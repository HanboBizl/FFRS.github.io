#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from typing import Any, Dict, List, Tuple

# 你已有的封装（用于编码 query）
from gme_inference import GmeQwen2VL


def load_samples(path: str) -> List[Dict[str, Any]]:
    """读取输入样本（支持 .json / .jsonl）"""
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(json.loads(ln))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_corpus(index_or_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    读取片段语料库：
      - index_or_dir: 可以是 index.jsonl 或其所在目录
    返回：
      - embs: (M, D) numpy float32，已做行向量 L2 归一化
      - meta: 长度 M 的元信息列表（含 key/dialogue_id/fragment_id/path/dim）
    """
    index_path = index_or_dir
    if os.path.isdir(index_or_dir):
        index_path = os.path.join(index_or_dir, "index.jsonl")
    assert os.path.exists(index_path), f"index.jsonl 不存在: {index_path}"

    metas: List[Dict[str, Any]] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                metas.append(json.loads(ln))

    assert len(metas) > 0, "index.jsonl 为空"
    dim = int(metas[0]["dim"])
    embs = np.empty((len(metas), dim), dtype=np.float32)

    for i, m in enumerate(metas):
        vec = np.load(m["path"])
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(f"向量维度不一致: {m['path']} 维度={vec.shape[0]} 期望={dim}")
        # 防 NaN/Inf
        if not np.isfinite(vec).all():
            raise ValueError(f"向量含 NaN/Inf: {m['path']}")
        embs[i] = vec

    # L2 归一化（余弦相似度 = 点积）
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs, metas


def collect_queries(samples: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[int, int]], List[List[int]], List[str]]:
    """
    从样本集中收集所有 query 文本，并记录索引关系
    返回：
      - queries: List[str]
      - q_index: List[(sample_idx, query_idx)]
      - gt_turns_per_query: 与 queries 对齐的 gt_turns（若不存在则 []）
      - dialogue_ids_per_query: 与 queries 对齐的 dialogue_id
    """
    queries: List[str] = []
    q_index: List[Tuple[int, int]] = []
    gt_list: List[List[int]] = []
    dids: List[str] = []

    for si, s in enumerate(samples):
        did = str(s.get("dialogue_id", ""))
        q_texts = s.get("query_text", []) or []
        gt_turns_all = s.get("gt_turns", []) or []

        for qi, q in enumerate(q_texts):
            if not isinstance(q, str) or not q.strip():
                # 空 query 也保留，但会产生低相似度；如需跳过，可 continue
                pass
            queries.append(q.strip())
            q_index.append((si, qi))
            # 对齐 gt_turns（长度不匹配时给空）
            if isinstance(gt_turns_all, list) and qi < len(gt_turns_all):
                gt_list.append(gt_turns_all[qi])
            else:
                gt_list.append([])
            dids.append(did)
    return queries, q_index, gt_list, dids


def encode_queries(model_path: str, queries: List[str], max_length: int = 15536, batch_size: int = 16) -> np.ndarray:
    """用 GME/Qwen2VL 编码 query，返回 (Q, D) float32 且 L2 归一化"""
    if len(queries) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    encoder = GmeQwen2VL(model_path, max_length=max_length)
    vecs = encoder.get_text_embeddings(queries, batch_size=batch_size)
    # 兼容返回 list 或 ndarray
    if isinstance(vecs, list):
        vecs = np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in vecs])
    else:
        vecs = np.asarray(vecs, dtype=np.float32)
    # 归一化
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs


def topk_indices_by_dot(A: np.ndarray, B: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 A @ B^T 取每行 TopK（A: QxD，B: MxD，均已归一化 -> 点积=cosine）
    返回：
      - idx: (Q, k) TopK 索引（按分数降序）
      - val: (Q, k) TopK 分数
    采用 argpartition 降低内存与计算开销
    """
    Q, D = A.shape
    M, D2 = B.shape
    assert D == D2, f"维度不匹配: A={A.shape}, B={B.shape}"

    # 分块计算防内存暴涨（可按需调 block_size）
    block_size = max(1, 8192 // max(1, M // 10000 + 1))  # 粗略自适应
    all_top_idx = np.empty((Q, k), dtype=np.int32)
    all_top_val = np.empty((Q, k), dtype=np.float32)

    start = 0
    while start < Q:
        end = min(Q, start + block_size)
        scores = A[start:end].dot(B.T)  # (b, M)

        # TopK（先 argpartition，再局部排序）
        if k < M:
            part = np.argpartition(scores, -k, axis=1)[:, -k:]  # (b, k) 无序
            part_vals = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-part_vals, axis=1)
            top_idx = np.take_along_axis(part, order, axis=1)
            top_val = np.take_along_axis(part_vals, order, axis=1)
        else:
            order = np.argsort(-scores, axis=1)
            top_idx = order[:, :k]
            top_val = np.take_along_axis(scores, top_idx, axis=1)

        all_top_idx[start:end] = top_idx
        all_top_val[start:end] = top_val
        start = end

    return all_top_idx, all_top_val


def main():
    ap = argparse.ArgumentParser(description="按 query_text 检索片段语料库，输出 Top-20 结果（每条 query 一行）")
    ap.add_argument("--input", default='', help="输入样本 JSON/JSONL（含 dialogue_id / query_text / gt_turns）")
    ap.add_argument("--corpus", default='', help="片段语料库目录或 index.jsonl 路径")
    ap.add_argument("--model", default=', help="用于编码 query 的模型路径（你的 GME/Qwen2VL checkpoint）")
    ap.add_argument("--max_length", type=int, default=15536)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--exclude-same-dialogue", action="store_true", help="检索时排除与 query 同 dialogue_id 的片段")
    ap.add_argument("--output_jsonl", default='', help="输出 JSONL（每条 query 一行）")
    ap.add_argument("--output_json", default="", help="（可选）同时输出一个汇总 JSON 文件")
    args = ap.parse_args()

    # 1) 载入数据与语料库
    samples = load_samples(args.input)
    corpus_embs, corpus_meta = load_corpus(args.corpus)
    print(f"[INFO] 片段语料库大小: {len(corpus_meta)}，向量维度: {corpus_embs.shape[1]}")

    # 2) 收集并编码所有 query
    queries, q_index, gt_turns_per_query, dids_per_query = collect_queries(samples)
    print(f"[INFO] 待检索的 query 数: {len(queries)}")
    q_embs = encode_queries(args.model, queries, max_length=args.max_length, batch_size=args.batch_size)
    assert q_embs.shape[0] == len(queries), "query 编码数量不一致"

    # 3) 若排除同对话，预先构造 mask（把同 dialogue_id 的片段分数置极小）
    if args.exclude_same_dialogue:
        # 为每个 query 列出需屏蔽的片段索引
        did_to_indices: Dict[str, List[int]] = {}
        for i, meta in enumerate(corpus_meta):
            did_to_indices.setdefault(str(meta["dialogue_id"]), []).append(i)

    # 4) TopK 检索
    topk = args.topk
    if not args.exclude_same_dialogue:
        idx, val = topk_indices_by_dot(q_embs, corpus_embs, k=topk)
    else:
        # 分批计算并屏蔽
        Q = q_embs.shape[0]
        idx = np.empty((Q, topk), dtype=np.int32)
        val = np.empty((Q, topk), dtype=np.float32)
        step = 1024
        for s in range(0, Q, step):
            e = min(Q, s + step)
            scores = q_embs[s:e].dot(corpus_embs.T)
            # 屏蔽同 dialogue
            for r in range(s, e):
                did = dids_per_query[r]
                for j in did_to_indices.get(did, []):
                    scores[r - s, j] = -1e9
            # 取 topk
            part = np.argpartition(scores, -topk, axis=1)[:, -topk:]
            part_vals = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-part_vals, axis=1)
            idx[s:e] = np.take_along_axis(part, order, axis=1)
            val[s:e] = np.take_along_axis(part_vals, order, axis=1)

    # 5) 组织输出（每条 query 一条）
    results_per_query: List[Dict[str, Any]] = []
    for qi, (si, qj) in enumerate(q_index):
        did = str(samples[si].get("dialogue_id", ""))
        qtext = queries[qi]
        gt_turns = gt_turns_per_query[qi]

        top_list = []
        for rank, ci in enumerate(idx[qi].tolist(), start=1):
            m = corpus_meta[ci]
            top_list.append({
                "rank": rank,
                "key": m["key"],  # e.g. "4525591374+0"
                "dialogue_id": str(m["dialogue_id"]),
                "fragment_id": int(m["fragment_id"]),
                "score": float(val[qi][rank-1]),
            })

        results_per_query.append({
            "dialogue_id": did,
            "query_text": qtext,
            "gt_turns": gt_turns,
            "top20": top_list,
        })

    # 6) 落盘
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as fw:
        for rec in results_per_query:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] 已写出 JSONL: {args.output_jsonl} （共 {len(results_per_query)} 行）")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fw:
            json.dump(results_per_query, fw, ensure_ascii=False, indent=2)
        print(f"[OK] 另存 JSON: {args.output_json}")


if __name__ == "__main__":
    main()