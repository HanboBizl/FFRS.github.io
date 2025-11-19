#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from typing import Any, Dict, List, Tuple

# ============== 语料库载入 ==============

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
            s = ln.strip()
            if s:
                metas.append(json.loads(s))

    assert len(metas) > 0, "index.jsonl 为空"
    dim = int(metas[0]["dim"])
    embs = np.empty((len(metas), dim), dtype=np.float32)

    for i, m in enumerate(metas):
        vec = np.load(m["path"])
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != dim:
            raise ValueError(f"向量维度不一致: {m['path']} 维度={vec.shape[0]} 期望={dim}")
        if not np.isfinite(vec).all():
            raise ValueError(f"向量含 NaN/Inf: {m['path']}")
        embs[i] = vec

    # L2 归一化（余弦相似度 = 点积）
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs, metas

# ============== Query 编码（根据你的环境替换为真实编码器） ==============

class GmeQwen2VLEncoder:
    """
    这里假设你已有一个文本编码器（比如你之前的 GME/Qwen2VL 封装）。
    把 `encode()` 改成调用你现有的接口即可。
    """
    def __init__(self, model_path: str, max_length: int = 15536):
        from gme_inference import GmeQwen2VL  # 你已有的封装
        self.encoder = GmeQwen2VL(model_path, max_length=max_length)

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self.encoder.get_text_embeddings(texts, batch_size=batch_size)
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        return vecs

# ============== TopK 计算 ==============

def topk_indices_by_dot(A: np.ndarray, B: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 A @ B^T 取每行 TopK（A: QxD，B: MxD，均已归一化 -> 点积=cosine）
    返回：
      - idx: (Q, k) TopK 索引（按分数降序）
      - val: (Q, k) TopK 分数
    """
    Q, D = A.shape
    M, D2 = B.shape
    assert D == D2, f"维度不匹配: A={A.shape}, B={B.shape}"

    k = min(k, M)
    all_top_idx = np.empty((Q, k), dtype=np.int32)
    all_top_val = np.empty((Q, k), dtype=np.float32)

    # 简单整块计算（如需更低内存，可改成分块）
    scores = A @ B.T  # (Q, M)

    if k < M:
        part = np.argpartition(scores, -k, axis=1)[:, -k:]  # 无序
        part_vals = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_vals, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)
        top_val = np.take_along_axis(part_vals, order, axis=1)
    else:
        order = np.argsort(-scores, axis=1)
        top_idx = order[:, :k]
        top_val = np.take_along_axis(scores, top_idx, axis=1)

    all_top_idx[:] = top_idx
    all_top_val[:] = top_val
    return all_top_idx, all_top_val

# ============== 主逻辑 ==============

def parse_query_text_arg(arg: str) -> List[str]:
    """
    支持：
      --query_text "单条字符串"
      --query_text '["q1","q2","q3"]'
    """
    s = (arg or "").strip()
    if not s:
        return []
    # 如果是 JSON 数组
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception as e:
            raise SystemExit(f"--query_text 解析 JSON 数组失败: {e}")
    # 否则按单条字符串处理
    return [s]

def main():
    ap = argparse.ArgumentParser(description="按 --query_text（支持 JSON 数组）检索片段语料库并输出 TopK")
    ap.add_argument("--query_text", required=True,
                    help='可传：单条字符串；或 JSON 数组字符串，如：\'["讨论数据集运行结果","讨论上传数据集"]\'')
    ap.add_argument("--corpus", default='', help="片段语料库目录或 index.jsonl 路径")
    ap.add_argument("--model", default='', help="用于编码 query 的模型路径（你的 GME/Qwen2VL checkpoint）")
    ap.add_argument("--max_length", type=int, default=15536)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--output_jsonl", default='', help="输出 JSONL（每条 query 一行）")
    ap.add_argument("--output_json", default="", help="（可选）同时输出一个汇总 JSON 文件")
    args = ap.parse_args()

    # 1) 解析 query_text
    queries = parse_query_text_arg(args.query_text)
    if not queries:
        raise SystemExit("空的 --query_text。示例：--query_text '\"讨论数据集运行结果\"' 或 --query_text '[\"q1\",\"q2\"]'")
    print(f"[INFO] queries: {len(queries)} -> {queries[:3]}{'...' if len(queries) > 3 else ''}")

    # 2) 载入语料
    corpus_embs, corpus_meta = load_corpus(args.corpus)
    print(f"[INFO] 语料库条数: {len(corpus_meta)}  向量维度: {corpus_embs.shape[1]}")

    # 3) 编码 query
    encoder = GmeQwen2VLEncoder(args.model, max_length=args.max_length)
    q_embs = encoder.encode(queries, batch_size=args.batch_size)
    assert q_embs.shape[0] == len(queries), "query 编码数量不一致"

    # 4) 检索 TopK
    idx, val = topk_indices_by_dot(q_embs, corpus_embs, k=args.topk)

    # 5) 组织输出（为了兼容你后续 step6，字段尽量保持一致）
    results: List[Dict[str, Any]] = []
    for qi, qtext in enumerate(queries):
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
        results.append({
            "dialogue_id": "",        # 这里没有原始 query 对话，可留空
            "query_text": qtext,      # 关键字段：用于 step6 继续构造 F2RVLM 输入
            "gt_turns": [],           # 兼容占位
            "top20": top_list         # 注意：字段名保持为 top20 以兼容你当前 6_fragment2F2RVLM.py
        })

    # 6) 落盘
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as fw:
        for rec in results:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] 写出 JSONL: {args.output_jsonl}  共 {len(results)} 条")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fw:
            json.dump(results, fw, ensure_ascii=False, indent=2)
        print(f"[OK] 另存 JSON: {args.output_json}")

if __name__ == "__main__":
    main()