#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
from typing import List, Any, Dict
from tqdm import tqdm

import torch
from gme_inference import GmeQwen2VL  # 你已有的封装

# ---------------- utils: 读入/清洗 ----------------

def load_samples(path: str) -> List[Dict[str, Any]]:
    # 兼容 .json / .jsonl
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def safe_name(s: str) -> str:
    # 文件名清洗：只保留字母数字-_:+，其他字符替换为_
    return "".join(ch if ch.isalnum() or ch in "-_+:" else "_" for ch in str(s))

def extract_image_paths(images_field: Any) -> List[str]:
    """
    支持两种格式：
      - [{'path': '...'}, ...]
      - ['...', ...]
    统一成: [str, ...]
    """
    paths: List[str] = []
    if isinstance(images_field, list):
        for it in images_field:
            if isinstance(it, dict):
                p = it.get("path") or it.get("image_url") or it.get("url")
                if isinstance(p, str) and p:
                    paths.append(p)
            elif isinstance(it, str) and it:
                paths.append(it)
    return paths

# ---------------- utils: <image> 对齐检查/修复 ----------------

IMG_TOKEN = "<image>"

def count_img_tokens(text: str) -> int:
    # 稳定计数（字面匹配）
    return len(re.findall(r"<image>", text))

def fix_text_and_images(text: str, images: List[str]) -> str:
    """
    若 <image> 多于图片：删除多余占位。
    若 图片多于 <image>：在文本末尾追加占位（以空格分隔）。
    返回修正后的文本（图片列表不变）。
    """
    m = count_img_tokens(text)
    n = len(images)
    if m == n:
        return text
    if m > n:
        parts = re.split(r"(<image>)", text)
        kept, used = [], 0
        for p in parts:
            if p == "<image>":
                if used < n:
                    kept.append(p)
                    used += 1
                else:
                    # 丢弃多余占位
                    continue
            else:
                kept.append(p)
        return "".join(kept)
    else:
        need = n - m
        pad = ("" if text.endswith(" ") else " ") + " ".join([IMG_TOKEN] * need)
        return text + pad

def preflight_normalize(
    samples: List[Dict[str, Any]],
    policy: str = "auto"  # auto|check|drop|error
) -> List[Dict[str, Any]]:
    """
    在编码前做片段级一致性检查/修复：
      - auto: 自动修复（默认）
      - check: 仅统计并保留原样
      - drop: 不一致的样本直接丢弃
      - error: 发现不一致直接抛异常
    同时把 images 字段统一为 [str,...]
    """
    total = len(samples)
    aligned = mismatched = fixed = dropped = 0
    out = []
    for s in samples:
        text = s.get("response", "")
        # 统一图片列表为字符串
        imgs = extract_image_paths(s.get("images", []))
        # 回写，保证后续流程一致
        s = dict(s)
        s["images"] = imgs

        m = count_img_tokens(text)
        n = len(imgs)

        if m == n:
            aligned += 1
            out.append(s)
            continue

        mismatched += 1
        if policy == "auto":
            new_text = fix_text_and_images(text, imgs)
            if count_img_tokens(new_text) != len(imgs):
                dropped += 1
                continue
            s["response"] = new_text
            out.append(s)
            fixed += 1
        elif policy == "check":
            out.append(s)
        elif policy == "drop":
            dropped += 1
            continue
        elif policy == "error":
            did = s.get("dialogue_id", "?")
            fid = s.get("fragment_id", "?")
            raise RuntimeError(
                f"[Mismatch] dialogue_id={did} fragment_id={fid} <image>={m} images={n}"
            )
        else:
            raise ValueError(f"Unknown mismatch policy: {policy}")

    print("[CHECK] <image> 对齐统计："
          f" total={total} aligned={aligned} mismatched={mismatched} fixed={fixed} dropped={dropped}")
    return out

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Encode response embeddings and save as dialogue_id+fragment_id.npy (with <image>-images check)"
    )
    ap.add_argument("--input_json", default='', help="输入数据（.json 或 .jsonl）")
    ap.add_argument("--model_path", default='', help="GME/Qwen2VL 检查点路径")
    ap.add_argument("--out_dir", default='', help="向量输出目录")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=31072)
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="bf16", help="保存到磁盘的向量精度（模型仍按封装跑）")
    ap.add_argument("--mismatch_policy", choices=["auto","check","drop","error"], default="error",
                    help="片段 <image> 与图片数不一致时的处理策略")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "index.jsonl")

    print("[INFO] 读取样本中...")
    samples = load_samples(args.input_json)
    print(f"[INFO] 原始样本数: {len(samples)}")

    # ★ 一致性检查 / 归一化（含 images 统一为字符串路径）
    print(f"[INFO] 片段一致性检查策略: {args.mismatch_policy}")
    samples = preflight_normalize(samples, policy=args.mismatch_policy)
    N = len(samples)
    print(f"[INFO] 规范化后样本数: {N}")

    print("[INFO] 加载模型中...")
    gme = GmeQwen2VL(args.model_path, max_length=args.max_length)

    # 准备两路：纯文本/多模态
    texts_textonly, idxs_textonly = [], []
    texts_mm, images_mm, idxs_mm = [], [], []

    for i, s in enumerate(samples):
        text = s.get("response", "")
        imgs = s.get("images", []) or []
        if not imgs or all(v is None for v in imgs):
            texts_textonly.append(text)
            idxs_textonly.append(i)
        else:
            texts_mm.append(text)
            images_mm.append(imgs)
            idxs_mm.append(i)

    # 编码
    embs_text = []
    embs_mm = []

    if texts_textonly:
        print(f"[INFO] 纯文本片段: {len(texts_textonly)} -> 编码中...")
        embs_text = gme.get_text_embeddings(texts_textonly, batch_size=args.batch_size)

    if texts_mm:
        print(f"[INFO] 图文片段: {len(texts_mm)} -> 编码中...")
        embs_mm = gme.get_fragment_embeddings(texts=texts_mm, images=images_mm, batch_size=args.batch_size)

    # 回填到原始顺序
    all_embs: List[np.ndarray] = [None] * N  # type: ignore
    for i, emb in zip(idxs_textonly, embs_text):
        all_embs[i] = emb
    for i, emb in zip(idxs_mm, embs_mm):
        all_embs[i] = emb

    # 维度检测
    dims = {None}
    for e in all_embs:
        if e is not None:
            dims.add(len(e))
    dims.discard(None)
    assert len(dims) == 1, f"[ERR] 向量维度不一致: {dims}"
    dim = list(dims)[0]
    print(f"[INFO] 向量维度: {dim}")

    # 精度选择
    save_dtype = {
        "fp16": np.float16,
        "bf16": np.float32,  # numpy 没有 bfloat16，用 float32 存；若需要节省体积可用 float16
        "fp32": np.float32
    }[args.dtype]

    # 落盘 + 索引
    skipped = 0
    with open(index_path, "w", encoding="utf-8") as fw:
        for i, s in enumerate(tqdm(samples, desc="[SAVE]")):
            emb = all_embs[i]
            if emb is None:
                skipped += 1
                continue
            # 防 NaN
            if not np.isfinite(emb).all():
                print(f"[WARN] 第{i}条 embedding 含 NaN/Inf，已跳过。")
                skipped += 1
                continue
            did = str(s.get("dialogue_id", i))
            fid = s.get("fragment_id", 0)
            key = f"{safe_name(did)}+{fid}"
            out_path = os.path.join(args.out_dir, f"{key}.npy")
            np.save(out_path, np.asarray(emb, dtype=save_dtype))
            # 写索引
            rec = {
                "key": key,
                "dialogue_id": did,
                "fragment_id": fid,
                "path": out_path,
                "dim": int(dim),
                "images": len(s.get("images", []) or []),
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] 完成：共 {N - skipped} 条保存到 {args.out_dir} ；索引 => {index_path}")
    if skipped:
        print(f"[WARN] 跳过 {skipped} 条（无向量或含 NaN/Inf）")

if __name__ == "__main__":
    main()
