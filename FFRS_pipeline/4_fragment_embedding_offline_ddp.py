#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any

from tqdm import tqdm
import torch
import torch.distributed as dist

# 你已有的封装：必须在 set_device 之后再 import/实例化模型更稳
from gme_inference import GmeQwen2VL

def load_samples(path: str) -> List[Dict[str, Any]]:
    # 支持 .json / .jsonl
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
    return "".join(ch if ch.isalnum() or ch in "-_+:" else "_" for ch in str(s))

def shard_range(n: int, rank: int, world: int, contiguous: bool = True):
    if contiguous:
        per = (n + world - 1) // world
        st = rank * per
        ed = min((rank + 1) * per, n)
        return st, ed
    # 交错切分（需要返回显式索引列表）
    idxs = list(range(rank, n, world))
    return idxs, None

def main():
    ap = argparse.ArgumentParser(description="DDP: encode response embeddings (dialogue_id+fragment_id.npy)")
    ap.add_argument("--input_json", default='', help="输入数据（.json 或 .jsonl）")
    ap.add_argument("--model_path", default='', help="GME/Qwen2VL 检查点路径")
    ap.add_argument("--out_dir", default='', help="向量输出目录")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=31072)
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="bf16",
                    help="落盘精度（numpy 类型）；模型内部精度按封装为准")
    ap.add_argument("--interleave", action="store_true",
                    help="使用交错切分(默认连续切分)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- init DDP ----------------
    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("[INFO] 读取数据...")
    samples = load_samples(args.input_json)
    N = len(samples)
    if rank == 0:
        print(f"[INFO] 样本数: {N} | world_size={world}")

    # ---------------- shard ----------------
    if args.interleave:
        shard, _ = shard_range(N, rank, world, contiguous=False)
        idx_list = shard
        if len(idx_list) == 0:
            if use_ddp:
                dist.barrier(); dist.destroy_process_group()
            return
    else:
        st, ed = shard_range(N, rank, world, contiguous=True)
        idx_list = list(range(st, ed))
        if st >= ed:
            if use_ddp:
                dist.barrier(); dist.destroy_process_group()
            return

    print(f"[rank{rank}] shard size = {len(idx_list)}")

    # ---------------- model ----------------
    if rank == 0:
        print("[INFO] 加载模型...")
    gme = GmeQwen2VL(args.model_path, max_length=args.max_length)  # 封装内部应走当前cuda设备

    # ---------------- dtype for saving ----------------
    save_dtype = {
        "fp16": np.float16,
        "bf16": np.float32,   # numpy 无 bfloat16，用 fp32 落盘更稳；需省体积可改成 fp16
        "fp32": np.float32
    }[args.dtype]

    # ---------------- encode loop (分批：纯文本/多模态) ----------------
    # 先把本 shard 的样本拆成两类
    text_only_texts, text_only_ids = [], []
    mm_texts, mm_images, mm_ids = [], [], []
    for i in idx_list:
        s = samples[i]
        text = s.get("response", "")
        imgs = s.get("images", []) or []
        if not imgs or all(v is None for v in imgs):
            text_only_texts.append(text)
            text_only_ids.append(i)
        else:
            mm_texts.append(text)
            mm_images.append(imgs)
            mm_ids.append(i)

    # 编码
    embs_text = []
    embs_mm = []
    if text_only_texts:
        if rank == 0:
            print(f"[INFO] 纯文本 shard 总数: {len(text_only_texts)}")
        embs_text = gme.get_text_embeddings(text_only_texts, batch_size=args.batch_size)
    if mm_texts:
        if rank == 0:
            print(f"[INFO] 图文 shard 总数: {len(mm_texts)}")
        embs_mm = gme.get_fragment_embeddings(texts=mm_texts, images=mm_images, batch_size=args.batch_size)

    # 回填 -> 保存
    index_shard = os.path.join(args.out_dir, f"index.rank{rank}.jsonl")
    saved = 0
    with open(index_shard, "w", encoding="utf-8") as fw:
        # 纯文本
        for i, emb in zip(text_only_ids, embs_text):
            if emb is None or not np.isfinite(emb).all():
                continue
            s = samples[i]
            did = str(s.get("dialogue_id", i))
            fid = s.get("fragment_id", 0)
            key = f"{safe_name(did)}+{fid}"
            np.save(os.path.join(args.out_dir, f"{key}.npy"), np.asarray(emb, dtype=save_dtype))
            fw.write(json.dumps({
                "key": key,
                "dialogue_id": did,
                "fragment_id": fid,
                "path": os.path.join(args.out_dir, f"{key}.npy"),
                "dim": int(len(emb)),
                "images": len(s.get("images", []) or [])
            }, ensure_ascii=False) + "\n")
            saved += 1

        # 多模态
        for i, emb in zip(mm_ids, embs_mm):
            if emb is None or not np.isfinite(emb).all():
                continue
            s = samples[i]
            did = str(s.get("dialogue_id", i))
            fid = s.get("fragment_id", 0)
            key = f"{safe_name(did)}+{fid}"
            np.save(os.path.join(args.out_dir, f"{key}.npy"), np.asarray(emb, dtype=save_dtype))
            fw.write(json.dumps({
                "key": key,
                "dialogue_id": did,
                "fragment_id": fid,
                "path": os.path.join(args.out_dir, f"{key}.npy"),
                "dim": int(len(emb)),
                "images": len(s.get("images", []) or [])
            }, ensure_ascii=False) + "\n")
            saved += 1

    print(f"[rank{rank}] saved = {saved}  -> {index_shard}")

    # ---------------- merge index (rank0) ----------------
    if use_ddp:
        dist.barrier()
    if rank == 0:
        merged = os.path.join(args.out_dir, "index.jsonl")
        with open(merged, "w", encoding="utf-8") as fout:
            for r in range(world):
                shard_p = os.path.join(args.out_dir, f"index.rank{r}.jsonl")
                if not os.path.exists(shard_p):
                    continue
                with open(shard_p, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
        print(f"[OK] 索引合并完成：{merged}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()