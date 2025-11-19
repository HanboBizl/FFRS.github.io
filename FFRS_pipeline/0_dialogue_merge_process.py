#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- IO ----------

def load_any(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # 先尝试整体 JSON
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    # 退回 JSONL
    out = []
    for ln in txt.splitlines():
        s = ln.strip()
        if s:
            out.append(json.loads(s))
    return out

def save_json(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 指纹：只看 dialogue 内容，确保“内容完全一致”才归为一组 ----------

def _norm_utt(utt: Dict[str, Any]) -> Dict[str, Any]:
    """仅用于生成稳定指纹；不会回写数据。"""
    who = utt.get("speaker", "")
    idx = utt.get("utterance_idx", None)
    content = utt.get("utterance", "")

    if isinstance(content, str):
        payload = {"t": "text", "v": content}
    elif isinstance(content, list):
        items = []
        for it in content:
            if isinstance(it, dict) and ("image_url" in it or "caption" in it):
                items.append({"u": it.get("image_url", ""), "c": it.get("caption", "")})
            else:
                items.append({"raw": str(it)})
        payload = {"t": "images", "v": items}
    else:
        payload = {"t": "raw", "v": str(content)}

    out = {"who": who, "p": payload}
    if idx is not None:
        out["i"] = int(idx)
    return out

def dialogue_fingerprint(dialogue: List[Dict[str, Any]]) -> str:
    norm = [_norm_utt(u or {}) for u in (dialogue or [])]
    s = json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------- 合并策略（最小改动） ----------

def lists_equal(a, b) -> bool:
    try:
        return json.dumps(a, ensure_ascii=False, sort_keys=False) == \
               json.dumps(b, ensure_ascii=False, sort_keys=False)
    except Exception:
        return a == b

def all_qt_gt_identical(items: List[Dict[str, Any]]) -> bool:
    """若组内所有样本的 (query_text, gt_turns) 完全一致（顺序也一致）则返回 True。"""
    if not items:
        return True
    q0 = items[0].get("query_text", []) or []
    g0 = items[0].get("gt_turns", []) or []
    for it in items[1:]:
        if not lists_equal(q0, it.get("query_text", []) or []):
            return False
        if not lists_equal(g0, it.get("gt_turns", []) or []):
            return False
    return True

def minimal_merge(items: List[Dict[str, Any]],
                  emit_merged_from: bool = False,
                  merge_strategy: str = "keep-first") -> Dict[str, Any]:
    """
    最小改动合并：
    - 若 (query_text, gt_turns) 全一致：直接返回第一条“原样拷贝”（不加任何字段）
    - 否则：以第一条为基准：
        * 已存在的 query：默认不改其 gt_turns（merge_strategy='keep-first'）
          （可选 'union' 时才把所有样本对应的 turns 做并集去重升序）
        * 其它样本中出现的“新 query”：按首次出现的顺序依次追加到末尾，gt_turns 直接复制，不做排序/去重
    """
    primary = items[0]
    # 情况1：完全一致 -> 原样复制
    if all_qt_gt_identical(items):
        return {
            "dialogue_id": primary.get("dialogue_id"),
            "dialogue": primary.get("dialogue"),
            "query_text": primary.get("query_text", []) or [],
            "gt_turns": primary.get("gt_turns", []) or [],
            **({} if not emit_merged_from else {})  # 不添加 merged_from，保持完全不变
        }

    # 情况2：存在差异 -> 最小化合并
    out = {
        "dialogue_id": primary.get("dialogue_id"),
        "dialogue": primary.get("dialogue"),
        "query_text": list(primary.get("query_text", []) or []),
        "gt_turns":  list(primary.get("gt_turns", [])  or []),
    }

    # 建立 query -> index 映射（以首条为准）
    q2idx: Dict[str, int] = {}
    for i, q in enumerate(out["query_text"]):
        q2idx[str(q)] = i

    # 遍历剩余样本，合并“新 query”，以及（可选）处理已存在 query。
    for it in items[1:]:
        ql = it.get("query_text", []) or []
        gl = it.get("gt_turns", []) or []
        for i, q in enumerate(ql):
            q = str(q)
            turns = gl[i] if i < len(gl) and isinstance(gl[i], list) else []
            if q in q2idx:
                if merge_strategy == "union":
                    # 仅在 union 模式下才合并；否则严格保留首条的 gt_turns
                    old = out["gt_turns"][q2idx[q]]
                    # 做并集，但尽量保持稳定：先旧后新，再去重，最后不排序（保留出现顺序）
                    seen = set()
                    merged = []
                    for x in list(old) + list(turns):
                        if isinstance(x, int):
                            key = x
                        else:
                            # 尽量不抛异常，保持用户数据
                            try:
                                key = int(x)
                            except Exception:
                                key = x  # 若真的不是 int，就按原样
                        if key not in seen:
                            seen.add(key)
                            merged.append(key)
                    out["gt_turns"][q2idx[q]] = merged
                else:
                    # keep-first：不动
                    pass
            else:
                # 新 query：追加到末尾（不改顺序、不排序、不去重）
                q2idx[q] = len(out["query_text"])
                out["query_text"].append(q)
                out["gt_turns"].append(list(turns))

    if emit_merged_from:
        out["merged_from"] = [it.get("dialogue_id") for it in items[1:] if it.get("dialogue_id")]

    return out

# ---------- 主程序 ----------

def main():
    ap = argparse.ArgumentParser(description="按对话内容去重（最小改动），确保相同内容时首条完全不变")
    ap.add_argument("--input", default="data/Dialogcc/process_0825/pipeline_process/0_wechat_dialogue_v2.json", help="输入 JSON/JSONL（含 dialogue_id, dialogue, query_text, gt_turns）")
    ap.add_argument("--output", default="data/Dialogcc/process_0825/pipeline_process/0_wechat_dialogue_v3.json", help="输出 JSON（合并后）")
    ap.add_argument("--emit-merged-from", action="store_true", default=False,
                    help="是否在合并结果中添加 merged_from（默认不添加，以免改变结构）")
    ap.add_argument("--merge-gt-strategy", choices=["keep-first", "union"], default="keep-first",
                    help="当同名 query 的 gt_turns 不一致时的处理策略：默认 keep-first；可选 union")
    args = ap.parse_args()

    data = load_any(args.input)
    print(f"[INFO] loaded {len(data)} items")

    # 1) 按 dialogue 内容指纹分组
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for obj in data:
        fp = dialogue_fingerprint(obj.get("dialogue", []) or [])
        groups.setdefault(fp, []).append(obj)

    print(f"[INFO] unique dialogues by content: {len(groups)}")

    # 2) 逐组合并（最小改动）
    merged: List[Dict[str, Any]] = []
    removed = 0
    for fp, items in groups.items():
        if len(items) == 1:
            # 单条：绝对不改
            x = items[0]
            merged.append({
                "dialogue_id": x.get("dialogue_id"),
                "dialogue": x.get("dialogue"),
                "query_text": x.get("query_text", []) or [],
                "gt_turns":  x.get("gt_turns", [])  or [],
            })
        else:
            removed += (len(items) - 1)
            out = minimal_merge(items,
                                emit_merged_from=args.emit_merged_from,
                                merge_strategy=args.merge_gt_strategy)
            merged.append(out)

    save_json(merged, args.output)
    print(f"[OK] wrote {len(merged)} items -> {args.output} (removed {removed} duplicates)")
    print("[NOTE] 若组内 (query_text, gt_turns) 完全一致，则首条样本保持 bit-by-bit 不变（不加任何额外字段）。")

if __name__ == "__main__":
    main()