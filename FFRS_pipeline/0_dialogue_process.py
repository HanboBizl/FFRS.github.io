#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List


def _to_str_id(did: Any) -> str:
    """对话ID统一成字符串，但不做任何裁剪/去后缀处理。"""
    return str(did)


def _dialogue_equal(d1: Any, d2: Any) -> bool:
    """判等（防御性：结构一致即可）。"""
    return json.dumps(d1, ensure_ascii=False, sort_keys=True) == \
           json.dumps(d2, ensure_ascii=False, sort_keys=True)


def _dedup_preserve_order(seq):
    """顺序去重（用于 query_text 与 gt_turns 对）。"""
    seen = set()
    out = []
    for x in seq:
        key = json.dumps(x, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def merge_by_exact_dialogue_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 按完全相同的 dialogue_id 分组
    groups = defaultdict(list)
    for s in items:
        did = _to_str_id(s.get("dialogue_id", ""))
        groups[did].append(s)

    merged = []
    for did, group in groups.items():
        # 1) 对话固定取第一条（若不一致仅警告，不修改）
        dialogue = group[0].get("dialogue", [])
        for g in group[1:]:
            if not _dialogue_equal(dialogue, g.get("dialogue", [])):
                # 仅打印一次警告；为脚本简洁，这里不抛错
                print(f"[WARN] dialogue_id={did} 的对话内容存在不一致，已使用首条。")
                break

        # 2) 收集 query_text 与 gt_turns（保持输入顺序，去重）
        q_list: List[str] = []
        t_list: List[List[int]] = []
        for g in group:
            if "query_text" in g:
                q_list.append("" if g["query_text"] is None else str(g["query_text"]))
            if "gt_turns" in g and isinstance(g["gt_turns"], list):
                # 兜底强转为 int
                turns = []
                for v in g["gt_turns"]:
                    try:
                        turns.append(int(v))
                    except Exception:
                        pass
                t_list.append(turns)

        # 去重（按整体一对一去重，保持配对关系）
        pair_list = []
        for q, t in zip(q_list, t_list):
            pair_list.append({"q": q, "t": t})
        pair_list = _dedup_preserve_order(pair_list)

        merged_q = [p["q"] for p in pair_list]
        merged_t = [p["t"] for p in pair_list]

        merged.append({
            "dialogue_id": did,
            "dialogue": dialogue,
            "query_text": merged_q,   # 例如 ["问题A","问题B",...]
            "gt_turns": merged_t      # 例如 [[1,2,3],[10,11],...]
        })

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default='Wechat/query_with_dialogue_0928.json', help="输入 JSON 文件（列表或单条对象）")
    ap.add_argument("--output", default='data/Dialogcc/process_0825/pipeline_process/0_wechat_dialogue_v2.json', help="输出 JSON 文件（列表）")
    ap.add_argument("--pretty", action="store_true", help="美化缩进输出")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    assert isinstance(data, list), "输入必须是 JSON 数组或单条对象"

    out = merge_by_exact_dialogue_id(data)

    with open(args.output, "w", encoding="utf-8") as f:

        json.dump(out, f, ensure_ascii=False, indent=2)


    print(f"[DONE] {len(out)} merged dialogues -> {args.output}")


if __name__ == "__main__":
    main()