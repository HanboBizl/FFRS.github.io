#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

# -------- regex --------
UTT_LINE_RE = re.compile(
    r"\<\|utt_id_start\|\>(\d+)\<\|utt_id_end\|\>\s*([^:：]+)\s*[:：]\s*(.*)$"
)
ANS_RE = re.compile(
    r"<\|utt_ids_start\|\>\s*\[(.*?)\]\s*<\|utt_ids_end\|\>\s*;"
    r"\s*<\|img_ids_start\|\>\s*\[(.*?)\]\s*<\|img_ids_end\|\>"
)
IMG_TAG_RE = re.compile(r"<\|img_id_start\|\>(\d+)\<\|img_id_end\|\>")

# -------- io --------
def load_any(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception as e:
                    print(f"[WARN] {path} line {ln} parse error: {e}")
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def save_like_input(objs: List[Dict[str, Any]], in_path: str, out_path: str):
    in_is_jsonl = Path(in_path).suffix.lower() in (".jsonl", ".ndjson")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if in_is_jsonl:
        with open(out_path, "w", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(objs, f, ensure_ascii=False, indent=2)

# -------- helpers --------
def pick_user_content(messages: List[Dict[str, Any]]) -> str:
    for m in messages or []:
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""

def get_assistant_index_and_text(messages: List[Dict[str, Any]]) -> Tuple[int, str]:
    if not isinstance(messages, list):
        return -1, ""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return i, (m.get("content", "") or "")
    return -1, ""

def parse_pred_lists(ans: str) -> Tuple[List[int], List[int]]:
    """解析 '<|utt_ids_start|>[..]</..>;<|img_ids_start|>[..]</..>' -> (utt_ids, img_ids)
       保留原顺序，顺序去重；解析失败则返回([],[])."""
    m = ANS_RE.search(ans or "")
    if not m:
        return [], []
    def parse_int_list(seg: str) -> List[int]:
        seg = seg.strip()
        if not seg:
            return []
        out, seen = [], set()
        for p in seg.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                v = int(p)
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            except Exception:
                pass
        return out
    return parse_int_list(m.group(1)), parse_int_list(m.group(2))

def build_img2utt_map(user_content: str) -> Dict[int, int]:
    """扫描用户对话文本，建立 img_id -> utt_id 映射；同一 img_id 多次出现取首次。"""
    mapping: Dict[int, int] = {}
    for line in user_content.splitlines():
        m = UTT_LINE_RE.match(line.strip())
        if not m:
            continue
        utt_id = int(m.group(1))
        rest = m.group(3)
        for s in IMG_TAG_RE.findall(rest):
            img_id = int(s)
            if img_id not in mapping:
                mapping[img_id] = utt_id
    return mapping

def uniq_preserve_order(seq: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def to_utt_only_string(utt_ids: List[int]) -> str:
    return f"<|utt_ids_start|>[{', '.join(str(x) for x in utt_ids)}]<|utt_ids_end|>;"

def convert_field_to_utt_only(field_text: str, img2utt: Dict[int, int]) -> Tuple[str, List[int], List[int], List[int]]:
    """把一个字段（response或assistant）转换为仅句子ID输出。
       返回: (new_text, fused_utt_ids, orig_utt_ids, missed_img_ids)"""
    utt_ids, img_ids = parse_pred_lists(field_text)
    if not utt_ids and not img_ids:
        return field_text, [], [], []
    mapped_from_imgs = [img2utt[k] for k in img_ids if k in img2utt]
    missed = [k for k in img_ids if k not in img2utt]
    fused = uniq_preserve_order([*utt_ids, *mapped_from_imgs])
    return to_utt_only_string(fused), fused, utt_ids, missed

# -------- main --------
def main():
    ap = argparse.ArgumentParser(
        description="把候选的 '<|utt_ids|>...;<|img_ids|>...' 转为仅 '<|utt_ids|>...;'（img_id 映射到其所在的句子ID），对顶层 response 与 assistant 同步处理。"
    )
    ap.add_argument("--input", default='', help="输入 .json / .jsonl（每行/每项为一个样本）")
    ap.add_argument("--output", default='', help="输出文件（自动跟随输入类型写 json / jsonl）")
    args = ap.parse_args()

    data = load_any(args.input)
    print(f"[INFO] loaded {len(data)} items")

    out = []
    for rec in data:
        rec2 = dict(rec)
        rec2.setdefault("meta", {})
        # 1) img_id -> utt_id
        user_text = pick_user_content(rec.get("messages") or [])
        img2utt = build_img2utt_map(user_text)

        # 2) 顶层 response 转换
        resp_raw = rec.get("response", "")
        new_resp, fused_utt, orig_utt, missed_imgs = convert_field_to_utt_only(resp_raw, img2utt)
        if new_resp != resp_raw:
            rec2["meta"]["response_raw"] = resp_raw
            rec2["meta"]["response_fused_utt_ids"] = fused_utt
            rec2["meta"]["response_orig_utt_ids"] = orig_utt
            rec2["meta"]["response_missed_img_ids"] = missed_imgs
            rec2["response"] = new_resp

        # 3) assistant 转换
        msgs = rec.get("messages") or []
        aidx, atext = get_assistant_index_and_text(msgs)
        if aidx >= 0 and atext:
            new_atext, fused_utt_a, orig_utt_a, missed_imgs_a = convert_field_to_utt_only(atext, img2utt)
            if new_atext != atext:
                msgs2 = list(msgs)
                msgs2[aidx] = dict(msgs2[aidx])
                msgs2[aidx]["content"] = new_atext
                rec2["messages"] = msgs2
                rec2["meta"]["assistant_raw"] = atext
                rec2["meta"]["assistant_fused_utt_ids"] = fused_utt_a
                rec2["meta"]["assistant_orig_utt_ids"] = orig_utt_a
                rec2["meta"]["assistant_missed_img_ids"] = missed_imgs_a

        out.append(rec2)

    save_like_input(out, args.input, args.output)
    print(f"[OK] wrote {len(out)} items -> {args.output}")

if __name__ == "__main__":
    main()