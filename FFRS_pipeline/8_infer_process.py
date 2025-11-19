#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable, Optional

# ============ 正则 ============
UTT_ONLY_RE = re.compile(
    r"<\|utt_ids_start\|\>\s*\[(.*?)\]\s*<\|utt_ids_end\|\>\s*;?"
)
QUERY_RE = re.compile(r"<\|query_start\|\>(.*?)<\|query_end\|\>")

# ============ IO ============
def load_any(path: str) -> List[Dict[str, Any]]:
    """
    读取 .json / .jsonl：
    - 先整体读入并尝试 json.loads；
    - 失败则按行解析（JSONL），逐行 json.loads（自动跳过空行和尾逗号）。
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    # 尝试作为 JSON
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass  # 回退为 JSONL
    # 作为 JSONL 逐行读
    out: List[Dict[str, Any]] = []
    for ln, line in enumerate(raw.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1].rstrip()
        try:
            out.append(json.loads(s))
        except Exception as e:
            print(f"[WARN] {path} line {ln} parse error: {e}")
    return out

def save_json(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ============ 工具 ============
def uniq_preserve_order(seq: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_query_text_from_user(messages: List[Dict[str, Any]]) -> str:
    """从 user 的 content 里兜底抓取 <|query_start|>...<|query_end|>。"""
    for m in messages or []:
        if m.get("role") != "user":
            continue
        c = m.get("content", "") or ""
        mm = QUERY_RE.search(c)
        if mm:
            return (mm.group(1) or "").strip()
    return ""

def get_query_text(rec: Dict[str, Any]) -> str:
    """优先 meta.query_text；否则从 user prompt 解析；兜底空字符串。"""
    meta = rec.get("meta") or {}
    qt = meta.get("query_text")
    if isinstance(qt, str) and qt.strip():
        return qt.strip()
    return extract_query_text_from_user(rec.get("messages") or []) or ""

def parse_utt_ids_from_response(resp: str) -> List[int]:
    """从 response 里解析 <|utt_ids_start|>[...]<|utt_ids_end|>；顺序去重；跳过非法。"""
    m = UTT_ONLY_RE.search(resp or "")
    if not m:
        return []
    seg = m.group(1).strip()
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

# ============ 分段元数据 ============
def load_fragment_meta(path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取“原始对话分片”元数据，返回：
      meta_map[dialogue_id] = {
         "fragment_count": int,
         "fragments": List[List[int]],  # 每个 fragment 的“全局句子ID”列表
      }
    """
    items = load_any(path)
    meta_map: Dict[str, Dict[str, Any]] = {}
    for obj in items:
        did = str(obj.get("dialogue_id", ""))
        if did:
            meta_map[did] = obj
    return meta_map

def fragments_list_for(meta_map: Dict[str, Dict[str, Any]], dialogue_id: str, frag_id: int) -> Optional[List[int]]:
    md = meta_map.get(str(dialogue_id))
    if not md:
        return None
    frags = md.get("fragments")
    if not isinstance(frags, list):
        return None
    if 0 <= frag_id < len(frags) and isinstance(frags[frag_id], list):
        return [int(x) for x in frags[frag_id]]
    return None

def map_local_to_global_utt_ids(local_utt_ids: List[int], frag_global_ids: Optional[List[int]]) -> Tuple[List[int], bool]:
    """
    把“片段内局部ID”映射到“原始对话全局ID”。
    - 若 local_utt_ids 全部落在 [0, len(frag_global_ids))，按索引映射；
    - 否则视为已是全局ID，原样返回。
    返回：(mapped_ids, used_local_mapping_flag)
    """
    if not frag_global_ids:
        return local_utt_ids, False
    L = len(frag_global_ids)
    if all((0 <= u < L) for u in local_utt_ids):
        return [frag_global_ids[u] for u in local_utt_ids], True
    return local_utt_ids, False

# ============ 诊断 ============
def diagnose_missing_fragments(cand_list, meta_for_did, dialogue_id):
    """
    打印该 dialogue 下哪些 fragment 没被任何候选命中，以及原因计数。
    先将局部ID映射为全局ID，再判断是否命中 fragment。
    """
    import json

    frags = meta_for_did.get("fragments") or []
    frag_count = int(meta_for_did.get("fragment_count", len(frags)))

    per_fid_hit = {fid: False for fid in range(frag_count)}
    reasons = {"fid_out_of_range": 0, "all_ids_oof": 0, "empty_response": 0, "ok": 0}

    for r in cand_list:
        resp = (r.get("response") or "").strip()
        utt_ids_local = parse_utt_ids_from_response(resp)
        if not utt_ids_local:
            reasons["empty_response"] += 1
            continue

        fid = int(r.get("fragment_id", -1))
        if not (0 <= fid < frag_count):
            reasons["fid_out_of_range"] += 1
            continue

        frag_global = frags[fid] if (0 <= fid < len(frags)) else []
        utt_ids_global, _ = map_local_to_global_utt_ids(utt_ids_local, frag_global)

        allow = set(int(x) for x in frag_global)
        kept = [u for u in utt_ids_global if u in allow]
        if not kept:
            reasons["all_ids_oof"] += 1
            continue

        per_fid_hit[fid] = True
        reasons["ok"] += 1

    missing = sorted([fid for fid, hit in per_fid_hit.items() if not hit])

    print(json.dumps({
        "dialogue_id": dialogue_id,
        "frag_count": frag_count,
        "missing_fragments": missing,
        "reasons": reasons
    }, ensure_ascii=False))

# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser(
        description="按 query_text 聚合候选片段（仅使用 <|utt_ids|>）；把片段“局部ID”映射为对话“全局ID”，再按 fragment 元数据过滤并汇总"
    )
    ap.add_argument("--infer", default='', help="推理结果 .json/.jsonl（每条为一个候选片段，含字段 response）")
    ap.add_argument("--fragmeta", default='', help="原始对话分片信息 .json/.jsonl（含 fragments 等）")
    ap.add_argument("--output", default='', help="输出 .json")
    ap.add_argument("--keep-duplicate", action="store_true",
                    help="是否保留同一 dialogue_id 下重复 utt_ids 的候选（默认去重）")
    ap.add_argument("--diagnose", action="store_true", help="打印每个 dialogue 的缺失 fragment 诊断信息")
    args = ap.parse_args()

    records = load_any(args.infer)
    meta_map = load_fragment_meta(args.fragmeta)

    print(f"[INFO] loaded infer items: {len(records)}; fragment metas: {len(meta_map)}")

    # 1) 固定按 query_text 分组
    by_qtext: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        qtext = get_query_text(r)
        by_qtext.setdefault(qtext, []).append(r)

    outputs = []
    for qtext, items in by_qtext.items():
        if not items:
            continue

        # 2) 组内按 dialogue_id 分组
        by_did: Dict[str, List[Dict[str, Any]]] = {}
        for r in items:
            did = str(r.get("dialogue_id", ""))
            by_did.setdefault(did, []).append(r)

        dialogue_results = []
        for did, cand_list in by_did.items():
            per_frag_pred: Dict[int, Dict[str, Any]] = {}
            seen_sig = set()

            md = meta_map.get(did, {})
            frags = (md or {}).get("fragments") or []
            frag_count = int(md.get("fragment_count", len(frags)))

            if args.diagnose and md:
                diagnose_missing_fragments(cand_list, md, did)

            for r in cand_list:
                # 只从 sample['response'] 解析局部 utt_ids
                utt_ids_local = parse_utt_ids_from_response(r.get("response", ""))
                if not utt_ids_local:
                    continue

                frag_id = int(r.get("fragment_id", -1))
                frag_global = fragments_list_for(meta_map, did, frag_id)

                # ---- 核心：局部 -> 全局 ----
                utt_ids_global, used_local = map_local_to_global_utt_ids(utt_ids_local, frag_global)

                # 去重签名（对同一个片段，仅基于“全局ID序列”）
                sig = (frag_id, tuple(utt_ids_global))
                if (not args.keep_duplicate) and sig in seen_sig:
                    continue
                seen_sig.add(sig)

                # 过滤到该 fragment（允许为空 meta 时跳过过滤）
                oof_utt = []
                if frag_global is not None:
                    allow = set(frag_global)
                    in_utt, out_utt, seen_local = [], [], set()
                    for u in utt_ids_global:
                        if u in allow:
                            if u not in seen_local:
                                seen_local.add(u)
                                in_utt.append(u)
                        else:
                            out_utt.append(u)
                    utt_ids_global = in_utt
                    oof_utt = uniq_preserve_order(out_utt)

                if not utt_ids_global:
                    continue

                node = per_frag_pred.setdefault(frag_id, {
                    "utt_ids": [],
                    "oof_utt_ids": [],
                    "used_candidates": 0
                })
                node["utt_ids"] = uniq_preserve_order([*node["utt_ids"], *utt_ids_global])
                node["oof_utt_ids"] = uniq_preserve_order([*node["oof_utt_ids"], *oof_utt])
                node["used_candidates"] += 1

            if not per_frag_pred:
                continue

            # 3) 按 fragment_id 升序合并（全局ID）
            ordered_frag_ids = sorted(per_frag_pred.keys())
            merged_utt = []
            for fid in ordered_frag_ids:
                merged_utt.extend(per_frag_pred[fid]["utt_ids"])

            result = {
                "dialogue_id": did,
                "fragment_count": frag_count,
                "merged": {
                    "utt_ids": merged_utt
                },
                "covered_fragments": sorted([fid for fid in per_frag_pred.keys() if fid >= 0]),
            }
            dialogue_results.append(result)

        outputs.append({
            "query_text": qtext,
            "n_candidates": len(items),
            "results": dialogue_results
        })

    save_json(outputs, args.output)
    print(f"[OK] wrote {len(outputs)} query groups -> {args.output}")

if __name__ == "__main__":
    main()

