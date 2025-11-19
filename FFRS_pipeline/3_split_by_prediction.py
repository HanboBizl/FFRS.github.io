
import json
import re
import ast
import argparse
from typing import Any, Dict, List, Tuple, Iterable, Union
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # 简化依赖

IMG_TOKEN = "<image>"
UTT_LINE_RE = re.compile(
    r"\<\|utt_id_start\|\>(\d+)\<\|utt_id_end\|\>\s*([^:：]+)\s*[:：]\s*(.*)$"
)

# ---------------- utils ----------------

def parse_segments(seg: Union[str, list]) -> List[List[int]]:
    """把 '[[0,1],[2,3]]' / 已解析列表 -> list[list[int]]，失败返回空列表。"""
    if seg is None:
        return []
    if isinstance(seg, list):
        return [[int(x) for x in g] for g in seg]
    s = str(seg).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

def pick_user_message(messages: List[Dict[str, Any]]) -> str:
    """从 messages 里找 user 的 content（整段对话文本）。兜底取第一个有 content 的。"""
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "") or ""
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str) and c:
            return c
    return ""

def parse_dialogue_text(user_content: str) -> Dict[int, Dict[str, Any]]:
    """
    解析：
    <|utt_id_start|>i<|utt_id_end|> UserX: 内容...
    返回 {i: {"speaker": "UserX", "text": "..."}}；text 可能含 <image>
    """
    out: Dict[int, Dict[str, Any]] = {}
    for ln in user_content.splitlines():
        m = UTT_LINE_RE.match(ln.strip())
        if not m:
            continue
        idx = int(m.group(1))
        speaker = m.group(2).strip()
        text = m.group(3)
        out[idx] = {"speaker": speaker, "text": text}
    return out

def extract_image_paths(images_field: Any) -> List[str]:
    """
    支持：
      - [{'path': '...'}, ...] / [{'image_url': '...'}, ...] / [{'url': '...'}, ...]
      - ['...', ...]
    忽略 bytes / None / 空字符串。
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

def count_img_tokens(text: str) -> int:
    return text.count(IMG_TOKEN)

def assign_images_per_utt(utt_map: Dict[int, Dict[str, Any]], all_image_paths: List[str]) -> Dict[int, List[str]]:
    """
    按 utterance 顺序，把全局 images 逐个分配到每条 utt 的 <image> 占位。
    图片不够：该 utt 少图；图片过多：剩余图片丢弃。
    """
    assigned: Dict[int, List[str]] = {}
    img_ptr = 0
    for uid in sorted(utt_map.keys()):
        need = count_img_tokens(utt_map[uid]["text"])
        cur = []
        for _ in range(need):
            if img_ptr < len(all_image_paths):
                cur.append(all_image_paths[img_ptr])
                img_ptr += 1
        assigned[uid] = cur
    return assigned

def make_query_from_fragment(utt_ids: List[int], utt_map: Dict[int, Dict[str, Any]]) -> str:
    """取片段内第一条非纯图片的文本作为 query，去掉 <image> 与多余空白。"""
    for uid in sorted(utt_ids):
        t = utt_map.get(uid, {}).get("text", "")
        if not t:
            continue
        plain = t.replace(IMG_TOKEN, "").strip()
        if plain:
            return re.sub(r"\s+", " ", plain)
    return ""

def build_response_text(utt_ids: List[int], utt_map: Dict[int, Dict[str, Any]]) -> str:
    """拼接片段文本：UserX: 原文（保留 <image>），按 utt_id 升序，一行一个。"""
    lines = []
    for uid in sorted(utt_ids):
        u = utt_map.get(uid)
        if u:
            lines.append(f'{u["speaker"]}: {u["text"]}')
    return "\n".join(lines)

def collect_fragment_images(utt_ids: List[int], per_utt_imgs: Dict[int, List[str]]) -> List[str]:
    """按 utt 顺序收集图片路径，保持出现顺序。"""
    out: List[str] = []
    for uid in sorted(utt_ids):
        out.extend([p for p in per_utt_imgs.get(uid, []) if p])
    return out

def validate_coverage(utt_map: Dict[int, Dict[str, Any]], seg_groups: List[List[int]]) -> Dict[str, Any]:
    """检查 response 覆盖情况，返回告警信息（不阻断流程）。"""
    utt_ids = sorted(utt_map.keys())
    resp_ids = [x for g in seg_groups for x in g]
    missing = sorted(list(set(utt_ids) - set(resp_ids)))
    extra = sorted(list(set(resp_ids) - set(utt_ids)))
    dup = sorted([k for k, c in Counter(resp_ids).items() if c > 1])
    return {
        "total_utt": len(utt_ids),
        "covered": len(missing) == 0,
        "missing_ids": missing,
        "extra_ids": extra,
        "duplicate_ids": dup,
        "all_utt_ids": utt_ids,
        "union_pred_ids": sorted(set(resp_ids)),
        "fragments": seg_groups,
        "fragment_count": len(seg_groups),
        "fragment_lengths": [len(g) for g in seg_groups],
    }

# ------------- core ---------------

def convert_one_sample(sample: Dict[str, Any], query_strategy: str = "empty", do_validate: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    按 response 分段生成多个片段：
      返回 (片段列表, 覆盖告警字典, 分段摘要summary)
    片段元素：
      {dialogue_id, query, response, images, fragment_id}
    摘要：
      {dialogue_id, fragment_count, fragments, fragment_lengths, all_utt_ids, covered, missing_ids, extra_ids, duplicate_ids, ...}
    """
    dialogue_id = str(sample.get("dialogue_id", ""))
    seg_groups = parse_segments(sample.get("response", "[]"))
    user_content = pick_user_message(sample.get("messages", []))
    utt_map = parse_dialogue_text(user_content)
    all_img_paths = extract_image_paths(sample.get("images", []))
    per_utt_imgs = assign_images_per_utt(utt_map, all_img_paths)

    warn = {}
    if do_validate:
        warn = validate_coverage(utt_map, seg_groups)

    out: List[Dict[str, Any]] = []
    for fid, group in enumerate(seg_groups):
        resp_text = build_response_text(group, utt_map)
        imgs = collect_fragment_images(group, per_utt_imgs)
        if query_strategy == "first_text":
            query = make_query_from_fragment(group, utt_map)
        else:
            query = ""
        out.append({
            "dialogue_id": dialogue_id,
            "query": query,
            "response": resp_text,
            "images": imgs,
            "fragment_id": fid
        })

    # 生成摘要
    summary = {
        "dialogue_id": dialogue_id,
        "fragment_count": len(seg_groups),
        "fragments": seg_groups,                           # [[ids], [ids], ...]
        "fragment_lengths": [len(g) for g in seg_groups],  # 每个fragment覆盖的ID数量
        "all_utt_ids": sorted(utt_map.keys()),            # 原始对话中全部句子ID
        "covered": bool(warn.get("covered", False)),
        "missing_ids": warn.get("missing_ids", []),
        "extra_ids": warn.get("extra_ids", []),
        "duplicate_ids": warn.get("duplicate_ids", []),
        "total_utt": warn.get("total_utt", len(utt_map)),
        "union_pred_ids": warn.get("union_pred_ids", []),
    }

    return out, warn, summary

# ---------- IO (json / jsonl) ----------

def iter_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """自动识别 json / jsonl。json 允许是单个 dict 或 list[dict]。"""
    p = Path(path)
    is_jsonl = p.suffix.lower() in {".jsonl", ".ndjson"}

    if is_jsonl:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception as e:
                    print(f"[WARN] JSONL line {ln} 解析失败: {e}")
        return

    # JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                yield it
            else:
                print("[WARN] JSON 列表中含非对象，已跳过一项。")
    else:
        print("[WARN] 非法 JSON 顶层结构，已跳过。")

def write_output(objs: Iterable[Dict[str, Any]], path: str, jsonl: bool):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with open(path, "w", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    else:
        buf = list(objs)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(buf, f, ensure_ascii=False, indent=2)

# ------------- main ---------------

def main():
    ap = argparse.ArgumentParser(description="按 response 分段，导出片段样本（支持 JSON/JSONL 流式处理）")
    ap.add_argument("--input", default='', help="输入 JSON（list 或单条 dict）")
    ap.add_argument("--output", default='', help="输出 JSON（list）")
    ap.add_argument("--summary-output", default=None, help="分段摘要 JSON（每个 dialogue 的片段ID与覆盖检查）。默认与 --output 同名加 _frag_summary.json")
    ap.add_argument("--query-strategy", choices=["empty", "first_text"], default="empty",
                    help="query 生成策略：留空 / 取片段首条非纯图片文本")
    ap.add_argument("--no-validate", action="store_true", help="不做覆盖校验（更快）")
    ap.add_argument("--show-warn", action="store_true", help="打印存在缺失/多余/重复 ID 的样本")
    args = ap.parse_args()

    out_is_jsonl = Path(args.output).suffix.lower() in {".jsonl", ".ndjson"}

    # 默认 summary 输出路径
    if args.summary_output is None:
        op = Path(args.output)
        args.summary_output = str(op.with_name(op.stem + "_frag_summary.json"))

    produced = 0
    warned = 0
    summaries: List[Dict[str, Any]] = []

    def gen():
        nonlocal produced, warned, summaries
        for sample in tqdm(iter_json_or_jsonl(args.input), desc="Converting"):
            try:
                frags, warn, summary = convert_one_sample(
                    sample,
                    query_strategy=args.query_strategy,
                    do_validate=not args.no_validate
                )
                summaries.append(summary)

                if args.show_warn and warn:
                    if (not warn["covered"]) or warn["extra_ids"] or warn["duplicate_ids"]:
                        warned += 1
                        print(json.dumps({
                            "dialogue_id": sample.get("dialogue_id"),
                            "warn": warn
                        }, ensure_ascii=False))
                for f in frags:
                    produced += 1
                    yield f
            except Exception as e:
                print(f"[WARN] dialogue_id={sample.get('dialogue_id')}: {e}")

    # 写片段样本
    write_output(gen(), args.output, jsonl=out_is_jsonl)
    # 写分段摘要
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"[OK] 片段共输出 {produced} 条 -> {args.output}")
    print(f"[OK] 分段摘要已写入 -> {args.summary_output}")
    if args.show_warn:
        print(f"[WARN] 有覆盖问题的样本：{warned} 条（仅告警，未中断）")

if __name__ == "__main__":
    main()