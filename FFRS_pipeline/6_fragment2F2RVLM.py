#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple

# ---------- helpers ----------

IMG_TOKEN = "<image>"

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
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

def load_fragments(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    读取 3_prediction_fragment.json，建立 (dialogue_id, fragment_id) -> 片段 的索引
    """
    data = load_json_or_jsonl(path)
    table: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in data:
        did = str(r.get("dialogue_id", ""))
        fid = int(r.get("fragment_id", 0))
        table[(did, fid)] = r
    return table

def parse_key(key: str) -> Tuple[str, int]:
    """
    从 key 里解析出 dialogue_id 和 fragment_id
    兼容形如 '4503313089_1_1+0'（左边可能带下划线等），以最后一个 '+' 为界。
    """
    if "+" not in key:
        raise ValueError(f"bad key: {key}")
    left, right = key.rsplit("+", 1)
    return left, int(right)

def speaker_to_lower(line: str) -> str:
    """
    把 'User1: xxx' / 'User2: xxx' 标成小写开头的 'user1: xxx' 等（仅美化，非必需）
    """
    m = re.match(r"^\s*([^:：]+)\s*[:：]\s*(.*)$", line)
    if not m:
        return line
    spk = m.group(1).strip()
    rest = m.group(2)
    spk = spk.lower()
    return f"{spk}: {rest}"

def annotate_images_in_line(line: str, next_img_id: int) -> Tuple[str, int]:
    """
    把一行中出现的每个 <image> 替换为 <|img_id_start|>i<|img_id_end|><image>，并递增 i
    返回 (替换后文本, 更新后的 next_img_id)
    """
    parts = line.split(IMG_TOKEN)
    if len(parts) == 1:
        return line, next_img_id
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i < len(parts) - 1:
            out.append(f"<|img_id_start|>{next_img_id}<|img_id_end|>{IMG_TOKEN}")
            next_img_id += 1
    return "".join(out), next_img_id

def build_dialogue_block(resp_text: str) -> Tuple[str, int]:
    """
    把片段的 response（多行 'UserX: ...'）转成：
      <|utt_id_start|>0<|utt_id_end|> user1: ...
      <|utt_id_start|>1<|utt_id_end|> user2: ...
      ...
    同时在文本中标注每个 <image> 的 img_id 序号（从 0 递增）。
    返回：(拼好的多行字符串, 总图片数)
    """
    lines = [ln for ln in resp_text.splitlines() if ln.strip() != ""]
    buf: List[str] = []
    next_img_id = 0
    for i, ln in enumerate(lines):
        ln2 = speaker_to_lower(ln)
        ln2, next_img_id = annotate_images_in_line(ln2, next_img_id)
        buf.append(f"<|utt_id_start|>{i}<|utt_id_end|> {ln2}")
    return "\n".join(buf), next_img_id

def make_user_prompt(query: str, dialogue_block: str) -> str:
    head = (
        f"请根据下列多模态长对话内容,检索与<|query_start|>{query}<|query_end|>匹配的句子和图片.\n"
        "要求: \n"
        "1.检索结果以句子ID和图片ID输出，若不存在匹配的句子或图片，对应部分返回空列表\n"
        "2.格式严格遵循:<|utt_ids_start|>[句子ID列表]<|utt_ids_end|>;"
        "<|img_ids_start|>[图片ID列表]<|img_ids_end|>\n\n"
        "对话内容: "
    )
    return head + dialogue_block + "\n"

def make_solution_str(utt_ids: List[int], img_ids: List[int]) -> str:
    return (
        f"<|utt_ids_start|>{json.dumps(utt_ids, ensure_ascii=False)}<|utt_ids_end|>;"
        f"<|img_ids_start|>{json.dumps(img_ids, ensure_ascii=False)}<|img_ids_end|>"
    )

# ---------- main pipeline ----------

def main():
    ap = argparse.ArgumentParser(
        description="把检索结果(top20) + 片段库(3_prediction_fragment.json) 组装成多模态检索训练样本 JSON"
    )
    ap.add_argument("--retrieval", default='',
                    help="检索结果 JSON/JSONL（含 dialogue_id, query_text, gt_turns, top20[key,...]）")
    ap.add_argument("--fragments", default='',
                    help="3_prediction_fragment.json（或 JSONL），含 {dialogue_id, response, images, fragment_id}")
    ap.add_argument("--output", default='', help="输出 JSON 文件（list）")
    ap.add_argument("--topn", type=int, default=20, help="对每条 query 取前 N 个片段")
    ap.add_argument("--fill-gt-when-same-dialogue", action="store_true",
                    help="若候选片段与query同 dialogue_id，则用原 gt_turns 作为 utt_ids（img_ids 仍为空）")
    args = ap.parse_args()

    # 1) 载入
    items = load_json_or_jsonl(args.retrieval)
    frag_table = load_fragments(args.fragments)
    print(f"[INFO] loaded queries={len(items)}, fragments={len(frag_table)}")

    out_records: List[Dict[str, Any]] = []

    # 2) 遍历每条 query 的 topN
    for it in items:
        q_did = str(it.get("dialogue_id", ""))
        query = str(it.get("query_text", "") or "")
        gt_turns = it.get("gt_turns", []) or []
        top_list = it.get("top20", [])[: args.topn]

        for cand in top_list:
            key = str(cand.get("key", ""))
            try:
                did, fid = parse_key(key)
            except Exception:
                # 跳过无法解析的 key
                continue

            frag = frag_table.get((did, fid))
            if frag is None:
                # 片段库未命中，跳过
                continue

            resp = str(frag.get("response", "") or "")
            images = frag.get("images", []) or []
            # 构建对话块 + 标注图片 id
            dialogue_block, img_count = build_dialogue_block(resp)

            # 组装 messages
            user_content = make_user_prompt(query, dialogue_block)

            # solution：默认空
            utt_sol: List[int] = []
            img_sol: List[int] = []

            # 可选：同对话时，用原 gt_turns 作为答案（注意这与本地编号未对齐，仅在你确认片段=整段且编号一致时再开启）
            if args.fill_gt_when_same_dialogue and (did == q_did) and isinstance(gt_turns, list):
                # 强行使用原始 gt_turns 作为 utt_ids
                utt_sol = [int(x) for x in gt_turns if isinstance(x, int)]

            solution = make_solution_str(utt_sol, img_sol)

            rec = {
                "messages": [
                    {"role": "system", "content": "你是一个多模态信息检索助手。"},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": solution},
                ],
                "images": images,                 # 与 <|img_id_start|> 编号一致（0..img_count-1）
                "dialogue_id": did,               # 用候选片段的对话 ID
                "fragment_id": int(frag.get("fragment_id", 0)),
                "solution": solution,             # 再冗余一份在顶层，方便训练脚本读取
                # 额外留点溯源信息（如不需要可删掉）
                "meta": {
                    "source_query_dialogue_id": q_did,
                    "query_text": query,
                    "key": key,
                    "rank": int(cand.get("rank", 0)),
                    "score": float(cand.get("score", 0.0)),

                }
            }
            out_records.append(rec)

    # 3) 写出
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {len(out_records)} records to {args.output}")

if __name__ == "__main__":
    main()