#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import Any, Dict, List

def _speaker_name(raw: str) -> str:
    """统一成 User1 / User2（其余保持首字母大写）"""
    if not isinstance(raw, str):
        return "User"
    raw = raw.strip()
    if raw.lower() in ("user1", "user 1", "u1"):
        return "User1"
    if raw.lower() in ("user2", "user 2", "u2"):
        return "User2"
    return raw[:1].upper() + raw[1:]

def _extract_images_from_utterance(utt: Any) -> List[str]:
    """
    从一次 utterance 中提取图片 URL/路径。
    可能的结构：
      - str：纯文本 -> 无图
      - list[dict]: [{"image_url": "...", "caption": ""}, ...]
      - 兼容键名: image_url / image / path
    """
    imgs: List[str] = []
    if isinstance(utt, list):
        for it in utt:
            if isinstance(it, dict):
                path = it.get("image_url") or it.get("image") or it.get("path")
                if isinstance(path, str) and path:
                    imgs.append(path)
    return imgs

def _count_images_in_utterance(utt: Any) -> int:
    return len(_extract_images_from_utterance(utt))

def _utterance_text(utt: Any) -> str:
    """
    将一次 utterance 转成目标里的文本片段：
      - 纯文本：原文
      - 仅图片：用若干个 <image> 占位，空格分隔
    """
    if isinstance(utt, str):
        return utt
    # 图片列表
    n_img = _count_images_in_utterance(utt)
    if n_img > 0:
        return " ".join(["<image>"] * n_img)
    # 兜底
    return ""

def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def convert_one(record: Dict[str, Any]) -> Dict[str, Any]:
    dialogue_id = str(record.get("dialogue_id", ""))

    # 1) 收集 images（按出现顺序，不去重！）
    all_images: List[str] = []
    for turn in record.get("dialogue", []):
        utt = turn.get("utterance")
        all_images.extend(_extract_images_from_utterance(utt))
    images = all_images  # 关键：不再去重

    # 2) 拼 user content（带编号与说话人）
    lines: List[str] = []
    total_img_tokens = 0
    for turn in record.get("dialogue", []):
        idx = turn.get("utterance_idx", 0)
        spk = _speaker_name(turn.get("speaker", "User"))
        text = _utterance_text(turn.get("utterance"))
        # 统计 <image> 个数用于一致性校验
        if isinstance(text, str):
            total_img_tokens += text.count("<image>")
        lines.append(f"<|utt_id_start|>{idx}<|utt_id_end|> {spk}: {text}")

    instruction = (
        "请根据下列多模态长对话内容，按照主题语义将对话划分为多个子片段。\n\n"
        "每条发言以<|utt_id_start|>句子ID<|utt_id_end|>标注，可能为文本、图像。请将语义连贯的句子归为一组。\n\n"
        "输出格式严格为：[[id,id,...],[id,...],...]，仅输出句子ID，不含其他内容。\n\n"
        "对话内容如下：\n"
    )
    user_content = instruction + "\n".join(lines)

    # 3) solution：把 gt_turns（列表的列表）转成字符串
    gt = record.get("gt_turns", [])
    try:
        import json as _json
        solution_str = _json.dumps(gt, ensure_ascii=False)
    except Exception:
        solution_str = "[]"

    # 4) 一致性校验（可选：不一致时丢弃或报警）
    if len(images) != total_img_tokens:
        print(
            f"[WARN] dialogue_id={dialogue_id}: images({len(images)}) "
            f"!= <image> tokens({total_img_tokens})."
        )
        # 如果你想强制一致，可以在这里选择：
        # - 丢弃样本：raise ValueError(...)
        # - 或者修补：比如截断多余图片或追加占位符（不推荐改文本）
        # 这里先仅告警，不改动

    # 5) 组装目标结构
    out = {
        "images": images,                 # 不去重，保证与 <image> 个数一致
        "dialogue_id": dialogue_id,
        "solution": solution_str,
        "messages": [
            {"role": "system", "content": "你是一个多模态对话分段助手。"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": solution_str},
        ],
    }
    return out
    

def main():
    ap = argparse.ArgumentParser(
        description="将原始多模态对话样本转换为分段任务格式（含 images / messages / solution）"
    )
    ap.add_argument("--input", default='data/Dialogcc/process_0825/pipeline_process/0_wechat_dialogue_v1.json', help="输入 JSON 文件（list 或单条 dict）")
    ap.add_argument("--output", default='data/Dialogcc/process_0825/pipeline_process/1_wechat_dialogue_split.json', help="输出 JSON 文件")
    ap.add_argument("--no-pretty", action="store_true", help="不进行缩进，单行输出")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容：单条 -> 列表
    if isinstance(data, dict):
        data = [data]
    assert isinstance(data, list), "输入 JSON 顶层应为 list 或单个样本 dict"

    converted = []
    for rec in data:
        try:
            converted.append(convert_one(rec))
        except Exception as e:
            # 出错也不中断，打印并跳过
            print(f"[WARN] 转换样本失败 dialogue_id={rec.get('dialogue_id')}: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"[OK] 转换完成：{len(converted)} 条，已写入 {args.output}")

if __name__ == "__main__":
    main()