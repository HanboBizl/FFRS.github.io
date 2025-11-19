

import argparse
import json
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Iterable, FrozenSet

# ---------- IO ----------

def load_json_any(path: str):
    """读取 .json 或 .jsonl；顶层可为 list 或单对象。"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
        return data
    except json.JSONDecodeError:
        # jsonl 兜底
        out = []
        for ln in txt.splitlines():
            s = ln.strip()
            if s:
                out.append(json.loads(s))
        return out

# ---------- utils ----------

def group_consecutive(sorted_ids: List[int], gap: int = 1) -> List[List[int]]:
    """把已排好序的 id 列表按连续性分组（默认差值=1 连续）"""
    if not sorted_ids:
        return []
    groups = [[sorted_ids[0]]]
    for x in sorted_ids[1:]:
        if x - groups[-1][-1] <= gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def to_file_url(path: str) -> str:
    if path.startswith("file://"):
        return path
    return "file://" + path

# 将本地路径映射为公网 URL
def to_web_img_url(p: str, local_prefix: str, web_prefix: str) -> str:
    if not isinstance(p, str):
        return ""
    p = p.strip()
    if not p:
        return ""
    if p.startswith(local_prefix):
        tail = p[len(local_prefix):].lstrip("/")
        return web_prefix.rstrip("/") + "/" + tail
    # 兜底：路径里包含 hanbobi/Wechat/
    m = re.search(r"hanbobi/Wechat/(.+)$", p)
    if m:
        return web_prefix.rstrip("/") + "/" + m.group(1)
    return p

def build_dialogue_map(corpus: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """dialogue_id -> 对话数组"""
    mp: Dict[str, List[Dict[str, Any]]] = {}
    for obj in corpus:
        did = str(obj.get("dialogue_id", "") or "")
        dlg = obj.get("dialogue") or []
        if did:
            mp[did] = dlg
    return mp

def normalize_text(s: str) -> str:
    """规整文本：压缩空白、去首尾空格"""
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_id(s: str) -> str:
    """生成可用于 HTML id 的安全字符串"""
    return re.sub(r"[^a-zA-Z0-9_\-:]", "_", s or "")

# ---------- 去重 helpers（按“命中文本集合 × 图片集合”去重） ----------

def _utt_by_idx(dialogue: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(u.get("utterance_idx", -1)): u for u in (dialogue or [])}

def _content_sets_for_dialogue(
    dlg: List[Dict[str, Any]],
    utt_ids: List[int],
    url_mapper
) -> Tuple[FrozenSet[Tuple[str, str]], FrozenSet[str]]:
    """
    把“命中句子”提取为两类集合：
      text_set: {(speaker, normalized_text), ...}
      img_set : {mapped_image_url, ...}
    缺失的命中句用 ('__MISSING__', f'UID:{uid}') 占位，避免误合并。
    """
    idx2utt = _utt_by_idx(dlg)
    text_items: List[Tuple[str, str]] = []
    img_items: List[str] = []
    for uid in utt_ids:
        u = idx2utt.get(int(uid))
        if not u:
            text_items.append(("__MISSING__", f"UID:{uid}"))
            continue
        spk = str(u.get("speaker", "") or "")
        content = u.get("utterance")
        if isinstance(content, list) and content and isinstance(content[0], dict) and "image_url" in content[0]:
            for it in content:
                if isinstance(it, dict) and "image_url" in it:
                    url = url_mapper(str(it.get("image_url", "") or ""))
                    img_items.append(url)
        else:
            txt = content if isinstance(content, str) else ""
            text_items.append((spk, normalize_text(txt)))
    return frozenset(text_items), frozenset(img_items)

def _is_dominated(tA: FrozenSet[Tuple[str, str]], iA: FrozenSet[str],
                  tB: FrozenSet[Tuple[str, str]], iB: FrozenSet[str]) -> bool:
    """A 是否被 B 支配（A ⊆ B 且至少一处严格包含）"""
    return (tA.issubset(tB) and iA.issubset(iB)) and (tA != tB or iA != iB)

def dedup_results(
    results: List[Dict[str, Any]],
    did2dlg: Dict[str, List[Dict[str, Any]]],
    url_mapper,
    mode: str = "subset"  # "subset" | "maximal" | "exact"
) -> List[Dict[str, Any]]:
    """
    - subset  ：按顺序保留第一个；后面只要“相等或包含关系（互为子集）”均丢弃后者。（默认）
    - maximal ：两两比较，去掉被“更大集合”支配的结果（保留最大元）。完全相等则保留最早出现的一个。
    - exact   ：仅严格相等时才去重；保留第一个。
    """
    if not results:
        return results

    # 预先计算“文本集合 × 图片集合”的签名
    sigs: List[Tuple[FrozenSet[Tuple[str, str]], FrozenSet[str]]] = []
    for r in results:
        did = str(r.get("dialogue_id", "") or "")
        utt_ids = list(map(int, (r.get("merged", {}) or {}).get("utt_ids", []) or []))
        dlg = did2dlg.get(did, [])
        tset, iset = _content_sets_for_dialogue(dlg, utt_ids, url_mapper)
        sigs.append((tset, iset))

    if mode == "subset":
        kept, kept_sigs = [], []
        for r, sig in zip(results, sigs):
            tS, iS = sig
            is_dup = False
            for tK, iK in kept_sigs:
                # 相等或互为包含都视为重复，保留第一个
                if (tS == tK and iS == iK) or \
                   (tS.issubset(tK) and iS.issubset(iK)) or \
                   (tK.issubset(tS) and iK.issubset(iS)):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)
                kept_sigs.append(sig)
        return kept

    if mode == "maximal":
        n = len(results)
        keep = [True] * n

        # 先去掉严格相等的重复：保留最早的一个
        first_seen: Dict[Tuple[FrozenSet[Tuple[str, str]], FrozenSet[str]], int] = {}
        for i, sig in enumerate(sigs):
            if sig in first_seen:
                keep[i] = False
            else:
                first_seen[sig] = i

        # 再做支配判断：被更大集合覆盖的都丢弃（不区分先后）
        for i in range(n):
            if not keep[i]:
                continue
            tA, iA = sigs[i]
            for j in range(n):
                if i == j or not keep[i]:
                    continue
                tB, iB = sigs[j]
                if _is_dominated(tA, iA, tB, iB):
                    keep[i] = False
        return [r for i, r in enumerate(results) if keep[i]]

    # exact
    kept, seen = [], set()
    for r, sig in zip(results, sigs):
        if sig in seen:
            continue
        kept.append(r)
        seen.add(sig)
    return kept

# ---------- HTML 生成 ----------

CSS = """
<style>
  :root{
    --bg:#fff;
    --fg:#111827;
    --muted:#6b7280;
    --border:#e5e7eb;
    --chip-bg:#eef2ff;
    --chip-fg:#3730a3;
    --hit:#fff7ed;
    --ctx:#f9fafb;
  }
  html,body{background:#fafafa}
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;
    margin:24px;
  }
  h1{font-size:22px;margin:0 0 16px}
  h2{font-size:20px;margin:24px 0 8px}
  small{color:#666}
  hr.sep{border:0;border-top:1px solid #eee;margin:24px 0}

  .query-nav{margin:12px 0 24px}
  .query-nav a{display:inline-block;margin:4px 8px 4px 0;padding:4px 8px;background:#f5f5f7;border-radius:6px;color:#333;text-decoration:none}
  .query-nav a:hover{background:#eaeaec}

  .card{border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin:12px 0;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .meta{color:#555;font-size:12px;margin-bottom:8px}
  .badge{display:inline-block;font-size:12px;padding:2px 6px;border-radius:999px;background:var(--chip-bg);color:var(--chip-fg);margin-left:6px}

  .block{border-left:3px solid var(--border);padding-left:8px;margin:10px 0;background:#fff}

  /* 一行展示：#idx | speaker：text | [images] */
  .turn{
    display:flex; align-items:flex-start; gap:8px;
    padding:6px 8px; border-radius:8px;
  }
  .turn .idx{
    color:#9ca3af; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;
    width:44px; flex:0 0 auto;
  }
  .turn .who{font-weight:600;color:#374151;flex:0 0 auto;white-space:nowrap}
  .turn .who::after{content:"："}
  .turn .text{color:var(--fg);white-space:pre-wrap;word-break:break-word;flex:1 1 auto}
  .turn .imgs{flex:1 1 auto;display:flex;flex-wrap:wrap;gap:8px}

  .hit{background:var(--hit)}
  .ctx{background:var(--ctx)}

  .thumb{
    max-height:120px;max-width:100%;
    border-radius:6px;border:1px solid var(--border);
    display:block;cursor:zoom-in;
  }

  .footer-note{color:var(--muted);font-size:12px;margin-top:8px}
  code{background:#f3f4f6;padding:1px 4px;border-radius:4px}

  /* 预览层（轻量 lightbox） */
  .img-modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:9999}
  .img-modal.show{display:flex}
  .img-modal img{max-width:92vw;max-height:92vh;border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,.5)}
  .img-modal .close{position:absolute;top:14px;right:18px;color:#fff;font-size:22px;cursor:pointer;user-select:none}

  /* 对话全文弹窗 */
  .dlg-modal{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;align-items:center;justify-content:center;z-index:9998}
  .dlg-modal.show{display:flex}
  .dlg-box{background:#fff;max-width:960px;width:92vw;max-height:92vh;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,.35);display:flex;flex-direction:column;overflow:hidden}
  .dlg-head{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #eee}
  .dlg-title{font-weight:600}
  .dlg-close{cursor:pointer;font-size:20px;color:#555}
  .dlg-body{padding:12px 14px;overflow:auto;background:#fafafa}
  .dlg-body .turn{background:#fff} /* 全文里的 turn 背景统一白色 */
</style>
"""

HTML_HEAD = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pipeline 可视化报告</title>
""" + CSS + """
<body>
<h1>Pipeline 可视化报告</h1>
"""

HTML_TAIL = """
<!-- 图片预览脚本 + 对话全文弹窗脚本 -->
<script>
(function(){
  // 轻量图片预览
  const imgModal = document.createElement('div');
  imgModal.className = 'img-modal';
  imgModal.innerHTML = '<span class="close" title="关闭">✕</span><img alt="preview">';
  document.body.appendChild(imgModal);
  const imgEl = imgModal.querySelector('img');
  const imgCloseBtn = imgModal.querySelector('.close');

  document.addEventListener('click', function(e){
    const img = e.target.closest('img.thumb');
    if (img){
      const full = img.getAttribute('data-full') || img.src;
      imgEl.src = full;
      imgModal.classList.add('show');
      e.preventDefault();
    }
    if (e.target === imgModal || e.target === imgCloseBtn){
      imgModal.classList.remove('show');
      imgEl.removeAttribute('src');
    }
  });
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape' && imgModal.classList.contains('show')){
      imgModal.classList.remove('show');
      imgEl.removeAttribute('src');
    }
  });

  // 对话全文弹窗
  const dlgModal = document.createElement('div');
  dlgModal.className = 'dlg-modal';
  dlgModal.innerHTML = '<div class="dlg-box"><div class="dlg-head"><div class="dlg-title"></div><span class="dlg-close" title="关闭">✕</span></div><div class="dlg-body"></div></div>';
  document.body.appendChild(dlgModal);
  const dlgTitle = dlgModal.querySelector('.dlg-title');
  const dlgBody  = dlgModal.querySelector('.dlg-body');
  const dlgClose = dlgModal.querySelector('.dlg-close');

  function openDialogue(targetId, did){
    const tpl = document.getElementById(targetId);
    if (!tpl) return;
    dlgTitle.textContent = did || '';
    dlgBody.innerHTML = tpl.innerHTML; // 填充全文 HTML
    dlgModal.classList.add('show');
  }

  document.addEventListener('click', function(e){
    const link = e.target.closest('a.dlg-link');
    if (link){
      const target = link.getAttribute('data-target');
      const did = link.getAttribute('data-did') || '';
      openDialogue(target, did);
      e.preventDefault();
    }
  });

  function closeDlg(){
    dlgModal.classList.remove('show');
    dlgBody.innerHTML = '';
    dlgTitle.textContent = '';
  }

  dlgClose.addEventListener('click', closeDlg);
  dlgModal.addEventListener('click', function(e){
    if (e.target === dlgModal) closeDlg();
  });
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape' && dlgModal.classList.contains('show')) closeDlg();
  });
})();
</script>
</body>
</html>
"""

def render_image_item(img_obj: Dict[str, Any],
                      url_mapper,
                      use_file_url: bool) -> str:
    url = str(img_obj.get("image_url", "") or "")
    cap = str(img_obj.get("caption", "") or "")

    if not url:
        return ""

    # 先做前缀映射
    url = url_mapper(url)

    # 非 http(s) 且需要 file://
    if use_file_url and not (url.startswith("http://") or url.startswith("https://")):
        url = to_file_url(url)

    # 可点击：a + 缩略图；点击会弹出预览（JS 拦截），也可 Cmd/Ctrl+点击在新标签打开
    img_html = (
        f'<a href="{html.escape(url)}" target="_blank" rel="noreferrer noopener">'
        f'  <img class="thumb" src="{html.escape(url)}" data-full="{html.escape(url)}" loading="lazy"/>'
        f'</a>'
    )
    if cap:
        img_html += f'<div class="footer-note">{html.escape(cap)}</div>'
    return img_html

def render_utterance(utt: Dict[str, Any],
                     cls: str,
                     url_mapper,
                     use_file_url: bool,
                     show_idx: bool = True) -> str:
    idx = int(utt.get("utterance_idx", -1))
    who = str(utt.get("speaker", "") or "")
    content = utt.get("utterance")

    # 文本或图片
    if isinstance(content, list) and content and isinstance(content[0], dict) and "image_url" in content[0]:
        thumbs = []
        for item in content:
            if isinstance(item, dict) and "image_url" in item:
                thumbs.append(render_image_item(item, url_mapper, use_file_url))
        body_html = f'<div class="imgs">{"".join(thumbs)}</div>'
    else:
        text = content if isinstance(content, str) else ""
        body_html = f'<div class="text">{html.escape(text)}</div>'

    idx_html = f'<div class="idx">#{idx}</div>' if show_idx else ""
    who_html = f'<div class="who">{html.escape(who)}</div>'
    return f'<div class="turn {cls}">{idx_html}{who_html}{body_html}</div>'

def render_dialogue_block(dialogue: List[Dict[str, Any]],
                          hit_ids: Set[int],
                          ctx: int,
                          url_mapper,
                          use_file_url: bool) -> str:
    """只渲染命中句子及其上下文，把连续的区段拆成多个 block。"""
    if not dialogue or not hit_ids:
        return ""

    all_idx = [int(x.get("utterance_idx", -1)) for x in dialogue]
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_idx)}

    sorted_hits = sorted([i for i in hit_ids if i in idx_to_pos])
    if not sorted_hits:
        return '<div class="footer-note">（命中 ID 未在原始对话中找到）</div>'

    groups = group_consecutive(sorted_hits, gap=1)
    out_parts = []
    for g in groups:
        lo, hi = g[0], g[-1]
        win_lo = clamp(idx_to_pos[lo] - ctx, 0, len(dialogue) - 1)
        win_hi = clamp(idx_to_pos[hi] + ctx, 0, len(dialogue) - 1)

        part = ['<div class="block">']
        for pos in range(win_lo, win_hi + 1):
            utt = dialogue[pos]
            uid = int(utt.get("utterance_idx", -1))
            cls = "hit" if uid in hit_ids else "ctx"
            part.append(
                render_utterance(utt, cls=cls, url_mapper=url_mapper, use_file_url=use_file_url)
            )
        part.append("</div>")
        out_parts.append("\n".join(part))

    return "\n".join(out_parts)

def render_full_dialogue(dlg: List[Dict[str, Any]],
                         url_mapper,
                         use_file_url: bool) -> str:
    """渲染完整对话（每条都用 ctx 风格）。"""
    if not dlg:
        return '<div class="footer-note">（该对话为空）</div>'
    parts = ['<div class="block">']
    for utt in dlg:
        parts.append(render_utterance(utt, cls="ctx", url_mapper=url_mapper, use_file_url=use_file_url))
    parts.append('</div>')
    return "\n".join(parts)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="将 run_pipeline 的合并结果与原始语料拼接成 HTML 可视化报告（支持包含/相等等价去重 & 点击 Dialogue 查看全文）"
    )
    ap.add_argument("--final_json",
                    default="",
                    help="step8 的合并结果（8_final_merged.json）")
    ap.add_argument("--corpus_json",
                    default="",
                    help="原始长对话（0_wechat_dialogue_v1.json）")
    ap.add_argument("--out",
                    default="",
                    help="输出 HTML 文件路径")
    ap.add_argument("--context", type=int, default=0,
                    help="命中句子上下文行数（默认0，仅展示命中）")
    ap.add_argument("--limit_dialogues", type=int, default=0,
                    help="每个 query 最大展示多少个对话（0=不限制）")
    ap.add_argument("--file_url_images", action="store_true",
                    help="把不是 http(s) 的本地图片路径转成 file:// URL（若前缀替换后仍为本地路径）")

    # 图片前缀替换（默认就是你要的前缀）
    ap.add_argument("--img_local_prefix",
                    default="Wechat/",
                    help="本地图片路径前缀")
    ap.add_argument("--img_web_prefix",
                    default="",
                    help="公网图片路径前缀（替换后会拼接本地前缀后的尾部）")

    # 去重控制
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="是否对同一 query 的结果去重（默认开启）")
    ap.add_argument("--dedup_mode", choices=["subset", "maximal", "exact"], default="maximal",
                    help="去重模式：subset=按顺序保留第一个；maximal=保留信息量最大的集合；exact=严格相等去重")

    args = ap.parse_args()

    final_data = load_json_any(args.final_json)
    corpus = load_json_any(args.corpus_json)
    did2dlg = build_dialogue_map(corpus)

    groups = final_data if isinstance(final_data, list) else [final_data]

    # URL 映射函数（闭包）
    def map_url(p: str) -> str:
        return to_web_img_url(p, args.img_local_prefix, args.img_web_prefix)

    # 导航（使用去重后的数量）
    nav_links = []
    for gi, g in enumerate(groups, start=1):
        qtext = str(g.get("query_text", "") or f"Query #{gi}")
        anchor = f"q{gi}"
        results_raw = g.get("results", []) or []
        results_nav = dedup_results(results_raw, did2dlg, map_url, mode=args.dedup_mode) if args.dedup else results_raw
        nav_links.append(
            f'<a href="#{anchor}">{html.escape(qtext)[:32] or "(empty)"}'
            f'<span class="badge">{len(results_nav)} dialogues</span></a>'
        )

    # 开始正文
    body_parts: List[str] = [HTML_HEAD]
    body_parts.append('<div class="query-nav">' + "\n".join(nav_links) + "</div>")
    body_parts.append('<hr class="sep"/>')

    # 收集本页用到的对话 ID，用于生成隐藏模板
    used_dids: List[str] = []

    # 渲染每个 query
    for gi, g in enumerate(groups, start=1):
        qtext = str(g.get("query_text", "") or f"Query #{gi}")
        anchor = f"q{gi}"
        results_raw = g.get("results", []) or []
        results = dedup_results(results_raw, did2dlg, map_url, mode=args.dedup_mode) if args.dedup else results_raw

        # limit 在去重之后
        if args.limit_dialogues and args.limit_dialogues > 0:
            results = results[: args.limit_dialogues]

        body_parts.append(f'<h2 id="{anchor}">Query {gi}: {html.escape(qtext)}</h2>')
        if len(results) != len(results_raw):
            body_parts.append(f'<div class="meta">命中对话数：{len(results)} / 原始 {len(results_raw)}（已去重：{args.dedup_mode}）</div>')
        else:
            body_parts.append(f'<div class="meta">命中对话数：{len(results)}</div>')

        if not results:
            body_parts.append('<div class="footer-note">（该 query 无结果）</div>')
            body_parts.append('<hr class="sep"/>')
            continue

        for r in results:
            did = str(r.get("dialogue_id", ""))
            frag_cnt = int(r.get("fragment_count", 0) or 0)
            utt_ids = list(map(int, (r.get("merged", {}) or {}).get("utt_ids", []) or []))
            covered = r.get("covered_fragments", []) or []

            # 记录用于生成模板
            used_dids.append(did)

            # link target id
            tpl_id = "tpl-" + safe_id(did)

            body_parts.append('<div class="card">')
            body_parts.append(
                f'<div class="meta"><b>Dialogue:</b> '
                f'<a href="#" class="dlg-link" data-target="{html.escape(tpl_id)}" data-did="{html.escape(did)}">'
                f'<code>{html.escape(did)}</code></a>'
                f' <span class="badge">fragments: {frag_cnt}</span>'
                f' <span class="badge">covered: {",".join(map(str, covered)) if covered else "-"}</span>'
                f'</div>'
            )

            dlg = did2dlg.get(did)
            if not dlg:
                body_parts.append('<div class="footer-note">原始对话未找到，可能语料不一致或 ID 清洗过。</div>')
                body_parts.append('</div>')
                continue

            hit_set: Set[int] = set(utt_ids)
            block_html = render_dialogue_block(
                dlg, hit_set, ctx=args.context,
                url_mapper=map_url, use_file_url=args.file_url_images
            )
            if not block_html:
                body_parts.append('<div class="footer-note">没有可展示的命中区段。</div>')
            else:
                body_parts.append(block_html)

            if utt_ids:
                ids_str = ", ".join(map(str, utt_ids))
                body_parts.append(f'<div class="footer-note">命中 ID（{len(utt_ids)}）: {ids_str}</div>')

            body_parts.append('</div>')  # .card

        body_parts.append('<hr class="sep"/>')

    # 生成隐藏的“完整对话”模板（仅对用到的 did）
    body_parts.append('<div id="dlg-templates" style="display:none">')
    seen_tpl = set()
    for did in used_dids:
        if did in seen_tpl:
            continue
        seen_tpl.add(did)
        dlg = did2dlg.get(did)
        if not dlg:
            continue
        tpl_id = "tpl-" + safe_id(did)
        body_parts.append(f'<div class="dlg-template" id="{html.escape(tpl_id)}" data-did="{html.escape(did)}">')
        body_parts.append(f'<h3 style="margin:6px 0 10px">{html.escape(did)}</h3>')
        body_parts.append(render_full_dialogue(dlg, url_mapper=map_url, use_file_url=args.file_url_images))
        body_parts.append('</div>')
    body_parts.append('</div>')  # #dlg-templates

    body_parts.append(HTML_TAIL)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"[OK] Wrote HTML -> {out_path}")

if __name__ == "__main__":
    main()