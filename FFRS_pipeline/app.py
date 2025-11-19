

import gradio as gr
import json
import os
import re
import sys
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# ========= 基础工具 =========

def _mk_jobdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="wechat_rag_job_"))
    (d / "outs").mkdir(exist_ok=True)
    return d

def _normalize_query(q: str) -> str:
    """
    输入可为：
      - 纯文本（单条）
      - 多行文本（多条）
      - 用逗号/中文逗号分隔
      - 原生 JSON 数组字符串
    统一转成 JSON 数组字符串传给 --query_text
    """
    s = (q or "").strip()
    if not s:
        raise ValueError("请输入 query")
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return json.dumps(arr, ensure_ascii=False)
        except Exception:
            pass
    parts = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        for seg in [p.strip() for p in line.replace("，", ",").split(",")]:
            if seg:
                parts.append(seg)
    if not parts:
        raise ValueError("query 解析为空")
    return json.dumps(parts, ensure_ascii=False)

def _normalize_base_url(base_url: str) -> str:
    """
    允许 host:port / http://host:port / http://host:port/v1，统一转 http://host:port/v1
    """
    s = (base_url or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s
    s = s.rstrip("/")
    if not s.endswith("/v1"):
        s = s + "/v1"
    return s

def _run(cmd: List[str], env=None, cwd=None) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, env=env, cwd=cwd,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return p.returncode, p.stdout

def _extract_step9_html_path(logs: str) -> str:
    """
    兼容两种打印：
      [OK] Step9 可视化完成 -> /path/xxx.html
      [OK] Wrote HTML -> /path/xxx.html
    """
    m = re.search(r"Step9\s*可视化完成\s*->\s*(.+)", logs)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Wrote\s+HTML\s*->\s*(.+)", logs)
    if m2:
        return m2.group(1).strip()
    return ""

def _extract_service_used(logs: str) -> str:
    """
    从日志中提取 Step2/Step7 使用的 base_url 与 model（若 run_pipeline 打印了）：
    例如：
      [INFO] Step2(OpenAI) base_url=http://127.0.0.1:8081/v1, model=SplitVLM
      [INFO] Step7(OpenAI) base_url=http://127.0.0.1:8080/v1, model=F2RVLM
    """
    lines = []
    p = re.compile(r"Step([27])\(OpenAI\)\s*base_url=(\S+),\s*model=([^\s,]+)")
    for m in p.finditer(logs):
        step, url, model = m.group(1), m.group(2), m.group(3)
        lines.append(f"Step{step} 使用服务: {url}  | 模型: {model}")
    return "\n".join(lines)

def _zip_job(job_dir: Path) -> Optional[str]:
    try:
        # 将 outs/ 与 run 日志（若有）打包
        base = job_dir / "artifact"
        zip_path = shutil.make_archive(str(base), "zip", root_dir=str(job_dir))
        return zip_path
    except Exception:
        return None

# ========= 核心：调用 run_pipeline（把远程服务与部署参数也透传） =========

def launch_pipeline(
    run_pipeline_path: str,
    query_text_raw: str,
    topk: int,

    # ======== Step7（F2R）远程服务/部署 ========
    f2r_vllm_server: str,
    served_model_name: str,
    openai_api_key: str,
    auto_deploy_step7: bool,
    deploy_host7: str,
    deploy_port7: int,

    # ======== Step2（Split）远程服务/部署 ========
    step2_vllm_server: str,
    step2_served_model_name: str,
    auto_deploy_step2: bool,
    deploy_host2: str,
    deploy_port2: int,

    # ======== 其它高级项（本地回退/环境） ========
    f2r_backend: str,
    step7_nproc: int,
    cuda_visible: str,
    prepare_corpus: bool,
):
    """
    - 不传 step9 任何参数；全部使用 run_pipeline.py 内置的 step9 配置
    - 为避免覆盖，仍把 step5-8 输出指向 job 目录
    - 统一规范化并透传 Step2/Step7 的远程服务与部署参数
    - 生成 ZIP 归档
    """
    try:
        query_text = _normalize_query(query_text_raw)
    except Exception as e:
        return gr.update(value=f"<pre>Query 解析失败：{e}</pre>"), None, f"Query 解析失败：{e}"

    run_pipeline = Path(run_pipeline_path).resolve()
    if not run_pipeline.is_file():
        err = f"未找到 run_pipeline.py: {run_pipeline}"
        return gr.update(value=f"<pre>{err}</pre>"), None, err

    # 规范化 base_url（允许为空）
    f2r_url_norm = _normalize_base_url(f2r_vllm_server) if f2r_vllm_server else ""
    step2_url_norm = _normalize_base_url(step2_vllm_server) if step2_vllm_server else ""

    job = _mk_jobdir()
    outs = job / "outs"

    # 将 5-8 的输出定向到 job 目录；step9 走 run_pipeline 默认
    step5_json   = outs / "5_retrieval.json"
    step5_jsonl  = outs / "5_retrieval.jsonl"
    step6_json   = outs / "6_f2rvlm_fragment.json"
    step7_jsonl  = outs / "7_f2r_infer.jsonl"
    step7p_json  = outs / "7p_fused.json"
    step8_json   = outs / "8_final_merged.json"

    # 组装命令
    cmd = [
        sys.executable, str(run_pipeline),
        "--query_text", query_text,
        "--topk", str(topk),

        # ===== Step7（服务优先，其次回退） =====
        "--served_model_name", served_model_name or "F2RVLM",
        "--openai_api_key", openai_api_key or "EMPTY",
        "--f2r_backend", f2r_backend,  # 回退本地用
        "--step7_nproc", str(step7_nproc),

        # ===== 输出重定向（避免覆盖全局文件） =====
        "--step5_output_json", str(step5_json),
        "--step5_output_jsonl", str(step5_jsonl),
        "--step6_output_dataset", str(step6_json),
        "--swift_f2r_result_path", str(step7_jsonl),
        "--step7p_output_fused", str(step7p_json),
        "--step8_output_final", str(step8_json),
    ]

    # Step7 服务：显式传入 base_url 则不必 auto_deploy
    if f2r_url_norm:
        cmd += ["--f2r_vllm_server", f2r_url_norm]
    elif auto_deploy_step7:
        cmd += ["--auto_deploy", "--deploy_host", deploy_host7 or "0.0.0.0", "--deploy_port", str(deploy_port7 or 8080)]

    # ===== Step2（服务优先，其次本地） =====
    if step2_served_model_name:
        cmd += ["--step2_served_model_name", step2_served_model_name]
    if step2_url_norm:
        cmd += ["--step2_vllm_server", step2_url_norm]
    elif auto_deploy_step2:
        cmd += ["--step2_auto_deploy", "--step2_deploy_host", deploy_host2 or "0.0.0.0", "--step2_deploy_port", str(deploy_port2 or 8081)]

    # 环境 GPU
    env = os.environ.copy()
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        cmd += ["--env_cuda_visible_devices", cuda_visible]

    if prepare_corpus:
        cmd.append("--prepare_corpus")

    t0 = time.time()
    rc, logs = _run(cmd, env=env)
    elapsed = time.time() - t0

    # —— 失败直接返回 —— 
    if rc != 0:
        html_err = f"<h3>Pipeline 失败（退出码 {rc}）</h3><pre>{logs}</pre>"
        return gr.update(value=html_err), None, logs

    # —— 解析 Step9 HTML 并内嵌 —— 
    html_path = _extract_step9_html_path(logs)
    if not html_path:
        msg = "Pipeline 完成，但日志中未找到 Step9 HTML 路径（检查 run_pipeline 的打印是否包含 'Step9 可视化完成' 或 'Wrote HTML'）。"
        return gr.update(value=f"<pre>{msg}</pre>"), None, logs

    html_file = Path(html_path)
    if not html_file.exists():
        msg = f"日志解析到的 HTML 路径不存在：{html_file}"
        return gr.update(value=f"<pre>{msg}</pre>"), None, logs

    try:
        html_text = html_file.read_text(encoding="utf-8")
    except Exception as e:
        html_text = f"<pre>读取 HTML 出错：{e}</pre>"

    # —— 附加运行摘要（总耗时 + 服务信息）——
    svc_used = _extract_service_used(logs)
    summary = f"<!-- Job dir: {job} | total: {elapsed:.1f}s | html: {html_file} -->\n"
    if svc_used:
        summary += "<!-- " + svc_used.replace("\n", " | ") + " -->\n"

    html_embed = summary + html_text

    # —— 打包 ZIP —— 
    zip_path = _zip_job(job)
    file_to_offer = str(html_file) if not zip_path else zip_path

    # 把“运行摘要 + 首尾 2000 字日志”拼到日志输出顶部，方便查看
    head = logs[:2000]
    tail = logs[-2000:] if len(logs) > 4000 else ""
    pretty_logs = (
        f"[SUMMARY] total={elapsed:.1f}s\n" +
        (svc_used + "\n" if svc_used else "") +
        "----- LOG HEAD -----\n" + head +
        ("\n----- LOG TAIL -----\n" + tail if tail else "")
    )

    return gr.update(value=html_embed, visible=True), file_to_offer, pretty_logs


# ========= Gradio UI =========

INTRO = """
**使用说明（微信对话检索 & 可视化）**  
- 在下方输入一条或多条 *query*（支持换行、逗号或 JSON 数组）。  
- 点击 **运行** 后，会执行 Pipeline 的 Step5–Step9：检索、F2R 推理、后处理与可视化。  
- 页面仅展示命中片段；**点击 Dialogue ID** 会在**新窗口**打开该对话的完整内容；**点击图片**在**新窗口**查看原图（由可视化脚本控制）。  
- 去重方式、上下文行数、展示数量等策略由 `run_pipeline.py` / 可视化脚本内置。  
"""

def build_ui():
    with gr.Blocks(title="WeChat Retrieval & Viz", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 微信对话检索 & 可视化")
        gr.Markdown(INTRO)

        # —— 用户侧只保留 query + topk ——
        with gr.Row():
            query = gr.Textbox(
                label="输入 query（多条可用换行/逗号/JSON数组）",
                placeholder="例如：讨论数据集运行结果\n讨论上传数据集",
                lines=4
            )
        with gr.Row():
            topk = gr.Slider(1, 50, value=20, step=1, label="TopK")

        run_btn = gr.Button("运行 Pipeline 并可视化", variant="primary")

        # —— 输出区域 —— 
        with gr.Row():
            html_out = gr.HTML(label="可视化结果", visible=True)
        with gr.Row():
            file_out = gr.File(label="下载（HTML 或打包 ZIP）")

        with gr.Accordion("运行日志", open=False):
            logs_out = gr.Textbox(label="运行日志", lines=16)

        # —— 高级设置（默认收起） ——
        with gr.Accordion("高级设置", open=False):
            run_pipeline_path = gr.Textbox(
                label="run_pipeline.py 路径",
                value="data/Dialogcc/process_0825/pipeline_process/run_pipeline_deploy.py",
                lines=1
            )

            gr.Markdown("### Step7（F2R 推理服务）")
            with gr.Row():
                f2r_vllm_server = gr.Textbox(label="Step7 服务 base_url（可留空，例如 29.226.3.166:8000 或 http://29.226.3.166:8000/v1）", value="http://29.226.3.166:8000/v1/")
            with gr.Row():
                served_model_name = gr.Textbox(label="Step7 served_model_name", value="F2RVLM")
                openai_api_key = gr.Textbox(label="OpenAI API Key（如服务端需校验）", value="EMPTY", type="password")
            with gr.Row():
                auto_deploy_step7 = gr.Checkbox(value=False, label="若未提供 base_url，则自动部署 Step7 服务（swift deploy）")
                deploy_host7 = gr.Textbox(label="Step7 部署 host", value="0.0.0.0")
                deploy_port7 = gr.Number(label="Step7 部署端口", value=8080, precision=0)

            gr.Markdown("### Step2（长对话切分服务）")
            with gr.Row():
                step2_vllm_server = gr.Textbox(label="Step2 服务 base_url（可留空，例如 127.0.0.1:8081 或 http://127.0.0.1:8081/v1）", value="")
            with gr.Row():
                step2_served_model_name = gr.Textbox(label="Step2 served_model_name", value="SplitVLM")
            with gr.Row():
                auto_deploy_step2 = gr.Checkbox(value=False, label="若未提供 base_url，则自动部署 Step2 服务（swift deploy）")
                deploy_host2 = gr.Textbox(label="Step2 部署 host", value="0.0.0.0")
                deploy_port2 = gr.Number(label="Step2 部署端口", value=8081, precision=0)

            gr.Markdown("### 其它（本地回退/环境）")
            with gr.Row():
                f2r_backend = gr.Dropdown(["pt", "vllm"], value="pt", label="Step7 本地回退后端")
                step7_nproc = gr.Slider(1, 8, value=8, step=1, label="Step7 nproc（pt 回退有效）")
                cuda_visible = gr.Textbox(value="0,1,2,3,4,5,6,7", label="CUDA_VISIBLE_DEVICES", lines=1)
            prepare_corpus = gr.Checkbox(value=False, label="首次运行需要准备语料（Step1-4）")

        # 绑定事件
        run_btn.click(
            fn=launch_pipeline,
            inputs=[
                run_pipeline_path,
                query,
                topk,

                # Step7
                f2r_vllm_server,
                served_model_name,
                openai_api_key,
                auto_deploy_step7,
                deploy_host7,
                deploy_port7,

                # Step2
                step2_vllm_server,
                step2_served_model_name,
                auto_deploy_step2,
                deploy_host2,
                deploy_port2,

                # 其它
                f2r_backend,
                step7_nproc,
                cuda_visible,
                prepare_corpus,
            ],
            outputs=[html_out, file_out, logs_out],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()

    ui.launch(server_name="XXX.XXX.XXX.XXX", server_port=8081, share=True)