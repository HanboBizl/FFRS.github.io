

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# ======（可选）用于存活探测：OpenAI 兼容探活 ======
try:
    from openai import OpenAI  # pip install openai>=1.40
except Exception:
    OpenAI = None


# ====================== helpers ======================

def run(cmd: List[str], env: Optional[dict] = None, cwd: Optional[str] = None):
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, env=env, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}")

def popen_background(cmd: List[str], log_file: Path, env: Optional[dict] = None, cwd: Optional[str] = None):
    print("[BG]", " ".join(shlex.quote(x) for x in cmd), ">>", str(log_file))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "ab", buffering=0)
    return subprocess.Popen(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)

def find_latest_jsonl(root: str) -> str:
    root_p = Path(root)
    if root_p.is_file() and root_p.suffix == ".jsonl":
        return str(root_p)
    candidates = list(root_p.rglob("*.jsonl"))
    if not candidates:
        raise SystemExit(f"No .jsonl found under: {root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] latest jsonl: {latest}")
    return str(latest)

def resolve_result_jsonl(path_or_dir: str) -> str:
    p = Path(path_or_dir)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        print(f"[INFO] using result file: {p}")
        return str(p)
    return find_latest_jsonl(path_or_dir)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def summary_path_for(output_path: str) -> str:
    p = Path(output_path)
    return str(p.with_name(p.stem + "_frag_summary.json"))

# -------- OpenAI/vLLM 探测 & URL 工具 --------

def _normalize_base_url(base_url: str) -> str:
    s = (base_url or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s
    s = s.rstrip("/")
    if not s.endswith("/v1"):
        s = s + "/v1"
    return s

def _server_base_from_host_port(host: str, port: int) -> str:
    bad = {"0.0.0.0", "::", "", None}
    if host in bad:
        host = "127.0.0.1"
    return _normalize_base_url(f"http://{host}:{int(port)}")

def _is_server_alive(base_url: str, api_key: str = "EMPTY") -> bool:
    if OpenAI is None:
        return False
    try:
        cli = OpenAI(base_url=_normalize_base_url(base_url), api_key=api_key)
        _ = cli.models.list()
        return True
    except Exception:
        return False


# ====================== Step0：部署（可选） ======================

def _deploy_one_service(*, model_path: str, host: str, port: int, tp: int, gpu_mem_util: float,
                        max_new_tokens: int, limit_mm_per_prompt: str, served_model_name: str,
                        wait_s: float, log_path: str, cuda_visible: str, max_pixels: int,
                        api_key: str) -> str:
    base_url = _server_base_from_host_port(host, port)
    if _is_server_alive(base_url, api_key):
        print(f"[INFO] Found existing server at {base_url}, skip deploy.")
        return base_url

    env = os.environ.copy()
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if max_pixels:
        env["MAX_PIXELS"] = str(max_pixels)

    log_file = Path(log_path)
    cmd = [
        "swift", "deploy",
        "--model", model_path,
        "--infer_backend", "vllm",
        "--tensor_parallel_size", str(tp),
        "--gpu_memory_utilization", str(gpu_mem_util),
        "--max_new_tokens", str(max_new_tokens),
        "--limit_mm_per_prompt", limit_mm_per_prompt,
        "--served_model_name", served_model_name,
        "--host", host,
        "--port", str(port),
    ]
    _ = popen_background(cmd, log_file=log_file, env=env)

    print(f"[INFO] Waiting for server ready @ {host}:{port} ...")
    deadline = time.time() + float(wait_s)
    while time.time() < deadline:
        if _is_server_alive(base_url, api_key):
            print(f"[OK] Server is ready at {base_url}")
            return base_url
        time.sleep(1.0)
        print("[WAIT] vLLM service booting ...")
    print(f"[WARN] Server not ready within timeout ({wait_s}s): {base_url}")
    return base_url  # 仍返回，后续会再次探活

def step0_deploy_servers(args):
    # Step2 服务
    if args.step2_auto_deploy and not args.step2_vllm_server:
        args.step2_vllm_server = _deploy_one_service(
            model_path=args.step2_deploy_model_path,
            host=args.step2_deploy_host,
            port=args.step2_deploy_port,
            tp=args.step2_deploy_tp,
            gpu_mem_util=args.step2_deploy_gpu_mem_util,
            max_new_tokens=args.step2_deploy_max_new_tokens,
            limit_mm_per_prompt=args.step2_deploy_limit_mm_per_prompt,
            served_model_name=args.step2_served_model_name,
            wait_s=args.step2_deploy_wait_s,
            log_path=args.step2_deploy_log,
            cuda_visible=args.env_cuda_visible_devices,
            max_pixels=args.env_max_pixels,
            api_key=args.openai_api_key,
        )

    # Step7 服务
    if args.auto_deploy and not args.f2r_vllm_server:
        args.f2r_vllm_server = _deploy_one_service(
            model_path=args.deploy_model_path,
            host=args.deploy_host,
            port=args.deploy_port,
            tp=args.deploy_tp,
            gpu_mem_util=args.deploy_gpu_mem_util,
            max_new_tokens=args.deploy_max_new_tokens,
            limit_mm_per_prompt=args.deploy_limit_mm_per_prompt,
            served_model_name=args.served_model_name,
            wait_s=args.deploy_wait_s,
            log_path=args.deploy_log,
            cuda_visible=args.env_cuda_visible_devices,
            max_pixels=args.env_max_pixels,
            api_key=args.openai_api_key,
        )


# ====================== pipeline steps ======================

def step1_convert_wechat(args):
    if not args.prepare_corpus:
        print("[SKIP] step1_convert_wechat (use --prepare_corpus to run).")
        return
    ensure_parent(args.step1_output)
    run([sys.executable, args.script_step1, "--input", args.step1_input, "--output", args.step1_output])

# ---------- Step2：支持“服务推理或本地推理” ----------
def step2_split_infer_via_external_or_local(args):
    if not args.prepare_corpus:
        print("[SKIP] step2_split_infer (use --prepare_corpus to run).")
        return

    # 优先：OpenAI 兼容服务（独立于 Step7 的服务）
    base_url = _normalize_base_url(
        args.step2_vllm_server or ( _server_base_from_host_port(args.step2_deploy_host, args.step2_deploy_port) if args.step2_auto_deploy else "" )
    )
    if base_url and _is_server_alive(base_url, args.openai_api_key):
        print(f"[INFO] Step2(OpenAI) base_url={base_url}, model={args.step2_served_model_name}")
        ensure_parent(args.swift_split_result_path)
        cmd = [
            sys.executable, args.script_step2_service,
            "--input", args.step1_output,                  # step1 输出（长对话）
            "--output", args.swift_split_result_path,      # 期望 JSONL 输出
            "--base_url", base_url,
            "--model", args.step2_served_model_name,
            "--api_key", args.openai_api_key,
            "--timeout", str(args.step2_openai_timeout_s),
            "--system_prompt", args.step2_system_prompt,
            "--max_pixels", str(args.env_max_pixels or 1_000_000),
        ]
        run(cmd)
        return
    elif base_url:
        print(f"[WARN] Step2 OpenAI 兼容服务不可达：{base_url}，将回退到本地 swift infer。")

    # 回退：本地 swift.cli.infer（原逻辑）
    env = os.environ.copy()
    if args.env_max_pixels: env["MAX_PIXELS"] = str(args.env_max_pixels)
    if args.env_master_port: env["MASTER_PORT"] = str(args.env_master_port)
    if args.env_cuda_visible_devices: env["CUDA_VISIBLE_DEVICES"] = args.env_cuda_visible_devices

    ensure_parent(args.swift_split_result_path)
    if int(args.step2_nproc) <= 1:
        cmd = [sys.executable, "-m", "swift.cli.infer",
               "--model", args.swift_split_model, "--infer_backend", "pt",
               "--val_dataset", args.step1_output, "--temperature", "0",
               "--max_new_tokens", str(args.split_max_new_tokens),
               "--max_batch_size", "1", "--result_path", args.swift_split_result_path]
    else:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={args.step2_nproc}",
               f"--master_port={args.env_master_port}", "--module", "swift.cli.infer",
               "--model", args.swift_split_model, "--infer_backend", "pt",
               "--val_dataset", args.step1_output, "--temperature", "0",
               "--max_new_tokens", str(args.split_max_new_tokens),
               "--max_batch_size", "1", "--result_path", args.swift_split_result_path]
    run(cmd, env=env)

def step3_split_by_prediction(args):
    if not args.prepare_corpus:
        print("[SKIP] step3_split_by_prediction (use --prepare_corpus to run).")
        return
    ensure_parent(args.step3_output_fragments)
    split_infer_jsonl = resolve_result_jsonl(args.swift_split_result_path)
    summary_out = summary_path_for(args.step3_output_fragments)
    cmd = [sys.executable, args.script_step3, "--input", split_infer_jsonl,
           "--output", args.step3_output_fragments, "--summary-output", summary_out]
    run(cmd)

def step4_fragment_embedding(args):
    if not args.prepare_corpus:
        print("[SKIP] step4_fragment_embedding (use --prepare_corpus to run).")
        return
    ensure_parent(args.step4_output_index)

    env = os.environ.copy()
    if args.env_cuda_visible_devices: env["CUDA_VISIBLE_DEVICES"] = args.env_cuda_visible_devices

    nproc = int(args.step4_nproc)
    if nproc <= 1:
        cmd = [sys.executable, args.script_step4,
               "--input_json", args.step3_output_fragments,
               "--out_dir", args.step4_output_index,
               "--model_path", args.embed_model]
    else:
        master_port = str(args.step4_master_port or args.env_master_port)
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", f"--master_port={master_port}",
               args.script_step4, "--input_json", args.step3_output_fragments,
               "--out_dir", args.step4_output_index, "--model_path", args.embed_model]
        env.setdefault("OMP_NUM_THREADS", "1")
    run(cmd, env=env)

def step5_query_retrieve(args):
    ensure_parent(args.step5_output_json)
    ensure_parent(args.step5_output_jsonl)
    if not args.query_text:
        raise SystemExit("Please pass --query_text.")
    cmd = [sys.executable, args.script_step5, "--query_text", args.query_text,
           "--corpus", args.step4_output_index, "--model", args.query_model or args.embed_model,
           "--topk", str(args.topk), "--output_jsonl", args.step5_output_jsonl,
           "--output_json", args.step5_output_json]
    run(cmd)

def step6_build_f2r_dataset(args):
    ensure_parent(args.step6_output_dataset)
    cmd = [sys.executable, args.script_step6, "--retrieval", args.step5_output_json,
           "--fragments", args.step3_output_fragments, "--output", args.step6_output_dataset,
           "--topn", str(args.topk)]
    if args.fill_gt_when_same_dialogue: cmd.append("--fill-gt-when-same-dialogue")
    run(cmd)

# ---------- Step7：服务推理优先，回退本地 ----------
def step7_f2r_infer_via_external_script(args):
    base_url = _normalize_base_url(
        args.f2r_vllm_server or ( _server_base_from_host_port(args.deploy_host, args.deploy_port) if args.auto_deploy else "" )
    )
    if base_url and _is_server_alive(base_url, args.openai_api_key):
        print(f"[INFO] Step7(OpenAI) base_url={base_url}, model={args.served_model_name}")
        ensure_parent(args.swift_f2r_result_path)
        cmd = [
            sys.executable, args.script_step7_service,
            "--input", args.step6_output_dataset,
            "--output", args.swift_f2r_result_path,
            "--base_url", base_url,
            "--model", args.served_model_name,
            "--api_key", args.openai_api_key,
            "--timeout", str(args.openai_timeout_s),
            "--system_prompt", args.f2r_system_prompt,
            "--max_images", str(args.openai_max_images),
            "--max_pixels", str(args.env_max_pixels or 1_000_000),
        ]
        run(cmd)
        return
    elif base_url:
        print(f"[WARN] Step7 OpenAI 兼容服务不可达：{base_url}，将回退到本地 swift infer。")

    env = os.environ.copy()
    if args.env_max_pixels: env["MAX_PIXELS"] = str(args.env_max_pixels)
    if args.env_master_port: env["MASTER_PORT"] = str(args.env_master_port)
    if args.env_cuda_visible_devices: env["CUDA_VISIBLE_DEVICES"] = args.env_cuda_visible_devices

    ensure_parent(args.swift_f2r_result_path)
    if args.f2r_backend == "vllm":
        cmd = ["swift", "infer", "--model", args.swift_f2r_model, "--infer_backend", "vllm",
               "--gpu_memory_utilization", str(args.vllm_gpu_mem_util),
               "--tensor_parallel_size", str(args.vllm_tp_size),
               "--limit_mm_per_prompt", '{"image": 32, "video": 2}',
               "--val_dataset", args.step6_output_dataset, "--temperature", "0",
               "--max_new_tokens", str(args.f2r_max_new_tokens),
               "--max_batch_size", str(args.f2r_max_batch_size),
               "--result_path", args.swift_f2r_result_path]
        run(cmd, env=env)
    else:
        base = ["--model", args.swift_f2r_model, "--infer_backend", "pt",
                "--val_dataset", args.step6_output_dataset, "--temperature", "0",
                "--max_new_tokens", str(args.f2r_max_new_tokens),
                "--max_batch_size", str(args.f2r_max_batch_size),
                "--result_path", args.swift_f2r_result_path]
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={max(1, int(args.step7_nproc))}",
               f"--master_port={args.env_master_port}", "--module", "swift.cli.infer", *base]
        run(cmd, env=env)

def step7p_f2r_post_process(args):
    ensure_parent(args.step7p_output_fused)
    infer_jsonl = resolve_result_jsonl(args.swift_f2r_result_path)
    cmd = [sys.executable, args.script_step7p, "--input", infer_jsonl, "--output", args.step7p_output_fused]
    run(cmd)

def step8_merge_local2global(args):
    ensure_parent(args.step8_output_final)
    frag_summary = summary_path_for(args.step3_output_fragments)
    cmd = [sys.executable, args.script_step8, "--infer", args.step7p_output_fused,
           "--fragmeta", frag_summary, "--output", args.step8_output_final]
    if args.diagnose: cmd.append("--diagnose")
    run(cmd)

def step9_visualize(args):
    ensure_parent(args.step9_out_html)
    cmd = [sys.executable, args.script_step9, "--final_json", args.step8_output_final,
           "--corpus_json", args.step1_input, "--out", args.step9_out_html,
           "--context", str(args.step9_context),
           "--limit_dialogues", str(args.step9_limit_dialogues),
           "--img_local_prefix", args.step9_img_local_prefix,
           "--img_web_prefix", args.step9_img_web_prefix]
    if args.step9_file_url_images: cmd.append("--file_url_images")
    if getattr(args, "step9_dedup", False): cmd.append("--dedup")
    run(cmd)
    print(f"[OK] Step9 可视化完成 -> {args.step9_out_html}")


# ====================== main ======================

def main():
    ap = argparse.ArgumentParser(
        description="One-click pipeline: corpus prep (Step2支持服务) + retrieval + F2RVLM(服务) + merge + viz"
    )

    # ==== modes ====
    ap.add_argument("--prepare_corpus", action="store_true",
                    help="Run Steps 1-4 to prepare corpus (usually offline once)")
    ap.add_argument("--diagnose", action="store_true", help="Pass --diagnose to step8 merge")

    # ==== Step2 service deploy ====
    ap.add_argument("--step2_auto_deploy", action="store_true",
                    help="启动时后台 swift deploy 长对话切分服务（Step2）")
    ap.add_argument("--step2_deploy_model_path",
                    default="")
    ap.add_argument("--step2_deploy_host", default="0.0.0.0")
    ap.add_argument("--step2_deploy_port", type=int, default=8081)
    ap.add_argument("--step2_deploy_tp", type=int, default=1)
    ap.add_argument("--step2_deploy_gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--step2_deploy_max_new_tokens", type=int, default=15536)
    ap.add_argument("--step2_deploy_limit_mm_per_prompt", default='{"image": 0, "video": 0}')
    ap.add_argument("--step2_served_model_name", default="SplitVLM")
    ap.add_argument("--step2_deploy_wait_s", type=float, default=180.0)
    ap.add_argument("--step2_deploy_log", default="./swift_deploy_step2.log")
    ap.add_argument("--step2_vllm_server", default=None,
                    help="已有 Step2 OpenAI 兼容服务 base url")
    ap.add_argument("--step2_openai_timeout_s", type=float, default=200.0)
    ap.add_argument("--step2_system_prompt", default="你是一个对微信长对话进行片段切分的助手。")

    # ==== Step7 service deploy ====
    ap.add_argument("--auto_deploy", action="store_true",
                    help="启动时后台 swift deploy F2R 服务（Step7）")
    ap.add_argument("--deploy_model_path",
                    default="")
    ap.add_argument("--deploy_host", default="0.0.0.0")
    ap.add_argument("--deploy_port", type=int, default=8080)
    ap.add_argument("--deploy_tp", type=int, default=4)
    ap.add_argument("--deploy_gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--deploy_max_new_tokens", type=int, default=15536)
    ap.add_argument("--deploy_limit_mm_per_prompt", default='{"image": 32, "video": 2}')
    ap.add_argument("--served_model_name", default="F2RVLM")
    ap.add_argument("--deploy_wait_s", type=float, default=180.0)
    ap.add_argument("--deploy_log", default="./swift_deploy_step7.log")
    ap.add_argument("--f2r_vllm_server", default=None,
                    help="已有 Step7 OpenAI 兼容服务 base url")
    ap.add_argument("--openai_api_key", default="EMPTY")
    ap.add_argument("--openai_max_images", type=int, default=32)
    ap.add_argument("--openai_timeout_s", type=float, default=200.0)
    ap.add_argument("--f2r_system_prompt", default="你是一个多模态信息检索助手。")

    # ==== paths: scripts ====
    ap.add_argument("--script_step1", default="")
    ap.add_argument("--script_step3", default="")
    ap.add_argument("--script_step4", default="")
    ap.add_argument("--script_step5", default="")
    ap.add_argument("--script_step6", default="")
    ap.add_argument("--script_step7p", default="")
    ap.add_argument("--script_step8", default="")
    ap.add_argument("--script_step9", default="",
                    help="可视化脚本 9_viz_pipeline_results.py 的路径")
    # 外部服务推理脚本（Step2/Step7）
    ap.add_argument("--script_step2_service", default="",
                    help="外部 Step2 脚本（OpenAI 兼容服务推理）的路径")
    ap.add_argument("--script_step7_service", default="",
                    help="外部 Step7 脚本（OpenAI 兼容服务推理）的路径")

    # ==== step1: wechat -> split ====
    ap.add_argument("--step1_input",
                    default="")
    ap.add_argument("--step1_output",
                    default="")

    # ==== step2: split infer ====
    ap.add_argument("--swift_split_model",
                    default="")
    ap.add_argument("--swift_split_result_path",
                    default="",
                    help="Step2 推理结果路径（文件或目录）")
    ap.add_argument("--split_max_new_tokens", type=int, default=15536)
    ap.add_argument("--step2_nproc", type=int, default=8)

    # ==== step3: split by prediction ====
    ap.add_argument("--step3_output_fragments",
                    default="")

    # ==== step4: fragment embedding ====
    ap.add_argument("--embed_model", default="")
    ap.add_argument("--step4_output_index",
                    default="")
    ap.add_argument("--step4_nproc", type=int, default=8)
    ap.add_argument("--step4_master_port", type=int, default=None)

    # ==== step5: query retrieve ====
    ap.add_argument("--query_text", default=None, help="Single query text (quick run).")
    ap.add_argument("--query_model", default="")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--step5_output_json",
                    default="")
    ap.add_argument("--step5_output_jsonl",
                    default="")

    # ==== step6: build F2R dataset ====
    ap.add_argument("--step6_output_dataset",
                    default="")
    ap.add_argument("--fill_gt_when_same_dialogue", action="store_true")

    # ==== step7: F2RVLM infer（外部脚本/或回退） ====
    ap.add_argument("--swift_f2r_model",
                    default="")
    ap.add_argument("--swift_f2r_result_path",
                    default="")
    ap.add_argument("--f2r_backend", choices=["vllm", "pt"], default="pt")
    ap.add_argument("--step7_nproc", type=int, default=8)
    ap.add_argument("--vllm_gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--vllm_tp_size", type=int, default=4)
    ap.add_argument("--f2r_max_new_tokens", type=int, default=15536)
    ap.add_argument("--f2r_max_batch_size", type=int, default=4)

    # ==== step7': post-process ====
    ap.add_argument("--step7p_output_fused",
                    default="")

    # ==== step8: merge ====
    ap.add_argument("--step8_output_final",
                    default="")

    # ==== step9: visualize ====
    ap.add_argument("--step9_out_html",
                    default="l")
    ap.add_argument("--step9_context", type=int, default=0)
    ap.add_argument("--step9_limit_dialogues", type=int, default=0)
    ap.add_argument("--step9_file_url_images", action="store_true")
    ap.add_argument("--step9_img_local_prefix",
                    default="Wechat/")
    ap.add_argument("--step9_img_web_prefix",
                    default="")
    ap.add_argument("--step9_dedup", default=True, action="store_true")

    # ==== env ====
    ap.add_argument("--env_max_pixels", type=int, default=602112)
    ap.add_argument("--env_master_port", type=int, default=33645)
    ap.add_argument("--env_cuda_visible_devices", default="0,1,2,3,4,5,6,7")

    args = ap.parse_args()

    # 部署（可选）
    step0_deploy_servers(args)

    # Offline（可选）
    if args.prepare_corpus:
        step1_convert_wechat(args)
        step2_split_infer_via_external_or_local(args)
        step3_split_by_prediction(args)
        step4_fragment_embedding(args)

    # Query-time
    step5_query_retrieve(args)
    step6_build_f2r_dataset(args)
    step7_f2r_infer_via_external_script(args)
    step7p_f2r_post_process(args)
    step8_merge_local2global(args)
    step9_visualize(args)
    print("[OK] Pipeline finished.")


if __name__ == "__main__":
    main()