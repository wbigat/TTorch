"""XSwift 退役的完整模型验证：功能 + 精度 + 性能。

三个 XSwift 状态（同一份生产脚本副本、同一模型、同一数据、同一 seed）：

  plugin_on    XSWIFT_PLUGIN_ENABLE=1（= 生产当前配置，基线）
  plugin_off   XSWIFT_PLUGIN_ENABLE=0（插件关闭，包仍在环境里）
  pkg_removed  物理移走 xswift_ext 包与 xswift_ext_auto_init.pth（真·退役）

精度口径（按既有做事基准）：先看 loss 差的**符号分布**判定系统性 bias vs 规约
噪声，再看幅度；同时对 HF 导出张量做逐位比对——它比 loss 更严格，因为 loss 只有
4 位有效数字的打印地板。

性能口径：**边际 s/it**，由 swift 的累计平均 `train_speed(s/it)` 反解
（`marginal[n] = n*avg[n] - (n-1)*avg[n-1]`），不用累计平均本身（首步编译会被摊薄），
去 warmup 后取中位数。

用法：
    python validate_retirement.py --model dense --states plugin_on plugin_off pkg_removed
    python validate_retirement.py --model moe --large-weight 1 --states plugin_on plugin_off

`--tt`：在三个状态里都激活 TT（经 `probe_bootstrap/sitecustomize.py`，等价于
安装 `ttorch_auto_init.pth`）。这是**退役终局形态**的验证口径——XSwift 的功能
（B1 THD cp_group / B5 / B7 gather_object）由 TT 的功能补丁承接，所以"能不能
退役"必须在 TT 在场的前提下回答；不带 `--tt` 的运行只能证明"包可以移走"。
"""
import argparse
import contextlib
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
SITE = Path(
    "/root/miniconda/envs/python312_torch29_cuda/lib/python3.12/site-packages")
XSWIFT_PKG = SITE / "xswift_ext"
XSWIFT_PTH = SITE / "xswift_ext_auto_init.pth"
RETIRED_SUFFIX = ".__tt_retired__"
PROBE_BOOTSTRAP = Path(__file__).resolve().parent / "probe_bootstrap"

STATES = ("plugin_on", "plugin_off", "pkg_removed")


# --------------------------------------------------------------------------
# 包的临时移除与恢复
# --------------------------------------------------------------------------

@contextlib.contextmanager
def xswift_package_removed(enabled):
    """把 xswift_ext 包与其 .pth 临时移走；退出时必须恢复（含异常路径）。

    用 rename 而不是 delete：可逆，且能在恢复后用 md5 清单核对未损坏。
    """
    if not enabled:
        yield {"removed": False}
        return
    moves = []
    for path in (XSWIFT_PKG, XSWIFT_PTH):
        if path.exists():
            moves.append((path, Path(str(path) + RETIRED_SUFFIX)))
    for src, dst in moves:
        if dst.exists():
            raise RuntimeError(f"残留的上次退役备份：{dst}（请先人工确认并恢复）")
    try:
        for src, dst in moves:
            src.rename(dst)
        yield {"removed": True, "moved": [str(s) for s, _ in moves]}
    finally:
        failed = []
        for src, dst in moves:
            try:
                if dst.exists():
                    dst.rename(src)
            except Exception as error:
                failed.append(f"{dst} -> {src}: {error}")
        for src, _ in moves:
            if not src.exists():
                failed.append(f"恢复后仍缺失: {src}")
        if failed:
            raise RuntimeError("XSwift 包恢复失败，环境已被破坏，必须人工介入：\n"
                               + "\n".join(failed))


def package_fingerprint():
    """包目录的文件数与总字节，用于验证恢复后未损坏。"""
    if not XSWIFT_PKG.exists():
        return {"present": False}
    files = sorted(p for p in XSWIFT_PKG.rglob("*.py"))
    return {"present": True, "py_files": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
            "pth_present": XSWIFT_PTH.is_file()}


# --------------------------------------------------------------------------
# 资产
# --------------------------------------------------------------------------

def ensure_moe_assets(root, tokenizer):
    sys.path.insert(0, str(REPO / "tests" / "e2e"))
    from big_leaf_assets import ensure_assets
    return ensure_assets(root, tokenizer_source=tokenizer)


MODEL_PRESETS = {
    # 生产默认形态：真实 Qwen3.5-0.8B（dense）+ LoRA。
    # SEQ_LEN 必须用生产原值 20480：该数据集是多模态 caption，把 max_length 调小会
    # 在图像 token 块中间截断，`get_rope_index` 的 MRoPE position_ids 与 grid 不再
    # 一致并抛 shape mismatch（已实测，与 XSwift 无关的既有缺陷）。
    "dense": {
        "TEMPLATE": "qwen3_5",
        "TUNER_TYPE": "lora",
        "SEQ_LEN": "20480",
        "BATCH_SIZE": "1",
        "GLOBAL_BATCH_SIZE": "4",
        "ENABLE_CHANNEL_LOSS": "true",
        "ADD_NON_THINKING_PREFIX": "true",
    },
    # 合成 BF16 Qwen3-MoE：唯一能在本机走到 GroupedLinear / large-weight 的形态
    "moe": {
        "TEMPLATE": "qwen3",
        "TUNER_TYPE": "full",
        "SEQ_LEN": "512",
        "BATCH_SIZE": "1",
        "GLOBAL_BATCH_SIZE": "4",
        "ENABLE_CHANNEL_LOSS": "false",
        "ADD_NON_THINKING_PREFIX": "false",
        "LOSS_SCALE": "default",
        "RECOMPUTE_GRANULARITY": "full",
    },
}


def build_env(args, run_dir, state):
    env = os.environ.copy()
    env.pop("XSWIFT_PLUGIN_ENABLE", None)
    env.pop("XME_PLUGIN_ENABLE", None)
    nproc = str(args.nproc)
    env.update({
        "NODE_IP_LIST": f"{env.get('LOCAL_IP', '127.0.0.1')}:{nproc}",
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(args.nproc)),
        # 三个被测开关
        "XSWIFT_PLUGIN_ENABLE": "0" if state != "plugin_on" else "1",
        "XME_PLUGIN_ENABLE": str(args.xme),
        "XTE_GROUPED_GEMM_LARGE_WEIGHT": str(args.large_weight),
        # 产物隔离：**必须用 TTV_ 前缀**。`run.sh` 里的 `source ~/.bashrc` 会重新
        # export 全部 HUNYUAN_*，直接设 HUNYUAN_CKPT_PATH 会被顶掉，训练会写进
        # 生产 ckpt 目录（已实测踩过）。脚本副本在 bashrc 之后重新应用 TTV_*。
        "TTV_HUNYUAN_CKPT_PATH": str(run_dir / "output"),
        "TTV_HUNYUAN_LOG_PATH": str(run_dir / "logs"),
        "TTV_HUNYUAN_TF_EVENTS_PATH": str(run_dir / "events"),
        "HUNYUAN_CKPT_PATH": str(run_dir / "output"),
        "HUNYUAN_LOG_PATH": str(run_dir / "logs"),
        "HUNYUAN_TF_EVENTS_PATH": str(run_dir / "events"),
        # 可复现：固定步数、关闭多进程 map（曾复现 fork-map 阻塞）
        "TRAIN_ITERS": str(args.steps),
        "SAVE_STEPS": str(args.steps),
        "EVAL_STEPS": str(args.eval_steps),
        "EVAL_ITERS": "2",
        "LOGGING_STEPS": "1",
        "SPLIT_DATASET_RATIO": "0.1",
        "DATALOADER_NUM_WORKERS": "0",
        "DATASET_NUM_PROC": "1",
        "SWANLAB_MODE": "offline",
        "TT_XSWIFT_PROBE": str(run_dir / "xswift_probe.jsonl"),
        "PYTHONPATH": f"{PROBE_BOOTSTRAP}{os.pathsep}{env.get('PYTHONPATH', '')}",
    })
    env.update(MODEL_PRESETS[args.model])
    if args.tt:
        # TT 未装 .pth，靠 probe_bootstrap 的 sitecustomize 在训练 worker 上激活；
        # 仓库根要进 PYTHONPATH 才 import 得到 ttorch。
        env["TT_PLUGIN_ENABLE"] = "1"
        env["PYTHONPATH"] = f"{PROBE_BOOTSTRAP}{os.pathsep}{REPO}" \
                            f"{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    else:
        env.pop("TT_PLUGIN_ENABLE", None)
    for item in (args.extra_env or []):
        key, _, value = item.partition("=")
        env[key] = value
    env["HUNYUAN_MODEL_PATH"] = str(args.model_path)
    env["TTV_HUNYUAN_MODEL_PATH"] = str(args.model_path)
    env["TTV_HUNYUAN_FINETUNING_MODE"] = (
        "FULL" if MODEL_PRESETS[args.model]["TUNER_TYPE"] == "full" else "LORA")
    if args.dataset:
        env["DATASET_PATH"] = str(args.dataset)
    # 不上报外网
    env.pop("SWANLAB_API_KEY", None)
    for key in ("http_proxy", "https_proxy", "ftp_proxy"):
        env.pop(key, None)
    return env


# --------------------------------------------------------------------------
# 单次运行
# --------------------------------------------------------------------------

def run_one(args, run_root, state):
    run_dir = run_root / state
    run_dir.mkdir(parents=True, exist_ok=False)
    env = build_env(args, run_dir, state)
    console = run_dir / "console.log"
    print(f"\n===== {state}: XSWIFT_PLUGIN_ENABLE="
          f"{env['XSWIFT_PLUGIN_ENABLE']} pkg_removed={state == 'pkg_removed'} "
          f"LW={env['XTE_GROUPED_GEMM_LARGE_WEIGHT']} =====", flush=True)
    with xswift_package_removed(state == "pkg_removed") as removal:
        with console.open("w") as log:
            process = subprocess.Popen(
                ["bash", "run.sh"], cwd=str(args.script_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1)
            for line in process.stdout:
                log.write(line)
            code = process.wait()
    text = console.read_text(errors="replace")
    # 硬断言：产物必须落在隔离目录。run.sh 的 bashrc 覆盖曾让训练写进生产 ckpt
    # 路径，这里不允许再静默发生。
    if code == 0 and not (run_dir / "output").is_dir():
        raise RuntimeError(
            f"{state}: 退出码 0 但 {run_dir / 'output'} 不存在——路径覆盖未生效，"
            "产物可能落到了生产目录；停止验证")
    result = {
        "state": state,
        "exit_code": code,
        "removal": removal,
        "xswift_banner_present": "XSwift Plugin Init" in text,
        "xswift_warning_present": "[XSwift][WARNING]" in text,
        "xme_banner_present": bool(re.search(r"XME|xmegatron_ext", text)),
        "console": str(console),
    }
    result["probe"] = summarize_probe(run_dir / "xswift_probe.jsonl")
    # 硬断言：开关状态必须与实际可观测行为一致。
    # 实测踩过：`~/.bashrc:129` 把 RUNTIME_SCRIPT_DIR 覆盖回生产目录，于是
    # `source "${RUNTIME_SCRIPT_DIR}/env.sh"` 读的是生产 env.sh（硬编码
    # XSWIFT_PLUGIN_ENABLE=1），外部传入的 0 被吃掉——"插件已关闭"的运行里
    # XSwift 仍完整初始化。没有这道断言会拿到一份看似通过的无效证据。
    expected_banner = state == "plugin_on"
    worker_env = result["probe"].get("worker_env_values")
    inconsistent = []
    if result["xswift_banner_present"] != expected_banner:
        inconsistent.append(
            f"banner={result['xswift_banner_present']} 但期望 {expected_banner}")
    if state != "plugin_on" and worker_env and worker_env != ["0"]:
        inconsistent.append(f"worker 的 XSWIFT_PLUGIN_ENABLE={worker_env}，期望全为 0")
    if state == "pkg_removed" and result["probe"].get("any_importable"):
        inconsistent.append("pkg_removed 下仍有进程可 import xswift_ext")
    inconsistent.extend(tt_inconsistencies(args, state, result["probe"], code))
    if inconsistent:
        raise RuntimeError(f"{state}: 开关未真正生效，证据无效 —— " + "；".join(inconsistent))
    if code != 0:
        result["tail"] = "".join(text.splitlines(keepends=True)[-40:])
        result["failure_signature"] = failure_signature(text)
        return result, run_dir
    result.update(parse_metrics(run_dir / "output", args.steps))
    result["hf_fingerprint"] = hf_fingerprint(run_dir / "output", args.steps)
    return result, run_dir


def tt_inconsistencies(args, state, probe, exit_code):
    """`--tt` 下的证据有效性断言：TT 必须真的在场且补丁必须真的生效。

    不加这些断言，一次"TT 其实没激活"的运行会给出一份看起来通过的无效证据
    —— 和之前 bashrc 顶掉 `XSWIFT_PLUGIN_ENABLE=0` 是同一类坑。
    """
    if not args.tt:
        return []
    issues = []
    if not probe.get("tt_mode"):
        return [f"{state}: --tt 已指定但探针里没有任何 TT 记录"]
    if probe.get("tt_worker_activated") != [True]:
        issues.append(f"worker 的 tt_activated={probe.get('tt_worker_activated')}，"
                      f"期望全为 True；错误={probe.get('tt_worker_errors')}")
    if exit_code != 0:
        return issues      # 训练本身失败时不追究终局证据（atexit 可能没跑完）
    if not probe.get("tt_final_records"):
        issues.append("缺少 tt_final 记录（atexit 取证未落盘）")
    if probe.get("tt_final_errors"):
        issues.append(f"tt_final 取证报错: {probe['tt_final_errors']}")
    # 退役态必须由 TT 独立承接：补丁被顶掉就说明缺口仍靠厂商插件挡着
    if state != "plugin_on" and probe.get("tt_overwritten"):
        issues.append(f"插件已关闭/移除，但 TT 补丁仍被覆盖: {probe['tt_overwritten']}")
    if probe.get("tt_export_issues"):
        issues.append(f"导出安全契约有未消除问题: {probe['tt_export_issues']}")
    if probe.get("tt_dummy_traces"):
        issues.append(f"发现 /klx_dummy_lib 假模块残留: {probe['tt_dummy_traces']}")
    return issues


def failure_signature(text):
    """从日志里抽出最具体的失败原因（用于验证「shim 承重」的预测）。"""
    patterns = [
        r"AttributeError: '?\w+'? object has no attribute '(\w+)'",
        r"AttributeError: (.+)",
        r"ValueError: (.+)",
        r"RuntimeError: (.+)",
        r"KeyError: (.+)",
    ]
    hits = []
    for pattern in patterns:
        hits.extend(re.findall(pattern, text))
    seen, ordered = set(), []
    for hit in hits:
        key = str(hit)[:160]
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered[:6]


def summarize_probe(path):
    if not path.is_file():
        return {"records": 0}
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    finals = [r for r in rows if r.get("kind") == "tt_final"]
    rows = [r for r in rows if r.get("kind") != "tt_final"]
    # 只看训练 worker（argv0=sft.py 且带 RANK）——swanlab 会顺带起 pip 子进程，
    # 它也带 RANK，但不是训练进程，不能混进判据。
    workers = [r for r in rows
               if r.get("rank") is not None and r.get("argv0") == "sft.py"]
    summary = {
        "records": len(rows),
        "worker_records": len(workers),
        "worker_ranks": sorted({r["rank"] for r in workers}),
        "worker_env_values": sorted({r["env_XSWIFT_PLUGIN_ENABLE"] for r in workers}),
        "worker_importable": sorted({r["xswift_importable"] for r in workers}),
        "worker_import_hooked": sorted({r["import_hooked_by_xswift"] for r in workers}),
        "any_importable": any(r["xswift_importable"] for r in rows),
        "all_importable": all(r["xswift_importable"] for r in rows) if rows else None,
        "any_import_hooked": any(r["import_hooked_by_xswift"] for r in rows),
        "any_in_sys_modules": any(r["xswift_in_sys_modules_at_start"] for r in rows),
        "any_dummy_lib": any(r["dummy_lib_modules"] for r in rows),
    }
    summary.update(summarize_tt(workers, finals))
    return summary


def summarize_tt(workers, finals):
    """TT 侧证据：启动时是否激活 + 退出时补丁/契约是否干净。

    `tt_final` 由探针在 `atexit` 写入——补丁状态只有在框架 import 完成之后
    看才算数（厂商插件走 lazy hook，启动阶段看全是 pending）。
    """
    if not any(w.get("tt_requested") for w in workers) and not finals:
        return {"tt_mode": False}
    return {
        "tt_mode": True,
        "tt_worker_activated": sorted({w.get("tt_activated") for w in workers}),
        "tt_worker_errors": sorted({w["tt_error"] for w in workers
                                    if w.get("tt_error")}),
        "tt_final_records": len(finals),
        "tt_final_active": sorted({f.get("tt_active") for f in finals}),
        # 退役终局的核心断言面：补丁没被顶掉、契约无问题、无假模块残留
        "tt_patch_status": sorted({json.dumps(f.get("patch_status"), sort_keys=True,
                                              ensure_ascii=False) for f in finals}),
        "tt_overwritten": [f["overwritten_patches"] for f in finals
                           if f.get("overwritten_patches")],
        "tt_export_issues": [f["retirement_report"]["export_safety_issues"]
                             for f in finals
                             if f.get("retirement_report", {}).get(
                                 "export_safety_issues")],
        "tt_dummy_traces": [f["retirement_report"]["dummy_module_traces"]
                            for f in finals
                            if f.get("retirement_report", {}).get(
                                "dummy_module_traces")],
        "tt_bindings_tampered": sorted({f.get("bindings_tampered") for f in finals}),
        "tt_final_errors": sorted({f["error"] for f in finals if f.get("error")}),
    }


def resolve_run_output(output_dir):
    """swift 会在 output_dir 下再建 `v{N}-{时间戳}` 版本目录（ADD_VERSION）。

    train_megatron.sh 没有暴露 add_version 参数，所以这里按时间取最新的版本
    目录；没有版本目录时回退到 output_dir 本身。
    """
    versions = sorted((p for p in output_dir.glob("v*-*") if p.is_dir()),
                      key=lambda p: p.stat().st_mtime)
    return versions[-1] if versions else output_dir


def parse_metrics(output_dir, steps):
    output_dir = resolve_run_output(output_dir)
    path = output_dir / "logging.jsonl"
    if not path.is_file():
        return {"metrics_error": f"缺少 {path}"}
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    train, evals = [], []
    for row in rows:
        if "loss" in row and "iteration" in row:
            train.append(row)
        if "eval_loss" in row:
            evals.append(row)

    def iteration(row):
        return int(str(row["iteration"]).split("/", 1)[0])

    train.sort(key=iteration)
    losses = {iteration(r): float(r["loss"]) for r in train}
    # 累计平均 -> 边际 s/it（exact：marginal[n] = n*avg[n] - (n-1)*avg[n-1]）
    cumulative = {iteration(r): float(r["train_speed(s/it)"]) for r in train
                  if "train_speed(s/it)" in r}
    marginal = {}
    for index in sorted(cumulative):
        previous = index - 1
        if previous in cumulative:
            marginal[index] = (index * cumulative[index]
                               - previous * cumulative[previous])
        elif index == 1:
            marginal[index] = cumulative[index]
    return {
        "steps_logged": len(train),
        "steps_expected": steps,
        "losses": losses,
        "grad_norms": {iteration(r): float(r["grad_norm"]) for r in train
                       if "grad_norm" in r},
        "eval_losses": [float(r["eval_loss"]) for r in evals],
        "memory_gib": max((float(r["memory(GiB)"]) for r in train
                           if "memory(GiB)" in r), default=None),
        "marginal_s_per_it": marginal,
        "cumulative_s_per_it_last": cumulative.get(max(cumulative)) if cumulative else None,
        "all_finite": all(math.isfinite(v) for v in losses.values()),
    }


def hf_fingerprint(output_dir, steps):
    """HF 导出张量的逐位指纹（比 loss 更严格的精度证据）。"""
    output_dir = resolve_run_output(output_dir)
    checkpoint = output_dir / f"checkpoint-{steps}"
    files = sorted(checkpoint.rglob("*.safetensors"))
    if not files:
        return {"present": False}
    import hashlib

    import torch
    from safetensors import safe_open
    digests = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                tensor = reader.get_tensor(key)
                digests[key] = hashlib.sha256(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
    return {"present": True, "tensor_count": len(digests), "digests": digests,
            "checkpoint": str(checkpoint)}


# --------------------------------------------------------------------------
# 跨状态对比
# --------------------------------------------------------------------------

def compare_accuracy(base, other, label):
    if "losses" not in base or "losses" not in other:
        return {"skipped": "一方无 metrics"}
    shared = sorted(set(base["losses"]) & set(other["losses"]))
    deltas = [other["losses"][i] - base["losses"][i] for i in shared]
    if not deltas:
        return {"skipped": "无共同步"}
    positive = sum(d > 0 for d in deltas)
    negative = sum(d < 0 for d in deltas)
    zero = sum(d == 0 for d in deltas)
    mean_delta = sum(deltas) / len(deltas)
    mean_abs = sum(abs(d) for d in deltas) / len(deltas)
    max_abs = max(abs(d) for d in deltas)
    base_scale = max(abs(v) for v in base["losses"].values()) or 1.0
    # 判据：先符号后幅度。同号 = 系统性 bias；零均值 = 规约顺序噪声
    if zero == len(deltas):
        verdict = "bitwise_identical"
    elif positive == 0 or negative == 0:
        verdict = "systematic_bias"
    elif abs(mean_delta) < mean_abs / 4:
        verdict = "zero_mean_noise"
    else:
        verdict = "mixed"
    out = {
        "label": label, "steps": len(deltas),
        "positive": positive, "negative": negative, "zero": zero,
        "mean_delta": mean_delta, "mean_abs_delta": mean_abs,
        "max_abs_delta": max_abs, "max_rel_delta": max_abs / base_scale,
        "verdict": verdict,
        "first_step_equal": (shared[0] in base["losses"]
                             and base["losses"][shared[0]] == other["losses"][shared[0]]),
        "eval_losses_equal": base.get("eval_losses") == other.get("eval_losses"),
    }
    base_hf = base.get("hf_fingerprint", {})
    other_hf = other.get("hf_fingerprint", {})
    if base_hf.get("present") and other_hf.get("present"):
        keys_base = set(base_hf["digests"])
        keys_other = set(other_hf["digests"])
        mismatched = [k for k in sorted(keys_base & keys_other)
                      if base_hf["digests"][k] != other_hf["digests"][k]]
        out["hf"] = {
            "tensor_count": len(keys_base),
            "key_sets_equal": keys_base == keys_other,
            "bitwise_identical": not mismatched and keys_base == keys_other,
            "mismatched_count": len(mismatched),
            "mismatched_sample": mismatched[:10],
        }
    return out


def compare_performance(base, other, label, warmup):
    if "marginal_s_per_it" not in base or "marginal_s_per_it" not in other:
        return {"skipped": "一方无 metrics"}

    def series(result):
        return [v for k, v in sorted(result["marginal_s_per_it"].items())
                if int(k) > warmup]

    a, b = series(base), series(other)
    if not a or not b:
        return {"skipped": "warmup 后无可统计步"}

    def stats(values):
        ordered = sorted(values)
        return {
            "samples": len(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "p90": ordered[max(0, math.ceil(len(ordered) * 0.9) - 1)],
            "min": ordered[0], "max": ordered[-1],
        }

    sa, sb = stats(a), stats(b)
    return {
        "label": label, "warmup_excluded": warmup,
        "base": sa, "other": sb,
        "median_ratio_base_over_other": sa["median"] / sb["median"],
        "median_delta_ms": (sb["median"] - sa["median"]) * 1000,
        "first_step_s": {
            "base": base["marginal_s_per_it"].get(1)
            or base["marginal_s_per_it"].get("1"),
            "other": other["marginal_s_per_it"].get(1)
            or other["marginal_s_per_it"].get("1"),
        },
        "memory_gib": {"base": base.get("memory_gib"),
                       "other": other.get("memory_gib")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), default="dense")
    parser.add_argument("--states", nargs="+", default=list(STATES),
                        choices=STATES)
    parser.add_argument("--script-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--xme", type=int, default=1)
    parser.add_argument("--large-weight", type=int, default=1)
    parser.add_argument("--extra-env", nargs="*", default=[],
                        help="额外注入训练环境的 KEY=VALUE（归因实验用）")
    parser.add_argument("--tt", action="store_true",
                        help="三个状态都激活 TT（退役终局口径：XSwift 的功能由 "
                             "TT 功能补丁承接）")
    parser.add_argument("--allow-failure", action="store_true",
                        help="某个状态失败也继续（用于验证 shim 承重的预测）")
    args = parser.parse_args()

    args.script_dir = Path(args.script_dir or
                           REPO / ".benchmarks/xswift_retire_20261118/script")
    if not (args.script_dir / "run.sh").is_file():
        parser.error(f"脚本目录无效: {args.script_dir}")

    asset_info = {"model": args.model}
    if args.model == "moe" and not args.model_path:
        root = REPO / ".benchmarks/xswift_retire_20261118/moe_assets"
        model_dir, dataset, info = ensure_moe_assets(root, args.tokenizer)
        args.model_path, args.dataset = model_dir, dataset
        asset_info.update(info)
    if not args.model_path:
        args.model_path = os.environ["HUNYUAN_MODEL_PATH"]

    # 必须绝对路径：run.sh 会 `cd` 到脚本目录，相对路径会被子进程按它的 cwd 解析，
    # 导致产物与探针落到脚本目录下（已实测踩过）。
    run_root = (Path(args.run_dir).resolve() if args.run_dir else (
        REPO / ".benchmarks/xswift_retire_20261118"
        / f"{args.model}_lw{args.large_weight}_"
          f"{datetime.datetime.now().strftime('%H%M%S')}").resolve())
    run_root.mkdir(parents=True, exist_ok=True)

    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "model_path": str(args.model_path),
        "dataset": str(args.dataset) if args.dataset else "run.json",
        "asset_info": asset_info,
        "config": {"steps": args.steps, "eval_steps": args.eval_steps,
                   "warmup": args.warmup, "nproc": args.nproc,
                   "xme": args.xme, "large_weight": args.large_weight,
                   "tt": bool(args.tt),
                   "extra_env": list(args.extra_env or []),
                   "script_dir": str(args.script_dir)},
        "package_fingerprint_before": package_fingerprint(),
        "runs": {}, "accuracy": {}, "performance": {},
    }
    try:
        for state in args.states:
            result, _ = run_one(args, run_root, state)
            report["runs"][state] = result
            status = "PASS" if result["exit_code"] == 0 else "FAIL"
            print(f"[{status}] {state}: exit={result['exit_code']} "
                  f"banner={result['xswift_banner_present']} "
                  f"probe_importable={result['probe'].get('any_importable')}"
                  + (f" tt={result['probe'].get('tt_final_active')}"
                     if args.tt else ""), flush=True)
            if result["exit_code"] != 0:
                print(f"  失败签名: {result.get('failure_signature')}", flush=True)
                if not args.allow_failure:
                    raise RuntimeError(f"{state} 运行失败；用 --allow-failure 继续")
        base_state = args.states[0]
        base = report["runs"][base_state]
        for state in args.states[1:]:
            other = report["runs"][state]
            if base.get("exit_code") == 0 and other.get("exit_code") == 0:
                label = f"{base_state} vs {state}"
                report["accuracy"][state] = compare_accuracy(base, other, label)
                report["performance"][state] = compare_performance(
                    base, other, label, args.warmup)
    finally:
        report["package_fingerprint_after"] = package_fingerprint()
        (run_root / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n报告: {run_root / 'report.json'}", flush=True)

    before = report["package_fingerprint_before"]
    after = report["package_fingerprint_after"]
    if before != after:
        raise RuntimeError(f"XSwift 包未完整恢复: {before} -> {after}")
    print("XSwift 包指纹已复原一致", flush=True)
    for state, acc in report["accuracy"].items():
        print(f"[accuracy] {state}: {acc.get('verdict')} "
              f"max_abs={acc.get('max_abs_delta')} "
              f"hf_bitwise={acc.get('hf', {}).get('bitwise_identical')}", flush=True)
    for state, perf in report["performance"].items():
        if "median_ratio_base_over_other" in perf:
            print(f"[perf] {state}: base/other="
                  f"{perf['median_ratio_base_over_other']:.4f}x "
                  f"median {perf['base']['median']:.4f}s -> "
                  f"{perf['other']['median']:.4f}s", flush=True)


if __name__ == "__main__":
    main()
