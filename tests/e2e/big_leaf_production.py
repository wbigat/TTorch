"""通过真实生产 run.sh 验收 TTorch big_leaf 的 train/eval/save/load/resume。"""
import argparse
import dataclasses
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch
from safetensors import safe_open

from big_leaf_assets import ensure_assets


ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class RunResult:
    name: str
    output_dir: Path
    console_log: Path
    records: list


def _validate_production_script_contract(production_dir):
    required = {
        "run.sh": ("DATASET_PATH", "TT_BIG_LEAF"),
        "train_megatron.sh": (
            "TEMPLATE:-", "SAVE_SAFETENSORS:-", "ADD_NON_THINKING_PREFIX:-",
            "ENABLE_CHANNEL_LOSS:-", "TT_BIG_LEAF:-",
        ),
    }
    missing = []
    for filename, variables in required.items():
        content = (production_dir / filename).read_text()
        missing.extend(f"{filename}:{variable}" for variable in variables
                       if variable not in content)
    if missing:
        raise ValueError(
            "生产脚本缺少 big_leaf 验收所需的环境参数化伴随改动: "
            + ", ".join(missing))


def _base_env(args, run_dir):
    env = os.environ.copy()
    nproc = str(args.nproc)
    env.setdefault("NODE_IP_LIST", f"127.0.0.1:{nproc}")
    env.setdefault("LOCAL_IP", "127.0.0.1")
    env.setdefault("CHIEF_IP", env["LOCAL_IP"])
    bootstrap = Path(__file__).resolve().parent / "big_leaf_bootstrap"
    env.update({
        "PYTHONPATH": f"{bootstrap}{os.pathsep}{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}",
        "PYTHON_BIN": sys.executable,
        "TT_PLUGIN_ENABLE": "1",
        "XSWIFT_PLUGIN_ENABLE": "0",
        "XME_PLUGIN_ENABLE": "0",
        "XTE_GROUPED_GEMM_LARGE_WEIGHT": "0",
        "XTE_USE_GROUPED_FC_FUSION": "0",
        "XTE_DISABLE_FAST_BF16_CACHE": "1",
        "XMLIR_ENABLE_FAST_FC": "true",
        "TEMPLATE": "qwen3" if args.synthetic else env.get("TEMPLATE", "qwen3_5"),
        "ENABLE_CHANNEL_LOSS": "false" if args.synthetic else "true",
        "ADD_NON_THINKING_PREFIX": "false" if args.synthetic else "true",
        "HUNYUAN_MODEL_PATH": str(args.model),
        "DATASET_PATH": str(args.dataset),
        "TUNER_TYPE": "full",
        "GLOBAL_BATCH_SIZE": str(args.global_batch_size),
        "BATCH_SIZE": str(args.micro_batch_size),
        "TRAIN_ITERS": str(args.steps),
        "SAVE_STEPS": str(args.save_steps),
        "EVAL_STEPS": str(args.eval_steps),
        "EVAL_ITERS": "1",
        "SPLIT_DATASET_RATIO": "0.2",
        "NO_SAVE_OPTIM": "false",
        "NO_SAVE_RNG": "false",
        "NO_LOAD_OPTIM": "false",
        "NO_LOAD_RNG": "false",
        "DIST_CKPT_OPTIM_FULLY_RESHARDABLE": "true",
        "ASYNC_SAVE": "false",
        "ADD_VERSION": "false",
        "DATALOADER_NUM_WORKERS": "0",
        "DATASET_NUM_PROC": "1",
        "SWANLAB_MODE": "offline",
        "HUNYUAN_CKPT_PATH": str(run_dir / "output"),
        "HUNYUAN_LOG_PATH": str(run_dir / "logs"),
        "HUNYUAN_TF_EVENTS_PATH": str(run_dir / "events"),
    })
    if args.seq_len is not None:
        env["SEQ_LEN"] = str(args.seq_len)
    elif args.synthetic:
        env["SEQ_LEN"] = "512"
    env.pop("MCORE_MODEL", None)
    env.pop("MCORE_ADAPTER", None)
    return env


def _read_records(output_dir):
    path = output_dir / "logging.jsonl"
    if not path.is_file():
        raise RuntimeError(f"缺少训练指标文件: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(args, run_root, name, big_leaf, resume_from=None, benchmark=False):
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    env = _base_env(args, run_dir)
    env["TT_BIG_LEAF"] = "1" if big_leaf else "0"
    if big_leaf:
        # big 路径覆盖总开关的安全默认值；stock 通过显式 0 安装同一生命周期。
        env.pop("XTE_DISABLE_FAST_BF16_CACHE", None)
        env.pop("XMLIR_ENABLE_FAST_FC", None)
    if benchmark:
        env["TT_BIG_LEAF_STEP_TRACE"] = str(run_dir / "steps.jsonl")
        env["NO_SAVE_OPTIM"] = "true"
        env["NO_SAVE_RNG"] = "true"
        env["SAVE_SAFETENSORS"] = "false"
    if resume_from is None:
        env["FINETUNE"] = "true"
    else:
        env["FINETUNE"] = "false"
        env["MCORE_MODEL"] = str(resume_from)
    console_log = run_dir / "console.log"
    command = ["bash", "run.sh"]
    print(f"\n===== {name}: big_leaf={big_leaf} resume_from={resume_from} =====", flush=True)
    with console_log.open("w") as log:
        process = subprocess.Popen(
            command, cwd=args.production_script_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            if not benchmark:
                print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        tail = "".join(console_log.read_text(errors="replace").splitlines(keepends=True)[-80:])
        raise RuntimeError(f"{name} 失败（exit={return_code}），日志={console_log}\n{tail}")
    console = console_log.read_text(errors="replace")
    expected = "ON" if big_leaf else "OFF"
    if f"[tt] big_leaf={expected}" not in console:
        raise RuntimeError(f"{name} 未出现 big_leaf={expected} 启动证据")
    attached_counts = [int(value) for value in re.findall(
        r"\[tt\] big_leaf DDP attached: modules=(\d+)", console)]
    if big_leaf and (not attached_counts or any(count <= 0 for count in attached_counts)):
        raise RuntimeError(f"{name} 没有非零 big_leaf attach: {attached_counts}")
    if not big_leaf and attached_counts:
        raise RuntimeError(f"{name} stock 意外启用了 big_leaf: {attached_counts}")
    for marker in ("xmegatron_ext auto-initialized", "xswift_ext auto-initialized"):
        if marker in console:
            raise RuntimeError(f"{name} 出现被禁用的厂商插件标记: {marker}")
    output_dir = run_dir / "output"
    result = RunResult(name, output_dir, console_log, _read_records(output_dir))
    if benchmark:
        _verify_benchmark_training(result, args.steps)
    else:
        _verify_training(result, args.steps)
    return result


def _iteration(row):
    return int(str(row["iteration"]).split("/", 1)[0])


def _train_rows(result):
    return [row for row in result.records if "loss" in row and "iteration" in row]


def _eval_rows(result):
    return [row for row in result.records if "eval_loss" in row]


def _verify_training(result, expected_steps):
    rows = _train_rows(result)
    if not rows or _iteration(rows[-1]) != expected_steps:
        raise RuntimeError(f"{result.name} 训练步数不符: {[r.get('iteration') for r in rows]}")
    losses = [float(row["loss"]) for row in rows]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"{result.name} 出现非有限 loss: {losses}")
    if not _eval_rows(result):
        raise RuntimeError(f"{result.name} 缺少 eval_loss")
    checkpoint = result.output_dir / f"checkpoint-{expected_steps}"
    if not checkpoint.is_dir() or not list(checkpoint.glob("iter_*")):
        raise RuntimeError(f"{result.name} 缺少 dist checkpoint: {checkpoint}")
    if not list(checkpoint.rglob("*.safetensors")):
        raise RuntimeError(f"{result.name} 缺少 HF safetensors: {checkpoint}")
    print(f"[verify] {result.name}: train/eval/save PASS, losses={losses}", flush=True)


def _verify_benchmark_training(result, expected_steps):
    rows = _train_rows(result)
    iterations = [_iteration(row) for row in rows]
    if iterations != list(range(1, expected_steps + 1)):
        raise RuntimeError(f"{result.name} benchmark 步数不符: {iterations}")
    losses = [float(row["loss"]) for row in rows]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"{result.name} benchmark 出现非有限 loss")
    print(f"[verify] {result.name}: {expected_steps} 步有限 loss PASS", flush=True)


def _compare_losses(left, right, label, iterations, *, metric="loss",
                    rtol=0.0, atol=0.0, exact_iterations=()):
    row_filter = _train_rows if metric == "loss" else _eval_rows
    left_rows = {_iteration(row): row for row in row_filter(left)}
    right_rows = {_iteration(row): row for row in row_filter(right)}
    comparisons = {}
    deltas = []
    for iteration in iterations:
        a = float(left_rows[iteration][metric])
        b = float(right_rows[iteration][metric])
        delta = b - a
        exact_required = iteration in exact_iterations or (rtol == 0 and atol == 0)
        accepted = a == b if exact_required else math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        comparisons[str(iteration)] = {
            "left": a, "right": b, "delta": delta, "equal": a == b,
            "relative_error": abs(delta) / max(abs(a), 1e-30),
            "exact_required": exact_required, "accepted": accepted,
        }
        deltas.append(delta)
    if not all(item["accepted"] for item in comparisons.values()):
        raise RuntimeError(f"{label} {metric} 超出验收阈值: {comparisons}")
    summary = {
        "mean_delta": sum(deltas) / len(deltas),
        "mean_abs_delta": sum(map(abs, deltas)) / len(deltas),
        "positive": sum(delta > 0 for delta in deltas),
        "negative": sum(delta < 0 for delta in deltas),
        "rtol": rtol, "atol": atol,
    }
    print(f"[verify] {label}: {metric} {iterations} PASS, summary={summary}", flush=True)
    return {"steps": comparisons, "summary": summary}


def _hf_tensors(checkpoint):
    files = sorted(checkpoint.rglob("*.safetensors"))
    output = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in output:
                    raise RuntimeError(f"HF 权重键重复: {key}")
                output[key] = reader.get_tensor(key)
    return output


def _compare_hf(left, right, label, *, bitwise=True, rtol=0.0, atol=0.0):
    left_dir = left.output_dir / "checkpoint-4"
    right_dir = right.output_dir / "checkpoint-4"
    a = _hf_tensors(left_dir)
    b = _hf_tensors(right_dir)
    if a.keys() != b.keys():
        raise RuntimeError(f"{label} HF 键集合不一致: {len(a)} vs {len(b)}")
    mismatches = [key for key in a if not torch.equal(a[key], b[key])]
    if mismatches and bitwise:
        raise RuntimeError(f"{label} HF 权重不一致: {mismatches[:20]}")
    max_abs = 0.0
    signed_sum = 0.0
    absolute_sum = 0.0
    element_count = 0
    positive = 0
    negative = 0
    for key in mismatches:
        delta = (b[key].float() - a[key].float()).double()
        if not bitwise:
            torch.testing.assert_close(a[key], b[key], rtol=rtol, atol=atol,
                                       msg=lambda msg, key=key: f"{label}/{key}: {msg}")
        max_abs = max(max_abs, float(delta.abs().max()))
        signed_sum += float(delta.sum())
        absolute_sum += float(delta.abs().sum())
        element_count += delta.numel()
        positive += int((delta > 0).sum())
        negative += int((delta < 0).sum())
    result = {
        "tensor_count": len(a), "bitwise": not mismatches,
        "mismatched_tensor_count": len(mismatches), "max_abs_delta": max_abs,
        "mean_delta": signed_sum / max(element_count, 1),
        "mean_abs_delta": absolute_sum / max(element_count, 1),
        "positive": positive, "negative": negative, "rtol": rtol, "atol": atol,
    }
    print(f"[verify] {label}: HF {result}", flush=True)
    return result


def lifecycle(args, run_root):
    results = {}
    for mode, big_leaf in (("stock", False), ("big", True)):
        initial = _run(args, run_root, f"{mode}-init", big_leaf)
        resume = _run(
            args, run_root, f"{mode}-resume", big_leaf,
            resume_from=initial.output_dir / "checkpoint-2",
        )
        results[f"{mode}_resume_losses"] = _compare_losses(
            initial, resume, f"{mode} uninterrupted-vs-resume", [3, 4])
        results[f"{mode}_resume_hf"] = _compare_hf(
            initial, resume, f"{mode} uninterrupted-vs-resume")
        results[mode] = (initial, resume)
    stock = results["stock"][0]
    big = results["big"][0]
    results["stock_big_losses"] = _compare_losses(
        stock, big, "stock-vs-big/train", [1, 2, 3, 4],
        rtol=5e-5, atol=1e-5, exact_iterations=(1, 2))
    results["stock_big_eval_losses"] = _compare_losses(
        stock, big, "stock-vs-big/eval", [2, 4], metric="eval_loss",
        rtol=5e-5, atol=1e-5, exact_iterations=(2,))
    results["stock_big_hf"] = _compare_hf(
        stock, big, "stock-vs-big", bitwise=False, rtol=1e-2, atol=2e-4)
    results.pop("stock")
    results.pop("big")
    return results


def _step_summary(run_dir, steps, warmup):
    path = run_dir / "steps.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [row["iteration"] for row in records] != list(range(1, steps + 1)):
        raise RuntimeError(f"step trace 不完整: {path}")
    if not all(row["success"] for row in records):
        raise RuntimeError(f"step trace 包含失败步: {path}")
    values = [float(row["elapsed_ms"]) for row in records if row["iteration"] > warmup]
    if not values:
        raise ValueError("benchmark warmup 后没有可统计步数")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "samples": len(values), "warmup": warmup,
        "median_ms": statistics.median(values), "mean_ms": statistics.mean(values),
        "p95_ms": ordered[p95_index], "min_ms": min(values), "max_ms": max(values),
    }


def benchmark(args, run_root):
    bench_args = argparse.Namespace(**vars(args))
    bench_args.save_steps = 0
    bench_args.eval_steps = 0
    rounds = []
    for index in range(args.rounds):
        pair = (("stock", False), ("big", True))
        if index % 2:
            pair = pair[::-1]
        result = {"round": index + 1}
        for mode, enabled in pair:
            name = f"round-{index + 1}-{mode}"
            _run(bench_args, run_root, name, enabled, benchmark=True)
            result[mode] = _step_summary(run_root / name, args.steps, args.warmup)
        result["speedup"] = result["stock"]["median_ms"] / result["big"]["median_ms"]
        result["big_faster"] = result["speedup"] > 1.0
        rounds.append(result)
        print(f"[benchmark] round={index + 1} speedup={result['speedup']:.6f}x "
              f"stock={result['stock']['median_ms']:.3f}ms "
              f"big={result['big']['median_ms']:.3f}ms", flush=True)
    stock_median = statistics.median(item["stock"]["median_ms"] for item in rounds)
    big_median = statistics.median(item["big"]["median_ms"] for item in rounds)
    stable = all(item["big_faster"] for item in rounds)
    return {
        "steps_per_run": args.steps, "warmup_steps": args.warmup,
        "round_count": args.rounds, "rounds": rounds,
        "stock_median_of_rounds_ms": stock_median,
        "big_median_of_rounds_ms": big_median,
        "speedup": stock_median / big_median,
        "stable_gain": stable,
        "timing": "rank0 train_step, device synchronized, no eval/checkpoint",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "lifecycle", "benchmark"),
                        default=os.environ.get("BIG_LEAF_MODE", "smoke"))
    parser.add_argument("--production-script-dir", default=os.environ.get("TRAIN_SCRIPT_DIR"))
    parser.add_argument("--run-dir", default=os.environ.get("BIG_LEAF_RUN_DIR"))
    parser.add_argument("--steps", type=int,
                        default=int(os.environ["BENCH_STEPS"]) if "BENCH_STEPS" in os.environ else None)
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("BENCH_ROUNDS", "3")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP", "20")))
    parser.add_argument("--save-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC_PER_NODE", "4")))
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--model", default=os.environ.get("TT_BIG_LEAF_MODEL"))
    parser.add_argument("--dataset", default=os.environ.get("TT_BIG_LEAF_DATA"))
    parser.add_argument("--tokenizer", default=os.environ.get("TT_BIG_LEAF_TOKENIZER"))
    args = parser.parse_args()
    if not args.production_script_dir:
        parser.error("请通过 --production-script-dir 或 TRAIN_SCRIPT_DIR 指定生产脚本目录")
    production_dir = Path(args.production_script_dir)
    missing_scripts = [name for name in ("run.sh", "train_megatron.sh")
                       if not (production_dir / name).is_file()]
    if missing_scripts:
        raise ValueError(
            f"生产脚本目录 {production_dir} 缺少 {missing_scripts}；请通过 "
            "TRAIN_SCRIPT_DIR 或 --production-script-dir 指定正确目录")
    _validate_production_script_contract(production_dir)
    args.synthetic = args.model is None
    if args.synthetic:
        if args.dataset is not None:
            raise ValueError("--dataset 必须与 --model 一起使用；默认资产由 runner 统一生成")
        args.model, args.dataset, asset_info = ensure_assets(
            ROOT / ".benchmarks/big_leaf_assets", tokenizer_source=args.tokenizer)
    else:
        if args.dataset is None:
            raise ValueError("外部 --model 必须同时提供 --dataset")
        asset_info = {"model_type": "external"}
    model_config = json.loads((Path(args.model) / "config.json").read_text())
    text_config = model_config.get("text_config", model_config)
    if model_config.get("quantization_config"):
        raise ValueError("big_leaf 生产验收拒绝 FP8/量化 checkpoint，请使用 BF16 权重")
    num_experts = text_config.get("num_experts", text_config.get("num_local_experts"))
    if not num_experts:
        raise ValueError("big_leaf 生产验收要求真实 MoE checkpoint（num_experts > 0）")
    if args.steps is None:
        args.steps = 200 if args.mode == "benchmark" else 4
    if args.mode == "smoke":
        args.steps = 2
    if args.steps <= 0 or args.rounds <= 0 or args.warmup < 0:
        raise ValueError("steps/rounds 必须为正数，warmup 不能为负数")
    if args.mode == "benchmark" and args.steps <= args.warmup:
        raise ValueError("benchmark 必须保留 warmup 后的测量步")
    if args.mode == "lifecycle" and (args.steps != 4 or args.save_steps != 2):
        raise ValueError("lifecycle 固定为连续 4 步、checkpoint-2 恢复至 4")
    run_root = Path(args.run_dir) if args.run_dir else ROOT / ".benchmarks" / (
        "big_leaf_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_root.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": args.mode, "run_dir": str(run_root),
        "model": str(args.model), "dataset": str(args.dataset),
        "synthetic": args.synthetic, "asset_info": asset_info,
        "config": {
            "steps": args.steps, "rounds": args.rounds, "warmup": args.warmup,
            "nproc": args.nproc, "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "xswift": False, "xme": False, "large_weight": False,
            "bf16_pin_cache": False,
        },
        "checks": {},
    }
    try:
        if args.mode == "smoke":
            _run(args, run_root, "big-smoke", True)
        elif args.mode == "lifecycle":
            report["checks"] = lifecycle(args, run_root)
        else:
            report["checks"] = benchmark(args, run_root)
            if not report["checks"]["stable_gain"]:
                raise RuntimeError("big_leaf 未在每一轮整网 benchmark 中获得稳定收益")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        (run_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nBIG_LEAF_PRODUCTION {report['status']}: {run_root / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
