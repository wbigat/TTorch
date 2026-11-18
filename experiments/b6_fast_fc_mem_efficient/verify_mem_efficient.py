# Copyright (c) 2026 TTorch Authors. All Rights Reserved.
"""B6 真实环境验证：HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT 的行为与性能对比。

厂商口径：fast FC 为计算高效需要对权重做转置；训练过程可以 inplace
（减少峰值显存 + 减少反向开销），eval 不需要。本脚本在 hydrax linear
（XME linear 路径的真实本体）上验证：

1. 布局变异：mem-efficient 开/关 × train/eval，观察 weight 的
   stride / is_contiguous / data_ptr / 逻辑值 是否被改写；
2. train->eval->train 序列：eval 后再训练，输出与梯度是否逐位一致；
3. 性能对比：train fwd+bwd / eval fwd-only 的时延与峰值显存。

复现 XME `LinearWithGradAccumulationAndAsyncCommunication` 的调用方式
（allreduce/sequence_parallel 关闭，不涉及分布式），weight 的
`state_of(...).memory_efficient=True` 与 XME 默认（cast cache 开）一致。
"""
import argparse
import os
import statistics
import time

import torch
import torch_xmlir  # noqa: F401  # 加载 KL3 厂商栈

from hydrax.xaccelerator.linear import (
    linear_bwd_bgrad,
    linear_bwd_dgrad,
    linear_bwd_wgrad,
    linear_fwd,
)
from hydrax.xaccelerator.stateful import StatefulConfig, state_of
from torch_xmlir.config import config

ENV = "HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT"
stateful_config = StatefulConfig(retain_state_for_parameter=not config.disable_cast_cache)


def snapshot(t):
    return {
        "shape": tuple(t.shape),
        "stride": tuple(t.stride()),
        "contig": t.is_contiguous(),
        "data_ptr": t.data_ptr(),
    }


def train_step(x, w, b, grad_out):
    """复制 XME 路径的一次 fwd+bwd（无通信）。"""
    x = x.contiguous()
    with stateful_config.make_tensors_stateful(w):
        state_of(w).memory_efficient = True  # XME 默认：cast cache 开、非共享权重
        with stateful_config.make_tensors_stateful(x):
            out = linear_fwd(x, w, b)
    grad_in = torch.empty_like(x)
    with stateful_config.make_tensors_stateful(w):
        linear_bwd_dgrad(grad_out, w, grad_in)
    grad_w = torch.empty_like(w)
    with stateful_config.make_tensors_stateful(x):
        linear_bwd_wgrad(grad_out, x, grad_w)
    grad_b = linear_bwd_bgrad(grad_out)
    return out, grad_in, grad_w, grad_b


def eval_step(x, w, b):
    with torch.no_grad():
        with stateful_config.make_tensors_stateful(w):
            state_of(w).memory_efficient = True
            with stateful_config.make_tensors_stateful(x):
                return linear_fwd(x, w, b)


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def run_mode(mem_eff_enabled, T, H, warmup, iters, dtype=torch.bfloat16):
    """mem_eff_enabled=True 表示不设置禁用 env（厂商 mem-efficient 生效）。"""
    if mem_eff_enabled:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = "1"

    torch.manual_seed(0)
    dev = "cuda"
    w0 = torch.randn(H, H, device=dev, dtype=dtype)
    b0 = torch.randn(H, device=dev, dtype=dtype)
    x = torch.randn(T, H, device=dev, dtype=dtype)
    grad_out = torch.randn(T, H, device=dev, dtype=dtype)

    w = w0.clone()
    b = b0.clone()
    before = snapshot(w)
    w_logical_before = w.clone()

    # ---- 正确性/布局：train -> eval -> train ----
    out1, gi1, gw1, gb1 = train_step(x, w, b, grad_out)
    after_train1 = snapshot(w)
    out_eval = eval_step(x, w, b)
    after_eval = snapshot(w)
    out2, gi2, gw2, gb2 = train_step(x, w, b, grad_out)
    after_train2 = snapshot(w)

    checks = {
        "weight_value_unchanged": torch.equal(w, w_logical_before),
        "train2_out_eq_train1": torch.equal(out2, out1),
        "train2_gw_eq_train1": torch.equal(gw2, gw1),
        "train2_gi_eq_train1": torch.equal(gi2, gi1),
        "eval_out_eq_train_out": torch.equal(out_eval, out1),
    }

    # ---- 性能：train fwd+bwd ----
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_us = timed(lambda: train_step(x, w, b, grad_out), warmup, iters)
    train_peak_mib = torch.cuda.max_memory_allocated() / 2**20

    # ---- 性能：eval fwd-only ----
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    eval_us = timed(lambda: eval_step(x, w, b), warmup, iters)
    eval_peak_mib = torch.cuda.max_memory_allocated() / 2**20

    return {
        "mode": "mem_efficient_ON" if mem_eff_enabled else "mem_efficient_OFF(env=1)",
        "layout": {
            "before": before,
            "after_train1": after_train1,
            "after_eval": after_eval,
            "after_train2": after_train2,
        },
        "checks": checks,
        "train_us": train_us,
        "train_peak_mib": train_peak_mib,
        "eval_us": eval_us,
        "eval_peak_mib": eval_peak_mib,
        # 供跨模式数值对比
        "_tensors": (out1, gw1, gi1, out_eval),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"device={torch.cuda.get_device_name(0)} shape=[{args.tokens},{args.hidden}]x[{args.hidden},{args.hidden}] {args.dtype}")
    print(f"torch_xmlir disable_cast_cache={config.disable_cast_cache}")

    res_on = run_mode(True, args.tokens, args.hidden, args.warmup, args.iters, dtype)
    res_off = run_mode(False, args.tokens, args.hidden, args.warmup, args.iters, dtype)

    for r in (res_on, res_off):
        print(f"\n===== {r['mode']} =====")
        for k, v in r["layout"].items():
            print(f"  weight {k}: stride={v['stride']} contig={v['contig']} ptr={v['data_ptr']:#x}")
        for k, v in r["checks"].items():
            print(f"  check {k}: {v}")
        print(f"  train fwd+bwd: {r['train_us']:.1f} us/it, peak {r['train_peak_mib']:.1f} MiB")
        print(f"  eval  fwd    : {r['eval_us']:.1f} us/it, peak {r['eval_peak_mib']:.1f} MiB")

    # 跨模式数值一致性
    names = ["out", "grad_w", "grad_in", "eval_out"]
    print("\n===== cross-mode numeric =====")
    for n, a, b in zip(names, res_on["_tensors"], res_off["_tensors"]):
        print(f"  {n}: bitwise_equal={torch.equal(a, b)}")

    print("\n===== summary =====")
    print(f"  train speedup ON/OFF: {res_off['train_us']/res_on['train_us']:.3f}x "
          f"({res_on['train_us']:.1f} vs {res_off['train_us']:.1f} us)")
    print(f"  eval  speedup ON/OFF: {res_off['eval_us']/res_on['eval_us']:.3f}x "
          f"({res_on['eval_us']:.1f} vs {res_off['eval_us']:.1f} us)")
    print(f"  train peak mem delta (OFF-ON): {res_off['train_peak_mib']-res_on['train_peak_mib']:+.1f} MiB")
    print(f"  eval  peak mem delta (OFF-ON): {res_off['eval_peak_mib']-res_on['eval_peak_mib']:+.1f} MiB")


if __name__ == "__main__":
    main()
