"""provider 实现的精度/性能验证原语（三级门禁的测量本体）。

归位 witness 的理由（2026-10-15）：验证 = 测量（精度对账 + 稳态吞吐探测），是
独立的取证能力，与"用测量结果做选择"解耦——选择缓存/编排留在 provider.manager，
本模块只提供可复用、可独立测试的测量原语。

三级门禁（对应 TT_Provider_Function_Map.md §2 gate）：
1. 功能：能跑、输出形状/类型与 native 一致；
2. 精度：对 native abs/rel err 在阈值内；
3. 收益：稳态吞吐加速比（声明了 min_speedup 才强制）。
"""
import time

import torch

__all__ = ["measure_throughput_us", "verify_impl"]


def _sync():
    """设备同步：provider 面向加速器，无 CUDA 的平台（cpu）跳过以避免报错。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_throughput_us(fn, args, kw, warmup=10, reps=200):
    """稳态吞吐探测（us/次）：warmup 后连续 reps 次、末尾一次同步，摊薄同步开销。"""
    for _ in range(warmup):
        fn(*args, **kw)
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args, **kw)
    _sync()
    return (time.perf_counter() - t0) / reps * 1e6


def verify_impl(impl_fn, native_fn, probe_input, *, min_speedup=0.0,
                enforce_benefit=True, rtol=5e-3, atol=5e-3):
    """对一个高性能实现跑三级验证（与 native 基准对比）。返回 (ok, reason, evidence)。

    impl_fn / native_fn：可调用的实现本体；probe_input: () -> (args, kwargs)。
    enforce_benefit：False 时收益只作证据记录、不强制（native/reference 用）。
    本函数纯测量，不认识 ProviderImpl——threshold 由调用方（provider.manager）传入。
    """
    args, kw = probe_input()
    ref_out = native_fn(*args, **kw)

    # 1. 功能准入：能跑、输出形状/类型与 native 一致
    try:
        out = impl_fn(*args, **kw)
    except Exception as e:
        return False, f"functional: 探测输入上抛异常 {type(e).__name__}: {e}", {}
    if not (isinstance(out, torch.Tensor) and out.shape == ref_out.shape
            and out.dtype == ref_out.dtype):
        return False, (f"functional: 输出形状/类型不符 "
                       f"(expect {tuple(ref_out.shape)}/{ref_out.dtype})"), {}

    # 2. 精度准入：对 native abs/rel err 在阈值内
    max_err = (out - ref_out).abs().max().item()
    if not torch.allclose(out, ref_out, rtol=rtol, atol=atol):
        return False, f"precision: max_abs_err={max_err:.2e} 超阈值", {}

    # 3. 收益证据：稳态吞吐加速比；enforce_benefit 时才强制
    probe_us = measure_throughput_us(impl_fn, args, kw)
    ref_us = measure_throughput_us(native_fn, args, kw)
    speedup = ref_us / probe_us if probe_us > 0 else float("inf")
    evidence = {"probe_us": round(probe_us, 1), "ref_us": round(ref_us, 1),
                "speedup": round(speedup, 2), "max_abs_err": round(max_err, 3)}
    if enforce_benefit and speedup < min_speedup:
        return False, (f"no_benefit: speedup={speedup:.2f}x "
                       f"< min_speedup={min_speedup}x"), evidence
    return True, "pass", evidence
