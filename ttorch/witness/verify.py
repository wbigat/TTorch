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


def _sync(sync=None):
    """设备同步：优先用注入的平台同步函数（生产由 manager 从激活 handle 注入）；
    未注入时退回 torch.cuda 检查（独立使用 provider、未经 bridge 激活的场景）。
    不经 bridge 单独用时 xpu-native 上 torch.cuda.is_available() 为假——
    所以生产路径必须注入，否则不同步、计时失真。"""
    if sync is not None:
        sync()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_throughput_us(fn, args, kw, warmup=10, reps=200, sync=None):
    """稳态吞吐探测（us/次）：warmup 后连续 reps 次、末尾一次同步，摊薄同步开销。"""
    for _ in range(warmup):
        fn(*args, **kw)
    _sync(sync)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args, **kw)
    _sync(sync)
    return (time.perf_counter() - t0) / reps * 1e6


def verify_impl(impl_fn, native_fn, probe_input, *, min_speedup=0.0,
                enforce_benefit=True, sync=None, rtol=5e-3, atol=5e-3):
    """对一个高性能实现跑三级验证（与 native 基准对比）。返回 (ok, reason, evidence)。

    impl_fn / native_fn：可调用的实现本体；probe_input: () -> (args, kwargs)。
    enforce_benefit：False 时收益只作证据记录、不强制（native/reference 用）。
    sync：设备同步函数（当前平台 handle.synchronize），由 manager 注入。
    本函数纯测量，不认识 ProviderImpl——threshold 由调用方（provider.manager）传入。

    门禁基础设施自身失败（probe 构造 / 基准前向异常）按"该 impl 不可用"返回
    fail，不向上抛——训练不能被测量工具搞崩（不变式：失败只降性能不降功能）。
    """
    def _f32(seq):
        return [a.float() if isinstance(a, torch.Tensor) and a.is_floating_point()
                else a for a in seq]

    try:
        args, kw = probe_input()
        # kwargs 里的浮点张量同样升 fp32 且 clone 隔离——native/impl 共享同一
        # 对象时，impl 原地写会污染基准（当前语义未触发，机制层面堵上）。
        kw = {k: (v.float().clone() if isinstance(v, torch.Tensor) and v.is_floating_point()
                  else v) for k, v in kw.items()}
        # 精度对账升 fp32 轨道（判数学等价）：bf16 下同语义实现差（1-3 ulp，
        # 实测 swiglu 达 3e-2）与语义偏差只差一个数量级分不开；fp32 下同语义
        # ulp 级、语义差不变——阈值依然稳。需要梯度：复制两份独立叶子。
        args = [a.clone().requires_grad_(True)
                if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in _f32(args)]
        args2 = [a.detach().clone().requires_grad_(True)
                 if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                 for a in args]
        ref_out = native_fn(*args, **kw)
    except Exception as e:
        return False, f"gate_infra: 探测输入/基准前向失败 {type(e).__name__}: {e}", {}

    # 1. 功能准入：能跑、输出形状/类型与 native 一致
    try:
        out = impl_fn(*args2, **kw)
    except Exception as e:
        return False, f"functional: 探测输入上抛异常 {type(e).__name__}: {e}", {}
    if not (isinstance(out, torch.Tensor) and out.shape == ref_out.shape
            and out.dtype == ref_out.dtype):
        return False, (f"functional: 输出形状/类型不符 "
                       f"(expect {tuple(ref_out.shape)}/{ref_out.dtype})"), {}

    # 2. 精度准入：fwd 对账 + bwd 对账（训练态必须双验——实测教训：厂商
    #    rotary_pos_emb kernel fwd 逐位一致但 bwd 梯度错 2.43，只验 fwd 会漏）
    max_err = (out - ref_out).abs().max().item()
    if not torch.allclose(out, ref_out, rtol=rtol, atol=atol):
        return False, f"precision: max_abs_err={max_err:.2e} 超阈值", {}
    bwd_err = None
    if ref_out.requires_grad and out.requires_grad:
        try:
            g = torch.ones_like(ref_out)
            ref_out.backward(g)
            out.backward(g)
            for a1, a2 in zip(args, args2):
                if isinstance(a1, torch.Tensor) and a1.grad is not None:
                    ge = (a2.grad - a1.grad).abs().max().item()
                    bwd_err = ge if bwd_err is None else max(bwd_err, ge)
                    if not torch.allclose(a2.grad, a1.grad, rtol=rtol, atol=atol):
                        return False, f"precision(bwd): max_grad_err={ge:.2e} 超阈值", {}
        except Exception as e:
            return False, f"precision(bwd): 反向对账异常 {type(e).__name__}: {e}", {}

    # 3. 收益证据：稳态吞吐加速比；enforce_benefit 时才强制。
    #    重新取一份原始 probe（生产 dtype）——精度判定升 fp32 轨道，性能测量
    #    必须是生产负载本身，两轨道分开。
    perf_args, perf_kw = probe_input()
    probe_us = measure_throughput_us(impl_fn, perf_args, perf_kw, sync=sync)
    ref_us = measure_throughput_us(native_fn, perf_args, perf_kw, sync=sync)
    speedup = ref_us / probe_us if probe_us > 0 else float("inf")
    evidence = {"probe_us": round(probe_us, 1), "ref_us": round(ref_us, 1),
                "speedup": round(speedup, 2), "max_abs_err": round(max_err, 3),
                "max_grad_err": (round(bwd_err, 6) if bwd_err is not None else None)}
    if enforce_benefit and speedup < min_speedup:
        return False, (f"no_benefit: speedup={speedup:.2f}x "
                       f"< min_speedup={min_speedup}x"), evidence
    return True, "pass", evidence
