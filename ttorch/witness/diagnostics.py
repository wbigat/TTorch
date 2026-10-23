"""witness/diagnostics：带副作用的按需诊断（用户知情主动调用）。

与 conformance 的分工（2026-10-22 确立）：
- conformance：激活期准入门禁——无副作用、fail fast、失败即平台不可用；
- 本模块：**运行期按需诊断**——验证项带不可逆副作用（如清峰值统计），
  必须用户知情主动调用，故不放激活期强制套件。

激活期验证不破坏被观测对象的状态，这是 layering 纪律。

依赖方向：只用 `torch.cuda.*` 门面符号（激活后已是 TT 接管的门面），不依赖
platform.manager——遵守 witness → 仅 platform 门面的单向依赖。

（历史呼应：witness 2026-10-15 由 ttorch/diagnostics.py 收拢而成，本文件重新
独立出来专放"带副作用的显式诊断"，与无副作用的 verify/provenance 区分。）
"""
from ttorch.witness import provenance

__all__ = ["mem_reset_semantics", "measure_forward_tax",
           "verify_provider_generalization", "measure_dispatch_tax"]


def measure_forward_tax(adapter=None, symbol="current_device", n=5000):
    """D8：热路径税量化**基线**——同一符号在「实现本体 / handle / 门面」三层的
    单次调用开销（us）。

    TT 的收束把每次调用从"直接调厂商实现"变成"门面 → handle → 实现"，代价是转发
    开销（历史上实测 16ns→85ns→155ns，约 10x）。这个数字**不是性能问题**（相对
    kernel 调用可忽略），但**没有基线就无法回答**"收束收了多少税"，也无法在改动
    转发链后判断是否劣化——D8 要的就是这个基线，不是优化。

    adapter 由调用方传入（本模块只依赖 torch.cuda 门面，不依赖 platform.manager）；
    为 None 时只测本体与门面两层（handle 层标 None）。

    返回 dict：各层 us/次 + 门面相对本体的倍数（tax_x）。
    """
    import time
    import torch
    from ttorch.utils import _rebind

    mod = torch.cuda
    # 实现本体：绑定前的原实现（owner 留痕）；未绑定则回落当前值
    native_fn = getattr(mod, _rebind.origin_attr("bridge", symbol), getattr(mod, symbol))
    facade_fn = getattr(mod, symbol)
    handle_fn = getattr(adapter, symbol, None) if adapter is not None else None

    def bench(fn):
        fn()                                          # warmup（首次可能触发惰性解析）
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6   # us/次

    native_us = bench(native_fn)
    facade_us = bench(facade_fn)
    handle_us = bench(handle_fn) if handle_fn is not None else None
    tax = facade_us / native_us if native_us else float("inf")
    return {"symbol": symbol, "n": n,
            "native_us": round(native_us, 4),
            "handle_us": (round(handle_us, 4) if handle_us is not None else None),
            "facade_us": round(facade_us, 4), "tax_x": round(tax, 2),
            "detail": (f"{symbol}: 本体 {native_us:.3f}us -> handle "
                       f"{('%.3f' % handle_us) if handle_us is not None else 'n/a'}us "
                       f"-> 门面 {facade_us:.3f}us（税率 {tax:.2f}x）")}


def mem_reset_semantics():
    """验证 max_memory_allocated 的峰值窗口与 reset_peak_memory_stats 语义
    （D7 第三口径：峰值统计窗口与 reset 语义）。

    **副作用警告**：会调 reset_peak_memory_stats() **清除峰值统计**——请在
    没有需要保留的峰值数据时调用（如训练开始前）。这正是它不放激活期
    强制门禁的原因。

    判据（有标杆、自洽）：分配已知量抬高峰值 → reset → 峰值应回落到当前
    allocated 附近（torch 语义：reset 后峰值重置为当前值）。

    返回 {"ok": bool|None, "detail": str, ...}；ok=None 表示平台未实现跳过。
    结果同时记 provenance（LAYER_PLATFORM, event=caliber）。
    """
    import torch
    tc = torch.cuda  # 激活后这些符号已是 TT 接管的门面
    mb = 1024**2
    try:
        x = torch.zeros(64 * mb, dtype=torch.uint8, device="cuda")
        peak_with = tc.max_memory_allocated()        # 含刚分配的 64M
        cur = tc.memory_allocated()
        tc.reset_peak_memory_stats()
        peak_after = tc.max_memory_allocated()       # 应回落到 ≈ 当前 allocated
        del x
        tc.empty_cache()
    except Exception as e:
        return {"ok": None,
                "detail": f"平台未实现/不支持该诊断 {type(e).__name__}: {e}"}
    ok = peak_after < peak_with and peak_after <= cur + 1 * mb
    provenance.record(provenance.LAYER_PLATFORM, "caliber", "mem_reset_semantics",
                      "pass" if ok else "fail",
                      peak_before=int(peak_with), peak_after_reset=int(peak_after),
                      allocated_at_reset=int(cur))
    return {"ok": ok,
            "detail": (f"峰值 {peak_with // mb}M →reset→ {peak_after // mb}M "
                       f"（reset 时 allocated={cur // mb}M，应回落≈当前值）"),
            "peak_before": int(peak_with), "peak_after_reset": int(peak_after)}


def _edge_input_variants(args, kw):
    """边界输入变体：特征维（最后一维）×2/÷2 + 批次维（第一维）空(0)/单(1)。

    覆盖 kernel 正确性的高危路径（实测教训：D=256→512 特征维静默算错；KLX
    fused_bias_swiglu 空张量特判——MoE 某 expert 分到 0 token 是常态）。
    非张量参数（eps 等标量）原样带上；任一变体构造失败即跳过（不为取证破坏
    已能工作的路径）。基础组始终在首位。
    """
    import torch
    groups = [(args, kw)]
    for factor in (2, 0.5):
        try:
            new_args = []
            for v in args:
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    last = v.shape[-1]
                    n_last = int(last * factor)
                    if n_last <= 0:
                        raise ValueError("特征维变体非法")
                    if factor >= 1:
                        idx = torch.arange(n_last, device=v.device) % last
                        new_args.append(v.index_select(-1, idx))
                    else:
                        new_args.append(v.narrow(-1, 0, n_last))
                else:
                    new_args.append(v)
            groups.append((tuple(new_args), kw))
        except Exception:
            continue
    for n_seq in (0, 1):
        try:
            new_args = [(v[:n_seq] if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
                        for v in args]
            groups.append((tuple(new_args), kw))
        except Exception:
            continue
    return groups


def verify_provider_generalization(provider, ref_fn=None, device=None, tol=5e-3):
    """D9：provider 实现的泛化校验（witness 层，针对问题多的 case）。

    **分层定位（2026-10-23）**：provider 的兜底校验（`_admit_native_vs_orig` 对账 /
    `_self_check` / 门禁 `verify_impl`）用**单点、轻量**——只确认"这次接管安全"，
    不可能全面。本函数是 witness 层的**更全校验**：对"问题多的 case"（如
    bias_swiglu 的空张量），用边界输入集（特征维×2/÷2 + 批次维空/单）全面校验
    各实现在边界下的正确性。

    **触发方式**：手动调用，或 e2e 测试时触发——不放激活期/热路径（全面校验有
    多次前向开销）。

    ref_fn：参考实现（框架原实现，校验 native/impl vs 原实现）；None 时校验
    impl vs native（native 为基准）。返回 {"ok", "worst_rel", "checks"}。
    """
    import torch
    from ttorch.platform import manager as _pfm
    if device is None:
        device = _pfm.activation().device_spec()
    native = provider.native
    ref = ref_fn if ref_fn is not None else native
    if ref_fn is not None:
        targets = [native] + list(provider.impls)      # 校验 native/impl vs 原实现
    else:
        targets = list(provider.impls) or [native]     # impl vs native；无 impl 则校验 native 自身边界

    checks = []
    worst = 0.0
    base_args, base_kw = provider.probe_input(device=device)
    for args, kw in _edge_input_variants(base_args, base_kw):
        shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        for impl in targets:
            try:
                ref_out = ref(*args, **kw)
                out = impl(*args, **kw)
                if ref_out.numel() == 0:
                    # 空张量：无数值可比（max 报错），校验"不崩 + 形状一致"即可——
                    # 空张量特判的正确性就是"不崩 + 形状对"，数值无意义。
                    checks.append({"impl": impl.name, "shapes": shapes,
                                   "ok": out.shape == ref_out.shape,
                                   "note": "空张量：形状一致"})
                    continue
                scale = float(ref_out.float().abs().max()) + 1e-12
                rel = float((out.float() - ref_out.float()).abs().max()) / scale
                worst = max(worst, rel)
                checks.append({"impl": impl.name, "shapes": shapes,
                               "rel": round(rel, 6), "ok": rel < tol})
            except Exception as e:
                checks.append({"impl": impl.name, "shapes": shapes, "ok": False,
                               "error": f"{type(e).__name__}: {e}"})
                worst = max(worst, 1.0)
    ok = all(c["ok"] for c in checks)
    provenance.record(provenance.LAYER_PROVIDER, "generalization",
                      provider.semantic, "pass" if ok else "fail",
                      worst_rel=round(worst, 6), n_checks=len(checks))
    return {"ok": ok, "worst_rel": round(worst, 6), "checks": checks}


def measure_dispatch_tax(semantic="rmsnorm", device=None, n=2000):
    """D8 ②：provider 函数级通道的分发热路径税基线。

    `dispatcher.call` 完整开销（`_call_sig` 构造 + 门禁缓存查找 + 分发）vs 小
    shape kernel 耗时。警惕场景（D8）：小 shape 算子，Python 分发开销与 kernel
    耗时同量级。实测（2026-10-23，D=256 rmsnorm）：call 77.7us vs kernel 68us，
    分发税 14%。**只作基线（防劣化），不优化**——若未来超阈值再考虑优化手段。

    返回 {"kernel_us", "call_us", "tax_us", "tax_pct", "detail"}；provider 未注册
    返回 {"ok": False}。
    """
    import time
    import torch
    from ttorch.provider import manager as _pm
    mgr = _pm.instance()
    provider = mgr.registry.get(semantic)
    if provider is None:
        return {"ok": False, "detail": f"provider {semantic} 未注册"}
    args, kw = provider.probe_input(device=device)

    def _sync():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    def bench(fn, reps):
        fn(); _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        _sync()
        return (time.perf_counter() - t0) / reps * 1e6  # us

    kernel_us = bench(lambda: provider.native(*args, **kw), n)
    call_us = bench(lambda: mgr.call(semantic, *args, slot=provider.slot, **kw), n)
    tax_us = call_us - kernel_us
    tax_pct = tax_us / kernel_us * 100 if kernel_us else float("inf")
    return {"ok": True, "semantic": semantic, "n": n,
            "kernel_us": round(kernel_us, 3), "call_us": round(call_us, 3),
            "tax_us": round(tax_us, 3), "tax_pct": round(tax_pct, 1),
            "detail": (f"{semantic}: kernel {kernel_us:.2f}us -> call {call_us:.2f}us"
                       f"（分发税 {tax_us:.2f}us / {tax_pct:.1f}%）")}
