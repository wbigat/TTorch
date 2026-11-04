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


# 精度轨道的低精度相对阈值（A2，2026-11-03）。放宽的代价必须写明：
#   能拦住：倍数级/语义级偏差（实测反例：bf16 下输出翻倍 rel≈1.0）、梯度方向性错误、
#           kernel 走错 dtype 分支导致的结构性偏差；
#   拦不住：同量级的舍入顺序差异（如 zcg 的 (1+w) 在 weight dtype 域 vs fp32 域
#           求值）。要区分这类需引入 fp64/fp32 真值的三方对比（err_impl vs
#           err_native 同量级判据），属后续精化项，当前刻意不做。
_LOWP_REL_TOL: dict = {torch.bfloat16: 4e-2, torch.float16: 1e-2}


def _rel_tol(dtype, base):
    """精度轨道的 dtype 自适应相对阈值（A2，2026-11-03）。

    低精度 dtype 按机器 epsilon 放宽约 10 ulp（bf16 eps=3.9e-3、fp16 eps=9.8e-4）；
    fp32/fp64 用调用方给的 base（保持原有严格度不变）。
    """
    return _LOWP_REL_TOL.get(dtype, base)


def _rel_err(out, ref):
    """max 相对误差，口径沿用 provider.spec._self_check（不另造一套）：
    rel = max|out-ref| / (max|ref| + tiny)。"""
    scale = ref.abs().max().item() + 1e-12
    return (out - ref).abs().max().item() / scale


def _sync(sync=None):
    """设备同步：优先用注入的平台同步函数（生产由 manager 从激活 handle 注入）；
    未注入时退回 torch.cuda 检查（独立使用 provider、未经 bridge 激活的场景）。
    不经 bridge 单独用时 xpu-native 上 torch.cuda.is_available() 为假——
    所以生产路径必须注入，否则不同步、计时失真。"""
    if sync is not None:
        sync()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_throughput_us(fn, args, kw, warmup=10, reps=200, sync=None, bwd=False):
    """稳态吞吐探测（us/次）：warmup 后连续 reps 次、末尾一次同步，摊薄同步开销。

    bwd=True（A1 步骤 5，2026-11-03）：每次迭代跑 **forward + backward**，对齐
    训练态口径。只测前向的收益是假象——建图与反向都不计入时，省掉建图的实现
    （含断图实现）反而更快，speedup 偏高（实测教训：整网曾因此把 2.5x 误判，
    真实约 1x）。用 autograd.grad 而非 .backward()：不写入 .grad，避免跨迭代
    梯度累加干扰计时与显存。
    """
    leaves = []
    if bwd:
        leaves = [a for a in args if isinstance(a, torch.Tensor)
                  and a.is_leaf and a.requires_grad]

    def _once():
        out = fn(*args, **kw)
        if leaves and isinstance(out, torch.Tensor) and out.requires_grad:
            torch.autograd.grad(out, leaves, torch.ones_like(out))

    for _ in range(warmup):
        _once()
    _sync(sync)
    t0 = time.perf_counter()
    for _ in range(reps):
        _once()
    _sync(sync)
    return (time.perf_counter() - t0) / reps * 1e6


def verify_impl(impl_fn, native_fn, probe_input, *, min_speedup=0.0,
                enforce_benefit=True, sync=None, rtol=5e-3, atol=5e-3):
    """对一个高性能实现跑三级验证（与 native 基准对比）。返回 (ok, reason, evidence)。

    impl_fn / native_fn：可调用的实现本体；probe_input: () -> (args, kwargs)。
    enforce_benefit：False 时收益只作证据记录、不强制（native/reference 用）。
    sync：设备同步函数（当前平台 handle.synchronize），由 manager 注入。
    rtol/atol：fp32 轨道的相对阈值基准（低精度 dtype 由 _LOWP_REL_TOL 覆盖；
    atol 当前未参与判据，保留以兼容调用方签名）。
    本函数纯测量，不认识 ProviderImpl——threshold 由调用方（provider.manager）传入。

    门禁基础设施自身失败（probe 构造 / 基准前向异常）按"该 impl 不可用"返回
    fail，不向上抛——训练不能被测量工具搞崩（不变式：失败只降性能不降功能）。

    **梯度契约（A1，2026-11-03）**：梯度契约以 native 为真值推导（native 产出
    哪些梯度就是本语义的契约），候选缺图/缺梯度按**契约违约**拒绝，不再当作
    "不需要验反向"。evidence 的 bwd_checked/bwd_skip_reason/grad_contract 三态
    化，区分"native 不可微合法未测"与"候选断图"。收益口径同步对齐训练态
    （native 需反向时两侧都测 fwd+bwd，见 evidence.perf_scope）。
    """
    try:
        args, kw = probe_input()
        # kwargs 里的浮点张量 clone 隔离——native/impl 共享同一对象时，impl
        # 原地写会污染基准（当前语义未触发，机制层面堵上）。
        kw = {k: (v.clone() if isinstance(v, torch.Tensor) and v.is_floating_point()
                  else v) for k, v in kw.items()}
        # 精度对账在**生产 dtype** 下执行（A2，2026-11-03 修正）。
        # 旧实现把输入升 fp32 再比，改变的是 kernel 实际走的执行路径而不只是
        # 比较精度——厂商 kernel 常按 dtype 分派（fastbf16/fp32 走不同符号），
        # 于是生产用的那份 kernel 从未被门禁调用过。实测反例：候选 fp32 正确、
        # bf16 返回两倍，门禁仍 pass 且 max_abs_err=0.0（bf16 路径被执行 210 次，
        # 一次都没看输出）。代价是低精度下舍入噪声与小幅语义偏差分不开——
        # 按 _LOWP_REL_TOL 放宽阈值，拦不住的那类已在该常量注释中写明。
        # 需要梯度：复制两份独立叶子。
        args = [a.clone().requires_grad_(True)
                if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args]
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
    #    判据改为 max 相对误差 vs dtype 自适应阈值（A2）：生产 dtype 轨道下
    #    allclose 的固定 rtol/atol 会被 bf16 自身的 ulp 噪声顶穿（基准第 6 条：
    #    噪声随 dtype 变，固定阈值必然失效）。
    #    atol 在新判据里的角色 = "什么算有效为零"：基准输出整体近零时相对判据
    #    的分母是 1e-12 量级，既判不出通过也判不出失败（与 A3 的退化输入同根），
    #    按门禁基础设施问题诚实报出，而不是让 0/1e-12=0 混成 pass。
    ref_scale = ref_out.abs().max().item()
    if ref_scale <= atol:
        return False, (f"gate_infra: 基准输出近零（max|ref|={ref_scale:.2e} "
                       f"<= atol={atol:.1e}），精度判据无判别力"), {}
    tol = _rel_tol(ref_out.dtype, rtol)
    max_err = (out - ref_out).abs().max().item()
    rel_err = _rel_err(out, ref_out)
    if rel_err > tol:
        return False, (f"precision: max_rel_err={rel_err:.2e} > tol={tol:.1e} "
                       f"(dtype={ref_out.dtype}, max_abs_err={max_err:.2e})"), {}

    # 2b. 梯度契约 + 梯度存在性，再比梯度数值（2026-11-03 修正，A1）。
    #     旧实现把 `out.requires_grad` 写在守卫位（`if ref_out.requires_grad and
    #     out.requires_grad`）——等于让候选自证豁免反向验证：厂商推理 kernel
    #     （如 rms_layer_norm，输出无 grad_fn）前向逐位正确即可拿到 pass，
    #     evidence 还记成 max_grad_err=None（"零误差"），上线后梯度图在该模块
    #     被切断。实测反例：native x.square() vs 候选 x.square().detach()
    #     → True/pass。叠加 measure_throughput_us 只跑前向，detached 候选省掉
    #     建图开销 speedup 反而更高——门禁不仅漏判，还会**优先选中**错误实现。
    #     正确顺序：契约以 native 为真值推导 -> 验候选契约 -> 验梯度存在性 ->
    #     才比数值。缺图是契约违约，不是"不需要验反向"。
    bwd_err = None
    expect_graph = bool(ref_out.requires_grad)
    contract = "match"
    skip_reason = None if expect_graph else "native_no_graph"

    def _ev(extra=None):
        """失败/成功共用的证据底座：bwd_checked 三态化——区分"未测"与"测过=0"。
        旧实现 max_grad_err 只有 float|None 两态，None 同时表示"native 不可微，
        合法未测"与"候选断图被漏放行"，provenance/summary 无法分辨。"""
        ev = {"bwd_checked": expect_graph and contract == "match",
              "bwd_skip_reason": skip_reason, "grad_contract": contract,
              "probe_dtype": str(ref_out.dtype), "rel_tol": tol,
              "max_abs_err": round(max_err, 6), "max_rel_err": round(rel_err, 6)}
        if extra:
            ev.update(extra)
        return ev

    if expect_graph:
        try:
            g = torch.ones_like(ref_out)
            ref_out.backward(g)
        except Exception as e:
            # native 是我们自己的基准：它反向不了属门禁基础设施问题，
            # 不能记成候选的精度问题（归因要分清）。
            return False, f"gate_infra: 基准反向失败 {type(e).__name__}: {e}", _ev()
        # native 产出哪些梯度 = 本语义的梯度契约（真值来自 native，不采信候选自述）
        expect_grad = [isinstance(a, torch.Tensor) and a.grad is not None for a in args]
        if not out.requires_grad:
            contract = "candidate_detached"
            return False, ("grad_contract: 候选输出无 autograd 图而 native 需要反向"
                           "（常见原因：厂商推理 kernel 未包 autograd.Function）"), _ev()
        try:
            out.backward(g)
        except Exception as e:
            return False, f"precision(bwd): 候选反向异常 {type(e).__name__}: {e}", _ev()
        try:
            for i, (a1, a2) in enumerate(zip(args, args2)):
                if not expect_grad[i]:
                    continue
                if a2.grad is None:
                    contract = "grad_missing"
                    return False, (f"grad_missing: native 产出 arg[{i}] 梯度，"
                                   f"候选未产出"), _ev()
                # 梯度同样走生产 dtype + 相对阈值（与前向同一口径，A2）
                ge = _rel_err(a2.grad, a1.grad)
                gtol = _rel_tol(a1.grad.dtype, rtol)
                bwd_err = ge if bwd_err is None else max(bwd_err, ge)
                if ge > gtol:
                    return False, (f"precision(bwd): arg[{i}] max_rel_grad_err="
                                   f"{ge:.2e} > tol={gtol:.1e}"), _ev()
        except Exception as e:
            return False, f"precision(bwd): 反向对账异常 {type(e).__name__}: {e}", _ev()
    elif out.requires_grad:
        # 候选比 native 多建图：更保守、不损失正确性，不阻断，只留痕。
        contract = "candidate_extra_graph"

    # 3. 收益证据：稳态吞吐加速比；enforce_benefit 时才强制。
    #    重新取一份干净 probe（避免复用上面已建图/已带梯度的张量影响计时）。
    #    口径（A1 步骤 5 + A2）：与精度轨道同为**生产 dtype**；native 需要反向
    #    时两侧都测 forward+backward——否则建图开销不计入，省掉建图的实现
    #    speedup 虚高，收益门禁会偏好错误实现。
    perf_args, perf_kw = probe_input()
    if expect_graph:
        perf_args = [a.clone().requires_grad_(True)
                     if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                     for a in perf_args]
    probe_us = measure_throughput_us(impl_fn, perf_args, perf_kw,
                                     sync=sync, bwd=expect_graph)
    ref_us = measure_throughput_us(native_fn, perf_args, perf_kw,
                                   sync=sync, bwd=expect_graph)
    speedup = ref_us / probe_us if probe_us > 0 else float("inf")
    evidence = _ev({"probe_us": round(probe_us, 1), "ref_us": round(ref_us, 1),
                    "speedup": round(speedup, 2),
                    # 口径标注：speedup 是 fwd+bwd 还是仅 fwd，两者不可混比
                    "perf_scope": "fwd+bwd" if expect_graph else "fwd",
                    "max_grad_err": (round(bwd_err, 6)
                                     if bwd_err is not None else None)})
    if enforce_benefit and speedup < min_speedup:
        return False, (f"no_benefit: speedup={speedup:.2f}x "
                       f"< min_speedup={min_speedup}x"), evidence
    return True, "pass", evidence
