"""conformance 套件：验证 adapter 是否履行统一接口契约（实现正确性）。

与 admission 的分层：
- admission（cuda-compat 侧）验"绑定正确性"——框架侧语义在替换后保持；
- conformance（本模块）验"adapter 正确性"——契约被真正实现。

激活期由 manager._activate 调用：任一强制测试 FAIL 即拒绝激活。

裁剪依据是**实现**而非声明（2026-10-14）：测试标注所需的平台接口，平台
没实现（`adapter.supports(api)` 为假）就跳过——真实栈可能合法地没有 mem
查询 / stream 族，一刀切 FAIL 会误杀。关键区分：**没实现=跳过**（合法降级，
调用方收 unsupported），**实现了但没过契约=FAIL**（拒绝激活）。

可选能力（graph 捕获 / 异步通信）走 _ENDORSEMENT：无条件实测，结果即事实，
不阻断激活——取代了原先"平台自报 capability=True 再验证它没说谎"的两步走。

测试分层：mock adapter 带 hw 钩子（submit/pending/alloc/wait_edges）走
深测版（能构造确定性反例）；真实 adapter 没有钩子，走 tensor 级契约版
（用真实算子入队 + 数值校验推断语义是否履行——弱于深测但真实可跑）。
"""
from collections import namedtuple

Result = namedtuple("Result", ["name", "ok", "detail"])

__all__ = ["Result", "run",
           "sync_scope_device_wide", "stream_dynamic_coverage",
           "mem_query_realtime", "graph_capture_roundtrip",
           "async_comm_smoke", "rng_state_roundtrip", "stream_fencing_semantics",
           "stream_ctx_restore", "mem_info_consistency", "device_identity_consumable",
           "mem_caliber_coherence"]


def _dev(adapter):
    """当前设备的 torch device spec（tensor 级测试用）。"""
    return f"{adapter.device_name}:{adapter.current_device()}"


def _hw_hooks(adapter):
    """mock 深测钩子；真实 adapter 返回 None。"""
    hw = getattr(adapter, "hw", None)
    return hw if hw is not None and hasattr(hw, "submit") else None


def _stream_key(s):
    """stream 身份的稳定可比较 key（测试判等用，不是平台契约）。

    真实 torch Stream 每次包装可能是新对象（`current_stream()` 内部按
    stream_id/device_index 重新构造），直接比对象会误判；取
    (device_index, stream_id) 作稳定 key。mock 的 stream 对象自带可比较
    语义，原样返回即可。
    """
    if hasattr(s, "stream_id"):
        return (getattr(s, "device_index", None), s.stream_id)
    return s


def sync_scope_device_wide(adapter):
    """device-wide 等待（强制契约）：侧流有未完成 work，sync 后必须完成。"""
    hw = _hw_hooks(adapter)
    if hw is not None:
        hw.submit(1, "w1")
        hw.submit(2, "w2")
        adapter.synchronize()
        left = hw.pending(1) + hw.pending(2)
        return Result("sync_scope_device_wide", left == 0,
                      f"{adapter.name}: s1={hw.pending(1)} s2={hw.pending(2)} 残留")
    import torch
    dev = _dev(adapter)
    x = torch.randn(256, 256, device=dev)
    expect = (x @ x).cpu()
    out = torch.empty(256, 256, device=dev)
    side = adapter.Stream()
    main = adapter.current_stream()
    adapter.set_stream(side)
    out.copy_(x @ x)  # 侧流上的异步 work
    adapter.set_stream(main)
    adapter.synchronize()
    ok = torch.allclose(out.cpu(), expect)
    return Result("sync_scope_device_wide", ok,
                  f"{adapter.name}: tensor 级校验侧流 work 完成 -> {ok}")


def stream_dynamic_coverage(adapter):
    """运行期新建的 stream 也必须被 device-wide sync 覆盖（强制契约）。"""
    hw = _hw_hooks(adapter)
    if hw is not None:
        sid = hw.create_stream()
        hw.submit(sid, "late-work")
        adapter.synchronize()
        left = hw.pending(sid)
        return Result("stream_dynamic_coverage", left == 0,
                      f"{adapter.name}: 新建 stream {sid} 残留 {left}")
    import torch
    dev = _dev(adapter)
    sid = adapter.Stream()
    prev = adapter.current_stream()
    out = torch.empty(128, device=dev)
    adapter.set_stream(sid)
    out.copy_(torch.full((128,), 7.0, device=dev))
    adapter.set_stream(prev)
    adapter.synchronize()
    ok = bool((out.cpu() == 7.0).all())
    return Result("stream_dynamic_coverage", ok,
                  f"{adapter.name}: 新建 stream 上的 work sync 后可见 -> {ok}")


def mem_query_realtime(adapter):
    """状态查询契约：实时读 + 口径一致（alloc 前后差值覆盖申请量）。"""
    hw = _hw_hooks(adapter)
    if hw is not None:
        before = adapter.memory_allocated()
        hw.alloc(256)
        after = adapter.memory_allocated()
        hw.free(256)
        ok = (after - before) == 256
        return Result("mem_query_realtime", ok,
                      f"{adapter.name}: alloc(256) 前后 {before}->{after}"
                      + "（变=未缓存，+256=口径一致）" if ok else "（口径或缓存违约）")
    import torch
    nbytes = 64 * 1024**2  # 64 MiB，远大于 allocator 粒度噪音
    before = adapter.memory_allocated()
    t = torch.empty(nbytes, dtype=torch.uint8, device=_dev(adapter))
    after = adapter.memory_allocated()
    del t
    adapter.empty_cache()
    delta = after - before
    ok = delta >= nbytes * 0.9  # caching allocator 向上取整，不苛求精确
    return Result("mem_query_realtime", ok,
                  f"{adapter.name}: tensor alloc 64MiB 前后 delta={delta}")


def graph_capture_roundtrip(adapter):
    """graph 捕获能力的**实测**：能捕获并重放即支持，捕获不了即不支持。

    不阻断激活（不在强制套件里）：不支持 graph 捕获是合法的平台状态。
    结果本身就是事实——取代了原先"平台自报 graph_capture=True 再来验证
    它有没有说谎"的两步走。
    """
    hw = _hw_hooks(adapter)
    if hw is not None:
        if not hasattr(hw, "capture"):
            return Result("graph_capture_roundtrip", True,
                          f"{adapter.name}: 不支持 graph 捕获（硬件无 capture）")
        r = hw.capture(lambda: "ok")
        return Result("graph_capture_roundtrip", r == "ok",
                      f"{adapter.name}: capture 执行 -> {r}")
    # 真实栈：用 torch 原生 graph 实测捕获+重放（判据=replay 真的执行了被
    # 捕获的 op，而非 capture 期顺带执行）。
    import torch
    mod = getattr(adapter, "mod", None)
    if not adapter.supports("Stream") or mod is None or not hasattr(mod, "CUDAGraph"):
        return Result("graph_capture_roundtrip", True,
                      f"{adapter.name}: 不支持 graph 捕获（无 CUDAGraph 或无 stream 接口）")
    # Denied 占位区分：CUDAGraph 被 TT Denied 替换后实例化抛 TTCompatError，若被
    # 下方 try/except 吞成"实测不支持"会产生误导性结论（复跑/重复激活时）——
    # 显式识别占位标记，如实说明（2026-10-22 专家审视发现）。
    if getattr(mod.CUDAGraph, "_tt_denied", False):
        return Result("graph_capture_roundtrip", True,
                      f"{adapter.name}: graph 被 TT Denied 占位（特性点未验收），"
                      f"非平台能力结论——启用需先验收转 Managed")
    try:
        x = torch.ones(8, device=_dev(adapter))
        main = adapter.current_stream()     # 捕获前的 warmup 必须在 side stream 上
        side = adapter.Stream()
        side.wait_stream(main)              # 原生 stream 的对象方法（非平台契约）
        adapter.set_stream(side)
        x.mul_(1)
        adapter.set_stream(main)
        main.wait_stream(side)
        g = mod.CUDAGraph()
        with mod.graph(g):
            x.mul_(2)
        before = x.clone()
        g.replay()
        adapter.synchronize()
        ok = bool((x == before * 2).all())
        detail = f"{adapter.name}: capture+replay 生效={ok}（replay 前 {before[0]} -> 后 {x[0]}）"
    except Exception as e:
        # 捕获失败 = 该平台不支持 graph 捕获，如实记录而非判 FAIL
        ok = True
        detail = (f"{adapter.name}: 实测不支持 graph 捕获 "
                  f"{type(e).__name__}: {e}")
    return Result("graph_capture_roundtrip", ok, detail)


def async_comm_smoke(adapter):
    """异步通信能力的**实测**：异步提交句柄可用且 wait 后完成。

    同 graph_capture_roundtrip：不阻断激活，结果即事实。
    """
    hw = _hw_hooks(adapter)
    if hw is None:
        # 真实栈的异步通信由 provider 收益门禁实测，此处不下结论
        return Result("async_comm_smoke", True,
                      f"{adapter.name}: 无 probe 钩子，留待 provider 收益门禁实测")
    hw.submit(0, "a")
    h = hw.submit_async(0, "b")
    h.wait()
    adapter.synchronize()
    left = hw.pending(0)
    return Result("async_comm_smoke", left == 0, f"{adapter.name}: 残留 {left}")


def rng_state_roundtrip(adapter):
    """RNG 契约：get->set->get 回环一致；非法值必须抛统一错误（checkpoint
    resume 正确性；xmegatron 的 XPU RNG 对齐补丁证明此路径必踩）。"""
    from ttorch.platform.platform_handle import TTPlatformError
    s = adapter.get_rng_state()
    adapter.set_rng_state(s)
    back = adapter.get_rng_state()
    # state 是 torch 原生对象（ByteTensor）时 `==` 返回逐元素 tensor，
    # 直接用作 bool 会抛 "Boolean value of Tensor is ambiguous"——
    # 契约只要求"回环相等"，比较方式随对象类型走
    ok = bool((back == s).all()) if hasattr(back, "all") else back == s
    try:
        adapter.set_rng_state(12345)  # 非法类型
        rejected = False
    except TTPlatformError as e:
        rejected = e.category == "invalid_rng_state"
    adapter.set_rng_state(s)  # 恢复
    return Result("rng_state_roundtrip", ok and rejected,
                  f"{adapter.name}: 回环一致={ok}，非法值统一拒绝={rejected}")


def stream_fencing_semantics(adapter):
    """stream fencing 契约：跨流等待确实落成依赖边（megatron ddp 侧流
    fencing 的语义基础——只做对形状不做对依赖等于没做）。"""
    hw = _hw_hooks(adapter)
    if hw is not None:
        side = adapter.Stream()
        main = adapter.current_stream()
        side.wait_stream(main)
        ok = (side, main) in adapter.hw.wait_edges
        return Result("stream_fencing_semantics", ok,
                      f"{adapter.name}: 依赖边({side},{main}) 记录={ok}")
    import torch
    dev = _dev(adapter)
    main = adapter.current_stream()
    side = adapter.Stream()
    x = torch.full((128,), 2.0, device=dev)
    y = torch.empty(128, device=dev)
    side.wait_stream(main)  # side 等 main 上的生产完成（原生对象方法）
    adapter.set_stream(side)
    y.copy_(x * 3)
    adapter.set_stream(main)
    adapter.synchronize()
    ok = bool((y.cpu() == 6.0).all())
    return Result("stream_fencing_semantics", ok,
                  f"{adapter.name}: tensor 级 fencing 校验 -> {ok}")


def stream_ctx_restore(adapter):
    """stream 上下文契约：切换后恢复，当前流读数正确。
    判等走 _stream_key（真实 torch Stream 对象每次包装可能不同）。"""
    key = _stream_key
    cur = adapter.current_stream()
    side = adapter.Stream()
    adapter.set_stream(side)
    mid = adapter.current_stream()
    adapter.set_stream(cur)
    ok = key(mid) == key(side) and key(adapter.current_stream()) == key(cur)
    return Result("stream_ctx_restore", ok,
                  f"{adapter.name}: cur={key(cur)} 切换后={key(mid)} 恢复={key(adapter.current_stream())}")


def mem_info_consistency(adapter):
    """mem_get_info 契约：(free, total) 字节元组，free+used≈total
    （容忍 1 GiB 的 allocator 保留池；单位换算错误差 1e6 倍，必然越界），
    且与 memory_allocated 口径一致。"""
    free, total = adapter.mem_get_info()
    used = adapter.memory_allocated()
    ok = isinstance(free, int) and isinstance(total, int) \
        and 0 < free <= total and abs(free + used - total) <= 1024**3
    return Result("mem_info_consistency", ok,
                  f"{adapter.name}: free={free} total={total} used={used}")


def mem_caliber_coherence(adapter):
    """内存口径**自洽性**验收（D7，2026-10-22）。

    无外部标杆——masquerade 下"真实该预留多少"无从知晓（实测：分配 16MiB 导致
    free 降 148MiB，差额构成外部不可知）。故验**自洽性/合理性**而非准确性：
    1. allocated 增量对账（唯一有标杆的）：分配已知序列，增量单调且量级精确
       （允许 allocator 粒度向上取整，不允许少于分配量或量级偏差）；
    2. 不变量：0 <= allocated <= reserved（caching allocator 语义，若有 reserved）；
    3. 释放回落：del + empty_cache 后 allocated 回基线附近（验统计无泄漏）。

    口径画像（reserved/allocated 关系、free/total 构成）记 provenance——不判对错
    （无标杆），让口径差异可见，供上层策略（ZeRO offload 阈值、容量规划）参考。

    判据刻意宽松（宁放过不误杀）：本测试在 fail-fast 强制套件里，误杀会拒绝激活。

    边界纪律（2026-10-22 修正）：本测试**不做 reset 语义验证**——
    `reset_peak_memory_stats()` 清峰值统计且不可恢复，而 activate() 是公开 API、
    允许训练中途显式调用，激活期执行它会清掉框架已累积的真实峰值。reset 语义
    挪到 witness 按需诊断（`witness/diagnostics.mem_reset_semantics`，用户知情
    主动调用）。激活期验证不破坏被观测对象的状态。
    """
    import torch
    from ttorch.witness import provenance
    hw = _hw_hooks(adapter)
    mb = 1024**2
    if hw is not None:
        base = adapter.memory_allocated()
        hw.alloc(64)
        after = adapter.memory_allocated()
        hw.free(64)
        ok = (after - base) == 64
        return Result("mem_caliber_coherence", ok,
                      f"{adapter.name}: alloc(64) 增量={after-base}（精确=口径一致）")
    dev = _dev(adapter)
    base_alloc = adapter.memory_allocated()
    xs, deltas = [], []
    seq = (16, 32, 64)
    for m in seq:
        xs.append(torch.zeros(m * mb, dtype=torch.uint8, device=dev))
        deltas.append(adapter.memory_allocated() - base_alloc)
    total_expect = sum(seq) * mb
    monotone = deltas[0] <= deltas[1] <= deltas[2]           # 增量单调不减
    precise = 0.9 * total_expect <= deltas[-1] <= 3 * total_expect  # 量级精确（容忍粒度/预留）
    alloc_now = adapter.memory_allocated()
    reserved = adapter.memory_reserved() if adapter.supports("memory_reserved") else None
    invariant = alloc_now >= 0 and (reserved is None or alloc_now <= reserved)
    del xs
    if adapter.supports("empty_cache"):
        adapter.empty_cache()
    fall_alloc = adapter.memory_allocated()
    fall = fall_alloc <= base_alloc + 1 * mb                 # 释放回落（允许小残留）
    # 口径画像：不判对错，记录供上层参考。free/total 依赖 mem_get_info（可选接口）
    free, total_b = (adapter.mem_get_info()
                     if adapter.supports("mem_get_info") else (None, None))
    provenance.record(provenance.LAYER_PLATFORM, "caliber", "memory_profile",
                      "observed", allocated=int(alloc_now),
                      reserved=(int(reserved) if reserved is not None else None),
                      free=(int(free) if free is not None else None),
                      total=(int(total_b) if total_b is not None else None),
                      alloc_delta_total=int(deltas[-1]),
                      alloc_expect_total=int(total_expect))
    ok = monotone and precise and invariant and fall
    return Result("mem_caliber_coherence", ok,
                  f"{adapter.name}: 增量{[d // mb for d in deltas]}MiB(期望{sum(seq)}) "
                  f"单调={monotone} 量级={precise} alloc<=reserved={invariant} 回落={fall}")


def device_identity_consumable(adapter):
    """设备标识契约：current_device 的返回值能回喂 set_device——
    swift 6+ 处把它直接传给 torch.tensor(device=)，demo 用回喂模拟。
    （框架侧构造 torch.device 应走 device_spec 门面，见 real smoke。）"""
    idx = adapter.current_device()
    try:
        adapter.set_device(idx)
        ok = adapter.current_device() == idx
    except Exception:
        ok = False
    return Result("device_identity_consumable", ok,
                  f"{adapter.name}: current_device={idx} 回喂 set_device 成功={ok}")


# 强制契约测试：所有 adapter 必过（平台未实现所需接口时自动跳过）
_MANDATORY = [sync_scope_device_wide, stream_dynamic_coverage,
             mem_query_realtime,
             rng_state_roundtrip, stream_fencing_semantics,
             stream_ctx_restore, mem_info_consistency,
             device_identity_consumable, mem_caliber_coherence]

# 各强制测试所需的平台接口：平台没实现就跳过该测试（2026-10-14 取代原
# _REQUIRED_CAP 的能力声明裁剪——裁剪依据从"平台自报 cap=False"改成
# "实现里有没有这个方法"，事实派生自实现，不经自报）。
_REQUIRED_API = {
    "sync_scope_device_wide": "Stream",
    "stream_dynamic_coverage": "Stream",
    "stream_fencing_semantics": "Stream",
    "stream_ctx_restore": "Stream",
    "mem_query_realtime": "memory_allocated",
    "mem_info_consistency": "mem_get_info",
    "mem_caliber_coherence": "memory_allocated",
    "rng_state_roundtrip": "get_rng_state",
}

# 可选能力的背书测试：**无条件跑**，结果即事实（2026-10-14 取代原 _BACKING
# 的"声明 True 才跑"——不再有自报的 graph_capture / async_comm 可信问题，
# 跑不通就是不支持。这两项不进强制套件：不支持是合法的，不该拒绝激活）。
_ENDORSEMENT = [graph_capture_roundtrip, async_comm_smoke]


def _run_one(t, adapter):
    """单个测试的异常托底：测试内异常转成 Result，不让它崩掉整个激活流程。

    _REQUIRED_API 裁剪只覆盖每个测试的主接口，测试内部可能触及**其他可选接口**
    （如 mem_caliber_coherence 用到 mem_get_info）。此前测试内的 TTPlatformError
    会直接冒泡崩掉激活（2026-10-22 专家审视发现）。区分：
    - TTPlatformError("unsupported")：触及平台未实现接口 → 合法降级 skip；
    - 其他异常：平台违约或测试缺陷 → FAIL（fail fast 走正常 Result 通道）。
    """
    from ttorch.platform.platform_handle import TTPlatformError
    try:
        return t(adapter)
    except TTPlatformError as e:
        if e.category == "unsupported":
            return Result(t.__name__, True,
                          f"skipped: 测试触及未实现接口（{e}），合法降级")
        return Result(t.__name__, False, f"平台错误[{e.category}]: {e}")
    except Exception as e:
        return Result(t.__name__, False, f"测试异常 {type(e).__name__}: {e}")


def run(adapter):
    """跑强制套件（按实现裁剪）+ 可选能力背书（实测产出事实，不阻断激活）。

    裁剪与失败的区分：平台**没实现**该接口 -> 跳过（合法的能力裁剪，框架
    会收到 unsupported 走降级）；实现了但**没过契约** -> FAIL（拒绝激活）。
    """
    results = []
    for t in _MANDATORY:
        api = _REQUIRED_API.get(t.__name__)
        if api and not adapter.supports(api):
            results.append(Result(t.__name__, True,
                                  f"skipped: 平台未实现 {api}()（调用方收 unsupported，走降级）"))
            continue
        results.append(_run_one(t, adapter))
    for t in _ENDORSEMENT:
        results.append(_run_one(t, adapter))
    return results
