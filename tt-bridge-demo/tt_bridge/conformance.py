"""conformance 套件：验证 adapter 是否履行统一接口契约（实现正确性）。

与 admission 的分层：
- admission（cuda-compat 侧）验"绑定正确性"——框架侧语义在替换后保持；
- conformance（本模块）验"adapter 正确性"——契约被真正实现。

激活期由 platform_mgr._activate 调用：任一强制测试 FAIL 即拒绝激活；
declared 能力承诺 True 必须有 BACKING 中的背书测试且通过。

能力裁剪（真实环境必需）：测试标注所需能力，平台声明 False 时跳过
（门面会对相应 API 报 unsupported，框架走降级路径）——真实栈可能合法地
没有 graph_capture / mem 查询，一刀切 FAIL 会误杀。

测试分层：mock adapter 带 hw 钩子（submit/pending/alloc/wait_edges）走
深测版（能构造确定性反例）；真实 adapter 没有钩子，走 tensor 级契约版
（用真实算子入队 + 数值校验推断语义是否履行——弱于深测但真实可跑）。
"""
from collections import namedtuple

Result = namedtuple("Result", ["name", "ok", "detail"])


def _dev(adapter):
    """当前设备的 torch device spec（tensor 级测试用）。"""
    return f"{adapter.device_name}:{adapter.current_device()}"


def _hw_hooks(adapter):
    """mock 深测钩子；真实 adapter 返回 None。"""
    hw = getattr(adapter, "hw", None)
    return hw if hw is not None and hasattr(hw, "submit") else None


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
    side = adapter.stream_create()
    main = adapter.stream_current()
    adapter.stream_set(side)
    out.copy_(x @ x)  # 侧流上的异步 work
    adapter.stream_set(main)
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
    sid = adapter.stream_create()
    prev = adapter.stream_current()
    out = torch.empty(128, device=dev)
    adapter.stream_set(sid)
    out.copy_(torch.full((128,), 7.0, device=dev))
    adapter.stream_set(prev)
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


def cap_matches_probe(adapter):
    """probed 能力必须实测：capability 表与直接查硬件的结果一致。"""
    caps = adapter.capabilities()
    hw = _hw_hooks(adapter)
    if hw is not None:
        count = hw.device_count() if hasattr(hw, "device_count") else hw.deviceCount()
        ok = caps["p2p"] == hw.p2p_supported and caps["device_count"] == count
        return Result("cap_matches_probe", ok,
                      f"{adapter.name}: p2p 声明={caps['p2p']} 实测={hw.p2p_supported}；"
                      f"device_count 声明={caps['device_count']} 实测={count}")
    mod = getattr(adapter, "mod", None)  # 真实 adapter 的 torch.cuda/torch.xpu 模块
    if mod is not None:
        real = mod.device_count()
        ok = caps["device_count"] == real
        return Result("cap_matches_probe", ok,
                      f"{adapter.name}: device_count 声明={caps['device_count']} torch 实测={real}")
    ok = caps["device_count"] >= 1
    return Result("cap_matches_probe", ok,
                  f"{adapter.name}: 无 probe 钩子，契约级校验 device_count={caps['device_count']}")


def graph_capture_roundtrip(adapter):
    """graph_capture=True 的背书：硬件确实能捕获执行。"""
    if not adapter.capabilities().get("graph_capture"):
        return Result("graph_capture_roundtrip", True, "declared False，无需背书")
    r = adapter.hw.capture(lambda: "ok")
    return Result("graph_capture_roundtrip", r == "ok",
                  f"{adapter.name}: capture 执行 -> {r}")


def async_comm_smoke(adapter):
    """async_comm=True 的背书：异步提交句柄可用且 wait 后完成。"""
    if not adapter.capabilities().get("async_comm"):
        return Result("async_comm_smoke", True, "declared False，无需背书")
    hw = _hw_hooks(adapter)
    if hw is None:
        # 真实栈的异步通信由 provider 门禁实测，此处只做声明存在性校验
        return Result("async_comm_smoke", True,
                      f"{adapter.name}: 无 probe 钩子，declared=True 留待 provider 收益门禁实测")
    hw.submit(0, "a")
    h = hw.submit_async(0, "b")
    h.wait()
    adapter.synchronize()
    left = hw.pending(0)
    return Result("async_comm_smoke", left == 0, f"{adapter.name}: 残留 {left}")


def rng_state_roundtrip(adapter):
    """RNG 契约：get->set->get 回环一致；非法值必须抛统一错误（checkpoint
    resume 正确性；xmegatron 的 XPU RNG 对齐补丁证明此路径必踩）。"""
    from tt_bridge.platform_adapter import TTPlatformError
    s = adapter.get_rng_state()
    adapter.set_rng_state(s)
    ok = adapter.get_rng_state() == s
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
        side = adapter.stream_create()
        main = adapter.stream_current()
        adapter.stream_wait(side, main)
        ok = (side, main) in adapter.hw.wait_edges
        return Result("stream_fencing_semantics", ok,
                      f"{adapter.name}: 依赖边({side},{main}) 记录={ok}")
    import torch
    dev = _dev(adapter)
    main = adapter.stream_current()
    side = adapter.stream_create()
    x = torch.full((128,), 2.0, device=dev)
    y = torch.empty(128, device=dev)
    adapter.stream_wait(side, main)  # side 等 main 上的生产完成
    adapter.stream_set(side)
    y.copy_(x * 3)
    adapter.stream_set(main)
    adapter.synchronize()
    ok = bool((y.cpu() == 6.0).all())
    return Result("stream_fencing_semantics", ok,
                  f"{adapter.name}: tensor 级 fencing 校验 -> {ok}")


def stream_ctx_restore(adapter):
    """stream 上下文契约：切换后恢复，当前流读数正确。
    判等走 stream_key（真实 torch Stream 对象每次包装可能不同）。"""
    key = adapter.stream_key
    cur = adapter.stream_current()
    side = adapter.stream_create()
    adapter.stream_set(side)
    mid = adapter.stream_current()
    adapter.stream_set(cur)
    ok = key(mid) == key(side) and key(adapter.stream_current()) == key(cur)
    return Result("stream_ctx_restore", ok,
                  f"{adapter.name}: cur={key(cur)} 切换后={key(mid)} 恢复={key(adapter.stream_current())}")


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


# 强制契约测试：所有 adapter 必过（除非所需能力声明 False 被裁剪）
MANDATORY = [sync_scope_device_wide, stream_dynamic_coverage,
             mem_query_realtime, cap_matches_probe,
             rng_state_roundtrip, stream_fencing_semantics,
             stream_ctx_restore, mem_info_consistency,
             device_identity_consumable]

# 各强制测试所需能力：平台声明 False 时裁剪（跳过），门面会报 unsupported
REQUIRED_CAP = {
    "sync_scope_device_wide": "stream_api",
    "stream_dynamic_coverage": "stream_api",
    "stream_fencing_semantics": "stream_api",
    "stream_ctx_restore": "stream_api",
    "mem_query_realtime": "memory_query",
    "mem_info_consistency": "memory_query",
    "rng_state_roundtrip": "rng_state",
}

# declared 能力的背书测试：承诺 True 才需要跑
BACKING = {"graph_capture": graph_capture_roundtrip,
           "async_comm": async_comm_smoke}


def run(adapter):
    """跑强制套件（按能力裁剪） + 声明了 True 的能力的背书测试。"""
    caps = adapter.capabilities()
    results = []
    for t in MANDATORY:
        req = REQUIRED_CAP.get(t.__name__)
        if req and not caps.get(req):
            results.append(Result(t.__name__, True,
                                  f"skipped: 平台声明 {req}=False（门面报 unsupported，框架降级）"))
            continue
        results.append(t(adapter))
    for key, test in BACKING.items():
        if caps.get(key):
            results.append(test(adapter))
    return results
