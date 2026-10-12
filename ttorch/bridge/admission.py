"""准入测试（admission）：只验"绑定正确性"——框架侧符号替换后语义保持。

与 conformance 的分层：
- admission（本模块，cuda-compat 侧）验绑定动作本身：符号是否绑到预期
  目标、两种引用观测是否一致、替换后是否可达硬件；
- conformance（platform 侧）验 adapter 契约履行：sync 等待范围、状态
  查询口径、capability 声明真实性——激活期强制执行，见 conformance.py。

不再有字符串 id 注册表——引用即对象，写错名字 import 期就炸。
每个测试返回 (ok, detail)。
"""
from ttorch.bridge import platform_mgr


def set_device_equivalence(ctx):
    dev = ctx["framework"].init_device(1)
    ok = dev == 1 and platform_mgr.current_device() == 1
    return ok, f"绑定后 current_device={dev}，两种引用观测一致"


def current_device_consistency(ctx):
    v = ctx["vendor_cuda"].current_device()
    u = platform_mgr.current_device()
    return v == u, f"框架侧读到 {v}，门面读到 {u}"


def annotate_delivered(ctx):
    hw = platform_mgr.active().hw
    platform_mgr.range_push("__probe__")
    ok = "__probe__" in hw.marks
    return ok, f"range_push 落到硬件: {hw.marks}"


def sync_binding(ctx):
    ok = ctx["vendor_cuda"].synchronize is platform_mgr.synchronize
    return ok, f"vendor_cuda.synchronize 绑到门面 -> {ok}"


def mem_binding(ctx):
    ok = ctx["vendor_cuda"].memory_allocated is platform_mgr.memory_allocated
    return ok, f"vendor_cuda.memory_allocated 绑到门面 -> {ok}"


def mem_info_binding(ctx):
    ok = ctx["vendor_cuda"].mem_get_info is platform_mgr.mem_get_info
    return ok, f"vendor_cuda.mem_get_info 绑到门面 -> {ok}"


def rng_binding(ctx):
    vc = ctx["vendor_cuda"]
    ok = (vc.get_rng_state is platform_mgr.get_rng_state
          and vc.set_rng_state is platform_mgr.set_rng_state)
    return ok, f"get/set_rng_state 绑到门面 -> {ok}"


def empty_cache_binding(ctx):
    ok = ctx["vendor_cuda"].empty_cache is platform_mgr.empty_cache
    return ok, f"vendor_cuda.empty_cache 绑到门面 -> {ok}"


def stream_binding(ctx):
    vc = ctx["vendor_cuda"]
    ok = (vc.Stream is platform_mgr.Stream
          and vc.stream is platform_mgr.stream
          and vc.current_stream is platform_mgr.current_stream)
    return ok, f"Stream/stream/current_stream 绑到门面 -> {ok}"


def vendor_internal_construction(ctx):
    """G14：绑定的爆炸半径——被绑符号所在模块的**原实现内部**也会引用
    同模块符号，必须仍拿到厂商原生对象。

    `torch.cuda.current_stream` 的全局名空间就是被我们改过的
    `torch.cuda.__dict__`，其内部 `Stream(stream_id=...)` 会落到门面；
    门面若不分流，返回的 TTStream 会被当作原生 stream 继续流进厂商栈
    （实测表现：`'TTStream' object has no attribute 'record_event'`）。

    判据：经门面取当前流，其 .sid 必须是厂商原生类型而非 TTStream。
    """
    from ttorch.bridge.platform_adapter import TTStream
    if not platform_mgr.capability("stream_api"):
        return True, "平台声明 stream_api=False，无 stream 对象族"
    sid = platform_mgr.current_stream().sid
    ok = not isinstance(sid, TTStream)
    return ok, (f"原实现内部构造回落厂商原生对象 -> {type(sid).__name__}"
                f"（非 TTStream={ok}）")


def make_cap_binding_test(attr, key):
    """能力类绑定正确性：框架侧符号读到的是 capability 缓存里的值。"""
    def cap_binding(ctx):
        v = getattr(ctx["vendor_cuda"], attr)()
        c = platform_mgr.capability(key)
        return v == c, f"{attr}()={v}，capability[{key}]={c}"
    cap_binding.__name__ = f"cap_binding_{key}"
    return cap_binding
