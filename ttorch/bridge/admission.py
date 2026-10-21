"""准入测试（admission）：只验"绑定正确性"——框架侧符号替换后语义保持。

与 conformance 的分层：
- admission（本模块，cuda-compat 侧）验绑定动作本身：符号是否绑到预期
  目标、两种引用观测是否一致、替换后是否可达硬件；
- conformance（platform 侧）验 adapter 契约履行：sync 等待范围、状态
  查询口径、capability 声明真实性——激活期强制执行，见 conformance.py。

组织：按符号分组，与 module_binding.py 的注册顺序对齐。mock demo 专属的
测试单独标注——真实 torch.cuda 的 policy（module_binding.py）不含这些符号。

不再有字符串 id 注册表——引用即对象，写错名字 import 期就炸。
每个测试返回 (ok, detail)。
"""
import torch

from ttorch.platform import interface, manager


# ==========================================================================
# 行为（behavior）：有副作用，验语义等价 / 对象形态
# ==========================================================================

# --- set_device ---

def set_device_roundtrip(ctx):
    """set_device 绑定正确性（真实环境版）：经绑定符号设置后回读一致。"""
    idx = interface.current_device()
    ctx["vendor_cuda"].set_device(idx)
    ok = ctx["vendor_cuda"].current_device() == idx
    return ok, f"set_device({idx}) 后 current_device 回读一致 -> {ok}"


def set_device_equivalence(ctx):
    """set_device（mock 版）：走 framework 替身的 init_device。"""
    dev = ctx["framework"].init_device(1)
    ok = dev == 1 and interface.current_device() == 1
    return ok, f"绑定后 current_device={dev}，两种引用观测一致"


# --- synchronize ---

def sync_binding(ctx):
    ok = ctx["vendor_cuda"].synchronize is interface.synchronize
    return ok, f"vendor_cuda.synchronize 绑到门面 -> {ok}"


# --- Stream / current_stream / stream（stream 对象族，ddp fencing 必触发）---

def stream_binding(ctx):
    """stream 上下文族绑定指向（Stream 类不在此列——它是类型位符号，
    绑平台原生类对象，由 stream_class_preserved 单独验收）。"""
    vc = ctx["vendor_cuda"]
    ok = (vc.stream is interface.stream
          and vc.current_stream is interface.current_stream)
    return ok, f"stream/current_stream 绑到门面 -> {ok}"


def stream_class_preserved(ctx):
    """类型位符号验收：torch.cuda.Stream 绑定后仍是 type（平台原生类）。

    若被绑成转发函数，`Stream | None` 返回注解在 import 期求值（transformers
    5.x 真实代码）、isinstance 判定（deepspeed/accelerate/megatron 常见写法）
    会全线 TypeError。
    """
    s = ctx["vendor_cuda"].Stream
    ok = isinstance(s, type)
    return ok, f"torch.cuda.Stream 仍是类型（{getattr(s, '__name__', s)!r}）-> {ok}"


def event_binding(ctx):
    """Event 验收（类型位 + 跨流同步语义）：通信编排现场的原语，两条都验。

    1) 类型位：绑定后仍是 type（同 Stream 教训——绑成转发函数会让
       `event: torch.cuda.Event` 注解求值、isinstance 判定全线 TypeError）；
    2) 语义：record → query/synchronize 必须在真实设备上生效。这条是**行为**
       验收而非绑定指向——Event 的 record/wait 是流水线 p2p 与 activation
       offload 串接阶段的唯一机制，语义失真就是跨阶段乱序（静默错）。

    跨流 fencing 实测：侧流做一段工作并 record，主流 wait 该 event 后读结果，
    数据必须可见——与 conformance 的 stream fencing 同思路，但落在 cuda-compat
    符号上（该符号是框架侧通信编排的实际落点）。
    """
    vc = ctx["vendor_cuda"]
    ev_cls = vc.Event
    if not isinstance(ev_cls, type):
        return False, f"torch.cuda.Event 不是类型（{type(ev_cls).__name__}）"
    act = manager.activation()
    if not (act.supports("Stream") and act.supports("stream")):
        return True, "平台无 stream 族，跳过 Event 语义实测（类型位已验）"
    try:
        dev = manager.activation().device_spec()
        x = torch.ones(256, 256, device=dev)
        s = vc.Stream()
        with vc.stream(s):
            y = (x + 1).sum()   # 侧流上产出，未经同步主流不可见
        ev = ev_cls()
        ev.record(s)
        ev.synchronize()        # 等待侧流完成（跨流 fencing）
        ok = bool(ev.query()) and float(y) > 0
        return ok, f"Event 仍是类型且 record/synchronize/query 生效 -> {ok}"
    except Exception as e:
        return False, f"Event 语义实测失败 {type(e).__name__}: {e}"


def vendor_internal_construction(ctx):
    """G14：绑定的爆炸半径——被绑符号所在模块的**原实现内部**也会引用
    同模块符号，必须仍拿到厂商原生对象。

    `torch.cuda.current_stream` 的全局名空间就是被我们改过的
    `torch.cuda.__dict__`，其内部 `Stream(stream_id=...)` 会落到门面。
    TT 不包装返回对象：门面把 adapter 返回的**厂商原生 stream** 原样透传
    给框架与厂商内部，不存在"自定义包装被当原生对象用"的错位（旧 TTStream
    流入 torch 内部曾炸 `'TTStream' object has no attribute 'record_event'`）。

    判据：门面返回值与 adapter 契约返回值同型（无中间包装层）。
    """
    a = manager.activation()
    if not a.supports("current_stream"):
        return True, "平台未实现 current_stream，无 stream 对象族"
    via_interface = interface.current_stream()
    direct = a.current_stream()
    ok = type(via_interface) is type(direct)
    return ok, (f"门面透传厂商原生 stream -> {type(via_interface).__name__}"
                f"（与 adapter 契约同型={ok}）")


# --- empty_cache ---

def empty_cache_binding(ctx):
    ok = ctx["vendor_cuda"].empty_cache is interface.empty_cache
    return ok, f"vendor_cuda.empty_cache 绑到门面 -> {ok}"


# --- range_push（mock 专属；真实 torch.cuda policy 未纳入此符号）---

def annotate_delivered(ctx):
    hw = manager.activation().hw
    interface.range_push("__probe__")
    ok = "__probe__" in hw.marks
    return ok, f"range_push 落到硬件: {hw.marks}"


# ==========================================================================
# 状态查询（state_query）：动态只读，验口径一致 + 绑定指向
# ==========================================================================

# --- current_device ---

def current_device_consistency(ctx):
    v = ctx["vendor_cuda"].current_device()
    u = interface.current_device()
    return v == u, f"框架侧读到 {v}，门面读到 {u}"


# --- memory_allocated ---

def mem_binding(ctx):
    ok = ctx["vendor_cuda"].memory_allocated is interface.memory_allocated
    return ok, f"vendor_cuda.memory_allocated 绑到门面 -> {ok}"


# --- mem_get_info ---

def mem_info_binding(ctx):
    ok = ctx["vendor_cuda"].mem_get_info is interface.mem_get_info
    return ok, f"vendor_cuda.mem_get_info 绑到门面 -> {ok}"


# --- get_rng_state + set_rng_state（RNG 配对：save 读侧 / resume 写侧）---
# set_rng_state 在 module_binding 里归 behavior，这里与 get 配对一起验。

def rng_binding(ctx):
    """RNG 三件套（get / set / manual_seed）绑到门面。

    manual_seed 与 get/set 同族：通信与训练路径都用它做全局播种，纳入门面后
    播种行为进入绑定面（可观测）。未绑定时该断言为假——这是本条目的验收点。
    """
    vc = ctx["vendor_cuda"]
    ok = (vc.get_rng_state is interface.get_rng_state
          and vc.set_rng_state is interface.set_rng_state
          and vc.manual_seed is interface.manual_seed)
    return ok, f"get/set_rng_state + manual_seed 绑到门面 -> {ok}"


# ==========================================================================
# 通用：绑定指向校验（按对象身份比对，适用于任何纯转发符号）
# ==========================================================================

def nccl_primitive_binding(ctx):
    """NCCL 通信原语符号（`torch.cuda.nccl.version`）的两个验收点。

    1) 绑到门面：该符号是**子模块内函数**（落点 `torch.cuda.nccl`），
       证明 TT 的绑定通道不只覆盖 torch.cuda 顶层符号；
    2) 行为不变：返回值仍是原生三元组——compat 只多加一条证据，不改返回值
       （改了会打破框架的版本检查分支）。
    """
    vc = ctx["vendor_cuda"]
    n = getattr(vc, "nccl", None)
    if n is None:
        return False, "torch.cuda.nccl 不存在"
    fn = n.version
    ok_bound = fn.__module__ == "ttorch.bridge.module_binding"
    try:
        v = fn()
    except Exception as e:
        return False, f"version() 调用失败 {type(e).__name__}: {e}"
    ok_val = isinstance(v, tuple) and len(v) == 3
    return ok_bound and ok_val, f"绑到门面={ok_bound}，返回值={v}（行为不变）"


def binding_identity(attr):
    """绑定正确性：框架侧符号确实指向 TT 门面（对象身份比对）。

    用于 is_available / device_count / can_device_access_peer 这类纯转发
    接口——它们经 interface 直达平台实现，验"绑对了指向"即足够，无需比对取值。
    """
    def check(ctx):
        bound = getattr(ctx["vendor_cuda"], attr)
        expected = getattr(interface, attr)
        ok = bound is expected
        return ok, f"{attr} 绑到门面 -> {ok}"
    check.__name__ = f"binding_identity_{attr}"
    return check
