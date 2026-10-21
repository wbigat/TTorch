"""torch.cuda 的符号绑定策略表（与 demo 侧 mock 盘点同构）。

每条 Entry 把一个 torch.cuda 符号绑定到 interface 的转发函数；activate 时由
cuda_compat 逐个 setattr 到 torch.cuda 上。设计要点：

- 落点必须是框架可见路径：这些符号定义在 torch.cuda.memory / streams /
  random 等子模块再 re-export 到 torch.cuda，框架访问的是 torch.cuda.X，
  所以每条显式 bind_module="torch.cuda"——只改子模块，框架侧仍指原实现；
- default_deny=False：torch.cuda.__all__ 里含 torch 自身内部要用的符号，
  白名单只管框架调用面，不接管 torch 自身；
- impl 签名对齐 torch 真实签名（含 device 可选参），带参调用点不破；
- 准入测试复用 admission.py 的 ctx 驱动条目（ctx["vendor_cuda"]=torch.cuda）；
  hw 钩子类测试（annotate_delivered）是 mock 专属，不在此注册。
"""
import torch.cuda

from ttorch.bridge import admission
from ttorch.platform import interface
from ttorch.bridge.binding_registry import (
    Decision, Entry, get_entries, register_binding)

R = Decision.REMAP
M = "torch.cuda"  # 绑定落点：框架可见的 re-export 路径


def _native_type_class(entry, original):
    """类型位符号（Stream / Event）：bind 期解析为平台原生**类对象**。

    不能绑转发函数（`type(torch.cuda.Stream)` 变成 function 后，`Stream | None`
    返回注解在 import 期求值、isinstance 判定全线 TypeError——transformers 5.x
    直接导不进来）。现有平台的正确答案都是原生类本身：masquerade/cuda 上原生
    类即正确（G14：torch 内部 `Stream(stream_id=...)` 构造也经它完成）；
    cpu/xpu-native 上框架不走这两个符号，保持原生不接管语义。

    Event 与 Stream 同族（通信编排现场实证：megatron 流水线 p2p / activation
    offload 有 38 处 `torch.cuda.Event` 调用，record/wait 做跨阶段同步；框架还
    会写 `event: torch.cuda.Event` 注解、对返回值做 isinstance 判定）。
    """
    if not isinstance(original, type):
        raise TypeError(f"{entry.dotted}: 绑定前值不是类型（{type(original).__name__}），"
                        f"类型位符号拒绝接管")
    return original

_native_stream_class = _native_type_class  # 存量名字保留（文档与注释引用）


def _comm_version_factory(entry, original):
    """`torch.cuda.nccl.version` —— NCCL 通信原语的 compat 实例（2026-10-20）。

    这是"框架绕过 c10d ProcessGroup 直调 NCCL 原语"的真实场景（实证：
    `deepspeed/accelerator/cuda_accelerator.py:71` 的
    `communication_backend_version()` 直接调它报告后端版本）。KL3 实测：
    `torch.cuda.nccl` 仍是 **torch 原生模块**（厂商未替换），`version()`
    返回编译期链接的 NCCL 版本 `(2, 27, 5)`——而实际通信跑的是 XCCL。

    处置三原则（不伪造、不阻断、留证据）：
    - 不伪造：厂商栈未暴露 XCCL 版本 API，TT 不编一个版本号出来；
    - 不阻断：原样回落原生实现，框架行为不变（改返回值或抛错都会打破
      框架的版本检查逻辑）；
    - 留证据：首次查询记一条 provenance，把"这个数字在 masquerade 平台
      上不可信"变成可见事实，供版本分支误判时溯源。

    机制价值：本条同时示范了**子模块内函数符号**的绑法（symbol 是
    `torch.cuda.nccl.version`，落点模块即 `torch.cuda.nccl`）与
    `impl_factory` 捕获原生实现避免门面递归的用法。
    """
    from ttorch.witness import provenance
    state = {"logged": False}

    def impl():
        v = original()
        if not state["logged"]:
            state["logged"] = True
            provenance.record(provenance.LAYER_BRIDGE, "bind",
                              "torch.cuda.nccl.version", "unverified",
                              value=repr(v),
                              detail="NCCL 原语模块版本号：masquerade 平台下该值为 "
                                     "torch 编译期链接版本，可能不是当前实际通信后端")
        return v

    impl.__name__ = "version"
    return impl


# --- 设备 · 同步 · 流 · RNG·显存写 ---

register_binding(Entry(symbol=torch.cuda.set_device, decision=R,
               impl=interface.set_device, test=admission.set_device_roundtrip,
               intent="把当前进程绑定到一张加速设备", bind_module=M))

register_binding(Entry(symbol=torch.cuda.synchronize, decision=R,
               impl=interface.synchronize, test=admission.sync_binding,
               intent="device-wide 等待；torch 原生语义与契约一致", bind_module=M))

register_binding(Entry(symbol=torch.cuda.Stream, decision=R,
               impl_factory=_native_stream_class, test=admission.stream_class_preserved,
               intent="stream 类（类型位符号）：绑平台原生类对象而非转发函数——"
                      "注解求值/isinstance 依赖它是 type；G14 的内部构造经原生类完成",
               bind_module=M))
register_binding(Entry(symbol=torch.cuda.current_stream, decision=R,
               impl=interface.current_stream, test=admission.vendor_internal_construction,
               intent="当前 stream 句柄（动态只读）；G14 验收：门面透传原生 stream "
                      "对象同型，无中间包装层", bind_module=M))
register_binding(Entry(symbol=torch.cuda.set_stream, decision=R,
               impl=interface.set_stream, test=admission.binding_identity("set_stream"),
               intent="切换当前 stream；契约（TORCH_API）含此符号 policy 就必须管控，"
                      "否则框架调用绕过门面直达厂商实现——不在链路上、diagnostics 不可见",
               bind_module=M))
register_binding(Entry(symbol=torch.cuda.stream, decision=R,
               impl=interface.stream, test=admission.stream_binding,
               intent="stream 上下文管理器", bind_module=M))
register_binding(Entry(symbol=torch.cuda.Event, decision=R,
               impl_factory=_native_type_class, test=admission.event_binding,
               intent="event 类（类型位符号）：通信编排现场的跨流同步原语——"
                      "流水线 p2p / activation offload 用 record+wait 串接阶段（megatron "
                      "实测 38 处）。同 Stream 教训：必须绑原生类对象，类型注解求值/"
                      "isinstance 依赖它是 type",
               bind_module=M))

register_binding(Entry(symbol=torch.cuda.manual_seed, decision=R,
               impl=interface.manual_seed, test=admission.rng_binding,
               intent="全局 RNG 播种；与 get/set_rng_state 同族，纳入门面后播种"
                      "行为进入绑定面（通信/训练路径 megatron 实测 4 处）",
               bind_module=M))
register_binding(Entry(symbol=torch.cuda.set_rng_state, decision=R,
               impl=interface.set_rng_state, test=admission.rng_binding,
               intent="checkpoint resume 恢复 RNG；torch 原生 ByteTensor 契约", bind_module=M))
register_binding(Entry(symbol=torch.cuda.empty_cache, decision=R,
               impl=interface.empty_cache, test=admission.empty_cache_binding,
               intent="allocator 提示；尽力语义", bind_module=M))

# --- 只读查询（设备 / 显存 / RNG） ---

register_binding(Entry(symbol=torch.cuda.current_device, decision=R,
               impl=interface.current_device, test=admission.current_device_consistency,
               intent="当前设备序号；构造 torch.device 请走 device_spec 门面",
               bind_module=M))
register_binding(Entry(symbol=torch.cuda.memory_allocated, decision=R,
               impl=interface.memory_allocated, test=admission.mem_binding,
               intent="显存水位（禁缓存）；平台未实现时抛 unsupported",
               bind_module=M))
register_binding(Entry(symbol=torch.cuda.mem_get_info, decision=R,
               impl=interface.mem_get_info, test=admission.mem_info_binding,
               intent="(free, total) 字节元组", bind_module=M))
register_binding(Entry(symbol=torch.cuda.get_rng_state, decision=R,
               impl=interface.get_rng_state, test=admission.rng_binding,
               intent="checkpoint save 读 RNG（torch 原生 ByteTensor会传给 Generator.set_state）", bind_module=M))

# --- 硬件存在性 / 规模 / 拓扑 ---

register_binding(Entry(symbol=torch.cuda.is_available, decision=R,
               impl=interface.is_available, test=admission.binding_identity("is_available"),
               intent="分支选择器；masquerade 语义由双身份兜底", bind_module=M))
register_binding(Entry(symbol=torch.cuda.device_count, decision=R,
               impl=interface.device_count, test=admission.binding_identity("device_count"),
               intent="可见设备数", bind_module=M))
register_binding(Entry(symbol=torch.cuda.can_device_access_peer, decision=R,
               impl=interface.can_device_access_peer,
               test=admission.binding_identity("can_device_access_peer"),
               intent="两卡间能否 P2P（逐对回答，转发厂商原生实现）", bind_module=M))

# --- NCCL 通信原语（子模块符号：框架绕过 c10d 直调的实例）---

register_binding(Entry(symbol=torch.cuda.nccl.version, decision=R,
               impl_factory=_comm_version_factory,
               test=admission.nccl_primitive_binding,
               intent="NCCL 通信原语版本查询（deepseed 等直调 torch.cuda.nccl）；"
                      "masquerade 下该值是编译期链接版本而非实际后端，TT 不伪造"
                      "不阻断，回落原生并记一条 unverified 证据",
               bind_module="torch.cuda.nccl"))


_CUDA_NS = "torch.cuda"  # 落点命名空间：含子模块（如 torch.cuda.nccl 的通信原语）


def entries():
    """本文件的条目（binding_registry 全局共享，按落点命名空间过滤）。

    用命名空间前缀而非精确模块名：NCCL 原语这类符号的落点是子模块
    `torch.cuda.nccl`（框架直调 `torch.cuda.nccl.version()`），精确过滤会
    漏掉它们。全局注册表里还有 mock demo 的 `vendor_cuda` 条目，
    startswith("torch.cuda") 天然把它们排除。
    """
    return [e for e in get_entries()
            if _in_cuda_ns(e.bind_module or e.symbol.__module__)]


def _in_cuda_ns(module_name):
    """落点是否属于 torch.cuda 命名空间（精确匹配或子模块）。

    精确判等 + `xxx.` 前缀：裸 `startswith("torch.cuda")` 会误匹配
    `torch.cudafoo` 这类前缀同名的包（当前无实害，但语义必须收紧）。
    """
    return module_name == _CUDA_NS or module_name.startswith(_CUDA_NS + ".")
