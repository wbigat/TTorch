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


def _native_stream_class(entry, original):
    """Stream 是类型位符号：bind 期解析为平台原生 Stream **类对象**。

    不能绑转发函数（`type(torch.cuda.Stream)` 变成 function 后，`Stream | None`
    返回注解在 import 期求值、isinstance 判定全线 TypeError——transformers 5.x
    直接导不进来）。现有平台的正确答案都是原生类本身：masquerade/cuda 上原生
    类即正确（G14：torch 内部 `Stream(stream_id=...)` 构造也经它完成）；
    cpu/xpu-native 上框架不走 torch.cuda.Stream，保持原生不接管语义。
    """
    if not isinstance(original, type):
        raise TypeError(f"{entry.dotted}: 绑定前值不是类型（{type(original).__name__}），"
                        f"类型位符号拒绝接管")
    return original


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


def entries():
    """本文件的条目（binding_registry 全局共享，按绑定落点过滤）。"""
    return list(get_entries(bind_module=M))
