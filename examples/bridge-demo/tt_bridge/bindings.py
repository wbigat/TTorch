"""policy 条目：每个符号的决策、绑定实现、准入测试同处一个文件。

显式 register_binding()，真实对象引用——symbol/impl/test 拼写在 import 期即暴露。
评审一个符号的处置时不需要在多个文件之间跳。

remap 的 impl 直接引用 interface 门面函数（签名天然匹配，无需包装）。
"""
import vendor_cuda

from tt_bridge import admission, interface
from tt_bridge.binding_registry import Entry, Decision, register_binding

R, F, X = Decision.REMAP, Decision.FALLBACK, Decision.EXCEPTION

# --- 行为 ---

register_binding(Entry(symbol=vendor_cuda.set_device, decision=R,
               impl=interface.set_device, test=admission.set_device_equivalence,
               intent="把当前进程绑定到一张加速设备"))

register_binding(Entry(symbol=vendor_cuda.synchronize, decision=R,
               impl=interface.synchronize, test=admission.sync_binding,
               intent="等待设备上全部工作完成（device-wide，不是 current-stream）"))

register_binding(Entry(symbol=vendor_cuda.range_push, decision=R,
               impl=interface.range_push, test=admission.annotate_delivered,
               intent="profiler 区间标注；torch.accelerator 暂无对应，走 TT 扩展位"))

# --- 行为：P0 扩展（swift 主链路） ---

register_binding(Entry(symbol=vendor_cuda.Stream, decision=R,
               impl=interface.Stream, test=admission.stream_binding,
               intent="stream 对象族（创建/当前流/上下文/跨流等待）；"
                      "megatron ddp fencing 必触发，整体绑到门面"))
register_binding(Entry(symbol=vendor_cuda.current_stream, decision=R,
               impl=interface.current_stream, test=admission.stream_binding,
               intent="当前 stream 句柄（动态只读）"))
register_binding(Entry(symbol=vendor_cuda.stream, decision=R,
               impl=interface.stream, test=admission.stream_binding,
               intent="stream 上下文管理器：进入切换、退出恢复"))
register_binding(Entry(symbol=vendor_cuda.set_rng_state, decision=R,
               impl=interface.set_rng_state, test=admission.rng_binding,
               intent="checkpoint resume 恢复 RNG；非法值经 adapter 翻译成统一错误"))
register_binding(Entry(symbol=vendor_cuda.manual_seed, decision=R,
               impl=interface.manual_seed, test=admission.rng_binding,
               intent="全局 RNG 播种；与 get/set_rng_state 同族，真实 policy 已纳"
                      "（megatron 通信/训练路径 4 处调用点）"))
register_binding(Entry(symbol=vendor_cuda.empty_cache, decision=R,
               impl=interface.empty_cache, test=admission.empty_cache_binding,
               intent="allocator 提示；rollout/转换路径高频"))

# --- 状态查询（动态只读，禁缓存） ---

register_binding(Entry(symbol=vendor_cuda.current_device, decision=R,
               impl=interface.current_device, test=admission.current_device_consistency,
               intent="查询当前设备；返回值必须能当设备标识回喂"
                      "（swift 6+ 处传给 torch.tensor(device=)）"))

register_binding(Entry(symbol=vendor_cuda.memory_allocated, decision=R,
               impl=interface.memory_allocated, test=admission.mem_binding,
               intent="查询显存水位（动态只读，口径=已分配字节，禁缓存）"))

register_binding(Entry(symbol=vendor_cuda.mem_get_info, decision=R,
               impl=interface.mem_get_info, test=admission.mem_info_binding,
               intent="(free, total) 字节元组；patcher max_memory 初始化形态"))
register_binding(Entry(symbol=vendor_cuda.get_rng_state, decision=R,
               impl=interface.get_rng_state, test=admission.rng_binding,
               intent="checkpoint save 读 RNG 状态（不透明 str 契约）"))

# --- 能力（静态事实，读 probe 缓存，不进门面） ---

register_binding(Entry(symbol=vendor_cuda.is_available, decision=R,
               impl=interface.is_available,
               test=admission.binding_identity("is_available"),
               intent="分支选择器（设备字符串形态决策）；masquerade 语义由双身份兜底"))

register_binding(Entry(symbol=vendor_cuda.device_count, decision=R,
               impl=interface.device_count,
               test=admission.binding_identity("device_count"),
               intent="可见设备数"))

register_binding(Entry(symbol=vendor_cuda.can_device_access_peer, decision=R,
               impl=interface.can_device_access_peer,
               test=admission.binding_identity("can_device_access_peer"),
               intent="两卡间能否 P2P（逐对回答，转发厂商原生实现）"))

# --- fallback / exception ---

register_binding(Entry(symbol=vendor_cuda.graph_capture, decision=F,
               fallback_target="eager",
               intent="CUDA Graph 加速；当前平台无等价物，退回 eager，功能不受影响"))

register_binding(Entry(symbol=vendor_cuda.profiler_start, decision=X,
               intent="CUDA 专属 profiler 入口；无等价且非必需，调用方应删除"))
