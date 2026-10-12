"""policy 条目：每个符号的决策、绑定实现、准入测试同处一个文件。

显式 binding_register()，真实对象引用——symbol/impl/test 拼写在 import 期即暴露。
评审一个符号的处置时不需要在多个文件之间跳。

remap 的 impl 直接引用 platform_mgr 门面函数（签名天然匹配，无需包装）。
"""
import vendor_cuda

from tt_bridge import admission, platform_mgr
from tt_bridge.registry import Category, Entry, Decision, binding_register

B, S, C = Category.BEHAVIOR, Category.STATE_QUERY, Category.CAPABILITY
R, F, X = Decision.REMAP, Decision.FALLBACK, Decision.EXCEPTION

# --- 行为 ---

binding_register(Entry(symbol=vendor_cuda.set_device, category=B, decision=R,
               impl=platform_mgr.set_device, test=admission.set_device_equivalence,
               intent="把当前进程绑定到一张加速设备"))

binding_register(Entry(symbol=vendor_cuda.synchronize, category=B, decision=R,
               impl=platform_mgr.synchronize, test=admission.sync_binding,
               intent="等待设备上全部工作完成（device-wide，不是 current-stream）"))

binding_register(Entry(symbol=vendor_cuda.range_push, category=B, decision=R,
               impl=platform_mgr.range_push, test=admission.annotate_delivered,
               intent="profiler 区间标注；torch.accelerator 暂无对应，走 TT 扩展位"))

# --- 行为：P0 扩展（swift 主链路） ---

binding_register(Entry(symbol=vendor_cuda.Stream, category=B, decision=R,
               impl=platform_mgr.Stream, test=admission.stream_binding,
               intent="stream 对象族（创建/当前流/上下文/跨流等待）；"
                      "megatron ddp fencing 必触发，整体绑到门面"))
binding_register(Entry(symbol=vendor_cuda.current_stream, category=S, decision=R,
               impl=platform_mgr.current_stream, test=admission.stream_binding,
               intent="当前 stream 句柄（动态只读）"))
binding_register(Entry(symbol=vendor_cuda.stream, category=B, decision=R,
               impl=platform_mgr.stream, test=admission.stream_binding,
               intent="stream 上下文管理器：进入切换、退出恢复"))
binding_register(Entry(symbol=vendor_cuda.set_rng_state, category=B, decision=R,
               impl=platform_mgr.set_rng_state, test=admission.rng_binding,
               intent="checkpoint resume 恢复 RNG；非法值经 adapter 翻译成统一错误"))
binding_register(Entry(symbol=vendor_cuda.empty_cache, category=B, decision=R,
               impl=platform_mgr.empty_cache, test=admission.empty_cache_binding,
               intent="allocator 提示；rollout/转换路径高频"))

# --- 状态查询（动态只读，禁缓存） ---

binding_register(Entry(symbol=vendor_cuda.current_device, category=S, decision=R,
               impl=platform_mgr.current_device, test=admission.current_device_consistency,
               intent="查询当前设备；返回值必须能当设备标识回喂"
                      "（swift 6+ 处传给 torch.tensor(device=)）"))

binding_register(Entry(symbol=vendor_cuda.memory_allocated, category=S, decision=R,
               impl=platform_mgr.memory_allocated, test=admission.mem_binding,
               intent="查询显存水位（动态只读，口径=已分配字节，禁缓存）"))

binding_register(Entry(symbol=vendor_cuda.mem_get_info, category=S, decision=R,
               impl=platform_mgr.mem_get_info, test=admission.mem_info_binding,
               intent="(free, total) 字节元组；patcher max_memory 初始化形态"))
binding_register(Entry(symbol=vendor_cuda.get_rng_state, category=S, decision=R,
               impl=platform_mgr.get_rng_state, test=admission.rng_binding,
               intent="checkpoint save 读 RNG 状态（不透明 str 契约）"))

# --- 能力（静态事实，读 probe 缓存，不进门面） ---

binding_register(Entry(symbol=vendor_cuda.is_available, category=C, decision=R,
               cap_key="device_available",
               test=admission.make_cap_binding_test("is_available", "device_available"),
               intent="分支选择器（设备字符串形态决策）；masquerade 语义由双身份兜底"))

binding_register(Entry(symbol=vendor_cuda.device_count, category=C, decision=R,
               cap_key="device_count",
               test=admission.make_cap_binding_test("device_count", "device_count"),
               intent="可见设备数（静态事实，probe 期填充，不进 runtime API）"))

binding_register(Entry(symbol=vendor_cuda.can_device_access_peer, category=C, decision=R,
               cap_key="p2p",
               test=admission.make_cap_binding_test("can_device_access_peer", "p2p"),
               intent="P2P 是否可用（静态事实，选择期消费，不进 runtime API）"))

# --- fallback / exception ---

binding_register(Entry(symbol=vendor_cuda.graph_capture, category=B, decision=F,
               fallback_target="eager",
               intent="CUDA Graph 加速；当前平台无等价物，退回 eager，功能不受影响"))

binding_register(Entry(symbol=vendor_cuda.profiler_start, category=B, decision=X,
               intent="CUDA 专属 profiler 入口；无等价且非必需，调用方应删除"))
