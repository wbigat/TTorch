"""真实 torch.cuda 的 policy 条目（与 bindings.py 的 mock 盘点同构）。

差异：
- symbol 是真实 torch.cuda 符号；impl 同一套 platform_mgr 门面——
  门面签名已对齐 torch 真实签名（device 可选参），带参调用点不破；
- 每条都显式 bind_module="torch.cuda"：这些符号定义在 torch.cuda.memory /
  streams / random 等子模块再 re-export，框架访问的是 torch.cuda.X，
  绑定落点必须是框架可见路径（mock 单模块察觉不到此坑）；
- 真实环境 activate 时 default_deny=False：torch.cuda.__all__ 含 torch
  自身内部所需符号，白名单管的是框架调用面（swift 盘点的 20 符号），
  不接管 torch 自身内部；
- 准入测试复用 admission.py 中 ctx 驱动的条目（ctx["vendor_cuda"]=torch.cuda）；
  hw 钩子类测试（annotate_delivered）是 mock 专属，不在此注册。
"""
import torch.cuda

from ttorch.bridge import admission, platform_mgr
from ttorch.bridge import registry as _registry
from ttorch.bridge.registry import Category, Decision, Entry, binding_register

B, S, C = Category.BEHAVIOR, Category.STATE_QUERY, Category.CAPABILITY
R = Decision.REMAP
M = "torch.cuda"  # 绑定落点：框架可见的 re-export 路径


def set_device_roundtrip(ctx):
    """set_device 绑定正确性（真实环境版）：经绑定符号设置后回读一致。"""
    idx = platform_mgr.current_device()
    ctx["vendor_cuda"].set_device(idx)
    ok = ctx["vendor_cuda"].current_device() == idx
    return ok, f"set_device({idx}) 后 current_device 回读一致 -> {ok}"


# --- 行为 ---

binding_register(Entry(symbol=torch.cuda.set_device, category=B, decision=R,
               impl=platform_mgr.set_device, test=set_device_roundtrip,
               intent="把当前进程绑定到一张加速设备", bind_module=M))

binding_register(Entry(symbol=torch.cuda.synchronize, category=B, decision=R,
               impl=platform_mgr.synchronize, test=admission.sync_binding,
               intent="device-wide 等待；torch 原生语义与契约一致", bind_module=M))

binding_register(Entry(symbol=torch.cuda.Stream, category=B, decision=R,
               impl=platform_mgr.Stream, test=admission.vendor_internal_construction,
               intent="stream 对象族；ddp fencing 必触发。门面签名是厂商构造签名的"
                      "超集（G14）：原实现内部的 Stream(stream_id=...) 必须回落原生对象",
               bind_module=M))
binding_register(Entry(symbol=torch.cuda.current_stream, category=S, decision=R,
               impl=platform_mgr.current_stream, test=admission.stream_binding,
               intent="当前 stream 句柄（动态只读）", bind_module=M))
binding_register(Entry(symbol=torch.cuda.stream, category=B, decision=R,
               impl=platform_mgr.stream, test=admission.stream_binding,
               intent="stream 上下文管理器", bind_module=M))

binding_register(Entry(symbol=torch.cuda.set_rng_state, category=B, decision=R,
               impl=platform_mgr.set_rng_state, test=admission.rng_binding,
               intent="checkpoint resume 恢复 RNG；torch 原生 ByteTensor 契约", bind_module=M))
binding_register(Entry(symbol=torch.cuda.empty_cache, category=B, decision=R,
               impl=platform_mgr.empty_cache, test=admission.empty_cache_binding,
               intent="allocator 提示；尽力语义", bind_module=M))

# --- 状态查询 ---

binding_register(Entry(symbol=torch.cuda.current_device, category=S, decision=R,
               impl=platform_mgr.current_device, test=admission.current_device_consistency,
               intent="当前设备序号；构造 torch.device 请走 device_spec 门面",
               bind_module=M))
binding_register(Entry(symbol=torch.cuda.memory_allocated, category=S, decision=R,
               impl=platform_mgr.memory_allocated, test=admission.mem_binding,
               intent="显存水位（禁缓存）；平台声明 memory_query=False 时报 unsupported",
               bind_module=M))
binding_register(Entry(symbol=torch.cuda.mem_get_info, category=S, decision=R,
               impl=platform_mgr.mem_get_info, test=admission.mem_info_binding,
               intent="(free, total) 字节元组", bind_module=M))
binding_register(Entry(symbol=torch.cuda.get_rng_state, category=S, decision=R,
               impl=platform_mgr.get_rng_state, test=admission.rng_binding,
               intent="checkpoint save 读 RNG（torch 原生 ByteTensor；会被 megatron 回喂 Generator.set_state）", bind_module=M))

# --- 能力 ---

binding_register(Entry(symbol=torch.cuda.is_available, category=C, decision=R,
               cap_key="device_available",
               test=admission.make_cap_binding_test("is_available", "device_available"),
               intent="分支选择器；masquerade 语义由双身份兜底", bind_module=M))
binding_register(Entry(symbol=torch.cuda.device_count, category=C, decision=R,
               cap_key="device_count",
               test=admission.make_cap_binding_test("device_count", "device_count"),
               intent="可见设备数（静态事实）", bind_module=M))
binding_register(Entry(symbol=torch.cuda.can_device_access_peer, category=C, decision=R,
               cap_key="p2p",
               test=admission.make_cap_binding_test("can_device_access_peer", "p2p"),
               intent="P2P 是否可用（参数忽略，答 probe 期事实）", bind_module=M))


def entries(exclude_capability=False):
    """本文件的条目（registry 全局共享，按绑定落点过滤）。

    exclude_capability=True 时剔除能力类绑定（is_available/device_count/
    can_device_access_peer）——平台 overlay 的最小实例：在 torch.cuda 为
    非功能桩的栈上（cpu-dev），把 is_available 绑成 True 会误导 torch
    内部守卫（AdamW 的 graph capture 健康检查经 torch.accelerator 走到
    dummy cuda 直接炸）。masquerade 栈无此问题（shim 真实可工作，
    原答案本就是 True）；这些栈上框架的 is_available 语义改由
    platform_mgr.capability("device_available") 直接供给。
    """
    es = [e for e in _registry.entries() if e.bind_module == M]
    if exclude_capability:
        es = [e for e in es if e.category is not Category.CAPABILITY]
    return es
