"""torch.cuda 契约面盘点——D4「面是否闭合」的可执行判据。

闭合的定义（2026-10-22 与作者对齐）：**不是支持多少接口，而是每个公开符号都有
且仅有一个显式归属**。契约面 = `torch.cuda.__all__`（122 个公开符号），每个符号
必须落在三态之一，不允许存在第四种「没人决定过却能被调到」的静默穿透：

- Managed ：TT 接管到 interface 门面（Decision.REMAP），行为由 TT 定义并验收；
- Denied  ：特性功能点入口，显式「未实现」占位（Decision.EXCEPTION），报错指向
            所属功能点，遇到该功能点时再放行补实现；
- Exempt  ：显式豁免穿透——不涉及硬件差异、或 masquerade 下语义天然正确，逐条有据。

横切约束（贯穿三态）：**torch 内部依赖的符号禁止 Denied**（deny 会让 torch.compile /
AMP / FSDP 等 torch 自身功能崩——这是当初 default_deny=False 的真实原因，不是
「存在面外符号」）。公开 `__all__` 里无纯私有符号，122 个全部在契约面内。

分期（作者定的策略）：
- 第一轮：基础 device 状态/能力查询**必须 Managed 覆盖**（框架真实会调）；
- 特性功能点（CUDA graph / 显存池 / IPC / 编译调优）**先 Denied 占位**，按需补；
- 内存/rng 的完整变体随第一轮一并纳入 Managed（成组接管，口径验见识 D7）。
"""
import enum
import inspect

import torch.cuda as _tc


class Surface(enum.Enum):
    MANAGED = "managed"    # 接管
    DENIED = "denied"      # 特性点占位，显式未实现
    EXEMPT = "exempt"      # 显式豁免穿透


# --- 现有 18 条 policy 已纳管的顶层符号（nccl.version 是子模块内符号，不在 __all__）---
_MANAGED_EXISTING = {
    "set_device", "current_device", "synchronize", "empty_cache", "memory_allocated",
    "mem_get_info", "get_rng_state", "set_rng_state", "manual_seed", "Stream",
    "current_stream", "set_stream", "stream", "Event", "is_available", "device_count",
    "can_device_access_peer",
}

# --- 第一轮补 Managed：基础 device 状态 / 能力 / 硬件状态查询（框架真实会调）---
_MANAGED_ROUND1 = {
    # device 控制与状态
    "init", "is_initialized", "device",
    # 能力查询（masquerade 伪装信号的核心载体，必须给正确口径——D5 治理点）
    "get_device_capability", "get_device_name", "get_device_properties",
    "is_bf16_supported", "utilization",
    # 硬件身份/能力查询（masquerade 下穿透会暴露 CUDA 原生身份——实测
    # get_arch_list/get_gencode_flags 返回 NVIDIA sm_* 架构。这类符号豁免即
    # 伪装信号外溢 D5，必须接管给正确口径）
    "get_arch_list", "get_gencode_flags", "is_tf32_supported",
    # stream/event 基础（注：streams 是子模块，走 ismodule 兜底归 Exempt；
    # device/ExternalStream/StreamContext 是类型位类，接管须用 impl_factory 绑原生类）
    "default_stream", "ExternalStream", "StreamContext",
    "get_stream_from_external",
    # rng 完整族
    "seed", "seed_all", "initial_seed", "manual_seed_all",
    "get_rng_state_all", "set_rng_state_all",
    # 内存基础与派生查询（接管时必须验口径——D7）。
    # 注：memory_usage / device_memory_used 实测穿透裸奔 NVMLError（它们内部
    # 用 pynvml 查硬件，不是 allocator 口径查询）——归 Denied 的 hw-monitor 特性点
    "memory_reserved", "memory_cached", "max_memory_allocated", "max_memory_cached",
    "max_memory_reserved", "memory_stats", "memory_summary",
    "memory_stats_as_nested_dict",
    "get_per_process_memory_fraction", "host_memory_stats",
    "host_memory_stats_as_nested_dict", "reset_peak_memory_stats",
    "reset_max_memory_allocated", "reset_max_memory_cached",
    "reset_accumulated_memory_stats", "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    # debug 模式
    "get_sync_debug_mode", "set_sync_debug_mode",
}

# --- Denied：特性功能点入口，显式「未实现」占位，遇到对应功能点再放行补实现 ---
# 每个 Denied 符号标注所属功能点——占位报错信息据此指向"是哪个功能点、
# 如何放行"，而非泛泛的"未实现"。
#
# 横切约束（Denied ∩ torch 内部默认路径依赖 == ∅）：Denied 一个 torch 内部
# 默认路径依赖的符号会让 torch 自己崩（2026-10-22 实测：CUDAGraph 被 Denied
# 成函数后，transformers `-> torch.cuda.CUDAGraph | None` 注解在 import 期求值
# 全线 TypeError，4 卡训练崩）。新增 Denied 符号前必须核查依赖面：
#   grep -rn "torch\.cuda\.<name>" <torch>/ <megatron>/ <deepspeed>/ <transformers>/
# 且类型位符号（类）即使 Denied 也必须保持 type 身份（_denied_impl 已处理）。
_DENIED_FEATURE = {
    # CUDA graph 捕获（torch 内部 dynamo/inductor 的 cudagraph 路径用 graph，但那是
    # 特性路径（mode="reduce-overhead" 才走），非默认；框架手动捕获也走这里——
    # 昆仑芯已有 graph-safe 前科，未验收前显式占位）
    "CUDAGraph": "cuda-graph", "graph": "cuda-graph",
    "graph_pool_handle": "cuda-graph", "make_graphed_callables": "cuda-graph",
    # 显存池 / 自定义分配器（nccl_ub user-buffer 路径依赖 MemPool）
    "MemPool": "memory-pool", "use_mem_pool": "memory-pool",
    "set_per_process_memory_fraction": "memory-pool", "memory_snapshot": "memory-pool",
    "CUDAPluggableAllocator": "custom-allocator",
    "change_current_allocator": "custom-allocator",
    "caching_allocator_alloc": "custom-allocator",
    "caching_allocator_delete": "custom-allocator",
    "caching_allocator_enable": "custom-allocator",
    "get_allocator_backend": "custom-allocator",
    # CUDA IPC（多进程显存共享特性点）
    "ipc_collect": "cuda-ipc",
    # 硬件监控（XPU 无 NVML 监控库，穿透裸奔 NVMLError；megatron 的 GPU 监控特性
    # 才调——utils.py 的 temperature/power_draw/clock_rate，非训练主路径。属"特定
    # 功能点才用"，先 Denied 占位，启用监控功能时再验收转 Managed）
    "clock_rate": "hw-monitor", "power_draw": "hw-monitor",
    "temperature": "hw-monitor", "list_gpu_processes": "hw-monitor",
    # 显存使用率/设备用量——名字像口径查询，实测内部用 pynvml 查硬件（NVML），
    # 穿透裸奔 NVMLError；是监控不是 allocator 口径
    "memory_usage": "hw-monitor", "device_memory_used": "hw-monitor",
    # CUDA 专有 cuBLAS 句柄（XPU 无对应物，穿透裸奔 CUBLAS_STATUS_NOT_INITIALIZED）
    "current_blas_handle": "cuda-blas",
}

# --- Exempt：显式豁免穿透（规则 + 显式清单）---
_EXEMPT_EXPLICIT = {
    # 异常类型（不涉及硬件语义）
    "CudaError", "DeferredCudaCallError",
    # torch 内部粘合层（torch 自用、外部几乎不调、穿透形态正常——实测 cudart 返回
    # torch._C._cudart 模块、check_error 为函数、device_of 为类。注意 cudart 是
    # CUDA runtime 句柄，masquerade 下语义存疑，但仅 torch 内部使用故豁免）
    "check_error", "device_of", "cudaStatus", "cudart",
    # rng 值对象（tuple of Generator，无接管价值——通信部分已判定）
    "default_generators",
    # 模块级布尔**值**属性（非函数）：masquerade 伪装值正确（实测 True），穿透即可；
    # 转发函数机制（REMAP impl）不适配值属性，豁免
    "has_half", "has_magma",
    # graph 捕获状态查询：被 torch 默认路径依赖（optim/optimizer.py 每个 step、
    # nn/modules/_functions.py），Denied 会让训练崩——穿透语义正确（实测返回
    # False，厂商栈 masquerade 已实现该状态查询），豁免
    "is_current_stream_capturing",
}


def _classify(name):
    """返回 (Surface, reason)。规则顺序：显式表 → 类型规则 → 兜底。"""
    if name in _MANAGED_EXISTING:
        return Surface.MANAGED, "已纳管(现有 policy)"
    if name in _MANAGED_ROUND1:
        return Surface.MANAGED, "基础 device/能力/内存/rng 查询(第一轮)"
    if name in _DENIED_FEATURE:
        return Surface.DENIED, f"特性功能点[{_DENIED_FEATURE[name]}]入口(占位,按需补)"
    if name in _EXEMPT_EXPLICIT:
        return Surface.EXEMPT, "显式豁免穿透"
    obj = getattr(_tc, name, None)
    if name.endswith(("Tensor", "Storage")):
        return Surface.EXEMPT, "tensor/storage 类型别名(masquerade 下即原生类型)"
    if inspect.ismodule(obj):
        return Surface.EXEMPT, "子模块(穿透;内部符号按需单独纳管,如 nccl.version)"
    return None, f"未归类: {type(obj).__name__}"


def audit_surface(check_registry=True):
    """闭合判据：torch.cuda.__all__ 每个符号都有且仅有三态之一的显式归属。

    check_registry=True 时额外断言**声称 Managed == 实际注册 Entry**（证据先于
    声明）：surface 的分类是手写清单，若与 module_binding 的实际注册漂移，
    闭合判据就成了未经验证的声明（2026-10-22 实测曾漂移 9 个——9 个符号
    "声称纳管、实则静默穿透"）。漂移即 closed=False。

    返回 (closed, rows, unmanaged, drift)：
    - closed：True 表示面闭合（无未归类符号 且 声称与实际一致）；
    - rows：[(symbol, Surface, reason)] 全量画像；
    - unmanaged：未归类符号清单；
    - drift：None 或 {"claimed_not_registered": [...],
      "registered_not_classified": [...]}。
    """
    rows, unmanaged = [], []
    for name in sorted(getattr(_tc, "__all__", [])):
        st, reason = _classify(name)
        if st is None:
            unmanaged.append((name, reason))
        rows.append((name, st, reason))
    drift = None
    if check_registry:
        all_names = set(getattr(_tc, "__all__", []))
        # 延迟 import：surface 被 bootstrap→cuda_compat 间接依赖，此处才需注册表
        from ttorch.bridge import module_binding
        actual = {(e.attr_name or e.symbol.__name__) for e in module_binding.entries()}
        actual &= all_names  # 子模块符号（nccl.version）不在 __all__，排除出对账范围
        claimed = {n for n, st, _ in rows if st == Surface.MANAGED}
        d_claim = sorted(claimed - actual)
        d_reg = sorted(actual - claimed)
        if d_claim or d_reg:
            drift = {"claimed_not_registered": d_claim,
                     "registered_not_classified": d_reg}
    return (not unmanaged and drift is None), rows, unmanaged, drift


def denied_items():
    """Denied 占位清单：[(dotted, feature_point)]，供 activate 绑显式异常。

    dotted 为完整路径（torch.cuda.<name>），feature_point 用于报错信息指向
    所属功能点（cuda-graph / memory-pool / custom-allocator / cuda-ipc）。
    """
    return [(f"torch.cuda.{name}", feat) for name, feat in _DENIED_FEATURE.items()]


def summary():
    """闭合画像的紧凑摘要（供激活期打印 / 回归基线断言）。"""
    closed, rows, unmanaged, drift = audit_surface()
    from collections import Counter
    cnt = Counter(st.value for _, st, _ in rows if st)
    return {
        "closed": closed,
        "total": len(rows),
        "managed": cnt.get("managed", 0),
        "denied": cnt.get("denied", 0),
        "exempt": cnt.get("exempt", 0),
        "unmanaged": [n for n, _ in unmanaged],
        "drift": drift,
    }


if __name__ == "__main__":
    import json
    s = summary()
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if not s["closed"]:
        raise SystemExit(f"契约面不闭合：未归类={s['unmanaged']} 漂移={s['drift']}")
