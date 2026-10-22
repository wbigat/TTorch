"""TorchDeviceModuleBase：**torch 设备模块范式**的公共实现。

命名辨正（2026-10-13，原名 TorchAccelBase 名不符实）：本类不是"CUDA 的实现
逻辑"，而是 torch 对设备模块的**统一接口范式**——`torch.cuda` / `torch.xpu` /
`torch.npu` 这些模块暴露同名同义的 `set_device` / `synchronize` /
`current_device` / `memory_allocated` / `get_rng_state` / `Stream` /
`current_stream` / `set_stream`。凡走这套范式的平台都可以继承它，与是不是
NVIDIA 无关。

所以继承关系是按"接口范式"而非"厂商"划分的：

    TorchDeviceModuleBase            # 本文件：范式通用实现
      ├── TorchCudaPlatform          # cuda.py：补 CUDA 命名空间专属项
      │     └── TorchXpuMasqueradePlatform   # 背后就是 torch.cuda，复用其实现
      └── TorchXpuNativePlatform     # xpu_native.py：torch.xpu，自报版本与 graph

**本类刻意不实现两件事**，因为它们不是范式的一部分而是各平台专属（放进来
就等于默认所有平台都是 CUDA，正是原设计的错处）：

- `sdk_version()`：版本来源各家不同（`torch.version.cuda` vs
  `torch.version.xpu`），本类不猜 -> 保持 PlatformBase 的抽象，子类必须实现。

所有 torch 调用必须经 `_orig()` 拿绑定前的原实现——cuda-compat 绑定后 mod 上
的同名符号已是门面，直接调会无限递归（门面 -> adapter -> 门面）。
`_origin_<name>` 留痕是绑定机制保存的原实现，未绑定则回落当前值（未激活
绑定的进程里行为不变）。
"""
import torch

from ttorch.platform.platform_handle import TTPlatformError
from ttorch.platform.adapters._rng import rng_from_state, rng_to_state
from ttorch.platform.platform_base import PlatformBase

__all__ = ["TorchDeviceModuleBase", "resolve_device_module"]


def resolve_device_module(path):
    """惰性解析设备模块：PrivateUse1 栈的 torch.xpu / torch.npu 是厂商包
    运行时注册出来的属性（rename_privateuse1_backend +
    _register_device_module），在 import 期求值成类属性必然拿到 None——
    必须等到激活期、厂商包已就位时再取。"""
    obj = torch
    for part in path.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            raise TTPlatformError(
                "platform", f"设备模块 '{path}' 不可用（'{part}' 未注册）")
    return obj


class TorchDeviceModuleBase(PlatformBase):
    """torch 设备模块（torch.cuda / torch.xpu / ...）的公共映射。

    子类只需指定 `mod_path` 并实现 `sdk_version()`。该栈缺某个接口时覆盖
    对应方法抛 unsupported 即可（或不覆盖、沿用 PlatformBase 默认）。
    """

    mod_path = None  # "torch.cuda" / "torch.xpu"：路径而非对象，实例化时才解析
    sdk_range = ("0.0", "999")  # 版本范围按厂商校准，默认不卡

    def __init__(self):
        if self.mod_path is None:
            raise TTPlatformError("platform", f"{type(self).__name__}: mod_path 未指定")
        # 实例属性（conformance 的 graph 实测也读它取 CUDAGraph）
        self.mod = resolve_device_module(self.mod_path)

    def _orig(self, name):
        """取符号的**原生实现**（绑定前）：优先 TT 的 owner 留痕，未绑定则取当前值。

        为什么必须走留痕而不是直接 getattr：绑定后 `torch.cuda.<name>` 已经是
        TT 门面，直接取会形成"门面 → activation().<name>() → _orig → 门面"的
        无限递归（2026-10-20 留痕改名后实测 RecursionError）。留痕名随
        `_rebind.origin_attr` 的 owner 规则（bridge 是绑定方），不能写死
        `_origin_`——那是改名前的旧前缀，也是与 provider 撞名的那个前缀。
        """
        from ttorch.utils import _rebind
        return getattr(self.mod, _rebind.origin_attr("bridge", name),
                       getattr(self.mod, name))

    # --- 平台自述 ---
    # sdk_version() 刻意不实现：版本来源是平台专属，见模块注释。

    # --- 硬件存在性 / 规模查询（直接问厂商模块，不缓存） ---
    def is_available(self):
        return bool(self._orig("is_available")())

    def device_count(self):
        return self._orig("device_count")()

    def can_device_access_peer(self, device, peer_device):
        if not hasattr(self.mod, "can_device_access_peer"):
            raise TTPlatformError(
                "unsupported", f"{self.mod_path} 无 can_device_access_peer")
        return bool(self._orig("can_device_access_peer")(device, peer_device))

    # --- 能力 / 身份查询（masquerade 下转发厂商伪装值，不改；G13 None 省略实参） ---
    def get_device_capability(self, device=None):
        fn = self._orig("get_device_capability")
        major, minor = fn() if device is None else fn(device)
        return (int(major), int(minor))

    def get_device_name(self, device=None):
        fn = self._orig("get_device_name")
        return fn() if device is None else fn(device)

    def get_device_properties(self, device):
        return self._orig("get_device_properties")(device)

    def is_bf16_supported(self, including_emulation=True):
        # G12：torch 原生 is_tf32_supported 内部调它且传 including_emulation
        return bool(self._orig("is_bf16_supported")(including_emulation))

    def utilization(self, device=None):
        fn = self._orig("utilization")
        return int(fn() if device is None else fn(device))

    def is_tf32_supported(self):
        return bool(self._orig("is_tf32_supported")())

    def get_arch_list(self):
        return self._orig("get_arch_list")()

    def get_gencode_flags(self):
        return self._orig("get_gencode_flags")()

    def get_sync_debug_mode(self):
        return self._orig("get_sync_debug_mode")()

    def set_sync_debug_mode(self, debug_mode):
        self._orig("set_sync_debug_mode")(debug_mode)

    def init(self):
        self._orig("init")()

    def is_initialized(self):
        return bool(self._orig("is_initialized")())

    # --- 设备 ---
    def set_device(self, device):
        try:
            self._orig("set_device")(device)
        except Exception as e:
            raise TTPlatformError("invalid_device", str(e)) from e

    def current_device(self):
        return self._orig("current_device")()

    def synchronize(self, device=None):
        # G13 统一约定：None=当前设备必须**省略实参**，不位置透传——torch 原生
        # 接受 None，但厂商栈不一定接受（get_rng_state 曾在 torch 2.9 因此炸）。
        fn = self._orig("synchronize")
        fn() if device is None else fn(device)

    # --- 显存 ---
    def empty_cache(self):
        if hasattr(self.mod, "empty_cache"):
            self._orig("empty_cache")()

    def memory_allocated(self, device=None):
        fn = self._orig("memory_allocated")  # G13：None 省略实参，见 synchronize
        return fn() if device is None else fn(device)

    def mem_get_info(self, device=None):
        fn = self._orig("mem_get_info")  # G13：None 省略实参，见 synchronize
        free, total = fn() if device is None else fn(device)
        return (int(free), int(total))

    def _q(self, name, device):
        """(device=None) 显存口径查询的统一转发（G13 None 省略实参）。

        这批 17 个查询签名、语义同构（动态只读），用一处转发避免重复；
        口径正确性由 conformance D7 验收，不在转发层处理。
        """
        fn = self._orig(name)
        return fn() if device is None else fn(device)

    def memory_reserved(self, device=None):
        return self._q("memory_reserved", device)

    def memory_cached(self, device=None):
        return self._q("memory_cached", device)

    def max_memory_allocated(self, device=None):
        return self._q("max_memory_allocated", device)

    def max_memory_cached(self, device=None):
        return self._q("max_memory_cached", device)

    def max_memory_reserved(self, device=None):
        return self._q("max_memory_reserved", device)

    def memory_stats(self, device=None):
        return self._q("memory_stats", device)

    def memory_stats_as_nested_dict(self, device=None):
        return self._q("memory_stats_as_nested_dict", device)

    def get_per_process_memory_fraction(self, device=None):
        return self._q("get_per_process_memory_fraction", device)

    def host_memory_stats(self, device=None):
        return self._q("host_memory_stats", device)

    def host_memory_stats_as_nested_dict(self, device=None):
        return self._q("host_memory_stats_as_nested_dict", device)

    def reset_peak_memory_stats(self, device=None):
        self._q("reset_peak_memory_stats", device)

    def reset_max_memory_allocated(self, device=None):
        self._q("reset_max_memory_allocated", device)

    def reset_max_memory_cached(self, device=None):
        self._q("reset_max_memory_cached", device)

    def reset_accumulated_memory_stats(self, device=None):
        self._q("reset_accumulated_memory_stats", device)

    def reset_accumulated_host_memory_stats(self, device=None):
        self._q("reset_accumulated_host_memory_stats", device)

    def reset_peak_host_memory_stats(self, device=None):
        self._q("reset_peak_host_memory_stats", device)

    def memory_summary(self, device=None, abbreviated=False):
        # 唯一带额外参数的（abbreviated），不并入 _q
        fn = self._orig("memory_summary")
        if device is None:
            return fn(abbreviated=abbreviated)
        return fn(device, abbreviated=abbreviated)

    # --- RNG（torch 原生 ByteTensor 契约，见 _rng.rng_to_state 的 G15 说明）---
    def get_rng_state(self, device=None):
        try:
            # torch>=2.x 的 get_rng_state(device) 不接受 None（内部直接取
            # device.index），None=当前设备必须**省略实参**，不能位置透传（G13）
            fn = self._orig("get_rng_state")
            state = fn() if device is None else fn(device)
            return rng_to_state(state)
        except Exception as e:
            raise TTPlatformError("rng_state", str(e)) from e

    def set_rng_state(self, state, device=None):
        t = rng_from_state(state)  # 类型不对在这里抛 invalid_rng_state
        try:
            if device is None:
                self._orig("set_rng_state")(t)
            else:
                self._orig("set_rng_state")(t, device)
        except TTPlatformError:
            raise
        except Exception as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    def manual_seed(self, seed, device=None):
        """全局 RNG 播种（通信/训练路径真实调用点，megatron 实测 4 处）。

        与 set_rng_state 同族：经门面接管后框架播种行为可观测（进 provenance 的
        绑定面）。G13：device=None 省略实参，不位置透传。
        """
        fn = self._orig("manual_seed")
        try:
            fn(seed) if device is None else fn(seed, device)
        except Exception as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    # --- RNG 完整族（全设备变体） ---
    def seed(self, device=None):
        fn = self._orig("seed")  # G13：None 省略实参
        fn() if device is None else fn(device)

    def seed_all(self):
        self._orig("seed_all")()

    def initial_seed(self, device=None):
        fn = self._orig("initial_seed")  # G13
        return fn() if device is None else fn(device)

    def manual_seed_all(self, seed):
        self._orig("manual_seed_all")(seed)

    def get_rng_state_all(self):
        # 原生 ByteTensor 列表（G15），门面不包装，直接流出
        return self._orig("get_rng_state_all")()

    def set_rng_state_all(self, new_states):
        self._orig("set_rng_state_all")(new_states)

    # --- stream（返回该栈的原生 Stream 对象，门面不包装） ---
    def Stream(self, **kwargs):  # noqa: N802  # 对齐 torch.cuda.Stream 符号名
        """原生 Stream 构造：kwargs 原样透传。

        无参=框架创建；带 stream_id/device_index/device_type=torch 原实现
        内部构造（G14）。两者都交给原生类即正确，无需分流。
        """
        return self._orig("Stream")(**kwargs)

    def current_stream(self, device=None):
        fn = self._orig("current_stream")
        return fn() if device is None else fn(device)

    def set_stream(self, stream):
        self._orig("set_stream")(stream)

    def stream(self, stream):
        """转发厂商原生 `mod.stream(s)`（返回其 StreamContext）。

        不自己用 set_stream/current_stream 组装：厂商的 StreamContext 还要
        处理与 allocator / graph 捕获的交互，自造一个只会丢语义。
        """
        return self._orig("stream")(stream)

    def default_stream(self, device=None):
        return self._q("default_stream", device)

    def get_stream_from_external(self, data_ptr, device=None):
        fn = self._orig("get_stream_from_external")  # G13：device=None 省略实参
        return fn(data_ptr) if device is None else fn(data_ptr, device)
