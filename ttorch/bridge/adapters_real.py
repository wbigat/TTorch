"""真实环境 adapter：背靠真实 torch 的 CUDA / XPU 栈（与 mock adapters.py 平级）。

与 mock 的差异：
- 没有 hw 钩子，conformance 自动走 tensor 级契约版；
- 厂商 API 缺项（如 native xpu 可能没有 mem_get_info/get_rng_state）用
  getattr 探测，缺失即声明对应 capability=False——门面报 unsupported，
  框架走降级路径，而不是在厂商 API 深处炸原生错误；
- RNG 契约落地：torch 原生 ByteTensor（会流出 TT 边界被框架直接消费，见 G15）。

三个真实平台 + 一个 dev 平台（身份与 detect 见 platform_specs.py）：
- TorchCudaPlatform          真 NVIDIA CUDA
- TorchXpuMasqueradePlatform KL3/XPYTORCH 风格：torch.cuda.* 实际跑 XPU
- TorchXpuNativePlatform     torch.xpu 原生栈（torch>=2.4 / torch_xpu 扩展）
- TorchCpuPlatform           本地开发验证专用，用于在没有加速卡的机器上
                             跑通 smoke 全链路

本文件是**适配面**：只有实现，不含 detect，也不自报身份——两者属管理面，
在 platform_specs.py，由 manager 激活期注入。本模块由 _load() 在平台确定
之后才 import，因此这里可以安全地 import 厂商包（反之 platform_specs.py
严禁）。

待真实环境校准项（本地无法验证，已标注）：
- masquerade 栈的 sdk_version 来源（torch.version.cuda 在定制栈上的取值）；
- torch.xpu 原生栈的 mem_get_info/get_rng_state 是否存在（缺失会自动降级）。
"""
import torch

from ttorch.bridge.platform_adapter import PlatformAdapter, TTPlatformError


def _resolve_mod(path):
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


def _rng_to_state(state_tensor):
    """RNG state 契约 = **torch 原生 ByteTensor**（不是 bytes）。

    契约修正（G15，2026-10-12 megatron 真实链路实测）：原先定为"不透明
    可序列化对象（bytes）"，理由是跨厂商可移植。但真实调用面否决了这个
    假设——megatron 的 RNG tracker 把 get_rng_state() 的返回值**原样回喂
    给 torch 原生 API**，中间不经我们的 set_rng_state：

        megatron/core/tensor_parallel/random.py:289
            orig_rng_state = torch.cuda.get_rng_state()   # 经 TT
        同文件 :164（_set_cuda_rng_state 的 _lazy_call 回调里）
            default_generator.set_state(new_state)        # 未经 TT，torch 原生
        -> TypeError: expected a torch.ByteTensor, but got bytes

    即状态对象会**流出 TT 边界**被厂商/框架原生代码直接消费，因此它不能
    是我们自定义的表示。门面的返回值类型必须与被替换符号一致——这是
    G12"签名超集"原则在**返回值**上的对应面：签名管入参，类型管出参。

    可移植性由 adapter 负责（各平台都返回自家 torch 栈的 ByteTensor），
    而不是靠统一成 bytes。
    """
    return state_tensor


def _rng_from_state(state):
    """接受 torch 原生 ByteTensor；bytes 仍兼容（旧 checkpoint 可读）。"""
    if isinstance(state, torch.Tensor):
        return state
    if isinstance(state, (bytes, bytearray)):  # 向后兼容：旧格式 checkpoint
        return torch.frombuffer(bytearray(state), dtype=torch.uint8)
    raise TTPlatformError(
        "invalid_rng_state",
        f"RNG state 契约=torch ByteTensor（兼容 bytes），实际 {type(state).__name__}")


class _TorchAccelBase(PlatformAdapter):
    """torch.cuda / torch.xpu 模块的公共映射。mod 由子类指定。

    所有 torch 调用必须经 _orig()：cuda-compat 绑定后 mod 上的同名符号
    已被换成门面，直接调 mod.synchronize 会无限递归（门面 -> adapter ->
    门面）。_origin_<name> 留痕是绑定机制保存的原实现，未绑定则回落
    当前值（未激活绑定的进程里行为不变）。
    """

    mod_path = None  # "torch.cuda" / "torch.xpu"：路径而非对象，激活期才解析
    sdk_range = ("0.0", "999")  # 真实栈版本范围待按厂商校准，默认不卡

    def __init__(self):
        if self.mod_path is None:
            raise TTPlatformError("platform", f"{self.name}: mod_path 未指定")
        # 实例属性（conformance 的 cap_matches_probe 也读它）
        self.mod = _resolve_mod(self.mod_path)

    def _orig(self, name):
        return getattr(self.mod, f"_origin_{name}", getattr(self.mod, name))

    def sdk_version(self):
        return getattr(torch.version, "cuda", None) or "0.0"

    # --- 能力探测：缺 API 即声明 False，不编 ---
    def capabilities(self):
        m = self.mod
        p2p = False
        if m.device_count() > 1 and hasattr(m, "can_device_access_peer"):
            try:
                p2p = bool(m.can_device_access_peer(0, 1))
            except Exception:
                p2p = False
        return {"device_available": bool(m.is_available()),
                "device_count": m.device_count(),
                "p2p": p2p,
                "device_wide_sync": True,  # torch 语义 synchronize 即 device-wide
                # 能力驱动：只看本平台 mod 的 API 存在性，不靠 device_name 身份推断
                # （硬编码 torch.cuda 会让 mod=torch.xpu 的 native 栈也误答 True）；
                # 承诺 True 由 conformance 的 graph_capture_roundtrip 实测背书
                "graph_capture": hasattr(m, "CUDAGraph"),
                "async_comm": False,  # 保守声明；真实收益由 provider 门禁实测
                "rng_state": hasattr(m, "get_rng_state") and hasattr(m, "set_rng_state"),
                "stream_api": all(hasattr(m, a) for a in ("Stream", "current_stream", "set_stream")),
                "memory_query": hasattr(m, "memory_allocated") and hasattr(m, "mem_get_info")}

    def active_streams(self):
        # torch 不暴露活跃流枚举；device-wide 语义由 mod.synchronize 原生保证
        raise TTPlatformError("unsupported", "torch 栈不暴露活跃流枚举（也不需要：sync 原生 device-wide）")

    # --- 行为 ---
    def set_device(self, idx):
        try:
            self._orig("set_device")(idx)
        except Exception as e:
            raise TTPlatformError("invalid_device", str(e)) from e

    def synchronize(self, device=None):
        self._orig("synchronize")(device)

    def range_push(self, name):
        # 对齐 torch.accelerator 之前的扩展位：nvtx 风格区间标注
        nvtx = getattr(self.mod, "nvtx", None)
        if nvtx is not None and hasattr(nvtx, "range_push"):
            nvtx.range_push(name)  # 尽力语义：无 nvtx 的栈不标注、不阻塞训练

    def empty_cache(self):
        if hasattr(self.mod, "empty_cache"):
            self._orig("empty_cache")()

    # --- 状态查询 ---
    def current_device(self):
        return self._orig("current_device")()

    def memory_allocated(self, device=None):
        return self._orig("memory_allocated")(device)

    def mem_get_info(self, device=None):
        free, total = self._orig("mem_get_info")(device)
        return (int(free), int(total))

    # --- RNG（torch 原生 ByteTensor 契约，见 _rng_to_state 的 G15 说明）---
    def get_rng_state(self, device=None):
        try:
            # torch>=2.x 的 get_rng_state(device) 不接受 None（内部 device.index），
            # None=当前设备必须省略实参，不能位置透传
            fn = self._orig("get_rng_state")
            state = fn() if device is None else fn(device)
            return _rng_to_state(state)
        except Exception as e:
            raise TTPlatformError("rng_state", str(e)) from e

    def set_rng_state(self, state, device=None):
        t = _rng_from_state(state)  # 类型不对在这里抛 invalid_rng_state
        try:
            if device is None:
                self._orig("set_rng_state")(t)
            else:
                self._orig("set_rng_state")(t, device)
        except TTPlatformError:
            raise
        except Exception as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    # --- stream 原语：sid 即真实 torch Stream 对象 ---
    def stream_create(self):
        return self._orig("Stream")()

    def stream_current(self):
        return self._orig("current_stream")()

    def stream_set(self, sid):
        self._orig("set_stream")(sid)

    def stream_wait(self, waiter, waitee):
        waiter.wait_stream(waitee)  # torch 原生语义：waiter 等 waitee

    def stream_wrap(self, **kwargs):
        """原生 torch Stream 构造（_origin_Stream 是绑定前的真类）。"""
        return self._orig("Stream")(**kwargs)

    def stream_key(self, sid):
        """stream 身份的稳定可比较 key（torch Stream 对象每次包装可能不同）。"""
        return (getattr(sid, "device_index", None),
                getattr(sid, "stream_id", None)) if not isinstance(sid, int) else sid


class TorchCudaPlatform(_TorchAccelBase):
    """真 NVIDIA CUDA。"""

    mod_path = "torch.cuda"
    sdk_range = ("11.0", "13.5")


class TorchXpuMasqueradePlatform(_TorchAccelBase):
    """KL3/XPYTORCH 风格：torch.cuda.* 实际跑 XPU。身份分清由 spec 负责：
    vendor_name=xpu 系（provider 分发用），device_name="cuda"（torch 只认得它）。
    """

    mod_path = "torch.cuda"


class TorchXpuNativePlatform(_TorchAccelBase):
    """torch.xpu 原生栈（torch>=2.4 内建或 torch_xpu 扩展注册）。"""

    mod_path = "torch.xpu"


class TorchCpuPlatform(PlatformAdapter):
    """本地开发验证专用：无加速卡的机器上跑通 smoke 全链路。

    诚实降级：stream_api / memory_query 声明 False（CPU 没有这些概念），
    门面报 unsupported，框架走降级路径——本身就是能力驱动降级的活样本。
    RNG 用 torch 全局 CPU RNG。同步语义：CPU 执行天然同步，device-wide 平凡成立。
    """

    def sdk_version(self):
        return torch.__version__.split("+")[0]

    def capabilities(self):
        return {"device_available": True,
                "device_count": 1,
                "p2p": False,
                "device_wide_sync": True,   # CPU 执行天然同步，平凡成立
                "graph_capture": False,
                "async_comm": False,
                "rng_state": True,
                "stream_api": False,
                "memory_query": False}

    def set_device(self, idx):
        if idx != 0:
            raise TTPlatformError("invalid_device", f"CPU 平台只有 device 0，收到 {idx}")

    def current_device(self):
        return 0

    def synchronize(self, device=None):
        pass  # CPU 天然同步

    def range_push(self, name):
        pass

    def empty_cache(self):
        pass

    def get_rng_state(self, device=None):
        return _rng_to_state(torch.get_rng_state())

    def set_rng_state(self, state, device=None):
        torch.set_rng_state(_rng_from_state(state))
