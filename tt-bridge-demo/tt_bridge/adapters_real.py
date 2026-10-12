"""真实环境 adapter：背靠真实 torch 的 CUDA / XPU 栈（与 mock adapters.py 平级）。

与 mock 的差异：
- 没有 hw 钩子，conformance 自动走 tensor 级契约版；
- 厂商 API 缺项（如 native xpu 可能没有 mem_get_info/get_rng_state）用
  getattr 探测，缺失即声明对应 capability=False——门面报 unsupported，
  框架走降级路径，而不是在厂商 API 深处炸原生错误；
- RNG 契约落地：torch ByteTensor <-> bytes（不透明可序列化对象）。

三个真实平台 + 一个 dev 平台：
- TorchCudaPlatform          真 NVIDIA CUDA
- TorchXpuMasqueradePlatform KL3/XPYTORCH 风格：torch.cuda.* 实际跑 XPU
                             （masquerades_as={"cuda"}，device_name="cuda"）
- TorchXpuNativePlatform     torch.xpu 原生栈（torch>=2.4 / torch_xpu 扩展）
- TorchCpuPlatform           本地开发验证专用（auto_detect=False），
                             用于在没有加速卡的机器上跑通 smoke 全链路

待真实环境校准项（本地无法验证，已标注）：
- XPU masquerade 的 device name 标记（"xpu"/"kl3"）是否覆盖真实卡名；
- masquerade 栈的 sdk_version 来源（torch.version.cuda 在定制栈上的取值）；
- torch.xpu 原生栈的 mem_get_info/get_rng_state 是否存在（缺失会自动降级）。
"""
import importlib.util

import torch

from tt_bridge.platform_adapter import PlatformAdapter, TTPlatformError

# masquerade 探测标记：torch.cuda.get_device_name() 含这些子串时，
# 说明 torch.cuda 背后不是 NVIDIA（待真实环境校准的卡名清单）
_XPU_MARKERS = ("xpu", "kl3")

# KL3 厂商栈的 native 探测通道：torch_xmlir 包存在性。
# 真实环境校准结论（2026-10-12，KL3 机器实测）：卡名是通用的 "gpu"，
# _XPU_MARKERS 不命中；但该环境 torch_xmlir 随栈预装（libbkcl/XCCL 由其
# 加载），包存在性是厂商私有通道，不会被 NVIDIA 栈误命中。
_XPU_NATIVE_PACKAGES = ("torch_xmlir",)


def _cuda_device_name():
    try:
        return torch.cuda.get_device_name(0).lower()
    except Exception:
        return ""


def _rng_to_bytes(state_tensor):
    return state_tensor.cpu().numpy().tobytes()


def _rng_from_bytes(state):
    if not isinstance(state, (bytes, bytearray)):
        raise TTPlatformError(
            "invalid_rng_state",
            f"RNG state 契约=bytes（torch ByteTensor 序列化），实际 {type(state).__name__}")
    return torch.frombuffer(bytearray(state), dtype=torch.uint8)


class _TorchAccelBase(PlatformAdapter):
    """torch.cuda / torch.xpu 模块的公共映射。mod 由子类指定。

    所有 torch 调用必须经 _orig()：cuda-compat 绑定后 mod 上的同名符号
    已被换成门面，直接调 mod.synchronize 会无限递归（门面 -> adapter ->
    门面）。_origin_<name> 留痕是绑定机制保存的原实现，未绑定则回落
    当前值（未激活绑定的进程里行为不变）。
    """

    mod = None  # torch.cuda 或 torch.xpu（类属性，conformance 的 cap_matches_probe 也读它）
    sdk_range = ("0.0", "999")  # 真实栈版本范围待按厂商校准，默认不卡

    def __init__(self):
        if self.mod is None:
            raise TTPlatformError("platform", f"{self.name}: mod 未指定")

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
                "graph_capture": hasattr(torch.cuda, "CUDAGraph") and self.device_name == "cuda",
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

    # --- RNG（bytes 契约）---
    def get_rng_state(self, device=None):
        try:
            # torch>=2.x 的 get_rng_state(device) 不接受 None（内部 device.index），
            # None=当前设备必须省略实参，不能位置透传
            fn = self._orig("get_rng_state")
            state = fn() if device is None else fn(device)
            return _rng_to_bytes(state)
        except Exception as e:
            raise TTPlatformError("rng_state", str(e)) from e

    def set_rng_state(self, state, device=None):
        t = _rng_from_bytes(state)  # 类型不对在这里抛 invalid_rng_state
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

    def stream_key(self, sid):
        """stream 身份的稳定可比较 key（torch Stream 对象每次包装可能不同）。"""
        return (getattr(sid, "device_index", None),
                getattr(sid, "stream_id", None)) if not isinstance(sid, int) else sid


class TorchCudaPlatform(_TorchAccelBase):
    """真 NVIDIA CUDA。detect 是 compat claim（可能被 masquerade 栈冒充）。"""

    name = "cuda"
    vendor_name = "nvidia"
    device_name = "cuda"
    mod = torch.cuda
    sdk_range = ("11.0", "13.5")

    @classmethod
    def detect(cls):
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available()=False"
        for pkg in _XPU_NATIVE_PACKAGES:
            if importlib.util.find_spec(pkg) is not None:
                return False, f"厂商包 {pkg} 已安装：疑似 masquerade 栈，cuda claim 不成立"
        name = _cuda_device_name()
        if any(k in name for k in _XPU_MARKERS):
            return False, f"CUDA API 存在但卡名 '{name}' 命中 XPU 标记（masquerade，不计为 NVIDIA）"
        return True, f"CUDA device: {name}"


class TorchXpuMasqueradePlatform(_TorchAccelBase):
    """KL3/XPYTORCH 风格：torch.cuda.* 实际跑 XPU。身份必须分清：
    vendor_name=xpu 系（provider 分发用），device_name="cuda"（torch 只认得它）。
    """

    name = "xpu-masquerade"
    vendor_name = "kl3-xpu"
    device_name = "cuda"
    mod = torch.cuda
    masquerades_as = frozenset({"cuda"})

    @classmethod
    def detect(cls):
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available()=False"
        hits = []
        for pkg in _XPU_NATIVE_PACKAGES:
            if importlib.util.find_spec(pkg) is not None:
                hits.append(f"厂商包 {pkg} 已安装（native 通道）")
        name = _cuda_device_name()
        if any(k in name for k in _XPU_MARKERS):
            hits.append(f"卡名 '{name}' 命中 XPU 标记")
        if hits:
            return True, "torch.cuda 背后是 masquerade 栈：" + "；".join(hits)
        return False, f"卡名 '{name}' 无 XPU 标记，厂商包 {list(_XPU_NATIVE_PACKAGES)} 均未安装"


class TorchXpuNativePlatform(_TorchAccelBase):
    """torch.xpu 原生栈（torch>=2.4 内建或 torch_xpu 扩展注册）。"""

    name = "xpu-native"
    vendor_name = "xpu-native"
    device_name = "xpu"
    mod = getattr(torch, "xpu", None)

    @classmethod
    def detect(cls):
        m = getattr(torch, "xpu", None)
        if m is None:
            return False, "torch.xpu 不存在（torch<2.4 或未装 torch_xpu）"
        try:
            ok = bool(m.is_available())
        except Exception as e:
            return False, f"torch.xpu.is_available() 异常: {e}"
        return (True, f"torch.xpu device: {m.get_device_name(0)}") if ok \
            else (False, "torch.xpu.is_available()=False")


class TorchCpuPlatform(PlatformAdapter):
    """本地开发验证专用：无加速卡的机器上跑通 smoke 全链路。

    诚实降级：stream_api / memory_query 声明 False（CPU 没有这些概念），
    门面报 unsupported，框架走降级路径——本身就是能力驱动降级的活样本。
    RNG 用 torch 全局 CPU RNG。同步语义：CPU 执行天然同步，device-wide 平凡成立。
    """

    name = "cpu"
    vendor_name = "cpu-dev"
    device_name = "cpu"
    auto_detect = False  # 只能显式 select("cpu")，不参与自动检测
    mod = None

    @classmethod
    def detect(cls):
        return False, "dev-only 平台，不参与自动检测"

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
        return _rng_to_bytes(torch.get_rng_state())

    def set_rng_state(self, state, device=None):
        torch.set_rng_state(_rng_from_bytes(state))
