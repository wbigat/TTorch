"""Platform Adapter 各实现：把统一契约映射到具体硬件的非标接口。

每个 adapter 的职责只有两件：
1. 填 gap——接口名/参数形式/语义范围不一致时，在 adapter 内补齐
   （如 XpuPlatform.synchronize 枚举全部活跃 stream 凑齐 device-wide）；
2. 错误翻译——厂商原生错误（CudaHwError/XpuHwError）-> TTPlatformError。

adapter 不知道也不关心 Manager 的激活门禁；契约履行由 conformance.py
在激活期验证，履行不了的 adapter（见 NaiveXpuPlatform）根本活不进运行期。
"""
from mock_hw import CudaHw, CudaHwError, GcuHw, XpuHw, XpuHwError

from tt_bridge.platform_adapter import PlatformAdapter, TTPlatformError


class CudaPlatform(PlatformAdapter):
    """CUDA 语义与契约一致，adapter 是透传 + 错误映射。

    detect 是 compat claim（CUDA API 存在可能被 GCU 类栈冒充），
    真正的 native 纠正靠冒充分辨机制，见 GcuPlatform。
    """

    name = "cuda"
    vendor_name = "nvidia-cuda"
    device_name = "cuda"
    sdk_range = ("11.0", "13.0")

    @classmethod
    def detect(cls):
        ok = CudaHw.probe_present()
        return ok, "CUDA driver present" if ok else "no CUDA driver"

    def __init__(self):
        self.hw = CudaHw()

    def sdk_version(self):
        return CudaHw.sdk_version

    def capabilities(self):
        # probed 项从硬件实测填充；declared 项是承诺，由 conformance 背书
        return {"device_available": CudaHw.probe_present(),
                "device_count": self.hw.device_count(),
                "p2p": self.hw.p2p_supported,
                "device_wide_sync": True,
                "graph_capture": True,
                "async_comm": True,
                "rng_state": True,
                "stream_api": True,
                "memory_query": True}

    def active_streams(self):
        return list(self.hw.streams)

    def set_device(self, idx):
        try:
            self.hw.set_device(idx)
        except CudaHwError as e:
            raise TTPlatformError("invalid_device", str(e)) from e

    def current_device(self):
        return self.hw.current_device()

    def memory_allocated(self, device=None):
        return self.hw.memory_allocated()

    def mem_get_info(self, device=None):
        return self.hw.mem_get_info()  # 结构/单位本就一致

    def get_rng_state(self, device=None):
        return self.hw.get_rng_state()

    def set_rng_state(self, state, device=None):
        try:
            self.hw.set_rng_state(state)
        except CudaHwError as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    def empty_cache(self):
        self.hw.empty_cache()

    def stream_create(self):
        return self.hw.create_stream()

    def stream_current(self):
        return self.hw.current_stream

    def stream_set(self, sid):
        self.hw.set_stream(sid)

    def stream_wait(self, waiter, waitee):
        self.hw.wait_stream(waiter, waitee)

    def range_push(self, name):
        self.hw.annotate(name)

    def synchronize(self, device=None):
        self.hw.sync_device()


class XpuPlatform(PlatformAdapter):
    """XPU 接口非标，adapter 负责填 gap：

    - set_device: int -> "XPU:n" 字符串；XpuHwError -> TTPlatformError
    - current_device: "XPU:n" -> int
    - synchronize: 硬件 sync() 只等当前 stream -> adapter 枚举全部活跃
      stream 逐一等待，凑齐 device-wide 语义（对外承诺与 CUDA 一致）
    - memory_allocated / range_push: 非标接口名 -> 统一契约
    """

    name = "xpu"
    vendor_name = "kl3-xpu"
    device_name = "xpu"
    sdk_range = ("2.0", "3.0")

    @classmethod
    def detect(cls):
        ok = XpuHw.probe_present()
        return ok, "XPU device nodes present" if ok else "no XPU device nodes"

    def __init__(self):
        self.hw = XpuHw()

    def sdk_version(self):
        return XpuHw.sdk_version

    def capabilities(self):
        return {"device_available": XpuHw.probe_present(),  # probed
                "device_count": self.hw.deviceCount(),   # probed
                "p2p": self.hw.p2p_supported,            # probed（实测 False）
                "device_wide_sync": True,                # declared：adapter 补齐后的承诺
                "graph_capture": False,
                "async_comm": True,
                "rng_state": True,
                "stream_api": True,
                "memory_query": True}

    def active_streams(self):
        return list(self.hw.streams)

    def set_device(self, idx):
        try:
            self.hw.setDevice(f"XPU:{idx}")
        except XpuHwError as e:
            raise TTPlatformError("invalid_device", str(e)) from e

    def current_device(self):
        return int(self.hw.currentDevice().split(":")[1])

    def memory_allocated(self, device=None):
        return self.hw.memUsed()  # 非标接口名 -> 统一契约

    def mem_get_info(self, device=None):
        info = self.hw.memInfo()  # 非标：dict + MiB 单位
        return (info["freeMiB"] * 1024**2, info["totalMiB"] * 1024**2)  # -> 字节 tuple

    def get_rng_state(self, device=None):
        return self.hw.rngState()  # 非标接口名

    def set_rng_state(self, state, device=None):
        try:
            self.hw.setRngState(state)  # 非标接口名
        except XpuHwError as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    def empty_cache(self):
        self.hw.purge()  # 非标接口名

    def stream_create(self):
        return self.hw.create_stream()

    def stream_current(self):
        return self.hw.current_stream

    def stream_set(self, sid):
        self.hw.set_stream(sid)

    def stream_wait(self, waiter, waitee):
        self.hw.streamWait(waiter, waitee)  # 非标接口名

    def range_push(self, name):
        self.hw.profMark(name)  # 非标接口名 -> 统一契约

    def synchronize(self, device=None):
        for s in self.active_streams():
            self.hw.set_stream(s)
            self.hw.sync()


class NaiveXpuPlatform(XpuPlatform):
    """反面教材：不填 gap 的 adapter，直接透传 sync()。

    等待范围从 device-wide 缩成 current-stream —— 不报错、只变慢/出错，
    conformance 门禁必须在激活期拦住它。
    """

    name = "xpu-naive"
    auto_detect = False  # 反面教材/测试夹具：不参与自动检测，只能显式构造

    def capabilities(self):
        caps = dict(super().capabilities())
        caps["device_wide_sync"] = False  # 诚实声明（但仍过不了强制契约测试）
        return caps

    def synchronize(self, device=None):
        self.hw.sync()  # 只等当前 stream


class GcuPlatform(CudaPlatform):
    """GCU 栈：CUDA 兼容（语义标准，继承 CudaPlatform 全部映射），但身份必须分清。

    - vendor_name="gcu"：provider 分发 / provenance 认这个，厂商 kernel
      不能因为它冒充 CUDA 就按 NVIDIA 发
    - device_name="cuda"：torch 只认得 cuda，框架里 device.type 比较
      必须继续成立，否则优化器等代码静默失效（FlagOS 真坑）
    - masquerades_as={"cuda"}：native 探测命中后，cuda 的 compat claim
      被降级为"我的冒充证据"，不再构成多平台冲突
    """

    name = "gcu"
    vendor_name = "gcu"
    device_name = "cuda"
    sdk_range = ("4.0", "5.0")
    masquerades_as = frozenset({"cuda"})

    @classmethod
    def detect(cls):
        ok = GcuHw.probe_present()
        return ok, "GCU native device nodes present" if ok else "no GCU native nodes"

    def __init__(self):
        self.hw = GcuHw()

    def sdk_version(self):
        return GcuHw.sdk_version
