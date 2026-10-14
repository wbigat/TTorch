"""mock 硬件的 PlatformBase 实现：把 torch 接口映射到假硬件的非标接口。

每个实现的职责只有两件：
1. 填 gap——接口名/参数形式/语义范围不一致时在实现内补齐
   （如 XpuPlatform.synchronize 枚举全部活跃 stream 凑齐 device-wide）；
2. 错误翻译——厂商原生错误（CudaHwError/XpuHwError）-> TTPlatformError。

实现类不知道也不关心激活门禁；契约履行由 conformance.py 在激活期验证，
履行不了的实现（见 NaiveXpuPlatform）根本活不进运行期。

身份与 detect 不在实现类里——实现类不自报身份（防冒充），身份元数据与
detect 由文件末尾的 PlatformSpec 承载（与真实平台 spec.py 同口径），由
registry.create 装配进 PlatformHandle。

方法名对齐 torch.cuda 真实符号（set_device/current_device/synchronize/
Stream/current_stream/set_stream/...），见 ttorch.platform.platform_base。
`range_push` 是 mock 的**扩展接口**（torch.cuda 下无此符号，真实位置是
torch.cuda.nvtx.range_push）——不占 PlatformBase 契约，但 manager 的动态
转发照样能把它暴露成 interface.range_push 供 mock policy 绑定。
"""
from contextlib import contextmanager

from mock_hw import CudaHw, CudaHwError, GcuHw, XpuHw, XpuHwError

from ttorch.platform.platform_base import PlatformBase
from ttorch.platform.spec import PlatformSpec
from tt_bridge.platform_adapter import TTPlatformError


@contextmanager
def _stream_ctx(hw, stream):
    """mock 的 StreamContext 等价物：进入切流、退出恢复。

    真实栈直接转发厂商 `mod.stream(s)`；mock 硬件没有这个概念，于是在实现侧
    补齐——"填 gap"本就是 adapter 的职责，不该上移到门面。
    """
    prev = hw.current_stream
    hw.set_stream(stream)
    try:
        yield stream
    finally:
        hw.set_stream(prev)


class CudaPlatform(PlatformBase):
    """CUDA 语义与契约一致，实现是透传 + 错误映射。"""

    sdk_range = ("11.0", "13.0")

    def __init__(self):
        self.hw = CudaHw()

    # --- 平台自述 ---
    def sdk_version(self):
        return CudaHw.sdk_version

    def is_available(self):
        return CudaHw.probe_present()

    def device_count(self):
        return self.hw.device_count()

    def can_device_access_peer(self, device, peer_device):
        return self.hw.p2p_supported

    # --- 设备 ---
    def set_device(self, device):
        try:
            self.hw.set_device(device)
        except CudaHwError as e:
            raise TTPlatformError("invalid_device", str(e)) from e

    def current_device(self):
        return self.hw.current_device()

    def synchronize(self, device=None):
        self.hw.sync_device()

    # --- 显存 ---
    def empty_cache(self):
        self.hw.empty_cache()

    def memory_allocated(self, device=None):
        return self.hw.memory_allocated()

    def mem_get_info(self, device=None):
        return self.hw.mem_get_info()  # 结构/单位本就一致

    # --- RNG ---
    def get_rng_state(self, device=None):
        return self.hw.get_rng_state()

    def set_rng_state(self, state, device=None):
        try:
            self.hw.set_rng_state(state)
        except CudaHwError as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    # --- stream（返回 mock 原生 stream 对象 MockStream，门面不包装） ---
    def Stream(self, **kwargs):  # noqa: N802  # 对齐 torch.cuda.Stream 符号名
        return self.hw.create_stream()  # mock 无厂商内部构造场景，kwargs 不适用

    def current_stream(self, device=None):
        return self.hw.current_stream

    def set_stream(self, stream):
        self.hw.set_stream(stream)

    def stream(self, stream):
        """mock 硬件没有 StreamContext，实现侧自己填 gap（这正是 adapter 的活）。"""
        return _stream_ctx(self.hw, stream)

    # --- 扩展接口（非 torch.cuda 符号；mock policy 绑定它）---
    def range_push(self, name):
        self.hw.annotate(name)


class XpuPlatform(PlatformBase):
    """XPU 接口非标，实现负责填 gap：

    - set_device: int -> "XPU:n" 字符串；XpuHwError -> TTPlatformError
    - current_device: "XPU:n" -> int
    - synchronize: 硬件 sync() 只等当前 stream -> 枚举全部活跃 stream 逐一
      等待，凑齐 device-wide 语义（对外承诺与 CUDA 一致）
    - memory_allocated / range_push: 非标接口名 -> 统一契约
    """

    sdk_range = ("2.0", "3.0")

    def __init__(self):
        self.hw = XpuHw()

    # --- 平台自述 ---
    def sdk_version(self):
        return XpuHw.sdk_version

    def is_available(self):
        return XpuHw.probe_present()

    def device_count(self):
        return self.hw.deviceCount()   # 非标接口名 -> 统一契约

    def can_device_access_peer(self, device, peer_device):
        return self.hw.p2p_supported   # 实测 False

    # --- 设备 ---
    def set_device(self, device):
        try:
            self.hw.setDevice(f"XPU:{device}")
        except XpuHwError as e:
            raise TTPlatformError("invalid_device", str(e)) from e

    def current_device(self):
        return int(self.hw.currentDevice().split(":")[1])

    def synchronize(self, device=None):
        # 硬件 sync() 只等当前 stream，枚举补齐 device-wide。枚举不得污染
        # 当前流：保存进入值、收尾恢复（门面透传原生对象，current_stream
        # 残留成别的流会让框架后续 fencing 落错地方）。
        cur = self.hw.current_stream
        for s in list(self.hw.streams):
            self.hw.set_stream(s)
            self.hw.sync()
        self.hw.set_stream(cur)

    # --- 显存 ---
    def empty_cache(self):
        self.hw.purge()  # 非标接口名

    def memory_allocated(self, device=None):
        return self.hw.memUsed()  # 非标接口名 -> 统一契约

    def mem_get_info(self, device=None):
        info = self.hw.memInfo()  # 非标：dict + MiB 单位
        return (info["freeMiB"] * 1024**2, info["totalMiB"] * 1024**2)  # -> 字节 tuple

    # --- RNG ---
    def get_rng_state(self, device=None):
        return self.hw.rngState()  # 非标接口名

    def set_rng_state(self, state, device=None):
        try:
            self.hw.setRngState(state)  # 非标接口名
        except XpuHwError as e:
            raise TTPlatformError("invalid_rng_state", str(e)) from e

    # --- stream ---
    def Stream(self, **kwargs):  # noqa: N802
        return self.hw.create_stream()

    def current_stream(self, device=None):
        return self.hw.current_stream

    def set_stream(self, stream):
        self.hw.set_stream(stream)

    def stream(self, stream):
        return _stream_ctx(self.hw, stream)

    # --- 扩展接口 ---
    def range_push(self, name):
        self.hw.profMark(name)  # 非标接口名 -> 统一契约


class NaiveXpuPlatform(XpuPlatform):
    """反面教材：不填 gap 的实现，直接透传 sync()。

    等待范围从 device-wide 缩成 current-stream —— 不报错、只变慢/出错，
    conformance 门禁必须在激活期拦住它。auto_detect=False 在 spec
    （NAIVE_XPU_SPEC）：不参与自动检测，只能显式 select。
    """

    def synchronize(self, device=None):
        self.hw.sync()  # 只等当前 stream


class GcuPlatform(CudaPlatform):
    """GCU 栈：CUDA 兼容（语义标准，继承 CudaPlatform 全部映射），身份见 GCU_SPEC。

    - vendor_name="gcu"：provider 分发 / provenance 认这个，厂商 kernel
      不能因为它冒充 CUDA 就按 NVIDIA 发
    - device_name="cuda"：torch 只认得 cuda，框架里 device.type 比较
      必须继续成立——写成品牌名会让这类比较静默失效（不报错、行为错）
    - masquerades_cuda=True：native 探测命中后，cuda 的 compat claim
      被降级为"我的冒充证据"，不再构成多平台冲突
    """

    sdk_range = ("4.0", "5.0")

    def __init__(self):
        self.hw = GcuHw()

    def sdk_version(self):
        return GcuHw.sdk_version


# ------------------------------------------------------------------
# 平台规格（管理面元数据 + detect）：注册用，身份的唯一真相。
# mock 无厂商包副作用，spec 与实现同文件内聚；detect 只看不初始化。
# ------------------------------------------------------------------

def _detect_cuda():
    ok = CudaHw.probe_present()
    return ok, "CUDA driver present" if ok else "no CUDA driver"


def _detect_xpu():
    ok = XpuHw.probe_present()
    return ok, "XPU device nodes present" if ok else "no XPU device nodes"


def _detect_gcu():
    ok = GcuHw.probe_present()
    return ok, "GCU native device nodes present" if ok else "no GCU native nodes"


CUDA_SPEC = PlatformSpec(
    name="cuda", vendor_name="nvidia-cuda", device_name="cuda",
    detect=_detect_cuda, base_cls=CudaPlatform)

XPU_SPEC = PlatformSpec(
    name="xpu", vendor_name="kl3-xpu", device_name="xpu",
    detect=_detect_xpu, base_cls=XpuPlatform)

GCU_SPEC = PlatformSpec(
    name="gcu", vendor_name="gcu", device_name="cuda",
    detect=_detect_gcu, base_cls=GcuPlatform,
    masquerades_cuda=True)

NAIVE_XPU_SPEC = PlatformSpec(
    name="xpu-naive", vendor_name="kl3-xpu", device_name="xpu",
    detect=_detect_xpu, base_cls=NaiveXpuPlatform, auto_detect=False)
