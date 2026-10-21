"""模拟 torch.cuda —— 框架直接使用的 CUDA 接口，Bridge 的替换对象。

_hw 为 None 时模拟"这台机器没有 CUDA 卡"（如 xpu-only 机器）：
框架里任何真实 CUDA 调用都会 RuntimeError —— 这就是没有 Bridge 时的现状。
"""
from contextlib import contextmanager

from mock_hw import CudaHw

# CUDA 风格的公共 API 面（盘点范围；辅助函数不在其列）
__all__ = [
    "set_device", "current_device", "device_count", "synchronize",
    "memory_allocated", "can_device_access_peer", "range_push",
    "graph_capture", "profiler_start",
    # P0 扩展（对照 TT_Swift_Cuda_API_Inventory.md）：
    "is_available", "Stream", "current_stream", "stream",
    "get_rng_state", "set_rng_state", "manual_seed", "empty_cache", "mem_get_info",
    "set_sync_debug_mode",  # 故意不纳入 policy：演示默认拒绝
]

_hw = None


def attach_cuda_hw(hw: CudaHw):
    """测试辅助：给有 CUDA 卡的环境挂驱动。"""
    global _hw
    _hw = hw


def _require() -> CudaHw:
    if _hw is None:
        raise RuntimeError("CUDA driver not present on this machine")
    return _hw


def set_device(idx):
    _require().set_device(idx)


def current_device():
    return _require().current_device()


def device_count():
    return _require().device_count()


def synchronize():
    _require().sync_device()


def memory_allocated():
    """状态查询：动态显存水位（只读，答案随分配变化）。"""
    return _require().memory_allocated()


def can_device_access_peer(device, peer_device):
    """对齐真实 torch 签名：两张卡之间能否 P2P（逐对回答）。

    mock 拓扑单一，硬件用统一的 p2p_supported 作答；真实栈应逐对查询。
    """
    return _require().p2p_supported


def range_push(name):
    """行为接口：profiler 区间标注（有副作用，CUDA 侧走 nvtx）。"""
    _require().annotate(name)


def graph_capture(fn):
    """CUDA Graph 加速路径（CUDA 专属能力）。"""
    _require()
    return fn()


def profiler_start():
    """CUDA 专属 profiler 入口。"""
    _require()


def set_sync_debug_mode(mode):
    """CUDA 同步调试开关（故意不纳入 policy，用于演示默认拒绝）。"""
    _require()


# --- P0 扩展（swift 主链路用到的 CUDA 形态） ---

def is_available():
    """分支选择器：swift 用它决定设备字符串形态（rollout.py 形态）。"""
    return _hw is not None


class Stream:
    """CUDA stream 对象（megatron ddp fencing 形态）。"""

    def __init__(self):
        self._sid = _require().create_stream()

    def wait_stream(self, other):
        _require().wait_stream(self._sid, other._sid)


def current_stream():
    s = Stream.__new__(Stream)
    s._sid = _require().current_stream
    return s


@contextmanager
def stream(s):
    """stream 上下文管理器。"""
    hw = _require()
    prev = hw.current_stream
    hw.set_stream(s._sid)
    try:
        yield s
    finally:
        hw.set_stream(prev)


def get_rng_state():
    """checkpoint save：读 RNG 状态。"""
    return _require().get_rng_state()


def set_rng_state(state):
    """checkpoint resume：恢复 RNG 状态。"""
    _require().set_rng_state(state)


def manual_seed(seed):
    """全局 RNG 播种（通信/训练路径真实调用点，megatron 实测 4 处）。"""
    return _require().manual_seed(seed)


def empty_cache():
    """allocator 提示（rollout/转换路径高频调用）。"""
    _require().empty_cache()


def mem_get_info():
    """(free, total) 字节元组（patcher.py 的 max_memory 初始化形态）。"""
    return _require().mem_get_info()
