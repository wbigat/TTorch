"""CPU dev 平台实现（惰性加载，激活时才 import）。

本地开发验证专用：无加速卡的机器上跑通 smoke 全链路。直接继承 PlatformBase，
不走 TorchDeviceModuleBase——CPU 没有 torch 设备模块那套 API，只共用 _rng
的类型契约辅助。
"""
import torch

from ttorch.platform.platform_handle import TTPlatformError
from ttorch.platform.adapters._rng import rng_from_state, rng_to_state
from ttorch.platform.platform_base import PlatformBase

__all__ = ["TorchCpuPlatform"]


class TorchCpuPlatform(PlatformBase):
    """本地开发验证专用。

    诚实降级：stream 族 / 显存查询索性不实现（CPU 没有这些概念）——
    PlatformBase 的默认实现会抛 unsupported，框架走降级路径。
    "不实现"就是"不支持"的全部表达，不需要另外声明能力表。
    RNG 用 torch 全局 CPU RNG。同步语义：CPU 执行天然同步，device-wide 平凡成立。
    """

    def sdk_version(self):
        return torch.__version__.split("+")[0]

    def is_available(self):
        return True   # CPU 总是在场

    def device_count(self):
        return 1

    def set_device(self, device):
        if device != 0:
            raise TTPlatformError("invalid_device", f"CPU 平台只有 device 0，收到 {device}")

    def current_device(self):
        return 0

    def synchronize(self, device=None):
        pass  # CPU 天然同步

    def empty_cache(self):
        pass

    def get_rng_state(self, device=None):
        return rng_to_state(torch.get_rng_state())

    def set_rng_state(self, state, device=None):
        torch.set_rng_state(rng_from_state(state))
