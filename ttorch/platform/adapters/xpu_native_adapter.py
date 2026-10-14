"""torch.xpu 原生栈平台实现（惰性加载，激活时才 import）。

骨架实现，未在真实 xpu-native 环境验证（本机是 masquerade 栈）。

继承 TorchDeviceModuleBase 而非 TorchCudaPlatform：它走 torch.xpu 命名空间，
与 CUDA 命名空间专属的（torch.version.cuda、CUDAGraph）无关。

待真实环境校准项：
- torch.version.xpu 是否存在及其取值格式（当前回落 torch 主版本号）；
- sdk_range 的有效区间（现默认不卡）；
- graph 捕获的真实 API 名（conformance 的 graph_capture_roundtrip 现按
  CUDAGraph 实测，torch.xpu 若用别的类名则测不到，确认后补）；
- mem_get_info / get_rng_state 是否齐备（缺失则调用时抛 unsupported，
  conformance 对应测试自动跳过）。
"""
import torch

from ttorch.platform.adapters._device_module import TorchDeviceModuleBase

__all__ = ["TorchXpuNativePlatform"]


class TorchXpuNativePlatform(TorchDeviceModuleBase):
    """torch.xpu 原生栈（torch>=2.4 内建或厂商扩展注册）。"""

    mod_path = "torch.xpu"

    def sdk_version(self):
        """XPU 栈版本：优先 torch.version.xpu，缺失回落 torch 主版本。"""
        return getattr(torch.version, "xpu", None) or torch.__version__.split("+")[0]
