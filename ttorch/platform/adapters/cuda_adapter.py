"""真 NVIDIA CUDA 平台实现（惰性加载：激活时才 import，可自由 import 厂商包）。

detect 与 SPEC 在同目录 cuda.py（注册单元，import 安全）。

本文件承载 CUDA 命名空间专属的部分：SDK 版本来源 torch.version.cuda。
通用范式在 _device_module.TorchDeviceModuleBase。

graph 捕获不再需要在此声明（2026-10-14）：conformance 的
graph_capture_roundtrip 直接实测 mod 上的 CUDAGraph 能否捕获+重放，
结果即事实，不需要平台先自报一个 graph_capture=True 再去验证。
"""
import torch

from ttorch.platform.adapters._device_module import TorchDeviceModuleBase

__all__ = ["TorchCudaPlatform"]


class TorchCudaPlatform(TorchDeviceModuleBase):
    """真 NVIDIA CUDA（实现本体）。"""

    mod_path = "torch.cuda"
    sdk_range = ("11.0", "13.5")

    def sdk_version(self):
        return getattr(torch.version, "cuda", None) or "0.0"
