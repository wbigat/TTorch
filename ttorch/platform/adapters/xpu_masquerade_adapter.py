"""KL3/XPYTORCH 风格 masquerade 平台实现（惰性加载，激活时才 import）。

继承 TorchCudaPlatform 是正确的，不是偷懒：该栈背后就是 torch.cuda 命名空间，
厂商把 XPU 全栈伪装成 CUDA——torch.version.cuda 有值、torch.cuda.CUDAGraph
实测可用（KL3 实测 capture+replay 生效）。实现与真 CUDA 无差异，若后续实测
出差异在此覆盖。detect 与身份见同目录 xpu_masquerade.py 的 SPEC。
"""
from ttorch.platform.adapters.cuda_adapter import TorchCudaPlatform

__all__ = ["TorchXpuMasqueradePlatform"]


class TorchXpuMasqueradePlatform(TorchCudaPlatform):
    """实现与真 CUDA 一致（同一 torch.cuda 命名空间），身份由 SPEC 分清。"""
