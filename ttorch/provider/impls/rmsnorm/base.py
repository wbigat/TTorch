"""RMSNorm 模块语义的 provider（第一层）与 native 默认实现。

RMSNorm 语义：y = x * rsqrt(mean(x^2, dim=-1) + eps) * w
调用约定：(x: Tensor[S, D], w: Tensor[D], eps: float) -> Tensor[S, D]

native 全硬件共享一份（torch 原生单算子流程），是功能底线 + 精度/收益基准；
各硬件的高性能实现作为 ProviderImpl 子类放在同文件夹的 <平台>.py（如 kl3.py）。
"""
import torch

from ttorch.provider.spec import Provider, ProviderImpl

_EPS = 1e-6


class NativeRMSNorm(ProviderImpl):
    """torch 原生单算子流程（native）：x * rsqrt(mean(x^2)+eps) * w。

    永远可用、永远启用、不可禁用——摘掉全部高性能实现后的功能底线，
    同时是门禁的精度/收益基准。
    """

    name = "native:rmsnorm"
    kind = "reference"
    intent = "torch 原生单算子流程；功能底线 + 精度/收益基准"

    def __call__(self, x, w, eps=_EPS):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + eps) * w


class RMSNormProvider(Provider):
    """RMSNorm 模块的高性能 provider：native（默认）+ 各硬件高性能实现。"""

    semantic = "rmsnorm"
    slot = "compute"
    module_names = ("RMSNorm",)  # Qwen2RMSNorm / Qwen3RMSNorm / LlamaRMSNorm 等

    def build_native(self):
        return NativeRMSNorm()

    @classmethod
    def probe_input(cls):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        x = torch.randn(4096, 4096, device=dev)
        w = torch.ones(4096, device=dev)
        return (x, w), {}
