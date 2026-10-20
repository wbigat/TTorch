"""SwiGLU 函数语义的 provider（第一层）与 native 默认实现。

语义：out = silu(y1) * y2（y 按最后一维劈半）
调用约定：(y: Tensor[..., 2D]) -> Tensor[..., D]

函数级接入（bind_points）：megatron 的 MLP.forward 里 swiglu 是函数调用而非
nn.Module——绑定后框架符号 = dispatcher，调用经 manager.call 分发，与模块级
共享门禁/回退/开关/provenance。
"""
import torch
import torch.nn.functional as F

from ttorch.provider.spec import Provider, ProviderImpl


class NativeSwiGLU(ProviderImpl):
    """torch 原生分解（与 megatron 公式逐字一致）：功能底线 + 精度/收益基准。"""

    name = "native:swiglu"
    kind = "reference"
    intent = "torch 原生分解 silu(y1)*y2；功能底线 + 精度/收益基准"

    def __call__(self, y):
        y1, y2 = torch.chunk(y, 2, -1)
        return F.silu(y1) * y2


class SwiGLUProvider(Provider):
    """SwiGLU 激活的高性能 provider：native（默认）+ 各硬件融合实现。"""

    semantic = "swiglu"
    slot = "compute"
    module_names = ()
    bind_points = (("megatron.core.fusions.fused_bias_swiglu", "swiglu"),)

    def build_native(self):
        return NativeSwiGLU()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        """探测输入：跟随首次真实调用现场（shape/dtype/device 与生产一致）。"""
        if ref is not None:
            return (torch.randn_like(ref[0][0]),), {}
        if device is None:
            device = "cpu"
        return (torch.randn(4, 4096, device=device),), {}
