"""bias_swiglu 函数语义的 provider（第一层）与 native 默认实现。

语义：out = silu((x+bias)1) * (x+bias)2（加 bias 后按最后一维劈半做 SwiGLU）
调用约定：(input: Tensor[..., 2D], bias: Tensor[D] | None, ...) -> Tensor[..., D]

函数级接入（bind_points）：megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl。
"""
import torch
import torch.nn.functional as F

from ttorch.provider.spec import Provider, ProviderImpl


class NativeBiasSwiGLU(ProviderImpl):
    """torch 原生分解（与 megatron 公式逐字一致）：功能底线 + 精度/收益基准。"""

    name = "native:bias_swiglu"
    kind = "reference"
    intent = "torch 原生分解 (x+bias) 后 silu(y1)*y2；功能底线 + 精度/收益基准"

    def __call__(self, input, bias, fp8_input_store=False, cpu_offload_input=False):
        y = input + bias if bias is not None else input
        y1, y2 = torch.chunk(y, 2, -1)
        return F.silu(y1) * y2


class BiasSwiGLUProvider(Provider):
    """bias SwiGLU 的高性能 provider：native（默认）+ 各硬件融合实现。"""

    semantic = "bias_swiglu"
    slot = "compute"
    module_names = ()
    bind_points = (("megatron.core.fusions.fused_bias_swiglu", "bias_swiglu_impl"),)

    def build_native(self):
        return NativeBiasSwiGLU()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        """探测输入：跟随首次真实调用现场（shape/dtype/device/标量参数一致）。"""
        if ref is not None:
            args, kw = ref
            x, b = args[0], args[1]
            return (torch.randn_like(x),
                    torch.randn_like(b) if b is not None else None) + tuple(args[2:]), dict(kw)
        if device is None:
            device = "cpu"
        return (torch.randn(4, 4096, device=device),
                torch.randn(4096, device=device)), {}
