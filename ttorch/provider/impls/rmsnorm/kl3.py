"""KL3（xpu-masquerade）的厂商 AOT 融合 RMSNorm 实现（第二层）。

实测（2026-10-14，KL3）：前向对**未融合 native 分解** ~2.5x，精度 bf16 err=0。
注意性能语境：KL3 的 torch_xmlir 已把 transformers 的 RMSNorm 符号替换成它自己的
融合实现（RMSNormFunction，同源 rms_layer_norm kernel），所以整网上本 provider
与厂商现状**持平（~1x）**——它的价值是统一的可插拔机制（门禁/回退/provenance/
开关），以及在厂商**未**预融合某算子的硬件上补上融合。

训练态必须带 autograd：rms_layer_norm 是纯前向推理 kernel（输出无 grad_fn），
直接用于训练会把梯度图在 RMSNorm 处切断（loss 不学）。故包一个
autograd.Function——forward 走快 kernel，backward 走厂商反向 kernel
rms_layer_norm_backward。
"""
import torch

from ttorch.provider.spec import ProviderImpl

_EPS = 1e-6


class _RMSNormFn(torch.autograd.Function):
    """RMSNorm 的可训练封装：forward/backward 都走厂商 AOT kernel。"""

    @staticmethod
    def forward(ctx, x, w, eps):
        y = torch.empty_like(x)
        rstd = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
        torch.ops.custom_ops.rms_layer_norm(x, float(eps), False, w, y=y, rstd=rstd)
        ctx.save_for_backward(x, w, rstd)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        torch.ops.custom_ops.rms_layer_norm_backward(
            dy.contiguous(), x, rstd, ctx.eps, False, w, dx=dx, dweight=dw)
        return dx, dw, None  # eps 不需要梯度


class KL3FusedRMSNorm(ProviderImpl):
    """KL3 厂商融合 RMSNorm（可训练）：torch.ops.custom_ops.rms_layer_norm(+_backward)。

    填 gap：厂商是 out 变体签名 `rms_layer_norm(x, eps, is_rstd, scale, *, y, rstd)`，
    这里包成干净的 `(x, w, eps) -> y`（rstd 是该栈要求的输出侧缓冲，
    shape=x.shape[:-1]，float32）。算子引用写在方法体内，保持非 KL3 机器 import 安全。
    """

    name = "vendor-fused:rmsnorm"
    kind = "fused_aot_vendor"
    platforms = frozenset({"xpu-masquerade"})
    intent = "KL3 厂商 AOT 融合 RMSNorm（forward+backward kernel）；前向 vs 未融合 native 2.5x"

    def __call__(self, x, w, eps=_EPS):
        return _RMSNormFn.apply(x, w, eps)
