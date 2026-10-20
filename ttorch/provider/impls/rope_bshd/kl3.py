"""KL3（xpu-masquerade）的厂商融合 RoPE（第二层）。

厂商 kernel：torch.ops.custom_ops.rotary_pos_emb(+_backward)。

已知问题（2026-10-15 实测）：厂商 backward 与 megatron 原生公式**语义不等价**
（fwd fp32 逐位一致 2.4e-7，bwd 梯度错 2.43，疑似 rotate_half 的 chunk/
interleaved 约定在反向不一致）——verify_impl 的 bwd 对账会拒绝本 impl，
回退 native（正确姿态）。厂商修复 kernel 后无需改代码，门禁自动放行。
"""
import torch

from ttorch.provider.spec import ProviderImpl


class _RotaryPosEmbFn(torch.autograd.Function):
    """RoPE 的可训练封装：forward/backward 都走厂商 kernel。"""

    @staticmethod
    def forward(ctx, t, freqs):
        output = t.new_empty(t.shape)
        torch.ops.custom_ops.rotary_pos_emb(t, freqs, out=output)
        ctx.freqs = freqs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_t = grad_output.new_empty(grad_output.shape)
        torch.ops.custom_ops.rotary_pos_emb_backward(grad_output, ctx.freqs, out=grad_t)
        return grad_t, None


class KL3FusedRoPEBshd(ProviderImpl):
    """KL3 厂商融合 RoPE（可训练）：custom_ops.rotary_pos_emb(+_backward)。"""

    name = "vendor-fused:rope_bshd"
    kind = "fused_aot_vendor"
    platforms = frozenset({"xpu-masquerade"})
    min_speedup = 1.0
    intent = "KL3 厂商融合 RoPE；注意 bwd kernel 与原生不等价（门禁会拒，见模块 docstring）"

    def supports(self, t, freqs, rotary_interleaved=False,
                 multi_latent_attention=False, mscale=1.0):
        # 厂商 kernel 只覆盖主路径；其余回退 native（原生公式含 mscale/重排）
        return mscale == 1.0 and not rotary_interleaved and not multi_latent_attention

    def __call__(self, t, freqs, rotary_interleaved=False,
                 multi_latent_attention=False, mscale=1.0):
        rot_dim = freqs.shape[-1]
        t_r, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        t_r = _RotaryPosEmbFn.apply(t_r.contiguous(), freqs)
        return torch.cat((t_r, t_pass), dim=-1)
