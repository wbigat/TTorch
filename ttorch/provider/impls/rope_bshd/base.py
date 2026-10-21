"""RoPE（bshd 布局）函数语义的 provider（第一层）与 native 默认实现。

语义：对 t[..., :rot_dim] 应用旋转位置编码（rot_dim = freqs.shape[-1]），
t[..., rot_dim:] 直通；mscale != 1.0 时带缩放。
调用约定（对齐 megatron `_apply_rotary_pos_emb_bshd`）：
    (t, freqs, rotary_interleaved=False, multi_latent_attention=False, mscale=1.0)

函数级接入（bind_points）：megatron.core.models.common.embeddings.rope_utils。
"""
import torch

from ttorch.provider import bind_targets
from ttorch.provider.spec import Provider, ProviderImpl


def _rotate_half(x, rotary_interleaved):
    """与 megatron rope_utils._rotate_half 逐字一致（[-odd, +even]）。"""
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


class NativeRoPEBshd(ProviderImpl):
    """torch 原生公式（与 megatron 逐字一致）：功能底线 + 精度/收益基准。"""

    name = "native:rope_bshd"
    kind = "reference"
    intent = "torch 原生 RoPE 公式（megatron 逐字）；功能底线 + 精度/收益基准"

    def __call__(self, t, freqs, rotary_interleaved=False,
                 multi_latent_attention=False, mscale=1.0):
        rot_dim = freqs.shape[-1]
        t_r, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        if multi_latent_attention:
            x1 = t_r[..., 0::2]
            x2 = t_r[..., 1::2]
            t_r = torch.cat((x1, x2), dim=-1)
        cos_ = (torch.cos(freqs) * mscale).to(t_r.dtype)
        sin_ = (torch.sin(freqs) * mscale).to(t_r.dtype)
        t_r = (t_r * cos_) + (_rotate_half(t_r, rotary_interleaved) * sin_)
        return torch.cat((t_r, t_pass), dim=-1)


class RoPEBshdProvider(Provider):
    """RoPE（bshd）的高性能 provider：native（默认）+ 各硬件融合实现。"""

    semantic = "rope_bshd"
    slot = "compute"
    module_names = ()
    bind_points = bind_targets.bind_points("megatron", "rope_bshd")

    def build_native(self):
        return NativeRoPEBshd()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        """探测输入：跟随首次真实调用现场（shape/dtype/device/标量参数一致）。"""
        if ref is not None:
            args, kw = ref
            return (torch.randn_like(args[0]),
                    torch.randn_like(args[1])) + tuple(args[2:]), dict(kw)
        if device is None:
            device = "cpu"
        return (torch.randn(8, 2, 4, 64, device=device),
                torch.randn(8, 1, 1, 32, device=device)), {}
