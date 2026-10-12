"""模拟厂商 custom_ops 命名空间（torch.ops.custom_ops 的 demo 对应物）。

真实形态见 xmegatron_ext/components/core/models/common/embeddings/rope_utils.py：
RotaryPositionalEmbeddingWithFreqFunction 内调 torch.ops.custom_ops.rotary_pos_emb
——厂商编译的 AOT kernel，黑盒、平台限定、有隐含适用条件（融合路径只在
mscale==1.0 时被选中，见 _apply_rotary_pos_emb_bshd 的分支）。

数值语义与 reference 一致；本 mock 只保留结构：fwd/bwd 成对、out= 传参风格。
"""
import math


def rotary_pos_emb(t, freqs, out=None):
    """fused RoPE 前向（mock）。t/freqs 为等长 float list（单行演示）。

    真实 kernel 是 inplace out= 写回；demo 返回 list 以便对账。
    """
    half = len(freqs) // 2
    out = out if out is not None else [0.0] * len(t)
    for i, v in enumerate(t):
        c, s = math.cos(freqs[i]), math.sin(freqs[i])
        rot = -t[i + half] if i < half else t[i - half]  # rotate_half
        out[i] = v * c + rot * s
    return out


def rotary_pos_emb_backward(grad_output, freqs, out=None):
    """fused RoPE 反向（mock）：与前向同构（RoPE 的 Jacobian 仍是旋转）。"""
    return rotary_pos_emb(grad_output, freqs, out=out)
