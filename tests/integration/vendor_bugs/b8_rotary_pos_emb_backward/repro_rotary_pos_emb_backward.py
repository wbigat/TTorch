"""最小复现：custom_ops.rotary_pos_emb_backward 在部分旋转下反向计算错误。

**零 TT 依赖**——只用 torch + 厂商栈自带包，可直接发给厂商复现。

运行：python tests/integration/vendor_bugs/b8_rotary_pos_emb_backward/repro_rotary_pos_emb_backward.py
期望输出：全旋转 fwd/bwd 均正常；部分旋转 fwd 正常、bwd 误差 ~2 倍梯度量级。

参考值用纯 torch 手写的 rotate_half 公式（非交错 RoPE，mscale=1），
与 megatron native 的语义一致但与其代码零共享——排除"参考实现版本差异"争议。

2026-11-05 从 experiments/b8_rope_backward/ 归档至 tests/integration/vendor_bugs/，
供与厂商对齐时直接引用；详见同目录 BUG_REPORT.md。
"""
import torch

from xmegatron_ext.components.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd as fused,
)


def reference_rope(t, freqs):
    """非交错 RoPE 纯 torch 参考实现（与 megatron rope_utils 同公式）。"""
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    x1, x2 = torch.chunk(t_rot, 2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return torch.cat((t_rot * cos_ + rot * sin_, t_pass), dim=-1)


def trial(s, h, d, rot, dtype, noncontig):
    torch.manual_seed(0)
    base = torch.randn(s, 2, h, d, dtype=dtype, device="cuda")
    t0 = base[:, :1] if noncontig else base[:, :1].contiguous()
    freqs = torch.randn(s, 1, 1, rot, dtype=torch.float32, device="cuda")

    t_fused = t0.clone().requires_grad_(True)
    t_ref = t0.clone().requires_grad_(True)
    out_fused = fused(t_fused, freqs)
    out_ref = reference_rope(t_ref, freqs)
    g = torch.randn_like(out_fused)
    out_fused.backward(g)
    out_ref.backward(g)

    fwd_err = (out_fused.float() - out_ref.float()).abs().max().item()
    # rot / pass 分段误差（2026-11-04 定位结论：pass 段恒等正确，rot 段错）
    g_err = (t_fused.grad.float() - t_ref.grad.float()).abs()
    bwd_err = g_err.max().item()
    rot_err = g_err[..., :rot].max().item() if rot < d else bwd_err
    pass_err = g_err[..., rot:].max().item() if rot < d else 0.0
    scale = t_ref.grad.float().abs().max().item()
    tag = f"s={s} h={h} d={d} rot={rot} {str(dtype).split('.')[-1]:8s} " \
          f"{'非连续' if noncontig else 'contig '}"
    # dtype 相关的精度地板：fp32 取 1e-3 rel，bf16 取 3e-2 rel（bf16 eps≈7.8e-3，
    # 量级 5 时单 ulp≈0.03）——ulp 噪声不得误报 BUG，逻辑错（≈2×量级）必然越线
    rel_tol = 3e-2 if dtype == torch.bfloat16 else 1e-3
    verdict = "OK " if bwd_err < rel_tol * max(scale, 1.0) else "BUG"
    seg = f" rot_err={rot_err:.3e} pass_err={pass_err:.3e}" if rot < d else ""
    print(f"[{verdict}] {tag} fwd_err={fwd_err:.3e}  "
          f"bwd_err={bwd_err:.3e}  (grad_scale={scale:.2e}){seg}")
    return verdict == "BUG"


if __name__ == "__main__":
    print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))
    print(f"{'=' * 78}")
    bugs = []
    bugs.append(trial(128, 4, 64, 64, torch.float32, False))   # 全旋转 fp32
    bugs.append(trial(128, 4, 64, 64, torch.float32, True))    # 全旋转 fp32 非连续
    bugs.append(trial(128, 4, 64, 32, torch.float32, False))   # 部分旋转 fp32
    bugs.append(trial(128, 4, 64, 32, torch.float32, True))    # 部分旋转 fp32 非连续
    bugs.append(trial(128, 4, 64, 32, torch.bfloat16, False))  # 部分旋转 bf16
    bugs.append(trial(128, 4, 64, 64, torch.bfloat16, False))  # 全旋转 bf16
    print(f"{'=' * 78}")
    n = sum(bugs)
    print(f"结论：{n}/6 组合反向错误" if n else "结论：全部通过")
    raise SystemExit(1 if n else 0)
