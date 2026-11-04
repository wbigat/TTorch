"""verify_impl 的生产 dtype 精度轨道回归（A2，复核 20261028 §4.2）。

背景：旧实现把输入升 fp32 再比较，改变的是 kernel 实际走的执行路径——厂商
kernel 常按 dtype 分派，生产用的那份从未被门禁调用。实测反例：候选 fp32 正确、
bf16 返回两倍，门禁仍 pass 且 max_abs_err=0.0（bf16 路径被性能轨道执行 210 次，
一次都没看输出）。本文件把该反例入库。
"""
import torch

from ttorch.witness.verify import _LOWP_REL_TOL, _rel_tol, verify_impl


def _native(x):
    return x.square()


def _probe_bf16():
    return (torch.randn(8, 16, dtype=torch.bfloat16),), {}


def test_rejects_candidate_wrong_only_in_bf16():
    """候选 fp32 正确、bf16 翻倍：必须被拒（旧实现给 pass）。"""
    def impl(x):
        return x.square() * 2 if x.dtype == torch.bfloat16 else x.square()

    ok, reason, _ = verify_impl(impl, _native, _probe_bf16, enforce_benefit=False)
    assert ok is False, f"bf16 下翻倍的候选被放行: {reason}"
    assert reason.startswith("precision:"), reason


def test_impl_sees_production_dtype_in_precision_track():
    """精度轨道必须以 probe 的原始 dtype 调用候选，不得升 fp32。"""
    seen = []

    def impl(x):
        seen.append(x.dtype)
        return x.square()

    verify_impl(impl, _native, _probe_bf16, enforce_benefit=False)
    assert seen, "候选未被调用"
    assert seen[0] == torch.bfloat16, f"精度轨道首次调用 dtype={seen[0]}，应为 bfloat16"
    assert torch.float32 not in seen, "出现 fp32 调用，说明仍在升精度轨道"


def test_accepts_same_semantic_bf16_noise():
    """同语义、仅内部规约顺序不同的 bf16 实现：ulp 级噪声不应被误拒。"""
    def impl(x):
        # 先升 fp32 算再降回——与 native 数学同义，仅舍入路径不同
        return (x.float() * x.float()).to(x.dtype)

    ok, reason, ev = verify_impl(impl, _native, _probe_bf16, enforce_benefit=False)
    assert ok is True, reason
    assert ev["probe_dtype"] == "torch.bfloat16"
    assert ev["rel_tol"] == _LOWP_REL_TOL[torch.bfloat16]


def test_fp32_track_keeps_original_strictness():
    """fp32 下沿用调用方阈值（不被低精度放宽波及）。"""
    assert _rel_tol(torch.float32, 5e-3) == 5e-3
    assert _rel_tol(torch.bfloat16, 5e-3) == _LOWP_REL_TOL[torch.bfloat16]

    def impl(x):
        return x.square() * 1.05  # 5% 偏差，fp32 阈值下必须被拒

    ok, reason, _ = verify_impl(impl, _native,
                                lambda: ((torch.randn(8, 16),), {}),
                                enforce_benefit=False)
    assert ok is False, f"fp32 下 5% 偏差被放行: {reason}"


def test_near_zero_reference_reported_as_infra():
    """基准输出近零：相对判据无判别力，必须诚实报 gate_infra 而非混成 pass。"""
    ok, reason, _ = verify_impl(lambda x: x * 0, lambda x: x * 0,
                                lambda: ((torch.zeros(4, 8),), {}),
                                enforce_benefit=False)
    assert ok is False
    assert reason.startswith("gate_infra:"), reason
    assert "判别力" in reason


def test_rejects_bf16_grad_error():
    """前向在 bf16 下正确、反向梯度错：梯度也走生产 dtype 轨道，必须被拒。"""
    class _BadGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.square()

        @staticmethod
        def backward(ctx, go):
            (x,) = ctx.saved_tensors
            return go * 2 * x * 3  # 梯度放大 3 倍

    ok, reason, _ = verify_impl(_BadGrad.apply, _native, _probe_bf16,
                                enforce_benefit=False)
    assert ok is False, f"bf16 梯度错的候选被放行: {reason}"
    assert "precision(bwd)" in reason, reason
