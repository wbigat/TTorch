"""D9：provider 实现的泛化校验（witness 层，针对问题多的 case）。

分层定位：provider 兜底校验（_admit_native_vs_orig 对账 / _self_check / 门禁
verify_impl）用**单点、轻量**——只确认"这次接管安全"，不可能全面。witness 层
`verify_provider_generalization` 做**更全校验**（边界输入集），手动/e2e 触发。

KLX 教训（fused_bias_swiglu 空张量特判）：MoE 某 expert 分到 0 token 是常态。
"""
import torch

from ttorch.witness.diagnostics import (
    _edge_input_variants, verify_provider_generalization)
from ttorch.provider.impls.bias_swiglu.base import (
    NativeBiasSwiGLU, BiasSwiGLUProvider)


def test_edge_input_variants_covers_boundary():
    """边界变体覆盖特征维 ×2/÷2 + 批次维空/单元素。"""
    x = torch.randn(4, 4096)
    b = torch.randn(4096)
    variants = _edge_input_variants((x, b), {})
    shapes = [tuple(a[0].shape) for a, _ in variants]
    assert len(variants) == 5, "基础 + 特征维×2 + ÷2 + 批次维空 + 单"
    assert (4, 4096) in shapes   # 基础（跟随真实负载）
    assert (4, 8192) in shapes   # 特征维 ×2（D=256→512 静默算错教训）
    assert (4, 2048) in shapes   # 特征维 ÷2
    assert (0, 4096) in shapes   # 批次维空（KLX 空张量特判：MoE 0 token）
    assert (1, 4096) in shapes   # 批次维单元素


def test_bias_swiglu_native_edge_input_correct():
    """bias_swiglu native 在空张量/单元素下正确（KLX 空张量特判）。"""
    native = NativeBiasSwiGLU()
    for batch in (0, 1, 4):
        x = torch.randn(batch, 4096)
        b = torch.randn(4096)
        out = native(x, b)
        assert out.shape == (batch, 2048), f"batch={batch} 输出形状正确且不崩"


def test_verify_provider_generalization_cpu():
    """泛化校验（witness 层）：bias_swiglu native 在边界输入集下 vs 参考一致。"""
    provider = BiasSwiGLUProvider()
    # ref_fn=None：校验 impl vs native；无 impl 时校验 native 自身（边界不崩）
    r = verify_provider_generalization(provider, device="cpu")
    assert "checks" in r and len(r["checks"]) > 0
    # native 自身在边界输入集下应全部正确（不崩、一致）
    assert all(c["ok"] for c in r["checks"]), \
        f"边界校验有失败: {[c for c in r['checks'] if not c['ok']]}"
