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


def test_fx_ssa_compare():
    """fx SSA 校验：分解 module 结构等价 / zcg 识别语义差异 / 融合不适用。

    设计（2026-10-23）：fx SSA 对比分解 vs 分解（外部框架实现 vs provider native，
    都是小算子）；融合算子（transformers custom_ops）语义等价需开发者确认 +
    精度兜底，SSA 不适用。
    """
    from ttorch.provider.impls.rmsnorm.base import RMSNormProvider
    provider = RMSNormProvider()
    x = torch.randn(4, 256)

    class DecompRMSNorm(torch.nn.Module):  # 标准分解（w 直乘）
        def __init__(s, d):
            super().__init__(); s.weight = torch.nn.Parameter(torch.ones(d)); s.variance_epsilon = 1e-6

        def forward(s, xx):
            d = xx.dtype; x32 = xx.float(); var = x32.pow(2).mean(-1, keepdim=True)
            return s.weight.to(d) * (x32 * torch.rsqrt(var + 1e-6)).to(d)

    class ZcgDecomp(torch.nn.Module):  # zcg 分解（1+w）
        def __init__(s, d):
            super().__init__(); s.weight = torch.nn.Parameter(torch.zeros(d)); s.variance_epsilon = 1e-6

        def forward(s, xx):
            d = xx.dtype; x32 = xx.float(); var = x32.pow(2).mean(-1, keepdim=True)
            return (1 + s.weight).to(d) * (x32 * torch.rsqrt(var + 1e-6)).to(d)

    # 标准分解 → SSA 一致（native 复刻了框架的标准分解语义）
    fx = provider._fx_ssa_compare(DecompRMSNorm(256), 1e-6, x)
    assert fx is not None and fx["applicable"] and fx["ssa_equal"], \
        "标准分解应与 native 结构一致"
    # zcg 分解 → SSA 识别语义差异（1+w 多一个 add）
    fx = provider._fx_ssa_compare(ZcgDecomp(256), 1e-6, x)
    assert fx is not None and fx["applicable"] and not fx["ssa_equal"], \
        "zcg（1+w）应被识别出语义差异"
    # Qwen2RMSNorm：transformers 原生是分解；在 KL3 环境被 torch_xmlir
    # symbrewrite patch 成融合 mock 类（custom_ops 命名空间是厂商 .so 注册的，
    # 非 transformers 原生——2026-10-26 口径修正）。两种形态都验证：
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        fx = provider._fx_ssa_compare(Qwen2RMSNorm(256, 1e-6), 1e-6, x)
        assert fx is not None
        if fx["applicable"]:
            # 无厂商 patch（或已清场）：原生分解，SSA 应与 native 一致
            assert fx["ssa_equal"], "原生分解应与 native 结构一致"
        # else：厂商 patch 的融合 mock 类 → SSA 不适用（融合 vs 分解不可比）
    except ImportError:
        pass  # 无 transformers 则跳过
