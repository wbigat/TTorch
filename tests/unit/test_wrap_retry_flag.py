"""重试阶段不再执行 flag 预拒的回归（A4，复核 20261028 §4.4）。

背景：自检失败后 `_retry_wrap` 会重试其余变体 provider，但重试仍执行
`wrap_module` 的 flag 预拒。模块 flag 写错（显式 False 的 zcg 模块）时形成
死锁：标准 provider 自检被数值否定 -> 重试 zcg provider 又被同一个错误 flag
预拒 -> 只能退 native（数值对、白丢加速）。实测该场景最终绑定 native。
"""
import torch

import ttorch
from ttorch.provider import manager as pm
from ttorch.provider.impls import build_all


class ZcgWrongFlagRMSNorm(torch.nn.Module):
    """zcg 语义（(1+w) 乘）但 flag 显式写成 False——先验与真实语义矛盾。"""

    def __init__(self, d=16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(d) * 0.1)
        self.eps = 1e-6
        self.zero_centered_gamma = False  # 故意写错

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        normed = x32 * torch.rsqrt(var + self.eps)
        return (normed * (1 + self.weight.float())).to(in_dtype)


class ZcgRightFlagRMSNorm(ZcgWrongFlagRMSNorm):
    """同语义但 flag 正确——用于证明首轮先验仍然生效（不被 A4 破坏）。"""

    def __init__(self, d=16):
        super().__init__(d)
        self.zero_centered_gamma = True


def _bind(module, x):
    ttorch.activate()
    build_all()
    m = pm.instance()
    orig = module.forward
    assert m._wrap_one(module) is not None, "未被任何 provider 包装"
    module.forward(x)  # 触发延迟自检（可能走重试）
    return getattr(module.forward, "_semantic", None), orig


def test_wrong_flag_does_not_block_correct_variant():
    """flag 写错的 zcg 模块：重试必须能绑到 zcg provider，而不是退 native。"""
    mod = ZcgWrongFlagRMSNorm()
    final, orig = _bind(mod, torch.randn(4, 16))
    assert final == "rmsnorm_zcg", f"错误 flag 挡住了正确变体，最终绑定={final}"
    x = torch.randn(4, 16)
    assert (mod.forward(x) - orig(x)).abs().max().item() < 1e-5


def test_first_round_flag_prior_still_honored():
    """首轮先验不得被 A4 破坏：flag 正确时应一次命中，不经重试。"""
    mod = ZcgRightFlagRMSNorm()
    final, orig = _bind(mod, torch.randn(4, 16))
    assert final == "rmsnorm_zcg", final
    x = torch.randn(4, 16)
    assert (mod.forward(x) - orig(x)).abs().max().item() < 1e-5


def test_wrap_module_trust_flag_contract():
    """wrap_module 的 trust_flag 契约：True 预拒不符变体，False 放行交给兜底。"""
    ttorch.activate()
    build_all()
    m = pm.instance()
    zcg = m.registry.get("rmsnorm_zcg")
    mod = ZcgWrongFlagRMSNorm()
    assert zcg.wrap_module(mod, trust_flag=True) is None, "首轮应按 flag 预拒"
    assert zcg.wrap_module(mod, trust_flag=False) is not None, "重试阶段应放行"
