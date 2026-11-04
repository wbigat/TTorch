"""_self_check 的判别力回归（A3，复核 20261028 §4.3）。

背景：自检只验"本 provider 与模块匹配"，判别力依赖输入非退化。x=0 时标准
语义（w 直乘）与 zcg 语义（(1+w) 乘）输出都塌成零，rel=0/1e-12=0 恒过，
旧实现据此接管并永久缓存——首个 batch 一个全零输入即把模块永久绑错语义，
实测后续非退化输入上误差 0.9999995。本文件把该反例入库。
"""
import torch

import ttorch
from ttorch.provider import manager as pm
from ttorch.provider.impls import build_all
from ttorch.provider.impls.rmsnorm.base import _probe_weight


class StdRMSNorm(torch.nn.Module):
    """标准 RMSNorm（w 直乘）。"""

    def __init__(self, d=16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(d) + 0.5)
        self.eps = 1e-6

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        return self.weight.to(in_dtype) * (x32 * torch.rsqrt(var + self.eps)).to(in_dtype)


class ZcgRMSNorm(torch.nn.Module):
    """zcg RMSNorm（(1+w) 乘），零初始化权重（模拟 Qwen3.5），无 flag 属性。"""

    def __init__(self, d=16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d))
        self.eps = 1e-6

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        normed = x32 * torch.rsqrt(var + self.eps)
        return (normed * (1 + self.weight.float())).to(in_dtype)


def _activated():
    ttorch.activate()
    build_all()
    return pm.instance()


def _bind_and_run(module, first_x):
    """包装 -> 用 first_x 触发延迟自检 -> 返回最终绑定语义。"""
    m = _activated()
    orig = module.forward
    fwd = m._wrap_one(module)
    assert fwd is not None, "未被任何 provider 包装"
    module.forward(first_x)  # 触发延迟自检
    final = getattr(module.forward, "_semantic", None)
    return final, orig


def test_degenerate_first_input_does_not_lock_wrong_variant():
    """zcg 模块 + 首次全零输入：不得停留在标准 provider（旧实现会）。"""
    mod = ZcgRMSNorm()
    final, orig = _bind_and_run(mod, torch.zeros(4, 16))
    assert final != "rmsnorm", "退化输入把 zcg 模块绑到了标准 provider"
    # 用非退化输入复测真实误差——这是旧实现暴露 0.9999995 的地方
    x = torch.randn(4, 16)
    err = (mod.forward(x) - orig(x)).abs().max().item()
    assert err < 1e-5, f"绑定后数值不等价，err={err}"


def test_standard_module_degenerate_first_input_still_correct():
    """标准模块 + 首次全零输入：同样不得提前锁错，且后续数值等价。"""
    mod = StdRMSNorm()
    final, orig = _bind_and_run(mod, torch.zeros(4, 16))
    assert final != "rmsnorm_zcg", "退化输入把标准模块绑到了 zcg provider"
    x = torch.randn(4, 16)
    err = (mod.forward(x) - orig(x)).abs().max().item()
    assert err < 1e-5, f"绑定后数值不等价，err={err}"


def test_normal_input_binding_unchanged():
    """非退化输入下的既有行为不得被 A3 改变（防回归）。"""
    for cls, expect in [(StdRMSNorm, "rmsnorm"), (ZcgRMSNorm, "rmsnorm_zcg")]:
        mod = cls()
        final, orig = _bind_and_run(mod, torch.randn(4, 16))
        assert final == expect, f"{cls.__name__} 绑到了 {final}，应为 {expect}"
        x = torch.randn(4, 16)
        assert (mod.forward(x) - orig(x)).abs().max().item() < 1e-5


def test_self_check_does_not_mutate_params_or_global_rng():
    """合成输入方案的约束守卫：不得改真实参数，不得污染全局 RNG。"""
    mod = ZcgRMSNorm()
    w_before = mod.weight.detach().clone()
    rng_before = torch.get_rng_state()
    _bind_and_run(mod, torch.zeros(4, 16))
    assert torch.equal(mod.weight.detach(), w_before), "自检改动了真实权重"
    assert mod.weight.grad is None, "自检在真实权重上留下了梯度"
    assert torch.equal(torch.get_rng_state(), rng_before), "自检污染了全局 RNG"


def test_semantically_equivalent_sibling_is_not_a_rival():
    """语义等价的兄弟 provider 不得被当成竞争者而关闭接管。

    根因回归（2026-11-03 实测）：判别力参照若定义为"所有同 module_names 的
    兄弟 provider"，则语义等价兄弟（sep=0.000e+00）使 rival_rel≈0 恒成立，
    判据从"判别力检查"退化为"永远拒绝"，全部加速被静默关闭。
    参照集必须按 native 间分离度筛出真竞争者。
    """
    from ttorch.provider.spec import Provider, ProviderImpl

    class _EquivNative(ProviderImpl):
        name = "native:equiv_probe"
        kind = "reference"

        def __call__(self, x, w, eps=1e-6):
            # 与 NativeRMSNorm 语义完全相同
            x32 = x.to(torch.float32)
            var = x32.pow(2).mean(-1, keepdim=True)
            return w.to(x.dtype) * (x32 * torch.rsqrt(var + eps)).to(x.dtype)

    class _EquivProvider(Provider):
        semantic = "equiv_probe"
        module_names = ("RMSNorm",)

        def build_native(self):
            return _EquivNative()

        @classmethod
        def probe_input(cls, ref=None, device=None):
            return (torch.randn(4, 16), torch.ones(16)), {}

    m = _activated()
    std = m.registry.get("rmsnorm")
    equiv = _EquivProvider()
    # 先证明它确实是语义等价兄弟（分离度为 0），否则本测试没有针对性
    x, w = torch.randn(4, 16), torch.rand(16) + 0.5
    base = std.native(x, w, 1e-6)
    sep = (equiv.native(x, w, 1e-6) - base).abs().max().item() / base.abs().max().item()
    assert sep == 0.0, f"构造的兄弟并非语义等价（sep={sep}），测试失去针对性"

    if m.registry.get("equiv_probe") is None:
        m.register(equiv)
    try:
        mod = StdRMSNorm()
        orig = mod.forward
        fwd = m._wrap_one(mod)
        assert fwd is not None
        mod.forward(torch.randn(4, 16))
        final = getattr(mod.forward, "_semantic", None)
        assert final is not None, "语义等价兄弟在场时接管被整体关闭（根因回归）"
        x = torch.randn(4, 16)
        assert (mod.forward(x) - orig(x)).abs().max().item() < 1e-5
    finally:
        m.registry._providers.pop("equiv_probe", None)


def test_probe_weight_keeps_real_values_and_breaks_constant():
    """A3b：probe 权重用真实值；真实权重为常量时叠确定性扰动以恢复判别力。"""
    w = torch.nn.Parameter(torch.rand(8) + 0.5)
    wp = _probe_weight(w)
    assert torch.equal(wp, w.detach()), "非常量真实权重应原样使用"
    assert wp is not w.data, "必须 clone 以防 impl 原地写污染真实参数"

    w0 = torch.nn.Parameter(torch.zeros(8))
    wp0 = _probe_weight(w0)
    assert wp0.float().std().item() > 1e-3, "常量权重未被打破，忽略 weight 的 kernel 仍可蒙过"
    assert torch.equal(_probe_weight(w0), wp0), "扰动必须确定性可复现"
