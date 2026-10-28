"""auto_wrap：框架建模入口的定点包装（TT_PROVIDER_AUTO_WRAP=1 门控）。

形态（2026-10-28 定案）：包装框架建模函数（如 mcore_bridge.get_mcore_model），
建模返回后对每个模型调 provider_wrap——与手动同一代码路径；不动 torch 本体
（此前的 Module._call_impl 全局 patch 已废除）。

验证点：
- enable 后建模入口返回的模型被自动 wrap（含 from-import 已加载场景的立即补绑）；
- disable 恢复原始建模函数；
- 环境变量门控（activate 分支）；
- 无 zcg flag 模块的变体误接管被延迟自检 + 同步重试纠正（手动 wrap 路径）；
- wrap 机制内部异常不穿透（不接管 + 模块保持原生）；
- bypass 走原始实现。
"""
import sys
import types

import pytest
import torch

from ttorch.provider import auto_wrap, fn_binding, manager
from ttorch.provider.spec import Provider, ProviderImpl, ProviderModuleForward

_FAKE_DOTTED = "fake_fw.build_model"


class _AutoWrapNative(ProviderImpl):
    name = "native:autowrap_norm"
    kind = "reference"

    def __call__(self, x, w, eps=1e-6):
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        return w.to(x.dtype) * (x32 * torch.rsqrt(var + eps)).to(x.dtype)


class _AutoWrapProvider(Provider):
    semantic = "autowrap_norm"
    # 同族匹配（与真实 std/zcg provider 同 module_names 的形态一致）——
    # zcg 类（*RMSNorm）也会被 std 先匹配，靠自检 + 重试纠正
    module_names = ("AutoWrapNorm", "RMSNorm")

    def build_native(self):
        return _AutoWrapNative()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        device = device or "cpu"
        return (torch.randn(4, 16, device=device),
                torch.ones(16, device=device)), {}


class _BrokenProvider(Provider):
    """wrap_module 抛非 TypeError 异常的反面夹具（机制失败不许影响功能）。"""
    semantic = "autowrap_broken"
    module_names = ("AutoWrapBrokenNorm",)

    def build_native(self):
        return _AutoWrapNative()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        return (torch.randn(4, 16), torch.ones(16)), {}

    def wrap_module(self, module):
        raise RuntimeError("模拟机制内部失败")


class _AutoWrapZcgProvider(Provider):
    semantic = "autowrap_norm_zcg"
    module_names = ("RMSNorm",)  # 与标准同族匹配，靠 zcg flag + 精度兜底分流
    zero_centered_gamma = True

    def build_native(self):
        class _ZcgNative(_AutoWrapNative):
            name = "native:autowrap_norm_zcg"

            def __call__(self, x, w, eps=1e-6):
                x32 = x.float()
                var = x32.pow(2).mean(-1, keepdim=True)
                return (1 + w).to(x.dtype) * (x32 * torch.rsqrt(var + eps)).to(x.dtype)
        return _ZcgNative()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        device = device or "cpu"
        return (torch.randn(4, 16, device=device),
                torch.zeros(16, device=device)), {}


class AutoWrapNorm(torch.nn.Module):
    """类名匹配 module_names 的 RMSNorm 分解实现（w 直乘）。"""

    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        return self.weight * (x32 * torch.rsqrt(var + self.variance_epsilon)).to(x.dtype)


class AutoWrapBrokenNorm(AutoWrapNorm):
    pass


class AutoWrapNoFlagZcgRMSNorm(torch.nn.Module):
    """无 zero_centered_gamma flag 的 zcg 分解——原始 Qwen3_5RMSNorm 形态。
    flag 预检不预拒，靠延迟自检 + 同步重试完成变体分流。"""

    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        return (1 + self.weight) * (x32 * torch.rsqrt(var + self.variance_epsilon)).to(x.dtype)


def _ensure_registered():
    m = manager.instance()
    if m.registry.get("autowrap_norm") is None:
        m.register(_AutoWrapProvider())
    if m.registry.get("autowrap_broken") is None:
        m.register(_BrokenProvider())
    if m.registry.get("autowrap_norm_zcg") is None:
        m.register(_AutoWrapZcgProvider())


def _make_fake_fw():
    """假框架：建模函数 build_model 返回含匹配模块的模型。"""
    mod = types.ModuleType("fake_fw")

    def build_model():
        return torch.nn.Sequential(AutoWrapNorm(16), torch.nn.Linear(16, 16))

    mod.build_model = build_model
    sys.modules["fake_fw"] = mod
    return mod


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    auto_wrap.disable()
    fn_binding.unregister(_FAKE_DOTTED)  # monkeypatch 恢复后 disable 按原清单
    sys.modules.pop("fake_fw", None)     # 摘不到假入口——显式清
    manager.instance().set_bypass(False)


def _enable_with_fake_point(monkeypatch):
    monkeypatch.setattr(auto_wrap, "MODEL_BUILD_POINTS", (("fake_fw", "build_model"),))
    _make_fake_fw()
    auto_wrap.enable()


def test_enable_wraps_via_build_hook(monkeypatch, capsys):
    """enable 后建模入口（已加载模块的立即补绑路径）返回的模型被自动 wrap。"""
    _ensure_registered()
    _enable_with_fake_point(monkeypatch)
    model = sys.modules["fake_fw"].build_model()
    assert isinstance(model[0].forward, ProviderModuleForward)  # 自动 wrap 生效
    assert not isinstance(model[1].forward, ProviderModuleForward)  # Linear 不动
    # 可感知：终端打印
    out = capsys.readouterr().out
    assert "auto-wrap" in out and "1 modules wrapped" in out
    # 数值正确（provider native 与模块原始实现同公式）
    x = torch.randn(4, 16)
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(model[0](x), ref, rtol=1e-5, atol=1e-6)


def test_disable_restores_builder(monkeypatch):
    _ensure_registered()
    fake = sys.modules.get("fake_fw") or _make_fake_fw()
    _enable_with_fake_point(monkeypatch)
    auto_wrap.disable()
    model = sys.modules["fake_fw"].build_model()
    assert not isinstance(model[0].forward, ProviderModuleForward)  # 不再 wrap


def test_enable_idempotent(monkeypatch):
    _enable_with_fake_point(monkeypatch)
    auto_wrap.enable()  # 重复使能不报错
    assert auto_wrap.is_enabled()


def test_env_gate_on_activate(monkeypatch):
    """activate 按 TT_PROVIDER_AUTO_WRAP=1 使能；不设则不使能。"""
    from ttorch.platform import manager as _pfm
    try:
        _pfm.activation()
    except RuntimeError:
        _pfm.auto_activate()  # activate 的 session provenance 需要平台激活
    monkeypatch.setenv(auto_wrap.ENV_VAR, "1")
    manager.instance().activate()
    assert auto_wrap.is_enabled()
    auto_wrap.disable()
    monkeypatch.delenv(auto_wrap.ENV_VAR)
    manager.instance().activate()
    assert not auto_wrap.is_enabled()


def test_no_flag_zcg_corrected_by_retry():
    """无 flag 的 zcg 模块（手动 wrap 路径）：标准 provider 先 wrap（不预拒）
    -> 首次调用自检超差 -> 记录 rejected + 同步重试 -> zcg provider 接管。"""
    _ensure_registered()
    mod = AutoWrapNoFlagZcgRMSNorm(16)
    n = manager.instance().wrap_model(torch.nn.Sequential(mod))
    assert n == 1
    x = torch.randn(4, 16)
    out = mod(x)
    assert mod._tt_provider_wrapped
    # 分流到 zcg 变体 provider（全量跑时可能已被 activate 注册的真实
    # rmsnorm_zcg 接管——语义等价，不断言具体 semantic 名）
    assert mod.forward._provider.zero_centered_gamma is True
    # 重试发生：std 变体被自检拒绝并记录（单跑/全量都应走重试路径）
    assert len(mod._tt_wrap_rejected) >= 1
    assert mod.forward._provider.semantic not in mod._tt_wrap_rejected
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)  # w=0 → 1+w=1
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_wrap_error_no_propagation():
    """wrap_module 内部异常（非 TypeError）收敛为不接管，不穿透训练流程。"""
    _ensure_registered()
    mod = AutoWrapBrokenNorm(16)
    n = manager.instance().wrap_model(torch.nn.Sequential(mod))
    assert n == 0  # 机制异常被 _wrap_one 兜底
    x = torch.randn(4, 16)
    out = mod(x)  # 不炸
    assert not isinstance(mod.forward, ProviderModuleForward)  # 保持原生
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_bypass_routes_to_orig():
    _ensure_registered()
    mod = AutoWrapNorm(16)
    manager.instance().wrap_model(torch.nn.Sequential(mod))
    x = torch.randn(4, 16)
    mod(x)  # 触发延迟自检
    manager.instance().set_bypass(True)
    out = mod(x)
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
