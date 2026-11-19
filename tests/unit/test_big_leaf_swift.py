"""big_leaf Swift 入口的单开关与安全默认值。"""
import os

import pytest

import ttorch
from ttorch.bridge import framework_patch as fw
from ttorch.bridge.big_leaf import swift


@pytest.fixture(autouse=True)
def _isolate_config_env():
    names = ("TT_BIG_LEAF", "XMLIR_ENABLE_FAST_FC", "XTE_DISABLE_FAST_BF16_CACHE",
             "TT_FAST_FC_LIFECYCLE", "TT_BIG_LEAF_FORWARD_MODE")
    original = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _mock_install(monkeypatch):
    registered = []
    monkeypatch.setattr(ttorch, "is_active", lambda: True)
    monkeypatch.setattr(swift, "_ACTIVE_CONFIG", None)
    monkeypatch.setattr(swift, "_register", registered.append)
    monkeypatch.setattr(swift.importlib, "import_module", lambda _: None)
    monkeypatch.setattr(fw, "install", lambda verbose: None)
    monkeypatch.setattr(fw, "patch_status",
                        lambda: {patch.dotted: "applied" for patch in registered})
    return registered


def test_explicit_pin_on_is_rejected(monkeypatch):
    monkeypatch.setattr(ttorch, "is_active", lambda: True)
    monkeypatch.setattr(swift, "_ACTIVE_CONFIG", None)
    monkeypatch.setenv("XTE_DISABLE_FAST_BF16_CACHE", "0")
    with pytest.raises(ValueError, match="XTE_DISABLE_FAST_BF16_CACHE=1"):
        swift.enable(big_leaf=True)


@pytest.mark.parametrize("big_leaf", [False, True])
def test_single_switch_enables_safe_fast_fc_defaults(monkeypatch, big_leaf):
    registered = _mock_install(monkeypatch)

    swift.enable(big_leaf=big_leaf)

    assert swift.os.environ[swift.ENV_VAR] == str(int(big_leaf))
    assert swift.os.environ["XTE_DISABLE_FAST_BF16_CACHE"] == "1"
    assert swift.os.environ["XMLIR_ENABLE_FAST_FC"] == "1"
    assert len(registered) == (4 if big_leaf else 3)
    assert "TT_FAST_FC_LIFECYCLE" not in swift.os.environ
    assert "TT_BIG_LEAF_FORWARD_MODE" not in swift.os.environ


def test_explicit_fast_fc_disable_is_preserved(monkeypatch):
    _mock_install(monkeypatch)
    monkeypatch.setenv("XTE_DISABLE_FAST_BF16_CACHE", "1")
    monkeypatch.setenv("XMLIR_ENABLE_FAST_FC", "0")

    swift.enable(big_leaf=True)

    assert swift.os.environ["XMLIR_ENABLE_FAST_FC"] == "0"


def test_enable_does_not_run_export_contract(monkeypatch):
    """导出安全契约的唯一挂载点是 ttorch.activate()（2026-11-19 收敛）。

    enable() 曾同时挂一份不带 worker gate 的契约：裸调它的工具/单测进程
    会被一个"保存时才兑现"的风险拦住，并被悄悄改掉环境变量——与 activate()
    的 worker gate 语义直接冲突。这里钉住收敛后的分工：enable() 不碰契约。
    契约行为本身的回归在 test_vendor_retire.py。
    """
    _mock_install(monkeypatch)
    from ttorch.bridge import vendor_retire as vr
    monkeypatch.setattr(vr, "assert_export_safe",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("enable() 不应执行导出安全契约")))
    monkeypatch.setattr(vr, "assert_export_safe_for_worker",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("enable() 不应执行导出安全契约")))

    swift.enable(big_leaf=True)   # 不抛即通过


def test_post_import_recheck_reports_without_raising(monkeypatch, capsys):
    """B17b：厂商 lazy hook 在 swift import 时才覆盖补丁——这一眼必须有，
    但不能 raise（过渡期厂商实现本身在干同一件事）。"""
    monkeypatch.setattr(fw, "overwritten_patches",
                        lambda: {"swift.x.wrap_model": "xswift_ext.components.y"})

    assert swift._report_overwritten_patches() == {
        "swift.x.wrap_model": "xswift_ext.components.y"}
    out = capsys.readouterr().out
    assert "xswift_ext" in out and "wrap_model" in out


def test_enable_from_env_only_forwards_master_switch(monkeypatch):
    captured = {}
    monkeypatch.setattr(swift, "enable", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv(swift.ENV_VAR, "on")
    monkeypatch.setenv("TT_FAST_FC_LIFECYCLE", "0")
    monkeypatch.setenv("TT_BIG_LEAF_FORWARD_MODE", "switch")

    swift.enable_from_env()

    assert captured == {"big_leaf": True}
