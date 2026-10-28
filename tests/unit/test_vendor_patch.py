"""vendor_patch：厂商性能 patch 清场（torch_xmlir symbrewrite 符号替换的逆操作）。

_restore_one 的单元验证用伪造模块（__origin_ 留痕形态）；注册表发现接口
只验健壮性（不崩 + 返回 list），真实恢复链路在 e2e（real_qwen_smoke
--restore-vendor）验证。
"""
import sys
import types

from ttorch.provider import vendor_patch


def _fake_module(name, with_origin=True):
    mod = types.ModuleType(name)

    def orig():
        return "orig"

    def patched():
        return "patched"

    mod.func = patched
    if with_origin:
        mod.__origin_func = orig  # symbrewrite set_attr 的留痕形态
    sys.modules[name] = mod
    return mod, orig


def test_restore_one_roundtrip():
    mod, orig = _fake_module("tt_fake_fw_rt")
    try:
        status, _ = vendor_patch._restore_one("tt_fake_fw_rt.func")
        assert status == "restored"
        assert mod.func is orig
        # 幂等：再次恢复报 already，不再动作
        status, _ = vendor_patch._restore_one("tt_fake_fw_rt.func")
        assert status == "already"
        assert mod.func is orig
    finally:
        del sys.modules["tt_fake_fw_rt"]


def test_restore_one_no_origin():
    mod, _ = _fake_module("tt_fake_fw_noorig", with_origin=False)
    try:
        status, detail = vendor_patch._restore_one("tt_fake_fw_noorig.func")
        assert status == "failed" and "留痕" in detail
    finally:
        del sys.modules["tt_fake_fw_noorig"]


def test_restore_one_module_not_imported():
    status, detail = vendor_patch._restore_one("tt_no_such_mod.func")
    assert status == "failed" and "未 import" in detail


def test_registered_rewrites_robust():
    """注册表发现：有/无 torch_xmlir 都不崩，返回 list。"""
    syms = vendor_patch._registered_rewrites()
    assert isinstance(syms, list)


def test_restore_scope_guard():
    import pytest
    with pytest.raises(AssertionError):
        vendor_patch.restore("bogus")


def test_activate_reads_restore_env(monkeypatch):
    """启动期口径开关：ttorch.activate 按 TT_RESTORE_VENDOR 调 restore；
    不设则不调（xmlir patch 在场 = xmlir 性能基线）。"""
    import ttorch
    calls = []
    monkeypatch.setattr(vendor_patch, "restore",
                        lambda scope: calls.append(scope) or ([], []))
    monkeypatch.setenv("TT_RESTORE_VENDOR", "perf")
    ttorch.activate()
    assert calls == ["perf"]
    calls.clear()
    monkeypatch.delenv("TT_RESTORE_VENDOR")
    ttorch.activate()
    assert calls == []
