"""fn_binding / _rebind / bind_targets 的单测（纯 CPU，无需 GPU）。

起源：2026-10-21 PR review P0-1——`uninstall()` 读旧留痕名 `_origin_X` 而
`_bind_one` 已改走 `_rebind`（留痕 `_tt_origin_provider_X`），回滚静默失效。
skills 文档自列"必须提供 uninstall 并验证 finder 未泄漏"，验证这一半缺位。

本文件同时覆盖 review 指出的同族风险：
- 快照基准清理（uninstall 后 _rebind.snapshot 不得残留）；
- 双 owner 冲突留痕（P1-1 测试 2）；
- 重复绑定不叠门面（P2-1）；
- 绑定坐标表查寻行为（P1-1 测试 3）；
- cuda 命名空间匹配不误伤前缀同名包（P2-3）。

运行：pytest tests/unit/test_rebind_uninstall.py -v
"""
import importlib
import sys
import types

import pytest

from ttorch.provider import fn_binding
from ttorch.utils import _rebind


@pytest.fixture()
def fake_module():
    """造一个假框架模块（注册进 sys.modules 以便 fn_binding 的 import 路径工作）。"""
    mod = types.ModuleType("fakefw")

    def _swiglu(x):
        return x

    mod.swiglu = _swiglu
    sys.modules["fakefw"] = mod
    yield mod
    sys.modules.pop("fakefw", None)


@pytest.fixture(autouse=True)
def _clean_fn_binding():
    """每个测试前后保证 fn_binding 干净（不泄漏 finder / 注册表 / 快照）。"""
    fn_binding.uninstall()
    yield
    fn_binding.uninstall()
    _rebind.clear()


def _register_fake(attr="swiglu", semantic=""):
    fn_binding.register_binding(fn_binding.PerfBinding(
        dotted=f"fakefw.{attr}", module="fakefw", attr=attr,
        factory=lambda orig: (lambda x: orig(x)), semantic=semantic))


def test_install_binds_then_uninstall_restores(fake_module):
    """P0-1 回归：bind → uninstall 必须完整恢复（符号/留痕/快照/finder 四件套）。"""
    orig_fn = fake_module.swiglu
    _register_fake()
    fn_binding.install()

    # 接管生效：符号被替换、留痕按新命名、快照有条目、finder 在场
    assert fake_module.swiglu is not orig_fn
    assert _rebind.has_origin(fake_module, "swiglu")
    assert ("fakefw", "swiglu") in _rebind.snapshot()
    assert any(isinstance(f, fn_binding._PostImportFinder) for f in sys.meta_path)

    fn_binding.uninstall()

    # 四件套全部恢复
    assert fake_module.swiglu is orig_fn, "uninstall 必须恢复原实现"
    assert not _rebind.has_origin(fake_module, "swiglu"), "留痕必须清除"
    assert _rebind.snapshot() == {}, "快照基准必须清空（否则篡改检测误报）"
    assert not any(isinstance(f, fn_binding._PostImportFinder)
                   for f in sys.meta_path), "meta_path 不得残留 finder"
    assert fn_binding._INSTALLED is False


def test_uninstall_clears_snapshot_for_integrity_baseline(fake_module):
    """P1-2：uninstall 后快照必须清——否则第三方改写已卸载符号会被误报篡改。"""
    _register_fake()
    fn_binding.install()
    assert len(_rebind.snapshot()) == 1
    fn_binding.uninstall()
    assert ("fakefw", "swiglu") not in _rebind.snapshot()


def test_rebind_owner_conflict_recorded_and_both_traces_kept():
    """双 owner 先后绑同一符号：记 conflict 证据，两个 owner 留痕并存。"""
    mod = types.ModuleType("fakeconflict")
    mod.fn = lambda x: x
    sys.modules["fakeconflict"] = mod

    from ttorch.witness import provenance
    _rebind.rebind(mod, "fn", lambda x: 1, owner="owner_a")
    _rebind.rebind(mod, "fn", lambda x: 2, owner="owner_b")

    attrs = [n for n in dir(mod) if n.startswith("_tt_origin_")]
    assert sorted(attrs) == ["_tt_origin_owner_a_fn", "_tt_origin_owner_b_fn"], \
        "两个 owner 的留痕必须并存（属性名即可看出谁绑的）"

    conflicts = [r for r in provenance.records(provenance.LAYER_BRIDGE)
                 if r["subject"] == "fakeconflict.fn" and r["outcome"] == "conflict"]
    assert conflicts, "必须记一条 conflict 证据"
    assert conflicts[-1]["evidence"]["other_owner"] == "owner_a"


def test_rebind_repeat_same_owner_does_not_stack_wrapper():
    """P2-1：同一 owner 重复 rebind 不得'门面叠门面'——origin 始终是真原实现。"""
    mod = types.ModuleType("fakerepeat")

    def real(x):
        return x

    mod.fn = real
    sys.modules["fakerepeat"] = mod

    _rebind.rebind(mod, "fn", lambda x: x + 1, owner="p")
    first = _rebind.origin_attr("p", "fn")
    assert getattr(mod, first) is real

    _rebind.rebind(mod, "fn", lambda x: x + 2, owner="p")  # 重复绑（未先 unbind）
    assert getattr(mod, first) is real, "重复绑定的 origin 必须是真原实现，不能是上一轮门面"


def test_bind_targets_lookup_and_summary():
    """bind_targets：未登记显式 KeyError；summary 可序列化。"""
    import json

    from ttorch.provider import bind_targets

    with pytest.raises(KeyError):
        bind_targets.target("megatron", "不存在")
    with pytest.raises(KeyError):
        bind_targets.target("unknown_framework", "swiglu")

    json.dumps(bind_targets.summary())  # 可序列化（供评审 diff / provenance）
    m, a = bind_targets.target("megatron", "swiglu")
    assert m == "megatron.core.fusions.fused_bias_swiglu" and a == "swiglu"
    assert "megatron" in bind_targets.window("megatron", "swiglu")


def test_entries_namespace_no_prefix_collision():
    """P2-3：cuda 命名空间匹配不得误伤 `torch.cudafoo` 这类前缀同名包。"""
    from ttorch.bridge import module_binding

    assert module_binding._in_cuda_ns("torch.cuda")
    assert module_binding._in_cuda_ns("torch.cuda.nccl")
    assert not module_binding._in_cuda_ns("torch.cudafoo")
    assert not module_binding._in_cuda_ns("vendor_cuda")
