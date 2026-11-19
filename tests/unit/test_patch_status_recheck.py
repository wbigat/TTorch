"""B17：`patch_status()` 的运行期复核（补丁被第三方整体覆盖必须可见）。

缺陷（2026-11-18 实测，退役文档 §15.3-A）：`_APPLIED` 是**安装那一刻**的结论。
厂商插件 XSwift 的 `KlxSymbolRewriter.replace(is_wrapper=False)` 是整函数替换，
而生产的真实时序是 `ttorch.activate()` 先打补丁、再 import swift 触发厂商
lazy hook —— 旧实现下三个共享符号被整体顶掉后 `patch_status()` 仍报全
`applied`（谎报），`big_leaf/swift.py` 的「禁止 pending」fail-fast 也拦不住
（检查发生在覆盖之前）。

复核基准取自 `_rebind.snapshot()`（与 `assert_bindings_intact()` 同源），
不是本模块私藏的对象引用——两个报告必须对得上账。
"""
import functools
import sys
import types

import pytest

from ttorch.bridge import framework_patch as fw
from ttorch.utils import _rebind
from ttorch.witness import provenance


_MODULE = "_tt_fake_recheck_mod"
_ATTR = "target"
_DOTTED = f"{_MODULE}.{_ATTR}"


def _plus_one(orig):
    @functools.wraps(orig)
    def wrapper(x):
        return orig(x) + 1
    return wrapper


@pytest.fixture()
def applied(monkeypatch):
    """装好一条补丁的假模块；yield (模块, 补丁, TT 装上去的那个对象)。"""
    mod = types.ModuleType(_MODULE)
    mod.target = lambda x: x
    monkeypatch.setitem(sys.modules, _MODULE, mod)

    patch = fw.FunctionPatch(dotted=_DOTTED, module=_MODULE, attr=_ATTR,
                             wrapper=_plus_one, intent="test recheck")
    # 直接进 _PATCHES：patch_status 只看注册表，而 register_patch 无反注册接口
    fw._PATCHES.append(patch)
    try:
        assert fw._apply_one(patch) is True
        assert fw.patch_status()[_DOTTED] == "applied"
        yield mod, patch, mod.target
    finally:
        fw._PATCHES.remove(patch)
        fw._APPLIED.pop(_DOTTED, None)
        fw._BOUND.pop(_DOTTED, None)
        fw._OVERWRITE_SEEN.pop(_DOTTED, None)
        _rebind.unbind(mod, _ATTR, owner="fwpatch")


def test_whole_function_replacement_is_reported(applied):
    """厂商整函数替换（无 `__wrapped__`）→ 状态必须变为 overwritten 并指名篡夺者。"""
    mod, _patch, _ours = applied

    def vendor_impl(x):
        return x * 100
    # 真实形态：XSwift 的实现定义在 xswift_ext.components.* 里（实测 impl_module）
    vendor_impl.__module__ = "xswift_ext.components.fake"
    mod.target = vendor_impl

    state = fw.patch_status()[_DOTTED]
    assert state.startswith("overwritten("), f"覆盖未被察觉: {state}"
    assert "xswift_ext" in state, f"状态里必须指名篡夺者: {state}"


def test_recheck_off_keeps_install_time_result(applied):
    """recheck=False 给安装时结论——两个口径必须能分辨，否则无法定位覆盖发生在何时。"""
    mod, _patch, _ours = applied
    mod.target = lambda x: x * 100
    assert fw.patch_status(recheck=False)[_DOTTED] == "applied"
    assert fw.patch_status()[_DOTTED].startswith("overwritten(")


def test_overwritten_patches_lists_dotted(applied):
    mod, _patch, _ours = applied
    assert fw.overwritten_patches() == {}
    mod.target = lambda x: x * 100
    assert _DOTTED in fw.overwritten_patches()


def test_third_party_wrapper_is_not_overwrite(applied):
    """第三方在我们**外面**再套一层：TT 的修正仍在链上，不能报假警报。

    （big_leaf 入口的 fail-fast 直接读这个状态，假警报会把能跑的训练打挂。）
    """
    mod, _patch, ours = applied

    @functools.wraps(ours)
    def outer(x):
        return ours(x)
    mod.target = outer

    assert fw.patch_status()[_DOTTED] == "applied"
    assert fw.overwritten_patches() == {}


def test_overwrite_leaves_one_provenance_record(applied):
    """覆盖必须留痕，且重复查询不刷屏（同一篡夺者只记一次）。"""
    mod, _patch, _ours = applied
    before = len([r for r in provenance.records(provenance.LAYER_BRIDGE)
                  if r["outcome"] == "overwritten" and r["subject"] == _DOTTED])

    def vendor_impl(x):
        return x
    vendor_impl.__module__ = "xswift_ext.components.fake"
    mod.target = vendor_impl
    for _ in range(3):
        fw.patch_status()

    rows = [r for r in provenance.records(provenance.LAYER_BRIDGE)
            if r["outcome"] == "overwritten" and r["subject"] == _DOTTED]
    assert len(rows) - before == 1, f"证据条数异常: {len(rows) - before}"
    assert "xswift_ext" in rows[-1]["evidence"]["current"]


def test_symbol_disappearing_is_reported(applied):
    """落点被整个删掉也算失效（不能因为 getattr 返回 None 就当还在）。"""
    mod, _patch, _ours = applied
    del mod.target
    assert fw.patch_status()[_DOTTED].startswith("overwritten(")
