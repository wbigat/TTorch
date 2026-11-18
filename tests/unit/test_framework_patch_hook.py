"""framework_patch 的 import 后置钩子回归。

真实事故（2026-11-04 review 复验发现）：`_PatchFinder` 旧实现只在
`find_spec` 里把模块名记入 `_PENDING`，**模块 exec 完成后没有任何地方
调用 drain**——于是"先 install 后 import 模块"的路径（正是生产 activate
的真实时序：ttorch.activate() 早于 swift.megatron 导入）补丁永远打不上，
patch_status 长期停在 pending(模块未到场)。
"""
import sys

import pytest

from ttorch.bridge import framework_patch as fw
from ttorch.bridge.patches import ddp_stream_fence_wrapper


@pytest.fixture()
def post_module(tmp_path, monkeypatch):
    """造一个真实可 import 的临时模块（让真 import 机制走 find_spec）。"""
    f = tmp_path / "fakefw_post.py"
    f.write_text("def wrap_model(args, models):\n    return models\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fakefw_post", None)
    yield "fakefw_post"
    sys.modules.pop("fakefw_post", None)


def _tmp_patch(module):
    p = fw.FunctionPatch(dotted=f"{module}.wrap_model", module=module,
                         attr="wrap_model", wrapper=ddp_stream_fence_wrapper,
                         intent="test")
    fw._PATCHES.append(p)
    return p


def _cleanup(p, module_name):
    fw._PATCHES.remove(p)
    fw.uninstall()
    fw._APPLIED.pop(p.dotted, None)
    mod = sys.modules.get(module_name)
    if mod is not None:
        from ttorch.utils import _rebind

        _rebind.unbind(mod, "wrap_model", owner="fwpatch")


def test_patch_applies_when_module_imported_after_install(post_module):
    """核心回归：先 install、后 import 目标模块，补丁必须在 exec 完成时打上。"""
    p = _tmp_patch(post_module)
    try:
        fw.install(verbose=False)
        assert fw._APPLIED.get(p.dotted) is None, "模块未到场时不应有绑定记录"
        import importlib

        mod = importlib.import_module(post_module)
        assert getattr(mod.wrap_model, "_tt_patched", False) is True, \
            "import 后置钩子未在模块到场时接管"
        assert fw._APPLIED.get(p.dotted) == "applied"
    finally:
        _cleanup(p, post_module)


def test_late_import_gets_patched_version_via_from_import(post_module):
    """exec 后立即绑定 → from-import 拿到的就是新实现（机制的立身之本）。"""
    p = _tmp_patch(post_module)
    try:
        fw.install(verbose=False)
        import importlib

        mod = importlib.import_module(post_module)
        from fakefw_post import wrap_model as via_from_import

        assert via_from_import is mod.wrap_model
        assert getattr(via_from_import, "_tt_patched", False) is True
    finally:
        _cleanup(p, post_module)


def test_retry_after_retries_when_dependency_module_arrives(tmp_path, monkeypatch):
    """前置条件依赖另一模块的安装动作：先失败，依赖模块 exec 后必须重试成功。

    模拟 B1 的真实时序（2026-11-04 实测）：rope_utils 先于 mcore_bridge
    被 import 时，闭包不存在 → 首次判据失败；mcore_bridge exec 完成后
    （其 __init__ 安装包装）必须重试并接管。
    """
    (tmp_path / "fake_target.py").write_text(
        "wrapped = False\n"
        "def fn(x):\n"
        "    return x\n")
    (tmp_path / "fake_dep.py").write_text(
        "import fake_target\n"
        "fake_target.wrapped = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("fake_target", "fake_dep"):
        sys.modules.pop(name, None)

    p = fw.FunctionPatch(
        dotted="fake_target.fn", module="fake_target", attr="fn",
        wrapper=lambda orig: (lambda x: orig(x)),
        intent="test",
        applies_when=lambda: sys.modules["fake_target"].wrapped,
        retry_after=("fake_dep",),
    )
    fw._PATCHES.append(p)
    try:
        fw.install(verbose=False)
        import importlib

        importlib.import_module("fake_target")
        assert fw._APPLIED[p.dotted].startswith("skipped"), "依赖未就位时应跳过"
        dep = importlib.import_module("fake_dep")
        assert dep is not None
        assert fw._APPLIED[p.dotted] == "applied", "依赖模块 exec 后未重试"
        assert getattr(sys.modules["fake_target"].fn, "_tt_patched", False)
    finally:
        mod = sys.modules.get("fake_target")
        if mod is not None:
            from ttorch.utils import _rebind

            _rebind.unbind(mod, "fn", owner="fwpatch")  # 快照残留会打断 rebind 测试
        for name in ("fake_target", "fake_dep"):
            sys.modules.pop(name, None)
        fw._PATCHES.remove(p)
        fw.uninstall()
        fw._APPLIED.pop(p.dotted, None)
