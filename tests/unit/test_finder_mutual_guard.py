"""两个后置 import finder 互委托递归的回归（2026-11-05 真机发现）。

真实触发链：`ttorch.activate()`（fn_binding 的 `_PostImportFinder` 盯
`megatron...rope_utils` 的 provider 绑定）+ B1 补丁（`_PatchFinder` 盯
**同一模块**）→ `import megatron.core.models.common.embeddings.rope_utils`
时两个 finder 在 `find_spec` 里互相委托对方（各自只跳过 `finder is self`），
无限互递归 → RecursionError，megatron 路径整个起不来。

此前未暴露的原因：B1 的四卡 e2e 未与 provider fn_binding 同进程叠加；
单测两条通道各自为战，从没有"同一模块同时是两通道目标"的用例。

修复（本测试的红绿基准）：两个 finder 加 per-module 重入防护——被另一个
TT finder 委托时返回 None 让位，外层循环继续找到真实 loader；两通道的
exec 包装形成嵌套（patch 钩子里调 binding 钩子再调真实 exec），两个钩子
都必须在 exec 完成后执行。
"""
import importlib
import sys

import pytest

from ttorch.bridge import framework_patch as fw
from ttorch.provider import fn_binding as fb


@pytest.fixture()
def shared_module(tmp_path, monkeypatch):
    """同一模块同时承载 patch 目标（fn）与 binding 目标（perf_fn）。"""
    f = tmp_path / "fake_shared.py"
    f.write_text(
        "def fn(x):\n    return x\n"
        "def perf_fn(x):\n    return x\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_shared", None)
    yield "fake_shared"
    sys.modules.pop("fake_shared", None)


def _install_both(module):
    p = fw.FunctionPatch(dotted=f"{module}.fn", module=module, attr="fn",
                         wrapper=lambda orig: (lambda x: orig(x)),
                         intent="test-mutual-guard")
    fw._PATCHES.append(p)
    b = fb.PerfBinding(dotted=f"{module}.perf_fn", module=module,
                       attr="perf_fn",
                       factory=lambda orig: (lambda x: orig(x)),
                       semantic="")  # 空 semantic 跳过 native 对账
    fb._BINDINGS.append(b)
    fw.install(verbose=False)
    fb.install()
    return p, b


def _cleanup(p, b, module):
    fw._PATCHES.remove(p)
    fw.uninstall()
    fw._APPLIED.pop(p.dotted, None)
    fb.uninstall()  # 内含 unbind(owner="provider") + _BINDINGS.clear()
    fb._TRIED.discard(b.dotted)
    fb._SKIPPED.pop(b.dotted, None)
    mod = sys.modules.get(module)
    if mod is not None:
        from ttorch.utils import _rebind

        _rebind.unbind(mod, "fn", owner="fwpatch")


def test_same_module_two_channels_no_mutual_recursion(shared_module):
    """红绿基准：修复前 RecursionError；修复后两通道都接管成功。"""
    p, b = _install_both(shared_module)
    try:
        mod = importlib.import_module(shared_module)  # 修复前此处 RecursionError
        assert fw._APPLIED.get(p.dotted) == "applied", "patch 通道未接管"
        assert b.dotted in fb._TRIED and b.dotted not in fb._SKIPPED, \
            f"binding 通道未接管: {fb._SKIPPED.get(b.dotted, 'not tried')}"
        assert mod.fn(1) == 1 and mod.perf_fn(2) == 2
    finally:
        _cleanup(p, b, shared_module)


def test_single_channel_target_still_works(shared_module, tmp_path, monkeypatch):
    """防护不得误伤单通道场景：只被 patch 盯上的模块正常打补丁。"""
    g = tmp_path / "fake_patch_only.py"
    g.write_text("def fn(x):\n    return x\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_patch_only", None)
    p = fw.FunctionPatch(dotted="fake_patch_only.fn", module="fake_patch_only",
                         attr="fn", wrapper=lambda orig: (lambda x: orig(x)),
                         intent="test-single")
    fw._PATCHES.append(p)
    try:
        fw.install(verbose=False)
        mod = importlib.import_module("fake_patch_only")
        assert fw._APPLIED.get(p.dotted) == "applied"
        assert mod.fn(3) == 3
    finally:
        mod = sys.modules.get("fake_patch_only")
        if mod is not None:
            from ttorch.utils import _rebind

            _rebind.unbind(mod, "fn", owner="fwpatch")  # 快照残留会污染其他测试
        sys.modules.pop("fake_patch_only", None)
        fw._PATCHES.remove(p)
        fw.uninstall()
        fw._APPLIED.pop(p.dotted, None)
