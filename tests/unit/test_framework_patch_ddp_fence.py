"""B5（DDP 初始化侧流双栅栏）补丁的回归。

缺陷（源码核实）：swift ≤4.3.1 wrap_model 在裸侧流里建 DDP，两端无 join；
社区 ≥4.3.2 已含双栅栏（v4.3.2 核实）。补丁在函数外加 synchronize 栅栏，
语义与社区修复等价（同一组 happens-before），零复制。
"""
import sys
import types

import pytest

from ttorch.bridge import framework_patch as fw
from ttorch.bridge.patches import (
    _WRAP_MODEL_ATTR, _WRAP_MODEL_DOTTED, _WRAP_MODEL_MODULE,
    _ddp_fence_applies, ddp_stream_fence_wrapper,
)


@pytest.fixture()
def sync_calls(monkeypatch):
    """记录 synchronize 调用顺序，不真碰设备。"""
    import torch
    calls = []
    # 必须打在类上：torch.cuda.current_stream() 每次返回新实例，
    # patch 实例属性会被丢弃（首版踩过：entry 没记录到）
    monkeypatch.setattr(torch.cuda.Stream, "synchronize",
                        lambda self: calls.append("entry"))
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a, **k: calls.append("exit"))
    return calls


def _b5_patch():
    """局部构造补丁对象——不走全局 register_all（B7 踩过的污染坑）。"""
    return fw.FunctionPatch(
        dotted=_WRAP_MODEL_DOTTED, module=_WRAP_MODEL_MODULE,
        attr=_WRAP_MODEL_ATTR, wrapper=ddp_stream_fence_wrapper,
        intent="test",
    )


def test_fence_wraps_orig_in_order(sync_calls):
    def orig(a, b=1):
        sync_calls.append("orig")
        return a + b

    wrapped = ddp_stream_fence_wrapper(orig)
    assert wrapped(1, b=2) == 3
    assert sync_calls == ["entry", "orig", "exit"], "栅栏必须夹住原调用"


def test_exit_fence_runs_on_orig_error(sync_calls):
    def orig():
        sync_calls.append("orig")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ddp_stream_fence_wrapper(orig)()
    assert sync_calls == ["entry", "orig", "exit"], "异常路径出口栅栏也必须执行"


def test_applies_rejects_upstream_fixed(monkeypatch):
    """上游 ≥4.3.2 形态（源码含 wait_stream）→ 不接管。"""

    def wrap_model(args, models):
        ddp_stream = None  # 上游修复形态：named stream + 双 wait_stream
        ddp_stream and ddp_stream.wait_stream(None)
        return models

    mod = types.ModuleType(_WRAP_MODEL_MODULE)
    mod.wrap_model = wrap_model
    monkeypatch.setitem(sys.modules, _WRAP_MODEL_MODULE, mod)
    assert _ddp_fence_applies() is False


def test_applies_accepts_defective(monkeypatch):
    """缺陷形态（裸侧流，源码无 wait_stream）→ 接管。"""
    import torch

    def wrap_model(args, models):
        with torch.cuda.stream(torch.cuda.Stream()):
            pass
        return models

    mod = types.ModuleType(_WRAP_MODEL_MODULE)
    mod.wrap_model = wrap_model
    monkeypatch.setitem(sys.modules, _WRAP_MODEL_MODULE, mod)
    assert _ddp_fence_applies() is True


def test_apply_sweeps_all_from_import_namespaces(monkeypatch):
    """关键：wrap_model 被两层 from-import 转链，三个命名空间必须全部换绑。"""
    def wrap_model(args, models):
        return models

    lm = types.ModuleType(_WRAP_MODEL_MODULE)
    lm.wrap_model = wrap_model
    utils_pkg = types.ModuleType("swift.megatron.utils")
    utils_pkg.wrap_model = wrap_model          # from .megatron_lm_utils import
    base = types.ModuleType("swift.megatron.trainers.base")
    base.wrap_model = wrap_model               # from ...utils import
    for name, m in ((_WRAP_MODEL_MODULE, lm),
                    ("swift.megatron.utils", utils_pkg),
                    ("swift.megatron.trainers.base", base)):
        monkeypatch.setitem(sys.modules, name, m)

    fw._APPLIED.pop(_WRAP_MODEL_DOTTED, None)
    assert fw._apply_one(_b5_patch()) is True
    for m in (lm, utils_pkg, base):
        assert m.wrap_model is not wrap_model, f"{m.__name__} 未被换绑"
        assert getattr(m.wrap_model, "_tt_patched", False)

    from ttorch.utils import _rebind
    _rebind.unbind(lm, _WRAP_MODEL_ATTR, owner="fwpatch")
    _rebind.unbind(utils_pkg, _WRAP_MODEL_ATTR, owner="fwpatch")
    _rebind.unbind(base, _WRAP_MODEL_ATTR, owner="fwpatch")
    fw._APPLIED.pop(_WRAP_MODEL_DOTTED, None)


def test_double_apply_does_not_stack(monkeypatch):
    def wrap_model(args, models):
        return models

    mod = types.ModuleType(_WRAP_MODEL_MODULE)
    mod.wrap_model = wrap_model
    monkeypatch.setitem(sys.modules, _WRAP_MODEL_MODULE, mod)

    fw._APPLIED.pop(_WRAP_MODEL_DOTTED, None)
    assert fw._apply_one(_b5_patch()) is True
    first = mod.wrap_model
    assert fw._apply_one(_b5_patch()) is True   # 幂等：返回 True 但不再包
    assert mod.wrap_model is first, "重复 apply 产生了叠层"

    from ttorch.utils import _rebind
    _rebind.unbind(mod, _WRAP_MODEL_ATTR, owner="fwpatch")
    fw._APPLIED.pop(_WRAP_MODEL_DOTTED, None)
