"""B7（Embedding object payload CPU gather）补丁的回归。

定性（2026-11-04 用户确认）：**未实际复现故障，但理论风险成立**——
swift.loss.embedding 对 CUDA 张量直接调 accelerate.gather_object，
object pickle 触发 CUDA storage 反序列化，是 PyTorch 约定中的脆弱用法，
KL3 masquerade 栈放大风险。补丁语义零争议（对齐推荐的 CPU payload 实践）。
"""
import sys
import types

import pytest
import torch

from ttorch.bridge import framework_patch as fw
from ttorch.bridge.patches import gather_object_cpu_wrapper


class _CudaLike:
    """不依赖真实设备的"非 CPU"张量替身——CPU-only 环境也能跑判别逻辑。"""

    def __init__(self, t):
        self.t = t


def _nested(device):
    return {
        "sentences": torch.randn(2, 3, device=device, requires_grad=True),
        "labels": [torch.ones(4, device=device), torch.zeros(2, device=device)],
        "meta": ("keep", 42),
        "nested": {"t": torch.randn(1, device=device)},
    }


def test_cpuify_detaches_and_keeps_values():
    captured = {}

    def fake_orig(obj):
        captured["payload"] = obj
        return "ok"

    wrapped = gather_object_cpu_wrapper(fake_orig)
    payload = _nested("cpu")
    out = wrapped(payload)
    assert out == "ok"
    got = captured["payload"]
    # 递归到位 + detach（梯度必须被剥掉——pickle 带图既冗余又危险）
    assert got["sentences"].requires_grad is False
    assert torch.equal(got["sentences"], payload["sentences"].detach())
    assert torch.equal(got["labels"][0], payload["labels"][0])
    assert torch.equal(got["nested"]["t"], payload["nested"]["t"])


def test_cpuify_preserves_containers_and_scalars():
    def fake_orig(obj):
        return obj

    wrapped = gather_object_cpu_wrapper(fake_orig)
    payload = _nested("cpu")
    got = wrapped(payload)
    assert isinstance(got["labels"], list)
    assert isinstance(got["meta"], tuple)
    assert got["meta"] == ("keep", 42)


def test_cpuify_preserves_namedtuple_and_handles_set():
    """review 修正的回归：namedtuple 不得降级为 plain tuple；set 系要递归。"""
    from collections import namedtuple

    Pair = namedtuple("Pair", ["x", "y"])

    def fake_orig(obj):
        return obj

    wrapped = gather_object_cpu_wrapper(fake_orig)
    t = torch.ones(2, requires_grad=True)
    got = wrapped({"pair": Pair(t, 7), "s": {t, 3}, "fs": frozenset({t})})
    pair = got["pair"]
    assert type(pair) is Pair and pair.y == 7
    assert pair.x.requires_grad is False
    inner = next(v for v in got["s"] if torch.is_tensor(v))
    assert inner.requires_grad is False
    assert any(torch.is_tensor(v) for v in got["fs"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="无设备")
def test_cuda_payload_moved_to_cpu():
    captured = {}

    def fake_orig(obj):
        captured["payload"] = obj
        return obj

    wrapped = gather_object_cpu_wrapper(fake_orig)
    payload = _nested("cuda")
    wrapped(payload)
    got = captured["payload"]
    assert got["sentences"].device.type == "cpu"
    assert got["labels"][0].device.type == "cpu"
    assert got["nested"]["t"].device.type == "cpu"
    assert torch.equal(got["sentences"], payload["sentences"].detach().cpu())


def _b7_patch():
    """局部构造补丁对象——不走全局 register_all，避免污染共享注册表
    （曾因此打断 test_rebind_uninstall：全局注册后其他用例的 install 行为被改）。"""
    import ttorch.bridge.patches as patches

    return fw.FunctionPatch(
        dotted=patches._GATHER_DOTTED,
        module=patches._GATHER_MODULE,
        attr=patches._GATHER_ATTR,
        wrapper=patches.gather_object_cpu_wrapper,
        intent="test",
    )


def test_patch_binds_callsite_namespace(monkeypatch):
    """绑定调用点命名空间即生效：swift.loss.embedding 里的名字被换成包装版。"""
    import ttorch.bridge.patches as patches

    mod = types.ModuleType("swift.loss.embedding")
    def orig(obj):
        return obj
    mod.gather_object = orig
    monkeypatch.setitem(sys.modules, "swift.loss.embedding", mod)

    fw._APPLIED.pop(patches._GATHER_DOTTED, None)
    assert fw._apply_one(_b7_patch()) is True
    assert mod.gather_object is not orig
    assert getattr(mod.gather_object, "_tt_patched", False) is True
    # 清理 rebind 全局快照——否则 test_rebind_uninstall 的
    # `snapshot() == {}` 断言会被这条残留打断（已踩过一次）
    from ttorch.utils import _rebind

    _rebind.unbind(mod, patches._GATHER_ATTR, owner="fwpatch")
    fw._APPLIED.pop(patches._GATHER_DOTTED, None)


def test_patch_skipped_when_symbol_absent(monkeypatch):
    import ttorch.bridge.patches as patches

    mod = types.ModuleType("swift.loss.embedding")
    monkeypatch.setitem(sys.modules, "swift.loss.embedding", mod)
    fw._APPLIED.pop(patches._GATHER_DOTTED, None)
    assert fw._apply_one(_b7_patch()) is False
    assert fw._APPLIED[patches._GATHER_DOTTED].startswith("skipped")
    fw._APPLIED.pop(patches._GATHER_DOTTED, None)
