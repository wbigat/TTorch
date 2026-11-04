"""large-weight per-expert 视图 shim 的回归（**已从 TT 撤出，存档备查**）。

原位置：`tests/unit/test_large_weight_shim.py`。撤出决策见同目录 README.md。
**不在 `tests/unit` 的默认回归范围内**（`pytest tests/unit` 不会跑到它）。
手动跑：`python -m pytest experiments/large_weight/ -q`（从仓库根目录）。

缺陷（实证 + 源码核实）：
  XTE_GROUPED_GEMM_LARGE_WEIGHT=1 时 grouped linear 只注册 3D large_weight，
  weight{i} 不存在（实证：LW=0 有 weight0..3 / LW=1 只有 large_weight）；
  而 mcore_bridge 全程按 weight{i} 寻址（gpt_bridge 12 处）→ AttributeError。
  更隐蔽的是 LoRA merge 在 **safe_merge 分支**做 `weight.data = merged`
  （重绑临时视图而非透写存储），合并结果静默丢失；非 safe_merge 分支与
  unmerge 是原地运算，真 view 可透写（口径见 patch_wrappers.py 的说明）。

本文件用同构最小替身（不依赖真实 TE/mcore_bridge）验证 shim 的读写语义。
"""
import torch

from . import large_weight_shim as lw
from .patch_wrappers import lora_merge_wrapper, set_mlp_state_wrapper


class FakeGroupedLinear(torch.nn.Module):
    """同构替身：只注册 3D large_weight/large_bias，不注册 weight{i}。"""

    def __init__(self, num_gemms=3, out_f=4, in_f=2, bias=True):
        super().__init__()
        self.num_gemms = num_gemms
        self.large_weight = torch.nn.Parameter(
            torch.arange(num_gemms * out_f * in_f, dtype=torch.float32)
            .reshape(num_gemms, out_f, in_f).clone())
        if bias:
            self.large_bias = torch.nn.Parameter(
                torch.zeros(num_gemms, out_f))
        else:
            # TE 在无 bias 时注册零尺寸占位
            self.large_bias = torch.Tensor()


class FakeMLP(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.linear_fc1 = FakeGroupedLinear(**kw)
        self.linear_fc2 = FakeGroupedLinear(**kw)


def test_gate_requires_real_large_weight():
    """门控：只有真实存在非空 large_weight 才接管（LW=0 时零副作用）。"""
    assert lw.has_large_weight(FakeGroupedLinear()) is True

    plain = torch.nn.Linear(2, 4)
    assert lw.has_large_weight(plain) is False

    empty = FakeGroupedLinear()
    empty.large_weight = None
    assert lw.has_large_weight(empty) is False


def test_num_experts_prefers_layer_fact_and_detects_conflict():
    """专家数取层自己的 num_gemms；与 large_weight 首维冲突时拒绝接管。

    这是待办 C8「专家数口径不一致」的修正：XSwift 用
    config.num_moe_experts // ep_size 推算，推算值与层实际切分不符时会
    静默产生错误的专家映射。
    """
    layer = FakeGroupedLinear(num_gemms=3)
    assert lw.num_experts_of(layer) == 3

    layer.num_gemms = 5            # 人为制造口径冲突
    assert lw.num_experts_of(layer) is None, "口径冲突时必须拒绝接管"


def test_expose_and_restore_roundtrip():
    """暴露 per-expert 视图后 weight{i} 可读且为对应切片；还原后痕迹清零。"""
    layer = FakeGroupedLinear(num_gemms=3)
    before = set(layer.__dict__.keys())

    items, names = lw.expose_per_expert(layer, 3)
    for i in range(3):
        assert torch.equal(getattr(layer, f"weight{i}"), layer.large_weight[i])
        assert torch.equal(getattr(layer, f"bias{i}"), layer.large_bias[i])

    lw.restore_layer(layer, items, names)
    for i in range(3):
        assert not hasattr(layer, f"weight{i}") or \
            f"weight{i}" not in layer.__dict__
    assert set(layer.__dict__.keys()) == before, "还原后残留了临时属性"


def test_write_back_catches_rebound_data():
    """核心：模拟 safe_merge 的 `weight.data = merged` 重绑，必须写回存储。"""
    layer = FakeGroupedLinear(num_gemms=3)
    orig = layer.large_weight.detach().clone()
    items, names = lw.expose_per_expert(layer, 3)

    # 复现 mcore_bridge/tuners/lora.py:538-540 的重绑写法
    merged = torch.full_like(layer.weight1, 99.0)
    layer.weight1.data = merged

    n = lw.write_back(layer, 3)
    lw.restore_layer(layer, items, names)

    assert n == 1, f"应写回 1 个切片，实际 {n}"
    assert torch.allclose(layer.large_weight[1], torch.full_like(orig[1], 99.0)), \
        "重绑的合并结果未落回 large_weight（即原缺陷）"
    assert torch.equal(layer.large_weight[0], orig[0]), "未被改动的专家被污染"
    assert torch.equal(layer.large_weight[2], orig[2]), "未被改动的专家被污染"


def test_write_back_skips_inplace_path():
    """原地写路径不需要搬运：指针相同则 write_back 为空操作。"""
    layer = FakeGroupedLinear(num_gemms=3)
    items, names = lw.expose_per_expert(layer, 3)
    with torch.no_grad():
        layer.weight0.add_(5.0)        # 原地写，直接落在 large 存储上
    n = lw.write_back(layer, 3)
    lw.restore_layer(layer, items, names)
    assert n == 0, "原地写被误判为需要搬运"
    assert torch.all(layer.large_weight[0] >= 5.0)


def test_write_back_refuses_shape_mismatch():
    """形状不符时不写（避免静默污染存储）。"""
    layer = FakeGroupedLinear(num_gemms=3)
    items, names = lw.expose_per_expert(layer, 3)
    layer.weight1.data = torch.zeros(99)    # 形状完全不符
    n = lw.write_back(layer, 3)
    lw.restore_layer(layer, items, names)
    assert n == 0, "形状不符仍写回，会污染 large_weight"


def test_no_bias_placeholder_is_ignored():
    """TE 无 bias 时的零尺寸占位不得被当成真参数。"""
    layer = FakeGroupedLinear(num_gemms=2, bias=False)
    items, names = lw.expose_per_expert(layer, 2)
    assert not any(n.startswith("bias") for n in names), "零尺寸占位被误暴露"
    lw.restore_layer(layer, items, names)


def test_synthesized_view_not_recorded_as_real_attr():
    """__getattr__ 合成的 weight{i} 不得进 restore 清单。

    否则还原时会被 setattr 成真实属性，永久遮蔽 shim 并固化过期视图。
    """
    class WithGetattrShim(FakeGroupedLinear):
        def __getattr__(self, name):
            if name.startswith("weight") and name[6:].isdigit():
                return self.large_weight[int(name[6:])]
            return super().__getattr__(name)

    layer = WithGetattrShim(num_gemms=2)
    assert layer.weight0 is not None          # 合成视图可读
    items, names = lw.expose_per_expert(layer, 2)
    assert all(not n.startswith("weight") for n, _ in items), \
        "合成视图被记入 restore 清单（会永久遮蔽 shim）"
    lw.restore_layer(layer, items, names)
    assert "weight0" not in layer.__dict__


def test_iter_expert_layers_covers_base_and_lora():
    """遍历必须覆盖 base layer 与 LoRA 的 A/B。"""
    mlp = FakeMLP(num_gemms=2)
    mlp.linear_fc1.lora_A = torch.nn.ModuleDict(
        {"default": FakeGroupedLinear(num_gemms=2)})
    mlp.linear_fc1.base_layer = FakeGroupedLinear(num_gemms=2)
    layers = list(lw.iter_expert_layers(mlp))
    assert mlp.linear_fc1 in layers and mlp.linear_fc2 in layers
    assert mlp.linear_fc1.lora_A["default"] in layers
    assert mlp.linear_fc1.base_layer in layers
    assert list(lw.iter_expert_layers(None)) == []


# ---------------------------------------------------------------- B2 wrapper

def test_b2_conversion_sees_per_expert_names():
    """B2：转换期间 weight{i} 可见；调用结束后还原。"""
    seen = {}

    def fake_set_mlp_state(self, mg_mlp, *args, **kwargs):
        layer = mg_mlp.linear_fc1
        seen["w0"] = getattr(layer, "weight0").clone()
        seen["in_dict"] = "weight0" in layer.__dict__
        return "ok"

    wrapped = set_mlp_state_wrapper(fake_set_mlp_state)
    mlp = FakeMLP(num_gemms=3)
    assert wrapped(None, mlp) == "ok"
    assert seen["in_dict"] is True, "转换期间 weight0 不是真实属性"
    assert torch.equal(seen["w0"], mlp.linear_fc1.large_weight[0])
    assert "weight0" not in mlp.linear_fc1.__dict__, "调用后未还原"


def test_b2_reproduces_attribute_error_without_patch():
    """先证明缺陷存在：未打补丁时按 weight{i} 寻址必然 AttributeError。"""
    mlp = FakeMLP(num_gemms=3)
    try:
        _ = mlp.linear_fc1.weight0
    except AttributeError:
        return
    raise AssertionError("替身未复现 large-weight 的 weight{i} 缺失")


def test_b2_declines_without_large_weight():
    """LW=0 布局（无 large_weight）：不接管，原实现原样执行。"""
    calls = []

    def fake(self, mg_mlp, *a, **kw):
        calls.append(mg_mlp)
        return "plain"

    class PlainMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = torch.nn.Linear(2, 4)
            self.linear_fc2 = torch.nn.Linear(2, 4)

    mlp = PlainMLP()
    assert set_mlp_state_wrapper(fake)(None, mlp) == "plain"
    assert calls == [mlp]


def test_b2_restores_on_exception():
    """原实现抛异常时也必须还原（finally 语义）。"""
    def boom(self, mg_mlp, *a, **kw):
        raise RuntimeError("转换失败")

    mlp = FakeMLP(num_gemms=2)
    try:
        set_mlp_state_wrapper(boom)(None, mlp)
    except RuntimeError:
        pass
    assert "weight0" not in mlp.linear_fc1.__dict__, "异常路径未还原"


# ---------------------------------------------------------------- B3 wrapper

class FakeLoraLinear:
    """同构 LoraParallelLinear：merge 用 `weight.data =` 重绑（即缺陷写法）。"""

    def __init__(self, base):
        self._base = base

    def get_base_layer(self):
        return self._base

    def merge(self, value=7.0):
        for i in range(self._base.num_gemms):
            w = getattr(self._base, f"weight{i}")
            w.data = torch.full_like(w, value)      # 重绑，非透写
        return "merged"


def test_b3_merge_result_reaches_large_weight():
    """核心：打补丁后 merge 结果真正落回 large_weight。"""
    base = FakeGroupedLinear(num_gemms=3)
    lora = FakeLoraLinear(base)
    FakeLoraLinear.merge = lora_merge_wrapper(FakeLoraLinear.merge)
    try:
        assert lora.merge(7.0) == "merged"
        assert torch.allclose(base.large_weight,
                              torch.full_like(base.large_weight, 7.0)), \
            "merge 结果未落回 large_weight（即原缺陷：静默丢写）"
    finally:
        FakeLoraLinear.merge = FakeLoraLinear.merge.__wrapped__


def test_b3_reproduces_silent_drop_without_patch():
    """先证明缺陷存在：未打补丁时 merge 结果丢失（且无任何报错）。"""
    base = FakeGroupedLinear(num_gemms=3)
    orig = base.large_weight.detach().clone()
    items, names = lw.expose_per_expert(base, 3)
    FakeLoraLinear(base).merge(7.0)              # 不走 write_back
    lw.restore_layer(base, items, names)
    assert torch.equal(base.large_weight, orig), \
        "替身未复现静默丢写（期望 large_weight 完全未变）"


def test_class_method_patch_keeps_integrity_check_working():
    """类方法落点不得打断完整性检测（2026-11-04 实测缺陷的回归）。

    rebind 的快照 key 原本是 holder.__name__；holder 为类时只有类名，
    assert_bindings_intact() 按它 import_module("GPTBridge") 直接
    ModuleNotFoundError，**整个检测被一条绑定打断**。
    现改为 "pkg.mod:Class" 并由检测侧解析。
    """
    import sys
    import types

    from ttorch.bridge import framework_patch as fw
    from ttorch.bridge import cuda_compat
    from ttorch.witness import bindings

    mod = types.ModuleType("_tt_fake_cls_mod")

    class Holder:
        def method(self):
            return "orig"

    mod.Holder = Holder
    sys.modules["_tt_fake_cls_mod"] = mod
    dotted = "_tt_fake_cls_mod.Holder.method"
    try:
        p = fw.FunctionPatch(dotted=dotted, module="_tt_fake_cls_mod",
                             attr="Holder.method",
                             wrapper=lambda o: lambda self: "patched",
                             intent="test")
        assert fw._apply_one(p) is True
        assert Holder().method() == "patched"
        # 关键断言：带类方法 key 的快照不得让检测抛异常
        n, rows = bindings.assert_bindings_intact(cuda_compat.post_bind_snapshot())
        names = [nm for _, nm in rows]
        assert "_tt_fake_cls_mod:Holder.method" in names, names[-3:]
        assert n == 0, [nm for ok, nm in rows if not ok]
    finally:
        from ttorch.utils import _rebind
        _rebind.unbind(Holder, "method", "fwpatch",
                       holder_path="_tt_fake_cls_mod:Holder")
        sys.modules.pop("_tt_fake_cls_mod", None)
        fw._APPLIED.pop(dotted, None)


def test_b3_declines_when_base_has_no_large_weight():
    """base layer 非 large-weight 布局：不接管。"""
    class PlainLora:
        def __init__(self):
            self.base = torch.nn.Linear(2, 4)

        def get_base_layer(self):
            return self.base

        def merge(self):
            return "plain"

    PlainLora.merge = lora_merge_wrapper(PlainLora.merge)
    try:
        assert PlainLora().merge() == "plain"
    finally:
        PlainLora.merge = PlainLora.merge.__wrapped__
