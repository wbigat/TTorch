"""review（2026-11-23）指认缺陷的红绿复现与修复守卫。

每个测试对应一个被指认的缺陷；红阶段确认缺陷真实存在，修复后转绿。
覆盖：问题 1a/1b/1c（distopt_state 转换器）、问题 2a/2b（序列化范围）、
问题 3（格式闸门可绕过）、问题 4（属性传播）。
"""
import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from ttorch.bridge.big_leaf import distopt_state as dstate
from ttorch.bridge.big_leaf import optimizer as bl_optimizer

_KEYS = ("param", "exp_avg", "exp_avg_sq")


class _FakePlan:
    def __init__(self, model, modules_to_bind, active=True):
        self.model = model
        self.modules_to_bind = modules_to_bind
        self._active = active

    def entity_swap(self):
        return contextlib.nullcontext()


def _self_with_plans(plans):
    return SimpleNamespace(
        model_chunks=[SimpleNamespace(_tt_big_leaf_plan=p) for p in plans])


def _fake_orig(self, model_sharded_state_dict=None, is_loading=False,
               sharding_type=None, metadata=None):
    return ("orig-called", model_sharded_state_dict)


# ───────────────────────────── 问题 1a：_name_map 普通参数直通

def test_name_map_plain_weight_bias_are_passthrough():
    """普通 Linear/LayerNorm 的 weight/bias（无专家数字后缀）必须直通，
    不能 int("") 崩溃——否则任何含非 GroupedLinear 参数的真实模型都会炸。"""
    model = nn.Module()
    model.lin = nn.Linear(4, 3)
    model.ln = nn.LayerNorm(5)
    m = dstate._name_map(model)
    for p in model.parameters():
        _, kind, idx = m[id(p)]
        assert kind == "" and idx is None, f"普通参数应直通，得 kind={kind} idx={idx}"


def test_name_map_expert_weight_still_indexed():
    """专家 weight0/weight1 仍应识别为带索引（回归保护）。"""
    model = nn.Module()
    gl = nn.Module()
    gl.weight0 = nn.Parameter(torch.randn(3, 4))
    gl.weight1 = nn.Parameter(torch.randn(3, 4))
    model.gl = gl
    m = dstate._name_map(model)
    assert m[id(gl.weight0)] == ("gl", "weight", 0)
    assert m[id(gl.weight1)] == ("gl", "weight", 1)


# ───────────────────────────── 问题 1b：忽略目标声明 shape

def test_convert_rejects_expert_count_mismatch():
    """目标声明 (3,3,4)，源只有 2 个专家：必须拒绝，不能隐式产出 (2,3,4)。"""
    model = nn.Module()
    model.add_module("gl", nn.Module())
    plan = _FakePlan(model, [model.gl])
    state = {i: {k: torch.ones(3, 4) * (i + 1) for k in _KEYS} for i in range(2)}
    src = [("gl", "weight", i, (3, 4)) for i in range(2)]
    tgt = [("gl", "weight", None, (3, 3, 4))]  # 声明 G=3，源只有 2
    with pytest.raises((ValueError, RuntimeError)):
        dstate.convert_param_state(state, src, tgt, plan=plan)


# ───────────────────────────── 问题 1c：entity→entity 隐式失败

def test_convert_entity_to_entity_is_explicit():
    """entity→entity（源只有 None 键）：bound 分支 indices 为空会 stack([]) 崩。
    应明确支持恒等直通（形状校验）。"""
    model = nn.Module()
    model.add_module("gl", nn.Module())
    plan = _FakePlan(model, [model.gl])
    ent = {k: torch.randn(2, 3, 4) for k in _KEYS}
    src = [("gl", "weight", None, (2, 3, 4))]
    tgt = [("gl", "weight", None, (2, 3, 4))]
    out = dstate.convert_param_state({0: ent}, src, tgt, plan=plan)
    for sk in _KEYS:
        assert torch.equal(out[0][sk], ent[sk])


# ───────────────────────────── 问题 3：格式闸门可绕过

def test_gate_explicit_dp_reshardable_rejected_despite_metadata_fully():
    """metadata=fully_reshardable + 显式 sharding_type=dp_reshardable：
    mcore 用显式值（dp_reshardable），闸门却只看 metadata → 会放行，必须拒绝。"""
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    with pytest.raises(ValueError):
        wrapped(_self_with_plans([_FakePlan(nn.Module(), [])]), None, False,
                "dp_reshardable", {"distrib_optim_sharding_type": "fully_reshardable"})


def test_gate_explicit_fully_reshardable_passes_with_empty_metadata(monkeypatch):
    """显式 fully_reshardable + metadata 空：mcore 用显式值（fully_reshardable），
    闸门不应误拒（当前却因 metadata 空而拒绝）。"""
    monkeypatch.setattr(bl_optimizer, "_rebuilt_sharded_state_dict",
                        lambda plan: {"_rebuilt": True})
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    result = wrapped(_self_with_plans([_FakePlan(nn.Module(), [])]), None, False,
                     "fully_reshardable", {})
    assert result == ("orig-called", {"_rebuilt": True})


# ───────────────────────────── 问题 2a：多模型覆盖

def test_serialization_rejects_multi_model_chunk():
    """多 plan 循环覆盖赋值只剩最后一个 → 必须显式拒绝多模型 chunk。"""
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    plans = [_FakePlan(nn.Module(), []), _FakePlan(nn.Module(), [])]
    with pytest.raises((ValueError, RuntimeError),
                       match="多模型|multi|单模型|chunk|chunk"):
        wrapped(_self_with_plans(plans), None, False, "fully_reshardable", {})


# ───────────────────────────── 问题 2b：TP/EP 未验收即写盘

def test_serialization_rejects_tp_or_ep(monkeypatch):
    """TP>1 或 EP>1 的 optimizer checkpoint 元数据未覆盖，必须 fail-fast。"""
    monkeypatch.setattr(bl_optimizer, "_tp_ep_world_sizes", lambda: (2, 1, 1))
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    with pytest.raises((ValueError, RuntimeError), match="TP|EP"):
        wrapped(_self_with_plans([_FakePlan(nn.Module(), [])]), None, False,
                "fully_reshardable", {})


# ───────────────────────────── 问题 4：实体属性传播

def test_entity_propagates_allreduce_and_custom_flags():
    """内部线实体应传播 allreduce（buffer 分组依据）与非分片的自定义属性
    （如 attr-based lr/wd 分组标记）——否则 optimizer 构造窗口内按属性分组失效。"""
    from ttorch.bridge.big_leaf.megatron import BigLeafMegatronPlan
    src = nn.Parameter(torch.randn(3, 4))
    src.allreduce = False                 # 专家参数：expert-parallel 分组
    src.is_expert = True
    src.no_weight_decay = True            # 自定义分组属性（示例）
    plan = BigLeafMegatronPlan.__new__(BigLeafMegatronPlan)
    big = torch.nn.Parameter(torch.randn(2, 3, 4))
    plan._propagate_param_attrs(big, src)  # 待实现的方法
    assert big.allreduce is False
    assert big.is_expert is True
    assert big.no_weight_decay is True


def test_entity_does_not_blindly_copy_expert_sharding_spec():
    """维度分片属性（partition_dim/partition_stride）是 per-expert 语义，实体
    shape=(G,out,in) 分片维度不同，照抄在 TP>1 下语义错误——不得照抄，由实体
    自决（TP>1 已由序列化闸门 fail-fast，review 问题 4 的边界）。"""
    from ttorch.bridge.big_leaf.megatron import BigLeafMegatronPlan
    src = nn.Parameter(torch.randn(3, 4))
    src.partition_dim = 0
    src.partition_stride = 2
    plan = BigLeafMegatronPlan.__new__(BigLeafMegatronPlan)
    big = torch.nn.Parameter(torch.randn(2, 3, 4))
    plan._propagate_param_attrs(big, src)
    # 照抄会把 per-expert 的 dim0 分片错误安到实体的 G 维上——必须不照抄。
    assert getattr(big, "partition_dim", None) != 0
    assert getattr(big, "partition_stride", None) != 2


# ───────────────────────────── 问题 5a：版本契约守卫

def _fresh_binding():
    from ttorch.bridge.big_leaf import core as bl_core  # noqa: F401
    mod = SimpleNamespace(
        num_gemms=2, use_bias=False,
        weight0=torch.nn.Parameter(torch.randn(3, 4)),
        weight1=torch.nn.Parameter(torch.randn(3, 4)))
    return SimpleNamespace(plan=None, module=mod,
                           big=torch.nn.Parameter(torch.randn(2, 3, 4)), big_b=None)


def test_version_guard_detects_unsanctioned_external_write():
    """共享 storage 不共享 _version：外部原地写 weight{i} 不改变 big._version，
    backward 的版本检查查不到。守卫必须显式检测 weight{i}._version 漂移。"""
    from ttorch.bridge.big_leaf import core as bl_core
    binding = _fresh_binding()
    bl_core._check_binding_runtime(binding)   # 记录基线
    with torch.no_grad():
        binding.module.weight0.add_(1.0)      # 未授权外部原地写 → bump _version
    with pytest.raises(RuntimeError, match="external_weight_update|_version"):
        bl_core._check_binding_runtime(binding)


def test_version_guard_optimizer_path_no_false_positive():
    """优化器经 big 更新（不碰 weight{i}）时 _version 不变，守卫不得误报。"""
    from ttorch.bridge.big_leaf import core as bl_core
    binding = _fresh_binding()
    bl_core._check_binding_runtime(binding)
    with torch.no_grad():
        binding.big.add_(1.0)                 # 优化器路径：big._version++，weight 不变
    bl_core._check_binding_runtime(binding)   # 不抛即通过


def test_version_guard_baseline_refresh_allows_sanctioned():
    """sanctioned 更新（merge/load，走 external_weight_update）在 repair 后重记
    基线，守卫放行。"""
    from ttorch.bridge.big_leaf import core as bl_core
    binding = _fresh_binding()
    bl_core._check_binding_runtime(binding)
    with torch.no_grad():
        binding.module.weight0.add_(1.0)      # sanctioned 路径会 bump _version
    bl_core._refresh_version_baseline(binding, "weight")   # repair 后重记
    bl_core._check_binding_runtime(binding)   # 不抛即通过
