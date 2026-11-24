"""第三轮外部 review（P1–P4）的红绿复现与修复守卫。

P1 多模型闸门漏普通 chunk；P2 版本守卫时序；P3 拓扑闸门 fail-open + ETP；
P4 目标 shape 校验只覆盖 entity 分支。红阶段确认缺陷存在，修复后转绿。
"""
import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer  # noqa: F401
from megatron.core import parallel_state as mpu
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


def _fake_orig(self, model_sharded_state_dict=None, is_loading=False,
               sharding_type=None, metadata=None):
    return ("orig-called", model_sharded_state_dict)


def _fresh_binding():
    mod = SimpleNamespace(
        num_gemms=2, use_bias=False,
        weight0=torch.nn.Parameter(torch.randn(3, 4)),
        weight1=torch.nn.Parameter(torch.randn(3, 4)))
    return SimpleNamespace(plan=None, module=mod,
                           big=torch.nn.Parameter(torch.randn(2, 3, 4)), big_b=None)


# ───────────────────────────── P1：多模型闸门只数 active plan

def test_gate_rejects_mixed_bigleaf_and_plain_chunk():
    """model_chunks=[有 plan 的模型, 普通模型]：active plans=1 会被放行，但重建只含
    big_leaf 模型、普通 chunk 的可训参数查表缺失。闸门必须按 model_chunks 总数判。"""
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    plan = _FakePlan(nn.Module(), [])
    big_chunk = SimpleNamespace(_tt_big_leaf_plan=plan)
    plain_chunk = SimpleNamespace(_tt_big_leaf_plan=None)   # 普通 chunk 无 plan
    self_ = SimpleNamespace(model_chunks=[big_chunk, plain_chunk])
    with pytest.raises((ValueError, RuntimeError), match="多模型|chunk|multi"):
        wrapped(self_, None, False, "fully_reshardable", {})


def test_gate_single_chunk_with_plan_still_allowed(monkeypatch):
    """单模型 chunk + plan 仍放行（回归保护：不过度拦截合法单模型）。"""
    monkeypatch.setattr(bl_optimizer, "_rebuilt_sharded_state_dict", lambda plan: {})
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    plan = _FakePlan(nn.Module(), [])
    self_ = SimpleNamespace(model_chunks=[SimpleNamespace(_tt_big_leaf_plan=plan)])
    result = wrapped(self_, None, False, "fully_reshardable", {})
    assert result[0] == "orig-called"


# ───────────────────────────── P3：拓扑闸门 fail-open + ETP

def test_tp_ep_query_failure_raises_not_failopen(monkeypatch):
    """已初始化后拓扑查询失败：必须报错，不能按 (1,1) 放行（P3 fail-open）。"""
    monkeypatch.setattr(mpu, "model_parallel_is_initialized", lambda: True)

    def boom():
        raise RuntimeError("查询失败")

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", boom)
    with pytest.raises(RuntimeError, match="查询失败"):
        bl_optimizer._tp_ep_world_sizes()


def test_tp_ep_uninitialized_returns_one(monkeypatch):
    """明确未初始化（非 worker 探针）：返回 (1,1,1)，不报错。"""
    monkeypatch.setattr(mpu, "model_parallel_is_initialized", lambda: False)
    assert bl_optimizer._tp_ep_world_sizes() == (1, 1, 1)


def test_gate_rejects_etp(monkeypatch):
    """独立专家张量并行 ETP>1：TP1/EP1 不等于 ETP1，必须拒绝（P3）。"""
    monkeypatch.setattr(bl_optimizer, "_tp_ep_world_sizes", lambda: (1, 1, 2))
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(_fake_orig)
    plan = _FakePlan(nn.Module(), [])
    self_ = SimpleNamespace(model_chunks=[SimpleNamespace(_tt_big_leaf_plan=plan)])
    with pytest.raises((ValueError, RuntimeError), match="ETP|TP|EP"):
        wrapped(self_, None, False, "fully_reshardable", {})


# ───────────────────────────── P4：目标 shape 校验只覆盖 entity 分支

def test_convert_passthrough_shape_mismatch_rejected():
    """直通分支：目标声明 (5,)，实际 (4,) → 必须拒绝（P4 校验缺口）。"""
    state = {0: {k: torch.ones(4) for k in _KEYS}}
    src = [("fc.weight", "", None, (4,))]
    tgt = [("fc.weight", "", None, (5,))]
    with pytest.raises((ValueError, RuntimeError)):
        dstate.convert_param_state(state, src, tgt)


def test_convert_entity_to_expert_shape_mismatch_rejected():
    """entity→expert：目标声明 (4,3)，实际 expert slice 是 (3,4) → 拒绝。"""
    ent = {k: torch.randn(2, 3, 4) for k in _KEYS}   # (G=2, out=3, in=4)
    src = [("gl", "weight", None, (2, 3, 4))]
    tgt = [("gl", "weight", 0, (4, 3)), ("gl", "weight", 1, (4, 3))]
    with pytest.raises((ValueError, RuntimeError)):
        dstate.convert_param_state({0: ent}, src, tgt)   # plan=None（stock 目标）


# ───────────────────────────── P2：版本守卫接到真实 backward 入口

def test_backward_guard_fires_at_backward_entry():
    """forward → 外部写 weight（Parameter 级）→ backward：检查必须在 backward 入口
    触发（而非等下一次 forward），阻止错误梯度被计算。"""
    from ttorch.bridge.big_leaf import core as bl_core
    binding = _fresh_binding()
    bl_core._check_binding_runtime(binding)     # 记录基线
    x = torch.randn(3, requires_grad=True)
    out = (x * 2).sum()
    out.register_hook(bl_core._backward_version_guard(binding))   # 待实现的钩子
    with torch.no_grad():
        binding.module.weight0.add_(5.0)        # forward 后外部写（bump _version）
    with pytest.raises(RuntimeError, match="external_weight_update|_version"):
        out.backward()                          # 必须在 backward 入口即失败
