"""big_leaf 双线实体版的 CPU 可测逻辑：param_state 转换形状、sharded_state_dict
补丁的签名守卫与格式闸门措辞分流。"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from ttorch.bridge.big_leaf import distopt_state as dstate
from ttorch.bridge.big_leaf import optimizer as bl_optimizer


# ----------------------------------------------- P2-1: expert→entity 名义形状

class _FakePlan:
    def __init__(self, model, modules_to_bind, active=True):
        self.model = model
        self.modules_to_bind = modules_to_bind
        self._active = active


def _expert_state(g, out, inp, keys=("param", "exp_avg", "exp_avg_sq")):
    state = {}
    for i in range(g):
        state[i] = {k: torch.arange(out * inp, dtype=torch.float32).reshape(out, inp)
                    + (i * 100 + hash(k) % 7) for k in keys}
    return state


def test_convert_expert_to_entity_shape_is_canonical():
    """entity 目标必须是 (G, out, in) 名义形状（P2-1：stack 而非 cat）。"""
    g, out, inp = 2, 3, 4
    model = nn.Module()
    model.add_module("gl", nn.Module())
    plan = _FakePlan(model, [model.gl])
    state = _expert_state(g, out, inp)
    src_positions = [("gl", "weight", i, (out, inp)) for i in range(g)]
    tgt_positions = [("gl", "weight", None, (g, out, inp))]

    converted = dstate.convert_param_state(state, src_positions, tgt_positions,
                                           plan=plan)

    for sk in ("param", "exp_avg", "exp_avg_sq"):
        got = converted[0][sk]
        assert tuple(got.shape) == (g, out, inp), (sk, got.shape)
        expected = torch.stack([state[0][sk], state[1][sk]], dim=0)
        assert torch.equal(got, expected), sk


def test_convert_entity_flatten_equivalent_to_cat():
    """stack 改动不得改变数值：flatten 后与旧 cat 字节序一致。"""
    g, out, inp = 2, 3, 4
    model = nn.Module()
    model.add_module("gl", nn.Module())
    plan = _FakePlan(model, [model.gl])
    state = _expert_state(g, out, inp)
    src_positions = [("gl", "weight", i, (out, inp)) for i in range(g)]
    tgt_positions = [("gl", "weight", None, (g, out, inp))]

    converted = dstate.convert_param_state(state, src_positions, tgt_positions,
                                           plan=plan)
    flat = converted[0]["param"].reshape(-1)
    legacy = torch.cat([state[0]["param"], state[1]["param"]], dim=0).reshape(-1)
    assert torch.equal(flat, legacy)


# ----------------------------------------------- P1-1: sharded_state_dict 签名守卫

def test_signature_guard_accepts_real_mcore():
    """真实 mcore 签名必须通过（锚定当前依赖的参数序）。"""
    bl_optimizer._assert_sd_signature(DistributedOptimizer.sharded_state_dict)


def test_signature_guard_rejects_drift():
    """签名漂移（参数名/序变化）必须 fail-fast，而非静默按位置错位转发。"""
    def drifted(self, model_sharded_state_dict=None, renamed=False):
        return None

    with pytest.raises(RuntimeError, match="签名"):
        bl_optimizer._assert_sd_signature(drifted)


def test_wrapper_factory_runs_signature_guard():
    """补丁工厂在安装期即校验签名（启动期 fail-fast，不等运行到保存）。"""
    def drifted(self, model_sharded_state_dict=None):
        return None

    with pytest.raises(RuntimeError, match="签名"):
        bl_optimizer.mcore_sharded_state_dict_wrapper(drifted)


# ----------------------------------------------- P2-2: 格式闸门措辞按 is_loading 分流

def _self_with_active_plan():
    model = nn.Module()
    plan = _FakePlan(model, [])
    chunk = SimpleNamespace(_tt_big_leaf_plan=plan)
    return SimpleNamespace(model_chunks=[chunk])


def test_format_gate_save_message_points_to_save_flag():
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(
        DistributedOptimizer.sharded_state_dict)
    with pytest.raises(ValueError, match="dist_ckpt_optim_fully_reshardable"):
        wrapped(_self_with_active_plan(), None, False, None, {})


def test_format_gate_load_message_points_to_migration():
    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(
        DistributedOptimizer.sharded_state_dict)
    with pytest.raises(ValueError, match="迁移"):
        wrapped(_self_with_active_plan(), None, True, None, {})


def test_format_gate_no_plan_passes_through(monkeypatch):
    """无 active plan 时补丁零侵入：原样转发（这里只验证不进闸门）。"""
    sentinel = object()

    def fake_orig(self, model_sharded_state_dict=None, is_loading=False,
                  sharding_type=None, metadata=None):
        return sentinel

    wrapped = bl_optimizer.mcore_sharded_state_dict_wrapper(fake_orig)
    result = wrapped(SimpleNamespace(model_chunks=[]), None, False, None, {})
    assert result is sentinel
