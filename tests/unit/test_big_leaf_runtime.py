"""big_leaf Megatron DDP 生产接线的参数与失败边界。"""
from types import SimpleNamespace

import pytest

from megatron.core import parallel_state as mpu
from ttorch.bridge.big_leaf import megatron as big_leaf_megatron
from ttorch.bridge.big_leaf import runtime
from ttorch.bridge.big_leaf.runtime import ddp_init_wrapper


class _Module:
    def parameters(self):
        return ["p0", "p1"]


class _Plan:
    instances = []

    def __init__(self, module, mode):
        self.module = module
        self.mode = mode
        self.layout_args = None
        self.attached = None
        self.bindings = ["binding"]
        self.__class__.instances.append(self)

    @classmethod
    def prepare(cls, module, forward_mode):
        return cls(module, forward_mode)

    def compute_full_param_layout(self, *args):
        self.layout_args = args
        return "canonical-layout"

    def attach_ddp(self, ddp):
        self.attached = ddp


@pytest.fixture(autouse=True)
def _fake_plan(monkeypatch):
    _Plan.instances = []
    monkeypatch.setattr(big_leaf_megatron, "BigLeafMegatronPlan", _Plan)
    monkeypatch.setattr(runtime, "_has_grouped_linear", lambda module: True)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda with_context_parallel: 8)
    monkeypatch.setattr(mpu, "get_expert_data_parallel_world_size", lambda: 4)


def _config():
    return SimpleNamespace(use_distributed_optimizer=True, bucket_size=1234)


def test_ddp_wrapper_injects_layout_and_attaches():
    calls = []

    def orig(self, config, ddp_config, module, **kwargs):
        calls.append((self, config, ddp_config, module, kwargs))

    ddp = SimpleNamespace()
    config = object()
    ddp_config = _config()
    module = _Module()
    ddp_init_wrapper(orig)(ddp, config, ddp_config, module)

    assert len(calls) == 1
    assert calls[0][4] == {
        "disable_bucketing": False,
        "pg_collection": None,
        "full_param_layout": "canonical-layout",
    }
    plan = _Plan.instances[0]
    assert plan.mode == "patch"
    assert plan.layout_args == (["p0", "p1"], 1234, 8, ddp_config, 4)
    assert plan.attached is ddp
    assert ddp._tt_big_leaf_plan is plan


def test_ddp_wrapper_uses_effective_bucket_size():
    def orig(*args, **kwargs):
        pass

    ddp_init_wrapper(orig)(SimpleNamespace(), object(), _config(), _Module(),
                           disable_bucketing=True)
    assert _Plan.instances[-1].layout_args[1] is None


@pytest.mark.parametrize("ddp_config,pg_collection,layout,match", [
    (SimpleNamespace(use_distributed_optimizer=False, bucket_size=1), None, None,
     "use_distributed_optimizer"),
    (_config(), object(), None, "pg_collection"),
    (_config(), None, object(), "full_param_layout"),
])
def test_ddp_wrapper_rejects_before_original(ddp_config, pg_collection, layout, match):
    called = False

    def orig(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match=match):
        ddp_init_wrapper(orig)(SimpleNamespace(), object(), ddp_config, _Module(),
                               pg_collection=pg_collection, full_param_layout=layout)
    assert not called


def test_ddp_wrapper_rejects_unverified_pipeline_parallel(monkeypatch):
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 2)

    def orig(*args, **kwargs):
        raise AssertionError("不应进入原 DDP")

    with pytest.raises(ValueError, match="PP=1"):
        ddp_init_wrapper(orig)(SimpleNamespace(), object(), _config(), _Module())


def test_non_moe_ddp_preserves_original_layout_and_skips_plan(monkeypatch):
    monkeypatch.setattr(runtime, "_has_grouped_linear", lambda module: False)
    external_layout = object()
    calls = []

    def orig(self, config, ddp_config, module, **kwargs):
        calls.append(kwargs)

    ddp_init_wrapper(orig)(SimpleNamespace(), object(), _config(), _Module(),
                           full_param_layout=external_layout)
    assert calls == [{
        "disable_bucketing": False,
        "pg_collection": None,
        "full_param_layout": external_layout,
    }]
    assert _Plan.instances == []
