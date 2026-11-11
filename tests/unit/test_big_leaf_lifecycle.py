"""fast-FC 缓存生命周期与 Swift 包装顺序。"""
from types import SimpleNamespace

from hydrax.xaccelerator import stateful
from ttorch.bridge.big_leaf import lifecycle


def test_train_step_invalidates_inside_optimizer_step(monkeypatch):
    events = []
    optimizer = SimpleNamespace()
    optimizer.step = lambda: events.append("optimizer_step")
    trainer = SimpleNamespace(optimizer=optimizer)
    monkeypatch.setattr(lifecycle, "invalidate_fast_fc_cache",
                        lambda current: events.append(("invalidate", current)))

    def orig(self, value):
        events.append(("before_step", value))
        self.optimizer.step()
        events.append("before_param_sync")
        return "result"

    result = lifecycle.train_step_wrapper(orig)(trainer, 7)
    assert result == "result"
    assert events == [
        ("before_step", 7), "optimizer_step", ("invalidate", optimizer),
        "before_param_sync",
    ]


def test_checkpoint_save_force_syncs_before_original():
    events = []
    model = SimpleNamespace(
        ddp_config=SimpleNamespace(use_distributed_optimizer=True, overlap_param_gather=False),
        start_param_sync=lambda force_sync: events.append(("sync", force_sync)))
    trainer = SimpleNamespace(wrapped_models=[model])

    def orig(self):
        events.append("save")
        return "saved"

    assert lifecycle.save_checkpoint_wrapper(orig)(trainer) == "saved"
    assert events == [("sync", True), "save"]


def test_checkpoint_save_skips_non_dist_opt_model():
    model = SimpleNamespace(
        ddp_config=SimpleNamespace(use_distributed_optimizer=False, overlap_param_gather=False),
        start_param_sync=lambda force_sync: (_ for _ in ()).throw(
            AssertionError("非 dist-opt 不应调用 start_param_sync")))
    trainer = SimpleNamespace(wrapped_models=[model])
    assert lifecycle.save_checkpoint_wrapper(lambda self: "saved")(trainer) == "saved"


def test_checkpoint_load_invalidates_only_when_weights_are_loaded(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "invalidate_fast_fc_cache", calls.append)
    trainer = SimpleNamespace(
        optimizer="optimizer",
        args=SimpleNamespace(mcore_model="checkpoint", mcore_adapter=None),
    )
    lifecycle.load_checkpoint_wrapper(lambda self: 3)(trainer)
    assert calls == ["optimizer"]

    calls.clear()
    trainer.args.mcore_model = None
    lifecycle.load_checkpoint_wrapper(lambda self: 4)(trainer)
    assert calls == []


def test_invalidation_uses_hydrax_post_hook_signature(monkeypatch):
    calls = []
    monkeypatch.setenv("XMLIR_ENABLE_FAST_FC", "true")
    monkeypatch.setattr(stateful, "reinitialize_all_after_param_update",
                        lambda *args: calls.append(args))
    assert lifecycle.invalidate_fast_fc_cache("optimizer") is True
    assert calls == [("optimizer", (), {})]


def test_invalidation_is_noop_when_fast_fc_is_disabled(monkeypatch):
    monkeypatch.setenv("XMLIR_ENABLE_FAST_FC", "0")

    def unexpected(*args):
        raise AssertionError("fast-FC 关闭时不应触碰 stateful cache")

    monkeypatch.setattr(stateful, "reinitialize_all_after_param_update", unexpected)
    assert lifecycle.invalidate_fast_fc_cache("optimizer") is False
