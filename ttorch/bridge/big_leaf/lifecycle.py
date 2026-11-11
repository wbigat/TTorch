"""XME 离场后的 fast-FC stateful cache 生命周期。

big_leaf 自身使用 fresh-pack，但整网其他 fast-FC Parameter 仍需要在参数更新或
checkpoint 写回后失效旧缓存。stock/big_leaf 必须共用这一层，A/B 才只有布局变量。
"""
import functools
import os

from ttorch.witness.step_timer import train_step_timer


SWIFT_TRAINER_MODULE = "swift.megatron.trainers.base"
TRAIN_STEP_ATTR = "BaseMegatronTrainer.train_step"
LOAD_CHECKPOINT_ATTR = "BaseMegatronTrainer._load_checkpoint"
TRAIN_STEP_DOTTED = f"{SWIFT_TRAINER_MODULE}.{TRAIN_STEP_ATTR}"
LOAD_CHECKPOINT_DOTTED = f"{SWIFT_TRAINER_MODULE}.{LOAD_CHECKPOINT_ATTR}"
SAVE_CHECKPOINT_ATTR = "BaseMegatronTrainer.save_checkpoint"
SAVE_CHECKPOINT_DOTTED = f"{SWIFT_TRAINER_MODULE}.{SAVE_CHECKPOINT_ATTR}"


def fast_fc_enabled():
    return os.environ.get("XMLIR_ENABLE_FAST_FC", "0").strip().lower() in (
        "1", "true", "yes", "on")


def invalidate_fast_fc_cache(optimizer):
    """按 hydrax optimizer post-hook 的原签名显式失效缓存。"""
    if not fast_fc_enabled():
        return False
    from hydrax.xaccelerator.stateful import reinitialize_all_after_param_update
    reinitialize_all_after_param_update(optimizer, (), {})
    return True


def _install_optimizer_step_hook(trainer):
    optimizer = trainer.optimizer
    if getattr(optimizer, "_tt_fast_fc_lifecycle", False):
        return
    original_step = optimizer.step

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        # Swift 在 optimizer.step 返回后才 start_param_sync；必须在此之前失效，
        # 与 XME register_optimizer_step_post_hook 的相位一致。
        invalidate_fast_fc_cache(optimizer)
        return result

    optimizer.step = step
    optimizer._tt_fast_fc_lifecycle = True


def train_step_wrapper(orig):
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        _install_optimizer_step_hook(self)
        with train_step_timer(self):
            return orig(self, *args, **kwargs)
    return wrapped


def save_checkpoint_wrapper(orig):
    """恢复 Megatron 上游保存前的参数同步，Swift 自定义路径默认缺失。"""
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        for model in self.wrapped_models:
            config = model.ddp_config
            if config.use_distributed_optimizer or config.overlap_param_gather:
                model.start_param_sync(force_sync=True)
        return orig(self, *args, **kwargs)
    return wrapped


def load_checkpoint_wrapper(orig):
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if self.args.mcore_model is not None or self.args.mcore_adapter is not None:
            invalidate_fast_fc_cache(self.optimizer)
        return result
    return wrapped
