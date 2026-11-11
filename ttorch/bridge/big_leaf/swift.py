"""ms-swift Megatron CLI 的 big_leaf 正式入口。

用法与 ``megatron`` 一致，例如：
``python -m ttorch.bridge.big_leaf.swift sft ...``。

``TT_BIG_LEAF`` 是唯一生产功能开关：1 安装 big_leaf DDP layout/attach，显式 0
用于 stock A/B。两种取值都安装 fast-FC 生命周期，固定生产 forward 为 patch，默认
启用 fast-FC 并关闭 BF16 pin；显式 pin-on 仍拒绝。
"""
import importlib
import os

from . import lifecycle
from . import runtime


ENV_VAR = "TT_BIG_LEAF"
_ACTIVE_CONFIG = None


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _export_config(big_leaf):
    os.environ[ENV_VAR] = str(int(big_leaf))
    os.environ.setdefault("XMLIR_ENABLE_FAST_FC", "1")
    os.environ.setdefault("XTE_DISABLE_FAST_BF16_CACHE", "1")


def _register(patch):
    from ttorch.bridge import framework_patch as fw
    if all(existing.dotted != patch.dotted for existing in fw.get_patches()):
        fw.register_patch(patch)


def enable(big_leaf=True):
    """在 TT 激活后、CLI 启动前安装补丁；任何落点未生效都立即失败。"""
    global _ACTIVE_CONFIG
    import ttorch
    if not ttorch.is_active():
        raise RuntimeError("请先调用 ttorch.activate()，再启用 big_leaf Swift 入口")
    config = bool(big_leaf)
    if _ACTIVE_CONFIG is not None:
        if _ACTIVE_CONFIG != config:
            raise ValueError(f"big_leaf Swift 已按 {_ACTIVE_CONFIG} 启用，不能改为 {config}")
        _export_config(config)
        return
    if "XTE_DISABLE_FAST_BF16_CACHE" in os.environ \
            and not _env_bool("XTE_DISABLE_FAST_BF16_CACHE"):
        raise ValueError(
            "XME-free train/eval/save/load 要求 XTE_DISABLE_FAST_BF16_CACHE=1；"
            "BF16 pin 会原地覆盖 Parameter 的导出视图，当前不作静默支持")
    _export_config(config)

    from ttorch.bridge import framework_patch as fw
    from ttorch.bridge.framework_patch import FunctionPatch

    patches = []
    if big_leaf:
        patches.append(FunctionPatch(
            dotted=runtime.DDP_DOTTED,
            module=runtime.DDP_MODULE,
            attr=runtime.DDP_ATTR,
            wrapper=runtime.ddp_init_wrapper,
            intent="big_leaf canonical DDP layout and main_grad lifecycle",
        ))
    patches.extend([
        FunctionPatch(
            dotted=lifecycle.TRAIN_STEP_DOTTED,
            module=lifecycle.SWIFT_TRAINER_MODULE,
            attr=lifecycle.TRAIN_STEP_ATTR,
            wrapper=lifecycle.train_step_wrapper,
            intent="invalidate fast-FC stateful cache after optimizer step",
        ),
        FunctionPatch(
            dotted=lifecycle.LOAD_CHECKPOINT_DOTTED,
            module=lifecycle.SWIFT_TRAINER_MODULE,
            attr=lifecycle.LOAD_CHECKPOINT_ATTR,
            wrapper=lifecycle.load_checkpoint_wrapper,
            intent="invalidate fast-FC stateful cache after checkpoint load",
        ),
        FunctionPatch(
            dotted=lifecycle.SAVE_CHECKPOINT_DOTTED,
            module=lifecycle.SWIFT_TRAINER_MODULE,
            attr=lifecycle.SAVE_CHECKPOINT_ATTR,
            wrapper=lifecycle.save_checkpoint_wrapper,
            intent="synchronize logical parameters before dist/HF checkpoint export",
        ),
    ])

    # CLI 在启用后才会启动训练，直接导入目标并核验 applied，杜绝 pending/静默缺补丁。
    for patch in patches:
        importlib.import_module(patch.module)
        _register(patch)
    fw.install(verbose=False)
    status = fw.patch_status()
    failed = {p.dotted: status.get(p.dotted) for p in patches
              if status.get(p.dotted) != "applied"}
    if failed:
        raise RuntimeError(f"big_leaf Swift 补丁未完整生效: {failed}")
    _ACTIVE_CONFIG = config
    print(f"[tt] big_leaf={'ON' if big_leaf else 'OFF'} forward_mode=patch "
          f"fast_fc={'ON' if _env_bool('XMLIR_ENABLE_FAST_FC') else 'OFF'} "
          "fast_fc_lifecycle=ON", flush=True)


def enable_from_env():
    """供显式 CLI 与 worker 的 ttorch.activate 共用同一个总开关。"""
    enable(big_leaf=_env_bool(ENV_VAR))


def main():
    import ttorch
    ttorch.activate()
    enable_from_env()
    from swift.cli._megatron.main import cli_main
    cli_main()


if __name__ == "__main__":
    main()
