"""DistributedOptimizer 构造边界的实体换装（双线设计的外部线守护）。

mcore 的 `get_megatron_optimizer` 通过 `model_chunk.named_parameters()` 收集
参数（`megatron/core/optimizer/__init__.py::_get_param_groups`），与 DDP 一样
必须让内部线实体（big/big_b）在该窗口内可见，optimizer 的 param_groups 才
持有内部线引用；构造结束后恢复，外部世界对内部线零感知。

为什么补在 trainer 方法而不是 get_megatron_optimizer 本身：swift 的调用点
是 `BaseMegatronTrainer.get_optimizer_and_scheduler`，方法体内还有
`get_optimizer_param_scheduler` 等动作，整方法窗口换装是覆盖该调用点的
最小侵入（换装只影响 GroupedLinear 的 parameters()/named_parameters()，
窗口内其余动作不消费它们）。
"""
import contextlib
import functools
import inspect

from .lifecycle import SWIFT_TRAINER_MODULE

OPTIMIZER_ATTR = "BaseMegatronTrainer.get_optimizer_and_scheduler"
OPTIMIZER_DOTTED = f"{SWIFT_TRAINER_MODULE}.{OPTIMIZER_ATTR}"

# mcore 的 optimizer state 序列化入口：生产 save/load 在 swift 内部调用它，
# 换装必须覆盖到这个边界（model_sharded_state_dict 参数要在换装窗口内重建，
# 使 `param_to_sharded_metadata` 的键 == 内部线实体）。
MCORE_SD_MODULE = "megatron.core.optimizer.distrib_optimizer"
MCORE_SD_ATTR = "DistributedOptimizer.sharded_state_dict"
MCORE_SD_DOTTED = f"{MCORE_SD_MODULE}.{MCORE_SD_ATTR}"

# wrapped() 按位置转发这些参数，必须与 mcore 的实际签名前缀逐一对齐。
_SD_EXPECTED_PARAMS = ("model_sharded_state_dict", "is_loading", "sharding_type",
                       "metadata")


def _assert_sd_signature(orig):
    """安装期校验 mcore DistributedOptimizer.sharded_state_dict 签名未漂移。

    本补丁按**位置**转发（见 wrapped），一旦 mcore 增删/重排参数，按位置
    转发会静默错位而非报错——framework_patch 的运行期复核只看符号是否被
    顶替，查不到签名漂移。这里锚定参数名与序，不匹配立即 fail-fast（启动期
    暴露，而不是等到保存/加载时才炸或更糟地静默写坏）。
    """
    names = [n for n in inspect.signature(orig).parameters if n != "self"]
    if names[:len(_SD_EXPECTED_PARAMS)] != list(_SD_EXPECTED_PARAMS):
        raise RuntimeError(
            f"mcore DistributedOptimizer.sharded_state_dict 签名已漂移："
            f"期望前缀 {_SD_EXPECTED_PARAMS}，实际 {names}；"
            "big_leaf 的按位置转发补丁需重新对齐 optimizer.py")


def _rebuilt_sharded_state_dict(plan):
    """换装窗口内按 mcore 的 plain-module fallback 重建 lookup dict。

    `sharded_state_dict_default` 会优先调用 MegatronModule 自带的
    `sharded_state_dict()`，其换装窗口内的产出与内部线实体不匹配（实测
    ValueError: Model param not in model_sharded_state_dict）；fallback 路径
    （`state_dict(keep_vars=True)` + `make_sharded_tensors_for_checkpoint`）
    把参数对象原样包成 ShardedTensor，正是 optimizer 侧
    `param_to_sharded_metadata` 查表需要的键。TP>1 的 axis-map 形态不在
    本轮范围（整网验证 TP=1）。
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.utils import (
        ensure_metadata_has_dp_cp_group,
        make_sharded_tensors_for_checkpoint,
    )
    metadata = ensure_metadata_has_dp_cp_group({})
    module_sd = plan.model.state_dict(prefix="", keep_vars=True)
    return make_sharded_tensors_for_checkpoint(
        module_sd, "", None, (),
        tp_group=mpu.get_tensor_model_parallel_group(),
        dp_cp_group=metadata["dp_cp_group"],
    )


def _active_plans(optimizer):
    plans = []
    for chunk in getattr(optimizer, "model_chunks", None) or []:
        plan = getattr(chunk, "_tt_big_leaf_plan", None)
        if plan is not None and plan._active and plan not in plans:
            plans.append(plan)
    return plans


def optimizer_boundary_wrapper(orig):
    """optimizer 构建窗口内换装，构建后恢复；无 plan 时原样通过。"""
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        plans = []
        for model in getattr(self, "wrapped_models", None) or []:
            plan = getattr(model, "_tt_big_leaf_plan", None)
            if plan is not None and plan._active and plan not in plans:
                plans.append(plan)
        if not plans:
            return orig(self, *args, **kwargs)
        with contextlib.ExitStack() as stack:
            for plan in plans:
                stack.enter_context(plan.entity_swap())
            return orig(self, *args, **kwargs)
    return wrapped


def mcore_sharded_state_dict_wrapper(orig):
    """DistributedOptimizer.sharded_state_dict 的换装 + 格式闸门。

    - 换装：model_sharded_state_dict 在换装窗口内重建，optimizer 侧按内部线
      实体查表（外部线 dict 里没有实体，会 KeyError——2026-11-19 整网实测）；
      模型侧 state dict 由调用方另行保存，键契约不受影响。
    - 格式闸门：双线实体版目前只验证了 fully-reshardable（dp_reshardable 的
      桶内 pad 尺寸计算与实体 layout 不兼容，实测 torch.empty(负数) 崩溃）。
      显式 fail-fast，不让它静默写出坏 checkpoint。
    """
    @functools.wraps(orig)
    def wrapped(self, model_sharded_state_dict=None, is_loading=False,
                sharding_type=None, metadata=None):
        plans = _active_plans(self)
        if not plans:
            return orig(self, model_sharded_state_dict, is_loading, sharding_type,
                        metadata)
        sharding = (metadata or {}).get("distrib_optim_sharding_type")
        if sharding != "fully_reshardable":
            # 缺失键时 mcore 默认 dp_reshardable，与实体 layout 不兼容——
            # 必须显式要求 fully-reshardable，不允许落到默认格式静默崩溃。
            if is_loading:
                raise ValueError(
                    f"big_leaf 双线实体版只能加载 fully-reshardable optimizer "
                    f"checkpoint，当前 distrib_optim_sharding_type={sharding!r}；"
                    "非 fully-reshardable 的旧 ckpt 需先用 distopt_state 迁移转换器转换")
            raise ValueError(
                f"big_leaf 双线实体版只支持 fully-reshardable optimizer checkpoint，"
                f"当前 distrib_optim_sharding_type={sharding!r}；请设置 "
                "dist_ckpt_optim_fully_reshardable=true")
        with contextlib.ExitStack() as stack:
            for plan in plans:
                stack.enter_context(plan.entity_swap())
                model_sharded_state_dict = _rebuilt_sharded_state_dict(plan)
            return orig(self, model_sharded_state_dict, is_loading, sharding_type,
                        metadata)
    _assert_sd_signature(orig)
    return wrapped

