"""Megatron DDP 的 big_leaf 生命周期接线。

只包装 ``DistributedDataParallel.__init__``：构造前注入 canonical layout，构造后
attach big_leaf。模型、optimizer、checkpoint 仍由框架原实现管理。
"""
import functools

import torch


DDP_MODULE = "megatron.core.distributed.distributed_data_parallel"
DDP_ATTR = "DistributedDataParallel.__init__"
DDP_DOTTED = f"{DDP_MODULE}.{DDP_ATTR}"


def _has_grouped_linear(module):
    from transformer_engine.pytorch import GroupedLinear
    return any(isinstance(child, GroupedLinear) for child in module.modules())


def ddp_init_wrapper(orig):
    """在 DDP 构造边界完成 prepare → layout → DDP → attach。"""
    @functools.wraps(orig)
    def wrapped(self, config, ddp_config, module, disable_bucketing=False,
                pg_collection=None, full_param_layout=None):
        # 多模态模型会为 visual 等独立子模型各建一个 DDP；非 MoE 实例必须原样通过。
        if not _has_grouped_linear(module):
            return orig(self, config, ddp_config, module,
                        disable_bucketing=disable_bucketing,
                        pg_collection=pg_collection,
                        full_param_layout=full_param_layout)
        if full_param_layout is not None:
            raise ValueError("big_leaf 必须自行构造 full_param_layout，不接受外部 layout")
        if not ddp_config.use_distributed_optimizer:
            raise ValueError("big_leaf Megatron 入口要求 use_distributed_optimizer=True")
        if pg_collection is not None:
            raise ValueError("big_leaf 当前入口不支持自定义 pg_collection")

        from megatron.core import parallel_state as mpu
        from .megatron import BigLeafMegatronPlan

        if mpu.get_pipeline_model_parallel_world_size() != 1:
            raise ValueError("big_leaf 当前生产入口仅支持 PP=1；PP/VPP 尚未验收")
        forward_mode = "patch"
        plan = BigLeafMegatronPlan.prepare(module, forward_mode=forward_mode)
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        bucket_size = None if disable_bucketing or pp_rank > 0 else ddp_config.bucket_size
        layout = plan.compute_full_param_layout(
            list(module.parameters()),
            bucket_size,
            mpu.get_data_parallel_world_size(with_context_parallel=True),
            ddp_config,
            mpu.get_expert_data_parallel_world_size(),
        )
        orig(self, config, ddp_config, module,
             disable_bucketing=disable_bucketing,
             pg_collection=pg_collection,
             full_param_layout=layout)
        plan.attach_ddp(self)
        self._tt_big_leaf_plan = plan
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[tt] big_leaf DDP attached: modules={len(plan.bindings)}, "
                  f"forward_mode={forward_mode}", flush=True)
    return wrapped
