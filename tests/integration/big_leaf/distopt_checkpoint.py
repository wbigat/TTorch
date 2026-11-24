"""big_leaf DDP layout + DistributedOptimizer fully-reshardable checkpoint 验证。"""
import argparse
import json
import os
import shutil
import tempfile

import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

from ttorch.bridge.big_leaf import BigLeafMegatronPlan
from distopt import G, FIN, FOUT, TOK, Model, init_model


def optimizer_model_sd(model, plan):
    """optimizer 侧 sharded state dict：big_leaf 时必须在换装窗口内构造，
    使 `param_to_sharded_metadata` 的键 == 内部线实体（mcore 按参数对象查表）。
    模型侧 checkpoint 仍用外部线 dict（weight{i} 键契约不变）。"""
    if plan is None:
        return sharded_state_dict_default(model)
    with plan.entity_swap():
        return sharded_state_dict_default(model)


def build_instance(big_leaf=True):
    model = init_model()
    plan = BigLeafMegatronPlan.prepare(model, forward_mode="patch") if big_leaf else None
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=False,
        overlap_param_gather=False, grad_reduce_in_fp32=True,
    )
    world_size = dist.get_world_size()
    layout = plan.compute_full_param_layout(
        plan.entities(), None, world_size, ddp_config, world_size) if big_leaf else None
    config = TransformerConfig(
        num_layers=1, hidden_size=FIN, num_attention_heads=1,
        num_moe_experts=G, moe_ffn_hidden_size=FOUT,
        bf16=True, params_dtype=torch.bfloat16,
    )
    if big_leaf:
        with plan.entity_swap():
            ddp = DistributedDataParallel(
                config, ddp_config, model, full_param_layout=layout)
        plan.attach_ddp(ddp)
    else:
        ddp = DistributedDataParallel(config, ddp_config, model, full_param_layout=layout)
    if big_leaf:
        with plan.entity_swap():
            optimizer = get_megatron_optimizer(
                OptimizerConfig(optimizer="adam", lr=1e-3, bf16=True,
                                use_distributed_optimizer=True, weight_decay=0.01),
                model_chunks=[ddp])
    else:
        optimizer = get_megatron_optimizer(
            OptimizerConfig(optimizer="adam", lr=1e-3, bf16=True,
                            use_distributed_optimizer=True, weight_decay=0.01),
            model_chunks=[ddp])
    return model, plan, ddp, optimizer


def train_step(model, plan, ddp, optimizer, x):
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    loss = ddp(x, [TOK] * G).float().square().mean()
    loss.backward()
    ddp.finish_grad_sync()
    optimizer.step()
    ddp.start_param_sync()
    if plan is not None:
        plan.assert_healthy()
    return loss.detach().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-stock", action="store_true")
    parser.add_argument("--target-stock", action="store_true")
    parser.add_argument("--ckpt-dir")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    mpu.initialize_model_parallel()
    ckpt_dir = args.ckpt_dir
    if rank == 0 and ckpt_dir is None:
        ckpt_dir = tempfile.mkdtemp(prefix="big_leaf_distopt_checkpoint_")
    shared_dir = [ckpt_dir]
    dist.broadcast_object_list(shared_dir, src=0)
    ckpt_dir = shared_dir[0]
    model, plan, ddp, optimizer = build_instance(big_leaf=not args.source_stock)
    torch.manual_seed(777)
    x = torch.randn(G * TOK, FIN, device="cuda", dtype=torch.bfloat16)
    train_step(model, plan, ddp, optimizer, x)

    metadata = {
        "distrib_optim_sharding_type": "fully_reshardable",
        "chained_optim_avoid_prefix": True,
    }
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()
    model_sd = sharded_state_dict_default(model)
    optimizer_sd = optimizer.sharded_state_dict(
        optimizer_model_sd(model, plan), metadata=metadata)
    dist_checkpointing.save({"model": model_sd, "optimizer": optimizer_sd},
                            ckpt_dir, async_sharded_save=False)
    dist.barrier()

    reference_loss = train_step(model, plan, ddp, optimizer, x)
    reference_params = {name: p.detach().clone() for name, p in model.named_parameters()}

    resumed, resumed_plan, resumed_ddp, resumed_optimizer = build_instance(
        big_leaf=not args.target_stock)
    resumed_model_sd = sharded_state_dict_default(resumed)
    resumed_optimizer_sd = resumed_optimizer.sharded_state_dict(
        optimizer_model_sd(resumed, resumed_plan), is_loading=True, metadata=metadata)
    cross_format = args.source_stock != args.target_stock
    if cross_format:
        # 跨格式：模型先行加载（键契约与格式无关），optimizer 状态走转换路径。
        loaded = dist_checkpointing.load({"model": resumed_model_sd}, ckpt_dir)
        resumed.load_state_dict(loaded["model"])
    else:
        loaded = dist_checkpointing.load(
            {"model": resumed_model_sd, "optimizer": resumed_optimizer_sd}, ckpt_dir)
        resumed.load_state_dict(loaded["model"])
        resumed_optimizer.load_state_dict(loaded["optimizer"])
    if cross_format:
        # 双线格式转换（expert ↔ entity）。读盘用**源格式参考实例**的
        # sharded_state_dict（mcore 原生机制）——这是无源恢复的正道：不需要原始
        # 训练实例，只需构造一个源格式的参考 model+DDP+optimizer 即可生成读模板。
        # （review 1d 已确认：手搓 build_param_state_template 的裸 from_rank_offsets
        # 缺 fully-reshardable 的真实分片元数据，驱动不了真实 load——本测试实测
        # "Invalid access pattern" 复现。它降级为布局对照诊断，见 distopt_state。）
        from ttorch.bridge.big_leaf import distopt_state as dstate
        ddp_config = resumed_ddp.ddp_config
        world = dist.get_world_size()
        # 布局对照诊断：position 推导的 FQN key/shape 必须与源格式参考实例的真实
        # 键完全一致（保证 convert 的键推导正确），但不用于驱动 load。
        if args.source_stock:
            src_positions = dstate.stock_positions(resumed, ddp_config, world, world)
        else:
            src_positions = dstate.entity_positions(plan, ddp_config, world, world)
        src_osd = optimizer.sharded_state_dict(
            optimizer_model_sd(model, plan), is_loading=True, metadata=metadata)
        built = dstate.build_param_state_template(src_positions)
        for idx, entry in src_osd["param_state"].items():
            for sk, sh_t in entry.items():
                assert sk in built[idx] and built[idx][sk].key == sh_t.key \
                    and tuple(built[idx][sk].data.shape) == tuple(sh_t.data.shape), (
                        idx, sk, "position 推导的 key/shape 与真实不符")
        raw = dist_checkpointing.load({"optimizer": src_osd}, ckpt_dir)["optimizer"]
        # 走生产可调用的迁移入口（review 1d：把内联逻辑抽为公开 API）。
        converted = dstate.migrate_param_state_to_target(
            raw["param_state"],
            source_format="expert" if args.source_stock else "entity",
            model=resumed, target_plan=resumed_plan, ddp_config=ddp_config,
            data_parallel_world_size=world, expert_data_parallel_world_size=world,
            source_plan=plan)
        resumed_optimizer.load_state_dict({
            "optimizer": raw["optimizer"],
            "param_state": converted,
            "param_state_sharding_type": "fully_reshardable",
        })
    if resumed_plan is not None:
        resumed_plan.assert_healthy()

    resumed_loss = train_step(resumed, resumed_plan, resumed_ddp, resumed_optimizer, x)
    assert torch.equal(resumed_loss, reference_loss), (rank, resumed_loss, reference_loss)
    for name, p in resumed.named_parameters():
        assert torch.equal(p, reference_params[name]), (
            rank, name, (p - reference_params[name]).abs().max())

    dist.barrier()
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print("BIG_LEAF_DISTOPT_CHECKPOINT PASS " + json.dumps({
            "dp": dist.get_world_size(), "format": "fully_reshardable",
            "source": "stock" if args.source_stock else "big_leaf",
            "target": "stock" if args.target_stock else "big_leaf",
            "parameters_compared": len(reference_params),
        }), flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
