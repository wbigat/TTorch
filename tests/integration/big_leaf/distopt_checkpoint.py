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


def build_instance(big_leaf=True):
    model = init_model()
    plan = BigLeafMegatronPlan.prepare(model, forward_mode="patch") if big_leaf else None
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=False,
        overlap_param_gather=False, grad_reduce_in_fp32=True,
    )
    world_size = dist.get_world_size()
    layout = plan.compute_full_param_layout(
        list(model.parameters()), None, world_size, ddp_config, world_size) if big_leaf else None
    config = TransformerConfig(
        num_layers=1, hidden_size=FIN, num_attention_heads=1,
        num_moe_experts=G, moe_ffn_hidden_size=FOUT,
        bf16=True, params_dtype=torch.bfloat16,
    )
    ddp = DistributedDataParallel(config, ddp_config, model, full_param_layout=layout)
    if big_leaf:
        plan.attach_ddp(ddp)
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
    optimizer_sd = optimizer.sharded_state_dict(model_sd, metadata=metadata)
    dist_checkpointing.save({"model": model_sd, "optimizer": optimizer_sd},
                            ckpt_dir, async_sharded_save=False)
    dist.barrier()

    reference_loss = train_step(model, plan, ddp, optimizer, x)
    reference_params = {name: p.detach().clone() for name, p in model.named_parameters()}

    resumed, resumed_plan, resumed_ddp, resumed_optimizer = build_instance(
        big_leaf=not args.target_stock)
    resumed_model_sd = sharded_state_dict_default(resumed)
    resumed_optimizer_sd = resumed_optimizer.sharded_state_dict(
        resumed_model_sd, is_loading=True, metadata=metadata)
    loaded = dist_checkpointing.load(
        {"model": resumed_model_sd, "optimizer": resumed_optimizer_sd}, ckpt_dir)
    resumed.load_state_dict(loaded["model"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
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
