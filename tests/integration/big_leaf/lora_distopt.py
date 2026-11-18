"""真实 LoRA + DP=2 DistributedOptimizer 的 stock/big_leaf 逐位对拍。"""
import os

import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from hydrax.xaccelerator.stateful import reinitialize_all_after_param_update

from ttorch.bridge.big_leaf import BigLeafMegatronPlan
from lora import make_model, forward


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    mpu.initialize_model_parallel()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    stock = make_model()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    big = make_model()
    big.load_state_dict(stock.state_dict())
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=False,
        overlap_param_gather=False, grad_reduce_in_fp32=True)
    stock_ddp = DistributedDataParallel(stock.config, ddp_config, stock)
    plan = BigLeafMegatronPlan.prepare(big, forward_mode="patch")
    world = dist.get_world_size()
    layout = plan.compute_full_param_layout(
        list(big.parameters()), None, world, ddp_config, world)
    big_ddp = DistributedDataParallel(big.config, ddp_config, big, full_param_layout=layout)
    plan.attach_ddp(big_ddp)
    optimizer_config = OptimizerConfig(
        optimizer="adam", lr=1e-3, bf16=True, use_distributed_optimizer=True,
        weight_decay=0.01)
    stock_optimizer = get_megatron_optimizer(optimizer_config, model_chunks=[stock_ddp])
    big_optimizer = get_megatron_optimizer(optimizer_config, model_chunks=[big_ddp])

    torch.manual_seed(2468)
    x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    expected = forward(stock_ddp, x)
    actual = forward(big_ddp, x)
    assert torch.equal(actual, expected)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    stock_params = dict(stock.named_parameters())
    pre_reduce = {name: p.main_grad.clone() for name, p in big.named_parameters()
                  if p.requires_grad}
    assert all(torch.equal(pre_reduce[name], stock_params[name].main_grad)
               for name in pre_reduce)
    stock_ddp.finish_grad_sync()
    big_ddp.finish_grad_sync()

    stock_optimizers = getattr(stock_optimizer, "chained_optimizers", [stock_optimizer])
    big_optimizers = getattr(big_optimizer, "chained_optimizers", [big_optimizer])
    compared = 0
    for name, big_param in big.named_parameters():
        if not big_param.requires_grad:
            continue
        bucket_group = big_ddp.param_to_bucket_group[big_param]
        scaling = bucket_group.buckets[0].gradient_scaling_factor
        expected_full = pre_reduce[name] * scaling
        dist.all_reduce(expected_full, group=big_ddp.intra_dp_cp_group)
        expected_full = expected_full.view(-1)
        for param, optimizers in ((stock_params[name], stock_optimizers),
                                  (big_param, big_optimizers)):
            owner = next((opt for opt in optimizers if param in opt.model_param_gbuf_map), None)
            if owner is None:
                continue
            param_range = owner._get_model_param_range_map(param)["param"]
            actual_shard = param.main_grad.view(-1)[param_range.start:param_range.end]
            expected_shard = expected_full[param_range.start:param_range.end]
            assert torch.equal(actual_shard, expected_shard), (name, param_range)
            compared += actual_shard.numel()
    assert compared > 0
    stock_optimizer.step()
    big_optimizer.step()
    reinitialize_all_after_param_update(big_optimizer, (), {})
    stock_ddp.start_param_sync()
    big_ddp.start_param_sync()
    plan.assert_healthy()
    for name, p in big.named_parameters():
        assert torch.equal(p, dict(stock.named_parameters())[name]), name

    stock.merge(safe_merge=True)
    big.merge(safe_merge=True)
    plan.assert_healthy()
    assert torch.equal(forward(big_ddp, x), forward(stock_ddp, x))
    stock.unmerge()
    big.unmerge()
    plan.assert_healthy()
    assert torch.equal(forward(big_ddp, x), forward(stock_ddp, x))
    dist.barrier()
    if dist.get_rank() == 0:
        print("BIG_LEAF_LORA_DISTOPT PASS", flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
