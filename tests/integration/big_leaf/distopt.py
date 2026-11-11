"""真实 Megatron DDP + DistributedOptimizer 的 big_leaf DP 测试。

用法：torchrun --standalone --nproc-per-node={1,2} tests/integration/big_leaf/distopt.py [--overlap]
"""
import argparse
import json
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from transformer_engine.pytorch import GroupedLinear

from ttorch.bridge.big_leaf import BigLeafMegatronPlan
from ttorch.bridge.big_leaf.runtime import ddp_init_wrapper

G, FIN, FOUT, TOK = 4, 32, 24, 4


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gl1 = GroupedLinear(G, FIN, FOUT, bias=True, params_dtype=torch.bfloat16)
        self.gl2 = GroupedLinear(G, FIN, FOUT, bias=True, params_dtype=torch.bfloat16)

    def forward(self, x, m_splits):
        return self.gl1(x, m_splits) + self.gl2(x, m_splits)


def init_model(seed=1234):
    torch.manual_seed(seed)
    model = Model().cuda()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.03)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--forward-mode", choices=("switch", "patch"), default="patch")
    parser.add_argument("--runtime-wrapper", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    mpu.initialize_model_parallel()
    world_size = dist.get_world_size()

    model = init_model()
    reference = init_model()
    reference.load_state_dict(model.state_dict())
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=args.overlap,
        overlap_param_gather=args.overlap,
        bucket_size=3000 if args.overlap else None,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )
    config = TransformerConfig(
        num_layers=1, hidden_size=FIN, num_attention_heads=1,
        num_moe_experts=G, moe_ffn_hidden_size=FOUT, bf16=True,
        params_dtype=torch.bfloat16,
    )
    if args.runtime_wrapper:
        if args.forward_mode != "patch":
            raise ValueError("生产 runtime wrapper 固定 patch；switch 请使用底层 plan API")
        DistributedDataParallel.__init__ = ddp_init_wrapper(DistributedDataParallel.__init__)
        ddp = DistributedDataParallel(config=config, ddp_config=ddp_config, module=model)
        plan = ddp._tt_big_leaf_plan
    else:
        plan = BigLeafMegatronPlan.prepare(model, forward_mode=args.forward_mode)
        layout = plan.compute_full_param_layout(
            list(model.parameters()), ddp_config.bucket_size, world_size, ddp_config, world_size)
        ddp = DistributedDataParallel(
            config=config, ddp_config=ddp_config, module=model, full_param_layout=layout)
        plan.attach_ddp(ddp)
    if args.overlap:
        bucket_counts = [len(group.buckets) for group in
                         ddp.bucket_groups + ddp.expert_parallel_bucket_groups]
        assert sum(bucket_counts) >= 2, bucket_counts

    for binding in plan.bindings:
        for i in range(binding.module.num_gemms):
            w = getattr(binding.module, f"weight{i}")
            assert w.data.data_ptr() == binding.buf[i].data_ptr()
        assert binding.big.data_ptr() == binding.buf.data_ptr()

    optimizer = get_megatron_optimizer(
        config=OptimizerConfig(
            optimizer="adam", lr=1e-3, bf16=True, use_distributed_optimizer=True,
            adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8, weight_decay=0.01,
        ),
        model_chunks=[ddp],
    )

    torch.manual_seed(5678)
    x = torch.randn(G * TOK, FIN, device="cuda", dtype=torch.bfloat16)
    m_splits = [TOK] * G
    expected_out = reference(x, m_splits)
    expected_loss = expected_out.float().square().mean()
    expected_loss.backward()

    for iteration in range(2):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        out = ddp(x, m_splits)
        loss = out.float().square().mean()
        if iteration == 0:
            assert torch.equal(out, expected_out), (rank, (out - expected_out).abs().max())
        loss.backward()
        ddp.finish_grad_sync()
        plan.assert_healthy()
        assert all(count == iteration + 1 for count in plan.flush_counts.values()), plan.flush_counts
        assert all(b.big.grad is None for b in plan.bindings)
        assert all(b.big_b.grad is None for b in plan.bindings)
        assert all(p.grad is None for p in model.parameters())

        if iteration == 0:
            reference_params = dict(reference.named_parameters())
            compared_elements = 0
            for name, p in model.named_parameters():
                expected_grad = reference_params[name].grad.float().view(-1)
                # dist-opt reduce-scatter 后仅本 rank 拥有的 param 子区间有效。
                optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
                owner = next((opt for opt in optimizers if p in opt.model_param_gbuf_map), None)
                if owner is None:  # 本 rank 不拥有该参数的任何 shard
                    continue
                param_range = owner._get_model_param_range_map(p)["param"]
                actual = p.main_grad.view(-1)[param_range.start:param_range.end]
                expected = expected_grad[param_range.start:param_range.end]
                assert torch.equal(actual, expected), (
                    rank, name, param_range, (actual - expected).abs().max())
                compared_elements += actual.numel()
            assert compared_elements > 0

        before = [b.buf.clone() for b in plan.bindings]
        optimizer.step()
        ddp.start_param_sync()
        plan.assert_healthy()
        for binding, old in zip(plan.bindings, before):
            assert not torch.equal(binding.buf, old), (rank, iteration, "权重未更新")
            for i in range(binding.module.num_gemms):
                w = getattr(binding.module, f"weight{i}")
                assert w.data.data_ptr() == binding.buf[i].data_ptr()
        torch.cuda.synchronize()

    # no_sync 梯度累积：在当前权重上先建立 stock 两次 backward 的参考。
    reference_current = init_model()
    reference_current.load_state_dict(model.state_dict())
    reference_current(x, m_splits).float().square().mean().backward()
    reference_single = {name: p.grad.clone() for name, p in reference_current.named_parameters()}
    reference_current(x, m_splits).float().square().mean().backward()
    for name, p in reference_current.named_parameters():
        assert torch.equal(p.grad, 2 * reference_single[name])
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    counts_before = dict(plan.flush_counts)
    with ddp.no_sync():
        ddp(x, m_splits).float().square().mean().backward()
    if args.overlap:
        groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
        assert all(group.grad_reduce_handle is None for group in groups)
    ddp(x, m_splits).float().square().mean().backward()
    ddp.finish_grad_sync()
    assert all(plan.flush_counts[key] == counts_before[key] + 2 for key in counts_before)
    for name, p in model.named_parameters():
        optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        owner = next((opt for opt in optimizers if p in opt.model_param_gbuf_map), None)
        if owner is None:
            continue
        param_range = owner._get_model_param_range_map(p)["param"]
        actual = p.main_grad.view(-1)[param_range.start:param_range.end]
        expected = dict(reference_current.named_parameters())[name].grad.float().view(-1)[
            param_range.start:param_range.end]
        assert torch.equal(actual, expected), (rank, "no_sync", name)

    plan.restore()
    # restore 只摘 big_leaf，DDP 原参数与 optimizer 仍可用。
    ddp.zero_grad_buffer()
    out = ddp(x, m_splits)
    out.float().square().mean().backward()
    ddp.finish_grad_sync()
    assert all(p.grad is None for p in model.parameters())
    dist.barrier()
    if rank == 0:
        print("BIG_LEAF_DISTOPT PASS " + json.dumps({
            "dp": world_size, "overlap": args.overlap, "forward_mode": args.forward_mode,
            "flush_counts": {str(k): v for k, v in plan.flush_counts.items()},
        }), flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
