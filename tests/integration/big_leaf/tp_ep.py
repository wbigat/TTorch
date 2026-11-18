"""真实 mcore_bridge TP/EP + DDP/dist-opt 的 big_leaf 验证。

TP_SIZE=2 EP_SIZE=1 torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/tp_ep.py
TP_SIZE=1 EP_SIZE=2 torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/tp_ep.py
"""
import json
import os

import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from hydrax.xaccelerator.stateful import reinitialize_all_after_param_update

import full_model as vfm
from ttorch.bridge.big_leaf import BigLeafMegatronPlan


def sharded_weights(model):
    result = {}
    if not hasattr(model, "sharded_state_dict"):
        return result
    sd = model.sharded_state_dict()
    for key, value in sd.items():
        if "experts.linear_fc" not in key or "weight" not in key:
            continue
        local = getattr(value, "data", value)
        if torch.is_tensor(local):
            result[key] = local.detach().clone()
    return result


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    vfm.initialize_model_parallel()
    tp = int(os.environ.get("TP_SIZE", "1"))
    ep = int(os.environ.get("EP_SIZE", "1"))
    model_parallel_cuda_manual_seed(1234)
    if rank == 0:
        print(f"STAGE build models tp={tp} ep={ep}", flush=True)
    stock, _ = vfm.build_model()
    model, config = vfm.build_model()
    vfm.deterministic_full_init(stock, seed=0)
    vfm.deterministic_full_init(model, seed=0)
    model.load_state_dict(stock.state_dict())

    assert set(stock.state_dict()) == set(model.state_dict())
    stock_shards = sharded_weights(stock)
    big_shards = sharded_weights(model)
    assert set(stock_shards) == set(big_shards) and stock_shards
    for key in stock_shards:
        assert torch.equal(stock_shards[key], big_shards[key]), key

    plan = BigLeafMegatronPlan.prepare(model, forward_mode="patch")
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        grad_reduce_in_fp32=True,
        check_for_nan_in_grad=True,
    )
    dp_size = mpu.get_data_parallel_world_size()
    edp_size = mpu.get_expert_data_parallel_world_size()
    layout = plan.compute_full_param_layout(
        list(model.parameters()), None, dp_size, ddp_config, edp_size)
    ddp = DistributedDataParallel(
        config=config, ddp_config=ddp_config, module=model, full_param_layout=layout)
    plan.attach_ddp(ddp)
    if rank == 0:
        print("STAGE DDP attached; build optimizer", flush=True)
    optimizer = get_megatron_optimizer(
        config=OptimizerConfig(
            optimizer="adam", lr=1e-4, bf16=True, use_distributed_optimizer=True,
            weight_decay=0.01,
        ), model_chunks=[ddp])
    for name, p in model.named_parameters():
        expected_weight = dict(stock.named_parameters())[name]
        assert torch.equal(p, expected_weight), (rank, "DDP 后权重变化", name)

    stock.eval()
    ddp.eval()
    model_modules = dict(model.named_modules())
    stock_modules = dict(stock.named_modules())
    captured_big_grads = {}
    for bi, binding in enumerate(plan.bindings):
        binding.big.register_hook(
            lambda grad, bi=bi: captured_big_grads.setdefault(bi, grad.detach().clone()))

    # 先绕开 MoE router 的跨模型非确定性，逐模块验证真实 TP/EP 分片算子。
    for bi, binding in enumerate(plan.bindings):
        module_name = next(name for name, module in model_modules.items() if module is binding.module)
        stock_module = stock_modules[module_name]
        torch.manual_seed(9000 + bi)
        x = torch.randn(binding.module.num_gemms * 4, binding.module.in_features,
                        device="cuda", dtype=torch.bfloat16)
        ms = [4] * binding.module.num_gemms
        stock_result = stock_module(x, ms)
        big_result = binding.module(x, ms)
        stock_out = stock_result[0] if isinstance(stock_result, tuple) else stock_result
        big_out = big_result[0] if isinstance(big_result, tuple) else big_result
        assert torch.equal(big_out, stock_out), (rank, module_name, "direct forward")
        stock_out.float().square().mean().backward()
        big_out.float().square().mean().backward()
        expected = torch.stack([getattr(stock_module, f"weight{i}").grad.float()
                                for i in range(binding.module.num_gemms)])
        assert torch.equal(captured_big_grads[bi].float(), expected), (
            rank, module_name, "direct grad")
        stock.zero_grad()
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        captured_big_grads.clear()

    combined_parallel = tp > 1 and ep > 1
    if combined_parallel:
        # stock/big 双整网实例会共享 fast FC stateful cache；组合模式只做单 big 整网 smoke，
        # 数值正确性由上面的每个真实 TP/EP GroupedLinear 逐位对拍承担。
        reinitialize_all_after_param_update(optimizer, (), {})
        tokens, labels, positions = vfm.make_batch(0)
        loss = vfm.forward_loss(ddp, tokens, labels, positions)
        assert torch.isfinite(loss)
        before = [b.buf.clone() for b in plan.bindings]
        loss.backward()
        ddp.finish_grad_sync()
        optimizer.step()
        reinitialize_all_after_param_update(optimizer, (), {})
        ddp.start_param_sync()
        plan.assert_healthy()
        for binding, old in zip(plan.bindings, before):
            assert not torch.equal(binding.buf, old) and torch.isfinite(binding.buf.float()).all()
        assert torch.isfinite(vfm.forward_loss(ddp, tokens, labels, positions))
        dist.barrier()
        if rank == 0:
            print("BIG_LEAF_TP_EP PASS " + json.dumps({
                "tp": tp, "ep": ep, "dp": dp_size, "edp": edp_size,
                "bindings": len(plan.bindings), "sharded_weight_keys": len(stock_shards),
                "direct_module_check": "bitwise", "full_model_check": "single-big-finite-smoke",
                "compared_grad_elements": 0,
            }), flush=True)
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
        return

    tokens, labels, positions = vfm.make_batch(0)
    expected_loss = vfm.forward_loss(stock, tokens, labels, positions)
    repeated_expected_loss = vfm.forward_loss(stock, tokens, labels, positions)
    forward_repeat_diff = (expected_loss - repeated_expected_loss).abs().item()
    if rank == 0:
        print("FORWARD_REPEAT_DIAG", forward_repeat_diff, "big_vs_stock", flush=True)
    expected_loss.backward()
    reference_grads = {name: p.grad.float().clone()
                       for name, p in stock.named_parameters() if p.grad is not None}
    stock.zero_grad()
    actual_loss = vfm.forward_loss(ddp, tokens, labels, positions)
    combined_parallel = tp > 1 and ep > 1
    if combined_parallel:
        assert torch.isfinite(actual_loss) and torch.isfinite(expected_loss)
    else:
        assert torch.equal(actual_loss, expected_loss), (rank, actual_loss, expected_loss)
    actual_loss.backward()
    ddp.finish_grad_sync()
    plan.assert_healthy()

    max_autograd_diff = 0.0
    for bi, binding in enumerate(plan.bindings):
        module_name = next(name for name, module in model_modules.items() if module is binding.module)
        expected = torch.stack([reference_grads[f"{module_name}.weight{i}"]
                                for i in range(binding.module.num_gemms)])
        max_autograd_diff = max(max_autograd_diff,
                                (captured_big_grads[bi] - expected).abs().max().item())
    repeated_loss = vfm.forward_loss(stock, tokens, labels, positions)
    repeated_loss.backward()
    max_stock_repeat_diff = max(
        (p.grad.float() - reference_grads[name]).abs().max().item()
        for name, p in stock.named_parameters() if name in reference_grads)
    if rank == 0:
        print("EP_GRAD_DIAG " + json.dumps({
            "big_vs_stock_autograd": max_autograd_diff,
            "stock_vs_repeated_stock": max_stock_repeat_diff,
        }), flush=True)

    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    compared = 0
    if not combined_parallel:
        for name, p in model.named_parameters():
            bucket_group = ddp.param_to_bucket_group[p]
            is_expert = bucket_group in ddp.expert_parallel_bucket_groups
            reduce_group = ddp.intra_expt_dp_group if is_expert else ddp.intra_dp_cp_group
            scaling_factor = bucket_group.buckets[0].gradient_scaling_factor
            expected_full = reference_grads[name].clone()
            expected_full.mul_(scaling_factor)
            if reduce_group.size() > 1:
                dist.all_reduce(expected_full, group=reduce_group)
            expected_full = expected_full.view(-1)

            owner = next((opt for opt in optimizers if p in opt.model_param_gbuf_map), None)
            if owner is None:
                continue
            param_range = owner._get_model_param_range_map(p)["param"]
            actual = p.main_grad.view(-1)[param_range.start:param_range.end]
            expected = expected_full[param_range.start:param_range.end]
            assert torch.equal(actual, expected), (
                rank, name, param_range, (actual - expected).abs().max())
            compared += actual.numel()
        assert compared > 0

    before = [b.buf.clone() for b in plan.bindings]
    optimizer.step()
    reinitialize_all_after_param_update(optimizer, (), {})
    ddp.start_param_sync()
    plan.assert_healthy()
    for binding, old in zip(plan.bindings, before):
        assert not torch.equal(binding.buf, old)
        assert torch.isfinite(binding.buf.float()).all()
        for i in range(binding.module.num_gemms):
            assert getattr(binding.module, f"weight{i}").data.data_ptr() == \
                binding.buf[i].data_ptr()
    next_loss = vfm.forward_loss(ddp, tokens, labels, positions)
    assert torch.isfinite(next_loss)

    dist.barrier()
    if rank == 0:
        print("BIG_LEAF_TP_EP PASS " + json.dumps({
            "tp": tp, "ep": ep, "dp": dp_size, "edp": edp_size,
            "bindings": len(plan.bindings), "sharded_weight_keys": len(stock_shards),
            "full_model_check": "finite-smoke" if combined_parallel else "bitwise",
            "stock_forward_repeat_diff": forward_repeat_diff,
            "compared_grad_elements": compared,
        }), flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
