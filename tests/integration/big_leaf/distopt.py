"""真实 Megatron DDP + DistributedOptimizer 的 big_leaf 双线实体 DP 测试。

用法：torchrun --standalone --nproc-per-node={1,2} tests/integration/big_leaf/distopt.py [--overlap]

与旧版（per-weight{i} 实体 + flush）的差异（2026-11-19 双线改版）：
- DDP/dist-opt 的参数实体是内部线 big/big_b（独立 nn.Parameter），外部线
  weight{i}/bias{i} 为其存储切片视图——optimizer 单实体与 DDP 元数据收益；
- 不再有 flush hook：big.grad 走标准 DDP main_grad/reduce 路径；
- restore 后不能继续训练（DDP 梯度实体是内部线），只验证 eval 可用与值完好。
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


def _reference_grads(plan, reference):
    """内部线实体对应的 stock 参考梯度（按实体顺序拼接 per-kind 梯度）。"""
    mod_names = {id(m): n for n, m in plan.model.named_modules()}
    ref = dict(reference.named_parameters())
    out = {}
    for module in plan.modules_to_bind:
        base = mod_names[id(module)]
        out[id(plan.proxies[id(module)][0])] = torch.cat(
            [ref[f"{base}.weight{i}"].grad.float().view(-1)
             for i in range(module.num_gemms)])
        if module.use_bias:
            out[id(plan.proxies[id(module)][1])] = torch.cat(
                [ref[f"{base}.bias{i}"].grad.float().view(-1)
                 for i in range(module.num_gemms)])
    return out


def _owned_range(optimizer, param):
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    owner = next((opt for opt in optimizers if param in opt.model_param_gbuf_map), None)
    if owner is None:
        return None
    return owner._get_model_param_range_map(param)["param"]


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
            plan.entities(), ddp_config.bucket_size, world_size, ddp_config, world_size)
        with plan.entity_swap():
            ddp = DistributedDataParallel(
                config=config, ddp_config=ddp_config, module=model, full_param_layout=layout)
        plan.attach_ddp(ddp)
    if args.overlap:
        bucket_counts = [len(group.buckets) for group in
                         ddp.bucket_groups + ddp.expert_parallel_bucket_groups]
        assert sum(bucket_counts) >= 2, bucket_counts

    # 结构断言：外部线 weight{i} 是内部线实体存储的切片视图。
    for binding in plan.bindings:
        for i in range(binding.module.num_gemms):
            w = getattr(binding.module, f"weight{i}")
            assert w.data.data_ptr() == binding.buf[i].data_ptr()
        assert binding.big.data_ptr() == binding.buf.data_ptr()
    # 结构断言：dist-opt 参数实体是内部线 + 其余参数，外部线 weight{i} 不在其列。
    with plan.entity_swap():
        optimizer = get_megatron_optimizer(
            config=OptimizerConfig(
                optimizer="adam", lr=1e-3, bf16=True, use_distributed_optimizer=True,
                adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8, weight_decay=0.01,
            ),
            model_chunks=[ddp],
        )
    # 结构断言：dist-opt 的 model 侧参数实体全是内部线；外部线 weight{i} 不进入
    # optimizer；shard 按实体级建立（G 倍缩减）。
    # （optimizer.param_groups 里是本 rank 拥有的 shard 张量；model_param_gbuf_map
    # 在 overlap 形态会含 gbuf=None 的簿记条目，拥有判据用 shard range 可取性。）
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    map_keys = set()
    for opt in optimizers:
        map_keys.update(opt.model_param_gbuf_map.keys())
    entities = set(plan.entities())
    expert_ids = {id(p) for p in model.parameters()}
    assert map_keys and map_keys <= entities, "optimizer 参数实体必须全部是内部线"
    assert not (map_keys & expert_ids), "外部线 weight{i} 混入 optimizer 参数实体"
    # G 倍缩减判据：shard 按实体级建立。非 overlap 形态 world 合计 == 实体数；
    # overlap 形态 mcore 对 first/rest chunk 的簿记会多出少量 shard 张量
    # （2026-11-19 实测 6 vs 4），只断言严格少于 stock 口径（G× 专家参数数）。
    count = torch.tensor(
        [sum(len(g["params"]) for g in optimizer.param_groups)], device="cuda")
    dist.all_reduce(count)
    stock_parity = G * len(plan.modules_to_bind) * 2
    if args.overlap:
        assert count.item() < stock_parity, (rank, count.item(), stock_parity)
    else:
        assert count.item() == len(entities), (rank, count.item(), len(entities))

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
        assert all(b.big.grad is None for b in plan.bindings)
        assert all(b.big_b.grad is None for b in plan.bindings)

        if iteration == 0:
            expected_grads = _reference_grads(plan, reference)
            compared_elements = 0
            for proxy in plan.entities():
                param_range = _owned_range(optimizer, proxy)
                if param_range is None:
                    continue
                actual = proxy.main_grad.view(-1)[param_range.start:param_range.end]
                expected = expected_grads[id(proxy)][param_range.start:param_range.end]
                assert torch.equal(actual, expected), (
                    rank, param_range, (actual - expected).abs().max())
                compared_elements += actual.numel()
            assert compared_elements > 0

        before = [b.buf.clone() for b in plan.bindings]
        optimizer.step()
        # overlap 形态 gather 在 zero_grad 时已异步启动，handle 可能残留——
        # force_sync 是 mcore 文档化安全路径（与 save_checkpoint_wrapper 一致）。
        ddp.start_param_sync(force_sync=args.overlap)
        plan.assert_healthy()
        for binding, old in zip(plan.bindings, before):
            assert not torch.equal(binding.buf, old), (rank, iteration, "权重未更新")
            for i in range(binding.module.num_gemms):
                w = getattr(binding.module, f"weight{i}")
                assert w.data.data_ptr() == binding.buf[i].data_ptr()
        torch.cuda.synchronize()

    # no_sync 梯度累积：两次 backward 后 main_grad 应等于单次参考的 2 倍。
    reference_current = init_model()
    reference_current.load_state_dict(model.state_dict())
    reference_current(x, m_splits).float().square().mean().backward()
    reference_current.zero_grad()
    reference_current(x, m_splits).float().square().mean().backward()
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    with ddp.no_sync():
        ddp(x, m_splits).float().square().mean().backward()
    if args.overlap:
        groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
        assert all(group.grad_reduce_handle is None for group in groups)
    ddp(x, m_splits).float().square().mean().backward()
    ddp.finish_grad_sync()
    single_grads = _reference_grads(plan, reference_current)
    for proxy in plan.entities():
        param_range = _owned_range(optimizer, proxy)
        if param_range is None:
            continue
        actual = proxy.main_grad.view(-1)[param_range.start:param_range.end]
        expected = single_grads[id(proxy)][param_range.start:param_range.end]
        assert torch.equal(actual, 2 * expected), (rank, "no_sync 累积")

    # restore 语义（2026-11-19 收紧）：摘除后 eval forward 可用、值完好；
    # 不再支持 restore 后继续训练（DDP 梯度实体是内部线）。
    entity_count = len(plan.entities())
    before_pairs = [(b.buf, b.buf.clone()) for b in plan.bindings]
    plan.restore()
    with torch.no_grad():
        out = model(x, m_splits)
    for buf, old in before_pairs:
        assert torch.equal(buf, old), (rank, "restore 后权重值漂移")
    dist.barrier()
    if rank == 0:
        print("BIG_LEAF_DISTOPT PASS " + json.dumps({
            "dp": world_size, "overlap": args.overlap, "forward_mode": args.forward_mode,
            "entities": entity_count,
        }), flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
