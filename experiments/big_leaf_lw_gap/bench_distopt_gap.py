"""big_leaf vs LW=1 在**真实 Megatron DDP + DistributedOptimizer** 口径下的缺口基准。

文档 §6.4/§6.6.2 明确标注待测项：正式自动入口保留原 `weight{i}` 作为 optimizer /
checkpoint 实体，因此「optimizer 单实体 + DDP 元数据缩减」收益没有完整拿到，
「每 rank 的实际 shard 数和 optimizer 开销需单独测量」。本脚本直接测量这项。

三臂（同进程交错采样，抵消设备频率漂移）：
  stock    = LW=0，Megatron 默认 layout（G 个 weight{i} 进 DDP/dist-opt）
  big_leaf = LW=0 + canonical layout + 大叶子 + main_grad flush（参数实体同 stock）
  lw1      = LW=1，每模块 1 个 large_weight Parameter 进 DDP/dist-opt

分相位计时：forward / backward / finish_grad_sync / optimizer.step，并记录
DDP buffer 参数数、bucket 数、dist-opt shard 张量数。

用法：
    torchrun --standalone --nproc-per-node=1 \
        experiments/big_leaf_lw_gap/bench_distopt_gap.py --iters 50 --out distopt.json
"""
import argparse
import json
import os
import statistics
import time

os.environ.setdefault("XTE_USE_GROUPED_FC_FUSION", "0")
os.environ.setdefault("XMLIR_ENABLE_FAST_FC", "1")
os.environ.setdefault("XTE_DISABLE_FAST_BF16_CACHE", "1")  # 与 big_leaf 生产入口一致

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from transformer_engine.pytorch.env_config import xte_env
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear


def _set(name, value):
    xte_env.set(name, value)
    os.environ[name] = value


class MoEStack(torch.nn.Module):
    """L 层 grouped-expert 堆叠，形态对齐 MoE 的 linear_fc1。"""

    def __init__(self, layers, G, fin, fout, large_weight):
        super().__init__()
        _set("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1" if large_weight else "0")
        torch.manual_seed(1234)
        self.layers = torch.nn.ModuleList([
            GroupedLinear(G, fin, fout, bias=False, params_dtype=torch.bfloat16,
                          device="cuda")
            for _ in range(layers)
        ])
        _set("XTE_GROUPED_GEMM_LARGE_WEIGHT", "0")
        self.back = torch.nn.ModuleList()
        _set("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1" if large_weight else "0")
        self.back = torch.nn.ModuleList([
            GroupedLinear(G, fout, fin, bias=False, params_dtype=torch.bfloat16,
                          device="cuda")
            for _ in range(layers)
        ])
        _set("XTE_GROUPED_GEMM_LARGE_WEIGHT", "0")
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.02)

    def forward(self, x, m_splits):
        for up, down in zip(self.layers, self.back):
            h = up(x, m_splits)
            if isinstance(h, tuple):
                h = h[0]
            h = torch.nn.functional.gelu(h)
            y = down(h, m_splits)
            if isinstance(y, tuple):
                y = y[0]
            x = x + y
        return x


class Arm:
    def __init__(self, name, cfg):
        self.name = name
        self.plan = None
        large_weight = name == "lw1"
        self.model = MoEStack(cfg["layers"], cfg["G"], cfg["fin"], cfg["fout"],
                              large_weight)
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            bucket_size=None,
            check_for_nan_in_grad=False,
            grad_reduce_in_fp32=True,
        )
        config = TransformerConfig(
            num_layers=1, hidden_size=cfg["fin"], num_attention_heads=1,
            num_moe_experts=cfg["G"], moe_ffn_hidden_size=cfg["fout"],
            bf16=True, params_dtype=torch.bfloat16,
        )
        world = dist.get_world_size()
        if name == "big_leaf":
            from ttorch.bridge.big_leaf import BigLeafMegatronPlan
            self.plan = BigLeafMegatronPlan.prepare(self.model, forward_mode="patch")
            layout = self.plan.compute_full_param_layout(
                list(self.model.parameters()), ddp_config.bucket_size, world,
                ddp_config, world)
            self.ddp = DistributedDataParallel(
                config=config, ddp_config=ddp_config, module=self.model,
                full_param_layout=layout)
            self.plan.attach_ddp(self.ddp)
        else:
            self.ddp = DistributedDataParallel(
                config=config, ddp_config=ddp_config, module=self.model)
        self.optimizer = get_megatron_optimizer(
            config=OptimizerConfig(
                optimizer="adam", lr=1e-4, bf16=True, use_distributed_optimizer=True,
                adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8, weight_decay=0.01,
            ),
            model_chunks=[self.ddp],
        )
        torch.manual_seed(99)
        tokens = cfg["tokens"]
        self.x = torch.randn(tokens, cfg["fin"], device="cuda", dtype=torch.bfloat16)
        splits = [tokens // cfg["G"]] * cfg["G"]
        splits[-1] += tokens - sum(splits)
        self.m_splits = splits
        self.stats = self._structure_stats()

    def _structure_stats(self):
        buffers = list(self.ddp.buffers) + list(self.ddp.expert_parallel_buffers)
        ddp_params = sum(len(buffer.param_index_map) for buffer in buffers)
        buckets = sum(len(buffer.buckets) for buffer in buffers)
        shard_tensors = 0
        for group in self.optimizer.param_groups:
            shard_tensors += len(group["params"])
        return {
            "ddp_buffer_params": ddp_params,
            "ddp_buckets": buckets,
            "optimizer_shard_tensors": shard_tensors,
            "model_params": len(list(self.model.parameters())),
            "big_leaf_bindings": len(self.plan.bindings) if self.plan else 0,
        }

    def step_phases(self):
        """返回一步内各相位耗时（us），每相位前后设备同步。"""
        out = {}
        self.ddp.zero_grad_buffer()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()

        start = time.perf_counter()
        y = self.ddp(self.x, self.m_splits)
        loss = y.float().square().mean()
        torch.cuda.synchronize()
        out["forward_us"] = (time.perf_counter() - start) * 1e6

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        out["backward_us"] = (time.perf_counter() - start) * 1e6

        start = time.perf_counter()
        self.ddp.finish_grad_sync()
        torch.cuda.synchronize()
        out["grad_sync_us"] = (time.perf_counter() - start) * 1e6

        start = time.perf_counter()
        self.optimizer.step()
        torch.cuda.synchronize()
        out["optimizer_us"] = (time.perf_counter() - start) * 1e6

        out["total_us"] = sum(out.values())
        return out


PHASES = ("forward_us", "backward_us", "grad_sync_us", "optimizer_us", "total_us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--fin", type=int, default=1024)
    parser.add_argument("--fout", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--arms", nargs="*", default=["stock", "big_leaf", "lw1"])
    parser.add_argument("--out")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    mpu.initialize_model_parallel()

    cfg = {"layers": args.layers, "G": args.experts, "fin": args.fin,
           "fout": args.fout, "tokens": args.tokens}
    arms = [Arm(name, cfg) for name in args.arms]
    for arm in arms:
        print(f"[struct] {arm.name:9s} {json.dumps(arm.stats)}", flush=True)

    for _ in range(args.warmup):
        for arm in arms:
            arm.step_phases()
    samples = {arm.name: [] for arm in arms}
    for index in range(args.iters):
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        for arm in order:
            samples[arm.name].append(arm.step_phases())

    report = {"config": cfg, "iters": args.iters, "warmup": args.warmup,
              "arms": {}}
    for arm in arms:
        rows = samples[arm.name]
        report["arms"][arm.name] = {
            "structure": arm.stats,
            "phases": {phase: statistics.median(row[phase] for row in rows)
                       for phase in PHASES},
        }
    if dist.get_rank() == 0:
        header = f"{'arm':10s}" + "".join(f"{p.replace('_us',''):>14s}" for p in PHASES)
        print(header, flush=True)
        for name in args.arms:
            phases = report["arms"][name]["phases"]
            print(f"{name:10s}" + "".join(f"{phases[p]:14.1f}" for p in PHASES),
                  flush=True)
        if {"lw1", "big_leaf"} <= set(args.arms):
            lw1 = report["arms"]["lw1"]["phases"]
            big = report["arms"]["big_leaf"]["phases"]
            stock = report["arms"].get("stock", {}).get("phases", big)
            print("\n[gap] big_leaf - lw1 (us/step):", flush=True)
            for phase in PHASES:
                print(f"  {phase:14s} big-lw1={big[phase] - lw1[phase]:9.1f}  "
                      f"stock-lw1={stock[phase] - lw1[phase]:9.1f}  "
                      f"big/lw1={big[phase] / lw1[phase]:6.4f}", flush=True)
            report["gap_vs_lw1"] = {
                phase: {"big_minus_lw1_us": big[phase] - lw1[phase],
                        "stock_minus_lw1_us": stock[phase] - lw1[phase],
                        "big_over_lw1": big[phase] / lw1[phase]}
                for phase in PHASES
            }
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
            print(f"written {args.out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
