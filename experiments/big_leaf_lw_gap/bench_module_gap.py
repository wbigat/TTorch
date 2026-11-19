"""big_leaf vs LW=1 模块级缺口基准（真实 P800 + 厂商 TE GroupedLinear）。

相对 RESULTS.md §9.3 旧基准的口径修正：
1. LW=1 pin 臂**每步显式失效** stateful cache（与生产 XME / TTorch optimizer-step
   post-hook 相位一致）；旧脚本缺这一步，存在 stale-cache 偏向 LW=1 的风险；
2. 新增 microbatch>1（梯度累积）与 activation recompute 形态，用于暴露「pin 缓存
   每步只 pack 一次 vs fresh pack 每次 forward 都 pack」的放大效应；
3. 新增 frozen（LoRA base 冻结）形态：LW=1 的 packed cache 对冻结权重永不失效；
4. **同一轮内交错计时**（A/B/C/... 轮转采样），抵消设备频率漂移——实测单臂连续
   计时的轮间噪声可达 10%，大于待分辨的 2-7% 效应。

用法：
    python experiments/big_leaf_lw_gap/bench_module_gap.py --iters 200 --out gap.json
    python experiments/big_leaf_lw_gap/bench_module_gap.py --mem --arm big_leaf \
        --shapes G8_2048_1536 --cases train_mb1
"""
import argparse
import json
import os
import statistics
import time

# 构造期决定 LW 形态，必须在建模前设好；运行期开关（pin）逐臂设置。
os.environ.setdefault("XTE_USE_GROUPED_FC_FUSION", "0")
os.environ.setdefault("XMLIR_ENABLE_FAST_FC", "1")

import torch
import torch.utils.checkpoint as ckpt
from transformer_engine.pytorch.env_config import xte_env
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear


SHAPES = {
    # 与 RESULTS.md §9.3 同形态，用于与历史记录交叉校验
    "G8_2048_1536": {"G": 8, "fin": 2048, "fout": 1536, "tokens": 8192, "bias": False},
    # Qwen3.5/3.6-35B-A3B：hidden 2048、moe_inter 512、256 专家 EP8 → 本地 32 专家；
    # linear_fc1 输出 = 2*512（gate+up 融合）
    "G32_2048_1024": {"G": 32, "fin": 2048, "fout": 1024, "tokens": 4096, "bias": False},
    # 每专家 1 token：device 计算量压到最小，用于隔离 forward 的 host 自开销
    "G32_host_only": {"G": 32, "fin": 2048, "fout": 1024, "tokens": 32, "bias": False},
    "G8_host_only": {"G": 8, "fin": 2048, "fout": 1536, "tokens": 8, "bias": False},
}

ARMS = ("stock_sep", "stock_contig", "lw1_pin", "lw1_nopin", "big_leaf",
        "big_leaf_nocheck")
# big_leaf_nocheck：仅用于归因——临时屏蔽 patch forward 每次调用的
# _check_binding_runtime（G 次 getattr + requires_grad/hook 巡检），衡量运行期
# 健康检查在 host-bound 形态下的占比。不是可交付配置。

CASES = {
    # name:            (mode,         microbatches, recompute, frozen)
    "train_mb1":       ("fwdbwdstep", 1,            False,     False),
    "train_mb4":       ("fwdbwdstep", 4,            False,     False),
    "train_mb1_rc":    ("fwdbwdstep", 1,            True,      False),
    "train_mb4_rc":    ("fwdbwdstep", 4,            True,      False),
    "frozen_fwdbwd":   ("fwdbwd",     1,            False,     True),
    "frozen_mb4_rc":   ("fwdbwd",     4,            True,      True),
    # forward-only + 冻结：配合 *_host_only 形状，读出每次 forward 的 host 自开销
    "host_fwd":        ("fwd",        1,            False,     True),
}


def _set(name, value):
    xte_env.set(name, value)
    os.environ[name] = value


def _pin_for(arm):
    """只有 LW=1 的 Parameter 会进 pin 路径；stock/big_leaf 入参都不是 Parameter。"""
    return arm == "lw1_pin"


def build_module(large_weight, shape):
    """构造一个 TE GroupedLinear；large_weight=True 即 LW=1 单 3D Parameter。"""
    _set("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1" if large_weight else "0")
    torch.manual_seed(1234)
    module = GroupedLinear(
        shape["G"], shape["fin"], shape["fout"],
        bias=shape["bias"], params_dtype=torch.bfloat16, device="cuda",
    )
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.02)
    _set("XTE_GROUPED_GEMM_LARGE_WEIGHT", "0")
    return module


def repack_contiguous(module):
    """route C：把 G 个专家权重搬进连续 buffer，仍用厂商 forward。"""
    G = module.num_gemms
    for kind in (("weight", "bias") if module.use_bias else ("weight",)):
        params = [getattr(module, f"{kind}{i}") for i in range(G)]
        buf = torch.empty(G, *params[0].shape, device=params[0].device,
                          dtype=params[0].dtype)
        with torch.no_grad():
            for i, p in enumerate(params):
                buf[i].copy_(p)
                p.data = buf[i]
        module.register_buffer(f"_routec_{kind}", buf, persistent=False)


class Arm:
    """一条被测形态：持有 module、optimizer、输入与一次迭代的闭包。"""

    def __init__(self, name, shape, case):
        mode, microbatches, recompute, frozen = case
        self.name = name
        self.pin = _pin_for(name)
        self.mode = mode
        self.microbatches = microbatches
        self.recompute = recompute
        self.plan = None

        self.skip_runtime_check = name == "big_leaf_nocheck"
        if name in ("stock_sep", "stock_contig", "big_leaf", "big_leaf_nocheck"):
            self.module = build_module(False, shape)
            if name == "stock_contig":
                repack_contiguous(self.module)
            if frozen:
                # 与真实 LoRA 流程一致：先冻结 base，再 apply（大叶子继承冻结语义）
                for p in self.module.parameters():
                    p.requires_grad_(False)
            if name.startswith("big_leaf"):
                from ttorch.bridge.big_leaf.core import apply_big_leaf
                container = torch.nn.Module()
                container.gl = self.module
                self.container = container
                self.plan = apply_big_leaf(container, forward_mode="patch")
        elif name in ("lw1_pin", "lw1_nopin"):
            self.module = build_module(True, shape)
            if frozen:
                for p in self.module.parameters():
                    p.requires_grad_(False)
        else:
            raise ValueError(name)

        if frozen:
            trainable = []
        elif self.plan is not None:
            trainable = self.plan.leafs
        else:
            trainable = [p for p in self.module.parameters() if p.requires_grad]
        self.trainable = trainable
        self.param_count = len(trainable)

        per_mb = shape["tokens"] // microbatches
        splits = [per_mb // shape["G"]] * shape["G"]
        splits[-1] += per_mb - sum(splits)
        self.m_splits = splits
        torch.manual_seed(99)
        self.x = torch.randn(per_mb, shape["fin"], device="cuda",
                             dtype=torch.bfloat16,
                             requires_grad=frozen or mode != "fwd")

        self.optimizer = None
        if mode == "fwdbwdstep" and trainable:
            from transformer_engine.pytorch.optimizers import FusedAdam
            self.optimizer = FusedAdam(trainable, lr=1e-4)

    def _call(self):
        if self.recompute:
            # 必须用 reentrant：Megatron 的 recompute 本身是 reentrant 实现，而
            # use_reentrant=False 会与 TE 自定义 Function 的两次 ctx.saved_tensors
            # 解包冲突（CheckpointError: already unpacked once）。
            return ckpt.checkpoint(self.module, self.x, self.m_splits,
                                   use_reentrant=True)
        return self.module(self.x, self.m_splits)

    def one_iter(self):
        _set("XTE_DISABLE_FAST_BF16_CACHE", "0" if self.pin else "1")
        if self.skip_runtime_check:
            from ttorch.bridge.big_leaf import core as bl_core
            original = bl_core._check_binding_runtime
            bl_core._check_binding_runtime = lambda binding: None
            try:
                return self._body()
            finally:
                bl_core._check_binding_runtime = original
        return self._body()

    def _body(self):
        for _ in range(self.microbatches):
            out = self._call()
            if isinstance(out, tuple):
                out = out[0]
            if self.mode != "fwd":
                out.float().square().mean().backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.pin:
                # 生产相位：optimizer.step 之后立即失效 stateful cache
                from hydrax.xaccelerator.stateful import \
                    reinitialize_all_after_param_update
                reinitialize_all_after_param_update(self.optimizer, (), {})


def measure_interleaved(arms, iters, warmup):
    """同轮交错采样：每次迭代按轮转顺序各测一次，抵消设备频率漂移。"""
    for _ in range(warmup):
        for arm in arms:
            arm.one_iter()
    torch.cuda.synchronize()
    samples = {arm.name: [] for arm in arms}
    for index in range(iters):
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        for arm in order:
            start = time.perf_counter()
            arm.one_iter()
            torch.cuda.synchronize()
            samples[arm.name].append((time.perf_counter() - start) * 1e6)
    out = {}
    for name, values in samples.items():
        ordered = sorted(values)
        out[name] = {
            "median_us": statistics.median(values),
            "p25_us": ordered[len(ordered) // 4],
            "p75_us": ordered[3 * len(ordered) // 4],
            "min_us": ordered[0],
            "samples": len(values),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--shapes", nargs="*", default=list(SHAPES))
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--mem", action="store_true",
                        help="单臂显存模式：只建一个臂并报告峰值显存")
    parser.add_argument("--arm", help="--mem 模式下的被测臂")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.mem:
        shape = SHAPES[args.shapes[0]]
        case = CASES[args.cases[0]]
        torch.cuda.reset_peak_memory_stats()
        arm = Arm(args.arm, shape, case)
        after_build = torch.cuda.memory_allocated() / 2 ** 20
        for _ in range(5):
            arm.one_iter()
        torch.cuda.synchronize()
        record = {
            "arm": args.arm, "shape": args.shapes[0], "case": args.cases[0],
            "resident_after_build_mib": after_build,
            "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            "param_count": arm.param_count,
        }
        print("MEM " + json.dumps(record), flush=True)
        return

    report = {"iters": args.iters, "warmup": args.warmup, "results": {}}
    for shape_name in args.shapes:
        for case_name in args.cases:
            key = f"{shape_name}/{case_name}"
            arms = [Arm(name, SHAPES[shape_name], CASES[case_name])
                    for name in args.arms]
            row = measure_interleaved(arms, args.iters, args.warmup)
            for arm in arms:
                row[arm.name]["param_count"] = arm.param_count
            report["results"][key] = row
            base = row.get("big_leaf", {}).get("median_us")
            for name in args.arms:
                stats = row[name]
                ratio = f"{stats['median_us'] / base:7.4f}x" if base else "   n/a "
                print(f"{key:30s} {name:14s} median={stats['median_us']:9.1f}us "
                      f"[p25={stats['p25_us']:8.1f} p75={stats['p75_us']:8.1f}] "
                      f"/big_leaf={ratio} params={stats['param_count']}", flush=True)
            for arm in arms:
                del arm
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
