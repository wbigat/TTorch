# big_leaf vs LW=1 性能缺口基准（研究用）

回答一个问题：**big_leaf 相对厂商 `large_weight`（LW=1）到底有哪些收益没拿到，
各值多少。** 结论与完整表格写在 `docs/big_leaf_方案设计与实现_20261109.md` §12；
本目录只放脚本与原始 JSON，按仓库约定不随功能 PR 合入。

## 两个基准

| 脚本 | 口径 | 回答的问题 |
|---|---|---|
| `bench_module_gap.py` | 单卡、真实 TE `GroupedLinear`、fwd+bwd+TE FusedAdam、同轮交错采样 | forward/backward/pack/pin 的模块级三方差；冻结（LoRA）形态反转；host 自开销归因；峰值显存 |
| `bench_distopt_gap.py` | 真实 `megatron.core` DDP + `get_megatron_optimizer`（dist-opt）、三臂交错、分相位计时 | 生产路径下「optimizer 单实体 + DDP 元数据」收益到底没拿到多少 |

五个臂：`stock_sep`（Megatron 默认独立存储）、`stock_contig`（route C 连续化但仍用
厂商 forward）、`lw1_pin`（LW=1 + 每步失效重 pack，生产 XME 相位）、`lw1_nopin`
（LW=1 + `XTE_DISABLE_FAST_BF16_CACHE=1`，big_leaf 入口强制的配置）、`big_leaf`，
外加归因臂 `big_leaf_nocheck`（临时屏蔽 patch forward 的运行期健康检查，仅用于
量化其占比，不是可交付配置）。

## 口径注意

- **必须交错采样**：单臂连续计时在本机的轮间噪声可达 10%，大于待分辨的 2–7% 效应；
- **pin 臂必须每步失效**：否则 stale cache 会让 LW=1 虚高，这是旧 `RESULTS.md §9.3`
  标注的已知限制；
- recompute 用 `use_reentrant=True`：Megatron 的 recompute 本身是 reentrant 实现，
  `use_reentrant=False` 会与 TE 自定义 Function 的两次 `ctx.saved_tensors` 解包冲突；
- 显存用单臂单进程 `--mem` 模式测：同进程多臂会累积分配，峰值不可比；
- `bench_distopt_gap.py` 当前是 DP-only、非 overlap、4 层代理堆叠，**不是整网
  `train_step` 净比值**。

## 复现

```bash
cd /root/dl/runtime/TTorch
PYTHONPATH=$PWD python3 experiments/big_leaf_lw_gap/bench_module_gap.py \
    --iters 120 --warmup 20 --out experiments/big_leaf_lw_gap/out/module_gap.json
PYTHONPATH=$PWD python3 experiments/big_leaf_lw_gap/bench_module_gap.py \
    --iters 200 --warmup 30 --shapes G8_host_only G32_host_only --cases host_fwd \
    --out experiments/big_leaf_lw_gap/out/host_only.json
for arm in stock_sep stock_contig lw1_pin lw1_nopin big_leaf; do
  PYTHONPATH=$PWD python3 experiments/big_leaf_lw_gap/bench_module_gap.py \
      --mem --arm $arm --shapes G32_2048_1024 --cases train_mb1
done
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
    experiments/big_leaf_lw_gap/bench_distopt_gap.py --iters 40 --warmup 8 \
    --layers 4 --experts 32 --fin 2048 --fout 1024 --tokens 4096 \
    --out experiments/big_leaf_lw_gap/out/distopt_dp4.json
```

环境：4×P800 OAM，镜像
`cu129-py3.12-torch2.9.0-mcore1.5.1-megatron0.18.0-swift4.3.1-init-p800:20261009`，
TTorch `bd347b2`。原始结果在 `out/`。
