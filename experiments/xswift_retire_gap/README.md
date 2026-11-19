# XSwift 退役：缺口探测 + 完整退役验证（研究用）

两件事：

1. **缺口探测**（`coexist_probe.py`）：以今天的代码与生产配置为准，XSwift 退役还差什么
   → `docs/xswift_退役能力分析与实施计划_20261029.md` **§15**；
2. **完整退役验证**（`validate_retirement.py`）：插件关闭 → 包物理移除的两级退役，
   整网功能 + 精度 + 性能 → 同文档 **§16**。

本目录只放脚本与原始 JSON，按仓库约定不随功能 PR 合入。

## validate_retirement.py

三状态同模型/同数据/同 seed 对比：`plugin_on`（生产基线）、`plugin_off`
（`XSWIFT_PLUGIN_ENABLE=0`）、`pkg_removed`（物理 rename 移走 `xswift_ext/` 与
`.pth`，`finally` 恢复 + 前后指纹校验）。

```bash
cd /root/dl/runtime/TTorch
# dense：真实 Qwen3.5-0.8B + LoRA（生产默认形态）
python3 experiments/xswift_retire_gap/validate_retirement.py --model dense \
    --states plugin_on plugin_off pkg_removed --steps 100 --eval-steps 50 --warmup 10 \
    --extra-env XTE_DISABLE_FAST_BF16_CACHE=1 HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT=1
# MoE：合成 BF16 Qwen3-MoE（本机唯一能走到 GroupedLinear / large-weight 的形态）
python3 experiments/xswift_retire_gap/validate_retirement.py --model moe --large-weight 0 \
    --states plugin_on plugin_off pkg_removed --steps 100 --eval-steps 50 --warmup 10 \
    --extra-env XTE_DISABLE_FAST_BF16_CACHE=1 HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT=1 \
    --model-path .benchmarks/xswift_retire_20261118/moe_assets/qwen3_moe_bf16 \
    --dataset .benchmarks/xswift_retire_20261118/moe_assets/text_sft.jsonl
```

### 必须知道的三个坑（都会让你拿到「看似通过」的无效证据）

1. **`~/.bashrc:129` 覆盖 `RUNTIME_SCRIPT_DIR`** → `run.sh` 的
   `source "${RUNTIME_SCRIPT_DIR}/env.sh"` 读的是**生产 env.sh**，外部传入的
   `XSWIFT_PLUGIN_ENABLE=0` 被吃掉，XSwift 全程在场。脚本副本已修正；验证器加了
   `banner == (state=="plugin_on")` 与 worker 探针双重硬断言；
2. **`~/.bashrc` 重新 export 全部 `HUNYUAN_*`** → 外部设 `HUNYUAN_CKPT_PATH` 无效，
   训练产物写进生产 ckpt 目录。脚本副本改用 `TTV_<VAR>` 在 bashrc 之后重新应用，
   验证器加了「exit=0 时 `run_dir/output` 必须存在」断言；
3. **`loss` 逐位一致 ≠ checkpoint 正确**。LW=1 下移除 XSwift 会在 exit=0、loss
   100% 逐位一致、零报错的情况下把 180 个 MoE 专家权重导出成 ~3.7e33。
   **必须比对导出张量**（本脚本做 sha256 逐位 + 数值幅度双检）。
   另注意 **LoRA 口径验不出来**——只导 adapter，base 权重不在导出集合里。

### 退役必需的三个环境开关（缺任一即静默损坏）

| 开关 | 缺失后果 |
|---|---|
| `XTE_GROUPED_GEMM_LARGE_WEIGHT=0` | 180 个 MoE 专家权重 → ~3.7e33 |
| `HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT=1` | `lm_head.weight` → 3.7e33 |
| `XTE_DISABLE_FAST_BF16_CACHE=1` | 全体张量污染（含 LoRA adapter） |

## probe_bootstrap/sitecustomize.py

进程级 XSwift 在场探针：每个 Python 进程启动时记录是否可 import xswift_ext、
`builtins.__import__` 是否被它劫持、`sys.modules` 里的 xswift 模块、
`/klx_dummy_lib` 假模块痕迹。**不只看 rank0 banner**，worker 级证据才算数。
链式加载原 sitecustomize，不顶掉环境已有的。写入失败会打到 stderr（不静默）。

## coexist_probe.py

探测 TT 的 bridge 功能补丁与 XSwift 的 `KlxSymbolRewriter.replace()` 在**同样三个
符号**上的共存行为（`wrap_model` / `gather_object` / `_apply_rotary_pos_emb_thd`）。
只读，不修任何源码。

```bash
cd /root/dl/runtime/TTorch
source /root/dl/runtime/script/env.sh
export PYTHONPATH=$PWD TT_PLUGIN_ENABLE=1 XSWIFT_PLUGIN_ENABLE=1 \
       XME_PLUGIN_ENABLE=0 TT_PROVIDER_AUTO_WRAP=0
python3 experiments/xswift_retire_gap/coexist_probe.py --order xswift_first \
    --out experiments/xswift_retire_gap/out/coexist_xswift_first.json
python3 experiments/xswift_retire_gap/coexist_probe.py --order tt_first \
    --out experiments/xswift_retire_gap/out/coexist_tt_first.json
```

两种顺序的含义：

| order | 含义 | 对应真实场景 |
|---|---|---|
| `xswift_first` | 先 import swift（XSwift 的 lazy `__import__` hook 接管）→ 再 `ttorch.activate()` | 调用方在 import swift 之后才激活 TT |
| `tt_first` | 先 `ttorch.activate()` → 再 import swift | **big_leaf 生产入口的真实顺序**（`big_leaf/swift.py:main()`） |

关键结果（2026-11-18，4×P800，**修复前**）：

| order | 最终实现 owner | TT `patch_status()` | `assert_bindings_intact()` |
|---|---|---|---|
| `xswift_first` | XSwift（gather 外套一层 TT） | wrap_model=`skipped`，其余 `applied` | tampered=**0** |
| `tt_first` | **XSwift（TT 补丁被整体覆盖）** | 三项全 **`applied`（谎报）** | tampered=**4** |

**修复后复测（2026-11-19，证据 `out/coexist_*_afterfix.json`）**：

| order | `patch_status()`（复核后） | `tt_overwritten_patches` | tampered |
|---|---|---|---|
| `xswift_first` | wrap_model/THD=`skipped`（归因改为 `vendor_plugin_present`）、gather=`applied` | `{}` | 0 |
| `tt_first` | wrap_model/gather=**`overwritten(xswift_ext...)`**、THD=`applied` | 2 条 | 4（fwpatch 两条与复核逐项一致） |

脚本同时输出 `tt_patch_status_install_time`（安装时结论，用于分辨覆盖发生的时点）
与 `retirement_report`（在场探针 + `/klx_dummy_lib` 痕迹 + 三开关导出安全核对，
来自包内 `ttorch.bridge.vendor_retire`）。机制说明见退役文档 §17 与
`TT_Bridge_Deep_Dive.md` §9.2/§9.6。

## TT 在场的整网 e2e（`--tt`，退役终局口径）

```bash
python3 validate_retirement.py --model dense --tt --large-weight 0 \
    --steps 100 --eval-steps 50 --warmup 10 --nproc 4 \
    --run-dir "$PWD/../../.benchmarks/xswift_retire_20261118/dense_tt" \
    --extra-env XTE_DISABLE_FAST_BF16_CACHE=1 HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT=1
```

`--tt` 经 `probe_bootstrap/sitecustomize.py` 在**训练 worker** 上走 TT 的生产
激活路径（`ttorch.utils._auto_init`，等价于装 `ttorch_auto_init.pth`；TT 当前
未装 .pth，sitecustomize 是唯一早于框架 import 的钩子），并在进程退出时追加
一条 `kind="tt_final"` 取证——补丁是否被厂商顶掉只有在框架 import 完成之后看
才算数。

**不带 `--tt` 的运行只能证明"包可以移走"，证明不了"退役后由 TT 承接"**
（§16 的那批运行就是这种）。结果见退役文档 §18：dense 372/372、MoE 231/231
逐位一致，退役态三条补丁全 `applied` 且无覆盖。

`--tt` 下验证器会额外做证据有效性断言（`tt_inconsistencies`）：worker 未真正
激活、缺 `tt_final`、退役态仍有补丁被覆盖、契约有未消除问题、发现
`/klx_dummy_lib` 残留——任一命中即 `RuntimeError` 终止，不允许产出看似通过的
无效证据。

## 口径注意

- `XSWIFT_PLUGIN_ENABLE=1` 是**生产现值**（`script/env.sh:33`），不是实验设定；
  退役分析此前的「两插件均关」前提已失效；
- XSwift 的 THD 落点写作 `mcore_bridge.patcher.rope_utils`，它**不是模块名**——
  是 `mcore_bridge.patcher` 里 import 进来的 megatron rope_utils 模块对象，
  只查 `sys.modules` 会误报 `module_not_imported`（脚本里 `_resolve_module` 处理）；
- B1 的落点是 mcore_bridge 的**闭包 cell**（`_origin_apply_rotary_pos_emb_thd`），
  不在模块属性上，需单独看 `_tt_cp_group_fixed` 标记；
- 本探测止于符号层，**没有**在 XSwift=1 下跑整网训练，也没有验证补丁被覆盖后的
  数值后果；
- `xswift_first` 下 THD 判为 `skipped` 不是缺口：XSwift 连 `mcore_bridge.patcher`
  一起换成了自己的拷贝（`__module__` 仍报 `mcore_bridge.patcher`，但
  `co_filename` 在 `xswift_ext/components/`），而**那份拷贝自带 cp_group 回退**
  （`components/mcore_bridge/patcher.py:110-115`，与 B1 修法一致）。
  插件关闭后复测 3/3 `applied`。
