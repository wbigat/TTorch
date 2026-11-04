# large-weight 存档（已从 TTorch 撤出，备查用）

> 撤出日期：2026-11-04（用户决策）
> 原位置：`ttorch/bridge/large_weight_shim.py`、`ttorch/bridge/patches.py` 的 B2/B3 注册、
> `tests/unit/test_large_weight_shim.py`
> 关联文档：`docs/TT_Bridge_Deep_Dive.md` §9.5、
> `../../docs/待办总览_持续推进清单_20261103.md` §B.1a

## 这里是什么

厂商 TE 在 `XTE_GROUPED_GEMM_LARGE_WEIGHT=1` 下把 G 份专家权重合并成单个 3D
`large_weight`，**不再注册 `weight{i}`**。而 `mcore_bridge` 的 HF↔Megatron 转换与
LoRA merge 全程按 `getattr(layer, f'weight{i}')` 寻址。本目录是为此写的 Python 层
适配 shim 及其回归。

| 文件 | 内容 |
|---|---|
| `large_weight_shim.py` | per-expert 视图 shim：`has_large_weight` / `num_experts_of` / `expose_per_expert` / `write_back` / `restore_layer` / `iter_expert_layers` |
| `patch_wrappers.py` | 两个补丁 wrapper：`set_mlp_state_wrapper`（B2）、`lora_merge_wrapper`（B3） |
| `test_large_weight_shim.py` | 17 例回归，用同构最小替身，**不依赖真实 TE / mcore_bridge** |

## 为什么撤出

**触发条件决定归属**：这些缺口**仅在主动开启 `XTE_GROUPED_GEMM_LARGE_WEIGHT=1`
时存在**。源码核实（2026-11-04）：

| 缺口 | LW=1 的失败点 | LW=0 时 |
|---|---|---|
| 权重转换 | `gpt_bridge.py` 12 处 `getattr(..., f'weight{i}')` → `AttributeError` | `weight{i}` 是真实 Parameter，正常 |
| LoRA merge | `tuners/lora.py:518` `base_layer.weight0.device` —— **一进门就 `AttributeError`** | 同上，正常 |
| 跨栈 ckpt | 盘上 key 集合 `large_weight` vs `weight{i}` 分叉 | 两侧同为 `weight{i}`，无分叉 |

于是它不是「硬件/框架缺口」，而是「**一个可选开关的代价**」：

1. 开关**默认 False**（`xmegatron_ext/envs.py:68`），厂商源码自标
   「large weight is temporary solution，need to be removed in the future」
   （`transformer_engine/pytorch/module/grouped_linear.py:786`）——
   给厂商自己准备废弃的临时特性在 TT 里建长期兼容层，方向不对；
2. shim 要替厂商维护「3D 合并参数 ↔ per-expert 视图」的映射，这是**厂商权重布局的
   实现细节**，不属于 TT bridge 的「功能接入」语义；
3. 继续这条路线还必须吃下 C1（优化器 `param_state` 裸索引重排，**高危且错位不可
   检测**），风险最大的一块反而由 TT 背。

## 怎么跑

**不参与 `ttorch` 包导入**，也不被 `register_all()` 注册 —— 本目录的代码在正常
训练中完全不会被执行。

```bash
# 从仓库根目录
python -m pytest experiments/large_weight/ -q     # 17 passed
```

**回归口径提醒**（避免数字漂移）：

| 命令 | 例数 | 含义 |
|---|---|---|
| `pytest tests/unit` | **88** | TT 的主回归口径 |
| `pytest`（根目录） | 105 | 会连带收集本目录的 17 例 |

仓库没有 `pytest.ini` / `testpaths` 配置，所以上面两个数都会出现。
引用「回归多少例」时**务必说明用的是哪个口径**。

## 留给独立方案的分析结论

即使 shim 路线不被采用，下面这些结论对任何方案都成立（**都有源码或实测支撑**）：

| 结论 | 内容 |
|---|---|
| **门控判据** | 用「`large_weight` 真实存在**且非空**」，**不读环境变量** —— 厂商在 `noop_cat` 零拷贝成立时也会把 `large_weight` 设成小权重的视图（`grouped_linear.py:1005`），此时布局其实是 per-expert。混用两种判据正是待办 C2「属性存在判据失真」的根因 |
| **专家数口径** | 优先 `layer.num_gemms`（层自己的事实），与 `large_weight.shape[0]` 冲突时**拒绝接管**；**不用** `config.num_moe_experts // ep_size` 推算 —— 那是待办 C8「专家数口径不一致」的根因，推算值与层实际切分不符时会静默产生错误的专家映射 |
| **切片方式** | 用直接索引 `large[i]`，**不用 `chunk`**（不整除时 chunk 静默错切） |
| **读写不对称** | XME 把 `TEGroupedLinear.__getattr__` 改写成合成 `weight{i}` 视图（`xmegatron_ext/plugin.py:113-117`）——**读路径够用，写路径不行**。`__getattr__` 只在属性查找失败时触发，这个不对称是它的固有性质 |
| **merge/unmerge 的差别**（口径更正） | **只有 `merge` 的 safe_merge 分支（`lora.py:540`）是重新绑定**（`weight.data = orig_weights[i]`）；`:546`（非 safe_merge）与 `:577`（unmerge）都是**原地运算**（`+=` / `-=`），真 view 可透写。此前「merge/unmerge 都静默丢写」的表述不准确 |
| **缺陷的两层性** | 裸环境下 `lora.py:518` 先抛 `AttributeError`，**根本走不到 `:540`**。丢写是厂商 `__getattr__` shim 把 `weight0` 变可读之后才浮现的**第二层**问题 |

## 新方案的关键输入（尚缺）

`LW=1` vs `LW=0 + XTE_USE_GROUPED_FC_FUSION=1` 的 **XPU 性能对比**
（LW=1 时厂商会强制把 `GROUPED_FC_FUSION` 置 0）。
若差距可接受，「关开关」就是最简方案 —— B2/B3/B4 与待办 C 组整组的触发面同时消失。
口径：边际 s/it（elapsed 差分），对齐 trainable params / batch / cutoff_len。

## 未解决的风险（如实登记）

生产脚本 `runtime/script/env.sh:31` 当前是**显式** `XTE_GROUPED_GEMM_LARGE_WEIGHT=1`
（注释已记录症状「与 mcore_bridge 权重命名曾出现不兼容，no attribute error」）。
撤出后，LW=1 下的 `AttributeError` 会**回归** —— 此前靠 XSwift 插件挡着，插件退役后
靠这两条补丁挡。按 2026-11-04 决策，**TT 侧不加告警也不接管**，生产侧风险由使用方把控。
