# TTorch 收束层缺陷：核实结论与可选解法（2026-10-21）

> 本文是 `TT_Convergence_Layer_Gap_Analysis.md`（14 条缺陷）的**核实与解法分析**。
> 定位：**只做分析，不含代码修改**。每条给三段：①本机核实结果（问题是否真实存在）
> ②参考框架的解法（具体到代码位置）③TT 可选方案（推荐 / 备选 / 代价 / 不做）。
>
> 核实环境：KL3（torch 2.9.0+cu129 / xpu-masquerade，4 卡）；对照源码见 `/root/dl/train-fmk`
> （Megatron-LM-FL / Torch-FL / TE-FL / verl-FL / vllm-plugin-FL）与本机 site-packages。

---

## 0. 核实结果总表

| # | 缺陷 | 核实结论 | 关键证据（本机） |
|---|---|---|---|
| D1 | 辅助进程重跑激活流水线 | **属实（已修）** | spawn 重跑 conformance；fork 继承。已按两档+分类修复 |
| D2 | 跨 rank 选择无一致性约束 | **属实** | rank0 签名 `cuda:0` / rank1 `cuda:1` → **缓存键天然不同** |
| D3 | 半激活态可达且进程继续 | **属实（已修）** | 第 3 条绑定失败 → `is_active=False` 但 6 符号已改写。已回滚修复 |
| D4 | 契约面无闭合判据 | **属实** | 142 公开符号 / 纳管 18 = **12.7%**；12 个高危符号**全部**未纳管 |
| D5 | 伪装信号外溢无治理 | **属实** | DeepSpeed `cuda_accelerator.py:204/215` 直接消费 `is_bf16_supported`/`get_device_capability`；TT 无 pin 机制 |
| D6 | 缺 torch 版本窗口 | **属实** | `TORCH_RANGE` 不存在；本机 torch=2.9.0+cu129 |
| D7 | 状态查询口径未验 | **属实** | `mem_query_realtime` 判据为量级（0.9×）；`mem_info_consistency` 容忍 1 GiB |
| D8 | 热路径税无基线 | **属实**（非性能问题） | 实测 handle→门面净增 **426 ns**（CPU 环境文献值 139 ns，量级一致） |
| D9 | 对账输入单点 | **属实** | `_admit_native_vs_orig` 调 `probe_input` 1 次；`_self_check` 固定 `randn(4, ...)` |
| D10 | 收束覆盖率不可见 | **属实** | 无收束画像接口；call_trace 只统计经门面的调用 |
| D11 | provenance 无 pid/rank | **属实（已修）** | 已加 pid/rank + fork_reset |
| D12 | detect 无异常隔离 | **属实** | `PlatformManager.auto_activate` 源码**无 try** |
| D13 | 缺 no-op 降级档 | **属实** | `PlatformBase.empty_cache` docstring 写"可 no-op"，实现却 `raise` |
| D14 | wrap_model 无并行包装检测 | **属实** | `wrap_model` 源码无 DataParallel/DistributedDataParallel 判据 |

---

## 1. D2 · 跨 rank 实现选择一致性【P0】

**核实**：`_call_sig` 把 `str(v.device)` 入签名 → rank0 得 `(2,256,'float32','cuda:0')`、rank1 得 `cuda:1`，缓存键天然不同，各 rank 独立跑门禁、独立决定 `chosen`。计时抖动 + 门禁异常静默回退进一步放大分叉概率。

**社区做法**：
| 框架 | 做法 |
|---|---|
| Megatron | 并行配置（TP/PP/EP）是**启动期静态参数**，各 rank 从同一份 args 派生，**不存在运行期分叉**；集合通信域在 `initialize_model_parallel` 里一次建好 |
| torchrun | 靠**全体一致性语义**：任一 rank 失败即作业失败（`ProcessFailure`），不追求"部分降级" |
| DeepSpeed | 配置驱动 + `barrier()` 对齐初始化；运行期不做 per-rank 自适应选择 |
| FlagOS / xExt | 静态 override 注册，**无运行期决策** |

即**没有任何参考框架在训练运行期做 per-rank 自适应算子选择**——TT 的三级门禁是独创能力，也因此独自承担了分叉风险。

**TT 可选方案**：

| 方案 | 做法 | 代价 | 评价 |
|---|---|---|---|
| **A（推荐，1 行）** | `_call_sig` 的 device 收敛为 `device.type`（丢卡序号） | 近乎零 | 消除根因，收益最大。卡序号对 kernel 选择无语义价值 |
| **B（推荐，~40 行）** | 首 step 后 `all_gather_object((semantic, selected))` 对账，**分歧则全体回退 native 并打印分歧表** | 一次集合通信；分歧时全体降到 native（一定正确） | fail-fast 而非协商：不变式 1 保证 native 必然正确，"某些 rank 用快实现"的收益远小于权重漂移风险 |
| C（备选） | rank0 决策后广播全体 | 引入广播依赖 + 时序耦合 | 文档已明确不建议（破坏 provider 单向只读 platform）；且广播失败时处置更复杂 |
| D（不做） | 跨 rank 协商/投票 | — | 过度设计 |

**建议组合**：A + B，且 B 的触发点放在"首个 step 边界"（此时各 rank 已完成首次 select），不要做成每次 select 都对账。

> **处置结论（2026-10-22）**：**只采纳 A，放弃 B 与 C、D**。
> - **A 已落地**：`_call_sig` device → `device.type`（`provider/manager.py`），
>   单测 `tests/unit/test_provider_sig.py` 7 例全过。
> - **B 经决策不采纳**：卡序号归一后同构集群各 rank 输入已一致，剩余分叉源仅
>   计时抖动（概率性）；为它引入首 step 集合通信 + 分歧处置协议，收益不成比例，
>   违背「避免过度设计」。残留风险与缓解因素（`min_speedup` 裕量 + D11 使分叉
>   可观测可归因）已记录于缺口分析文档 D2 节。若实测出现真实分叉再启用（成本
>   不变）。本文表格中 B/C 行保留，作为"已评估并放弃"的决策痕迹。

---

## 2. D4 · 契约面闭合性【P0】

**核实**：142/18 = 12.7%；`memory_stats`、`is_bf16_supported`、`get_device_capability`、`graphs`、`amp` 等 12 个高危符号全部静默穿透。

**社区做法**（"面怎么定义"的三种范式）：
| 框架 | 面 | 特点 |
|---|---|---|
| **PyTorch `torch.accelerator`** | 官方新抽象：`current_accelerator/is_available/device_count/empty_cache/max_memory_allocated...` | **极小且官方**——PyTorch 自己在收束硬件差异，是最值得对齐的方向 |
| DeepSpeed | `abstract_accelerator` **68 个方法** | 大而全，但每个后端（cuda/cpu/...）各自实现 |
| FlagOS | platform base ABC ~60 抽象方法 | 显式枚举 |
| symbrewrite | 按目标框架分插件 `SYMBOL_REPLACE` 枚举（148 条） | 框架视角切分 |
| DeepInfer | **4 属性 3 方法** | 极小，硬件差异点全部 `PLATFORM.` 前缀可 grep |

**关键洞察**：TT 的问题不在"18 太少"（DeepInfer 只有 7 个也够用），而在**没有"面是否闭合"的判据**，也没有"未纳管"的第二种处置（当前只有静默穿透）。

**TT 可选方案**：
1. **面盘点工具 + 覆盖率进回归基线**（~40 行，推荐先做）：输出"公开符号 × 是否纳管 × 是否被框架调用过"，与现有基线（demo 8 PASS / 等价性 B≡A / QWEN_SMOKE）并列；
2. **高危两类改为 EXCEPTION 占位**（~20 行）：能力查询类（`is_bf16_supported`/`get_device_capability`/`get_device_name`）与特性入口类（`graphs`/`amp`/`nvtx`/`is_current_stream_capturing`）——框架用到时**显式报错并指向开关**，强于静默返回伪装值；
3. **对齐 `torch.accelerator`**（观察项，不急）：PyTorch 官方抽象若成为框架的事实入口，TT 的契约面应向它收敛而非各自定义。

**不做**：把 142 符号全纳管（掉进"完整模拟 torch.cuda"的非目标）。

> **处置结论（2026-10-22，已落地）**：采纳方案 1+2 并强化——
> - 面盘点工具落地为 `bridge/surface.py`，且**不仅是盘点**：三态归属（Managed/Denied/
>   Exempt）+ `audit_surface()` 一致性断言（**声称 Managed == 实际注册 Entry**，漂移即
>   不闭合）——"证据先于声明"，判据受事实约束；
> - 高危两类落地为 **Denied 显式占位**（22 个特性点，报错指向功能点）；
> - 显存口径类**接管到门面**（17 个）+ D7 口径验收；
> - 实测 `closed=true, drift=null`、4 卡 DP 整网回归通过。
> 落地中实证修正的分类：值属性（`has_half`/`has_magma`）与子模块归 Exempt；
> `memory_usage`/`device_memory_used` 内部用 pynvml（非口径查询）归 Denied。

---

## 3. D5 · 伪装信号外溢治理【P1】

**核实**：DeepSpeed `cuda_accelerator.py:204/207/215` 直接消费 `torch.cuda.is_bf16_supported()` 与 `get_device_capability()` 做 dtype/kernel 决策——**下游消费真实存在**；TT 无任何 pin 或清单。

**社区做法**（本机就有的三个现成样本）：
| 来源 | 做法 |
|---|---|
| **torch_xmlir `_inductor/_triton_target.py`** | **直接在 triton 编译 target 层打 patch**（`_patch_triton_target` 等一系列），带 `_activated` 幂等标志 + `XMLIR_FORCE_XBLOCK` 显式开关 |
| torch_xmlir `xpu_fla` | patch triton 的 `jit/autotune/heuristics` 为 noop、`driver.active` 指向 mock(cuda) |
| DeepInfer 笔记 | `env pin + setdefault`，但**pin 后不验证** |

**TT 可选方案**（优先级递减，符合文档"① 官方显式配置 > ② 探测函数 pin > ③ 返回值修正"）：
1. **`PlatformSpec.env_pins` 机制化**（~30 行）：平台知识的一部分（而非散在 adapter 里的裸 `os.environ`），`activate()` 时 `setdefault` 注入并**逐条进 provenance**；
2. **pin 后验证**（TT 可超越 DeepInfer 的一步）：pin 完跑一次最小探测断言生效——下游库升级改 env 名时 pin 会静默失效，这是 DeepInfer 没做的；
3. **时机断言**：pin 必须早于下游库 import（TT 的 `.pth` 天然满足），但应显式检查 `sys.modules` 若已有 tilelang/triton 则告警；
4. **学习 torch_xmlir 的"代码级 patch target"**（备选）：对可控的下游库，直接 patch 其 target 决策比 env pin 更稳——但这是厂商栈的职责，TT 作为第二层**优先用 pin，不重复 patch**。

---

## 4. D6 · torch 版本窗口【P1】

**核实**：只有厂商 SDK 窗口（`sdk_range`），torch 自身无任何门禁；本机 torch=2.9.0+cu129。

**社区做法**：
- **symbrewrite**：按 transformers 版本开目录（`transformers_4_32_1` / `4_42_3` / `4_43_3` / `4_49_0` / `4_52_3`）——**版本即代码分支**；
- KLX/xExt：版本漂移**只告警不阻断**；
- FlagOS：签名兼容测试兜底。

**TT 可选方案**（~15 行）：`TORCH_RANGE = ("2.6", "2.13")`（按实测校准），激活期比对，**warn 不 block**（理由同 KLX：TT 有 admission/conformance 作为真闸门）。warn 文案要可执行——"torch X.Y 超出实测窗口，G12/G13/G15 类签名契约未在此版本验证，请跑 `check_bindings()` 确认"。形态与 `bind_targets.version_window` 同款（字符串，不建版本引擎）。

---

## 5. D7 · 状态查询口径等价【P1】

**核实**：`mem_query_realtime` 只验量级（≥0.9×），`mem_info_consistency` 容忍 1 GiB 误差——能挡量级错，挡不住口径差异。

**社区做法**：PyTorch 自身对 `memory_allocated` / `memory_reserved` / `max_memory_allocated` 有明确语义区分（是否含 cache、峰值窗口、reset 语义）；DeepSpeed 的 ZeRO 依赖 `memory_allocated` 做 offload 阈值决策。参考框架**没有**做"跨实现口径对账"（因为它们是单一实现）。

**TT 可选方案**（~25 行）：加一条 conformance——分配 `[16,32,64] MiB` 序列验增量单调与精确（允许 allocator 粒度向上取整，不允许量级偏差）；`del + empty_cache()` 后验回落基线（验 cache 是否计入）；把"口径画像"（含 cache？粒度？free/total 差额构成）写进 provenance 平台画像——**即使不强制 FAIL，也让口径差异可见**。

> **处置结论（2026-10-22，已落地，含两处修正）**：
> - `conformance.mem_caliber_coherence` 落地为**自洽性验收**（非准确性：masquerade 下
>   reserved/free/total 无外部标杆——实测分配 16MiB 致 free 降 148MiB，构成不可知）；
> - **修正 1（无标杆约束）**：准确性判定做不到，改为"allocated 增量对账（有标杆）+
>   不变量 + 回落"自洽性验收 + 口径画像（记录不判定）；
> - **修正 2（副作用归属）**：reset 语义验证涉及 `reset_peak_memory_stats()` 不可逆
>   副作用，从强制门禁挪到 `witness/diagnostics.mem_reset_semantics()` 按需诊断——
>   激活期验证不破坏被观测对象状态；
> - 口径画像进 `LAYER_PLATFORM` 且 `report()` 可见（实测 `reserved > allocated` 被记录）；
> - 实测：conformance 真实栈 + mock 双路径全过，4 卡 DP 各 rank 验收通过。

---

## 6. D13 · 接口三档【P2】

**核实**：`PlatformBase.empty_cache` docstring 写"尽力语义（可 no-op）"，实现却 `raise`——自相矛盾。

**社区做法**（直接先例）：
- **DeepSpeed `cpu_accelerator.empty_cache()` = `return`（no-op）**，而 `abstract_accelerator` 用 `@abc.abstractmethod` 约束必需能力——**abstract + no-op 双档**已是成熟实践；
- DeepInfer：abstract + no-op 双档；
- FlagOS：全 abstract（无 no-op 档）。

**TT 可选方案**（~20 行）：三档化——MANDATORY（`current_device/set_device/synchronize/is_available/device_count`）、OPTIONAL-RAISE（现状）、**OPTIONAL-NOOP（`empty_cache` 等，配 `_tt_noop_default` 哨兵）**。唯一要动的是 `supports()` 的派生逻辑需区分两档默认。MANDATORY 集合保持极小（abstract 化会让部分实现的 adapter 实例化即炸）。

---

## 7. D9 / D10 / D8 / D12 / D14 · 其余五条

| # | 社区做法 | TT 可选方案 |
|---|---|---|
| **D9** 对账输入单点 | KLX 教训原文指明**边界输入才是出错处**（MoE 空张量、`weights.float()` 升精度） | `probe_input` 增可选边界集（典型 + 空张量 + 单元素 + 敏感维度），遍历取最差值（~20 行）。成本仍是启动期几次小前向 |
| **D10** 覆盖率不可见 | DeepInfer：76 处硬件分支里 92% 是裸字段查询，结论是"**泄漏点集中且可 grep**"而非消灭泄漏 | 激活末尾产"收束画像"：①符号面覆盖率 ②在场不可收束组件（TE/apex/flash_attn/triton/tilelang 列单并注明不经 TT）③未纳管但被实际调用的符号（诊断模式采样，~40 行） |
| **D8** 热路径税无基线 | FlagOS：模块级 `cur_platform` 缓存 | 加两条 micro-bench 进回归基线（门面转发开销、`_call_sig`+dispatcher 占小 kernel 耗时比）。**现在不优化**（实测 0.014%/step） |
| **D12** detect 无隔离 | 各框架 detect 均有 try/except（本机 `get_device_name` 已包裹、`xpu-native` 已包裹，唯独 detect 调用点没有） | detect 调用包 try/except，异常记为 `(False, "detect 异常: ...")` 并照常打印（~5 行）——探测失败不该阻断其他平台，且异常信息本身是有价值的证据 |
| **D14** wrap 时机契约 | PyTorch DDP 建议在 `DistributedDataParallel` **构造前**完成模型改造；`torch.compile`/FSDP 同样要求包装前 | `wrap_model` 入口检测模型是否已被并行包装，拒绝并给出明确指引（~20 行） |

---

## 8. 建议落地序（不改本文档之外的任何代码）

**第一序（P0，堵静默错）**：D2-A（1 行）→ D2-B（40 行）→ D4-① 面盘点（40 行）→ D4-② 高危占位（20 行）
**第二序（P1）**：D5（30 行）→ D9（20 行）→ D14（20 行）→ D6（15 行）→ D7（25 行）→ D10（40 行）
**第三序（P2）**：D12（5 行）→ D13（20 行）→ D8（20 行）

**明确不做**（继承既有降级决策）：通用事务/回滚栈、ActivationPlan/CapabilityEvidence、跨 rank 广播协商、现在优化转发开销、完整模拟 torch.cuda。

---

## 9. TT 已领先、不要因"别人没做"而砍掉的三条

1. **三级门禁（功能/精度/收益）+ fp32 参考轨道**——symbrewrite 148 条替换**零等价性证明**，xExt 同样为零；
2. **kill switch + bypass 双层开关 + native 不可禁用**——DeepInfer"缺 kernel 即报错"，没有"降级到慢但正确"这一档；
3. **provenance 全量事件信封 + 绑定完整性检测**——FlagOS override 命中只有 `logger.info`。

这三点是 TT 相对四个参考框架的真实差异化，任何一条的削减都会让 TT 退化成"多一层包装的 symbrewrite"。
