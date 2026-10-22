# TTorch 作为收束层的设计缺陷分析（2026-10-21 更新版）

> 代码基线：`/Users/vector/code/train/TTorch` @ `73f98df`（dev，P0 uninstall 修复 + unit 单测层之后）
> 代码规模：`ttorch/` 5196 行 Python / 4 层（platform · bridge · provider · witness）
> 本机实测环境：CPU venv（torch 2.12.0，`~/.workbuddy/binaries/python/envs/default`）
> 参考系：torch-xmlir `symbol_rewrite`、KLX `xswift_ext`/`xmegatron_ext`、FlagOS、DeepInfer
>
> **本文的定位与既有 9 份文档的关系**：既有文档已沉淀了「已修复（A 类）」「刻意不做（B 类）」「待办（C 类）」三张清单。
> 本文**不重复**其中任何一条，只提出**尚未被任何文档覆盖的设计缺陷**，并对其中 1 条挂起决策（跨 rank 一致性）
> 给出「触发条件已满足」的新证据。全部结论标注证据强度：`[实测]` = 本轮本机复现；`[代码]` = 当前代码直读；
> `[他方]` = 参考框架素材，不作为 TT 自身实测结论。
>
> **配套文档**：各条的社区解法对照与可选方案见 `TT_Convergence_Fix_Options.md`。
>
> **修复状态（2026-10-22）**：
> - ✅ **D1 / D3 / D11 已修复并实测**（两档激活 + 进程分类 + 完整回滚含扫尾 +
>   provenance pid/rank + fork 事件流按进程重置；单测 14 例 + 4 卡 DP 回归通过）
> - ✅ **D2 已修复（采纳方案 A）**：`_call_sig` 的 device 收敛为 `device.type`，
>   卡序号不再入签名 → 同构集群各 rank 从同一缓存键出发，消除分叉的结构性
>   放大器。**方案 B（跨 rank 对账）经评估不采纳**：无实际收益预期且引入集合
>   通信依赖、违背"不过度设计"原则；残留风险（各 rank 仍 per-process 自治门禁，
>   计时抖动理论可致分叉）如实记录于代码注释与本节末尾，实测出现再议。
>   另修复**同源漏洞**：非张量对象参数 `repr(v)` 含内存地址会让签名跨 rank/跨
>   调用都不同（缓存永不命中）——现按类型名归并。
> - ✅ **D4 已修复并实测**：契约面闭合判据落地（`bridge/surface.py`：122 个公开
>   符号三态归属 + **Managed==实际注册一致性断言**）+ Denied 22 个特性点显式
>   占位 + Managed 接管扩至 58 个（`closed=true, drift=null`）。专家审视修复三个
>   真实缺陷：类型位 Denied 保持 type 身份（CUDAGraph 注解崩溃）、`symbol.__name__`
>   被 symbrewrite 改写失真（加 `Entry.attr_name`）、`is_bf16_supported` 签名超集。
> - ✅ **D7 已修复并实测**：`conformance.mem_caliber_coherence` 自洽性验收
>   （allocated 增量对账 + alloc≤reserved 不变量 + 释放回落）+ 口径画像进
>   provenance 与 report（可见）。reset 语义挪 `witness/diagnostics.mem_reset_semantics`
>   按需诊断——避免激活期 `reset_peak_memory_stats` 清峰值统计的不可逆副作用。
> - ⏳ D5 / D6 / D8 / D9 / D10 / D12 / D13 / D14 待决策（解法见配套文档）

---

## 0. 结论先行

TT 的四层架构（Platform/Bridge/Provider/Witness）与「实现即能力」「证据先于声明」「避免过度设计」三条文化**方向正确**，
本轮未发现需要推倒的结构性错误。但作为**上层训练框架对底层硬件的收束层**，它当前缺的不是机制精度，而是**四个维度的「边界定义」**：

| 维度 | 一句话问题 | 现状 | 最重缺陷 |
|---|---|---|---|
| **作用域边界** | 「哪些进程该被收束」没有定义 | `.pth` 对所有 Python 进程生效，只豁免 pip/conda | **D1**（spawn 子进程重跑整条激活流水线） |
| **一致性边界** | 「多 rank 必须选同一个实现」无约束 | provider 门禁是 per-process 自治实测 | **D2**（跨 rank 选择分叉，4 卡已跑通→触发条件已满足） |
| **契约面边界** | 「面有多大、覆盖多少」无量化、无闭合判据 | 145 个 torch.cuda 符号纳管 18 个，其余静默穿透 | **D4**（覆盖率 12.4%，含能力查询/显存口径两类高危） |
| **伪装外溢边界** | 「被伪装的信号流出 TT 之外」无治理 | 只治理「框架→torch.cuda」一条链 | **D5**（第三方库据伪装信号做不可逆编译决策） |

共 **14 条**缺陷，按类别与优先级：

| # | 缺陷 | 类别 | 证据 | 优先级 |
|---|---|---|---|---|
| D1 | 收束作用域未定义：辅助进程重跑激活流水线 | 作用域 | [实测] | **P0** |
| D2 | 跨 rank 实现选择无一致性约束 | 一致性 | [代码]+[实测] | **P0** |
| D3 | 半激活态可达且进程继续（.pth 吞异常 × 非原子绑定） | 作用域 | [实测] | **P0** |
| D4 | 契约面无闭合性判据，覆盖 12.4% 且静默穿透 | 契约面 | [实测] | **P0** |
| D5 | masquerade 伪装信号外溢，下游不可逆决策无 pin | 伪装 | [实测]+[他方] | **P1** |
| D6 | 缺 torch 自身版本窗口（只有厂商 SDK 窗口） | 契约面 | [代码] | **P1** |
| D7 | 状态查询类符号的「口径等价」未验 | 契约面 | [代码] | **P1** |
| D8 | 热路径税无量化基线（方法论缺口，非性能问题） | 可观测 | [实测] | P2 |
| D9 | native 复刻对账的输入覆盖面是单点 | 验证 | [代码] | **P1** |
| D10 | 收束覆盖率不可见（绕过路径无观测） | 可观测 | [代码] | **P1** |
| D11 | provenance 无 pid/rank 来源标识 | 可观测 | [实测] | P2（D2 前置） |
| D12 | platform detect 无异常隔离 | 健壮性 | [代码] | P2 |
| D13 | 接口两档不够，缺 no-op 降级档 | 契约面 | [代码]+[他方] | P2 |
| D14 | `provider_wrap` 时机契约未定义（与并行包装冲突） | 作用域 | [代码] | **P1** |

---

## 1. 方法与证据基线

本轮做了三组本机实测（脚本见附录 B），CPU 平台**仅作机制载体**——按项目既定基线，CPU/xpu-native 是预留平台、
其自身 bug 不计入架构缺陷，故下文只采信**与平台无关的机制行为**（进程模型、符号面大小、转发链开销、
provenance 信封结构），不把 CPU 特有的数值/能力问题当作缺陷。

| 实测项 | 关键数据 |
|---|---|
| 符号面 | `torch.cuda` 公开符号 **145** 个，policy 纳管 **18** 个（12.4%），`default_deny=False` |
| 未纳管但框架常用 | `memory_stats`/`memory_reserved`/`max_memory_allocated`/`reset_peak_memory_stats`/`is_initialized`/`is_bf16_supported`/`get_device_capability`/`get_device_name`/`is_current_stream_capturing`/`nvtx`/`amp`/`graphs` |
| 转发链开销 | 实现本体 16 ns → handle 转发 85 ns → 门面 155 ns（**净增 139 ns，9.97x**） |
| spawn 子进程 ×3 | 各自重跑 conformance **10 项**（合计 30 项次）+ 各自 lazy-import 厂商实现 |
| fork 子进程 | 继承绑定（`symbol_still_bound=True`）与 19 条 provenance，之后各自演化 |
| provenance 信封 | `[event, evidence, layer, outcome, seq, subject, ts]`——**无 pid/rank** |
| 半激活态 | platform 已激活 + 18 符号已绑定，而 `ttorch.is_active()` = **False** |

---

## 2. A 类：作用域与生命周期边界

> 共同根因：TT 把「激活」定义为**进程级事件**，但没有定义**哪些进程属于收束域**，也没有定义**激活动作与框架生命周期的时序契约**。

### D1 · 收束作用域未定义：辅助进程重跑整条激活流水线 【P0】

**现状** `[代码]`：`ttorch/utils/_auto_init.py` 的 `.pth` 钩子对**每个** Python 进程生效，豁免逻辑只有一条：

```python
def _is_tooling_process():
    """包管理/构建工具进程豁免"""
    return (argv0 in {"pip", "pip3", "setup.py", "conda"} or ...)
```

DataLoader worker、`torch.distributed` spawn 的子进程、tokenizer 多进程、监控进程的 `sys.argv[0]`
都是训练脚本本身 → **一律不豁免 → 一律走完整激活流水线**。

**实测** `[实测]`：spawn 子进程 ×3，每个都重跑 **10 项 conformance** + 重新 lazy-import 厂商实现：

```
父进程 pid=64032  conformance 跑了 10 项，provenance 19 条
spawn 子进程 ×3（每个都是全新解释器）:
  pid=64089  重跑 conformance 10 项，重新 lazy-import 实现 1 次
  pid=64090  重跑 conformance 10 项，重新 lazy-import 实现 1 次
  pid=64091  重跑 conformance 10 项，重新 lazy-import 实现 1 次
  → 激活流水线被重复执行 30 项次
```

**为什么是设计缺陷而非性能问题**：conformance 不是纯查询，它在真实卡上执行**真实设备操作**（`conformance.py` 代码直读）：

| 测试 | 真实副作用 |
|---|---|
| `mem_query_realtime` | 分配 **64 MiB** 真实显存 + `empty_cache()` |
| `sync_scope_device_wide` | `randn(256,256)` ×2 + 矩阵乘 + 建侧流 + device-wide sync |
| `stream_dynamic_coverage` / `stream_fencing_semantics` / `stream_ctx_restore` | 各建一条新 stream + 跨流拷贝 |
| `rng_state_roundtrip` | **改写全局 RNG state**（get→set→set 非法→set 恢复） |
| `graph_capture_roundtrip` | 尝试 CUDA graph 捕获 + replay |

`num_workers=8` 的 DataLoader 意味着 8 份上述操作。三类真实后果：

1. **资源竞争**：主进程已占满显存时，worker 的 64 MiB 探测分配可能 OOM → worker 崩 → 训练中断，且报错点在 conformance 里，归因困难；
2. **厂商 runtime 重复初始化**：`spec.load()` 会 import 厂商实现模块（KL3 上即 `torch_xmlir` 相关路径），每个 worker 一次 context 初始化；
3. **RNG 污染**：`rng_state_roundtrip` 改写全局 RNG state。虽然它最后恢复，但在 fork 语义下父子 RNG 分歧的时序更复杂。

**与既有决策的关系**：文档 C1 列过「import hook 的 spawn worker 重放 ⏸ 多卡前再管」，那是**绑定重放**问题；
本条是**激活流水线（含 conformance 真实设备操作）在非训练进程中执行**，是不同问题且不依赖多卡。

**参考框架对照**：DeepInfer 用 `functools.cache` 惰性单例 + `__init__.py` 全局编排保证「一次初始化」`[他方]`；
FlagOS 的 `cur_platform` 是模块级单例 `[他方]`。两者都只解决**同进程内**幂等，**都没有**多进程作用域概念——
这是 TT 可以领先的点，不是追赶。

**最小落地建议**（不建新子系统，约 30 行）：
把激活拆成**两档**，因为两档的成本结构完全不同：

```
轻档（绑定）  = 纯 setattr，廉价、无设备操作 → 所有进程都做（worker 也需要符号正确）
重档（门禁）  = conformance，真实设备操作 → 仅「训练主体进程」做
```

判据用**显式信号**而非启发式（符合项目「反魔法反的是隐式选择」文化）：
- 存在 `RANK`/`LOCAL_RANK` 环境变量且进程未被标记为 worker → 训练主体；
- `_auto_init` 激活后在环境里打一个标记（如 `TT_ACTIVATED_PID=<pid>`），子进程发现 `os.getpid() != 该值` 即判为派生进程 → 走轻档；
- 轻档/重档的选择逐条进 provenance（谁、为什么走了哪档）。

---

### D3 · 半激活态可达且进程继续 【P0】

**现状** `[代码]`：两个独立设计叠加出一个未被考虑的状态。

(a) `bridge/bootstrap.py` 的激活顺序 —— `_ACTIVATION` 在**全部绑定完成之后**才赋值：

```python
manager.auto_activate()                      # ① platform 激活（含 conformance）
cuda_compat.activate(entries, ...)           # ② 逐符号 setattr（18 次）
stale = cuda_compat.verify_sweep(...)         # ③ 扫尾验收
_ACTIVATION = BridgeActivation(...)           # ④ 此刻才算"已激活"
```

②在 `cuda_compat.activate` 内是**逐条 setattr 的循环**，没有事务性——中途任一条抛异常（符号不存在、
`importlib.import_module` 失败、厂商模块属性异常）→ **前面若干符号已改写，后面没改，`_ACTIVATION` 仍为 None**。

(b) `utils/_auto_init.py` 的异常处置 —— **捕获一切并让进程继续**：

```python
except Exception as e:  # noqa: BLE001
    print(f"[ttorch] 自动激活失败（进程继续，TT 未生效）...", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    return False      # ← 不 re-raise，进程继续
```

**实测** `[实测]`：构造「platform 已激活 + 符号已绑定 + `_ACTIVATION` 未赋值」状态，确认该状态可观测：

```
当前 bridge is_active() = False
但 torch.cuda.synchronize 已被改写 = True
```

**为什么是设计缺陷**：文档 B1 撤回「五阶段事务 + 回滚栈」的理由原文是——
> 「激活失败目前就是 fail fast 退出进程，没有『半激活后继续跑』的需求。回滚栈是为长驻服务设计的，TT 是训练进程」

这个前提**在 `.pth` 路径上不成立**：`.pth` 是生产推荐路径（文档明言「对 `megatron sft` 这类第三方 CLI 入口，
手动 import 无处安放；.pth 是 Python 官方提供的钩子」），而它恰恰**不退出进程**。
于是真实可达的状态是：torch.cuda 被改了一半、`ttorch.is_active()` 报 False、训练照常开始。
此时框架的部分调用经 TT 门面、部分直达厂商实现 —— 正是 `verify_sweep` 要防的「静默失效」形态，但 `verify_sweep` 此时根本没跑到。

**注意**：这**不是**要求恢复被撤回的五阶段事务。`_rebind` 收敛后，回滚能力**已经具备**
（`rebind` 有全局 `_SNAPSHOT`、`unbind` 在 P0 修复中已补齐），缺的只是**调用它**。

**最小落地建议**（约 15 行）：
1. `bootstrap.activate` 用 try/except 包住 ②③，失败时对 `_rebind.snapshot()` 里本次已写入的条目逐个 `unbind` 后 re-raise ——
   「要么全绑、要么全不绑」，不需要通用事务机制；
2. `_auto_init` 失败后写一个进程级中毒标记，并让 `ttorch.is_active()` 之外多一个 `assert_healthy()`：
   有标记时抛错并指明「TT 激活失败且符号可能部分改写，请勿在此进程训练」。

---

### D14 · `provider_wrap` 的时机契约未定义（与并行包装的时序冲突）【P1】

**现状** `[代码]`：契约只有 docstring 一句话：

```python
def provider_wrap(model):
    """... 在模型建好（from_pretrained / 搬到设备）之后调用一次 ..."""
```

而 `wrap_model` → `Provider.wrap_module` → `_self_check` 会对每个匹配模块做两件**侵入性**操作（`provider/spec.py:171-202`）：

```python
module.to(torch.float32)            # ① 改模块 dtype
ref = module.forward(x)             # ② 跑一次真实前向（x 形状取自 w.numel()）
...
finally:
    module.to(orig_dtype)           # 恢复
```

**为什么是设计缺陷**：「模型建好之后」没有排除**并行包装之后**，而两者冲突：

| 包装层 | 冲突点 |
|---|---|
| FSDP | 参数被 flatten 成 `FlatParameter`；`module.weight` 可能是 sharded view 或已被替换，`w.numel()` 不是完整维度；`module.to(float32)` 改动 flat param dtype 会破坏 FSDP 的 dtype 假设 |
| DeepSpeed ZeRO-3 | 参数被 partition，forward 外 `weight` 是占位（`ds_status` 为 NOT_AVAILABLE），`_self_check` 在 wrap 期直接读会拿到空 tensor |
| DDP | bucket 已按 param 顺序构建；替换 `module.forward` 不影响 bucket，但 `to(float32)` 往返会触发 param 的 dtype 变更 |
| activation checkpointing | 被包装的 module 的 forward 已被 checkpoint wrapper 接管，`module.forward(x)` 的语义不再是单次前向 |

当前这个坑**还没暴露**，原因是真实链路（megatron）走的是**函数级绑定通道**（`fn_binding`），
模块级 `wrap_model` 的实测只到纯 transformers 路径（`real_qwen_smoke.py`）`[代码]`。
一旦模块级通道接入 FSDP/ZeRO-3 的框架（文档 §7 规划的 megatron ParallelLinear 方向），这是必踩的。

**最小落地建议**（约 20 行）：
`wrap_model` 入口加**显式拒绝**（不是修复兼容，而是把隐式约束变成显式报错）：
检测 model 树上是否存在 FSDP/DDP/DeepSpeedEngine/CheckpointWrapper 类名，
或参数上是否带 `_is_sharded`/`ds_status`/`_flat_param` 标记 → 命中则抛错并指明
「`provider_wrap` 必须在并行包装之前调用」。docstring 的约束同步写进 README 的接入顺序说明。

---

## 3. B 类：分布式一致性边界

### D2 · 跨 rank 实现选择无一致性约束 【P0】

**现状** `[代码]`：provider 的三级门禁是**完全 per-process 自治**的实测，且第三级（收益）依赖**计时**：

```python
# provider/manager.py:_gate → witness.verify.verify_impl(..., min_speedup=..., sync=sync)
chosen = max(gated, key=lambda i: _KIND_PRIORITY.get(i.kind, 0)) if gated else native
self._binding[(semantic, slot, sig)] = (provider, chosen)
```

没有任何跨进程通信、广播或对账。每个 rank 独立：
① 跑自己的精度对账 → ② 跑自己的计时 → ③ 自己决定 `chosen`。

**为什么必然分叉**：
- **计时抖动**：`speedup` 是实测比值，受同机邻居进程、显存碎片、时钟频率、首次 kernel JIT 影响。`min_speedup` 阈值附近的 impl 在 rank 间**必然出现过/不过的分化**；
- **负载签名不同**：缓存键是 `_call_sig`（含 `shape[-1]`、dtype、**device**）。`str(v.device)` 在 rank0 是 `cuda:0`、rank1 是 `cuda:1` → **签名天然不同** → 各 rank 独立门禁，没有任何机制保证结论相同；
- **门禁基础设施失败的回退是静默的**：`call()` 里 gate 异常 → 该 rank 绑定 native 并记 `fallback`，其他 rank 可能正常选中 impl。

**后果分级**：

| 后果 | 机制 | 严重度 |
|---|---|---|
| DP 下各 rank 权重漂移 | 不同 kernel 的 reduction 顺序不同 → 本地梯度有 ulp 级差异 → all-reduce 后仍一致，但**本地参数更新路径**不同（优化器状态按本地梯度演化）→ 长训练后 rank 间权重缓慢分歧 | 静默（最危险） |
| 性能长尾 | 一个 rank 走 native、其余走融合 kernel → 该 rank 成为每步的 straggler，全局被它拖慢 | 可观测但难归因 |
| 潜在 hang | 若某 impl 的 `wraps` 链改变 autograd 图结构（`build="wrap_native"` 的实现包装了不同的原生算子链），backward 的执行序在 rank 间不同 | 显式但偶发 |

**与既有决策的关系 —— 这是本文唯一对挂起决策的挑战**：
文档 B3 把「跨 rank fingerprint 对账 / ActivationPlan 重放」挂起，触发条件写的是**「上多卡时」**：
> 「单进程单平台下，provenance 的 session 快照已经够复现。多 rank fingerprint 对账是上多卡时的真需求，**那时再做**」

而 `TT_Comm_Architecture.md` §7 的实测记录是：
> 「**4 卡 DP 跑通**，`megatron sft`，NNODES=1 / WORLD_GPUS=4；policy 18 条全通过」

**触发条件已满足**。而且此刻正是成本最低的时点：现在只有 4 个语义、门禁缓存结构简单；
等 provider 数量翻倍、上到 TP/PP/EP 混合并行后，追溯「哪个 rank 什么时候选了什么」的成本会高一个量级。

**最小落地建议**（不建 ActivationPlan/CapabilityEvidence 大机制，约 40 行）：
采用**「rank0 决策 + 全体对账」**而非「全体协商」：

1. **签名归一**：`_call_sig` 里把 `str(v.device)` 收敛为 `v.device.type`（丢掉卡序号）——
   卡序号对 kernel 选择没有语义价值，却让各 rank 的缓存键天然不同。这是 1 行改动，收益最大；
2. **门禁结论对账（fail-fast 而非协商）**：在首个训练 step 结束后（或 `select` 完成时）做一次
   `all_gather_object((semantic, selected_impl_name))`，不一致则**全体回退 native 并打印分歧表**。
   理由：回退 native 必然正确（不变式 1），而「让某些 rank 用快实现」的收益远小于权重漂移的风险；
3. **provenance 带 rank**（即 D11）：这是 ①②的前置——没有 rank 字段，分歧表打印不出谁是谁。

**明确不建议**：跨 rank 的门禁结果广播/同步选择（需要在 provider 层引入集合通信依赖，
违反「provider 单向只读 platform」的依赖方向），以及 ActivationPlan 重放机制（文档已正确降级）。

> **修复实录（2026-10-22，采纳第 ① 条 / 方案 A，放弃第 ② 条 / 方案 B）**：
> `_call_sig` 的 device 收敛为 `v.device.type`，卡序号不再入签名（`manager.py`）。
> 单测 7 例（`tests/unit/test_provider_sig.py`）+ 实测签名分量 `cuda` 无序号。
>
> **方案 B（首 step 后 all_gather 对账、分歧全体回退 native）经决策不采纳**：
> 卡序号归一后，同构集群各 rank 的**输入**已一致，剩下的分叉源只有计时抖动这一
> **概率性**因素；为它引入"每个作业首 step 后一次集合通信 + 分歧处置协议"的复杂度，
> 收益不成比例——这违背「避免过度设计」原则。**接受的残留风险**：各 rank 仍
> per-process 自治门禁，理论上计时抖动可致分叉；缓解因素有二——
> (1) `min_speedup` 阈值留有裕量，真正的临界 impl 极少；(2) 门禁基础设施失败回退
> native 已记 provenance（带 pid/rank，D11），分叉**可观测、可归因**，只是不阻断。
> 若未来实测出现真实分叉案例，再回来启用方案 B（成本不变，约 40 行）。

---

### D11 · provenance 无 pid/rank 来源标识 【P2，但是 D2 的前置】

**实测** `[实测]`：信封字段 = `[event, evidence, layer, outcome, seq, subject, ts]`，无 pid/rank。
fork 子进程继承 19 条父进程记录后各自演化，聚合时无法区分来源。

**现状** `[代码]`：`witness/provenance.py` 的 `record()` 写死 7 个字段；`export_jsonl` 要调用方手动调。

**后果**：
- 多 rank 训练产出 N 份 JSONL，内容格式相同、无来源字段 → 跨 rank 对账（D2）无从下手；
- fork 场景下父子记录混杂（`seq` 各自从继承值继续递增 → **seq 会重复**）；
- 文档把 provenance 列为 TT 对 FlagOS/xExt 的**首要差异化**（「provenance 全量审计」），
  但差异化在多进程下打了折扣。

**最小落地建议**（2 行）：`record()` 的信封增 `pid`（`os.getpid()`）与 `rank`（读 `RANK`/`LOCAL_RANK`，缺省 None）。
`export_jsonl` 的默认路径带 pid/rank 后缀，避免多 rank 互相覆盖。

---

## 4. C 类：契约面与伪装治理边界

### D4 · 契约面无闭合性判据：覆盖 12.4% 且静默穿透 【P0】

**实测** `[实测]`：

```
torch.cuda 公开符号总数 = 145
接管清单（18）= ['Event','Stream','can_device_access_peer','current_device','current_stream',
                 'device_count','empty_cache','get_rng_state','is_available','manual_seed',
                 'mem_get_info','memory_allocated','set_device','set_rng_state','set_stream',
                 'stream','synchronize','version']
存在于 torch.cuda 但未纳入 policy（静默穿透）=
  ['memory_stats','memory_reserved','max_memory_allocated','reset_peak_memory_stats',
   'is_initialized','is_bf16_supported','get_device_capability','get_device_name',
   'is_current_stream_capturing','nvtx','amp','graphs']
```

**现状** `[代码]`：`bootstrap.activate` 显式传 `default_deny=False`，注释理由是
「torch.cuda.`__all__` 含 torch 自身内部所需符号，白名单管的是框架调用面，不接管 torch 内部」。

**为什么是设计缺陷而非待办**：文档 C2 已承认「`default_deny=True` 尚未开启」并列了 2 个未纳管符号
（`is_initialized`、`is_current_stream_capturing`），但那是**逐个补符号**的视角。真正的缺陷在**覆盖面的定义方式**：

当前的面是 **trace-driven（追踪式）**——来源是「ms-swift 主链路实测用到的 20 符号 / 6 语义族」。
这意味着：
- **换框架就有新缺口**（LLaMA-Factory 的 `memory_stats` 显存日志、verl 的 `is_bf16_supported`、DeepSpeed 的 `get_device_capability`）；
- **换特性就有新缺口**（开 CUDA graph → `graphs`/`is_current_stream_capturing`；开 profiling → `nvtx`；开 autocast → `amp`）；
- **缺口形态是静默穿透**：在 masquerade 平台上，穿透的终点是厂商栈——可能对，也可能悄悄返回伪装值。

三类高危**不是等量的**：

| 类别 | 符号 | 穿透后果 | 为什么高危 |
|---|---|---|---|
| **能力查询类** | `is_bf16_supported`、`get_device_capability`、`get_device_name` | 返回伪装的 NVIDIA 架构信息（如 sm_80） | 框架据此**选 dtype / 开 flash-attn / 选 kernel 变体**，决策错误是静默的，且会向下传播到编译/算子选择 |
| **显存口径类** | `memory_stats`、`memory_reserved`、`max_memory_allocated`、`reset_peak_memory_stats` | 返回厂商 allocator 的口径 | 框架据此做 **OOM 预判 / auto batch size / ZeRO offload 阈值 / 显存报告**，偏差让上层策略失准（见 D7） |
| **特性入口类** | `graphs`、`amp`、`nvtx`、`is_current_stream_capturing` | 走到厂商栈的未验收实现 | 文档 §4.5 已明确「特性级符号不预先接管，建议以 default-deny 占位」——但**占位本身没做**，所以当前是静默穿透而非显式报错 |

**参考框架对照**：
- DeepInfer 的 Platform 只有 **4 属性 3 方法** `[他方]`——极小，但**显式且闭合**（硬件差异的 76 个消费点全部 `PLATFORM.` 前缀可 grep）；
- FlagOS 的 platform base ABC 有 **~60 抽象方法** `[他方]`——大而全，但也是显式枚举；
- TT 的 17 个 TORCH_API 介于两者之间，问题不在数量，而在**没有「面是否闭合」的判据**，也没有「未纳管=静默穿透」之外的第二种处置。

**最小落地建议**（不要求立刻开 `default_deny=True`，分两步）：
1. **先让缺口可见**（约 40 行脚本 + 进回归基线）：加一个面盘点工具，输出
   「torch.cuda 公开符号 × 是否纳管 × 是否被框架实际调用过（可用一次性 audit 采样）」的覆盖率报告。
   项目已有「demo 8 PASS / fn 三函数加权 +13.5%」的回归基线文化，覆盖率数字进同一张基线表；
2. **对两类高危先占位**：能力查询类与特性入口类按文档 §4.5 自己的建议落地为 **EXCEPTION 占位**
   （框架开启相关特性时显式报错并指向开关，强于静默跑错）；显存口径类走 D7 的口径对账。
   显存/能力类**不要**一上来就 remap——那会掉进「完整模拟 torch.cuda」的非目标（B4 第②条）。

> **修复实录（2026-10-22，已落地并实测）**：
> - **闭合判据** `bridge/surface.py`：122 个公开符号三态归属（Managed/Denied/Exempt）
>   + **Managed==实际注册一致性断言**（`audit_surface()`，漂移即 `closed=false`）——
>   证据先于声明，判据本身受事实约束（修复前曾漂移 9 个："声称纳管、实则穿透"）；
> - **Denied 22 个特性点**显式占位（cuda-graph / memory-pool / custom-allocator /
>   cuda-ipc / hw-monitor / cuda-blas），报错指向功能点（即上面建议 2 的 EXCEPTION 占位）；
> - **Managed 接管扩至 58 个**：能力查询 + rng 完整族 + 显存口径族 + stream/device 变体；
> - 实测：`closed=true, drift=null`、`check_bindings 58/58`、4 卡 DP 整网回归通过；
> - **专家审视修复的三个真实缺陷**：
>   1. 类型位 Denied 曾把 `CUDAGraph` 绑成函数 → transformers `-> CUDAGraph | None`
>      注解在 import 期求值全线 TypeError（训练崩）；修为占位类（保持 type 身份，
>      实例化才报错），并加 `_tt_denied` 标记供 conformance 区分"被占位 vs 平台不支持"；
>   2. `symbol.__name__` 被 symbrewrite 包装改写（`utilization`→`mock_torch_utilization`），
>      dotted/activate/verify_sweep 三处落点推导失效；给 `Entry` 加 `attr_name` 显式落点名；
>   3. `is_bf16_supported` 缺 `including_emulation` 参数 → torch 内部 `is_tf32_supported`
>      崩溃；修为签名超集（G12），并泛化核查全部 policy 符号的 torch 内部调用形态。
> - **遗留**：能力查询伪装值的 pin/留证治理属 D5；显存口径深度验收见 D7。

---

### D5 · masquerade 伪装信号外溢，下游不可逆决策无 pin 【P1】

**实测** `[实测]`：TT 激活前后 `torch.cuda.is_available()` 由 `False` 变 `True`（CPU 平台，`is_available` 在 policy 清单内）。
在真实 masquerade 平台上这个值本来就是 True（厂商栈设的）——**问题不在这个值本身，而在它的下游消费方**。

**现状** `[代码]`：TT 治理的链路只有一条：

```
训练框架 ──→ torch.cuda.*  ──(TT 门面)──→ platform adapter ──→ 厂商实现
                    ↑
                    └── 第三方库（triton / tilelang / flash-attn / bitsandbytes /
                        xformers / apex / DeepSpeed op_builder）与 torch 内部
                        （torch.accelerator / optim 的 _get_available_device_type）
                        也在读这个信号 —— TT 对这条链无任何机制
```

**污染链的一般形态**（DeepInfer 笔记 §8.1 的抽象，`[他方]` 实证）：

```
A（torch.cuda 门面，被有意伪装）→ B（下游库自动探测）→ C（不可逆决策）
```

DeepInfer 的实证案例原文注释：
> `transfer_to_npu` maps `torch.cuda.*` onto `torch.npu.*`, so `torch.cuda.is_available()`
> reports True on this NPU-only host. **TileLang's target auto-detection gates its CUDA probe
> on that call, and would compile every kernel without an explicit `target=` for CUDA.**

**关键在 C 不可逆**：编译 target、JIT 缓存 key、架构分发一旦定错，运行期纠正不了。
所以必须在 B 的入口截断，不能等 C 出错再修。

**TT 的现实暴露面**（KL3 栈实测事实，文档 E1 已记录）：该环境自带
`SYMBOL_REWRITE（torch/deepspeed/transformers）+ xpu_fla triton patch` —— 说明**下游库确实在被改写**，
而改写方是厂商栈，不是 TT。TT 作为「第二层」叠加其上，对第三方库的探测结果**既不感知也不治理**。
本机已踩过同类坑（CPU 平台上 `is_available=True` 导致 torch 2.12 的 Adam 经
`_get_available_device_type()` 误判 cuda）——那是 torch **内部**消费方，第三方库是同构问题的更大集合。

**为什么是设计缺陷**：TT 的定位是「收束层」，而收束的边界被隐含定义为「框架 → torch.cuda」。
但 masquerade 平台的本质是**信号伪装**，伪装信号不会停在 TT 的边界内——
凡读 `torch.cuda.*` 做决策的代码都是 TT 的事实影响范围，而 TT 只接管了其中一部分（18/145，见 D4），
对接管之外的消费方连**清单**都没有。

**最小落地建议**（DeepInfer 笔记 §8.1 已给出完整方案，TT 零落地，约 30 行）：
1. **机制化**：`PlatformSpec` 增 `env_pins: dict[str, str]` 字段（如 KL3 上 `{"TILELANG_DEFAULT_TARGET": ...}`），
   `activate()` 时统一 `os.environ.setdefault` 注入，**逐条进 provenance**（谁设的、key、value、为什么）——
   pin 表成为平台知识的一部分，而不是散在 adapter `__init__` 里的裸 `os.environ`；
2. **pin 后验证（TT 可超越 DeepInfer 的一步）**：DeepInfer pin 完不校验是否生效，
   下游库升级改 env 名时 pin 会**静默失效**。TT 的 conformance 文化可以做「pin 后跑一次最小探测并断言结果」；
3. **时机断言**：pin 必须早于下游库 import。TT 的 `.pth` 天然满足，但应显式断言
   （「若 tilelang/triton 已在 `sys.modules` 则 pin 失效，告警」）；
4. **决策优先级准则**（直接采用笔记结论）：
   **① B 的官方显式配置 > ② B 的探测函数 pin > ③ A 的返回值修正** —— 副作用面递增，③ 是最后手段
   且必须全量盘点 A 的消费方（即 D4 要建的那张面盘点表）。

---

### D6 · 缺 torch 自身版本窗口（只有厂商 SDK 窗口）【P1】

**现状** `[代码]`：版本门禁只有一维，且卡的是**厂商 SDK**：

```python
class TorchCudaPlatform(TorchDeviceModuleBase):
    sdk_range = ("11.0", "13.5")
    def sdk_version(self):
        return getattr(torch.version, "cuda", None) or "0.0"   # ← CUDA 版本，不是 torch 版本
```

文档 C2 的 P1-5 列的是**训练框架**版本窗口（swift 4.3.x / mcore_bridge 1.5.x）。
于是三维版本里，**torch 自身这一维完全没有门禁**：

| 维度 | 谁管 | 现状 |
|---|---|---|
| 厂商 SDK（CUDA/XPU 版本） | `PlatformSpec.sdk_range` | ✅ 已有，激活期 fail fast |
| 训练框架（swift/megatron） | — | ⏸ 文档 C2 P1-5 待办 |
| **torch 本身** | — | ❌ **无任何机制** |

**为什么这一维最重要**：TT 绑定的是 `torch.cuda` 的符号契约，而**契约的定义方就是 torch**。
已踩过的坑全部源于 torch 版本演进 `[代码]`（文档 A4 的 G 系列）：

| 坑 | 根因 | torch 版本相关性 |
|---|---|---|
| G12 | 门面签名必须是 torch 真实签名超集 | 签名由 torch 定义，随版本变 |
| G13 | `get_rng_state(None)` 在 **torch 2.9** 会炸（内部直接 `device.index`） | 明确标注了 torch 版本 |
| G15 | RNG state 必须是原生 ByteTensor 而非 bytes | torch `Generator.set_state()` 的类型要求 |
| 本机实测 | torch **2.12** 的 Adam 经 `_get_available_device_type()` 误判 | torch 内部实现变更 |

四个坑、四个 torch 版本相关——而 TT 对「本次运行的 torch 版本是否在实测验证过的窗口内」毫无判断。
`provenance.session()` 记了 `torch_version` 作为**事后证据**，但没有**事前门禁**。

**最小落地建议**（约 15 行，采用 KLX 的 warn-not-block 姿态）：
- `platform_base` 或 `bootstrap` 增 `TORCH_RANGE = ("2.6", "2.13")`（按实测校准），激活期比对；
- 超窗 **warn 不 block**（理由同 KLX「版本漂移只告警不阻断」，且 TT 有 admission/conformance 作为真正闸门）；
- warn 文案要可执行：「torch X.Y 超出实测窗口 [a,b)，TORCH_API 的签名契约（G12/G13/G15 类）未在此版本验证，
  请跑 `check_bindings()` 确认」；
- `TORCH_API` 旁注明「实测通过的 torch 版本」，与 `bind_targets.py` 的 `version_window` 同形态（字符串即可，不建版本引擎）。

---

### D7 · 状态查询类符号的「口径等价」未验 【P1】

**现状** `[代码]`：conformance 对显存查询的两条判据都是**量级/实时性**，不是**口径等价**：

```python
def mem_query_realtime(adapter):
    nbytes = 64 * 1024**2
    delta = after - before
    ok = delta >= nbytes * 0.9     # ← 只验"实时读到了、量级对"

def mem_info_consistency(adapter):
    ok = ... and abs(free + used - total) <= 1024**3   # ← 容忍 1 GiB
```

**为什么不够**：这两条能挡住「返回缓存值」「单位换算错 1e6 倍」这类粗错，
但挡不住**语义口径差异**——而框架消费这些数字做决策：

| 口径问题 | 框架消费点 | 失准后果 |
|---|---|---|
| `memory_allocated` 含不含 allocator cache / 碎片 | OOM 预判、显存日志 | 预判偏松 → 真 OOM；偏紧 → batch 开不满 |
| `mem_get_info().free` 是否扣除厂商 runtime 保留池 | auto batch size、ZeRO offload 阈值 | 策略按错误的可用量决策 |
| `max_memory_allocated` 的峰值统计窗口与 reset 语义 | 调优报告、容量规划 | 报告数字不可比 |

这是 G11（「能力类绑定的说谎半径」，讲 `is_available` 这类**布尔**的误导）的**数值版本**——
文档 A4 覆盖了布尔，没覆盖数值口径。而数值口径的错更隐蔽：布尔错会让守卫炸（G11 实测 AdamW graph capture 直接炸），
数值错**只让上层策略失准，不报任何错**。

**最小落地建议**（约 25 行，加一条 conformance）：
设计一个**已知总量的分配序列**对账，而不是单点阈值：
- 分配 `[16, 32, 64] MiB` 三次，每次读 `memory_allocated`，验**增量的单调性与精确性**（允许 allocator 粒度向上取整，但不允许量级偏差）；
- `del` + `empty_cache()` 后验**回落到基线附近**（验 cache 是否计入）；
- 把测得的「口径画像」（是否含 cache、粒度、free 与 total 的差额构成）写进 provenance 的平台画像字段——
  即使不强制 FAIL，也让口径差异从**不可见**变成**可查**。

> **修复实录（2026-10-22，已落地并实测）**：
> - `conformance.mem_caliber_coherence` **自洽性验收**（非准确性——masquerade 下
>   reserved/free/total 无外部标杆，实测分配 16MiB 导致 free 降 148MiB、构成不可知）：
>   allocated 增量对账（唯一有标杆）+ `alloc≤reserved` 不变量 + 释放回落；
> - **口径画像**进 provenance（新增 `LAYER_PLATFORM` 层）且 **report 可见**（实测 KL3：
>   `reserved=116M > allocated=112M`，allocator 粒度预留关系被记录）——达成"可查"；
> - **reset 语义**（第三口径）挪 `witness/diagnostics.mem_reset_semantics()` 按需诊断：
>   `reset_peak_memory_stats()` 清峰值统计**不可逆**，而 `activate()` 是公开 API、允许
>   训练中途调用——激活期强制门禁不应破坏被观测对象状态（专家审视后修正归属）；
> - 顺带强化：`conformance.run` 加异常托底（测试内 unsupported→skip、其他→FAIL，
>   不再崩栈）；显存口径族 17 个符号接管到门面（D4 配套）。
> - 实测：conformance 全过（真实栈 + mock 双路径），4 卡 DP 各 rank 验收通过。

---

### D13 · 接口两档不够，缺 no-op 降级档 【P2】

**现状** `[代码]`：`PlatformBase` 是**全 raise 档**——未实现即 `_unsupported()` 抛 `TTPlatformError("unsupported")`。
`platform_base.py` 自己留了备案注释：「若引入 `@abstractmethod` 需先理清两类接口边界」。

**实测侧证** `[实测]`：CPU 平台的 conformance 有 **5 项 skipped**（未实现 `Stream`/`memory_allocated`/`mem_get_info`），
说明「未实现」在真实降级平台上是**常态**而非异常——档位细分有实际价值。

**三档设计**（DeepInfer 笔记 §8.3 已给出，判别准则一句话：**调用方能否在不知道该能力是否存在时写出正确代码？**）：

| 档 | 表达 | 语义 | TT 的归属 | 调用方形态 |
|---|---|---|---|---|
| MANDATORY | `@abstractmethod` | 没有它进程无法工作 | `current_device`/`set_device`/`synchronize`/`is_available`/`device_count` | 无条件调用 |
| OPTIONAL-RAISE | 抛 `unsupported`（TT 现状） | 能力缺失，上层必须绕行 | stream 族、`can_device_access_peer`、`mem_get_info` | `if supports():` 或 try/except |
| **OPTIONAL-NOOP** | 默认 no-op / 等价弱实现 | 能力弱化但**功能正确**，只是慢 | **`empty_cache`** —— 自相矛盾的铁证见下 | **无脑调用，零分支** |

`empty_cache` 是档位缺失的直接证据（`platform_base.py:130-132`，docstring 与实现自相矛盾）：

```python
def empty_cache(self):
    """对齐 torch.cuda.empty_cache()：allocator 提示，尽力语义（可 no-op）。"""
    raise _unsupported(self, "empty_cache")   # ← 注释说"可 no-op"，实现却抛错
```

**为什么 no-op 档不可省**：把「不支持就什么都不做也正确」的接口放进 raise 档，
等于把分支成本推给每一个调用方。DeepInfer 的反面事实（76 处硬件分支里 70 处是裸字段查询，92%）正说明：
**能在基类默认实现里吸收的差异，就不要漏成调用方分支** `[他方]`。

**最小落地建议**（约 20 行）：
1. 给 `empty_cache` 这类加 no-op 默认实现 + 哨兵属性（如 `_tt_noop_default = True`）；
2. `supports()` 的派生逻辑区分「raise 档默认」与「no-op 档默认」——这是三档化**唯一需要动** `supports()` 的地方；
3. MANDATORY 集合**保持极小**（abstract 化的代价是部分实现的 adapter 实例化即炸），
   测试/预留平台给显式逃生门而非让测试类硬抄全部签名；
4. 配 mypy/pyright strict 进 CI，形成「静态检查 → 实例化 → conformance」三道门。
   注意 `@abstractmethod` 的真实失败点是**实例化期**（不是编译期），别让它产生「静态过了就安全」的错觉——
   abstract 保证接口存在，conformance 保证语义正确，两者不可替代。

---

## 5. D 类：可观测性与验证覆盖

### D9 · native 复刻对账的输入覆盖面是单点 【P1】

**现状** `[代码]`：本轮新增的两条对账都是**单点输入**：

```python
# fn_binding._admit_native_vs_orig —— 函数级接管前对账
args, kw = provider.probe_input(device=dev)     # 单次
ref = orig(*args, **kw); out = provider.native(*args, **kw)

# spec.Provider._self_check —— 模块级接管前自检
x = torch.randn(4, w.numel(), dtype=torch.float32, ...)   # 固定形状
```

**为什么不够**：项目自己吸收的 KLX 教训原文（文档 D2）就指明了**边界输入才是出错处**：
> `fused_bias_swiglu` 的两个正确性细节：**空张量特判**（「MoE 某 expert 分到 0 token 是常态」）+ `weights.float()` 升精度

当前对账若 probe 是典型 shape，则：空张量路径、单 token 路径、极端 dtype 路径的语义偏差**不会被发现**。
而 `_admit_native_vs_orig` 的价值主张正是「把『逐字一致』从声明变成每次启动都复验的证据」——
单点输入只能证明「在这一点上一致」，不能支撑「逐字一致」这个全域声明。

这条**不是**重复 A2-2（那条是「有没有对账」，已落地），而是「对账的**域**有多大」。

**最小落地建议**（约 20 行）：
`probe_input` 增一个可选的边界输入集（典型 + 空张量 + 单元素 + 该语义已知的敏感维度），
对账与自检遍历该集合、取最差值判定。成本是启动期多几次小前向（`_self_check` 当前成本注释为
「每模块至多两次小输入前向」，扩到 4-5 次仍在同一量级）。

---

### D10 · 收束覆盖率不可见（绕过路径无观测）【P1】

**现状** `[代码]`：两个观测机制都只看「经过 TT 的部分」：
- `witness/call_trace.py`：给**门面**装计数探针 → 只统计经 TT 中转的调用；
- `witness/provenance.py`：只记 TT 自己的动作（bind/gate/select/call/switch）。

于是无法回答一个基本问题：**这次训练有多少硬件交互绕过了 TT？** 绕过路径至少三类：

| 绕过路径 | 为什么 TT 看不见 |
|---|---|
| 127 个未纳管的 `torch.cuda.*` 符号（D4） | 没绑定就没探针 |
| C++ 扩展（TE / apex / flash-attn / DeepSpeed ops） | 文档 B4 已明确「没有可 remap 的调用点」——但**在场与否**也没检测 |
| codegen（inductor / triton / tilelang） | 同上，且叠加 D5 的伪装外溢 |

**为什么是设计缺陷而非职责边界问题**：文档 B4 的「C++ 扩展不是兼容问题，是替换或拒绝的二选一」
是**职责边界**结论（TT 不负责管）——这个结论正确。但「用户不知道有多少没被管」是**可观测性缺口**，是另一个问题。
TT 的核心价值主张原文是：
> 「TT 不保证『任何框架在任何硬件上都能跑』，**只保证『跑没跑对可被观测』**」

那么「哪些部分不在观测范围内」本身就必须可观测，否则这个主张是不完整的。

**参考框架对照**：DeepInfer 的结论 `[他方]`（笔记 §9-4）：
> 「76 处硬件分支里 70 处（92%）是裸字段查询，抽象面只有 6 处。这说明**抽象层设计得再窄，差异仍会泄漏**；
> 真正让系统可维护的是『泄漏点集中且可 grep』，而不是消灭它们。」

**最小落地建议**（约 40 行，激活末尾加一份摘要）：
产出「**收束画像**」并打印 + 进 provenance：
1. **符号面**：已纳管 / torch.cuda 公开总数（即 D4 的覆盖率数字）；
2. **在场的不可收束组件**：检测 `sys.modules` 里是否有
   `transformer_engine`/`apex`/`flash_attn`/`deepspeed.ops`/`triton`/`tilelang` → 列出清单并注明「这些组件的硬件交互不经 TT」；
3. **未纳管但被实际调用的符号**：可用一次性的 `torch.cuda.__getattr__` 代理或 `sys.addaudithook` 采样
   （只在诊断模式开启，避免热路径成本）。

---

### D8 · 热路径税无量化基线 【P2 · 方法论缺口，非性能问题】

**实测** `[实测]`（`is_available`，CPU adapter 实现了该方法，N=300000）：

```
1. base.is_available()             16 ns/call   (实现本体)
2. handle.is_available()           85 ns/call   (+PlatformHandle.__getattr__ 转发)
3. interface.is_available()       157 ns/call   (+manager.activation() 查找)
4. torch.cuda.is_available()      155 ns/call   (框架入口 = 门面)

收束层净开销 = 139 ns/call，倍率 9.97x
分解：__getattr__ 转发 +69 ns，activation() 查找 +72 ns（各占一半）
```

**诚实结论先行**：按 100 次调用/step 估算约 **14 us/step**，相对典型 100 ms/step 是 **0.014%**——
**当前绝对量级可忽略，这不是性能问题**。

**那缺陷在哪**：缺的是**基线本身**。项目文化是「证据先于声明」，而：
- `provider/manager.py` 的注释写的是推断：「签名计算是几个 tuple/str 操作，**相对 kernel 调用可忽略**」；
- `interface.py` 的三条硬约束（必须缓存/延迟求值/白名单）都是正确性约束，没有一条是性能约束；
- 转发链开销从未被测量过，也不在回归基线里（现有基线是「demo 8 PASS / 等价性 B≡A / QWEN_SMOKE resume diff=0.0 / fn 三函数加权 +13.5%」）。

两个真实需要警惕的方向：
1. **符号面扩大**（D4 的直接后果）：若把 `memory_stats` 这类高频显存查询纳管，调用频次是另一量级；
2. **provider 函数级通道进入小 kernel 高频路径**：`FunctionDispatcher` 每次调用构造 `_call_sig`
   （对每个 arg 做 isinstance + tuple 构造），对小 shape 算子，Python 开销与 kernel 耗时同量级。

**最小落地建议**（约 20 行）：加两条 micro-bench 进回归基线：
① 门面转发开销（本文数据可作首个基线值）；② `_call_sig` + dispatcher 分发开销 vs 小 shape kernel 耗时占比。
若将来某项超阈值，再考虑优化手段（如 FlagOS 式的模块级 `cur_platform` 缓存 `[他方]`）——**现在不要优化**。

---

## 6. E 类：健壮性与扩展性

### D12 · platform detect 无异常隔离 【P2】

**现状** `[代码]`：`manager.auto_activate()` 对 detect 的调用**无 try/except**：

```python
for spec in self.registry.auto_detectable():
    ok, evidence = spec.detect()          # ← 任一 detect 抛异常，整个激活崩
    print(f"[platform] detect {spec.name}: ...")
```

而 detect 的实现里有多个可能抛出的调用（`catalog.py`）：`torch.cuda.is_available()`
（半初始化的厂商栈上可能抛）、`importlib.util.find_spec()`（遇损坏/冲突的包元数据会抛 `ValueError`/`ImportError`）、
`torch.cuda.get_device_name(0)`（已有 try/except 包裹 ✅）、`m.is_available()`（xpu-native 已包裹 ✅）。

**后果**：一个无关平台的 detect 异常会让**整个自动激活崩溃**，且无法用 `select(name)` 绕过——
因为 `auto_activate()` 在 `select` 之前就崩了。这与项目「多命中 fail fast，可用 select 显式 override 排查」的
逃生通道设计意图冲突：逃生通道在这个失败模式下不可达。

**附带一个扩展性观察（非缺陷，spec.py 已声明为刻意取舍）**：
`masquerades_cuda: bool` + `manager.py:91-96` 硬编码字符串 `"cuda"` 的降级逻辑，
表达不了「伪装成 xpu/npu」或多目标伪装。spec 注释已写明「若未来出现伪装非 cuda / 多目标的栈再改为集合」——
记录在案，接第二个伪装平台时触发。

**最小落地建议**（约 5 行）：detect 调用包 try/except，异常记为 `(False, f"detect 异常: {type(e).__name__}: {e}")`
并照常打印——探测失败不该阻断其他平台的探测，且异常信息本身就是有价值的证据。

---

## 7. 与四个参考框架的对照矩阵

| 维度 | symbol_rewrite | xswift/xmegatron_ext | FlagOS | DeepInfer | **TT 现状** | 本文建议 |
|---|---|---|---|---|---|---|
| 多进程作用域 | 无概念 | 无概念 | 模块级单例（同进程） | `functools.cache` 单例（同进程） | **无概念，全进程激活** | **D1** 轻档/重档二分 |
| 跨 rank 一致性 | 无（无运行期决策） | 无 | 无（override 静态注册） | 无（构建期定死） | **无（per-process 门禁）** | **D2** 签名归一 + 对账回退 |
| 激活原子性 | 无 | 无 | 无 | 构建期无需 | **非原子 + .pth 吞异常** | **D3** 失败即 unbind |
| 符号面定义 | 148 条枚举 | `_RewriteSpec` 枚举 | ~60 抽象方法 | 4 属性 3 方法（极小显式） | **18/145 追踪式** | **D4** 覆盖率进基线 + 高危占位 |
| 伪装外溢治理 | 无 | 无 | 无 | **env pin + setdefault** | **无** | **D5** `env_pins` + pin 后验证 |
| 版本门禁 | 6 个版本目录 | 漂移告警 | 签名兼容测试兜底 | 构建期分叉 | 厂商 SDK ✅ / torch ❌ | **D6** `TORCH_RANGE` warn |
| 接口档位 | 无基类 | 无基类 | abstract（全 MANDATORY） | **abstract + no-op 双档** | 全 raise 档 | **D13** 三档 |
| 等价性验证 | **0 条证明** | **0 条证明** | 无 | 平台内自洽测试 | 三级门禁 ✅（TT 领先） | **D9** 扩到边界输入集 |
| 泄漏可见性 | — | — | 泄漏散落 | **`PLATFORM.` 统一前缀可 grep** | 只见已纳管 | **D10** 收束画像 |
| 运行期自愈 | 无 | 无（替换后无退路） | 无 | 无 | 两级回退 + kill switch ✅（TT 领先） | 保留，勿砍 |

**三条反向确认（TT 已领先，不要因为「别人没做」而砍掉）**：
1. **三级门禁（功能/精度/收益）+ fp32 参考轨道** —— symbrewrite 148 条替换零等价性证明，xExt 同样为零；
2. **kill switch + bypass 双层开关 + native 不可禁用** —— DeepInfer「缺 kernel 即报错」，没有「降级到慢但正确」这一档；
3. **provenance 全量事件信封 + 绑定完整性检测** —— FlagOS override 命中只有 `logger.info`。

---

## 8. 优先级与落地序

按「**边界先定义，机制后补全**」排序。每项都给了约 5–40 行的最小形态，总量可控，不引入新子系统
（符合项目「避免过度设计」文化与 B 节已有的降级决策）。

### 第一序：堵住静默错（P0，建议本轮做）

| # | 动作 | 规模 | 为什么最先 |
|---|---|---|---|
| D11 | provenance 信封加 `pid`/`rank` | ~2 行 | D2 的前置，零风险 |
| D2-① | `_call_sig` 的 device 收敛为 `device.type` | ~1 行 | 消除各 rank 缓存键天然不同的根因，收益最大的单行改动 |
| D3 | `bootstrap.activate` 失败时 unbind 已写入条目 | ~15 行 | `_rebind.unbind` 已具备能力，只缺调用 |
| D1 | 激活二分（轻档绑定 / 重档门禁）+ 派生进程判据 | ~30 行 | 真实资源竞争与 RNG 污染风险 |
| D4-① | 面盘点工具 + 覆盖率进回归基线 | ~40 行 | 先让缺口可见，不急着开 `default_deny` |

### 第二序：补齐契约与验证（P1）

| # | 动作 | 规模 |
|---|---|---|
| D2-② | 首 step 后 `all_gather_object` 对账，分歧则全体回退 native | ~40 行 |
| D5 | `PlatformSpec.env_pins` + setdefault 注入 + provenance + pin 后验证 | ~30 行 |
| D9 | `probe_input` 支持边界输入集，对账取最差值 | ~20 行 |
| D14 | `wrap_model` 入口拒绝并行包装后的模型 | ~20 行 |
| D6 | `TORCH_RANGE` 窗口 + warn-not-block | ~15 行 |
| D7 | conformance 增显存口径对账 + 口径画像进 provenance | ~25 行 |
| D10 | 激活末尾「收束画像」摘要 | ~40 行 |
| D4-② | 能力查询类 / 特性入口类按文档 §4.5 落地 EXCEPTION 占位 | ~20 行 |

### 第三序：质量与扩展（P2）

D12（detect 异常隔离 ~5 行）、D13（三档接口 ~20 行）、D8（两条 micro-bench 进基线 ~20 行）。

### 明确不建议做（继承既有降级决策，并新增两条）

| 不做 | 理由 |
|---|---|
| 通用五阶段事务 / 回滚栈 | 文档 B1 已降级且理由仍成立；D3 只需「失败即 unbind」，不需要通用机制 |
| ActivationPlan / CapabilityEvidence / Contract Pack | 文档 B1 已降级；D2 用「对账 + 回退」替代「计划 + 重放」，成本低一个量级 |
| **跨 rank 门禁结果广播/协商选择** | 会把集合通信依赖引入 provider 层，破坏「provider 单向只读 platform」的依赖方向。对账 + 全体回退 native 已足够安全（不变式 1 保证 native 必然正确） |
| **现在优化门面转发开销** | D8 实测 0.014%/step，无优化必要。先建基线，超阈值再动 |
| 完整模拟 torch.cuda（把 145 符号全纳管） | 文档 B4 非目标第②条。D4 要的是「面可量化 + 高危可见」，不是「面做大」 |

---

## 9. 一句话总结

TT 的四层机制已经相当完备，且在**等价性验证、运行期自愈、全量审计**三个维度明确领先四个参考框架；
本轮发现的 14 条缺陷**没有一条需要改变架构**，它们共同指向同一件尚未做的事——
**把「收束层的边界」从隐含假设变成显式定义**：哪些进程在域内（D1/D3/D14）、
多 rank 必须一致（D2/D11）、面有多大且未纳管的怎么处置（D4/D6/D7/D13）、
伪装信号流出去之后谁管（D5）、以及「不在收束范围内的部分」本身也要可观测（D8/D9/D10/D12）。

---

## 附录 A：本文引用的代码位置速查

| 结论 | 文件:关键位置 |
|---|---|
| D1 豁免逻辑 | `ttorch/utils/_auto_init.py:_is_tooling_process` |
| D1 conformance 真实副作用 | `ttorch/platform/conformance.py:mem_query_realtime`（64 MiB）、`rng_state_roundtrip` |
| D2 per-process 门禁 | `ttorch/provider/manager.py:_gate` / `select` / `_call_sig`（device 入签名） |
| D3 激活顺序 | `ttorch/bridge/bootstrap.py:activate`（`_ACTIVATION` 末尾赋值） |
| D3 吞异常 | `ttorch/utils/_auto_init.py:_init`（`except Exception` → `return False`） |
| D4 default_deny=False | `ttorch/bridge/bootstrap.py:activate` |
| D5 is_available 在 policy 内 | `ttorch/bridge/module_binding.py`（`register_binding(... is_available ...)`） |
| D6 SDK 窗口 | `ttorch/platform/adapters/cuda_adapter.py:sdk_range` / `sdk_version` |
| D7 显存判据 | `ttorch/platform/conformance.py:mem_query_realtime` / `mem_info_consistency` |
| D8 转发链 | `ttorch/platform/interface.py:_make_forward` → `platform_handle.py:__getattr__` |
| D9 单点对账 | `ttorch/provider/fn_binding.py:_admit_native_vs_orig`、`provider/spec.py:_self_check` |
| D10 观测范围 | `ttorch/witness/call_trace.py`（门面探针）、`witness/provenance.py` |
| D11 信封字段 | `ttorch/witness/provenance.py:record` |
| D12 detect 无隔离 | `ttorch/platform/manager.py:auto_activate` |
| D13 全 raise 档 | `ttorch/platform/platform_base.py:_unsupported` + `:72-77` 备案注释 |
| D14 自检侵入性 | `ttorch/provider/spec.py:_self_check`（`module.to(float32)` + `module.forward`） |

## 附录 B：实测脚本

三组脚本已验证并在本文引用其输出，核心片段如下（完整脚本为一次性诊断，未入库）：

```python
# ① 符号面盘点（D4/D5）
pm.select("cpu"); cuda_compat.activate(module_binding.entries(), default_deny=False, ...)
bound = sorted({e.symbol.__name__ for e in entries})          # 18
total = len([n for n in dir(torch.cuda) if not n.startswith("_")])   # 145
missing = [s for s in ["memory_stats", "memory_reserved", "max_memory_allocated",
                       "reset_peak_memory_stats", "is_initialized", "is_bf16_supported",
                       "get_device_capability", "get_device_name",
                       "is_current_stream_capturing", "nvtx", "amp", "graphs"]
           if s not in bound and hasattr(torch.cuda, s)]      # 12 个全部命中

# ② 转发链开销分解（D8）
bench(base.is_available)        # 16 ns  实现本体
bench(handle.is_available)      # 85 ns  +__getattr__
bench(interface.is_available)   # 157 ns +activation() 查找
bench(torch.cuda.is_available)  # 155 ns 框架入口

# ③ 多进程激活行为（D1/D11）
ctx = mp.get_context("spawn")   # 子进程各自: conformance 10 项 + lazy-import 1 次
ctx = mp.get_context("fork")    # 子进程: platform_activated=True, symbol_still_bound=True,
                                #         provenance=19 条（继承后各自演化，seq 会重复）
```
