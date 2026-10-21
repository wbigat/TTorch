# TTorch 替换机制设计：现状、缺口与按需演进

> 刷新于 2026-10-20，基线 commit `bd1b90f`（新增 `provider/fn_binding.py` 函数级绑定通道）。
> 本机实测环境：torch 2.12.0 / transformers 5.19.0（CPU 隔离 venv）。
>
> **本版相对 2026-10-15 版的三处改动**：
> 1. 代码已落地 import hook（`_PostImportFinder`），与上版"import hook 暂不进 P0"的结论不同——
>    经复核**代码是对的，上版判据过粗**，§3 重写判据；
> 2. 上版提出的 `RewriteSpec` 十字段 DSL、五阶段事务、`ActivationPlan`、`CapabilityEvidence`
>    对当前体量属**过度设计**，本版降级或撤回，见 §6；
> 3. 新增本轮实测证据（静默偏差 1.337、完整性检测覆盖 15/19、`_origin_` 前缀撞名），见 §4。

---

## 0. 结论先行

当前替换机制**已经够用**，不需要新建"Rewrite Engine"子系统。真实缺口是三条**小而具体**的问题，
都是在 provider 从 1 个长到 4 个、绑定通道从 1 条长到 2 条时**自然暴露**的，不是设计预判：

| # | 缺口 | 实测证据 | 修复体量 |
|---|---|---|---|
| **A** | 替换动作有 3 份独立实现，`_origin_` 前缀撞名，完整性检测只覆盖其中 1 份 | `assert_bindings_intact()` 返回 15 条，provider 函数级绑定 **0 条**被覆盖 | 抽 ~40 行公共函数 |
| **B** | 函数级绑定**没有**把"被替换的原实现"纳入对账，`orig` 已在手上却只用于 bypass | 故意改框架原实现后接管，输出差 **1.337**，provenance 三条事件全绿、零告警 | ~15 行 |
| **C** | 绑定坐标散在各语义包、无版本维度、失效静默 | 3 个 `bind_points` 分散在 3 个文件；`bias_swiglu_impl` 绑定 `skipped` 只进 JSONL，终端无提示 | 一张表 + 一行打印 |

**一句话**：机制不缺，缺的是"同一个动作只写一遍"和"替换前后要对账"。这两件事加起来不到 100 行。

---

## 1. 当前替换机制的真实形态（bd1b90f）

TT 现在有 **2 条绑定通道 + 1 种实例级接管 + 1 个观测探针**，共 4 个 patch writer：

| # | writer | 代码位置 | 落点 | 时机 | 目的 |
|---|---|---|---|---|---|
| 1 | `bridge/cuda_compat.activate` | `cuda_compat.py:71-84` | `torch.cuda.X`（re-export 点）+ 白名单包 sweep | 激活期一次性 | 功能正确性 |
| 2 | `provider/fn_binding._bind_one` | `fn_binding.py:128-144` | 框架内部函数定义点（`megatron...swiglu`）+ 白名单包 sweep | **import 钩子后置** | 性能增量 |
| 3 | `provider/manager.wrap_model` | `manager.py:175` | `module.forward`（实例属性） | 模型构建后显式调用 | 性能增量 |
| 4 | `witness/call_trace` | `call_trace.py:61` | `torch.cuda.X`（覆盖门面） | 按需开启 | 可观测 |

这个分工本身是对的——**落点不同、时机不同、owner 不同**，不该合并成一条通道。
问题出在"怎么写"这一层（§4-A）。

### 1.1 通道 2 的关键设计（本轮新增，值得保留）

```python
# fn_binding.py:51-73
class FunctionDispatcher:
    def __call__(self, *args, **kw):
        m = _pm.instance()
        if m.is_bypassed():
            return self._orig(*args, **kw)
        return m.call(self._semantic, *args, **kw)
```

**绑定目标不是厂商 kernel，而是 dispatcher**。这一步做得很好：函数级接入因此自动继承了
模块级已有的三级门禁 / 负载签名缓存 / 两级回退 / kill switch / bypass / provenance，
而不是另起一套。新增一个函数级 provider 的成本 = 写 `bind_points` 一行。

同样值得保留的是 `FunctionDispatcher` 必须是**模块级类**（而非闭包）的决策及其注释——
megatron 的 MLP 会把激活函数存进实例属性，闭包会让 `torch.save(整模型)` 炸 `PicklingError`。
这条和 `ProviderModuleForward` 是同一个坑的两个现场，注释里都留了证据。

---

## 2. symbrewrite 对照：哪些已吸收，哪些还没

| 机制 | symbrewrite | TT 现状 | 判定 |
|---|---|---|---|
| 工厂延迟构造 | `is_wrapper=True` → enable 期调工厂 | `Entry.impl_factory` ✅ | **已吸收**（并修掉了 `torch.cuda.Stream` 被绑成函数的阻断项） |
| 定义点直打 | 字符串 FQN → `setattr` | `PerfBinding(module, attr)` ✅ | **已吸收** |
| 原实现留痕 | `__origin_<name>` | `_origin_<name>` ✅ | 已吸收（但撞名，见 §4-A） |
| from-import 扫尾 | 不做 | `_rebind_and_sweep` + `verify_sweep` | **TT 更强**（保留） |
| 等价性门禁 | 无 | 三级门禁 + fp32 参考轨道 | **TT 更强**（但函数级有漏，见 §4-B） |
| 运行时回退 | 无 | 两级回退 + kill switch + bypass | **TT 更强**（保留） |
| 按"目标框架"组织 | `plugins/<框架>/` | 散在 `impls/<语义>/` ❌ | **未吸收**（见 §4-C） |
| 版本维度 | 目录即版本（`transformers_4_32_1`…） | 无 ❌ | **未吸收**（见 §4-C） |
| 插件清单与开关 | `plugins.xflags` | 无 ❌ | 未吸收（按需，§6 降级） |
| caller→impl 源位置日志 | `LOG_SYMBOL_REPLACE=1` | 只统计次数 ❌ | 未吸收（按需，§6 降级） |
| 整包替换 | `MODULE_REPLACE` | 无 ❌ | 未吸收（当前无需求，§6 撤回） |

---

## 3. 重写判据：什么时候该用 import hook

上版结论是"主动拉起更确定，import hook 暂不进 P0"。代码选择了 import hook，**复核后代码是对的**，
上版判据漏了"主动拉起的代价"这一维。修正后的判据：

```
目标模块能否被 TT 在激活期安全且廉价地主动 import？
  ├─ 能  → 主动拉起 + 定义点直打（确定性更高，激活返回即状态确定）
  └─ 不能 → import 钩子后置绑定（唯一可行，须满足下方硬要求）
```

"不能"的三种典型情形，megatron **三条全中**：

| 情形 | megatron 的具体表现 |
|---|---|
| import 代价高 | `import megatron.core` 牵出大量子模块与 CUDA 扩展检查 |
| import 有副作用 | 部分路径依赖分布式环境变量/初始化状态 |
| 会制造反向依赖 | TT 的 `.pth` 在**任何框架 import 之前**激活；此时主动 import megatron 等于让 TT 强依赖一个训练框架 |

对比：transformers 的 `modeling_*.py` 主动 import 是廉价的（纯 Python 定义），
所以**模型层走主动拉起、训练框架内部走 import hook** 是正确的分工，不是二选一。

### 3.1 用 import hook 的硬要求（现状对照）

| 要求 | 现状 |
|---|---|
| 走标准 `MetaPathFinder`，不 patch 私有 `_handle_fromlist` | ✅ `_PostImportFinder` |
| `exec_module()` 完成后才绑，防 partially initialized | ✅ `fn_binding.py:179-181` |
| per-module 幂等 | ✅ `register_binding` 按 dotted 查重 + `_INSTALLED` 闸 |
| 已在场模块立即绑 + 扫尾兜底 | ✅ `install()` 末尾 |
| import 失败不吞异常 | ✅ `orig_exec` 异常直接向上抛 |
| 提供 uninstall，验证 finder 未泄漏 | ❌ **缺**（单测/多次 activate 场景会累积 finder） |
| spawn worker 重放 | ⏸ 多卡前再管（§6） |

补 uninstall 是 ~10 行，建议随手做掉；其余不动。

---

## 4. 三个真实缺口（本轮实测）

### 4-A 同一个动作写了三遍，检测只认一遍

> **状态（2026-10-21）：已落地**——替换动作收进 `ttorch/utils/_rebind.py`，
> 留痕改带 owner（`_tt_origin_<owner>_<attr>`）；检测基准统一后覆盖 21 条
> （bridge 18 + provider 函数级 3，此前 provider 0 条）。

`cuda_compat.py:71-88` 与 `fn_binding.py:128-147` 的 sweep 循环**近乎逐行重复**：
同样的"遍历 `sys.modules` → 白名单前缀过滤 → `id()` 比对 → 双 `setattr` 留痕"。
差异只有两处：白名单来源（bridge 是参数、fn_binding **硬编码** `("megatron","mcore_bridge","swift")`）、
provenance 的 layer 标签。

实测后果：

```
bridge   写入 torch.cuda 的 _origin_* : 15 个
provider 写入 megatron 模块的 _origin_*: _origin_swiglu（同前缀）
assert_bindings_intact() 覆盖条目      : 15 条
  其中 provider 函数级绑定             : 0 条   ← 完全不在检测范围
```

两个后果：

1. **完整性检测有盲区**。`_POST_BIND` 快照只在 `cuda_compat.activate` 里写，
   provider 的函数级绑定被第三方（symbrewrite / Liger / 框架 kernelize）覆盖掉，TT 无从察觉。
2. **`_origin_` 撞名**。两个 writer 用同一前缀，且 symbrewrite 用 `__origin_`。
   一旦出现同一符号被两方先后绑定，后写者会把前写者的门面存成"原实现"——回滚就回滚错了。
   当前没有符号重叠，所以是**潜在雷不是现炸**，但也没有任何机制保证不重叠。

**按需的最小修法**（不是建 Engine）：抽一个 `ttorch/utils/_rebind.py`，只提供两个函数：

```python
def rebind(module, attr, new, *, owner, sweep_packages):
    """源头 setattr + 扫尾 + 留痕（带 owner），返回 (n_swept, snapshot_key)。"""

def snapshot():
    """全部 writer 的 {(mod, attr): (obj, owner)}——完整性检测的唯一基准。"""
```

- 留痕改 `_tt_origin_<owner>_<attr>`，消灭撞名，同时让"谁绑的"从属性名即可见；
- `cuda_compat` 与 `fn_binding` 都改调它，删掉各自的 sweep 循环；
- `witness.assert_bindings_intact()` 改读 `_rebind.snapshot()`，一次覆盖全部 writer；
- 冲突检测顺带就有了：`rebind` 发现目标已有他人 owner 留痕 → 记 provenance 并按策略拒绝。

体量约 40 行 + 两处改调用。**不要**在这一步引入 `RewriteSpec`、事务、回滚栈——
当前只有 2 个 writer、4 种落点，DSL 的维护成本高于它解决的问题。

### 4-B 函数级绑定没有对账"被替换的原实现"

> **状态（2026-10-21）：已落地**——`fn_binding._admit_native_vs_orig()` 在绑定
> 时对账 native vs orig（tol 5e-3），超差拒绝接管并留证；红绿双验通过。

`_bind_one` 在 `fn_binding.py:116` 拿到了 `orig`，但它只流向 `FunctionDispatcher._orig`，
**仅在 bypass 时使用**。门禁链路 `manager.call → select → _gate → verify_impl(impl, provider.native)`
的参考系是 TT 自家的 `native`，从头到尾没见过 `orig`。

而 TT 的 native 是**手工复刻**的——`NativeSwiGLU` 的注释写着"与 megatron 公式逐字一致"，
`NativeRoPEBshd` 写着"与 megatron rope_utils 逐字一致"。这个"一致"目前是**注释里的声明，不是证据**。

实测（故意把框架原实现改成 `silu(y1)*y2*1.5`，模拟上游公式演进或方言分支）：

```
provenance: bind→remap ✅   select→selected ✅   call→called ✅
实际数值  : 框架原实现 vs TT 接管后  max_abs = 1.337
告警      : 无
```

即 **TT 把框架的计算语义换掉了，全链路零告警**。这恰好是文档里一直强调的
"TT 相对 symbrewrite 的核心差异化 = 替换有证据"，而函数级这条新通道把它漏了。

**按需的最小修法**：`_bind_one` 在绑定时多做一次对账——

```python
# orig 在手，probe_input 已有，verify 已有测量原语
args, kw = provider.probe_input(device=...)
rel = max_rel_err(provider.native(*args, **kw), orig(*args, **kw))
provenance.record(..., "bind", dotted, "remap", native_vs_orig=rel)
if rel > TOL:     # 复刻漂了：不绑，留证，框架保持原样
    return
```

约 15 行，复用现成的 `probe_input` 与 `witness.verify`。
收益：把"逐字一致"从注释变成每次启动都复验的证据，**顺带解决版本漂移的静默失效**——
megatron 改了公式，这里立刻红。

> 注：模块级接入（`Provider._self_check`）**已经**用 `module.forward`（原实现）做基准并
> 实测择优，做得是对的。缺口只在函数级——两条通道的正确性姿态应当拉齐。

### 4-C 绑定坐标散落、无版本维度、失效静默

> **状态（2026-10-21）：已落地**——坐标集中到 `provider/bind_targets.py`（含
> 版本窗口），`fn_binding.install()` 末尾打印摘要且 skipped 列出符号名。

三个 `bind_points` 分别写在三个语义包里：

```
impls/swiglu/base.py:34       ("megatron.core.fusions.fused_bias_swiglu", "swiglu")
impls/bias_swiglu/base.py:33  ("megatron.core.fusions.fused_bias_swiglu", "bias_swiglu_impl")
impls/rope_bshd/base.py:53    ("megatron.core.models.common.embeddings.rope_utils",
                               "_apply_rotary_pos_emb_bshd")
```

三个问题：

1. **没有任何一处能回答"TT 对 megatron 动了哪些手脚"**。现在靠 grep。provider 涨到 10 个、
   再支持 transformers 定义点时，这个问题会变成排障的主要成本。
   symbrewrite 用 `plugins/<目标框架>/` 的组织方式天然能回答（见 `TT_Symbrewrite_Absorption.md` §3）。
2. **无版本维度**。`_apply_rotary_pos_emb_bshd` 是 megatron 的**私有函数**（前导下划线），
   路径和签名随版本漂移是常态。symbrewrite 为此维护了 `transformers_4_32_1`…`4_52_3` 多个版本目录。
3. **失效是静默的**。实测 `bias_swiglu_impl` 在 mock 环境绑定 `skipped`，
   终端只打印 `[cuda_compat] activated: 18 policy symbols`，skipped 仅进 JSONL。
   生产上 megatron 换版本导致全部性能绑定失效，现象是"没报错但不加速了"——最难排查的一类。

**按需的最小修法**（不是 Contract Pack）：

- 建 `ttorch/provider/bind_targets.py`，把三条坐标集中成一张表，加一列 `version_window`
  （字符串即可，如 `"megatron>=0.9,<0.14"`），`Provider.bind_points` 改为引用表里的 key；
- `install()` 结束时打印一行摘要：`[tt] fn_binding: 3 bound, 0 skipped`，
  **skipped > 0 时额外打印具体符号名**。失效从此可见。

体量约 30 行。暂不做版本窗口的自动匹配逻辑——先把版本写下来，人能看见就够了；
真遇到两版共存再加解析。

---

## 5. 继续保留的设计（不要动）

1. **Bridge / Provider 落点分离**：`torch.cuda.*` 走 re-export + sweep；框架内部符号走定义点直打。
   两种落点对应两类代码组织，不是谁取代谁。
2. **绑定目标是 dispatcher 而非厂商实现**：让函数级自动继承全部治理能力。
3. **native 作为永不可禁用的功能底线** + 两级回退 + kill switch + bypass。
4. **fp32 参考轨道做精度门禁**：bf16 下同语义实现差（1-2 ulp ≈ 6e-3）与语义偏差（4e-2）
   只差一个数量级，阈值怎么拍都误伤；fp32 下拉开 5 个数量级。这条判据写在 `spec.py:164-167`，
   是本项目最有价值的工程决策之一。
5. **负载签名缓存按"特征维归并"**：最后一维进签名（kernel 正确性差异的实际载体），
   序列/批次维归并（变长训练下每个新长度都重跑门禁不可接受）。
6. **模块级 `_self_check` 用实测择优取代声明式变体表**：见 §6 撤回项 2。

---

## 6. 从上版撤回/降级的建议（避免过度设计）

| 上版建议 | 处置 | 理由 |
|---|---|---|
| `RewriteSpec` 十字段 DSL + Rewrite Engine 子系统 | **降级为 §4-A 的 40 行公共函数** | 当前 2 个 writer、4 种落点。DSL 的维护成本高于收益；等出现第 3 类 owner 或真实冲突再升级 |
| 五阶段事务 `DISCOVER→…→COMMIT` + 回滚栈 | **撤回** | 激活失败目前就是 fail fast 退出进程，没有"半激活后继续跑"的需求。回滚栈是为长驻服务设计的，TT 是训练进程 |
| `ActivationPlan` / `EffectiveRuntime` / dry-run | **降级到多卡前** | 单进程单平台下，`provenance` 的 session 快照已经够复现。多 rank fingerprint 对账是上多卡时的真需求，那时再做 |
| `CapabilityEvidence` 结构化证据模型 | **降级** | conformance 已在实测并打印。最小改进是把结果写进 provenance（几行），不需要新模型 |
| `VariantSpec` 精确点名变体表 | **撤回** | 代码走了更好的路：`_self_check` 用候选语义实测择优（`spec.py:154-216`）。symbrewrite 要为 7 个模型 × 3 变体维护一张表，TT 一次前向就定了，且对无 flag 的 Gemma 式硬编码同样有效。**实测优于声明**，符合项目文化 |
| Framework Contract Pack | **降级为 §4-C 的一列 `version_window`** | 完整契约包（调用点/seam/sentinel/feature gate）在只支持一套框架时是空壳 |
| 接 HF Kernels / Megatron ModuleSpec 官方 seam | **保留为中期选项，不进当期** | 当前主战场是 megatron eager 路径的函数符号，官方 seam 覆盖不到。等做 transformers 模型层时再评估 |
| `torch.library` / dispatcher 接入 | **挂起** | eager 路径的增量还没吃透（`torch_xmlir/nn` 下仍有十几个未覆盖的融合算子）。dispatcher 要求 schema/fake/autograd/opcheck 全套，投入产出比现在不划算 |
| Distributed Contract | **挂起到多卡前** | 还没上多卡 |
| 整包替换 `MODULE_REPLACE` | **撤回** | 当前无 apex/flash_attn 的接管需求 |

---

## 7. 建议实施顺序

### 当期（CUDA + KL3 主线，约 100 行）

1. **§4-B 函数级对账 `native vs orig`**——最高优先。它直接保护"替换有证据"这条差异化，
   且能顺带把版本漂移从静默变成显式。15 行。
2. **§4-C 绑定坐标集中 + skipped 摘要打印**——让"TT 动了什么""什么时候失效了"可见。30 行。
3. **§4-A 抽 `_rebind` 公共函数 + owner 留痕 + 检测覆盖全 writer**——消除重复与盲区。40 行。
4. 顺手：`fn_binding` 补 `uninstall()`；硬编码的 sweep 白名单改为参数。

### 中期（第二个框架 / 第二个伪装平台到来时）

5. `version_window` 从"写下来"升级为"启动时核对并告警"；
6. 按目标框架聚合的绑定清单导出（`TT 对 megatron/transformers 的改动全表`），供评审 diff；
7. 评估 HF Kernels seam adapter（仅当开始做 transformers 模型层）。

### 挂起（有明确触发条件才启动）

- 多卡 → Distributed Contract + 跨 rank fingerprint + ActivationPlan 重放；
- 真实 owner 冲突出现 → 升级为事务与回滚栈；
- eager 增量吃尽 → dispatcher / `torch.library`。

---

## 8. 验收标准（只列当期三条的）

| 验收项 | 通过标准 |
|---|---|
| 函数级对账 | 构造"框架原实现与 TT native 不等价"的夹具（如 `×1.5`），绑定必须被拒绝 + provenance 留 `native_vs_orig`；红绿校验：改回等价后必须放行 |
| 失效可见 | mock 掉一个 `bind_points` 的目标符号，终端必须出现 `skipped: <符号名>`，不能只进 JSONL |
| 检测无盲区 | 人为覆盖一个 provider 函数级绑定，`assert_bindings_intact()` 返回的 tampered 必须包含它（当前返回 0） |
| owner 可辨 | 同一符号被两个 owner 先后绑定时，留痕属性名能区分，且第二次绑定产生 conflict 记录 |
| 无回归 | KL3 实测全绿基线不变：demo 8 PASS / 等价性实验 B≡A 逐位一致 / QWEN_SMOKE resume diff=0.0 / fn 三函数加权 +13.5% |

---

## 9. 主要资料

### 官方/一手

- Python import/reload caveats（外部引用不自动重绑、旧实例不更新类）：
  https://docs.python.org/3.11/library/importlib.html
- PyTorch custom operators（schema / fake / autograd / opcheck）：
  https://pytorch.org/tutorials/advanced/python_custom_ops.html
- PyTorch compile guards / recompile：
  https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html
- Transformers monkey patching（官方标注为结构修改的最后手段）：
  https://huggingface.co/docs/transformers/v5.5.3/monkey_patching
- Hugging Face Kernels layers（官方 seam、mode、第三方 patch 警告）：
  https://huggingface.co/docs/kernels/v0.17.1/en/layers
- FlagGems selective enablement：
  https://docs.flagos.io/projects/FlagGems/en/latest/usage/selective.html
- Megatron-Core TransformerLayerSubmodules：
  https://docs.nvidia.com/megatron-core/developer-guide/0.16.1/apidocs/core/core.transformer.transformer_layer.html
- torch_npu 自动迁移约束（JIT、三方冲突、NCCL/HCCL 字符串）：
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/devaids/migrationtools/atlasfmkt_16_0019.html

### 生产案例

- SGLang PR #27774：按值 import 导致 NPU monkey patch 失效

### 本轮实测证据（本机 CPU venv，torch 2.12.0 / transformers 5.19.0）

- `assert_bindings_intact()` 覆盖 15 条，provider 函数级 0 条；
- 框架原实现与 TT 接管后 `max_abs=1.337`，provenance 三条事件全 pass；
- `_origin_` 前缀 bridge 与 fn_binding 完全相同；
- `setattr` 写入点共 3 处模块级 + 1 处实例级。

### 本地源码

- `/Users/vector/code/train/TTorch/ttorch/{bridge,provider,witness}/`
- `/Users/vector/code/train/train_klx/torch_xmlir/symbrewrite/`
