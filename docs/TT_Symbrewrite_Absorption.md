# torch_xmlir/symbrewrite 吸收分析：结构层面

> 刷新于 2026-10-20，TTorch 基线 commit `bd1b90f`。
> 分析对象：`torch_xmlir/symbrewrite/`（17 插件 / 176 文件 / 13352 行 / 148 处 `SYMBOL_REPLACE`）
> 与 `torch_xmlir/nn/`（28 个厂商融合算子封装）。
> 与 `TT_KLX_Plugin_Absorption.md` 的分工：那份看 xswift_ext / xmegatron_ext（框架集成层），
> 本份看 symbrewrite（替换引擎）+ `nn/`（kernel 封装层）。
>
> **本版变化**：上两版的重点是"有哪些算子可以搬"与"机制缺什么"。`bd1b90f` 之后，
> 算子已搬进 4 个（rmsnorm / swiglu / bias_swiglu / rope_bshd），机制缺口也补了大半。
> 本版把重点整体移到**结构（组织方式）层面**——这是 symbrewrite 真正还没被吸收、
> 而且 TT 正在开始吃亏的部分。单点算子选题收进 §5，不再是主体。

---

## 0. 结论先行

| # | 结论 | 证据 |
|---|---|---|
| 1 | **机制层面已基本吸收完毕**：工厂延迟构造、定义点直打、原实现留痕三项都已落地，且 TT 在门禁/回退/可观测上本就更强。机制维度已无大额欠账 | §1 |
| 2 | **真正没吸收的是「组织方式」**：symbrewrite 按**目标框架**组织（`plugins/<框架>/`），TT 按**语义**组织（`impls/<语义>/`）。两种切法各有适用面，但 TT 当前只有一种，导致"TT 对 megatron 动了哪些手脚"没有任何一处能回答 | §2.1 |
| 3 | **版本维度缺失的代价马上会显现**：TT 绑的 `_apply_rotary_pos_emb_bshd` 是 megatron **私有函数**，symbrewrite 为同类问题维护了 6 个 transformers 版本目录；TT 一个都没有，且失效时静默 | §2.2 |
| 4 | **分层开关（装了≠生效）是被低估的结构能力**：symbrewrite 17 个插件默认只开 4 个。TT 现在是"注册即全开"，没有按框架/按语义的灰度粒度 | §2.3 |
| 5 | **上版建议的 `VariantSpec` 精确点名表应当撤回**：代码走了更好的路——`_self_check` 候选语义实测择优。symbrewrite 要为 7 模型 × 3 变体维护一张表，TT 一次前向就定了，对无 flag 的 Gemma 式硬编码同样有效 | §4 |
| 6 | **吸收的同时必须补上 symbrewrite 唯一的结构性短板**：它 148 条替换没有一条带等价性证明。TT 的函数级通道当前**也没有**（实测静默偏差 1.337），这条必须先补，否则 TT 在该通道上退化成"多一层包装的 symbrewrite" | §3 |

---

## 1. 机制维度：已吸收清单（可以翻篇了）

| 机制 | symbrewrite 实现 | TT 现状 | 状态 |
|---|---|---|---|
| 工厂延迟构造 | `is_wrapper=True` → enable 期调工厂取真身（`symbol_rewrite.py:139-140`） | `Entry.impl_factory`（`binding_registry.py:51`） | ✅ 已吸收，并修掉了 `torch.cuda.Stream` 被绑成函数导致 transformers 5.x 无法 import 的阻断项 |
| 字符串 FQN + 定义点直打 | `SYMBOL_REPLACE("a.b.C", target)` | `PerfBinding(dotted, module, attr, factory)` | ✅ 已吸收 |
| 原实现留痕 | `__origin_<name>` | `_origin_<name>` | ✅ 已吸收（前缀撞名问题见 `TT_Rewrite_Mechanism_Design.md` §4-A） |
| 依赖预拉起 | `MODULE_DPES` | import 钩子后置绑定（等价目的、不同路径，判据见 Rewrite 文档 §3） | ✅ 目的已达成 |
| from-import 扫尾 | 不做 | `_rebind_and_sweep` + `verify_sweep` | TT 更强 |
| 等价性门禁 | **无** | 三级门禁 + fp32 参考轨道 | TT 更强（函数级有漏，§3） |
| 运行时回退 | **无** | 两级回退 + kill switch + bypass | TT 更强 |
| provenance | 启动打一行 | 统一事件流 + JSONL | TT 更强 |

**判断**：机制维度不必再向 symbrewrite 学。剩下的欠账全在组织层。

---

## 2. 结构维度：三项可吸收的组织方式

### 2.1 按「目标框架」切，还是按「语义」切

两套系统的目录结构反映了两种完全不同的切法：

```
symbrewrite（按目标框架切）          TTorch（按语义切）
plugins/                             impls/
  torch/          ← 改 torch 的        rmsnorm/   ← RMSNorm 的 native + 各硬件 impl
  transformers/   ← 改 HF 的           swiglu/    ← SwiGLU 的 ...
  apex/           ← 改 apex 的         bias_swiglu/
  flash_attn/                          rope_bshd/
  swift/
  transformers_4_32_1/  ← 版本专用
  transformers_4_52_3/
```

**TT 的切法对"一个语义的多硬件实现"是对的**——`rmsnorm/base.py` 放 native、`kl3.py` 放厂商实现，
加一块硬件就加一个文件，这个维度不该变。

但"**改谁**"这个维度现在被摊进了各语义包。实测确认三条绑定坐标散在三个文件：

```
impls/swiglu/base.py:34       ("megatron.core.fusions.fused_bias_swiglu", "swiglu")
impls/bias_swiglu/base.py:33  ("megatron.core.fusions.fused_bias_swiglu", "bias_swiglu_impl")
impls/rope_bshd/base.py:53    ("megatron.core.models.common.embeddings.rope_utils",
                               "_apply_rotary_pos_emb_bshd")
```

后果：**没有任何一处能回答"TT 对 megatron 动了哪些手脚"**，现在只能靠 grep。
这个问题在 provider = 1 时不存在，在 = 4 时开始别扭，在 = 10 且跨两个框架时会变成排障主成本。
symbrewrite 看一眼 `plugins/transformers/` 就能回答。

**按需的吸收方式**（不是推翻现有结构）：保留 `impls/<语义>/` 不动，
只把「目标坐标」这一维抽出来集中：

```python
# ttorch/provider/bind_targets.py —— 一张表，按框架分组
MEGATRON = {
    "swiglu":      ("megatron.core.fusions.fused_bias_swiglu", "swiglu",
                    "megatron>=0.9,<0.14"),
    "bias_swiglu": ("megatron.core.fusions.fused_bias_swiglu", "bias_swiglu_impl",
                    "megatron>=0.9,<0.14"),
    "rope_bshd":   ("megatron.core.models.common.embeddings.rope_utils",
                    "_apply_rotary_pos_emb_bshd", "megatron>=0.9,<0.14"),
}
```

语义包里 `bind_points` 改为引用这张表的 key。这样：
语义包仍然只管"语义 + native + impls"，而"TT 改了哪个框架的什么"集中可见、可 diff、可导出评审。
体量约 30 行。

### 2.2 版本维度：symbrewrite 用目录承载，TT 一个都没有

symbrewrite 的 17 个插件里有 6 个是**版本专用**的：
`transformers_4_32_1` / `4_38_2` / `4_43_3` / `4_45_2` / `4_49_0` / `4_52_3`。
这不是过度设计——它是被上游漂移逼出来的。

TT 当前绑定的三个符号里，`_apply_rotary_pos_emb_bshd` 带**前导下划线**，
是 megatron 的私有函数：路径、函数名、签名随版本变动都属正常，上游没有任何兼容承诺。

**而 TT 的失效是静默的**。实测：目标符号不存在时 `_bind_one` 记一条
`skipped: 目标符号不存在` 进 JSONL，终端只显示
`[cuda_compat] activated: 18 policy symbols + 0 default-deny` —— 没有任何提示。
生产上的现象会是"没报错，但性能回到基线"，这是最难排查的一类故障。

**按需的吸收方式**（不做自动版本解析）：
1. §2.1 的表里加一列 `version_window`，**先只是写下来**，人能看见即可；
2. `install()` 末尾打印一行摘要：`[tt] fn_binding: 3 bound, 0 skipped`，
   **skipped > 0 时列出具体符号名**。

真遇到两个 megatron 版本需要不同坐标时，再加解析逻辑。现在不要建版本匹配引擎。

### 2.3 分层开关：「装了」不等于「生效」

symbrewrite 17 个插件，`plugins.default.xflags` 里**默认只开 4 个**
（`torch` / `torch_vision` / `torchaudio` / `transformers`），13 个默认关闭
（含全部版本专用插件、`apex`、`flash_attn`、`profiler`、`reproduce`、`swift`）。

这个结构能力的价值在于**灰度粒度**：新接一个框架的替换，可以先随包发布、默认关闭、
指定环境打开验证，再改默认值。TT 现在是 `SEMANTICS` 里登记了就全量注册、全量绑定，
只有"全局 bypass"和"按 impl 的 kill switch"两个极端粒度，中间缺一档
"按框架/按语义关掉这一组绑定"。

**判断：当前不急**。TT 只接一个框架、4 个语义，`disable(impl_name)` 够用。
但接第二个框架时这一档会立刻需要——建议届时直接对标 `xflags` 的形态
（环境变量 `TT_BIND_DISABLE=megatron.rope_bshd` 这类），不要自造复杂的配置系统。

> 一个必须避开的坑：symbrewrite 的 xflags 文件头声明"列表从上到下优先级依次降低，
> 重复改写低优先级被忽略"，但 Python 侧 `add()`（`symbol_rewrite.py:43-48`）的实际行为是
> **打 WARN 后用新 target 覆盖**，且 `enable_rewrite_all` 在 py 源码中无调用方
> （实际加载顺序在编译产物里），**优先级语义无法从源码验证**。
> TT 若做优先级，必须做成可审计的——覆盖事件进 provenance，而不是只 print。

---

## 3. 吸收的前提：先补上 symbrewrite 唯一的结构短板

symbrewrite 的 148 条替换，**没有一条带等价性证明**。这是 TT 最大的差异化来源，
但本轮实测发现 **TT 的函数级通道当前同样没有**：

```
实测：把框架原实现改成 silu(y1)*y2*1.5（模拟上游公式演进），绑定后
  provenance : bind→remap ✅  select→selected ✅  call→called ✅
  实际数值   : 框架原实现 vs TT 接管后 max_abs = 1.337
  告警       : 无
```

根因是参考系选错了：门禁链路 `manager.call → select → _gate → verify_impl(impl, provider.native)`
拿 TT 自家的 `native` 当基准，而 `native` 是**手工复刻**的
（`NativeSwiGLU` 注释写"与 megatron 公式逐字一致"、`NativeRoPEBshd` 写"与 megatron rope_utils 逐字一致"）。
这个"一致"目前是注释里的**声明**，不是**证据**。

`_bind_one` 在 `fn_binding.py:116` 已经拿到了 `orig`，但它只用于 bypass。
补这条对账约 15 行，复用现成的 `probe_input` 与 `witness.verify`——
详见 `TT_Rewrite_Mechanism_Design.md` §4-B。

**这条必须排在所有吸收动作之前**：否则每多吸收一个 symbrewrite 的知识点，
就多一处"没有等价性证明的替换"，TT 在函数级通道上就退化成"多一层包装的 symbrewrite"。

对照：模块级接入（`Provider._self_check`，`spec.py:154-216`）**已经**用 `module.forward`
原实现做基准，而且做得很好。两条通道的正确性姿态应当拉齐，不是新增机制。

---

## 4. 撤回：`VariantSpec` 精确点名表

上版（2026-10-15）建议 Provider 的键从 `semantic` 扩成 `semantic × variant`，
用 `VariantSpec` 精确点名各模型的 RMSNorm 变体——对标 symbrewrite 的
`wrapper_mock_RMSNorm` / `wrapper_mock_zero_centered_gamma_RMSNorm` / `wrapper_mock_Gemma4RMSNorm`。

**这条撤回。** 代码走了更好的路（`spec.py:154-216`）：

```
_self_check 两判据（分离）：
  1) 变体识别（选择题，无阈值）：候选 scale 语义（w 直乘 / 1+w）各跑一次，取误差最小者
     —— flag 嗅探只作先验排序，不作结论
  2) 正确性准入（判断题）：最小误差必须 < admit_tol 才接管
轨道用 fp32：同语义实现差 5e-7 级、语义偏差 4e-2，拉开 5 个数量级
```

为什么比声明式变体表好：

| | VariantSpec（声明） | `_self_check`（实测） |
|---|---|---|
| 维护成本 | symbrewrite 为 7 模型 × 3 变体维护一张表，上游加一个模型就要加一条 | 零维护，新模型自动适配 |
| 对无 flag 的硬编码变体 | 必须手工点名（Gemma 的 `(1+w)` 写死在 forward 里，无 flag 可嗅探） | 自动识别——实测哪个对用哪个 |
| 厂商替换丢/伪造 flag | 表会失配 | 以实测为准，flag 只作先验 |
| 符合项目文化 | 声明 | **证据先于声明** |

**保留的边界**：`_self_check` 当前只覆盖 scale 语义（`w` vs `1+w`）这一个自由度。
Gemma4 的 `with_scale=False`（**无 weight 属性**）和 RoPE 的 layout 变体不在其覆盖内——
前者会在 `wrap_module` 的 `isinstance(weight, Parameter)` 处抛 `TypeError` 被安全跳过（正确行为），
后者走 `supports()` 回退 native。
结论：**实测择优作为主路径，只在实测无法区分时才需要声明**，而不是反过来。

---

## 5. 单点选题：剩余增量空间（已不是本文重点）

`torch_xmlir/nn/` 有 28 个厂商融合封装，symbrewrite 默认开启的 transformers 插件
只替换了 RMSNorm 与 `apply_rotary_pos_emb*`。两者相减即 Provider 的真实增量空间。

**已接入（4/28）**：`rmsnorm`、`swiglu`、`bias_swiglu`、`rope_bshd`

**剩余候选**，按 LLM SFT 主链路优先级：

| 优先级 | 候选 | 理由 |
|---|---|---|
| 1 | `dropout_add_rms_layernorm` / `bias_dropout_add` | 残差+norm 融合，省中间张量，对显存也有收益 |
| 2 | `clip_grad` | 整网一次但张量多，`multi_tensor` 形态收益明显 |
| 3 | `scaled_softmax` / `softmax_with_mask` | attention 路径（需先确认与 flash/eager 分支的关系） |
| 4 | `gelu_with_bias`、`rotary_pos_emb_index`、`bmm`/`baddbmm` | 覆盖面较窄 |

**已被厂商默认融合的不要碰**：`rms_norm`、`rope` 在 KL3 上 symbrewrite 默认已替换，
Provider 在那里净收益≈0（已实测验证：等价性实验 B≡A 逐位一致），只剩机制价值。

每项接入前按 §3 准备 reference（与**被替换的原实现**对账，不是与自家 native）。

---

## 6. 明确不吸收

| 不吸收 | 位置 | 理由 |
|---|---|---|
| `torch.compile` → no-op 装饰器 | `plugins/torch/__init__.py:127-133` | 静默吞掉用户的编译意图。TT 应报 unsupported，不假装编译了 |
| `is_nccl_available` → `lambda: True` | `plugins/torch/__init__.py:83-86` | 直接撒谎。分布式面应走 backend 别名注册，不伪造能力查询 |
| `mock_torch_init.py`（1346 行）整包 | `plugins/torch/` | 目标是"CPU 模拟 GPU 随机数路径"，与 TT 目标不同 |
| `caching_allocator_warmup` 改写 | `plugins/transformers/mock_modeling_utils.py` | 复刻 transformers 4.57 旧行为，属版本回退补丁，随上游演进必失效 |
| NUMA 亲和写死 `splitlines()[1:9]`（8 卡） | `plugins/torch/enhance_context.py:43` | 写死单机 8 卡，与 KLX 插件里 `gpu_idx < 7` 同类问题 |
| 「零验证直接替换」的工作方式 | 全局 | symbrewrite 148 条无一条等价性证明。这是 TT 的反面教材，不是可吸收项（见 §3） |

---

## 7. 行动清单（按需，逐步）

### 当期（约 50 行，排在扩算子之前）

1. **§3 函数级补 `native vs orig` 对账**——保护差异化，顺带把版本漂移从静默变显式；
2. **§2.1 + §2.2 绑定坐标集中成一张表（含 `version_window` 列）+ skipped 摘要打印**。

### 中期（触发条件：接第二个框架 / 第二块伪装硬件）

3. `version_window` 从"写下来"升级为"启动核对并告警"；
4. 按目标框架聚合的绑定清单导出（供评审 diff）；
5. §2.3 的分层开关，对标 `xflags` 形态，不自造配置系统。

### 挂起（有明确需求再启动）

- 整包替换（`MODULE_REPLACE`）——当前无 apex/flash_attn 接管需求；
- 优先级/overlay 机制——当前无符号重叠；
- `LOG_SYMBOL_REPLACE` 式 caller→impl 源位置日志——现有 `call_trace` 够用，
  排障真的卡住了再加。

---

## 8. 风险提示

1. **版本漂移是当前最大的未设防风险**：绑的是 megatron 私有函数，上游无兼容承诺，
   失效静默。§2.2 的两条（写下版本 + 打印 skipped）是最低成本的防线。
2. **双栈共存无 owner 检测**：symbrewrite 与 TT 都在改符号、都留 `*origin_` 痕
   （前缀还相同），且 TT 的完整性检测只覆盖 bridge 的 15 条，provider 函数级 0 条覆盖。
   详见 `TT_Rewrite_Mechanism_Design.md` §4-A。
3. **不要因为"吸收"而放宽门禁**：每吸收一个 symbrewrite 知识点，都必须带着
   reference 与门禁一起进来。这是 TT 与 symbrewrite 的唯一本质差别。
