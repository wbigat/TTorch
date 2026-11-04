# transformers symbrewrite 插件：TT 最新代码覆盖能力复核

> 复核日期：2026-10-28
> 原始文档：`transformers_symbrewrite_插件_TT覆盖能力分析_20261023.md`
> TT 仓库：`/root/dl/TTorch`
> 代码基线：`4ba6c6c4a201dfa6f3757ed02435843e07243eb3`（dev）
> 环境：torch `2.9.0+cu129`、transformers `5.12.1`、ms-swift `4.3.1`
> 本轮方式：源码交叉审查、隔离进程最小反例、现有单测；未修改源码，未重跑整网性能。
> 本文记录问题与后续计划，不表示相关问题已经修复。

## 0. 结论

原文的职责划分大体合理，但把“机制可以接入”与“实际实现已覆盖”混为一谈，且低估了 transformers RoPE 的补齐成本。

当前代码已经具备标准/zcg RMSNorm 注册、native、KL3 实现、延迟自检和框架建模入口包装；但 transformers RoPE、Gemma4 no-scale、框架版本执行校验仍未闭环。

比扩展算子覆盖更应优先处理的是：梯度准入、生产 dtype 精度验证、退化输入误接管、绑定生命周期和启动失败原子性。本轮独立反例已确认其中多个问题。现有 44 条单测通过，不能排除这些缺陷。

此前“没有未解决关键问题”“SSA 相同即结构等价”“all 证明不依赖任何 xmlir patch”等结论需要收紧；应以本文的证据与适用范围为准。

## 1. 分析口径修正

### 1.1 机制能力不等于实际覆盖

以下状态必须分别登记：

1. 机制是否支持接线；
2. 是否定义了准确的语义与接口契约；
3. 是否有 native/vendor 实现和适用范围；
4. 是否登记了真实框架调用点；
5. 是否完成前向、反向、生产 dtype、边界和端到端验证。

建议以“唯一语义 → 框架接口 → 实现范围 → 验证证据”为覆盖口径，不再按 patch 条目数直接计算覆盖率。

### 1.2 原文统计与版本前提

原文“9 条 RMSNorm”清单包含 `BertAttention`，后续又将其归为策略强制和结构替换，分类有交叠。不同版本目录中的重复注册，也不等于本机同时生效的能力。

原文异机 transformers `5.19.0` 与当前 `5.12.1` 不同。旧版本包元数据故障、历史性能数字、旧插件失效行为需要分别重验，不能直接外推。

因此，本轮不继续使用“22/39 已覆盖、补小改可达 28/39”的比例。

## 2. 已落地的能力

| 能力 | 当前状态 | 限制 |
|---|---|---|
| 标准 RMSNorm | provider、native、KL3 impl 已注册 | 不代表所有同名模块的 dtype/乘法顺序完全等价 |
| zcg RMSNorm | 独立 provider 已注册，KL3 impl 已补齐 | 无 flag 重试存在退化输入风险；显式错误 flag 仍阻挡正确变体 |
| zcg native fp32 scale | 对齐 Qwen3.5 的 `1+w.float()` 与 fp32 乘法 | vendor 生产低精度还需专项验证 |
| 延迟自检 | wrap 期不跑前向、不改 dtype，首次调用验证 | 单点输入、真实 weight、结果缓存；不能据此证明任意并行包装/任意输入安全 |
| 函数级通道 | Megatron 三个坐标已登记 | 没有 transformers RoPE 接入 |
| 建模入口 auto-wrap | 环境变量门控 `mcore_bridge.get_mcore_model` | 入口 scoped，不是调用者 scoped；生命周期缺陷见 §4 |
| vendor 符号恢复 | 启动时读取 `TT_RESTORE_VENDOR` | 一次性 SymbolRewrite 恢复，不是整个厂商 patch 体系关闭 |
| FX 对比 | 实现了算子名称序列比较与 provenance | 不是完整 SSA 图结构等价证明 |

主要实现：

- `ttorch/provider/impls/rmsnorm/__init__.py:23–29`
- `ttorch/provider/impls/rmsnorm/zcg_base.py:32–38`
- `ttorch/provider/spec.py:73–106,161–189,191–322`
- `ttorch/provider/bind_targets.py:20–36`
- `ttorch/provider/auto_wrap.py:30–100`
- `ttorch/provider/vendor_patch.py:43–105`

以上行号均相对本文代码基线，后续修改后可能漂移。

## 3. 对照原文六个 Gap

### Gap-1：Attention 语义

**判定：对应实现确实缺失，但原文对机制限制的描述过宽。**

默认 `ProviderModuleForward` 最终调用 `(x,w,eps)`，正常接管路径不转发原始额外参数，直接复用它包装 Attention 不成立。

但 `FunctionDispatcher`、`manager.call`、`ProviderImpl.supports` 接受任意参数，provider 子类也可以重写 `wrap_module`、`probe_input`。不能把整个分发体系说成固定 RMSNorm 三参数。

更实际的机制限制是：`verify_impl` 当前按单 Tensor 输出处理，尚不能直接验证 Attention 常见的 output/weights/cache 结构。

不建议为补齐统计数字复制整个 Attention 类。应先界定具体后端契约，包括 mask、dropout、GQA、causal、padding、KV-cache，再决定是否由 TT 接入；利用框架原生 attention 后端选择机制仍是合理方向。

证据：

- `ttorch/provider/spec.py:73–106,161–189`
- `ttorch/provider/fn_binding.py:77–82`
- `ttorch/provider/manager.py:437–483`
- `ttorch/witness/verify.py:77–85`

### Gap-2：transformers RoPE

**判定：仍存在，且不是“新增约 30 行坐标表”的纯数据工作。**

当前 `bind_targets.py` 只有 Megatron 三个坐标。现有 `rope_bshd` 输入单个 `t` 与角度 `freqs`，内部计算 cos/sin，返回单 Tensor。

transformers 至少有四类不同契约：

| 类型 | 差异 |
|---|---|
| 文本 RoPE | 输入 q/k/cos/sin，双输出；广播维度可变，部分模型保留未旋转尾部 |
| 视觉 RoPE | 常见三维输入；明确 fp32 计算后恢复各自 dtype |
| 多模态 RoPE | 增加 `mrope_section`，先组合 temporal/height/width |
| Gemma4 per-tensor | 单 Tensor 输入输出，布局和广播约定不同 |

厂商 transformers mock 使用双输入 cos/sin 接口 `FusedRoPEFunc`，不是当前 TT 的角度输入 kernel 包装。

`supports()` 只能判适用性，不能替代签名、布局、输出结构和精度转换。直接把 transformers 坐标指向 `rope_bshd` 不成立。

正确顺序：定义契约/native → 支持结构化输出验证 → adapter/vendor → 梯度与 dtype 验证 → 登记坐标 → 端到端验证。

证据：

- `ttorch/provider/bind_targets.py:20–36`
- `ttorch/provider/impls/rope_bshd/base.py:34–69`
- `ttorch/provider/impls/rope_bshd/kl3.py:19–29,41–51`
- `ttorch/witness/verify.py:77–85`
- 安装包 `transformers/models/qwen3/modeling_qwen3.py:159–181`
- 安装包 `transformers/models/qwen3_5/modeling_qwen3_5.py:590–605`
- 安装包 `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:161–172,564–606`
- 安装包 `transformers/models/gemma4/modeling_gemma4.py:787–806,1245–1265`

安装包根目录：`/root/miniconda/envs/python312_torch29_cuda/lib/python3.12/site-packages/`。

### Gap-3：Gemma4 无 scale

**判定：仍存在，已用当前社区原始类实测确认。**

`with_scale=False` 没有 weight，`wrap_module` 抛 TypeError，接管被跳过。并非仅改入口即可：forward、自检、FX 比较仍直接读取 `module.weight`。

本机 Gemma4 社区实现：

- `with_scale=True`：weight 初始化 ones，直接乘 `weight.float()`，不是 zcg；
- `with_scale=False`：无 weight，不只用于 v_norm，也用于 router norm；
- 归一化与 scale 乘法均在 fp32，最后恢复输入 dtype；
- 归一化使用 `pow(...,-0.5)`，不能只凭算子序列与标准 rsqrt 相异就认定数学不等价。

当前标准 native 的降精度乘法顺序不完全复刻 Gemma4。有 scale 分支也需非单位权重、bf16/fp16 专项验证。

应明确 no-scale 语义，不能任意把所有无 weight 的同名模块补成 ones；若使用 zcg 表达，恒等 scale 对应 zeros，不能混用。

证据：

- `ttorch/provider/spec.py:104,179–180,219,274`
- `ttorch/provider/impls/rmsnorm/base.py:27–36`
- 安装包 `transformers/models/gemma4/modeling_gemma4.py:193–211,1214,1341`

### Gap-4：FlashAttention 包元数据

**判定：平台兼容风险成立，但本机没有复现原文的元数据缺失故障。**

本轮实测：

- `importlib.metadata.version('flash_attn')` → `2.4.2+0e1a3d1`
- `_is_package_available('flash_attn',return_version=True)` → `(True,'2.4.2+0e1a3d1')`
- 当前 `is_flash_attn_2_available()` → `True`

当前 transformers 5.12.1 已捕获部分元数据缺失，并检查 distribution mapping；与旧版本的检查逻辑不同。无有效 `__spec__` 的桩仍可能使 `find_spec` 报错，须按部署环境验证。

TT 未内置探测替换或元数据补齐。原文“机制提供、策略归平台栈”的职责判断仍合理，不建议内置恒真返回。

证据：安装包 `transformers/utils/import_utils.py:50–76,996–1010`。

### Gap-5：框架版本策略

**判定：仍存在。**

torch 版本窗口有执行检查；框架 `version_window` 只是说明字符串，`PerfBinding` 没有执行版本窗口字段。

符号存在检查不是签名/语义版本检查；当前 native-vs-orig 无法验证时允许 unverified 继续绑定，不能替代框架兼容管理。

应补明确的支持版本、接口契约验证和失效证据，不必为每个版本复制整套目录。

证据：

- `ttorch/provider/bind_targets.py:11–14,50–69`
- `ttorch/provider/fn_binding.py:43–57,186–206`
- `ttorch/platform/manager.py:53–68,165`

### Gap-6：分发热路径开销

**判定：值得观察，但现有测量不够纯净。**

`measure_dispatch_tax` 将直接调用 `provider.native` 与 `manager.call` 选中的实现比较。若后者选 vendor，两边 kernel 不同，会把融合收益混进分发税。

此外，当前默认 RMSNorm probe 不是历史注释里的 D=256，小 shape 历史 14% 不能作为当前通用量级；测试只验证测量可用，不设退化阈值。

应对同一选中实现比较直接调用、manager 分发、FunctionDispatcher、模块 wrapper，并给 shape/dtype 分组结果与多次波动。

证据：`ttorch/witness/diagnostics.py:202–243`。

## 4. 新确认的关键问题

### 4.1 训练准入会放行断梯度实现〔最高优先级，实测〕

`verify_impl` 仅在 native、候选输出都 `requires_grad` 时检查 backward。候选输出数值正确但 detached，反向检查被跳过。

最小反例：native 为 `x.square()`，候选为 `x.square().detach()`。

结果：门禁 `True / pass`，`max_grad_err=None`。

应先检查梯度契约与梯度存在性，再比较要求存在的梯度，不能把缺图当作不需要反向验证。

证据：`ttorch/witness/verify.py:82–105`。

### 4.2 生产 dtype 没有正确性对账〔高优先级，实测〕

精度与反向统一升 fp32；原始 dtype 输入仅用于性能计时。

最小反例：候选 fp32 时正确，bf16 时返回两倍结果。门禁仍 `True / pass`。

这不证明当前厂商 kernel 已经出错，但证明门禁无法排除该类错误。应同时保留 fp32 数学对账与生产 dtype 前反向对账。

zcg vendor 的 `w+1.0` 在 weight dtype 执行，而 native 使用 `1+w.float()`，存在舍入顺序差异，需要真实低精度与非单位权重验证。

证据：

- `ttorch/witness/verify.py:54–73,108–120`
- `ttorch/provider/impls/rmsnorm/zcg_kl3.py:30–31`

### 4.3 首次退化输入误接管〔高优先级，实测〕

模块自检只使用首次真实输入、实际 weight，成功后缓存。

最小反例：无 flag 的 zcg 模块、weight=zeros、首次 x=zeros。标准 provider 自检通过。后续 x=ones 时，标准 native 与 zcg 原模块最大误差 `0.9999995`。

需要非退化验证输入；不能修改真实参数或污染训练状态。单点数据一致不能证明语义一致。

impl 门禁的 RMSNorm probe 又恒用 ones weight，不能充分证明非单位 weight 行为。

证据：

- `ttorch/provider/spec.py:71–83,281–303`
- `ttorch/provider/impls/rmsnorm/base.py:62–72`
- `ttorch/provider/impls/rmsnorm/zcg_base.py:41–50`

### 4.4 显式错误 flag 阻挡正确变体〔实测/源码确认〕

自检失败虽重试，但重试仍执行 flag 预拒条件。错误变体被 rejected，正确变体又被错误 flag 挡住，最终只能退原生。

本轮确认：实际 zcg 模块显式 flag=False 时，正确 zcg provider 的 `wrap_module` 返回 None。

“伪造 flag 同样被纠正”的注释不成立。需区分首次先验分流与已经由自检否定先验后的重试。

证据：`ttorch/provider/spec.py:84–98,181–188`、`ttorch/provider/manager.py:238–271`。

### 4.5 FX 算子序列不是 SSA 结构等价〔实测〕

仅比较 `call_function` 的 target 名称序列，没有比较数据依赖、常量、归约维度和输出连接。

最小反例：模块 epsilon=1e-6，native epsilon=0.1。仍 `ssa_equal=True`，实际输出最大误差 `0.0465369`。

相同算子序列不能证明结构或语义等价；不同序列也可能数学等价。当前结果仅辅助留痕，不参与拒绝决策。

应先纠正命名与声明；若确需 SSA 结构证据，则比较规范化节点参数与数据依赖，不将图一致进一步等同于任意输入数学证明。

证据：`ttorch/provider/spec.py:224–236,303–314`。

### 4.6 auto-wrap/函数绑定生命周期〔多个最小反例确认〕

| 问题 | 本轮结果与影响 |
|---|---|
| 已加载别名漏扫 | 同名引用被修改，`_build_model` 别名仍指向原函数；真实 swift 使用 `_get_mcore_model` 别名 |
| unregister 后消费者残留 | 定义点恢复，但扫尾修改过的消费者仍持有 wrapper |
| 已加载入口重复包装 | `bind_loaded` 后 `install` 再绑一次，形成两层 wrapper |
| 后续缺符号不打印 | install 之后的 import-hook skip 不刷新摘要 |
| false 状态错误标成 pending | 已加载但缺符号/被拒也可能显示“等待 import” |
| 成功重试状态陈旧 | 成功路径没有清除旧 `_SKIPPED` |
| uninstall 再 activate 不完整 | `_BINDINGS` 清空，但 manager `_fn_registered`、auto-wrap `_ENABLED` 未同步重置 |

本轮伪框架执行结果：

- `alias_not_swept=True`
- `consumer_was_swept=True`
- `unregister_source_restored=True, consumer_still_wrapped=True`
- 二次绑定调用结果：`('wrapper',('wrapper','original'))`

应分别验证源头、已加载同名/别名消费者、未来 import、关闭后未来建模、重复启停。auto-wrap 目前仅对入口 scoped，任何调用该 mcore_bridge 入口的用户都会受影响，不只 swift 调用者。

证据：

- `ttorch/utils/_rebind.py:102–145`
- `ttorch/provider/fn_binding.py:121–142,209–242,281–325,339–349`
- `ttorch/provider/auto_wrap.py:43–96`
- 安装包 `swift/megatron/model/utils.py:4,83–85`

### 4.7 完整启动链失败原子性〔高优先级，实测〕

bootstrap 成功后继续 provider 激活、vendor restore、fn_binding 安装，但这些阶段没有总的回滚保护。

最小反例：设置 `TT_RESTORE_VENDOR=bogus` 后调用 `activate()`。接口抛 AssertionError，但 `tt.is_active()` 仍 True。

应在修改状态前校验配置，并为完整激活链提供一致的失败状态与恢复语义。不能仅凭 bootstrap 回滚测试宣称所有阶段均具原子性。

证据：

- `ttorch/__init__.py:42–67`
- `ttorch/bridge/bootstrap.py:112–136`
- `ttorch/provider/vendor_patch.py:87`

### 4.8 扩展前需要收紧的机制边界〔源码确认〕

1. 函数 native-vs-orig 对账异常会 `unverified → True`，接口/返回结构不兼容也可能进入绑定，之后回退是 TT native 而非原框架函数。新增接入不能依赖这条宽松路径保证安全。
2. `supports()` 调用位于运行时异常回退 try 之外；谓词异常可直接穿透调用链。
3. 缓存签名不含完整 shape、stride、kwargs 名称；容器参数只按类型归并。并不意味着当前所有调用已经错误，但扩展 mRoPE/布局相关契约前必须重新定义键与验证边界。
4. RoPE 门禁强制所有浮点位置参数 requires_grad，包括 freqs；native 对 freqs 可微，而当前 vendor backward 返回 freqs 梯度 None。即便 t 的厂商 backward 修复，也不代表门禁自然放行。需要先声明 freqs 是否属于训练梯度契约。

证据：

- `ttorch/provider/fn_binding.py:186–201`
- `ttorch/provider/manager.py:423–435,466–483`
- `ttorch/witness/verify.py:67–72,98–105`
- `ttorch/provider/impls/rope_bshd/kl3.py:25–29`

## 5. vendor restore 与性能对比边界

### 5.1 当前不是严格互斥的性能模式

| 开关 | 实际控制 |
|---|---|
| `TT_PLUGIN_ENABLE=1` | 自动激活 TT，不关闭 xmlir 性能 patch |
| `TT_RESTORE_VENDOR=perf/all` | 一次性符号恢复，不自动旁路 TT、不关闭厂商 hook |
| `TT_PROVIDER_AUTO_WRAP=1` | 包装建模入口，不保证厂商性能替换已关闭 |
| provider bypass | 回到保存的原实现；该实现可能已被 xmlir 融合 |
| XSWIFT/XME 插件开关 | 控制对应扩展，不等于关闭 torch_xmlir symbrewrite |

不设 restore 不等于“纯 xmlir 基线”：TT 函数绑定仍会安装，手动/自动 wrap 也可接管。

### 5.2 perf 会恢复 TT 尚未覆盖的 RoPE

`PERF_HINTS` 按名称匹配 RMSNorm/rotary；TT 当前只登记 Megatron RoPE，没有 transformers RoPE。

因此 perf 的准确含义是“按名字恢复一批性能符号”，不是“只关闭 TT 已覆盖的优化”。恢复后的 transformers RoPE 可能回到社区分解，而没有 TT 接管。

### 5.3 all 不是关闭整个 xmlir 插件体系

当前仅读 `SymbolRewriteRegister`，按 `__origin_<attr>` 恢复属性：

- 不处理独立 `ModuleReplace` 注册表与 `sys.modules` 替换；
- 不卸载厂商 import hook；
- 不恢复已 from-import 的变量、闭包、实例或容器引用；
- 原先不存在、后注入且 origin=None 的符号不会被删除；
- 不持续拦截后续其他 patch writer。

普通再次 import 不必然再次 patch，厂商有一次性保护；但不能由此推导 TT 恢复是持续禁用策略。

此前 all 档整网 PASS 至多证明测试路径不依赖已恢复的那批符号替换，不能证明不依赖任何 xmlir patch。

### 5.4 建议定义三组对照

| 模式 | xmlir 目标性能替换 | TT provider |
|---|---|---|
| 社区分解基线 | 关闭 | 旁路 |
| xmlir 优化 | 保留 | 旁路 |
| TT 优化 | 关闭 | 启用 |

三组使用相同硬件、模型、attention 后端、数据、训练配置；功能对接层始终保留，不混进性能对比。启动应输出实际绑定/恢复清单与模式，不能只依据 env 名称判断路径。

证据：`ttorch/__init__.py:42–62`、`ttorch/provider/vendor_patch.py:43–105`、`ttorch/provider/manager.py:157–180`。

## 6. 职责边界与待验证项

### 6.1 不应为了覆盖率直接搬入 TT

- DataLoader/DDP 用户配置强制；
- 强制所有模型走 SDPA；
- 整段 swift 多模态训练 forward；
- 静默绕过加载安全检查。

优化器可另行评估，但没有必要仅为一条历史 patch 新建接管通道。

### 6.2 安全检查观测尚未落地

原文建议初始化探测、警告与 provenance，当前 TT 未实现相关闭环。

“TT 不主动绕过安全护栏”与“TT 已发现并告知厂商绕过”是两回事。默认/perf 未必撤销厂商已有的安全函数 pass。应保留安全语义，不把恒真绕行作为兼容方案。

### 6.3 caching_allocator_warmup 仍需专项对照

当前社区 warmup 确实消费 `mem_get_info/memory_reserved/memory_allocated`，TT 已纳管相关符号。

但查询口径正确不等于预热算法正确。厂商 mock 明确回退到 4.57 式 sizing。此前模型加载/整网成功不证明该 patch 可退役。

应单独对比原生与厂商 warmup 的加载时间、峰值显存、实际分配/释放与 device_map 行为，而不是仅检查“能加载”。本轮未完成这项真机实验。

证据：

- 安装包 `transformers/modeling_utils.py:5025–5073`
- 安装包 `torch_xmlir/symbrewrite/plugins/transformers/mock_modeling_utils.py:67` 起

## 7. 本轮验证结果

### 7.1 现有测试

运行：`cd /root/dl/TTorch && python3 -m pytest tests/unit/ -q`

结果：`44 passed, 1 warning in 10.33s`，退出码 0。warning 为 pytest 对预加载 anyio 的重写提示。

现有测试没有覆盖下列负向情形，不能用通过结果否定独立反例。

### 7.2 隔离进程最小反例

| 检查 | 实际结果 |
|---|---|
| 正确值但 detached 候选 | gate=True/pass，梯度证据 None |
| fp32 正确、bf16 结果乘二候选 | gate=True/pass |
| 无 flag zcg 首次零输入 | 标准自检=True，后续最大误差约 1.0 |
| 错误 epsilon 的 FX 比较 | ssa_equal=True，输出最大误差约 0.0465 |
| 显式错误 flag | 正确 zcg provider 被预拒 |
| 同名/别名扫尾 | 同名已替换，别名未替换 |
| unregister | 源头恢复，消费者 wrapper 残留 |
| bind_loaded 后 install | 两层 wrapper |
| Gemma4 原始 no-scale 类 | TypeError，无法接管 |
| 非法启动 restore scope | activate 报错，但 is_active=True |
| 当前 FlashAttention 元数据/原生探测 | 版本存在，原生探测 True，无原文故障 |

这些实验以独立进程退出清理状态，没有修改项目或环境源码，也未写入新的永久测试。

### 7.3 未验证范围

- 本轮未重跑整网性能，不将历史 1.05x/1.128x 当作新鲜证据；
- 未完成 transformers text/vision/multimodal RoPE 接入及前反向矩阵；
- 未完成 Gemma4 非单位权重、低精度 vendor 专项精度；
- 未完成原生/厂商 allocator warmup 真机专项对照；
- 未完成多 rank、FSDP/ZeRO、compile 等完整兼容矩阵；
- 未认定当前每个厂商 impl 均存在数值错误；门禁反例证明的是检测漏洞。

## 8. 建议优先级与验收

| 顺序 | 工作项 | 主要验收标准 |
|---|---|---|
| 1 | 正确性底座 | detached/低精度错误候选被拒；退化输入不误接管；错误 flag 可安全纠正或明确退场；启动失败状态一致 |
| 2 | 绑定生命周期 | 同名/别名/未来 import 覆盖；关闭后未来建模不再 wrap；重复启停不叠层；状态与对象一致 |
| 3 | SSA 与对比证据口径 | 常量/连接反例不再标为结构相等；明确三模式；all 范围准确且可审计 |
| 4 | transformers 文本 RoPE | 契约/native/adapter/坐标齐备；双输出、布局、partial rotary、前反向及生产 dtype 验证 |
| 5 | 视觉/mRoPE/Gemma4 | 在文本闭环基础上逐语义扩展，不盲加坐标 |
| 6 | 框架版本与观测 | 支持窗口、接口检查、失败可见、安全检查证据与 warmup 专项结论 |
| 7 | Attention 与性能专项 | 先定义后端契约，再评估接入；分发税按同一实现测量 |

最终建议：下一步不只是加 TRANSFORMERS 坐标表。先修准入与生命周期，再扩 RoPE 契约，否则覆盖扩大将放大现有门禁和回退漏洞。

## 9. 暂停点

本文已落盘，源码仍保持 `4ba6c6c`，上述问题尚未实施修复。后续恢复处理时应先确认代码基线是否变化，再将最小反例固化为回归测试，按 §8 分批修复与验证。
