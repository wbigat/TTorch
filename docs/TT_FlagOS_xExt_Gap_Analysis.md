# TT Bridge/Provider vs FlagOS / xswift_ext / xmegatron_ext 差距分析

日期：2026-10-11。基准：tt-bridge-demo v8（Bridge 两块）+ tt_provider v2（三级门禁/supports/switch/provenance/tt_ops）。

> 注（2026-10-15）：本文的 mock `tt_provider v2` 已落地为真实生产包 `ttorch/provider`
> （两层 Provider/ProviderImpl + 自动注册/生效/选择 + 统一 provenance），首个真实
> provider（RMSNorm, KL3）实测与经验教训见 `TT_Provider_Function_Map.md` §6。
证据来源：Megatron-LM-FL 全量扫描（1165 py，重点 `megatron/plugin/`）、xswift_ext 与 xmegatron_ext 全部 60+ 文件精读。

---

## 1. FlagOS（Megatron-LM-FL）核心逻辑与主要功能点

### 1.1 两套正交机制 + 一条接缝

| 机制 | 位置 | 职责 |
|---|---|---|
| Platform 抽象层（纵向） | `megatron/plugin/platform/`（base ABC ~60 抽象方法 + 7 平台实现 + register/manager） | 替换硬件 API：`torch.cuda.X` → `cur_platform.X`；69 个 core 文件模块级 `cur_platform = get_platform()` |
| Override 注册表（横向） | `megatron/plugin/decorators.py`（541 行）+ `override_registry.py`（217 行） | 替换函数/方法/类实现：core 预埋 `@overridable`（实测 16 个文件），vendor 侧 `register(target, impl, vendor=)` 只存字符串懒加载 |
| 接缝 | `decorators.py::_get_preferred_vendor()` | vendor 默认 = `platform_manager.cur_platform.platform_name()`——"检测到什么硬件就用哪个 vendor 的 override" |

对照我们：**Bridge ≈ Platform 层，Provider ≈ Override 层**，分层判断一致；FlagOS 的接缝极小化（Provider 侧只读一个 vendor 字符串）与我们"provider 单向只读 Platform Manager"一致。

### 1.2 值得记录的关键设计点

1. **`platform_name()` vs `device_name()` 双身份分离**（`platform_base.py:40-69`）：vendor 身份（kunlunxin）与 torch 认得的设备类型（cuda）严格分开。历史坑在 `platform_kunlunxin.py:57-65`：device_name 返回品牌名 → `torch.tensor(device=...)` 抛错 + 优化器 `device.type` 比较**静默失效**。
2. **detect 优先级：native vendor > CUDA 兼容声明 > cpu**（`platform_manager.py:12-35`）：冒充 CUDA 的栈（Enflame GCU 会把 `torch.cuda.is_available()` 报成 True）下 native 优先；多 native vendor 直接 RuntimeError。
3. **懒加载注册**：`register()` 只存字符串路径，`_resolve_lazy_impl()` 首次调用才 import——不在昇腾上跑就永远不 import 昇腾库。
4. **命中/未命中双缓存**（`_plugin_impl_cache` + `_original_impl_cache`），热路径零开销。
5. **vendor 回退链**：preferred → `default` → None + warning；MACA 二次甄别（cuda 平台 + `torch.version.maca` → vendor=metax）。
6. **改动标记纪律**：`######## FlagScale Begin/End ########`（90+ 文件），fork diff 可机器识别。
7. **三套 patch 机制并存**：`@overridable`（FlagScale 式预埋接缝）+ NPU `PatchesManager`（MindSpeed 式 sys.modules 暴力扫尾，含 `torch.compile` 禁用、`Parameter.type` 字符串改写、TE 版本号伪装等对 torch 本身的补丁）+ 昆仑芯闭源 XME `megatron_plugin_init`。

### 1.3 主要功能增量清单

多硬件插件层（platform/override/4 vendor 包）；hetero 异构混训（`plugin/hetero/parallel_context.py` 700+ 行，flagcx 跨芯片通信，**同进程多芯片类型**）；DualPipeV 双流水线（`plugin/dualpipev/` 2000 行，移植自 MindSpeed，接入 `core/pipeline_parallel/schedules.py:155`）；DSA 稀疏注意力 Triton kernel（SM90/SM100 分发）；实验注意力变体（DSA/CSA/Engram）；Flextron 弹性架构；RL（`megatron/rl/` + `train_rl.py`）；post_training 增强。

### 1.4 FlagOS 的薄弱点（我们已覆盖，方向性确认）

- **无 provenance**：override 命中只有 `logger.info`，无结构化审计；
- **门禁薄弱**：只有 `is_current_platform_supported()` 一个 assert；能力查询泄漏——`transformer_config.py:1496-1545` DSA 门禁直接调 `torch.cuda.get_device_capability()` 绕过抽象；
- **无 switch**：`cur_platform` 单例选定后不可换，vendor 无运行时开关；
- **env 散落**：`MG_FL_PREFER`/`XPU=1`/`XMLIR_BATCH_PARALLEL` 等无集中注册表；
- **method_key 脆弱**：key 只取路径末两段，跨模块同名撞 key；版本/签名兼容靠 `test_kunlunxin_signature_compat.py` 手工兜底；
- **无多硬件 CI 准入**：functional_tests 全 NVIDIA 金标。

---

## 2. xswift_ext / xmegatron_ext：按我们三层模型重新归类

之前已吸收：symbol_rewriter 扫尾机制（cuda-compat `_bind`）、fused RoPE（provider v2 完整实例）。本次精读全部文件后，剩余功能归类如下。

### 2.1 属于 cuda-compat 层（兼容/正确性补丁，非性能）

| 补丁 | 出处 | 暴露的机制缺口 |
|---|---|---|
| `os_environ_wrapper`：劫持 `os.environ.get` 强制 `CUDA_DEVICE_MAX_CONNECTIONS=1` | xmegatron_ext `training/arguments.py:63` | **patch 目标不限于 vendor_cuda 模块**——标准库对象也在盘点面内 |
| `validate_global_plan_wrapper`：inspect 调用方源码判断返回 list 还是 bool | xswift_ext `dcp.py:10-18` | **契约探测**：按调用方实际契约自适应，一个 patch 兼容多个 torch 版本 |
| `conv3d_forward_wrapper`：Swift 只在 torch [2.9,2.10) 打补丁，XPU 无条件需要 | xswift_ext `conv3d.py` | **版本窗口兜底**：上游的版本判断在非 CUDA 平台不成立 |
| `wrap_model`：DDP 初始化侧流双向 fencing（上游漏 join，静默读到半写权重） | xswift_ext `megatron_lm_utils.py` | 正确性修复；证明 **behavior 类补丁需要 conformance 背书**（此类 bug 不报错只出 NaN） |
| `CudaRNGStatesTracker_wrapper`：XPU RNG 语义对齐 CUDA | xmegatron_ext `tensor_parallel/random.py` | RNG 是 conformance 必测项（之前定稿已列，未实现） |
| `_ParamAndGradBuffer._get` 的 Proteus flat-owner 标记 | xmegatron_ext `distributed/param_and_grad_buffer.py` | 厂商内存系统（Proteus）兼容钩子 |
| `FusedDispatch/Combine_wrapper`：fwd/bwd 前后 `torch.cuda.synchronize()` 关掉 DeepEP NaN 路径 | xmegatron_ext `moe/fused_a2a.py` | **硬件规避框架缺陷的暴力正确性补丁**——注意它是"加 sync 降级"而非性能优化 |
| `gather_object_cpu` / `disable_fast_fc_memory_efficient_wrapper` | xswift_ext `loss/embedding.py`、`common/utils.py` | CUDA storage 反序列化规避；评测期厂商库行为开关 |

### 2.2 属于 Provider 层（性能增量，尚未做成 provider 的）

| 实现 | 出处 | 对我们机制的印证 |
|---|---|---|
| SwiGLU 三件套（`bias_swiglu_impl`/`weighted_bias_swiglu_impl`，torch_xmlir SwiGLUFunction） | xmegatron_ext `fusions/fused_bias_swiglu.py` | 自带 supports 条件（fp8=False、shape∈{2,3}、bf16+axis=-1 fast path）；`XME_SWIGLU_USE_OKEANOS_DL` 环境开关 = switch 的 env 后端 |
| TP 线性层融合 `LinearWithGradAccumulationAndAsyncCommunication`（hydrax.xaccelerator，fwd/dgrad/wgrad 分解 + async 通信） | xmegatron_ext `tensor_parallel/layers.py` | 重量级 compute provider；ImportError 时类置 None 的懒加载形态 |
| DeepEP `get_buffer` + fused a2a | xmegatron_ext `moe/fused_a2a.py` | MoE 通信融合 provider 候选 |
| **max-seqlen pad warmup**（前几个 microbatch 补齐到最大长度，预建最大 kernel/显存 shape；MoE pad 随机化防 aux spike + GEMM 不均衡；GDN NT 对齐） | xswift_ext `trainers/base.py:255-396` | **流程级性能优化**：跨 data collator/ViT/GPTModel/router mask 四点协作，不是单算子也不是单模块 schedule——我们 Provider 模型里没有这一层（见 §3 遗漏 4） |

### 2.3 属于 checkpoint/权重布局适配（三层之外的第四领域）

`te_grouped_linear_getattr/sharded_state_dict`（XTE large_weight ↔ per-expert weight{i} 视图）、`_get_param_groups`（PP 各 stage param group 顺序对齐，分布式 checkpoint 正确性）、`_load_base_checkpoint`/`generate_state_dict`、bridge conversion 全套（streaming 权重、conversion 内存控制、modelopt mock）。**这些不是 runtime API 也不是性能 provider**，应归入之前定稿的"模型 conformance"层的权重布局子域，或独立的 conversion 工作流。

### 2.4 机制层亮点（xswift_ext `entry.py`/`auto_init.py`/`envs.py`）

1. **wrapper 链**：`is_wrapper=True` 包裹原实现而非替换，且单符号可叠多个 wrapper（`_wrappers.extend`）——大量补丁是"增强"（init 注入 callback、padding_mask 传递、stream fencing）不是"替换"。
2. **框架版本窗口**：`SUPPORTED_SWIFT_VERSION_SERIES="4.3.x"` + drift warning（warn-not-block），rewrite spec 带 `version_checker`。
3. **lazy init hook**：劫持 `builtins.__import__`/`importlib.import_module`，在 swift/mcore_bridge 首次 import 前触发 patch；线程安全（Condition + owner 防重入）；pip 等工具进程豁免。
4. **init summary**：rank0-only 打印 enabled/skipped specs + env 值 + rewritten symbols——初始化期 provenance。
5. **envs.py 集中注册**：EnvSpec 表 + init_envs 一次读入 + 模块代理属性访问 + 未初始化访问报错。

---

## 3. 对照后我们设计的遗漏清单（按优先级）

### P0 —— 真实需求已被两处代码证明

1. **cuda-compat 缺 WRAP 决策（仅正确性 wrapper）**。当前只有 remap/fallback/exception 三种替换式决策；xswift 里一半补丁是 wrapper（增强/围栏/注入），且支持一符号多 wrapper 叠加。→ `Decision` 增加 `WRAP`，Entry 增加 `wraps` 列表（有序叠加），绑定后仍留 `_origin_` 痕。**注意边界**：这只承载正确性 wrapper；性能目的的 wrapper（JIT 包装原生小算子链）属 Provider 的 wrap_native 构建策略，已落地 demo v3——见 §5 二维模型。
2. **Platform 双身份：`platform_name()`（vendor 身份）与 `device_name()`（torch 设备类型）分离**。冒充 CUDA 的栈（torch_xmlir/MACA/GCU）会让 CUDA detect 误命中；`device_name` 写错品牌名会导致 `torch.tensor(device=)` 报错、优化器 `device.type` 比较静默失效（FlagOS 有注释在案的真实坑）。→ **已落地（demo v9）**：`PlatformHandle` 增加 `vendor_name`/`device_name`/`masquerades_cuda` 类属性（双身份由 spec 承载，实现类不许自报，防冒充）；`auto_activate` 中 native 命中把被冒充的 compat claim 降级为证据；red 用例证明无 native adapter 时冒充必然得逞。
3. **盘点面扩展**：patch 目标不止 vendor_cuda 模块函数——`os.environ.get`、`torch.nn.Conv3d.forward`、`torch.distributed.checkpoint` 内部函数都在列。default-deny 的覆盖面定义要从"vendor_cuda.__all__"扩为"框架用到的全部硬件相关符号表"。

### P1 —— 工程化强度

4. **流程级性能 provider 缺位**。max-seqlen pad warmup 这类优化跨 data collator/模型 forward/router mask 多点协作，不是模块语义 + compute/schedule 插槽能表达的。→ 需要第三种 provider 形态：**workflow/hook 型 provider**（注册到训练流程的钩子点集合，统一门禁与 switch）。建议先只建概念与注册位，不急着实现。
5. **框架版本协商**。Platform Manager 有 SDK 版本门禁；policy/rewrite 侧缺框架版本窗口（swift 4.3.x / mcore_bridge 1.5.x）+ drift warning（warn-not-block）。→ Entry 增加 `framework_versions` 适用窗口字段。
6. **集中 env registry**。switch 需要环境变量后端；EnvSpec 表 + init_envs + 未初始化报错是值得照搬的形态（FlagOS 恰好缺这个，散落各处）。
7. **lazy init hook**（在框架首次 import 前自动触发 patch）+ 线程安全单次初始化 + rank0-only init summary。我们的 `cuda_compat.activate()` 目前是显式调用，真实接入框架时需要这个时机机制。

### P2 —— 记录为已知边界，暂不实现

8. **懒加载 vs code-first 的张力**：FlagOS register 只存字符串路径（不在该平台就绝不 import 其库）；我们 code-first 真实对象引用 import 期即解析（拼写即炸），代价是全部 vendor 实现 eager import。折中候选：provider impl 支持惰性工厂（`impl_factory: Callable[[], Callable]`），门禁期才解析。
   **platform 侧已落地（2026-10-12；现 `spec.py`）**：按"管理面 eager / 适配面 lazy"切开——detect 与身份元数据保持真实引用（code-first 优势不丢），只有实现本体降级为 `impl="module:Class"` 字符串，`_activate()` 才 import。实测佐证了 FlagOS 这条设计的必要性：本机 `import torch_xmlir` 即触发 SYMBOL_REWRITE 与 libbkcl 加载，eager import 全部 vendor adapter 在异构机群上不可接受。另有一条 FlagOS 未记录的收益：PrivateUse1 的 `torch.xpu` 是厂商包运行时注册的属性，类属性 import 期求值必拿 None，惰性是正确性要求而不只是性能优化。**provider 侧仍待办**（见 §4 行动顺序）。
9. **异构混训（hetero）与"进程级单 platform"假设冲突**。FlagOS hetero 支持同进程多芯片类型（flagcx）；我们 auto_activate 多命中 fail fast。文档明记：若未来做异构混训，该假设要打破。
10. **契约探测技巧**（inspect 调用方源码判断契约形态，dcp.py 实例）——用于一个 patch 兼容多个上游版本。技巧保留，遇具体案例再用。
11. **RNG conformance 测试**未实现（定稿已列）；wrap_model 类"不报错只出 NaN"的补丁证明 behavior 类必须配 conformance 背书。

### 我们独有、FlagOS/xExt 都没有的（方向性确认）

provenance 全量审计（gate/select/call/switch + fallback_actual）；三级准入门禁（功能/精度/收益）；switch 运行时控制 + reference 不可禁用；supports 逐调用适用性谓词；conformance 激活门禁（接口不支持抛 unsupported、可选能力实测派生）；集中 env registry（FlagOS 也缺）。

---

## 4. 建议行动顺序

1. cuda-compat 增加 WRAP 决策 + wrapper 链（P0-1），demo 用 wrap_model 式围栏补丁验证；
2. PlatformHandle 双身份 + detect 优先级（P0-2），demo 加一个"冒充 CUDA 的厂商栈"用例（detect cuda=True 但 native vendor 优先）；
3. Entry 增加框架版本窗口 + drift warning（P1-5）；
4. env registry（P1-6）作为 switch 的环境变量后端；
5. workflow 型 provider 概念与注册位（P1-4），先设计评审后实现。

---

## 5. wrapper 的双归属：目的 × 形态二维模型（用户纠偏后定稿）

### 5.1 纠偏：归属判据是"目的"，不是"wrapper 形态"

初版把 WRAP 整体塞进 cuda-compat 是不对的。wrapper 只是一种**实现形态**（保留原实现、只表达 delta）；归属由已定稿的判据决定——**摘掉后的后果**：

| | cuda-compat 的 WRAP（第四种决策） | Provider 的 wrap_native（impl 构建策略） |
|---|---|---|
| 目的 | 正确性/兼容性 delta：原实现必须跑，补丁在其前后/参数上加东西 | 性能增量：包装社区原生小算子链，JIT/AOT 编译融合 |
| 摘掉后果 | 会错 / NaN / 不兼容 → 必须在 compat 层 | 只是变慢（回到 reference 分解链）→ 必须在 Provider |
| 典型形态 | 原函数 + 围栏/回调/参数适配 | `jit_compile(native_chain)`：链包装 + 编译缓存 |
| 验证重心 | conformance 背书（不报错只出 NaN 的补丁必须激活期拦） | 收益门禁（数值语义继承原生链，精度门禁天然对齐） |
| remap/fallback 的关系 | 并列的决策枚举（remap/fallback/exception/**wrap**） | 无关——Provider 不做符号替换决策，只做实现选择 |

### 5.2 cuda-compat 侧：WRAP 决策的需求场景（正确性 delta）

这些案例的原实现**完全能跑且必须跑**，remap 硬套会复制-漂移、不可叠加：

| 场景类型 | 真实案例 | wrapper 做什么 |
|---|---|---|
| **注入 callback** | `init_wrapper`（xswift `megatron_lm_utils.py`） | 原 init 执行完后触发 XME 侧初始化 |
| **执行围栏** | `wrap_model`：DDP 侧流 broadcast 前后 event wait | 加顺序约束，不改原行为 |
| **参数透传/改写** | `padding_mask` 传递补丁 | 只适配签名窗口，主体仍调原函数 |
| **降级性包裹** | `FusedDispatch/Combine_wrapper`（xmegatron `fused_a2a.py`）加 sync 关 DeepEP NaN 路径 | 牺牲性能换正确性——注意方向是"降级"，不是优化 |
| **契约自适应** | `validate_global_plan_wrapper`（xswift `dcp.py`） | inspect 调用方决定返回形态 |
| **环境切面** | `os_environ_wrapper`（xmegatron `arguments.py:63`） | 只改特定 key 的返回值，其余透传 |

机制要求（超出"加一种 Decision 枚举"本身）：**有序叠加**（wraps 列表，应用顺序=注册顺序）；**原实现可达**（`_origin_` 记整条链）；**类别不变**（包多少层仍是原三类归属）；**单层可摘除**（switch/调试用）。

### 5.3 Provider 侧：wrap_native 构建策略（性能 wrapper）

社区原生接口性能差的典型形态不是单算子慢，而是**分解链**：每个小算子独立 kernel、中间结果逐步物化（如 RMSNorm 的 `square→mean→rsqrt→mul→mul`）。性能优化手段是把整条链 wrap 起来编译融合——这就是"Provider 里的 wrapper"：

- `Provider.build = "self_contained" | "wrap_native"`（与 kind 正交）：厂商 AOT 黑盒是 self_contained；`jit_compile(native_chain)` 产物是 wrap_native；
- `Provider.wraps` 声明被包装的原生算子链，进 gate/select provenance（可审计"这个性能来自对哪条链的融合"）；
- 门禁含义不对称：wrap_native 数值语义继承原生链，**精度门禁天然过，收益门禁是主过滤器**；self_contained 自研/厂商 kernel 相反，精度门禁是主过滤器；
- 低精度编译（fp16 截断类）是 wrap_native 自己的责任——demo 里 `fused-jit` 就是这个反例，精度门禁拒掉它。

已落地（demo v3）：`implementations.py::jit_compile()` 包装器 + `_rmsnorm_native_chain` 原生链；`fused-jit-wrap` provider 过门禁后在 vendor/ABO 被禁用时接管流量，结果与 reference 逐位一致。

### 5.4 两层交界处：性能 wrapper 的"挂载点"

还有一个容易混淆的点：对**不受我们控制的框架**（Megatron/LMF），性能 wrapper 的挂载动作（把框架的 `apply_rotary_pos_emb` 指向 tt_ops 语义入口）用的是 cuda-compat 的绑定机制（remap 到语义 shim）；但挂载点后面的性能实现本体在 Provider。即：**cuda-compat 提供拦截机制，Provider 提供被拦截后的实现**——xswift 的 `apply_rotary_pos_emb_bshd_wrapper` 裸替换 = 两者没分层时的混沌形态。

---

## 6. 盘点面扩展详解：default-deny 要 cover 什么

### 6.1 问题定义

cuda-compat 当前的盘点面 = "vendor_cuda 模块的全部公开符号"，default-deny 拦截未盘点符号。这个模型隐含假设：**框架的硬件痕迹集中在一个 vendor 模块里**。xswift/xmegatron 的全部补丁证明这个假设不成立——硬件痕迹散布在至少四类目标上，每类的盘点方式、归属判定、default-deny 形态都不同。

### 6.2 四类 patch 目标（按盘点难度递增）

**A 类：vendor 设备模块**（当前已覆盖）
- 形态：`vendor_cuda.synchronize` / `torch.cuda.*` / `torch.accelerator.*`
- 盘点方式：模块 `__all__` + 显式符号表；from-import 扫尾按 `id()` 匹配
- 现状：demo 已验证，9 条 policy 全覆盖

**B 类：标准库/通用库的环境切面**
- 实例：`os.environ.get`（xmegatron 强制 `CUDA_DEVICE_MAX_CONNECTIONS=1`）；`os.environ.setdefault`；`socket`/`datetime` 暂无但同类
- 特殊性：**全局单例对象的方法**，patch 后影响进程内所有调用方（包括框架之外的库），爆炸半径远大于 A 类
- 盘点要求：Entry 必须带 **key 级白名单**——wrapper 只许改指定 key 的行为，其余 key 必须原样透传（xmegatron 的实现就是这么做的）；admission 测试要验证"非目标 key 行为不变"
- default-deny 形态：不能对整个 `os.environ.get` deny（框架正常读环境变量不该被拦），只对**被注册的 key**做策略裁决，未注册 key 透传——default-deny 的粒度从"符号"细化到"符号 × 关键参数值"

**C 类：torch/框架内部函数（非设备 API）**
- 实例：`torch.nn.Conv3d.forward`（xswift conv3d.py：上游只在 torch [2.9,2.10) 打补丁，XPU 无条件需要）；`torch.distributed.checkpoint` 的 `validate_global_plan`/`_load_base_checkpoint`；DCP filesystem 策略内部函数
- 特殊性：这些是**上游版本敏感**目标——上游下个版本可能自己修了，补丁从"必需"变"有害"。必须配版本窗口（P1-5 的 `framework_versions` 字段）：窗口外自动跳过 + drift warning
- 盘点要求：Entry 增加"上游状态"注解（`upstream_fixed_in` / `upstream_issue`），CI 在上游版本升级时提醒复核
- default-deny 形态：维持符号级 deny，但注册表要按"上游版本 × 平台"二维索引——同一个 `Conv3d.forward`，torch 2.9+CUDA 不打、torch 2.9+XPU 打、torch 2.10+任何平台都不打

**D 类：调用现场契约（最隐蔽）**
- 实例：`validate_global_plan_wrapper` 用 inspect 读调用方源码决定返回类型；`gather_object_cpu` 按序列化来源决定 storage 位置
- 特殊性：补丁的正确性依赖**调用方**的契约形态而非被调函数本身；同一函数被不同版本框架调用时要求不同行为
- 盘点要求：这类补丁必须在 Entry 里显式声明"我依赖调用方契约"（`caller_contract=True`），admission 测试要覆盖全部已知调用方形态；新调用方出现时应 fail fast 而非猜测

### 6.3 盘点面的组织形态结论

单一"vendor 模块符号表"扩为**四张分表 + 统一注册**：

| 分表 | 索引键 | default-deny 粒度 | 必备字段 |
|---|---|---|---|
| A. vendor 设备符号 | 模块.符号 | 符号级 | 三类归属 + 决策 |
| B. 环境切面 | 模块.符号 × key | 符号 × 参数值 | key 白名单 + 透传承诺 |
| C. 框架内部函数 | 符号 × 上游版本 × 平台 | 符号级（版本窗口内） | 版本窗口 + 上游状态注解 |
| D. 调用现场契约 | 符号 × 调用方形态 | 符号级 + 调用方枚举 | caller_contract 声明 + 调用方清单 |

统一进 `binding_register` 一张注册表（Entry 增加 `surface` 字段区分分表），导出 YAML 快照按分表分节——CI diff 时能一眼看出"这次改动碰了哪类目标"，B/D 类的 diff 评审强度应高于 A/C。

---

### 6.4 与"平台维度"的交叉

四张分表都要带平台维度（policy 按 platform overlay）：同一个 `Conv3d.forward`，CUDA 平台无补丁、XPU 平台 WRAP；同一个环境 key，KL3 强制 "1"、CUDA 透传。这正好落在之前定稿的"policy 多框架 overlay"遗留项上——分表 × 平台 × 框架版本构成 policy 的完整索引空间，demo 目前只实现了 A 表 × 单平台 × 无版本窗口的最小切片。

---

## 7. schedule provider 场景盘点（真实代码佐证）

时序优化不改算子结果、只改执行时序，是三框架中真实存在且反复出现的一类性能手段。按调度范围分两档：

### 7.1 模块内 schedule（归模块 Provider 的 schedule 插槽）

| 场景 | 真实出处 | 时序手法 | demo 对应 |
|---|---|---|---|
| **TP 通算 overlap** | xmegatron_ext `tensor_parallel/layers.py` `LinearWithGradAccumulationAndAsyncCommunication`（hydrax.xaccelerator） | forward：all-gather 异步化与本地 matmul 交叠；backward：wgrad/dgrad 分解，wgrad 不等通信 | `tp_linear` 实例（demo v4）：串行 reference vs `overlap_tp_schedule`（通信上 comm stream、计算先行、末尾统一等） |
| **MoE dispatch/combine 与计算 overlap** | xmegatron_ext `moe/fused_a2a.py`（DeepEP fused a2a 的异步收发能力） | dispatch 异步在途时先算 shared expert / 本地 expert | `moe_block` 实例（demo v1 起）：serial vs overlap |
| **梯度 bucket 通信与反向计算 overlap** | Megatron DDP `param_and_grad_buffer.py`（xmegatron_ext 有 Proteus 适配痕迹） | grad bucket 满即异步 all-reduce，不等整网反传完 | 候选，未实现（语义同构于 tp_linear，优先级低） |

共同判据：这类优化**摘掉只是变慢**（回退 serial 仍正确），且消费 Platform Manager 的异步原语（stream/event）——正是 provider → bridge 单向只读依赖的合法用法。

### 7.2 跨模块 schedule（RegionScheduleProvider 的领域，先建概念不实现）

| 场景 | 真实出处 | 为什么超出现有插槽 |
|---|---|---|
| **PP 双流水线 DualPipeV** | FlagOS（Megatron-LM-FL） | 调度单位是"流水线 stage 的 fwd/bwd 块"，跨越多个模块语义；需要全局流水视图，不是单模块内的时序安排 |
| **max-seqlen pad warmup** | xswift_ext `trainers/base.py:255-396` | 跨 data collator/ViT/GPTModel/router mask 四点协作——之前归类为"流程级 provider"（§3 遗漏 4），本质也是 region 级调度的一种 |

结论与行动：模块内 schedule 已有 demo 双实例（moe_block + tp_linear）覆盖机制验证；region 级（DualPipeV/warmup）**只建概念与注册位**，待模块内形态稳定后再设计——它要求 schedule plan 能引用多个模块语义的边界，是会改变 Provider 核心数据结构的决策，不宜提前铺开。

### 7.3 与 cuda-compat 的边界再确认

`fused_a2a.py` 里 fwd/bwd 前后加 `torch.cuda.synchronize()` 关 DeepEP NaN 路径——形态上也是"动时序"，但目的是**正确性**（摘掉会 NaN），归 cuda-compat WRAP（§5.2 降级性包裹）。schedule provider 的判别铁律不变：摘掉只是变慢。
