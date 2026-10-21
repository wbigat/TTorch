# TT Bridge 深入分析

> 本文聚焦**机制层**：Bridge 到底怎么工作、难点在哪、失败怎么办。  
> 功能组成见 `TT_Bridge_Provider_Architecture.md` 第 4 章，性能侧见 `TT_Provider_Function_Map.md`。

---

## 0. Bridge 的两大块与依赖方向（2026-10-10 二次定稿）

Bridge 由两大块组成，**组织上同属 Bridge**：

| 模块               | 职责                                                                          | 变更频率          |
| ---------------- | --------------------------------------------------------------------------- | ------------- |
| `cuda_compat`    | 框架 CUDA 调用点的处理策略：**remap / fallback / exception**，含准入测试与 trace              | 随框架上游漂移，最高    |
| Platform Manager | 多 platform 注册/选择/构建/probe + 每平台 adapter（填非标接口 gap）；**统一 runtime 接口是它的顺带产物** | 随硬件/SDK 演进，中等 |

> **统一接口不再是独立模块**（2026-10-10 修正，替代之前"三大块"的划分）：  
> 原 TT Universal API 没有任何独立语义内容——契约由 `PlatformHandle` 接口定义，  
> 分发是 `select()` 之后一行转发的机械门面。所以它作为**概念/契约**存在（见 §3），  
> 作为**代码**并入 Platform Manager：`PlatformHandle`（契约本体）+ 模块级门面函数（分发器）。  
> 这也理顺了 accelerator 代理路径：官方已支持的符号，policy 的 remap target 以后可直接指  
> `torch.accelerator.*`，连门面都绕过——门面只为扩展位存在，更不该独立成模块。

执行链路：

```text
框架 CUDA 调用点（不改框架源码，运行时重绑）
   → cuda_compat 逐符号决策
       ├─ remap      → interface 转发函数 → 激活的 platform 实现 → 厂商接口
       ├─ fallback   → torch-native 等价路径（不经过门面）
       └─ exception  → 明确报错
```

**要守的是依赖方向，不是包归属**：

- Provider 可依赖 Platform Manager（门面函数、接口存在性内省、stream/event 契约）；
- Provider **不依赖** `cuda_compat`（不碰框架调用点替换）；
- `cuda_compat` **不依赖** Provider（功能判定里没有"等 Provider 来救"）。

四个边界条件：

1. **remap 只对 Python 层可拦截调用点有效**；C++ 扩展（TE/apex/flash-attn）与 codegen（inductor/triton）没有可 remap 的调用点，只能整体 fallback 或 exception。
2. **统一接口取交集、不收算子**——框架的张量计算走 ATen dispatcher → 厂商 PrivateUse1 后端，本就不在 cuda_compat 覆盖范围内。门面只装 runtime 语义：device / stream / event / memory / RNG / profiler + c10d 入口。
3. **Platform Manager 的管理面与适配面内部拆开**：管理面 = 注册/选择/构建/probe/生命周期；适配面 = 每平台 adapter 实现契约接口、填非标 gap。门面是薄分发层，不含硬件代码——新增硬件 = 新增一个 adapter，不改门面、不改 cuda_compat policy。
4. **"向上对接框架"的是 cuda_compat**（每框架一份调用点清单 + 绑定）；框架不 import TT API，否则会滑成"要求框架适配我们"，失去无侵入性。

---

## 1. Bridge 本体：框架侧适配的四个真难点

### 1.1 seam 稳定性决定上游跟随成本

每个框架能挂 Bridge 的位置不同，稳定性差异极大：

| 框架                   | 可用 seam                                    | 稳定性          |
| -------------------- | ------------------------------------------ | ------------ |
| MCore                | `ModuleSpec` / Builder / TransformerConfig | 较好，官方扩展点     |
| Swift / Transformers | 模块替换规则、trainer 回调                          | 中等，部分需 patch |
| DeepSpeed            | engine / optimizer seam                    | 中等           |
| verl                 | Model Engine seam                          | 视版本          |

**adapters 的稳定性 = 上游跟随成本。**&#x6BCF;个 adapter 应记录：seam 类型（官方扩展点 / 稳定 hook / monkey patch）、上游版本范围、失效检测方式。monkey patch 类 seam 要单独标红——它是最先被上游改动打碎的部分。

### 1.2 激活顺序与幂等

Bridge 必须在 **framework import / model build 之前**激活。多框架组合（Swift-Megatron）时顺序更复杂。要求：

- 激活**幂等**（重复调用不叠加 patch）；
- 激活顺序**显式声明**（谁先谁后有定义，不是靠 import 顺序碰运气）；
- 激活失败 → fail fast，不允许"部分激活后继续跑"。

### 1.3 feature gate 的粒度

不是"全局关掉某个能力"就够了。需要至少三层粒度：全局 / 按能力类（如 graph capture）/ 按模块路径。且被禁用的能力必须进 diagnostics——**训练跑得慢时，第一件事是看哪些能力被 gate 掉了，而不是猜**。

### 1.4 checkpoint bridge 容易被低估

它不只是"存盘读盘"：state schema 校验、RNG 状态、并行布局变化时的 reshard、resume 后 loss 连续性。这是功能验收里最容易在后期才暴露的一环，v0 就该有。

---

## 2. `cuda_compat`：Bridge 里最硬的子模块

### 2.1 盘点能覆盖什么，盲区在哪

| 来源                             | 能否静态盘点 | 处理                          |
| ------------------------------ | ------ | --------------------------- |
| Python 层显式 `torch.cuda.*`      | ✅      | 主力战场                        |
| 动态 `getattr(torch.cuda, name)` | ⚠️ 部分  | 需运行时覆盖检测兜底                  |
| C++ 扩展内部 CUDA 调用               | ❌      | 只能整体替换该扩展或 reject           |
| 第三方库（TE / apex / flash-attn）   | ❌      | 走 fallback 或禁用，不进 Bridge 模拟 |

**要承认边界**：Bridge 覆盖的是 Python 层显式痕迹。C++ 扩展不是"兼容"问题，是"替换或拒绝"的二选一。

### 2.2 policy 表是核心资产——但必须是"经过校验的数据"，不是代码里的大字典

每条记录是一个符号的完整决策，不是一行映射：

```yaml
- symbol: torch.cuda.set_device
  intent: 把当前进程绑定到一张加速设备
  decision: remap                      # remap / fallback / exception
  target: set_device                   # interface 的转发函数
  test_id: compat.test.device.set_device_equivalence
  framework_range: "swift>=3.0"
  notes: 仅覆盖"选择当前设备"语义，不含可见性过滤
```

`intent` 字段最关键——**没有 intent 记录的映射，半年后没人知道当初为什么这么绑。**

**载体演进：从"字符串数据 + 校验器"到"真实对象引用 + 显式注册"**（2026-10-10 v5 demo 落地，`ttorch/bridge/binding_registry.py` + `examples/bridge-demo/tt_bridge/bindings.py`）：

v4 的 YAML+校验器是在弥补字符串引用的缺陷，v5 让大部分缺陷**不发生**：

| 脆弱点              | v4（YAML+校验器） | v5（code-first 注册表，定稿）                                                       |
| ---------------- | ------------ | --------------------------------------------------------------------------- |
| 引用拼写             | load 期被校验器拦  | **import 期 AttributeError / IDE 实时**（`symbol=vendor_cuda.set_device` 是真实对象） |
| 枚举非法             | load 期拦      | 枚举类型，书写期就无法表达                                                               |
| test 悬空 id       | load 期查注册表   | **真实函数引用**，不存在即 NameError                                                   |
| policy/实现/测试三处分离 | 仍在三个文件       | **同处 `bindings.py`**，评审不用跳文件                                                |
| 忘注册              | —            | 默认拒绝兜底（未注册符号绑 exception）                                                    |
| 可评审 diff         | 手改 YAML      | **YAML 降级为生成物**：`export_yaml()` 快照，CI diff 必须一致，永远与代码同步                     |

保留下来的机制不变：显式 `register()`（不用装饰器——数据流显式，且"忘注册"有默认拒绝兜底）、`__post_init__` 形态一致性检查（remap 不许免检）、默认拒绝 + coverage 对账、冻结（frozen dataclass）。

**唯一新增的设计约束**：`symbol=vendor_cuda.set_device` 要求 policy 模块 import 目标模块。真实场景中 `torch.cuda` 很轻，但若框架的 CUDA 痕迹藏在重量级模块里，policy 加载时机要与 bootstrap 顺序协调。

### 2.3 准入测试的判据模板

"跑起来不报错"不是等价性证明。按符号类别给可观测判据：

| 类别             | 等价判据（必须可观测）                                     |
| -------------- | ----------------------------------------------- |
| 设备选择           | 绑定后 `current_device` 一致；新建 tensor 的 device 一致   |
| 设备查询           | `device_count` / `is_available` 与 platform 实测一致 |
| 同步             | **等待范围**一致（device-wide vs stream）；并发语义不被破坏      |
| stream / event | stream 语义、event 记录与等待语义一致                       |
| RNG            | 相同 seed 下序列一致                                   |
| 内存查询           | 数值口径一致（总量/已用/峰值口径需明确）                           |

`synchronize` 是最危险的一类：同步范围不对，会**静默杀掉 overlap**，而且现象是"变慢"而不是"报错"。

### 2.4 绑定机制

- 导入期**一次性绑定**，绑定后不可变；
- 热路径不做 `if` 判断；
- 绑定动作本身要可枚举（不能散落在各 adapter 里偷偷 patch）。

### 2.5 决策三类（注意：没有"交给 Provider"）

| 决策                   | 行为                                                        |
| -------------------- | --------------------------------------------------------- |
| `remap`              | 已证明等价，改绑到 interface 的转发函数（统一转发到激活的 platform 实现） |
| `fallback`（含 forbid） | 放弃该能力，走 torch-native 等价路径或显式禁用，**不经过门面**                  |
| `exception`          | 无等价且必需 → 明确报错，不静默伪装                                       |

**准入测试没过闸的符号，要么 fallback 要么 exception，不允许"等 Provider 来救"。**

### 2.6 trace 与 Provider 侧统一 schema

记录：符号、决策、target、测试结果、实际生效路径。与 Provider provenance 合并成一份报告——回答"这次训练用了哪些替身和加速"。

---

## 3. 统一 runtime 接口：契约层（非独立模块，Provider 也消费）

> 本章内容是**契约**——接口的语义约定与分类规则。代码形态上它已并入  
> Platform Manager（`PlatformHandle` + 门面函数），见 §0。

### 3.1 取交集 + 扩展协议，避免膨胀

- **核心契约** = 所有 platform 必须实现（取交集）
- **扩展协议** = 可选能力，由 Provider 消费（是否存在由实测/内禀判定，不经平台自报）

不这么做，"universal"会膨胀成谁都实现不全的接口。

### 3.2 社区未支持 API 的处理（2026-10-10 定稿）

对 `torch.accelerator` 尚未覆盖的符号，三层判定：

1. **语义可跨硬件等价？** 否（CUDAGraph、`current_blas_handle` 等 CUDA 专属）→ fallback / exception，此为终态，与社区进度无关。
2. **`torch.accelerator` 已支持？** 是 → remap 目标直接代理 accelerator，零自造。
3. **未支持 → 两个二元问题分类**（2026-10-10 修正：不靠"能力/行为"的定义感觉，靠可操作判定；实为三类）：
   | 判定问题                                   | 是                      | 否                            |
   | -------------------------------------- | ---------------------- | ---------------------------- |
   | Q1 调用有副作用？（改设备/runtime 状态、入队工作、影响后续执行） | **行为接口** → 统一门面 / 扩展位  | 进 Q2                         |
   | Q2 答案在进程生命周期内会变？                       | **状态查询** → 门面只读接口，不许缓存 | **静态事实** → 也按只读接口处理 |
   - 状态查询 API：只读但动态（`memory_allocated`、`mem_get_info`），热路径实时读；
   - 行为 API：有副作用，走门面分发到激活的 platform 实现；
   - 静态事实（`is_available`/`device_count` 等）：查询本身廉价。2026-10-14 起
     不再为它们单设 capability 缓存表，与状态查询同为普通只读接口——
     不值得为廉价查询建一套 probe 期填充 + 校验的机制。
   - **平局规则**：分不清时默认归 runtime API——动态查询错当静态缓存是正确性 bug（读到过期值），静态事实错当 API 只是多一次调用，代价不对称，倒向安全侧。

**处置原则（2026-10-10 定稿）：社区的不确定性不进入设计。** 所有扩展位按"上游永远不会支持"建设——契约冻结、版本化、准入测试齐全，是 TT 的永久资产；上游哪天支持了，只是 policy `target` 改一行的免费替换项，不是设计依赖。`upstream_status` 降级为 policy 表上的备注字段，不参与任何决策。

**三类接口在社区未支持时的处置**：

| 类别             | 社区未支持时怎么办                                                                                                                  | 对上游的免疫力     |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------- |
| **状态查询**       | 开只读扩展位，契约**写死口径**（如显存统计含不含 cache/碎片），各 adapter 实现对齐口径，准入测试验证口径一致性；调用方不许缓存                                                  | 免疫：契约 TT 冻结 |
| **行为**         | 开扩展位，契约写清副作用范围（如 synchronize 的 device-wide 等待范围），adapter 填 gap，准入测试验证副作用语义                                                 | 免疫：契约 TT 冻结 |

**扩展位三条规则**（修订版）：

| 规则               | 内容                                        |
| ---------------- | ----------------------------------------- |
| 按永久资产建设          | 契约文档 + 版本化 + 准入测试，假设上游永远不会有；不留"等上游"的半成品状态 |
| 风格对齐 accelerator | 命名/参数风格参考官方规范，降低未来**若**切换的成本——这只是习惯，不是等待  |
| 测试照跑             | 扩展位不是免检区                                  |

**元规则**：每个扩展位都必须能写出所有 platform 适用的契约文档；写不出来说明语义不通用，退回第一层。扩展位数量应为个位数——持续膨胀说明在重造 accelerator，方向错了。

### 3.3 分成三类的意义

> **2026-10-14 修订**：三类已简化为两类 runtime 接口（行为 / 状态查询）——
> 原"capability"一类作为**交付机制**被移除：静态事实（is_available/device_count）
> 改普通只读接口（查询廉价，不值得缓存表）；可选能力（graph 捕获/异步通信）改
> conformance 无条件实测、结果即事实，不再"平台自报 + 验证自报"。
> 行为 vs 状态查询的二分仍是写 adapter 时的有效思考框架（决定契约写法与
> conformance 测试形态），但它现在是 `PlatformBase` docstring 里的接入指引，
> 不再是绑定表里的 `Category` 枚举字段。下面保留原三类推理作演进记录。

1. **生命周期与变更频率不同**：capability 随硬件/SDK 演进（最慢）；状态查询契约稳定但值动态；行为接口涉及并发/正确性语义，最易踩坑。混在一起管，就会把静态事实当行为绑定、把动态查询当事实缓存——两类典型 bug 的温床。
2. **验证方法不同**：capability 验"实测与声明一致"；状态查询验"口径一致"（与原生对账）；行为验"语义等价"（副作用/等待范围）。一套准入测试模板套不了三类——**分类决定测试形态**。
3. **消费者与性能预算不同**：capability 在选择期消费、可缓存；状态查询可能进低频热路径（显存水位检查）；行为在热路径上、绑定后零判断。**分类决定绑定策略**。
4. **错误成本不对称**：动态查询错当 capability 是正确性 bug（读到过期值），静态事实错当 API 只是多一次调用——这是平局规则倒向 runtime API 的依据。
5. **对上游依赖的免疫力不同**：capability 天然免疫；状态查询/行为按永久资产建扩展位后同样免疫——"社区是否支持"这个不确定项由此被彻底移出设计决策链。

### 3.4 能用官方的就用官方的

`torch.accelerator` 已覆盖的，remap 目标直接代理，不重复造；未覆盖的按 §3.2 建设为 TT 自有永久资产。代理是省事，自建是常态，两者都不依赖上游 roadmap。

### 3.5 init 期绑定的功能接口

RNG、allocator、profiler、graph capture、pin memory 等**功能性质接口必须 init 期绑定**——热路径分派代价高，且它们的选择不随输入变化。

### 3.6 契约版本化

Platform 声明支持的契约版本；契约变更需全平台回归。

---

## 4. Platform Manager：Bridge 两块之二（Provider 读其接口与门面）

### 4.1 构建期与启动期严格分离

| 阶段          | 职责                                                 |
| ----------- | -------------------------------------------------- |
| Bazel / 编译期 | 确定哪些 SDK、ABI、CCL、Platform 产物能安全链接                  |
| 进程启动期       | 在**已构建**的候选中自动检测激活（见 §4.1a），初始化 runtime、注册 backend |

运行时不做 dlopen 查找——找不到就是构建配置错了，不是运行时的锅。

### 4.1a 激活方式：自动检测 + 证据打印 + override 逃生通道（2026-10-10 三次定稿）

> **管理面/适配面分文件 + 实现惰性导入（2026-10-12 落地；现 `spec.py` + `adapters/`）**：
> register 登记的是轻量 `PlatformSpec`（身份元数据 + `detect` + 实现模块路径 `impl`），
> 实现本体由 `_activate()` 在平台确定后才 import。两条硬理由：
> ① **厂商包 import 有副作用**——KL3 的 `torch_xmlir` 一经 import 即执行
> SYMBOL_REWRITE（torch/deepspeed/transformers）并加载 libbkcl/XCCL，在非该硬件的
> 机器上轻则污染进程重则 ImportError；② **PrivateUse1 栈的设备模块是运行时注册的**
> （`_register_device_module`），`mod = getattr(torch, "xpu", None)` 作为类属性在
> import 期求值必然拿到 None，故 adapter 改持 `mod_path` 字符串、激活期解析。
> 由此 `spec.py` 有一条硬约束：**detect 只许 `find_spec`（查找不导入）/
> torch 自带 API / 设备节点探测，禁止 import 任何厂商包**——这正是"detect 只看不
> 初始化"在代码组织上的落地。detect 保持真实函数引用（code-first 不丢），只有实现
> 本体降级为字符串路径。

Platform Manager 管理**所有 platform 类型的逻辑注册**（register 只登记元数据，不初始化、不导入实现），启动期**自动分析当前环境**决定激活哪一个：

- 每个 adapter 实现 `detect()` 存在性探测（查驱动/SO/设备节点，**只看不初始化**），返回 `(present, evidence)`；
- `auto_activate()` 对全部候选逐一探测并**逐条打印证据**，恰好一个命中 → 激活；零命中 → fail fast 并列出探测报告；**多命中 → fail fast 拒绝猜测**（进程级单 platform 假设被破坏，必须人工介入）；
- `select(name)` 保留为调试/测试的显式 override 通道，打印 `via=manual-override`；
- 测试夹具/变体 adapter 置 `auto_detect = False`，不参与自动检测（demo 的 `NaiveXpuPlatform` 即如此）。

与"不要环境变量魔法"不冲突：反魔法反的是**隐式选择**，不是自动发现——自动检测的每一步证据都必须可打印、可审计。demo（`examples/bridge-demo`）已落地：xpu-only 机器自动激活 xpu 并打印探测证据；模拟双硬件同时在机 → fail fast 并提示 override。

### 4.2 probe 清单

device count、stream/event、async comm、IPC/P2P、allocator 特性、dtype 支持、**算子覆盖度**。

**算子覆盖度是最关键的一项**——它直接决定 `fail_fast` 的判定：某个必需算子没有实现，就该明确报错而不是训练中途崩。

### 4.2a 接口的"不支持"与可选能力的实测（2026-10-14 重写，取代 capability schema）

不再有一张平台自报的 capability 表（原 `CAP_SCHEMA` 的 probed/declared/identity
分档已删除）。取而代之的是两条更简单的规则：

- **不支持 = 不实现**：`PlatformBase` 的每个接口默认抛 `TTPlatformError("unsupported")`，
  平台不实现某接口即不支持，调到才报错；调用方（如 checkpoint）`except
  TTPlatformError` 走降级。接口存在性用 `supports(name)` 内省（子类是否覆盖
  默认实现）——派生自实现、不会撒谎，也不需要 conformance 回头验证声明的真假。
- **可选能力 = 实测，不声明**：graph 捕获 / 异步通信这类"平台可能有也可能没有"
  的能力，conformance 无条件跑一次实测，**结果即事实**。取代了原先"平台自报
  True 再验证它没说谎"的两步走——没有自报这个不可信输入，就不需要验证它的机制。

为什么删掉那张表（演进教训）：capability 表的 keys 曾退化为"复述实现已知事实"
（device_wide_sync 无人消费、graph_capture 是我们自己探测后又自验的自证循环），
字符串 key 散落在 6 处且只有"平台自报"一处被校验。删掉后这些信息全部由
实现本身或实测派生，字符串脆弱问题一并消失。

### 4.3 conformance 套件：激活期门禁

与 admission 分层：**admission（cuda-compat 侧）验绑定正确性**（符号绑对、两种引用观测一致）；**conformance（platform 侧）验 adapter 契约履行**。任一强制测试 FAIL 即**拒绝激活且不切换 activation**——不合格 adapter 活不进运行期（demo：naive adapter 被拦）。

强制套件（demo 已落地）：`sync_scope_device_wide`、`stream_dynamic_coverage`（运行期新建 stream 也必须被 device-wide sync 覆盖）、`mem_query_realtime`（口径 + 禁缓存）、`mem_info_consistency`、`rng_state_roundtrip`、`stream_fencing_semantics`、`stream_ctx_restore`、`device_identity_consumable`。可选能力实测（不阻断激活，结果即事实）：`graph_capture_roundtrip`、`async_comm_smoke`。

完整版另含：算子覆盖、数值、分布式（DP/TP）、checkpoint、RNG 一致性。


### 4.4 错误映射、生命周期与版本协商（demo v7 落地）

- **错误映射**：厂商原生异常（`XpuHwError`/`CudaHwError`）在 adapter 内翻译为统一 `TTPlatformError(category, detail)`，上层按 category 写可移植处理。
- **生命周期**：`init()`/`finalize()` 钩子，激活/退出各调一次；重新激活时先过门禁再切换、旧实例 finalize。
- **版本协商**：adapter 声明 `sdk_range`，激活期与实测 SDK 版本比对，越界 fail fast（demo：`sdk_version=9.9` 被拦）。

### 4.5 C++ 接口的两级绑定

Python 层门面只解决 Python 符号的分发；C++ 侧接口（自定义算子、CCL hook、allocator 等）的实现选择分两级：

| 级 | 机制 | 适用 |
|---|---|---|
| 编译期 | **Bazel select/alias**：按目标平台选择链接哪个 adapter 的 native 产物 | 默认路径。ABI 不兼容的厂商栈必须编译期隔离，不能共存于一个二进制 |
| 启动期动态绑定 | 已编译进二进制的多个实现，注册到 native 注册表，随 `auto_activate()` 的结果绑定函数指针/虚表 | 仅当产物 ABI 兼容、确实需要同二进制多平台时；**不做运行时 dlopen 搜寻** |

原则与 Python 侧一致：选择动作集中在启动期一次完成并打印，运行期热路径不做任何平台判断。

---

## 5. 时序状态机与失败行为

| 阶段 | 产物 | 失败行为 |
|---|---|---|
| 构建期 | 可链接的 platform 产物清单 | 构建失败 |
| 启动期 | platform 自动检测激活（证据打印）+ runtime report | fail fast（含多命中拒绝猜测） |
| 导入期 | cuda_compat 绑定结果 + trace | 按 policy：fallback / exception |
| 运行期 | 只执行已绑定结果 | 不做重新探测；异常走错误映射 |

**关键约束**：运行期不允许重新做硬件判断。所有需要在运行期"再看一眼"的需求，说明前一阶段的绑定设计有问题。

---

## 6. 落地顺序与理由

| 顺序 | 步骤 | 理由 |
|---|---|---|
| 1 | **符号盘点**（只读，产出清单） | 先知道真实工作量，避免拍脑袋定范围 |
| 2 | **定 tt_api 契约**（小） | 先有地面，才知道把东西绑到哪 |
| 3 | **platform_mgmt + 一个硬件跑通** | 先有绑定目标 |
| 4 | **cuda_compat 绑定** | 最后才动手绑，此时每步都有依据 |

反过来先做 compat 的常见问题：绑了一堆符号，却没有等价性判据，也没有可绑的目标契约。

---

## 7. 判据

| 判据 | 期望 |
|---|---|
| 换硬件 | **不改** cuda_compat policy（改了说明通用语义被塞进了兼容层） |
| Provider 全摘掉 | 功能不受影响 |
| 训练变慢时 | 能从 diagnostics 看出是哪些能力被 gate、哪些符号走了回退 |
| 新框架接入 | 只加 adapter，不动契约与 platform |
| 每个 remap 决策 | 都有 intent 记录 + 准入测试 id |

---

## 8. 社区对应工作（2026-10-10 调研）

PyTorch 官方在三个层面做了与 Bridge 对应的事，TT 应对齐而非重造：

| TT 概念 | 社区对应物 | 状态 |
|---|---|---|
| Universal API | `torch.accelerator` + 泛化 `torch.Stream/Event` | 概念 2.5、正式 2.6（Intel 主导）；2.9 大批 API ready；`set_device_index`、device ctx、`mem_get_info` 2.10 |
| Platform / adapter | PrivateUse1 三步（`rename_privateuse1_backend` + `_register_device_module` + `generate_methods_for_privateuse1_backend`）+ entry-points autoload + **OpenReg**（官方参考实现 + CI 守门） | 成熟，torch_npu / torch_musa 均走此路 |
| cuda-compat | 生态**源码迁移**运动：vLLM RFC #30679（2025-12，目标 2026 Q2）、compressed-tensors RFC #655 | 进行中 |
| 分布式后端映射 | `Backend.register_backend(devices=[...])` + `dist.get_default_backend_for_device()` | 成熟 |
| 编译侧设备接入 | Inductor `DeviceInterface` | 成熟 |

**关键背书**：vLLM RFC 的 NA 清单（`CUDAGraph`、`nvtx`、`profiler`、`CUDAPluggableAllocator`、`MemPool`、`can_device_access_peer`、`is_current_stream_capturing` 等无通用等价物）与 cuda-compat 的 fallback/exception 边界几乎完全重合——"通用语义取交集、专属能力走 gate"是社区实践验证过的划分。

**路线差异**：社区靠改框架源码 + lint 防新增；TT 靠运行时 remap（不改源码，对所有接入框架统一生效）+ 准入测试（语义等价验证是社区不管的空缺）。TT 定位 = 上游迁移完成前的运行时摆渡 + 迁移完成后残留的审计与门禁；随着上游迁移，policy 的 remap 条目应自然萎缩。

**社区盲区（TT 的增量空间）**：
1. `torch.accelerator` 设计上单机只允许一种 accelerator（编译期检测）——多异构卡同机官方不支持；
2. 框架迁到 accelerator 后，KLX 类 CUDA-masquerade 硬件的伪装价值下降，`vendor_native` 策略需重新评估；
3. 官方不提供逐符号的语义等价准入测试（sync 范围、RNG、显存口径）。

**行动项**：TT Universal API 直接代理/对齐 `torch.accelerator`，只补缺口（RNG 统一、comm metadata、provenance 钩子），不自造平行接口。
