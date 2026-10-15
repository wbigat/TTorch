# TT Bridge + Provider：异构训练收束层架构设计

> **定位**：TT（暂定名）是面向异构设备的训练执行收束层。它不替代 Swift、Megatron/MCore、DeepSpeed、verl，也不替代厂商的 runtime、编译器、通信库、Megatron-XPU 或 kernel 栈；其职责是将上层训练框架的基础功能接入与模块级性能优化解耦，并以统一、可验证、可追踪的方式管理多硬件执行。

---

## 1. 结论与目标

多硬件训练适配的真实组合并非只有“模型 × 硬件”，而是：

[  
M;(模型) \times F;(训练框架) \times H;(硬件) \times S;(场景)  
]

其中场景包括 Dense、长序列、MoE、多模态、SFT、RL 等。若每个组合都通过 fork、patch 或模型特判处理，复杂度会随组合相乘增长，并在模型、框架和厂商 runtime 演进时产生交叉回归。

TT 的目标不是承诺所有模型在所有硬件上自动达到同等性能，而是将绝大多数工作收束为：

[  
M;conformance + F;Bridge + H;Platform + P;Provider + 少量;Sentinel  
]

- **模型 conformance**：模型可被目标框架与模块语义表达，模型代码不含硬件分支；
- **框架 Bridge**：框架的启动、设备绑定、分布式语义、有限 CUDA 兼容和 feature gate；
- **硬件 Platform**：PyTorch device backend、c10d ProcessGroup、runtime、通信、dtype、RNG 等基础正确性；
- **性能 Provider**：以模块语义为单位实现 Reference、JIT 或 Native 的高性能执行；
- **Sentinel**：Dense、长序列、MoE、VLM、RL 等少量不可避免的交叉验证场景。

核心原则是：**新增模型、框架或硬件时，大多数变更只落在一个维度，而不是复制整套模型 × 框架 × 硬件适配。**

---

## 2. TT 的整体架构

TT 有两条独立但共享底座的主链路：

```text
功能路径：让训练框架先正确跑通

模型 / 训练框架
Swift / MCore / DeepSpeed / verl
          │
          ▼
  Framework Bridge
  - 启动与设备绑定
  - 有限 CUDA compatibility
  - feature gate
  - checkpoint / diagnostics
          │
          ▼
  torch.tt.runtime / torch.tt.comm
          │
          ▼
PyTorch ATen + autograd + torch.distributed/c10d
          │
          ▼
TT Platform（当前硬件）
          │
          ▼
厂商 PyTorch plugin / runtime / CCL / SDK


性能路径：让模块以可替换方式逐步获得性能

框架已有 module seam
          │
          ▼
Framework Module Adapter
          │
          ▼
ModuleSemanticSpec
          │
          ▼
Execution Provider
  ├─ ReferenceProvider
  ├─ JitProvider
  └─ NativeProvider
          │
          ▼
ExecutionPlan（构建期或首次编译期绑定）
          │
          ├─ PyTorch/MCore reference path
          ├─ Compiler Provider → lowering / codegen
          └─ torch.ops.tt.* → native fused op
          │
          ▼
TT Platform → 厂商 runtime / compiler / kernel / CCL
```

两条链路的关系：

- **Bridge 的目标是功能正确性**。基础训练应可走 PyTorch reference path，不因某个融合算子或编译器不可用而被阻塞。
- **Provider 的目标是性能兑现**。性能实现失败时必须能够回退到 reference path，并提供明确原因。
- **Platform 是共同底座**。它决定当前硬件是否具备正确的 PyTorch、通信和 runtime 基础能力。

---

## 3. 设计边界：TT 做什么、不做什么

### 3.1 TT 必须拥有的能力

1. 多框架 Bridge 与启动期适配；
2. `runtime / comm / ops` 的稳定公共契约；
3. 多硬件 Platform 注册、选择、能力探测与正确性资格验证；
4. 从框架模块 seam 到统一模块语义的适配协议；
5. Reference / JIT / Native Provider 的统一选择、回退和执行计划；
6. 数值、性能、fallback、编译产物和版本的 provenance；
7. M、F、H 加 Sentinel 的持续 conformance / qualification 体系。

### 3.2 TT 明确不应承担的职责

1. 不重写 Swift、Megatron、DeepSpeed、verl 的 trainer、scheduler 或算法控制流；
2. 不完整模拟或伪造 `torch.cuda`；
3. 不自己重写厂商 kernel、runtime、CCL 或编译器纵深栈；
4. 不在逐算子 forward 热路径做硬件识别和分发；
5. 不让 Provider 以 `model_name` 或 `framework_name` 作为主输入；
6. 不用手写 supported/unsupported JSON 作为能力真相；
7. 不为了统一而抹掉 MCore 5D 并行、DeepSpeed ZeRO、verl RL 权重流等真实语义差异。

---

## 4. Framework Bridge：功能接入层

### 4.1 Bridge 的职责

Bridge 解决的问题是：**某个框架能否在当前硬件上以 torch native 语义正确启动、训练、通信、保存和恢复。**

Bridge 是有限兼容层，应满足：

- 启动期生效，发生在目标框架 import / model build 前；
- 白名单化，仅覆盖已分析和已验证的调用；
- 每条兼容规则有框架版本范围、替代语义、测试用例和失效策略；
- 不支持时 fail fast 或显式 feature gate，绝不静默伪装；
- 不持有模型名，也不写 `Qwen × KLX` 一类组合分支。

推荐入口：

```python
tt.bridge.swift.activate(...)
tt.bridge.mcore.activate(...)
tt.bridge.deepspeed.activate(...)
tt.bridge.verl.activate(...)
```

而不是：

```python
import tt
# 隐式修改全局 torch.cuda，不可审计
```

### 4.2 Bridge 子模块

Bridge 组织上由两大块构成（2026-10-10 二次定稿）：**cuda-compat**（CUDA 调用点策略：remap / fallback / exception）与 **Platform Manager**（platform 类型注册/环境自动检测激活/构建/probe + per-platform adapter，填非标接口 gap；C++ 接口走编译期 Bazel 选择或启动期动态绑定）。**统一 runtime 接口不是独立子模块**——它没有独立语义内容：契约由 adapter 接口定义，分发是 `select()` 之后一行转发的机械门面，因此作为代码并入 Platform Manager（契约本体 + 门面函数），作为概念/契约见 `TT_Bridge_Deep_Dive.md` §3。执行链路：框架 CUDA 调用点 → cuda-compat 决策 → remap 到 interface 转发函数 → 激活的 platform 实现 → 厂商接口；fallback 走 torch-native；exception 明确报错。要守的约束是依赖方向（Provider 可消费 Platform Manager 的门面与接口，但 Provider 不依赖 cuda-compat、cuda-compat 不依赖 Provider），不是包归属。

下表是两大块内部的细分子模块：

| 子模块                   | 职责                                                 | v0 范围         |
| --------------------- | -------------------------------------------------- | ------------- |
| `bootstrap`           | 激活 Platform、加载 PyTorch plugin、注册 c10d backend、执行预检 | 必做            |
| `device_binding`      | rank/local-rank 到 device 映射、设备上下文和可见卡校验            | 必做            |
| `distributed_binding` | 初始化和校验 ProcessGroup；维护 DP/TP/PP/EP group metadata  | 先 DP、TP       |
| `cuda_compat`         | 对已验证的 CUDA 痕迹做定向映射                                 | 必做，但严格收缩      |
| `feature_gate`        | 明确禁用无等价实现的 CUDA-only 路径                            | 必做            |
| `framework_adapter`   | Swift、MCore 等的启动时序、参数和模型构建 seam                    | Swift + MCore |
| `checkpoint_bridge`   | 保存/恢复、RNG 状态、state schema 校验                       | 必做            |
| `diagnostics`         | 输出 Platform、backend、patch、禁用能力、版本与 fallback        | 必做            |
| `qualification`       | 单卡、DP、TP、resume、数值一致性测试入口                          | 必做            |

推荐目录：

```text
tt/
  bridge/
    bootstrap.py
    diagnostics.py
    feature_gate.py
    device/
      binding.py
      capability_probe.py
    distributed/
      process_group.py
      group_metadata.py
      validation.py
    compat/
      cuda_dependency_inventory.py
      cuda_compat_registry.py
      swift.py
      megatron.py
      deepspeed.py
      verl.py
    frameworks/
      swift_adapter.py
      mcore_adapter.py
      deepspeed_adapter.py
      verl_adapter.py
    checkpoint/
      state_schema.py
      resume_validation.py
```

### 4.3 CUDA 依赖治理：分析所有调用点，但不翻译所有调用

Swift、MCore、DeepSpeed、verl 以及其第三方依赖中的 CUDA 痕迹应持续盘点，但要分四类治理：

| 类别          | 典型调用                                                                                     | Bridge 处理方式                                                   |
| ----------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 通用设备语义      | `is_available`、`device_count`、`set_device`、`synchronize`、RNG、autocast、基础内存查询             | **remap** 到 interface 转发函数，经激活的 platform 实现落到厂商接口；对齐 PyTorch accelerator 语义 |
| 分布式语义       | NCCL 初始化、collective、P2P、异步 Work                                                          | 保持 `torch.distributed/c10d` 语义；`torch.tt.comm` 只包装元数据与扩展 hint |
| CUDA 专属性能能力 | CUDA Graph、NVTX、TF32、SM capability、`CUDA_DEVICE_MAX_CONNECTIONS`                         | 不伪兼容；feature gate 禁用或回退 torch-native 路径。Provider 可提供更优实现，但**功能可达不等待 Provider** |
| 第三方 CUDA 扩展 | Apex、Transformer Engine、FlashAttention、bitsandbytes、DeepSpeed OpBuilder、Triton CUDA path | 纯 torch-native fallback 或显式禁用；不放进 Bridge 模拟。厂商 / Provider 可提供加速实现，但**其缺失时功能路径仍成立** |

特别注意：`torch.cuda` 混杂了通用 accelerator 语义、CUDA runtime 语义和 NVIDIA 生态探测入口。全局替换 `torch.cuda` 会导致框架误选 fast path，且错误往往在训练中后段才暴露。因此，Bridge 的正确形态是**版本化依赖清单 + 定向规则**，不是 `TTUniversalCudaProxy`。

依赖清单示例：

```text
Swift <version>:
  torch.cuda.is_available        → runtime.is_available
  torch.cuda.device_count        → runtime.device_count
  torch.cuda.set_device          → runtime.set_device
  torch.cuda.amp.autocast        → runtime.autocast
  torch.cuda.get_rng_state       → runtime.get_rng_state
  torch.cuda.CUDAGraph           → disabled
  flash_attn import              → disabled / torch fallback
```

该清单应由静态扫描、启动期调用覆盖、capability probe 和框架 functional test 共同维护，而非人工维护一份静态能力 JSON。

### 4.4 Bridge 的输入与输出

```python
BridgeRequest(
    framework="mcore",
    framework_version="<version>",
    platform="klx",
    world_size=2,
    local_rank=0,
    parallelism={"tp": 2, "pp": 1, "dp": 1},
    precision="bf16",
)
```

```python
EffectiveRuntime(
    platform="klx",
    device_backend="torch_klx",
    process_group_backend="bkcl",
    applied_compat_patches=[...],
    disabled_features=[...],
    checkpoint_schema="mcore-tp-v1",
)
```

输出必须体现“有效运行态”，而不只是“应用了哪些 patch”。

---

## 5. TT 公共 API：Runtime、Comm、Ops

`torch.tt` 是 TT 内部和自有代码的稳定公共契约，不要求将所有外部框架源码全面迁移到新 namespace。

### 5.1 `torch.tt.runtime`

职责是硬件无关的运行时语义：

```python
torch.tt.runtime.is_available()
torch.tt.runtime.device_count()
torch.tt.runtime.current_device()
torch.tt.runtime.set_device(...)
torch.tt.runtime.current_stream()
torch.tt.runtime.synchronize()
torch.tt.runtime.get_rng_state()
torch.tt.runtime.set_rng_state(...)
torch.tt.runtime.autocast(...)
torch.tt.runtime.memory_info()
torch.tt.runtime.capabilities()
```

覆盖：device context、stream/event、RNG、AMP/dtype、allocator/memory 信息、runtime 初始化状态和 capability query。

自有新代码应优先对齐 PyTorch 的 `torch.accelerator` 语义；`torch.tt.runtime` 只承载 TT 所需、尚未被 PyTorch 公共接口覆盖的统一扩展。

### 5.2 `torch.tt.comm`

`torch.tt.comm` 不能另造一个脱离 `torch.distributed` 的私有 group 世界。

功能路径应继续通过厂商注册的 c10d `ProcessGroup` 执行：

```python
torch.distributed.init_process_group(...)
torch.distributed.all_reduce(...)
torch.distributed.all_gather(...)
```

TT 通信层的职责是提供统一 group metadata、拓扑/stream hint、overlap hint 和诊断：

```python
torch.tt.comm.get_dp_group()
torch.tt.comm.get_tp_group()
torch.tt.comm.get_pp_group()
torch.tt.comm.get_ep_group()
torch.tt.comm.all_reduce(...)
torch.tt.comm.all_gather(...)
torch.tt.comm.reduce_scatter(...)
torch.tt.comm.all_to_all(...)
torch.tt.comm.send(...)
torch.tt.comm.recv(...)
torch.tt.comm.describe_group(...)
```

通信原语至少需覆盖：

- DP：all-reduce、reduce-scatter；
- TP：all-reduce、all-gather、reduce-scatter；
- PP：send/recv、batch_isend_irecv；
- EP/MoE：all-to-all；
- 公共：barrier、异步 Work、timeout、错误传播。

### 5.3 `torch.tt.ops`

手工优化的算子建议最终注册为：

```python
torch.ops.tt.fused_swiglu(...)
torch.ops.tt.fused_rmsnorm_linear(...)
torch.ops.tt.column_parallel_linear(...)
```

Python 侧可以提供更友好的入口：

```python
torch.tt.ops.fused_swiglu(...)
```

但 canonical implementation 必须是 `torch.ops.tt.*`，并完成：

- operator schema；
- device kernel；
- autograd backward；
- fake/meta kernel；
- alias / mutation 语义；
- `torch.compile`、FX、export、activation checkpoint、FSDP/DTensor 的兼容性；
- `torch.library.opcheck()` 与 `gradcheck()` 验证。

---

## 6. TT Platform：每种硬件的基础正确性底座

### 6.1 Platform 的职责

每种硬件对应一个 Platform，例如：

```text
tt_platform_cuda
tt_platform_klx
tt_platform_xpu
tt_platform_npu
```

Platform 负责“能否正确训练”：

```text
- PyTorch device plugin 与 ATen dispatcher registration
- runtime 初始化、device、stream/event
- allocator 与基础 memory 语义
- RNG、AMP、dtype capability
- c10d ProcessGroup / CCL backend
- collective、P2P、异步 Work、错误传播
- 基础 checkpoint / serialization
- runtime / compiler / SDK capability probe
```

它不负责：模型判断、框架判断、每 op runtime dispatch、fused kernel 实现或 TP/PP 调度逻辑。

### 6.2 绑定时机

| 时机                   | 负责事项                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------- |
| Bazel / 编译期          | 厂商 SDK 头文件和 ABI、CCL、compiler 链接、PyTorch plugin、native op、Platform 动态库构建                 |
| 进程启动期                | 探测可见硬件、加载匹配 Platform、初始化 runtime、注册 device/c10d backend、探测 capability、生成 runtime report |
| 模块构建期 / 首次编译期        | Provider 根据 Platform capability 与模块语义选择 ExecutionPlan                                   |
| forward/backward 热路径 | 只执行已绑定 plan；禁止重复硬件探测                                                                    |

Bazel/编译期的意义是确定哪些 SDK、ABI 和 Platform 包能够安全链接与加载；运行期只应从已构建、已加载的候选中选择当前 Platform。

---

## 7. Execution Provider：模块级性能兑现层

> **实施状态（2026-10-15）**：本节是 Provider 的**目标形态**（面向 ParallelLinear/
> MoE、ModuleSemanticSpec 归一化、framework_module_adapter、Reference/Jit/Native
> 三分、megatron 路径）。当前**首个落地**走的是更小的 transformers/eager 路径——
> RMSNorm provider：两层结构（`Provider`/`ProviderImpl`）+ 自动注册/生效/选择 +
> 统一 provenance，代码在 `ttorch/provider/`。as-built 结构、实测结论与经验教训见
> `TT_Provider_Function_Map.md` §6。ModuleSemanticSpec / framework_module_adapter /
> megatron 路径的 ParallelLinear 仍是后续方向（本节愿景不变）。

### 7.1 Provider 的职责

Provider 解决的问题是：**某类训练模块在当前 Platform 上如何以可替换、可验证、可回退的方式执行。**

Provider 必须面向模块语义，而非框架 Python 类名、模型名或硬件名。例如：

```text
ParallelLinearProvider
AttentionProvider
MLPProvider
NormProvider
MoEProvider
EmbeddingProvider
LossProvider
OptimizerProvider
```

首个 Provider 建议选择 `ParallelLinearProvider`：

- GPT/Qwen Dense MLP 和 attention projection 可复用；
- MCore 已具备 `ColumnParallelLinear / RowParallelLinear` 和 Builder / ModuleSpec seam；
- 可覆盖 TP 通信、参数分片、反向、checkpoint shard 等关键语义；
- 比 Attention、MoE 更适合建立第一个 correctness 闭环。

### 7.2 Framework Module Adapter：不要让 Provider 直接理解框架

性能路径必须在 Provider 前增加框架模块适配：

```text
MCore ModuleSpec / BackendSpecProvider / Builder
Swift / Transformers 模块替换规则
DeepSpeed engine / optimizer seam
verl Model Engine seam
            │
            ▼
      ModuleSemanticSpec
            │
            ▼
      Execution Provider
```


Provider 不应该写：

```python
if framework == "megatron":
    ...
```

它只接收归一化的模块与并行语义：

```python
ModuleSemanticSpec(
    kind="column_parallel_linear",
    dtype="bf16",
    training=True,
    backward_required=True,
    shape_bucket={"seq": 4096, "hidden": 4096},
    parallel_layout={
        "tp": 2,
        "sp": False,
        "ep": 1,
        "input": "replicated",
        "output": "hidden_sharded",
    },
    checkpoint_schema="mcore-tp-v1",
)
```

这也是避免重新形成 `F × H`、`M × H` 专用 patch 债务的关键。

### 7.3 三种统一 Provider 实现

三种实现应服从同一个 Execution Provider 契约，而不是形成两套 wrapper：

| Provider            | 实现                                                      | 作用                              |
| ------------------- | ------------------------------------------------------- | ------------------------------- |
| `ReferenceProvider` | 原始 PyTorch / MCore reference module                     | 正确性基线、bring-up 默认、数值对照、fallback |
| `JitProvider`       | FX/ATen → lowering → per-platform compiler → executable | 图/算子编译与自动融合                     |
| `NativeProvider`    | `torch.ops.tt.*` → 厂商 native fused op                   | 手工优化、特化 layout/通信/算子实现          |

Provider 统一返回：

```python
ExecutionPlan(
    provider="tt.native.parallel_linear.v1",
    implementation="torch.ops.tt.column_parallel_linear",
    guards=[...],
    fallback="tt.reference.parallel_linear.v1",
    provenance_id="...",
)
```

Provider 的选择发生在**模块构建期或首次编译期**：

```text
ModuleSemanticSpec
+ Platform capability
+ dtype / training mode
+ shape bucket
+ TP / SP / EP layout
        │
        ▼
Candidate ranking
        │
        ▼
ExecutionPlan binding + cache + provenance
```

之后的 forward/backward 只调用已绑定 plan。允许重新选择的情形是 shape bucket、dtype、并行布局、训练/推理模式变化或 guard 失效，而不是每次执行都通过 `if platform == ...` 分支。

### 7.4 Provider 子模块

| 子模块                        | 职责                                          | v0 范围            |
| -------------------------- | ------------------------------------------- | ---------------- |
| `module_semantics`         | 模块语义、并行布局、状态 schema                         | 必做               |
| `framework_module_adapter` | 从 MCore/HF module seam 提取语义                 | MCore + HF/Swift |
| `provider_registry`        | 注册 Reference/Native/JIT 与能力声明               | 必做               |
| `plan_selector`            | 根据 capability、dtype、shape、布局选择 plan         | 必做               |
| `reference_provider`       | reference path 与正确性 fallback                | 必做               |
| `native_provider`          | 通过 `torch.ops.tt.*` 调用手工融合算子                | 先一个模块            |
| `jit_provider`             | FX/ATen、lowering、compiler provider、artifact | 后置               |
| `artifact_cache`           | 编译产物、guard、版本、cache key                     | JIT 阶段引入         |
| `provenance`               | provider 选择、fallback、性能、数值证据                | 必做               |
| `provider_qualification`   | 前反向、checkpoint、性能验证                         | 必做               |

推荐目录：

```text
tt/
  provider/
    semantics/
      module_spec.py
      parallel_layout.py
      state_schema.py
    adapters/
      mcore.py
      transformers.py
    registry/
      provider_registry.py
      capability.py
      plan_selector.py
    implementations/
      reference/
        parallel_linear.py
      native/
        parallel_linear.py
      jit/
        compiler_provider.py
    execution/
      execution_plan.py
      fallback.py
      artifact_cache.py
    observability/
      provenance.py
      metrics.py
      numerical_check.py
```

> 实施状态（2026-10-15）：此 `observability/` 层已落地为 **`ttorch/witness/`**——
> `witness/provenance.py`（统一事件流）、`witness/bindings.py`（bridge API 验证）、
> `witness/call_trace.py`（运行期中转证据）、`witness/verify.py`（provider 精度/
> 性能验证原语）。witness 与 bridge/platform/provider 平行为第四层。

---

## 8. Bridge、Platform、Provider 的接口边界

```text
Framework Bridge
  负责：框架启动、device/rank 绑定、有限 CUDA 兼容、feature gate、基础训练与 checkpoint 可用
  不负责：选择 fused kernel、JIT 编译、模块性能替换

Framework Module Adapter
  负责：从框架 seam 提取模块、布局、状态和训练语义
  不负责：硬件实现选择、框架启动

TT Platform
  负责：PyTorch/c10d/runtime/CCL 的基础正确性与 capability
  不负责：模型/框架判定、模块性能策略

Execution Provider
  负责：语义模块的 Reference/JIT/Native ExecutionPlan 选择、回退和观察
  不负责：框架生命周期、模型名/框架名分支

Hardware / Vendor Stack
  负责：runtime、compiler、CCL、kernel、allocator、native PyTorch plugin
  不负责：上层模型与框架语义
```

因此应维护如下单向依赖：

```text
Framework → Bridge / ModuleAdapter → TT Contract → Platform / Provider → Hardware
```

禁止反向依赖：

```text
Model / Framework → vendor-specific kernel/runtime
Provider → framework-private trainer lifecycle
Platform → model_name
```

---

## 9. 可观测性与 Debug：TT 的核心工程资产

TT 的价值不能只体现在“选择了哪个插件”，而必须把每一次执行决策变成可追踪证据。每个模块绑定应记录：

```text
module_path:
  decoder.layers.3.mlp.linear_fc1

semantic_kind:
  column_parallel_linear

platform:
  klx

request:
  dtype=bf16
  training=true
  TP=8, SP=true
  shape_bucket=S8K_H4096

selected_provider:
  tt.native.parallel_linear.v2

selected_implementation:
  torch.ops.tt.fused_swiglu_column_parallel

rejected_candidates:
  tt.jit.parallel_linear.v1
  reason=sequence_parallel_backward_unsupported

fallback:
  false

artifact:
  kernel_version
  compiler_version
  cache_key
  build_id

runtime_metrics:
  compile_ms
  calls
  latency_p50 / latency_p99
  peak_workspace
  communication_time
  numerical_check_status
```

建议提供：

```python
torch.tt.diagnostics.report()
torch.tt.diagnostics.explain("decoder.layers.3.mlp.linear_fc1")
torch.tt.providers.pin(module_path, "reference")
torch.tt.providers.pin(module_path, "jit")
torch.tt.providers.pin(module_path, "native")
```

这使得性能回退、数值偏差和框架升级问题可以定位到：Bridge、Platform、Provider、编译器产物或厂商 runtime，而不是停留在“某硬件不支持”的模糊结论。

---

## 10. v0 开发范围与阶段计划

### 10.1 v0 最小闭环

```text
硬件：KLX（Platform-1）
模型：Dense GPT/Qwen 小模型
框架：Swift Transformers + 直接 MCore
功能：单卡 → 2 卡 DP → 2 卡 TP
Bridge：有限 CUDA compatibility + ProcessGroup + checkpoint resume
Provider：ParallelLinear 的 Reference → Native
观测：runtime report + provider provenance + fallback + 数值/性能指标
```

### 10.2 分阶段计划

| 阶段                    | 目标与交付物                                                                                   | 验收标准                                                       |
| --------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| P0：契约与空壳              | 定义 Platform、`EffectiveRuntime`、`ModuleSemanticSpec`、`ExecutionPlan`、provenance schema    | 启动时打印完整 runtime report；无模型 × 硬件专用分支                        |
| P1：Bridge 功能闭环        | KLX bootstrap、device/c10d binding、Swift/MCore 有限 CUDA compat、feature gate、同并行度 resume    | 单卡 fp32/bf16、2 卡 DP；loss/梯度/参数更新正确；resume 后 loss 连续        |
| P2：MCore TP 基线        | MCore Module Adapter；Column/RowParallelLinear reference path；TP group metadata 与 c10d 对齐 | TP=2 Dense 前反向正确；TP checkpoint 可恢复；与参考硬件数值满足门槛             |
| P3：首个 Native Provider | `ParallelLinearProvider`、`torch.ops.tt.*`、Reference fallback、provenance 与 pin            | Native/reference 前反向、梯度、state_dict 一致；可强制回退；fallback 原因可定位 |
| P4：性能资格体系             | provider 性能/数值回归、profile attribution、shape bucket、capability probe                       | 输出 tokens/s、step P50/P99、峰值显存、通信占比、数值校验和实现来源               |
| P5：第二框架与第二模块          | Swift-Megatron 或 DeepSpeed 明确 Bridge；RMSNorm 或 MLP Provider                              | 新框架仅增加 Adapter；新模块不引入模型名特判                                 |
| P6：JIT 与复杂场景          | JitProvider、Attention、MoE、长序列、ZeRO-3、verl 权重流                                            | JIT/native/reference 共用 ExecutionPlan；复杂 sentinel 可稳定回归    |

### 10.3 v0 非目标

第一版明确不做：

- 全局完整模拟 `torch.cuda`；
- DeepSpeed ZeRO-3、verl RL、MoE、CP、长序列同时上线；
- Attention 和 MoE 的高性能实现；
- Triton/JIT 与 native fused op 同时开发；
- 多硬件 Platform 并行铺开；
- 自动处理所有第三方 CUDA 依赖；
- 全并行度 checkpoint reshard。

---

## 11. 验收体系

### 11.1 功能 conformance

- 单卡：forward、backward、AdamW、AMP、RNG、save/load；
- DP：梯度同步、固定 seed、断点恢复；
- TP：Column/RowParallelLinear、all-reduce/all-gather/reduce-scatter、TP checkpoint；
- checkpoint：同并行度恢复后 loss 连续，参数和 optimizer state 合法；
- Bridge：所有 patch、禁用 feature 和 runtime binding 可报告、可测试、可回滚。

### 11.2 Provider conformance

- Reference 与 Native/JIT 的 forward、input grad、parameter grad、optimizer update 对齐；
- state_dict key、parameter alias、TP shard layout、checkpoint metadata 不被破坏；
- activation checkpoint、autograd、FakeTensor/meta、`torch.compile` 组合行为可验证；
- Native/JIT 不满足 guard 时可回退 Reference，且不会静默改变语义。

### 11.3 数值与性能

- bring-up 先使用 fp32 定位问题；
- bf16/fp16 阶段采用任务定义的误差门槛，例如 abs/rel error < 5e-3；
- 性能持续记录 tokens/s、step latency、P99、通信占比、峰值显存、编译耗时、fallback 比例；
- 不以“native 路径被选中”作为性能结论，必须以 profile 和端到端指标为准。

---

## 12. 最终架构判断

1. **Bridge 不是完整 CUDA 模拟层。** 它持续分析上层框架 CUDA 依赖，但只桥接具有稳定跨硬件语义的 runtime/comm 部分；CUDA 专属性能能力通过 feature gate、fallback 或 Provider/Platform 承接。
2. **Platform 决定能否正确训练。** 设备 plugin、ATen、autograd、c10d、CCL、RNG、AMP、checkpoint 是基础底座。
3. **Provider 决定能否高性能训练。** 它以模块语义而非模型或框架名称组织，统一管理 Reference、JIT 和 Native 三类实现。
4. **编译后端是 Provider 的实现方式，而非收束层入口。** 图编译、算子编译、lowering 和 codegen 可以跨硬件复用，但必须挂在已经稳定的模块语义与 ExecutionPlan 之后。
5. **厂商纵深栈仍不可替代。** 厂商 runtime、compiler、CCL、kernel、MoE 调度、通信重叠应作为 Platform/Provider 的实现能力接入，而非被 TT 重写。
6. **TT 的长期价值是可验证的正交性。** 模型升级、框架升级、硬件增加或性能优化发生时，变更应落在 M、F、H、P 中的一个清晰边界，并通过少量 Sentinel 发现真实交叉泄漏。
