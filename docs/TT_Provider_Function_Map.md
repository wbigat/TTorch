# TT Provider 功能地图

> 前置：Bridge 负责**功能可达**（torch-native），Provider 负责**性能增量**。两者唯一接口是调用边界。
> Provider 的一切设计都服从"失败无损"——它只能让训练更快，不能让训练跑不起来。

## 0. 四条不变式

| # | 不变式 | 违反后果 |
|---|---|---|
| 1 | Provider 缺失只降性能，不降功能 | Bridge 目标被破坏 |
| 2 | 任何 Provider 失败 → 回退 reference，**无失败模式** | 性能模块变成可用性风险 |
| 3 | 选择发生在**构建期/首次编译期**，热路径零探测 | 每 step 判断吞掉收益 |
| 4 | **无收益证据不启用** | "发起了异步"被误当成优化 |

---

## 1. 调用边界

> **实现注记（2026-10-15）**：最初设想"只是一组签名+元数据、不是抽象基类体系"，
> 落地时演化为**最小两层 ABC**（见 §6）：`Provider`（第一层，一个模块语义一个类型，
> 持有 native + 各平台 impl）+ `ProviderImpl`（第二层，某平台的一份高性能实现）。
> 层次仍刻意保持薄——基类只承载"调用签名 + probe_input + supports + __call__"
> 这组契约，不引入额外抽象。

注册元数据（`ProviderImpl` 类属性）：`name` / `semantic` / `slot` / `kind` /
`platforms` / `min_speedup` / `build` / `intent`；调用签名即 `__call__`（语义专属，
如 RMSNorm 的 `(x, w, eps) -> y`）+ `supports`（逐调用适用性）+ `probe_input`
（语义级门禁探测输入）。

为什么不能没有：正因为实现是 N 份异构的，manager 才需要统一方式去注册、询问、调用、回退；否则 manager 自己变成 M×N 的 if-else 中心。

---

## 2. provider_manager：六个职责

| 职责 | 做什么 | 关键约束 | 失败行为 |
|---|---|---|---|
| **registry** | 注册与发现实现 | 按**模块语义**注册，禁止模型名/硬件名作为主键 | 无实现 → reference |
| **selection** | capability 过滤 + 优先级 + 输出 `selection_trace` | 必须能力驱动，**禁止 `if hardware == "KLX"`** | 无满足者 → reference |
| **gate** | 数值等价 + 执行合法 + 收益证据 | 三项缺一不启用，自动化可回归 | 未过闸 → 不启用并记原因 |
| **binding & cache** | 构建期/首次编译期绑定；JIT 编译 key、缓存与失效 | key 含 shape/dtype/并行/版本 | 缓存失效受控，不每 step 重编 |
| **switch** | 开关与灰度：per-module / per-layer / per-kind / **全局 kill switch** | kill switch 是一等公民 | 关闭 → 该插槽回退 |
| **observability** | 性能观测（吞吐/延迟/显存）+ provenance | 在线持续做性能，数值等价靠离线 gate + 哨兵抽样 | 观测失败不影响训练 |

---

## 3. implementations：模块语义 × 两个独立插槽

```
parallel_linear/          attention/          moe/          norm/     comm_fusion/
    compute : reference | fused_ops | jit_compile
    schedule: serial | overlap | builtin        # 可选，缺省 serial
```

**schedule 不与 jit / fused_op 并列**，是单独可选的一环，可与之叠加：

1. **可选**：缺省 `serial`，不存在也是合法状态。
2. **可叠加**：fused MoE kernel 也能配 `a2a_overlap`，不是二选一。
3. **`builtin` 例外**：计算实现内部已固化时序时（如 fused a2a+gemm），插槽让位。

两个插槽**独立注册、独立选择、独立开关、独立回退**——线上可把"换算子"和"改时序"分开灰度归因。

### 分叉轴与压缩

**分叉轴必须是「模块语义 × 硬件」，绝不能让模型名渗进来。** 模块语义集合小且稳定，不随模型演进。

| 实现类型 | 硬件专属？ | 份数 |
|---|---|---|
| reference（ATen 组合） | 否 | **1 份全硬件共享** |
| 可移植实现（Triton 等，需硬件提供后端） | 视支持 | 少量，可跨硬件复用 |
| JIT / 编译 | 管线共享 | **1 管线 + N backend** |
| 厂商原生库 | 是（多为薄适配） | N 份 |
| schedule 计划 | 计划共享，门槛不同 | **1 份计划 + N 个能力判断** |

---

## 4. Provenance（生效溯源）

**定义**：能回答"这次训练每个模块用了谁的实现、为什么选它、被拒的是谁、实际退到哪、收益多少"。

它不是日志，是**一等能力**：没有 provenance，就无法区分"跑得慢是因为没启用优化"还是"启用了但无效"，也无法在出问题时快速归因。

### 四个产生阶段

| 阶段 | 记录内容 |
|---|---|
| **构建期** | 链接了哪些 platform/实现产物：版本、commit、编译选项 |
| **启动期** | platform 选择结果、capability probe 实测值 |
| **绑定期** | 每个 `module_kind × slot`：候选列表、被选中者、被拒者及原因、收益证据引用 |
| **运行期** | 是否保持绑定、是否发生运行时回退、编译/重编译次数、性能观测序列 |

### 建议 schema

```json
{
  "module_kind": "moe",
  "slot": "compute",
  "selected": {"impl": "vendor_fused_moe", "kind": "fused_ops",
               "version": "1.4.2", "source": "vendor"},
  "rejected": [
    {"impl": "tt_triton_moe", "reason": "capability_missing", "detail": "no_triton_backend"},
    {"impl": "jit_moe", "reason": "gate_failed", "detail": "perf_gain 0.97x < 1.05x"}
  ],
  "fallback_actual": null,
  "evidence": {"equiv_test": "moe_fp16_ref_v3", "perf_gain": "1.31x",
               "profiler_ref": "trace/2026-09-30-moe-a2a"},
  "switch": {"global_kill": false, "per_module": "enabled"}
}
```

`slot: "schedule"` 另有一条记录，结构相同。

### 三条硬要求

1. **可打印**：启动时 dump 生效来源，**不隐藏 fallback 顺序**。
2. **可导出**：与 Bridge 的 `cuda_compat.trace`（哪些 CUDA 符号被替换、走了哪条回退）**统一 schema**，合并成一份报告——回答"这次训练用了哪些替身和加速"。分开两份，排障要对账。
3. **可复现**：provenance 应足以让另一次运行重放同一套选择（含 capability 快照）。

### 消费者

排障归因 · 灰度 A/B · 性能回归基线 · 交付与验收记录 · 硬件能力演进追踪

---

## 5. 验收与判据

| 判据 | 期望 |
|---|---|
| kill switch 全开 | 回到 Bridge 基线，功能不受影响，数值不变 |
| 新增一个模型 | Provider 实现份数**不变**，只增一条 conformance |
| 新增一个硬件 | N 侧加一列，**优先继承共享层**，只为有收益证据的模块写专属 |
| 单个 provider 关闭 | 该模块回退 reference，其余照常，无级联失败 |
| 每个已启用 provider | 必须同时有等价证据、合法性证据、收益证据 |

反例判据：若新增模型要写新 provider 实现 → 模块语义抽象失败；若新增硬件要重写所有模块 → 共享层太薄。

---

## 6. 实施状态（as-built，2026-10-15）与经验教训

本节记录首个真实 provider（RMSNorm，KL3）落地后的**实际结构**与**实测结论**，
与前面的设计愿景对照。代码在 `ttorch/provider/`，demo 在 `tt-provider-demo/`。

### 6.1 落地的结构（两层 + 三段自动化）

> **术语**：设计文档里的"reference"（功能底线/精度收益基准）在代码里即
> **native**——`Provider.native` 属性、impl 名 `native:<语义>`，但 `kind="reference"`。
> 用 native 是为了避开 bridge 里"native 通道"（torch.xpu/device_name）的撞词。

```
ttorch/provider/
  spec.py      Provider（第一层：一个模块语义一个类型，持有 native + 各平台 impl）
               + ProviderImpl（第二层：某平台一份高性能实现）
  registry.py  ProviderRegistry（按 semantic 注册第一层 Provider）
  manager.py   ProviderManager（门禁/选择/分发/开关/bypass）+ 进程级单例
  impls/<语义>/  base.py（Provider + native）+ <平台>.py（各硬件 ProviderImpl）
```

三段自动化（对接 framework 的成本压到最低）：

| 段 | 时机 | 入口 |
|---|---|---|
| 自动注册 | `ttorch.activate()` 时 | `provider.activate()` 注册全部已知 provider + 接线平台 |
| 自动生效 | 模型建好后 | `ttorch.provider_wrap(model)` 按 `module_names` 自动发现并 patch |
| 自动选择 | 首次前向 | 惰性 `select()`：platform 过滤 + 三级门禁 + kind 优先级，写绑定缓存 |

provenance 与 bridge **统一进 `ttorch.witness.provenance`**（同一事件流，信封
`seq/ts/layer/event/subject/outcome/evidence`），实现了 §4"与 bridge trace
统一 schema 合并成一份报告"的硬要求。

> **观测层独立（2026-10-15）**：provenance 与 bridge 的 API 验证、运行期中转证据、
> provider 的精度/性能验证，已收拢成与 bridge/platform/provider 平行的第四层
> **`ttorch/witness/`**（witness = 见证：采集完整执行数据 + 对 bridge/provider 取证）：
> `witness/provenance.py`（统一事件流 + session 快照 + summary 聚合）、
> `witness/bindings.py`（bridge API 验证 check_bindings）、`witness/call_trace.py`
> （运行期中转证据）、`witness/verify.py`（provider 三级验证的测量本体，gate 复用）。
> 对应愿景文档里的 `observability/` 层。

### 6.2 KL3 RMSNorm 案例的实测结论（诚实）

- **数值正确性成立**：provider 选中厂商融合 kernel 后，整网前向 loss 与 baseline
  **逐位一致**（bf16 err=0），训练正常收敛，checkpoint 连续。
- **整网性能持平（~1x），不是加速**：因为 KL3 的 torch_xmlir **早已**把
  transformers 的 RMSNorm 符号替换成它自己的融合实现（RMSNormFunction，同源
  `rms_layer_norm` kernel）。本 provider 与之同源同速。
- **价值定位**：本 provider 的意义是统一的可插拔机制（门禁/回退/provenance/
  kill switch），以及在厂商**未预融合**某算子的硬件/算子上补融合——RMSNorm
  恰好在 KL3 已被厂商吃掉，所以这个具体算子在这里没有净增益。

### 6.3 经验教训（后续 provider 必守）

1. **推理 kernel ≠ 训练可用**。厂商 `rms_layer_norm` 是纯前向、无 autograd
   （输出无 `grad_fn`），直接进训练会切断梯度图、loss 不学。**训练态实现必须
   包 autograd.Function**（forward 快 kernel + backward 用厂商反向 kernel
   `rms_layer_norm_backward`）。
2. **小心 RMSNorm 变体**。Qwen3.5 是 zero_centered_gamma（有效 scale=`1+weight`
   而非 `weight`）；wrap 时必须折成有效权重，否则与厂商实现差 5.16。
3. **"松的收敛检查"会掩盖数值错**。过拟合单 batch 时即使 RMSNorm 数值偏了
   loss 也照降——必须用"同权重前向 loss 逐位对齐"来验数值等价，不能只看收敛。
4. **forward-only 的"提速"是假象**。backward 没真跑时整网会假性变快——测整网
   收益必须 forward+backward 都真实跑。
5. **先查厂商是否已优化该算子**（torch_xmlir 的 symbrewrite 符号替换清单），
   避免重复造轮子；provider 的增量价值在厂商没覆盖的地方。
