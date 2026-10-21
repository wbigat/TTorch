# TTorch 通信（分布式）架构设计

> 创建于 2026-10-20，基线 commit `bd1b90f`（provider 函数级绑定通道之后）。
> 本机实测环境：KL3（torch 2.9+cu129 / torch_xmlir masquerade / 4 卡，
> megatron-core 0.18.0）。
>
> 与既有文档的分工：
> - `TT_Symbrewrite_Absorption.md` —— 替换机制与算子吸收（eager 路径）；
> - `TT_Rewrite_Mechanism_Design.md` —— 替换动作的实现收敛；
> - 本文档 —— **通信侧**的架构：问题定义、方案取舍、边界与判据。
>
> 本文只讲**设计逻辑与取舍**；具体问题的修复过程不进架构文档。

---

## 1. 问题定义：通信兼容到底要解什么

先看通信调用的真实形态（扫描 `megatron/core` 与 `torch/distributed` 本体）：

```
框架代码
  ├─ collective：dist.all_reduce / all_to_all / p2p ...     ← 设备无关
  └─ 使用现场：device guard、stream、event、RNG、显存       ← 全是 torch.cuda.*
```

所以"通信兼容"其实是**两件不同的事**：

| 关注点 | 归属 | 谁负责 |
|---|---|---|
| collective 语义本身（能不能跑通、结果对不对） | `torch.distributed` 原语层 | 厂商 backend（KL3 上 `xccl` 已注册为合法 Backend） |
| 通信的**使用现场**（CUDA 接口能不能工作） | `torch.cuda.*` 符号 | TT 的 bridge（cuda-compat） |

**TT 的定位由此确定**：不重做通信抽象，而是让"使用现场"可靠 + 让"语义仿真度"可验收。

---

## 2. 方案取舍：为什么不是 universal-comm-api

三个候选方案对比：

| 方案 | 做法 | 优点 | 致命问题 |
|---|---|---|---|
| **A. 自建统一通信 API** | TT 定义 `tt.comm.all_reduce(...)`，框架改调它 | 语义差异可在一处抹平 | **没人会迁移过来**：所有框架直调 `torch.distributed`；要它们改用 TT API 只能靠符号替换——既然最终都靠符号接管，中间那层 API 是纯负担 |
| **B. 注册真 backend**（FlagOS `ProcessGroupFlagOS` 路线） | 自己实现 ProcessGroup 注册进 `torch.distributed` | 覆盖最彻底、可承载私有设备类型 | 要对**全部**语义差异负责（device-scoped stream 静默 no-op、HCCL 无隐式 scaling、fp32 reduce-scatter 语义不齐…）；FlagOS 为此维护 1241 行 + 厂商表，TT 背上同等维护成本即偏离"证据驱动"定位 |
| **C. 原语层不接管 + CUDA 层符号接管 + 观测验收**（**选定**） | collective 交给厂商 backend；`torch.cuda.*` 按符号纳 policy；差异做成可验收项 | 与现有四层架构同构、零新子系统、维护面最小 | 语义差异需靠"验收"发现而非"抹平"，因此对 witness 的完备性有要求 |

**为什么 C 够用**：KL3 的现实是厂商栈已经把通信跑通了（`xccl` 是合法 backend，
`ProcessGroupXCCL` 甚至冒充了 `ProcessGroupNCCL`）。TT 在这个前提下再建抽象层，
是在**已工作的栈之上加一层**，收益为负。

**C 的风险与对冲**：C 不抹平差异，只把差异变成可观测证据。这要求验收项足够密——
§5 的差异化清单就是验收项的来源。

**什么时候重新考虑 B**（明确触发条件，避免提前建设）：
1. 接入的硬件**没有**合法 backend 注册（厂商没做 XCCL 这类工作）；
2. 语义抹平 shim 积累到 ≥3 处、跨 ≥2 个框架；
3. 需要**跨异构 mesh** 通信（如 P800 与 H20 混部流水线）——这是唯一没有现成抽象的场景。

---

## 3. 架构：两层拼合与边界判据

```
collective 原语层（torch.distributed.*）
    厂商 backend 承接；TT 只做验收与观测（不出实现）
             ↑ 使用现场依赖
CUDA 接口层（torch.cuda.*）
    TT bridge 按符号接管（policy 表：Entry + 准入测试 + provenance）
```

边界判据沿用项目既有的**"摘掉后果"**规则：

| 摘掉后果 | 归属 |
|---|---|
| 起不来 / 算错（backend 名、device guard、流序） | bridge（功能域） |
| 只是变慢（通算 overlap、融合 collective、通信调度） | provider（性能域） |

这条判据让通信侧与算子侧完全同构——通信没有特殊地位，也不需要专门的子系统。

---

## 4. 关键设计决策（逐条：选择 → 理由 → 代价）

### 4.1 符号接管 vs 语义抹平

**选择**：接管符号，不抹平语义。

框架写死 `"nccl"` 的地方（如 `megatron/training/gpu_sniff_test.py`），在 KL3 上由
厂商栈完成字符串改写；TT 不重复接管（同一符号两个 owner 会让回滚语义错乱）。
TT 只在**厂商未覆盖的平台**上才用现有 REMAP 决策接管，`impl` 由 platform adapter
提供。

**代价**：TT 不保证"任何框架在任何硬件上都能跑"，只保证"跑没跑对可被观测"。

### 4.2 类型位符号必须绑类对象

`torch.cuda.Event` 与 `Stream` 同族：`record/wait` 是流水线 p2p 与 activation
offload 串接阶段的唯一机制（megatron 实测 38 处调用点），且框架会写
`event: torch.cuda.Event` 注解、做 `isinstance` 判定。

**选择**：`impl_factory` 在 bind 期解析为平台**原生类对象**（不是转发函数）。
绑成函数会让注解求值在 import 期炸、isinstance 全线 TypeError。

**配套**：类型位不能只验"绑对了指向"，还要验语义——`admission.event_binding`
除类型位断言外，实测"侧流产出 → record → synchronize → query"跨流 fencing。
理由：Event 语义失真是**跨阶段乱序**（静默错），比崩掉更危险。

### 4.3 通信原语的版本谎报：不伪造、不阻断、留证据

**现象**：`torch.cuda.nccl` 是 torch 原生模块（厂商未替换），KL3 上 `version()`
返回编译期链接的 `(2, 27, 5)`，而实际通信跑 XCCL。DeepSpeed 的
`communication_backend_version()` 直调它报告后端版本。

三个可选处置：

| 处置 | 评价 |
|---|---|
| 编一个 XCCL 版本号返回 | ❌ 撒谎（symbrewrite 的 `is_nccl_available → True` 即此类，明确不吸收） |
| 抛 unsupported | ❌ 阻断：框架的版本检查分支会崩，为取证而破坏功能 |
| **回落原生值 + 记一条 unverified 证据**（选定） | ✅ 行为等价，且"这个数字不可信"变成可见事实 |

### 4.4 `default_generators` 明确不接管

它是 `tuple[Generator]` **值对象**，不是调用点。接管无价值，且属"值类型替换"
难题（symbrewrite 支持的整包/值替换，TT 当前不接）。不接管是设计结论，不是遗漏。

### 4.5 特性级符号按需触发，不预先接管

`MemPool`（user buffer）、`CUDAGraph`/`graph`（通信被图捕获）只在特定特性开启时
才进入调用面（megatron 中分别 14 / 10 处）。

**选择**：未验收前不接管；建议形态是 policy 表内以 **default-deny（EXCEPTION）**
占位——框架开启相关特性时 TT 显式报错并指向开关，**强于静默跑错**。

理由：这两类有真实的静默错前科（昆仑芯需 graph-safe all_reduce patch；
`ncclMemAlloc` 是编译期绑死的 C API，符号替换够不着——后者是平台能力边界，
应在框架开启 `nccl_ub` 时显式告知，而非假装支持）。

### 4.6 绑定坐标集中 + 失效可见

**问题**：坐标散在各语义包时，"TT 对 megatron 改了什么"只能靠 grep——provider
只有 1 个时无所谓，4 个时开始别扭，10 个且跨框架时成为排障主成本。

**选择**：保留 `impls/<语义>/`（按"一个语义的多硬件实现"切是对的），只把"改谁"
这一维抽到 `provider/bind_targets.py`，按目标框架分组，并带 `version_window`。

**为什么版本窗口只是字符串**：绑的 `_apply_rotary_pos_emb_bshd` 是 megatron
**私有函数**（前导下划线），上游无兼容承诺；symbrewrite 为此维护 6 个 transformers
版本目录。当前先把版本**写下来**让人能看见即可——真遇到两版共存再加解析，
不建版本匹配引擎。

**失效可见**：绑定目标失效时现象是"没报错但不加速"，最难排查。故 `install()`
末尾强制打印摘要，`skipped > 0` 时列出具体符号名（此前只进 JSONL，终端无感）。

---

## 5. 验收项来源：厂商栈的已知语义差异

TT 不抹平差异，就必须把这些差异做成验收项。来源两类：本项目实测 + FlagOS 跨硬件
适配暴露的差异（后者是前者的素材库）：

| # | 差异 | 类型 | 备注 |
|---|------|------|------|
| 1 | device-scoped stream（GCU/TsingMicro）：device 不匹配则**静默 no-op** | 静默错 | FlagOS device guard |
| 2 | `reduce_scatter_tensor` + fp32 累加语义各厂商不齐 | 静默错 | Megatron 用 A2A + 本地 sum 规避 |
| 3 | HCCL 无隐式梯度缩放语义（NCCL 有） | 静默错 | Ascend 需显式 scaling |
| 4 | 私有设备 tensor 非 cuda alias 时，cuda view 会正确性破 | 静默错 | kunlunxin |
| 5 | gloo 对私有设备 tensor 直接拒绝 | 显式炸 | Torch-FL redirect_gloo |
| 6 | MemPool `symmetric`/`symm_mem` 参数名随版本漂移 | 显式炸 | nccl_allocator |
| 7 | CUDA graph 捕获期 communicator 未初始化 | 显式炸 | vllm 惰性初始化 |
| 8 | `new_group(use_local_synchronization=True)` 在 xccl 上直接 raise | 显式炸 | **KL3 实测确认** |

**静默错类优先**（1-4）——显式炸会被立刻发现并修，静默错会污染训练结果。

---

## 6. 验收口径

| 层 | 验收什么 | 何时 |
|---|---|---|
| `admission` | 符号是否绑定到预期目标（含类型位、子模块符号） | 激活期 |
| `conformance` | 平台语义契约（同步范围、流序、状态口径、graph 捕获） | 激活期（fail fast） |
| `witness.verify` | provider 精度（fp32 参考轨道 fwd+bwd）与收益（生产 dtype） | 门禁期 |
| `assert_bindings_intact` | 运行期绑定是否被第三方覆盖 | 训练中/步边界可随时调用 |

通信侧新增的验收落点：`Event` 的类型位 + 跨流 fencing（admission）、
backend 相关符号状态（witness）。collective 语义本身的验收需要真实 process group，
**单机单 rank 自发自收即可测功能正确性**，不必等有多卡环境。

---

## 7. 当前覆盖与已知边界

**已覆盖**（KL3，单机）：policy 18 条全通过，含 `Event`（类型位 + fencing 实测）、
`manual_seed`、`torch.cuda.nccl.version`（子模块符号 compat 实例）。

**已验证场景**：4 卡 DP 训练跑通（`megatron sft`，NNODES=1 / WORLD_GPUS=4），
日志确认平台激活、policy 全量生效、函数级绑定在真实链路触发。

**已知边界**：
1. 多卡收益类验收（通算 overlap、融合 collective）尚未开展——provider 侧候选见
   `TT_Provider_Function_Map.md`，收益门禁需 `requires_multi_rank` 扩展后才有意义；
2. 跨 rank 数值对账（fingerprint）与 ActivationPlan 重放挂起到多卡前；
3. 语义差异验收目前只对 #8 有 KL3 实测，#1-#4 需按 §5 逐项补实测项。

---

## 8. 主要资料

- 本项目：`ttorch/bridge/module_binding.py`（policy 表）、`ttorch/utils/_rebind.py`
  （替换动作唯一实现）、`ttorch/provider/bind_targets.py`（坐标表）
- 对照实现：FlagOS `Torch-FL/torch_fl/comm/process_group.py`（真 backend 路线）、
  `symbrewrite/plugins/torch/mock_torch.py`（冒充路线）
- 官方：PyTorch custom operators / c10d Backend 扩展、Megatron TransformerLayerSubmodules
