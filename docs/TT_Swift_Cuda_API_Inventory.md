# ms-swift 原生框架 CUDA 接口盘点 × Bridge 补齐缺口分析

> 数据源：`/Users/vector/code/train/ms-swift`（main@21ada4620，528 个 py 文件）。
> 方法：全仓枚举 `torch.cuda.*` / `.cuda()` / CUDA 相关环境变量 / 厂商工具链调用，
> 按接口性质（有副作用的行为 / 动态只读的状态查询）与三种决策
> （remap / fallback / exception）归类，对照 demo 盘点面找缺口。
> （2026-10-14：原 capability 一类已删除——静态事实改普通只读接口，
> 可选能力改 conformance 实测，见 TT_Bridge_Deep_Dive.md §4.2a。）
> 日期：2026-10-12。

## 0. 两个结构性发现（先于 API 清单）

1. **swift 自带一层分支式设备抽象**（`utils/torch_utils.py`）：`synchronize /
   get_device / get_current_device / get_torch_device / set_device /
   get_device_count / empty_cache / ipc_collect` 都是 `if is_torch_npu elif
   is_torch_cuda elif is_torch_mps else cpu` 的硬编码分支。**每加一个平台要改
   N 处、且只对走这层 helper 的调用点有效**——下文清单里 17 处
   `torch.cuda.current_device` 直连说明大量调用点绕过了它。这正是我们
   Platform Manager 门面（单接口 + 内部分发）的替代价值所在。
2. **`model/npu_patch/` 目录（9 个 monkey-patch 文件）就是"没有 Bridge 时的
   现状"**：mindspeed.py / model.py / megatron_checkpoint.py /
   vllm_ascend*.py——每接一个新平台就要攒一摞裸 patch，无门禁、无
   provenance、无版本窗口。

## 1. CUDA 接口全量清单（按用途分组，频次=全仓出现次数）

### 1.1 设备发现与生命周期

| API | 频次 | 代表调用点 | 三类归属 | 语义/风险 |
|---|---|---|---|---|
| `torch.cuda.is_available` | 10 | `rollout.py:406`（`f'cuda:{rank}' if ... else 'cpu'`）、`activation_cpu_offload.py:16`、`trainers/mixin.py:698` | state_query | **分支选择器**：结果直接决定设备字符串形态。masquerade 平台上答案与 vendor 身份脱钩（GCU 报 True） |
| `torch.cuda.current_device` | 17 | **关键：`megatron/utils/utils.py:214-221`、`parallel_utils.py:14,24`、`trainers/base.py:1043` 把它当 `device=` 参数直接传给 `torch.tensor/empty`** | state_query | 返回值被当**设备标识**用（不只索引）。非 CUDA 平台上 `torch.tensor(device=<int>)` 语义直接断裂——这是最容易漏的隐性契约 |
| `torch.cuda.set_device` | 2+封装 | `torch_utils.py:165`、`infer_engine/utils.py:393` | behavior | 标准 |
| `torch.cuda.device_count` | 1+封装 | `torch_utils.py:172` | state_query | 标准 |
| `torch.cuda.is_initialized` | 1 | `dataset/packing.py:29` | state_query | lazy-init 判定，packing 的 device 放置前提 |
| `get_physical_device_count` 内部 | 1 处函数 | `torch_utils.py:187-244` | state_query | **直接调 pynvml / 读 `/proc/driver/nvidia` / 起 `nvidia-smi` 子进程**——厂商工具链硬依赖，绕过 mask 查物理卡数 |

### 1.2 同步与 stream 原语

| API | 频次 | 代表调用点 | 三类归属 | 语义/风险 |
|---|---|---|---|---|
| `torch.cuda.synchronize` | 4+封装 | `trainers/base.py:707`、`rollout.py:511` | behavior | device-wide 契约，demo 已覆盖 |
| `torch.cuda.Stream()` / `current_stream()` / `stream()` ctx / `wait_stream()` | 各 1-2 | **`megatron/utils/megatron_lm_utils.py:686-704`**：ddp 侧流 fencing | behavior | **stream 对象模型**（创建/当前流/上下文切换/跨流等待）。xswift 的 `wrap_model` 补丁正是修这段的上游缺陷——XPU 落地时这段必被触发 |
| `torch.cuda.empty_cache` | 5+封装 | `megatron_lm_utils.py:852`、`rollout_mixin.py:1439,1459` | behavior | allocator 提示；kl3 类栈语义可能不同（释放粒度） |
| `torch.cuda.ipc_collect` | 1+封装 | `torch_utils.py:258` | behavior | IPC 显存回收 |
| `torch.cuda.memory._set_allocator_settings` | 1 | `rlhf_trainers/utils.py:1680`：`expandable_segments:{enable}` | behavior | **CUDA allocator 私有配置**——非 CUDA 平台无对应物，是 fallback(no-op)/exception 决策的天然样板 |

### 1.3 显存状态查询（日志与 RL 显存管理依赖）

| API | 频次 | 代表调用点 | 三类归属 |
|---|---|---|---|
| `memory_allocated` | 3+封装 | `trainers/utils.py:301`、`rlhf_trainers/utils.py:423,447`、`tuners/base.py:695`（**带设备索引参数逐卡求和**） | state_query |
| `memory_reserved` | 2+ | `rlhf_trainers/utils.py:424,448`、`trainers/utils.py:302` | state_query |
| `max_memory_allocated` / `max_memory_reserved` | 2+1 | `rlhf_trainers/utils.py:425,449`、`callbacks/print.py:63` | state_query |
| `reset_peak_memory_stats` | 1 | `rlhf_trainers/utils.py:461` | behavior（重置峰值计数） |
| `mem_get_info` | 2 | `model/patcher.py:518`（max_memory 初始化）、`rollout_mixin` | state_query |

### 1.4 RNG 状态（checkpoint 正确性）

| API | 频次 | 调用点 | 三类归属 |
|---|---|---|---|
| `torch.cuda.get_rng_state` | 1 | `megatron/utils/megatron_lm_utils.py:107`：checkpoint 存 `cuda_rng_state` | state_query |
| `torch.cuda.set_rng_state` | 1 | `megatron_lm_utils.py:599`：resume 恢复 | behavior |
| `torch.cuda.random.get_rng_state_all` / `get_rng_state` | 1+1 | 同文件 | state_query |

xmegatron_ext 的 `CudaRNGStatesTracker_wrapper`（XPU RNG 语义对齐 CUDA）证明这不是冷门路径——**分布式 checkpoint resume 必踩**。

### 1.5 能力查询

| API | 频次 | 调用点 | 用途 |
|---|---|---|---|
| `get_device_capability` | 1 | `tuners/longlora/llama.py:399` | 判定 sm>=80 决定 flash-attn 可用性——**kernel 特性选择器**，非 CUDA 平台需要映射到等价能力 key |

### 1.6 环境变量与通信库（B 类盘点面实例）

| 目标 | 调用点 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` 读/写 | `cli/utils.py:7,13`、`utils/env.py:88`、`ui/llm_train.py:463` | ray 路径已有 `ASCEND_RT_VISIBLE_DEVICES` 分支（`resource_pool.py:119`、`vllm_server.py:63`、`replica.py:177`）——**env 分支化已经开始，又是 if/elif** |
| `NCCL_CUMEM_ENABLE=0` | `ray/megatron/worker_group.py:404`、`checkpoint_engine/nccl.py:206` | NCCL 行为开关，非 NCCL 栈需要等价物或 no-op |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | `arguments/sft_args.py:329-330` | SFT 默认强制设置——xmegatron 强制 `CUDA_DEVICE_MAX_CONNECTIONS=1` 的同类 |

### 1.7 模块/张量级 CUDA 硬编码（C 类盘点面实例）

| 形态 | 调用点 |
|---|---|
| `m.cuda(torch.cuda.current_device())` | `megatron_lm_utils.py:652`、`rollout_mixin.py:226`（teacher_model）、`ray/megatron/model_worker.py:44` |
| `torch.ones/empty/tensor(..., device=torch.cuda.current_device())` | `megatron/utils/utils.py:214-221`、`parallel_utils.py:14,24`、`trainers/base.py:1043`、`megatron_lm_utils.py:98` |

`.cuda()` 不是 `torch.cuda.*` 符号，是 `nn.Module` 方法——盘点面的 C 类（框架内部对象方法），替换语义与 A 类完全不同。

## 2. 对照 demo v9 盘点面的缺口清单

demo 现有 9 符号：`set_device/synchronize/range_push/current_device/
memory_allocated/device_count/can_device_access_peer/graph_capture/
profiler_start`(+default-deny)。打通一个 swift 模型（SFT 主链路 +
checkpoint save/resume）还需补齐：

### P0 —— 不补就跑不通/会算错（✅ 已全部落地，demo v10 + real smoke）（**已全部落地，demo v10**）

| 缺口 | 归属 | 建议决策 | 理由 |
|---|---|---|---|
| `is_available` | state_query | remap → `interface.is_available`（普通转发） | 分支选择器，10 处调用；masquerade 语义由双身份机制兜底 |
| **stream 对象族**（`Stream`/`current_stream`/`stream` ctx/`wait_stream`） | behavior | remap → Platform Manager stream 门面 | megatron 路径 ddp fencing 必触发；demo 完全没有 stream 对象模型，**是当前最大结构性缺口** |
| RNG state 三件套（`get/set_rng_state`、`get_rng_state_all`） | state_query+behavior | remap → adapter 的 RNG 语义对齐 | checkpoint save/resume 正确性；xmegatron 已证明 XPU 需要对齐 |
| `current_device` 的 **device 标识语义** | state_query | 契约澄清：返回值必须能被 `torch.tensor(device=)` 消费 | 17 处调用里至少 6 处当 device 用；adapter 契约文档化 + conformance 加测 |
| `empty_cache` | behavior | remap（语义=尽力释放，允许 no-op 化） | rollout/转换路径高频 |
| `mem_get_info` | state_query | remap → (free, total) 元组契约 | `model/patcher.py:518` 的 max_memory 初始化依赖 |

### P1 —— 主链路能跑但观测/RL 残缺

| 缺口 | 归属 | 建议决策 |
|---|---|---|
| `memory_reserved` / `max_memory_allocated` / `max_memory_reserved` / `reset_peak_memory_stats` | state_query+behavior | remap；峰值统计无对应的平台 fallback 到 allocated 口径并在 provenance 记录口径降级 |
| `is_initialized` | state_query | remap |
| `ipc_collect` | behavior | remap，平台无 IPC 时 fallback no-op |
| `get_device_capability` | state_query | remap → 等价能力查询接口（不是返回 sm 元组，是"支持哪档 kernel 特性"的映射） |
| `_set_allocator_settings`（expandable_segments） | behavior | **fallback no-op + trace**——私有配置的降级样板 |

### P2 —— 盘点面机制扩展（超出 torch.cuda.* 符号表）

| 缺口 | 盘点面分表 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` / `NCCL_*` env 切面 | B 类（符号×key） | swift ray 路径已自建 ASCEND 分支；bridge 应收编为 env registry + key 级策略 |
| `Module.cuda()` / `device=current_device()` 用法 | C 类（对象方法 + 语义契约） | 不做符号替换的话，至少要 conformance 验证"adapter 返回的设备标识能被 tensor 构造消费" |
| `pynvml`/`nvidia-smi`/procfs 物理设备发现 | 厂商工具链抽象 | `get_physical_device_count` 需要 adapter 提供"绕过 mask 的物理卡数"等价物，或 fallback 到 runtime count |

## 3. 结论

1. swift 主链路（SFT + checkpoint + rollout）的 CUDA 痕迹约 **20 个符号、
   6 个语义族**；demo 已覆盖 1.5 族（同步 + 部分显存查询）。
2. 最大的两个结构性缺口：**stream 对象模型**（P0，megatron ddp fencing
   与一切 overlap 的地基）和 **RNG state 存取**（P0，checkpoint resume
   正确性）。
3. swift 自带的 `torch_utils` 分支抽象和 `npu_patch` 目录证明：没有统一
   Bridge 时，平台适配 = 分支膨胀 + 裸 patch 堆积。我们的盘点面（四分表）
   + Platform Manager 门面 + 双身份正好收编这三类混乱。
4. 下一步候选：~~demo 补 stream 门面 + RNG state 两个 P0 族~~（已完成，
   demo v10 第 11 节基础 SFT 流程实战通过）；再把 `bindings.py` 扩到
   20 符号规模（P1 族）验证真实盘点面下 default-deny 的工作量。

## 4. 真实环境适配：mock → real 复查出的新坑（real_sft_smoke.py 落地）

P0 接口族在 mock demo 验证后，移植真实 torch 时又暴露出四个 mock
单模块/假硬件永远察觉不到的坑，已全部修复并沉淀进机制：

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| G9 | **绑定落点≠定义模块**：`torch.cuda.Stream/mem_get_info/get_rng_state` 等定义在 `torch.cuda.streams/memory/random` 子模块再 re-export；按 `symbol.__module__` 绑定只改了子模块，框架侧 `torch.cuda.X` 仍指原实现（14 条 policy 只生效 8 条，mem_get_info 直撞真实 CUDA 报 AssertionError） | mock vendor_cuda 是单模块，无 re-export 形态 | `Entry.bind_module` 显式指定绑定落点（框架可见路径），default-deny 分组同步 |
| G10 | **adapter 直调 `mod.X` 会无限递归**：绑定后 `torch.cuda.synchronize` 已是门面，adapter 再调就是 门面→adapter→门面 | mock adapter 用独立 hw 对象，不经过被 patch 模块 | 真实 adapter 全部经 `_orig(name)` 走 `_origin_` 留痕（绑定机制顺带产物成为官方逃生通道） |
| G11 | **能力类绑定的"说谎"半径**：cpu-dev 上把 `torch.cuda.is_available` 绑成 True，AdamW 的 graph capture 健康检查经 `torch.accelerator` 走到 dummy cuda 直接炸 | mock 没有"torch 内部也消费这些符号"的维度 | 教训是"硬件在场"类符号的回答会误导 torch 内部守卫。2026-10-14 起它是普通转发接口（无能力缓存表、无 overlay 剔除），治理方向是让平台如实回答；masquerade 栈 shim 真实可工作，无此问题 |
| G12 | 门面签名必须是 torch 真实签名超集（`synchronize(device)` 等可选 device 参） | mock 调用点全部无参 | 契约/门面/adapter 全链路 `device=None` |
| G13 | `get_rng_state(None)` 在 torch 2.9 会炸（内部直接 `device.index`）——None=当前设备必须**省略实参**，不能位置透传 None | KL3 真机 conformance 首跑 Traceback 实锤（2026-10-12） | adapter 内 `fn() if device is None else fn(device)` |
| G14 | **绑定的爆炸半径不止于框架调用点**：被绑符号所在模块的**原实现内部**也引用同模块符号。`torch.cuda.current_stream` 的全局名空间就是被改过的 `torch.cuda.__dict__`，其内部 `Stream(stream_id=..., device_index=..., device_type=...)` 落到门面 → `TypeError: Stream.__new__() got an unexpected keyword argument 'stream_id'`；放行 TTStream 则更隐蔽（`'TTStream' object has no attribute 'record_event'`） | mock vendor_cuda 的函数内部不互相构造对象；G12 只覆盖了"框架调用点的签名"，没覆盖"原实现内部的构造签名" | **G12 原则推广到构造型符号**：interface.Stream 签名取厂商构造签名超集（`**kwargs`），无参/带参都原样转发给原生 Stream 类，**两者都返回厂商原生对象**（门面不包装；2026-10-13 移除 TTStream：Bridge 只管入口路由，返回对象用各平台 adapter 原生，上层不主动管理）。admission `vendor_internal_construction` 守门（判据=门面返回值与 adapter 契约同型） |

**G14 的一般化结论**：凡 remap 的符号是**类/构造器**（Stream、Event、Graph…），
都要检查"该模块的原实现内部是否用它构造对象"。这类调用的身份判据是**调用者**
（带厂商私有构造参数的只可能是厂商栈自己），不是参数形态；放行原则是原样回落，
不拿门面冒充厂商内部数据结构。

**推论（2026-10-13 移除 TTStream 后确立）**：构造型符号的门面只做**入口路由**，
不包装返回对象——框架与厂商内部拿到的都是 adapter 返回的原生对象，对象的方法
形态（`wait_stream` 等）由各平台保证。这消除了"自定义包装流入厂商内部被当原生
对象用"的整类风险（G14 的第二失败形态 `'TTStream' object has no attribute
'record_event'` 因此从机制上不复存在）。

### 真实环境校准结果（2026-10-12 KL3 机器实测）

1. **masquerade 探测通道已闭环**：真实卡名是通用的 `"gpu"`，`_XPU_MARKERS`
   卡名标记法不可靠；改用**厂商包存在性**作为 native 探测通道
   （`_XPU_NATIVE_PACKAGES=("torch_xmlir",)`——该栈预装 torch_xmlir 并由此
   加载 libbkcl/XCCL）。同时 `TorchCudaPlatform.detect` 增加防御：厂商包在场时
   cuda claim 直接不成立（masquerade 防护双保险，不再只依赖降级机制）。
2. **附带观察**：KL3 栈自带 SYMBOL_REWRITE（torch/deepspeed/transformers）与
   xpu_fla triton patch——厂商栈有自己的 monkey-patch 层，TT cuda-compat 是
   叠加其上的第二层，绑定顺序须保证厂商栈先就位（import torch 时它已生效）。
3. mock 双 demo 在该环境完整跑通（含全部 conformance/admission/red 用例），
   证明 Bridge/Provider 机制与真实厂商栈共存无冲突。

### 真实 XPU 环境执行清单（real_sft_smoke.py）

```bash
cd examples/bridge-demo
python real_sft_smoke.py                    # 自动检测 cuda / xpu-masquerade / xpu-native
python real_sft_smoke.py --platform cpu     # 无卡机器 dev 验证
# 判定：退出码 0 = admission 全 PASS + fp32/bf16 loss 收敛 + fp32 对 CPU 参考 rel_err<5e-3
```

待真实环境校准（本地 CPU 路径已验证之外）：
1. ~~masquerade 卡名标记命中~~ **已闭环**（见上节：改用 torch_xmlir 厂商包探测）；
2. native xpu 栈 `mem_get_info/get_rng_state` 是否存在——不存在则调用时抛
   unsupported，conformance 对应测试自动跳过（看 skipped 行）；
3. `TorchCudaPlatform.sdk_range=("11.0","13.5")` 对定制栈可能需放宽
   （KL3 报 `torch=2.9.0+cu129`，当前取值范围内）；
4. bf16 vs fp32 的 rel_err 参考值（CPU 实测 7.2e-3，仅打印不卡阈值）。

## 5. 真实模型穿透：Qwen3.5-0.8B 脱离 xswift 跑通（2026-10-12，real_qwen_smoke.py）

开发方针是**穿透优先、不铺面**：完备性现阶段放低，未纳入 policy 的 API
直接报未实现（default-deny/exception），先把一条端到端链路打穿。

第一个穿透点：**不经 ms-swift / megatron / mcore_bridge，纯 transformers +
TT Bridge 跑通真实 Qwen3.5-0.8B 的 fwd/bwd/optimizer step。**

| 事实 | 证据 |
|---|---|
| xswift_ext/xmegatron_ext 由环境变量控制，无需删包改 `.pth` | `XSWIFT_PLUGIN_ENABLE=0` 即不激活（`auto_init._is_enabled()` 默认 "0"）；实测 `xswift_ext.entry`/`swift`/`megatron`/`mcore_bridge` 全部 `参与=False` |
| 厂商基座必须保留，且与 TT 共存无冲突 | `torch_xmlir` 经 `torch_xmlir.pth` 的 `hook_torch()` 加载，`参与=True`；XCCL/SYMBOL_REWRITE 照常 |
| transformers 5.12.1 **内置**支持 `qwen3_5`，无 remote code | 模型目录无 `.py`；`AutoConfig` 直出 `model_type=qwen3_5` |
| Bridge 绑定被真实框架消费 | `torch.cuda.{synchronize,current_stream,mem_get_info,get_rng_state}` 全部指向 `ttorch.platform.interface.*` |
| 功能穿透成立 | 单 batch 过拟合 `loss 15.6669 -> 0.0002`（20 步）；admission 14/14 PASS |

**本轮确立的真实接入形态（自造替身察觉不到的）**：

1. **import 时序是硬约束**（§1.2）：`transformers` 必须在 `cuda_compat.activate()`
   **之后**才 import，否则框架 from-import 持有绑定前的陈旧引用。落地形态是
   把 `from transformers import AutoModelForCausalLM` 放进函数体而非模块顶部。
2. **扫尾白名单 = 真实框架包**：`sweep_packages=["transformers", "fla"]`
   （对照 mock demo 的 `None` 全量扫尾、`real_sft_smoke` 的 `framework_real`）。
3. **混合架构已跑通**：Qwen3.5 的 `linear_attention`（fla/GatedDeltaNet 系）+
   `full_attention` 交替层，24 层 0.752B。fla 在该栈回落 CPU（厂商 `xpu_fla`
   patch 的 triton noop 所致，仅 warning），**功能可达成立**——性能是 Provider
   面的独立课题，不阻塞功能穿透。
4. `attn_implementation="eager"` 固定：`flash`/`sdpa` 后端在厂商栈的可用性属
   Provider 性能面，不进功能穿透的范围。

## 6. 第二个穿透点：`bash run.sh` 经 TT 跑通 swift-megatron（2026-10-12）

目标达成：**原训练脚本一行不改**（仅 env.sh 把两个开关改成可覆盖的默认值），
加三个环境变量即由 TT 顶替 xswift_ext/xmegatron_ext 跑完整 SFT。

```bash
TT_PLUGIN_ENABLE=1 XSWIFT_PLUGIN_ENABLE=0 XME_PLUGIN_ENABLE=0 bash run.sh
```

| 项 | TT 模式 | 对照（原 xswift） |
|---|---|---|
| 退出码 | **0** | 0 |
| 完成 step | **43/43** | 43/43 |
| checkpoint | 已保存 | 已保存 |
| 插件 | XSwift/XME Plugin Init **0 处** | 均 Init |

**loss 偏差性质判别**（按 crosshw-numeric-divergence-triage 的口径）：
43 步 Δ 的正号 24 / 负号 19、均值 +2.4e-05、平均 |Δ| 3.7e-04（相对量级 2e-04）
——**零均值、符号近似对半**，是规约顺序噪声而非系统性 bias，符合"两套不同
patch 实现同一语义"的预期。

### 本轮暴露并修复的缺口

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| **G15** | `TypeError: expected a torch.ByteTensor, but got bytes`，死在 `initialize_megatron -> set_random_seed` | **RNG state 契约定错**：我们定为"不透明可序列化对象（bytes）"，但 megatron 的 RNG tracker 把 `torch.cuda.get_rng_state()` 的返回值**原样回喂 torch 原生 `Generator.set_state()`**（`tensor_parallel/random.py:289` 取、`:164` 喂），中途不经我们的 `set_rng_state`——状态对象会**流出 TT 边界** | 契约改为 **torch 原生 ByteTensor**（兼容读旧 bytes）。一般化结论见下 |

**G15 的一般化结论**：**门面的返回值类型必须与被替换符号一致**。这是
G12「签名必须是真实签名超集」在**返回值**上的对应面——G12 管入参（谁来调），
G15 管出参（结果流向谁）。判别方法：问"这个返回值会不会被框架交给
TT 管不到的代码消费？"会，就不能用自定义表示。RNG state、Stream、Event、
Generator 都属此类；纯数值（`memory_allocated` 的 int、`mem_get_info` 的
tuple）不属此类，可以自由换算口径。

### 本轮验证的非缺口（避免误判）

- **`SEQ_LEN=2048` 下 `get_rope_index` 崩**（`expanded size (2048) must match
  existing size (802)`）：对照实验证明**原 xswift 配置同样崩在同一行**，
  是脚本参数与样本长度不匹配，与 TT 无关。默认 `SEQ_LEN=20480` 下两边都正常。
  —— 教训：接框架后的每个报错都要先做开关对照，再归因。

### 仍未覆盖（按方针不铺开）

- 多卡 / 分布式（环境不支持）；
- `default_deny=True` 尚未开启：当前 14 符号够跑通本链路，但"未盘点即报错"
  的白名单姿态还没立起来，缺口靠探针实测而非机制拦截；
- 实测仍被调用但未纳入 policy 的符号：`is_initialized`（torch `_lazy_init`）、
  `is_current_stream_capturing`（AdamW graph capture 健康检查）。

### 下一个穿透点候选（仍是一次只打一个）

| 候选 | 价值 | 代价 |
|---|---|---|
| 真实数据 + tokenizer（替合成 ids） | 补齐 SFT 数据面的 CUDA 痕迹（packing 的 `is_initialized` 等） | 低 |
| checkpoint save/resume（真实权重） | §1.4 "最容易后期暴露"的一环，RNG 契约已就位 | 中 |
| 单机 2 卡 DP（c10d + XCCL） | B 类 env 切面（`BKCL_*`），分布式是 swift 主链路前提 | 中高 |
| 接回 ms-swift（关 xswift_ext，用 TT 顶替） | 直面 §1 的 20 符号 6 语义族全集 | 高 |

### 遗留（不阻塞单卡 SFT smoke，按优先级）

- `Module.cuda()`/`torch.device("cuda")` 字面量（C 类面）：当前过渡形态是
  框架用 `device_spec()` 门面构造设备；正式解是 WRAP 决策落地后包
  `nn.Module.cuda`/`torch.device` 构造路径；
- 分布式/XCCL env 切面（多卡 SFT 前必做，B 类面）；
- provider 的 tensor 级 probe（接真实 kernel 时替换 list probe）。
