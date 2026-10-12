# ms-swift 原生框架 CUDA 接口盘点 × Bridge 补齐缺口分析

> 数据源：`/Users/vector/code/train/ms-swift`（main@21ada4620，528 个 py 文件）。
> 方法：全仓枚举 `torch.cuda.*` / `.cuda()` / CUDA 相关环境变量 / 厂商工具链调用，
> 按我们的三类接口（behavior / state_query / capability）与三种决策
> （remap / fallback / exception）归类，对照 demo v9 的 9 符号盘点面找缺口。
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
| `torch.cuda.is_available` | 10 | `rollout.py:406`（`f'cuda:{rank}' if ... else 'cpu'`）、`activation_cpu_offload.py:16`、`trainers/mixin.py:698` | capability | **分支选择器**：结果直接决定设备字符串形态。masquerade 平台上答案与 vendor 身份脱钩（GCU 报 True） |
| `torch.cuda.current_device` | 17 | **关键：`megatron/utils/utils.py:214-221`、`parallel_utils.py:14,24`、`trainers/base.py:1043` 把它当 `device=` 参数直接传给 `torch.tensor/empty`** | state_query | 返回值被当**设备标识**用（不只索引）。非 CUDA 平台上 `torch.tensor(device=<int>)` 语义直接断裂——这是最容易漏的隐性契约 |
| `torch.cuda.set_device` | 2+封装 | `torch_utils.py:165`、`infer_engine/utils.py:393` | behavior | 标准 |
| `torch.cuda.device_count` | 1+封装 | `torch_utils.py:172` | capability | 标准 |
| `torch.cuda.is_initialized` | 1 | `dataset/packing.py:29` | state_query | lazy-init 判定，packing 的 device 放置前提 |
| `get_physical_device_count` 内部 | 1 处函数 | `torch_utils.py:187-244` | capability | **直接调 pynvml / 读 `/proc/driver/nvidia` / 起 `nvidia-smi` 子进程**——厂商工具链硬依赖，绕过 mask 查物理卡数 |

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
| `is_available` | capability | remap → capability `device_available` | 分支选择器，10 处调用；masquerade 语义由双身份机制兜底 |
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
| `get_device_capability` | capability | remap → capability 等价 key（不是返回 sm 元组，是"支持哪档 kernel 特性"的映射表） |
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
   + Platform Manager 门面 + capability 双身份正好收编这三类混乱。
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
| G11 | **能力类绑定的"说谎"半径**：cpu-dev 上把 `torch.cuda.is_available` 绑成 True，AdamW 的 graph capture 健康检查经 `torch.accelerator` 走到 dummy cuda 直接炸 | mock 没有"torch 内部也消费这些符号"的维度 | 平台 overlay 最小实例：torch.cuda 为非功能桩的栈剔除能力类绑定，框架语义由 `platform_mgr.capability()` 直接供给；masquerade 栈无此问题（shim 真实可工作） |
| G12 | 门面签名必须是 torch 真实签名超集（`synchronize(device)` 等可选 device 参） | mock 调用点全部无参 | 契约/门面/adapter 全链路 `device=None` |
| G13 | `get_rng_state(None)` 在 torch 2.9 会炸（内部直接 `device.index`）——None=当前设备必须**省略实参**，不能位置透传 None | KL3 真机 conformance 首跑 Traceback 实锤（2026-10-12） | adapter 内 `fn() if device is None else fn(device)` |

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
cd tt-bridge-demo
python real_sft_smoke.py                    # 自动检测 cuda / xpu-masquerade / xpu-native
python real_sft_smoke.py --platform cpu     # 无卡机器 dev 验证
# 判定：退出码 0 = admission 全 PASS + fp32/bf16 loss 收敛 + fp32 对 CPU 参考 rel_err<5e-3
```

待真实环境校准（本地 CPU 路径已验证之外）：
1. ~~masquerade 卡名标记命中~~ **已闭环**（见上节：改用 torch_xmlir 厂商包探测）；
2. native xpu 栈 `mem_get_info/get_rng_state` 是否存在——不存在会自动
   声明 capability=False 并降级，看 conformance 的 skipped 行；
3. `TorchCudaPlatform.sdk_range=("11.0","13.5")` 对定制栈可能需放宽
   （KL3 报 `torch=2.9.0+cu129`，当前取值范围内）；
4. bf16 vs fp32 的 rel_err 参考值（CPU 实测 7.2e-3，仅打印不卡阈值）。

### 遗留（不阻塞单卡 SFT smoke，按优先级）

- `Module.cuda()`/`torch.device("cuda")` 字面量（C 类面）：当前过渡形态是
  框架用 `device_spec()` 门面构造设备；正式解是 WRAP 决策落地后包
  `nn.Module.cuda`/`torch.device` 构造路径；
- 分布式/XCCL env 切面（多卡 SFT 前必做，B 类面）；
- provider 的 tensor 级 probe（接真实 kernel 时替换 list probe）。
