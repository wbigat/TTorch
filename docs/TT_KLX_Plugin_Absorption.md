# KLX 插件（xswift_ext / xmegatron_ext）分析与 TT 吸收清单

> 静态只读分析，未在 XPU 上运行。规模：xswift_ext 30 文件 ~2100 行，xmegatron_ext 39 文件 ~3900 行。
> 结论先行：**这套插件实际上是一个"原型 TT"**——它已经独立发明了 cuda_compat 的绑定机制、feature gate 和一部分 provenance，但缺少准入测试、capability probe、compute/schedule 分离和运行时回退。

---

## 1. 机制层：它已经做对的四件事

两个仓库共用同一套骨架（文件名完全对称），这套骨架本身比里面的业务改动更值得吸收。

### 1.1 `symbol_rewriter.py` —— cuda_compat 绑定机制的现成原型

`Symbol.rewrite()`（`xswift_ext/symbol_rewriter.py:156-195`）做了两件关键的事：

```python
setattr(original_module, f"_origin_{name}", original_symbol_impl)   # L179 保留原实现
setattr(original_module, name, finalized_symbol_impl)               # L180
for module in sys.modules.copy().values():                          # L181-188
    if id(getattr(module, name, None)) == id(original_symbol_impl):
        setattr(module, name, finalized_symbol_impl)                # 连 from X import Y 的旧引用一起改
```

- **全局引用重绑定**（L181-188）解决了 `from X import Y` 拿到旧引用的经典问题。
- **`_origin_` 保留**让 wrapper 可以回退到原实现——这正是 TT 需要的 fallback 基础。
- `is_wrapper=True` 支持多个 wrapper 按注册顺序叠加（L171-176）。
- `use_dummy_module`（L51-87）允许在目标库未安装时也能注册（合成占位模块）。

**对 TT 的意义**：`cuda_compat.binding` 不必从零设计，这个 rewriter 可以直接作为参考实现。
**但要补**：它按 `id()` + 同名匹配，同名不同义的符号会误伤；且**没有记录"为什么替换"**。

### 1.2 `entry.py` 的 `_RewriteSpec` —— policy 表的雏形

```python
@dataclass(frozen=True)
class _RewriteSpec:                      # xmegatron_ext/entry.py:56-64
    name: str
    module_name: str          # 目标模块，不可导入则跳过
    register_name: str
    version_key: str          # 版本键
    version_checker: object   # 版本校验函数
    xpu_only: bool = False    # 硬件门控
    env_flag: str = None      # 环境变量门控
```

`_rewrite_skip_reason()`（L331-338）返回**结构化的跳过原因**，`_build_rewrite_plan()`（L370-379）产出 enabled/skipped 两张表。这已经非常接近我们定义的 policy 表。

### 1.3 启动摘要 —— provenance 的雏形

`_print_init_summary()`（`xmegatron_ext/entry.py:465-484`）会打印：Megatron 版本、XPU 模式、**每个组件 enabled/skipped 及原因**、全部环境变量、**全部被重写的符号列表**（按 `os.` / `megatron.core.` / `megatron.training.` / `megatron.bridge.` 排序）。

这正是我们要的「可打印、不隐藏 fallback」。**缺的是**：没有 rejected 候选的收益证据，没有 capability 快照，不可导出成机器可读格式。

### 1.4 硬件判定完全外提

**`components/` 下 70 个文件里没有一处 `if is_xpu()`。** 判定只在两个 `entry.py`：

```python
def is_xpu():                                    # xmegatron_ext/entry.py:257-264
    if os.getenv('XPU', "").lower() in ['1','true']: return True
    if _has_component('torch_xmlir'): return True
```

这是整套插件**架构上最好的决策**，与我们定的"运行期不做硬件判断"方向一致。
**但要改**：判据是「能否 import 厂商库」，不是 capability probe。换硬件要改这个函数，而不是加一列 capability。

---

## 2. 全量功能清单与动机分类

### (a) 功能性兼容——不改就跑不起来或跑错

| 位置 | 内容 | 是否硬件相关 |
|---|---|---|
| `xmegatron/core/optimizer.py:89` | 排序 key 加 `x[1]`(is_expert_parallel)，修 MoE+PP 下各 rank param group 顺序不一致 | **否，是上游 bug**（所以注册在 common，GPU 也打） |
| `xswift/.../megatron_lm_utils.py:77-93` | DDP 侧流双向 `wait_stream` 围栏，修写-写/读-写竞态 | **否，上游 Swift 4.3.x bug**，XPU 更易触发 |
| `xswift/swift/loss/embedding.py` | gather 前把 tensor 下沉 CPU，避免 CUDA storage 跨 rank 反序列化 | 部分 |
| `xswift/swift/megatron/dcp.py:10-18` | `inspect.getsource` 探测 pytorch#176728 契约变更，一个 stub 兼容新旧 torch | **否，torch 版本漂移** |
| `xmegatron/training/arguments.py:63-74` | 替换 `os.environ.get`，让 `CUDA_DEVICE_MAX_CONNECTIONS` 恒返回 "1" | 是 |
| `xmegatron/core/distributed/param_and_grad_buffer.py:22` | `proteus.mark_flat_owner` 消除 bf16 swap 的 double-offset | 是 |
| `xmegatron/core/tensor_parallel/random.py` | RNG tracker 改用 torch_xmlir mock state，对齐 CUDA 初始化序列 | 是 |
| `xswift/swift/model/conv3d.py` | 无条件打开 Swift 的 Conv3d unfold 实现（XPU 无原生 Conv3d kernel） | 是 |
| `xswift/mcore_bridge/patcher.py:110-115` | THD RoPE 补回丢失的 CP group | 否 |

### (b) 性能优化

| 位置 | 内容 |
|---|---|
| `xmegatron/core/tensor_parallel/layers.py` | **TP Linear 融合 + 反向 dgrad/wgrad overlap**（见 §4.1） |
| `xmegatron/core/fusions/fused_bias_swiglu.py` | SwiGLU 融合算子，含空张量与 weights.float() 处理 |
| `xmegatron/core/models/.../rope_utils.py` | RoPE 融合算子，`mscale != 1.0` 回退上游公式 |
| `xswift/.../trainers/base.py:255-395` | max_seqlen pad warmup（shape 预热） |
| `xmegatron/components/bridge/**` | 转换期内存优化：多卡 spill、流式保存、no-op cuda |

### (c) 正确性回退——牺牲性能换正确 ⚠️

| 位置 | 内容 | 代价 |
|---|---|---|
| **`xmegatron/core/transformer/moe/fused_a2a.py`** | FusedDispatch/Combine 的 forward/backward 前后各插 `torch.cuda.synchronize()`，**共 8 处**（L38,40,46,48,63,65,71,73） | **DeepEP 的 overlap 收益被完全抹除** |
| `xmegatron/core/.../filesystem_async.py:52-53` | 强制 `non_blocking=False`，警告"D2H might fail" | checkpoint 保存变慢 |
| `xswift/common/utils.py` + `xmegatron/core/utils.py` | eval 期间关闭 HYDRA/XMLIR fast FC | eval 变慢 |
| `xmegatron/core/utils.py:28` | **死代码**：`return None` 直接跳过 eval，注释 "tmp avoid kernel crash" | — |

### (d) 生态/格式适配

`checkpointing.py`(643行，XTE large_weight ⇄ per-expert 双向转换 + LoRA)、`extensions/transformer_engine.py`、两份 `sft_tokenizer`、`bridge/training/tokenizers/*`、`print.py`(吞吐监控)、`base.py:37-252`(ProfilerCallback 注入)。

### (e) 关键分层结论（2026-10-15 实测确认）

**性能 kernel 的物理本体全部在 torch_xmlir**（`torch.ops.custom_ops.*`，来自
`libxmlir_custom_ops.so`/`libxpu_dnn.so`）。xmegatron_ext 的"融合算子"
（SwiGLU/RoPE/MoE a2a）是把 torch_xmlir kernel 包成 autograd.Function/megatron
层的**封装**；xswift_ext 则是纯框架集成（trainer/checkpoint/lora/数据/conv3d），
无一行自有 kernel。所以"xswift 不做性能优化"成立；"性能全在 torch_xmlir"要修正为
"**kernel 在 torch_xmlir，megatron 融合层封装在 xmegatron_ext，swift 集成在
xswift_ext**"。两个开关（`XSWIFT_PLUGIN_ENABLE`/`XME_PLUGIN_ENABLE`）只门控这两层
框架集成（`.pth` auto-init + 惰性 import 钩子），不管 torch_xmlir 本体（它永远开）。

**torch_xmlir 自己在 eager/transformers 层也做符号替换**（`symbrewrite/plugins/
transformers`）：Qwen2/Qwen3/Qwen3.5/Gemma4 等的 `*RMSNorm`、`apply_rotary_pos_emb*`
已被换成厂商融合实现（RMSNormFunction/FusedRoPEFunc，同源 custom_ops kernel）。
**这是 eager 路径上 RMSNorm/RoPE 已是融合态的原因**——Provider 在这类算子上的增量
价值只剩机制（门禁/回退/provenance/开关），净性能持平（见 TT_Provider_Function_Map.md
§6.2）。**Provider 选题前先查这份符号替换清单，避免在厂商已融合的算子上重复。**

---

## 3. Bridge 应当吸收什么

### 3.1 直接吸收机制（高价值）

| 吸收项 | 来源 | 进入 TT 的位置 |
|---|---|---|
| 符号重写引擎（含全局引用重绑定、`_origin_` 保留、wrapper 叠加） | `symbol_rewriter.py` | `cuda_compat.binding` |
| `_RewriteSpec` 结构化门控 + skip reason | `entry.py:56-64, 331-338` | `cuda_compat.policy` 骨架 |
| 启动摘要（组件状态 + 环境 + 重写符号清单） | `entry.py:465-484` | `diagnostics` / provenance |
| 硬件判定外提、组件内零 if | 全局设计 | 确认我们的"运行期不做硬件判断"约束可行 |
| 版本漂移只告警不阻断 | `entry.py:205-219` | 但 TT 要加准入测试作为闸门 |
| lazy import hook（首次 import megatron 时才激活） | `auto_init.py:101-136` | `bootstrap` 激活时机 |

### 3.2 吸收为 cuda_compat 的 policy 条目

把这些现成的决策直接填进 policy 表（补上 `intent` 和准入测试）：

```yaml
- symbol: os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")
  intent: Megatron 用它保证 TP overlap 的 kernel 下发顺序；该概念在非 CUDA 设备不存在
  decision: native_fallback        # 不是 map——现实现是伪造返回值
  note: 现实现替换 os.environ.get 方法本身，污染全局；TT 应改为 feature_gate

- symbol: torch.distributed.all_gather_object (含 CUDA tensor)
  intent: 跨 rank 收集对象；CUDA storage 反序列化会按错误 device index 重建
  decision: map -> 下沉 CPU 后再 gather
  test_id: compat.test.gather_object_cuda_storage

- symbol: torch.nn.Conv3d.forward
  intent: XPU 无原生 Conv3d kernel
  decision: native_fallback -> unfold + F.linear
```

### 3.3 吸收为 Bridge 子模块能力

| Bridge 子模块 | 从插件吸收的内容 |
|---|---|
| `distributed_binding` | `megatron_lm_utils.py` 的 DDP 侧流双向 fence（**这是竞态 bug 修复，必须吸收**） |
| `checkpoint_bridge` | `dcp.py` 的契约版本探测模式；`checkpointing.py` 的 schema 校验思路（但不吸收 XTE 特定实现） |
| `feature_gate` | `envs.py` 的 EnvSpec 声明式定义 + 启动打印 |
| `qualification` | `optimizer.py` 的 param group 顺序问题 → 直接变成一条 PP+MoE 的 conformance 测试用例 |
| ShapeWarmupPolicy（数据侧） | `base.py:255-395` 的 pad warmup，但要重做（见 §5） |

---

## 4. Provider 应当吸收什么

### 4.1 `layers.py` → ParallelLinear Provider（**首个 case，直接可用**）

`xmegatron_ext/components/core/tensor_parallel/layers.py:196-199` 的反向：

```python
grad_input, handle = _async_backward_input(ctx, grad_output)    # L196 异步发起 RS/AR
grad_weight = _backward_weight(ctx, grad_output, wgrad_compute) # L197 wgrad 与之重叠
if handle is not None: handle.wait()                            # L198-199 汇合
```

**这与我们 `examples/schedule-demo` 里的 `OverlapTpSchedule` 是同一个模式**，而且它比上游更好：上游靠 `CUDA_DEVICE_MAX_CONNECTIONS=1` 的**隐式 kernel 下发顺序**，这里改成**显式 handle 传递**（也正因此才需要 §2(a) 那个 `os.environ` hack 去骗过 Megatron 的检查）。

拆进 TT 的两个插槽：

| 插槽 | 吸收内容 |
|---|---|
| **compute** | `linear_fwd` / `linear_bwd_dgrad` / `fc_fusion(beta=1.0)` 替代 apex `fused_weight_gradient_mlp_cuda`（该 CUDA extension 在非 NV 硬件不存在） |
| **schedule** | L196-199 的 async handle + wait 模式 → `overlap` plan |

**一个重要的设计抉择要单独评估**：它在 forward 把 **AG 后的 `total_input` 整个存进 `save_for_backward`**（L95），因此 backward 完全不需要再 all-gather（上游的 overlap ① 被删除）。代价是**激活显存 × TP_size**，且不能复用 mpu 全局 buffer（L78 改成 `new_empty`）。**这是用显存换通信，在长序列场景是风险**——应作为 schedule plan 的一个变体，由 capability + shape 决定是否启用，而不是写死。

### 4.2 op 级 Provider

| 来源 | 进入 TT |
|---|---|
| `fused_bias_swiglu.py` | SwiGLU op provider。**务必吸收它的两个正确性细节**：空张量特判（MoE 某 expert 分到 0 token 是常态）、`weights.float()` 升精度 |
| `rope_utils.py` | RoPE op provider。**吸收它的回退模式**：`mscale != 1.0` 时完整回退上游 cos/sin 公式（L53-59），这就是我们要的 reference fallback |

### 4.3 两条互斥加速路线 —— 现成的 compute 插槽证据

`envs.py:73-76` + `entry.py:233-254`：

- `XME_SWIGLU_USE_OKEANOS_DL=0` → 替换成**手写融合算子**
- `XME_SWIGLU_USE_OKEANOS_DL=1` → **不替换**，保留 `@jit_fuser`，并把 `megatron.core.jit.jit_fuser` 换成 `okeanos_jit_fuser` 走**编译器**路径

**这正是我们定义的 compute 插槽 `fused_ops | jit_compile` 二选一**，而且他们已经在生产里这么用了。可以直接作为 Provider registry 的设计依据。

---

## 5. 明确不吸收的部分

| 不吸收 | 位置 | 理由 |
|---|---|---|
| **替换 `os.environ.get` 方法本身** | `arguments.py:63-74` | 全局污染；且 `os.environ[...]` 下标访问不受影响，行为不一致。TT 应走 feature_gate |
| **`gpu_idx < 7` + 4GiB 写死** | `model_bridge.py:92`、`model_provider.py:21`、`mlm_compat/model.py:21` | 写死单机 8 卡；**同一段 31 行代码重复三份**；>8 卡或 <8 卡行为错误 |
| **8× `torch.cuda.synchronize()`** | `fused_a2a.py` | 全设备同步抹除 DeepEP 全部 overlap。TT 应定位 XPU event 语义缺陷本身，或退回普通 A2A |
| **模型架构名写死在注册表** | `xswift_ext/plugin.py:117-121`（`Qwen3_5Vit`/`Qwen3VL_Vit`） | 是 M×N 债务来源。新增 VL 架构要改注册表 |
| **XTE large_weight 的 optimizer state 下标重排** | `checkpointing.py` 643 行 | 整数下标 × 交错 weight/bias × expert 的下标算术，与 XTE 布局强绑定。TT 只吸收 schema 校验思路 |
| **整类替换 + 固定签名** | `layers.py` forward 9 参数 | 本机上游 Megatron-LM 已是 11 参数、Megatron-LM-FL 是 10 参数 → **升级即 TypeError**。TT 必须走 ModuleSpec/Builder seam，不整类替换 |
| **无条件 bf16 落盘** | `filesystem_async.py:230` | fp32 optimizer state 会被静默降精度 |
| **模板不匹配时静默丢 loss** | `sft_tokenizer.py:241,247` | 整条样本 target 置 IGNORE_INDEX 且不报错，训练"安静地"学不到东西 |

---

## 6. 这套插件缺的四样东西（正是 TT 的增量价值）

| 缺失 | 现状 | TT 补什么 |
|---|---|---|
| **准入测试** | 所有替换都是无条件的，没有任何等价性证明 | `cuda_compat.admission_test` 作为闸门：不过闸只能回退或报错 |
| **capability probe** | 用 `_has_component('torch_xmlir')` 判硬件 | 实测 probe：async_comm、event 语义、算子覆盖度 |
| **运行时回退** | 替换后无退路，`not_impl_wrapper` 只会抛 NotImplementedError | reference 永远可用，kill switch 一关回基线 |
| **compute / schedule 分离** | 融合算子与 overlap 时序耦合在同一个 autograd Function 里 | 两个独立插槽，可分开灰度归因 |

**第四条对 `fused_a2a.py` 尤其关键**：如果 compute 与 schedule 是分离的，就可以只关掉 schedule（退回同步 A2A）而保留 DeepEP 的 compute，而不是现在这种"用 8 次全设备同步把整个 DeepEP 废掉"。

---

## 7. 吸收优先级

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | 吸收 `symbol_rewriter` + `_RewriteSpec` + 启动摘要三件机制 | 现成且验证过，直接变成 cuda_compat 骨架 |
| **P0** | 把 §3.2 的 policy 条目填表，补 `intent` + 准入测试 | policy 表是核心资产 |
| **P1** | `layers.py` 拆成 ParallelLinear 的 compute + schedule 双插槽 | 与已有 demo 对齐，首个 Provider 闭环 |
| **P1** | DDP 侧流 fence 进 `distributed_binding`；param group 顺序进 conformance | 两个真实 bug，吸收成本低 |
| **P2** | SwiGLU / RoPE op provider（含空张量、`mscale` 回退两个正确性细节） | 第二批模块语义 |
| **P2** | 复刻 `XME_SWIGLU_USE_OKEANOS_DL` 的互斥路线作为 compute 插槽设计依据 | 生产验证过的 fused_ops ↔ jit_compile 切换 |
| **P3** | 用 compute/schedule 分离重新评估 `fused_a2a` | 需要先有 XPU 环境复现与 profiler 证据 |

**一句话**：这套插件把「怎么绑」做得相当完整（rewriter + 门控 + 摘要 + 硬件判定外提），把「凭什么绑」完全留空（无 intent、无准入、无 probe、无回退）。TT 的增量就落在后半句。
