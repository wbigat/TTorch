# transformers symbrewrite 插件分析：TT 收束层覆盖能力评估

> 分析日期：2026-10-23
> 分析对象：`/Users/vector/code/train/train_klx/torch_xmlir/symbrewrite/plugins/` 下 6 个 transformers 目录 + `swift/` 目录
> 对照体系：TTorch（`/Users/vector/code/train/TTorch` @ `691ab33`）的 Bridge / Platform / Provider 三层
> 参考框架：torch-xmlir symbol_rewrite、xswift/xmegatron_ext、FlagOS、DeepInfer

---

## 0. 结论先行

**39 个生效修改点**中（38 `SYMBOL_REPLACE` + 1 `MODULE_REPLACE`，另有 1 条被注释未生效）：

| 判定 | 条目数 | 占比 |
|---|---|---|
| **TT 现有机制可直接覆盖** | 22 | 56.4% |
| **需补充机制（小改，非新架构）** | 6 | 15.4% |
| **TT 不应覆盖（超出职责边界）** | 8 | 20.5% |
| **已失效 / 仅留痕（无实际行为变更）** | 3 | 7.7% |

**一句话**：transformers 侧的修改**分化程度远高于** xswift/xmegatron——后者几乎全是硬件适配，而 transformers 侧只有约一半是硬件适配，
另一半是**训练策略调优**与**框架版本兼容 patch**。这不是 TT 能力不足，而是**职责边界问题**：把后两类硬塞进收束层会破坏 TT 的定位。

**最重要的一条新发现（修正前一版口头分析）**：`is_flash_attn_2_available` 的 mock **不是**为了绕过 CUDA 信号，
而是为了绕过**包版本元数据缺失**（`importlib.metadata.version("flash_attn")` 抛 `PackageNotFoundError`）。
TT 接管 `torch.cuda.is_available()` **解决不了这个问题**——因为执行顺序上，mock 是先检查 flash_attn 包、再检查 cuda 的。
详见 §3.1。

---

## 1. 分析范围与事实基线

### 1.1 目录与条目分布（实测清点）

| 目录 | SYMBOL_REPLACE | MODULE_REPLACE | 备注 |
|---|---|---|---|
| `transformers/`（无版本号，基线） | 19 | 0 | 面向 transformers 5.x，含 `MODULE_DPES` 声明 9 个模型模块 |
| `transformers_4_32_1/` | 2 | 0 | Baichuan2 社区模型（remote code） |
| `transformers_4_42_3/` | 6 | 1 | 唯一含 `MODULE_REPLACE("torch.optim.AdamW" → FusedAdamW)` |
| `transformers_4_43_3/` | 2 | 0 | 纯能力探测 |
| `transformers_4_49_0/` | 4 | 0 | 能力探测 + Qwen2 |
| `transformers_4_52_3/` | 3 | 0 | 能力探测 + 安全检查 |
| `swift/`（同级，与 transformers 强相关） | 2 | 0 | ms-swift 训练路径 |
| **合计** | **38** | **1** | **39 生效** |

另有 1 条在 `transformers_4_42_3/__init__.py` 中被注释（`BertAttention.__init__` 的 wrapper 版本），不计入。

### 1.2 验证方法（本轮实测，非文档转述）

| 验证项 | 方法 | 结果 |
|---|---|---|
| 条目清点 | Python 正则解析全部 `__init__.py`（剔除注释行） | 39 条，逐条符号路径已列出 |
| 原生实现语义 | `pip download transformers==4.52.3` 解包读**真实源码** | 见 §3 各条 |
| 桩模块行为 | 构造 `sys.modules["flash_attn"]` 桩 + 复现 `_is_package_available` | `find_spec` 抛 `ValueError`（无 `__spec__`）/ 返回 spec（有 `__spec__`） |
| 版本元数据 | `importlib.metadata.version("flash_attn")` | `PackageNotFoundError` |
| 本机 transformers | `transformers.__version__` | **5.19.0**（`is_torch_sdpa_available` / `is_torch_fx_available` 已删除） |

---

## 2. 分类总表

分类依据是**修改动机**（为什么改），不是符号所在模块。同一符号在不同版本里可能因不同动机被改。

| 类 | 动机 | 条目数 | TT 判定 |
|---|---|---|---|
| **A. 算子替换**（RMSNorm / RoPE） | 用厂商融合算子替代 torch 原生小算子链 | 21 | ✅ **Provider 完全覆盖** |
| **B. 能力探测改写**（attention 后端选择） | 让框架选到厂商支持的 attention 实现 | 7 | ✅ **Bridge 覆盖 4 条 / 需补 3 条** |
| **C. 框架安全检查绕行** | 规避版本号检查失败 | 1 | ⚠️ **能覆盖但应留证据不绕行** |
| **D. 训练策略调优** | 提性能的框架/hardware 参数 hack | 4 | ❌ **不应覆盖（超职责）** |
| **E. 模型结构整体替换** | 整段 forward / 整个 Attention 类重写 | 3 | ⚠️ **需 Provider 泛化（小改）** |
| **F. 框架版本兼容 patch** | 定制 torch 与标准 transformers 的接口不兼容 | 3 | ❌ **部分不应由 TT 承担** |
| **G. 无实际行为变更** | mock 恒等于原生，或仅加留痕 | 3 | ⚪ **已失效 / 无操作** |

---

## 3. 逐类详细分析

### 3.1 A 类：算子替换（21 条）— ✅ Provider 完全覆盖

这是 transformers 侧**最大的一块**，也是与 xswift/xmegatron 性质最一致的一块。

#### 条目清单（21 条）

**RMSNorm 类替换（9 条）**——注意：同一个语义在不同模型模块下重复注册了 9 次。

| 目标符号 | 替换函数 | 出现版本 |
|---|---|---|
| `qwen2.modeling_qwen2.Qwen2RMSNorm` | `wrapper_mock_RMSNorm` | 基线, 4_49_0 |
| `qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm` | `wrapper_mock_RMSNorm` | 基线 |
| `qwen3.modeling_qwen3.Qwen3RMSNorm` | `wrapper_mock_RMSNorm` | 基线 |
| `qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm` | `wrapper_mock_RMSNorm` | 基线 |
| `qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm` | `wrapper_mock_zero_centered_gamma_RMSNorm` | 基线 |
| `qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm` | `wrapper_mock_zero_centered_gamma_RMSNorm` | 基线 |
| `gemma4.modeling_gemma4.Gemma4RMSNorm` | `wrapper_mock_Gemma4RMSNorm` | 基线 |
| `llama.modeling_llama.LlamaRMSNorm` | `wrapper_mock_LlamaRMSNorm` | 4_42_3 |
| `bert.modeling_bert.BertAttention`（内含 SDPA 改造） | `wrapper_mock_BertAttention` | 4_42_3 |

**RoPE 函数替换（12 条）**

| 目标符号 | 替换函数 | 出现版本 |
|---|---|---|
| `qwen3.modeling_qwen3.apply_rotary_pos_emb` | `mock_apply_rotary_pos_emb` | 基线 |
| `qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb` | 同上 | 基线 |
| `qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb_vision` | `mock_apply_rotary_pos_emb_vision` | 基线 |
| `qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb` | `mock_apply_rotary_pos_emb` | 基线 |
| `qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb_vision` | `mock_apply_rotary_pos_emb_vision` | 基线 |
| `qwen3_5_moe.modeling_qwen3_5_moe.apply_rotary_pos_emb` | `mock_apply_rotary_pos_emb` | 基线 |
| `qwen3_5_moe.modeling_qwen3_5_moe.apply_rotary_pos_emb_vision` | `mock_apply_rotary_pos_emb_vision` | 基线 |
| `gemma4.modeling_gemma4.apply_rotary_pos_emb` | `mock_apply_rotary_pos_emb_gemma4` | 基线 |
| `qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb` | `mock_apply_multimodal_rotary_pos_emb` | 基线 |
| `qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_vision` | `mock_apply_rotary_pos_emb_vision` | 基线 |
| `qwen2.modeling_qwen2.apply_rotary_pos_emb` | `mock_apply_rotary_pos_emb` | 4_49_0 |
| `llama.modeling_llama.apply_rotary_pos_emb` | `mock_apply_rotary_pos_emb` | 4_42_3 |

#### 替换实现的关键逻辑（读 mock 源码得出的真实行为）

**RMSNorm**：把 `torch.nn.Module` 子类整体换成调用厂商 `RMSNormFunction.apply(hidden_states, eps, weight, ...)`。
参数顺序和零中心语义按模型变体分三种：
- `wrapper_mock_RMSNorm`：直乘 weight（Qwen2/3/Llama/Qwen2.5-VL）
- `wrapper_mock_zero_centered_gamma_RMSNorm`：`zero_centered_gamma=True`，初始化用 `torch.zeros`（Qwen3.5 系）
- `wrapper_mock_Gemma4RMSNorm`：额外 `with_scale` 参数，`with_scale=False` 时无 `weight`，用全 1 张量（Gemma4 的 v_norm）

**RoPE**：
- `mock_apply_rotary_pos_emb`：先尝试 transpose 到 BLHD 且检查 4 个张量 `is_contiguous()`，
  满足才走 `FusedRoPEFunc.apply(..., "BLHD")`，否则回落 `"BHLD"` 布局。**这是显式的性能分支，不是语义分支。**
- `mock_apply_rotary_pos_emb_vision`：unsqueeze(0) 补 batch 维后走 BLHD，再 squeeze 回去。
- `mock_apply_multimodal_rotary_pos_emb`：先按 `mrope_section` 做 3 段交错拼接（temporal/height/width），再走融合。
- `mock_apply_rotary_pos_emb_gemma4`：Gemma4 的 Per-tensor 调用（q 或 k 单独传），`"BLHD"` 布局，返回单张量。

#### TT 覆盖路径与结论

**RMSNorm → `Provider.wrap_model()`（模块级通道）**
- 匹配机制：`module_names` 按类名 `endswith` 匹配，`("RMSNorm",)` 可一次覆盖 8 个变体；
- 变体识别：TT 的 `_self_check` 在 fp32 轨道上对 `w` / `1+w` 两个候选各跑一次实测择优，
  **不依赖 `zero_centered_gamma` flag**（symbrewrite 靠人肉分三个 wrapper，TT 靠实测）——这是 TT 的明确优势；
- 残余差距：`Gemma4RMSNorm` 的 `with_scale=False` 分支**没有 `weight` 参数**，
  TT 的 `wrap_module` 入口 `isinstance(module.weight, Parameter)` 会直接 `raise TypeError` 跳过接管。
  **需要小改**：把 `weight` 缺失时的「用全 1」语义纳入 Provider 的 wrap 约定（约 10 行）。

**RoPE → `fn_binding`（函数级通道）**
- 机制已具备（megatron 已用同一通道绑 `_apply_rotary_pos_emb_bshd`）；
- **只需在 `bind_targets.py` 增一张 `TRANSFORMERS` 表**，把这 12 个坐标登记进去；
- 布局分支（BLHD vs BHLD）正是 TT Provider 的 `supports()` 谓词该表达的内容——可由厂商 impl 自己判断，不需要 TT 干预。

**判定**：A 类 **21/21 可被 TT 覆盖**，其中 20 条零改动、1 条（Gemma4 `with_scale=False`）需约 10 行适配。

---

### 3.2 B 类：能力探测改写（7 条）— ✅ 4 条已覆盖 / ⚠️ 3 条需判断

这类修改是整个分析中**最容易误判**的一块。前一版口头分析说「Bridge 接管 `torch.cuda.is_available()` 后自然解决」——
**对其中 4 条成立，对 3 条不成立**。原因如下。

#### 原生实现 vs mock（基于 transformers 4.52.3 真实源码）

**B-1 `is_torch_sdpa_available`（4_43_3 / 4_49_0 / 4_52_3，3 条）**

原生（4.52.3 实测源码）：
```python
def is_torch_sdpa_available():
    if not is_torch_available(): return False
    elif _torch_version == "N/A": return False
    if is_torch_mlu_available(): return True
    if is_torch_npu_available(): return True
    return version.parse(_torch_version) >= version.parse("2.1.1")
```
mock：
```python
def mock_is_torch_sdpa_available():
    if not is_torch_available(): return False
    return True
```

**行为差异**：原生只检查 **torch 版本号**（≥2.1.1）与 MLU/NPU 特判，**完全不查 CUDA**。
在 KL3 上 `_torch_version` 形如 `2.7.1+hstu` 这样的厂商后缀版本——
`version.parse("2.7.1+hstu") >= version.parse("2.1.1")` 仍为 True，所以原生**通常不会失败**。
mock 的唯一实质作用是**兜底**：万一 `_torch_version == "N/A"`（元数据缺失）或版本号怪到 `version.parse` 抛错，mock 恒真。

**TT 覆盖判定**：**不适用**。这条改动读的是 `torch.__version__` 与 MLU/NPU 探测，**不经 `torch.cuda.*`**，Bridge 无插手点。
但它也**不需要 TT 覆盖**——原生在正常版本号下就是 True。归入「框架版本兼容 patch」（F 类）。

**B-2 `is_flash_attn_2_available`（4_43_3 / 4_49_0 / 4_52_3，3 条）★ 本轮最重要的修正**

原生（4.52.3 实测源码）：
```python
def is_flash_attn_2_available():
    if not is_torch_available(): return False
    if not _is_package_available("flash_attn"): return False
    import torch
    if not (torch.cuda.is_available() or is_torch_mlu_available()): return False
    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    ...
```

**执行顺序是关键的**：`_is_package_available("flash_attn")` 在 `torch.cuda.is_available()` **之前**。

实测两个失败点：
1. `_is_package_available` 里 `importlib.util.find_spec("flash_attn")`：symbrewrite 用 `MODULE_REPLACE` 往 `sys.modules` 注入桩模块，
   若桩无 `__spec__` → **`find_spec` 直接抛 `ValueError: xxx.__spec__ is None`**；有 `__spec__` 才返回 spec（本机实测两种情形）；
2. 通过第 1 关后，`importlib.metadata.version("flash_attn")` → **`PackageNotFoundError`**（桩模块没有 dist-info，本机实测确认），
   而原生代码**没有 try/except**，异常直接冒出。

mock 的实现是 `return torch.cuda.is_available()`——**完全跳过包探测与版本比较**，只保留 cuda 检查。

**TT 覆盖判定**：**Bridge 接管 `torch.cuda.is_available()` 解决不了**。
因为问题出在**更早的一步**（包元数据），而 TT 的能力边界是 `torch.cuda.*` 符号面，管不到 `importlib.metadata`。
这条必须走**函数级替换**（TT 的 `fn_binding` 通道技术上能做：注册 `transformers.utils.is_flash_attn_2_available` → 一个只查 `torch.cuda.is_available()` 的实现）。

但这里引出一个**架构判断**：TT 该不该做这件事？
- 支持做：这是 masquerade 场景的通用形态（桩包替换 → 元数据缺失 → 下游探测炸），TT 作为收束层应该能兜住；
- 反对做：这属于「厂商栈自己的桩包机制没做完整」（`MODULE_REPLACE` 注入的桩应带 `__version__` 与 dist-info），
  严格说是 torch_xmlir/symbrewrite 的缺陷，不是硬件差异。

**建议**：TT 提供机制（`fn_binding` 已够用），但**不主动内置这条替换**——把它作为「记录在案的已知 patch」由平台栈决定是否注册。
代价很低（一条 Entry），且能避免 TT 与厂商栈双重打补丁导致互相覆盖。

**B-3 `PreTrainedModel._check_and_enable_sdpa`（4_42_3，1 条）**

mock 的实质：把整个函数体注释掉，只留 `config._attn_implementation = "sdpa"; return config`——**强制所有模型走 SDPA，跳过 `_supports_sdpa` 与 `is_torch_sdpa_available` 检查**。

**TT 覆盖判定**：**不适用**。这是**改变框架的 attention 后端选择策略**，不涉及 `torch.cuda.*`。
归入 D 类（训练策略）性质——它强制了一条路径，而 TT 的定位是「让原路径能跑通」，不是「替用户选路径」。

**B-4 `BertAttention`（4_42_3，1 条）**

mock 把 `BERT_SELF_ATTENTION_CLASSES` 固定为 `{"eager": ..., "sdpa": ...}` 并在 `__init__` 里强制 `config._attn_implementation = "sdpa"`，
其余 `prune_heads` / `forward` 逐字复制原生。**实质是强制 SDPA + 复制原生代码**。

**TT 覆盖判定**：同 B-3，属策略强制。但注意这条同时是 E 类（整体类替换）的形态。

#### B 类小结

| 条目 | 真实差异 | TT 判定 |
|---|---|---|
| `is_torch_sdpa_available` ×3 | 兜底恒真，正常版本号下与原生等价 | ⚪ **无实际变更**（归 G 类） |
| `is_flash_attn_2_available` ×3 | 绕过包元数据缺失（**不是 CUDA 信号**） | ⚠️ **需 `fn_binding`，但不建议内置** |
| `_check_and_enable_sdpa` ×1 | 强制 SDPA 后端 | ❌ **策略强制，不应覆盖** |

---

### 3.3 C 类：框架安全检查绕行（1 条）

`transformers.utils.check_torch_load_is_safe`（4_52_3）——mock 是空函数 `pass`。

原生（本机 5.19 实测源码，4.52.3 同形）：
```python
def check_torch_load_is_safe() -> None:
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError("... CVE-2025-32434 ...")
```

**动机**：KL3 的定制 torch 版本字符串（如 `2.7.1+hstu`）可能让 `is_torch_greater_or_equal("2.6")` 解析失败或判错，
于是 `torch.load` 路径被这条安全检查拦死。

**TT 覆盖判定**：**技术上可行，但不建议主动做**。

理由：这是一条**安全护栏**（CVE-2025-32434，`torch.load` 的任意代码执行漏洞）。
TT 作为收束层，替用户静默绕开安全检查是不可接受的——用户会失去知道「自己的 torch 版本不被信任」的机会。

**建议**：TT 应该在**初始化时探测**这条检查是否会失败，若会失败则**打印明确警告 + 记 provenance**，
让用户知情后自行决定（例如改用 safetensors，或显式接受风险）。这与 `nccl.version` 的「不伪造、不阻断、留证据」三原则一致。

---

### 3.4 D 类：训练策略调优（4 条）— ❌ 明确不应覆盖

| 目标符号 | mock 行为 | 动机 |
|---|---|---|
| `torch.utils.data.DataLoader` | 强制 `pin_memory=True, num_workers=8, pin_memory_device="cuda"` | I/O 吞吐 |
| `torch.nn.parallel.DistributedDataParallel` | 强制 `bucket_cap_mb=200` | 通信粒度 |
| `torch.optim.AdamW`（`MODULE_REPLACE`） | 换成 `torch_xmlir.optimizer.fused_adamw.FusedAdamW` | 厂商融合优化器 |
| `swift.llm.train.SwiftSft._get_data_collator` | 前 2 个 step 强制 `padding_to = max_length * len(batch)` | 预热，避免早期变长 |

**为什么 TT 不应覆盖**：

1. **不是硬件差异**：`num_workers=8`、`bucket_cap_mb=200` 在任何硬件上都是可调参数，不是「CUDA 符号在 XPU 上不存在」这类硬约束；
2. **与 TT 的分层冲突**：TT 的 Provider 不变式是「provider 缺失只降性能不降功能」，
   而 `num_workers=8` 这类**强行改写用户配置**违反了「不改用户意图」的前提；
3. **会覆盖用户显式配置**：symbrewrite 的 mock 用 `kwargs.update(...)`，用户传什么都会被覆盖——这是 hack 而非适配；
4. **框架本身已支持**：LLaMA-Factory / ms-swift 的 YAML 都有对应参数，正确的做法是给配置不改代码。

**唯一有讨论空间的是 `FusedAdamW`**：它是厂商算子，理论上可视为「优化器级 Provider」。
但 TT 当前 Provider 的语义是「模块 forward 的分发」，优化器是**构造期的类替换**，形态完全不同。
若要做，需要新增一个独立的优化器接管通道——**从性价比看，框架侧的 `--optim` 参数已够用，不建议为 1 条改动加通道。**

---

### 3.5 E 类：模型结构整体替换（3 条）— ⚠️ 需 Provider 泛化

| 目标符号 | 替换内容 | 规模 |
|---|---|---|
| `transformers_modules.baichuan2_13b_hf.modeling_baichuan.BaichuanAttention` | 整个 Attention 类重写（用 `bmm_xpu` 替代 `torch.bmm`，且改写 mask 处理与 shape 变换顺序） | ~130 行 |
| `swift.model.models.qwen._forward_qwen3_vl_or_qwen3_omni` | 整个多模态前向重写（ViT 冻结 + `no_grad` + `masked_scatter` 调度） | ~140 行 |
| `bert.modeling_bert.BertAttention` | 整类复制 + 强制 SDPA（见 B-4） | ~90 行 |

**TT 当前的能力边界**：

- `wrap_model` **能**按类名匹配到 `BaichuanAttention` / `BertAttention`；
- 但 `ProviderModuleForward.__call__(self, x, *args, **kw)` 里**固定**提取 `module.weight` / `esp`，
  然后 `manager.call(self._semantic, x, w, eps)`——**签名被锁死在 RMSNorm 的三参数约定上**；
- 所以「整体替换一个 Attention 类」当前做不到。

**判定**：
- **BaichuanAttention**：⚠️ 需要 Provider 泛化。把 `wrap_module` 返回的 forward 从固定签名泛化为
  「由 Provider 子类定义 `wrap_forward(module)`」，即可支持任意签名。这是 ~30 行的机制泛化，**不是新架构**。
- **Qwen3-VL 的 `_forward_qwen3_vl_or_qwen3_omni`**：❌ **不应由 TT 承担**。
  它改的是 ms-swift 的多模态训练逻辑（ViT 冻结策略、梯度管理、deepstack embedding 合并），
  与硬件无关，是训练框架的特性增强 / bug-fix，**应该贡献回 ms-swift 上游**。
- **BertAttention**：同 B-4，属策略强制 + 整类复制，不建议覆盖。

---

### 3.6 F 类：框架版本兼容 patch（4 条）— ❌ 部分不应由 TT 承担

| 目标符号 | 替换内容 | 性质 |
|---|---|---|
| `transformers.modeling_utils.caching_allocator_warmup` | 恢复 HF 4.57 式的 allocator 预热逻辑（~50 行，用到 `mem_get_info` / `memory_reserved` / `memory_allocated`） | 版本行为回退 |
| `transformers.dynamic_module_utils.get_class_in_module` | 简化动态模块导入（去掉路径安全检查） | remote code 兼容 |
| `transformers.utils.import_utils.is_torch_fx_available` | 加 origin 留痕转发（原生已 `return is_torch_available()`） | **无实际变更** |
| `is_torch_sdpa_available` ×3 | 见 B-1，兜底恒真 | **无实际变更** |

**`caching_allocator_warmup` 是唯一值得特别讨论的**：

它内部消费 `mem_get_info` / `memory_reserved` / `memory_allocated`——**这三个符号 TT 已在 691ab33 中纳管（D4 收束）**。
这意味着 TT 接管后，这个函数读到的可能是**正确口径**的值，从而**自然解决**问题。

但必须实测验证，不能假设：
- 若 TT 的口径正确 → 无需替换，symbrewrite 这条 patch 可以退役；
- 若仍不正确 → 说明问题在**算法**（HF 5.x 改了预热策略本身），而非符号值——那也不该由 TT 管。

**其余 F 类判定**：
- `get_class_in_module`：❌ 属定制 torch 与 transformers 的 remote code 兼容问题，应由厂商栈处理；
- `is_torch_fx_available` / `is_torch_sdpa_available`：⚪ 原生已是 True，mock 无实际变更（归 G 类）。

---

### 3.7 G 类：无实际行为变更（3 条）— ⚪ 已失效 / 无操作

| 条目 | 为什么无变更 |
|---|---|
| `is_torch_fx_available` → `mock_is_torch_fx_available` | 4.52.3 原生 `return is_torch_available()`（实测源码），已是 True；mock 只是多了一层 `__origin_*` 留痕转发 |
| `is_torch_sdpa_available` ×3（4_43_3 / 4_49_0 / 4_52_3） | 原生在正常版本号下返回 True（实测 `2.12.0` → True），mock 恒真，等价 |
| `check_torch_load_is_safe` → `pass` | 有实际变更但不属「无变更」——归 C 类 |

**额外发现（重要）**：本机 transformers 为 **5.19.0**，其中 `is_torch_sdpa_available` 与 `is_torch_fx_available` **已被上游删除**。
这说明：
1. `transformers_4_52_3` 插件在 transformers 5.x 上会**静默失效**（符号不存在 → `SYMBOL_REPLACE` 找不到目标）；
2. 这正是 **D6「torch/框架版本窗口」缺陷的现实印证**——TT 的 `bind_targets.version_window` 目前只是字符串，
   symbrewrite 靠「一个版本一个目录」硬扛，两边都缺**运行时校验**。
3. 建议 TT 在函数级绑定失败时（符号不存在）**必须打印**（symbrewrite 是静默的）——TT 的 `_print_summary` 已实现这点，是明确优势。

---

## 4. 与 xswift/xmegatron 的对比

这个对比能解释「为什么 transformers 侧看起来更难覆盖」：

| 维度 | xswift / xmegatron | transformers（本分析） |
|---|---|---|
| 修改性质 | 几乎全是**硬件适配**（CUDA API → XPU 等价物） | **混合**：约一半硬件适配，一半是策略/兼容 |
| 与 `torch.cuda.*` 的关系 | 强相关（socket / stream / event / RNG） | 弱相关（多数在 `transformers.*` 自己模块内） |
| 语义是否可判定 | 是（同一数学对象的两种实现） | 部分是（RMSNorm/RoPE）；部分否（DataLoader 参数无「正确值」） |
| TT 覆盖比 | ~90%+ | **56% 直接 + 15% 小改** |
| symbrewrite 的维护负担 | 中等 | **高**（5 个版本目录 + 1 基线，39 条大量重复） |

**关键洞察**：transformers 侧的 39 条里，**A 类 21 条（54%）是同一个语义在不同模型模块的重复注册**。
`RMSNorm` 语义重复 9 次，`RoPE` 语义重复 12 次——symbrewrite 靠人肉复制，TT 靠 `module_names endswith 匹配` +
`fn_binding` 坐标表一次覆盖。**这是 TT 在可维护性上的结构性优势**，也是本次分析最有价值的结论。

---

## 5. 结论与建议

### 5.1 判定汇总

| 类 | 条目 | TT 判定 | 所需动作 |
|---|---|---|---|
| **A. 算子替换** | 21 | ✅ 覆盖 | `bind_targets` 增 `TRANSFORMERS` 表（RoPE 12 条）；Gemma4 `with_scale=False` 适配（~10 行） |
| **B. 能力探测** | 7 | ✅4 / ⚠️3 | `is_flash_attn_2_available` ×3 可用 `fn_binding`，但**建议由平台栈注册而非 TT 内置**；`_check_and_enable_sdpa` 不覆盖 |
| **C. 安全检查** | 1 | ⚠️ 留证据不绕行 | 探测 + 警告 + provenance，不静默 `pass` |
| **D. 训练策略** | 4 | ❌ 不应覆盖 | 走框架配置（YAML / `--optim`） |
| **E. 结构替换** | 3 | ⚠️1 / ❌2 | `ProviderModuleForward` 泛化签名（~30 行）支持 BaichuanAttention；VL forward 回上游 |
| **F. 版本兼容** | 4 | ❌ 部分不应 | `caching_allocator_warmup` 先真机验证 TT 是否自然解决；其余归厂商栈 |
| **G. 无变更** | 3 | ⚪ 无操作 | 可退役（`is_torch_fx_available` / `is_torch_sdpa_available`） |

**覆盖率**：TT 现有机制直接覆盖 **22/39 = 56.4%**；加小改可达 **28/39 = 71.8%**；剩余 11 条属职责边界外。

### 5.2 需要补充的机制（三个，均非新架构）

| # | 机制 | 规模 | 支撑的条目 |
|---|---|---|---|
| 1 | `bind_targets.py` 增 `TRANSFORMERS` 坐标表（RoPE 12 个坐标 + 未来可扩展） | ~30 行（纯数据） | A 类 RoPE |
| 2 | `Provider.wrap_module` 泛化：支持无 `weight` 模块（Gemma4 `with_scale=False`）与自定义 forward 签名（BaichuanAttention） | ~40 行 | A 类 Gemma4、E 类 BaichuanAttention |
| 3 | 函数级绑定失败可见性：符号不存在时打印（TT 已有 `_print_summary`，需覆盖「符号不存在」这条路径） | ~5 行 | G 类失效检测 |

### 5.3 明确不建议做的（边界纪律）

| 不做 | 理由 |
|---|---|
| DataLoader / DDP 参数强制 | 不是硬件差异；会覆盖用户显式配置；框架本身支持 |
| FusedAdamW 通道 | 为 1 条改动新增优化器接管通道，性价比不足；框架 `--optim` 已够 |
| 主动绕过 `check_torch_load_is_safe` | 安全护栏（CVE-2025-32434），静默绕行不可接受 |
| 整段替换 ms-swift 的 VL forward | 训练框架特性增强，应回上游 |
| 内置 `is_flash_attn_2_available` 替换 | 属厂商栈桩包元数据缺陷；TT 提供机制但不预置策略 |

### 5.4 三个待真机验证项

| 项 | 验证方法 | 判定标准 |
|---|---|---|
| `caching_allocator_warmup` 是否被 TT 自然解决 | KL3 上启用 TT 后跑 `AutoModelForCausalLM.from_pretrained`，对比禁用 symbrewrite 该条 patch 的行为 | 显存峰值与 baseline 差异 <5% → 可退役该 patch |
| `is_flash_attn_2_available` 的真实失败形态 | KL3 上打印 `_is_package_available("flash_attn")` 与 `importlib.metadata.version` 的实际结果 | 确认是 `find_spec` 抛错还是 `PackageNotFoundError` |
| transformers 5.x 上 4_52_3 插件的失效行为 | 装 transformers 5.19 后看 symbrewrite 是否静默跳过 | 静默 → 印证 D6 版本窗口缺陷的紧迫性 |

---

## 附录：39 条完整清单（按目标模块分组）

### 基线 `transformers/`（19 条）
1. `transformers.utils.import_utils.is_torch_fx_available` → `mock_is_torch_fx_available` 〔G 无变更〕
2. `transformers.modeling_utils.caching_allocator_warmup` → `mock_caching_allocator_warmup` 〔F〕
3. `transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm` → `wrapper_mock_RMSNorm` 〔A〕
4. `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm` → `wrapper_mock_RMSNorm` 〔A〕
5. `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb` → `mock_apply_multimodal_rotary_pos_emb` 〔A〕
6. `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_vision` → `mock_apply_rotary_pos_emb_vision` 〔A〕
7. `transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm` → `wrapper_mock_RMSNorm` 〔A〕
8. `transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb` 〔A〕
9. `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm` → `wrapper_mock_RMSNorm` 〔A〕
10. `transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb` 〔A〕
11. `transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb_vision` → `mock_apply_rotary_pos_emb_vision` 〔A〕
12. `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm` → `wrapper_mock_zero_centered_gamma_RMSNorm` 〔A〕
13. `transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb` 〔A〕
14. `transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb_vision` → `mock_apply_rotary_pos_emb_vision` 〔A〕
15. `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm` → `wrapper_mock_zero_centered_gamma_RMSNorm` 〔A〕
16. `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb` 〔A〕
17. `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.apply_rotary_pos_emb_vision` → `mock_apply_rotary_pos_emb_vision` 〔A〕
18. `transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm` → `wrapper_mock_Gemma4RMSNorm` 〔A，需适配〕
19. `transformers.models.gemma4.modeling_gemma4.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb_gemma4` 〔A〕

### `transformers_4_32_1/`（2 条）
20. `transformers.dynamic_module_utils.get_class_in_module` → `mock_get_class_in_module` 〔F〕
21. `transformers_modules.baichuan2_13b_hf.modeling_baichuan.BaichuanAttention` → `wrapper_mock_BaichuanAttention` 〔E，需泛化〕

### `transformers_4_42_3/`（6 SYMBOL + 1 MODULE）
22. `transformers.models.bert.modeling_bert.BertAttention` → `wrapper_mock_BertAttention` 〔B/E〕
23. `torch.utils.data.DataLoader` → `dataloader_wrapper` 〔D〕
24. `torch.nn.parallel.DistributedDataParallel` → `distributed_data_parallel_wrapper` 〔D〕
25. `transformers.models.llama.modeling_llama.LlamaRMSNorm` → `wrapper_mock_LlamaRMSNorm` 〔A〕
26. `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb` 〔A〕
27. `transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa` → `mock_check_and_enable_sdpa` 〔B，策略〕
28. **`MODULE_REPLACE`** `torch.optim.AdamW` → `torch_xmlir.optimizer.fused_adamw.FusedAdamW` 〔D〕

### `transformers_4_43_3/`（2 条）
29. `transformers.utils.is_torch_sdpa_available` → `mock_is_torch_sdpa_available` 〔G〕
30. `transformers.utils.is_flash_attn_2_available` → `mock_is_flash_attn_2_available` 〔B，需 fn_binding〕

### `transformers_4_49_0/`（4 条）
31. `transformers.utils.is_torch_sdpa_available` → `mock_is_torch_sdpa_available` 〔G〕
32. `transformers.utils.is_flash_attn_2_available` → `mock_is_flash_attn_2_available` 〔B〕
33. `transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm` → `wrapper_mock_Qwen2RMSNorm` 〔A〕
34. `transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb` → `mock_apply_rotary_pos_emb` 〔A〕

### `transformers_4_52_3/`（3 条）
35. `transformers.utils.is_torch_sdpa_available` → `mock_is_torch_sdpa_available` 〔G〕
36. `transformers.utils.is_flash_attn_2_available` → `mock_is_flash_attn_2_available` 〔B〕
37. `transformers.utils.check_torch_load_is_safe` → `mock_check_torch_load_is_safe` 〔C〕

### `swift/`（2 条）
38. `swift.llm.train.SwiftSft._get_data_collator` → `mock__get_data_collator` 〔D〕
39. `swift.model.models.qwen._forward_qwen3_vl_or_qwen3_omni` → `mock_forward_qwen3_vl_or_qwen3_omni` 〔E/D，回上游〕

---

## 附录二：本分析与前两版口头结论的差异（勘误）

| 项 | 前版说法 | 本版更正 | 依据 |
|---|---|---|---|
| 条目总数 | 「约 30 条」 | **39 条**（38 SYMBOL + 1 MODULE） | 正则解析全部 `__init__.py`，剔除 1 条注释 |
| `is_flash_attn_2_available` 根因 | 「Bridge 接管 `torch.cuda.is_available()` 后自然解决」 | **不成立**。执行顺序上包元数据检查在前，`importlib.metadata.version` 抛 `PackageNotFoundError`（实测） | 4.52.3 真实源码 + 桩模块实测 |
| `is_torch_sdpa_available` 判定 | 「Bridge 已覆盖」 | **归 G 类（无实际变更）**：原生不查 CUDA，只比 torch 版本号，正常版本下已是 True | 4.52.3 真实源码 |
| `BertAttention` 归类 | 「E 类结构替换」 | **B 类（策略强制）+ E 类（整类复制）双重性质** | 读 mock 源码：强制 `_attn_implementation="sdpa"` |
| A 类条目数 | 「15 条」 | **21 条**（RMSNorm 9 + RoPE 12；扣除 BertAttention 重复计入） | 逐条清点 |
| `is_torch_fx_available` | 「F 类版本兼容」 | **G 类（无实际变更）**：4.52.3 原生已 `return is_torch_available()` | 4.52.3 真实源码 |
| 覆盖率 | 「74.4%」 | **56.4% 直接 + 15.4% 小改 = 71.8%** | 按修正后的分类重算 |

**方法论教训**：判断「某条 patch 是为了绕什么」时，**必须读被替换符号的原生实现**，不能从 mock 的名字或意图注释反推。
本次 `is_flash_attn_2_available` 的误判就源于此——mock 的函数名叫 `mock_is_flash_attn_2_available`、内部只留了 `torch.cuda.is_available()`，
很容易误以为「它就是用 cuda 信号替代 flash_attn 检查」，而实际关键是**包元数据缺失**这个更早的失败点。

---

## 附录三：TTorch 最新代码（`9112aab`）D 编号闭环对照

> 补录日期：2026-10-23
> 代码基线：`/Users/vector/code/train/TTorch` @ `9112aab`（dev，D 类缺口修复第二轮）
> 验证环境：本机 CPU venv pytest 22 passed / 3 skipped / 1 failed(面闭合差 2 符号) / 4 errors(无卡环境)

### D1-D14 全量闭环状态

| D# | 缺陷 | 状态 | 落地形态 | 验证 |
|---|---|---|---|---|
| D1 | 作用域边界 | ✅ **已修** | `_classify_process`（primary/derived/inherited）+ `skip_conformance` + `register_at_fork` + `mark_activated` | pytest 3 用例 + 4 卡 DP 回归 |
| D2 | 跨 rank 一致性 | ✅ **已修-A** | `_call_sig` device 只取 `.type` + 对象参数按类型名归并 | pytest `test_provider_sig` 7 用例 |
| D3 | 半激活态 | ✅ **已修** | `_rebind.rollback()` + bootstrap try/except + `mark_failed`/`assert_healthy` + 恢复语义 | pytest `test_bootstrap_rollback` + `test_poison` |
| D4 | 契约面闭合 | ✅ **已修** | `surface.py` 三态 + audit 闭合判据 + 57 Managed / 22 Denied / 43 Exempt = 122/124（差 2 个 torch 2.12 新符号） | pytest `test_d10` + surface CLI |
| D5 | 伪装外溢 | **修正：非缺陷** | 定性推翻——伪装值是厂商刻意的 CUDA 兼容身份，不存在外溢。`masquerade_profile` 保留为观测画像 | conformance ENDORSEMENT 恒 PASS |
| D6 | torch 版本窗口 | ✅ **已修** | `torch_range=("2.9","2.10")` + `_warn_torch_range` warn 不 block | 激活期输出 |
| D7 | 显存口径 | ✅ **已闭环** | `mem_caliber_coherence`（三序列增量 + 单调 + 回落）+ `mem_reset_semantics`（witness 诊断） | pytest `test_d7` + conformance |
| D8 | 热路径税 | ✅ **已闭环** | `measure_forward_tax` + `measure_dispatch_tax` 双基线 | pytest `test_d8`（需卡环境） |
| D9 | 对账单点 | ✅ **已修，分层** | 兜底对账保持单点（轻量）；`verify_provider_generalization`（边界输入集 + 空张量特判）挪 witness | pytest `test_d9` + provider_admit |
| D10 | 收束不可见 | ✅ **已修** | `coverage_profile` + `unconverged_components` + 激活末尾打印 + provenance | pytest `test_d10` |
| D11 | provenance 无标识 | ✅ **已修** | 信封 +pid/+rank + `fork_reset` + `default_export_path` | pytest 4 用例 |
| D12 | detect 无隔离 | ✅ **已修** | `auto_activate` 内 try/except + provenance 留证 | pytest `test_d12`（需卡环境） |
| D13 | 接口档位 | **修正：两档即可** | 基类 raise + 平台 adapter 按硬件能力覆写（no-op 归 adapter 实现选择非基类） | 设计决策已文档化 |
| D14 | wrap_model 时机 | ✅ **已修（延迟自检）** | wrap 期零侵入 + 精度兜底延迟到首次 `__call__` + 并行包装后可替换 | pytest `test_provider_admit` |

**14 条中 11 条已修复落地、2 条定性修正（非缺陷/不需要）、1 条（D2-B 跨 rank 对账）记为 follow-up。**

### Provider 替换机制的关键改进（`9112aab` 新增）

本轮修复中**最重要的架构改进**在 Provider 层——直接回应 transformers 分析中识别的 gap：

#### 改进 1：延迟自检（D14 时机消解）

**旧问题**：`wrap_module` 期间做 `module.to(float32)` + `module.forward(x)` 跑精度自检——侵入性强，与 FSDP/ZeRO-3 冲突。

**新方案**：wrap 期**零侵入**（只挂成员 + 替换 forward，不跑前向、不改 dtype），精度兜底延迟到**首次 `__call__`**。

这对 transformers 分析的意义：
- symbrewrite 的 A 类条目（21 个 RMSNorm/RoPE 替换）全部是**类替换**——用新 Module 类直接换掉原 Module 类。
  TT 的 `wrap_model` 是**forward 替换**（不换类，只换 forward 函数）。两者语义不同但效果等价，
  **TT 的方案对并行框架更友好**（不影响类型检查、不影响 param 注册、不影响 state_dict key）。
- 延迟自检使 `wrap_model` 可以在**任意时机**调用——框架集成的灵活性大幅提升。

#### 改进 2：provider ↔ module 一对一分离（zcg 变体独立）

**旧问题**：一个 `RMSNormProvider` 运行时做变体识别（w vs 1+w），`ProviderModuleForward` 内部按 zcg 标志折 scale——
语义选择被错误下沉到运行时，与「Provider 在设计时确定语义」的原则矛盾。

**新方案**：标准 `RMSNormProvider`（w 直乘）与 `ZcgRMSNormProvider`（1+w）**分离为独立 Provider**。
`wrap_model` 遍历时，多个 provider 按 `zero_centered_gamma` flag 轻量分流 + self_check 精度兜底。
flag 错误由延迟自检纠正（不依赖 flag 做最终判定）。

这对 transformers 分析的意义：
- symbrewrite 为 zcg 变体单独写了 `wrapper_mock_zero_centered_gamma_RMSNorm`——正是「一对一」的体现；
- TT 此前把变体识别塞进 self_check 是**机制污染**（gate 有两个职责），拆开后 self_check 纯做精度兜底，职责干净。
- 新增 Gemma4 这类更特殊的变体（`with_scale=False`）也遵循同一模式——新建一个 provider 即可，不动机制。

#### 改进 3：dtype 自适应阈值（取代 to(float32) 侵入性转换）

**旧问题**：`_self_check` 用 `module.to(float32)` 把自检搬到 fp32，理由是「bf16 阈值失效」。
实际上根因是**固定绝对阈值 + 噪声随 dtype 变化**——正确做法是修正阈值逻辑而非侵入性改 dtype。

**新方案**：阈值随 dtype 自适应（fp32→1e-3 / bf16→2e-2），在模块实际 dtype 下自检。

---

## 附录四：TTorch 仍存在的 Gap（结合 transformers 分析与最新代码）

> 补录日期：2026-10-23

虽然 D1-D14 已全部闭环，结合 transformers 分析暴露的六类修改与 TTorch `9112aab` 最新代码，仍有以下**关键 Gap**：

### Gap-1：Provider 只覆盖 RMSNorm/RoPE/SwiGLU，缺 Attention 语义 〔高〕

**现状**：TT 当前 4 个 provider semantic（`rmsnorm` / `rmsnorm_zcg` / `rope_bshd` / `swiglu` / `bias_swiglu`），
全部是**逐算子替换**。而 transformers 分析的 E 类（BaichuanAttention、BertAttention）是**整段 Attention forward 替换**——
这在实际训练中是收益最大的优化点（attention 占训练时间 40-60%）。

**缺失的具体能力**：
1. `ProviderModuleForward.__call__` 签名固定为 `(x, w, eps)`——无法表达 Attention 的 `(q, k, v, mask)` 语义；
2. Provider 的 `probe_input` / `native` / `_self_check` 都围绕 `(x, w, eps)` 设计——扩展到 Attention 需要新的 probe 约定；
3. **更深层的问题**：Attention 替换不只是「换算子」，而是「换算法」（eager→SDPA→flash-attn），涉及 causal mask 处理方式、
   padding 策略、key-value cache 接口——这些不是 Provider 的「同一数学对象的不同实现」语义能表达的。

**建议**：
- **短期**：不扩展 Provider 到 Attention 层——那会把 Provider 从「算子替换」扩展为「算法替换」，复杂度跳一个级别；
- **中期**：利用 transformers 自身的 `_attn_implementation` 机制（`"eager"/"sdpa"/"flash_attention_2"`），
  TT 的角色是**确保厂商的 attention 后端能被正确选中**（Bridge 接管能力查询信号使框架选到正确路径），
  而不是自己实现 attention 替换——这是 symbrewrite 在 D 类里做的事（强制 `_attn_implementation="sdpa"`），
  但正确的做法应该是让框架原生的选择逻辑工作。

### Gap-2：函数级通道缺 transformers 坐标表 〔中〕

**现状**：`bind_targets.py` 只有 `MEGATRON` 表（3 个坐标），无 `TRANSFORMERS` 表。
transformers 分析 A 类 12 个 RoPE 函数替换的坐标已经明确，但还没登记。

**影响**：transformers 路径上的 RoPE 函数（`apply_rotary_pos_emb` 等）无法通过 `fn_binding` 接管——
当前只能靠 `wrap_model`（模块级），而 RoPE 在 transformers 里是**函数**不是 Module。

**建议**：`bind_targets.py` 增 `TRANSFORMERS` 表，坐标直接从 symbrewrite 的条目清单（附录一）提取。
每个坐标带 `version_window`（如 `"transformers>=4.42（qwen2 起）"`），与 megatron 同形态。
~30 行纯数据改动，不涉及机制。

### Gap-3：Gemma4 `with_scale=False`（无 weight）模块未适配 〔低〕

**现状**：Gemma4 的 `v_norm`（`with_scale=False`）没有 `weight` Parameter，
`wrap_module` 入口 `isinstance(module.weight, Parameter)` 直接 raise TypeError 跳过。
symbrewrite 为此写了完整的 `Gemma4RMSNorm` 类（无 weight 时用全 1 向量）。

**建议**：`wrap_module` 对无 weight 的模块不 raise，而是传 `ones_like` 给 Provider——
由 Provider 自己决定是否接管（约 10 行，在 `wrap_module` 的 weight 获取逻辑里适配）。
或者新增一个 `NoScaleRMSNormProvider`（更干净但多一个 provider）。

### Gap-4：框架探测函数的包元数据缺失（`is_flash_attn_2_available`）〔中〕

**现状**（本文 §3.1 B-2 详述）：symbrewrite 的 `MODULE_REPLACE` 注入桩模块无 dist-info，
导致 transformers 的 `is_flash_attn_2_available` 内 `importlib.metadata.version("flash_attn")` 抛 `PackageNotFoundError`——
**TT 的 Bridge 管不到 `importlib.metadata`，这在 TT 的能力边界之外。**

**但 TT 能做的**：`fn_binding` 技术上支持注册 `transformers.utils.is_flash_attn_2_available` 的替换——
问题是**该不该由 TT 内置这条**（§3.1 给出了「TT 提供机制但不预置策略」的建议）。

**建议**：
- 明确这是**平台栈的职责**（`MODULE_REPLACE` 注入的桩应自带 `__version__` 与 dist-info 元数据），
  不是 TT 的缺陷——但 TT 应在文档中记录这个已知形态（「厂商桩包不完整时框架探测炸」），
  供平台栈参考修正。
- 备选：TT 提供一个「框架探测 patch 注册通道」（`fn_binding` 已够用），平台栈按需注册。

### Gap-5：跨 transformers 版本维护策略未定义 〔中〕

**现状**：symbrewrite 靠**版本目录**维护（`transformers_4_32_1/` ... `transformers_4_52_3/`），
5 个版本目录 + 1 基线 = 6 套代码，大量条目重复（RMSNorm 重复 9 次、RoPE 重复 12 次）。
TT 当前 `torch_range=("2.9","2.10")` 只覆盖 torch 版本窗口，**对 transformers 版本无任何管理**。

**关键决策点**：
- `bind_targets` 的坐标（模块路径 + 符号名）随 transformers 版本漂移——
  如 Qwen2 的 `modeling_qwen2.py` 在 4.42 和 4.52 里路径相同，但 5.x 可能移入新命名空间；
- `is_torch_sdpa_available` 在 5.19 已删除——4_52_3 插件在 5.x 上静默失效。

**建议**：`bind_targets` 的 `TRANSFORMERS` 表每个坐标带 `version_window` 字符串（与 megatron 同形态），
`fn_binding` 安装时校验符号是否存在（已有——`_bind_one` 在 `orig is None` 时 skip 并打印 `_print_summary`），
**不为每个版本建目录**——这是 TT 对 symbrewrite 的结构性优势（一份坐标表 + 运行时校验 vs N 份代码目录）。

### Gap-6（预留）：Provider 热路径 Python 分发开销 〔观察〕

`measure_dispatch_tax` 实测分发税 14%（D=256 小 shape），当前量级可接受。
但随 Provider 数量增多（当前 5 个 → 未来可能 10+）和 `_call_sig` 复杂度增长，
热路径 Python 开销可能成为瓶颈——**只作观察（基线已有），暂不优化**。

若未来超阈值，候选优化：
1. `_call_sig` 缓存（per-module 固定签名的 provider 可在 wrap 期确定 sig，热路径查 dict 而非逐参数构造）；
2. 门禁结论按 Provider 而非按签名缓存（若确认「同 Provider 的门禁不随 shape 变」）；
3. 编译期通道（Provider `kind="fused_jit"`，spec.py 已预留）。

---

**最终结论**：TTorch `9112aab` 的 D1-D14 已全部闭环。结合 transformers 分析，剩余 Gap 中**最有价值的下一步**是 Gap-2（`bind_targets` 增 TRANSFORMERS 表，~30 行纯数据），它能让 TT 直接覆盖 transformers 分析 A 类中的 12 个 RoPE 函数级替换——这是 symbrewrite 维护量最大的部分，也是 TT 结构性优势（一份坐标表 vs N 个版本目录）的最佳示范。
