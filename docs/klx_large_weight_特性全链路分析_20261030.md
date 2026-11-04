# KLX large weight（`XTE_GROUPED_GEMM_LARGE_WEIGHT`）特性全链路分析

- 整理日期：2026-10-30
- 来源：从《XSwift 退役能力分析与实施计划》（`xswift_退役能力分析与实施计划_20261029.md` §14）抽离为独立文档，主文档 §14 已替换为指向本文的指针。
- 环境基线：KL3 / XTE `transformer_engine==2.12.0+c9c3d0d9`、Megatron 0.18、`mcore_bridge`、XME `xmegatron_ext`、XSwift `xswift_ext`。
- 一句话定性：**large weight 是 KL3 专属的性能特性，不是跨平台通用优化，更不是正确性特性**；本质是"list kernel 有精度 bug 被互斥禁用、唯一可用 kernel 只吃连续 3D 张量"这一现状下的结构解。

## 目录

1. 特性定位与上游等价物
2. 自上而下修改点清单（L0–L8）
3. 逐层主体逻辑与合理性
4. 精度正确性是怎么保证的
5. 性能收益是怎么拿到的
6. 三个追问的实证结论（连续性 / 根因 / grouped GEMM）
7. 问题分级
8. 升级到 TE 2.14 / 上游 single_grouped_weight 的可行性
9. 对 XSwift 退役的结论
10. 证据位置

---

## 1. 特性定位与上游等价物

large weight 不是算子优化，而是一次**参数存储布局变更**：把 TE `GroupedLinear` 的
`weight0..weight{G-1}`（G 个 2D `Parameter`）替换为单个 3D `Parameter`
`large_weight[G, out, in]`（bias 对应 `large_bias[G, out]`）。

它只改动 TE 一层的存储，但 `weight{i}` 这个名字是 Megatron、mcore_bridge、LoRA tuner、
dist-checkpoint 四条链路共同依赖的契约，因此其上压了三个包、八个补丁点把名字"翻译"回来。

上游存在**官方等价能力**：`moe_single_grouped_weight` / `moe_single_grouped_bias`
（`megatron/core/transformer/transformer_config.py:766-774`），要求 `transformer-engine>=2.14.0`。
本机 XTE 为 `2.12.0+c9c3d0d9`，够不到 2.14，所以厂商另起了一套 `large_weight`。
**整条 shim 链的根因是参数命名没有沿用上游的 grouped key。**

## 2. 自上而下修改点清单（L0–L8）

| 层 | 文件:位置 | 归属 | 做了什么 |
| --- | --- | --- | --- |
| L0 开关 | `transformer_engine/pytorch/env_config.py:67` | XTE | 注册 `XTE_GROUPED_GEMM_LARGE_WEIGHT`，默认 `0`；`get()` 直读 `os.getenv`，无缓存 |
| L0 开关镜像 | `xmegatron_ext/envs.py:68` | XME | 同名 EnvSpec，供 plugin 门控 |
| L0 校验降级 | `xmegatron_ext/components/training/arguments.py:127-142` | XME | 非 `torch_dist`、或 `ep==1 且要存 optim` → 改写 `os.environ` 自动关闭；`moe_layer_freq==1` 强制展开为 list |
| L1 存储 | `transformer_engine/.../module/grouped_linear.py:756-794` | XTE | 注册 `large_weight[G,out,in]`、`large_bias[G,out]`；无 bias 时放 0-size 占位张量（非 Parameter） |
| L1 TP 属性 | 同上 `:857-881` | XTE | `set_tensor_model_parallel_attributes` 维度整体 +1（row→2、column→1） |
| L2 计算 | 同上 `:985-1227` | XTE | forward 取 `large_weight`，按需切成 per-expert list 喂各分支；`return_bias` 同样按 index 切 |
| L2 互斥 | 同上 `:758-762` | XTE | 强制 `XTE_USE_GROUPED_FC_FUSION=0`，注释自承与 fastbf16 inplace 共存有精度回退 |
| L3 Megatron 本体 | — | — | **零修改**。`megatron/` 全仓 grep `large_weight`/`large_bias` = 0 命中 |
| L4 dist-ckpt 分片 | `xmegatron_ext/.../core/extensions/transformer_engine.py:54-155` | XME | 重写 `TEGroupedLinear._sharded_state_dict_grouped`：无 `large_weight` 走原函数；有则自建 ShardedTensor，由 `tp_axis_map['0.weight']` 推出 `large_weight: 1/2` |
| L5 存盘转换 | `xmegatron_ext/.../training/checkpointing.py:494-641` | XME | 包 `generate_state_dict`：large → per-expert 拆回盘上格式，含 SwiGLU factory 与 optimizer `param_state` 整数索引重排 |
| L5 读盘转换 | 同文件 `:53-73, 310-410` | XME | 包 `_load_base_checkpoint`：per-expert → `torch.stack` 成 large；丢弃 `_extra_state{1..}`；optimizer 索引反向重排 |
| L6 属性 shim | `xmegatron_ext/.../core/extensions/transformer_engine.py:19-51` | XME | 重写 `TEGroupedLinear.__getattr__`，miss 时返回 `large_param.data[i]`（detached 只读视图） |
| L7 HF↔MG 转换 | `xswift_ext/.../mcore_bridge/bridge/gpt_bridge.py:1-105` | XSwift | 包 `GPTBridge._set_mlp_state`：调用期把 `large_param.chunk()` 挂成真实属性 `weight{i}`，`finally` 还原 |
| L8 LoRA 折叠 | `xswift_ext/.../mcore_bridge/tuners/lora.py:1-66` | XSwift | 包 `LoraParallelLinear.merge/unmerge`：同样挂真实属性，并在退出前写回 `large_weight` |
| L8 注册门控 | `xswift_ext/plugin.py:74-90` | XSwift | 仅当 env 为真才注册 L7/L8 |

## 3. 逐层主体逻辑与合理性

**L1/L2（XTE 存储与计算）—— 逻辑成立，两处设计债。**

主体逻辑：G 个独立 2D 张量合成一个 3D 张量，forward 用时再 `weight_tensor[i]` 切回 list。

合理之处：TP 属性维度统一 +1，与前置的 expert 轴一致，和非 large 分支对称；无 bias 时用
0-size 张量占位而非 `None`，使下游 `numel()==0` 判据统一（XME 与 XSwift 两处 shim 都复用了它）。

设计债：

1. `large_weight` 与 `weight{i}` 语义串台。`grouped_linear.py:1004-1007` 显示**非** large 分支在
   `noop_cat` 返回视图时会把该视图回写 `self.large_weight`，于是"`large_weight` 属性存在"不等价于
   "处于 large 布局"。XSwift `tuners/lora.py:55` 恰以 `getattr(base_layer,"large_weight",None) is not None`
   为判据。**修正说明：在 KL3 上 `noop_cat` 必然退化（见 §5），`_is_noop_cat` 永不为真，
   该回写不会发生，隐患在本机不触发；在 NV 上才会触发。** 属于平台相关的潜在误触发。
2. 与 FC fusion 互斥：large weight 的收益与 grouped FC fusion 的收益拿不到一起，这是未在任何文档
   暴露的隐式性能取舍。
3. 初始化：`init_fn` 一次性作用在 3D 张量上。若 `init_method` 与形状相关（xavier 类按 fan_in/fan_out），
   per-expert 2D 初始化与 3D 一次性初始化的方差不等价。Megatron 默认 `init_method_normal` 与形状无关，
   当前配置无碍，但这是**配置敏感的隐藏前提**。

**L4/L5（XME checkpoint）—— 读写对称，但索引算术是最大脆弱点。**

主体逻辑：盘上格式保持上游 per-expert 不变，内存格式为 large，存盘前拆、读盘后合。

合理之处：盘格式不变是正确取舍，否则 ckpt 无法与上游互转；`_is_swiglu_grouped_weight` 把 SwiGLU
factory 严格限定在 `linear_fc1` 的 base weight（不含 adapter），与 `mcore_bridge/tuners/lora.py:466`
的作用面精确对齐；`_param_state_has_large_bias` 不信 `add_bias_linear` 全局 flag，改为探测实际 key，
因为 LoRA 下只有 adapter 进优化器而 adapter 无 bias——这两处判断都很到位。

设计债：

4. **优化器状态靠裸整数索引算术重排**。存侧 `num_local_experts*(param_idx+1)-2*i-2`、
   读侧 `2*(linear_idx*num_local_experts+i)+1` 且带 `sub_weight_list[::-1]` 逆序。整套逻辑依赖
   "Megatron 参数注册顺序为 weight/bias 交错、按 linear 分组"这一**未契约化的隐含约定**。
   Megatron 一改注册顺序即**静默错位**（权重与 moment 配错专家），而 `_validate_*` 系列只校验
   "键连续 / 状态名一致 / 首维 == num_local_experts"，错位时断言全部通过。本特性风险最高的一块。
5. `ep_size==1 且需存 optimizer` 直接降级关特性，等价于承认上述索引逻辑只在 ep>1 下验证过。
6. `parse_large_weight_state_dict` 内无条件 `pop(f'{prefix}.weight{i}')`，缺键抛 `KeyError` 而非可读诊断。

**L6（XME `__getattr__` shim）—— 定位准确，能力不足。**

只走属性 miss 路径、真实属性优先 shadow，这个最小侵入选择是对的；不注册成 Parameter
以避免污染 state_dict 与优化器也是对的。但"读得到"不等于"写得进"：detached 视图在
"调用方重新绑定引用"的场景静默失效。**已实测**：只装 XME shim 时 LoRA `safe_merge` 后
权重误差为 1 且层被标记为已合并（静默漏写）；补上 XSwift 写回后误差 0。

**L7（XSwift `_set_mlp_state`）—— 正确，且依赖一个已核实的前提。**

`_has_real_attr` 不用 `hasattr`，因为 L6 的 shim 会对任意 `weight{i}` 答 True，`hasattr` 会导致
还原阶段把临时视图 `setattr` 成永久真实属性，**永久遮蔽 shim**，这个坑看得很准。
不需要写回的前提已核实成立：`mcore_bridge/bridge/gpt_bridge.py:228` 的 `_set_param` 使用
`param.data.copy_(tensor)`，in-place，直接穿透视图写进 `large_weight`。

设计债：

7. 专家数口径不一致：L7 用 `config.num_moe_experts // bridge.ep_size`，L8 用 `base_layer.num_gemms`。
   正常相等，但 L7 在 `num_moe_experts is None` 时返回 1，一旦将来出现 dense 路径的 grouped linear
   会静默切错。应统一取 `num_gemms`。

**L8（XSwift LoRA merge）—— 症状修复，位置不在根因处。**

`data_ptr()` 比对精准区分了两条路径：非 safe 路径 `orig_weight.data += delta`（in-place，指针相同，
跳过拷贝以避免自拷贝）；safe 路径 `weight.data = orig_weights[i]`（`mcore_bridge/tuners/lora.py:540`，
重绑定，指针不同，需要拷回）。

设计债：

8. 这是在调用方修补被调用方的写入语义。真正的问题是 `merge` 在 `safe_merge` 分支用了赋值而非
   `copy_`，对任何"weight 是视图"的布局都会丢写。**与 THD 那条定性同构：这是 mcore_bridge 自身的
   写入契约缺陷，不是厂商专属能力。** 若 `mcore_bridge` 的 safe 分支改成
   `weight.data.copy_(orig_weights[i])`，L8 整个 wrapper 可以删除。

## 4. 精度正确性是怎么保证的

分三类，其中两类是"规避"而非"验证"。

**（1）结构性数值等价（设计层面保证，不依赖测试）**

`grouped_linear.py` forward 的两条分支最终汇聚到同一处：非 large 路径用
`noop_cat(weights_to_cat, dim=0)`（`:1000-1001`）得到一个 3D 张量，large 路径直接使用 3D
`large_weight` 本体（`:985-987`），两者都作为单个 `weight_tensor` 喂给 `MoeFcFusionImpl`
（`:1081-1083`）。`torch.cat` 只是拷贝，不改变数值，所以**GEMM 的输入逐位相同 → 前向数值逐位等价**。

反向侧：非 large 路径由 `_NoopCatFunc.backward`（`module/_common.py:154-165`）按 `split_ranges`
切片把 grad 分发回各 Parameter；large 路径 grad 直接落在 3D Parameter 上。两者数学等价，差异
只在**梯度的物理归属**（G 个 grad 张量 vs 1 个），影响的是优化器与 dist-opt 分片，不是数值本身。

**（2）规避式保证（两处"关掉"而非"修好"）**

- `grouped_linear.py:758-762`：开启 large weight 时强制 `XTE_USE_GROUPED_FC_FUSION=0`，
  注释自承"fastbf16 inplace mode 与 large weight 共存有潜在精度回退"。即缺陷未修，用关闭另一个
  优化来规避。注意这同时改变了算子路径：关闭 FC fusion 后走 `MoeFcFusionImpl`（单 3D 张量）
  而非 `GroupedFcFusionImpl`（per-expert list，`:1073-1079`）。
- `arguments.py:134-139`：`ep==1 且要存 optim` 时直接关闭特性，承认优化器状态转换路径
  只在 ep>1 验证过。

值得并列记录的是：**上游在同一个地方踩了同一个坑。**
`transformer_config.py:1425-1436` 把 `moe_single_grouped_weight` 硬门控到 fp8 + mxfp8，理由是
"dist-optimizer 的 quantized-param shard 路径在 single-grouped-weight 存储上只在 mxfp8 验证过，
其他组合有已知数值问题（upstream PR NVIDIA/Megatron-LM#4621）"。两家的结论一致：
**单个大张量进 dist-optimizer 的分片路径是数值风险区**，XTE 的规避方式是关 FC fusion + 限制 ep>1，
上游的规避方式是只允许 mxfp8。

**（3）本项目已有实测的覆盖面与缺口**

已实测的只是**转换与合并的正确性**，不是训练数值精度：

- 真实 `TEGroupedLinear` 在 large 布局下访问 `weight0` 抛 `AttributeError`，`GPTBridge` 导入失败；
- 仅装 XME getter 时 LoRA `safe_merge` 误差 1 且被标记为已合并（静默漏写）；
- 补上 XSwift 写回后误差 0。

**尚未实测（精度口径的真实缺口）**：

- large weight on/off 的同配置 loss 逐步对齐（验证 §4(1) 的结构性等价在整网成立）；
- checkpoint save → load 往返一致性，尤其**优化器 moment 与专家的配对关系**是否保持
  （这正是 §3 第 4 条标为最高风险的那块，当前无任何测试能发现错位）；
- 关闭 FC fusion 带来的算子路径切换（`MoeFcFusionImpl` vs `GroupedFcFusionImpl`）本身的数值差异。

## 5. 性能收益是怎么拿到的

**根因证据：`noop_cat` 不是 view 类算子，而是一个"条件零拷贝"包装；在 KL3/XMLIR 上两道闸都会让它退化成真 `torch.cat`。**

`noop_cat`（`module/_common.py:168-191`）不是"view 类算子"。它有两个分支——
`_NoopCatFunc.forward` 内部用 `set_` 在同一块 storage 上伪造出 3D 视图（这才是"view"来源），
外层再兜一道 XMLIR 语义检测。因此它是不是 view 取决于两道闸，而非算子本身：

1. `_NoopCatFunc.forward:133-141` 检查 G 个张量在内存中是否首尾相接（`data_ptr` 连续、`stride` 相同），
   不满足就直接 `torch.cat`。而 `grouped_linear.py:796-797` 的注释明确写了
   "Keep expert storages independent so each one can be released separately"——per-expert 权重是
   **独立分配**的，`data_ptr` 不连续，**这一闸即已退化**。
2. 即使连续，`noop_cat:185-191` 还发现 XMLIR 后端下 `torch.Tensor.set_` 会篡改
   `tensors[0].data_ptr()`，一旦检测到不一致就**再退化成 `torch.cat`**。

因此在本机栈上，small weight 路径**每层每次 forward 都会完整拷贝一遍本地专家权重**，
并分配一份等大的临时 3D 缓冲。这解释了 large weight 在 KL3 上的性能动机。

**修正（2026-10-30 实测）：** "NV 侧 `noop_cat` 零拷贝"此前是无依据推测，撤回。本机 CPU 实测：
复刻 small 分支的 back-to-back 分配（G=8 个等大 bf16 张量连续 `torch.empty`），`data_ptr` **不连续**，
`noop_cat` 返回的是真实拷贝（`out.data_ptr() != tensors[0].data_ptr()`）。NV 上是否连续未测——
取决于 CUDA caching allocator 的分配顺序巧合，**不应作为设计前提**。另外 XTE 内部存在自相矛盾的
两条注释：`grouped_linear.py:797` "Keep expert storages independent"（当前实现）与 `:1003`
"small weights are already allocated in contiguous buffers"（noop_cat 的假设），说明该优化的前提
在当前实现里已经被自己的存储选择破坏。

还要注意：**只有"单 3D 张量"接口的 kernel 才需要 cat**（INT8 路径、`MoeFcFusionImpl`、SmoothQuant 入口）。
`GroupedFcFusionImpl`（list 接口，`:1073-1079`）、非融合 `_Linear` 逐 GEMM 循环（`:1183-1203`）、
以及 `backward_dw`（`:1317-1322`）都只消费 per-expert list，不产生 cat。所以"不做 large weight 的代价"
精确地说就是：**被迫走单 3D 张量接口的那几条 kernel 路径，每步多一次全权重拷贝**。

**收益构成：**

| 项 | 来源 | 量级 |
| --- | --- | --- |
| 主：消除 forward 权重拷贝 | 权重天生就是那个 3D 连续张量，不再 `torch.cat` | `num_local_experts × (fc1+fc2 权重字节) × 层数 × 每步前向次数`（含重算则再乘）；需按实际 MoE 配置代入 |
| 次：参数数量下降 | 每 grouped linear 从 G 个 Parameter 降到 1 个 | dist-optimizer 分片元数据、grad buffer 桶数、`param_groups` 遍历开销按 G 倍缩减 |
| 显存 | 省掉每层一份临时 3D cat 缓冲 | 降峰值，同时减少变长场景下的碎片 |
| 代价 | 与 `XTE_USE_GROUPED_FC_FUSION` 互斥 | 拿不到 grouped FC fusion 的收益，二者只能选一 |

**未实测**：本机没有 large weight on/off 的吞吐与显存对照数据。上表的主收益是从代码路径
推出的结构性结论，量级需实测确认；在补齐 §4 的 loss 对齐实验时应同时采集吞吐与峰值显存。

## 6. 三个追问的实证结论（2026-10-30，用户确认）

**Q1：KLX 的 large weight 是否保证了多个 weight 的连续性？**

保证，且是**结构性保证**。`grouped_linear.py:764-777` 用单个 `torch.empty(num_gemms, out, in)`
分配一整块 storage，再注册成 `Parameter`；G 个专家权重 = 这块 storage 沿 dim0 的切片
（`large_weight[i]`），天然首尾相接，**不依赖分配器行为**。这与 small 分支的根本差异在于：
small 分支靠 G 次 back-to-back `torch.empty`（`:796-811`），连续性只是分配器顺序的**巧合**，
本机 CPU 实测已证伪（`data_ptr` 不连续）；large 分支靠**单次分配**，连续性是确定性的。

**Q2：KLX 必须做 large-weight 的根因是不是"算子没有非连续访问能力"？**

**不是。** 证据链如下——

XTE 栈上**已经存在吃 per-expert list、不需要拼接的分支**：`XTE_USE_GROUPED_FC_FUSION=1` 时走
`GroupedFcFusionImpl`（`grouped_linear.py:1073-1079`，直接消费 `weight_tensor_list`），
zero cat。所以"算子只能访问连续 HBM"不成立。

真正的机制是**精度兜底，不是能力兜底**：

- `grouped_linear.py:757-762`：开启 `XTE_GROUPED_GEMM_LARGE_WEIGHT` 时，在**构造期**
  就强制 `XTE_USE_GROUPED_FC_FUSION=0`（注释自承：与 fastbf16 inplace 共存有精度回退）；
- 构造期的 `set` 把环境变量钉死，forward 期（`:982`）再读时已为 `0`——mutex 在建模时落下；
- 于是 large weight 路径**被迫**走 `MoeFcFusionImpl`（单 3D 张量接口，`:1080-1084`）。

因此更准确的三层因果是：

1. **触发层**：`GroupedFcFusionImpl`（list 接口）在 fastbf16 inplace 下有已知精度回退
   → 厂商没有修它，而是**用互斥把它关掉**；
2. **承接层**：关掉 list kernel 后只能走 `MoeFcFusionImpl`（单 3D 张量接口），
   而 small 布局下喂这个接口必须 `noop_cat`，`noop_cat` 在本栈又因
   "存储不连续 + XMLIR `set_` 语义问题"**两道闸都退化成真拷贝**；
3. **解法层**：large weight 用"单次分配"把 3D 张量变成**天然存在**，省掉那次每步拷贝。

所以 large weight 不是"KLX 算子访问不了非连续内存"的补救，而是
**"在 list kernel 有精度 bug 被弃用、唯一可用 kernel 只吃连续 3D 张量"这一现状下的结构解**。

**Q3：如果能访问非连续 HBM，是不是直接做 grouped GEMM 就行？**

**是，而且这正是 NV / 昇腾 / CUTLASS 的常规形态。** 真正的 grouped GEMM 定义就是
**pointer-list 接口**——一组专家权重指针（`void*[]` 或 `TensorList`），各专家存储独立、
无需连续、无需拷贝，kernel 按指针逐个寻址。对照：

| 平台 | grouped GEMM 的接口形态 | 是否需要连续/拷贝 |
| --- | --- | --- |
| NV | CUTLASS grouped GEMM（pointer array）/ 上游 TE ops | 不需要 |
| 昇腾 | `torch_npu.npu_grouped_matmul` 吃 list（`mcore_bridge` 里有 `NpuGroupedLoraLinear` 佐证 list 契约存在） | 不需要（待 MindSpeed 源码核实） |
| KL3 `GroupedFcFusionImpl` | list 接口（`w_list`） | 接口上不需要；**kernel 内部实现为厂商二进制，是否内部再打包无法核实** |
| KL3 `MoeFcFusionImpl` | 单 3D 张量 | **需要连续，否则每步真 cat** |

也就是说：KL3 不是"没有"非连续访问形态，`GroupedFcFusionImpl` 的**接口**已经是 list 形态；
它被禁用纯粹是精度问题。若该精度 bug 修复（或换非 inplace 路径验证通过），
理论上可以不用 large weight 而直接用 list kernel 拿到"免拷贝"。
**未验证项**：`GroupedFcFusionImpl` 的二进制内部是否真的逐指针寻址（而非内部再 cat），
这一点源码不可见，不能把"接口是 list"直接等同于"硬件做了非连续访问"。

## 7. 问题分级

| 级别 | 问题 | 位置 |
| --- | --- | --- |
| 高 | 优化器 `param_state` 裸索引重排，错位不可检测，现有断言无法发现 | XME `checkpointing.py` 读/写两侧 |
| 高 | `large_weight` 属性在非 large 布局也可能被设置，使"属性存在"判据失真（本机因 `noop_cat` 退化而不触发，NV 侧会触发） | XTE `grouped_linear.py:1004-1007` + XSwift `lora.py:55` |
| 中 | `safe_merge` 用赋值而非 `copy_`，对视图布局静默丢写（已实测误差 1） | 根因 `mcore_bridge/tuners/lora.py:540`，补丁在 XSwift |
| 中 | `GPTBridge` 原生导入在 large 布局下缺 `weight0`（已实测） | `mcore_bridge/bridge/gpt_bridge.py:867/1044/1153/1255` 等 10+ 处 |
| 中 | L6 detached shim 与 L7/L8 真实属性 shim 语义不等价却叠加，靠 Python shadow 规则耦合 | XME + XSwift |
| 中 | 训练态 loss 对齐与 ckpt 往返一致性均未实测 | 本项目验证缺口 |
| 低 | 3D 一次性初始化对形状相关 `init_method` 不等价 | XTE `grouped_linear.py:764-777` |
| 低 | 专家数口径 L7/L8 不一致 | XSwift `gpt_bridge.py:79-84` |
| 低 | 读盘 `pop` 缺键直接 `KeyError`，无诊断信息 | XME `checkpointing.py:331` |

## 8. 升级到 TE 2.14 / 上游 single_grouped_weight 的可行性

上游方案的完整修改面（**均已在上游完成，不需要我们写**）：

| 位置 | 作用 |
| --- | --- |
| `transformer_config.py:766-774` | 两个 config flag |
| `transformer_config.py:1414-1438` | 前置校验（依赖 `moe_grouped_gemm`、TE>=2.14、fp8+mxfp8、`add_bias_linear`） |
| `core/extensions/transformer_engine.py:1995-2001` | 把 `single_grouped_weight/bias` 透传给 TE `GroupedLinear` |
| `core/extensions/transformer_engine.py:2141-2209` | `_normalize_grouped_parameter_keys` load hook + `_split_grouped_checkpoint_tensor`，**双向**转换 grouped ↔ indexed |
| `core/transformer/moe/experts.py:380-519` | TE ops fusion 路径按 flag 分支取 `weight` 或 `weight0` |
| `core/optimizer/distrib_optimizer.py:1069-1074, 2827-2868` | `_is_grouped_quantized_tensor` + 优化器 state_dict 的 grouped/indexed key 归一化 |
| `training/checkpointing.py:1565-1566` | 两个 arg 的 ckpt 透传 |

**可以删掉的（升级的最大价值）**

- **L4 + L5 全部可删**（约 400 行，XME `checkpointing.py` 的格式转换与优化器索引重排、
  以及 `_sharded_state_dict_grouped` 重写）。上游 `_normalize_grouped_parameter_keys` 配合
  `distrib_optimizer` 的 key 归一化已覆盖双向转换，且其 `_split_grouped_checkpoint_tensor`
  对 `num_gemms` 整除、`unbind`、quantized tensor 三种情形都有处理与显式报错。
  **§7 两条"高"级别风险中最危险的那条（裸索引算术静默错位）由此消除。**
- **L6 可删**：上游不走 `weight{i}` 契约，不需要属性 shim。
- L0 的校验降级可删，改为上游 `__post_init__` 的显式 `ValueError`（fail fast 优于静默降级）。

**不能删的**

- **L7 / L8 必须保留**（或在 mcore_bridge 内做掉）。关键证据：上游 grouped 模式下参数名是
  **`weight`（单数，无索引）**，不是 `weight{i}`。见 `moe/experts.py:383-387`：
  `self.linear_fc1.weight if fc1_single_grouped_weight else self.linear_fc1.weight0`
  ——**上游自己也在每个消费点分支，并未提供 `weight{i}` 兼容 shim。** 因此 mcore_bridge 里
  10+ 处硬编码的 `getattr(layer, f'weight{i}')` 在上游方案下**同样会失败**，XSwift 这两个补丁
  升级后依然必需，只是 shim 目标从 `large_weight` 改成 `weight`。

**阻塞项（结论：现阶段换不过去）**

1. **bf16 被上游直接拒绝**。`transformer_config.py:1432-1436`：`fp4 or not fp8 or fp8_recipe != mxfp8`
   时构造期抛 `ValueError`。KL3 走 bf16/fastbf16，**上游路径在当前训练配置下不可用**。
2. **依赖 TE `GroupedTensor` 接口**。`distrib_optimizer.py:1070-1074` 与
   `extensions/transformer_engine.py:2190-2201` 依赖 `split_into_quantized_tensors()` 等量化存储接口。
   厂商 XTE 要对接上游路径，必须实现这一套，而不只是改参数名。
3. 需要上游 PR NVIDIA/Megatron-LM#4621 解掉非 mxfp8 的数值问题，或自行承担风险放开门控。

**性能是否同等**：原理同等。上游用单一 grouped 存储，同样消除了 per-step 的权重拼接，
KL3 的主收益（消除真 `torch.cat`）在上游路径上同样能拿到——因为参数本身就是单个连续存储。
但上游做这个特性的首要动机是 fp8 量化存储与参数数量，不是省 cat（NV 上 `noop_cat` 本就零拷贝），
所以不应期待上游会为 bf16 路径优先解门控。

**结论（回答"能否更简要地拿到同等精度和性能"）**：

- **方向正确但现在换不了**：升级能删掉 L4/L5/L6（含最高风险的索引算术），L7/L8 仍需保留，
  这是一次实质简化；但 fp8+mxfp8 门控与 `GroupedTensor` 接口依赖两项阻塞，使它在当前
  bf16 训练配置下不可用。
- **建议的中间路线**：不急于升级，优先做两件低成本高价值的事：
  (a) 补 §4 列出的两个实测（loss 对齐、ckpt 往返含优化器配对），把最高风险项从"不可检测"
  变成"有回归"；(b) 把 L8 的根因修回 `mcore_bridge`（`safe_merge` 改用 `copy_`），
  因为这一改对两套方案都有效，且与 THD 同属 bridge 契约缺陷。

## 9. 对 XSwift 退役的结论

XSwift 在 large weight 上只提供 L7 + L8 两个补丁，二者都**不是厂商硬件能力**，而是填
mcore_bridge 对非标准权重布局的两个契约漏洞：

- **L7（HF↔MG 转换）必须有替代**，否则 large weight + MoE 无法导入导出。建议在 mcore_bridge
  内引入统一的 per-expert 权重取用入口（替换 10+ 处 `getattr(layer, f'weight{i}')`），
  这同时兼容 `large_weight` 与上游 `weight` 两种布局；否则保留等价 shim。
- **L8（LoRA merge 写回）建议直接在 mcore_bridge 修根因**（`tuners/lora.py:540` 赋值改 `copy_`），
  而不是继续在外层包。与 THD 的归属判定同一条原则：缺陷在 bridge 本体，不装 XSwift 时依然存在。

另需留意：真正承载 large weight 的是 **XTE（存储/计算）+ XME（ckpt/属性）**，这两个包不在本次
退役范围内。**退役 XSwift 不会让 large weight 失效，只会让 MoE 的 HF 转换与 LoRA 合并失效。**

## 10. 证据位置

- `transformer_engine/pytorch/env_config.py:63-74, 138-166`
- `transformer_engine/pytorch/module/grouped_linear.py:756-806, 857-891, 985-1015, 1064-1090, 1177-1227, 1257-1261`
- `transformer_engine/pytorch/module/_common.py:100-152, 168-191`
- `megatron/core/transformer/transformer_config.py:766-774, 1414-1438`
- `megatron/core/extensions/transformer_engine.py:1995-2001, 2141-2209`
- `megatron/core/transformer/moe/experts.py:380-519`
- `megatron/core/optimizer/distrib_optimizer.py:1069-1074, 2827-2868`
- `megatron/training/checkpointing.py:1565-1566`
- `xmegatron_ext/envs.py:68`，`xmegatron_ext/plugin.py:45-51, 105-117, 182-188`
- `xmegatron_ext/components/training/arguments.py:127-142`
- `xmegatron_ext/components/training/checkpointing.py:53-73, 296-410, 413-491, 494-641`
- `xmegatron_ext/components/core/extensions/transformer_engine.py:16-155`
- `xswift_ext/plugin.py:74-90`
- `xswift_ext/components/mcore_bridge/bridge/gpt_bridge.py:1-105`
- `xswift_ext/components/mcore_bridge/tuners/lora.py:1-66`
- `mcore_bridge/bridge/gpt_bridge.py:181-229, 780-1259`
- `mcore_bridge/tuners/lora.py:499-546`
- 版本基线：`transformer_engine.__version__ == 2.12.0+c9c3d0d9`（低于上游要求的 2.14.0）
- 实测证据：`/root/dl/TTorch/.benchmarks/xswift_full_20261029/`（15 项 rank-case）
---

## 11. ckpt 补丁归因与 swift 路径缺口（2026-10-30）

### 11.1 XME 全部 ckpt 相关补丁：门控、用途、是否 large-weight 引入

以下逐条经过**源码核实**（非仅注释）。`xmegatron_ext/plugin.py` 的 ckpt/优化器相关补丁：

| 符号 | 挂载点 | 门控 | 真实用途（代码核实） | large-weight 引入？ |
| --- | --- | --- | --- | --- |
| `megatron.training.checkpointing._load_base_checkpoint` | `_common_replace_training:45-51` | `XTE_GROUPED_GEMM_LARGE_WEIGHT` | 读盘 per-expert→large stack + 优化器索引合并 | **是** |
| `TEGroupedLinear._sharded_state_dict_grouped` | `_xpu_replace_core:105-111` | `XTE_GROUPED_GEMM_LARGE_WEIGHT` | large_weight 的 dist-ckpt 分片（ep/tp 轴） | **是** |
| `megatron.training.checkpointing.generate_state_dict` | `_xpu_replace_training:182-188` | `XTE_GROUPED_GEMM_LARGE_WEIGHT` | 存盘 large→per-expert split + 优化器索引重排 | **是** |
| `FileSystemWriterAsync.preload_tensors` | `_xpu_replace_core:100-104` | XPU 全程 | XPU 同步 D2H staging | 否 |
| `FileSystemWriterAsync.write_preloaded_data` | `_xpu_replace_bridge_conversion:309-313` | `XME_BRIDGE_CONVERSION_PATCH` | 转换时序列化异步写、降峰值显存 | 否 |
| `FileSystemWriterAsync.prepare_write_data` | 同 `:314-318` | `XME_BRIDGE_CONVERSION_PATCH` | 延迟 D2H 到 bucket flush | 否 |
| `apply_factory_merges` | 同 `:335-340` | `XME_BRIDGE_CONVERSION_PATCH` | factory merge 就地化，避免冗余张量物化 | 否 |
| `load_preprocess` | 同 `:341-344` | `XME_BRIDGE_CONVERSION_PATCH` | 加载 state_dict 预处理（转换显存控制） | 否 |
| `megatron.core.optimizer._get_param_groups` | `_common_replace_core:19-20` | 无条件 | MoE+PP 下优化器 child 顺序跨 stage 对齐 | 否（MoE+PP 通用） |

**结论：ckpt 的修改里恰好 3 个是 large-weight 引入的，且全部是 `XTE_GROUPED_GEMM_LARGE_WEIGHT` 门控。**
其余 ckpt 相关补丁是 Bridge 转换显存控制（4 个）与 XPU D2H staging（1 个），与 large-weight 无关；
`_get_param_groups` 是 MoE+PP 通用正确性补丁，也与 large-weight 无关。

### 11.2 关键：3 个 large-weight ckpt 补丁里，被 swift 旁路的恰好是 2 个函数级的

| 补丁 | 层级 | 触发条件 | swift 路径是否触发 |
| --- | --- | --- | --- |
| `_sharded_state_dict_grouped` | **类方法** | 谁调 `TEGroupedLinear.sharded_state_dict()` 都触发 | **触发**（模型分片正确） |
| `generate_state_dict` | megatron.training **函数** | 只有走 `megatron.training.checkpointing.save_checkpoint` 才触发 | **不触发** |
| `_load_base_checkpoint` | megatron.training **函数** | 只有走 `megatron.training.checkpointing.load_checkpoint` 才触发 | **不触发** |

swift 的实际路径（`swift/megatron/utils/megatron_lm_utils.py`）：

- 存：`save_mcore_checkpoint:255` → swift 自己的 `_generate_state_dict:124`（`:143` 直接 `models[i].sharded_state_dict()`）→ `dist_checkpointing.save:285`。
- 取：`load_mcore_checkpoint:442` → swift 自己的 `_generate_state_dict` → `dist_checkpointing.load:462`。
- 调用方：`trainers/base.py:116/753`（训练）、`convert.py:61/91/120`（转换）、`export.py:41/96/116`（导出）。

**已被代码核实**：swift 的 `_generate_state_dict` 不 import / 不调 `megatron.training.checkpointing.generate_state_dict`；`load_mcore_checkpoint` 不调 `_load_base_checkpoint`。

### 11.3 由此产生的盘上格式分叉（large-weight + MoE + swift 的 ckpt 缺口）

因为只有"类方法分片"在 swift 路径触发、而"函数级格式转换"不触发：

| | megatron training 路径 | swift 路径（KL3 实际） |
| --- | --- | --- |
| 模型盘格式 | per-expert `weight{i}`（`generate_state_dict` wrapper 拆开） | `large_weight` 单键（没人拆） |
| 优化器索引 | 重排成 per-expert（高风险算术） | 保持运行时 large 顺序（没跑那段算术） |
| 自洽性 | 自洽 | 自洽（swift→swift 反而绕开高风险算术） |

**缺口 = 跨栈不兼容**（逻辑强推，待运行时实证）：

1. megatron training 存的 per-expert ckpt → swift 加载：swift 运行时按 `large_weight` 键取，盘上是 `weight{i}`，`dist_checkpointing.load` 键不匹配 → 加载失败/错载。上游 `_normalize_grouped_parameter_keys` hook 救不了（它是 `load_state_dict` 的 hook，dist 加载在键匹配阶段就失败）。
2. swift 存的 `large_weight` ckpt → megatron training 加载：`_load_base_checkpoint` 找 `weight0` stack，找到的是 `large_weight` → 空转 → 对不上。
3. 同一运行时两套盘格式，无版本标记调和，只能靠 key 名碰巧区分。

典型踩坑场景：megatron 预训练出 per-expert ckpt → ms-swift MoE SFT → resume 即挂。

### 11.4 把握度与待实证

- **代码已核实**：XME 3 个补丁的 large-weight 门控、swift 旁路两个函数挂点、类补丁两路径都触发、模型分片两路径正确、4 个 Bridge 补丁与 `_get_param_groups` 均与 large-weight 无关。
- **待运行时实证**（需多卡 XPU，当前设备不稳定）：两路径各存一个 large-weight MoE dist-ckpt，diff key 集合（`weight{i}` vs `large_weight`）+ 交叉加载，把"格式分叉"从推断变成实测。

### 11.5 对 XSwift 退役的含义

- 这 3 个 ckpt 补丁都在 **XME**（不在退役范围），退役 XSwift 不影响它们。
- XSwift 自己的 ckpt 触点只有 L7（`_set_mlp_state` HF↔MG）和 L8（LoRA merge），与上述 ckpt 格式转换是两件事。
- **但** §11.3 的格式分叉是 XME 的**挂载位置**问题（挂在 `megatron.training` 而 swift 旁路），与是否退役 XSwift 无关——它在本就运行的 swift 路径上就是一个待补缺口。
