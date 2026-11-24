# big_leaf 双线实体版 PR 说明（2026-11-23）

> 本文是 big_leaf「双线实体版」一揽子改动的完整 PR 文档：设计意图、方案细节、
> 核心代码思想、实现拆解、demo 验证、整网验证、自审修复与已知边界。
> 深度设计演进与历史取舍见《big_leaf_方案设计与实现_20261109.md》§12–§13，
> 本文自足，可作评审单文档使用。

---

## 0. TL;DR（一屏看完）

**做了什么**：把 MoE GroupedLinear 的 dist-opt/DDP 参数实体从「每专家一个
`weight{i}`」换成「内部线大实体 `big`」，同时外部线 `weight{i}` 保持注册并以
`.data` 视图共享同一份存储——**性能/显存拿到 LW=1 的全部收益，外部可见性
（state_dict 键、查询、HF/模型 checkpoint 导出）零感知**。

**核心交付（同轮三臂交错实测，P800）**：

| 口径 | 结果 |
|---|---|
| dist-opt ON DP1（big/lw1 total） | **0.987** |
| dist-opt ON DP4（big/lw1 total） | **1.005** |
| dist-opt ON optimizer 相位（DP1/DP4） | 0.992 / 0.988 |
| dist-opt ON 结构（与 LW=1 完全一致） | DDP buffer 参数 256→8、optimizer shard（DP1 256→8 / DP4 64→2） |
| dist-opt OFF（手工优化器，big/lw1_nopin） | **0.995** |
| 冻结/LoRA 形态（big/lw1_nopin，fwd+bwd） | **1.027**（旧 1.18–1.29，pack-once 回收） |

**验证状态**：单测 184 全过；集成五形态（distopt）+ 四向 checkpoint + TP/EP 三配置
+ LoRA + LoRA-distopt 全 PASS；整网（run.sh 拉起）全 exit=0，HF 导出无污染。

---

## 1. 设计意图（为什么做）

### 1.1 问题：big_leaf 只有"权重"是双线，梯度和 optimizer 参数仍在外部线

big_leaf 已把 G 个专家权重 repack 成一块连续存储、`weight{i}` 以视图共享。但
Megatron dist-opt 场景下，DDP buffer 与 DistributedOptimizer 看到的仍是 G 个
`weight{i}` 实体。对照 LW=1（`XTE_GROUPED_GEMM_LARGE_WEIGHT=1`，厂商原生单
大权重），§12.3 实测出两项结构性残差：

| 阶段 | 旧 big_leaf | LW=1 | 后果 |
|---|---|---|---|
| dist-opt 参数实体 | G 个 `weight{i}`（DP4 每 rank 64 shard 张量） | 1 个大权重（2 shard） | optimizer 单实体收益零回收 |
| DDP buffer 元数据 | G 个 param 条目 | 1 个 | DDP 元数据缩减零回收 |
| big.grad | 自建 flush hook 拆 G 份 `add_` 进 `main_grad` | 整体一次 | 非标准路径、复杂度/风险 |

实测残差：DP1 总步 big/lw1=1.179（optimizer 相位 +6.77ms/step）、DP4 big/lw1=1.046
（+1.77ms/step）。**差距不在 kernel，而在参数实体的数量级**。

### 1.2 目标与约束

- **目标**：性能/显存对齐 LW=1（dist-opt 开关两种形态、可训与冻结两种形态）。
- **硬约束**：对外契约（`weight{i}` 注册、state_dict 键、HF 导出、旧 checkpoint
  可迁移）一律不动；不得引入静默损坏风险。
- **取舍**：LW=1 走厂商 pin 路径，存在「原地覆写 Parameter 导出视图」的 checkpoint
  污染风险（退役文档 §16.3）——big_leaf 生产入口刻意关 pin 规避之，因此**不能简单
  复用 LW=1 的注册形态**，必须自造一条不依赖 pin 的双线。

### 1.3 核心洞察（用户给定思路，本 PR 落地）

> **双线进行**。对外一条线（原始 parameters 与 optimizer 视角不变）；对内一条线
> （大权重、大梯度、大优化器参数）。两线通过 torch 的 `.data` 视图共享内存。
> 对外访问/查询/导出 ckpt 都是正常 module，不做额外感知；内部流程访问聚合后的
> big-leaf 实体。既要性能与显存，又保住对外的易用性与代码内聚性。

---

## 2. 方案设计细节

### 2.1 总体架构：两线共享一份存储

```
外部线（世界看到的，永不改变）                内部线（训练基础设施看到的）
  module.weight{i}  ──.data=切片视图──►   ┌────────────────────────────┐
  module.bias{i}    ──.data=切片视图──►   │ DDP param_data 连续区        │
  state_dict / HF 导出读 weight{i}        │  big    (G, out, in) 实体    │◄─ 单一 DDP buffer 参数
  module.parameters() 对用户原样          │  big_b  (G, out)     实体    │◄─ 单一 optimizer 实体
  （查询/导出/键契约零感知）              │  big.grad 整体 accumulate     │◄─ 标准 DDP grad hook
                                          │  Adam m/v 按 big 形态存      │◄─ 大优化器参数
                                          └────────────────────────────┘
```

- **外部线**：`weight{i}`/`bias{i}` 始终注册在 module 上，`state_dict` 键、HF 导出、
  查询语义完全不变；它们的 `.data` 被重绑为内部线存储的切片视图，读到的是实时值。
- **内部线**：每个 GroupedLinear 一个 `big`（含可选 `big_b`）——独立的
  `nn.Parameter`，**不注册进任何 module**。它同时是 DDP buffer 的参数实体（梯度
  整体 accumulate/reduce-scatter 一次完成）和 DistributedOptimizer 的参数实体
  （Adam 状态按大实体建，shard 数比 per-weight{i} 少 G 倍）。

### 2.2 关键机制：entity_swap（实体换装）

要让内部线实体被 DDP 与 optimizer「看见」，遇到 mcore 的一个硬约束：

- DDP 构造（`distributed_data_parallel.py`）从 `module.named_parameters()` 收集参数，
  并**断言 layout 的键 == 该集合**；
- `get_megatron_optimizer` 同样从 `model_chunk.named_parameters()` 收集 param_groups；
- 而 `nn.Module.named_parameters()` 的递归实现**直读子模块的 `_parameters` 字典**，
  不调用子模块同名方法（2026-11-19 实测，方法级覆盖完全无效）。

因此换装必须发生在 `_parameters` 层，且只在边界调用窗口内：

```python
@contextlib.contextmanager
def entity_swap(self):            # megatron.py
    for module in self.modules_to_bind:
        orig = module._parameters
        swapped = {n: p for n, p in orig.items()
                   if not (n.startswith("weight") or n.startswith("bias"))}
        swapped["weight"] = big            # 内部线实体顶替
        swapped["bias"] = big_b
        module._parameters = swapped
    yield                                 # DDP / optimizer 构造在此窗口内
    module._parameters = orig             # 窗口外恢复，外部零感知
```

**三个边界窗口**（每处都复用同一 `entity_swap()` 原语）：

| 边界 | 落点 | 目的 |
|---|---|---|
| DDP 构造 | `runtime.py` 的 `ddp_init_wrapper` | buffer 参数实体 = `big`，layout 键一致 |
| optimizer 构造 | `optimizer.py` 的 `optimizer_boundary_wrapper`（swift trainer 方法） | param_groups 捕获 `big` 引用 |
| optimizer state 序列化 | `optimizer.py` 的 `mcore_sharded_state_dict_wrapper`（mcore 方法） | save/load 的 lookup dict 按实体重建 |

窗口内 `_parameters` 不含 `weight{i}`（若被误访问会 AttributeError），但两个消费者
只做泛型参数遍历；`state_dict()` 只在窗口外调用，键契约不受影响。

### 2.3 实体形态：为什么必须是 `nn.Parameter` 而非 plain tensor

mcore buffer 用 `param.data = 视图` 重绑存储（`param_and_grad_buffer.py`），只有
`nn.Parameter` 的 `.data` setter 才真正换 storage。实体不注册进任何 module，因此
**不会进入 TE 的 kernel pin 路径**（pin 遍历的是 module `_parameters`）——原
plain-tensor 的「避开 pin 污染」安全理由依然成立。

### 2.4 冻结模块（LoRA base / full 的后续 case）

`prepare()` 把整模块冻结的 GroupedLinear 归入 `frozen_modules`，`attach_ddp` 用
独立 buf（复用 `_repack_one`）绑定，**只拿 forward 收益**（forward 直取 + 免 cat），
不进 DDP buffer / optimizer（无梯度、无 Adam 状态）。LoRA base 即此形态；LoRA 适配器
（`lora_A`/`lora_B` 本身也是 GroupedLinear）照常走双线实体路径。`restore` 后冻结
权重保持独立 buf 视图（视图 refcount 保活 storage，不悬空、无需 clone，见 §8.1）。

### 2.5 optimizer checkpoint：fully-reshardable + 格式闸门 + 旧格式迁移

- **格式闸门**：双线实体版只支持 fully-reshardable。`dp_reshardable` 的桶内 pad
  尺寸计算与实体 layout 不兼容（实测 `torch.empty(负数)` 崩溃），显式 fail-fast，
  不许静默写出坏 checkpoint；生产脚本需 `dist_ckpt_optim_fully_reshardable=true`。
  save/load 措辞按 `is_loading` 分流（P2-2）。
- **序列化换装**：`sharded_state_dict` 的 lookup dict 在换装窗口内用 mcore 的
  plain-module fallback（`state_dict(keep_vars=True)` +
  `make_sharded_tensors_for_checkpoint`）重建，使 optimizer 侧
  `param_to_sharded_metadata` 的键 == 内部线实体。
- **旧格式迁移**（`distopt_state.py`）：fully-reshardable `param_state` 是
  `{param_idx: {state_key: tensor}}` 位置字典，两种格式（expert 粒度 per-weight{i}
  ↔ entity 粒度 big）的位置序都可确定性复刻（`stock_positions`/`entity_positions`），
  `convert_param_state` 做 dim 0 的 **stack/slice** 完成粒度转换。读盘用**源格式参考
  实例**（fresh model+DDP+optimizer，同架构、格式相反）的 mcore 原生
  `sharded_state_dict`——不需原始训练实例；公开入口 `migrate_param_state_to_target()`
  封装位置序复刻 + 转换（见 §8.1 的 1d 澄清）。

---

## 3. 主要代码思想

1. **双线视图共享**：外部线保持注册、内部线持有存储，`.data` 切片视图连接——
   一份存储同时满足「对外无感知」与「对内高性能」。
2. **换装即最小侵入**：不 fork mcore，只在三个边界窗口临时让 `named_parameters()`
   产出内部线实体，窗口外恢复。换装原语 `entity_swap()` 三处复用，逻辑零重复。
3. **删繁就简**：内部线实体成为标准 DDP/optimizer 参数后，旧版「flush hook 拆
   `main_grad`」整套机制（~60 行）**删除**——`big.grad` 回归标准路径，
   `overlap_grad_reduce`/`no_sync` 原生兼容，少一台自管理状态机。
4. **fail-fast 而非静默兜底**：格式不兼容（dp_reshardable）、签名漂移、绑定破坏
   （`assert_healthy`）、冻结 pack-once 的 pin 风险，一律显式拒绝而非隐式降级。

---

## 4. 代码实现（模块拆解）

生产核心共 6 个文件、1856 行（`ttorch/bridge/big_leaf/`）：

| 文件 | 行 | 职责（本轮改动） |
|---|---:|---|
| `megatron.py` | 475 | **核心重写**：`BigLeafMegatronPlan` 双线实体——`_build_proxies`（构造内部线实体 + `_propagate_param_attrs` 属性传播）、`entities()`（DDP/optimizer 可见实体集）、`entity_swap()`（换装原语）、`compute_full_param_layout`（按实体建 layout）、`attach_ddp`（外部线重绑视图 + 冻结模块绑定）、`restore()` |
| `optimizer.py` | 171 | **新增**：两个边界补丁——`optimizer_boundary_wrapper`（optimizer 构造换装）、`mcore_sharded_state_dict_wrapper`（序列化换装 + fully-reshardable 闸门 + `_assert_sd_signature` 签名守卫 + 多模型/TP/EP 范围闸门） |
| `distopt_state.py` | 218 | **新增**：旧↔新格式迁移——`stock_positions`/`entity_positions`（位置序复刻）、`convert_param_state`（expert↔entity 粒度 stack/slice）、`migrate_param_state_to_target`（生产可调用迁移入口）、`build_param_state_template`（布局诊断） |
| `core.py` | 762 | `_repack_one` 加 hook/main_grad 容忍参数（冻结绑定用）；冻结 + pin 关时大叶子转 `nn.Parameter`（pack-once，回收 item 7）；`_check_binding_runtime` 版本契约（检测外部写权重） |
| `runtime.py` | 68 | `ddp_init_wrapper`：layout 改传 `plan.entities()`，DDP 构造纳入 `entity_swap()` 窗口 |
| `swift.py` | 162 | 注册两个 optimizer 边界补丁（补丁计数 5→6） |

**改动规模**（git diff，含测试/文档/基准）：14 个已跟踪文件 +606/-244（首轮双线）
+ 二轮 review 修复，另新增 4 个文件（`optimizer.py`、`distopt_state.py`、
`test_big_leaf_distopt.py`、`test_big_leaf_review_fixes.py`）与 4 份基准证据 JSON。

---

## 5. 性能结果（核心交付）

### 5.1 dist-opt ON（`bench_distopt_gap.py`，同轮三臂交错分相位，中位数 us）

DP1（4 层 × 32 专家，2048→1024，4096 tokens）：

| arm | fwd | bwd | grad_sync | **optimizer** | **total** |
|---|---:|---:|---:|---:|---:|
| stock | 6388 | 39341 | 440 | 23361 | 70417 |
| **big_leaf** | 4585 | 29946 | 418 | **14039** | **49302** |
| lw1 | 4133 | 30750 | 486 | 14152 | 49931 |

→ big/lw1 total **0.987**、optimizer **0.992**；big/stock total **0.700**。

DP4：

| arm | fwd | bwd | grad_sync | **optimizer** | **total** |
|---|---:|---:|---:|---:|---:|
| stock | 6618 | 39699 | 21876 | 16580 | 87964 |
| **big_leaf** | 4395 | 31042 | 21198 | **13898** | **72921** |
| lw1 | 4298 | 30519 | 21788 | 14063 | 72525 |

→ big/lw1 total **1.005**、optimizer **0.988**；big/stock total **0.829**。

**结构数字（收益确实回收，与 LW=1 完全一致）**：

| 指标 | stock | big_leaf | lw1 |
|---|---:|---:|---:|
| DDP buffer 参数 | 256 | **8** | 8 |
| optimizer shard 张量（DP1 / DP4） | 256 / 64 | **8 / 2** | 8 / 2 |
| big_leaf bindings | 0 | 8 | 0 |

### 5.2 dist-opt OFF（`bench_module_gap.py`，手工优化器六臂回归）

big/lw1_nopin **0.995**（旧 1.008）——手工路径无回退。

### 5.3 冻结 / LoRA 形态（pack-once 回收 item 7）

冻结 fwd+bwd（G32，受控基准）：big/lw1_nopin **1.027**（旧 1.18–1.29），
`big_leaf_nocheck` 0.975。机制：TE 对 `nn.Parameter` 权重走常驻转置缓存、plain
tensor 每次 fresh pack；冻结 + pin 关时大叶子转 `nn.Parameter`，交错实测
param/plain = **0.771**。安全论证：pin 关无导出污染路径，pin 开保持 plain tensor
（宁慢 23% 不污染导出视图）。

---

## 6. 完整 demo 验证

### 6.1 单元测试（`tests/unit`，184 全过）

big_leaf 相关 6 个文件；本轮新增 `test_big_leaf_distopt.py`（8 例，CPU-only）：
param_state 转换形状/字节序等价、`sharded_state_dict` 签名守卫（真实签名锚定 +
漂移 fail-fast + 工厂安装期校验）、格式闸门 save/load 措辞分流、无 plan 零侵入透传。
（163 → 171 见 §13.6，171 → 184 见 §13.7 二轮 review。）

### 6.2 集成测试（`tests/integration/big_leaf`，torchrun 多卡）

| 测试 | 覆盖 | 结果 |
|---|---|---|
| `distopt.py` | DP1/DP2 × 基础/overlap/runtime-wrapper：forward 逐位、owned shard 梯度逐位、no_sync 累积 2×、step 后权重与视图 ptr、结构断言（owned⊆实体、shard 按实体级） | 5 形态全 PASS |
| `distopt_checkpoint.py` | fully-reshardable **四向组合**（stock/big × stock/big）+ 跨格式转换 + 模板键对照；resume 后 `train_step` + 逐位 loss/param 对拍 | 4 组合全 PASS |
| `tp_ep.py` | TP2×EP2 逐模块 forward/grad 逐位 + 整网 finite smoke | PASS |
| `lora.py` | 手工 LoRA（真实 mcore_bridge LoraParallelLinear） | PASS |
| `lora_distopt.py` | LoRA + DP2 dist-opt：forward 逐位、step 后权重逐位、merge/unmerge 逐位，冻结 base 绑定 + 适配器实体（bindings=3） | PASS |

**精度口径是强断言**：checkpoint resume 后先 `train_step` 再逐位比 loss 与全部
参数——optimizer state 转换错误必然暴露（不是只看"跑通"）。

---

## 7. 整网验证（`bash run.sh` 拉起，4×P800，Qwen3-MoE 合成 BF16 资产）

复用退役 e2e 的参数化脚本副本（`.benchmarks/big_leaf_dual_20261119/script/`），
`megatron sft` 整网拉起，验证 dist-opt 开关 × big_leaf 开关 × LoRA 五连：

| run | 配置 | 步数 | loss | exit | 备注 |
|---|---|---:|---|---:|---|
| **A** | big_leaf ON，dist-opt ON（fully-reshardable） | 6 | 12.56→10.94 | 0 | 6 步 + eval + optimizer ckpt 保存 |
| **B** | big_leaf OFF（stock），dist-opt ON | 6 | 12.56→10.94 | 0 | 参照臂；首步 loss 与 A 逐位一致（12.56361103） |
| **C** | big_leaf OFF，**dist-opt OFF** | 3 | 12.56→12.22 | 0 | 手工优化器路径 |
| **D** | big_leaf ON，dist-opt ON，**resume from A 的 ckpt** | 8 | →10.41 | 0 | 生产 resume：从实体格式 optimizer ckpt 续训，loss 从 10.94 连续降到 10.41 |
| **E** | big_leaf ON，**LoRA**（TUNER_TYPE=lora） | 3 | 12.56→12.58 | 0 | 冻结 base 绑定生效（`modules=24`，旧版为 0） |

- 五连全 exit=0，loss 轨迹正常（A/D 连续下降，B 与 A 末步 10.94156265 vs
  10.94063663，第三位小数级一致，为 6 步累积序差）。
- **HF 导出无污染**：runE（LoRA，pin 关生产配置）导出 231 键 **0 非有限值**，
  权重 absmax 0.111——验证了 pack-once 的 pin-gate 不引入 §16.3 污染。
- **二轮 review 修复后复验**（2026-11-23）：runF（full + dist-opt + save）exit=0、
  loss 12.16→10.94（与 runA 一致），新增的多模型/TP/EP/版本/格式闸门在生产
  路径（TP=1、EP=1、单模型）正常放行；runG（LoRA + merge/unmerge）exit=0、
  `modules=24`，**版本契约守卫零误报**，HF 导出 231 键 0 非有限值。

---

## 8. 自审修复（评审后落码，红绿齐全）

提交前 infra 评审识别 2 P1 + 3 P2，其中 4 项按红绿流程修复（红阶段 5 失败、绿
阶段 8/8、全量回归 163→171、四向 ckpt 复测全 PASS）：

| 编号 | 问题 | 修复 |
|---|---|---|
| P1-1 | `sharded_state_dict` 按位置镜像 mcore 签名转发，mcore 升级会**静默错位**（B17 复核查不到签名漂移） | 新增 `_assert_sd_signature()`：安装期 `inspect.signature` 锚定，漂移即 fail-fast |
| P1-2 | `restore` 只对冻结模块 clone 脱离，可训 `weight{i}` 仍指 DDP param_data 视图，生命周期契约隐式 | restore 加 `detach_storage` 参数 + docstring 显式契约（DDP 存活期 ⩾ weight{i} 使用期） |
| P2-1 | `convert_param_state` expert→entity 用 `cat`，名义形状 `(G*out,in)` 与实体模板 `(G,out,in)` 不符 | 改 `stack`，名义形状自洽，flatten 字节序不变 |
| P2-2 | 格式闸门 load 路径错误信息按 save 侧措辞，误导排障 | 按 `is_loading` 分流 save/load 措辞 |
| P2-3 | prepare 瞬时 ~3× 显存峰值 | 未改代码，登记为 §13.5-5 待实测项 |

### 8.1 复审修复（2026-11-23 二轮，外部 review 指认 6 项，红绿齐全）

二轮外部 review 指认 6 个问题 + 性能口径提醒，逐项**真实复现确认后修复**（新增
`test_big_leaf_review_fixes.py` 13 例，红阶段 8 失败确认缺陷存在；全量回归 171→184）：

| # | 问题 | 修复 |
|---|---|---|
| 1a | `_name_map` 把普通 Linear/LayerNorm 的 weight/bias 当 weight{i}，`int("")` 崩 | 只认数字后缀专家参数，普通直通 |
| 1b/1c | 转换器忽略目标 shape（专家数不符静默产错形状）；entity→entity 空 stack | 校验声明 shape fail-fast；entity→entity 恒等直通 |
| 1d | 迁移读盘依赖源实例、生产无可调用 API | 读盘走**源格式参考实例**（mcore 原生）；手搓模板降级为布局诊断；新增公开入口 `migrate_param_state_to_target()` |
| 2a/2b | 序列化多模型覆盖、TP/EP 元数据未覆盖 | ckpt 边界 fail-fast 多模型/TP>1/EP>1 |
| 3 | 格式闸门只查 metadata，但 mcore **显式 sharding_type 优先**（可绕过） | 闸门改按 mcore 同一有效值判 |
| 4 | 实体只复制 `allreduce`，TP/分组属性未传；lr/wd 等价性 | `_propagate_param_attrs` 传安全属性、不照抄 per-expert 分片 spec；docstring 明确 name/shape 级 lr/wd 需重验证 |
| 5a | 共享 storage ≠ 共享 `_version`，外部写权重 backward 静默用错值 | `_check_binding_runtime` 版本契约检测；sanctioned 路径（merge/load）重记基线 |
| 5b/5c | 「防悬空」clone 解释过强（.data 别名保活）；restore 语义应分开 | **收敛掉** P1-2 的 `detach_storage`+冻结 clone（错误前提的过度设计），分清存储生命周期 vs 训练有效性 |
| tp_ep | TP-only/EP-only 用外部参数查内部 DDP 映射 | 改用 `plan.entities()`+实体→参考梯度映射，~1.25M 元素逐位对拍 |

**1d 诚实闭环**：手搓 `build_param_state_template` 的裸 `from_rank_offsets` 缺
fully-reshardable 真实分片元数据，实测驱动不了 load（`Invalid access pattern`）；
要修复等于重写 mcore 内部构造（过度设计），故改走参考实例机制。完整记录见设计文档
§13.7；定位与口径澄清（非框架无关、compile 未验收、性能基准口径）见 §13.8。

---

## 9. 已知边界与后续（如实登记）

1. **dp_reshardable 格式 fail-fast**（生产需 `dist_ckpt_optim_fully_reshardable=true`），
   未做兼容——如需支持要单独论证实体 layout 的桶内 pad 对齐。
2. **optimizer ckpt 的多模型/TP>1/EP>1 未验收即 fail-fast**（`_tp_ep_world_sizes`
   闸门）：`_rebuilt_sharded_state_dict` 用 plain-module fallback（整模型按
   replicated），只覆盖单模型 chunk + TP=1 + EP=1；扩展时需保留原生 metadata 只转换
   受影响实体。
3. **`restore` 语义**：摘除后**不能继续训练**（DDP 梯度实体是内部线，hook 与参数
   实体不匹配）——这是训练有效性问题；`weight{i}` 保持内部线存储视图，refcount
   保活 storage、不悬空（存储生命周期与训练有效性是两回事，§13.7-5b/5c）。
4. **prepare 瞬时 ~3× 显存峰值**（big 中转 + 原 weight{i} + param_data 并存，构造后
   回落 1×）——大 MoE（35B-A3B 量级）峰值待实测登记。
5. **部分冻结不支持**：只支持整模块可训或整模块冻结，`prepare()` 对部分冻结
   fail-fast。
6. **lr/wd 自定义分组**：换装窗口内实体名/维数变化，按 `weight{i}` 名字模式或
   per-expert shape 的分组规则不命中实体，需在实体层重写；按属性的规则有效（属性
   已传播）。**迁移读盘需构造源格式参考实例**（非全自动内联检测）。

---

## 10. 兼容性与回滚

- **外部契约不变**：`weight{i}`/`bias{i}` 注册、state_dict 键、HF 导出、查询语义
  与 stock 完全一致（整网 HF 导出键集合与数值无污染实证）。
- **旧 checkpoint 迁移**：per-weight{i}（expert 格式）↔ big（entity 格式）双向
  迁移转换器，四向组合 resume 逐位一致，无需源实例在场。
- **生产开关**：`TT_BIG_LEAF`（1 安装双线实体 attach，0 stock A/B）；dist-opt
  checkpoint 需 `dist_ckpt_optim_fully_reshardable=true`。
- **回滚策略**：优先新进程（restore 后不继续训练）；`TT_BIG_LEAF=0` 即回到 stock。

---

## 11. 改动清单

**生产代码**（`ttorch/bridge/big_leaf/`）：`megatron.py`（重写）、`optimizer.py`（新增）、
`distopt_state.py`（新增）、`core.py`、`runtime.py`、`swift.py`。

**测试**：`tests/unit/test_big_leaf_distopt.py`（新增 8 例）、
`tests/unit/test_big_leaf_review_fixes.py`（新增 13 例，二轮 review 守卫）、
`test_big_leaf_swift.py`（补丁计数）、`test_big_leaf_runtime.py`；集成
`distopt.py`、`distopt_checkpoint.py`（迁移走公开 API）、`tp_ep.py`（TP-only/EP-only
映射修复）、`lora_distopt.py` 适配双线 API。

**基准**：`experiments/big_leaf_lw_gap/bench_distopt_gap.py`、`bench_module_gap.py`
（双线适配）+ 证据 `out/distopt_dp1_dual.json`、`distopt_dp4_dual.json`、
`module_gap_dual.json`、`module_gap_frozen_dual.json`。

**文档**：`big_leaf_方案设计与实现_20261109.md`（§13 全节 + §13.7 二轮 review +
§13.8 定位澄清）、本 PR 说明、`待办总览`（回归口径 184）。

**规模**：首轮双线 14 个已跟踪文件 +606/-244 + 二轮 review 修复（8 文件 +243/-80）；
新增 4 个文件 + 4 份基准 JSON。
