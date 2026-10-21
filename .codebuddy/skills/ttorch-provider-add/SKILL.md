---
name: ttorch-provider-add
description: TTorch 新增性能优化 provider 模块（模块级高性能实现）：选题、native 基准、平台 impl、门禁与 wrap 自检、实测验证。当新增一个模块语义的 provider（swiglu/dropout_add_rms_layernorm/softmax 族等）、写平台 ProviderImpl、调门禁/wrap 逻辑、排查 provider 数值或性能问题时使用。关键词：provider、ProviderImpl、native、门禁、wrap_model、融合算子、厂商 kernel、min_speedup、probe、zero_centered_gamma、RMSNorm、SwiGLU。
---

# TTorch 新增性能优化 Provider

provider 层职责：性能实现（native 底线 + 各平台高性能 impl + 三级门禁 + 两级回退）。
依赖方向：provider → platform / witness；函数级接入机制 `provider/fn_binding.py`
同层闭环（性能替换的执行归 provider——§8：Bridge 不负责模块性能替换）。

## 三级接入形态（同一套 Provider 抽象，只是接线器不同）

| 层级 | 适用 | 接线器 | 实例 |
|---|---|---|---|
| **L1 函数级**（算子符号） | 框架内部的函数调用点（megatron 的 swiglu/rope 等非 nn.Module） | `Provider.bind_points` 声明 `(模块路径, 符号名)`；activate 时注册到 `provider/fn_binding.py` 的后置钩子，绑定后符号 = `FunctionDispatcher`（`manager.call` 分发） | swiglu、bias_swiglu、rope_bshd |
| **L2 模块级** | nn.Module（RMSNorm 等） | `Provider.module_names` + `wrap_module`（forward → manager.call） | rmsnorm |
| **L3 编译级**（预留） | 局部修改过、无法固定替换的模块 | `kind="fused_jit"` 的 ProviderImpl（torch.compile 产物） | 未实现（依据：XME_SWIGLU_USE_OKEANOS_DL 互斥路线） |

关键：**函数级绑定的目标是 dispatcher 而非厂商实现本身**——三级门禁（功能/
精度/收益）、负载签名缓存、两级回退、kill switch、bypass、provenance 全部共享。
绑定层不做数值判断（测量与选择解耦）。

## 门禁口径（fp32 精度轨道 + 生产 dtype 性能轨道）

- **精度对账在 fp32 轨道**（判数学等价，fwd+bwd 都验）：bf16 下同语义实现差
  （1-3 ulp，实测 swiglu 达 3e-2）与语义偏差分不开；fp32 下 ulp 级 vs 语义差。
  **bwd 必验**——厂商 rotary_pos_emb kernel fwd 逐位一致但 bwd 错 2.43（实测），
  只验 fwd 会漏。
- **性能测量用生产 dtype**（probe 跟随首次真实调用现场，两轨道分开）。
- **负载签名 = (ndim, 特征维, dtype, device) + 标量参数**：特征维变化重新门禁
  （shape-sensitive 防护），序列/批次维归并（变长训练每步重门禁不可接受）。

## 0. 选题：先查厂商已融合清单（最重要的一步）

读 `references/vendor-stack.md`：eager 路径已被 symbrewrite 替换的算子（RMSNorm/RoPE）
**只有机制价值**（性能持平 ~1x）；净收益在未覆盖处。当前差集顺序：**swiglu →
dropout_add_rms_layernorm / bias_dropout_add → clip_grad → softmax 族**。
选题结论写进 impl 的 `intent`。

## 1. 语义与调用约定

一个语义一个文件夹 `ttorch/provider/impls/<语义>/`：`base.py`（Provider + native）+
`<平台>.py`（ProviderImpl）。调用约定写进模块 docstring，发布后不改——wrap 层负责把
模块语义折算进约定（如 zcg 折成有效权重 `1+w`）。

## 2. native（功能底线 + 精度/收益基准）

torch 原生单算子流程，`kind="reference"`，永远启用不可禁用。**语义必须与被替换的原始
模块等价**（如 HF RMSNorm 的 fp32 upcast）——否则 kill switch 回退即精度漂移，且门禁
拿 native 当基准、结构上测不出。写之前先读框架原始实现。

## 3. 平台 impl（ProviderImpl）

- 元数据必设：`name` / `platforms` / `kind` / `version` / **`min_speedup`（缺省 0.0
  收益门禁恒过）** / `build` / `intent`。
- 厂商算子引用写在 `__call__` 体内（非该硬件机器 import 安全）。
- **推理 kernel ≠ 训练可用**：无 autograd 的 kernel 必须包 `autograd.Function`
  （fwd 快 kernel + bwd 反向 kernel）。教学参照：`impls/rmsnorm/kl3.py` 的 `_RMSNormFn`。
- `supports()` = 逐调用适用性谓词（融合 kernel 只覆盖部分 shape/dtype），不满足安静
  回退 native，不许抛异常。

## 4. 注册

`impls/__init__.py` 的 `SEMANTICS` 登记；注册 eager、门禁/选择惰性（首次 call 才跑）。

## 5. wrap 与门禁（改这段逻辑时的要点）

- 接管前 `_self_check` **两判据分离**：变体识别（w / 1+w 候选取误差最小者，
  选择题不设阈值）+ 正确性准入（最小误差 < 1e-3 才接管，判断题）。判定在
  **fp32 轨道**做（模块临时 upcast，往返无损）——bf16 下同语义实现差（≈6e-3）
  与语义偏差（实测 4e-2）只差一个数量级分不开，fp32 下拉开 5 个数量级。
- 门禁三级（功能/精度 5e-3/收益），测量在 `witness/verify.py`、编排在 `manager._gate`
  （测量与选择解耦）。probe 跟随首次真实调用现场：**完整复制** shape/dtype/device
  与非张量参数（eps 等）及 kwargs；设备走 `handle.device_spec()`；不动全局 RNG 种子；
  门禁整段在回退保护内（挂了回退 native）。
- **门禁与绑定缓存按负载签名分键**（`(impl, sig)` / `(semantic, slot, sig)`）：
  门禁结论只对测量过的负载成立，shape/标量参数变了必须重新门禁（shape-sensitive
  impl 换 shape 静默算错的实测教训）。热路径签名计算是几个 tuple/str 操作，
  不逐调用同步、成功只计数。

## 6. 实测验证（不验不收工，细则见 references/verification.md）

1. 单算子对账：impl vs native，**fwd+bwd 都真跑**（forward-only 提速是假象）；
2. `examples/provider-demo/demo.py`：门禁 + 开关 + provenance（含反面夹具）；
3. 整网等价性：`demo_e2e.py`；对比厂商替换时用 `demo_replace_torchxmlir.py` 的
   A/B/C 三态范式；
4. `tests/e2e/real_qwen_smoke.py`：端到端收敛 + checkpoint 连续性。

数值判据：同权重仅前向 loss 逐位比对（不看"loss 在降"）。性能判据：厂商已融合算子
0.85~1.15x 为等价，未覆盖算子预期 >1.1x。

## 收尾

删临时文件；运行产物（provenance.jsonl 等）不入 git；新坑固化回本 skill 的
references（验证方法论/厂商栈清单）。
