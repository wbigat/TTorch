# TTorch 方案背景与厂商栈事实（2026-10 实测校准）

## 厂商栈分层（KL3）

- **torch_xmlir**：kernel 物理本体（`torch.ops.custom_ops.*`，libxmlir_custom_ops.so /
  libxpu_dnn.so）+ eager 层符号替换（symbrewrite）。永远开启，不受环境开关控制。
- **xmegatron_ext**：把 torch_xmlir kernel 包成 megatron 融合层（SwiGLU/RoPE/MoE a2a 的
  autograd.Function）。开关 `XME_PLUGIN_ENABLE`。
- **xswift_ext**：纯 swift 框架集成（trainer/checkpoint/lora/数据），无自有 kernel。
  开关 `XSWIFT_PLUGIN_ENABLE`。

## symbrewrite 符号替换（eager 路径）

- 机制：import 钩子全局替换，原实现存 `__origin_{name}`（可恢复，"去掉厂商 patch"实验靠它）；
  `is_wrapper=True` 表示 bind 期调工厂取真身。默认只开 torch/torch_vision/torchaudio/transformers。
- transformers 插件替换清单：**RMSNorm 类 7 条**（Qwen2/Qwen2.5VL/Qwen3/Qwen3VL/Qwen3.5/
  Qwen3.5MoE 的 RMSNorm + Gemma4 专门 wrapper，均换成厂商 RMSNormFunction）+ **RoPE 函数
  9 条**（FusedRoPEFunc，BLHD/BHLD 布局）+ 框架兼容 2 条（transformers 5.x）。
- **推论：eager 路径 RMSNorm/RoPE 已是融合态，provider 在其上只有机制价值（~1x）。
  净收益选题差集（torch_xmlir/nn/ 28 个封装减去已替换）：swiglu → dropout_add_rms_layernorm
  / bias_dropout_add → clip_grad → softmax 族。**（口径：eager 路径；megatron 路径
  SwiGLU/RoPE 已被 xmegatron_ext 融合。）
- 副作用：替换产物是局部类 → KL3 栈上 `torch.save(整模型)` 本就失败（与 TT 无关，
  state_dict 不受影响）。

## RMSNorm kernel 细节

- `rms_layer_norm(x, eps, is_rstd, scale, *, y, rstd)` 纯前向（无 grad_fn）+
  `rms_layer_norm_backward` 配套反向；rstd 为 fp32 输出缓冲（shape 带不带 trailing (1,)
  实测无差）。
- Qwen3.5 是 **zero_centered_gamma**：weight 零初始化、有效 scale=`1+w`（直乘 w 实测
  err 达 5.16）。
- 厂商封装 `torch_xmlir.nn.rms_norm.RMSNormFunction` 内建 zcg/fp32 rstd/needs_input_grad；
  TTorch 的 kl3.py 保留自包 `_RMSNormFn` 作为教学参照（用户决策 2026-10-15），两者输出
  逐位一致。

## 厂商 kernel 的已知缺陷（实测发现，2026-10-15）

- **`rotary_pos_emb_backward` 与 megatron 原生公式语义不等价**：fwd 完全一致
  （fp32 err 2.4e-7），bwd 梯度错（bf16 轨道错 2.43、梯度落位模式不同，疑似
  rotate_half 的 chunk/interleaved 约定在反向不一致；**fp32 轨道直接产 nan**）。
  XME 无证绑定把该 kernel 直接接入训练——实测整网 loss 偏移（rope 原生组
  1.4588x vs 厂商组 1.45833，同 seed 同数据）。**TT 的准入对账拒绑了它
  （正确姿态）；向厂商报修前，rope 保持原生。** 教训：厂商 kernel 的 fwd 对
  不代表 bwd 对，训练态必须 fwd+bwd 都验。

## 平台探测（catalog）

- KL3 卡名是通用 "gpu"；native 探测通道 = `torch_xmlir` 包存在性（find_spec 只查不导入）。
- masquerade 栈 `torch.cuda.is_available()` 报 True——detect cuda 需厂商包防御排除。
- TT 的 `.pth` 与厂商 `.pth` 顺序取决于文件名**字母序**（ttorch 须在 torch_xmlir 之后）。

## 关键设计决策（为什么这么设计）

- **能力自报表废除**："接口在不在"是实现的事实，`supports()` 从子类覆盖派生；自报=
  不可信输入复述事实，两头成本。残留调用 `capabilities()` 会炸出半激活态。
- **双身份**：`vendor_name`（厂商，provider 分发用）≠ `device_name`（torch 认得的类型，
  框架 `device.type` 比较存活线）；masquerade 栈 device_name 必须 `"cuda"`。
- **激活时序证据制**：不猜 import 先后（厂商 .pth 早于一切用户代码），激活末尾按对象
  身份 `verify_sweep()` 验收，残留即 fail fast。
- **观测与操作分离**：witness 只取证不改绑定；bridge/provider 的事件统一进
  `witness.provenance` 一条事件流。
