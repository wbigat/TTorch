# Bug Report：`rotary_pos_emb_backward` 部分旋转（partial rotation）反向梯度计算错误

> 日期：2026-11-04 ｜ 报告方：TAIJI 训练平台（TTorch 项目组）
> 算子：`torch.ops.custom_ops.rotary_pos_emb_backward`（`libxmlir_custom_ops.so`）
> 调用链：`xmegatron_ext.components.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd`
> → `RotaryPositionalEmbeddingWithFreqFunction.backward`
>
> 2026-11-05 归档：自 `experiments/b8_rope_backward/` 移至
> `tests/integration/vendor_bugs/b8_rotary_pos_emb_backward/`，供与厂商对齐直接引用。

## 环境

| 组件 | 版本 |
|---|---|
| torch | 2.9.0+cu129 |
| hydrax | 0.1.0 |
| xmegatron-ext | 0.18+20260922.ge680013cb |
| transformer-engine | 2.12.0+c9c3d0d9 |
| 硬件 | KL3 XPU ×4 |

## 现象

部分旋转（`rot_dim < head_dim`，即 `freqs.shape[-1] < t.shape[-1]`）时，
**前向逐位正确，反向旋转段梯度错误**；pass-through 段梯度正确（恒等，误差精确为 0）。
全旋转（`rot_dim == head_dim`）前反向均正常。

## 实测矩阵（复现脚本输出，2026-11-04）

输入 `t` [s=128, 1, h=4, d=64]，`freqs` [128, 1, 1, rot]，参考为纯 torch 手写
非交错 RoPE 公式（与 megatron native 同语义、零代码共享）：

| 场景 | fwd_err | bwd_err（grad_scale≈4.8） | rot 段 | pass 段 | 判定 |
|---|---|---|---|---|---|
| 全旋转 fp32 contiguous | 2.4e-7 | 1.0e-5 | — | — | ✅ |
| 全旋转 fp32 非连续 | 2.4e-7 | 1.0e-5 | — | — | ✅ |
| 全旋转 bf16 | 0.0 | 3.1e-2（≈1 ulp） | — | — | ✅ |
| **部分旋转 fp32（rot=32）contiguous** | 2.4e-7 | **8.9** | **8.9** | **0.0** | ❌ |
| **部分旋转 fp32（rot=32）非连续** | 2.4e-7 | **8.9** | **8.9** | **0.0** | ❌ |
| **部分旋转 bf16（rot=32）** | 1.6e-2 | **8.9** | **8.9** | **0.0** | ❌ |

要点：

1. **fp32 下误差 ≈ 梯度量级的 2 倍**——不是精度噪声，是实现逻辑错；
2. **误差仅出现在 rot 段**，pass 段恒等正确（误差精确为 0）——定位到反向
   kernel 对 rot 段的处理；
3. **误差幅度随张量布局/分配器状态变化**（不同 harness 下 fp32 观测到
   8.9 / 10.97 / 206 不等）——疑似反向 kernel 按全宽 stride/offset 处理
   rot 段，读到了非预期内存。

## 最小复现

`repro_rotary_pos_emb_backward.py`（同目录，零 TT 依赖，仅 torch + 厂商栈）：

```bash
python tests/integration/vendor_bugs/b8_rotary_pos_emb_backward/repro_rotary_pos_emb_backward.py
# 期望：3 OK（全旋转）/ 3 BUG（部分旋转），exit code 1
```

## 诉求（两条，按优先级）

1. **修复部分旋转的反向梯度**（rot_dim < head_dim 时 rot 段 grad 计算）；
2. **若短期不修复：对部分旋转显式报"不支持"**——前向对、反向错的静默
   数值错误是最危险的形态（loss 照降、权重静默污染，且前向检查无法发现）。
   kernel 在 `rot_dim != head_dim` 时直接 raise 远好过返回错误梯度。

## 影响面

- **Qwen3 全系**：RoPE 为全旋转，**当前不受影响**；
- **MLA 类模型**（DeepSeek 系等部分旋转 RoPE）：训练路径**不可用**——
  前向正常、反向静默错误，是比直接报错更危险的形态；
- 我们侧（TTorch）已用「梯度契约」机制在准入层挡住该变体（前向对+反向缺/错
  的候选无法通过门禁），本报告旨在推动 kernel 本体修复。

## 附：关键定位过程

1. 初测整体 bwd_err 巨大（10.97），fwd 逐位正确；
2. 分段定位：pass 段误差精确为 0（恒等正确），rot 段误差 ~2× 梯度量级；
3. 误差幅度随布局/分配器状态漂移 → 疑似 stride/offset 误用读取非预期内存；
4. 全旋转所有组合（含非连续、bf16）均正常 → 缺陷条件精确为
   `freqs.shape[-1] != t.shape[-1]`。
