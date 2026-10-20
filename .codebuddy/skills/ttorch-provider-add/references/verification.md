# TTorch 验证方法论与坑判据（血泪教训，逐条来自真实事故）

## 五条必守

1. **松的收敛检查掩盖数值错**：过拟合单 batch 时数值偏了 loss 也照降。验数值等价必须
   "同权重、仅前向、loss 逐位对齐"。
2. **forward-only 的提速是假象**：backward 没真跑时整网假性变快（曾把 2.5x 误判为 ~1x
   的反向）。测收益必须 fwd+bwd 都真跑。
3. **推理 kernel ≠ 训练可用**：无 autograd 的 kernel 必须包 `autograd.Function`；
   检查 `y.grad_fn is not None` 且梯度有值。
4. **变体陷阱**：zero_centered_gamma 等改变 scale 语义（1+w vs w）；wrap 自检按实测
   对齐探测，不信 flag（厂商替换可能丢 flag）。
5. **先查厂商已融合清单**再写 provider：已融合处性能持平 ~1x，净价值在未覆盖处。

## 口径与判据

- 精度：abs/rel err < 5e-3。bf16 下同一语义不同算法路径有 eps 级舍入差（rel≈5e-3 正常）；
  语义错配是 O(1) 相对差——自检用相对口径（max_err/ref 幅度，阈值 0.05）区分两者。
- 性能：厂商已融合算子等价区间 0.85~1.15x；未覆盖算子预期 >1.1x 才值得启用。
  多次测量先看 run 间噪声量级。
- 吞吐探测：warmup 10 + reps 200、末尾一次同步；sync 经平台注入（不经 bridge 激活时
  `torch.cuda.is_available()` 在 xpu-native 为假，会计时失真）。
- 门禁探测不动全局 RNG 种子（门禁在训练循环中途触发，manual_seed 会污染训练）。

## 归因纪律

- 先做**未接入对照组**（不激活 TT / bypass=True）再归因到 TT。实例：KL3 栈
  `torch.save(整模型)` 失败是厂商局部类替换所致，与 TT wrap 无关。
- 实验判定阈值与实测物理量对齐（例：RMSNorm 占整网比例有限，"慢 >5% 且远超噪声"
  即可证融合有效，">10%"偏严）。
- 新踩的坑固化回本文件或 SKILL.md（收尾规则）。

## 实测基线（2026-10-15，KL3 / Qwen3.5-0.8B / transformers 5.12.1）

- RMSNorm 整网：TT provider vs 厂商融合 0.994x（等价）；同权重前向 loss 逐位一致；
  未融合 native 慢 9.1%。
- gate 证据：vendor-fused probe 23.8us vs native 91.5us（3.84x，err=0.0）；反面夹具
  sloppy（err 1.31e-01 被拒）/ no-benefit（0.20x < min_speedup 被拒）。
- 单算子 bwd：自包 `_RMSNormFn` vs native dx err ≈ 1.9e-06。
