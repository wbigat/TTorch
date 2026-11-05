# B6 验证报告：`HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT` 行为与性能对比

> 日期：2026-11-05 ｜ 硬件：KL3 XPU ｜ torch 2.9.0 / hydrax 0.1.0 / torch_xmlir（生产同栈）
> 复现脚本：`verify_mem_efficient.py`（同目录）
> 结论先行：**该行为由 `HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT` 环境变量控制**；
> 开关两态数值逐位一致、权重无改写；mem-efficient=ON 峰值显存显著更低
> （bf16 省 224 MiB / fp32 省 448 MiB）；速度差 ~1.00x（未复现"减少反向开销"）；
> 生产 swift+megatron 路径（XME/XSwift 均关）不经此流程，env 在生产是 no-op。

## 1. 厂商口径（2026-11-05 提供）

> fast FC 为计算高效需要对权重做一个**转置**；训练过程可以 **inplace**
> （减少峰值显存 + 减少反向开销），eval 过程不需要。

XSwift/XME 的 evaluate 保护就是在 eval 期间设置
`HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT=1`，finally 恢复原值
（`xswift_ext/components/common/utils.py`、`xmegatron_ext/components/core/utils.py:46-66`）。

## 2. 控制点核实（代码级）

- env 读取点在 **hydrax `libHydraOps.so`**：符号
  `hydra::xpu::env::disable_fast_fc_memory_efficient()`，配套 kernel 参数
  `memory_efficient`（strings 实测命中）；Python 侧同名 helper 见于
  `hydrax.xaccelerator.stateful.disable_fast_fc_memory_efficient`。
- Python 层消费者（谁设这个 env）：**XSwift evaluate wrapper**、
  **XME `disable_fast_fc_memory_efficient_wrapper`**（`xmegatron_ext/plugin.py:173-179`）、
  xdeepspeed。三者在生产配置（`env.sh`：`XME_PLUGIN_ENABLE=0`、XSwift 关）下**全部不在场**。
- kernel 本体消费者：XME tensor_parallel linear
  （`xmegatron_ext/components/core/tensor_parallel/layers.py`）经
  `hydrax.xaccelerator.linear.{linear_fwd,linear_bwd_*}` 与
  `fc_fusion`，并用 `state_of(weight).memory_efficient = True`
  （cast cache 开、非共享权重时）允许 inplace 行为。
- **生产路径不到达**：torch_xmlir 全库唯一 `fc_fusion` 调用在 apex
  wgrad 插件（`symbrewrite/plugins/apex/fused_weight_gradient_mlp_cuda/
  wgrad_gemm_accum.py`，仅 TE fused-wgrad 权重梯度路径）；`F.linear` /
  `nn.Linear.forward` / `te.Linear.forward` 运行时核实均未被 reroute 到
  hydrax。生产 swift+megatron（mcore-bridge，megatron core 原生层）linear
  不走 fc_fusion → **该 env 在生产路径无作用对象**。

## 3. 实验设计

直接调用 hydrax `linear_fwd` / `linear_bwd_{dgrad,wgrad,bgrad}`，逐字复制
XME `LinearWithGradAccumulationAndAsyncCommunication` 的调用方式
（stateful 上下文 + `memory_efficient=True`，无通信），对比两种模式：

| 模式 | env | 含义 |
|---|---|---|
| ON | 不设置（默认） | mem-efficient 生效（训练 inplace 口径） |
| OFF | `HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT=1` | XSwift eval 保护态 |

形状 `[tokens=8192, hidden=4096] × [4096, 4096]`，bf16 与 fp32 两组；
20 warmup + 100 iters 取中位；`torch.cuda.max_memory_allocated` 记峰值。

## 4. 实测结果

### 4.1 正确性 / 布局变异（train → eval → train 序列）

两种模式、两种 dtype 下全部成立：

- weight 的 `stride / is_contiguous / data_ptr` 在 train1、eval、train2 后**均不变**
  （torch 可见层无 inplace 转置痕迹——inplace 发生在 hydrax stateful 内部缓冲，
  不触及用户权重张量）；
- weight 逻辑值逐位不变；train2 的 out/grad_w/grad_in 与 train1 **逐位一致**；
- eval 输出与 train 前向**逐位一致**；
- **跨模式**（ON vs OFF）out / grad_w / grad_in / eval_out 全部**逐位一致**。

### 4.2 性能与峰值显存

| dtype | 阶段 | ON 时延 | OFF 时延 | ON/OFF 加速比 | ON 峰值 | OFF 峰值 | OFF−ON |
|---|---|---|---|---|---|---|---|
| bf16 | train fwd+bwd | 3909.6 µs | 3908.4 µs | 1.000x | 896.0 MiB | 1120.0 MiB | **+224.0 MiB** |
| bf16 | eval fwd | 1333.7 µs | 1332.9 µs | 0.999x | 768.0 MiB | 992.0 MiB | **+224.0 MiB** |
| fp32 | train fwd+bwd | 7076.2 µs | 7077.8 µs | 1.000x | 1536.3 MiB | 1984.3 MiB | **+448.0 MiB** |
| fp32 | eval fwd | 2616.9 µs | 2669.9 µs | 1.020x | 1344.1 MiB | 1792.1 MiB | **+448.0 MiB** |

显存差定量自洽（bf16 train）：x(64) + grad_out(64) + grad_in(64) + w(32)
= **224 MiB** 整；fp32 翻倍为 448 MiB——OFF 模式为参与转置的每个张量
多持有一份转置副本，ON 模式 inplace 复用，与厂商"减少峰值显存"口径**精确吻合**。

## 5. 结论

1. **控制变量确认**：行为由 `HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT` 控制，
   hydrax C++ 层动态读取；XSwift/XME 的 eval 保护 = eval 期间置 1。
2. **显存口径证实**：mem-efficient=ON 省峰值显存（每参与张量省一份副本），
   train/eval 都生效（厂商说"eval 不需要"是需求口径，不是行为差异——
   实测 eval 下 ON 同样省显存且数值逐位一致）。
3. **速度口径未复现**：本形状下 train/eval 均 ~1.00x（fp32 eval 1.02x 边缘），
   "减少反向开销"在 8192×4096 规模未观测到；如需可向厂商追问受益形状区间。
4. **生产路径无关**：XME/XSwift/xdeepspeed 均关时，swift+megatron 的 linear
   不经 hydrax fc_fusion，该 env 是 no-op；XSwift eval 保护在生产配置下
   本就无保护对象。B6 维持"已关闭（不承接）"。
