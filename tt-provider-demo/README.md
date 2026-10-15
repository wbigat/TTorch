# tt-provider-demo：真实硬件上的 module 级性能 provider 验证

验证 `ttorch.provider`（性能增量层）在真实硬件（KL3 / xpu-masquerade）上的完整机制。
设计依据 `docs/TT_Provider_Function_Map.md`（§6 是 as-built 实施状态与经验教训）。

## 与 tt-bridge-demo 的分工

- `tt-bridge-demo`：mock 虚拟硬件，验证 **bridge** 的功能可达机制。
- `tt-provider-demo`：**真实硬件**，验证 **provider** 的性能增量机制——用真实
  RMSNorm provider（torch 原生分解 vs KL3 厂商 AOT 融合 kernel）。

## 运行

```bash
# 环境：KL3 真机（torch_xmlir 厂商栈在场），TT_PLUGIN_ENABLE=0 之外无需特殊配置
cd tt-provider-demo
python demo.py                      # 单模块：门禁/选择/回退/provenance 机制
python demo_e2e.py                  # 整网（Qwen2）：自动注册/生效/选择 + A-B 性能
python demo_replace_torchxmlir.py   # 等价性：TT provider vs torch_xmlir 的 RMSNorm
```

## 三个 demo 各验证什么

| demo | 验证点 |
|---|---|
| `demo.py` | 单模块机制：三级门禁（功能/精度/收益）拦住坏候选、kind 优先级选厂商融合、kill switch 回退 native 功能无损、统一 provenance（bridge+provider 一份事件流） |
| `demo_e2e.py` | 整网自动化：`tt.activate()` 自动注册 → `tt.provider_wrap(model)` 自动发现 patch 全部 RMSNorm → 首次前向自动选择；A-B（bypass）对比原始 vs provider；数值逐位一致 |
| `demo_replace_torchxmlir.py` | 等价性：torch_xmlir 已把 RMSNorm 符号替换为融合实现，本实验对比 A(厂商融合)/B(TT provider 融合)/C(未融合)，验证 TT provider 能否替代厂商优化拿到等价整网性能 |

## 实测结论（诚实，详见功能地图 §6.2）

- **数值正确**：provider 选中厂商融合后整网前向 loss 与 baseline 逐位一致（bf16 err=0），训练正常收敛。
- **性能持平（~1x），非加速**：KL3 的 torch_xmlir 已融合 RMSNorm（同源 kernel），
  本 provider 与之同源同速。价值在统一可插拔机制 + 厂商未预融合处补融合。
- **关键教训**：训练态 provider 必须包 autograd.Function（厂商 forward-only kernel
  无梯度）；zero_centered_gamma 变体要折有效权重 `1+w`。

## 结构

```
demo.py / demo_e2e.py / demo_replace_torchxmlir.py   # 三个验证脚本
provenance.jsonl                                      # 运行产物（统一 provenance 导出，gitignore）
```
被验证的实现在 `ttorch/provider/`（生产包），本目录只有驱动脚本。
