# examples：可运行案例

放**独立可跑的验证案例**（不是库代码）。每个案例演示 TT 某一层机制在
真实或模拟环境下的完整链路，同时充当回归入口——改库代码后跑一遍即可确认
机制没被打坏。

依赖关系：案例只依赖 `ttorch` 本体（已安装，或 `PYTHONPATH` 指向仓库根），
彼此独立、不互相引用；案例内路径全部用 `__file__` 自定位，故目录可整体移动。

## bridge-demo（mock 全链路，无加速卡也能跑）

演示 **Bridge 的 CUDA 兼容面**：mock 出一个假厂商栈（非标接口名、CUDA/XPU/GCU
三种方言），验证"框架用 torch 原生写法调用 → 经门面落到厂商实现"的整条链路：
符号绑定、from-import 扫尾、准入测试、checkpoint/RNG、生命周期。

```
cd examples/bridge-demo && python3 demo_bridge.py      # mock 全链路（无卡可跑）
cd examples/bridge-demo && python3 demo_provider.py    # provider 接入形态
```

## provider-demo（真实硬件，KL3 实测基线）

演示 **Provider 的性能增量面**：真实 KL3 上的门禁、开关、回退与溯源。

| 脚本 | 演示什么 |
|---|---|
| `demo.py` | 三级门禁（功能/精度/收益）+ kill switch + bypass + provenance（含反面夹具） |
| `demo_e2e.py` | 整网端到端：接管前后对比（厂商已融合平台上的口径是等价，不是提速） |
| `demo_replace_torchxmlir.py` | A/B/C 三态等价性实验（厂商融合 / TT 融合 / 未融合） |

## schedule-demo（机制演示，通信与计算 overlap）

演示 **schedule 类 provider**：把通信与计算的时序安排（comm/compute thunk 编排）
作为可替换实现，验证 TP 通算 overlap 形态。当前是概念与机制演示，不进生产路径。

## 案例与单元测试的分工

案例是**机制演示 + 端到端回归**（要能讲清"为什么这样设计"，有可读输出）；
逐函数的边界条件归 `tests/unit/`。案例失败优先看机制是否被破坏，不是看阈值。
