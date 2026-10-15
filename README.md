# TTorch

TTorch是面向异构设备的训练执行收束层，其职责是将上层训练框架的基础功能接入与模块级性能优化解耦，并以统一、可验证、可追踪的方式管理多硬件执行。

## 当前状态

**已穿透**：Qwen3.5-0.8B 在 KL3-XPU 上，由 TT 顶替厂商插件（xswift_ext /
xmegatron_ext）跑通完整 swift-megatron SFT —— 训练脚本一行不改，`exit 0`，
43/43 步，checkpoint 正常；与厂商插件的 loss 偏差为零均值噪声（正号 24 /
负号 19，均值 +2.4e-05）而非系统性 bias。

开发方针是**穿透优先、不铺开面**：完备性现阶段放低，未纳入 policy 的 API
直接报未实现，先把一条端到端链路打穿再扩面。当前 policy 14 个符号。

**provider 已起步**：bridge 之外补齐了性能实现层与观测层——首个真实 provider
（RMSNorm，KL3）跑通"自动注册→自动生效→自动选择"的整网使能（数值与厂商逐位一致），
并收拢观测/验证/溯源为独立的 witness 层。实测结论与经验教训见
`docs/TT_Provider_Function_Map.md` §6。

## 目录

```
ttorch/              正式包（四层收束，顶层只做转发）
  __init__.py          统一入口：activate / device_spec / provider_wrap / check_bindings
  _auto_init.py        .pth 钩子：环境变量控制的自动激活
  bridge/              CUDA 接口兼容（cuda_compat 符号决策）+ bootstrap（激活编排）
                       + checkpoint（save/resume + schema 校验 + RNG）
  platform/            Platform Manager：平台注册/检测/门禁/生命周期 + torch 接口暴露
  provider/            module 级性能实现：两层 Provider/ProviderImpl + 自动注册/生效/选择
  witness/             观测/验证/溯源：统一 provenance + bridge API 验证 + 运行期中转
                       + provider 精度/性能验证原语
tests/               测试与验收（分层说明见 tests/README.md）
scripts/             工程工具（打包安装等）
tt-bridge-demo/      mock 机制演示（假硬件走通 Bridge 设计意图，含 red 用例）
tt-schedule-demo/    schedule provider 机制演示
tt-provider-demo/    真实硬件（KL3）provider 机制演示（RMSNorm 门禁/选择/回退/整网使能）
docs/                架构设计与真实环境踩坑记录（G 系列）
```

## 快速开始

```bash
bash scripts/build_and_install.sh    # 打包 + 安装 + 校验

# 方式一：零侵入（.pth 自动激活，用户代码不动）
export TT_PLUGIN_ENABLE=1 XSWIFT_PLUGIN_ENABLE=0 XME_PLUGIN_ENABLE=0
python your_training_script.py

# 方式二：显式激活（脚本最开头，所有框架 import 之前）
import ttorch; ttorch.activate()
```

默认不激活——装了包不改变行为，这是可安全合入的前提。

## 设计要点

四层收束，依赖单向（bridge→platform、provider→platform、bridge/provider→witness）：
**bridge** 功能接入（CUDA 调用点接管 + checkpoint）、**platform** 硬件抽象、
**provider** module 级性能实现、**witness** 观测/验证/溯源。

- **两类决策分离**：`cuda_compat` 管框架 CUDA 调用点（remap / fallback /
  exception），Platform Manager 管平台注册/检测/激活与生命周期，统一接口是
  后者的顺带产物；
- **provider 失败只降性能不降功能**：每个 module provider 自带 native 底线
  （torch 原生单算子），高性能 impl 经三级门禁（功能/精度/收益）选用，
  失败/关闭即回退 native，可 kill switch 整体退场；
- **不支持即不实现，禁止身份判定**：接口默认抛 unsupported，平台不实现即不支持
  （`supports()` 内省存在性）；可选能力由 conformance 实测派生，不经平台自报；
  身份（vendor/device）由 spec 承载、实现类不许自报；
- **两层门禁**：admission 验绑定正确性，conformance 验 adapter 契约履行，
  任一不过拒绝激活；
- **惰性导入**：register 只登记轻量 spec，厂商实现在平台确定后才 import——
  厂商包 import 有副作用，且 PrivateUse1 设备模块是运行时注册的。

详见 `docs/`。真实环境踩坑（G9~G15）记录在
`docs/TT_Swift_Cuda_API_Inventory.md`，每条含症状、根因与一般化结论。
