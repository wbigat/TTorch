# 测试与验收脚本

按**验证范围**分层，不按技术形态分。每层回答一个不同的问题，上层通过不代表下层可省。

| 层 | 问的问题 | 依赖 | 耗时 |
|---|---|---|---|
| `unit/` | 单个机制是否成立 | 无（mock） | 秒级 |
| `integration/` | 真实 torch + 真实硬件下契约是否成立 | 加速卡 | 分钟级 |
| `e2e/` | 真实模型 / 真实框架 / 真实数据能否跑通 | 加速卡 + 模型 + 数据集 | 十分钟级 |

工程工具（打包安装等）在仓库根目录的 `scripts/`，不属于测试分层。

mock demo（`tt-bridge-demo/`、`tt-schedule-demo/`）是**机制演示**而非测试，
保留在原位：它们用假硬件把 Bridge/Provider 的设计意图走一遍，包括正常路径
与 red 用例（门禁拦截、冲突 fail fast），是设计文档的可执行版本。

## integration

```bash
cd tests/integration
XSWIFT_PLUGIN_ENABLE=0 python real_sft_smoke.py                 # 自动检测平台
XSWIFT_PLUGIN_ENABLE=0 python real_sft_smoke.py --platform cpu  # 无卡机器
```

自造 TinyLLM（~1.8M），验 P0 接口族的契约与**跨设备精度**（加速卡 fp32 对
CPU fp32 参考 rel_err < 5e-3）。模型自造是有意的：精度对账需要逐位可控的
参考实现，真实模型做不到。

## e2e

```bash
cd tests/e2e
XSWIFT_PLUGIN_ENABLE=0 python real_qwen_smoke.py                 # 真实 Qwen3.5-0.8B
bash run_swift_megatron.sh                                       # 真实 swift-megatron SFT
TT_MODE=0 bash run_swift_megatron.sh                             # 对照组（原厂商插件）
```

`real_qwen_smoke.py` 覆盖：admission 绑定 / 数据 loss mask / 训练收敛 /
checkpoint resume 连续性 / schema 守门 / **运行期中转证据**（证明框架调用
真的经过 TT，而非仅符号指向）。

`run_swift_megatron.sh` 不复制生产训练脚本，只套环境变量后转调——生产脚本
含集群路径与 API Key，不进仓库。前置条件见脚本头部注释。

**对照组是归因工具**：接框架后遇到任何报错，先用 `TT_MODE=0` 跑一遍。实测
踩过一次——`SEQ_LEN=2048` 下 `get_rope_index` 崩，对照组同样崩在同一行，
是参数与样本长度不匹配，与 TT 无关。

## 打包（仓库根目录 scripts/）

```bash
bash scripts/build_and_install.sh    # 打包 + 安装 + 校验
```

开发期重装必须走它：直接 `pip install` 踩过两个静默坑（`build/lib` 残留、
editable 安装残留遮蔽），表现都是"代码改了但装进去的是旧的"且不报错。

## 环境变量

| 变量 | 含义 |
|---|---|
| `TT_PLUGIN_ENABLE=1` | `.pth` 自动激活 TT（默认 0，装了包不改变行为） |
| `TT_PLATFORM` | 指定平台，默认 auto 检测 |
| `XSWIFT_PLUGIN_ENABLE=0` `XME_PLUGIN_ENABLE=0` | 关闭厂商插件，让 TT 顶替 |
| `TT_QWEN_MODEL` / `TT_SFT_DATA` | 覆盖 e2e 的模型与数据路径 |
