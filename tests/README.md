# 测试与验收脚本

按**验证范围**分层，不按技术形态分。每层回答一个不同的问题，上层通过不代表下层可省。

| 层 | 问的问题 | 依赖 | 耗时 |
|---|---|---|---|
| `unit/` | 单个机制是否成立 | 无（mock） | 秒级 |
| `integration/` | 真实 torch + 真实硬件下契约是否成立 | 加速卡 | 分钟级 |
| `e2e/` | 真实模型 / 真实框架 / 真实数据能否跑通 | 加速卡 + 模型 + 数据集 | 十分钟级 |

工程工具（打包安装等）在仓库根目录的 `scripts/`，不属于测试分层。

demo 是**机制演示**而非测试，保留在原位，是设计文档的可执行版本：

- mock demo（`examples/bridge-demo/`、`examples/schedule-demo/`）用假硬件把 Bridge/Provider
  的设计意图走一遍，包括正常路径与 red 用例（门禁拦截、冲突 fail fast）。
- `examples/provider-demo/` 用**真实硬件**（KL3）验证 provider 性能增量机制——真实
  RMSNorm provider 的门禁/选择/回退/provenance + 整网自动使能（`tt.activate()`
  自动注册 → `tt.provider_wrap(model)` 自动生效 → 首次前向自动选择）。详见其 README。

e2e 的 `real_qwen_smoke.py` 已默认开启 provider 自动使能（`--no-provider` 可关）。

## integration

```bash
cd tests/integration
XSWIFT_PLUGIN_ENABLE=0 python real_sft_smoke.py                 # 自动检测平台
XSWIFT_PLUGIN_ENABLE=0 python real_sft_smoke.py --platform cpu  # 无卡机器
```

自造 TinyLLM（~1.8M），验 P0 接口族的契约与**跨设备精度**（加速卡 fp32 对
CPU fp32 参考 rel_err < 5e-3）。模型自造是有意的：精度对账需要逐位可控的
参考实现，真实模型做不到。

`integration/vendor_bugs/` 归档**厂商算子缺陷**的最小复现与报告（零 TT 依赖，
可直接发厂商）：当前有 `b8_rotary_pos_emb_backward/`（部分旋转反向梯度错，
复现退出码 1 为预期）。

### big_leaf 真实硬件集成测试

从仓库根执行，并显式指定当前源码，避免导入旧安装包：

```bash
export PYTHONPATH="$PWD"
python tests/integration/big_leaf/core.py
python tests/integration/big_leaf/checkpoint.py
python tests/integration/big_leaf/forward_patch.py
python tests/integration/big_leaf/regressions.py
python tests/integration/big_leaf/lora.py
```

分布式与全模型测试：

```bash
torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/distopt.py --runtime-wrapper
torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/distopt_checkpoint.py
torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/lora_distopt.py
TP_SIZE=2 EP_SIZE=1 torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/tp_ep.py
TP_SIZE=1 EP_SIZE=2 torchrun --standalone --nproc-per-node=2 tests/integration/big_leaf/tp_ep.py

python tests/integration/big_leaf/full_model.py --bigleaf 0 --out /tmp/big_leaf_stock.json
python tests/integration/big_leaf/full_model.py --bigleaf 1 --out /tmp/big_leaf_big.json
python tests/integration/big_leaf/full_model.py --compare /tmp/big_leaf_stock.json /tmp/big_leaf_big.json
```

这些脚本使用真实加速卡，不会被 `pytest tests/unit` 收集；性能 benchmark 不属于核心
正确性套件。

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

big_leaf 生产验收使用独立 Python CLI，不在仓库根放置 `run.sh`：

```bash
python tests/e2e/big_leaf_production.py --mode smoke --production-script-dir /path/to/production
python tests/e2e/big_leaf_production.py --mode lifecycle --production-script-dir /path/to/production
python tests/e2e/big_leaf_production.py --mode benchmark --steps 200 --rounds 3 --warmup 20 \
  --production-script-dir /path/to/production
```

`--production-script-dir` 指向含 `run.sh + train_megatron.sh` 的目标生产目录，也可用
`TRAIN_SCRIPT_DIR` 设置。首次生成合成资产需 `--tokenizer`，完整缓存或
`--model + --dataset` 不需要 tokenizer。

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
