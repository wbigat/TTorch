#!/bin/bash
# E2E：用 TT 顶替 xswift_ext/xmegatron_ext，跑真实的 swift-megatron SFT。
#
# 本脚本**不复制**生产训练脚本，只是给它套上 TT 的环境变量后转调——
# 生产脚本含集群路径与 API Key，不进仓库；且复制会与真身分叉。
#
# 用法：
#   bash tests/e2e/run_swift_megatron.sh                       # TT 模式
#   TT_MODE=0 bash tests/e2e/run_swift_megatron.sh             # 对照（原 xswift）
#   TRAIN_SCRIPT_DIR=/path/to/script bash tests/e2e/...        # 指定脚本位置
#
# 前置条件：
#   1. pip install dist/ttorch-*.whl（需要 .pth，editable 装不生效）
#   2. 训练脚本的 env.sh 里两个开关需可被外部覆盖，即写成：
#        export XSWIFT_PLUGIN_ENABLE=${XSWIFT_PLUGIN_ENABLE:-1}
#        export XME_PLUGIN_ENABLE=${XME_PLUGIN_ENABLE:-1}
#      （默认值不变，不传变量时行为与原来完全一致）
#
# 判定：exit 0 且日志出现 `[tt] <tt activated ...>`；对照组 loss 偏差应为
# 零均值噪声（符号近似对半），而非系统性 bias——见
# docs/TT_Swift_Cuda_API_Inventory.md §6。
set -eo pipefail

TRAIN_SCRIPT_DIR="${TRAIN_SCRIPT_DIR:-/root/dl/runtime/script}"
TT_MODE="${TT_MODE:-1}"

if [ ! -f "${TRAIN_SCRIPT_DIR}/run.sh" ]; then
    echo "[e2e] ERROR: 找不到 ${TRAIN_SCRIPT_DIR}/run.sh，请用 TRAIN_SCRIPT_DIR 指定" >&2
    exit 1
fi

if [ "${TT_MODE}" = "1" ]; then
    echo "[e2e] TT 模式：TT 激活，xswift_ext / xmegatron_ext 关闭"
    export TT_PLUGIN_ENABLE=1 XSWIFT_PLUGIN_ENABLE=0 XME_PLUGIN_ENABLE=0
else
    echo "[e2e] 对照模式：TT 关闭，沿用厂商插件（用于归因——报错先在此确认是否同样复现）"
    export TT_PLUGIN_ENABLE=0
fi

cd "${TRAIN_SCRIPT_DIR}"
bash run.sh
