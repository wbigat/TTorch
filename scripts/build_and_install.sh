#!/bin/bash
# 打包并安装 ttorch wheel（开发期反复重装用）。
#
# 为什么要这个脚本而不是直接 pip install：踩过两个坑，都表现为
# "代码改了但装进去的还是旧的"，且不报错——
#
#   1. `build/lib/` 残留：setuptools 从它打包而非源码树，改动不生效；
#   2. **editable 安装残留**：`pip install -e .` 留下的
#      site-packages/ttorch/ 目录在 `pip uninstall` 后不会被清掉，
#      之后正常安装的包被它遮蔽。
#
# 另：必须用 wheel 安装而非 editable——`.pth` 走 data_files，
# editable 模式不处理，装了也不会自动激活。
set -eo pipefail

# 本脚本在 scripts/ 下，项目根是上一级
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="$(python -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
WHEEL_DIR="${WHEEL_DIR:-${PROJ_DIR}/dist}"

cd "${PROJ_DIR}"
echo "[build] 清理构建残留"
rm -rf build ttorch.egg-info "${WHEEL_DIR}"
pip uninstall -y ttorch -q 2>/dev/null || true
rm -rf "${SP}/ttorch" "${SP}"/ttorch-*.dist-info "${SP}/ttorch_auto_init.pth"

echo "[build] 构建 wheel -> ${WHEEL_DIR}"
python -m pip wheel . --no-deps --no-build-isolation --no-cache-dir -w "${WHEEL_DIR}" -q

echo "[build] 安装"
pip install --no-cache-dir --no-deps -q "${WHEEL_DIR}"/ttorch-*.whl

echo "[build] 校验"
python - <<'PY'
import importlib, sysconfig, os, sys
m = importlib.import_module("ttorch")
assert "site-packages" in m.__file__, f"装的不是 site-packages 版本: {m.__file__}"
for sub in ("ttorch.bridge", "ttorch.checkpoint", "ttorch.data"):
    importlib.import_module(sub)
pth = os.path.join(sysconfig.get_path("purelib"), "ttorch_auto_init.pth")
assert os.path.exists(pth), f".pth 未落位: {pth}"
print(f"  ok: {m.__file__}")
print(f"  ok: {pth}")
PY
echo "[build] 完成。自动激活需 export TT_PLUGIN_ENABLE=1"
