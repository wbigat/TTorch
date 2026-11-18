"""仓库内验收的 auto-init 引导，等价于安装 ttorch_auto_init.pth。

仅在本目录被显式放入 PYTHONPATH 时生效，是否激活仍由 TT_PLUGIN_ENABLE 门控。
"""
import importlib.util
import os
import sys


script = os.path.abspath(sys.argv[0]).replace("\\", "/")
is_training_worker = script.endswith("/swift/cli/_megatron/sft.py")
# 数据解析、HF 导出等派生工具进程不重复激活；父 CLI 会显式 activate。
if is_training_worker and importlib.util.find_spec("torch") is not None:
    import ttorch
    import ttorch.utils._auto_init  # noqa: F401
    if not ttorch.is_active():
        raise SystemExit("big_leaf 验收要求训练 worker 的 TT 激活成功，禁止降级混跑")
