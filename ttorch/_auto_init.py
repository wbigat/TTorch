"""`.pth` 自动激活钩子：解释器启动阶段按开关激活 TT。

机制与厂商栈同构（吸收 xswift_ext_auto_init.pth 的两段式）：
    ttorch_auto_init.pth  ->  import ttorch._auto_init
本模块据环境变量决定是否真的激活，**默认不激活**——装了包不等于改变
行为，这是可安全合入的前提（与 xswift 的 XSWIFT_PLUGIN_ENABLE 同口径）。

    export TT_PLUGIN_ENABLE=1        # 启用
    export TT_PLATFORM=xpu-masquerade  # 可选，默认 auto 检测

**.pth 加载顺序**（site 按文件名字母序处理，本机实测）：
    distutils-precedence < torch_xmlir < torch_xray < ttorch
    < xdeepspeed_ext < xmegatron_ext < xswift_ext
恰好满足依赖：厂商基座（torch_xmlir，拉起 torch）先就位 -> TT 接管
torch.cuda 符号 -> swift/megatron 扩展在后。命名若改动需重新校核此序。

为什么 .pth 优于"脚本里手动 import"：对 `megatron sft` 这类**第三方
CLI 入口**，我们改不了它的 main 函数，手动 import 无处安放；.pth 是
Python 官方提供的、早于一切用户代码的钩子。
"""
import os
import sys


def _enabled():
    return os.getenv("TT_PLUGIN_ENABLE", "0").upper() in {"1", "TRUE", "YES", "Y", "ON"}


def _is_tooling_process():
    """包管理/构建工具进程豁免：这些进程不训练，且其 stdout 可能被解析
    （pip 用 stdout 传数据），激活日志会污染。同 xswift 的豁免口径。"""
    argv = [a.lower() for a in sys.argv[:3]]
    argv0 = os.path.basename(argv[0]) if argv else ""
    return (argv0 in {"pip", "pip3", "setup.py", "conda"}
            or argv[:2] == ["-m", "pip"]
            or "PIP_BUILD_TRACKER" in os.environ
            or "PIP_REQ_TRACKER" in os.environ)


def _init():
    if not _enabled() or _is_tooling_process():
        return False
    try:
        import ttorch
        # 日志走 stderr：训练脚本常把 stdout 作为结果通道
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            ttorch.activate(platform=os.getenv("TT_PLATFORM", "auto"))
        return True
    except Exception as e:  # noqa: BLE001
        # .pth 里的异常会被 site 吞掉成一行警告，排障极难——显式打全栈。
        # 但不 re-raise：解释器启动阶段抛异常会让所有 python 进程不可用
        # （含 pip/conda），代价远超"TT 未激活"。激活与否由后续
        # ttorch.is_active() 判定，调用方据此 fail fast。
        import traceback
        print(f"[ttorch] 自动激活失败（进程继续，TT 未生效）: {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


_init()
