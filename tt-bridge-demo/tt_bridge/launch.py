"""训练命令 launcher：先激活 TT Bridge，再以原参数跑目标脚本。

用法（替换原命令的 python 前缀即可）：
    python -m tt_bridge.launch train.py --args ...
    python -m tt_bridge.launch "$(which swift)" sft --model ...   # swift CLI

原理：console script（如 swift）本质是 python 文件，runpy 等价执行；
激活先于目标脚本的任何 import，框架 from-import 快照到的天然是门面符号。
"""
import runpy
import sys

from tt_bridge.entry import activate


def main():
    if len(sys.argv) < 2:
        print("usage: python -m tt_bridge.launch <script.py> [args...]",
              file=sys.stderr)
        return 2
    script = sys.argv[1]
    activate()                      # 先激活（此时框架尚未 import）
    sys.argv = sys.argv[1:]         # 目标脚本看到原始形态 argv
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
