"""`.pth` 自动激活钩子：解释器启动阶段按开关激活 TT。

两段式机制：
    ttorch_auto_init.pth  ->  import ttorch.utils._auto_init
.pth 只负责触发 import；是否真激活由本模块据环境变量决定，**默认不激活**——
装了包不等于改变行为，这是可安全合入的前提。

    export TT_PLUGIN_ENABLE=1          # 启用
    export TT_PLATFORM=xpu-masquerade  # 可选，默认 auto 检测

**.pth 加载顺序**（site 按文件名字母序处理）：依赖方向要求"厂商基座先拉起
torch -> TT 接管 torch.cuda 符号 -> 训练框架扩展在后"，因此 ttorch 的 .pth
文件名必须在字母序上落在厂商基座（torch_xmlir）之后、框架扩展之前。
文件名若改动，需重新校核这个相对位置仍成立。

为什么 .pth 优于"脚本里手动 import"：对 `megatron sft` 这类**第三方 CLI
入口**，我们改不了它的 main 函数，手动 import 无处安放；.pth 是 Python
官方提供的、早于一切用户代码的钩子。
"""
import os
import sys


def _enabled():
    return os.getenv("TT_PLUGIN_ENABLE", "0").upper() in {"1", "TRUE", "YES", "Y", "ON"}


# 进程级标记：本进程已完成激活的 pid（**内部信息传递，非用户配置项**）。
# 子进程（spawn/fork）会继承环境变量，据此可把它判为派生进程（D1）。
_ACTIVATED_PID_ENV = "TT_ACTIVATED_PID"


def _on_fork_child():
    """fork 后的子进程回调（D1 + D11）：标注继承态并重置事件流。

    为什么必须用 `os.register_at_fork` 而不是在 `_init()` 里判断：
    **fork 出来的进程不会重新执行 .pth**（.pth 只在解释器启动时跑一次），
    所以 DataLoader worker 这类派生进程根本走不到 `_classify_process()`，
    继承态的处理与事件流重置都会漏掉（2026-10-21 实测确认：fork 后
    `fork_events=0`）。`after_in_child` 是唯一能在 fork 后自动触发的钩子。

    回调刻意保持简单（fork 后进程环境受限，不碰锁/不 import 重模块）。
    """
    try:
        from ttorch.witness import provenance
        provenance.fork_reset(os.getenv(_ACTIVATED_PID_ENV))
        print("[ttorch] fork 派生进程：沿用父进程激活态（事件流已按进程重置）。"
              "注意 PyTorch 禁止在 fork 子进程重新初始化 CUDA，"
              "worker 内请勿使用 CUDA。", file=sys.stderr)
    except Exception:  # noqa: BLE001  fork 回调不得抛（会让子进程直接失败）
        pass


def mark_activated():
    """激活完成仪式（**.pth 与显式 ttorch.activate() 两条路径共用**）。

    做什么：打进程标记 + 挂 fork 钩子。为什么必须在两条路径上都执行
    （2026-10-22 审视发现）：这三件绑定在 .pth 路径时，显式 `ttorch.activate()`
    的进程既不写标记也不挂 fork 钩子——其后代 spawn 子进程判不出"派生"
    （会走重档跑 conformance），fork 出的 worker 事件流也不重置。所以这不是
    .pth 的事，是"激活完成"本身的事，由 ttorch.activate() 末尾统一调用。
    """
    os.environ[_ACTIVATED_PID_ENV] = str(os.getpid())
    _register_fork_hook()


def _register_fork_hook():
    """注册 fork 子进程回调（幂等；不支持的平台静默跳过）。"""
    if getattr(_register_fork_hook, "_done", False):
        return
    _register_fork_hook._done = True
    register = getattr(os, "register_at_fork", None)
    if register is None:  # 非 POSIX（Windows）无 fork 语义，无需注册
        return
    try:
        register(after_in_child=_on_fork_child)
    except Exception:  # noqa: BLE001
        pass


def _record_scope(kind, reason, skip_conformance=None):
    """把「本进程为什么走这一档」写进 provenance（谁、为什么，可审计）。"""
    try:
        from ttorch.witness import provenance
        provenance.record(provenance.LAYER_BRIDGE, "scope", "process", kind,
                          reason=reason, skip_conformance=skip_conformance)
    except Exception:  # noqa: BLE001  观测不得影响激活主流程
        pass


def _classify_process():
    """判定本进程在收束域中的角色，返回 (kind, reason)。

    kind：
      primary   训练主体 → 重档（绑定 + conformance）
      derived   派生进程（全新解释器，如 spawn 的 worker）→ 轻档（只绑定）
      inherited 已继承激活态（fork 派生）→ 什么都不做

    判据全部是**显式信号**，不做启发式猜测（符合项目反魔法文化）：
      - RANK/LOCAL_RANK 由框架（torchrun/megatron）注入 → 训练 rank；
      - TT_ACTIVATED_PID 由 TT 自己写入，子进程继承后 pid 必然不同 → 派生。
    """
    # 判据顺序是实测校正出来的（2026-10-21）：**"是否已激活"必须排在 RANK 之前**。
    # 反例：rank 进程 fork 出来的 DataLoader worker 会继承父进程的 RANK=0，
    # 若先判 RANK 就把它当成训练主体 → 走重档跑 conformance（真实设备操作）。
    # 而它其实已继承父进程激活态，应判 inherited。
    marked = os.getenv(_ACTIVATED_PID_ENV)
    derived = bool(marked) and marked != str(os.getpid())
    if derived:
        try:
            from ttorch.platform import manager as _pm
            already = _pm.is_activated()
        except Exception:  # noqa: BLE001  判据失败按"未激活"处理，不猜
            already = False
        if already:
            return "inherited", (f"fork 派生进程（{_ACTIVATED_PID_ENV}={marked}）："
                                 f"已继承父进程的绑定与 provenance，无需再激活")
    if os.getenv("RANK") or os.getenv("LOCAL_RANK"):
        return "primary", "RANK/LOCAL_RANK 存在（框架注入，判为训练 rank 进程）"
    if derived:
        return "derived", (f"{_ACTIVATED_PID_ENV}={marked} != 本进程 "
                           f"{os.getpid()}（spawn 派生进程，无 rank 身份）")
    return "primary", "本进程首次激活（无派生标记）"


def _is_tooling_process():
    """包管理/构建工具进程豁免：这些进程不训练，且其 stdout 可能被解析
    （pip 用 stdout 传数据），激活日志会污染。"""
    argv = [a.lower() for a in sys.argv[:3]]
    argv0 = os.path.basename(argv[0]) if argv else ""
    return (argv0 in {"pip", "pip3", "setup.py", "conda"}
            or argv[:2] == ["-m", "pip"]
            or "PIP_BUILD_TRACKER" in os.environ
            or "PIP_REQ_TRACKER" in os.environ)


def _init():
    if not _enabled() or _is_tooling_process():
        return False
    kind, reason = _classify_process()

    if kind == "inherited":
        # fork 派生：状态已继承，不再做任何绑定。留证据并提示 PyTorch 的
        # fork-CUDA 限制——若框架在此类 worker 里做 GPU 预处理会踩
        # "Cannot re-initialize CUDA in forked subprocess"。
        # 事件流也必须重置：继承来的记录是父进程的，seq 会从父进程值继续递增
        # （多 worker 聚合时 seq 重复，D11）。
        from ttorch.witness import provenance
        provenance.fork_reset(os.getenv(_ACTIVATED_PID_ENV))
        _record_scope(kind, reason, skip_conformance=None)
        print(f"[ttorch] 本进程为 fork 派生（{reason}）：沿用父进程激活态。"
              f"注意 PyTorch 禁止在 fork 子进程重新初始化 CUDA，"
              f"worker 内请勿使用 CUDA。", file=sys.stderr)
        return True

    try:
        import ttorch
        # 日志走 stderr：训练脚本常把 stdout 作为结果通道
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            ttorch.activate(platform=os.getenv("TT_PLATFORM", "auto"),
                            skip_conformance=(kind == "derived"))
        # 标记与 fork 钩子由 ttorch.activate() 末尾的 mark_activated() 统一执行
        # （两条路径共用）；这里只记档位证据。
        _record_scope(kind, reason, skip_conformance=(kind == "derived"))
        return True
    except Exception as e:  # noqa: BLE001
        # .pth 里的异常会被 site 吞掉成一行警告，排障极难——显式打全栈。
        # 但不 re-raise：解释器启动阶段抛异常会让所有 python 进程不可用
        # （含 pip/conda），代价远超"TT 未激活"。激活与否由后续
        # ttorch.is_active() 判定，调用方据此 fail fast。
        # 中毒标记（D3）：本进程继续跑，但"TT 失败过且已回滚"必须可查——
        # 否则调用方无法区分"TT 接管中"与"跑在纯厂商栈上"。
        try:
            from ttorch.bridge import bootstrap
            bootstrap.mark_failed(e)
        except Exception:  # noqa: BLE001  观测不得影响失败处置主流程
            pass
        import traceback
        print(f"[ttorch] 自动激活失败（进程继续，TT 未生效）: {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


_init()
