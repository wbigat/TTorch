"""进程级 XSwift 在场探针（+ 可选的 TT 引导）。

放进 PYTHONPATH 后，每个 Python 进程启动时把「xswift_ext 是否可导入 / 是否
已在 sys.modules / auto-init 是否劫持了 __import__」写成一行 JSON，用于证明
退役验证里训练 worker 真的没有 XSwift 运行代码——而不是只看 rank0 banner。

`TT_PLUGIN_ENABLE=1` 时额外在**训练 worker** 上激活 TT（等价于安装
`ttorch_auto_init.pth`；TT 当前未装 .pth，sitecustomize 是唯一早于框架
import 的钩子），并在进程退出时追加一条 `kind="tt_final"` 记录——补丁是否
被厂商顶掉、导出安全契约是否干净，只有在框架 import 完成之后看才算数。

必须先链式加载环境里原有的 sitecustomize，否则会把它顶掉。
"""
import atexit
import builtins
import importlib.util
import json
import os
import sys


def _chain_original():
    """加载被本文件遮蔽的原 sitecustomize（按 sys.path 顺序找下一个）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        try:
            if not entry or os.path.abspath(entry) == here:
                continue
            candidate = os.path.join(entry, "sitecustomize.py")
            if os.path.isfile(candidate):
                spec = importlib.util.spec_from_file_location(
                    "_tt_original_sitecustomize", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return candidate
        except Exception:
            continue
    return None


# 进程级幂等卫：本文件在真实环境里会被执行**两次**（2026-11-19 实测）。
# 机制是两个链式加载器递归一层——环境里另一个 sitecustomize（PYTHONPATH 中的
# IDE shim）同样做链式加载，它的做法是 `sys.modules.pop("sitecustomize")` +
# 重新 `__import__`，但只把**自己**的目录移出 sys.path，于是又命中本文件。
# 后果：每进程两条探针记录、`atexit` 注册两次 —— 证据计数翻倍（内容一致，
# 不影响任何 set/any/all 聚合判定，但"8 条记录"会被误读成 8 个 worker）。
# 标记挂在 `sys` 上而不是模块全局：重复执行时模块全局是全新的。
_GUARD = "_tt_xswift_probe_done"


def _write(record):
    out = os.environ.get("TT_XSWIFT_PROBE")
    try:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as error:
        # 不能让探针打断训练，但**必须可见**——静默失败会让「worker 里没有
        # XSwift」这个结论失去证据（已实测踩过：相对路径被子进程按自身 cwd 解析）。
        sys.stderr.write(
            f"[tt-probe] 写入失败 path={out!r} err={type(error).__name__}: {error}\n")


def _is_training_worker():
    """是否为 swift megatron 的训练 worker（与 big_leaf 验收引导同一判据）。"""
    script = os.path.abspath(sys.argv[0]).replace("\\", "/") if sys.argv else ""
    return script.endswith("/swift/cli/_megatron/sft.py")


def _tt_requested():
    return os.environ.get("TT_PLUGIN_ENABLE", "0").upper() in {
        "1", "TRUE", "YES", "Y", "ON"}


def _dump_tt_final():
    """进程退出时的 TT 终局取证。

    为什么必须等到退出：厂商插件走 lazy hook，`import swift` 才会整体替换
    符号；在 sitecustomize 阶段看 `patch_status()` 只会得到一片 pending，
    证明不了任何事。
    """
    record = {"kind": "tt_final", "pid": os.getpid(),
              "rank": os.environ.get("RANK")}
    try:
        import ttorch
        from ttorch.bridge import framework_patch as fw
        from ttorch.bridge import vendor_retire
        record["tt_active"] = ttorch.is_active()
        record["patch_status"] = dict(fw.patch_status())
        record["overwritten_patches"] = fw.overwritten_patches()
        record["retirement_report"] = vendor_retire.retirement_report()
        n_tampered, rows = ttorch.assert_bindings_intact()
        record["bindings_tampered"] = n_tampered
        record["bindings_tampered_rows"] = [name for ok, name in rows if not ok]
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"[:300]
    _write(record)


def _activate_tt(record):
    """按开关在训练 worker 上走 TT 的**生产激活路径**（`ttorch.utils._auto_init`）。

    刻意复用生产钩子而不是自己调 `ttorch.activate()`：进程分档（训练主体/派生/
    工具进程豁免）、fork 钩子、以及"导出安全契约失败 → 转 SystemExit 拒绝启动"
    都在 `_auto_init` 里。自己重写一遍就等于测了个假的——实测踩过：探针自调
    `activate()` 时 `ExportUnsafeError` 被 `except Exception` 吞掉，worker 照跑。

    TT 当前未安装 `.pth`，sitecustomize 是唯一早于框架 import 的钩子，这里显式
    import 一次等价于 `ttorch_auto_init.pth` 的那一行。只给训练 worker 激活：
    数据解析等辅助进程不训练也不保存。
    """
    record["tt_requested"] = _tt_requested()
    record["tt_training_worker"] = _is_training_worker()
    record["tt_activated"] = False
    if not (record["tt_requested"] and record["tt_training_worker"]):
        return
    try:
        import ttorch
        import ttorch.utils._auto_init  # noqa: F401  触发生产激活逻辑
        record["tt_activated"] = ttorch.is_active()
        if record["tt_activated"]:
            atexit.register(_dump_tt_final)
    except SystemExit:
        # 契约拒绝启动是故意的。先把证据落盘再放它走，否则现场丢失。
        record["tt_refused"] = True
        _write(record)
        raise
    except Exception as error:
        record["tt_error"] = f"{type(error).__name__}: {error}"[:300]
        sys.stderr.write(f"[tt-probe] TT 激活失败: {record['tt_error']}\n")


def _probe():
    out = os.environ.get("TT_XSWIFT_PROBE")
    if not out:
        return
    try:
        spec = importlib.util.find_spec("xswift_ext")
        importable = spec is not None
        origin = getattr(spec, "origin", None) if spec else None
    except Exception as error:
        importable = False
        origin = f"find_spec_error:{type(error).__name__}"
    record = {
        "pid": os.getpid(),
        "argv0": os.path.basename(sys.argv[0]) if sys.argv else "",
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "env_XSWIFT_PLUGIN_ENABLE": os.environ.get("XSWIFT_PLUGIN_ENABLE"),
        "env_XME_PLUGIN_ENABLE": os.environ.get("XME_PLUGIN_ENABLE"),
        "env_XTE_GROUPED_GEMM_LARGE_WEIGHT": os.environ.get(
            "XTE_GROUPED_GEMM_LARGE_WEIGHT"),
        # 包是否还在环境里（退役第二阶段应为 False）
        "xswift_importable": importable,
        "xswift_origin": origin,
        # auto_init 是否已劫持 import（它替换 builtins.__import__）
        "import_hooked_by_xswift": "xswift_ext" in getattr(
            builtins.__import__, "__module__", "") or "xswift" in repr(
            builtins.__import__).lower(),
        "xswift_in_sys_modules_at_start": [
            name for name in sys.modules if name.startswith("xswift_ext")],
        "dummy_lib_modules": [
            name for name, mod in list(sys.modules.items())
            if getattr(mod, "__file__", None)
            and "/klx_dummy_lib/" in str(mod.__file__)],
    }
    # XSwift 画像先采集、后激活 TT：保证 `*_at_start` 字段仍是"进程启动时"的
    # 语义，不被 TT 自己的 import 动作污染。
    _activate_tt(record)
    _write(record)


if not getattr(sys, _GUARD, False):
    setattr(sys, _GUARD, True)
    _chain_original()
    _probe()
