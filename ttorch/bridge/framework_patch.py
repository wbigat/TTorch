"""框架功能兼容补丁（bridge 层）——与 provider 的性能接管严格分开。

**为什么在 bridge 而不是 provider**：本模块的补丁修正的是框架自身实现的
**功能/正确性缺陷**（参数丢失、契约不一致），不替换任何 kernel、不追求吞吐。
设计文档 §8 的边界：provider 管"性能增量"（选 fused kernel、模块性能替换），
bridge 管"功能接入"。把功能补丁塞进 provider 会污染三级门禁的语义——门禁
以 speedup 为准入条件之一，而功能补丁的收益恒为 1x，必然被收益门禁误判。

**为什么不复用 provider.fn_binding**：那条通道的绑定目标是 FunctionDispatcher
（经 provider.manager.call 走门禁选实现）。功能补丁不需要门禁、不需要变体
选择，只需要"在原实现外面补一层正确性修正"。共用会让 provider 的 semantic
注册表里出现没有 native/impl 的空语义——此前已明确禁止"非性能生命周期适配
通过空 semantic 偷渡为算子已通过门禁"。

机制：复用 fn_binding 已验证的 import 钩子（它已覆盖 megatron/mcore_bridge/
swift 包），但走独立的注册表与 wrap 语义。

补丁清单见 patches.py；本模块只提供机制。
"""
import importlib.abc
import sys
from dataclasses import dataclass, field
from typing import Callable

from ttorch.utils._rebind import rebind
from ttorch.witness import provenance

# 替换动作的 owner 标识（留痕属性名 `_tt_origin_fwpatch_<attr>`）。
# 必经 _rebind.rebind()——它是替换动作的**唯一实现**：带 owner 留痕消灭
# 撞名、写全局快照让 assert_bindings_intact() 能覆盖本通道。自己 setattr
# 会在完整性检测里留下盲区（第三方覆盖掉补丁时 TT 无从察觉）。
#
# 类方法落点的快照 key 用 `"pkg.mod:Class"`（经 rebind 的 holder_path）——
# 只传类对象时 `type.__name__` 不含模块路径，会让 assert_bindings_intact()
# 按类名 import_module 而中断整个完整性检测（2026-11-04 实测后修正）。
_OWNER = "fwpatch"
_SWEEP_PACKAGES = ("megatron", "mcore_bridge", "swift")

__all__ = ["FunctionPatch", "register_patch", "get_patches", "install",
           "uninstall", "patch_status"]


@dataclass(frozen=True)
class FunctionPatch:
    """一条框架功能补丁的注册。

    dotted/module/attr：补丁落点（框架可见路径）。

    **attr 支持类方法**（2026-11-04）：写成 ``"ClassName.method"`` 即补到类上
    （如 ``GPTBridge._set_mlp_state``）。框架的功能缺陷大量落在类方法上，
    只支持模块级函数会把这类补丁逼成"改调用方"的大改动。
    嵌套类用点号继续下钻（``Outer.Inner.method``）。

    wrapper：(orig) -> 新实现。必须在 orig 不可用时返回 orig 本身（不接管）。
      类方法的 orig 是**未绑定函数**，wrapper 返回的也必须是未绑定函数
      （签名含 self），由 Python 描述符协议在访问时完成绑定。
    intent：补丁要修的是什么缺陷——没有 intent 半年后没人敢删。
    applies_when：() -> bool 的前置判据（如"仅 CP>1 时需要"）。返回 False
        则不接管并留痕，而不是静默跳过（失效必须可见）。
    retry_after：前置条件依赖**另一个模块的安装动作**时，给出这些模块名——
        它们 exec 完成时会重试本补丁。典型场景（2026-11-04 实测）：B1 的
        判据要求 mcore_bridge 已把上游符号包进闭包，而该动作发生在
        ``import mcore_bridge``（其 __init__ 调 apply_patch）期间——若
        rope_utils 先于它被 import，首次判据必然失败，必须靠
        ``retry_after=("mcore_bridge",)`` 在它 exec 完成后重试。
    """
    dotted: str
    module: str
    attr: str
    wrapper: Callable
    intent: str = ""
    applies_when: Callable = None
    upstream_ref: str = ""     # 对应的上游 issue/PR，便于升级时判断能否删
    retry_after: tuple = ()


_PATCHES = []
_APPLIED = {}      # dotted -> "applied" / 跳过原因
_INSTALLED = False


def register_patch(p: FunctionPatch):
    """注册一条功能补丁（幂等按 dotted 查重）。"""
    assert all(x.dotted != p.dotted for x in _PATCHES), f"重复注册补丁: {p.dotted}"
    _PATCHES.append(p)
    return p


def get_patches():
    """已注册补丁的只读副本。"""
    return tuple(_PATCHES)


def patch_status():
    """每条补丁的当前状态（applied / 跳过原因 / pending）。失效可见。"""
    out = {}
    for p in _PATCHES:
        out[p.dotted] = _APPLIED.get(p.dotted, "pending(模块未到场)")
    return out


def _resolve_holder(mod, attr):
    """把 attr 解析为 (持有者, 叶子名)。

    ``"func"``            -> (模块, "func")
    ``"Class.method"``    -> (类,   "method")
    ``"Outer.Inner.m"``   -> (内层类, "m")

    中间任一层缺失返回 (None, leaf)——调用方据此记 skipped（框架结构变了，
    不接管比猜一个落点安全）。
    """
    parts = attr.split(".")
    holder = mod
    for name in parts[:-1]:
        holder = getattr(holder, name, None)
        if holder is None:
            return None, parts[-1]
    return holder, parts[-1]


def _apply_one(p: FunctionPatch):
    """对一条补丁执行绑定。机制异常收敛为"不接管"+留痕，绝不向上抛——
    功能补丁失败应退回框架原实现（可能有缺陷但能跑），不能让 TT 搞崩训练。"""
    if p.dotted in _APPLIED and _APPLIED[p.dotted] == "applied":
        return True
    mod = sys.modules.get(p.module)
    if mod is None:
        return False
    try:
        holder, leaf = _resolve_holder(mod, p.attr)
        if holder is None:
            _APPLIED[p.dotted] = "skipped(符号不存在)"
            provenance.record(provenance.LAYER_BRIDGE, "patch", p.dotted,
                              "skipped", reason="holder_absent", intent=p.intent)
            return False
        orig = getattr(holder, leaf, None)
        if orig is None:
            _APPLIED[p.dotted] = "skipped(符号不存在)"
            provenance.record(provenance.LAYER_BRIDGE, "patch", p.dotted,
                              "skipped", reason="symbol_absent", intent=p.intent)
            return False
        if getattr(orig, "_tt_patched", False):
            _APPLIED[p.dotted] = "applied"
            return True
        if p.applies_when is not None and not p.applies_when():
            _APPLIED[p.dotted] = "skipped(前置条件不满足)"
            provenance.record(provenance.LAYER_BRIDGE, "patch", p.dotted,
                              "skipped", reason="precondition_false",
                              intent=p.intent)
            return False
        new = p.wrapper(orig)
        if new is orig:
            _APPLIED[p.dotted] = "skipped(wrapper 判定不需要接管)"
            provenance.record(provenance.LAYER_BRIDGE, "patch", p.dotted,
                              "skipped", reason="wrapper_declined",
                              intent=p.intent)
            return False
        try:
            new._tt_patched = True
            new._tt_patch_intent = p.intent
        except (AttributeError, TypeError):
            pass  # 不可写属性的可调用对象：不影响绑定本体
        # 扫尾只对**模块级函数**做（修 from-import 的陈旧引用）。类方法必须
        # 禁用：_sweep 是拿叶子名去各模块 getattr 比对的，类方法的叶子名
        # （如 "merge"）在模块命名空间里可能巧合同名，扫尾会误改无关函数。
        # 类方法也不存在 from-import 陈旧引用问题——持有者是类本身，
        # 经描述符协议访问，改类属性即全局生效。
        is_method = "." in p.attr
        # 非模块 holder 必须给出可 import 的定位路径（"pkg.mod:Class"）——
        # 否则快照 key 只有类名，assert_bindings_intact() 会按它 import_module
        # 而 ModuleNotFoundError，打断整个完整性检测（2026-11-04 实测）。
        holder_path = (f"{p.module}:{p.attr.rsplit('.', 1)[0]}"
                       if is_method else None)
        swept = rebind(holder, leaf, new, owner=_OWNER,
                       sweep_packages=() if is_method else _SWEEP_PACKAGES,
                       holder_path=holder_path)
        _APPLIED[p.dotted] = "applied"
        provenance.record(provenance.LAYER_BRIDGE, "patch", p.dotted,
                          "applied", intent=p.intent, swept=swept,
                          upstream_ref=p.upstream_ref)
        return True
    except Exception as e:
        _APPLIED[p.dotted] = f"error({type(e).__name__})"
        provenance.record(provenance.LAYER_BRIDGE, "patch", p.dotted,
                          "error", reason=f"{type(e).__name__}: {e}",
                          intent=p.intent)
        return False


class _PatchFinder(importlib.abc.MetaPathFinder):
    """import 钩子：目标模块 exec_module 完成的瞬间打补丁（后置通道）。

    与 fn_binding._PostImportFinder 同构但独立——两条通道的注册表与语义不同，
    共用一个 finder 会让"性能绑定"与"功能补丁"的失效诊断混在一起。

    **实现要点（2026-11-04 修复）**：必须**包装真实 loader 的 exec_module**
    并在模块体执行完后立即绑定——旧实现只在 find_spec 里把模块名记入
    _PENDING，exec 完成后没有任何地方 drain，"先 install 后 import"的路径
    （正是生产 activate 的真实时序）补丁永远打不上。回归
    tests/unit/test_framework_patch_hook.py。
    """

    def __init__(self):
        # 重入防护：本 finder 委托遍历 meta_path 时，若另一个 TT 后置 finder
        # （fn_binding._PostImportFinder）盯上同一模块，对方会再委托回来——
        # 各自只跳过 `finder is self`，形成无限互递归（2026-11-05 实测
        # RecursionError：activate 的 rope_utils 绑定 + B1 补丁同模块）。
        # 被委托而重入时返回 None 让位，外层循环继续找到真实 loader；
        # 两通道的 exec 包装自然嵌套，两个钩子都会执行。回归
        # tests/unit/test_finder_mutual_guard.py。
        self._resolving = set()

    def find_spec(self, fullname, path=None, target=None):
        targets = {p.module for p in _PATCHES}
        triggers = {m for p in _PATCHES for m in p.retry_after}
        if fullname not in targets and fullname not in triggers:
            return None
        if fullname in self._resolving:
            return None
        self._resolving.add(fullname)
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    break
            else:
                return None
        finally:
            self._resolving.discard(fullname)
        orig_exec = spec.loader.exec_module

        def _exec(module):
            orig_exec(module)
            # 模块体执行完立即绑定：from-import 方拿到的就是新实现。
            # 除目标模块自身外，retry_after 声明的依赖模块 exec 完成也重试——
            # 前置条件依赖其他模块的安装动作时，首次判据必然失败（见字段说明）。
            for p in _PATCHES:
                if _APPLIED.get(p.dotted) == "applied":
                    continue
                if p.module == module.__name__ or module.__name__ in p.retry_after:
                    _apply_one(p)

        spec.loader.exec_module = _exec
        return spec


def install(verbose=True):
    """安装功能补丁（幂等）。已在场的模块立即打，未到场的等 import 钩子。

    返回 patch_status() 快照。摘要打印让"补丁没生效"立刻可见——静默失效
    是最危险的状态（训练照跑，但缺陷依旧）。
    """
    global _INSTALLED
    if not _INSTALLED:
        _INSTALLED = True
        sys.meta_path.insert(0, _PatchFinder())
    for p in _PATCHES:
        _apply_one(p)
    status = patch_status()
    if verbose and status:
        applied = sum(1 for v in status.values() if v == "applied")
        print(f"[tt] framework_patch: {applied}/{len(status)} applied"
              + ("" if applied == len(status) else f"  详情: {status}"))
    return status


def uninstall():
    """移除 finder 并清状态（测试隔离用；不回滚已打的补丁——
    回滚需要持有 orig，而框架可能已把新实现 from-import 到别处）。"""
    global _INSTALLED
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not isinstance(f, _PatchFinder)]
    _INSTALLED = False
