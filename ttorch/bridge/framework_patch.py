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
    """
    dotted: str
    module: str
    attr: str
    wrapper: Callable
    intent: str = ""
    applies_when: Callable = None
    upstream_ref: str = ""     # 对应的上游 issue/PR，便于升级时判断能否删


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


class _PatchFinder:
    """import 钩子：目标模块加载完成的瞬间打补丁。

    与 fn_binding._PostImportFinder 同构但独立——两条通道的注册表与语义不同，
    共用一个 finder 会让"性能绑定"与"功能补丁"的失效诊断混在一起。
    """

    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        if any(p.module == fullname for p in _PATCHES):
            _PENDING.add(fullname)
        return None


_PENDING = set()


def _drain_pending():
    """把已完成 import 的待处理模块逐个打上补丁。"""
    for name in list(_PENDING):
        if name in sys.modules:
            for p in _PATCHES:
                if p.module == name:
                    _apply_one(p)
            _PENDING.discard(name)


def install(verbose=True):
    """安装功能补丁（幂等）。已在场的模块立即打，未到场的等 import。

    返回 patch_status() 快照。摘要打印让"补丁没生效"立刻可见——静默失效
    是最危险的状态（训练照跑，但缺陷依旧）。
    """
    global _INSTALLED
    if not _INSTALLED:
        _INSTALLED = True
        sys.meta_path.insert(0, _PatchFinder())
    for p in _PATCHES:
        _apply_one(p)
    _drain_pending()
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
    _PENDING.clear()
