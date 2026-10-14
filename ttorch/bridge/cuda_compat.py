"""cuda-compat：Bridge 两块之一，CUDA 调用点的逐符号绑定机制。

policy 载体（v5）：code-first 注册表（binding_registry.py + bindings.py），
真实对象引用——字符串引用的拼写/悬空问题在 import 期即暴露，不再需要事后校验器。
YAML 降级为生成物（binding_registry.export_entries 快照，供评审 diff 与 provenance）。

机制不变：
- activate()：导入期一次性绑定；源头 setattr + sys.modules 扫尾
  （修 from-import 陈旧引用，吸收 KLX symbol_rewriter 机制），并留 _origin_ 痕
- 默认拒绝：盘点面（__all__）内未注册的符号一律绑 exception，白名单姿态
- trace()：每个符号的决策与去向的只读快照（Bridge 侧 provenance）

决策三类（没有"交给 Provider"）：remap / fallback / exception。
"""
import importlib
import sys

from ttorch.platform import manager
from ttorch.bridge.binding_registry import Decision, Entry

__all__ = ["TTCompatError", "activate", "coverage_report", "verify_sweep", "trace"]


class TTCompatError(RuntimeError):
    pass


# 绑定去向记录（Bridge 侧 provenance）：append-only 日志，运行期会增长
# （fallback 每次执行都会追加）。匿名为 _TRACE——暴露成 public 可变 list
# 会被外部 append/clear 破坏（同 binding_registry 的 _REGISTRY 考量），
# 外部经 trace() 取只读快照。
# 不搞 class：它只是无规则的追加日志（无注册/查重逻辑），一个 list +
# 只读 getter 足够；BindingRegistry 那种有规则的才值得封装成类。
_TRACE = []


def trace():
    """绑定去向记录的只读快照（tuple）。每项含 symbol/decision/target/
    intent/swept_modules，可导出（Bridge 侧 provenance）。"""
    return tuple(_TRACE)


def _rebind_and_sweep(source_module, attr, new_impl, original, sweep_packages=None):
    """源头 setattr 重绑 + 扫尾修 from-import 陈旧引用（按对象身份匹配）。

    命名即行为：rebind = 源头 setattr 重绑符号；sweep = 遍历框架包修陈旧
    引用。返回扫尾修复的模块数（n_swept）。

    扫尾范围由 sweep_packages 白名单限定（模块名前缀）：真实环境上
    全量扫 sys.modules 会把 torch 内部/第三方库对 torch.cuda.X 的引用
    一并改写，越界风险不可控——框架包（如 swift / framework_real）才是
    from-import 陈旧引用的产生地。None 表示不限（仅 mock demo 用）。
    """
    setattr(source_module, f"_origin_{attr}", original)
    setattr(source_module, attr, new_impl)
    n_swept = 0
    for m in sys.modules.copy().values():
        if m is source_module:
            continue
        mod_name = getattr(m, "__name__", "")
        if sweep_packages is not None and not any(
                mod_name == p or mod_name.startswith(p + ".") for p in sweep_packages):
            continue
        try:
            if id(getattr(m, attr, None)) == id(original):
                setattr(m, f"_origin_{attr}", original)
                setattr(m, attr, new_impl)
                n_swept += 1
        except Exception:
            continue
    return n_swept


def _fallback_eager(dotted):
    def impl(fn, *a, **k):
        _TRACE.append(dict(symbol=dotted, decision="fallback-exec",
                           target="eager", intent="运行时退回 eager 执行", swept_modules=0))
        return fn(*a, **k)
    return impl


def _exception_impl(dotted, intent):
    def impl(*a, **k):
        raise TTCompatError(f"{dotted}: {intent}")
    return impl


def _impl_for(e: Entry):
    if e.decision is Decision.REMAP:
        return e.impl
    if e.decision is Decision.FALLBACK:
        return _fallback_eager(e.dotted)
    return _exception_impl(e.dotted, f"当前平台无等价实现且不可禁用。intent: {e.intent}")


def activate(entries, default_deny=True, sweep_packages=None):
    """导入期一次性绑定。绑定后不可变，热路径零判断。

    sweep_packages：扫尾白名单（模块名前缀）。真实环境必须显式给出
    （如 ["swift"] 或 ["framework_real"]），只修框架包的 from-import
    陈旧引用；mock demo 传 None 保持全量扫尾以演示机制本身。

    default_deny：对盘点面（源头模块 __all__）内未注册符号绑 exception。
    真实 torch.cuda 的 __all__ 含 torch 自身内部所需符号，真实环境
    应传 False——白名单管的是框架调用面，不接管 torch 自身内部。
    """
    managed = set()
    for e in entries:
        module = importlib.import_module(e.bind_module or e.symbol.__module__)
        attr = e.symbol.__name__
        n_swept = _rebind_and_sweep(module, attr, _impl_for(e), getattr(module, attr), sweep_packages)
        managed.add(e.dotted)
        _TRACE.append(dict(symbol=e.dotted, decision=e.decision.value,
                           target=e.target, intent=e.intent, swept_modules=n_swept))

    # 默认拒绝：盘点面内未注册的符号，绑 exception，白名单姿态
    denied = 0
    if default_deny:
        for mod_name in {e.bind_module or e.symbol.__module__ for e in entries}:
            module = importlib.import_module(mod_name)
            for name in getattr(module, "__all__", []):
                if f"{mod_name}.{name}" in managed:
                    continue
                original = getattr(module, name)
                n_swept = _rebind_and_sweep(module, name, _exception_impl(
                    f"{mod_name}.{name}", "未纳入 policy（默认拒绝）：请先盘点、归类、补准入测试"),
                    original, sweep_packages)
                denied += 1
                _TRACE.append(dict(symbol=f"{mod_name}.{name}",
                                   decision="default-deny", target=None,
                                   intent="未盘点符号，白名单姿态", swept_modules=n_swept))

    print(f"[cuda_compat] activated: {len(entries)} policy symbols + {denied} default-deny")


def coverage_report(entries, module_names):
    """盘点对账：模块 __all__ 中哪些符号有 policy、哪些被默认拒绝。"""
    managed = {e.dotted for e in entries}
    rows = []
    for mod_name in module_names:
        module = importlib.import_module(mod_name)
        for name in getattr(module, "__all__", []):
            dotted = f"{mod_name}.{name}"
            rows.append((dotted, "managed" if dotted in managed else "default-deny"))
    return rows


def verify_sweep(entries, packages):
    """扫尾验收：检查框架包里是否还有指向绑定前实现的陈旧引用。

    `from torch.cuda import synchronize` 会在框架模块里留下独立引用，
    源头 setattr 改不动它——这正是 _rebind_and_sweep 扫尾（吸收 KLX
    symbol_rewriter 机制）要解决的。本函数是扫尾的验收，与
    _rebind_and_sweep 同属绑定机制：按对象身份比对，框架模块里的同名
    属性必须 is 门面实现。

    返回陈旧引用列表（空 = 扫尾干净）。激活编排据此 fail fast——
    扫尾声称覆盖就必须验证到位，否则"接管"只是说法。
    """
    bound = {e.symbol.__name__: getattr(
        sys.modules[e.bind_module], e.symbol.__name__) for e in entries}
    stale = []
    for mod_name, mod in list(sys.modules.items()):
        if not any(mod_name == p or mod_name.startswith(p + ".") for p in packages):
            continue
        for attr, impl in bound.items():
            got = getattr(mod, attr, None)
            # 只认"确实持有该符号的旧实现"：同名但无关的对象（如框架自己的
            # 同名函数）不在此列，判据是存在 _origin_ 留痕且当前值不是门面
            if got is not None and got is not impl and hasattr(mod, f"_origin_{attr}"):
                stale.append(f"{mod_name}.{attr}")
    return stale
