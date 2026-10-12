"""cuda-compat：Bridge 两块之一，CUDA 调用点的逐符号绑定机制。

policy 载体（v5）：code-first 注册表（registry.py + bindings.py），
真实对象引用——字符串引用的拼写/悬空问题在 import 期即暴露，不再需要事后校验器。
YAML 降级为生成物（registry.export_yaml 快照，供评审 diff 与 provenance）。

机制不变：
- activate()：导入期一次性绑定；源头 setattr + sys.modules 扫尾
  （修 from-import 陈旧引用，吸收 KLX symbol_rewriter 机制），并留 _origin_ 痕
- 默认拒绝：盘点面（__all__）内未注册的符号一律绑 exception，白名单姿态
- TRACE：每个符号的决策与去向，可导出（Bridge 侧 provenance）

决策三类（没有"交给 Provider"）：remap / fallback / exception。
接口三类（category，与 decision 正交）：behavior / state_query / capability。
"""
import importlib
import sys

from tt_bridge import platform_mgr
from tt_bridge.registry import Decision, Entry


class TTCompatError(RuntimeError):
    pass


TRACE = []


def _bind(source_module, attr, new_impl, original, sweep_packages=None):
    """源头 setattr + 扫尾修 from-import 陈旧引用（按对象身份匹配）。

    扫尾范围由 sweep_packages 白名单限定（模块名前缀）：真实环境上
    全量扫 sys.modules 会把 torch 内部/第三方库对 torch.cuda.X 的引用
    一并改写，越界风险不可控——框架包（如 swift / framework_real）才是
    from-import 陈旧引用的产生地。None 表示不限（仅 mock demo 用）。
    """
    setattr(source_module, f"_origin_{attr}", original)
    setattr(source_module, attr, new_impl)
    swept = 0
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
                swept += 1
        except Exception:
            continue
    return swept


def _capability_impl(key):
    """能力类符号的绑定目标：读 probe 期缓存的静态事实（可缓存，不进 runtime API）。"""
    def impl(*a, **k):
        return platform_mgr.capability(key)
    return impl


def _fallback_eager(dotted):
    def impl(fn, *a, **k):
        TRACE.append(dict(symbol=dotted, category="behavior", decision="fallback-exec",
                          target="eager", intent="运行时退回 eager 执行", swept_modules=0))
        return fn(*a, **k)
    return impl


def _exception_impl(dotted, intent):
    def impl(*a, **k):
        raise TTCompatError(f"{dotted}: {intent}")
    return impl


def _impl_for(e: Entry):
    if e.decision is Decision.REMAP:
        return _capability_impl(e.cap_key) if e.cap_key else e.impl
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
        swept = _bind(module, attr, _impl_for(e), getattr(module, attr), sweep_packages)
        managed.add(e.dotted)
        TRACE.append(dict(symbol=e.dotted, category=e.category.value,
                          decision=e.decision.value, target=e.target,
                          intent=e.intent, swept_modules=swept))

    # 默认拒绝：盘点面内未注册的符号，绑 exception，白名单姿态
    denied = 0
    if default_deny:
        for mod_name in {e.bind_module or e.symbol.__module__ for e in entries}:
            module = importlib.import_module(mod_name)
            for name in getattr(module, "__all__", []):
                if f"{mod_name}.{name}" in managed:
                    continue
                original = getattr(module, name)
                swept = _bind(module, name, _exception_impl(
                    f"{mod_name}.{name}", "未纳入 policy（默认拒绝）：请先盘点、归类、补准入测试"),
                    original, sweep_packages)
                denied += 1
                TRACE.append(dict(symbol=f"{mod_name}.{name}", category="unknown",
                                  decision="default-deny", target=None,
                                  intent="未盘点符号，白名单姿态", swept_modules=swept))

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
