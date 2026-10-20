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

from ttorch.witness import provenance
from ttorch.platform import manager
from ttorch.bridge.binding_registry import Decision, Entry

__all__ = ["TTCompatError", "activate", "coverage_report", "verify_sweep", "trace"]


class TTCompatError(RuntimeError):
    pass


# 绑定完成那一刻的对象快照 {(mod_name, attr): obj}——运行期篡改检测的基准
# （verify_sweep 只在激活末尾跑一次；import 触发的符号替换可能在激活后才
# 覆盖门面，witness.assert_bindings_intact 据此随时复检）。
_POST_BIND = {}


def post_bind_snapshot():
    """绑定完成时刻的对象快照（只读副本）。篡改检测的基准，供 witness 用。"""
    return dict(_POST_BIND)


# 绑定去向记录统一进 witness.provenance（layer="bridge"），与 provider 的
# 门禁/选择/调用事件共一份事件流——排障时一份报告同时覆盖"符号被替换成什么"
# （bridge）与"模块加速选了谁"（provider）。trace() 保留为兼容视图：把统一
# 信封适配回历史字段（symbol/decision/target/swept_modules），供既有打印用。


def trace():
    """绑定去向记录的只读快照（tuple）。每项含 symbol/decision/target/
    intent/swept_modules。

    存储已统一进 witness.provenance（layer="bridge"）；本函数把统一信封适配回
    历史字段，供既有消费方用。新代码请直接读 ttorch.witness.provenance。
    """
    return tuple(
        {"symbol": r["subject"], "decision": r["outcome"],
         "target": r["evidence"].get("target"), "intent": r["evidence"].get("intent"),
         "swept_modules": r["evidence"].get("swept_modules")}
        for r in provenance.records(provenance.LAYER_BRIDGE))


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


def _fallback_eager(dotted, original):
    """FALLBACK = 退回 eager 等价路径（收窄后的显式语义，2026-10-15 修正）。

    仅适配"加速壳"类符号：首参是可调用 body，退回即 eager 执行它
    （如 graph_capture(body) -> body()）。等价路径**不是**调 original——
    原生壳本身往往就是当前要避开的平台专属路径（mock 的 graph_capture 会
    _require CUDA driver）。

    旧形态对首参非 callable 的符号会静默错乱（把第一个参数当函数调）；
    现在明确报错——该符号应改用 remap/exception 决策。original 参数保留
    在签名里供未来扩展（如转发到指定等价路径），当前不转发。
    """
    def impl(*a, **k):
        if not (a and callable(a[0])):
            raise TTCompatError(
                f"{dotted}: FALLBACK 仅适配首参为 callable 的加速壳符号"
                f"（如 graph_capture(body)），收到首参 {type(a[0]).__name__ if a else None}；"
                f"该符号请改用 remap/exception 决策")
        provenance.record(provenance.LAYER_BRIDGE, "bind", dotted, "fallback-exec",
                          target="eager", intent="运行时退回 eager 执行", swept_modules=0)
        return a[0](*a[1:], **k)
    impl.__name__ = dotted.rsplit(".", 1)[-1]
    return impl


def _exception_impl(dotted, intent):
    def impl(*a, **k):
        raise TTCompatError(f"{dotted}: {intent}")
    return impl


def _impl_for(e: Entry, original):
    if e.decision is Decision.REMAP:
        if e.impl_factory is not None:
            # bind 期解析（平台已激活）：类型位符号等需要"绑定对象"而非
            # 转发函数的条目在此物化（见 Entry.impl_factory）
            return e.impl_factory(e, original)
        return e.impl
    if e.decision is Decision.FALLBACK:
        return _fallback_eager(e.dotted, original)
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
        original = getattr(module, attr)
        n_swept = _rebind_and_sweep(module, attr, _impl_for(e, original), original, sweep_packages)
        _POST_BIND[(e.bind_module or e.symbol.__module__, attr)] = getattr(module, attr)
        managed.add(e.dotted)
        provenance.record(provenance.LAYER_BRIDGE, "bind", e.dotted, e.decision.value,
                          target=e.target, intent=e.intent, swept_modules=n_swept)

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
                _POST_BIND[(mod_name, name)] = getattr(module, name)
                denied += 1
                provenance.record(provenance.LAYER_BRIDGE, "bind", f"{mod_name}.{name}",
                                  "default-deny", target=None,
                                  intent="未盘点符号，白名单姿态", swept_modules=n_swept)

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
