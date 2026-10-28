"""厂商性能 patch 清场：恢复 torch_xmlir symbrewrite 的符号替换。

定位（2026-10-28 收敛）：**启动期口径开关**——`TT_RESTORE_VENDOR=perf/all`
由 `ttorch.activate()` 读取并在启动时自动执行（本文件 restore），用于
**TT vs xmlir 的性能对比**：不设=xmlir patch 在场（xmlir 性能基线）；
设=baseline 回到社区原生分解、TT provider 接管（TT 真实收益口径）。
生产上 TT 在场时无需恢复 xmlir 的任何接口（功能对接类绝不能恢复——
perf 档已排除；性能类不恢复也能跑，回退落点是 xmlir 融合）。显式 API
`tt.restore_vendor_symbols(scope)` 保留给脚本内手动控制。

背景（2026-10-26 职责重划）：torch_xmlir 经 symbrewrite 把框架符号替换成厂商
融合实现（如 transformers 的 Qwen3_5RMSNorm 类 -> RMSNormFunction 融合的 mock
类，原始类留痕在 `__origin_<名>`）。这与 TT provider 的性能职责重叠——目标
形态：**torch_xmlir 只做 torch 框架对接 + 算子提供，性能优化全归 TT**。
不清场时 baseline（bypass）本身就是融合态，provider 的整网收益测不出来
（结构性持平 ~1x）；清场后 baseline 回到未融合分解，门禁/收益口径才真实。

两档：
- perf（默认）：只恢复 TT provider 接管语义的性能符号（RMSNorm/RoPE 类，
  按符号名命中 _PERF_HINTS）——框架对接类 patch（is_torch_fx_available /
  caching_allocator_warmup 等）保留，这是职责重划后的生产形态；
- all：一键全关（含对接类）——仅测试 TT 整网独立性用，可能触发框架对接
  缺失的崩溃，勿用于生产。

时序：torch_xmlir 在目标 modeling 模块 import 完成后才 patch（其插件的
module_deps 机制）——restore 内部先 import 目标模块触发 patch 再恢复，故
activate 启动期调用即安全（此时 bridge 绑定已完成，modeling 模块被提前
拉起无碍）；手工调用只需在框架 import 后、模型构建前。幂等：已恢复的符号
跳过。派生进程（skip_conformance）跳过 restore。

依赖方向：vendor_patch → witness.provenance（留痕）；不碰 torch_xmlir 内部
实现细节，只读其公开注册表（SymbolRewriteRegister）与 __origin_ 留痕约定。
"""
import importlib
import sys

from ttorch.witness import provenance

__all__ = ["restore", "PERF_HINTS"]

# 性能类符号的识别 hints：符号名命中 TT provider 已接管的模块语义。
# 接新语义时在此登记（与 provider impls 的语义一一对应）。
PERF_HINTS = ("RMSNorm", "rotary_pos_emb", "RotaryEmbedding")


def _registered_rewrites():
    """从 torch_xmlir symbrewrite 注册表拿全部符号替换清单（未装/无注册表则空）。"""
    try:
        from torch_xmlir.symbrewrite.symbol_rewrite import SymbolRewriteRegister
        reg = SymbolRewriteRegister().reg
        return [sym for mod in reg.values() for sym in mod.rewrites]
    except Exception:
        return []


def _restore_one(dotted):
    """按 __origin_ 留痕恢复一个符号（symbrewrite set_attr 的逆操作）。

    返回 (状态, detail)：状态 restored / already / failed。
    """
    parts = dotted.split(".")
    obj = sys.modules.get(parts[0])
    if obj is None:
        return "failed", "顶层模块未 import"
    for name in parts[1:-1]:
        obj = getattr(obj, name, None)
        if obj is None:
            return "failed", f"路径断于 {name}"
    attr = parts[-1]
    origin = getattr(obj, f"__origin_{attr}", None)
    if origin is None:
        return "failed", "无 __origin_ 留痕（未 patch）"
    if getattr(obj, attr, None) is origin:
        return "already", "已是原始符号"
    setattr(obj, attr, origin)
    return "restored", "已恢复"


def restore(scope="perf"):
    """恢复 torch_xmlir 的符号 patch（幂等）。返回 (restored, skipped) 符号清单。

    scope="perf"：只恢复性能类（默认，生产形态）；scope="all"：一键全关
    （测试 TT 整网功能/性能独立性）。目标 modeling 模块是 lazy import 的
    （patch 在其 import 时才发生）——本函数先 import 目标模块触发 patch，
    再按留痕恢复，故调用    点只需在框架 import 后、模型构建前。
    """
    assert scope in ("perf", "all"), f"scope 只能是 perf/all: {scope}"
    restored, skipped = [], []
    for sym in _registered_rewrites():
        if scope == "perf" and not any(h in sym for h in PERF_HINTS):
            continue
        mod_path = sym.rpartition(".")[0]
        try:
            importlib.import_module(mod_path)  # 触发 lazy import → 厂商 patch 生效
        except Exception:
            pass  # 模块不可 import（框架版本无此模型）——恢复跳过，进 skipped
        status, detail = _restore_one(sym)
        if status == "restored":
            restored.append(sym)
        elif status == "failed":
            skipped.append(f"{sym}: {detail}")
    provenance.record(provenance.LAYER_PROVIDER, "wrap", "*",
                      "vendor_patch_restored", scope=scope,
                      restored=restored, skipped=skipped)
    return restored, skipped
