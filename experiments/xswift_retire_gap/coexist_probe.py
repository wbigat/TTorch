"""XSwift 与 TT 功能补丁通道的共存/顺序探测。

背景：生产 `script/env.sh` 当前是 `XSWIFT_PLUGIN_ENABLE=1`，而 TT 的 bridge
功能补丁（B1 THD RoPE / B5 wrap_model / B7 gather_object）与 XSwift 的
`KlxSymbolRewriter.replace()` **落在同样的三个符号上**。退役过渡期两者必然
同时在场，但两边都没有共存契约。本脚本只做只读探测，不修源码。

两个顺序各测一遍：
  xswift_first = XSwift 先 patch，再 ttorch.activate()
  tt_first     = ttorch.activate() 先 patch，再触发 XSwift

每个符号报告：当前实现的 owner、TT 留痕属性是否还在、TT 自己的
`patch_status()` 与 `assert_bindings_intact()` 是否察觉被覆盖。

用法：
    XSWIFT_PLUGIN_ENABLE=1 python coexist_probe.py --order xswift_first
    XSWIFT_PLUGIN_ENABLE=1 python coexist_probe.py --order tt_first
"""
import argparse
import json
import os
import sys


SYMBOLS = (
    ("swift.megatron.utils.megatron_lm_utils", "wrap_model"),
    ("swift.loss.embedding", "gather_object"),
    ("mcore_bridge.patcher.rope_utils", "_apply_rotary_pos_emb_thd"),
)


def _resolve_module(module_name):
    """按模块名取；取不到再按「父模块的属性」解析。

    XSwift 的 THD 落点写作 ``mcore_bridge.patcher.rope_utils``，它不是一个
    模块名——是 mcore_bridge.patcher 里 import 进来的 megatron rope_utils
    模块对象。只查 sys.modules 会误报 module_not_imported。
    """
    mod = sys.modules.get(module_name)
    if mod is not None:
        return mod
    parent_name, _, child = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    return getattr(parent, child, None) if parent is not None else None


def _describe(module_name, attr):
    mod = _resolve_module(module_name)
    if mod is None:
        return {"state": "module_not_imported"}
    fn = getattr(mod, attr, None)
    if fn is None:
        return {"state": "attr_missing"}
    origin = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    # TT 的 rebind 会留 `_tt_origin_fwpatch_<attr>`；它消失即说明被覆盖
    trace_attr = f"_tt_origin_fwpatch_{attr}"
    wrapped = getattr(fn, "__wrapped__", None)
    return {
        "state": "present",
        "impl_module": origin,
        "impl_qualname": qual,
        "tt_origin_trace_present": hasattr(mod, trace_attr),
        "has_functools_wrapped": wrapped is not None,
        "wrapped_module": getattr(wrapped, "__module__", None) if wrapped else None,
        "is_xswift_impl": bool(origin and origin.startswith("xswift_ext")),
        "closure_marker_tt_cp_group_fixed": _closure_marker(module_name),
    }


def _closure_marker(module_name):
    """B1 的落点是 mcore_bridge 闭包 cell，不在模块属性上，需单独看标记。"""
    if module_name != "mcore_bridge.patcher.rope_utils":
        return None
    patcher = sys.modules.get("mcore_bridge.patcher")
    if patcher is None:
        return "patcher_not_imported"
    found = []
    for name in dir(patcher):
        obj = getattr(patcher, name, None)
        cells = getattr(obj, "__closure__", None) or ()
        for cell in cells:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if getattr(value, "_tt_cp_group_fixed", False):
                found.append(name)
    return found or "marker_absent"


def _force_imports():
    """让三个目标模块都真实 import（XSwift 的 lazy hook 也在这里触发）。"""
    import swift.megatron.utils.megatron_lm_utils  # noqa: F401
    import swift.loss.embedding  # noqa: F401
    import mcore_bridge  # noqa: F401
    import mcore_bridge.patcher  # noqa: F401


def _tt_status():
    from ttorch.bridge import framework_patch as fw
    status = dict(fw.patch_status())
    try:
        from ttorch.bridge import cuda_compat
        from ttorch.witness import bindings
        # assert_bindings_intact 返回 (n_tampered, rows)，不 raise——旧版这里
        # 用 getattr(report, "tampered") 必然拿到 None，只好打印整个 repr。
        n_tampered, rows = bindings.assert_bindings_intact(
            cuda_compat.post_bind_snapshot())
        intact = {"raised": False, "tampered": n_tampered,
                  "tampered_rows": [name for ok, name in rows if not ok]}
    except Exception as error:  # 完整性检测报警即是我们要的信号
        intact = {"raised": True, "error": f"{type(error).__name__}: {error}"[:300]}
    return status, intact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("xswift_first", "tt_first"),
                        required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    record = {"order": args.order,
              "xswift_plugin_enable": os.environ.get("XSWIFT_PLUGIN_ENABLE"),
              "xme_plugin_enable": os.environ.get("XME_PLUGIN_ENABLE"),
              "large_weight": os.environ.get("XTE_GROUPED_GEMM_LARGE_WEIGHT")}

    import ttorch

    if args.order == "xswift_first":
        _force_imports()                       # XSwift 的 .pth hook 在此接管
        record["after_xswift"] = {f"{m}.{a}": _describe(m, a) for m, a in SYMBOLS}
        ttorch.activate()
        record["after_tt"] = {f"{m}.{a}": _describe(m, a) for m, a in SYMBOLS}
    else:
        ttorch.activate()
        _force_imports()
        record["after_tt_then_xswift"] = {f"{m}.{a}": _describe(m, a)
                                          for m, a in SYMBOLS}

    record["tt_active"] = ttorch.is_active()
    status, intact = _tt_status()
    record["tt_patch_status"] = status
    record["tt_bindings_intact"] = intact
    # B17/B18/B21 落地后的新证据：补丁是否被覆盖（运行期复核）、
    # 厂商在场探针与 /klx_dummy_lib 痕迹、三开关的导出安全核对
    from ttorch.bridge import framework_patch as fw
    from ttorch.bridge import vendor_retire
    record["tt_overwritten_patches"] = fw.overwritten_patches()
    record["tt_patch_status_install_time"] = dict(fw.patch_status(recheck=False))
    record["retirement_report"] = vendor_retire.retirement_report()
    record["xswift_initialized"] = any(
        name.startswith("xswift_ext.components") for name in sys.modules)

    print("COEXIST " + json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
