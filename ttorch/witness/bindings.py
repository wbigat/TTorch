"""bridge API 验证：绑定正确性自检（静态/激活期）。

回答"bridge 的 CUDA API 是否真的被接管"——逐条跑 admission 测试（含 G14 的
vendor_internal_construction）。这是**激活期**的取证：符号是否绑到预期门面。

（运行期的"是否真的被调用到"是另一层，见 call_trace.py。）

归位 witness 的理由（2026-10-15）：观测/验证是 TT 的独立工程资产，不是入口函数
的附属品。本模块只**读**（entries 由调用方传入），不改绑定——观测与操作分离。
"""

__all__ = ["check_bindings", "assert_bindings_intact"]


def check_bindings(entries):
    """绑定正确性自检：逐条跑 admission 测试。

    entries 由调用方传入（ttorch.active().entries），本模块不反向依赖入口层——
    保持观测模块独立，可被 Bridge 内层复用而不成环。
    返回 (n_fail, rows)，rows 为 (ok, test_name, detail)。
    """
    import torch
    ctx = {"vendor_cuda": torch.cuda}
    rows = []
    n_fail = 0
    for e in entries:
        ok, detail = e.test(ctx)
        rows.append((ok, e.test.__name__, detail))
        n_fail += 0 if ok else 1
    return n_fail, rows


def assert_bindings_intact(snapshot):
    """运行期绑定完整性校验（篡改检测）：当前值是否仍是绑定完成那一刻的对象。

    verify_sweep() 只在激活末尾跑一次；import 触发的符号替换（symbrewrite
    钩子）可能在 TT 激活**之后**才覆盖门面（并把门面存成 `__origin_X`）——
    没有本函数，TT 对此无人察觉。可在训练启动后 / step 边界随时调用。

    snapshot：`cuda_compat.post_bind_snapshot()` 的产出（调用方传入——
    witness 不反向依赖 bridge，观测与操作分离）。篡改记 provenance
    （outcome="tampered"）。返回 (n_tampered, rows)。
    """
    import importlib
    from ttorch.witness import provenance
    rows = []
    n = 0
    for (mod_name, attr), expected in snapshot.items():
        # key 支持两种形态（2026-11-04）：
        #   "pkg.mod"             -> 模块级符号
        #   "pkg.mod:Class[.Sub]" -> 类（方法）落点
        # 不支持后者时，一条类方法绑定就会让 import_module("GPTBridge") 抛
        # ModuleNotFoundError，**整个完整性检测被打断**（实测）。
        path, _, cls_chain = mod_name.partition(":")
        try:
            holder = importlib.import_module(path)
            for part in filter(None, cls_chain.split(".")):
                holder = getattr(holder, part)
        except (ImportError, AttributeError) as e:
            # 定位不到 holder 本身也是一种"完整性丢失"，按篡改计并留证据，
            # 不让异常穿透打断对其余条目的检查。
            n += 1
            provenance.record(provenance.LAYER_BRIDGE, "bind",
                              f"{mod_name}.{attr}", "tampered",
                              current=f"holder 不可定位: {type(e).__name__}")
            rows.append((False, f"{mod_name}.{attr}"))
            continue
        cur = getattr(holder, attr, None)
        ok = cur is expected
        if not ok:
            n += 1
            provenance.record(provenance.LAYER_BRIDGE, "bind", f"{mod_name}.{attr}",
                              "tampered", current=repr(cur)[:80])
        rows.append((ok, f"{mod_name}.{attr}"))
    return n, rows
