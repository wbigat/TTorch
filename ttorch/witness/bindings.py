"""bridge API 验证：绑定正确性自检（静态/激活期）。

回答"bridge 的 CUDA API 是否真的被接管"——逐条跑 admission 测试（含 G14 的
vendor_internal_construction）。这是**激活期**的取证：符号是否绑到预期门面。

（运行期的"是否真的被调用到"是另一层，见 call_trace.py。）

归位 witness 的理由（2026-10-15）：观测/验证是 TT 的独立工程资产，不是入口函数
的附属品。本模块只**读**（entries 由调用方传入），不改绑定——观测与操作分离。
"""

__all__ = ["check_bindings"]


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
