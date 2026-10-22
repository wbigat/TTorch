"""witness/diagnostics：带副作用的按需诊断（用户知情主动调用）。

与 conformance 的分工（2026-10-22 确立）：
- conformance：激活期准入门禁——无副作用、fail fast、失败即平台不可用；
- 本模块：**运行期按需诊断**——验证项带不可逆副作用（如清峰值统计），
  必须用户知情主动调用，故不放激活期强制套件。

激活期验证不破坏被观测对象的状态，这是 layering 纪律。

依赖方向：只用 `torch.cuda.*` 门面符号（激活后已是 TT 接管的门面），不依赖
platform.manager——遵守 witness → 仅 platform 门面的单向依赖。

（历史呼应：witness 2026-10-15 由 ttorch/diagnostics.py 收拢而成，本文件重新
独立出来专放"带副作用的显式诊断"，与无副作用的 verify/provenance 区分。）
"""
from ttorch.witness import provenance

__all__ = ["mem_reset_semantics"]


def mem_reset_semantics():
    """验证 max_memory_allocated 的峰值窗口与 reset_peak_memory_stats 语义
    （D7 第三口径：峰值统计窗口与 reset 语义）。

    **副作用警告**：会调 reset_peak_memory_stats() **清除峰值统计**——请在
    没有需要保留的峰值数据时调用（如训练开始前）。这正是它不放激活期
    强制门禁的原因。

    判据（有标杆、自洽）：分配已知量抬高峰值 → reset → 峰值应回落到当前
    allocated 附近（torch 语义：reset 后峰值重置为当前值）。

    返回 {"ok": bool|None, "detail": str, ...}；ok=None 表示平台未实现跳过。
    结果同时记 provenance（LAYER_PLATFORM, event=caliber）。
    """
    import torch
    tc = torch.cuda  # 激活后这些符号已是 TT 接管的门面
    mb = 1024**2
    try:
        x = torch.zeros(64 * mb, dtype=torch.uint8, device="cuda")
        peak_with = tc.max_memory_allocated()        # 含刚分配的 64M
        cur = tc.memory_allocated()
        tc.reset_peak_memory_stats()
        peak_after = tc.max_memory_allocated()       # 应回落到 ≈ 当前 allocated
        del x
        tc.empty_cache()
    except Exception as e:
        return {"ok": None,
                "detail": f"平台未实现/不支持该诊断 {type(e).__name__}: {e}"}
    ok = peak_after < peak_with and peak_after <= cur + 1 * mb
    provenance.record(provenance.LAYER_PLATFORM, "caliber", "mem_reset_semantics",
                      "pass" if ok else "fail",
                      peak_before=int(peak_with), peak_after_reset=int(peak_after),
                      allocated_at_reset=int(cur))
    return {"ok": ok,
            "detail": (f"峰值 {peak_with // mb}M →reset→ {peak_after // mb}M "
                       f"（reset 时 allocated={cur // mb}M，应回落≈当前值）"),
            "peak_before": int(peak_with), "peak_after_reset": int(peak_after)}
