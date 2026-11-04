"""收益口径对齐训练态的回归（A1 步骤 5，2026-11-03）。

背景：measure_throughput_us 原先只跑 forward，建图与反向开销都不计入。
后果不只是数值不准——省掉建图的实现（含断图实现）speedup 反而更高，收益
门禁会**偏好错误实现**。实测教训：整网曾因 forward-only 口径把 2.5x 误判，
真实约 1x。
"""
import torch

from ttorch.witness.verify import measure_throughput_us, verify_impl


def _native(x):
    return x.square()


def _probe():
    return (torch.randn(64, 64),), {}


class _HeavyBackward(torch.autograd.Function):
    """前向与 native 等价、反向故意很重——用于区分两种收益口径。"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, go):
        (x,) = ctx.saved_tensors
        acc = go * 2 * x
        for _ in range(30):            # 反向额外负载
            acc = acc + go * x * 1e-6
        return acc


def test_measure_counts_backward_when_bwd_true():
    """bwd=True 时反向必须被计入：同一实现的 fwd+bwd 耗时应显著高于仅 fwd。"""
    x = torch.randn(64, 64, requires_grad=True)
    fwd_only = measure_throughput_us(_HeavyBackward.apply, (x,), {},
                                     warmup=5, reps=50, bwd=False)
    fwd_bwd = measure_throughput_us(_HeavyBackward.apply, (x,), {},
                                    warmup=5, reps=50, bwd=True)
    assert fwd_bwd > fwd_only * 1.5, \
        f"反向未被计入：fwd_only={fwd_only:.1f}us fwd_bwd={fwd_bwd:.1f}us"


def test_measure_does_not_accumulate_grad():
    """用 autograd.grad 而非 .backward()：不得写入 .grad（跨迭代累加会污染）。"""
    x = torch.randn(32, 32, requires_grad=True)
    measure_throughput_us(_native, (x,), {}, warmup=2, reps=10, bwd=True)
    assert x.grad is None, "性能测量写入了 .grad，会跨迭代累加并污染训练状态"


def test_perf_scope_recorded_as_fwd_bwd():
    """native 需反向时，evidence 必须标注 fwd+bwd 口径（否则无法与仅 fwd 对比）。"""
    ok, reason, ev = verify_impl(_HeavyBackward.apply, _native, _probe,
                                 enforce_benefit=False)
    assert ok is True, reason
    assert ev["perf_scope"] == "fwd+bwd", ev


def test_perf_scope_fwd_when_native_has_no_graph():
    """native 不可微：两侧都只测前向，口径标注为 fwd。"""
    def nd(x):
        return torch.argmax(x, dim=-1)

    ok, reason, ev = verify_impl(nd, nd, _probe, enforce_benefit=False)
    assert ok is True, reason
    assert ev["perf_scope"] == "fwd", ev


def test_heavy_backward_penalized_in_speedup():
    """反向重的实现在新口径下 speedup 必须明显低于仅 fwd 口径。

    这是"门禁取向偏差"的直接回归：若 speedup 仍按仅 fwd 计，反向开销不计入，
    反向重/断图的实现 speedup 虚高，更容易通过 min_speedup。

    断言用**两种口径的比值**而非绝对阈值——实测 autograd.Function 的 Python
    调度开销本身就让仅 fwd 口径的 speedup 降到 0.38，用 `<1.0` 两种口径都满足，
    没有区分力（本测试初版即因此 red-green 不成立）。
    """
    _, _, ev = verify_impl(_HeavyBackward.apply, _native, _probe,
                           enforce_benefit=False)
    assert ev["perf_scope"] == "fwd+bwd", ev
    args, kw = _probe()
    fwd_only = (measure_throughput_us(_native, args, kw, warmup=5, reps=50, bwd=False)
                / measure_throughput_us(_HeavyBackward.apply, args, kw,
                                        warmup=5, reps=50, bwd=False))
    assert ev["speedup"] < fwd_only * 0.8, (
        f"反向开销未反映到 speedup：fwd+bwd={ev['speedup']} "
        f"vs 仅 fwd={fwd_only:.2f}（口径仍是仅 fwd？）")
