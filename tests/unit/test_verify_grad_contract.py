"""verify_impl 的梯度契约门禁回归（A1，复核 20261028 §4.1 的最小反例固化）。

背景：旧实现把 `out.requires_grad` 写在守卫位，候选断图即跳过反向验证，
门禁给 pass 且 evidence 记 max_grad_err=None。厂商推理 kernel（无 autograd）
会被放行，上线后梯度图在该模块被切断。本文件把当时隔离进程的反例入库。
"""
import torch

from ttorch.witness.verify import verify_impl


def _probe():
    return (torch.randn(8, 16),), {}


def _native(x):
    return x.square()


def test_rejects_detached_candidate():
    """候选前向数值正确但 detach：必须因梯度契约被拒（旧实现给 pass）。"""
    ok, reason, ev = verify_impl(lambda x: x.square().detach(), _native,
                                 _probe, enforce_benefit=False)
    assert ok is False, f"detached 候选被放行: {reason}"
    assert reason.startswith("grad_contract:"), reason
    assert ev["grad_contract"] == "candidate_detached"
    assert ev["bwd_checked"] is False


def test_rejects_candidate_missing_one_grad():
    """native 对两个输入都回梯度、候选只回其中一个：按 grad_missing 拒绝。

    旧实现在 `a2.grad - a1.grad` 处抛 TypeError 被兜住，reason 记成
    precision(bwd) 反向对账异常——拒绝对了但归因错（契约缺失 != 精度超差）。
    """
    class _OnlyFirstGrad(torch.autograd.Function):
        """arg0 梯度**正确**（隔离变量：只缺 arg1 的梯度，不混入数值误差）。"""

        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return a * b.sum()

        @staticmethod
        def backward(ctx, go):
            a, b = ctx.saved_tensors
            return go * b.sum(), None  # d/da 正确；第二个输入不回梯度

    def probe():
        return (torch.randn(4, 4), torch.randn(4, 4)), {}

    def native(a, b):
        return a * b.sum()

    ok, reason, ev = verify_impl(lambda a, b: _OnlyFirstGrad.apply(a, b), native,
                                 probe, enforce_benefit=False)
    assert ok is False, f"缺梯度候选被放行: {reason}"
    assert reason.startswith("grad_missing:"), reason
    assert ev["grad_contract"] == "grad_missing"


def test_accepts_proper_autograd_function():
    """正确包了 autograd.Function 的候选：通过且 bwd_checked 为真。"""
    class _Sq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.square()

        @staticmethod
        def backward(ctx, go):
            (x,) = ctx.saved_tensors
            return go * 2 * x

    ok, reason, ev = verify_impl(_Sq.apply, _native, _probe, enforce_benefit=False)
    assert ok is True, reason
    assert ev["bwd_checked"] is True
    assert ev["grad_contract"] == "match"
    assert ev["max_grad_err"] is not None and ev["max_grad_err"] < 1e-5


def test_evidence_marks_legit_skip_when_native_has_no_graph():
    """native 本身不可微：合法跳过反向，但必须显式留痕（不是静默 None）。"""
    ok, reason, ev = verify_impl(lambda x: torch.argmax(x, dim=-1),
                                 lambda x: torch.argmax(x, dim=-1),
                                 _probe, enforce_benefit=False)
    assert ok is True, reason
    assert ev["bwd_checked"] is False
    assert ev["bwd_skip_reason"] == "native_no_graph"


def test_candidate_extra_graph_not_rejected():
    """native 无图、候选有图：候选更保守，不应被拒，只留痕。"""
    def native(x):
        with torch.no_grad():
            return x.square()

    ok, reason, ev = verify_impl(_native, native, _probe, enforce_benefit=False)
    assert ok is True, reason
    assert ev["grad_contract"] == "candidate_extra_graph"
