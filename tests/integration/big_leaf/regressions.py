"""独立代码审查发现的 big_leaf 生命周期与失败路径回归。"""
import os
import types

import torch
from transformer_engine.pytorch import GroupedLinear

from ttorch.bridge.big_leaf import BigLeafMegatronPlan, apply_big_leaf
from ttorch.bridge.big_leaf import core as big_leaf

G, FIN, FOUT, TOK = 4, 16, 12, 8


def make(bias=False, cls=GroupedLinear, **kwargs):
    torch.manual_seed(123)
    module = cls(G, FIN, FOUT, bias=bias, params_dtype=torch.bfloat16, **kwargs).cuda()
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.03)
    return module


def input_for(module=GroupedLinear):
    return torch.randn(G * TOK, FIN, device="cuda", dtype=torch.bfloat16), [TOK] * G


def expect_raises(fn, text):
    try:
        fn()
        raise AssertionError(f"应拒绝但未拒绝: {text}")
    except (ValueError, RuntimeError) as error:
        if text not in str(error):
            raise AssertionError(f"异常文本不匹配: 期望 {text}，实际 {error}") from error


def test_restore_idempotent_after_stock_update():
    module = make()
    plan = apply_big_leaf(module)
    plan.restore()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    for p in module.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    expected = [p.detach().clone() for p in module.parameters()]
    plan.restore()
    assert all(torch.equal(p, value) for p, value in zip(module.parameters(), expected))
    print("PASS repeated restore preserves later stock updates", flush=True)


def test_dynamic_freeze_and_late_hooks_fail_fast():
    for mode in ("switch", "patch"):
        module = make()
        plan = apply_big_leaf(module, forward_mode=mode)
        module.requires_grad_(False)
        expect_raises(lambda: module(*input_for()), "动态冻结")
        plan.restore()

        module = make()
        plan = apply_big_leaf(module, forward_mode=mode)
        module.weight1.requires_grad_(False)
        expect_raises(plan.assert_healthy, "混合冻结")
        plan.restore()

        module = make()
        plan = apply_big_leaf(module, forward_mode=mode)
        module.weight0.register_hook(lambda grad: grad)
        expect_raises(lambda: module(*input_for()), "新增原参数 hook")
        plan.restore()
    print("PASS dynamic freeze/partial freeze/late hooks", flush=True)


def test_assign_positional_child_and_existing_grad():
    model = TwoGL()
    plan = apply_big_leaf(model)
    state = model.state_dict()
    expect_raises(lambda: model.load_state_dict(state, True, True), "assign=True")
    expect_raises(lambda: model.first.load_state_dict(model.first.state_dict(), assign=True),
                  "assign=True")
    plan.assert_healthy()
    plan.restore()

    module = make()
    module.weight0.grad = torch.ones_like(module.weight0)
    ptr = module.weight0.data_ptr()
    expect_raises(lambda: apply_big_leaf(module), "已有 grad/main_grad")
    assert module.weight0.data_ptr() == ptr and not hasattr(module, "_big_leaf_plan")

    module = make()
    module.register_forward_hook(lambda *args: None)
    expect_raises(lambda: apply_big_leaf(module), "_forward_hooks")
    module = make()
    module.weight0.main_grad = torch.zeros_like(module.weight0)
    expect_raises(lambda: apply_big_leaf(module), "grad/main_grad")
    print("PASS positional assign, existing grad/DDP and forward-hook rejection", flush=True)


class TwoGL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = make()
        self.second = make()


def test_apply_install_failure_rolls_back_all_modules():
    model = TwoGL()
    expected = model.state_dict()
    original_install = big_leaf._install_forward
    calls = 0

    def fail_second(binding, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected install failure")
        original_install(binding, mode)

    big_leaf._install_forward = fail_second
    try:
        expect_raises(lambda: apply_big_leaf(model), "injected install failure")
    finally:
        big_leaf._install_forward = original_install
    assert not hasattr(model, "_big_leaf_plan")
    assert not model.first.large_weight_switch and not model.second.large_weight_switch
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name]), name
    print("PASS partial install rollback", flush=True)


def test_failed_external_repair_blocks_forward():
    module = make()
    plan = apply_big_leaf(module)

    def invalid_update():
        with plan.external_weight_update():
            module.weight0.data = torch.empty_like(module.weight0, dtype=torch.float32)

    expect_raises(invalid_update, "shape/dtype/device")
    assert plan._broken
    expect_raises(lambda: module(*input_for()), "broken")
    print("PASS failed external repair blocks later forward", flush=True)


def test_environment_subclass_probe_and_delayed_wgrad():
    previous = os.environ.get("XTE_USE_GROUPED_FC_FUSION")
    os.environ["XTE_USE_GROUPED_FC_FUSION"] = "1"
    module = make()
    ptr = module.weight0.data_ptr()
    try:
        expect_raises(lambda: apply_big_leaf(module), "XTE_USE_GROUPED_FC_FUSION")
    finally:
        if previous is None:
            os.environ.pop("XTE_USE_GROUPED_FC_FUSION", None)
        else:
            os.environ["XTE_USE_GROUPED_FC_FUSION"] = previous
    assert module.weight0.data_ptr() == ptr

    class CustomGroupedLinear(GroupedLinear):
        def forward(self, inp, m_splits):
            return super().forward(inp, m_splits)

    for mode in ("switch", "patch"):
        custom = make(cls=CustomGroupedLinear)
        expect_raises(lambda: apply_big_leaf(custom, forward_mode=mode), "子类")

    module = make()
    plan = apply_big_leaf(module, probe_input=torch.randn(G * TOK + 1, FIN, device="cuda",
                                                        dtype=torch.bfloat16))
    plan.assert_healthy()
    plan.restore()

    delayed = make(delay_wgrad_compute=True)
    expect_raises(lambda: apply_big_leaf(delayed), "delayed-wgrad")

    module = make()
    module.config = types.SimpleNamespace(overlap_dispatch_backward_with_experts_wgrad=True)
    expect_raises(lambda: BigLeafMegatronPlan.prepare(module), "dispatch-wgrad")
    print("PASS environment/subclass/probe/delayed-wgrad", flush=True)


def test_switch_health_check_survives_python_optimization():
    module = make()
    plan = apply_big_leaf(module, forward_mode="switch")
    module.large_weight_switch = False
    expect_raises(plan.assert_healthy, "large_weight_switch")
    plan.restore()
    print("PASS switch health check under python -O", flush=True)


if __name__ == "__main__":
    test_restore_idempotent_after_stock_update()
    test_dynamic_freeze_and_late_hooks_fail_fast()
    test_assign_positional_child_and_existing_grad()
    test_apply_install_failure_rolls_back_all_modules()
    test_failed_external_repair_blocks_forward()
    test_environment_subclass_probe_and_delayed_wgrad()
    test_switch_health_check_survives_python_optimization()
    print("BIG_LEAF_REVIEW_REGRESSIONS PASS", flush=True)
