"""big_leaf 参数分组、optimizer checkpoint 与外部权重更新测试。"""
import copy
import os

import torch
from transformer_engine.pytorch import GroupedLinear

from ttorch.bridge.big_leaf import (
    apply_big_leaf, convert_from_big_leaf, convert_to_big_leaf,
)

G, FIN, FOUT = 4, 32, 24
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def make(use_bias=True, lw1=False):
    previous = os.environ.get("XTE_GROUPED_GEMM_LARGE_WEIGHT")
    os.environ["XTE_GROUPED_GEMM_LARGE_WEIGHT"] = "1" if lw1 else "0"
    try:
        m = GroupedLinear(G, FIN, FOUT, bias=use_bias, params_dtype=torch.bfloat16).cuda()
    finally:
        if previous is None:
            os.environ.pop("XTE_GROUPED_GEMM_LARGE_WEIGHT", None)
        else:
            os.environ["XTE_GROUPED_GEMM_LARGE_WEIGHT"] = previous
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.03)
    return m


def expert_groups(m):
    weights = [getattr(m, f"weight{i}") for i in range(G)]
    groups = [{"params": weights, "lr": 1e-3, "weight_decay": 0.1}]
    if m.use_bias:
        groups.append({"params": [getattr(m, f"bias{i}") for i in range(G)],
                       "lr": 2e-3, "weight_decay": 0.0})
    return groups


def set_manual_grads(m, step):
    torch.manual_seed(9000 + step)
    if getattr(m, "large_weight_switch", False):
        m.large_weight.grad = torch.randn_like(m.large_weight) * 0.01
        if m.large_bias is not None:
            m.large_bias.grad = torch.randn_like(m.large_bias) * 0.01
    elif hasattr(m, "large_weight"):
        m.large_weight.grad = torch.randn_like(m.large_weight) * 0.01
        if getattr(m, "large_bias", None) is not None:
            m.large_bias.grad = torch.randn_like(m.large_bias) * 0.01
    else:
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 0.01


def copy_stock_grads_to_big(stock, big, step):
    set_manual_grads(stock, step)
    big.large_weight.grad = torch.stack([getattr(stock, f"weight{i}").grad
                                         for i in range(G)]).clone()
    if stock.use_bias:
        big.large_bias.grad = torch.stack([getattr(stock, f"bias{i}").grad
                                           for i in range(G)]).clone()


def assert_stock_big_equal(stock, big):
    for i in range(G):
        assert torch.equal(getattr(stock, f"weight{i}"), getattr(big, f"weight{i}")), i
        if stock.use_bias:
            assert torch.equal(getattr(stock, f"bias{i}"), getattr(big, f"bias{i}")), i


@check
def test_group_mapping_and_roundtrip_checkpoint():
    stock, big = make(), make()
    big.load_state_dict(stock.state_dict())
    source_groups = expert_groups(stock)
    target_stock_groups = expert_groups(big)
    source_opt = torch.optim.AdamW(source_groups, foreach=False)
    plan = apply_big_leaf(big)
    target_groups = plan.optimizer_param_groups(big, target_stock_groups)
    assert len(target_groups) == 2 and target_groups[0]["params"] == [plan.bindings[0].big]
    assert target_groups[1]["params"] == [plan.bindings[0].big_b]
    assert target_groups[0]["weight_decay"] == 0.1 and target_groups[1]["weight_decay"] == 0.0
    target_opt = torch.optim.AdamW(target_groups, foreach=False)

    for step in range(2):
        set_manual_grads(stock, step)
        source_opt.step()
    big.load_state_dict(stock.state_dict())
    converted = convert_to_big_leaf(plan, source_opt, stock, target_opt, big)
    target_opt.load_state_dict(converted)
    big_state = target_opt.state[plan.bindings[0].big]
    for i in range(G):
        source_state = source_opt.state[getattr(stock, f"weight{i}")]
        assert torch.equal(big_state["exp_avg"][i], source_state["exp_avg"])
        assert torch.equal(big_state["exp_avg_sq"][i], source_state["exp_avg_sq"])

    copy_stock_grads_to_big(stock, big, 10)
    source_opt.step()
    target_opt.step()
    assert_stock_big_equal(stock, big)

    restored = make()
    restored.load_state_dict(big.state_dict())
    restored_opt = torch.optim.AdamW(expert_groups(restored), foreach=False)
    reverse = convert_from_big_leaf(plan, target_opt, big, restored_opt, restored)
    restored_opt.load_state_dict(reverse)
    for i in range(G):
        state = restored_opt.state[getattr(restored, f"weight{i}")]
        assert torch.equal(state["exp_avg"], big_state["exp_avg"][i])
        assert torch.equal(state["exp_avg_sq"], big_state["exp_avg_sq"][i])

    copy_stock_grads_to_big(restored, big, 11)
    restored_opt.step()
    target_opt.step()
    assert_stock_big_equal(restored, big)
    plan.restore()


@check
def test_lw1_checkpoint_resume():
    lw1 = make(use_bias=False, lw1=True)
    source_opt = torch.optim.AdamW(lw1.parameters(), lr=1e-3, foreach=False)
    for step in range(2):
        set_manual_grads(lw1, step)
        source_opt.step()

    big = make(use_bias=False)
    with torch.no_grad():
        for i in range(G):
            getattr(big, f"weight{i}").copy_(lw1.large_weight[i])
    plan = apply_big_leaf(big)
    target_opt = torch.optim.AdamW(plan.optimizer_params(big), lr=1e-3, foreach=False)
    converted = convert_to_big_leaf(plan, source_opt, lw1, target_opt, big)
    target_opt.load_state_dict(converted)
    source_state = source_opt.state[lw1.large_weight]
    target_state = target_opt.state[plan.bindings[0].big]
    assert torch.equal(target_state["exp_avg"], source_state["exp_avg"])
    assert torch.equal(target_state["exp_avg_sq"], source_state["exp_avg_sq"])
    plan.restore()


@check
def test_mixed_group_rejected():
    m = make(use_bias=False)
    groups = [{"params": [m.weight0], "lr": 0.1},
              {"params": [getattr(m, f"weight{i}") for i in range(1, G)], "lr": 0.2}]
    plan = apply_big_leaf(m)
    try:
        plan.optimizer_param_groups(m, groups)
        raise AssertionError("专家跨 lr 组未被拒绝")
    except ValueError as e:
        assert "同一 optimizer group" in str(e)
    plan.restore()


class FakeLora(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.merged = False

    def get_base_layer(self):
        return self.base

    def merge(self, safe_merge=False, adapter_names=None):
        with torch.no_grad():
            for i in range(G):
                w = getattr(self.base, f"weight{i}")
                if safe_merge:
                    assert torch.isfinite(w).all()
                w.data = (w.data + (i + 1) * 0.01).clone()
        self.merged = True

    def unmerge(self):
        with torch.no_grad():
            for i in range(G):
                w = getattr(self.base, f"weight{i}")
                w.data = (w.data - (i + 1) * 0.01).clone()
        self.merged = False


@check
def test_external_update_and_lora_guard():
    base = make(use_bias=False)
    model = torch.nn.Module()
    model.adapter = FakeLora(base)
    plan = apply_big_leaf(model)
    expected = plan.bindings[0].buf.clone()
    for i in range(G):
        expected[i] += (i + 1) * 0.01

    model.adapter.merge(safe_merge=True)
    assert model.adapter.merged
    assert torch.equal(plan.bindings[0].buf, expected)
    plan.assert_healthy()
    merged = plan.bindings[0].buf.clone()
    model.adapter.unmerge()
    assert not model.adapter.merged
    expected_unmerged = merged.clone()
    for i in range(G):
        expected_unmerged[i] -= (i + 1) * 0.01
    assert torch.equal(plan.bindings[0].buf, expected_unmerged)

    sd = copy.deepcopy(model.state_dict())
    try:
        model.load_state_dict(sd, assign=True)
        raise AssertionError("assign=True 未被拒绝")
    except ValueError as e:
        assert "assign=True" in str(e)
    model.load_state_dict(sd)
    plan.assert_healthy()
    plan.restore()
    model.adapter.merge()
    assert all(getattr(base, f"weight{i}").data.data_ptr() != plan.bindings[0].buf[i].data_ptr()
               for i in range(G))


@check
def test_invalid_core_forms_fail_fast():
    m = make(use_bias=False)
    m.weight0.requires_grad_(False)
    try:
        apply_big_leaf(m)
        raise AssertionError("混合冻结未被拒绝")
    except ValueError as e:
        assert "混合冻结" in str(e)

    m = make(use_bias=False)
    m.weight0.register_hook(lambda grad: grad)
    try:
        apply_big_leaf(m)
        raise AssertionError("旧 per-param hook 未被拒绝")
    except ValueError as e:
        assert "per-param hook" in str(e)

    m = make(use_bias=False)
    m.weight0 = torch.nn.Parameter(m.weight0.detach().to(torch.int8), requires_grad=False)
    try:
        apply_big_leaf(m)
        raise AssertionError("量化参数未被拒绝")
    except ValueError as e:
        assert "量化" in str(e) or "非浮点" in str(e)


if __name__ == "__main__":
    for fn in CHECKS:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"BIG_LEAF_CHECKPOINT_SUITE PASS ({len(CHECKS)}/{len(CHECKS)})", flush=True)
