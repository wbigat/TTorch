"""big_leaf 独立 forward（不依赖 large_weight_switch）的数值与失败注入测试。"""
import os

import torch
from transformer_engine.pytorch import GroupedLinear

from ttorch.bridge.big_leaf import apply_big_leaf
from ttorch.bridge.big_leaf import core as big_leaf

G, FIN, FOUT, TOK = 4, 32, 24, 4


def make(bias=True):
    torch.manual_seed(123)
    m = GroupedLinear(G, FIN, FOUT, bias=bias, params_dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.03)
    return m


def grads(m):
    if getattr(m, "_big_leaf_binding", None) is not None:
        binding = m._big_leaf_binding
        return binding.big.grad, binding.big_b.grad if binding.big_b is not None else None
    if m.large_weight_switch:
        return m.large_weight.grad, m.large_bias.grad if m.use_bias else None
    return (torch.stack([getattr(m, f"weight{i}").grad for i in range(G)]),
            torch.stack([getattr(m, f"bias{i}").grad for i in range(G)]) if m.use_bias else None)


def run_pair(bias, fast_fc):
    previous = os.environ.get("XMLIR_ENABLE_FAST_FC")
    os.environ["XMLIR_ENABLE_FAST_FC"] = "1" if fast_fc else "0"
    try:
        stock, patched = make(bias), make(bias)
        patched.load_state_dict(stock.state_dict())
        plan = apply_big_leaf(patched, forward_mode="patch")
        module = plan.bindings[0].module
        assert not module.large_weight_switch
        assert "large_weight" not in module.__dict__ and "large_bias" not in module.__dict__
        assert module.forward.__func__ is big_leaf._big_leaf_forward
        plan.assert_healthy()

        torch.manual_seed(456)
        x = torch.randn(G * TOK, FIN, device="cuda", dtype=torch.bfloat16)
        x3d = x.view(2, G * TOK // 2, FIN)
        ms = [TOK] * G
        expected_out = stock(x3d, ms)
        actual_out = patched(x3d, ms)
        assert torch.equal(actual_out, expected_out)

        sx = x.clone().requires_grad_(True)
        px = x.clone().requires_grad_(True)
        stock(sx, ms).float().square().mean().backward()
        patched(px, ms).float().square().mean().backward()
        assert torch.equal(px.grad, sx.grad)
        expected_wgrad, expected_bgrad = grads(stock)
        actual_wgrad, actual_bgrad = grads(patched)
        assert torch.equal(actual_wgrad, expected_wgrad)
        if bias:
            assert torch.equal(actual_bgrad, expected_bgrad)

        # 第二个 microbatch 继续累积，验证没有切回 stock 叶子。
        stock(sx, ms).float().square().mean().backward()
        patched(px, ms).float().square().mean().backward()
        expected_wgrad, expected_bgrad = grads(stock)
        actual_wgrad, actual_bgrad = grads(patched)
        assert torch.equal(actual_wgrad, expected_wgrad)
        if bias:
            assert torch.equal(actual_bgrad, expected_bgrad)
        assert all(getattr(module, f"weight{i}").grad is None for i in range(G))

        sd = stock.state_dict()
        patched.load_state_dict(sd)
        plan.assert_healthy()
        try:
            patched.load_state_dict(sd, assign=True)
            raise AssertionError("patch 模式 assign=True 未被拒绝")
        except ValueError:
            pass

        original_forward = module.forward
        module.forward = plan.bindings[0].original_forward
        try:
            plan.assert_healthy()
            raise AssertionError("forward 被篡改未被健康断言发现")
        except RuntimeError as e:
            assert "独立 forward" in str(e)
        module.forward = original_forward

        plan.restore()
        assert "large_weight" not in module.__dict__
        assert module.forward.__func__ is not big_leaf._big_leaf_forward
        assert torch.equal(patched(x3d, ms), stock(x3d, ms))
    finally:
        if previous is None:
            os.environ.pop("XMLIR_ENABLE_FAST_FC", None)
        else:
            os.environ["XMLIR_ENABLE_FAST_FC"] = previous


def test_fast_paths():
    for bias in (False, True):
        for fast_fc in (False, True):
            run_pair(bias, fast_fc)
            print(f"PASS patch forward bias={bias} fast_fc={fast_fc}", flush=True)


def test_rejections_and_atomicity():
    m = make(False)
    original_ptr = m.weight0.data_ptr()
    original = big_leaf._SUPPORTED_FORWARD_SHA256
    big_leaf._SUPPORTED_FORWARD_SHA256 = "bad"
    try:
        apply_big_leaf(m, forward_mode="patch")
        raise AssertionError("错误指纹未被拒绝")
    except RuntimeError as e:
        assert "源码已变化" in str(e)
    finally:
        big_leaf._SUPPORTED_FORWARD_SHA256 = original
    assert m.weight0.data_ptr() == original_ptr and not hasattr(m, "_big_leaf_plan")

    plan = apply_big_leaf(m, forward_mode="patch")
    previous = os.environ.get("XTE_USE_GROUPED_FC_FUSION")
    os.environ["XTE_USE_GROUPED_FC_FUSION"] = "1"
    try:
        m(torch.randn(G * TOK, FIN, device="cuda", dtype=torch.bfloat16), [TOK] * G)
        raise AssertionError("list grouped-fusion 未被拒绝")
    except RuntimeError as e:
        assert "不支持" in str(e)
    finally:
        if previous is None:
            os.environ.pop("XTE_USE_GROUPED_FC_FUSION", None)
        else:
            os.environ["XTE_USE_GROUPED_FC_FUSION"] = previous
    plan.restore()


if __name__ == "__main__":
    test_fast_paths()
    test_rejections_and_atomicity()
    print("PASS patch rejection/fingerprint/rollback", flush=True)
    print("BIG_LEAF_FORWARD_PATCH PASS", flush=True)
