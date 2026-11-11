"""真实 mcore_bridge LoraParallelLinear + TEGroupedLinear 的 big_leaf 验证。"""
import os
import random

import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.core.extensions.transformer_engine import TEGroupedLinear
from megatron.core.transformer import TransformerConfig
from mcore_bridge.tuners.lora import LoraParallelLinear

from ttorch.bridge.big_leaf import (
    apply_big_leaf, convert_from_big_leaf, convert_to_big_leaf,
)

G, FIN, FOUT, TOK = 4, 32, 24, 4


def init_dist():
    if not dist.is_initialized():
        port = random.randint(30100, 30900)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                rank=0, world_size=1)
        mpu.initialize_model_parallel()


def make_model():
    def init_method(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=0.02)

    config = TransformerConfig(
        num_layers=1, hidden_size=FIN, num_attention_heads=1,
        bf16=True, params_dtype=torch.bfloat16, init_method=init_method,
        expert_model_parallel_size=1, gradient_accumulation_fusion=False,
        num_moe_experts=G,
    )
    base = TEGroupedLinear(
        num_gemms=G, input_size=FIN, output_size=FOUT, parallel_mode=None,
        config=config, init_method=init_method, bias=False, skip_bias_add=True,
        is_expert=True,
    )
    base.is_expert = True
    model = LoraParallelLinear(
        base_layer=base, adapter_name="default", r=4, lora_alpha=8,
        lora_dropout=0.0, init_lora_weights=True, lora_bias=False,
    ).cuda()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * 0.02)
    for p in model.base_layer.parameters():
        p.requires_grad_(False)
    return model


def forward(model, x):
    result = model(x, [TOK] * G)
    return result[0] if isinstance(result, tuple) else result


def compare_trainable_weights(stock, big):
    for name, p in big.named_parameters():
        if p.requires_grad:
            expected = dict(stock.named_parameters())[name]
            assert torch.equal(p, expected), name


def main():
    init_dist()
    stock = make_model()
    big = make_model()
    big.load_state_dict(stock.state_dict())
    source_groups = [{"params": [p for p in stock.parameters() if p.requires_grad],
                      "lr": 1e-3, "weight_decay": 0.0}]
    target_groups = [{"params": [p for p in big.parameters() if p.requires_grad],
                      "lr": 1e-3, "weight_decay": 0.0}]
    stock_opt = torch.optim.AdamW(source_groups, foreach=False)
    plan = apply_big_leaf(big, forward_mode="patch")
    mapped_groups = plan.optimizer_param_groups(big, target_groups)
    assert sum(len(group["params"]) for group in mapped_groups) == 2, mapped_groups
    big_opt = torch.optim.AdamW(mapped_groups, foreach=False)

    torch.manual_seed(321)
    x = torch.randn(G * TOK, FIN, device="cuda", dtype=torch.bfloat16)
    expected_out = forward(stock, x)
    actual_out = forward(big, x)
    assert torch.equal(actual_out, expected_out)
    expected_out.float().square().mean().backward()
    actual_out.float().square().mean().backward()

    stock_modules = dict(stock.named_modules())
    for name, binding_module in big.named_modules():
        if name not in ("lora_A.default", "lora_B.default"):
            continue
        binding = next(b for b in plan.bindings if b.module is binding_module)
        stock_module = stock_modules[name]
        expected_grad = torch.stack([getattr(stock_module, f"weight{i}").grad
                                     for i in range(G)])
        assert torch.equal(binding.big.grad, expected_grad), name
    base_binding = next(b for b in plan.bindings if b.module is big.base_layer)
    assert not base_binding.big.requires_grad and base_binding.big.grad is None

    stock_opt.step()
    big_opt.step()
    compare_trainable_weights(stock, big)

    stock.merge(safe_merge=True)
    big.merge(safe_merge=True)
    plan.assert_healthy()
    assert torch.equal(forward(big, x), forward(stock, x))
    for i in range(G):
        assert torch.equal(getattr(big.base_layer, f"weight{i}"),
                           getattr(stock.base_layer, f"weight{i}"))

    stock.unmerge()
    big.unmerge()
    plan.assert_healthy()
    assert torch.equal(forward(big, x), forward(stock, x))
    for i in range(G):
        assert torch.equal(getattr(big.base_layer, f"weight{i}"),
                           getattr(stock.base_layer, f"weight{i}"))

    # unmerge 后继续训练，确认 adapter big_leaf 与 optimizer 仍有效。
    stock_opt.zero_grad()
    big_opt.zero_grad()
    forward(stock, x).float().square().mean().backward()
    forward(big, x).float().square().mean().backward()
    stock_opt.step()
    big_opt.step()
    compare_trainable_weights(stock, big)

    # 冻结 base 不在两个 optimizer 中；双向转换只处理真实 adapter 实体。
    converted = convert_to_big_leaf(plan, stock_opt, stock, big_opt, big)
    big_opt.load_state_dict(converted)
    reverse = convert_from_big_leaf(plan, big_opt, big, stock_opt, stock)
    stock_opt.load_state_dict(reverse)
    stock_opt.zero_grad()
    big_opt.zero_grad()
    forward(stock, x).float().square().mean().backward()
    forward(big, x).float().square().mean().backward()
    stock_opt.step()
    big_opt.step()
    compare_trainable_weights(stock, big)

    plan.restore()
    stock.merge(safe_merge=True)
    big.merge(safe_merge=True)
    assert torch.equal(forward(big, x), forward(stock, x))
    print("BIG_LEAF_LORA PASS (real mcore_bridge LoraParallelLinear)", flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
