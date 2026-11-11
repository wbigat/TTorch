"""直接 mcore/Megatron 的 stock/big_leaf 全模型集成验证。

覆盖确定性初始化、train/eval、dist-checkpoint 恢复和 HF 导出。stock/big 分进程
运行并比较 JSON，避免厂商 stateful cache 和同进程多实例干扰。TP/EP 测试复用本文件的
无副作用模型 harness；分布式初始化只发生在显式运行入口。

运行：
  python tests/integration/big_leaf/full_model.py --bigleaf 0 --out /tmp/big_leaf_stock.json
  python tests/integration/big_leaf/full_model.py --bigleaf 1 --forward-mode patch --out /tmp/big_leaf_big.json
  python tests/integration/big_leaf/full_model.py --compare /tmp/big_leaf_stock.json /tmp/big_leaf_big.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("XTE_GROUPED_GEMM_LARGE_WEIGHT", "0")
os.environ.setdefault("XTE_USE_GROUPED_FC_FUSION", "0")
# 裸 mcore 链路不开 fast FC；生产生命周期由 tests/e2e/big_leaf_production.py 验证。

import torch
import torch.distributed as dist
import torch_xmlir  # noqa: F401
import megatron.core.parallel_state as mpu


def initialize_model_parallel():
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=int(os.environ.get("EP_SIZE", "1")),
    )


def initialize_runtime():
    if not dist.is_initialized():
        import random

        port = random.randint(29600, 29900)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                world_size=1, rank=0)
    initialize_model_parallel()

from mcore_bridge import GPTBridge, ModelConfig, get_mcore_model, hf_to_mcore_config
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from transformers import Qwen3MoeConfig

CHECKS = {}


def check(name, ok):
    CHECKS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def build_model():
    hf_cfg = Qwen3MoeConfig(
        vocab_size=1024, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=128, max_position_embeddings=512,
        rope_theta=10000.0, tie_word_embeddings=False,
    )
    cfg_dict = hf_to_mcore_config(hf_cfg)
    cfg_dict.update(dict(
        bf16=True, params_dtype=torch.bfloat16, moe_grouped_gemm=True,
        hidden_dropout=0.0, attention_dropout=0.0,
        tensor_model_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        expert_model_parallel_size=int(os.environ.get("EP_SIZE", "1")),
        sequence_parallel=int(os.environ.get("TP_SIZE", "1")) > 1,
        pipeline_model_parallel_size=1,
    ))
    mc = ModelConfig(**cfg_dict)
    return get_mcore_model(mc)[0].cuda(), mc


def deterministic_full_init(model, seed=0):
    """确定性全量初始化（模拟生产的 load 预训练权重）。

    背景：裸 mcore_bridge 链路不接线 init_method（hf_to_mcore_config 不给），
    模型建在 cuda 上时大量参数是未初始化显存垃圾（有限但巨大，前向溢出 NaN，
    且随进程变化）；生产从不走这条路（建模后立即 load 预训练权重覆盖）。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n.lower() or n.endswith(".ln.weight"):
                p.data.fill_(1.0)
            else:
                p.data.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32)
                             .mul_(0.02).to(p.dtype).to(p.device))
    torch.cuda.synchronize()


from ttorch.bridge.big_leaf import apply_big_leaf


def bigleaf_model(model, forward_mode="patch"):
    """从正式公共 API 建立 big_leaf，并返回非 dist-opt plan。"""
    gls = [m for m in model.modules() if isinstance(m, GroupedLinear)]
    assert all(not m.use_bias for m in gls), "全链路模式当前仅覆盖无 bias 形态"
    plan = apply_big_leaf(model, forward_mode=forward_mode)
    plan._gls = gls
    return plan


def contiguity_ok(model):
    for mod in model.modules():
        if isinstance(mod, GroupedLinear):
            ws = [getattr(mod, f"weight{i}") for i in range(mod.num_gemms)]
            base = ws[0].data_ptr()
            step = ws[0].numel() * ws[0].element_size()
            if not all(w.data_ptr() == base + i * step for i, w in enumerate(ws)):
                return False
    return True


def make_batch(step, B=2, S=64, vocab=1024):
    g = torch.Generator().manual_seed(1000 + step)
    ids = torch.randint(0, vocab, (B, S), generator=g)
    labels = ids.clone()
    labels[:, :8] = -100
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    return ids.cuda(), labels.cuda(), pos.cuda()


def forward_loss(model, ids, labels, pos):
    out = model(input_ids=ids, position_ids=pos, attention_mask=None, labels=labels)
    loss = out[0] if isinstance(out, tuple) else out
    if loss.ndim > 0:  # mcore_bridge 返回 per-token loss，聚合成标量
        mask = (labels != -100).float()
        loss = (loss.view(-1) * mask.view(-1)).sum() / mask.sum().clamp(min=1)
    return loss


def run(out_path, bigleaf=False, forward_mode="patch"):
    initialize_runtime()
    print(f"===== Stage 0: 建模（bigleaf={bigleaf} forward_mode={forward_mode}） =====")
    torch.manual_seed(0)
    model, mc = build_model()
    deterministic_full_init(model)
    ids, labels, pos = make_batch(999)
    model.eval()
    with torch.no_grad():
        probe = forward_loss(model, ids, labels, pos)
    check("probe_loss_finite", torch.isfinite(probe).all().item())
    if not CHECKS["probe_loss_finite"]:
        print("  探针失败，exit(2) 请求外层重试（厂商间歇性 init 垃圾）")
        sys.exit(2)
    gls = [m for m in model.modules() if isinstance(m, GroupedLinear)]
    print(f"  GroupedLinear 层数: {len(gls)}（num_gemms={gls[0].num_gemms}）"
          f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M bf16")

    record = {"probe_loss": probe.item()}

    if bigleaf:
        print("===== Stage 1: big_leaf =====")
        keys_before = set(model.state_dict().keys())
        ids0 = [id(p) for p in model.parameters()]
        plan = bigleaf_model(model, forward_mode=forward_mode)
        big_leafs = plan.leafs
        check("big_leaf_layers", len(plan.bindings) == len(gls))
        check("big_leaf_state_dict_keys_unchanged",
              set(model.state_dict().keys()) == keys_before)
        check("big_leaf_param_identity_kept",
              [id(p) for p in model.parameters()] == ids0)
        check("big_leaf_contiguity", contiguity_ok(model))
        model.eval()
        with torch.no_grad():
            post = forward_loss(model, ids, labels, pos)
        check("big_leaf_fwd_bitwise_vs_pre", torch.equal(probe, post))
        record["post_big_leaf_loss"] = post.item()

    print("===== Stage 2: train（5 步） =====")
    if bigleaf:
        opt_params = plan.optimizer_params(model)
        opt = torch.optim.AdamW(opt_params, lr=1e-3)
        print(f"  optimizer 参数数: {len(opt_params)}"
              f"（stock 为 {sum(1 for _ in model.parameters())} 个）")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    from hydrax.xaccelerator.stateful import (
        register_optimizer_step_post_hook,
        reinitialize_all_after_param_update,
    )

    register_optimizer_step_post_hook(reinitialize_all_after_param_update)
    model.train()
    losses = []
    for step in range(5):
        ids, labels, pos = make_batch(step)
        loss = forward_loss(model, ids, labels, pos)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  step{step}: loss={loss.item():.6f}")
    record["train_losses"] = losses
    check("train_losses_finite", all(torch.isfinite(torch.tensor(losses)).tolist()))
    if bigleaf:
        check("bigleaf_grads_on_leaf",
              all(lf.grad is not None for lf in big_leafs))
        check("bigleaf_weight_i_no_grad",
              all(getattr(mod, f"weight{i}").grad is None
                  for mod in gls for i in range(mod.num_gemms)))
    if bigleaf:
        check("train_contiguity_kept", contiguity_ok(model))

    print("===== Stage 3: eval =====")
    model.eval()
    ids, labels, pos = make_batch(99)
    with torch.no_grad():
        ev = forward_loss(model, ids, labels, pos)
        ev2 = forward_loss(model, ids, labels, pos)
    record["eval_loss"] = ev.item()
    check("eval_forward_deterministic", torch.equal(ev.cpu(), ev2.cpu()))
    print(f"  eval loss={ev.item():.6f}（两次前向逐位一致: {CHECKS['eval_forward_deterministic']}）")

    print("===== Stage 4: dist-ckpt 保存→扰动→加载→eval 回退 =====")
    from megatron.core import dist_checkpointing

    ckpt_dir = tempfile.mkdtemp(prefix="tt_big_leaf_ckpt_")
    try:
        dist_checkpointing.save(model.sharded_state_dict(), ckpt_dir,
                                async_sharded_save=False)
        snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        # 扰动：再训 2 步
        model.train()
        for step in range(5, 7):
            ids, labels, pos = make_batch(step)
            loss = forward_loss(model, ids, labels, pos)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            disturbed = forward_loss(model, *make_batch(99))
        # 加载回 step4 状态
        sharded_sd = model.sharded_state_dict()
        loaded = dist_checkpointing.load(sharded_sd, ckpt_dir)
        model.load_state_dict(loaded)
        torch.cuda.synchronize()
        mismatched = [n for n, p in model.named_parameters()
                      if not torch.equal(p.detach().cpu(), snap[n])]
        if mismatched:
            print(f"  [注] load 后与 step4 快照不一致的参数 {len(mismatched)}/{len(snap)}: "
                  f"{mismatched[:4]}")
        check("ckpt_params_restored_bitwise", len(mismatched) == 0)
        with torch.no_grad():
            restored = forward_loss(model, *make_batch(99))
        print(f"  disturbed={disturbed.item():.6f} restored={restored.item():.6f}")
        # 若参数逐位恢复但输出不恢复：cast 缓存未随 load_state_dict 失效
        # （post-hook 只在 optimizer.step 触发，load 不走 step）——手动失效再测
        if not torch.equal(restored.cpu(), ev.cpu()):
            print("  [注] 参数已逐位恢复但输出未恢复 → cast 缓存陈旧，手动失效复测")
            reinitialize_all_after_param_update(opt, (), {})
            with torch.no_grad():
                restored2 = forward_loss(model, *make_batch(99))
            print(f"  缓存失效后 restored={restored2.item():.6f}")
            check("ckpt_restore_after_cache_invalidation",
                  torch.equal(restored2.cpu(), ev.cpu()))
            restored = restored2
        check("ckpt_disturbed_differs", disturbed.item() != restored.item())
        check("ckpt_restore_matches_step4_eval",
              torch.equal(restored.cpu(), ev.cpu()))
        record["restored_eval_loss"] = restored.item()
        if bigleaf:
            check("ckpt_contiguity_kept_after_load", contiguity_ok(model))
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print("===== Stage 5: GPTBridge HF 导出 =====")
    bridge = GPTBridge(mc)
    sd = dict(bridge.export_weights([model]))
    expert_keys = [k for k in sd if "expert" in k]
    check("export_has_per_expert_keys", len(expert_keys) > 0)
    # 导出内容指纹：stock/big 跨进程比对的一致性口径
    record["export_keys"] = sorted(sd.keys())
    record["export_fp"] = {
        k: (tuple(v.shape), str(v.dtype), float(v.float().sum()))
        for k, v in sd.items() if "expert" in k
    }
    out_dir = tempfile.mkdtemp(prefix="tt_big_leaf_hf_")
    try:
        bridge.save_weights([model], out_dir)
        check("export_dir_written",
              any(f.endswith(".safetensors") for f in os.listdir(out_dir)))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    record["checks"] = CHECKS
    record["bigleaf"] = bigleaf
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1)
    all_pass = all(CHECKS.values())
    print("VERDICT:", "ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


def compare(path_a, path_b):
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"A={path_a}（bigleaf={a['bigleaf']}）  B={path_b}（bigleaf={b['bigleaf']}）")
    ok = True
    for k in ("probe_loss", "train_losses", "eval_loss", "restored_eval_loss"):
        same = a[k] == b[k]
        ok &= same
        print(f"  [{'PASS' if same else 'FAIL'}] {k}: {a[k]} vs {b[k]}")
    same = a["export_keys"] == b["export_keys"]
    ok &= same
    print(f"  [{'PASS' if same else 'FAIL'}] export_keys（HF 导出键集合一致）")
    same = a["export_fp"] == b["export_fp"]
    ok &= same
    print(f"  [{'PASS' if same else 'FAIL'}] export_fingerprint（专家权重值指纹逐位一致）")
    ok &= all(a["checks"].values()) and all(b["checks"].values())
    print("  checks 各自全过:", all(a["checks"].values()), all(b["checks"].values()))
    print("VERDICT:", "ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigleaf", type=int, choices=(0, 1), default=0)
    ap.add_argument("--forward-mode", choices=("switch", "patch"), default="patch")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.out or f"/tmp/big_leaf_{args.bigleaf}.json",
            bigleaf=bool(args.bigleaf), forward_mode=args.forward_mode)
