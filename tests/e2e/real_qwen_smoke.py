"""真实 Qwen3.5 SFT 穿透 smoke：不经厂商训练框架封装，纯 transformers + TT。

目的（穿透优先，不铺面）：验证 TT 在**真实模型 + 真实框架 + 真实数据**下
成立。与 real_sft_smoke.py 的分工：后者用自造 TinyLLM 验 P0 接口族的契约
与跨设备精度；本文件验真实链路的调用面、import 时序与 checkpoint。

覆盖的 swift 主链路环节：
  tt.activate()  ->  tokenizer + chat template + loss mask  ->  模型加载
  ->  SFT 训练循环（含 Bridge 设备交互）  ->  checkpoint save
  ->  resume 后 loss 连续性验证

明确不做（当前穿透点之外，撞到就报未实现）：
- 不经 ms-swift / megatron / mcore_bridge（XSWIFT_PLUGIN_ENABLE=0 关闭，
  厂商基座 torch_xmlir 保留）；
- 不做 LoRA、TP/PP/EP 并行、MoE dispatcher、多模态 ViT、packing；
- 多卡分布式（环境不支持）；
- policy 表仍是 module_binding 的 14 符号，未纳入的走 default-deny。

用法：
  python real_qwen_smoke.py
  python real_qwen_smoke.py --platform cpu --steps 4 --max-length 256
环境：XSWIFT_PLUGIN_ENABLE=0 XME_PLUGIN_ENABLE=0

判定（退出码反映）：
  绑定  : admission 全部 PASS
  数据  : loss mask 边界正确（prompt 段全 masked、answer 段非空）
  功能  : 训练 loss 下降且全程有限
  检查点: schema 校验通过 + resume 后 loss 轨迹与原轨迹吻合
"""
# ---- import 时序硬约束：TT 必须先于训练框架（见 tt.py 模块注释）----
import ttorch as tt  # noqa: I001  # isort: skip  —— 必须是第一个项目内 import

import argparse
import math
import os
import sys
import time

import torch

DEFAULT_MODEL = os.environ.get(
    "TT_QWEN_MODEL",
    "/apdcephfs_bj36/share_306186457/download/huggingface/Qwen/Qwen3.5-0.8B")
DEFAULT_DATA = os.environ.get(
    "TT_SFT_DATA",
    "/cfs_hunyuanaide/172076/caption_merged_messages_sample150_llamafactory_bj.jsonl")


def _bridge_io(step):
    """swift/megatron 主链路形态的设备交互，全走 torch.cuda.*（已绑到门面）。"""
    if step % 5 == 0:  # checkpoint 形态：RNG 存取回环
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    if tt.active().platform != "cpu" and torch.cuda.is_available():
        side, main = torch.cuda.Stream(), torch.cuda.current_stream()
        side.wait_stream(main)          # ddp 侧流 fencing 形态
        with torch.cuda.stream(side):
            pass                        # 梯度 bucket 通信占位
        main.wait_stream(side)
    torch.cuda.synchronize()


def train_loop(model, opt, batches, device, steps, bridged=True, start=0, fixed=False):
    """SFT 训练循环。返回 losses。start 用于 resume 后对齐 batch 顺序。

    fixed=True 时每步复用同一个 batch（过拟合 sanity check）：loss 在
    **同一批数据**上单调下降才是有意义的收敛证据。轮换 batch 时相邻两步
    的 loss 来自不同样本，首尾比大小只反映样本难度差异，不反映学习效果。
    """
    losses = []
    model.train()
    for i in range(steps):
        ids, labels, attn = batches[start % len(batches) if fixed
                                    else (start + i) % len(batches)]
        out = model(input_ids=ids.to(device), labels=labels.to(device),
                    attention_mask=attn.to(device))
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if bridged:
            _bridge_io(i)
        losses.append(out.loss.detach().float().item())
    return losses


def _time_fwd_bwd(model, batches, device, n):
    """计时不更新权重的 前向+反向（不动 opt.step）——测 RMSNorm provider 的整网
    计算开销，同时不污染后续训练轨迹的初始权重。"""
    ids, labels, attn = batches[0]
    model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = model(input_ids=ids.to(device), labels=labels.to(device),
                    attention_mask=attn.to(device))
        out.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="auto",
                    choices=["auto", "cpu", "cuda", "xpu-masquerade", "xpu-native"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--resume-at", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    # 默认 8192：本机数据集样本总长中位数约 6k token，512 会让 100% 样本的
    # answer 被整段截掉（实测）。真实训练脚本用 SEQ_LEN=20480
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ckpt", default="/tmp/tt_qwen_ckpt.pt")
    ap.add_argument("--no-provider", action="store_true",
                    help="禁用 provider 自动使能（默认开启：RMSNorm 路由到厂商融合）")
    args = ap.parse_args()

    print(f"[env] torch={torch.__version__}")
    act = tt.activate(platform=args.platform)
    print(f"[env] vendor={act.vendor_name} device_name={act.device_name} "
          f"device_spec={act.device_spec()}")

    print("\n== 1. admission：绑定正确性 ==")
    n_fail, rows = tt.check_bindings()
    for ok, name, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    tt.start_call_trace()  # 装探针：后续统计运行期真实中转

    # 框架 import 必须在 tt.activate() 之后
    import tt_data
    import ttorch.bridge.checkpoint as tt_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n== 2. 真实数据 + tokenizer（{os.path.basename(args.data)}）==")
    tok = AutoTokenizer.from_pretrained(args.model)
    rows_raw = tt_data.load_alpaca_jsonl(args.data)
    batches, st = tt_data.build_batches(
        tok, rows_raw, args.batch_size, args.max_length)
    ids0, lab0, _ = batches[0]
    n_mask = int((lab0[0] == tt_data.IGNORE_INDEX).sum())
    n_ans = int(lab0.numel() // lab0.shape[0] - n_mask)
    usable = st["n_valid"] / max(st["n_rows"], 1)
    print(f"  纯文本样本={st['n_rows']} 有效={st['n_valid']}（{usable:.0%}）"
          f" batch 数={st['n_batches']} shape={tuple(ids0.shape)}")
    print(f"  max_length={st['max_length']}：被截断={st['n_truncated']}，"
          f"因 answer 整段被截而丢弃={st['n_dropped_answer_truncated']}")
    if usable < 0.5:
        print(f"  [warn] 过半样本不可用——该数据集样本很长（中位数约 6k token），"
              f"max_length 偏小会让训练只发生在少数短样本上（真实脚本 SEQ_LEN=20480）")
    print(f"  样本0 loss mask: prompt 段 masked={n_mask} answer 段={n_ans}")
    print(f"  answer 片段: {tok.decode([x for x in lab0[0].tolist() if x != -100][:24])!r}"[:140])
    # 判据含数据利用率：loss mask 对了但只剩少数样本，不算数据面打通
    data_ok = n_ans > 0 and n_mask > 0 and st["n_batches"] > 0 and usable >= 0.5
    print(f"  数据面达标（mask 正确 + 利用率≥50%）: {data_ok}")

    dev = torch.device(act.device_spec())
    dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16

    # feature gate：不支持就明确报错，不静默伪装（TT_Bridge_Provider_Architecture §4.1）。
    # Qwen3.5 的 linear_attention 层在本环境走厂商算子 fla_xpu_ops.causal_conv1d_fwd
    # （经 causal_conv1d 包分发），该算子只支持加速卡——CPU 上会崩在算子深处
    # （RuntimeError: CUDAGuardImpl initialized with non-CUDA DeviceType: cpu），
    # 堆栈指向 transformers 内部，与真实原因（平台能力不足）相距甚远。
    # 这类"第三方 CUDA 扩展"按 §4.3 第四类治理：不进 Bridge 模拟，显式禁用。
    if dev.type == "cpu":
        print(f"\n[gate] 平台 {act.vendor_name} 不支持本模型：Qwen3.5 的 "
              f"linear_attention 依赖厂商算子 causal_conv1d（仅加速卡），"
              f"CPU 无等价实现。")
        print("[gate] 这是平台能力边界，非 TT 缺陷；CPU 路径的 Bridge 机制"
              "验证请走 real_sft_smoke.py --platform cpu（自造模型，无厂商算子依赖）。")
        print("QWEN_SMOKE SKIPPED (platform unsupported)")
        tt.finalize()
        return 0

    print(f"\n== 3. 加载真实 Qwen（device={dev} dtype={dtype}）==")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager").to(dev)
    print(f"  {type(model).__name__} 参数量="
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B")

    # provider 自动使能：把模型的 RMSNorm 模块路由到 provider（自动发现+选择厂商
    # 融合）。A-B 计时不更新权重（只前向+反向），不污染后续训练轨迹。
    if not args.no_provider:
        n_wrapped = tt.provider_wrap(model)
        from ttorch.provider import manager as _pm
        _pm.set_bypass(True)   # baseline：原始 RMSNorm（厂商栈接管的 RMSNormFunction）
        _time_fwd_bwd(model, batches, dev, 3)  # warmup
        t_base = _time_fwd_bwd(model, batches, dev, 6)
        _pm.set_bypass(False)  # vendor：provider 路由到 torch.ops.custom_ops AOT kernel
        _time_fwd_bwd(model, batches, dev, 3)  # warmup（首次前向触发自动选择+门禁）
        t_vendor = _time_fwd_bwd(model, batches, dev, 6)
        print(f"  provider 自动使能: patch {n_wrapped} 个 RMSNorm 模块；"
              f"整网前向+反向 {t_base*1e3:.1f} -> {t_vendor*1e3:.1f} ms "
              f"({t_base/t_vendor:.2f}x)")
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"  mem_get_info（经门面）free={free / 2**30:.1f}GB "
              f"total={total / 2**30:.1f}GB")
    except tt.TTPlatformError as e:
        print(f"  mem_get_info 能力驱动降级: {e}")

    print(f"\n== 4. SFT 训练 {args.steps} 步（固定 batch 过拟合；"
          f"第 {args.resume_at} 步存 checkpoint）==")
    torch.manual_seed(42)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses_a = train_loop(model, opt, batches, dev, args.resume_at, fixed=True)
    tt_checkpoint.save(args.ckpt, model, opt, step=args.resume_at)
    rest = args.steps - args.resume_at
    losses_b = train_loop(model, opt, batches, dev, rest, start=0, fixed=True)
    losses = losses_a + losses_b
    for i, v in enumerate(losses):
        if i % 3 == 0 or i == len(losses) - 1:
            print(f"  step {i:>3} loss={v:.4f}")
    # 判据：同一 batch 上显著下降。轮换 batch 时首尾比大小只反映样本难度
    # 差异（实测 4 步轮换会得到 1.84 -> 2.38 的"上升"假象）
    func_ok = (all(math.isfinite(x) for x in losses)
               and losses[-1] < losses[0] * 0.9)
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"同批次收敛（末值<初值x0.9）: {func_ok}")

    print(f"\n== 5. checkpoint resume：从 step {args.resume_at} 恢复重跑 ==")
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager").to(dev)
    if not args.no_provider:
        tt.provider_wrap(model2)   # resume 模型同样走 provider（轨迹一致）
    opt2 = torch.optim.AdamW(model2.parameters(), lr=args.lr)
    step = tt_checkpoint.load(args.ckpt, model2, opt2)
    losses_r = train_loop(model2, opt2, batches, dev, rest, start=0, fixed=True)
    # bf16 下逐位复现不现实，取与训练 loss 量级相称的容差
    tol = 1e-5 if dtype is torch.float32 else 5e-2
    ckpt_ok, max_diff = tt_checkpoint.verify_resume_continuity(
        losses_b, losses_r, tol=tol)
    print(f"  原轨迹 {[f'{x:.4f}' for x in losses_b]}")
    print(f"  恢复后 {[f'{x:.4f}' for x in losses_r]}")
    print(f"  loss 连续性: max_diff={max_diff:.3e}（阈值 {tol:.0e}）-> {ckpt_ok}")

    print("\n== 6. checkpoint 负向：结构指纹不符必须拒绝 ==")
    guard_ok = False
    try:
        model2.model.embed_tokens.weight = torch.nn.Parameter(
            model2.model.embed_tokens.weight[:-1].clone())  # 改 shape
        tt_checkpoint.load(args.ckpt, model2, opt2)
        print("  意外成功（不应发生）")
    except RuntimeError as e:
        guard_ok = "指纹不符" in str(e)
        print(f"  fail fast（符合预期）: {str(e)[:96]}")

    print("\n== 7. 中转证据：运行期真的经过 TT 门面（非仅符号指向）==")
    trace_ok, trace_rows = tt.call_trace_report()
    for name, n, callers, external in trace_rows:
        tag = "框架侧" if external else "smoke 自身"
        print(f"  {name:16} 调用 {n:>4} 次  [{tag}] 调用方={(external or callers)[:3]}")
    print(f"  框架侧调用经 TT 中转: {trace_ok}"
          f"（若为 False 说明框架绕过了门面，绑定形同虚设）")

    tt.finalize()
    if os.path.exists(args.ckpt):
        os.remove(args.ckpt)

    print("\n== 判定 ==")
    checks = {"admission": n_fail == 0, "data_loss_mask": data_ok,
              "functional": func_ok, "ckpt_resume": ckpt_ok,
              "ckpt_schema_guard": guard_ok, "tt_in_path": trace_ok}
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"QWEN_SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
