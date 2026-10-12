"""真实环境 LLM-SFT smoke：Bridge 激活 -> torch.cuda 绑定 -> 功能 + 精度验收。

用法：
  python real_sft_smoke.py                    # 自动检测 cuda / xpu-masquerade / xpu-native
  python real_sft_smoke.py --platform cpu     # 本地 dev 验证（无加速卡机器）
  python real_sft_smoke.py --steps 50

判定标准（退出码反映）：
  绑定: 全部 admission 条目 PASS
  功能: fp32 SFT loss 末值 < 初值 x 0.5；bf16 完跑且 loss 全部有限且下降
  精度: 加速卡 fp32 与 CPU fp32 参考的 eval logits rel_err < 5e-3
        （跨硬件精度口径；bf16 vs fp32 的 rel_err 只打印不卡阈值）

真实 XPU 环境待验证项（本地 CPU 已验证路径之外）：
  - masquerade 探测的卡名标记是否命中真实 XPU 卡（ adapters_real._XPU_MARKERS ）
  - native xpu 栈的 mem_get_info/get_rng_state 是否存在（缺失自动降级，看日志）
  - SDK 版本协商范围（TorchCudaPlatform.sdk_range 对定制栈可能需放宽）
"""
import argparse
import math
import sys

import torch

import framework_real
from tt_bridge import bindings_torch, cuda_compat, platform_mgr
from tt_bridge.adapters_real import (TorchCpuPlatform, TorchCudaPlatform,
                                     TorchXpuMasqueradePlatform,
                                     TorchXpuNativePlatform)


def rel_err(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="auto",
                    choices=["auto", "cpu", "cuda", "xpu-masquerade", "xpu-native"])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    for cls in (TorchCudaPlatform, TorchXpuMasqueradePlatform,
                TorchXpuNativePlatform, TorchCpuPlatform):
        platform_mgr.register(cls)

    print(f"[env] torch={torch.__version__}")
    if args.platform == "auto":
        platform_mgr.auto_activate()
    else:
        platform_mgr.select(args.platform)
    print(f"[env] vendor={platform_mgr.vendor_name()} "
          f"device_name={platform_mgr.device_name()} "
          f"device_spec={platform_mgr.device_spec()}")

    print("\n== 1. cuda-compat 绑定真实 torch.cuda（扫尾白名单=framework_real）==")
    # 平台 overlay：torch.cuda 为非功能桩的栈（cpu-dev）剔除能力类绑定，
    # 避免 is_available 的"真"回答误导 torch 内部守卫（见 bindings_torch.entries）
    exclude_cap = platform_mgr.device_name() == "cpu"
    entries = bindings_torch.entries(exclude_capability=exclude_cap)
    cuda_compat.activate(entries, default_deny=False,
                         sweep_packages=["framework_real"])
    if exclude_cap:
        print("  平台 overlay：cpu-dev 剔除能力类绑定，"
              "is_available 语义由 platform_mgr.capability 直接供给")

    print("\n== 2. admission：绑定正确性 ==")
    ctx = {"vendor_cuda": torch.cuda}
    n_fail = 0
    for e in entries:
        ok, detail = e.test(ctx)
        print(f"  [{'PASS' if ok else 'FAIL'}] {e.test.__name__}: {detail}")
        n_fail += 0 if ok else 1

    dev = torch.device(platform_mgr.device_spec())
    print(f"\n== 3. 功能验收：fp32 SFT（device={dev}）==")
    losses, logits_fp32, mem_info, n_params = framework_real.sft_run(
        dev, args.steps, "fp32")
    print(f"  参数量={n_params}  mem_get_info={mem_info}")
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    func_ok = losses[-1] < losses[0] * 0.5 and all(math.isfinite(x) for x in losses)
    print(f"  loss 下降达标（末值<初值x0.5）: {func_ok}")

    print("\n== 4. 功能验收：bf16 autocast 完跑 ==")
    losses_bf16, logits_bf16, _, _ = framework_real.sft_run(dev, args.steps, "bf16")
    bf16_ok = (all(math.isfinite(x) for x in losses_bf16)
               and losses_bf16[-1] < losses_bf16[0])
    print(f"  loss: {losses_bf16[0]:.4f} -> {losses_bf16[-1]:.4f}  有限且下降: {bf16_ok}")

    print("\n== 5. 精度验收：accelerator fp32 vs CPU fp32 参考 ==")
    ref_losses, ref_logits, _, _ = framework_real.sft_run(
        torch.device("cpu"), args.steps, "fp32", bridged=False)
    err_fp32 = rel_err(logits_fp32, ref_logits)
    err_bf16 = rel_err(logits_bf16, ref_logits)
    print(f"  fp32 rel_err={err_fp32:.3e}（阈值 5e-3）  bf16 rel_err={err_bf16:.3e}（参考值）")
    prec_ok = err_fp32 < 5e-3

    platform_mgr.finalize()

    print("\n== 判定 ==")
    checks = {"admission": n_fail == 0, "functional_fp32": func_ok,
              "functional_bf16": bf16_ok, "precision_fp32": prec_ok}
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
