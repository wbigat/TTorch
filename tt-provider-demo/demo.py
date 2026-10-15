"""真实 RMSNorm provider 端到端 demo（KL3 / xpu-masquerade），跑在 ttorch.provider 上。

演示两层 provider 结构：第一层模块级 Provider（RMSNormProvider）持有 native
（torch 原生单算子）+ 各硬件高性能 impl（KL3 厂商 AOT 融合）；执行时按 platform
auto_detect 结果在该 provider 内选实现，经三级门禁 -> selection -> 真实调用 ->
回退 -> 统一 provenance（bridge 绑定 + provider 生效合并成一份报告）。

运行：  cd tt-provider-demo && python demo.py
"""
import os
import sys
import time

import torch

import ttorch
from ttorch.platform import manager as platform_manager
from ttorch.witness import provenance
from ttorch.provider import manager as provider_manager
from ttorch.provider.spec import ProviderImpl

DEV = "cuda"
S, D = 4096, 4096
_EPS = 1e-6


# ----------------------------------------------------------------------
# 两个"反面候选"（教学夹具，演示门禁把关；不进生产 impls 文件夹）
# ----------------------------------------------------------------------
class SloppyRMSNorm(ProviderImpl):
    """精度不达标：功能能跑、形状对，但中间累加被截断（模拟低精度融合）。"""
    name = "sloppy:rmsnorm"
    kind = "fused_jit"
    intent = "反面候选：精度超阈值，应被精度门禁拒绝"

    def __call__(self, x, w, eps=_EPS):
        ms = x.pow(2).mean(-1, keepdim=True)
        ms = (ms * 10).round() / 10
        return x * torch.rsqrt(ms + eps) * w


class NoBenefitRMSNorm(ProviderImpl):
    """无收益：数值正确但冗余重复计算（模拟"融合了但没快"）。"""
    name = "no-benefit:rmsnorm"
    kind = "fused_jit"
    min_speedup = 1.05
    intent = "反面候选：数值对但无收益，应被收益门禁拒绝"

    def __call__(self, x, w, eps=_EPS):
        out = None
        for _ in range(5):
            out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        return out


def _fresh():
    torch.manual_seed(0)
    return torch.randn(S, D, device=DEV), torch.ones(D, device=DEV)


def _bench_call(semantic, reps=200):
    """经 provider call() 路径跑 reps 次，测稳态吞吐。"""
    x, w = _fresh()
    for _ in range(10):
        provider_manager.call(semantic, x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        provider_manager.call(semantic, x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    checks = []
    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))

    # 0. 激活（ttorch.activate 内自动注册全部 provider + 接线平台探测）
    ttorch.activate()
    print(f"== 0. 平台激活（provider 已自动注册）==\n  platform={platform_manager.activation().name}")

    # 反面候选追加到已自动注册的 rmsnorm provider（教学夹具，演示门禁把关）
    provider_manager.register_impl("rmsnorm", SloppyRMSNorm())
    provider_manager.register_impl("rmsnorm", NoBenefitRMSNorm())
    provider = provider_manager.instance().registry.get("rmsnorm")

    # 1. selection：按平台在 provider 内选 impl，三级门禁 + kind 优先级
    print("== 1. selection（功能/精度/收益门禁 + kind 优先级）==")
    summ = provider_manager.select("rmsnorm")
    print(f"  selected={summ['selected']}  speedup={summ['selected_speedup']}x")
    for name, reason in summ["rejected"]:
        print(f"  rejected: {name}  ({reason})")
    check("选中厂商融合实现", summ["selected"] == "vendor-fused:rmsnorm", summ["selected"])
    check("精度不达标候选被精度门禁拒绝",
          any("sloppy" in n and "precision" in r for n, r in summ["rejected"]))
    check("无收益候选被收益门禁拒绝",
          any("no-benefit" in n and "no_benefit" in r for n, r in summ["rejected"]))

    # 2. 精度对账：选中实现 vs native
    print("== 2. 精度对账 ==")
    x, w = _fresh()
    out = provider_manager.call("rmsnorm", x, w)
    ref = provider.native(x, w)
    err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, rtol=5e-3, atol=5e-3)
    print(f"  vendor_fused vs native max_abs_err={err:.2e}")
    check("选中实现精度达标（<5e-3）", ok, f"max_abs_err={err:.2e}")

    # 3. 端到端收益：provider 路径 vs 纯 native 基线（kill switch）
    print("== 3. 端到端收益 ==")
    t_opt = _bench_call("rmsnorm")
    provider_manager.disable("vendor-fused:rmsnorm")   # kill switch：回退 native
    t_base = _bench_call("rmsnorm")
    provider_manager.enable("vendor-fused:rmsnorm")    # 恢复
    speedup = t_base / t_opt
    print(f"  native 基线 {t_base:.3f}ms -> vendor_fused {t_opt:.3f}ms，端到端 {speedup:.2f}x")
    check("端到端有真实收益（>1.3x）", speedup > 1.3, f"{speedup:.2f}x")

    # 4. 回退正确性：禁用 vendor 后输出仍等于 native（功能无损）
    provider_manager.disable("vendor-fused:rmsnorm")
    out2 = provider_manager.call("rmsnorm", x, w)
    provider_manager.enable("vendor-fused:rmsnorm")
    check("kill switch 后回退 native 且功能无损",
          torch.allclose(out2, ref, rtol=5e-3, atol=5e-3))

    # 5. 统一 provenance：bridge 绑定 + provider 生效合并成一份报告
    print("== 4. 统一 provenance（bridge + provider 一份事件流）==")
    print(provenance.report())
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance.jsonl")
    provenance.export_jsonl(out_path)
    layers = {r["layer"] for r in provenance.records()}
    events = {r["event"] for r in provenance.records()}
    print(f"  已导出 {out_path}（{len(provenance.records())} 条；layer={sorted(layers)}，"
          f"provider event={sorted(e for e in events if e != 'bind')}）")
    check("统一 provenance 同时覆盖 bridge(bind) 与 provider(gate/select/call/switch)",
          {"bridge", "provider"} <= layers
          and {"gate", "select", "call", "switch"} <= events)

    # 判定
    print("== 判定 ==")
    n_fail = checks.count(False)
    print("RMSNORM_PROVIDER " + ("FAIL" if n_fail else "PASS"))
    platform_manager.finalize()
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
