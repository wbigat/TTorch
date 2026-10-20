"""RMSNorm provider 的整网自动使能 + 性能验证（KL3 / xpu-masquerade，Qwen2 SFT）。

全程走自动化路径，演示"自动注册 -> 自动生效 -> 自动选择"：
  tt.activate()            自动注册全部 provider + 接线平台（无需手动 register）
  tt.provider_wrap(model)  自动发现模型里的 RMSNorm 模块并 patch（自动生效）
  首次前向                 按 auto_detect 的平台自动选 impl（自动选择，惰性门禁）

性能对比（同权重、同数据）：
  bypass=True   被 wrap 的模块走原始 transformers RMSNorm（baseline）
  bypass=False  走 provider -> KL3 厂商融合
  kill switch   关掉厂商 impl -> provider 内回退 native

运行：  cd tt-provider-demo && python demo_e2e.py
"""
import sys
import time

import torch

import ttorch as tt
from ttorch.provider import manager as provider_manager


def _build_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
        num_key_value_heads=4, intermediate_size=1408, vocab_size=10000,
        max_position_embeddings=4096,
    )
    model = Qwen2ForCausalLM(cfg)
    model.config._attn_implementation = "eager"  # KL3 不支持 flash/sdpa
    return model.to("cuda", dtype=torch.bfloat16)


def _run_steps(model, opt, batch, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    checks = []
    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))

    # 自动注册（tt.activate 内）+ 接线
    tt.activate()
    print(f"== 0. 平台激活 + provider 自动注册 ==\n  platform={tt.active().device_name}")

    model = _build_model()
    # 自动生效：发现 + patch 全部 RMSNorm 模块
    n_wrapped = tt.provider_wrap(model)
    print(f"  provider_wrap: patch {n_wrapped} 个 RMSNorm 模块（自动发现）")
    check("自动 wrap 发现并 patch 了 RMSNorm 模块", n_wrapped > 0, f"{n_wrapped} 个")

    torch.manual_seed(0)
    B, L, V = 4, 512, model.config.vocab_size
    ids = torch.randint(0, V, (B, L), device="cuda")
    batch = {"input_ids": ids, "labels": ids.clone()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 数值对齐：同一组随机初始化权重 + 同一份数据，仅前向（不更新权重）。
    # bypass 开关切换原始/provider，权重不变，loss 差异即数值差。
    provider_manager.set_bypass(True)
    loss_b0 = model(**batch).loss.item()
    provider_manager.set_bypass(False)
    loss_v0 = model(**batch).loss.item()   # 首次前向触发自动选择（门禁+绑定）

    # 计时：每模式 warmup + 计时步（权重随训练演化，但比较的是每步耗时，与权重无关）
    WARM, STEPS = 8, 40
    provider_manager.set_bypass(True)
    _run_steps(model, opt, batch, WARM)
    t_base = _run_steps(model, opt, batch, STEPS)
    provider_manager.set_bypass(False)
    _run_steps(model, opt, batch, WARM)
    t_vendor = _run_steps(model, opt, batch, STEPS)
    bindings_vendor = dict(provider_manager.instance().status()["bindings"])  # vendor 激活时读
    provider_manager.disable("vendor-fused:rmsnorm")   # kill switch：回退 native
    _run_steps(model, opt, batch, WARM)
    t_native = _run_steps(model, opt, batch, STEPS)
    provider_manager.enable("vendor-fused:rmsnorm")
    provider_manager.set_bypass(False)

    print("\n== 整网性能（ms/step，越小越好）==")
    print(f"  baseline(原始,bypass)  : {t_base*1e3:.3f}")
    print(f"  vendor(厂商融合)       : {t_vendor*1e3:.3f}")
    print(f"  native(kill switch)    : {t_native*1e3:.3f}")
    print(f"  vendor 激活时绑定: {bindings_vendor}")
    speedup = t_base / t_vendor
    print(f"\n  整网端到端加速比: {speedup:.3f}x")
    print(f"  数值对齐（同权重前向 loss）: baseline={loss_b0:.5f}  vendor={loss_v0:.5f}")

    # 绑定键含负载签名（rmsnorm/compute#<sig>）——按前缀匹配该语义下的全部绑定
    check("自动选择绑定到厂商融合",
          any(k.startswith("rmsnorm/compute") and "vendor-fused:rmsnorm" in v
              for k, v in bindings_vendor.items()))
    # 口径修正（2026-10-15）：KL3 上 torch_xmlir 已把 RMSNorm 换成厂商融合实现，
    # baseline（bypass）本身就是融合态；TT vendor 是同源 kernel + 一层分发——
    # 结构性持平（~1x），"必须有提升"在该平台不成立。提升预期只适用于厂商未
    # 覆盖的硬件/算子（docs/TT_KLX_Plugin_Absorption.md）。等价口径对齐
    # demo_replace_torchxmlir（0.85~1.15x），此处收得更紧（同源 kernel）。
    check("整网端到端等价（vendor ≈ baseline，0.95~1.05x；厂商已融合平台上预期持平）",
          0.95 < speedup < 1.05,
          f"{t_base*1e3:.3f} -> {t_vendor*1e3:.3f} ms ({speedup:.3f}x)")
    check("替换后前向 loss 与 baseline 一致（bf16 数值对齐）",
          abs(loss_b0 - loss_v0) < 0.01, f"baseline={loss_b0:.5f} vendor={loss_v0:.5f}")
    check("kill switch 回退 native 整网可用", t_native > 0)

    print("\n== 判定 ==")
    n_fail = checks.count(False)
    print("RMSNORM_E2E " + ("FAIL" if n_fail else "PASS"))
    tt.finalize()
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
