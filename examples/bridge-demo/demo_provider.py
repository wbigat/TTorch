"""Provider 垂直切片 demo v4：schedule 插槽第二实例（TP 通算 overlap）。

v4 新增：tp_linear 语义的 schedule 完整实例——TP 通算 overlap（all-gather
通信异步化 + wgrad/dgrad 分解）。抽象为 (comm_thunks, compute_thunks)
时序安排：缺省串行 -> overlap provider 过门禁接管，结果逐位对账。

v3 新增（用户纠偏）：wrapper 按目的分属两层——cuda-compat 的 WRAP 是
正确性 delta（摘掉会错）；性能目的的 wrapper 属 Provider，作为 impl 的
构建策略 build=wrap_native：包装社区原生小算子链 + JIT 编译融合。
数值语义继承原生链，精度门禁天然对齐，收益门禁才是主过滤器。
build/wraps 进入 gate/select provenance。

v2 在 v1（registry/selection/门禁/switch/回退/provenance）之上补齐三个坑，
并以 fused RoPE（BSHD）为完整实例——一个"厂商融合 kernel + 裸替换 wrapper"
的真实组合：
- 厂商融合 kernel：仅 mscale==1.0 走融合路径，其余回退非融合
- 裸替换 wrapper：能 import 到融合实现就换——无门禁、无 provenance、无 switch

补齐的坑：
1. 逐调用适用性谓词 supports：mscale!=1.0 时融合 kernel 不适用，
   安静回退 reference（记 reason=inapplicable，区别于 error）
2. 真实收益证据：门禁第三级——probe 加速比（best-of-3），
   默认只记录，provider 声明 min_speedup 才强制拒绝（no_benefit）
3. torch.ops.tt.* 注册位：tt_ops 模块，框架只面向语义名编程
"""
from tt_bridge import manager
from tt_bridge.adapters import XPU_SPEC
from tt_provider import implementations as impl
from tt_provider import provenance, provider_mgr, tt_ops

# 独立实例隔离（真实平台已在 import 时自动注册进默认单例，见 demo_bridge 注释）
from ttorch.platform import PlatformManager, PlatformRegistry  # noqa: E402
manager.set_instance(PlatformManager(PlatformRegistry()))

X8 = [0.13, -1.27, 0.85, 0.44, -0.62, 0.99, -0.31, 0.71]
W8 = [1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 0.95]
T8 = [0.41, -0.97, 0.22, 1.13, -0.55, 0.68, -0.21, 0.94]
F8 = [0.01, 0.10, 0.35, 0.80, 1.30, 2.10, 3.20, 4.50]


def print_select(summary):
    sp = summary.get("selected_speedup")
    print(f"  selected: {summary['selected']} (build={summary.get('selected_build')})"
          + (f"（probe speedup={sp}x）" if sp else ""))
    for name, reason in summary["rejected"]:
        print(f"  rejected: {name:<28} <- {reason}")


print("== 0. Bridge 平台激活（Provider 消费 Platform Manager，单向只读）==")
manager.register(XPU_SPEC)
manager.auto_activate()
provider_mgr.attach_platform(lambda: manager.activation().name)

print("\n== 1. 无性能 Provider 基线：只有 reference（摘掉全部 Provider 的世界）==")
out = provider_mgr.call("rmsnorm", X8, W8)
assert out == impl.reference_rmsnorm(X8, W8)
print(f"  rmsnorm(len=8) 走 reference，结果正确: [{out[0]:.4f}, {out[1]:.4f}, ...]")

print("\n== 2. 注册 rmsnorm 性能 Provider + 三级门禁 + 选择 ==")
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-aot-vendor-xpu", kind="fused_aot_vendor", semantic="rmsnorm",
    slot="compute", impl=impl.fused_aot_vendor_rmsnorm, platforms=frozenset({"xpu"}),
    intent="厂商融合算子（AOT 黑盒），只支持 len=8"))
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-aot-vendor-cuda", kind="fused_aot_vendor", semantic="rmsnorm",
    slot="compute", impl=impl.fused_aot_vendor_cuda_rmsnorm, platforms=frozenset({"cuda"}),
    intent="厂商融合算子的 CUDA 版本"))
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-aot-abo", kind="fused_aot_abo", semantic="rmsnorm",
    slot="compute", impl=impl.fused_aot_abo_rmsnorm, platforms=None,
    intent="ABO 融合算子（图编译 AOT 产物），无平台限定"))
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-jit", kind="fused_jit", semantic="rmsnorm",
    slot="compute", impl=impl.fused_jit_rmsnorm, platforms=None,
    build="wrap_native",
    wraps=("aten::square", "aten::mean", "aten::rsqrt", "aten::mul"),
    intent="JIT 包装原生链但低精度编译（fp16 截断），应被精度门禁拒绝——"
           "wrap_native 的精度语义继承链，主动降精度是 provider 自己的责任"))
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-jit-wrap", kind="fused_jit", semantic="rmsnorm",
    slot="compute",
    impl=impl.jit_compile(impl._rmsnorm_native_chain, "jit_wrap_rmsnorm"),
    platforms=None, build="wrap_native",
    wraps=("aten::square", "aten::mean", "aten::rsqrt", "aten::mul", "aten::mul"),
    intent="JIT 包装社区原生小算子链（wrap_native 正例）：性能来自融合消物化，"
           "数值语义继承原生链——精度门禁天然过，收益门禁才是主过滤器"))
print_select(provider_mgr.select("rmsnorm"))

print("\n== 3. 调用（走选定的 vendor impl，结果对账 reference）==")
out = provider_mgr.call("rmsnorm", X8, W8)
assert out == impl.reference_rmsnorm(X8, W8)
print(f"  结果与 reference 逐位一致: [{out[0]:.4f}, {out[1]:.4f}, ...]")

print("\n== 4. switch：禁用 vendor -> 重选落到 ABO ==")
provider_mgr.disable("fused-aot-vendor-xpu")
print_select(provider_mgr.select("rmsnorm"))
out = provider_mgr.call("rmsnorm", X8, W8)
assert out == impl.reference_rmsnorm(X8, W8)
print("  禁用后功能不受影响（ABO 接管，结果仍与 reference 一致）")

print("\n  -- 4a. 再禁用 ABO -> wrap_native（JIT 包装原生小算子链）接管 --")
provider_mgr.disable("fused-aot-abo")
print_select(provider_mgr.select("rmsnorm"))
out = provider_mgr.call("rmsnorm", X8, W8)
assert out == impl.reference_rmsnorm(X8, W8)
print("  wrap_native 接管：数值语义继承原生链，结果逐位一致")
provider_mgr.enable("fused-aot-abo")

print("\n== 5. 运行时失败回退：vendor 不支持 len=16 -> reference ==")
provider_mgr.enable("fused-aot-vendor-xpu")
provider_mgr.select("rmsnorm")  # 重新选中 vendor
x16, w16 = X8 + X8, W8 + W8
out = provider_mgr.call("rmsnorm", x16, w16)
assert out == impl.reference_rmsnorm(x16, w16)
print("  vendor 抛 XpuHwError(unsupported shape) -> 已回退 reference，结果正确")
print("  （不 fail fast：性能增量失败不是功能事故）")

print("\n== 6. schedule 插槽：缺省 serial -> overlap provider 接管 ==")
r = provider_mgr.call("moe_block", list(impl.MOE_THUNKS), slot="schedule")
print(f"  无 schedule provider 时缺省 serial: {r}")
provider_mgr.provider_register(provider_mgr.Provider(
    name="schedule-overlap", kind="schedule", semantic="moe_block",
    slot="schedule", impl=impl.overlap_schedule, platforms=None,
    intent="expert 计算全部异步下发后统一等待（模拟通算 overlap）"))
print_select(provider_mgr.select("moe_block", slot="schedule"))
r2 = provider_mgr.call("moe_block", list(impl.MOE_THUNKS), slot="schedule")
assert r2 == r == [2.0, 4.0, 6.0]
print(f"  overlap 接管后结果完全一致: {r2}（schedule 只改时序不改语义）")

print("\n  -- 6a. TP 通算 overlap（all-gather 异步化 + wgrad/dgrad 分解）--")
r = provider_mgr.call("tp_linear", list(impl.TP_COMM_THUNKS),
                      list(impl.TP_COMPUTE_THUNKS), slot="schedule")
print(f"  无 provider 时缺省串行（通信完再计算）: {r}")
provider_mgr.provider_register(provider_mgr.Provider(
    name="schedule-tp-overlap", kind="schedule", semantic="tp_linear",
    slot="schedule", impl=impl.overlap_tp_schedule, platforms=None,
    intent="通信异步下发到 comm stream、计算先行、末尾统一等通信——"
           "对照真实代码的 all-gather/matmul 交叠与 wgrad/dgrad 分解"))
print_select(provider_mgr.select("tp_linear", slot="schedule"))
r2 = provider_mgr.call("tp_linear", list(impl.TP_COMM_THUNKS),
                       list(impl.TP_COMPUTE_THUNKS), slot="schedule")
assert r2 == r == [1.5, 2.5]
print(f"  overlap 接管后结果完全一致: {r2}（只改时序不改语义）")

print("\n== 7. 完整实例：fused RoPE（BSHD）==")
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-aot-vendor-rope", kind="fused_aot_vendor", semantic="rope_bshd",
    slot="compute", impl=impl.fused_aot_vendor_rope_bshd, platforms=frozenset({"xpu"}),
    supports=lambda t, freqs, mscale=1.0: mscale == 1.0,
    intent="厂商 fused RoPE（torch.ops.custom_ops.rotary_pos_emb 的 mock）；"
           "supports 谓词替代真实代码里 impl 内部的 mscale==1.0 分支"))
provider_mgr.provider_register(provider_mgr.Provider(
    name="fused-jit-rope", kind="fused_jit", semantic="rope_bshd",
    slot="compute", impl=impl.fused_jit_rope_bshd, platforms=None, min_speedup=1.0,
    build="wrap_native", wraps=("megatron::rope_unfused",),
    intent="JIT 包装 Megatron unfused 分解；数值正确但无收益，"
           "应被收益门禁拒绝（no_benefit）"))
print_select(provider_mgr.select("rope_bshd"))

print("\n  -- 7a. mscale=1.0：适用，走厂商融合 kernel --")
out = provider_mgr.call("rope_bshd", T8, F8, mscale=1.0)
assert out == impl.reference_rope_bshd(T8, F8, 1.0)
print(f"  结果与 reference 逐位一致: [{out[0]:.4f}, {out[1]:.4f}, ...]")

print("\n  -- 7b. mscale=1.2：supports 不满足，安静回退 reference --")
out = provider_mgr.call("rope_bshd", T8, F8, mscale=1.2)
assert out == impl.reference_rope_bshd(T8, F8, 1.2)
print(f"  回退 reference（inapplicable，非 error），结果正确: [{out[0]:.4f}, {out[1]:.4f}, ...]")

print("\n== 8. torch.ops.tt.* 注册位：框架只面向语义名编程 ==")
out1 = tt_ops.rmsnorm(X8, W8)
out2 = tt_ops.rope_bshd(T8, F8)
assert out1 == impl.reference_rmsnorm(X8, W8)
assert out2 == impl.reference_rope_bshd(T8, F8, 1.0)
print("  tt_ops.rmsnorm(x, w) / tt_ops.rope_bshd(t, freqs) 正常返回")
print("  框架侧不感知 provider 名字、不知道选中谁、不知道是否回退")

print("\n== 9. 红线：reference 不可被 switch 禁用 ==")
try:
    provider_mgr.disable("reference:rmsnorm")
    print("  意外成功（不应发生）")
except ValueError as e:
    print(f"  拒绝（符合预期）: {e}")

print("\n== 10. 状态观测 + provenance 全量 ==")
st = provider_mgr.status()
for name, info in st["providers"].items():
    print(f"  provider {name:<28} kind={info['kind']:<16} enabled={info['enabled']}")
print(f"  bindings: {st['bindings']}")
print()
print(f"  {'seq':<4}{'event':<7}{'关键内容'}")
for r in provenance.RECORDS:
    keep = {k: v for k, v in r.items() if k not in ("seq", "ts", "event")}
    print(f"  {r['seq']:<4}{r['event']:<7}{keep}")
import os
snap = os.path.join(os.path.dirname(__file__), "provenance.jsonl")
provenance.export_jsonl(snap)
print(f"\n  已导出 {os.path.basename(snap)}（{len(provenance.RECORDS)} 条）")

print("\n== 11. 生命周期收尾 ==")
manager.finalize()
