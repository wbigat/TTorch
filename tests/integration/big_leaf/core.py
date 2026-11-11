"""big_leaf C1–C11 单卡正确性集成测试；性能矩阵不属于本套件。"""
import os

os.environ.setdefault("XTE_GROUPED_GEMM_LARGE_WEIGHT", "0")
os.environ.setdefault("XTE_USE_GROUPED_FC_FUSION", "0")
os.environ.setdefault("XMLIR_ENABLE_FAST_FC", "1")

import torch
import torch_xmlir  # noqa: F401
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from ttorch.bridge.big_leaf import apply_big_leaf

dt = torch.bfloat16
RESULTS = {}


def check(name, ok):
    RESULTS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def build(G, fin, fout, bias, lw1=False, seed=0):
    if lw1:
        os.environ["XTE_GROUPED_GEMM_LARGE_WEIGHT"] = "1"
    torch.manual_seed(seed)
    m = GroupedLinear(G, fin, fout, bias=bias, params_dtype=dt, device="cuda")
    os.environ["XTE_GROUPED_GEMM_LARGE_WEIGHT"] = "0"
    return m


def repack(module):
    G = module.num_gemms
    ws = [getattr(module, f"weight{i}") for i in range(G)]
    buf = torch.empty(G, *ws[0].shape, device=ws[0].device, dtype=ws[0].dtype)
    with torch.no_grad():
        for i, w in enumerate(ws):
            buf[i].copy_(w)
            w.data = buf[i]
    if module.use_bias:
        bs = [getattr(module, f"bias{i}") for i in range(G)]
        bbuf = torch.empty(G, *bs[0].shape, device=bs[0].device, dtype=bs[0].dtype)
        with torch.no_grad():
            for i, b in enumerate(bs):
                bbuf[i].copy_(b)
                b.data = bbuf[i]
        return buf, bbuf
    return buf, None


def make_big_leaf(module, buf, bbuf):
    big = buf.detach().requires_grad_(True)
    module.large_weight = big
    if bbuf is not None:
        module.large_bias = bbuf.detach().requires_grad_(True)
    else:
        module.large_bias = None  # switch 分支无条件 getattr，占位
    module.large_weight_switch = True
    return ([big, module.large_bias] if bbuf is not None else [big])


def grads_of(model, G, bias):
    out = [getattr(model, f"weight{i}").grad for i in range(G)]
    if bias:
        out += [getattr(model, f"bias{i}").grad for i in range(G)]
    return out


def zero_grads(model, G, bias):
    for i in range(G):
        getattr(model, f"weight{i}").grad = None
        if bias:
            getattr(model, f"bias{i}").grad = None


def run_pair(G, FIN, FOUT, TOK, bias, tag, steps=3):
    """stock ref vs big_leaf test：fwd/xgrad/wgrad + steps 步轨迹全逐位。"""
    ref = build(G, FIN, FOUT, bias)
    test = build(G, FIN, FOUT, bias)
    with torch.no_grad():
        for p_r, p_t in zip(ref.parameters(), test.parameters()):
            p_t.copy_(p_r)
    buf, bbuf = repack(test)
    leafs = make_big_leaf(test, buf, bbuf)
    torch.manual_seed(7)
    x1 = torch.randn(TOK, FIN, device="cuda", dtype=dt, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    ms = [TOK // G] * G
    o1, o2 = ref(x1, ms), test(x2, ms)
    check(f"{tag}: fwd bitwise", torch.equal(o1, o2))
    g = torch.randn_like(o1)
    o1.backward(g)
    o2.backward(g)
    check(f"{tag}: xgrad bitwise", torch.equal(x1.grad, x2.grad))
    wg = all(torch.equal(leafs[0].grad[i], getattr(ref, f"weight{i}").grad)
             for i in range(G))
    check(f"{tag}: wgrad per-slice bitwise", wg)
    if bias:
        bg = all(torch.equal(leafs[1].grad[i], getattr(ref, f"bias{i}").grad)
                 for i in range(G))
        check(f"{tag}: bgrad per-slice bitwise", bg)
    # 轨迹
    optT = torch.optim.AdamW(leafs, lr=1e-3)
    optR = torch.optim.AdamW(list(ref.parameters()), lr=1e-3)
    g_fixed = torch.randn_like(o1.detach())
    ok = True
    for s in range(steps):
        optT.step()
        optR.step()
        ok &= all(torch.equal(getattr(test, f"weight{i}").detach(),
                              getattr(ref, f"weight{i}").detach()) for i in range(G))
        if s < steps - 1:
            for lf in leafs:
                lf.grad = None
            zero_grads(ref, G, bias)
            ref(x1, ms).backward(g_fixed)
            test(x2, ms).backward(g_fixed)
    check(f"{tag}: {steps}-step trajectory bitwise", ok)
    return ref, test, leafs, x1, x2


print("===== C1/C2: bias × shape 矩阵 =====")
run_pair(8, 2048, 1536, 8192, True, "C1a bias=True G8 2048->1536")
run_pair(8, 2048, 1536, 8192, False, "C1b bias=False G8 2048->1536")
run_pair(4, 512, 384, 2048, True, "C2a bias=True G4 512->384")
run_pair(4, 512, 384, 2048, False, "C2b bias=False G4 512->384")

print("===== C3: state_dict 往返 =====")
ref, test, leafs, x1, x2 = None, None, None, None, None
G, FIN, FOUT, TOK, bias = 8, 512, 384, 2048, False
ref, test, leafs, x1, x2 = run_pair(G, FIN, FOUT, TOK, bias, "C3-base", steps=1)
sd = test.state_dict()
fresh = build(G, FIN, FOUT, bias)
fresh.load_state_dict(sd)
ms = [TOK // G] * G
with torch.no_grad():
    o_fresh = fresh(x1.detach(), ms)
    o_ref = ref(x1.detach(), ms)
check("C3: state_dict roundtrip fwd bitwise", torch.equal(o_fresh, o_ref))
check("C3: state_dict keys unchanged vs ref",
      set(sd.keys()) == set(ref.state_dict().keys()))

print("===== C4: eval/train 混用 =====")
with torch.no_grad():
    _ = test(x2.detach(), ms)  # eval 前向
o_t = test(x2, ms)  # train 前向
o_t.square().mean().backward()
check("C4: eval->train 后 big_leaf.grad 正常", all(lf.grad is not None for lf in leafs))
o_r = ref(x1, ms)
o_r.square().mean().backward()
check("C4: eval/train 混用后 grad 逐位", torch.equal(leafs[0].grad[0],
                                                     ref.weight0.grad))

print("===== C5: 梯度累积（2 microbatch）=====")
for lf in leafs:
    lf.grad = None
zero_grads(ref, G, bias)
for mb in range(2):
    torch.manual_seed(100 + mb)
    xa = torch.randn(TOK, FIN, device="cuda", dtype=dt, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    ref(xa, ms).square().mean().backward()
    test(xb, ms).square().mean().backward()
check("C5: 梯度累积后 big_leaf.grad 逐位",
      all(torch.equal(leafs[0].grad[i], getattr(ref, f"weight{i}").grad)
          for i in range(G)))

print("===== C6: clip_grad_norm_ 语义等价 =====")
gn_test = torch.nn.utils.clip_grad_norm_(leafs, max_norm=1.0)
gn_ref = torch.nn.utils.clip_grad_norm_(list(ref.parameters()), max_norm=1.0)
check("C6: 合并范数 == 分立范数", torch.equal(gn_test, gn_ref))
check("C6: clip 后 grad 逐位",
      all(torch.equal(leafs[0].grad[i], getattr(ref, f"weight{i}").grad)
          for i in range(G)))

print("===== C7: 三种 optimizer 轨迹对拍 =====")
from apex.optimizers import FusedAdam

for impl, tag in [({"foreach": True}, "foreach"), ({"fused": True}, "torch-fused")]:
    r = build(G, FIN, FOUT, bias)
    t = build(G, FIN, FOUT, bias)
    with torch.no_grad():
        for pr, pt in zip(r.parameters(), t.parameters()):
            pt.copy_(pr)
    buf, bbuf = repack(t)
    lfs = make_big_leaf(t, buf, bbuf)
    torch.manual_seed(7)
    xa = torch.randn(TOK, FIN, device="cuda", dtype=dt, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    oA, oB = r(xa, ms), t(xb, ms)
    gf = torch.randn_like(oA.detach())
    oA.backward(gf)
    oB = t(xb, ms)  # 重算 test 前向（消耗同值）
    oB.backward(gf)
    oR = torch.optim.AdamW(list(r.parameters()), lr=1e-3, **impl)
    oT = torch.optim.AdamW(lfs, lr=1e-3, **impl)
    ok = True
    for s in range(3):
        oR.step()
        oT.step()
        ok &= all(torch.equal(getattr(t, f"weight{i}").detach(),
                              getattr(r, f"weight{i}").detach()) for i in range(G))
        if s < 2:
            for lf in lfs:
                lf.grad = None
            zero_grads(r, G, bias)
            r(xa, ms).backward(gf)
            t(xb, ms).backward(gf)
    check(f"C7-{tag}: 3-step trajectory bitwise", ok)

# apex FusedAdam 对拍（同实现两侧）
r = build(G, FIN, FOUT, bias)
t = build(G, FIN, FOUT, bias)
with torch.no_grad():
    for pr, pt in zip(r.parameters(), t.parameters()):
        pt.copy_(pr)
buf, bbuf = repack(t)
lfs = make_big_leaf(t, buf, bbuf)
torch.manual_seed(7)
xa = torch.randn(TOK, FIN, device="cuda", dtype=dt, requires_grad=True)
xb = xa.detach().clone().requires_grad_(True)
r(xa, ms).square().mean().backward()
t(xb, ms).square().mean().backward()
oR = FusedAdam(list(r.parameters()), lr=1e-3, adam_w_mode=True)
oT = FusedAdam(lfs, lr=1e-3, adam_w_mode=True)
ok = True
for s in range(3):
    oR.step()
    oT.step()
    ok &= all(torch.equal(getattr(t, f"weight{i}").detach(),
                          getattr(r, f"weight{i}").detach()) for i in range(G))
    if s < 2:
        for lf in lfs:
            lf.grad = None
        zero_grads(r, G, bias)
        r(xa, ms).square().mean().backward()
        t(xb, ms).square().mean().backward()
check("C7-apex: 3-step trajectory bitwise", ok)

print("===== C8: restore 回滚保留训练更新 + 幂等保护 =====")

G8, FIN8, FOUT8, TOK8 = 8, 512, 384, 2048
m8 = build(G8, FIN8, FOUT8, False)
plan8 = apply_big_leaf(m8)
torch.manual_seed(7)
x8 = torch.randn(TOK8, FIN8, device="cuda", dtype=dt, requires_grad=True)
ms8 = [TOK8 // G8] * G8
opt8 = torch.optim.AdamW(plan8.leafs, lr=1e-3)
m8(x8, ms8).square().mean().backward()
opt8.step()  # 训练一步，buf 已更新
trained = [getattr(m8, f"weight{i}").detach().clone() for i in range(G8)]
plan8.restore()
restored_eq = all(torch.equal(getattr(m8, f"weight{i}").detach(), trained[i])
                  for i in range(G8))
check("C8a: restore 后训练更新保留（不丢）", restored_eq)
with torch.no_grad():
    o_restored = m8(x8.detach(), ms8)
o_expect = None
with torch.no_grad():  # 参考：同值权重 stock 路径前向
    ref8 = build(G8, FIN8, FOUT8, False)
    for pr, pt in zip(ref8.parameters(), m8.parameters()):
        pr.copy_(pt)
    o_expect = ref8(x8.detach(), ms8)
check("C8b: restore 后前向与 stock 逐位", torch.equal(o_restored, o_expect))
m8b = build(G8, FIN8, FOUT8, False)
apply_big_leaf(m8b)
try:
    apply_big_leaf(m8b)  # 未 restore 直接重复 apply → 必须拒绝
    check("C8c: 重复 apply 拒绝", False)
except ValueError:
    check("C8c: 重复 apply 拒绝", True)
# restore 后可再次 apply（先 restore 的合法路径）
plan8b = apply_big_leaf(m8)
check("C8d: restore 后可重新 apply", len(plan8b.leafs) == 1)
plan8b.restore()

print("===== C9/C10: 显存中性 + named_leafs + restore 后 optimizer 重建 =====")
import dataclasses as _dc

m9 = build(G8, FIN8, FOUT8, True)  # 有 bias 形态覆盖
torch.cuda.empty_cache()
mem0 = torch.cuda.memory_allocated()
plan9 = apply_big_leaf(m9)
torch.cuda.empty_cache()
mem1 = torch.cuda.memory_allocated()
# 显存中性：不持有旧 storage（D1 修复）→ apply 后常驻增量 ≈ 0
check("C9: apply 后常驻显存增量 < 1MiB（不持有旧 storage）",
      abs(mem1 - mem0) < 2**20)
nl = plan9.named_leafs(m9)
check("C10a: named_leafs 数量与命名", len(nl) == 2 and
      all("large_weight" in n or "large_bias" in n for n, _ in nl))
# restore 后 optimizer 重建路径（restore 使旧 leaf 失效 → 用 model.parameters()）
opt9 = torch.optim.AdamW(plan9.leafs, lr=1e-3)
x9 = torch.randn(TOK8, FIN8, device="cuda", dtype=dt, requires_grad=True)
ms9 = [TOK8 // G8] * G8
m9(x9, ms9).square().mean().backward()
opt9.step()
# 实际篡改 bias ptr，确认 healthy 不是常量断言。
m9.bias0.data = m9.bias0.data.clone()
try:
    plan9.assert_healthy()
    check("C10c: bias ptr 篡改被 healthy 检出", False)
except RuntimeError as e:
    check("C10c: bias ptr 篡改被 healthy 检出", "bias0" in str(e))
m9.bias0.data = plan9.bindings[0].bbuf[0]
plan9.assert_healthy()
plan9.restore()
opt9b = torch.optim.AdamW(m9.parameters(), lr=1e-3)  # stock 参数列表
m9(x9, ms9).square().mean().backward()
opt9b.step()
check("C10b: restore 后 stock optimizer 正常续训",
      all(p.grad is not None for p in m9.parameters()))

print("===== C11: 冻结权重（GRPO ref model 形态）=====")
m11 = build(G8, FIN8, FOUT8, False)
for p in m11.parameters():
    p.requires_grad_(False)  # 全部冻结
plan11 = apply_big_leaf(m11)
check("C11a: 冻结权重 big_leaf.requires_grad=False（不强制可训）",
      not plan11.leafs[0].requires_grad)
plan11.assert_healthy()  # 冻结形态下断言不误报
check("C11b: 冻结形态 healthy 断言通过", True)
x11 = torch.randn(TOK8, FIN8, device="cuda", dtype=dt, requires_grad=True)
ms11 = [TOK8 // G8] * G8
o11 = m11(x11, ms11)
o11.square().mean().backward()
check("C11c: 冻结形态 big_leaf.grad 为 None（不累积）",
      plan11.leafs[0].grad is None)
check("C11d: 冻结形态输入梯度正常", x11.grad is not None)
# 与 stock 冻结参考逐位
ref11 = build(G8, FIN8, FOUT8, False)
with torch.no_grad():
    for pr, pt in zip(ref11.parameters(), m11.parameters()):
        pr.copy_(pt)
x11r = x11.detach().clone().requires_grad_(True)
o11r = ref11(x11r, ms11)
o11r.square().mean().backward()
check("C11e: 冻结形态 fwd 与 stock 逐位", torch.equal(o11, o11r))

all_pass = all(RESULTS.values())
print(f"\n正确性 {sum(RESULTS.values())}/{len(RESULTS)} PASS")
print("VERDICT:", "ALL PASS" if all_pass else "SOME FAILED")
raise SystemExit(0 if all_pass else 1)
