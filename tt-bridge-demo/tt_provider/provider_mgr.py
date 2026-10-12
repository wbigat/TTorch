"""Provider Manager：所有 provider 类型的 registry 与 selection。

职责（对照定稿）：
- registry    provider_register() 逻辑注册，只登记元数据，不执行
- selection   select()：平台过滤 → switch 过滤 → 功能&精度&收益准入门禁 →
              按 kind 优先级选定，写入绑定缓存（热路径零判断）
- gate        每个 provider 的准入测试，三级：
              1. 功能：能跑、输出形状/类型与 reference 一致
              2. 精度：对 reference abs/rel err < 5e-3
              3. 收益：probe 加速比 >= provider 声明的 min_speedup
                 （Provider 是性能增量，无收益即无存在意义；
                 默认 0 = 只记录证据不强制，阈值由注册方显式声明）
- switch      enable()/disable() 运行时控制每个 provider 启用状态，
              禁用即退出后续 selection；reference 是功能底线，不可禁用
- 分发与回退  call() 走绑定缓存；两级回退都不 fail fast：
              1. supports 谓词不满足（如 mscale!=1.0）-> 回退 reference，
                 记 reason=inapplicable——不是错误，是适用范围外
              2. impl 抛异常 -> 回退 reference，记 fallback_actual+error

Manager 不 import Bridge：当前平台名通过 attach_platform() 注入，
依赖方向保持 provider -> bridge 单向（且只读 Platform Manager）。
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tt_provider import provenance

# kind 优先级：数值越大越优先。compute 插槽内 厂商 AOT > ABO > JIT > reference
# （厂商 AOT 通常收益最高）；schedule 插槽内任何过门禁的调度 provider 都优先于
# reference（serial 只是缺省兜底，有过门禁的优化就该接管）。
_KIND_PRIORITY = {"reference": 0, "fused_jit": 1, "fused_aot_abo": 2,
                  "fused_aot_vendor": 3, "schedule": 1}

# 精度门禁阈值：与跨硬件精度验证口径一致（bf16/fp16 abs/rel err < 5e-3）
_RTOL = 5e-3
_ATOL = 5e-3

# probe 计时取 best-of-N，抑制单次抖动
_PROBE_REPS = 3


@dataclass(frozen=True)
class Provider:
    """一个性能实现的元数据。impl/supports 是真实对象引用，拼错 import 期即炸。

    platforms=None 表示全平台；否则仅列出的平台名可参与 selection。
    supports 是逐调用适用性谓词（与 impl 同签名）：真实融合 kernel 常有
    适用条件（如 fused RoPE 仅 mscale==1.0），不满足时 call() 回退
    reference，记 inapplicable 而非 error。
    min_speedup 是收益门禁阈值：probe 加速比低于它则拒绝注册生效；
    默认 0 表示收益只作为 provenance 证据记录、不强制。

    build 是 impl 的构建策略（形态维度，与 kind 正交）：
    - self_contained  自带完整 kernel（厂商 AOT 黑盒）——门禁重点在精度
    - wrap_native     包装社区原生小算子链 + 编译融合——数值语义继承
                      原生链，精度门禁天然对齐，收益门禁才是主过滤器；
                      wraps 声明被包装的链（provenance 可查）
    """
    name: str
    kind: str            # reference | fused_aot_vendor | fused_aot_abo | fused_jit | schedule
    semantic: str        # 模块语义，如 "rmsnorm" / "rope_bshd" / "moe_block"
    slot: str            # compute | schedule
    impl: Callable
    platforms: Optional[frozenset] = None
    supports: Optional[Callable] = None
    min_speedup: float = 0.0
    build: str = "self_contained"  # self_contained | wrap_native
    wraps: tuple = ()    # build=wrap_native 时被包装的原生小算子链
    intent: str = ""


_PROVIDERS = {}     # name -> Provider
_ENABLED = {}       # name -> bool（switch 状态，缺省 True）
_GATE_CACHE = {}    # name -> (ok, reason, evidence) 门禁裁决缓存
_BINDING = {}       # (semantic, slot) -> Provider 选择期写入的绑定缓存
_platform_probe = None  # attach_platform() 注入：() -> 当前平台名


def attach_platform(probe):
    """注入平台名来源（demo: lambda: platform_mgr.active().name）。"""
    global _platform_probe
    _platform_probe = probe


def _platform():
    return _platform_probe() if _platform_probe else None


def provider_register(p: Provider) -> Provider:
    """逻辑注册一个 provider：只登记元数据。重名立即报错。"""
    assert p.name not in _PROVIDERS, f"provider 重复注册: {p.name}"
    _PROVIDERS[p.name] = p
    _ENABLED[p.name] = True
    return p


def _reference(semantic, slot):
    """(semantic, slot) 的 reference：精度基准 + 收益基准 + 回退终点。"""
    for p in _PROVIDERS.values():
        if p.semantic == semantic and p.slot == slot and p.kind == "reference":
            return p
    return None


# ------------------------------------------------------------------
# gate：功能 & 精度 & 收益准入测试
# ------------------------------------------------------------------

def _close(a, b):
    """逐元素判定 |a-b| <= atol + rtol*|b|（b 为 reference 输出）。"""
    return all(abs(x - y) <= _ATOL + _RTOL * abs(y) for x, y in zip(a, b))


def _probe_us(fn, args, kw):
    """best-of-N 探测耗时（us）。"""
    best = float("inf")
    for _ in range(_PROBE_REPS):
        t0 = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def _gate(p):
    """对一个 provider 跑三级准入测试。返回 (ok, reason, evidence)。

    evidence 含 probe_us / ref_us / speedup——收益证据在门禁期采集，
    即使不做强制拒绝（min_speedup=0）也进入 provenance。
    """
    from tt_provider.implementations import PROBE  # 延迟 import 避免环
    ref = _reference(p.semantic, p.slot)
    if ref is None:
        return False, "gate: 无 reference 可作基准", {}
    args, kw = PROBE[p.semantic]
    ref_out = ref.impl(*args, **kw)

    # 1. 功能准入：能跑、输出形状/类型与 reference 一致
    try:
        out = p.impl(*args, **kw)
    except Exception as e:
        return False, f"functional: 探测输入上抛异常 {e!r}", {}
    if type(out) is not type(ref_out) or len(out) != len(ref_out):
        return False, (f"functional: 输出形状/类型不符 "
                       f"(expect {type(ref_out).__name__}[{len(ref_out)}])"), {}

    # 2. 精度准入：对 reference abs/rel err < 5e-3
    if not _close(out, ref_out):
        err = max(abs(x - y) / max(abs(y), 1e-12) for x, y in zip(out, ref_out))
        return False, f"precision: max_rel_err={err:.2e} 超阈值 {_RTOL:.0e}", {}

    # 3. 收益证据：probe 加速比；声明了 min_speedup 才强制执行
    probe_us = _probe_us(p.impl, args, kw)
    ref_us = _probe_us(ref.impl, args, kw)
    speedup = ref_us / probe_us if probe_us > 0 else float("inf")
    evidence = {"probe_us": round(probe_us, 1), "ref_us": round(ref_us, 1),
                "speedup": round(speedup, 2)}
    if p.kind != "reference" and speedup < p.min_speedup:
        return False, (f"no_benefit: speedup={speedup:.2f}x "
                       f"< min_speedup={p.min_speedup}x"), evidence
    return True, "pass", evidence


def _gate_cached(p):
    if p.name not in _GATE_CACHE:
        ok, reason, evidence = _gate(p)
        _GATE_CACHE[p.name] = (ok, reason, evidence)
        provenance.record("gate", provider=p.name, semantic=p.semantic,
                          slot=p.slot, ok=ok, reason=reason,
                          build=p.build, wraps=list(p.wraps), **evidence)
    return _GATE_CACHE[p.name]


# ------------------------------------------------------------------
# selection：平台过滤 -> switch 过滤 -> 门禁 -> 优先级选定 -> 绑定缓存
# ------------------------------------------------------------------

def select(semantic, slot="compute"):
    """为一个 (semantic, slot) 选定实现。返回摘要（同时写 provenance）。"""
    platform = _platform()
    rejected = []
    gated = []
    for p in _PROVIDERS.values():
        if p.semantic != semantic or p.slot != slot:
            continue
        if p.platforms is not None and platform not in p.platforms:
            rejected.append((p.name, f"platform_unsupported: 仅 {sorted(p.platforms)}，当前 {platform}"))
            continue
        if not _ENABLED[p.name]:
            rejected.append((p.name, "disabled_by_switch"))
            continue
        ok, reason, evidence = _gate_cached(p)
        if not ok:
            rejected.append((p.name, reason))
            continue
        gated.append((p, evidence))

    # schedule 插槽缺省 serial：它以 reference kind 注册，必然在 gated 里
    if not gated:
        raise RuntimeError(f"{semantic}/{slot}: 无可选实现（reference 缺失？）")
    chosen, ev = max(gated, key=lambda kv: _KIND_PRIORITY.get(kv[0].kind, 0))
    _BINDING[(semantic, slot)] = chosen
    summary = {"semantic": semantic, "slot": slot, "selected": chosen.name,
               "selected_build": chosen.build,
               "selected_speedup": ev.get("speedup"), "rejected": rejected}
    provenance.record("select", platform=platform, **summary)
    return summary


def enable(name):
    _set_switch(name, True)


def disable(name):
    _set_switch(name, False)


def _set_switch(name, on):
    """switch：运行时控制启用状态。变更后失效相关绑定，下次 call 重选。

    reference 不可禁用——它是功能底线，禁用它等于把 Provider 层
    从"性能增量"变成"功能依赖"，违背定位。
    """
    p = _PROVIDERS.get(name)
    if p is None:
        raise KeyError(f"provider '{name}' 未注册，可选: {sorted(_PROVIDERS)}")
    if p.kind == "reference" and not on:
        raise ValueError(f"{name} 是 reference（功能底线），不可禁用")
    _ENABLED[name] = on
    _BINDING.pop((p.semantic, p.slot), None)
    provenance.record("switch", provider=name, enabled=on)


# ------------------------------------------------------------------
# 分发与回退
# ------------------------------------------------------------------

def _args_sig(args, kw):
    """参数签名：shape/flags 级信息，不记数据本体。"""
    def s(v):
        if isinstance(v, (list, tuple)):
            return f"len={len(v)}"
        if callable(v):
            return getattr(v, "__name__", "fn")
        return repr(v)
    return [s(v) for v in args] + [f"{k}={s(v)}" for k, v in kw.items()]


def call(semantic, *args, slot="compute", **kw):
    """经绑定缓存分发；两级回退（inapplicable / error）都只回退 reference。"""
    key = (semantic, slot)
    if key not in _BINDING:
        select(semantic, slot)
    p = _BINDING[key]
    ref = _reference(semantic, slot)
    sig = _args_sig(args, kw)

    # 回退 1：逐调用适用性。supports 不满足不是错误——如 fused RoPE
    # 只覆盖 mscale==1.0，超出范围安静回退 reference。
    if p.supports is not None and ref is not None and not p.supports(*args, **kw):
        out = ref.impl(*args, **kw)
        provenance.record("call", semantic=semantic, slot=slot, impl=p.name,
                          args_sig=sig, fallback_actual=ref.name,
                          reason="inapplicable")
        return out

    t0 = time.perf_counter()
    try:
        out = p.impl(*args, **kw)
    except Exception as e:
        # 回退 2：运行时失败。reference 自身失败是功能问题，不在 Provider 范畴。
        if ref is None or p.impl is ref.impl:
            raise
        out = ref.impl(*args, **kw)
        provenance.record("call", semantic=semantic, slot=slot, impl=p.name,
                          args_sig=sig, duration_us=round((time.perf_counter() - t0) * 1e6, 1),
                          fallback_actual=ref.name, error=repr(e))
        return out
    provenance.record("call", semantic=semantic, slot=slot, impl=p.name,
                      args_sig=sig, duration_us=round((time.perf_counter() - t0) * 1e6, 1),
                      fallback_actual=None)
    return out


def status():
    """当前全部 provider 的启用状态与绑定缓存（观测性入口）。"""
    return {"providers": {n: {"kind": p.kind, "enabled": _ENABLED[n]}
                          for n, p in _PROVIDERS.items()},
            "bindings": {f"{s}/{sl}": p.name for (s, sl), p in _BINDING.items()}}
