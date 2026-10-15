"""ProviderManager：module 级性能实现的门禁 / 选择 / 分发 / 开关（运行态）。

层级：registry 注册第一层 Provider（一个模块语义一个）；执行时按 platform
auto_detect 结果在该 provider 内选第二层实现（各硬件高性能 impl），无该平台
实现或门禁不过 -> 回退 native（torch 原生单算子流程）。

职责（docs/TT_Provider_Function_Map.md §2）与四条不变式：
1. Provider 缺失只降性能，不降功能——每个 provider 自带 native，永远可用、
   永远启用、不可禁用；
2. 任何高性能 impl 失败 -> 回退 native，无失败模式（inapplicable / error 两级）；
3. 选择发生在首次调用期，写入绑定缓存后热路径零判断；
4. 无收益证据不启用——门禁第三级（probe 加速比）兜底。

分层（对齐 platform 的 registry/manager 拆分）：
- registry.py  ProviderRegistry：只管第一层 Provider 的注册存储；
- 本模块       ProviderManager：管运行态——门禁缓存、绑定缓存、开关状态、
               选择、分发；provenance 写入统一事件流（witness.provenance），
               精度/性能测量复用 witness.verify。

依赖方向：provider -> witness（记录+测量）、provider -> platform（只读激活平台名，
经 attach_platform 注入）单向；绝不碰 cuda-compat。
"""
import torch

from ttorch.witness import provenance, verify
from ttorch.provider.registry import ProviderRegistry

# kind 优先级：数值越大越优先。厂商 AOT 通常收益最高；native(reference) 是底线。
_KIND_PRIORITY = {"reference": 0, "fused_jit": 1, "fused_aot_abo": 2,
                  "fused_aot_vendor": 3}

# 精度门禁阈值：abs/rel err < 5e-3（与跨硬件精度验证口径一致；测量在 witness.verify）
_RTOL = 5e-3
_ATOL = 5e-3


__all__ = [
    "ProviderManager",
    "activate", "register", "register_impl", "select", "call", "wrap_model",
    "enable", "disable", "set_bypass", "status", "attach_platform", "instance",
]


class ProviderManager:
    """module 级性能实现的运行态管理。可独立实例化（测试隔离）。"""

    def __init__(self, registry=None):
        self.registry = registry if registry is not None else ProviderRegistry()
        self._enabled = {}     # impl name -> bool（switch 状态，缺省 True）
        self._gate_cache = {}  # impl name -> (ok, reason, evidence) 门禁裁决缓存
        self._binding = {}     # (semantic, slot) -> (provider, 选定的 impl)
        self._platform_probe = None  # attach_platform() 注入：() -> 当前平台名
        self._call_count = {}    # impl name -> 成功调用次数（热路径观测，轻量）
        self._called_once = set()  # 已记录"首次成功调用"provenance 的 impl
        self._bypass = False     # 全局旁路：True 时被 wrap 的模块走原始实现（A-B 对比/退场）

    def set_bypass(self, on):
        """全局旁路开关：on=True 时所有被 wrap 的模块走原始实现（不经 provider）。

        用于 A-B 性能对比（baseline vs provider）与紧急整体退场（区别于按 impl
        的 kill switch——那是一实现一开关，在 provider 内回退 native）。
        """
        self._bypass = bool(on)
        provenance.record(provenance.LAYER_PROVIDER, "switch", "*", 
                          "bypass" if on else "active")

    def attach_platform(self, probe):
        """注入当前平台名来源（生产：lambda: platform_manager.activation().name）。"""
        self._platform_probe = probe

    def _platform(self):
        return self._platform_probe() if self._platform_probe else None

    # --------------------------------------------------------------
    # registry（登记第一层 Provider；其 impl 运行态默认启用）
    # --------------------------------------------------------------
    def register(self, provider):
        self.registry.register(provider)
        self._enabled[provider.native.name] = True
        for impl in provider.impls:
            self._enabled[impl.name] = True
        return provider

    def register_impl(self, semantic, impl):
        """往已注册的 provider 追加一个候选 impl（扩展/教学用），默认启用。"""
        provider = self.registry.get(semantic)
        if provider is None:
            raise KeyError(f"provider '{semantic}' 未注册，可选: {self.registry.names()}")
        provider.add_impl(impl)
        self._enabled[impl.name] = True
        return impl

    # --------------------------------------------------------------
    # 自动注册 + 自动生效
    # --------------------------------------------------------------
    def activate(self):
        """provider 层激活：注册全部已知 provider + 默认接线到 platform manager。

        由 ttorch.activate() 在 bridge 激活后调用（此时平台已 auto_detect）。
        幂等——重复调用不重复注册。注册 eager（import 安全、便宜）；门禁/选择
        惰性（首次 call 才探测，不为没用到的 provider 白跑）。
        """
        from ttorch.platform import manager as platform_manager
        from ttorch.provider import impls
        registered = []
        for provider in impls.build_all():
            if self.registry.get(provider.semantic) is None:
                self.register(provider)
                registered.append(provider.semantic)
        if self._platform_probe is None:
            self.attach_platform(lambda: platform_manager.activation().name)
        # 启动期 provenance：可复现快照（平台 + 能力 + 版本 + 注册了哪些 provider）。
        # §4"可复现"硬要求——另一次运行凭这份快照重放同一套选择。
        if registered:
            act = platform_manager.activation()
            provenance.record(provenance.LAYER_PROVIDER, "session", "provider",
                              "activated", platform=act.name, vendor=act.vendor_name,
                              device_name=act.device_name,
                              capabilities=act.capabilities(),
                              torch_version=torch.__version__, registered=registered)

    def wrap_model(self, model):
        """自动发现模型里匹配已注册 provider 的模块，patch 其 forward 走 provider
        （自动生效）。幂等（已 patch 的模块跳过）。返回 patch 的模块数。

        匹配规则：模块类名 endswith 某 provider 的 module_names。weight 在
        调用期现读（见 Provider.wrap_module），故 .to(device)/训练更新后仍正确。
        """
        providers = [p for p in self.registry.all() if p.module_names]
        n = 0
        wrapped = []
        for module in model.modules():
            if getattr(module, "_tt_provider_wrapped", False):
                continue
            cls_name = type(module).__name__
            for p in providers:
                if any(cls_name == mn or cls_name.endswith(mn) for mn in p.module_names):
                    try:
                        module.forward = p.wrap_module(module)
                    except TypeError:
                        break  # 无 weight Parameter 等，跳过该模块
                    module._tt_provider_wrapped = True
                    wrapped.append(f"{cls_name}->{p.semantic}")
                    n += 1
                    break
        if n:
            provenance.record(provenance.LAYER_PROVIDER, "wrap",
                              type(model).__name__, "wrapped",
                              count=n, modules=wrapped)
        return n

    def _find_impl(self, name):
        """按 impl 名定位 (provider, impl)（含 native）。"""
        for provider in self.registry.all():
            if provider.native.name == name:
                return provider, provider.native
            for impl in provider.impls:
                if impl.name == name:
                    return provider, impl
        return None, None

    # --------------------------------------------------------------
    # gate：功能 & 精度 & 收益三级准入（候选 vs 该 provider 的 native）
    # --------------------------------------------------------------
    def _gate(self, provider, impl):
        """对一个高性能 impl 跑三级准入（功能/精度/收益的测量本体在 witness.verify）。

        返回 (ok, reason, evidence)。选择缓存与 provenance 记录在本层，测量复用
        witness.verify.verify_impl——验证（测量）与选择（编排）解耦。
        """
        return verify.verify_impl(impl, provider.native, provider.probe_input,
                                  min_speedup=impl.min_speedup,
                                  enforce_benefit=(impl.kind != "reference"),
                                  rtol=_RTOL, atol=_ATOL)

    def _gate_cached(self, provider, impl):
        if impl.name not in self._gate_cache:
            ok, reason, evidence = self._gate(provider, impl)
            self._gate_cache[impl.name] = (ok, reason, evidence)
            provenance.record(provenance.LAYER_PROVIDER, "gate", impl.name,
                              "pass" if ok else "fail",
                              semantic=provider.semantic, slot=provider.slot,
                              reason=reason, build=impl.build, version=impl.version,
                              wraps=list(impl.wraps), **evidence)
        return self._gate_cache[impl.name]

    # --------------------------------------------------------------
    # selection：按平台在该 provider 内选 impl，门禁 + 优先级，回退 native
    # --------------------------------------------------------------
    def select(self, semantic, slot="compute"):
        """为一个模块语义选定实现。返回摘要（同时写 provenance）。"""
        provider = self.registry.get(semantic)
        if provider is None:
            raise KeyError(f"provider '{semantic}' 未注册，可选: {self.registry.names()}")
        platform = self._platform()
        native = provider.native
        rejected = []
        gated = []
        for impl in provider.candidates_for(platform):
            if not self._enabled.get(impl.name, True):
                rejected.append((impl.name, "disabled_by_switch"))
                continue
            ok, reason, _ = self._gate_cached(provider, impl)
            if not ok:
                rejected.append((impl.name, reason))
                continue
            gated.append(impl)

        chosen = max(gated, key=lambda i: _KIND_PRIORITY.get(i.kind, 0)) if gated else native
        self._binding[(semantic, slot)] = (provider, chosen)
        speedup = None
        if chosen is not native and chosen.name in self._gate_cache:
            speedup = self._gate_cache[chosen.name][2].get("speedup")
        summary = {"semantic": semantic, "slot": slot, "selected": chosen.name,
                   "selected_build": chosen.build, "selected_speedup": speedup,
                   "rejected": rejected}
        provenance.record(provenance.LAYER_PROVIDER, "select", f"{semantic}/{slot}",
                          "selected", platform=platform, selected=chosen.name,
                          selected_build=chosen.build, selected_version=chosen.version,
                          selected_speedup=speedup, rejected=rejected)
        return summary

    # --------------------------------------------------------------
    # switch：运行时启用/禁用（native 不可禁用）
    # --------------------------------------------------------------
    def enable(self, name):
        self._set_switch(name, True)

    def disable(self, name):
        self._set_switch(name, False)

    def _set_switch(self, name, on):
        provider, impl = self._find_impl(name)
        if impl is None:
            raise KeyError(f"impl '{name}' 未注册")
        if impl is provider.native and not on:
            raise ValueError(f"{name} 是 native（功能底线），不可禁用")
        self._enabled[name] = on
        self._binding.pop((provider.semantic, provider.slot), None)  # 失效绑定，下次 call 重选
        provenance.record(provenance.LAYER_PROVIDER, "switch", name,
                          "enabled" if on else "disabled")

    # --------------------------------------------------------------
    # 分发与回退
    # --------------------------------------------------------------
    def _args_sig(self, args, kw):
        """参数签名：shape/dtype/flags 级信息，不记数据本体（观测用）。"""
        def s(v):
            if isinstance(v, torch.Tensor):
                return f"{tuple(v.shape)}/{v.dtype}"
            if callable(v):
                return getattr(v, "__name__", "fn")
            return repr(v)
        return [s(v) for v in args] + [f"{k}={s(v)}" for k, v in kw.items()]

    def call(self, semantic, *args, slot="compute", **kw):
        """经绑定缓存分发；两级回退（inapplicable / error）都只回退 native。

        热路径对齐 torch 原生调用开销：**不逐调用同步、不逐条记录**——成功调用
        只累加计数，首次成功记一条 provenance（证明分发到了哪个 impl）；回退
        （罕见）才逐条记录。设备同步只在门禁探测（_probe_us）做，那里才是测
        吞吐的地方。
        """
        key = (semantic, slot)
        if key not in self._binding:
            self.select(semantic, slot)
        provider, impl = self._binding[key]
        native = provider.native
        subject = f"{semantic}/{slot}"

        # 回退 1：逐调用适用性。supports 不满足不是错误——融合 kernel 只覆盖
        # 特定 shape/dtype，超出范围安静回退 native。
        if not impl.supports(*args, **kw):
            provenance.record(provenance.LAYER_PROVIDER, "call", subject, "fallback",
                              impl=impl.name, args_sig=self._args_sig(args, kw),
                              fallback_actual=native.name, reason="inapplicable")
            return native(*args, **kw)

        try:
            out = impl(*args, **kw)
        except Exception as e:
            # 回退 2：运行时失败。native 自身失败是功能问题，不在 Provider 范畴。
            if impl is native:
                raise
            provenance.record(provenance.LAYER_PROVIDER, "call", subject, "fallback",
                              impl=impl.name, args_sig=self._args_sig(args, kw),
                              fallback_actual=native.name, error=repr(e))
            return native(*args, **kw)

        self._call_count[impl.name] = self._call_count.get(impl.name, 0) + 1
        if impl.name not in self._called_once:
            self._called_once.add(impl.name)
            provenance.record(provenance.LAYER_PROVIDER, "call", subject, "called",
                              impl=impl.name, args_sig=self._args_sig(args, kw))
        return out

    def status(self):
        """当前全部 provider 的启用状态与绑定缓存（观测性入口）。"""
        return {"platform": self._platform(),
                "providers": {p.semantic: {
                    "native": p.native.name,
                    "impls": {i.name: {"kind": i.kind, "enabled": self._enabled.get(i.name, True),
                                       "platforms": (sorted(i.platforms) if i.platforms else "all")}
                              for i in p.impls}}
                    for p in self.registry.all()},
                "bindings": {f"{s}/{sl}": impl.name for (s, sl), (pr, impl) in self._binding.items()},
                "calls": dict(self._call_count)}


# ----------------------------------------------------------------------
# 进程级单例 + 模块函数（常规使用方共享一个 manager；测试可另建实例隔离）
# ----------------------------------------------------------------------
_instance = ProviderManager()


def instance():
    return _instance


def attach_platform(probe):
    return _instance.attach_platform(probe)


def activate():
    """provider 层激活（自动注册全部 provider + 接线平台）。ttorch.activate() 调用。"""
    return _instance.activate()


def wrap_model(model):
    """自动发现并 patch 模型里匹配已注册 provider 的模块（自动生效）。返回 patch 数。"""
    return _instance.wrap_model(model)


def register(provider):
    return _instance.register(provider)


def register_impl(semantic, impl):
    return _instance.register_impl(semantic, impl)


def set_bypass(on):
    """全局旁路开关：on=True 时所有被 wrap 的模块走原始实现（A-B 对比/整体退场）。"""
    return _instance.set_bypass(on)


def select(semantic, slot="compute"):
    return _instance.select(semantic, slot)


def call(semantic, *args, slot="compute", **kw):
    return _instance.call(semantic, *args, slot=slot, **kw)


def enable(name):
    return _instance.enable(name)


def disable(name):
    return _instance.disable(name)


def status():
    return _instance.status()
