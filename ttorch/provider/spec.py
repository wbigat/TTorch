"""provider 的两层抽象：Provider（模块级）+ ProviderImpl（平台级实现）。

层级（docs/TT_Provider_Function_Map.md §3 的分叉轴 = 模块语义 × 硬件）：

  第一层  Provider       一个模块语义一个类型（RMSNormProvider / RoPEProvider…）。
                        持有 native 默认实现 + 各硬件高性能实现；定义该语义的
                        调用约定（__call__ 签名）与门禁探测输入（probe_input）。
  第二层  ProviderImpl   provider 内部、面向某平台的一份高性能实现
                        （KL3FusedRMSNorm / CudaFusedRMSNorm…）。每个硬件重写
                        自己的一份；platforms 标记它适用的平台。

注册单元是第一层的 Provider（按 semantic 注册进 ProviderRegistry）；执行时按
platform auto_detect 结果在该 provider 内选实现，无该平台实现或门禁不过 ->
回退 native（torch 原生单算子流程，功能底线）。

一种模块语义一个文件夹（impls/<语义>/）：base.py 放 Provider 类型 + native，
<平台>.py 放各硬件的 ProviderImpl。厂商算子引用一律写在 __call__ 方法体内，
保持非该硬件机器 import 安全。
"""
from abc import ABC
from typing import Optional


class ProviderImpl(ABC):
    """第二层：provider 内部、面向某（些）平台的一份高性能实现。

    类属性即元数据（拼错在 import 期即暴露）；实现本体是 __call__（子类重写）。
    """

    name = ""                  # 唯一名（switch/观测用），子类必设
    platforms: Optional[frozenset] = None  # None=全平台通用；否则仅列出的平台可被选
    kind = "fused_aot_vendor"  # reference | fused_aot_vendor | fused_aot_abo | fused_jit
    version = ""               # 实现版本（provenance 可复现/回归基线用）
    min_speedup = 0.0          # 收益门禁阈值；0=收益只作证据记录、不强制
    build = "self_contained"   # self_contained(厂商黑盒) | wrap_native(包装原生链融合)
    wraps: tuple = ()          # build=wrap_native 时被包装的原生小算子链
    intent = ""                # 为什么这样实现（没有 intent 半年后没人懂）

    def supports(self, *args, **kw):
        """逐调用适用性谓词（与 __call__ 同签名）。融合 kernel 常有适用条件
        （只覆盖某 shape/dtype）；不满足时 call() 回退 native（不是错误）。默认恒适用。"""
        return True

    def __call__(self, *args, **kw):
        """实现本体（子类重写；厂商算子引用写在方法体内保持 import 安全）。"""
        raise NotImplementedError


class ProviderModuleForward:
    """wrap_module 的产物：模块级可调用对象，替代原先的内部闭包 forward。

    为什么是模块级类而非闭包：闭包/局部函数不可 pickle——`torch.save(model)`
    整模型序列化（EMA、部分框架习惯）会炸
    `Can't get local object 'wrap_module.<locals>.forward'`。模块级类的实例
    可 pickle/deepcopy；持有 module 形成循环引用，pickle memo 机制可处理。

    weight/eps 调用期现读（不经构造期固化），.to(device)/训练更新后仍正确。

    **延迟自检（2026-10-23，D14 时机契约消解）**：wrap 期只挂成员 + 替换 forward
    （零前向、零 dtype 改动——零侵入，故可在并行包装**之后**安全调用）；精度
    兜底延迟到**首次 __call__**——此时经 __call__ 流程，FSDP/DDP 的 all-gather
    hook 已执行、weight 完整。自检未过则恢复原生 forward 回退（功能不降级）。
    """

    def __init__(self, module, semantic, slot, provider):
        self._module = module
        self._orig_forward = module.forward  # bypass 回退 / A-B 对比基线 / 延迟自检基准
        self._semantic = semantic
        self._slot = slot
        self._provider = provider  # 延迟自检用（首次 __call__ 调其 _self_check）
        self._checked = None       # None=待自检; True=通过; False=已回退原生

    def __call__(self, x, *args, **kw):
        from ttorch.provider import manager as _pm  # 延迟 import 避免环
        m = _pm.instance()
        if m.is_bypassed():  # 全局旁路：走原始实现（baseline）
            return self._orig_forward(x, *args, **kw)
        if self._checked is None:
            # 延迟自检：首次真实调用时做（真实输入 x，经 __call__ weight 完整）。
            eps0 = getattr(self._module, "variance_epsilon",
                           getattr(self._module, "eps", 1e-6))
            self._checked = self._provider._self_check(
                self._module, eps0, x=x, ref_forward=self._orig_forward)
            if not self._checked:
                # 自检未过：恢复原生 forward，本模块不接管（功能不降级）。
                self._module.forward = self._orig_forward
                return self._orig_forward(x, *args, **kw)
        if not self._checked:
            return self._orig_forward(x, *args, **kw)
        module = self._module
        # 传原始 weight——scale 语义由 provider 的 native/impl 内部固定
        # （标准 provider 用 w 直乘；zcg provider 内部做 1+w）。一对一，
        # 不在此处按 flag 折 scale。
        w = module.weight
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        return m.call(self._semantic, x, w, eps, slot=self._slot)


class Provider(ABC):
    """第一层：一个模块语义的高性能 provider（注册单元）。

    子类按模块语义定义（semantic + probe_input + build_native）。native 是
    torch 原生单算子流程（kind="reference"），默认自带、永远启用、不可禁用——
    摘掉全部高性能实现后的功能底线 + 精度/收益基准。各硬件的高性能实现经
    add_impl 挂到 impls，执行时按平台选。
    """

    semantic = ""              # 模块语义，如 "rmsnorm" / "rope_bshd"
    slot = "compute"           # compute（schedule 插槽本期挂起）
    module_names: tuple = ()   # 模块级接入：nn.Module 类名（endswith 匹配），自动发现用
    zero_centered_gamma = False  # 本 provider 处理的 RMSNorm 变体：False=w 直乘（标准），
        # True=1+w（zero_centered_gamma，zcg）。一对一——每种变体一个 provider，
        # wrap_module 据此 flag 做轻量分流（精度兜底在首次 __call__ 延迟复检）。
    bind_points: tuple = ()    # 函数级接入：((模块路径, 符号名), ...)——框架内部的
        # 函数符号（如 megatron 的 swiglu）。绑定后符号 = dispatcher，调用经
        # manager.call 分发——与模块级共享门禁/回退/开关/provenance。
        # （第三级接入形态"编译级"= kind="fused_jit" 的 ProviderImpl，预留）

    def __init__(self):
        self.native = self.build_native()
        assert self.native is not None and self.native.kind == "reference", \
            f"{type(self).__name__}.build_native() 必须返回 kind='reference' 的 native 实现"
        self.impls = []        # 各平台高性能实现（ProviderImpl）

    def build_native(self):
        """默认 native 实现（torch 原生单算子流程）。子类必须返回。"""
        raise NotImplementedError

    def add_impl(self, impl):
        """挂载一份平台高性能实现（返回 impl 便于链式）。"""
        self.impls.append(impl)
        return impl

    def candidates_for(self, platform):
        """当前平台可候选的高性能实现（不含 native——native 是独立回退底线）。"""
        return [i for i in self.impls
                if i.platforms is None or platform in i.platforms]

    @classmethod
    def probe_input(cls, ref=None, device=None):
        """语义专属探测输入（真实设备张量，(args, kwargs)）。子类必须实现。

        ref：首次真实调用的 (args, kwargs)——探测输入跟随生产负载的
        shape/dtype/device/标量参数（完整复制非张量参数与 kwargs，eps 这类
        参数丢了门禁结论就不对应生产）。device：无 ref 时由 manager 经
        handle.device_spec() 注入（不硬编码 "cuda"）。不定全局 RNG 种子
        （门禁在训练循环中途触发，manual_seed 会污染训练 RNG）。
        """
        raise NotImplementedError

    def wrap_module(self, module):
        """把一个匹配的 nn.Module 包成走 provider.call 的 forward（自动生效）。

        默认按 RMSNorm 类约定从模块提取 weight（Parameter）与 eps
        （variance_epsilon/eps），语义特殊的子类可重写。返回的 forward 每次
        调用经 manager.call 分发——享受门禁选择、两级回退与 provenance。
        weight/eps 在调用期现读（不经构造期固化），保证 .to(device)/训练更新
        后仍正确。

        返回 None = 本 provider 变体与模块不符（由对应变体 provider 接管）。
        抛 TypeError = 模块形态不符该语义（无 weight Parameter）。

        **延迟自检（2026-10-23，D14 时机消解）**：wrap 期**零侵入**——不跑前向、
        不改 dtype，只挂成员 + 替换 forward（故可在并行包装**之后**安全调用）。
        精度兜底延迟到首次 __call__（经 __call__ 流程，FSDP/DDP all-gather 后
        weight 完整），未过则恢复原生 forward（见 ProviderModuleForward）。
        """
        import torch
        if not isinstance(getattr(module, "weight", None), torch.nn.Parameter):
            raise TypeError(f"{type(module).__name__} 无可包装的 weight Parameter")
        # 轻量分流（flag 先验，不跑前向）：本 provider 的变体语义与模块的
        # zero_centered_gamma flag 不符则不接管（由对应变体 provider 接管）。
        # flag 可能丢/伪造——精度兜底在首次 __call__ 延迟复检（纠正 flag 错误）。
        if getattr(module, "zero_centered_gamma", False) != self.zero_centered_gamma:
            return None
        return ProviderModuleForward(module, self.semantic, self.slot, self)

    def _self_check(self, module, eps, x=None, ref_forward=None, admit_tol=None):
        """精度兜底：本 provider 固定语义的实现 vs 模块参考 forward 一致性。

        **定位（2026-10-23 修正）**：provider ↔ 社区标准 module **一对一**——语义
        在 provider 设计时确定（本 provider 的 native/impl 固定一种 scale 语义），
        self_check **不做变体选择**，只验证"本 provider 实现 vs 模块真实 forward"
        的精度一致性：一致才接管，不一致拒绝（兜底）。语义不匹配的模块由对应
        变体的独立 provider 接管（zcg 由 ZcgRMSNormProvider，见 rmsnorm/zcg_base.py）。
        此前把"变体识别（w vs 1+w 选择题）"塞进自检，是把本应在 provider 设计时
        定的语义选择错误下沉到运行时——已去除。

        **dtype 自适应阈值（取代 to(float32) 侵入性转换）**：固定绝对阈值 + 噪声
        随 dtype 变化是此前"bf16 阈值失效"的根因——修正阈值逻辑（随 dtype 自适应）
        而非侵入性改模块 dtype。实测（32 组采样）：bf16 同语义实现差上限 7.7e-3、
        语义偏差 4e-2，bf16 阈值取 2e-2 即可分；fp32 同语义差 ~0，阈值 1e-3。
        在模块实际 dtype 下自检——零侵入、更贴近生产负载。

        **预留扩展点（2026-10-23）**：fx graph SSA 结构对比将作为 module 等价性
        check 的一环（functorch make_fx 转译 module 与 provider 实现的计算图做 SSA
        直接对比——非侵入、图结构级等价证据），与数值精度兜底互补。当前先预留位置。

        **延迟模式（2026-10-23，D14 时机消解）**：传入 x/ref_forward 时为延迟自检
        （首次 __call__，真实输入，经 __call__ 流程——FSDP/DDP 的 all-gather hook
        已执行、weight 完整，并行包装后安全）；不传 x 则用合成输入（独立验证）。
        ref_forward 用于延迟自检时传 orig_forward，避免递归调已替换的 module.forward。

        返回 ok（bool）。成本 = 每模块一次小输入前向（生产 dtype）。
        """
        import torch
        from ttorch.witness import provenance
        w = module.weight
        dtype = w.dtype
        if admit_tol is None:
            # 阈值随 dtype 自适应（失效根因：固定阈值 + 噪声随 dtype 变）：
            # fp32 同语义差 ~0 → 1e-3；bf16/fp16 同语义差 ~7.7e-3 → 2e-2
            # （与语义偏差 4e-2 拉开约 2 倍 margin）。
            admit_tol = 1e-3 if dtype == torch.float32 else 2e-2
        if x is None:
            # 独立验证（无真实输入）：合成输入
            try:
                gen = torch.Generator(device=w.device).manual_seed(0)
            except Exception:
                gen = None  # 部分设备栈不支持 Generator 构造；随机输入无需定种子
            x = torch.randn(4, w.numel(), dtype=dtype, device=w.device, generator=gen)
        # 延迟自检时 ref_forward=orig_forward（避免递归调已替换的 module.forward）
        ref_fn = ref_forward if ref_forward is not None else module.forward
        try:
            ref = ref_fn(x)               # 模块参考实现（原生/被接管对象）
            out = self.native(x, w, eps)  # 本 provider 固定语义的参考实现
            scale = ref.abs().max().item() + 1e-12
            rel = (out - ref).abs().max().item() / scale
        except Exception as e:
            # 自检失败（如模块不支持该 dtype 输入）必须收敛为"不接管"——
            # 不许穿透 wrap_model（不变式 1：只降性能不降功能）。
            provenance.record(provenance.LAYER_PROVIDER, "wrap",
                              type(module).__name__, "rejected",
                              semantic=self.semantic, reason="self_check_error",
                              error=f"{type(e).__name__}: {e}")
            return False
        ok = rel < admit_tol
        if not ok:
            # 精度不一致（兜底生效）：本 provider 语义与模块真实实现不匹配——
            # 拒绝接管。可能是变体不符（应由对应变体的独立 provider 接管）或实现偏差。
            provenance.record(provenance.LAYER_PROVIDER, "wrap",
                              type(module).__name__, "rejected",
                              semantic=self.semantic, reason="self_check",
                              rel_err=round(rel, 6))
        return ok
