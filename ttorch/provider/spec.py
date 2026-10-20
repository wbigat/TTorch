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
    """

    def __init__(self, module, semantic, slot, zcg):
        self._module = module
        self._orig_forward = module.forward  # bypass 回退 / A-B 对比基线
        self._semantic = semantic
        self._slot = slot
        self._zcg = zcg

    def __call__(self, x, *args, **kw):
        from ttorch.provider import manager as _pm  # 延迟 import 避免环
        if _pm.instance().is_bypassed():  # 全局旁路：走原始实现（baseline）
            return self._orig_forward(x, *args, **kw)
        module = self._module
        w = module.weight
        if self._zcg:
            w = 1 + w  # 折成有效 scale；+1 可微，梯度正确回传 weight
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        return _pm.call(self._semantic, x, w, eps, slot=self._slot)


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

        返回 None = 接管前数值自检未过（_self_check），模块保持原样——
        证据先于声明，对不齐就不接管（功能不降级）。抛 TypeError = 模块形态
        不符该语义（无 weight Parameter）。
        """
        import torch
        if not isinstance(getattr(module, "weight", None), torch.nn.Parameter):
            raise TypeError(f"{type(module).__name__} 无可包装的 weight Parameter")
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        # zero_centered_gamma 变体（如 Qwen3.5）：有效 scale 是 (1+weight) 而非 weight。
        # 若不处理，直乘 weight 与厂商 RMSNormFunction 数值不一致（实测 err 达 5.16）。
        # flag 嗅探只作先验——自检按实测对齐结果定有效语义（见 _self_check）。
        zcg = getattr(module, "zero_centered_gamma", False)
        ok, zcg = self._self_check(module, eps, zcg)
        if not ok:
            return None
        return ProviderModuleForward(module, self.semantic, self.slot, zcg)

    def _self_check(self, module, eps, zcg, admit_tol=1e-3):
        """接管前两判据（分离）：变体识别 + 正确性准入。

        1) 变体识别（选择题，无阈值）：候选 scale 语义（w 直乘 / 1+w）各跑一次，
           取误差最小者为有效语义——flag 嗅探只作先验排序，不作结论（厂商符号
           替换可能丢/伪造 flag；Gemma 式 (1+w) 硬编码无 flag）。
        2) 正确性准入（判断题）：最小误差必须低于 admit_tol 才接管——识别出的
           语义与模块真实实现仍可能不是同一数学对象（实测：rel=4% 的模块曾被
           宽松阈值放过，违反 5e-3 全局口径），两判据必须独立。

        轨道用 fp32（模块临时 upcast + fp32 输入，dtype 往返无损）：bf16 下
        同语义实现差（1-2 ulp ≈ 6e-3）与语义偏差（4e-2）只差一个数量级，阈值
        怎么拍都误伤；fp32 下同语义差 5e-7 级、语义差不变，拉开 5 个数量级。
        厂商 kernel 已实测支持 fp32 输入。

        返回 (ok, effective_zcg)。成本 = 每模块至多两次小输入前向（fp32）。
        """
        import torch
        from ttorch.witness import provenance
        w = module.weight
        orig_dtype = w.dtype
        try:
            gen = torch.Generator(device=w.device).manual_seed(0)
        except Exception:
            gen = None  # 部分设备栈不支持 Generator 构造；随机输入无需定种子
        x = torch.randn(4, w.numel(), dtype=torch.float32, device=w.device,
                        generator=gen)
        module.to(torch.float32)
        try:
            ref = module.forward(x)
            scale = ref.abs().max().item() + 1e-12
            candidates = [(True, 1 + w.float()), (False, w.float())] if zcg \
                else [(False, w.float()), (True, 1 + w.float())]
            best = None
            for cand_zcg, eff_w in candidates:
                out = self.native(x, eff_w, eps)
                rel = (out - ref).abs().max().item() / scale
                if best is None or rel < best[1]:
                    best = (cand_zcg, rel)
        except Exception as e:
            # 自检是 provider 内部机制，其失败（如模块不支持 fp32 输入）必须收敛为
            # "不接管"——不许穿透 wrap_model（不变式 1：只降性能不降功能）。
            provenance.record(provenance.LAYER_PROVIDER, "wrap",
                              type(module).__name__, "rejected",
                              semantic=self.semantic, reason="self_check_error",
                              error=f"{type(e).__name__}: {e}")
            return False, zcg
        finally:
            module.to(orig_dtype)  # fp32→bf16 回到原值（bf16 值域 ⊂ fp32）
        ok = best[1] < admit_tol
        if not ok:
            provenance.record(provenance.LAYER_PROVIDER, "wrap",
                              type(module).__name__, "rejected",
                              semantic=self.semantic, reason="self_check",
                              rel_err=round(best[1], 6))
        elif best[0] != zcg:
            # flag 与实测不一致（厂商替换丢 flag 等）：按实测语义接管，留证
            provenance.record(provenance.LAYER_PROVIDER, "wrap",
                              type(module).__name__, "self_check",
                              semantic=self.semantic,
                              detail=f"flag zcg={zcg} 与实测不符，按实测 zcg={best[0]} 接管",
                              rel_err=round(best[1], 6))
        return ok, best[0]
