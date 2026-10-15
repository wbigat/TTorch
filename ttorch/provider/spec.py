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


class Provider(ABC):
    """第一层：一个模块语义的高性能 provider（注册单元）。

    子类按模块语义定义（semantic + probe_input + build_native）。native 是
    torch 原生单算子流程（kind="reference"），默认自带、永远启用、不可禁用——
    摘掉全部高性能实现后的功能底线 + 精度/收益基准。各硬件的高性能实现经
    add_impl 挂到 impls，执行时按平台选。
    """

    semantic = ""              # 模块语义，如 "rmsnorm" / "rope_bshd"
    slot = "compute"           # compute（schedule 插槽本期挂起）
    module_names: tuple = ()   # 该语义覆盖的 nn.Module 类名（endswith 匹配），自动发现用

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
    def probe_input(cls):
        """语义专属探测输入（真实设备张量，(args, kwargs)）。跟随语义、硬件无关，
        各候选共用——门禁拿它跑候选并与 native 对精度/收益。子类必须实现。"""
        raise NotImplementedError

    def wrap_module(self, module):
        """把一个匹配的 nn.Module 包成走 provider.call 的 forward（自动生效）。

        默认按 RMSNorm 类约定从模块提取 weight（Parameter）与 eps
        （variance_epsilon/eps），语义特殊的子类可重写。返回的 forward 每次
        调用经 manager.call 分发——享受门禁选择、两级回退与 provenance。
        weight 在调用期现读（不经闭包固化），保证 .to(device)/训练更新后仍正确。
        """
        import torch
        if not isinstance(getattr(module, "weight", None), torch.nn.Parameter):
            raise TypeError(f"{type(module).__name__} 无可包装的 weight Parameter")
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        # zero_centered_gamma 变体（如 Qwen3.5）：有效 scale 是 (1+weight) 而非 weight。
        # 若不处理，直乘 weight 与厂商 RMSNormFunction 数值不一致（实测 err 达 5.16）。
        zcg = getattr(module, "zero_centered_gamma", False)
        semantic, slot = self.semantic, self.slot
        orig_forward = module.forward          # 留原始实现：bypass 时回退/A-B 对比
        from ttorch.provider import manager as _pm  # 延迟 import 避免环

        def forward(x, *args, **kw):
            if _pm.instance()._bypass:         # 全局旁路：走原始实现（baseline）
                return orig_forward(x, *args, **kw)
            w = module.weight
            if zcg:
                w = 1 + w  # 折成有效 scale；+1 可微，梯度正确回传 weight
            return _pm.call(semantic, x, w, eps, slot=slot)
        return forward
