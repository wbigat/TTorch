"""框架性能符号的后置绑定机制（通道 B）——纯机制，不含绑定清单与实现。

为什么需要：TT 的 .pth 激活早于训练框架 import——megatron 等框架的性能符号
（swiglu/rope）在激活时不存在，cuda_compat 既绑不到也 sweep 不到。本模块装
import 钩子，在目标模块加载完成的瞬间绑定定义点——后续 from-import 自然拿到
新实现（import 顺序保证）；已在场的模块直接绑 + 按对象身份扫尾兜底。

盘点留档：megatron 的 `weighted_bias_swiglu_impl`（token-wise 加权 SwiGLU）
当前**未纳入**绑定（原生实现 bias 即 NotImplementedError，且本配置训练路径
不走它）——若未来接入，建 `weighted_bias_swiglu` provider 语义并按
bind_points 注册，勿在体系外单独绑。

归 provider 的理由（2026-10-20，二度修正：曾连机制一起误置 bridge）：性能符号
替换的执行动作属"性能增量"语义——设计文档 §8 边界明确 Bridge"不负责选择
fused kernel、模块性能替换"，故机制与实现同归 provider（本模块只依赖
witness.provenance，provider 内部闭环）；bridge/cuda_compat 只管功能域的
torch.cuda 收束。

**绑定目标不是厂商实现，而是 dispatcher**（FunctionDispatcher）：符号被绑成
透明分发器，调用经 provider.manager.call 分发——函数级接入与模块级 wrap
共享三级门禁/负载签名缓存/两级回退/kill switch/bypass/provenance。
绑定清单由 provider.manager.activate() 按各 Provider 的 bind_points 注册。

依赖方向：fn_binding → witness（provenance）；manager → fn_binding（注册）。
"""
import importlib
import importlib.abc
import sys
from dataclasses import dataclass
from typing import Callable

from ttorch.witness import provenance

__all__ = ["PerfBinding", "FunctionDispatcher", "make_dispatcher",
           "register_binding", "install"]


@dataclass(frozen=True)
class PerfBinding:
    """一条框架性能符号的绑定注册（provider 层按 Provider.bind_points 生成）。

    factory：bind 期以原符号为参构造替换对象（生产就是 make_dispatcher —
    dispatcher 内部走 provider 门禁选择实现，绑定层不再单独对账）。
    """
    dotted: str
    module: str
    attr: str
    factory: Callable  # (orig) -> 替换对象


class FunctionDispatcher:
    """函数级接入的 dispatcher：透明分发到 provider 体系。

    bypass（全局旁路）走原实现；正常路径经 manager.call（门禁选择 impl /
    失败回退 native）。dispatcher 本身不做任何数值判断——正确性由
    provider 的门禁 + native 底线保证（测量与选择解耦，绑定只是接线）。

    必须是模块级类而非闭包：megatron MLP 会把绑定后的函数存进实例属性
    （self.activation_func = config.activation_func），闭包让
    torch.save(整模型) 炸 PicklingError（实测）——与 ProviderModuleForward
    同理。orig 是 megatron 模块级函数，可 pickle。
    """

    def __init__(self, semantic, orig):
        self._semantic = semantic
        self._orig = orig

    def __call__(self, *args, **kw):
        from ttorch.provider import manager as _pm
        m = _pm.instance()
        if m.is_bypassed():
            return self._orig(*args, **kw)
        return m.call(self._semantic, *args, **kw)

    def __reduce__(self):
        """pickle 支持：符号替换后 orig 的"模块+名字"引用校验必然失败
        （getattr(module, attr) 已是 dispatcher 而非 orig）——按语义 +
        orig 的坐标存，unpickle 时从 _origin_ 留痕恢复原实现。"""
        return (_restore_dispatcher,
                (self._semantic, self._orig.__module__, self._orig.__name__))

    def __repr__(self):
        return f"<FunctionDispatcher {self._semantic}>"


def _restore_dispatcher(semantic, mod_name, attr):
    """unpickle 恢复：从定义点模块的 _origin_ 留痕拿原实现（未绑定的
    环境则取当前值）。"""
    module = importlib.import_module(mod_name)
    orig = getattr(module, f"_origin_{attr}", getattr(module, attr))
    return FunctionDispatcher(semantic, orig)


def make_dispatcher(semantic, orig):
    """构造函数级 dispatcher（bind 期以原符号为参）。"""
    return FunctionDispatcher(semantic, orig)


_BINDINGS = []  # 已注册清单（provider 层经 register_binding 写入）


def register_binding(b: PerfBinding):
    """注册一条框架性能符号绑定（provider.activate() 时调用）。幂等按 dotted 查重。"""
    assert all(x.dotted != b.dotted for x in _BINDINGS), f"重复注册: {b.dotted}"
    _BINDINGS.append(b)
    return b


# ----------------------------------------------------------------------
# 绑定执行（机制：装 dispatcher + 留痕 + 扫尾 + provenance）
# ----------------------------------------------------------------------

def _bind_one(b: PerfBinding):
    """绑一条：取原符号 → 构造 dispatcher → setattr + _origin_ 留痕 + 扫尾。"""
    module = importlib.import_module(b.module)
    orig = getattr(module, b.attr, None)
    if orig is None:
        provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted, "skipped",
                          reason="目标符号不存在")
        return
    try:
        new = b.factory(orig)
    except Exception as e:
        provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted, "skipped",
                          reason=f"dispatcher 构造失败: {type(e).__name__} {e}")
        return

    setattr(module, f"_origin_{b.attr}", orig)
    setattr(module, b.attr, new)
    # 扫尾兜底：本模块若已被其他框架模块 from-import，按对象身份修陈旧引用
    # （正常路径钩子在其 exec_module 完成时即绑，后续 from-import 自然拿新实现；
    # 这里是"激活晚于框架 import"场景的补救）。
    swept = 0
    for m in sys.modules.copy().values():
        if m is module:
            continue
        mod_name = getattr(m, "__name__", "")
        if not any(mod_name == p or mod_name.startswith(p + ".")
                   for p in ("megatron", "mcore_bridge", "swift")):
            continue
        try:
            if id(getattr(m, b.attr, None)) == id(orig):
                setattr(m, f"_origin_{b.attr}", orig)
                setattr(m, b.attr, new)
                swept += 1
        except Exception:
            continue
    provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted, "remap",
                      target=repr(new),
                      intent="函数级 dispatcher（provider 门禁选择实现）",
                      swept_modules=swept)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """目标模块 exec_module 完成时触发绑定（后置通道）。

    包装真实 loader 的 exec_module：模块体执行完 → 立即绑定义点 → 返回给
    import 方，后续 from-import 拿到的就是新实现（无需扫尾）。
    """

    def __init__(self, on_module):
        self._on_module = on_module

    def find_spec(self, fullname, path=None, target=None):
        pending = {b.module for b in _BINDINGS}
        if fullname not in pending:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        orig_exec = spec.loader.exec_module
        on_module = self._on_module

        def _exec(module):
            orig_exec(module)
            on_module(module.__name__)

        spec.loader.exec_module = _exec
        return spec


_INSTALLED = False


def _on_module_imported(module_name):
    for b in _BINDINGS:
        if b.module == module_name:
            _bind_one(b)


def install():
    """安装后置绑定钩子（幂等）。注册完成后调用（ttorch.activate 末尾）。

    已在场的目标模块立即绑定（含扫尾兜底）；未到场的等 import 钩子。
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    finder = _PostImportFinder(_on_module_imported)
    sys.meta_path.insert(0, finder)
    seen = {b.module for b in _BINDINGS}
    for name in list(seen):
        if name in sys.modules:
            _on_module_imported(name)
