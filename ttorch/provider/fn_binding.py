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

from ttorch.utils import _rebind
from ttorch.witness import provenance

__all__ = ["PerfBinding", "FunctionDispatcher", "make_dispatcher",
           "register_binding", "install", "uninstall", "SWEEP_PACKAGES"]

# 扫尾白名单（模块名前缀）：修这些包内持有原对象的 from-import 陈旧引用。
# 接新框架时改这一处（此前硬编码在 _bind_one 体内）。
SWEEP_PACKAGES = ("megatron", "mcore_bridge", "swift")


@dataclass(frozen=True)
class PerfBinding:
    """一条框架性能符号的绑定注册（provider 层按 Provider.bind_points 生成）。

    factory：bind 期以原符号为参构造替换对象（生产就是 make_dispatcher —
    dispatcher 内部走 provider 门禁选择实现）。

    semantic：绑定对应的 provider 语义——接管前对账（native vs orig）需要它
    去取该语义的 native 与 probe_input（见 `_admit_native_vs_orig`）。
    """
    dotted: str
    module: str
    attr: str
    factory: Callable  # (orig) -> 替换对象
    semantic: str = ""


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
    """unpickle 恢复：从定义点模块的 owner 留痕拿原实现（未绑定的环境则取
    当前值）。留痕名随 _rebind 的 owner 规则（见 _rebind.origin_attr）。"""
    module = importlib.import_module(mod_name)
    orig = getattr(module, _rebind.origin_attr("provider", attr),
                   getattr(module, attr))
    return FunctionDispatcher(semantic, orig)


def make_dispatcher(semantic, orig):
    """构造函数级 dispatcher（bind 期以原符号为参）。"""
    return FunctionDispatcher(semantic, orig)


_BINDINGS = []      # 已注册清单（provider 层经 register_binding 写入）
_SKIPPED = {}       # dotted -> 跳过原因（失效可见：install 摘要打印）
_TRIED = set()      # 已尝试绑定的 dotted（用于区分 skipped 与 pending）


def register_binding(b: PerfBinding):
    """注册一条框架性能符号绑定（provider.activate() 时调用）。幂等按 dotted 查重。"""
    assert all(x.dotted != b.dotted for x in _BINDINGS), f"重复注册: {b.dotted}"
    _BINDINGS.append(b)
    return b


# ----------------------------------------------------------------------
# 绑定执行（机制：装 dispatcher + 留痕 + 扫尾 + provenance）
# ----------------------------------------------------------------------

_ADMIT_REL_TOL = 5e-3   # 与 provider 精度门禁同口径（spec.py 的 admit 容差）


def _admit_native_vs_orig(b: PerfBinding, orig):
    """接管前对账：TT 手工复刻的 native vs **被替换的框架原实现**。

    为什么必须有（Symbrewrite 吸收文档 §3 / Rewrite 设计文档 §4-B）：门禁链路
    `call → select → _gate → verify_impl(impl, provider.native)` 的参考系是 TT
    自家的 native，而 native 是**手工复刻**的——"与 megatron 公式逐字一致"此前
    只是注释里的声明，不是证据。实测把框架原实现改成 ×1.5 后接管，输出差
    1.337 而 provenance 三条事件全绿：TT 换掉了框架的计算语义却零告警。

    这条对账把"逐字一致"从声明变成**每次启动都复验的证据**，顺带让上游公式
    演进 / 版本漂移从静默失效变成显式拒绝。

    与模块级的关系：模块级 `Provider._self_check` 早已用 `module.forward`
    （原实现）做基准并实测择优；这里是函数级通道的同一姿态——两条通道的
    正确性姿态拉齐，不是新增机制。

    对账本身无法进行（签名/设备不匹配等）时**不阻断**绑定，只记 `unverified`
    留痕——不为取证而破坏已验证能工作的路径。

    返回 True 表示允许接管。
    """
    from ttorch.provider import manager as _pm
    from ttorch.platform import manager as _pfm

    provider = _pm.instance().registry.get(b.semantic)
    if provider is None:
        return True
    try:
        dev = _pfm.activation().device_spec()
        args, kw = provider.probe_input(device=dev)
        ref = orig(*args, **kw)
        out = provider.native(*args, **kw)
        scale = float(ref.float().abs().max()) + 1e-12
        rel = float((out.float() - ref.float()).abs().max()) / scale
    except Exception as e:
        provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted,
                          "unverified", semantic=b.semantic,
                          reason=f"native_vs_orig 对账无法进行: "
                                 f"{type(e).__name__}: {e}")
        return True
    ok = rel < _ADMIT_REL_TOL
    provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted,
                      "admitted" if ok else "rejected", semantic=b.semantic,
                      native_vs_orig=round(rel, 6), tol=_ADMIT_REL_TOL)
    return ok


def _bind_one(b: PerfBinding):
    """绑一条：取原符号 → 接管前对账 → 构造 dispatcher → setattr + 留痕 + 扫尾。"""
    _TRIED.add(b.dotted)
    module = importlib.import_module(b.module)
    orig = getattr(module, b.attr, None)
    if orig is None:
        _SKIPPED[b.dotted] = "目标符号不存在"
        provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted, "skipped",
                          reason="目标符号不存在")
        return
    if b.semantic and not _admit_native_vs_orig(b, orig):
        _SKIPPED[b.dotted] = f"native_vs_orig 超差（tol={_ADMIT_REL_TOL}）"
        provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted, "skipped",
                          reason=f"native_vs_orig 超差（tol={_ADMIT_REL_TOL}），"
                                 f"拒绝接管：框架保持原实现")
        return
    try:
        new = b.factory(orig)
    except Exception as e:
        _SKIPPED[b.dotted] = f"dispatcher 构造失败: {type(e).__name__}"
        provenance.record(provenance.LAYER_PROVIDER, "bind", b.dotted, "skipped",
                          reason=f"dispatcher 构造失败: {type(e).__name__} {e}")
        return

    # 替换动作走 _rebind（与 bridge 共用一份实现）：owner 留痕 `_tt_origin_
    # provider_<attr>` 不再与 bridge 的 `_origin_` 撞名，且自动进入全局快照——
    # assert_bindings_intact 从此覆盖 provider 的函数级绑定（此前 0 条）。
    # 扫尾白名单（此前硬编码在函数体内）提到模块级常量，接新框架时改一处。
    swept = _rebind.rebind(module, b.attr, new, owner="provider",
                           sweep_packages=SWEEP_PACKAGES)
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


def _print_summary():
    """绑定摘要（**失效可见**）：skipped 必须出现在终端，不能只进 JSONL。

    为什么必须有（Symbrewrite 吸收文档 §2.2）：绑定目标是 megatron 的**私有
    函数**（`_apply_rotary_pos_emb_bshd` 带前导下划线），路径/签名随版本漂移
    是常态、上游无兼容承诺。失效时现象是"没报错，但性能回到基线"——最难排查
    的一类故障。此前 skipped 只写进 provenance JSONL，终端只显示 cuda_compat
    的 policy 数，用户对失效毫无感知。
    """
    bound = sum(1 for b in _BINDINGS
                if b.dotted in _TRIED and b.dotted not in _SKIPPED)
    pending = sum(1 for b in _BINDINGS if b.dotted not in _TRIED)
    print(f"[tt] fn_binding: {bound} bound, {len(_SKIPPED)} skipped, "
          f"{pending} pending（目标框架未 import）")
    for dotted, reason in _SKIPPED.items():
        print(f"[tt] fn_binding skipped: {dotted} —— {reason}")


def install():
    """安装后置绑定钩子（幂等）。注册完成后调用（ttorch.activate 末尾）。

    已在场的目标模块立即绑定（含扫尾兜底）；未到场的等 import 钩子。
    末尾打印绑定摘要——失效可见（见 _print_summary）。
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
    _print_summary()


def uninstall():
    """卸载 import 钩子并恢复全部绑定（单测 / 多次 activate 场景）。

    Rewrite 设计文档 §3.1 列的 import hook 硬要求之一：必须提供 uninstall 并
    验证 finder 未泄漏——否则重复 activate 会在 sys.meta_path 上累积 finder，
    且 _origin_ 留痕被二次覆盖后回滚会回滚错。
    """
    global _INSTALLED
    for f in list(sys.meta_path):
        if isinstance(f, _PostImportFinder):
            sys.meta_path.remove(f)
    for b in list(_BINDINGS):
        module = sys.modules.get(b.module)
        if module is None:
            continue
        orig = getattr(module, f"_origin_{b.attr}", None)
        if orig is None:
            continue
        setattr(module, b.attr, orig)
        try:
            delattr(module, f"_origin_{b.attr}")
        except AttributeError:
            pass
    _BINDINGS.clear()
    _SKIPPED.clear()
    _TRIED.clear()
    _INSTALLED = False
