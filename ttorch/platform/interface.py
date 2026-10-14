"""interface：torch 接口转发层——把当前激活 handle 的 torch 接口原样暴露出去。

本模块是 policy 表的绑定目标（`impl=interface.synchronize`），职责**只有转发**：
`base.TORCH_API` 列的符号不在这里逐个定义，由模块级 `__getattr__`（PEP 562）
按需生成转发函数。加一个 torch 接口只需改 base 的契约，这里零改动。

**刻意不在本层处理任何具体特性或类型**（2026-10-13 修正）：曾经这里有个
`class Stream` 和一个自己用 set_stream/current_stream 组装的 `stream` 上下文
管理器——那是收束层越界去实现厂商语义。现在两者都是普通转发，差异由各平台
的 PlatformBase 子类消化。Bridge 只保证"框架的调用能照原样走进来"。

三条硬约束决定了实现细节（都是真实调用面的要求，不能简化）：

1. **必须缓存**：admission 用对象身份断言绑定（`torch.cuda.synchronize is
   interface.synchronize`），同名每次取必须返回同一个函数对象；diagnostics 装
   探针也按身份比对（`getattr(torch.cuda, name) is interface_fn`）。
2. **必须延迟求值**：policy 表在**模块 import 期**就取门面
   （`impl=interface.set_device`，见 module_binding.py），此时 platform 往往还
   没激活——转发函数要到"被调用"时才 activation()。
3. **必须白名单**：否则 `interface.synchronze` 这类拼写错误会静默返回一个转发
   函数，调用时才在实现深处炸，错误被推迟且难归因。白名单 = base 契约 ∪
   厂商扩展（`EXTRA_FORWARD`）。

能力裁剪不在本层：平台不支持某接口时，`PlatformBase` 的默认实现抛统一的
`TTPlatformError("unsupported")`，框架据此走降级路径（而不是在厂商 API
深处炸原生错误）。本模块不查任何能力表，纯转发。

与 manager 的分工：manager 管"激活哪个平台、当前是谁"（有状态）；
本模块只管"把当前平台的 torch 接口转出去"（无状态）。
"""
from ttorch.platform import platform_base as _base
from ttorch.platform import manager as _manager
from ttorch.platform.platform_handle import TTPlatformError

# 厂商/mock 的扩展接口：不属 torch 社区契约（torch.cuda 下无此符号），但实现
# 类可提供，门面照样转发。range_push 对应的 torch 真实位置是
# torch.cuda.nvtx.range_push，真实 policy 未绑定它，仅 mock demo 的
# vendor_cuda.range_push 绑到此门面。
EXTRA_FORWARD = ("range_push",)

# 可转发名单：base 契约 ∪ 厂商扩展
FORWARDABLE = frozenset(_base.TORCH_API) | frozenset(EXTRA_FORWARD)

_cache = {}

__all__ = sorted(FORWARDABLE)


def _make_forward(name):
    """生成一个转发函数：延迟取当前激活 handle，原样转发。

    不做能力门控（2026-10-14 删除 CAP_GUARD）：平台不支持某接口时，
    PlatformBase 的默认实现本就抛 `TTPlatformError("unsupported")`，
    调用方的降级分支照常生效——不需要门面再查一张能力表。
    """
    def forward(*args, **kwargs):
        return getattr(_manager.activation(), name)(*args, **kwargs)

    forward.__name__ = name
    forward.__qualname__ = name
    forward.__module__ = __name__
    forward.__doc__ = f"TT 门面：转发到当前 platform 实现的 {name}()。"
    return forward


def __getattr__(name):
    """模块级动态属性（PEP 562）：按白名单生成并缓存 torch 接口门面。"""
    if name not in FORWARDABLE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    fn = _cache.get(name)
    if fn is None:
        fn = _make_forward(name)
        _cache[name] = fn   # 身份稳定：绑定与断言都依赖它
    return fn


def __dir__():
    return sorted(set(globals()) | FORWARDABLE)
