"""PlatformBase：厂商硬件的实现契约——主体是 torch 社区接口，原样对齐。

厂商接入一块新卡 = 写一个 PlatformBase 子类（放 adapters/ 下独立文件），
把本文件声明的接口映射到自家栈。PlatformHandle 再持有它 + 一份 PlatformSpec
对外呈现（见 platform_handle.py）。

**本文件只放 torch 社区真实存在的接口**（2026-10-13 清理）：方法名逐一对齐
`torch.cuda.*` 的真实符号名，不自造名字——门面要绑回 torch.cuda 的同名符号，
名字对不上就是给自己造一套伪 API。被清掉的伪接口与理由：

- `range_push`：torch.cuda 下没有（真实位置 `torch.cuda.nvtx.range_push`），
  且真实 policy 未绑定它 -> 作为厂商/mock 的**扩展接口**由实现类自行提供，
  manager 的动态转发照样能暴露，不占契约；
- `stream_current` / `stream_set` / `stream_create`：torch 真名是
  `current_stream` / `set_stream` / `Stream(...)` 构造 -> 改名对齐；
- `stream_wrap`：G14 的分流产物。归并进 `Stream(**kwargs)` 后自然消解——
  同一个方法同时覆盖"框架无参创建"与"torch 内部带 kwargs 构造"，两者都
  原样转发给厂商原生 Stream 类，本就不需要分流（见该方法注释）；
- `stream_wait`：torch 里是 `Stream.wait_stream(other)` **对象方法**，框架
  直接调原生对象，不经门面 -> 契约不需要它；
- `stream_key`：conformance 判等用的测试工具，不是平台能力 -> 挪进 conformance；
- `active_streams`：torch 不暴露活跃流枚举（真实栈实现本就只能抛
  unsupported）-> 需要枚举的实现（mock）自己在内部处理。

契约要点：
- `synchronize()` = device-wide 等待（不是 current-stream）；
- 状态查询（`current_device` / `memory_allocated`）动态只读，不许缓存；
- RNG state 是**平台 torch 栈的原生对象**（真实栈 ByteTensor），会流出 TT
  边界被框架原生代码直接消费（G15），类型必须与被替换符号一致；
- stream 返回值是平台**原生** stream 对象，门面不包装、上层不主动管理。

**不支持 = 不实现**（2026-10-14 简化，取代原 capability 声明表）：本类所有
接口的默认实现直接抛 `TTPlatformError("unsupported")`，平台不实现某组接口
（如 CPU 没有 stream / 显存查询概念）即视为不支持，调到才报错。

为什么不再要平台自报 `capabilities()`：
- "接口在不在"是**实现的事实**，自报等于让不可信输入复述已知事实，还要
  conformance 反过来验证它说得对不对——声明、验证两头都是成本；
- 统一抛 `TTPlatformError("unsupported")` 而非 `NotImplementedError`，调用方
  （checkpoint 等）照常 `except TTPlatformError` 走降级，门控效果不变；
- 需要"先问再调"的调用方用 `PlatformBase.supports(obj, name)` 内省，答案
  来自"子类有没有覆盖"，派生出的事实不会撒谎。
"""

from abc import ABC

from ttorch.platform.platform_handle import TTPlatformError

__all__ = ["PlatformBase", "TORCH_API"]

# 本契约覆盖的 torch.cuda 符号名（interface 的动态转发白名单据此生成）。
# 每一项都必须是 torch.cuda 下真实存在的符号——加项前先去 torch 源码核对。
TORCH_API = (
    "set_device", "current_device", "synchronize",
    "empty_cache", "memory_allocated", "mem_get_info",
    "get_rng_state", "set_rng_state",
    "Stream", "current_stream", "set_stream", "stream",
    "is_available", "device_count", "can_device_access_peer",
)


def _unsupported(obj, name):
    """默认实现的统一报错：本平台不支持该接口。

    用 TTPlatformError("unsupported") 而非 NotImplementedError，是为了让
    调用方能用一个统一类别做降级（checkpoint 的 RNG 存取即如此）。
    """
    return TTPlatformError(
        "unsupported", f"{type(obj).__name__} 未实现 {name}()：本平台不支持该接口")


# 备案（2026-10-14，暂不做）：可给"核心必备接口"（synchronize / set_device /
# current_device / rng 存取——每个平台都必须实现的）加 @abstractmethod，与
# "能力可裁剪接口"（stream 族 / 显存查询）在抽象层级上区分开。当前刻意统一用
# 方法体抛 unsupported 而非 @abstractmethod：好让只实现部分接口的诚实降级
# 平台（如 TorchCpuPlatform 不实现 stream/memory）仍能实例化，
# 未实现的接口调到时才报错。若引入 @abstractmethod 需先理清两类接口边界。
class PlatformBase(ABC):
    """torch 社区接口契约 + 该平台的自述信息（版本 / 生命周期）。

    接入一块卡 = 覆盖本类中该平台支持的接口。没覆盖的保持默认实现，调用时
    抛 `TTPlatformError("unsupported")`——这就是"不支持"的全部表达方式，
    不需要另外声明一张能力表。

    写实现时值得先想清楚每个接口的性质（影响契约与 conformance 验证）：
    - 有副作用（set_device / synchronize / empty_cache / set_stream）：
      契约是**副作用范围**，如 synchronize 必须 device-wide；
    - 动态只读（current_device / memory_allocated / current_stream）：
      契约是**口径一致 + 不许缓存**，答案进程内会变。
    """

    @staticmethod
    def supports(obj, name):
        """该实现是否提供了 `name` 接口（子类是否覆盖了默认实现）。

        给"先问再调"的调用方用（conformance 裁剪、框架侧分支）。事实派生自
        实现本身，不经平台自报——继承链上的覆盖同样算数（如
        TorchXpuMasqueradePlatform 继承 TorchCudaPlatform 的实现）。
        """
        impl = getattr(type(obj), name, None)
        return impl is not None and impl is not getattr(PlatformBase, name, None)

    # ==================================================================
    # torch 社区接口（方法名 = torch.cuda 符号名，G12：不可改名）
    # ==================================================================

    # --- 设备 ---
    def set_device(self, device):
        """对齐 torch.cuda.set_device(device)。"""
        raise _unsupported(self, "set_device")

    def current_device(self):
        """对齐 torch.cuda.current_device()。

        返回值必须能当设备标识回喂（set_device / 张量 device= 参数）——
        swift 有 6+ 处把它直接传给 torch.tensor(device=)，conformance 的
        device_identity_consumable 强制此契约。
        """
        raise _unsupported(self, "current_device")

    def synchronize(self, device=None):
        """对齐 torch.cuda.synchronize(device=None)。语义 = **device-wide** 等待。

        device 可选参是 torch 真实签名的一部分——torch 内部与框架都有带参
        调用点，实现签名必须是其超集，否则绑定后带参调用点直接 TypeError（G12）。
        """
        raise _unsupported(self, "synchronize")

    # --- 显存 ---
    def empty_cache(self):
        """对齐 torch.cuda.empty_cache()：allocator 提示，尽力语义（可 no-op）。"""
        raise _unsupported(self, "empty_cache")

    def memory_allocated(self, device=None):
        """对齐 torch.cuda.memory_allocated(device=None)。口径=已分配字节，实时读。"""
        raise _unsupported(self, "memory_allocated")

    def mem_get_info(self, device=None):
        """对齐 torch.cuda.mem_get_info(device=None)：返回 (free_bytes, total_bytes)。

        厂商侧结构/单位不同由实现类换算（如某栈的 MiB dict -> 字节 tuple）。
        """
        raise _unsupported(self, "mem_get_info")

    # --- RNG（checkpoint save / resume） ---
    def get_rng_state(self, device=None):
        """对齐 torch.cuda.get_rng_state(device)。契约见 set_rng_state。"""
        raise _unsupported(self, "get_rng_state")

    def set_rng_state(self, state, device=None):
        """对齐 torch.cuda.set_rng_state(state, device)。

        state 契约 = **该平台 torch 栈的原生 RNG state 对象**（真实栈是
        torch ByteTensor）。要求：get/set 回环相等；非法值抛
        TTPlatformError("invalid_rng_state")。

        为什么不是"不透明可序列化对象"（G15）：该对象会**流出 TT 边界**——
        megatron 的 RNG tracker 把 get_rng_state() 的结果直接回喂 torch 原生
        `Generator.set_state()`，不经我们的 set_rng_state。返回值类型必须与
        被替换符号一致，否则框架在 TT 管不到的地方炸。
        """
        raise _unsupported(self, "set_rng_state")

    # --- stream ---
    def Stream(self, **kwargs):  # noqa: N802  # 刻意对齐 torch.cuda.Stream 符号名
        """对齐 torch.cuda.Stream(...) 构造：返回平台**原生** stream 对象。

        `**kwargs` 覆盖两类调用者，两者都原样转发给厂商原生 Stream 类：
        - 框架无参创建：`torch.cuda.Stream()`；
        - torch 原实现内部带厂商构造参数：`Stream(stream_id=..., device_index=...,
          device_type=...)`。后者是 G14——`torch.cuda.current_stream` 的全局名
          空间就是被我们改过的 `torch.cuda.__dict__`，绑定 `torch.cuda.Stream`
          后它内部的构造也会落到门面。

        因为门面不包装返回对象（无自定义 stream 类型），两类调用者的正确行为
        是同一个：交给原生类。所以这里一个方法足够，不需要再按调用者分流。
        """
        raise _unsupported(self, "Stream")

    def current_stream(self, device=None):
        """对齐 torch.cuda.current_stream(device=None)：当前原生 stream 对象。"""
        raise _unsupported(self, "current_stream")

    def set_stream(self, stream):
        """对齐 torch.cuda.set_stream(stream)：切换当前 stream。"""
        raise _unsupported(self, "set_stream")

    def stream(self, stream):
        """对齐 torch.cuda.stream(stream)：返回切流的**上下文管理器**。

        由平台自己实现而非门面用 set_stream/current_stream 组装（2026-10-13
        修正）：`torch.cuda.stream` 本就是厂商侧的真实符号（返回其
        `StreamContext`），门面自己组装一个等于在收束层重新实现厂商语义——
        切流的进入/退出细节、与厂商 allocator/graph 的交互都该由厂商负责。
        门面只保证"框架的调用能进来"，差异在各平台实现里消化。
        """
        raise _unsupported(self, "stream")

    # --- 硬件存在性 / 规模查询 ---
    # 曾经是 capability 表里的 probed 档（device_available/device_count/p2p），
    # 由门面读缓存作答。2026-10-14 改回普通接口：查询本身廉价，不值得为它建
    # 一套 probe 期缓存 + 校验机制；且缓存版 can_device_access_peer 会忽略
    # 入参、对任意设备对返回同一个值，在拓扑不对称的多卡机上是错的。
    def is_available(self):
        """对齐 torch.cuda.is_available()：本平台的加速设备是否可用。"""
        raise _unsupported(self, "is_available")

    def device_count(self):
        """对齐 torch.cuda.device_count()：可见设备数。"""
        raise _unsupported(self, "device_count")

    def can_device_access_peer(self, device, peer_device):
        """对齐 torch.cuda.can_device_access_peer(device, peer_device)：
        两张卡之间能否 P2P 访问（逐对回答，不是全局静态事实）。"""
        raise _unsupported(self, "can_device_access_peer")

    # ==================================================================
    # 平台自述（非 torch 接口；TT 要求实现类提供的元信息与生命周期）
    # ==================================================================

    sdk_range = ("0.0", "999")  # 声明支持的 SDK 版本范围 (min, max)，左闭右开

    def sdk_version(self):
        """当前环境的 SDK 版本（激活期与 sdk_range 协商）。每个平台必须实现。"""
        raise NotImplementedError

    def initialize(self):
        """激活时调用一次：runtime 初始化、进程级资源准备。

        刻意不叫 `init`——torch.cuda 下有个同名但语义不同的
        `torch.cuda.init()`（惰性初始化 CUDA 上下文），重名会让人以为这是
        在对齐那个符号。本方法是 TT 的平台生命周期钩子，不是 torch 接口。
        """

    def finalize(self):
        """退出/切换前调用一次：释放进程级资源。"""
