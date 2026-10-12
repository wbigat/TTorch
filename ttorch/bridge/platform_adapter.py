"""Platform Adapter：统一接口的后端契约（与 Platform Manager 分文件）。

职责分界：
- Manager（platform_mgr.py）：注册 / 自动检测 / 激活门禁 / 分发门面，
  不关心厂商接口长什么样；
- Adapter（本文件契约 + adapters.py 各实现）：把统一契约映射到厂商
  非标接口（填 gap），把厂商原生错误翻译成 TTPlatformError。

统一错误类别定义在本文件而非 manager：抛出方是 adapter，
manager 与上层只是透传/捕获，定义跟着抛出方走。
"""


class TTPlatformError(Exception):
    """统一错误类别：厂商原生错误经 adapter 翻译成它，上层据此写可移植处理。"""

    def __init__(self, category, detail):
        self.category = category
        super().__init__(f"[{category}] {detail}")


class TTStream:
    """统一 stream 句柄：包装 adapter + stream id，对框架暴露与 CUDA 一致
    的对象形态（wait_stream 方法），内部走 adapter 的填 gap 实现。"""

    def __init__(self, adapter, sid):
        self._adapter = adapter
        self.sid = sid

    def wait_stream(self, other):
        """跨流依赖：本 stream 等 other 完成（ddp fencing 的原语）。"""
        self._adapter.stream_wait(self.sid, other.sid)

    def __repr__(self):
        return f"<TTStream {self._adapter.name}:{self.sid}>"


class PlatformAdapter:
    """统一接口的后端契约。synchronize() 语义 = device-wide 等待。

    接口按三类区分：
    - 行为：set_device / synchronize / range_push（有副作用）
    - 状态查询：current_device / memory_allocated（动态只读，不许缓存）
    - 能力：capabilities() 返回的静态事实表（probe 期填充一次，可缓存）

    另有三条平台管理契约：
    - detect()       存在性探测（只看不初始化），供自动检测
    - sdk_range      声明支持的 SDK 版本范围，激活期校验
    - init/finalize  生命周期钩子，激活/退出时各调一次

    双身份（缺一不可， FlagOS 真坑沉淀）：
    - vendor_name  厂商身份：provider 分发 / 门禁 / provenance 用
    - device_name  torch 认得的设备类型：框架 `device.type == "cuda"`
                   这类比较的存活线。冒充 CUDA 的栈（GCU/MACA）上
                   vendor_name="gcu" 但 device_name 必须是 "cuda"，
                   写成品牌名会让框架里的 device.type 比较静默失效
    """

    name = "base"
    vendor_name = "base"     # 厂商身份（静态事实，激活期注入 capability）
    device_name = "base"     # torch 视角的设备类型（同上）
    auto_detect = True       # 测试夹具类 adapter 应置 False，不参与自动检测
    sdk_range = ("0.0", "999")  # (min, max)，左闭右开
    masquerades_as = frozenset()  # 本栈冒充的 compat 身份：native 探测命中时，
                                  # 被冒充身份的 detect 命中降级为证据而非冲突

    @classmethod
    def detect(cls):
        """存在性探测：只看不初始化。返回 (present: bool, evidence: str)。"""
        raise NotImplementedError

    def sdk_version(self):
        """当前环境的 SDK 版本（激活期与 sdk_range 协商）。"""
        raise NotImplementedError

    def init(self):
        """激活时调用一次：runtime 初始化、进程级资源准备。"""

    def finalize(self):
        """退出/切换前调用一次：释放进程级资源。"""

    def capabilities(self):
        """静态平台事实：probe 期填充，进程生命周期内不变。"""
        raise NotImplementedError

    def active_streams(self):
        """枚举当前全部活跃 stream——device-wide 语义的实现基础。"""
        raise NotImplementedError

    # --- 行为 ---
    def set_device(self, idx):
        raise NotImplementedError

    def synchronize(self, device=None):
        """device-wide 等待。device 可选参对齐 torch.cuda.synchronize(device)
        真实签名——torch 内部与框架都有带参调用点，门面签名必须是其超集，
        否则绑定后带参调用点直接 TypeError。"""
        raise NotImplementedError

    def range_push(self, name):
        raise NotImplementedError

    def empty_cache(self):
        """allocator 提示：尽力释放未用缓存（允许实现为 no-op）。"""
        raise NotImplementedError

    def set_rng_state(self, state, device=None):
        """恢复 RNG 状态（checkpoint resume）。

        state 契约 = **该平台 torch 栈的原生 RNG state 对象**（真实 adapter
        是 torch ByteTensor；mock adapter 用 str 自洽即可）。要求有二：
        get/set 回环相等；非法值抛 TTPlatformError("invalid_rng_state")。

        为什么不是"不透明可序列化对象"（G15 修正）：该对象会**流出 TT
        边界**——megatron 的 RNG tracker 把 get_rng_state() 的结果直接
        回喂 torch 原生 `Generator.set_state()`，不经我们的 set_rng_state。
        门面的返回值类型必须与被替换符号一致，否则框架拿到的是我们的
        自定义表示，在 TT 管不到的地方炸。
        """
        raise NotImplementedError

    # --- stream 原语（swift megatron 路径 ddp fencing 必触发） ---
    def stream_create(self):
        """新建 stream，返回 stream id。"""
        raise NotImplementedError

    def stream_current(self):
        """当前 stream id。"""
        raise NotImplementedError

    def stream_set(self, sid):
        """切换当前 stream（stream 上下文管理器的实现基础）。"""
        raise NotImplementedError

    def stream_wait(self, waiter, waitee):
        """跨流依赖：waiter 等 waitee 完成。"""
        raise NotImplementedError

    def stream_wrap(self, **kwargs):
        """按厂商原生构造参数造一个**原生** stream 对象（不是 TTStream）。

        存在理由（G14，G12"门面签名是 torch 真实签名超集"的推广）：
        remap 的爆炸半径不止于框架调用点——被绑定符号所在模块的**原实现
        内部**也会引用同模块的其他符号。`torch.cuda.current_stream` 的
        全局命名空间就是 `torch.cuda.__dict__`，绑定 `torch.cuda.Stream`
        后，它内部 `Stream(stream_id=..., device_index=..., device_type=...)`
        拿到的是我们的门面。

        所以构造型符号的门面必须覆盖两种调用者：
        - 框架调用（无参）-> 门面语义，返回 TTStream 句柄；
        - 原实现内部调用（带厂商构造 kwargs）-> 本方法，返回原生对象。

        判据是"调用者是谁"，不是"参数长什么样"：带厂商私有构造参数的
        只可能是厂商栈自己，此时必须原样放行，否则等于拿门面去冒充厂商
        内部数据结构。
        """
        raise TTPlatformError(
            "unsupported", f"{self.name}: 不支持按厂商构造参数包装 stream {sorted(kwargs)}")

    def stream_key(self, sid):
        """stream 身份的稳定可比较 key（conformance 判等用）。

        mock 的 sid 是 int，默认恒等即可；真实 torch Stream 每次包装可能是
        新对象，adapter 应覆盖为 (device_index, stream_id) 形态的稳定 key。
        """
        return sid

    # --- 状态查询（动态只读） ---
    def current_device(self):
        """返回值必须能当设备标识回喂（set_device / 张量 device= 参数）——
        swift 有 6+ 处把 current_device() 直接传给 torch.tensor(device=)，
        conformance 的 device_identity_consumable 强制此契约。"""
        raise NotImplementedError

    def memory_allocated(self, device=None):
        """口径=已分配字节，实时读。device 可选参对齐 torch 真实签名。"""
        raise NotImplementedError

    def mem_get_info(self, device=None):
        """(free_bytes, total_bytes) 元组契约——厂商侧结构/单位不同
        由 adapter 换算（如 XpuHw.memInfo 的 MiB dict -> 字节 tuple）。"""
        raise NotImplementedError

    def get_rng_state(self, device=None):
        """读 RNG 状态（checkpoint save）。契约见 set_rng_state：平台 torch
        栈的原生 state 对象，会被框架直接回喂 torch 原生 API。"""
        raise NotImplementedError
