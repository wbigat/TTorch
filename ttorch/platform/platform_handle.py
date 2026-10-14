"""PlatformHandle：把 PlatformBase 的实现与 PlatformSpec 的描述关联起来。

它是一层很薄的句柄（handle），职责只有两件：
1. **身份对外呈现**：name / vendor_name / device_name / masquerades_cuda /
   auto_detect 一律取自注入的 spec——spec 是唯一真相，实现类无从自报
   （防冒充：身份在结构上就不可能来自实现侧）；
2. **接口透出**：torch 接口与平台自述行为（sdk_version / initialize /
   finalize）以及厂商扩展（range_push、mock 的 .hw 钩子）经 `__getattr__`
   动态转发给持有的 base 实例——**不逐个方法写委托样板**，base 加接口这里
   零改动。另有两个 handle 自己呈现的：`device_spec()`（身份 + 序号拼设备
   标识）与 `supports(name)`（接口存在性内省）。

组合而非继承（2026-10-13 定稿）：handle 持有 base + spec，构造时传入
（`PlatformHandle(base, spec)`，由 registry.create 装配）。好处是
"torch 接口实现"（base 子类，厂商写）与"身份描述"（spec，管理面写）彻底
分离，handle 自己没有实现、也不需要厂商子类。

分层：
- platform_base.py  PlatformBase  纯 torch 社区接口（厂商实现子类）
- spec.py     PlatformSpec     纯元数据 + detect（管理面，禁 import 厂商包）
- 本文件       PlatformHandle  句柄：关联两者，对外呈现
- registry.py PlatformRegistry 注册 + 按 spec 装配出 handle
- manager.py  PlatformManager  激活 / 生命周期 / 能力查询
"""

__all__ = ["TTPlatformError", "PlatformHandle"]


class TTPlatformError(Exception):
    """统一错误类别：厂商原生错误经实现类翻译成它，上层据此写可移植处理。"""

    def __init__(self, category, detail):
        self.category = category
        super().__init__(f"[{category}] {detail}")


class PlatformHandle:
    """句柄 = 一个 PlatformBase 实例 + 一份 PlatformSpec。

    双身份（缺一不可）由 spec 承载：
    - vendor_name  厂商身份：provider 分发 / 门禁 / provenance 用
    - device_name  torch 认得的设备类型：框架 `device.type == "cuda"` 这类
                   比较的存活线。冒充 CUDA 的栈（KL3/GCU/MACA）上
                   vendor_name="kl3-xpu" 但 device_name 必须是 "cuda"，
                   写成品牌名会让框架里的 device.type 比较静默失效。
    """

    def __init__(self, base, spec):
        self._base = base
        self._spec = spec

    # --- 关联的两端（显式暴露，便于门禁/诊断取证） ---
    @property
    def base(self):
        """torch 接口的实现本体（PlatformBase 子类实例）。"""
        return self._base

    @property
    def spec(self):
        """平台描述（身份元数据 + detect）。"""
        return self._spec

    # --- 身份：一律从 spec 读（实现类不自报） ---
    @property
    def name(self):
        return self._spec.name

    @property
    def vendor_name(self):
        return self._spec.vendor_name

    @property
    def device_name(self):
        return self._spec.device_name

    @property
    def masquerades_cuda(self):
        return self._spec.masquerades_cuda

    @property
    def auto_detect(self):
        return self._spec.auto_detect

    # --- 平台呈现 ---
    def supports(self, name):
        """本平台是否提供 `name` 接口（派生自实现，不经平台自报）。

        取代原 capability 表的"接口存在性"声明（stream_api / memory_query /
        rng_state）：事实由"子类有没有覆盖默认实现"直接给出，不会撒谎，也
        不需要 conformance 再去验证声明的真假。
        """
        from ttorch.platform.platform_base import PlatformBase
        return PlatformBase.supports(self._base, name)

    def device_spec(self, idx=None):
        """设备标识字符串（"xpu:0" / "cuda:0" 形态），供框架构造 torch.device。

        current_device() 的 int 返回值在 native xpu 上会被 torch 当 cuda 索引
        解析（torch.tensor(device=0) 的隐含约定）——设备构造必须走 device_name
        + 序号的完整 spec，这是双身份在框架侧的落地形态。
        """
        return f"{self.device_name}:{self.current_device() if idx is None else idx}"

    # --- 其余一切转发给 base ---
    def __getattr__(self, name):
        """torch 接口 + 平台自述 + 厂商扩展，全部转发到 base。

        只在常规属性查找失败时触发，所以上面的 property 优先级更高（身份
        不会被 base 抢答）。下划线开头直接 AttributeError：既挡住内部属性
        穿透，也避免 `self._base` 尚未赋值时的无限递归。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def __repr__(self):
        return (f"<PlatformHandle {self.name} vendor={self.vendor_name} "
                f"device_name={self.device_name} base={type(self._base).__name__}>")
