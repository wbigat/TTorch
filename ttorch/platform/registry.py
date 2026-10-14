"""PlatformRegistry：平台注册表——只管"登记了哪些平台"。

职责边界（2026-10-13 收窄两次）：本类**只负责注册相关行为**——登记 / 摘除 /
查询 / 遍历 PlatformSpec。

不在这里的：
- "把 spec 变成实现实例"-> `PlatformSpec.load()`（impl/base_cls 是 spec 自己
  的字段，解释权归 spec；注册表不该伸手读）；
- 自动检测 / 激活流水线 / 生命周期 / 能力查询 -> `PlatformManager`
  （manager.py）。注册表不持有"当前激活的是谁"。

于是本类就是一张 name -> spec 的表加几个查询口，没有别的知识。
"""
from ttorch.platform.platform_handle import TTPlatformError
from ttorch.platform.spec import PlatformSpec

__all__ = ["PlatformRegistry"]


class PlatformRegistry:
    """name -> PlatformSpec 的登记表。"""

    def __init__(self):
        self._specs = {}

    # ------------------------------------------------------------------
    # 登记 / 摘除
    # ------------------------------------------------------------------
    def register(self, spec):
        """登记一个平台：只存元数据，不初始化、不导入实现。"""
        if not isinstance(spec, PlatformSpec):
            raise TTPlatformError(
                "platform",
                f"register 只接受 PlatformSpec，实际 {type(spec).__name__}")
        self._specs[spec.name] = spec
        return spec

    def unregister(self, name):
        """摘除登记（测试/调试用）。返回被摘除的 spec 或 None。"""
        return self._specs.pop(name, None)

    # ------------------------------------------------------------------
    # 查询 / 遍历
    # ------------------------------------------------------------------
    def get(self, name):
        if name not in self._specs:
            raise KeyError(f"platform '{name}' 未注册，可选: {self.names()}")
        return self._specs[name]

    def names(self):
        return sorted(self._specs)

    def items(self):
        """(name, spec) 对，供 manager 做检测遍历。"""
        return self._specs.items()

    def auto_detectable(self):
        """参与自动检测的 spec（测试夹具类 auto_detect=False，被排除）。"""
        return [s for s in self._specs.values() if s.auto_detect]

    def __contains__(self, name):
        return name in self._specs

    def __len__(self):
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())
