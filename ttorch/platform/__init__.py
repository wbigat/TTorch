"""TT Platform：平台注册 / 检测激活 / 门禁 / 生命周期 / torch 接口暴露。

按功能分层，每个文件一个职责：

| 文件 | 角色 | 内容 |
|---|---|---|
| platform_base.py | 契约 | `PlatformBase`：纯 torch 社区接口（厂商实现子类） |
| spec.py      | 类型   | `PlatformSpec`：平台自述数据结构（元数据 + detect + 实现类） |
| platform_handle.py | 关联 | `PlatformHandle`：持有 base 实例 + spec，动态透出 |
| registry.py  | 注册   | `PlatformRegistry`：登记哪些 spec（不持激活态） |
| manager.py   | 运行态 | 自动检测 / 激活门禁 / 生命周期 / 能力查询（含进程级单例） |
| interface.py    | 暴露   | 把 active 平台的 torch 接口原样转发（policy 表绑这里） |
| conformance.py | 门禁 | 验实现是否履行契约 |
| adapters/    | 实现   | catalog.py 集中登记各平台（detect+SPEC），实现类在 `<平台>_adapter.py` |

**注册时机（2026-10-13）**：register 是纯元数据登记、无副作用，不必等到
activate()——import 本包即由 `adapters/catalog.py` 注册全部内置平台（登记
的是 impl 字符串）。生产链路只剩 `manager.auto_activate()` 一步。实现类在
`<平台>_adapter.py`，激活时 spec.load() 才 import，可自由 import 厂商包；
catalog 本身 **import 安全**（顶层不 import 厂商包，detect 只看不初始化）。

两个入口，各司其职：

    from ttorch.platform import manager, interface
    manager.auto_activate()   # 管理面：激活谁、当前是谁、能力如何
    interface.synchronize        # torch 接口面：policy 表的绑定目标
"""
from ttorch.platform import interface, manager, spec  # noqa: F401
from ttorch.platform.platform_handle import PlatformHandle, TTPlatformError  # noqa: F401
from ttorch.platform.platform_base import PlatformBase  # noqa: F401
from ttorch.platform.manager import PlatformManager  # noqa: F401
from ttorch.platform.registry import PlatformRegistry  # noqa: F401
from ttorch.platform.spec import PlatformSpec  # noqa: F401

# 触发内置平台的自动发现并注册（见上方"注册时机"）。放在末尾：adapters 需要
# 上面已就绪的 manager 单例来登记。
from ttorch.platform import adapters as _adapters  # noqa: F401,E402

__all__ = ["manager", "interface", "spec", "adapters",
           "PlatformBase", "PlatformSpec", "PlatformHandle",
           "PlatformRegistry", "PlatformManager", "TTPlatformError"]
