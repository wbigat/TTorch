"""ttorch.provider：module 级性能实现层（性能增量，不是功能接入）。

与 bridge 的分工：bridge 负责功能可达（torch-native），provider 负责性能增量
（厂商融合 kernel / 编译融合）。两者唯一接口是调用边界——provider 只读
platform manager 的激活平台名（attach_platform 注入），绝不碰 cuda-compat。

组成（两层）：
- spec.py       Provider（第一层：一个模块语义一个类型）+ ProviderImpl（第二层：
                provider 内、面向某平台的高性能实现）
- registry.py   ProviderRegistry（按 semantic 注册第一层 Provider）
- manager.py    ProviderManager（门禁/选择/分发/开关）+ 进程级单例
- impls/        各模块语义的实现集合（一种语义一个文件夹，内含 Provider 类型
                + native + 各平台 ProviderImpl）

provenance 与 bridge 共用统一事件流 ttorch.witness.provenance（layer="provider"）。
"""
from ttorch.witness import provenance  # noqa: F401  # 统一事件流（bridge/provider 共享）
from ttorch.provider.spec import Provider, ProviderImpl  # noqa: F401
from ttorch.provider.registry import ProviderRegistry  # noqa: F401
from ttorch.provider.manager import (  # noqa: F401
    ProviderManager, activate, register, register_impl, select, call, wrap_model,
    enable, disable, set_bypass, status, attach_platform, instance)

__all__ = [
    "Provider", "ProviderImpl", "ProviderRegistry", "ProviderManager",
    "activate", "register", "register_impl", "select", "call", "wrap_model",
    "enable", "disable", "set_bypass", "status", "attach_platform", "instance",
    "provenance",
]
