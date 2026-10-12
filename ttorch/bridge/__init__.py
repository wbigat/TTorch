"""TT Bridge：功能接入层（Part 1）。

两大块（TT_Bridge_Deep_Dive.md §0）：
- cuda_compat     框架 CUDA 调用点的逐符号决策：remap / fallback / exception
- Platform Manager 多 platform 注册/自动检测/激活门禁 + per-platform adapter；
                  统一 runtime 接口是它的顺带产物（契约 platform_adapter +
                  门面函数 platform_mgr），不是独立模块

本包只放**真实平台**的实现；mock adapter 与 mock policy 属 demo 资产，
留在 tt-bridge-demo/tt_bridge/（adapters.py / bindings.py）。
"""
from ttorch.bridge import (admission, bindings_torch, conformance,  # noqa: F401
                           cuda_compat, platform_adapter, platform_mgr,
                           platform_specs, registry)
from ttorch.bridge.platform_adapter import (TTPlatformError,  # noqa: F401
                                            TTStream)

__all__ = ["admission", "bindings_torch", "conformance", "cuda_compat",
           "platform_adapter", "platform_mgr", "platform_specs", "registry",
           "TTPlatformError", "TTStream"]
