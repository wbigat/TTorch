"""mock demo 的 Bridge 命名空间：真实模块转发自 ttorch.bridge，mock 留本地。

打包迁移后（2026-10-12）真实实现的唯一来源是 `ttorch.bridge`，本包只保留
**mock demo 专属**资产：
- adapters.py：假硬件 adapter（CudaPlatform/XpuPlatform/GcuPlatform/NaiveXpuPlatform）
- bindings.py：针对 mock vendor_cuda 的 policy 表

转发用 sys.modules 别名而非 `from X import *`：`from tt_bridge import
platform_mgr` 这类子模块导入必须有真实的 sys.modules 条目才成立，
且别名保证 demo 与正式包操作的是**同一份模块对象**（platform_mgr 有
_REGISTRY/_active 全局状态，复制一份会让状态分裂）。
"""
import importlib
import sys

_FORWARDED = ("platform_adapter", "platform_specs", "platform_mgr",
              "registry", "bindings_torch", "admission", "conformance",
              "cuda_compat")

for _name in _FORWARDED:
    _mod = importlib.import_module(f"ttorch.bridge.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

# adapters_real 不在此列：它是惰性导入的目标（platform_specs 的 impl 路径），
# eager import 会抵消惰性设计——厂商实现应在平台确定后才加载。
# 需要它的代码走 ttorch.bridge.adapters_real，不经本转发层。

from ttorch.bridge.platform_adapter import TTPlatformError, TTStream  # noqa: E402,F401
