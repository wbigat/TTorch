"""mock demo 的 Bridge 命名空间：真实模块转发自 ttorch，mock 留本地。

打包迁移后（2026-10-12）真实实现的唯一来源是 `ttorch` 包，本包只保留
**mock demo 专属**资产：
- adapters.py：假硬件 adapter（CudaPlatform/XpuPlatform/GcuPlatform/
  NaiveXpuPlatform）+ 其 PlatformSpec（CUDA_SPEC/XPU_SPEC/GCU_SPEC/...）
- bindings.py：针对 mock vendor_cuda 的 policy 表

转发用 sys.modules 别名而非 `from X import *`：`from tt_bridge import
manager` 这类子模块导入必须有真实的 sys.modules 条目才成立，
且别名保证 demo 与正式包操作的是**同一份模块对象**（manager 的
单例持有激活态，复制一份会让状态分裂）。

来源分两个包（2026-10-12 迁移；2026-10-13 platform 内部按功能拆分，模块改名）：
- 平台相关 -> ttorch.platform
  （platform_adapter 现名 platform_handle、platform_specs 现名 spec；
   platform_mgr 按功能拆成 manager「管理面」+ interface「torch 接口暴露」）
- CUDA 兼容 -> ttorch.bridge（binding_registry/admission/cuda_compat）
"""
import importlib
import sys

# tt_bridge 别名 -> ttorch.platform 目标模块。platform 分层后模块名变了，
# 别名保持不变，使 mock 代码 `from tt_bridge.platform_adapter import ...` 仍可用。
# 注意 platform_mgr 已按功能拆分，mock demo 需按用途分别用 manager / interface，
# 不再提供 platform_mgr 这个混合别名（它正是被拆掉的那层）。
_PLATFORM_ALIAS = {
    "platform_adapter": "platform_handle",
    "platform_specs": "spec",
    "manager": "manager",
    "interface": "interface",
    "conformance": "conformance",
}
_FROM_BRIDGE = ("binding_registry", "admission", "cuda_compat")

for _alias, _target in _PLATFORM_ALIAS.items():
    _mod = importlib.import_module(f"ttorch.platform.{_target}")
    sys.modules[f"{__name__}.{_alias}"] = _mod
    globals()[_alias] = _mod
for _name in _FROM_BRIDGE:
    _mod = importlib.import_module(f"ttorch.bridge.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

# 两个不在此列，理由相同（都是"真实环境专属，mock demo 不该加载"）：
# - adapters/ 实现目录：惰性导入目标（spec 的 impl 路径），eager import
#   会抵消惰性设计——厂商实现应在平台确定后才加载；
# - module_binding：真实 torch.cuda 的 policy 表，import 即在模块级
#   register_binding 14 条真实条目——mock demo 用 vendor_cuda 假模块，
#   加载它会把真实 policy 混进全局注册表，污染 demo 的盘点与快照。
# 需要它们的代码走 ttorch.platform.adapters.* / ttorch.bridge.module_binding。

from ttorch.platform.platform_handle import TTPlatformError  # noqa: E402,F401
