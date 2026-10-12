"""TT Bridge 真实环境激活入口（对齐 xswift_ext.auto_init 的挂载形态）。

两种用法：
1. 显式（launcher 内，推荐）：
       import tt_bridge.entry as tt_entry
       tt_entry.activate()          # 之后再 import swift / mcore_bridge
2. 懒加载钩子（sitecustomize / .pth 场景）：
       tt_entry.install_lazy_hook() # 首次 import swift/mcore_bridge 前自动激活

设计要点（真实环境教训沉淀）：
- **必须先激活再 import 框架**：框架里的 `from torch.cuda import X` 在 import 时
  快照符号；激活在前则快照到的就是门面，扫尾只兜底"激活前已被 import 的模块"
  （sweep_packages 白名单，见 cuda_compat.activate）。
- 平台 overlay：torch.cuda 为非功能桩的栈剔除能力类绑定（G11，AdamW 健康检查
  会被误导）；masquerade 栈 shim 真实工作，能力类保留。
- rank0-only 摘要：分布式下每个进程都会激活，摘要只打一份。
"""
import os
import sys

from tt_bridge import cuda_compat, platform_mgr

_ACTIVATED = False
_HOOK_INSTALLED = False

# 默认注册的真实平台（cpu 仅供本地 dev，auto_detect=False 不参与自动检测）
_DEFAULT_PLATFORMS = ("cuda", "xpu-masquerade", "xpu-native", "cpu")


def _rank0():
    dist = getattr(sys.modules.get("torch"), "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return os.getenv("LOCAL_RANK", "0") == "0"


def _register_defaults():
    from tt_bridge.adapters_real import (TorchCpuPlatform, TorchCudaPlatform,
                                         TorchXpuMasqueradePlatform,
                                         TorchXpuNativePlatform)
    for cls in (TorchCudaPlatform, TorchXpuMasqueradePlatform,
                TorchXpuNativePlatform, TorchCpuPlatform):
        platform_mgr.register(cls)


def activate(sweep_packages=(), platforms=_DEFAULT_PLATFORMS, extra_entries=(),
             platform=None):
    """一次性激活 Platform Manager + cuda-compat。幂等；返回激活的 platform 名。

    sweep_packages：除框架主包外需要扫尾 from-import 旧引用的包名（默认空——
    遵循"先激活后 import"时框架符号快照到的天然是门面，无需扫尾）。
    platform：调试 override（如 "cpu" 本地验证）；默认 None=自动检测。
    """
    global _ACTIVATED
    if _ACTIVATED:
        return platform_mgr.active().name

    _register_defaults()
    platform = platform or os.getenv("TT_PLATFORM") or None
    if platform is None:
        platform_mgr.auto_activate()
    else:
        platform_mgr.select(platform)

    from tt_bridge import bindings_torch  # noqa: 延迟到激活后，注册表隔离
    exclude_cap = platform_mgr.device_name() == "cpu"
    entries = bindings_torch.entries(exclude_capability=exclude_cap) + list(extra_entries)
    cuda_compat.activate(entries, default_deny=False,
                         sweep_packages=list(sweep_packages))

    _ACTIVATED = True
    if _rank0():
        caps = platform_mgr.capabilities()
        print(f"[tt_bridge] activated: vendor={platform_mgr.vendor_name()} "
              f"device_name={platform_mgr.device_name()} "
              f"device_spec={platform_mgr.device_spec()} "
              f"bound={len(entries)} symbols "
              f"(capability={'excluded' if exclude_cap else 'bound'})")
        print(f"[tt_bridge] capabilities: {caps}")
    return platform_mgr.active().name


def install_lazy_hook(targets=("swift", "mcore_bridge", "megatron")):
    """装 import 钩子：首次 import 目标框架前自动 activate()。

    与 xswift_ext.auto_init 同形态但刻意简化：单钩子、幂等、无 env 门禁
    （用到本函数即表示要激活）。钩子触发一次后自摘除。
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED or _ACTIVATED:
        return
    import builtins
    import importlib

    def _is_target(name):
        return any(name == t or name.startswith(f"{t}.") for t in targets)

    orig_import = builtins.__import__
    orig_import_module = importlib.import_module

    def lazy_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and _is_target(name) and not _ACTIVATED:
            activate()
        return orig_import(name, globals, locals, fromlist, level)

    def lazy_import_module(name, package=None):
        if package is None and _is_target(name) and not _ACTIVATED:
            activate()
        return orig_import_module(name, package)

    builtins.__import__ = lazy_import
    importlib.import_module = lazy_import_module
    _HOOK_INSTALLED = True
