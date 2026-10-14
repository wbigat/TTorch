"""内置平台目录：全部已知平台的 detect + SPEC，import 即自动注册。

这些是**静态事实**（有哪些平台、各自身份、如何探测、实现在哪），集中一处
一眼看全，比分散到每平台一个文件好读。实现类在各自的 `<平台>_adapter.py`
（惰性加载，激活时 spec.load() 才 import，可自由 import 厂商包），这里只
登记 impl 字符串。

import 安全硬约定：本文件在注册期被 import（见 adapters/__init__.py），顶层
只 import torch + 共享辅助，detect 经 find_spec / torch API 探测（只看不
初始化），**禁止 import 任何厂商包**。

加一块新卡 = 本文件加一段（detect + SPEC）+ 一个 `<平台>_adapter.py` 实现。
"""
import importlib.util

import torch

from ttorch.platform import manager as _manager
from ttorch.platform.spec import PlatformSpec

__all__ = ["SPECS", "register_all",
           "CUDA", "XPU_MASQUERADE", "XPU_NATIVE", "CPU"]

# ------------------------------------------------------------------
# detect 共享辅助（只看不初始化，禁 import 厂商包）
# ------------------------------------------------------------------

# masquerade 探测标记：torch.cuda.get_device_name() 含这些子串时，
# 说明 torch.cuda 背后不是 NVIDIA（待真实环境校准的卡名清单）
_XPU_MARKERS = ("xpu", "kl3")

# KL3 厂商栈的 native 探测通道：torch_xmlir 包存在性。
# 真实环境校准结论（2026-10-12，KL3 机器实测）：卡名是通用的 "gpu"，
# _XPU_MARKERS 不命中；但该环境 torch_xmlir 随栈预装（libbkcl/XCCL 由其
# 加载），包存在性是厂商私有通道，不会被 NVIDIA 栈误命中。
_XPU_NATIVE_PACKAGES = ("torch_xmlir",)


def _installed(pkg):
    """包是否已安装——find_spec 只查找不导入，厂商包的副作用不会被触发。"""
    return importlib.util.find_spec(pkg) is not None


def _cuda_device_name():
    try:
        return torch.cuda.get_device_name(0).lower()
    except Exception:
        return ""


# ------------------------------------------------------------------
# 各平台：detect + SPEC（实现类在 <平台>_adapter.py，激活时才 import）
# ------------------------------------------------------------------

def _detect_cuda():
    """真 NVIDIA。compat claim 会被 masquerade 栈冒充，故加厂商包防御。"""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available()=False"
    for pkg in _XPU_NATIVE_PACKAGES:
        if _installed(pkg):
            return False, f"厂商包 {pkg} 已安装：疑似 masquerade 栈，cuda claim 不成立"
    name = _cuda_device_name()
    if any(k in name for k in _XPU_MARKERS):
        return False, f"CUDA API 存在但卡名 '{name}' 命中 XPU 标记（masquerade，不计为 NVIDIA）"
    return True, f"CUDA device: {name}"


def _detect_xpu_masquerade():
    """KL3/XPYTORCH 风格：torch.cuda.* 实际跑 XPU。"""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available()=False"
    hits = []
    for pkg in _XPU_NATIVE_PACKAGES:
        if _installed(pkg):
            hits.append(f"厂商包 {pkg} 已安装（native 通道）")
    name = _cuda_device_name()
    if any(k in name for k in _XPU_MARKERS):
        hits.append(f"卡名 '{name}' 命中 XPU 标记")
    if hits:
        return True, "torch.cuda 背后是 masquerade 栈：" + "；".join(hits)
    return False, f"卡名 '{name}' 无 XPU 标记，厂商包 {list(_XPU_NATIVE_PACKAGES)} 均未安装"


def _detect_xpu_native():
    """torch.xpu 原生栈。每次调用都重新 getattr，不缓存——torch.xpu 可能由
    厂商包运行时才注册，import 期求值正是要避免的坑。"""
    m = getattr(torch, "xpu", None)
    if m is None:
        return False, "torch.xpu 不存在（torch<2.4 或未装 torch_xpu）"
    try:
        ok = bool(m.is_available())
    except Exception as e:
        return False, f"torch.xpu.is_available() 异常: {e}"
    return (True, f"torch.xpu device: {m.get_device_name(0)}") if ok \
        else (False, "torch.xpu.is_available()=False")


def _detect_cpu():
    return False, "dev-only 平台，不参与自动检测"


CUDA = PlatformSpec(
    name="cuda", vendor_name="nvidia", device_name="cuda",
    detect=_detect_cuda, impl="ttorch.platform.adapters.cuda_adapter:TorchCudaPlatform")

XPU_MASQUERADE = PlatformSpec(
    name="xpu-masquerade", vendor_name="kl3-xpu", device_name="cuda",
    detect=_detect_xpu_masquerade, masquerades_cuda=True,
    impl="ttorch.platform.adapters.xpu_masquerade_adapter:TorchXpuMasqueradePlatform")

XPU_NATIVE = PlatformSpec(
    name="xpu-native", vendor_name="xpu-native", device_name="xpu",
    detect=_detect_xpu_native, impl="ttorch.platform.adapters.xpu_native_adapter:TorchXpuNativePlatform")

CPU = PlatformSpec(
    name="cpu", vendor_name="cpu-dev", device_name="cpu",
    detect=_detect_cpu, auto_detect=False,  # dev-only，只能显式 select("cpu")
    impl="ttorch.platform.adapters.cpu_adapter:TorchCpuPlatform")

SPECS = (CUDA, XPU_MASQUERADE, XPU_NATIVE, CPU)


def register_all():
    """把全部内置平台登记进 manager 的进程级单例。

    register 是纯元数据登记、无副作用，提前到 import 即可——生产链路只剩
    manager.auto_activate() 一步。
    """
    for s in SPECS:
        _manager.register(s)


# import 即注册（见模块 docstring）
register_all()
