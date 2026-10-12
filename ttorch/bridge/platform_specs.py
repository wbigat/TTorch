"""平台规格表：轻量元数据 + 零厂商依赖的 detect + 实现模块路径（惰性）。

为什么 spec 与实现分文件（2026-10-12 定稿）：
进程里最终只激活一个 platform，但 register 必须登记全部候选。若 spec 与
实现同处一模块，"注册"就等于 import 了所有厂商实现——代价是真实的：

- **厂商包 import 有副作用**：KL3 栈的 torch_xmlir 一经 import 就执行
  SYMBOL_REWRITE（torch/deepspeed/transformers）并加载 libbkcl/XCCL
  （本机日志可见）。在非该硬件的机器上，轻则污染进程、重则 ImportError；
- **PrivateUse1 栈的设备模块是运行时注册的**：torch.xpu / torch.npu 由厂商包
  经 rename_privateuse1_backend + _register_device_module 注册出来，
  类属性 `mod = getattr(torch, "xpu", None)` 在 import 期求值必然拿到 None。

因此按管理面/适配面切开（与 TT_Bridge_Deep_Dive.md §0 的分层一致）：

- **管理面（本文件）**：detect「只看不初始化」。只允许 importlib.util.find_spec
  （查找但不导入）、torch 自带 API、文件系统/设备节点探测；
  **禁止 import 任何厂商包**——这是本文件的硬约束。
- **适配面（adapters_real.py）**：实现本体，由 platform_mgr._activate()
  在平台确定之后才按 `impl` 路径导入。

detect 保持真实函数引用（code-first 的"拼写即炸"不丢），只有实现本体降级为
字符串路径——这是惰性必须付出的最小代价，且 `impl` 的解析错误在激活期即暴露。

身份（name/vendor_name/device_name/masquerades_as）以本表为唯一真相，激活期
由 manager 注入实例，实现类不自报——与 CAP_SCHEMA 中 identity 档"adapter
不许自报（防冒充）"的既有设计一致。
"""
import importlib.util
from dataclasses import dataclass, field
from typing import Callable, FrozenSet

import torch

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


@dataclass(frozen=True)
class PlatformSpec:
    """平台规格：注册期可得的全部轻量信息，不含实现。"""

    name: str
    vendor_name: str          # 厂商身份：provider 分发 / 门禁 / provenance
    device_name: str          # torch 视角的设备类型：框架 device.type 比较的存活线
    impl: str                 # "module:Class"，激活期才 import
    detect: Callable          # () -> (present, evidence)，零厂商 import
    masquerades_as: FrozenSet[str] = field(default_factory=frozenset)
    auto_detect: bool = True


# ------------------------------------------------------------------
# detect 实现：只看不初始化
# ------------------------------------------------------------------

def detect_cuda():
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


def detect_xpu_masquerade():
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


def detect_xpu_native():
    """torch.xpu 原生栈。注意 torch.xpu 可能由厂商包运行时注册——
    这里每次调用都重新 getattr，不缓存（import 期求值正是要避免的坑）。"""
    m = getattr(torch, "xpu", None)
    if m is None:
        return False, "torch.xpu 不存在（torch<2.4 或未装 torch_xpu）"
    try:
        ok = bool(m.is_available())
    except Exception as e:
        return False, f"torch.xpu.is_available() 异常: {e}"
    return (True, f"torch.xpu device: {m.get_device_name(0)}") if ok \
        else (False, "torch.xpu.is_available()=False")


def detect_cpu():
    return False, "dev-only 平台，不参与自动检测"


# ------------------------------------------------------------------
# 规格表：注册入口
# ------------------------------------------------------------------

CUDA = PlatformSpec(
    name="cuda", vendor_name="nvidia", device_name="cuda",
    impl="ttorch.bridge.adapters_real:TorchCudaPlatform", detect=detect_cuda)

XPU_MASQUERADE = PlatformSpec(
    name="xpu-masquerade", vendor_name="kl3-xpu", device_name="cuda",
    impl="ttorch.bridge.adapters_real:TorchXpuMasqueradePlatform",
    detect=detect_xpu_masquerade, masquerades_as=frozenset({"cuda"}))

XPU_NATIVE = PlatformSpec(
    name="xpu-native", vendor_name="xpu-native", device_name="xpu",
    impl="ttorch.bridge.adapters_real:TorchXpuNativePlatform", detect=detect_xpu_native)

CPU = PlatformSpec(
    name="cpu", vendor_name="cpu-dev", device_name="cpu",
    impl="ttorch.bridge.adapters_real:TorchCpuPlatform", detect=detect_cpu,
    auto_detect=False)  # dev-only，只能显式 select("cpu")

REAL_PLATFORMS = (CUDA, XPU_MASQUERADE, XPU_NATIVE, CPU)
