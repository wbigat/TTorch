"""RMSNorm 模块语义的 provider 装配：native（默认）+ 各平台高性能实现。

一种模块语义一个文件夹，内部放不同 platform 的 ProviderImpl（kl3.py、cuda.py…）。
惰性导入暂缓（2026-10-14）：build_provider() 显式组装该模块的 Provider，
由使用方注册进 manager。新 hardware 的新实现 = 加一个 <平台>.py 子类并在此
add_impl。

本语义含**两个变体 provider**（一对一，语义设计时确定，靠 flag + 精度兜底
分流）：标准（w 直乘，RMSNormProvider）+ zcg（1+w，ZcgRMSNormProvider，
Gemma/Qwen3.5 变体）——build_provider 返回列表，build_all 扁平化注册。
"""
from ttorch.provider.impls.rmsnorm.base import RMSNormProvider, NativeRMSNorm
from ttorch.provider.impls.rmsnorm.kl3 import KL3FusedRMSNorm
from ttorch.provider.impls.rmsnorm.zcg_base import (
    ZcgRMSNormProvider, ZcgNativeRMSNorm)
from ttorch.provider.impls.rmsnorm.zcg_kl3 import KL3FusedZcgRMSNorm

__all__ = ["RMSNormProvider", "NativeRMSNorm", "KL3FusedRMSNorm",
           "ZcgRMSNormProvider", "ZcgNativeRMSNorm", "KL3FusedZcgRMSNorm",
           "build_provider"]


def build_provider():
    """组装 RMSNorm provider 族：标准（w 直乘）+ zcg 变体（1+w），各挂平台 impl。"""
    std = RMSNormProvider()
    std.add_impl(KL3FusedRMSNorm())
    zcg = ZcgRMSNormProvider()
    zcg.add_impl(KL3FusedZcgRMSNorm())
    return [std, zcg]
