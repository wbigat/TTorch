"""RMSNorm 模块语义的 provider 装配：native（默认）+ 各平台高性能实现。

一种模块语义一个文件夹，内部放不同 platform 的 ProviderImpl（kl3.py、cuda.py…）。
惰性导入暂缓（2026-10-14）：build_provider() 显式组装该模块的 Provider，
由使用方注册进 manager。新 hardware 的新实现 = 加一个 <平台>.py 子类并在此
add_impl。
"""
from ttorch.provider.impls.rmsnorm.base import RMSNormProvider, NativeRMSNorm
from ttorch.provider.impls.rmsnorm.kl3 import KL3FusedRMSNorm

__all__ = ["RMSNormProvider", "NativeRMSNorm", "KL3FusedRMSNorm", "build_provider"]


def build_provider():
    """组装 RMSNorm provider：native（默认自带）+ 各平台高性能实现。"""
    provider = RMSNormProvider()
    provider.add_impl(KL3FusedRMSNorm())
    return provider
