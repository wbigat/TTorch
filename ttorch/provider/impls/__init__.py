"""各模块语义的 provider 实现集合（一种语义一个子文件夹）。

自动注册：SEMANTICS 登记已知模块语义；build_all() 逐个 import 其包并调
build_provider() 组装（含 native + 各平台 impl），供 manager.activate() 注册。
新增一个模块语义的 provider = 建 impls/<语义>/ 文件夹（base.py 放 Provider 类型
+ native，<平台>.py 放 ProviderImpl），并在 SEMANTICS 登记。

各语义包 import 安全（厂商算子引用都写在 __call__ 方法体内），故 activate 期
eager 注册无副作用；惰性 import 优化暂缓（2026-10-14）。
"""
import importlib

# 已知模块语义（新增 provider 在此登记）
SEMANTICS = ("rmsnorm", "swiglu", "bias_swiglu", "rope_bshd")


def build_all():
    """构建全部已知模块语义的 Provider（自动注册的来源）。

    一个语义包可返回单个 Provider 或列表（一变体一 provider，如 rmsnorm 的
    标准 + zcg）——列表在此扁平化。
    """
    providers = []
    for sem in SEMANTICS:
        mod = importlib.import_module(f"ttorch.provider.impls.{sem}")
        built = mod.build_provider()
        if isinstance(built, (list, tuple)):
            providers.extend(built)
        else:
            providers.append(built)
    return providers
