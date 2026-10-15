"""ProviderRegistry：模块级 provider 的注册存储（第一层）。

注册单元是 Provider（一个模块语义一个），按 semantic 键控。存储封装：注册表
是实例私有 dict，对外不暴露可变表——只能经 register() 写入、get()/all()/names()
读，杜绝外部绕过查重直接改表。注册只登记，不执行（构建/门禁在选择期才发生）。
"""
from typing import Tuple

from ttorch.provider.spec import Provider

__all__ = ["ProviderRegistry"]


class ProviderRegistry:
    """模块级 provider 的注册表。进程内可实例化多个（测试隔离）。"""

    def __init__(self):
        self._providers = {}

    def register(self, provider: Provider) -> Provider:
        """逻辑注册一个模块 provider。semantic 重复 / impl 重名立即报错。"""
        assert provider.semantic, f"{type(provider).__name__} 未设 semantic（模块语义）"
        assert provider.semantic not in self._providers, \
            f"provider 重复注册 semantic: {provider.semantic}"
        # impl 名全局唯一（switch/观测按名索引）
        seen = {provider.native.name}
        assert provider.native.name, f"{provider.semantic} 的 native 未设 name"
        for impl in provider.impls:
            assert impl.name, f"{provider.semantic} 有 impl 未设 name"
            assert impl.name not in seen, f"impl 重名: {impl.name}"
            seen.add(impl.name)
        self._providers[provider.semantic] = provider
        return provider

    def get(self, semantic):
        return self._providers.get(semantic)

    def all(self) -> Tuple[Provider, ...]:
        """全部已注册 provider（只读快照）。"""
        return tuple(self._providers.values())

    def names(self):
        return sorted(self._providers)
