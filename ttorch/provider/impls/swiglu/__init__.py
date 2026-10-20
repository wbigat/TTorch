"""SwiGLU 语义包的组装入口（impls.build_all 调用）。"""
from ttorch.provider.impls.swiglu.base import SwiGLUProvider


def build_provider():
    p = SwiGLUProvider()
    from ttorch.provider.impls.swiglu.kl3 import KL3FusedSwiGLU
    p.add_impl(KL3FusedSwiGLU())
    return p
