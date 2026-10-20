"""bias_swiglu 语义包的组装入口（impls.build_all 调用）。"""
from ttorch.provider.impls.bias_swiglu.base import BiasSwiGLUProvider


def build_provider():
    p = BiasSwiGLUProvider()
    from ttorch.provider.impls.bias_swiglu.kl3 import KL3FusedBiasSwiGLU
    p.add_impl(KL3FusedBiasSwiGLU())
    return p
