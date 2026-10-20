"""rope_bshd 语义包的组装入口（impls.build_all 调用）。"""
from ttorch.provider.impls.rope_bshd.base import RoPEBshdProvider


def build_provider():
    p = RoPEBshdProvider()
    from ttorch.provider.impls.rope_bshd.kl3 import KL3FusedRoPEBshd
    p.add_impl(KL3FusedRoPEBshd())
    return p
