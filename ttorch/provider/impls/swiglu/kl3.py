"""KL3（xpu-masquerade）的厂商融合 SwiGLU（第二层）。

厂商 kernel：`torch_xmlir.nn.swiglu.SwiGLUFunction`（autograd.Function，
fwd/bwd 都走 custom_ops.swiglu_forward/backward，含空张量特判——MoE expert
分到 0 token 的常态）。算子引用写在 __call__ 体内，非 KL3 机器 import 安全。
"""
import torch

from ttorch.provider.spec import ProviderImpl


class KL3FusedSwiGLU(ProviderImpl):
    """KL3 厂商融合 SwiGLU（可训练）：torch_xmlir.nn.swiglu.SwiGLUFunction。"""

    name = "vendor-fused:swiglu"
    kind = "fused_aot_vendor"
    platforms = frozenset({"xpu-masquerade"})
    min_speedup = 1.0  # 至少不劣于 native 分解才启用
    intent = "KL3 厂商融合 SwiGLU（SwiGLUFunction，fwd+bwd kernel）"

    def supports(self, y):
        # 厂商 kernel 要求最后一维偶数（劈半）
        return y.shape[-1] % 2 == 0

    def __call__(self, y):
        from torch_xmlir.nn.swiglu import SwiGLUFunction
        ori = y.shape
        out = SwiGLUFunction.apply(y.reshape(-1, ori[-1]))
        return out if len(ori) == 2 else out.view(*ori[:-1], -1)
