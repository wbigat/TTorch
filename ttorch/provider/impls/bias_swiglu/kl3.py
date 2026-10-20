"""KL3（xpu-masquerade）的厂商融合 bias SwiGLU（第二层）。

厂商 kernel：`torch_xmlir.nn.swiglu.SwiGLUFunction`（bias 先行相加）。
算子引用写在 __call__ 体内，非 KL3 机器 import 安全。
"""
import torch

from ttorch.provider.spec import ProviderImpl


class KL3FusedBiasSwiGLU(ProviderImpl):
    """KL3 厂商融合 bias SwiGLU（可训练）：bias + SwiGLUFunction。"""

    name = "vendor-fused:bias_swiglu"
    kind = "fused_aot_vendor"
    platforms = frozenset({"xpu-masquerade"})
    min_speedup = 1.0
    intent = "KL3 厂商融合 bias SwiGLU（bias 相加后走 SwiGLUFunction）"

    def supports(self, input, bias=None, **kw):
        return input.shape[-1] % 2 == 0

    def __call__(self, input, bias, fp8_input_store=False, cpu_offload_input=False):
        from torch_xmlir.nn.swiglu import SwiGLUFunction
        assert not fp8_input_store, "fp8_input_store 未支持（厂商 kernel 无此路径）"
        ori = input.shape
        x = input.view(-1, ori[-1])
        if bias is not None:
            x = x + bias
        out = SwiGLUFunction.apply(x)
        return out if len(ori) == 2 else out.view(ori[0], ori[1], -1)
