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

    def supports(self, input, bias=None, *args, **kw):
        """适用性：最后一维偶数（劈半）且非 fp8 输入路径。

        签名必须收下**全部真实调用形态**（4 卡 DP 整网验证实测捕获，2026-10-20）：
        megatron 以**位置参数**调用
        `bias_swiglu_impl(input, bias, fp8_input_store, cpu_offload_input)`，
        原签名 `(self, input, bias=None, **kw)` 只能吃关键字，真实训练里抛
        `TypeError: supports() takes from 2 to 3 positional arguments but 5
        were given`——整网直接起不来。与 G12"门面签名取框架签名的超集"同源：
        supports 是逐调用判定，必须扛住调用面的全部形态。

        fp8_input_store 在 supports 层就拒（而非 __call__ 里 assert）：让该
        调用按"不适用"回退 native，不是让训练崩。
        """
        if input.shape[-1] % 2 != 0:
            return False
        fp8 = args[0] if args else kw.get("fp8_input_store", False)
        return not fp8

    def __call__(self, input, bias, fp8_input_store=False, cpu_offload_input=False):
        from torch_xmlir.nn.swiglu import SwiGLUFunction
        assert not fp8_input_store, "fp8_input_store 未支持（厂商 kernel 无此路径）"
        ori = input.shape
        x = input.view(-1, ori[-1])
        if bias is not None:
            x = x + bias
        out = SwiGLUFunction.apply(x)
        return out if len(ori) == 2 else out.view(ori[0], ori[1], -1)
