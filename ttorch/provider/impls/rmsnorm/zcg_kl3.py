"""KL3（xpu-masquerade）的厂商 AOT 融合 RMSNorm zcg 变体实现（第二层）。

zcg（zero_centered_gamma）：y = normalized(x) * (1 + w)——Gemma 系 / Qwen3.5 /
Qwen3-Next 的 RMSNorm 变体。厂商 kernel 不区分 zcg：厂商自家 RMSNormFunction
的做法就是 `weight + 1.0` 预折算后调同一个 rms_layer_norm（见 torch_xmlir
.nn.rms_norm.RMSNormFunction.forward）——本 impl 复刻同一姿态：fwd 传入 (1+w)，
bwd 得到的 d(1+w) 即 dw（链式法则 d/dw = d/d(w+1)），梯度正确。

训练态复用 kl3.py 的 _RMSNormFn（fwd/bwd 都走厂商 AOT kernel）。
"""
from ttorch.provider.impls.rmsnorm.kl3 import _RMSNormFn
from ttorch.provider.spec import ProviderImpl

_EPS = 1e-6


class KL3FusedZcgRMSNorm(ProviderImpl):
    """KL3 厂商融合 RMSNorm zcg 变体（可训练）：(1+w) 预折算 + rms_layer_norm。

    与厂商 RMSNormFunction(zero_centered_gamma=True) 同源同姿态（weight+1.0
    后调同一 kernel），精度/性能口径与 KL3FusedRMSNorm 一致。
    """

    name = "vendor-fused:rmsnorm_zcg"
    kind = "fused_aot_vendor"
    platforms = frozenset({"xpu-masquerade"})
    min_speedup = 1.0  # 收益门禁生效：至少不劣于 zcg native 分解才启用
    intent = "KL3 厂商 AOT 融合 RMSNorm zcg 变体（1+w 预折算，同源 kernel）"

    def __call__(self, x, w, eps=_EPS):
        return _RMSNormFn.apply(x, w + 1.0, eps)
