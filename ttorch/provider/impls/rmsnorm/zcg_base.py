"""zcg（zero_centered_gamma）变体 RMSNorm 的 provider 骨架。

zcg 语义：y = normalized(x) * (1 + w)——Gemma 系 / Qwen3.5 / Qwen3-Next 的
RMSNorm 变体。与标准 RMSNormProvider（w 直乘）**一对一分离**：语义在设计时
确定（1+w），不做运行时变体识别；self_check 精度兜底自然分流——标准 provider
验证 w 直乘，遇 zcg 模块精度不一致拒绝；本 provider 验证 1+w，zcg 模块一致
才接管。

当前只 native 实现（功能底线 + 精度/收益基准）；高性能 impl 后续 qwen3_5
case 来了再补（挂到本 provider 的 impls）。
"""
import torch

from ttorch.provider.spec import ProviderImpl
from ttorch.provider.impls.rmsnorm.base import RMSNormProvider, _EPS


class ZcgNativeRMSNorm(ProviderImpl):
    """zcg 变体的 native：y = normalized(x) * (1 + w)。

    精度姿态对齐社区原始实现（transformers Qwen3_5RMSNorm，2026-10-26 经 fx
    SSA 校验修正）：归一化 upcast fp32，**(1+w) 与缩放乘都在 fp32 域**（原始
    是 `output * (1.0 + weight.float())` 后 `type_as`），最后才 cast 回输入
    dtype。此前 (1+w) 与乘法在输入 dtype 域——SSA 对比识别出该结构差异
    （数值差在 bf16 自检阈值内，但 native 是精度基准，应复刻原始语义）。
    """

    name = "native:rmsnorm_zcg"
    kind = "reference"
    intent = "zcg RMSNorm torch 原生流程（fp32 域 1+w 与缩放乘，对齐社区原始语义）；功能底线 + 精度/收益基准"

    def __call__(self, x, w, eps=_EPS):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        # 求值顺序对齐原始实现（norm 完成后再算 1+w 并乘）——fx SSA 逐算子一致
        normed = x32 * torch.rsqrt(var + eps)
        return (normed * (1 + w.float())).to(in_dtype)


class ZcgRMSNormProvider(RMSNormProvider):
    """zcg 变体 RMSNorm 的 provider（骨架：当前只 native，高性能 impl 后续补）。

    module_names 继承标准 provider（都匹配 *RMSNorm）——靠各自 self_check 的
    精度兜底分流：语义在设计时固定，谁与模块精度一致谁接管，注册顺序不影响
    正确性。probe_input 复用标准（输入形态 (x, w, eps) 相同）。
    """

    semantic = "rmsnorm_zcg"
    zero_centered_gamma = True   # zcg 变体（1+w）：据此与标准 provider 分流

    def build_native(self):
        return ZcgNativeRMSNorm()
