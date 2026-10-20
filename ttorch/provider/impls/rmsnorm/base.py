"""RMSNorm 模块语义的 provider（第一层）与 native 默认实现。

RMSNorm 语义：y = x * rsqrt(mean(x^2, dim=-1) + eps) * w
调用约定：(x: Tensor[S, D], w: Tensor[D], eps: float) -> Tensor[S, D]

native 全硬件共享一份（torch 原生单算子流程），是功能底线 + 精度/收益基准；
各硬件的高性能实现作为 ProviderImpl 子类放在同文件夹的 <平台>.py（如 kl3.py）。
"""
import torch

from ttorch.provider.spec import Provider, ProviderImpl

_EPS = 1e-6


class NativeRMSNorm(ProviderImpl):
    """torch 原生单算子流程（native）：x * rsqrt(mean(x^2)+eps) * w。

    永远可用、永远启用、不可禁用——摘掉全部高性能实现后的功能底线，
    同时是门禁的精度/收益基准。
    """

    name = "native:rmsnorm"
    kind = "reference"
    intent = "torch 原生单算子流程；功能底线 + 精度/收益基准"

    def __call__(self, x, w, eps=_EPS):
        # 对齐 HF RMSNorm 语义（Qwen2/3 等）：归一化 upcast 到 fp32 计算，
        # 缩放乘回原 dtype。native 是精度基准兼 kill switch 的回退落点——
        # 若全程在 x.dtype 里算，与被替换的原始模块不等价（bf16 实测
        # max_rel≈1.5e-2，超 5e-3 口径 3 倍），回退即精度漂移，且门禁拿
        # native 当基准、结构上永远测不出这个偏差。
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        return w.to(in_dtype) * (x32 * torch.rsqrt(var + eps)).to(in_dtype)


class RMSNormProvider(Provider):
    """RMSNorm 模块的高性能 provider：native（默认）+ 各硬件高性能实现。"""

    semantic = "rmsnorm"
    slot = "compute"
    module_names = ("RMSNorm",)  # Qwen2RMSNorm / Qwen3RMSNorm / LlamaRMSNorm 等

    def build_native(self):
        return NativeRMSNorm()

    @classmethod
    def probe_input(cls, ref=None, device=None):
        """探测输入：优先跟随首次真实调用现场（ref = 真实 (args, kwargs)），
        shape/dtype/device 与生产一致——门禁结论才可外推（基准第 5 条：对比
        先对齐变量）；且不在首个训练 step 中途白占一块固定大显存。

        无 ref（独立验证/测试）时退回固定形状 + device（由 manager 经
        handle.device_spec() 注入，不硬编码 "cuda"——激活门面后
        torch.cuda.is_available() 在 cpu 平台也报 True，会造错设备）。

        不动全局 RNG 种子（门禁在训练循环中途触发，manual_seed 会污染训练
        RNG）；native 与 impl 共用同一次构造的输入，相对比较无需定种子。
        """
        if ref is not None:
            args, kw = ref
            x, w = args[0], args[1]
            # 完整保留非张量参数（eps 等）与 kwargs——门禁必须在生产同语义的
            # 负载上测；丢 eps（真实 1e-2 vs 默认 1e-6）门禁结论就不对应生产。
            return (torch.randn_like(x), torch.ones_like(w)) + tuple(args[2:]), dict(kw)
        if device is None:
            device = "cpu"
        x = torch.randn(4096, 4096, device=device)
        w = torch.ones(4096, device=device)
        return (x, w), {}
