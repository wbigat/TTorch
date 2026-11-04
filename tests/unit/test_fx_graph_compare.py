"""fx 图对比的判别力与契约回归（A5，复核 20261028 §4.5）。

背景：旧实现（_fx_ssa_compare / ssa_equal）只比较 call_function 的 target
名称有序列表，丢掉常量、维度参数与 def-use 边。实测假阳性：eps 1e-6->0.1、
归约维 -1->0 都仍判 True，实际输出误差 0.25 / 0.91；另有假阴性：mean vs
sum/N 数学等价但序列不等。本文件把这些反例入库，并钉住"图不等不拒绝"的契约。
"""
import torch

from ttorch.provider.spec import Provider, ProviderImpl


class _Nat(ProviderImpl):
    name, kind = "native:fxprobe", "reference"

    def __init__(self, variant="correct"):
        self.variant = variant

    def __call__(self, x, w, eps=1e-6):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        if self.variant == "eps_wrong":            # 只改常量
            var = x32.pow(2).mean(-1, keepdim=True)
            return w.to(in_dtype) * (x32 * torch.rsqrt(var + 0.1)).to(in_dtype)
        if self.variant == "dim_wrong":            # 只改归约维度
            var = x32.pow(2).mean(0, keepdim=True)
            return w.to(in_dtype) * (x32 * torch.rsqrt(var + eps)).to(in_dtype)
        if self.variant == "equiv_reorder":        # 数学等价、序列不同
            var = x32.pow(2).sum(-1, keepdim=True) / x32.shape[-1]
            return w.to(in_dtype) * (x32 * torch.rsqrt(var + eps)).to(in_dtype)
        var = x32.pow(2).mean(-1, keepdim=True)
        return w.to(in_dtype) * (x32 * torch.rsqrt(var + eps)).to(in_dtype)


class _P(Provider):
    semantic, slot, module_names = "fxprobe", "compute", ("FxProbeNorm",)

    def __init__(self, variant="correct"):
        self._v = variant
        super().__init__()

    def build_native(self):
        return _Nat(self._v)

    @classmethod
    def probe_input(cls, ref=None, device=None):
        return (torch.randn(4, 8), torch.ones(8)), {}


class RefNorm(torch.nn.Module):
    """参照模块：标准 RMSNorm，eps=1e-6，归约 -1 维。"""

    def __init__(self, d=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(d) + 0.5)
        self.eps = 1e-6

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        return self.weight.to(in_dtype) * (x32 * torch.rsqrt(var + self.eps)).to(in_dtype)


def _cmp(variant, x=None):
    mod = RefNorm()
    x = torch.randn(4, 8) if x is None else x
    return _P(variant)._fx_graph_compare(mod, mod.eps, x, ref_forward=mod.forward)


def test_detects_constant_difference():
    """eps 常量差异必须被识别（旧实现判 True，实际误差 0.25）。"""
    fx = _cmp("eps_wrong")
    assert fx["applicable"]
    assert fx["op_seq_equal"] is True, "名称序列本就相同，这正是旧实现的盲区"
    assert fx["norm_equal"] is False, "规范化比较应识别出 eps 常量差异"


def test_detects_reduction_dim_difference():
    """归约维度差异必须被识别（旧实现判 True，实际误差 0.91）。"""
    fx = _cmp("dim_wrong")
    assert fx["applicable"]
    assert fx["norm_equal"] is False, "规范化比较应识别出归约维度差异"


def test_correct_variant_reports_equal():
    """正确实现：名称序列与规范化结构都应判等（防过度收紧造成假拒绝）。"""
    fx = _cmp("correct")
    assert fx["applicable"]
    assert fx["op_seq_equal"] is True
    assert fx["norm_equal"] is True, "正确实现被判结构不等，说明规范化过严"


def test_equivalent_reorder_is_known_false_negative():
    """数学等价但序列不同：仍判不等——这是已知且不修的假阴性。

    契约：图不等**不构成拒绝理由**，所以该假阴性无害。本测试钉住这一事实，
    防止后人误把图对比接入接管裁决。
    """
    fx = _cmp("equiv_reorder")
    assert fx["applicable"]
    assert fx["op_seq_equal"] is False
    assert fx["norm_equal"] is False


def test_result_is_dtype_local():
    """图随 dtype 变化，结论只对 probe_dtype 成立。"""
    f32 = _cmp("correct", torch.randn(4, 8))
    bf16 = _cmp("correct", torch.randn(4, 8, dtype=torch.bfloat16))
    assert f32["probe_dtype"] == "torch.float32"
    assert bf16["probe_dtype"] == "torch.bfloat16"
    assert len(bf16["native_ops"]) > len(f32["native_ops"]), \
        "bf16 下应多出 _to_copy 节点（fp32 下 upcast 被消除）"


def test_empty_graph_is_not_applicable():
    """空图不是"结构等价"：旧实现 any([])=False 且 []==[] 会报通过（真空真理）。"""
    class _Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))
            self.eps = 1e-6

        def forward(self, x):
            return x  # 无 call_function 节点

    mod = _Empty()
    fx = _P("correct")._fx_graph_compare(mod, mod.eps, torch.randn(4, 8),
                                         ref_forward=mod.forward)
    assert fx["applicable"] is False
    assert "无 call_function" in fx["reason"]


def test_grad_context_does_not_flip_result():
    """autograd 伪影剥离：同一正确实现在不同 grad 上下文下结论必须一致。

    实测伪影：对 requires_grad 的 Parameter 建图会插入 aten.detach.default，
    使序列长度变化——曾造成"看似检出了数据依赖错误"的误判。
    """
    x = torch.randn(4, 8)
    a = _cmp("correct", x)
    mod = RefNorm()
    mod.weight.requires_grad_(False)
    b = _P("correct")._fx_graph_compare(mod, mod.eps, x, ref_forward=mod.forward)
    assert a["norm_equal"] == b["norm_equal"] is True
    assert a["op_seq_equal"] == b["op_seq_equal"] is True
