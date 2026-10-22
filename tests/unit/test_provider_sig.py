"""D2-A：负载签名的 device 收敛为 type（卡序号不入键）的验证。

根因（2026-10-22 修复前）：`_call_sig` 用 `str(v.device)` 入签名 → rank0 得
`cuda:0`、rank1 得 `cuda:1`，缓存键天然不同，各 rank 独立跑门禁、独立决定
chosen——跨 rank 选择分叉的结构性放大器。修复后 device 只取 `.type`。

卡序号对 kernel 选择无语义价值（同型号卡上同一语义的正确实现是同一个），
但携带序号会让同构集群各 rank 从不同的缓存键出发——这正是要消除的。

这些用例不依赖多卡：单机即可验证"序号被丢弃"这一性质（cuda:0 的签名分量
必须是 "cuda" 而非 "cuda:0"，旧实现会失败）。
"""
import torch

from ttorch.provider import manager


def _sig(*args, **kw):
    return manager.instance()._call_sig(args, kw)


def test_cuda_device_signature_drops_card_index():
    """cuda:0 的签名分量必须是 'cuda'，不是 'cuda:0'（旧实现会产出后者）。"""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("无 cuda 设备")
    x = torch.randn(2, 8, device="cuda")
    sig = _sig(x)
    # 张量分量: (ndim, 特征维, dtype, device_type)
    assert sig[0][3] == "cuda", f"device 序号未丢弃: {sig[0][3]}"
    assert ":" not in sig[0][3]


def test_cpu_device_signature_is_type():
    x = torch.randn(2, 8)
    assert _sig(x)[0][3] == "cpu"


def test_same_shape_dtype_same_signature():
    """同 shape/dtype/device_type 的两个张量 → 同签名（缓存命中语义）。"""
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    assert _sig(a) == _sig(b)


def test_feature_dim_still_distinguishes():
    """特征维（最后一维）仍是正确性载体，必须保留区分能力。"""
    a = torch.randn(4, 16)
    b = torch.randn(4, 32)
    assert _sig(a) != _sig(b)


def test_dtype_still_distinguishes():
    a = torch.randn(4, 16)
    b = torch.randn(4, 16, dtype=torch.float64)
    assert _sig(a) != _sig(b)


def test_seq_and_batch_dims_merged():
    """序列/批次维归并：只影响工作量不影响语义，新长度不应换键重跑门禁。"""
    a = torch.randn(4, 16)   # batch=4
    b = torch.randn(8, 16)   # batch=8，特征维相同
    assert _sig(a) == _sig(b)


def test_non_tensor_args_in_signature():
    """非张量参数（eps 等）按值入键。"""
    x = torch.randn(2, 8)
    assert _sig(x, eps=1e-5) != _sig(x, eps=1e-6)
