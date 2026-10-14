"""RNG state 的类型契约辅助（全平台通用，不限 torch 加速栈）。

独立成文件的理由：CPU dev 平台也要用它，但 CPU 不属于"torch 设备模块范式"
（没有 torch.cpu 设备模块那套 API），不该为了拿两个函数去 import
_device_module 把加速栈的基类一并拖进来。
"""
import torch

from ttorch.platform.platform_handle import TTPlatformError

__all__ = ["rng_to_state", "rng_from_state"]


def rng_to_state(state_tensor):
    """RNG state 契约 = **torch 原生 ByteTensor**（不是 bytes）。

    契约修正（G15，2026-10-12 megatron 真实链路实测）：megatron 的 RNG
    tracker 把 get_rng_state() 的返回值**原样回喂给 torch 原生 API**，中间
    不经我们的 set_rng_state（`default_generator.set_state(new_state)` 期望
    ByteTensor）。即状态对象会**流出 TT 边界**被厂商/框架原生代码直接消费，
    因此返回值类型必须与被替换符号一致——这是 G12"签名超集"原则在**返回值**
    上的对应面：签名管入参，类型管出参。可移植性由各平台实现负责（都返回
    自家 torch 栈的 ByteTensor），而不是靠统一成 bytes。
    """
    return state_tensor


def rng_from_state(state):
    """接受 torch 原生 ByteTensor；bytes 仍兼容（旧 checkpoint 可读）。"""
    if isinstance(state, torch.Tensor):
        return state
    if isinstance(state, (bytes, bytearray)):  # 向后兼容：旧格式 checkpoint
        return torch.frombuffer(bytearray(state), dtype=torch.uint8)
    raise TTPlatformError(
        "invalid_rng_state",
        f"RNG state 契约=torch ByteTensor（兼容 bytes），实际 {type(state).__name__}")
