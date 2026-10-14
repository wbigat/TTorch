"""checkpoint bridge：save / resume + state schema 校验 + RNG 状态。

TT_Bridge_Deep_Dive.md §1.4 把这一环列为"最容易被低估、最容易后期才
暴露"，要求 v0 就有。它不只是存盘读盘，至少要覆盖：

1. **state schema 校验**：resume 前核对 schema 版本、模型结构指纹、
   并行布局——对不上必须 fail fast，而不是让 load_state_dict 静默
   strict=False 地漏掉一半权重（那会表现为"resume 后 loss 跳变"，
   等训练跑完才发现）；
2. **RNG 状态**：设备 RNG 经 Bridge 门面存取（ByteTensor 契约），CPU RNG
   走 torch 原生。不存 RNG 的 resume，数据顺序与 dropout 都不连续，
   loss 必然跳变——xmegatron 的 CudaRNGStatesTracker 补丁证明这是
   真实必踩路径；
3. **resume 后 loss 连续性**：验收判据本身（见 verify_resume_continuity）。

当前穿透点范围：单卡、同并行度。**不做** reshard（并行布局变化时的
权重重切分）——那要等分布式打通，现在遇到直接 fail fast。
"""
import os

import torch

from ttorch.platform import interface, manager
from ttorch.platform.platform_handle import TTPlatformError

SCHEMA_VERSION = "tt-single-card-v1"


def _fingerprint(model):
    """模型结构指纹：参数名 + shape。resume 时结构必须一致。

    只取名字与形状、不取值——指纹要对"结构是否同一个模型"敏感，
    对"训练到第几步"不敏感。
    """
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(f"{name}:{tuple(p.shape)}".encode())
    return h.hexdigest()[:16]


def _rng_state():
    """RNG 快照：设备侧经 Bridge 门面（ByteTensor 契约），CPU 侧走 torch 原生。

    平台声明 rng_state=False 时门面报 unsupported——此处如实记 None 并
    在 resume 时告警，而不是假装存了（能力驱动降级，不静默伪装）。
    """
    try:
        device = interface.get_rng_state()
    except TTPlatformError as e:
        device = None
        print(f"[ckpt] 设备 RNG 不可用，降级: {e}")
    return {"cpu": torch.get_rng_state(), "device": device}


def save(path, model, optimizer, step, extra=None):
    """保存 checkpoint。torch.save 到临时文件再 rename —— 避免写一半被中断
    留下无法分辨的半截文件（resume 时才发现，代价高）。"""
    payload = {
        "schema": SCHEMA_VERSION,
        "fingerprint": _fingerprint(model),
        "platform": {"vendor": manager.activation().vendor_name,
                     "device": manager.activation().device_name},
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": _rng_state(),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[ckpt] saved step={step} -> {path} "
          f"(schema={SCHEMA_VERSION}, fp={payload['fingerprint']})")
    return payload["fingerprint"]


def load(path, model, optimizer, strict_platform=False):
    """恢复 checkpoint，返回 step。

    schema/指纹不符一律 fail fast（§1.4：state schema 校验是硬要求）。
    strict_platform：跨平台 resume 是合法需求（CUDA 存的权重在 XPU 上
    续训），默认只告警；要求同平台时置 True。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("schema") != SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint schema 不符: 期望 {SCHEMA_VERSION}，"
            f"实际 {payload.get('schema')}——拒绝加载（静默兼容会让"
            f"结构差异变成训练中后段才暴露的精度问题）")
    fp_now, fp_ckpt = _fingerprint(model), payload.get("fingerprint")
    if fp_now != fp_ckpt:
        raise RuntimeError(
            f"模型结构指纹不符: 当前 {fp_now} vs checkpoint {fp_ckpt}"
            f"（参数名或 shape 变了，strict 加载会漏权重）")

    saved_plat = payload.get("platform", {})
    if saved_plat.get("vendor") != manager.activation().vendor_name:
        msg = (f"[ckpt] 跨平台 resume: 存于 {saved_plat.get('vendor')}，"
               f"当前 {manager.activation().vendor_name}")
        if strict_platform:
            raise RuntimeError(msg + "（strict_platform=True）")
        print(msg + "（权重可移植，RNG 序列不保证逐位一致）")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])

    rng = payload.get("rng") or {}
    if rng.get("cpu") is not None:
        torch.set_rng_state(rng["cpu"])
    if rng.get("device") is not None:
        try:
            interface.set_rng_state(rng["device"])
        except TTPlatformError as e:
            print(f"[ckpt] 设备 RNG 恢复失败，降级: {e}")
    else:
        print("[ckpt] checkpoint 无设备 RNG，resume 后随机序列不连续")

    print(f"[ckpt] loaded step={payload['step']} <- {path}")
    return payload["step"]


def verify_resume_continuity(losses_ref, losses_resumed, tol=1e-4):
    """resume 正确性判据：从 step k 恢复后重跑，loss 必须与原轨迹吻合。

    这是 §1.4 点名的验收口径——"能加载"不等于"恢复正确"：漏掉
    optimizer state（动量）或 RNG，loss 都会跳变，但 load 本身不报错。
    返回 (ok, max_diff)。
    """
    n = min(len(losses_ref), len(losses_resumed))
    if n == 0:
        return False, float("inf")
    diffs = [abs(a - b) for a, b in zip(losses_ref[:n], losses_resumed[:n])]
    max_diff = max(diffs)
    return max_diff <= tol, max_diff
