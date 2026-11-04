"""B1 端到端验证：真实 mcore_bridge 1.5.1 + 真实 megatron 上游 + 真实 CP 进程组。

两个模式（各自独立进程，因为补丁改写闭包 cell 不可回滚）：

  repro   不打 TT 补丁：证明缺陷在**真实链路**上稳定复现
          （mcore_bridge 的 wrapper pop 掉 cp_group → 上游硬 raise ValueError）。
  verify  打 TT 补丁：同一调用成功，且输出与**独立参考实现**（纯 torch 手算
          RoPE 公式，不经过任何被测代码路径）逐位一致。

运行（从仓库根；**必须 PYTHONPATH 指向仓库**——site-packages 里有旧版
ttorch 安装，不带 framework_patch，会静默用错代码）：
  PYTHONPATH=/root/dl/TTorch python tests/e2e/b1_thd_rope_e2e.py repro
  PYTHONPATH=/root/dl/TTorch python tests/e2e/b1_thd_rope_e2e.py verify
  # 多卡 CP 模式（真实 zigzag 跨 rank 切片，XPU 张量 + nccl/BKCL）：
  PYTHONPATH=/root/dl/TTorch torchrun --nproc_per_node=2 --master_port=29531 \
      tests/e2e/b1_thd_rope_e2e.py repro-cp
  PYTHONPATH=/root/dl/TTorch torchrun --nproc_per_node=2 --master_port=29531 \
      tests/e2e/b1_thd_rope_e2e.py verify-cp

设计要点：
- gloo 后端 + world_size=1 + CP=1——全程 CPU 张量，**零设备触碰**
  （上游 `_get_thd_freqs_on_this_cp_rank` 的 .cuda() 只在 cp_size>1 分支）。
- freqs.shape[0] != cu_seqlens[-1] 强制走 non-batched fallback（缺陷路径）。
- CP>1 的跨 rank 切片正确性不属本验证（需多卡，见待办 E13）。
"""
import os
import sys

import torch
import torch.distributed as dist

T, H, D = 8, 2, 8          # 两个序列 3+5，2 头，head_dim 8（全旋转，无 pass 段）
CU_SEQLENS = torch.tensor([0, 3, 8], dtype=torch.int32)
MAX_S = 5                  # != cu_seqlens[-1]=8 → 强制 non-batched fallback


def _setup_dist():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(context_parallel_size=1)
    return mpu.get_context_parallel_group()


def _inputs():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(T, H, D, generator=g)
    freqs = torch.randn(MAX_S, 1, 1, D, generator=g)
    return t, CU_SEQLENS.clone(), freqs


def _reference_rope_thd(t, cu_seqlens, freqs):
    """独立参考实现：非交错 RoPE，CP=1（offset=0，每序列取 freqs[:seqlen]）。

    与上游 rope_utils 的 CASE 2 同公式，但完全手写——不 import 任何被测函数。
    """
    outs = []
    for i in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        x = t[s:e]                                  # (s_i, h, d)
        f = freqs[: e - s].squeeze(1).squeeze(1)    # (s_i, d)
        cos, sin = torch.cos(f), torch.sin(f)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        outs.append(x * cos.unsqueeze(1) + rot * sin.unsqueeze(1))
    return torch.cat(outs, dim=0)


def _call_thd(cp_group):
    from megatron.core.models.common.embeddings import rope_utils

    t, cu, freqs = _inputs()
    return rope_utils._apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=cp_group)


def mode_repro():
    cp_group = _setup_dist()
    import swift.megatron  # noqa: F401  触发 mcore_bridge patcher（缺陷在场）

    try:
        _call_thd(cp_group)
    except ValueError as e:
        assert "cp_group must be provided" in str(e), f"意料外的错误: {e}"
        print("B1-REPRO PASS：真实链路复现缺陷（上游 ValueError，cp_group 被 pop 丢失）")
        return
    print("B1-REPRO FAIL：未打补丁却没报错——缺陷未复现（mcore_bridge 版本已修？）")
    sys.exit(1)


def mode_verify():
    cp_group = _setup_dist()
    import swift.megatron  # noqa: F401

    from ttorch.bridge import framework_patch as fw
    from ttorch.bridge import patches

    patches.register_all()
    status = fw.install(verbose=False)
    dotted = patches._THD_DOTTED
    assert status.get(dotted) == "applied", f"补丁未生效: {status}"

    out = _call_thd(cp_group)
    ref = _reference_rope_thd(*_inputs())
    assert out.shape == ref.shape == (T, H, D)
    max_err = (out - ref).abs().max().item()
    assert torch.equal(out, ref), f"与独立参考实现不一致: max_abs_err={max_err}"
    print(f"B1-VERIFY PASS：fallback 路径修复生效，输出与独立参考逐位一致 "
          f"(max_abs_err={max_err})")


# ---------------- 多卡 CP 模式（torchrun 启动） ----------------
#
# 全局两条序列，长度 4C 与 6C（C=cp_size，保证 zigzag 半块 2C 整除）。
# zigzag 负载均衡：序列切成 2C 块，rank r 拿第 r 块与第 2C-1-r 块。
# freqs 长度 6C ≠ 本地 cu_seqlens[-1]=10 → 强制 non-batched fallback（缺陷路径）。


def _zigzag_idx(seq_len, cp_size, cp_rank):
    """rank `cp_rank` 在长度为 seq_len 的序列上持有的全局 token 位置。"""
    seg = seq_len // (2 * cp_size)
    first = range(cp_rank * seg, (cp_rank + 1) * seg)
    second = range(seq_len - (cp_rank + 1) * seg, seq_len - cp_rank * seg)
    return list(first) + list(second)


def _cp_inputs(cp_size, cp_rank, device):
    """每 rank 构造本地 shard；返回 (t_local, cu_local, freqs, expected_local)。

    expected_local 由全局序列的手写参考 RoPE 按 zigzag 位置切出——
    与被测路径零共享代码。
    """
    g = torch.Generator().manual_seed(0)           # 所有 rank 同一全局输入
    la, lb = 4 * cp_size, 6 * cp_size
    t_global = torch.randn(la + lb, H, D, generator=g)
    freqs = torch.randn(lb, 1, 1, D, generator=g)  # max_s = 6C

    idx_a = _zigzag_idx(la, cp_size, cp_rank)
    idx_b = _zigzag_idx(lb, cp_size, cp_rank)
    ref = torch.cat([
        _reference_rope_thd(t_global[:la], torch.tensor([0, la]), freqs)[idx_a],
        _reference_rope_thd(t_global[la:], torch.tensor([0, lb]), freqs)[idx_b],
    ], dim=0)

    t_local = torch.cat([t_global[idx_a], t_global[la:][idx_b]], dim=0).to(device)
    # 上游契约（rope_utils.py:204）：cu_seqlens 传**全局**值，上游内部
    # 自行 // cp_size 得到本地段长。x 是本地 shard（总长 = 全局总长 / C）。
    cu_global = torch.tensor([0, la, la + lb], dtype=torch.int32, device=device)
    return t_local, cu_global, freqs.to(device), ref.to(device)


def _setup_dist_cp():
    """torchrun 多进程：nccl 后端（KL3 栈 masquerade 到 BKCL）+ XPU 张量。"""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(context_parallel_size=world)
    return rank, world, mpu.get_context_parallel_group()


def _run_cp(with_patch):
    from megatron.core.models.common.embeddings import rope_utils as ru

    orig_upstream = ru._apply_rotary_pos_emb_thd     # mcore_bridge 替换前留存
    rank, cp_size, cp_group = _setup_dist_cp()

    import swift.megatron  # noqa: F401  mcore_bridge patcher 替换模块符号

    if with_patch:
        from ttorch.bridge import framework_patch as fw
        from ttorch.bridge import patches

        patches.register_all()
        status = fw.install(verbose=False)
        assert status.get(patches._THD_DOTTED) == "applied", f"补丁未生效: {status}"

    t, cu, freqs, expected = _cp_inputs(cp_size, rank, f"cuda:{rank % 4}")
    tag = f"[rank{rank}/{cp_size}]"

    if not with_patch:
        try:
            ru._apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=cp_group)
        except ValueError as e:
            assert "cp_group must be provided" in str(e)
            print(f"{tag} B1-REPRO-CP PASS：缺陷在真实多卡链路复现（ValueError）",
                  flush=True)
            dist.barrier()
            dist.destroy_process_group()
            return
        print(f"{tag} B1-REPRO-CP FAIL：未打补丁却未报错", flush=True)
        sys.exit(1)

    out_patched = ru._apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=cp_group)
    out_direct = orig_upstream(t, cu, freqs, cp_group=cp_group)   # 直调上游基准
    assert torch.equal(out_patched, out_direct), \
        f"{tag} 补丁路径与直调上游不一致 " \
        f"err={(out_patched - out_direct).abs().max().item()}"
    # 手写参考在 CPU 上算，与 XPU kernel 存在 fp32 ulp 级噪声（实测 ~2e-7，
    # 跨设备逐位不可比），用 atol 判据；A/B 同设备同 kernel 才要求逐位。
    err = (out_patched.cpu() - expected.cpu()).abs().max().item()
    assert torch.allclose(out_patched.cpu(), expected.cpu(), rtol=1e-5, atol=1e-6), \
        f"{tag} 与 zigzag 手写参考超差 err={err}"
    print(f"{tag} B1-VERIFY-CP PASS：cp_size={cp_size} zigzag 切片正确，"
          f"补丁路径 ≡ 直调上游（逐位）；vs 手写参考 err={err:.2e}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def mode_repro_cp():
    _run_cp(with_patch=False)


def mode_verify_cp():
    _run_cp(with_patch=True)


if __name__ == "__main__":
    modes = {"repro": mode_repro, "verify": mode_verify,
             "repro-cp": mode_repro_cp, "verify-cp": mode_verify_cp}
    assert len(sys.argv) == 2 and sys.argv[1] in modes, \
        "用法: b1_thd_rope_e2e.py [repro|verify|repro-cp|verify-cp]"
    modes[sys.argv[1]]()
