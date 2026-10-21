"""模拟训练框架：混用 from-import 与模块属性两种方式调 CUDA 接口。

from-import 会在 import 时把对象引用拷进本模块命名空间，
是普通 monkey-patch 的盲区，用来考验 cuda-compat 的扫尾机制。
"""
import vendor_cuda
from vendor_cuda import (Stream, can_device_access_peer, current_stream,
                         empty_cache, get_rng_state, graph_capture, is_available,
                         manual_seed, mem_get_info, memory_allocated,
                         profiler_start, range_push, set_device, set_rng_state,
                         stream, synchronize)


def init_device(local_rank):
    set_device(local_rank)               # from-import 陈旧引用
    return vendor_cuda.current_device()  # 模块属性引用


def pick_comm_path():
    """初始化期按两卡 P2P 可达性选通信路径。"""
    return "p2p-direct" if can_device_access_peer(0, 1) else "staging-buffer"


def memory_report():
    """训练控制流里的显存水位观测（state_query：动态只读）。"""
    return memory_allocated()


def annotated_step(fn):
    """行为扩展位：给训练 step 打 profiler 区间。"""
    range_push("train_step")
    return fn()


def all_reduce_hook():
    synchronize()                        # 训练里的同步点


def capture_step(fn):
    return graph_capture(fn)             # CUDA Graph 加速路径


def profile_step():
    profiler_start()


# ------------------------------------------------------------------
# 基础 SFT 流程（对照 swift 主链路真实形态，见 TT_Swift_Cuda_API_Inventory.md）
# ------------------------------------------------------------------

def pick_device_or_cpu():
    """rollout.py:406 形态：is_available 分支决定设备字符串。"""
    return "accelerator" if is_available() else "cpu"


def device_budget_gb():
    """patcher.py:518 形态：mem_get_info 初始化 per-device 预算。"""
    free, total = mem_get_info()
    return round(free / 1024**3, 1), round(total / 1024**3, 1)


def ddp_fencing():
    """megatron_lm_utils.py:686-704 形态：ddp 侧流双向 fencing。"""
    side = Stream()
    side.wait_stream(current_stream())   # 侧流等主流
    with stream(side):
        pass                             # 侧流上干活（mock 无实际 work）
    current_stream().wait_stream(side)   # 主流等侧流


def tensor_on_current_device(n):
    """megatron/utils/utils.py:214 形态：current_device() 当 device 标识用。"""
    dev = vendor_cuda.current_device()   # 模块属性引用
    return [0.0] * n, dev                # mock: 张量 + 设备标识元组


def sft_step():
    """最小 SFT step：存 RNG（checkpoint save）-> 同步点 -> ddp fencing
    -> 显存清理（rollout 形态）。"""
    ckpt = {"rng": get_rng_state()}      # megatron_lm_utils.py:107 形态
    synchronize()
    ddp_fencing()
    empty_cache()
    return ckpt


def resume_from(ckpt):
    """megatron_lm_utils.py:599 形态：resume 恢复 RNG 状态。"""
    set_rng_state(ckpt["rng"])
