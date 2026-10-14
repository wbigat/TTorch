"""Implementations：模块语义的候选实现集合。

本 demo 三个模块语义：
- rmsnorm（compute 插槽）：(x, w, eps) -> y，张量用 float list 模拟
- rope_bshd（compute 插槽，完整实例）：一个厂商融合实现 + 一个裸替换
  wrapper 的组合，见下文出处注释
- moe_block（schedule 插槽）：一组 expert thunk 的执行时序

三类实现（对照定稿）：
- reference       torch-native 基线（demo 用纯 Python 模拟 ATen 语义）。
                  永远可用、永远启用、不可被 switch 禁用——摘掉全部
                  Provider 后的功能底线。import 本模块即注册。
- Fused 算子 AOT   编译期固化，分两小类：
    - 厂商融合算子  厂商库黑盒 kernel：通常最快，但可能有平台限定、
                    shape 支持有限（本 demo 的 vendor fused 只支持 len=8，
                    运行时遇到不支持 shape 会抛厂商错误 -> 演示回退）
    - ABO 融合算子  图编译产物的 AOT build：无平台限定，收益次之
- Fused 算子 JIT   运行期编译：本 demo 的 JIT 实现中间累加精度不足
                   （模拟 fp16 截断），应被精度门禁拒绝

schedule 插槽（可选，缺省 serial）：不改变算子结果，只改变执行时序。
overlap 实现消费 Platform Manager（provider -> bridge 的唯一依赖方向：
provider 可读 platform 能力/接口，绝不碰 cuda-compat）。
"""
import math

from mock_hw import XpuHwError

import vendor_ops
from tt_bridge import manager
from tt_provider import provider_mgr

# ------------------------------------------------------------------
# rmsnorm：compute 插槽
# ------------------------------------------------------------------


def reference_rmsnorm(x, w, eps=1e-6):
    """(S,) fp32 -> (S,)。纯 Python 模拟 ATen 语义，功能底线。"""
    ms = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [v * inv * wi for v, wi in zip(x, w)]


def fused_aot_vendor_rmsnorm(x, w, eps=1e-6):
    """厂商融合算子（AOT 黑盒）：只支持 len=8，其余 shape 抛厂商错误。

    这是真实的厂商库形态：AOT kernel 为固定 shape 编译，遇到不支持的
    shape 不是算错而是报错——Provider Manager 必须能回退 reference。
    """
    if len(x) != 8:
        raise XpuHwError(f"vendor fused rmsnorm: unsupported shape len={len(x)}")
    return reference_rmsnorm(x, w, eps)  # 数值与 reference 一致（精度达标）


def fused_aot_vendor_cuda_rmsnorm(x, w, eps=1e-6):
    """厂商融合算子的 CUDA 版本：platforms={"cuda"}，在 xpu 机器上
    应在 selection 期被平台过滤，根本进不了门禁。"""
    return reference_rmsnorm(x, w, eps)


def fused_aot_abo_rmsnorm(x, w, eps=1e-6):
    """ABO 融合算子（图编译 AOT 产物）：无平台限定、无 shape 限制。"""
    return reference_rmsnorm(x, w, eps)


def fused_jit_rmsnorm(x, w, eps=1e-6):
    """JIT 融合算子：中间累加精度不足（模拟 fp16 截断）。

    功能上能跑、形状也对，只有数值偏——专用来证明精度门禁的必要性：
    功能测试拦不住它，精度测试（对 reference abs/rel err < 5e-3）能。
    """
    ms = sum(v * v for v in x) / len(x)
    ms = round(ms, 1)  # 精度截断：累加结果只剩一位小数
    inv = 1.0 / math.sqrt(ms + eps)
    return [v * inv * wi for v, wi in zip(x, w)]


# ------------------------------------------------------------------
# wrap_native 构建策略：JIT 包装社区原生小算子链
#
# 社区原生接口性能差的典型形态不是单算子慢，而是分解链：每个小算子
# 独立 kernel、中间结果逐步物化。性能优化手段是把整条链 wrap 起来
# 编译融合（torch.compile / 厂商 JIT），数值语义继承原生链——这正是
# "Provider 里的 wrapper"：形态是包装，目的是性能，归属 Provider。
# ------------------------------------------------------------------


def _rmsnorm_native_chain(x, w, eps=1e-6):
    """社区原生小算子链（ATen 分解的模拟）：

    square -> mean -> rsqrt -> mul -> mul，五段独立"kernel"，
    中间结果逐步物化——性能差的来源，也是 wrap 的包装对象。
    """
    sq = [v * v for v in x]                       # aten::square
    ms = sum(sq) / len(sq)                        # aten::mean
    inv = 1.0 / math.sqrt(ms + eps)               # aten::rsqrt
    return [(v * inv) * wi for v, wi in zip(x, w)]  # aten::mul x2


def jit_compile(chain, name):
    """JIT 包装器：把原生小算子链包装成单个融合可调用体。

    真实对应物：torch.compile / 厂商 JIT 对 ATen 分解链做融合，
    消除中间物化与 launch 开销。首次调用触发 trace/compile（一次性
    开销），后续走编译缓存——demo 用计数器留痕模拟这一过程。
    """
    def wrapped(*args, **kw):
        if wrapped.compile_count == 0:
            wrapped.compile_count += 1  # mock: trace + compile 一次性开销
        return chain(*args, **kw)
    wrapped.compile_count = 0
    wrapped.__name__ = name
    return wrapped


# ------------------------------------------------------------------
# moe_block：schedule 插槽（缺省 serial，不是 provider）
# ------------------------------------------------------------------


def serial_schedule(thunks):
    """内建缺省调度：逐个执行。不注册为 provider——摘光 Provider 它也在。"""
    return [t() for t in thunks]


def overlap_schedule(thunks):
    """overlap 调度：全部异步下发后统一等待（模拟计算/通信 overlap）。

    消费 Platform Manager 的 active adapter 拿异步句柄——Provider 对
    Bridge 的合法依赖方式。
    """
    hw = manager.activation().hw
    handles = [hw.submit_async(0, f"thunk-{i}") for i in range(len(thunks))]
    results = [t() for t in thunks]
    for h in handles:
        h.wait()
    return results


# ------------------------------------------------------------------
# tp_linear：schedule 插槽第二个完整实例——TP 通算 overlap
#
# 技术原型是 TP 通算 overlap：forward 里把 all-gather 通信异步化、与本地
# matmul 交叠；backward 把 wgrad/dgrad 分解，wgrad 不等通信。本 demo 抽象为
# (comm_thunks, compute_thunks) 两组任务的时序安排——结果不变，只改时序。
# ------------------------------------------------------------------


def serial_tp_schedule(comm_thunks, compute_thunks):
    """缺省串行：通信全部完成后再开始计算（通算无交叠）。"""
    for t in comm_thunks:
        t()
    return [t() for t in compute_thunks]


def overlap_tp_schedule(comm_thunks, compute_thunks):
    """TP 通算 overlap：通信异步下发到独立 comm stream，计算不等待先跑，
    末尾统一等通信句柄——结果与串行逐位一致，只改时序不改语义。

    mock 说明：thunk 本体仍是 Python 同步执行，异步只体现在 stream
    提交/等待的留痕上；真实场景 comm_thunks 由 comm stream 承载，
    与 compute stream 并行。
    """
    hw = manager.activation().hw
    handles = [hw.submit_async(1, f"comm-{i}")  # stream 1 = comm stream
               for i, _ in enumerate(comm_thunks)]
    for t in comm_thunks:
        t()
    results = [t() for t in compute_thunks]     # 计算不等通信完成
    for h in handles:
        h.wait()
    return results


# ------------------------------------------------------------------
# rope_bshd：compute 插槽（完整实例，对照真实代码）
#
# 真实出处：
# - reference = 非融合路径：cos/sin * mscale + _rotate_half（mscale!=1.0 也适用）
# - 厂商融合 = 融合 kernel（fwd/bwd 成对），仅 mscale==1.0 适用
# - 选择逻辑：能拿到融合实现就替换——本 demo 把它正规化为 provider + 门禁
#
# 签名：rope_bshd(t, freqs, mscale=1.0) -> out，三者均为等长 float list
# （单行演示；真实张量 [s, ..., d] 的逐元素结构相同）。
# ------------------------------------------------------------------


def reference_rope_bshd(t, freqs, mscale=1.0):
    """Megatron unfused RoPE：t*cos(freqs*mscale) + rotate_half(t)*sin(freqs*mscale)。

    rotate_half（rotary_interleaved=False）：cat(-x2, x1)，x1/x2 为前后半段。
    """
    half = len(freqs) // 2
    out = []
    for i, v in enumerate(t):
        c = math.cos(freqs[i]) * mscale
        s = math.sin(freqs[i]) * mscale
        rot = -t[i + half] if i < half else t[i - half]
        out.append(v * c + rot * s)
    return out


def fused_aot_vendor_rope_bshd(t, freqs, mscale=1.0):
    """厂商融合算子：转调 vendor_ops.rotary_pos_emb（custom_ops mock）。

    真实 kernel 内不含 mscale 分支——融合路径只在 mscale==1.0 被选中，
    该适用条件由 Provider 的 supports 谓词表达（见 demo 注册处），
    而不是像真实代码那样在 impl 内部 if/else。
    """
    return vendor_ops.rotary_pos_emb(t, freqs)


def fused_jit_rope_bshd(t, freqs, mscale=1.0):
    """JIT 融合算子：数值正确但无收益（冗余重复计算模拟编译产物开销）。

    用来证明收益门禁的必要性：功能、精度都能过，只有 probe 加速比
    远低于 1——性能增量无收益即无存在意义。
    """
    out = None
    for _ in range(50):  # 冗余 50 遍，结果取最后一遍（数值与 reference 一致）
        out = reference_rope_bshd(t, freqs, mscale)
    return out


# --- moe_block 的探测/演示用 thunk（有名函数，provenance 可读） ---

def expert0():
    return 2.0


def expert1():
    return 4.0


def expert2():
    return 6.0


MOE_THUNKS = (expert0, expert1, expert2)


# --- tp_linear 的探测/演示用 thunk ---
# comm thunk 模拟 all-gather 分片（有副作用、无返回值）；compute thunk
# 模拟本地 matmul 分片（返回分片结果，供对账）

def comm_shard0():
    return None


def comm_shard1():
    return None


def matmul_shard0():
    return 1.5


def matmul_shard1():
    return 2.5


TP_COMM_THUNKS = (comm_shard0, comm_shard1)
TP_COMPUTE_THUNKS = (matmul_shard0, matmul_shard1)

# ------------------------------------------------------------------
# 探测输入：门禁用它跑候选实现。语义专属知识，跟随 implementations。
# ------------------------------------------------------------------
PROBE = {
    "rmsnorm": (([0.13, -1.27, 0.85, 0.44, -0.62, 0.99, -0.31, 0.71],
                 [1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 0.95]), {}),
    "rope_bshd": (([0.41, -0.97, 0.22, 1.13, -0.55, 0.68, -0.21, 0.94],
                   [0.01, 0.10, 0.35, 0.80, 1.30, 2.10, 3.20, 4.50]), {}),
    "moe_block": ((list(MOE_THUNKS),), {}),
    "tp_linear": ((list(TP_COMM_THUNKS), list(TP_COMPUTE_THUNKS)), {}),
}

# reference 实现注册表：门禁的精度基准 + 运行时回退终点。
# reference 本身也是 provider（kind="reference"），import 期注册，
# 永远启用、不可禁用。
provider_mgr.provider_register(provider_mgr.Provider(
    name="reference:rmsnorm", kind="reference", semantic="rmsnorm", slot="compute",
    impl=reference_rmsnorm, platforms=None,
    intent="torch-native 基线；摘掉全部 Provider 后的功能底线"))

# schedule 插槽的缺省 serial 也以 reference kind 注册：它是 schedule 的
# 精度基准与回退终点——没有任何 schedule provider 通过门禁时，selection
# 自然落到它，"缺省 serial"由此实现而不是 Manager 里的特例分支。
provider_mgr.provider_register(provider_mgr.Provider(
    name="reference:moe_block/schedule", kind="reference",
    semantic="moe_block", slot="schedule",
    impl=serial_schedule, platforms=None,
    intent="内建串行调度；schedule 插槽的缺省与回退终点"))

provider_mgr.provider_register(provider_mgr.Provider(
    name="reference:rope_bshd", kind="reference", semantic="rope_bshd",
    slot="compute", impl=reference_rope_bshd, platforms=None,
    intent="Megatron unfused RoPE 基线；mscale!=1.0 等融合盲区由它兜底"))

provider_mgr.provider_register(provider_mgr.Provider(
    name="reference:tp_linear/schedule", kind="reference",
    semantic="tp_linear", slot="schedule",
    impl=serial_tp_schedule, platforms=None,
    intent="通算串行基线；tp_linear schedule 插槽的缺省与回退终点"))
