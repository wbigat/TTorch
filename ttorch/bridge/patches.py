"""框架功能补丁清单（bridge 层）。机制见 framework_patch.py。

每条补丁必须写清三件事：修什么缺陷、证据在哪、上游修好后能否删。

**撤出记录（2026-11-04 决策）**：本模块曾登记 large-weight 相关的三条补丁
（`GPTBridge._set_mlp_state` + `LoraParallelLinear.merge/unmerge`，共享
`large_weight_shim`）。这些缺口**仅在主动开启 `XTE_GROUPED_GEMM_LARGE_WEIGHT=1`
时存在**（LW=0 下 `weight{i}` 是真实 Parameter，一条都不成立），属"某个可选
开关的代价"而非硬件/框架缺口，且 shim 要替厂商维护"3D 合并参数 ↔ per-expert
视图"的映射——是厂商权重布局的实现细节，不属 bridge 的功能接入语义。
故不在 TT 内解决，转独立方案。代码移至 `experiments/large_weight/`
（不参与包导入），分析结论见 `docs/TT_Bridge_Deep_Dive.md` §9.5。
"""
from ttorch.bridge.framework_patch import FunctionPatch, register_patch

__all__ = ["register_all", "thd_rope_cp_group_wrapper", "gather_object_cpu_wrapper",
           "ddp_stream_fence_wrapper"]

# mcore_bridge 在自己的 patch 闭包里保存的上游实现名
_MCORE_BRIDGE_ORIGIN = "_origin_apply_rotary_pos_emb_thd"

_THD_MODULE = "megatron.core.models.common.embeddings.rope_utils"
_THD_ATTR = "_apply_rotary_pos_emb_thd"
_THD_DOTTED = f"{_THD_MODULE}.{_THD_ATTR}"

_GATHER_MODULE = "swift.loss.embedding"
_GATHER_ATTR = "gather_object"
_GATHER_DOTTED = f"{_GATHER_MODULE}.{_GATHER_ATTR}"

_WRAP_MODEL_MODULE = "swift.megatron.utils.megatron_lm_utils"
_WRAP_MODEL_ATTR = "wrap_model"
_WRAP_MODEL_DOTTED = f"{_WRAP_MODEL_MODULE}.{_WRAP_MODEL_ATTR}"


def _fn_source_contains(fn, needle):
    """源码形态探测：fn 的源码里是否出现 needle。

    升级安全共用件（B1 的 ``_upstream_has_fix`` 同款判据）：用源码形态而非
    版本号——内部 fork / 定制打包会让版本号说谎。拿不到源码按「不含」处理：
    该方向是安全的（误判未修复 → 补丁生效但运行时判据兜底；误判已修复
    才会漏补，而 needle 出现在未修复代码里的概率可忽略）。
    """
    import inspect

    try:
        return needle in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


def ddp_stream_fence_wrapper(orig):
    """B5：给 swift ``wrap_model`` 的 DDP 初始化侧流补双向同步栅栏。

    **缺陷（源码核实）**：swift ≤4.3.1 的 ``wrap_model``
    （`swift/megatron/utils/megatron_lm_utils.py:540`）在**裸侧流**里构造
    DDP 包装（``with torch.cuda.stream(torch.cuda.Stream())``），侧流与默认
    流**双向都没有 join**。两个静默隐患：① 侧流的参数拷贝可能读到主流上
    尚未完成的 ``.cuda()``/cast 结果；② ``broadcast_params()`` 与首次前向
    可能在侧流写完 flat buffer 前读取它。是否显形取决于 kernel 调度
    （640 rank-case 无错 ≠ 无风险）。

    **社区修复（已存在）**：ms-swift **≥4.3.2** 的 ``wrap_model`` 已带双栅栏
    （``ddp_stream.wait_stream(主流)`` + ``主流.wait_stream(ddp_stream)``；
    v4.3.2 / v4.4.0 / main 均已核实）。本补丁是该修复在 ≤4.3.1 窗口的摆渡。

    **与社区修复的关系——语义等价、实现更强、零复制**：社区的
    ``wait_stream`` 是事件排序（不阻塞 host）；本补丁在**函数外**加
    ``synchronize``——入口保证内部侧流启动前主流既有工作全部完成，出口
    保证内部侧流的 DDP 初始化全部完成后再返回，提供的 happens-before
    保证与社区修复相同（且更强：host 阻塞；出口的
    ``torch.cuda.synchronize()`` 是**全设备**同步，会把同进程其它并发流
    一起等住——``wrap_model`` 是每次启动只跑一次的初始化路径，此代价
    可接受）。换来的是**不复制** ``wrap_model`` 的 50 行本体（XSwift
    原补丁是逐字复制 + 两行，框架演进即漂移）。
    """
    import functools

    import torch

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        """swift wrap_model + DDP 侧流双向栅栏。"""
        torch.cuda.current_stream().synchronize()   # 入口栅栏
        try:
            return orig(*args, **kwargs)
        finally:
            torch.cuda.synchronize()                # 出口栅栏（异常也执行）

    return wrapper


def _ddp_fence_applies():
    """前置判据：swift 的 wrap_model 在场 **且未含社区修复**（双栅栏）。

    升级安全：源码检出 ``ddp_stream``（≥4.3.2 修复引入的命名流变量）→
    整个补丁不接管（skipped + 留痕），保证社区更新后零影响。
    """
    import sys

    mod = sys.modules.get(_WRAP_MODEL_MODULE)
    if mod is None:
        return False
    fn = getattr(mod, _WRAP_MODEL_ATTR, None)
    if fn is None:
        return False
    # needle 用 "ddp_stream" 而非 "wait_stream"：前者是 4.3.2 修复引入的
    # 命名流变量，特异性高；后者是通用 API 名，上游无关改动含它就误判
    # 已修复（方向危险：不接管 → 裸侧流风险静默回归）。
    return not _fn_source_contains(fn, "ddp_stream")


def thd_rope_cp_group_wrapper(orig):
    """补回 THD RoPE non-batched fallback 路径丢失的 cp_group。

    **根因链（2026-11-04 补全，全部有出处）**

    1. **源头变更**：Megatron-LM **0.18.0** 给上游
       ``_apply_rotary_pos_emb_thd`` 新增了 ``cp_group`` 硬要求
       （`megatron/core/models/common/embeddings/rope_utils.py:200-201`）::

           if cp_group is None:
               raise ValueError("cp_group must be provided for THD format RoPE")

       且拿它做 CP 分片（``cp_group.size()`` / ``cp_group.rank()`` 切 freqs）。
    2. **适配层未跟上**：mcore-bridge ≤1.5.x 的包装
       （`mcore_bridge/patcher.py:202`，属 ModelScope 开源项目）按旧签名写成
       ``cp_group = kwargs.pop('cp_group', None)`` —— 取出后**没有放回**，
       non-batched fallback（`:211`）把 kwargs 原样转给上游时 cp_group 已丢失。
    3. **后果**：CP>1 且走 non-batched fallback → 直接 ``ValueError``；若调用方
       自带默认值绕过报错，则 freqs 按错误 rank 取片 —— **静默数值错，比报错
       更危险**。

    **社区修复（已存在，本补丁是它的版本窗口摆渡）**：
    modelscope/mcore-bridge **PR #146**「FIX APPLY_ROTARY_POS_EMB_THD MISSING
    CP_GROUP」（作者/合并 hjh0119，功能提交 ``f42016e``，合并提交 ``2220f77``，
    2026-07-11 合入；**首个含修复版本 1.6.0**，2026-07-13 发布；页面未见关联
    issue）。社区修复逻辑::

        cp_group = kwargs.pop('cp_group', None)
        if cp_group is None:
            cp_group = mpu.get_context_parallel_group()      # 兜底
        ...
        return _origin_apply_rotary_pos_emb_thd(..., cp_group=cp_group, **kwargs)

    **受影响版本窗口**：≤1.5.2 受影响（1.5.2 发布于 2026-06-28，早于修复）；
    ≥1.6.0 已修。ms-swift 4.3.1 的约束是 ``mcore-bridge>=1.4.0``，升级在约束内
    （兼容性冒烟属待办 E 组）。

    **与社区修复完全对齐（2026-11-04 用户决策，取代此前的保守偏差）**：
    社区修法对 ``mpu.get_context_parallel_group()`` 是**无条件调用**
    （PR 评审中 gemini-code-assist 指出 parallel_state 未初始化时会抛
    ``AssertionError``，建议 ``if cp_size > 1 else None``，**未被采纳**）。
    本补丁同样无条件回退——打过补丁的 ≤1.5.x 与社区 ≥1.6.0 **行为逐点一致**
    （含边界：未初始化时抛同一个 ``AssertionError``），升级是行为无感的。
    回归 ``test_parallel_state_failure_matches_community_behavior`` 钉住该对齐。

    一个已知的语义差（位置决定的，生产不可达）：本补丁在闭包 cell 层注入，
    若调用方显式传的 cp_group 与 parallel_state 当前组**不同**且走了
    fallback，注入的是后者而非前者。真实链路中两者恒为同一对象（attention
    传的就是 parallel_state 的 CP 组），且不改时该路径直接 ValueError——
    严格属于改善，但如实登记。

    **升级安全保障（四层，2026-11-04 按用户决策补齐）**——社区更新后本补丁
    不得产生任何负面影响：

    1. **源码形态探测**（本层）：``_thd_rope_applies`` 先调 ``_upstream_has_fix``，
       在上游包装的源码里检出 ``cp_group=cp_group`` 显式回传 → 判定已修复 →
       **整个补丁不注册生效**（skipped + provenance 留痕）。用源码形态而非版本号，
       因为内部 fork / 定制打包会让版本号说谎；
    2. **结构探测**：mcore_bridge 重构导致闭包 cell 找不到 → ``_find_origin_cell``
       返回 None → 不接管；
    3. **运行时透传**：上游修复后 fallback 显式传 ``cp_group=cp_group``，本 wrapper
       「缺了才补」的判据命中不了 → 无害透传（即使前两层都漏了）；
    4. **幂等**：``_tt_cp_group_fixed`` 标记防重复激活叠层。

    **修补位置为什么必须是上游侧，而不是入口侧**（2026-11-04 实测修正）

    最初的实现是在入口处"把 cp_group 补进 kwargs"再转给 mcore_bridge。
    这**无效**：丢失动作（`pop`）发生在 mcore_bridge **内部**，入口补得再全，
    它照样 pop 掉，fallback 依旧缺参。最小复现测试直接证伪了该设计
    （`test_patch_restores_cp_group_on_fallback` 首版即失败）。

    正确做法：包住**上游实现本身**。mcore_bridge 的闭包变量
    `_origin_apply_rotary_pos_emb_thd` 指向上游函数，我们把它替换成一个
    "缺 cp_group 就补上再转发"的版本。这样：
      - batched 分支：不经上游，行为完全不变；
      - non-batched fallback：进入我们的包装，cp_group 被补回；
      - 不复制 mcore_bridge 的分支判据（XSwift 原补丁复制了 use_batched_rope
        的计算，框架改判据就会漂移），职责更窄。

    实现手段是改写闭包 cell（`__closure__` 的 cell_contents 可写）——这是
    Python 层唯一能在不改框架源码的前提下替换闭包捕获值的办法。失败时
    （cell 不可写 / 结构不符）返回 orig 不接管，由 framework_patch 留痕。

    **可删除条件（已具备明确触发点）**：mcore-bridge **≥1.6.0** 已含社区修复
    （PR #146）。升级后 ``_upstream_has_fix`` 会让本补丁在注册期就不接管；
    即使探测漏掉，运行时「缺了才补」也命中不了（上游已显式回传）。环境完成
    升级并经过兼容性冒烟（待办 E 组）后，本补丁可整个删除。
    """
    import functools

    cell = _find_origin_cell(orig)
    if cell is None:
        # 链上没有 mcore_bridge 的 patch：要么框架未加载该 patch，要么其
        # 内部结构已变。不接管（返回 orig 本身，framework_patch 会留痕）。
        return orig
    upstream = cell.cell_contents
    if getattr(upstream, "_tt_cp_group_fixed", False):
        return orig  # 已补过（重复激活）：不叠层

    @functools.wraps(upstream)
    def upstream_with_cp_group(t, cu_seqlens, freqs, *args, **kwargs):
        """Megatron 上游 THD RoPE + cp_group 补齐（与社区修复逻辑完全一致）。

        对齐 PR #146：cp_group 缺失时**无条件**回退 parallel_state 的当前
        CP 组——包括其边界行为（parallel_state 未初始化时抛
        ``AssertionError``，不吞、不改写）。打过补丁的 ≤1.5.x 与社区
        ≥1.6.0 行为逐点一致，未来升级因此是行为无感的。

        不覆盖：上游侧已拿到 cp_group 时原样透传。
        """
        if kwargs.get("cp_group") is None:
            kwargs["cp_group"] = _current_cp_group()
        return upstream(t, cu_seqlens, freqs, *args, **kwargs)

    upstream_with_cp_group._tt_cp_group_fixed = True
    try:
        cell.cell_contents = upstream_with_cp_group
    except (ValueError, AttributeError):
        return orig  # cell 不可写：不接管
    # 入口函数对象本身不变（分支判据仍是 mcore_bridge 的），但语义已修正。
    # 返回一个 wraps 后的透传层，让 framework_patch 的"是否接管"判据
    # （new is not orig）成立并留下 provenance。
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        """mcore_bridge THD RoPE（上游侧已补 cp_group）。"""
        return orig(*args, **kwargs)

    return wrapper


def _current_cp_group():
    """当前 CP 进程组（与社区修复同一来源、同一行为：megatron parallel_state）。

    **不加 try/except**（2026-11-04 用户决策：完全对齐 PR #146）——社区修复
    就是无条件调用，parallel_state 未初始化时的 ``AssertionError`` 是社区
    ≥1.6.0 的固有行为，本补丁保持一致（THD 路径本就要求分布式已初始化，
    该场景不可能正常工作，两个版本只是抛同一个错）。此前"拿不到就返回
    None、让上游 ValueError 暴露"的保守版本已废弃——它让打补丁的旧版与
    社区新版在边界上**行为不一致**，违背逻辑兼容目标。
    """
    from megatron.core import parallel_state

    return parallel_state.get_context_parallel_group()


def _find_origin_cell(func):
    """沿 wrapper 链定位持有上游实现的**闭包 cell**（而非其值）。

    拿 cell 而非值，是因为要把捕获的上游函数**替换**掉——取到值只能读。
    """
    import inspect

    seen = set()
    while func is not None and id(func) not in seen:
        seen.add(id(func))
        code = getattr(func, "__code__", None)
        closure = getattr(func, "__closure__", None)
        if code is not None and closure:
            names = code.co_freevars
            if _MCORE_BRIDGE_ORIGIN in names:
                return closure[names.index(_MCORE_BRIDGE_ORIGIN)]
        func = getattr(func, "__wrapped__", None)
    return None


def _upstream_has_fix(fn):
    """探测 mcore_bridge 的包装是否已含社区修复（PR #146，≥1.6.0）。

    判据是**源码形态**：已修复版本的 fallback 显式回传 ``cp_group=cp_group``。
    不用版本号——内部 fork / 定制打包会让版本号说谎，源码形态不会。
    拿不到源码（C 扩展 / 动态生成）时按「未修复」处理，交后续探测兜底：
    该方向是安全的——误判未修复只会让运行时透传层生效（「缺了才补」判据
    在已修复版本上永远命中不了），误判已修复才会漏补，而 ``cp_group=cp_group``
    出现在未修复代码里的概率可忽略。
    """
    return _fn_source_contains(fn, "cp_group=cp_group")


def _thd_rope_applies():
    """前置判据：仅当「mcore_bridge 的 THD patch 在场 **且未含社区修复**」。

    不按 CP size 判断——补丁注册发生在训练启动早期，此时 parallel_state
    可能尚未初始化；而 wrapper 本身在 CP=1 下是无害的（补一个 size=1 的
    group，上游行为不变）。按"缺陷是否存在"判断比按"当前配置是否触发"
    更稳，也避免了初始化时序依赖。

    升级安全（2026-11-04 用户决策）：先检 ``_upstream_has_fix``——上游
    ≥1.6.0 已修复时整个补丁不接管（skipped + 留痕），保证社区更新后
    本补丁零影响。详见 ``thd_rope_cp_group_wrapper`` 的「升级安全保障」。
    """
    import sys

    mod = sys.modules.get(_THD_MODULE)
    if mod is None:
        return False
    fn = getattr(mod, _THD_ATTR, None)
    if fn is None:
        return False
    if _upstream_has_fix(fn):
        return False   # 上游已修复（≥1.6.0）：不接管
    return _find_origin_cell(fn) is not None


def gather_object_cpu_wrapper(orig):
    """B7：object gather 前把 payload 里的张量递归搬到 CPU（detach）。

    **定性（2026-11-04 用户确认）**：**未实际复现故障，但理论风险成立**。

    **风险点**：``swift/loss/embedding.py:159-160``（embedding 模型 InfoNCE
    对比损失跨 DP gather 句向量）对 **CUDA 张量**直接调 accelerate 的
    ``gather_object``。object collective 的 pickle 序列化会触发 CUDA storage
    的反序列化——这在 PyTorch/accelerate 的约定中本来就是脆弱用法
    （推荐 CPU payload），KL3 masquerade 栈的 storage 反序列化路径把该
    风险进一步放大（XSwift 原补丁注释：Avoid CUDA storage deserialization）。

    **为什么语义零争议**：把 payload 搬 CPU 是 PyTorch 文档对 object
    collective 的推荐实践；业务侧本来就在 gather 后自己
    ``.to(sentences.device)`` 搬回（`swift/loss/embedding.py:167`），
    数值路径不变。与 XSwift 原补丁（``xswift_ext/components/swift/loss/
    embedding.py``）的差别：原补丁硬编码调 accelerate 的实现，本补丁
    **包装当前绑定的原函数**（``orig``），不假设其来源。

    **绑定位置为什么有效**：调用点 ``gather_object(...)`` 是
    ``swift.loss.embedding`` 模块命名空间里的 from-import 绑定
    （`embedding.py:7`），调用时按模块全局解析——rebind 该命名空间即可
    生效，无需扫尾。

    **后续**：作为外部框架的基础功能修复点收录；可推 swift 社区
    （CPU-ify 是普适健壮性改进，所有后端受益），上游采纳后按升级安全退场。
    """
    import functools

    import torch

    def _to_cpu_object(obj):
        """递归 CPU-ify。容器语义：dict/list/tuple 重建保型；namedtuple 用
        ``_make`` 保型（2026-11-04 review 修正：此前会降级为 plain tuple）；
        set/frozenset 递归保型；其余 tuple 子类（torch.Size 等）按 plain
        tuple 重建——payload 实际只含张量与 list，其余类型属防御性覆盖。"""
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu_object(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu_object(v) for v in obj]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)._make(_to_cpu_object(v) for v in obj)
        if isinstance(obj, tuple):
            return tuple(_to_cpu_object(v) for v in obj)
        if isinstance(obj, (set, frozenset)):
            return type(obj)(_to_cpu_object(v) for v in obj)
        return obj

    @functools.wraps(orig)
    def wrapper(obj):
        """accelerate gather_object + payload 递归 CPU-ify。"""
        return orig(_to_cpu_object(obj))

    return wrapper


def register_all():
    """注册全部框架功能补丁（ttorch.activate 时调用）。

    幂等：activate 可被重复调用（显式调用 + .pth 自动激活），而
    register_patch 对重复 dotted 是 assert。这里按已注册清单早退，不让
    重复激活炸在断言上。
    """
    from ttorch.bridge.framework_patch import get_patches

    registered = {p.dotted for p in get_patches()}
    if _WRAP_MODEL_DOTTED not in registered:
        register_patch(FunctionPatch(
            dotted=_WRAP_MODEL_DOTTED,
            module=_WRAP_MODEL_MODULE,
            attr=_WRAP_MODEL_ATTR,
            wrapper=ddp_stream_fence_wrapper,
            intent="补 swift ≤4.3.1 wrap_model 的 DDP 初始化侧流双栅栏"
                   "（裸侧流建 DDP，参数拷贝与 broadcast/首次前向存在跨流"
                   "竞态；是否显形取决于 kernel 调度）",
            applies_when=_ddp_fence_applies,
            upstream_ref="社区已修复：ms-swift ≥4.3.2 wrap_model 双栅栏"
                         "（v4.3.2/v4.4.0/main 均核实）；升级后由源码形态"
                         "探测（wait_stream）自动不接管",
        ))
    if _GATHER_DOTTED not in registered:
        register_patch(FunctionPatch(
            dotted=_GATHER_DOTTED,
            module=_GATHER_MODULE,
            attr=_GATHER_ATTR,
            wrapper=gather_object_cpu_wrapper,
            intent="object gather 的 payload 递归 CPU-ify：CUDA storage 反序列化"
                   "是 object collective 的脆弱用法（未实际复现，理论风险成立），"
                   "CPU payload 是 PyTorch 推荐实践，语义零争议",
            upstream_ref="拟推 swift 社区 PR（普适健壮性改进）；上游采纳后"
                         "按升级安全退场",
        ))
    if _THD_DOTTED in registered:
        return
    register_patch(FunctionPatch(
        dotted=_THD_DOTTED,
        module=_THD_MODULE,
        attr=_THD_ATTR,
        wrapper=thd_rope_cp_group_wrapper,
        intent="补回 mcore_bridge ≤1.5.x THD RoPE non-batched fallback 丢失的"
               " cp_group（megatron-core 0.18 起上游对其为硬要求；缺失即"
               " ValueError 或 freqs 按错误 cp_rank 取片）",
        applies_when=_thd_rope_applies,
        # 判据要求 mcore_bridge 已把上游包进闭包——该动作发生在
        # import mcore_bridge（__init__ 调 apply_patch）期间；rope_utils
        # 若先被 import，首次判据必失败，必须在它 exec 完成后重试
        # （2026-11-04 生产时序实测发现）。
        retry_after=("mcore_bridge",),
        upstream_ref="社区已修复：modelscope/mcore-bridge PR #146"
                     "（commit 2220f77，2026-07-11，首个含修复版本 1.6.0）；"
                     "本补丁是 ≤1.5.2 窗口的摆渡，升级后由 _upstream_has_fix"
                     " 探测自动不接管",
    ))
