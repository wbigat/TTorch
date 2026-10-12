"""Platform Manager：Bridge 两块之二（管理面 + 分发门面）。

职责（定稿）：多种 platform 类型的注册 / 自动检测 / 激活管理；
**外部暴露一份统一接口**——行为 API、状态查询、能力查询三类，全部经
active adapter 分发。填非标 gap、错误翻译是 adapter 的事，见
platform_adapter.py（契约）与 adapters.py（各实现）。

管理面：register / auto_activate / select(调试 override) / active /
        finalize；激活期依次执行：版本协商 → init → capability schema
        校验 → conformance 门禁，任一不过则拒绝激活（不切换 _active）。
"""
from contextlib import contextmanager

from tt_bridge.platform_adapter import TTPlatformError, TTStream  # noqa: F401  # 再导出


# ------------------------------------------------------------------
# capability schema：合法 key 注册表（与 registry.py 同构的防手滑设计）
#   tier=probed    必须从硬件实测填充，缺失即激活失败（不许编）
#   tier=declared  adapter 承诺；承诺 True 必须有 conformance 背书测试
#   tier=identity  平台双身份，由 manager 从 adapter 类属性注入，
#                  adapter 的 capabilities() 不许自报（防冒充）
# ------------------------------------------------------------------
CAP_SCHEMA = {
    "vendor_name":     {"type": str,  "tier": "identity"},
    "device_name":     {"type": str,  "tier": "identity"},
    "device_available": {"type": bool, "tier": "probed"},  # is_available 分支选择器的依据
    "device_count":    {"type": int,  "tier": "probed"},
    "p2p":             {"type": bool, "tier": "probed"},
    "device_wide_sync": {"type": bool, "tier": "declared"},  # 训练强制项，见 MANDATORY_CAPS
    "graph_capture":   {"type": bool, "tier": "declared"},
    "async_comm":      {"type": bool, "tier": "declared"},
    "rng_state":       {"type": bool, "tier": "declared"},  # False -> 门面报 unsupported
    "stream_api":      {"type": bool, "tier": "declared"},  # False -> stream 族门面报 unsupported
    "memory_query":    {"type": bool, "tier": "declared"},  # False -> 显存查询门面报 unsupported
}

# 训练平台强制能力：声明 False 直接拒绝激活（不是跳过测试——
# 一个做不了 device-wide sync 的平台不能拿来训练，声明失败即淘汰）
MANDATORY_CAPS = {"device_wide_sync": True}

_REGISTRY = {}
_active = None
_caps_cache = None  # probe 期填充一次的静态事实缓存


def register(adapter_cls):
    """逻辑注册 platform 类型：只登记元数据，不初始化。"""
    _REGISTRY[adapter_cls.name] = adapter_cls


def _ver_tuple(s):
    return tuple(int(x) for x in s.split("."))


def _validate_caps(caps):
    """schema 校验：未知 key 报错；identity 不许自报；probed 缺失报错；
    declared 缺失补 False。"""
    unknown = set(caps) - set(CAP_SCHEMA)
    if unknown:
        raise TTPlatformError("capability_schema", f"未知 capability key: {sorted(unknown)}")
    identity_in_caps = {k for k in caps if CAP_SCHEMA[k]["tier"] == "identity"}
    if identity_in_caps:
        raise TTPlatformError(
            "capability_schema",
            f"identity 项 {sorted(identity_in_caps)} 由 manager 从类属性注入，"
            f"adapter 不许自报（防冒充）")
    out = {}
    for key, spec in CAP_SCHEMA.items():
        if spec["tier"] == "identity":
            continue  # 由 _activate 注入
        if key not in caps:
            if spec["tier"] == "probed":
                raise TTPlatformError("capability_schema", f"probed 项 {key} 缺失（不许编）")
            out[key] = False
            continue
        if not isinstance(caps[key], spec["type"]):
            raise TTPlatformError(
                "capability_schema",
                f"{key} 类型应为 {spec['type'].__name__}，实际 {type(caps[key]).__name__}")
        out[key] = caps[key]
    return out


def _activate(name, how):
    """激活流水线：实例化 → 版本协商 → init → caps schema 校验 →
    conformance 门禁。任一不过则拒绝激活，不切换 _active。"""
    global _active, _caps_cache
    if name not in _REGISTRY:
        raise KeyError(f"platform '{name}' 未注册，可选: {sorted(_REGISTRY)}")
    candidate = _REGISTRY[name]()

    lo, hi = _ver_tuple(candidate.sdk_range[0]), _ver_tuple(candidate.sdk_range[1])
    sdk = _ver_tuple(candidate.sdk_version())
    if not lo <= sdk < hi:
        raise TTPlatformError(
            "sdk_version",
            f"{name} adapter 支持 SDK [{candidate.sdk_range[0]}, {candidate.sdk_range[1]})，"
            f"检测到 {candidate.sdk_version()}")

    candidate.init()
    caps = _validate_caps(candidate.capabilities())
    caps["vendor_name"] = candidate.vendor_name  # identity 档：manager 注入
    caps["device_name"] = candidate.device_name

    lacking = [k for k, want in MANDATORY_CAPS.items() if caps.get(k) != want]
    if lacking:
        candidate.finalize()
        raise TTPlatformError(
            "mandatory_capability",
            f"{name} 训练强制能力不满足: {lacking}（声明 False 即淘汰，不靠测试偶遇）")

    from tt_bridge import conformance  # 延迟 import 避免环
    results = conformance.run(candidate)
    failed = [r for r in results if not r.ok]
    for r in results:
        print(f"[conformance] {'PASS' if r.ok else 'FAIL'} {r.name}: {r.detail}")
    if failed:
        candidate.finalize()
        raise TTPlatformError(
            "conformance",
            f"{name} 未通过 conformance 门禁，拒绝激活: {[r.name for r in failed]}")

    old = _active
    _active = candidate
    _caps_cache = caps
    if old is not None:
        old.finalize()
    print(f"[platform_mgr] activated={name} via={how} capabilities={_caps_cache}")
    return _active


def auto_activate():
    """正常路径：自动分析当前环境，激活唯一匹配的 platform。

    对每个 auto_detect 的已注册 adapter 跑 detect() 存在性探测，
    证据逐条打印（provenance）。恰好一个命中 → 激活；零命中或多命中
    都 fail fast——多命中说明进程级单 platform 的假设被破坏，宁可报错
    也不猜测，此时可用 select(name) 显式 override 排查。

    masquerade 降级：命中者声明了 masquerades_as 时，被冒充身份的命中
    降级为"冒充证据"（打印，不参与冲突判定）——GCU 类栈会让
    torch.cuda.is_available() 报 True，没有这层会把冒充者当真 NVIDIA。
    """
    hits = {}  # name -> evidence
    for name, cls in _REGISTRY.items():
        if not getattr(cls, "auto_detect", True):
            continue
        ok, evidence = cls.detect()
        print(f"[platform_mgr] detect {name}: {'present' if ok else 'absent'} ({evidence})")
        if ok:
            hits[name] = evidence
    demoted = set()
    for name in hits:
        for fake in getattr(_REGISTRY[name], "masquerades_as", ()):
            if fake in hits:
                demoted.add(fake)
                print(f"[platform_mgr] detect {fake}: 命中被 {name} 的 native 探测降级"
                      f"（{fake} 是 {name} 的 masquerade 身份，非独立硬件）")
    matched = [n for n in hits if n not in demoted]
    if len(matched) == 1:
        return _activate(matched[0], how="auto-detect")
    if not matched:
        raise RuntimeError(f"未检测到任何已注册 platform，已探测: {sorted(_REGISTRY)}")
    raise RuntimeError(
        f"检测到多个 platform: {matched}——进程级单 platform，拒绝猜测；"
        f"调试可用 platform_mgr.select(name) 显式 override")


def select(name):
    """显式 override：仅调试/测试逃生通道，正常路径走 auto_activate()。"""
    return _activate(name, how="manual-override")


def active():
    if _active is None:
        raise RuntimeError("platform 未激活：请先 platform_mgr.auto_activate()")
    return _active


def finalize():
    """进程退出前的生命周期收尾。"""
    global _active, _caps_cache
    if _active is not None:
        _active.finalize()
        print(f"[platform_mgr] finalized={_active.name}")
    _active = None
    _caps_cache = None


def capability(key):
    """能力查询（单 key）：读 probe 期缓存的静态事实。

    不暴露为动态 API——答案是进程内不变的事实，选择期消费、可缓存。
    """
    if _caps_cache is None:
        raise RuntimeError("platform 未激活：请先 platform_mgr.auto_activate()")
    return _caps_cache[key]


def capabilities():
    """能力查询（全量快照）：返回静态事实表副本，调用方改不动缓存。"""
    if _caps_cache is None:
        raise RuntimeError("platform 未激活：请先 platform_mgr.auto_activate()")
    return dict(_caps_cache)


def vendor_name():
    """厂商身份（capability/identity 档）：provider 分发、门禁、provenance 用。"""
    return capability("vendor_name")


def device_name():
    """torch 视角的设备类型（capability/identity 档）。

    框架里 `device.type == "cuda"` 这类比较必须用它——冒充 CUDA 的栈上
    它是 "cuda" 而非品牌名，写成 vendor_name 会让比较静默失效。
    """
    return capability("device_name")


# =====================================================================
# 统一对外接口：行为 API + 状态查询（能力查询走上面的 capability()）
#
# 它不是独立的一层：契约由 PlatformAdapter 定义，这里只是激活之后
# 顺带的分发门面——每个函数向 active adapter 转发一行，无自身逻辑。
#
# 只承载两类接口（判定：Q1 有副作用？Q2 答案进程内会变？）：
# - 行为（有副作用）：set_device / synchronize / range_push
# - 状态查询（动态只读，禁缓存）：current_device / memory_allocated
#
# 契约示例：synchronize() = device-wide 等待（不是 current-stream）；
# memory_allocated() 口径 = 已分配字节数，每次调用实时读硬件。
# =====================================================================

def device_spec(idx=None):
    """设备标识字符串（"xpu:0" / "cuda:0" 形态），供框架构造 torch.device。

    current_device() 的 int 返回值在 native xpu 上被 torch 当 cuda 索引
    解析（torch.tensor(device=0) 的隐含约定）——设备构造必须走
    device_name + 序号的完整 spec，这是双身份在框架侧的落地形态。
    """
    a = active()
    return f"{a.device_name}:{a.current_device() if idx is None else idx}"


def set_device(idx):
    active().set_device(idx)


def synchronize(device=None):
    """device 可选参对齐 torch.cuda.synchronize(device) 真实签名。"""
    active().synchronize(device)


def range_push(name):
    """扩展位：profiler 区间标注（torch.accelerator 暂无对应，按永久资产建）。"""
    active().range_push(name)


def current_device():
    return active().current_device()


def _require_cap(key, api):
    """可选能力保护：平台声明不支持时，门面报统一 unsupported 错误，
    框架据此走降级路径（而不是在厂商 API 深处炸原生错误）。"""
    if not capability(key):
        raise TTPlatformError("unsupported", f"{api}: 平台声明 {key}=False")


def memory_allocated(device=None):
    _require_cap("memory_query", "memory_allocated")
    return active().memory_allocated(device)


def mem_get_info(device=None):
    """(free_bytes, total_bytes)；字节 tuple 契约，厂商结构/单位差异在 adapter 内换算。"""
    _require_cap("memory_query", "mem_get_info")
    return active().mem_get_info(device)


def get_rng_state(device=None):
    """checkpoint save 的 RNG 状态读取（不透明可序列化对象契约）。"""
    _require_cap("rng_state", "get_rng_state")
    return active().get_rng_state(device)


def empty_cache():
    active().empty_cache()


def set_rng_state(state, device=None):
    """checkpoint resume 的 RNG 状态恢复；非法值经 adapter 翻译成
    TTPlatformError(category="invalid_rng_state")。"""
    _require_cap("rng_state", "set_rng_state")
    active().set_rng_state(state, device)


# --- stream 对象族（swift megatron 路径 ddp fencing 必触发） ---

class Stream:
    """门面 Stream：框架 new 出来的是 TTStream 句柄（对象形态与 CUDA 一致）。"""

    def __new__(cls):
        _require_cap("stream_api", "Stream")
        return TTStream(active(), active().stream_create())


def current_stream():
    _require_cap("stream_api", "current_stream")
    return TTStream(active(), active().stream_current())


@contextmanager
def stream(s):
    """stream 上下文管理器：进入切当前流，退出恢复原流。"""
    a = active()
    prev = a.stream_current()
    a.stream_set(s.sid)
    try:
        yield s
    finally:
        a.stream_set(prev)
