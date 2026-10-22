"""Bridge 启动编排（bootstrap）：cuda-compat 使能的主控。

职责（docs §4.2 子模块表的 bootstrap 项）：激活 Platform、按 policy 完成
cuda_compat 绑定、执行预检（扫尾验收）。

依赖方向是本模块存在的理由：**platform 是纯底座，不认识 bridge；bridge
依赖 platform**。激活编排是"使用 platform 来使能 cuda-compat"的一方，
因此必须站在 bridge 侧——放进 platform 的 manager 会让 platform 反向知道
bridge 的存在（2026-10-12 曾这么做过，已纠正）：

    bootstrap.activate()
        ├──> manager.register/auto_activate()   # bridge → platform（正向）
        └──> cuda_compat.activate()/verify_sweep()    # bridge 内部

运行时数据流同样单向：框架调用 → cuda_compat → platform interface 门面 → 平台实现。
platform 任何模块都不 import bridge（含本模块）——可用
`grep -rn "ttorch.bridge" ttorch/platform/` 验证应为空。
"""
import sys

from ttorch.platform import manager

# 扫尾白名单：会产生 `from torch.cuda import X` 陈旧引用的框架包。
# 不全量扫 sys.modules——会把 torch 内部与无关第三方库的引用一并改写，
# 越界风险不可控（cuda_compat._bind 的 sweep_packages 说明）。
# 新接入一个框架就在这里加一项，这是**有意的显式清单**而非配置项。
FRAMEWORK_PACKAGES = [
    "transformers",   # HF 主链路
    "fla",            # Qwen3.5 linear_attention 依赖
    "swift",          # ms-swift（接入时生效，未安装则无副作用）
    "megatron",
    "mcore_bridge",
]

_ACTIVATION = None  # Bridge 激活状态：BridgeActivation 实例或 None，本模块唯一持有

# 进程级中毒标记：激活失败并已回滚的原因（D3）。`.pth` 路径不 re-raise，
# "失败过"必须用显式状态留下——见 mark_failed / assert_healthy。
_LAST_FAILURE = None


class BridgeActivation:
    """一次 Bridge 激活的结果快照：对外只读，便于调用方做断言与日志。

    身份/设备信息转发自 platform 的 manager（bridge → platform，正向依赖）。
    """

    def __init__(self, platform, entries):
        self.platform = platform
        self.entries = entries

    @property
    def vendor_name(self):
        return manager.activation().vendor_name

    @property
    def device_name(self):
        return manager.activation().device_name

    def device_spec(self, idx=None):
        """设备标识字符串（"cuda:0"/"xpu:0"）——框架构造 torch.device 的唯一正途。"""
        return manager.activation().device_spec(idx)

    def __repr__(self):
        return (f"<tt activated platform={self.platform} vendor={self.vendor_name} "
                f"device={self.device_spec()} symbols={len(self.entries)}>")


def activate(platform="auto", sweep_packages=None, skip_conformance=False):
    """激活 TT Bridge：platform 检测激活 + conformance 门禁 + cuda-compat 绑定。

    platform: "auto" 走自动检测（正常路径）；传具体名字是调试 override。
    sweep_packages: 覆盖扫尾白名单，None 用 FRAMEWORK_PACKAGES。
    skip_conformance: 跳过 conformance（D1 轻档，派生进程用）；默认 False。

    时序正确性靠**证据**而非启发式（2026-10-12 修正）。曾经的做法是拍
    sys.modules 快照、把"activate 之前已 import 的框架"判成时序违规并
    fail fast，真实环境证明该判据不可靠：
      - 厂商栈 torch_xmlir.pth 在 site 处理阶段就拉起 torch+transformers+fla，
        **早于一切用户代码**（`python -S` 对照：跳过 site 后 sys.modules 为空），
        TT 不可能早于它们，这是环境既成事实而非用户过错；
      - 装成 whl 后 TT 自身也由 .pth 加载，与厂商 .pth 的先后取决于文件名
        字母序，快照时机随打包方式漂移。
    所以不猜"谁先 import"，直接验"绑定有没有生效"：末尾用
    cuda_compat.verify_sweep() 按对象身份逐个比对框架模块，有残留即
    fail fast。这是充分条件——扫尾干净时，框架何时 import 都不影响正确性。

    幂等：已激活则直接返回上次结果，不重复绑定。
    """
    global _ACTIVATION, _LAST_FAILURE
    if _ACTIVATION is not None:
        # 幂等早退：本进程已处于激活成功态，中毒标记不应残留
        # （"失败→重试成功"的恢复语义依赖这里与成功路径都清标记）。
        _LAST_FAILURE = None
        return _ACTIVATION

    packages = sweep_packages or FRAMEWORK_PACKAGES
    already = [p for p in packages if p in sys.modules]
    if already:
        print(f"[tt] 框架已在场: {already}（其 from-import 陈旧引用由扫尾接管，"
              f"激活末尾验证无残留）")

    # 内置平台已在 import ttorch.platform 时自动注册（adapters/__init__ 扫描），
    # 这里只激活
    # 延迟 import：policy 表引用真实 torch.cuda 符号，须在 platform 激活之后
    from ttorch.bridge import module_binding, cuda_compat, surface
    entries = module_binding.entries()

    # 失败即完整回滚（D3）：②③ 任一环节失败都要把符号面还原，否则进程带着
    # "部分符号已改写"的混合态继续跑——比完全不激活更危险（部分调用经门面、
    # 部分直达厂商）。回滚后系统是**一致的纯厂商栈**，即 TT 的降级语义。
    try:
        if platform == "auto":
            manager.auto_activate(skip_conformance=skip_conformance)
        else:
            manager.select(platform, skip_conformance=skip_conformance)

        # default_deny=False：torch.cuda.__all__ 含 torch 自身内部所需符号，
        # 白名单管的是框架调用面，不接管 torch 内部。
        # denied=surface.denied_items()：特性功能点（CUDA graph/显存池/自定义
        # 分配器/IPC）显式占位——即使 default_deny=False，这 15 个也绑显式异常，
        # 避免特性语义未验收的静默穿透（昆仑芯 graph-safe 前科）。
        cuda_compat.activate(entries, default_deny=False, sweep_packages=packages,
                             denied=surface.denied_items())

        stale = cuda_compat.verify_sweep(entries, packages)
        if stale:  # 扫尾声称覆盖，就必须验证到位——否则"接管"只是说法
            raise RuntimeError(
                f"扫尾后仍有陈旧引用未被接管: {stale}。这些调用点会绕过 Bridge "
                f"直达厂商实现，属静默失效（不报错、行为不对），拒绝带病激活。")
    except Exception as e:
        mark_failed(e)          # 中毒标记先落——即使回滚本身出问题也有痕迹
        _rollback_activation()
        raise

    _ACTIVATION = BridgeActivation(manager.activation().name, entries)
    # 成功路径必须清除中毒标记：否则"失败一次→重试成功"后 assert_healthy()
    # 仍误报失败（2026-10-22 审视发现的恢复语义缺陷）。
    _LAST_FAILURE = None
    print(f"[tt] {_ACTIVATION}")
    return _ACTIVATION


def mark_failed(error):
    """记一次"激活失败并已回滚"（进程级中毒标记，D3 建议 ②）。

    `.pth` 路径不 re-raise（保护 pip/conda 等一切 python 进程），于是"失败过"
    这件事必须用**显式状态**留下来，否则调用方无从判断本进程到底是
    "TT 正常接管"还是"TT 已失败、跑在纯厂商栈上"。
    """
    global _LAST_FAILURE
    _LAST_FAILURE = {"error": f"{type(error).__name__}: {error}"}


def is_healthy():
    """本进程 TT 是否处于健康态（无"激活失败已回滚"记录）。"""
    return _LAST_FAILURE is None


def failure():
    """上一次激活失败的原因（未失败返回 None）。"""
    return dict(_LAST_FAILURE) if _LAST_FAILURE else None


def assert_healthy():
    """健康断言：激活失败已回滚的进程，不允许继续承载训练。

    与 `is_active()` 的分工：`is_active()` 答"TT 有没有接管"，本函数答
    "TT 有没有失败过"。半激活态时期望的组合是
    `is_active() == False and is_healthy() == False`——此时系统一致的纯厂商栈，
    是否继续由调用方决定（默认 .pth 路径继续，显式调用路径可据此 fail fast）。
    """
    if _LAST_FAILURE is None:
        return
    raise RuntimeError(
        f"TT 激活失败且已回滚（本进程运行在纯厂商栈上，TT 未生效）: "
        f"{_LAST_FAILURE['error']}。若本进程承担训练，请先排除该问题；"
        f"确认可接受降级运行则不要调用 assert_healthy()。")


def _rollback_activation():
    """激活失败回滚：符号（含扫尾）+ provider 绑定状态 + 激活标记全部还原。

    回滚顺序由 `_rebind.rollback` 保证（扫尾先于源头——否则第三方模块里的
    引用仍指向已卸载的门面，比不回滚更隐蔽）。

    **不可回滚且无害的部分**（勿在此尝试逆转）：
    - 厂商实现模块 import（spec.load 已进 sys.modules，仅内存）；
    - conformance 的设备副作用（张量/stream 是局部变量→GC；RNG 由
      rng_state_roundtrip 内部恢复；CUDA context 厂商栈本就要建）；
    - provenance 事件流（观测数据，另记一条 rollback 事件说明）；
    - platform 激活态：`auto_activate` 幂等，保持激活无害，且 finalize 有
      厂商反初始化副作用，故不动。
    """
    global _ACTIVATION
    from ttorch.utils import _rebind
    from ttorch.witness import provenance

    n_source, n_sweep = _rebind.rollback(owner="bridge")
    try:
        from ttorch.provider import fn_binding
        fn_binding.uninstall()
    except Exception:  # noqa: BLE001  回滚路径不得因 provider 侧异常中断
        pass
    _ACTIVATION = None
    provenance.record(provenance.LAYER_BRIDGE, "activate", "bootstrap", "rollback",
                      restored_source=n_source, restored_sweep=n_sweep,
                      detail="激活失败已回滚：符号面还原为纯厂商栈（一致降级态）")


def activation():
    """当前 Bridge 激活结果（BridgeActivation）；未激活返回 None。

    激活状态的唯一查询入口——对应 manager.activation()（platform 侧
    当前 adapter），本接口是 bridge 侧当前激活快照。
    """
    return _ACTIVATION


def finalize():
    """进程退出前收尾：清 Bridge 激活态 + 平台生命周期。绑定不回滚——
    绑定后不可变是既定约束。"""
    global _ACTIVATION
    if _ACTIVATION is not None:
        manager.finalize()
        _ACTIVATION = None
