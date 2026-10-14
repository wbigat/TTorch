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


def activate(platform="auto", sweep_packages=None):
    """激活 TT Bridge：platform 检测激活 + conformance 门禁 + cuda-compat 绑定。

    platform: "auto" 走自动检测（正常路径）；传具体名字是调试 override。
    sweep_packages: 覆盖扫尾白名单，None 用 FRAMEWORK_PACKAGES。

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
    global _ACTIVATION
    if _ACTIVATION is not None:
        return _ACTIVATION

    packages = sweep_packages or FRAMEWORK_PACKAGES
    already = [p for p in packages if p in sys.modules]
    if already:
        print(f"[tt] 框架已在场: {already}（其 from-import 陈旧引用由扫尾接管，"
              f"激活末尾验证无残留）")

    # 内置平台已在 import ttorch.platform 时自动注册（adapters/__init__ 扫描），
    # 这里只激活
    if platform == "auto":
        manager.auto_activate()
    else:
        manager.select(platform)

    # 延迟 import：policy 表引用真实 torch.cuda 符号，须在 platform 激活之后
    from ttorch.bridge import module_binding, cuda_compat
    entries = module_binding.entries()

    # default_deny=False：torch.cuda.__all__ 含 torch 自身内部所需符号，
    # 白名单管的是框架调用面，不接管 torch 内部
    cuda_compat.activate(entries, default_deny=False, sweep_packages=packages)

    stale = cuda_compat.verify_sweep(entries, packages)
    if stale:  # 扫尾声称覆盖，就必须验证到位——否则"接管"只是说法
        raise RuntimeError(
            f"扫尾后仍有陈旧引用未被接管: {stale}。这些调用点会绕过 Bridge "
            f"直达厂商实现，属静默失效（不报错、行为不对），拒绝带病激活。")

    _ACTIVATION = BridgeActivation(manager.activation().name, entries)
    print(f"[tt] {_ACTIVATION}")
    return _ACTIVATION


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
