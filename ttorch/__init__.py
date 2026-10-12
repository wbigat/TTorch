"""TT 统一入口：框架之前一行激活 Bridge。

    import ttorch; ttorch.activate()   # 必须在 import 任何训练框架之前

为什么需要显式入口（TT_Bridge_Deep_Dive.md §1.2 的落地形态）：
Bridge 必须在 framework import / model build **之前**激活，否则框架的
`from torch.cuda import X` 会持有绑定前的陈旧引用。扫尾机制（cuda_compat
的 sys.modules 遍历）只能补救已 import 的模块，顺序本身无法补救——
所以激活时机是硬约束，不是建议。

现阶段形态：一组函数 + 本模块；后续做成独立代码库与 whl（届时可改用
`.pth` / entry-points autoload，对调用方零改动——激活动作本身不变）。
对照厂商栈的 `xswift_ext_auto_init.pth`：同一机制的两段式（.pth 触发 +
环境变量开关），TT 先做手动段，自动段是打包问题不是机制问题。

设计约束：
- **幂等**（§1.2）：重复调用不叠加 patch，返回同一份激活结果；
- **fail fast**：激活失败不允许"部分激活后继续跑"；
- **证据可打印**：platform 探测、绑定去向、被 gate 的能力全部见日志。
"""
import sys

from ttorch.bridge import cuda_compat, platform_mgr, platform_specs
from ttorch.bridge.platform_adapter import TTPlatformError  # noqa: F401  # 再导出

# 时序正确性靠**证据**而非启发式（2026-10-12 打包后修正）。
#
# 曾经的做法：拍 sys.modules 快照，把"activate 之前已 import 的框架"
# 判成时序违规并 fail fast。真实环境证明该判据不可靠——
#   - 厂商栈 torch_xmlir.pth 在 site 处理阶段就拉起 torch+transformers+fla，
#     **早于一切用户代码**（`python -S` 对照：跳过 site 后 sys.modules 为空），
#     TT 不可能早于它们，这是环境既成事实而非用户过错；
#   - 装成 whl 后 TT 自身也由 .pth 加载，与厂商 .pth 的先后取决于文件名
#     字母序，快照时机随打包方式漂移（实测：editable 模式与 wheel 模式
#     的快照内容不同，同一份代码在两种安装方式下判定相反）。
#
# 改为不猜"谁先 import"，直接验"绑定有没有生效"：activate() 末尾用
# _stale_references() 按对象身份逐个比对框架模块，有残留即 fail fast。
# 这是充分条件——扫尾干净时，框架何时 import 都不影响正确性。

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

_state = None


class _Activation:
    """激活结果：对外只读，便于调用方做断言与日志。"""

    def __init__(self, platform, entries, excluded_capability):
        self.platform = platform
        self.entries = entries
        self.excluded_capability = excluded_capability

    @property
    def vendor_name(self):
        return platform_mgr.vendor_name()

    @property
    def device_name(self):
        return platform_mgr.device_name()

    def device_spec(self, idx=None):
        """设备标识字符串（"cuda:0"/"xpu:0"）——框架构造 torch.device 的唯一正途。"""
        return platform_mgr.device_spec(idx)

    def __repr__(self):
        return (f"<tt activated platform={self.platform} vendor={self.vendor_name} "
                f"device={self.device_spec()} symbols={len(self.entries)}>")


def activate(platform="auto", sweep_packages=None):
    """激活 TT Bridge：platform 自动检测 + conformance 门禁 + cuda-compat 绑定。

    platform: "auto" 走自动检测（正常路径）；传具体名字是调试 override。
    sweep_packages: 覆盖扫尾白名单，None 用 FRAMEWORK_PACKAGES。

    时序正确性由末尾的 _stale_references() 证据校验保证，不靠"框架是否
    已 import"的启发式判断（见模块注释）。

    幂等：已激活则直接返回上次结果，不重复绑定。
    """
    global _state
    if _state is not None:
        return _state

    packages = sweep_packages or FRAMEWORK_PACKAGES
    already = [p for p in packages if p in sys.modules]
    if already:
        print(f"[tt] 框架已在场: {already}（其 from-import 陈旧引用由扫尾接管，"
              f"激活末尾验证无残留）")

    for spec in platform_specs.REAL_PLATFORMS:
        platform_mgr.register(spec)
    if platform == "auto":
        platform_mgr.auto_activate()
    else:
        platform_mgr.select(platform)

    # 平台 overlay：torch.cuda 为非功能桩的栈（cpu-dev）剔除能力类绑定，
    # 否则 is_available 的"真"回答会误导 torch 内部守卫（G11）
    exclude_cap = platform_mgr.device_name() == "cpu"
    from ttorch.bridge import bindings_torch  # 延迟 import：policy 表引用真实 torch.cuda 符号
    entries = bindings_torch.entries(exclude_capability=exclude_cap)

    # default_deny=False：torch.cuda.__all__ 含 torch 自身内部所需符号，
    # 白名单管的是框架调用面，不接管 torch 内部
    cuda_compat.activate(entries, default_deny=False, sweep_packages=packages)

    stale = _stale_references(entries, packages)
    if stale:  # 扫尾声称覆盖，就必须验证到位——否则"接管"只是说法
        raise RuntimeError(
            f"扫尾后仍有陈旧引用未被接管: {stale}。这些调用点会绕过 Bridge "
            f"直达厂商实现，属静默失效（不报错、行为不对），拒绝带病激活。")

    _state = _Activation(platform_mgr.active().name, entries, exclude_cap)
    print(f"[tt] {_state}")
    return _state


def _stale_references(entries, packages):
    """检查框架包里是否还有指向绑定前实现的陈旧引用。

    `from torch.cuda import synchronize` 会在框架模块里留下独立引用，
    源头 setattr 改不动它——这正是 cuda_compat 扫尾（吸收 KLX
    symbol_rewriter 机制）要解决的。本函数是扫尾的验收：按对象身份比对，
    框架模块里的同名属性必须 is 门面实现。
    """
    bound = {e.symbol.__name__: getattr(
        sys.modules[e.bind_module], e.symbol.__name__) for e in entries}
    stale = []
    for mod_name, mod in list(sys.modules.items()):
        if not any(mod_name == p or mod_name.startswith(p + ".") for p in packages):
            continue
        for attr, impl in bound.items():
            got = getattr(mod, attr, None)
            # 只认"确实持有该符号的旧实现"：同名但无关的对象（如框架自己的
            # 同名函数）不在此列，判据是存在 _origin_ 留痕且当前值不是门面
            if got is not None and got is not impl and hasattr(mod, f"_origin_{attr}"):
                stale.append(f"{mod_name}.{attr}")
    return stale


def active():
    """当前激活结果；未激活则报错（不隐式激活——时序必须由调用方掌控）。"""
    if _state is None:
        raise RuntimeError("TT 未激活：请在脚本最开头 `import ttorch; ttorch.activate()`")
    return _state


def is_active():
    return _state is not None


def device_spec(idx=None):
    """快捷方式：框架侧构造 torch.device 用。"""
    return active().device_spec(idx)


def check_bindings():
    """绑定正确性自检（admission 全套）。返回 (n_fail, rows)。"""
    import torch
    ctx = {"vendor_cuda": torch.cuda}
    rows = []
    n_fail = 0
    for e in active().entries:
        ok, detail = e.test(ctx)
        rows.append((ok, e.test.__name__, detail))
        n_fail += 0 if ok else 1
    return n_fail, rows


# --- 运行期中转计数（证明"真的经过 TT"，而非只是符号指向门面）---
_calls = None


def start_call_trace():
    """给门面装计数探针，统计运行期**实际**经 TT 中转的调用及其调用方。

    为什么需要：`torch.cuda.X is platform_mgr.X` 只证明绑定成立（静态），
    不证明训练期真的走到了——框架可能根本没调用该符号，也可能从别处
    拿到了未被接管的引用。中转是运行期事实，必须用运行期证据。

    返回 collections.Counter（符号 -> 次数）；调用方见 call_trace_report()。
    """
    import collections
    import traceback

    from ttorch.bridge import platform_mgr as pm

    global _calls
    if _calls is not None:
        return _calls
    _calls = collections.Counter()
    callers = collections.defaultdict(set)
    _calls.callers = callers  # 附在同一对象上，调用方不必管两份状态

    import torch
    for e in active().entries:
        name = e.symbol.__name__
        facade = getattr(pm, name, None)
        if facade is None or not callable(facade):
            continue

        def probe(*a, _n=name, _f=facade, **k):
            _calls[_n] += 1
            for fr in traceback.extract_stack()[:-1][::-1]:
                if "/ttorch/" not in fr.filename:
                    callers[_n].add(f"{fr.filename.split('/')[-1]}:{fr.name}")
                    break
            return _f(*a, **k)

        # 探针插在门面与框架之间：框架侧看到的仍是"绑定后的符号"
        if getattr(torch.cuda, name, None) is facade:
            setattr(torch.cuda, name, probe)
    return _calls


def call_trace_report(min_calls=1):
    """中转证据报告。返回 (ok, rows)：ok 表示确有框架侧（非本 smoke 自身）
    的调用经过 TT——这是"TT 在链路上"的最低证据。"""
    if _calls is None:
        return False, []
    rows = []
    for name, n in _calls.most_common():
        callers = sorted(_calls.callers.get(name, ()))
        # 框架侧调用方：排除我们自己的 smoke/验收代码
        external = [c for c in callers
                    if not c.startswith(("real_qwen_smoke", "real_sft_smoke",
                                         "checkpoint", "framework_real"))]
        rows.append((name, n, callers, external))
    ok = sum(n for _, n, _, ext in rows if ext) >= min_calls
    return ok, rows


def finalize():
    """进程退出前收尾（平台生命周期）。绑定不回滚——绑定后不可变是既定约束。"""
    global _state
    if _state is not None:
        platform_mgr.finalize()
        _state = None
