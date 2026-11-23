"""厂商插件（XSwift / XME）退役的前置检查：在场探针 + checkpoint 导出安全契约。

**为什么需要这个模块**（2026-11-18 整网实测，退役文档 §16）：关闭或移除 XSwift
之后，三个环境开关里缺任一条都会在 **exit=0、loss 逐位一致、零报错**的情况下
导出垃圾权重（~3.7e33）：

| 开关 | 缺失后果 | 实测污染面 |
|---|---|---|
| `XTE_GROUPED_GEMM_LARGE_WEIGHT` 必须为 0 | XME 的 `__getattr__` 让 `weight{i}` 可读但值错（**不是** AttributeError） | 180 个 `mlp.experts.*` |
| `HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT` 必须为 1 | fast-FC mem-efficient 原地复用转置缓冲 | `lm_head.weight` |
| `XTE_DISABLE_FAST_BF16_CACHE` 必须为 1 | TE BF16 pin 在 forward 原地覆盖 Parameter 导出视图 | 全体张量（含 LoRA adapter） |

三者互相独立，各自解释一类张量。最危险的是失败模式：**没有任何运行期信号**，
而且 **LoRA 口径验不出前两条**（HF 只导出 adapter，base 权重不在导出集合里）。
所以这件事不能靠"写进文档让人记得设"，必须在代码里有一道闸。

分工：
- 开关为「可安全兜底」的（`fixable=True`）走 `apply_export_safe_defaults()`
  的 setdefault——安全值不改变任何显式性能选择；
- `XTE_GROUPED_GEMM_LARGE_WEIGHT=1` 是**显式的性能决策**（MoE 层 ~12-14%），
  不能被悄悄翻成 0，只能 fail-fast 让使用方自己选（关 LW 退役 / 留 LW 不退役）。

另外提供 XSwift 的两处机制副作用探针（退役文档 §15.3-C）：
`import_module_safe` 在 import 失败时往 `sys.modules` 注入指向
`/klx_dummy_lib/...` 的空模块，会掩盖真实 import 失败并污染"无包验证"的
`sys.modules` 审计——退役检查必须显式扫这个痕迹。

依赖方向：只读 `os.environ` / `sys.modules` / `importlib.util.find_spec`，
不 import 厂商包（探测本身不能把被探测对象拉起来）。
"""
import importlib.util
import os
import sys
from collections import namedtuple

__all__ = ["PLUGIN_PACKAGES", "DUMMY_LIB_MARKER", "EXPORT_ENV_RULES",
           "ExportUnsafeError",
           "implementation_owner", "plugin_presence", "plugin_protection_expected",
           "dummy_module_traces", "export_safety_issues",
           "apply_export_safe_defaults", "assert_export_safe",
           "assert_export_safe_for_worker", "retirement_report"]

# 厂商插件包：两者都会改写框架符号，升级安全探测必须能把它们与"上游已修复"分开
PLUGIN_PACKAGES = ("xswift_ext", "xmegatron_ext")

# XSwift `symbol_rewriter.import_module_safe` 造假模块时写入的 spec.origin 前缀
DUMMY_LIB_MARKER = "/klx_dummy_lib"

_TRUTHY = ("1", "true", "yes", "y", "on")

_Rule = namedtuple("_Rule", "env safe_value unset_is_safe symptom fixable")


class ExportUnsafeError(RuntimeError):
    """导出安全契约不满足。

    单独立类型是为了让 `.pth` 自动激活路径能把它与"普通激活失败"分开：
    后者降级继续（保护 pip/conda 与辅助进程）是对的，而前者降级继续就等于
    回到静默损坏——`utils/_auto_init` 据此把它转成 `SystemExit` 穿透
    `site` 的异常捕获，让训练 worker 立刻失败。
    """

EXPORT_ENV_RULES = (
    _Rule("XTE_GROUPED_GEMM_LARGE_WEIGHT", False, True,
          "MoE 专家权重（mlp.experts.*，实测 180 个张量）导出为 ~3.7e33 垃圾值",
          False),
    _Rule("HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT", True, False,
          "lm_head.weight 导出为 ~3.7e33 垃圾值", True),
    _Rule("XTE_DISABLE_FAST_BF16_CACHE", True, False,
          "全体导出张量被 TE BF16 pin 原地覆盖（含 LoRA adapter）", True),
)


def _env_truth(name):
    """环境变量的布尔值；未设置返回 None（"未设置"与"显式关"必须能分辨）。"""
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def implementation_owner(fn):
    """fn 的实现是否来自厂商插件；是则返回包名，否则 None。

    **为什么必须有这个判据**（退役文档 §15.3-A②）：TT 的升级安全探测原本只看
    源码 needle，而 XSwift 的 `wrap_model` 替换实现是 swift ≥4.3.2 的逐字拷贝、
    变量名恰好就叫 `ddp_stream` —— TT 于是判「上游已修复 → 不接管」并如此留痕。
    净效果无害（XSwift 确实提供了栅栏），但 provenance 把"依赖厂商插件"记成
    "上游已修复"，正是退役期最不该搞错的归因。

    沿 `__wrapped__` 链找：第三方可能在厂商实现外面再套一层，只看最外层会漏判。
    两条证据都认（`__module__` 可被改写，`__code__.co_filename` 更硬）。
    """
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        module = getattr(fn, "__module__", "") or ""
        filename = getattr(getattr(fn, "__code__", None), "co_filename", "") or ""
        for pkg in PLUGIN_PACKAGES:
            if module == pkg or module.startswith(pkg + "."):
                return pkg
            if f"/{pkg}/" in filename or f"\\{pkg}\\" in filename:
                return pkg
        fn = getattr(fn, "__wrapped__", None)
    return None


def _importable(pkg):
    """包是否可 import（不 import 它本体）。"""
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def plugin_presence():
    """各厂商插件的在场画像（只读，不触发 import）。

    `initialized` 看的是 `<pkg>.components` 子模块——插件的符号重写发生在
    components 被加载时，顶层包在 sys.modules 里只说明"被 import 过"。
    """
    out = {}
    for pkg in PLUGIN_PACKAGES:
        enable_env = f"{'XSWIFT' if pkg == 'xswift_ext' else 'XME'}_PLUGIN_ENABLE"
        out[pkg] = {
            "importable": _importable(pkg),
            "imported": pkg in sys.modules,
            "initialized": any(name.startswith(pkg + ".components")
                               for name in list(sys.modules)),
            "enable_env": enable_env,
            "enable_env_value": os.environ.get(enable_env),
        }
    return out


def plugin_protection_expected(package="xswift_ext"):
    """XSwift 是否会在本进程接管（= 三个开关的问题是否已被它挡住）。

    判据：包可 import **且** `XSWIFT_PLUGIN_ENABLE` 为真。默认值取 0 ——
    与厂商 `xswift_ext/auto_init.py:25` 的 `os.getenv("XSWIFT_PLUGIN_ENABLE", "0")`
    一致：不设即不启用，不能按"插件装了就会生效"推断。
    """
    env = f"{'XSWIFT' if package == 'xswift_ext' else 'XME'}_PLUGIN_ENABLE"
    return bool(_env_truth(env)) and _importable(package)


def dummy_module_traces():
    """`sys.modules` 里由 XSwift 注入的 `/klx_dummy_lib/...` 假模块清单。

    退役检查项（§15.3-C1）：这些模块对象指向**不存在的路径**，会掩盖真实
    import 失败，也让无包验证的 `sys.modules` 审计失真。返回
    `[(模块名, origin)]`，空列表 = 干净。
    """
    traces = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        spec = getattr(mod, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(mod, "__file__", None) or ""
        if DUMMY_LIB_MARKER in str(origin):
            traces.append((name, str(origin)))
    return sorted(traces)


def export_safety_issues(protected=None):
    """checkpoint 导出安全的逐条核对。返回问题列表（空 = 安全）。

    protected=None 时自动判定 XSwift 是否在场；在场则三条都由插件挡着，
    不报问题（这就是当前生产的形态：LW=1 + 插件开）。
    """
    if protected is None:
        protected = plugin_protection_expected()
    if protected:
        return []
    issues = []
    for rule in EXPORT_ENV_RULES:
        current = _env_truth(rule.env)
        if current is None:
            if rule.unset_is_safe:
                continue
            shown = "未设置"
        elif current == rule.safe_value:
            continue
        else:
            shown = os.environ[rule.env]
        issues.append({"env": rule.env, "current": shown,
                       "required": "1" if rule.safe_value else "0",
                       "symptom": rule.symptom, "auto_fixable": rule.fixable})
    return issues


def apply_export_safe_defaults():
    """给**未设置**且可安全兜底的开关写安全值（setdefault 语义）。

    返回实际写入的 `{env: value}`。只在 XSwift 不在场时动手——插件在场时
    这些开关各有性能代价（mem-efficient 关闭实测峰值 +224/448 MiB），
    没必要为一个已经被挡住的问题付钱。

    不碰 `XTE_GROUPED_GEMM_LARGE_WEIGHT`：它是显式的性能决策，悄悄翻成 0
    会让 MoE 性能无声退化（~12-14%），必须由使用方决定。
    """
    if plugin_protection_expected():
        return {}
    written = {}
    for rule in EXPORT_ENV_RULES:
        if not rule.fixable or rule.env in os.environ:
            continue
        value = "1" if rule.safe_value else "0"
        os.environ[rule.env] = value
        written[rule.env] = value
    return written


def assert_export_safe(auto_fix=True):
    """导出安全契约的闸门：能兜底的兜底，兜不住的 fail-fast。

    返回 `{"applied": 兜底写入的 env, "protected": 是否由插件挡着}`。
    不安全时 raise RuntimeError，消息里带每条的症状与可选处置——
    静默损坏的 checkpoint 比训练起不来贵得多。
    """
    from ttorch.witness import provenance

    applied = apply_export_safe_defaults() if auto_fix else {}
    protected = plugin_protection_expected()
    issues = export_safety_issues(protected=protected)
    provenance.record(provenance.LAYER_BRIDGE, "env_contract", "checkpoint_export",
                      "fail" if issues else "pass", protected=protected,
                      applied=applied, issues=issues)
    if issues:
        lines = [f"  {i['env']}={i['current']}（需要 {i['required']}）→ {i['symptom']}"
                 for i in issues]
        raise ExportUnsafeError(
            "checkpoint 导出不安全：XSwift 未在场，以下开关会让训练在 exit=0、"
            "loss 逐位一致的情况下导出垃圾权重（退役文档 §16.3）：\n"
            + "\n".join(lines)
            + "\n处置：按上面的要求设置环境变量；若必须保留 "
              "XTE_GROUPED_GEMM_LARGE_WEIGHT=1，则不能退役 XSwift（它的 "
              "large-weight shim 是承重件）。")
    return {"applied": applied, "protected": protected}


def assert_export_safe_for_worker(auto_fix=True):
    """只在**训练 worker** 上执行导出安全契约；其余进程原样放行（返回 None）。

    判据是 `RANK`/`LOCAL_RANK`（torchrun/megatron 注入的显式信号，不做启发式
    猜测）——只有训练 worker 会写 checkpoint。CPU 探针、数据解析工具、单测
    进程不该被一个"保存时才兑现"的风险拦住，更不该被悄悄改掉环境变量。

    挂载点收敛（2026-11-19）：本函数是契约的**唯一入口**，由
    `ttorch.activate()` 无条件调用，是否执行只看上面的 worker 判据。
    此前的两处挂法都已撤掉：`big_leaf.swift.enable()` 那份不带 worker
    gate（裸调它的工具进程会被拦、被改环境）；`activate()` 一度把它置于
    `skip_conformance` 之下（显式跳过一致性门禁会静默失去导出保护）。
    收敛后判据单一，派生进程继承父进程已兜底的环境、幂等通过。
    """
    if not (os.environ.get("RANK") or os.environ.get("LOCAL_RANK")):
        return None
    return assert_export_safe(auto_fix=auto_fix)


def retirement_report():
    """退役检查的一次性画像（在场 + 假模块痕迹 + 开关核对）。只读，不改环境。"""
    protected = plugin_protection_expected()
    return {"plugins": plugin_presence(),
            "xswift_protection_expected": protected,
            "dummy_module_traces": dummy_module_traces(),
            "export_safety_issues": export_safety_issues(protected=protected)}
