"""B18 / B21 / B22：厂商插件退役的前置检查。

三件事的回归：
- B18 `implementation_owner`：把"厂商插件在场"与"上游已修复"分开（XSwift 的
  `wrap_model` 是 swift ≥4.3.2 的逐字拷贝，源码 needle `ddp_stream` 会命中，
  旧实现据此判"上游已修复"——归因反了）；
- B21① `dummy_module_traces`：XSwift `import_module_safe` 注入的
  `/klx_dummy_lib/...` 假模块必须被退役检查扫出来；
- B22 导出安全契约：三个开关缺任一都会在 exit=0 + loss 逐位一致的情况下
  导出 ~3.7e33 垃圾权重（退役文档 §16.3），必须在训练前闸住。
"""
import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest

from ttorch.bridge import vendor_retire as vr


_ENV_NAMES = tuple(rule.env for rule in vr.EXPORT_ENV_RULES) + \
    ("XSWIFT_PLUGIN_ENABLE", "RANK", "LOCAL_RANK")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """三个开关 + 插件开关全部清空：契约测试必须从"什么都没设"开始。"""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def unprotected(monkeypatch):
    """XSwift 不在场（退役后的形态）。"""
    monkeypatch.setattr(vr, "plugin_protection_expected", lambda *a, **k: False)


# ---------------------------------------------------------------- B18

def test_vendor_impl_detected_by_module():
    def fn():
        pass
    fn.__module__ = "xswift_ext.components.swift.megatron.utils.megatron_lm_utils"
    assert vr.implementation_owner(fn) == "xswift_ext"


def test_vendor_impl_detected_through_wrapped_chain():
    """第三方在厂商实现外面再套一层时，只看最外层会漏判。"""
    def vendor():
        pass
    vendor.__module__ = "xmegatron_ext.plugin"

    def outer():
        pass
    outer.__wrapped__ = vendor
    assert vr.implementation_owner(outer) == "xmegatron_ext"


def test_upstream_impl_is_not_attributed_to_vendor():
    """上游真修复（swift 自己的模块）不能被记成厂商插件。"""
    def fn():
        pass
    fn.__module__ = "swift.megatron.utils.megatron_lm_utils"
    assert vr.implementation_owner(fn) is None


def test_ddp_fence_declines_for_vendor_impl_without_needle(monkeypatch):
    """关键判别：厂商实现里**没有** needle 时，旧实现会接管（归因无从体现），
    新实现按"厂商在场"拒绝接管并留下 vendor_plugin_present 证据。"""
    from ttorch.bridge.patches import _WRAP_MODEL_ATTR, _WRAP_MODEL_MODULE, \
        _ddp_fence_applies
    from ttorch.witness import provenance

    def wrap_model(args, models):
        return models            # 刻意不含 ddp_stream / wait_stream
    wrap_model.__module__ = "xswift_ext.components.swift.megatron.utils" \
                            ".megatron_lm_utils"

    mod = types.ModuleType(_WRAP_MODEL_MODULE)
    setattr(mod, _WRAP_MODEL_ATTR, wrap_model)
    monkeypatch.setitem(sys.modules, _WRAP_MODEL_MODULE, mod)

    assert _ddp_fence_applies() is False
    reasons = [r["evidence"].get("reason")
               for r in provenance.records(provenance.LAYER_BRIDGE)
               if r["event"] == "patch"]
    assert reasons[-1] == "vendor_plugin_present", \
        "厂商在场必须如实留痕，不能记成上游修复"


# ---------------------------------------------------------------- B21①

def test_dummy_lib_trace_is_detected(monkeypatch):
    """XSwift import 失败时注入的假模块：origin 指向不存在的 /klx_dummy_lib。"""
    name = "_tt_fake_dummy_pkg"
    dummy_path = f"{vr.DUMMY_LIB_MARKER}/{name}.py"
    loader = importlib.machinery.SourceFileLoader(name, dummy_path)
    spec = importlib.util.spec_from_loader(name, loader, origin=dummy_path)
    spec.has_location = True
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)

    traces = dict(vr.dummy_module_traces())
    assert name in traces and vr.DUMMY_LIB_MARKER in traces[name]


def test_clean_process_has_no_dummy_trace():
    assert vr.dummy_module_traces() == []


# ---------------------------------------------------------------- B22

def test_unset_switches_report_two_issues(unprotected):
    """未设置 = 厂商默认（mem-efficient 开、pin 开）= 不安全；LW 未设默认 0 安全。"""
    envs = {i["env"] for i in vr.export_safety_issues()}
    assert envs == {"HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT",
                    "XTE_DISABLE_FAST_BF16_CACHE"}


def test_large_weight_on_is_an_issue_and_not_auto_fixable(monkeypatch, unprotected):
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")
    issue = next(i for i in vr.export_safety_issues()
                 if i["env"] == "XTE_GROUPED_GEMM_LARGE_WEIGHT")
    assert issue["auto_fixable"] is False, "LW=1 是显式性能决策，不能被悄悄翻掉"
    assert "experts" in issue["symptom"]


def test_safe_config_has_no_issue(monkeypatch, unprotected):
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "0")
    monkeypatch.setenv("HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT", "1")
    monkeypatch.setenv("XTE_DISABLE_FAST_BF16_CACHE", "1")
    assert vr.export_safety_issues() == []


def test_plugin_presence_suppresses_issues(monkeypatch):
    """当前生产形态（LW=1 + 插件开）：由 XSwift 挡着，不该报问题也不该改环境。"""
    monkeypatch.setattr(vr, "plugin_protection_expected", lambda *a, **k: True)
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")
    assert vr.export_safety_issues() == []
    assert vr.apply_export_safe_defaults() == {}


def test_assert_auto_fixes_then_passes(unprotected):
    result = vr.assert_export_safe()
    assert result["applied"] == {"HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT": "1",
                                 "XTE_DISABLE_FAST_BF16_CACHE": "1"}
    assert vr.export_safety_issues() == []


def test_assert_raises_on_large_weight(monkeypatch, unprotected):
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")
    with pytest.raises(RuntimeError, match="XTE_GROUPED_GEMM_LARGE_WEIGHT"):
        vr.assert_export_safe()


def test_assert_raises_on_explicitly_unsafe_mem_efficient(monkeypatch, unprotected):
    """显式关掉保护比不设更危险（使用方以为自己在省显存）——不能被 setdefault 掩盖。"""
    monkeypatch.setenv("HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT", "0")
    with pytest.raises(RuntimeError, match="lm_head"):
        vr.assert_export_safe()


def test_protection_requires_both_env_and_package(monkeypatch):
    """插件装了但 XSWIFT_PLUGIN_ENABLE 未设 → 厂商 auto_init 默认 0，不会生效。"""
    monkeypatch.setattr(vr, "_importable", lambda pkg: True)
    assert vr.plugin_protection_expected() is False
    monkeypatch.setenv("XSWIFT_PLUGIN_ENABLE", "1")
    assert vr.plugin_protection_expected() is True


def test_worker_gate_skips_non_training_process(monkeypatch, unprotected):
    """CPU 探针/工具进程（无 RANK）不该被一个"保存时才兑现"的风险拦住，
    更不该被悄悄改掉环境变量。"""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")

    assert vr.assert_export_safe_for_worker() is None
    assert "HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT" not in os.environ


def test_worker_gate_fires_with_rank(monkeypatch, unprotected):
    """torchrun/megatron 注入 RANK = 真训练 worker = 会写 checkpoint。"""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")

    with pytest.raises(RuntimeError, match="XTE_GROUPED_GEMM_LARGE_WEIGHT"):
        vr.assert_export_safe_for_worker()


def test_worker_gate_accepts_local_rank_only(monkeypatch, unprotected):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")

    result = vr.assert_export_safe_for_worker()
    assert result is not None and result["applied"]


# ------------------------------------------- activate() 挂载点（唯一入口）

def _stub_activate_stages(monkeypatch):
    """stub 掉 activate() 的重阶段，只让契约阶段真实运行（wiring 测试）。"""
    from ttorch.bridge import bootstrap, framework_patch, patches
    from ttorch.provider import fn_binding, manager
    from ttorch.utils import _auto_init
    monkeypatch.setattr(bootstrap, "activate", lambda **kwargs: None)
    monkeypatch.setattr(manager, "activate", lambda: None)
    monkeypatch.setattr(fn_binding, "install", lambda: None)
    monkeypatch.setattr(patches, "register_all", lambda: None)
    monkeypatch.setattr(framework_patch, "install", lambda verbose: None)
    monkeypatch.setattr(_auto_init, "mark_activated", lambda: None)
    for name in ("TT_BIG_LEAF", "TT_RESTORE_VENDOR"):
        monkeypatch.delenv(name, raising=False)


def test_activate_runs_contract_on_worker(monkeypatch, unprotected):
    """worker（RANK 在场）+ LW=1 无插件 → activate() 必须 fail-fast。"""
    import ttorch
    _stub_activate_stages(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")

    with pytest.raises(vr.ExportUnsafeError, match="XTE_GROUPED_GEMM_LARGE_WEIGHT"):
        ttorch.activate()


def test_activate_skips_contract_off_worker(monkeypatch, unprotected):
    """非 worker（无 RANK）：不安全配置也不拦、不改环境（保存时才兑现的风险）。"""
    import ttorch
    _stub_activate_stages(monkeypatch)
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")

    ttorch.activate()

    assert "HYDRA_DISABLE_FAST_FC_MEMORY_EFFICIENT" not in os.environ


def test_activate_skip_conformance_still_protects_worker(monkeypatch, unprotected):
    """skip_conformance 只跳一致性门禁，不许静默失去导出保护（2026-11-19 解耦）。"""
    import ttorch
    _stub_activate_stages(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("XTE_GROUPED_GEMM_LARGE_WEIGHT", "1")

    with pytest.raises(vr.ExportUnsafeError):
        ttorch.activate(skip_conformance=True)


def test_auto_init_turns_export_unsafe_into_systemexit(monkeypatch):
    """`.pth`/sitecustomize 路径对契约失败必须致命。

    `site` 的 `except Exception` 会把普通异常吞成一行警告、训练照跑——那正是
    回到静默损坏。`SystemExit` 是 BaseException，能穿透它。
    """
    import ttorch
    from ttorch.utils import _auto_init

    monkeypatch.setenv("TT_PLUGIN_ENABLE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("TT_ACTIVATED_PID", raising=False)
    monkeypatch.setattr(ttorch, "activate", lambda *a, **k: (_ for _ in ()).throw(
        vr.ExportUnsafeError("XTE_GROUPED_GEMM_LARGE_WEIGHT=1 → 专家权重垃圾值")))

    with pytest.raises(SystemExit, match="拒绝启动"):
        _auto_init._init()


def test_auto_init_keeps_other_failures_non_fatal(monkeypatch):
    """其余激活失败仍按原策略降级继续——否则 pip/conda 等进程会被连带打挂。"""
    import ttorch
    from ttorch.bridge import bootstrap
    from ttorch.utils import _auto_init

    monkeypatch.setenv("TT_PLUGIN_ENABLE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("TT_ACTIVATED_PID", raising=False)
    monkeypatch.setattr(bootstrap, "mark_failed", lambda *a, **k: None)
    monkeypatch.setattr(ttorch, "activate", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("平台检测失败")))

    assert _auto_init._init() is False


def test_retirement_report_shape(unprotected):
    report = vr.retirement_report()
    assert set(report) == {"plugins", "xswift_protection_expected",
                           "dummy_module_traces", "export_safety_issues"}
    assert set(report["plugins"]) == set(vr.PLUGIN_PACKAGES)
