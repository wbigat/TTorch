"""激活作用域与失败原子性单测（D1 分类 / D3 回滚 / D11 来源标识）。

起源：2026-10-21 PR review 缺陷分析
- D3：绑定中途失败 → 半激活态（部分符号已改写、`is_active()=False`）且进程继续；
- D1：spawn 辅助进程重跑含真实设备操作的 conformance；fork 派生静默继承；
- D11：provenance 无 pid/rank，多进程下无法区分来源。

本文件是这三条的回归网：
1. `test_rebind_rollback_*`：回滚必须覆盖**源头 + 扫尾**（只回滚源头会让第三方
   模块里的引用仍指向已卸载的门面，比不回滚更隐蔽——实测确认过）；
2. `test_bootstrap_rollback_*`：绑定失败 → 符号面还原、`_ACTIVATION` 置 None、
   provenance 留 rollback 事件；
3. `test_provenance_*`：信封含 pid/rank；
4. `test_scope_classification`：判据只用显式信号（RANK / TT_ACTIVATED_PID）。

运行：pytest tests/unit/test_activation_scope.py -v
"""
import importlib
import os
import sys
import types

import pytest

from ttorch.utils import _rebind


@pytest.fixture(autouse=True)
def _clean_rebind():
    """每个测试前后保证 _rebind 状态干净。"""
    _rebind.rollback()
    _rebind.clear()
    yield
    _rebind.rollback()
    _rebind.clear()


def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------- D3 回滚

def test_rebind_rollback_restores_source_and_swept_modules():
    """D3 核心：回滚必须同时恢复源头与扫尾改写的模块（顺序不可反）。"""
    src = _fake_module("fw_src", fn=lambda: 1)
    # 第三方模块 from-import 了源头符号（扫尾的作用对象）
    consumer = _fake_module("fw_consumer", fn=src.fn)

    new = lambda: 2  # noqa: E731
    _rebind.rebind(src, "fn", new, owner="bridge", sweep_packages=("fw_consumer",))

    # 绑定生效：源头与消费者都被改写
    assert src.fn is new
    assert consumer.fn is new

    n_source, n_sweep = _rebind.rollback(owner="bridge")

    assert n_source == 1, "源头必须恢复"
    assert n_sweep == 1, "扫尾改写的模块必须恢复（否则残留指向已卸载的门面）"
    assert src.fn is not new and consumer.fn is not new
    assert src.fn is consumer.fn, "回滚后两者应回到同一个原实现"
    assert not _rebind.has_origin(src, "fn"), "留痕必须清除"
    assert not _rebind.has_origin(consumer, "fn"), "扫尾留痕必须清除"
    assert _rebind.snapshot() == {}


def test_rebind_rollback_only_target_owner():
    """多 writer 场景：指定 owner 时只回滚该 writer 的源头条目。"""
    src = _fake_module("fw_multi", fn=lambda: 1)
    a, b = (lambda: 2), (lambda: 3)
    _rebind.rebind(src, "fn", a, owner="bridge")
    _rebind.rebind(src, "fn", b, owner="provider")

    n_source, _ = _rebind.rollback(owner="provider")
    assert n_source == 1
    assert src.fn is a, "只回滚 provider 那一层，回到 bridge 的门面"


def test_bootstrap_rollback_on_bind_failure():
    """绑定中途失败 → 符号面还原 + _ACTIVATION 置 None + 留 rollback 证据。"""
    import torch

    # Bug1 修复：无卡环境本用例必失败——bootstrap 完整激活依赖真实 platform
    # （detect 无命中会在 platform 阶段就抛，走不到本用例要测的"绑定失败"）。
    # 此用例本质是端到端激活测试，跳过无卡环境。
    if not torch.cuda.is_available():
        pytest.skip("无加速设备：完整激活需要真实 platform")

    import ttorch
    import ttorch.bridge.bootstrap as bootstrap
    import ttorch.bridge.cuda_compat as cuda_compat
    from ttorch.witness import provenance

    # 幂等保护：若已激活，先复位（本用例要完整走一遍激活）
    bootstrap._ACTIVATION = None

    real_impl_for = cuda_compat._impl_for
    calls = {"n": 0}

    def flaky(entry, original):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("模拟第 3 条绑定失败")
        return real_impl_for(entry, original)

    cuda_compat._impl_for = flaky
    try:
        with pytest.raises(RuntimeError, match="模拟第 3 条绑定失败"):
            ttorch.activate()
    finally:
        cuda_compat._impl_for = real_impl_for

    # ① 激活标记必须复位（否则 is_active() 与符号面不一致 = 半激活态）
    assert bootstrap._ACTIVATION is None
    assert ttorch.is_active() is False
    # ② 符号面必须还原：前 2 条改过的符号回到非门面
    from ttorch.platform import interface
    managed = ("synchronize", "set_device", "current_device", "Stream", "Event",
               "manual_seed", "empty_cache", "current_stream", "get_rng_state")
    leaked = [n for n in managed
              if getattr(torch.cuda, n, None) is getattr(interface, n, object())]
    assert leaked == [], f"仍有符号指向 TT 门面（回滚不完整）: {leaked}"
    # ③ 快照与证据
    assert _rebind.snapshot() == {}
    rolls = [r for r in provenance.records(provenance.LAYER_BRIDGE)
             if r["outcome"] == "rollback"]
    assert rolls, "必须留一条 rollback 证据"


def test_activation_failure_poisons_process_and_assert_healthy():
    """D3 建议②：激活失败必须留下进程级中毒标记，`assert_healthy()` 可查。"""
    import ttorch
    import ttorch.bridge.bootstrap as bootstrap

    saved = bootstrap._LAST_FAILURE
    try:
        bootstrap._LAST_FAILURE = None
        assert bootstrap.is_healthy()
        ttorch.assert_healthy()          # 健康态不应抛

        err = RuntimeError("模拟激活失败")
        bootstrap.mark_failed(err)
        assert not bootstrap.is_healthy()
        assert "模拟激活失败" in bootstrap.failure()["error"]
        with pytest.raises(RuntimeError, match="激活失败且已回滚"):
            ttorch.assert_healthy()
    finally:
        bootstrap._LAST_FAILURE = saved


def test_successful_retry_clears_failure_mark():
    """D3 恢复语义：失败一次（中毒）→ 重试成功 → 标记必须清除。

    2026-10-22 审视发现的缺陷：`_LAST_FAILURE` 只在失败时写、成功时不清，
    会导致"失败→重试成功"后 assert_healthy() 仍误报。本用例用真实激活
    （KL3 有设备）验证成功路径清标记；无平台环境跳过。
    """
    import ttorch
    import ttorch.bridge.bootstrap as bootstrap
    from ttorch.platform import manager as pfm

    if not pfm.is_activated():
        pytest.skip("无已激活平台（非真实设备环境），跳过真实激活验证")

    saved = bootstrap._LAST_FAILURE
    try:
        bootstrap.mark_failed(RuntimeError("第一次失败"))
        assert not bootstrap.is_healthy()
        ttorch.activate()   # 幂等路径：已激活直接返回，但也应清标记
        assert bootstrap.is_healthy(), "成功激活后中毒标记必须清除"
        ttorch.assert_healthy()   # 不再抛
    finally:
        bootstrap._LAST_FAILURE = saved


# ---------------------------------------------------------------- D11 信封

def test_provenance_envelope_has_pid_and_rank():
    """D11：信封自带 pid/rank，多进程可区分来源。"""
    from ttorch.witness import provenance

    provenance.record(provenance.LAYER_BRIDGE, "unit", "test.subject", "ok", k=1)
    r = [x for x in provenance.records() if x["subject"] == "test.subject"][-1]
    assert r["pid"] == os.getpid()
    assert "rank" in r, "rank 字段必须存在（非分布式时为 None）"
    # None 与 0 必须可区分
    assert r["rank"] is None or isinstance(r["rank"], int)


def test_provenance_fork_reset_gives_clean_per_process_stream():
    """D11 后果②：fork 继承会让 seq 从父进程值继续递增（多 worker 聚合重复）。

    fork_reset 后必须：继承记录清空、seq 从 1 重新计数、留一条 fork 事件
    指明来源（parent_pid）——"一份 JSONL = 一个进程的事件流"。
    """
    from ttorch.witness import provenance

    provenance.record(provenance.LAYER_BRIDGE, "bind", "parent.symbol", "remap")
    provenance.record(provenance.LAYER_BRIDGE, "bind", "parent.symbol2", "remap")
    assert len(provenance.records()) >= 2

    provenance.fork_reset("4242")
    recs = provenance.records()
    assert len(recs) == 1, f"继承的父进程记录必须丢弃，实际剩 {len(recs)} 条"
    r = recs[0]
    assert r["event"] == "fork" and r["outcome"] == "inherited"
    assert r["seq"] == 1, "seq 必须从 1 重新计数"
    assert r["evidence"]["parent_pid"] == "4242"

    # 后续事件 seq 连续递增（不重复）
    provenance.record(provenance.LAYER_BRIDGE, "call", "worker.sym", "called")
    seqs = [x["seq"] for x in provenance.records()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_export_path_includes_pid_and_rank(monkeypatch, tmp_path):
    """D11：默认导出名带 pid/rank 后缀，避免多 rank 互相覆盖。"""
    from ttorch.witness import provenance

    monkeypatch.setenv("RANK", "3")
    name = provenance.default_export_path()
    assert str(os.getpid()) in name
    assert "r3" in name
    monkeypatch.delenv("RANK", raising=False)
    assert "r" not in provenance.default_export_path().rsplit(".", 2)[-2]


# ---------------------------------------------------------------- D1 分类

def test_scope_classification_explicit_signals(monkeypatch):
    """D1：判据只用显式信号（RANK / TT_ACTIVATED_PID），不做启发式猜测。"""
    from ttorch.utils import _auto_init

    monkeypatch.setenv("RANK", "0")
    kind, _ = _auto_init._classify_process()
    assert kind == "primary", "RANK 存在 → 训练 rank，走重档"

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv(_auto_init._ACTIVATED_PID_ENV, str(os.getpid() + 9999))
    kind, reason = _auto_init._classify_process()
    assert kind in {"derived", "inherited"}, f"派生标记命中，实际={kind}（{reason}）"

    monkeypatch.setenv(_auto_init._ACTIVATED_PID_ENV, str(os.getpid()))
    kind, _ = _auto_init._classify_process()
    assert kind == "primary", "pid 与标记一致 → 就是激活者本人"

    monkeypatch.delenv(_auto_init._ACTIVATED_PID_ENV, raising=False)
    kind, _ = _auto_init._classify_process()
    assert kind == "primary", "无标记 → 首次激活进程"


def test_d10_convergence_profile_observability():
    """D10：收束画像——符号面三态可见 + 在场不可收束组件可检测。"""
    import sys
    from ttorch.bridge import surface
    profile = surface.coverage_profile()
    assert profile["closed"] is True
    assert profile["managed"] > 0 and profile["denied"] > 0 and profile["exempt"] > 0
    assert profile["managed"] + profile["denied"] + profile["exempt"] == 122
    # 不可收束组件检测：模拟组件在场，应被捕获且带原因
    sys.modules.setdefault("transformer_engine", None)
    try:
        unconverged = surface.unconverged_components()
        names = [c for c, _ in unconverged]
        assert "transformer_engine" in names, "在场不可收束组件应被检测到"
        assert all(isinstance(r, str) and r for _, r in unconverged), "每个组件须带原因"
    finally:
        sys.modules.pop("transformer_engine", None)
