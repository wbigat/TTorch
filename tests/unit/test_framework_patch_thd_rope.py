"""THD RoPE cp_group 补齐的回归（B1）。

根因（源码核实）：
  mcore_bridge/patcher.py:202  cp_group = kwargs.pop('cp_group', None)
  —— pop 后未放回；non-batched 分支（:211）把 kwargs 转给 Megatron 上游时
  cp_group 已丢失。
上游对此为硬要求（megatron/.../rope_utils.py:200-201）：
  if cp_group is None: raise ValueError("cp_group must be provided ...")
且用 cp_group.rank() 按 CP rank 切 freqs —— 丢失即 ValueError，或（调用方
自带默认值时）freqs 取错片造成静默数值错。

本文件不依赖真实 megatron/mcore_bridge：用同构的最小复现（同样的 pop 语义、
同样的 fallback 分支、同样的闭包变量名）验证补丁行为。
"""
import functools

import pytest

from ttorch.bridge import framework_patch as fw
from ttorch.bridge import patches
from ttorch.bridge.patches import thd_rope_cp_group_wrapper


class _FakeGroup:
    """最小 ProcessGroup 替身：只需 size()/rank()。"""

    def __init__(self, size=2, rank=1):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _upstream_thd(t, cu_seqlens, freqs, *args, **kwargs):
    """同构上游实现：cp_group 缺失即报错（对齐 rope_utils.py:200-201）。"""
    cp_group = kwargs.get("cp_group")
    if cp_group is None:
        raise ValueError("cp_group must be provided for THD format RoPE")
    return ("upstream", cp_group.size(), cp_group.rank())


def _make_mcore_bridge_patched(use_batched=False):
    """构造带缺陷的 mcore_bridge patch（闭包变量名与真实一致）。"""
    _origin_apply_rotary_pos_emb_thd = _upstream_thd

    def _apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, *args, **kwargs):
        cp_group = kwargs.pop("cp_group", None)   # 缺陷：pop 后不放回
        cp_size = cp_group.size() if cp_group is not None else 1
        if not use_batched:
            # fallback 到上游——此时 kwargs 已无 cp_group
            return _origin_apply_rotary_pos_emb_thd(
                t, cu_seqlens, freqs, *args, **kwargs)
        return ("batched", cp_size)

    return _apply_rotary_pos_emb_thd


def test_reproduces_upstream_failure_without_patch():
    """先证明缺陷真实存在：未打补丁时 fallback 路径必然 ValueError。"""
    patched = _make_mcore_bridge_patched(use_batched=False)
    with pytest.raises(ValueError, match="cp_group must be provided"):
        patched(None, None, None, cp_group=_FakeGroup())


def _make_upstream_fixed_patched():
    """构造**社区已修复形态**的 mcore_bridge patch（对齐上游 1.6.0+，PR #146）。

    上游修复逻辑：pop 出 cp_group；为 None 时用 parallel_state 兜底；
    fallback 显式回传 ``cp_group=cp_group``。
    """
    _origin_apply_rotary_pos_emb_thd = _upstream_thd

    def _apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, *args, **kwargs):
        cp_group = kwargs.pop("cp_group", None)
        if cp_group is None:
            cp_group = _FakeGroup(1, 0)   # 对应上游 mpu.get_context_parallel_group()
        return _origin_apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, *args, cp_group=cp_group, **kwargs)

    return _apply_rotary_pos_emb_thd


def test_applies_when_rejects_upstream_fixed_version(monkeypatch):
    """升级安全（2026-11-04 用户决策）：上游 ≥1.6.0 已含修复时**不再接管**。

    依据是源码形态探测（fallback 显式回传 cp_group=cp_group），不是版本号——
    版本号在内部 fork / 定制打包下会说谎，源码形态不会。探测不到修复才进入
    后续的闭包 cell 探测。
    """
    import sys
    import types

    mod = types.ModuleType(patches._THD_MODULE)
    mod._apply_rotary_pos_emb_thd = _make_upstream_fixed_patched()
    monkeypatch.setitem(sys.modules, patches._THD_MODULE, mod)
    assert patches._thd_rope_applies() is False, "上游已修复仍接管"


def test_applies_when_accepts_defective_version(monkeypatch):
    """对照组：缺陷形态（pop 不放回）在场时前置判据必须放行接管。"""
    import sys
    import types

    mod = types.ModuleType(patches._THD_MODULE)
    mod._apply_rotary_pos_emb_thd = _make_mcore_bridge_patched(use_batched=False)
    monkeypatch.setitem(sys.modules, patches._THD_MODULE, mod)
    assert patches._thd_rope_applies() is True


@pytest.fixture
def cp_group_from_parallel_state(monkeypatch):
    """模拟"分布式已初始化"：让 _current_cp_group() 返回受控的组。

    单测进程没有 CP 进程组，而本补丁的兜底值正是从 parallel_state 取的。
    不打这个桩，测到的只是"拿不到组"的分支，覆盖不到修复本体。
    """
    group = _FakeGroup(4, 3)
    monkeypatch.setattr(patches, "_current_cp_group", lambda: group)
    return group


def test_patch_restores_cp_group_on_fallback(cp_group_from_parallel_state):
    """核心：fallback 路径拿回 cp_group，且 size/rank 来自 parallel_state。"""
    patched = _make_mcore_bridge_patched(use_batched=False)
    wrapped = thd_rope_cp_group_wrapper(patched)
    assert wrapped is not patched, "补丁未接管（未找到 mcore_bridge 闭包？）"
    kind, size, rank = wrapped(None, None, None, cp_group=_FakeGroup(4, 3))
    assert (kind, size, rank) == ("upstream", 4, 3)


def test_parallel_state_failure_matches_community_behavior(monkeypatch):
    """边界行为与社区修复逐点一致（2026-11-04 用户决策：完全对齐 PR #146）。

    社区修法对 ``mpu.get_context_parallel_group()`` 是**无条件调用**——
    parallel_state 未初始化时抛 ``AssertionError``。本补丁不得吞掉它、也不得
    改写成上游的 ValueError：打过补丁的 ≤1.5.x 必须与 ≥1.6.0 行为一致，
    升级才是行为无感的。

    注意打桩打在 **parallel_state 层**而非 ``_current_cp_group``——后者整体
    替换会连旧实现的 try/except 一起换掉，测不到真实接缝（红绿校验因此成立）。

    用 sys.modules 伪模块而非真 import megatron：全量套件中其他测试可能已在
    sys.modules 留了 ``transformer_engine`` 假模块，真 import megatron.core 的
    依赖链会被污染（本测试曾因此在全量下失败、单跑通过——测试隔离缺陷）。
    """
    import sys
    import types

    def _uninitialized():
        raise AssertionError("context parallel group is not initialized")

    fake_ps = types.ModuleType("megatron.core.parallel_state")
    fake_ps.get_context_parallel_group = _uninitialized
    fake_core = types.ModuleType("megatron.core")
    fake_core.parallel_state = fake_ps
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", fake_ps)

    patched = _make_mcore_bridge_patched(use_batched=False)
    wrapped = thd_rope_cp_group_wrapper(patched)
    with pytest.raises(AssertionError, match="context parallel group"):
        wrapped(None, None, None, cp_group=_FakeGroup())


def test_patch_preserves_caller_supplied_group(monkeypatch):
    """不覆盖：上游侧已拿到 cp_group 时，补丁不得改写成 parallel_state 的值。

    对应 mcore_bridge 未来把 pop 改成 get 的情形——届时上游直接拿到调用方的
    组，补丁应是无害透传层。
    """
    monkeypatch.setattr(patches, "_current_cp_group", lambda: _FakeGroup(99, 98))

    # 构造"不丢参数"的 mcore_bridge（pop 后放回，即上游已修复的形态）
    _origin_apply_rotary_pos_emb_thd = _upstream_thd

    def fixed_bridge(t, cu_seqlens, freqs, *args, **kwargs):
        cp_group = kwargs.pop("cp_group", None)
        kwargs["cp_group"] = cp_group          # 放回
        return _origin_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs,
                                                *args, **kwargs)

    wrapped = thd_rope_cp_group_wrapper(fixed_bridge)
    _, size, rank = wrapped(None, None, None, cp_group=_FakeGroup(8, 5))
    assert (size, rank) == (8, 5), "补丁覆盖了调用方传入的 cp_group"


def test_double_apply_does_not_stack(cp_group_from_parallel_state):
    """重复激活不叠层：第二次 wrapper 调用必须直接返回 orig（不接管）。

    对应 activate() 可被重复调用（显式 + .pth 自动激活）的现实；叠层会让
    同一调用被补两次参数、provenance 重复计数。本例钉住既有行为
    （`_tt_cp_group_fixed` 标记），非红绿新增。
    """
    patched = _make_mcore_bridge_patched(use_batched=False)
    w1 = thd_rope_cp_group_wrapper(patched)
    assert w1 is not patched, "首次未接管"
    w2 = thd_rope_cp_group_wrapper(patched)
    assert w2 is patched, "重复调用产生了叠层"
    # 行为仍正确（补一次且只补一次）
    kind, size, rank = patched(None, None, None, cp_group=_FakeGroup(9, 9))
    assert (kind, size, rank) == ("upstream", 4, 3)


def test_patch_does_not_change_batched_branch():
    """batched 分支行为不得被改变（补丁只补参数，不碰分支判据）。"""
    patched = _make_mcore_bridge_patched(use_batched=True)
    wrapped = thd_rope_cp_group_wrapper(patched)
    assert wrapped(None, None, None, cp_group=_FakeGroup(2, 0)) == ("batched", 2)


def test_declines_when_no_mcore_bridge_patch():
    """链上没有 mcore_bridge patch 时不接管（返回原函数本身）。"""
    def plain(t, cu_seqlens, freqs, *args, **kwargs):
        return "plain"

    assert thd_rope_cp_group_wrapper(plain) is plain


def test_finds_origin_through_wrapper_chain():
    """mcore_bridge patch 被别人再包一层时仍能找到上游实现。"""
    patched = _make_mcore_bridge_patched(use_batched=False)

    @functools.wraps(patched)
    def outer(*a, **kw):
        return patched(*a, **kw)

    outer.__wrapped__ = patched
    wrapped = thd_rope_cp_group_wrapper(outer)
    assert wrapped is not outer, "未能穿过 wrapper 链找到 mcore_bridge 闭包"


def test_patch_mechanism_skips_absent_symbol():
    """机制层：符号不存在时记 skipped，不抛异常（失效可见但不影响功能）。"""
    import sys
    import types

    mod = types.ModuleType("_tt_fake_rope_mod")
    sys.modules["_tt_fake_rope_mod"] = mod
    try:
        p = fw.FunctionPatch(dotted="_tt_fake_rope_mod.nonexistent",
                             module="_tt_fake_rope_mod", attr="nonexistent",
                             wrapper=lambda o: o, intent="test")
        assert fw._apply_one(p) is False
        # 查 _APPLIED 而非 patch_status()：后者只遍历已注册清单，这里是
        # 直接调 _apply_one 的机制测试，未走 register_patch。
        assert "符号不存在" in fw._APPLIED.get(p.dotted, "")
    finally:
        sys.modules.pop("_tt_fake_rope_mod", None)
        fw._APPLIED.pop("_tt_fake_rope_mod.nonexistent", None)


def test_patch_mechanism_survives_wrapper_exception():
    """机制层：wrapper 本身抛异常必须收敛为"不接管"，不得穿透。"""
    import sys
    import types

    mod = types.ModuleType("_tt_fake_rope_mod2")
    mod.target = lambda: "orig"
    sys.modules["_tt_fake_rope_mod2"] = mod
    try:
        def boom(orig):
            raise RuntimeError("wrapper 内部故障")

        p = fw.FunctionPatch(dotted="_tt_fake_rope_mod2.target",
                             module="_tt_fake_rope_mod2", attr="target",
                             wrapper=boom, intent="test")
        assert fw._apply_one(p) is False
        assert mod.target() == "orig", "异常后原实现被破坏"
        assert "error" in fw._APPLIED.get(p.dotted, "")
    finally:
        sys.modules.pop("_tt_fake_rope_mod2", None)
        fw._APPLIED.pop("_tt_fake_rope_mod2.target", None)
