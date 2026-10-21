"""替换动作的**唯一实现**（owner 留痕 + 扫尾 + 完整性检测基准）。

为什么抽出来（Rewrite 设计文档 §4-A，2026-10-20 实测）：

替换动作此前有 3 份独立实现——`cuda_compat.activate`（bridge，torch.cuda 符号）、
`fn_binding._bind_one`（provider，框架函数符号）、`call_trace` 探针（覆盖门面）。
两处 sweep 循环近乎逐行重复（遍历 sys.modules → 白名单前缀过滤 → id() 比对 →
双 setattr 留痕），且带来两个实测后果：

1. **`_origin_` 前缀撞名**：两个 writer 用同一前缀，symbrewrite 又用 `__origin_`。
   同一符号被两方先后绑定时，后写者会把前写者的门面存成"原实现"——**回滚就
   回滚错了**。当前无符号重叠（潜在雷非现炸），但没有任何机制保证不重叠。
2. **完整性检测有盲区**：`_POST_BIND` 快照只在 cuda_compat 里写，实测
   `assert_bindings_intact()` 覆盖 15 条中 **provider 函数级 0 条**——第三方
   （symbrewrite / Liger / 框架 kernelize）覆盖掉 provider 的绑定，TT 无从察觉。

本模块提供唯一入口 `rebind()` 与唯一快照 `snapshot()`：所有 writer 都改调它，
留痕带 owner（属性名即可看出"谁绑的"），检测一次覆盖全部 writer。

**不引入** RewriteSpec / 事务 / 回滚栈——当前只有 2 个 writer、4 种落点，DSL
的维护成本高于它解决的问题（Rewrite 文档 §6 明确降级）。
"""

__all__ = ["rebind", "snapshot", "origin_attr", "has_origin"]

# 留痕属性前缀：带 owner，消灭撞名且"谁绑的"从属性名即可见
_ORIGIN_PREFIX = "_tt_origin_"

# 全部 writer 的绑定快照 {(module_name, attr): (bound_obj, owner)}
# ——完整性检测的唯一基准（替代 cuda_compat 局部的 _POST_BIND）
_SNAPSHOT = {}


def origin_attr(owner, attr):
    """留痕属性名：`_tt_origin_<owner>_<attr>`。"""
    return f"{_ORIGIN_PREFIX}{owner}_{attr}"


def _find_other_owner(module, attr, owner):
    """是否已有**别的** owner 留痕（同符号多方接管的冲突探测）。"""
    pre = _ORIGIN_PREFIX
    suf = f"_{attr}"
    for name in dir(module):
        if name.startswith(pre) and name.endswith(suf):
            other = name[len(pre):-len(suf)]
            if other != owner:
                return other
    return None


def rebind(module, attr, new, *, owner, sweep_packages=()):
    """执行一次符号替换：源头 setattr + 扫尾 + 带 owner 留痕。

    - 留痕写 `_tt_origin_<owner>_<attr>`（非 `_origin_<attr>`），消灭撞名；
    - 扫尾修白名单包内持有**原对象**的陈旧 from-import 引用；
    - 写入全局快照，供 `assert_bindings_intact()` 一次覆盖全部 writer。

    发现已有他人 owner 留痕时记一条 `conflict` 证据但**继续绑定**（保持现有
    行为不变）；真正的优先级/overlay 策略留待出现真实重叠时再定（Rewrite
    文档 §6 挂起项）——当前无符号重叠，不提前建机制。

    返回 swept 的模块数。
    """
    from ttorch.witness import provenance

    original = getattr(module, attr)
    other = _find_other_owner(module, attr, owner)
    if other is not None:
        provenance.record(provenance.LAYER_BRIDGE, "bind",
                          f"{module.__name__}.{attr}", "conflict",
                          owner=owner, other_owner=other,
                          detail=f"该符号已被 {other} 绑定过；owner 留痕可区分，"
                                 f"但回滚语义需人工确认（当前无优先级策略）")
    setattr(module, origin_attr(owner, attr), original)
    setattr(module, attr, new)
    swept = _sweep(module, attr, new, original, sweep_packages)
    _SNAPSHOT[(module.__name__, attr)] = (new, owner)
    return swept


def _sweep(module, attr, new, original, packages):
    """扫尾：白名单包内仍持有原对象的模块，一并换掉并留痕。

    packages 为 None 表示不限（仅 mock demo 演示机制用）；真实环境必须显式
    给出白名单（框架包），否则全量扫 sys.modules 会越界改写 torch 内部/
    第三方库的引用，风险不可控。
    """
    import sys
    swept = 0
    for m in list(sys.modules.values()):
        if m is module:
            continue
        name = getattr(m, "__name__", "")
        if packages is not None and not any(
                name == p or name.startswith(p + ".") for p in packages):
            continue
        try:
            if getattr(m, attr, None) is original:
                setattr(m, origin_attr("sweep", attr), original)
                setattr(m, attr, new)
                swept += 1
        except Exception:
            continue
    return swept


def snapshot():
    """完整性检测的唯一基准：{(module_name, attr): (bound_obj, owner)}。"""
    return dict(_SNAPSHOT)


def has_origin(module, attr):
    """该模块上是否存在 TT 写入的 owner 留痕（任一 owner，含扫尾）。

    扫尾验收（cuda_compat.verify_sweep）据此判定"这个模块确实持有过被替换的
    旧实现"——留痕名随 owner 变（`_tt_origin_<owner>_<attr>`），不能按固定
    前缀 `_origin_` 找。
    """
    pre, suf = _ORIGIN_PREFIX, f"_{attr}"
    return any(n.startswith(pre) and n.endswith(suf) for n in dir(module))
