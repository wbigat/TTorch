"""函数级绑定坐标表（按目标框架分组）——"TT 对外改动了什么"的唯一答案。

为什么集中（Symbrewrite 吸收文档 §2.1）：坐标原本散在 `impls/<语义>/base.py`
里——provider 只有 1 个时无所谓，涨到 4 个后"TT 对 megatron 动了哪些手脚"
**没有任何一处能回答**，只能 grep。symbrewrite 用 `plugins/<框架>/` 的组织
天然能回答这个问题（看一眼目录即可）。

吸收方式不是推翻现有切法：`impls/<语义>/` 按"一个语义的多硬件实现"切是对的，
保留不动；本表只把"**改谁**"这一维抽出来集中，可 diff、可导出评审。

version_window 为什么只是字符串（§2.2）：TT 绑的 `_apply_rotary_pos_emb_bshd`
是 megatron **私有函数**（前导下划线），路径/函数名/签名随版本漂移是常态，
上游无兼容承诺——symbrewrite 为此维护了 6 个 transformers 版本目录。当前先把
版本**写下来**让人能看见即可；真遇到两版共存再加解析，不建版本匹配引擎。
"""

__all__ = ["MEGATRON", "bind_points", "target", "window", "summary"]

# 目标框架 -> {语义: (模块路径, 符号名, 版本窗口)}
MEGATRON = {
    "swiglu": (
        "megatron.core.fusions.fused_bias_swiglu", "swiglu",
        "megatron-core>=0.9（0.18.0 实测）",
    ),
    "bias_swiglu": (
        "megatron.core.fusions.fused_bias_swiglu", "bias_swiglu_impl",
        "megatron-core>=0.9（0.18.0 实测）",
    ),
    "rope_bshd": (
        "megatron.core.models.common.embeddings.rope_utils",
        "_apply_rotary_pos_emb_bshd",
        "megatron-core>=0.9（0.18.0 实测；私有符号，上游无兼容承诺）",
    ),
}

_FRAMEWORKS = {"megatron": MEGATRON}


def _lookup(framework, semantic):
    table = _FRAMEWORKS.get(framework)
    if table is None:
        raise KeyError(f"未知目标框架: {framework!r}（已登记: {sorted(_FRAMEWORKS)}）")
    item = table.get(semantic)
    if item is None:
        raise KeyError(f"{framework} 下未登记语义 {semantic!r}"
                       f"（已登记: {sorted(table)}）")
    return item


def target(framework, semantic):
    """取绑定坐标 (module, attr)。未登记显式报错——不静默返回 None。"""
    return _lookup(framework, semantic)[:2]


def window(framework, semantic):
    """取版本窗口字符串（登记时的实测依据写在串里）。"""
    return _lookup(framework, semantic)[2]


def bind_points(framework, *semantics):
    """生成 `Provider.bind_points` 需要的元组（语义名与表 key 同名）。"""
    return tuple(target(framework, s) for s in semantics)


def summary():
    """按框架导出全部坐标（供评审 diff / provenance session 快照）。"""
    return {fw: {sem: {"module": m, "attr": a, "version_window": w}
                 for sem, (m, a, w) in table.items()}
            for fw, table in _FRAMEWORKS.items()}
