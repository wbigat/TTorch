"""MoE GroupedLinear 的 big_leaf 权重布局与训练生命周期适配。

所有实现均按需加载，避免把 TE/Megatron 依赖扩散到未启用该特性的 TTorch 进程。
"""

__all__ = [
    "BigLeafPlan", "BigLeafMegatronPlan", "apply_big_leaf",
    "convert_from_big_leaf", "convert_to_big_leaf",
]


def __getattr__(name):
    if name in ("BigLeafPlan", "apply_big_leaf"):
        from . import core
        return getattr(core, name)
    if name == "BigLeafMegatronPlan":
        from .megatron import BigLeafMegatronPlan
        return BigLeafMegatronPlan
    if name in ("convert_from_big_leaf", "convert_to_big_leaf"):
        from . import checkpoint
        return getattr(checkpoint, name)
    raise AttributeError(name)
