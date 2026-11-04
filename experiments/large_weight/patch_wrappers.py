"""large-weight 补丁的两个 wrapper（**已从 TT 撤出，仅存档备查**）。

原位置：`ttorch/bridge/patches.py`（B2 / B3 两条注册）。
撤出决策与理由见 `experiments/large_weight/README.md` 与
`docs/TT_Bridge_Deep_Dive.md` §9.5。

本文件**不参与 ttorch 包导入**，也不被 `register_all()` 注册。保留原因：
若后续的 large-weight 独立方案仍需在 Python 层做 per-expert 视图适配，
这两个 wrapper 与配套的 17 例回归可直接取用，不必重写。

与原版唯一的差别：`large_weight_shim` 改为同目录相对导入
（原为 `from ttorch.bridge import large_weight_shim as lw`）。
"""
import functools

from . import large_weight_shim as lw

__all__ = ["set_mlp_state_wrapper", "lora_merge_wrapper"]


def set_mlp_state_wrapper(orig):
    """B2：让 mcore_bridge 的 HF↔Megatron MoE 转换能寻址 large-weight 布局。

    **缺陷**：``GPTBridge._set_mlp_state`` 及其下游按
    ``getattr(layer, f'weight{i}')`` 读专家权重（``gpt_bridge.py`` 12 处），
    而 ``XTE_GROUPED_GEMM_LARGE_WEIGHT=1`` 时该名字不注册（只有 3D
    ``large_weight``）→ AttributeError。生产 ``env.sh:31`` 正是 =1。

    **处置**：调用期把专家切片设为真实属性，调完还原。只在 large-weight
    布局下接管（``has_large_weight``），LW=0 时零副作用。

    转换是**读**路径为主，但 to_mcore=True 方向会写入——故同样做 write_back，
    与 B3 共用一套 shim（写回是幂等的：没有重绑就什么都不做）。
    """
    @functools.wraps(orig)
    def wrapper(self, mg_mlp, *args, **kwargs):
        """mcore_bridge GPTBridge._set_mlp_state + large-weight 视图。"""
        patched = []
        for layer in lw.iter_expert_layers(mg_mlp):
            if not lw.has_large_weight(layer):
                continue
            n = lw.num_experts_of(layer)
            if not n:
                continue   # 专家数口径不可信（见 num_experts_of）：不接管
            items, names = lw.expose_per_expert(layer, n)
            if items or names:
                patched.append((layer, n, items, names))
        try:
            return orig(self, mg_mlp, *args, **kwargs)
        finally:
            for layer, n, items, names in reversed(patched):
                lw.write_back(layer, n)
                lw.restore_layer(layer, items, names)

    return wrapper


def lora_merge_wrapper(orig):
    """B3：让 LoRA merge/unmerge 的结果真正落回 large_weight 存储。

    **缺陷（已实测，误差 1）**：``LoraParallelLinear.merge`` 在 ``safe_merge``
    下执行 ``weight.data = orig_weights[i]``（``tuners/lora.py:540``）——
    这是**重新绑定**属性对象，不是透写回存储。large-weight 布局下该属性是
    临时视图，重绑后合并结果**静默丢失**（权重损坏，无任何报错）。

    口径更正（2026-11-04 复核源码）：**只有 merge 的 safe_merge 分支**是重绑；
    ``lora.py:546``（非 safe_merge）与 ``:577``（unmerge）都是**原地运算**
    （``orig_weight.data += / -= delta``），真 view 可透写。另：裸环境下
    ``lora.py:518`` 的 ``base_layer.weight0.device`` 会先抛 AttributeError，
    **根本走不到 :540** —— 丢写是厂商 ``__getattr__`` shim 把 weight0 变可读
    之后才浮现的第二层问题。

    XME 的 ``__getattr__`` 合成视图只解决读，解决不了这个写路径
    （``xmegatron_ext/plugin.py:113``）。

    **处置**：真实属性遮蔽 → 调原实现 → ``write_back`` 把重绑值 copy_ 回
    large_weight 切片 → 还原。写回按指针比对跳过原地写路径。
    """
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        """mcore_bridge LoraParallelLinear.merge/unmerge + 写回 large_weight。"""
        try:
            base = self.get_base_layer()
        except Exception:
            return orig(self, *args, **kwargs)
        if not lw.has_large_weight(base):
            return orig(self, *args, **kwargs)
        n = lw.num_experts_of(base)
        if not n:
            return orig(self, *args, **kwargs)
        items, names = lw.expose_per_expert(base, n)
        try:
            return orig(self, *args, **kwargs)
        finally:
            lw.write_back(base, n)
            lw.restore_layer(base, items, names)

    return wrapper
