"""框架建模入口的自动 wrap（auto-wrap）：环境变量门控的定点包装。

形态（2026-10-28 设计定案，替代此前的 Module._call_impl 全局 patch）：
- **定点**：只包装清单内框架的建模入口函数（如 mcore_bridge.get_mcore_model）
  ——建模返回后对每个模型调 `provider_wrap`，与手动路径**同一代码路径**，心智
  统一；不动 torch 本体：零热路径开销、无私有方法侵入、无 patch 叠加风险；
- **门控**：`TT_PROVIDER_AUTO_WRAP=1` 时由 `provider.activate()` 使能
  （opt-in，最小惊讶）；默认关——用户手动 `ttorch.provider_wrap(model)`；
- **可感知**：建模入口触发且 wrap 数 >0 时终端打印一行 + provenance 留痕；
- 绑定复用 fn_binding 通道（import hook 兜未来加载的框架模块，已加载的立即
  补绑；_rebind 扫尾修正 from-import 陈旧引用）；semantic="" 跳过性能符号的
  native 对账（建模函数无对账对象）。

覆盖盲区（如实记录）：只覆盖清单入口建的模型——建模后再加工替换模块的流程
不兜，纯 transformers 手搭路径也不经此通道（这些场景用手动 provider_wrap）。

接新框架：往 MODEL_BUILD_POINTS 加 (模块, 函数名)——函数须返回模型或模型
list/tuple（list 形态对应 megatron virtual pipeline 的 model chunks）。
"""
import functools
import sys

import torch

from ttorch.provider import fn_binding
from ttorch.witness import provenance

__all__ = ["enable", "disable", "is_enabled", "MODEL_BUILD_POINTS", "ENV_VAR"]

ENV_VAR = "TT_PROVIDER_AUTO_WRAP"

# 框架建模入口清单：(模块, 函数名)。
# mcore_bridge.get_mcore_model：swift-megatron 路径的建模源头
# （swift.megatron.model.utils.get_mcore_model 是它的薄封装；swift 侧的
# from-import 陈旧引用由 _rebind sweep 修正，白名单已含 mcore_bridge/swift）。
MODEL_BUILD_POINTS = (
    ("mcore_bridge", "get_mcore_model"),
)

_ENABLED = False


def _wrap_model_builder(orig, dotted):
    """建模函数包装器：建模后对返回的每个模型 provider_wrap（与手动同路径）。"""
    @functools.wraps(orig)
    def _wrapped(*args, **kw):
        models = orig(*args, **kw)
        from ttorch.provider import manager as _pm  # 延迟 import 避免环
        iterable = models if isinstance(models, (list, tuple)) else (models,)
        total = 0
        for model in iterable:
            if isinstance(model, torch.nn.Module):
                total += _pm.instance().wrap_model(model)
        if total:
            # 可感知：终端一行 + provenance（fn_binding 的 remap 事件另记绑定）
            print(f"[tt] provider auto-wrap: {dotted} -> {total} modules wrapped",
                  flush=True)
            provenance.record(provenance.LAYER_PROVIDER, "wrap", dotted,
                              "auto_wrapped_via_build_hook", count=total)
        return models
    return _wrapped


def enable():
    """使能建模入口自动 wrap（幂等）。activate() 按 TT_PROVIDER_AUTO_WRAP=1 调用；
    也可代码侧经 ttorch.provider_auto_wrap(True) 使能。"""
    global _ENABLED
    if _ENABLED:
        return
    _ENABLED = True
    bound, pending = [], []
    for mod, attr in MODEL_BUILD_POINTS:
        dotted = f"{mod}.{attr}"
        b = fn_binding.PerfBinding(
            dotted=dotted, module=mod, attr=attr, semantic="",
            factory=lambda orig, _d=dotted: _wrap_model_builder(orig, _d))
        try:
            fn_binding.register_binding(b)
        except AssertionError:
            bound.append(dotted)  # 已注册（重复 enable）——绑定仍在
            continue
        (bound if fn_binding.bind_loaded(b) else pending).append(dotted)
    # 失效可见（与 fn_binding._print_summary 同哲学）：绑没绑上必须出现在终端
    print(f"[tt] provider auto-wrap: ON（{ENV_VAR}=1；bound={bound}，"
          f"pending（目标框架未 import，等 hook）={pending}）", flush=True)


def disable():
    """摘除全部建模入口包装（恢复原始函数）。幂等。不影响 fn_binding 里其他
    性能符号绑定（按 dotted 选择性移除）。"""
    global _ENABLED
    if not _ENABLED:
        return
    _ENABLED = False
    for mod, attr in MODEL_BUILD_POINTS:
        fn_binding.unregister(f"{mod}.{attr}")


def is_enabled():
    return _ENABLED
