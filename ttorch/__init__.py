"""TTorch 对外入口：只做接口的转发 / 封装 / 暴露，不含功能性实现。

    import ttorch; ttorch.activate()   # 必须在 import 任何训练框架之前

为什么需要显式入口（TT_Bridge_Deep_Dive.md §1.2 的落地形态）：
Bridge 必须在 framework import / model build **之前**激活，否则框架的
`from torch.cuda import X` 会持有绑定前的陈旧引用。扫尾机制只能补救已
import 的模块，顺序本身无法补救——激活时机是硬约束，不是建议。

职责划分（本文件只做转发，实现各归其位）：
- 激活编排 + 激活状态：ttorch.bridge.bootstrap（cuda-compat 使能的主控）
- 平台注册/检测/门禁/生命周期：ttorch.platform.manager
- 平台的 torch 接口暴露（policy 绑定目标）：ttorch.platform.interface
- 绑定状态确认 + 运行期中转统计：ttorch.diagnostics（观测是独立工程资产）
- checkpoint：ttorch.checkpoint
- 打包/自动激活：ttorch._auto_init（.pth 钩子）

依赖方向（单向）：bootstrap / cuda_compat → platform；platform 不认识 bridge。
"""
from ttorch.bridge import bootstrap as _bootstrap
from ttorch.platform.platform_handle import TTPlatformError  # noqa: F401  # 再导出

__all__ = ["activate", "active", "is_active", "device_spec", "finalize",
           "check_bindings", "start_call_trace", "call_trace_report",
           "TTPlatformError", "FRAMEWORK_PACKAGES"]


def activate(platform="auto", sweep_packages=None):
    """激活 TT Bridge（platform 检测 + 门禁 + cuda-compat 绑定 + 扫尾验收）。

    实现见 bridge.bootstrap.activate()。幂等：重复调用返回同一结果。
    """
    return _bootstrap.activate(platform=platform, sweep_packages=sweep_packages)


def active():
    """当前激活结果；未激活则报错（不隐式激活——时序必须由调用方掌控）。"""
    act = _bootstrap.activation()
    if act is None:
        raise RuntimeError("TT 未激活：请在脚本最开头 `import ttorch; ttorch.activate()`")
    return act


def is_active():
    """Bridge 是否已激活。激活状态的唯一来源在 bridge.bootstrap。"""
    return _bootstrap.activation() is not None


def device_spec(idx=None):
    """快捷方式：框架侧构造 torch.device 用。"""
    return active().device_spec(idx)


def finalize():
    """进程退出前收尾（Bridge 激活态 + 平台生命周期）。绑定不回滚。"""
    _bootstrap.finalize()


# --- 观测：转发到 ttorch.diagnostics，本体不在此 ---

def check_bindings():
    """绑定正确性自检（admission 全套）。返回 (n_fail, rows)。"""
    from ttorch import diagnostics
    return diagnostics.check_bindings(active().entries)


def start_call_trace():
    """给门面装计数探针，统计运行期实际经 TT 中转的调用。"""
    from ttorch import diagnostics
    return diagnostics.start_call_trace(active().entries)


def call_trace_report(min_calls=1, exclude_prefixes=None):
    """中转证据报告。exclude_prefixes 默认排除本项目自带的 smoke/验收代码
    （它们对门面的调用不算"框架侧"证据）。"""
    from ttorch import diagnostics
    if exclude_prefixes is None:
        exclude_prefixes = ("real_qwen_smoke", "real_sft_smoke",
                            "checkpoint", "framework_real")
    return diagnostics.call_trace_report(min_calls, exclude_prefixes)


# 扫尾白名单再导出（本体在 bridge.bootstrap，激活编排的默认配置）
FRAMEWORK_PACKAGES = _bootstrap.FRAMEWORK_PACKAGES
