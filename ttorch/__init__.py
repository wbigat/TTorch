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
- 模块级性能实现（provider）：ttorch.provider（自动注册/生效/选择）
- 观测/验证/溯源（witness）：ttorch.witness（执行数据采集 + bridge/provider 取证）
- checkpoint（save/resume + schema 校验 + RNG）：ttorch.bridge.checkpoint
- 打包/自动激活：ttorch._auto_init（.pth 钩子）

依赖方向（单向）：bootstrap / cuda_compat → platform；provider → platform；
platform 不认识 bridge / provider。
"""
from ttorch.bridge import bootstrap as _bootstrap
from ttorch.platform.platform_handle import TTPlatformError  # noqa: F401  # 再导出

__all__ = ["activate", "active", "is_active", "device_spec", "finalize",
           "provider_wrap", "check_bindings", "assert_bindings_intact",
           "start_call_trace", "call_trace_report", "TTPlatformError",
           "FRAMEWORK_PACKAGES"]


def activate(platform="auto", sweep_packages=None):
    """激活 TT（platform 检测 + cuda-compat 绑定 + provider 自动注册）。

    Bridge 实现见 bridge.bootstrap.activate()；随后 provider 层自动注册全部
    已知 provider 并接线平台探测（见 provider.manager.activate()）。幂等。
    """
    act = _bootstrap.activate(platform=platform, sweep_packages=sweep_packages)
    # provider 自动注册 + 接线（平台已激活）。惰性 import 保持 import ttorch 轻。
    from ttorch.provider import manager as _provider_mgr
    _provider_mgr.activate()
    # 通道 B：框架性能符号的后置绑定——清单已由 provider.activate() 注册，
    # 此时装 import 钩子（框架在本进程大多尚未 import）。机制在 provider 层
    # （性能替换的执行归 provider，bridge 只管功能域的 torch.cuda 收束）。
    from ttorch.provider import fn_binding
    fn_binding.install()
    return act


def provider_wrap(model):
    """自动发现并 patch 模型里匹配已注册 provider 的模块（自动生效）。

    在模型建好（from_pretrained / 搬到设备）之后调用一次：模型里的
    RMSNorm/RoPE 等模块被路由到对应 provider，首次前向时按平台自动选实现。
    返回 patch 的模块数。
    """
    from ttorch.provider import manager as _provider_mgr
    return _provider_mgr.wrap_model(model)


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


# --- 观测/验证：转发到 ttorch.witness，本体不在此 ---

def check_bindings():
    """绑定正确性自检（admission 全套）。返回 (n_fail, rows)。"""
    from ttorch import witness
    return witness.check_bindings(active().entries)


def assert_bindings_intact():
    """运行期绑定完整性校验（篡改检测）：门面是否仍是绑定完成那一刻的对象。

    import 触发的符号替换（symbrewrite 钩子）可能在 TT 激活之后才覆盖门面，
    verify_sweep 只在激活末尾跑一次——本函数可在训练启动后/step 边界随时复检。
    返回 (n_tampered, rows)。
    """
    from ttorch import witness
    from ttorch.bridge import cuda_compat
    return witness.assert_bindings_intact(cuda_compat.post_bind_snapshot())


def start_call_trace():
    """给门面装计数探针，统计运行期实际经 TT 中转的调用。"""
    from ttorch import witness
    return witness.start_call_trace(active().entries)


def call_trace_report(min_calls=1, exclude_prefixes=None):
    """中转证据报告。exclude_prefixes 默认排除本项目自带的 smoke/验收代码
    （它们对门面的调用不算"框架侧"证据）。"""
    from ttorch import witness
    if exclude_prefixes is None:
        exclude_prefixes = ("real_qwen_smoke", "real_sft_smoke",
                            "checkpoint", "framework_real")
    return witness.call_trace_report(min_calls, exclude_prefixes)


# 扫尾白名单再导出（本体在 bridge.bootstrap，激活编排的默认配置）
FRAMEWORK_PACKAGES = _bootstrap.FRAMEWORK_PACKAGES
