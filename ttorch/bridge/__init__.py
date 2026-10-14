"""TT Bridge：cuda_compat（Bridge 两大块之一，CUDA 接口兼容层）。

职责：框架 CUDA 调用点的逐符号决策与绑定——remap / fallback / exception，
含准入测试与绑定去向 trace。

注意命名：docs 里的 "Bridge" 是**大概念**（含本包 cuda_compat + 平行的
ttorch.platform Platform Manager + 顶层 checkpoint.py）。本包只承载其中的
CUDA 接口兼容这一块。

模块划分：
- bootstrap.py       启动编排：激活 Platform + 绑定 cuda-compat + 扫尾验收，
                     是 cuda-compat 使能的主控（依赖 platform，正向）
- binding_registry.py  policy 注册表：Entry 数据结构 + register_binding + YAML 快照
- module_binding.py  真实 torch.cuda 的 policy 条目（14 符号）
- cuda_compat.py     绑定机制：setattr + from-import 扫尾 + default-deny + 扫尾验收
- admission.py       准入测试：验绑定正确性（cuda-compat 侧，与 platform 侧
                     conformance 分层）

统一 runtime 接口不在此——它是 Platform Manager 的顺带产物，见
ttorch.platform（docs §0：契约由 PlatformHandle 定义，分发门面并入 manager）。
"""
from ttorch.bridge import admission, cuda_compat, binding_registry  # noqa: F401
from ttorch.bridge import bootstrap  # noqa: F401
from ttorch.bridge.binding_registry import (  # noqa: F401
    Decision, Entry, register_binding)

# module_binding 不在此 eager import：它引用真实 torch.cuda 符号，
# 须在 platform 激活之后才加载（bootstrap 延迟 import）。
