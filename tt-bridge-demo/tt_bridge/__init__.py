"""TT Bridge 垂直切片：cuda_compat + Platform Manager 两块。

cuda_compat：框架 CUDA 调用点的逐符号绑定机制（registry/bindings/
admission 配套）。

Platform Manager（platform_mgr.py，管理面 + 分发门面）与 adapter
分文件：契约在 platform_adapter.py，各硬件实现在 adapters.py。
统一 runtime 接口不是独立模块：契约由 PlatformAdapter 定义，分发门面
是激活之后顺带的产物（platform_mgr 模块级函数）。
"""
