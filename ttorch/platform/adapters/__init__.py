"""平台实现（适配面）。

- `catalog.py`           内置平台目录：detect + SPEC + import 即自动注册
                         （import 安全，只看不初始化）；
- `<平台>_adapter.py`    平台实现类：惰性加载（激活时 spec.load() 才 import），
                         可自由 import 厂商包；
- `_*.py`                共享辅助（_device_module / _rng），非平台。

本 __init__ 只导入 catalog——导入动作即触发全部内置平台的注册（提前到 import，
不等 activate；register 是纯元数据登记、无副作用）。
"""
from ttorch.platform.adapters import catalog  # noqa: F401  # import 即注册全部内置平台
