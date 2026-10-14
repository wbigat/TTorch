"""打包补充：把 ttorch_auto_init.pth 装进 site-packages 根。

元数据在 pyproject.toml，这里只处理 `.pth` 的安装位置——它无法用
纯声明式配置可靠表达：

- `package_data` 会落进包目录内（site-packages/ttorch/），site 不扫那里；
- `[tool.setuptools] data-files = { "." = [...] }` 的 key 是相对
  **sysconfig data 路径**（即 sys.prefix）的，实测落到
  `<env>/ttorch_auto_init.pth`，site 同样不扫（site 只扫 site-packages）；
- 写死 `lib/python3.12/site-packages` 可行但绑定 Python 小版本。

因此构建期动态计算 purelib 相对 data 的路径（通常是
`lib/pythonX.Y/site-packages`）。代价：产出的 wheel 与构建时的
Python 小版本绑定——这是 `.pth` 类包的固有性质，不是本项目的设计缺陷。
"""
import os
import sysconfig

from setuptools import setup

_purelib = sysconfig.get_path("purelib")
_data = sysconfig.get_path("data")
_rel_site_packages = os.path.relpath(_purelib, _data)

setup(data_files=[(_rel_site_packages, ["ttorch_auto_init.pth"])])
