"""ttorch.utils：跨层共享的内部工具（不属于任何一层的对外契约）。

- `_rebind`：符号替换动作的唯一实现（bridge 与 provider 两个 writer 共用）
  ——放 utils 是因为它同时被 bridge 与 provider 依赖，放任何一层都会让
  另一层反向依赖该层。
- `_auto_init`：`.pth` 自动激活钩子（进程启动期执行，先于任何层被 import）。
"""
