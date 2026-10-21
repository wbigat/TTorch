"""torch.ops.tt.* 的 demo 对应物：Provider 实现对框架暴露的注册位。

框架只面向语义名编程（tt_ops.rmsnorm / tt_ops.rope_bshd），不感知
provider 名字、不知道当前选中谁、不知道是否发生回退——选择、门禁、
switch、回退全部封装在 Provider Manager 内。这正是定稿中
"torch.ops.tt.* 只是 Provider 注册位"的含义：op 命名空间是隔离缝，
不是功能载体。

模块级 __getattr__ 动态分发：注册新语义（reference provider）后，
框架侧无需任何改动即可按名调用。
"""
from tt_provider import provider_mgr


def __getattr__(semantic):
    if semantic.startswith("_"):
        raise AttributeError(semantic)

    def op(*args, **kw):
        return provider_mgr.call(semantic, *args, **kw)

    op.__name__ = f"tt.{semantic}"
    return op
