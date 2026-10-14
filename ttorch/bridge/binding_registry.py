"""binding 注册表：code-first，真实对象引用，显式注册。

命名：本模块原名 registry.py，改为 binding_registry.py——"注册"在 TT 里有
两处（platform 的 manager.register 注册平台、这里注册绑定策略），泛名 registry
会让人分不清是哪一层。绑定阶段的注册表名实相符地叫 binding_registry。

对比 YAML 载体的变化：
- symbol / test 是真实对象（函数引用），拼写在 import 期就炸，不需要事后校验器
- decision 是枚举，非法值在写代码时就无法通过
- policy 条目与绑定实现、准入测试同处一个文件（见 bindings.py），内聚
- YAML 降级为生成物：export_entries() 导出快照供评审 diff / provenance，
  永远与代码同步，不存在手改数据文件导致的漂移

显式 register_binding() 而非装饰器：数据流显式可见；"忘注册"由默认拒绝兜底。

存储封装（BindingRegistry）：注册表是进程级单例，内部条目列表对外**匿名**——
外部摸不到它，只能经公开接口访问：register_binding() 写入、get_entries()
只读副本、export_entries() 导出快照。不用裸模块级 list，是为了杜绝外部
绕过查重直接改表（registry.REGISTRY.clear() 这类破坏不变式的操作）。

公开接口一律动宾结构（动词开头）：register_binding / get_entries /
export_entries——避免与调用方同名函数撞名（如 module_binding.entries()），
也使"读/写/导出"的意图从名字即可见。
"""
import enum
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

__all__ = ["Decision", "Entry", "BindingRegistry",
           "register_binding", "get_entries", "export_entries"]


class Decision(enum.Enum):
    REMAP = "remap"          # 绑到 impl（interface 的转发函数）
    FALLBACK = "fallback"    # 退回 torch-native 等价路径
    EXCEPTION = "exception"  # 无等价且必需，明确报错


@dataclass(frozen=True)
class Entry:
    """一个 CUDA 符号的完整决策记录。引用均为真实对象，import 期即解析。

    构造参数字段（symbol..test）在 bindings.py 里显式给出；
    dotted / target 是**派生属性**，不是构造参数——只在报错信息、
    trace、YAML 快照里用于展示，不参与绑定逻辑本身。
    """
    symbol: Callable                 # 框架侧的原始符号（真实引用，拼错 import 期即炸）
    decision: Decision               # 处置策略：remap / fallback / exception
    intent: str                      # 为什么这样处置（没有 intent 的映射半年后没人懂）
    impl: Optional[Callable] = None        # 仅 remap 用：要绑过去的目标实现（真实引用）
    fallback_target: Optional[str] = None  # 仅 fallback 用：等价路径名（仅展示用途，不进 trace 的 target 时也可读）
    test: Optional[Callable] = None        # remap 必填：准入测试（真实函数引用，不许免检）
    bind_module: Optional[str] = None      # 绑定落点模块（框架可见路径）。
        # 默认 None=符号的定义模块（symbol.__module__）。真实 torch.cuda 的符号
        # 定义在 torch.cuda.memory/streams/random 等子模块再 re-export，
        # 框架访问的是 torch.cuda.X——必须显式 bind_module="torch.cuda"，
        # 否则只改了子模块，框架侧属性仍指原实现（mock 单模块察觉不到此坑）

    def __post_init__(self):
        # 形态一致性检查（对象引用已由 import 期保证存在，这里只查组合逻辑）
        if self.decision is Decision.REMAP:
            assert self.impl is not None, f"{self.dotted}: remap 必须给 impl"
            assert self.test is not None, f"{self.dotted}: remap 必须有准入测试（不许免检）"
        elif self.decision is Decision.FALLBACK:
            assert self.fallback_target, f"{self.dotted}: fallback 缺等价路径名"
        else:
            assert self.impl is None, f"{self.dotted}: exception 不应有实现"

    @property
    def dotted(self) -> str:
        """符号的点分全名（绑定落点路径，如 torch.cuda.synchronize）。

        派生自 bind_module（缺省=定义模块）+ symbol 名，用于：__post_init__
        报错信息、cuda_compat 的 trace/coverage_report、YAML 快照的 symbol
        列、default-deny 覆盖面的 key。
        """
        return f"{self.bind_module or self.symbol.__module__}.{self.symbol.__name__}"

    @property
    def target(self) -> Optional[str]:
        """决策去向的展示名：remap→impl 函数名；fallback→等价路径名。

        纯展示用途（trace / YAML 快照的 target 列），绑定逻辑不读它——
        真正被绑的是 impl 对象。
        """
        if self.decision is Decision.REMAP:
            return self.impl.__name__
        return self.fallback_target


class BindingRegistry:
    """绑定策略注册表：封装存储与访问规则。

    存储是实例私有（self._entries），注册表本体是模块级私有单例
    `_REGISTRY`——外部拿不到可变列表，只能经下方模块级公开接口访问。
    规则（查重、只读副本、按落点过滤）集中在 class 内，未来按
    platform × 框架版本二维索引（policy 多平台 overlay 遗留项）时在此扩展。
    """

    def __init__(self):
        self._entries = []

    def register_binding(self, entry: Entry) -> Entry:
        """显式注册一条绑定策略。重复符号立即报错。"""
        assert all(e.symbol is not entry.symbol for e in self._entries), \
            f"符号重复注册: {entry.dotted}"
        self._entries.append(entry)
        return entry

    def get_entries(self, bind_module=None) -> Tuple[Entry, ...]:
        """只读副本（tuple，外部改不动内部表）。

        bind_module：只返回绑定到该模块的条目。注册表是**进程级全局**，
        一个进程里可能同时注册了多套 policy（mock 的 vendor_cuda 与真实的
        torch.cuda）——按落点过滤是分离它们的手段。None 表示全部。
        """
        if bind_module is None:
            return tuple(self._entries)
        return tuple(e for e in self._entries
                     if (e.bind_module or e.symbol.__module__) == bind_module)

    def export_entries(self, path, bind_module=None):
        """导出快照：评审 diff / provenance 用。生成物，永不手改。

        bind_module 过滤的意义：快照要满足"CI diff 必须一致"，混入不相关
        policy 的条目会让 diff 失真（实测：demo 导出的快照曾混进 14 条
        torch.cuda 条目）。None 表示导出全部（单一 policy 的场景）。
        """
        import yaml
        rows = [dict(symbol=e.dotted, decision=e.decision.value,
                     target=e.target, intent=e.intent,
                     test=e.test.__name__ if e.test else None)
                for e in self.get_entries(bind_module)]
        with open(path, "w") as f:
            yaml.safe_dump({"version": 1, "symbols": rows}, f,
                           allow_unicode=True, sort_keys=False)


_REGISTRY = BindingRegistry()  # 进程级单例（模块级私有，外部摸不到内部表）


def register_binding(entry: Entry) -> Entry:
    """显式注册一条绑定策略。"""
    return _REGISTRY.register_binding(entry)


def get_entries(bind_module=None) -> Tuple[Entry, ...]:
    """全部（或按落点过滤的）已注册条目，只读副本。"""
    return _REGISTRY.get_entries(bind_module)


def export_entries(path, bind_module=None):
    """导出快照：评审 diff / provenance 用。生成物，永不手改。"""
    return _REGISTRY.export_entries(path, bind_module)
