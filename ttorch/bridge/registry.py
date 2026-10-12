"""policy 注册表：code-first，真实对象引用，显式注册。

对比 YAML 载体的变化：
- symbol / test 是真实对象（函数引用），拼写在 import 期就炸，不需要事后校验器
- category / decision 是枚举，非法值在写代码时就无法通过
- policy 条目与绑定实现、准入测试同处一个文件（见 bindings.py），内聚
- YAML 降级为生成物：export_yaml() 导出快照供评审 diff / provenance，
  永远与代码同步，不存在手改数据文件导致的漂移

显式 binding_register() 而非装饰器：数据流显式可见；"忘注册"由默认拒绝兜底。
（命名区别于 platform_mgr.register：那里注册的是平台，这里注册的是绑定策略）
"""
import enum
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


class Category(enum.Enum):
    BEHAVIOR = "behavior"          # 有副作用：验语义等价
    STATE_QUERY = "state_query"    # 动态只读：验口径一致 + 禁缓存
    CAPABILITY = "capability"      # 静态事实：验声明 vs 实测；不进门面


class Decision(enum.Enum):
    REMAP = "remap"          # 绑到 impl（行为/状态查询）或 capability 缓存（能力）
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
    category: Category               # 接口类型：行为 / 状态查询 / 能力（决定绑定与验证形态）
    decision: Decision               # 处置策略：remap / fallback / exception
    intent: str                      # 为什么这样处置（没有 intent 的映射半年后没人懂）
    impl: Optional[Callable] = None        # 仅 remap+行为/状态查询用：要绑过去的目标实现（真实引用）
    cap_key: Optional[str] = None          # 仅 remap+能力类用：capability 缓存键（probe 期填充，选择期消费）
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
            if self.category is Category.CAPABILITY:
                assert self.cap_key and self.impl is None, \
                    f"{self.dotted}: capability 类只能给 cap_key，不许给 impl"
            else:
                assert self.impl is not None and self.cap_key is None, \
                    f"{self.dotted}: 行为/状态查询 remap 必须给 impl"
            assert self.test is not None, f"{self.dotted}: remap 必须有准入测试（不许免检）"
        elif self.decision is Decision.FALLBACK:
            assert self.fallback_target, f"{self.dotted}: fallback 缺等价路径名"
        else:
            assert self.impl is None and self.cap_key is None, \
                f"{self.dotted}: exception 不应有实现"

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
        """决策去向的展示名：remap→impl 函数名或 cap:<key>；fallback→等价路径名。

        纯展示用途（trace / YAML 快照的 target 列），绑定逻辑不读它——
        真正被绑的是 impl 对象或 capability 闭包。
        """
        if self.decision is Decision.REMAP:
            return f"cap:{self.cap_key}" if self.cap_key else self.impl.__name__
        return self.fallback_target


REGISTRY: list = []


def binding_register(entry: Entry) -> Entry:
    """显式注册一条绑定策略。重复符号立即报错。"""
    assert all(e.symbol is not entry.symbol for e in REGISTRY), \
        f"符号重复注册: {entry.dotted}"
    REGISTRY.append(entry)
    return entry


def entries() -> Tuple[Entry, ...]:
    return tuple(REGISTRY)


def export_yaml(path, bind_module=None):
    """导出快照：评审 diff / provenance 用。生成物，永不手改。

    bind_module：只导出绑定到该模块的条目。REGISTRY 是**进程级全局**，
    一个进程里可能同时注册了多套 policy（mock 的 vendor_cuda 与真实的
    torch.cuda）——不过滤会把不相关的条目写进快照，导致"CI diff 必须
    一致"的约定失效（实测：demo 导出的快照混进了 14 条 torch.cuda 条目）。
    None 表示导出全部（单一 policy 的场景）。
    """
    import yaml
    rows = [dict(symbol=e.dotted, category=e.category.value, decision=e.decision.value,
                 target=e.target, intent=e.intent,
                 test=e.test.__name__ if e.test else None)
            for e in REGISTRY
            if bind_module is None or (e.bind_module or e.symbol.__module__) == bind_module]
    with open(path, "w") as f:
        yaml.safe_dump({"version": 1, "symbols": rows}, f,
                       allow_unicode=True, sort_keys=False)
