"""PlatformSpec：平台自述的数据结构（元数据 + detect + 实现类引用）。

spec 是平台的**唯一身份真相**：name/vendor_name/device_name/masquerades_cuda
等元数据由它承载，adapter 实例的身份一律从注入的 spec 读（实现类不自报，
防冒充）。

**不再集中登记平台实例**：每个平台一份 spec，与 detect 同处一个 import 安全的
注册单元（adapters/<平台>.py）；集中登记改为 adapters/__init__.py 的自动发现
+ 注册。本文件只定义 PlatformSpec 这个类型。

**延迟加载**：注册期只登记 `impl` 字符串，实现类到 `load()`（激活时）才真正
import。这样实现模块（adapters/<平台>_adapter.py）可以自由 import 厂商包——
副作用被推迟到"确认要激活这个平台"之后，而不必要求实现模块顶层保持纯净。
"""
import importlib
from dataclasses import dataclass
from typing import Callable, Optional

from ttorch.platform.platform_handle import TTPlatformError

__all__ = ["PlatformSpec"]


@dataclass(frozen=True)
class PlatformSpec:
    """平台自述：注册期可得的全部轻量信息 + 实现类引用。

    - name/vendor_name/device_name/masquerades_cuda/auto_detect：身份与检测元数据；
    - detect：存在性探测（只看不初始化），() -> (present, evidence)，零厂商 import；
    - impl：实现类路径 "module:Class"（真实平台，load() 才惰性 import）；
    - base_cls：已 import 的实现类（mock/测试夹具，直接实例化）。

    import 安全是硬约定：spec 与 detect 会在注册/检测期被读取（决定激活谁），
    因此承载它们的注册单元顶层不许 import 厂商包；厂商访问经 find_spec（查找
    不导入）、torch 自带 API，或在实现模块里（那个模块是惰性加载的）。
    """

    name: str
    vendor_name: str          # 厂商身份：provider 分发 / 门禁 / provenance
    device_name: str          # torch 视角的设备类型：框架 device.type 比较的存活线
    detect: Callable          # () -> (present, evidence)，零厂商 import
    impl: str = ""            # 实现类路径 "module:Class"（惰性 import，load 才加载）
    base_cls: Optional[type] = None  # 已 import 的实现类（mock/测试夹具）
    # 是否伪装成 cuda：伪装者会让 torch.cuda.is_available() 报 True，污染真
    # NVIDIA 的探测；为 True 时冲突判定把 "cuda" 命中降级为"伪装证据"。
    # bool 而非目标集合：masquerade 的现实就是伪装成 cuda（cuda 是生态唯一被
    # 框架硬编码的设备类型）；若未来出现伪装非 cuda / 多目标的栈再改为集合。
    masquerades_cuda: bool = False
    auto_detect: bool = True

    def load(self):
        """惰性加载并实例化本平台的实现类——**延迟 import 的唯一落点**。

        归在 spec 而非 registry：`impl` / `base_cls` 是 spec 自己的字段，"我的
        实现在哪、怎么取出来"属于平台自述，注册表不该伸手读。

        带 impl 的 spec 在此才 import 实现模块：进程里只会激活一个平台，其余
        候选的厂商实现永远不被加载（厂商包 import 有副作用）。
        """
        if self.base_cls is not None:
            return self.base_cls()  # 已 import，直接实例化
        if not self.impl:
            raise TTPlatformError(
                "platform", f"{self.name}: 既无 impl 也无 base_cls，无法取得实现")
        mod_path, _, cls_name = self.impl.partition(":")
        print(f"[platform] lazy-import {self.name}: {self.impl}")
        try:
            module = importlib.import_module(mod_path)
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError) as e:
            raise TTPlatformError(
                "platform", f"{self.name}: 实现 '{self.impl}' 导入失败: {e}") from e
        return cls()
