"""manager：平台管理面——自动检测 / 激活门禁 / 生命周期 / 能力查询。

本模块是**管理面的对外入口**：既有 `PlatformManager` 类（可独立实例化，便于
测试隔离），也持有一个进程级单例并把管理动作暴露成模块函数
（`manager.auto_activate()` 等）。

三层分工（2026-10-13 按功能拆分，此前 manager.py 与 platform_mgr.py 定位
和命名都重复，已合并/改名）：
- registry.py  只管**注册**：登记哪些 spec（不持有激活态）；
- 本模块       管**运行态**：决定激活谁（detect）、激活流水线与门禁、当前
               激活的是谁（activation）、生命周期收尾；
- interface.py    管**torch 接口暴露**：把当前平台的 torch 接口原样转发出去
               （policy 表的绑定目标，无状态）。

平台呈现（torch 接口 / 平台自述 / spec 身份 / 能力表 / 设备标识）一律由
PlatformHandle 对外承担——manager 是管理者，不重新暴露平台接口；要平台
信息先 `activation()` 拿到 handle 再问它。

激活流水线（任一不过即拒绝激活，不切换 active）：
    PlatformHandle(spec.load(), spec)   # 装配：惰性导入实现 + 关联 spec
      -> SDK 版本协商
      -> initialize()
      -> capability schema 校验（identity 由本类从 spec 注入）
      -> conformance 门禁
"""
from ttorch.platform.platform_handle import PlatformHandle, TTPlatformError
from ttorch.platform.registry import PlatformRegistry

__all__ = [
    # 类（可独立实例化）
    "PlatformManager",
    # 单例上的管理面模块函数
    "register", "unregister", "auto_activate", "select", "activation",
    "is_activated", "finalize",
    "registry", "instance",
    # 再导出
    "TTPlatformError",
]


def _ver_tuple(s):
    return tuple(int(x) for x in s.split("."))


class PlatformManager:
    """平台运行态：激活哪个、当前是谁、能力如何。注册委托给 registry。"""

    def __init__(self, registry=None):
        self.registry = registry if registry is not None else PlatformRegistry()
        self._activation = None  # 当前激活的 PlatformHandle（名词：这次激活）

    # ------------------------------------------------------------------
    # 注册（转发 registry，让调用方只认一个入口）
    # ------------------------------------------------------------------
    def register(self, spec):
        return self.registry.register(spec)

    def unregister(self, name):
        return self.registry.unregister(name)

    # ------------------------------------------------------------------
    # 激活
    # ------------------------------------------------------------------
    def auto_activate(self):
        """正常路径：自动分析当前环境，激活唯一匹配的 platform。

        对每个 auto_detect 的已注册 spec 跑 detect() 存在性探测，证据逐条
        打印（provenance）。恰好一个命中 -> 激活；零命中或多命中都 fail
        fast——多命中说明进程级单 platform 的假设被破坏，宁可报错也不猜测，
        此时可用 select(name) 显式 override 排查。

        masquerade 降级：命中者声明 masquerades_cuda 时，"cuda" 的命中
        降级为"冒充证据"（打印，不参与冲突判定）——冒充 CUDA 的栈会让
        torch.cuda.is_available() 报 True，没有这层会把冒充者当真 NVIDIA。
        """
        hits = {}  # name -> evidence
        for spec in self.registry.auto_detectable():
            ok, evidence = spec.detect()  # 只看不初始化
            print(f"[platform] detect {spec.name}: "
                  f"{'present' if ok else 'absent'} ({evidence})")
            if ok:
                hits[spec.name] = evidence
        demoted = set()
        for name in hits:
            if (self.registry.get(name).masquerades_cuda
                    and "cuda" in hits and "cuda" not in demoted):
                demoted.add("cuda")
                print(f"[platform] detect cuda: 命中被 {name} 的 native 探测降级"
                      f"（cuda 是 {name} 伪装的身份，非独立硬件）")
        matched = [n for n in hits if n not in demoted]
        if len(matched) == 1:
            return self._activate(matched[0], how="auto-detect")
        if not matched:
            raise RuntimeError(
                f"未检测到任何已注册 platform，已探测: {self.registry.names()}")
        raise RuntimeError(
            f"检测到多个 platform: {matched}——进程级单 platform，拒绝猜测；"
            f"调试可用 select(name) 显式 override")

    def select(self, name):
        """显式 override：仅调试/测试逃生通道，正常路径走 auto_activate()。"""
        return self._activate(name, how="manual-override")

    def _activate(self, name, how):
        """激活流水线：装配 -> 版本协商 -> initialize -> caps 校验 -> conformance 门禁。"""
        spec = self.registry.get(name)
        # 装配：spec.load() 取实现（惰性导入的唯一落点），与 spec 关联成 handle
        candidate = PlatformHandle(spec.load(), spec)

        lo, hi = _ver_tuple(candidate.sdk_range[0]), _ver_tuple(candidate.sdk_range[1])
        sdk = _ver_tuple(candidate.sdk_version())
        if not lo <= sdk < hi:
            raise TTPlatformError(
                "sdk_version",
                f"{name} 实现支持 SDK [{candidate.sdk_range[0]}, {candidate.sdk_range[1]})，"
                f"检测到 {candidate.sdk_version()}")

        candidate.initialize()

        from ttorch.platform import conformance  # 延迟 import 避免环
        results = conformance.run(candidate)
        failed = [r for r in results if not r.ok]
        for r in results:
            print(f"[conformance] {'PASS' if r.ok else 'FAIL'} {r.name}: {r.detail}")
        if failed:
            candidate.finalize()
            raise TTPlatformError(
                "conformance",
                f"{name} 未通过 conformance 门禁，拒绝激活: {[r.name for r in failed]}")

        old = self._activation
        self._activation = candidate
        if old is not None:
            old.finalize()
        print(f"[platform] activated={name} via={how} "
              f"vendor={candidate.vendor_name} device_name={candidate.device_name}")
        return self._activation

    # ------------------------------------------------------------------
    # 当前状态 / 生命周期
    # ------------------------------------------------------------------
    def activation(self):
        """当前激活的 PlatformHandle（名词：这次激活）；未激活则报错。"""
        if self._activation is None:
            raise RuntimeError("platform 未激活：请先 auto_activate()")
        return self._activation

    def is_activated(self):
        return self._activation is not None

    def finalize(self):
        """进程退出前的生命周期收尾。"""
        if self._activation is not None:
            self._activation.finalize()
            print(f"[platform] finalized={self._activation.name}")
        self._activation = None

# =====================================================================
# 进程级单例 + 管理面模块函数
#
# 为什么要单例：平台激活是进程级事实（一个进程只跑一块卡），且 cuda-compat
# 的绑定是全局的——门面必须都指向同一份激活态，复制一份会让状态分裂。
# 类本身保持可独立实例化，便于测试隔离。
# =====================================================================

_instance = PlatformManager()


def instance():
    """进程级 PlatformManager 单例（需要直接操作管理器时用）。"""
    return _instance


def set_instance(m):
    """换掉进程级单例（测试/mock 隔离用）。

    内置平台在 import ttorch.platform 时自动注册进了默认单例；mock/demo 要
    用一套完全不同的假平台，就换一个干净的实例，避免与生产单例里的真实平台
    冲突。interface/admission 经模块函数 `activation()` 读取，换单例后它们随之
    指向新实例。
    """
    global _instance
    if not isinstance(m, PlatformManager):
        raise TTPlatformError("platform", f"set_instance 需要 PlatformManager，实际 {type(m).__name__}")
    _instance = m
    return m


def registry():
    """单例持有的 PlatformRegistry（注册表本体）。"""
    return _instance.registry


def register(spec):
    """登记一个平台（PlatformSpec）：只存元数据，不初始化、不导入实现。"""
    return _instance.register(spec)


def unregister(name):
    return _instance.unregister(name)


def auto_activate():
    """自动检测并激活唯一匹配的 platform（含 conformance 门禁）。"""
    return _instance.auto_activate()


def select(name):
    """显式指定激活哪个 platform（调试/测试逃生通道）。"""
    return _instance.select(name)


def activation():
    """当前激活的 PlatformHandle；未激活则报错。"""
    return _instance.activation()


def is_activated():
    return _instance.is_activated()


def finalize():
    """进程退出前的生命周期收尾。"""
    return _instance.finalize()
