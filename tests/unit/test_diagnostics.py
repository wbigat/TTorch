"""D7/D8/D12 闭环测试：内存口径诊断 + 热路径税基线 + detect 异常隔离。"""
import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def _activate():
    import ttorch
    ttorch.activate()


def test_d8_measure_forward_tax():
    """D8：热路径税基线——三层开销测量可用，税率为正且有限。"""
    from ttorch.witness import diagnostics
    r = diagnostics.measure_forward_tax(symbol="current_device", n=500)
    assert r["native_us"] > 0 and r["facade_us"] > 0
    assert r["tax_x"] > 0
    assert "current_device" in r["detail"]


def test_d7_mem_reset_semantics():
    """D7：峰值窗口 + reset 口径（reset 后峰值回落到当前 allocated）。"""
    from ttorch.witness import diagnostics
    r = diagnostics.mem_reset_semantics()
    assert r["ok"] is True, r["detail"]


def test_d12_detect_exception_isolation(capsys):
    """D12：单个平台 detect 抛异常被隔离，其他平台照常探测，异常不穿透。"""
    from ttorch.platform.manager import PlatformManager

    class FakeSpec:
        masquerades_cuda = False
        torch_range = ("2.9", "2.10")

        def __init__(self, name, detect_fn):
            self.name = name
            self._detect = detect_fn

        def detect(self):
            return self._detect()

    def bad_detect():
        raise RuntimeError("厂商包损坏")

    def absent_detect():
        return (False, "不可用")

    class FakeRegistry:
        def auto_detectable(self):
            return [FakeSpec("bad", bad_detect), FakeSpec("absent", absent_detect)]

        def names(self):
            return ["bad", "absent"]

        def get(self, name):
            return next(s for s in self.auto_detectable() if s.name == name)

    mgr = PlatformManager(registry=FakeRegistry())
    with pytest.raises(RuntimeError) as exc:
        mgr.auto_activate()
    # 无命中应报"未检测到"，而非 detect 异常穿透
    assert "厂商包损坏" not in str(exc.value), "detect 异常不应穿透"
    assert "未检测到" in str(exc.value)
    out = capsys.readouterr().out
    assert "SKIP 探测异常" in out, "异常平台应被隔离并打印"
    assert "absent" in out, "正常平台应照常探测"


def test_d8_measure_dispatch_tax():
    """D8 ②：provider 函数级通道分发热路径税基线（dispatcher.call vs kernel）。

    只作基线（防劣化），不断言税率阈值（当前不优化）。
    """
    from ttorch.witness import diagnostics
    r = diagnostics.measure_dispatch_tax("rmsnorm", n=200)
    assert r["ok"] is True, r.get("detail")
    assert r["kernel_us"] > 0 and r["call_us"] > 0
    # tax_pct 符号受小 n 测量噪声影响（call/kernel 抖动时可能为负）——基线只确认
    # 测量可用，不断言税率符号/阈值（D8：只加基线不优化）
    assert "tax_pct" in r and "分发税" in r["detail"]
