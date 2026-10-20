"""ttorch.witness：观测 / 验证 / 溯源层——TT 的独立工程资产。

定位（与 bridge / platform / provider 平行的第四层）：witness 不在数据通路上，
而是**旁观并作证**——开启后采集完整执行数据，并对 bridge 与 provider 的行为取证：

- provenance.py   统一事件流：bridge 绑定 + provider 门禁/选择/回退，合并成一份
                  "这次训练用了哪些替身和加速"的可复现报告（record/records/report/
                  summary/session/export_jsonl）
- bindings.py     bridge API 验证：激活期每个符号是否绑到预期门面（check_bindings）
- call_trace.py   bridge 运行期中转证据：训练时实际经 TT 中转的调用（start_call_trace/
                  call_trace_report）
- verify.py       provider 精度/性能验证原语：三级门禁的测量本体（verify_impl /
                  measure_throughput_us），provider.manager 的 gate 复用；
                  精度对账在 fp32 轨道（判数学等价，fwd+bwd 双验），性能测量用
                  生产 dtype——两轨道分离

依赖方向（单向、无环）：bridge / provider → witness（记录+验证）；witness → 仅
platform.interface（call_trace 读门面符号）。witness 不改绑定、不做选择——观测
与操作分离。

2026-10-15 由 ttorch/provenance.py + ttorch/diagnostics.py 收拢而成。
"""
from ttorch.witness import provenance  # noqa: F401  # 统一事件流（子模块）
from ttorch.witness.provenance import (  # noqa: F401
    record, records, report, summary, session, export_jsonl,
    LAYER_BRIDGE, LAYER_PROVIDER)
from ttorch.witness.bindings import check_bindings, assert_bindings_intact  # noqa: F401
from ttorch.witness.call_trace import start_call_trace, call_trace_report  # noqa: F401
from ttorch.witness.verify import verify_impl, measure_throughput_us  # noqa: F401

__all__ = [
    "provenance",
    "record", "records", "report", "summary", "session", "export_jsonl",
    "LAYER_BRIDGE", "LAYER_PROVIDER",
    "check_bindings", "assert_bindings_intact",
    "start_call_trace", "call_trace_report",
    "verify_impl", "measure_throughput_us",
]
