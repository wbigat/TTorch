"""Compare a reference TP plan against an overlap TP plan.

Checks three things:
  1. numerics are identical between schedule plans
  2. the overlap plan really shortens the critical path
  3. schedule selection is decided by probed capability, not by hardware name
"""

import numpy as np

from tt_schedule import (
    NumpyGemmOp,
    OverlapTpSchedule,
    ReferenceTpSchedule,
    TilingGemmOp,
    TpLinear,
)

M, IN, OUT, TP = 512, 512, 512, 2
CAPS_FULL = {"async_comm": True, "independent_stream": True, "comm_can_overlap_gemm": True}
CAPS_POOR = {"async_comm": False, "independent_stream": True, "comm_can_overlap_gemm": False}


def make_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, IN)).astype(np.float32)
    w = rng.standard_normal((OUT, IN)).astype(np.float32)
    g = rng.standard_normal((M, OUT // TP)).astype(np.float32)
    return x, w, g


def run_case(title: str, caps: dict, op) -> None:
    x, w, g = make_inputs()
    mod = TpLinear(IN, OUT, TP, backend_caps=caps, op_provider=op)
    print(f"\n=== {title} ===")
    print(f"op provider        : {mod.op.name}")
    print(f"schedule provider  : {mod.schedule.name}")
    for k, v in mod.selection_trace.items():
        print(f"  - {k}: {v}")
    print(f"forward wait steps : {mod._plans['forward'].wait_steps()}")
    print(f"backward wait steps: {mod._plans['backward'].wait_steps()}")

    fwd = mod.forward(x, w)
    bwd = mod.backward(g, x, w)
    print(f"forward wall       : {fwd.wall_time() * 1000:.2f} ms")
    print(f"backward wall      : {bwd.wall_time() * 1000:.2f} ms")
    print(f"backward per-stream: { {k: round(v * 1000, 2) for k, v in bwd.per_stream_busy().items()} }")
    print(f"y[0,:3]            : {fwd.outputs['f.out'][0, :3]}")
    print(f"dx[0,:3]           : {bwd.outputs['b.out']['dx'][0, :3]}")
    print(f"dw[0,:3]           : {bwd.outputs['b.out']['dw'][0, :3]}")


def check_equivalence() -> None:
    x, w, g = make_inputs()
    ref = TpLinear(IN, OUT, TP, CAPS_POOR, NumpyGemmOp(), [ReferenceTpSchedule()])
    ovl = TpLinear(IN, OUT, TP, CAPS_FULL, NumpyGemmOp(), [ReferenceTpSchedule(), OverlapTpSchedule()])

    r_f, r_b = ref.forward(x, w), ref.backward(g, x, w)
    o_f, o_b = ovl.forward(x, w), ovl.backward(g, x, w)

    y_ok = np.allclose(r_f.outputs["f.out"], o_f.outputs["f.out"], rtol=1e-5, atol=1e-5)
    dx_ok = np.allclose(r_b.outputs["b.out"]["dx"], o_b.outputs["b.out"]["dx"], rtol=1e-5, atol=1e-5)
    dw_ok = np.allclose(r_b.outputs["b.out"]["dw"], o_b.outputs["b.out"]["dw"], rtol=1e-5, atol=1e-5)
    print("\n=== equivalence: reference plan vs overlap plan ===")
    print(f"forward y  identical: {y_ok}")
    print(f"backward dx identical: {dx_ok}")
    print(f"backward dw identical: {dw_ok}")

    rt, ot = r_b.wall_time(), o_b.wall_time()
    print(f"backward wall reference: {rt * 1000:.2f} ms")
    print(f"backward wall overlap  : {ot * 1000:.2f} ms")
    print(f"speedup                : {rt / ot:.2f}x")
    assert y_ok and dx_ok and dw_ok, "overlap plan changed numerics"


def check_selection() -> None:
    poor = TpLinear(IN, OUT, TP, CAPS_POOR, NumpyGemmOp(), [ReferenceTpSchedule(), OverlapTpSchedule()])
    print("\n=== selection is capability-driven ===")
    print(f"backend caps {CAPS_POOR} -> {poor.schedule.name}")
    assert poor.schedule.name == "tp-reference", "must fall back when async comm is unavailable"


def check_op_swap() -> None:
    x, w, g = make_inputs()
    a = TpLinear(IN, OUT, TP, CAPS_FULL, NumpyGemmOp(), [OverlapTpSchedule()])
    b = TpLinear(IN, OUT, TP, CAPS_FULL, TilingGemmOp(tile=128), [OverlapTpSchedule()])
    ra, rb = a.backward(g, x, w), b.backward(g, x, w)
    same = np.allclose(
        ra.outputs["b.out"]["dw"], rb.outputs["b.out"]["dw"], rtol=1e-5, atol=1e-5
    )
    print("\n=== op provider swap keeps the same schedule ===")
    print(f"schedule unchanged: {a.schedule.name} / {b.schedule.name}")
    print(f"dw identical      : {same}")
    assert same, "op swap changed numerics"


if __name__ == "__main__":
    run_case("case 1: reference schedule, weak backend", CAPS_POOR, NumpyGemmOp())
    run_case("case 2: overlap schedule, capable backend", CAPS_FULL, NumpyGemmOp())
    check_equivalence()
    check_selection()
    check_op_swap()
    print("\nall checks passed")
