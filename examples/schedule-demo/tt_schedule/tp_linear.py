"""TP linear: a ModuleProvider that consumes schedule plans and op providers.

Layering shown here:
    ScheduleProvider -> ExecPlan (when to wait, what may overlap)
    ModuleProvider   -> module math + parallel semantics (this file)
    OpProvider       -> a single kernel substitution (gemm here)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .runner import run_plan
from .schedule import ExecPlan, ModuleRequest, ScheduleProvider, Step


class NumpyGemmOp:
    """OpProvider: one kernel-level substitution. Knows nothing about schedules."""

    name = "numpy-gemm"
    kind = "op"

    def __call__(self, a: np.ndarray, b: np.ndarray, trans_b: bool = False) -> np.ndarray:
        time.sleep(0.004)  # stand-in for a real GEMM, releases the GIL
        return a @ (b.T if trans_b else b)


class TilingGemmOp:
    """Alternative op implementation, same math, different kernel path."""

    name = "tiling-gemm"
    kind = "op"

    def __init__(self, tile: int = 256) -> None:
        self.tile = tile

    def __call__(self, a: np.ndarray, b: np.ndarray, trans_b: bool = False) -> np.ndarray:
        time.sleep(0.004)
        bb = b.T if trans_b else b
        out = np.zeros((a.shape[0], bb.shape[1]), dtype=a.dtype)
        for i in range(0, a.shape[0], self.tile):
            out[i : i + self.tile] = a[i : i + self.tile] @ bb
        return out


class ReferenceTpSchedule(ScheduleProvider):
    """Portable plan: every collective is synchronous and joined immediately."""

    name = "tp-reference"
    kind = "schedule"

    def capabilities(self, backend: Any) -> Dict[str, bool]:
        return {"async_comm": False, "independent_stream": False}

    def build_plan(self, req: ModuleRequest) -> ExecPlan:
        if req.phase == "forward":
            return ExecPlan(
                "tp-forward-reference",
                [
                    Step("f.gemm", "op", "compute", False, ()),
                    Step("f.rs", "comm", "comm", False, ("f.gemm",)),
                    Step("f.wait", "wait", "comm", False, ("f.rs",)),
                    Step("f.out", "op", "compute", False, ("f.wait",)),
                ],
            )
        return ExecPlan(
            "tp-backward-reference",
            [
                Step("b.ag", "comm", "comm", False, ()),
                Step("b.wait_ag", "wait", "comm", False, ("b.ag",)),
                Step("b.dx", "op", "compute", False, ("b.wait_ag",)),
                Step("b.rs", "comm", "comm", False, ("b.dx",)),
                Step("b.wait_rs", "wait", "comm", False, ("b.rs",)),
                Step("b.dw", "op", "compute", False, ("b.wait_rs",)),
                Step("b.out", "op", "compute", False, ("b.dw",)),
            ],
        )


class OverlapTpSchedule(ScheduleProvider):
    """Overlap plan: dgrad reduce-scatter runs while wgrad GEMM computes."""

    name = "tp-overlap"
    kind = "schedule"

    def capabilities(self, backend: Any) -> Dict[str, bool]:
        return {
            "async_comm": True,
            "independent_stream": True,
            "comm_can_overlap_gemm": True,
        }

    def build_plan(self, req: ModuleRequest) -> ExecPlan:
        if req.phase == "forward":
            return ExecPlan(
                "tp-forward-overlap",
                [
                    Step("f.gemm", "op", "compute", False, ()),
                    Step("f.rs", "comm", "comm", True, ("f.gemm",)),
                    Step("f.wait", "wait", "comm", False, ("f.rs",)),
                    Step("f.out", "op", "compute", False, ("f.wait",)),
                ],
            )
        return ExecPlan(
            "tp-backward-overlap",
            [
                Step("b.ag", "comm", "comm", True, ()),
                Step("b.wait_ag", "wait", "comm", False, ("b.ag",)),
                Step("b.dx", "op", "compute", False, ("b.wait_ag",)),
                Step("b.rs", "comm", "comm", True, ("b.dx",)),
                Step("b.dw", "op", "compute", False, ("b.wait_ag",)),
                Step("b.wait_rs", "wait", "comm", False, ("b.rs",)),
                Step("b.out", "op", "compute", False, ("b.wait_rs", "b.dw")),
            ],
        )


def select_schedule(
    candidates, backend_caps: Dict[str, bool]
) -> tuple[ScheduleProvider, Dict[str, str]]:
    """Choose once, at construction time. Never called on the hot path."""
    trace: Dict[str, str] = {}
    eligible: list = []
    for cand in candidates:
        need = cand.capabilities(None)
        missing = [k for k, v in need.items() if v and not backend_caps.get(k, False)]
        if missing:
            trace[cand.name] = f"rejected: backend lacks {missing}"
        else:
            trace[cand.name] = "eligible"
            eligible.append(cand)

    if not eligible:
        chosen = candidates[0]
        trace[chosen.name] = "selected (forced reference)"
        return chosen, trace

    chosen = max(eligible, key=lambda c: len([v for v in c.capabilities(None).values() if v]))
    for cand in eligible:
        trace[cand.name] = "selected" if cand is chosen else "eligible, lower priority"
    return chosen, trace


class TpLinear:
    """ModuleProvider. Owns math and parallel semantics; borrows the schedule."""

    module_kind = "parallel_linear"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        tp: int,
        backend_caps: Dict[str, bool],
        op_provider=None,
        schedule_candidates=None,
    ) -> None:
        self.in_dim, self.out_dim, self.tp = in_dim, out_dim, tp
        self.backend_caps = backend_caps
        self.op = op_provider or NumpyGemmOp()
        cands = schedule_candidates or [ReferenceTpSchedule(), OverlapTpSchedule()]
        self.schedule, self.selection_trace = select_schedule(cands, backend_caps)
        # Plans are built once, not per step.
        self._plans = {
            "forward": self.schedule.build_plan(self._request("forward")),
            "backward": self.schedule.build_plan(self._request("backward")),
        }

    def _request(self, phase: str) -> ModuleRequest:
        return ModuleRequest(
            module_kind=self.module_kind,
            phase=phase,
            shape={"in": self.in_dim, "out": self.out_dim},
            dtype="fp32",
            parallel={"tp": self.tp},
            backend_caps=self.backend_caps,
        )

    def _comm(self, payload, latency: float = 0.02):
        def _run(state: Dict[str, Any]):
            time.sleep(latency)  # stand-in for a collective; releases the GIL
            return payload(state)

        return _run

    def forward(self, x: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
        def rs(state):
            y = state["f.gemm"]
            m, out = y.shape
            return y.reshape(m, self.tp, out // self.tp).sum(axis=1)

        plan = self._plans["forward"]
        steps = list(plan.steps)

        def bind(step: Step) -> Step:
            if step.name == "f.gemm":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            lambda s: self.op(s["x"], s["w"], trans_b=True))
            if step.name == "f.rs":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            self._comm(rs))
            if step.name == "f.out":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            lambda s: s["f.rs"])
            return step

        bound = ExecPlan(plan.name, [bind(s) for s in steps])
        return run_plan(bound, {"x": x, "w": w})

    def backward(self, grad_out: np.ndarray, x: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
        def ag(state):
            g = state["g"]
            return np.repeat(g, self.tp, axis=1)

        def rs_dx(state):
            dx = state["b.dx"]
            m, _ = dx.shape
            return dx.reshape(m, self.tp, dx.shape[1] // self.tp).sum(axis=1)

        plan = self._plans["backward"]
        steps = list(plan.steps)

        def bind(step: Step) -> Step:
            if step.name == "b.ag":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            self._comm(ag))
            if step.name == "b.dx":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            lambda s: self.op(s["b.ag"], s["w"]))
            if step.name == "b.rs":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            self._comm(rs_dx))
            if step.name == "b.dw":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            lambda s: self.op(s["b.ag"], s["x"], trans_b=True))
            if step.name == "b.out":
                return Step(step.name, step.kind, step.stream, step.async_, step.needs,
                            lambda s: {"dx": s["b.rs"], "dw": s["b.dw"]})
            return step

        bound = ExecPlan(plan.name, [bind(s) for s in steps])
        return run_plan(bound, {"g": grad_out, "x": x, "w": w})
