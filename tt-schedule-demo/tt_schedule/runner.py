"""Plan interpreter: turns an ExecPlan into real execution.

The runner is deliberately dumb: it honours dependencies, launches async steps
on their stream, and joins at explicit wait steps. All timing intelligence lives
in the ScheduleProvider that produced the plan.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class StepRecord:
    name: str
    stream: str
    start: float
    end: float


@dataclass
class RunResult:
    outputs: Dict[str, Any]
    records: List[StepRecord] = field(default_factory=list)
    wait_steps: List[str] = field(default_factory=list)

    def wall_time(self) -> float:
        if not self.records:
            return 0.0
        return max(r.end for r in self.records) - min(r.start for r in self.records)

    def per_stream_busy(self) -> Dict[str, float]:
        busy: Dict[str, float] = {}
        for r in self.records:
            if r.end > r.start:
                busy[r.stream] = busy.get(r.stream, 0.0) + (r.end - r.start)
        return busy


def _topo_order(steps: List[Step]) -> List[Step]:
    seen, order = set(), []

    def visit(step: Step, stack: Tuple[str, ...]) -> None:
        if step.name in seen:
            return
        if step.name in stack:
            raise ValueError(f"cycle in ExecPlan: {stack} -> {step.name}")
        for dep in step.needs:
            match = next((s for s in steps if s.name == dep), None)
            if match is None:
                raise ValueError(f"step {step.name} depends on unknown step {dep}")
            visit(match, stack + (step.name,))
        seen.add(step.name)
        order.append(step)

    for s in steps:
        visit(s, ())
    return order


def run_plan(plan: ExecPlan, state: Dict[str, Any]) -> RunResult:
    """Execute a plan. Steps without async_ run inline; async steps get a thread."""
    handles: Dict[str, threading.Thread] = {}
    records: List[StepRecord] = []
    wait_steps: List[str] = []
    t0 = time.perf_counter()

    def launch(step: Step) -> None:
        start = time.perf_counter()
        value = step.payload(state) if step.payload else None
        end = time.perf_counter()
        if value is not None:
            state[step.name] = value
        records.append(StepRecord(step.name, step.stream, start, end))

    def join(stream: str | None) -> None:
        for name, th in list(handles.items()):
            if stream is None or th.stream == stream:
                th.join()
                del handles[name]

    class _Handle(threading.Thread):
        def __init__(self, target, stream: str):
            super().__init__(target=target)
            self.stream = stream

    for step in _topo_order(plan.steps):
        if step.kind == "wait":
            join(step.stream)
            wait_steps.append(step.name)
            continue
        if step.async_:
            th = _Handle(target=lambda s=step: launch(s), stream=step.stream)
            th.start()
            handles[step.name] = th
        else:
            launch(step)

    join(None)
    return RunResult(outputs=state, records=records, wait_steps=wait_steps)
