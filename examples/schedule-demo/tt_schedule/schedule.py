"""Schedule layer: what may overlap, and where we must wait.

This module owns ONLY the schedule description. It does not compute, it does not
call collectives, and it does not know the model name or the framework name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Step:
    """One unit of an execution plan.

    kind:
        "op"   -> compute, executed synchronously on its stream
        "comm" -> collective, may be launched asynchronously on its stream
        "wait" -> join barrier for one stream, or all streams when stream is None
    """

    name: str
    kind: str
    stream: str = "default"
    async_: bool = False
    needs: tuple = ()
    payload: Optional[Callable[[Dict[str, Any]], Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecPlan:
    """A concrete, inspectable schedule produced by a ScheduleProvider."""

    name: str
    steps: List[Step]

    def wait_steps(self) -> List[str]:
        return [s.name for s in self.steps if s.kind == "wait"]

    def async_ratio(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.async_) / len(self.steps)


class ScheduleProvider:
    """Answers: given a module request, which steps may overlap?"""

    name = "schedule-base"
    kind = "schedule"

    def capabilities(self, backend: Any) -> Dict[str, bool]:
        return {}

    def build_plan(self, req: "ModuleRequest") -> ExecPlan:
        raise NotImplementedError


@dataclass
class ModuleRequest:
    """What the schedule provider is allowed to see.

    No model name, no framework name. Only semantics, shapes and backend facts.
    """

    module_kind: str
    phase: str
    shape: Dict[str, int]
    dtype: str
    parallel: Dict[str, Any]
    backend_caps: Dict[str, bool]
    state: Dict[str, Any] = field(default_factory=dict)
