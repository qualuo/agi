"""Reference coordinator.

A *coordination engine* drives the Runtime: it decides what to attempt,
in what order, with what budgets, and what to do with the results. The
Runtime executes; the coordinator decides.

`SimpleCoordinator` is a thin reference implementation that demonstrates
the contract a real coordination engine would honor:

  - Submit goals (one-shot or as a Plan).
  - Stream events for observability.
  - Inspect status, cancel, retry.
  - Aggregate metrics across runs.

It's deliberately small. Real coordinators will live outside this repo
(LangGraph, an orchestrator service, a workflow engine, your own); the
point of this file is that the surface they need is small and clean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from agi.budget import Budget
from agi.events import Event
from agi.plan import Plan, SubgoalResult, execute_plan
from agi.runtime import Runtime


@dataclass
class RunSummary:
    session_id: str
    status: str
    final_text: str
    cost_usd: float
    turns: int


class SimpleCoordinator:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def run(
        self,
        goal: str,
        *,
        budget: Optional[Budget] = None,
        timeout: Optional[float] = None,
    ) -> RunSummary:
        sid = self.runtime.submit(goal, budget=budget)
        record = self.runtime.wait(sid, timeout=timeout)
        return RunSummary(
            session_id=sid,
            status=record["status"],
            final_text=record["final_text"],
            cost_usd=record["total_cost_usd"],
            turns=record["turns"],
        )

    def run_plan(
        self,
        plan: Plan,
        *,
        timeout: Optional[float] = None,
        stop_on_error: bool = False,
    ) -> dict[str, SubgoalResult]:
        return execute_plan(
            self.runtime, plan, timeout=timeout, stop_on_error=stop_on_error
        )

    def stream(
        self, goal: str, *, budget: Optional[Budget] = None
    ) -> tuple[str, Iterator[Event]]:
        """Submit and return (session_id, live event iterator)."""
        sid = self.runtime.submit(goal, budget=budget)
        return sid, self.runtime.events(sid, replay=True, follow=True)
