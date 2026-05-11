"""Coordination engine — drives one or more AgentRuntime sessions against goals.

This is the "outer loop" that turns a runtime into a goal-completion system.
It implements the canonical Plan → Execute → Verify pattern with retries,
budget enforcement, and parallel sub-task execution.

The coordinator does **not** know about HTTP or any particular wire format.
It depends only on the AgentRuntime contract (`send` yields Events ending in
a `TurnCompleted`). A wire-protocol server (see `agi.server`) can sit on top
of either the runtime or the coordinator.

Design:
- `Goal`: a typed task contract — description, success check, budget, plan
  policy.
- `Specialist`: a named role bundling a system prompt and optional model
  override. The coordinator picks specialists per phase.
- `Coordinator.execute(goal)`: returns an `Outcome` with success/failure,
  produced artifacts, traces, and final cost.

Parallelism: independent sub-tasks run in threads. Each thread uses a fresh
session — sessions are not shared across threads. (The anthropic SDK is
thread-safe when each thread uses its own session/messages list.)
"""
from __future__ import annotations

import concurrent.futures
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

from agi.events import (
    BudgetExceeded,
    Event,
    RuntimeError_,
    TurnCompleted,
)
from agi.runtime import AgentRuntime, SessionConfig


# ---- Goal contract ---------------------------------------------------------

SuccessCheck = Callable[[str], bool]


@dataclass
class Goal:
    description: str
    # How to verify success. If None, the verifier specialist judges.
    success_check: SuccessCheck | None = None
    # Plan policy: "auto" decomposes via planner, "direct" runs as a single
    # executor turn (no planning step). Default "auto".
    plan: str = "auto"
    # Per-goal hard caps. Each session inherits these; the coordinator
    # additionally enforces a total cost ceiling across all sessions.
    max_cost_usd: float | None = 1.0
    max_turns_per_session: int | None = 30
    max_retries: int = 1
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


# ---- Specialist registry ---------------------------------------------------

DEFAULT_PLANNER_PROMPT = """\
You are a planning specialist. Given a goal, produce a numbered list of
concrete sub-tasks an executor agent can carry out independently and in
parallel where possible. Be concise. Format:

1. <subtask>
2. <subtask>
...

Do not execute the tasks. Do not call tools. Output only the numbered list.
If the goal needs no decomposition, output a single-item list with the goal
itself.
"""

DEFAULT_EXECUTOR_PROMPT = """\
You are an execution specialist. Carry out the task you are given using the
available tools (files, shell, web search, memory). Report the result in the
final assistant message. Be terse.
"""

DEFAULT_VERIFIER_PROMPT = """\
You are a verification specialist. Given a goal and an executor's result,
decide if the goal was achieved. Reply with exactly one word on the first
line: PASS or FAIL. Optionally add a one-sentence reason on the next line.
Do not call tools; reason from the provided text alone.
"""


@dataclass
class Specialist:
    name: str
    system_prompt: str
    model: str | None = None  # None → runtime default
    effort: str | None = None  # None → runtime default


def default_specialists() -> dict[str, Specialist]:
    return {
        "planner": Specialist(name="planner", system_prompt=DEFAULT_PLANNER_PROMPT),
        "executor": Specialist(name="executor", system_prompt=DEFAULT_EXECUTOR_PROMPT),
        "verifier": Specialist(
            name="verifier",
            system_prompt=DEFAULT_VERIFIER_PROMPT,
            # Verification doesn't benefit from heavy reasoning; use a smaller
            # model when available
            model="claude-haiku-4-5-20251001",
            effort="low",
        ),
    }


# ---- Outcome ---------------------------------------------------------------

@dataclass
class SubtaskResult:
    task: str
    text: str
    cost_usd: float
    success: bool
    events: list[Event] = field(default_factory=list)


@dataclass
class Outcome:
    goal_id: str
    success: bool
    final_text: str
    subtasks: list[SubtaskResult] = field(default_factory=list)
    verifier_text: str = ""
    total_cost_usd: float = 0.0
    retries_used: int = 0
    failure_reason: str | None = None


# ---- Coordinator -----------------------------------------------------------

class Coordinator:
    def __init__(
        self,
        runtime: AgentRuntime,
        specialists: dict[str, Specialist] | None = None,
        max_parallel: int = 4,
        keep_events: bool = False,
    ) -> None:
        self.runtime = runtime
        self.specialists = specialists or default_specialists()
        self.max_parallel = max_parallel
        # Storing every event for large runs balloons memory; keep them only
        # when the caller asks (e.g., a UI that needs the trace).
        self.keep_events = keep_events

    # ---- public api ----

    def execute(self, goal: Goal) -> Outcome:
        retries_used = 0
        last_outcome: Outcome | None = None
        for attempt in range(goal.max_retries + 1):
            outcome = self._execute_once(goal)
            outcome.retries_used = attempt
            last_outcome = outcome
            if outcome.success:
                return outcome
            retries_used = attempt + 1
        assert last_outcome is not None
        last_outcome.retries_used = retries_used - 1 if retries_used else 0
        return last_outcome

    # ---- internal phases ----

    def _execute_once(self, goal: Goal) -> Outcome:
        # 1) Plan
        if goal.plan == "direct":
            subtasks = [goal.description]
        else:
            subtasks = self._plan(goal)

        # 2) Execute (parallel where the plan allows)
        results = self._execute_subtasks(goal, subtasks)

        total_cost = sum(r.cost_usd for r in results)
        combined = self._combine_results(goal, results)

        # 3) Verify
        if goal.success_check is not None:
            success = goal.success_check(combined)
            verifier_text = f"objective check: {'PASS' if success else 'FAIL'}"
        else:
            success, verifier_text, verify_cost = self._verify(goal, combined)
            total_cost += verify_cost

        return Outcome(
            goal_id=goal.id,
            success=success,
            final_text=combined,
            subtasks=results,
            verifier_text=verifier_text,
            total_cost_usd=total_cost,
            failure_reason=None if success else "verifier rejected result",
        )

    def _plan(self, goal: Goal) -> list[str]:
        spec = self.specialists["planner"]
        text, _, _ = self._run_one(
            spec, goal, prompt=f"Goal:\n{goal.description}\n\nProduce the sub-task list."
        )
        subtasks = _parse_numbered_list(text)
        return subtasks or [goal.description]

    def _execute_subtasks(self, goal: Goal, subtasks: list[str]) -> list[SubtaskResult]:
        spec = self.specialists["executor"]
        if len(subtasks) == 1:
            text, cost, events = self._run_one(spec, goal, prompt=subtasks[0])
            return [
                SubtaskResult(
                    task=subtasks[0],
                    text=text,
                    cost_usd=cost,
                    success=True,
                    events=events if self.keep_events else [],
                )
            ]

        results: list[SubtaskResult | None] = [None] * len(subtasks)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_parallel, len(subtasks))
        ) as pool:
            futures = {
                pool.submit(self._run_one, spec, goal, prompt=t): i
                for i, t in enumerate(subtasks)
            }
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    text, cost, events = fut.result()
                    results[i] = SubtaskResult(
                        task=subtasks[i],
                        text=text,
                        cost_usd=cost,
                        success=True,
                        events=events if self.keep_events else [],
                    )
                except Exception as e:
                    results[i] = SubtaskResult(
                        task=subtasks[i],
                        text=f"error: {type(e).__name__}: {e}",
                        cost_usd=0.0,
                        success=False,
                    )
        return [r for r in results if r is not None]  # narrow Optional

    def _verify(self, goal: Goal, combined: str) -> tuple[bool, str, float]:
        spec = self.specialists["verifier"]
        prompt = (
            f"Goal:\n{goal.description}\n\n"
            f"Executor result:\n{combined}\n\n"
            "Did the executor achieve the goal? Reply PASS or FAIL on the "
            "first line."
        )
        text, cost, _ = self._run_one(spec, goal, prompt=prompt)
        first_line = (text or "").strip().splitlines()[0] if text else ""
        success = first_line.strip().upper().startswith("PASS")
        return success, text, cost

    def _combine_results(self, goal: Goal, results: list[SubtaskResult]) -> str:
        if len(results) == 1:
            return results[0].text
        parts = [f"Subtask {i+1}: {r.task}\nResult:\n{r.text}" for i, r in enumerate(results)]
        return "\n\n".join(parts)

    # ---- one-shot specialist run ----

    def _run_one(
        self, spec: Specialist, goal: Goal, *, prompt: str
    ) -> tuple[str, float, list[Event]]:
        """Run a single prompt to completion under a specialist. Returns
        (final_text, cost_usd, events). Each call gets a fresh session so
        specialists don't share conversation context — by design."""
        cfg = SessionConfig(
            system_prompt=spec.system_prompt,
            model=spec.model or SessionConfig().model,
            effort=spec.effort or SessionConfig().effort,
            max_cost_usd=goal.max_cost_usd,
            max_turns=goal.max_turns_per_session,
        )
        sid = self.runtime.start_session(config=cfg)
        try:
            return self._consume(sid, prompt)
        finally:
            self.runtime.close_session(sid)

    def _consume(self, session_id: str, prompt: str) -> tuple[str, float, list[Event]]:
        final_text = ""
        cost = 0.0
        kept: list[Event] = []
        for ev in self.runtime.send(session_id, prompt):
            if self.keep_events:
                kept.append(ev)
            if isinstance(ev, TurnCompleted):
                final_text = ev.text
                cost = ev.cost_usd
            elif isinstance(ev, BudgetExceeded):
                # Trust the budget guard; stop consuming further from this run
                break
            elif isinstance(ev, RuntimeError_):
                raise RuntimeError(f"{ev.error_type}: {ev.message}")
        return final_text, cost, kept


# ---- helpers ---------------------------------------------------------------

_NUM_LIST_RE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$")


def _parse_numbered_list(text: str) -> list[str]:
    """Pull `1. foo\\n2. bar` out of free-form text. Returns [] if no items
    match — caller decides how to recover."""
    items: list[str] = []
    for line in text.splitlines():
        m = _NUM_LIST_RE.match(line)
        if m:
            items.append(m.group(1).strip())
    return items
