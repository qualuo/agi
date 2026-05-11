"""Coordinator — a reference engine that drives the Runtime.

A Coordinator sits *above* the Runtime. It accepts a Goal (high-level
intent + acceptance criteria + budget), decomposes it into a Plan of
ordered or parallel steps, dispatches each step as a Task to the
Runtime's queue, monitors progress via the event stream, and aggregates
step results into a final outcome.

This module is intentionally a **reference implementation**: the
coordinator is one of *many* possible drivers of the Runtime. The point
is to demonstrate that the Runtime exposes enough surface for a planner
to do its job without holes:

  - Goal: declarative intent + scoring rubric (optional)
  - Plan: ordered list of PlanSteps (each step is a `delegate` unit)
  - Decomposer: produces a Plan from a Goal (pluggable; LLM-driven in
    production, rule/template-driven for tests and offline workflows)
  - Aggregator: turns step results into a final answer (default: concat
    with step labels)
  - Coordinator: glue. Submits the plan, waits for completion, returns
    the aggregated answer.

The Coordinator does NOT subclass or extend the Runtime — it talks to
the Runtime exclusively through the public API. Any other planner can
do the same.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.events import Event
from agi.runtime import Runtime, SessionConfig
from agi.tasks import (
    TASK_COMPLETED,
    TASK_FAILED,
    Task,
    TaskQueue,
    TaskRunner,
    submit_task,
)


@dataclass
class Goal:
    """A unit of intent for the Coordinator.

    `acceptance` is optional: a callable (result_text) → bool that
    decides whether the Plan satisfied the Goal. If absent, any
    non-empty result counts as success.
    """
    intent: str
    acceptance: Callable[[str], bool] | None = None
    budget_usd: float | None = None
    deadline_ts: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    """One unit of executable work inside a Plan.

    Dependencies (`depends_on`) name earlier step ids; the Coordinator
    won't dispatch a step until all its dependencies are done. This is
    how a planner expresses ordering or fan-out.
    """
    id: str
    prompt: str
    role: str = "executor"
    model: str | None = None
    depends_on: list[str] = field(default_factory=list)
    use_skills: bool = True
    priority: int = 0
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: list[PlanStep]
    rationale: str = ""

    def step_ids(self) -> set[str]:
        return {s.id for s in self.steps}


@dataclass
class StepOutcome:
    step_id: str
    task_id: str
    session_id: str | None
    status: str
    result: str | None
    error: str | None
    duration_seconds: float
    cost_usd: float


@dataclass
class CoordinationResult:
    goal: Goal
    plan: Plan
    outcomes: list[StepOutcome]
    final_text: str
    success: bool
    total_cost_usd: float
    total_duration_seconds: float


Decomposer = Callable[[Goal], Plan]
Aggregator = Callable[[Goal, Plan, list[StepOutcome]], str]


def single_step_decomposer(goal: Goal) -> Plan:
    """Default: one step, the goal's intent verbatim. Useful baseline.

    Production deployments swap this for an LLM-driven planner that
    reads the runtime's capabilities() and the skill library before
    deciding decomposition.
    """
    step = PlanStep(id="root", prompt=goal.intent, role="executor")
    return Plan(steps=[step], rationale="single-step fallback")


def label_aggregator(_goal: Goal, plan: Plan, outcomes: list[StepOutcome]) -> str:
    """Concatenate step results with step labels."""
    by_id = {o.step_id: o for o in outcomes}
    parts = []
    for step in plan.steps:
        o = by_id.get(step.id)
        if o is None or o.result is None:
            continue
        parts.append(f"## Step {step.id} ({step.role})\n{o.result}")
    return "\n\n".join(parts)


class Coordinator:
    """Reference coordination engine driving a Runtime.

    Lifecycle for one run():

      1. decomposer(goal) → Plan
      2. For each step in topological order (respecting depends_on),
         submit a Task to the queue, run the runner, capture the
         outcome.
      3. Enforce goal.budget_usd: stop dispatching new steps once
         accrued cost is at the ceiling. Pending steps go un-run.
      4. aggregator(goal, plan, outcomes) → final_text.
      5. acceptance(final_text) → success/failure.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        queue: TaskQueue | None = None,
        decomposer: Decomposer = single_step_decomposer,
        aggregator: Aggregator = label_aggregator,
        default_session_config: SessionConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.queue = queue or TaskQueue()
        self.decomposer = decomposer
        self.aggregator = aggregator
        self.default_session_config = default_session_config or SessionConfig(use_skills=True)
        self.runner = TaskRunner(self.runtime, self.queue)

    def run(self, goal: Goal) -> CoordinationResult:
        plan = self.decomposer(goal)
        outcomes: list[StepOutcome] = []
        cost_so_far = 0.0
        start_ts = time.time()
        done: set[str] = set()

        # Topological-ish dispatch: a step is dispatched once its deps are done.
        # If a step's deps are missing or cyclic, it never runs.
        remaining = {s.id: s for s in plan.steps}
        guard = 0
        while remaining and guard < 1000:
            guard += 1
            ready = [s for s in remaining.values() if set(s.depends_on).issubset(done)]
            if not ready:
                break
            for step in ready:
                if goal.budget_usd is not None and cost_so_far >= goal.budget_usd:
                    self.runtime.bus.publish(Event(
                        kind="coordinator.budget_exhausted",
                        data={"goal_intent": goal.intent, "remaining": list(remaining.keys())},
                    ))
                    remaining = {}
                    break
                outcome = self._run_step(goal, step)
                outcomes.append(outcome)
                cost_so_far += outcome.cost_usd
                done.add(step.id)
                remaining.pop(step.id, None)

        final_text = self.aggregator(goal, plan, outcomes)
        success = bool(final_text) and all(o.status == "done" for o in outcomes)
        if goal.acceptance is not None:
            success = success and goal.acceptance(final_text)
        result = CoordinationResult(
            goal=goal,
            plan=plan,
            outcomes=outcomes,
            final_text=final_text,
            success=success,
            total_cost_usd=cost_so_far,
            total_duration_seconds=time.time() - start_ts,
        )
        self.runtime.bus.publish(Event(
            kind="coordinator.completed" if success else "coordinator.failed",
            data={
                "goal_intent": goal.intent,
                "step_count": len(plan.steps),
                "cost_usd": cost_so_far,
                "success": success,
            },
        ))
        return result

    def _run_step(self, goal: Goal, step: PlanStep) -> StepOutcome:
        cfg = SessionConfig(
            **{**self.default_session_config.__dict__, "use_skills": step.use_skills,
               "system_prompt_extra": f"Role: {step.role}. Return a concise final answer.",
               "role": step.role,
               "cost_ceiling_usd": goal.budget_usd}
        )
        if step.model:
            cfg.model = step.model
        started = time.time()
        tid = submit_task(
            self.queue,
            prompt=step.prompt,
            session_config=cfg,
            priority=step.priority,
            deadline_ts=goal.deadline_ts,
            namespace=step.namespace,
            tag=goal.metadata.get("tag"),
        )
        # Drain this single task (the runner is single-threaded by design;
        # parallelism comes from running multiple runners in production).
        self.runner.tick()
        task = self.queue.get(tid)
        session = self.runtime.get_session(task.session_id) if task.session_id else None
        cost = session.state.total_cost_usd if session else 0.0
        return StepOutcome(
            step_id=step.id,
            task_id=tid,
            session_id=task.session_id,
            status=task.status,
            result=task.result,
            error=task.error,
            duration_seconds=time.time() - started,
            cost_usd=cost,
        )
