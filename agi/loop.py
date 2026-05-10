"""Autonomous goal-loop.

Given a goal and a stop predicate, the loop drives a Session step-by-step
until: the agent reports DONE, the budget is exhausted (USD or steps), or
the stop predicate fires. This is the unit a coordination engine
schedules — "achieve X, you have 50 cents and 8 turns" — and the unit a
human would call "the agent did the task."

The loop is intentionally thin. It does not retry, does not branch, does
not parallelize. A coordinator that needs those behaviors composes
multiple goal-loops on top of this primitive.

Stop conditions, in priority order:
1. Goal already complete (per the agent's own DONE marker).
2. Hard budget exhausted (steps or USD cost).
3. Custom stop predicate returns True.
4. Step error (model refusal, API failure, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from agi.agent import StepResult
from agi.runtime import Session


_DONE_PATTERN = re.compile(
    r"(?:\b(?:GOAL_COMPLETE|GOAL DONE|TASK_COMPLETE|TASK DONE)\b)|(?:\bDONE\.)",
    re.IGNORECASE,
)


@dataclass
class GoalBudget:
    max_steps: int = 8
    max_cost_usd: float = 1.00


@dataclass
class GoalRunResult:
    goal: str
    completed: bool
    stop_reason: str  # "done" | "max_steps" | "max_cost" | "predicate" | "error"
    steps: list[StepResult] = field(default_factory=list)
    final_text: str = ""
    cost_usd: float = 0.0
    error: str | None = None

    def summary(self) -> dict:
        return {
            "goal": self.goal,
            "completed": self.completed,
            "stop_reason": self.stop_reason,
            "steps": len(self.steps),
            "cost_usd": self.cost_usd,
            "final_text": self.final_text,
            "error": self.error,
        }


GOAL_KICKOFF_TEMPLATE = """\
Your goal for this loop is:

{goal}

Operate autonomously: plan, then execute step by step using your tools.
After each turn, decide whether the goal is fully achieved. When (and only
when) you are confident the goal is complete and verified, write the
literal token GOAL_COMPLETE on its own line at the end of your reply.

Hard budget: at most {max_steps} turns and ${max_cost:.2f} of model spend.
The loop will stop you if you exceed it. Be efficient.
"""

GOAL_CONTINUE_TEMPLATE = """\
Continue toward the goal. {budget_hint} Take the next concrete action; if
the goal is now fully achieved and verified, end your reply with
GOAL_COMPLETE on its own line.
"""


def _budget_hint(steps_left: int, cost_left: float) -> str:
    return f"({steps_left} turns and ${cost_left:.2f} remaining.)"


def is_done(text: str) -> bool:
    return bool(_DONE_PATTERN.search(text or ""))


def run_goal(
    session: Session,
    goal: str,
    *,
    budget: GoalBudget | None = None,
    stop: Callable[[Session, StepResult], bool] | None = None,
    max_iterations_per_step: int = 25,
) -> GoalRunResult:
    """Drive `session` toward `goal` until done, budget out, or predicate fires.

    The session's existing state is preserved — call this on a fresh
    session if you want a clean run. Each call to `run_goal` does push
    a new "kickoff" message describing the goal and budget.
    """
    budget = budget or GoalBudget()
    result = GoalRunResult(goal=goal, completed=False, stop_reason="error")

    initial_cost = session.agent.usage.cost_usd(session.agent.model)
    kickoff = GOAL_KICKOFF_TEMPLATE.format(
        goal=goal.strip(),
        max_steps=budget.max_steps,
        max_cost=budget.max_cost_usd,
    )

    for step_i in range(budget.max_steps):
        if step_i == 0:
            user_input = kickoff
        else:
            cost_used = session.agent.usage.cost_usd(session.agent.model) - initial_cost
            user_input = GOAL_CONTINUE_TEMPLATE.format(
                budget_hint=_budget_hint(
                    steps_left=budget.max_steps - step_i,
                    cost_left=max(budget.max_cost_usd - cost_used, 0.0),
                )
            )

        try:
            step = session.step(user_input, max_iterations=max_iterations_per_step)
        except Exception as e:  # noqa: BLE001
            result.error = f"{type(e).__name__}: {e}"
            result.stop_reason = "error"
            break

        result.steps.append(step)
        result.final_text = step.text
        result.cost_usd = session.agent.usage.cost_usd(session.agent.model) - initial_cost

        if step.error:
            result.error = step.error
            result.stop_reason = "error"
            break

        if is_done(step.text):
            result.completed = True
            result.stop_reason = "done"
            break

        if stop is not None and stop(session, step):
            result.stop_reason = "predicate"
            break

        if result.cost_usd >= budget.max_cost_usd:
            result.stop_reason = "max_cost"
            break
    else:
        # Loop exited via for-else: budget.max_steps reached without `break`
        result.stop_reason = "max_steps"

    return result
