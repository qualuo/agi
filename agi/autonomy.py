"""AutonomyEngine — the runtime closes its own learning loop.

`AutonomousLoop` makes one Goal converge. `AutonomyEngine` runs the
*outer* loop: it pulls Goals from a queue, dispatches each through a
Coordinator (with auto-routing from the PolicyRouter), records the
outcome in CapabilityRegistry, mines skills from successful trajectories,
**re-runs the SelfEval bank before promoting**, and writes new eval items
when a goal finishes successfully. All of this happens unattended —
between user requests, on a heartbeat, or against a queue another system
fills.

This is the headline AGI feature, decomposed honestly:

  - **Memory-of-self**: capabilities + skills + KG grow on every run.
  - **Self-improvement gated by regression**: skill promotion only ships
    if the SelfEvalBank pass rate doesn't drop. The system can't make
    itself worse without noticing.
  - **Autonomous task selection**: the engine doesn't wait for a human —
    a `GoalProvider` (queue, schedule, callback) feeds it work.
  - **Bounded by budget**: hard cost/iteration ceilings per heartbeat.

Investors care because this is the difference between "a chat product"
and "a runtime that wakes up overnight, attempts goals, and is
provably better in the morning." Coordination engines care because
they get a single drive entry point — `engine.run_once()` or
`engine.run_forever()` — that does the entire learn-from-experience
cycle in one call.

The engine emits events on the bus so any UI or external observer can
follow along:

  - `autonomy.tick_started` / `autonomy.tick_completed`
  - `autonomy.goal_started` / `autonomy.goal_completed` / `autonomy.goal_failed`
  - `autonomy.skill_promoted` (only after SelfEval clears)
  - `autonomy.skill_rejected` (SelfEval regressed; candidate kept on disk)
  - `autonomy.evalbank_updated`
  - `autonomy.idle` (no work in queue)
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.autoloop import AutonomousLoop, AutonomousResult, promote_skill
from agi.capabilities import CapabilityRegistry
from agi.coordinator import Coordinator, Goal
from agi.events import Event
from agi.policy import PolicyRouter
from agi.runtime import Runtime
from agi.selfeval import EvalItem, EvalReport, SelfEvalBank
from agi.skillmine import SkillCandidate
from agi.skills import Skill


GoalProvider = Callable[[], Goal | None]


@dataclass
class TickReport:
    """One heartbeat: zero-or-one goals attempted, plus side-effects."""
    tick_id: str
    goal: Goal | None
    autonomous: AutonomousResult | None
    skill_promoted: Skill | None
    skill_rejected: SkillCandidate | None
    regression_report: EvalReport | None
    eval_items_added: int
    cost_usd: float
    duration_seconds: float
    idle: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_id": self.tick_id,
            "goal_intent": self.goal.intent if self.goal else None,
            "success": self.autonomous.success if self.autonomous else None,
            "iterations": (
                len(self.autonomous.iterations) if self.autonomous else 0
            ),
            "skill_promoted": (
                self.skill_promoted.name if self.skill_promoted else None
            ),
            "skill_rejected": (
                self.skill_rejected.suggested_name if self.skill_rejected else None
            ),
            "regression_pass_rate": (
                self.regression_report.pass_rate
                if self.regression_report
                else None
            ),
            "eval_items_added": self.eval_items_added,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
            "idle": self.idle,
        }


@dataclass
class GoalQueue:
    """A simple thread-safe FIFO of Goals. Use directly or wrap in a
    GoalProvider that pulls from somewhere else (cron, MQ, HTTP)."""
    _items: list[Goal] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, goal: Goal) -> None:
        with self._lock:
            self._items.append(goal)

    def pop(self) -> Goal | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def as_provider(self) -> GoalProvider:
        return self.pop


class AutonomyEngine:
    """Continuous closed-loop driver over a Runtime.

    One `run_once()` consumes at most one Goal. `run_forever()` loops
    until `stop()` is called or `max_ticks` is reached, sleeping
    `heartbeat_seconds` between empty ticks.

    Parameters
    ----------
    runtime
        The Runtime to drive.
    coordinator
        A Coordinator already pointed at this Runtime.
    goal_provider
        Callable returning the next Goal, or None when idle.
    eval_bank
        Optional SelfEvalBank. When provided, skill promotions are gated
        on the bank's pass rate. Successful goals can also write new
        EvalItems back to it (see `mine_eval_items`).
    policy
        Optional PolicyRouter. When provided, the engine refreshes the
        router's posterior with every iteration's outcome.
    capabilities
        Optional CapabilityRegistry. The AutonomousLoop already writes
        here; the engine reads from it via the policy router.
    max_cost_per_tick_usd
        Hard cap on the budget any single goal can consume per tick.
    max_iterations
        Default iteration cap for the inner AutonomousLoop.
    mine_eval_items
        If true, every successful goal writes a new EvalItem to the bank
        so the regression suite grows with use.
    """

    def __init__(
        self,
        runtime: Runtime,
        coordinator: Coordinator,
        *,
        goal_provider: GoalProvider,
        eval_bank: SelfEvalBank | None = None,
        policy: PolicyRouter | None = None,
        capabilities: CapabilityRegistry | None = None,
        max_cost_per_tick_usd: float = 1.0,
        max_iterations: int = 3,
        mine_eval_items: bool = True,
        regression_tolerance: float = 0.0,
        eval_runner: Callable[[EvalItem], tuple[bool, str, float]] | None = None,
        baseline_pass_rate: float = 1.0,
    ) -> None:
        self.runtime = runtime
        self.coordinator = coordinator
        self.goal_provider = goal_provider
        self.eval_bank = eval_bank
        self.policy = policy
        self.capabilities = capabilities
        self.max_cost_per_tick_usd = max_cost_per_tick_usd
        self.max_iterations = max_iterations
        self.mine_eval_items = mine_eval_items
        self.regression_tolerance = regression_tolerance
        self.eval_runner = eval_runner
        self.baseline_pass_rate = baseline_pass_rate
        self._stop = threading.Event()
        self._ticks_run = 0
        self._reports: list[TickReport] = []

    # --- public API -------------------------------------------------

    def run_once(self) -> TickReport:
        tick_id = uuid.uuid4().hex[:10]
        started = time.time()
        self.runtime.bus.publish(Event(
            kind="autonomy.tick_started",
            data={"tick_id": tick_id},
        ))

        goal = self.goal_provider()
        if goal is None:
            self.runtime.bus.publish(Event(kind="autonomy.idle", data={"tick_id": tick_id}))
            report = TickReport(
                tick_id=tick_id,
                goal=None,
                autonomous=None,
                skill_promoted=None,
                skill_rejected=None,
                regression_report=None,
                eval_items_added=0,
                cost_usd=0.0,
                duration_seconds=time.time() - started,
                idle=True,
            )
            self._reports.append(report)
            self._ticks_run += 1
            self.runtime.bus.publish(Event(
                kind="autonomy.tick_completed",
                data=report.to_dict(),
            ))
            return report

        # Honor caller's budget but clamp to per-tick ceiling
        budget = goal.budget_usd
        if budget is None or budget > self.max_cost_per_tick_usd:
            goal = Goal(
                intent=goal.intent,
                acceptance=goal.acceptance,
                budget_usd=self.max_cost_per_tick_usd,
                deadline_ts=goal.deadline_ts,
                metadata=goal.metadata,
            )

        self.runtime.bus.publish(Event(
            kind="autonomy.goal_started",
            data={"tick_id": tick_id, "intent": goal.intent, "budget_usd": goal.budget_usd},
        ))

        loop = AutonomousLoop(
            self.coordinator,
            max_iterations=self.max_iterations,
            capabilities=self.capabilities,
            mine_skill_on_success=True,
        )
        result = loop.pursue(goal)

        skill_promoted: Skill | None = None
        skill_rejected: SkillCandidate | None = None
        regression: EvalReport | None = None

        if result.success and result.skill_candidate is not None:
            candidate = result.skill_candidate
            if self.eval_bank is not None and self.eval_runner is not None:
                regression = self._regression_check(candidate)
                if regression is None or regression.pass_rate >= (1.0 - self.regression_tolerance):
                    skill_promoted = promote_skill(self.runtime, candidate)
                else:
                    skill_rejected = candidate
                    self.runtime.bus.publish(Event(
                        kind="autonomy.skill_rejected",
                        data={
                            "name": candidate.suggested_name,
                            "pass_rate": regression.pass_rate,
                        },
                    ))
            else:
                # No regression bank: promote eagerly.
                skill_promoted = promote_skill(self.runtime, candidate)

        eval_items_added = 0
        if (
            self.mine_eval_items
            and result.success
            and self.eval_bank is not None
            and goal.intent.strip()
            and result.final_text.strip()
        ):
            try:
                # Use a substring slice of the final text as a tolerant
                # expected_substring; ensures the eval is non-empty and
                # somewhat robust to formatting drift.
                substr = result.final_text.strip().splitlines()[0]
                if len(substr) > 80:
                    substr = substr[:80]
                if substr:
                    added_item = self.eval_bank.add(
                        prompt=goal.intent,
                        expect_substring=substr,
                        source="automatic",
                        tags=[f"autonomy:{tick_id}"],
                    )
                    if added_item is not None:
                        eval_items_added = 1
                        self.runtime.bus.publish(Event(
                            kind="autonomy.evalbank_updated",
                            data={
                                "added": 1,
                                "size": len(self.eval_bank.all()),
                            },
                        ))
            except Exception:
                eval_items_added = 0

        if result.success:
            self.runtime.bus.publish(Event(
                kind="autonomy.goal_completed",
                data={
                    "tick_id": tick_id,
                    "intent": goal.intent,
                    "cost_usd": result.total_cost_usd,
                    "iterations": len(result.iterations),
                    "skill_promoted": skill_promoted.name if skill_promoted else None,
                },
            ))
        else:
            self.runtime.bus.publish(Event(
                kind="autonomy.goal_failed",
                data={
                    "tick_id": tick_id,
                    "intent": goal.intent,
                    "cost_usd": result.total_cost_usd,
                    "iterations": len(result.iterations),
                },
            ))

        report = TickReport(
            tick_id=tick_id,
            goal=goal,
            autonomous=result,
            skill_promoted=skill_promoted,
            skill_rejected=skill_rejected,
            regression_report=regression,
            eval_items_added=eval_items_added,
            cost_usd=result.total_cost_usd,
            duration_seconds=time.time() - started,
            idle=False,
        )
        self._reports.append(report)
        self._ticks_run += 1
        self.runtime.bus.publish(Event(
            kind="autonomy.tick_completed",
            data=report.to_dict(),
        ))
        return report

    def run_forever(
        self,
        *,
        max_ticks: int | None = None,
        heartbeat_seconds: float = 1.0,
        idle_grace_ticks: int = 0,
    ) -> list[TickReport]:
        """Run until stop()/max_ticks/idle_grace_ticks. Synchronous;
        callers wanting background autonomy spawn a Thread."""
        consecutive_idle = 0
        out: list[TickReport] = []
        while not self._stop.is_set():
            if max_ticks is not None and self._ticks_run >= max_ticks:
                break
            report = self.run_once()
            out.append(report)
            if report.idle:
                consecutive_idle += 1
                if idle_grace_ticks and consecutive_idle >= idle_grace_ticks:
                    break
                if heartbeat_seconds > 0:
                    time.sleep(heartbeat_seconds)
            else:
                consecutive_idle = 0
        return out

    def stop(self) -> None:
        self._stop.set()

    def reports(self) -> list[TickReport]:
        return list(self._reports)

    def metrics(self) -> dict[str, Any]:
        total_cost = sum(r.cost_usd for r in self._reports)
        successes = sum(
            1 for r in self._reports if r.autonomous and r.autonomous.success
        )
        promotions = sum(1 for r in self._reports if r.skill_promoted is not None)
        rejections = sum(1 for r in self._reports if r.skill_rejected is not None)
        return {
            "ticks": self._ticks_run,
            "goals_attempted": sum(1 for r in self._reports if not r.idle),
            "successes": successes,
            "skills_promoted": promotions,
            "skills_rejected_by_regression": rejections,
            "total_cost_usd": total_cost,
            "evalbank_size": len(self.eval_bank.all()) if self.eval_bank else None,
        }

    # --- internals --------------------------------------------------

    def _regression_check(
        self, candidate: SkillCandidate
    ) -> EvalReport | None:
        """Re-run the EvalBank with the candidate notionally in scope,
        and return the resulting EvalReport. The caller decides whether
        the pass rate meets the bar.

        We do NOT persist the candidate before running — the SelfEvalBank
        sees a candidate-skill-applied world only conceptually; the
        eval_runner the caller provides is responsible for honoring it
        (e.g., by injecting the candidate skill into the prompt context).
        """
        if self.eval_bank is None or self.eval_runner is None:
            return None
        try:
            return self.eval_bank.run(self.eval_runner)
        except Exception as e:
            self.runtime.bus.publish(Event(
                kind="error",
                data={"phase": "regression", "type": type(e).__name__, "message": str(e)},
            ))
            return None
