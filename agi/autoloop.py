"""Autonomous goal loop — iterate until the goal is met or budget exhausted.

A `Coordinator.run(goal)` is *one* attempt. Real autonomous behavior is:

    while not acceptance(result) and budget > 0 and iterations < max:
        attempt = run(goal_with_lessons_so_far)
        if attempt.success: return attempt
        lessons = analyze(attempt)
        goal = augment_with_lessons(goal, lessons)

That is what `AutonomousLoop` implements. Each retry attaches the
prior failures' lessons to the goal's intent so the next plan can
avoid the same mistakes. On final success or after iteration cap,
optionally mines a Skill candidate from the winning trajectory so the
next *similar* goal is cheaper.

This is what a coordination engine asks the runtime for when it
wants "pursue this until done" semantics. The runtime supplies the
observable per-attempt steps; the AutonomousLoop supplies the
*persistence*.

A few invariants:
  - Each iteration runs inside the same overall budget. Halt the
    moment cumulative cost ≥ goal.budget_usd.
  - Critic scores, when present, count: a result with score ≥
    `accept_critic_score` is treated as success even if no explicit
    acceptance callback is given.
  - Lessons are bounded in size; we keep the most recent K lessons,
    not the full trace history, so the prompt doesn't grow unboundedly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.capabilities import CapabilityRegistry
from agi.coordinator import (
    CoordinationResult,
    Coordinator,
    Goal,
)
from agi.events import Event
from agi.skillmine import SkillCandidate, propose_skill_from_cluster
from agi.skills import Skill


@dataclass
class IterationOutcome:
    iteration: int
    success: bool
    final_text: str
    cost_usd: float
    duration_seconds: float
    lesson: str | None = None
    coordination_result: CoordinationResult | None = None


@dataclass
class AutonomousResult:
    goal: Goal
    iterations: list[IterationOutcome]
    success: bool
    final_text: str
    total_cost_usd: float
    total_duration_seconds: float
    skill_candidate: SkillCandidate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


LessonAnalyzer = Callable[[Goal, CoordinationResult], str | None]


def default_lesson_analyzer(goal: Goal, result: CoordinationResult) -> str | None:
    """Extract a one-sentence lesson from a failed attempt.

    Heuristic baseline; coordinators with an LLM available swap in an
    LLM-driven analyzer. The default looks for the first failed step
    and reports its error or empty result.
    """
    if result.success:
        return None
    if not result.outcomes:
        return f"No step ran for goal {goal.intent!r}; refine the plan before retrying."
    failed = [o for o in result.outcomes if o.status != "done" or not (o.result or "").strip()]
    if not failed:
        # Acceptance failed despite all steps "done"
        first = result.outcomes[0]
        return (
            f"Acceptance failed on step {first.step_id}: produced "
            f"{(first.result or '').strip()[:160]!r}. Tighten the response."
        )
    f = failed[0]
    why = f.error or "empty result"
    return f"Step {f.step_id} did not succeed ({why}). Address this on the next attempt."


class AutonomousLoop:
    """Coordinator-driver that retries a goal with lessons until success.

    Parameters
    ----------
    coordinator
        The reference Coordinator (or any Coordinator subclass).
    max_iterations
        Hard cap on retries.
    accept_critic_score
        If the final attempt produces a critic score ≥ this, count it
        as success even without explicit acceptance.
    lesson_analyzer
        Pluggable (goal, result) → lesson; default is heuristic.
    capabilities
        Optional CapabilityRegistry; every iteration records its
        outcome so future routing can learn from this run.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        *,
        max_iterations: int = 4,
        accept_critic_score: float | None = None,
        lesson_analyzer: LessonAnalyzer = default_lesson_analyzer,
        capabilities: CapabilityRegistry | None = None,
        mine_skill_on_success: bool = True,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be ≥ 1")
        self.coordinator = coordinator
        self.runtime = coordinator.runtime
        self.max_iterations = max_iterations
        self.accept_critic_score = accept_critic_score
        self.lesson_analyzer = lesson_analyzer
        self.capabilities = capabilities
        self.mine_skill_on_success = mine_skill_on_success

    def pursue(self, goal: Goal) -> AutonomousResult:
        iterations: list[IterationOutcome] = []
        lessons: list[str] = []
        start = time.time()
        cost_so_far = 0.0
        final_text = ""
        success = False
        last_result: CoordinationResult | None = None

        for i in range(1, self.max_iterations + 1):
            attempt_goal = self._augment_goal(goal, lessons)
            self.runtime.bus.publish(
                Event(
                    kind="autoloop.iteration_started",
                    data={
                        "iteration": i,
                        "goal_intent": goal.intent,
                        "lessons_so_far": len(lessons),
                        "cost_so_far": cost_so_far,
                    },
                )
            )
            attempt_start = time.time()
            result = self.coordinator.run(attempt_goal)
            last_result = result
            cost_so_far += result.total_cost_usd
            attempt_success = result.success or self._critic_accepted(result)

            lesson: str | None = None
            if not attempt_success:
                try:
                    lesson = self.lesson_analyzer(goal, result)
                except Exception:
                    lesson = None
                if lesson:
                    lessons.append(lesson)
                    # Keep only the last 5 lessons in the prompt window
                    lessons = lessons[-5:]

            outcome = IterationOutcome(
                iteration=i,
                success=attempt_success,
                final_text=result.final_text,
                cost_usd=result.total_cost_usd,
                duration_seconds=time.time() - attempt_start,
                lesson=lesson,
                coordination_result=result,
            )
            iterations.append(outcome)
            self._record_to_capabilities(goal, result, attempt_success)
            self.runtime.bus.publish(
                Event(
                    kind="autoloop.iteration_completed",
                    data={
                        "iteration": i,
                        "success": attempt_success,
                        "cost_so_far": cost_so_far,
                        "lesson": lesson,
                    },
                )
            )

            if attempt_success:
                success = True
                final_text = result.final_text
                break

            if goal.budget_usd is not None and cost_so_far >= goal.budget_usd:
                self.runtime.bus.publish(
                    Event(
                        kind="autoloop.budget_exhausted",
                        data={
                            "goal_intent": goal.intent,
                            "iterations": i,
                            "cost_so_far": cost_so_far,
                        },
                    )
                )
                final_text = result.final_text
                break

            if goal.deadline_ts is not None and time.time() >= goal.deadline_ts:
                self.runtime.bus.publish(
                    Event(
                        kind="autoloop.deadline_exceeded",
                        data={
                            "goal_intent": goal.intent,
                            "iterations": i,
                        },
                    )
                )
                final_text = result.final_text
                break
        else:
            # for/else: loop exited via exhaustion of iterations
            if last_result is not None:
                final_text = last_result.final_text

        skill_candidate: SkillCandidate | None = None
        if success and self.mine_skill_on_success and last_result is not None:
            skill_candidate = self._mine_skill(goal, last_result)

        self.runtime.bus.publish(
            Event(
                kind="autoloop.completed" if success else "autoloop.failed",
                data={
                    "goal_intent": goal.intent,
                    "iterations": len(iterations),
                    "success": success,
                    "cost_usd": cost_so_far,
                },
            )
        )
        return AutonomousResult(
            goal=goal,
            iterations=iterations,
            success=success,
            final_text=final_text,
            total_cost_usd=cost_so_far,
            total_duration_seconds=time.time() - start,
            skill_candidate=skill_candidate,
        )

    # --- helpers ---------------------------------------------------------

    def _augment_goal(self, goal: Goal, lessons: list[str]) -> Goal:
        if not lessons:
            return goal
        lesson_block = "\n".join(f"- {l}" for l in lessons)
        new_intent = (
            f"{goal.intent}\n\n"
            f"# Lessons from prior attempts (do NOT repeat these mistakes):\n"
            f"{lesson_block}"
        )
        return Goal(
            intent=new_intent,
            acceptance=goal.acceptance,
            budget_usd=goal.budget_usd,
            deadline_ts=goal.deadline_ts,
            metadata={**goal.metadata, "autoloop_lessons": len(lessons)},
        )

    def _critic_accepted(self, result: CoordinationResult) -> bool:
        if self.accept_critic_score is None or not result.outcomes:
            return False
        # Look at the last outcome's session, if any, for a critic score.
        last = result.outcomes[-1]
        if not last.session_id:
            return False
        try:
            sess = self.runtime.get_session(last.session_id)
        except KeyError:
            return False
        score = sess.state.last_critic_score
        return score is not None and score >= self.accept_critic_score

    def _record_to_capabilities(
        self, goal: Goal, result: CoordinationResult, success: bool
    ) -> None:
        if self.capabilities is None:
            return
        for o in result.outcomes:
            role = "executor"
            model = "claude-opus-4-7"
            skills_used: list[str] = []
            critic_score: float | None = None
            if o.session_id:
                try:
                    sess = self.runtime.get_session(o.session_id)
                    role = sess.state.config.role or "executor"
                    model = sess.state.config.model
                    critic_score = sess.state.last_critic_score
                except KeyError:
                    pass
            # Per-step success: did this step return non-empty without error?
            step_success = (
                success and o.status == "done" and bool((o.result or "").strip())
            )
            self.capabilities.record(
                prompt=goal.intent,
                role=role,
                model=model,
                skills_used=skills_used,
                success=step_success,
                cost_usd=o.cost_usd,
                duration_seconds=o.duration_seconds,
                critic_score=critic_score,
                tag=goal.metadata.get("tag") if isinstance(goal.metadata, dict) else None,
            )

    def _mine_skill(
        self, goal: Goal, result: CoordinationResult
    ) -> SkillCandidate | None:
        """Compile a SkillCandidate from a winning attempt. The
        coordinator decides whether to promote it (auto or via
        review). The candidate is also embedded in AutonomousResult so
        a UI can show it."""
        prompts = [goal.intent]
        responses = [result.final_text]
        try:
            return propose_skill_from_cluster(
                prompts, responses, name_hint=goal.metadata.get("skill_name")
            )
        except Exception:
            return None


def promote_skill(
    runtime, candidate: SkillCandidate, *, min_confidence: int = 1
) -> Skill | None:
    """Convenience: save a SkillCandidate to the Runtime's library iff
    it meets a confidence floor (number of supporting traces).

    Returns the saved Skill, or None if not promoted.
    """
    if candidate.trace_count < min_confidence:
        return None
    skill = candidate.to_skill()
    runtime.save_skill(skill)
    runtime.bus.publish(
        Event(
            kind="autoloop.skill_promoted",
            data={
                "name": skill.name,
                "trace_count": candidate.trace_count,
            },
        )
    )
    return skill
