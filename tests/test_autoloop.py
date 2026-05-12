"""Tests for AutonomousLoop — self-improving goal pursuit."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.autoloop import (
    AutonomousLoop,
    default_lesson_analyzer,
    promote_skill,
)
from agi.capabilities import CapabilityRegistry
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.events import Event
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent, FakeUsage


class FailingFakeAgent(FakeAgent):
    """Fails the first N calls (returns empty), then succeeds."""

    _fail_remaining_global: int = 0

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.received_prompts.append(prompt)
        self.usage.input_tokens += 100
        self.usage.output_tokens += 50
        self.usage.turns += 1
        if FailingFakeAgent._fail_remaining_global > 0:
            FailingFakeAgent._fail_remaining_global -= 1
            return ""  # empty result trips acceptance
        return "answer: ok"


def _make_runtime(agent_factory=FakeAgent) -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=agent_factory,
    )


class TestAutoLoopHappyPath(unittest.TestCase):
    def test_single_iteration_when_first_attempt_succeeds(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=3)
        result = loop.pursue(Goal(intent="trivial task"))
        self.assertTrue(result.success)
        self.assertEqual(len(result.iterations), 1)
        self.assertIn("ok", result.final_text)

    def test_acceptance_callback_drives_retry(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        # FakeAgent always returns "ok"; acceptance rejects it
        loop = AutonomousLoop(coord, max_iterations=2)
        result = loop.pursue(Goal(intent="thing", acceptance=lambda t: "NO" in t))
        self.assertFalse(result.success)
        # Should have done max_iterations attempts
        self.assertEqual(len(result.iterations), 2)

    def test_skill_candidate_mined_on_success(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=2, mine_skill_on_success=True)
        result = loop.pursue(
            Goal(intent="cluster these prompts together", metadata={"skill_name": "test"})
        )
        self.assertTrue(result.success)
        self.assertIsNotNone(result.skill_candidate)


class TestAutoLoopRetry(unittest.TestCase):
    def test_lessons_accumulate_across_failed_attempts(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=3)

        # Acceptance fails first two, passes on the third (we won't reach it
        # because acceptance is just a stub here; we check lesson accumulation
        # via the augmented prompt instead).
        seen_prompts: list[str] = []

        def always_fail_acceptance(_t: str) -> bool:
            return False

        # Spy on agent prompts via a session subscriber
        rt.subscribe(
            lambda e: seen_prompts.append(e.data.get("user_input", ""))
            if e.kind == "chat.started"
            else None,
            kind="chat.started",
        )

        loop.pursue(Goal(intent="hard task", acceptance=always_fail_acceptance))
        # Second + third iteration prompts should mention prior lessons
        self.assertGreaterEqual(len(seen_prompts), 2)
        self.assertIn("Lessons from prior attempts", seen_prompts[1])

    def test_failing_agent_eventually_succeeds(self):
        FailingFakeAgent._fail_remaining_global = 2
        rt = _make_runtime(agent_factory=FailingFakeAgent)
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=5)
        # Acceptance requires "answer:" — flagged "" early-iteration results
        # are not accepted, forcing retry.
        result = loop.pursue(
            Goal(intent="retry me", acceptance=lambda t: "answer:" in t)
        )
        # Eventually succeeded once FailingFakeAgent stopped failing.
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.iterations), 3)
        self.assertFalse(result.iterations[0].success)
        self.assertTrue(result.iterations[-1].success)

    def test_budget_halts_loop(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=10)
        # Acceptance always false; budget should cut us off early.
        result = loop.pursue(
            Goal(intent="never", acceptance=lambda t: False, budget_usd=0.002)
        )
        self.assertFalse(result.success)
        # At least one attempt ran, but not all 10
        self.assertLess(len(result.iterations), 10)


class TestEventsEmitted(unittest.TestCase):
    def test_events_emitted_per_iteration(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=2)

        events: list[Event] = []
        rt.subscribe(events.append)

        loop.pursue(Goal(intent="x"))
        kinds = [e.kind for e in events]
        self.assertIn("autoloop.iteration_started", kinds)
        self.assertIn("autoloop.iteration_completed", kinds)
        self.assertIn("autoloop.completed", kinds)

    def test_failure_emits_failed_event(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=1)
        events: list[Event] = []
        rt.subscribe(events.append, kind="autoloop.failed")
        loop.pursue(Goal(intent="x", acceptance=lambda t: False))
        self.assertEqual(len(events), 1)


class TestCapabilitiesIntegration(unittest.TestCase):
    def test_records_to_capability_registry(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        tmp = tempfile.mkdtemp()
        caps = CapabilityRegistry(path=Path(tmp) / "caps.jsonl")
        loop = AutonomousLoop(coord, max_iterations=1, capabilities=caps)
        loop.pursue(Goal(intent="record me", metadata={"tag": "demo"}))
        recs = caps.all()
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0].success)
        self.assertEqual(recs[0].tag, "demo")


class TestSkillPromotion(unittest.TestCase):
    def test_promote_skill_writes_to_runtime_library(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=1)
        result = loop.pursue(Goal(intent="abc def ghi"))
        self.assertIsNotNone(result.skill_candidate)
        skill = promote_skill(rt, result.skill_candidate)
        self.assertIsNotNone(skill)
        self.assertIn(skill.name, [s.name for s in rt.skills.all()])

    def test_promote_skill_below_threshold_skipped(self):
        rt = _make_runtime()
        coord = Coordinator(rt)
        loop = AutonomousLoop(coord, max_iterations=1)
        result = loop.pursue(Goal(intent="abc def ghi"))
        skill = promote_skill(rt, result.skill_candidate, min_confidence=99)
        self.assertIsNone(skill)


class TestLessonAnalyzer(unittest.TestCase):
    def test_default_returns_none_on_success(self):
        from agi.coordinator import CoordinationResult, Plan, StepOutcome

        result = CoordinationResult(
            goal=Goal(intent="x"),
            plan=Plan(steps=[]),
            outcomes=[],
            final_text="ok",
            success=True,
            total_cost_usd=0.0,
            total_duration_seconds=0.0,
        )
        self.assertIsNone(default_lesson_analyzer(Goal(intent="x"), result))

    def test_default_extracts_lesson_on_empty_step(self):
        from agi.coordinator import CoordinationResult, Plan, StepOutcome

        outcome = StepOutcome(
            step_id="s1",
            task_id="t1",
            session_id=None,
            status="done",
            result="",
            error=None,
            duration_seconds=0.0,
            cost_usd=0.0,
        )
        result = CoordinationResult(
            goal=Goal(intent="x"),
            plan=Plan(steps=[]),
            outcomes=[outcome],
            final_text="",
            success=False,
            total_cost_usd=0.0,
            total_duration_seconds=0.0,
        )
        lesson = default_lesson_analyzer(Goal(intent="x"), result)
        self.assertIsNotNone(lesson)
        self.assertIn("s1", lesson)


if __name__ == "__main__":
    unittest.main()
