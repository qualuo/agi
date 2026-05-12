"""Tests for SessionFork — parallel-hypothesis racing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event
from agi.fork import ForkOutcome, ForkVariant, SessionFork, default_judge
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent, FakeUsage


class ScoredFakeAgent(FakeAgent):
    """FakeAgent whose response and critic score depend on its role."""

    _by_role = {
        "good": ("the right answer", 0.9),
        "ok": ("a mediocre answer", 0.5),
        "bad": ("", None),
    }

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.received_prompts.append(prompt)
        self.usage.input_tokens += 100
        self.usage.output_tokens += 50
        self.usage.turns += 1
        role = self.extra_system or "ok"
        # extra_system has the form "Role: good. Return..."
        if role.startswith("Role:"):
            role_token = role.split(":", 1)[1].split(".", 1)[0].strip()
        else:
            role_token = "ok"
        resp, score = self._by_role.get(role_token, ("default", 0.4))
        self.last_critic_score = score
        return resp


def _make_runtime(agent_factory=ScoredFakeAgent) -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=agent_factory,
    )


def _cfg(role: str) -> SessionConfig:
    return SessionConfig(
        system_prompt_extra=f"Role: {role}. Return a concise final answer.",
        role=role,
    )


class TestFork(unittest.TestCase):
    def test_race_returns_best_by_critic_score(self):
        rt = _make_runtime()
        fork = SessionFork(rt, max_workers=3)
        result = fork.race(
            "what is x?",
            [
                ForkVariant("good", _cfg("good")),
                ForkVariant("ok", _cfg("ok")),
                ForkVariant("bad", _cfg("bad")),
            ],
        )
        self.assertEqual(len(result.outcomes), 3)
        self.assertIsNotNone(result.winner)
        self.assertEqual(result.winner.variant.name, "good")
        self.assertTrue(result.success)

    def test_all_variants_run(self):
        rt = _make_runtime()
        fork = SessionFork(rt, max_workers=4)
        result = fork.race(
            "x",
            [
                ForkVariant(f"v{i}", _cfg("ok")) for i in range(4)
            ],
        )
        self.assertEqual(len(result.outcomes), 4)
        # All should have an attempt outcome
        statuses = [o.status for o in result.outcomes]
        self.assertEqual(statuses, ["done"] * 4)

    def test_cost_aggregates_across_variants(self):
        rt = _make_runtime()
        fork = SessionFork(rt, max_workers=2)
        result = fork.race(
            "y",
            [
                ForkVariant("good", _cfg("good")),
                ForkVariant("ok", _cfg("ok")),
            ],
        )
        per_variant = [o.cost_usd for o in result.outcomes]
        self.assertGreater(result.total_cost_usd, 0)
        self.assertAlmostEqual(result.total_cost_usd, sum(per_variant), places=6)

    def test_empty_variants_raises(self):
        rt = _make_runtime()
        fork = SessionFork(rt)
        with self.assertRaises(ValueError):
            fork.race("z", [])

    def test_default_judge_prefers_higher_critic(self):
        high = ForkOutcome(
            variant=ForkVariant("a", SessionConfig()),
            task_id="t1", session_id="s1", status="done",
            result="x", error=None, cost_usd=0.001, duration_seconds=1.0,
            critic_score=0.9,
        )
        low = ForkOutcome(
            variant=ForkVariant("b", SessionConfig()),
            task_id="t2", session_id="s2", status="done",
            result="y", error=None, cost_usd=0.001, duration_seconds=1.0,
            critic_score=0.2,
        )
        self.assertGreater(default_judge(high), default_judge(low))

    def test_default_judge_rejects_empty_result(self):
        empty = ForkOutcome(
            variant=ForkVariant("e", SessionConfig()),
            task_id="t", session_id="s", status="done",
            result="", error=None, cost_usd=0.0, duration_seconds=0.0,
            critic_score=1.0,
        )
        self.assertLess(default_judge(empty), 0)

    def test_default_judge_penalizes_cost_for_tie_break(self):
        cheap = ForkOutcome(
            variant=ForkVariant("c", SessionConfig()),
            task_id="t1", session_id="s1", status="done",
            result="x", error=None, cost_usd=0.001, duration_seconds=1.0,
            critic_score=None,
        )
        expensive = ForkOutcome(
            variant=ForkVariant("e", SessionConfig()),
            task_id="t2", session_id="s2", status="done",
            result="x", error=None, cost_usd=0.10, duration_seconds=1.0,
            critic_score=None,
        )
        # Both ties on "succeeded"; cheaper should win
        self.assertGreater(default_judge(cheap), default_judge(expensive))


class TestForkEvents(unittest.TestCase):
    def test_race_emits_started_and_completed_events(self):
        rt = _make_runtime()
        events: list[Event] = []
        rt.subscribe(events.append, kind="fork.race_started")
        rt.subscribe(events.append, kind="fork.race_completed")
        fork = SessionFork(rt, max_workers=2)
        fork.race("p", [ForkVariant("a", _cfg("ok")), ForkVariant("b", _cfg("good"))])
        kinds = [e.kind for e in events]
        self.assertIn("fork.race_started", kinds)
        self.assertIn("fork.race_completed", kinds)


class TestCustomJudge(unittest.TestCase):
    def test_custom_judge_chooses_winner(self):
        rt = _make_runtime()
        # A judge that prefers variant named "ok" regardless of score
        def judge(o: ForkOutcome) -> float:
            return 1.0 if o.variant.name == "ok" else -1.0

        fork = SessionFork(rt, judge=judge, max_workers=3)
        result = fork.race(
            "p",
            [
                ForkVariant("good", _cfg("good")),
                ForkVariant("ok", _cfg("ok")),
                ForkVariant("bad", _cfg("bad")),
            ],
        )
        self.assertEqual(result.winner.variant.name, "ok")


if __name__ == "__main__":
    unittest.main()
