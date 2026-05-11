"""Runtime, Session, and tool-integration tests using a fake Agent."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import Budget, EventBus, Memory, Runtime, SkillLibrary, capability_manifest, filter_types
from tests._fakes import constant_factory, counting_factory


class TestManifest(unittest.TestCase):
    def test_manifest_has_required_fields(self):
        m = capability_manifest()
        self.assertEqual(m["name"], "agi-runtime")
        self.assertIn("roles", m)
        self.assertIn("tools", m)
        self.assertIn("events", m)
        names = {r["name"] for r in m["roles"]}
        self.assertIn("planner", names)
        self.assertIn("executor", names)
        self.assertIn("critic", names)


class TestRuntimeBasics(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.skills = SkillLibrary(path=self.tmp / "skills")
        self.memory = Memory(path=self.tmp / "mem.jsonl")
        self.bus = EventBus(history=64)

    def tearDown(self):
        self._tmp.cleanup()

    def _runtime(self, factory):
        return Runtime(
            skill_library=self.skills,
            memory=self.memory,
            bus=self.bus,
            tracer=None,
            agent_factory=factory,
        )

    def test_open_chat_close(self):
        rt = self._runtime(constant_factory("hello"))
        s = rt.open_session(role="general")
        result = s.chat("hi")
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["stop_reason"], "ok")
        self.assertEqual(result["session_id"], s.id)
        self.assertEqual(len(rt.list_sessions()), 1)
        rt.close_session(s.id)
        self.assertEqual(len(rt.list_sessions()), 0)

    def test_session_emits_lifecycle_events(self):
        rt = self._runtime(constant_factory("hi"))
        s = rt.open_session()
        s.chat("ping")
        rt.close_session(s.id)
        types = [e["type"] for e in rt.history()]
        self.assertIn("session_opened", types)
        self.assertIn("turn_started", types)
        self.assertIn("turn_finished", types)
        self.assertIn("session_closed", types)

    def test_role_prompt_appears_in_system_extra(self):
        rt = self._runtime(constant_factory("ok"))
        s = rt.open_session(role="planner")
        # Inspect the agent's composed system_extra (constructor arg).
        self.assertIn("PLANNER", s.agent.system_extra)

    def test_skill_loaded_event(self):
        self.skills.save(
            name="addition",
            description="solve addition word problems",
            body="add the numbers; return the sum.",
            tags=["math"],
        )
        rt = self._runtime(constant_factory("3"))
        s = rt.open_session()
        s.chat("solve this addition problem: 1+2")
        events = filter_types(rt.history(), "skill_loaded")
        self.assertGreaterEqual(len(events), 1)

    def test_max_sessions_cap(self):
        rt = Runtime(
            skill_library=self.skills,
            memory=self.memory,
            bus=self.bus,
            max_sessions=2,
            agent_factory=constant_factory("ok"),
        )
        rt.open_session()
        rt.open_session()
        with self.assertRaises(RuntimeError):
            rt.open_session()

    def test_close_all(self):
        rt = self._runtime(constant_factory("ok"))
        rt.open_session()
        rt.open_session()
        n = rt.close_all()
        self.assertEqual(n, 2)
        self.assertEqual(rt.list_sessions(), [])

    def test_chat_unknown_session_raises(self):
        rt = self._runtime(constant_factory("ok"))
        with self.assertRaises(KeyError):
            rt.chat("nope", "hi")

    def test_unknown_session_returns_none(self):
        rt = self._runtime(constant_factory("ok"))
        self.assertIsNone(rt.get_session("nope"))


class TestDelegation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_delegate_emits_spawn_and_return(self):
        factory, counter = counting_factory("done")
        rt = Runtime(
            skill_library=SkillLibrary(path=self.tmp / "s"),
            memory=Memory(path=self.tmp / "m.jsonl"),
            bus=EventBus(history=128),
            agent_factory=factory,
        )
        parent = rt.open_session()
        result = rt.delegate(parent_id=parent.id, task="do thing", role="executor")
        self.assertEqual(result["stop_reason"], "ok")
        self.assertGreaterEqual(counter["n"], 1)
        types = [e["type"] for e in rt.history()]
        self.assertIn("delegate_spawn", types)
        self.assertIn("delegate_return", types)
        # child got closed
        self.assertEqual(len(rt.list_sessions()), 1)  # only parent left

    def test_delegate_unknown_parent_raises(self):
        rt = Runtime(
            skill_library=SkillLibrary(path=self.tmp / "s"),
            memory=Memory(path=self.tmp / "m.jsonl"),
            bus=EventBus(),
            agent_factory=constant_factory(),
        )
        with self.assertRaises(KeyError):
            rt.delegate(parent_id="nope", task="x")


class TestBudgetIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_zero_turn_budget_short_circuits(self):
        rt = Runtime(
            skill_library=SkillLibrary(path=self.tmp / "s"),
            memory=Memory(path=self.tmp / "m.jsonl"),
            bus=EventBus(history=64),
            agent_factory=constant_factory("hi"),
        )
        # max_turns=0 → BudgetExceeded fires before the first model call.
        s = rt.open_session(budget=Budget(max_turns=0))
        result = s.chat("hello")
        self.assertTrue(result["stop_reason"].startswith("budget_exceeded"))
        self.assertEqual(result["text"], "")


if __name__ == "__main__":
    unittest.main()
