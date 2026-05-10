"""Tests for agi.runtime, agi.events, agi.coordination — without API calls.

We use a FakeAgent that conforms to the duck-typed surface Session needs
so we can test the runtime contract end-to-end without hitting Claude.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.costs import Usage
from agi.events import EventBus, SESSION_CREATED, TURN_COMPLETED, TURN_STARTED
from agi.memory import Memory
from agi.runtime import Budget, Runtime, Session


class FakeAgent:
    """Minimal stand-in for agi.agent.Agent that drives Session.

    Has the same attribute surface Session/Runtime touch: model, usage,
    bus, budget, last_critic_score, extra_system, snapshot/restore, chat.
    """

    def __init__(
        self,
        *,
        memory=None,
        model: str = "claude-haiku-4-5",
        bus=None,
        budget=None,
        extra_system=None,
        reply: str = "ok",
        in_tokens: int = 100,
        out_tokens: int = 50,
        verbose: bool = False,
        **_: object,
    ) -> None:
        self.memory = memory
        self.model = model
        self.bus = bus
        self.budget = budget
        self.extra_system = extra_system
        self.usage = Usage()
        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self.verbose = verbose
        self._reply = reply
        self._in = in_tokens
        self._out = out_tokens

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        if self.bus is not None:
            self.bus.emit(TURN_STARTED, {"prompt": user_input})
        self.messages.append({"role": "user", "content": user_input})
        self.usage.input_tokens += self._in
        self.usage.output_tokens += self._out
        self.usage.turns += 1
        self.messages.append({"role": "assistant", "content": self._reply})
        if self.bus is not None:
            # Emit per-turn deltas, mirroring the real Agent's TURN_COMPLETED.
            from agi.costs import Usage as _U

            turn_u = _U(input_tokens=self._in, output_tokens=self._out)
            self.bus.emit(
                TURN_COMPLETED,
                {
                    "text": self._reply,
                    "cost_usd": turn_u.cost_usd(self.model),
                    "input_tokens": self._in,
                    "output_tokens": self._out,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            )
        return self._reply

    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "messages": list(self.messages),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "turns": self.usage.turns,
            },
            "extra_system": self.extra_system,
        }

    def restore(self, snap: dict) -> None:
        self.messages = list(snap.get("messages", []))
        u = snap.get("usage", {})
        self.usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
            turns=u.get("turns", 0),
        )
        self.extra_system = snap.get("extra_system", self.extra_system)


class TestEventBus(unittest.TestCase):
    def test_emit_and_replay(self):
        bus = EventBus()
        bus.session_id = "s1"
        e1 = bus.emit("a", {"x": 1})
        e2 = bus.emit("b", {"y": 2})
        self.assertEqual(e1.seq, 1)
        self.assertEqual(e2.seq, 2)
        self.assertEqual(e1.session_id, "s1")
        replayed = bus.replay()
        self.assertEqual([e.type for e in replayed], ["a", "b"])

    def test_replay_since(self):
        bus = EventBus()
        bus.emit("a")
        bus.emit("b")
        bus.emit("c")
        replayed = bus.replay(since_seq=1)
        self.assertEqual([e.type for e in replayed], ["b", "c"])

    def test_subscribe_receives_live_events(self):
        bus = EventBus()
        received: list[str] = []
        unsub = bus.subscribe(lambda e: received.append(e.type))
        bus.emit("x")
        bus.emit("y")
        unsub()
        bus.emit("z")
        self.assertEqual(received, ["x", "y"])

    def test_handler_exception_does_not_break_emit(self):
        bus = EventBus()
        bus.subscribe(lambda e: 1 / 0)
        # Should not raise
        bus.emit("a")
        self.assertEqual(len(bus.replay()), 1)

    def test_buffer_caps(self):
        bus = EventBus(buffer_size=3)
        for i in range(10):
            bus.emit("e", {"i": i})
        tail = bus.replay()
        self.assertEqual(len(tail), 3)
        self.assertEqual([e.data["i"] for e in tail], [7, 8, 9])


class TestBudget(unittest.TestCase):
    def test_no_limits(self):
        b = Budget()
        u = Usage(input_tokens=10**9, output_tokens=10**9, turns=1000)
        self.assertIsNone(b.violation(u, "claude-opus-4-7"))

    def test_max_usd_triggered(self):
        b = Budget(max_usd=0.001)
        u = Usage(input_tokens=1_000_000)  # $5 on opus, well over
        self.assertIn("max_usd", b.violation(u, "claude-opus-4-7"))

    def test_max_turns_triggered(self):
        b = Budget(max_turns=2)
        u = Usage(turns=2)
        self.assertIn("max_turns", b.violation(u, "claude-opus-4-7"))

    def test_round_trip_dict(self):
        b = Budget(max_usd=0.5, max_turns=10)
        b2 = Budget.from_dict(b.to_dict())
        self.assertEqual(b.to_dict(), b2.to_dict())

    def test_from_dict_none(self):
        b = Budget.from_dict(None)
        self.assertEqual(b.to_dict(), Budget().to_dict())


class TestSession(unittest.TestCase):
    def test_step_runs_chat_and_returns_result(self):
        agent = FakeAgent(reply="hello world")
        session = Session(agent)
        result = session.step("hi")
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.input_tokens, 100)
        self.assertFalse(result.budget_exceeded)
        self.assertIsNone(result.error)

    def test_step_emits_session_created(self):
        agent = FakeAgent()
        bus_events: list[str] = []
        agent.bus = EventBus()
        agent.bus.subscribe(lambda e: bus_events.append(e.type))
        Session(agent)
        self.assertIn(SESSION_CREATED, bus_events)

    def test_session_info_reflects_state(self):
        agent = FakeAgent(reply="r", in_tokens=50, out_tokens=20)
        session = Session(agent, role="planner")
        info = session.info()
        self.assertEqual(info.role, "planner")
        self.assertEqual(info.turns, 0)
        session.step("x")
        info2 = session.info()
        self.assertEqual(info2.turns, 1)

    def test_budget_blocks_step_after_exhaustion(self):
        # Tight ceiling: one step blows it.
        agent = FakeAgent(in_tokens=1_000_000, out_tokens=0, model="claude-opus-4-7")
        session = Session(agent, budget=Budget(max_usd=0.10))
        r1 = session.step("first")
        # First step: pre-flight check passes (usage was 0), but post-step
        # the session is exhausted.
        self.assertTrue(r1.budget_exceeded)
        r2 = session.step("second")
        # Second step: rejected pre-flight.
        self.assertTrue(r2.budget_exceeded)
        self.assertEqual(r2.input_tokens, 0)  # never ran
        self.assertIn("max_usd", r2.error or "")

    def test_snapshot_round_trip(self):
        agent = FakeAgent(reply="resp")
        session = Session(agent, role="executor")
        session.step("first")
        snap = session.snapshot()
        self.assertEqual(snap["role"], "executor")
        self.assertEqual(snap["agent"]["usage"]["turns"], 1)
        # Round-trip into a fresh Session.
        agent2 = FakeAgent()
        agent2.restore(snap["agent"])
        self.assertEqual(agent2.usage.turns, 1)
        self.assertEqual(agent2.usage.input_tokens, 100)

    def test_step_serializes_concurrent_calls(self):
        # FakeAgent.chat is fast; verify two threads don't deadlock and
        # both complete; lock guarantees serialization, not order.
        agent = FakeAgent(reply="ok")
        session = Session(agent)
        results: list[str] = []
        errors: list[str] = []

        def worker(tag: str) -> None:
            try:
                results.append(session.step(tag).text)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        self.assertEqual(agent.usage.turns, 3)


class _FakeRuntime(Runtime):
    """Runtime that uses FakeAgent so we never call the API."""

    def __init__(self, **kwargs):
        super().__init__(agent_factory=FakeAgent, **kwargs)


class TestRuntime(unittest.TestCase):
    def test_create_and_step_and_close(self):
        rt = _FakeRuntime()
        s = rt.create_session(budget=Budget(max_turns=3))
        r = s.step("hello")
        self.assertEqual(r.text, "ok")
        self.assertEqual(len(rt.list_sessions()), 1)
        self.assertEqual(rt.list_sessions()[0].id, s.id)
        self.assertTrue(rt.close(s.id))
        self.assertEqual(len(rt.list_sessions()), 0)

    def test_capabilities_lists_tools(self):
        rt = _FakeRuntime()
        caps = rt.capabilities().to_dict()
        self.assertIn("read_file", caps["tools"])
        self.assertIn("save_memory", caps["tools"])
        self.assertIn("web_search", caps["server_tools"])
        self.assertTrue(caps["snapshots"])

    def test_restore_round_trip(self):
        rt = _FakeRuntime()
        s = rt.create_session(role="executor")
        s.step("hi")
        snap = s.snapshot()
        rt.close(s.id)

        rt2 = _FakeRuntime()
        s2 = rt2.restore_session(snap)
        self.assertEqual(s2.id, s.id)
        self.assertEqual(s2.role, "executor")
        self.assertEqual(s2.agent.usage.turns, 1)

    def test_get_unknown_session(self):
        rt = _FakeRuntime()
        self.assertIsNone(rt.get("does-not-exist"))

    def test_close_unknown_session(self):
        rt = _FakeRuntime()
        self.assertFalse(rt.close("nope"))

    def test_capabilities_includes_skills_when_library_present(self):
        from learner.skills import SkillLibrary

        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=Path(tmp))
            lib.write("greet", "user says hi", "Say hello back.")
            rt = Runtime(agent_factory=FakeAgent, skill_library=lib)
            caps = rt.capabilities().to_dict()
            self.assertIn("greet", caps["skills"])

    def test_runtime_threads_skill_addendum_into_extra_system(self):
        from learner.skills import SkillLibrary

        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=Path(tmp))
            lib.write("git-recovery", "rebase has gone wrong", "1. git reflog. 2. ...")
            rt = Runtime(agent_factory=FakeAgent, skill_library=lib)
            s = rt.create_session()
            self.assertIn("git-recovery", s.agent.extra_system or "")


if __name__ == "__main__":
    unittest.main()
