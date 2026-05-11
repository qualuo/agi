"""Tests for the Runtime / Session machinery.

We use a FakeAgent factory so tests run without hitting the Anthropic API.
The FakeAgent honors the same interface the Runtime expects: chat(prompt,
max_iterations) → str, plus `usage`, `last_critic_score`, optional
attach_* hooks. This is intentional: anyone integrating an alternative
backend (local model, simulator, etc.) implements this same interface.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import (
    CHAT_COMPLETED,
    CHAT_STARTED,
    ERROR,
    SESSION_CREATED,
    SESSION_ENDED,
    SKILL_LOADED,
    Event,
)
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import Skill, SkillLibrary


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None
        self.received_prompts: list[str] = []
        self._tool_synth = None
        self._delegate_fn = None
        # Configurable response
        self._response = "ok"

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.received_prompts.append(prompt)
        self.usage.input_tokens += 100
        self.usage.output_tokens += 50
        self.usage.turns += 1
        return self._response

    def attach_tool_synth(self, registry, bus=None, session_id=None):
        self._tool_synth = registry

    def attach_delegation(self, fn, bus=None, session_id=None):
        self._delegate_fn = fn

    def reset(self):
        self.usage = FakeUsage()


def _make_runtime(**overrides) -> tuple[Runtime, Path]:
    tmp = tempfile.mkdtemp()
    runtime = Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=FakeAgent,
        **overrides,
    )
    return runtime, Path(tmp)


class TestRuntimeBasics(unittest.TestCase):
    def test_create_session_emits_event(self):
        rt, _ = _make_runtime()
        events: list[Event] = []
        rt.subscribe(events.append, kind=SESSION_CREATED)
        sid = rt.create_session()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].session_id, sid)

    def test_chat_emits_lifecycle_events(self):
        rt, _ = _make_runtime()
        events: list[Event] = []
        rt.subscribe(events.append)
        sid = rt.create_session()
        rt.chat(sid, "hello")
        kinds = [e.kind for e in events]
        self.assertIn(CHAT_STARTED, kinds)
        self.assertIn(CHAT_COMPLETED, kinds)

    def test_chat_returns_final_text(self):
        rt, _ = _make_runtime()
        sid = rt.create_session()
        out = rt.chat(sid, "hi")
        self.assertEqual(out, "ok")

    def test_session_state_tracks_usage(self):
        rt, _ = _make_runtime()
        sid = rt.create_session()
        rt.chat(sid, "a")
        rt.chat(sid, "b")
        s = rt.get_session(sid).to_dict()
        self.assertEqual(s["turn_count"], 2)
        self.assertEqual(s["total_input_tokens"], 200)
        self.assertEqual(s["total_output_tokens"], 100)
        self.assertGreater(s["total_cost_usd"], 0)

    def test_end_session_emits_event_and_blocks_further_chat(self):
        rt, _ = _make_runtime()
        sid = rt.create_session()
        rt.chat(sid, "hi")
        events: list[Event] = []
        rt.subscribe(events.append, kind=SESSION_ENDED)
        rt.end_session(sid)
        self.assertEqual(len(events), 1)
        with self.assertRaises(RuntimeError):
            rt.chat(sid, "more")

    def test_unknown_session_raises(self):
        rt, _ = _make_runtime()
        with self.assertRaises(KeyError):
            rt.chat("nope", "hi")

    def test_reset_clears_usage(self):
        rt, _ = _make_runtime()
        sid = rt.create_session()
        rt.chat(sid, "x")
        rt.reset_session(sid)
        s = rt.get_session(sid).to_dict()
        self.assertEqual(s["turn_count"], 0)
        self.assertEqual(s["total_cost_usd"], 0.0)

    def test_capabilities(self):
        rt, _ = _make_runtime()
        rt.skills.save(Skill(name="cap_test", description="cap", body="b"))
        caps = rt.capabilities()
        self.assertIn("models", caps)
        self.assertIn("skills", caps)
        self.assertEqual(len(caps["skills"]), 1)
        self.assertEqual(caps["skills"][0]["name"], "cap_test")


class TestSkillInjection(unittest.TestCase):
    def test_relevant_skill_appended_to_prompt(self):
        rt, _ = _make_runtime()
        rt.skills.save(Skill(name="add_numbers", description="arithmetic addition", body="parse; add", tags=["math"]))

        events: list[Event] = []
        rt.subscribe(events.append, kind=SKILL_LOADED)

        sid = rt.create_session(SessionConfig(use_skills=True))
        rt.chat(sid, "please do addition of 2 and 2")

        # The skill should have been emitted
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["name"], "add_numbers")

        # The FakeAgent's prompt should include the skill block
        session = rt.get_session(sid)
        agent: FakeAgent = session._agent
        self.assertIn("Skill: add_numbers", agent.received_prompts[0])

    def test_disabled_skips_skill_block(self):
        rt, _ = _make_runtime()
        rt.skills.save(Skill(name="add", description="arithmetic addition", body="..."))
        sid = rt.create_session(SessionConfig(use_skills=False))
        rt.chat(sid, "addition of 2 and 2")
        agent: FakeAgent = rt.get_session(sid)._agent
        self.assertNotIn("Skill: add", agent.received_prompts[0])


class TestDelegation(unittest.TestCase):
    def test_subagent_costs_roll_up(self):
        # Custom agent factory whose chat() invokes delegation
        class DelegatingAgent(FakeAgent):
            def chat(self, prompt: str, max_iterations: int = 25) -> str:
                super().chat(prompt, max_iterations)
                if self._delegate_fn is not None and "delegate" in prompt:
                    return self._delegate_fn(task="subtask", role="executor", model=None)
                return self._response

        rt = Runtime(
            memory=Memory(path=Path(tempfile.mkdtemp()) / "m.jsonl"),
            skills=SkillLibrary(path=Path(tempfile.mkdtemp())),
            agent_factory=DelegatingAgent,
        )
        sid = rt.create_session(SessionConfig(enable_delegation=True, use_skills=False))
        out = rt.chat(sid, "please delegate the work")
        # The result is the subagent's final_text from its FakeAgent.chat()
        self.assertEqual(out, "ok")
        parent = rt.get_session(sid).to_dict()
        # Parent should account for both its own and child's tokens
        self.assertGreaterEqual(parent["total_input_tokens"], 200)


class TestEventStreamConcurrency(unittest.TestCase):
    def test_thread_safe_publish(self):
        from agi.events import Event, EventBus
        bus = EventBus()
        received: list[Event] = []
        lock = threading.Lock()

        def cb(e):
            with lock:
                received.append(e)

        bus.subscribe(cb)

        def worker(prefix: str):
            for i in range(50):
                bus.publish(Event(kind="x", session_id=f"{prefix}-{i}"))

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(received), 200)


if __name__ == "__main__":
    unittest.main()
