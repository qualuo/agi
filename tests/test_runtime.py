"""Tests for the Runtime layer.

The Runtime wraps Agent. To keep the tests offline, we substitute a fake
agent that mimics the relevant attributes of `agi.agent.Agent` without
making any API calls.
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
from agi.memory import Memory
from agi.runtime import Runtime, TurnResult


class FakeAgent:
    """Stand-in for agi.agent.Agent for offline testing.

    Tracks usage as a real-looking `Usage` object, exposes `last_critic_score`
    and `last_skills_used`, and implements `chat()` deterministically.
    """

    def __init__(self, model: str = "claude-opus-4-7", reply: str = "ok") -> None:
        self.model = model
        self.max_tokens = 16000
        self.effort = "high"
        self.usage = Usage()
        self.last_critic_score: float | None = None
        self.last_skills_used: list[str] = []
        self.tool_schemas: list[dict] = [
            {"name": "read_file"},
            {"name": "write_file"},
            {"type": "web_search_20260209", "name": "web_search"},
        ]
        self.skills = None
        self._reply = reply
        self.chat_calls: list[str] = []
        self.memory = Memory(path=Path(tempfile.gettempdir()) / "agi-fake-memory.jsonl")

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.chat_calls.append(user_input)
        # Mimic API-side accounting so the runtime's per-turn diff is non-zero.
        self.usage.input_tokens += 100
        self.usage.output_tokens += 50
        self.usage.turns += 1
        if user_input.startswith("BOOM"):
            raise RuntimeError("simulated failure")
        return f"{self._reply}: {user_input}"

    def reset(self) -> None:
        self.usage = Usage()


class TestRuntime(unittest.TestCase):
    def _runtime_with_fake(self, **agent_kwargs):
        def factory(_sid: str) -> FakeAgent:
            return FakeAgent(**agent_kwargs)
        return Runtime(agent_factory=factory)

    def test_create_and_list_session(self):
        rt = self._runtime_with_fake()
        sid = rt.create_session()
        sessions = rt.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].id, sid)
        self.assertEqual(sessions[0].turn_count, 0)

    def test_create_with_explicit_id(self):
        rt = self._runtime_with_fake()
        sid = rt.create_session("custom-id")
        self.assertEqual(sid, "custom-id")
        info = rt.session_info("custom-id")
        self.assertEqual(info.id, "custom-id")

    def test_duplicate_session_id_rejected(self):
        rt = self._runtime_with_fake()
        rt.create_session("x")
        with self.assertRaises(ValueError):
            rt.create_session("x")

    def test_turn_returns_structured_result(self):
        rt = self._runtime_with_fake(reply="hello")
        sid = rt.create_session()
        result = rt.turn(sid, "hi there")
        self.assertIsInstance(result, TurnResult)
        self.assertEqual(result.text, "hello: hi there")
        self.assertEqual(result.finish_reason, "ok")
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertEqual(result.usage["output_tokens"], 50)
        self.assertGreater(result.cost_usd, 0)
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        self.assertIsNone(result.error)

    def test_turn_captures_errors(self):
        rt = self._runtime_with_fake()
        sid = rt.create_session()
        result = rt.turn(sid, "BOOM please")
        self.assertEqual(result.finish_reason, "error")
        self.assertEqual(result.text, "")
        self.assertIn("simulated failure", result.error or "")

    def test_session_info_accumulates(self):
        rt = self._runtime_with_fake()
        sid = rt.create_session()
        rt.turn(sid, "first")
        rt.turn(sid, "second")
        info = rt.session_info(sid)
        self.assertEqual(info.turn_count, 2)
        self.assertEqual(info.cumulative_usage["input_tokens"], 200)

    def test_reset_clears_history_but_keeps_session(self):
        rt = self._runtime_with_fake()
        sid = rt.create_session()
        rt.turn(sid, "first")
        rt.reset_session(sid)
        info = rt.session_info(sid)
        self.assertEqual(info.turn_count, 0)
        self.assertEqual(info.cumulative_usage["input_tokens"], 0)

    def test_delete_session_removes_it(self):
        rt = self._runtime_with_fake()
        sid = rt.create_session()
        rt.delete_session(sid)
        with self.assertRaises(KeyError):
            rt.session_info(sid)

    def test_unknown_session_raises_keyerror(self):
        rt = self._runtime_with_fake()
        with self.assertRaises(KeyError):
            rt.turn("nope", "hi")

    def test_describe_includes_tools_and_pricing(self):
        rt = self._runtime_with_fake()
        d = rt.describe()
        self.assertEqual(d["model"], "claude-opus-4-7")
        tool_names = {t["name"] for t in d["tools"]}
        self.assertIn("read_file", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertIn("claude-opus-4-7", d["pricing"])

    def test_concurrent_turns_on_different_sessions(self):
        rt = self._runtime_with_fake()
        sid_a = rt.create_session("a")
        sid_b = rt.create_session("b")
        results: dict[str, TurnResult] = {}

        def run(sid: str, prompt: str) -> None:
            results[sid] = rt.turn(sid, prompt)

        threads = [
            threading.Thread(target=run, args=(sid_a, "for-a")),
            threading.Thread(target=run, args=(sid_b, "for-b")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(results["a"].finish_reason, "ok")
        self.assertEqual(results["b"].finish_reason, "ok")
        self.assertIn("for-a", results["a"].text)
        self.assertIn("for-b", results["b"].text)


if __name__ == "__main__":
    unittest.main()
