"""Tests for AgentRuntime that don't hit the Anthropic API.

We exercise the session lifecycle, snapshot/restore, tool interception, and
the event-conversion helpers directly. Anything that requires a real
streaming response is tested via the Coordinator-level fake (see
test_coordinator.py)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Dummy API key so anthropic.Anthropic() can be constructed without a real one.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import TextDelta, ThinkingDelta, ToolResult, ToolUseRequested
from agi.memory import Memory
from agi.runtime import AgentRuntime, SessionConfig, _Session


class _FakeAgent:
    """Stand-in for agi.Agent that doesn't open a network client."""

    def __init__(self):
        self.client = None
        self.memory = Memory(path=Path(tempfile.mkdtemp()) / "m.jsonl")
        self.model = "claude-opus-4-7"
        self.max_tokens = 1000
        self.effort = "high"
        self.messages: list[dict] = []
        self.tool_schemas: list[dict] = []
        self.handlers: dict = {
            "echo": lambda text: f"echoed:{text}",
            "boom": lambda: (_ for _ in ()).throw(ValueError("kaboom")),
        }
        from agi.costs import Usage
        self.usage = Usage()
        self.tracer = None
        self.critic = None
        self.critic_threshold = 0.5
        self.last_critic_score = None

    def _apply_critic_gate(self, prompt, response):
        return response, None


def _make_runtime_with_fake_session(interceptor=None):
    rt = AgentRuntime()
    sid = "test-sess"
    cfg = SessionConfig()
    rt.sessions[sid] = _Session(
        id=sid, agent=_FakeAgent(), config=cfg, interceptor=interceptor
    )
    return rt, sid


class TestSessionLifecycle(unittest.TestCase):
    def test_start_and_close(self):
        rt = AgentRuntime()
        # Use a fake agent so we don't touch the API
        sid = rt.start_session(agent=_FakeAgent())
        self.assertIn(sid, rt.sessions)
        rt.close_session(sid)
        self.assertNotIn(sid, rt.sessions)

    def test_duplicate_session_id_raises(self):
        rt = AgentRuntime()
        rt.start_session(session_id="dup", agent=_FakeAgent())
        with self.assertRaises(ValueError):
            rt.start_session(session_id="dup", agent=_FakeAgent())

    def test_get_session_missing_raises(self):
        rt = AgentRuntime()
        with self.assertRaises(KeyError):
            rt.get_session("nope")

    def test_get_session_closed_raises(self):
        rt = AgentRuntime()
        sid = rt.start_session(agent=_FakeAgent())
        rt.sessions[sid].closed = True
        with self.assertRaises(RuntimeError):
            rt.get_session(sid)


class TestEventConversion(unittest.TestCase):
    def test_text_delta(self):
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        out: list = []
        ev = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hello"),
        )
        rt._convert_stream_event(s, ev, out)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], TextDelta)
        self.assertEqual(out[0].text, "hello")
        self.assertEqual(out[0].session_id, sid)

    def test_thinking_delta(self):
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        out: list = []
        ev = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="hmm"),
        )
        rt._convert_stream_event(s, ev, out)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], ThinkingDelta)
        self.assertEqual(out[0].text, "hmm")

    def test_irrelevant_events_ignored(self):
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        out: list = []
        for ev in [
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
            SimpleNamespace(type="message_start"),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="input_json_delta")),
        ]:
            rt._convert_stream_event(s, ev, out)
        self.assertEqual(out, [])


def _make_tool_use_block(name, tool_input, block_id="t1"):
    return SimpleNamespace(
        type="tool_use", name=name, input=tool_input, id=block_id
    )


class TestToolDispatch(unittest.TestCase):
    def test_normal_dispatch(self):
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        content = [_make_tool_use_block("echo", {"text": "hi"})]
        results, events = rt._handle_tools(s, content)
        # Two events: requested, result
        kinds = [e.kind for e in events]
        self.assertEqual(kinds, ["tool_use_requested", "tool_result"])
        req, res = events
        self.assertIsInstance(req, ToolUseRequested)
        self.assertIsInstance(res, ToolResult)
        self.assertEqual(res.output, "echoed:hi")
        self.assertFalse(res.is_error)
        self.assertFalse(res.intercepted)
        # And one tool_result message for the model
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "echoed:hi")

    def test_handler_exception_becomes_error_result(self):
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        content = [_make_tool_use_block("boom", {})]
        results, events = rt._handle_tools(s, content)
        res = events[-1]
        self.assertTrue(res.is_error)
        self.assertIn("kaboom", res.output)
        self.assertTrue(results[0]["is_error"])

    def test_interceptor_replaces_result(self):
        def intercept(session_id, name, inp):
            return "MOCKED"

        rt, sid = _make_runtime_with_fake_session(interceptor=intercept)
        s = rt.sessions[sid]
        content = [_make_tool_use_block("echo", {"text": "hi"})]
        _, events = rt._handle_tools(s, content)
        res = events[-1]
        self.assertEqual(res.output, "MOCKED")
        self.assertTrue(res.intercepted)
        self.assertFalse(res.is_error)

    def test_interceptor_returning_none_falls_through(self):
        def intercept(session_id, name, inp):
            return None  # don't intercept

        rt, sid = _make_runtime_with_fake_session(interceptor=intercept)
        s = rt.sessions[sid]
        content = [_make_tool_use_block("echo", {"text": "hi"})]
        _, events = rt._handle_tools(s, content)
        res = events[-1]
        self.assertEqual(res.output, "echoed:hi")
        self.assertFalse(res.intercepted)

    def test_interceptor_raises_becomes_intercepted_error(self):
        def intercept(session_id, name, inp):
            raise PermissionError("denied")

        rt, sid = _make_runtime_with_fake_session(interceptor=intercept)
        s = rt.sessions[sid]
        content = [_make_tool_use_block("echo", {"text": "hi"})]
        _, events = rt._handle_tools(s, content)
        res = events[-1]
        self.assertTrue(res.intercepted)
        self.assertTrue(res.is_error)
        self.assertIn("denied", res.output)

    def test_unknown_tool_skipped(self):
        # Server-side tools (web_search/web_fetch) end up here and should be
        # dropped without emitting events.
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        content = [_make_tool_use_block("web_search", {"query": "x"})]
        results, events = rt._handle_tools(s, content)
        self.assertEqual(events, [])
        self.assertEqual(results, [])


class TestSnapshotRestore(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        rt, sid = _make_runtime_with_fake_session()
        s = rt.sessions[sid]
        s.agent.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        s.agent.usage.input_tokens = 100
        s.agent.usage.output_tokens = 50
        s.seq = 7

        snap = rt.snapshot(sid)
        self.assertEqual(snap["id"], sid)
        self.assertEqual(snap["seq"], 7)
        self.assertEqual(snap["usage"]["input_tokens"], 100)
        self.assertEqual(len(snap["messages"]), 2)

        # Restore into a different runtime
        rt2 = AgentRuntime()
        # restore() rebuilds a real Agent. To avoid hitting the API, override
        # the constructed Agent by deleting and reusing the snapshot path —
        # restore() doesn't open the network until used, so just check state.
        sid2 = rt2.restore(snap)
        self.assertEqual(sid2, sid)
        s2 = rt2.sessions[sid]
        self.assertEqual(s2.agent.usage.input_tokens, 100)
        self.assertEqual(len(s2.agent.messages), 2)
        self.assertEqual(s2.seq, 7)

    def test_save_and_load_snapshot_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = AgentRuntime(sessions_dir=tmp)
            # Insert a fake-agent session directly
            cfg = SessionConfig()
            rt.sessions["disk-test"] = _Session(
                id="disk-test", agent=_FakeAgent(), config=cfg
            )
            rt.sessions["disk-test"].agent.messages = [
                {"role": "user", "content": "saved"}
            ]
            path = rt.save_snapshot("disk-test")
            self.assertTrue(path.exists())

            rt2 = AgentRuntime(sessions_dir=tmp)
            sid = rt2.load_snapshot("disk-test")
            self.assertEqual(sid, "disk-test")
            self.assertEqual(
                rt2.sessions["disk-test"].agent.messages[0]["content"], "saved"
            )


if __name__ == "__main__":
    unittest.main()
