"""Runtime tests.

These tests avoid hitting the real Anthropic API by injecting a fake
agent_factory. They exercise: event ordering, cancellation, subrun
propagation, and the runtime's bookkeeping.
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
from agi.events import (
    Event, make, DONE, USAGE, RUN_STARTED, TEXT, TOOL_CALL, TOOL_RESULT,
    CANCELLED, ERROR, SUBRUN_STARTED, SUBRUN_COMPLETED, SKILLS_LOADED,
)
from agi.runtime import Runtime, RunRequest


class FakeAgent:
    """A scripted agent for runtime tests. Emits canned events; returns text."""

    def __init__(self, runtime, handle, *, text: str = "ok",
                 cost: float = 0.0, emit_tool: bool = False, sleep_ms: int = 0,
                 fail: bool = False):
        self.runtime = runtime
        self.handle = handle
        self.run_id = handle.run_id
        self.text = text
        self.cost = cost
        self.emit_tool = emit_tool
        self.sleep_ms = sleep_ms
        self.fail = fail
        self.last_critic_score = None
        self.model = "fake-model"
        self.usage = Usage(input_tokens=10, output_tokens=20)

    def chat(self, task: str, max_iterations: int = 25) -> str:
        if self.fail:
            raise RuntimeError("scripted failure")
        if self.emit_tool:
            self.handle.emit(make(TOOL_CALL, self.run_id, name="ping", input={}, id="t1"))
            self.handle.emit(make(TOOL_RESULT, self.run_id, id="t1", content="pong", is_error=False))
        if self.sleep_ms:
            # Sleep in small steps so cancellation can take effect.
            elapsed = 0
            while elapsed < self.sleep_ms:
                if self.handle.is_cancelled():
                    return ""
                time.sleep(0.01)
                elapsed += 10
        self.handle.emit(make(TEXT, self.run_id, text=self.text))
        return self.text


def factory_returning(text: str = "ok", **kw):
    def _factory(runtime, handle):
        return FakeAgent(runtime, handle, text=text, **kw)
    return _factory


class TestRuntimeBasic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_submit_and_complete(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning("hello"))
        run = rt.submit(RunRequest(task="say hello"))
        events = list(run.events(timeout=5))
        run.wait(timeout=5)
        types = [e.type for e in events]
        self.assertEqual(types[0], RUN_STARTED)
        self.assertIn(TEXT, types)
        self.assertIn(USAGE, types)
        self.assertEqual(types[-1], DONE)
        self.assertEqual(run.result.text, "hello")
        self.assertIsNone(run.result.error)

    def test_event_run_id_matches_handle(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning("x"))
        run = rt.submit(RunRequest(task="anything"))
        run.wait(timeout=5)
        for evt in run.replay():
            self.assertEqual(evt.run_id, run.run_id)

    def test_tool_events(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning(emit_tool=True))
        run = rt.submit(RunRequest(task="tool me"))
        run.wait(timeout=5)
        types = [e.type for e in run.replay()]
        self.assertIn(TOOL_CALL, types)
        self.assertIn(TOOL_RESULT, types)

    def test_error_propagation(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning(fail=True))
        run = rt.submit(RunRequest(task="boom"))
        run.wait(timeout=5)
        types = [e.type for e in run.replay()]
        self.assertEqual(types[-1], ERROR)
        self.assertIsNotNone(run.result.error)

    def test_cancellation(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning(sleep_ms=2000))
        run = rt.submit(RunRequest(task="slow"))
        time.sleep(0.05)
        run.cancel()
        run.wait(timeout=5)
        self.assertTrue(run.result.cancelled)
        types = [e.type for e in run.replay()]
        self.assertIn(CANCELLED, types)

    def test_list_runs_includes_submitted(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning("done"))
        run = rt.submit(RunRequest(task="hi"))
        run.wait(timeout=5)
        rows = rt.list_runs()
        ids = {r["run_id"] for r in rows}
        self.assertIn(run.run_id, ids)

    def test_skills_event_when_match(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning("ok"))
        rt.skills.add("Greet", "Greet the user warmly.", "Say hi.", tags=["greet"])
        run = rt.submit(RunRequest(task="greet the user"))
        run.wait(timeout=5)
        types = [e.type for e in run.replay()]
        self.assertIn(SKILLS_LOADED, types)


class TestSubruns(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_submit_child_emits_subrun_events(self):
        rt = Runtime(root_dir=self.root, agent_factory=factory_returning("parent"))
        parent = rt.submit(RunRequest(task="parent task"))
        # Parent thread is already running; submit a child via the runtime API.
        # Wait until parent is mid-run so events queue up properly.
        time.sleep(0.05)
        child = rt.submit_child(parent, RunRequest(task="child task"))
        child.wait(timeout=5)
        parent.wait(timeout=5)
        types = [e.type for e in parent.replay()]
        self.assertIn(SUBRUN_STARTED, types)
        self.assertIn(SUBRUN_COMPLETED, types)


if __name__ == "__main__":
    unittest.main()
