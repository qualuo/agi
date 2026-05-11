"""Runtime tests using FakeAgent — no API calls."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests._fakes import FakeAgent


def _patched_runtime(**kwargs):
    """Build a Runtime with FakeAgent in place of the real Agent."""
    from agi import runtime as runtime_module
    patcher = patch.object(runtime_module, "Agent", FakeAgent)
    patcher.start()
    rt = runtime_module.Runtime(**kwargs)
    return rt, patcher


class TestSessionLifecycle(unittest.TestCase):
    def setUp(self):
        self.rt, self._patcher = _patched_runtime()

    def tearDown(self):
        self._patcher.stop()

    def test_create_and_destroy(self):
        sid = self.rt.create_session()
        self.assertIn(sid, [s.session_id for s in self.rt.list_sessions()])
        self.assertTrue(self.rt.destroy_session(sid))
        self.assertFalse(self.rt.destroy_session(sid))

    def test_create_with_explicit_id(self):
        sid = self.rt.create_session(session_id="my-id")
        self.assertEqual(sid, "my-id")

    def test_duplicate_id_rejected(self):
        self.rt.create_session(session_id="dup")
        with self.assertRaises(ValueError):
            self.rt.create_session(session_id="dup")

    def test_get_unknown_session_raises(self):
        with self.assertRaises(KeyError):
            self.rt.get_session("nope")


class TestRun(unittest.TestCase):
    def setUp(self):
        self.rt, self._patcher = _patched_runtime()

    def tearDown(self):
        self._patcher.stop()

    def test_run_returns_structured_result(self):
        sid = self.rt.create_session()
        result = self.rt.run(sid, "hello world")
        self.assertEqual(result.session_id, sid)
        self.assertIn("echo:", result.output_text)
        self.assertEqual(result.usage["input_tokens"], 10)
        self.assertEqual(result.usage["output_tokens"], 20)
        self.assertFalse(result.cancelled)
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertGreaterEqual(result.cost_usd, 0)

    def test_idempotency_returns_cached_result(self):
        sid = self.rt.create_session()
        r1 = self.rt.run(sid, "hi", idempotency_key="abc")
        r2 = self.rt.run(sid, "different prompt", idempotency_key="abc")
        # The second call returns the cached result, ignoring the new prompt.
        self.assertEqual(r1.run_id, r2.run_id)
        self.assertEqual(r1.output_text, r2.output_text)

    def test_multiple_runs_accumulate_usage(self):
        sid = self.rt.create_session()
        self.rt.run(sid, "a")
        self.rt.run(sid, "b")
        info = self.rt.get_session(sid)
        self.assertEqual(info.runs, 2)
        self.assertEqual(info.cumulative_usage["input_tokens"], 20)

    def test_cancel_short_circuits_next_run(self):
        sid = self.rt.create_session()
        self.rt.cancel(sid)
        result = self.rt.run(sid, "this should not produce output")
        self.assertTrue(result.cancelled)
        self.assertEqual(result.output_text, "")


class TestCapabilities(unittest.TestCase):
    def setUp(self):
        self.rt, self._patcher = _patched_runtime()

    def tearDown(self):
        self._patcher.stop()

    def test_capabilities_manifest_shape(self):
        caps = self.rt.capabilities()
        self.assertIn("runtime_api_version", caps)
        self.assertIn("tools", caps)
        self.assertIn("events", caps)
        self.assertIn("features", caps)
        # Each declared tool has a name and kind.
        for tool in caps["tools"]:
            self.assertIn("name", tool)
            self.assertIn("kind", tool)

    def test_capabilities_lists_skills(self):
        from agi.skills import Skill, SkillLibrary
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.save(Skill(name="x", description="test skill", body="..."))
            self._patcher.stop()
            self.rt, self._patcher = _patched_runtime(skills=lib)
            caps = self.rt.capabilities()
            names = [s["name"] for s in caps["skills"]]
            self.assertIn("x", names)

    def test_health(self):
        h = self.rt.health()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["sessions"], 0)
        sid = self.rt.create_session()
        h = self.rt.health()
        self.assertEqual(h["sessions"], 1)
        self.rt.destroy_session(sid)


class TestEventStream(unittest.TestCase):
    def setUp(self):
        self.rt, self._patcher = _patched_runtime()

    def tearDown(self):
        self._patcher.stop()

    def test_subscribe_receives_events(self):
        sid = self.rt.create_session()
        q, unsubscribe = self.rt.subscribe(sid)
        self.rt.run(sid, "hi")
        kinds = []
        # Drain everything queued so far.
        import queue
        while True:
            try:
                evt = q.get(timeout=0.1)
            except queue.Empty:
                break
            kinds.append(evt.kind)
        unsubscribe()
        self.assertIn("run.started", kinds)
        self.assertIn("run.finished", kinds)

    def test_recent_events_after_run(self):
        sid = self.rt.create_session()
        self.rt.run(sid, "hi")
        events = self.rt.recent_events(sid, limit=20)
        kinds = [e.kind for e in events]
        self.assertIn("run.started", kinds)


class TestUsageRollup(unittest.TestCase):
    def setUp(self):
        self.rt, self._patcher = _patched_runtime()

    def tearDown(self):
        self._patcher.stop()

    def test_aggregate_across_sessions(self):
        for _ in range(3):
            sid = self.rt.create_session()
            self.rt.run(sid, "x")
        agg = self.rt.aggregate_usage()
        self.assertEqual(agg["runs"], 3)
        self.assertEqual(agg["input_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
