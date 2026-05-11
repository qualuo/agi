"""Tests for the runtime engine.

All tests use the mock agent backend so no API key or network is required.
They cover: budgets, sessions, jobs (sync + async + cancel + streaming),
metrics, capability manifest, HTTP server end-to-end via the Python client.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.costs import Usage
from runtime import Budget, BudgetError, JobState, Runtime
from runtime.capabilities import build_manifest
from runtime.client import Client, ClientError
from runtime.mock_agent import MockAgent
from runtime.server import serve_in_thread


def _mock_factory(**kwargs):
    kwargs.setdefault("model", "mock-1")
    return MockAgent(**kwargs)


class TestBudget(unittest.TestCase):
    def test_below_caps_passes(self):
        b = Budget(max_input_tokens=1000, max_output_tokens=1000, max_usd=10.0)
        u = Usage(input_tokens=100, output_tokens=100)
        b.check(u, "claude-opus-4-7", jobs_run=0)  # should not raise

    def test_input_cap_blocks(self):
        b = Budget(max_input_tokens=50)
        u = Usage(input_tokens=100)
        with self.assertRaises(BudgetError):
            b.check(u, "claude-opus-4-7", jobs_run=0)

    def test_total_cap_blocks(self):
        b = Budget(max_total_tokens=150)
        u = Usage(input_tokens=100, output_tokens=100)
        with self.assertRaises(BudgetError):
            b.check(u, "claude-opus-4-7", jobs_run=0)

    def test_usd_cap_blocks(self):
        b = Budget(max_usd=0.001)
        u = Usage(input_tokens=1_000_000)  # = $5
        with self.assertRaises(BudgetError):
            b.check(u, "claude-opus-4-7", jobs_run=0)

    def test_turns_and_jobs_caps(self):
        b = Budget(max_turns=2, max_jobs=3)
        b.check(Usage(turns=1), "claude-opus-4-7", jobs_run=2)  # ok
        with self.assertRaises(BudgetError):
            b.check(Usage(turns=2), "claude-opus-4-7", jobs_run=0)
        with self.assertRaises(BudgetError):
            b.check(Usage(turns=0), "claude-opus-4-7", jobs_run=3)

    def test_roundtrip_dict(self):
        b = Budget(max_usd=1.0, max_turns=10)
        d = b.to_dict()
        b2 = Budget.from_dict(d)
        self.assertEqual(b2.max_usd, 1.0)
        self.assertEqual(b2.max_turns, 10)
        self.assertIsNone(b2.max_input_tokens)


class TestRuntimeInProcess(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rt = Runtime(agent_factory=_mock_factory, root=self._tmp.name, max_workers=4)

    def tearDown(self):
        self.rt.shutdown()
        self._tmp.cleanup()

    def test_create_and_get_session(self):
        s = self.rt.create_session()
        info = self.rt.sessions.get(s.id).info()
        self.assertEqual(info.id, s.id)
        self.assertEqual(info.model, "mock-1")
        self.assertFalse(info.closed)

    def test_sync_chat_increments_usage(self):
        s = self.rt.create_session()
        result = self.rt.chat_sync(s.id, "2 + 2")
        self.assertEqual(result["text"], "4")
        self.assertGreaterEqual(result["usage_delta"]["output_tokens"], 1)
        self.assertEqual(result["session"]["turns"], 1)

    def test_isolated_sessions_have_isolated_memory(self):
        s1 = self.rt.create_session()
        s2 = self.rt.create_session()
        s1.agent.memory.save("only-s1", tags=["t"])
        self.assertEqual(len(s1.agent.memory.search("only-s1")), 1)
        self.assertEqual(len(s2.agent.memory.search("only-s1")), 0)

    def test_budget_blocks_after_first_turn(self):
        s = self.rt.create_session(budget=Budget(max_turns=1))
        self.rt.chat_sync(s.id, "hi")
        with self.assertRaises(BudgetError):
            self.rt.chat_sync(s.id, "hi again")

    def test_async_job_runs_to_completion(self):
        s = self.rt.create_session()
        job = self.rt.submit_job(s.id, "3 * 4")
        done = self.rt.wait(job.id, timeout=5.0)
        self.assertIsNotNone(done)
        self.assertEqual(done.state, JobState.SUCCEEDED)
        self.assertEqual(done.result_text, "12")
        kinds = [e.kind for e in done._events]
        self.assertIn("text_delta", kinds)
        self.assertIn("done", kinds)

    def test_async_job_cancel_before_run(self):
        rt = Runtime(
            agent_factory=lambda **kw: MockAgent(delay_seconds=0.2, **{k: v for k, v in kw.items() if k != "delay_seconds"}),
            root=self._tmp.name,
            max_workers=1,
        )
        try:
            s = rt.create_session()
            blocker = rt.submit_job(s.id, "blocking-prompt")
            # Saturate the pool with the blocker, then submit + cancel a second job
            second = rt.submit_job(s.id, "2 + 2")
            rt.jobs.cancel(second.id)
            rt.wait(blocker.id, timeout=5.0)
            done = rt.wait(second.id, timeout=5.0)
            self.assertEqual(done.state, JobState.CANCELLED)
        finally:
            rt.shutdown()

    def test_metrics_record_activity(self):
        s = self.rt.create_session()
        self.rt.chat_sync(s.id, "1 + 1")
        snap = self.rt.metrics_snapshot()
        self.assertGreaterEqual(snap["counters"].get("sessions_created", 0), 1)
        self.assertGreaterEqual(snap["counters"].get("turns_total", 0), 1)
        self.assertIn("chat_sync_ms", snap["latency"])

    def test_health(self):
        h = self.rt.health()
        self.assertEqual(h["status"], "ok")
        self.assertIn("uptime_seconds", h)


class TestManifest(unittest.TestCase):
    def test_manifest_has_required_keys(self):
        m = build_manifest()
        self.assertIn("tools", m)
        self.assertIn("models", m)
        self.assertIn("features", m)
        self.assertIn("protocol", m)
        tool_names = {t["name"] for t in m["tools"]}
        self.assertIn("read_file", tool_names)
        self.assertIn("run_bash", tool_names)


class TestHttpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.rt = Runtime(agent_factory=_mock_factory, root=cls._tmp.name, max_workers=4)
        cls.server, cls.thread, cls.base = serve_in_thread(cls.rt, port=0)
        cls.client = Client(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.rt.shutdown()
        cls._tmp.cleanup()

    def test_health_endpoint(self):
        self.assertEqual(self.client.health()["status"], "ok")

    def test_capabilities_endpoint(self):
        manifest = self.client.capabilities()
        self.assertIn("tools", manifest)
        self.assertEqual(manifest["protocol"], "1.0")

    def test_full_session_lifecycle(self):
        s = self.client.create_session(budget={"max_turns": 5})
        sid = s["id"]
        self.assertIn(sid, [x["id"] for x in self.client.list_sessions()])

        out = self.client.chat(sid, "7 - 2")
        self.assertEqual(out["text"], "5")

        info = self.client.get_session(sid)
        self.assertEqual(info["turns"], 1)

        deleted = self.client.delete_session(sid)
        self.assertEqual(deleted["deleted"], sid)
        with self.assertRaises(ClientError) as ctx:
            self.client.get_session(sid)
        self.assertEqual(ctx.exception.status, 404)

    def test_async_job_and_stream(self):
        sid = self.client.create_session()["id"]
        job = self.client.submit_job(sid, "10 / 2")
        events = list(self.client.stream(job["id"]))
        kinds = [e.get("kind") for e in events]
        self.assertIn("text_delta", kinds)
        self.assertIn("done", kinds)
        final = self.client.get_job(job["id"])
        self.assertEqual(final["state"], "succeeded")
        self.assertEqual(final["result_text"], "5.0")

    def test_budget_returns_402(self):
        sid = self.client.create_session(budget={"max_turns": 1})["id"]
        self.client.chat(sid, "hi")
        with self.assertRaises(ClientError) as ctx:
            self.client.chat(sid, "hi again")
        self.assertEqual(ctx.exception.status, 402)

    def test_memory_endpoints(self):
        sid = self.client.create_session()["id"]
        self.client.save_memory(sid, "user likes broccoli", tags=["pref"])
        results = self.client.get_memory(sid, q="broccoli")
        self.assertEqual(len(results), 1)
        self.assertIn("broccoli", results[0]["text"])

    def test_metrics_endpoint(self):
        snap = self.client.metrics()
        self.assertIn("counters", snap)
        self.assertIn("uptime_seconds", snap)

    def test_missing_content_returns_400(self):
        sid = self.client.create_session()["id"]
        with self.assertRaises(ClientError) as ctx:
            self.client._req("POST", f"/v1/sessions/{sid}/messages", {})
        self.assertEqual(ctx.exception.status, 400)


class TestAuth(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rt = Runtime(agent_factory=_mock_factory, root=self._tmp.name, max_workers=2)
        from runtime.server import make_server
        self.server = make_server(self.rt, host="127.0.0.1", port=0, auth_token="secret")
        import threading
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.rt.shutdown()
        self._tmp.cleanup()

    def test_token_required(self):
        c_no = Client(self.base)
        with self.assertRaises(ClientError) as ctx:
            c_no.health()
        self.assertEqual(ctx.exception.status, 401)

    def test_token_accepted(self):
        c = Client(self.base, token="secret")
        self.assertEqual(c.health()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
