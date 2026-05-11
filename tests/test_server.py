"""Tests for the HTTP server. Round-trips via the Python client.

Uses MockBackend so no network or API key is needed.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.skills import SkillLibrary
from runtime.backend import MockBackend
from runtime.client import RuntimeClient, RuntimeError_
from runtime.engine import Engine
from runtime.server import serve_in_background


class _ServerCase(unittest.TestCase):
    backend = None
    extra_skills = ()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        backend = self.backend or MockBackend.echo("ok")
        self.lib = SkillLibrary(path=self.tmp / "skills")
        for sk in self.extra_skills:
            self.lib.add_from_text(**sk)
        self.engine = Engine(backend=backend, skill_library=self.lib)
        self.server, self.thread, self.base_url = serve_in_background(
            self.engine, skill_library=self.lib
        )
        self.client = RuntimeClient(self.base_url, timeout=5.0)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.engine.shutdown()
        self._tmp.cleanup()


class TestHealth(_ServerCase):
    def test_health(self):
        self.assertTrue(self.client.health()["ok"])

    def test_metrics_empty(self):
        m = self.client.metrics()
        self.assertEqual(m["total_tasks"], 0)


class TestTasksEndToEnd(_ServerCase):
    def test_submit_and_wait(self):
        task = self.client.submit("hello")
        # Status race: the task may already be running by the time the
        # initial snapshot was serialized. Either is acceptable.
        self.assertIn(task["status"], ("queued", "running", "completed"))
        self.assertIsNotNone(task["id"])

        final = self.client.wait(task["id"], timeout=5)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["result"], "ok")

    def test_get_events(self):
        task = self.client.submit("hello")
        self.client.wait(task["id"], timeout=5)
        events = self.client.events(task["id"])
        kinds = [e["kind"] for e in events]
        self.assertIn("status_changed", kinds)
        self.assertIn("text", kinds)

    def test_list_filtered(self):
        for _ in range(3):
            self.client.submit("anything")
        time.sleep(0.2)  # let them finish
        all_tasks = self.client.list()
        self.assertEqual(len(all_tasks), 3)
        completed = self.client.list(status="completed")
        self.assertEqual(len(completed), 3)

    def test_404_on_unknown(self):
        with self.assertRaises(RuntimeError_) as ctx:
            self.client.get("nonexistent")
        self.assertEqual(ctx.exception.status, 404)

    def test_metrics_after_run(self):
        task = self.client.submit("hi")
        self.client.wait(task["id"], timeout=5)
        m = self.client.metrics()
        self.assertEqual(m["total_tasks"], 1)
        self.assertEqual(m["task_counts"]["completed"], 1)


class TestSkills(_ServerCase):
    def test_skill_crud(self):
        added = self.client.add_skill(
            name="echo_pattern",
            when="user wants an echo",
            body="say it back",
            tags=["mock"],
        )
        self.assertEqual(added["name"], "echo_pattern")
        listed = self.client.list_skills()
        self.assertEqual(len(listed), 1)

        got = self.client.get_skill("echo_pattern")
        self.assertEqual(got["when"], "user wants an echo")

        self.client.remove_skill("echo_pattern")
        self.assertEqual(self.client.list_skills(), [])


class TestBudgetEnforcedOverHTTP(_ServerCase):
    backend = MockBackend.scripted([
        MockBackend.tool_call("run_bash", {"command": "echo a"}) for _ in range(50)
    ])

    def test_budget_failure_reflected_in_snapshot(self):
        task = self.client.submit("loop", budget={"max_turns": 2})
        final = self.client.wait(task["id"], timeout=10)
        self.assertEqual(final["status"], "failed")
        self.assertIn("turns", (final["error"] or "").lower())


class TestStreamSSE(_ServerCase):
    def test_sse_stream(self):
        task = self.client.submit("stream me")
        events = []
        # Stream returns when task hits a terminal state.
        for ev in self.client.stream(task["id"]):
            events.append(ev["kind"])
            if "status_changed" in events and "to" in ev["data"] and ev["data"]["to"] == "completed":
                break
        # Should observe at least these milestones over the stream
        self.assertIn("status_changed", events)


if __name__ == "__main__":
    unittest.main()
