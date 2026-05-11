"""Tests for the JSON-lines runtime server.

We drive the server with StringIO instead of stdin/stdout, inject a fake
Agent factory so no API calls happen, and verify the protocol contract a
coordination engine would rely on.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import events as ev
from agi.costs import Usage
from agi.runtime import Runtime
from agi.serve import JSONLinesServer


class _FakeAgent:
    def __init__(self, *, event_sink=None, model="fake", **kw):
        self.event_sink = event_sink
        self.model = model
        self.usage = Usage()
        self.last_critic_score = None
        self.handlers: dict = {}
        self.tool_schemas: list = []

    def _rid(self):
        return self.event_sink.id if self.event_sink else "local"

    def chat(self, prompt, max_iterations=25):
        class U:
            input_tokens = 10
            output_tokens = 20
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0
        self.usage.add(U())
        if self.event_sink:
            self.event_sink.emit(ev.task_started(self._rid(), prompt))
            self.event_sink.emit(ev.task_completed(self._rid(), f"echo:{prompt}", None))
        return f"echo:{prompt}"


def _run_server_with(lines: list[dict], runtime: Runtime) -> list[dict]:
    """Feed `lines` to a server, return the JSON objects it writes."""
    stdin = io.StringIO("".join(json.dumps(l) + "\n" for l in lines))
    stdout = io.StringIO()
    server = JSONLinesServer(runtime=runtime, stdin=stdin, stdout=stdout)
    server.serve_forever()
    out = stdout.getvalue().strip().splitlines()
    return [json.loads(line) for line in out]


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rt = Runtime(
            registry_dir=Path(self._tmp.name) / "runs",
            agent_factory=_FakeAgent,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_submit_returns_submitted_then_shutdown(self):
        out = _run_server_with(
            [{"cmd": "submit", "prompt": "hi"}, {"cmd": "shutdown"}],
            self.rt,
        )
        types = [o["type"] for o in out]
        self.assertIn("submitted", types)
        self.assertEqual(types[-1], "shutdown")
        submitted = next(o for o in out if o["type"] == "submitted")
        self.assertIn("run_id", submitted)

    def test_submit_with_invalid_prompt_returns_error(self):
        out = _run_server_with(
            [{"cmd": "submit", "prompt": ""}, {"cmd": "shutdown"}],
            self.rt,
        )
        errors = [o for o in out if o["type"] == "error"]
        self.assertTrue(errors)
        self.assertIn("prompt", errors[0]["message"])

    def test_unknown_command_returns_error(self):
        out = _run_server_with(
            [{"cmd": "wat"}, {"cmd": "shutdown"}],
            self.rt,
        )
        errors = [o for o in out if o["type"] == "error"]
        self.assertTrue(errors)
        self.assertIn("unknown", errors[0]["message"])

    def test_invalid_json_returns_error(self):
        stdin = io.StringIO("{not-json\n" + json.dumps({"cmd": "shutdown"}) + "\n")
        stdout = io.StringIO()
        server = JSONLinesServer(runtime=self.rt, stdin=stdin, stdout=stdout)
        server.serve_forever()
        objs = [json.loads(l) for l in stdout.getvalue().strip().splitlines()]
        self.assertTrue(any(o["type"] == "error" for o in objs))

    def test_status_for_unknown_run(self):
        out = _run_server_with(
            [{"cmd": "status", "run_id": "missing"}, {"cmd": "shutdown"}],
            self.rt,
        )
        self.assertTrue(any(o["type"] == "error" for o in out))

    def test_submit_then_status_returns_run(self):
        # Submit, wait briefly, ask for status.
        run = self.rt.submit("hi")
        run.wait(timeout=2)
        out = _run_server_with(
            [{"cmd": "status", "run_id": run.id}, {"cmd": "shutdown"}],
            self.rt,
        )
        status = next(o for o in out if o["type"] == "status")
        self.assertEqual(status["run"]["id"], run.id)
        self.assertEqual(status["run"]["status"], "completed")

    def test_list_returns_all_runs(self):
        self.rt.submit("a").wait(timeout=2)
        self.rt.submit("b").wait(timeout=2)
        out = _run_server_with([{"cmd": "list"}, {"cmd": "shutdown"}], self.rt)
        lst = next(o for o in out if o["type"] == "list")
        self.assertEqual(len(lst["runs"]), 2)

    def test_submit_with_subscribe_streams_events(self):
        # Subscribe=True forks a forwarder thread; events should appear.
        out = _run_server_with(
            [{"cmd": "submit", "prompt": "x", "subscribe": True}, {"cmd": "shutdown"}],
            self.rt,
        )
        # Give the forwarder a moment to flush before the server exits.
        time.sleep(0.1)
        types = [o.get("type") for o in out]
        self.assertIn("submitted", types)


if __name__ == "__main__":
    unittest.main()
