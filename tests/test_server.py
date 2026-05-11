"""HTTP server tests — exercises the wire surface with a fake agent.

Uses urllib (stdlib) to keep the test runner dep-free. The server is
started on an ephemeral port in a background thread, then shut down in
tearDown.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.runtime import Run, Runtime
from agi.server import serve


class _Usage:
    input_tokens = 10
    output_tokens = 20
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return 0.001


class FastAgent:
    def __init__(self, run: Run, output: str = "ok") -> None:
        self.run = run
        self.output = output
        self.usage = _Usage()
        self.model = "claude-opus-4-7"

    def chat(self, task: str) -> str:
        return f"{self.output}: {task}"


def _fast_factory(run: Run, runtime: Runtime) -> FastAgent:
    return FastAgent(run)


class TestServer(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(agent_factory=_fast_factory)
        self.httpd = serve(runtime=self.runtime, host="127.0.0.1", port=0)
        self.host, self.port = self.httpd.server_address
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=2.0)
        self.httpd.server_close()

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def _request(self, path: str, *, method: str = "GET", body: dict | None = None, timeout: float = 5.0):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._url(path), data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_healthz(self):
        status, body = self._request("/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_submit_then_get(self):
        status, body = self._request("/v1/runs", method="POST", body={"task": "do a thing"})
        self.assertEqual(status, 201)
        self.assertEqual(body["task"], "do a thing")
        run_id = body["id"]

        # Poll until terminal (should be near-instant).
        run = self.runtime.get(run_id)
        assert run is not None
        run.wait(timeout=2.0)

        status, body = self._request(f"/v1/runs/{run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "succeeded")
        self.assertIn("ok: do a thing", body["result"])

    def test_submit_missing_task(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._request("/v1/runs", method="POST", body={})
        self.assertEqual(ctx.exception.code, 400)

    def test_list_runs(self):
        self._request("/v1/runs", method="POST", body={"task": "a"})
        self._request("/v1/runs", method="POST", body={"task": "b"})
        status, body = self._request("/v1/runs")
        self.assertEqual(status, 200)
        self.assertEqual(len(body), 2)

    def test_cancel(self):
        # Use a slow agent for this test only.
        class SlowAgent(FastAgent):
            def chat(self, task: str) -> str:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if self.run._cancel.is_set():
                        from agi.runtime import Cancelled

                        raise Cancelled("cancelled")
                    time.sleep(0.01)
                return "should not reach"

        self.runtime._agent_factory = lambda run, runtime: SlowAgent(run)
        status, body = self._request("/v1/runs", method="POST", body={"task": "slow"})
        run_id = body["id"]
        time.sleep(0.05)
        status, body = self._request(f"/v1/runs/{run_id}/cancel", method="POST")
        self.assertEqual(status, 200)
        self.assertTrue(body["cancelled"])

        run = self.runtime.get(run_id)
        assert run is not None
        run.wait(timeout=2.0)
        self.assertEqual(run.status.value, "cancelled")

    def test_get_unknown(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._request("/v1/runs/does-not-exist")
        self.assertEqual(ctx.exception.code, 404)

    def test_events_sse(self):
        # Submit, wait for terminal, then read back the event stream.
        status, body = self._request("/v1/runs", method="POST", body={"task": "x"})
        run_id = body["id"]
        run = self.runtime.get(run_id)
        assert run is not None
        run.wait(timeout=2.0)

        req = urllib.request.Request(self._url(f"/v1/runs/{run_id}/events"))
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.headers.get("Content-Type"), "text/event-stream")
            raw = resp.read().decode("utf-8")
        # SSE format: 'event: <type>\ndata: <json>\n\n'
        self.assertIn("event: run.started", raw)
        self.assertIn("event: run.succeeded", raw)


if __name__ == "__main__":
    unittest.main()
