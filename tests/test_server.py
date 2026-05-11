"""End-to-end tests for the HTTP runtime server.

These spin up the server on a localhost port, hit it with urllib, and
verify the full Task lifecycle: submit → poll → success. Uses the same
FakeAgent as the runtime tests so no network calls happen.
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

from agi.runtime import RuntimeEngine
from agi.server import serve
from tests.test_runtime import FakeAgent


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestServer(unittest.TestCase):
    def setUp(self):
        self.port = _free_port()
        engine = RuntimeEngine(lambda: FakeAgent(output="server says hi"))
        self.httpd = serve(host="127.0.0.1", port=self.port, engine=engine)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        # Wait a beat for the listener to be ready.
        self._wait_for_health()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _wait_for_health(self) -> None:
        for _ in range(50):
            try:
                with urllib.request.urlopen(self._url("/v1/health"), timeout=0.5) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.02)
        self.fail("server did not become healthy in time")

    def _get(self, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(self._url(path), timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def _post(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            self._url(path),
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    # ---- tests ----------------------------------------------------------

    def test_health(self):
        status, body = self._get("/v1/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_capabilities(self):
        status, body = self._get("/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(body["runtime_name"], "agi-runtime")
        self.assertIn("model", body)
        self.assertIn("default_budget", body)

    def test_submit_and_poll_to_success(self):
        status, body = self._post("/v1/tasks", {"instruction": "go"})
        self.assertEqual(status, 202)
        task_id = body["task_id"]
        # Poll a few times until terminal.
        for _ in range(50):
            status, body = self._get(f"/v1/tasks/{task_id}")
            self.assertEqual(status, 200)
            if body["status"] not in ("pending", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(body["status"], "succeeded")
        self.assertEqual(body["output"], "server says hi")

    def test_submit_validates_instruction(self):
        status, body = self._post("/v1/tasks", {"instruction": ""})
        self.assertEqual(status, 400)
        self.assertIn("instruction", body["error"])

    def test_submit_rejects_unknown_budget_field(self):
        status, body = self._post("/v1/tasks", {
            "instruction": "go", "budget": {"not_a_field": 1},
        })
        self.assertEqual(status, 400)

    def test_unknown_task_returns_404(self):
        status, body = self._get("/v1/tasks/does-not-exist")
        self.assertEqual(status, 404)

    def test_cancel_unknown_returns_404(self):
        status, body = self._post("/v1/tasks/nope/cancel")
        self.assertEqual(status, 404)

    def test_unknown_route_returns_404(self):
        status, _ = self._get("/v1/whatever")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
