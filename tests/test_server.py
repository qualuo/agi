"""HTTP server tests against a live RuntimeServer with FakeAgent backing."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests._fakes import FakeAgent


def _http(method: str, url: str, body: dict | None = None, token: str | None = None, timeout: float = 5):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agi import runtime as runtime_module
        from agi.server import RuntimeServer
        cls._patcher = patch.object(runtime_module, "Agent", FakeAgent)
        cls._patcher.start()
        cls.runtime = runtime_module.Runtime()
        # port 0 → OS assigns a free port
        cls.server = RuntimeServer(cls.runtime, host="127.0.0.1", port=0)
        cls.server.start()
        host, port = cls.server.address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._patcher.stop()

    def test_health_endpoint(self):
        status, body = _http("GET", f"{self.base}/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIn("runtime_api_version", body)

    def test_capabilities_endpoint(self):
        status, body = _http("GET", f"{self.base}/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertIn("tools", body)
        self.assertIn("events", body)
        self.assertIn("features", body)

    def test_session_create_run_destroy(self):
        # create
        status, body = _http("POST", f"{self.base}/v1/sessions", body={})
        self.assertEqual(status, 201)
        sid = body["session_id"]

        # run
        status, body = _http("POST", f"{self.base}/v1/sessions/{sid}/run",
                             body={"prompt": "hello"})
        self.assertEqual(status, 200)
        self.assertIn("echo:", body["output_text"])
        self.assertEqual(body["usage"]["input_tokens"], 10)

        # get
        status, body = _http("GET", f"{self.base}/v1/sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertEqual(body["runs"], 1)

        # destroy
        status, body = _http("DELETE", f"{self.base}/v1/sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertTrue(body["destroyed"])

    def test_idempotency_via_http(self):
        sid = _http("POST", f"{self.base}/v1/sessions", body={})[1]["session_id"]
        _, r1 = _http("POST", f"{self.base}/v1/sessions/{sid}/run",
                      body={"prompt": "a", "idempotency_key": "K"})
        _, r2 = _http("POST", f"{self.base}/v1/sessions/{sid}/run",
                      body={"prompt": "b", "idempotency_key": "K"})
        self.assertEqual(r1["run_id"], r2["run_id"])
        # destroy
        _http("DELETE", f"{self.base}/v1/sessions/{sid}")

    def test_unknown_session_returns_404(self):
        status, _ = _http("POST", f"{self.base}/v1/sessions/missing/run",
                          body={"prompt": "x"})
        self.assertEqual(status, 404)

    def test_missing_prompt_returns_400(self):
        sid = _http("POST", f"{self.base}/v1/sessions", body={})[1]["session_id"]
        status, _ = _http("POST", f"{self.base}/v1/sessions/{sid}/run", body={})
        self.assertEqual(status, 400)
        _http("DELETE", f"{self.base}/v1/sessions/{sid}")

    def test_recent_events(self):
        sid = _http("POST", f"{self.base}/v1/sessions", body={})[1]["session_id"]
        _http("POST", f"{self.base}/v1/sessions/{sid}/run", body={"prompt": "go"})
        status, body = _http("GET", f"{self.base}/v1/sessions/{sid}/events/recent")
        self.assertEqual(status, 200)
        kinds = [e["kind"] for e in body["events"]]
        self.assertIn("run.finished", kinds)
        _http("DELETE", f"{self.base}/v1/sessions/{sid}")

    def test_sse_stream(self):
        sid = _http("POST", f"{self.base}/v1/sessions", body={})[1]["session_id"]

        # Open the SSE stream in a background thread, then trigger a run.
        events = []

        def reader():
            try:
                with urllib.request.urlopen(
                    f"{self.base}/v1/sessions/{sid}/events", timeout=3
                ) as resp:
                    while len(events) < 5:
                        line = resp.readline()
                        if not line:
                            break
                        line = line.decode().rstrip("\n")
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
            except Exception:
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        # Small delay so the subscriber registers before we run.
        time.sleep(0.2)
        _http("POST", f"{self.base}/v1/sessions/{sid}/run", body={"prompt": "hi"})
        t.join(timeout=3)
        kinds = [e["kind"] for e in events]
        self.assertIn("run.started", kinds)
        _http("DELETE", f"{self.base}/v1/sessions/{sid}")


class TestServerAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agi import runtime as runtime_module
        from agi.server import RuntimeServer
        cls._patcher = patch.object(runtime_module, "Agent", FakeAgent)
        cls._patcher.start()
        cls.runtime = runtime_module.Runtime()
        cls.server = RuntimeServer(cls.runtime, host="127.0.0.1", port=0, auth_token="secret")
        cls.server.start()
        host, port = cls.server.address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._patcher.stop()

    def test_missing_token_returns_401(self):
        status, _ = _http("GET", f"{self.base}/v1/health")
        self.assertEqual(status, 401)

    def test_wrong_token_returns_401(self):
        status, _ = _http("GET", f"{self.base}/v1/health", token="wrong")
        self.assertEqual(status, 401)

    def test_correct_token_works(self):
        status, body = _http("GET", f"{self.base}/v1/health", token="secret")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")


if __name__ == "__main__":
    unittest.main()
