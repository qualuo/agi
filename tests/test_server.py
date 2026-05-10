"""End-to-end tests for the HTTP server.

We start the stdlib HTTP server on an ephemeral port using a Runtime built
with a fake agent factory (no API calls), then drive it with urllib.
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

from agi.runtime import Runtime
from agi.server import serve

from tests.test_runtime import FakeAgent  # reuse the offline agent


def _request(method: str, url: str, body: dict | None = None,
             token: str | None = None, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class _ServerFixture:
    def __init__(self, *, auth_token: str | None = None):
        def factory(_sid: str) -> FakeAgent:
            return FakeAgent()
        self.runtime = Runtime(agent_factory=factory)
        self.server = serve(self.runtime, host="127.0.0.1", port=0,
                            auth_token=auth_token)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)


class TestServer(unittest.TestCase):
    def setUp(self):
        self.fix = _ServerFixture()

    def tearDown(self):
        self.fix.stop()

    def test_health(self):
        status, body = _request("GET", self.fix.url("/v1/health"))
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_describe(self):
        status, body = _request("GET", self.fix.url("/v1/describe"))
        self.assertEqual(status, 200)
        self.assertIn("model", body)
        self.assertIn("tools", body)
        self.assertIn("pricing", body)

    def test_session_lifecycle(self):
        # create
        status, body = _request("POST", self.fix.url("/v1/sessions"), body={})
        self.assertEqual(status, 201)
        sid = body["id"]
        # list
        status, body = _request("GET", self.fix.url("/v1/sessions"))
        self.assertEqual(status, 200)
        self.assertEqual([s["id"] for s in body["sessions"]], [sid])
        # turn
        status, body = _request(
            "POST",
            self.fix.url(f"/v1/sessions/{sid}/turn"),
            body={"input": "hello world"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["finish_reason"], "ok")
        self.assertIn("hello world", body["text"])
        self.assertGreater(body["cost_usd"], 0)
        # session info reflects the turn
        status, body = _request("GET", self.fix.url(f"/v1/sessions/{sid}"))
        self.assertEqual(status, 200)
        self.assertEqual(body["turn_count"], 1)
        # reset
        status, body = _request(
            "POST", self.fix.url(f"/v1/sessions/{sid}/reset"), body={}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["turn_count"], 0)
        # delete
        status, body = _request("DELETE", self.fix.url(f"/v1/sessions/{sid}"))
        self.assertEqual(status, 200)
        # subsequent get is 404
        status, _ = _request("GET", self.fix.url(f"/v1/sessions/{sid}"))
        self.assertEqual(status, 404)

    def test_turn_requires_input(self):
        sid = self.fix.runtime.create_session()
        status, body = _request(
            "POST", self.fix.url(f"/v1/sessions/{sid}/turn"), body={}
        )
        self.assertEqual(status, 400)
        self.assertIn("input", body["error"])

    def test_turn_for_unknown_session_is_404(self):
        status, body = _request(
            "POST", self.fix.url("/v1/sessions/no-such/turn"), body={"input": "hi"}
        )
        self.assertEqual(status, 404)

    def test_unknown_path_404(self):
        status, _ = _request("GET", self.fix.url("/v1/nowhere"))
        self.assertEqual(status, 404)


class TestServerAuth(unittest.TestCase):
    def setUp(self):
        self.fix = _ServerFixture(auth_token="secret")

    def tearDown(self):
        self.fix.stop()

    def test_unauthenticated_request_rejected(self):
        status, body = _request("GET", self.fix.url("/v1/health"))
        self.assertEqual(status, 401)

    def test_authenticated_request_accepted(self):
        status, body = _request("GET", self.fix.url("/v1/health"), token="secret")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
