"""HTTP/SSE server tests.

Spin up the real server on a random port with a runtime backed by a fake
Anthropic client. Drive it with urllib so we exercise the exact wire format
a coordination engine would consume.
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.coord import Coordinator
from agi.runtime import Runtime
from agi.server import make_server

from tests.fake_client import FakeAnthropic, text_reply


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


class ServerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = FakeAnthropic()

        def factory(**kwargs):
            return Agent(
                client=self.client,
                enable_web_search=False,
                enable_web_fetch=False,
                **kwargs,
            )

        self.rt = Runtime(agent_factory=factory, memory_root=Path(self.tmp.name))
        self.co = Coordinator(self.rt)
        self.port = _free_port()
        self.server = make_server(self.rt, self.co, host="127.0.0.1", port=self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        # Give the server a beat to bind. healthz keeps this honest.
        for _ in range(50):
            try:
                status, _ = _http("GET", self._url("/v1/healthz"))
                if status == 200:
                    break
            except Exception:
                time.sleep(0.02)
        else:
            self.fail("server failed to start")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.tmp.cleanup()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"


class TestServerRoutes(ServerBase):
    def test_healthz(self):
        status, body = _http("GET", self._url("/v1/healthz"))
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_create_send_status_delete(self):
        self.client.responses.append(text_reply("server ok"))

        # POST /v1/sessions
        status, body = _http("POST", self._url("/v1/sessions"), {})
        self.assertEqual(status, 201)
        sid = body["session_id"]
        self.assertTrue(sid)

        # POST /v1/sessions/{sid}/messages
        status, _ = _http(
            "POST",
            self._url(f"/v1/sessions/{sid}/messages"),
            {"input": "hello"},
        )
        self.assertEqual(status, 202)

        # Wait for the turn to complete via the in-process runtime, then
        # query status to confirm.
        end = self.rt.wait_for_turn_end(sid, timeout=5.0)
        self.assertIsNotNone(end)
        self.assertEqual(end.final_text, "server ok")

        status, body = _http("GET", self._url(f"/v1/sessions/{sid}"))
        self.assertEqual(status, 200)
        self.assertEqual(body["turns"], 1)
        self.assertEqual(body["state"], "idle")

        # DELETE
        status, _ = _http("DELETE", self._url(f"/v1/sessions/{sid}"))
        self.assertEqual(status, 200)

    def test_unknown_session_returns_404(self):
        status, body = _http("GET", self._url("/v1/sessions/deadbeef00"))
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_missing_input_returns_400(self):
        status, body = _http("POST", self._url("/v1/sessions"), {})
        self.assertEqual(status, 201)
        sid = body["session_id"]
        status, body = _http(
            "POST", self._url(f"/v1/sessions/{sid}/messages"), {}
        )
        self.assertEqual(status, 400)

    def test_tasks_endpoint_runs_coordinator(self):
        self.client.responses.append(text_reply("task done"))
        status, body = _http(
            "POST",
            self._url("/v1/tasks"),
            {"prompt": "do a thing", "role": "executor"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["final_text"], "task done")
        self.assertEqual(body["task"]["prompt"], "do a thing")


class TestSSEStream(ServerBase):
    def test_event_stream_yields_turn_end(self):
        self.client.responses.append(text_reply("stream ok"))

        status, body = _http("POST", self._url("/v1/sessions"), {})
        sid = body["session_id"]

        # Open the SSE stream on a thread so we can send the message after.
        events: list[dict] = []
        done = threading.Event()

        def consume():
            req = urllib.request.Request(self._url(f"/v1/sessions/{sid}/events"))
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                buf = b""
                while not done.is_set():
                    # read1 returns whatever's available (1..n bytes) instead
                    # of blocking until n bytes are buffered — required for
                    # streaming reads of an SSE response.
                    chunk = resp.read1(1024)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n\n" in buf:
                        block, buf = buf.split(b"\n\n", 1)
                        ev_type = None
                        data = None
                        for line in block.decode().splitlines():
                            if line.startswith("event: "):
                                ev_type = line[len("event: "):]
                            elif line.startswith("data: "):
                                data = json.loads(line[len("data: "):])
                        if data is not None:
                            events.append({"type": ev_type, **data})
                        if ev_type == "TurnEnd":
                            done.set()
                            return

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        # tiny pause so the consumer is attached before we send
        time.sleep(0.1)
        _http(
            "POST",
            self._url(f"/v1/sessions/{sid}/messages"),
            {"input": "hi"},
        )
        t.join(timeout=8.0)
        self.assertTrue(done.is_set(), "did not receive TurnEnd over SSE")
        types = [e["type"] for e in events]
        self.assertIn("TurnStart", types)
        self.assertIn("TurnEnd", types)
        final = [e for e in events if e["type"] == "TurnEnd"][-1]
        self.assertEqual(final["final_text"], "stream ok")


if __name__ == "__main__":
    unittest.main()
