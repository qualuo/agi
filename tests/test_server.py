"""End-to-end test of the runtime HTTP server.

Boots a real ThreadingHTTPServer on a free port, drives it via stdlib
urllib, and verifies the JSON-over-HTTP contract a coordination engine
would consume. The Runtime is wired with FakeAgent so no API calls happen.
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

# Reuse the FakeAgent from test_runtime so we don't hit the real API.
from tests.test_runtime import FakeAgent


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _request(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {}
        return e.code, payload


class TestRuntimeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.runtime = Runtime(agent_factory=FakeAgent)
        cls.server = serve(host="127.0.0.1", port=cls.port, runtime=cls.runtime)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # Wait briefly for the listener.
        for _ in range(50):
            try:
                _request("GET", cls.url("/"))
                break
            except Exception:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    @classmethod
    def url(cls, path: str) -> str:
        return f"http://127.0.0.1:{cls.port}{path}"

    def test_root_returns_capabilities(self):
        status, body = _request("GET", self.url("/"))
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertIn("tools", body["capabilities"])

    def test_capabilities_endpoint(self):
        status, body = _request("GET", self.url("/capabilities"))
        self.assertEqual(status, 200)
        self.assertIn("read_file", body["tools"])

    def test_session_lifecycle(self):
        # create
        status, body = _request("POST", self.url("/sessions"), {"role": "executor"})
        self.assertEqual(status, 201)
        sid = body["id"]
        self.assertEqual(body["role"], "executor")
        # list
        status, listed = _request("GET", self.url("/sessions"))
        self.assertEqual(status, 200)
        self.assertTrue(any(s["id"] == sid for s in listed))
        # step
        status, step_body = _request("POST", self.url(f"/sessions/{sid}/step"), {"input": "hello"})
        self.assertEqual(status, 200)
        self.assertEqual(step_body["text"], "ok")
        self.assertGreaterEqual(step_body["input_tokens"], 0)
        # info
        status, info = _request("GET", self.url(f"/sessions/{sid}"))
        self.assertEqual(status, 200)
        self.assertEqual(info["turns"], 1)
        # snapshot
        status, snap = _request("POST", self.url(f"/sessions/{sid}/snapshot"))
        self.assertEqual(status, 200)
        self.assertEqual(snap["id"], sid)
        # delete
        status, body = _request("DELETE", self.url(f"/sessions/{sid}"))
        self.assertEqual(status, 200)
        self.assertTrue(body["closed"])

    def test_step_requires_input(self):
        status, body = _request("POST", self.url("/sessions"), {})
        sid = body["id"]
        status, payload = _request("POST", self.url(f"/sessions/{sid}/step"), {})
        self.assertEqual(status, 400)
        self.assertIn("input", payload["error"])

    def test_step_unknown_session_404s(self):
        status, payload = _request("POST", self.url("/sessions/does-not-exist/step"), {"input": "hi"})
        self.assertEqual(status, 404)

    def test_unknown_path_404s(self):
        status, _ = _request("GET", self.url("/no-such-endpoint"))
        self.assertEqual(status, 404)

    def test_restore_session_endpoint(self):
        # create + step + snapshot + close
        _, body = _request("POST", self.url("/sessions"), {})
        sid = body["id"]
        _request("POST", self.url(f"/sessions/{sid}/step"), {"input": "first"})
        _, snap = _request("POST", self.url(f"/sessions/{sid}/snapshot"))
        _request("DELETE", self.url(f"/sessions/{sid}"))
        # restore
        status, restored = _request("POST", self.url("/sessions/restore"), {"snapshot": snap})
        self.assertEqual(status, 201)
        self.assertEqual(restored["id"], sid)
        self.assertEqual(restored["turns"], 1)

    def test_sse_replay_returns_buffered_events(self):
        # Create a session, take a step, then connect to events with since=0
        # and read a few events. The test reads only headers + a chunk and
        # disconnects so we don't block forever.
        _, body = _request("POST", self.url("/sessions"), {})
        sid = body["id"]
        _request("POST", self.url(f"/sessions/{sid}/step"), {"input": "hi"})

        import socket

        s = socket.create_connection(("127.0.0.1", self.port), timeout=5.0)
        try:
            s.sendall(
                f"GET /sessions/{sid}/events?since=0 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("ascii")
            )
            s.settimeout(3.0)
            buf = b""
            deadline = time.time() + 3.0
            # Read until we see at least one "event:" line emitted.
            while time.time() < deadline and b"\nevent:" not in buf:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()

        text = buf.decode("utf-8", errors="replace")
        self.assertIn("Content-Type: text/event-stream", text)
        self.assertIn("event: ", text)


if __name__ == "__main__":
    unittest.main()
