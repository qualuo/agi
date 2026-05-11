"""Tests for the stdlib HTTP+SSE server.

Each test boots the server on an ephemeral port against a fake runtime, hits
it with urllib, and shuts it down. Networking is loopback only.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event, TextDelta, TurnCompleted
from agi.server import serve


class FakeServerRuntime:
    """Implements only the surface the server uses."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self.scripted_events: list[Event] = []
        self.start_calls: list[dict | None] = []

    def start_session(self, session_id=None, config=None, *, interceptor=None, agent=None):
        sid = session_id or f"sess-{len(self.sessions) + 1}"
        if sid in self.sessions:
            raise ValueError(f"session {sid} already exists")
        self.sessions[sid] = {"config": config}
        self.start_calls.append({"sid": sid, "config": config})
        return sid

    def close_session(self, sid):
        self.sessions.pop(sid, None)

    def get_session(self, sid):
        if sid not in self.sessions:
            raise KeyError(sid)
        return self.sessions[sid]

    def send(self, sid, prompt, max_iterations=25) -> Iterator[Event]:
        for e in self.scripted_events:
            # Re-label session_id so the server matches expectations
            e.session_id = sid
            yield e

    def snapshot(self, sid):
        if sid not in self.sessions:
            raise KeyError(sid)
        return {"id": sid, "fake": True}

    def restore(self, snap):
        if "id" not in snap:
            raise ValueError("missing id")
        self.sessions[snap["id"]] = {"config": None}
        return snap["id"]


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerHarness:
    def __init__(self, runtime, token=None):
        self.runtime = runtime
        self.port = _free_port()
        self.httpd = serve(runtime=runtime, host="127.0.0.1", port=self.port, token=token)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None, raw=False):
        url = self.base + path
        data = None
        hdrs = dict(headers or {})
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = resp.read()
            status = resp.status
        if raw:
            return status, payload
        if not payload:
            return status, {}
        return status, json.loads(payload.decode("utf-8"))


class TestHealthAndAuth(unittest.TestCase):
    def test_health_no_auth(self):
        h = _ServerHarness(FakeServerRuntime())
        try:
            status, body = h.request("GET", "/v1/health")
            self.assertEqual(status, 200)
            self.assertTrue(body["ok"])
        finally:
            h.stop()

    def test_auth_required_when_token_set(self):
        h = _ServerHarness(FakeServerRuntime(), token="secret-token")
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                h.request("GET", "/v1/health")
            self.assertEqual(cm.exception.code, 401)
            # With the correct token it should work
            status, body = h.request(
                "GET",
                "/v1/health",
                headers={"Authorization": "Bearer secret-token"},
            )
            self.assertEqual(status, 200)
        finally:
            h.stop()

    def test_auth_wrong_token_rejected(self):
        h = _ServerHarness(FakeServerRuntime(), token="secret-token")
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                h.request(
                    "GET",
                    "/v1/health",
                    headers={"Authorization": "Bearer wrong"},
                )
            self.assertEqual(cm.exception.code, 401)
        finally:
            h.stop()


class TestSessionRoutes(unittest.TestCase):
    def test_create_and_delete_session(self):
        rt = FakeServerRuntime()
        h = _ServerHarness(rt)
        try:
            status, body = h.request("POST", "/v1/sessions", body={"session_id": "abc"})
            self.assertEqual(status, 201)
            self.assertEqual(body["session_id"], "abc")
            self.assertIn("abc", rt.sessions)

            status, _ = h.request("DELETE", "/v1/sessions/abc")
            self.assertEqual(status, 200)
            self.assertNotIn("abc", rt.sessions)
        finally:
            h.stop()

    def test_create_with_config(self):
        rt = FakeServerRuntime()
        h = _ServerHarness(rt)
        try:
            cfg = {"model": "claude-haiku-4-5-20251001", "effort": "low"}
            status, body = h.request("POST", "/v1/sessions", body={"config": cfg})
            self.assertEqual(status, 201)
            sid = body["session_id"]
            self.assertEqual(rt.sessions[sid]["config"].model, "claude-haiku-4-5-20251001")
        finally:
            h.stop()

    def test_duplicate_session_returns_409(self):
        rt = FakeServerRuntime()
        h = _ServerHarness(rt)
        try:
            h.request("POST", "/v1/sessions", body={"session_id": "dup"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                h.request("POST", "/v1/sessions", body={"session_id": "dup"})
            self.assertEqual(cm.exception.code, 409)
        finally:
            h.stop()

    def test_snapshot_roundtrip(self):
        rt = FakeServerRuntime()
        h = _ServerHarness(rt)
        try:
            h.request("POST", "/v1/sessions", body={"session_id": "snap"})
            status, snap = h.request("GET", "/v1/sessions/snap/snapshot")
            self.assertEqual(status, 200)
            self.assertEqual(snap["id"], "snap")

            # Restore into a fresh runtime context
            rt.sessions.clear()
            status, body = h.request("POST", "/v1/sessions/snap/snapshot", body=snap)
            self.assertEqual(status, 200)
            self.assertTrue(body["restored"])
            self.assertIn("snap", rt.sessions)
        finally:
            h.stop()


class TestSSEStream(unittest.TestCase):
    def test_send_streams_events(self):
        rt = FakeServerRuntime()
        rt.scripted_events = [
            TextDelta(session_id="placeholder", seq=1, text="hel"),
            TextDelta(session_id="placeholder", seq=2, text="lo"),
            TurnCompleted(
                session_id="placeholder",
                seq=3,
                text="hello",
                stop_reason="end_turn",
                cost_usd=0.0001,
            ),
        ]
        h = _ServerHarness(rt)
        try:
            h.request("POST", "/v1/sessions", body={"session_id": "s"})
            # POST /v1/sessions/s/send with streamed body
            url = h.base + "/v1/sessions/s/send"
            data = json.dumps({"prompt": "say hi"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                # Read until the connection closes; server closes after
                # turn_completed
                body = resp.read().decode("utf-8")

            # SSE frame: lines like `event: <kind>\ndata: <json>\n\n`
            frames = [f for f in body.split("\n\n") if f.strip()]
            kinds = []
            for f in frames:
                first = f.splitlines()[0]
                self.assertTrue(first.startswith("event: "))
                kinds.append(first[len("event: ") :])
            self.assertEqual(kinds, ["text_delta", "text_delta", "turn_completed"])

            # Final turn_completed data includes the text
            last_data = frames[-1].splitlines()[1][len("data: ") :]
            payload = json.loads(last_data)
            self.assertEqual(payload["text"], "hello")
            self.assertEqual(payload["kind"], "turn_completed")
        finally:
            h.stop()


if __name__ == "__main__":
    unittest.main()
