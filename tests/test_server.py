"""Tests for the HTTP+SSE runtime server.

Spins up a real server on an ephemeral port with a FakeAgent backend, then
exercises endpoints via the stdlib urllib. SSE streaming is verified with
a raw socket so we can read events as they arrive without blocking forever.
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

from agi.memory import Memory
from agi.runtime import Runtime
from agi.server import RuntimeServer
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


def _make_runtime() -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=FakeAgent,
    )


def _post(url: str, payload: dict | None = None, token: str | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8") or "{}"
        return resp.status, json.loads(body) if body else {}


def _get(url: str, token: str | None = None):
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _delete(url: str, token: str | None = None):
    req = urllib.request.Request(url, method="DELETE")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status


class TestServer(unittest.TestCase):
    def setUp(self):
        self.runtime = _make_runtime()
        self.server = RuntimeServer(self.runtime, host="127.0.0.1", port=0)
        self.server.start()
        self.base = self.server.base_url

    def tearDown(self):
        self.server.stop()

    def test_healthz(self):
        status, body = _get(f"{self.base}/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_capabilities(self):
        status, body = _get(f"{self.base}/capabilities")
        self.assertEqual(status, 200)
        self.assertIn("models", body)
        self.assertEqual(body["active_sessions"], 0)

    def test_create_chat_get_delete(self):
        status, body = _post(f"{self.base}/sessions", {})
        self.assertEqual(status, 201)
        sid = body["id"]

        status, body = _post(f"{self.base}/sessions/{sid}/chat", {"prompt": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(body["final_text"], "ok")
        self.assertEqual(body["session"]["turn_count"], 1)

        status, body = _get(f"{self.base}/sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertEqual(body["turn_count"], 1)

        status = _delete(f"{self.base}/sessions/{sid}")
        self.assertEqual(status, 204)

        status, body = _get(f"{self.base}/sessions/{sid}")
        self.assertEqual(body["ended"], True)

    def test_chat_missing_prompt(self):
        _, body = _post(f"{self.base}/sessions", {})
        sid = body["id"]
        try:
            _post(f"{self.base}/sessions/{sid}/chat", {})
            self.fail("expected error")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_unknown_session_404(self):
        try:
            _get(f"{self.base}/sessions/nope")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_create_skill(self):
        status, body = _post(
            f"{self.base}/skills",
            {"name": "s1", "description": "d", "body": "b", "tags": ["t"]},
        )
        self.assertEqual(status, 201)
        status, body = _get(f"{self.base}/skills")
        self.assertEqual(status, 200)
        names = [s["name"] for s in body]
        self.assertIn("s1", names)

    def test_synthesize_tool_endpoint(self):
        status, body = _post(
            f"{self.base}/tools",
            {
                "name": "echo",
                "description": "echo input",
                "code": "def run(text=''):\n    return text",
                "smoke_test_kwargs": {"text": "x"},
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["name"], "echo")

    def test_events_history(self):
        _, body = _post(f"{self.base}/sessions", {})
        sid = body["id"]
        _post(f"{self.base}/sessions/{sid}/chat", {"prompt": "hi"})
        status, body = _get(f"{self.base}/events/history?session_id=" + sid)
        self.assertEqual(status, 200)
        kinds = {e["kind"] for e in body}
        self.assertIn("session.created", kinds)
        self.assertIn("chat.completed", kinds)

    def test_metrics(self):
        status, body = _get(f"{self.base}/metrics")
        self.assertEqual(status, 200)
        self.assertIn("sessions_created", body)
        self.assertIn("total_cost_usd", body)

    def test_tasks_endpoint_404_without_queue(self):
        try:
            _get(f"{self.base}/tasks")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


class TestServerWithTaskQueue(unittest.TestCase):
    def setUp(self):
        from agi.tasks import TaskQueue
        self.runtime = _make_runtime()
        self.queue = TaskQueue()
        self.server = RuntimeServer(
            self.runtime, host="127.0.0.1", port=0, task_queue=self.queue
        )
        self.server.start()
        self.base = self.server.base_url

    def tearDown(self):
        self.server.stop()

    def test_submit_drain_get(self):
        status, body = _post(f"{self.base}/tasks", {"prompt": "hi"})
        self.assertEqual(status, 201)
        tid = body["id"]

        status, body = _post(f"{self.base}/tasks/drain", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["executed"], 1)

        status, body = _get(f"{self.base}/tasks/{tid}")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "done")
        self.assertEqual(body["result"], "ok")


class TestServerWithSessionStore(unittest.TestCase):
    def setUp(self):
        import tempfile
        from agi.persistence import SessionStore
        self.tmp = tempfile.mkdtemp()
        self.runtime = Runtime(
            memory=Memory(path=Path(self.tmp) / "m.jsonl"),
            skills=SkillLibrary(path=Path(self.tmp) / "skills"),
            agent_factory=FakeAgent,
            session_store=SessionStore(path=Path(self.tmp) / "sessions"),
        )
        self.server = RuntimeServer(self.runtime, host="127.0.0.1", port=0)
        self.server.start()
        self.base = self.server.base_url

    def tearDown(self):
        self.server.stop()

    def test_checkpoint_and_restore_roundtrip(self):
        _, body = _post(f"{self.base}/sessions", {})
        sid = body["id"]
        _post(f"{self.base}/sessions/{sid}/chat", {"prompt": "hi"})
        status, body = _post(f"{self.base}/sessions/{sid}/checkpoint", {})
        self.assertEqual(status, 200)
        self.assertTrue(Path(body["path"]).exists())

        # Reload by simulating a new server backed by a fresh runtime that
        # points at the same session store.
        from agi.persistence import SessionStore
        runtime2 = Runtime(
            memory=Memory(path=Path(self.tmp) / "m.jsonl"),
            skills=SkillLibrary(path=Path(self.tmp) / "skills"),
            agent_factory=FakeAgent,
            session_store=SessionStore(path=Path(self.tmp) / "sessions"),
        )
        server2 = RuntimeServer(runtime2, host="127.0.0.1", port=0)
        server2.start()
        try:
            status, body = _post(f"{server2.base_url}/sessions/restore", {"session_id": sid})
            self.assertEqual(status, 200)
            self.assertEqual(body["id"], sid)
            status, body = _get(f"{server2.base_url}/sessions/{sid}")
            self.assertEqual(body["turn_count"], 1)
        finally:
            server2.stop()


class TestServerAuth(unittest.TestCase):
    def setUp(self):
        self.runtime = _make_runtime()
        self.server = RuntimeServer(self.runtime, host="127.0.0.1", port=0, auth_token="secret")
        self.server.start()
        self.base = self.server.base_url

    def tearDown(self):
        self.server.stop()

    def test_unauthorized_without_token(self):
        try:
            _get(f"{self.base}/healthz")
            self.fail("expected 401")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)

    def test_authorized_with_token(self):
        status, body = _get(f"{self.base}/healthz", token="secret")
        self.assertEqual(status, 200)


class TestSSE(unittest.TestCase):
    def setUp(self):
        self.runtime = _make_runtime()
        self.server = RuntimeServer(self.runtime, host="127.0.0.1", port=0)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def test_sse_delivers_events(self):
        # Open SSE connection in a thread, capture events.
        host, port = self.server.host, self.server.port
        events: list[str] = []
        ready = threading.Event()

        def reader():
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.sendall(
                    b"GET /events HTTP/1.1\r\n"
                    b"Host: localhost\r\n"
                    b"Accept: text/event-stream\r\n\r\n"
                )
                sock.settimeout(3.0)
                buf = b""
                # Read headers
                while b"\r\n\r\n" not in buf:
                    chunk = sock.recv(4096)
                    if not chunk:
                        return
                    buf += chunk
                ready.set()
                buf = buf.split(b"\r\n\r\n", 1)[1]
                deadline = time.time() + 2.5
                while time.time() < deadline and len(events) < 2:
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    # parse event blocks separated by blank lines
                    while b"\n\n" in buf:
                        block, buf = buf.split(b"\n\n", 1)
                        for line in block.decode("utf-8", errors="replace").splitlines():
                            if line.startswith("event: "):
                                events.append(line[len("event: "):])

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        self.assertTrue(ready.wait(2.0))

        # Trigger some events
        sid = self.runtime.create_session()
        # Need an Agent to deliver chat — use the runtime's chat
        self.runtime.chat(sid, "hi")

        t.join(timeout=4.0)
        self.assertIn("session.created", events)


if __name__ == "__main__":
    unittest.main()
