"""End-to-end tests for the HTTP runtime server.

We bring up the server in a thread, hit it with urllib, and verify the
JSON wire shape. No API calls are made — only routes that don't require
the model are exercised. The route that *does* take a step (POST step)
is not invoked here.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.runtime import Runtime
from agi.server import make_server
from agi.skills import SkillLibrary


class FakeClient:
    pass


def _request(url, method="GET", body=None, token=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name)
        cls.runtime = Runtime(
            memory=Memory(path=path / "m.jsonl"),
            skills=SkillLibrary(path=path / "skills"),
            tracer=None,
            client=FakeClient(),
        )
        cls.server = make_server(cls.runtime, host="127.0.0.1", port=0)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://{cls.host}:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls._tmp.cleanup()

    def test_health(self):
        status, body = _request(f"{self.base}/v1/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_capabilities(self):
        status, body = _request(f"{self.base}/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertIn("roles", body)
        self.assertIn("tools", body)

    def test_session_lifecycle(self):
        status, body = _request(
            f"{self.base}/v1/sessions",
            method="POST",
            body={"role": "general", "goal": "test session"},
        )
        self.assertEqual(status, 201)
        sid = body["session_id"]
        self.assertEqual(body["role"], "general")
        self.assertEqual(body["goal"], "test session")
        self.assertTrue(body["active"])

        # Snapshot
        status, snap = _request(f"{self.base}/v1/sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertEqual(snap["session_id"], sid)

        # Inject
        status, body = _request(
            f"{self.base}/v1/sessions/{sid}/inject",
            method="POST",
            body={"text": "an env event"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        # End
        status, body = _request(
            f"{self.base}/v1/sessions/{sid}",
            method="DELETE",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["end_reason"], "complete")
        self.assertFalse(body["active"])

    def test_create_session_unknown_role(self):
        status, body = _request(
            f"{self.base}/v1/sessions",
            method="POST",
            body={"role": "no-such-role"},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_skill_crud(self):
        status, body = _request(
            f"{self.base}/v1/skills",
            method="POST",
            body={"title": "T", "when": "always", "procedure": "do it", "triggers": ["t"]},
        )
        self.assertEqual(status, 201)
        sid = body["id"]
        self.assertEqual(body["title"], "T")

        status, body = _request(f"{self.base}/v1/skills?q=t")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["skills"]), 1)

        status, _ = _request(f"{self.base}/v1/skills/{sid}", method="DELETE")
        self.assertEqual(status, 200)

    def test_memory_round_trip(self):
        status, body = _request(
            f"{self.base}/v1/memory",
            method="POST",
            body={"text": "remember this", "tags": ["x"]},
        )
        self.assertEqual(status, 201)

        status, body = _request(f"{self.base}/v1/memory/search?q=remember")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(body["results"]), 1)

    def test_unknown_route(self):
        status, body = _request(f"{self.base}/v1/nope")
        self.assertEqual(status, 404)


class TestServerAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name)
        cls.runtime = Runtime(
            memory=Memory(path=path / "m.jsonl"),
            skills=SkillLibrary(path=path / "skills"),
            tracer=None,
            client=FakeClient(),
        )
        cls.server = make_server(cls.runtime, host="127.0.0.1", port=0, auth_token="secret123")
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://{cls.host}:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls._tmp.cleanup()

    def test_unauthorized_without_token(self):
        status, body = _request(f"{self.base}/v1/health")
        self.assertEqual(status, 401)

    def test_authorized_with_token(self):
        status, body = _request(f"{self.base}/v1/health", token="secret123")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
