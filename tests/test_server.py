"""HTTP server: control plane + SSE.

Spins up the server on an ephemeral port against a fake-Agent runtime
so no API key or network is needed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import EventBus, Memory, Runtime, SkillLibrary
from agi.server import serve_in_thread
from tests._fakes import constant_factory


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.runtime = Runtime(
            skill_library=SkillLibrary(path=tmp / "s"),
            memory=Memory(path=tmp / "m.jsonl"),
            bus=EventBus(history=128),
            agent_factory=constant_factory("hello from fake"),
        )
        cls.httpd, cls.thread, cls.port = serve_in_thread(cls.runtime)
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close_all()
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls._tmp.cleanup()

    def _get(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path)
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.getcode(), json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.getcode(), json.loads(r.read().decode("utf-8"))

    def _delete(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, method="DELETE")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.getcode(), json.loads(r.read().decode("utf-8"))

    def test_healthz(self):
        with urllib.request.urlopen(self.base + "/healthz", timeout=5) as r:
            self.assertEqual(r.getcode(), 200)
            self.assertIn("ok", r.read().decode())

    def test_manifest(self):
        code, body = self._get("/v1/manifest")
        self.assertEqual(code, 200)
        self.assertEqual(body["name"], "agi-runtime")
        self.assertIn("roles", body)

    def test_open_chat_close_round_trip(self):
        # before
        _, listing = self._get("/v1/sessions")
        before = len(listing["sessions"])

        # open
        code, info = self._post("/v1/sessions", {"role": "general"})
        self.assertEqual(code, 201)
        sid = info["id"]
        self.assertEqual(info["role"], "general")

        # chat
        code, result = self._post(f"/v1/sessions/{sid}/chat", {"input": "hi"})
        self.assertEqual(code, 200)
        self.assertEqual(result["text"], "hello from fake")
        self.assertEqual(result["stop_reason"], "ok")

        # get
        code, fetched = self._get(f"/v1/sessions/{sid}")
        self.assertEqual(code, 200)
        self.assertEqual(fetched["id"], sid)
        self.assertGreaterEqual(fetched["turns"], 1)

        # listing grew
        _, listing = self._get("/v1/sessions")
        self.assertEqual(len(listing["sessions"]), before + 1)

        # close
        code, _ = self._delete(f"/v1/sessions/{sid}")
        self.assertEqual(code, 200)

    def test_chat_missing_input_400s(self):
        _, info = self._post("/v1/sessions", {})
        sid = info["id"]
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{self.base}/v1/sessions/{sid}/chat",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=5,
            ):
                self.fail("expected 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
        finally:
            self._delete(f"/v1/sessions/{sid}")

    def test_unknown_session_404s(self):
        try:
            with urllib.request.urlopen(self.base + "/v1/sessions/nope", timeout=5):
                self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_history_returns_recent_events(self):
        # generate some events
        _, info = self._post("/v1/sessions", {})
        sid = info["id"]
        self._post(f"/v1/sessions/{sid}/chat", {"input": "x"})
        self._delete(f"/v1/sessions/{sid}")

        code, body = self._get("/v1/history")
        self.assertEqual(code, 200)
        types = {e["type"] for e in body["events"]}
        self.assertTrue({"session_opened", "turn_started", "turn_finished"} <= types)


if __name__ == "__main__":
    unittest.main()
