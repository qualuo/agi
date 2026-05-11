"""Tests for agi.server — exercises the HTTP surface against a runtime that
uses a fake agent (no API calls)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-server-tests")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import runtime as runtime_mod
from agi.runtime import Runtime
from agi.server import _serve_in_thread
from tests.test_runtime import _FakeAgent  # reuse the fake


class TestServer(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_agent = runtime_mod.Agent
        runtime_mod.Agent = _FakeAgent
        self.tmp = tempfile.TemporaryDirectory()
        self.rt = Runtime(
            memory_path=Path(self.tmp.name) / "mem.jsonl",
            skill_root=Path(self.tmp.name) / "skills",
            synth_root=Path(self.tmp.name) / "synth",
            trace_path=Path(self.tmp.name) / "traces.jsonl",
            max_workers=2,
        )
        self.httpd, _ = _serve_in_thread(self.rt, host="127.0.0.1", port=0)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.rt.shutdown(wait=True)
        runtime_mod.Agent = self._orig_agent
        self.tmp.cleanup()

    def _get(self, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(self.base + path, timeout=5) as r:
                return r.status, json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")

    def test_healthz(self) -> None:
        status, body = self._get("/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_manifest(self) -> None:
        status, body = self._get("/manifest")
        self.assertEqual(status, 200)
        self.assertIn("runtime_version", body)
        self.assertIn("tools", body)
        self.assertIn("roles", body)

    def test_submit_then_wait_then_status(self) -> None:
        status, body = self._post("/tasks", {"prompt": "hi"})
        self.assertEqual(status, 202)
        task_id = body["id"]
        # wait for completion
        status, snap = self._get(f"/tasks/{task_id}/wait?timeout=5")
        self.assertEqual(status, 200)
        self.assertEqual(snap["status"], "succeeded")
        self.assertEqual(snap["result"], "fake: hi")

    def test_list_tasks(self) -> None:
        self._post("/tasks", {"prompt": "a"})
        self._post("/tasks", {"prompt": "b"})
        time.sleep(0.3)
        status, body = self._get("/tasks")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(body["tasks"]), 2)

    def test_unknown_task_returns_404(self) -> None:
        status, body = self._get("/tasks/nonexistent")
        self.assertEqual(status, 404)

    def test_cancel(self) -> None:
        status, body = self._post("/tasks", {"prompt": "SLEEP 2"})
        task_id = body["id"]
        time.sleep(0.1)
        status, body = self._post(f"/tasks/{task_id}/cancel", {})
        self.assertEqual(status, 200)
        self.assertTrue(body["cancelled"])

    def test_post_missing_prompt(self) -> None:
        status, body = self._post("/tasks", {})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
