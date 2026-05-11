"""HTTP / SSE server tests."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.costs import Usage
from agi.events import TEXT, make
from agi.runtime import Runtime, RunRequest
from agi.server import serve


class FakeAgent:
    def __init__(self, runtime, handle):
        self.runtime = runtime
        self.handle = handle
        self.run_id = handle.run_id
        self.last_critic_score = None
        self.model = "fake"
        self.usage = Usage(input_tokens=1, output_tokens=1)

    def chat(self, task: str, max_iterations: int = 25) -> str:
        self.handle.emit(make(TEXT, self.run_id, text=f"echo:{task}"))
        return f"echo:{task}"


def _factory(runtime, handle):
    return FakeAgent(runtime, handle)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.runtime = Runtime(root_dir=cls.root, agent_factory=_factory)
        cls.server = serve(cls.runtime, host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._tmp.cleanup()

    def test_health(self):
        with urllib.request.urlopen(f"{self.base}/v1/health") as r:
            body = json.loads(r.read())
        self.assertTrue(body["ok"])

    def test_create_run_and_get_status(self):
        req = urllib.request.Request(
            f"{self.base}/v1/runs",
            data=json.dumps({"task": "hi", "reflect": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            create = json.loads(r.read())
        self.assertIn("run_id", create)

        # Poll until done.
        for _ in range(50):
            with urllib.request.urlopen(f"{self.base}/v1/runs/{create['run_id']}") as r:
                status = json.loads(r.read())
            if status["done"]:
                break
            time.sleep(0.05)
        self.assertTrue(status["done"])
        self.assertEqual(status["result"]["text"], "echo:hi")

    def test_event_stream_yields_events(self):
        req = urllib.request.Request(
            f"{self.base}/v1/runs",
            data=json.dumps({"task": "stream-me", "reflect": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            create = json.loads(r.read())

        # Read the SSE stream until DONE.
        seen_types: list[str] = []
        with urllib.request.urlopen(f"{self.base}/v1/runs/{create['run_id']}/events", timeout=5) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    seen_types.append(line.split(":", 1)[1].strip())
                if "done" in seen_types:
                    break
        self.assertIn("run_started", seen_types)
        self.assertIn("done", seen_types)

    def test_metrics_includes_event_counts(self):
        with urllib.request.urlopen(f"{self.base}/v1/metrics") as r:
            body = json.loads(r.read())
        self.assertIn("event_counts", body)
        self.assertIn("runs_started", body)

    def test_skills_endpoints(self):
        # Create a skill.
        req = urllib.request.Request(
            f"{self.base}/v1/skills",
            data=json.dumps({
                "name": "Server Test Skill",
                "description": "for tests",
                "body": "do the thing",
                "tags": ["test"],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            created = json.loads(r.read())
        self.assertEqual(created["name"], "Server Test Skill")

        # List skills.
        with urllib.request.urlopen(f"{self.base}/v1/skills") as r:
            listing = json.loads(r.read())
        names = [s["name"] for s in listing["skills"]]
        self.assertIn("Server Test Skill", names)

    def test_404_for_unknown_route(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self.base}/nope")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
