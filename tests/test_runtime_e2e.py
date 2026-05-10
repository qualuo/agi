"""End-to-end test: spawn the runtime as a real subprocess and drive it.

Uses memory/skill operations (no API key needed). This validates the wire
protocol, the CLI flag, and the RuntimeClient together.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coord.client import RuntimeClient


class TestRuntimeE2E(unittest.TestCase):
    def setUp(self):
        # Isolate memory/skills under a temp HOME so we don't touch user state.
        # Preserve real $HOME's user-site so pip-installed packages still load.
        self._tmp = tempfile.TemporaryDirectory()
        env = os.environ.copy()
        real_home = env.get("HOME", "")
        if real_home and "PYTHONUSERBASE" not in env:
            env["PYTHONUSERBASE"] = str(Path(real_home) / ".local")
        env["HOME"] = self._tmp.name
        # Ensure the subprocess doesn't try to authenticate to Anthropic.
        env.pop("ANTHROPIC_API_KEY", None)
        self.client = RuntimeClient(env=env)
        self.client.start()

    def tearDown(self):
        self.client.close()
        self._tmp.cleanup()

    def test_capabilities(self):
        caps = self.client.capabilities()
        self.assertTrue(caps["ok"])
        self.assertEqual(caps["model"], "claude-opus-4-7")
        self.assertIn("read_file", caps["tools"])
        self.assertTrue(caps["has_skills"])

    def test_memory_round_trip(self):
        save = self.client.memory_save("project alpha uses postgres", tags=["proj"])
        self.assertTrue(save["ok"])
        result = self.client.memory_search("postgres")
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("postgres", result["results"][0]["text"])

    def test_skill_round_trip(self):
        # Save a skill via memory.* doesn't apply, so use the save_skill tool
        # path through the runtime: we go through the request directly.
        self.client.request(
            "skills.list"
        )  # ensure the request type exists
        # Save via the FS directly (the runtime exposes read but not save in the
        # protocol — agent does that via tools). We exercise list/find/read.
        skills_dir = Path(self._tmp.name) / ".agi" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "deploy.md").write_text(
            "---\nname: deploy\nwhen: Deploy to prod.\ntags: [deploy]\n---\nrun ./deploy.sh\n"
        )
        listed = self.client.skills_list()
        names = [s["name"] for s in listed["skills"]]
        self.assertIn("deploy", names)

        found = self.client.skills_find("prod", k=5)
        self.assertEqual([s["name"] for s in found["skills"]], ["deploy"])

        read = self.client.request("skills.read", name="deploy")
        self.assertEqual(read["name"], "deploy")
        self.assertIn("deploy.sh", read["body"])


if __name__ == "__main__":
    unittest.main()
