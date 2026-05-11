"""Tests for agi.world_model — pure filesystem, no API."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.world_model import Observation, WorldModel


class TestWorldModel(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "world.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_observe_records_and_indexes_latest(self) -> None:
        wm = WorldModel(path=self.path)
        wm.observe(kind="file", id="/a", action="read", outcome="success")
        wm.observe(kind="file", id="/a", action="write", outcome="failure", detail={"errno": 13})
        latest = wm.latest("file", "/a")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.action, "write")
        self.assertEqual(latest.outcome, "failure")
        self.assertEqual(latest.detail.get("errno"), 13)

    def test_summary_counts_distinct_entities(self) -> None:
        wm = WorldModel(path=self.path)
        wm.observe(kind="file", id="/a", action="read")
        wm.observe(kind="file", id="/a", action="read")  # same entity, doesn't add
        wm.observe(kind="file", id="/b", action="write")
        wm.observe(kind="url", id="https://x.test", action="fetch")
        s = wm.summary()
        self.assertEqual(s["entity_counts"]["file"], 2)
        self.assertEqual(s["entity_counts"]["url"], 1)
        self.assertEqual(s["total_entities"], 3)

    def test_summary_surfaces_recent_failures(self) -> None:
        wm = WorldModel(path=self.path)
        wm.observe(kind="command", id="ls /nope", action="run", outcome="failure")
        wm.observe(kind="command", id="ls /nope2", action="run", outcome="failure")
        wm.observe(kind="command", id="ls /yes", action="run", outcome="success")
        s = wm.summary()
        self.assertEqual(len(s["recent_failures"]), 2)
        self.assertTrue(all(f["outcome"] == "failure" for f in s["recent_failures"]))

    def test_persists_across_reload(self) -> None:
        wm = WorldModel(path=self.path)
        wm.observe(kind="url", id="https://x.test/", action="fetch")
        wm.observe(kind="url", id="https://y.test/", action="fetch", outcome="failure")
        wm2 = WorldModel(path=self.path)
        self.assertIsNotNone(wm2.latest("url", "https://x.test/"))
        self.assertEqual(wm2.latest("url", "https://y.test/").outcome, "failure")

    def test_known_filters_by_kind(self) -> None:
        wm = WorldModel(path=self.path)
        wm.observe(kind="file", id="/a", action="read")
        wm.observe(kind="url", id="https://x", action="fetch")
        files = wm.known("file")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].entity_id, "/a")
        self.assertEqual(wm.known("entity"), [])

    def test_skips_malformed_lines(self) -> None:
        # Pre-seed the file with a junk line, then construct.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            f.write("not-json\n")
            f.write(json.dumps({
                "entity_kind": "file", "entity_id": "/a", "action": "read",
                "outcome": "success", "ts": 1.0, "detail": {},
            }) + "\n")
        wm = WorldModel(path=self.path)
        self.assertIsNotNone(wm.latest("file", "/a"))


class TestToolsAutoRecord(unittest.TestCase):
    """Confirm make_tools auto-records observations when a WorldModel is set,
    and that it's a no-op when world_model=None (backward compat)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make(self, with_world: bool):
        from agi.memory import Memory
        from agi.tools import make_tools
        memory = Memory(path=self.tmp / "m.jsonl")
        wm = WorldModel(path=self.tmp / "w.jsonl") if with_world else None
        schemas, handlers = make_tools(memory, world_model=wm)
        return wm, schemas, handlers

    def test_read_write_record_observations(self) -> None:
        wm, _, handlers = self._make(with_world=True)
        target = self.tmp / "x.txt"
        handlers["write_file"](path=str(target), content="hi")
        handlers["read_file"](path=str(target))
        assert wm is not None
        latest = wm.latest("file", str(target))
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.action, "read")
        self.assertEqual(latest.outcome, "success")

    def test_failed_read_records_failure(self) -> None:
        wm, _, handlers = self._make(with_world=True)
        handlers["read_file"](path=str(self.tmp / "missing.txt"))
        assert wm is not None
        obs = wm.known("file")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].outcome, "failure")

    def test_run_bash_records_exit_code(self) -> None:
        wm, _, handlers = self._make(with_world=True)
        handlers["run_bash"](command="true")
        handlers["run_bash"](command="false")
        assert wm is not None
        cmds = {o.entity_id: o for o in wm.known("command")}
        self.assertEqual(cmds["true"].outcome, "success")
        self.assertEqual(cmds["false"].outcome, "failure")
        self.assertEqual(cmds["false"].detail.get("exit_code"), 1)

    def test_tools_unchanged_without_world_model(self) -> None:
        # Backward compat: existing make_tools(memory) still works.
        wm, schemas, handlers = self._make(with_world=False)
        self.assertIsNone(wm)
        self.assertNotIn("world_summary", handlers)
        # Existing tools still functional.
        target = self.tmp / "y.txt"
        handlers["write_file"](path=str(target), content="ok")
        self.assertEqual(handlers["read_file"](path=str(target)), "ok")

    def test_world_summary_tool_returns_json(self) -> None:
        import json as _json
        wm, schemas, handlers = self._make(with_world=True)
        handlers["write_file"](path=str(self.tmp / "z"), content="x")
        out = handlers["world_summary"]()
        parsed = _json.loads(out)
        self.assertIn("entity_counts", parsed)
        self.assertEqual(parsed["entity_counts"]["file"], 1)
        # Schema is exposed only when world_model is set.
        names = {s["name"] for s in schemas}
        self.assertIn("world_summary", names)
        self.assertIn("world_known", names)


if __name__ == "__main__":
    unittest.main()
