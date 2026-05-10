"""Tests for learner.skills."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.skills import SkillLibrary, make_skill_tools


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.lib = SkillLibrary(path=self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_and_read_roundtrip(self):
        s = self.lib.write("git-rebase", "a rebase has gone wrong", "1. git reflog\n2. checkout that hash")
        again = self.lib.get("git-rebase")
        self.assertIsNotNone(again)
        self.assertEqual(again.name, "git-rebase")
        self.assertIn("git reflog", again.body)
        self.assertEqual(again.when, "a rebase has gone wrong")

    def test_search_by_keyword(self):
        self.lib.write("rebase-recovery", "git rebase has gone wrong", "...")
        self.lib.write("merge-conflict", "merge conflict needs resolution", "...")
        self.lib.write("postgres-tuning", "postgres slow query", "...")
        hits = self.lib.search("rebase")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "rebase-recovery")

    def test_search_returns_empty_for_no_match(self):
        self.lib.write("alpha", "first thing", "body")
        self.assertEqual(self.lib.search("kubernetes"), [])

    def test_search_empty_query(self):
        self.lib.write("alpha", "first thing", "body")
        self.assertEqual(self.lib.search(""), [])

    def test_reload_picks_up_external_files(self):
        # Drop a markdown file directly, no frontmatter.
        (self.tmp / "ad-hoc.md").write_text("just a body without frontmatter")
        lib = SkillLibrary(path=self.tmp)
        self.assertIsNotNone(lib.get("ad-hoc"))

    def test_delete_removes_file(self):
        self.lib.write("temp", "to be deleted", "body")
        self.assertTrue(self.lib.delete("temp"))
        self.assertIsNone(self.lib.get("temp"))
        # Idempotent
        self.assertFalse(self.lib.delete("temp"))

    def test_addendum_index_when_no_query(self):
        self.lib.write("a", "use when alpha", "x")
        self.lib.write("b", "use when beta", "y")
        s = self.lib.system_prompt_addendum(query=None)
        self.assertIn("Available skills", s)
        self.assertIn("a:", s)
        self.assertIn("b:", s)

    def test_addendum_with_query(self):
        self.lib.write("a", "git rebase has gone wrong", "x")
        self.lib.write("b", "postgres slow query", "y")
        s = self.lib.system_prompt_addendum(query="rebase")
        self.assertIn("skill: a", s)
        self.assertNotIn("skill: b", s)

    def test_addendum_empty_when_no_skills(self):
        self.assertEqual(self.lib.system_prompt_addendum(), "")

    def test_save_skill_overwrites_existing_with_same_name(self):
        self.lib.write("dup", "v1", "first")
        self.lib.write("dup", "v2", "second")
        self.assertEqual(len([s for s in self.lib.all() if s.name == "dup"]), 1)
        self.assertEqual(self.lib.get("dup").when, "v2")


class TestSkillTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=Path(self._tmp.name))
        self.schemas, self.handlers = make_skill_tools(self.lib)

    def tearDown(self):
        self._tmp.cleanup()

    def test_schemas_match_handlers(self):
        self.assertEqual({s["name"] for s in self.schemas}, set(self.handlers))

    def test_save_then_list(self):
        out = self.handlers["save_skill"](name="foo", when="user says foo", body="say bar")
        self.assertIn("foo", out)
        listed = self.handlers["list_skills"]()
        self.assertIn("foo", listed)

    def test_search_via_tool(self):
        self.handlers["save_skill"](name="rebase-recovery", when="git rebase mishap", body="x")
        self.handlers["save_skill"](name="docker-compose", when="container orchestration", body="x")
        out = self.handlers["search_skills"](query="rebase")
        self.assertIn("rebase-recovery", out)
        self.assertNotIn("docker-compose", out)

    def test_read_missing(self):
        self.assertIn("not found", self.handlers["read_skill"](name="missing"))


if __name__ == "__main__":
    unittest.main()
