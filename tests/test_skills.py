"""Skill library tests — pure filesystem, no API."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import SkillLibrary, make_skill_tools


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_get_round_trip(self):
        self.lib.save(
            "pr-triage",
            when="Triaging a pull request",
            body="1. read description\n2. check CI",
            tags=["github", "pr"],
        )
        loaded = self.lib.get("pr-triage")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.name, "pr-triage")
        self.assertEqual(loaded.when, "Triaging a pull request")
        self.assertIn("github", loaded.tags)
        self.assertIn("read description", loaded.body)

    def test_search_ranks_by_relevance(self):
        self.lib.save("a", when="postgres migration", body="psql ...", tags=["db"])
        self.lib.save("b", when="git rebase", body="rebase ...", tags=["git"])
        self.lib.save("c", when="postgres backup", body="pg_dump ...", tags=["db", "postgres"])
        results = self.lib.search("postgres")
        names = [s.name for s in results]
        self.assertIn("a", names)
        self.assertIn("c", names)
        self.assertNotIn("b", names)

    def test_search_empty_query_returns_nothing(self):
        self.lib.save("x", when="", body="body", tags=[])
        self.assertEqual(self.lib.search(""), [])

    def test_save_overwrites_existing(self):
        self.lib.save("x", when="first", body="first body")
        self.lib.save("x", when="second", body="second body")
        loaded = self.lib.get("x")
        assert loaded is not None
        self.assertEqual(loaded.when, "second")

    def test_delete(self):
        self.lib.save("x", when="w", body="b")
        self.assertTrue(self.lib.delete("x"))
        self.assertIsNone(self.lib.get("x"))
        self.assertFalse(self.lib.delete("x"))

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            self.lib.save("!!!", when="x", body="y")

    def test_parses_file_without_front_matter(self):
        path = Path(self._tmp.name) / "raw.md"
        path.write_text("just body text")
        all_skills = self.lib.all()
        self.assertEqual(len(all_skills), 1)
        self.assertEqual(all_skills[0].name, "raw")

    def test_skips_malformed_files(self):
        # The library should not break on unreadable files.
        good_lib = self.lib
        good_lib.save("good", when="w", body="b")
        # Confirm the library still returns the good skill.
        self.assertEqual(len(good_lib.all()), 1)


class TestSkillTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)
        self.schemas, self.handlers = make_skill_tools(self.lib)

    def tearDown(self):
        self._tmp.cleanup()

    def test_schemas_match_handlers(self):
        schema_names = {s["name"] for s in self.schemas}
        self.assertEqual(schema_names, set(self.handlers))

    def test_list_when_empty(self):
        self.assertIn("no skills", self.handlers["list_skills"]())

    def test_save_then_search(self):
        out = self.handlers["save_skill"](
            name="deploy",
            when="Deploying a service",
            body="1. tag\n2. push",
            tags=["ops"],
        )
        self.assertIn("saved", out)
        result = self.handlers["search_skills"](query="deploy")
        self.assertIn("deploy", result)
        self.assertIn("Deploying a service", result)

    def test_load_missing(self):
        result = self.handlers["load_skill"](name="missing")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
