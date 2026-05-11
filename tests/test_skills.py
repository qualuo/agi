"""Tests for the skill library."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import Skill, SkillLibrary, _parse_frontmatter


class TestFrontmatter(unittest.TestCase):
    def test_parses_basic(self):
        text = "---\nname: foo\ndescription: bar\ntags: [a, b]\n---\nbody here"
        meta, body = _parse_frontmatter(text)
        self.assertEqual(meta["name"], "foo")
        self.assertEqual(meta["description"], "bar")
        self.assertEqual(meta["tags"], ["a", "b"])
        self.assertEqual(body, "body here")

    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("hello world")
        self.assertEqual(meta, {})
        self.assertEqual(body, "hello world")

    def test_quotes_stripped(self):
        meta, _ = _parse_frontmatter("---\nname: \"quoted\"\n---\n")
        self.assertEqual(meta["name"], "quoted")


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load(self):
        s = Skill(name="solve_addition", description="add two ints", body="1) parse\n2) add\n3) emit", tags=["math"])
        self.lib.save(s)
        loaded = self.lib.get("solve_addition")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.description, "add two ints")
        self.assertEqual(loaded.tags, ["math"])
        self.assertIn("parse", loaded.body)

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            self.lib.save(Skill(name="bad name", description="", body=""))

    def test_retrieve_finds_relevant(self):
        self.lib.save(Skill(name="add", description="arithmetic addition", body="...", tags=["math"]))
        self.lib.save(Skill(name="bake_cake", description="baking instructions", body="...", tags=["food"]))
        results = self.lib.retrieve("how do I do addition", k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].name, "add")

    def test_retrieve_returns_nothing_when_no_overlap(self):
        self.lib.save(Skill(name="bake_cake", description="baking instructions", body="..."))
        results = self.lib.retrieve("astrophysics", k=2)
        self.assertEqual(results, [])

    def test_delete(self):
        self.lib.save(Skill(name="x", description="", body=""))
        self.assertTrue(self.lib.delete("x"))
        self.assertFalse(self.lib.delete("x"))

    def test_format_for_prompt_includes_all(self):
        s1 = Skill(name="a", description="d1", body="p1")
        s2 = Skill(name="b", description="d2", body="p2")
        out = self.lib.format_for_prompt([s1, s2])
        self.assertIn("Skill: a", out)
        self.assertIn("Skill: b", out)
        self.assertIn("p1", out)
        self.assertIn("p2", out)

    def test_format_for_prompt_empty(self):
        self.assertEqual(self.lib.format_for_prompt([]), "")

    def test_persists_across_instances(self):
        self.lib.save(Skill(name="persists", description="d", body="b"))
        other = SkillLibrary(path=self._tmp.name)
        self.assertIsNotNone(other.get("persists"))


if __name__ == "__main__":
    unittest.main()
