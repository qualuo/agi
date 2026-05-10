"""Tests for the skill library — author/parse/search/render."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import SkillLibrary, render_skill_block


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_and_read(self):
        s = self.lib.add(
            title="Roman numeral conversion",
            when="When the user asks to convert integers to Roman numerals.",
            procedure="1. parse int.\n2. greedy subtract from value table.",
            failure_modes="zero and negatives are out of domain.",
            triggers=["roman", "numeral"],
        )
        self.assertTrue(Path(s.path).exists())
        loaded = self.lib.get(s.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.title, "Roman numeral conversion")
        self.assertEqual(loaded.triggers, ["roman", "numeral"])
        self.assertIn("greedy", loaded.procedure)

    def test_search_matches_trigger(self):
        self.lib.add(
            title="A",
            when="for foo tasks",
            procedure="step",
            triggers=["foo"],
        )
        self.lib.add(
            title="B",
            when="for bar tasks",
            procedure="step",
            triggers=["bar"],
        )
        hits = self.lib.search("convert this foo")
        self.assertEqual([s.title for s in hits], ["A"])

    def test_search_empty_returns_nothing(self):
        self.lib.add(title="A", when="x", procedure="y", triggers=["foo"])
        self.assertEqual(self.lib.search(""), [])

    def test_remove(self):
        s = self.lib.add(title="X", when="x", procedure="y")
        self.assertTrue(self.lib.remove(s.id))
        self.assertIsNone(self.lib.get(s.id))
        self.assertFalse(self.lib.remove("nonexistent"))

    def test_render_block_includes_title_and_procedure(self):
        s = self.lib.add(title="ABC", when="always", procedure="just do it")
        block = render_skill_block([s])
        self.assertIn("ABC", block)
        self.assertIn("just do it", block)
        self.assertIn("Relevant skills", block)

    def test_render_block_empty(self):
        self.assertEqual(render_skill_block([]), "")

    def test_malformed_skill_is_skipped(self):
        bad = Path(self.lib.path) / "broken.md"
        bad.write_text("no frontmatter here")
        good = self.lib.add(title="OK", when="x", procedure="y")
        all_skills = self.lib.all()
        self.assertEqual([s.id for s in all_skills], [good.id])


if __name__ == "__main__":
    unittest.main()
