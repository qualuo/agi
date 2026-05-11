"""Tests for the skill library."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import Skill, SkillLibrary


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_round_trip(self):
        self.lib.save(
            name="summarize-file",
            description="Read a file and produce a 3-bullet summary.",
            body="1. read_file\n2. compress to 3 bullets\n3. return",
            tags=["io", "summarization"],
        )
        loaded = self.lib.load("summarize-file")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.name, "summarize-file")
        self.assertEqual(loaded.tags, ["io", "summarization"])
        self.assertIn("read_file", loaded.body)
        self.assertIn("compress", loaded.body)

    def test_save_preserves_success_count(self):
        self.lib.save("foo", "desc", "body")
        self.lib.record_success("foo")
        self.lib.record_success("foo")
        self.lib.save("foo", "desc2", "body2")  # update preserves count
        loaded = self.lib.load("foo")
        assert loaded is not None
        self.assertEqual(loaded.success_count, 2)
        self.assertEqual(loaded.description, "desc2")

    def test_search_by_keyword(self):
        self.lib.save("a", "How to fizzbuzz", "...", tags=["math"])
        self.lib.save("b", "Compute factorials", "...", tags=["math"])
        self.lib.save("c", "Email parsing", "...", tags=["text"])
        hits = self.lib.search("fizzbuzz")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "a")

    def test_search_breaks_ties_on_success_count(self):
        self.lib.save("a", "common task", "...", tags=["x"])
        self.lib.save("b", "common task", "...", tags=["x"])
        self.lib.record_success("b")
        hits = self.lib.search("common task")
        self.assertEqual(hits[0].name, "b")

    def test_load_missing_returns_none(self):
        self.assertIsNone(self.lib.load("nope"))

    def test_render_includes_name_and_body(self):
        s = Skill(name="x", description="d", body="step1\nstep2", tags=["t"])
        rendered = s.render()
        self.assertIn("x", rendered)
        self.assertIn("d", rendered)
        self.assertIn("step1", rendered)
        self.assertIn("t", rendered)

    def test_all_returns_every_skill(self):
        self.lib.save("a", "a", "a")
        self.lib.save("b", "b", "b")
        names = {s.name for s in self.lib.all()}
        self.assertEqual(names, {"a", "b"})

    def test_delete(self):
        self.lib.save("a", "a", "a")
        self.assertTrue(self.lib.delete("a"))
        self.assertIsNone(self.lib.load("a"))
        self.assertFalse(self.lib.delete("a"))

    def test_slug_handles_unsafe_names(self):
        # Skills get filename-slugged; weird names still round-trip.
        self.lib.save("My Cool Skill!", "d", "b")
        # The same name (case-insensitive, punctuation-stripped) finds it.
        loaded = self.lib.load("My Cool Skill!")
        self.assertIsNotNone(loaded)

    def test_persists_across_instances(self):
        self.lib.save("x", "d", "b")
        other = SkillLibrary(path=self._tmp.name)
        self.assertIsNotNone(other.load("x"))


if __name__ == "__main__":
    unittest.main()
