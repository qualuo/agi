"""Tests for the skill library."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import SkillLibrary


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.lib = SkillLibrary(path=self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_get(self):
        self.lib.save("foo_bar", "Do the foo bar thing", "Step 1: do it.")
        s = self.lib.get("foo_bar")
        assert s is not None
        self.assertEqual(s.name, "foo_bar")
        self.assertEqual(s.description, "Do the foo bar thing")
        self.assertIn("Step 1", s.body)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.lib.get("nope"))

    def test_invalid_names_rejected(self):
        for bad in ["FooBar", "1foo", "foo bar", "foo!", "", "a" * 100]:
            with self.assertRaises(ValueError):
                self.lib.save(bad, "x", "y")

    def test_list_alpha_order(self):
        self.lib.save("zeta", "Zeta", "z")
        self.lib.save("alpha", "Alpha", "a")
        self.lib.save("mid", "Mid", "m")
        names = [s.name for s in self.lib.list()]
        self.assertEqual(names, ["alpha", "mid", "zeta"])

    def test_find_keyword_match(self):
        self.lib.save("greet_user", "How to greet a user politely", "say hi")
        self.lib.save("sort_csv", "How to sort a CSV file", "use pandas")
        found = self.lib.find("greet")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "greet_user")

    def test_find_ranks_by_overlap(self):
        self.lib.save("file_read", "Read a file from disk", "open it")
        self.lib.save("file_write", "Write a file to disk", "open it")
        # "file disk" matches both equally; not asserting order, just count.
        found = self.lib.find("file disk")
        self.assertEqual(len(found), 2)

    def test_find_empty_query_returns_empty(self):
        self.lib.save("a", "desc", "body")
        self.assertEqual(self.lib.find(""), [])

    def test_delete(self):
        self.lib.save("temp", "Temporary", "ephemeral")
        self.assertTrue(self.lib.delete("temp"))
        self.assertIsNone(self.lib.get("temp"))
        self.assertFalse(self.lib.delete("temp"))

    def test_render_skips_unknown(self):
        self.lib.save("known", "Known skill", "body")
        rendered = self.lib.render(["known", "missing"])
        self.assertIn("skill: known", rendered)
        self.assertNotIn("missing", rendered)

    def test_render_empty(self):
        self.assertEqual(self.lib.render([]), "")
        self.assertEqual(self.lib.render(["nothing-saved"]), "")


if __name__ == "__main__":
    unittest.main()
