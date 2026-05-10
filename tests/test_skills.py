"""Tests for the skill library."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.skills import Skill, SkillLibrary, parse_skill


class TestParseSkill(unittest.TestCase):
    def test_parses_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.md"
            p.write_text(
                "---\nname: solve-quadratic\ndescription: Solve quadratics.\n"
                "tags: [math, algebra]\n---\n\n## When to use\nA quadratic.\n"
            )
            s = parse_skill(p)
            self.assertEqual(s.name, "solve-quadratic")
            self.assertEqual(s.description, "Solve quadratics.")
            self.assertEqual(s.tags, ["math", "algebra"])
            self.assertIn("When to use", s.body)

    def test_falls_back_to_filename_and_first_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fallback.md"
            p.write_text("# A skill description\n\nbody here\n")
            s = parse_skill(p)
            self.assertEqual(s.name, "fallback")
            self.assertEqual(s.description, "A skill description")


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_library_returns_nothing(self):
        self.assertEqual(self.lib.all(), [])
        self.assertEqual(self.lib.search("anything"), [])
        self.assertEqual(self.lib.render_for_prompt("anything"), "")

    def test_write_and_search(self):
        self.lib.write(
            name="solve-quadratic",
            description="Solve quadratic equations using the formula.",
            body="1. Identify a, b, c.\n2. Compute discriminant.",
            tags=["math", "algebra"],
        )
        self.lib.write(
            name="fetch-and-summarize",
            description="Fetch a URL and summarize its contents.",
            body="Use web_fetch then write a summary.",
            tags=["web", "summary"],
        )
        hits = self.lib.search("quadratic equation")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "solve-quadratic")
        hits = self.lib.search("summarize a web page")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "fetch-and-summarize")

    def test_render_for_prompt_includes_skill_names(self):
        self.lib.write(name="alpha", description="Do alpha things.", body="alpha body")
        block = self.lib.render_for_prompt("alpha")
        self.assertIn("Skill: alpha", block)
        self.assertIn("alpha body", block)

    def test_search_ignores_short_query_terms(self):
        # Single-character query terms shouldn't match everything.
        self.lib.write(name="topic", description="Some topic.", body="x")
        self.assertEqual(self.lib.search("a"), [])

    def test_malformed_files_dont_break_library(self):
        bad = Path(self._tmp.name) / "bad.md"
        bad.write_bytes(b"\xff\xfe\x00\x00not valid utf8")
        self.lib.write(name="good", description="A good skill.", body="ok")
        # all() should skip the malformed one but still return the good one.
        names = [s.name for s in self.lib.all()]
        self.assertIn("good", names)

    def test_safe_filename(self):
        s = self.lib.write(name="Has Spaces & Slashes!", description="x", body="y")
        self.assertTrue(s.path.exists())
        self.assertNotIn(" ", s.path.name)


if __name__ == "__main__":
    unittest.main()
