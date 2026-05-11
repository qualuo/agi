"""Skill library tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import Skill, SkillLibrary


class TestSkillRoundTrip(unittest.TestCase):
    def test_markdown_round_trip(self):
        s = Skill(
            name="Debug Flaky Tests",
            description="Reproduce, bisect, fix.",
            body="1. Run the test in a loop.\n2. Bisect by commit.\n3. Land a fix with a regression case.",
            tags=["testing", "debugging"],
        )
        md = s.to_markdown()
        s2 = Skill.from_markdown(md)
        self.assertEqual(s2.name, s.name)
        self.assertEqual(s2.description, s.description)
        self.assertEqual(s2.body.strip(), s.body.strip())
        self.assertEqual(s2.tags, s.tags)

    def test_slug_normalizes(self):
        self.assertEqual(Skill(name="Hello World!!", description="", body="x").slug, "hello-world")


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_then_get(self):
        self.lib.add("Greet User", "Friendly greeting protocol.", "Say hi.", tags=["chat"])
        s = self.lib.get("greet-user")
        self.assertIsNotNone(s)
        self.assertEqual(s.name, "Greet User")

    def test_search_keyword_overlap(self):
        self.lib.add("Debug Flaky Tests", "Reproduce, bisect, fix.", "...", tags=["testing"])
        self.lib.add("Write CRUD APIs", "REST endpoints with validation.", "...", tags=["api"])
        results = self.lib.search("how do I debug a flaky test in CI?")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Debug Flaky Tests")

    def test_render_prompt_increments_uses(self):
        self.lib.add("Search Memory First", "Search memory before answering.", "Search memory.")
        prompt = self.lib.render_prompt("search memory before answering")
        self.assertIn("Search Memory First", prompt)
        s = self.lib.get("search-memory-first")
        self.assertEqual(s.uses, 1)
        # And again — uses should increment.
        self.lib.render_prompt("search memory before answering")
        self.assertEqual(self.lib.get("search-memory-first").uses, 2)

    def test_render_prompt_empty_when_no_match(self):
        self.lib.add("X", "Y", "Z")
        self.assertEqual(self.lib.render_prompt("totally unrelated lorem ipsum dolor sit"), "")

    def test_persists_across_instances(self):
        self.lib.add("Persisted", "Should survive a reopen.", "body")
        lib2 = SkillLibrary(path=self.lib.path)
        all_skills = lib2.all()
        self.assertEqual(len(all_skills), 1)
        self.assertEqual(all_skills[0].name, "Persisted")

    def test_delete(self):
        self.lib.add("Doomed", "Will be deleted.", "body")
        self.assertTrue(self.lib.delete("doomed"))
        self.assertIsNone(self.lib.get("doomed"))
        self.assertFalse(self.lib.delete("doomed"))


if __name__ == "__main__":
    unittest.main()
