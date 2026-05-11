"""Skill library smoke tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.skills import Skill, SkillLibrary


class TestSkill(unittest.TestCase):
    def test_round_trip(self):
        s = Skill(name="foo", triggers=["a", "b"], body="step one\nstep two")
        parsed = Skill.parse(s.render())
        self.assertEqual(parsed.name, "foo")
        self.assertEqual(parsed.triggers, ["a", "b"])
        self.assertIn("step one", parsed.body)

    def test_parse_without_frontmatter(self):
        s = Skill.parse("just a body")
        self.assertEqual(s.body, "just a body")
        self.assertEqual(s.triggers, [])

    def test_parse_strips_trailing_whitespace(self):
        s = Skill.parse("---\nname: x\ntriggers: [a]\n---\nbody\n\n")
        self.assertEqual(s.body, "body")


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_get(self):
        self.lib.write(Skill(name="hello", triggers=["greet"], body="say hi"))
        got = self.lib.get("hello")
        assert got is not None
        self.assertEqual(got.triggers, ["greet"])
        self.assertIn("say hi", got.body)

    def test_slugifies_names(self):
        self.lib.write(Skill(name="Make Cake!", body="x"))
        # Directory should contain a sluggified filename.
        names = [p.name for p in Path(self.lib.path).iterdir()]
        self.assertIn("make-cake.md", names)

    def test_retrieve_ranks_trigger_hits_higher(self):
        self.lib.write(Skill(name="A", triggers=["python"], body="general"))
        self.lib.write(Skill(name="B", triggers=[], body="python python general"))
        # Trigger match has weight 3, B has 2 body hits → A scores 3, B scores 2.
        results = self.lib.retrieve("python")
        self.assertEqual([s.name for s in results[:1]], ["A"])

    def test_retrieve_empty_query_returns_nothing(self):
        self.lib.write(Skill(name="A", triggers=["x"], body="y"))
        self.assertEqual(self.lib.retrieve("   "), [])

    def test_delete(self):
        self.lib.write(Skill(name="gone", body="x"))
        self.assertTrue(self.lib.delete("gone"))
        self.assertFalse(self.lib.delete("gone"))  # idempotent-ish
        self.assertIsNone(self.lib.get("gone"))

    def test_all_skips_non_markdown(self):
        (Path(self.lib.path) / "readme.txt").write_text("ignore me")
        self.lib.write(Skill(name="real", body="x"))
        names = [s.name for s in self.lib.all()]
        self.assertEqual(names, ["real"])

    def test_malformed_file_does_not_crash(self):
        (Path(self.lib.path) / "bad.md").write_text("---\nmalformed without close")
        # Should still parse (tolerated: no frontmatter found).
        all_skills = self.lib.all()
        # Either parsed as bodyful skill or skipped — must not raise.
        self.assertIsInstance(all_skills, list)


if __name__ == "__main__":
    unittest.main()
