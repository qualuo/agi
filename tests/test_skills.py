"""Smoke tests for the skill library."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.skills import SkillLibrary, Skill, _parse
from agi.tools import make_tools


class TestSkillParse(unittest.TestCase):
    def test_parses_full_frontmatter(self):
        text = (
            "---\n"
            "name: summarize-file\n"
            "when: Summarize a text file.\n"
            "tags: [io, summarize]\n"
            "---\n"
            "1. read_file\n"
            "2. summarize\n"
        )
        s = _parse(text)
        self.assertIsNotNone(s)
        self.assertEqual(s.name, "summarize-file")
        self.assertEqual(s.when, "Summarize a text file.")
        self.assertEqual(s.tags, ["io", "summarize"])
        self.assertIn("read_file", s.body)

    def test_returns_none_without_frontmatter(self):
        self.assertIsNone(_parse("just a markdown doc"))

    def test_handles_empty_tags(self):
        text = "---\nname: x\nwhen: y\ntags: []\n---\nbody\n"
        s = _parse(text)
        self.assertEqual(s.tags, [])


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_round_trip(self):
        self.lib.save("hello-world", "Greet the user.", "1. say hi\n", tags=["greet"])
        s = self.lib.get("hello-world")
        self.assertIsNotNone(s)
        self.assertEqual(s.when, "Greet the user.")
        self.assertEqual(s.tags, ["greet"])
        self.assertIn("say hi", s.body)

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            self.lib.save("Bad Name", "x", "y")
        with self.assertRaises(ValueError):
            self.lib.save("../escape", "x", "y")

    def test_search_finds_by_when_and_tags(self):
        self.lib.save("write-tests", "Write a unit test for a function.", "body", tags=["test"])
        self.lib.save("read-file", "Read a file from disk.", "body", tags=["io"])
        results = self.lib.search("unit test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "write-tests")

    def test_search_weights_name_higher(self):
        # The first skill has the term "deploy" only in its body. The second
        # has it in name + when. The second should rank higher.
        self.lib.save("alpha", "unrelated trigger", "deploy notes in body", tags=[])
        self.lib.save("deploy-service", "Deploy a service to prod.", "body", tags=[])
        results = self.lib.search("deploy")
        self.assertEqual(results[0].name, "deploy-service")

    def test_delete(self):
        self.lib.save("temp", "transient", "body")
        self.assertTrue(self.lib.delete("temp"))
        self.assertIsNone(self.lib.get("temp"))
        self.assertFalse(self.lib.delete("nope"))

    def test_all_skips_non_md_and_unparseable(self):
        # Drop a non-md file and a malformed md.
        (self.lib.path / "junk.txt").write_text("ignore me")
        (self.lib.path / "broken.md").write_text("no frontmatter here")
        self.lib.save("good", "ok", "body")
        names = [s.name for s in self.lib.all()]
        self.assertEqual(names, ["good"])


class TestSkillTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.memory = Memory(path=tmp / "m.jsonl")
        self.skills = SkillLibrary(path=tmp / "skills")
        self.schemas, self.handlers = make_tools(self.memory, skills=self.skills)

    def tearDown(self):
        self._tmp.cleanup()

    def test_skill_tools_registered_when_lib_provided(self):
        names = {s["name"] for s in self.schemas}
        for n in ("find_skills", "read_skill", "save_skill", "list_skills"):
            self.assertIn(n, names)

    def test_skill_tools_absent_without_lib(self):
        schemas, handlers = make_tools(self.memory, skills=None)
        names = {s["name"] for s in schemas}
        for n in ("find_skills", "read_skill", "save_skill", "list_skills"):
            self.assertNotIn(n, names)
            self.assertNotIn(n, handlers)

    def test_save_then_find_via_tools(self):
        out = self.handlers["save_skill"](
            name="git-blame",
            when="Find who changed a line.",
            body="1. run git blame\n",
            tags=["git"],
        )
        self.assertIn("saved", out)
        result = self.handlers["find_skills"](query="git blame")
        self.assertIn("git-blame", result)

    def test_read_skill_returns_full_markdown(self):
        self.handlers["save_skill"](name="x", when="y", body="z\n")
        out = self.handlers["read_skill"](name="x")
        self.assertIn("---", out)
        self.assertIn("name: x", out)

    def test_schema_set_matches_handlers(self):
        schema_names = {s["name"] for s in self.schemas}
        self.assertEqual(schema_names, set(self.handlers))


if __name__ == "__main__":
    unittest.main()
