"""Tests for agi.skills."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import SkillLibrary, _parse_sections, _slugify


class TestSkillLibrary(unittest.TestCase):
    def test_add_then_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            s = lib.add(
                name="run pytest",
                when_to_use="user asks to run the test suite",
                procedure="1. cd to repo root\n2. run `pytest -q`\n3. summarize failures",
                failure_modes="missing deps → suggest `pip install -e .`",
            )
            self.assertTrue(s.path.exists())
            loaded = lib.get("run pytest")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.name, "run pytest")
            self.assertIn("cd to repo root", loaded.procedure)

    def test_search_finds_by_when_to_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.add("alpha", "user asks about pytest failures", "do x")
            lib.add("beta", "user asks for code review", "do y")
            results = lib.search("how do I debug pytest failures?")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].name, "alpha")

    def test_search_short_query_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.add("alpha", "do alpha", "step")
            self.assertEqual(lib.search("a"), [])

    def test_mark_used_increments_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.add("alpha", "do alpha thing", "step")
            lib.mark_used("alpha")
            lib.mark_used("alpha")
            loaded = lib.get("alpha")
            self.assertEqual(loaded.usage_count, 2)
            self.assertGreater(loaded.last_used_ts, 0)

    def test_archive_moves_skill_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.add("alpha", "do alpha thing", "step")
            self.assertTrue(lib.archive("alpha"))
            self.assertIsNone(lib.get("alpha"))
            self.assertTrue((Path(tmp) / "archive" / "alpha.md").exists())

    def test_to_prompt_block_is_compact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            s = lib.add("alpha", "trigger condition", "do this", "fail like that")
            block = s.to_prompt_block()
            self.assertIn("skill: alpha", block)
            self.assertIn("trigger condition", block)


class TestParseSections(unittest.TestCase):
    def test_parses_three_sections(self) -> None:
        md = "# my skill\n\n## When to use\ntrigger\n\n## Procedure\nstep one\nstep two\n\n## Failure modes\nfail here"
        out = _parse_sections(md)
        self.assertEqual(out["name"], "my skill")
        self.assertEqual(out["when to use"], "trigger")
        self.assertIn("step one", out["procedure"])
        self.assertEqual(out["failure modes"], "fail here")

    def test_slugify_handles_special_chars(self) -> None:
        self.assertEqual(_slugify("Run Pytest!"), "run-pytest")
        self.assertEqual(_slugify("   "), "skill")


if __name__ == "__main__":
    unittest.main()
