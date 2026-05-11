"""Skill library tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import Skill, SkillLibrary, parse_skill, render_skill


SAMPLE = """\
---
name: sql-migration
description: how to safely add a NOT NULL column to a large table
tags: [postgres, ops]
---
1. Add the column NULLABLE with a default.
2. Backfill in batches.
3. Add the NOT NULL constraint.
"""


class TestParse(unittest.TestCase):
    def test_parse_front_matter(self):
        s = parse_skill(SAMPLE)
        self.assertEqual(s.name, "sql-migration")
        self.assertIn("Backfill in batches", s.body)
        self.assertEqual(s.tags, ["postgres", "ops"])

    def test_parse_no_front_matter(self):
        s = parse_skill("just a body, no metadata")
        self.assertEqual(s.name, "unnamed")
        self.assertIn("just a body", s.body)

    def test_render_roundtrip(self):
        s = parse_skill(SAMPLE)
        rendered = render_skill(s)
        s2 = parse_skill(rendered)
        self.assertEqual(s.name, s2.name)
        self.assertEqual(s.description, s2.description)
        self.assertEqual(s.tags, s2.tags)

    def test_match_score_overlap(self):
        s = parse_skill(SAMPLE)
        self.assertGreater(s.matches_score("safely add a NOT NULL column"), 0)
        self.assertEqual(s.matches_score("kitten photography"), 0)


class TestLibrary(unittest.TestCase):
    def test_save_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.save(Skill(name="hello", description="say hi", body="print('hi')"))
            lib.save(Skill(name="bye", description="say bye", body="print('bye')"))
            names = sorted(s.name for s in lib.list())
            self.assertEqual(names, ["bye", "hello"])

    def test_retrieve_by_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.save(Skill(name="migration", description="postgres NOT NULL column add", body="..."))
            lib.save(Skill(name="search", description="elasticsearch tuning", body="..."))
            hits = lib.retrieve("how do I add a postgres column safely?", k=2)
            self.assertTrue(hits)
            self.assertEqual(hits[0].name, "migration")

    def test_retrieve_no_match_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.save(Skill(name="m", description="postgres ops", body="..."))
            self.assertEqual(lib.retrieve("baking cakes"), [])

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.save(Skill(name="x", description="d", body="b"))
            self.assertTrue(lib.delete("x"))
            self.assertEqual(lib.list(), [])
            self.assertFalse(lib.delete("missing"))

    def test_empty_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            self.assertEqual(lib.list(), [])
            self.assertEqual(lib.retrieve("anything"), [])


if __name__ == "__main__":
    unittest.main()
