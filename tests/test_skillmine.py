"""Tests for skill mining from trace pairs."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skillmine import (
    cluster_traces_by_keyword,
    mine_skills,
    propose_skill_from_cluster,
)
from agi.skills import SkillLibrary


class TestClustering(unittest.TestCase):
    def test_groups_by_leading_keyword(self):
        prompts = [
            "summarize this report",
            "summarize that document",
            "summarize my email",
            "calculate 2+2",
        ]
        clusters = cluster_traces_by_keyword(prompts, min_cluster_size=3)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0]), [0, 1, 2])

    def test_small_clusters_dropped(self):
        prompts = ["sum a", "sum b", "different"]
        clusters = cluster_traces_by_keyword(prompts, min_cluster_size=3)
        self.assertEqual(clusters, [])


class TestPropose(unittest.TestCase):
    def test_candidate_has_required_fields(self):
        prompts = ["summarize X", "summarize Y", "summarize Z"]
        responses = ["X bullet", "Y bullet", "Z bullet"]
        cand = propose_skill_from_cluster(prompts, responses)
        self.assertTrue(cand.suggested_name)
        self.assertGreater(len(cand.suggested_description), 0)
        self.assertIn("Procedure", cand.body)
        self.assertEqual(cand.trace_count, 3)

    def test_candidate_to_skill_round_trips_via_library(self):
        prompts = ["debug error A", "debug error B", "debug error C"]
        responses = ["fix 1", "fix 2", "fix 3"]
        cand = propose_skill_from_cluster(prompts, responses, name_hint="handle_debug")
        skill = cand.to_skill()
        self.assertEqual(skill.name, "handle_debug")

        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.save(skill)
            loaded = lib.get("handle_debug")
            self.assertIsNotNone(loaded)
            self.assertIn("Procedure", loaded.body)


class TestMineSkills(unittest.TestCase):
    def test_end_to_end(self):
        pairs = [
            ("summarize this report", "bullet 1"),
            ("summarize that one too", "bullet 2"),
            ("summarize the third doc", "bullet 3"),
            ("calculate 1+1", "2"),
            ("calculate 3+4", "7"),
        ]
        candidates = mine_skills(pairs, min_cluster_size=3)
        # Only the 'summarize' cluster has ≥3 examples
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertEqual(cand.trace_count, 3)


if __name__ == "__main__":
    unittest.main()
