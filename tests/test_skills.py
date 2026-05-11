"""SkillLibrary: load, save, search, render."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills import SkillLibrary, _parse_frontmatter, _slug


class TestSlug(unittest.TestCase):
    def test_slug_lowercases(self):
        self.assertEqual(_slug("Hello World"), "hello-world")

    def test_slug_strips_punct(self):
        self.assertEqual(_slug("Foo / Bar / Baz!"), "foo-bar-baz")

    def test_slug_empty_safe(self):
        self.assertEqual(_slug("!!!"), "skill")


class TestFrontmatter(unittest.TestCase):
    def test_parse_basic(self):
        raw = "---\nname: foo\ndescription: bar\ntags: [a, b]\n---\nbody here\n"
        meta, body = _parse_frontmatter(raw)
        self.assertEqual(meta["name"], "foo")
        self.assertEqual(meta["description"], "bar")
        self.assertEqual(meta["tags"], ["a", "b"])
        self.assertEqual(body.strip(), "body here")

    def test_no_frontmatter_returns_raw(self):
        meta, body = _parse_frontmatter("just body\nmore\n")
        self.assertEqual(meta, {})
        self.assertIn("just body", body)


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = SkillLibrary(path=Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_get(self):
        s = self.lib.save(
            name="Resolve git conflicts",
            description="Use when a merge has unresolved conflict markers",
            body="Open the file, find <<<<<<<, choose a side, run git add, then git commit.",
            tags=["git"],
        )
        self.assertIsNotNone(s.path)
        self.assertTrue(s.path.exists())
        self.assertEqual(self.lib.get("Resolve git conflicts").body.strip(),
                         s.body.strip())

    def test_search_finds_by_description_overlap(self):
        self.lib.save(
            name="git-merge",
            description="Resolve unresolved git merge conflicts",
            body="...",
            tags=["git"],
        )
        self.lib.save(
            name="docker-deploy",
            description="Build and push a docker image",
            body="...",
            tags=["docker"],
        )
        hits = self.lib.search("how do I resolve a merge conflict?")
        self.assertEqual(hits[0].name, "git-merge")

    def test_search_returns_empty_on_no_overlap(self):
        self.lib.save(name="x", description="y z", body="...", tags=[])
        self.assertEqual(self.lib.search("alpha beta gamma"), [])

    def test_save_disambiguates_collisions(self):
        a = self.lib.save(name="dup", description="a", body="x", tags=[])
        b = self.lib.save(name="dup", description="b", body="y", tags=[])
        self.assertNotEqual(a.path, b.path)

    def test_render_for_prompt_empty_when_no_hits(self):
        self.assertEqual(self.lib.render_for_prompt("nothing matches xyzzy"), "")

    def test_render_for_prompt_includes_skill_body(self):
        self.lib.save(
            name="k8s-rollback",
            description="rollback a kubernetes deployment",
            body="kubectl rollout undo deployment/<name>",
            tags=["k8s"],
        )
        out = self.lib.render_for_prompt("how do I rollback the kubernetes deployment?")
        self.assertIn("k8s-rollback", out)
        self.assertIn("kubectl rollout undo", out)

    def test_persists_across_instances(self):
        path = Path(self._tmp.name)
        SkillLibrary(path).save(name="a", description="b c d", body="body", tags=[])
        again = SkillLibrary(path)
        self.assertEqual(len(again.all()), 1)


if __name__ == "__main__":
    unittest.main()
