"""Tests for the skill library and its engine integration."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.skills import Skill, SkillLibrary, parse_skill, render_skill
from runtime.backend import MockBackend
from runtime.engine import Engine


SAMPLE = """\
---
name: greet
when: user says hello
tags: [greeting, simple]
---
# Procedure
Respond with a friendly greeting.
"""


class TestParse(unittest.TestCase):
    def test_parses_frontmatter(self):
        s = parse_skill(SAMPLE)
        self.assertEqual(s.name, "greet")
        self.assertEqual(s.when, "user says hello")
        self.assertEqual(s.tags, ["greeting", "simple"])
        self.assertIn("Respond with a friendly", s.body)

    def test_roundtrip(self):
        s = parse_skill(SAMPLE)
        rendered = render_skill(s)
        s2 = parse_skill(rendered)
        self.assertEqual(s.name, s2.name)
        self.assertEqual(s.tags, s2.tags)
        self.assertEqual(s.body, s2.body)

    def test_tolerates_missing_frontmatter(self):
        s = parse_skill("just a body", default_name="raw")
        self.assertEqual(s.name, "raw")
        self.assertEqual(s.body, "just a body")
        self.assertEqual(s.tags, [])


class TestLibrary(unittest.TestCase):
    def test_add_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.add_from_text(
                name="greet",
                when="user says hello",
                body="say hi back",
                tags=["greeting"],
            )
            s = lib.get("greet")
            self.assertIsNotNone(s)
            self.assertEqual(s.when, "user says hello")
            self.assertEqual(s.tags, ["greeting"])

    def test_retrieve_by_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.add_from_text(name="shell", when="run shell command", body="bash stuff", tags=["shell"])
            lib.add_from_text(name="math", when="solve arithmetic", body="addition subtraction", tags=["math"])
            hits = lib.retrieve("how do I run a shell command to list files?")
            self.assertTrue(hits)
            self.assertEqual(hits[0].name, "shell")
            hits = lib.retrieve("addition problem")
            self.assertEqual(hits[0].name, "math")

    def test_retrieve_empty_query_returns_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.add_from_text(name="x", when="x", body="x", tags=[])
            self.assertEqual(lib.retrieve(""), [])

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.add_from_text(name="doomed", when="never", body="x")
            self.assertIsNotNone(lib.get("doomed"))
            self.assertTrue(lib.remove("doomed"))
            self.assertIsNone(lib.get("doomed"))
            self.assertFalse(lib.remove("doomed"))  # already gone

    def test_propose_from_trace_returns_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            draft = lib.propose_from_trace(
                "summarize a long file efficiently",
                "Use head -50, then read carefully, then write a 3-line summary.",
            )
            self.assertIsInstance(draft, Skill)
            # propose does NOT auto-write
            self.assertIsNone(lib.get(draft.name))


class TestEngineSkillInjection(unittest.TestCase):
    def test_skills_injected_into_system_prompt(self):
        """If the library has a relevant skill, the engine surfaces a
        skills_loaded event and the skill body appears in the system prompt
        passed to the backend."""
        captured: list[dict] = []

        def responder(messages):
            return MockBackend.text("done")

        backend = MockBackend(responder=responder)

        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.add_from_text(
                name="summarize_file",
                when="user asks to summarize a file",
                body="Read with head, identify topic, write 3 lines.",
                tags=["summary"],
            )

            engine = Engine(backend=backend, skill_library=lib)
            try:
                task = engine.submit("Please summarize the README file.")
                task.wait(timeout=5)
                self.assertEqual(task.result, "done")

                kinds = [e.kind for e in task.events()]
                self.assertIn("skills_loaded", kinds)

                # Inspect what was actually sent to the backend
                self.assertTrue(backend.calls)
                system = backend.calls[0]["system"]
                system_text = "".join(s["text"] for s in system) if isinstance(system, list) else str(system)
                self.assertIn("summarize_file", system_text)
                self.assertIn("Read with head", system_text)
            finally:
                engine.shutdown()

    def test_unrelated_query_loads_no_skills(self):
        backend = MockBackend.echo("ok")
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(path=tmp)
            lib.add_from_text(name="quantum", when="quantum mechanics", body="QM stuff", tags=["physics"])
            engine = Engine(backend=backend, skill_library=lib)
            try:
                task = engine.submit("Tell me about pizza.")
                task.wait(timeout=5)
                kinds = [e.kind for e in task.events()]
                self.assertNotIn("skills_loaded", kinds)
            finally:
                engine.shutdown()


if __name__ == "__main__":
    unittest.main()
