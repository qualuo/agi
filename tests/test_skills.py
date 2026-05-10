"""Skill library tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.skills.library import Skill, SkillLibrary


def test_skill_library_loads_project_skills():
    lib = SkillLibrary()
    names = {s.name for s in lib.all()}
    assert "research-question" in names
    assert "decompose-goal" in names
    assert "code-with-tests" in names
    assert "reflect-and-save" in names


def test_skill_render_substitutes_args():
    s = Skill(name="t", description="d", body="Search for ${question}",
              args=["question"], tags=[])
    assert s.render({"question": "the cat"}) == "Search for the cat"


def test_skill_retrieve_ranks_by_keyword_match():
    lib = SkillLibrary()
    # "research" appears in the research-question skill (tags + name + description)
    hits = lib.retrieve("research factual question web sources", k=2)
    assert hits
    assert hits[0].name == "research-question"


def test_skill_add_and_reload(tmp_path):
    lib = SkillLibrary(tmp_path)
    s = Skill(name="hello", description="say hi",
              body="Say hello to ${name}",
              args=["name"], tags=["greet"])
    lib.add(s)
    lib2 = SkillLibrary(tmp_path)
    got = lib2.get("hello")
    assert got is not None
    assert got.description == "say hi"
    assert got.render({"name": "World"}) == "Say hello to World"
