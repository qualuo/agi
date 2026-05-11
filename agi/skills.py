"""Skill library — durable procedural memory.

A skill is a markdown snippet describing how to do a class of task: when
to use it, the procedure, known failure modes. The runtime injects the
relevant skill(s) at the top of the prompt; the reasoning core treats
them as standing instructions.

Three retrieval modes:
- explicit:  Task.skills lists skill names; load exactly those.
- keyword:   `find(query)` scores by term overlap against name+description.
- all:       `list()` returns everything (small N early on).

Storage: a flat directory of `<name>.md` files. The first line is the
description (used for retrieval); the rest is the procedure. This format
was chosen because it's trivially editable by hand — humans can review
and curate the skill set, which is the bottleneck once the agent starts
proposing new skills automatically.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str

    def render(self) -> str:
        return f"## skill: {self.name}\n{self.description}\n\n{self.body}".strip()


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class SkillLibrary:
    """Filesystem-backed skill store. Default location is
    `~/.agi/skills/`. Pass `path=` to use a different directory (tests do)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, description: str, body: str) -> Skill:
        if not _NAME_RE.match(name):
            raise ValueError(f"skill name must match {_NAME_RE.pattern!r}, got {name!r}")
        skill = Skill(name=name, description=description.strip(), body=body.strip())
        contents = f"{skill.description}\n\n{skill.body}\n"
        (self.path / f"{name}.md").write_text(contents)
        return skill

    def get(self, name: str) -> Skill | None:
        p = self.path / f"{name}.md"
        if not p.exists():
            return None
        text = p.read_text()
        lines = text.splitlines()
        description = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]).strip()
        return Skill(name=name, description=description, body=body)

    def list(self) -> list[Skill]:
        skills = []
        for p in sorted(self.path.glob("*.md")):
            skill = self.get(p.stem)
            if skill is not None:
                skills.append(skill)
        return skills

    def find(self, query: str, k: int = 3) -> list[Skill]:
        q = query.lower().strip()
        terms = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 1]
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for skill in self.list():
            hay = (skill.name + " " + skill.description).lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def delete(self, name: str) -> bool:
        p = self.path / f"{name}.md"
        if p.exists():
            p.unlink()
            return True
        return False

    def render(self, names: list[str]) -> str:
        """Render a set of skills into a single prompt-ready block. Skip
        unknown names silently — coordinators may ask for skills that
        were removed; better to proceed without them than fail the task."""
        parts: list[str] = []
        for name in names:
            skill = self.get(name)
            if skill is None:
                continue
            parts.append(skill.render())
        if not parts:
            return ""
        return "[loaded skills]\n\n" + "\n\n---\n\n".join(parts)
