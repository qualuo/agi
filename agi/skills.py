"""Skill library — durable procedural knowledge.

A `Skill` is a markdown file with a YAML-ish front-matter header:

    ---
    name: github-pr-triage
    when: Triaging a GitHub pull request
    tags: github, pr, review
    ---
    1. Read the PR description and diff via gh api...
    2. Check CI status before commenting...
    3. ...

The library keyword-matches the query against name + when + tags + body and
returns the top-K. The agent loads relevant skills into its system context at
the start of a task. Successful task decompositions can be promoted into new
skills by the agent itself via the `save_skill` tool.

This is the "hours" timescale in ARCHITECTURE.md — procedures the agent learns
without retraining the base model. Stored in `~/.agi/skills/` by default.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9-]+")


@dataclass
class Skill:
    name: str
    when: str = ""
    tags: list[str] = field(default_factory=list)
    body: str = ""
    path: Path | None = None

    def to_prompt(self) -> str:
        header = f"## Skill: {self.name}"
        if self.when:
            header += f"\nWhen: {self.when}"
        if self.tags:
            header += f"\nTags: {', '.join(self.tags)}"
        return f"{header}\n\n{self.body.strip()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when": self.when,
            "tags": list(self.tags),
            "body": self.body,
            "path": str(self.path) if self.path else None,
        }


class SkillLibrary:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for md in sorted(self.path.glob("*.md")):
            try:
                skills.append(_parse_skill(md))
            except Exception:
                # skip malformed files — don't break the loop
                continue
        return skills

    def get(self, name: str) -> Skill | None:
        slug = _slugify(name)
        target = self.path / f"{slug}.md"
        if target.exists():
            return _parse_skill(target)
        return None

    def search(self, query: str, k: int = 3) -> list[Skill]:
        q = query.lower().strip()
        if not q:
            return []
        terms = [t for t in re.split(r"\s+", q) if t]
        scored: list[tuple[int, Skill]] = []
        for sk in self.all():
            hay = " ".join([sk.name.lower(), sk.when.lower(), " ".join(sk.tags).lower(), sk.body.lower()])
            # name + when get a weight boost — that's where intent lives
            boost_hay = " ".join([sk.name.lower(), sk.when.lower()])
            score = sum(hay.count(t) + 2 * boost_hay.count(t) for t in terms)
            if score:
                scored.append((score, sk))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def save(
        self,
        name: str,
        when: str,
        body: str,
        tags: list[str] | None = None,
    ) -> Skill:
        slug = _slugify(name)
        if not slug:
            raise ValueError("skill name must contain at least one alphanumeric character")
        target = self.path / f"{slug}.md"
        front: list[str] = [f"name: {name}"]
        if when:
            front.append(f"when: {when}")
        if tags:
            front.append(f"tags: {', '.join(tags)}")
        content = "---\n" + "\n".join(front) + "\n---\n" + body.strip() + "\n"
        target.write_text(content)
        return _parse_skill(target)

    def delete(self, name: str) -> bool:
        slug = _slugify(name)
        target = self.path / f"{slug}.md"
        if target.exists():
            target.unlink()
            return True
        return False


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower().strip()).strip("-")


def _parse_skill(path: Path) -> Skill:
    text = path.read_text()
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return Skill(name=path.stem, body=text.strip(), path=path)

    front = m.group(1)
    body = text[m.end():]
    name = path.stem
    when = ""
    tags: list[str] = []
    for line in front.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "name":
            name = value
        elif key == "when":
            when = value
        elif key == "tags":
            tags = [t.strip() for t in value.split(",") if t.strip()]
    return Skill(name=name, when=when, tags=tags, body=body.strip(), path=path)


def make_skill_tools(
    library: SkillLibrary,
) -> tuple[list[dict], dict[str, Callable[..., str]]]:
    """Expose the library as agent tools: list, search, load, save."""

    def list_skills() -> str:
        skills = library.all()
        if not skills:
            return "(no skills saved yet)"
        lines = [f"- {s.name}: {s.when}" for s in skills]
        return "\n".join(lines)

    def search_skills(query: str, k: int = 3) -> str:
        results = library.search(query, k=k)
        if not results:
            return "(no matching skills)"
        return "\n\n".join(s.to_prompt() for s in results)

    def load_skill(name: str) -> str:
        sk = library.get(name)
        if sk is None:
            return f"error: skill '{name}' not found"
        return sk.to_prompt()

    def save_skill(name: str, when: str, body: str, tags: list[str] | None = None) -> str:
        sk = library.save(name, when, body, tags or [])
        return f"saved skill '{sk.name}' at {sk.path}"

    schemas = [
        {
            "name": "list_skills",
            "description": "List all saved skills (name + when-to-use) so you can decide which to load.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "search_skills",
            "description": "Search the skill library by keyword. Returns full skill bodies of the top matches.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you're trying to do."},
                    "k": {"type": "integer", "default": 3, "description": "Max skills to return."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "load_skill",
            "description": "Load a named skill in full. Use after list_skills surfaces a relevant one.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        {
            "name": "save_skill",
            "description": (
                "Save a new skill (or overwrite an existing one by name) — a markdown "
                "procedure that future tasks of the same shape can reuse. Capture "
                "the procedure, the trigger condition ('when'), and known pitfalls. "
                "Only save skills you've actually verified work."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short kebab-case-ish name."},
                    "when": {"type": "string", "description": "One-sentence trigger condition."},
                    "body": {"type": "string", "description": "Markdown procedure body."},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "when", "body"],
            },
        },
    ]

    handlers: dict[str, Callable[..., str]] = {
        "list_skills": list_skills,
        "search_skills": search_skills,
        "load_skill": load_skill,
        "save_skill": save_skill,
    }
    return schemas, handlers
