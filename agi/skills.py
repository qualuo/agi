"""Skill library — procedural memory.

A skill is a markdown file with a 1-line description and a body. The
description is what we retrieve against; the body is what gets injected
into the system prompt when the skill is selected for a task.

This is the medium-timescale learning channel from ARCHITECTURE.md: when
the agent solves a class of task, it can write a skill so the next
instance is cheaper. Retrieval is keyword-overlap on the description,
which is good enough until we wire embeddings.

File format:
    ---
    name: short-name
    description: one-line trigger description
    tags: [foo, bar]
    ---
    # Skill body in markdown.
    Steps, gotchas, examples — whatever helps the next run.

Files live under `~/.agi/skills/` by default. One file per skill.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Skill:
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    path: Path | None = None
    used: int = 0

    def render(self) -> str:
        """Format for injection into the system prompt."""
        head = f"## Skill: {self.name}\n_{self.description}_\n\n"
        return head + self.body.strip() + "\n"


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:60] or "skill"


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta_block, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in meta_block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [t.strip() for t in v[1:-1].split(",") if t.strip()]
        meta[k.strip()] = v
    return meta, body


class SkillLibrary:
    """Directory of skill markdown files."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for f in sorted(self.path.glob("*.md")):
            try:
                raw = f.read_text()
            except OSError:
                continue
            meta, body = _parse_frontmatter(raw)
            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            skills.append(
                Skill(
                    name=meta.get("name") or f.stem,
                    description=meta.get("description") or "",
                    body=body,
                    tags=list(tags),
                    path=f,
                )
            )
        return skills

    def save(
        self,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Skill:
        slug = _slug(name)
        path = self.path / f"{slug}.md"
        if path.exists() and not overwrite:
            # disambiguate with a suffix so we don't silently clobber
            path = self.path / f"{slug}-{int(time.time())}.md"
        tag_list = tags or []
        front = (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"tags: [{', '.join(tag_list)}]\n"
            "---\n"
        )
        path.write_text(front + body.rstrip() + "\n")
        return Skill(name=name, description=description, body=body, tags=list(tag_list), path=path)

    def search(self, query: str, k: int = 3) -> list[Skill]:
        """Rank skills by token overlap against (description + tags + name).

        Cheap and dependency-free; embeddings can replace this behind the
        same interface later.
        """
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for s in self.all():
            hay_text = f"{s.name} {s.description} {' '.join(s.tags)}".lower()
            hay_terms = set(re.findall(r"[a-z0-9]+", hay_text))
            score = len(terms & hay_terms)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def get(self, name: str) -> Skill | None:
        for s in self.all():
            if s.name == name:
                return s
        return None

    def render_for_prompt(self, query: str, k: int = 3) -> str:
        """Format the top-K skills as a system-prompt section. Empty if none."""
        hits = self.search(query, k)
        if not hits:
            return ""
        body = "\n\n".join(s.render() for s in hits)
        return (
            "# Relevant skills (procedural memory)\n"
            "These are SOPs you've previously distilled. Use them when applicable.\n\n"
            + body
        )
