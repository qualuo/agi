"""Skill library.

A skill is a named, retrievable procedure stored as a markdown file. When the
agent faces a new task we search the library by keyword overlap with the task
description and inject the top-K skills into the system prompt. Successful task
patterns get compiled back into the library over time.

This is the medium-timescale learning channel from ARCHITECTURE.md — slower
than memory writes, faster than weight updates. Plain markdown so a human can
read, edit, and curate skills directly.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)

# Conservative English stopword set — small enough not to hide real signal.
_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "are", "but", "this", "that",
    "from", "into", "have", "has", "had", "was", "were", "will", "would",
    "can", "could", "should", "how", "why", "what", "when", "where", "who",
    "which", "about", "they", "their", "them", "there", "here", "than",
    "then", "some", "any", "all", "out", "not", "over", "under", "just",
    "very", "really", "much", "more", "most", "such", "also", "too",
    "between", "before", "after", "during", "while",
}


@dataclass
class Skill:
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    uses: int = 0
    ts: float = 0.0

    @property
    def slug(self) -> str:
        return _slugify(self.name)

    def render(self) -> str:
        """Format the skill for injection into the system prompt."""
        return f"## Skill: {self.name}\n\n{self.body.strip()}"

    def to_markdown(self) -> str:
        meta = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "uses": self.uses,
            "ts": self.ts,
        }
        meta_json = json.dumps(meta, indent=2)
        return f"---\n{meta_json}\n---\n{self.body.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str) -> "Skill":
        m = _FRONTMATTER_RE.match(text)
        if not m:
            raise ValueError("skill markdown is missing frontmatter")
        meta = json.loads(m.group(1))
        return cls(
            name=meta["name"],
            description=meta.get("description", ""),
            body=m.group(2).strip(),
            tags=list(meta.get("tags", [])),
            uses=int(meta.get("uses", 0)),
            ts=float(meta.get("ts", 0.0)),
        )


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "skill"


class SkillLibrary:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        name: str,
        description: str,
        body: str,
        tags: Iterable[str] | None = None,
    ) -> Skill:
        skill = Skill(
            name=name,
            description=description,
            body=body,
            tags=list(tags or []),
            ts=time.time(),
        )
        (self.path / f"{skill.slug}.md").write_text(skill.to_markdown())
        return skill

    def get(self, name_or_slug: str) -> Skill | None:
        slug = _slugify(name_or_slug)
        p = self.path / f"{slug}.md"
        if not p.exists():
            return None
        return Skill.from_markdown(p.read_text())

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for p in sorted(self.path.glob("*.md")):
            try:
                skills.append(Skill.from_markdown(p.read_text()))
            except Exception:
                # Skip malformed skills rather than crash the agent.
                continue
        return skills

    def search(self, query: str, k: int = 3) -> list[Skill]:
        """Keyword overlap over name + description + tags. Embeddings later."""
        terms = [t for t in re.split(r"\W+", query.lower())
                 if len(t) >= 3 and t not in _STOPWORDS]
        if not terms:
            return []
        scored: list[tuple[float, Skill]] = []
        for s in self.all():
            hay = f"{s.name} {s.description} {' '.join(s.tags)}".lower()
            # Word-boundary match per term so "api" doesn't half-match "apis".
            score = 0.0
            for t in terms:
                score += len(re.findall(rf"\b{re.escape(t)}\w*", hay))
            if score:
                # Boost frequently-used skills mildly — a soft popularity prior.
                score += min(s.uses, 5) * 0.1
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def mark_used(self, name_or_slug: str) -> None:
        skill = self.get(name_or_slug)
        if skill is None:
            return
        skill.uses += 1
        (self.path / f"{skill.slug}.md").write_text(skill.to_markdown())

    def delete(self, name_or_slug: str) -> bool:
        p = self.path / f"{_slugify(name_or_slug)}.md"
        if p.exists():
            p.unlink()
            return True
        return False

    def render_prompt(self, query: str, k: int = 3) -> str:
        """Top-K skills as a system-prompt block. Empty string if none match."""
        hits = self.search(query, k)
        if not hits:
            return ""
        for s in hits:
            self.mark_used(s.name)
        body = "\n\n".join(s.render() for s in hits)
        return (
            "Skills relevant to the current task (apply when they fit):\n\n"
            f"{body}"
        )
