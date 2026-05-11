"""Skill library.

A skill is a small named procedure with a description, a body, and optional
preconditions. It's a markdown file with YAML-ish frontmatter, written by
the agent (or a human) when a task pattern is worth memorising.

The library exists so the same task family doesn't re-pay the planning cost
on every attempt. Stage 2 of the plan calls for "$/passed-task should fall
on repeat workloads" — this is the mechanism.

On-disk layout (default ~/.agi/skills/):

    ~/.agi/skills/
      summarize-file.md
      run-fizzbuzz.md
      ...

File format:

    ---
    name: summarize-file
    description: Read a file and produce a 3-bullet summary.
    tags: [io, summarization]
    success_count: 4
    ---

    1. Read the file with `read_file`.
    2. Compress to <=3 bullets.
    3. Return the bullets verbatim.

The retrieval layer is keyword for now — slot a real embedding model in
later behind `SkillLibrary.search`.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


_FILENAME_OK = re.compile(r"[^a-z0-9_\-]+")


@dataclass
class Skill:
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def render(self) -> str:
        """Format the skill for injection into a system prompt."""
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return (
            f"### {self.name}{tag_str}\n"
            f"{self.description}\n\n"
            f"{self.body.strip()}\n"
        )

    def to_markdown(self) -> str:
        tags = "[" + ", ".join(self.tags) + "]" if self.tags else "[]"
        return (
            "---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"tags: {tags}\n"
            f"success_count: {self.success_count}\n"
            f"created_at: {self.created_at}\n"
            f"updated_at: {self.updated_at}\n"
            "---\n\n"
            f"{self.body.strip()}\n"
        )


def _slug(name: str) -> str:
    s = name.strip().lower().replace(" ", "-")
    return _FILENAME_OK.sub("", s) or "skill"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    header = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    meta: dict = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    return meta, body


def _parse_tags(s: str) -> list[str]:
    s = s.strip().strip("[]")
    if not s:
        return []
    return [t.strip().strip("'\"") for t in s.split(",") if t.strip()]


class SkillLibrary:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def _file_for(self, name: str) -> Path:
        return self.path / f"{_slug(name)}.md"

    def save(
        self,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
    ) -> Skill:
        existing = self.load(name)
        now = time.time()
        skill = Skill(
            name=name,
            description=description,
            body=body,
            tags=tags or (existing.tags if existing else []),
            success_count=existing.success_count if existing else 0,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._file_for(name).write_text(skill.to_markdown())
        return skill

    def load(self, name: str) -> Skill | None:
        path = self._file_for(name)
        if not path.exists():
            return None
        text = path.read_text()
        meta, body = _parse_frontmatter(text)
        try:
            success_count = int(meta.get("success_count", "0"))
        except ValueError:
            success_count = 0
        try:
            created_at = float(meta.get("created_at", "0") or 0)
        except ValueError:
            created_at = 0.0
        try:
            updated_at = float(meta.get("updated_at", "0") or 0)
        except ValueError:
            updated_at = 0.0
        return Skill(
            name=meta.get("name") or name,
            description=meta.get("description", ""),
            body=body.strip(),
            tags=_parse_tags(meta.get("tags", "")),
            success_count=success_count,
            created_at=created_at,
            updated_at=updated_at,
        )

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for path in sorted(self.path.glob("*.md")):
            s = self.load(path.stem)
            if s is not None:
                skills.append(s)
        return skills

    def search(self, query: str, k: int = 3) -> list[Skill]:
        """Keyword search over name+description+tags.

        Lightweight on purpose; replace with embeddings later behind this
        same signature.
        """
        terms = [t for t in query.lower().split() if t]
        scored: list[tuple[int, Skill]] = []
        for s in self.all():
            hay = (s.name + " " + s.description + " " + " ".join(s.tags)).lower()
            score = sum(hay.count(t) for t in terms)
            # Tie-breaker: prefer skills that have actually worked before.
            score = score * 10 + min(s.success_count, 9)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def record_success(self, name: str) -> Skill | None:
        s = self.load(name)
        if s is None:
            return None
        s.success_count += 1
        s.updated_at = time.time()
        self._file_for(name).write_text(s.to_markdown())
        return s

    def delete(self, name: str) -> bool:
        path = self._file_for(name)
        if path.exists():
            path.unlink()
            return True
        return False
