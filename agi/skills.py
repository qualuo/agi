"""Skill library — durable procedural memory.

A skill is a markdown file with frontmatter describing when to use it and the
procedure to follow. The agent loads relevant skills into its system prompt
before reasoning. This is the medium-timescale learning channel from
ARCHITECTURE.md §3: when the system solves a novel class of task well, the
procedure gets compiled into a SKILL.md so the next instance is cheaper.

Skill files live in a directory (default `~/.agi/skills/`). Format:

    ---
    name: solve_addition
    description: arithmetic addition of two integers
    tags: [math, arithmetic]
    ---

    Procedure:
    1. Parse the two operands.
    2. Add them.
    3. Return the result with no commentary.

Retrieval today is keyword overlap on `description + tags`. v2 will swap in
embedding similarity behind the same `retrieve()` interface.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    path: Path | None = None
    created_ts: float = field(default_factory=time.time)

    def to_prompt_block(self) -> str:
        tags_str = f" ({', '.join(self.tags)})" if self.tags else ""
        return f"## Skill: {self.name}{tags_str}\n{self.description}\n\n{self.body.strip()}"


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse a tiny YAML-like header: key: value, with list values as [a, b]."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    header, body = match.group(1), match.group(2)
    meta: dict[str, object] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key.strip()] = [
                item.strip().strip("\"'") for item in val[1:-1].split(",") if item.strip()
            ]
        else:
            meta[key.strip()] = val.strip("\"'")
    return meta, body


def _format_frontmatter(skill: Skill) -> str:
    tags = "[" + ", ".join(skill.tags) + "]"
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {skill.description}\n"
        f"tags: {tags}\n"
        "---\n\n"
    )


class SkillLibrary:
    """Directory of markdown skill files with keyword retrieval."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def _load_file(self, p: Path) -> Skill | None:
        try:
            text = p.read_text()
        except (OSError, UnicodeDecodeError):
            return None
        meta, body = _parse_frontmatter(text)
        name = str(meta.get("name") or p.stem)
        description = str(meta.get("description") or "")
        tags_obj = meta.get("tags") or []
        tags = list(tags_obj) if isinstance(tags_obj, list) else []
        return Skill(name=name, description=description, body=body, tags=tags, path=p)

    def all(self) -> list[Skill]:
        out: list[Skill] = []
        for p in sorted(self.path.glob("*.md")):
            s = self._load_file(p)
            if s is not None:
                out.append(s)
        return out

    def get(self, name: str) -> Skill | None:
        for s in self.all():
            if s.name == name:
                return s
        return None

    def save(self, skill: Skill) -> Path:
        if not skill.name or not re.match(r"^[a-zA-Z0-9_-]+$", skill.name):
            raise ValueError(f"invalid skill name: {skill.name!r}")
        target = self.path / f"{skill.name}.md"
        target.write_text(_format_frontmatter(skill) + skill.body.strip() + "\n")
        skill.path = target
        return target

    def delete(self, name: str) -> bool:
        target = self.path / f"{name}.md"
        if target.exists():
            target.unlink()
            return True
        return False

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        """Keyword-overlap ranking. Scores against description + tags + name.

        Conservative: a skill needs at least one keyword hit to be returned.
        """
        q = query.lower()
        terms = [t for t in re.split(r"\W+", q) if t]
        scored: list[tuple[float, Skill]] = []
        for s in self.all():
            hay = " ".join([s.name, s.description, " ".join(s.tags)]).lower()
            score = 0.0
            for t in terms:
                if t in hay:
                    score += 1.0
                    # Bonus for exact-token hits in description (vs substring)
                    if re.search(rf"\b{re.escape(t)}\b", s.description.lower()):
                        score += 0.5
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def format_for_prompt(self, skills: list[Skill]) -> str:
        if not skills:
            return ""
        blocks = "\n\n".join(s.to_prompt_block() for s in skills)
        return (
            "# Loaded skills\n\n"
            "The following skills are relevant to the user's request. "
            "Follow the procedure exactly unless the user contradicts it.\n\n"
            f"{blocks}"
        )
