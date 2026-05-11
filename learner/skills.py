"""Skill library.

A skill is a named, retrievable procedure stored as a small text file:

    ---
    name: <slug>
    triggers: [keyword, keyword, ...]
    ---
    <markdown body: when to use, the procedure, failure modes>

Skills are the medium-timescale learning channel from `ARCHITECTURE.md`:
the system distills successful task decompositions into reusable SOPs, and
the next time a similar task comes in, the relevant skills are loaded into
the system prompt instead of re-derived from scratch.

This module is intentionally minimal:
- on-disk format is plain markdown so humans can edit skills in an editor;
- retrieval is keyword based (triggers + body) — embeddings slot in later;
- writes are atomic via tempfile + rename so an interrupted save can't
  leave a half-written file in the library.

Pluggable boundaries (in the spirit of the architecture's "shape, not
specifics" rule): the retrieval function takes a query string and returns
ranked skills, so a vector backend can replace `_score` without changing
callers.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def _slugify(name: str) -> str:
    s = SLUG_RE.sub("-", name.lower()).strip("-")
    return s or "skill"


@dataclass
class Skill:
    name: str
    triggers: list[str] = field(default_factory=list)
    body: str = ""
    path: Path | None = None  # set by the loader for round-tripping

    def render(self) -> str:
        trigger_str = ", ".join(self.triggers)
        return f"---\nname: {self.name}\ntriggers: [{trigger_str}]\n---\n{self.body.rstrip()}\n"

    @classmethod
    def parse(cls, text: str, path: Path | None = None) -> "Skill":
        m = FRONTMATTER_RE.match(text)
        if not m:
            # Tolerate skills without frontmatter — name comes from filename.
            name = path.stem if path else "unnamed"
            return cls(name=name, triggers=[], body=text.strip(), path=path)
        meta_block, body = m.group(1), m.group(2)
        name = path.stem if path else "unnamed"
        triggers: list[str] = []
        for line in meta_block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "name" and value:
                name = value
            elif key == "triggers":
                triggers = _parse_list(value)
        return cls(name=name, triggers=triggers, body=body.strip(), path=path)


def _parse_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [t.strip() for t in value.split(",") if t.strip()]


class SkillLibrary:
    """A directory of skill files. Default location: ~/.agi/skills/."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for entry in sorted(self.path.iterdir()):
            if entry.suffix.lower() != ".md" or not entry.is_file():
                continue
            try:
                skills.append(Skill.parse(entry.read_text(), path=entry))
            except Exception:
                # A malformed file shouldn't crash the agent. Skip it.
                continue
        return skills

    def get(self, name: str) -> Skill | None:
        slug = _slugify(name)
        target = self.path / f"{slug}.md"
        if not target.exists():
            return None
        return Skill.parse(target.read_text(), path=target)

    def write(self, skill: Skill) -> Path:
        slug = _slugify(skill.name)
        target = self.path / f"{slug}.md"
        skill.path = target
        # Atomic write so an interrupted save doesn't corrupt the library.
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=self.path, prefix=f".{slug}.", suffix=".md.tmp"
        ) as tf:
            tf.write(skill.render())
            tmp_path = Path(tf.name)
        os.replace(tmp_path, target)
        return target

    def delete(self, name: str) -> bool:
        target = self.path / f"{_slugify(name)}.md"
        if not target.exists():
            return False
        target.unlink()
        return True

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        """Rank skills by trigger and body keyword match.

        Trigger matches weigh 3x body matches — triggers are the curated
        index. This is the v1 retrieval; the interface stays stable when
        we swap in embeddings.
        """
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for skill in self.all():
            score = _score(skill, terms)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]


def _score(skill: Skill, terms: list[str]) -> int:
    body = skill.body.lower()
    triggers = " ".join(skill.triggers).lower()
    return sum(triggers.count(t) * 3 + body.count(t) for t in terms)
