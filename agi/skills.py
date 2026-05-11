"""Skill library.

Procedural memory: named, retrievable how-tos that the agent has learned
or been taught. v1 is a flat directory of markdown files with YAML-ish
front matter. Retrieval is keyword overlap between the user task and the
skill's `description`. Good enough to demonstrate the architecture's
medium-timescale learning channel without taking a dependency on
embeddings.

Format (one file per skill):

    ---
    name: short-name
    description: one-line trigger phrase
    tags: [optional, list]
    ---
    Markdown body — the actual procedure, examples, known failure modes.

A SkillLibrary points at a directory; missing dir → empty library.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


@dataclass
class Skill:
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    path: Path | None = None

    def matches_score(self, query: str) -> int:
        q = set(_tokenize(query))
        hay = set(_tokenize(self.description)) | set(_tokenize(" ".join(self.tags)))
        return len(q & hay)


def parse_skill(text: str, path: Path | None = None) -> Skill:
    m = _FRONT_MATTER_RE.match(text)
    meta: dict = {}
    body = text
    if m:
        body = text[m.end():]
        for raw in m.group(1).splitlines():
            if ":" not in raw:
                continue
            key, _, value = raw.partition(":")
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                items = [x.strip() for x in value[1:-1].split(",") if x.strip()]
                meta[key] = items
            else:
                meta[key] = value
    name = str(meta.get("name") or (path.stem if path else "unnamed"))
    description = str(meta.get("description") or "")
    tags = meta.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    return Skill(name=name, description=description, body=body.strip(), tags=list(tags), path=path)


def render_skill(skill: Skill) -> str:
    tags = ", ".join(skill.tags)
    return (
        f"---\nname: {skill.name}\ndescription: {skill.description}\n"
        f"tags: [{tags}]\n---\n{skill.body}\n"
    )


class SkillLibrary:
    """Read/write skills in a directory. Retrieval is keyword overlap."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root else Path.home() / ".agi" / "skills"
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Skill]:
        skills: list[Skill] = []
        for path in sorted(self.root.glob("*.md")):
            try:
                skills.append(parse_skill(path.read_text(), path=path))
            except Exception:
                continue
        return skills

    def get(self, name: str) -> Skill | None:
        for s in self.list():
            if s.name == name:
                return s
        return None

    def save(self, skill: Skill) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", skill.name).strip("-") or "skill"
        path = self.root / f"{safe}.md"
        if path.exists():
            # Avoid clobbering: append timestamp to distinguish revisions.
            path = self.root / f"{safe}-{int(time.time())}.md"
        path.write_text(render_skill(skill))
        skill.path = path
        return path

    def delete(self, name: str) -> bool:
        skill = self.get(name)
        if skill is None or skill.path is None:
            return False
        skill.path.unlink()
        return True

    def retrieve(self, query: str, k: int = 2) -> list[Skill]:
        scored: list[tuple[int, Skill]] = []
        for skill in self.list():
            score = skill.matches_score(query)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]
