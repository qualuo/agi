"""Skill library.

Skills are reusable, named procedures stored as markdown files on disk. Each
skill captures *how* the agent has previously succeeded at a class of task —
when to use it, the steps, known failure modes.

Format on disk (`~/.agi/skills/<name>.md` by default):

    ---
    name: summarize-file
    when: Summarize a text file the user points at.
    tags: [io, summarize]
    ---
    1. Read the file with read_file.
    2. Identify the top-level structure (headers, sections).
    3. Write 3-7 bullet points capturing what each section asserts.
    4. Verify by re-reading the headers.

Retrieval is keyword over `name`, `when`, `tags`, and body. The agent can
also add new skills at runtime via `save`.

This is the medium-timescale learning channel from ARCHITECTURE.md §3:
successful task decompositions accumulate and lower the cost of the next
similar task.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_WHEN_RE = re.compile(r"^when:\s*(.+?)\s*$", re.MULTILINE)
_TAGS_RE = re.compile(r"^tags:\s*\[(.*?)\]\s*$", re.MULTILINE)
_SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass
class Skill:
    name: str
    when: str
    tags: list[str] = field(default_factory=list)
    body: str = ""
    path: Path | None = None

    def render(self) -> str:
        tags = ", ".join(self.tags)
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"when: {self.when}\n"
            f"tags: [{tags}]\n"
            f"---\n"
            f"{self.body.rstrip()}\n"
        )

    def summary(self) -> str:
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"{self.name}{tag_str}: {self.when}"


def _parse(text: str, path: Path | None = None) -> Skill | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm = m.group(1)
    body = text[m.end():]
    name_m = _NAME_RE.search(fm)
    when_m = _WHEN_RE.search(fm)
    tags_m = _TAGS_RE.search(fm)
    if not name_m:
        return None
    raw_tags = tags_m.group(1) if tags_m else ""
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    return Skill(
        name=name_m.group(1).strip(),
        when=(when_m.group(1).strip() if when_m else ""),
        tags=tags,
        body=body,
        path=path,
    )


class SkillLibrary:
    """Flat directory of markdown skills. Lazy-reads files on each call so
    edits on disk (by the user, or by the agent via save) are picked up
    without restart."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Skill]:
        out: list[Skill] = []
        for f in sorted(self.path.glob("*.md")):
            try:
                text = f.read_text()
            except OSError:
                continue
            skill = _parse(text, path=f)
            if skill is not None:
                out.append(skill)
        return out

    def get(self, name: str) -> Skill | None:
        if not _SAFE_NAME_RE.match(name):
            return None
        f = self.path / f"{name}.md"
        if not f.exists():
            return None
        return _parse(f.read_text(), path=f)

    def search(self, query: str, k: int = 5) -> list[Skill]:
        q = query.lower().strip()
        terms = [t for t in q.split() if t]
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for s in self.all():
            hay = (
                s.name.lower()
                + " "
                + s.when.lower()
                + " "
                + " ".join(t.lower() for t in s.tags)
                + " "
                + s.body.lower()
            )
            # Heavier weight on name/when/tags than body.
            head = (
                s.name.lower()
                + " "
                + s.when.lower()
                + " "
                + " ".join(t.lower() for t in s.tags)
            )
            score = sum(head.count(t) * 3 + hay.count(t) for t in terms)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:k]]

    def save(
        self,
        name: str,
        when: str,
        body: str,
        tags: list[str] | None = None,
    ) -> Skill:
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(
                f"invalid skill name {name!r}: lowercase letters/digits/hyphens, "
                "must start with letter or digit, max 64 chars"
            )
        skill = Skill(name=name, when=when, tags=tags or [], body=body)
        out_path = self.path / f"{name}.md"
        out_path.write_text(skill.render())
        skill.path = out_path
        return skill

    def delete(self, name: str) -> bool:
        if not _SAFE_NAME_RE.match(name):
            return False
        f = self.path / f"{name}.md"
        if not f.exists():
            return False
        f.unlink()
        return True
