"""Skill library — Stage 3 from ARCHITECTURE.md.

A directory of named procedures. Each skill is a markdown file describing:
- when to use it
- the procedure
- known failure modes

The runtime engine looks up skills relevant to the current task instruction
and injects the top-K into the agent's system prompt. Cost of a skill is
~a kilobyte of prompt — cheap, and prompt caching makes it cheaper.

This is the medium-timescale learning channel: distillation of repeated
successful patterns. v1 is keyword retrieval; later we swap in embeddings
behind the same `retrieve()` interface.

File format
-----------
Each `.md` file in the library has optional YAML-ish frontmatter:

    ---
    name: shell_command_runner
    when: user asks to execute or test a shell command
    tags: [shell, tooling]
    ---
    # When to use
    ...

If frontmatter is absent, the filename stem is the name and the entire body
is searched.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Skill:
    name: str
    when: str  # plaintext description of when this skill applies
    body: str  # full markdown body (without frontmatter)
    tags: list[str] = field(default_factory=list)
    path: Path | None = None
    created_at: float = field(default_factory=time.time)
    times_used: int = 0


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_KV_RE = re.compile(r"^([a-zA-Z_][a-zA-Z_0-9]*)\s*:\s*(.*)$")


def parse_skill(text: str, *, default_name: str = "skill", path: Path | None = None) -> Skill:
    """Parse a markdown skill file. Tolerant of missing frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    meta: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            kv = _KV_RE.match(line.strip())
            if kv:
                meta[kv.group(1)] = kv.group(2).strip()
        body = m.group(2)
    else:
        body = text

    raw_tags = meta.get("tags", "").strip()
    tags: list[str] = []
    if raw_tags:
        raw_tags = raw_tags.strip("[]")
        tags = [t.strip().strip("'\"") for t in raw_tags.split(",") if t.strip()]

    return Skill(
        name=meta.get("name", default_name),
        when=meta.get("when", ""),
        body=body.strip(),
        tags=tags,
        path=path,
    )


def render_skill(skill: Skill) -> str:
    """Inverse of parse_skill — useful for `add_from_text`."""
    tag_str = "[" + ", ".join(skill.tags) + "]" if skill.tags else "[]"
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"when: {skill.when}\n"
        f"tags: {tag_str}\n"
        "---\n"
        f"{skill.body}\n"
    )


class SkillLibrary:
    """A flat directory of markdown skills.

    `retrieve(query, k)` returns the top-K skills by keyword overlap against
    (name + when + tags + body). This is intentionally simple; semantic
    retrieval drops in behind the same method.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    # --- IO ----------------------------------------------------------------

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for p in sorted(self.path.glob("*.md")):
            try:
                text = p.read_text()
            except OSError:
                continue
            skills.append(parse_skill(text, default_name=p.stem, path=p))
        return skills

    def get(self, name: str) -> Skill | None:
        for s in self.all():
            if s.name == name:
                return s
        return None

    def add(self, skill: Skill) -> Skill:
        """Write a skill to disk. Overwrites if a skill with the same name exists.
        If `skill.path` is unset, picks a filename from `skill.name`.
        """
        if skill.path is None:
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", skill.name).strip("_") or uuid.uuid4().hex[:8]
            skill.path = self.path / f"{safe}.md"
        skill.path.write_text(render_skill(skill))
        return skill

    def add_from_text(
        self,
        *,
        name: str,
        when: str,
        body: str,
        tags: Iterable[str] = (),
    ) -> Skill:
        skill = Skill(name=name, when=when, body=body.strip(), tags=list(tags))
        return self.add(skill)

    def remove(self, name: str) -> bool:
        for p in self.path.glob("*.md"):
            try:
                s = parse_skill(p.read_text(), default_name=p.stem, path=p)
            except Exception:
                continue
            if s.name == name:
                p.unlink()
                return True
        return False

    # --- retrieval ---------------------------------------------------------

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        """Return the top-K skills with non-zero keyword overlap with `query`.
        Scoring: term frequency across (name + when + tags + body), lowercased."""
        terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for s in self.all():
            hay = " ".join([s.name, s.when, " ".join(s.tags), s.body]).lower()
            score = sum(hay.count(t) for t in terms)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:k]]

    # --- distillation seam -------------------------------------------------

    def propose_from_trace(self, instruction: str, final_text: str, *, name: str | None = None) -> Skill:
        """Stub for trace-driven skill compilation.

        In v1 the human is the LLM-judge: this returns a draft skill that a
        reviewer (or a later LLM pass) can refine before committing. We don't
        auto-write — that's how a noisy skill library is born.

        The real distillation loop will live in `learner/skill_compile.py`:
        scan recent successful traces, cluster, prompt an LLM to draft a
        skill per cluster, and surface them for human review.
        """
        draft_name = name or _slug(instruction)
        body = (
            f"## When to use\n{instruction}\n\n"
            f"## Procedure (proposed; refine before committing)\n{final_text}\n\n"
            f"## Failure modes\nUnknown — verify on a few examples before saving.\n"
        )
        return Skill(name=draft_name, when=instruction[:200], body=body)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "skill"
