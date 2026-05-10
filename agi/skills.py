"""Skill library — procedural memory the agent can retrieve.

A skill is a markdown file with three sections: when to use it, the
procedure to follow, and known failure modes. Skills live as plain `.md`
files in a directory so a human can author or edit them by hand. The
agent retrieves the top-K relevant skills for a prompt and pastes them
into the system prompt for that turn.

Retrieval is keyword-based for v1 (intersection over title + description
+ trigger keywords). Same shape as `Memory.search`; embedding retrieval
slots in behind the same `search` method later.

Two write paths exist:
- `add_skill(...)` — agent or human-authored skill, written immediately.
- `compile_skills_from_traces(...)` — periodic distillation from
  successful traces (Stage 3 in ARCHITECTURE.md). Lives next to the
  library to keep the I/O coupled.

The compile step is deliberately conservative: it proposes skills, it
does not commit them. A human (or a future critic) approves before the
file is written.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


SKILL_TEMPLATE = """\
---
id: {id}
title: {title}
created: {ts}
triggers: {triggers}
---

# When to use

{when}

# Procedure

{procedure}

# Failure modes

{failure_modes}
"""


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n(.*)$", re.DOTALL)


@dataclass
class Skill:
    id: str
    title: str
    path: str
    when: str
    procedure: str
    failure_modes: str
    triggers: list[str] = field(default_factory=list)
    created: float = 0.0

    def to_prompt_block(self) -> str:
        """Compact form suitable for inclusion in a system prompt."""
        triggers = f" (triggers: {', '.join(self.triggers)})" if self.triggers else ""
        return (
            f"## SKILL: {self.title}{triggers}\n"
            f"When to use: {self.when.strip()}\n"
            f"Procedure:\n{self.procedure.strip()}\n"
            f"Failure modes: {self.failure_modes.strip()}\n"
        )


class SkillLibrary:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    # ---- read ----

    def all(self) -> list[Skill]:
        out: list[Skill] = []
        for p in sorted(self.path.glob("*.md")):
            try:
                out.append(self._parse(p))
            except ValueError:
                # Malformed skill — skip rather than crash the agent.
                continue
        return out

    def get(self, skill_id: str) -> Skill | None:
        for s in self.all():
            if s.id == skill_id:
                return s
        return None

    def search(self, query: str, k: int = 3) -> list[Skill]:
        q = query.lower().strip()
        terms = [t for t in re.findall(r"[a-z0-9]+", q) if t]
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for s in self.all():
            hay = (
                f"{s.title} {' '.join(s.triggers)} {s.when} {s.procedure}"
            ).lower()
            score = sum(hay.count(t) for t in terms)
            # Triggers count double — they're authored signal.
            for t in terms:
                if any(t in tr.lower() for tr in s.triggers):
                    score += 5
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], -x[1].created))
        return [s for _, s in scored[:k]]

    # ---- write ----

    def add(
        self,
        *,
        title: str,
        when: str,
        procedure: str,
        failure_modes: str = "(none recorded)",
        triggers: list[str] | None = None,
    ) -> Skill:
        sid = uuid.uuid4().hex[:8]
        triggers = triggers or []
        ts = time.time()
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "skill"
        path = self.path / f"{slug}-{sid}.md"
        path.write_text(
            SKILL_TEMPLATE.format(
                id=sid,
                title=title,
                ts=ts,
                triggers=json.dumps(triggers),
                when=when.strip(),
                procedure=procedure.strip(),
                failure_modes=failure_modes.strip(),
            )
        )
        return Skill(
            id=sid,
            title=title,
            path=str(path),
            when=when,
            procedure=procedure,
            failure_modes=failure_modes,
            triggers=triggers,
            created=ts,
        )

    def remove(self, skill_id: str) -> bool:
        s = self.get(skill_id)
        if s is None:
            return False
        Path(s.path).unlink(missing_ok=True)
        return True

    # ---- internal ----

    def _parse(self, p: Path) -> Skill:
        text = p.read_text()
        m = _FRONTMATTER_RE.match(text)
        if not m:
            raise ValueError(f"missing frontmatter in {p}")
        meta_block, body = m.group(1), m.group(2)
        meta: dict = {}
        for line in meta_block.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        triggers = []
        if meta.get("triggers"):
            try:
                triggers = json.loads(meta["triggers"])
            except json.JSONDecodeError:
                triggers = [t.strip() for t in meta["triggers"].split(",") if t.strip()]
        sections = _split_sections(body)
        return Skill(
            id=meta.get("id", p.stem),
            title=meta.get("title", p.stem),
            path=str(p),
            when=sections.get("when to use", "").strip(),
            procedure=sections.get("procedure", "").strip(),
            failure_modes=sections.get("failure modes", "").strip(),
            triggers=triggers,
            created=float(meta.get("created", 0) or 0),
        )


def _split_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("# "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[2:].strip().lower()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def render_skill_block(skills: Iterable[Skill]) -> str:
    """Render retrieved skills as a single block to splice into a prompt."""
    skills = list(skills)
    if not skills:
        return ""
    parts = ["# Relevant skills (retrieved from skill library)\n"]
    for s in skills:
        parts.append(s.to_prompt_block())
    parts.append(
        "Use the skill if it applies; do not narrate its retrieval. If none apply, ignore."
    )
    return "\n".join(parts)
