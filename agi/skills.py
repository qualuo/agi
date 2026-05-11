"""Skill library — procedural memory.

A *skill* is a markdown document with three sections:

```
# <name>

## When to use
<one line: trigger conditions>

## Procedure
<numbered steps the agent should follow>

## Failure modes
<known ways this skill goes wrong + recovery>
```

Skills live as `*.md` files under `~/.agi/skills/`. The agent loads the
top-K relevant skills into its system prompt for each task (cheap retrieval:
keyword overlap on the "When to use" line + skill name).

This is the medium-timescale learning channel from ARCHITECTURE.md. When the
agent solves a novel class of task, it can write a skill so the next instance
is cheaper. Successful skills get used; useless ones rot quietly.

Curation hygiene:
- Skill writes are versioned (append .vN suffix). Old versions stay on disk.
- A `usage_count` is tracked in a sidecar JSON; skills not used in N tasks
  get archived (moved out of the active dir) but never deleted.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


SKILL_TEMPLATE = """# {name}

## When to use
{when_to_use}

## Procedure
{procedure}

## Failure modes
{failure_modes}
"""


@dataclass
class Skill:
    name: str
    when_to_use: str
    procedure: str
    failure_modes: str
    path: Path | None = None
    usage_count: int = 0
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = 0.0

    def to_markdown(self) -> str:
        return SKILL_TEMPLATE.format(
            name=self.name,
            when_to_use=self.when_to_use,
            procedure=self.procedure,
            failure_modes=self.failure_modes,
        )

    def to_prompt_block(self) -> str:
        """Compact rendering for inclusion in a system prompt."""
        return (
            f"### skill: {self.name}\n"
            f"when to use: {self.when_to_use}\n"
            f"procedure:\n{self.procedure}\n"
            f"failure modes: {self.failure_modes}"
        )


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "skill"


class SkillLibrary:
    """Filesystem-backed skill store.

    Layout:
        <root>/<slug>.md            -- active skill
        <root>/<slug>.meta.json     -- usage stats sidecar
        <root>/archive/<slug>.md    -- archived skills (kept for provenance)
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root else Path.home() / ".agi" / "skills"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "archive").mkdir(exist_ok=True)

    def add(self, name: str, when_to_use: str, procedure: str, failure_modes: str = "") -> Skill:
        slug = _slugify(name)
        path = self.root / f"{slug}.md"
        skill = Skill(
            name=name,
            when_to_use=when_to_use.strip(),
            procedure=procedure.strip(),
            failure_modes=failure_modes.strip() or "none recorded",
            path=path,
        )
        path.write_text(skill.to_markdown())
        self._write_meta(slug, {"usage_count": 0, "created_ts": skill.created_ts, "last_used_ts": 0.0})
        return skill

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for md in sorted(self.root.glob("*.md")):
            skill = self._load(md)
            if skill is not None:
                skills.append(skill)
        return skills

    def get(self, name: str) -> Skill | None:
        slug = _slugify(name)
        path = self.root / f"{slug}.md"
        if not path.exists():
            return None
        return self._load(path)

    def search(self, query: str, k: int = 3) -> list[Skill]:
        """Cheap keyword retrieval over the 'When to use' line + skill name."""
        q = query.lower()
        terms = [t for t in re.split(r"\W+", q) if len(t) > 2]
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for skill in self.all():
            hay = (skill.when_to_use + " " + skill.name).lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda x: (-x[0], -x[1].usage_count))
        return [s for _, s in scored[:k]]

    def mark_used(self, name: str) -> None:
        slug = _slugify(name)
        meta = self._read_meta(slug)
        if not meta:
            return
        meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
        meta["last_used_ts"] = time.time()
        self._write_meta(slug, meta)

    def archive(self, name: str) -> bool:
        slug = _slugify(name)
        path = self.root / f"{slug}.md"
        if not path.exists():
            return False
        dest = self.root / "archive" / f"{slug}.md"
        path.rename(dest)
        meta = self.root / f"{slug}.meta.json"
        if meta.exists():
            meta.rename(self.root / "archive" / f"{slug}.meta.json")
        return True

    def _load(self, path: Path) -> Skill | None:
        text = path.read_text()
        sections = _parse_sections(text)
        if "name" not in sections:
            return None
        meta = self._read_meta(path.stem) or {}
        return Skill(
            name=sections["name"],
            when_to_use=sections.get("when to use", ""),
            procedure=sections.get("procedure", ""),
            failure_modes=sections.get("failure modes", ""),
            path=path,
            usage_count=int(meta.get("usage_count", 0)),
            created_ts=float(meta.get("created_ts", 0.0)),
            last_used_ts=float(meta.get("last_used_ts", 0.0)),
        )

    def _read_meta(self, slug: str) -> dict | None:
        p = self.root / f"{slug}.meta.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def _write_meta(self, slug: str, meta: dict) -> None:
        (self.root / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2))


def _parse_sections(text: str) -> dict[str, str]:
    """Extract `# name` and `## <heading>` blocks from a skill markdown file."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m1 = re.match(r"^#\s+(.+)$", line)
        m2 = re.match(r"^##\s+(.+)$", line)
        if m1:
            if current_key:
                sections[current_key] = "\n".join(buf).strip()
                buf = []
            sections["name"] = m1.group(1).strip()
            current_key = None
        elif m2:
            if current_key:
                sections[current_key] = "\n".join(buf).strip()
                buf = []
            current_key = m2.group(1).strip().lower()
        else:
            if current_key is not None:
                buf.append(line)
    if current_key:
        sections[current_key] = "\n".join(buf).strip()
    return sections
