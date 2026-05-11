"""Skill library — durable procedural memory.

A skill is a named, retrievable recipe for a task class. Stored as
markdown on disk (frontmatter for metadata, body for the SOP) so a
human can edit, review, and version them with git.

A coordination engine attaches relevant skills to a session by name or
by description-match; the runtime injects them into the agent's system
prompt for that task. New skills are proposed by a `compile_skill` call
and only **promoted** to the active library after passing the eval gate.

Schema (markdown file at `<root>/<id>.md`):

    ---
    id: <slug>
    description: <one-line description>
    triggers: ["substring1", "regex2"]
    created_at: 2026-05-11T...
    promoted: true|false
    eval_pass_rate: 0.83
    parent: <id|null>
    ---
    # Body
    ...

The library is keyword-retrievable by default (same as Memory) — embeddings
can slot in behind `match()` later.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    s = _SLUG_RE.sub("-", s.lower()).strip("-")
    return s[:48] or uuid.uuid4().hex[:8]


@dataclass
class Skill:
    id: str
    description: str
    body: str
    triggers: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    promoted: bool = False
    eval_pass_rate: float | None = None
    parent: str | None = None

    def render_for_prompt(self) -> str:
        return f"## skill: {self.id}\n{self.description}\n\n{self.body.strip()}\n"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "body": self.body,
            "triggers": list(self.triggers),
            "created_at": self.created_at,
            "promoted": self.promoted,
            "eval_pass_rate": self.eval_pass_rate,
            "parent": self.parent,
        }


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta_raw = m.group(1)
    body = text[m.end():]
    meta: dict = {}
    for line in meta_raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, body


def _serialize(skill: Skill) -> str:
    triggers = json.dumps(skill.triggers)
    parts = [
        "---",
        f"id: {skill.id}",
        f"description: {skill.description}",
        f"triggers: {triggers}",
        f"created_at: {skill.created_at}",
        f"promoted: {'true' if skill.promoted else 'false'}",
    ]
    if skill.eval_pass_rate is not None:
        parts.append(f"eval_pass_rate: {skill.eval_pass_rate}")
    if skill.parent:
        parts.append(f"parent: {skill.parent}")
    parts.append("---\n")
    parts.append(skill.body.rstrip() + "\n")
    return "\n".join(parts)


class SkillLibrary:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path.home() / ".agi" / "skills"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, skill_id: str) -> Path:
        return self.root / f"{skill_id}.md"

    def add(
        self,
        description: str,
        body: str,
        *,
        triggers: list[str] | None = None,
        promoted: bool = False,
        parent: str | None = None,
        id: str | None = None,
    ) -> Skill:
        skill = Skill(
            id=id or _slugify(description),
            description=description.strip(),
            body=body.strip(),
            triggers=list(triggers or []),
            promoted=promoted,
            parent=parent,
        )
        # Avoid silent overwrite — append a short suffix if the id collides.
        path = self._path(skill.id)
        if path.exists():
            skill.id = f"{skill.id}-{uuid.uuid4().hex[:6]}"
            path = self._path(skill.id)
        path.write_text(_serialize(skill))
        return skill

    def get(self, skill_id: str) -> Skill | None:
        p = self._path(skill_id)
        if not p.exists():
            return None
        meta, body = _parse(p.read_text())
        return Skill(
            id=meta.get("id", skill_id),
            description=meta.get("description", ""),
            body=body,
            triggers=_safe_json_list(meta.get("triggers")),
            created_at=float(meta.get("created_at", 0) or 0),
            promoted=str(meta.get("promoted", "")).lower() == "true",
            eval_pass_rate=_safe_float(meta.get("eval_pass_rate")),
            parent=meta.get("parent") or None,
        )

    def all(self, *, promoted_only: bool = False) -> list[Skill]:
        skills: list[Skill] = []
        for p in sorted(self.root.glob("*.md")):
            s = self.get(p.stem)
            if s is None:
                continue
            if promoted_only and not s.promoted:
                continue
            skills.append(s)
        return skills

    def remove(self, skill_id: str) -> bool:
        p = self._path(skill_id)
        if not p.exists():
            return False
        p.unlink()
        return True

    def promote(self, skill_id: str, *, eval_pass_rate: float | None = None) -> Skill | None:
        s = self.get(skill_id)
        if s is None:
            return None
        s.promoted = True
        if eval_pass_rate is not None:
            s.eval_pass_rate = eval_pass_rate
        self._path(skill_id).write_text(_serialize(s))
        return s

    def match(self, prompt: str, *, k: int = 3, promoted_only: bool = True) -> list[Skill]:
        """Return up to k skills whose triggers or description appear in prompt.

        Scoring: trigger substring hit > description term hit; ties broken
        by recency. Keyword-based; semantic search slots in behind the same
        method later.
        """
        q = prompt.lower()
        scored: list[tuple[float, Skill]] = []
        for s in self.all(promoted_only=promoted_only):
            score = 0.0
            for t in s.triggers:
                if t.lower() in q:
                    score += 2.0
            for term in re.findall(r"[a-z0-9]+", s.description.lower()):
                if len(term) < 4:
                    continue
                if term in q:
                    score += 0.25
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], -x[1].created_at))
        return [s for _, s in scored[:k]]


def _safe_json_list(v) -> list[str]:
    if not v:
        return []
    try:
        out = json.loads(v)
        return [str(x) for x in out] if isinstance(out, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _safe_float(v) -> float | None:
    if v in (None, "", "null"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
