"""Skill library.

A *skill* is a versioned markdown SOP: when to use it, the procedure (as
plain text the agent will read), known failure modes, and an optional
parameter schema. Skills live as `.md` files with a YAML-ish frontmatter in
either the project-local `skills_library/` or `~/.agi/skills/`.

Format:

    ---
    name: research-question
    description: Find a credible answer to a factual question on the open web.
    args:
      - question
      - depth: int = 2
    tags: [research, web]
    version: 1
    ---

    When you receive a research question, do the following:
    1. Search the web for `${question}`...

`SkillLibrary.retrieve(query, k)` returns the top-k most relevant skills via
keyword scoring on (name + description + tags). The Agent calls this at the
start of a task and prepends the matching skills to its system prompt.

Skills are content-addressable by `name`. Updating a skill bumps `version`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Skill:
    name: str
    description: str
    body: str
    args: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    version: int = 1
    path: str | None = None

    def render(self, args: dict[str, Any] | None = None) -> str:
        """Substitute ${var} placeholders in the body with provided args."""
        out = self.body
        for k, v in (args or {}).items():
            out = out.replace("${" + k + "}", str(v))
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse(path: Path) -> Skill | None:
    text = path.read_text()
    m = _FM.match(text)
    body = text[m.end():].strip() if m else text.strip()
    meta: dict[str, Any] = {}
    if m:
        for line in m.group(1).splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                inside = val[1:-1].strip()
                items = [x.strip() for x in inside.split(",") if x.strip()]
                meta[key] = items
            elif val.isdigit():
                meta[key] = int(val)
            else:
                meta[key] = val
    name = meta.get("name") or path.stem
    desc = meta.get("description", "")
    args = meta.get("args", []) if isinstance(meta.get("args"), list) else []
    tags = meta.get("tags", []) if isinstance(meta.get("tags"), list) else []
    version = int(meta.get("version", 1))
    return Skill(
        name=str(name),
        description=str(desc),
        body=body,
        args=[str(a) for a in args],
        tags=[str(t) for t in tags],
        version=version,
        path=str(path),
    )


class SkillLibrary:
    def __init__(self, *paths: str | os.PathLike[str]) -> None:
        # Search order: project skills_library/, then ~/.agi/skills/.
        roots: list[Path] = []
        if paths:
            roots = [Path(p) for p in paths]
        else:
            # Package-relative project skills/
            here = Path(__file__).resolve().parent.parent.parent
            roots.append(here / "skills_library")
            roots.append(Path.home() / ".agi" / "skills")
        self.roots = [r for r in roots if r.exists() or self._mkdir_safe(r)]
        self._skills: dict[str, Skill] = {}
        self._load()

    def _mkdir_safe(self, p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def _load(self) -> None:
        for root in self.roots:
            if not root.exists():
                continue
            for f in sorted(root.glob("*.md")):
                s = _parse(f)
                if s is None:
                    continue
                existing = self._skills.get(s.name)
                if existing is None or s.version >= existing.version:
                    self._skills[s.name] = s

    def reload(self) -> None:
        self._skills.clear()
        self._load()

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def add(self, skill: Skill, root: Path | None = None) -> Path:
        target_root = root or (self.roots[0] if self.roots else Path.home() / ".agi" / "skills")
        target_root.mkdir(parents=True, exist_ok=True)
        path = target_root / f"{skill.name}.md"
        fm_lines = [
            "---",
            f"name: {skill.name}",
            f"description: {skill.description}",
            f"args: [{', '.join(skill.args)}]",
            f"tags: [{', '.join(skill.tags)}]",
            f"version: {skill.version}",
            "---",
            "",
            skill.body,
            "",
        ]
        path.write_text("\n".join(fm_lines))
        self._skills[skill.name] = Skill(**{**asdict(skill), "path": str(path)})
        return path

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        if not query:
            return []
        q = query.lower()
        terms = [t for t in re.split(r"\W+", q) if t and len(t) > 2]
        scored: list[tuple[int, Skill]] = []
        for s in self._skills.values():
            hay = f"{s.name} {s.description} {' '.join(s.tags)}".lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:k]]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "description": s.description,
             "tags": s.tags, "args": s.args, "version": s.version}
            for s in self._skills.values()
        ]
