"""Skill library — the medium-timescale learning channel.

A skill is a markdown file describing a procedure: when to use it, the
steps, known failure modes. The agent loads relevant skills into its system
prompt at the start of each task. When the agent solves a novel class of
task, the operator (or a future reflection step) can save a SKILL.md so the
next instance is cheaper.

Storage: a directory of `.md` files. Default: `~/.agi/skills/`.

Each file has a YAML-ish frontmatter with `name` and `when` (a short
description of when to use the skill); the body is the procedure.

Example skill file (skills/git-rebase-recovery.md):

    ---
    name: git-rebase-recovery
    when: a git rebase has gone wrong and the user wants to recover
    ---

    1. Check `git reflog` for the pre-rebase HEAD.
    2. ...

Retrieval is keyword-based against the `when` field. Replaceable with
embedding search later — the public interface stays the same.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Skill:
    name: str
    when: str
    body: str
    path: Path
    mtime: float = 0.0
    tags: list[str] = field(default_factory=list)

    def render(self) -> str:
        return f"## skill: {self.name}\nuse when: {self.when}\n\n{self.body}"


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_skill(path: Path) -> Skill:
    raw = path.read_text()
    m = _FRONTMATTER_RE.match(raw)
    if m:
        meta_block = m.group(1)
        body = raw[m.end() :].strip()
        meta = _parse_simple_yaml(meta_block)
    else:
        meta = {}
        body = raw.strip()
    name = meta.get("name") or path.stem
    when = meta.get("when") or _first_line(body) or name
    tags_raw = meta.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    return Skill(name=name, when=when, body=body, path=path, mtime=path.stat().st_mtime, tags=tags)


def _parse_simple_yaml(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def _first_line(s: str) -> str:
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


class SkillLibrary:
    """Directory-backed skill library.

    Cheap to instantiate; reads the directory once. Call `reload()` to pick
    up new files.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)
        self._skills: list[Skill] = []
        self.reload()

    def reload(self) -> None:
        self._skills = []
        for p in sorted(self.path.glob("*.md")):
            try:
                self._skills.append(_parse_skill(p))
            except Exception:
                # Malformed skill — skip rather than crash the runtime.
                continue

    def all(self) -> list[Skill]:
        return list(self._skills)

    def search(self, query: str, k: int = 3) -> list[Skill]:
        q = query.lower().strip()
        if not q:
            return []
        terms = [t for t in re.split(r"\W+", q) if len(t) > 2]
        if not terms:
            terms = [q]
        scored: list[tuple[int, Skill]] = []
        for skill in self._skills:
            hay = (skill.when + " " + skill.name + " " + " ".join(skill.tags)).lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:k]]

    def get(self, name: str) -> Skill | None:
        for s in self._skills:
            if s.name == name:
                return s
        return None

    def write(self, name: str, when: str, body: str, tags: Iterable[str] | None = None) -> Skill:
        slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "skill"
        path = self.path / f"{slug}.md"
        tag_line = f"tags: {', '.join(tags)}\n" if tags else ""
        content = f"---\nname: {name}\nwhen: {when}\n{tag_line}---\n\n{body.strip()}\n"
        path.write_text(content)
        skill = _parse_skill(path)
        # Replace any existing skill with the same name.
        self._skills = [s for s in self._skills if s.name != skill.name]
        self._skills.append(skill)
        return skill

    def delete(self, name: str) -> bool:
        skill = self.get(name)
        if skill is None:
            return False
        try:
            skill.path.unlink()
        except FileNotFoundError:
            pass
        self._skills = [s for s in self._skills if s.name != name]
        return True

    def system_prompt_addendum(self, query: str | None = None, k: int = 5) -> str:
        """A block to append to the agent's system prompt.

        With a query, returns up to k matched skills. Without a query, lists
        every skill's `when` line so the model knows what's available and
        can request it implicitly through behavior.
        """
        if not self._skills:
            return ""
        if query is not None:
            picked = self.search(query, k=k)
            if not picked:
                return ""
            blocks = [s.render() for s in picked]
            return "\n\n## Skill library (matched)\n\n" + "\n\n---\n\n".join(blocks)
        # Index mode: just the table of contents.
        toc = "\n".join(f"- {s.name}: {s.when}" for s in self._skills)
        return f"\n\n## Available skills\n{toc}\n"


# Tool wrappers — let the agent self-extend by writing skills during a session.

def make_skill_tools(library: SkillLibrary) -> tuple[list[dict], dict]:
    def list_skills() -> str:
        skills = library.all()
        if not skills:
            return "(no skills yet)"
        return "\n".join(f"- {s.name}: {s.when}" for s in skills)

    def read_skill(name: str) -> str:
        skill = library.get(name)
        if skill is None:
            return f"error: skill {name!r} not found"
        return skill.render()

    def save_skill(name: str, when: str, body: str, tags: list[str] | None = None) -> str:
        skill = library.write(name, when, body, tags or [])
        return f"saved skill {skill.name!r} to {skill.path}"

    def search_skills(query: str, k: int = 3) -> str:
        results = library.search(query, k=k)
        if not results:
            return "no matching skills"
        return "\n".join(f"- {s.name}: {s.when}" for s in results)

    schemas = [
        {
            "name": "list_skills",
            "description": "List every skill in the library with its 'when to use' line.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "read_skill",
            "description": "Read the full body of a named skill.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        {
            "name": "save_skill",
            "description": (
                "Distill a successful procedure into a reusable skill. The skill "
                "becomes available to future sessions. Use after solving a novel task "
                "that's likely to recur."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short slug-like name."},
                    "when": {"type": "string", "description": "One-line description of when this applies."},
                    "body": {"type": "string", "description": "Markdown body: steps and known failure modes."},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "when", "body"],
            },
        },
        {
            "name": "search_skills",
            "description": "Find skills relevant to a query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    ]
    handlers = {
        "list_skills": list_skills,
        "read_skill": read_skill,
        "save_skill": save_skill,
        "search_skills": search_skills,
    }
    return schemas, handlers
