"""Skill library — procedural memory at the medium timescale.

A skill is a markdown file describing how to do a class of task: when to use
it, the procedure, and known failure modes. The agent retrieves relevant
skills at the start of a task and gets them injected into the system prompt.

This is the architecture's medium-timescale learning channel (per
ARCHITECTURE.md §3): faster than training a new adapter, slower than working
memory, and durable across sessions. v1 is a flat directory of `.md` files
with keyword retrieval. Embeddings would slot in behind the same `search`
interface.

A skill file looks like:

    ---
    name: solve-quadratic
    description: Solve quadratic equations using the quadratic formula.
    tags: [math, algebra]
    ---

    ## When to use
    The user gives you a quadratic equation `ax^2 + bx + c = 0` and asks for
    its roots.

    ## Procedure
    1. Identify a, b, c.
    2. Compute discriminant: b^2 - 4ac.
    3. ...

    ## Known failure modes
    - Floating-point error on near-zero discriminants.

The frontmatter is optional; if missing, name is taken from the filename and
the first non-empty line of the body becomes the description.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    path: Path | None = None

    def render(self) -> str:
        """Render the skill as a block to inject into a system prompt."""
        header = f"### Skill: {self.name}"
        if self.description:
            header += f"\n_{self.description}_"
        return f"{header}\n\n{self.body.strip()}"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_block, body = m.group(1), m.group(2)
    fm: dict = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
            fm[key] = items
        else:
            fm[key] = val.strip("'\"")
    return fm, body


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return ""


def parse_skill(path: Path) -> Skill:
    raw = path.read_text()
    fm, body = _parse_frontmatter(raw)
    name = fm.get("name") or path.stem
    description = fm.get("description") or _first_nonempty_line(body)
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return Skill(name=name, description=description, body=body, tags=tags, path=path)


class SkillLibrary:
    """Directory of markdown skills.

    The default location is `~/.agi/skills/`. Falls back gracefully if the
    directory doesn't exist — an empty library returns no skills, which lets
    the agent run identically to before.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "skills"
        self.path.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Skill]:
        skills: list[Skill] = []
        for p in sorted(self.path.glob("*.md")):
            try:
                skills.append(parse_skill(p))
            except Exception:
                # A malformed skill file shouldn't break the agent.
                continue
        return skills

    def search(self, query: str, k: int = 3) -> list[Skill]:
        """Rank skills by keyword overlap against name + description + tags.

        Body text is *not* searched: skill bodies are long and would dominate
        on incidental term matches. The frontmatter is the index.
        """
        q = query.lower().strip()
        terms = [t for t in re.split(r"\W+", q) if len(t) >= 2]
        scored: list[tuple[int, Skill]] = []
        for skill in self.all():
            haystack = " ".join(
                [skill.name, skill.description, " ".join(skill.tags)]
            ).lower()
            score = sum(haystack.count(t) for t in terms)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def write(
        self,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
    ) -> Skill:
        """Persist a new skill (or overwrite an existing one with the same name)."""
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
        if not safe_name:
            raise ValueError("skill name produced an empty filename")
        path = self.path / f"{safe_name}.md"
        tag_list = tags or []
        tag_str = "[" + ", ".join(tag_list) + "]"
        content = (
            f"---\nname: {name}\ndescription: {description}\ntags: {tag_str}\n---\n\n"
            f"{body.strip()}\n"
        )
        path.write_text(content)
        return parse_skill(path)

    def render_for_prompt(self, query: str, k: int = 3) -> str:
        """Render the top-k matching skills as a single string block for the
        system prompt. Returns "" if no skills match — caller can skip
        injection.
        """
        hits = self.search(query, k)
        if not hits:
            return ""
        blocks = [s.render() for s in hits]
        return "## Relevant skills from your library\n\n" + "\n\n".join(blocks)
