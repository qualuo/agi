"""Registry for tools the agent synthesized at runtime.

Synthesized tools live in two places:

1. **Session-scoped** — registered in-memory on the active Agent for the
   duration of a task; gone when the runtime restarts. Default for new
   `define_tool` calls (safe: nothing persists without explicit promotion).
2. **Persistent** — promoted to disk under `~/.agi/synth_tools/<name>.json`
   with the source + schema. Loaded by the registry at startup so subsequent
   runtime instances see them.

The promotion step is intentionally a separate tool (`promote_tool`) so the
agent must take an explicit action — the user (or coordinator) can audit
the source before promotion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agi.sandbox import SynthTool, synthesize_tool


class SynthToolRegistry:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root else Path.home() / ".agi" / "synth_tools"
        self.root.mkdir(parents=True, exist_ok=True)
        self._session: dict[str, SynthTool] = {}
        self._persistent: dict[str, SynthTool] = {}
        self._load_persistent()

    def define(self, *, name: str, description: str, input_schema: dict, source: str) -> SynthTool:
        tool = synthesize_tool(name=name, description=description, input_schema=input_schema, source=source)
        self._session[name] = tool
        return tool

    def promote(self, name: str) -> bool:
        """Promote a session tool to persistent storage."""
        tool = self._session.get(name) or self._persistent.get(name)
        if tool is None:
            return False
        path = self.root / f"{name}.json"
        path.write_text(json.dumps({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "source": tool.source,
            "created_ts": tool.created_ts,
        }, indent=2))
        self._persistent[name] = tool
        return True

    def remove(self, name: str) -> bool:
        existed = False
        if name in self._session:
            del self._session[name]
            existed = True
        if name in self._persistent:
            del self._persistent[name]
            path = self.root / f"{name}.json"
            if path.exists():
                path.unlink()
            existed = True
        return existed

    def all(self) -> dict[str, SynthTool]:
        """All tools, session-scoped overriding persistent on name conflict."""
        return {**self._persistent, **self._session}

    def _load_persistent(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                tool = synthesize_tool(
                    name=d["name"],
                    description=d["description"],
                    input_schema=d.get("input_schema") or {"type": "object", "properties": {}},
                    source=d["source"],
                )
                self._persistent[tool.name] = tool
            except Exception:
                # Skip malformed/incompatible persistent tools rather than crashing.
                continue
