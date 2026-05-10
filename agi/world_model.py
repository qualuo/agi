"""World model — observed-entity tracker.

The agent's working memory tracks the *conversation*. The world model tracks
*things the agent has interacted with* in a structured form: files it has
read/written, URLs it has fetched, shell commands it has run, named entities
the user has mentioned. Each observation is timestamped with an outcome,
allowing the agent to answer "have I seen this before?" and "what happened
last time?" without re-deriving from raw memory.

This is intentionally minimal — it's a structured log, not a causal graph.
But it gives the coordination engine real state to plan over: "the runtime
has read /etc/foo, last result was success; skip the read."
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Observation:
    entity_kind: str          # "file" | "url" | "command" | "entity"
    entity_id: str             # path / url / canonical command / name
    action: str                # "read" | "write" | "fetch" | "run" | "mention"
    outcome: str               # "success" | "failure" | "unknown"
    ts: float
    detail: dict[str, Any] = field(default_factory=dict)


class WorldModel:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "world.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        # In-memory index: (kind, id) -> latest Observation
        self._latest: dict[tuple[str, str], Observation] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    obs = Observation(**d)
                    self._latest[(obs.entity_kind, obs.entity_id)] = obs
                except Exception:
                    continue

    def observe(
        self,
        *,
        kind: str,
        id: str,
        action: str,
        outcome: str = "success",
        detail: dict[str, Any] | None = None,
    ) -> Observation:
        obs = Observation(
            entity_kind=kind, entity_id=id, action=action, outcome=outcome,
            ts=time.time(), detail=detail or {},
        )
        with self._lock:
            self._latest[(kind, id)] = obs
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(obs), default=str) + "\n")
        return obs

    def latest(self, kind: str, id: str) -> Observation | None:
        return self._latest.get((kind, id))

    def known(self, kind: str) -> list[Observation]:
        return [o for (k, _), o in self._latest.items() if k == kind]

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        last_failures: list[Observation] = []
        for obs in self._latest.values():
            by_kind[obs.entity_kind] = by_kind.get(obs.entity_kind, 0) + 1
            if obs.outcome == "failure":
                last_failures.append(obs)
        last_failures.sort(key=lambda o: o.ts, reverse=True)
        return {
            "entity_counts": by_kind,
            "total_entities": len(self._latest),
            "recent_failures": [asdict(o) for o in last_failures[:5]],
        }
