"""Persistent memory store.

Append-only JSONL on disk. Search is keyword + tag for now; embeddings would
slot in behind the same `search` interface.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Note:
    id: str
    ts: float
    text: str
    tags: list[str] = field(default_factory=list)


class Memory:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "memory.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def save(self, text: str, tags: list[str] | None = None) -> Note:
        note = Note(id=uuid.uuid4().hex[:12], ts=time.time(), text=text, tags=tags or [])
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(note)) + "\n")
        return note

    def all(self) -> list[Note]:
        notes: list[Note] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                notes.append(Note(**d))
        return notes

    def search(self, query: str, k: int = 5) -> list[Note]:
        q = query.lower().strip()
        terms = [t for t in q.split() if t]
        scored: list[tuple[int, Note]] = []
        for note in self.all():
            hay = (note.text + " " + " ".join(note.tags)).lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, note))
        scored.sort(key=lambda x: (-x[0], -x[1].ts))
        return [n for _, n in scored[:k]]

    def recent(self, k: int = 10) -> list[Note]:
        return list(reversed(self.all()))[:k]
