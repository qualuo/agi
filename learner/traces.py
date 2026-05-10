"""Trace logger.

Every agent interaction is logged as a Trace — a JSONL record of the full
conversation, tool usage, final output, and metadata (task id, eval result,
user rating, anything we want to filter on later). The training pipeline
consumes these traces.

Traces are append-only and durable across restarts. The default location
is `~/.agi/traces.jsonl` but the path is configurable.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Trace:
    id: str
    ts: float
    model: str
    messages: list[dict]
    final_text: str
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceLogger:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "traces.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(
        self,
        *,
        model: str,
        messages: list[dict],
        final_text: str,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trace:
        trace = Trace(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            model=model,
            messages=_serialize_messages(messages),
            final_text=final_text,
            usage=usage or {},
            metadata=metadata or {},
        )
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(trace), default=str) + "\n")
        return trace

    def all(self) -> list[Trace]:
        traces: list[Trace] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                traces.append(Trace(**d))
        return traces


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Convert in-memory messages (which may contain SDK Pydantic blocks)
    into plain dicts safe to JSON-encode."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif isinstance(content, list):
            blocks = []
            for b in content:
                if hasattr(b, "model_dump"):
                    blocks.append(b.model_dump(exclude_none=True))
                elif isinstance(b, dict):
                    blocks.append(b)
                else:
                    blocks.append({"type": "unknown", "repr": repr(b)})
            out.append({"role": m["role"], "content": blocks})
        else:
            out.append({"role": m["role"], "content": repr(content)})
    return out
