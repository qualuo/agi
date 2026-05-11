"""Session persistence.

Sessions need to survive process restart for any production coordinator.
This module checkpoints SessionState plus the underlying conversation
messages to disk as JSON and reloads them on demand.

Storage layout:

    {root}/
      {session_id}.json       — SessionState + agent.messages
      {session_id}.meta.json  — last-write metadata (size, ts, schema_ver)

Conversation messages are serialized via the same path the trace logger
uses — Pydantic-like content blocks lose any internal SDK refs but
retain shape sufficient for re-prompting.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agi.runtime import Session, SessionConfig, SessionState

SCHEMA_VERSION = 1


def _serialize_messages(messages: list[dict]) -> list[dict]:
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


class SessionStore:
    """File-backed session checkpoint store."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "sessions"
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session) -> Path:
        agent = session._agent
        messages = _serialize_messages(getattr(agent, "messages", []) or []) if agent else []
        last_critic = session.state.last_critic_score
        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": {
                **{k: v for k, v in asdict(session.state).items() if k != "config"},
                "config": session.state.config.__dict__,
            },
            "messages": messages,
            "last_critic_score": last_critic,
            "saved_ts": time.time(),
        }
        target = self.path / f"{session.state.id}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(target)
        return target

    def load(self, session_id: str) -> dict[str, Any]:
        target = self.path / f"{session_id}.json"
        if not target.exists():
            raise KeyError(f"no checkpoint for session {session_id}")
        return json.loads(target.read_text())

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.path.glob("*.json") if not p.name.endswith(".tmp"))

    def delete(self, session_id: str) -> bool:
        target = self.path / f"{session_id}.json"
        if target.exists():
            target.unlink()
            return True
        return False

    def hydrate(self, payload: dict[str, Any]) -> tuple[SessionState, list[dict]]:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema_version: {payload.get('schema_version')}"
            )
        state_dict = dict(payload["state"])
        config_dict = state_dict.pop("config")
        cfg = SessionConfig(**{
            k: v for k, v in config_dict.items()
            if k in SessionConfig.__dataclass_fields__
        })
        state = SessionState(config=cfg, id=state_dict["id"])
        for k, v in state_dict.items():
            if k == "id":
                continue
            if hasattr(state, k):
                setattr(state, k, v)
        return state, payload.get("messages") or []
