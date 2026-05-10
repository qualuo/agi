"""Runtime protocol — JSON-line stdio control surface.

Lets an external coordination engine drive an agent as a subprocess. Each
line on stdin is a JSON request; the runtime emits JSON-line events and a
terminal result/error on stdout. Both sides are append-only; no in-band
control characters.

Why a subprocess protocol?
- The agent becomes a swappable building block. A coordinator can pool many
  runtimes (different models, different skill libraries, different memory
  scopes) and route work between them.
- Crash isolation: a misbehaving agent doesn't take down the coordinator.
- Language-agnostic: a coordinator written in TypeScript, Go, or Rust can
  drive the runtime without knowing Python.

## Wire format

Every message is a single line of JSON. Newlines inside string values are
escaped as `\n` per JSON spec.

### Coordinator → Runtime (requests)

    {"id": "<arbitrary>", "type": "hello"}
    {"id": "...", "type": "capabilities"}
    {"id": "...", "type": "chat", "input": "<user message>", "max_iterations": 25}
    {"id": "...", "type": "reset"}
    {"id": "...", "type": "memory.save", "text": "...", "tags": [...]}
    {"id": "...", "type": "memory.search", "query": "...", "k": 5}
    {"id": "...", "type": "memory.recent", "k": 10}
    {"id": "...", "type": "skills.list"}
    {"id": "...", "type": "skills.find", "query": "...", "k": 5}
    {"id": "...", "type": "skills.read", "name": "..."}
    {"id": "...", "type": "shutdown"}

### Runtime → Coordinator (responses)

    {"type": "ready", "version": "<n>"}                   # on startup
    {"type": "event", "req_id": "...", "kind": "...", ...} # streamed during chat
    {"type": "result", "req_id": "...", "ok": true, ...}  # terminal success
    {"type": "error", "req_id": "...", "message": "..."}  # terminal failure
    {"type": "bye"}                                       # on shutdown

### Event kinds (during chat)

    thinking_start | thinking_delta | text_start | text_delta
    tool_use_start | tool_call (full input) | tool_result
    server_tool_use_start

Coordinators that only care about the final answer can ignore events and
wait for the `result` envelope.

## Concurrency

One request at a time. The runtime is intentionally single-threaded; if you
need parallelism, run multiple runtime subprocesses behind the coordinator.
That's the whole point of this design.

## Versioning

Bump `PROTOCOL_VERSION` when the wire format changes. The `ready` and
`capabilities` responses both carry the version so coordinators can refuse
incompatible runtimes.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import IO, Any

from agi.agent import Agent
from agi.memory import Memory
from agi.skills import SkillLibrary


PROTOCOL_VERSION = 1


def _write(stream: IO[str], obj: dict[str, Any]) -> None:
    stream.write(json.dumps(obj, default=str) + "\n")
    stream.flush()


class Runtime:
    """JSON-line runtime around an `Agent`.

    `serve()` blocks reading from `stdin` and writing to `stdout` until a
    `shutdown` request arrives (or stdin closes). Inject custom streams in
    tests by passing `stdin`/`stdout`.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
    ) -> None:
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self._current_req_id: str | None = None
        # The runtime owns the Agent; it wires its own event callback in.
        if agent is None:
            agent = Agent(
                skills=SkillLibrary(),
                verbose=False,
                on_event=self._on_agent_event,
            )
        else:
            agent.verbose = False
            agent.on_event = self._on_agent_event
        self.agent: Agent = agent

    # -- event plumbing -------------------------------------------------

    def _on_agent_event(self, event: dict) -> None:
        if self._current_req_id is None:
            return
        out = {"type": "event", "req_id": self._current_req_id}
        out.update(event)
        _write(self.stdout, out)

    # -- request handlers ----------------------------------------------

    def _handle(self, req: dict) -> dict:
        kind = req.get("type")
        if kind == "hello":
            return {"ok": True, "version": PROTOCOL_VERSION, "model": self.agent.model}

        if kind == "capabilities":
            return {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "model": self.agent.model,
                "tools": [s["name"] for s in self.agent.tool_schemas],
                "has_skills": self.agent.skills is not None,
                "has_critic": self.agent.critic is not None,
                "has_tracer": self.agent.tracer is not None,
            }

        if kind == "chat":
            user_input = req.get("input")
            if not isinstance(user_input, str) or not user_input:
                raise ValueError("chat requires non-empty string 'input'")
            max_iterations = int(req.get("max_iterations", 25))
            text = self.agent.chat(user_input, max_iterations=max_iterations)
            return {
                "ok": True,
                "text": text,
                "model": self.agent.model,
                "usage": {
                    "input_tokens": self.agent.usage.input_tokens,
                    "output_tokens": self.agent.usage.output_tokens,
                    "cache_creation_input_tokens": self.agent.usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": self.agent.usage.cache_read_input_tokens,
                    "turns": self.agent.usage.turns,
                    "cost_usd": self.agent.usage.cost_usd(self.agent.model),
                },
                "critic_score": self.agent.last_critic_score,
            }

        if kind == "reset":
            self.agent.reset()
            return {"ok": True}

        if kind == "memory.save":
            text = req.get("text")
            tags = req.get("tags") or []
            if not isinstance(text, str) or not text:
                raise ValueError("memory.save requires non-empty string 'text'")
            note = self.agent.memory.save(text, list(tags))
            return {"ok": True, "id": note.id, "ts": note.ts}

        if kind == "memory.search":
            query = req.get("query") or ""
            k = int(req.get("k", 5))
            results = self.agent.memory.search(query, k)
            return {
                "ok": True,
                "results": [
                    {"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags}
                    for n in results
                ],
            }

        if kind == "memory.recent":
            k = int(req.get("k", 10))
            results = self.agent.memory.recent(k)
            return {
                "ok": True,
                "results": [
                    {"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags}
                    for n in results
                ],
            }

        if kind == "skills.list":
            if self.agent.skills is None:
                return {"ok": True, "skills": []}
            return {
                "ok": True,
                "skills": [
                    {"name": s.name, "when": s.when, "tags": s.tags}
                    for s in self.agent.skills.all()
                ],
            }

        if kind == "skills.find":
            if self.agent.skills is None:
                return {"ok": True, "skills": []}
            query = req.get("query") or ""
            k = int(req.get("k", 5))
            return {
                "ok": True,
                "skills": [
                    {"name": s.name, "when": s.when, "tags": s.tags}
                    for s in self.agent.skills.search(query, k)
                ],
            }

        if kind == "skills.read":
            if self.agent.skills is None:
                return {"ok": False, "error": "no skill library configured"}
            name = req.get("name") or ""
            s = self.agent.skills.get(name)
            if s is None:
                return {"ok": False, "error": f"no skill named {name!r}"}
            return {
                "ok": True,
                "name": s.name,
                "when": s.when,
                "tags": s.tags,
                "body": s.body,
            }

        raise ValueError(f"unknown request type {kind!r}")

    # -- main loop ------------------------------------------------------

    def serve(self) -> int:
        _write(self.stdout, {"type": "ready", "version": PROTOCOL_VERSION})
        for raw_line in self.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                _write(
                    self.stdout,
                    {"type": "error", "req_id": None, "message": f"invalid json: {e}"},
                )
                continue
            if not isinstance(req, dict):
                _write(
                    self.stdout,
                    {"type": "error", "req_id": None, "message": "request must be an object"},
                )
                continue
            req_id = req.get("id")
            self._current_req_id = req_id if isinstance(req_id, str) else None

            if req.get("type") == "shutdown":
                _write(self.stdout, {"type": "result", "req_id": req_id, "ok": True})
                _write(self.stdout, {"type": "bye"})
                return 0

            try:
                payload = self._handle(req)
                _write(
                    self.stdout,
                    {"type": "result", "req_id": req_id, **payload},
                )
            except Exception as e:
                _write(
                    self.stdout,
                    {
                        "type": "error",
                        "req_id": req_id,
                        "message": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    },
                )
            finally:
                self._current_req_id = None
        return 0


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Allow the runtime to start without a key only if the caller plans
        # to use memory/skill ops; chat will fail at API time anyway.
        pass
    return Runtime().serve()


if __name__ == "__main__":
    sys.exit(main())
