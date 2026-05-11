"""JSON-Lines runtime server.

The runtime engine exposed as a line-oriented protocol over stdin/stdout (or
any pair of file-like objects). A coordination engine drives this:

    Client → server (one JSON object per line):
      {"cmd": "submit",  "prompt": "...", "budget": {...}, "role": "...",
       "skills": [...], "metadata": {...}}
      {"cmd": "events",  "run_id": "..."}    # subscribe to live stream
      {"cmd": "status",  "run_id": "..."}    # snapshot
      {"cmd": "list"}                        # all runs
      {"cmd": "cancel",  "run_id": "...", "reason": "..."}
      {"cmd": "shutdown"}

    Server → client (one JSON object per line):
      {"type": "submitted", "run_id": "...", ...}
      {"type": "event", "run_id": "...", "event": {...}}
      {"type": "status",  "run_id": "...", "run": {...}}
      {"type": "list",    "runs": [...]}
      {"type": "ack",     "cmd": "cancel", "run_id": "..."}
      {"type": "error",   "message": "..."}
      {"type": "shutdown"}

Concurrency: a single background thread per `events` subscription forwards
the run's queue to stdout. Multiple subscriptions on the same run are
supported.

This module is intentionally dependency-free beyond the rest of `agi`.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import IO, Any

from agi import events as ev
from agi.runtime import Budget, RunSpec, Runtime


class JSONLinesServer:
    def __init__(
        self,
        runtime: Runtime | None = None,
        *,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
    ) -> None:
        self.runtime = runtime or Runtime()
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self._write_lock = threading.Lock()
        self._shutdown = threading.Event()

    def serve_forever(self) -> int:
        """Read commands line-by-line; return 0 on clean shutdown."""
        for line in iter(self.stdin.readline, ""):
            line = line.strip()
            if not line:
                continue
            if self._shutdown.is_set():
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                self._send({"type": "error", "message": f"invalid JSON: {e}"})
                continue
            self._dispatch(msg)
            if self._shutdown.is_set():
                break
        self._send({"type": "shutdown"})
        return 0

    def _send(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, default=str)
        with self._write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def _dispatch(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "submit":
            self._cmd_submit(msg)
        elif cmd == "events":
            self._cmd_events(msg)
        elif cmd == "status":
            self._cmd_status(msg)
        elif cmd == "list":
            self._cmd_list(msg)
        elif cmd == "cancel":
            self._cmd_cancel(msg)
        elif cmd == "shutdown":
            self._shutdown.set()
        else:
            self._send({"type": "error", "message": f"unknown cmd {cmd!r}"})

    def _cmd_submit(self, msg: dict) -> None:
        prompt = msg.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send({"type": "error", "message": "submit requires non-empty 'prompt'"})
            return
        budget_dict = msg.get("budget") or {}
        try:
            budget = Budget(**{k: v for k, v in budget_dict.items() if v is not None})
        except TypeError as e:
            self._send({"type": "error", "message": f"bad budget: {e}"})
            return
        spec = RunSpec(
            prompt=prompt,
            role=msg.get("role", "general"),
            model=msg.get("model", "claude-opus-4-7"),
            effort=msg.get("effort", "high"),
            max_iterations=int(msg.get("max_iterations", 25)),
            budget=budget,
            skills=list(msg.get("skills") or []),
            metadata=dict(msg.get("metadata") or {}),
        )
        run = self.runtime.submit(spec)
        self._send({
            "type": "submitted",
            "run_id": run.id,
            "spec": spec.to_dict(),
        })
        if msg.get("subscribe"):
            self._spawn_event_forwarder(run.id)

    def _cmd_events(self, msg: dict) -> None:
        run_id = msg.get("run_id")
        if not isinstance(run_id, str):
            self._send({"type": "error", "message": "events requires 'run_id'"})
            return
        if self.runtime.get(run_id) is None:
            self._send({"type": "error", "message": f"unknown run {run_id!r}"})
            return
        self._spawn_event_forwarder(run_id)

    def _spawn_event_forwarder(self, run_id: str) -> None:
        run = self.runtime.get(run_id)
        if run is None:
            return
        # Forward what's already buffered before subscribing to future events.
        for past in run.events():
            self._send({"type": "event", "run_id": run_id, "event": past.to_dict()})
            if past.type in ("run_completed", "run_failed", "run_cancelled"):
                return

        def forward() -> None:
            for event in run.iter_events():
                self._send({"type": "event", "run_id": run_id, "event": event.to_dict()})

        t = threading.Thread(target=forward, daemon=True, name=f"forward-{run_id}")
        t.start()

    def _cmd_status(self, msg: dict) -> None:
        run_id = msg.get("run_id")
        if not isinstance(run_id, str):
            self._send({"type": "error", "message": "status requires 'run_id'"})
            return
        run = self.runtime.get(run_id)
        if run is None:
            # Try the on-disk registry.
            record = self.runtime.load_record(run_id)
            if record is None:
                self._send({"type": "error", "message": f"unknown run {run_id!r}"})
                return
            self._send({"type": "status", "run_id": run_id, "run": record})
            return
        self._send({"type": "status", "run_id": run_id, "run": run.to_dict()})

    def _cmd_list(self, msg: dict) -> None:
        self._send({
            "type": "list",
            "runs": [r.to_dict() for r in self.runtime.list_runs()],
        })

    def _cmd_cancel(self, msg: dict) -> None:
        run_id = msg.get("run_id")
        if not isinstance(run_id, str):
            self._send({"type": "error", "message": "cancel requires 'run_id'"})
            return
        run = self.runtime.get(run_id)
        if run is None:
            self._send({"type": "error", "message": f"unknown run {run_id!r}"})
            return
        run.cancel(msg.get("reason") or "cancelled by coordinator")
        self._send({"type": "ack", "cmd": "cancel", "run_id": run_id})


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write("warning: ANTHROPIC_API_KEY not set — submit will fail\n")
    server = JSONLinesServer()
    return server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
