"""Reusable runtime client.

Spawns `python -m agi --runtime` as a subprocess and exposes a simple
synchronous request/response API. Streamed events (tool calls, text deltas)
are surfaced via an `on_event` callback the caller supplies.

Usage:

    with RuntimeClient() as rt:
        caps = rt.request("capabilities")
        result = rt.chat("write hello world to /tmp/x.txt", on_event=print)
        print(result["text"])

Designed to be the substrate a coordination engine uses to pool many
runtimes. Open as many `RuntimeClient` instances as you want; each is an
isolated process.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


class RuntimeError(Exception):
    """Raised when the runtime returns a structured error envelope."""


class RuntimeClient:
    def __init__(
        self,
        cmd: list[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if cmd is None:
            cmd = [sys.executable, "-m", "agi", "--runtime"]
        if cwd is None:
            cwd = Path(__file__).resolve().parent.parent
        self._cmd = cmd
        self._cwd = str(cwd)
        self._env = env
        self._proc: subprocess.Popen | None = None
        self._counter = itertools.count(1)

    # -- lifecycle ------------------------------------------------------

    def __enter__(self) -> "RuntimeClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> dict:
        if self._proc is not None:
            raise RuntimeError("runtime already started")
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self._cwd,
            env=self._env or os.environ.copy(),
        )
        ready = self._read_message()
        if ready.get("type") != "ready":
            raise RuntimeError(f"expected ready, got {ready!r}")
        return ready

    def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        try:
            if proc.poll() is None:
                # Best-effort graceful shutdown.
                try:
                    self._send({"id": self._next_id(), "type": "shutdown"})
                    # Drain remaining output.
                    while proc.poll() is None:
                        line = proc.stdout.readline()
                        if not line:
                            break
                except (BrokenPipeError, OSError):
                    pass
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, BrokenPipeError):
                        pass
            self._proc = None

    # -- protocol -------------------------------------------------------

    def _next_id(self) -> str:
        return f"r{next(self._counter)}"

    def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _read_message(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            stderr_tail = ""
            if self._proc.stderr is not None:
                stderr_tail = self._proc.stderr.read() or ""
            raise RuntimeError(
                "runtime closed unexpectedly"
                + (f" — stderr:\n{stderr_tail}" if stderr_tail else "")
            )
        return json.loads(line)

    def request(
        self,
        type_: str,
        on_event: Callable[[dict], None] | None = None,
        **fields: Any,
    ) -> dict:
        """Send one request and read until the terminal `result` or `error`.

        Forwards any `event` messages to `on_event` if provided.
        """
        if self._proc is None:
            raise RuntimeError("runtime not started")
        rid = self._next_id()
        self._send({"id": rid, "type": type_, **fields})
        while True:
            msg = self._read_message()
            mtype = msg.get("type")
            if mtype == "event":
                if on_event is not None:
                    on_event(msg)
                continue
            if mtype == "result" and msg.get("req_id") == rid:
                return msg
            if mtype == "error" and msg.get("req_id") == rid:
                raise RuntimeError(msg.get("message", "unknown error"))
            # Out-of-band messages (e.g., events from earlier calls): drop.

    # -- ergonomic shortcuts -------------------------------------------

    def chat(
        self,
        text: str,
        max_iterations: int = 25,
        on_event: Callable[[dict], None] | None = None,
    ) -> dict:
        return self.request(
            "chat", on_event=on_event, input=text, max_iterations=max_iterations
        )

    def reset(self) -> dict:
        return self.request("reset")

    def capabilities(self) -> dict:
        return self.request("capabilities")

    def memory_save(self, text: str, tags: Iterable[str] | None = None) -> dict:
        return self.request("memory.save", text=text, tags=list(tags or []))

    def memory_search(self, query: str, k: int = 5) -> dict:
        return self.request("memory.search", query=query, k=k)

    def skills_list(self) -> dict:
        return self.request("skills.list")

    def skills_find(self, query: str, k: int = 5) -> dict:
        return self.request("skills.find", query=query, k=k)
