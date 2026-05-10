"""HTTP server: the wire contract a coordination engine talks to.

Stdlib-only (http.server + threading). No Flask/FastAPI dependency. The
intended consumer is a coordination engine: it POSTs work to /sessions and
streams progress over /sessions/{id}/events (SSE).

Endpoints
---------

GET  /                         -> health + capabilities
GET  /capabilities             -> capabilities only
POST /sessions                 -> create a session; body: { budget?, role?, agent? }
GET  /sessions                 -> list sessions
GET  /sessions/{id}            -> session info
POST /sessions/{id}/step       -> body: { input }; returns StepResult
POST /sessions/{id}/snapshot   -> returns serialized session state
DELETE /sessions/{id}          -> close the session
GET  /sessions/{id}/events     -> SSE stream of lifecycle events
                                  (?since=<seq> replays from a sequence number)
POST /sessions/restore         -> body: snapshot dict; resumes a parked session

This is *intentionally* not REST-pure — it's a JSON-over-HTTP control plane
optimized for a coordinator driving multiple agents. Auth is left to the
embedder (run behind a private network or front with a reverse proxy).
"""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agi.events import Event, EventBus
from agi.runtime import Budget, Runtime


def _json_body(handler: "RuntimeHandler") -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _send_json(handler: "RuntimeHandler", status: int, payload: Any) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler: "RuntimeHandler", status: int, msg: str) -> None:
    _send_json(handler, status, {"error": msg})


class RuntimeHandler(BaseHTTPRequestHandler):
    runtime: Runtime  # set by serve()

    # Suppress noisy default access logs.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    # -- routing --

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not parts:
            return self._handle_root()
        if parts == ["capabilities"]:
            return _send_json(self, 200, self.runtime.capabilities().to_dict())
        if parts == ["sessions"]:
            return _send_json(self, 200, [s.to_dict() for s in self.runtime.list_sessions()])
        if len(parts) == 2 and parts[0] == "sessions":
            session = self.runtime.get(parts[1])
            if session is None:
                return _send_error(self, 404, "session not found")
            return _send_json(self, 200, session.info().to_dict())
        if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "events":
            return self._handle_sse(parts[1], parse_qs(url.query))
        _send_error(self, 404, f"unknown path: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        body = _json_body(self)
        if parts == ["sessions"]:
            return self._handle_create_session(body)
        if parts == ["sessions", "restore"]:
            return self._handle_restore_session(body)
        if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "step":
            return self._handle_step(parts[1], body)
        if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "snapshot":
            return self._handle_snapshot(parts[1])
        _send_error(self, 404, f"unknown path: {self.path}")

    def do_DELETE(self) -> None:  # noqa: N802
        parts = [p for p in self.path.split("/") if p]
        if len(parts) == 2 and parts[0] == "sessions":
            ok = self.runtime.close(parts[1])
            return _send_json(self, 200 if ok else 404, {"closed": ok})
        _send_error(self, 404, f"unknown path: {self.path}")

    # -- handlers --

    def _handle_root(self) -> None:
        _send_json(
            self,
            200,
            {
                "service": "agi-runtime",
                "ok": True,
                "capabilities": self.runtime.capabilities().to_dict(),
            },
        )

    def _handle_create_session(self, body: dict) -> None:
        budget = Budget.from_dict(body.get("budget"))
        role = body.get("role")
        agent_kwargs = body.get("agent") or {}
        # Whitelist agent kwargs to avoid arbitrary attribute injection.
        allowed = {"model", "max_tokens", "effort", "enable_web_search", "enable_web_fetch", "extra_system"}
        agent_kwargs = {k: v for k, v in agent_kwargs.items() if k in allowed}
        try:
            session = self.runtime.create_session(budget=budget, role=role, agent_kwargs=agent_kwargs)
        except Exception as e:
            return _send_error(self, 500, f"failed to create session: {type(e).__name__}: {e}")
        _send_json(self, 201, session.info().to_dict())

    def _handle_restore_session(self, body: dict) -> None:
        snapshot = body.get("snapshot") or body
        try:
            session = self.runtime.restore_session(snapshot)
        except Exception as e:
            return _send_error(self, 400, f"failed to restore: {type(e).__name__}: {e}")
        _send_json(self, 201, session.info().to_dict())

    def _handle_step(self, session_id: str, body: dict) -> None:
        session = self.runtime.get(session_id)
        if session is None:
            return _send_error(self, 404, "session not found")
        user_input = body.get("input")
        if not isinstance(user_input, str) or not user_input.strip():
            return _send_error(self, 400, "missing 'input' string in body")
        result = session.step(user_input)
        _send_json(self, 200, result.to_dict())

    def _handle_snapshot(self, session_id: str) -> None:
        session = self.runtime.get(session_id)
        if session is None:
            return _send_error(self, 404, "session not found")
        _send_json(self, 200, session.snapshot())

    def _handle_sse(self, session_id: str, params: dict[str, list[str]]) -> None:
        session = self.runtime.get(session_id)
        if session is None:
            return _send_error(self, 404, "session not found")
        since_seq = 0
        if "since" in params:
            try:
                since_seq = int(params["since"][0])
            except (ValueError, IndexError):
                since_seq = 0

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Replay buffered tail first.
        for ev in session.bus.replay(since_seq):
            if not self._write_sse(ev):
                return

        # Then live-subscribe.
        q: "queue.Queue[Event | None]" = queue.Queue(maxsize=2048)

        def push(ev: Event) -> None:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass

        unsubscribe = session.bus.subscribe(push)
        try:
            while True:
                try:
                    ev = q.get(timeout=15.0)
                except queue.Empty:
                    # Keep-alive comment so proxies don't time out.
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    continue
                if ev is None or not self._write_sse(ev):
                    return
        finally:
            unsubscribe()

    def _write_sse(self, ev: Event) -> bool:
        try:
            payload = json.dumps(ev.to_dict(), default=str)
            self.wfile.write(f"event: {ev.type}\nid: {ev.seq}\ndata: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False


def serve(host: str = "127.0.0.1", port: int = 8765, runtime: Runtime | None = None) -> ThreadingHTTPServer:
    """Start the runtime HTTP server on (host, port). Returns the server
    object; call .serve_forever() (blocks) or .shutdown() in another thread."""
    rt = runtime or Runtime()

    class _BoundHandler(RuntimeHandler):
        pass

    _BoundHandler.runtime = rt
    server = ThreadingHTTPServer((host, port), _BoundHandler)
    server.runtime = rt  # type: ignore[attr-defined]
    return server


def serve_forever(host: str = "127.0.0.1", port: int = 8765, runtime: Runtime | None = None) -> None:
    server = serve(host, port, runtime)
    print(f"agi-runtime listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
