"""HTTP + SSE surface for the Runtime.

Lets an external coordination engine drive a Runtime over the network.
Built on stdlib `http.server` to avoid pulling in FastAPI/uvicorn —
trade-off: it's not high-throughput, but it's correct, dependency-free,
and easy to reason about. Swap for FastAPI/Starlette if you need scale.

Endpoints:
  POST /v1/sessions               { "goal": "...", "budget": {...}, ... } -> { "session_id" }
  GET  /v1/sessions               -> [ {record}, ... ]
  GET  /v1/sessions/{id}          -> {record}
  POST /v1/sessions/{id}/cancel   -> { "ok": true }
  GET  /v1/sessions/{id}/events   -> SSE stream of events
  GET  /v1/health                 -> { "ok": true, "concurrent": N }
  POST /v1/plans                  { "name": "...", "subgoals": [...] } -> { "results": {...} }
                                                      (synchronous; useful for tests + small plans)

Auth: optional shared-secret header `Authorization: Bearer <token>`.
Set `AGI_API_TOKEN` env var to require it; absence disables auth.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

from agi.budget import Budget
from agi.plan import execute_plan, plan_from_dict
from agi.runtime import Runtime


def _budget_from_dict(d: Optional[dict[str, Any]]) -> Optional[Budget]:
    if not d:
        return None
    return Budget(
        max_cost_usd=d.get("max_cost_usd"),
        max_tokens=d.get("max_tokens"),
        max_iterations=d.get("max_iterations"),
        max_wall_seconds=d.get("max_wall_seconds"),
    )


class _Handler(BaseHTTPRequestHandler):
    runtime: Runtime  # set on the subclass below
    auth_token: Optional[str] = None

    # Quieter access logs by default.
    def log_message(self, format: str, *args: Any) -> None:
        pass

    # ----- helpers -----------------------------------------------------------

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bad(self, msg: str, status: int = 400) -> None:
        self._json(status, {"error": msg})

    def _auth_ok(self) -> bool:
        if not self.auth_token:
            return True
        hdr = self.headers.get("Authorization", "")
        return hdr == f"Bearer {self.auth_token}"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode())

    # ----- dispatch ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return self._bad("unauthorized", 401)
        path = urlparse(self.path).path
        if path == "/v1/health":
            return self._json(200, {"ok": True, "concurrent": self.runtime.max_concurrent})
        if path == "/v1/sessions":
            return self._json(200, self.runtime.list_sessions())
        if path.startswith("/v1/sessions/"):
            rest = path[len("/v1/sessions/"):]
            if rest.endswith("/events"):
                sid = rest[: -len("/events")]
                return self._sse(sid)
            try:
                return self._json(200, self.runtime.status(rest))
            except KeyError:
                return self._bad("no such session", 404)
        return self._bad("not found", 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return self._bad("unauthorized", 401)
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            return self._bad("invalid json")

        if path == "/v1/sessions":
            goal = body.get("goal")
            if not goal or not isinstance(goal, str):
                return self._bad("missing 'goal'")
            try:
                sid = self.runtime.submit(
                    goal,
                    budget=_budget_from_dict(body.get("budget")),
                    max_iterations=int(body.get("max_iterations", 25)),
                    tags=dict(body.get("tags", {})),
                )
            except Exception as e:
                return self._bad(f"{type(e).__name__}: {e}", 500)
            return self._json(202, {"session_id": sid})

        if path.startswith("/v1/sessions/") and path.endswith("/cancel"):
            sid = path[len("/v1/sessions/"): -len("/cancel")]
            try:
                self.runtime.cancel(sid)
            except KeyError:
                return self._bad("no such session", 404)
            return self._json(200, {"ok": True})

        if path == "/v1/plans":
            try:
                plan = plan_from_dict(body)
            except (KeyError, ValueError) as e:
                return self._bad(f"invalid plan: {e}")
            timeout = body.get("timeout")
            results = execute_plan(self.runtime, plan, timeout=timeout)
            return self._json(200, {"results": {k: asdict(v) for k, v in results.items()}})

        return self._bad("not found", 404)

    # ----- SSE ---------------------------------------------------------------

    def _sse(self, session_id: str) -> None:
        try:
            stream = self.runtime.events(session_id, replay=True, follow=True)
        except KeyError:
            return self._bad("no such session", 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for event in stream:
                payload = json.dumps(event.to_dict(), default=str)
                chunk = f"event: {event.kind}\ndata: {payload}\n\n".encode()
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def make_server(
    runtime: Runtime,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: Optional[str] = None,
) -> ThreadingHTTPServer:
    token = auth_token if auth_token is not None else os.environ.get("AGI_API_TOKEN")

    class Handler(_Handler):
        pass

    Handler.runtime = runtime
    Handler.auth_token = token
    return ThreadingHTTPServer((host, port), Handler)


def serve(
    runtime: Optional[Runtime] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: Optional[str] = None,
) -> None:
    """Blocking serve. Creates a default Runtime if none is provided."""
    rt = runtime or Runtime()
    server = make_server(rt, host=host, port=port, auth_token=auth_token)
    print(f"agi runtime listening on http://{host}:{port}")
    if (auth_token or os.environ.get("AGI_API_TOKEN")):
        print("auth: Bearer token required")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        rt.shutdown()


def serve_in_thread(
    runtime: Runtime,
    host: str = "127.0.0.1",
    port: int = 0,
    auth_token: Optional[str] = None,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the server on a background thread. Useful for tests."""
    server = make_server(runtime, host=host, port=port, auth_token=auth_token)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t
