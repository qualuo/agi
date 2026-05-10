"""HTTP control plane for the Runtime.

Stdlib-only (http.server) so a non-Python coordination engine can drive the
runtime over HTTP without needing FastAPI/Flask. JSON in, JSON out. Events
are streamed as Server-Sent Events (text/event-stream) so curl can tail a
job in real time.

This is deliberately small: routing is a switch on (method, path). For
production a proper web framework + auth + TLS belongs in front. The point
of this module is to give a coordinator a stable wire protocol to test
against.

    GET  /healthz                  → {"ok": true}
    GET  /metrics                  → runtime metrics snapshot
    POST /v1/sessions              → create session  {role?, system_prompt?, model?}
    GET  /v1/sessions              → list sessions
    GET  /v1/sessions/{id}         → get one session
    POST /v1/sessions/{id}/jobs    → submit job  {prompt, budget_usd?, max_iterations?}
    GET  /v1/jobs                  → list jobs (?session_id=...)
    GET  /v1/jobs/{id}             → get job (poll)
    POST /v1/jobs/{id}/cancel      → cancel
    GET  /v1/jobs/{id}/events      → SSE stream of events until job finishes
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agi.runtime import Runtime


_ROUTES: list[tuple[str, str, str]] = [
    # (method, regex, handler_name)
    ("GET",  r"^/healthz$",                                 "healthz"),
    ("GET",  r"^/metrics$",                                 "metrics"),
    ("POST", r"^/v1/sessions$",                             "create_session"),
    ("GET",  r"^/v1/sessions$",                             "list_sessions"),
    ("GET",  r"^/v1/sessions/(?P<sid>[^/]+)$",              "get_session"),
    ("POST", r"^/v1/sessions/(?P<sid>[^/]+)/jobs$",         "submit_job"),
    ("GET",  r"^/v1/jobs$",                                 "list_jobs"),
    ("GET",  r"^/v1/jobs/(?P<jid>[^/]+)$",                  "get_job"),
    ("POST", r"^/v1/jobs/(?P<jid>[^/]+)/cancel$",           "cancel_job"),
    ("GET",  r"^/v1/jobs/(?P<jid>[^/]+)/events$",           "stream_events"),
]


def make_handler(runtime: "Runtime") -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Silence default access logging — the runtime owns observability.
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            for m, pattern, name in _ROUTES:
                if m != method:
                    continue
                match = re.match(pattern, path)
                if match:
                    handler = getattr(self, "_h_" + name)
                    try:
                        handler(match.groupdict(), _parse_query(query))
                    except _HTTPError as e:
                        self._json(e.status, {"error": e.message})
                    except Exception as e:
                        self._json(500, {"error": f"{type(e).__name__}: {e}"})
                    return
            self._json(404, {"error": "not found", "path": path})

        # ---- handlers ----

        def _h_healthz(self, params, query) -> None:
            self._json(200, {"ok": True})

        def _h_metrics(self, params, query) -> None:
            self._json(200, runtime.metrics())

        def _h_create_session(self, params, query) -> None:
            body = self._read_json()
            rec = runtime.create_session(
                role=body.get("role"),
                system_prompt=body.get("system_prompt"),
                model=body.get("model", "claude-opus-4-7"),
                metadata=body.get("metadata") or {},
            )
            self._json(200, asdict(rec))

        def _h_list_sessions(self, params, query) -> None:
            self._json(200, [asdict(s) for s in runtime.list_sessions()])

        def _h_get_session(self, params, query) -> None:
            try:
                rec = runtime.get_session(params["sid"])
            except KeyError as e:
                raise _HTTPError(404, str(e))
            self._json(200, asdict(rec))

        def _h_submit_job(self, params, query) -> None:
            body = self._read_json()
            if "prompt" not in body:
                raise _HTTPError(400, "missing 'prompt'")
            try:
                rec = runtime.submit(
                    params["sid"],
                    body["prompt"],
                    budget_usd=body.get("budget_usd"),
                    max_iterations=body.get("max_iterations", 25),
                    metadata=body.get("metadata") or {},
                )
            except KeyError as e:
                raise _HTTPError(404, str(e))
            self._json(200, asdict(rec))

        def _h_list_jobs(self, params, query) -> None:
            sid = query.get("session_id")
            self._json(200, [asdict(j) for j in runtime.list_jobs(session_id=sid)])

        def _h_get_job(self, params, query) -> None:
            try:
                rec = runtime.get_job(params["jid"])
            except KeyError as e:
                raise _HTTPError(404, str(e))
            self._json(200, asdict(rec))

        def _h_cancel_job(self, params, query) -> None:
            try:
                rec = runtime.cancel(params["jid"])
            except KeyError as e:
                raise _HTTPError(404, str(e))
            self._json(200, asdict(rec))

        def _h_stream_events(self, params, query) -> None:
            try:
                stream = runtime.stream(params["jid"])
            except KeyError as e:
                raise _HTTPError(404, str(e))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for ev in stream:
                    chunk = (
                        f"event: {ev.kind}\n"
                        f"data: {json.dumps({'ts': ev.ts, 'payload': ev.payload})}\n\n"
                    )
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected; leave the job running.
                return

        # ---- helpers ----

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or "0")
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode())
            except json.JSONDecodeError as e:
                raise _HTTPError(400, f"invalid json: {e}")

        def _json(self, status: int, body) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(runtime: "Runtime", host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Start the control-plane HTTP server in a background thread.

    Returns the server; call .shutdown() to stop. For long-running deployment
    use serve_forever() in the foreground."""
    handler_cls = make_handler(runtime)
    server = ThreadingHTTPServer((host, port), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True, name="agi-http").start()
    return server


class _HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _parse_query(qs: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not qs:
        return out
    for pair in qs.split("&"):
        if not pair:
            continue
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
        else:
            out[pair] = ""
    return out
