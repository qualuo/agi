"""HTTP server for the runtime engine.

Stdlib-only, threaded, JSON-over-HTTP with a Server-Sent Events stream for
job output. The shape is intentionally small and stable — every endpoint is
useful to a coordination engine, and nothing depends on a third-party web
framework. Authentication is a bearer token (env: `AGI_RUNTIME_TOKEN`); if
unset, the server is open (intended for local development).

Endpoints
---------
GET    /v1/health
GET    /v1/capabilities
GET    /v1/metrics
POST   /v1/sessions                       body: {model?, budget?, agent_kwargs?}
GET    /v1/sessions
GET    /v1/sessions/{sid}
DELETE /v1/sessions/{sid}
POST   /v1/sessions/{sid}/messages        body: {content, max_iterations?}
POST   /v1/sessions/{sid}/jobs            body: {content, max_iterations?, metadata?}
GET    /v1/jobs/{jid}
GET    /v1/jobs/{jid}/stream              SSE
POST   /v1/jobs/{jid}/cancel
GET    /v1/sessions/{sid}/memory          query: q, k
POST   /v1/sessions/{sid}/memory          body: {text, tags?}
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from runtime.budgets import Budget, BudgetError
from runtime.jobs import Job
from runtime.runtime import Runtime


log = logging.getLogger("runtime.server")


_ROUTES_GET: list[tuple[re.Pattern[str], str]] = []
_ROUTES_POST: list[tuple[re.Pattern[str], str]] = []
_ROUTES_DELETE: list[tuple[re.Pattern[str], str]] = []


def _route(method_table: list[tuple[re.Pattern[str], str]], pattern: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        method_table.append((re.compile(f"^{pattern}$"), fn.__name__))
        return fn
    return deco


class Handler(BaseHTTPRequestHandler):
    runtime: Runtime  # injected by make_server
    auth_token: str | None  # injected by make_server

    server_version = "agi-runtime/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), format % args)

    # ---- helpers ----

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message, "status": status})

    def _check_auth(self) -> bool:
        if not self.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[len("Bearer ") :] == self.auth_token:
            return True
        self._send_error(HTTPStatus.UNAUTHORIZED, "missing or invalid bearer token")
        return False

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw)

    def _dispatch(self, method: str, table: list[tuple[re.Pattern[str], str]]) -> None:
        # Strip query string for matching.
        path = self.path.split("?", 1)[0]
        for pattern, name in table:
            m = pattern.match(path)
            if m:
                if not self._check_auth():
                    return
                try:
                    getattr(self, name)(**m.groupdict())
                except KeyError as e:
                    self._send_error(HTTPStatus.NOT_FOUND, str(e))
                except BudgetError as e:
                    self._send_error(HTTPStatus.PAYMENT_REQUIRED, f"budget: {e}")
                except json.JSONDecodeError as e:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {e}")
                except Exception as e:
                    log.exception("handler error in %s", name)
                    self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")
                return
        self._send_error(HTTPStatus.NOT_FOUND, f"no route for {method} {path}")

    def do_GET(self) -> None:
        self._dispatch("GET", _ROUTES_GET)

    def do_POST(self) -> None:
        self._dispatch("POST", _ROUTES_POST)

    def do_DELETE(self) -> None:
        self._dispatch("DELETE", _ROUTES_DELETE)

    # ---- routes ----

    @_route(_ROUTES_GET, r"/v1/health")
    def health(self) -> None:
        self._send_json(HTTPStatus.OK, self.runtime.health())

    @_route(_ROUTES_GET, r"/v1/capabilities")
    def capabilities(self) -> None:
        self._send_json(HTTPStatus.OK, self.runtime.manifest())

    @_route(_ROUTES_GET, r"/v1/metrics")
    def metrics(self) -> None:
        self._send_json(HTTPStatus.OK, self.runtime.metrics_snapshot())

    @_route(_ROUTES_POST, r"/v1/sessions")
    def create_session(self) -> None:
        body = self._read_json()
        session = self.runtime.create_session(
            model=body.get("model"),
            budget=Budget.from_dict(body.get("budget")),
            agent_kwargs=body.get("agent_kwargs"),
        )
        self._send_json(HTTPStatus.CREATED, session.info().to_dict())

    @_route(_ROUTES_GET, r"/v1/sessions")
    def list_sessions(self) -> None:
        self._send_json(HTTPStatus.OK, {"sessions": [s.to_dict() for s in self.runtime.sessions.list()]})

    @_route(_ROUTES_GET, r"/v1/sessions/(?P<sid>[a-zA-Z0-9_-]+)")
    def get_session(self, sid: str) -> None:
        session = self.runtime.sessions.get(sid)
        if session is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no session {sid}")
            return
        self._send_json(HTTPStatus.OK, session.info().to_dict())

    @_route(_ROUTES_DELETE, r"/v1/sessions/(?P<sid>[a-zA-Z0-9_-]+)")
    def delete_session(self, sid: str) -> None:
        ok = self.runtime.delete_session(sid)
        if not ok:
            self._send_error(HTTPStatus.NOT_FOUND, f"no session {sid}")
            return
        self._send_json(HTTPStatus.OK, {"deleted": sid})

    @_route(_ROUTES_POST, r"/v1/sessions/(?P<sid>[a-zA-Z0-9_-]+)/messages")
    def post_message(self, sid: str) -> None:
        body = self._read_json()
        content = body.get("content")
        if not content:
            self._send_error(HTTPStatus.BAD_REQUEST, "missing 'content'")
            return
        result = self.runtime.chat_sync(
            sid,
            content,
            max_iterations=int(body.get("max_iterations", 25)),
        )
        self._send_json(HTTPStatus.OK, result)

    @_route(_ROUTES_POST, r"/v1/sessions/(?P<sid>[a-zA-Z0-9_-]+)/jobs")
    def post_job(self, sid: str) -> None:
        body = self._read_json()
        content = body.get("content")
        if not content:
            self._send_error(HTTPStatus.BAD_REQUEST, "missing 'content'")
            return
        job = self.runtime.submit_job(
            sid,
            content,
            max_iterations=int(body.get("max_iterations", 25)),
            metadata=body.get("metadata"),
        )
        self._send_json(HTTPStatus.ACCEPTED, job.to_dict())

    @_route(_ROUTES_GET, r"/v1/jobs/(?P<jid>[a-zA-Z0-9_-]+)")
    def get_job(self, jid: str) -> None:
        job = self.runtime.jobs.get(jid)
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no job {jid}")
            return
        self._send_json(HTTPStatus.OK, job.to_dict())

    @_route(_ROUTES_POST, r"/v1/jobs/(?P<jid>[a-zA-Z0-9_-]+)/cancel")
    def cancel_job(self, jid: str) -> None:
        ok = self.runtime.jobs.cancel(jid)
        if not ok:
            self._send_error(HTTPStatus.NOT_FOUND, f"no job {jid}")
            return
        self._send_json(HTTPStatus.OK, {"cancel_requested": jid})

    @_route(_ROUTES_GET, r"/v1/jobs/(?P<jid>[a-zA-Z0-9_-]+)/stream")
    def stream_job(self, jid: str) -> None:
        job = self.runtime.jobs.get(jid)
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no job {jid}")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        seq = 0
        try:
            while True:
                batch = job.events_since(seq, timeout=15.0)
                if not batch:
                    # Heartbeat so proxies don't drop the connection.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    if job._done.is_set():
                        break
                    continue
                for ev in batch:
                    payload = json.dumps(ev.to_dict(), default=str).encode()
                    self.wfile.write(b"event: " + ev.kind.encode() + b"\n")
                    self.wfile.write(b"data: " + payload + b"\n\n")
                seq = batch[-1].seq + 1
                self.wfile.flush()
                if any(e.kind == "done" for e in batch):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    @_route(_ROUTES_GET, r"/v1/sessions/(?P<sid>[a-zA-Z0-9_-]+)/memory")
    def get_memory(self, sid: str) -> None:
        session = self.runtime.sessions.get(sid)
        if session is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no session {sid}")
            return
        q = self._query_param("q")
        k = int(self._query_param("k") or 10)
        if q:
            notes = session.agent.memory.search(q, k=k)
        else:
            notes = session.agent.memory.recent(k=k)
        self._send_json(
            HTTPStatus.OK,
            {"notes": [{"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags} for n in notes]},
        )

    @_route(_ROUTES_POST, r"/v1/sessions/(?P<sid>[a-zA-Z0-9_-]+)/memory")
    def post_memory(self, sid: str) -> None:
        session = self.runtime.sessions.get(sid)
        if session is None:
            self._send_error(HTTPStatus.NOT_FOUND, f"no session {sid}")
            return
        body = self._read_json()
        text = body.get("text")
        if not text:
            self._send_error(HTTPStatus.BAD_REQUEST, "missing 'text'")
            return
        note = session.agent.memory.save(text, tags=body.get("tags") or [])
        self._send_json(HTTPStatus.CREATED, {"id": note.id, "ts": note.ts, "text": note.text, "tags": note.tags})

    def _query_param(self, name: str) -> str | None:
        if "?" not in self.path:
            return None
        from urllib.parse import parse_qs
        qs = parse_qs(self.path.split("?", 1)[1])
        vals = qs.get(name)
        return vals[0] if vals else None


def make_server(
    runtime: Runtime,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    token = auth_token if auth_token is not None else os.environ.get("AGI_RUNTIME_TOKEN")
    handler_cls = type(
        "BoundHandler",
        (Handler,),
        {"runtime": runtime, "auth_token": token},
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    return server


def serve_forever(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
    runtime: Runtime | None = None,
) -> None:
    runtime = runtime or Runtime()
    server = make_server(runtime, host=host, port=port, auth_token=auth_token)
    log.info("agi-runtime listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.shutdown()
        runtime.shutdown()


def serve_in_thread(runtime: Runtime, *, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Start the server on a background thread. Returns (server, thread, base_url).

    Mainly for tests: port=0 binds an ephemeral port; read it back from the
    server's socket.
    """
    server = make_server(runtime, host=host, port=port)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="agi-runtime-server")
    thread.start()
    return server, thread, f"http://{host}:{actual_port}"
