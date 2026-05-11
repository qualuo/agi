"""HTTP + SSE transport for the Runtime.

Wraps `agi.runtime.Runtime` in a small stdlib-only HTTP server so a
coordination engine in a different process (or different language) can
drive it. The wire format is JSON; long-running tasks stream Server-Sent
Events on `/v1/sessions/{id}/tasks/{tid}/events`.

This is intentionally stdlib-only — no FastAPI / uvicorn dependency — so
the runtime stays cheap to embed in any environment. Production callers
that need TLS, auth, or concurrency limits should put this behind a
reverse proxy or replace the transport.

Endpoints:

    GET    /v1/health
    GET    /v1/capabilities
    POST   /v1/sessions                     -> open a session
    GET    /v1/sessions/{sid}               -> session info
    DELETE /v1/sessions/{sid}               -> close a session
    POST   /v1/sessions/{sid}/tasks         -> run a task (sync, blocking)
    POST   /v1/sessions/{sid}/tasks?async=1 -> queue a task (non-blocking)
    GET    /v1/sessions/{sid}/tasks/{tid}   -> poll task state
    POST   /v1/sessions/{sid}/tasks/{tid}/cancel
    GET    /v1/sessions/{sid}/tasks/{tid}/events  (SSE stream)
    GET    /v1/skills                       -> list skills
    POST   /v1/skills                       -> add a skill
    DELETE /v1/skills/{id}                  -> remove a skill
    POST   /v1/skills/{id}/promote          -> promote a skill

Auth: optional bearer token via the AGI_RUNTIME_TOKEN env var. When set,
every request must carry `Authorization: Bearer <token>`. When unset (dev
default) the server is open — bind to localhost only.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

from agi.budget import Budget
from agi.runtime import Runtime, SessionConfig


_AUTH_TOKEN = os.environ.get("AGI_RUNTIME_TOKEN")


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload) -> None:
    body = json.dumps(payload, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("X-AGI-Runtime", "1")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _authed(handler: BaseHTTPRequestHandler) -> bool:
    if not _AUTH_TOKEN:
        return True
    header = handler.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return False
    return header[len("Bearer "):].strip() == _AUTH_TOKEN


# Route patterns compiled once.
_ROUTES = [
    ("GET",    re.compile(r"^/v1/health$")),
    ("GET",    re.compile(r"^/v1/capabilities$")),
    ("POST",   re.compile(r"^/v1/sessions$")),
    ("GET",    re.compile(r"^/v1/sessions/(?P<sid>[^/]+)$")),
    ("DELETE", re.compile(r"^/v1/sessions/(?P<sid>[^/]+)$")),
    ("POST",   re.compile(r"^/v1/sessions/(?P<sid>[^/]+)/tasks$")),
    ("GET",    re.compile(r"^/v1/sessions/(?P<sid>[^/]+)/tasks/(?P<tid>[^/]+)$")),
    ("POST",   re.compile(r"^/v1/sessions/(?P<sid>[^/]+)/tasks/(?P<tid>[^/]+)/cancel$")),
    ("GET",    re.compile(r"^/v1/sessions/(?P<sid>[^/]+)/tasks/(?P<tid>[^/]+)/events$")),
    ("GET",    re.compile(r"^/v1/skills$")),
    ("POST",   re.compile(r"^/v1/skills$")),
    ("DELETE", re.compile(r"^/v1/skills/(?P<id>[^/]+)$")),
    ("POST",   re.compile(r"^/v1/skills/(?P<id>[^/]+)/promote$")),
]


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Quieter logs
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        # Dispatch each method through one matcher so we don't repeat routing.
        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            if not _authed(self):
                _json_response(self, 401, {"error": "unauthorized"})
                return
            url = urlparse(self.path)
            path = url.path
            for m, rx in _ROUTES:
                if m != method:
                    continue
                match = rx.match(path)
                if not match:
                    continue
                params = match.groupdict()
                params["_query"] = parse_qs(url.query)
                try:
                    self._handle(path, method, params)
                except Exception as e:
                    _json_response(self, 500, {"error": str(e), "type": type(e).__name__})
                return
            _json_response(self, 404, {"error": "not found", "path": path})

        # ---- handlers ----

        def _handle(self, path: str, method: str, params: dict) -> None:
            if path == "/v1/health":
                _json_response(self, 200, {"ok": True, "version": runtime.version, "uptime_s": time.time() - runtime.started_at})
                return
            if path == "/v1/capabilities":
                _json_response(self, 200, runtime.capabilities())
                return
            if path == "/v1/sessions" and method == "POST":
                self._open_session()
                return
            if path == "/v1/skills" and method == "GET":
                self._list_skills()
                return
            if path == "/v1/skills" and method == "POST":
                self._add_skill()
                return

            sid = params.get("sid")
            tid = params.get("tid")
            skill_id = params.get("id")

            if sid and tid and path.endswith("/events"):
                self._stream_events(sid, tid)
                return
            if sid and tid and path.endswith("/cancel"):
                ok = runtime.cancel_task(sid, tid)
                _json_response(self, 200 if ok else 404, {"cancelled": ok})
                return
            if sid and tid:
                self._task_status(sid, tid)
                return
            if sid and path.endswith("/tasks") and method == "POST":
                self._run_task(sid, params["_query"])
                return
            if sid and method == "GET":
                self._session_info(sid)
                return
            if sid and method == "DELETE":
                ok = runtime.close_session(sid)
                _json_response(self, 200 if ok else 404, {"closed": ok})
                return
            if skill_id and method == "DELETE":
                ok = runtime.skills.remove(skill_id)
                _json_response(self, 200 if ok else 404, {"removed": ok})
                return
            if skill_id and path.endswith("/promote") and method == "POST":
                body = _read_json(self)
                s = runtime.skills.promote(skill_id, eval_pass_rate=body.get("eval_pass_rate"))
                _json_response(self, 200 if s else 404, s.to_dict() if s else {"error": "not found"})
                return
            _json_response(self, 404, {"error": "no handler", "path": path})

        def _open_session(self) -> None:
            body = _read_json(self)
            cfg = SessionConfig.from_dict(body)
            session = runtime.open_session(cfg)
            _json_response(self, 201, session.to_dict())

        def _session_info(self, sid: str) -> None:
            session = runtime.get_session(sid)
            if session is None:
                _json_response(self, 404, {"error": "unknown session"})
                return
            _json_response(self, 200, session.to_dict())

        def _run_task(self, sid: str, query: dict) -> None:
            body = _read_json(self)
            input_text = body.get("input") or body.get("prompt") or ""
            if not input_text:
                _json_response(self, 400, {"error": "missing 'input'"})
                return
            budget = Budget.from_dict(body.get("budget")) if body.get("budget") else None
            task_id = body.get("task_id")
            is_async = bool(query.get("async") or query.get("stream"))
            try:
                if is_async:
                    tid = runtime.start_task(sid, input_text, budget=budget, task_id=task_id)
                    _json_response(self, 202, {"task_id": tid, "session_id": sid, "status": "running"})
                else:
                    result = runtime.run_task(sid, input_text, budget=budget, task_id=task_id)
                    _json_response(self, 200, result.to_dict())
            except KeyError:
                _json_response(self, 404, {"error": "unknown session"})

        def _task_status(self, sid: str, tid: str) -> None:
            t = runtime.get_task(sid, tid)
            if t is None:
                _json_response(self, 404, {"error": "unknown task"})
                return
            if t.result is not None:
                _json_response(self, 200, {**t.result.to_dict(), "task_status": t.status})
                return
            _json_response(self, 200, {"task_id": tid, "session_id": sid, "task_status": t.status})

        def _stream_events(self, sid: str, tid: str) -> None:
            t = runtime.get_task(sid, tid)
            if t is None:
                _json_response(self, 404, {"error": "unknown task"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for event in runtime.events(sid, tid, timeout=300.0):
                    chunk = f"event: {event.get('type', 'event')}\ndata: {json.dumps(event, default=str)}\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def _list_skills(self) -> None:
            promoted_only = self.path.endswith("?promoted_only=1")  # cheap query handling
            skills = runtime.skills.all(promoted_only=promoted_only)
            _json_response(self, 200, {"skills": [s.to_dict() for s in skills]})

        def _add_skill(self) -> None:
            body = _read_json(self)
            desc = body.get("description") or ""
            content = body.get("body") or body.get("content") or ""
            triggers = body.get("triggers") or []
            promoted = bool(body.get("promoted", False))
            if not desc or not content:
                _json_response(self, 400, {"error": "description and body are required"})
                return
            skill = runtime.skills.add(desc, content, triggers=triggers, promoted=promoted)
            _json_response(self, 201, skill.to_dict())

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8088,
    runtime: Runtime | None = None,
    *,
    on_ready: Callable[[ThreadingHTTPServer], None] | None = None,
) -> ThreadingHTTPServer:
    runtime = runtime or Runtime()
    server = ThreadingHTTPServer((host, port), make_handler(runtime))
    server.runtime = runtime  # type: ignore[attr-defined]
    if on_ready:
        on_ready(server)
    return server


def serve_forever(host: str = "127.0.0.1", port: int = 8088, runtime: Runtime | None = None) -> None:
    server = serve(host, port, runtime)
    print(f"agi runtime serving on http://{host}:{port} (auth: {'on' if _AUTH_TOKEN else 'off'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.shutdown()


def _spawn_in_thread(host: str = "127.0.0.1", port: int = 0, runtime: Runtime | None = None) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Test helper — start the server on a background thread, return (server, thread).

    Use port=0 to bind an ephemeral port; the assigned port is on
    `server.server_address[1]`.
    """
    server = serve(host, port, runtime)
    thread = threading.Thread(target=server.serve_forever, name="agi-runtime-http", daemon=True)
    thread.start()
    return server, thread
