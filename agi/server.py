"""HTTP control plane + SSE event stream.

Stdlib-only — no FastAPI / Flask / Starlette dependency. The point is
that any coordination engine in any language can drive this runtime by
making JSON requests and tailing /events.

Endpoints
---------
GET  /                       human-readable index
GET  /healthz                liveness check
GET  /v1/manifest            capability descriptor (roles, tools, events)
GET  /v1/sessions            list live sessions (JSON array of SessionInfo)
POST /v1/sessions            open a new session
                             body: {role?, model?, max_usd?, max_turns?, system_extra?, tags?}
GET  /v1/sessions/<id>       one SessionInfo
DEL  /v1/sessions/<id>       close a session
POST /v1/sessions/<id>/chat  body: {input: str}
                             returns: chat() result dict
GET  /v1/events              Server-Sent Events stream of all runtime events
GET  /v1/history             ring-buffer of recent events (JSON array)

The server is multi-threaded (one thread per request) so SSE clients
don't block control-plane requests. Auth is opt-in via a static bearer
token: set `AGI_API_TOKEN=...` and clients must send `Authorization:
Bearer <token>`.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from agi.budget import Budget
from agi.runtime import Runtime


_SSE_HEARTBEAT = 15.0  # seconds between keepalive pings


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, default=str).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    # set by build_server
    runtime: Runtime
    auth_token: str | None

    # -- helpers -----------------------------------------------------------

    def log_message(self, format: str, *args) -> None:  # quieter access log
        pass

    def _check_auth(self) -> bool:
        if not self.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.auth_token}"

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _send_json(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, body: str, content_type: str = "text/plain") -> None:
        b = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _route(self) -> tuple[str, list[str]]:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        parts = [p for p in path.split("/") if p]
        return path, parts

    # -- dispatch ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (HTTP server convention)
        path, parts = self._route()
        if path == "/":
            self._send_text(
                200,
                "agi-runtime — see /v1/manifest, /v1/sessions, /v1/events\n",
            )
            return
        if path == "/healthz":
            self._send_text(200, "ok\n")
            return
        if not self._check_auth():
            self._err(401, "unauthorized")
            return
        if path == "/v1/manifest":
            self._send_json(200, self.runtime.manifest())
            return
        if path == "/v1/sessions":
            sessions = [self._session_info_dict(i) for i in self.runtime.list_sessions()]
            self._send_json(200, {"sessions": sessions})
            return
        if len(parts) == 3 and parts[0] == "v1" and parts[1] == "sessions":
            sid = parts[2]
            session = self.runtime.get_session(sid)
            if session is None:
                self._err(404, f"unknown session: {sid}")
                return
            self._send_json(200, self._session_info_dict(session.info()))
            return
        if path == "/v1/history":
            self._send_json(200, {"events": self.runtime.history()})
            return
        if path == "/v1/events":
            self._stream_events()
            return
        self._err(404, f"no route for GET {path}")

    def do_POST(self) -> None:  # noqa: N802
        path, parts = self._route()
        if not self._check_auth():
            self._err(401, "unauthorized")
            return
        try:
            body = self._read_body()
        except ValueError as exc:
            self._err(400, str(exc))
            return
        if path == "/v1/sessions":
            self._open_session(body)
            return
        if len(parts) == 4 and parts[0] == "v1" and parts[1] == "sessions" and parts[3] == "chat":
            self._chat(parts[2], body)
            return
        self._err(404, f"no route for POST {path}")

    def do_DELETE(self) -> None:  # noqa: N802
        path, parts = self._route()
        if not self._check_auth():
            self._err(401, "unauthorized")
            return
        if len(parts) == 3 and parts[0] == "v1" and parts[1] == "sessions":
            sid = parts[2]
            ok = self.runtime.close_session(sid)
            if not ok:
                self._err(404, f"unknown session: {sid}")
                return
            self._send_json(200, {"closed": sid})
            return
        self._err(404, f"no route for DELETE {path}")

    # -- handlers ----------------------------------------------------------

    def _open_session(self, body: dict) -> None:
        try:
            budget = None
            if body.get("max_usd") is not None or body.get("max_turns") is not None:
                budget = Budget(
                    max_usd=body.get("max_usd"),
                    max_turns=body.get("max_turns"),
                    model=body.get("model") or self.runtime.model,
                )
            session = self.runtime.open_session(
                role=body.get("role", "general"),
                model=body.get("model"),
                budget=budget,
                system_extra=body.get("system_extra"),
                tags=body.get("tags"),
                effort=body.get("effort", "high"),
                parent_id=body.get("parent_id"),
            )
        except Exception as exc:
            self._err(400, f"open_session failed: {type(exc).__name__}: {exc}")
            return
        self._send_json(201, self._session_info_dict(session.info()))

    def _chat(self, session_id: str, body: dict) -> None:
        text = body.get("input") or body.get("text")
        if not text:
            self._err(400, "missing 'input' field")
            return
        session = self.runtime.get_session(session_id)
        if session is None:
            self._err(404, f"unknown session: {session_id}")
            return
        try:
            result = session.chat(text)
        except Exception as exc:
            self._err(
                500,
                f"chat failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
            return
        self._send_json(200, result)

    def _stream_events(self) -> None:
        # Per-connection queue subscribed to the runtime bus.
        q: queue.Queue[dict] = queue.Queue(maxsize=1024)
        unsub = self.runtime.subscribe(lambda e: _safe_put(q, e))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # nginx hint
        self.end_headers()

        try:
            # Replay recent history so a late subscriber catches up.
            for e in self.runtime.history():
                self._write_event(e)
            last_ping = time.time()
            while True:
                try:
                    evt = q.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - last_ping >= _SSE_HEARTBEAT:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
                    continue
                self._write_event(evt)
                last_ping = time.time()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            unsub()

    def _write_event(self, evt: dict) -> None:
        type_ = evt.get("type", "message")
        data = json.dumps(evt, default=str)
        # SSE frame: event line + data line + blank
        self.wfile.write(f"event: {type_}\n".encode("utf-8"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _session_info_dict(self, info) -> dict:
        return {
            "id": info.id,
            "role": info.role,
            "model": info.model,
            "created_at": info.created_at,
            "parent_id": info.parent_id,
            "turns": info.turns,
            "cost_usd": info.cost_usd,
            "last_critic_score": info.last_critic_score,
            "closed": info.closed,
            "tags": list(info.tags),
        }


def _safe_put(q: queue.Queue, evt: dict) -> None:
    try:
        q.put_nowait(evt)
    except queue.Full:
        # drop oldest then enqueue — better than blocking the bus
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(evt)
        except queue.Full:
            pass


def build_server(
    runtime: Runtime,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    """Construct (but don't start) an HTTP server bound to `runtime`."""
    handler_cls = type(
        "BoundHandler",
        (_Handler,),
        {"runtime": runtime, "auth_token": auth_token},
    )
    return ThreadingHTTPServer((host, port), handler_cls)


def serve(
    runtime: Runtime | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
) -> None:
    """Start the server in the foreground. Ctrl-C to stop."""
    if runtime is None:
        runtime = Runtime()
    if auth_token is None:
        auth_token = os.environ.get("AGI_API_TOKEN")
    httpd = build_server(runtime, host=host, port=port, auth_token=auth_token)
    print(f"agi-runtime serving on http://{host}:{port}")
    print(f"  manifest:  http://{host}:{port}/v1/manifest")
    print(f"  events:    http://{host}:{port}/v1/events")
    if auth_token:
        print(f"  auth:      Bearer {auth_token[:6]}…")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
        runtime.close_all()


def serve_in_thread(
    runtime: Runtime,
    host: str = "127.0.0.1",
    port: int = 0,
    auth_token: str | None = None,
) -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    """Start the server in a daemon thread; return (httpd, thread, port).

    Useful for tests and for in-process embedding. port=0 binds an
    ephemeral port; the returned int is the actual port chosen.
    """
    httpd = build_server(runtime, host=host, port=port, auth_token=auth_token)
    actual_port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, name="agi-http", daemon=True)
    t.start()
    return httpd, t, actual_port
