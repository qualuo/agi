"""Stdlib HTTP + SSE server over AgentRuntime.

This makes the runtime callable from any language. A coordination engine
written in TypeScript, Go, Rust, or another Python process can drive sessions
over HTTP with no SDK install.

Endpoints (JSON in/out unless noted):

  GET  /v1/health
       → {"ok": true, "sessions": <count>}

  POST /v1/sessions
       body: {"session_id"?: str, "config"?: {...SessionConfig fields...}}
       → {"session_id": str}

  DELETE /v1/sessions/{id}
       → {"closed": true}

  POST /v1/sessions/{id}/send
       body: {"prompt": str, "max_iterations"?: int}
       → text/event-stream; each line `event: <kind>\\ndata: <json>\\n\\n`
       Terminates after `turn_completed` (or `budget_exceeded` /
       `runtime_error`).

  GET  /v1/sessions/{id}/snapshot
       → snapshot JSON

  POST /v1/sessions/{id}/snapshot
       body: snapshot JSON (from a prior GET)
       → {"session_id": str, "restored": true}

Auth: if env var `AGI_RUNTIME_TOKEN` is set, every request must carry
`Authorization: Bearer <token>` (constant-time compared).

Threading model: `ThreadingHTTPServer` handles each request on a thread.
Sessions are guarded by a simple per-session lock so concurrent `send`
requests on the same session serialize cleanly (preventing message-list
corruption). Cross-session traffic is fully parallel.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agi.events import Event
from agi.runtime import AgentRuntime, SessionConfig


log = logging.getLogger("agi.server")


SESSION_PATH = re.compile(r"^/v1/sessions/([A-Za-z0-9_-]+)(/[^/?]*)?$")


def _auth_ok(headers, expected_token: str | None) -> bool:
    if not expected_token:
        return True
    auth = headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth[len("Bearer ") :].strip(), expected_token)


def _make_handler(runtime: AgentRuntime, token: str | None):
    """Return a handler class closed over the runtime instance."""

    # Per-session locks ensure two concurrent `send` calls on the same
    # session don't interleave on `agent.messages`. We don't bother locking
    # snapshot/start/close — those manipulate dicts that are already covered
    # by the GIL for single ops.
    session_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def _lock_for(sid: str) -> threading.Lock:
        with locks_guard:
            lk = session_locks.get(sid)
            if lk is None:
                lk = threading.Lock()
                session_locks[sid] = lk
            return lk

    class _Handler(BaseHTTPRequestHandler):
        server_version = "agi-runtime/0.1"

        def log_message(self, fmt, *args):  # quieter than the default stderr spew
            log.debug("%s - " + fmt, self.address_string(), *args)

        # ---- helpers ----

        def _json(self, status: int, body: dict | list) -> None:
            payload = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, msg: str) -> None:
            self._json(status, {"error": msg})

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON: {e}") from e

        def _authed(self) -> bool:
            if _auth_ok(self.headers, token):
                return True
            self._error(HTTPStatus.UNAUTHORIZED, "missing or invalid bearer token")
            return False

        # ---- routes ----

        def do_GET(self):  # noqa: N802 (http.server contract)
            if not self._authed():
                return
            if self.path == "/v1/health":
                self._json(HTTPStatus.OK, {"ok": True, "sessions": len(runtime.sessions)})
                return
            m = SESSION_PATH.match(self.path)
            if m and m.group(2) == "/snapshot":
                sid = m.group(1)
                try:
                    snap = runtime.snapshot(sid)
                except KeyError:
                    self._error(HTTPStatus.NOT_FOUND, f"no session {sid}")
                    return
                self._json(HTTPStatus.OK, snap)
                return
            self._error(HTTPStatus.NOT_FOUND, f"no route for GET {self.path}")

        def do_DELETE(self):  # noqa: N802
            if not self._authed():
                return
            m = SESSION_PATH.match(self.path)
            if m and m.group(2) is None:
                sid = m.group(1)
                runtime.close_session(sid)
                self._json(HTTPStatus.OK, {"closed": True})
                return
            self._error(HTTPStatus.NOT_FOUND, f"no route for DELETE {self.path}")

        def do_POST(self):  # noqa: N802
            if not self._authed():
                return
            try:
                body = self._read_body()
            except ValueError as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
                return

            if self.path == "/v1/sessions":
                cfg_dict = body.get("config") or {}
                try:
                    cfg = SessionConfig(**cfg_dict)
                except TypeError as e:
                    self._error(HTTPStatus.BAD_REQUEST, f"bad config: {e}")
                    return
                try:
                    sid = runtime.start_session(
                        session_id=body.get("session_id"), config=cfg
                    )
                except ValueError as e:
                    self._error(HTTPStatus.CONFLICT, str(e))
                    return
                self._json(HTTPStatus.CREATED, {"session_id": sid})
                return

            m = SESSION_PATH.match(self.path)
            if not m:
                self._error(HTTPStatus.NOT_FOUND, f"no route for POST {self.path}")
                return
            sid = m.group(1)
            tail = m.group(2)

            if tail == "/send":
                prompt = body.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    self._error(HTTPStatus.BAD_REQUEST, "missing 'prompt'")
                    return
                max_it = int(body.get("max_iterations") or 25)
                self._stream_send(sid, prompt, max_it)
                return

            if tail == "/snapshot":
                try:
                    runtime.restore(body)
                except (KeyError, ValueError) as e:
                    self._error(HTTPStatus.BAD_REQUEST, str(e))
                    return
                self._json(HTTPStatus.OK, {"session_id": sid, "restored": True})
                return

            self._error(HTTPStatus.NOT_FOUND, f"no route for POST {self.path}")

        # ---- SSE streaming ----

        def _stream_send(self, sid: str, prompt: str, max_iterations: int) -> None:
            try:
                runtime.get_session(sid)
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, f"no session {sid}")
                return

            # Open SSE response. Force connection close so the client knows
            # the response is complete once the stream ends — there's no
            # Content-Length on a streamed response, and we don't multiplex
            # multiple SSE streams on one connection.
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
            self.end_headers()

            lock = _lock_for(sid)
            with lock:
                try:
                    for ev in runtime.send(sid, prompt, max_iterations=max_iterations):
                        self._send_event(ev)
                        if ev.kind in ("turn_completed", "budget_exceeded", "runtime_error"):
                            break
                except BrokenPipeError:
                    # Client disconnected mid-stream; not an error.
                    return
                except Exception as e:
                    self._send_raw_event(
                        "runtime_error",
                        {"error_type": type(e).__name__, "message": str(e)},
                    )

        def _send_event(self, event: Event) -> None:
            self._send_raw_event(event.kind, event.to_dict())

        def _send_raw_event(self, kind: str, data: dict) -> None:
            payload = f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
            try:
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                raise

    return _Handler


def serve(
    runtime: AgentRuntime | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """Build and return a server bound to (host, port). Call `serve_forever`
    on the returned instance to run it, or use `server.shutdown()` from
    another thread to stop it cleanly.

    Auth token defaults to the `AGI_RUNTIME_TOKEN` env var.
    """
    runtime = runtime or AgentRuntime()
    token = token if token is not None else os.environ.get("AGI_RUNTIME_TOKEN")
    handler_cls = _make_handler(runtime, token)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    # Stash the runtime so tests can inspect it
    httpd.runtime = runtime  # type: ignore[attr-defined]
    return httpd


def main() -> int:
    """`python -m agi.server` — start the runtime server."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("AGI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGI_PORT", "8765")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    httpd = serve(host=args.host, port=args.port)
    log.info("agi runtime listening on http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
