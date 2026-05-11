"""HTTP/JSON surface for the Runtime.

A coordination engine in another process (or another language) talks to
the runtime through this server. Stdlib-only — no FastAPI dependency to
keep the install footprint small. Sufficient for a v1 product surface
that demonstrates the contract.

Endpoints (all JSON unless noted):

  GET    /v1/health
  GET    /v1/capabilities
  GET    /v1/usage
  POST   /v1/sessions                     {metadata?} -> {session_id, ...}
  GET    /v1/sessions
  GET    /v1/sessions/{id}
  DELETE /v1/sessions/{id}
  POST   /v1/sessions/{id}/run            {prompt, idempotency_key?, run_id?, max_iterations?}
  POST   /v1/sessions/{id}/cancel
  GET    /v1/sessions/{id}/events         (text/event-stream; SSE)
  GET    /v1/sessions/{id}/events/recent  {limit?} -> list of events
  GET    /v1/skills
  POST   /v1/skills                       {name, description, body, tags?}

Auth: a single shared bearer token, read from env. v1 is single-tenant;
real multi-tenant auth lives one layer up.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from agi.events import Event
from agi.runtime import Runtime, RUNTIME_API_VERSION


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Event):
        return o.to_dict()
    return str(o)


class _Handler(BaseHTTPRequestHandler):
    server_version = f"agi-runtime/{RUNTIME_API_VERSION}"

    # Sub-classed with .runtime / .auth_token wired in via make_handler.
    runtime: Runtime
    auth_token: str | None

    # ---- helpers ----

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e

    def _authorized(self) -> bool:
        if not self.auth_token:
            return True
        h = self.headers.get("Authorization", "")
        return h == f"Bearer {self.auth_token}"

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib name
        # Quiet by default; the runtime emits its own structured events.
        pass

    # ---- dispatch ----

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if not self._authorized():
            return self._send_error_json(401, "unauthorized")
        try:
            self._dispatch_get()
        except KeyError as e:
            self._send_error_json(404, str(e))
        except Exception as e:
            self._send_error_json(500, f"{type(e).__name__}: {e}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        if not self._authorized():
            return self._send_error_json(401, "unauthorized")
        try:
            self._dispatch_post()
        except KeyError as e:
            self._send_error_json(404, str(e))
        except ValueError as e:
            self._send_error_json(400, str(e))
        except Exception as e:
            self._send_error_json(500, f"{type(e).__name__}: {e}")

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib name
        if not self._authorized():
            return self._send_error_json(401, "unauthorized")
        try:
            self._dispatch_delete()
        except Exception as e:
            self._send_error_json(500, f"{type(e).__name__}: {e}")

    def _dispatch_get(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/v1/health":
            return self._send_json(200, self.runtime.health())
        if path == "/v1/capabilities":
            return self._send_json(200, self.runtime.capabilities())
        if path == "/v1/usage":
            return self._send_json(200, self.runtime.aggregate_usage())
        if path == "/v1/sessions":
            return self._send_json(200, {"sessions": [asdict(s) for s in self.runtime.list_sessions()]})
        if path == "/v1/skills":
            if self.runtime.skills is None:
                return self._send_json(200, {"skills": []})
            return self._send_json(200, {
                "skills": [
                    {"name": s.name, "description": s.description, "tags": s.tags}
                    for s in self.runtime.skills.list()
                ]
            })
        parts = path.strip("/").split("/")
        # /v1/sessions/{id}
        if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            return self._send_json(200, asdict(self.runtime.get_session(parts[2])))
        # /v1/sessions/{id}/events
        if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "events":
            return self._stream_events(parts[2])
        # /v1/sessions/{id}/events/recent
        if len(parts) == 5 and parts[:2] == ["v1", "sessions"] and parts[3:5] == ["events", "recent"]:
            limit = 100
            qs = urlparse(self.path).query
            for kv in qs.split("&"):
                if kv.startswith("limit="):
                    try:
                        limit = int(kv.split("=", 1)[1])
                    except ValueError:
                        pass
            evts = self.runtime.recent_events(parts[2], limit=limit)
            return self._send_json(200, {"events": [e.to_dict() for e in evts]})
        self._send_error_json(404, f"no route for GET {path}")

    def _dispatch_post(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/v1/sessions":
            body = self._read_json()
            sid = self.runtime.create_session(
                session_id=body.get("session_id"),
                metadata=body.get("metadata"),
            )
            return self._send_json(201, asdict(self.runtime.get_session(sid)))
        if path == "/v1/skills":
            body = self._read_json()
            for k in ("name", "description", "body"):
                if k not in body:
                    raise ValueError(f"missing required field: {k}")
            info = self.runtime.upsert_skill(
                name=body["name"],
                description=body["description"],
                body=body["body"],
                tags=body.get("tags") or [],
            )
            return self._send_json(201, info)

        parts = path.strip("/").split("/")
        # /v1/sessions/{id}/run
        if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "run":
            sid = parts[2]
            body = self._read_json()
            if "prompt" not in body:
                raise ValueError("missing required field: prompt")
            result = self.runtime.run(
                sid,
                body["prompt"],
                max_iterations=int(body.get("max_iterations", 25)),
                idempotency_key=body.get("idempotency_key"),
                run_id=body.get("run_id"),
            )
            return self._send_json(200, result.to_dict())
        # /v1/sessions/{id}/cancel
        if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "cancel":
            self.runtime.cancel(parts[2])
            return self._send_json(202, {"cancelled": True})
        self._send_error_json(404, f"no route for POST {path}")

    def _dispatch_delete(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        parts = path.strip("/").split("/")
        # /v1/sessions/{id}
        if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            ok = self.runtime.destroy_session(parts[2])
            return self._send_json(200, {"destroyed": ok})
        self._send_error_json(404, f"no route for DELETE {path}")

    # ---- SSE ----

    def _stream_events(self, session_id: str) -> None:
        qs = urlparse(self.path).query
        replay = "replay=1" in qs or "replay=true" in qs

        try:
            q, unsubscribe = self.runtime.subscribe(session_id, replay=replay)
        except KeyError:
            return self._send_error_json(404, f"unknown session {session_id!r}")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()

        # Heartbeat once a session is idle so coord engines can detect liveness.
        try:
            last_beat = time.time()
            while True:
                try:
                    evt = q.get(timeout=10)
                except queue.Empty:
                    if time.time() - last_beat >= 10:
                        try:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        last_beat = time.time()
                    continue
                payload = json.dumps(evt.to_dict(), default=_json_default)
                try:
                    self.wfile.write(f"event: {evt.kind}\n".encode())
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_beat = time.time()
                if evt.kind == "run.finished":
                    # Don't auto-close; the coordinator may want to drive
                    # multiple runs over the same stream. A client closes
                    # by disconnecting (handled above).
                    pass
        finally:
            unsubscribe()


def make_handler(runtime: Runtime, auth_token: str | None = None) -> type[_Handler]:
    """Bind a runtime + token into a handler class usable by ThreadingHTTPServer."""

    class Bound(_Handler):
        pass

    Bound.runtime = runtime
    Bound.auth_token = auth_token
    return Bound


class RuntimeServer:
    """Thin wrapper around ThreadingHTTPServer. Run with `serve_forever`
    or `start`/`stop` for tests."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        host: str = "127.0.0.1",
        port: int = 7777,
        auth_token: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._httpd is None:
            return (self.host, self.port)
        return self._httpd.server_address  # type: ignore[return-value]

    def start(self) -> None:
        handler = make_handler(self.runtime, auth_token=self.auth_token)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def serve_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> int:
    host = os.environ.get("AGI_RUNTIME_HOST", "127.0.0.1")
    port = int(os.environ.get("AGI_RUNTIME_PORT", "7777"))
    token = os.environ.get("AGI_RUNTIME_TOKEN") or None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: ANTHROPIC_API_KEY is not set — /v1/sessions/{id}/run will fail.")
    runtime = Runtime()
    server = RuntimeServer(runtime, host=host, port=port, auth_token=token)
    print(f"agi-runtime serving on http://{host}:{port}  (api v{RUNTIME_API_VERSION})")
    if token:
        print("auth: Bearer token required (AGI_RUNTIME_TOKEN)")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
