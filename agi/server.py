"""HTTP server exposing the Runtime over JSON.

This is the surface a coordination engine binds to. Stdlib-only — no flask,
no fastapi — to keep the deploy story trivial: anywhere Python runs.

Endpoints:

    GET  /v1/health
    GET  /v1/describe
    GET  /v1/sessions
    POST /v1/sessions                       — body: {"session_id": "..."} (optional)
    GET  /v1/sessions/{id}
    DELETE /v1/sessions/{id}
    POST /v1/sessions/{id}/reset
    POST /v1/sessions/{id}/turn             — body: {"input": "...", "max_iterations": 25}

Auth is intentionally out of scope here. Wrap behind your own gateway, or
add a bearer-token check in `_authenticated()`. Errors return JSON with an
`error` field. All requests/responses are JSON.

Run with:

    python -m agi serve --host 127.0.0.1 --port 8088 [--memory-root path/]
"""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from agi.runtime import Runtime


_log = logging.getLogger("agi.server")


def _make_handler(runtime: Runtime, auth_token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # ---------- helpers ----------

        def _write_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json: {e}")

        def _authenticated(self) -> bool:
            if auth_token is None:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {auth_token}"

        def log_message(self, format: str, *args) -> None:  # quieter default logs
            _log.info("%s - %s", self.address_string(), format % args)

        # ---------- routing ----------

        def do_GET(self) -> None:  # noqa: N802
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            try:
                self._route_get()
            except Exception as e:
                self._write_json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            try:
                self._route_post()
            except ValueError as e:
                self._write_json(400, {"error": str(e)})
            except KeyError as e:
                self._write_json(404, {"error": str(e)})
            except Exception as e:
                self._write_json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            try:
                self._route_delete()
            except KeyError as e:
                self._write_json(404, {"error": str(e)})
            except Exception as e:
                self._write_json(500, {"error": f"{type(e).__name__}: {e}"})

        def _route_get(self) -> None:
            path = urlparse(self.path).path
            if path == "/v1/health":
                self._write_json(200, {"status": "ok"})
                return
            if path == "/v1/describe":
                self._write_json(200, runtime.describe())
                return
            if path == "/v1/sessions":
                self._write_json(
                    200,
                    {"sessions": [s.to_dict() for s in runtime.list_sessions()]},
                )
                return
            if path.startswith("/v1/sessions/"):
                sid = path.rsplit("/", 1)[-1]
                try:
                    self._write_json(200, runtime.session_info(sid).to_dict())
                except KeyError:
                    self._write_json(404, {"error": f"unknown session: {sid}"})
                return
            self._write_json(404, {"error": f"unknown path: {path}"})

        def _route_post(self) -> None:
            path = urlparse(self.path).path
            if path == "/v1/sessions":
                body = self._read_json()
                sid = runtime.create_session(body.get("session_id"))
                self._write_json(201, runtime.session_info(sid).to_dict())
                return
            if path.startswith("/v1/sessions/") and path.endswith("/turn"):
                sid = path[len("/v1/sessions/"):-len("/turn")]
                body = self._read_json()
                user_input = body.get("input")
                if not isinstance(user_input, str) or not user_input.strip():
                    raise ValueError("body.input is required and must be a non-empty string")
                max_iter = int(body.get("max_iterations", 25))
                result = runtime.turn(sid, user_input, max_iterations=max_iter)
                self._write_json(200, result.to_dict())
                return
            if path.startswith("/v1/sessions/") and path.endswith("/reset"):
                sid = path[len("/v1/sessions/"):-len("/reset")]
                runtime.reset_session(sid)
                self._write_json(200, runtime.session_info(sid).to_dict())
                return
            self._write_json(404, {"error": f"unknown path: {path}"})

        def _route_delete(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/v1/sessions/"):
                sid = path.rsplit("/", 1)[-1]
                runtime.delete_session(sid)
                self._write_json(200, {"deleted": sid})
                return
            self._write_json(404, {"error": f"unknown path: {path}"})

    return Handler


def serve(
    runtime: Runtime,
    *,
    host: str = "127.0.0.1",
    port: int = 8088,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    """Start the server. Returns the underlying server object so callers can
    `serve_forever()` themselves or shut it down. Useful for tests:

        srv = serve(Runtime(), host="127.0.0.1", port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        ...
        srv.shutdown()
    """
    handler_cls = _make_handler(runtime, auth_token)
    server = ThreadingHTTPServer((host, port), handler_cls)
    return server


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="agi serve")
    parser.add_argument("--host", default=os.environ.get("AGI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGI_PORT", "8088")))
    parser.add_argument(
        "--memory-root",
        default=os.environ.get("AGI_MEMORY_ROOT"),
        help="Directory under which per-session memory JSONLs are written.",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("AGI_AUTH_TOKEN"),
        help="If set, require Authorization: Bearer <token> on every request.",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: ANTHROPIC_API_KEY is not set — agent.chat() will fail when called.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    runtime = Runtime(memory_root=args.memory_root)
    server = serve(runtime, host=args.host, port=args.port, auth_token=args.auth_token)
    addr = server.server_address
    _log.info("agi runtime listening on http://%s:%d", addr[0], addr[1])
    stop_event = threading.Event()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("shutting down")
    finally:
        server.shutdown()
        stop_event.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
