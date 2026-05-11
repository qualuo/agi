"""HTTP / SSE server exposing the Runtime.

This is the external surface a coordination engine talks to when the agent
isn't running in-process. Endpoints:

  POST   /v1/sessions                 -> {session_id}
  GET    /v1/sessions                 -> list of SessionStatus
  GET    /v1/sessions/{sid}           -> SessionStatus
  POST   /v1/sessions/{sid}/messages  -> {ok: true}   body: {"input": "..."}
  GET    /v1/sessions/{sid}/events    -> Server-Sent Events stream
  DELETE /v1/sessions/{sid}           -> {ok: true}
  POST   /v1/tasks                    -> TaskResult (runs through Coordinator)
                                          body: see Task dataclass

  GET    /v1/healthz                  -> {ok: true}

The server uses only the Python stdlib (`http.server`, `socketserver`) to keep
the dependency surface tight. For production you'd put this behind a real
HTTP server, but the wire format is stable and the contract is what matters.

SSE lines look like:

  event: TextDelta
  data: {"session_id": "...", "seq": 7, "text": "hello"}
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from agi.coord import Coordinator, Task, TaskResult
from agi.runtime import Runtime


def _serialize_task_result(tr: TaskResult) -> dict[str, Any]:
    return {
        "task": {
            "prompt": tr.task.prompt,
            "role": tr.task.role,
            "budget_usd": tr.task.budget_usd,
            "metadata": tr.task.metadata,
        },
        "session_id": tr.session_id,
        "final_text": tr.final_text,
        "cost_usd": tr.cost_usd,
        "total_cost_usd": tr.total_cost_usd(),
        "elapsed_s": tr.elapsed_s,
        "error": tr.error,
        "children": [_serialize_task_result(c) for c in tr.children],
    }


def _parse_task(d: dict[str, Any]) -> Task:
    subtasks_raw = d.get("subtasks") or []
    subtasks = [_parse_task(s) for s in subtasks_raw if isinstance(s, dict)]
    return Task(
        prompt=str(d.get("prompt", "")),
        role=str(d.get("role", "executor")),
        budget_usd=d.get("budget_usd"),
        metadata=dict(d.get("metadata") or {}),
        subtasks=subtasks,
    )


class _Handler(BaseHTTPRequestHandler):
    runtime: Runtime  # set on the class by make_server
    coordinator: Coordinator  # set on the class by make_server

    # Quieten the default access log; production deploys can flip this back on.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    # ---------- response helpers ----------

    def _json(self, status: int, body: Any) -> None:
        payload = json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON body: {e}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # ---------- routing ----------

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path == "/v1/healthz":
            return self._json(HTTPStatus.OK, {"ok": True})

        if path == "/v1/sessions":
            return self._json(
                HTTPStatus.OK,
                {"sessions": [asdict(s) for s in self.runtime.list_sessions()]},
            )

        if path.startswith("/v1/sessions/"):
            parts = path.split("/")
            # /v1/sessions/{sid} or /v1/sessions/{sid}/events
            if len(parts) == 4:
                sid = parts[3]
                try:
                    return self._json(HTTPStatus.OK, asdict(self.runtime.status(sid)))
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, f"unknown session: {sid}")
            if len(parts) == 5 and parts[4] == "events":
                return self._stream_events(parts[3])

        return self._error(HTTPStatus.NOT_FOUND, f"no route for GET {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json_body()
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))

        if path == "/v1/sessions":
            sid = self.runtime.open(
                budget_usd=body.get("budget_usd"),
                system_prompt_extra=str(body.get("system_prompt_extra", "")),
                metadata=dict(body.get("metadata") or {}),
            )
            return self._json(HTTPStatus.CREATED, {"session_id": sid})

        if path.startswith("/v1/sessions/") and path.endswith("/messages"):
            sid = path.split("/")[3]
            user_input = body.get("input")
            if not isinstance(user_input, str) or not user_input.strip():
                return self._error(HTTPStatus.BAD_REQUEST, "missing 'input' field")
            try:
                self.runtime.send(sid, user_input)
            except KeyError:
                return self._error(HTTPStatus.NOT_FOUND, f"unknown session: {sid}")
            except RuntimeError as e:
                return self._error(HTTPStatus.CONFLICT, str(e))
            return self._json(HTTPStatus.ACCEPTED, {"ok": True})

        if path == "/v1/tasks":
            task = _parse_task(body)
            if not task.prompt.strip():
                return self._error(HTTPStatus.BAD_REQUEST, "missing 'prompt' field")
            result = self.coordinator.run_one(task)
            return self._json(HTTPStatus.OK, _serialize_task_result(result))

        return self._error(HTTPStatus.NOT_FOUND, f"no route for POST {path}")

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/v1/sessions/"):
            sid = path.split("/")[3]
            try:
                self.runtime.close(sid)
            except KeyError:
                return self._error(HTTPStatus.NOT_FOUND, f"unknown session: {sid}")
            return self._json(HTTPStatus.OK, {"ok": True})
        return self._error(HTTPStatus.NOT_FOUND, f"no route for DELETE {path}")

    # ---------- SSE ----------

    def _stream_events(self, sid: str) -> None:
        try:
            self.runtime.status(sid)
        except KeyError:
            return self._error(HTTPStatus.NOT_FOUND, f"unknown session: {sid}")

        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        # Disable proxy buffering when behind nginx/cloudflare.
        self.send_header("x-accel-buffering", "no")
        self.end_headers()

        try:
            for ev in self.runtime.stream(sid, timeout=30.0):
                line = (
                    f"event: {ev.type}\n"
                    f"data: {json.dumps(ev.to_dict(), default=str)}\n\n"
                )
                try:
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected. Drop the stream; the session keeps
                    # running and can be re-attached to (events queued after
                    # us will still be there for the next reader).
                    return
        except KeyError:
            return  # session was closed mid-stream


def make_server(
    runtime: Runtime,
    coordinator: Coordinator,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server bound to a given runtime.

    The runtime and coordinator are bound onto a fresh handler subclass so
    multiple servers can run in the same process with different configs.
    """
    handler_cls = type(
        "BoundHandler",
        (_Handler,),
        {"runtime": runtime, "coordinator": coordinator},
    )
    return ThreadingHTTPServer((host, port), handler_cls)  # type: ignore[arg-type]


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    runtime: Runtime | None = None,
    coordinator: Coordinator | None = None,
) -> None:
    """Convenience: build a runtime + coordinator and serve forever."""
    rt = runtime or Runtime()
    co = coordinator or Coordinator(rt)
    srv = make_server(rt, co, host=host, port=port)
    print(f"agi runtime listening on http://{host}:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    serve(host=args.host, port=args.port)
