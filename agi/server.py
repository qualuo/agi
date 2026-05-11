"""HTTP server exposing the runtime to a coordination engine.

Stdlib-only (http.server + json + threading) so this installs anywhere
the rest of the package does, with no new dependencies. A coordinator
speaks HTTP/JSON to drive tasks; for streaming progress it consumes
Server-Sent Events on a per-task endpoint.

Endpoints:
  GET  /v1/health               liveness
  GET  /v1/capabilities         what this runtime can do (Capabilities JSON)
  POST /v1/tasks                submit a Task; returns {task_id}
  GET  /v1/tasks/{id}           current TaskResult (any state)
  GET  /v1/tasks/{id}/events    SSE stream of Events until terminal
  POST /v1/tasks/{id}/cancel    request cooperative cancellation

The server is intentionally minimal — no auth, no rate limiting. Put it
behind a real ingress (nginx, an API gateway, mTLS) for production. The
goal here is "addressable runtime", not "internet-exposed service".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from agi.runtime import Budget, RuntimeEngine, Task


def _agent_factory_default():
    """Default agent factory. Built lazily so importing the server module
    doesn't require the Anthropic SDK or an API key on disk."""
    from agi.agent import Agent
    return Agent(verbose=False)


class RuntimeHTTPHandler(BaseHTTPRequestHandler):
    engine: RuntimeEngine  # populated by make_handler

    def log_message(self, fmt: str, *args) -> None:  # quiet by default
        if os.environ.get("AGI_HTTP_LOG"):
            super().log_message(fmt, *args)

    # ---- routing --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib signature)
        path = self.path.split("?", 1)[0]
        if path == "/v1/health":
            return self._json(200, {"ok": True})
        if path == "/v1/capabilities":
            return self._json(200, self.engine.capabilities.to_dict())
        if path.startswith("/v1/tasks/") and path.endswith("/events"):
            task_id = path[len("/v1/tasks/"):-len("/events")]
            return self._sse_events(task_id)
        if path.startswith("/v1/tasks/"):
            task_id = path[len("/v1/tasks/"):]
            return self._task_result(task_id)
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/v1/tasks":
            return self._submit_task()
        if path.startswith("/v1/tasks/") and path.endswith("/cancel"):
            task_id = path[len("/v1/tasks/"):-len("/cancel")]
            try:
                self.engine.cancel(task_id)
                return self._json(202, {"task_id": task_id, "cancel_requested": True})
            except KeyError:
                return self._json(404, {"error": f"unknown task {task_id}"})
        self._json(404, {"error": "not found"})

    # ---- handlers -------------------------------------------------------

    def _submit_task(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        instruction = body.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            return self._json(400, {"error": "field 'instruction' is required"})

        budget_args = body.get("budget") or {}
        try:
            budget = Budget(**budget_args)
        except TypeError as e:
            return self._json(400, {"error": f"invalid budget: {e}"})

        task = Task(
            instruction=instruction,
            inputs=body.get("inputs") or {},
            skills=body.get("skills") or [],
            metadata=body.get("metadata") or {},
            budget=budget,
        )
        if "id" in body and isinstance(body["id"], str) and body["id"]:
            task.id = body["id"]

        self.engine.submit(task)
        self._json(202, {"task_id": task.id, "status": "running"})

    def _task_result(self, task_id: str) -> None:
        try:
            result = self.engine.get_result(task_id)
        except KeyError:
            return self._json(404, {"error": f"unknown task {task_id}"})
        self._json(200, result.to_dict())

    def _sse_events(self, task_id: str) -> None:
        try:
            self.engine.get_result(task_id)
        except KeyError:
            return self._json(404, {"error": f"unknown task {task_id}"})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for ev in self.engine.stream_events(task_id):
                payload = json.dumps(asdict(ev), default=str)
                self.wfile.write(f"event: {ev.type}\n".encode())
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    # ---- helpers --------------------------------------------------------

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return None
        try:
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError) as e:
            self._json(400, {"error": f"invalid JSON: {e}"})
            return None

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_handler(engine: RuntimeEngine) -> type[RuntimeHTTPHandler]:
    """Bind a RuntimeEngine into a handler class. ThreadingHTTPServer
    instantiates the handler per-connection, so we attach the engine as
    a class attribute on a fresh subclass."""
    return type("BoundRuntimeHandler", (RuntimeHTTPHandler,), {"engine": engine})


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    agent_factory: Callable[[], object] | None = None,
    engine: RuntimeEngine | None = None,
) -> ThreadingHTTPServer:
    """Start the runtime HTTP server. Blocks on serve_forever().

    Returns the server so callers can stop it from another thread
    (used in tests). For production-style use prefer running this from
    `python -m agi.server` and stopping via signal.
    """
    if engine is None:
        engine = RuntimeEngine(agent_factory or _agent_factory_default)
    handler_cls = make_handler(engine)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    httpd.engine = engine  # type: ignore[attr-defined]
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(description="agi-runtime HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: ANTHROPIC_API_KEY not set — task execution will fail", file=sys.stderr)

    httpd = serve(host=args.host, port=args.port)
    print(f"agi-runtime listening on http://{args.host}:{args.port}")
    print("  GET  /v1/health")
    print("  GET  /v1/capabilities")
    print("  POST /v1/tasks")
    print("  GET  /v1/tasks/{id}")
    print("  GET  /v1/tasks/{id}/events  (SSE)")
    print("  POST /v1/tasks/{id}/cancel")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
