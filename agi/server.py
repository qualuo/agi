"""HTTP / JSON-RPC server exposing the Runtime.

A coordination engine running in another process (or another machine) talks
to the runtime over this surface. Plain stdlib HTTP — no Flask, no FastAPI,
no extra dependencies.

Endpoints:

    GET  /healthz               liveness
    GET  /manifest              capability manifest (discovery)
    POST /tasks                 submit a task; returns {id}
    GET  /tasks                 list tasks (?status=running etc.)
    GET  /tasks/{id}            task status snapshot
    POST /tasks/{id}/cancel     cancel a running task
    GET  /tasks/{id}/wait       block until task finishes (server-side wait)
    GET  /events                Server-Sent Events stream of runtime events

The /tasks POST body is JSON:
    {
      "prompt": "...",
      "role": "executor",
      "model": "claude-opus-4-7",
      "budget": {"max_usd": 0.1, "max_tokens": 200000, "max_wall_seconds": 60},
      "system_prompt": "optional override",
      "max_iterations": 25
    }

Authentication: none built-in. Front this with an auth proxy (Cloudflare
Access, nginx + JWT, etc.) before exposing to anything that isn't trusted.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agi.budget import Budget
from agi.events import Event
from agi.runtime import Runtime, TaskStatus


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: Any) -> None:
    payload = json.dumps(body, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)


def _parse_budget(d: dict | None) -> Budget | None:
    if not d:
        return None
    return Budget(
        max_usd=d.get("max_usd"),
        max_tokens=d.get("max_tokens"),
        max_wall_seconds=d.get("max_wall_seconds"),
    )


def make_handler(runtime: Runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "agi-runtime/0.2"

        # Quieter logs — default BaseHTTPRequestHandler writes to stderr.
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            pass

        # --------------------------------------------------------- routing
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path, qs = parsed.path, urllib.parse.parse_qs(parsed.query)
            if path == "/healthz":
                _json_response(self, 200, {"ok": True, "ts": time.time()})
                return
            if path == "/manifest":
                _json_response(self, 200, runtime.manifest().to_dict())
                return
            if path == "/tasks":
                status_filter = None
                if "status" in qs and qs["status"]:
                    try:
                        status_filter = TaskStatus(qs["status"][0])
                    except ValueError:
                        _json_response(self, 400, {"error": f"unknown status: {qs['status'][0]}"})
                        return
                _json_response(self, 200, {"tasks": runtime.list_tasks(status=status_filter)})
                return
            if path.startswith("/tasks/") and path.endswith("/wait"):
                task_id = path[len("/tasks/"):-len("/wait")]
                timeout = float(qs.get("timeout", ["60"])[0])
                try:
                    snap = runtime.wait(task_id, timeout=timeout)
                except KeyError:
                    _json_response(self, 404, {"error": "no such task"})
                    return
                _json_response(self, 200, snap)
                return
            if path.startswith("/tasks/"):
                task_id = path[len("/tasks/"):]
                try:
                    _json_response(self, 200, runtime.status(task_id))
                except KeyError:
                    _json_response(self, 404, {"error": "no such task"})
                return
            if path == "/events":
                self._stream_events(qs)
                return
            _json_response(self, 404, {"error": f"unknown path: {path}"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as e:
                _json_response(self, 400, {"error": f"bad json: {e}"})
                return

            if path == "/tasks":
                if not isinstance(body, dict) or "prompt" not in body:
                    _json_response(self, 400, {"error": "missing 'prompt'"})
                    return
                handle = runtime.submit(
                    body["prompt"],
                    role=body.get("role", "executor"),
                    model=body.get("model"),
                    budget=_parse_budget(body.get("budget")),
                    system_prompt=body.get("system_prompt"),
                    max_iterations=int(body.get("max_iterations", 25)),
                )
                _json_response(self, 202, {"id": handle.id, "status": "pending"})
                return

            if path.startswith("/tasks/") and path.endswith("/cancel"):
                task_id = path[len("/tasks/"):-len("/cancel")]
                ok = runtime.cancel(task_id)
                _json_response(self, 200 if ok else 409, {"cancelled": ok, "id": task_id})
                return

            _json_response(self, 404, {"error": f"unknown path: {path}"})

        # ------------------------------------------------------- SSE stream
        def _stream_events(self, qs: dict[str, list[str]]) -> None:
            """Server-Sent Events stream of runtime events.

            Optional ?task_id=... filters to events for that task and its
            descendants.
            """
            task_filter = qs.get("task_id", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            buf: queue.Queue[Event] = queue.Queue(maxsize=1000)

            def on_event(ev: Event) -> None:
                if task_filter and ev.task_id != task_filter and ev.parent_task_id != task_filter:
                    return
                try:
                    buf.put_nowait(ev)
                except queue.Full:
                    pass

            unsub = runtime.subscribe(on_event)
            try:
                last_heartbeat = time.time()
                while True:
                    try:
                        ev = buf.get(timeout=5.0)
                        payload = json.dumps(ev.to_dict(), default=str)
                        chunk = f"event: {ev.kind}\ndata: {payload}\n\n"
                        try:
                            self.wfile.write(chunk.encode("utf-8"))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    except queue.Empty:
                        # Heartbeat so proxies don't kill the connection.
                        if time.time() - last_heartbeat > 15:
                            try:
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                                last_heartbeat = time.time()
                            except (BrokenPipeError, ConnectionResetError):
                                return
            finally:
                unsub()

    return Handler


def serve(runtime: Runtime, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Start the HTTP server. Returns the server instance; caller is
    responsible for serve_forever() or shutdown()."""
    handler_cls = make_handler(runtime)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    return httpd


def serve_blocking(runtime: Runtime, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = serve(runtime, host=host, port=port)
    print(f"agi-runtime serving on http://{host}:{port}")
    print(f"  GET  /healthz            liveness")
    print(f"  GET  /manifest           capability manifest")
    print(f"  POST /tasks              submit a task")
    print(f"  GET  /tasks/{{id}}         task status")
    print(f"  GET  /tasks/{{id}}/wait    block until done")
    print(f"  POST /tasks/{{id}}/cancel  cancel a task")
    print(f"  GET  /events             SSE stream of runtime events")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        runtime.shutdown()


def _serve_in_thread(runtime: Runtime, *, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Test helper: start the server in a background thread."""
    httpd = serve(runtime, host=host, port=port)
    th = threading.Thread(target=httpd.serve_forever, daemon=True, name="agi-server")
    th.start()
    return httpd, th
