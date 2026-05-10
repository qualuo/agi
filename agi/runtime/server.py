"""HTTP server exposing the runtime API.

stdlib only — `http.server.ThreadingHTTPServer`. The protocol:

  GET  /capabilities                 -> JSON capability descriptor
  POST /tasks                        -> {task_id} (body: TaskSpec JSON)
  GET  /tasks                        -> list of tasks
  GET  /tasks/{id}                   -> task JSON
  GET  /tasks/{id}/stream            -> text/event-stream of task events
  POST /tasks/{id}/cancel            -> 202
  POST /graphs                       -> {graph_id} (body: GraphSpec JSON)
  GET  /graphs/{id}/stream           -> text/event-stream of graph events
  GET  /graphs/{id}                  -> latest GraphResult or in-progress status
  GET  /events                       -> text/event-stream of all runtime events
  GET  /skills                       -> list of skills
  GET  /skills/{name}                -> skill content
  GET  /healthz                      -> {ok: true}

The server is meant as the boundary a coordination engine talks to. It's
intentionally small; production deployments would wrap it with auth/TLS.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agi.runtime.capabilities import RUNTIME_CAPABILITIES
from agi.runtime.events import EventBus
from agi.runtime.graph import GraphExecutor, GraphSpec
from agi.runtime.tasks import TaskSpec, TaskStore
from agi.runtime.worker import Worker, make_default_registry


class Runtime:
    """Composition root: store + bus + workers + graph executor."""

    def __init__(self, *, num_workers: int = 2, agent_factory=None) -> None:
        self.store = TaskStore()
        self.bus = EventBus()
        self.registry = make_default_registry(agent_factory=agent_factory)
        self.workers = [
            Worker(self.store, self.bus, self.registry, name=f"worker-{i}")
            for i in range(num_workers)
        ]
        self.graph = GraphExecutor(self.store, self.bus, self.workers)
        for w in self.workers:
            w.start()
        self.bus.publish("runtime", "runtime.startup",
                         {"workers": num_workers,
                          "capabilities": RUNTIME_CAPABILITIES["name"]})

    def submit_task(self, spec: TaskSpec):
        task = self.store.submit(spec)
        # Round-robin over workers.
        worker = self.workers[hash(task.id) % len(self.workers)]
        worker.enqueue(task.id)
        self.bus.publish(f"task.{task.id}", "task.queued", {"task": task.to_dict()})
        return task

    def shutdown(self) -> None:
        for w in self.workers:
            w.stop(drain=True)


class _RuntimeHTTPHandler(BaseHTTPRequestHandler):
    runtime: Runtime  # patched in by serve()

    # Quiet the default request log noise; emit one line per request only.
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse_header(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {e}"})
            raise

    # --- routing ---
    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        path = url.path.rstrip("/")
        try:
            if path == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True, "ts": time.time()})
                return
            if path == "/capabilities":
                self._send_json(HTTPStatus.OK, RUNTIME_CAPABILITIES)
                return
            if path == "/tasks":
                self._send_json(HTTPStatus.OK,
                                {"tasks": [t.to_dict() for t in self.runtime.store.all()]})
                return
            if path.startswith("/tasks/") and path.endswith("/stream"):
                tid = path[len("/tasks/"):-len("/stream")]
                self._stream(f"task.{tid}")
                return
            if path.startswith("/tasks/"):
                tid = path[len("/tasks/"):]
                t = self.runtime.store.get(tid)
                if t is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "task not found"})
                    return
                self._send_json(HTTPStatus.OK, t.to_dict())
                return
            if path.startswith("/graphs/") and path.endswith("/stream"):
                gid = path[len("/graphs/"):-len("/stream")]
                self._stream(f"graph.{gid}")
                return
            if path.startswith("/graphs/"):
                gid = path[len("/graphs/"):]
                # Return latest event for the graph or 404.
                hist = self.runtime.bus.history(f"graph.{gid}")
                if not hist:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "graph not found"})
                    return
                latest = hist[-1]
                self._send_json(HTTPStatus.OK,
                                {"graph_id": gid, "latest": {"kind": latest.kind,
                                                              "payload": latest.payload}})
                return
            if path == "/events":
                self._stream("")
                return
            if path == "/skills":
                from agi.skills.library import SkillLibrary
                self._send_json(HTTPStatus.OK,
                                {"skills": SkillLibrary().describe()})
                return
            if path.startswith("/skills/"):
                from agi.skills.library import SkillLibrary
                name = path[len("/skills/"):]
                s = SkillLibrary().get(name)
                if s is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "skill not found"})
                    return
                self._send_json(HTTPStatus.OK, s.to_dict())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            try:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            except Exception:
                pass

    def do_POST(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        path = url.path.rstrip("/")
        try:
            if path == "/tasks":
                body = self._read_json()
                spec = TaskSpec(
                    kind=body["kind"],
                    input=body.get("input", {}),
                    dedup_key=body.get("dedup_key"),
                    role=body.get("role"),
                    parent_id=body.get("parent_id"),
                    budget_tokens=body.get("budget_tokens"),
                    budget_seconds=body.get("budget_seconds"),
                    tags=body.get("tags", []),
                )
                task = self.runtime.submit_task(spec)
                self._send_json(HTTPStatus.ACCEPTED, {"task_id": task.id, "status": task.status.value})
                return
            if path.startswith("/tasks/") and path.endswith("/cancel"):
                tid = path[len("/tasks/"):-len("/cancel")]
                self.runtime.store.mark_cancelled(tid)
                self._send_json(HTTPStatus.ACCEPTED, {"task_id": tid, "cancel_requested": True})
                return
            if path == "/graphs":
                body = self._read_json()
                graph = GraphSpec.from_dict(body)
                gid = self.runtime.graph.submit(graph)
                self._send_json(HTTPStatus.ACCEPTED, {"graph_id": gid})
                return
            if path.startswith("/graphs/") and path.endswith("/cancel"):
                gid = path[len("/graphs/"):-len("/cancel")]
                self.runtime.graph.cancel(gid)
                self._send_json(HTTPStatus.ACCEPTED, {"graph_id": gid, "cancel_requested": True})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})
        except KeyError as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"missing field: {e}"})
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            try:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            except Exception:
                pass

    def _stream(self, prefix: str) -> None:
        self._send_sse_header()
        # Send any retained history first.
        try:
            for e in self.runtime.bus.history(prefix):
                self._sse_emit(e)
            sub = self.runtime.bus.subscribe(prefix=prefix)
            try:
                while True:
                    assert sub.queue is not None
                    try:
                        event = sub.queue.get(timeout=15.0)
                    except Exception:
                        # Heartbeat to keep connection open.
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    self._sse_emit(event)
                    # Terminal graph events close the stream so clients can exit.
                    if event.kind in ("graph.completed", "graph.failed", "graph.cancelled"):
                        break
            finally:
                self.runtime.bus.unsubscribe(sub)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

    def _sse_emit(self, event) -> None:
        line = f"event: {event.kind}\ndata: {event.to_json()}\n\n"
        try:
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()
        except BrokenPipeError:
            raise
        except ConnectionResetError:
            raise


def serve(runtime: Runtime, host: str = "127.0.0.1", port: int = 7777) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the HTTP server in a background thread. Returns (server, thread)."""
    handler = type("RuntimeHandler", (_RuntimeHTTPHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="agi-runtime-http", daemon=True)
    thread.start()
    return server, thread


def main() -> int:
    import os
    import signal
    import sys

    host = os.environ.get("AGI_RUNTIME_HOST", "127.0.0.1")
    port = int(os.environ.get("AGI_RUNTIME_PORT", "7777"))
    workers = int(os.environ.get("AGI_RUNTIME_WORKERS", "2"))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: ANTHROPIC_API_KEY not set; chat/plan/critique tasks will fail.",
              file=sys.stderr)

    runtime = Runtime(num_workers=workers)
    server, _ = serve(runtime, host=host, port=port)
    print(f"agi-runtime listening on http://{host}:{port}  ({workers} workers)")
    print(f"  GET  /capabilities      describe what this runtime can do")
    print(f"  POST /tasks             submit a task (kind: chat|plan|critique|skill.invoke|tool)")
    print(f"  POST /graphs            submit a typed task DAG")
    print(f"  GET  /events            SSE stream of all runtime events")

    stop = threading.Event()
    def _handle_signal(*_):
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass
    try:
        while not stop.is_set():
            stop.wait(timeout=1.0)
    finally:
        print("shutting down")
        server.shutdown()
        runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
