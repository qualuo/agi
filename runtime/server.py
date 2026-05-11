"""HTTP server: a wire protocol for the runtime engine.

A coordination engine that doesn't share the Python process can talk to
the runtime over plain HTTP. The protocol is small enough to inspect with
curl and stable enough to script against.

Endpoints
---------
GET  /health                        Liveness probe.
GET  /metrics                       Counts by status, totals.
POST /tasks                         Submit a task. Body: {instruction, budget?, metadata?, parent_id?}.
                                    Returns: task snapshot. 202.
GET  /tasks                         List tasks. Query: ?status=running etc.
GET  /tasks/{id}                    Get task snapshot.
GET  /tasks/{id}/tree               Task subtree (children populated).
GET  /tasks/{id}/events             All events so far (JSON array).
GET  /tasks/{id}/stream             Server-Sent Events stream of live events.
POST /tasks/{id}/cancel             Request cancellation. 202 if accepted.
POST /tasks/{id}/wait?timeout=N     Block (server-side) up to N seconds, return snapshot.
GET  /skills                        List skills in the library.
GET  /skills/{name}                 Get a skill.
POST /skills                        Add a skill. Body: {name, when, body, tags?}.
DELETE /skills/{name}               Remove a skill.

This is a stdlib-only implementation — no FastAPI dependency added. It is
adequate for coordination traffic (low QPS, structured). Swap in something
fancier when traffic justifies it.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

from runtime.engine import Engine, event_to_jsonable
from runtime.task import Budget, TaskStatus

logger = logging.getLogger("runtime.server")


def make_app(
    engine: Engine,
    *,
    skill_library=None,
):
    """Build the HTTP handler class bound to a particular engine.

    Returns a `BaseHTTPRequestHandler` subclass suitable for passing to
    `ThreadingHTTPServer`.
    """

    routes: list[tuple[str, re.Pattern, str]] = [
        ("GET", re.compile(r"^/health$"), "health"),
        ("GET", re.compile(r"^/metrics$"), "metrics"),
        ("POST", re.compile(r"^/tasks$"), "submit_task"),
        ("GET", re.compile(r"^/tasks$"), "list_tasks"),
        ("GET", re.compile(r"^/tasks/(?P<id>[a-zA-Z0-9_-]+)$"), "get_task"),
        ("GET", re.compile(r"^/tasks/(?P<id>[a-zA-Z0-9_-]+)/tree$"), "get_tree"),
        ("GET", re.compile(r"^/tasks/(?P<id>[a-zA-Z0-9_-]+)/events$"), "get_events"),
        ("GET", re.compile(r"^/tasks/(?P<id>[a-zA-Z0-9_-]+)/stream$"), "stream_events"),
        ("POST", re.compile(r"^/tasks/(?P<id>[a-zA-Z0-9_-]+)/cancel$"), "cancel_task"),
        ("POST", re.compile(r"^/tasks/(?P<id>[a-zA-Z0-9_-]+)/wait$"), "wait_task"),
        ("GET", re.compile(r"^/skills$"), "list_skills"),
        ("GET", re.compile(r"^/skills/(?P<name>[a-zA-Z0-9_-]+)$"), "get_skill"),
        ("POST", re.compile(r"^/skills$"), "add_skill"),
        ("DELETE", re.compile(r"^/skills/(?P<name>[a-zA-Z0-9_-]+)$"), "remove_skill"),
    ]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # Silence the noisy default access log; route through `logger` instead.
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - - %s", self.address_string(), fmt % args)

        # --- dispatch -----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            url = urlsplit(self.path)
            for m, pattern, handler_name in routes:
                if m != method:
                    continue
                match = pattern.match(url.path)
                if match is None:
                    continue
                params = match.groupdict()
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                try:
                    getattr(self, handler_name)(params=params, query=query)
                except _ClientError as e:
                    self._send_json(e.status, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    logger.exception("handler %s failed", handler_name)
                    self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self._send_json(404, {"error": "not found", "path": self.path})

        # --- helpers ------------------------------------------------------

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                d = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise _ClientError(400, f"invalid JSON: {e}")
            if not isinstance(d, dict):
                raise _ClientError(400, "request body must be a JSON object")
            return d

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        # --- route handlers ----------------------------------------------

        def health(self, *, params, query) -> None:
            self._send_json(200, {"ok": True, "ts": time.time()})

        def metrics(self, *, params, query) -> None:
            records = engine.list_tasks()
            counts: dict[str, int] = {}
            total_cost = 0.0
            total_input = total_output = 0
            for r in records:
                counts[r.status.value] = counts.get(r.status.value, 0) + 1
                total_cost += r.cost_usd
                total_input += r.input_tokens
                total_output += r.output_tokens
            self._send_json(200, {
                "task_counts": counts,
                "total_cost_usd": round(total_cost, 6),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tasks": len(records),
            })

        def submit_task(self, *, params, query) -> None:
            body = self._read_json()
            instruction = body.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise _ClientError(400, "instruction is required")
            budget_kwargs = body.get("budget") or {}
            try:
                budget = Budget(**budget_kwargs) if budget_kwargs else Budget()
            except TypeError as e:
                raise _ClientError(400, f"bad budget: {e}")
            task = engine.submit(
                instruction=instruction,
                budget=budget,
                parent_id=body.get("parent_id"),
                metadata=body.get("metadata") or {},
            )
            self._send_json(202, task.snapshot().as_dict())

        def list_tasks(self, *, params, query) -> None:
            status_filter = None
            if "status" in query:
                try:
                    status_filter = TaskStatus(query["status"])
                except ValueError:
                    raise _ClientError(400, f"bad status: {query['status']}")
            records = engine.list_tasks(status=status_filter)
            self._send_json(200, {"tasks": [r.as_dict() for r in records]})

        def get_task(self, *, params, query) -> None:
            task = engine.get(params["id"])
            if task is None:
                raise _ClientError(404, f"no such task: {params['id']}")
            self._send_json(200, task.snapshot().as_dict())

        def get_tree(self, *, params, query) -> None:
            tree = engine.task_tree(params["id"])
            if not tree:
                raise _ClientError(404, f"no such task: {params['id']}")
            self._send_json(200, tree)

        def get_events(self, *, params, query) -> None:
            task = engine.get(params["id"])
            if task is None:
                raise _ClientError(404, f"no such task: {params['id']}")
            self._send_json(200, {"events": [event_to_jsonable(e) for e in task.events()]})

        def stream_events(self, *, params, query) -> None:
            task = engine.get(params["id"])
            if task is None:
                raise _ClientError(404, f"no such task: {params['id']}")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for ev in task.stream_events(timeout=1.0):
                    chunk = f"data: {json.dumps(event_to_jsonable(ev), default=str)}\n\n"
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def cancel_task(self, *, params, query) -> None:
            ok = engine.cancel(params["id"])
            if not ok:
                raise _ClientError(404, f"task not cancellable: {params['id']}")
            self._send_json(202, {"cancel_requested": True, "id": params["id"]})

        def wait_task(self, *, params, query) -> None:
            task = engine.get(params["id"])
            if task is None:
                raise _ClientError(404, f"no such task: {params['id']}")
            try:
                timeout = float(query.get("timeout", "30"))
            except ValueError:
                raise _ClientError(400, "timeout must be a number")
            task.wait(timeout=timeout)
            self._send_json(200, task.snapshot().as_dict())

        # --- skills --------------------------------------------------------

        def list_skills(self, *, params, query) -> None:
            if skill_library is None:
                raise _ClientError(404, "no skill library configured")
            skills = skill_library.all()
            self._send_json(200, {"skills": [
                {"name": s.name, "when": s.when, "tags": s.tags, "body": s.body}
                for s in skills
            ]})

        def get_skill(self, *, params, query) -> None:
            if skill_library is None:
                raise _ClientError(404, "no skill library configured")
            s = skill_library.get(params["name"])
            if s is None:
                raise _ClientError(404, f"no such skill: {params['name']}")
            self._send_json(200, {"name": s.name, "when": s.when, "tags": s.tags, "body": s.body})

        def add_skill(self, *, params, query) -> None:
            if skill_library is None:
                raise _ClientError(404, "no skill library configured")
            body = self._read_json()
            for required in ("name", "when", "body"):
                if not isinstance(body.get(required), str):
                    raise _ClientError(400, f"{required} is required")
            s = skill_library.add_from_text(
                name=body["name"],
                when=body["when"],
                body=body["body"],
                tags=body.get("tags") or [],
            )
            self._send_json(201, {"name": s.name, "when": s.when, "tags": s.tags, "body": s.body})

        def remove_skill(self, *, params, query) -> None:
            if skill_library is None:
                raise _ClientError(404, "no skill library configured")
            ok = skill_library.remove(params["name"])
            if not ok:
                raise _ClientError(404, f"no such skill: {params['name']}")
            self._send_json(200, {"removed": params["name"]})

    return Handler


class _ClientError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Convenience: run a server
# ---------------------------------------------------------------------------


def serve(
    engine: Engine,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    skill_library=None,
) -> ThreadingHTTPServer:
    """Build a ThreadingHTTPServer bound to `engine`. Returns it; caller
    decides whether to `.serve_forever()` or run in a background thread
    (the test suite does the latter)."""
    handler = make_app(engine, skill_library=skill_library)
    return ThreadingHTTPServer((host, port), handler)


def serve_in_background(
    engine: Engine,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    skill_library=None,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Run a server on a daemon thread. Returns (server, thread, base_url).

    Pass `port=0` to let the OS choose a free port — useful for tests."""
    server = serve(engine, host=host, port=port, skill_library=skill_library)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="runtime.server", daemon=True)
    thread.start()
    base_url = f"http://{host}:{actual_port}"
    return server, thread, base_url
