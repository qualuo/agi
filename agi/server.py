"""HTTP + SSE server exposing the Runtime to a coordination engine.

Endpoints (JSON in/out unless noted):

  GET  /healthz                  → {"status": "ok"}
  GET  /capabilities             → Runtime.capabilities()
  GET  /sessions                 → list of session dicts
  POST /sessions                 → create. Body: SessionConfig fields. → {"id": ...}
  GET  /sessions/{id}            → session dict
  POST /sessions/{id}/chat       → {"prompt": ...} → {"final_text": ..., "session": ...}
  POST /sessions/{id}/cancel     → 204
  POST /sessions/{id}/reset      → 204
  DELETE /sessions/{id}          → 204
  GET  /events                   → Server-Sent Events stream of all runtime events
  GET  /events?session_id=…      → filtered stream
  GET  /events/history           → past events as JSON
  POST /skills                   → save a skill. Body: {name, description, body, tags}
  GET  /skills                   → list skills
  POST /tools                    → synthesize a tool

The transport is intentionally minimal — Python stdlib only — so the runtime
has no required runtime deps beyond `anthropic`. Production deployments
should put this behind a reverse proxy with auth and TLS.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agi.events import Event
from agi.runtime import Runtime, SessionConfig
from agi.skills import Skill
from agi.tasks import TaskQueue, TaskRunner, submit_task


def _config_from_body(body: dict[str, Any]) -> SessionConfig:
    valid = {f for f in SessionConfig.__dataclass_fields__}
    return SessionConfig(**{k: v for k, v in body.items() if k in valid})


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_empty(handler: BaseHTTPRequestHandler, status: int = 204) -> None:
    handler.send_response(status)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


def make_handler(
    runtime: Runtime,
    *,
    auth_token: str | None = None,
    task_queue: TaskQueue | None = None,
):
    """Build a BaseHTTPRequestHandler class bound to a Runtime instance."""
    tq = task_queue
    runner = TaskRunner(runtime, tq) if tq is not None else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # quiet by default
            return

        def _check_auth(self) -> bool:
            if auth_token is None:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {auth_token}"

        # --- routing ----------------------------------------------------

        def do_GET(self):  # noqa: N802
            if not self._check_auth():
                return _send_json(self, 401, {"error": "unauthorized"})
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/healthz":
                return _send_json(self, 200, {"status": "ok"})
            if path == "/capabilities":
                return _send_json(self, 200, runtime.capabilities())
            if path == "/metrics":
                return _send_json(self, 200, runtime.metrics())
            if path == "/sessions":
                return _send_json(self, 200, runtime.sessions())
            if path == "/tasks":
                if tq is None:
                    return _send_json(self, 404, {"error": "task queue not configured"})
                status_filter = query.get("status", [None])[0]
                tag_filter = query.get("tag", [None])[0]
                return _send_json(
                    self,
                    200,
                    [t.to_dict() for t in tq.list(status=status_filter, tag=tag_filter)],
                )
            if path.startswith("/tasks/"):
                if tq is None:
                    return _send_json(self, 404, {"error": "task queue not configured"})
                tid = path[len("/tasks/"):]
                try:
                    return _send_json(self, 200, tq.get(tid).to_dict())
                except KeyError:
                    return _send_json(self, 404, {"error": f"unknown task: {tid}"})
            if path.startswith("/sessions/"):
                sid = path[len("/sessions/"):]
                try:
                    return _send_json(self, 200, runtime.get_session(sid).to_dict())
                except KeyError:
                    return _send_json(self, 404, {"error": f"unknown session: {sid}"})
            if path == "/skills":
                return _send_json(
                    self,
                    200,
                    [
                        {"name": s.name, "description": s.description, "tags": s.tags, "body": s.body}
                        for s in runtime.skills.all()
                    ],
                )
            if path == "/events/history":
                kwargs: dict[str, Any] = {}
                if "session_id" in query:
                    kwargs["session_id"] = query["session_id"][0]
                if "kind" in query:
                    kwargs["kind"] = query["kind"][0]
                if "limit" in query:
                    kwargs["limit"] = int(query["limit"][0])
                return _send_json(
                    self,
                    200,
                    [e.to_dict() for e in runtime.events(**kwargs)],
                )
            if path == "/events":
                return self._stream_events(query)
            return _send_json(self, 404, {"error": f"not found: {path}"})

        def do_POST(self):  # noqa: N802
            if not self._check_auth():
                return _send_json(self, 401, {"error": "unauthorized"})
            parsed = urlparse(self.path)
            path = parsed.path
            body = _read_body(self)

            if path == "/sessions":
                cfg = _config_from_body(body)
                namespace = body.get("namespace")
                sid = runtime.create_session(cfg, namespace=namespace)
                return _send_json(self, 201, {"id": sid})
            if path == "/sessions/restore":
                if runtime.session_store is None:
                    return _send_json(self, 400, {"error": "no session store configured"})
                sid = body.get("session_id")
                if not sid:
                    return _send_json(self, 400, {"error": "missing 'session_id'"})
                try:
                    sid = runtime.restore_session(sid)
                except (KeyError, ValueError) as e:
                    return _send_json(self, 404, {"error": str(e)})
                return _send_json(self, 200, {"id": sid})
            if path == "/tasks":
                if tq is None:
                    return _send_json(self, 404, {"error": "task queue not configured"})
                try:
                    cfg = _config_from_body(body.get("session_config") or {})
                    tid = submit_task(
                        tq,
                        prompt=body["prompt"],
                        session_config=cfg,
                        priority=int(body.get("priority", 0)),
                        deadline_ts=body.get("deadline_ts"),
                        max_attempts=int(body.get("max_attempts", 1)),
                        namespace=body.get("namespace"),
                        tag=body.get("tag"),
                    )
                except (KeyError, ValueError) as e:
                    return _send_json(self, 400, {"error": str(e)})
                return _send_json(self, 201, {"id": tid})
            if path == "/tasks/drain":
                if runner is None:
                    return _send_json(self, 404, {"error": "task queue not configured"})
                max_ticks = int(body.get("max_ticks", 100))
                executed = runner.run_until_empty(max_ticks=max_ticks)
                return _send_json(self, 200, {"executed": executed})
            if path == "/skills":
                try:
                    skill = Skill(
                        name=body["name"],
                        description=body.get("description", ""),
                        body=body.get("body", ""),
                        tags=list(body.get("tags") or []),
                    )
                    runtime.save_skill(skill)
                except (KeyError, ValueError) as e:
                    return _send_json(self, 400, {"error": str(e)})
                return _send_json(self, 201, {"name": skill.name})
            if path == "/tools":
                try:
                    tool = runtime.synthesize_tool(
                        name=body["name"],
                        description=body.get("description", ""),
                        code=body["code"],
                        input_schema=body.get("input_schema"),
                        smoke_test_kwargs=body.get("smoke_test_kwargs"),
                    )
                except Exception as e:
                    return _send_json(self, 400, {"error": str(e)})
                return _send_json(self, 201, {"name": tool.name})
            if path.startswith("/sessions/"):
                rest = path[len("/sessions/"):]
                if "/" in rest:
                    sid, action = rest.split("/", 1)
                else:
                    sid, action = rest, ""
                try:
                    if action == "chat":
                        prompt = body.get("prompt", "")
                        if not prompt:
                            return _send_json(self, 400, {"error": "missing 'prompt'"})
                        final = runtime.chat(sid, prompt)
                        return _send_json(self, 200, {
                            "final_text": final,
                            "session": runtime.get_session(sid).to_dict(),
                        })
                    if action == "cancel":
                        runtime.cancel(sid)
                        return _send_empty(self)
                    if action == "reset":
                        runtime.reset_session(sid)
                        return _send_empty(self)
                    if action == "checkpoint":
                        if runtime.session_store is None:
                            return _send_json(self, 400, {"error": "no session store configured"})
                        path_written = runtime.checkpoint_session(sid)
                        return _send_json(self, 200, {"path": str(path_written)})
                except KeyError:
                    return _send_json(self, 404, {"error": f"unknown session: {sid}"})
                except Exception as e:
                    return _send_json(self, 500, {"error": f"{type(e).__name__}: {e}"})
            return _send_json(self, 404, {"error": f"not found: {path}"})

        def do_DELETE(self):  # noqa: N802
            if not self._check_auth():
                return _send_json(self, 401, {"error": "unauthorized"})
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/sessions/"):
                sid = path[len("/sessions/"):]
                try:
                    runtime.end_session(sid)
                except KeyError:
                    return _send_json(self, 404, {"error": f"unknown session: {sid}"})
                return _send_empty(self)
            return _send_json(self, 404, {"error": f"not found: {path}"})

        # --- SSE event stream ------------------------------------------

        def _stream_events(self, query: dict[str, list[str]]):
            session_filter = query.get("session_id", [None])[0]
            kind_filter = query.get("kind", [None])[0]
            include_history = query.get("history", ["false"])[0].lower() in ("1", "true", "yes")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q: queue.Queue[Event | None] = queue.Queue()

            def deliver(event: Event):
                if kind_filter is not None and event.kind != kind_filter:
                    return
                q.put(event)

            sub_id = runtime.subscribe(deliver, session_id=session_filter)
            try:
                if include_history:
                    for e in runtime.events(session_id=session_filter, kind=kind_filter):
                        self._write_sse(e)
                while True:
                    try:
                        event = q.get(timeout=15.0)
                    except queue.Empty:
                        try:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        continue
                    if event is None:
                        break
                    try:
                        self._write_sse(event)
                    except (BrokenPipeError, ConnectionResetError):
                        break
            finally:
                runtime.unsubscribe(sub_id)

        def _write_sse(self, event: Event) -> None:
            data = json.dumps(event.to_dict(), default=str)
            self.wfile.write(f"event: {event.kind}\n".encode("utf-8"))
            self.wfile.write(f"id: {event.id}\n".encode("utf-8"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


class RuntimeServer:
    """Thin wrapper around ThreadingHTTPServer. Supports start/stop in tests."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        auth_token: str | None = None,
        task_queue: TaskQueue | None = None,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.task_queue = task_queue
        self._handler_cls = make_handler(runtime, auth_token=auth_token, task_queue=task_queue)
        self._server = ThreadingHTTPServer((host, port), self._handler_cls)
        self.port = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    parser = argparse.ArgumentParser(description="agi runtime HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auth-token", default=os.environ.get("AGI_AUTH_TOKEN"))
    args = parser.parse_args(argv)

    runtime = Runtime()
    server = RuntimeServer(runtime, host=args.host, port=args.port, auth_token=args.auth_token)
    server.start()
    print(f"agi runtime listening on {server.base_url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
