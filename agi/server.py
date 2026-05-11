"""HTTP / SSE adapter so a coordination engine can drive the runtime over the wire.

Stdlib-only (`http.server` + `socketserver.ThreadingMixIn`). One thread per
request; each `/runs/{id}/events` request streams a single run's events as
Server-Sent Events until completion.

Endpoints (v1):

    POST   /v1/runs                  body: {"task": "...", "skills": true, ...}
                                     → 201 {"run_id": "...", "stream": "..."}
    GET    /v1/runs                  → list runs
    GET    /v1/runs/{id}             → status snapshot
    GET    /v1/runs/{id}/events      → SSE stream of Events
    POST   /v1/runs/{id}/cancel      → 200 {"cancelled": true}
    GET    /v1/skills                → list skills
    POST   /v1/skills                → create skill
    GET    /v1/memory?q=...          → search memory
    GET    /v1/metrics               → aggregate counters
    GET    /v1/health                → liveness

The wire format for events is JSON-per-SSE-message: each event lands as
`event: <type>\\ndata: {...}\\n\\n`. Compatible with EventSource on the
client side.
"""
from __future__ import annotations

import json
import socketserver
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agi.runtime import Runtime, RunRequest


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    metrics = {
        "runs_started": 0,
        "runs_completed": 0,
        "runs_cancelled": 0,
        "runs_errored": 0,
        "total_cost_usd": 0.0,
    }
    event_counts: Counter[str] = Counter()

    class Handler(BaseHTTPRequestHandler):
        # Keep the runtime reachable on the handler class.
        runtime = None  # type: ignore[assignment]

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            return

        # --- helpers ------------------------------------------------

        def _write_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def _split_path(self) -> tuple[str, dict[str, list[str]]]:
            u = urlparse(self.path)
            return u.path.rstrip("/"), parse_qs(u.query)

        # --- routes -------------------------------------------------

        def do_GET(self) -> None:
            path, qs = self._split_path()
            if path == "/v1/health":
                return self._write_json(200, {"ok": True, "ts": time.time()})
            if path == "/v1/metrics":
                snapshot = dict(metrics)
                snapshot["event_counts"] = dict(event_counts)
                snapshot["live_runs"] = sum(
                    1 for r in runtime.list_runs() if not r["done"]
                )
                return self._write_json(200, snapshot)
            if path == "/v1/runs":
                return self._write_json(200, {"runs": runtime.list_runs()})
            if path == "/v1/skills":
                return self._write_json(200, {
                    "skills": [
                        {"name": s.name, "description": s.description,
                         "tags": s.tags, "uses": s.uses}
                        for s in runtime.skills.all()
                    ],
                })
            if path == "/v1/memory":
                q = (qs.get("q") or [""])[0]
                k = int((qs.get("k") or ["10"])[0])
                if q:
                    notes = runtime.memory.search(q, k=k)
                else:
                    notes = runtime.memory.recent(k=k)
                return self._write_json(200, {
                    "notes": [{"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags}
                              for n in notes],
                })
            if path.startswith("/v1/runs/") and path.endswith("/events"):
                run_id = path.split("/")[3]
                return self._stream_events(run_id)
            if path.startswith("/v1/runs/"):
                run_id = path.split("/")[3]
                return self._run_status(run_id)
            self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path, _ = self._split_path()
            if path == "/v1/runs":
                body = self._read_json()
                task = body.get("task")
                if not task:
                    return self._write_json(400, {"error": "missing 'task'"})
                req = RunRequest(
                    task=task,
                    skills=bool(body.get("skills", True)),
                    reflect=bool(body.get("reflect", True)),
                    max_iterations=int(body.get("max_iterations", 25)),
                    metadata=dict(body.get("metadata") or {}),
                )
                handle = runtime.submit(req)
                metrics["runs_started"] += 1
                return self._write_json(201, {
                    "run_id": handle.run_id,
                    "stream": f"/v1/runs/{handle.run_id}/events",
                })
            if path == "/v1/skills":
                body = self._read_json()
                name = body.get("name")
                description = body.get("description", "")
                content = body.get("body") or body.get("content")
                if not name or not content:
                    return self._write_json(400, {"error": "name and body required"})
                skill = runtime.skills.add(name, description, content, tags=body.get("tags"))
                return self._write_json(201, {
                    "name": skill.name, "slug": skill.slug, "tags": skill.tags,
                })
            if path.startswith("/v1/runs/") and path.endswith("/cancel"):
                run_id = path.split("/")[3]
                handle = runtime.get(run_id)
                if handle is None:
                    return self._write_json(404, {"error": "run not found"})
                handle.cancel()
                metrics["runs_cancelled"] += 1
                return self._write_json(200, {"cancelled": True, "run_id": run_id})
            self._write_json(404, {"error": "not found"})

        # --- streaming ---------------------------------------------

        def _stream_events(self, run_id: str) -> None:
            handle = runtime.get(run_id)
            if handle is None:
                return self._write_json(404, {"error": "run not found"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            # Replay anything already emitted (the consumer might have raced).
            for evt in handle.replay():
                self._write_sse(evt.type, evt.to_dict())
                event_counts[evt.type] += 1
            for evt in handle.events():
                self._write_sse(evt.type, evt.to_dict())
                event_counts[evt.type] += 1
            # Update terminal counters once the stream closes.
            r = handle.result
            if r is not None:
                if r.error:
                    metrics["runs_errored"] += 1
                elif r.cancelled:
                    pass  # already counted on cancel
                else:
                    metrics["runs_completed"] += 1
                metrics["total_cost_usd"] = round(
                    metrics["total_cost_usd"] + (r.cost_usd or 0.0), 6
                )

        def _write_sse(self, event_type: str, payload: dict) -> None:
            line = f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"
            try:
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _run_status(self, run_id: str) -> None:
            handle = runtime.get(run_id)
            if handle is None:
                return self._write_json(404, {"error": "run not found"})
            r = handle.result
            return self._write_json(200, {
                "run_id": handle.run_id,
                "task": handle.request.task,
                "done": handle.is_done(),
                "cancelled": handle.is_cancelled(),
                "elapsed_seconds": time.time() - handle.t0,
                "result": r and {
                    "text": r.text,
                    "passed": r.passed,
                    "critic_score": r.critic_score,
                    "usage": r.usage,
                    "cost_usd": r.cost_usd,
                    "error": r.error,
                },
                "event_count": len(handle.replay()),
            })

    return Handler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(runtime: Runtime, host: str = "127.0.0.1", port: int = 8088) -> ThreadingHTTPServer:
    handler = make_handler(runtime)
    server = ThreadingHTTPServer((host, port), handler)
    return server


def run_forever(host: str = "127.0.0.1", port: int = 8088) -> None:
    rt = Runtime()
    server = serve(rt, host, port)
    print(f"agi runtime listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    run_forever(port=port)
