"""HTTP surface for the agent runtime.

A coordination engine (Temporal, Inngest, a custom scheduler, a CrewAI-
style orchestrator, a per-tenant queue) calls this. Stdlib only — no
fastapi, no uvicorn — so the runtime ships as a single `pip install -e .`
and runs anywhere Python runs.

Endpoints (all under `/v1`):

  POST   /runs                 submit a run; returns the Run as JSON
  GET    /runs                 list runs
  GET    /runs/{id}            get one run
  POST   /runs/{id}/cancel     cooperative cancel
  GET    /runs/{id}/events     Server-Sent Events stream (replays history)
  GET    /healthz              liveness

The contract is intentionally narrow. The runtime is the executor; the
coordinator handles persistence, retries, and scheduling.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from agi.runtime import Runtime, RunStatus

log = logging.getLogger(__name__)


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict | list) -> None:
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except json.JSONDecodeError:
        return {}


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a Runtime instance."""

    class Handler(BaseHTTPRequestHandler):
        # Quiet the default access logging — we route through `log` below.
        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("%s - %s", self.address_string(), fmt % args)

        # ---- routing ----

        def do_GET(self) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts == ["healthz"]:
                    return _json_response(self, 200, {"ok": True})
                if parts == ["v1", "runs"]:
                    return self._list_runs()
                if len(parts) == 3 and parts[:2] == ["v1", "runs"]:
                    return self._get_run(parts[2])
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "events":
                    return self._stream_events(parts[2])
            except BrokenPipeError:
                # client disconnected mid-stream; nothing to do
                return
            return _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts == ["v1", "runs"]:
                    return self._submit_run()
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "cancel":
                    return self._cancel_run(parts[2])
            except BrokenPipeError:
                return
            return _json_response(self, 404, {"error": "not found"})

        # ---- handlers ----

        def _submit_run(self) -> None:
            body = _read_json_body(self)
            task = body.get("task")
            if not isinstance(task, str) or not task.strip():
                return _json_response(self, 400, {"error": "missing 'task'"})

            try:
                run = runtime.submit(
                    task=task,
                    cost_ceiling_usd=body.get("cost_ceiling_usd"),
                    timeout_seconds=body.get("timeout_seconds"),
                    metadata=body.get("metadata") or {},
                )
            except Exception as e:  # noqa: BLE001
                return _json_response(self, 500, {"error": f"{type(e).__name__}: {e}"})
            return _json_response(self, 201, run.to_public_dict())

        def _list_runs(self) -> None:
            return _json_response(self, 200, [r.to_public_dict() for r in runtime.list_runs()])

        def _get_run(self, run_id: str) -> None:
            run = runtime.get(run_id)
            if run is None:
                return _json_response(self, 404, {"error": "run not found"})
            return _json_response(self, 200, run.to_public_dict())

        def _cancel_run(self, run_id: str) -> None:
            run = runtime.get(run_id)
            if run is None:
                return _json_response(self, 404, {"error": "run not found"})
            ok = run.cancel()
            return _json_response(self, 200, {"id": run_id, "cancelled": ok, "status": run.status.value})

        def _stream_events(self, run_id: str) -> None:
            run = runtime.get(run_id)
            if run is None:
                return _json_response(self, 404, {"error": "run not found"})

            # SSE has no Content-Length; rely on connection close so HTTP/1.0
            # and 1.1 clients both reach EOF cleanly when the run terminates.
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                for event in run.stream(replay=True):
                    msg = f"event: {event.type}\ndata: {json.dumps(event.to_dict())}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # client disconnected; that's fine
                return

    return Handler


def serve(
    runtime: Runtime | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Start a blocking HTTP server. Returns the server object (already
    serving) — used by tests; the script entry point calls
    `serve_forever` after start.
    """
    runtime = runtime or Runtime()
    handler_cls = make_handler(runtime)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    # Stash the runtime on the server object so tests can reach it.
    httpd.runtime = runtime  # type: ignore[attr-defined]
    return httpd


def main() -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="agi runtime HTTP server")
    parser.add_argument("--host", default=os.environ.get("AGI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGI_PORT", "8000")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    httpd = serve(host=args.host, port=args.port)
    log.info("agi runtime serving on http://%s:%s", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
