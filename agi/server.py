"""HTTP/JSON façade over Runtime.

A coordination engine that lives in another process or another language
talks to the agent through this server. The wire format is intentionally
boring: JSON requests and responses, REST-shaped paths, no server-side
state beyond the Runtime itself.

Endpoints (all under /v1):

  GET  /health
  GET  /capabilities
  GET  /stats

  POST   /sessions               body: {role, goal, model?, max_tokens?}
  GET    /sessions               list (active_only=1 to filter)
  GET    /sessions/{id}          snapshot
  DELETE /sessions/{id}          end (?reason=...)
  GET    /sessions/{id}/transcript
  POST   /sessions/{id}/step     body: {input, max_iterations?}
  POST   /sessions/{id}/inject   body: {text, role?}

  GET    /skills                 list (?q=... search)
  POST   /skills                 body: {title, when, procedure, failure_modes?, triggers?}
  DELETE /skills/{id}

  GET    /memory/search          ?q=...&k=...
  GET    /memory/recent          ?k=...
  POST   /memory                 body: {text, tags?}

Authentication: an optional bearer token (env AGI_API_TOKEN). When set,
requests must include `Authorization: Bearer <token>`.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from agi.memory import Memory
from agi.runtime import Runtime
from agi.skills import SkillLibrary


# ---- routing helpers ----

# Each route is (method, regex) → handler(self, match, body) returning (status, payload).
RouteHandler = Callable[["RuntimeHandler", "re.Match[str]", dict | None], tuple[int, Any]]


class _Router:
    def __init__(self) -> None:
        self.routes: list[tuple[str, "re.Pattern[str]", RouteHandler]] = []

    def add(self, method: str, pattern: str, handler: RouteHandler) -> None:
        self.routes.append((method, re.compile(f"^{pattern}$"), handler))

    def match(self, method: str, path: str):
        for m, pat, h in self.routes:
            if m != method:
                continue
            mo = pat.match(path)
            if mo:
                return h, mo
        return None, None


def _ok(payload: Any) -> tuple[int, Any]:
    return 200, payload


def _err(status: int, message: str) -> tuple[int, Any]:
    return status, {"error": message}


# ---- handler ----

class RuntimeHandler(BaseHTTPRequestHandler):
    server_version = "agi-runtime/1"
    runtime: Runtime  # injected at server build time
    router: _Router
    auth_token: str | None

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet by default; route through stderr so it doesn't fight stdout.
        sys.stderr.write("[runtime] " + (format % args) + "\n")

    # ---- request loop ----

    def _check_auth(self) -> bool:
        if not self.auth_token:
            return True
        got = self.headers.get("Authorization", "")
        if got == f"Bearer {self.auth_token}":
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return None
        raw = self.rfile.read(length)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"invalid JSON: {e}"})
            return None

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str) -> None:
        if not self._check_auth():
            return
        body = self._read_body() if method in ("POST", "PUT", "PATCH", "DELETE") else None
        url = urlparse(self.path)
        handler, mo = self.router.match(method, url.path)
        if handler is None:
            self._send(404, {"error": f"no route: {method} {url.path}"})
            return
        # Make query string available to handlers via self.query
        self.query = parse_qs(url.query, keep_blank_values=True)  # type: ignore[attr-defined]
        try:
            status, payload = handler(self, mo, body)
        except KeyError as e:
            status, payload = 404, {"error": str(e)}
        except ValueError as e:
            status, payload = 400, {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            status, payload = 500, {"error": f"{type(e).__name__}: {e}"}
        self._send(status, payload)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


# ---- routes ----

def _build_router() -> _Router:
    r = _Router()

    def health(h: RuntimeHandler, m, body):
        return _ok({"ok": True})

    def capabilities(h, m, body):
        return _ok(h.runtime.capabilities())

    def stats(h, m, body):
        return _ok(h.runtime.stats())

    def list_sessions(h, m, body):
        active_only = h.query.get("active_only", ["0"])[0] in ("1", "true", "yes")
        return _ok({"sessions": h.runtime.list_sessions(active_only=active_only)})

    def create_session(h, m, body):
        body = body or {}
        sess = h.runtime.create_session(
            role=body.get("role", "general"),
            goal=body.get("goal"),
            model=body.get("model"),
            max_tokens=body.get("max_tokens", 16000),
            verbose=False,
        )
        return 201, sess.snapshot()

    def get_session(h, m, body):
        return _ok(h.runtime.get(m.group("id")).snapshot())

    def end_session(h, m, body):
        reason = h.query.get("reason", ["complete"])[0]
        return _ok(h.runtime.end_session(m.group("id"), reason=reason))

    def transcript(h, m, body):
        sess = h.runtime.get(m.group("id"))
        return _ok({"transcript": sess.transcript()})

    def step_session(h, m, body):
        body = body or {}
        text = body.get("input")
        if not isinstance(text, str) or not text:
            return _err(400, "missing 'input' string")
        sess = h.runtime.get(m.group("id"))
        result = sess.step(text, max_iterations=body.get("max_iterations", 25))
        return _ok(
            {
                "text": result.text,
                "tool_calls": result.tool_calls,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "duration_seconds": result.duration_seconds,
                "critic_score": result.critic_score,
                "error": result.error,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cost_usd": result.usage.cost_usd(sess.agent.model),
                },
                "session": sess.snapshot(),
            }
        )

    def inject(h, m, body):
        body = body or {}
        text = body.get("text")
        if not isinstance(text, str):
            return _err(400, "missing 'text'")
        sess = h.runtime.get(m.group("id"))
        sess.inject_observation(text, role=body.get("role", "user"))
        return _ok({"ok": True, "session": sess.snapshot()})

    # Skills

    def list_skills(h, m, body):
        q = h.query.get("q", [""])[0]
        skills = h.runtime.skills.search(q, k=20) if q else h.runtime.skills.all()
        return _ok({
            "skills": [
                {
                    "id": s.id,
                    "title": s.title,
                    "when": s.when,
                    "procedure": s.procedure,
                    "failure_modes": s.failure_modes,
                    "triggers": s.triggers,
                    "created": s.created,
                }
                for s in skills
            ]
        })

    def add_skill(h, m, body):
        body = body or {}
        for k in ("title", "when", "procedure"):
            if not isinstance(body.get(k), str):
                return _err(400, f"missing string field {k!r}")
        s = h.runtime.skills.add(
            title=body["title"],
            when=body["when"],
            procedure=body["procedure"],
            failure_modes=body.get("failure_modes", "(none recorded)"),
            triggers=body.get("triggers") or [],
        )
        return 201, {"id": s.id, "title": s.title, "path": s.path}

    def delete_skill(h, m, body):
        ok = h.runtime.skills.remove(m.group("id"))
        if not ok:
            return _err(404, f"no skill {m.group('id')}")
        return _ok({"ok": True})

    # Memory

    def search_memory(h, m, body):
        q = h.query.get("q", [""])[0]
        k = int(h.query.get("k", ["5"])[0])
        return _ok({
            "results": [
                {"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags}
                for n in h.runtime.memory.search(q, k)
            ]
        })

    def recent_memory(h, m, body):
        k = int(h.query.get("k", ["10"])[0])
        return _ok({
            "results": [
                {"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags}
                for n in h.runtime.memory.recent(k)
            ]
        })

    def add_memory(h, m, body):
        body = body or {}
        text = body.get("text")
        if not isinstance(text, str) or not text:
            return _err(400, "missing 'text'")
        tags = body.get("tags") or []
        if not isinstance(tags, list):
            return _err(400, "'tags' must be a list of strings")
        n = h.runtime.memory.save(text, tags=[str(t) for t in tags])
        return 201, {"id": n.id, "ts": n.ts, "text": n.text, "tags": n.tags}

    # Register routes

    r.add("GET",    r"/v1/health",                     health)
    r.add("GET",    r"/v1/capabilities",               capabilities)
    r.add("GET",    r"/v1/stats",                      stats)

    r.add("GET",    r"/v1/sessions",                   list_sessions)
    r.add("POST",   r"/v1/sessions",                   create_session)
    r.add("GET",    r"/v1/sessions/(?P<id>[a-f0-9]+)", get_session)
    r.add("DELETE", r"/v1/sessions/(?P<id>[a-f0-9]+)", end_session)
    r.add("GET",    r"/v1/sessions/(?P<id>[a-f0-9]+)/transcript", transcript)
    r.add("POST",   r"/v1/sessions/(?P<id>[a-f0-9]+)/step",       step_session)
    r.add("POST",   r"/v1/sessions/(?P<id>[a-f0-9]+)/inject",     inject)

    r.add("GET",    r"/v1/skills",                     list_skills)
    r.add("POST",   r"/v1/skills",                     add_skill)
    r.add("DELETE", r"/v1/skills/(?P<id>[A-Za-z0-9]+)", delete_skill)

    r.add("GET",    r"/v1/memory/search",              search_memory)
    r.add("GET",    r"/v1/memory/recent",              recent_memory)
    r.add("POST",   r"/v1/memory",                     add_memory)

    return r


def make_server(
    runtime: Runtime,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    """Build an HTTP server bound to the given runtime.

    The handler class is constructed dynamically so each server instance
    can hold its own runtime / token without globals.
    """
    router = _build_router()

    class BoundHandler(RuntimeHandler):
        pass

    BoundHandler.runtime = runtime
    BoundHandler.router = router
    BoundHandler.auth_token = auth_token or os.environ.get("AGI_API_TOKEN")

    return ThreadingHTTPServer((host, port), BoundHandler)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    enable_reflection: bool = False,
    auth_token: str | None = None,
) -> None:
    """Block forever serving the runtime. Used by `python -m agi.server`."""
    runtime = Runtime(enable_reflection=enable_reflection)
    server = make_server(runtime, host=host, port=port, auth_token=auth_token)
    sys.stderr.write(f"[runtime] listening on http://{host}:{port}\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[runtime] shutting down\n")
        server.shutdown()


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="agi runtime HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reflect", action="store_true", help="Enable reflection (Haiku call per turn)")
    p.add_argument("--token", default=None, help="Required bearer token; falls back to AGI_API_TOKEN env")
    args = p.parse_args()
    serve(host=args.host, port=args.port, enable_reflection=args.reflect, auth_token=args.token)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
