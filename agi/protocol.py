"""Coordination protocol — a stdio JSON-RPC surface for the Runtime.

A coordination engine in another process (any language) needs a clean,
versioned wire format to drive a Runtime. The HTTP/SSE server already
exists for browser-style use. This module ships the *other* common
surface: newline-delimited JSON-RPC 2.0 over stdio.

Why both? HTTP is right for browser/UI; stdio is right for a
coordinator that spawns the runtime as a subprocess — same shape as
Anthropic's MCP, OpenAI function calling subprocesses, Bazel
workers. A higher-level coordination engine drops a binary that
speaks this protocol and gets a fully-fledged AGI runtime as a
callable primitive.

The protocol exposes (in v1):

  - `runtime.capabilities()`            → dict
  - `runtime.metrics()`                  → dict
  - `session.create(config)`             → {session_id}
  - `session.chat(session_id, input)`    → {final_text}
  - `session.cancel(session_id)`         → {ok}
  - `session.end(session_id)`            → {ok}
  - `session.get(session_id)`            → dict
  - `tasks.submit(prompt, config, ...)`  → {task_id}
  - `tasks.get(task_id)`                 → dict
  - `tasks.drain(max_ticks)`             → {executed}
  - `skills.save(skill)`                 → {ok}
  - `tools.synthesize(name, code, ...)`  → {ok, tool}
  - `events.subscribe()`                  → streams notifications
  - `events.history(since_ts, kind)`     → [event...]

Notifications (server-initiated, no id):

  - `event` — one event from the bus while a subscription is active

This is enough surface for a coordination engine to: discover
capabilities, allocate sessions, run work, cancel, and observe.
Anything richer (autoloop, fork, pool) is composed by the
coordinator on top.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import IO, Any, Callable

from agi.coordinator import Plan, PlanStep
from agi.events import Event
from agi.runtime import Runtime, SessionConfig
from agi.scheduler import (
    CycleError,
    ParallelScheduler,
    RetryPolicy,
    SchedulerConfig,
)
from agi.skills import Skill
from agi.tasks import Task, TaskQueue, TaskRunner, submit_task


PROTOCOL_VERSION = "1.0"


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(obj.__dict__)
        except Exception:
            pass
    return str(obj)


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class CoordinationProtocol:
    """JSON-RPC 2.0 server exposing the Runtime over stdio (or any pair of
    file-like streams).

    Usage:

        from agi.runtime import Runtime
        from agi.protocol import CoordinationProtocol
        CoordinationProtocol(Runtime()).serve_stdio()

    For testing, pass any `read`/`write` streams.

    Thread model: one reader thread parses incoming requests; method
    handlers run on that thread. Notifications are written from the
    bus-subscriber callback (event-emitting thread).
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        queue: TaskQueue | None = None,
        runner: TaskRunner | None = None,
        scheduler: ParallelScheduler | None = None,
    ) -> None:
        self.runtime = runtime
        self.queue = queue or TaskQueue()
        self.runner = runner or TaskRunner(runtime, self.queue)
        self.scheduler = scheduler or ParallelScheduler(runtime)
        self._write_lock = threading.Lock()
        self._writer: IO[str] | None = None
        self._subscribed = False
        self._sub_id: int | None = None
        self._stop = threading.Event()
        self._methods: dict[str, Callable[..., Any]] = self._register_methods()

    # --- registration --------------------------------------------------

    def _register_methods(self) -> dict[str, Callable[..., Any]]:
        return {
            "ping": self._m_ping,
            "version": lambda: {"protocol": PROTOCOL_VERSION},
            "runtime.capabilities": self._m_capabilities,
            "runtime.metrics": self._m_metrics,
            "session.create": self._m_session_create,
            "session.chat": self._m_session_chat,
            "session.cancel": self._m_session_cancel,
            "session.end": self._m_session_end,
            "session.get": self._m_session_get,
            "session.list": self._m_session_list,
            "tasks.submit": self._m_tasks_submit,
            "tasks.get": self._m_tasks_get,
            "tasks.drain": self._m_tasks_drain,
            "plans.submit": self._m_plans_submit,
            "plans.run": self._m_plans_run,
            "plans.get": self._m_plans_get,
            "plans.list": self._m_plans_list,
            "plans.cancel": self._m_plans_cancel,
            "skills.save": self._m_skills_save,
            "tools.synthesize": self._m_tools_synthesize,
            "events.subscribe": self._m_events_subscribe,
            "events.unsubscribe": self._m_events_unsubscribe,
            "events.history": self._m_events_history,
        }

    # --- transport -----------------------------------------------------

    def serve_stdio(self) -> None:  # pragma: no cover - thin entrypoint
        self.serve_streams(sys.stdin, sys.stdout)

    def serve_streams(self, reader: IO[str], writer: IO[str]) -> None:
        """Read newline-delimited JSON-RPC requests; write responses."""
        self._writer = writer
        self._announce()
        for line in reader:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            self._handle_line(line)

    def stop(self) -> None:
        self._stop.set()

    def _announce(self) -> None:
        # Server-initiated banner: lets a fresh coordinator confirm the
        # protocol version before sending anything.
        self._send({
            "jsonrpc": "2.0",
            "method": "ready",
            "params": {"protocol": PROTOCOL_VERSION},
        })

    def _send(self, payload: dict[str, Any]) -> None:
        if self._writer is None:
            return
        line = json.dumps(_jsonable(payload), default=str)
        with self._write_lock:
            self._writer.write(line + "\n")
            self._writer.flush()

    # --- request handling ---------------------------------------------

    def _handle_line(self, line: str) -> None:
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            self._send_error(None, PARSE_ERROR, f"parse error: {e}")
            return
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            self._send_error(req.get("id") if isinstance(req, dict) else None,
                             INVALID_REQUEST, "not a valid JSON-RPC 2.0 request")
            return
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if not isinstance(method, str):
            self._send_error(rid, INVALID_REQUEST, "missing method")
            return
        handler = self._methods.get(method)
        if handler is None:
            self._send_error(rid, METHOD_NOT_FOUND, f"unknown method: {method}")
            return
        try:
            if isinstance(params, list):
                result = handler(*params)
            elif isinstance(params, dict):
                result = handler(**params)
            else:
                self._send_error(rid, INVALID_PARAMS, "params must be list or dict")
                return
        except JsonRpcError as je:
            self._send_error(rid, je.code, je.message, je.data)
            return
        except TypeError as te:
            self._send_error(rid, INVALID_PARAMS, str(te))
            return
        except KeyError as ke:
            self._send_error(rid, INVALID_PARAMS, f"unknown id: {ke}")
            return
        except Exception as e:
            self._send_error(rid, INTERNAL_ERROR, f"{type(e).__name__}: {e}")
            return
        if rid is not None:
            self._send({"jsonrpc": "2.0", "id": rid, "result": _jsonable(result)})

    def _send_error(self, rid: Any, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"jsonrpc": "2.0", "id": rid, "error": err})

    # --- methods -------------------------------------------------------

    def _m_ping(self) -> dict[str, Any]:
        return {"pong": True, "ts": time.time(), "protocol": PROTOCOL_VERSION}

    def _m_capabilities(self) -> dict[str, Any]:
        return self.runtime.capabilities()

    def _m_metrics(self) -> dict[str, Any]:
        return self.runtime.metrics()

    def _m_session_create(self, **config_kwargs: Any) -> dict[str, Any]:
        namespace = config_kwargs.pop("namespace", None)
        parent = config_kwargs.pop("parent_session_id", None)
        # Filter to known SessionConfig fields; anything else is rejected.
        allowed = set(SessionConfig.__dataclass_fields__.keys())
        unknown = set(config_kwargs.keys()) - allowed
        if unknown:
            raise JsonRpcError(INVALID_PARAMS, f"unknown config fields: {sorted(unknown)}")
        config = SessionConfig(**config_kwargs)
        sid = self.runtime.create_session(config, namespace=namespace, parent_session_id=parent)
        return {"session_id": sid}

    def _m_session_chat(self, *, session_id: str, user_input: str) -> dict[str, Any]:
        final = self.runtime.chat(session_id, user_input)
        sess = self.runtime.get_session(session_id)
        return {
            "final_text": final,
            "session": sess.to_dict(),
        }

    def _m_session_cancel(self, *, session_id: str) -> dict[str, Any]:
        self.runtime.cancel(session_id)
        return {"ok": True}

    def _m_session_end(self, *, session_id: str) -> dict[str, Any]:
        self.runtime.end_session(session_id)
        return {"ok": True}

    def _m_session_get(self, *, session_id: str) -> dict[str, Any]:
        return self.runtime.get_session(session_id).to_dict()

    def _m_session_list(self) -> list[dict[str, Any]]:
        return self.runtime.sessions()

    def _m_tasks_submit(
        self,
        *,
        prompt: str,
        session_config: dict[str, Any] | None = None,
        priority: int = 0,
        deadline_ts: float | None = None,
        max_attempts: int = 1,
        namespace: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        cfg = SessionConfig(**(session_config or {}))
        tid = submit_task(
            self.queue,
            prompt=prompt,
            session_config=cfg,
            priority=priority,
            deadline_ts=deadline_ts,
            max_attempts=max_attempts,
            namespace=namespace,
            tag=tag,
        )
        return {"task_id": tid}

    def _m_tasks_get(self, *, task_id: str) -> dict[str, Any]:
        task = self.queue.get(task_id)
        return task.to_dict()

    def _m_tasks_drain(self, *, max_ticks: int = 100) -> dict[str, Any]:
        executed = self.runner.run_until_empty(max_ticks=max_ticks)
        return {"executed": executed}

    def _m_plans_submit(
        self,
        *,
        steps: list[dict[str, Any]],
        rationale: str = "",
        budget_usd: float | None = None,
        deadline_ts: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan = _plan_from_dicts(steps, rationale=rationale)
        try:
            eid = self.scheduler.submit(
                plan,
                budget_usd=budget_usd,
                deadline_ts=deadline_ts,
                metadata=metadata,
            )
        except CycleError as ce:
            raise JsonRpcError(INVALID_PARAMS, f"invalid plan: {ce}") from ce
        return {"execution_id": eid}

    def _m_plans_run(
        self,
        *,
        steps: list[dict[str, Any]],
        rationale: str = "",
        budget_usd: float | None = None,
        deadline_ts: float | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        plan = _plan_from_dicts(steps, rationale=rationale)
        try:
            eid = self.scheduler.submit(
                plan,
                budget_usd=budget_usd,
                deadline_ts=deadline_ts,
                metadata=metadata,
            )
        except CycleError as ce:
            raise JsonRpcError(INVALID_PARAMS, f"invalid plan: {ce}") from ce
        execution = self.scheduler.wait(eid, timeout=timeout)
        return execution.to_dict()

    def _m_plans_get(self, *, execution_id: str) -> dict[str, Any]:
        return self.scheduler.get(execution_id).to_dict()

    def _m_plans_list(self) -> list[dict[str, Any]]:
        return self.scheduler.list_executions()

    def _m_plans_cancel(self, *, execution_id: str) -> dict[str, Any]:
        ok = self.scheduler.cancel(execution_id)
        return {"ok": ok}

    def _m_skills_save(self, *, name: str, description: str, body: str,
                       tags: list[str] | None = None) -> dict[str, Any]:
        self.runtime.save_skill(Skill(name=name, description=description, body=body, tags=tags or []))
        return {"ok": True}

    def _m_tools_synthesize(
        self,
        *,
        name: str,
        description: str,
        code: str,
        input_schema: dict[str, Any] | None = None,
        smoke_test_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool = self.runtime.synthesize_tool(
            name=name,
            description=description,
            code=code,
            input_schema=input_schema,
            smoke_test_kwargs=smoke_test_kwargs,
        )
        return {"ok": True, "tool": {"name": tool.name, "description": tool.description}}

    def _m_events_subscribe(
        self, *, kind: str | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        if self._subscribed:
            return {"already_subscribed": True}

        def _on(e: Event) -> None:
            self._send({
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "kind": e.kind,
                    "session_id": e.session_id,
                    "ts": e.ts,
                    "id": e.id,
                    "data": e.data,
                },
            })

        self._sub_id = self.runtime.bus.subscribe(_on, session_id=session_id, kind=kind)
        self._subscribed = True
        return {"ok": True, "subscription_id": self._sub_id}

    def _m_events_unsubscribe(self) -> dict[str, Any]:
        if self._sub_id is None:
            return {"ok": False}
        self.runtime.bus.unsubscribe(self._sub_id)
        self._sub_id = None
        self._subscribed = False
        return {"ok": True}

    def _m_events_history(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.runtime.events(
            session_id=session_id, kind=kind, since_ts=since_ts, limit=limit
        )]


def _plan_from_dicts(steps: list[dict[str, Any]], rationale: str = "") -> Plan:
    """Build a Plan from JSON-ish step dicts. Accepts the JSON-RPC payload
    shape directly, with friendly errors for missing fields."""
    if not isinstance(steps, list) or not steps:
        raise JsonRpcError(INVALID_PARAMS, "steps must be a non-empty list")
    out: list[PlanStep] = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise JsonRpcError(INVALID_PARAMS, f"step[{i}] must be an object")
        sid = s.get("id")
        prompt = s.get("prompt")
        if not isinstance(sid, str) or not sid:
            raise JsonRpcError(INVALID_PARAMS, f"step[{i}].id required")
        if not isinstance(prompt, str) or not prompt:
            raise JsonRpcError(INVALID_PARAMS, f"step[{i}].prompt required")
        out.append(PlanStep(
            id=sid,
            prompt=prompt,
            role=s.get("role", "executor"),
            model=s.get("model"),
            depends_on=list(s.get("depends_on") or []),
            use_skills=bool(s.get("use_skills", True)),
            priority=int(s.get("priority", 0)),
            namespace=s.get("namespace"),
            metadata=dict(s.get("metadata") or {}),
        ))
    return Plan(steps=out, rationale=rationale)


