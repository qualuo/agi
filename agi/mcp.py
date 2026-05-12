"""MCP adapter — expose the Runtime as a Model Context Protocol server.

The Model Context Protocol is Anthropic's open spec for letting clients
(Claude Desktop, Claude Code, claude.ai, any MCP-aware tool) talk to
servers that publish tools, resources, and prompts. Implementing MCP on
top of the Runtime turns this project into something an MCP-aware
coordination engine can drive *without writing any custom integration*.

This module ships a minimal MCP-compatible server speaking newline-
delimited JSON-RPC 2.0 over stdio — the canonical MCP transport. It
exposes the Runtime's primitives as MCP tools:

  - `agi.chat`               — one-turn chat against a session
  - `agi.create_session`     — open a new session
  - `agi.end_session`        — close a session
  - `agi.list_sessions`      — list active sessions
  - `agi.capabilities`       — runtime capabilities snapshot
  - `agi.metrics`            — counters + totals
  - `agi.save_skill`         — persist a Skill
  - `agi.list_skills`        — list available skills
  - `agi.recall`             — query Memory + KnowledgeGraph
  - `agi.run_goal`           — submit a Goal to the AutonomousLoop
  - `agi.autonomy.tick`      — run one AutonomyEngine tick

We also expose two MCP resource kinds for streaming-state read:

  - `agi://events/{session_id}`   — recent events for a session
  - `agi://sessions/{session_id}` — session state JSON

This is intentionally a *subset* of MCP — enough for tools-to-Claude
interop, not the full prompts + sampling surface. Adopters who want
more swap in a fuller framework; the runtime's protocol layer is
already richer (`agi/protocol.py`) and the two coexist.

Investors care because publishing as MCP is the cheapest distribution
play: Claude Desktop and Code can each drive this runtime as a server
with one config line, no custom client code.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict
from typing import Any, Callable, TextIO

from agi.coordinator import Coordinator, Goal
from agi.runtime import Runtime, SessionConfig
from agi.skills import Skill


MCP_VERSION = "2024-11-05"
SERVER_NAME = "agi-runtime"
SERVER_VERSION = "0.2.0"


# --- Tool catalog -----------------------------------------------------

def _tool_catalog() -> list[dict[str, Any]]:
    """Static tool definitions in MCP shape.

    Each entry has name, description, and input_schema (JSON Schema).
    """
    return [
        {
            "name": "agi.create_session",
            "description": (
                "Create a new agent session. Returns {session_id}. The session "
                "is fully isolated: its memory namespace and budget are scoped."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "default": "claude-opus-4-7"},
                    "effort": {"type": "string", "enum": ["low", "medium", "high"], "default": "high"},
                    "namespace": {"type": "string"},
                    "cost_ceiling_usd": {"type": "number"},
                    "enable_tool_synthesis": {"type": "boolean", "default": False},
                    "enable_delegation": {"type": "boolean", "default": False},
                    "use_skills": {"type": "boolean", "default": True},
                    "role": {"type": "string"},
                    "system_prompt_extra": {"type": "string"},
                },
            },
        },
        {
            "name": "agi.chat",
            "description": "Send one user message to a session and return the final text response.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "input": {"type": "string"},
                },
                "required": ["session_id", "input"],
            },
        },
        {
            "name": "agi.end_session",
            "description": "Close a session and roll up its accounting.",
            "inputSchema": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
        {
            "name": "agi.list_sessions",
            "description": "List all sessions on this runtime with state + accounting.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "agi.capabilities",
            "description": (
                "Snapshot of what the runtime can do right now: models, skills, "
                "synthesized tools, active sessions, memory size. Coordinators "
                "call this before dispatching to know what's available."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "agi.metrics",
            "description": "Counters + totals (cost, sessions, tokens) since startup.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "agi.save_skill",
            "description": (
                "Persist a Skill to the procedural memory library. The Skill is "
                "auto-loaded when the agent encounters a matching prompt."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description", "body"],
            },
        },
        {
            "name": "agi.list_skills",
            "description": "List all skills currently in the procedural memory library.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "agi.recall",
            "description": (
                "Search long-term memory and (if attached) the knowledge graph for "
                "anything matching the query. Returns notes + graph entities."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "namespace": {"type": "string"},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
        {
            "name": "agi.run_goal",
            "description": (
                "Submit a Goal to the reference Coordinator and run synchronously. "
                "Returns the final aggregated text + step outcomes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "budget_usd": {"type": "number"},
                    "expect_substring": {"type": "string", "description": "If present, used as the acceptance check."},
                },
                "required": ["intent"],
            },
        },
        {
            "name": "agi.autonomy.tick",
            "description": (
                "Run one tick of the AutonomyEngine. The engine pulls the next goal "
                "from its queue, pursues it, and updates registries/skills. Returns "
                "the TickReport."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


# --- Server -----------------------------------------------------------


class McpServer:
    """Minimal MCP server bound to a Runtime.

    Speaks newline-delimited JSON-RPC 2.0 over the streams it's given.
    Default streams are stdin/stdout so an MCP host can spawn the server
    as a subprocess.

    Construct with optional `coordinator`, `knowledge`, and
    `autonomy_engine` to enable the corresponding tool surface. Without
    them the calls return a typed `not_configured` error.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        coordinator: Coordinator | None = None,
        knowledge=None,            # KnowledgeGraph | None — avoid hard import cycle
        autonomy_engine=None,      # AutonomyEngine | None
        memory=None,               # falls back to runtime.memory
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.runtime = runtime
        self.coordinator = coordinator
        self.knowledge = knowledge
        self.autonomy_engine = autonomy_engine
        self.memory = memory or runtime.memory
        self._in = stdin or sys.stdin
        self._out = stdout or sys.stdout
        self._lock = threading.Lock()
        self._initialized = False

    # --- transport --------------------------------------------------

    def _send(self, obj: dict[str, Any]) -> None:
        with self._lock:
            line = json.dumps(obj, default=str)
            self._out.write(line + "\n")
            try:
                self._out.flush()
            except Exception:
                pass

    def _reply(self, req_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _error(self, req_id: Any, code: int, message: str, *, data: Any = None) -> None:
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"jsonrpc": "2.0", "id": req_id, "error": err})

    # --- main loop --------------------------------------------------

    def serve_forever(self) -> None:
        for line in self._in:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception as e:
                self._error(None, -32700, f"parse error: {e}")
                continue
            self.handle(msg)

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Process one message. For non-notification calls, sends a
        response and also returns the response dict (useful in tests)."""
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": MCP_VERSION,
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {"listChanged": False}, "resources": {}},
                }
                self._initialized = True
                self._reply(req_id, result)
                return result
            if method == "initialized":
                # Notification — no reply
                return None
            if method == "ping":
                self._reply(req_id, {})
                return {}
            if method == "tools/list":
                tools = _tool_catalog()
                self._reply(req_id, {"tools": tools})
                return {"tools": tools}
            if method == "tools/call":
                result = self._dispatch_tool(
                    params.get("name"), params.get("arguments") or {}
                )
                self._reply(req_id, result)
                return result
            if method == "resources/list":
                resources = self._list_resources()
                self._reply(req_id, {"resources": resources})
                return {"resources": resources}
            if method == "resources/read":
                contents = self._read_resource(params.get("uri", ""))
                self._reply(req_id, contents)
                return contents
            if method == "shutdown":
                self._reply(req_id, {})
                return {}
            self._error(req_id, -32601, f"method not found: {method}")
        except Exception as e:
            self._error(req_id, -32000, f"{type(e).__name__}: {e}")
        return None

    # --- tool dispatch ---------------------------------------------

    def _dispatch_tool(self, name: str | None, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(name, str):
            raise ValueError("tools/call requires a name")
        if name == "agi.create_session":
            return self._tool_create_session(args)
        if name == "agi.chat":
            return self._tool_chat(args)
        if name == "agi.end_session":
            return self._tool_end_session(args)
        if name == "agi.list_sessions":
            return {"content": [{"type": "text", "text": json.dumps(self.runtime.sessions(), default=str)}]}
        if name == "agi.capabilities":
            return {"content": [{"type": "text", "text": json.dumps(self.runtime.capabilities(), default=str)}]}
        if name == "agi.metrics":
            return {"content": [{"type": "text", "text": json.dumps(self.runtime.metrics(), default=str)}]}
        if name == "agi.save_skill":
            return self._tool_save_skill(args)
        if name == "agi.list_skills":
            skills = [
                {"name": s.name, "description": s.description, "tags": s.tags}
                for s in self.runtime.skills.all()
            ]
            return {"content": [{"type": "text", "text": json.dumps(skills, default=str)}]}
        if name == "agi.recall":
            return self._tool_recall(args)
        if name == "agi.run_goal":
            return self._tool_run_goal(args)
        if name == "agi.autonomy.tick":
            return self._tool_autonomy_tick()
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
        }

    # --- tool implementations --------------------------------------

    def _tool_create_session(self, args: dict[str, Any]) -> dict[str, Any]:
        cfg_kwargs = {k: v for k, v in args.items() if k != "namespace"}
        cfg = SessionConfig(**cfg_kwargs) if cfg_kwargs else SessionConfig()
        sid = self.runtime.create_session(cfg, namespace=args.get("namespace"))
        return {
            "content": [{"type": "text", "text": json.dumps({"session_id": sid})}]
        }

    def _tool_chat(self, args: dict[str, Any]) -> dict[str, Any]:
        sid = args["session_id"]
        text = self.runtime.chat(sid, args["input"])
        return {"content": [{"type": "text", "text": text}]}

    def _tool_end_session(self, args: dict[str, Any]) -> dict[str, Any]:
        self.runtime.end_session(args["session_id"])
        return {"content": [{"type": "text", "text": "ok"}]}

    def _tool_save_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        skill = Skill(
            name=args["name"],
            description=args["description"],
            body=args["body"],
            tags=list(args.get("tags") or []),
        )
        self.runtime.save_skill(skill)
        return {"content": [{"type": "text", "text": f"saved skill: {skill.name}"}]}

    def _tool_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        q = args["query"]
        ns = args.get("namespace")
        limit = int(args.get("limit", 8))
        mem = self.memory.namespaced(ns) if ns else self.memory
        notes = mem.search(q)[:limit] if hasattr(mem, "search") else []
        graph_hits: list[dict[str, Any]] = []
        if self.knowledge is not None:
            for n in self.knowledge.query_text(q, limit=limit):
                graph_hits.append({"id": n.id, "kind": n.kind, "attrs": n.attrs})
        payload = {
            "notes": [
                {"id": getattr(n, "id", ""), "text": getattr(n, "text", ""),
                 "tags": getattr(n, "tags", [])} for n in notes
            ],
            "graph": graph_hits,
        }
        return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}

    def _tool_run_goal(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.coordinator is None:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "not_configured: no coordinator attached"}],
            }
        intent = args["intent"]
        expected = args.get("expect_substring")
        budget = args.get("budget_usd")
        acceptance = None
        if isinstance(expected, str) and expected:
            es = expected.lower()
            def acceptance(text: str, _es=es) -> bool:
                return _es in text.lower()
        goal = Goal(intent=intent, acceptance=acceptance, budget_usd=budget)
        result = self.coordinator.run(goal)
        payload = {
            "success": result.success,
            "final_text": result.final_text,
            "total_cost_usd": result.total_cost_usd,
            "outcomes": [asdict(o) for o in result.outcomes],
        }
        return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}

    def _tool_autonomy_tick(self) -> dict[str, Any]:
        if self.autonomy_engine is None:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "not_configured: no autonomy engine attached"}],
            }
        report = self.autonomy_engine.run_once()
        return {"content": [{"type": "text", "text": json.dumps(report.to_dict(), default=str)}]}

    # --- resources -------------------------------------------------

    def _list_resources(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in self.runtime.sessions():
            sid = s["id"]
            out.append({
                "uri": f"agi://sessions/{sid}",
                "name": f"session-{sid}",
                "mimeType": "application/json",
            })
            out.append({
                "uri": f"agi://events/{sid}",
                "name": f"events-{sid}",
                "mimeType": "application/json",
            })
        return out

    def _read_resource(self, uri: str) -> dict[str, Any]:
        if uri.startswith("agi://sessions/"):
            sid = uri.removeprefix("agi://sessions/")
            session = self.runtime.get_session(sid)
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(session.to_dict(), default=str),
                }]
            }
        if uri.startswith("agi://events/"):
            sid = uri.removeprefix("agi://events/")
            events = [e.to_dict() for e in self.runtime.events(session_id=sid)]
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(events, default=str),
                }]
            }
        raise ValueError(f"unknown resource uri: {uri}")


def run_stdio(
    runtime: Runtime,
    *,
    coordinator: Coordinator | None = None,
    knowledge=None,
    autonomy_engine=None,
) -> None:
    """Convenience entry point: serve MCP over stdin/stdout until EOF."""
    server = McpServer(
        runtime,
        coordinator=coordinator,
        knowledge=knowledge,
        autonomy_engine=autonomy_engine,
    )
    server.serve_forever()
