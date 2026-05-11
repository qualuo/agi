"""AgentRuntime — embeddable agent runtime for a coordination engine.

This is the layer that makes `agi.Agent` consumable by other systems. Where
`Agent.chat()` is a blocking string-in/string-out call meant for a REPL,
`AgentRuntime` exposes:

- **Sessions** with explicit ids, isolated state, and lifecycle management
- **Structured event streams** (`agi.events`) instead of stdout printing
- **Programmatic tool interception**: a hook the coordinator can use to
  approve, deny, or replace tool calls before they execute
- **Budget enforcement**: stop a run when it exceeds cost or turn limits
- **Snapshot / resume**: serialize session state to dict (or disk), rehydrate
  later — required for durable workflows and crash recovery

The runtime is intentionally synchronous. A coordination engine that needs
parallelism wraps it in threads (the SDK and runtime are thread-safe across
distinct sessions). Async support can layer on later without breaking the
sync API.

`AgentRuntime` does not subclass `Agent`. It composes one per session and
drives its own stream loop so it can emit events at the granularity the
coordinator needs. The Agent's `_apply_critic_gate` and `_dispatch_tool_calls`
helpers are reused; the streaming loop is reimplemented to yield events.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agi.agent import SYSTEM_PROMPT, Agent
from agi.costs import Usage
from agi.events import (
    BudgetExceeded,
    Event,
    RuntimeError_,
    SessionStarted,
    TextDelta,
    ThinkingDelta,
    ToolResult,
    ToolUseRequested,
    TurnCompleted,
)
from agi.memory import Memory


# Coordinator hook: called before a client-side tool executes.
# Return None to allow normal execution. Return a string to replace the tool's
# output. Raise to deny with an error.
ToolInterceptor = Callable[[str, str, dict], "str | None"]


@dataclass
class SessionConfig:
    """Per-session knobs the coordinator controls."""
    system_prompt: str = SYSTEM_PROMPT
    model: str = "claude-opus-4-7"
    max_tokens: int = 16000
    effort: str = "high"
    enable_web_search: bool = True
    enable_web_fetch: bool = True
    # Budget caps. None means no limit.
    max_cost_usd: float | None = None
    max_turns: int | None = None
    # Persistence
    memory_path: str | None = None  # None → ephemeral in-process memory


@dataclass
class _Session:
    id: str
    agent: Agent
    config: SessionConfig
    seq: int = 0
    interceptor: ToolInterceptor | None = None
    closed: bool = False
    created_at: float = field(default_factory=time.time)

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


class AgentRuntime:
    """Multi-session agent runtime. One instance can drive many parallel
    sessions; each session owns its own conversation state, memory, and
    budget."""

    def __init__(self, sessions_dir: str | Path | None = None) -> None:
        self.sessions: dict[str, _Session] = {}
        # Optional on-disk session snapshots (~/.agi/sessions by default)
        self.sessions_dir = Path(sessions_dir) if sessions_dir else None
        if self.sessions_dir is not None:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ---- session lifecycle ------------------------------------------------

    def start_session(
        self,
        session_id: str | None = None,
        config: SessionConfig | None = None,
        *,
        interceptor: ToolInterceptor | None = None,
        agent: Agent | None = None,
    ) -> str:
        """Create a session. Returns its id.

        Pass an existing `agent` to reuse one (advanced; mainly for tests).
        Otherwise the runtime constructs an Agent from `config`.
        """
        sid = session_id or uuid.uuid4().hex[:12]
        if sid in self.sessions:
            raise ValueError(f"session {sid} already exists")

        cfg = config or SessionConfig()
        if agent is None:
            memory = Memory(path=cfg.memory_path) if cfg.memory_path else Memory(
                path=Path("/tmp") / f"agi-rt-{sid}.jsonl"
            )
            agent = Agent(
                memory=memory,
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                effort=cfg.effort,
                enable_web_search=cfg.enable_web_search,
                enable_web_fetch=cfg.enable_web_fetch,
                verbose=False,
            )

        self.sessions[sid] = _Session(
            id=sid, agent=agent, config=cfg, interceptor=interceptor
        )
        return sid

    def close_session(self, session_id: str) -> None:
        s = self.sessions.pop(session_id, None)
        if s is not None:
            s.closed = True

    def get_session(self, session_id: str) -> _Session:
        s = self.sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown session {session_id}")
        if s.closed:
            raise RuntimeError(f"session {session_id} is closed")
        return s

    # ---- core driving loop ------------------------------------------------

    def send(
        self,
        session_id: str,
        user_input: str,
        max_iterations: int = 25,
    ) -> Iterator[Event]:
        """Send a user message and yield events from the run.

        The iterator is single-use. Consume it to completion before sending
        another message on the same session — concurrent send() calls on the
        same session are unsupported (race on `agent.messages`).
        """
        s = self.get_session(session_id)
        agent = s.agent

        # Announce the session on its first turn
        if s.seq == 0:
            yield SessionStarted(session_id=s.id, seq=s.next_seq(), model=agent.model)

        agent.messages.append({"role": "user", "content": user_input})

        turn_usage = Usage()
        last_text = ""
        stop_reason = "end_turn"
        iterations = 0

        for _ in range(max_iterations):
            iterations += 1

            try:
                final, events = self._stream_one(s)
            except Exception as e:
                yield RuntimeError_(
                    session_id=s.id,
                    seq=s.next_seq(),
                    error_type=type(e).__name__,
                    message=str(e),
                )
                return

            yield from events

            agent.messages.append({"role": "assistant", "content": final.content})
            agent.usage.add(final.usage)
            turn_usage.add(final.usage)
            stop_reason = final.stop_reason or "end_turn"

            for block in final.content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", ""):
                    last_text = block.text

            # Budget enforcement: check after assistant turn, before tool dispatch
            cfg = s.config
            if cfg.max_cost_usd is not None:
                cost = agent.usage.cost_usd(agent.model)
                if cost > cfg.max_cost_usd:
                    yield BudgetExceeded(
                        session_id=s.id,
                        seq=s.next_seq(),
                        limit_kind="cost_usd",
                        limit_value=cfg.max_cost_usd,
                        actual=cost,
                    )
                    break
            if cfg.max_turns is not None and agent.usage.turns >= cfg.max_turns:
                yield BudgetExceeded(
                    session_id=s.id,
                    seq=s.next_seq(),
                    limit_kind="max_turns",
                    limit_value=cfg.max_turns,
                    actual=agent.usage.turns,
                )
                break

            if stop_reason == "end_turn":
                break
            if stop_reason == "pause_turn":
                # Server-side tool hit its limit; re-send to continue.
                continue
            if stop_reason == "tool_use":
                results, tool_events = self._handle_tools(s, final.content)
                yield from tool_events
                if not results:
                    break
                agent.messages.append({"role": "user", "content": results})
                continue

            # refusal, max_tokens, model_context_window_exceeded, etc.
            break

        last_text, critic_score = agent._apply_critic_gate(user_input, last_text)
        agent.last_critic_score = critic_score

        yield TurnCompleted(
            session_id=s.id,
            seq=s.next_seq(),
            text=last_text,
            stop_reason=stop_reason,
            input_tokens=turn_usage.input_tokens,
            output_tokens=turn_usage.output_tokens,
            cache_creation_input_tokens=turn_usage.cache_creation_input_tokens,
            cache_read_input_tokens=turn_usage.cache_read_input_tokens,
            cost_usd=turn_usage.cost_usd(agent.model),
            critic_score=critic_score,
        )

        if agent.tracer is not None:
            metadata: dict = {}
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            agent.tracer.log(
                model=agent.model,
                messages=agent.messages,
                final_text=last_text,
                usage={
                    "input_tokens": turn_usage.input_tokens,
                    "output_tokens": turn_usage.output_tokens,
                    "cache_creation_input_tokens": turn_usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": turn_usage.cache_read_input_tokens,
                },
                metadata=metadata,
            )

    def _stream_one(self, s: _Session) -> tuple[Any, list[Event]]:
        """Open one streaming request, collect events into a list, return
        (final_message, events). Buffering events here keeps the SDK stream
        context manager tightly scoped — the caller yields after close."""
        events: list[Event] = []
        agent = s.agent
        with agent.client.messages.stream(
            model=agent.model,
            max_tokens=agent.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": s.config.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=agent.tool_schemas,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": agent.effort},
            messages=agent.messages,
        ) as stream:
            for ev in stream:
                self._convert_stream_event(s, ev, events)
            return stream.get_final_message(), events

    def _convert_stream_event(self, s: _Session, ev: Any, out: list[Event]) -> None:
        et = getattr(ev, "type", None)
        if et == "content_block_delta":
            d = getattr(ev, "delta", None)
            dt = getattr(d, "type", None)
            if dt == "text_delta":
                out.append(TextDelta(session_id=s.id, seq=s.next_seq(), text=d.text))
            elif dt == "thinking_delta":
                out.append(ThinkingDelta(session_id=s.id, seq=s.next_seq(), text=d.thinking))

    def _handle_tools(self, s: _Session, content) -> tuple[list[dict], list[Event]]:
        """Dispatch client-side tool calls, emitting ToolUseRequested + ToolResult
        events. Honors the session's `interceptor` hook.

        Server-side tools (web_search, web_fetch) are skipped — the API already
        executed them and inlined the results into the assistant turn.
        """
        agent = s.agent
        events: list[Event] = []
        results: list[dict] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = block.name
            tool_input = block.input or {}
            tool_id = block.id

            handler = agent.handlers.get(name)
            if handler is None:
                # server-side tool; the API already handled it
                continue

            events.append(
                ToolUseRequested(
                    session_id=s.id,
                    seq=s.next_seq(),
                    tool_id=tool_id,
                    tool_name=name,
                    tool_input=dict(tool_input),
                )
            )

            # Coordinator interception
            intercepted = False
            replaced: str | None = None
            if s.interceptor is not None:
                try:
                    replaced = s.interceptor(s.id, name, dict(tool_input))
                except Exception as e:
                    replaced = f"error: interceptor raised {type(e).__name__}: {e}"
                    intercepted = True
                if replaced is not None:
                    intercepted = True

            if intercepted:
                output_str = replaced or ""
                is_error = output_str.startswith("error:")
            else:
                try:
                    raw = handler(**tool_input)
                    output_str = raw if isinstance(raw, str) else json.dumps(raw, default=str)
                    is_error = False
                except Exception as e:
                    output_str = f"error: {type(e).__name__}: {e}"
                    is_error = True

            events.append(
                ToolResult(
                    session_id=s.id,
                    seq=s.next_seq(),
                    tool_id=tool_id,
                    tool_name=name,
                    output=output_str,
                    is_error=is_error,
                    intercepted=intercepted,
                )
            )

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output_str,
                    "is_error": is_error,
                }
            )

        return results, events

    # ---- snapshot / resume ------------------------------------------------

    def snapshot(self, session_id: str) -> dict:
        """Serialize session state to a JSON-safe dict. Conversation history
        is preserved; tool interceptor and memory backend are not (they must
        be re-supplied on restore)."""
        s = self.get_session(session_id)
        return {
            "id": s.id,
            "created_at": s.created_at,
            "seq": s.seq,
            "config": {
                "system_prompt": s.config.system_prompt,
                "model": s.config.model,
                "max_tokens": s.config.max_tokens,
                "effort": s.config.effort,
                "enable_web_search": s.config.enable_web_search,
                "enable_web_fetch": s.config.enable_web_fetch,
                "max_cost_usd": s.config.max_cost_usd,
                "max_turns": s.config.max_turns,
                "memory_path": s.config.memory_path,
            },
            "messages": _serialize_messages(s.agent.messages),
            "usage": {
                "input_tokens": s.agent.usage.input_tokens,
                "output_tokens": s.agent.usage.output_tokens,
                "cache_creation_input_tokens": s.agent.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": s.agent.usage.cache_read_input_tokens,
                "turns": s.agent.usage.turns,
            },
        }

    def save_snapshot(self, session_id: str) -> Path:
        """Persist a snapshot to `sessions_dir`."""
        if self.sessions_dir is None:
            raise ValueError("sessions_dir not configured")
        snap = self.snapshot(session_id)
        path = self.sessions_dir / f"{session_id}.json"
        path.write_text(json.dumps(snap, indent=2, default=str))
        return path

    def restore(
        self,
        snapshot: dict,
        *,
        interceptor: ToolInterceptor | None = None,
    ) -> str:
        """Rebuild a session from a snapshot dict. Returns the session id."""
        cfg = SessionConfig(**snapshot["config"])
        sid = snapshot["id"]
        if sid in self.sessions:
            raise ValueError(f"session {sid} already exists in this runtime")

        memory = Memory(path=cfg.memory_path) if cfg.memory_path else Memory(
            path=Path("/tmp") / f"agi-rt-{sid}.jsonl"
        )
        agent = Agent(
            memory=memory,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            effort=cfg.effort,
            enable_web_search=cfg.enable_web_search,
            enable_web_fetch=cfg.enable_web_fetch,
            verbose=False,
        )
        # Re-hydrate conversation and usage
        agent.messages = list(snapshot.get("messages", []))
        u = snapshot.get("usage", {})
        agent.usage.input_tokens = u.get("input_tokens", 0)
        agent.usage.output_tokens = u.get("output_tokens", 0)
        agent.usage.cache_creation_input_tokens = u.get("cache_creation_input_tokens", 0)
        agent.usage.cache_read_input_tokens = u.get("cache_read_input_tokens", 0)
        agent.usage.turns = u.get("turns", 0)

        self.sessions[sid] = _Session(
            id=sid,
            agent=agent,
            config=cfg,
            seq=snapshot.get("seq", 0),
            interceptor=interceptor,
            created_at=snapshot.get("created_at", time.time()),
        )
        return sid

    def load_snapshot(
        self, session_id: str, *, interceptor: ToolInterceptor | None = None
    ) -> str:
        if self.sessions_dir is None:
            raise ValueError("sessions_dir not configured")
        path = self.sessions_dir / f"{session_id}.json"
        snap = json.loads(path.read_text())
        return self.restore(snap, interceptor=interceptor)


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Convert agent.messages (which may contain SDK Pydantic blocks) into
    plain dicts safe to JSON-encode.

    This is a small mirror of learner.traces._serialize_messages — duplicated
    here to keep the runtime import-light (the learner package pulls torch
    transitively, which we don't want as a runtime dependency).
    """
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif isinstance(content, list):
            blocks = []
            for b in content:
                if hasattr(b, "model_dump"):
                    blocks.append(b.model_dump(exclude_none=True))
                elif isinstance(b, dict):
                    blocks.append(b)
                else:
                    blocks.append({"type": "unknown", "repr": repr(b)})
            out.append({"role": m["role"], "content": blocks})
        else:
            out.append({"role": m["role"], "content": repr(content)})
    return out
