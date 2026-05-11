"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

The agent emits structured events through an optional `event_sink` callable.
That sink is what `agi.runtime` uses to fan turns out to subscribers
(coordination engines, HTTP/SSE clients, UIs). When no sink is provided the
agent still prints to stdout in verbose mode, so the CLI is unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from agi.costs import Usage
from agi.events import (
    ErrorEvent,
    Event,
    ServerToolUse,
    TextDelta,
    ThinkingDelta,
    ToolUseResult,
    ToolUseStart,
    TurnEnd,
    TurnStart,
    UsageDelta,
)
from agi.memory import Memory
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


SYSTEM_PROMPT = """\
You are an agent built on Claude Opus 4.7. You have tools to read and write files,
run shell commands, search and fetch the web, and manage a persistent long-term
memory that survives across sessions.

Operating principles:
- Plan before acting on multi-step tasks. Decompose, then execute.
- Use tools instead of guessing. If you need a file's contents, read it. If you
  need a fact, search the web. If you remember something useful, save it.
- Use long-term memory deliberately. Save user preferences, project facts, and
  durable lessons learned. Search memory at the start of related tasks.
- Verify before claiming success. Read back files you wrote. Run tests where
  applicable. State limitations honestly.
- Be terse. Skip preamble. Show the work, not the throat-clearing.
"""


def _stringify_tool_result(result) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


EventSink = Callable[[Event], None]


class Agent:
    def __init__(
        self,
        memory: Memory | None = None,
        model: str = "claude-opus-4-7",
        max_tokens: int = 16000,
        effort: str = "high",
        enable_web_search: bool = True,
        enable_web_fetch: bool = True,
        verbose: bool = True,
        tracer=None,
        critic=None,
        critic_threshold: float = 0.5,
        client: Any | None = None,
        event_sink: EventSink | None = None,
        session_id: str = "local",
        extra_tools: tuple[list[dict], dict[str, Callable[..., str]]] | None = None,
        extra_system_prompt: str = "",
    ) -> None:
        self.client = client if client is not None else anthropic.Anthropic()
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer  # optional TraceLogger for the learning loop
        self.critic = critic  # optional learner.Critic; gates final output
        self.critic_threshold = critic_threshold
        self.last_critic_score: float | None = None
        self.event_sink = event_sink
        self.session_id = session_id
        self.extra_system_prompt = extra_system_prompt
        self.messages: list[dict] = []
        self.usage = Usage()

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = dict(handlers)

        if extra_tools is not None:
            extra_schemas, extra_handlers = extra_tools
            self.tool_schemas.extend(extra_schemas)
            self.handlers.update(extra_handlers)

        if enable_web_search:
            self.tool_schemas.append(
                {"type": "web_search_20260209", "name": "web_search"}
            )
        if enable_web_fetch:
            self.tool_schemas.append(
                {"type": "web_fetch_20260209", "name": "web_fetch"}
            )

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def _emit(self, event: Event) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(event)
        except Exception:
            # An event sink must never break the agent loop. If a subscriber
            # blew up, drop the event and continue — the trace logger still
            # captures the canonical record.
            pass

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._emit(TurnStart(session_id=self.session_id, user_input=user_input))

        last_text = ""
        stop_reason = "end_turn"
        turn_usage = Usage()

        try:
            for _ in range(max_iterations):
                response = self._stream_one()

                self.messages.append({"role": "assistant", "content": response.content})
                self.usage.add(response.usage)
                turn_usage.add(response.usage)

                # Emit per-LLM-call usage so the coordinator can meter spend.
                self._emit(
                    UsageDelta(
                        session_id=self.session_id,
                        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                        cache_creation_input_tokens=getattr(
                            response.usage, "cache_creation_input_tokens", 0
                        ) or 0,
                        cache_read_input_tokens=getattr(
                            response.usage, "cache_read_input_tokens", 0
                        ) or 0,
                        cost_usd=turn_usage.cost_usd(self.model),
                    )
                )

                for block in response.content:
                    if block.type == "text" and block.text:
                        last_text = block.text
                    if block.type == "server_tool_use":
                        self._emit(
                            ServerToolUse(session_id=self.session_id, name=block.name)
                        )

                stop_reason = response.stop_reason or "end_turn"

                if response.stop_reason == "end_turn":
                    break

                if response.stop_reason == "pause_turn":
                    continue

                if response.stop_reason == "tool_use":
                    tool_results = self._dispatch_tool_calls(response.content)
                    if not tool_results:
                        break
                    self.messages.append({"role": "user", "content": tool_results})
                    continue

                break
        except Exception as e:
            self._emit(
                ErrorEvent(
                    session_id=self.session_id,
                    message=str(e),
                    exc_type=type(e).__name__,
                )
            )
            raise

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        if self.tracer is not None:
            metadata: dict = {"session_id": self.session_id}
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            self.tracer.log(
                model=self.model,
                messages=self.messages,
                final_text=last_text,
                usage={
                    "input_tokens": turn_usage.input_tokens,
                    "output_tokens": turn_usage.output_tokens,
                    "cache_creation_input_tokens": turn_usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": turn_usage.cache_read_input_tokens,
                },
                metadata=metadata,
            )

        self._emit(
            TurnEnd(
                session_id=self.session_id,
                final_text=last_text,
                stop_reason=stop_reason,
                critic_score=critic_score,
                cost_usd=turn_usage.cost_usd(self.model),
            )
        )
        return last_text

    def _apply_critic_gate(self, prompt: str, response: str) -> tuple[str, float | None]:
        """Score the response with the critic; annotate if below threshold.

        Returns (possibly-annotated response, score). If no critic is set,
        returns (response, None) — opt-in feature, default off.

        v1: annotate only. Future options: regenerate with hint, refuse,
        surface a structured uncertainty signal to the caller.
        """
        if self.critic is None:
            return response, None
        score = self.critic.predict_proba(prompt, response)
        if score < self.critic_threshold:
            warning = f"\n\n[critic confidence: {score:.2f} (< {self.critic_threshold}) — response may be unreliable]"
            if self.verbose:
                print(warning, flush=True)
            return response + warning, score
        return response, score

    def _system_blocks(self) -> list[dict]:
        text = SYSTEM_PROMPT
        if self.extra_system_prompt:
            text = text + "\n\n" + self.extra_system_prompt
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _stream_one(self):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_blocks(),
            tools=self.tool_schemas,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
        ) as stream:
            if self.verbose or self.event_sink is not None:
                for event in stream:
                    self._handle_stream_event(event)
            else:
                for _ in stream:
                    pass
            return stream.get_final_message()

    def _handle_stream_event(self, event) -> None:
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                if self.verbose:
                    print("\n[thinking] ", end="", flush=True)
            elif block.type == "tool_use":
                if self.verbose:
                    print(f"\n[tool: {block.name}]", end="", flush=True)
                self._emit(
                    ToolUseStart(
                        session_id=self.session_id,
                        tool_use_id=block.id,
                        name=block.name,
                        input=dict(block.input or {}) if block.input else {},
                    )
                )
            elif block.type == "server_tool_use":
                if self.verbose:
                    print(f"\n[server: {block.name}]", end="", flush=True)
            elif block.type == "text":
                if self.verbose:
                    print()
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                if self.verbose:
                    print(d.thinking, end="", flush=True)
                self._emit(
                    ThinkingDelta(session_id=self.session_id, text=d.thinking)
                )
            elif d.type == "text_delta":
                if self.verbose:
                    print(d.text, end="", flush=True)
                self._emit(TextDelta(session_id=self.session_id, text=d.text))

    def _dispatch_tool_calls(self, content) -> list[dict]:
        tool_results: list[dict] = []
        for block in content:
            if block.type != "tool_use":
                continue
            handler = self.handlers.get(block.name)
            if handler is None:
                # web_search / web_fetch and other server-side tools land here;
                # skip — the API already handled them and inlined the results.
                continue
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            output_str = _stringify_tool_result(result)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output_str,
                    "is_error": is_error,
                }
            )
            self._emit(
                ToolUseResult(
                    session_id=self.session_id,
                    tool_use_id=block.id,
                    name=block.name,
                    output=output_str,
                    is_error=is_error,
                )
            )
        return tool_results
