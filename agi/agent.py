"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

The agent can run standalone (build it, call `chat`) or as the executor
inside a runtime Task. The runtime hooks in via these optional params:

  backend       — LLM transport. If None, builds anthropic.Anthropic() directly.
  event_sink    — callable that receives structured events for the engine to surface.
  cancel_check  — called between turns and at tool boundaries; raise to abort.
  budget_check  — called between turns with usage so far; raise to abort.
  extra_tools   — list of (schema, handler) pairs injected at construction.
  system_suffix — appended to SYSTEM_PROMPT (skill library uses this).
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from agi.costs import Usage
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
        backend=None,
        event_sink: Optional[Callable[[str, dict], Any]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
        budget_check: Optional[Callable[..., None]] = None,
        extra_tools: Optional[list[tuple[dict, Callable[..., str]]]] = None,
        system_suffix: str = "",
    ) -> None:
        # Backend abstraction: defaults to AnthropicBackend so standalone use
        # (python -m agi) keeps working with no extra wiring.
        if backend is None:
            from runtime.backend import AnthropicBackend
            backend = AnthropicBackend()
        self.backend = backend
        # Keep `self.client` for backwards-compatible attribute access.
        self.client = getattr(backend, "client", None)
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer  # optional TraceLogger for the learning loop
        self.critic = critic  # optional learner.Critic; gates final output
        self.critic_threshold = critic_threshold
        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self.usage = Usage()

        self._event_sink = event_sink
        self._cancel_check = cancel_check
        self._budget_check = budget_check
        self._system_suffix = system_suffix
        self._start_time = time.time()

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = dict(handlers)

        if extra_tools:
            for schema, handler in extra_tools:
                self.tool_schemas.append(schema)
                self.handlers[schema["name"]] = handler

        if enable_web_search:
            self.tool_schemas.append(
                {"type": "web_search_20260209", "name": "web_search"}
            )
        if enable_web_fetch:
            self.tool_schemas.append(
                {"type": "web_fetch_20260209", "name": "web_fetch"}
            )

    # ----- public API ------------------------------------------------------

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()
        self._start_time = time.time()

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()

        for _ in range(max_iterations):
            self._check_cancel()
            self._check_budget()

            response = self._stream_one()

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)

            self._emit_turn_events(response)
            self._emit(
                "turn_complete",
                {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                    "stop_reason": response.stop_reason,
                },
            )

            for block in response.content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    last_text = block.text

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "pause_turn":
                # Server-side tool hit its iteration limit; re-send to continue.
                continue

            if response.stop_reason == "tool_use":
                tool_results = self._dispatch_tool_calls(response.content)
                if not tool_results:
                    break
                self.messages.append({"role": "user", "content": tool_results})
                continue

            # refusal, max_tokens, stop_sequence, model_context_window_exceeded, ...
            break

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        if self.tracer is not None:
            metadata: dict = {}
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

        return last_text

    # ----- internals -------------------------------------------------------

    def _check_cancel(self) -> None:
        if self._cancel_check is not None:
            self._cancel_check()

    def _check_budget(self) -> None:
        if self._budget_check is None:
            return
        self._budget_check(
            cost_usd=self.usage.cost_usd(self.model),
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            turns=self.usage.turns,
            elapsed_seconds=time.time() - self._start_time,
        )

    def _emit(self, kind: str, data: dict) -> None:
        if self._event_sink is not None:
            try:
                self._event_sink(kind, data)
            except Exception:
                # Never let a misbehaving sink break the agent loop.
                pass

    def _emit_turn_events(self, response) -> None:
        if self._event_sink is None:
            return
        for block in response.content:
            t = getattr(block, "type", None)
            if t == "text" and getattr(block, "text", None):
                self._emit("text", {"text": block.text})
            elif t == "tool_use":
                self._emit(
                    "tool_call",
                    {"name": block.name, "input": dict(block.input or {})},
                )
            elif t == "thinking":
                summary = getattr(block, "summary", None) or getattr(block, "thinking", None)
                if summary:
                    self._emit("thinking_summary", {"text": summary})

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

    def _stream_one(self):
        system_text = SYSTEM_PROMPT + self._system_suffix
        with self.backend.stream_message(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self.tool_schemas,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
        ) as stream:
            if self.verbose:
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
                print("\n[thinking] ", end="", flush=True)
            elif block.type == "tool_use":
                print(f"\n[tool: {block.name}]", end="", flush=True)
            elif block.type == "server_tool_use":
                print(f"\n[server: {block.name}]", end="", flush=True)
            elif block.type == "text":
                print()
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                print(d.thinking, end="", flush=True)
            elif d.type == "text_delta":
                print(d.text, end="", flush=True)
            # input_json_delta is too noisy to show during streaming

    def _dispatch_tool_calls(self, content) -> list[dict]:
        tool_results: list[dict] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            self._check_cancel()
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
            self._emit(
                "tool_result",
                {"name": block.name, "output": _stringify_tool_result(result), "is_error": is_error},
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _stringify_tool_result(result),
                    "is_error": is_error,
                }
            )
        return tool_results
