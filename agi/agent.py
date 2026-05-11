"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

The agent integrates with `agi.runtime` via three optional hooks:
- `cancel_event` — checked between turns and after each tool dispatch.
- `cost_ceiling_usd` — checked after each turn; raises BudgetExceeded.
- `event_callback` — emits structured events for the runtime event bus.

`runtime` + `run_id` together let tools (e.g. `delegate`) spawn child
runs on the same Runtime. None of these hooks are required — the agent
runs standalone without any of them set.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

import anthropic

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
- If a `delegate` tool is available, use it for clearly-separable subtasks
  that benefit from a fresh context — research, parallel exploration, narrow
  verification. Don't delegate trivial work; the coordination overhead costs.
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
        # Runtime integration (all optional)
        event_callback: Callable[[str, dict], None] | None = None,
        cancel_event: threading.Event | None = None,
        cost_ceiling_usd: float | None = None,
        runtime: Any = None,
        run_id: str | None = None,
        extra_tools: list[dict] | None = None,
        extra_handlers: dict[str, Callable[..., str]] | None = None,
    ) -> None:
        self.client = anthropic.Anthropic()
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

        # Runtime hooks
        self.event_callback = event_callback
        self.cancel_event = cancel_event
        self.cost_ceiling_usd = cost_ceiling_usd
        self.runtime = runtime
        self.run_id = run_id

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = handlers

        # The runtime injects extra tools (delegate, skill ops, etc.) without
        # this module needing to know about them.
        if extra_tools:
            self.tool_schemas.extend(extra_tools)
        if extra_handlers:
            self.handlers.update(extra_handlers)

        # If wired to a Runtime, expose the delegate tool automatically.
        if runtime is not None and "delegate" not in self.handlers:
            from agi.delegation import make_delegate_tool

            schema, handler = make_delegate_tool(runtime, parent_run_id=run_id)
            self.tool_schemas.append(schema)
            self.handlers["delegate"] = handler

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

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        from agi.runtime import BudgetExceeded, Cancelled

        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()

        for _ in range(max_iterations):
            self._check_cancel(Cancelled)

            response = self._stream_one()

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)

            self._emit(
                "turn.completed",
                {
                    "input_tokens": response.usage.input_tokens or 0,
                    "output_tokens": response.usage.output_tokens or 0,
                    "stop_reason": response.stop_reason,
                    "cumulative_cost_usd": round(self.usage.cost_usd(self.model), 6),
                },
            )

            self._check_budget(BudgetExceeded)

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "pause_turn":
                # Server-side tool hit its iteration limit; re-send to continue.
                continue

            if response.stop_reason == "tool_use":
                self._check_cancel(Cancelled)
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

    def _check_cancel(self, exc_cls) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise exc_cls("cancellation requested")

    def _check_budget(self, exc_cls) -> None:
        if self.cost_ceiling_usd is None:
            return
        cost = self.usage.cost_usd(self.model)
        if cost > self.cost_ceiling_usd:
            raise exc_cls(
                f"cost ${cost:.4f} exceeded ceiling ${self.cost_ceiling_usd:.4f}"
            )

    def _emit(self, event_type: str, payload: dict) -> None:
        if self.event_callback is not None:
            try:
                self.event_callback(event_type, payload)
            except Exception:
                # Never let an observer crash the agent.
                pass

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
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
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
            if block.type != "tool_use":
                continue
            handler = self.handlers.get(block.name)
            if handler is None:
                # web_search / web_fetch and other server-side tools land here;
                # skip — the API already handled them and inlined the results.
                continue
            self._emit("tool.call", {"name": block.name, "id": block.id})
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            self._emit(
                "tool.result",
                {"name": block.name, "id": block.id, "is_error": is_error},
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
