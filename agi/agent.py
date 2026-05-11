"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

Two ways to observe activity:
  - `verbose=True` (default): pretty-print to stdout as before.
  - `on_event=callable`: receive structured `agi.events.Event` objects.
The Runtime uses the callback path; the REPL uses verbose. Both can be on.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

import anthropic

from agi.budget import Budget
from agi.costs import Usage
from agi.events import (
    BudgetExceeded,
    Event,
    SessionFinished,
    SessionStarted,
    TextDelta,
    ThinkingDelta,
    ToolResult,
    ToolUse,
    TurnFinished,
    TurnStarted,
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


EventCallback = Callable[[Event], None]


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
        on_event: Optional[EventCallback] = None,
        budget: Optional[Budget] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        system_prompt: str = SYSTEM_PROMPT,
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
        self.on_event = on_event
        self.budget = budget
        self.cancel_check = cancel_check
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self.usage = Usage()

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = handlers

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
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            # Subscribers must not break the agent loop.
            pass

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()
        if self.budget is not None:
            self.budget.reset_clock()

        budget_hit: tuple[str, float, float] | None = None

        for iteration in range(max_iterations):
            self._emit(TurnStarted(iteration=iteration))
            response = self._stream_one()

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text

            self._emit(
                TurnFinished(
                    stop_reason=response.stop_reason or "",
                    input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                    cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                    cost_usd=self.usage.cost_usd(self.model),
                )
            )

            if self.cancel_check is not None and self.cancel_check():
                budget_hit = ("cancelled", 0.0, 0.0)
                self._emit(BudgetExceeded(reason="cancelled", limit=0.0, actual=0.0))
                break

            if self.budget is not None:
                budget_hit = self.budget.check(
                    cost_usd=self.usage.cost_usd(self.model),
                    input_tokens=self.usage.input_tokens,
                    output_tokens=self.usage.output_tokens,
                    iterations=iteration + 1,
                )
                if budget_hit is not None:
                    reason, limit, actual = budget_hit
                    self._emit(BudgetExceeded(reason=reason, limit=limit, actual=actual))
                    break

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
            if budget_hit is not None:
                metadata["budget_exceeded"] = budget_hit[0]
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
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self.tool_schemas,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
        ) as stream:
            for event in stream:
                self._handle_stream_event(event)
            return stream.get_final_message()

    def _handle_stream_event(self, event) -> None:
        # Pretty-print path for the REPL.
        if self.verbose:
            self._render_stream_event(event)

        # Structured-event path for the Runtime / external subscribers.
        if self.on_event is None:
            return

        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                self._emit(ToolUse(name=block.name, server_side=False, input={}))
            elif block.type == "server_tool_use":
                self._emit(ToolUse(name=block.name, server_side=True, input={}))
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                self._emit(ThinkingDelta(text=d.thinking))
            elif d.type == "text_delta":
                self._emit(TextDelta(text=d.text))

    def _render_stream_event(self, event) -> None:
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
            t0 = time.time()
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            elapsed_ms = int((time.time() - t0) * 1000)
            output = _stringify_tool_result(result)
            self._emit(
                ToolResult(
                    name=block.name,
                    output=output[:4096],  # cap for event stream; full result still goes to the model
                    is_error=is_error,
                    elapsed_ms=elapsed_ms,
                )
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        return tool_results
