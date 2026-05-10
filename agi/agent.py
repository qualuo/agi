"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.
"""
from __future__ import annotations

import json
import threading
from typing import Callable

import anthropic

from agi.costs import Usage
from agi.memory import Memory
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore

# Runtime control-plane errors. Re-raised through chat_controlled so the
# Runtime worker can mark the job correctly. Imported here (not from
# agi.runtime) to avoid a circular import — runtime imports Agent.
class BudgetExceeded(RuntimeError):
    pass


class JobCanceled(RuntimeError):
    pass


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
        system_prompt: str | None = None,
        runtime=None,
        session_id: str | None = None,
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
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # Runtime hooks — set by chat_controlled per call, accessible to tools.
        self.runtime = runtime
        self.session_id = session_id
        self._cancel_event: threading.Event | None = None
        self._budget_usd: float | None = None
        self._event_sink: Callable[[str, dict], None] | None = None

        schemas, handlers = make_tools(self.memory, runtime=runtime, parent_agent=self)
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

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        return self._chat_inner(user_input, max_iterations)

    def chat_controlled(
        self,
        user_input: str,
        *,
        cancel_event: threading.Event | None = None,
        budget_usd: float | None = None,
        event_sink: Callable[[str, dict], None] | None = None,
        max_iterations: int = 25,
    ) -> str:
        """Runtime-driven entry point. Same as chat() but honors a cancel
        signal, a $ budget checked between turns, and emits structured events
        to a sink. Used by agi.runtime.Runtime to drive jobs."""
        self._cancel_event = cancel_event
        self._budget_usd = budget_usd
        self._event_sink = event_sink
        try:
            return self._chat_inner(user_input, max_iterations)
        finally:
            self._cancel_event = None
            self._budget_usd = None
            self._event_sink = None

    def _emit(self, kind: str, payload: dict | None = None) -> None:
        if self._event_sink is not None:
            try:
                self._event_sink(kind, payload or {})
            except Exception:
                # Never let an observer crash the worker.
                pass

    def _check_control(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise JobCanceled("job canceled")
        if self._budget_usd is not None:
            spent = self.usage.cost_usd(self.model)
            if spent > self._budget_usd:
                raise BudgetExceeded(
                    f"spent ${spent:.4f} > budget ${self._budget_usd:.4f}"
                )

    def _chat_inner(self, user_input: str, max_iterations: int) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()
        iterations = 0

        for _ in range(max_iterations):
            self._check_control()
            response = self._stream_one()
            iterations += 1

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)
            self._emit("usage", {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cumulative_cost_usd": round(self.usage.cost_usd(self.model), 6),
            })

            for block in response.content:
                if block.type == "text" and block.text:
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
        # Iterate events when *either* the user wants printed output or a
        # runtime is consuming the structured event sink. Skip the loop only
        # when neither is true (small perf win for purely silent runs).
        consume_events = self.verbose or self._event_sink is not None
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
            if consume_events:
                for event in stream:
                    self._handle_stream_event(event)
            else:
                for _ in stream:
                    pass
            return stream.get_final_message()

    def _handle_stream_event(self, event) -> None:
        v = self.verbose
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                if v: print("\n[thinking] ", end="", flush=True)
            elif block.type == "tool_use":
                if v: print(f"\n[tool: {block.name}]", end="", flush=True)
                self._emit("tool_use", {"name": block.name})
            elif block.type == "server_tool_use":
                if v: print(f"\n[server: {block.name}]", end="", flush=True)
                self._emit("tool_use", {"name": block.name, "server": True})
            elif block.type == "text":
                if v: print()
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                if v: print(d.thinking, end="", flush=True)
                self._emit("thinking_delta", {"text": d.thinking})
            elif d.type == "text_delta":
                if v: print(d.text, end="", flush=True)
                self._emit("text_delta", {"text": d.text})
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
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            self._emit("tool_result", {"name": block.name, "is_error": is_error})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _stringify_tool_result(result),
                    "is_error": is_error,
                }
            )
        return tool_results
