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
from typing import Callable

import anthropic

from agi.costs import Usage
from agi.events import (
    BUDGET_EXCEEDED,
    CRITIC_SCORED,
    EventBus,
    TEXT_DELTA,
    THINKING_DELTA,
    TOOL_COMPLETED,
    TOOL_ERRORED,
    TOOL_INVOKED,
    TURN_COMPLETED,
    TURN_ERRORED,
    TURN_STARTED,
)
from agi.memory import Memory
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger, _serialize_messages
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore

    def _serialize_messages(messages):  # type: ignore
        return list(messages)


class BudgetExceeded(Exception):
    """Raised mid-turn when the runtime budget is exhausted."""

    def __init__(self, reason: str, partial_text: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial_text = partial_text


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
        bus: EventBus | None = None,
        budget=None,
        extra_system: str | None = None,
        extra_tool_schemas: list[dict] | None = None,
        extra_tool_handlers: dict | None = None,
        enable_coordination: bool = True,
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
        self.bus = bus  # optional EventBus for runtime observers
        self.budget = budget  # optional Budget enforced per-turn / per-session
        self.extra_system = extra_system  # appended to base SYSTEM_PROMPT (skills, role, etc.)

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = handlers

        if enable_coordination:
            from agi.coordination import make_coordination_tools

            cs, ch = make_coordination_tools(
                parent_usage=self.usage,
                parent_memory=self.memory,
                parent_model=self.model,
                parent_extra_system=self.extra_system,
            )
            self.tool_schemas.extend(cs)
            self.handlers.update(ch)

        if extra_tool_schemas:
            self.tool_schemas.extend(extra_tool_schemas)
        if extra_tool_handlers:
            self.handlers.update(extra_tool_handlers)

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

    def snapshot(self) -> dict:
        """Serialize the agent's resumable state. Pairs with `restore`."""
        return {
            "model": self.model,
            "messages": _serialize_messages(self.messages),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "turns": self.usage.turns,
            },
            "extra_system": self.extra_system,
        }

    def restore(self, snapshot: dict) -> None:
        self.messages = list(snapshot.get("messages", []))
        u = snapshot.get("usage", {})
        self.usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
            turns=u.get("turns", 0),
        )
        self.extra_system = snapshot.get("extra_system", self.extra_system)

    def _emit(self, type: str, data: dict | None = None) -> None:
        if self.bus is not None:
            self.bus.emit(type, data or {})

    def _check_budget(self, partial_text: str = "") -> None:
        if self.budget is None:
            return
        reason = self.budget.violation(self.usage, self.model)
        if reason is not None:
            self._emit(BUDGET_EXCEEDED, {"reason": reason})
            raise BudgetExceeded(reason, partial_text=partial_text)

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()
        self._emit(TURN_STARTED, {"prompt": user_input})

        try:
            for _ in range(max_iterations):
                self._check_budget(partial_text=last_text)
                response = self._stream_one()

                self.messages.append({"role": "assistant", "content": response.content})
                self.usage.add(response.usage)
                turn_usage.add(response.usage)

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
        except BudgetExceeded as e:
            last_text = (last_text + f"\n\n[runtime: {e.reason} — turn aborted]").strip()

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score
        if critic_score is not None:
            self._emit(CRITIC_SCORED, {"score": critic_score, "threshold": self.critic_threshold})

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        self._emit(
            TURN_COMPLETED,
            {
                "text": last_text,
                "cost_usd": turn_usage.cost_usd(self.model),
                "input_tokens": turn_usage.input_tokens,
                "output_tokens": turn_usage.output_tokens,
                "cache_read_input_tokens": turn_usage.cache_read_input_tokens,
                "cache_creation_input_tokens": turn_usage.cache_creation_input_tokens,
            },
        )

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

    def _system_blocks(self) -> list[dict]:
        blocks: list[dict] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.extra_system:
            blocks.append({"type": "text", "text": self.extra_system})
        return blocks

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
            # Always iterate so the bus gets events; verbose controls printing only.
            for event in stream:
                self._handle_stream_event(event)
            return stream.get_final_message()

    def _handle_stream_event(self, event) -> None:
        v = self.verbose
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                if v:
                    print("\n[thinking] ", end="", flush=True)
            elif block.type == "tool_use":
                if v:
                    print(f"\n[tool: {block.name}]", end="", flush=True)
            elif block.type == "server_tool_use":
                if v:
                    print(f"\n[server: {block.name}]", end="", flush=True)
            elif block.type == "text":
                if v:
                    print()
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                if v:
                    print(d.thinking, end="", flush=True)
                self._emit(THINKING_DELTA, {"text": d.thinking})
            elif d.type == "text_delta":
                if v:
                    print(d.text, end="", flush=True)
                self._emit(TEXT_DELTA, {"text": d.text})
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
            self._emit(TOOL_INVOKED, {"name": block.name, "input": block.input or {}, "tool_use_id": block.id})
            try:
                result = handler(**(block.input or {}))
                is_error = False
                self._emit(TOOL_COMPLETED, {"name": block.name, "tool_use_id": block.id, "result_preview": _stringify_tool_result(result)[:200]})
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
                self._emit(TOOL_ERRORED, {"name": block.name, "tool_use_id": block.id, "error": str(e)})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _stringify_tool_result(result),
                    "is_error": is_error,
                }
            )
        return tool_results
