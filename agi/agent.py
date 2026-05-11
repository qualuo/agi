"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

When driven by a `Runtime`, the agent is also wired to:
  - an event sink (the parent `Run`) — receives structured events,
  - a `Budget` — enforced between turns and after tool calls,
  - the `Runtime` itself — used by the `delegate` tool to spawn subagents,
  - `depth` — current subagent depth (to cap recursion),
  - `role_addendum` — extra system-prompt text for planner/executor/critic,
  - `preload_skills` — skills loaded into context up-front.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Callable

import anthropic

from agi import events as ev
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

When you discover a reusable procedure, save it as a skill with `save_skill` so
future tasks of the same shape are cheaper. When a task is plausibly already
covered, call `search_skills` before planning.

When a sub-problem is well-scoped and self-contained, you may `delegate` it to
a sub-agent (planner, executor, critic, researcher, or general). Delegation
costs roll up to this run.
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
        # Runtime integration — all optional. Set by Runtime._build_agent.
        event_sink=None,
        budget=None,
        runtime=None,
        depth: int = 0,
        role_addendum: str = "",
        preload_skills: list[str] | None = None,
        skill_library=None,
    ) -> None:
        self.client = anthropic.Anthropic()
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self.usage = Usage()

        self.event_sink = event_sink
        self.budget = budget
        self.runtime = runtime
        self.depth = depth
        self.role_addendum = role_addendum
        self.skill_library = skill_library
        self._start_time = time.time()

        schemas, handlers = make_tools(
            self.memory,
            runtime=runtime,
            current_agent=self,
            skill_library=skill_library,
        )
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

        if preload_skills and skill_library is not None:
            self._preload_skills(preload_skills)

    def _preload_skills(self, names: list[str]) -> None:
        """Inject named skills as a system reminder before the first user turn."""
        if self.skill_library is None:
            return
        loaded = [self.skill_library.load(n) for n in names]
        loaded = [s for s in loaded if s is not None]
        if not loaded:
            return
        block = "\n\n".join(s.render() for s in loaded)
        self.messages.append({
            "role": "user",
            "content": (
                "[preloaded skills — apply when relevant]\n\n" + block
            ),
        })
        # And acknowledge as the assistant so the conversation alternates cleanly.
        self.messages.append({
            "role": "assistant",
            "content": "Acknowledged. I'll apply the relevant skills.",
        })

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._emit(ev.task_started(self._rid(), user_input))
        last_text = ""
        turn_usage = Usage()

        for _ in range(max_iterations):
            self._check_budget_and_cancel(turn_count=turn_usage.turns)

            response = self._stream_one()

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text

            self._emit(ev.turn_completed(
                self._rid(),
                {
                    "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                    "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                },
                response.stop_reason,
            ))

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

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score
        self._emit(ev.task_completed(self._rid(), last_text, critic_score))

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

    def _check_budget_and_cancel(self, *, turn_count: int) -> None:
        if self.event_sink is not None and hasattr(self.event_sink, "check_cancelled"):
            self.event_sink.check_cancelled()
        if self.budget is not None:
            elapsed = time.time() - self._start_time
            self.budget.check(self.usage, self.model, turn_count, elapsed)

    def _apply_critic_gate(self, prompt: str, response: str) -> tuple[str, float | None]:
        if self.critic is None:
            return response, None
        score = self.critic.predict_proba(prompt, response)
        if score < self.critic_threshold:
            warning = f"\n\n[critic confidence: {score:.2f} (< {self.critic_threshold}) — response may be unreliable]"
            if self.verbose:
                print(warning, flush=True)
            return response + warning, score
        return response, score

    def _system_prompt(self) -> list[dict]:
        text = SYSTEM_PROMPT
        if self.role_addendum:
            text = text + "\n\nRole-specific guidance:\n" + self.role_addendum
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def _stream_one(self):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt(),
            tools=self.tool_schemas,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
        ) as stream:
            if self.verbose:
                for event in stream:
                    self._handle_stream_event(event)
            else:
                for event in stream:
                    self._handle_stream_event_silent(event)
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
                self._emit(ev.thinking_delta(self._rid(), d.thinking))
            elif d.type == "text_delta":
                print(d.text, end="", flush=True)
                self._emit(ev.text_delta(self._rid(), d.text))

    def _handle_stream_event_silent(self, event) -> None:
        if event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                self._emit(ev.thinking_delta(self._rid(), d.thinking))
            elif d.type == "text_delta":
                self._emit(ev.text_delta(self._rid(), d.text))

    def _dispatch_tool_calls(self, content) -> list[dict]:
        tool_results: list[dict] = []
        for block in content:
            if block.type == "server_tool_use":
                self._emit(ev.tool_call(
                    self._rid(), block.name, block.id, dict(block.input or {}), server_side=True,
                ))
                continue
            if block.type != "tool_use":
                continue
            handler = self.handlers.get(block.name)
            self._emit(ev.tool_call(
                self._rid(), block.name, block.id, dict(block.input or {}), server_side=handler is None,
            ))
            if handler is None:
                continue
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            content_str = _stringify_tool_result(result)
            self._emit(ev.tool_result(self._rid(), block.id, content_str, is_error))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_str,
                    "is_error": is_error,
                }
            )
            # Post-tool budget check — fast tool that bursts cost can still trip caps.
            self._check_budget_and_cancel(turn_count=self.usage.turns)
        return tool_results

    def _emit(self, event) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink.emit(event)
        except Exception:
            # Never let event emission break the agent loop.
            pass

    def _rid(self) -> str:
        if self.event_sink is not None and hasattr(self.event_sink, "id"):
            return self.event_sink.id
        return "local"
