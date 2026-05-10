"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; tool-result dispatch skips them.

Lifecycle hooks let a coordination engine observe text/tool/usage events
in real time without subclassing the agent.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import anthropic

from agi.costs import Usage
from agi.memory import Memory
from agi.skills import SkillLibrary, render_skill_block
from agi.tools import ToolRegistry, make_tools

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


SYSTEM_PROMPT = """\
You are an agent built on Claude Opus 4.7. You have tools to read and write files,
run shell commands, search and fetch the web, manage a persistent long-term
memory, retrieve and author procedural skills, and synthesize new tools or
delegate to sub-agents on hard tasks.

Operating principles:
- Plan before acting on multi-step tasks. Decompose, then execute.
- Use tools instead of guessing. If you need a file's contents, read it. If you
  need a fact, search the web. If you remember something useful, save it.
- Use long-term memory deliberately. Save user preferences, project facts, and
  durable lessons learned. Search memory at the start of related tasks.
- Reach for skills before reinventing. If a relevant skill is loaded below, follow
  it. After solving a novel reusable problem, consider adding a skill.
- Verify before claiming success. Read back files you wrote. Run tests where
  applicable. State limitations honestly.
- Be terse. Skip preamble. Show the work, not the throat-clearing.
"""


@dataclass
class StepResult:
    """Compact summary of one chat turn — what the runtime exposes upstream."""
    text: str
    usage: Usage
    tool_calls: list[str]
    iterations: int
    stop_reason: str
    duration_seconds: float
    critic_score: float | None = None
    error: str | None = None


@dataclass
class Hooks:
    """Optional callbacks for lifecycle observation.

    All callbacks are best-effort. Exceptions inside a hook are swallowed
    with a printed warning — the agent loop must keep running for the
    coordination engine that depends on it.
    """
    on_text_delta: Callable[[str], None] | None = None
    on_thinking_delta: Callable[[str], None] | None = None
    on_tool_use: Callable[[str, dict], None] | None = None
    on_tool_result: Callable[[str, str, bool], None] | None = None
    on_iteration: Callable[[int, str], None] | None = None  # (iter, stop_reason)
    on_complete: Callable[["StepResult"], None] | None = None


def _stringify_tool_result(result) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


def _safe(callback, *args):
    if callback is None:
        return
    try:
        callback(*args)
    except Exception as e:  # noqa: BLE001
        print(f"[hook error: {type(e).__name__}: {e}]", flush=True)


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
        skills: SkillLibrary | None = None,
        skills_top_k: int = 3,
        reflector=None,
        hooks: Hooks | None = None,
        registry: ToolRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        client: anthropic.Anthropic | None = None,
        delegate_fn: Callable[..., str] | None = None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.memory = memory or Memory()
        self.skills = skills
        self.skills_top_k = skills_top_k
        self.reflector = reflector
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer  # optional TraceLogger for the learning loop
        self.critic = critic  # optional learner.Critic; gates final output
        self.critic_threshold = critic_threshold
        self.last_critic_score: float | None = None
        self.last_step: StepResult | None = None
        self.messages: list[dict] = []
        self.usage = Usage()
        self.system_prompt = system_prompt
        self.hooks = hooks or Hooks()

        self.registry = registry or make_tools(
            self.memory,
            skills=skills,
            delegate_fn=delegate_fn,
        )

        self.server_tools: list[dict] = []
        if enable_web_search:
            self.server_tools.append(
                {"type": "web_search_20260209", "name": "web_search"}
            )
        if enable_web_fetch:
            self.server_tools.append(
                {"type": "web_fetch_20260209", "name": "web_fetch"}
            )

    # ---- public state ----

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()
        self.registry.call_log.clear()

    def snapshot(self) -> dict:
        """Lightweight serializable view of the agent's current state.

        The coordination engine reads this between steps to decide what
        to do next without parsing the full message log.
        """
        return {
            "model": self.model,
            "messages": len(self.messages),
            "tools_called_session": list(self.registry.call_log),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "turns": self.usage.turns,
                "cost_usd": self.usage.cost_usd(self.model),
            },
            "last_critic_score": self.last_critic_score,
            "registered_tools": self.registry.names(),
        }

    # ---- the loop ----

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        result = self.step(user_input, max_iterations=max_iterations)
        return result.text

    def step(self, user_input: str, max_iterations: int = 25) -> StepResult:
        """One user-facing turn. Returns a StepResult with text + telemetry."""
        t0 = time.time()
        self.messages.append({"role": "user", "content": user_input})

        last_text = ""
        turn_usage = Usage()
        turn_tool_calls: list[str] = []
        iterations = 0
        stop_reason = "end_turn"
        error: str | None = None

        try:
            for i in range(max_iterations):
                iterations = i + 1
                response = self._stream_one(user_input)
                self.messages.append({"role": "assistant", "content": response.content})
                self.usage.add(response.usage)
                turn_usage.add(response.usage)

                for block in response.content:
                    if block.type == "text" and block.text:
                        last_text = block.text

                stop_reason = response.stop_reason
                _safe(self.hooks.on_iteration, iterations, stop_reason)

                if stop_reason == "end_turn":
                    break
                if stop_reason == "pause_turn":
                    continue
                if stop_reason == "tool_use":
                    tool_results = self._dispatch_tool_calls(
                        response.content, turn_tool_calls
                    )
                    if not tool_results:
                        break
                    self.messages.append({"role": "user", "content": tool_results})
                    continue
                # refusal, max_tokens, stop_sequence, model_context_window_exceeded
                break
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            stop_reason = "error"

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        if self.tracer is not None:
            metadata: dict = {
                "tool_calls": turn_tool_calls,
                "iterations": iterations,
                "stop_reason": stop_reason,
            }
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            if error:
                metadata["error"] = error
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

        # Reflect: best-effort, never blocks the response, never raises.
        if self.reflector is not None and not error:
            try:
                self.reflector.reflect(
                    user_prompt=user_input,
                    final_text=last_text,
                    tools_used=turn_tool_calls,
                )
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[reflect failed: {type(e).__name__}: {e}]", flush=True)

        result = StepResult(
            text=last_text,
            usage=turn_usage,
            tool_calls=turn_tool_calls,
            iterations=iterations,
            stop_reason=stop_reason,
            duration_seconds=time.time() - t0,
            critic_score=critic_score,
            error=error,
        )
        self.last_step = result
        _safe(self.hooks.on_complete, result)
        return result

    # ---- critic gate ----

    def _apply_critic_gate(self, prompt: str, response: str) -> tuple[str, float | None]:
        if self.critic is None:
            return response, None
        score = self.critic.predict_proba(prompt, response)
        if score < self.critic_threshold:
            warning = (
                f"\n\n[critic confidence: {score:.2f} (< {self.critic_threshold}) — "
                "response may be unreliable]"
            )
            if self.verbose:
                print(warning, flush=True)
            return response + warning, score
        return response, score

    # ---- streaming ----

    def _stream_one(self, current_user_input: str):
        system_blocks = self._build_system_blocks(current_user_input)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            tools=self.registry.schemas + self.server_tools,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
        ) as stream:
            if self.verbose or self.hooks.on_text_delta or self.hooks.on_thinking_delta:
                for event in stream:
                    self._handle_stream_event(event)
            else:
                for _ in stream:
                    pass
            return stream.get_final_message()

    def _build_system_blocks(self, user_input: str) -> list[dict]:
        blocks: list[dict] = [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.skills is not None:
            relevant = self.skills.search(user_input, k=self.skills_top_k)
            block = render_skill_block(relevant)
            if block:
                blocks.append({"type": "text", "text": block})
        return blocks

    def _handle_stream_event(self, event) -> None:
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                if self.verbose:
                    print("\n[thinking] ", end="", flush=True)
            elif block.type == "tool_use":
                if self.verbose:
                    print(f"\n[tool: {block.name}]", end="", flush=True)
                _safe(self.hooks.on_tool_use, block.name, getattr(block, "input", {}) or {})
            elif block.type == "server_tool_use":
                if self.verbose:
                    print(f"\n[server: {block.name}]", end="", flush=True)
                _safe(self.hooks.on_tool_use, block.name, getattr(block, "input", {}) or {})
            elif block.type == "text":
                if self.verbose:
                    print()
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                if self.verbose:
                    print(d.thinking, end="", flush=True)
                _safe(self.hooks.on_thinking_delta, d.thinking)
            elif d.type == "text_delta":
                if self.verbose:
                    print(d.text, end="", flush=True)
                _safe(self.hooks.on_text_delta, d.text)

    def _dispatch_tool_calls(self, content, tool_calls_log: list[str]) -> list[dict]:
        tool_results: list[dict] = []
        for block in content:
            if block.type != "tool_use":
                continue
            if block.name not in self.registry.handlers:
                # web_search / web_fetch — server-side, already inlined.
                continue
            tool_calls_log.append(block.name)
            result, is_error = self.registry.dispatch(block.name, block.input or {})
            _safe(self.hooks.on_tool_result, block.name, result, is_error)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _stringify_tool_result(result),
                    "is_error": is_error,
                }
            )
        return tool_results
