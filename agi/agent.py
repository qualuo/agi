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
import time
from typing import Callable

import anthropic

from agi.budget import Budget
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
        budget: Budget | None = None,
        skills=None,
        system_prompt_extra: str | None = None,
        on_event: Callable[[dict], None] | None = None,
        extra_tools: tuple[list[dict], dict[str, Callable[..., str]]] | None = None,
        client=None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer  # optional TraceLogger for the learning loop
        self.critic = critic  # optional learner.Critic; gates final output
        self.critic_threshold = critic_threshold
        self.budget = budget
        self.skills = skills  # optional SkillLibrary
        self.system_prompt_extra = system_prompt_extra
        self.on_event = on_event
        self.last_critic_score: float | None = None
        self.last_stop_reason: str | None = None
        self.last_skills_used: list[str] = []
        self.messages: list[dict] = []
        self.usage = Usage()

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = handlers

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
        self.last_stop_reason = None
        self.last_skills_used = []

    def chat(
        self,
        user_input: str,
        max_iterations: int = 25,
        budget: Budget | None = None,
    ) -> str:
        skills_block = self._retrieve_skills(user_input)
        self.last_skills_used = [s.id for s in skills_block]
        self.messages.append({"role": "user", "content": user_input})

        effective_budget = self.budget.merged_with(budget) if self.budget else budget
        budget_capped_iterations = False
        if effective_budget and effective_budget.max_iterations is not None:
            if effective_budget.max_iterations < max_iterations:
                max_iterations = effective_budget.max_iterations
                budget_capped_iterations = True

        last_text = ""
        turn_usage = Usage()
        started_at = time.time()
        iterations = 0
        stop_reason: str | None = None

        for _ in range(max_iterations):
            if effective_budget is not None:
                over = effective_budget.check(
                    usage=turn_usage,
                    model=self.model,
                    started_at=started_at,
                    iterations=iterations,
                )
                if over is not None:
                    stop_reason = f"over_budget:{over}"
                    self._emit({"type": "budget", "reason": over})
                    break

            response = self._stream_one(skills_block)
            iterations += 1

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)
            self._emit({
                "type": "turn",
                "iteration": iterations,
                "stop_reason": response.stop_reason,
                "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
            })

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text

            if response.stop_reason == "end_turn":
                stop_reason = "end_turn"
                break

            if response.stop_reason == "pause_turn":
                # Server-side tool hit its iteration limit; re-send to continue.
                continue

            if response.stop_reason == "tool_use":
                tool_results = self._dispatch_tool_calls(response.content)
                if not tool_results:
                    stop_reason = "tool_use_empty"
                    break
                self.messages.append({"role": "user", "content": tool_results})
                continue

            stop_reason = response.stop_reason
            break

        else:
            stop_reason = "over_budget:max_iterations" if budget_capped_iterations else "max_iterations"

        self.last_stop_reason = stop_reason
        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        if self.tracer is not None:
            metadata: dict = {}
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            if self.last_skills_used:
                metadata["skills_used"] = list(self.last_skills_used)
            if self.last_stop_reason:
                metadata["stop_reason"] = self.last_stop_reason
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

    def _retrieve_skills(self, prompt: str):
        if self.skills is None:
            return []
        try:
            return self.skills.match(prompt, k=3, promoted_only=True)
        except Exception:  # never let skill retrieval break a turn
            return []

    def _build_system(self, skills) -> list[dict]:
        # System prompt blocks (cached separately so skills don't bust the cache
        # of the core prompt). Order: stable base → user-added → matched skills.
        blocks: list[dict] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.system_prompt_extra:
            blocks.append({"type": "text", "text": self.system_prompt_extra})
        if skills:
            skills_text = "## Relevant skills\n\n" + "\n\n".join(
                s.render_for_prompt() for s in skills
            )
            blocks.append({"type": "text", "text": skills_text})
        return blocks

    def _stream_one(self, skills=None):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._build_system(skills or []),
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

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            pass

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
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _stringify_tool_result(result),
                    "is_error": is_error,
                }
            )
        return tool_results
