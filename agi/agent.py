"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

Runtime hooks (all optional, default off):
  - skills:        SkillLibrary; top-K injected per turn into the system prompt
  - budget:        Budget; checked before each model call, raises BudgetExceeded
  - event_bus:     EventBus; emits turn/tool/text/skill/critic events
  - system_extra:  string appended to SYSTEM_PROMPT for role specialization
  - session_id:    propagated on every emitted event for routing
"""
from __future__ import annotations

import json
from typing import Callable

import anthropic

from agi.budget import Budget, BudgetExceeded
from agi.costs import Usage
from agi.events import EventBus
from agi.memory import Memory
from agi.skills import SkillLibrary
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
        skills: SkillLibrary | None = None,
        budget: Budget | None = None,
        event_bus: EventBus | None = None,
        system_extra: str = "",
        session_id: str | None = None,
        client=None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.skills = skills
        self.budget = budget
        self.bus = event_bus
        self.system_extra = system_extra
        self.session_id = session_id

        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self.usage = Usage()

        schemas, handlers = make_tools(self.memory, skills=skills)
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
        self.last_critic_score = None

    # ---- internal helpers ----

    def _emit(self, type_: str, **fields) -> None:
        if self.bus is None:
            return
        if self.session_id is not None and "session_id" not in fields:
            fields["session_id"] = self.session_id
        self.bus.emit(type_, **fields)

    def _build_system(self, user_input: str) -> list[dict]:
        """Compose the system prompt: base + role extra + relevant skills.

        Only the static base block is cache-marked. The dynamic skill block
        changes per task and would just thrash the cache if marked.
        """
        blocks: list[dict] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.system_extra:
            blocks.append({"type": "text", "text": self.system_extra})
        if self.skills is not None:
            skills_section = self.skills.render_for_prompt(user_input, k=3)
            if skills_section:
                blocks.append({"type": "text", "text": skills_section})
                # tell the bus which skills were loaded
                for s in self.skills.search(user_input, k=3):
                    self._emit("skill_loaded", name=s.name, description=s.description)
        return blocks

    # ---- public API ----

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()
        system = self._build_system(user_input)

        self._emit(
            "turn_started",
            input_preview=user_input[:200],
            iteration_cap=max_iterations,
        )

        try:
            for _ in range(max_iterations):
                # Budget check BEFORE each model call so we never blow past the cap.
                if self.budget is not None:
                    self.budget.check(self.usage)

                response = self._stream_one(system)

                self.messages.append({"role": "assistant", "content": response.content})
                self.usage.add(response.usage)
                turn_usage.add(response.usage)

                for block in response.content:
                    if block.type == "text" and block.text:
                        last_text = block.text
                        self._emit("text", text=block.text)
                    elif block.type == "tool_use":
                        self._emit(
                            "tool_call",
                            name=block.name,
                            input=block.input,
                            tool_use_id=block.id,
                        )

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

                # refusal, max_tokens, stop_sequence, model_context_window_exceeded, ...
                break

        except BudgetExceeded as exc:
            self._emit("budget_warning", reason=exc.reason)
            raise

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score
        if critic_score is not None:
            self._emit("critic_score", score=critic_score)

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        self._emit(
            "turn_finished",
            text_preview=last_text[:200],
            cost_usd=turn_usage.cost_usd(self.model),
            input_tokens=turn_usage.input_tokens,
            output_tokens=turn_usage.output_tokens,
        )

        if self.tracer is not None:
            metadata: dict = {}
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            if self.session_id is not None:
                metadata["session_id"] = self.session_id
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

    def _stream_one(self, system: list[dict]):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
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
                self._emit("server_tool", name=block.name)
            elif block.type == "text":
                print()
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                print(d.thinking, end="", flush=True)
                self._emit("thinking", delta=d.thinking)
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
            self._emit(
                "tool_result",
                tool_use_id=block.id,
                name=block.name,
                is_error=is_error,
                preview=_stringify_tool_result(result)[:300],
            )
        return tool_results
