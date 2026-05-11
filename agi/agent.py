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
from agi.events import Event, EventBus, stdout_printer
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
        events: EventBus | None = None,
        skills: "SkillLibrary | None" = None,
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
        self.events = events or EventBus()
        self.skills = skills
        self.session_id = session_id
        self._cancelled = False
        if verbose:
            # The REPL/CLI keep their stdout behavior by default.
            self.events.subscribe(stdout_printer)

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
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation. Checked between turns; honoured at the
        next loop boundary. In-flight streaming responses still complete
        their current turn — clean cancellation mid-stream is not yet wired."""
        self._cancelled = True

    def chat(self, user_input: str, max_iterations: int = 25, run_id: str | None = None) -> str:
        self._cancelled = False
        self._emit("run.started", run_id=run_id, prompt_chars=len(user_input))
        self._inject_skill_hints(user_input)
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()
        stop_reason: str | None = None
        cancelled = False

        for _ in range(max_iterations):
            if self._cancelled:
                cancelled = True
                self._emit("cancelled", run_id=run_id)
                break

            self._emit("turn.started", run_id=run_id)
            try:
                response = self._stream_one()
            except Exception as e:
                self._emit("error", run_id=run_id, message=f"{type(e).__name__}: {e}")
                raise

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)
            self._emit(
                "turn.finished",
                run_id=run_id,
                stop_reason=response.stop_reason,
                usage_formatted=Usage(
                    input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                    cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                ).format(self.model),
            )

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text

            stop_reason = response.stop_reason

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "pause_turn":
                # Server-side tool hit its iteration limit; re-send to continue.
                continue

            if response.stop_reason == "tool_use":
                tool_results = self._dispatch_tool_calls(response.content, run_id=run_id)
                if not tool_results:
                    break
                self.messages.append({"role": "user", "content": tool_results})
                continue

            # refusal, max_tokens, stop_sequence, model_context_window_exceeded, ...
            break

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score
        if critic_score is not None:
            self._emit(
                "critic.scored",
                run_id=run_id,
                score=critic_score,
                threshold=self.critic_threshold,
            )

        self._emit(
            "run.finished",
            run_id=run_id,
            stop_reason=stop_reason,
            cancelled=cancelled,
            turn_usage={
                "input_tokens": turn_usage.input_tokens,
                "output_tokens": turn_usage.output_tokens,
                "cache_creation_input_tokens": turn_usage.cache_creation_input_tokens,
                "cache_read_input_tokens": turn_usage.cache_read_input_tokens,
                "cost_usd": turn_usage.cost_usd(self.model),
            },
            response_chars=len(last_text),
        )

        if self.tracer is not None:
            metadata: dict = {}
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            if cancelled:
                metadata["cancelled"] = True
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

    def _emit(self, kind: str, **data) -> None:
        evt = Event(kind=kind, data=data, session_id=self.session_id, run_id=data.get("run_id"))
        self.events.publish(evt)

    def _inject_skill_hints(self, user_input: str) -> None:
        """If a SkillLibrary is wired, inject the top matching skill bodies
        as a system-style hint message before the user input."""
        if self.skills is None:
            return
        hits = self.skills.retrieve(user_input, k=2)
        if not hits:
            return
        body = "\n\n".join(f"### Skill: {s.name}\n{s.body}" for s in hits)
        hint = (
            "The runtime has surfaced procedural skills that may apply to this task. "
            "Use them as guidance; you are not required to follow them verbatim.\n\n"
            + body
        )
        self.messages.append({"role": "user", "content": hint})
        self._emit("skills.injected", names=[s.name for s in hits])

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
            for event in stream:
                self._handle_stream_event(event)
            return stream.get_final_message()

    def _handle_stream_event(self, event) -> None:
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                self._emit("thinking.started")
            elif block.type == "tool_use":
                self._emit("tool.requested", name=block.name, id=getattr(block, "id", None))
            elif block.type == "server_tool_use":
                self._emit("server_tool.requested", name=block.name)
            elif block.type == "text":
                self._emit("text.started")
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                self._emit("thinking.delta", text=d.thinking)
            elif d.type == "text_delta":
                self._emit("text.delta", text=d.text)
            # input_json_delta is too noisy to surface during streaming

    def _dispatch_tool_calls(self, content, run_id: str | None = None) -> list[dict]:
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
            self._emit(
                "tool.result",
                run_id=run_id,
                name=block.name,
                id=block.id,
                is_error=is_error,
                result_preview=_stringify_tool_result(result)[:200],
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
