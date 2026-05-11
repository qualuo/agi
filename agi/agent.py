"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

The agent can also be wired with optional runtime components:
- `event_bus` → structured events for a coordination engine to subscribe to.
- `skills` → procedural-memory library; relevant skills loaded into the prompt.
- `synth` → registry for tools the agent defines at runtime.
- `delegate_fn` → callback to spawn a subagent in a named role.
- `budget` → per-task usage/time ceiling; tripping it ends the loop.
- `reflect` → bool, run a small reflection pass after each task.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from agi.budget import Budget
from agi.costs import Usage
from agi.events import EventBus
from agi.memory import Memory
from agi.reflect import REFLECT_PROMPT, parse_lesson, should_reflect
from agi.skills import SkillLibrary
from agi.synth_registry import SynthToolRegistry
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

Self-extension (when available):
- Use `define_tool` when you need a deterministic helper that's worth reusing
  (parsing, math, formatting). Promote with `promote_tool` only after testing.
- Use `add_skill` when you've solved a non-trivial task and the procedure
  would speed up the next instance.
- Use `delegate` to hand off well-scoped subtasks to a cheaper specialist
  (planner, executor, critic, researcher, coder, summarizer).
"""


def _stringify_tool_result(result) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


def _build_system_prompt(skills: SkillLibrary | None, user_query: str | None) -> list[dict]:
    """Build the system prompt blocks. The base prompt is prompt-cached;
    relevant skills are appended as a non-cached block so changes take effect
    immediately when new skills are added."""
    blocks: list[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    if skills is not None and user_query:
        relevant = skills.search(user_query, k=3)
        if relevant:
            joined = "\n\n".join(s.to_prompt_block() for s in relevant)
            blocks.append({
                "type": "text",
                "text": f"Relevant skills retrieved for this task:\n\n{joined}",
            })
            for s in relevant:
                skills.mark_used(s.name)
    return blocks


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
        # --- runtime integration --------------------------------------------
        event_bus: EventBus | None = None,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        skills: SkillLibrary | None = None,
        synth: SynthToolRegistry | None = None,
        delegate_fn: Callable[[str, str, dict | None], str] | None = None,
        budget: Budget | None = None,
        system_prompt_override: str | None = None,
        reflect: bool = False,
        cancel_event=None,  # threading.Event-like, .is_set() ends the loop
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
        self.event_bus = event_bus
        self.task_id = task_id
        self.parent_task_id = parent_task_id
        self.skills = skills
        self.synth = synth
        self.delegate_fn = delegate_fn
        self.budget = budget
        self.system_prompt_override = system_prompt_override
        self.reflect = reflect
        self.cancel_event = cancel_event

        schemas, handlers = make_tools(
            self.memory,
            skills=self.skills,
            synth=self.synth,
            delegate_fn=self.delegate_fn,
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

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, **data: Any) -> None:
        if self.event_bus is None or self.task_id is None:
            return
        self.event_bus.emit(kind, self.task_id, parent_task_id=self.parent_task_id, **data)

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        if self.budget is not None:
            self.budget.reset_clock()
        self._emit("task.started", prompt=user_input, model=self.model)

        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()
        n_tool_calls = 0
        n_iterations = 0
        # Refresh synthesized tools at the start of each turn so newly-defined
        # tools become callable on the next API call.
        self._refresh_synth_tools()
        # Snapshot relevant skills at task start.
        system_blocks = _build_system_prompt(self.skills, user_input)
        if self.skills is not None:
            for s in self.skills.search(user_input, k=3):
                self._emit("task.skill_loaded", name=s.name)

        stop_reason: str | None = None
        for _ in range(max_iterations):
            if self.cancel_event is not None and self.cancel_event.is_set():
                self._emit("task.cancelled", reason="cancel_event set")
                break

            n_iterations += 1
            response = self._stream_one(system_blocks=system_blocks)

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)
            self._emit(
                "task.usage",
                input_tokens=turn_usage.input_tokens,
                output_tokens=turn_usage.output_tokens,
                cache_read=turn_usage.cache_read_input_tokens,
                cache_write=turn_usage.cache_creation_input_tokens,
                cost_usd=turn_usage.cost_usd(self.model),
            )

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text
                    self._emit("task.text", text=block.text)
                elif block.type == "tool_use":
                    self._emit("task.tool_call", name=block.name, input=block.input)
                    n_tool_calls += 1

            stop_reason = response.stop_reason

            # Budget gate between turns
            if self.budget is not None:
                reason = self.budget.check(turn_usage, self.model)
                if reason:
                    self._emit("task.budget_exceeded", reason=reason)
                    break

            if stop_reason == "end_turn":
                break

            if stop_reason == "pause_turn":
                continue

            if stop_reason == "tool_use":
                tool_results = self._dispatch_tool_calls(response.content)
                if not tool_results:
                    break
                self.messages.append({"role": "user", "content": tool_results})
                # Synthesized tools may have just been defined; refresh schema list
                self._refresh_synth_tools()
                continue

            break  # refusal / max_tokens / stop_sequence / context_window

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score
        if critic_score is not None:
            self._emit("task.critic_score", score=critic_score, threshold=self.critic_threshold)

        if self.reflect and should_reflect(
            n_turns=n_iterations,
            n_tool_calls=n_tool_calls,
            response_chars=len(last_text or ""),
        ):
            lesson = self._run_reflection(user_input, last_text)
            if lesson:
                self.memory.save(lesson, tags=["lesson"])
                self._emit("task.reflection", lesson=lesson)

        if self.verbose:
            print(f"\n[{turn_usage.format(self.model)}]", flush=True)

        if self.tracer is not None:
            metadata: dict = {}
            if critic_score is not None:
                metadata["critic_score"] = critic_score
            if self.task_id:
                metadata["task_id"] = self.task_id
            if self.parent_task_id:
                metadata["parent_task_id"] = self.parent_task_id
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

        self._emit(
            "task.completed",
            final_text=last_text,
            n_iterations=n_iterations,
            n_tool_calls=n_tool_calls,
            cost_usd=turn_usage.cost_usd(self.model),
            stop_reason=stop_reason,
        )
        return last_text

    # ------------------------------------------------------------- reflection
    def _run_reflection(self, prompt: str, response: str) -> str | None:
        """One cheap turn against the model asking for a durable lesson."""
        try:
            r = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=200,
                system=[{"type": "text", "text": REFLECT_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": f"TASK:\n{prompt}\n\nRESPONSE:\n{response}",
                    }
                ],
            )
            text_parts = [b.text for b in r.content if b.type == "text"]
            return parse_lesson("\n".join(text_parts))
        except Exception:
            # Reflection is best-effort; never let it kill the task.
            return None

    # --------------------------------------------------------- critic / synth
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

    def _refresh_synth_tools(self) -> None:
        """Sync synthesized tools into the live tool_schemas/handlers."""
        if self.synth is None:
            return
        # Drop any prior synth tool schemas; we re-add fresh.
        synth_names = set(self.synth.all().keys())
        builtin_names = {"read_file", "write_file", "list_dir", "run_bash",
                         "save_memory", "search_memory", "recent_memory",
                         "list_skills", "add_skill",
                         "define_tool", "list_synth_tools", "promote_tool", "call_synth",
                         "delegate"}
        # Keep schemas that aren't synth-tool schemas; keep server-side typed ones.
        new_schemas: list[dict] = []
        for s in self.tool_schemas:
            name = s.get("name")
            if "type" in s and s.get("type", "").startswith(("web_search", "web_fetch")):
                new_schemas.append(s)
                continue
            if name in builtin_names:
                new_schemas.append(s)
        for name, tool in self.synth.all().items():
            new_schemas.append({
                "name": name,
                "description": f"[synthesized] {tool.description}",
                "input_schema": tool.input_schema or {"type": "object", "properties": {}},
            })
            self.handlers[name] = lambda _t=tool, **kw: _stringify_synth_result(_t.func(**kw))
        self.tool_schemas = new_schemas

    # ------------------------------------------------------------ streaming
    def _stream_one(self, *, system_blocks: list[dict] | None = None):
        sys_blocks = system_blocks
        if sys_blocks is None:
            if self.system_prompt_override is not None:
                sys_blocks = [{"type": "text", "text": self.system_prompt_override,
                               "cache_control": {"type": "ephemeral"}}]
            else:
                sys_blocks = [{"type": "text", "text": SYSTEM_PROMPT,
                               "cache_control": {"type": "ephemeral"}}]

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=sys_blocks,
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
                self._emit("task.thinking", started=True)
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

    def _dispatch_tool_calls(self, content) -> list[dict]:
        tool_results: list[dict] = []
        for block in content:
            if block.type != "tool_use":
                continue
            handler = self.handlers.get(block.name)
            if handler is None:
                continue
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            self._emit(
                "task.tool_result",
                name=block.name,
                is_error=is_error,
                result_preview=str(result)[:240],
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


def _stringify_synth_result(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        return repr(value)
