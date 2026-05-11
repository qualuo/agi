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
        world_model=None,
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
        self.world_model = world_model  # optional WorldModel for entity tracking
        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self.usage = Usage()

        # Runtime-injected attachments. Set by Runtime via attach_*().
        self.extra_system: str | None = None
        self._tool_synth = None  # ToolSynthRegistry
        self._tool_synth_bus = None
        self._tool_synth_session_id = None
        self._delegate_fn = None  # Callable[[task, role, model?], str]
        self._delegate_bus = None
        self._delegate_session_id = None

        schemas, handlers = make_tools(self.memory, world_model=self.world_model)
        self._builtin_tool_schemas: list[dict] = list(schemas)
        self._builtin_handlers: dict[str, Callable[..., str]] = dict(handlers)

        if enable_web_search:
            self._builtin_tool_schemas.append(
                {"type": "web_search_20260209", "name": "web_search"}
            )
        if enable_web_fetch:
            self._builtin_tool_schemas.append(
                {"type": "web_fetch_20260209", "name": "web_fetch"}
            )

        # tool_schemas/handlers are recomputed each turn so attached
        # synthesizers and delegation can add tools on the fly.
        self.tool_schemas: list[dict] = list(self._builtin_tool_schemas)
        self.handlers: dict[str, Callable[..., str]] = dict(self._builtin_handlers)

    def attach_tool_synth(self, registry, bus=None, session_id: str | None = None) -> None:
        """Wire a ToolSynthRegistry so the agent can `make_tool` at runtime."""
        self._tool_synth = registry
        self._tool_synth_bus = bus
        self._tool_synth_session_id = session_id

    def attach_delegation(self, delegate_fn: Callable[..., str], bus=None, session_id: str | None = None) -> None:
        """Wire a delegation function so the agent can spawn subagents."""
        self._delegate_fn = delegate_fn
        self._delegate_bus = bus
        self._delegate_session_id = session_id

    def _refresh_tools(self) -> None:
        """Rebuild tool_schemas and handlers from builtins + attachments.

        Called between turns so newly synthesized or just-attached tools
        become visible on the next API request without rebuilding the Agent.
        """
        schemas: list[dict] = list(self._builtin_tool_schemas)
        handlers: dict[str, Callable[..., str]] = dict(self._builtin_handlers)

        if self._tool_synth is not None:
            from agi.toolsynth import ToolSynthError  # local to avoid cycles
            # the make_tool meta-tool itself
            def _make_tool(name: str, description: str, code: str,
                           input_schema: dict | None = None,
                           smoke_test_kwargs: dict | None = None) -> str:
                try:
                    tool = self._tool_synth.register(
                        name=name,
                        description=description,
                        code=code,
                        input_schema=input_schema,
                        smoke_test_kwargs=smoke_test_kwargs,
                    )
                except ToolSynthError as e:
                    return f"error: {e}"
                if self._tool_synth_bus is not None:
                    from agi.events import Event, TOOL_SYNTHESIZED
                    self._tool_synth_bus.publish(Event(
                        kind=TOOL_SYNTHESIZED,
                        session_id=self._tool_synth_session_id,
                        data={"name": tool.name, "description": tool.description},
                    ))
                return f"registered tool {name!r}; it will be callable on the next turn"

            schemas.append({
                "name": "make_tool",
                "description": (
                    "Synthesize a new Python tool for this session. The "
                    "code must define `run(**kwargs) -> str` at module top. "
                    "Imports of os/subprocess/socket/shutil are banned; "
                    "eval/exec/compile/open are banned. Code runs in a "
                    "sandboxed subprocess on each call."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Snake_case identifier."},
                        "description": {"type": "string"},
                        "code": {"type": "string", "description": "Python source defining run(**kwargs) -> str."},
                        "input_schema": {"type": "object", "description": "JSON Schema for the kwargs the tool accepts."},
                        "smoke_test_kwargs": {"type": "object", "description": "Sample kwargs for the registration-time smoke test."},
                    },
                    "required": ["name", "description", "code"],
                },
            })
            handlers["make_tool"] = _make_tool

            for synth_schema in self._tool_synth.schemas():
                schemas.append(synth_schema)
            for name, fn in self._tool_synth.handlers().items():
                handlers[name] = fn

        if self._delegate_fn is not None:
            def _delegate(task: str, role: str, model: str | None = None) -> str:
                try:
                    return self._delegate_fn(task=task, role=role, model=model)
                except Exception as e:
                    return f"error: {type(e).__name__}: {e}"

            schemas.append({
                "name": "delegate",
                "description": (
                    "Spawn a specialist subagent to solve a self-contained "
                    "subtask. Returns the subagent's final answer. Use for "
                    "decomposable problems; costs roll up into this session."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The subtask, written as a standalone instruction."},
                        "role": {"type": "string", "description": "Role hint, e.g. 'planner', 'researcher', 'executor', 'critic'."},
                        "model": {"type": "string", "description": "Optional model override (e.g. 'claude-haiku-4-5' for cheap subtasks)."},
                    },
                    "required": ["task", "role"],
                },
            })
            handlers["delegate"] = _delegate

        self.tool_schemas = schemas
        self.handlers = handlers

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        # Recompute tools so newly attached/synthesized tools are visible
        # this turn (when wired from a Runtime).
        self._refresh_tools()

        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()

        for _ in range(max_iterations):
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
                # Newly synthesized tools should be visible next turn.
                self._refresh_tools()
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

    def _system_blocks(self) -> list[dict]:
        blocks = [
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
