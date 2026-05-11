"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

The agent emits structured Events via `on_event` so a runtime can stream
thinking/tool-use/text/usage in real time without parsing the SDK objects.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

import anthropic

from agi.costs import Usage
from agi.events import (
    Event, make,
    THINKING, TEXT_DELTA, TEXT, TOOL_CALL, TOOL_RESULT,
)
from agi.memory import Memory
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


SYSTEM_PROMPT = """\
You are an agent built on Claude Opus 4.7. You have tools to read and write files,
run shell commands, search and fetch the web, manage a persistent long-term
memory that survives across sessions, delegate sub-tasks to focused sub-agents,
and forge new tools at runtime for problems your current toolbox can't handle.

Operating principles:
- Plan before acting on multi-step tasks. Decompose, then execute.
- Use tools instead of guessing. If you need a file's contents, read it. If you
  need a fact, search the web. If you remember something useful, save it.
- Use long-term memory deliberately. Save user preferences, project facts, and
  durable lessons learned. Search memory at the start of related tasks.
- Delegate when subtasks are independent or specialized — a sub-agent gets a
  fresh context window and a focused brief.
- Forge a new tool when a needed primitive is missing. Smoke-test before use.
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
        on_event: "Callable[[Event], None] | None" = None,
        runtime: Any = None,
        run_id: str | None = None,
        extra_system_prompt: str = "",
        enable_delegate: bool = True,
        enable_make_tool: bool = True,
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
        self.runtime = runtime
        self.run_id = run_id or "local"
        self.extra_system_prompt = extra_system_prompt
        self.messages: list[dict] = []
        self.usage = Usage()
        self._cancel = threading.Event()

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

        # Composable feature tools that need the agent in scope.
        if enable_delegate and runtime is not None:
            self._install_delegate_tool()
        if enable_make_tool:
            self._install_make_tool_tool()

    # --- agent lifecycle -----------------------------------------------

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def cancel(self) -> None:
        """Cooperatively stop the current chat loop at the next safe point."""
        self._cancel.set()

    # --- inner-loop emission helper ------------------------------------

    def _emit(self, type_: str, **data: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(make(type_, self.run_id, **data))
        except Exception:
            # Event-bus failures must never break the agent loop.
            pass

    # --- main chat loop ------------------------------------------------

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()

        for _ in range(max_iterations):
            if self._cancel.is_set() or self._runtime_cancelled():
                break

            response = self._stream_one()

            self.messages.append({"role": "assistant", "content": response.content})
            self.usage.add(response.usage)
            turn_usage.add(response.usage)

            for block in response.content:
                if block.type == "text" and block.text:
                    last_text = block.text
                    self._emit(TEXT, text=block.text)

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
            metadata: dict = {"run_id": self.run_id}
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

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        """One-shot non-tool LLM call. Used by the reflector and other helpers.

        Bypasses the conversation history and tool surface so it can run
        cheaply alongside an in-flight chat.
        """
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.usage.add(msg.usage)
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    # --- critic gate ---------------------------------------------------

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

    # --- streaming helpers ---------------------------------------------

    def _runtime_cancelled(self) -> bool:
        if self.runtime is None or self.run_id is None:
            return False
        handle = getattr(self.runtime, "get", lambda _id: None)(self.run_id)
        return bool(handle and handle.is_cancelled())

    def _stream_one(self):
        system_blocks = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.extra_system_prompt:
            system_blocks.append({"type": "text", "text": self.extra_system_prompt})

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
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
                self._emit(THINKING, text=d.thinking)
            elif d.type == "text_delta":
                print(d.text, end="", flush=True)
                self._emit(TEXT_DELTA, text=d.text)
            # input_json_delta is too noisy to show during streaming

    def _handle_stream_event_silent(self, event) -> None:
        if event.type == "content_block_delta":
            d = event.delta
            if d.type == "thinking_delta":
                self._emit(THINKING, text=d.thinking)
            elif d.type == "text_delta":
                self._emit(TEXT_DELTA, text=d.text)

    # --- tool dispatch -------------------------------------------------

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
            self._emit(TOOL_CALL, name=block.name, input=block.input or {}, id=block.id)
            try:
                result = handler(**(block.input or {}))
                is_error = False
            except Exception as e:  # tool failures are reported back to the model
                result = f"error: {type(e).__name__}: {e}"
                is_error = True
            content_str = _stringify_tool_result(result)
            self._emit(TOOL_RESULT, id=block.id, content=content_str[:2000], is_error=is_error)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_str,
                    "is_error": is_error,
                }
            )
        return tool_results

    # --- composable feature tools --------------------------------------

    def _install_delegate_tool(self) -> None:
        """Add a `delegate(task, role?)` tool that spawns a sub-run on the runtime."""
        runtime = self.runtime

        def delegate(task: str, role: str = "executor", max_iterations: int = 15) -> str:
            from agi.runtime import RunRequest  # local import to avoid cycles
            parent = runtime.get(self.run_id) if self.run_id else None
            req = RunRequest(
                task=task,
                skills=True,
                reflect=False,
                max_iterations=max_iterations,
                metadata={"role": role},
            )
            if parent is not None:
                child = runtime.submit_child(parent, req)
            else:
                child = runtime.submit(req)
            child.wait()
            r = child.result
            if r is None:
                return "delegate: no result"
            if r.error:
                return f"delegate failed: {r.error}"
            return (
                f"sub-run {child.run_id} ({role})\n"
                f"cost: ${r.cost_usd:.4f}\n"
                f"---\n{r.text}"
            )

        self.tool_schemas.append({
            "name": "delegate",
            "description": (
                "Spawn a focused sub-agent on a sub-task. Gets a fresh context, "
                "the shared skill library, and rolls its usage up to the parent."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What the sub-agent should do."},
                    "role": {
                        "type": "string",
                        "description": "Hint at sub-agent's role (planner/executor/critic/researcher).",
                        "default": "executor",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Hard cap on sub-agent tool-use turns.",
                        "default": 15,
                    },
                },
                "required": ["task"],
            },
        })
        self.handlers["delegate"] = delegate

    def _install_make_tool_tool(self) -> None:
        """Add a `make_tool(name, description, code, input_schema, ...)` tool.

        On success the new tool is registered for the rest of this Agent's
        lifetime — subsequent turns can call it by name.
        """
        from agi.synthesis import ToolForge, ToolSynthesisError
        forge = ToolForge()

        def make_tool(
            name: str,
            description: str,
            code: str,
            input_schema: dict,
            smoke_test: dict | None = None,
            expected: Any = None,
        ) -> str:
            try:
                tool = forge.compile(name, description, code, input_schema,
                                     smoke_test=smoke_test, expected=expected)
            except ToolSynthesisError as e:
                return f"make_tool failed: {e}"
            if any(s.get("name") == tool.name for s in self.tool_schemas):
                return f"make_tool failed: tool {tool.name!r} already exists"
            self.tool_schemas.append(tool.as_schema())
            self.handlers[tool.name] = tool.handler
            return f"registered tool {tool.name!r}"

        self.tool_schemas.append({
            "name": "make_tool",
            "description": (
                "Author a new Python tool at runtime. Provide a single function "
                "definition matching `name`, a JSON input schema, and ideally a "
                "smoke_test input + expected output. On success the tool is "
                "callable for the rest of the session."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "code": {
                        "type": "string",
                        "description": "Python source. Must define a top-level function whose name matches `name`.",
                    },
                    "input_schema": {
                        "type": "object",
                        "description": "JSON Schema for the new tool's arguments.",
                    },
                    "smoke_test": {
                        "type": "object",
                        "description": "Optional kwargs to test the function before registering.",
                    },
                    "expected": {
                        "description": "Optional expected return value for the smoke_test.",
                    },
                },
                "required": ["name", "description", "code", "input_schema"],
            },
        })
        self.handlers["make_tool"] = make_tool
