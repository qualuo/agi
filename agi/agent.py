"""Core agent loop.

Streaming Messages API on top of a manual tool-use loop. Adaptive thinking
with summarized display. Prompt-cached system prompt. Conversation history
and cumulative usage persist for the lifetime of the Agent instance.

Server-side `web_search_20260209` and `web_fetch_20260209` are mixed in
alongside the client-side tools. Anthropic executes them server-side and
returns results inline; our `_dispatch_tool_calls` skips them.

Extension hooks for the coordination-engine runtime:
- `skills`: a `learner.SkillLibrary` that powers recall/save/list_skills
  tools, and gets sampled into the system prompt before each turn.
- `delegate_runtime`: a callable `(task: str) -> str` that the agent can
  invoke via the `delegate` tool. Wire this from the runtime layer.
- `enable_tool_synthesis`: when True, exposes `make_tool` and lets the
  agent register sandboxed Python tools at runtime.
- `reflect`: when True, the agent writes a `lesson`-tagged memory note
  after each task — the per-task durable-improvement loop from
  `ARCHITECTURE.md`.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from agi.costs import Usage
from agi.memory import Memory
from agi.tools import make_tools

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


BASE_SYSTEM_PROMPT = """\
You are an agent built on Claude Opus 4.7. You have tools to read and write files,
run shell commands, search and fetch the web, and manage a persistent long-term
memory that survives across sessions.

Operating principles:
- Plan before acting on multi-step tasks. Decompose, then execute.
- Use tools instead of guessing. If you need a file's contents, read it. If you
  need a fact, search the web. If you remember something useful, save it.
- Use long-term memory deliberately. Save user preferences, project facts, and
  durable lessons learned. Search memory at the start of related tasks.
- Check the skill library at the start of unfamiliar tasks (`recall_skill`).
  When you solve something whose procedure will recur, save it (`save_skill`).
- Verify before claiming success. Read back files you wrote. Run tests where
  applicable. State limitations honestly.
- Be terse. Skip preamble. Show the work, not the throat-clearing.
"""

# Backwards compatible alias for callers that imported the old name.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT


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
        skills=None,
        delegate_runtime: Callable[[str], str] | None = None,
        enable_tool_synthesis: bool = False,
        reflect: bool = False,
        system_prompt_extra: str = "",
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
        self.skills = skills
        self.reflect = reflect
        self.system_prompt_extra = system_prompt_extra
        self.messages: list[dict] = []
        self.usage = Usage()
        self._dynamic_tool_names: list[str] = []  # for capability introspection

        schemas, handlers = make_tools(
            self.memory,
            skills=skills,
            runtime=delegate_runtime,
            on_register_tool=(self._register_tool if enable_tool_synthesis else None),
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

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    # ----- capability introspection (for the coordination engine) -----

    def capabilities(self) -> dict[str, Any]:
        """Describe what this agent can do. Coordinators introspect this to
        route work and to know what's safe to ask. Output is JSON-clean."""
        client_tools: list[str] = []
        server_tools: list[str] = []
        for s in self.tool_schemas:
            # Anthropic server-side tools include a `type` field like
            # "web_search_20260209"; client-side tools use only `name` +
            # `input_schema`. We separate them so coordinators know which
            # are sandboxed by the API vs run locally.
            if "type" in s and s.get("type", "").endswith(("_20260209",)):
                server_tools.append(s["type"])
            elif "name" in s:
                client_tools.append(s["name"])
        return {
            "model": self.model,
            "effort": self.effort,
            "tools": sorted(client_tools),
            "server_tools": sorted(server_tools),
            "dynamic_tools": list(self._dynamic_tool_names),
            "has_skill_library": self.skills is not None,
            "has_critic": self.critic is not None,
            "has_tracer": self.tracer is not None,
            "reflect": self.reflect,
        }

    # ----- main entry point -----

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
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
                continue

            # refusal, max_tokens, stop_sequence, model_context_window_exceeded, ...
            break

        last_text, critic_score = self._apply_critic_gate(user_input, last_text)
        self.last_critic_score = critic_score

        if self.reflect:
            self._write_reflection(user_input, last_text)

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

    # ----- helpers -----

    def _register_tool(self, schema: dict, handler: Callable[..., str]) -> None:
        """Callback passed to `make_tools` so synthesized tools register live.

        Duplicate names overwrite the previous handler — that's intentional
        so the model can iteratively refine a tool without rename ceremony.
        """
        name = schema["name"]
        # Replace existing schema with the same name, else append.
        self.tool_schemas = [s for s in self.tool_schemas if s.get("name") != name]
        self.tool_schemas.append(schema)
        self.handlers[name] = handler
        if name not in self._dynamic_tool_names:
            self._dynamic_tool_names.append(name)

    def _write_reflection(self, prompt: str, response: str) -> None:
        # Best-effort and silent on failure. Reflections are an optimization,
        # never required for correctness.
        try:
            note = (
                f"task: {prompt[:200]}\n"
                f"outcome: {response[:300]}\n"
            )
            self.memory.save(note, tags=["lesson", "reflection"])
        except Exception:
            pass

    def _system_prompt(self) -> list[dict]:
        """Build the system prompt with optional skill snippets and caller-extra.

        Skill snippets are kept short and bullet-summarized to stay within
        the prompt-cache breakpoint. If the library is empty or absent, we
        just return the base prompt.
        """
        parts = [BASE_SYSTEM_PROMPT]
        if self.skills is not None:
            try:
                items = self.skills.all()
            except Exception:
                items = []
            if items:
                lines = ["Skills currently available (call `recall_skill` to load the full body):"]
                for s in items[:25]:  # cap to keep the cached prefix bounded
                    triggers = ", ".join(s.triggers) if s.triggers else "—"
                    lines.append(f"- {s.name}: triggers=[{triggers}]")
                parts.append("\n".join(lines))
        if self.system_prompt_extra:
            parts.append(self.system_prompt_extra)
        text = "\n\n".join(parts)
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

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
