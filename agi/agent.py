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
import os
from pathlib import Path
from typing import Callable

import anthropic

from agi.costs import Usage
from agi.memory import Memory
from agi.tools import make_tools
from agi.tools_extension import make_extension_tools
from agi.world_model import WorldModel

try:
    from learner.traces import TraceLogger
except ImportError:  # learner package optional
    TraceLogger = None  # type: ignore


SYSTEM_PROMPT_BASE = """\
You are an agent built on Claude Opus 4.7. You have tools to read and write files,
run shell commands, search and fetch the web, manage a persistent long-term
memory, retrieve and invoke skills from a skill library, delegate to subagents,
synthesize new tools, propose typed task graphs for a coordination engine to
execute, and record structured observations in a world model.

Operating principles:
- Plan before acting on multi-step tasks. Decompose, then execute. For ambiguous
  multi-step work, call `plan_graph` and let the coordinator dispatch it.
- Retrieve skills before re-deriving. Call `list_skills(query)` at task start;
  if a skill matches, `invoke_skill(name, args_json)` to read its SOP, then follow it.
- Use tools instead of guessing. If you need a file's contents, read it. If you
  need a fact, search the web. If you remember something useful, save it.
- Use long-term memory deliberately. Save user preferences, project facts, and
  durable lessons learned. Search memory at the start of related tasks.
- Use the world model. After meaningful actions (file write, URL fetch, command
  run), call `remember_observation` so future tasks can avoid redoing work.
- Delegate when a subtask is independently scoped. `delegate(task, role)` runs
  a focused subagent and returns its answer.
- Verify before claiming success. Read back files you wrote. Run tests where
  applicable. State limitations honestly.
- Be terse. Skip preamble. Show the work, not the throat-clearing.
"""

ROLE_PROMPTS: dict[str, str] = {
    "planner": (
        "ROLE: planner. Your job is to decompose a goal into a typed task DAG "
        "(GraphSpec JSON). Do NOT execute the work yourself; produce the graph "
        "and stop. Use the `decompose-goal` skill if uncertain about the shape."
    ),
    "executor": (
        "ROLE: executor. Your job is to make the goal happen end-to-end. Use "
        "tools, retrieve skills, verify your output."
    ),
    "critic": (
        "ROLE: critic. Score a candidate response for correctness and "
        "trustworthiness on a 0.0–1.0 scale. Reply with JSON only: "
        '{"score": <float>, "explanation": "<one sentence>"}.'
    ),
}


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
        role: str | None = None,
        world_model: WorldModel | None = None,
        enable_extensions: bool = True,
        extra_system_prompt: str | None = None,
        skill_library=None,
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
        self.role = role
        self.world_model = world_model or WorldModel()
        self.extra_system_prompt = extra_system_prompt
        self.messages: list[dict] = []
        self.usage = Usage()
        self.synthesized_tools: dict[str, dict] = {}

        if skill_library is None:
            try:
                from agi.skills.library import SkillLibrary
                self.skill_library = SkillLibrary()
            except Exception:
                self.skill_library = None
        else:
            self.skill_library = skill_library

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = handlers

        if enable_extensions:
            def _sub_factory(r):
                # Defer construction so we don't recurse through extensions.
                return Agent(verbose=False, role=r, enable_extensions=False,
                             skill_library=self.skill_library,
                             world_model=self.world_model)
            ext_schemas, ext_handlers, synthesized = make_extension_tools(
                world_model=self.world_model,
                agent_factory=_sub_factory,
            )
            self.tool_schemas.extend(ext_schemas)
            self.handlers.update(ext_handlers)
            self.synthesized_tools = synthesized

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

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        # Retrieve relevant skills before adding the user turn so they appear
        # in earlier context (better prompt-cache reuse on repeat queries).
        if not self.messages:
            self._maybe_load_skills(user_input)
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

    def _system_prompt(self) -> str:
        parts = [SYSTEM_PROMPT_BASE]
        if self.role and self.role in ROLE_PROMPTS:
            parts.append(ROLE_PROMPTS[self.role])
        if self.extra_system_prompt:
            parts.append(self.extra_system_prompt)
        # Surface synthesized tools so the model knows they exist.
        if self.synthesized_tools:
            names = ", ".join(self.synthesized_tools.keys())
            parts.append(f"Synthesized tools available this session: {names}")
        return "\n\n".join(parts)

    def _maybe_load_skills(self, prompt: str) -> None:
        """Inject the top-K skills relevant to the prompt as a user-role note.

        Done once per chat call. Skills are only retrieved; whether to follow
        one is the model's call.
        """
        if not self.skill_library:
            return
        try:
            skills = self.skill_library.retrieve(prompt, k=2)
        except Exception:
            return
        if not skills:
            return
        block = ["[skills you may find useful for this task]"]
        for s in skills:
            block.append(f"\n--- {s.name}: {s.description}\n{s.body}\n")
        self.messages.append({"role": "user", "content": "\n".join(block)})

    def _stream_one(self):
        # Refresh extension tools that may have been synthesized mid-session.
        if self.synthesized_tools:
            for name, entry in self.synthesized_tools.items():
                if entry["schema"] not in self.tool_schemas:
                    self.tool_schemas.append(entry["schema"])
                self.handlers[name] = entry["fn"]
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
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
