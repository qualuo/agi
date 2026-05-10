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

try:
    from learner.skills import SkillLibrary
except ImportError:  # learner package optional
    SkillLibrary = None  # type: ignore


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
        skills=None,
        skills_top_k: int = 3,
        enable_delegate: bool = False,
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
        # skills is an optional learner.SkillLibrary; when set, the top-k
        # matching skills are appended to the system prompt at each chat() call.
        self.skills = skills
        self.skills_top_k = skills_top_k
        self.last_skills_used: list[str] = []
        self.messages: list[dict] = []
        self.usage = Usage()

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

        if enable_delegate:
            self._install_delegate_tool()

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""
        turn_usage = Usage()

        skills_block, skills_used = self._select_skills(user_input)
        self.last_skills_used = skills_used

        for _ in range(max_iterations):
            response = self._stream_one(skills_block=skills_block)

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
            if skills_used:
                metadata["skills_used"] = skills_used
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

    def _select_skills(self, user_input: str) -> tuple[str, list[str]]:
        """Pick skills relevant to this turn and render them for the prompt.

        Returns (block, names). The block is "" when no skills match or no
        library is configured, in which case the system prompt is unchanged.
        """
        if self.skills is None:
            return "", []
        try:
            hits = self.skills.search(user_input, k=self.skills_top_k)
        except Exception:
            return "", []
        if not hits:
            return "", []
        block = "## Relevant skills from your library\n\n" + "\n\n".join(
            s.render() for s in hits
        )
        return block, [s.name for s in hits]

    def _stream_one(self, skills_block: str = ""):
        # The base prompt is cached separately so adding ephemeral skill blocks
        # doesn't invalidate the prompt cache. Each system block is its own
        # cache key; only blocks that actually change get rebuilt.
        system_blocks: list[dict] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if skills_block:
            system_blocks.append({"type": "text", "text": skills_block})

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

    def _install_delegate_tool(self) -> None:
        """Add a `delegate` tool that spawns a sub-Agent for a focused task.

        Subagent token usage rolls up into the parent's `usage` so cost
        accounting remains correct. Sub-agents do not inherit the parent's
        memory unless explicitly handed one — the default is isolated to
        prevent the sub-task from polluting long-term memory.
        """
        parent = self

        def delegate(task: str, role: str = "executor", max_iterations: int = 8) -> str:
            sub = Agent(
                memory=Memory(path=parent.memory.path),  # share read view
                model=parent.model,
                max_tokens=parent.max_tokens,
                effort=parent.effort,
                enable_web_search=False,  # disable in subagents by default
                enable_web_fetch=False,
                verbose=False,
                # No tracer/critic/skills inheritance in v1: subagent traces
                # are inlined into the parent's final trace via tool results.
                enable_delegate=False,  # one level of delegation only
            )
            framed = (
                f"You are a sub-agent with role: {role}. "
                f"Complete the focused task below and return only the final answer.\n\n"
                f"Task:\n{task}"
            )
            try:
                result = sub.chat(framed, max_iterations=max_iterations)
            except Exception as e:
                return f"error: subagent failed: {type(e).__name__}: {e}"
            # Roll usage up to the parent so cost accounting stays accurate.
            parent.usage.input_tokens += sub.usage.input_tokens
            parent.usage.output_tokens += sub.usage.output_tokens
            parent.usage.cache_creation_input_tokens += sub.usage.cache_creation_input_tokens
            parent.usage.cache_read_input_tokens += sub.usage.cache_read_input_tokens
            return result

        self.tool_schemas.append(
            {
                "name": "delegate",
                "description": (
                    "Spawn a focused sub-agent to handle a self-contained subtask "
                    "and return its final answer. Use for parallel decomposition or "
                    "when a sub-task benefits from a clean context. The sub-agent "
                    "has filesystem and shell tools but no web/delegate."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The full task description for the sub-agent.",
                        },
                        "role": {
                            "type": "string",
                            "description": "Role hint, e.g. 'planner', 'executor', 'critic'.",
                            "default": "executor",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Max tool-loop iterations the sub-agent may run.",
                            "default": 8,
                        },
                    },
                    "required": ["task"],
                },
            }
        )
        self.handlers["delegate"] = delegate

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
