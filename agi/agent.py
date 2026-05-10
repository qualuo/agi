"""Core agent loop.

Manual tool-use loop on top of the Messages API. Adaptive thinking + high
effort. Prompt-cached system prompt. Conversation history persists for the
lifetime of the Agent instance.

The web search server-side tool is mixed into the tools list alongside the
custom tools — Claude calls them by the same surface, but Anthropic executes
web_search server-side and returns results inline.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable

import anthropic

from agi.memory import Memory
from agi.tools import make_tools


SYSTEM_PROMPT = """\
You are an agent built on Claude Opus 4.7. You have tools to read and write files,
run shell commands, search the web, and manage a persistent long-term memory that
survives across sessions.

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
        verbose: bool = True,
    ) -> None:
        self.client = anthropic.Anthropic()
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.messages: list[dict] = []

        schemas, handlers = make_tools(self.memory)
        self.tool_schemas: list[dict] = list(schemas)
        self.handlers: dict[str, Callable[..., str]] = handlers

        if enable_web_search:
            self.tool_schemas.append(
                {"type": "web_search_20260209", "name": "web_search"}
            )

    def reset(self) -> None:
        self.messages = []

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self.messages.append({"role": "user", "content": user_input})
        last_text = ""

        for _ in range(max_iterations):
            response = self.client.messages.create(
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
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=self.messages,
            )

            if self.verbose:
                self._print_response(response)

            self.messages.append({"role": "assistant", "content": response.content})

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

        return last_text

    def _dispatch_tool_calls(self, content) -> list[dict]:
        tool_results: list[dict] = []
        for block in content:
            if block.type != "tool_use":
                continue
            handler = self.handlers.get(block.name)
            if handler is None:
                # web_search and other server-side tools land here; skip — the API
                # already handled them and inlined the results.
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

    def _print_response(self, response) -> None:
        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text, flush=True)
            elif block.type == "tool_use":
                preview = self._short_input(block.input)
                print(f"\n[tool: {block.name}({preview})]", flush=True)
            elif block.type == "thinking":
                # adaptive thinking on Opus 4.7 returns omitted text by default
                pass
            elif block.type == "server_tool_use":
                print(f"\n[server: {block.name}]", flush=True)

    @staticmethod
    def _short_input(inp) -> str:
        if not inp:
            return ""
        s = json.dumps(inp, default=str)
        return s if len(s) <= 120 else s[:117] + "..."
