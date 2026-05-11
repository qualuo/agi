"""Mock agent for offline testing and demos.

Drop-in replacement for `agi.Agent` that needs no API key and no network. It
exposes the same `chat()` shape so the runtime, server, tests, and client can
exercise the full lifecycle (sessions, jobs, streaming, budgets) without
spending tokens. Behaviour is deterministic and trivial: it echoes the prompt
and, for inputs that look like arithmetic, returns the answer.

Real Agents are still injected via factory in `Runtime`; this only kicks in
when explicitly selected (`backend="mock"` or `AGI_RUNTIME_BACKEND=mock`).
"""
from __future__ import annotations

import re
import time
from typing import Callable

from agi.costs import Usage
from agi.memory import Memory


class MockAgent:
    """Compatible-shape stand-in for `agi.Agent`."""

    def __init__(
        self,
        memory: Memory | None = None,
        model: str = "mock-1",
        max_tokens: int = 16000,
        effort: str = "high",
        enable_web_search: bool = False,
        enable_web_fetch: bool = False,
        verbose: bool = False,
        tracer=None,
        critic=None,
        critic_threshold: float = 0.5,
        delay_seconds: float = 0.0,
    ) -> None:
        self.memory = memory or Memory()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.verbose = verbose
        self.tracer = tracer
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self.usage = Usage()
        self._delay = delay_seconds
        # Hooks: runtime injects a callback to stream tokens.
        self.on_text_chunk: Callable[[str], None] | None = None

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        if self._delay:
            time.sleep(self._delay)
        self.messages.append({"role": "user", "content": user_input})
        text = self._respond(user_input)

        # Stream tokens to any subscriber so SSE tests see realistic chunks.
        if self.on_text_chunk is not None:
            for chunk in _split_into_chunks(text, n=4):
                self.on_text_chunk(chunk)

        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

        in_toks = max(1, len(user_input) // 4)
        out_toks = max(1, len(text) // 4)
        self.usage.input_tokens += in_toks
        self.usage.output_tokens += out_toks
        self.usage.turns += 1

        if self.tracer is not None:
            self.tracer.log(
                model=self.model,
                messages=self.messages,
                final_text=text,
                usage={
                    "input_tokens": in_toks,
                    "output_tokens": out_toks,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                metadata={"mock": True},
            )
        return text

    @staticmethod
    def _respond(prompt: str) -> str:
        m = re.fullmatch(r"\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*", prompt)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            try:
                if op == "+":
                    return str(a + b)
                if op == "-":
                    return str(a - b)
                if op == "*":
                    return str(a * b)
                if op == "/":
                    return str(a / b) if b != 0 else "error: division by zero"
            except Exception as e:
                return f"error: {e}"
        return f"mock-echo: {prompt}"


def _split_into_chunks(text: str, n: int) -> list[str]:
    if not text:
        return []
    size = max(1, len(text) // max(1, n))
    return [text[i : i + size] for i in range(0, len(text), size)]
