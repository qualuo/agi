"""Test fakes for the Agent surface.

The real Agent requires an Anthropic client + network. The Runtime only
cares about the Agent's behavioral contract, not the model under the
hood, so tests inject a FakeAgent that mimics the same surface.
"""
from __future__ import annotations

import time
from typing import Any

from agi.costs import Usage
from agi.events import Event, EventBus, stdout_printer


class FakeAgent:
    """Drop-in for agi.agent.Agent in tests.

    Behavior:
      - `chat()` emits the standard event sequence, advances Usage, and
        returns a canned response derived from the prompt.
      - `cancel()` flips a flag; subsequent `chat()` calls are short-circuited.
      - Constructor accepts everything the Runtime passes.
    """

    def __init__(
        self,
        memory=None,
        model: str = "claude-opus-4-7",
        max_tokens: int = 16000,
        effort: str = "high",
        enable_web_search: bool = True,
        enable_web_fetch: bool = True,
        verbose: bool = False,
        tracer=None,
        critic=None,
        critic_threshold: float = 0.5,
        events: EventBus | None = None,
        skills: Any = None,
        session_id: str | None = None,
    ) -> None:
        self.memory = memory
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.enable_web_search = enable_web_search
        self.enable_web_fetch = enable_web_fetch
        self.verbose = verbose
        self.tracer = tracer
        self.critic = critic
        self.critic_threshold = critic_threshold
        self.events = events or EventBus()
        self.skills = skills
        self.session_id = session_id
        self.usage = Usage()
        self.last_critic_score: float | None = None
        self.messages: list[dict] = []
        self._cancelled = False
        if verbose:
            self.events.subscribe(stdout_printer)

    def reset(self) -> None:
        self.messages = []
        self.usage = Usage()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def chat(self, user_input: str, max_iterations: int = 25, run_id: str | None = None) -> str:
        self.events.publish(Event(
            kind="run.started",
            data={"run_id": run_id, "prompt_chars": len(user_input)},
            session_id=self.session_id,
            run_id=run_id,
        ))
        if self._cancelled:
            self.events.publish(Event(
                kind="cancelled", data={"run_id": run_id},
                session_id=self.session_id, run_id=run_id,
            ))
            self.events.publish(Event(
                kind="run.finished",
                data={"run_id": run_id, "stop_reason": None, "cancelled": True,
                      "turn_usage": {"input_tokens": 0, "output_tokens": 0,
                                     "cache_creation_input_tokens": 0,
                                     "cache_read_input_tokens": 0, "cost_usd": 0.0},
                      "response_chars": 0},
                session_id=self.session_id, run_id=run_id,
            ))
            return ""

        self.messages.append({"role": "user", "content": user_input})

        # Inject a deterministic "turn" so per-run usage is non-zero.
        self.events.publish(Event(kind="turn.started", data={"run_id": run_id},
                                  session_id=self.session_id, run_id=run_id))
        response = f"echo: {user_input.strip()}"
        self.events.publish(Event(kind="text.delta", data={"text": response},
                                  session_id=self.session_id, run_id=run_id))

        class _FakeRespUsage:
            input_tokens = 10
            output_tokens = 20
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0

        self.usage.add(_FakeRespUsage())
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        self.events.publish(Event(
            kind="turn.finished",
            data={"run_id": run_id, "stop_reason": "end_turn", "usage_formatted": "fake"},
            session_id=self.session_id, run_id=run_id,
        ))
        self.events.publish(Event(
            kind="run.finished",
            data={"run_id": run_id, "stop_reason": "end_turn", "cancelled": False,
                  "turn_usage": {"input_tokens": 10, "output_tokens": 20,
                                 "cache_creation_input_tokens": 0,
                                 "cache_read_input_tokens": 0, "cost_usd": 0.00055},
                  "response_chars": len(response)},
            session_id=self.session_id, run_id=run_id,
        ))
        # Tiny sleep so timestamps separate.
        time.sleep(0.0001)
        return response
