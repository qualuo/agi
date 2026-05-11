"""Test doubles: fake Anthropic SDK objects + fake Agent.

These let the runtime/session/server tests run with no API key, no
network, and no anthropic install.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---- minimal stand-ins for SDK content blocks -------------------------------


class TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id = id
        self.name = name
        self.input = input


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: list
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


# ---- minimal stand-in for `client.messages.stream(...)` ---------------------


class FakeStream:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([])

    def get_final_message(self) -> FakeResponse:
        return self._response


class FakeMessages:
    def __init__(self, scripted: Callable[..., FakeResponse]) -> None:
        self._scripted = scripted

    def stream(self, **kwargs):
        return FakeStream(self._scripted(**kwargs))


class FakeClient:
    """Drop-in for anthropic.Anthropic; one scripted response per call.

    `scripted(**kwargs)` is invoked each time the agent streams; return
    a FakeResponse. Closure-state lets a test sequence multi-turn flows.
    """

    def __init__(self, scripted: Callable[..., FakeResponse]) -> None:
        self.messages = FakeMessages(scripted)


def text_response(text: str, usage: FakeUsage | None = None) -> FakeResponse:
    return FakeResponse(content=[TextBlock(text)], usage=usage or FakeUsage())


def tool_call_response(name: str, input: dict, *, tool_id: str = "t1") -> FakeResponse:
    return FakeResponse(
        content=[ToolUseBlock(tool_id, name, input)],
        usage=FakeUsage(input_tokens=200, output_tokens=20),
        stop_reason="tool_use",
    )


# ---- factories -------------------------------------------------------------


def make_fake_agent_factory(scripted: Callable[..., FakeResponse]):
    """Return an `agent_factory` for Session/Runtime that uses FakeClient.

    The factory must accept all the kwargs Session passes; we forward
    them to the real Agent class with our fake client injected.
    """
    from agi.agent import Agent

    def factory(**kwargs):
        kwargs.setdefault("enable_web_search", False)
        kwargs.setdefault("enable_web_fetch", False)
        kwargs["client"] = FakeClient(scripted)
        return Agent(**kwargs)

    return factory


def constant_factory(text: str = "ok"):
    """Trivial factory: every turn ends with the given text. No tools."""
    def scripted(**_kwargs):
        return text_response(text)
    return make_fake_agent_factory(scripted)


def counting_factory(text: str = "ok"):
    """Returns (factory, counter_dict) — counter['n'] is incremented per call."""
    counter = {"n": 0}

    def scripted(**_kwargs):
        counter["n"] += 1
        return text_response(f"{text} #{counter['n']}")

    return make_fake_agent_factory(scripted), counter
