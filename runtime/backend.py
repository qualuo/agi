"""Backend abstraction for the LLM call.

`Agent._stream_one` used to call `anthropic.Anthropic().messages.stream(...)`
directly. The runtime needs to:

- Run end-to-end tests without an `ANTHROPIC_API_KEY` (CI, dev loop)
- Plug different model providers in eventually (the architecture is agnostic)

A `Backend` exposes one method: `stream_message(**api_kwargs) -> ContextManager`.
The returned context manager is duck-compatible with what the Anthropic SDK
returns from `client.messages.stream(...)`:

- iterating it yields stream events (objects with `.type`, `.delta`, `.content_block`)
- calling `.get_final_message()` returns the final `Message` with `.content`,
  `.stop_reason`, `.usage`

`AnthropicBackend` is the production path. `MockBackend` produces canned
responses from a script — used by tests and the example coordinator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator


class Backend:
    """Interface. Subclass and implement `stream_message`."""

    def stream_message(self, **api_kwargs):  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real backend — thin wrapper around anthropic.Anthropic
# ---------------------------------------------------------------------------


class AnthropicBackend(Backend):
    """Production backend. Requires `ANTHROPIC_API_KEY` in the environment."""

    def __init__(self, client=None) -> None:
        import anthropic
        self.client = client or anthropic.Anthropic()

    def stream_message(self, **api_kwargs):
        return self.client.messages.stream(**api_kwargs)


# ---------------------------------------------------------------------------
# Mock backend — for tests and offline demos
# ---------------------------------------------------------------------------


@dataclass
class MockBlock:
    """Minimal stand-in for an anthropic content block."""
    type: str
    text: str | None = None
    name: str | None = None
    id: str | None = None
    input: dict | None = None

    def model_dump(self, exclude_none: bool = False) -> dict:
        d = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.name is not None:
            d["name"] = self.name
        if self.id is not None:
            d["id"] = self.id
        if self.input is not None:
            d["input"] = self.input
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


@dataclass
class MockUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class MockMessage:
    content: list[MockBlock]
    stop_reason: str
    usage: MockUsage = field(default_factory=MockUsage)


class _MockStream:
    """Iterable + context manager matching the SDK's stream shape."""

    def __init__(self, message: MockMessage) -> None:
        self._message = message

    def __enter__(self) -> "_MockStream":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def __iter__(self) -> Iterator:
        # Mock streams are silent — no per-token events. Production code that
        # iterates them (the verbose display path) just sees an empty stream.
        return iter([])

    def get_final_message(self) -> MockMessage:
        return self._message


# Type for a script: callable that takes the in-flight messages and returns
# the next MockMessage. This lets a test author script a multi-turn response
# pattern (e.g., first turn asks for a tool, second turn finishes).
MockResponder = Callable[[list[dict]], MockMessage]


class MockBackend(Backend):
    """Returns scripted responses. Useful when you want determinism without
    an API key.

    Construct one of three ways:

      MockBackend.echo("hi")
          -> always returns a single text block "hi" and ends the turn.

      MockBackend.scripted([msg1, msg2, ...])
          -> returns the next message from the list on each call. After the
             list is exhausted, returns a benign end_turn with empty text.

      MockBackend(responder=lambda messages: MockMessage(...))
          -> full control. The responder sees the current message list and
             returns whatever it wants.
    """

    def __init__(self, responder: MockResponder | None = None) -> None:
        self.responder: MockResponder = responder or (
            lambda messages: MockMessage(
                content=[MockBlock(type="text", text="")],
                stop_reason="end_turn",
            )
        )
        self.calls: list[dict] = []  # for assertions in tests

    def stream_message(self, **api_kwargs):
        self.calls.append(api_kwargs)
        msg = self.responder(api_kwargs.get("messages", []))
        return _MockStream(msg)

    # --- convenience constructors -----------------------------------------

    @classmethod
    def echo(cls, text: str) -> "MockBackend":
        msg = MockMessage(
            content=[MockBlock(type="text", text=text)],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=5, output_tokens=len(text) // 4 + 1),
        )
        return cls(responder=lambda messages: msg)

    @classmethod
    def scripted(cls, messages: Iterable[MockMessage]) -> "MockBackend":
        queue = list(messages)
        idx = {"i": 0}

        def step(_msgs: list[dict]) -> MockMessage:
            if idx["i"] < len(queue):
                m = queue[idx["i"]]
                idx["i"] += 1
                return m
            return MockMessage(
                content=[MockBlock(type="text", text="")],
                stop_reason="end_turn",
            )
        return cls(responder=step)

    # --- helpers for constructing MockMessages in tests --------------------

    @staticmethod
    def text(text: str, *, input_tokens: int = 5, output_tokens: int = 5) -> MockMessage:
        return MockMessage(
            content=[MockBlock(type="text", text=text)],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    @staticmethod
    def tool_call(
        name: str,
        tool_input: dict,
        *,
        block_id: str = "toolu_mock",
        leading_text: str | None = None,
        input_tokens: int = 5,
        output_tokens: int = 5,
    ) -> MockMessage:
        content: list[MockBlock] = []
        if leading_text:
            content.append(MockBlock(type="text", text=leading_text))
        content.append(MockBlock(type="tool_use", name=name, id=block_id, input=tool_input))
        return MockMessage(
            content=content,
            stop_reason="tool_use",
            usage=MockUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    @staticmethod
    def from_json(blob: str) -> MockMessage:
        """Parse a `{content: [...], stop_reason: ...}` dict into a MockMessage.
        Convenient for fixture files."""
        d = json.loads(blob)
        blocks = [MockBlock(**b) for b in d["content"]]
        return MockMessage(content=blocks, stop_reason=d.get("stop_reason", "end_turn"))
