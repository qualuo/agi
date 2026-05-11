"""Fake Anthropic client for tests.

Mimics enough of the `anthropic.Anthropic` interface that the streaming agent
loop runs without a network call. Each call to `messages.stream` pops a
scripted `FakeResponse` off the queue. A `FakeResponse` is a final-message
shape plus the deltas the agent's stream handler will see.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeServerToolUseBlock:
    name: str
    type: str = "server_tool_use"


@dataclass
class FakeFinalMessage:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class _Delta:
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class _StreamEvent:
    type: str
    content_block: Any = None
    delta: _Delta | None = None


def _events_for_message(msg: FakeFinalMessage) -> list[_StreamEvent]:
    """Synthesize a plausible event stream for a final message. Tests don't
    rely on the deltas exactly mirroring the content blocks; they just need
    the agent to emit the right structured events."""
    events: list[_StreamEvent] = []
    for block in msg.content:
        if isinstance(block, FakeTextBlock):
            events.append(_StreamEvent("content_block_start", content_block=block))
            if block.text:
                events.append(
                    _StreamEvent(
                        "content_block_delta",
                        delta=_Delta(type="text_delta", text=block.text),
                    )
                )
        elif isinstance(block, FakeToolUseBlock):
            events.append(_StreamEvent("content_block_start", content_block=block))
        elif isinstance(block, FakeServerToolUseBlock):
            events.append(_StreamEvent("content_block_start", content_block=block))
    return events


class _FakeStream:
    def __init__(self, msg: FakeFinalMessage) -> None:
        self._msg = msg
        self._events = _events_for_message(msg)

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def __iter__(self) -> Iterator[_StreamEvent]:
        return iter(self._events)

    def get_final_message(self) -> FakeFinalMessage:
        return self._msg


class _Messages:
    def __init__(self, parent: "FakeAnthropic") -> None:
        self._parent = parent

    def stream(self, **kwargs) -> _FakeStream:
        self._parent.requests.append(kwargs)
        if not self._parent.responses:
            # If nothing queued, just end the turn with a trivial text reply
            # so the agent doesn't loop forever.
            msg = FakeFinalMessage(content=[FakeTextBlock(text="ok")])
        else:
            msg = self._parent.responses.pop(0)
        return _FakeStream(msg)


class FakeAnthropic:
    """Drop-in replacement for `anthropic.Anthropic` in tests.

    Usage:
        client = FakeAnthropic()
        client.responses.append(FakeFinalMessage(content=[FakeTextBlock("hi")]))
        agent = Agent(client=client, ...)
    """

    def __init__(self) -> None:
        self.responses: list[FakeFinalMessage] = []
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self)


def text_reply(text: str, **usage_kwargs) -> FakeFinalMessage:
    return FakeFinalMessage(
        content=[FakeTextBlock(text=text)],
        usage=FakeUsage(**usage_kwargs) if usage_kwargs else FakeUsage(),
    )


def tool_call_reply(
    *,
    tool_use_id: str,
    name: str,
    input: dict[str, Any],
    preceding_text: str = "",
) -> FakeFinalMessage:
    content: list[Any] = []
    if preceding_text:
        content.append(FakeTextBlock(text=preceding_text))
    content.append(FakeToolUseBlock(id=tool_use_id, name=name, input=input))
    return FakeFinalMessage(content=content, stop_reason="tool_use")
