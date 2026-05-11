"""Runtime + Agent-event tests.

All tests run network-free using `FakeAnthropic`. The point is to lock down
the structured-event contract and the runtime's behavior around sessions,
budgets, and concurrency — that's the surface a coordination engine codes
against.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.events import (
    ErrorEvent,
    TextDelta,
    ToolUseResult,
    ToolUseStart,
    TurnEnd,
    TurnStart,
    UsageDelta,
    from_dict,
)
from agi.memory import Memory
from agi.runtime import Runtime, BudgetExceededError

from tests.fake_client import (
    FakeAnthropic,
    FakeFinalMessage,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
    text_reply,
    tool_call_reply,
)


def _make_agent(client, tmp: Path, **kwargs) -> Agent:
    return Agent(
        memory=Memory(path=tmp / "m.jsonl"),
        client=client,
        verbose=False,
        enable_web_search=False,
        enable_web_fetch=False,
        **kwargs,
    )


class TestAgentEvents(unittest.TestCase):
    def test_emits_turn_start_text_delta_turn_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("hello world"))
            events = []
            agent = _make_agent(
                client,
                Path(tmp),
                event_sink=events.append,
                session_id="s1",
            )
            out = agent.chat("hi")
            self.assertEqual(out, "hello world")
            types = [e.type for e in events]
            self.assertEqual(types[0], "TurnStart")
            self.assertIn("TextDelta", types)
            self.assertEqual(types[-1], "TurnEnd")
            turn_end = events[-1]
            self.assertIsInstance(turn_end, TurnEnd)
            self.assertEqual(turn_end.final_text, "hello world")
            self.assertEqual(turn_end.session_id, "s1")

    def test_emits_tool_use_and_result_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            # Turn 1: agent calls write_file. Turn 2: agent finishes.
            client.responses.append(
                tool_call_reply(
                    tool_use_id="tu_1",
                    name="write_file",
                    input={"path": str(Path(tmp) / "out.txt"), "content": "hi"},
                )
            )
            client.responses.append(text_reply("done"))

            events = []
            agent = _make_agent(client, Path(tmp), event_sink=events.append)
            agent.chat("write 'hi' to out.txt")

            tool_starts = [e for e in events if isinstance(e, ToolUseStart)]
            tool_results = [e for e in events if isinstance(e, ToolUseResult)]
            self.assertEqual(len(tool_starts), 1)
            self.assertEqual(tool_starts[0].name, "write_file")
            self.assertEqual(len(tool_results), 1)
            self.assertFalse(tool_results[0].is_error)
            self.assertEqual((Path(tmp) / "out.txt").read_text(), "hi")

    def test_usage_delta_carries_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(
                FakeFinalMessage(
                    content=[FakeTextBlock("ok")],
                    usage=FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000),
                )
            )
            events = []
            agent = _make_agent(client, Path(tmp), event_sink=events.append)
            agent.chat("hi")
            deltas = [e for e in events if isinstance(e, UsageDelta)]
            self.assertEqual(len(deltas), 1)
            # opus-4-7: 1M in @ $5 + 1M out @ $25 = $30
            self.assertAlmostEqual(deltas[0].cost_usd, 30.0, places=4)

    def test_event_sink_exception_does_not_break_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("ok"))

            def bad_sink(_ev):
                raise RuntimeError("subscriber broke")

            agent = _make_agent(client, Path(tmp), event_sink=bad_sink)
            # The agent must still complete the turn even when the sink throws.
            out = agent.chat("hi")
            self.assertEqual(out, "ok")

    def test_event_round_trip_via_dict(self):
        ev = TextDelta(session_id="s1", text="hello")
        ev.seq = 3
        roundtrip = from_dict(ev.to_dict())
        self.assertEqual(roundtrip.type, "TextDelta")
        self.assertEqual(roundtrip.session_id, "s1")
        self.assertEqual(roundtrip.text, "hello")
        self.assertEqual(roundtrip.seq, 3)


class TestRuntime(unittest.TestCase):
    def _factory(self, client, tmp: Path):
        def factory(**kwargs):
            return Agent(
                client=client,
                enable_web_search=False,
                enable_web_fetch=False,
                **kwargs,
            )
        return factory

    def test_open_send_stream_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("hi back"))
            rt = Runtime(
                agent_factory=self._factory(client, Path(tmp)),
                memory_root=Path(tmp),
            )
            sid = rt.open()
            rt.send(sid, "hi")
            end = rt.wait_for_turn_end(sid, timeout=5.0)
            self.assertIsNotNone(end)
            self.assertEqual(end.final_text, "hi back")
            status = rt.status(sid)
            self.assertEqual(status.state, "idle")
            self.assertEqual(status.turns, 1)
            self.assertGreater(status.input_tokens, 0)
            rt.close(sid)
            # After close, stream should drain and end.
            list(rt.stream(sid, timeout=2.0))
            self.assertEqual(rt.status(sid).state, "closed")

    def test_seq_is_monotonic(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("ok"))
            rt = Runtime(
                agent_factory=self._factory(client, Path(tmp)),
                memory_root=Path(tmp),
            )
            sid = rt.open()
            rt.send(sid, "hi")
            seen = []
            deadline = time.time() + 5
            for ev in rt.stream(sid, timeout=2.0):
                seen.append(ev)
                if isinstance(ev, TurnEnd):
                    break
                if time.time() > deadline:
                    self.fail("timeout waiting for TurnEnd")
            rt.close(sid)
            seqs = [e.seq for e in seen]
            self.assertEqual(seqs, sorted(seqs))
            self.assertTrue(all(s > 0 for s in seqs))

    def test_budget_exceeded_blocks_next_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            # Each call burns $30.
            for _ in range(3):
                client.responses.append(
                    FakeFinalMessage(
                        content=[FakeTextBlock("ok")],
                        usage=FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000),
                    )
                )
            rt = Runtime(
                agent_factory=self._factory(client, Path(tmp)),
                memory_root=Path(tmp),
            )
            # $25 budget — the first turn ($30) consumes it.
            sid = rt.open(budget_usd=25.0)
            rt.send(sid, "first")
            end = rt.wait_for_turn_end(sid, timeout=5.0)
            self.assertIsNotNone(end)
            # Second turn should be refused with an ErrorEvent and never call
            # the (faked) API again.
            rt.send(sid, "second")
            saw_budget_error = False
            for ev in rt.stream(sid, timeout=2.0):
                if isinstance(ev, ErrorEvent) and "budget" in ev.message.lower():
                    saw_budget_error = True
                    break
            self.assertTrue(saw_budget_error, "expected an ErrorEvent about budget")
            rt.close(sid)

    def test_two_sessions_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("A"))
            client.responses.append(text_reply("B"))
            rt = Runtime(
                agent_factory=self._factory(client, Path(tmp)),
                memory_root=Path(tmp),
            )
            a = rt.open()
            b = rt.open()
            rt.send(a, "for A")
            rt.send(b, "for B")
            end_a = rt.wait_for_turn_end(a, timeout=5.0)
            end_b = rt.wait_for_turn_end(b, timeout=5.0)
            self.assertIsNotNone(end_a)
            self.assertIsNotNone(end_b)
            self.assertEqual({end_a.final_text, end_b.final_text}, {"A", "B"})
            self.assertNotEqual(a, b)
            rt.close(a)
            rt.close(b)


if __name__ == "__main__":
    unittest.main()
