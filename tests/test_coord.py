"""Coordinator tests.

These exercise the orchestration patterns a real coordination engine would
use against this runtime: single-task execution, parallel fan-out, decompose
and synthesize, budget propagation. Everything runs network-free with a
scripted FakeAnthropic client.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.coord import Coordinator, Task, _parse_subtasks_json
from agi.runtime import Runtime

from tests.fake_client import FakeAnthropic, text_reply


def _runtime_with(client: FakeAnthropic, tmp: Path) -> Runtime:
    def factory(**kwargs):
        return Agent(
            client=client,
            enable_web_search=False,
            enable_web_fetch=False,
            **kwargs,
        )
    return Runtime(agent_factory=factory, memory_root=tmp)


class TestParseSubtasks(unittest.TestCase):
    def test_clean_array(self):
        text = '[{"prompt": "do A", "role": "executor"}]'
        tasks = _parse_subtasks_json(text)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].prompt, "do A")
        self.assertEqual(tasks[0].role, "executor")

    def test_array_in_fenced_block(self):
        text = '```json\n[{"prompt": "do B"}]\n```'
        tasks = _parse_subtasks_json(text)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].prompt, "do B")
        # Default role is executor when omitted.
        self.assertEqual(tasks[0].role, "executor")

    def test_array_with_prose(self):
        text = "Here are the subtasks:\n[{\"prompt\": \"X\"}, {\"prompt\": \"Y\"}]\nDone."
        tasks = _parse_subtasks_json(text)
        self.assertEqual([t.prompt for t in tasks], ["X", "Y"])

    def test_garbage_returns_empty(self):
        self.assertEqual(_parse_subtasks_json("no json here"), [])
        self.assertEqual(_parse_subtasks_json(""), [])

    def test_invalid_entries_are_filtered(self):
        text = '[{"prompt": ""}, {"role": "executor"}, {"prompt": "ok"}]'
        tasks = _parse_subtasks_json(text)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].prompt, "ok")


class _ScriptedClient:
    """Thread-safe variant of FakeAnthropic that pops scripted responses
    keyed on a substring of the system prompt. Lets us test parallel-fan-out
    where two subagents run concurrently with different roles."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._responses: list[tuple[str | None, object]] = []
        self.messages = self
        self.calls = 0

    def expect(self, system_substring: str | None, msg) -> None:
        self._responses.append((system_substring, msg))

    def stream(self, **kwargs):
        with self.lock:
            self.calls += 1
            sys_text = ""
            for s in kwargs.get("system") or []:
                sys_text += s.get("text", "") if isinstance(s, dict) else ""
            # Find the first matching scripted response (by substring) and
            # remove it. Fall back to first available if no match.
            match_idx = None
            for i, (needle, _msg) in enumerate(self._responses):
                if needle is None or needle in sys_text:
                    match_idx = i
                    break
            if match_idx is None:
                if not self._responses:
                    from tests.fake_client import text_reply as _tr
                    _, msg = (None, _tr("ok"))
                else:
                    _, msg = self._responses.pop(0)
            else:
                _, msg = self._responses.pop(match_idx)
        from tests.fake_client import _FakeStream
        return _FakeStream(msg)


class TestCoordinator(unittest.TestCase):
    def test_run_one_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("the answer is 42"))
            rt = _runtime_with(client, Path(tmp))
            co = Coordinator(rt)
            result = co.run_one(Task(prompt="what's the answer?", role="executor"))
            self.assertEqual(result.final_text, "the answer is 42")
            self.assertEqual(result.error, None)
            self.assertGreater(result.elapsed_s, 0)

    def test_run_parallel_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            client.responses.append(text_reply("first done"))
            client.responses.append(text_reply("second done"))
            rt = _runtime_with(client, Path(tmp))
            co = Coordinator(rt, max_parallel=2)
            results = co.run_parallel([
                Task(prompt="A"),
                Task(prompt="B"),
            ])
            self.assertEqual(len(results), 2)
            texts = {r.final_text for r in results}
            self.assertEqual(texts, {"first done", "second done"})

    def test_run_with_subtasks_synthesizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _ScriptedClient()
            client.expect("RESEARCHER", text_reply("found fact X"))
            client.expect("EXECUTOR", text_reply("executed step Y"))
            client.expect("SYNTHESIZER", text_reply("final report: X + Y"))
            rt = _runtime_with(client, Path(tmp))
            co = Coordinator(rt, max_parallel=4)
            parent = Task(
                prompt="combined task",
                role="planner",
                subtasks=[
                    Task(prompt="research X", role="researcher"),
                    Task(prompt="execute Y", role="executor"),
                ],
            )
            result = co.run_one(parent)
            self.assertEqual(result.final_text, "final report: X + Y")
            self.assertEqual(len(result.children), 2)
            child_texts = {c.final_text for c in result.children}
            self.assertEqual(child_texts, {"found fact X", "executed step Y"})

    def test_total_cost_rolls_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeAnthropic()
            from tests.fake_client import FakeFinalMessage, FakeTextBlock, FakeUsage
            # Two children each cost $30 (1M in + 1M out). Synthesizer too.
            for _ in range(3):
                client.responses.append(
                    FakeFinalMessage(
                        content=[FakeTextBlock("done")],
                        usage=FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000),
                    )
                )
            rt = _runtime_with(client, Path(tmp))
            co = Coordinator(rt)
            result = co.run_one(
                Task(
                    prompt="parent",
                    subtasks=[Task(prompt="a"), Task(prompt="b")],
                )
            )
            # 2 children + 1 synthesizer = $90.
            self.assertAlmostEqual(result.total_cost_usd(), 90.0, places=2)


if __name__ == "__main__":
    unittest.main()
