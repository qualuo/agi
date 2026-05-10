"""Runtime protocol tests.

Drives the runtime by injecting in-memory stdin/stdout streams and a fake
Agent. No API key required.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.skills import SkillLibrary
from agi.runtime import Runtime, PROTOCOL_VERSION


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 12
        self.output_tokens = 34
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 1

    def cost_usd(self, model: str) -> float:
        return 0.0001


class FakeAgent:
    """Minimal stand-in for `Agent` that doesn't talk to the API.

    Implements only the surface the runtime touches.
    """

    def __init__(self, memory: Memory, skills: SkillLibrary | None) -> None:
        self.memory = memory
        self.skills = skills
        self.model = "fake-model"
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.critic = None
        self.tracer = None
        self.verbose = False
        self.on_event: Callable[[dict], None] | None = None
        self.tool_schemas: list[dict] = [{"name": "read_file"}, {"name": "write_file"}]
        self._reset_calls = 0
        self._last_chat: str | None = None

    def reset(self) -> None:
        self._reset_calls += 1

    def chat(self, user_input: str, max_iterations: int = 25) -> str:
        self._last_chat = user_input
        if self.on_event:
            self.on_event({"kind": "text_start"})
            self.on_event({"kind": "text_delta", "text": "echo: " + user_input})
        return "echo: " + user_input


def _drive(requests: list[dict], agent: FakeAgent) -> list[dict]:
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    runtime = Runtime(agent=agent, stdin=stdin, stdout=stdout)
    runtime.serve()
    out: list[dict] = []
    for line in stdout.getvalue().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class TestRuntime(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.memory = Memory(path=tmp / "m.jsonl")
        self.skills = SkillLibrary(path=tmp / "skills")
        self.agent = FakeAgent(self.memory, self.skills)

    def tearDown(self):
        self._tmp.cleanup()

    def test_emits_ready_on_start_then_bye_on_shutdown(self):
        out = _drive([{"id": "1", "type": "shutdown"}], self.agent)
        self.assertEqual(out[0], {"type": "ready", "version": PROTOCOL_VERSION})
        # shutdown emits {result, bye}
        self.assertEqual(out[-1], {"type": "bye"})
        self.assertEqual(out[-2]["type"], "result")
        self.assertEqual(out[-2]["req_id"], "1")

    def test_hello_returns_version(self):
        out = _drive(
            [{"id": "h", "type": "hello"}, {"id": "s", "type": "shutdown"}],
            self.agent,
        )
        results = [m for m in out if m["type"] == "result"]
        hello = results[0]
        self.assertEqual(hello["req_id"], "h")
        self.assertEqual(hello["version"], PROTOCOL_VERSION)
        self.assertEqual(hello["model"], "fake-model")

    def test_capabilities_lists_tools(self):
        out = _drive(
            [{"id": "c", "type": "capabilities"}, {"id": "s", "type": "shutdown"}],
            self.agent,
        )
        caps = next(m for m in out if m["type"] == "result" and m["req_id"] == "c")
        self.assertIn("read_file", caps["tools"])
        self.assertTrue(caps["has_skills"])

    def test_chat_emits_events_and_result(self):
        out = _drive(
            [{"id": "c", "type": "chat", "input": "hello"}, {"id": "s", "type": "shutdown"}],
            self.agent,
        )
        events = [m for m in out if m["type"] == "event" and m["req_id"] == "c"]
        self.assertTrue(any(e["kind"] == "text_delta" for e in events))
        result = next(m for m in out if m["type"] == "result" and m["req_id"] == "c")
        self.assertEqual(result["text"], "echo: hello")
        self.assertIn("usage", result)
        self.assertEqual(result["usage"]["input_tokens"], 12)

    def test_memory_save_then_search_roundtrip(self):
        out = _drive(
            [
                {"id": "a", "type": "memory.save", "text": "favorite color is teal", "tags": ["pref"]},
                {"id": "b", "type": "memory.search", "query": "teal", "k": 5},
                {"id": "s", "type": "shutdown"},
            ],
            self.agent,
        )
        save = next(m for m in out if m["type"] == "result" and m["req_id"] == "a")
        self.assertTrue(save["ok"])
        search = next(m for m in out if m["type"] == "result" and m["req_id"] == "b")
        self.assertEqual(len(search["results"]), 1)
        self.assertIn("teal", search["results"][0]["text"])

    def test_skills_save_via_lib_then_listed_by_runtime(self):
        self.skills.save("ship", "Ship code.", "1. push.\n", tags=["deploy"])
        out = _drive(
            [{"id": "l", "type": "skills.list"}, {"id": "s", "type": "shutdown"}],
            self.agent,
        )
        result = next(m for m in out if m["type"] == "result" and m["req_id"] == "l")
        names = [s["name"] for s in result["skills"]]
        self.assertIn("ship", names)

    def test_skills_find_filters(self):
        self.skills.save("deploy-svc", "Deploy a service.", "body", tags=["deploy"])
        self.skills.save("read-config", "Read config file.", "body", tags=["io"])
        out = _drive(
            [
                {"id": "f", "type": "skills.find", "query": "deploy"},
                {"id": "s", "type": "shutdown"},
            ],
            self.agent,
        )
        result = next(m for m in out if m["type"] == "result" and m["req_id"] == "f")
        names = [s["name"] for s in result["skills"]]
        self.assertEqual(names, ["deploy-svc"])

    def test_invalid_json_emits_error(self):
        stdin = io.StringIO("not json\n" + json.dumps({"id": "s", "type": "shutdown"}) + "\n")
        stdout = io.StringIO()
        Runtime(agent=self.agent, stdin=stdin, stdout=stdout).serve()
        msgs = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        errs = [m for m in msgs if m["type"] == "error"]
        self.assertEqual(len(errs), 1)
        self.assertIn("invalid json", errs[0]["message"])

    def test_unknown_type_emits_error(self):
        out = _drive(
            [
                {"id": "u", "type": "no-such-thing"},
                {"id": "s", "type": "shutdown"},
            ],
            self.agent,
        )
        err = next(m for m in out if m["type"] == "error" and m["req_id"] == "u")
        self.assertIn("unknown request type", err["message"])

    def test_reset_calls_through(self):
        _drive(
            [{"id": "r", "type": "reset"}, {"id": "s", "type": "shutdown"}],
            self.agent,
        )
        self.assertEqual(self.agent._reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
