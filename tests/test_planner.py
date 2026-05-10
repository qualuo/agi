"""Planner tests with a stubbed LLM."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.planner import _extract_json, propose_graph
from agi.runtime.graph import GraphSpec, NodeSpec


class FakeAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict] = []
        self.usage = type("U", (), {"input_tokens": 0, "output_tokens": 0,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0})()

    def chat(self, msg: str, **_) -> str:
        self.messages.append({"role": "user", "content": msg})
        return self.reply


def test_extract_json_with_and_without_fences():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}
    assert _extract_json("blah {\"a\":2} blah") == {"a": 2}
    assert _extract_json("no json here") is None


def test_propose_graph_parses_well_formed_response():
    reply = """
    {
      "name": "research",
      "nodes": [
        {"id": "find", "kind": "chat", "input": {"message": "search"}, "depends_on": []},
        {"id": "sum", "kind": "chat", "input": {"message": "summarize ${find.text}"}, "depends_on": ["find"]}
      ]
    }
    """
    fake = FakeAgent(reply)
    out = propose_graph(goal="research X", agent_factory=lambda r: fake)
    assert out["name"] == "research"
    assert len(out["nodes"]) == 2
    assert out["nodes"][1]["depends_on"] == ["find"]


def test_propose_graph_falls_back_on_garbage():
    fake = FakeAgent("I cannot decompose this. Sorry.")
    out = propose_graph(goal="vague goal", agent_factory=lambda r: fake)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["kind"] == "chat"
    assert out["nodes"][0]["input"]["message"] == "vague goal"
