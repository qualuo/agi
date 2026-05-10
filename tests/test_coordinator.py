"""Coordinator tests using an in-process runtime with stub handlers.

We sidestep the API by injecting fake `plan`, `chat`, and `critique` handlers
into a Runtime instance. This validates that the coordinator wires plan ->
graph submission -> stream -> result -> revision correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordination.coordinator import Coordinator
from agi.runtime.graph import GraphSpec, NodeSpec
from agi.runtime.server import Runtime


def _stub_runtime(plan_graph: dict, critic_score: float = 1.0) -> Runtime:
    """Build a Runtime whose handlers don't call the API."""
    runtime = Runtime(num_workers=2)
    chat_counter = {"n": 0}

    def stub_plan(ctx, task):
        return {"result": {"graph": plan_graph}}

    def stub_chat(ctx, task):
        chat_counter["n"] += 1
        text = f"answer-{chat_counter['n']}: " + str(task.spec.input.get("message", ""))[:80]
        return {"result": {"text": text}}

    def stub_critique(ctx, task):
        return {"result": {"score": critic_score, "explanation": "stubbed"}}

    for w in runtime.workers:
        w.registry["plan"] = stub_plan
        w.registry["chat"] = stub_chat
        w.registry["critique"] = stub_critique
    return runtime


def test_coordinator_executes_simple_graph_and_passes_critic():
    plan = {
        "name": "demo",
        "nodes": [
            {"id": "research", "kind": "chat", "role": "executor",
             "input": {"message": "find facts about X"}, "depends_on": []},
            {"id": "summary", "kind": "chat", "role": "executor",
             "input": {"message": "summarize ${research.text}"},
             "depends_on": ["research"]},
        ],
    }
    runtime = _stub_runtime(plan, critic_score=0.9)
    try:
        coord = Coordinator(runtime, verify=True, max_iterations=1)
        report = coord.run("Tell me about X")
        assert report.status == "succeeded"
        assert "answer-2" in report.final_text
        assert report.critic_score == 0.9
        assert report.iterations == 1
    finally:
        runtime.shutdown()


def test_coordinator_retries_when_critic_rejects():
    plan = {
        "name": "demo",
        "nodes": [
            {"id": "draft", "kind": "chat", "role": "executor",
             "input": {"message": "answer the question"}, "depends_on": []},
        ],
    }
    runtime = _stub_runtime(plan, critic_score=0.2)
    try:
        coord = Coordinator(runtime, verify=True, verify_threshold=0.6,
                            max_iterations=2)
        report = coord.run("Tell me about X")
        # Two iterations because each fails the critic gate.
        assert report.iterations == 2
        assert report.status in ("failed_after_revisions", "succeeded")
    finally:
        runtime.shutdown()
