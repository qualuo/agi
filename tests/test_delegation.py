"""Delegation tool tests — exercise the parent/child Run relationship.

Uses the runtime with a fake agent factory that *itself* can call
`delegate` from a "tool" hook so we can test depth limits and result
roll-up without standing up the real Agent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.delegation import make_delegate_tool
from agi.runtime import Run, Runtime, RunStatus


class _FakeUsage:
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return 0.0


class EchoAgent:
    """Returns its task as output. The fake agent factory captures this so
    tests can assert end-to-end behavior."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.usage = _FakeUsage()
        self.model = "claude-opus-4-7"

    def chat(self, task: str) -> str:
        return f"echo: {task}"


def _echo_factory(run: Run, runtime: Runtime) -> EchoAgent:
    return EchoAgent(run)


class TestDelegate(unittest.TestCase):
    def test_delegate_runs_child_and_returns_result(self):
        rt = Runtime(agent_factory=_echo_factory)
        parent = rt.submit("parent task")
        parent.wait(timeout=2.0)
        _, delegate = make_delegate_tool(rt, parent_run_id=parent.id)
        result = delegate(task="hello child")
        self.assertEqual(result, "echo: hello child")

    def test_delegate_depth_limit(self):
        rt = Runtime(agent_factory=_echo_factory)
        parent = rt.submit("parent")
        parent.wait(timeout=2.0)
        # Manually set depth to the cap so the next delegate is refused.
        parent.metadata["depth"] = 3
        _, delegate = make_delegate_tool(rt, parent_run_id=parent.id, max_depth=3)
        result = delegate(task="nope")
        self.assertIn("error", result)
        self.assertIn("max", result)

    def test_delegate_emits_spawned_event(self):
        rt = Runtime(agent_factory=_echo_factory)
        parent = rt.submit("parent")
        parent.wait(timeout=2.0)
        _, delegate = make_delegate_tool(rt, parent_run_id=parent.id)
        delegate(task="child", role="researcher")
        types = [e.type for e in parent.events()]
        self.assertIn("delegate.spawned", types)
        spawned = [e for e in parent.events() if e.type == "delegate.spawned"][0]
        self.assertEqual(spawned.payload["role"], "researcher")
        self.assertIn("child_id", spawned.payload)

    def test_delegate_returns_child_failure_message(self):
        class FailingAgent(EchoAgent):
            def chat(self, task: str) -> str:
                raise RuntimeError("child boom")

        rt = Runtime(agent_factory=lambda run, runtime: FailingAgent(run))
        parent = rt.submit("parent")
        parent.wait(timeout=2.0)
        _, delegate = make_delegate_tool(rt, parent_run_id=parent.id)
        result = delegate(task="child")
        self.assertIn("failed", result)
        self.assertIn("child boom", result)

    def test_delegate_child_inherits_default_budget(self):
        rt = Runtime(agent_factory=_echo_factory)
        parent = rt.submit("parent")
        parent.wait(timeout=2.0)
        _, delegate = make_delegate_tool(rt, parent_run_id=parent.id, default_budget_usd=0.42)
        delegate(task="child")
        children = [r for r in rt.list_runs() if r.parent_id == parent.id]
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].cost_ceiling_usd, 0.42)


if __name__ == "__main__":
    unittest.main()
