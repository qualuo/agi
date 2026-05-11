"""Tests for Runtime + Coordinator + protocol.

No network. The Agent is replaced with a fake whose `chat()` method
deterministically pretends to do work, accumulating fake usage so the
budget code paths are exercised end-to-end.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.costs import Usage
from agi.protocol import (
    Job,
    JobResult,
    JobStatus,
    ProgressEvent,
    RuntimeCapabilities,
    ToolDescriptor,
)
from agi.runtime import Runtime
from coord import Coordinator, RoutingPolicy
from coord.coordinator import CoordinatorBudget


class FakeAgent:
    """Stand-in for agi.Agent — same surface, no API."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        per_turn_input_tokens: int = 100_000,
        per_turn_output_tokens: int = 50_000,
        replies: list[str] | None = None,
        n_turns_to_finish: int = 1,
        raise_in_chat: Exception | None = None,
    ) -> None:
        self.model = model
        self.per_turn_in = per_turn_input_tokens
        self.per_turn_out = per_turn_output_tokens
        self.replies = list(replies) if replies else ["ok"]
        self.n_turns_to_finish = n_turns_to_finish
        self.raise_in_chat = raise_in_chat

        self.tool_schemas = [
            {"name": "read_file", "description": "Read a UTF-8 text file."},
            {"type": "web_search_20260209", "name": "web_search"},
        ]
        self.critic = None
        self.memory = None
        self.tracer = None
        self.messages: list[dict] = []
        self.usage = Usage()
        self.last_critic_score: float | None = None
        self.last_iterations = 0
        self.last_aborted = False
        self.last_turn_usage = Usage()

    def chat(self, prompt: str, max_iterations: int = 25, should_continue=None) -> str:
        if self.raise_in_chat:
            raise self.raise_in_chat
        self.messages.append({"role": "user", "content": prompt})
        turn_usage = Usage()
        text = ""
        self.last_iterations = 0
        self.last_aborted = False
        for i in range(min(max_iterations, self.n_turns_to_finish)):
            class FakeUsage:
                input_tokens = self.per_turn_in
                output_tokens = self.per_turn_out
                cache_creation_input_tokens = 0
                cache_read_input_tokens = 0
            self.usage.add(FakeUsage())
            turn_usage.add(FakeUsage())
            self.last_iterations += 1
            text = self.replies[min(i, len(self.replies) - 1)]
            self.messages.append({"role": "assistant", "content": text})
            if should_continue is not None and not should_continue(turn_usage):
                self.last_aborted = True
                break
        self.last_turn_usage = turn_usage
        return text


# -- Protocol round-trip ------------------------------------------------------


class TestProtocol(unittest.TestCase):
    def test_job_round_trip(self):
        j = Job(prompt="hi", max_cost_usd=0.25, output_contract="JSON")
        d = j.to_dict()
        j2 = Job.from_dict(d)
        self.assertEqual(j2.prompt, "hi")
        self.assertEqual(j2.max_cost_usd, 0.25)
        self.assertEqual(j2.output_contract, "JSON")
        self.assertEqual(j2.job_id, j.job_id)

    def test_job_drops_unknown_fields(self):
        j = Job.from_dict({"prompt": "hi", "future_field": 42})
        self.assertEqual(j.prompt, "hi")

    def test_job_result_status_round_trip(self):
        r = JobResult(job_id="x", status=JobStatus.SUCCEEDED, output="done", cost_usd=0.01)
        d = r.to_dict()
        self.assertEqual(d["status"], "succeeded")
        r2 = JobResult.from_dict(d)
        self.assertEqual(r2.status, JobStatus.SUCCEEDED)
        self.assertTrue(r2.succeeded)

    def test_capabilities_round_trip(self):
        cap = RuntimeCapabilities(
            runtime_id="rt-1",
            model="m",
            cost_per_1m_input_usd=1.0,
            cost_per_1m_output_usd=5.0,
            tools=[ToolDescriptor(name="t", description="d", server_side=True)],
            tags=["a", "b"],
        )
        d = cap.to_dict()
        self.assertIsInstance(d["tools"][0], dict)
        cap2 = RuntimeCapabilities.from_dict(d)
        self.assertEqual(cap2.tools[0].name, "t")
        self.assertTrue(cap2.tools[0].server_side)
        self.assertEqual(cap2.tags, ["a", "b"])


# -- Runtime ------------------------------------------------------------------


class TestRuntime(unittest.TestCase):
    def test_capabilities_lists_tools(self):
        rt = Runtime(agent_factory=lambda: FakeAgent())
        cap = rt.capabilities()
        self.assertEqual(cap.model, "claude-opus-4-7")
        names = {t.name for t in cap.tools}
        self.assertIn("read_file", names)
        self.assertIn("web_search", names)
        self.assertTrue(any(t.server_side for t in cap.tools if t.name == "web_search"))

    def test_submit_succeeds_and_charges(self):
        # tiny per-turn cost: 100k in + 50k out on opus
        # = 100000 * 5/1M + 50000 * 25/1M = 0.50 + 1.25 = $1.75
        rt = Runtime(agent_factory=lambda: FakeAgent(replies=["42"]))
        result = rt.submit(Job(prompt="2+2", max_cost_usd=10.0))
        self.assertEqual(result.status, JobStatus.SUCCEEDED)
        self.assertEqual(result.output, "42")
        self.assertGreater(result.cost_usd, 0)
        self.assertEqual(result.iterations, 1)
        self.assertIsNotNone(result.session_id)

    def test_submit_idempotent(self):
        rt = Runtime(agent_factory=lambda: FakeAgent(replies=["a"]))
        job = Job(prompt="hi", max_cost_usd=10.0)
        r1 = rt.submit(job)
        r2 = rt.submit(job)
        self.assertIs(r1, r2)

    def test_budget_exceeded_aborts(self):
        # Per-turn cost > $0.001 budget; should abort after first turn.
        rt = Runtime(agent_factory=lambda: FakeAgent(
            replies=["partial-1", "partial-2", "final"],
            n_turns_to_finish=3,
        ))
        result = rt.submit(Job(prompt="long", max_cost_usd=0.001))
        self.assertEqual(result.status, JobStatus.BUDGET_EXCEEDED)
        self.assertEqual(result.iterations, 1)  # aborted after the first turn
        self.assertIn("budget", (result.error or ""))

    def test_failure_caught_in_result(self):
        rt = Runtime(agent_factory=lambda: FakeAgent(raise_in_chat=RuntimeError("boom")))
        result = rt.submit(Job(prompt="x", max_cost_usd=10.0))
        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertIn("boom", result.error or "")

    def test_cancel_stops_in_flight(self):
        rt = Runtime(agent_factory=lambda: FakeAgent(
            replies=["a", "b", "c"], n_turns_to_finish=3,
        ))
        # Pre-cancel before submit; first should_continue returns False.
        job = Job(prompt="x", max_cost_usd=10.0)
        rt.cancel(job.job_id)
        result = rt.submit(job)
        self.assertEqual(result.status, JobStatus.CANCELLED)

    def test_snapshot_and_resume(self):
        rt = Runtime(agent_factory=lambda: FakeAgent(replies=["acknowledged"]))
        first = rt.submit(Job(prompt="codeword: zephyr", max_cost_usd=10.0))
        snap = rt.snapshot(first.session_id)
        new_session = rt.resume(snap)
        self.assertNotEqual(new_session, first.session_id)
        # Resumed agent has the conversation history of the original.
        resumed = rt._sessions[new_session]
        self.assertGreaterEqual(len(resumed.messages), 2)

    def test_output_contract_appears_in_prompt(self):
        captured: list[str] = []

        class CapAgent(FakeAgent):
            def chat(self2, prompt, max_iterations=25, should_continue=None):
                captured.append(prompt)
                return super().chat(prompt, max_iterations, should_continue)

        rt = Runtime(agent_factory=lambda: CapAgent(replies=["ok"]))
        rt.submit(Job(prompt="summarize", output_contract='{"summary": str}',
                      max_cost_usd=10.0))
        self.assertIn("summary", captured[0])

    def test_progress_events_emitted(self):
        events: list[ProgressEvent] = []
        rt = Runtime(agent_factory=lambda: FakeAgent(replies=["x"]))
        rt.subscribe(events.append)
        rt.submit(Job(prompt="x", max_cost_usd=10.0))
        kinds = [e.kind for e in events]
        self.assertIn("job_started", kinds)
        self.assertIn("budget_check", kinds)
        self.assertIn("job_finished", kinds)


# -- Coordinator --------------------------------------------------------------


class TestCoordinator(unittest.TestCase):
    def _two_runtime_coord(self) -> tuple[Coordinator, Runtime, Runtime]:
        cheap = Runtime(
            agent_factory=lambda: FakeAgent(model="claude-haiku-4-5", replies=["c"]),
            tags=["fast", "cheap"],
        )
        smart = Runtime(
            agent_factory=lambda: FakeAgent(model="claude-opus-4-7", replies=["s"]),
            tags=["frontier"],
        )
        c = Coordinator(budget=CoordinatorBudget(max_total_usd=10.0))
        c.register(cheap)
        c.register(smart)
        return c, cheap, smart

    def test_route_cheapest(self):
        c, cheap, smart = self._two_runtime_coord()
        chosen = c.route(Job(prompt="hi"), policy=RoutingPolicy.CHEAPEST)
        self.assertIs(chosen, cheap)

    def test_route_required_tags(self):
        c, cheap, smart = self._two_runtime_coord()
        chosen = c.route(Job(prompt="hi"), required_tags=["frontier"])
        self.assertIs(chosen, smart)
        chosen = c.route(Job(prompt="hi"), required_tags=["does-not-exist"])
        self.assertIsNone(chosen)

    def test_route_round_robin(self):
        c, cheap, smart = self._two_runtime_coord()
        a = c.route(Job(prompt="hi"), policy=RoutingPolicy.ROUND_ROBIN)
        b = c.route(Job(prompt="hi"), policy=RoutingPolicy.ROUND_ROBIN)
        self.assertIsNot(a, b)

    def test_run_charges_global_budget(self):
        c, cheap, smart = self._two_runtime_coord()
        before = c.budget.spent_usd
        result = c.run(Job(prompt="hi", max_cost_usd=10.0))
        self.assertEqual(result.status, JobStatus.SUCCEEDED)
        self.assertGreater(c.budget.spent_usd, before)
        self.assertEqual(c.stats.succeeded, 1)

    def test_run_rejects_when_global_budget_exhausted(self):
        c = Coordinator(budget=CoordinatorBudget(max_total_usd=0.0001))
        c.register(Runtime(agent_factory=lambda: FakeAgent()))
        result = c.run(Job(prompt="hi", max_cost_usd=1.00))
        self.assertEqual(result.status, JobStatus.BUDGET_EXCEEDED)
        self.assertEqual(c.stats.rejected_no_budget, 1)

    def test_run_no_route_returns_failed(self):
        c = Coordinator()
        result = c.run(Job(prompt="hi"))
        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertEqual(c.stats.rejected_no_route, 1)

    def test_race_returns_first_acceptable(self):
        c, cheap, smart = self._two_runtime_coord()
        winner = c.race(
            Job(prompt="hi", max_cost_usd=10.0),
            runtime_ids=[cheap.runtime_id, smart.runtime_id],
            accept=lambda r: r.succeeded,
        )
        self.assertEqual(winner.status, JobStatus.SUCCEEDED)
        self.assertEqual(winner.output, "c")

    def test_race_falls_through_on_unacceptable(self):
        # Both runtimes succeed, but accept rejects "c"; should fall to smart.
        c, cheap, smart = self._two_runtime_coord()
        winner = c.race(
            Job(prompt="hi", max_cost_usd=10.0),
            runtime_ids=[cheap.runtime_id, smart.runtime_id],
            accept=lambda r: r.output == "s",
        )
        self.assertEqual(winner.output, "s")


if __name__ == "__main__":
    unittest.main()
