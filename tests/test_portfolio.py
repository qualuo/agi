"""Tests for the PortfolioOptimizer + RuntimeDriver.submit_portfolio.

The optimizer is the runtime's batch-allocation layer: given N tickets
and a single shared budget B, pick one model per ticket (or skip) to
maximize total expected p_success subject to the budget. These tests
pin the algorithm's correctness and the driver-wiring contract.
"""
from __future__ import annotations

import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.driver import (
    COMPLETED,
    RuntimeDriver,
    Ticket,
    TicketRequest,
)
from agi.memory import Memory
from agi.portfolio import (
    DEFAULT_CANDIDATE_MODELS,
    SKIP_MODEL,
    PortfolioOptimizer,
    PortfolioPlan,
)
from agi.preflight import PreflightEstimator
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# ---------- fakes (mirror test_driver.py) ----------


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    response = "fake-response"

    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None
        self.messages: list[Any] = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return FakeAgent.response

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


def _make_runtime() -> Runtime:
    tmp = Path(tempfile.mkdtemp())
    return Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )


# ---------- optimizer unit tests ----------


class TestPortfolioOptimizer(unittest.TestCase):

    def test_plan_picks_one_candidate_per_request(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent=f"task {i}") for i in range(3)]
        plan = opt.plan(reqs, total_budget_usd=1.0)
        self.assertIsInstance(plan, PortfolioPlan)
        self.assertEqual(len(plan.allocations), 3)
        for a in plan.allocations:
            # Each allocation has a chosen candidate; candidates list
            # includes one per model plus a skip slot.
            self.assertIn(a.chosen.model, set(DEFAULT_CANDIDATE_MODELS) | {SKIP_MODEL})
            self.assertEqual(
                len(a.candidates), len(DEFAULT_CANDIDATE_MODELS) + 1,
            )

    def test_plan_respects_total_budget(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent="hello world") for _ in range(4)]
        plan = opt.plan(reqs, total_budget_usd=0.30)
        self.assertLessEqual(plan.expected_cost_usd, 0.30 + 1e-6)
        self.assertGreaterEqual(plan.expected_value, 0.0)

    def test_zero_budget_skips_everything(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent=f"q{i}") for i in range(3)]
        plan = opt.plan(reqs, total_budget_usd=0.0)
        # All forecasts cost >0 → every request must be skipped.
        self.assertEqual(plan.skipped_count, 3)
        self.assertEqual(plan.expected_cost_usd, 0.0)
        for a in plan.allocations:
            self.assertTrue(a.skipped)

    def test_disallow_skip_forces_dispatch(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent="hi") for _ in range(2)]
        plan = opt.plan(reqs, total_budget_usd=10.0, allow_skip=False)
        self.assertEqual(plan.skipped_count, 0)
        for a in plan.allocations:
            self.assertFalse(a.skipped)
            self.assertNotEqual(a.chosen.model, SKIP_MODEL)

    def test_more_budget_never_lowers_expected_value(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent=f"task {i} body content") for i in range(5)]
        budgets = [0.01, 0.05, 0.20, 0.50, 1.0, 5.0]
        prev = -1.0
        for b in budgets:
            plan = opt.plan(reqs, total_budget_usd=b)
            self.assertGreaterEqual(
                plan.expected_value + 1e-9, prev,
                msg=f"value regressed at budget {b}",
            )
            prev = plan.expected_value

    def test_large_budget_picks_top_tier_models(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent="x") for _ in range(2)]
        plan = opt.plan(reqs, total_budget_usd=100.0)
        # With infinite budget, every request gets the highest-p_success
        # candidate, which is the most expensive model (opus-4-7 has the
        # highest prior). At minimum, none should be skipped.
        for a in plan.allocations:
            self.assertFalse(a.skipped)

    def test_value_weights_steer_allocation(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [
            TicketRequest(intent="low priority"),
            TicketRequest(intent="high priority"),
        ]
        # Tight budget — only one request can run expensively.
        # Weight the second request 100x.
        plan = opt.plan(
            reqs,
            total_budget_usd=0.05,
            value_weights=[1.0, 100.0],
        )
        # If anything is dispatched, the high-weight request should be it.
        dispatched = [a for a in plan.allocations if not a.skipped]
        if dispatched:
            self.assertTrue(
                any(a.request_index == 1 for a in dispatched),
                "high-value request was skipped while a low-value one ran",
            )

    def test_frontier_is_monotone_in_budget(self):
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent=f"q {i}") for i in range(4)]
        points = opt.frontier(reqs, budgets=[0.0, 0.02, 0.10, 0.50, 2.0])
        for prev, cur in zip(points, points[1:]):
            self.assertGreaterEqual(
                cur.expected_value + 1e-9, prev.expected_value,
            )
            self.assertGreaterEqual(
                cur.expected_p_success_sum + 1e-9, prev.expected_p_success_sum,
            )

    def test_to_dict_is_json_safe(self):
        import json
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent="ping")]
        plan = opt.plan(reqs, total_budget_usd=0.10)
        s = json.dumps(plan.to_dict(), default=str)
        data = json.loads(s)
        self.assertIn("allocations", data)
        self.assertEqual(data["total_budget_usd"], 0.10)
        self.assertEqual(len(data["allocations"]), 1)

    def test_custom_candidate_models_overrides_default(self):
        opt = PortfolioOptimizer(
            PreflightEstimator(),
            candidate_models=["claude-haiku-4-5"],
        )
        reqs = [TicketRequest(intent="x")]
        plan = opt.plan(reqs, total_budget_usd=1.0)
        # Only haiku or skip permitted.
        for a in plan.allocations:
            self.assertIn(a.chosen.model, {"claude-haiku-4-5", SKIP_MODEL})

    def test_greedy_matches_dp_within_tolerance(self):
        """The greedy fallback should be near-optimal on typical inputs."""
        opt = PortfolioOptimizer(PreflightEstimator())
        reqs = [TicketRequest(intent=f"task {i}") for i in range(6)]
        dp = opt.plan(reqs, total_budget_usd=0.30, method="dp")
        greedy = opt.plan(reqs, total_budget_usd=0.30, method="greedy")
        # Greedy is at worst one upgrade away from optimal; with
        # uniform-ish costs/values it usually matches.
        self.assertGreaterEqual(dp.expected_value + 1e-9, greedy.expected_value)
        # Greedy should be within 20% of DP's value (loose envelope).
        if dp.expected_value > 0:
            ratio = greedy.expected_value / dp.expected_value
            self.assertGreater(ratio, 0.8, f"greedy={greedy.expected_value} dp={dp.expected_value}")


# ---------- driver wiring ----------


class TestDriverSubmitPortfolio(unittest.TestCase):

    def test_plan_only_returns_plan_no_tickets(self):
        rt = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        reqs = [TicketRequest(intent="x"), TicketRequest(intent="y")]
        tickets, plan = driver.submit_portfolio(
            reqs, total_budget_usd=0.20, plan_only=True,
        )
        self.assertEqual(tickets, [])
        self.assertIsInstance(plan, PortfolioPlan)
        self.assertEqual(len(plan.allocations), 2)

    def test_submit_portfolio_dispatches_non_skipped(self):
        rt = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        reqs = [TicketRequest(intent=f"task {i}") for i in range(3)]
        tickets, plan = driver.submit_portfolio(
            reqs, total_budget_usd=2.0, allow_skip=False,
        )
        # allow_skip=False → all three are dispatched as real tickets.
        self.assertEqual(len(tickets), 3)
        for t in tickets:
            self.assertIsNotNone(t)
        receipts = [t.result(timeout=5.0) for t in tickets if t is not None]
        for r in receipts:
            self.assertEqual(r.status, COMPLETED)
        # Each receipt's metadata records the optimizer's pick.
        for r, alloc in zip(receipts, plan.allocations):
            self.assertEqual(r.model, alloc.chosen.model)

    def test_submit_portfolio_skips_under_zero_budget(self):
        rt = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        reqs = [TicketRequest(intent="x"), TicketRequest(intent="y")]
        tickets, plan = driver.submit_portfolio(reqs, total_budget_usd=0.0)
        # Everything skipped → tickets is [None, None]; plan says so.
        self.assertEqual(tickets, [None, None])
        self.assertEqual(plan.skipped_count, 2)

    def test_driver_stats_track_batches(self):
        rt = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        reqs = [TicketRequest(intent="hello") for _ in range(2)]
        driver.submit_portfolio(reqs, total_budget_usd=1.0, allow_skip=False)
        stats = driver.stats()
        self.assertEqual(stats["portfolio_batches"], 1)

    def test_driver_portfolio_lazy_singleton(self):
        rt = _make_runtime()
        driver = RuntimeDriver(runtime=rt)
        opt1 = driver.portfolio
        opt2 = driver.portfolio
        self.assertIs(opt1, opt2)
        self.assertIsInstance(opt1, PortfolioOptimizer)


if __name__ == "__main__":
    unittest.main()
