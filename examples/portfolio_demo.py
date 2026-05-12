"""Portfolio demo — fixed budget, many tickets, optimal allocation.

A coordination engine has 10 tasks of varying priority and exactly
$0.50 to spend. Which tasks get Opus, which get Sonnet, which get
Haiku, and which don't run at all?

`PortfolioOptimizer` answers this end-to-end:

  1. Forecast every (task × candidate model) cell via
     `PreflightEstimator`.
  2. Solve multiple-choice knapsack — pick one model per task (or
     skip) to maximize total expected successes within the budget.
  3. Optionally dispatch the plan via `RuntimeDriver.submit_portfolio`
     for live execution under shared accounting.

The demo also plots the budget → expected-value frontier: how many
more successful task completions does each extra dollar buy you? The
curve flattens once every task is already at its cheapest viable
model — a clear cue to operators about where the next dollar stops
paying off.

Run:
    python examples/portfolio_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.driver import RuntimeDriver, TicketRequest
from agi.memory import Memory
from agi.portfolio import PortfolioOptimizer
from agi.preflight import PreflightEstimator
from agi.runtime import Runtime
from agi.skills import SkillLibrary


# Reuse the same fake agent pattern the test suite uses so the demo
# runs without an Anthropic API key.
class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None
        self.messages: list = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return f"[{self.model}] done: {prompt[:30]}"

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


# A realistic mix of tasks at different intent lengths and (notional)
# priorities. Priority becomes the value_weight — high-priority work
# competes harder for the budget.
WORKLOAD = [
    ("Summarize the Q3 earnings call transcript",                    3.0),
    ("Triage 200 inbound support tickets and label by topic",        2.5),
    ("Generate a list of 50 unit-test ideas for the auth module",    1.0),
    ("Translate the user-facing settings page to Spanish",           1.0),
    ("Draft the weekly internal newsletter from the changelog",      0.8),
    ("Audit the API for OWASP top 10 issues",                        4.0),
    ("Suggest copy improvements for the landing page hero",          0.5),
    ("Extract action items from yesterday's all-hands transcript",   1.5),
    ("Write release notes from the merged-PR list",                  1.0),
    ("Propose tags for 1,000 newly uploaded blog posts",             1.2),
]


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def main() -> None:
    runtime = Runtime(
        memory=Memory(),
        skills=SkillLibrary(),
        agent_factory=FakeAgent,
    )
    driver = RuntimeDriver(runtime=runtime)

    requests = [TicketRequest(intent=intent) for intent, _ in WORKLOAD]
    weights = [w for _, w in WORKLOAD]

    banner("Plan-only quote at three budget tiers")
    for budget in (0.05, 0.20, 1.00):
        _, plan = driver.submit_portfolio(
            requests,
            total_budget_usd=budget,
            value_weights=weights,
            plan_only=True,
        )
        dispatched = sum(1 for a in plan.allocations if not a.skipped)
        print(
            f"\nbudget ${budget:>5.2f} | method={plan.method:>6} | "
            f"expected_cost=${plan.expected_cost_usd:.4f} | "
            f"util={plan.utilization * 100:5.1f}% | "
            f"E[success-sum]={plan.expected_p_success_sum:.2f} | "
            f"weighted_value={plan.expected_value:.2f} | "
            f"dispatched={dispatched}/{len(requests)}"
        )
        print("  --- per-task picks ---")
        for a in plan.allocations:
            tag = "skip" if a.skipped else a.chosen.model
            print(
                f"   [{a.request_index:>2}] w={a.value_weight:>4.1f} "
                f"E[$]={a.chosen.estimated_cost_usd:.4f} "
                f"E[p]={a.chosen.estimated_p_success:.2f} "
                f"-> {tag:<22} "
                f"intent={a.request.intent[:40]!r}"
            )

    banner("Budget → expected-value frontier")
    points = driver.portfolio.frontier(
        requests,
        budgets=[0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00],
        value_weights=weights,
    )
    print(f"  {'budget':>9} | {'spend':>9} | {'E[value]':>9} | {'E[succ]':>8} | {'dispatched':>10}")
    for p in points:
        print(
            f"  ${p.budget_usd:>7.2f} | "
            f"${p.expected_cost_usd:>7.4f} | "
            f"{p.expected_value:>9.3f} | "
            f"{p.expected_p_success_sum:>8.3f} | "
            f"{p.dispatched_count:>10}"
        )
    print(
        "\n  Read: each row is one operator choice. Marginal value per "
        "dollar shrinks\n  as budget grows — by ~$1 every task is already "
        "at a viable model and\n  further spend can only buy quality, not "
        "coverage."
    )

    banner("Dispatch the $0.50 plan and read back receipts")
    tickets, plan = driver.submit_portfolio(
        requests,
        total_budget_usd=0.50,
        value_weights=weights,
    )
    for i, t in enumerate(tickets):
        if t is None:
            print(f"  [{i:>2}] SKIPPED (no budget for {WORKLOAD[i][0][:48]!r})")
            continue
        r = t.result(timeout=5.0)
        print(
            f"  [{i:>2}] {r.status:<10} model={r.model:<22} "
            f"E[$]={r.estimated_cost_usd:.4f} actual=${r.actual_cost_usd:.4f} "
            f"text={(r.final_text or '')[:48]!r}"
        )

    print("\nPortfolio plan (JSON-serializable for audit):")
    print(json.dumps({
        "total_budget_usd": plan.total_budget_usd,
        "expected_cost_usd": plan.expected_cost_usd,
        "expected_value": plan.expected_value,
        "expected_p_success_sum": plan.expected_p_success_sum,
        "skipped_count": plan.skipped_count,
        "method": plan.method,
        "candidate_models": list(plan.candidate_models),
    }, indent=2))


if __name__ == "__main__":
    main()
