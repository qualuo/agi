"""SLO contract demo — coordination-engine drives the runtime by objective.

The story this demo tells:

  A coordination engine no longer picks a model. It declares an objective:

      slo = TicketSLO(
          min_p_success=0.95, max_cost_usd=0.40,
          max_latency_s=30.0, hedge_policy="auto",
          refund_on_breach=1.0,
      )

  The runtime compiles the SLO into the cheapest plan that meets it:
  one model when feasible, a parallel hedge across two or three models
  when not. Receipts carry an SLO compliance verdict the operator can
  bill or refund against.

Three scenarios:

  1.  Easy SLO          → single haiku, $0.003-ish, compliant.
  2.  Tight quality SLO → hedge across haiku + sonnet, raises hedged
                          p_success to ~0.97, both children run in
                          parallel, the first success wins.
  3.  Tight budget SLO  → no candidate fits; compiler reports infeasible.

Then prints:

  - the frontier (budget → expected p_success / strategy)
  - the compliance ledger summary across all three submissions

Uses FakeAgent — no API key required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.contract import TicketSLO
from agi.driver import RuntimeDriver, TicketRequest
from agi.memory import Memory
from agi.runtime import Runtime
from agi.skills import SkillLibrary


# ----- FakeAgent: model-aware fake so the demo is honest about costs -----


class _FakeUsage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        if "haiku" in model:
            return self.input_tokens * 1e-6 + self.output_tokens * 5e-6
        if "sonnet" in model:
            return self.input_tokens * 3e-6 + self.output_tokens * 1.5e-5
        return self.input_tokens * 5e-6 + self.output_tokens * 2.5e-5


class _FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw):
        self.memory = memory
        self.model = model
        self.usage = _FakeUsage()
        self.last_critic_score = None
        self.extra_system = None
        self.messages = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        # Simulate a model-tier-dependent latency. Cheaper tiers respond
        # faster, which lets the hedge race produce a visible winner.
        if "haiku" in self.model:
            time.sleep(0.02)
            self.usage.input_tokens += 120
            self.usage.output_tokens += 40
            return f"[haiku] {prompt[:40]}"
        if "sonnet" in self.model:
            time.sleep(0.05)
            self.usage.input_tokens += 200
            self.usage.output_tokens += 100
            return f"[sonnet] {prompt[:40]}"
        time.sleep(0.10)
        self.usage.input_tokens += 400
        self.usage.output_tokens += 200
        return f"[opus] {prompt[:40]}"

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = _FakeUsage()


def _bar(label: str) -> None:
    print(f"\n{'=' * 8}  {label}  {'=' * 8}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="slo_demo_"))
    print(f"== SLO contract demo ==  (workspace: {tmp})\n")

    runtime = Runtime(
        memory=Memory(path=tmp / "memory.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=_FakeAgent,
    )
    driver = RuntimeDriver(
        runtime=runtime,
        compliance_path=tmp / "compliance.jsonl",
    )

    intents = [
        "Summarise this paragraph in two sentences.",
        "Draft a board update with a tight cost analysis.",
        "Solve the following constraint puzzle exactly.",
    ]

    # -------- Frontier preview ---------------------------------------
    _bar("Cost frontier for 'min_p_success=0.95'")
    rows = driver.frontier_for_slo(
        TicketRequest(intent=intents[1]),
        TicketSLO(min_p_success=0.95, hedge_policy="auto"),
        budgets=[0.001, 0.01, 0.05, 0.20, 1.0],
    )
    print(f"  {'budget':>8}  {'expected_p':>11}  {'expected_$':>11}  strategy  feasible  models")
    for r in rows:
        print(
            f"  ${r['budget_usd']:>6.4f}  {r['expected_p_success']:>11.4f}  "
            f"${r['expected_cost_usd']:>9.4f}  {r['strategy']:>8}  "
            f"{str(r['feasible']):>8}  {','.join(r['models'])}"
        )

    # -------- Scenario A: easy SLO -----------------------------------
    _bar("Scenario A — easy SLO  (default min_p_success, budget $1)")
    slo_a = TicketSLO(max_cost_usd=1.0)
    t_a = driver.submit_with_slo(TicketRequest(intent=intents[0]), slo_a)
    r_a = t_a.result(timeout=10.0)
    print(f"  strategy   = {r_a.plan.strategy}")
    print(f"  candidates = {[c.model for c in r_a.plan.candidates]}")
    print(f"  status     = {r_a.status}     slo_status = {r_a.slo_status}")
    print(f"  cost       = ${r_a.actual_cost_usd:.6f}   duration = {r_a.actual_duration_s:.3f}s")
    print(f"  final_text = {r_a.final_text!r}")

    # -------- Scenario B: tight quality SLO --------------------------
    _bar("Scenario B — tight quality SLO  (p≥0.99, budget $1, auto-hedge)")
    slo_b = TicketSLO(
        min_p_success=0.99, max_cost_usd=1.0,
        hedge_policy="auto", refund_on_breach=1.0,
    )
    t_b = driver.submit_with_slo(TicketRequest(intent=intents[1]), slo_b)
    r_b = t_b.result(timeout=10.0)
    print(f"  strategy        = {r_b.plan.strategy}")
    print(f"  candidates      = {[c.model for c in r_b.plan.candidates]}")
    print(f"  expected_p      = {r_b.plan.expected_p_success:.4f}")
    print(f"  winner          = {r_b.winner_model}")
    print(f"  status          = {r_b.status}     slo_status = {r_b.slo_status}")
    print(f"  aggregate_cost  = ${r_b.actual_cost_usd:.6f}  (sum across all hedged children)")
    print(f"  duration        = {r_b.actual_duration_s:.3f}s")
    print(f"  per-child       = " + ", ".join(
        f"{c.model}:${c.actual_cost_usd:.5f}" for c in r_b.children
    ))

    # -------- Scenario C: tight budget, infeasible -------------------
    _bar("Scenario C — tight budget  (p≥0.99, budget $0.0001 → infeasible)")
    slo_c = TicketSLO(
        min_p_success=0.99, max_cost_usd=0.0001,
        hedge_policy="auto", refund_on_breach=1.0,
    )
    t_c = driver.submit_with_slo(
        TicketRequest(intent=intents[2]), slo_c,
        dispatch_infeasible=False,   # reject up front rather than burn $
    )
    r_c = t_c.result(timeout=2.0)
    print(f"  status     = {r_c.status}")
    print(f"  slo_status = {r_c.slo_status}")
    print(f"  reason     = {r_c.plan.reason}")
    print(f"  cost       = ${r_c.actual_cost_usd:.6f}   (no spend; rejected by SLO compiler)")

    # -------- Compliance ledger summary ------------------------------
    _bar("Compliance ledger summary")
    summary = driver.compliance_report()
    print(json.dumps(summary, indent=2))

    print(f"\nReceipts + ledger persisted under: {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
