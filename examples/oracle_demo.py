"""TicketOracle demo — counterfactual replay + auto-tuning admission.

The story this demo tells:

  After a runtime has dispatched a few hundred tickets, the receipts
  themselves contain the answer to "which knobs would have saved us
  money?" The oracle replays every receipt under alternative
  `(min_p_success, max_cost_per_turn_usd)` knobs, picks the best,
  and (optionally) applies it back to the live `AdmissionAdvisor`.

  Investors care about three things from this:

    1. Every routing decision is **causally durable** — the trace on
       each `Receipt` is enough to replay alternate worlds.
    2. The system **self-tunes**: the longer it runs, the cheaper its
       admission policy becomes for the same hit rate.
    3. Operators can ask **what-if** questions ("provider raises
       prices 20% next quarter — what does our P&L look like?")
       without booking any real spend.

Three scenes:

  Scene 1 — Mixed historical workload
      6 cheap tickets that succeed, 6 expensive tickets that failed
      in production. The runtime is bleeding money on the bad bucket.
  Scene 2 — Oracle recommendation
      `recommend()` searches the knob grid and proposes a cost cap
      that would have rejected the bad bucket before spend.
  Scene 3 — What-if: provider price hike
      Layer a 1.25x cost multiplier on top of the same population to
      project the financial impact of an upstream pricing shock.
  Scene 4 — Auto-tune & verify
      Apply the recommendation; submit a new ticket; show that the
      live advisor reflects the new knobs.

Uses FakeAgent so no API key is required.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.driver import (
    COMPLETED,
    D_ADMISSION,
    D_ESTIMATE,
    Decision,
    FAILED,
    Receipt,
    RuntimeDriver,
    TicketRequest,
)
from agi.memory import Memory
from agi.oracle import PolicyKnobs
from agi.preflight import ADMIT
from agi.runtime import Runtime
from agi.skills import SkillLibrary


class _FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 5e-6 + self.output_tokens * 25e-6


class _FakeAgent:
    response = "ok"

    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = _FakeUsage()
        self.last_critic_score = None
        self.extra_system = None
        self.messages: list = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.usage.input_tokens += 200
        self.usage.output_tokens += 80
        return _FakeAgent.response

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = _FakeUsage()


def _synth_receipt(
    *,
    ticket_id: str,
    est_cost: float,
    est_cost_p90: float,
    actual_cost: float,
    est_p_success: float,
    status: str,
    model: str = "claude-opus-4-7",
) -> Receipt:
    """A receipt with a decision trace the oracle can replay."""
    now = time.time()
    return Receipt(
        ticket_id=ticket_id,
        intent=f"intent for {ticket_id}",
        status=status,
        tenant_id="acme",
        model=model,
        actual_cost_usd=actual_cost,
        estimated_cost_usd=est_cost,
        estimated_p_success=est_p_success,
        decisions=[
            Decision(
                kind=D_ESTIMATE,
                ts=now,
                payload={
                    "cost_usd": est_cost,
                    "cost_p10_usd": est_cost * 0.6,
                    "cost_p90_usd": est_cost_p90,
                    "duration_s": 5.0,
                    "p_success": est_p_success,
                    "confidence": "high",
                    "samples": 60,
                    "model": model,
                },
            ),
            Decision(
                kind=D_ADMISSION,
                ts=now,
                payload={"verdict": ADMIT, "reason": "ok"},
            ),
        ],
    )


def _print_header(text: str) -> None:
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76)


def main() -> None:
    rt = Runtime(
        memory=Memory(),
        skills=SkillLibrary(),
        agent_factory=_FakeAgent,
    )
    driver = RuntimeDriver(runtime=rt)
    oracle = driver.oracle

    # ----- Scene 1: historical workload --------------------------------
    _print_header("Scene 1 — Historical workload (mixed)")
    for i in range(6):
        oracle.record(_synth_receipt(
            ticket_id=f"cheap-{i}",
            est_cost=0.02,
            est_cost_p90=0.03,
            actual_cost=0.018,
            est_p_success=0.95,
            status=COMPLETED,
        ))
    for i in range(6):
        oracle.record(_synth_receipt(
            ticket_id=f"big-{i}",
            est_cost=0.60,
            est_cost_p90=1.50,
            actual_cost=0.85,
            est_p_success=0.45,
            status=FAILED,
        ))
    baseline = oracle.replay(oracle.receipts(), oracle.baseline)
    print(f"Tickets in window  : {baseline.n_tickets}")
    print(f"Baseline spend     : ${baseline.baseline_cost_usd:.2f}")
    print(f"Baseline hit-rate  : {baseline.baseline_success_rate:.0%}")
    print(f"Verdict mix        : all ADMIT (no policy filter)")

    # ----- Scene 2: oracle recommends ----------------------------------
    _print_header("Scene 2 — Oracle recommends")
    rec = oracle.recommend()
    if rec is None:
        print("Oracle has too little data to draw a conclusion.")
        return
    print(rec.summary)
    print(f"  proposed knobs   : {rec.knobs.to_dict()}")
    print(f"  baseline knobs   : {rec.baseline_knobs.to_dict()}")
    print(f"  verdict changes  : {rec.improvement.verdict_changes}")
    print(
        f"  projected spend  : ${rec.improvement.alt_cost_usd:.2f} "
        f"(was ${rec.improvement.baseline_cost_usd:.2f})"
    )
    print(
        f"  alt hit-rate     : {rec.improvement.alt_success_rate:.0%} "
        f"(baseline {rec.improvement.baseline_success_rate:.0%})"
    )

    # ----- Scene 3: what-if pricing shock ------------------------------
    _print_header("Scene 3 — What-if: provider raises prices 25%")
    wi = oracle.what_if(cost_multiplier=1.25)
    print(f"Tickets in window  : {wi.n_tickets}")
    print(f"Spend at current pricing : ${wi.baseline_cost_usd:.2f}")
    print(f"Spend with +25% shock    : ${wi.shocked_cost_usd:.2f}")
    print(f"Δ over the window        : ${wi.projected_cost_delta_usd:+.2f}")

    # ----- Scene 4: auto-tune & confirm --------------------------------
    _print_header("Scene 4 — Auto-tune & confirm")
    applied = oracle.auto_tune(driver, min_savings_usd=0.10)
    if applied is None:
        print("Auto-tune declined: no recommendation cleared the savings floor.")
        return
    print(f"Applied knobs : {applied.knobs.to_dict()}")
    print(
        f"Advisor knobs : min_p_success={driver.advisor._min_p_success}, "
        f"max_cost_per_turn_usd={driver.advisor._max_cost_per_turn_usd}"
    )
    # A new live ticket would now be evaluated under the new knobs.
    receipt = driver.submit_sync(TicketRequest(intent="hello after tune"), timeout=5.0)
    print(f"\nNew ticket status     : {receipt.status}")
    print(f"New ticket model      : {receipt.model}")
    print(f"New ticket cost (USD) : ${receipt.actual_cost_usd:.4f}")


if __name__ == "__main__":
    main()
