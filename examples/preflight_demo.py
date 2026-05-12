"""Preflight demo — what a coordination engine sees before dispatch.

Shows the four-state admission verdict (ADMIT / DEFER / DOWNGRADE /
REJECT) emerging from realistic configurations:

  1. Cold start: no history, heuristic prior used, ADMIT
  2. After 30 chats: estimator is calibrated, p_success is empirical
  3. Per-turn cost cap: opus over-budget → DOWNGRADE recommendation
  4. Tenant budget: daily cap exhausted → DEFER until reset
  5. Capacity: runtime at session cap → DEFER

Run:
    python examples/preflight_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.governance import PolicyManager, TenantLimits
from agi.memory import Memory
from agi.preflight import AdmissionAdvisor, PreflightEstimator
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.usage = FakeUsage()
        self.last_critic_score: float | None = 0.9
        self.extra_system: str | None = None

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.usage.input_tokens += 1500
        self.usage.output_tokens += 800
        self.usage.turns += 1
        return "done"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(estimate) -> None:
    print(json.dumps(estimate.to_dict(), indent=2, default=str))


def main() -> int:
    runtime = Runtime(
        memory=Memory(),
        skills=SkillLibrary(),
        agent_factory=FakeAgent,
        max_concurrent_sessions=3,
    )

    section("1. Cold start — heuristic prior")
    e1 = runtime.estimate("Summarize this PDF and extract action items.")
    print(f"  cost_usd:   ${e1.cost_usd:.4f}  (p10 ${e1.cost_p10_usd:.4f} … p90 ${e1.cost_p90_usd:.4f})")
    print(f"  duration:   {e1.duration_s:.1f}s  (p10 {e1.duration_p10_s:.1f}s … p90 {e1.duration_p90_s:.1f}s)")
    print(f"  p_success:  {e1.p_success:.2f}")
    print(f"  confidence: {e1.confidence}  ({e1.samples} samples)")
    print(f"  breakdown:  {e1.breakdown.input_tokens}in / {e1.breakdown.output_tokens}out")

    section("2. After 30 simulated chats — empirical calibration")
    sid = runtime.create_session(SessionConfig(use_skills=False))
    for _ in range(30):
        runtime.chat(sid, "Summarize this PDF and extract action items.")
    e2 = runtime.estimate("Summarize this PDF and extract action items.", session_id=sid)
    print(f"  cost_usd:   ${e2.cost_usd:.4f}  (p10 ${e2.cost_p10_usd:.4f} … p90 ${e2.cost_p90_usd:.4f})")
    print(f"  p_success:  {e2.p_success:.2f}")
    print(f"  confidence: {e2.confidence}  ({e2.samples} samples)")
    print(f"  → estimator now reflects measured cost, not heuristic prior")

    section("3. Per-turn cost cap — DOWNGRADE suggestion")
    advisor = AdmissionAdvisor(runtime.estimator, runtime=runtime, max_cost_per_turn_usd=0.08)
    advice = advisor.advise(
        prompt="x" * 5000,
        config=SessionConfig(model="claude-opus-4-7"),
    )
    print(f"  verdict: {advice.verdict}")
    print(f"  reason:  {advice.reason}")
    if advice.alternative:
        a = advice.alternative
        print(f"  alternative: model={a['model']}, est=${a['est_cost_usd']:.4f}, "
              f"savings=${a['expected_savings_usd']:.4f}")

    section("4. Tenant budget exhausted — DEFER until daily reset")
    policy = PolicyManager()
    policy.set_limits(TenantLimits(tenant_id="acme", daily_cost_usd=0.01))
    policy.commit(tenant_id="acme", cost_usd=0.009)  # nearly tapped out
    advisor2 = AdmissionAdvisor(runtime.estimator, policy=policy)
    advice2 = advisor2.advise(prompt="another expensive task", tenant_id="acme")
    print(f"  verdict: {advice2.verdict}")
    print(f"  reason:  {advice2.reason}")
    print(f"  governance_code: {advice2.governance_code}")
    print(f"  retry_after_s:   {advice2.retry_after_s}")

    section("5. Runtime at session cap — DEFER until capacity frees")
    # Fill remaining slots up to cap (one was already created in section 2).
    while True:
        try:
            runtime.create_session()
        except RuntimeError:
            break
    advice3 = runtime.advise("new work")
    print(f"  verdict: {advice3.verdict}")
    print(f"  reason:  {advice3.reason}")
    print(f"  retry_after_s: {advice3.retry_after_s}")

    section("6. Runtime capabilities snapshot — for coordinator UI")
    caps = runtime.capabilities()
    print(f"  total_sessions: {caps['total_sessions']}")
    print(f"  active_sessions: {caps['active_sessions']}")
    print(f"  preflight bins: {caps['preflight']['bin_count']}")
    print(f"  preflight samples: {caps['preflight']['total_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
