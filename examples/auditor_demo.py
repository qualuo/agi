"""Auditor demo — multi-hypothesis testing as a runtime primitive.

Walks an investor through the Auditor end-to-end:

  1. Set the scene: a coordination engine is running 200 simultaneous
     A/B experiments. Without multiplicity control, the expected
     number of "winners" is ``200 · 0.05 = 10`` under the null.
  2. Run BH at α = 0.05 and watch the FDR-controlled rejection set
     shrink to the *real* winners.
  3. Show Holm, BY, and Storey side-by-side and explain when each
     wins. Holm for FWER under arbitrary dependence; BY for FDR
     under arbitrary dependence; Storey when nulls are abundant.
  4. Demonstrate e-BH for the case of correlated test statistics
     where BH formal validity is suspect.
  5. Switch to streaming mode: tests arrive one at a time and LORD-3
     decides each *online* with provable FDR guarantees over the
     entire (unbounded) future stream.
  6. Show the attestation receipt and event stream — every batch
     decision is reproducible and tamper-evident.

Run:
    python examples/auditor_demo.py

Stdlib-only, CPU-bound, ~200ms on a laptop.
"""
from __future__ import annotations

import random

from agi.attest import AttestationLedger, RuntimeAttestor
from agi.auditor import (
    AUDIT_DECIDED,
    Auditor,
    METHOD_BH,
    METHOD_BY,
    METHOD_EBH,
    METHOD_HOLM,
    METHOD_LORD,
    METHOD_STOREY,
)
from agi.events import EventBus


def simulated_experiments(seed: int = 42, m: int = 200, n_alt: int = 20):
    """Generate a realistic test scenario.

    ``m`` simultaneous tests; ``n_alt`` have a real effect (p ~ U[0, 0.001]),
    the remaining nulls have p ~ U[0, 1]. Returns (p-values, alt-mask).
    """
    r = random.Random(seed)
    ps = [r.random() for _ in range(m)]
    alt_idx = set(r.sample(range(m), n_alt))
    for i in alt_idx:
        ps[i] = r.uniform(0.0, 0.001)
    return ps, alt_idx


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def show_decision(name: str, rejected: list[bool], alt_idx: set[int]) -> None:
    n = len(rejected)
    true_pos = sum(1 for i, r in enumerate(rejected) if r and i in alt_idx)
    false_pos = sum(1 for i, r in enumerate(rejected) if r and i not in alt_idx)
    total = sum(rejected)
    fdp = false_pos / max(total, 1)
    power = true_pos / max(len(alt_idx), 1)
    print(
        f"  {name:<24s} rejected={total:>3d}  true_pos={true_pos:>3d}  "
        f"false_pos={false_pos:>3d}  FDP={fdp:.3f}  power={power:.2f}"
    )


def main() -> None:
    banner("Step 1: the multiplicity problem")
    print("A coordination engine runs 200 concurrent A/B experiments.")
    print("Of those, 20 have a real winner. The other 180 are nulls.")
    print("Naively at α=0.05, ~5% of nulls (≈9) become false positives.")
    ps, alt_idx = simulated_experiments()
    naive = sum(1 for p in ps if p <= 0.05)
    naive_fp = sum(1 for i, p in enumerate(ps) if p <= 0.05 and i not in alt_idx)
    print(f"  Naive α=0.05:           rejected={naive}  false_pos={naive_fp}")

    banner("Step 2-3: BH/BY/Holm/Storey side by side")
    bus = EventBus()
    ledger = AttestationLedger()
    attestor = RuntimeAttestor(ledger)
    auditor = Auditor(bus=bus, attestor=attestor)
    for i, p in enumerate(ps):
        auditor.observe(f"exp{i:03d}", p_value=p)

    for method in (METHOD_BH, METHOD_BY, METHOD_HOLM, METHOD_STOREY):
        rpt = auditor.decide(method=method, alpha=0.05)
        rejects = [rpt.decisions[f"exp{i:03d}"].rejected for i in range(len(ps))]
        show_decision(method, rejects, alt_idx)

    banner("Step 4: e-BH for correlated tests (arbitrary dependence)")
    # Shafer-Vovk "betting" e-value: e_c(p) = (1/c) · 1{p ≤ c}, which has
    # E[e | H₀] = (1/c)·c = 1, valid for any c ∈ (0,1). Best power when
    # c is set near the typical alt-p mass; here we average over several c
    # (a valid e-value averaging trick).
    auditor_e = Auditor()
    for i, p in enumerate(ps):
        # Mixed-betting calibrator averaging over c ∈ {0.001, 0.01, 0.05}
        e_components = [(1.0 / c) * (1.0 if p <= c else 0.0) for c in (0.001, 0.01, 0.05)]
        e = sum(e_components) / len(e_components)
        auditor_e.observe(f"exp{i:03d}", e_value=e)
    rpt = auditor_e.decide(method=METHOD_EBH, alpha=0.05)
    rejects = [rpt.decisions[f"exp{i:03d}"].rejected for i in range(len(ps))]
    show_decision("ebh", rejects, alt_idx)

    banner("Step 5: streaming LORD-3 online FDR")
    auditor_online = Auditor()
    rejected_online = []
    for i, p in enumerate(ps):
        is_r = auditor_online.decide_online(
            f"stream{i:03d}", p_value=p, method=METHOD_LORD, alpha=0.05
        )
        rejected_online.append(is_r)
    show_decision("lord (online)", rejected_online, alt_idx)

    banner("Step 6: attestation receipt")
    last_decided = bus.history(kind=AUDIT_DECIDED)[-1]
    print(f"  Last batch decision event: {last_decided.kind}")
    print(f"  Report id: {last_decided.data['report_id']}")
    print(f"  Method: {last_decided.data['method']}  α={last_decided.data['alpha']}")
    print(f"  n_tests={last_decided.data['n_tests']}  "
          f"n_rejected={last_decided.data['n_rejected']}")
    print(f"  Ledger entries: {len(ledger)}")
    print(f"  Tamper-evident chain head: {ledger.head_hash()[:24]}…")


if __name__ == "__main__":
    main()
