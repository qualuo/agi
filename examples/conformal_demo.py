"""ConformalPredictor demo: distribution-free 95% upper-bound on cost
that a coordination engine can drive directly.

We simulate two tenants with very different cost distributions, feed
the receipt stream into a `ConformalPredictor`, and show that:

  - The marginal 95% interval has empirical coverage ≥ 95%.
  - Mondrian intervals give group-conditional coverage so neither
    tenant is systematically under-covered.
  - Adaptive Conformal Inference (Gibbs-Candès 2021) recovers
    long-run coverage when the data-generating process shifts.

Run:

    python examples/conformal_demo.py
"""
from __future__ import annotations

import random
import statistics

from agi.conformal import (
    METHOD_MONDRIAN,
    METHOD_SPLIT,
    ConformalPredictor,
    PredictionInterval,
)


def synth_cost(tenant: str, prediction: float, *, regime_shift: bool = False) -> float:
    base_sigma = 0.05 if tenant == "small" else 0.20
    drift = 0.30 if regime_shift else 0.0
    return max(0.0, prediction + drift + random.gauss(0, base_sigma))


def main() -> None:
    random.seed(0)
    cp = ConformalPredictor(target_coverage=0.95, adaptive=True, drift_threshold=0.05)

    # 1) Drain a fake receipt stream into the calibrator.
    for _ in range(1000):
        tenant = random.choice(("small", "large"))
        pred = random.uniform(0.10, 1.50)
        actual = synth_cost(tenant, pred)
        cp.record(
            features={"tenant": tenant, "estimated_cost": pred},
            prediction=pred,
            outcome=actual,
            group=tenant,
        )

    # 2) Score a held-out set marginally + group-conditionally.
    held_marginal_hits = 0
    held_widths = []
    per_tenant_hits = {"small": 0, "large": 0}
    per_tenant_n = {"small": 0, "large": 0}
    per_tenant_widths = {"small": [], "large": []}
    held_n = 500
    for _ in range(held_n):
        tenant = random.choice(("small", "large"))
        pred = random.uniform(0.10, 1.50)
        actual = synth_cost(tenant, pred)
        marg = cp.predict_interval(prediction=pred, method=METHOD_SPLIT)
        held_marginal_hits += int(marg.contains(actual))
        held_widths.append(marg.width)
        cond = cp.predict_interval(prediction=pred, method=METHOD_MONDRIAN, group=tenant)
        per_tenant_hits[tenant] += int(cond.contains(actual))
        per_tenant_n[tenant] += 1
        per_tenant_widths[tenant].append(cond.width)

    print("=== ConformalPredictor demo ===")
    print(f"Calibration set size:                {len(cp)}")
    print(f"Marginal empirical coverage:         {held_marginal_hits/held_n:.3f}  (target 0.950)")
    print(f"Mean marginal interval width:        ${statistics.mean(held_widths):.4f}")
    for t in ("small", "large"):
        emp = per_tenant_hits[t] / max(1, per_tenant_n[t])
        w = statistics.mean(per_tenant_widths[t]) if per_tenant_widths[t] else 0.0
        print(f"Mondrian coverage tenant={t!r}:".ljust(40)
              + f"{emp:.3f}  width=${w:.4f}")

    # 3) Now drive a regime shift through the adaptive loop and watch
    #    ACI recover coverage.
    last_interval: PredictionInterval | None = None
    for i in range(400):
        tenant = random.choice(("small", "large"))
        pred = random.uniform(0.10, 1.50)
        shifted = i > 100  # halfway through, costs rise systematically
        actual = synth_cost(tenant, pred, regime_shift=shifted)
        last_interval = cp.predict_interval(prediction=pred, method=METHOD_SPLIT)
        cp.update_adaptive(outcome=actual, last_interval=last_interval)

    rep = cp.report()
    print(f"\nAfter regime shift + adaptive online α:")
    print(f"  streaming empirical coverage:      {rep.empirical_coverage:.3f}")
    print(f"  drift detected:                    {rep.drift_detected}")
    print(f"  current ACI α:                     {cp._aci.alpha_t:.4f}  (started 0.05)")

    print("\nA coordination engine can now read these intervals to:")
    print("  - cap admission: defer if upper > budget_remaining")
    print("  - guardrail: pause forecaster if drift_detected")
    print("  - bill: charge the upper bound to the tenant on prepay")


if __name__ == "__main__":
    main()
