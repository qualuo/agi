"""Reconciler demo — Aumann agreement across runtime primitives.

Run::

    python examples/reconciler_demo.py

Four scenarios show off the Reconciler runtime primitive:

  1. Three primitives disagree about which arm wins — Aumann iteration
     in finitely many rounds, with outlier identification and
     anytime-valid CIs.

  2. Linear vs logarithmic vs KL-barycenter pooling on the same beliefs.

  3. Calibration: each source's PIT-history runs the KS test; the
     primitive whose predictions are systematically over/under-confident
     gets a low p-value.

  4. Coordination-engine handshake — the coordinator queries the
     consensus belief, the outlier identity, and the audit-chain head
     before forwarding to a downstream primitive.
"""
from __future__ import annotations

import random

from agi.reconciler import (
    METHOD_AUMANN,
    METHOD_KL_BARYCENTER,
    METHOD_LINEAR,
    METHOD_LOGARITHMIC,
    Reconciler,
    ReconcilerConfig,
)


def _hdr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_arm_consensus() -> None:
    _hdr("1. Three primitives disagree about which arm wins")
    rec = Reconciler(ReconcilerConfig(method=METHOD_AUMANN))
    rec.register_topic("arm_a_wins", outcomes=("yes", "no"))
    rec.contribute(
        "arm_a_wins", source="bandit",   belief={"yes": 0.70, "no": 0.30}
    )
    rec.contribute(
        "arm_a_wins", source="bayesopt", belief={"yes": 0.60, "no": 0.40}
    )
    rec.contribute(
        "arm_a_wins", source="psrl",     belief={"yes": 0.65, "no": 0.35}
    )

    print("  contributed beliefs:")
    for s in rec.sources("arm_a_wins"):
        print(f"    {s.source_id:10}  yes={s.belief['yes']:.2f}  no={s.belief['no']:.2f}")

    report = rec.consensus("arm_a_wins")
    print(
        f"  Aumann consensus  →  yes={report.consensus['yes']:.4f}, "
        f"no={report.consensus['no']:.4f}"
    )
    print(
        f"  converged={report.converged} in {report.rounds} rounds; "
        f"effective_n={report.effective_n_sources:.2f}"
    )
    print(f"  per-source KL gap (KL(p_i ‖ q)):")
    for sid, kl in report.per_source_kl.items():
        print(f"    {sid:10}  KL = {kl:.4f}")
    print(
        f"  outlier:  {report.outlier[0]}  (KL gap = {report.outlier[1]:.4f})"
    )
    print(f"  HRMS 95% anytime-valid CI per outcome:")
    for o, (lo, hi) in report.confidence_interval.items():
        print(f"    {o}:  [{lo:.3f}, {hi:.3f}]")
    print(f"  audit-chain head: {report.fingerprint_hash[:24]}...")


def demo_methods() -> None:
    _hdr("2. Linear vs logarithmic vs KL-barycenter pooling")
    rec = Reconciler()
    rec.register_topic("rain", outcomes=("rain", "shine"))
    # One confident, two unconfident
    rec.contribute("rain", source="meteorologist", belief={"rain": 0.9, "shine": 0.1})
    rec.contribute("rain", source="model_a",       belief={"rain": 0.55, "shine": 0.45})
    rec.contribute("rain", source="model_b",       belief={"rain": 0.50, "shine": 0.50})

    for method in (METHOD_LINEAR, METHOD_LOGARITHMIC, METHOD_KL_BARYCENTER):
        r = rec.consensus("rain", method=method)
        print(
            f"  {method:14}  rain={r.consensus['rain']:.4f},  "
            f"shine={r.consensus['shine']:.4f}"
        )
    # Aumann ends up between linear and logarithmic.
    r_aumann = rec.consensus("rain", method=METHOD_AUMANN)
    print(
        f"  {METHOD_AUMANN:14}  rain={r_aumann.consensus['rain']:.4f},  "
        f"shine={r_aumann.consensus['shine']:.4f}  "
        f"(converged={r_aumann.converged} in {r_aumann.rounds} rounds)"
    )


def demo_calibration() -> None:
    _hdr("3. Calibration: which primitive's predictions are over-confident?")
    rec = Reconciler()
    rec.register_topic("rain_today", outcomes=("rain", "shine"))
    rng = random.Random(0)

    # Calibrated primitive: predictions agree with actual frequencies.
    for _ in range(200):
        p = rng.uniform(0.0, 1.0)
        outcome = "rain" if rng.random() < p else "shine"
        rec.contribute(
            "rain_today",
            source="calibrated_model",
            belief={"rain": p, "shine": 1 - p},
            realised=outcome,
        )

    # Over-confident primitive: always 0.99.
    for _ in range(200):
        outcome = "rain" if rng.random() < 0.5 else "shine"
        rec.contribute(
            "rain_today",
            source="over_confident_model",
            belief={"rain": 0.99, "shine": 0.01},
            realised=outcome,
        )

    cal_good = rec.calibration("rain_today", source="calibrated_model")
    cal_bad = rec.calibration("rain_today", source="over_confident_model")
    print(
        f"  calibrated_model:     KS={cal_good.ks_statistic:.3f}, "
        f"log-loss={cal_good.log_loss:.3f}  ← lower is better-calibrated"
    )
    print(
        f"  over_confident_model: KS={cal_bad.ks_statistic:.3f}, "
        f"log-loss={cal_bad.log_loss:.3f}"
    )
    if cal_good.log_loss < cal_bad.log_loss:
        print(
            "  → log-loss flags over-confident as poorly calibrated; "
            "the coordination engine downweights it."
        )
    print(
        "  (KS p-values are not meaningful for binary outcomes — log-loss is. "
        "Reconciler ships both so the caller picks the right test for the topic.)"
    )


def demo_coordination() -> None:
    _hdr("4. Coordination-engine handshake")
    rec = Reconciler(ReconcilerConfig(method=METHOD_AUMANN))
    rec.register_topic(
        "user_intent_is_research", outcomes=("research", "transactional")
    )
    rec.contribute(
        "user_intent_is_research",
        source="intent_classifier",
        belief={"research": 0.65, "transactional": 0.35},
        weight=2.0,
    )
    rec.contribute(
        "user_intent_is_research",
        source="goal_compiler",
        belief={"research": 0.75, "transactional": 0.25},
        weight=1.0,
    )
    rec.contribute(
        "user_intent_is_research",
        source="usage_history",
        belief={"research": 0.55, "transactional": 0.45},
        weight=0.5,
    )

    report = rec.consensus("user_intent_is_research")
    consensus = report.consensus
    print(f"  consensus P(research) = {consensus['research']:.4f}")
    print(f"  effective_n_sources    = {report.effective_n_sources:.2f}")
    print(f"  outlier (source most surprising under consensus): {report.outlier[0]}")
    print(f"  → coordinator routes by majority intent")
    if consensus["research"] > 0.5:
        print("    DISPATCH:  Researcher primitive (heavyweight, broad)")
    else:
        print("    DISPATCH:  Transactional primitive (fast, narrow)")
    print(f"  fingerprint head for compliance: {report.fingerprint_hash[:24]}...")

    ident = rec.identifiability_report("user_intent_is_research")
    print(
        f"  identifiability:  zero-mass outcomes={ident.zero_mass_outcomes}; "
        f"effective_n={ident.effective_n_sources:.2f}"
    )


def main() -> None:
    print("=" * 70)
    print("Reconciler — Aumann agreement as a runtime primitive.")
    print(
        "Pure stdlib • Aumann 1976 • Stone 1961 linear • Bordley 1982 log"
    )
    print(
        "Geanakoplos-Polemarchakis 1982 finite-time convergence • "
        "HRMS 2021 anytime-valid CI"
    )
    print("=" * 70)
    demo_arm_consensus()
    demo_methods()
    demo_calibration()
    demo_coordination()
    print()
    print("Done.  Compose Reconciler with:")
    print("  Bandit/BayesOpt/Imaginator/Forecaster/Predictor — contribute posteriors")
    print("  Auditor       — FDR-control across many topics' outliers")
    print("  DriftSentinel — CUSUM on consensus stability")
    print("  Aligner       — preferences over (topic, consensus) pairs")
    print("  Mentalist     — rationality-weighted counterparty contributions")
    print("  Conformal     — finite-sample-valid prediction sets on consensus")
    print("  Coordinator   — every Goal that depends on >1 primitive routes here")


if __name__ == "__main__":
    main()
