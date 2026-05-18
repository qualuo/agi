"""Attributor coordination demo — Attributor as a primitive a coordination
engine *discovers* via the Manifest, *invokes* on a contested decision,
and *audits* via the certificate ledger.

A coordination engine drives the runtime to deliver:

  Goal:  *Explain a contested ML decision and identify the smallest
         training-data intervention that would change it.*  A regulator
         (or product manager, or aggrieved end user) asked
         "why was this loan denied?".  The coordination engine must:

           1. *Discover* the primitive whose ``manifest.kind ==
              observability`` and whose tags overlap {bayesian,
              numerical, replay-verifiable} and whose summary mentions
              ``attribution`` / ``influence``.  That's Attributor.
           2. *Fit* the contested model on the same training data that
              produced the decision.
           3. *Diagnose* — Cook's distance, leverage, PRESS residuals.
              Flag any high-influence training points.
           4. *Attribute* the decision to specific training points
              via the first-order influence function.
           5. *Certify* with a decision-flip search: what is the
              smallest set of training points whose removal flips
              ACCEPT → REJECT, and what is its likelihood-ratio
              Bayes factor?
           6. *Replay* — every step is fingerprinted; the regulator can
              re-run the entire trace from the event log and
              independently verify the certificate.

This is the **runtime-as-coordination-surface** investor pitch: the
coordination engine doesn't know how Attributor computes influence —
it discovers that some primitive offers data attribution via the
manifest, calls it through a uniform API, reads a uniform certificate
(an e-value with a Jeffreys label), and acts.

Run::

  python examples/attributor_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.manifest import default_manifest  # noqa: E402
from agi.attributor import (  # noqa: E402
    Attributor,
    EXACT_LOO,
    INFLUENCE_FUNCTION,
    LINEAR,
    QUERY_PREDICTION,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Phase 1: discovery — the coordination engine asks the manifest
# "who handles data attribution?".  It does not know the primitive's name.


def phase_1_discovery() -> str:
    banner("Phase 1 — discovery via the manifest")
    m = default_manifest()
    print(f"  catalog size: {len(m.list())} primitives")
    print("  query: 'data attribution influence functions'")
    ranked = m.recommend("data attribution influence functions", k=3)
    print("\n  top matches:")
    for spec, score in ranked:
        print(f"    {score:>5.2f}  {spec.name:<14} kind={spec.kind:<14} "
              f"tags={','.join(spec.tags) or '(none)'}")
    chosen = ranked[0][0]
    print(f"\n  → coordination engine routes to '{chosen.name}'")
    print(f"     certificate class:  {chosen.certificate}")
    print(f"     composes_with:      {', '.join(chosen.composes_with)}")
    print(f"     demo:               {chosen.demo_path}")
    return chosen.name


# ---------------------------------------------------------------------------
# Phase 2: fit — same training data that produced the contested decision.


def phase_2_fit_contested_model() -> tuple[Attributor, list[list[float]], list[float], list[float]]:
    banner("Phase 2 — refit the contested model on the original training set")
    # Synthetic credit-score regression.  The clean relationship is
    # score = +0.4 · income − 0.2 · debt + noise.  We add one anomalous
    # training example that distorts the slope.
    rnd = random.Random(0xC0DE)
    X = [[1.0]]  # placeholder, overwritten
    X = []
    y = []
    for _ in range(25):
        income = rnd.gauss(0, 1)
        debt = rnd.gauss(0, 1)
        score = 0.4 * income - 0.2 * debt + rnd.gauss(0, 0.05)
        X.append([1.0, income, debt])
        y.append(score)
    # Anomalous training point — same input distribution, wildly wrong y.
    X.append([1.0, 0.5, -0.1])
    y.append(8.0)

    a = Attributor()
    fit = a.fit("credit_model", LINEAR, X=X, y=y)
    print(f"  trained on n={fit.n} points, p={fit.p} features")
    print(f"  β̂ = ({fit.theta[0]:+.3f}, {fit.theta[1]:+.3f}, {fit.theta[2]:+.3f})")
    print(f"  RSS = {fit.residual_sum_squares:.3f}")
    # The contested applicant.
    applicant_features = [1.0, 0.2, 0.0]
    pred = sum(applicant_features[j] * fit.theta[j] for j in range(3))
    decision = "ACCEPT" if pred > 0.3 else "REJECT"
    print(f"\n  contested applicant features: income={applicant_features[1]}, "
          f"debt={applicant_features[2]}")
    print(f"  baseline prediction:           ŷ = {pred:+.3f}")
    print(f"  baseline decision:             {decision!r}")
    return a, X, y, applicant_features


# ---------------------------------------------------------------------------
# Phase 3: diagnose — case-influence diagnostics.


def phase_3_diagnose(a: Attributor, n_show: int = 6) -> list[int]:
    banner("Phase 3 — case-influence diagnostics")
    diag = a.linear_diagnostics("credit_model")
    n = len(diag.cooks_distance)
    threshold = 4.0 / n
    print(f"  conventional 4/n threshold on Cook's D = {threshold:.3f}")
    flagged = [i for i, d in enumerate(diag.cooks_distance) if d > threshold]
    print(f"  observations flagged as high-influence: {flagged}")
    print()
    print(f"  {'i':>3} {'leverage':>10} {'press_resid':>12} {'cooks_D':>10}")
    # Show top-K rows by Cook's D.
    order = sorted(range(n), key=lambda i: -diag.cooks_distance[i])
    for i in order[:n_show]:
        flag = " ★" if diag.cooks_distance[i] > threshold else "  "
        print(f"  {i:>3}{flag}{diag.leverage[i]:>9.4f}  "
              f"{diag.press_residual[i]:>+12.3f}  {diag.cooks_distance[i]:>10.4f}")
    return flagged


# ---------------------------------------------------------------------------
# Phase 4: attribute — first-order influence on the contested prediction.


def phase_4_attribute(a: Attributor, applicant: list[float]) -> list[tuple[int, float]]:
    banner("Phase 4 — attribute the contested prediction")
    inf = a.influence("credit_model", query=QUERY_PREDICTION, test_point=applicant)
    print(f"  baseline prediction:        ŷ = {inf.q_baseline:+.3f}")
    print(f"  Σᵢ |influence| over n:      "
          f"{sum(abs(v) for v in inf.per_point):.3f}")
    print()
    print("  top-5 most-influential training points (by |IF|):")
    print(f"  {'i':>3}  {'influence':>10}  {'leverage':>10}")
    top5 = inf.most_influential(5)
    lev = a.leverage("credit_model")
    for i, v in top5:
        print(f"  {i:>3}  {v:>+10.4f}  {lev[i]:>10.4f}")
    return top5


# ---------------------------------------------------------------------------
# Phase 5: certify — decision-flip search with e-value certificate.


def phase_5_certify(a: Attributor, applicant: list[float]) -> None:
    banner("Phase 5 — decision-flip certificate")
    pred_full = sum(
        applicant[j] * a._state("credit_model").theta[j] for j in range(3)
    )
    # Acceptance threshold chosen so the baseline is ACCEPT but the
    # clean fit (without the anomalous training point) would REJECT —
    # the demo flips ACCEPT → REJECT.
    threshold = 0.3
    decision_fn = lambda theta: (
        "ACCEPT"
        if sum(applicant[j] * theta[j] for j in range(3)) > threshold
        else "REJECT"
    )
    flip = a.decision_flip(
        "credit_model",
        decision_fn=decision_fn,
        budget_k=5,
        query=QUERY_PREDICTION, test_point=applicant,
    )
    if flip.flipped:
        print(f"  flip found at K = {len(flip.minimal_set)}")
        print(f"  minimal set:           {flip.minimal_set}")
        print(f"  decision_full:         {flip.decision_full!r}")
        print(f"  decision_after:        {flip.decision_after!r}")
        print(f"  e-value (cf : full):   {flip.e_value:.4f}")
        print(f"  log₁₀ Bayes factor:    {flip.log10_bayes_factor:+.4f}")
        # Now show the counterfactual prediction.
        cf = a.counterfactual_refit(
            "credit_model", remove=flip.minimal_set,
            query=QUERY_PREDICTION, test_point=applicant,
        )
        print(f"\n  counterfactual θ:      "
              f"({cf.theta_counterfactual[0]:+.3f}, "
              f"{cf.theta_counterfactual[1]:+.3f}, "
              f"{cf.theta_counterfactual[2]:+.3f})")
        print(f"  counterfactual ŷ:      {cf.q_counterfactual:+.3f}")
        print(f"  Δ vs baseline:         {cf.delta_q:+.3f}")
        print(f"\n  regulator-facing receipt:")
        print(f"    'The decision ACCEPT was driven by training points "
              f"{flip.minimal_set}.")
        print(f"     Removing them and refitting flips the decision to REJECT.")
        print(f"     The counterfactual model fits the data with e-value "
              f"{flip.e_value:.3g}")
        print(f"     vs the full model — i.e., the counterfactual is ")
        if flip.e_value < 0.1:
            print("     less consistent with training data than the full fit.'")
        else:
            print("     comparably consistent with training data.'")
    else:
        print("  no flipping set found within budget — decision is robust.")
    print(f"\n  attestation fingerprint:  {flip.fingerprint[:16]}…")


# ---------------------------------------------------------------------------
# Phase 6: replay — verify the chain.


def phase_6_replay(a: Attributor) -> None:
    banner("Phase 6 — attestation replay")
    events = a.events()
    print(f"  events emitted:        {len(events)}")
    for ev in events[-6:]:
        print(f"    seq {ev.seq:>3}: {ev.kind:<32}  "
              f"hash = {ev.this_hash[:12]}…")
    print(f"\n  fingerprint:           {a.fingerprint[:16]}…")
    print(f"  chain valid:           {a.verify_chain()}")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    chosen = phase_1_discovery()
    assert chosen == "attributor", f"manifest discovery returned {chosen!r}"
    a, X, y, applicant = phase_2_fit_contested_model()
    flagged = phase_3_diagnose(a)
    top5 = phase_4_attribute(a, applicant)
    phase_5_certify(a, applicant)
    phase_6_replay(a)
    print("\n" + "=" * 72)
    print("  Coordination engine completed end-to-end attribution workflow.")
    print("=" * 72)
