"""Scientist demo — automated discovery of closed-form physical laws.

What this shows
---------------

The ``Scientist`` primitive takes a stream of ``(x, y)`` observations
and returns a *closed-form mathematical expression* — a Law — that
explains ``y`` as a sparse linear combination of a library of basis
functions.  The selection is by Sequential Thresholded Least Squares
(Brunton-Proctor-Kutz 2016) over a grid of sparsity thresholds, and
the Law carries

* in-sample R² and out-of-sample R² on a held-out set;
* Akaike (AIC), Schwarz (BIC) and MDL ranking;
* per-coefficient bootstrap 95 % CIs;
* Meinshausen-Bühlmann stability-selection inclusion frequencies;
* the full Pareto frontier of (complexity, residual);
* a SHA-256 fingerprint chain for replay attestation.

The runtime is what an investor wants to see in a slide:
``Scientist.fit(observations) → "y ≈ -4.905·t² + 3.00·t + 100.00"``,
recovered from data alone, with a bound on overfitting.

Run with::

    python -m examples.scientist_demo

What you see
------------

1. Galileo's falling-body law recovered from noisy time-vs-altitude
   observations.
2. The full Pareto frontier (complexity vs. residual) showing the
   parsimony-fit tradeoff.
3. Bootstrap 95 % confidence intervals on every recovered coefficient.
4. Stability selection identifying the *robust support* — which basis
   functions survive resampling.
5. A two-input law with cross-term ``y = 3 x0 - 2 x1 + 0.5 x0 x1``.
6. A trigonometric law showing the basis library is composable.
"""
from __future__ import annotations

import math
import random
import textwrap

from agi.scientist import (
    Basis,
    Scientist,
    SELECT_AIC,
    SELECT_BIC,
    SELECT_MDL,
    default_library,
)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------
# 1. Galileo: y(t) = ½ g t² + v₀ t + h₀ from noisy data.
# ---------------------------------------------------------------------

def demo_falling_body() -> None:
    banner("1. Galileo's falling body — recover y(t) = ½ g t² + v₀ t + h₀")

    g, v0, h0 = -9.81, 3.0, 100.0  # truth
    rng = random.Random(0)
    n = 80
    ts = sorted(rng.uniform(0.0, 4.0) for _ in range(n))
    ys = [0.5 * g * t * t + v0 * t + h0 + rng.gauss(0.0, 0.2) for t in ts]

    sci = Scientist.create(input_dim=1, max_degree=3, seed=0)
    sci.observe_many([[t] for t in ts], ys)
    law = sci.fit(criterion=SELECT_AIC)
    print(f"\n   Recovered law: {law}")
    print(f"   R²            = {law.r2:.6f}")
    print(f"   k (terms)     = {law.k}")
    print(f"   AIC           = {law.aic:.2f}")
    print(f"   BIC           = {law.bic:.2f}")
    print(f"   MDL (bits/n)  = {law.mdl:.4f}")
    print(f"   AICc          = {sci.aicc_correction(law):.2f}")
    print(f"   λ (sparsity)  = {law.lam:.4g}")
    print(f"   Fingerprint   = {law.fingerprint[:16]}…")

    # Held-out R²
    ts_test = [rng.uniform(0.0, 4.0) for _ in range(40)]
    xs_test = [[t] for t in ts_test]
    ys_test = [0.5 * g * t * t + v0 * t + h0 for t in ts_test]
    r2_out = sci.evaluate_r2(xs_test, ys_test, law=law)
    print(f"   Out-of-sample R² = {r2_out:.6f}")


# ---------------------------------------------------------------------
# 2. Pareto frontier
# ---------------------------------------------------------------------

def demo_pareto() -> None:
    banner("2. Pareto frontier — complexity vs. residual sum of squares")

    rng = random.Random(1)
    n = 80
    xs = [[rng.uniform(-2.0, 2.0)] for _ in range(n)]
    ys = [0.5 * x[0] ** 2 - 1.0 * x[0] + 0.25 + rng.gauss(0.0, 0.05) for x in xs]

    sci = Scientist.create(input_dim=1, max_degree=4, seed=0)
    sci.observe_many(xs, ys)
    front = sci.pareto()
    print(f"\n   {'k':>3} {'λ':>10} {'RSS':>12} {'R²':>8} {'AIC':>10} {'BIC':>10} law")
    for p in front:
        print(
            f"   {p.k:>3} {p.lam:>10.4g} {p.rss:>12.4g} {p.r2:>8.4f} "
            f"{p.aic:>10.2f} {p.bic:>10.2f} {p.law}"
        )

    # Akaike model averaging weights:
    weights = sci.akaike_weights()
    if weights:
        best_lam = max(weights, key=lambda k: weights[k])
        print(f"\n   Best Akaike weight: λ = {best_lam:.4g} (w = {weights[best_lam]:.4f})")


# ---------------------------------------------------------------------
# 3. Bootstrap CI on coefficients
# ---------------------------------------------------------------------

def demo_bootstrap() -> None:
    banner("3. Bootstrap 95% CI — coefficient uncertainty under noise")

    rng = random.Random(2024)
    n = 200
    xs = [[rng.uniform(-3.0, 3.0)] for _ in range(n)]
    ys = [2.0 * x[0] + 1.0 + rng.gauss(0.0, 0.3) for x in xs]

    sci = Scientist.create(input_dim=1, max_degree=3, seed=42)
    sci.observe_many(xs, ys)
    law = sci.fit()
    boot = sci.bootstrap(law=law, n_resamples=200, alpha=0.05)
    print(f"\n   Recovered law: {law}")
    print(f"\n   {'term':>16} {'estimate':>12} {'CI lo':>12} {'CI hi':>12} {'SE':>10}")
    for t in law.terms:
        lo, hi = boot.ci[t.name]
        se = boot.se[t.name]
        marker = "" if lo <= 0.0 <= hi else "  ←✓"
        print(f"   {t.name:>16} {t.coefficient:>12.4f} {lo:>12.4f} {hi:>12.4f} {se:>10.4f}{marker}")
    print(
        textwrap.indent(
            "← ✓ marks terms whose CI excludes zero (significant at α = 0.05)",
            "   ",
        )
    )


# ---------------------------------------------------------------------
# 4. Stability selection
# ---------------------------------------------------------------------

def demo_stability() -> None:
    banner("4. Stability selection — Meinshausen-Bühlmann inclusion frequency")

    rng = random.Random(7)
    n = 150
    xs = [[rng.uniform(-2.0, 2.0)] for _ in range(n)]
    ys = [
        1.5 * x[0] + 0.7 * x[0] ** 2 + rng.gauss(0.0, 0.1)
        for x in xs
    ]
    sci = Scientist.create(input_dim=1, max_degree=4, seed=0)
    sci.observe_many(xs, ys)
    stab = sci.stability_selection(
        n_resamples=100,
        lam=0.05,
        subsample_fraction=0.5,
        pi_thr=0.6,
    )
    print(f"\n   Library size: {sci.library_size}")
    print(f"   Subsample size: {int(n * 0.5)} / {n}")
    print(f"   Stability threshold π = {stab.pi_thr}")
    print(f"\n   {'basis':>20} {'inclusion freq':>15}")
    for name, pi in sorted(stab.inclusion.items(), key=lambda kv: -kv[1]):
        marker = "  ✓ stable" if pi >= stab.pi_thr else ""
        print(f"   {name:>20} {pi:>15.3f}{marker}")


# ---------------------------------------------------------------------
# 5. Multi-input cross-term
# ---------------------------------------------------------------------

def demo_two_input() -> None:
    banner("5. Two-input law — y = 3 x0 - 2 x1 + 0.5 x0·x1")

    rng = random.Random(3)
    n = 200
    xs = [[rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)] for _ in range(n)]
    ys = [3.0 * x[0] - 2.0 * x[1] + 0.5 * x[0] * x[1] + rng.gauss(0.0, 0.05) for x in xs]

    sci = Scientist.create(input_dim=2, max_degree=2, seed=0)
    sci.observe_many(xs, ys)
    law = sci.fit()
    print(f"\n   Recovered: {law}")
    print(f"   R² = {law.r2:.6f}")
    print(f"   Out-of-sample prediction at (1, 1) → {sci.predict([1.0, 1.0], law=law):.4f}")
    print(f"   Truth                              → {3.0 - 2.0 + 0.5:.4f}")


# ---------------------------------------------------------------------
# 6. Trigonometric law
# ---------------------------------------------------------------------

def demo_trig() -> None:
    banner("6. Trigonometric law — y = 2 sin(x) + 0.5 cos(2x)")

    rng = random.Random(4)
    n = 200
    xs = [[rng.uniform(0.0, 6.0)] for _ in range(n)]
    ys = [2.0 * math.sin(x[0]) + 0.5 * math.cos(2.0 * x[0]) + rng.gauss(0.0, 0.05) for x in xs]

    sci = Scientist.create(
        input_dim=1,
        max_degree=1,
        include_trig=True,
        trig_frequencies=(1.0, 2.0),
        seed=0,
    )
    sci.observe_many(xs, ys)
    law = sci.fit()
    print(f"\n   Recovered: {law}")
    print(f"   R² = {law.r2:.6f}")
    print(f"   Library size: {sci.library_size}")


# ---------------------------------------------------------------------
# 7. MDL certificate
# ---------------------------------------------------------------------

def demo_mdl() -> None:
    banner("7. MDL certificate — two-part code length, in bits per sample")

    rng = random.Random(5)
    n = 100
    xs = [[rng.uniform(-3, 3)] for _ in range(n)]
    ys = [1.5 * x[0] - 0.4 + rng.gauss(0, 0.1) for x in xs]
    sci = Scientist.create(input_dim=1, max_degree=3, seed=0)
    sci.observe_many(xs, ys)
    law = sci.fit(criterion=SELECT_MDL)
    cert = sci.mdl_certificate(law)
    print(f"\n   Law: {law}")
    print(f"   model bits / sample = {cert['model_bits_per_sample']:.4f}")
    print(f"   data  bits / sample = {cert['data_bits_per_sample']:.4f}")
    print(f"   total bits / sample = {cert['total_bits_per_sample']:.4f}")
    print(f"   log₂ |library|      = {cert['library_log2_p']:.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    demo_falling_body()
    demo_pareto()
    demo_bootstrap()
    demo_stability()
    demo_two_input()
    demo_trig()
    demo_mdl()


if __name__ == "__main__":
    main()
