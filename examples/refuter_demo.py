"""Refuter — automated falsification as a runtime primitive.

This demo runs five scenarios end-to-end (no API key required):

  1. **Refute a known-false hypothesis** — find an x with x² < x in (0, 1).
  2. **Support a known-true hypothesis** — x² + 1 > 0; report the
     Clopper-Pearson finite-sample UCB on failure rate.
  3. **Metamorphic refutation** — refute a buggy sort that occasionally
     drops an element.
  4. **Bound refutation** — drive a continuous function toward a tight
     upper bound and refute the bound when the function exceeds it.
  5. **CEGIS** with the Refuter — synthesise a constant c such that
     ``max(L) <= c`` over a finite list space, by alternating
     refute → resynthesise (Solar-Lezama 2008).

Each report carries a SHA-256 fingerprint over `(predicate signature,
space, seed, witnesses, strategy counts)` — replay-verifiable by any
auditor.

Run:  python examples/refuter_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.refuter import (
    Refuter,
    ContinuousSpace,
    IntegerSpace,
    ListSpace,
    Product,
    cegis_loop,
    e_value_binomial,
    rule_of_three,
)


def section(title: str) -> None:
    line = "=" * 78
    print()
    print(line)
    print(f"  {title}")
    print(line)


def main() -> None:
    R = Refuter(seed=2026)

    # -------------------------------------------------------------------
    # 1. Refute a known-false hypothesis: x² ≥ x on (-3, 3)
    # -------------------------------------------------------------------
    section("1. Refute a known-false hypothesis: ∀x ∈ (-3, 3): x² ≥ x")
    def H1(x): return x["v"] ** 2 >= x["v"]
    rep = R.try_refute(
        H1,
        Product(v=ContinuousSpace(-3.0, 3.0, include_corners=False)),
        n_trials=1_000,
        alpha=0.05,
    )
    print(rep.support_claim())
    print(f"  strategy breakdown: {rep.strategy_counts}")
    print(f"  fingerprint: {rep.fingerprint}")

    # -------------------------------------------------------------------
    # 2. Support a known-true hypothesis: x² + 1 > 0
    # -------------------------------------------------------------------
    section("2. Support a known-true hypothesis: ∀x ∈ ℝ: x² + 1 > 0")
    def H2(x): return x["v"] ** 2 + 1 > 0
    rep = R.try_refute(
        H2,
        Product(v=ContinuousSpace(-100.0, 100.0, include_corners=False)),
        n_trials=2_000,
        alpha=0.05,
    )
    print(rep.support_claim())
    print(f"  failure rate Clopper-Pearson UCB: {rep.failure_rate_ucb:.4g}")
    print(f"  rule-of-three (≈3/n):             {rule_of_three(rep.n_trials):.4g}")

    # -------------------------------------------------------------------
    # 3. Metamorphic refutation: ``sorted(reversed(L)) == sorted(L)``
    #    on a buggy `sort` that drops one element when len > 2.
    # -------------------------------------------------------------------
    section("3. Metamorphic refutation: ∀L: sorted(reversed(L)) == sorted(L) (buggy_sort)")

    def buggy_sort(L: list[int]) -> list[int]:
        # An evil sorter — when input is long, drops the head.
        if len(L) > 2:
            return sorted(L[1:])
        return sorted(L)

    def relation(x, fx, x2, fx2):
        return fx == fx2

    rep = R.try_refute_relation(
        f=buggy_sort,
        relation=relation,
        space=ListSpace(IntegerSpace(0, 50), min_len=3, max_len=6),
        x_to_x2=lambda L: list(reversed(L)),
        n_trials=400,
    )
    if rep.refuted:
        print(f"REFUTED: sorted(reversed({rep.counterexample.x})) "
              f"= {sorted(reversed(rep.counterexample.x))} but "
              f"buggy_sort({rep.counterexample.x}) = {buggy_sort(rep.counterexample.x)}")
        print(f"  strategy: {rep.counterexample.strategy}, trial: {rep.counterexample.trial}")
    else:
        print("supported")

    # -------------------------------------------------------------------
    # 4. Bound refutation: ∀x ∈ [0, 5]: f(x) = x² - 4x + 5 ≤ 2.5
    #    (real min is 1 at x=2, but the parabola climbs to 5 at x=0).
    # -------------------------------------------------------------------
    section("4. Bound refutation: ∀x ∈ [0, 5]: x² - 4x + 5 ≤ 2.5")

    def scalar(x): return x["v"] ** 2 - 4 * x["v"] + 5

    rep = R.try_refute_bound(
        scalar=scalar,
        threshold=2.5,
        direction="<=",
        space=Product(v=ContinuousSpace(0.0, 5.0, include_corners=False)),
        n_trials=400,
    )
    if rep.refuted:
        cex_v = rep.counterexample.x["v"]
        print(f"REFUTED: f({cex_v:.4g}) = {scalar(rep.counterexample.x):.4g} > 2.5")
        print(f"  tightness margin: {rep.extra['tightness_margin']:.4g}")
        print(f"  found by {rep.counterexample.strategy} at trial {rep.counterexample.trial}")
    else:
        print(f"supported; tightness margin: {rep.extra.get('tightness_margin'):.4g}")

    # -------------------------------------------------------------------
    # 5. CEGIS scaffold: synthesise c such that ∀L: max(L) ≤ c.
    # -------------------------------------------------------------------
    section("5. CEGIS: synthesise c such that ∀L: max(L) ≤ c on lists of [0..15]")

    space = ListSpace(IntegerSpace(0, 15), min_len=1, max_len=3)

    def refute_c(c):
        return R.try_refute(lambda L: (max(L) <= c) if L else True,
                            space, n_trials=200)

    def resynth(c, cex):
        return max(c, max(cex.x))

    final_c, witnesses = cegis_loop(0, refute_c, resynth, max_rounds=20)
    print(f"final c = {final_c}")
    print(f"witnesses produced (showing first 5): "
          f"{[w.x for w in witnesses[:5]]}")
    print(f"total CEGIS rounds: {len(witnesses) + 1}")

    # -------------------------------------------------------------------
    # 6. Sequential / anytime-valid refutation of a rate claim
    # -------------------------------------------------------------------
    section("6. Sequential rate refutation (anytime-valid e-process)")

    import random as rnd

    def H_with_5pct_rate(x):
        # deterministic in x; fails ~5% of inputs
        return rnd.Random(hash(repr(x)) & 0xFFFFFFFF).random() > 0.05

    rep = R.refute_until(
        predicate=H_with_5pct_rate,
        space=Product(v=ContinuousSpace(-10.0, 10.0, include_corners=False)),
        p0=0.01,           # claim: failure rate ≤ 1%
        alpha=0.05,        # type-I error
        n_max=5_000,
        block_size=64,
    )
    print(rep.support_claim())
    print(f"  H₀: Pr[H fails] ≤ 0.01")
    print(f"  decision: {'rejected H₀ — rate evidently > 0.01' if rep.refuted or rep.e_value >= 20 else 'no rejection — data consistent with H₀'}")
    print(f"  evidence (e-value):     {rep.e_value:.4g}")
    print(f"  rejection threshold:    {1.0 / rep.alpha:.4g}  (Ville)")
    print(f"  empirical failure rate: {rep.failure_rate_emp:.4g}")
    print(f"  trials consumed:        {rep.n_trials} / 5000")


if __name__ == "__main__":
    main()
