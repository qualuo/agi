"""Ranker demo — eight paired-comparison algorithms on the same data.

A coordination engine running a Chatbot-Arena-style judge pipeline needs
to turn pairwise verdicts ("judge said model A beat model B on this
prompt") into a *global* ranking with confidence.  This demo runs
Bradley-Terry MM / Bradley-Terry MAP / Plackett-Luce MM / Thurstone-
Mosteller / Elo / Glicko / Glicko-2 / TrueSkill on the same synthetic
data and compares:

  * recovered ranking vs. ground truth (Kendall τ);
  * log-likelihood + McFadden pseudo-R² (where defined);
  * identifiability diagnostic via Tarjan SCC;
  * P(D beats A) point estimate + anytime-valid 95% confidence interval;
  * Hajek-Oh-Xu (2014) sample-complexity bound for top-K recovery;
  * Tamper-evident fingerprint.

Run::

    python -m examples.ranker_demo
"""

from __future__ import annotations

import math
import random

from agi.ranker import (
    BRADLEY_TERRY_MAP,
    BRADLEY_TERRY_MM,
    ELO,
    GLICKO,
    GLICKO2,
    PLACKETT_LUCE_MM,
    THURSTONE_MM,
    TRUE_SKILL,
    Ranker,
    hox_sample_complexity,
    rank_correlation_kendall,
    sigmoid,
)


def stationary_demo() -> None:
    """Run eight algorithms on the same 4-item Bradley-Terry stream."""
    truth = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    items = list(truth.keys())
    T = 3_000
    rng = random.Random(0)
    stream: list[tuple[str, str]] = []
    for _ in range(T):
        a, b = rng.sample(items, 2)
        p = sigmoid(truth[a] - truth[b])
        if rng.random() < p:
            stream.append((a, b))
        else:
            stream.append((b, a))

    algos = [
        BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP, PLACKETT_LUCE_MM,
        THURSTONE_MM, ELO, GLICKO, GLICKO2, TRUE_SKILL,
    ]
    print(
        f"{'algorithm':<22} {'ranking':<22} {'τ':>5} "
        f"{'P(D>A)':>8} {'CI95':>16} {'ident':>5} {'fp':>10}"
    )
    print("-" * 96)
    true_order = sorted(items, key=lambda k: -truth[k])
    for algo in algos:
        R = Ranker(items=items, algorithm=algo, seed=0,
                   auto_fit=(algo == BRADLEY_TERRY_MM))
        for w, l in stream:
            R.observe_pair(w, l)
        if not R._dirty:                  # not auto-fit
            R.fit()
        order = R.rank()
        tau = rank_correlation_kendall(order, true_order)
        cp = R.compare("D", "A")
        rep = R.report(delta_bound=0.05)
        fp_short = rep.fingerprint.split(":")[1][:8]
        ci = f"[{cp.ci_low:.2f},{cp.ci_high:.2f}]"
        print(
            f"{algo:<22} {','.join(order):<22} {tau:>5.2f} "
            f"{cp.mean_win_prob:>8.3f} {ci:>16} "
            f"{('y' if rep.identifiable else 'n'):>5} {fp_short:>10}"
        )


def sample_complexity_demo() -> None:
    """Show the Hajek-Oh-Xu (2014) sample-complexity envelope."""
    print()
    print("Hajek-Oh-Xu (2014) sample-complexity bound for top-K recovery (δ=0.01):")
    print(f"{'K':>4} {'Δ_min=0.05':>12} {'Δ=0.10':>10} {'Δ=0.20':>10}  {'Δ=0.50':>10}")
    print("-" * 56)
    for K in [5, 10, 25, 50, 100]:
        print(
            f"{K:>4} "
            f"{hox_sample_complexity(K, 0.05, delta=0.01):>12,} "
            f"{hox_sample_complexity(K, 0.10, delta=0.01):>10,} "
            f"{hox_sample_complexity(K, 0.20, delta=0.01):>10,} "
            f"{hox_sample_complexity(K, 0.50, delta=0.01):>10,}"
        )


def online_demo() -> None:
    """Show online updates on a stream: ratings evolve match-by-match."""
    print()
    print("Online TrueSkill — ratings after every 200 matches:")
    items = ["A", "B", "C", "D"]
    truth = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    rng = random.Random(7)
    R = Ranker(items=items, algorithm=TRUE_SKILL)
    print(f"{'matches':>8} "
          f"{'  A (μ ± σ)':>14} {'  B':>14} {'  C':>14} {'  D':>14}")
    print("-" * 70)
    for t in range(2_000):
        a, b = rng.sample(items, 2)
        p = sigmoid(truth[a] - truth[b])
        if rng.random() < p:
            R.observe_pair(a, b)
        else:
            R.observe_pair(b, a)
        if (t + 1) % 500 == 0:
            line = f"{t+1:>8} "
            for n in items:
                rt = R.rate(n)
                line += f" {rt.mean:5.1f}±{rt.stderr:4.2f}  "
            print(line)


def top_k_demo() -> None:
    """PAC-certified top-2 with a tight gap vs. a clear gap."""
    print()
    print("PAC-certified top-K (δ=0.05):")
    items = ["A", "B", "C", "D"]
    tight = {"A": 0.0, "B": 0.05, "C": 0.10, "D": 0.15}
    clear = {"A": 0.0, "B": 0.05, "C": 2.0, "D": 3.0}
    rng = random.Random(11)
    for label, truth in [("tight gaps", tight), ("clear gaps", clear)]:
        R = Ranker(items=items, algorithm=BRADLEY_TERRY_MM)
        for _ in range(2_000):
            a, b = rng.sample(items, 2)
            p = sigmoid(truth[a] - truth[b])
            if rng.random() < p:
                R.observe_pair(a, b)
            else:
                R.observe_pair(b, a)
        dec = R.top_k(2)
        print(
            f"{label:<15} → top-2 = {','.join(dec.items):<10} "
            f"margin = {dec.margin:.3f} "
            f"pac_certified = {dec.pac_certified}"
        )


def main() -> None:
    stationary_demo()
    sample_complexity_demo()
    online_demo()
    top_k_demo()


if __name__ == "__main__":
    main()
