"""PrivacyAccountant — differential privacy as a runtime primitive.

Six end-to-end scenarios (no API key required):

  1. **Laplace release** with budget debit (basic composition).
  2. **Gaussian release** with Balle-Wang 2018 analytic σ vs. classical.
  3. **Budget exhaustion**: the odometer refuses an over-budget request.
  4. **Exponential mechanism** for private "best of N" selection.
  5. **Sparse Vector Technique** — answer many threshold queries while
     paying budget only on positives (Lyu-Su-Li 2017).
  6. **Rényi accountant** for tight composition over many releases
     (Mironov 2017; the right accountant for DP-SGD).

Plus an audit-trail demonstration:
  7. **Ledger hash + per-release fingerprints** — every privacy-
     respecting decision the runtime made is tamper-evident.

Run:  python examples/privacy_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.privacy import (
    BudgetExhausted,
    PrivacyAccountant,
    RenyiAccountant,
    analytic_gaussian_sigma,
    classical_gaussian_sigma,
)


def section(title: str) -> None:
    line = "=" * 78
    print()
    print(line)
    print(f"  {title}")
    print(line)


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Laplace + basic composition
    # -------------------------------------------------------------------
    section("1. Laplace mechanism + basic composition")
    A = PrivacyAccountant(epsilon=1.0, delta=1e-6, composition="basic", seed=42)
    true_count = 1_000
    print(f"true count = {true_count}, sensitivity = 1 (one-record-change)")
    for budget in (0.1, 0.2, 0.5):
        noisy = A.laplace(value=true_count, sensitivity=1.0, epsilon=budget)
        print(f"  ε={budget}: released {noisy:.3f}  (error {noisy-true_count:+.3f})")
    print(f"  spent ε = {A.spent_epsilon:.3f} / target {A.target_epsilon}")
    print(f"  remaining ε = {A.remaining_epsilon:.3f}")

    # -------------------------------------------------------------------
    # 2. Gaussian: analytic vs. classical calibration
    # -------------------------------------------------------------------
    section("2. Gaussian mechanism — analytic vs. classical calibration")
    for eps in (0.5, 1.0, 2.0, 5.0):
        sa = analytic_gaussian_sigma(1.0, eps, 1e-5)
        try:
            sc = classical_gaussian_sigma(1.0, eps, 1e-5)
            print(f"  ε={eps:>3}: σ_analytic = {sa:7.4f}  σ_classical = {sc:7.4f}  "
                  f"(saved {(1.0 - sa/sc)*100:.1f}% noise)")
        except Exception:
            print(f"  ε={eps:>3}: σ_analytic = {sa:7.4f}  σ_classical = invalid (ε > 1)")

    # -------------------------------------------------------------------
    # 3. Budget exhaustion
    # -------------------------------------------------------------------
    section("3. Budget exhaustion: the odometer trips")
    A = PrivacyAccountant(epsilon=0.5, seed=0)
    A.laplace(value=0, sensitivity=1, epsilon=0.3)
    print(f"  after first 0.3-release: spent={A.spent_epsilon}, remaining={A.remaining_epsilon}")
    try:
        A.laplace(value=0, sensitivity=1, epsilon=0.3)
    except BudgetExhausted as e:
        print(f"  second 0.3-release refused: {e}")

    # -------------------------------------------------------------------
    # 4. Exponential mechanism
    # -------------------------------------------------------------------
    section("4. Exponential mechanism — private 'best of N' selection")
    A = PrivacyAccountant(epsilon=5.0, seed=7)
    items = ["pizza", "tacos", "sushi", "burgers", "salad"]
    votes = {"pizza": 100, "tacos": 25, "sushi": 60, "burgers": 70, "salad": 5}
    print(f"  true vote counts: {votes}")
    print(f"  100 private 'best' draws at ε=0.5 each (sensitivity=1):")
    A_demo = PrivacyAccountant(epsilon=100.0, seed=7)
    picks = []
    for _ in range(100):
        c = A_demo.exponential(
            items, lambda r: votes[r], sensitivity=1.0, epsilon=0.5
        )
        picks.append(c)
    summary = {it: picks.count(it) for it in items}
    print(f"  → {summary}")
    print(f"  ε spent so far: {A_demo.spent_epsilon} / {A_demo.target_epsilon}")

    # -------------------------------------------------------------------
    # 5. Sparse Vector Technique
    # -------------------------------------------------------------------
    section("5. Sparse Vector Technique — pay only for positive answers")
    A = PrivacyAccountant(epsilon=20.0, seed=12)
    svt = A.sparse_vector(threshold=50.0, sensitivity=1.0,
                          epsilon_threshold=2.0, epsilon_answer=2.0,
                          max_positive=3)
    queries = [10, 20, 90, 30, 80, 100, 25, 5, 75, 50, 35]
    print(f"  queries (threshold = 50): {queries}")
    hits = []
    for q in queries:
        if svt.query(q):
            hits.append(q)
    print(f"  reported positives: {hits}  (max_positive=3)")
    print(f"  total ε spent: {A.spent_epsilon}")

    # -------------------------------------------------------------------
    # 6. Rényi accountant: tight composition over many releases
    # -------------------------------------------------------------------
    section("6. Rényi DP accountant — tight composition for many releases")
    R = RenyiAccountant()
    sigma = 5.0
    n_steps = 1_000
    for _ in range(n_steps):
        R.gaussian(sensitivity=1.0, sigma=sigma)
    delta = 1e-6
    eps, alpha_opt = R.to_epsilon_delta(delta=delta)
    # For comparison: basic composition (each Gaussian as ε(ε,δ)-DP via the
    # tight analytic σ inversion at the same δ — approximately one per step)
    print(f"  N = {n_steps} releases, σ = {sigma}, δ = {delta}")
    print(f"  Rényi-DP composed ε = {eps:.4f} at optimal α = {alpha_opt}")

    # -------------------------------------------------------------------
    # 7. Audit trail: per-release fingerprints + ledger hash
    # -------------------------------------------------------------------
    section("7. Audit trail — tamper-evident receipts")
    A = PrivacyAccountant(epsilon=1.0, delta=1e-6, seed=99, label="user-42")
    A.laplace(value=10, sensitivity=1, epsilon=0.2)
    A.gaussian(value=20, sensitivity=1, epsilon=0.3, delta=1e-7)
    for rel in A.releases:
        print(f"  {rel.mechanism:10s} ε={rel.epsilon:>5} δ={rel.delta:>8.0e}  "
              f"σ/b={rel.noise_param:.4g}   fp={rel.fingerprint[:16]}…")
    print(f"  ledger hash:  {A.ledger_hash()}")
    print(f"  remaining:    ε={A.remaining_epsilon}, δ={A.remaining_delta:.0e}")


if __name__ == "__main__":
    main()
