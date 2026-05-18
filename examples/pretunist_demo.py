"""Pretunist — Test-Time Training as a runtime primitive.

A coordination engine drives the runtime to deliver:

  Goal:  When the *current task* is far from the base policy's training
         distribution, adapt at inference time on a small support set
         (a handful of in-context demos), produce a provably-bounded
         specialised predictor, and hand the coordinator either a
         certified prediction *or* a defensible abstention.

         No GPU.  No torch.  Closed-form ridge fit + Cholesky.  PAC-Bayes
         bound.  KL-budget projection.  Anytime-valid e-process on the
         "is the adapter genuinely helping?" question.

This is the runtime primitive that operationalises the ARC-AGI 2024
prize-winning technique (Akyürek et al. 2024, *The Surprising
Effectiveness of Test-Time Training for Few-Shot Learning*) — a small
adapter fit on the test instance's own support outperforms an order-of-
magnitude bigger frozen model.  The runtime hands the coordinator the
closed-form fit, a McAllester-style generalisation guarantee, and a
mixture-martingale e-process that monitors whether to keep adapting.

This demo shows the investor-grade test-time-adaptation story in a
single runnable script (no API key required, pure stdlib):

  1. PretunistConfig            → declare adapter dim / λ / KL budget
  2. Pretunist.set_base         → install the base policy θ_0
  3. Pretunist.observe_support  → ingest (x_i, y_i) demos
  4. Pretunist.adapt(query)     → closed-form ridge solve + project
  5. Pretunist.should_abstain   → leverage / variance / KL gates
  6. Pretunist.certify          → PAC-Bayes generalisation cert
  7. Pretunist.report           → coordinator-readable summary
  8. Pretunist.snapshot/restore → ship adapter across the federation

Run::

  python examples/pretunist_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.pretunist import (  # noqa: E402
    ABSTAIN_LEVERAGE,
    ABSTAIN_VARIANCE,
    Pretunist,
    PretunistConfig,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def make_task(d: int, n_demo: int, n_query: int, *, seed: int, noise: float = 0.05):
    """A new task is a fresh ground-truth weight vector w*.  The runtime
    has *never* seen this w* — we simulate the situation where the base
    policy is generic and the test-time task is OOD."""
    rng = random.Random(seed)
    w_star = [rng.gauss(0.0, 1.0) for _ in range(d)]
    def gen(n):
        xs, ys = [], []
        for _ in range(n):
            x = [rng.gauss(0.0, 1.0) for _ in range(d)]
            y = sum(wi * xi for wi, xi in zip(w_star, x)) + rng.gauss(0.0, noise)
            xs.append(x)
            ys.append([y])
        return xs, ys
    demos = gen(n_demo)
    queries = gen(n_query)
    return w_star, demos, queries


def mse(predictions: list[tuple[float, ...]], targets: list[list[float]]) -> float:
    if not predictions:
        return float("inf")
    s = 0.0
    n = 0
    for p, y in zip(predictions, targets):
        for i in range(len(p)):
            s += (p[i] - y[i]) ** 2
            n += 1
    return s / max(1, n)


def main() -> int:
    banner("0 — Setup")
    d = 8
    n_demo = 60
    n_query = 25
    seed = 2026

    w_star, (X_demo, y_demo), (X_query, y_query) = make_task(
        d=d, n_demo=n_demo, n_query=n_query, seed=seed,
    )
    # Base policy = "best guess before seeing the task" = zero vector.
    base = [[0.0] for _ in range(d)]

    print(f"  task dim          d = {d}")
    print(f"  demos             n = {n_demo}  (in-context support)")
    print(f"  queries           m = {n_query}")
    print(f"  base policy       θ_0 = 0  (no prior knowledge of this task)")
    print(f"  true target       w* = [{w_star[0]:+.3f}, {w_star[1]:+.3f}, …]")

    banner("1 — Base policy on the queries (NO adaptation)")
    # Naive baseline: predict 0 for everything (because base is zero).
    base_preds = [tuple(0.0 for _ in range(1)) for _ in X_query]
    base_mse = mse(base_preds, y_query)
    print(f"  base MSE on queries:                              {base_mse:.4f}")
    print(f"  → useless predictor; no demo absorbed.")

    banner("2 — Pretunist: adapt on demos")
    cfg = PretunistConfig(
        adapter_dim=d,
        output_dim=1,
        ridge_lambda=1e-3,
        prior_variance=1.0,
        posterior_variance=1.0,
        noise_variance=0.05,
        kl_budget=math.inf,                # unconstrained for now
        abstain_rules=(ABSTAIN_LEVERAGE, ABSTAIN_VARIANCE),
        abstain_leverage_threshold=0.95,
        abstain_variance_threshold=0.5,
    )
    pre = Pretunist(cfg)
    pre.set_base(base)

    # 2a. Ingest demos.
    pre.observe_batch(X_demo, y_demo)
    print(f"  observed:                                         n={pre.n_observed}")

    # 2b. Adapt + predict on queries.
    adapted_preds = []
    abstained = 0
    for xq in X_query:
        ab = pre.should_abstain(xq)
        if ab.triggered:
            abstained += 1
            adapted_preds.append((0.0,))  # caller would route to fallback
            continue
        r = pre.adapt(query=xq)
        adapted_preds.append(r.prediction)
    adapted_mse = mse(adapted_preds, y_query)
    print(f"  adapted MSE on queries:                           {adapted_mse:.4f}")
    print(f"  abstentions:                                      {abstained}/{n_query}")
    print(f"  MSE reduction vs base:                            "
          f"{(base_mse - adapted_mse):.4f}  ({100*(base_mse - adapted_mse)/max(base_mse,1e-12):.1f}%)")

    banner("3 — One adaptation in detail")
    sample = X_query[0]
    target = y_query[0][0]
    r = pre.adapt(query=sample)
    print(f"  query:               x = [{sample[0]:+.3f}, {sample[1]:+.3f}, …]")
    print(f"  true target:                              y  = {target:+.4f}")
    print(f"  base prediction:                       ŷ_0  = {r.base_prediction[0]:+.4f}")
    print(f"  adapted prediction:                      ŷ  = {r.prediction[0]:+.4f}")
    print(f"  leverage h(x):                            {r.leverage:.4f}")
    print(f"  predictive variance σ²(1+h):              {r.predictive_variance:.4f}")
    print(f"  adaptation gain  (bits):                  {r.adaptation_gain_bits:.3f}")
    print(f"  KL drift  ||θ*−θ_0||²/(2σ_p²) (nats):     {r.kl_drift_nats:.3f}")
    print(f"  KL-budget projected?                       {r.kl_budget_active}")
    print(f"  fit residual ||y − Xθ*||₂:                {r.fit_residual_norm:.4f}")

    banner("4 — PAC-Bayes certificate")
    cert = pre.certify(delta=0.05)
    print(f"  n:                                        {cert.n}")
    print(f"  δ:                                        {cert.delta:.3f}")
    print(f"  empirical risk:                           {cert.empirical_risk:.4f}")
    print(f"  KL(Q‖P) (nats):                           {cert.kl_qp_nats:.3f}")
    print(f"  PAC-Bayes upper bound on test risk:       {cert.pac_bayes_bound:.4f}")
    print(f"  bound is vacuous?                          {cert.pac_bayes_is_vacuous}")
    print(f"  leave-one-out risk:                       {cert.loo_risk:.4f}")
    print(f"  max leverage on support:                  {cert.leverage_max:.4f}")
    print(f"  e-process log-value:                      {cert.e_process_log:.3f}")
    print(f"  e-process rejected (1-α level)?            {cert.e_process_rejected}")
    print(f"  adapter fingerprint (SHA-256):            {cert.adapter_fingerprint[:16]}…")
    print(f"  ledger root (SHA-256):                    {cert.ledger_root[:16]}…")
    print(f"  references:")
    for ref in cert.references:
        print(f"    · {ref}")

    banner("5 — KL-budget projection (coordinator demands bounded drift)")
    # Now run with a tight KL budget — emulate "preserve safety
    # guarantees that hold under the base policy".
    budget = 0.5 * r.kl_drift_nats
    print(f"  unconstrained KL drift was:               {r.kl_drift_nats:.3f}")
    print(f"  setting budget to:                        {budget:.3f}")
    pre_c = Pretunist(PretunistConfig(
        adapter_dim=d,
        ridge_lambda=cfg.ridge_lambda,
        kl_budget=budget,
        abstain_rules=cfg.abstain_rules,
    ))
    pre_c.set_base(base)
    pre_c.observe_batch(X_demo, y_demo)
    r_c = pre_c.adapt(query=sample)
    print(f"  projected KL drift:                       {r_c.kl_drift_nats:.3f}")
    print(f"  projection α (1 = no project):            {r_c.kl_projection_alpha:.4f}")
    print(f"  projected prediction:                      ŷ  = {r_c.prediction[0]:+.4f}")
    print(f"  projected adaptation gain (bits):          {r_c.adaptation_gain_bits:.3f}")

    banner("6 — Abstention on an out-of-support query")
    extreme = [100.0] + [0.0] * (d - 1)
    ab = pre.should_abstain(extreme)
    print(f"  query: x = [100, 0, 0, …]")
    print(f"  triggered:                                 {ab.triggered}")
    print(f"  rules fired:                               {list(ab.rules_fired)}")
    print(f"  leverage:                                 {ab.leverage:.3f}")
    print(f"  predictive variance:                      {ab.predictive_variance:.3f}")
    print(f"  KL drift:                                 {ab.kl_drift_nats:.3f}")
    if ab.triggered:
        print(f"  → coordinator should route to a larger model / collect data.")

    banner("7 — Snapshot / restore (federation)")
    snap = pre.snapshot()
    print(f"  snapshot size (chars in JSON form): "
          f"{sum(len(repr(v)) for v in snap.values())}  (illustrative)")
    pre_restored = Pretunist.restore(snap)
    print(f"  ledger preserved across snapshot/restore? "
          f"{pre.ledger_root == pre_restored.ledger_root}")
    print(f"  adapter fingerprint preserved?            "
          f"{pre.adapter_fingerprint == pre_restored.adapter_fingerprint}")
    p_orig = pre.predict(sample)[0]
    p_rest = pre_restored.predict(sample)[0]
    print(f"  prediction at sample x[0] (original):     {p_orig:+.6f}")
    print(f"  prediction at sample x[0] (restored):     {p_rest:+.6f}")
    print(f"  predictions identical?                     "
          f"{abs(p_orig - p_rest) < 1e-12}")

    banner("8 — Coordinator-visible report")
    rep = pre.report()
    print(f"  schema:                {rep.schema}")
    print(f"  n_support:             {rep.n_support}")
    print(f"  n_adaptations:         {rep.n_adaptations}")
    print(f"  n_abstentions:         {rep.n_abstentions}")
    print(f"  last_loo_risk:         {rep.last_loo_risk:.4f}")
    print(f"  last_pac_bayes_bound:  {rep.last_pac_bayes_bound:.4f}")
    print(f"  last_kl_drift_nats:    {rep.last_kl_drift_nats:.3f}")
    print(f"  last_adapt_gain_bits:  {rep.last_adaptation_gain_bits:.3f}")
    print(f"  last_e_process_log:    {rep.last_e_process_log:.3f}")
    print(f"  ledger_root:           {rep.ledger_root[:16]}…")
    print(f"  adapter_fingerprint:   {rep.adapter_fingerprint[:16]}…")

    print()
    print("Done.  Pretunist closes the per-token-adaptation gap in the")
    print("architecture's learning-timescale table — and it ships a")
    print("McAllester PAC-Bayes certificate with every call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
