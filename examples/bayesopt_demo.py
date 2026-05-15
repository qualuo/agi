"""BayesOpt demo — three head-to-head problems showcasing the
coordination-engine surface of `agi.bayesopt.BayesOpt`.

Showcases:
  * Five acquisition policies (GP-UCB / EI / PI / Thompson / KG) on a
    common 2D objective.
  * Mixed continuous + categorical search via ``MixedDomain``.
  * Batch / parallel suggestions via the constant-liar heuristic.
  * Anytime cumulative-regret bound (Srinivas et al. 2010) printed at
    every step.
  * Tamper-evident replay fingerprint per run.

Run::

    python -m examples.bayesopt_demo
"""

from __future__ import annotations

import math

from agi.bayesopt import (
    ACQ_EI,
    ACQ_KG,
    ACQ_PI,
    ACQ_THOMPSON,
    ACQ_UCB,
    BayesOpt,
    BayesOptConfig,
    CategoricalDim,
    ContinuousBox,
    KERNEL_MATERN52,
    MAXIMISE,
    MixedDomain,
)


def head_to_head_2d() -> None:
    """Run all five acquisitions on a 2D quadratic with a known optimum
    at (0.4, 0.6); compare best-y after 30 evaluations.
    """

    def f(x):
        return 1.0 - ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2)

    domain = ContinuousBox(low=(0.0, 0.0), high=(1.0, 1.0))
    print("=" * 68)
    print("Head-to-head: 2D quadratic, target (0.4, 0.6), 30 evals/acq")
    print("=" * 68)
    print(f"{'acq':>10}  {'best_y':>8}  {'best_x':>20}  {'fingerprint':>14}")
    for acq in (ACQ_UCB, ACQ_EI, ACQ_PI, ACQ_THOMPSON, ACQ_KG):
        cfg = BayesOptConfig(
            direction=MAXIMISE, acquisition=acq, kernel=KERNEL_MATERN52,
            seed=2025, noise_var=1e-4, kg_fantasies=8,
        )
        bo = BayesOpt(domain=domain, config=cfg)
        for _ in range(30):
            sug = bo.suggest()
            bo.observe(sug.x, f(sug.x))
        best = bo.best()
        fp = bo.fingerprint()[:12]
        bx = "(" + ", ".join(f"{v:.3f}" for v in best.x) + ")"
        print(f"{acq:>10}  {best.y:8.5f}  {bx:>20}  {fp:>14}")


def mixed_domain() -> None:
    """Mixed continuous + categorical domain.  The objective rewards
    (continuous x ≈ 0.65) AND (categorical = 'b').
    """

    cont = ContinuousBox(low=(0.0,), high=(1.0,))
    cats = (CategoricalDim(name="kind", values=("a", "b", "c")),)
    domain = MixedDomain(cont=cont, cats=cats)

    def f(x):
        return -((x[0] - 0.65) ** 2 + (x[1] - 1.0) ** 2)

    cfg = BayesOptConfig(
        direction=MAXIMISE, acquisition=ACQ_EI, seed=11, noise_var=1e-4,
    )
    bo = BayesOpt(domain=domain, config=cfg)
    print()
    print("=" * 68)
    print("Mixed-domain BO: continuous [0,1] + categorical {a,b,c}")
    print("Target: x[0] ≈ 0.65, category = 'b' (index 1)")
    print("=" * 68)
    for t in range(30):
        sug = bo.suggest()
        y = f(sug.x)
        bo.observe(sug.x, y)
        if t in (4, 9, 14, 19, 29):
            cat_idx = int(round(sug.x[1]))
            print(
                f"  t={t+1:>3}  pick=({sug.x[0]:.3f}, '{('abc')[cat_idx]}')"
                f"   y={y:8.4f}   cum-regret-bound={bo.cumulative_regret_bound():.3f}"
            )
    best = bo.best()
    print(f"\nBest:  ({best.x[0]:.3f}, '{('abc')[int(round(best.x[1]))]}')"
          f"   y={best.y:.5f}")


def batch_parallel() -> None:
    """Constant-liar batch BO: request 4 parallel suggestions from a GP
    with a single seed observation.
    """

    domain = ContinuousBox(low=(0.0, 0.0), high=(1.0, 1.0))
    cfg = BayesOptConfig(
        direction=MAXIMISE, acquisition=ACQ_UCB, seed=3, noise_var=1e-4,
    )
    bo = BayesOpt(domain=domain, config=cfg)
    bo.observe((0.5, 0.5), 0.0)
    bo.observe((0.4, 0.6), 0.05)
    bo.observe((0.6, 0.4), 0.05)
    print()
    print("=" * 68)
    print("Batch / parallel BO (constant-liar, k=4)")
    print("=" * 68)
    batch = bo.suggest_batch(4)
    for i, sug in enumerate(batch):
        print(f"  batch[{i}] = ({sug.x[0]:.3f}, {sug.x[1]:.3f})"
              f"   acq={sug.acquisition_value:.4f}"
              f"   μ={sug.mean:.3f}   σ={sug.std:.3f}")
    print(f"\nDistinct? {len({s.x for s in batch}) == len(batch)}")


def coordination_engine_handshake() -> None:
    """The shape a coordination engine sees: register → suggest →
    observe → report.  Print the report verbatim — every field is a
    quantity a higher-level planner can act on.
    """
    domain = ContinuousBox(low=(0.0,), high=(1.0,))
    cfg = BayesOptConfig(direction=MAXIMISE, acquisition=ACQ_EI, seed=7,
                         noise_var=1e-4)
    bo = BayesOpt(domain=domain, config=cfg)

    def expensive_oracle(x):
        # 1D sin: argmax ≈ π/12 ≈ 0.261… in [0, 1].
        return math.sin(6.0 * x[0])

    for _ in range(20):
        sug = bo.suggest()
        bo.observe(sug.x, expensive_oracle(sug.x))
    print()
    print("=" * 68)
    print("Coordination-engine handshake: BayesOptReport on 1D sin(6x)")
    print("=" * 68)
    report = bo.report()
    print(f"  direction:                        {report.direction}")
    print(f"  n_observations:                   {report.n_observations}")
    print(f"  best_x:                           {report.best_x}")
    print(f"  best_y:                           {report.best_y:.6f}")
    print(f"  posterior_max_mean_x:             {report.posterior_max_mean_x}")
    print(f"  posterior_max_mean:               {report.posterior_max_mean:.6f}")
    print(f"  posterior_max_std:                {report.posterior_max_std:.6f}")
    print(f"  simple_regret_upper_bound:        {report.simple_regret_upper_bound:.6f}")
    print(f"  cumulative_regret_upper_bound:    {report.cumulative_regret_upper_bound:.6f}")
    print(f"  info_gain_estimate:               {report.info_gain_estimate:.6f}")
    print(f"  beta_t (GP-UCB schedule):         {report.beta_t:.6f}")
    print(f"  kernel:                           {report.kernel_name}")
    print(f"  fingerprint:                      {report.fingerprint[:16]}…")


if __name__ == "__main__":
    head_to_head_2d()
    mixed_domain()
    batch_parallel()
    coordination_engine_handshake()
