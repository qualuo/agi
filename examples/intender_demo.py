"""Intender — inverse reinforcement learning / preference-based reward
inference as a runtime primitive.

Run this from the repo root::

    python examples/intender_demo.py

Scenario
--------

A coordination engine has access to *expert behaviour* — completed
trajectories an analyst marked "good", and pairwise preferences over
candidate plans — but no explicit reward function.  Before it can plan
under the user's preferences (with ``ActiveInferencer``), evaluate a
candidate policy (with ``PolicyImprover``), or commit a deployment to
production (with ``Quantilizer``), it must *infer* what the user values.

The Intender primitive solves this in three flavours:

  1. **MaxEnt IRL** (Ziebart 2008) — fit reward weights ``θ`` such that
     the expert's expected feature visit ``E_τ[Σ_t γ^t φ(s_t, a_t)]``
     equals the observed empirical visit.  The unique max-entropy
     reward consistent with the data.

  2. **Bayesian IRL** (Ramachandran-Amir 2007) — sample the posterior
     ``p(θ | τ)`` via Metropolis-Hastings on the Boltzmann-rationality
     likelihood.  Returns credible regions plus a Geweke stationarity
     check on the chain.

  3. **Bradley-Terry preference learning** (Christiano-et-al 2017) —
     fit ``θ`` such that ``σ(θᵀ ΔΦ) > 0.5`` whenever the user prefers
     the winner trajectory.  Returns held-out agreement rate with an
     anytime-valid Howard-Ramdas-McAuliffe-Sekhon 2021 confidence
     sequence the coordinator can stop on without losing coverage.

The demo runs all three on a 3×3 gridworld with an unknown reward and
prints the receipts a coordination engine would actually consume.

What the coordination engine gets
---------------------------------

  * **Pointwise reward** — call ``intender.reward(state, action)`` and
    plug it into any downstream optimiser (Bandit, BayesOpt, Composer,
    Strategist, ActiveInferencer).

  * **Soft-optimal policy** — call ``intender.policy()`` to get a
    state-conditional action distribution that is the natural
    information-theoretic minimum-commitment surrogate for the
    expert's behaviour.

  * **KL-from-behavioural-cloning** — composes directly with
    Quantilizer's safe-deployment KL budget.

  * **Identifiability bound** (Cao-Cohen-Szepesvári 2021) — the
    nullity of the feature matrix tells the coordinator *which* reward
    perturbations are observationally indistinguishable; downstream
    primitives can then refuse to optimise over the null space.

  * **Anytime-valid certificates** on every aggregate statistic —
    preference agreement, log-likelihood, feature-matching residual.

  * **Tamper-evident fingerprint chain** — every observe / fit / report
    event SHA-256-hashes into ``AttestationLedger`` so an external
    auditor can replay the inference trace byte-for-byte.

Investor framing
----------------

> The runtime cannot align an agent to a user's preferences without
> first *inferring* those preferences from observed behaviour.  The
> Intender is the universal preference-elicitation primitive: it turns
> demonstrations and thumbs-up/down signals into a calibrated reward
> function the rest of the runtime can optimise — with explicit
> identifiability bounds, posterior uncertainty, and tamper-evident
> audit trails.  This is the RLHF kernel as a runtime call.
"""
from __future__ import annotations

import random

from agi.intender import (
    BIRL,
    MAXENT,
    PREFERENCE,
    Intender,
    identifiability_report,
    quick_gridworld_fixture,
)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _expert_trajectory_toward_goal(schema, start=(0, 0), goal=(2, 2)):
    """The expert is a Manhattan-walker that stays at the goal once it gets
    there.  The reward generating this behaviour is unknown to Intender."""
    s = start
    out = []
    while s != goal:
        x, y = s
        gx, gy = goal
        if x < gx:
            a = "E"
        elif x > gx:
            a = "W"
        elif y < gy:
            a = "N"
        else:
            a = "S"
        out.append((s, a))
        if a == "E":
            s = (x + 1, y)
        elif a == "W":
            s = (x - 1, y)
        elif a == "N":
            s = (x, y + 1)
        else:
            s = (x, y - 1)
    # A couple of stays at the goal so the absorbing reward is observed.
    out.append((goal, "X"))
    out.append((goal, "X"))
    return out


def _lazy_loser_trajectory():
    """A trajectory that never moves toward the goal — the loser side of
    every pairwise preference."""
    return [((0, 0), "X") for _ in range(7)]


def demo_maxent():
    _section("MaxEnt IRL (Ziebart 2008) — feature-matching reward fit")

    schema, _ = quick_gridworld_fixture(width=3, height=3, goal=(2, 2), gamma=0.95)
    intender = Intender.maxent(
        states=schema.states,
        actions=schema.actions,
        features=schema.feature_fn,
        transitions=schema.transitions,
        gamma=schema.gamma,
        horizon=schema.horizon,
        lr=0.5,
        max_iters=300,
        tol=1.0e-3,
        l2=1.0e-3,
    )
    rng = random.Random(0)
    starts = [(rng.randrange(3), rng.randrange(3)) for _ in range(10)]
    for start in starts:
        intender.observe_trajectory(_expert_trajectory_toward_goal(schema, start=start))

    intender.fit()
    report = intender.report()

    print(f"  algorithm                : {report.algorithm}")
    print(f"  trajectories observed    : {report.n_trajectories}")
    print(f"  feature dim              : {report.feature_dim}")
    print(f"  converged                : {report.converged}")
    print(f"  feature-residual ‖·‖     : {report.feature_residual_norm:.4f}")
    print(f"  mean-step log-likelihood : {report.log_likelihood:.4f}")
    print(f"  KL(π_soft ‖ π_BC)        : {report.kl_to_bc:.4f}")
    print(f"  soft policy entropy      : {report.soft_policy_entropy:.4f}")
    print()
    print("  fitted reward weights (goal-flag, step-cost, x-norm, y-norm):")
    for k, w in enumerate(report.theta):
        print(f"    θ[{k}] = {w:+.4f}")
    print()
    print("  identifiability:")
    print(f"    feature_dim  = {report.identifiability.feature_dim}")
    print(f"    rank         = {report.identifiability.rank}")
    print(f"    nullity      = {report.identifiability.nullity}")
    print(f"    conditioning = {report.identifiability.conditioning:.3f}")
    print()
    print("  certificates:")
    for name, cert in report.certificates.items():
        print(f"    {name:30s} estimate={cert.estimate:+.4f}  "
              f"halfwidth={cert.half_width:.4f}  method={cert.method}  "
              f"δ={cert.delta:.3f}  n={cert.n}")
    print(f"  fingerprint              : {report.fingerprint[:16]}…")


def demo_preference():
    _section("Preference IRL (Christiano 2017) — Bradley-Terry reward fit")

    schema, _ = quick_gridworld_fixture(width=3, height=3, goal=(2, 2), gamma=0.95)
    intender = Intender.preference(
        states=schema.states,
        actions=schema.actions,
        features=schema.feature_fn,
        transitions=schema.transitions,
        gamma=schema.gamma,
        horizon=schema.horizon,
        beta=1.5,
        lr=0.5,
        max_iters=300,
        tol=1.0e-4,
        l2=1.0e-3,
    )

    rng = random.Random(1)
    for _ in range(40):
        start = (rng.randrange(3), rng.randrange(3))
        winner = _expert_trajectory_toward_goal(schema, start=start)
        loser = _lazy_loser_trajectory()
        intender.observe_preference(winner, loser)

    intender.fit()
    report = intender.report()
    print(f"  preferences observed     : {report.n_preferences}")
    print(f"  converged                : {report.converged}")
    print(f"  training log-likelihood  : {report.log_likelihood:.4f}")
    print(f"  training agreement rate  : {report.agreement_rate:.4f}")
    print()

    # Held-out preference agreement with an anytime-valid CS.
    held_out = []
    for _ in range(20):
        start = (rng.randrange(3), rng.randrange(3))
        held_out.append(
            (_expert_trajectory_toward_goal(schema, start=start), _lazy_loser_trajectory())
        )
    stats = intender.evaluate_preferences(held_out)
    print("  held-out agreement (anytime-valid CS):")
    print(f"    n                = {stats['n']}")
    print(f"    estimate         = {stats['agreement_rate']:.4f}")
    print(f"    [lower, upper]   = [{stats['agreement_lower']:.4f}, "
          f"{stats['agreement_upper']:.4f}]")
    print(f"    method           = {stats['method']}  δ={stats['delta']:.3f}")
    print()
    print("  fitted reward weights:")
    for k, w in enumerate(report.theta):
        print(f"    θ[{k}] = {w:+.4f}")
    print(f"  fingerprint              : {report.fingerprint[:16]}…")


def demo_birl():
    _section("Bayesian IRL (Ramachandran-Amir 2007) — posterior over rewards")

    schema, _ = quick_gridworld_fixture(width=2, height=2, goal=(1, 1), gamma=0.9)
    intender = Intender.birl(
        states=schema.states,
        actions=schema.actions,
        features=schema.feature_fn,
        transitions=schema.transitions,
        gamma=schema.gamma,
        horizon=schema.horizon,
        beta=2.0,
        sigma_prior=1.0,
        proposal_scale=0.3,
        burn_in=100,
        n_steps=300,
        thin=1,
        vi_iters=50,
        seed=42,
    )

    intender.observe_trajectory([((0, 0), "E"), ((1, 0), "N"), ((1, 1), "X"), ((1, 1), "X")])
    intender.observe_trajectory([((0, 0), "N"), ((0, 1), "E"), ((1, 1), "X"), ((1, 1), "X")])
    intender.observe_trajectory([((0, 1), "E"), ((1, 1), "X")])
    intender.observe_trajectory([((1, 0), "N"), ((1, 1), "X"), ((1, 1), "X")])

    intender.fit()
    report = intender.report()

    print(f"  trajectories observed    : {report.n_trajectories}")
    print(f"  acceptance rate          : {report.acceptance_rate:.3f}  "
          "(target ≈ 0.234 in d > 1; Roberts-Rosenthal 2009)")
    print(f"  Geweke max |z|           : {report.geweke_max_abs_z:.3f}  "
          "(stationary if < 1.96)")
    print(f"  converged                : {report.converged}")
    print()
    print("  posterior summary:")
    for k, (lo, mu, hi) in enumerate(zip(
        report.posterior_lower, report.posterior_mean, report.posterior_upper
    )):
        print(f"    θ[{k}]  mean={mu:+.4f}   95% CI=[{lo:+.4f}, {hi:+.4f}]")
    print()
    print("  identifiability:")
    print(f"    feature_dim  = {report.identifiability.feature_dim}")
    print(f"    rank         = {report.identifiability.rank}")
    print(f"    nullity      = {report.identifiability.nullity}")
    print()
    samples = intender.sample_posterior(n=5)
    print(f"  posterior samples (showing 5 of {report.diagnostics['n_samples']}):")
    for i, s in enumerate(samples):
        print(f"    sample[{i}]: {['{:+.3f}'.format(x) for x in s]}")
    print(f"  fingerprint              : {report.fingerprint[:16]}…")


def demo_composition():
    _section("Composition — Intender feeds the rest of the runtime")
    schema, _ = quick_gridworld_fixture(width=3, height=3, goal=(2, 2), gamma=0.95)
    intender = Intender.maxent(
        states=schema.states, actions=schema.actions,
        features=schema.feature_fn, transitions=schema.transitions,
        gamma=schema.gamma, horizon=schema.horizon,
        lr=0.5, max_iters=200, tol=1.0e-3, l2=1.0e-3,
    )
    for _ in range(8):
        intender.observe_trajectory(_expert_trajectory_toward_goal(schema))
    intender.fit()
    policy = intender.policy()

    # The "coordinator" asks: which action does the learned policy prefer at
    # each state on the diagonal?
    print("  learned soft-optimal action by state (top-1):")
    for s in [(0, 0), (1, 1), (2, 2)]:
        best_a = max(schema.actions, key=lambda a: policy[(s, a)])
        p_best = policy[(s, best_a)]
        print(f"    state {s}: prefer action {best_a!r:>3}  (p={p_best:.3f})")
    print()
    print("  reward at a few state-action pairs (θᵀφ):")
    for s, a in [((2, 2), "X"), ((0, 0), "E"), ((0, 0), "W"), ((1, 1), "N")]:
        print(f"    r({s}, {a!r}) = {intender.reward(s, a):+.4f}")
    print()
    # Show what a coordination engine would consume.
    print("  receipts for downstream composition:")
    print("    • soft-optimal policy   → ActiveInferencer.C")
    print("    • posterior on θ        → Strategist risk weighting")
    print("    • KL(π_soft ‖ π_BC)     → Quantilizer KL budget")
    print("    • identifiability null  → Refuter falsification target")
    print("    • SHA-256 fingerprint   → AttestationLedger")


def main():
    print("# Intender demo — inverse reinforcement learning as a runtime primitive")
    demo_maxent()
    demo_preference()
    demo_birl()
    demo_composition()
    print()
    print("=" * 72)
    print("  Done. The Intender is the runtime's preference-elicitation kernel.")
    print("=" * 72)


if __name__ == "__main__":
    main()
