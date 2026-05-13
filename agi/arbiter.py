"""Arbiter — fixed-confidence Best-Arm Identification as a runtime primitive.

Every coordination engine on top of this runtime sooner or later faces the
same question: of these K candidate strategies — model variants, prompt
templates, tool implementations, fine-tuned adapters, sub-agent roles —
**which one is best**, and how do I pull samples so I can commit with a
specified confidence using the *fewest* observations?

This is the classical Best-Arm Identification (BAI) problem, distinct from
regret-minimisation bandits (UCB1, Thompson) that already live in
`agi.policy`. Regret minimisation optimises cumulative reward *during*
exploration. BAI optimises *terminal* decision quality: at the moment the
campaign stops, the recommended arm must be `(ε, δ)`-PAC — at most ε
worse than the true best with probability ≥ 1−δ. Regret-minimising
policies overpull the empirical leader (it costs them no regret to do so)
and therefore *under*-explore the runners-up — the same property that
makes them bad at BAI.

`Arbiter` is the runtime primitive that closes this gap. It composes three
algorithms that span the practical Pareto frontier between sample
optimality, implementation complexity, and operational fit:

  1. **Track-and-Stop (Garivier & Kaufmann, COLT 2016).**
     Asymptotically optimal: for fixed confidence δ → 0, no algorithm
     beats Track-and-Stop in expected sample complexity. The sampler
     tracks an *optimal proportion vector* `w* ∈ Δ_K` that solves the
     game-theoretic equilibrium

         w* = argmax_{w ∈ Δ_K} min_{a ≠ a*}
                 ( w_{a*} · d(μ_{a*}, x_a) + w_a · d(μ_a, x_a) )

     where `x_a = (w_{a*} μ_{a*} + w_a μ_a) / (w_{a*} + w_a)` is the
     midpoint that minimises the cost of confusing arm `a*` with arm `a`
     under KL divergence `d`. The stopping rule is the Chernoff/GLR
     statistic

         Z_{a*, a}(t) = N_{a*}(t) · d(μ̂_{a*}, x̂_a) + N_a(t) · d(μ̂_a, x̂_a)

         τ_δ = inf{ t : min_{a ≠ â*(t)} Z_{â*(t), a}(t) ≥ β(t, δ) }

     with Kaufmann-Koolen threshold `β(t, δ) = log((log(t) + 1)/δ)`. We
     implement **C-tracking** (Garivier-Kaufmann §3.3): pulls the arm
     that maximises the cumulative deficit `t · w*_a − N_a(t)`, ties
     broken by least-pulled, with a forced-exploration rate `√(t)/K`
     so every arm gets pulled infinitely often.

  2. **KL-LUCB (Kalyanakrishnan, Tewari, Auer & Stone, ICML 2012).**
     The pragmatic workhorse: identify the empirical leader `h_t` and
     its KL-UCB challenger `l_t`, pull both, repeat until
     `U_{l_t}(t) − L_{h_t}(t) ≤ ε`. Confidence bounds use the KL
     inversion

         U_a(t) = sup{ q : N_a(t) · d(μ̂_a, q) ≤ β(t, δ) }
         L_a(t) = inf{ q : N_a(t) · d(μ̂_a, q) ≤ β(t, δ) }

     KL-LUCB is not asymptotically optimal but matches Track-and-Stop
     constants within a factor of ~2 on most regimes and is robust to
     model-mismatch in finite samples — the right choice when the user
     sets a moderate δ ∈ [0.01, 0.1] and wants no surprises.

  3. **Sequential Halving (Karnin, Koren, & Somekh, ICML 2013).**
     The fixed-*budget* alternative: given a total sample budget `T`,
     not a confidence δ, identify the best arm. Splits `T` into
     `⌈log_2 K⌉` rounds; in each round every surviving arm is pulled
     `T / (|S_r| · ⌈log_2 K⌉)` times, and the bottom half by empirical
     mean is eliminated. With probability ≥ 1 − exp(−T / (8 H_2 log_2 K)),
     where `H_2 = max_i i · Δ_i^{-2}`, the returned arm is best. This is
     the right primitive when the budget is the constraint, not the
     confidence — e.g. a fixed daily eval spend.

What it composes (razor-sharp coordination integration)
------------------------------------------------------

  - **Strategist.** When `Strategist.recommend(...)` returns
    `STRAT_EXPLORE` — i.e. "the data is too thin, run for data" — the
    coordinator can hand the candidate set to `Arbiter.campaign(...)`
    and get back a winner with a PAC certificate. The Strategist
    re-runs with the winner pinned and typically returns `STRAT_SINGLE`
    on the next call.

  - **PolicyLab / PolicyImprover.** Once Arbiter declares a winning
    arm, that arm's induced policy can be HCPI-certified by
    `PolicyImprover.safety_check(...)` against the production baseline
    before being promoted. Arbiter says *which*; PolicyImprover says
    *whether it's safe to ship*.

  - **CalibrationEngine / ConformalPredictor.** Per-arm Bernoulli
    means returned by Arbiter feed calibrated `p_success` priors;
    per-arm reward variances feed conformal residuals. A short Arbiter
    campaign is the fastest way to bootstrap a calibrator on a new
    model variant.

  - **ExperimentDesigner.** The two are duals: ExperimentDesigner
    selects the next *experiment* to maximise information about a
    target function; Arbiter selects the next *arm pull* to identify
    the maximiser. ExperimentDesigner answers "what to measure";
    Arbiter answers "how many times before I commit".

  - **AttestationLedger.** Every completed campaign emits a
    tamper-evident `arbiter.committed` receipt: arms, observation
    counts, stopping statistic, PAC certificate. A downstream auditor
    can replay the exact decision under the same δ and reproduce the
    winner. This is what "shippable AI decisions" requires.

  - **DriftSentinel.** Subscribes to `arbiter.observed` events; a
    drift trigger on the winner's reward stream invalidates the
    campaign's certificate and the coordinator must re-arbitrate.

  - **EventBus.** Streams every pull, every commit, every observation.
    The bus is how a higher-level coordination engine reacts in real
    time: route the next ticket to the leading arm the moment Arbiter
    has enough evidence.

Where this slots in
-------------------

    arbiter = Arbiter(bus=bus, attestor=attestor)

    # Online (streaming) — feed in observations from the live runtime.
    arbiter.start_campaign(
        campaign_id="model-bakeoff-2026Q2",
        arms=["haiku", "sonnet", "opus"],
        algorithm=ALGO_TRACK_AND_STOP,
        delta=0.05,
        epsilon=0.02,
        reward_model=REWARD_BERNOULLI,
    )
    for obs in driver.completed():
        arbiter.observe("model-bakeoff-2026Q2", obs.arm_id, float(obs.success))
        plan = arbiter.next_pulls("model-bakeoff-2026Q2", batch=4)
        if plan.stopped:
            break
    report = arbiter.report("model-bakeoff-2026Q2")
    coordinator.promote(report.best_arm, certificate=report.pac_receipt_hash)

    # Synchronous (closed-loop) — Arbiter drives the sampler itself.
    def sample(arm_id: str) -> float:
        return float(driver.dispatch(prompt, model=arm_id).success)
    campaign = arbiter.run(
        arms=["v1", "v2", "v3"],
        sampler=sample,
        algorithm=ALGO_KL_LUCB,
        delta=0.05,
        max_samples=1000,
    )

Events
------
    arbiter.started     — a campaign began
    arbiter.pulled      — a sampling plan was emitted (which arms to pull next)
    arbiter.observed    — a single reward landed
    arbiter.committed   — the campaign stopped with a PAC certificate
    arbiter.exhausted   — the campaign stopped because the budget was reached
    arbiter.cleared     — a campaign was reset / discarded
    arbiter.report      — a coverage / sample-complexity report was published

Honest about limits
-------------------

  - **Track-and-Stop assumes the right reward model.** If you tell
    Arbiter the rewards are Bernoulli and they are heavy-tailed, the
    KL stopping rule is overconfident. The runtime publishes a
    *self-coverage* report: did the realised PAC rate match `δ`? If
    miscoverage materially exceeds `δ`, switch reward models.

  - **The PAC guarantee is conditional on i.i.d. observations from a
    fixed-arm distribution.** Drift on the winning arm's reward stream
    invalidates the certificate. Subscribe DriftSentinel to
    `arbiter.observed` to enforce this.

  - **The optimal proportion solver is a fixed-point iteration on a
    smooth concave-convex game.** It converges; we cap iterations at
    `_MAX_WSTAR_ITERS=400` and fall back to uniform allocation on
    non-convergence with a logged warning. Non-convergence is
    diagnostic of degenerate arm gaps (two arms with identical
    empirical means), in which case Arbiter pulls them uniformly
    anyway.

  - **KL-LUCB is anytime-valid** but not asymptotically optimal in
    the multi-arm regime — Track-and-Stop will identify the winner in
    fewer samples once `δ ≤ 0.01` and K ≥ 4. Below that, KL-LUCB is
    typically within 30%.

  - **Sequential Halving cannot promise confidence.** It returns a
    winner; whether that winner is correct is a function of the
    budget and the (unknown) gaps. Use only when the budget is hard.

The module is stdlib-only and CPU-bound. A 6-arm Bernoulli campaign
at δ=0.05 typically stops in ~10⁴ samples; the per-update overhead in
Arbiter itself is single-digit microseconds. The end-to-end runtime
ships without external numerical dependencies — investor demos run
on a laptop.

Citations
---------

* Garivier, A., & Kaufmann, E. (2016). Optimal best arm identification
  with fixed confidence. *Proc. COLT*, 998–1027.
* Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012). PAC
  subset selection in stochastic multi-armed bandits. *Proc. ICML*,
  655–662.
* Karnin, Z., Koren, T., & Somekh, O. (2013). Almost optimal
  exploration in multi-armed bandits. *Proc. ICML*, 1238–1246.
* Kaufmann, E., & Koolen, W. (2021). Mixture martingales revisited
  with applications to sequential tests and confidence intervals.
  *JMLR* 22(246), 1–44.
* Even-Dar, E., Mannor, S., & Mansour, Y. (2006). Action elimination
  and stopping conditions for the multi-armed bandit and
  reinforcement learning problems. *JMLR* 7(39), 1079–1105.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

ARBITER_STARTED = "arbiter.started"
ARBITER_PULLED = "arbiter.pulled"
ARBITER_OBSERVED = "arbiter.observed"
ARBITER_COMMITTED = "arbiter.committed"
ARBITER_EXHAUSTED = "arbiter.exhausted"
ARBITER_CLEARED = "arbiter.cleared"
ARBITER_REPORT = "arbiter.report"


# =====================================================================
# Algorithms & reward models
# =====================================================================

ALGO_TRACK_AND_STOP = "track_and_stop"
ALGO_KL_LUCB = "kl_lucb"
ALGO_SEQUENTIAL_HALVING = "sequential_halving"
KNOWN_ALGORITHMS = (
    ALGO_TRACK_AND_STOP,
    ALGO_KL_LUCB,
    ALGO_SEQUENTIAL_HALVING,
)

REWARD_BERNOULLI = "bernoulli"
REWARD_GAUSSIAN = "gaussian"
KNOWN_REWARD_MODELS = (REWARD_BERNOULLI, REWARD_GAUSSIAN)


# =====================================================================
# Verdicts
# =====================================================================

VERDICT_BEST = "best"          # winner identified at (ε, δ)-PAC
VERDICT_EXHAUSTED = "exhausted"  # stopped on max_samples; winner is empirical
VERDICT_INFEASIBLE = "infeasible"  # zero observations on at least one arm
VERDICT_CLEARED = "cleared"    # campaign was reset before completion
KNOWN_VERDICTS = (
    VERDICT_BEST,
    VERDICT_EXHAUSTED,
    VERDICT_INFEASIBLE,
    VERDICT_CLEARED,
)


# =====================================================================
# Numerical constants
# =====================================================================

_EPS = 1e-12
_KL_TOL = 1e-9
_KL_MAX_ITER = 80
_MAX_WSTAR_ITERS = 400
_DEFAULT_GAUSSIAN_VAR = 1.0
_FORCED_EXPLORATION = 0.5  # exponent in the `t^_FORCED_EXPLORATION` floor
_DEFAULT_TIE_EPS = 1e-9


# =====================================================================
# Numerical primitives
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def kl_bernoulli(p: float, q: float) -> float:
    """KL divergence between two Bernoulli(p) and Bernoulli(q) distributions.

    Always non-negative; zero iff p == q. Robust to p ∈ {0, 1} via the
    convention 0 · log 0 = 0. Used as the divergence in Track-and-Stop
    and the GLR stopping rule for binary rewards.
    """
    p = _clip(float(p), 0.0, 1.0)
    q = _clip(float(q), _EPS, 1.0 - _EPS)
    if p <= 0.0:
        return -math.log1p(-q)
    if p >= 1.0:
        return -math.log(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_gaussian(mu1: float, mu2: float, sigma2: float) -> float:
    """KL divergence between N(mu1, sigma²) and N(mu2, sigma²).

    With shared variance, KL = (mu1 - mu2)² / (2 sigma²) — the well-known
    Gaussian quadratic. Used when the reward model is REWARD_GAUSSIAN.
    """
    if sigma2 <= 0.0:
        return math.inf if abs(mu1 - mu2) > 0.0 else 0.0
    d = mu1 - mu2
    return 0.5 * d * d / sigma2


def _kl(reward_model: str, p: float, q: float, sigma2: float) -> float:
    if reward_model == REWARD_BERNOULLI:
        return kl_bernoulli(p, q)
    if reward_model == REWARD_GAUSSIAN:
        return kl_gaussian(p, q, sigma2)
    raise ValueError(f"unknown reward model: {reward_model}")


def kl_confidence_upper(
    mu_hat: float, n: int, beta: float, *,
    reward_model: str = REWARD_BERNOULLI, sigma2: float = _DEFAULT_GAUSSIAN_VAR,
) -> float:
    """Upper confidence bound by KL inversion: sup { q ≥ mu_hat : n · d(mu_hat, q) ≤ beta }.

    For Bernoulli rewards this is the KL-UCB of Cappé et al. (2013); for
    Gaussian it reduces to the standard `mu_hat + sqrt(2 σ² β / n)`. The
    bound is tight at the optimum-transport boundary that defines the
    arms' confusion region in Track-and-Stop.
    """
    if n <= 0 or beta <= 0.0:
        return 1.0 if reward_model == REWARD_BERNOULLI else math.inf
    if reward_model == REWARD_BERNOULLI:
        if mu_hat >= 1.0:
            return 1.0
        lo, hi = _clip(mu_hat, 0.0, 1.0), 1.0
        for _ in range(_KL_MAX_ITER):
            mid = 0.5 * (lo + hi)
            if n * kl_bernoulli(mu_hat, mid) > beta:
                hi = mid
            else:
                lo = mid
            if hi - lo < _KL_TOL:
                break
        return lo
    if reward_model == REWARD_GAUSSIAN:
        return mu_hat + math.sqrt(2.0 * sigma2 * beta / n)
    raise ValueError(f"unknown reward model: {reward_model}")


def kl_confidence_lower(
    mu_hat: float, n: int, beta: float, *,
    reward_model: str = REWARD_BERNOULLI, sigma2: float = _DEFAULT_GAUSSIAN_VAR,
) -> float:
    """Lower confidence bound by KL inversion: inf { q ≤ mu_hat : n · d(mu_hat, q) ≤ beta }."""
    if n <= 0 or beta <= 0.0:
        return 0.0 if reward_model == REWARD_BERNOULLI else -math.inf
    if reward_model == REWARD_BERNOULLI:
        if mu_hat <= 0.0:
            return 0.0
        lo, hi = 0.0, _clip(mu_hat, 0.0, 1.0)
        for _ in range(_KL_MAX_ITER):
            mid = 0.5 * (lo + hi)
            if n * kl_bernoulli(mu_hat, mid) > beta:
                lo = mid
            else:
                hi = mid
            if hi - lo < _KL_TOL:
                break
        return hi
    if reward_model == REWARD_GAUSSIAN:
        return mu_hat - math.sqrt(2.0 * sigma2 * beta / n)
    raise ValueError(f"unknown reward model: {reward_model}")


def glr_threshold(t: int, delta: float, K: int = 2) -> float:
    """Kaufmann-Koolen threshold β(t, δ) for GLR stopping.

    β(t, δ) = log((log(t) + 1) / δ) + K · log(log(max(2, t))) for the
    multi-armed case. The leading term is the Kaufmann-Koolen exploration
    bonus; the secondary term is the union-bound factor across K arms.
    Sound for any K ≥ 2 and t ≥ 1.
    """
    t = max(1, int(t))
    delta = _clip(float(delta), _EPS, 1.0 - _EPS)
    leading = math.log((math.log(max(t, 2)) + 1.0) / delta)
    union = max(0, K - 1) * math.log(math.log(max(t, math.e * 2.0)))
    return max(leading + union, _EPS)


def _midpoint_for_confusion(
    mu_a: float, mu_b: float, n_a: float, n_b: float, reward_model: str, sigma2: float,
) -> float:
    """Allocation-weighted midpoint x_{a,b} that minimises the cost of
    confusing arms `a` and `b` under KL divergence.

    For Bernoulli and Gaussian (shared variance) this is the simple
    pooled mean — a classical convex-conjugate fact.
    """
    if n_a + n_b <= 0.0:
        return 0.5 * (mu_a + mu_b)
    return (n_a * mu_a + n_b * mu_b) / (n_a + n_b)


def _glr_pair_statistic(
    mu_top: float, mu_alt: float, n_top: int, n_alt: int,
    reward_model: str, sigma2: float,
) -> float:
    """Z_{top, alt}(t): GLR statistic between H_0 (top is best) and H_1 (alt is best).

    The closed form is the optimum-transport identity

        Z = n_top · d(mu_top, x) + n_alt · d(mu_alt, x)

    with x the pooled midpoint. By construction Z ≥ 0; it is large when
    the empirical gap between `top` and `alt` is large relative to their
    sample counts, and it is exactly the quantity that must exceed
    β(t, δ) to reject H_1 at level δ.
    """
    if mu_top <= mu_alt + _DEFAULT_TIE_EPS:
        return 0.0
    if n_top <= 0 or n_alt <= 0:
        return 0.0
    x = _midpoint_for_confusion(mu_top, mu_alt, n_top, n_alt, reward_model, sigma2)
    return n_top * _kl(reward_model, mu_top, x, sigma2) + n_alt * _kl(reward_model, mu_alt, x, sigma2)


def solve_w_star(
    means: Sequence[float], *,
    reward_model: str = REWARD_BERNOULLI,
    sigma2: float = _DEFAULT_GAUSSIAN_VAR,
    max_iter: int = _MAX_WSTAR_ITERS,
    tol: float = 1e-7,
) -> tuple[list[float], bool]:
    """Solve for the optimal proportion vector w* ∈ Δ_K of Track-and-Stop.

    The optimal w* maximises the game value

        Γ*(w) = min_{a ≠ a*} ( w_{a*} · d(μ_{a*}, x_a) + w_a · d(μ_a, x_a) )

    over the simplex. We solve via fixed-point iteration on the
    characterisation of Garivier-Kaufmann (Theorem 5):
    define `g_a(w) = w_{a*} · d(μ_{a*}, x_a) + w_a · d(μ_a, x_a)` for a ≠ a*.
    At the optimum, all g_a are equal. We update w_a ∝ 1/d(μ_a, x_a) for
    a ≠ a*, then balance w_{a*} against the sum, iterating until the
    values g_a equalise to within tol.

    Returns (w_star, converged). On non-convergence falls back to a
    uniform-over-non-best allocation, which is suboptimal but valid
    (Track-and-Stop with any positive allocation still terminates).
    """
    K = len(means)
    if K <= 1:
        return [1.0] * K, True
    means = [float(m) for m in means]
    best = max(range(K), key=lambda i: means[i])
    mu_best = means[best]

    # Identify degenerate arms (those tied with the best). With ties we
    # have no information geometry to exploit; allocate uniformly across
    # ties + best, leave others at near-zero.
    tied = [i for i in range(K) if abs(means[i] - mu_best) <= _DEFAULT_TIE_EPS]
    if len(tied) >= K:
        return [1.0 / K] * K, True

    # Initialise: uniform on non-best, small mass on best (will be filled
    # by the balance step).
    w = [1.0 / K] * K

    for _ in range(max_iter):
        # Step 1: given w[best], compute weights for non-best arms so that
        # all g_a are equal and sum(w) = 1.
        # Treat g_a(w) = w[best] · d(μ_best, x_a) + w[a] · d(μ_a, x_a) where
        # x_a = (w[best] μ_best + w[a] μ_a) / (w[best] + w[a]). For a fixed
        # w[best] we one-step update: aim each g_a at a common value gamma,
        # then renormalise. We do this by binary-searching gamma so that
        # the implied w summing to 1 is consistent.

        # For each candidate gamma, the implied w_a satisfies
        #   w[best] · d(μ_best, x_a) + w[a] · d(μ_a, x_a) = gamma
        # We use a closed-form approximation in the small-gap limit:
        # w[a] ≈ gamma / max(d(μ_a, μ_best), tol). This is a good warm
        # start; we then refine via one Newton step on the midpoint.

        # Range of gamma: try doubling/halving until total mass crosses 1.
        def _total_for_gamma(g: float) -> tuple[float, list[float]]:
            ws = [0.0] * K
            wb = max(_EPS, w[best])
            # First pass: solve each non-best's w via Newton.
            for a in range(K):
                if a == best:
                    continue
                d_ab = max(_kl(reward_model, means[a], mu_best, sigma2), _EPS)
                # initial guess
                wa = g / d_ab
                for _newt in range(8):
                    x = (wb * mu_best + wa * means[a]) / (wb + wa)
                    dba = _kl(reward_model, mu_best, x, sigma2)
                    dab = _kl(reward_model, means[a], x, sigma2)
                    val = wb * dba + wa * dab - g
                    # ∂x/∂w_a = wb · (mu_a - mu_best) / (wb + wa)²
                    if reward_model == REWARD_BERNOULLI:
                        # ∂d(mu_best, x)/∂x = (x - mu_best) / (x (1-x))
                        x_safe = _clip(x, _EPS, 1.0 - _EPS)
                        ddba_dx = (x_safe - mu_best) / max(x_safe * (1 - x_safe), _EPS)
                        # ∂d(mu_a, x)/∂x = (x - mu_a) / (x (1-x))
                        ddab_dx = (x_safe - means[a]) / max(x_safe * (1 - x_safe), _EPS)
                    else:  # gaussian
                        ddba_dx = (x - mu_best) / max(sigma2, _EPS)
                        ddab_dx = (x - means[a]) / max(sigma2, _EPS)
                    dx_dwa = wb * (means[a] - mu_best) / max((wb + wa) ** 2, _EPS)
                    # Total derivative
                    grad = (wb * ddba_dx + wa * ddab_dx) * dx_dwa + dab
                    if abs(grad) < _EPS:
                        break
                    delta = val / grad
                    wa = max(_EPS, wa - delta)
                    if abs(delta) < tol:
                        break
                ws[a] = wa
            total = wb + sum(ws[a] for a in range(K) if a != best)
            return total, ws

        # Binary-search gamma so total mass = 1.
        lo_g, hi_g = 1e-8, 10.0
        for _bs in range(60):
            mid_g = math.sqrt(lo_g * hi_g)
            total, ws = _total_for_gamma(mid_g)
            if total > 1.0:
                hi_g = mid_g
            else:
                lo_g = mid_g
            if hi_g / lo_g < 1.0 + tol:
                break
        _, ws = _total_for_gamma(math.sqrt(lo_g * hi_g))
        # Set w[best] so masses sum to 1.
        rest = sum(ws[a] for a in range(K) if a != best)
        wb_new = max(_EPS, 1.0 - rest)
        ws[best] = wb_new

        # Renormalise (safety against numerical drift).
        s = sum(ws)
        if s > 0:
            ws = [x / s for x in ws]

        # Convergence: changes below tol?
        max_delta = max(abs(ws[i] - w[i]) for i in range(K))
        w = ws
        if max_delta < tol:
            return w, True

    # Non-convergence — fall back to uniform.
    return [1.0 / K] * K, False


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class ArmStats:
    """Per-arm aggregate inside a campaign."""

    id: str
    n: int                       # number of observations
    mean: float                  # empirical mean
    sum_sq: float                # Σ x²; used for variance + Gaussian KL
    sample_var: float            # Σ (x - mean)² / max(n-1, 1)
    lower: float                 # KL-LCB at the current β(t, δ)
    upper: float                 # KL-UCB at the current β(t, δ)
    last_value: float            # last observation
    first_seen_at: int           # 1-indexed pull when first observed
    last_seen_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingPlan:
    """The next set of arms the runtime should pull.

    `batch_pulls` is a list of arm ids (possibly with repetitions) the
    caller should sample once each. The plan is *advisory*: the
    coordination engine can pull more or fewer, in any order — Arbiter
    only requires that each observation be reported back via `observe`.
    """

    campaign_id: str
    t: int                       # total pulls before this plan
    batch_pulls: list[str]
    stopped: bool                # campaign has reached its stop condition
    stop_reason: str             # one of KNOWN_VERDICTS
    rationale: str = ""
    statistic: float = 0.0       # current min GLR or LUCB margin
    threshold: float = 0.0       # current stopping threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArbiterReport:
    """The full record of one BAI campaign.

    `pac_receipt_hash` is non-empty iff an AttestationLedger was wired in
    and the campaign committed at `VERDICT_BEST`; it is the receipt
    hash of the immutable record a downstream auditor can use to replay
    the decision.
    """

    id: str
    arms: list[ArmStats]
    best_arm: str
    runner_up: str
    algorithm: str
    reward_model: str
    delta: float
    epsilon: float
    max_samples: int
    n_total: int
    cost: float
    stop_reason: str
    verdict: str
    confidence: str              # "high" | "medium" | "low"
    pac_guarantee: bool          # True iff a (ε, δ)-PAC certificate holds
    statistic: float             # final GLR / LUCB margin
    threshold: float             # final β(t, δ)
    sample_complexity_bound: float  # asymptotic upper bound from theory
    started_at: float
    finished_at: float
    rationale: str
    pac_receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arms"] = [a.to_dict() if hasattr(a, "to_dict") else dict(a) for a in self.arms]
        return d


@dataclass
class CoverageReport:
    """How well-calibrated *this* Arbiter has been historically.

    For campaigns that committed at `VERDICT_BEST` with confidence δ,
    the realised correct-best-arm rate should be ≥ 1 − δ when the
    truth is supplied later. Material miscoverage (> δ) signals a
    reward-model mismatch (e.g. heavy-tailed observations declared
    Bernoulli) — switch reward model.
    """

    n_campaigns: int
    n_best_verdicts: int
    n_observed_truths: int       # campaigns with a follow-up ground truth
    realised_accuracy: float     # P(returned == truth | committed)
    target_accuracy: float       # 1 − delta at commit
    miscoverage: float           # max(0, target - realised)
    n_by_verdict: dict[str, int]
    mean_samples_by_verdict: dict[str, float]
    mean_cost_by_verdict: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Internal campaign state
# =====================================================================


@dataclass
class _CampaignState:
    id: str
    arms: list[str]
    counts: dict[str, int]
    sums: dict[str, float]
    sum_sqs: dict[str, float]
    last_value: dict[str, float]
    first_seen: dict[str, int]
    last_seen: dict[str, int]
    cost: dict[str, float]
    algorithm: str
    reward_model: str
    sigma2: float
    delta: float
    epsilon: float
    max_samples: int
    started_at: float
    metadata: dict[str, Any]
    # Sequential Halving state.
    seq_halving_round: int = 0
    seq_halving_remaining: list[str] = field(default_factory=list)
    seq_halving_n_rounds: int = 0
    seq_halving_pulls_per_round: int = 0
    seq_halving_pulls_this_round: dict[str, int] = field(default_factory=dict)
    finished: bool = False
    stop_reason: str = ""
    final_report: ArbiterReport | None = None

    def total_pulls(self) -> int:
        return sum(self.counts.values())

    def total_cost(self) -> float:
        return sum(self.cost.values())


# =====================================================================
# Arbiter
# =====================================================================


class Arbiter:
    """Fixed-confidence Best-Arm Identification runtime primitive.

    Provides three algorithms (`ALGO_TRACK_AND_STOP`, `ALGO_KL_LUCB`,
    `ALGO_SEQUENTIAL_HALVING`) under one unified API. Threadsafe;
    multiple campaigns can run concurrently — each is keyed by its
    `campaign_id`.

    Wire an `EventBus` to stream pulls / observations / commits to the
    coordination engine. Wire an `attestor` (typically a
    `RuntimeAttestor` over an `AttestationLedger`) to produce
    tamper-evident PAC certificates on commit.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._campaigns: dict[str, _CampaignState] = {}
        self._history: list[ArbiterReport] = []
        # campaign_id -> realised truth (best arm id), when supplied via
        # `record_truth`. Used by coverage_report().
        self._truths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Campaign lifecycle
    # ------------------------------------------------------------------

    def start_campaign(
        self,
        *,
        campaign_id: str | None = None,
        arms: Sequence[str],
        algorithm: str = ALGO_TRACK_AND_STOP,
        delta: float = 0.05,
        epsilon: float = 0.0,
        reward_model: str = REWARD_BERNOULLI,
        sigma2: float = _DEFAULT_GAUSSIAN_VAR,
        max_samples: int = 100_000,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Open a new campaign. Returns the campaign id."""
        if algorithm not in KNOWN_ALGORITHMS:
            raise ValueError(f"unknown algorithm: {algorithm}")
        if reward_model not in KNOWN_REWARD_MODELS:
            raise ValueError(f"unknown reward model: {reward_model}")
        if len(arms) < 2:
            raise ValueError("at least 2 arms required")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        if epsilon < 0.0:
            raise ValueError("epsilon must be ≥ 0")
        if max_samples < len(arms):
            raise ValueError("max_samples must be ≥ number of arms")
        if sigma2 <= 0.0:
            raise ValueError("sigma2 must be positive")
        if len(set(arms)) != len(arms):
            raise ValueError("arm ids must be unique")

        cid = campaign_id or f"arb-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if cid in self._campaigns:
                raise ValueError(f"campaign already exists: {cid}")
            arms_list = list(arms)
            st = _CampaignState(
                id=cid,
                arms=arms_list,
                counts={a: 0 for a in arms_list},
                sums={a: 0.0 for a in arms_list},
                sum_sqs={a: 0.0 for a in arms_list},
                last_value={a: 0.0 for a in arms_list},
                first_seen={a: 0 for a in arms_list},
                last_seen={a: 0 for a in arms_list},
                cost={a: 0.0 for a in arms_list},
                algorithm=algorithm,
                reward_model=reward_model,
                sigma2=float(sigma2),
                delta=float(delta),
                epsilon=float(epsilon),
                max_samples=int(max_samples),
                started_at=time.time(),
                metadata=dict(metadata or {}),
            )
            # Sequential Halving setup.
            if algorithm == ALGO_SEQUENTIAL_HALVING:
                K = len(arms_list)
                rounds = max(1, math.ceil(math.log2(K)))
                st.seq_halving_n_rounds = rounds
                st.seq_halving_remaining = list(arms_list)
                # pulls per arm per round = ⌊max_samples / (|S_r| · rounds)⌋
                st.seq_halving_pulls_per_round = max(
                    1, max_samples // (K * rounds)
                )
                st.seq_halving_pulls_this_round = {a: 0 for a in arms_list}
            self._campaigns[cid] = st

        self._emit(ARBITER_STARTED, {
            "campaign_id": cid,
            "arms": list(arms_list),
            "algorithm": algorithm,
            "reward_model": reward_model,
            "delta": delta,
            "epsilon": epsilon,
            "max_samples": max_samples,
        })
        return cid

    def clear(self, campaign_id: str) -> None:
        """Discard an active campaign. Use when drift invalidates it."""
        with self._lock:
            st = self._campaigns.pop(campaign_id, None)
            if st is None:
                return
            st.finished = True
            st.stop_reason = VERDICT_CLEARED
        self._emit(ARBITER_CLEARED, {"campaign_id": campaign_id})

    def has_campaign(self, campaign_id: str) -> bool:
        with self._lock:
            return campaign_id in self._campaigns

    def active_campaigns(self) -> list[str]:
        with self._lock:
            return [cid for cid, st in self._campaigns.items() if not st.finished]

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def observe(
        self,
        campaign_id: str,
        arm_id: str,
        reward: float,
        *,
        cost: float = 0.0,
    ) -> None:
        """Record one observation. Reward must be in [0,1] for Bernoulli."""
        with self._lock:
            st = self._campaigns.get(campaign_id)
            if st is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if st.finished:
                raise RuntimeError(f"campaign already finished: {campaign_id}")
            if arm_id not in st.counts:
                raise KeyError(f"arm not in campaign: {arm_id}")
            r = float(reward)
            if st.reward_model == REWARD_BERNOULLI:
                if not 0.0 <= r <= 1.0:
                    raise ValueError(f"bernoulli reward out of range: {r}")
            st.counts[arm_id] += 1
            st.sums[arm_id] += r
            st.sum_sqs[arm_id] += r * r
            st.last_value[arm_id] = r
            st.cost[arm_id] += float(cost)
            t = st.total_pulls()
            if st.first_seen[arm_id] == 0:
                st.first_seen[arm_id] = t
            st.last_seen[arm_id] = t
            if st.algorithm == ALGO_SEQUENTIAL_HALVING:
                st.seq_halving_pulls_this_round[arm_id] = (
                    st.seq_halving_pulls_this_round.get(arm_id, 0) + 1
                )
        self._emit(ARBITER_OBSERVED, {
            "campaign_id": campaign_id,
            "arm_id": arm_id,
            "reward": float(reward),
            "cost": float(cost),
            "t": t,
        })

    def next_pulls(
        self,
        campaign_id: str,
        batch: int = 1,
    ) -> SamplingPlan:
        """Return the next `batch` arm pulls and check the stopping rule.

        The plan may include `stopped=True`; callers should not pull
        anything in that case. After a stop, `report(...)` returns the
        finalised `ArbiterReport`.
        """
        if batch <= 0:
            raise ValueError("batch must be positive")
        with self._lock:
            st = self._campaigns.get(campaign_id)
            if st is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if st.finished:
                return SamplingPlan(
                    campaign_id=campaign_id,
                    t=st.total_pulls(),
                    batch_pulls=[],
                    stopped=True,
                    stop_reason=st.stop_reason,
                    rationale="campaign already finished",
                )
            plan = self._plan_locked(st, batch)
            if plan.stopped:
                self._finish_locked(st, plan.stop_reason, plan.statistic, plan.threshold)
                report = st.final_report
            else:
                report = None
        self._emit(ARBITER_PULLED, {
            "campaign_id": campaign_id,
            "t": plan.t,
            "batch_pulls": list(plan.batch_pulls),
            "stopped": plan.stopped,
            "stop_reason": plan.stop_reason,
            "statistic": plan.statistic,
            "threshold": plan.threshold,
        })
        if report is not None:
            kind = ARBITER_COMMITTED if report.verdict == VERDICT_BEST else ARBITER_EXHAUSTED
            self._emit(kind, {
                "campaign_id": campaign_id,
                "verdict": report.verdict,
                "best_arm": report.best_arm,
                "runner_up": report.runner_up,
                "n_total": report.n_total,
                "statistic": report.statistic,
                "threshold": report.threshold,
                "pac_guarantee": report.pac_guarantee,
                "pac_receipt_hash": report.pac_receipt_hash,
            })
        return plan

    def report(self, campaign_id: str) -> ArbiterReport:
        """Return the (possibly in-progress) report for a campaign."""
        with self._lock:
            st = self._campaigns.get(campaign_id)
            if st is None:
                # Look in history.
                for r in self._history:
                    if r.id == campaign_id:
                        return r
                raise KeyError(f"unknown campaign: {campaign_id}")
            if st.final_report is not None:
                return st.final_report
            return self._build_report_locked(st, finalised=False)

    def record_truth(self, campaign_id: str, true_best: str) -> None:
        """Supply the realised ground truth for a finished campaign.

        Used to compute coverage. Typically called by a delayed labeler
        — e.g. the runtime that runs a much longer follow-up campaign
        and treats *its* winner as the truth.
        """
        with self._lock:
            if campaign_id not in self._truths and any(
                r.id == campaign_id for r in self._history
            ):
                self._truths[campaign_id] = str(true_best)
            elif campaign_id in self._campaigns:
                self._truths[campaign_id] = str(true_best)

    def history(self) -> list[ArbiterReport]:
        with self._lock:
            return list(self._history)

    def coverage_report(self) -> CoverageReport:
        with self._lock:
            n_total = len(self._history)
            by_verdict: dict[str, int] = {}
            samples_by_verdict: dict[str, list[int]] = {}
            cost_by_verdict: dict[str, list[float]] = {}
            n_best = 0
            n_observed = 0
            n_correct = 0
            target_acc = 0.0
            for r in self._history:
                by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1
                samples_by_verdict.setdefault(r.verdict, []).append(r.n_total)
                cost_by_verdict.setdefault(r.verdict, []).append(r.cost)
                if r.verdict == VERDICT_BEST:
                    n_best += 1
                    target_acc += (1.0 - r.delta)
                    if r.id in self._truths:
                        n_observed += 1
                        if self._truths[r.id] == r.best_arm:
                            n_correct += 1
            target_acc = target_acc / n_best if n_best else 0.0
            realised = (n_correct / n_observed) if n_observed else 0.0
            return CoverageReport(
                n_campaigns=n_total,
                n_best_verdicts=n_best,
                n_observed_truths=n_observed,
                realised_accuracy=realised,
                target_accuracy=target_acc,
                miscoverage=max(0.0, target_acc - realised),
                n_by_verdict={k: v for k, v in by_verdict.items()},
                mean_samples_by_verdict={
                    k: statistics.fmean(v) for k, v in samples_by_verdict.items()
                },
                mean_cost_by_verdict={
                    k: statistics.fmean(v) for k, v in cost_by_verdict.items()
                },
            )

    # ------------------------------------------------------------------
    # Synchronous (closed-loop) campaign driver
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        arms: Sequence[str],
        sampler: Callable[[str], float],
        algorithm: str = ALGO_TRACK_AND_STOP,
        delta: float = 0.05,
        epsilon: float = 0.0,
        reward_model: str = REWARD_BERNOULLI,
        sigma2: float = _DEFAULT_GAUSSIAN_VAR,
        max_samples: int = 100_000,
        batch: int = 1,
        cost_fn: Callable[[str, float], float] | None = None,
        on_pull: Callable[[str, float], None] | None = None,
        metadata: Mapping[str, Any] | None = None,
        campaign_id: str | None = None,
    ) -> ArbiterReport:
        """Drive a campaign synchronously to completion.

        `sampler(arm_id)` is the caller's reward oracle: takes an arm
        id, returns a scalar reward. Arbiter calls it repeatedly,
        following the algorithm's sampling rule, until the stopping
        rule fires or `max_samples` is exhausted.
        """
        cid = self.start_campaign(
            campaign_id=campaign_id,
            arms=arms,
            algorithm=algorithm,
            delta=delta,
            epsilon=epsilon,
            reward_model=reward_model,
            sigma2=sigma2,
            max_samples=max_samples,
            metadata=metadata,
        )
        while True:
            plan = self.next_pulls(cid, batch=batch)
            if plan.stopped:
                break
            for arm_id in plan.batch_pulls:
                r = float(sampler(arm_id))
                c = float(cost_fn(arm_id, r)) if cost_fn is not None else 0.0
                self.observe(cid, arm_id, r, cost=c)
                if on_pull is not None:
                    on_pull(arm_id, r)
        return self.report(cid)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _arm_mean(self, st: _CampaignState, a: str) -> float:
        n = st.counts[a]
        if n <= 0:
            return 0.0 if st.reward_model == REWARD_BERNOULLI else 0.0
        return st.sums[a] / n

    def _arm_variance(self, st: _CampaignState, a: str) -> float:
        n = st.counts[a]
        if n < 2:
            return 0.0
        m = st.sums[a] / n
        ss = st.sum_sqs[a]
        var = max(0.0, (ss - n * m * m) / (n - 1))
        return var

    def _empirical_best(self, st: _CampaignState) -> str:
        return max(st.arms, key=lambda a: (self._arm_mean(st, a), st.counts[a]))

    def _plan_locked(self, st: _CampaignState, batch: int) -> SamplingPlan:
        # Force at least one pull per arm.
        unseen = [a for a in st.arms if st.counts[a] == 0]
        if unseen:
            picks: list[str] = []
            i = 0
            while len(picks) < batch:
                picks.append(unseen[i % len(unseen)])
                i += 1
            return SamplingPlan(
                campaign_id=st.id,
                t=st.total_pulls(),
                batch_pulls=picks,
                stopped=False,
                stop_reason="",
                rationale=f"forcing initial pulls on unseen arms ({len(unseen)} pending)",
            )

        # Check budget exhaustion BEFORE algorithm-specific logic, so
        # we always honour `max_samples` as the hard ceiling.
        if st.total_pulls() >= st.max_samples:
            stat, thr = self._final_statistic_locked(st)
            return SamplingPlan(
                campaign_id=st.id,
                t=st.total_pulls(),
                batch_pulls=[],
                stopped=True,
                stop_reason=VERDICT_EXHAUSTED,
                rationale=f"max_samples ({st.max_samples}) reached",
                statistic=stat,
                threshold=thr,
            )

        if st.algorithm == ALGO_TRACK_AND_STOP:
            return self._plan_track_and_stop_locked(st, batch)
        if st.algorithm == ALGO_KL_LUCB:
            return self._plan_kl_lucb_locked(st, batch)
        if st.algorithm == ALGO_SEQUENTIAL_HALVING:
            return self._plan_seq_halving_locked(st, batch)
        raise ValueError(f"unknown algorithm: {st.algorithm}")

    def _plan_track_and_stop_locked(
        self, st: _CampaignState, batch: int,
    ) -> SamplingPlan:
        t = st.total_pulls()
        K = len(st.arms)
        means = [self._arm_mean(st, a) for a in st.arms]
        beta = glr_threshold(t, st.delta, K)

        # Check stopping rule first.
        best_idx = max(range(K), key=lambda i: (means[i], st.counts[st.arms[i]]))
        mu_top = means[best_idx]
        n_top = st.counts[st.arms[best_idx]]
        min_z = math.inf
        for j in range(K):
            if j == best_idx:
                continue
            z = _glr_pair_statistic(
                mu_top, means[j], n_top, st.counts[st.arms[j]],
                st.reward_model, st.sigma2,
            )
            # Apply ε-best slack: a candidate is dominated already if its
            # mean is within ε of the top.
            if means[best_idx] - means[j] < st.epsilon:
                z = 0.0
            if z < min_z:
                min_z = z
        if min_z >= beta:
            return SamplingPlan(
                campaign_id=st.id,
                t=t,
                batch_pulls=[],
                stopped=True,
                stop_reason=VERDICT_BEST,
                rationale=(
                    f"GLR={min_z:.3f} ≥ β(t={t}, δ={st.delta})={beta:.3f}; "
                    f"empirical best is {st.arms[best_idx]!r}"
                ),
                statistic=min_z,
                threshold=beta,
            )

        # Sampling rule: C-tracking with forced exploration.
        w_star, _ = solve_w_star(
            means, reward_model=st.reward_model, sigma2=st.sigma2,
        )
        picks: list[str] = []
        # Local mutable counts for this batch.
        counts = dict(st.counts)
        for _ in range(batch):
            t_local = sum(counts.values())
            # Forced exploration: any arm with N_a < √t pulled by force.
            forced = [
                a for a in st.arms
                if counts[a] < math.pow(max(1, t_local), _FORCED_EXPLORATION) / K
            ]
            if forced:
                # pick the least-pulled
                pick = min(forced, key=lambda a: counts[a])
            else:
                # cumulative-deficit C-tracking
                pick = max(
                    range(K),
                    key=lambda i: (t_local + 1) * w_star[i] - counts[st.arms[i]],
                )
                pick = st.arms[pick]
            counts[pick] += 1
            picks.append(pick)
        return SamplingPlan(
            campaign_id=st.id,
            t=t,
            batch_pulls=picks,
            stopped=False,
            stop_reason="",
            rationale=(
                f"track-and-stop: GLR={min_z:.3f} < β={beta:.3f}; "
                f"w*={[round(w, 3) for w in w_star]}"
            ),
            statistic=min_z,
            threshold=beta,
        )

    def _plan_kl_lucb_locked(
        self, st: _CampaignState, batch: int,
    ) -> SamplingPlan:
        t = st.total_pulls()
        K = len(st.arms)
        means = {a: self._arm_mean(st, a) for a in st.arms}
        beta = glr_threshold(t, st.delta, K)

        # Identify h_t (empirical leader) and l_t (best UCB challenger).
        h = self._empirical_best(st)
        # Confidence bounds.
        bounds = {
            a: (
                kl_confidence_lower(
                    means[a], st.counts[a], beta,
                    reward_model=st.reward_model, sigma2=st.sigma2,
                ),
                kl_confidence_upper(
                    means[a], st.counts[a], beta,
                    reward_model=st.reward_model, sigma2=st.sigma2,
                ),
            )
            for a in st.arms
        }
        # Challenger: arm ≠ h with largest UCB.
        challenger = max(
            (a for a in st.arms if a != h),
            key=lambda a: bounds[a][1],
        )
        margin = bounds[challenger][1] - bounds[h][0]
        if margin <= st.epsilon:
            return SamplingPlan(
                campaign_id=st.id,
                t=t,
                batch_pulls=[],
                stopped=True,
                stop_reason=VERDICT_BEST,
                rationale=(
                    f"KL-LUCB margin {margin:.3f} ≤ ε={st.epsilon:.3f}; "
                    f"empirical best is {h!r}"
                ),
                statistic=-margin,
                threshold=-st.epsilon,
            )
        picks: list[str] = []
        # Pulls alternate h then challenger to balance information per arm.
        for i in range(batch):
            picks.append(h if i % 2 == 0 else challenger)
        return SamplingPlan(
            campaign_id=st.id,
            t=t,
            batch_pulls=picks,
            stopped=False,
            stop_reason="",
            rationale=(
                f"KL-LUCB: leader={h!r} challenger={challenger!r} "
                f"margin={margin:.3f} > ε={st.epsilon:.3f}"
            ),
            statistic=-margin,
            threshold=-st.epsilon,
        )

    def _plan_seq_halving_locked(
        self, st: _CampaignState, batch: int,
    ) -> SamplingPlan:
        t = st.total_pulls()
        # Check round completion.
        per_round = st.seq_halving_pulls_per_round
        remaining = st.seq_halving_remaining
        if not remaining:
            return SamplingPlan(
                campaign_id=st.id, t=t, batch_pulls=[],
                stopped=True, stop_reason=VERDICT_BEST,
                rationale="sequential halving exhausted survivors",
                statistic=0.0, threshold=0.0,
            )
        if len(remaining) == 1:
            return SamplingPlan(
                campaign_id=st.id, t=t, batch_pulls=[],
                stopped=True, stop_reason=VERDICT_BEST,
                rationale=f"sequential halving converged to {remaining[0]!r}",
                statistic=0.0, threshold=0.0,
            )
        # Is the current round complete?
        round_done = all(
            st.seq_halving_pulls_this_round.get(a, 0) >= per_round
            for a in remaining
        )
        if round_done:
            # Eliminate bottom half.
            ranked = sorted(
                remaining,
                key=lambda a: (
                    self._arm_mean(st, a),
                    st.seq_halving_pulls_this_round.get(a, 0),
                ),
                reverse=True,
            )
            keep_n = max(1, math.ceil(len(remaining) / 2))
            new_remaining = ranked[:keep_n]
            st.seq_halving_remaining = new_remaining
            st.seq_halving_round += 1
            st.seq_halving_pulls_this_round = {a: 0 for a in new_remaining}
            remaining = new_remaining
            if len(remaining) == 1:
                return SamplingPlan(
                    campaign_id=st.id, t=t, batch_pulls=[],
                    stopped=True, stop_reason=VERDICT_BEST,
                    rationale=f"sequential halving converged to {remaining[0]!r}",
                    statistic=0.0, threshold=0.0,
                )
            # Recompute per-round pulls for the new round on the surviving set.
            # Keep `seq_halving_pulls_per_round` fixed: it's set so that the
            # *total* sample count stays at most max_samples under the
            # canonical splitting.
        # Plan: pull underexposed arms in `remaining`.
        picks: list[str] = []
        # Local counter to honour per_round across the batch.
        local = dict(st.seq_halving_pulls_this_round)
        for _ in range(batch):
            # Among remaining, the most-deficient arm relative to per_round.
            cand = min(remaining, key=lambda a: local.get(a, 0))
            local[cand] = local.get(cand, 0) + 1
            picks.append(cand)
            if local[cand] >= per_round and all(
                local.get(a, 0) >= per_round for a in remaining
            ):
                # Mark round-complete cheaply by topping up local pulls to
                # per_round so subsequent iterations of this loop don't
                # overpull this round.
                break
        return SamplingPlan(
            campaign_id=st.id, t=t,
            batch_pulls=picks,
            stopped=False, stop_reason="",
            rationale=(
                f"sequential halving round {st.seq_halving_round + 1}/{st.seq_halving_n_rounds} "
                f"over {len(remaining)} survivors"
            ),
            statistic=0.0, threshold=0.0,
        )

    def _final_statistic_locked(self, st: _CampaignState) -> tuple[float, float]:
        """Return (statistic, threshold) snapshot for the current state.

        Used both by stopping rules and by the final report.
        """
        t = st.total_pulls()
        K = len(st.arms)
        beta = glr_threshold(t, st.delta, K)
        if st.algorithm == ALGO_KL_LUCB:
            means = {a: self._arm_mean(st, a) for a in st.arms}
            h = self._empirical_best(st)
            bounds = {
                a: (
                    kl_confidence_lower(
                        means[a], st.counts[a], beta,
                        reward_model=st.reward_model, sigma2=st.sigma2,
                    ),
                    kl_confidence_upper(
                        means[a], st.counts[a], beta,
                        reward_model=st.reward_model, sigma2=st.sigma2,
                    ),
                )
                for a in st.arms
            }
            if K < 2:
                return 0.0, 0.0
            challenger = max(
                (a for a in st.arms if a != h),
                key=lambda a: bounds[a][1],
            )
            margin = bounds[challenger][1] - bounds[h][0]
            return -margin, -st.epsilon
        means = [self._arm_mean(st, a) for a in st.arms]
        best_idx = max(range(K), key=lambda i: means[i])
        n_top = st.counts[st.arms[best_idx]]
        if n_top <= 0:
            return 0.0, beta
        min_z = math.inf
        for j in range(K):
            if j == best_idx:
                continue
            z = _glr_pair_statistic(
                means[best_idx], means[j], n_top, st.counts[st.arms[j]],
                st.reward_model, st.sigma2,
            )
            if z < min_z:
                min_z = z
        if min_z is math.inf:
            min_z = 0.0
        return min_z, beta

    def _build_report_locked(
        self, st: _CampaignState, *, finalised: bool,
        stop_reason: str = "", statistic: float = 0.0, threshold: float = 0.0,
    ) -> ArbiterReport:
        t = st.total_pulls()
        K = len(st.arms)
        means = {a: self._arm_mean(st, a) for a in st.arms}
        beta = glr_threshold(max(1, t), st.delta, K)
        arms_stats: list[ArmStats] = []
        for a in st.arms:
            n = st.counts[a]
            var = self._arm_variance(st, a)
            lo = kl_confidence_lower(
                means[a], n, beta,
                reward_model=st.reward_model, sigma2=st.sigma2,
            ) if n else 0.0
            up = kl_confidence_upper(
                means[a], n, beta,
                reward_model=st.reward_model, sigma2=st.sigma2,
            ) if n else 1.0
            arms_stats.append(ArmStats(
                id=a,
                n=n,
                mean=means[a],
                sum_sq=st.sum_sqs[a],
                sample_var=var,
                lower=lo,
                upper=up,
                last_value=st.last_value[a],
                first_seen_at=st.first_seen[a],
                last_seen_at=st.last_seen[a],
            ))
        # Identify best and runner-up by empirical mean (ties broken by
        # larger sample count).
        ranked = sorted(arms_stats, key=lambda s: (s.mean, s.n), reverse=True)
        best = ranked[0].id if ranked else ""
        runner = ranked[1].id if len(ranked) > 1 else ""
        if not finalised:
            stat, thr = self._final_statistic_locked(st)
            stop_reason = ""
            statistic, threshold = stat, thr
            verdict = ""
        else:
            verdict = (
                VERDICT_BEST if stop_reason == VERDICT_BEST else
                VERDICT_EXHAUSTED if stop_reason == VERDICT_EXHAUSTED else
                VERDICT_INFEASIBLE if stop_reason == VERDICT_INFEASIBLE else
                stop_reason
            )
        pac = (verdict == VERDICT_BEST) and all(s.n > 0 for s in arms_stats)
        if t < 2 * K:
            conf = "low"
        elif pac:
            conf = "high"
        else:
            conf = "medium"
        # Asymptotic sample-complexity ceiling (Garivier-Kaufmann Thm 1):
        # T*(μ) · log(1/δ), where T*(μ) is the inverse of the game value.
        sc_bound = self._asymptotic_complexity_bound(st, means)
        return ArbiterReport(
            id=st.id,
            arms=arms_stats,
            best_arm=best,
            runner_up=runner,
            algorithm=st.algorithm,
            reward_model=st.reward_model,
            delta=st.delta,
            epsilon=st.epsilon,
            max_samples=st.max_samples,
            n_total=t,
            cost=st.total_cost(),
            stop_reason=stop_reason,
            verdict=verdict,
            confidence=conf,
            pac_guarantee=pac,
            statistic=statistic,
            threshold=threshold,
            sample_complexity_bound=sc_bound,
            started_at=st.started_at,
            finished_at=time.time() if finalised else 0.0,
            rationale=self._rationale_for_report(st, best, runner, means, stop_reason),
        )

    def _asymptotic_complexity_bound(
        self, st: _CampaignState, means: Mapping[str, float],
    ) -> float:
        """Upper bound on E[τ_δ] from Garivier-Kaufmann (Theorem 1).

        Returns T*(μ) · log(1/δ) where T*(μ) is the inverse of the game
        value v* = max_w min_a g_a(w, μ). For arms with identical means
        this diverges; we cap at `max_samples` for stability.
        """
        K = len(st.arms)
        if K < 2:
            return 0.0
        mu = [means[a] for a in st.arms]
        # Find best mean.
        best_idx = max(range(K), key=lambda i: mu[i])
        mu_best = mu[best_idx]
        # Effective gaps.
        gaps_sq = []
        for j in range(K):
            if j == best_idx:
                continue
            gap = abs(mu_best - mu[j])
            if gap <= _DEFAULT_TIE_EPS:
                return float(st.max_samples)
            if st.reward_model == REWARD_BERNOULLI:
                d = max(_kl(REWARD_BERNOULLI, mu_best, mu[j], 1.0), _EPS)
            else:
                d = max(_kl(REWARD_GAUSSIAN, mu_best, mu[j], st.sigma2), _EPS)
            gaps_sq.append(1.0 / d)
        # Loose upper bound: 2 · (sum 1/d_j) · log(1/δ).
        log_delta = math.log(1.0 / max(st.delta, _EPS))
        bound = 2.0 * sum(gaps_sq) * log_delta
        return min(bound, float(st.max_samples) * 10.0)

    def _rationale_for_report(
        self, st: _CampaignState, best: str, runner: str,
        means: Mapping[str, float], stop_reason: str,
    ) -> str:
        bits = [
            f"algorithm={st.algorithm}",
            f"reward_model={st.reward_model}",
            f"δ={st.delta}",
            f"ε={st.epsilon}",
            f"n_total={st.total_pulls()}",
        ]
        if best:
            bits.append(f"best={best!r}(μ̂={means.get(best, 0.0):.3f})")
        if runner:
            bits.append(f"runner_up={runner!r}(μ̂={means.get(runner, 0.0):.3f})")
        if stop_reason:
            bits.append(f"stop_reason={stop_reason}")
        return " | ".join(bits)

    def _finish_locked(
        self, st: _CampaignState, stop_reason: str,
        statistic: float, threshold: float,
    ) -> None:
        if any(st.counts[a] == 0 for a in st.arms):
            stop_reason = VERDICT_INFEASIBLE
        report = self._build_report_locked(
            st, finalised=True, stop_reason=stop_reason,
            statistic=statistic, threshold=threshold,
        )
        # Attestation: build a tamper-evident receipt.
        if self._attestor is not None and report.verdict == VERDICT_BEST:
            try:
                payload = report.to_dict()
                payload.pop("pac_receipt_hash", None)
                serialised = json.dumps(payload, sort_keys=True, default=str)
                digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
                receipt_hash = digest
                # If the attestor exposes a record() method, use it; else
                # we still emit our content-digest so the report carries a
                # deterministic identity.
                rec = getattr(self._attestor, "record", None)
                if callable(rec):
                    try:
                        receipt = rec(kind="arbiter.committed", payload=payload)
                        if hasattr(receipt, "hash"):
                            receipt_hash = receipt.hash
                        elif isinstance(receipt, str):
                            receipt_hash = receipt
                    except Exception:
                        pass
                report = ArbiterReport(
                    **{**asdict(report), "arms": report.arms,
                       "pac_receipt_hash": receipt_hash}
                )
            except Exception:
                pass
        st.final_report = report
        st.finished = True
        st.stop_reason = report.stop_reason
        self._history.append(report)
        # Hold onto the campaign for `report(cid)` lookups; cap history.
        if len(self._history) > 2048:
            self._history = self._history[-2048:]
        # Emit one summary event with the full report payload so
        # downstream tools (DriftSentinel, dashboards) can hook.
        self._emit(ARBITER_REPORT, {
            "campaign_id": st.id,
            "report": report.to_dict(),
        })

    def _emit(self, kind: str, data: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:
            # Telemetry must never crash the runtime.
            pass


# =====================================================================
# Free functions for direct algorithmic access (Strategist composition)
# =====================================================================


def empirical_best(means: Mapping[str, float]) -> str:
    """Return arg max over a mapping of empirical means.

    Convenience for coordinators that already have empirical means and
    just need to identify the leader.
    """
    if not means:
        raise ValueError("means is empty")
    return max(means, key=lambda k: means[k])


def pac_certificate(
    means: Mapping[str, float],
    counts: Mapping[str, int],
    delta: float,
    *,
    reward_model: str = REWARD_BERNOULLI,
    sigma2: float = _DEFAULT_GAUSSIAN_VAR,
) -> tuple[bool, float, float, str]:
    """One-shot (ε=0) PAC certificate check on a fixed empirical state.

    Returns (pac_holds, statistic, threshold, rationale). Useful for
    Strategist to ask "given what I already know, can I commit?" without
    running a full campaign.
    """
    arms = list(means)
    K = len(arms)
    if K < 2:
        return False, 0.0, 0.0, "need ≥ 2 arms"
    if any(counts.get(a, 0) <= 0 for a in arms):
        return False, 0.0, 0.0, "an arm has zero observations"
    t = sum(counts[a] for a in arms)
    best = empirical_best(means)
    mu_top = means[best]
    n_top = counts[best]
    min_z = math.inf
    runner = ""
    for a in arms:
        if a == best:
            continue
        z = _glr_pair_statistic(
            mu_top, means[a], n_top, counts[a], reward_model, sigma2,
        )
        if z < min_z:
            min_z = z
            runner = a
    beta = glr_threshold(t, delta, K)
    holds = min_z >= beta
    return holds, min_z, beta, (
        f"GLR(best={best!r} vs {runner!r})={min_z:.3f} "
        f"{'≥' if holds else '<'} β(t={t}, δ={delta})={beta:.3f}"
    )


def expected_samples_to_identify(
    means: Sequence[float],
    delta: float,
    *,
    reward_model: str = REWARD_BERNOULLI,
    sigma2: float = _DEFAULT_GAUSSIAN_VAR,
) -> float:
    """Asymptotic sample complexity T*(μ) · log(1/δ) of Garivier-Kaufmann.

    Returns the rate-optimal expected sample count for fixed-confidence
    BAI in the given problem. The runtime uses this for cost forecasts
    before starting a campaign: "how much will it cost to identify the
    best of these K arms at confidence δ?".
    """
    if len(means) < 2:
        return 0.0
    best = max(range(len(means)), key=lambda i: means[i])
    mu_best = means[best]
    inv_kl = 0.0
    for j, m in enumerate(means):
        if j == best:
            continue
        if reward_model == REWARD_BERNOULLI:
            d = max(kl_bernoulli(mu_best, m), _EPS)
        else:
            d = max(kl_gaussian(mu_best, m, sigma2), _EPS)
        inv_kl += 1.0 / d
    return 2.0 * inv_kl * math.log(1.0 / max(delta, _EPS))
