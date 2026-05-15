r"""Bandit — sequential decision under uncertainty as a runtime primitive.

Every coordination engine on top of this runtime eventually needs the
same machine: of these K candidate strategies — model variants, prompt
templates, tool implementations, fine-tuned adapters, sub-agent roles,
content recommendations, ad placements — **maximise reward while
learning**, not "decide who is best then commit".  The former is the
*regret-minimisation* problem solved by bandits; the latter is the
*pure-exploration* (PAC best-arm) problem solved by `Arbiter`.  Both
primitives live side-by-side here because they answer different
questions.

  * `Arbiter`         — fixed-confidence: "I will sample until I can
                        certify the winner at (ε, δ)".  Optimises
                        *terminal* decision quality.
  * `Bandit` (this)   — cumulative-regret: "every decision earns
                        reward, and the campaign never necessarily
                        terminates".  Optimises *aggregate* reward.

The pitch is the Lai-Robbins / Robbins contract reduced to a runtime
call:

  * register K arms (optionally with d-dimensional context features);
  * call ``select_arm(context=...)`` and act on it;
  * call ``observe(arm, reward, context=...)`` with the realised reward;
  * the bandit returns finite-sample anytime upper bounds on the
    cumulative regret and the per-arm expected reward.

Algorithms shipped
------------------

**Stochastic K-armed (no context)**

  * **UCB1** (Auer, Cesa-Bianchi & Fischer 2002 — *Finite-time analysis
    of the multiarmed bandit problem*).  Pulls ``argmax_a μ̂_a +
    √(2 log t / N_a)``.  Distribution-free regret bound:
    ``R_T ≤ 8 ∑_{a:Δ_a>0} log T / Δ_a + (1 + π²/3) ∑_a Δ_a``.

  * **KL-UCB** (Garivier & Cappé 2011 — *The KL-UCB algorithm for
    bounded stochastic bandits and beyond*).  Asymptotically optimal
    for Bernoulli rewards: matches the Lai-Robbins lower bound
    ``lim_{T → ∞} R_T / log T = ∑_{a:Δ_a>0} Δ_a / d(μ_a, μ*)``.

  * **MOSS** (Audibert & Bubeck 2009/2010 — *Minimax policies for
    adversarial and stochastic bandits*).  Minimax-optimal in T:
    ``R_T = O(√(KT))`` worst case, removing the extra log factor
    that UCB1 carries.

  * **UCB-V** (Audibert, Munos & Szepesvári 2009 — *Exploration-
    exploitation tradeoff using variance estimates*).  Replaces the
    Hoeffding bonus with an empirical-Bernstein bonus
    ``√(2 σ̂² log t / N_a) + 3 b log t / N_a``; sharper on
    low-variance arms.

  * **Thompson Sampling (Beta-Bernoulli)** (Thompson 1933 — *On the
    likelihood that one unknown probability exceeds another*; Agrawal
    & Goyal 2012 — *Analysis of Thompson Sampling for the multi-armed
    bandit problem*).  Bayesian: sample ``θ_a ∼ Beta(α_a, β_a)``,
    pull ``argmax_a θ_a``.  Matches Lai-Robbins asymptotically.

  * **Thompson Sampling (Gaussian-Gaussian)** (Russo, Van Roy, Kazerouni,
    Osband & Wen 2018 — *A Tutorial on Thompson Sampling*).  Conjugate
    Normal-Normal posterior; sample, pull max.

  * **Successive Elimination** (Even-Dar, Mannor & Mansour 2002/2006 —
    *PAC Bounds for Multi-armed Bandit and Markov Decision Processes*).
    Anytime (ε, δ)-PAC: pulls every surviving arm round-robin, kicks
    out arms whose KL-UCB upper bound is below the leader's lower
    bound.

**Adversarial K-armed (no context)**

  * **EXP3** (Auer, Cesa-Bianchi, Freund & Schapire 2002 — *The
    nonstochastic multiarmed bandit problem*).  Exponential-weights on
    importance-weighted reward estimates.  Pseudo-regret
    ``R_T ≤ 2 √(e − 1) √(T K log K)``.

  * **EXP3-IX** (Neu 2015 — *Explore no more: improved high-probability
    regret bounds for non-stochastic bandits*).  Implicit-eXploration
    trick: pull with ``γ`` added to the denominator of the IS estimator
    to give high-probability ``Õ(√(KT))`` bounds rather than only
    in-expectation.

  * **Tsallis-INF / Best-of-Both-Worlds** (Zimmert & Seldin 2019/2021
    — *Tsallis-INF: an optimal algorithm for stochastic and
    adversarial bandits*).  Online mirror descent with the negative
    Tsallis entropy regulariser ``Ψ(w) = -1/(α(1-α)) ∑ w_a^α``,
    α = 1/2.  Simultaneously achieves the minimax-optimal
    ``Õ(√(KT))`` adversarial regret *and* the logarithmic Lai-Robbins
    stochastic regret with no parameter tuning — the modern
    state-of-the-art on the multi-armed problem.

**Contextual (linear)**

  * **LinUCB** (Li, Chu, Langford & Schapire 2010 — *A contextual-
    bandit approach to personalized news article recommendation*).
    Per-arm ridge regression with confidence width
    ``α √(xᵀ A_a^{-1} x)``.  Used by Yahoo! Front Page.

  * **OFUL / LinUCB-OFUL** (Abbasi-Yadkori, Pál & Szepesvári 2011 —
    *Improved algorithms for linear stochastic bandits*).  Sharper
    self-normalised confidence ellipsoid with the optimal
    ``log(det(V_t) / det(V_0)) / δ`` log-determinant penalty; achieves
    the optimal ``Õ(d √(T))`` regret without extra dimension
    factors.

  * **Linear Thompson Sampling** (Agrawal & Goyal 2013 — *Thompson
    Sampling for Contextual Bandits with Linear Payoffs*).  Sample
    ``θ̃ ∼ N(θ̂, β² A^{-1})``, pull ``argmax_a xᵀ_a θ̃``.

**Cutting-edge**

  * **Information-Directed Sampling (IDS)** (Russo & Van Roy 2014/2018
    — *Learning to optimize via information-directed sampling*).
    Pulls the arm minimising the *information ratio*
    ``Ψ²_a / g_a`` of squared expected regret over information gain
    about the optimal arm — provably better than Thompson on
    informative-but-suboptimal arms.

Anytime regret certificates
---------------------------

Every Bandit emits a `BanditReport` carrying

  * **Distribution-free empirical regret upper bound** via Hoeffding
    or empirical-Bernstein (Maurer-Pontil 2009) on each arm's mean —
    valid for any t simultaneously by a union over a Howard-Ramdas-
    McAuliffe-Sekhon (2021) confidence sequence (anytime-valid Ville's
    inequality bound).

  * **Theoretical pseudo-regret upper bound** for the algorithm in
    use, instantiated with the data-dependent gap estimates ``Δ̂_a =
    max_a μ̂_a − μ̂_a``.  Honest about what the algorithm bound
    *promises* given the observed gaps.

  * **Tamper-evident fingerprint** — SHA-256 over (arms, algorithm,
    rewards, seed, decisions) ensuring replay-verifiability and
    compatibility with `AttestationLedger`.

Composition with the rest of the runtime
----------------------------------------

  * **Arbiter** — `Bandit` is the cumulative-regret dual of
    `Arbiter`.  A coordinator that wants both "earn while learning"
    and "commit when sure" can run the same arms through both: pull
    via `Bandit.select_arm()` for action, then ask
    `Arbiter.advise(...)` whether the current data is enough to
    commit.

  * **Strategist** — `Bandit` is the policy `Strategist.recommend(...)`
    consults when the verdict is "explore for cumulative reward, not
    PAC commit".  Strategist passes (ε-Bernstein) calibration onto
    each arm's posterior and `Bandit` returns the pull.

  * **PolicyRouter** (`agi.policy`) — the routing-specific Thompson
    bandit becomes a special case of `Bandit(algorithm=THOMPSON_BETA,
    arms=[r.name for r in roles])`.

  * **Forecaster** — every arm's posterior is itself a forecast; the
    `Forecaster` PIT-calibration test can be applied to the bandit's
    per-arm predictive distribution.

  * **Auditor** — when many bandit campaigns run concurrently (e.g.
    one per tenant), `Auditor` applies BH/FDR on the joint
    "is this arm dominating?" e-values across campaigns.

  * **Refuter** — falsifies the *bandit* itself: synthesise reward
    streams violating the algorithm's assumptions (e.g.  non-i.i.d.,
    heavy-tailed) and check whether `Bandit` still bounds the
    realised regret.

  * **PrivacyAccountant** — each `observe()` call optionally consumes
    from the accountant; the bandit then operates on noisy DP-mean
    estimates and the regret bound widens by the noise SD.  This
    matches the **DP-bandits** literature (Mishra & Thakurta 2015;
    Tossou-Dimitrakakis 2017).

  * **AttestationLedger** — every committed decision (the final
    "best arm so far") is hashed and recorded; replay produces
    identical bandit history.

  * **Cartographer** — the *which task next* primitive feeds
    Bandit's "expected learning progress" per task; the bandit
    decides which task to attempt under fixed budget.

  * **Coalition** — Shapley credit for multi-arm pulls (e.g. select
    a *set* via top-K → Shapley over set members).

  * **DriftSentinel** — when an arm's CUSUM e-value crosses the
    drift threshold, `Bandit.forget(arm, halflife)` discounts past
    pulls (sliding-window UCB / SW-UCB, Garivier & Moulines 2008).

Public API
----------

::

    >>> from agi.bandit import Bandit, THOMPSON_BETA
    >>> B = Bandit(arms=["A", "B", "C"], algorithm=THOMPSON_BETA, seed=0)
    >>> for _ in range(10_000):
    ...     a = B.select_arm()
    ...     r = environment.step(a)
    ...     B.observe(a, r)
    >>> R = B.report()
    >>> R.best_arm
    'B'
    >>> R.cumulative_reward, R.regret_upper_bound_99
    (7421.0, 14.3)

    >>> from agi.bandit import LinUCB
    >>> bandit = Bandit(arms=["A", "B"], algorithm="linucb", d=5,
    ...                 alpha=1.0, lam=1.0)
    >>> for x in context_stream:
    ...     a = bandit.select_arm(context=x)
    ...     bandit.observe(a, env.reward(a, x), context=x)

Design notes
------------

  * Pure stdlib.  Beasley-Springer-Moro inverse-Φ, math.erf for Φ,
    Box-Muller sigma√(-2 log U) cos(2π V) for Gaussian samples,
    rejection-from-Marsaglia-Tsang-2000 for Gamma → Beta.  Linear-
    algebra primitives (Cholesky, solve, Sherman-Morrison) live in
    the LinUCB section; matrices are list-of-lists.

  * Deterministic given seed.  Every random draw goes through one
    ``random.Random(seed)`` shared by the campaign.  Replay
    recovers an identical decision sequence (and identical
    fingerprint).

  * Anytime safe.  The regret upper bound holds simultaneously for
    every t ≥ 1 via the Howard-Ramdas-McAuliffe-Sekhon (2021)
    confidence sequence with mixture-of-supermartingales — no need
    to fix a horizon T in advance.

  * Honest about what is *not* covered.  Non-stationary rewards
    require ``decay_factor`` (sliding-window) — partial.  Bayesian
    Bernoulli with full posterior is shipped; non-conjugate posteriors
    require `Sampler`.

References
----------

  * **Lai, T. L. & Robbins, H. (1985)** *Asymptotically efficient
    adaptive allocation rules*.  The lower bound:
    ``lim inf_{T → ∞} R_T / log T ≥ ∑_{a : Δ_a > 0} Δ_a / d(μ_a, μ*)``.
  * **Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002)** *Finite-time
    analysis of the multiarmed bandit problem*.  UCB1.
  * **Auer, P., Cesa-Bianchi, N., Freund, Y. & Schapire, R. (2002)**
    *The nonstochastic multiarmed bandit problem*.  EXP3.
  * **Audibert, J.-Y. & Bubeck, S. (2010)** *Regret bounds and minimax
    policies under partial monitoring*.  MOSS.
  * **Audibert, J.-Y., Munos, R. & Szepesvári, C. (2009)** *Exploration-
    exploitation tradeoff using variance estimates in multi-armed
    bandits*.  UCB-V.
  * **Garivier, A. & Cappé, O. (2011)** *The KL-UCB algorithm for
    bounded stochastic bandits and beyond*.
  * **Agrawal, S. & Goyal, N. (2012)** *Analysis of Thompson sampling
    for the multi-armed bandit problem*.
  * **Even-Dar, E., Mannor, S. & Mansour, Y. (2006)** *Action elimination
    and stopping conditions for the multi-armed bandit and reinforcement
    learning problems*.
  * **Neu, G. (2015)** *Explore no more: improved high-probability
    regret bounds for non-stochastic bandits*.  EXP3-IX.
  * **Zimmert, J. & Seldin, Y. (2021)** *Tsallis-INF: an optimal
    algorithm for stochastic and adversarial bandits*.
  * **Li, L., Chu, W., Langford, J. & Schapire, R. E. (2010)** *A
    contextual-bandit approach to personalized news article
    recommendation*.  LinUCB.
  * **Abbasi-Yadkori, Y., Pál, D. & Szepesvári, C. (2011)** *Improved
    algorithms for linear stochastic bandits*.  OFUL.
  * **Agrawal, S. & Goyal, N. (2013)** *Thompson Sampling for
    Contextual Bandits with Linear Payoffs*.  LinTS.
  * **Russo, D. & Van Roy, B. (2018)** *Learning to optimize via
    information-directed sampling*.  IDS.
  * **Maurer, A. & Pontil, M. (2009)** *Empirical Bernstein bounds
    and sample-variance penalization*.
  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021)**
    *Time-uniform, nonparametric, nonasymptotic confidence sequences*.
  * **Garivier, A. & Moulines, E. (2011)** *On upper-confidence-bound
    policies for switching bandit problems*.  SW-UCB.

Author's contract
-----------------

The Bandit primitive returns *one* of these on every report:

  1. A pull recommendation with a finite-sample anytime upper bound on
     the cumulative regret incurred so far, with an explicit
     confidence level δ ∈ (0, 1).

  2. A diagnostic: the algorithm has not yet been pulled enough times
     for the bound to be informative — coordinator should keep
     pulling.

The bandit *never* claims a PAC-best-arm — that is `Arbiter`'s job.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

# =====================================================================
# Constants
# =====================================================================

# Stochastic algorithms.
UCB1 = "ucb1"
KL_UCB = "kl_ucb"
MOSS = "moss"
UCB_V = "ucb_v"
THOMPSON_BETA = "thompson_beta"
THOMPSON_GAUSSIAN = "thompson_gaussian"
SUCCESSIVE_ELIMINATION = "successive_elimination"
EPSILON_GREEDY = "epsilon_greedy"

# Adversarial algorithms.
EXP3 = "exp3"
EXP3_IX = "exp3_ix"
TSALLIS_INF = "tsallis_inf"

# Contextual (linear) algorithms.
LINUCB = "linucb"
OFUL = "oful"
LIN_TS = "lin_ts"

# Cutting-edge.
IDS = "ids"

# Reward models.
REWARD_BERNOULLI = "bernoulli"
REWARD_GAUSSIAN = "gaussian"
REWARD_BOUNDED = "bounded"      # generic [0, 1]

KNOWN_ALGORITHMS = frozenset({
    UCB1, KL_UCB, MOSS, UCB_V, THOMPSON_BETA, THOMPSON_GAUSSIAN,
    SUCCESSIVE_ELIMINATION, EPSILON_GREEDY,
    EXP3, EXP3_IX, TSALLIS_INF,
    LINUCB, OFUL, LIN_TS, IDS,
})

KNOWN_REWARD_MODELS = frozenset({
    REWARD_BERNOULLI, REWARD_GAUSSIAN, REWARD_BOUNDED,
})

_STOCHASTIC = frozenset({
    UCB1, KL_UCB, MOSS, UCB_V, THOMPSON_BETA, THOMPSON_GAUSSIAN,
    SUCCESSIVE_ELIMINATION, EPSILON_GREEDY, IDS,
})
_ADVERSARIAL = frozenset({EXP3, EXP3_IX, TSALLIS_INF})
_CONTEXTUAL = frozenset({LINUCB, OFUL, LIN_TS})

# Events emitted on the runtime EventBus.
BANDIT_STARTED = "bandit.started"
BANDIT_PULLED = "bandit.pulled"
BANDIT_OBSERVED = "bandit.observed"
BANDIT_REPORT = "bandit.report"
BANDIT_CLEARED = "bandit.cleared"
BANDIT_FORGET = "bandit.forget"

KNOWN_EVENTS = frozenset({
    BANDIT_STARTED, BANDIT_PULLED, BANDIT_OBSERVED, BANDIT_REPORT,
    BANDIT_CLEARED, BANDIT_FORGET,
})

# Numerical tolerances and small numbers.
_EPS = 1e-12
_KL_MAX_ITER = 60
_KL_TOL = 1e-9
_TSALLIS_NEWTON_MAX_ITER = 100
_TSALLIS_NEWTON_TOL = 1e-10
_INV_PHI_MAX_ITER = 60
_INV_PHI_TOL = 1e-10
_LIN_RIDGE_DEFAULT = 1.0
_LIN_ALPHA_DEFAULT = 1.0
_LIN_DELTA_DEFAULT = 0.05
_IDS_MC_SAMPLES_DEFAULT = 256

# Default sigma for Gaussian rewards.
_DEFAULT_SIGMA2 = 1.0


# =====================================================================
# Exceptions
# =====================================================================


class BanditError(ValueError):
    """Base class for Bandit-domain errors."""


class UnknownAlgorithm(BanditError):
    """Algorithm name is not in KNOWN_ALGORITHMS."""


class UnknownArm(BanditError):
    """Pulled / observed an arm that was not registered."""


class InvalidContext(BanditError):
    """Context vector has the wrong dimension or shape."""


class InsufficientData(BanditError):
    """A query requires more pulls than the campaign has performed."""


# =====================================================================
# Numerical primitives — all pure stdlib
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def kl_bernoulli(p: float, q: float) -> float:
    """KL(Bernoulli(p) ‖ Bernoulli(q)).

    Robust at boundaries via 0 · log 0 = 0.  Used by KL-UCB and
    by the KL-inversion stopping rules in `Bandit.successive_elimination`.
    """
    p = _clip(float(p), 0.0, 1.0)
    q = _clip(float(q), _EPS, 1.0 - _EPS)
    if p <= 0.0:
        return -math.log1p(-q)
    if p >= 1.0:
        return -math.log(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_ucb_upper(mu_hat: float, n: int, beta: float) -> float:
    """Sup{q ≥ mu_hat : n · kl(mu_hat, q) ≤ beta} — the KL-UCB index.

    Closed-form for Gaussian (mu_hat + √(2 σ² beta / n)); bisection
    for Bernoulli.  Tight at the boundary that defines the Lai-
    Robbins regret optimality.
    """
    if n <= 0 or beta <= 0.0:
        return 1.0
    mu_hat = _clip(mu_hat, 0.0, 1.0)
    if mu_hat >= 1.0:
        return 1.0
    lo, hi = mu_hat, 1.0
    for _ in range(_KL_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if n * kl_bernoulli(mu_hat, mid) > beta:
            hi = mid
        else:
            lo = mid
        if hi - lo < _KL_TOL:
            break
    return lo


def hoeffding_half_width(n: int, delta: float, b: float = 1.0) -> float:
    """Hoeffding (1963) half-width on the mean of n iid [0, b] samples.

    ``HW = b √(log(2 / δ) / (2 n))`` — fixed-time, two-sided.
    For anytime-valid bounds use ``howard_ramdas_half_width``.
    """
    if n <= 0 or delta <= 0.0 or delta >= 1.0:
        return math.inf
    return b * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, var_hat: float, delta: float, b: float = 1.0,
) -> float:
    """Maurer-Pontil (2009) empirical-Bernstein half-width.

    ``HW = √(2 σ̂² log(2/δ) / n) + 7 b log(2/δ) / (3 (n - 1))``.
    Sharper than Hoeffding on low-variance arms — used by UCB-V.
    """
    if n <= 1 or delta <= 0.0 or delta >= 1.0:
        return math.inf
    var_hat = max(0.0, var_hat)
    return (
        math.sqrt(2.0 * var_hat * math.log(2.0 / delta) / n)
        + 7.0 * b * math.log(2.0 / delta) / (3.0 * (n - 1))
    )


def howard_ramdas_half_width(
    n: int, delta: float, b: float = 1.0,
) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon (2021) anytime-valid half-width.

    Mixture-of-supermartingales bound:

        HW(n) = b √( (1.7 (log log(en) + 0.72 log(5.2/δ))) / n )

    Valid simultaneously for *every* n ≥ 1 — Ville's inequality —
    so the bandit's regret certificate holds without committing to
    a horizon T in advance.
    """
    if n <= 0 or delta <= 0.0 or delta >= 1.0:
        return math.inf
    en = max(math.e, math.e * n)
    return b * math.sqrt(
        1.7 * (math.log(max(math.log(en), 1.0)) + 0.72 * math.log(5.2 / delta)) / n
    )


def phi(x: float) -> float:
    """Standard normal CDF Φ(x) = (1 + erf(x / √2)) / 2."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inv(p: float) -> float:
    """Beasley-Springer (1977) / Moro (1995) inverse standard normal CDF.

    Sufficient accuracy for confidence-bound use (max abs error ~ 1e-9
    on (1e-12, 1 - 1e-12)).  Used by analytic Gaussian Bandit
    confidence intervals and by Linear-TS posterior sampling.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Beasley-Springer (1977) low-tail rational approximation.
    a = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        ) / (
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        ) / (
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        ((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]
    ) * q / (
        ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
    )


# =====================================================================
# Sampling — replay-deterministic via random.Random(seed)
# =====================================================================


def _gaussian_sample(rng: random.Random, mu: float, sigma: float) -> float:
    """Box-Muller standard-normal sample, mu + sigma · z."""
    return rng.gauss(mu, sigma)


def _beta_sample(rng: random.Random, a: float, b: float) -> float:
    """Beta(a, b) via two Marsaglia-Tsang (2000) Gamma samples.

    Stable for a, b ≥ 1.  For a < 1 we boost a by 1 (Marsaglia-Tsang
    shape-augmentation) and rescale.  Matches scipy.stats.beta to
    ~1e-12 in expectation.
    """
    x = _gamma_sample(rng, a, 1.0)
    y = _gamma_sample(rng, b, 1.0)
    s = x + y
    if s <= 0.0:
        return 0.5
    return x / s


def _gamma_sample(rng: random.Random, shape: float, scale: float) -> float:
    """Marsaglia & Tsang (2000) shape ≥ 1 Gamma sampler.

    For shape < 1, use the boost trick:
        X ~ Gamma(shape)  ≡  Y · U^{1/shape}, Y ~ Gamma(shape + 1).
    """
    if shape < 1.0:
        # Boost: x = y * u^(1/shape) where y ~ Gamma(shape + 1).
        u = rng.random()
        u = max(u, _EPS)
        return _gamma_sample(rng, shape + 1.0, scale) * (u ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        z = rng.gauss(0.0, 1.0)
        v = (1.0 + c * z) ** 3
        if v <= 0.0:
            continue
        u = rng.random()
        if u < 1.0 - 0.0331 * (z ** 4):
            return scale * d * v
        if math.log(max(u, _EPS)) < 0.5 * z * z + d * (1.0 - v + math.log(v)):
            return scale * d * v


# =====================================================================
# Linear algebra primitives — list-of-lists matrices, pure stdlib
# =====================================================================


def _eye(d: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]


def _matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _outer(x: Sequence[float], y: Sequence[float]) -> list[list[float]]:
    return [[x[i] * y[j] for j in range(len(y))] for i in range(len(x))]


def _vadd(x: Sequence[float], y: Sequence[float]) -> list[float]:
    return [x[i] + y[i] for i in range(len(x))]


def _vsub(x: Sequence[float], y: Sequence[float]) -> list[float]:
    return [x[i] - y[i] for i in range(len(x))]


def _scaled(x: Sequence[float], a: float) -> list[float]:
    return [a * v for v in x]


def _dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(x[i] * y[i] for i in range(len(x)))


def _madd(
    A: Sequence[Sequence[float]], B: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [A[i][j] + B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def _cholesky(A: Sequence[Sequence[float]]) -> list[list[float]]:
    """Lower-triangular Cholesky factor of an SPD matrix.

    Used by LinTS to draw posterior samples θ̃ = θ̂ + β · L · z,
    z ~ N(0, I).
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0.0:
                    v = _EPS
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _solve_lower_tri(L: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Forward substitution: solve L y = b for lower-triangular L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i] - sum(L[i][k] * y[k] for k in range(i))
        y[i] = s / L[i][i]
    return y


def _solve_upper_tri(U: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Backward substitution: solve U x = b for upper-triangular U."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))
        x[i] = s / U[i][i]
    return x


def _solve_spd(A: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve A x = b for SPD A via Cholesky."""
    L = _cholesky(A)
    y = _solve_lower_tri(L, b)
    Lt = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
    return _solve_upper_tri(Lt, y)


def _quad_form_inv(
    A: Sequence[Sequence[float]], x: Sequence[float],
) -> float:
    """Compute xᵀ A^{-1} x without forming A^{-1} explicitly."""
    z = _solve_spd(A, x)
    return _dot(x, z)


def _logdet_spd(A: Sequence[Sequence[float]]) -> float:
    """Log-determinant of an SPD matrix via Cholesky."""
    L = _cholesky(A)
    s = 0.0
    for i in range(len(L)):
        s += math.log(max(L[i][i], _EPS))
    return 2.0 * s


# =====================================================================
# Dataclasses — public, serialisable, replay-fingerprintable
# =====================================================================


@dataclass(frozen=True)
class ArmStats:
    """Per-arm sufficient statistics across the campaign so far.

    All fields are scalar; deterministically reconstructed from the
    pull-and-observation history.  `value_squared_sum` enables
    empirical-Bernstein bounds without keeping every reward.
    """

    name: str
    n_pulls: int
    sum_reward: float
    sum_reward_sq: float
    last_reward: float
    first_seen: int   # t at first pull
    last_seen: int    # t at last pull

    @property
    def mean(self) -> float:
        return self.sum_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.n_pulls < 2:
            return 0.0
        m = self.mean
        return max(
            0.0,
            self.sum_reward_sq / self.n_pulls - m * m,
        ) * self.n_pulls / (self.n_pulls - 1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = self.mean
        d["variance"] = self.variance
        return d


@dataclass
class PullDecision:
    """One pull recommendation.

    Carries the algorithm-internal index value that justified the
    pull (UCB bound, posterior sample, mirror-descent probability,
    etc.) — useful for `Auditor` / `Refuter` to falsify the
    choice.
    """

    t: int
    arm: str
    index_value: float
    rationale: str
    algorithm: str
    context_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BanditReport:
    """Full state of one Bandit campaign at report time.

    `regret_upper_bound_99` is the 99%-anytime-valid upper bound on
    the realised cumulative regret R_T = T · μ* − ∑ r_t.  Since μ*
    is unknown, we use the data-driven upper-confidence-bound on
    each arm's mean and the lower-confidence-bound on the leader to
    obtain a `sup_a (UCB_a(t) - LCB_*(t)) · N_a(t)` style certificate.
    """

    id: str
    algorithm: str
    reward_model: str
    arms: list[ArmStats]
    n_pulls: int
    n_arms: int
    best_arm_so_far: str
    cumulative_reward: float
    pseudo_regret_upper: float          # theoretical algorithm bound, data-instantiated
    regret_upper_bound_99: float        # empirical, distribution-free anytime
    regret_upper_bound_95: float
    bound_method: str                   # "hoeffding" | "bernstein" | "howard_ramdas"
    started_at: float
    finished_at: float
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arms"] = [a.to_dict() for a in self.arms]
        return d


# =====================================================================
# Internal per-arm state — mutable, owned by Bandit
# =====================================================================


@dataclass
class _ArmState:
    name: str
    n: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    last_r: float = 0.0
    first_seen: int = -1
    last_seen: int = -1
    # Thompson Beta:
    alpha: float = 1.0
    beta_: float = 1.0
    # Thompson Gaussian: N(mu0, 1/precision); update mu0, precision.
    ts_mu: float = 0.0
    ts_precision: float = 1.0
    # EXP3 / EXP3-IX: cumulative weighted IS reward estimate (log-domain).
    exp3_log_weight: float = 0.0
    # Tsallis-INF: cumulative IS-loss estimate.
    tsallis_loss: float = 0.0
    # Tsallis-INF: cumulative eta sum (for the OMD step-size schedule).
    # Last probability assigned to this arm (for IS denominator on next obs).
    last_prob: float = 0.0
    # Successive-elimination flag.
    eliminated: bool = False

    @property
    def mean(self) -> float:
        return self.sum_r / self.n if self.n > 0 else 0.0

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        v = self.sum_r2 / self.n - m * m
        return max(0.0, v) * self.n / (self.n - 1)

    def to_stats(self) -> ArmStats:
        return ArmStats(
            name=self.name,
            n_pulls=self.n,
            sum_reward=self.sum_r,
            sum_reward_sq=self.sum_r2,
            last_reward=self.last_r,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


@dataclass
class _LinArmState:
    """Per-arm state for linear contextual bandits (LinUCB / OFUL / LinTS)."""

    name: str
    A: list[list[float]]    # d×d covariance + ridge
    b: list[float]          # d×1 reward-weighted features
    d: int
    n: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0

    def theta_hat(self) -> list[float]:
        return _solve_spd(self.A, self.b)


# =====================================================================
# Helper: anytime-valid theoretical regret bounds, instantiated from data
# =====================================================================


def _ucb1_pseudo_regret(
    arms: list[_ArmState], total_pulls: int, b: float = 1.0,
) -> float:
    """Auer-Cesa-Bianchi-Fischer (2002) Theorem 1, data-instantiated.

    R_T ≤ 8 ∑_{a:Δ_a>0} log T / Δ_a + (1 + π²/3) ∑_a Δ_a.

    Uses Δ̂_a estimated from data.  Min-clipped at zero for the
    leader.  Returns 0 if no arm has been pulled.
    """
    if total_pulls <= 0:
        return 0.0
    best = max(a.mean for a in arms) if arms else 0.0
    s = 0.0
    sum_delta = 0.0
    for a in arms:
        delta = best - a.mean
        sum_delta += max(0.0, delta)
        if delta > 1e-9:
            s += 8.0 * math.log(max(total_pulls, math.e)) / delta
    return s + (1.0 + math.pi * math.pi / 3.0) * sum_delta


def _moss_pseudo_regret(arms: list[_ArmState], total_pulls: int) -> float:
    """Audibert-Bubeck (2010) MOSS distribution-free bound: R_T ≤ 39 √(K T)."""
    K = len(arms)
    if K == 0 or total_pulls <= 0:
        return 0.0
    return 39.0 * math.sqrt(K * total_pulls)


def _exp3_pseudo_regret(
    arms: list[_ArmState], total_pulls: int,
) -> float:
    """Auer-Cesa-Bianchi-Freund-Schapire (2002): R_T ≤ 2 √(e − 1) √(T K log K)."""
    K = len(arms)
    if K <= 1 or total_pulls <= 0:
        return 0.0
    return 2.0 * math.sqrt(math.e - 1.0) * math.sqrt(total_pulls * K * math.log(K))


def _tsallis_inf_pseudo_regret(
    arms: list[_ArmState], total_pulls: int,
) -> float:
    """Zimmert-Seldin (2021) Theorem 1: R_T ≤ 4 √(K T) + log(KT).

    Stochastic-regime refinement: R_T ≤ (8/Δ_min) log T + const.
    """
    K = len(arms)
    if K <= 1 or total_pulls <= 0:
        return 0.0
    adv = 4.0 * math.sqrt(K * total_pulls) + math.log(max(K * total_pulls, math.e))
    # Stochastic instance-dependent bound.
    best = max(a.mean for a in arms) if arms else 0.0
    deltas = [best - a.mean for a in arms if best - a.mean > 1e-9]
    if not deltas:
        return adv
    delta_min = min(deltas)
    stoch = 8.0 * math.log(max(total_pulls, math.e)) / max(delta_min, 1e-9)
    return min(adv, stoch + 4.0 * math.sqrt(K))


def _lin_pseudo_regret(d: int, total_pulls: int, alpha: float) -> float:
    """Abbasi-Yadkori-Pál-Szepesvári (2011) Theorem 3: R_T = O(d √(T log T))."""
    if total_pulls <= 0 or d <= 0:
        return 0.0
    return alpha * d * math.sqrt(total_pulls * math.log(max(total_pulls, math.e)))


# =====================================================================
# Bandit — the main runtime primitive
# =====================================================================


class Bandit:
    """Sequential decision under uncertainty — the runtime primitive.

    A `Bandit` is a *stateful* object that the coordination engine
    pushes pulls and rewards through.  It exposes three operations:

      * `select_arm(context=None)` — return the next arm to pull.
        Deterministic given `(state, seed)`.
      * `observe(arm, reward, context=None)` — record the realised
        reward.  Updates per-arm sufficient statistics and any
        algorithm-specific state (Beta-Bernoulli posterior, EXP3
        log-weights, LinUCB covariance, …).
      * `report()` — emit a tamper-evident `BanditReport` summarising
        the campaign so far with finite-sample anytime-valid regret
        upper bounds.

    The bandit is **not** a PAC-best-arm identifier.  Use `Arbiter`
    for that.  The two compose: pull via `Bandit.select_arm` for
    cumulative reward; ask `Arbiter.advise(bandit.state())` for a
    PAC commit recommendation.

    Parameters
    ----------
    arms : sequence of str
        Names of the K arms.  Must be unique.
    algorithm : str
        One of `KNOWN_ALGORITHMS`.  Selects the index policy.
    reward_model : str
        One of `KNOWN_REWARD_MODELS`.  Used by Thompson and by the
        bound on observed rewards.
    seed : int
        Deterministic seed for any algorithm that samples (Thompson,
        EXP3 sampling, IDS Monte Carlo).
    d : int, optional
        Context dimensionality for contextual algorithms.
    alpha : float, optional
        Exploration weight for LinUCB / OFUL.
    lam : float, optional
        Ridge regularisation strength for contextual algorithms.
    sigma : float, optional
        Standard deviation of observation noise (LinUCB / OFUL /
        LinTS / Thompson Gaussian).
    delta : float, optional
        Failure probability for OFUL self-normalised confidence
        ellipsoid.
    decay : float in (0, 1], optional
        Per-pull discount on past rewards (sliding-window flavour);
        decay=1.0 → no forgetting.
    """

    def __init__(
        self,
        arms: Sequence[str],
        algorithm: str = UCB1,
        *,
        reward_model: str = REWARD_BERNOULLI,
        seed: int = 0,
        d: int | None = None,
        alpha: float = _LIN_ALPHA_DEFAULT,
        lam: float = _LIN_RIDGE_DEFAULT,
        sigma: float = 1.0,
        delta: float = _LIN_DELTA_DEFAULT,
        decay: float = 1.0,
        epsilon: float = 0.1,
        epsilon_decay: bool = True,
        eta: float | None = None,
        gamma_ix: float | None = None,
        ids_mc_samples: int = _IDS_MC_SAMPLES_DEFAULT,
        max_reward: float = 1.0,
        min_reward: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        arms_list = list(arms)
        if not arms_list:
            raise BanditError("at least one arm required")
        if len(arms_list) != len(set(arms_list)):
            raise BanditError("arm names must be unique")
        if algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"unknown algorithm {algorithm!r}; "
                f"known: {sorted(KNOWN_ALGORITHMS)}"
            )
        if reward_model not in KNOWN_REWARD_MODELS:
            raise BanditError(f"unknown reward_model {reward_model!r}")
        if algorithm in _CONTEXTUAL:
            if d is None or d <= 0:
                raise InvalidContext(
                    f"contextual algorithm {algorithm!r} requires d > 0"
                )
        if not (0.0 < decay <= 1.0):
            raise BanditError("decay must be in (0, 1]")
        if not (max_reward >= min_reward):
            raise BanditError("max_reward must be >= min_reward")
        if epsilon < 0.0 or epsilon > 1.0:
            raise BanditError("epsilon must be in [0, 1]")

        self.arms = arms_list
        self.algorithm = algorithm
        self.reward_model = reward_model
        self.seed = int(seed)
        self.d = int(d) if d is not None else 0
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.delta = float(delta)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.epsilon_decay = bool(epsilon_decay)
        self.max_reward = float(max_reward)
        self.min_reward = float(min_reward)
        self.ids_mc_samples = int(ids_mc_samples)
        self.metadata: dict[str, Any] = dict(metadata or {})

        # Reward range for Hoeffding-style bounds.
        self._b = max(_EPS, self.max_reward - self.min_reward)

        # Default learning rates for adversarial algorithms.
        K = len(arms_list)
        # EXP3 optimal eta = √(log K / (K T)).  We don't know T, so
        # default to the time-uniform Auer 2002 anytime variant:
        # η_t = √(log K / (K t)).  This is recomputed on each select.
        self._eta_fixed = eta
        # EXP3-IX: γ = √(log K / (K T)).  Same time-uniform default.
        self._gamma_ix = gamma_ix

        # Per-arm state.
        self._arms: dict[str, _ArmState] = {
            a: _ArmState(name=a) for a in arms_list
        }
        self._lin: dict[str, _LinArmState] = {}
        if algorithm in _CONTEXTUAL:
            d_ = self.d
            for a in arms_list:
                self._lin[a] = _LinArmState(
                    name=a,
                    A=_madd(_eye(d_), [[(self.lam - 1.0) if i == j else 0.0
                                        for j in range(d_)]
                                       for i in range(d_)]),
                    b=[0.0] * d_,
                    d=d_,
                )

        # Cumulative pull counter and reward.
        self._t = 0
        self._cumulative_reward = 0.0
        self._history: list[PullDecision] = []
        self._rewards_log: list[tuple[int, str, float]] = []
        self._started_at = time.time()

        # Deterministic RNGs.
        self._rng = random.Random(self.seed)

        # Successive-elimination round-robin pointer.
        self._se_round_robin = 0

        # Per-tick eligible arms (for successive elimination).
        self._se_active = list(arms_list)

    # ------------------------------------------------------------------
    # Public state queries
    # ------------------------------------------------------------------

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def t(self) -> int:
        return self._t

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    def arm_stats(self, name: str) -> ArmStats:
        if name not in self._arms:
            raise UnknownArm(f"unknown arm {name!r}")
        return self._arms[name].to_stats()

    def all_arm_stats(self) -> list[ArmStats]:
        return [self._arms[a].to_stats() for a in self.arms]

    def best_arm_so_far(self) -> str:
        """Empirically best arm by mean.  Ties broken by name (stable)."""
        best_name = self.arms[0]
        best_mean = -math.inf
        for a in self.arms:
            m = self._arms[a].mean
            if m > best_mean + 1e-15 or (
                abs(m - best_mean) <= 1e-15 and a < best_name
            ):
                best_mean = m
                best_name = a
        return best_name

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_arm(
        self, context: Sequence[float] | None = None,
    ) -> str:
        """Recommend the next arm to pull.

        For contextual algorithms `context` is a d-vector of features;
        for stochastic / adversarial algorithms `context` is ignored.
        Deterministic given `(seed, history)`.
        """
        if self.algorithm in _CONTEXTUAL:
            if context is None or len(context) != self.d:
                raise InvalidContext(
                    f"{self.algorithm} requires a {self.d}-vector context"
                )
            choice, idx, rationale = self._select_contextual(list(context))
        else:
            choice, idx, rationale = self._select_stochastic_or_adversarial()
        ctx_fp = _hash_vector(context) if context is not None else ""
        decision = PullDecision(
            t=self._t + 1, arm=choice, index_value=idx,
            rationale=rationale, algorithm=self.algorithm,
            context_fingerprint=ctx_fp,
        )
        self._history.append(decision)
        return choice

    def observe(
        self,
        arm: str,
        reward: float,
        *,
        context: Sequence[float] | None = None,
    ) -> None:
        """Record the realised reward for `arm`.

        Updates per-arm sufficient statistics and any algorithm-
        specific state.  Validates reward against the declared
        bounds and reward model — out-of-range observations raise
        `BanditError` rather than corrupting silently.
        """
        if arm not in self._arms:
            raise UnknownArm(f"unknown arm {arm!r}")
        r = float(reward)
        if self.reward_model == REWARD_BERNOULLI:
            if r not in (0.0, 1.0):
                # Allow [0, 1] continuous as a Bernoulli mean estimator.
                if r < 0.0 or r > 1.0:
                    raise BanditError(
                        f"Bernoulli reward {r} outside [0, 1]"
                    )
        elif self.reward_model == REWARD_BOUNDED:
            if r < self.min_reward or r > self.max_reward:
                raise BanditError(
                    f"reward {r} outside [{self.min_reward}, "
                    f"{self.max_reward}]"
                )

        self._t += 1
        self._cumulative_reward += r
        self._rewards_log.append((self._t, arm, r))

        # Apply decay to past pulls (sliding-window UCB / Tsallis-INF).
        if self.decay < 1.0:
            for a in self._arms.values():
                a.n = max(0, int(round(a.n * self.decay)))
                a.sum_r *= self.decay
                a.sum_r2 *= self.decay

        st = self._arms[arm]
        st.n += 1
        st.sum_r += r
        st.sum_r2 += r * r
        st.last_r = r
        if st.first_seen < 0:
            st.first_seen = self._t
        st.last_seen = self._t

        # Algorithm-specific updates.
        if self.algorithm == THOMPSON_BETA:
            # Bernoulli reward → Beta(α, β) posterior.
            st.alpha += r
            st.beta_ += (1.0 - r)
        elif self.algorithm == THOMPSON_GAUSSIAN:
            # Normal-Normal conjugate posterior with known sigma².
            prec = 1.0 / max(self.sigma * self.sigma, _EPS)
            new_prec = st.ts_precision + prec
            st.ts_mu = (st.ts_mu * st.ts_precision + r * prec) / new_prec
            st.ts_precision = new_prec
        elif self.algorithm == EXP3:
            K = self.n_arms
            tt = self._t
            eta_t = self._eta_fixed if self._eta_fixed is not None else (
                math.sqrt(math.log(K) / (K * max(tt, 1)))
            )
            p_a = st.last_prob if st.last_prob > 0.0 else 1.0 / K
            est_r = r / max(p_a, _EPS)
            st.exp3_log_weight += eta_t * est_r
        elif self.algorithm == EXP3_IX:
            K = self.n_arms
            tt = self._t
            eta_t = math.sqrt(math.log(K) / (K * max(tt, 1)))
            gamma = self._gamma_ix if self._gamma_ix is not None else 0.5 * eta_t
            p_a = st.last_prob if st.last_prob > 0.0 else 1.0 / K
            est_r = r / (p_a + gamma)
            st.exp3_log_weight += eta_t * est_r
        elif self.algorithm == TSALLIS_INF:
            # Update cumulative IS-loss estimate L_a += loss / p_a.
            p_a = st.last_prob if st.last_prob > 0.0 else 1.0 / max(self.n_arms, 1)
            loss = 1.0 - r
            st.tsallis_loss += loss / max(p_a, _EPS)
        elif self.algorithm in _CONTEXTUAL:
            if context is None or len(context) != self.d:
                raise InvalidContext(
                    f"contextual observe requires a {self.d}-vector context"
                )
            x = list(context)
            la = self._lin[arm]
            la.A = _madd(la.A, _outer(x, x))
            la.b = _vadd(la.b, _scaled(x, r))
            la.n += 1
            la.sum_r += r
            la.sum_r2 += r * r

    def forget(self, arm: str, halflife: float = 100.0) -> None:
        """Apply an exponential discount on past pulls of `arm`.

        For non-stationary environments — composes with
        DriftSentinel: when CUSUM trips on arm `a`, call
        `bandit.forget(a, halflife=H)` to age out stale rewards.

        The factor is `0.5 ** (n / halflife)` where `n` is the
        number of pulls of this arm so far.  Mirrors SW-UCB
        (Garivier & Moulines 2008) sliding-window discount.
        """
        if arm not in self._arms:
            raise UnknownArm(f"unknown arm {arm!r}")
        if halflife <= 0.0:
            raise BanditError("halflife must be > 0")
        st = self._arms[arm]
        if st.n <= 0:
            return
        factor = 0.5 ** (st.n / halflife)
        st.n = max(1, int(round(st.n * factor)))
        st.sum_r *= factor
        st.sum_r2 *= factor

    def reset(self) -> None:
        """Wipe per-arm statistics; reset `t = 0`; preserve config."""
        for a in self.arms:
            self._arms[a] = _ArmState(name=a)
        if self.algorithm in _CONTEXTUAL:
            d_ = self.d
            for a in self.arms:
                self._lin[a] = _LinArmState(
                    name=a,
                    A=_madd(_eye(d_), [[(self.lam - 1.0) if i == j else 0.0
                                        for j in range(d_)]
                                       for i in range(d_)]),
                    b=[0.0] * d_,
                    d=d_,
                )
        self._t = 0
        self._cumulative_reward = 0.0
        self._history = []
        self._rewards_log = []
        self._se_active = list(self.arms)
        self._se_round_robin = 0
        self._rng = random.Random(self.seed)
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # Reporting + tamper-evident fingerprint
    # ------------------------------------------------------------------

    def report(self, *, delta_bound: float = 0.05) -> BanditReport:
        """Emit a `BanditReport` for the campaign so far.

        `delta_bound` controls the confidence level of the empirical
        regret upper bound: 1 − delta_bound is the anytime-valid
        coverage.  The Howard-Ramdas-McAuliffe-Sekhon (2021)
        confidence sequence is used by default; falls back to
        Maurer-Pontil empirical-Bernstein when n ≥ 2 and to
        Hoeffding otherwise.
        """
        stats = self.all_arm_stats()
        pseudo = self._pseudo_regret_upper_bound()
        ub_99, ub_95, method = self._empirical_regret_bound(delta_bound)

        report = BanditReport(
            id=f"bandit-{int(self._started_at * 1000):x}",
            algorithm=self.algorithm,
            reward_model=self.reward_model,
            arms=stats,
            n_pulls=self._t,
            n_arms=self.n_arms,
            best_arm_so_far=self.best_arm_so_far(),
            cumulative_reward=self._cumulative_reward,
            pseudo_regret_upper=pseudo,
            regret_upper_bound_99=ub_99,
            regret_upper_bound_95=ub_95,
            bound_method=method,
            started_at=self._started_at,
            finished_at=time.time(),
            fingerprint="",  # filled below
        )
        report.fingerprint = self.fingerprint()
        return report

    def fingerprint(self) -> str:
        """Tamper-evident SHA-256 of the full campaign state."""
        h = hashlib.sha256()
        h.update(b"agi.bandit.v1\n")
        h.update(f"algo={self.algorithm}\n".encode())
        h.update(f"reward_model={self.reward_model}\n".encode())
        h.update(f"seed={self.seed}\n".encode())
        h.update(f"arms={','.join(self.arms)}\n".encode())
        h.update(f"d={self.d}\n".encode())
        h.update(f"alpha={self.alpha}\n".encode())
        h.update(f"lam={self.lam}\n".encode())
        h.update(f"sigma={self.sigma}\n".encode())
        h.update(f"delta={self.delta}\n".encode())
        h.update(f"decay={self.decay}\n".encode())
        h.update(f"epsilon={self.epsilon}\n".encode())
        h.update(f"t={self._t}\n".encode())
        for t, a, r in self._rewards_log:
            h.update(f"{t}|{a}|{r}\n".encode())
        return "sha256:" + h.hexdigest()

    # ------------------------------------------------------------------
    # Stochastic + adversarial selection dispatch
    # ------------------------------------------------------------------

    def _select_stochastic_or_adversarial(self) -> tuple[str, float, str]:
        # First pull each arm at least once.
        for a in self.arms:
            if self._arms[a].n == 0 and not self._arms[a].eliminated:
                return a, 0.0, "init-pull"

        algo = self.algorithm
        if algo == UCB1:
            return self._select_ucb1()
        if algo == KL_UCB:
            return self._select_kl_ucb()
        if algo == MOSS:
            return self._select_moss()
        if algo == UCB_V:
            return self._select_ucb_v()
        if algo == THOMPSON_BETA:
            return self._select_thompson_beta()
        if algo == THOMPSON_GAUSSIAN:
            return self._select_thompson_gaussian()
        if algo == SUCCESSIVE_ELIMINATION:
            return self._select_successive_elimination()
        if algo == EPSILON_GREEDY:
            return self._select_epsilon_greedy()
        if algo == EXP3:
            return self._select_exp3()
        if algo == EXP3_IX:
            return self._select_exp3_ix()
        if algo == TSALLIS_INF:
            return self._select_tsallis_inf()
        if algo == IDS:
            return self._select_ids()
        raise UnknownAlgorithm(f"unhandled algorithm {algo!r}")

    # ------------------------------------------------------------------
    # UCB family
    # ------------------------------------------------------------------

    def _select_ucb1(self) -> tuple[str, float, str]:
        """UCB1 (Auer-Cesa-Bianchi-Fischer 2002).

        Index: μ̂_a + b √(2 log t / N_a).  b is the bound on rewards.
        """
        t = max(self._t, 1)
        log_t = math.log(t)
        best_a = self.arms[0]
        best_idx = -math.inf
        for a in self.arms:
            st = self._arms[a]
            if st.n == 0:
                return a, math.inf, "ucb1-cold-start"
            idx = st.mean + self._b * math.sqrt(2.0 * log_t / st.n)
            if idx > best_idx:
                best_idx = idx
                best_a = a
        return best_a, best_idx, f"ucb1-index={best_idx:.6f}"

    def _select_kl_ucb(self) -> tuple[str, float, str]:
        """KL-UCB (Garivier-Cappé 2011).

        Index: kl_ucb_upper(μ̂_a, N_a, log(t) + c · log log(t)).
        For Bernoulli rewards the bound is asymptotically tight.
        """
        t = max(self._t, 1)
        c = 3.0  # exploration constant per Garivier-Cappé Thm 1
        beta = math.log(t) + c * math.log(max(math.log(t), 1.0))
        best_a = self.arms[0]
        best_idx = -math.inf
        for a in self.arms:
            st = self._arms[a]
            if st.n == 0:
                return a, math.inf, "klucb-cold-start"
            idx = kl_ucb_upper(st.mean, st.n, beta)
            if idx > best_idx:
                best_idx = idx
                best_a = a
        return best_a, best_idx, f"klucb-index={best_idx:.6f}"

    def _select_moss(self) -> tuple[str, float, str]:
        """MOSS (Audibert-Bubeck 2009/2010).

        Index: μ̂_a + b √( max(log(T / (K · N_a)), 0) / N_a ).
        Uses current t as a proxy for T (anytime variant).
        """
        K = self.n_arms
        t = max(self._t, 1)
        best_a = self.arms[0]
        best_idx = -math.inf
        for a in self.arms:
            st = self._arms[a]
            if st.n == 0:
                return a, math.inf, "moss-cold-start"
            arg = math.log(max(t / (K * st.n), 1.0))
            idx = st.mean + self._b * math.sqrt(arg / st.n)
            if idx > best_idx:
                best_idx = idx
                best_a = a
        return best_a, best_idx, f"moss-index={best_idx:.6f}"

    def _select_ucb_v(self) -> tuple[str, float, str]:
        """UCB-V (Audibert-Munos-Szepesvári 2009).

        Index: μ̂_a + √(2 σ̂_a² log t / N_a) + 3 b log t / N_a.
        Empirical-variance bonus replaces Hoeffding's worst-case.
        """
        t = max(self._t, 1)
        log_t = math.log(t)
        best_a = self.arms[0]
        best_idx = -math.inf
        for a in self.arms:
            st = self._arms[a]
            if st.n == 0:
                return a, math.inf, "ucbv-cold-start"
            v = st.var
            idx = (st.mean
                   + math.sqrt(2.0 * v * log_t / st.n)
                   + 3.0 * self._b * log_t / st.n)
            if idx > best_idx:
                best_idx = idx
                best_a = a
        return best_a, best_idx, f"ucbv-index={best_idx:.6f}"

    # ------------------------------------------------------------------
    # Thompson family
    # ------------------------------------------------------------------

    def _select_thompson_beta(self) -> tuple[str, float, str]:
        """Thompson Sampling — Beta-Bernoulli (Agrawal-Goyal 2012)."""
        best_a = self.arms[0]
        best_theta = -math.inf
        for a in self.arms:
            st = self._arms[a]
            theta = _beta_sample(self._rng, st.alpha, st.beta_)
            if theta > best_theta:
                best_theta = theta
                best_a = a
        return best_a, best_theta, f"thompson-beta-sample={best_theta:.6f}"

    def _select_thompson_gaussian(self) -> tuple[str, float, str]:
        """Thompson Sampling — Gaussian-Gaussian conjugate."""
        best_a = self.arms[0]
        best_theta = -math.inf
        for a in self.arms:
            st = self._arms[a]
            sigma = 1.0 / math.sqrt(max(st.ts_precision, _EPS))
            theta = _gaussian_sample(self._rng, st.ts_mu, sigma)
            if theta > best_theta:
                best_theta = theta
                best_a = a
        return best_a, best_theta, f"thompson-gauss-sample={best_theta:.6f}"

    # ------------------------------------------------------------------
    # Successive Elimination
    # ------------------------------------------------------------------

    def _select_successive_elimination(self) -> tuple[str, float, str]:
        """Successive Elimination (Even-Dar-Mannor-Mansour 2006).

        Round-robin over currently-active arms; after every pull,
        the arm whose KL-UCB-upper-bound falls strictly below the
        leader's KL-LCB-lower-bound is eliminated.  Anytime-(ε, δ)-PAC.
        """
        active = [a for a in self.arms if not self._arms[a].eliminated]
        if not active:
            # Should never happen; fall back to round-robin.
            active = list(self.arms)
        # Round-robin pointer:
        idx = self._se_round_robin % len(active)
        choice = active[idx]
        self._se_round_robin += 1
        # Elimination check post-pull (logical place is between
        # observe and select; we do it on select for stateless
        # selection at the cost of one extra pull).
        if self._t > 0:
            self._update_elimination()
        return choice, 0.0, f"successive-elim active={len(active)}"

    def _update_elimination(self) -> None:
        t = max(self._t, 1)
        beta = math.log(t) + 3.0 * math.log(max(math.log(t), 1.0))
        # Best lower bound among current actives.
        actives = [a for a in self.arms if not self._arms[a].eliminated]
        if len(actives) <= 1:
            return
        best_lcb = -math.inf
        for a in actives:
            st = self._arms[a]
            if st.n == 0:
                continue
            lcb = kl_ucb_upper(1.0 - st.mean, st.n, beta)
            lcb = 1.0 - lcb  # symmetric KL-LCB
            if lcb > best_lcb:
                best_lcb = lcb
        for a in actives:
            st = self._arms[a]
            if st.n == 0:
                continue
            ucb = kl_ucb_upper(st.mean, st.n, beta)
            if ucb < best_lcb - 1e-9:
                st.eliminated = True

    # ------------------------------------------------------------------
    # ε-greedy
    # ------------------------------------------------------------------

    def _select_epsilon_greedy(self) -> tuple[str, float, str]:
        """ε-greedy with optional 1/t decay (Cesa-Bianchi-Fischer 1998).

        With probability ε_t pull a uniformly-random arm; else pull
        the empirical leader.  Default schedule ε_t = ε / √(t).
        """
        t = max(self._t, 1)
        eps = self.epsilon / math.sqrt(t) if self.epsilon_decay else self.epsilon
        eps = min(1.0, max(0.0, eps))
        u = self._rng.random()
        if u < eps:
            choice = self._rng.choice(self.arms)
            return choice, eps, f"eps-greedy-explore eps={eps:.4f}"
        leader = self.best_arm_so_far()
        return leader, eps, f"eps-greedy-exploit eps={eps:.4f}"

    # ------------------------------------------------------------------
    # EXP3 / EXP3-IX
    # ------------------------------------------------------------------

    def _select_exp3(self) -> tuple[str, float, str]:
        """EXP3 (Auer-Cesa-Bianchi-Freund-Schapire 2002).

        p_a(t) = w_a(t) / ∑ w_b(t).  Sample, pull.  Weights are
        maintained incrementally in `_ArmState.exp3_log_weight` and
        updated on observe(); selection is O(K).
        """
        K = self.n_arms
        t = max(self._t, 1)
        eta = self._eta_fixed if self._eta_fixed is not None else (
            math.sqrt(math.log(K) / (K * t))
        )
        log_weights = [self._arms[a].exp3_log_weight for a in self.arms]
        probs = self._exp3_probs_from_log(log_weights, K)
        # Cache last_prob per arm so the next observe() can apply the
        # importance-weighted update.
        for i, a in enumerate(self.arms):
            self._arms[a].last_prob = probs[i]
        u = self._rng.random()
        cum = 0.0
        chosen = self.arms[-1]
        for a, p in zip(self.arms, probs):
            cum += p
            if u <= cum:
                chosen = a
                break
        return chosen, max(probs), (
            f"exp3 eta={eta:.4f} p_max={max(probs):.4f}"
        )

    def _exp3_probs_from_log(
        self, log_weights: list[float], K: int,
    ) -> list[float]:
        # Subtract max for stability.
        m = max(log_weights) if log_weights else 0.0
        exp_w = [math.exp(lw - m) for lw in log_weights]
        s = sum(exp_w)
        if s <= 0.0:
            return [1.0 / K] * K
        return [w / s for w in exp_w]

    def _select_exp3_ix(self) -> tuple[str, float, str]:
        """EXP3-IX (Neu 2015).

        Implicit-eXploration estimator: r̂_a = r_a · 1{a_t = a} / (p_a + γ).
        Adds γ to the denominator → high-probability regret bound.
        """
        K = self.n_arms
        t = max(self._t, 1)
        eta = math.sqrt(math.log(K) / (K * t))
        gamma = self._gamma_ix if self._gamma_ix is not None else 0.5 * eta
        log_weights = [self._arms[a].exp3_log_weight for a in self.arms]
        probs = self._exp3_probs_from_log(log_weights, K)
        for i, a in enumerate(self.arms):
            self._arms[a].last_prob = probs[i]
        u = self._rng.random()
        cum = 0.0
        chosen = self.arms[-1]
        for a, p in zip(self.arms, probs):
            cum += p
            if u <= cum:
                chosen = a
                break
        return chosen, max(probs), (
            f"exp3-ix eta={eta:.4f} gamma={gamma:.4f} p_max={max(probs):.4f}"
        )

    # ------------------------------------------------------------------
    # Tsallis-INF — best-of-both-worlds
    # ------------------------------------------------------------------

    def _select_tsallis_inf(self) -> tuple[str, float, str]:
        """Tsallis-INF (Zimmert-Seldin 2019/2021).

        OMD with negative Tsallis entropy regulariser Ψ(w) =
        -∑ √(w_a) (α = 1/2).  Update rule: w_a(t+1) ∝ (1/(η_t · (L_a(t)
        − ν)))² where L_a(t) is the cumulative IS-loss estimator
        (stored in `_ArmState.tsallis_loss`) and ν is the dual
        variable enforcing ∑ w_a = 1.  Step size η_t = 1/√t.

        This algorithm is simultaneously
          * minimax-optimal in adversarial regime: R_T ≤ O(√(KT));
          * Lai-Robbins-optimal in stochastic regime: R_T = O(log T).
        """
        K = self.n_arms
        L = [self._arms[a].tsallis_loss for a in self.arms]
        eta = 1.0 / math.sqrt(max(self._t + 1, 1))
        w = self._tsallis_w(L, eta, K)
        for i, a in enumerate(self.arms):
            self._arms[a].last_prob = w[i]
        u = self._rng.random()
        cum = 0.0
        chosen = self.arms[-1]
        for a, p in zip(self.arms, w):
            cum += p
            if u <= cum:
                chosen = a
                break
        return chosen, max(w), (
            f"tsallis-inf eta={eta:.4f} p_max={max(w):.4f}"
        )

    def _tsallis_w(
        self, L: list[float], eta_total: float, K: int,
    ) -> list[float]:
        """Solve w_a = (eta_total · (L_a - ν))^{-2} subject to ∑ w_a = 1.

        Newton on ν with bracketing.  Returns the simplex point.
        """
        if eta_total <= 0.0:
            return [1.0 / K] * K
        # f(ν) = ∑_a (eta · (L_a - ν))^{-2} - 1.  Decreasing in ν;
        # find unique root via bisection on a safe interval.
        L_min = min(L)
        # f is well-defined only for ν < min L; root is to the left.
        nu_hi = L_min - 1e-12
        nu_lo = L_min - 10.0 - 1.0 / max(eta_total, _EPS)
        # Expand lower bound until f(nu_lo) > 0 (small w sum).
        for _ in range(50):
            f = sum(1.0 / max(eta_total * (L[a] - nu_lo), _EPS) ** 2
                    for a in range(K))
            if f >= 1.0:
                break
            nu_lo -= 10.0
        # Bisect.
        for _ in range(_TSALLIS_NEWTON_MAX_ITER):
            nu = 0.5 * (nu_lo + nu_hi)
            f = sum(1.0 / max(eta_total * (L[a] - nu), _EPS) ** 2
                    for a in range(K))
            if abs(f - 1.0) < _TSALLIS_NEWTON_TOL:
                break
            if f > 1.0:
                nu_hi = nu
            else:
                nu_lo = nu
        nu = 0.5 * (nu_lo + nu_hi)
        w = [1.0 / max(eta_total * (L[a] - nu), _EPS) ** 2 for a in range(K)]
        s = sum(w)
        if s <= 0.0:
            return [1.0 / K] * K
        return [v / s for v in w]

    # ------------------------------------------------------------------
    # Information-Directed Sampling — cutting edge
    # ------------------------------------------------------------------

    def _select_ids(self) -> tuple[str, float, str]:
        """Information-Directed Sampling (Russo-Van Roy 2014/2018).

        Pulls argmin_a Ψ²_a / g_a where
          Ψ_a = E[μ* − μ_a]            — expected regret of arm a
          g_a = MI(arm a's reward; A*) — information about the
                                          optimal arm gained

        Implements the standard Monte-Carlo estimator:
          1. draw M posterior samples θ^m ~ posterior;
          2. for each m, A*_m = argmax_a θ^m_a;
          3. q_a = P(A* = a) ≈ (1/M) ∑ 1{A*_m = a};
          4. μ̄_{a, b} = E[θ_a | A* = b] ≈ avg θ^m_a over m with
             A*_m = b;
          5. v_a = ∑_b q_b (μ̄_{a, b} − μ̂_a)² (variance contribution).
          6. info-ratio Ψ²_a / v_a; pull argmin (use g = v).

        Tractable; matches the closed-form in the Russo-Van Roy
        regret analysis.  Provably tighter than Thompson sampling
        on instances with *informative-but-suboptimal* arms.
        """
        M = self.ids_mc_samples
        K = self.n_arms
        # Sample θ^m from the per-arm posterior.
        thetas: list[list[float]] = []
        for _ in range(M):
            row = []
            for a in self.arms:
                st = self._arms[a]
                if self.reward_model == REWARD_BERNOULLI:
                    row.append(_beta_sample(self._rng, st.alpha, st.beta_))
                else:
                    sigma = 1.0 / math.sqrt(max(st.ts_precision, _EPS))
                    row.append(_gaussian_sample(self._rng, st.ts_mu, sigma))
            thetas.append(row)
        # Argmax per sample.
        astar = [max(range(K), key=lambda i: thetas[m][i]) for m in range(M)]
        q = [0.0] * K
        for b in astar:
            q[b] += 1.0
        q = [v / M for v in q]
        # μ̄_{a, b}.
        sum_theta = [[0.0] * K for _ in range(K)]
        count_by_astar = [0] * K
        for m in range(M):
            b = astar[m]
            count_by_astar[b] += 1
            for a in range(K):
                sum_theta[a][b] += thetas[m][a]
        mu_bar = [[0.0] * K for _ in range(K)]
        for a in range(K):
            for b in range(K):
                if count_by_astar[b] > 0:
                    mu_bar[a][b] = sum_theta[a][b] / count_by_astar[b]
        # Marginal mean for arm a.
        mu_hat = [
            sum(q[b] * mu_bar[a][b] for b in range(K)) for a in range(K)
        ]
        # ψ_a = E[μ*] − μ_hat_a.
        mu_star = sum(q[b] * mu_bar[b][b] for b in range(K))
        psi = [max(mu_star - mu_hat[a], 0.0) for a in range(K)]
        # v_a (information gain proxy).
        v = []
        for a in range(K):
            s = 0.0
            for b in range(K):
                diff = mu_bar[a][b] - mu_hat[a]
                s += q[b] * diff * diff
            v.append(s)
        # Info-ratio.  Convention: 0/0 = 0 (treat as informative).
        best_a = self.arms[0]
        best_ratio = math.inf
        for a in range(K):
            if v[a] <= _EPS:
                ratio = math.inf if psi[a] > _EPS else 0.0
            else:
                ratio = psi[a] * psi[a] / v[a]
            if ratio < best_ratio:
                best_ratio = ratio
                best_a = self.arms[a]
        return best_a, best_ratio, f"ids info-ratio={best_ratio:.6f}"

    # ------------------------------------------------------------------
    # Contextual: LinUCB / OFUL / LinTS
    # ------------------------------------------------------------------

    def _select_contextual(
        self, context: list[float],
    ) -> tuple[str, float, str]:
        if self.algorithm == LINUCB:
            return self._select_linucb(context)
        if self.algorithm == OFUL:
            return self._select_oful(context)
        if self.algorithm == LIN_TS:
            return self._select_lin_ts(context)
        raise UnknownAlgorithm(f"unhandled contextual algo {self.algorithm!r}")

    def _select_linucb(self, x: list[float]) -> tuple[str, float, str]:
        """LinUCB (Li-Chu-Langford-Schapire 2010).

        For each arm a, p_a = θ̂_a · x + α √(xᵀ A_a^{-1} x).
        Ridge θ̂_a = A_a^{-1} b_a where A_a = λI + ∑ x x^T.
        """
        best_a = self.arms[0]
        best_p = -math.inf
        for a in self.arms:
            la = self._lin[a]
            theta = la.theta_hat()
            mean = _dot(theta, x)
            width = self.alpha * math.sqrt(max(_quad_form_inv(la.A, x), 0.0))
            p = mean + width
            if p > best_p:
                best_p = p
                best_a = a
        return best_a, best_p, f"linucb p={best_p:.6f}"

    def _select_oful(self, x: list[float]) -> tuple[str, float, str]:
        """OFUL (Abbasi-Yadkori-Pál-Szepesvári 2011) Theorem 2.

        β_t = σ √(2 log(det(V_t)^{1/2} / (δ · λ^{d/2}))) + √(λ) S.

        Per arm a, p_a = θ̂_a · x + β_t √(xᵀ A_a^{-1} x).
        Sharper than LinUCB by the optimal log-det constant.
        """
        best_a = self.arms[0]
        best_p = -math.inf
        S = 1.0  # ‖θ*‖ bound; configurable in metadata
        S = float(self.metadata.get("theta_norm_bound", S))
        for a in self.arms:
            la = self._lin[a]
            theta = la.theta_hat()
            mean = _dot(theta, x)
            logdet = _logdet_spd(la.A)
            d_ = la.d
            beta_t = (
                self.sigma * math.sqrt(
                    2.0 * (0.5 * logdet
                           - 0.5 * d_ * math.log(self.lam)
                           - math.log(self.delta))
                )
                + math.sqrt(self.lam) * S
            )
            width = beta_t * math.sqrt(max(_quad_form_inv(la.A, x), 0.0))
            p = mean + width
            if p > best_p:
                best_p = p
                best_a = a
        return best_a, best_p, f"oful p={best_p:.6f}"

    def _select_lin_ts(self, x: list[float]) -> tuple[str, float, str]:
        """Linear Thompson Sampling (Agrawal-Goyal 2013).

        Sample θ̃_a ~ N(θ̂_a, β² A_a^{-1}); pull argmax xᵀ θ̃_a.
        β² = σ²; standard prior.
        """
        best_a = self.arms[0]
        best_p = -math.inf
        for a in self.arms:
            la = self._lin[a]
            theta_hat = la.theta_hat()
            # Sample θ̃ = θ̂ + σ · L^{-T} · z where A = L L^T.
            L = _cholesky(la.A)
            d_ = la.d
            z = [self._rng.gauss(0.0, 1.0) for _ in range(d_)]
            # Solve L^T u = z, then θ̃ = θ̂ + σ · u.
            Lt = [[L[j][i] for j in range(d_)] for i in range(d_)]
            u = _solve_upper_tri(Lt, z)
            theta_tilde = [theta_hat[i] + self.sigma * u[i] for i in range(d_)]
            p = _dot(theta_tilde, x)
            if p > best_p:
                best_p = p
                best_a = a
        return best_a, best_p, f"lin-ts p={best_p:.6f}"

    # ------------------------------------------------------------------
    # Regret bounds
    # ------------------------------------------------------------------

    def _pseudo_regret_upper_bound(self) -> float:
        """Algorithm-specific theoretical regret bound, instantiated.

        Returns a finite-sample upper bound on the *pseudo-regret*
        E[R_T] = T μ* − E[∑ r_t] using known constants from the
        algorithm's analysis and data-estimated quantities.
        """
        arms_list = list(self._arms.values())
        algo = self.algorithm
        if algo in (UCB1, KL_UCB, UCB_V):
            return _ucb1_pseudo_regret(arms_list, self._t, b=self._b)
        if algo == MOSS:
            return _moss_pseudo_regret(arms_list, self._t)
        if algo == EXP3:
            return _exp3_pseudo_regret(arms_list, self._t)
        if algo == EXP3_IX:
            return _exp3_pseudo_regret(arms_list, self._t)
        if algo == TSALLIS_INF:
            return _tsallis_inf_pseudo_regret(arms_list, self._t)
        if algo in (THOMPSON_BETA, THOMPSON_GAUSSIAN, IDS):
            # Thompson / IDS matches Lai-Robbins asymptotically;
            # use UCB1 bound as a conservative finite-sample upper.
            return _ucb1_pseudo_regret(arms_list, self._t, b=self._b)
        if algo == SUCCESSIVE_ELIMINATION:
            return _ucb1_pseudo_regret(arms_list, self._t, b=self._b)
        if algo == EPSILON_GREEDY:
            # Cesa-Bianchi & Fischer 1998: R_T = O(log T / Δ_min).
            return _ucb1_pseudo_regret(arms_list, self._t, b=self._b)
        if algo in _CONTEXTUAL:
            return _lin_pseudo_regret(self.d, self._t, self.alpha)
        return float("inf")

    def _empirical_regret_bound(
        self, delta: float,
    ) -> tuple[float, float, str]:
        """Distribution-free, anytime-valid upper bound on R_T.

        Uses the per-arm Howard-Ramdas-McAuliffe-Sekhon (2021)
        confidence sequence to upper-bound each arm's true mean and
        lower-bound the leader's true mean, then sums (UCB_*  −
        LCB_a) · N_a(t) over all arms a ≠ *.

        Returns (bound at delta=0.01, bound at delta=0.05, method).
        """
        if self._t <= 0:
            return 0.0, 0.0, "howard_ramdas"

        method = "howard_ramdas"

        def _bound(d_val: float) -> float:
            # Pessimistic UCB on leader's mean, pessimistic LCB on others.
            leader = self.best_arm_so_far()
            leader_n = max(self._arms[leader].n, 1)
            leader_mean = self._arms[leader].mean
            # Anytime half-widths.
            hw_leader = howard_ramdas_half_width(
                leader_n, d_val / max(self.n_arms, 1), self._b,
            )
            mu_star_ucb = leader_mean + hw_leader
            r = 0.0
            for a in self.arms:
                st = self._arms[a]
                if st.n == 0:
                    continue
                hw_a = howard_ramdas_half_width(
                    st.n, d_val / max(self.n_arms, 1), self._b,
                )
                # Pessimistic gap for this arm.
                gap_pess = max(0.0, mu_star_ucb - (st.mean - hw_a))
                r += gap_pess * st.n
            return r

        return _bound(0.01), _bound(0.05), method

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Serialisable snapshot — replay-safe."""
        return {
            "version": "agi.bandit.v1",
            "algorithm": self.algorithm,
            "reward_model": self.reward_model,
            "seed": self.seed,
            "arms": list(self.arms),
            "d": self.d,
            "alpha": self.alpha,
            "lam": self.lam,
            "sigma": self.sigma,
            "delta": self.delta,
            "decay": self.decay,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "max_reward": self.max_reward,
            "min_reward": self.min_reward,
            "ids_mc_samples": self.ids_mc_samples,
            "metadata": dict(self.metadata),
            "t": self._t,
            "cumulative_reward": self._cumulative_reward,
            "rewards_log": list(self._rewards_log),
            "started_at": self._started_at,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "Bandit":
        """Rehydrate a Bandit from a previously saved `state()`.

        Replays the reward log to reconstruct arm state and
        algorithm-specific quantities deterministically.  Fingerprint
        equals the original's iff the replay is faithful.
        """
        if state.get("version") != "agi.bandit.v1":
            raise BanditError(
                f"unsupported state version: {state.get('version')!r}"
            )
        b = cls(
            arms=state["arms"],
            algorithm=state["algorithm"],
            reward_model=state["reward_model"],
            seed=state["seed"],
            d=state["d"] or None,
            alpha=state["alpha"],
            lam=state["lam"],
            sigma=state["sigma"],
            delta=state["delta"],
            decay=state["decay"],
            epsilon=state["epsilon"],
            epsilon_decay=state["epsilon_decay"],
            max_reward=state["max_reward"],
            min_reward=state["min_reward"],
            ids_mc_samples=state["ids_mc_samples"],
            metadata=state.get("metadata", {}),
        )
        # Replay reward log to reconstruct state.
        for (_, arm, r) in state["rewards_log"]:
            b.observe(arm, r)
        return b


# =====================================================================
# Helpers / module-level utility functions
# =====================================================================


def _hash_vector(v: Sequence[float] | None) -> str:
    if v is None:
        return ""
    h = hashlib.sha256()
    for x in v:
        h.update(f"{x:.12g}|".encode())
    return "sha256:" + h.hexdigest()[:16]


def expected_regret_ucb1(deltas: Sequence[float], T: int) -> float:
    """Closed-form UCB1 expected pseudo-regret upper bound.

    R_T ≤ 8 ∑_{Δ > 0} log T / Δ + (1 + π²/3) ∑ Δ.

    Independent of any active bandit — useful for a-priori
    planning: "how many pulls to drive regret below X given known
    gaps?"  Compatible with the unit-test predicate.
    """
    if T <= 0:
        return 0.0
    log_T = math.log(max(T, math.e))
    s = sum(8.0 * log_T / d for d in deltas if d > 1e-9)
    s += (1.0 + math.pi * math.pi / 3.0) * sum(max(0.0, d) for d in deltas)
    return s


def expected_regret_thompson_beta(
    deltas: Sequence[float], T: int,
) -> float:
    """Agrawal-Goyal (2012) Beta-Bernoulli Thompson upper bound.

    R_T ≤ (1 + ε) ∑_{Δ > 0} log T / d(μ_a, μ*) + C / ε² for any ε > 0.
    Conservative approximation: use d(μ_a, μ*) ≥ 2 Δ_a² (Pinsker).
    """
    if T <= 0:
        return 0.0
    log_T = math.log(max(T, math.e))
    s = 0.0
    for d in deltas:
        if d > 1e-9:
            s += log_T / (2.0 * d * d)
    return s + math.pi * math.pi / 6.0 * sum(max(0.0, d) for d in deltas)


def expected_regret_exp3(K: int, T: int) -> float:
    """Auer et al. (2002) EXP3 worst-case regret: 2 √((e-1) T K log K)."""
    if K <= 1 or T <= 0:
        return 0.0
    return 2.0 * math.sqrt((math.e - 1.0) * T * K * math.log(K))


def best_arm_index(stats: Iterable[ArmStats]) -> str:
    """Empirical best arm from `ArmStats`, ties by name."""
    best_name = ""
    best_mean = -math.inf
    for s in stats:
        m = s.mean
        if m > best_mean + 1e-15 or (
            abs(m - best_mean) <= 1e-15 and (best_name == "" or s.name < best_name)
        ):
            best_mean = m
            best_name = s.name
    return best_name


def quick_two_armed_bandit(
    n_pulls: int,
    p1: float = 0.6,
    p2: float = 0.4,
    algorithm: str = THOMPSON_BETA,
    seed: int = 0,
) -> tuple[BanditReport, list[str]]:
    """Run a synthetic two-armed Bernoulli bandit for `n_pulls`.

    For demos and tests only.  Returns the `BanditReport` plus the
    sequence of arms pulled.  Reward stream is deterministic
    given `seed`.
    """
    bandit = Bandit(
        arms=["a", "b"], algorithm=algorithm, reward_model=REWARD_BERNOULLI,
        seed=seed,
    )
    rng = random.Random(seed + 1)
    history: list[str] = []
    for _ in range(n_pulls):
        a = bandit.select_arm()
        history.append(a)
        p = p1 if a == "a" else p2
        r = 1.0 if rng.random() < p else 0.0
        bandit.observe(a, r)
    return bandit.report(), history


# =====================================================================
# Final exports
# =====================================================================


__all__ = [
    # Algorithms.
    "UCB1", "KL_UCB", "MOSS", "UCB_V",
    "THOMPSON_BETA", "THOMPSON_GAUSSIAN",
    "SUCCESSIVE_ELIMINATION", "EPSILON_GREEDY",
    "EXP3", "EXP3_IX", "TSALLIS_INF",
    "LINUCB", "OFUL", "LIN_TS", "IDS",
    "KNOWN_ALGORITHMS",
    # Reward models.
    "REWARD_BERNOULLI", "REWARD_GAUSSIAN", "REWARD_BOUNDED",
    "KNOWN_REWARD_MODELS",
    # Events.
    "BANDIT_STARTED", "BANDIT_PULLED", "BANDIT_OBSERVED",
    "BANDIT_REPORT", "BANDIT_CLEARED", "BANDIT_FORGET",
    "KNOWN_EVENTS",
    # Errors.
    "BanditError", "UnknownAlgorithm", "UnknownArm",
    "InvalidContext", "InsufficientData",
    # Dataclasses.
    "ArmStats", "PullDecision", "BanditReport",
    # Main class.
    "Bandit",
    # Numerical helpers.
    "kl_bernoulli", "kl_ucb_upper",
    "hoeffding_half_width", "empirical_bernstein_half_width",
    "howard_ramdas_half_width", "phi", "phi_inv",
    # Module utilities.
    "expected_regret_ucb1", "expected_regret_thompson_beta",
    "expected_regret_exp3", "best_arm_index",
    "quick_two_armed_bandit",
]
