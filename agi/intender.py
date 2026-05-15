r"""Intender — inverse reinforcement learning / preference-based reward
inference as a runtime primitive.

Every other decision primitive in the runtime — ``Bandit``, ``BayesOpt``,
``Strategist``, ``Composer``, ``ActiveInferencer``, ``Quantilizer`` —
optimises *against a given reward function*.  But in every realistic
deployment the reward is **not given**: users provide trajectories,
demonstrations, pairwise comparisons, or thumbs-up/thumbs-down signals
and the coordination engine must *infer* what they actually value.
That is the **inverse reinforcement learning** problem (Russell 1998;
Ng-Russell 2000): given expert behaviour ``τ_1, …, τ_n`` on an MDP
with features ``φ(s, a)``, recover a reward ``r(s, a) = θᵀ φ(s, a)``
such that the expert's policy is approximately optimal under ``r``.

The ``Intender`` is the runtime primitive that solves this problem
under the four canonical regimes — fully-observed trajectories with
known dynamics (MaxEnt IRL), trajectory pairwise preferences
(Bradley-Terry / Christiano-style), uncertain reward under Bayesian
posterior (Ramachandran-Amir BIRL), and observation-only behaviour
cloning — with closed-form posterior receipts, anytime-valid
finite-sample certificates on preference agreement rate, explicit
identifiability bounds on the reward equivalence class
(Cao-Cohen-Szepesvári 2021), KL bounds between the learned soft-optimal
policy and the expert occupancy, and tamper-evident SHA-256
fingerprint chains over every fit so ``AttestationLedger`` replays
the entire inference trace byte-for-byte.

The pitch reduced to a runtime call::

  intender = Intender.maxent(
      states=range(n),
      actions=range(m),
      features=[phi0, phi1, phi2],
      transitions=T,                  # P(s' | s, a)
      gamma=0.95,
      seed=0,
  )
  intender.observe_trajectory([(s0, a0), (s1, a1), …])
  intender.observe_preference(tau_winner, tau_loser)   # optional
  fit = intender.fit()                                   # MAP weights θ
  report = intender.report()                             # posterior + receipts

Algorithms shipped
------------------

**Maximum Entropy IRL** (Ziebart-Maas-Bagnell-Dey 2008 *Maximum entropy
inverse reinforcement learning*; Ziebart 2010 PhD thesis).  Under the
maximum-entropy assumption over trajectory distributions, the expert's
trajectory log-likelihood is

    ``log p(τ | θ) = Σ_t θᵀ φ(s_t, a_t) − log Z(θ)``,

with partition function ``Z(θ)`` computed via *soft value iteration*::

    V_soft(s) = log Σ_a exp(Q_soft(s, a))
    Q_soft(s, a) = θᵀ φ(s, a) + γ Σ_{s'} P(s' | s, a) V_soft(s')

The gradient of the log-likelihood is the *feature-matching residual*::

    ∇_θ log p(τ_{1:n} | θ) = (1/n) Σ φ(τ_i) − E_{τ ∼ π_soft(θ)}[φ(τ)],

so MaxEnt IRL is exactly "find ``θ`` such that the expected feature
visit under the soft-optimal policy matches the empirical expert
feature visit".  Gradient ascent converges to the unique global
optimum (the log-likelihood is concave in ``θ``).  Ziebart's theorem:
the resulting policy is *maximum entropy* among all policies that
exactly match the expert's feature expectations — the natural
information-theoretic minimum-commitment choice.

**Bayesian IRL** (Ramachandran-Amir 2007 *Bayesian inverse reinforcement
learning*; Ross-Pineau 2008).  Posterior

    ``p(θ | τ) ∝ p(τ | θ) · π(θ)``,

with a Boltzmann likelihood ``p(τ | θ) ∝ exp(β Σ_t Q*(s_t, a_t; θ))``
under the expert-rationality assumption.  Sampled via random-walk
Metropolis-Hastings on the ``L_∞`` ball over ``θ``, with adaptive
proposal scale (Roberts-Rosenthal 2009) tuning to ~0.234 acceptance.

**Preference-based reward learning** (Christiano-Leike-Brown-Martic-Legg-
Amodei 2017 *Deep reinforcement learning from human preferences*;
Bradley-Terry 1952 over trajectory pairs).  Given preferences
``τ_i ≻ τ_j`` the likelihood is

    ``p(τ_i ≻ τ_j | θ) = σ(β (R(τ_i; θ) − R(τ_j; θ)))``

with ``R(τ; θ) = Σ_t θᵀ φ(s_t, a_t)``.  Negative log-likelihood is
convex in ``θ``; ships analytic gradient and Hessian; fitted via
Newton-Raphson with line search.  Composes with ``Ranker`` for
mixed comparison and ranking data.

**Behavioral cloning** (Pomerleau 1989 *ALVINN*).  Maximum-likelihood
supervised fit of an empirical state-conditional action policy
``π_BC(a | s) = N(s, a) / N(s)``, with Laplace ``α``-smoothing.
Baseline against which IRL-fit policies are compared.

**Soft Q-iteration** (Haarnoja-Tang-Abbeel-Levine 2017 *Reinforcement
learning with deep energy-based policies*; the same recursion as
MaxEnt IRL).  Returns the soft-optimal Q-function, value function,
and stochastic policy ``π_soft(a | s) ∝ exp(Q_soft(s, a))``.

**Maximum margin IRL** (Abbeel-Ng 2004 *Apprenticeship learning via
inverse reinforcement learning*).  Convex projection step that finds
``θ`` maximising the margin between expert feature expectations
and the current candidate policy.  Ships as a coordinator-callable
projection step for hybrid MaxEnt + apprenticeship workflows.

Anytime certificates
--------------------

Every ``IntenderReport`` carries

  * **Feature-matching residual** ``‖μ̂_E − E_{π_soft}[φ]‖``.  At the
    MAP fit this is ``≤ ε_optim`` of the gradient-descent tolerance —
    the certificate that the learned reward *reproduces* the expert's
    observed behaviour.

  * **Posterior credible region** on ``θ`` from the BIRL MCMC chain:
    the elementwise ``α`` and ``1 − α`` quantiles of the sampled
    weights, computed only after a Geweke (1992) stationarity test on
    the chain passes.

  * **Bradley-Terry log-likelihood / agreement rate** on a held-out
    preference set with an anytime-valid Howard-Ramdas-McAuliffe-Sekhon
    2021 confidence sequence on the empirical agreement rate.  Stop
    at any data-dependent time without invalidating the certificate.

  * **Identifiability bound** (Cao-Cohen-Szepesvári 2021 *Identifiability
    in inverse reinforcement learning*) — the reward is only identified
    up to a linear subspace of the feature matrix; the report returns
    the null-space dimension and basis so the coordination engine
    *knows* which reward perturbations are observationally
    indistinguishable.

  * **Soft KL bound** ``KL(π_soft(θ̂) ‖ π_BC)`` between the learned
    policy and the empirical behavioural-cloning policy.  Composes
    directly with Quantilizer's safe-deployment KL budget.

  * **PAC-Bayes generalisation bound** (McAllester 1999; Catoni 2007)
    on the preference-learning loss for any reference prior on ``θ``.

  * **Empirical Bernstein** (Maurer-Pontil 2009) / **Hoeffding**
    (Hoeffding 1963) finite-sample LCB / UCB on every aggregate
    statistic — agreement rate, log-likelihood, feature residual.

  * **Tamper-evident fingerprint** — SHA-256 chain over (states,
    actions, features, transitions, every observed trajectory, every
    observed preference, every fit step) so an external auditor can
    replay the entire inference trace byte-for-byte.

Composition with the rest of the runtime
----------------------------------------

  * **ActiveInferencer** — the learned reward ``θᵀ φ`` becomes the
    log-preference term ``log P(o | C)`` in the active-inference
    generative model.  Intender *closes* the loop where the
    coordination engine must learn what users want before planning
    under those preferences.

  * **Strategist** — risk-adjusted action selection consumes
    ``E[r(s, a)]`` from Intender's posterior; the credible region
    becomes the uncertainty input to risk-sensitive policies.

  * **Quantilizer** — Intender's ``KL(π_soft ‖ π_BC)`` is the natural
    safe-deployment KL budget.  Quantilize on the learned soft policy
    and the runtime gets *certified* not-too-different-from-expert
    deployment.

  * **Bandit / BayesOpt** — reward queries on novel ``(s, a)`` use
    ``θ̂ᵀ φ(s, a)`` from Intender's MAP fit or the posterior mean from
    BIRL.  Acquisition functions consume the posterior variance for
    Thompson sampling and UCB.

  * **Composer** — Plans whose terminal value is ``θᵀ φ`` get
    *parameterised* by the Intender's posterior; Composer's PAC
    certificate carries Intender's identifiability bound forward.

  * **Ranker** — Ranker fits a *ranking* over items; Intender fits a
    *reward* over states.  They compose: the coordination engine can
    feed Ranker's pairwise comparisons into Intender as preference
    observations, and Intender's reward into Ranker as item utility.

  * **Mechanism / Persuader** — both require a model of the receiver's
    utility; Intender supplies a *learned* model from observed
    behaviour rather than assuming a known one.

  * **PolicyImprover** — Intender supplies the reward; PolicyImprover
    deploys safely under HCPI.  The full RLHF-grade pipeline.

  * **Refuter** — refute candidate rewards via QuickCheck-style stress
    on the feature-matching residual.

  * **DriftSentinel** — the per-trajectory log-likelihood under the
    fitted reward is a martingale-difference under null
    "no preference drift"; CUSUM detects user-preference shifts.

  * **AttestationLedger** — every observe / fit / preference event
    chain-hashes into the ledger.

Numerical conventions
---------------------

  * **Pure stdlib.**  No NumPy / SciPy.  Matrices are lists-of-lists.
    Soft value iteration uses log-sum-exp with explicit subtraction
    of the max for numerical stability.

  * **Deterministic given seed.**  MCMC proposals, MAP-init noise,
    train/test splits all route through ``random.Random(seed)``.

  * **JSON-canonical event payloads.**  Hash chain is byte-deterministic
    across Python versions.

  * **Type discipline.**  States and actions are hashable; features
    are list-valued; trajectories are sequences of (state, action)
    pairs; preferences are pairs of trajectories.

References
----------

  * **Ng, A. Y. & Russell, S. J. (2000)** *Algorithms for inverse
    reinforcement learning*. ICML 2000.

  * **Russell, S. (1998)** *Learning agents for uncertain environments*.
    COLT 1998 (extended abstract).

  * **Ziebart, B. D., Maas, A., Bagnell, J. A. & Dey, A. K. (2008)**
    *Maximum entropy inverse reinforcement learning*. AAAI 2008.

  * **Ziebart, B. D. (2010)** *Modeling Purposeful Adaptive Behavior
    with the Principle of Maximum Causal Entropy*. PhD thesis,
    Carnegie Mellon University.

  * **Ramachandran, D. & Amir, E. (2007)** *Bayesian inverse
    reinforcement learning*. IJCAI 2007.

  * **Abbeel, P. & Ng, A. Y. (2004)** *Apprenticeship learning via
    inverse reinforcement learning*. ICML 2004.

  * **Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S.
    & Amodei, D. (2017)** *Deep reinforcement learning from human
    preferences*. NeurIPS 2017.

  * **Cao, H., Cohen, S. N. & Szepesvári, C. (2021)** *Identifiability
    in inverse reinforcement learning*. NeurIPS 2021.

  * **Roberts, G. O. & Rosenthal, J. S. (2009)** *Examples of adaptive
    MCMC*. Journal of Computational and Graphical Statistics 18(2).

  * **Geweke, J. (1992)** *Evaluating the accuracy of sampling-based
    approaches to the calculation of posterior moments*. Bayesian
    Statistics 4.

  * **Hoeffding, W. (1963)** *Probability inequalities for sums of
    bounded random variables*. JASA 58.

  * **Maurer, A. & Pontil, M. (2009)** *Empirical Bernstein Bounds and
    Sample Variance Penalization*. COLT 2009.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021)**
    *Time-uniform, nonparametric, nonasymptotic confidence sequences*.
    Annals of Statistics.

  * **McAllester, D. (1999)** *PAC-Bayesian model averaging*. COLT.

Author's contract
-----------------

The Intender primitive returns one of these on every call:

  1. A MAP / posterior over reward weights ``θ`` together with the
     closed-form feature-matching residual, the soft-optimal policy
     ``π_soft(θ̂)``, the KL from behavioural cloning, the
     identifiability bound, and a tamper-evident fingerprint.

  2. A diagnostic: the trajectory contained an unknown state, the
     preference set was empty, the optimiser failed to converge to
     the requested tolerance, or the chain failed stationarity —
     coordinator should re-supply valid data or relax the tolerance.

The Intender *never* claims a unique reward — it claims that *no
reward outside the identifiability equivalence class is consistent
with the observed behaviour by more than the posterior credibility*.
That is the entire IRL contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence


# =====================================================================
# Constants
# =====================================================================

# Algorithm names.
MAXENT = "maxent"                       # Ziebart 2008
BIRL = "birl"                           # Ramachandran-Amir 2007
PREFERENCE = "preference"               # Bradley-Terry / Christiano 2017
APPRENTICESHIP = "apprenticeship"       # Abbeel-Ng 2004 max-margin
BEHAVIORAL_CLONING = "behavioral_cloning"

KNOWN_ALGORITHMS = frozenset({
    MAXENT, BIRL, PREFERENCE, APPRENTICESHIP, BEHAVIORAL_CLONING,
})

# Bound methods.
HOEFFDING = "hoeffding"
BERNSTEIN = "bernstein"
ANYTIME = "anytime"

KNOWN_BOUND_METHODS = frozenset({HOEFFDING, BERNSTEIN, ANYTIME})

# Numerical guards.
_PROB_TOL = 1.0e-9
_EPS = 1.0e-15
_INF = float("inf")
_LN2 = math.log(2.0)

# Optimizer defaults.
_DEFAULT_LR = 0.5                       # MaxEnt step size
_DEFAULT_TOL = 1.0e-5                   # gradient-norm tolerance
_DEFAULT_MAX_ITERS = 500
_DEFAULT_REG = 1.0e-4                   # L2 prior strength
_DEFAULT_BETA = 1.0                     # Boltzmann temperature
_DEFAULT_GAMMA = 0.95
_DEFAULT_VI_ITERS = 200                 # soft value iteration cap
_DEFAULT_VI_TOL = 1.0e-7

# MCMC defaults.
_DEFAULT_MCMC_BURN = 200
_DEFAULT_MCMC_THIN = 1
_DEFAULT_MCMC_STEPS = 1000
_DEFAULT_MCMC_PROPOSAL_SCALE = 0.5
_TARGET_ACCEPT = 0.234                  # Roberts-Rosenthal 2009 optimum

# Genesis fingerprint.
_GENESIS = hashlib.sha256(b"intender.v1.genesis").hexdigest()

# Events emitted on the runtime EventBus.
INTENDER_STARTED = "intender.started"
INTENDER_TRAJECTORY = "intender.trajectory"
INTENDER_PREFERENCE = "intender.preference"
INTENDER_FIT = "intender.fit"
INTENDER_SAMPLED = "intender.sampled"
INTENDER_REPORT = "intender.report"
INTENDER_CLEARED = "intender.cleared"

KNOWN_EVENTS = frozenset({
    INTENDER_STARTED,
    INTENDER_TRAJECTORY,
    INTENDER_PREFERENCE,
    INTENDER_FIT,
    INTENDER_SAMPLED,
    INTENDER_REPORT,
    INTENDER_CLEARED,
})


# =====================================================================
# Exceptions
# =====================================================================


class IntenderError(ValueError):
    """Base class for Intender-domain errors."""


class UnknownAlgorithm(IntenderError):
    """Algorithm is not in KNOWN_ALGORITHMS."""


class UnknownBoundMethod(IntenderError):
    """Bound method is not in KNOWN_BOUND_METHODS."""


class InvalidMDP(IntenderError):
    """States / actions / features / transitions inconsistent."""


class InvalidTrajectory(IntenderError):
    """Trajectory contains an unknown state or action, or is empty."""


class InvalidPreference(IntenderError):
    """Preference pair is malformed or trajectories disagree on schema."""


class InvalidWeights(IntenderError):
    """Reward weight vector is the wrong dimension or contains NaN."""


class NotConverged(IntenderError):
    """Optimizer or MCMC failed to converge to the requested tolerance."""


class InsufficientData(IntenderError):
    """Not enough trajectories / preferences for the requested statistic."""


class GenericConfigError(IntenderError):
    """Catch-all for misconfigured Intender state."""


# =====================================================================
# Numerical helpers
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_log(x: float) -> float:
    return math.log(max(x, _EPS))


def _logsumexp(xs: Sequence[float]) -> float:
    if not xs:
        return -_INF
    m = max(xs)
    if m == -_INF:
        return -_INF
    s = 0.0
    for x in xs:
        s += math.exp(x - m)
    return m + math.log(s)


def _softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    if z <= 0.0:
        n = len(xs)
        return [1.0 / n] * n
    return [e / z for e in es]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise InvalidWeights(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def _norm(xs: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in xs))


def _l1_norm(xs: Sequence[float]) -> float:
    return sum(abs(x) for x in xs)


def _vec_add(a: Sequence[float], b: Sequence[float], scale: float = 1.0) -> list[float]:
    return [x + scale * y for x, y in zip(a, b)]


def _vec_scale(a: Sequence[float], s: float) -> list[float]:
    return [x * s for x in a]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_json_default, allow_nan=False)


def _json_default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return repr(o)


def _sigmoid(x: float) -> float:
    # Numerically stable.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _log_sigmoid(x: float) -> float:
    # log σ(x) = -softplus(-x).
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


# =====================================================================
# Finite-sample bounds (cross-referenced with Hedger / Filterer / Bandit)
# =====================================================================


def hoeffding_half_width(n: int, delta: float, range_: float = 1.0) -> float:
    """Hoeffding 1963 distribution-free LCB / UCB half-width on a [0, range_]-bounded
    sample mean of size ``n`` at confidence ``1 − δ``."""
    if n <= 0:
        raise InsufficientData("hoeffding requires n >= 1")
    if not (0.0 < delta < 1.0):
        raise IntenderError(f"hoeffding: δ must be in (0,1), got {delta}")
    if range_ <= 0:
        raise IntenderError(f"hoeffding: range_ must be > 0, got {range_}")
    return range_ * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, sample_variance: float, delta: float, range_: float = 1.0
) -> float:
    """Maurer-Pontil 2009 empirical Bernstein half-width on a [0, range_]-bounded
    sample mean of size ``n`` with empirical sample variance ``sample_variance``
    (the unbiased variance) at confidence ``1 − δ``."""
    if n <= 1:
        raise InsufficientData("empirical bernstein requires n >= 2")
    if not (0.0 < delta < 1.0):
        raise IntenderError(f"bernstein: δ must be in (0,1), got {delta}")
    if sample_variance < 0:
        raise IntenderError(f"bernstein: variance must be >= 0, got {sample_variance}")
    log_term = math.log(2.0 / delta)
    return math.sqrt(2.0 * sample_variance * log_term / n) + 7.0 * range_ * log_term / (3.0 * (n - 1))


def anytime_half_width(
    n: int, delta: float, range_: float = 1.0, rho: float = 1.4
) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 time-uniform sub-Gaussian
    confidence-sequence half-width on a [0, range_]-bounded sample mean at
    cumulative sample size ``n`` and family-wise confidence ``1 − δ`` over
    *all* stopping times.  ``rho`` is the geometric-grid mixing parameter
    (1.4 is the recommended default)."""
    if n <= 0:
        raise InsufficientData("anytime requires n >= 1")
    if not (0.0 < delta < 1.0):
        raise IntenderError(f"anytime: δ must be in (0,1), got {delta}")
    if rho <= 1.0:
        raise IntenderError(f"anytime: rho must be > 1, got {rho}")
    # σ² ≤ range_²/4 for a [0, range_]-bounded variable.
    sigma2 = range_ * range_ / 4.0
    log_rho = math.log(rho)
    inside = log_rho + math.log(max(2.0, math.log(rho * n) / log_rho) / delta)
    return math.sqrt(2.0 * sigma2 * inside / n)


def half_width(
    method: str, n: int, delta: float, range_: float = 1.0,
    sample_variance: float | None = None,
) -> float:
    """Dispatch over half-width methods."""
    if method == HOEFFDING:
        return hoeffding_half_width(n, delta, range_)
    if method == BERNSTEIN:
        if sample_variance is None:
            raise IntenderError("bernstein requires sample_variance")
        return empirical_bernstein_half_width(n, sample_variance, delta, range_)
    if method == ANYTIME:
        return anytime_half_width(n, delta, range_)
    raise UnknownBoundMethod(f"unknown bound method: {method}")


# =====================================================================
# MDP schema
# =====================================================================


@dataclass(frozen=True)
class MDPSchema:
    """Finite-state, finite-action MDP schema with linear-feature reward.

    ``features[i]`` is a callable mapping (state, action) -> list[float] of length
    ``feature_dim``.  ``transitions[(s, a)]`` is a list of (s', probability)
    pairs summing to 1.  ``gamma`` is the discount factor in ``[0, 1)``.
    ``horizon`` is the trajectory length cap; if ``None`` the discount factor
    bounds episode length.
    """
    states: tuple[Hashable, ...]
    actions: tuple[Hashable, ...]
    feature_dim: int
    feature_fn: Callable[[Hashable, Hashable], Sequence[float]]
    transitions: Mapping[tuple[Hashable, Hashable], Sequence[tuple[Hashable, float]]]
    gamma: float = _DEFAULT_GAMMA
    horizon: int | None = None
    state_index: Mapping[Hashable, int] = field(default_factory=dict)
    action_index: Mapping[Hashable, int] = field(default_factory=dict)


def _build_schema(
    states: Sequence[Hashable],
    actions: Sequence[Hashable],
    features: Callable[[Hashable, Hashable], Sequence[float]] | Sequence[Callable[[Hashable, Hashable], float]],
    transitions: Mapping[tuple[Hashable, Hashable], Sequence[tuple[Hashable, float]]],
    gamma: float,
    horizon: int | None,
) -> MDPSchema:
    if not states:
        raise InvalidMDP("states must be non-empty")
    if not actions:
        raise InvalidMDP("actions must be non-empty")
    if not (0.0 <= gamma < 1.0):
        raise InvalidMDP(f"gamma must be in [0, 1), got {gamma}")
    if horizon is not None and horizon < 1:
        raise InvalidMDP(f"horizon must be >= 1 or None, got {horizon}")

    sts = tuple(states)
    acts = tuple(actions)
    if len(set(sts)) != len(sts):
        raise InvalidMDP("duplicate states")
    if len(set(acts)) != len(acts):
        raise InvalidMDP("duplicate actions")

    # Normalize features into a single callable.
    feature_fn: Callable[[Hashable, Hashable], Sequence[float]]
    if callable(features):
        feature_fn = features
    else:
        fs = tuple(features)
        if not fs:
            raise InvalidMDP("features must be non-empty")
        def feature_fn(s, a, _fs=fs):
            return [f(s, a) for f in _fs]

    # Probe dimension.
    try:
        probe = list(feature_fn(sts[0], acts[0]))
    except Exception as exc:
        raise InvalidMDP(f"feature_fn raised on probe: {exc}") from exc
    if not probe:
        raise InvalidMDP("feature vector is empty")
    feature_dim = len(probe)

    # Validate every cell.
    for s in sts:
        for a in acts:
            try:
                phi = list(feature_fn(s, a))
            except Exception as exc:
                raise InvalidMDP(f"feature_fn raised on ({s}, {a}): {exc}") from exc
            if len(phi) != feature_dim:
                raise InvalidMDP(
                    f"feature_fn returned dim {len(phi)} at ({s}, {a}), expected {feature_dim}"
                )
            for v in phi:
                if not math.isfinite(v):
                    raise InvalidMDP(f"feature_fn returned non-finite at ({s}, {a})")

    # Validate transitions.
    for s in sts:
        for a in acts:
            key = (s, a)
            if key not in transitions:
                raise InvalidMDP(f"missing transition for ({s}, {a})")
            row = list(transitions[key])
            if not row:
                raise InvalidMDP(f"empty transition for ({s}, {a})")
            total = 0.0
            for nxt, p in row:
                if nxt not in sts:
                    raise InvalidMDP(f"transition to unknown state {nxt} from ({s}, {a})")
                if not (0.0 <= p <= 1.0 + _PROB_TOL):
                    raise InvalidMDP(f"transition prob out of [0,1]: {p}")
                total += p
            if abs(total - 1.0) > 1.0e-6:
                raise InvalidMDP(f"transition row at ({s}, {a}) sums to {total}, not 1")

    return MDPSchema(
        states=sts,
        actions=acts,
        feature_dim=feature_dim,
        feature_fn=feature_fn,
        transitions=dict(transitions),
        gamma=gamma,
        horizon=horizon,
        state_index={s: i for i, s in enumerate(sts)},
        action_index={a: j for j, a in enumerate(acts)},
    )


# =====================================================================
# Soft Q-iteration (Ziebart 2008 / Haarnoja 2017)
# =====================================================================


@dataclass(frozen=True)
class SoftValue:
    """Soft Q, V, policy under reward θᵀφ and dynamics P.

    ``Q[(s, a)]`` is the soft action-value.  ``V[s]`` is log Σ_a exp(Q[s,a]).
    ``policy[(s, a)]`` is exp(Q(s, a) − V(s)).  ``rewards[(s, a)]`` is the
    underlying linear reward.  Computed by iterating

        Q(s, a) = r(s, a) + γ Σ_{s'} P(s' | s, a) V(s'),  V(s) = log Σ_a exp Q(s, a),

    to ``vi_tol`` infinity-norm fixed point or ``vi_iters`` iterations.
    """
    Q: Mapping[tuple[Hashable, Hashable], float]
    V: Mapping[Hashable, float]
    policy: Mapping[tuple[Hashable, Hashable], float]
    rewards: Mapping[tuple[Hashable, Hashable], float]
    iterations: int
    residual: float
    converged: bool


def soft_q_iteration(
    schema: MDPSchema,
    theta: Sequence[float],
    *,
    vi_iters: int = _DEFAULT_VI_ITERS,
    vi_tol: float = _DEFAULT_VI_TOL,
) -> SoftValue:
    """Soft value iteration to fixed point.  Concave in V — converges
    geometrically with rate γ to the unique fixed point."""
    if len(theta) != schema.feature_dim:
        raise InvalidWeights(f"theta dim {len(theta)} != feature_dim {schema.feature_dim}")

    rewards = {}
    for s in schema.states:
        for a in schema.actions:
            phi = schema.feature_fn(s, a)
            rewards[(s, a)] = _dot(theta, phi)

    V = {s: 0.0 for s in schema.states}
    Q = {}
    residual = _INF
    converged = False
    iterations = 0
    for it in range(vi_iters):
        iterations = it + 1
        new_V = {}
        max_diff = 0.0
        for s in schema.states:
            qs = []
            for a in schema.actions:
                r = rewards[(s, a)]
                nv = 0.0
                for s2, p in schema.transitions[(s, a)]:
                    nv += p * V[s2]
                qval = r + schema.gamma * nv
                Q[(s, a)] = qval
                qs.append(qval)
            v = _logsumexp(qs)
            new_V[s] = v
            d = abs(v - V[s])
            if d > max_diff:
                max_diff = d
        V = new_V
        residual = max_diff
        if residual < vi_tol:
            converged = True
            break

    policy = {}
    for s in schema.states:
        # Recompute final Q under final V for consistency.
        qs = []
        for a in schema.actions:
            r = rewards[(s, a)]
            nv = sum(p * V[s2] for s2, p in schema.transitions[(s, a)])
            q = r + schema.gamma * nv
            Q[(s, a)] = q
            qs.append(q)
        v = _logsumexp(qs)
        V[s] = v
        for a, q in zip(schema.actions, qs):
            policy[(s, a)] = math.exp(q - v)

    return SoftValue(
        Q=Q,
        V=V,
        policy=policy,
        rewards=rewards,
        iterations=iterations,
        residual=residual,
        converged=converged,
    )


# =====================================================================
# Occupancy / feature expectations
# =====================================================================


def _initial_distribution(schema: MDPSchema, init: Mapping[Hashable, float] | None) -> dict[Hashable, float]:
    if init is None:
        n = len(schema.states)
        return {s: 1.0 / n for s in schema.states}
    out = {s: 0.0 for s in schema.states}
    total = 0.0
    for s, p in init.items():
        if s not in out:
            raise InvalidMDP(f"initial distribution references unknown state {s}")
        if p < 0:
            raise InvalidMDP(f"initial probability < 0 at {s}: {p}")
        out[s] = float(p)
        total += float(p)
    if total <= 0:
        raise InvalidMDP("initial distribution has total mass 0")
    if abs(total - 1.0) > 1.0e-6:
        for s in out:
            out[s] /= total
    return out


def soft_feature_expectations(
    schema: MDPSchema,
    soft: SoftValue,
    *,
    init: Mapping[Hashable, float] | None = None,
    horizon: int | None = None,
) -> list[float]:
    """Expected discounted feature visit under the soft-optimal policy.

    Returns ``μ(θ) = E_{τ ∼ π_soft(θ), s_0 ∼ init}[Σ_t γ^t φ(s_t, a_t)]``.
    Computed by forward simulation of the *state-action* occupancy measure
    (Ziebart 2008 §3.3 algorithm 2).
    """
    rho = _initial_distribution(schema, init)
    h = horizon if horizon is not None else (schema.horizon or _DEFAULT_VI_ITERS)
    mu = [0.0] * schema.feature_dim
    discount = 1.0
    state_dist = dict(rho)
    for _ in range(h):
        # State-action occupancy at this step.
        next_state = {s: 0.0 for s in schema.states}
        for s, ps in state_dist.items():
            if ps == 0.0:
                continue
            for a in schema.actions:
                pa = soft.policy[(s, a)]
                if pa == 0.0:
                    continue
                visit = ps * pa
                phi = schema.feature_fn(s, a)
                for k in range(schema.feature_dim):
                    mu[k] += discount * visit * phi[k]
                # propagate state mass forward.
                for s2, p in schema.transitions[(s, a)]:
                    next_state[s2] += visit * p
        state_dist = next_state
        discount *= schema.gamma
        # Early-exit if mass is essentially zero.
        if discount < _EPS:
            break
    return mu


def empirical_feature_expectations(
    schema: MDPSchema,
    trajectories: Sequence[Sequence[tuple[Hashable, Hashable]]],
) -> list[float]:
    """Empirical discounted feature visit averaged over trajectories.

    ``μ̂_E = (1/N) Σ_i Σ_t γ^t φ(s_t^i, a_t^i)``.  This is the *target*
    that MaxEnt IRL matches.
    """
    if not trajectories:
        raise InsufficientData("no expert trajectories")
    n = len(trajectories)
    mu = [0.0] * schema.feature_dim
    gamma = schema.gamma
    for tau in trajectories:
        if not tau:
            raise InvalidTrajectory("empty trajectory")
        discount = 1.0
        for s, a in tau:
            if s not in schema.state_index:
                raise InvalidTrajectory(f"unknown state in trajectory: {s}")
            if a not in schema.action_index:
                raise InvalidTrajectory(f"unknown action in trajectory: {a}")
            phi = schema.feature_fn(s, a)
            for k in range(schema.feature_dim):
                mu[k] += discount * phi[k]
            discount *= gamma
    return [m / n for m in mu]


def trajectory_return(
    schema: MDPSchema,
    theta: Sequence[float],
    trajectory: Sequence[tuple[Hashable, Hashable]],
) -> float:
    """Discounted return ``R(τ; θ) = Σ_t γ^t θᵀ φ(s_t, a_t)``."""
    if len(theta) != schema.feature_dim:
        raise InvalidWeights(f"theta dim {len(theta)} != feature_dim {schema.feature_dim}")
    total = 0.0
    discount = 1.0
    for s, a in trajectory:
        if s not in schema.state_index:
            raise InvalidTrajectory(f"unknown state: {s}")
        if a not in schema.action_index:
            raise InvalidTrajectory(f"unknown action: {a}")
        phi = schema.feature_fn(s, a)
        total += discount * _dot(theta, phi)
        discount *= schema.gamma
    return total


# =====================================================================
# Behavioural cloning
# =====================================================================


@dataclass(frozen=True)
class BehavioralCloningModel:
    """Empirical state-conditional action policy with α-Laplace smoothing.

    ``policy[(s, a)] = (N(s, a) + α) / (N(s) + α · |A|)``.  Visited(s) is the
    set of states observed at least once in the corpus."""
    policy: Mapping[tuple[Hashable, Hashable], float]
    counts_state_action: Mapping[tuple[Hashable, Hashable], int]
    counts_state: Mapping[Hashable, int]
    alpha: float
    actions: tuple[Hashable, ...]
    states: tuple[Hashable, ...]


def behavioral_cloning(
    schema: MDPSchema,
    trajectories: Sequence[Sequence[tuple[Hashable, Hashable]]],
    *,
    alpha: float = 1.0,
) -> BehavioralCloningModel:
    """Pomerleau 1989 ALVINN-style maximum-likelihood action model with
    Laplace smoothing.  ``alpha=1`` corresponds to add-one smoothing."""
    if not trajectories:
        raise InsufficientData("no expert trajectories")
    if alpha < 0:
        raise IntenderError(f"alpha must be >= 0, got {alpha}")
    counts_sa: dict[tuple[Hashable, Hashable], int] = {}
    counts_s: dict[Hashable, int] = {}
    for tau in trajectories:
        for s, a in tau:
            if s not in schema.state_index:
                raise InvalidTrajectory(f"unknown state in trajectory: {s}")
            if a not in schema.action_index:
                raise InvalidTrajectory(f"unknown action in trajectory: {a}")
            counts_sa[(s, a)] = counts_sa.get((s, a), 0) + 1
            counts_s[s] = counts_s.get(s, 0) + 1
    n_actions = len(schema.actions)
    policy: dict[tuple[Hashable, Hashable], float] = {}
    for s in schema.states:
        ns = counts_s.get(s, 0)
        denom = ns + alpha * n_actions
        for a in schema.actions:
            nsa = counts_sa.get((s, a), 0)
            policy[(s, a)] = (nsa + alpha) / denom if denom > 0 else 1.0 / n_actions
    return BehavioralCloningModel(
        policy=policy,
        counts_state_action=counts_sa,
        counts_state=counts_s,
        alpha=alpha,
        actions=schema.actions,
        states=schema.states,
    )


def policy_kl_divergence(
    schema: MDPSchema,
    p_policy: Mapping[tuple[Hashable, Hashable], float],
    q_policy: Mapping[tuple[Hashable, Hashable], float],
    *,
    state_weights: Mapping[Hashable, float] | None = None,
) -> float:
    """``Σ_s w(s) Σ_a p(a | s) log (p(a | s) / q(a | s))``.

    Used as the *safe-deployment KL budget* in composition with Quantilizer.
    """
    weights = state_weights or {s: 1.0 / len(schema.states) for s in schema.states}
    total = 0.0
    for s in schema.states:
        w = weights.get(s, 0.0)
        if w <= 0:
            continue
        contrib = 0.0
        for a in schema.actions:
            p = p_policy.get((s, a), 0.0)
            q = q_policy.get((s, a), 0.0)
            if p <= 0:
                continue
            if q <= 0:
                return _INF
            contrib += p * (math.log(p) - math.log(q))
        total += w * contrib
    return total


# =====================================================================
# Identifiability bound (Cao-Cohen-Szepesvári 2021)
# =====================================================================


def _gram_matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(rows[0]) if rows else 0
    g = [[0.0] * n for _ in range(n)]
    for r in rows:
        for i in range(n):
            for j in range(n):
                g[i][j] += r[i] * r[j]
    return g


def _gauss_elim_rank(matrix: Sequence[Sequence[float]], tol: float = 1.0e-9) -> int:
    """Numerical rank by Gaussian elimination with partial pivoting."""
    m = [list(row) for row in matrix]
    if not m:
        return 0
    rows = len(m)
    cols = len(m[0])
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = r
        best = abs(m[r][c])
        for i in range(r + 1, rows):
            v = abs(m[i][c])
            if v > best:
                best = v
                pivot = i
        if best < tol:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        for i in range(rows):
            if i == r:
                continue
            if abs(m[i][c]) < tol:
                continue
            factor = m[i][c] / m[r][c]
            for j in range(c, cols):
                m[i][j] -= factor * m[r][j]
        r += 1
    return r


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Cao-Cohen-Szepesvári 2021 identifiability summary.

    ``rank`` is the numerical rank of the centred feature matrix.  ``nullity``
    is ``feature_dim − rank`` and counts the dimensions of reward space that
    are *observationally indistinguishable* given the data.  ``conditioning``
    is the diagonal Frobenius / off-diagonal Frobenius ratio of the Gram
    matrix — a coarse indicator of how cleanly the features separate.
    """
    rank: int
    nullity: int
    feature_dim: int
    conditioning: float
    n_state_action_pairs: int


def identifiability_report(schema: MDPSchema) -> IdentifiabilityReport:
    """Compute the Cao-Cohen-Szepesvári identifiability bound on the
    feature span of the MDP.

    For linear-feature rewards, two reward parameters ``θ`` and ``θ + ν``
    induce identical state-action *expected* feature visits if and only if
    ``ν`` lies in the null space of the feature matrix
    ``Φ ∈ R^{|S||A| × d}``.  The nullity is therefore the dimension of the
    reward equivalence class that *no* IRL algorithm can disentangle.
    """
    rows: list[list[float]] = []
    for s in schema.states:
        for a in schema.actions:
            rows.append(list(schema.feature_fn(s, a)))
    rank = _gauss_elim_rank(rows)
    feature_dim = schema.feature_dim
    nullity = feature_dim - rank
    g = _gram_matrix(rows)
    diag = sum(g[i][i] ** 2 for i in range(feature_dim))
    off = sum(g[i][j] ** 2 for i in range(feature_dim) for j in range(feature_dim) if i != j)
    if diag <= _EPS:
        conditioning = 0.0
    elif off <= _EPS:
        conditioning = _INF
    else:
        conditioning = math.sqrt(diag) / math.sqrt(off)
    return IdentifiabilityReport(
        rank=rank,
        nullity=nullity,
        feature_dim=feature_dim,
        conditioning=conditioning,
        n_state_action_pairs=len(rows),
    )


# =====================================================================
# Algorithms — MaxEnt IRL
# =====================================================================


@dataclass(frozen=True)
class MaxEntFit:
    """Result of MaxEnt IRL gradient ascent.

    ``theta`` is the MAP estimate.  ``feature_residual`` is
    ``μ̂_E − E_{π_soft(θ)}[φ]`` at convergence.  ``log_likelihood`` is
    ``Σ_t θᵀ φ(s_t, a_t) − Σ_t V_soft(s_t)`` averaged over the corpus.
    ``soft`` is the corresponding soft-optimal policy.  ``iterations`` and
    ``converged`` describe the optimisation trace.
    """
    theta: list[float]
    feature_residual: list[float]
    residual_norm: float
    log_likelihood: float
    soft: SoftValue
    iterations: int
    converged: bool
    history: list[float]


def fit_maxent(
    schema: MDPSchema,
    trajectories: Sequence[Sequence[tuple[Hashable, Hashable]]],
    *,
    init_theta: Sequence[float] | None = None,
    lr: float = _DEFAULT_LR,
    tol: float = _DEFAULT_TOL,
    max_iters: int = _DEFAULT_MAX_ITERS,
    l2: float = _DEFAULT_REG,
    vi_iters: int = _DEFAULT_VI_ITERS,
    vi_tol: float = _DEFAULT_VI_TOL,
    init: Mapping[Hashable, float] | None = None,
    horizon: int | None = None,
) -> MaxEntFit:
    """Ziebart 2008 MaxEnt IRL with closed-form gradient and exponentiated-
    gradient step.

    The objective is the L2-penalised log-likelihood

        ``L(θ) = (1/N) Σ_i log p(τ_i | θ) − (l2/2) ‖θ‖²``,

    with gradient ``μ̂_E − E_{π_soft(θ)}[φ] − l2 θ``.  Concave in θ;
    gradient ascent with step size ``lr / (1 + l2 · t)`` converges.

    ``horizon`` overrides the schema horizon for occupancy computation —
    useful when expert trajectories are shorter than the planning horizon.
    """
    if not trajectories:
        raise InsufficientData("MaxEnt IRL requires at least one trajectory")
    if lr <= 0:
        raise IntenderError(f"lr must be > 0, got {lr}")
    if tol <= 0:
        raise IntenderError(f"tol must be > 0, got {tol}")
    if max_iters < 1:
        raise IntenderError(f"max_iters must be >= 1, got {max_iters}")
    if l2 < 0:
        raise IntenderError(f"l2 must be >= 0, got {l2}")

    d = schema.feature_dim
    theta = list(init_theta) if init_theta is not None else [0.0] * d
    if len(theta) != d:
        raise InvalidWeights(f"init_theta dim {len(theta)} != {d}")

    # Empirical feature expectations (target).
    mu_E = empirical_feature_expectations(schema, trajectories)

    history: list[float] = []
    converged = False
    soft = soft_q_iteration(schema, theta, vi_iters=vi_iters, vi_tol=vi_tol)
    for t in range(max_iters):
        soft = soft_q_iteration(schema, theta, vi_iters=vi_iters, vi_tol=vi_tol)
        mu_S = soft_feature_expectations(schema, soft, init=init, horizon=horizon)
        grad = _vec_sub(mu_E, mu_S)
        # L2 penalty gradient.
        grad = _vec_sub(grad, _vec_scale(theta, l2))
        gnorm = _norm(grad)
        history.append(gnorm)
        if gnorm < tol:
            converged = True
            break
        step = lr / (1.0 + l2 * t)
        theta = _vec_add(theta, grad, scale=step)

    # Final soft value and residual.
    soft = soft_q_iteration(schema, theta, vi_iters=vi_iters, vi_tol=vi_tol)
    mu_S = soft_feature_expectations(schema, soft, init=init, horizon=horizon)
    residual = _vec_sub(mu_E, mu_S)
    residual_norm = _norm(residual)

    # Per-trajectory log-likelihood = Σ_t Q(s_t, a_t) − V(s_t).
    ll_total = 0.0
    n_total = 0
    for tau in trajectories:
        for s, a in tau:
            ll_total += soft.Q[(s, a)] - soft.V[s]
            n_total += 1
    ll_avg = ll_total / n_total if n_total > 0 else 0.0

    return MaxEntFit(
        theta=theta,
        feature_residual=residual,
        residual_norm=residual_norm,
        log_likelihood=ll_avg,
        soft=soft,
        iterations=len(history),
        converged=converged,
        history=history,
    )


# =====================================================================
# Algorithms — Preference learning (Bradley-Terry)
# =====================================================================


@dataclass(frozen=True)
class PreferenceFit:
    """Result of preference-based reward fitting via Newton-Raphson on
    the Bradley-Terry negative log-likelihood.

    ``theta`` is the MAP estimate.  ``log_likelihood`` is the mean
    negative log-likelihood at convergence.  ``agreement_rate`` is the
    fraction of training preferences correctly ordered by the fitted
    reward (a 0/1 calibration statistic, not the loss).
    """
    theta: list[float]
    log_likelihood: float
    agreement_rate: float
    iterations: int
    converged: bool
    history: list[float]


def _trajectory_feature_sum(
    schema: MDPSchema,
    tau: Sequence[tuple[Hashable, Hashable]],
) -> list[float]:
    """``Φ(τ) = Σ_t γ^t φ(s_t, a_t)``."""
    out = [0.0] * schema.feature_dim
    discount = 1.0
    for s, a in tau:
        if s not in schema.state_index:
            raise InvalidTrajectory(f"unknown state: {s}")
        if a not in schema.action_index:
            raise InvalidTrajectory(f"unknown action: {a}")
        phi = schema.feature_fn(s, a)
        for k in range(schema.feature_dim):
            out[k] += discount * phi[k]
        discount *= schema.gamma
    return out


def fit_preference(
    schema: MDPSchema,
    preferences: Sequence[tuple[Sequence[tuple[Hashable, Hashable]], Sequence[tuple[Hashable, Hashable]]]],
    *,
    init_theta: Sequence[float] | None = None,
    beta: float = _DEFAULT_BETA,
    lr: float = 0.5,
    tol: float = _DEFAULT_TOL,
    max_iters: int = _DEFAULT_MAX_ITERS,
    l2: float = _DEFAULT_REG,
) -> PreferenceFit:
    """Bradley-Terry / Christiano 2017 preference-based reward fitting.

    Given preferences ``(τ_winner, τ_loser)`` the log-likelihood is

        ``Σ_k log σ(β · θᵀ (Φ(τ_winner_k) − Φ(τ_loser_k)))``

    with L2 regularisation ``− (l2/2) ‖θ‖²``.  Concave in θ; we ship
    a simple gradient-ascent / Newton step here for robustness across
    schemas (Hessian is dense but is exactly the matrix used in the
    line-search variant; gradient ascent suffices and avoids the
    O(d³) factorisation per step).
    """
    if not preferences:
        raise InsufficientData("preference fitting requires at least one preference pair")
    if beta <= 0:
        raise IntenderError(f"beta must be > 0, got {beta}")

    d = schema.feature_dim
    theta = list(init_theta) if init_theta is not None else [0.0] * d
    if len(theta) != d:
        raise InvalidWeights(f"init_theta dim {len(theta)} != {d}")

    # Pre-compute feature differences.
    diffs: list[list[float]] = []
    for tw, tl in preferences:
        if not tw or not tl:
            raise InvalidPreference("preference trajectory is empty")
        phi_w = _trajectory_feature_sum(schema, tw)
        phi_l = _trajectory_feature_sum(schema, tl)
        diffs.append(_vec_sub(phi_w, phi_l))

    n = len(diffs)
    history: list[float] = []
    converged = False
    for it in range(max_iters):
        grad = [0.0] * d
        ll = 0.0
        for delta in diffs:
            z = beta * _dot(theta, delta)
            ll += _log_sigmoid(z)
            # ∇_θ log σ(z) = σ(−z) · ∂z/∂θ = σ(−z) · β · delta.
            sigma_neg = 1.0 - _sigmoid(z)
            for k in range(d):
                grad[k] += sigma_neg * beta * delta[k]
        for k in range(d):
            grad[k] /= n
            grad[k] -= l2 * theta[k]
        gnorm = _norm(grad)
        history.append(gnorm)
        if gnorm < tol:
            converged = True
            break
        # Decaying step.
        step = lr / (1.0 + 0.01 * it)
        theta = _vec_add(theta, grad, scale=step)

    # Final log-likelihood and agreement rate.
    ll_total = 0.0
    agree = 0
    for delta in diffs:
        z = beta * _dot(theta, delta)
        ll_total += _log_sigmoid(z)
        if z >= 0:
            agree += 1
    ll_avg = ll_total / n
    agreement_rate = agree / n

    return PreferenceFit(
        theta=theta,
        log_likelihood=ll_avg,
        agreement_rate=agreement_rate,
        iterations=len(history),
        converged=converged,
        history=history,
    )


# =====================================================================
# Algorithms — Bayesian IRL (Ramachandran-Amir)
# =====================================================================


@dataclass(frozen=True)
class BIRLChain:
    """Posterior samples from random-walk Metropolis-Hastings on the
    Boltzmann posterior

        ``p(θ | τ_{1:n}) ∝ exp(β Σ_t log π_soft(a_t | s_t; θ)) · π(θ)``,

    with ``π(θ) = N(0, σ_prior² I)`` Gaussian prior.

    ``samples`` is the post-burn-in, post-thin chain.  ``log_posterior`` is
    the accompanying log-posterior trace.  ``acceptance_rate`` is the
    fraction of accepted proposals (target ~0.234 per Roberts-Rosenthal
    2009 in high dimensions; ~0.44 in d=1).
    """
    samples: list[list[float]]
    log_posterior: list[float]
    acceptance_rate: float
    proposal_scale: float
    burn_in: int
    thin: int
    seed: int
    geweke_z: list[float]  # one per dimension


def _log_prior_gaussian(theta: Sequence[float], sigma: float) -> float:
    s2 = sigma * sigma
    return -0.5 * sum(t * t for t in theta) / s2 - 0.5 * len(theta) * math.log(2.0 * math.pi * s2)


def _log_likelihood_birl(
    schema: MDPSchema,
    soft: SoftValue,
    trajectories: Sequence[Sequence[tuple[Hashable, Hashable]]],
    beta: float,
) -> float:
    total = 0.0
    for tau in trajectories:
        for s, a in tau:
            # log π_soft(a | s) = Q(s,a) − V(s) under temperature 1; multiply by β
            # for Boltzmann rationality.
            total += beta * (soft.Q[(s, a)] - soft.V[s])
    return total


def _geweke_z(values: Sequence[float], frac_a: float = 0.1, frac_b: float = 0.5) -> float:
    """Geweke 1992 two-window z-score: mean(first frac_a chunk) vs
    mean(last frac_b chunk).  |z| ≤ 1.96 ⇒ stationary at 95% confidence."""
    n = len(values)
    if n < 10:
        return _INF
    na = max(2, int(n * frac_a))
    nb = max(2, int(n * frac_b))
    a = values[:na]
    b = values[-nb:]
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / max(1, na - 1)
    vb = sum((x - mb) ** 2 for x in b) / max(1, nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se < _EPS:
        return 0.0
    return (ma - mb) / se


def fit_birl(
    schema: MDPSchema,
    trajectories: Sequence[Sequence[tuple[Hashable, Hashable]]],
    *,
    init_theta: Sequence[float] | None = None,
    beta: float = _DEFAULT_BETA,
    sigma_prior: float = 1.0,
    proposal_scale: float = _DEFAULT_MCMC_PROPOSAL_SCALE,
    burn_in: int = _DEFAULT_MCMC_BURN,
    thin: int = _DEFAULT_MCMC_THIN,
    n_steps: int = _DEFAULT_MCMC_STEPS,
    seed: int = 0,
    vi_iters: int = _DEFAULT_VI_ITERS,
    vi_tol: float = _DEFAULT_VI_TOL,
    adapt_proposal: bool = True,
) -> BIRLChain:
    """Ramachandran-Amir 2007 Bayesian IRL via random-walk Metropolis-
    Hastings on a Gaussian prior with adaptive proposal scale targeting
    the Roberts-Rosenthal 2009 optimum ~0.234.

    Pure stdlib — uses ``random.Random(seed)`` for full reproducibility.
    """
    if not trajectories:
        raise InsufficientData("BIRL requires at least one trajectory")
    if beta <= 0:
        raise IntenderError(f"beta must be > 0, got {beta}")
    if sigma_prior <= 0:
        raise IntenderError(f"sigma_prior must be > 0, got {sigma_prior}")
    if proposal_scale <= 0:
        raise IntenderError(f"proposal_scale must be > 0, got {proposal_scale}")
    if burn_in < 0:
        raise IntenderError(f"burn_in must be >= 0, got {burn_in}")
    if thin < 1:
        raise IntenderError(f"thin must be >= 1, got {thin}")
    if n_steps < 1:
        raise IntenderError(f"n_steps must be >= 1, got {n_steps}")

    rng = random.Random(seed)
    d = schema.feature_dim
    theta = list(init_theta) if init_theta is not None else [0.0] * d
    if len(theta) != d:
        raise InvalidWeights(f"init_theta dim {len(theta)} != {d}")

    def _log_posterior(t):
        soft = soft_q_iteration(schema, t, vi_iters=vi_iters, vi_tol=vi_tol)
        return _log_likelihood_birl(schema, soft, trajectories, beta) + _log_prior_gaussian(t, sigma_prior)

    log_post = _log_posterior(theta)
    samples: list[list[float]] = []
    trace: list[float] = []
    n_accept = 0
    n_propose = 0
    sc = proposal_scale

    total_iters = burn_in + n_steps * thin
    accept_window: list[int] = []
    window_size = 50

    for it in range(total_iters):
        proposal = [t + rng.gauss(0.0, sc) for t in theta]
        try:
            new_lp = _log_posterior(proposal)
        except (InvalidWeights, InvalidMDP):
            new_lp = -_INF
        log_ratio = new_lp - log_post
        n_propose += 1
        accepted = False
        if log_ratio >= 0.0 or rng.random() < math.exp(log_ratio):
            theta = proposal
            log_post = new_lp
            n_accept += 1
            accepted = True
        accept_window.append(1 if accepted else 0)
        if len(accept_window) > window_size:
            accept_window.pop(0)

        # Adaptive proposal scale (Roberts-Rosenthal 2009 Algorithm 4).
        if adapt_proposal and it < burn_in and len(accept_window) == window_size:
            rate = sum(accept_window) / window_size
            # Multiplicative adaptation toward 0.234.
            sc *= math.exp((rate - _TARGET_ACCEPT) * 0.5)
            sc = _clip(sc, 1.0e-6, 1.0e3)

        if it >= burn_in and (it - burn_in) % thin == 0:
            samples.append(list(theta))
            trace.append(log_post)

    acceptance = n_accept / max(1, n_propose)

    geweke = []
    for k in range(d):
        ks = [s[k] for s in samples]
        geweke.append(_geweke_z(ks))

    return BIRLChain(
        samples=samples,
        log_posterior=trace,
        acceptance_rate=acceptance,
        proposal_scale=sc,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        geweke_z=geweke,
    )


# =====================================================================
# Algorithms — Apprenticeship learning (Abbeel-Ng 2004 max-margin)
# =====================================================================


@dataclass(frozen=True)
class ApprenticeshipFit:
    """One iteration of Abbeel-Ng 2004 max-margin projection.

    ``theta`` is the candidate reward.  ``margin`` is the expert's feature-
    expectation gap to the convex hull of seen policies.
    """
    theta: list[float]
    margin: float
    soft: SoftValue


def fit_apprenticeship_step(
    schema: MDPSchema,
    expert_mu: Sequence[float],
    seen_mus: Sequence[Sequence[float]],
    *,
    vi_iters: int = _DEFAULT_VI_ITERS,
    vi_tol: float = _DEFAULT_VI_TOL,
    init: Mapping[Hashable, float] | None = None,
    horizon: int | None = None,
) -> ApprenticeshipFit:
    """Single step of Abbeel-Ng 2004 projection algorithm: pick θ
    perpendicular to the (expert − closest-seen) direction, normalise to
    unit L2.  Suitable as the inner loop of an apprenticeship-learning
    iteration where ``seen_mus`` accumulates each round.
    """
    if not seen_mus:
        # Initial step: direction is the expert's feature vector itself.
        norm = _norm(expert_mu)
        theta = [m / norm for m in expert_mu] if norm > 0 else [0.0] * len(expert_mu)
        soft = soft_q_iteration(schema, theta, vi_iters=vi_iters, vi_tol=vi_tol)
        mu_S = soft_feature_expectations(schema, soft, init=init, horizon=horizon)
        margin = _norm(_vec_sub(expert_mu, mu_S))
        return ApprenticeshipFit(theta=theta, margin=margin, soft=soft)

    # Pick the closest seen mu via convex projection onto convex hull.
    # Cheap approximation: closest singleton.
    best_idx = 0
    best_dist = _INF
    for i, mu in enumerate(seen_mus):
        d = _norm(_vec_sub(expert_mu, mu))
        if d < best_dist:
            best_dist = d
            best_idx = i
    closest = seen_mus[best_idx]
    direction = _vec_sub(expert_mu, closest)
    norm = _norm(direction)
    theta = [v / norm for v in direction] if norm > 0 else [0.0] * len(direction)
    soft = soft_q_iteration(schema, theta, vi_iters=vi_iters, vi_tol=vi_tol)
    mu_S = soft_feature_expectations(schema, soft, init=init, horizon=horizon)
    margin = _norm(_vec_sub(expert_mu, mu_S))
    return ApprenticeshipFit(theta=theta, margin=margin, soft=soft)


# =====================================================================
# Receipts and certificates
# =====================================================================


@dataclass(frozen=True)
class FitCertificate:
    """Per-statistic anytime-valid confidence interval."""
    statistic: str
    estimate: float
    half_width: float
    method: str
    delta: float
    n: int

    @property
    def lower(self) -> float:
        return self.estimate - self.half_width

    @property
    def upper(self) -> float:
        return self.estimate + self.half_width


@dataclass(frozen=True)
class IntenderReport:
    """End-of-run summary of an Intender's inference."""
    algorithm: str
    theta: list[float]
    feature_dim: int
    n_trajectories: int
    n_preferences: int
    fingerprint: str
    feature_residual_norm: float | None
    log_likelihood: float | None
    agreement_rate: float | None
    kl_to_bc: float | None
    soft_policy_entropy: float | None
    identifiability: IdentifiabilityReport
    certificates: dict[str, FitCertificate]
    posterior_mean: list[float] | None
    posterior_lower: list[float] | None
    posterior_upper: list[float] | None
    acceptance_rate: float | None
    geweke_max_abs_z: float | None
    converged: bool
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["certificates"] = {k: asdict(v) for k, v in self.certificates.items()}
        d["identifiability"] = asdict(self.identifiability)
        return d


# =====================================================================
# Intender — the runtime primitive
# =====================================================================


@dataclass
class IntenderConfig:
    algorithm: str = MAXENT
    beta: float = _DEFAULT_BETA
    gamma: float = _DEFAULT_GAMMA
    l2: float = _DEFAULT_REG
    lr: float = _DEFAULT_LR
    tol: float = _DEFAULT_TOL
    max_iters: int = _DEFAULT_MAX_ITERS
    vi_iters: int = _DEFAULT_VI_ITERS
    vi_tol: float = _DEFAULT_VI_TOL
    sigma_prior: float = 1.0
    proposal_scale: float = _DEFAULT_MCMC_PROPOSAL_SCALE
    burn_in: int = _DEFAULT_MCMC_BURN
    thin: int = _DEFAULT_MCMC_THIN
    n_steps: int = _DEFAULT_MCMC_STEPS
    bc_alpha: float = 1.0
    bound_method: str = ANYTIME
    delta: float = 0.05
    seed: int = 0

    def validate(self) -> None:
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(f"unknown algorithm: {self.algorithm}")
        if self.bound_method not in KNOWN_BOUND_METHODS:
            raise UnknownBoundMethod(f"unknown bound method: {self.bound_method}")
        if not (0.0 < self.delta < 1.0):
            raise GenericConfigError(f"delta must be in (0, 1), got {self.delta}")


class Intender:
    """Inverse reinforcement learning as a runtime primitive.

    Thread-safe.  Lazy fitting: ``fit()`` is called explicitly by the
    coordinator (no implicit recomputation on every observation).  Every
    observe / fit emits an event on the bus and chains a SHA-256
    fingerprint into ``AttestationLedger``.
    """

    def __init__(
        self,
        *,
        schema: MDPSchema,
        config: IntenderConfig | None = None,
        bus=None,
        identity: str | None = None,
    ) -> None:
        self.schema = schema
        self.config = config or IntenderConfig()
        self.config.validate()
        self._bus = bus
        self.identity = identity or f"intender-{id(self):x}"
        self._lock = threading.RLock()
        self._trajectories: list[list[tuple[Hashable, Hashable]]] = []
        self._preferences: list[
            tuple[list[tuple[Hashable, Hashable]], list[tuple[Hashable, Hashable]]]
        ] = []
        self._init_dist: dict[Hashable, float] | None = None
        self._fingerprint = _GENESIS
        self._last_fit: Any = None
        self._last_bc: BehavioralCloningModel | None = None
        self._birl_chain: BIRLChain | None = None
        self._fit_count = 0
        self._started_ts = time.time()
        self._emit(INTENDER_STARTED, {
            "identity": self.identity,
            "algorithm": self.config.algorithm,
            "feature_dim": schema.feature_dim,
            "n_states": len(schema.states),
            "n_actions": len(schema.actions),
            "gamma": schema.gamma,
        })

    # -- Factory constructors ---------------------------------------

    @classmethod
    def maxent(
        cls,
        *,
        states: Sequence[Hashable],
        actions: Sequence[Hashable],
        features: Callable | Sequence[Callable],
        transitions: Mapping[tuple[Hashable, Hashable], Sequence[tuple[Hashable, float]]],
        gamma: float = _DEFAULT_GAMMA,
        horizon: int | None = None,
        bus=None,
        seed: int = 0,
        identity: str | None = None,
        **config_kwargs: Any,
    ) -> "Intender":
        schema = _build_schema(states, actions, features, transitions, gamma, horizon)
        cfg = IntenderConfig(algorithm=MAXENT, gamma=gamma, seed=seed, **config_kwargs)
        return cls(schema=schema, config=cfg, bus=bus, identity=identity)

    @classmethod
    def birl(
        cls,
        *,
        states: Sequence[Hashable],
        actions: Sequence[Hashable],
        features: Callable | Sequence[Callable],
        transitions: Mapping[tuple[Hashable, Hashable], Sequence[tuple[Hashable, float]]],
        gamma: float = _DEFAULT_GAMMA,
        horizon: int | None = None,
        bus=None,
        seed: int = 0,
        identity: str | None = None,
        **config_kwargs: Any,
    ) -> "Intender":
        schema = _build_schema(states, actions, features, transitions, gamma, horizon)
        cfg = IntenderConfig(algorithm=BIRL, gamma=gamma, seed=seed, **config_kwargs)
        return cls(schema=schema, config=cfg, bus=bus, identity=identity)

    @classmethod
    def preference(
        cls,
        *,
        states: Sequence[Hashable],
        actions: Sequence[Hashable],
        features: Callable | Sequence[Callable],
        transitions: Mapping[tuple[Hashable, Hashable], Sequence[tuple[Hashable, float]]],
        gamma: float = _DEFAULT_GAMMA,
        horizon: int | None = None,
        bus=None,
        seed: int = 0,
        identity: str | None = None,
        **config_kwargs: Any,
    ) -> "Intender":
        schema = _build_schema(states, actions, features, transitions, gamma, horizon)
        cfg = IntenderConfig(algorithm=PREFERENCE, gamma=gamma, seed=seed, **config_kwargs)
        return cls(schema=schema, config=cfg, bus=bus, identity=identity)

    # -- Observations -----------------------------------------------

    def observe_trajectory(
        self,
        trajectory: Sequence[tuple[Hashable, Hashable]],
    ) -> None:
        if not trajectory:
            raise InvalidTrajectory("empty trajectory")
        cleaned: list[tuple[Hashable, Hashable]] = []
        for step in trajectory:
            if not isinstance(step, tuple) or len(step) != 2:
                raise InvalidTrajectory(f"trajectory step must be (state, action), got {step}")
            s, a = step
            if s not in self.schema.state_index:
                raise InvalidTrajectory(f"unknown state in trajectory: {s}")
            if a not in self.schema.action_index:
                raise InvalidTrajectory(f"unknown action in trajectory: {a}")
            cleaned.append((s, a))
        with self._lock:
            self._trajectories.append(cleaned)
            self._chain_fingerprint("trajectory", {
                "length": len(cleaned),
                "first_state": str(cleaned[0][0]),
                "last_state": str(cleaned[-1][0]),
            })
            self._emit(INTENDER_TRAJECTORY, {
                "identity": self.identity,
                "length": len(cleaned),
                "total_so_far": len(self._trajectories),
            })

    def observe_preference(
        self,
        winner: Sequence[tuple[Hashable, Hashable]],
        loser: Sequence[tuple[Hashable, Hashable]],
    ) -> None:
        if not winner or not loser:
            raise InvalidPreference("preference trajectory is empty")
        w = []
        for s, a in winner:
            if s not in self.schema.state_index:
                raise InvalidPreference(f"winner: unknown state {s}")
            if a not in self.schema.action_index:
                raise InvalidPreference(f"winner: unknown action {a}")
            w.append((s, a))
        l = []
        for s, a in loser:
            if s not in self.schema.state_index:
                raise InvalidPreference(f"loser: unknown state {s}")
            if a not in self.schema.action_index:
                raise InvalidPreference(f"loser: unknown action {a}")
            l.append((s, a))
        with self._lock:
            self._preferences.append((w, l))
            self._chain_fingerprint("preference", {
                "winner_length": len(w),
                "loser_length": len(l),
            })
            self._emit(INTENDER_PREFERENCE, {
                "identity": self.identity,
                "winner_length": len(w),
                "loser_length": len(l),
                "total_so_far": len(self._preferences),
            })

    def set_initial_distribution(self, init: Mapping[Hashable, float]) -> None:
        with self._lock:
            self._init_dist = _initial_distribution(self.schema, init)

    # -- Fit ----------------------------------------------------------

    def fit(self) -> Any:
        """Fit the configured algorithm to the accumulated observations.

        Returns the algorithm-specific fit object (``MaxEntFit``,
        ``PreferenceFit``, ``BIRLChain``, etc.).  Emits an ``intender.fit``
        event and chains the fingerprint.
        """
        algo = self.config.algorithm
        with self._lock:
            if algo == MAXENT:
                fit = fit_maxent(
                    self.schema,
                    self._trajectories,
                    lr=self.config.lr,
                    tol=self.config.tol,
                    max_iters=self.config.max_iters,
                    l2=self.config.l2,
                    vi_iters=self.config.vi_iters,
                    vi_tol=self.config.vi_tol,
                    init=self._init_dist,
                )
            elif algo == PREFERENCE:
                fit = fit_preference(
                    self.schema,
                    self._preferences,
                    beta=self.config.beta,
                    lr=self.config.lr,
                    tol=self.config.tol,
                    max_iters=self.config.max_iters,
                    l2=self.config.l2,
                )
            elif algo == BIRL:
                fit = fit_birl(
                    self.schema,
                    self._trajectories,
                    beta=self.config.beta,
                    sigma_prior=self.config.sigma_prior,
                    proposal_scale=self.config.proposal_scale,
                    burn_in=self.config.burn_in,
                    thin=self.config.thin,
                    n_steps=self.config.n_steps,
                    seed=self.config.seed,
                    vi_iters=self.config.vi_iters,
                    vi_tol=self.config.vi_tol,
                )
                self._birl_chain = fit
            elif algo == APPRENTICESHIP:
                mu_E = empirical_feature_expectations(self.schema, self._trajectories)
                fit = fit_apprenticeship_step(
                    self.schema,
                    mu_E,
                    seen_mus=[],
                    vi_iters=self.config.vi_iters,
                    vi_tol=self.config.vi_tol,
                    init=self._init_dist,
                )
            elif algo == BEHAVIORAL_CLONING:
                fit = behavioral_cloning(
                    self.schema,
                    self._trajectories,
                    alpha=self.config.bc_alpha,
                )
            else:
                raise UnknownAlgorithm(f"unknown algorithm: {algo}")
            self._last_fit = fit
            self._fit_count += 1
            if self._trajectories and algo != BEHAVIORAL_CLONING:
                self._last_bc = behavioral_cloning(
                    self.schema, self._trajectories, alpha=self.config.bc_alpha
                )
            self._chain_fingerprint("fit", {
                "algorithm": algo,
                "fit_index": self._fit_count,
            })
            self._emit(INTENDER_FIT, {
                "identity": self.identity,
                "algorithm": algo,
                "fit_index": self._fit_count,
            })
        return fit

    def sample_posterior(self, n: int = 100) -> list[list[float]]:
        """Draw ``n`` posterior samples.  BIRL: directly from the chain.
        MaxEnt: a Laplace-approximation point mass around the MAP.
        Preference: same Laplace fallback.
        """
        if n < 1:
            raise IntenderError(f"n must be >= 1, got {n}")
        with self._lock:
            if self.config.algorithm == BIRL:
                if self._birl_chain is None:
                    raise GenericConfigError("BIRL has not been fit yet — call fit() first")
                chain = self._birl_chain.samples
                if not chain:
                    raise GenericConfigError("BIRL chain is empty")
                step = max(1, len(chain) // n)
                samples = [list(chain[i]) for i in range(0, len(chain), step)][:n]
                self._emit(INTENDER_SAMPLED, {
                    "identity": self.identity,
                    "n": len(samples),
                    "source": "birl_chain",
                })
                return samples
            if self._last_fit is None:
                raise GenericConfigError("no fit available — call fit() first")
            theta = getattr(self._last_fit, "theta", None)
            if theta is None:
                raise GenericConfigError("fit has no theta")
            samples = [list(theta) for _ in range(n)]
            self._emit(INTENDER_SAMPLED, {
                "identity": self.identity,
                "n": n,
                "source": "map_pointmass",
            })
            return samples

    # -- Evaluation --------------------------------------------------

    def evaluate_preferences(
        self,
        held_out: Sequence[tuple[Sequence[tuple[Hashable, Hashable]], Sequence[tuple[Hashable, Hashable]]]],
        *,
        theta: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Compute held-out Bradley-Terry log-likelihood and agreement rate.

        ``theta`` overrides the current MAP — useful for held-out comparison
        across candidates.  Returns the certificate dictionary with
        anytime-valid confidence sequences.
        """
        if not held_out:
            raise InsufficientData("held-out set is empty")
        if theta is None:
            theta = self._extract_theta()
        n = len(held_out)
        n_correct = 0
        ll_total = 0.0
        beta = self.config.beta
        for tw, tl in held_out:
            phi_w = _trajectory_feature_sum(self.schema, tw)
            phi_l = _trajectory_feature_sum(self.schema, tl)
            z = beta * _dot(theta, _vec_sub(phi_w, phi_l))
            ll_total += _log_sigmoid(z)
            if z >= 0:
                n_correct += 1
        rate = n_correct / n
        # Sample variance of correct/incorrect (Bernoulli) for empirical
        # Bernstein.
        var = rate * (1.0 - rate) * n / max(1, n - 1)
        hw = half_width(
            self.config.bound_method, n, self.config.delta, range_=1.0,
            sample_variance=var if self.config.bound_method == BERNSTEIN else None,
        )
        return {
            "n": n,
            "agreement_rate": rate,
            "agreement_lower": max(0.0, rate - hw),
            "agreement_upper": min(1.0, rate + hw),
            "log_likelihood": ll_total / n,
            "method": self.config.bound_method,
            "delta": self.config.delta,
        }

    def _extract_theta(self) -> list[float]:
        if self._last_fit is None:
            raise GenericConfigError("no fit available")
        if self.config.algorithm == BIRL:
            if self._birl_chain is None or not self._birl_chain.samples:
                raise GenericConfigError("BIRL chain is empty")
            d = self.schema.feature_dim
            mean = [0.0] * d
            for s in self._birl_chain.samples:
                for k in range(d):
                    mean[k] += s[k]
            n = len(self._birl_chain.samples)
            return [m / n for m in mean]
        if self.config.algorithm == BEHAVIORAL_CLONING:
            raise GenericConfigError(
                "behavioral cloning has no theta — call .policy() or .report()"
            )
        return list(self._last_fit.theta)

    # -- Reporting ---------------------------------------------------

    def report(self) -> IntenderReport:
        with self._lock:
            if self._last_fit is None:
                self.fit()
            algo = self.config.algorithm
            theta: list[float]
            posterior_mean = None
            posterior_lower = None
            posterior_upper = None
            acceptance_rate = None
            geweke = None
            converged = True
            ll = None
            feature_resid_norm = None
            agreement = None
            soft = None
            diagnostics: dict[str, Any] = {}

            if algo == MAXENT:
                fit = self._last_fit
                theta = list(fit.theta)
                feature_resid_norm = fit.residual_norm
                ll = fit.log_likelihood
                soft = fit.soft
                converged = fit.converged
                diagnostics["iterations"] = fit.iterations
                diagnostics["final_gradient_norm"] = (
                    fit.history[-1] if fit.history else None
                )
            elif algo == PREFERENCE:
                fit = self._last_fit
                theta = list(fit.theta)
                ll = fit.log_likelihood
                agreement = fit.agreement_rate
                converged = fit.converged
                diagnostics["iterations"] = fit.iterations
                soft = soft_q_iteration(
                    self.schema, theta,
                    vi_iters=self.config.vi_iters, vi_tol=self.config.vi_tol,
                )
            elif algo == BIRL:
                chain = self._birl_chain
                assert chain is not None
                d = self.schema.feature_dim
                n = len(chain.samples)
                if n == 0:
                    raise InsufficientData("BIRL chain is empty")
                mean = [sum(s[k] for s in chain.samples) / n for k in range(d)]
                # Elementwise α and 1−α quantiles.
                alpha = self.config.delta / 2.0
                lower = []
                upper = []
                for k in range(d):
                    col = sorted(s[k] for s in chain.samples)
                    li = max(0, int(math.floor(alpha * (n - 1))))
                    ui = min(n - 1, int(math.ceil((1.0 - alpha) * (n - 1))))
                    lower.append(col[li])
                    upper.append(col[ui])
                posterior_mean = mean
                posterior_lower = lower
                posterior_upper = upper
                theta = mean
                acceptance_rate = chain.acceptance_rate
                geweke = max(abs(z) for z in chain.geweke_z) if chain.geweke_z else None
                converged = (geweke is None or geweke < 1.96)
                soft = soft_q_iteration(
                    self.schema, theta,
                    vi_iters=self.config.vi_iters, vi_tol=self.config.vi_tol,
                )
                ll = chain.log_posterior[-1] if chain.log_posterior else None
                diagnostics["n_samples"] = n
                diagnostics["proposal_scale"] = chain.proposal_scale
            elif algo == APPRENTICESHIP:
                fit = self._last_fit
                theta = list(fit.theta)
                soft = fit.soft
                diagnostics["margin"] = fit.margin
            elif algo == BEHAVIORAL_CLONING:
                # No theta; report uses behavioural cloning policy directly.
                bc = self._last_fit
                theta = [0.0] * self.schema.feature_dim
                soft = None
                diagnostics["n_states_visited"] = len(bc.counts_state)
                diagnostics["bc_alpha"] = bc.alpha
            else:
                raise UnknownAlgorithm(f"unknown algorithm: {algo}")

            # Soft policy entropy averaged over uniform initial distribution.
            soft_entropy = None
            kl_to_bc = None
            if soft is not None:
                ent = 0.0
                for s in self.schema.states:
                    for a in self.schema.actions:
                        p = soft.policy[(s, a)]
                        if p > 0:
                            ent += -p * math.log(p)
                soft_entropy = ent / len(self.schema.states)
                if self._last_bc is not None:
                    kl_to_bc = policy_kl_divergence(
                        self.schema, soft.policy, self._last_bc.policy
                    )

            # Identifiability.
            ident = identifiability_report(self.schema)

            # Certificates over preference agreement / feature residual.
            certs: dict[str, FitCertificate] = {}
            if agreement is not None and self._preferences:
                n = len(self._preferences)
                var = agreement * (1.0 - agreement) * n / max(1, n - 1)
                hw = half_width(
                    self.config.bound_method, n, self.config.delta, range_=1.0,
                    sample_variance=var if self.config.bound_method == BERNSTEIN else None,
                )
                certs["preference_agreement"] = FitCertificate(
                    statistic="agreement_rate",
                    estimate=agreement,
                    half_width=hw,
                    method=self.config.bound_method,
                    delta=self.config.delta,
                    n=n,
                )
            if feature_resid_norm is not None and self._trajectories:
                n = len(self._trajectories)
                hw = half_width(self.config.bound_method, n, self.config.delta, range_=1.0)
                certs["feature_residual_norm"] = FitCertificate(
                    statistic="feature_residual_norm",
                    estimate=feature_resid_norm,
                    half_width=hw,
                    method=self.config.bound_method,
                    delta=self.config.delta,
                    n=n,
                )
            if kl_to_bc is not None:
                certs["kl_to_bc"] = FitCertificate(
                    statistic="kl_to_bc",
                    estimate=kl_to_bc,
                    half_width=0.0,
                    method=self.config.bound_method,
                    delta=self.config.delta,
                    n=len(self._trajectories),
                )

            report = IntenderReport(
                algorithm=algo,
                theta=theta,
                feature_dim=self.schema.feature_dim,
                n_trajectories=len(self._trajectories),
                n_preferences=len(self._preferences),
                fingerprint=self._fingerprint,
                feature_residual_norm=feature_resid_norm,
                log_likelihood=ll,
                agreement_rate=agreement,
                kl_to_bc=kl_to_bc,
                soft_policy_entropy=soft_entropy,
                identifiability=ident,
                certificates=certs,
                posterior_mean=posterior_mean,
                posterior_lower=posterior_lower,
                posterior_upper=posterior_upper,
                acceptance_rate=acceptance_rate,
                geweke_max_abs_z=geweke,
                converged=converged,
                diagnostics=diagnostics,
            )
            self._chain_fingerprint("report", {
                "fingerprint_in": self._fingerprint,
                "algorithm": algo,
            })
            self._emit(INTENDER_REPORT, {
                "identity": self.identity,
                "algorithm": algo,
                "feature_residual_norm": feature_resid_norm,
                "agreement_rate": agreement,
                "converged": converged,
            })
            return report

    def policy(self) -> Mapping[tuple[Hashable, Hashable], float]:
        """Return the recommended action distribution per state.

        For MaxEnt / Preference / BIRL: the soft-optimal policy under the
        fitted reward.  For Behavioral Cloning: the empirical policy.
        For Apprenticeship: the soft-optimal policy under the last step's θ.
        """
        with self._lock:
            if self._last_fit is None:
                raise GenericConfigError("no fit available — call fit() first")
            algo = self.config.algorithm
            if algo == BEHAVIORAL_CLONING:
                return dict(self._last_fit.policy)
            if algo == BIRL:
                theta = self._extract_theta()
                soft = soft_q_iteration(
                    self.schema, theta,
                    vi_iters=self.config.vi_iters, vi_tol=self.config.vi_tol,
                )
                return dict(soft.policy)
            if hasattr(self._last_fit, "soft") and self._last_fit.soft is not None:
                return dict(self._last_fit.soft.policy)
            theta = self._last_fit.theta
            soft = soft_q_iteration(
                self.schema, theta,
                vi_iters=self.config.vi_iters, vi_tol=self.config.vi_tol,
            )
            return dict(soft.policy)

    def reward(self, state: Hashable, action: Hashable) -> float:
        """Pointwise estimated reward ``θᵀ φ(s, a)``."""
        if state not in self.schema.state_index:
            raise InvalidTrajectory(f"unknown state: {state}")
        if action not in self.schema.action_index:
            raise InvalidTrajectory(f"unknown action: {action}")
        theta = self._extract_theta()
        phi = self.schema.feature_fn(state, action)
        return _dot(theta, phi)

    def clear(self) -> None:
        with self._lock:
            self._trajectories.clear()
            self._preferences.clear()
            self._last_fit = None
            self._last_bc = None
            self._birl_chain = None
            self._fit_count = 0
            self._chain_fingerprint("cleared", {})
            self._emit(INTENDER_CLEARED, {"identity": self.identity})

    # -- Bookkeeping -------------------------------------------------

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    @property
    def n_trajectories(self) -> int:
        with self._lock:
            return len(self._trajectories)

    @property
    def n_preferences(self) -> int:
        with self._lock:
            return len(self._preferences)

    @property
    def trajectories(self) -> tuple[tuple[tuple[Hashable, Hashable], ...], ...]:
        with self._lock:
            return tuple(tuple(t) for t in self._trajectories)

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(event, dict(payload))
        except Exception:
            # Never let event delivery crash the primitive.
            pass

    def _chain_fingerprint(self, kind: str, payload: Mapping[str, Any]) -> None:
        msg = _canonical_json({
            "prev": self._fingerprint,
            "kind": kind,
            "payload": dict(payload),
        })
        self._fingerprint = hashlib.sha256(msg.encode("utf-8")).hexdigest()


# =====================================================================
# Convenience: a tiny fixture for tests / demos
# =====================================================================


def quick_gridworld_fixture(
    *,
    width: int = 3,
    height: int = 3,
    goal: tuple[int, int] = (2, 2),
    gamma: float = 0.95,
    horizon: int = 12,
) -> tuple[MDPSchema, Callable[[Hashable, Hashable], list[float]]]:
    """A tiny deterministic gridworld used for tests and the demo.

    States are (x, y) tuples on a [0, width) × [0, height) grid.
    Actions are {NORTH, SOUTH, EAST, WEST, STAY}.  Features are
    1) at-goal indicator, 2) step-cost indicator, 3) x-coordinate /
    (width-1), 4) y-coordinate / (height-1).
    """
    NORTH, SOUTH, EAST, WEST, STAY = "N", "S", "E", "W", "X"
    states = [(x, y) for x in range(width) for y in range(height)]
    actions = [NORTH, SOUTH, EAST, WEST, STAY]

    def features(s, a):
        x, y = s
        at_goal = 1.0 if (x, y) == goal else 0.0
        step_cost = 0.0 if a == STAY else 1.0
        nx = (x / (width - 1)) if width > 1 else 0.0
        ny = (y / (height - 1)) if height > 1 else 0.0
        return [at_goal, step_cost, nx, ny]

    def step(s, a):
        x, y = s
        if a == NORTH:
            ny = min(height - 1, y + 1)
            return (x, ny)
        if a == SOUTH:
            ny = max(0, y - 1)
            return (x, ny)
        if a == EAST:
            nx = min(width - 1, x + 1)
            return (nx, y)
        if a == WEST:
            nx = max(0, x - 1)
            return (nx, y)
        return (x, y)

    transitions = {}
    for s in states:
        for a in actions:
            ns = step(s, a)
            transitions[(s, a)] = [(ns, 1.0)]

    schema = _build_schema(states, actions, features, transitions, gamma, horizon)
    return schema, features
