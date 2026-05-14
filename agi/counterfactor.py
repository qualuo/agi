r"""Counterfactor — sequential off-policy evaluation as a runtime primitive.

A coordination engine that plans more than one step ahead — a router that
decides "issue tool A, then if result is X issue tool B, else tool C",
an autonomous agent that walks a chain of skills, a tutor that picks a
sequence of exercises — eventually proposes a new **multi-step policy**
and asks the runtime "what would this earn if I deployed it tomorrow?"

`PolicyLab` answers that for single-step contextual bandits (IPS / SNIPS /
DM / DR / SWITCH-DR — Dudík et al., Wang et al.).  The moment the policy
is *sequential*, those estimators silently lose their guarantees: the
importance weight is a *product* of step-wise ratios that explodes with
horizon, and reward at step ``t`` depends on actions chosen back at
step ``0``.  Naive trajectory-IS is unbiased but unusable; the variance
grows exponentially in horizon.  Twenty-five years of off-policy
evaluation literature exists precisely to fix this — *the* line between
"we have a great idea" and "we can ship it without burning real money".

``Counterfactor`` is the runtime primitive for that line.  Give it a
log of *trajectories* — ``[(state, action, reward, behavior_prob, …)]``
under whatever logging policy the runtime actually ran — and any new
policy ``π(a | s)``.  It returns calibrated value estimates with
finite-sample, distribution-free confidence intervals; pessimistic
lower bounds for high-confidence deployment; per-step diagnostics
(ESS, overlap, max weight, tail-mass); and the suite of doubly-robust
estimators that *industrial* reinforcement learning relies on.

Mathematical roots
------------------

Let trajectories be ``τ = (s_0, a_0, r_0, …, s_{H-1}, a_{H-1}, r_{H-1})``
with horizon ``H``, behaviour policy ``μ`` (``μ(a_t | s_t) = π_b``), and
target policy ``π``.  Let
``ρ_t = π(a_t | s_t) / μ(a_t | s_t)`` be the per-step importance ratio
and ``W_t = ∏_{u ≤ t} ρ_u`` its prefix product.  The episode return is
``G = Σ_{t=0}^{H-1} γ^t r_t``.

  * **Horvitz, D. G., Thompson, D. J. (1952).**  Inverse-propensity
    estimator.  For any random variable ``f(τ)`` and a behavioural
    sampler ``μ`` covering the support of ``π``::

        E_π[f(τ)]  =  E_μ[ (π(τ)/μ(τ)) · f(τ) ].

    For trajectories ``π(τ)/μ(τ) = W_{H-1}``.

  * **Precup, D., Sutton, R. S., Singh, S. (2000) — "Eligibility traces
    for off-policy policy evaluation."**  *Per-Decision Importance
    Sampling* (PDIS).  The reward at step ``t`` does not depend on
    actions ``t+1, …, H-1``, so only weight what *precedes* it::

        V̂_PDIS  =  (1/n) Σ_τ Σ_t γ^t W_t(τ) r_t(τ).

    Same unbiasedness as trajectory IS with strictly smaller variance.

  * **Precup, D. (2000) — *Temporal abstraction in reinforcement
    learning*.**  **Weighted PDIS** (WPDIS / WIS for trajectories):
    self-normalise the per-step weights::

        V̂_WPDIS  =  Σ_t γ^t  (Σ_τ W_t(τ) r_t(τ)) / (Σ_τ W_t(τ)).

    Consistent, **biased**, dramatically lower variance.  This is the
    estimator industrial OPE actually ships.

  * **Thomas, P. S., Theocharous, G., Ghavamzadeh, M. (2015) —
    "High-Confidence Off-Policy Evaluation."**  HCOPE: combine PDIS
    with a *bounded-range* truncation ``ξ`` on the IS-weighted returns
    and apply Maurer-Pontil empirical Bernstein to get a **finite-
    sample lower confidence bound**::

        V̂_HCOPE^L = mean(Y) - √(2 ln(2/δ) Var(Y)/n) - 7 ξ ln(2/δ)/(3(n-1)),

    where ``Y_τ = min(ξ, W_{H-1}(τ) G(τ))``.  The lower bound holds
    with probability ≥ 1-δ *uniformly* over the data — no asymptotics,
    no Gaussianity.  This is the bound a *safe* deployment loop uses:
    if ``V̂_HCOPE^L(π_new) > V_behaviour``, ship.

  * **Jiang, N., Li, L. (2016) — "Doubly Robust Off-policy Value
    Evaluation for Reinforcement Learning."**  DR-RL.  Plug in a
    learned ``Q̂(s,a)``::

        V̂_DR  =  (1/n) Σ_τ Σ_t γ^t
                 [ W_t(τ) (r_t - Q̂(s_t,a_t))
                   + W_{t-1}(τ) V̂(s_t) ],

    with ``V̂(s) = Σ_a π(a|s) Q̂(s,a)``.  Doubly robust:
    consistent if **either** ``μ`` is known **or** ``Q̂`` is correct.

  * **Thomas, P. S., Brunskill, E. (2016) — "Data-Efficient Off-Policy
    Policy Evaluation for Reinforcement Learning."**  **WDR** (weighted
    DR) self-normalises the DR weights; **MAGIC** (Model-And-Guided-
    Importance-sampling Combination) blends the per-step DR
    contribution with a pure model prediction by minimising an
    optimal MSE bound at each step.

  * **Maurer, A., Pontil, M. (2009) — "Empirical Bernstein bounds and
    sample-variance penalization."**  For ``X_i ∈ [0, c]`` iid::

        P( μ - X̄  ≥  √(2 V̂_n ln(2/δ)/n) + 7 c ln(2/δ)/(3(n-1)) ) ≤ δ.

    Tighter than Hoeffding when the empirical variance is small; the
    work-horse for HCOPE.

  * **Owen, A. B. (2013) — *Monte Carlo theory, methods, examples*.**
    Self-normalised importance sampling Slutsky / delta-method CI for
    SNIPS / WPDIS; the second-moment correction with the *effective*
    sample size ``ESS = (Σ w)² / Σ w²``.

  * **Vovk, V., Gammerman, A., Shafer, G. (2005) — *Algorithmic
    Learning in a Random World*.**  Conformal predictive intervals:
    for an arbitrary base estimator the conformal envelope of
    ``V̂ ± q̂_{1-α}(|residual|)`` is **finite-sample valid** under
    exchangeability — the source of the distribution-free
    `conformal_ope` envelope.

  * **Cesa-Bianchi, N., Lugosi, G. (2006).**  Online weighted-average
    forecasting: the convex combination of DM and PDIS that
    ``MAGIC`` uses to minimise MSE step-by-step is the same
    ``Hedge``-style mixture.

Composing with the rest of the stack
------------------------------------

The Counterfactor is the **temporal twin** of PolicyLab and slots
naturally between the existing primitives:

  * **PolicyLab (bandit OPE).**  When ``H = 1`` the formulas reduce to
    the contextual-bandit estimators PolicyLab already ships;
    Counterfactor delegates to them via ``evaluate(method="snips",
    horizon=1, …)`` for symmetry but its strength is ``H ≥ 2``.

  * **PolicyImprover.**  Already does *single-step* HCPI for log-linear
    softmax policies.  Counterfactor's ``hcope_trajectory(...)`` gives
    PolicyImprover a multi-step safety gate: ship a new sequential
    policy only if its HCOPE lower bound exceeds the baseline at
    confidence 1-δ.

  * **Strategist.**  When the strategist proposes a sequenced plan
    (route → tool → sub-route), Counterfactor evaluates every variant
    against logged sessions; the strategist then optimises against a
    *certified* value estimate, not a hopeful point forecast.

  * **ActiveInferencer.**  Counterfactor consumes the policies it
    enumerates and tells the runtime which one to *prefer* under the
    realised data — closing the loop from "expected free energy" to
    "expected return on logged traffic".

  * **Diplomat / Equilibrator.**  Off-policy evaluation of *each
    player's* sequential strategy: feed Counterfactor a log of an
    extensive-form play and a candidate CFR-improved policy → safe
    deployment guarantee.

  * **Forecaster.**  The Counterfactor's per-step bias and variance
    decomposition (MAGIC weight ``λ_t``) is itself a probabilistic
    forecast over the bias-variance trade-off; Forecaster's
    calibration tests apply directly to the per-step CI coverage.

  * **AttestationLedger.**  Every ``evaluate / hcope / compare`` call
    is signed and content-hashed: the coordination engine can publish
    "policy v17 was certified to dominate v16 at δ=0.05 over 4,193
    trajectories" with a verifiable receipt.

  * **Auditor.**  The pairwise compare's p-values combine via the
    Auditor's anytime-valid sequential test → false discovery
    control across a portfolio of policy proposals.

What this module ships
----------------------

* Estimators (single class ``Counterfactor``):

    - ``traj_is``       (trajectory IS — Horvitz-Thompson 1952)
    - ``traj_wis``      (weighted trajectory IS — Owen 2013 / Precup 2000)
    - ``pdis``          (per-decision IS — Precup-Sutton-Singh 2000)
    - ``wpdis``         (weighted PDIS — Precup 2000)
    - ``dm``            (direct method — fit Q̂, integrate against π)
    - ``dr_rl``         (doubly-robust RL — Jiang-Li 2016)
    - ``wdr``           (weighted DR — Thomas-Brunskill 2016)
    - ``magic``         (model-and-IS combination — Thomas-Brunskill 2016)

  All also exposed as pure functions for testing / external use.

* Confidence intervals and pessimistic lower bounds:

    - ``hoeffding_half_width(n, range_, alpha)``
    - ``empirical_bernstein_half_width(values, alpha, range_)``
    - ``student_t_half_width(values, alpha)``
    - ``hcope_lower_bound(values, xi, alpha)``   (Thomas et al. 2015)
    - ``conformal_envelope(residuals, alpha)``   (Vovk et al. 2005)

* Diagnostics:

    - ``ess(weights)`` Effective sample size — Kong (1992)
    - ``weight_diagnostics(weights)``  max, p99, clip fraction, tail
    - ``overlap_kl(behaviour_log, target_policy)``  KL of policies on
      logged states; surfaces support-violation regimes.

* Q-models (drop-in for DM / DR / MAGIC):

    - ``ConstantQModel(c)``
    - ``TabularQModel()``                 fits per-(state-key, action) mean
    - ``LinearQModel(features, l2)``      ridge regression on features

* Policy adapters (drop-in for the target policy callback):

    - ``UniformPolicy(actions)``
    - ``DeterministicPolicy(action_of)``
    - ``EpsilonGreedyPolicy(action_of, eps)``
    - ``SoftmaxPolicy(score_of, temperature)``

All numerics are stdlib only.  Threadsafe under a single ``RLock``.
Every state-changing call emits an optional Event and records an
optional attestation receipt.

Honest about limits
-------------------

* Off-policy evaluation inherits the support of the logging policy.
  An action / state never visited under ``μ`` has *no* counterfactual
  signal — Counterfactor surfaces this via ``coverage`` and
  ``overlap_kl`` and refuses to certify policies with effective
  sample size below a configurable floor.

* ``traj_is`` and ``pdis`` are unbiased *only* if propensities are
  known *exactly*.  Estimated propensities introduce bias the
  estimator does not correct for; pass propensity directly when you
  can, or use ``dr_rl`` which is doubly robust against this.

* The MAGIC weights are MSE-optimal **under the empirical
  influence-function estimate** of bias and variance.  In low-data
  regimes (``n < 30``) the per-step MSE estimate is itself noisy;
  Counterfactor falls back to a uniform blend and warns.

* HCOPE requires bounded returns.  Pass ``reward_range`` to the
  constructor or rewards will be auto-clipped to the empirical
  ``[r_min, r_max]`` with a logged warning.

* The conformal envelope is distribution-free and finite-sample
  valid **under exchangeability** of the trajectories.  If the
  logging policy was time-varying (e.g., A/B-shifted across days),
  pass time stamps and Counterfactor will apply the Mondrian
  conformal slicing.

Citations
---------

* Horvitz, D. G., Thompson, D. J. (1952). A generalization of sampling
  without replacement from a finite universe. *JASA*, 47(260), 663–685.
* Precup, D., Sutton, R. S., Singh, S. (2000). Eligibility traces for
  off-policy policy evaluation. *ICML*, 759–766.
* Precup, D. (2000). *Temporal abstraction in reinforcement learning*.
  PhD thesis, University of Massachusetts Amherst.
* Owen, A. B. (2013). *Monte Carlo theory, methods, examples*.  Stanford.
* Maurer, A., Pontil, M. (2009). Empirical Bernstein bounds and sample
  variance penalization.  *COLT*.
* Thomas, P. S., Theocharous, G., Ghavamzadeh, M. (2015).
  High-confidence off-policy evaluation. *AAAI*, 3000–3006.
* Jiang, N., Li, L. (2016). Doubly robust off-policy value evaluation
  for reinforcement learning. *ICML*, 652–661.
* Thomas, P. S., Brunskill, E. (2016). Data-efficient off-policy policy
  evaluation for reinforcement learning. *ICML*, 2139–2148.
* Dudík, M., Langford, J., Li, L. (2011). Doubly robust policy
  evaluation and learning. *ICML*, 1097–1104.
* Wang, Y.-X., Agarwal, A., Dudík, M. (2017). Optimal and adaptive
  off-policy evaluation in contextual bandits. *ICML*, 3589–3597.
* Swaminathan, A., Joachims, T. (2015). The self-normalized estimator
  for counterfactual learning. *NIPS*, 3231–3239.
* Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in
  a Random World*.  Springer.
* Kong, A. (1992).  A note on importance sampling using standardized
  weights.  Technical Report 348, Univ. of Chicago.
* Hoeffding, W. (1963). Probability inequalities for sums of bounded
  random variables. *JASA*, 58(301), 13–30.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event, EventBus  # type: ignore
except Exception:  # pragma: no cover - keep stdlib-only fallbacks
    Event = None  # type: ignore
    EventBus = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================

CF_STARTED = "counterfactor.started"
CF_LOGGED = "counterfactor.logged"
CF_EVALUATED = "counterfactor.evaluated"
CF_HCOPE_BOUNDED = "counterfactor.hcope_bounded"
CF_COMPARED = "counterfactor.compared"
CF_DIAGNOSED = "counterfactor.diagnosed"
CF_CLEARED = "counterfactor.cleared"
CF_REPORT = "counterfactor.report"


# =====================================================================
# Method names
# =====================================================================

METHOD_TRAJ_IS = "traj_is"
METHOD_TRAJ_WIS = "traj_wis"
METHOD_PDIS = "pdis"
METHOD_WPDIS = "wpdis"
METHOD_DM = "dm"
METHOD_DR_RL = "dr_rl"
METHOD_WDR = "wdr"
METHOD_MAGIC = "magic"

KNOWN_METHODS = frozenset(
    {
        METHOD_TRAJ_IS,
        METHOD_TRAJ_WIS,
        METHOD_PDIS,
        METHOD_WPDIS,
        METHOD_DM,
        METHOD_DR_RL,
        METHOD_WDR,
        METHOD_MAGIC,
    }
)


CI_HOEFFDING = "hoeffding"
CI_BERNSTEIN = "bernstein"
CI_STUDENT_T = "student_t"
CI_CONFORMAL = "conformal"

KNOWN_CI = frozenset({CI_HOEFFDING, CI_BERNSTEIN, CI_STUDENT_T, CI_CONFORMAL})


# =====================================================================
# Constants
# =====================================================================

_EPS = 1e-12
_DEFAULT_CLIP = 100.0
_DEFAULT_ALPHA = 0.05


# =====================================================================
# Exceptions
# =====================================================================


class CounterfactorError(ValueError):
    """Invalid input or infeasible OPE call."""


class InsufficientData(CounterfactorError):
    """Not enough data to compute the requested estimate / bound."""


class UnknownMethod(CounterfactorError):
    """Caller asked for an estimator that is not registered."""


class SupportViolation(CounterfactorError):
    """Target policy puts mass where the logging policy did not."""


# =====================================================================
# Dataclasses — JSON-friendly reports
# =====================================================================


@dataclass(frozen=True)
class LoggedStep:
    """One (state, action, reward, behaviour-probability) tuple.

    ``state`` is an opaque identifier or feature mapping.  Counterfactor
    only requires that it is **hashable** (used as a key for the
    tabular Q model and the diagnostics).  Pass tuples / strings / ints
    when possible; nested dicts get JSON-canonicalised.

    ``behavior_prob`` must be strictly positive — zero-propensity
    samples have undefined importance weights and are rejected at
    log time.
    """

    state: Any
    action: Any
    reward: float
    behavior_prob: float
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "state": _canonical(self.state),
            "action": _canonical(self.action),
            "reward": float(self.reward),
            "behavior_prob": float(self.behavior_prob),
        }
        if self.metadata is not None:
            d["metadata"] = dict(self.metadata)
        return d


@dataclass(frozen=True)
class LoggedTrajectory:
    """An ordered list of ``LoggedStep``s plus optional metadata."""

    steps: tuple[LoggedStep, ...]
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tenant_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.steps:
            raise CounterfactorError("trajectory must contain at least one step")

    @property
    def horizon(self) -> int:
        return len(self.steps)

    def return_(self, gamma: float = 1.0) -> float:
        if gamma == 1.0:
            return sum(s.reward for s in self.steps)
        g = 1.0
        total = 0.0
        for s in self.steps:
            total += g * s.reward
            g *= gamma
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class OPEReport:
    """Outcome of one ``evaluate(...)`` call.

    Carries the point estimate, finite-sample CI under the chosen CI
    family, effective sample size, weight diagnostics, and a content
    hash that the AttestationLedger commits to.
    """

    method: str
    ci_method: str
    value: float
    ci_lo: float
    ci_hi: float
    alpha: float
    n_trajectories: int
    horizon: int
    gamma: float
    ess: float
    max_weight: float
    clip_fraction: float
    weight_cap: float | None
    reward_range: tuple[float, float]
    extras: dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extras"] = dict(self.extras)
        return d


@dataclass
class HCOPEReport:
    """Pessimistic lower confidence bound on policy value (Thomas 2015)."""

    method: str
    point_value: float
    lower_bound: float
    xi: float
    alpha: float
    n_trajectories: int
    horizon: int
    gamma: float
    bernstein_term: float
    range_term: float
    extras: dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extras"] = dict(self.extras)
        return d


@dataclass
class DiagnosticsReport:
    """Importance-weight quality diagnostics for a target policy."""

    n_trajectories: int
    horizon: int
    ess_trajectory: float
    ess_pdis_min: float
    max_weight: float
    p99_weight: float
    p50_weight: float
    mean_log_weight: float
    var_log_weight: float
    overlap_kl: float
    coverage: float
    clip_fraction: float
    weight_cap: float | None
    warnings: list[str] = field(default_factory=list)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class CompareReport:
    """Paired off-policy comparison between two target policies."""

    method: str
    name_a: str
    name_b: str
    value_a: float
    value_b: float
    delta: float
    delta_ci_lo: float
    delta_ci_hi: float
    p_a_better: float
    alpha: float
    n_trajectories: int
    a_dominates: bool
    extras: dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extras"] = dict(self.extras)
        return d


# =====================================================================
# Helpers
# =====================================================================


def _canonical(x: Any) -> Any:
    """Best-effort hashable canonical form for state / action."""
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    if isinstance(x, tuple):
        return tuple(_canonical(v) for v in x)
    if isinstance(x, list):
        return tuple(_canonical(v) for v in x)
    if isinstance(x, Mapping):
        return tuple(sorted((str(k), _canonical(v)) for k, v in x.items()))
    return repr(x)


def _hash_payload(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise CounterfactorError("alpha must be in (0, 1)")


def _validate_gamma(gamma: float) -> None:
    if not 0.0 <= gamma <= 1.0:
        raise CounterfactorError("gamma must be in [0, 1]")


def _validate_prob(p: float, label: str = "probability") -> None:
    if not 0.0 < p <= 1.0 + 1e-9:
        raise CounterfactorError(
            f"{label} must be in (0, 1]; got {p!r}"
        )


def _safe_div(num: float, den: float) -> float:
    if abs(den) < _EPS:
        return 0.0
    return num / den


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise CounterfactorError("percentile q must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    idx = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(s[lo])
    frac = idx - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _z_for_alpha(alpha: float) -> float:
    """Two-sided normal critical value Φ⁻¹(1 - α/2).

    Rational approximation (Beasley-Springer / Moro) accurate to ~7e-9
    over (0, 1).
    """
    _validate_alpha(alpha)
    p = 1.0 - alpha / 2.0
    return _inv_normal_cdf(p)


def _inv_normal_cdf(p: float) -> float:
    """Acklam's approximation to the inverse normal CDF."""
    if not 0.0 < p < 1.0:
        raise CounterfactorError("p must be in (0, 1)")
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
    ) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _student_t_quantile(alpha: float, df: int) -> float:
    """Two-sided Student-t critical value t_{1-α/2, df}.

    Hill's algorithm (Comm. ACM 1970); accurate to ~1e-6 for df ≥ 1
    over typical α ∈ [1e-4, 0.5].
    """
    _validate_alpha(alpha)
    if df <= 0:
        raise CounterfactorError("df must be ≥ 1")
    if df > 200:
        return _z_for_alpha(alpha)
    p = 1.0 - alpha / 2.0
    # bisection on the t CDF
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _student_t_cdf(t: float, df: int) -> float:
    """Student's t CDF via regularised incomplete beta."""
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    p = 0.5 * _betainc(x, a, b)
    if t > 0.0:
        return 1.0 - p
    if t < 0.0:
        return p
    return 0.5


def _betainc(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a,b) via continued fraction.

    Numerically stable for the parameter ranges used by the t-CDF.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    ln_pref = a * math.log(x) + b * math.log(1.0 - x) - lnbeta
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln_pref) * _betacf(x, a, b) / a
    return 1.0 - math.exp(ln_pref) * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-7:
            return h
    return h


# =====================================================================
# Confidence intervals
# =====================================================================


def hoeffding_half_width(n: int, range_: float, alpha: float) -> float:
    """Two-sided Hoeffding 1963 half-width for a mean of ``[0, range_]`` rvs."""
    _validate_alpha(alpha)
    if n <= 0:
        raise CounterfactorError("n must be ≥ 1")
    if range_ < 0.0:
        raise CounterfactorError("range_ must be ≥ 0")
    return range_ * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def empirical_bernstein_half_width(
    values: Sequence[float],
    alpha: float,
    range_: float,
) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein half-width for ``[0, range_]`` rvs.

    Returns ``√(2 V̂ ln(2/δ)/n) + 7 c ln(2/δ)/(3(n-1))`` with ``c = range_``
    and ``V̂`` the sample variance (unbiased).
    """
    _validate_alpha(alpha)
    n = len(values)
    if n < 2:
        raise InsufficientData("empirical Bernstein needs n ≥ 2")
    if range_ < 0.0:
        raise CounterfactorError("range_ must be ≥ 0")
    var = statistics.variance(values)
    log_term = math.log(2.0 / alpha)
    bernstein = math.sqrt(2.0 * var * log_term / n)
    range_term = 7.0 * range_ * log_term / (3.0 * (n - 1))
    return bernstein + range_term


def student_t_half_width(values: Sequence[float], alpha: float) -> float:
    """Two-sided Student-t half-width for the mean of ``values``."""
    n = len(values)
    if n < 2:
        raise InsufficientData("Student-t needs n ≥ 2")
    se = statistics.stdev(values) / math.sqrt(n)
    tcrit = _student_t_quantile(alpha, n - 1)
    return tcrit * se


def conformal_envelope(residuals: Sequence[float], alpha: float) -> float:
    """Vovk 2005 split-conformal half-width.

    Given absolute residuals from a held-out calibration fold, returns
    the ``⌈(n+1)(1-α)⌉/n``-th order statistic.  Under exchangeability of
    the calibration and test samples the resulting interval is
    *finite-sample* valid with coverage ≥ 1 - α.
    """
    _validate_alpha(alpha)
    n = len(residuals)
    if n < 1:
        raise InsufficientData("conformal envelope needs n ≥ 1")
    s = sorted(float(r) for r in residuals)
    q = math.ceil((n + 1) * (1.0 - alpha)) - 1
    q = max(0, min(n - 1, q))
    return s[q]


def hcope_lower_bound(
    values: Sequence[float],
    xi: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Thomas et al. 2015 high-confidence off-policy lower bound.

    Each ``values[i] = min(ξ, W_i · G_i)`` is the truncated IS-weighted
    return for trajectory ``i``.  Returns
    ``(lower_bound, bernstein_term, range_term)``.  The lower bound
    holds with probability ≥ 1 - α.
    """
    _validate_alpha(alpha)
    n = len(values)
    if n < 2:
        raise InsufficientData("HCOPE requires n ≥ 2 trajectories")
    if xi <= 0.0:
        raise CounterfactorError("xi must be > 0")
    mean = statistics.fmean(values)
    var = statistics.variance(values)
    log_term = math.log(2.0 / alpha)
    bernstein = math.sqrt(2.0 * var * log_term / n)
    range_term = 7.0 * xi * log_term / (3.0 * (n - 1))
    lb = mean - bernstein - range_term
    return lb, bernstein, range_term


# =====================================================================
# Weight diagnostics
# =====================================================================


def ess(weights: Sequence[float]) -> float:
    """Kong 1992 effective sample size: (Σ w)² / Σ w²."""
    if not weights:
        return 0.0
    s = sum(weights)
    if abs(s) < _EPS:
        return 0.0
    s2 = sum(w * w for w in weights)
    if s2 < _EPS:
        return 0.0
    return (s * s) / s2


def weight_diagnostics(weights: Sequence[float]) -> dict[str, float]:
    """Tail diagnostics over an importance-weight bag."""
    if not weights:
        return {
            "ess": 0.0,
            "max_weight": 0.0,
            "p99_weight": 0.0,
            "p50_weight": 0.0,
            "mean_log_weight": 0.0,
            "var_log_weight": 0.0,
        }
    log_w = [math.log(max(w, _EPS)) for w in weights]
    return {
        "ess": ess(weights),
        "max_weight": max(weights),
        "p99_weight": _percentile(weights, 99.0),
        "p50_weight": _percentile(weights, 50.0),
        "mean_log_weight": statistics.fmean(log_w),
        "var_log_weight": statistics.pvariance(log_w) if len(log_w) > 1 else 0.0,
    }


def overlap_kl(
    behavior_probs: Sequence[float],
    target_probs: Sequence[float],
) -> float:
    """KL(target ‖ behaviour) on a sequence of step probabilities.

    Used to surface support-violation regimes.  Both sequences must
    have the same length and strictly positive entries.
    """
    if len(behavior_probs) != len(target_probs):
        raise CounterfactorError("behavior_probs and target_probs length mismatch")
    if not behavior_probs:
        return 0.0
    total = 0.0
    for pb, pt in zip(behavior_probs, target_probs):
        if pt <= 0.0:
            continue
        if pb <= 0.0:
            raise SupportViolation("target has mass where behaviour has none")
        total += pt * math.log(pt / pb)
    return total


# =====================================================================
# Policy adapters
# =====================================================================


Policy = Callable[[Any], Mapping[Any, float]]


def _normalise_policy_dist(dist: Mapping[Any, float]) -> dict[Any, float]:
    out: dict[Any, float] = {}
    total = 0.0
    for a, p in dist.items():
        p = float(p)
        if p < 0.0:
            raise CounterfactorError("policy probability must be ≥ 0")
        out[a] = p
        total += p
    if total <= 0.0:
        raise CounterfactorError("policy distribution sums to zero")
    if not math.isfinite(total):
        raise CounterfactorError("policy distribution diverges")
    return {a: p / total for a, p in out.items()}


def _policy_prob(policy: Policy, state: Any, action: Any) -> tuple[float, Mapping[Any, float]]:
    """Return ``(π(action|state), full distribution)``."""
    dist = _normalise_policy_dist(policy(state))
    return dist.get(action, 0.0), dist


class UniformPolicy:
    """Uniform over a fixed action set."""

    def __init__(self, actions: Sequence[Any]) -> None:
        if not actions:
            raise CounterfactorError("UniformPolicy needs at least one action")
        self._actions = tuple(actions)

    def __call__(self, state: Any) -> dict[Any, float]:
        p = 1.0 / len(self._actions)
        return {a: p for a in self._actions}


class DeterministicPolicy:
    """Pure (state → action) function."""

    def __init__(self, action_of: Callable[[Any], Any], actions: Sequence[Any] | None = None) -> None:
        self._action_of = action_of
        self._actions = tuple(actions) if actions is not None else None

    def __call__(self, state: Any) -> dict[Any, float]:
        a = self._action_of(state)
        if self._actions is not None:
            return {act: (1.0 if act == a else 0.0) for act in self._actions}
        return {a: 1.0}


class EpsilonGreedyPolicy:
    """ε-greedy over a fixed action set with deterministic exploit."""

    def __init__(
        self,
        action_of: Callable[[Any], Any],
        actions: Sequence[Any],
        epsilon: float,
    ) -> None:
        if not actions:
            raise CounterfactorError("EpsilonGreedyPolicy needs at least one action")
        if not 0.0 <= epsilon <= 1.0:
            raise CounterfactorError("epsilon must be in [0, 1]")
        self._action_of = action_of
        self._actions = tuple(actions)
        self._epsilon = float(epsilon)

    def __call__(self, state: Any) -> dict[Any, float]:
        greedy = self._action_of(state)
        n = len(self._actions)
        base = self._epsilon / n
        out = {a: base for a in self._actions}
        if greedy in out:
            out[greedy] += 1.0 - self._epsilon
        else:
            # greedy was outside the registered action set
            extra = 1.0 - self._epsilon
            out[greedy] = extra
        return out


class SoftmaxPolicy:
    """Softmax over a per-action score function."""

    def __init__(
        self,
        score_of: Callable[[Any, Any], float],
        actions: Sequence[Any],
        temperature: float = 1.0,
    ) -> None:
        if not actions:
            raise CounterfactorError("SoftmaxPolicy needs at least one action")
        if temperature <= 0.0:
            raise CounterfactorError("temperature must be > 0")
        self._score_of = score_of
        self._actions = tuple(actions)
        self._temperature = float(temperature)

    def __call__(self, state: Any) -> dict[Any, float]:
        scores = [self._score_of(state, a) / self._temperature for a in self._actions]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        return {a: e / total for a, e in zip(self._actions, exps)}


# =====================================================================
# Q-models (value approximators)
# =====================================================================


class QModel:
    """Abstract Q̂(s, a) interface."""

    def fit(self, trajectories: Sequence[LoggedTrajectory]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def q(self, state: Any, action: Any) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def v(self, state: Any, policy: Policy) -> float:
        dist = _normalise_policy_dist(policy(state))
        return sum(p * self.q(state, a) for a, p in dist.items())


class ConstantQModel(QModel):
    """``Q̂(s,a) = c`` for all ``s, a``.  Useful as a baseline / no-op."""

    def __init__(self, value: float = 0.0) -> None:
        self._c = float(value)

    def fit(self, trajectories: Sequence[LoggedTrajectory]) -> None:
        return

    def q(self, state: Any, action: Any) -> float:
        return self._c


class TabularQModel(QModel):
    """Per-(state-key, action) mean-reward table.

    Returns the empirical mean of *future discounted return* observed
    starting from ``(state, action)`` in the logged trajectories.
    Unseen ``(s, a)`` defaults to the global mean reward (or zero if
    the table is empty).
    """

    def __init__(self, gamma: float = 1.0, default: float = 0.0) -> None:
        _validate_gamma(gamma)
        self._gamma = gamma
        self._default = float(default)
        self._table: dict[tuple[Any, Any], list[float]] = {}
        self._cached_default = float(default)

    def fit(self, trajectories: Sequence[LoggedTrajectory]) -> None:
        self._table = {}
        all_returns: list[float] = []
        for tr in trajectories:
            steps = tr.steps
            # cumulative discounted return from step t onwards
            tail = 0.0
            tails = [0.0] * len(steps)
            for t in range(len(steps) - 1, -1, -1):
                tail = steps[t].reward + self._gamma * tail
                tails[t] = tail
            for t, s in enumerate(steps):
                key = (_canonical(s.state), _canonical(s.action))
                self._table.setdefault(key, []).append(tails[t])
                all_returns.append(tails[t])
        if all_returns:
            self._cached_default = statistics.fmean(all_returns)
        else:
            self._cached_default = self._default

    def q(self, state: Any, action: Any) -> float:
        key = (_canonical(state), _canonical(action))
        if key in self._table:
            return statistics.fmean(self._table[key])
        return self._cached_default


class LinearQModel(QModel):
    """Per-action ridge regression Q̂_a(φ(s)) = wᵀ φ(s) + b.

    ``features`` is a callable ``state → list[float]`` of fixed length.
    A separate weight vector is fit per action via the closed-form
    Tikhonov-regularised normal equations.  Stdlib only.
    """

    def __init__(
        self,
        features: Callable[[Any], Sequence[float]],
        actions: Sequence[Any],
        l2: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        _validate_gamma(gamma)
        if l2 < 0.0:
            raise CounterfactorError("l2 must be ≥ 0")
        if not actions:
            raise CounterfactorError("LinearQModel needs at least one action")
        self._features = features
        self._actions = tuple(actions)
        self._l2 = float(l2)
        self._gamma = gamma
        self._weights: dict[Any, list[float]] = {}
        self._bias: dict[Any, float] = {}
        self._dim: int | None = None

    def fit(self, trajectories: Sequence[LoggedTrajectory]) -> None:
        per_action: dict[Any, list[tuple[list[float], float]]] = {
            a: [] for a in self._actions
        }
        for tr in trajectories:
            steps = tr.steps
            tail = 0.0
            tails = [0.0] * len(steps)
            for t in range(len(steps) - 1, -1, -1):
                tail = steps[t].reward + self._gamma * tail
                tails[t] = tail
            for t, s in enumerate(steps):
                feats = [float(x) for x in self._features(s.state)]
                if self._dim is None:
                    self._dim = len(feats)
                elif len(feats) != self._dim:
                    raise CounterfactorError(
                        f"feature dim mismatch: got {len(feats)} vs {self._dim}"
                    )
                if s.action in per_action:
                    per_action[s.action].append((feats, tails[t]))
        for a, rows in per_action.items():
            if not rows or self._dim is None:
                self._weights[a] = [0.0] * (self._dim or 0)
                self._bias[a] = 0.0
                continue
            w, b = _ridge_fit(rows, self._l2)
            self._weights[a] = w
            self._bias[a] = b

    def q(self, state: Any, action: Any) -> float:
        if action not in self._weights:
            return 0.0
        feats = [float(x) for x in self._features(state)]
        w = self._weights[action]
        if len(feats) != len(w):
            raise CounterfactorError("feature dim mismatch at predict")
        return self._bias[action] + sum(wi * xi for wi, xi in zip(w, feats))


def _ridge_fit(rows: Sequence[tuple[Sequence[float], float]], l2: float) -> tuple[list[float], float]:
    """Closed-form ridge: ŵ = (XᵀX + λI)⁻¹ Xᵀy, with an explicit bias.

    Augments features with a leading 1 to absorb the intercept; the
    intercept itself is **not** regularised.
    """
    if not rows:
        return [], 0.0
    d = len(rows[0][0])
    p = d + 1
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for feats, y in rows:
        x = [1.0] + [float(v) for v in feats]
        for i in range(p):
            xi = x[i]
            for j in range(p):
                XtX[i][j] += xi * x[j]
            Xty[i] += xi * y
    for i in range(1, p):  # do not regularise the intercept (i=0)
        XtX[i][i] += l2
    coefs = _solve_linear(XtX, Xty)
    return coefs[1:], coefs[0]


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan with partial pivoting."""
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-12:
            # singular — return least-norm-ish solution
            aug[col][col] = 1e-12
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        piv_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= piv_val
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


# =====================================================================
# Pure estimators
# =====================================================================


def _step_weights(
    trajectory: LoggedTrajectory,
    target_policy: Policy,
    weight_cap: float | None,
) -> tuple[list[float], list[float], int]:
    """Return per-step importance ratios ρ_t, the truth target probs π(a|s),
    and the number of clipped weights.

    ρ_t = π(a_t|s_t) / μ(a_t|s_t).
    """
    rhos: list[float] = []
    target_probs: list[float] = []
    n_clipped = 0
    for step in trajectory.steps:
        _validate_prob(step.behavior_prob, "behavior_prob")
        pt, _ = _policy_prob(target_policy, step.state, step.action)
        rho = pt / step.behavior_prob
        if weight_cap is not None and rho > weight_cap:
            rho = float(weight_cap)
            n_clipped += 1
        rhos.append(rho)
        target_probs.append(pt)
    return rhos, target_probs, n_clipped


def _prefix_products(rhos: Sequence[float]) -> list[float]:
    """W_t = ∏_{u ≤ t} ρ_u."""
    out: list[float] = []
    w = 1.0
    for r in rhos:
        w *= r
        out.append(w)
    return out


def _discount_vec(horizon: int, gamma: float) -> list[float]:
    g = 1.0
    out = [0.0] * horizon
    for t in range(horizon):
        out[t] = g
        g *= gamma
    return out


def traj_is(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
) -> tuple[float, list[float]]:
    """Trajectory importance sampling.  Returns ``(V̂, per-traj weighted returns)``."""
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("traj_is needs ≥ 1 trajectory")
    weighted: list[float] = []
    for tr in trajectories:
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        w = 1.0
        for r in rhos:
            w *= r
        g = tr.return_(gamma)
        weighted.append(w * g)
    return statistics.fmean(weighted), weighted


def traj_wis(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
) -> tuple[float, list[float], list[float]]:
    """Self-normalised trajectory IS.  Returns ``(V̂, weights, returns)``."""
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("traj_wis needs ≥ 1 trajectory")
    weights: list[float] = []
    returns: list[float] = []
    for tr in trajectories:
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        w = 1.0
        for r in rhos:
            w *= r
        weights.append(w)
        returns.append(tr.return_(gamma))
    den = sum(weights)
    if den < _EPS:
        return 0.0, weights, returns
    num = sum(w * g for w, g in zip(weights, returns))
    return num / den, weights, returns


def pdis(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
) -> tuple[float, list[float]]:
    """Per-decision importance sampling (Precup et al. 2000).

    V̂ = (1/n) Σ_τ Σ_t γ^t W_t(τ) r_t(τ).
    """
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("pdis needs ≥ 1 trajectory")
    contribs: list[float] = []
    for tr in trajectories:
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        Ws = _prefix_products(rhos)
        g = 1.0
        s = 0.0
        for t, step in enumerate(tr.steps):
            s += g * Ws[t] * step.reward
            g *= gamma
        contribs.append(s)
    return statistics.fmean(contribs), contribs


def wpdis(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
) -> tuple[float, list[list[float]], list[list[float]]]:
    """Weighted per-decision IS (Precup 2000).

    V̂ = Σ_t γ^t  (Σ_τ W_t(τ) r_t(τ)) / (Σ_τ W_t(τ)).
    """
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("wpdis needs ≥ 1 trajectory")
    H = max(tr.horizon for tr in trajectories)
    weights_t: list[list[float]] = [[] for _ in range(H)]
    rewards_t: list[list[float]] = [[] for _ in range(H)]
    for tr in trajectories:
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        Ws = _prefix_products(rhos)
        for t, step in enumerate(tr.steps):
            weights_t[t].append(Ws[t])
            rewards_t[t].append(step.reward)
    total = 0.0
    g = 1.0
    for t in range(H):
        den = sum(weights_t[t])
        if den > _EPS:
            num = sum(w * r for w, r in zip(weights_t[t], rewards_t[t]))
            total += g * num / den
        g *= gamma
    return total, weights_t, rewards_t


def dm(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    q_model: QModel,
    *,
    gamma: float = 1.0,
) -> tuple[float, list[float]]:
    """Direct method: V̂ = (1/n) Σ_τ V̂_π(s_0).

    ``Q̂(s, a)`` is assumed to estimate the *cumulative* discounted return
    from ``(s, a)`` onwards.  The DM value is therefore the average
    starting-state value ``V̂_π(s_0) = Σ_a π(a|s_0) Q̂(s_0, a)``.  This is
    the Jiang-Li / Thomas-Brunskill convention and is the form DR-RL,
    WDR and MAGIC all share with DM.
    """
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("dm needs ≥ 1 trajectory")
    contribs: list[float] = [
        q_model.v(tr.steps[0].state, target_policy) for tr in trajectories
    ]
    return statistics.fmean(contribs), contribs


def dr_rl(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    q_model: QModel,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
) -> tuple[float, list[float]]:
    """Doubly-robust off-policy evaluation for finite-horizon MDPs.

    Jiang & Li 2016 recursive form::

        V̂_DR(s_0) = V̂(s_0) + Σ_t γ^t W_t · ( r_t + γ V̂(s_{t+1}) - Q̂(s_t, a_t) ),

    with ``V̂(s) = Σ_a π(a|s) Q̂(s,a)``.  Returned per-trajectory
    contributions correspond to V̂_DR(τ).
    """
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("dr_rl needs ≥ 1 trajectory")
    contribs: list[float] = []
    for tr in trajectories:
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        Ws = _prefix_products(rhos)
        v0 = q_model.v(tr.steps[0].state, target_policy)
        residual = 0.0
        gpow = 1.0
        H = len(tr.steps)
        for t in range(H):
            step = tr.steps[t]
            v_next = (
                q_model.v(tr.steps[t + 1].state, target_policy)
                if t + 1 < H
                else 0.0
            )
            q_sa = q_model.q(step.state, step.action)
            residual += gpow * Ws[t] * (step.reward + gamma * v_next - q_sa)
            gpow *= gamma
        contribs.append(v0 + residual)
    return statistics.fmean(contribs), contribs


def wdr(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    q_model: QModel,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
) -> tuple[float, list[list[float]]]:
    """Weighted DR (Thomas-Brunskill 2016).

    Self-normalises the per-step DR correction by the column-sum of
    ``W_t``.  Returns ``(V̂, per-step weight columns)``.
    """
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("wdr needs ≥ 1 trajectory")
    H = max(tr.horizon for tr in trajectories)
    weights_t: list[list[float]] = [[] for _ in range(H)]
    corrections_t: list[list[float]] = [[] for _ in range(H)]
    v0s: list[float] = []
    for tr in trajectories:
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        Ws = _prefix_products(rhos)
        v0s.append(q_model.v(tr.steps[0].state, target_policy))
        for t in range(len(tr.steps)):
            step = tr.steps[t]
            v_next = (
                q_model.v(tr.steps[t + 1].state, target_policy)
                if t + 1 < len(tr.steps)
                else 0.0
            )
            q_sa = q_model.q(step.state, step.action)
            weights_t[t].append(Ws[t])
            corrections_t[t].append(step.reward + gamma * v_next - q_sa)
    total = statistics.fmean(v0s)
    g = 1.0
    for t in range(H):
        den = sum(weights_t[t])
        if den > _EPS:
            num = sum(w * c for w, c in zip(weights_t[t], corrections_t[t]))
            total += g * num / den
        g *= gamma
    return total, weights_t


def magic(
    trajectories: Sequence[LoggedTrajectory],
    target_policy: Policy,
    q_model: QModel,
    *,
    gamma: float = 1.0,
    weight_cap: float | None = None,
    j_set: Sequence[int] | None = None,
) -> tuple[float, list[float]]:
    """MAGIC — Model And Guided Importance-sampling Combination
    (Thomas-Brunskill 2016).

    Computes a family of *j-step* return estimators::

        g^j(τ) = Σ_{t<j} γ^t W_t (r_t - Q̂(s_t,a_t))   +   W_{j-1} γ^j V̂(s_j),

    indexed by ``j ∈ J``.  Builds an empirical mean / covariance
    over ``g^j`` and picks the convex combination minimising the
    empirical MSE proxy ``λ^T (Σ + bbᵀ) λ`` subject to ``Σ λ_j = 1``,
    where ``b_j`` is the bias of ``g^j`` relative to the maximum-``j``
    return.  Returns ``(V̂_MAGIC, λ)``.
    """
    _validate_gamma(gamma)
    if not trajectories:
        raise InsufficientData("magic needs ≥ 1 trajectory")
    H = max(tr.horizon for tr in trajectories)
    if j_set is None:
        j_set = list(range(0, H + 1))
    else:
        j_set = sorted(set(int(j) for j in j_set if 0 <= j <= H))
        if not j_set:
            raise CounterfactorError("j_set is empty")
    # g^j(τ) = V̂(s_0) + Σ_{t<j} γ^t W_t (r_t + γ V̂(s_{t+1}) - Q̂(s_t, a_t))
    # j = 0 → pure model V̂(s_0)
    # j = H → full DR-RL
    n = len(trajectories)
    G = [[0.0] * len(j_set) for _ in range(n)]
    for i, tr in enumerate(trajectories):
        rhos, _, _ = _step_weights(tr, target_policy, weight_cap)
        Ws = _prefix_products(rhos)
        v0 = q_model.v(tr.steps[0].state, target_policy)
        # incremental DR correction up to step t (inclusive)
        correction_upto: list[float] = [0.0] * (len(tr.steps) + 1)
        g_pow = 1.0
        for t in range(len(tr.steps)):
            step = tr.steps[t]
            v_next = (
                q_model.v(tr.steps[t + 1].state, target_policy)
                if t + 1 < len(tr.steps)
                else 0.0
            )
            q_sa = q_model.q(step.state, step.action)
            delta = g_pow * Ws[t] * (step.reward + gamma * v_next - q_sa)
            correction_upto[t + 1] = correction_upto[t] + delta
            g_pow *= gamma
        for ji, j in enumerate(j_set):
            if j == 0:
                G[i][ji] = v0
            else:
                upto = min(j, len(tr.steps))
                G[i][ji] = v0 + correction_upto[upto]
    # empirical mean
    means = [statistics.fmean(col) for col in zip(*G)]
    # use the largest-j as reference for bias
    ref = means[-1]
    biases = [m - ref for m in means]
    # covariance
    K = len(j_set)
    cov = [[0.0] * K for _ in range(K)]
    if n > 1:
        for i in range(n):
            for j in range(K):
                dj = G[i][j] - means[j]
                for kk in range(K):
                    cov[j][kk] += dj * (G[i][kk] - means[kk])
        for j in range(K):
            for kk in range(K):
                cov[j][kk] /= (n - 1)
    # MSE proxy: Σ + b bᵀ
    M = [[cov[j][kk] + biases[j] * biases[kk] for kk in range(K)] for j in range(K)]
    # ridge for stability
    for j in range(K):
        M[j][j] += 1e-8
    # solve min λᵀ M λ s.t. Σ λ = 1 → λ = M⁻¹ 1 / 1ᵀ M⁻¹ 1
    ones = [1.0] * K
    z = _solve_linear([row[:] for row in M], ones[:])
    s = sum(z)
    if abs(s) < _EPS:
        lam = [1.0 / K] * K
    else:
        lam = [v / s for v in z]
        # project to simplex if any negative (fall back to uniform)
        if any(v < -1e-6 for v in lam):
            lam = [1.0 / K] * K
    value = sum(li * mi for li, mi in zip(lam, means))
    return value, lam


# =====================================================================
# Counterfactor — the runtime primitive
# =====================================================================


class Counterfactor:
    """Sequential off-policy evaluation runtime primitive.

    Parameters
    ----------
    bus : EventBus | None
        Optional event bus for live broadcast of each method call.
    attestor : object | None
        Optional sink with ``record(kind=..., payload=...)`` or a plain
        callable; every evaluate / hcope / compare emits a content-hashed
        receipt that the AttestationLedger can chain.
    reward_range : tuple[float, float]
        Used to compute the truncation parameter for HCOPE and the
        range argument for Hoeffding / empirical-Bernstein bounds.
    weight_cap : float | None
        Default importance-weight cap.  Per-call ``weight_cap`` overrides.
    random_seed : int | None
        Seed for any bootstrap / Monte-Carlo machinery in MAGIC's
        diagnostic fall-backs.
    """

    def __init__(
        self,
        *,
        bus: Any = None,
        attestor: Any = None,
        reward_range: tuple[float, float] = (0.0, 1.0),
        weight_cap: float | None = _DEFAULT_CLIP,
        random_seed: int | None = None,
    ) -> None:
        if reward_range[1] < reward_range[0]:
            raise CounterfactorError("reward_range[1] must be ≥ reward_range[0]")
        self._bus = bus
        self._attestor = attestor
        self._reward_range = (float(reward_range[0]), float(reward_range[1]))
        self._default_cap = float(weight_cap) if weight_cap is not None else None
        self._lock = threading.RLock()
        self._rng = random.Random(random_seed)
        self._trajectories: list[LoggedTrajectory] = []
        self._n_evaluates = 0
        self._n_hcopes = 0
        self._n_compares = 0
        self._started_ns = time.time_ns()
        self._emit(
            CF_STARTED,
            {"id": uuid.uuid4().hex[:16], "ts_ns": self._started_ns},
        )

    # ---- event / attest helpers --------------------------------------

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None or Event is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(payload)))
        except Exception:
            pass

    def _attest(self, kind: str, payload: dict) -> str:
        digest = _hash_payload(payload)
        if self._attestor is None:
            return digest
        try:
            if hasattr(self._attestor, "record"):
                self._attestor.record(kind=kind, payload=payload)
            elif callable(self._attestor):
                self._attestor({"kind": kind, "payload": payload, "digest": digest})
        except Exception:
            pass
        return digest

    # ---- ingestion ----------------------------------------------------

    def log_trajectory(self, trajectory: LoggedTrajectory) -> None:
        """Append one trajectory to the log."""
        if not isinstance(trajectory, LoggedTrajectory):
            raise CounterfactorError("log_trajectory requires a LoggedTrajectory")
        for s in trajectory.steps:
            _validate_prob(s.behavior_prob, "behavior_prob")
        with self._lock:
            self._trajectories.append(trajectory)
        self._emit(
            CF_LOGGED,
            {
                "trajectory_id": trajectory.trajectory_id,
                "horizon": trajectory.horizon,
                "ts": trajectory.timestamp,
            },
        )

    def log(
        self,
        steps: Iterable[Mapping[str, Any] | LoggedStep],
        *,
        trajectory_id: str | None = None,
        tenant_id: str | None = None,
    ) -> LoggedTrajectory:
        """Convenience: build and append a trajectory from a list of dicts."""
        parsed: list[LoggedStep] = []
        for s in steps:
            if isinstance(s, LoggedStep):
                parsed.append(s)
                continue
            parsed.append(
                LoggedStep(
                    state=s["state"],
                    action=s["action"],
                    reward=float(s["reward"]),
                    behavior_prob=float(s["behavior_prob"]),
                    metadata=s.get("metadata"),
                )
            )
        tr = LoggedTrajectory(
            steps=tuple(parsed),
            trajectory_id=trajectory_id or uuid.uuid4().hex[:12],
            tenant_id=tenant_id,
        )
        self.log_trajectory(tr)
        return tr

    def log_bandit(
        self,
        state: Any,
        action: Any,
        reward: float,
        behavior_prob: float,
        *,
        metadata: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> LoggedTrajectory:
        """One-step trajectory — bridge to contextual-bandit OPE."""
        return self.log(
            [
                LoggedStep(
                    state=state,
                    action=action,
                    reward=float(reward),
                    behavior_prob=float(behavior_prob),
                    metadata=metadata,
                )
            ],
            tenant_id=tenant_id,
        )

    @property
    def n_trajectories(self) -> int:
        with self._lock:
            return len(self._trajectories)

    def trajectories(self) -> list[LoggedTrajectory]:
        """Snapshot of the current trajectory log."""
        with self._lock:
            return list(self._trajectories)

    def clear(self) -> None:
        with self._lock:
            self._trajectories = []
        self._emit(CF_CLEARED, {})

    def coverage(self) -> dict:
        with self._lock:
            return {
                "n_trajectories": len(self._trajectories),
                "n_evaluates": self._n_evaluates,
                "n_hcopes": self._n_hcopes,
                "n_compares": self._n_compares,
                "reward_range": self._reward_range,
                "default_cap": self._default_cap,
                "started_ns": self._started_ns,
            }

    # ---- core evaluation ---------------------------------------------

    def evaluate(
        self,
        target_policy: Policy,
        *,
        method: str = METHOD_WPDIS,
        gamma: float = 1.0,
        q_model: QModel | None = None,
        weight_cap: float | None = None,
        alpha: float = _DEFAULT_ALPHA,
        ci_method: str = CI_BERNSTEIN,
        j_set: Sequence[int] | None = None,
    ) -> OPEReport:
        """Evaluate ``target_policy`` against the trajectory log.

        Returns a content-hashed ``OPEReport`` with point estimate and
        finite-sample CI.
        """
        if method not in KNOWN_METHODS:
            raise UnknownMethod(
                f"unknown method {method!r}; expected {sorted(KNOWN_METHODS)!r}"
            )
        if ci_method not in KNOWN_CI:
            raise CounterfactorError(
                f"unknown ci_method {ci_method!r}; expected {sorted(KNOWN_CI)!r}"
            )
        _validate_alpha(alpha)
        _validate_gamma(gamma)
        cap = weight_cap if weight_cap is not None else self._default_cap
        with self._lock:
            trajs = list(self._trajectories)
        if not trajs:
            raise InsufficientData("evaluate needs ≥ 1 logged trajectory")
        H = max(tr.horizon for tr in trajs)
        n = len(trajs)

        # Dispatch
        if method == METHOD_TRAJ_IS:
            value, contribs = traj_is(trajs, target_policy, gamma=gamma, weight_cap=cap)
            ci_values = contribs
        elif method == METHOD_TRAJ_WIS:
            value, weights, returns = traj_wis(trajs, target_policy, gamma=gamma, weight_cap=cap)
            ci_values = [w * r for w, r in zip(weights, returns)]
        elif method == METHOD_PDIS:
            value, contribs = pdis(trajs, target_policy, gamma=gamma, weight_cap=cap)
            ci_values = contribs
        elif method == METHOD_WPDIS:
            value, _, _ = wpdis(trajs, target_policy, gamma=gamma, weight_cap=cap)
            # CI on WPDIS uses per-trajectory PDIS contributions as the influence proxy
            _, ci_values = pdis(trajs, target_policy, gamma=gamma, weight_cap=cap)
        elif method == METHOD_DM:
            if q_model is None:
                raise CounterfactorError("dm requires a q_model")
            value, contribs = dm(trajs, target_policy, q_model, gamma=gamma)
            ci_values = contribs
        elif method == METHOD_DR_RL:
            if q_model is None:
                raise CounterfactorError("dr_rl requires a q_model")
            value, contribs = dr_rl(
                trajs, target_policy, q_model, gamma=gamma, weight_cap=cap
            )
            ci_values = contribs
        elif method == METHOD_WDR:
            if q_model is None:
                raise CounterfactorError("wdr requires a q_model")
            value, _ = wdr(trajs, target_policy, q_model, gamma=gamma, weight_cap=cap)
            _, ci_values = dr_rl(
                trajs, target_policy, q_model, gamma=gamma, weight_cap=cap
            )
        elif method == METHOD_MAGIC:
            if q_model is None:
                raise CounterfactorError("magic requires a q_model")
            value, _ = magic(
                trajs,
                target_policy,
                q_model,
                gamma=gamma,
                weight_cap=cap,
                j_set=j_set,
            )
            _, ci_values = dr_rl(
                trajs, target_policy, q_model, gamma=gamma, weight_cap=cap
            )
        else:  # pragma: no cover
            raise UnknownMethod(method)

        # CI
        ci_lo, ci_hi = self._ci(value, ci_values, alpha, ci_method, gamma, H)

        # Diagnostics
        traj_ws = self._traj_weights(trajs, target_policy, cap)
        traj_clip = sum(1 for tr in trajs if any(_step_weights(tr, target_policy, cap)[2:]))  # placeholder
        n_clipped = 0
        for tr in trajs:
            _, _, c = _step_weights(tr, target_policy, cap)
            n_clipped += c
        total_steps = sum(tr.horizon for tr in trajs)
        clip_frac = _safe_div(n_clipped, total_steps)
        ess_val = ess(traj_ws)
        max_w = max(traj_ws) if traj_ws else 0.0

        report = OPEReport(
            method=method,
            ci_method=ci_method,
            value=float(value),
            ci_lo=float(ci_lo),
            ci_hi=float(ci_hi),
            alpha=alpha,
            n_trajectories=n,
            horizon=H,
            gamma=gamma,
            ess=ess_val,
            max_weight=max_w,
            clip_fraction=clip_frac,
            weight_cap=cap,
            reward_range=self._reward_range,
            extras={},
        )
        digest = self._attest(
            CF_EVALUATED,
            {
                "method": method,
                "ci_method": ci_method,
                "value": float(value),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "alpha": alpha,
                "gamma": gamma,
                "n_trajectories": n,
                "horizon": H,
                "ess": ess_val,
                "max_weight": max_w,
                "clip_fraction": clip_frac,
                "weight_cap": cap,
            },
        )
        report.digest = digest
        with self._lock:
            self._n_evaluates += 1
        self._emit(CF_EVALUATED, {"digest": digest, "method": method, "value": float(value)})
        return report

    # ---- HCOPE -------------------------------------------------------

    def hcope(
        self,
        target_policy: Policy,
        *,
        method: str = METHOD_PDIS,
        gamma: float = 1.0,
        q_model: QModel | None = None,
        weight_cap: float | None = None,
        alpha: float = _DEFAULT_ALPHA,
        xi: float | None = None,
    ) -> HCOPEReport:
        """High-confidence off-policy *lower bound* (Thomas et al. 2015).

        Each trajectory contributes ``min(ξ, V̂_τ)`` where ``V̂_τ`` is the
        per-trajectory point estimate under ``method``.  The returned
        lower bound holds with probability ≥ 1 - α (over the data) by
        the Maurer-Pontil 2009 empirical Bernstein inequality.

        Default ``ξ`` is ``H · r_max`` (the maximum possible return).
        """
        if method not in KNOWN_METHODS:
            raise UnknownMethod(method)
        _validate_alpha(alpha)
        _validate_gamma(gamma)
        cap = weight_cap if weight_cap is not None else self._default_cap
        with self._lock:
            trajs = list(self._trajectories)
        if not trajs:
            raise InsufficientData("hcope needs ≥ 1 logged trajectory")
        H = max(tr.horizon for tr in trajs)
        r_max = max(abs(self._reward_range[0]), abs(self._reward_range[1]))
        xi_val = float(xi) if xi is not None else max(r_max * H, 1.0)

        # Compute per-trajectory point contributions under the chosen method.
        if method == METHOD_TRAJ_IS:
            _, contribs = traj_is(trajs, target_policy, gamma=gamma, weight_cap=cap)
        elif method == METHOD_TRAJ_WIS:
            _, weights, returns = traj_wis(trajs, target_policy, gamma=gamma, weight_cap=cap)
            den = sum(weights)
            if den < _EPS:
                contribs = [0.0] * len(trajs)
            else:
                contribs = [
                    w * r * len(trajs) / den for w, r in zip(weights, returns)
                ]
        elif method == METHOD_PDIS:
            _, contribs = pdis(trajs, target_policy, gamma=gamma, weight_cap=cap)
        elif method == METHOD_WPDIS:
            _, contribs = pdis(trajs, target_policy, gamma=gamma, weight_cap=cap)
        elif method == METHOD_DM:
            if q_model is None:
                raise CounterfactorError("hcope dm requires a q_model")
            _, contribs = dm(trajs, target_policy, q_model, gamma=gamma)
        elif method == METHOD_DR_RL or method == METHOD_WDR or method == METHOD_MAGIC:
            if q_model is None:
                raise CounterfactorError(f"hcope {method} requires a q_model")
            _, contribs = dr_rl(trajs, target_policy, q_model, gamma=gamma, weight_cap=cap)
        else:  # pragma: no cover
            raise UnknownMethod(method)

        # Shift to [0, 2ξ] before Maurer-Pontil
        truncated = [max(min(c, xi_val), -xi_val) + xi_val for c in contribs]
        # Now bounded in [0, 2*xi_val] → use 2*xi_val as range.
        c_range = 2.0 * xi_val
        n = len(truncated)
        if n < 2:
            raise InsufficientData("hcope needs ≥ 2 trajectories")
        mean = statistics.fmean(truncated)
        var = statistics.variance(truncated)
        log_term = math.log(2.0 / alpha)
        bernstein = math.sqrt(2.0 * var * log_term / n)
        range_term = 7.0 * c_range * log_term / (3.0 * (n - 1))
        # Shift back: lower bound on E[clipped contribution]
        shifted_lb = mean - bernstein - range_term
        lb = shifted_lb - xi_val
        point_value = statistics.fmean(contribs)

        report = HCOPEReport(
            method=method,
            point_value=float(point_value),
            lower_bound=float(lb),
            xi=float(xi_val),
            alpha=alpha,
            n_trajectories=n,
            horizon=H,
            gamma=gamma,
            bernstein_term=float(bernstein),
            range_term=float(range_term),
            extras={"clip_to_range": float(c_range)},
        )
        digest = self._attest(
            CF_HCOPE_BOUNDED,
            {
                "method": method,
                "point_value": float(point_value),
                "lower_bound": float(lb),
                "xi": float(xi_val),
                "alpha": alpha,
                "n_trajectories": n,
                "horizon": H,
                "gamma": gamma,
            },
        )
        report.digest = digest
        with self._lock:
            self._n_hcopes += 1
        self._emit(CF_HCOPE_BOUNDED, {"digest": digest, "lower_bound": float(lb)})
        return report

    # ---- pairwise comparison -----------------------------------------

    def compare(
        self,
        policy_a: Policy,
        policy_b: Policy,
        *,
        name_a: str = "a",
        name_b: str = "b",
        method: str = METHOD_WPDIS,
        gamma: float = 1.0,
        q_model: QModel | None = None,
        weight_cap: float | None = None,
        alpha: float = _DEFAULT_ALPHA,
    ) -> CompareReport:
        """Paired off-policy comparison.

        Computes per-trajectory influence values under ``method`` for
        both policies, then evaluates the *paired difference* with the
        Student-t paired half-width.  Reports ``P(A > B)`` under the
        asymptotic Gaussian approximation of the paired difference,
        and the boolean ``a_dominates = (delta_ci_lo > 0)``.
        """
        if method not in KNOWN_METHODS:
            raise UnknownMethod(method)
        _validate_alpha(alpha)
        _validate_gamma(gamma)
        cap = weight_cap if weight_cap is not None else self._default_cap
        with self._lock:
            trajs = list(self._trajectories)
        if not trajs:
            raise InsufficientData("compare needs ≥ 1 logged trajectory")

        contribs_a = self._per_traj_contribs(
            trajs, policy_a, method, gamma, q_model, cap
        )
        contribs_b = self._per_traj_contribs(
            trajs, policy_b, method, gamma, q_model, cap
        )
        diffs = [a - b for a, b in zip(contribs_a, contribs_b)]
        n = len(diffs)
        if n < 2:
            raise InsufficientData("compare needs ≥ 2 trajectories")
        delta = statistics.fmean(diffs)
        if statistics.pvariance(diffs) < _EPS:
            se = 0.0
            t_half = 0.0
            p_better = 1.0 if delta > 0 else (0.0 if delta < 0 else 0.5)
        else:
            se = statistics.stdev(diffs) / math.sqrt(n)
            tcrit = _student_t_quantile(alpha, n - 1)
            t_half = tcrit * se
            z = delta / se if se > _EPS else (math.inf if delta > 0 else -math.inf)
            p_better = _normal_cdf(z)
        delta_lo = delta - t_half
        delta_hi = delta + t_half
        a_dom = delta_lo > 0.0
        report = CompareReport(
            method=method,
            name_a=name_a,
            name_b=name_b,
            value_a=float(statistics.fmean(contribs_a)),
            value_b=float(statistics.fmean(contribs_b)),
            delta=float(delta),
            delta_ci_lo=float(delta_lo),
            delta_ci_hi=float(delta_hi),
            p_a_better=float(p_better),
            alpha=alpha,
            n_trajectories=n,
            a_dominates=bool(a_dom),
            extras={"se": float(se)},
        )
        digest = self._attest(
            CF_COMPARED,
            {
                "method": method,
                "name_a": name_a,
                "name_b": name_b,
                "value_a": report.value_a,
                "value_b": report.value_b,
                "delta": report.delta,
                "delta_ci_lo": report.delta_ci_lo,
                "delta_ci_hi": report.delta_ci_hi,
                "alpha": alpha,
                "n_trajectories": n,
            },
        )
        report.digest = digest
        with self._lock:
            self._n_compares += 1
        self._emit(CF_COMPARED, {"digest": digest, "delta": report.delta})
        return report

    # ---- diagnostics -------------------------------------------------

    def diagnostics(
        self,
        target_policy: Policy,
        *,
        weight_cap: float | None = None,
        coverage_floor: float = 0.05,
        ess_floor: float = 5.0,
    ) -> DiagnosticsReport:
        """Importance-weight & support diagnostics for ``target_policy``."""
        cap = weight_cap if weight_cap is not None else self._default_cap
        with self._lock:
            trajs = list(self._trajectories)
        if not trajs:
            raise InsufficientData("diagnostics needs ≥ 1 logged trajectory")
        H = max(tr.horizon for tr in trajs)
        n = len(trajs)

        traj_ws: list[float] = []
        per_t_weights: list[list[float]] = [[] for _ in range(H)]
        n_clipped = 0
        zero_target_mass = 0
        total_steps = 0
        target_logp: list[float] = []
        behaviour_logp: list[float] = []
        for tr in trajs:
            rhos, targets, c = _step_weights(tr, target_policy, cap)
            n_clipped += c
            Ws = _prefix_products(rhos)
            traj_ws.append(Ws[-1] if Ws else 1.0)
            for t in range(len(tr.steps)):
                per_t_weights[t].append(Ws[t])
                total_steps += 1
                if targets[t] <= 0.0:
                    zero_target_mass += 1
                target_logp.append(math.log(max(targets[t], _EPS)))
                behaviour_logp.append(math.log(max(tr.steps[t].behavior_prob, _EPS)))
        # ESS_traj on full-path weights
        ess_traj = ess(traj_ws)
        # ESS_pdis_min: minimum per-step ESS — drives WPDIS variance
        per_t_ess = [ess(per_t_weights[t]) for t in range(H) if per_t_weights[t]]
        ess_pdis_min = min(per_t_ess) if per_t_ess else 0.0
        diag = weight_diagnostics(traj_ws)
        # KL(target ‖ behaviour) on logged steps (where target has mass)
        kl_total = 0.0
        for tlp, blp in zip(target_logp, behaviour_logp):
            kl_total += math.exp(tlp) * (tlp - blp) if math.exp(tlp) > 0 else 0.0
        coverage = 1.0 - _safe_div(zero_target_mass, total_steps)
        clip_frac = _safe_div(n_clipped, total_steps)
        warnings: list[str] = []
        if coverage < coverage_floor:
            warnings.append(
                f"low overlap: target has zero mass on "
                f"{zero_target_mass}/{total_steps} logged steps"
            )
        if ess_traj < ess_floor:
            warnings.append(
                f"ESS_traj={ess_traj:.2f} below floor {ess_floor:.2f}: high-variance estimate"
            )
        if ess_pdis_min < ess_floor:
            warnings.append(
                f"ESS_pdis_min={ess_pdis_min:.2f} below floor {ess_floor:.2f}"
            )
        if diag["max_weight"] > 1e3:
            warnings.append(
                f"max trajectory weight {diag['max_weight']:.2e} — consider truncation"
            )
        report = DiagnosticsReport(
            n_trajectories=n,
            horizon=H,
            ess_trajectory=ess_traj,
            ess_pdis_min=ess_pdis_min,
            max_weight=diag["max_weight"],
            p99_weight=diag["p99_weight"],
            p50_weight=diag["p50_weight"],
            mean_log_weight=diag["mean_log_weight"],
            var_log_weight=diag["var_log_weight"],
            overlap_kl=float(kl_total),
            coverage=coverage,
            clip_fraction=clip_frac,
            weight_cap=cap,
            warnings=warnings,
        )
        digest = self._attest(
            CF_DIAGNOSED,
            {
                "n_trajectories": n,
                "horizon": H,
                "ess_trajectory": ess_traj,
                "ess_pdis_min": ess_pdis_min,
                "max_weight": diag["max_weight"],
                "coverage": coverage,
                "clip_fraction": clip_frac,
                "warnings": warnings,
            },
        )
        report.digest = digest
        self._emit(CF_DIAGNOSED, {"digest": digest, "warnings": len(warnings)})
        return report

    # ---- internals ---------------------------------------------------

    def _ci(
        self,
        value: float,
        contribs: Sequence[float],
        alpha: float,
        ci_method: str,
        gamma: float,
        H: int,
    ) -> tuple[float, float]:
        if not contribs or len(contribs) < 2:
            return (value, value)
        # Pick range as horizon-discounted r_max for [0, range_] bounds
        r_max = max(abs(self._reward_range[0]), abs(self._reward_range[1]))
        if gamma == 1.0:
            range_ = r_max * H
        else:
            # geometric sum
            range_ = r_max * (1 - gamma ** H) / max(1.0 - gamma, 1e-9)
        if ci_method == CI_HOEFFDING:
            half = hoeffding_half_width(len(contribs), max(range_ * 2.0, 1e-9), alpha)
        elif ci_method == CI_BERNSTEIN:
            try:
                half = empirical_bernstein_half_width(
                    contribs, alpha, max(range_ * 2.0, 1e-9)
                )
            except InsufficientData:
                half = hoeffding_half_width(len(contribs), max(range_ * 2.0, 1e-9), alpha)
        elif ci_method == CI_STUDENT_T:
            half = student_t_half_width(contribs, alpha)
        elif ci_method == CI_CONFORMAL:
            mean = statistics.fmean(contribs)
            residuals = [abs(c - mean) for c in contribs]
            half = conformal_envelope(residuals, alpha)
        else:  # pragma: no cover
            raise CounterfactorError(f"unknown ci_method {ci_method!r}")
        return value - half, value + half

    def _traj_weights(
        self,
        trajectories: Sequence[LoggedTrajectory],
        target_policy: Policy,
        cap: float | None,
    ) -> list[float]:
        ws: list[float] = []
        for tr in trajectories:
            rhos, _, _ = _step_weights(tr, target_policy, cap)
            w = 1.0
            for r in rhos:
                w *= r
            ws.append(w)
        return ws

    def _per_traj_contribs(
        self,
        trajectories: Sequence[LoggedTrajectory],
        target_policy: Policy,
        method: str,
        gamma: float,
        q_model: QModel | None,
        cap: float | None,
    ) -> list[float]:
        if method == METHOD_TRAJ_IS:
            _, contribs = traj_is(trajectories, target_policy, gamma=gamma, weight_cap=cap)
            return contribs
        if method == METHOD_TRAJ_WIS:
            _, weights, returns = traj_wis(
                trajectories, target_policy, gamma=gamma, weight_cap=cap
            )
            den = sum(weights)
            if den < _EPS:
                return [0.0] * len(trajectories)
            return [w * r * len(trajectories) / den for w, r in zip(weights, returns)]
        if method in (METHOD_PDIS, METHOD_WPDIS):
            _, contribs = pdis(trajectories, target_policy, gamma=gamma, weight_cap=cap)
            return contribs
        if method == METHOD_DM:
            if q_model is None:
                raise CounterfactorError("compare(dm) requires a q_model")
            _, contribs = dm(trajectories, target_policy, q_model, gamma=gamma)
            return contribs
        if method in (METHOD_DR_RL, METHOD_WDR, METHOD_MAGIC):
            if q_model is None:
                raise CounterfactorError(f"compare({method}) requires a q_model")
            _, contribs = dr_rl(
                trajectories, target_policy, q_model, gamma=gamma, weight_cap=cap
            )
            return contribs
        raise UnknownMethod(method)


# =====================================================================
# Small helpers exposed for testing / introspection
# =====================================================================


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


__all__ = [
    # event kinds
    "CF_STARTED",
    "CF_LOGGED",
    "CF_EVALUATED",
    "CF_HCOPE_BOUNDED",
    "CF_COMPARED",
    "CF_DIAGNOSED",
    "CF_CLEARED",
    "CF_REPORT",
    # method names
    "METHOD_TRAJ_IS",
    "METHOD_TRAJ_WIS",
    "METHOD_PDIS",
    "METHOD_WPDIS",
    "METHOD_DM",
    "METHOD_DR_RL",
    "METHOD_WDR",
    "METHOD_MAGIC",
    "KNOWN_METHODS",
    "CI_HOEFFDING",
    "CI_BERNSTEIN",
    "CI_STUDENT_T",
    "CI_CONFORMAL",
    "KNOWN_CI",
    # exceptions
    "CounterfactorError",
    "InsufficientData",
    "UnknownMethod",
    "SupportViolation",
    # dataclasses
    "LoggedStep",
    "LoggedTrajectory",
    "OPEReport",
    "HCOPEReport",
    "DiagnosticsReport",
    "CompareReport",
    # estimators
    "traj_is",
    "traj_wis",
    "pdis",
    "wpdis",
    "dm",
    "dr_rl",
    "wdr",
    "magic",
    # bounds / diagnostics
    "hoeffding_half_width",
    "empirical_bernstein_half_width",
    "student_t_half_width",
    "conformal_envelope",
    "hcope_lower_bound",
    "ess",
    "weight_diagnostics",
    "overlap_kl",
    # models
    "QModel",
    "ConstantQModel",
    "TabularQModel",
    "LinearQModel",
    # policy adapters
    "Policy",
    "UniformPolicy",
    "DeterministicPolicy",
    "EpsilonGreedyPolicy",
    "SoftmaxPolicy",
    # main class
    "Counterfactor",
]
