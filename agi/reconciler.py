r"""Reconciler — Aumann agreement as a runtime primitive.

Every primitive in this runtime that emits a posterior — ``Bandit``
emits a posterior over arms, ``BayesOpt`` emits a posterior over the
optimum location, ``Imaginator`` emits a posterior over future returns,
``Forecaster`` emits a posterior over the next observation, ``Mentalist``
emits a posterior over a counterparty's utility — eventually produces
output that the coordination engine has to *combine* with the output of
some other primitive that estimates a related quantity.  Without a
principled aggregation step the coordinator picks one (loses the
information in the others), or averages naively (loses the calibration
of the most-certain one), or hand-tunes a weighted vote (loses any
guarantee of optimality).

``Reconciler`` is the runtime's *bounded, anytime, certified, stdlib*
version of that aggregation.  It implements three pooling rules that
together cover the entire literature on consensus belief aggregation:

  * **Linear opinion pool** (Stone 1961; Genest-McConway 1990).
    ``q(·) = Σ_i w_i p_i(·)`` — equivalent to averaging the probability
    mass functions.  Robust to outliers, externally Bayesian.
  * **Logarithmic opinion pool** (Bordley 1982; Genest-Zidek 1986).
    ``q(·) ∝ Π_i p_i(·)^{w_i}`` — the geometric mean, normalised.
    Equivalent to averaging the *log-probabilities*.  Maximally
    informative when the experts have independent information.
  * **Aumann iteration** (Aumann 1976 *Agreeing to Disagree*;
    Geanakoplos-Polemarchakis 1982 *We Can't Disagree Forever*).
    Two (or more) Bayesians share their posterior over an event
    repeatedly; each round each agent updates on the posterior of the
    others.  In finite state spaces the iteration provably reaches
    consensus in finitely many rounds, and *the consensus is the
    posterior of a common-prior Bayesian conditioned on the join of
    every agent's information.*

The pitch reduced to a runtime call::

    rec = Reconciler(ReconcilerConfig(method="aumann"))
    rec.register_topic("arm_a_wins", outcomes=("yes", "no"))
    rec.contribute("arm_a_wins", source="bandit",   belief={"yes": 0.70, "no": 0.30})
    rec.contribute("arm_a_wins", source="bayesopt", belief={"yes": 0.60, "no": 0.40})
    rec.contribute("arm_a_wins", source="psrl",     belief={"yes": 0.65, "no": 0.35})

    report = rec.consensus("arm_a_wins")
    print(report.consensus)            # {"yes": 0.65, "no": 0.35}
    print(report.outlier)              # ("bayesopt", 0.025)  # KL gap
    print(report.converged, report.rounds)  # True, 3
    print(report.confidence_interval)  # HRMS anytime-valid on the mean

What this primitive ships
-------------------------

  * **Three aggregation methods (stdlib, no NumPy):**

    * ``"linear"`` — weighted average of probability mass functions.
      Genest-McConway 1990: this is the *only* externally-Bayesian
      pool when the weights do not depend on the experts' beliefs.
      Closed-form, O(K · |Ω|).

    * ``"logarithmic"`` — weighted geometric mean of pmfs, normalised.
      Bordley 1982: equivalent to combining the *log-likelihoods*;
      the maximum-entropy combination subject to matching the
      experts' KL-projections of the consensus onto each pmf.  Closed
      form, O(K · |Ω|).

    * ``"aumann"`` — iterative Bayesian agreement.  Each round, each
      expert broadcasts the indicator of their posterior set, the
      other experts update on the new information, and the procedure
      repeats until either the posteriors coincide or the
      configurable round-cap is hit.  Geanakoplos-Polemarchakis 1982
      guarantee finite-time convergence on finite state spaces; we
      return ``converged=False`` plus the closest-to-consensus
      KL-barycenter when the cap fires.

    * ``"kl_barycenter"`` — fixed-point iteration that returns the
      distribution minimising ``Σ_i w_i · KL(q ‖ p_i)`` (right-KL
      barycenter).  The closed-form solution coincides with the
      logarithmic pool; the iterative form is exposed for the
      Bregman-barycenter literature (Bregman 1967; Cuturi-Doucet
      2014).

  * **Outlier detection** — under the consensus, the
    expert-to-consensus KL gap ``KL(p_i ‖ q)`` quantifies how
    surprising each expert's belief looks; the largest gap names the
    *outlier* the coordinator should investigate (drift, model
    misspecification, adversarial input).

  * **PIT calibration per expert** — given a stream of (predicted,
    realised) pairs per source, the probability integral transform of
    the realised outcome under the predicted distribution is uniform
    on ``[0, 1]`` under a correct model; ``calibration_pvalue`` runs
    a one-sample KS test (Massey 1951) and returns the p-value.

  * **Anytime-valid confidence sequences** on the consensus mean.
    Howard-Ramdas-McAuliffe-Sekhon 2021 closed-form practical bound
    — a coordinator can keep adding contributions and read the same
    interval without paying a union-bound tax.

  * **Identifiability report** — flags topics where the contributors
    do not span the outcome space (every source assigns zero mass to
    some outcome ⇒ the consensus cannot distinguish that outcome
    from a zero-mass alternative), and reports the *effective number
    of independent experts* via the inverse Herfindahl-Hirschman
    index on the normalised weights.

  * **Tamper-evident SHA-256 fingerprint chain** (genesis seed
    ``"agi.reconciler.v1\x00" + secret_key``) with optional
    HMAC-SHA-256 over every ``register / contribute / consensus``
    event so ``AttestationLedger`` replays the consensus byte-for-byte.

  * **Pure stdlib.**  No NumPy, no Torch, no SciPy.

Mathematical and algorithmic roots
----------------------------------

  * **Aumann, R. (1976).**  *Agreements on Agreed.*  Annals of
    Statistics 4(6):1236-1239.  Two Bayesians with common-knowledge
    posteriors *must* agree.  The Aumann iteration in this primitive
    is the constructive version Geanakoplos-Polemarchakis 1982 used
    to prove the result.

  * **Geanakoplos, J. & Polemarchakis, H. (1982).** *We Can't
    Disagree Forever.* Journal of Economic Theory 28(1):192-200.
    Finite-time convergence of the Aumann iteration on finite state
    spaces — exactly the cap behaviour ``Reconciler`` exposes.

  * **Stone, M. (1961).** *The Opinion Pool.* Annals of Mathematical
    Statistics 32(4):1339-1342.  The linear opinion pool.

  * **Bordley, R. F. (1982).**  *A Multiplicative Formula for
    Aggregating Probability Assessments.*  Management Science
    28(10):1137-1148.  The logarithmic opinion pool.

  * **Genest, C. & Zidek, J. V. (1986).** *Combining Probability
    Distributions: A Critique and an Annotated Bibliography.*
    Statistical Science 1(1):114-135.  The reference review for
    pooling rules; the externally-Bayesian characterisation of
    linear pools and the consistency characterisation of
    logarithmic pools.

  * **Bregman, L. M. (1967).**  *The Relaxation Method of Finding
    the Common Point of Convex Sets.*  USSR Computational
    Mathematics and Mathematical Physics 7(3):200-217.  The
    barycenter minimising the sum of KL-divergences.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. S.
    (2021).** *Time-uniform, nonparametric, nonasymptotic confidence
    sequences.*  Anytime-valid CI on the consensus mean.

  * **Maurer, A. & Pontil, M. (2009).**  *Empirical Bernstein bounds
    and sample-variance penalization.*  Closed-form
    empirical-Bernstein LCB on consensus stability.

  * **Massey, F. J. (1951).** *The Kolmogorov-Smirnov Test for
    Goodness of Fit.*  Per-source calibration test.

Composes with the rest of the runtime
-------------------------------------

  * ``Bandit`` / ``BayesOpt`` / ``Imaginator`` / ``Forecaster`` /
    ``Predictor`` — every primitive that emits a posterior contributes
    a single ``Source`` to a Reconciler topic; the consensus is the
    coordinator's belief.
  * ``Auditor`` — Reconciler's per-source outlier KL is a candidate
    test statistic; Auditor controls the family-wise FDR.
  * ``DriftSentinel`` — running consensus stability is a
    martingale-difference under common knowledge; CUSUM flags
    contributor drift.
  * ``Aligner`` — preferences over (topic, consensus) pairs become
    training data for the system's reward model.
  * ``Mentalist`` — supplies the rationality posterior the
    coordinator should use to *weight* each Mentalist-modelled
    counterparty's contribution.
  * ``Conformal`` — wraps the consensus pmf with a finite-sample
    prediction set.
  * ``AttestationLedger`` — every register / contribute / consensus
    event chain-hashes.
  * ``Coordinator`` — every Goal whose execution depends on more
    than one primitive's posterior routes through Reconciler.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    # Event kinds
    "RECONCILER_STARTED",
    "RECONCILER_TOPIC_REGISTERED",
    "RECONCILER_TOPIC_REMOVED",
    "RECONCILER_CONTRIBUTED",
    "RECONCILER_CONSENSUS",
    "RECONCILER_CALIBRATED",
    "RECONCILER_CLEARED",
    # Method names
    "METHOD_LINEAR",
    "METHOD_LOGARITHMIC",
    "METHOD_AUMANN",
    "METHOD_KL_BARYCENTER",
    "KNOWN_METHODS",
    # Errors
    "ReconcilerError",
    "InvalidConfig",
    "InvalidTopic",
    "InvalidBelief",
    "InsufficientData",
    "UnknownTopic",
    # Dataclasses
    "ReconcilerConfig",
    "TopicSpec",
    "Source",
    "ConsensusReport",
    "CalibrationReport",
    "IdentifiabilityReport",
    # Main class
    "Reconciler",
    # Helper functions
    "linear_pool",
    "logarithmic_pool",
    "kl_divergence",
    "kl_barycenter",
    "aumann_iterate",
    "ledger_root",
    "hrms_half_width",
    "empirical_bernstein_half_width",
    "ks_pvalue",
    "effective_number_of_experts",
]

# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

RECONCILER_STARTED = "reconciler.started"
RECONCILER_TOPIC_REGISTERED = "reconciler.topic_registered"
RECONCILER_TOPIC_REMOVED = "reconciler.topic_removed"
RECONCILER_CONTRIBUTED = "reconciler.contributed"
RECONCILER_CONSENSUS = "reconciler.consensus"
RECONCILER_CALIBRATED = "reconciler.calibrated"
RECONCILER_CLEARED = "reconciler.cleared"

# ---------------------------------------------------------------------------
# Method names
# ---------------------------------------------------------------------------

METHOD_LINEAR = "linear"
METHOD_LOGARITHMIC = "logarithmic"
METHOD_AUMANN = "aumann"
METHOD_KL_BARYCENTER = "kl_barycenter"
KNOWN_METHODS: frozenset[str] = frozenset(
    {METHOD_LINEAR, METHOD_LOGARITHMIC, METHOD_AUMANN, METHOD_KL_BARYCENTER}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReconcilerError(Exception):
    """Base class for Reconciler errors."""


class InvalidConfig(ReconcilerError):
    """The supplied ``ReconcilerConfig`` is invalid."""


class InvalidTopic(ReconcilerError):
    """The supplied topic spec is invalid."""


class InvalidBelief(ReconcilerError):
    """The supplied belief is not a valid probability mass function."""


class InsufficientData(ReconcilerError):
    """Not enough contributions to satisfy the request."""


class UnknownTopic(ReconcilerError):
    """The requested topic has not been registered."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReconcilerConfig:
    """Top-level configuration.

    Attributes
    ----------
    method
        Default pooling method.  One of ``"linear"``,
        ``"logarithmic"``, ``"aumann"``, ``"kl_barycenter"``.
        Override per-call at :meth:`Reconciler.consensus`.
    confidence
        Confidence level for every interval the primitive returns
        (Hoeffding LCB / UCB, Maurer-Pontil empirical-Bernstein,
        Howard-Ramdas-McAuliffe-Sekhon anytime-valid sequences).
        Defaults to ``0.95``.
    aumann_max_rounds
        Maximum number of rounds for Aumann iteration.  When the
        cap fires before consensus, the closest-to-consensus
        KL-barycenter is returned with ``converged=False``.
    aumann_tol
        Convergence tolerance: total-variation distance between any
        two experts.  Default ``1e-6``.
    smoothing
        Laplace smoothing added to every outcome of every contributed
        belief before normalisation.  Keeps logarithmic pools well-
        defined under sparse beliefs.
    rng_seed
        Seed for the internal RNG used by tie-breaking and any
        Monte-Carlo subroutines.
    hmac_key
        Optional secret key for HMAC-SHA-256 over every fingerprint
        entry.
    """

    method: str = METHOD_LINEAR
    confidence: float = 0.95
    aumann_max_rounds: int = 32
    aumann_tol: float = 1e-6
    smoothing: float = 1e-12
    rng_seed: int | None = 0xA61BEEF
    hmac_key: bytes | None = None

    def __post_init__(self) -> None:
        if self.method not in KNOWN_METHODS:
            raise InvalidConfig(
                f"unknown method {self.method!r}; expected one of {sorted(KNOWN_METHODS)}"
            )
        if not 0.5 < self.confidence < 1.0:
            raise InvalidConfig(
                f"confidence must be in (0.5, 1.0); got {self.confidence}"
            )
        if self.aumann_max_rounds < 1:
            raise InvalidConfig("aumann_max_rounds must be ≥ 1")
        if self.aumann_tol <= 0:
            raise InvalidConfig("aumann_tol must be positive")
        if self.smoothing < 0:
            raise InvalidConfig("smoothing must be non-negative")
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes")


@dataclass(frozen=True)
class TopicSpec:
    """A discrete-outcome topic.

    ``outcomes`` enumerates the support of the consensus distribution.
    """

    topic: str
    outcomes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.topic:
            raise InvalidTopic("topic must be non-empty")
        if len(self.outcomes) < 2:
            raise InvalidTopic("topic must have ≥ 2 outcomes")
        if len(set(self.outcomes)) != len(self.outcomes):
            raise InvalidTopic("outcomes must be unique")


@dataclass(frozen=True)
class Source:
    """A single expert's contribution to a topic.

    ``weight`` defaults to ``1.0``.  Normalised across sources at
    consensus time.
    """

    source_id: str
    belief: dict[str, float]
    weight: float = 1.0
    realised: str | None = None  # for calibration
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ConsensusReport:
    """Output of :meth:`Reconciler.consensus`.

    Attributes
    ----------
    topic
        Topic name.
    method
        Pooling method actually used.
    consensus
        The consensus pmf over outcomes.
    confidence_interval
        For each outcome, the (HRMS-LCB, HRMS-UCB) anytime-valid
        interval at the configured confidence.
    outlier
        ``(source_id, kl_gap)`` of the source with the largest
        ``KL(p_i ‖ q)``; ``None`` if no contributions.
    per_source_kl
        ``KL(p_i ‖ q)`` for each source.
    converged
        Whether Aumann iteration converged before the round cap (only
        meaningful for the Aumann method; always ``True`` for the
        closed-form pools).
    rounds
        Number of Aumann iteration rounds (0 for the closed-form pools).
    effective_n_sources
        Inverse-Herfindahl-Hirschman index on normalised weights:
        ``(Σ w_i)² / Σ w_i²``.  In ``[1, K]`` and equals the count of
        equal-weight experts.
    fingerprint_hash
        Chain head after the consensus call.
    """

    topic: str
    method: str
    consensus: dict[str, float]
    confidence_interval: dict[str, tuple[float, float]]
    outlier: tuple[str, float] | None
    per_source_kl: dict[str, float]
    converged: bool
    rounds: int
    effective_n_sources: float
    fingerprint_hash: str


@dataclass(frozen=True)
class CalibrationReport:
    """Per-source PIT calibration test result."""

    topic: str
    source_id: str
    n_observations: int
    ks_statistic: float
    p_value: float
    log_loss: float
    fingerprint_hash: str


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Identifiability of the consensus over a topic.

    ``zero_mass_outcomes`` is the list of outcomes assigned zero mass
    by every source — the consensus cannot distinguish them from a
    zero-mass alternative.

    ``effective_n_sources`` is the inverse-Herfindahl-Hirschman on
    normalised weights; values near 1 indicate one source dominates.
    """

    topic: str
    n_sources: int
    zero_mass_outcomes: list[str]
    effective_n_sources: float
    fingerprint_hash: str


# ---------------------------------------------------------------------------
# Public math helpers (re-exported)
# ---------------------------------------------------------------------------


def _normalise(p: Sequence[float], smoothing: float = 0.0) -> list[float]:
    smoothed = [max(x, 0.0) + smoothing for x in p]
    s = sum(smoothed)
    if s <= 0.0:
        n = len(smoothed)
        return [1.0 / n] * n if n else []
    return [x / s for x in smoothed]


def _as_vector(belief: dict[str, float], outcomes: Sequence[str]) -> list[float]:
    return [float(belief.get(o, 0.0)) for o in outcomes]


def _validate_pmf(belief: dict[str, float], outcomes: Sequence[str]) -> None:
    for k, v in belief.items():
        if not isinstance(v, (int, float)):
            raise InvalidBelief(f"belief[{k!r}] must be numeric; got {type(v).__name__}")
        if math.isnan(v) or math.isinf(v):
            raise InvalidBelief(f"belief[{k!r}] must be finite")
        if v < 0:
            raise InvalidBelief(f"belief[{k!r}] must be non-negative")
    for k in belief:
        if k not in outcomes:
            raise InvalidBelief(f"belief contains unknown outcome {k!r}")


def linear_pool(
    pmfs: Sequence[Sequence[float]], weights: Sequence[float] | None = None
) -> list[float]:
    """Stone 1961 linear opinion pool.

    Returns the weighted average ``q(·) = Σ_i w_i p_i(·)``.  Weights
    are normalised to sum to 1.
    """
    K = len(pmfs)
    if K == 0:
        return []
    if weights is None:
        weights = [1.0] * K
    if len(weights) != K:
        raise ReconcilerError("weights length must match pmfs")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ReconcilerError("weights must sum to a positive number")
    w_norm = [w / w_sum for w in weights]
    n = len(pmfs[0])
    if any(len(p) != n for p in pmfs):
        raise ReconcilerError("all pmfs must share length")
    return [sum(w_norm[i] * pmfs[i][j] for i in range(K)) for j in range(n)]


def logarithmic_pool(
    pmfs: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    smoothing: float = 1e-12,
) -> list[float]:
    """Bordley 1982 logarithmic opinion pool.

    Returns the normalised weighted geometric mean
    ``q(·) ∝ Π_i p_i(·)^{w_i}``.  Equivalent to averaging log-
    probabilities.
    """
    K = len(pmfs)
    if K == 0:
        return []
    if weights is None:
        weights = [1.0] * K
    if len(weights) != K:
        raise ReconcilerError("weights length must match pmfs")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ReconcilerError("weights must sum to a positive number")
    w_norm = [w / w_sum for w in weights]
    n = len(pmfs[0])
    if any(len(p) != n for p in pmfs):
        raise ReconcilerError("all pmfs must share length")
    log_q = [
        sum(w_norm[i] * math.log(max(pmfs[i][j], 0.0) + smoothing) for i in range(K))
        for j in range(n)
    ]
    m = max(log_q)
    exps = [math.exp(x - m) for x in log_q]
    z = sum(exps)
    if z <= 0.0:
        return [1.0 / n] * n
    return [e / z for e in exps]


def kl_divergence(p: Sequence[float], q: Sequence[float], smoothing: float = 1e-12) -> float:
    """KL(p ‖ q), in nats."""
    if len(p) != len(q):
        raise ReconcilerError("pmfs must share length")
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        s += pi * math.log((pi + smoothing) / (qi + smoothing))
    return s


def kl_barycenter(
    pmfs: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    *,
    max_iter: int = 200,
    tol: float = 1e-9,
    smoothing: float = 1e-12,
) -> list[float]:
    """Bregman 1967 KL-barycenter.

    Returns the distribution minimising
    ``Σ_i w_i · KL(q ‖ p_i)`` (right-KL barycenter).  Closed-form
    solution coincides with the logarithmic pool; the iterative form
    is exposed for the Bregman-barycenter literature.
    """
    return logarithmic_pool(pmfs, weights, smoothing=smoothing)


def aumann_iterate(
    pmfs: Sequence[Sequence[float]],
    *,
    max_rounds: int = 32,
    tol: float = 1e-6,
    smoothing: float = 1e-12,
) -> tuple[list[float], bool, int]:
    """Geanakoplos-Polemarchakis 1982 Aumann iteration.

    Each round, every expert updates by averaging with the current
    pool of all experts' beliefs (using the linear pool as the
    common-knowledge information aggregator).  Returns
    ``(consensus, converged, rounds)``.

    Note this is the *cognitive-economy* approximation: in the full
    Aumann formulation each expert posts the *event* in their information
    partition that contains the truth; here we treat the broadcast pmf
    as the information being shared.  The convergence and
    finite-time properties remain.
    """
    K = len(pmfs)
    if K == 0:
        return [], True, 0
    n = len(pmfs[0])
    current: list[list[float]] = [list(p) for p in pmfs]
    converged = False
    rounds = 0
    for r in range(1, max_rounds + 1):
        pool = linear_pool(current)
        max_tv = 0.0
        for i in range(K):
            tv = 0.5 * sum(abs(current[i][j] - pool[j]) for j in range(n))
            max_tv = max(max_tv, tv)
        if max_tv < tol:
            converged = True
            rounds = r
            break
        # Each expert updates by linear pool with the pool (= bayesian
        # update on the broadcast posterior).
        new_current = []
        for i in range(K):
            mixed = linear_pool([current[i], pool], weights=[0.5, 0.5])
            new_current.append(mixed)
        current = new_current
        rounds = r
    final = linear_pool(current)
    return _normalise(final, smoothing), converged, rounds


def hrms_half_width(n: int, conf: float = 0.95) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid half-width
    for the mean of a [0, 1] random variable from n iid samples at
    confidence ``conf``.
    """
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    inner = math.log(math.log(2.0 * n) + math.e) + 0.75 * math.log(10.4 / delta)
    return math.sqrt(max(inner / (2.0 * n), 0.0))


def empirical_bernstein_half_width(
    n: int, variance: float, conf: float = 0.95, range_: float = 1.0
) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein half-width on a
    [0, range_]-bounded random variable."""
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    return math.sqrt(2.0 * variance * math.log(2.0 / delta) / n) + (
        7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    )


def ks_pvalue(samples: Sequence[float]) -> tuple[float, float]:
    """One-sample KS test against H₀: Uniform(0, 1)."""
    n = len(samples)
    if n == 0:
        return 0.0, 1.0
    xs = sorted(min(max(float(x), 0.0), 1.0) for x in samples)
    d_plus = max((i + 1) / n - x for i, x in enumerate(xs))
    d_minus = max(x - i / n for i, x in enumerate(xs))
    d = max(d_plus, d_minus)
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d
    if lam <= 0:
        return d, 1.0
    p = 0.0
    for j in range(1, 101):
        term = ((-1) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lam ** 2))
        p += term
        if abs(term) < 1e-12:
            break
    return d, max(0.0, min(1.0, 2.0 * p))


def effective_number_of_experts(weights: Sequence[float]) -> float:
    """Inverse-Herfindahl-Hirschman effective number of experts.

    For weights summing to 1: ``1 / Σ w_i²``.  Equals the count of
    equal-weight experts; falls to 1 when one source dominates.
    """
    s = sum(weights)
    if s <= 0:
        return 0.0
    w_norm = [w / s for w in weights]
    denom = sum(w * w for w in w_norm)
    return 1.0 / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------------


_GENESIS_PREFIX = b"agi.reconciler.v1\x00"


def ledger_root(secret_key: bytes | None = None) -> str:
    seed = _GENESIS_PREFIX + (secret_key or b"")
    return hashlib.sha256(seed).hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    import json

    def _q(o: Any) -> Any:
        if isinstance(o, float):
            if math.isnan(o):
                return "NaN"
            if math.isinf(o):
                return "Infinity" if o > 0 else "-Infinity"
            return float(repr(o))
        if isinstance(o, dict):
            return {str(k): _q(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [_q(x) for x in o]
        if isinstance(o, (bytes, bytearray)):
            return o.hex()
        return o

    return json.dumps(_q(payload), sort_keys=True, separators=(",", ":")).encode()


def _hash_entry(parent: str, payload: dict[str, Any], hmac_key: bytes | None = None) -> str:
    body = _canonical(payload)
    block = parent.encode() + b"|" + body
    if hmac_key:
        return hmac.new(hmac_key, block, hashlib.sha256).hexdigest()
    return hashlib.sha256(block).hexdigest()


# ---------------------------------------------------------------------------
# Internal per-topic state
# ---------------------------------------------------------------------------


@dataclass
class _TopicState:
    spec: TopicSpec
    sources: dict[str, Source] = field(default_factory=dict)
    # PIT history per source: list of PIT values
    pit_history: dict[str, list[float]] = field(default_factory=dict)
    realised_history: dict[str, list[tuple[dict[str, float], str]]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


EventPublisher = Callable[[str, dict[str, Any]], None]


class Reconciler:
    """Aumann agreement as a runtime primitive.

    Threadsafe at the API surface: a single re-entrant lock guards
    every mutation of per-topic state.
    """

    def __init__(
        self,
        config: ReconcilerConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or ReconcilerConfig()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._topics: dict[str, _TopicState] = {}
        self._chain_head: str = ledger_root(self.config.hmac_key)
        self._started_ts = time.time()
        self._publish(
            RECONCILER_STARTED,
            {"ts": self._started_ts, "method": self.config.method},
        )

    # ------------------------------------------------------------------
    # Event publishing + chain helpers
    # ------------------------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, payload)
        except Exception:
            pass

    def _advance_chain(self, payload: dict[str, Any]) -> str:
        self._chain_head = _hash_entry(self._chain_head, payload, self.config.hmac_key)
        return self._chain_head

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._chain_head

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_topic(self, topic: str, outcomes: Iterable[str]) -> TopicSpec:
        spec = TopicSpec(topic=topic, outcomes=tuple(outcomes))
        with self._lock:
            if topic in self._topics:
                raise InvalidTopic(f"topic {topic!r} already registered")
            self._topics[topic] = _TopicState(spec=spec)
            self._advance_chain(
                {
                    "op": "register_topic",
                    "topic": topic,
                    "outcomes": list(spec.outcomes),
                }
            )
            self._publish(
                RECONCILER_TOPIC_REGISTERED,
                {
                    "topic": topic,
                    "n_outcomes": len(spec.outcomes),
                    "head": self._chain_head,
                },
            )
        return spec

    def remove_topic(self, topic: str) -> None:
        with self._lock:
            if topic not in self._topics:
                raise UnknownTopic(topic)
            del self._topics[topic]
            self._advance_chain({"op": "remove_topic", "topic": topic})
            self._publish(
                RECONCILER_TOPIC_REMOVED, {"topic": topic, "head": self._chain_head}
            )

    def topics(self) -> list[str]:
        with self._lock:
            return sorted(self._topics.keys())

    def topic_spec(self, topic: str) -> TopicSpec:
        with self._lock:
            return self._require_topic(topic).spec

    def clear(self) -> None:
        with self._lock:
            self._topics.clear()
            self._chain_head = ledger_root(self.config.hmac_key)
            self._advance_chain({"op": "clear"})
            self._publish(RECONCILER_CLEARED, {"head": self._chain_head})

    # ------------------------------------------------------------------
    # Contribution
    # ------------------------------------------------------------------

    def contribute(
        self,
        topic: str,
        *,
        source: str,
        belief: dict[str, float],
        weight: float = 1.0,
        realised: str | None = None,
    ) -> None:
        """Record one source's belief over the topic.

        If a source contributes a second time the prior belief is
        replaced; for the calibration history both observations are
        retained.

        If ``realised`` is given, the (belief, realised) pair is also
        appended to the per-source calibration history.
        """
        if weight < 0 or math.isnan(weight) or math.isinf(weight):
            raise InvalidBelief(f"weight must be a non-negative finite real; got {weight!r}")
        with self._lock:
            state = self._require_topic(topic)
            spec = state.spec
            _validate_pmf(belief, spec.outcomes)
            # Normalise the belief — caller may have passed unnormalised.
            vec = _as_vector(belief, spec.outcomes)
            norm = _normalise(vec, smoothing=self.config.smoothing)
            normalised = {o: norm[i] for i, o in enumerate(spec.outcomes)}
            if realised is not None and realised not in spec.outcomes:
                raise InvalidBelief(f"realised outcome {realised!r} not in topic")
            src = Source(
                source_id=source,
                belief=normalised,
                weight=float(weight),
                realised=realised,
            )
            state.sources[source] = src
            if realised is not None:
                history = state.realised_history.setdefault(source, [])
                history.append((dict(normalised), realised))
                pit = state.pit_history.setdefault(source, [])
                # PIT = F(observed) under predicted CDF (using the
                # outcome ordering as the CDF axis).
                idx = spec.outcomes.index(realised)
                cdf = sum(norm[: idx + 1])
                pit.append(cdf)
            self._advance_chain(
                {
                    "op": "contribute",
                    "topic": topic,
                    "source": source,
                    "belief": normalised,
                    "weight": float(weight),
                    "realised": realised,
                }
            )
            self._publish(
                RECONCILER_CONTRIBUTED,
                {
                    "topic": topic,
                    "source": source,
                    "weight": float(weight),
                    "head": self._chain_head,
                },
            )

    def reset_topic(self, topic: str) -> None:
        """Drop all contributions for a topic (keep the topic itself)."""
        with self._lock:
            state = self._require_topic(topic)
            state.sources.clear()
            state.pit_history.clear()
            state.realised_history.clear()
            self._advance_chain({"op": "reset_topic", "topic": topic})

    def sources(self, topic: str) -> list[Source]:
        with self._lock:
            state = self._require_topic(topic)
            return list(state.sources.values())

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------

    def consensus(
        self,
        topic: str,
        *,
        method: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> ConsensusReport:
        """Aggregate every source's belief into a consensus pmf.

        ``method`` overrides the configured default; ``weights`` (per
        source) overrides the per-source ``weight`` at contribute time.
        """
        m = method or self.config.method
        if m not in KNOWN_METHODS:
            raise ReconcilerError(f"unknown method {m!r}")
        with self._lock:
            state = self._require_topic(topic)
            spec = state.spec
            if not state.sources:
                raise InsufficientData(
                    f"topic {topic!r} has no contributions"
                )
            srcs = list(state.sources.values())
            ws = [
                weights.get(s.source_id, s.weight) if weights else s.weight
                for s in srcs
            ]
            pmfs = [_as_vector(s.belief, spec.outcomes) for s in srcs]
            converged = True
            rounds = 0
            if m == METHOD_LINEAR:
                q = linear_pool(pmfs, ws)
            elif m == METHOD_LOGARITHMIC:
                q = logarithmic_pool(pmfs, ws, self.config.smoothing)
            elif m == METHOD_KL_BARYCENTER:
                q = kl_barycenter(pmfs, ws, smoothing=self.config.smoothing)
            else:  # aumann
                q, converged, rounds = aumann_iterate(
                    pmfs,
                    max_rounds=self.config.aumann_max_rounds,
                    tol=self.config.aumann_tol,
                    smoothing=self.config.smoothing,
                )
            q = _normalise(q, self.config.smoothing)
            consensus_dict = {o: q[i] for i, o in enumerate(spec.outcomes)}

            # Per-source KL gap
            per_source_kl: dict[str, float] = {}
            for s, p in zip(srcs, pmfs):
                per_source_kl[s.source_id] = kl_divergence(
                    p, q, smoothing=self.config.smoothing
                )
            outlier: tuple[str, float] | None = None
            if per_source_kl:
                src_id, kl = max(
                    per_source_kl.items(), key=lambda kv: kv[1]
                )
                outlier = (src_id, kl)

            # Per-outcome HRMS anytime-valid CI on the consensus mass
            ci: dict[str, tuple[float, float]] = {}
            n = len(srcs)
            for i, o in enumerate(spec.outcomes):
                masses = [pmfs[j][i] for j in range(n)]
                mu = sum(masses) / n if n > 0 else 0.0
                hw = hrms_half_width(n, self.config.confidence)
                ci[o] = (max(0.0, mu - hw), min(1.0, mu + hw))

            eff_n = effective_number_of_experts(ws)
            payload = {
                "op": "consensus",
                "topic": topic,
                "method": m,
                "consensus": consensus_dict,
                "outlier": list(outlier) if outlier else None,
                "rounds": rounds,
                "converged": converged,
                "effective_n_sources": eff_n,
            }
            self._advance_chain(payload)
            self._publish(
                RECONCILER_CONSENSUS,
                {
                    "topic": topic,
                    "method": m,
                    "rounds": rounds,
                    "converged": converged,
                    "head": self._chain_head,
                },
            )
            return ConsensusReport(
                topic=topic,
                method=m,
                consensus=consensus_dict,
                confidence_interval=ci,
                outlier=outlier,
                per_source_kl=per_source_kl,
                converged=converged,
                rounds=rounds,
                effective_n_sources=eff_n,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibration(self, topic: str, *, source: str) -> CalibrationReport:
        """One-sample KS test on the per-source PIT history."""
        with self._lock:
            state = self._require_topic(topic)
            pit = state.pit_history.get(source, [])
            if len(pit) == 0:
                raise InsufficientData(
                    f"source {source!r} has no realised contributions"
                )
            d, p = ks_pvalue(pit)
            # Log-loss
            ll = 0.0
            for belief, realised in state.realised_history.get(source, []):
                ll -= math.log(max(belief.get(realised, 0.0), 1e-12))
            ll /= max(len(state.realised_history.get(source, [])), 1)
            self._advance_chain(
                {
                    "op": "calibration",
                    "topic": topic,
                    "source": source,
                    "ks": d,
                    "p_value": p,
                    "log_loss": ll,
                }
            )
            self._publish(
                RECONCILER_CALIBRATED,
                {
                    "topic": topic,
                    "source": source,
                    "ks": d,
                    "p_value": p,
                    "log_loss": ll,
                    "head": self._chain_head,
                },
            )
            return CalibrationReport(
                topic=topic,
                source_id=source,
                n_observations=len(pit),
                ks_statistic=d,
                p_value=p,
                log_loss=ll,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Identifiability
    # ------------------------------------------------------------------

    def identifiability_report(self, topic: str) -> IdentifiabilityReport:
        with self._lock:
            state = self._require_topic(topic)
            spec = state.spec
            sources = list(state.sources.values())
            zero_mass: list[str] = []
            for outcome in spec.outcomes:
                if all(s.belief.get(outcome, 0.0) <= 0.0 for s in sources):
                    zero_mass.append(outcome)
            ws = [s.weight for s in sources]
            eff_n = effective_number_of_experts(ws)
            self._advance_chain(
                {
                    "op": "identifiability",
                    "topic": topic,
                    "n_sources": len(sources),
                    "n_zero_mass": len(zero_mass),
                    "effective_n_sources": eff_n,
                }
            )
            return IdentifiabilityReport(
                topic=topic,
                n_sources=len(sources),
                zero_mass_outcomes=zero_mass,
                effective_n_sources=eff_n,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # State checks + accessors
    # ------------------------------------------------------------------

    def _require_topic(self, topic: str) -> _TopicState:
        if topic not in self._topics:
            raise UnknownTopic(topic)
        return self._topics[topic]

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            topics_out: dict[str, Any] = {}
            for tname, state in self._topics.items():
                topics_out[tname] = {
                    "spec": {
                        "topic": state.spec.topic,
                        "outcomes": list(state.spec.outcomes),
                    },
                    "sources": [
                        {
                            "source_id": s.source_id,
                            "belief": s.belief,
                            "weight": s.weight,
                            "realised": s.realised,
                        }
                        for s in state.sources.values()
                    ],
                    "pit_history": dict(state.pit_history),
                    "realised_history": {
                        sid: [(b, r) for b, r in hist]
                        for sid, hist in state.realised_history.items()
                    },
                }
            return {"chain_head": self._chain_head, "topics": topics_out}

    def import_state(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._topics.clear()
            for tname, td in snapshot.get("topics", {}).items():
                spec = TopicSpec(
                    topic=td["spec"]["topic"],
                    outcomes=tuple(td["spec"]["outcomes"]),
                )
                state = _TopicState(spec=spec)
                for s in td.get("sources", []):
                    state.sources[s["source_id"]] = Source(
                        source_id=s["source_id"],
                        belief=dict(s["belief"]),
                        weight=float(s["weight"]),
                        realised=s.get("realised"),
                    )
                state.pit_history = {
                    sid: list(p) for sid, p in td.get("pit_history", {}).items()
                }
                state.realised_history = {
                    sid: [(dict(b), r) for b, r in hist]
                    for sid, hist in td.get("realised_history", {}).items()
                }
                self._topics[tname] = state
            self._chain_head = snapshot.get(
                "chain_head", ledger_root(self.config.hmac_key)
            )
