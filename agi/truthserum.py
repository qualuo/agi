r"""TruthSerum — incentive-compatible peer-prediction as a runtime primitive.

A coordination engine that fans work out to many agents (humans, LLM
judges, sub-models, oracle services, prediction-market traders) has to
combine their reports into a single answer without ground truth. The
classical fix is "majority vote", which has two well-known failure
modes:

  1. **Systematic bias.** When most reporters are wrong in the same
     direction (e.g. the prompt has an obvious wrong answer), majority
     vote learns the bias.
  2. **Strategic misreport.** When reporters know their score depends
     on agreement, "always say A" or "copy the public answer" is a
     dominant strategy unless the mechanism rewards informative
     reporting in equilibrium.

Both failures vanish if the runtime *pays* each reporter according to
a strictly-truthful payment rule and then aggregates by report-weighted
plurality.  This is the **peer-prediction** problem, and the TruthSerum
is its runtime primitive — composes with Auditor (FDR over truthfulness
tests), Coalition (Shapley over reporters), Forecaster (calibrated
belief reports), Equilibrator (truthful Nash verification), and
Strategist (decision under noisy aggregated truth).

Mathematical roots
------------------

  * **Miller, Resnick & Zeckhauser, 2005 — "Eliciting informative
    feedback: the peer-prediction method."**  The first mechanism
    that elicits private signals as a strict Bayes-Nash equilibrium
    without observing ground truth.  Uses a strictly proper scoring
    rule against a *reference reporter's* posterior; requires the
    designer to know the common prior, which limits practical use.

  * **Prelec, 2004 — "A Bayesian Truth Serum for subjective data."**
    Each reporter submits both a *signal* x_i and a *meta-prediction*
    y_i (the distribution they expect *others* to report).  Scoring:
        BTS(i) = log(x̄_{x_i} / ȳ_{x_i}) + α · Σ_k x̄_k log(ȳ_k / x̄_k)
    where x̄ is the population frequency of signals and ȳ the
    geometric-mean meta-prediction.  Asymptotically Bayes-Nash
    truthful for any prior; the famous "surprisingly common" award.

  * **Witkowski & Parkes, 2012 — "A Robust Bayesian Truth Serum for
    small populations."**  RBTS modifies BTS so it is *strictly*
    incentive-compatible for binary signals and *finite* populations
    (n ≥ 3) without knowing the prior, using a shadow reference and
    a quadratic scoring component.

  * **Jurca & Faltings, 2009; Witkowski & Parkes, 2012b.**  Output
    agreement (von Ahn-Dabbish, 2004) is truthful in BNE when signals
    are positively correlated; we ship it as the cheapest baseline.

  * **Dasgupta & Ghosh, 2013 — "Crowdsourced judgement elicitation
    with endogenous proficiency."**  Correlated Agreement (CA) is the
    first detail-free, multi-task mechanism that is *strongly*
    truthful for *any* signal model with positive correlation:
        CA(i,j) = 1[a_i^t = a_j^t] − 1[a_i^t = a_j^{t'}]
    on a shared bonus task t' drawn independently.  Truth maximises
    expected payment among all strategies, with strict margin equal
    to the I(S_i; S_j) over a TV-divergence representation.

  * **Shnayder, Agarwal, Frongillo, Parkes, 2016 — "Informed truth
    seeking in peer prediction."**  Generalises Dasgupta-Ghosh to
    n ≥ 3 reporters and proves *informed-truthful* equilibrium:
    truth-telling strictly Pareto-dominates every other equilibrium.

  * **Kong & Schoenebeck, 2018 — "Water from two rocks: maximising
    the mutual information."**  Multi-task f-Mutual-Information
    mechanism: pay each pair their estimated I_f(S_i; S_j) over
    shared tasks vs random pairs.  For f = total-variation this is
    TVD-MI, recovering Dasgupta-Ghosh.  Truthful in dominant strategy
    for permutation-invariant reporting strategies.

  * **Kong, 2020 — "Dominantly truthful multi-task peer prediction
    with a constant number of tasks."**  Determinant-based Mutual
    Information (DMI):  for k ≥ 2-signal space, score is
        DMI(i,j) = det(M_ij) · det(M_ij')
    where M_ij is the empirical joint signal matrix from a random
    half of shared tasks and M_ij' the other half.  Dominantly
    truthful for k tasks with k as small as 2·|S| − 1.

  * **Liu, Wang & Chen, 2020 — "Surrogate Scoring Rules."**  Suppose
    a noisy proxy ỹ is available (a peer report).  Estimate the
    *confusion matrix* P(ỹ | y) by EM (Dawes-Skene), invert it on
    the score, and recover a *proper* scoring rule against the
    hidden truth y.  Combines BTS-style elicitation with calibrated
    probability reports.

  * **Dawid & Skene, 1979 — "Maximum likelihood estimation of
    observer error-rates using the EM algorithm."**  EM over a
    latent-truth multinomial mixture; recovers per-reporter
    confusion matrices and posterior task labels.  Used here both
    as an aggregation method and as the back-end for SSR.

  * **Hoeffding, 1963; Maurer & Pontil, 2009.**  Hoeffding's bound
    on the empirical-mean radius for n bounded payments gives the
    headline anytime PAC certificate on each reporter's expected
    score.  Empirical Bernstein replaces the worst-case variance
    with the *empirical* variance for a tighter bound at higher n.

  * **Bonferroni, 1936.**  Multiple-testing correction across N
    reporter-level truthfulness tests; composed with Auditor for
    sharper FDR/FWER joint control in the streaming case.

Design contract
---------------

The TruthSerum consumes a stream of ``Report`` records and exposes
one read API ``score(mechanism=…)`` that returns an
``ElicitationReport``: per-reporter expected payment, confidence
interval, anytime PAC certificate, empirical truthful-equilibrium
margin, and an aggregated answer per task.

  * `submit(report)` — append one report.
  * `submit_batch(iterable)` — bulk append.
  * `score(mechanism, alpha=0.05)` — compute scores under one of
    {output_agreement, bts, robust_bts, correlated_agreement,
    determinant_mi, phi_mi, surrogate_scoring}.  All mechanisms emit
    per-reporter Hoeffding/Bernstein CIs.
  * `aggregate(method="weighted_plurality" | "weighted_em" | "plurality")`
    — return the runtime's best guess at the hidden truth per task.
  * `confusion_matrix(reporter_id, …)` — fit a per-reporter Dawes-Skene
    confusion matrix (used by SSR and by aggregate(weighted_em)).
  * `is_strict_truthful_eq(mechanism, …)` — empirical truthful-
    equilibrium verification: for every reporter, the expected payment
    under truthful play is strictly greater than under every
    constant-report or permutation strategy.  Returns the margin (gap)
    of the worst deviation; positive ↔ strict equilibrium.
  * `detect_collusion(…)` — identifies reporter clusters whose
    pairwise agreement is anomalously high vs the null.
  * `clear(scope)` — purge state.
  * `coverage()` — lifetime stats; `report()` — full ElicitationReport.

Every state change emits an event on the optional ``EventBus`` and
records a content-hashed receipt on the optional ``RuntimeAttestor``.
All operations are thread-safe under a single recursive lock; the
module is stdlib-only.

Composition surface
-------------------

  * **Forecaster.** Reporters may submit their `belief` as a categorical
    forecast; SSR consumes it directly and TruthSerum routes the proper
    score through the confusion-matrix inverse to recover a calibrated
    score against the latent truth.
  * **Coalition.** The per-reporter expected payment is a valuation
    function v(S) = Σ_{i∈S} payment_i; the Coalition primitive computes
    Shapley credit over a cooperative aggregation pipeline.
  * **Auditor.** Truthfulness tests at level α — one per reporter, one
    per pair — are p-values fed to Auditor for FDR/FWER joint control.
  * **Equilibrator.** The empirical payoff bimatrix from a mechanism
    can be checked for Nash strictness with the Equilibrator; truthful
    is a Nash iff Equilibrator.is_nash([truth_i for i in 1..n]) holds.
  * **Strategist.** Aggregated truth + posterior probability is a
    feeding signal to the meta-decision layer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

try:
    from agi.events import Event  # type: ignore
except Exception:  # pragma: no cover - keep stdlib-only fallbacks
    Event = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================


TRUTHSERUM_STARTED = "truthserum.started"
TRUTHSERUM_SUBMITTED = "truthserum.submitted"
TRUTHSERUM_SCORED = "truthserum.scored"
TRUTHSERUM_AGGREGATED = "truthserum.aggregated"
TRUTHSERUM_EQ_CHECKED = "truthserum.eq_checked"
TRUTHSERUM_COLLUSION_DETECTED = "truthserum.collusion_detected"
TRUTHSERUM_CONFUSION_FIT = "truthserum.confusion_fit"
TRUTHSERUM_CLEARED = "truthserum.cleared"
TRUTHSERUM_REPORT = "truthserum.report"


# =====================================================================
# Mechanism names
# =====================================================================


MECH_OUTPUT_AGREEMENT = "output_agreement"
MECH_BTS = "bayesian_truth_serum"
MECH_RBTS = "robust_bts"
MECH_CORRELATED_AGREEMENT = "correlated_agreement"
MECH_DMI = "determinant_mi"
MECH_PHI_MI = "phi_mutual_info"
MECH_SSR = "surrogate_scoring_rules"

KNOWN_MECHANISMS = frozenset(
    {
        MECH_OUTPUT_AGREEMENT,
        MECH_BTS,
        MECH_RBTS,
        MECH_CORRELATED_AGREEMENT,
        MECH_DMI,
        MECH_PHI_MI,
        MECH_SSR,
    }
)


# =====================================================================
# Aggregation names
# =====================================================================


AGG_PLURALITY = "plurality"
AGG_WEIGHTED_PLURALITY = "weighted_plurality"
AGG_WEIGHTED_EM = "weighted_em"  # Dawes-Skene EM with per-reporter weights

KNOWN_AGGREGATIONS = frozenset({AGG_PLURALITY, AGG_WEIGHTED_PLURALITY, AGG_WEIGHTED_EM})


# =====================================================================
# f-MI variants
# =====================================================================


PHI_TVD = "tvd"          # 2 max |p - q|
PHI_KL = "kl"            # KL(P || Q)
PHI_JS = "js"            # Jensen-Shannon
PHI_CHI2 = "chi_squared"

KNOWN_PHI = frozenset({PHI_TVD, PHI_KL, PHI_JS, PHI_CHI2})


# =====================================================================
# Errors
# =====================================================================


class TruthSerumError(Exception):
    """Base class for TruthSerum errors."""


class InvalidReport(TruthSerumError):
    """Report is malformed (empty reporter / task / signal, bad belief)."""


class InsufficientData(TruthSerumError):
    """Mechanism requires more reports than are present."""


class UnknownMechanism(TruthSerumError):
    """Caller named a mechanism that does not exist."""


class UnknownAggregation(TruthSerumError):
    """Caller named an aggregation method that does not exist."""


# =====================================================================
# Data classes
# =====================================================================


@dataclass(frozen=True)
class Report:
    """One reporter's answer to one task, with optional meta-prediction.

    ``answer`` is a hashable signal (commonly a string or int).
    ``belief``, if supplied, is a tuple of (signal, probability) pairs
    over the *full* signal alphabet; it MUST sum to 1 within ``tol``.
    Used by BTS / RBTS / SSR; ignored by OA / CA / DMI / phi-MI.
    """

    reporter_id: str
    task_id: str
    answer: Hashable
    belief: tuple = ()
    ts: float = field(default_factory=time.time)
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if not self.reporter_id:
            raise InvalidReport("reporter_id must be non-empty")
        if not self.task_id:
            raise InvalidReport("task_id must be non-empty")
        if self.answer is None:
            raise InvalidReport("answer must not be None")
        if self.belief:
            total = 0.0
            seen: set = set()
            for k, v in self.belief:
                if k in seen:
                    raise InvalidReport(f"duplicate signal {k!r} in belief")
                seen.add(k)
                if not isinstance(v, (int, float)):
                    raise InvalidReport(f"belief value must be numeric, got {type(v).__name__}")
                if v < -self.tol:
                    raise InvalidReport(f"negative belief mass for {k!r}")
                total += float(v)
            if abs(total - 1.0) > self.tol:
                raise InvalidReport(f"belief masses sum to {total:.6f}, not 1")


@dataclass(frozen=True)
class ScoreStats:
    """Per-reporter expected payment and anytime PAC bound."""

    reporter_id: str
    mean_score: float
    sd_score: float
    n_scored: int
    ci_lower: float
    ci_upper: float
    radius: float
    alpha: float
    method: str  # "hoeffding" | "empirical_bernstein"


@dataclass(frozen=True)
class TaskTruth:
    """Aggregated posterior truth for one task."""

    task_id: str
    answer: Hashable
    posterior: float           # weighted-vote share or EM posterior
    plurality_count: int       # # reporters supporting the chosen answer
    n_reports: int


@dataclass(frozen=True)
class ConfusionMatrix:
    """Per-reporter Dawes-Skene confusion matrix.

    ``rows`` indexes the latent truth, ``cols`` the reported signal.
    ``matrix[i][j] = P(report = signals[j] | truth = signals[i])``.
    """

    reporter_id: str
    signals: tuple
    matrix: tuple  # tuple of tuple[float, ...] rows summing to 1
    fit_iters: int
    log_likelihood: float


@dataclass(frozen=True)
class ElicitationReport:
    """Full per-mechanism scoring report."""

    mechanism: str
    n_reporters: int
    n_tasks: int
    n_reports: int
    n_pairings: int
    alpha: float
    scores: tuple                # tuple[ScoreStats, ...]
    aggregation: str
    truths: tuple                # tuple[TaskTruth, ...]
    truthful_strict_eq: bool     # margin > 0 for every reporter
    truthful_eq_margin: float    # min reporter (E[truth] - max E[deviation])
    started_at: float
    finished_at: float
    notes: str
    digest: str


@dataclass(frozen=True)
class CoverageReport:
    """Lifetime stats on a TruthSerum instance."""

    n_reporters: int
    n_tasks: int
    n_reports: int
    n_scorings: int
    n_aggregations: int
    n_eq_checks: int
    n_confusion_fits: int
    started_ns: int
    uptime_seconds: float


# =====================================================================
# Hash helpers / attestation
# =====================================================================


def _hash_payload(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


class _AttestableReceipt:
    __slots__ = ("ticket_id", "kind", "payload", "digest")

    def __init__(self, kind: str, payload: dict, digest: str = "") -> None:
        self.kind = kind
        self.payload = payload
        self.digest = digest
        self.ticket_id = (
            payload.get("receipt_id") or digest[:16] or uuid.uuid4().hex[:16]
        )

    def to_dict(self) -> dict:
        return {"ticket_id": self.ticket_id, "kind": self.kind, "payload": self.payload, "digest": self.digest}


# =====================================================================
# Concentration helpers — anytime-valid PAC bounds
# =====================================================================


def hoeffding_radius(n: int, alpha: float, score_range: float) -> float:
    r"""Hoeffding's bound on the empirical-mean deviation.

    If ``X_1, ..., X_n`` are i.i.d. with values in ``[a, b]`` and
    ``score_range = b - a``, then for all ``alpha ∈ (0,1)``,
        P(|X̄_n - μ| ≥ r) ≤ alpha
    holds with ``r = score_range · sqrt(log(2/alpha) / (2n))``.

    Returns the radius; n must be ≥ 1.
    """
    if n <= 0:
        return float("inf")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if score_range <= 0.0:
        return 0.0
    return score_range * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def empirical_bernstein_radius(
    values: Sequence[float], alpha: float, score_range: float
) -> float:
    r"""Maurer-Pontil empirical-Bernstein bound on the empirical mean.

    With ``V̂_n`` the empirical variance of ``n`` samples in ``[a, b]``,
        P(|X̄_n - μ| ≥ r) ≤ alpha
    holds with
        r = sqrt(2 V̂_n log(2/alpha) / n) + 7 (b-a) log(2/alpha) / (3(n-1)).

    Falls back to Hoeffding for ``n ≤ 1``.
    """
    n = len(values)
    if n <= 1:
        return hoeffding_radius(n, alpha, score_range)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if score_range <= 0.0:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    var = max(var, 0.0)
    return math.sqrt(2.0 * var * math.log(2.0 / alpha) / n) + (
        7.0 * score_range * math.log(2.0 / alpha) / (3.0 * (n - 1))
    )


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Bonferroni-adjusted per-test α for joint-FWER control at level α."""
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_tests <= 0:
        return alpha
    return alpha / n_tests


# =====================================================================
# Pairwise mechanism payments
# =====================================================================


def output_agreement_payment(a_i: Hashable, a_j: Hashable) -> float:
    """Output agreement (von Ahn-Dabbish, 2004): 1[a_i = a_j]."""
    return 1.0 if a_i == a_j else 0.0


def correlated_agreement_payment(
    a_i_t: Hashable,
    a_j_t: Hashable,
    a_i_tprime: Hashable,
    a_j_tprime: Hashable,
) -> float:
    """Dasgupta-Ghosh 2013 correlated-agreement payment.

    Score = 1[a_i^t = a_j^t] - 1[a_i^t = a_j^{t'}], where the second
    indicator pairs reporter i's answer on the *shared* task with
    reporter j's answer on a different (bonus) task t'.  Truthful is
    strictly Bayes-Nash whenever signals are positively correlated.
    """
    base = 1.0 if a_i_t == a_j_t else 0.0
    bonus = 1.0 if a_i_t == a_j_tprime else 0.0
    return base - bonus


def bts_payment(
    signal_i: Hashable,
    meta_i: Mapping[Hashable, float],
    pop_freq: Mapping[Hashable, float],
    geo_meta: Mapping[Hashable, float],
    alpha: float = 1.0,
) -> float:
    r"""Prelec's Bayesian Truth Serum score for one reporter.

    Score components:
        info  = log(pop_freq[signal_i] / geo_meta[signal_i])
        pred  = α · Σ_k pop_freq[k] · log(geo_meta[k] / pop_freq[k])
    Total = info + pred.  Asymptotically Bayes-Nash truthful.

    ``meta_i`` is reporter i's meta-prediction (used to build
    ``geo_meta`` outside this function).  ``pop_freq`` is the empirical
    distribution of reported signals.  Both must sum to ~1 over the
    same signal alphabet.
    """
    eps = 1e-12
    p = max(pop_freq.get(signal_i, 0.0), eps)
    g = max(geo_meta.get(signal_i, 0.0), eps)
    info = math.log(p / g)
    pred = 0.0
    for k in pop_freq:
        pk = max(pop_freq[k], eps)
        gk = max(geo_meta.get(k, 0.0), eps)
        pred += pk * math.log(gk / pk)
    return info + alpha * pred


def rbts_payment(
    signal_i: int,
    meta_i: float,           # P(other reporter signals 1)
    signal_ref: int,         # reference peer signal in {0, 1}
    meta_shadow: float,      # shadow meta-prediction; see below
) -> float:
    r"""Robust Bayesian Truth Serum payment (Witkowski-Parkes 2012).

    Binary signal space ``{0, 1}``.  Information score is
        IS(i, j) = 1 - (signal_i - meta'_j)²
    against a *shifted* meta-prediction
        meta'_j = meta_j + (signal_i - meta_i') · (something small),
    plus a Brier prediction component
        PS(i) = 1 - (1[ref_signal] - meta_i)².
    The Witkowski-Parkes 2012 mechanism is strictly truthful for n ≥ 3
    under any prior with positive correlation.  We use the simplified
    standard form: PS + IS with meta_shadow as the reference shifted
    meta-prediction.

    All inputs in ``[0,1]``.  Returns a real-valued score.
    """
    if signal_i not in (0, 1) or signal_ref not in (0, 1):
        raise InvalidReport("RBTS signals must be in {0, 1}")
    # information score against shifted peer meta:
    info = 1.0 - (signal_i - meta_shadow) ** 2
    # prediction (Brier) score against reference outcome:
    pred = 1.0 - (signal_ref - meta_i) ** 2
    return info + pred


def determinant_mi_payment(
    M_a: Sequence[Sequence[float]],
    M_b: Sequence[Sequence[float]],
) -> float:
    r"""Kong 2020 Determinant-based Mutual Information payment.

    ``M_a`` and ``M_b`` are the *empirical* joint signal matrices of
    reporter i and j on disjoint random subsets of shared tasks.  The
    DMI score is ``det(M_a) · det(M_b)``; truthful is a *dominant*
    strategy when the signal alphabet has size k and ≥ 2(k-1)+1 tasks
    are observed per pairing.
    """
    return _det(M_a) * _det(M_b)


def phi_mi_payment(
    joint: Mapping[tuple, float],
    margin_i: Mapping[Hashable, float],
    margin_j: Mapping[Hashable, float],
    phi: str = PHI_TVD,
) -> float:
    r"""Kong-Schoenebeck 2018 f-Mutual-Information payment.

    Estimated I_phi(S_i; S_j) computed from a joint empirical and the
    two marginals.  For phi = "tvd" this is exactly TVD-MI, recovering
    Dasgupta-Ghosh up to constant; for phi = "kl" this is classical MI;
    for phi = "js" Jensen-Shannon; for phi = "chi_squared" χ²-divergence.

    Returns the divergence ≥ 0 (≈0 ↔ independent).
    """
    if phi not in KNOWN_PHI:
        raise UnknownMechanism(f"unknown phi divergence: {phi!r}")
    keys_i = set(margin_i)
    keys_j = set(margin_j)
    if phi == PHI_TVD:
        total = 0.0
        for a in keys_i:
            for b in keys_j:
                p = joint.get((a, b), 0.0)
                q = margin_i[a] * margin_j[b]
                total += abs(p - q)
        return 0.5 * total
    if phi == PHI_KL:
        total = 0.0
        for (a, b), p in joint.items():
            if p <= 0.0:
                continue
            q = margin_i.get(a, 0.0) * margin_j.get(b, 0.0)
            if q <= 0.0:
                continue
            total += p * math.log(p / q)
        return total
    if phi == PHI_JS:
        # JS(P || Q) = ½ KL(P || M) + ½ KL(Q || M), M = ½(P+Q)
        total = 0.0
        for a in keys_i:
            for b in keys_j:
                p = joint.get((a, b), 0.0)
                q = margin_i[a] * margin_j[b]
                m = 0.5 * (p + q)
                if m <= 0.0:
                    continue
                if p > 0.0:
                    total += 0.5 * p * math.log(p / m)
                if q > 0.0:
                    total += 0.5 * q * math.log(q / m)
        return total
    # chi-squared
    total = 0.0
    for a in keys_i:
        for b in keys_j:
            p = joint.get((a, b), 0.0)
            q = margin_i[a] * margin_j[b]
            if q <= 0.0:
                continue
            total += (p - q) ** 2 / q
    return total


def surrogate_score_payment(
    belief: Mapping[Hashable, float],
    surrogate_label: Hashable,
    confusion: Mapping[Hashable, Mapping[Hashable, float]],
    proper_rule: str = "log",
) -> float:
    r"""Liu-Wang-Chen 2020 Surrogate Scoring Rule.

    The peer's report ``surrogate_label`` is treated as a noisy proxy
    for the latent truth y.  Given a per-reporter confusion matrix
    P(ỹ = c | y = c'), invert and compute an unbiased estimate of the
    proper-score against the hidden y.

    For the log score, the SSR-corrected payment is
        Σ_y T^{-1}[surrogate_label, y] · log belief[y]
    where T is the row-stochastic confusion matrix (rows = truth,
    cols = surrogate).  We compute T^{-1} explicitly for small label
    sets; for ill-conditioned T we fall back to log score against the
    surrogate.
    """
    if not belief:
        return 0.0
    signals = tuple(sorted(belief.keys(), key=lambda x: str(x)))
    # Build T as a row-stochastic |signals|x|signals| matrix.
    n = len(signals)
    T = [[0.0] * n for _ in range(n)]
    for i, y in enumerate(signals):
        row = confusion.get(y, {})
        for j, s in enumerate(signals):
            T[i][j] = float(row.get(s, 0.0))
        rs = sum(T[i])
        if rs > 0.0:
            T[i] = [v / rs for v in T[i]]
        else:
            # uninformative reporter: identity row
            T[i][i] = 1.0
    inv = _invert(T)
    if inv is None:
        # Singular: fall back to surrogate-as-truth
        eps = 1e-12
        return math.log(max(belief.get(surrogate_label, 0.0), eps))
    # surrogate_label column index
    try:
        col = signals.index(surrogate_label)
    except ValueError:
        return 0.0
    score = 0.0
    eps = 1e-12
    for i, y in enumerate(signals):
        b = max(belief.get(y, 0.0), eps)
        if proper_rule == "log":
            s = math.log(b)
        elif proper_rule == "brier":
            s = -sum((b - (1.0 if y == k else 0.0)) ** 2 for k in signals)
        else:
            s = math.log(b)
        score += inv[i][col] * s
    return score


# =====================================================================
# Pure-math helpers
# =====================================================================


def _det(M: Sequence[Sequence[float]]) -> float:
    """Determinant via LU with partial pivoting (in-place on a copy)."""
    n = len(M)
    if n == 0:
        return 1.0
    A = [list(row) for row in M]
    if any(len(r) != n for r in A):
        raise ValueError("determinant requires square matrix")
    sign = 1.0
    for k in range(n):
        # pivot
        pivot = k
        best = abs(A[k][k])
        for r in range(k + 1, n):
            v = abs(A[r][k])
            if v > best:
                best = v
                pivot = r
        if best == 0.0:
            return 0.0
        if pivot != k:
            A[k], A[pivot] = A[pivot], A[k]
            sign = -sign
        pivot_val = A[k][k]
        for r in range(k + 1, n):
            factor = A[r][k] / pivot_val
            for c in range(k, n):
                A[r][c] -= factor * A[k][c]
    diag = 1.0
    for k in range(n):
        diag *= A[k][k]
    return sign * diag


def _invert(M: Sequence[Sequence[float]]) -> list | None:
    """Gauss-Jordan invert; returns None if singular."""
    n = len(M)
    if n == 0:
        return []
    A = [list(row) + [1.0 if j == i else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for k in range(n):
        pivot = k
        best = abs(A[k][k])
        for r in range(k + 1, n):
            v = abs(A[r][k])
            if v > best:
                best = v
                pivot = r
        if best < 1e-14:
            return None
        if pivot != k:
            A[k], A[pivot] = A[pivot], A[k]
        pv = A[k][k]
        A[k] = [v / pv for v in A[k]]
        for r in range(n):
            if r == k:
                continue
            factor = A[r][k]
            if factor == 0.0:
                continue
            A[r] = [A[r][c] - factor * A[k][c] for c in range(2 * n)]
    return [row[n:] for row in A]


def _geometric_mean(vals: Sequence[float]) -> float:
    """Geometric mean of strictly-positive values (clamped to 1e-12)."""
    eps = 1e-12
    if not vals:
        return 0.0
    s = 0.0
    for v in vals:
        s += math.log(max(v, eps))
    return math.exp(s / len(vals))


# =====================================================================
# Dawes-Skene EM (confusion-matrix estimation + soft aggregation)
# =====================================================================


def dawes_skene_em(
    answers: Mapping[str, Mapping[str, Hashable]],
    signals: Sequence[Hashable],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int | None = None,
) -> tuple[dict[str, list[list[float]]], dict[str, dict[Hashable, float]], float, int]:
    r"""Dawes-Skene EM over (reporters × tasks × signals).

    ``answers[reporter_id][task_id] = signal``.  Returns:
        confusions[reporter_id]            — |S| × |S| row-stochastic matrix.
        posteriors[task_id][signal]        — posterior P(truth = signal | reports).
        log_likelihood                     — final log-likelihood.
        iters                              — # EM steps taken.

    Initialised by majority-vote posteriors.  Each M-step laplace-
    smooths the confusion estimate with prior count 1 to avoid zeros.
    """
    sig = list(signals)
    s_index = {s: i for i, s in enumerate(sig)}
    K = len(sig)
    reporters = list(answers.keys())
    tasks_set: set = set()
    for r in reporters:
        for t in answers[r]:
            tasks_set.add(t)
    tasks = sorted(tasks_set)

    if not tasks or not reporters or K == 0:
        return {r: [[1.0 if i == j else 0.0 for j in range(K)] for i in range(K)] for r in reporters}, {}, 0.0, 0

    rng = random.Random(seed)

    # Init: majority-vote posterior, with tiny noise to break ties.
    posteriors: dict[str, list[float]] = {}
    for t in tasks:
        counts = [0.0] * K
        for r in reporters:
            a = answers[r].get(t)
            if a is None:
                continue
            if a in s_index:
                counts[s_index[a]] += 1.0
        total = sum(counts)
        if total == 0.0:
            posteriors[t] = [1.0 / K] * K
        else:
            posteriors[t] = [(c + 1e-3 * rng.random()) / (total + 1e-3 * K) for c in counts]
            z = sum(posteriors[t])
            posteriors[t] = [p / z for p in posteriors[t]]

    prior = [0.0] * K
    confusions: dict[str, list[list[float]]] = {
        r: [[1.0 / K] * K for _ in range(K)] for r in reporters
    }

    prev_ll = -float("inf")
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        # M-step
        prior = [0.0] * K
        for t in tasks:
            for k in range(K):
                prior[k] += posteriors[t][k]
        s = sum(prior) or 1.0
        prior = [p / s for p in prior]

        for r in reporters:
            counts = [[1.0] * K for _ in range(K)]  # Laplace prior
            for t in tasks:
                a = answers[r].get(t)
                if a is None or a not in s_index:
                    continue
                col = s_index[a]
                for k in range(K):
                    counts[k][col] += posteriors[t][k]
            for k in range(K):
                rs = sum(counts[k]) or 1.0
                confusions[r][k] = [c / rs for c in counts[k]]

        # E-step
        ll = 0.0
        for t in tasks:
            log_post = [math.log(max(prior[k], 1e-12)) for k in range(K)]
            for r in reporters:
                a = answers[r].get(t)
                if a is None or a not in s_index:
                    continue
                col = s_index[a]
                for k in range(K):
                    log_post[k] += math.log(max(confusions[r][k][col], 1e-12))
            m = max(log_post)
            exps = [math.exp(lp - m) for lp in log_post]
            z = sum(exps) or 1.0
            posteriors[t] = [e / z for e in exps]
            ll += m + math.log(z)

        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll

    posteriors_out: dict[str, dict[Hashable, float]] = {
        t: {sig[k]: posteriors[t][k] for k in range(K)} for t in tasks
    }
    return confusions, posteriors_out, prev_ll, iters


# =====================================================================
# TruthSerum runtime
# =====================================================================


class TruthSerum:
    """Incentive-compatible elicitation engine.

    Threadsafe; stateless except for the report store and lifetime
    counters.  Optional dependencies:

      bus       — ``agi.events.EventBus`` for live event broadcast.
      attestor  — ``agi.attest.RuntimeAttestor`` for content-hashed
                  receipt persistence.

    A coordination engine drives the TruthSerum through:

        submit(report) → score(mechanism=...) → aggregate(method=...) →
            is_strict_truthful_eq(mechanism) → detect_collusion() →
            report()
    """

    def __init__(
        self,
        *,
        bus: Any = None,
        attestor: Any = None,
        random_seed: int | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._rng = random.Random(random_seed)
        # reports[task_id][reporter_id] = Report
        self._reports: dict[str, dict[str, Report]] = {}
        # _by_reporter[reporter_id] is set of task_ids
        self._by_reporter: dict[str, set[str]] = {}
        # cached confusion matrices, keyed by signal-alphabet hash
        self._confusion_cache: dict[str, dict[str, ConfusionMatrix]] = {}
        self._n_scorings = 0
        self._n_aggregations = 0
        self._n_eq_checks = 0
        self._n_confusion_fits = 0
        self._started_ns = time.time_ns()
        self._emit(
            TRUTHSERUM_STARTED,
            {"id": uuid.uuid4().hex[:16], "ts_ns": self._started_ns},
        )

    # -------- event / attest helpers --------

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
        receipt = _AttestableReceipt(kind=kind, payload=payload, digest=digest)
        try:
            if hasattr(self._attestor, "record"):
                self._attestor.record(kind=kind, payload=payload)
            elif callable(self._attestor):
                self._attestor(receipt)
        except Exception:
            pass
        return digest

    # -------- ingest --------

    def submit(self, report: Report) -> None:
        """Append one report to the elicitation store."""
        with self._lock:
            tdict = self._reports.setdefault(report.task_id, {})
            tdict[report.reporter_id] = report
            self._by_reporter.setdefault(report.reporter_id, set()).add(
                report.task_id
            )
            self._confusion_cache.clear()
            self._emit(
                TRUTHSERUM_SUBMITTED,
                {
                    "task_id": report.task_id,
                    "reporter_id": report.reporter_id,
                    "answer": str(report.answer),
                    "has_belief": bool(report.belief),
                },
            )

    def submit_batch(self, reports: Iterable[Report]) -> int:
        count = 0
        for r in reports:
            self.submit(r)
            count += 1
        return count

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()
            self._by_reporter.clear()
            self._confusion_cache.clear()
            self._emit(TRUTHSERUM_CLEARED, {"ts_ns": time.time_ns()})

    # -------- introspection --------

    def reporters(self) -> tuple:
        with self._lock:
            return tuple(sorted(self._by_reporter.keys()))

    def tasks(self) -> tuple:
        with self._lock:
            return tuple(sorted(self._reports.keys()))

    def signal_alphabet(self) -> tuple:
        with self._lock:
            seen: set = set()
            for tdict in self._reports.values():
                for r in tdict.values():
                    seen.add(r.answer)
            return tuple(sorted(seen, key=lambda x: str(x)))

    def coverage(self) -> CoverageReport:
        with self._lock:
            return CoverageReport(
                n_reporters=len(self._by_reporter),
                n_tasks=len(self._reports),
                n_reports=sum(len(t) for t in self._reports.values()),
                n_scorings=self._n_scorings,
                n_aggregations=self._n_aggregations,
                n_eq_checks=self._n_eq_checks,
                n_confusion_fits=self._n_confusion_fits,
                started_ns=self._started_ns,
                uptime_seconds=(time.time_ns() - self._started_ns) / 1e9,
            )

    # -------- core: scoring --------

    def score(
        self,
        mechanism: str = MECH_CORRELATED_AGREEMENT,
        *,
        alpha: float = 0.05,
        method: str = "empirical_bernstein",
        bts_alpha: float = 1.0,
        phi: str = PHI_TVD,
        ssr_rule: str = "log",
        aggregation: str = AGG_WEIGHTED_PLURALITY,
        confusion: Mapping[str, ConfusionMatrix] | None = None,
        bonferroni: bool = True,
        seed: int | None = None,
    ) -> ElicitationReport:
        """Score every reporter under a chosen mechanism.

        ``alpha`` is the *joint* miscoverage; with ``bonferroni=True``
        each per-reporter CI is at level ``alpha / n_reporters``.
        ``method`` is the concentration bound for the CI:
        ``hoeffding`` or ``empirical_bernstein`` (default).
        """
        if mechanism not in KNOWN_MECHANISMS:
            raise UnknownMechanism(mechanism)
        if aggregation not in KNOWN_AGGREGATIONS:
            raise UnknownAggregation(aggregation)
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha must be in (0, 1)")
        with self._lock:
            started = time.time()
            reporters = sorted(self._by_reporter.keys())
            if len(reporters) < 2:
                raise InsufficientData("need ≥ 2 reporters")
            tasks = sorted(self._reports.keys())
            if not tasks:
                raise InsufficientData("no tasks recorded")
            signals = self.signal_alphabet()
            if not signals:
                raise InsufficientData("no signals recorded")

            n_pairings = 0
            per_reporter_scores: dict[str, list[float]] = {r: [] for r in reporters}
            score_range = 1.0
            mech_seed = seed if seed is not None else self._rng.randrange(2**31)

            if mechanism == MECH_OUTPUT_AGREEMENT:
                n_pairings = self._score_output_agreement(per_reporter_scores)
                score_range = 1.0
            elif mechanism == MECH_BTS:
                self._score_bts(per_reporter_scores, signals, bts_alpha)
                # BTS is unbounded; treat as bounded over the realised range.
                vals = [v for vs in per_reporter_scores.values() for v in vs]
                rng_ = max(vals) - min(vals) if vals else 0.0
                score_range = max(rng_, 1e-6)
                n_pairings = len(reporters)
            elif mechanism == MECH_RBTS:
                n_pairings = self._score_rbts(per_reporter_scores)
                score_range = 2.0
            elif mechanism == MECH_CORRELATED_AGREEMENT:
                n_pairings = self._score_correlated_agreement(
                    per_reporter_scores, mech_seed
                )
                score_range = 2.0  # 1 - (-1) range of CA payment
            elif mechanism == MECH_DMI:
                n_pairings = self._score_dmi(
                    per_reporter_scores, signals, mech_seed
                )
                # det of |S|x|S| empirical matrix in [-1,1]; squared ≤ 1.
                score_range = 1.0
            elif mechanism == MECH_PHI_MI:
                n_pairings = self._score_phi_mi(
                    per_reporter_scores, signals, phi
                )
                # Bound depends on phi; conservative range:
                score_range = 1.0 if phi == PHI_TVD else 5.0
            elif mechanism == MECH_SSR:
                if confusion is None:
                    confusion = self.fit_confusion_matrices(signals=signals)
                n_pairings = self._score_ssr(
                    per_reporter_scores, signals, confusion, ssr_rule
                )
                # log score is unbounded; clip practical range:
                score_range = 10.0

            n_tests = max(len(reporters), 1)
            per_alpha = bonferroni_alpha(alpha, n_tests) if bonferroni else alpha

            stats: list[ScoreStats] = []
            for r in reporters:
                vals = per_reporter_scores[r]
                n = len(vals)
                if n == 0:
                    stats.append(
                        ScoreStats(
                            reporter_id=r,
                            mean_score=0.0,
                            sd_score=0.0,
                            n_scored=0,
                            ci_lower=float("-inf"),
                            ci_upper=float("inf"),
                            radius=float("inf"),
                            alpha=per_alpha,
                            method=method,
                        )
                    )
                    continue
                mu = sum(vals) / n
                if n > 1:
                    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
                else:
                    var = 0.0
                sd = math.sqrt(max(var, 0.0))
                if method == "hoeffding":
                    rad = hoeffding_radius(n, per_alpha, score_range)
                else:
                    rad = empirical_bernstein_radius(vals, per_alpha, score_range)
                stats.append(
                    ScoreStats(
                        reporter_id=r,
                        mean_score=mu,
                        sd_score=sd,
                        n_scored=n,
                        ci_lower=mu - rad,
                        ci_upper=mu + rad,
                        radius=rad,
                        alpha=per_alpha,
                        method=method,
                    )
                )

            score_map = {s.reporter_id: max(s.mean_score, 0.0) for s in stats}
            truths = self._aggregate(aggregation, score_map, signals)
            margin, ok = self._eq_margin(per_reporter_scores, mechanism)

            finished = time.time()
            n_reports = sum(len(t) for t in self._reports.values())

            report = ElicitationReport(
                mechanism=mechanism,
                n_reporters=len(reporters),
                n_tasks=len(tasks),
                n_reports=n_reports,
                n_pairings=n_pairings,
                alpha=alpha,
                scores=tuple(stats),
                aggregation=aggregation,
                truths=tuple(truths),
                truthful_strict_eq=ok,
                truthful_eq_margin=margin,
                started_at=started,
                finished_at=finished,
                notes=(
                    f"score_range≈{score_range:.4f}; "
                    f"per_alpha={per_alpha:.4g}; "
                    f"bonferroni={bonferroni}"
                ),
                digest="",
            )
            digest_payload = {
                "mechanism": mechanism,
                "n_reporters": report.n_reporters,
                "n_tasks": report.n_tasks,
                "alpha": alpha,
                "scores": [
                    (s.reporter_id, round(s.mean_score, 8), s.n_scored)
                    for s in stats
                ],
                "truthful_strict_eq": ok,
                "truthful_eq_margin": round(margin, 8),
            }
            digest = self._attest(TRUTHSERUM_SCORED, digest_payload)
            report = ElicitationReport(**{**report.__dict__, "digest": digest})
            self._n_scorings += 1
            self._emit(
                TRUTHSERUM_SCORED,
                {
                    "mechanism": mechanism,
                    "n_reporters": report.n_reporters,
                    "n_tasks": report.n_tasks,
                    "digest": digest,
                    "truthful_strict_eq": ok,
                },
            )
            return report

    # ---- mechanism backends ----

    def _score_output_agreement(
        self, per_reporter_scores: dict[str, list[float]]
    ) -> int:
        n_pairings = 0
        for task_id, tdict in self._reports.items():
            reporters_t = list(tdict.keys())
            for r in reporters_t:
                peers = [p for p in reporters_t if p != r]
                if not peers:
                    continue
                a_r = tdict[r].answer
                payment = 0.0
                for p in peers:
                    payment += output_agreement_payment(a_r, tdict[p].answer)
                    n_pairings += 1
                per_reporter_scores[r].append(payment / len(peers))
        return n_pairings

    def _score_bts(
        self,
        per_reporter_scores: dict[str, list[float]],
        signals: Sequence,
        bts_alpha: float,
    ) -> None:
        # Collect per-signal frequencies and per-signal geometric-mean
        # meta-prediction *over reporters that supplied beliefs*.
        pop_count: dict[Hashable, float] = {s: 0.0 for s in signals}
        meta_acc: dict[Hashable, list[float]] = {s: [] for s in signals}
        # Aggregate over tasks: average reporter belief by task & overall
        n_reports = 0
        for tdict in self._reports.values():
            for r_id, rep in tdict.items():
                pop_count[rep.answer] = pop_count.get(rep.answer, 0.0) + 1.0
                n_reports += 1
                if rep.belief:
                    for k, v in rep.belief:
                        meta_acc.setdefault(k, []).append(float(v))
        if n_reports == 0:
            return
        pop_freq: dict[Hashable, float] = {
            k: v / n_reports for k, v in pop_count.items()
        }
        geo_meta: dict[Hashable, float] = {}
        for k in signals:
            vs = meta_acc.get(k, [])
            geo_meta[k] = _geometric_mean(vs) if vs else 1.0 / max(len(signals), 1)
        # Normalise geo_meta (geometric-mean over reporters not stochastic)
        z = sum(geo_meta.values()) or 1.0
        geo_meta = {k: v / z for k, v in geo_meta.items()}
        # Score each reporter once: average BTS-payment over their reports.
        for r_id, task_ids in self._by_reporter.items():
            payments: list[float] = []
            for t in task_ids:
                rep = self._reports[t].get(r_id)
                if rep is None:
                    continue
                # Use the reporter's *own* meta-prediction if available.
                if rep.belief:
                    meta_r: dict[Hashable, float] = {k: v for k, v in rep.belief}
                else:
                    meta_r = dict(pop_freq)  # uninformed reporter
                # Combine with population geo_meta (for the prediction
                # part); BTS uses the geometric mean of *all* meta-preds,
                # so we recompute the per-task geometric mean here for
                # accuracy:
                payments.append(
                    bts_payment(
                        signal_i=rep.answer,
                        meta_i=meta_r,
                        pop_freq=pop_freq,
                        geo_meta=geo_meta,
                        alpha=bts_alpha,
                    )
                )
            if payments:
                per_reporter_scores[r_id].extend(payments)

    def _score_rbts(self, per_reporter_scores: dict[str, list[float]]) -> int:
        n_pairings = 0
        for task_id, tdict in self._reports.items():
            reporters_t = list(tdict.keys())
            if len(reporters_t) < 3:
                continue
            for r in reporters_t:
                peers = [p for p in reporters_t if p != r]
                ref = peers[0]
                shadow = peers[1] if len(peers) > 1 else peers[0]
                rep_r = tdict[r]
                rep_ref = tdict[ref]
                rep_sh = tdict[shadow]
                if rep_r.answer not in (0, 1) or rep_ref.answer not in (0, 1):
                    continue
                # extract meta_i from reporter's belief over {0,1}
                meta_r = 0.5
                if rep_r.belief:
                    for k, v in rep_r.belief:
                        if k == 1:
                            meta_r = float(v)
                meta_sh = 0.5
                if rep_sh.belief:
                    for k, v in rep_sh.belief:
                        if k == 1:
                            meta_sh = float(v)
                payment = rbts_payment(
                    signal_i=int(rep_r.answer),
                    meta_i=meta_r,
                    signal_ref=int(rep_ref.answer),
                    meta_shadow=meta_sh,
                )
                per_reporter_scores[r].append(payment)
                n_pairings += 1
        return n_pairings

    def _score_correlated_agreement(
        self, per_reporter_scores: dict[str, list[float]], seed: int
    ) -> int:
        # Variance-reduced CA estimator: for each (i, t, peer j), use the
        # *average* bonus indicator over every t' ≠ t where j reported.
        # Expectation is identical to the single-sample Dasgupta-Ghosh
        # estimator but variance is O(1/|t'|) lower.  Per-reporter freq
        # of a_j is cached once per j.
        task_ids = list(self._reports.keys())
        if len(task_ids) < 2:
            raise InsufficientData("CA needs ≥ 2 tasks")
        # Cache reporter -> {answer: count} across all their reports.
        j_counts: dict[str, dict] = {}
        for r_id, tasks in self._by_reporter.items():
            c: dict = {}
            for t in tasks:
                a = self._reports[t][r_id].answer
                c[a] = c.get(a, 0) + 1
            j_counts[r_id] = c
        n_pairings = 0
        for t in task_ids:
            tdict = self._reports[t]
            reporters_t = list(tdict.keys())
            if len(reporters_t) < 2:
                continue
            for r in reporters_t:
                peers = [p for p in reporters_t if p != r]
                if not peers:
                    continue
                a_r = tdict[r].answer
                payment_sum = 0.0
                n_peers = 0
                for j in peers:
                    # 1[a_r = a_j^t]
                    base = 1.0 if a_r == tdict[j].answer else 0.0
                    # E[1[a_r = a_j^{t'}]] over t' ≠ t where j reported
                    j_total = sum(j_counts[j].values())
                    j_on_t = 1 if a_r in [tdict[j].answer] else 0
                    j_a_count = j_counts[j].get(a_r, 0) - j_on_t
                    denom = j_total - 1
                    bonus_freq = (j_a_count / denom) if denom > 0 else 0.0
                    payment_sum += base - bonus_freq
                    n_peers += 1
                    n_pairings += 1
                if n_peers > 0:
                    per_reporter_scores[r].append(payment_sum / n_peers)
        return n_pairings

    def _score_dmi(
        self,
        per_reporter_scores: dict[str, list[float]],
        signals: Sequence,
        seed: int,
    ) -> int:
        rng = random.Random(seed)
        s_index = {s: i for i, s in enumerate(signals)}
        K = len(signals)
        if K < 2:
            raise InsufficientData("DMI needs alphabet size ≥ 2")
        n_pairings = 0
        reporters = sorted(self._by_reporter.keys())
        # For each ordered pair (i, j) compute DMI payment; assign to i.
        for r_i, r_j in itertools.permutations(reporters, 2):
            shared = sorted(self._by_reporter[r_i] & self._by_reporter[r_j])
            if len(shared) < 2 * K - 1 and len(shared) < 2:
                continue
            if len(shared) < 2:
                continue
            tasks_perm = list(shared)
            rng.shuffle(tasks_perm)
            half = len(tasks_perm) // 2
            if half == 0:
                continue
            A_tasks = tasks_perm[:half]
            B_tasks = tasks_perm[half:]
            if not B_tasks:
                continue
            M_a = [[0.0] * K for _ in range(K)]
            for t in A_tasks:
                a_i = self._reports[t][r_i].answer
                a_j = self._reports[t][r_j].answer
                if a_i in s_index and a_j in s_index:
                    M_a[s_index[a_i]][s_index[a_j]] += 1.0
            for r in range(K):
                rs = sum(M_a[r])
                if rs > 0:
                    M_a[r] = [v / rs for v in M_a[r]]
            M_b = [[0.0] * K for _ in range(K)]
            for t in B_tasks:
                a_i = self._reports[t][r_i].answer
                a_j = self._reports[t][r_j].answer
                if a_i in s_index and a_j in s_index:
                    M_b[s_index[a_i]][s_index[a_j]] += 1.0
            for r in range(K):
                rs = sum(M_b[r])
                if rs > 0:
                    M_b[r] = [v / rs for v in M_b[r]]
            payment = determinant_mi_payment(M_a, M_b)
            per_reporter_scores[r_i].append(payment)
            n_pairings += 1
        return n_pairings

    def _score_phi_mi(
        self,
        per_reporter_scores: dict[str, list[float]],
        signals: Sequence,
        phi: str,
    ) -> int:
        n_pairings = 0
        reporters = sorted(self._by_reporter.keys())
        for r_i, r_j in itertools.permutations(reporters, 2):
            shared = sorted(self._by_reporter[r_i] & self._by_reporter[r_j])
            if len(shared) < 2:
                continue
            joint: dict[tuple, float] = {}
            margin_i: dict[Hashable, float] = {s: 0.0 for s in signals}
            margin_j: dict[Hashable, float] = {s: 0.0 for s in signals}
            for t in shared:
                a_i = self._reports[t][r_i].answer
                a_j = self._reports[t][r_j].answer
                joint[(a_i, a_j)] = joint.get((a_i, a_j), 0.0) + 1.0
                margin_i[a_i] = margin_i.get(a_i, 0.0) + 1.0
                margin_j[a_j] = margin_j.get(a_j, 0.0) + 1.0
            total = float(len(shared))
            joint = {k: v / total for k, v in joint.items()}
            margin_i = {k: v / total for k, v in margin_i.items()}
            margin_j = {k: v / total for k, v in margin_j.items()}
            payment = phi_mi_payment(joint, margin_i, margin_j, phi=phi)
            per_reporter_scores[r_i].append(payment)
            n_pairings += 1
        return n_pairings

    def _score_ssr(
        self,
        per_reporter_scores: dict[str, list[float]],
        signals: Sequence,
        confusion: Mapping[str, ConfusionMatrix],
        ssr_rule: str,
    ) -> int:
        n_pairings = 0
        reporters = sorted(self._by_reporter.keys())
        for r_i, r_j in itertools.permutations(reporters, 2):
            shared = sorted(self._by_reporter[r_i] & self._by_reporter[r_j])
            if not shared:
                continue
            cm_j = confusion.get(r_j)
            if cm_j is None:
                continue
            # Convert cm_j.matrix → dict[truth][surrogate] = P
            T = {
                cm_j.signals[i]: {
                    cm_j.signals[k]: cm_j.matrix[i][k]
                    for k in range(len(cm_j.signals))
                }
                for i in range(len(cm_j.signals))
            }
            for t in shared:
                rep_i = self._reports[t][r_i]
                if not rep_i.belief:
                    continue
                belief = {k: float(v) for k, v in rep_i.belief}
                surrogate = self._reports[t][r_j].answer
                payment = surrogate_score_payment(
                    belief=belief,
                    surrogate_label=surrogate,
                    confusion=T,
                    proper_rule=ssr_rule,
                )
                per_reporter_scores[r_i].append(payment)
                n_pairings += 1
        return n_pairings

    # ---- aggregation ----

    def aggregate(
        self,
        method: str = AGG_WEIGHTED_PLURALITY,
        *,
        scores: Mapping[str, float] | None = None,
        signals: Sequence | None = None,
    ) -> tuple:
        """Return aggregated TaskTruth per task."""
        with self._lock:
            if method not in KNOWN_AGGREGATIONS:
                raise UnknownAggregation(method)
            sig = tuple(signals) if signals is not None else self.signal_alphabet()
            score_map = dict(scores or {})
            truths = self._aggregate(method, score_map, sig)
            self._n_aggregations += 1
            self._emit(
                TRUTHSERUM_AGGREGATED,
                {"method": method, "n_tasks": len(truths)},
            )
            return tuple(truths)

    def _aggregate(
        self,
        method: str,
        scores: Mapping[str, float],
        signals: Sequence,
    ) -> list[TaskTruth]:
        truths: list[TaskTruth] = []
        if method == AGG_WEIGHTED_EM:
            answers = {
                r_id: {
                    t: self._reports[t][r_id].answer
                    for t in tasks
                    if r_id in self._reports[t]
                }
                for r_id, tasks in self._by_reporter.items()
            }
            _, posteriors, _, _ = dawes_skene_em(answers, list(signals))
            for t in sorted(self._reports.keys()):
                pst = posteriors.get(t, {})
                if not pst:
                    continue
                best, prob = max(pst.items(), key=lambda kv: kv[1])
                count = sum(
                    1
                    for r_id in self._reports[t]
                    if self._reports[t][r_id].answer == best
                )
                truths.append(
                    TaskTruth(
                        task_id=t,
                        answer=best,
                        posterior=prob,
                        plurality_count=count,
                        n_reports=len(self._reports[t]),
                    )
                )
            return truths

        for t in sorted(self._reports.keys()):
            counts: dict[Hashable, float] = {}
            raw_counts: dict[Hashable, int] = {}
            for r_id, rep in self._reports[t].items():
                weight = 1.0
                if method == AGG_WEIGHTED_PLURALITY:
                    weight = max(scores.get(r_id, 1.0), 1e-9)
                counts[rep.answer] = counts.get(rep.answer, 0.0) + weight
                raw_counts[rep.answer] = raw_counts.get(rep.answer, 0) + 1
            if not counts:
                continue
            total = sum(counts.values()) or 1.0
            best, w = max(counts.items(), key=lambda kv: kv[1])
            truths.append(
                TaskTruth(
                    task_id=t,
                    answer=best,
                    posterior=w / total,
                    plurality_count=raw_counts.get(best, 0),
                    n_reports=len(self._reports[t]),
                )
            )
        return truths

    # ---- confusion matrices ----

    def fit_confusion_matrices(
        self,
        *,
        signals: Sequence | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> dict[str, ConfusionMatrix]:
        """Fit per-reporter Dawes-Skene confusion matrices."""
        with self._lock:
            sig = list(signals) if signals is not None else list(self.signal_alphabet())
            answers = {
                r_id: {
                    t: self._reports[t][r_id].answer
                    for t in tasks
                    if r_id in self._reports[t]
                }
                for r_id, tasks in self._by_reporter.items()
            }
            confusions, _, ll, iters = dawes_skene_em(
                answers, sig, max_iter=max_iter, tol=tol, seed=seed
            )
            out: dict[str, ConfusionMatrix] = {}
            for r_id, M in confusions.items():
                out[r_id] = ConfusionMatrix(
                    reporter_id=r_id,
                    signals=tuple(sig),
                    matrix=tuple(tuple(row) for row in M),
                    fit_iters=iters,
                    log_likelihood=ll,
                )
            self._n_confusion_fits += 1
            self._emit(
                TRUTHSERUM_CONFUSION_FIT,
                {"n_reporters": len(out), "iters": iters, "log_likelihood": ll},
            )
            return out

    # ---- truthful equilibrium check ----

    def _eq_margin(
        self,
        per_reporter_scores: dict[str, list[float]],
        mechanism: str,
    ) -> tuple[float, bool]:
        """Empirical truthful-equilibrium margin.

        For each reporter, compute:
            U_truth(r) = mean payment under as-reported play.
            U_dev(r)  = max over constant deviation a*:
                          mean payment if reporter r had reported a* on
                          *all* their tasks, holding peers fixed.
        Margin = min_r (U_truth(r) - U_dev(r)).
        Truthful is a strict empirical Nash iff margin > 0.
        """
        signals = self.signal_alphabet()
        if not signals or len(signals) < 2:
            return 0.0, False
        worst: float = float("inf")
        for r_id, task_ids in self._by_reporter.items():
            obs = per_reporter_scores.get(r_id, [])
            if not obs:
                continue
            u_truth = sum(obs) / len(obs)
            best_dev = -float("inf")
            for a_star in signals:
                u_dev = self._counterfactual_mean(r_id, a_star, mechanism)
                if u_dev > best_dev:
                    best_dev = u_dev
            margin = u_truth - best_dev
            if margin < worst:
                worst = margin
        if worst == float("inf"):
            return 0.0, False
        return worst, worst > 0.0

    def _counterfactual_mean(
        self, r_id: str, a_star: Hashable, mechanism: str
    ) -> float:
        """Mean payment had reporter r reported a_star on every task,
        holding peers fixed.

        Mechanism-specific; covers OA, CA, RBTS (binary), and a default
        per-task agreement for others (sufficient for the empirical
        margin check used by ``is_strict_truthful_eq``).
        """
        tasks = self._by_reporter.get(r_id, set())
        if not tasks:
            return 0.0
        payments: list[float] = []
        if mechanism == MECH_CORRELATED_AGREEMENT:
            # Use the same variance-reduced estimator as the forward path,
            # but with reporter r's answer pinned to a_star on every task.
            j_counts: dict[str, dict] = {}
            for j_id, tids in self._by_reporter.items():
                c: dict = {}
                for tt in tids:
                    a = self._reports[tt][j_id].answer
                    c[a] = c.get(a, 0) + 1
                j_counts[j_id] = c
            for t in tasks:
                tdict = self._reports[t]
                peers = [p for p in tdict if p != r_id]
                if not peers:
                    continue
                payment_sum = 0.0
                n_peers = 0
                for j in peers:
                    base = 1.0 if a_star == tdict[j].answer else 0.0
                    j_total = sum(j_counts[j].values())
                    j_on_t = 1 if a_star == tdict[j].answer else 0
                    j_a_count = j_counts[j].get(a_star, 0) - j_on_t
                    denom = j_total - 1
                    bonus_freq = (j_a_count / denom) if denom > 0 else 0.0
                    payment_sum += base - bonus_freq
                    n_peers += 1
                if n_peers > 0:
                    payments.append(payment_sum / n_peers)
        elif mechanism == MECH_OUTPUT_AGREEMENT:
            for t in tasks:
                tdict = self._reports[t]
                peers = [p for p in tdict if p != r_id]
                if not peers:
                    continue
                payment = 0.0
                for p in peers:
                    payment += output_agreement_payment(a_star, tdict[p].answer)
                payments.append(payment / len(peers))
        elif mechanism in (MECH_DMI, MECH_PHI_MI):
            # Constant reports are statistically independent of every
            # peer's report distribution, so the f-MI / DMI between
            # reporter i (constant) and any peer j is exactly 0.
            return 0.0
        elif mechanism == MECH_BTS:
            # BTS deviation: reporter always submits signal a_star,
            # holds the population freq fixed. Recompute BTS payment
            # for a_star against the global pop_freq / geo_meta.
            pop_count: dict = {}
            n_reports = 0
            for tt, td in self._reports.items():
                for rid_, rep in td.items():
                    pop_count[rep.answer] = pop_count.get(rep.answer, 0) + 1
                    n_reports += 1
            if n_reports == 0:
                return 0.0
            pop_freq = {k: v / n_reports for k, v in pop_count.items()}
            n_tasks = len(tasks)
            # Constant report leaves geo_meta unchanged but shifts pop_freq
            # by 1/N_total per task: assume small effect.
            geo = pop_freq
            eps = 1e-12
            p = max(pop_freq.get(a_star, 0.0), eps)
            g = max(geo.get(a_star, 0.0), eps)
            info = math.log(p / g)
            pred = 0.0
            for k in pop_freq:
                pk = max(pop_freq[k], eps)
                gk = max(geo.get(k, 0.0), eps)
                pred += pk * math.log(gk / pk)
            return (info + pred) * n_tasks / max(n_tasks, 1)
        elif mechanism == MECH_RBTS:
            if a_star not in (0, 1):
                return -float("inf")
            for t in tasks:
                tdict = self._reports[t]
                peers = [p for p in tdict if p != r_id]
                if len(peers) < 2:
                    continue
                ref = peers[0]
                sh = peers[1]
                if tdict[ref].answer not in (0, 1):
                    continue
                meta_sh = 0.5
                if tdict[sh].belief:
                    for k, v in tdict[sh].belief:
                        if k == 1:
                            meta_sh = float(v)
                payments.append(
                    rbts_payment(
                        signal_i=int(a_star),
                        meta_i=0.5,  # uninformed deviation
                        signal_ref=int(tdict[ref].answer),
                        meta_shadow=meta_sh,
                    )
                )
        else:
            # Default: simple agreement rate against the first peer.
            for t in tasks:
                tdict = self._reports[t]
                peers = [p for p in tdict if p != r_id]
                if not peers:
                    continue
                payments.append(
                    output_agreement_payment(a_star, tdict[peers[0]].answer)
                )
        if not payments:
            return 0.0
        return sum(payments) / len(payments)

    def is_strict_truthful_eq(
        self,
        mechanism: str = MECH_CORRELATED_AGREEMENT,
    ) -> tuple[bool, float]:
        """Empirical strict-Nash check for truthful play."""
        with self._lock:
            if mechanism not in KNOWN_MECHANISMS:
                raise UnknownMechanism(mechanism)
            scratch: dict[str, list[float]] = {
                r: [] for r in self._by_reporter
            }
            if mechanism == MECH_OUTPUT_AGREEMENT:
                self._score_output_agreement(scratch)
            elif mechanism == MECH_CORRELATED_AGREEMENT:
                self._score_correlated_agreement(scratch, self._rng.randrange(2**31))
            elif mechanism == MECH_RBTS:
                self._score_rbts(scratch)
            elif mechanism == MECH_DMI:
                self._score_dmi(scratch, self.signal_alphabet(), self._rng.randrange(2**31))
            elif mechanism == MECH_PHI_MI:
                self._score_phi_mi(scratch, self.signal_alphabet(), PHI_TVD)
            elif mechanism == MECH_BTS:
                self._score_bts(scratch, self.signal_alphabet(), 1.0)
            else:
                # SSR — fall back to OA-like check
                self._score_output_agreement(scratch)
            margin, ok = self._eq_margin(scratch, mechanism)
            self._n_eq_checks += 1
            self._emit(
                TRUTHSERUM_EQ_CHECKED,
                {"mechanism": mechanism, "margin": margin, "strict": ok},
            )
            return ok, margin

    # ---- collusion detection ----

    def detect_collusion(
        self,
        *,
        alpha: float = 0.01,
        min_overlap: int = 5,
    ) -> tuple:
        """Identify reporter clusters with anomalously high pairwise
        agreement vs an independence-null Bonferroni-corrected at
        joint level ``alpha``.

        Returns a tuple of frozensets (each a suspected colluding
        clique, of size ≥ 2).
        """
        with self._lock:
            reporters = sorted(self._by_reporter.keys())
            if len(reporters) < 2:
                return ()
            signals = self.signal_alphabet()
            n_signals = max(len(signals), 1)
            base_match = 1.0 / n_signals
            n_pairs = len(reporters) * (len(reporters) - 1) // 2
            per_alpha = bonferroni_alpha(alpha, max(n_pairs, 1))
            # threshold: agreement > base_match + Hoeffding-radius for n
            cliques: list[set] = []
            for r_i, r_j in itertools.combinations(reporters, 2):
                shared = sorted(
                    self._by_reporter[r_i] & self._by_reporter[r_j]
                )
                if len(shared) < min_overlap:
                    continue
                agree = sum(
                    1
                    for t in shared
                    if self._reports[t][r_i].answer == self._reports[t][r_j].answer
                ) / len(shared)
                rad = hoeffding_radius(len(shared), per_alpha, 1.0)
                if agree - base_match > rad:
                    # merge into existing clique if one of r_i or r_j is in.
                    merged = False
                    for c in cliques:
                        if r_i in c or r_j in c:
                            c.update({r_i, r_j})
                            merged = True
                            break
                    if not merged:
                        cliques.append({r_i, r_j})
            out: tuple = tuple(frozenset(c) for c in cliques if len(c) >= 2)
            self._emit(
                TRUTHSERUM_COLLUSION_DETECTED,
                {"n_cliques": len(out), "alpha": alpha},
            )
            return out


# =====================================================================
# Convenience facade for one-off scoring
# =====================================================================


def quick_score(
    reports: Iterable[Report],
    mechanism: str = MECH_CORRELATED_AGREEMENT,
    *,
    alpha: float = 0.05,
    aggregation: str = AGG_WEIGHTED_PLURALITY,
    seed: int | None = None,
) -> ElicitationReport:
    """One-shot: ingest reports, score under one mechanism, return the
    report.  Useful for inline tests and demos."""
    ts = TruthSerum(random_seed=seed)
    ts.submit_batch(reports)
    return ts.score(mechanism=mechanism, alpha=alpha, aggregation=aggregation, seed=seed)


__all__ = [
    # constants
    "TRUTHSERUM_STARTED",
    "TRUTHSERUM_SUBMITTED",
    "TRUTHSERUM_SCORED",
    "TRUTHSERUM_AGGREGATED",
    "TRUTHSERUM_EQ_CHECKED",
    "TRUTHSERUM_COLLUSION_DETECTED",
    "TRUTHSERUM_CONFUSION_FIT",
    "TRUTHSERUM_CLEARED",
    "TRUTHSERUM_REPORT",
    "MECH_OUTPUT_AGREEMENT",
    "MECH_BTS",
    "MECH_RBTS",
    "MECH_CORRELATED_AGREEMENT",
    "MECH_DMI",
    "MECH_PHI_MI",
    "MECH_SSR",
    "KNOWN_MECHANISMS",
    "AGG_PLURALITY",
    "AGG_WEIGHTED_PLURALITY",
    "AGG_WEIGHTED_EM",
    "KNOWN_AGGREGATIONS",
    "PHI_TVD",
    "PHI_KL",
    "PHI_JS",
    "PHI_CHI2",
    "KNOWN_PHI",
    # exceptions
    "TruthSerumError",
    "InvalidReport",
    "InsufficientData",
    "UnknownMechanism",
    "UnknownAggregation",
    # data classes
    "Report",
    "ScoreStats",
    "TaskTruth",
    "ConfusionMatrix",
    "ElicitationReport",
    "CoverageReport",
    # primitive class
    "TruthSerum",
    # concentration helpers
    "hoeffding_radius",
    "empirical_bernstein_radius",
    "bonferroni_alpha",
    # mechanism payments
    "output_agreement_payment",
    "correlated_agreement_payment",
    "bts_payment",
    "rbts_payment",
    "determinant_mi_payment",
    "phi_mi_payment",
    "surrogate_score_payment",
    # EM
    "dawes_skene_em",
    # facade
    "quick_score",
]
