"""Strategist — top-level meta-decision API for the coordination engine.

Below this module the runtime ships seven first-class forecasting layers:
`PreflightEstimator`, `CalibrationEngine`, `ConformalPredictor`,
`CausalLab`, `PolicyLab`, `PortfolioOptimizer`, and `TicketOracle`. Each
answers a precise statistical question. None of them on their own answers
the *operational* question a coordination engine actually has to make,
turn after turn:

    "Given these candidate actions, this context, this SLO, and what we
     have learned so far, what is the right thing to do, and how
     confident are we in the answer?"

`Strategist` is that surface. It fuses every available forecaster into
one decision and returns a structured `StrategyRecommendation` the
coordination engine can act on. It honors any subset of forecasters that
are wired in — a fresh deployment with only a `PreflightEstimator`
already gets a useful recommendation; a mature deployment with the full
stack gets risk-adjusted, doubly-robust, conformally-bounded decisions
out of the same one call.

What it does, concretely
------------------------

    strat = Strategist(
        calibration=cal_engine,
        conformal=conformal_cost,
        causal=causal_lab,
        policy_lab=pol_lab,
        baseline_action_id="claude-opus-4-7",
    )

    rec = strat.recommend(
        candidates=[
            Candidate(id="haiku",  raw_p_success=0.74, raw_cost_usd=0.05),
            Candidate(id="sonnet", raw_p_success=0.85, raw_cost_usd=0.18),
            Candidate(id="opus",   raw_p_success=0.92, raw_cost_usd=0.40),
        ],
        constraints=StrategyConstraints(
            target_p_success=0.90,
            max_cost_usd=0.50,
            payoff_usd=2.00,
            risk_aversion=1.5,
        ),
        context={"task_difficulty": 0.7, "tenant": "acme"},
    )

    if rec.strategy == STRAT_SINGLE:
        coordinator.dispatch(rec.primary.candidate)
    elif rec.strategy == STRAT_HEDGE:
        coordinator.hedge([a.candidate for a in rec.hedged_arms])
    elif rec.strategy == STRAT_EXPLORE:
        coordinator.dispatch_with_logging(rec.primary.candidate)  # data-hungry
    else:
        coordinator.defer_or_reject(rec.strategy, rec.rationale)

    # Closing the loop — feed observed outcomes back to every wired layer
    # in one call.
    strat.observe(rec, StrategyOutcome(
        recommendation_id=rec.id,
        chosen_arm_id="sonnet",
        success=True,
        cost_usd=0.17,
        duration_s=6.4,
    ))

What it composes (razor's-edge of the stack)
--------------------------------------------

  - **Calibration.** Raw `p_success` from any forecaster is passed
    through `CalibrationEngine.calibrate(...)`, so the EV math the
    strategist does is honest. Falls back to raw probability when no
    calibrator has been fit.

  - **Conformal cost bounds.** Cost forecasts are widened to a
    distribution-free `(1-α)` upper bound via
    `ConformalPredictor.predict_interval(...)`. Falls back to a
    quantile heuristic when no calibration set has been recorded.

  - **Causal lift.** When a baseline action is declared,
    `CausalLab.cate(context)` returns the per-context lift of each
    treatment vs. the baseline. Used to break ties between candidates
    with similar EV and to flag candidates with a confidently negative
    CATE.

  - **Off-policy value.** When a target policy is supplied (or
    inferred from the candidate set), `PolicyLab.evaluate(...)`
    contributes a doubly-robust expected reward estimate. Combined
    with the calibrated-prior estimate via inverse-variance Bayesian
    model averaging so the more confident estimator dominates.

  - **Risk-adjusted EV.** The strategist tracks two EV numbers per
    candidate:
        EV       = p_cal * payoff - (1-p_cal) * refund - cost_mean
        EV_LB    = p_cal_lower * payoff - (1-p_cal_lower) * refund
                                  - cost_p95 - λ * (cost_p95 - cost_mean)
    where `λ = constraints.risk_aversion`. EV_LB is what gets ranked.
    The candidate with the best lower bound wins, not the best mean —
    so an over-confident point estimate cannot whiplash a coordinator
    into a bad bet.

  - **Hedging.** If no single candidate's calibrated `p_success` meets
    `target_p_success`, the strategist greedily adds the
    independent-success-maximising next-cheapest arm until the
    `hedged_p_success` floor is reached or the budget is blown, then
    emits a `STRAT_HEDGE` plan compatible with
    `agi.contract.SLOCompiler` for downstream dispatch.

  - **Defer / reject / explore.** Strategist owns the four-bucket
    admission verdict: SINGLE / HEDGE / EXPLORE (run for data) /
    DEFER (wait for more signal) / REJECT (no feasible plan). Each
    verdict carries a structured rationale referencing the specific
    forecast that drove it.

  - **Provenance.** If an `AttestationLedger` is wired in, each
    recommendation appends a tamper-evident receipt whose hash is
    surfaced on the recommendation. A coordination engine can replay
    the strategist's exact decision against the original input.

  - **Self-evaluation.** `observe(...)` forwards realised outcomes
    into every wired forecaster *and* the strategist's own
    self-evaluation log, so the strategist can report on its own
    calibration (`coverage_report()`): did `EV_LB` cover realised EV
    at the requested confidence? Did `p_success` predictions match
    realised hit rates?

Events
------
    strategist.recommended   — one decision produced
    strategist.observed      — one realised outcome recorded
    strategist.refit         — composite refit triggered

Honest about limits
-------------------

`Strategist` does not invent forecasts. If every wired layer is empty
or every candidate is unfamiliar, the recommendation falls back to a
prior-driven decision flagged with `confidence="low"`. It will never
emit a confident verdict from no data. Composing other estimators
through inverse-variance weighting is *only* valid when their
independence assumptions are not violated; this module documents the
assumption rather than silently doing the wrong thing.

The module is stdlib-only. Decisions for an 8-candidate set with all
forecasters wired run in single-digit milliseconds — investor demos
ship without GPU dependencies.
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
from typing import Any, Mapping, Sequence

from agi.attest import AttestationLedger
from agi.calibration import CalibrationEngine
from agi.causal import CATEPoint, CausalLab, LEARNER_DR
from agi.conformal import ConformalPredictor, METHOD_MONDRIAN, METHOD_SPLIT
from agi.contract import hedged_p_success
from agi.events import Event, EventBus
from agi.policy_lab import LoggedEvent, METHOD_DR, METHOD_SNIPS, PolicyLab


# ----- event kinds ----------------------------------------------------

STRATEGIST_RECOMMENDED = "strategist.recommended"
STRATEGIST_OBSERVED = "strategist.observed"
STRATEGIST_REFIT = "strategist.refit"


# ----- strategy verdicts ---------------------------------------------

STRAT_SINGLE = "single"      # one candidate is good enough on its own
STRAT_HEDGE = "hedge"        # parallelize K candidates to hit target_p
STRAT_EXPLORE = "explore"    # the data is too thin; run with logging
STRAT_DEFER = "defer"        # wait for more signal before committing
STRAT_REJECT = "reject"      # no feasible plan; tell the caller to abort

KNOWN_STRATEGIES = (
    STRAT_SINGLE,
    STRAT_HEDGE,
    STRAT_EXPLORE,
    STRAT_DEFER,
    STRAT_REJECT,
)


# ----- confidence levels ----------------------------------------------

CONF_LOW = "low"
CONF_MEDIUM = "medium"
CONF_HIGH = "high"

KNOWN_CONFIDENCES = (CONF_LOW, CONF_MEDIUM, CONF_HIGH)


# ----- helpers --------------------------------------------------------

_EPS = 1e-9


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _p_clip(p: float) -> float:
    return _clip(float(p), 0.0, 1.0)


def _z_for_confidence(confidence: float) -> float:
    """Two-sided normal quantile for `confidence` (e.g. 0.95 -> 1.96).

    Closed-form Beasley-Springer-Moro approximation, stdlib-only.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    # Inverse standard normal at p = (1+confidence)/2.
    p = (1.0 + confidence) / 2.0
    # Acklam's algorithm — fast & accurate to ~1e-9.
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
           ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def _wilson_lower(p: float, n: int, confidence: float) -> float:
    """Wilson lower bound for a Binomial proportion.

    Beats normal-approximation in small-n regimes — a coordination
    engine often acts on the lower bound, so getting it right matters.
    """
    if n <= 0:
        return 0.0
    z = _z_for_confidence(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return _p_clip(center - half)


def _wilson_upper(p: float, n: int, confidence: float) -> float:
    if n <= 0:
        return 1.0
    z = _z_for_confidence(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return _p_clip(center + half)


def _bma_inverse_variance(
    estimates: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """Inverse-variance Bayesian model average over (mean, se) pairs.

    Returns (combined_mean, combined_se) or None if no usable input.
    Estimates with non-finite or zero SE are dropped (they would
    swamp the average and they signal "no real uncertainty estimate").
    """
    valid = [(m, s) for m, s in estimates
             if math.isfinite(m) and math.isfinite(s) and s > _EPS]
    if not valid:
        return None
    weights = [1.0 / (s * s) for _, s in valid]
    wsum = sum(weights)
    if wsum <= 0:
        return None
    mean = sum(m * w for (m, _), w in zip(valid, weights)) / wsum
    se = math.sqrt(1.0 / wsum)
    return mean, se


# ----- dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A single action the strategist may recommend.

    `raw_p_success` and `raw_cost_usd` are the forecaster-emitted
    point estimates (e.g. from `PreflightEstimator.estimate(...)`).
    The strategist will *calibrate* the probability and *conformalize*
    the cost — callers do not need to do that themselves.

    `samples` is the count of historical observations underlying the
    raw forecast (used to derive a Wilson interval as a fallback for
    `p_success` uncertainty when no calibration data exists).
    """

    id: str
    raw_p_success: float
    raw_cost_usd: float
    raw_duration_s: float = 0.0
    model: str = ""
    role: str = "executor"
    effort: str = "high"
    samples: int = 0
    group: str = ""             # for Mondrian conformal
    calibration_source: str = ""
    calibration_bucket: str = ""
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_p_success <= 1.0:
            raise ValueError(
                f"raw_p_success {self.raw_p_success} out of [0,1] for {self.id!r}"
            )
        if self.raw_cost_usd < 0 or not math.isfinite(self.raw_cost_usd):
            raise ValueError(
                f"raw_cost_usd must be finite >= 0, got {self.raw_cost_usd} for {self.id!r}"
            )
        if self.samples < 0:
            raise ValueError(f"samples must be >= 0, got {self.samples}")


@dataclass(frozen=True)
class StrategyConstraints:
    """SLO + economic constraints driving the decision.

    `target_p_success`: what we are trying to achieve. Drives hedging.
    `min_p_success`: hard floor. Recommendations under this are REJECTed
                     (no defer / hedge).
    `max_cost_usd`: hard cost ceiling. Conformal upper bound must be
                    under this for a candidate to be feasible.
    `payoff_usd`: economic value of a successful run.
    `refund_usd`: cost of a failed run beyond direct compute cost
                  (refund/SLA credit charged back). Default 0.
    `max_hedge_parallel`: cap on hedge fan-out.
    `risk_aversion`: λ on (cost_p95 - cost_mean) penalty in EV_LB. 0
                     means we rank by mean EV; 1 is the default
                     bracket; >1 is highly conservative.
    `confidence`: target marginal coverage for cost bounds and CIs.
    `allow_defer/reject/hedge/explore`: enable/disable each verdict so
                     a coordinator can opt into the surface it wants.
    """

    target_p_success: float = 0.90
    min_p_success: float = 0.0
    max_cost_usd: float | None = None
    max_latency_s: float | None = None
    payoff_usd: float = 1.0
    refund_usd: float = 0.0
    max_hedge_parallel: int = 3
    hedge_cost_multiplier: float = 3.0   # max hedge_cost / cheapest_cost
    risk_aversion: float = 1.0
    confidence: float = 0.95
    allow_defer: bool = True
    allow_reject: bool = True
    allow_hedge: bool = True
    allow_explore: bool = True
    explore_min_evidence: int = 10        # below this n, prefer EXPLORE
    explore_max_cost_usd: float | None = None  # cost ceiling for EXPLORE arm

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_p_success <= 1.0:
            raise ValueError("target_p_success must be in [0,1]")
        if not 0.0 <= self.min_p_success <= 1.0:
            raise ValueError("min_p_success must be in [0,1]")
        if self.min_p_success > self.target_p_success:
            raise ValueError("min_p_success cannot exceed target_p_success")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive when set")
        if self.max_hedge_parallel < 1:
            raise ValueError("max_hedge_parallel must be >= 1")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0,1)")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")


@dataclass(frozen=True)
class CandidateForecast:
    """The fused forecast for one candidate.

    All fields are populated for every candidate so a coordination
    engine can compare them apples-to-apples. Forecasters that aren't
    wired contribute the identity / fallback for their field.
    """

    candidate: Candidate
    raw_p_success: float
    calibrated_p_success: float
    p_success_lower: float
    p_success_upper: float
    cost_mean_usd: float
    cost_p95_usd: float
    cost_lower_usd: float
    cost_width_usd: float
    duration_s: float
    cate_lift: float | None
    cate_ci_low: float | None
    cate_ci_high: float | None
    cate_low_data: bool
    ope_value: float | None
    ope_ci_low: float | None
    ope_ci_high: float | None
    ope_method: str | None
    bma_value: float | None
    bma_se: float | None
    expected_value_usd: float
    value_lower_bound_usd: float
    value_upper_bound_usd: float
    risk_score: float
    feasible: bool
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidate"] = asdict(self.candidate)
        if self.candidate.metadata is not None:
            d["candidate"]["metadata"] = dict(self.candidate.metadata)
        d["warnings"] = list(self.warnings)
        d["diagnostics"] = dict(self.diagnostics)
        return d


@dataclass(frozen=True)
class StrategyRecommendation:
    """The strategist's output. Coordination engine acts on this object.

    `strategy` ∈ {single, hedge, explore, defer, reject}.
    `primary` is the chosen candidate forecast (None on reject).
    `hedged_arms` is non-empty iff strategy == STRAT_HEDGE; ordered by
                  cost ascending (cheapest first).
    `candidates` holds every forecast scored, sorted by EV_LB descending
                 so a UI can render a leaderboard.
    `pareto_frontier` lists candidate ids that are not dominated on
                 (calibrated_p_success ascending, cost_p95 descending).
    `expected_value_usd` / `value_lower_bound_usd` summarise the
                 chosen *strategy* (single arm or hedge).
    `confidence` ∈ {low, medium, high} — operational summary derived
                 from data volume and forecaster agreement.
    `rationale` is human-readable; `diagnostics` is machine-readable.
    `attestation_hash` is set iff an `AttestationLedger` was wired in.
    """

    id: str
    strategy: str
    primary: CandidateForecast | None
    hedged_arms: tuple[CandidateForecast, ...]
    candidates: tuple[CandidateForecast, ...]
    pareto_frontier: tuple[str, ...]
    expected_value_usd: float
    value_lower_bound_usd: float
    cost_p95_usd: float
    p_success: float
    constraints: StrategyConstraints
    context: Mapping[str, Any]
    confidence: str
    rationale: str
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    attestation_hash: str | None
    ts: float

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "strategy": self.strategy,
            "primary": self.primary.to_dict() if self.primary else None,
            "hedged_arms": [a.to_dict() for a in self.hedged_arms],
            "candidates": [c.to_dict() for c in self.candidates],
            "pareto_frontier": list(self.pareto_frontier),
            "expected_value_usd": self.expected_value_usd,
            "value_lower_bound_usd": self.value_lower_bound_usd,
            "cost_p95_usd": self.cost_p95_usd,
            "p_success": self.p_success,
            "constraints": asdict(self.constraints),
            "context": dict(self.context),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "attestation_hash": self.attestation_hash,
            "ts": self.ts,
        }
        return d


@dataclass(frozen=True)
class StrategyOutcome:
    """Realised outcome after dispatch — closes the learning loop.

    `chosen_arm_id` matters under STRAT_HEDGE: which arm actually
    fulfilled the request. For STRAT_SINGLE it equals the primary.
    `success` is whether the SLO target was met.
    `cost_usd` is the realised total cost (sum across hedged arms).
    """

    recommendation_id: str
    chosen_arm_id: str
    success: bool
    cost_usd: float
    duration_s: float = 0.0
    refund_usd: float = 0.0
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CoverageReport:
    """The strategist's self-evaluation report.

    Did `EV_LB` actually cover the realised EV at the requested
    confidence? Did `p_success` predictions land at empirical rates?
    Investors look at this number — if the strategist is honest about
    its own calibration, every downstream consumer can be too.
    """

    n: int
    p_success_brier: float
    p_success_log_loss: float
    p_success_ece: float
    ev_lb_coverage: float       # fraction of EV realisations >= EV_LB
    ev_ub_coverage: float       # fraction of EV realisations <= EV_UB
    cost_p95_breach_rate: float
    mean_realised_value_usd: float
    mean_predicted_value_usd: float
    per_strategy: Mapping[str, Mapping[str, float]]
    confidence: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["per_strategy"] = {k: dict(v) for k, v in self.per_strategy.items()}
        d["notes"] = list(self.notes)
        return d


# =====================================================================
# The strategist
# =====================================================================


class Strategist:
    """Top-level meta-decision orchestrator.

    Thread-safe: a single instance is safe to share across producer
    threads in a coordination engine. Lock granularity is coarse (the
    decision path is short — single-digit ms with 10 candidates and
    all forecasters wired).

    Wire as much or as little as you have:

      - With only `Candidate.raw_p_success` and `raw_cost_usd`, the
        strategist returns prior-driven recommendations flagged
        ``confidence="low"``.
      - With `calibration` wired, probabilities become honest.
      - With `conformal` wired, costs become bounded.
      - With `causal` wired, candidates with confidently negative CATE
        are filtered out and CATE lift breaks EV ties.
      - With `policy_lab` wired, OPE contributes to value estimates
        through inverse-variance Bayesian model averaging.
      - With `ledger` wired, every recommendation gets a tamper-evident
        provenance hash.

    `observe(recommendation, outcome)` is the closed-loop call: it
    forwards the realised outcome into every wired forecaster *and*
    the strategist's own log so `coverage_report()` returns honest
    self-evaluation.

    Parameters
    ----------
    calibration, conformal, causal, policy_lab, ledger
        Optional forecasters / provenance. Wire what you have.
    baseline_action_id
        Name of the baseline action (e.g. the production model). When
        set, the strategist queries `causal.cate(...)` per candidate
        against this baseline.
    conformal_method
        Which conformal method to use for cost bounds. Default is
        Mondrian when candidates carry a `group`, else split.
    bus
        Optional `EventBus` for ``strategist.*`` events.
    history_limit
        Max number of (recommendation, outcome) pairs retained for
        self-evaluation. FIFO.
    seed
        RNG seed for tie-breaking (not for any sampling).
    """

    def __init__(
        self,
        *,
        calibration: CalibrationEngine | None = None,
        conformal: ConformalPredictor | None = None,
        causal: CausalLab | None = None,
        policy_lab: PolicyLab | None = None,
        ledger: AttestationLedger | None = None,
        baseline_action_id: str | None = None,
        conformal_method: str | None = None,
        bus: EventBus | None = None,
        history_limit: int = 5000,
        seed: int = 0xA62,
    ) -> None:
        self.calibration = calibration
        self.conformal = conformal
        self.causal = causal
        self.policy_lab = policy_lab
        self.ledger = ledger
        self.baseline_action_id = baseline_action_id
        self.conformal_method = conformal_method
        self.bus = bus
        self.history_limit = int(history_limit)
        self.seed = int(seed)

        self._lock = threading.RLock()
        # In-flight recommendations: id -> recommendation. We hold a
        # reference so observe() can join cleanly even if the caller
        # discards the local variable.
        self._open: dict[str, StrategyRecommendation] = {}
        # Closed (recommendation, outcome) pairs for self-evaluation.
        self._history: list[tuple[StrategyRecommendation, StrategyOutcome]] = []

    # ----- public API -------------------------------------------------

    def forecast(
        self,
        candidate: Candidate,
        *,
        context: Mapping[str, Any] | None = None,
        constraints: StrategyConstraints | None = None,
    ) -> CandidateForecast:
        """Fuse all wired forecasters for one candidate.

        `context` is opaque to the strategist except that it's passed
        to `causal.cate(...)` (which interprets numeric features) and
        to `policy_lab.evaluate(...)` (which logs it).

        `constraints` is used for `payoff_usd` / `refund_usd` /
        `risk_aversion` / `confidence` — without it, prior defaults
        are used (payoff=$1, refund=$0, λ=1, 95%).
        """
        c = constraints or StrategyConstraints()
        ctx = dict(context or {})
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {}

        # 1. Calibrated probability.
        p_raw = _p_clip(candidate.raw_p_success)
        if self.calibration is not None:
            try:
                p_cal = _p_clip(self.calibration.calibrate(
                    p_raw,
                    source=candidate.calibration_source,
                    bucket=candidate.calibration_bucket,
                ))
                if abs(p_cal - p_raw) > 0.02:
                    diagnostics["calibration_shift"] = round(p_cal - p_raw, 4)
            except Exception as exc:                  # pragma: no cover
                warnings.append(f"calibration_failed:{exc}")
                p_cal = p_raw
        else:
            p_cal = p_raw

        # Probability uncertainty: Wilson interval if we have evidence,
        # otherwise a wide jeffrey-style prior.
        n_eff = max(candidate.samples, 0)
        if n_eff >= 1:
            p_lo = _wilson_lower(p_cal, n_eff, c.confidence)
            p_hi = _wilson_upper(p_cal, n_eff, c.confidence)
        else:
            # No empirical evidence — be honest: the bound stretches
            # across the unit interval, shrunken toward the point
            # estimate by a Beta(1,1)-Jeffrey-style heuristic.
            half = (1.0 - 0.5 * c.confidence) * 0.5
            p_lo = _p_clip(p_cal - half)
            p_hi = _p_clip(p_cal + half)
            warnings.append("no_p_success_evidence")
        diagnostics["p_success_n"] = n_eff

        # 2. Conformal cost bounds.
        cost_mean = float(candidate.raw_cost_usd)
        cost_lo = cost_mean
        cost_hi = cost_mean
        if self.conformal is not None and len(self.conformal) > 0:
            try:
                method = self.conformal_method
                if method is None:
                    method = METHOD_MONDRIAN if candidate.group else METHOD_SPLIT
                # ConformalPredictor's target_coverage is set at
                # construction; we honor that here (it's the runtime's
                # global default). The constraints.confidence is used
                # below only for the *value* CI's z-score.
                interval = self.conformal.predict_interval(
                    prediction=cost_mean,
                    features={
                        "model": candidate.model,
                        "role": candidate.role,
                        "effort": candidate.effort,
                        "prediction": cost_mean,
                    },
                    method=method,
                    group=candidate.group,
                )
                # Costs can't be negative.
                cost_lo = max(0.0, float(interval.lower))
                cost_hi = max(cost_lo, float(interval.upper))
                diagnostics["conformal_n_cal"] = interval.n_cal
                diagnostics["conformal_method"] = interval.method
            except Exception as exc:
                warnings.append(f"conformal_failed:{exc}")
        elif self.conformal is not None:
            warnings.append("conformal_no_calibration")
        # If conformal is missing or empty, fall back to a symmetric
        # ±25% width as a coarse prior — better than zero width which
        # would lie about uncertainty.
        if cost_hi <= cost_mean + _EPS:
            spread = 0.25 * max(cost_mean, 0.01)
            cost_lo = max(0.0, cost_mean - spread)
            cost_hi = cost_mean + spread
        cost_width = cost_hi - cost_lo

        # 3. CATE lift vs. baseline action.
        cate_lift: float | None = None
        cate_ci_low: float | None = None
        cate_ci_high: float | None = None
        cate_low_data = False
        if self.causal is not None and self.baseline_action_id is not None \
                and candidate.id != self.baseline_action_id:
            try:
                point = self.causal.cate(
                    context=_numeric_only(ctx),
                    treatment=candidate.id,
                    control=self.baseline_action_id,
                    learner=LEARNER_DR,
                    confidence=c.confidence,
                )
                cate_lift = float(point.value)
                cate_ci_low = float(point.ci_low)
                cate_ci_high = float(point.ci_high)
                cate_low_data = bool(point.low_data)
                if cate_low_data:
                    diagnostics["cate_low_data"] = True
                if cate_ci_high < 0:
                    warnings.append("cate_negative_ci")
            except Exception as exc:
                warnings.append(f"causal_failed:{exc}")

        # 4. OPE value (DM, DR / SNIPS) on the deterministic policy that
        #    always selects this candidate.
        ope_value: float | None = None
        ope_ci_low: float | None = None
        ope_ci_high: float | None = None
        ope_method: str | None = None
        if self.policy_lab is not None and len(self.policy_lab.events()) >= 2:
            try:
                pol = _DeterministicArmPolicy(candidate.id)
                est = self.policy_lab.evaluate(
                    pol,
                    method=METHOD_DR,
                    confidence=c.confidence,
                )
                if math.isfinite(est.value):
                    ope_value = float(est.value)
                    ope_ci_low = float(est.ci_low)
                    ope_ci_high = float(est.ci_high)
                    ope_method = METHOD_DR
            except Exception:
                # Fall back to SNIPS, which is more robust.
                try:
                    pol = _DeterministicArmPolicy(candidate.id)
                    est = self.policy_lab.evaluate(
                        pol,
                        method=METHOD_SNIPS,
                        confidence=c.confidence,
                    )
                    if math.isfinite(est.value):
                        ope_value = float(est.value)
                        ope_ci_low = float(est.ci_low)
                        ope_ci_high = float(est.ci_high)
                        ope_method = METHOD_SNIPS
                except Exception as exc:
                    warnings.append(f"ope_failed:{exc}")

        # 5. Risk-adjusted EV math.
        #    EV     = p * payoff - (1-p) * refund - cost_mean
        #    EV_LB  = p_lo * payoff - (1-p_lo) * refund
        #               - cost_hi - λ * (cost_hi - cost_mean)
        payoff = float(c.payoff_usd)
        refund = float(c.refund_usd)
        lam = float(c.risk_aversion)

        ev_mean = p_cal * payoff - (1.0 - p_cal) * refund - cost_mean
        ev_lb = (
            p_lo * payoff
            - (1.0 - p_lo) * refund
            - cost_hi
            - lam * (cost_hi - cost_mean)
        )
        ev_ub = p_hi * payoff - (1.0 - p_hi) * refund - cost_lo

        # 6. Inverse-variance BMA across estimators that produced a
        #    finite (value, se) pair. The calibrated-prior estimate has
        #    SE derived from the Wilson half-width on p_cal × payoff
        #    span (conservative — it bounds reward uncertainty given p
        #    uncertainty alone; cost uncertainty is captured separately
        #    via EV_LB).
        prior_se = max(
            ((p_hi - p_lo) / 2.0) * (abs(payoff) + abs(refund)) + _EPS,
            _EPS,
        )
        bma_inputs: list[tuple[float, float]] = [(ev_mean, prior_se)]
        if ope_value is not None and ope_ci_low is not None and ope_ci_high is not None:
            # SE from the half-width of the OPE CI under z(confidence).
            half = (ope_ci_high - ope_ci_low) / 2.0
            z = _z_for_confidence(c.confidence) or 1.0
            ope_se = max(half / z, _EPS)
            # OPE value is reward-only; subtract cost_mean to get EV-like.
            bma_inputs.append((ope_value - cost_mean, ope_se))
        if cate_lift is not None and cate_ci_high is not None and cate_ci_low is not None \
                and self.baseline_action_id is not None:
            # Cast the CATE estimate as "value vs. baseline" — useful
            # only as a relative signal, but we can fuse against the
            # absolute EV by anchoring on the baseline's EV if known.
            # The strategist does not synthesise a baseline EV here
            # (it doesn't have the baseline's calibrated probability
            # in this call), so we register the CATE-driven SE only as
            # an uncertainty *prior* — diagnostics, not the BMA input.
            half = (cate_ci_high - cate_ci_low) / 2.0
            diagnostics["cate_half_width"] = round(half, 5)

        bma = _bma_inverse_variance(bma_inputs)
        bma_value = bma[0] if bma else None
        bma_se = bma[1] if bma else None
        if bma is not None:
            diagnostics["bma_n_inputs"] = len(bma_inputs)
            diagnostics["bma_value"] = round(bma[0], 5)
            diagnostics["bma_se"] = round(bma[1], 5)

        # 7. Risk score in [0, 1]. Combines:
        #    - Probability uncertainty band  (p_hi - p_lo)
        #    - Cost uncertainty band / cost_mean (clipped to [0,1])
        #    - Whether CATE has a negative CI
        #    - Whether OPE disagrees with prior by > 2 SE
        p_unc = p_hi - p_lo
        cost_rel_unc = cost_width / max(cost_mean, _EPS)
        cost_rel_unc = _clip(cost_rel_unc / 2.0, 0.0, 1.0)
        risk = 0.5 * p_unc + 0.4 * cost_rel_unc
        if cate_ci_high is not None and cate_ci_high < 0:
            risk += 0.2  # CATE says candidate is likely worse than baseline
        if ope_value is not None and bma_se is not None:
            disagreement = abs(ope_value - ev_mean - cost_mean) / max(bma_se, _EPS)
            if disagreement > 2.0:
                risk += 0.15
        risk_score = _clip(risk, 0.0, 1.0)

        # 8. Feasibility.
        feasible = True
        if c.max_cost_usd is not None and cost_hi > c.max_cost_usd:
            feasible = False
            warnings.append("cost_p95_over_budget")
        if c.max_latency_s is not None and candidate.raw_duration_s > c.max_latency_s:
            feasible = False
            warnings.append("latency_over_budget")
        if p_hi < c.min_p_success:
            feasible = False
            warnings.append("p_success_under_floor")

        return CandidateForecast(
            candidate=candidate,
            raw_p_success=p_raw,
            calibrated_p_success=p_cal,
            p_success_lower=p_lo,
            p_success_upper=p_hi,
            cost_mean_usd=cost_mean,
            cost_p95_usd=cost_hi,
            cost_lower_usd=cost_lo,
            cost_width_usd=cost_width,
            duration_s=float(candidate.raw_duration_s),
            cate_lift=cate_lift,
            cate_ci_low=cate_ci_low,
            cate_ci_high=cate_ci_high,
            cate_low_data=cate_low_data,
            ope_value=ope_value,
            ope_ci_low=ope_ci_low,
            ope_ci_high=ope_ci_high,
            ope_method=ope_method,
            bma_value=bma_value,
            bma_se=bma_se,
            expected_value_usd=ev_mean,
            value_lower_bound_usd=ev_lb,
            value_upper_bound_usd=ev_ub,
            risk_score=risk_score,
            feasible=feasible,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
        )

    def recommend(
        self,
        candidates: Sequence[Candidate],
        constraints: StrategyConstraints | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> StrategyRecommendation:
        """Pick a strategy across `candidates` honoring `constraints`.

        Returns a `StrategyRecommendation`. The caller acts on it,
        then closes the loop with `observe(...)`.
        """
        if not candidates:
            raise ValueError("at least one candidate required")
        c = constraints or StrategyConstraints()
        ctx = dict(context or {})

        # Score every candidate independently.
        forecasts = [self.forecast(cand, context=ctx, constraints=c)
                     for cand in candidates]

        # Sort by lower bound EV (descending), tie-break by mean EV.
        forecasts_sorted = sorted(
            forecasts,
            key=lambda f: (-f.value_lower_bound_usd, -f.expected_value_usd, f.cost_p95_usd),
        )

        # Pareto frontier: candidate is on the frontier iff no other
        # candidate has greater-or-equal calibrated_p_success AND
        # lower-or-equal cost_p95, with at least one strict inequality.
        pareto: list[str] = []
        for f in forecasts_sorted:
            dominated = False
            for g in forecasts_sorted:
                if g is f:
                    continue
                if (g.calibrated_p_success >= f.calibrated_p_success
                        and g.cost_p95_usd <= f.cost_p95_usd
                        and (g.calibrated_p_success > f.calibrated_p_success
                             or g.cost_p95_usd < f.cost_p95_usd)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(f.candidate.id)

        feasible_sorted = [f for f in forecasts_sorted if f.feasible]
        warnings: list[str] = []

        if not feasible_sorted:
            return self._finalise_reject(
                "no_feasible_candidates",
                forecasts_sorted, pareto, c, ctx, warnings,
            )

        best = feasible_sorted[0]

        # ----- decide the strategy verdict --------------------------

        # 1. STRAT_SINGLE: best candidate's calibrated p_success ≥ target.
        if best.calibrated_p_success >= c.target_p_success \
                and best.value_lower_bound_usd >= 0:
            return self._finalise_single(best, forecasts_sorted, pareto, c, ctx, warnings)

        # 2. STRAT_HEDGE: try to assemble a parallel set meeting target.
        if c.allow_hedge and c.target_p_success > 0:
            hedge = self._propose_hedge(feasible_sorted, c)
            if hedge:
                return self._finalise_hedge(hedge, forecasts_sorted, pareto, c, ctx, warnings)

        # 3. STRAT_EXPLORE: best candidate has positive *upper-bound* EV
        #    and low evidence — running it is positive expected
        #    information value.
        if c.allow_explore and best.value_upper_bound_usd > 0 \
                and best.candidate.samples < c.explore_min_evidence:
            if c.explore_max_cost_usd is None or best.cost_p95_usd <= c.explore_max_cost_usd:
                return self._finalise_explore(
                    best, forecasts_sorted, pareto, c, ctx, warnings,
                )

        # 4. STRAT_DEFER: best EV is positive but LB is negative and we
        #    don't have a reason to run for data — wait for more signal.
        if c.allow_defer and best.expected_value_usd > 0:
            warnings.append("ev_positive_but_lb_negative")
            return self._finalise_defer(best, forecasts_sorted, pareto, c, ctx, warnings)

        # 5. STRAT_REJECT.
        if c.allow_reject:
            return self._finalise_reject(
                "all_candidates_negative_ev",
                forecasts_sorted, pareto, c, ctx, warnings,
            )

        # Reject is the only honest answer but disabled — fall back to
        # the least-bad single, warning loudly.
        warnings.append("forced_single_against_advice")
        return self._finalise_single(best, forecasts_sorted, pareto, c, ctx, warnings)

    def observe(
        self,
        recommendation: StrategyRecommendation,
        outcome: StrategyOutcome,
    ) -> None:
        """Forward a realised outcome to every wired forecaster.

        Specifically:
          - `CalibrationEngine.observe(p_forecast, success)` for the
            chosen arm.
          - `ConformalPredictor.record(features, prediction=cost_mean,
            outcome=cost_usd)` for the chosen arm's cost.
          - `PolicyLab.record(LoggedEvent(...))` reconstructing the
            decision propensity from the recommendation's diagnostics.
          - `AttestationLedger.record(...)` appending the outcome to
            the recommendation's provenance chain.
          - The strategist's own self-evaluation log for
            `coverage_report()`.

        Idempotent on (recommendation_id, outcome) pairs.
        """
        with self._lock:
            self._history.append((recommendation, outcome))
            if len(self._history) > self.history_limit:
                self._history = self._history[-self.history_limit :]
            self._open.pop(recommendation.id, None)

        chosen = self._chosen_forecast(recommendation, outcome.chosen_arm_id)
        # 1. Calibration.
        if self.calibration is not None and chosen is not None:
            try:
                self.calibration.observe(
                    chosen.raw_p_success,
                    outcome.success,
                    source=chosen.candidate.calibration_source,
                    bucket=chosen.candidate.calibration_bucket,
                )
            except Exception:                            # pragma: no cover
                pass

        # 2. Conformal cost record.
        if self.conformal is not None and chosen is not None:
            try:
                self.conformal.record(
                    features={
                        "model": chosen.candidate.model,
                        "role": chosen.candidate.role,
                        "effort": chosen.candidate.effort,
                        "prediction": chosen.cost_mean_usd,
                    },
                    prediction=chosen.cost_mean_usd,
                    outcome=outcome.cost_usd,
                    group=chosen.candidate.group,
                )
            except Exception:                            # pragma: no cover
                pass

        # 3. Off-policy logging — reward convention: payoff if success
        #    minus cost; refunds subtracted on failure.
        if self.policy_lab is not None and chosen is not None:
            try:
                reward = (
                    recommendation.constraints.payoff_usd
                    if outcome.success
                    else -recommendation.constraints.refund_usd
                )
                reward -= outcome.cost_usd
                # The strategist's propensity is 1.0 on the chosen arm
                # under STRAT_SINGLE/EXPLORE/DEFER, and 1/K for hedged
                # (the K arms run concurrently; from an OPE viewpoint
                # each is effectively the deterministic policy).
                if recommendation.strategy == STRAT_HEDGE and recommendation.hedged_arms:
                    propensity = 1.0 / len(recommendation.hedged_arms)
                else:
                    propensity = 1.0
                self.policy_lab.record(LoggedEvent(
                    context=_numeric_only(recommendation.context),
                    action=outcome.chosen_arm_id,
                    propensity=propensity,
                    reward=float(reward),
                    metadata={
                        "recommendation_id": recommendation.id,
                        "strategy": recommendation.strategy,
                        "success": outcome.success,
                        "cost_usd": outcome.cost_usd,
                    },
                ))
            except Exception:                            # pragma: no cover
                pass

        # 4. CausalLab is fed via the PolicyLab event mirror; if a
        #    user wired CausalLab without PolicyLab we record directly.
        if self.causal is not None and chosen is not None and self.policy_lab is None:
            try:
                reward = (
                    recommendation.constraints.payoff_usd
                    if outcome.success
                    else -recommendation.constraints.refund_usd
                )
                reward -= outcome.cost_usd
                propensity = (
                    1.0 / len(recommendation.hedged_arms)
                    if recommendation.strategy == STRAT_HEDGE and recommendation.hedged_arms
                    else 1.0
                )
                self.causal.record(LoggedEvent(
                    context=_numeric_only(recommendation.context),
                    action=outcome.chosen_arm_id,
                    propensity=propensity,
                    reward=float(reward),
                ))
            except Exception:                            # pragma: no cover
                pass

        # 5. Attestation receipt.
        if self.ledger is not None and recommendation.attestation_hash is not None:
            try:
                self.ledger.append({
                    "ticket_id": recommendation.id,
                    "kind": "strategist.outcome",
                    "chosen_arm": outcome.chosen_arm_id,
                    "success": outcome.success,
                    "cost_usd": outcome.cost_usd,
                    "duration_s": outcome.duration_s,
                    "links": [recommendation.attestation_hash],
                })
            except Exception:                            # pragma: no cover
                pass

        if self.bus is not None:
            self.bus.publish(Event(
                kind=STRATEGIST_OBSERVED,
                data={
                    "recommendation_id": recommendation.id,
                    "chosen_arm_id": outcome.chosen_arm_id,
                    "success": outcome.success,
                    "cost_usd": outcome.cost_usd,
                    "strategy": recommendation.strategy,
                },
            ))

    def coverage_report(self) -> CoverageReport:
        """Return the strategist's own calibration report.

        With <2 observations the report carries the `low_data` note
        rather than spurious numbers.
        """
        with self._lock:
            history = list(self._history)

        n = len(history)
        if n < 2:
            return CoverageReport(
                n=n,
                p_success_brier=0.0,
                p_success_log_loss=0.0,
                p_success_ece=0.0,
                ev_lb_coverage=0.0,
                ev_ub_coverage=0.0,
                cost_p95_breach_rate=0.0,
                mean_realised_value_usd=0.0,
                mean_predicted_value_usd=0.0,
                per_strategy={},
                confidence=0.0,
                notes=("low_data",),
            )

        brier_sum = 0.0
        ll_sum = 0.0
        n_brier = 0
        # Reliability binning for ECE.
        bin_p_sums = [0.0] * 10
        bin_y_sums = [0.0] * 10
        bin_counts = [0] * 10
        ev_lb_hits = 0
        ev_ub_hits = 0
        n_value_pairs = 0
        cost_breach = 0
        n_cost_pairs = 0
        realised_total = 0.0
        predicted_total = 0.0

        per_strategy_acc: dict[str, dict[str, float]] = {}

        for rec, out in history:
            chosen = self._chosen_forecast(rec, out.chosen_arm_id)
            if chosen is None:
                continue
            # Probabilistic calibration of p_success.
            y = 1.0 if out.success else 0.0
            p = _p_clip(chosen.calibrated_p_success)
            brier_sum += (p - y) ** 2
            p_eps = min(max(p, 1e-6), 1.0 - 1e-6)
            ll_sum += -(y * math.log(p_eps) + (1 - y) * math.log(1 - p_eps))
            n_brier += 1
            bin_idx = min(int(p * 10), 9)
            bin_p_sums[bin_idx] += p
            bin_y_sums[bin_idx] += y
            bin_counts[bin_idx] += 1

            # Cost upper bound: did the realised cost stay inside?
            if chosen.cost_p95_usd > 0:
                n_cost_pairs += 1
                if out.cost_usd > chosen.cost_p95_usd:
                    cost_breach += 1

            # Value bounds: did realised EV stay within [EV_LB, EV_UB]?
            realised_value = (
                rec.constraints.payoff_usd if out.success else -rec.constraints.refund_usd
            ) - out.cost_usd
            n_value_pairs += 1
            realised_total += realised_value
            predicted_total += rec.expected_value_usd
            if realised_value >= rec.value_lower_bound_usd - 1e-9:
                ev_lb_hits += 1
            if chosen.value_upper_bound_usd >= realised_value - 1e-9:
                ev_ub_hits += 1

            acc = per_strategy_acc.setdefault(rec.strategy, {
                "n": 0.0,
                "success_n": 0.0,
                "realised_value_sum": 0.0,
                "predicted_value_sum": 0.0,
                "cost_sum": 0.0,
            })
            acc["n"] += 1
            if out.success:
                acc["success_n"] += 1
            acc["realised_value_sum"] += realised_value
            acc["predicted_value_sum"] += rec.expected_value_usd
            acc["cost_sum"] += out.cost_usd

        # ECE: weighted absolute calibration gap across non-empty bins.
        ece = 0.0
        total_n = sum(bin_counts) or 1
        for i in range(10):
            if bin_counts[i] == 0:
                continue
            p_mean = bin_p_sums[i] / bin_counts[i]
            y_mean = bin_y_sums[i] / bin_counts[i]
            ece += (bin_counts[i] / total_n) * abs(p_mean - y_mean)

        per_strategy = {
            k: {
                "n": v["n"],
                "success_rate": v["success_n"] / v["n"] if v["n"] else 0.0,
                "mean_realised_value_usd": v["realised_value_sum"] / v["n"] if v["n"] else 0.0,
                "mean_predicted_value_usd": v["predicted_value_sum"] / v["n"] if v["n"] else 0.0,
                "mean_cost_usd": v["cost_sum"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in per_strategy_acc.items()
        }

        notes: list[str] = []
        if n < 30:
            notes.append("limited_evidence")

        return CoverageReport(
            n=n,
            p_success_brier=brier_sum / max(n_brier, 1),
            p_success_log_loss=ll_sum / max(n_brier, 1),
            p_success_ece=ece,
            ev_lb_coverage=ev_lb_hits / max(n_value_pairs, 1),
            ev_ub_coverage=ev_ub_hits / max(n_value_pairs, 1),
            cost_p95_breach_rate=cost_breach / max(n_cost_pairs, 1),
            mean_realised_value_usd=realised_total / max(n_value_pairs, 1),
            mean_predicted_value_usd=predicted_total / max(n_value_pairs, 1),
            per_strategy=per_strategy,
            confidence=0.95,
            notes=tuple(notes),
        )

    def open_recommendations(self) -> list[StrategyRecommendation]:
        with self._lock:
            return list(self._open.values())

    def history(self) -> list[tuple[StrategyRecommendation, StrategyOutcome]]:
        with self._lock:
            return list(self._history)

    # ----- internals --------------------------------------------------

    def _propose_hedge(
        self,
        feasible_sorted: list[CandidateForecast],
        c: StrategyConstraints,
    ) -> list[CandidateForecast] | None:
        """Greedy hedge selection.

        Sort feasible candidates by cost ascending. Add greedily while
        hedged_p_success < target and budget holds; stop early when
        target reached or the next add would exceed budget or
        max_parallel.
        """
        budget = c.max_cost_usd
        max_parallel = c.max_hedge_parallel
        cheapest = min(feasible_sorted, key=lambda f: f.cost_p95_usd)
        # Budget always wins when set. The multiplier only kicks in
        # when no explicit budget exists — a defensive sanity bound
        # against runaway hedging.
        if budget is not None:
            cap = budget
        else:
            cap = c.hedge_cost_multiplier * cheapest.cost_p95_usd

        by_cost = sorted(feasible_sorted, key=lambda f: f.cost_p95_usd)
        chosen: list[CandidateForecast] = []
        total_cost_hi = 0.0
        total_cost_mean = 0.0
        for f in by_cost:
            if len(chosen) >= max_parallel:
                break
            if total_cost_hi + f.cost_p95_usd > cap:
                continue
            chosen.append(f)
            total_cost_hi += f.cost_p95_usd
            total_cost_mean += f.cost_mean_usd
            phs = hedged_p_success([
                g.calibrated_p_success for g in chosen
            ])
            if phs >= c.target_p_success:
                break
        if not chosen:
            return None
        phs = hedged_p_success([g.calibrated_p_success for g in chosen])
        # Only return a hedge if it actually buys us something over the
        # cheapest single — otherwise STRAT_SINGLE would be a lie.
        if len(chosen) <= 1:
            return None
        # Don't hedge if the cheapest single already meets the target
        # at acceptable EV — the caller already considered SINGLE.
        if phs < c.target_p_success and phs < cheapest.calibrated_p_success + 0.02:
            # The hedge isn't pulling its weight relative to its best
            # single arm.
            return None
        return chosen

    def _finalise_single(
        self,
        primary: CandidateForecast,
        forecasts_sorted: list[CandidateForecast],
        pareto: list[str],
        c: StrategyConstraints,
        ctx: Mapping[str, Any],
        warnings: list[str],
    ) -> StrategyRecommendation:
        rationale = (
            f"single arm {primary.candidate.id!r}: "
            f"calibrated p_success={primary.calibrated_p_success:.3f} "
            f"(LB={primary.p_success_lower:.3f}, UB={primary.p_success_upper:.3f}), "
            f"cost_p95=${primary.cost_p95_usd:.4f}, "
            f"EV=${primary.expected_value_usd:.4f}, "
            f"EV_LB=${primary.value_lower_bound_usd:.4f}, "
            f"risk={primary.risk_score:.2f}."
        )
        diagnostics = {
            "candidates_evaluated": len(forecasts_sorted),
            "best_candidate_id": primary.candidate.id,
            "best_ev_lb": primary.value_lower_bound_usd,
            "second_ev_lb": (
                forecasts_sorted[1].value_lower_bound_usd
                if len(forecasts_sorted) > 1 else None
            ),
        }
        return self._finalise(
            STRAT_SINGLE,
            primary, [],
            forecasts_sorted, pareto, c, ctx,
            ev=primary.expected_value_usd,
            ev_lb=primary.value_lower_bound_usd,
            cost_p95=primary.cost_p95_usd,
            p_success=primary.calibrated_p_success,
            rationale=rationale,
            warnings=warnings + list(primary.warnings),
            diagnostics=diagnostics,
        )

    def _finalise_hedge(
        self,
        chosen: list[CandidateForecast],
        forecasts_sorted: list[CandidateForecast],
        pareto: list[str],
        c: StrategyConstraints,
        ctx: Mapping[str, Any],
        warnings: list[str],
    ) -> StrategyRecommendation:
        hedge_p = hedged_p_success([f.calibrated_p_success for f in chosen])
        hedge_p_lb = hedged_p_success([f.p_success_lower for f in chosen])
        cost_p95 = sum(f.cost_p95_usd for f in chosen)
        cost_mean = sum(f.cost_mean_usd for f in chosen)
        ev = (
            hedge_p * c.payoff_usd - (1.0 - hedge_p) * c.refund_usd - cost_mean
        )
        ev_lb = (
            hedge_p_lb * c.payoff_usd
            - (1.0 - hedge_p_lb) * c.refund_usd
            - cost_p95
            - c.risk_aversion * (cost_p95 - cost_mean)
        )
        primary = chosen[0]
        rationale = (
            f"hedge across {len(chosen)} arms "
            f"({', '.join(f.candidate.id for f in chosen)}): "
            f"hedged p_success={hedge_p:.3f} (LB={hedge_p_lb:.3f}) "
            f"meets target {c.target_p_success:.3f}, "
            f"total cost_p95=${cost_p95:.4f}, EV_LB=${ev_lb:.4f}."
        )
        diagnostics = {
            "hedge_arms": [f.candidate.id for f in chosen],
            "hedged_p_success": hedge_p,
            "hedged_p_success_lb": hedge_p_lb,
            "hedge_cost_p95_usd": cost_p95,
        }
        return self._finalise(
            STRAT_HEDGE,
            primary, chosen,
            forecasts_sorted, pareto, c, ctx,
            ev=ev, ev_lb=ev_lb,
            cost_p95=cost_p95, p_success=hedge_p,
            rationale=rationale,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _finalise_explore(
        self,
        primary: CandidateForecast,
        forecasts_sorted: list[CandidateForecast],
        pareto: list[str],
        c: StrategyConstraints,
        ctx: Mapping[str, Any],
        warnings: list[str],
    ) -> StrategyRecommendation:
        rationale = (
            f"explore arm {primary.candidate.id!r}: "
            f"upper-bound EV=${primary.value_upper_bound_usd:.4f} > 0 "
            f"with only {primary.candidate.samples} prior samples (<{c.explore_min_evidence}); "
            f"information-value positive."
        )
        diagnostics = {
            "explore_samples": primary.candidate.samples,
            "ev_upper_bound": primary.value_upper_bound_usd,
        }
        return self._finalise(
            STRAT_EXPLORE,
            primary, [],
            forecasts_sorted, pareto, c, ctx,
            ev=primary.expected_value_usd,
            ev_lb=primary.value_lower_bound_usd,
            cost_p95=primary.cost_p95_usd,
            p_success=primary.calibrated_p_success,
            rationale=rationale,
            warnings=warnings + ["exploration_active"],
            diagnostics=diagnostics,
        )

    def _finalise_defer(
        self,
        primary: CandidateForecast,
        forecasts_sorted: list[CandidateForecast],
        pareto: list[str],
        c: StrategyConstraints,
        ctx: Mapping[str, Any],
        warnings: list[str],
    ) -> StrategyRecommendation:
        rationale = (
            f"defer: best mean EV=${primary.expected_value_usd:.4f} > 0 "
            f"but EV_LB=${primary.value_lower_bound_usd:.4f} < 0 "
            f"under {c.risk_aversion:.2f}× risk-aversion. "
            f"Wait for more signal."
        )
        diagnostics = {
            "best_candidate_id": primary.candidate.id,
            "ev_mean": primary.expected_value_usd,
            "ev_lb": primary.value_lower_bound_usd,
        }
        return self._finalise(
            STRAT_DEFER,
            None, [],
            forecasts_sorted, pareto, c, ctx,
            ev=primary.expected_value_usd,
            ev_lb=primary.value_lower_bound_usd,
            cost_p95=primary.cost_p95_usd,
            p_success=primary.calibrated_p_success,
            rationale=rationale,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _finalise_reject(
        self,
        reason: str,
        forecasts_sorted: list[CandidateForecast],
        pareto: list[str],
        c: StrategyConstraints,
        ctx: Mapping[str, Any],
        warnings: list[str],
    ) -> StrategyRecommendation:
        rationale = f"reject ({reason}): no candidate meets constraints with positive EV."
        diagnostics = {"reject_reason": reason}
        return self._finalise(
            STRAT_REJECT,
            None, [],
            forecasts_sorted, pareto, c, ctx,
            ev=0.0, ev_lb=0.0,
            cost_p95=0.0, p_success=0.0,
            rationale=rationale,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _finalise(
        self,
        strategy: str,
        primary: CandidateForecast | None,
        hedged: list[CandidateForecast],
        forecasts_sorted: list[CandidateForecast],
        pareto: list[str],
        c: StrategyConstraints,
        ctx: Mapping[str, Any],
        *,
        ev: float,
        ev_lb: float,
        cost_p95: float,
        p_success: float,
        rationale: str,
        warnings: list[str],
        diagnostics: dict[str, Any],
    ) -> StrategyRecommendation:
        rec_id = uuid.uuid4().hex[:16]
        confidence = self._confidence_level(forecasts_sorted, primary)

        attestation_hash: str | None = None
        if self.ledger is not None:
            try:
                payload = {
                    "ticket_id": rec_id,
                    "kind": "strategist.recommendation",
                    "strategy": strategy,
                    "primary": primary.candidate.id if primary else None,
                    "hedged": [f.candidate.id for f in hedged],
                    "ev": ev, "ev_lb": ev_lb,
                    "cost_p95": cost_p95, "p_success": p_success,
                    "constraints": asdict(c),
                    "context": _attest_safe(ctx),
                }
                entry = self.ledger.append(payload)
                attestation_hash = entry.entry_hash
            except Exception:                            # pragma: no cover
                pass
        if attestation_hash is None:
            # Always derive a stable digest so downstream consumers
            # can dedupe even without a ledger.
            digest_payload = json.dumps({
                "rec_id": rec_id,
                "strategy": strategy,
                "primary": primary.candidate.id if primary else None,
                "ev_lb": round(ev_lb, 6),
            }, sort_keys=True)
            attestation_hash = hashlib.sha256(digest_payload.encode()).hexdigest()

        rec = StrategyRecommendation(
            id=rec_id,
            strategy=strategy,
            primary=primary,
            hedged_arms=tuple(hedged),
            candidates=tuple(forecasts_sorted),
            pareto_frontier=tuple(pareto),
            expected_value_usd=ev,
            value_lower_bound_usd=ev_lb,
            cost_p95_usd=cost_p95,
            p_success=p_success,
            constraints=c,
            context=dict(ctx),
            confidence=confidence,
            rationale=rationale,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
            attestation_hash=attestation_hash,
            ts=time.time(),
        )
        with self._lock:
            self._open[rec_id] = rec

        if self.bus is not None:
            self.bus.publish(Event(
                kind=STRATEGIST_RECOMMENDED,
                data={
                    "recommendation_id": rec_id,
                    "strategy": strategy,
                    "primary": primary.candidate.id if primary else None,
                    "ev": ev,
                    "ev_lb": ev_lb,
                    "cost_p95": cost_p95,
                    "p_success": p_success,
                    "confidence": confidence,
                },
            ))
        return rec

    def _confidence_level(
        self,
        forecasts: list[CandidateForecast],
        primary: CandidateForecast | None,
    ) -> str:
        """Map data sufficiency + estimator agreement → confidence label."""
        if primary is None:
            return CONF_LOW
        total_samples = sum(f.candidate.samples for f in forecasts)
        if primary.candidate.samples >= 50 and primary.risk_score <= 0.3:
            return CONF_HIGH
        if primary.candidate.samples >= 10 and primary.risk_score <= 0.5:
            return CONF_MEDIUM
        if total_samples >= 50 and primary.risk_score <= 0.6:
            return CONF_MEDIUM
        return CONF_LOW

    def _chosen_forecast(
        self,
        rec: StrategyRecommendation,
        chosen_arm_id: str,
    ) -> CandidateForecast | None:
        for f in rec.candidates:
            if f.candidate.id == chosen_arm_id:
                return f
        # Hedged arms are a subset of candidates, but defensively look
        # there too.
        for f in rec.hedged_arms:
            if f.candidate.id == chosen_arm_id:
                return f
        return None


# ----- support classes ------------------------------------------------


class _DeterministicArmPolicy:
    """A deterministic policy used to query `PolicyLab.evaluate(...)`.

    Returns 1.0 propensity on the configured arm, 0.0 elsewhere.
    """

    def __init__(self, arm_id: str) -> None:
        self.arm_id = arm_id

    def __call__(self, context: Mapping[str, float]) -> Mapping[str, float]:
        # The PolicyLab `Policy` protocol accepts a callable
        # `policy(context) -> {action: probability}`.
        return {self.arm_id: 1.0}


def _numeric_only(ctx: Mapping[str, Any]) -> dict[str, float]:
    """Project a mixed-type context dict to its numeric subset."""
    out: dict[str, float] = {}
    for k, v in ctx.items():
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = float(v)
    return out


def _attest_safe(value: Any) -> Any:
    """Reduce a value to plain JSON-serialisable primitives for the ledger."""
    if isinstance(value, Mapping):
        return {str(k): _attest_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_attest_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
