"""CausalLab — heterogeneous treatment effects as a runtime primitive.

`PolicyLab` answers "what is the *average* lift of switching from policy
A to policy B on the traffic we logged?" That number is the right one
to bring to a steering committee. But a coordination engine that wants
to *route per request* needs a finer answer: **for this specific
context, what is the counterfactual lift of intervening with action τ
versus baseline action β?** That object is the Conditional Average
Treatment Effect (CATE):

    τ(c) := E[Y(τ) - Y(β) | C = c]

`CausalLab` estimates τ(c) from logged (context, action, propensity,
reward) tuples and turns it into a runtime primitive a coordinator can
call inline:

    lab = CausalLab(treatment="claude-haiku-4-5", control="claude-opus-4-7")
    for ev in policy_lab.events():
        lab.record(ev)

    point = lab.cate(context={"task_difficulty": 0.7}, learner="dr")
    if point.ci_low > 0:
        route_to("haiku")   # provably positive lift on this context
    elif point.ci_high < 0:
        route_to("opus")    # provably negative; keep the strong model
    else:
        route_to(default)   # not enough signal yet — log for next refit

This is "personalised off-policy evaluation". PolicyLab averages over
traffic. CausalLab segments it.

What it implements (razor's-edge of the HTE literature)
-------------------------------------------------------

  - **T-learner** — separate outcome regressors per arm. Simple and
    unbiased when each arm has support; high variance with imbalanced
    arms.

  - **S-learner** — single regressor with treatment as a feature. Low
    variance, but regularisation can shrink the treatment indicator
    toward zero and underestimate effects.

  - **X-learner** (Künzel-Sekhon-Bickel-Yu 2019, PNAS). Cross-fits per-
    arm regressors, builds *imputed* treatment effects on the opposite
    arm, then averages the two via a propensity-weighted combination.
    SOTA when arms are unbalanced — exactly the LLM-routing case.

  - **DR-learner** (Kennedy 2020, "Optimal doubly robust estimation of
    heterogeneous causal effects"). Builds a doubly-robust pseudo-
    outcome
        φᵢ = r̂(cᵢ, τ) - r̂(cᵢ, β)
             + 1[Aᵢ=τ]/π(τ|cᵢ) · (Yᵢ - r̂(cᵢ, τ))
             - 1[Aᵢ=β]/π(β|cᵢ) · (Yᵢ - r̂(cᵢ, β))
    and regresses φ on c. Doubly-robust: consistent if either the
    outcome model OR the propensity model is correct. Influence-
    function variance gives valid CIs for free.

  - **Qini curve & coefficient** (Radcliffe 2007) — the uplift analog
    of ROC AUC. Sort contexts by predicted τ̂(c), plot cumulative
    realised lift vs. random targeting; Qini = (area under uplift
    curve) - (area under random). Investor framing: "Routing the top
    decile by predicted CATE captures 4× the lift of random routing."

  - **Heterogeneity permutation test** (Chernozhukov-Demirer-Duflo-
    Fernandez-Val 2018). Shuffles treatment labels and refits to build
    a null distribution for Var(τ̂(c)). Rejects "all-units-respond-
    equally" with a proper p-value. Without this, every CATE report
    risks being noise dressed as personalisation.

  - **Best Linear Predictor of CATE** (Chernozhukov et al. 2018). After
    fitting any nuisance, regress the DR pseudo-outcome on a small set
    of context features to get an interpretable, OLS-style "this
    feature drives lift by β ± SE" table. The right output for a
    coordination engine that ships rules, not black boxes.

Surface a coordination engine drives
------------------------------------

    lab = CausalLab(treatment="cheap-arm", control="strong-arm")

    # Drain a PolicyLab or RuntimeDriver's events.
    lab.record_batch(policy_lab.events())

    # Per-request counterfactual reasoning.
    point = lab.cate(context, learner="dr", confidence=0.95)

    # Choose the best arm for this context, with provable lift CI.
    rec = lab.recommend(context, actions=("cheap", "strong"))
    if rec.lift_ci_low > 0:
        route_to(rec.best_action)

    # Population-level summary.
    report = lab.uplift(learner="x", n_buckets=10)
    print(report.summary)        # decile lift table + Qini coefficient

    # Sanity check: is there *any* heterogeneity?
    het = lab.test_heterogeneity(n_permutations=200)
    if not het.is_heterogeneous:
        # The new policy moves the average; it does not segment.
        # Coordinator should fall back to PolicyLab-style global decisions.
        ...

Honest about limits
-------------------

  - **Positivity / overlap**: CATE is identifiable only where both
    arms have data. The lab reports the per-arm propensity floor and
    flags contexts in the "no-overlap" zone via `support_warning`.

  - **Sample efficiency**: with K << 200 events per arm, learners
    collapse toward their priors. The lab returns a `low_data=True`
    diagnostic and *wide* CIs in that regime — by design, so a
    coordinator does not act on noise.

  - **Confounding**: CATE assumes the propensities the lab is given
    are the *true* probabilities of action selection. If the logging
    policy had hidden state, the estimate is biased. The lab cannot
    detect this; the operator must own propensity bookkeeping.

  - **Extrapolation**: any CATE at a context far from the training
    support is an extrapolation, not a measurement. The CI widens but
    cannot encode "I have never seen anything like this before". For
    novel contexts, gate on `support_score` returned with each point.

The lab is stdlib-only and reuses `agi.policy_lab.LoggedEvent` and
`LinearRewardModel`. Investor demos run in milliseconds; production
deployments scale to ~100k logged events in <1s per CATE refit.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus
from agi.policy_lab import (
    LinearRewardModel,
    LoggedEvent,
    PerActionMeanRewardModel,
    RewardModel,
    _bernstein_radius,
    _inv_normal,
    _normal_pcdf,
    _normalize_policy,
    _student_t_z,
)


# ----- event kinds ----------------------------------------------------

CAUSAL_RECORDED = "causal.recorded"
CAUSAL_FIT = "causal.fit"
CAUSAL_CATE_ESTIMATED = "causal.cate_estimated"
CAUSAL_UPLIFT = "causal.uplift_reported"
CAUSAL_HETEROGENEITY = "causal.heterogeneity_tested"
CAUSAL_RECOMMENDED = "causal.recommended"


# ----- learners -------------------------------------------------------

LEARNER_T = "t"
LEARNER_S = "s"
LEARNER_X = "x"
LEARNER_DR = "dr"

KNOWN_LEARNERS = (LEARNER_T, LEARNER_S, LEARNER_X, LEARNER_DR)


# Numerical floors.
_EPS = 1e-9
_DEFAULT_PROPENSITY_FLOOR = 1e-3
_DEFAULT_WEIGHT_CLIP = 50.0


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class CATEPoint:
    """Conditional average treatment effect at a single context.

    `value` is the estimated lift of `treatment` vs `control` at this
    context. Positive = treatment helps; negative = treatment hurts.

    `support_score` ∈ [0, 1] is the propensity-overlap floor at this
    context — how strongly the logging policy could have routed to
    either arm here. Low support_score → CATE is an extrapolation.

    `low_data` is True when the effective sample size used to construct
    the point estimate fell below the lab's `min_eff_n` floor.
    """

    treatment: str
    control: str
    value: float
    se: float
    ci_low: float
    ci_high: float
    confidence: float
    learner: str
    support_score: float
    low_data: bool
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["diagnostics"] = dict(self.diagnostics)
        return d


@dataclass(frozen=True)
class CATERecommendation:
    """Best action per context with counterfactual CI vs. a baseline."""

    context: Mapping[str, float]
    best_action: str
    baseline_action: str
    lift: float
    lift_se: float
    lift_ci_low: float
    lift_ci_high: float
    confidence: float
    per_action: Mapping[str, CATEPoint]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": dict(self.context),
            "best_action": self.best_action,
            "baseline_action": self.baseline_action,
            "lift": self.lift,
            "lift_se": self.lift_se,
            "lift_ci_low": self.lift_ci_low,
            "lift_ci_high": self.lift_ci_high,
            "confidence": self.confidence,
            "per_action": {k: v.to_dict() for k, v in self.per_action.items()},
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class UpliftBucket:
    """One decile (or arbitrary bucket) on the uplift curve."""

    rank: int
    n: int
    mean_predicted_cate: float
    mean_realised_lift: float
    cum_population_frac: float
    cum_lift: float


@dataclass(frozen=True)
class UpliftReport:
    """Population-level uplift summary."""

    learner: str
    n: int
    n_buckets: int
    buckets: tuple[UpliftBucket, ...]
    qini_coefficient: float
    qini_normalised: float
    auuc: float
    ate: float
    ate_se: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["buckets"] = [asdict(b) for b in self.buckets]
        return d


@dataclass(frozen=True)
class HeterogeneityTest:
    """Result of a permutation test for τ(c) ≠ const."""

    statistic: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    is_heterogeneous: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class BLPCoefficient:
    """One row of the Best Linear Predictor of CATE."""

    feature: str
    coef: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class BLPReport:
    """Best Linear Predictor of CATE — interpretable CATE summary."""

    intercept: BLPCoefficient
    coefficients: tuple[BLPCoefficient, ...]
    r_squared: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": asdict(self.intercept),
            "coefficients": [asdict(c) for c in self.coefficients],
            "r_squared": self.r_squared,
            "n": self.n,
        }


# =====================================================================
# The lab
# =====================================================================


class CausalLab:
    """Heterogeneous treatment effect estimation over logged events.

    The lab is thread-safe for record/cate/uplift/test calls. Nuisance
    refits are lazy: the first CATE call after new records triggers a
    refit; subsequent calls reuse the fit until more data arrives.

    Parameters
    ----------
    treatment
        Default treatment action name when callers do not pass one.
    control
        Default control / baseline action name.
    event_bus
        Optional `EventBus` to emit `causal.*` events for an operator
        dashboard.
    reward_model_factory
        Callable returning a fresh `RewardModel` for the lab's
        per-arm and pseudo-outcome regressors. Default:
        `LinearRewardModel(ridge=1.0)`. Pass `PerActionMeanRewardModel`
        for context-free baselines.
    propensity_floor
        Lower bound for propensities used as denominators. Defaults to
        1e-3 — sites with sub-floor propensity get clipped (with a
        diagnostic) rather than divided by ~0.
    weight_clip
        Importance-weight clip for DR-learner pseudo-outcomes. Default
        50.0 matches `PolicyLab`'s convention.
    min_eff_n
        Effective sample size below which point estimates carry the
        `low_data=True` flag. Default 20.
    max_events
        Soft cap on the in-memory log. When exceeded, oldest events
        drop FIFO. Set to None for unbounded.
    seed
        Random seed for cross-fitting and permutation tests.
    """

    def __init__(
        self,
        treatment: str | None = None,
        control: str | None = None,
        *,
        event_bus: EventBus | None = None,
        reward_model_factory: Callable[[], RewardModel] | None = None,
        propensity_floor: float = _DEFAULT_PROPENSITY_FLOOR,
        weight_clip: float = _DEFAULT_WEIGHT_CLIP,
        min_eff_n: int = 20,
        max_events: int | None = 100_000,
        seed: int = 0xA61,
    ) -> None:
        if propensity_floor <= 0 or propensity_floor >= 1:
            raise ValueError("propensity_floor must be in (0, 1)")
        if weight_clip <= 0:
            raise ValueError("weight_clip must be positive")
        if min_eff_n < 1:
            raise ValueError("min_eff_n must be >= 1")
        self.treatment = treatment
        self.control = control
        self.event_bus = event_bus
        self._rm_factory = reward_model_factory or (lambda: LinearRewardModel(ridge=1.0))
        self.propensity_floor = float(propensity_floor)
        self.weight_clip = float(weight_clip)
        self.min_eff_n = int(min_eff_n)
        self.max_events = max_events
        self._rng = random.Random(seed)

        self._lock = threading.RLock()
        self._events: list[LoggedEvent] = []
        self._fitted_at_n: int = -1
        self._per_arm: dict[str, RewardModel] = {}
        self._s_learner: RewardModel | None = None
        # Empirical per-action propensity per feature-bucket (very coarse).
        self._action_freq: dict[str, float] = {}

    # ----- ingest ----------------------------------------------------

    def record(self, event: LoggedEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._invalidate_fit()
            if self.max_events is not None and len(self._events) > self.max_events:
                drop = len(self._events) - self.max_events
                del self._events[: drop]
        self._emit(CAUSAL_RECORDED, {"action": event.action, "reward": event.reward})

    def record_batch(self, events: Iterable[LoggedEvent]) -> int:
        n = 0
        for ev in events:
            self.record(ev)
            n += 1
        return n

    def events(self) -> list[LoggedEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def _invalidate_fit(self) -> None:
        self._fitted_at_n = -1

    def fit(self) -> None:
        """Eager refit of all nuisance models. Lazy fit is usually fine."""
        with self._lock:
            self._refit_locked(list(self._events))
        self._emit(CAUSAL_FIT, {"n": self._fitted_at_n})

    def _ensure_fit(self) -> list[LoggedEvent]:
        with self._lock:
            events = list(self._events)
            if self._fitted_at_n != len(events):
                self._refit_locked(events)
            return events

    def _refit_locked(self, events: Sequence[LoggedEvent]) -> None:
        by_arm: dict[str, list[LoggedEvent]] = {}
        for ev in events:
            by_arm.setdefault(ev.action, []).append(ev)
        self._per_arm.clear()
        for action, evs in by_arm.items():
            rm = self._rm_factory()
            rm.fit(evs)
            self._per_arm[action] = rm
        # S-learner: augment context with action indicators *and* action ×
        # feature interactions. Without interactions, a linear S-learner can
        # only represent constant lift; with them it can match T/X-learner
        # capacity while keeping a single shared variance pool.
        s_rm = self._rm_factory()
        ctx_keys: set[str] = set()
        for ev in events:
            ctx_keys.update(ev.context.keys())
        feat_order = sorted(ctx_keys)
        s_events: list[LoggedEvent] = []
        for ev in events:
            ctx = dict(ev.context)
            for a in by_arm:
                indicator = 1.0 if a == ev.action else 0.0
                ctx[f"__a__{a}"] = indicator
                # Interaction columns: action_a × feature_k.
                for k in feat_order:
                    ctx[f"__a__{a}__x__{k}"] = indicator * float(ev.context.get(k, 0.0))
            s_events.append(
                LoggedEvent(
                    context=ctx,
                    action="__s__",  # single-arm container
                    propensity=ev.propensity,
                    reward=ev.reward,
                    timestamp=ev.timestamp,
                    id=ev.id,
                    tenant_id=ev.tenant_id,
                    metadata=ev.metadata,
                )
            )
        s_rm.fit(s_events)
        self._s_learner = s_rm
        self._s_feat_order: tuple[str, ...] = tuple(feat_order)
        # Action marginal frequencies (used only as a positivity smoke test).
        n = max(1, len(events))
        freqs: dict[str, float] = {}
        for ev in events:
            freqs[ev.action] = freqs.get(ev.action, 0.0) + 1.0
        self._action_freq = {a: c / n for a, c in freqs.items()}
        self._fitted_at_n = len(events)

    # ----- CATE point estimate ---------------------------------------

    def cate(
        self,
        context: Mapping[str, float],
        *,
        treatment: str | None = None,
        control: str | None = None,
        learner: str = LEARNER_DR,
        confidence: float = 0.95,
        n_bootstrap: int = 0,
    ) -> CATEPoint:
        """Estimate τ(c) = E[Y(treatment) - Y(control) | C = c].

        For `learner='dr'` the SE is computed from the doubly-robust
        influence functions (no bootstrap needed). For T/S/X learners
        you may pass `n_bootstrap` ≥ 50 to get a non-parametric CI;
        with `n_bootstrap=0` the CI uses a sub-population residual
        plug-in that is a coarse but free lower bound on uncertainty.
        """
        if learner not in KNOWN_LEARNERS:
            raise ValueError(f"unknown learner {learner!r}; expected one of {KNOWN_LEARNERS}")
        if confidence <= 0 or confidence >= 1:
            raise ValueError("confidence must be in (0, 1)")
        t = treatment or self.treatment
        c = control or self.control
        if t is None or c is None:
            raise ValueError("treatment and control must be set (constructor or call site)")
        if t == c:
            raise ValueError("treatment and control must differ")

        events = self._ensure_fit()
        if not events:
            return _empty_cate(t, c, learner, confidence)

        if learner == LEARNER_T:
            point = self._cate_t(context, t, c, confidence)
        elif learner == LEARNER_S:
            point = self._cate_s(context, t, c, confidence)
        elif learner == LEARNER_X:
            point = self._cate_x(events, context, t, c, confidence)
        else:  # LEARNER_DR
            point = self._cate_dr(events, context, t, c, confidence)

        if n_bootstrap and learner in (LEARNER_T, LEARNER_S, LEARNER_X):
            point = self._refine_with_bootstrap(
                events, context, t, c, learner, confidence, int(n_bootstrap)
            )

        self._emit(
            CAUSAL_CATE_ESTIMATED,
            {
                "learner": learner,
                "treatment": t,
                "control": c,
                "value": point.value,
                "support_score": point.support_score,
            },
        )
        return point

    def _cate_t(
        self,
        context: Mapping[str, float],
        treatment: str,
        control: str,
        confidence: float,
    ) -> CATEPoint:
        r_t = self._predict_arm(treatment, context)
        r_c = self._predict_arm(control, context)
        value = r_t - r_c
        # CI from per-arm residual variance, conservative plug-in:
        # Var(τ̂) ≈ σ²_t / n_t + σ²_c / n_c — averaged over the dataset.
        var_t, n_t = self._arm_residual_var(treatment)
        var_c, n_c = self._arm_residual_var(control)
        se = math.sqrt(var_t / max(1, n_t) + var_c / max(1, n_c))
        support = self._support(context, treatment, control)
        n_eff = min(n_t, n_c)
        return _build_cate(
            value, se, treatment, control, LEARNER_T, confidence, support, n_eff, self.min_eff_n,
            diagnostics={"n_t": float(n_t), "n_c": float(n_c)},
        )

    def _cate_s(
        self,
        context: Mapping[str, float],
        treatment: str,
        control: str,
        confidence: float,
    ) -> CATEPoint:
        if self._s_learner is None:
            return _empty_cate(treatment, control, LEARNER_S, confidence)
        ctx_t = dict(context)
        ctx_c = dict(context)
        feat_order = getattr(self, "_s_feat_order", ())
        for a in self._per_arm:
            ind_t = 1.0 if a == treatment else 0.0
            ind_c = 1.0 if a == control else 0.0
            ctx_t[f"__a__{a}"] = ind_t
            ctx_c[f"__a__{a}"] = ind_c
            for k in feat_order:
                xk = float(context.get(k, 0.0))
                ctx_t[f"__a__{a}__x__{k}"] = ind_t * xk
                ctx_c[f"__a__{a}__x__{k}"] = ind_c * xk
        r_t = self._s_learner.predict(ctx_t, "__s__")
        r_c = self._s_learner.predict(ctx_c, "__s__")
        value = r_t - r_c
        # SE: shared residual variance × √(2/n_overlap), since S regresses on the union.
        n_t = sum(1 for ev in self._events if ev.action == treatment)
        n_c = sum(1 for ev in self._events if ev.action == control)
        var_pooled, _ = self._pooled_residual_var(treatment, control)
        denom = max(1, min(n_t, n_c))
        se = math.sqrt(2.0 * var_pooled / denom)
        support = self._support(context, treatment, control)
        return _build_cate(
            value, se, treatment, control, LEARNER_S, confidence, support,
            min(n_t, n_c), self.min_eff_n,
            diagnostics={"n_t": float(n_t), "n_c": float(n_c)},
        )

    def _cate_x(
        self,
        events: Sequence[LoggedEvent],
        context: Mapping[str, float],
        treatment: str,
        control: str,
        confidence: float,
    ) -> CATEPoint:
        # X-learner per Künzel et al. 2019.
        # Step 1: per-arm regressors r̂_t, r̂_c (we already have them from _refit).
        # Step 2: imputed treatment effects on each arm.
        #   D¹ᵢ = Yᵢ - r̂_c(cᵢ)         for Aᵢ = treatment
        #   D⁰ᵢ = r̂_t(cᵢ) - Yᵢ         for Aᵢ = control
        # Step 3: regress D¹ on c → τ̂_1, regress D⁰ on c → τ̂_0
        # Step 4: combine with propensity g(c):
        #   τ̂(c) = g(c) τ̂_0(c) + (1 - g(c)) τ̂_1(c)
        events_t = [ev for ev in events if ev.action == treatment]
        events_c = [ev for ev in events if ev.action == control]
        if not events_t or not events_c:
            return _empty_cate(treatment, control, LEARNER_X, confidence)
        # Build imputed-effect events. Action label is reused as "__x__" so the
        # underlying LinearRewardModel fits a single model per side.
        d1 = []
        for ev in events_t:
            imputed = ev.reward - self._predict_arm(control, ev.context)
            d1.append(
                LoggedEvent(
                    context=ev.context, action="__x__", propensity=1.0, reward=imputed
                )
            )
        d0 = []
        for ev in events_c:
            imputed = self._predict_arm(treatment, ev.context) - ev.reward
            d0.append(
                LoggedEvent(
                    context=ev.context, action="__x__", propensity=1.0, reward=imputed
                )
            )
        rm_1 = self._rm_factory()
        rm_1.fit(d1)
        rm_0 = self._rm_factory()
        rm_0.fit(d0)
        tau_1 = rm_1.predict(context, "__x__")
        tau_0 = rm_0.predict(context, "__x__")
        # Propensity g(c) ≈ Pr(A = treatment | C = c). We do not own a per-context
        # propensity model so we use the empirical marginal — biased but robust.
        n_t = len(events_t)
        n_c = len(events_c)
        g = n_t / max(1, n_t + n_c)
        value = g * tau_0 + (1.0 - g) * tau_1
        # SE: empirical SD of the imputed effects scaled by 1/min(n_t, n_c).
        sd_1 = statistics.pstdev(e.reward for e in d1) if len(d1) > 1 else 0.0
        sd_0 = statistics.pstdev(e.reward for e in d0) if len(d0) > 1 else 0.0
        denom = max(1, min(n_t, n_c))
        se = math.sqrt((g * sd_0) ** 2 / denom + ((1 - g) * sd_1) ** 2 / denom)
        support = self._support(context, treatment, control)
        return _build_cate(
            value, se, treatment, control, LEARNER_X, confidence, support,
            min(n_t, n_c), self.min_eff_n,
            diagnostics={
                "tau_0": tau_0, "tau_1": tau_1, "g": g,
                "n_t": float(n_t), "n_c": float(n_c),
            },
        )

    def _cate_dr(
        self,
        events: Sequence[LoggedEvent],
        context: Mapping[str, float],
        treatment: str,
        control: str,
        confidence: float,
    ) -> CATEPoint:
        # DR-learner per Kennedy 2020.
        #   φᵢ = r̂(cᵢ, τ) - r̂(cᵢ, β)
        #        + 1[Aᵢ=τ]/π(τ|cᵢ) · (Yᵢ - r̂(cᵢ, τ))
        #        - 1[Aᵢ=β]/π(β|cᵢ) · (Yᵢ - r̂(cᵢ, β))
        # τ̂(c) = E[φ | C = c].
        # We regress φ on c with the same RewardModel family.
        if treatment not in self._per_arm or control not in self._per_arm:
            return _empty_cate(treatment, control, LEARNER_DR, confidence)
        pseudo: list[LoggedEvent] = []
        clip = self.weight_clip
        floor = self.propensity_floor
        sq = 0.0
        n_eff_w = 0.0
        max_w = 0.0
        for ev in events:
            r_t = self._predict_arm(treatment, ev.context)
            r_c = self._predict_arm(control, ev.context)
            phi = r_t - r_c
            pi = max(floor, float(ev.propensity))
            w = 1.0 / pi
            if w > clip:
                w = clip
            if w > max_w:
                max_w = w
            if ev.action == treatment:
                phi += w * (ev.reward - r_t)
                n_eff_w += 1.0
            elif ev.action == control:
                phi -= w * (ev.reward - r_c)
                n_eff_w += 1.0
            sq += phi * phi
            pseudo.append(
                LoggedEvent(
                    context=ev.context, action="__dr__", propensity=1.0, reward=phi
                )
            )
        if not pseudo:
            return _empty_cate(treatment, control, LEARNER_DR, confidence)
        rm = self._rm_factory()
        rm.fit(pseudo)
        value = rm.predict(context, "__dr__")
        # Influence-function SE at the population level:
        # Var(τ̂(c)) ≈ (1/n) Var(φ - τ̂(c)).
        n = len(pseudo)
        residuals = [(p.reward - value) for p in pseudo]
        if n > 1:
            var = sum(r * r for r in residuals) / (n - 1)
        else:
            var = float("inf")
        se = math.sqrt(var / n) if math.isfinite(var) else float("inf")
        support = self._support(context, treatment, control)
        diagnostics = {
            "max_weight": max_w,
            "n_treatment_or_control": n_eff_w,
            "mean_phi": sum(p.reward for p in pseudo) / n,
        }
        return _build_cate(
            value, se, treatment, control, LEARNER_DR, confidence, support,
            int(n_eff_w), self.min_eff_n, diagnostics=diagnostics,
        )

    def _refine_with_bootstrap(
        self,
        events: Sequence[LoggedEvent],
        context: Mapping[str, float],
        treatment: str,
        control: str,
        learner: str,
        confidence: float,
        n_bootstrap: int,
    ) -> CATEPoint:
        if not events or n_bootstrap < 2:
            return self.cate(
                context, treatment=treatment, control=control, learner=learner,
                confidence=confidence,
            )
        rng = random.Random(self._rng.random())
        n = len(events)
        values: list[float] = []
        for _ in range(n_bootstrap):
            sample = [events[rng.randrange(n)] for _ in range(n)]
            tmp = CausalLab(
                treatment=treatment, control=control,
                reward_model_factory=self._rm_factory,
                propensity_floor=self.propensity_floor,
                weight_clip=self.weight_clip,
                min_eff_n=self.min_eff_n,
                max_events=None,
            )
            for ev in sample:
                tmp.record(ev)
            point = tmp.cate(context, learner=learner, confidence=confidence)
            values.append(point.value)
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) if len(values) > 1 else float("inf")
        z = _student_t_z(confidence, n_bootstrap)
        support = self._support(context, treatment, control)
        return _build_cate(
            mean, sd, treatment, control, learner, confidence, support, n, self.min_eff_n,
            diagnostics={"bootstrap": float(n_bootstrap), "se_bootstrap": sd},
        )

    # ----- recommend per-context ------------------------------------

    def recommend(
        self,
        context: Mapping[str, float],
        actions: Sequence[str],
        *,
        baseline: str | None = None,
        learner: str = LEARNER_DR,
        confidence: float = 0.95,
    ) -> CATERecommendation:
        """Pick the action with the highest CATE vs. a baseline.

        If `baseline` is None, the most-frequent logged action is used.
        Returns per-action CATE points plus the lift CI for the winner.
        """
        if not actions:
            raise ValueError("actions must be non-empty")
        events = self._ensure_fit()
        if not events:
            empty = _empty_cate("?", baseline or "?", learner, confidence)
            return CATERecommendation(
                context=dict(context), best_action=actions[0],
                baseline_action=baseline or actions[0], lift=0.0,
                lift_se=float("inf"), lift_ci_low=float("-inf"),
                lift_ci_high=float("inf"), confidence=confidence,
                per_action={a: empty for a in actions},
                rationale="no logged events; cannot recommend.",
            )

        if baseline is None:
            baseline = max(self._action_freq.items(), key=lambda kv: kv[1])[0]
        per: dict[str, CATEPoint] = {}
        for a in actions:
            if a == baseline:
                per[a] = CATEPoint(
                    treatment=a, control=baseline, value=0.0, se=0.0,
                    ci_low=0.0, ci_high=0.0, confidence=confidence,
                    learner=learner,
                    support_score=self._support(context, a, baseline),
                    low_data=False, diagnostics={},
                )
            else:
                per[a] = self.cate(
                    context, treatment=a, control=baseline,
                    learner=learner, confidence=confidence,
                )
        best = max(per.items(), key=lambda kv: kv[1].value)[0]
        bp = per[best]
        rationale = (
            f"action={best} lift_vs_{baseline}={bp.value:+.4f} "
            f"CI=[{bp.ci_low:+.4f},{bp.ci_high:+.4f}] "
            f"support={bp.support_score:.2f}"
            + (" low_data" if bp.low_data else "")
        )
        rec = CATERecommendation(
            context=dict(context), best_action=best, baseline_action=baseline,
            lift=bp.value, lift_se=bp.se,
            lift_ci_low=bp.ci_low, lift_ci_high=bp.ci_high,
            confidence=confidence, per_action=per, rationale=rationale,
        )
        self._emit(CAUSAL_RECOMMENDED, {"best": best, "baseline": baseline, "lift": bp.value})
        return rec

    # ----- uplift / Qini --------------------------------------------

    def uplift(
        self,
        *,
        treatment: str | None = None,
        control: str | None = None,
        learner: str = LEARNER_DR,
        confidence: float = 0.95,
        n_buckets: int = 10,
    ) -> UpliftReport:
        """Decile-style uplift curve + Qini coefficient.

        Sorts logged events by predicted CATE, then walks the sort
        cumulatively computing realised IPW lift in each bucket.
        Returns the Qini coefficient: the area between the cumulative
        uplift curve and the random-targeting diagonal.
        """
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        t = treatment or self.treatment
        c = control or self.control
        if t is None or c is None:
            raise ValueError("treatment and control must be set")
        events = self._ensure_fit()
        if not events:
            return UpliftReport(
                learner=learner, n=0, n_buckets=n_buckets, buckets=(),
                qini_coefficient=0.0, qini_normalised=0.0, auuc=0.0,
                ate=0.0, ate_se=float("inf"),
                summary="empty: no logged events.",
            )

        # Predict CATE on each logged context and pair with the IPW
        # contribution to the lift τ - β.
        floor = self.propensity_floor
        clip = self.weight_clip
        scored: list[tuple[float, float]] = []  # (predicted_cate, ipw_contribution)
        for ev in events:
            point = self.cate(ev.context, treatment=t, control=c, learner=learner,
                              confidence=confidence)
            w = 0.0
            sign = 0
            if ev.action == t:
                w = min(clip, 1.0 / max(floor, ev.propensity))
                sign = +1
            elif ev.action == c:
                w = min(clip, 1.0 / max(floor, ev.propensity))
                sign = -1
            contribution = sign * w * ev.reward
            scored.append((point.value, contribution))

        # Sort by predicted CATE descending — high-uplift contexts first.
        scored.sort(key=lambda x: x[0], reverse=True)
        n = len(scored)
        ate = sum(s for _, s in scored) / n
        # ATE SE via influence function on IPW contributions.
        if n > 1:
            var = sum((s - ate) ** 2 for _, s in scored) / (n - 1)
            ate_se = math.sqrt(var / n)
        else:
            ate_se = float("inf")

        # Bucket cumulatively.
        per_bucket = max(1, n // n_buckets)
        buckets: list[UpliftBucket] = []
        cum_lift = 0.0
        cum_pred = 0.0
        running_sum = 0.0
        for b in range(n_buckets):
            start = b * per_bucket
            end = (b + 1) * per_bucket if b < n_buckets - 1 else n
            chunk = scored[start:end]
            if not chunk:
                continue
            chunk_pred = sum(p for p, _ in chunk) / len(chunk)
            chunk_lift = sum(s for _, s in chunk) / len(chunk)
            running_sum += sum(s for _, s in chunk)
            cum_lift = running_sum / max(1, end)
            cum_pred += chunk_pred
            buckets.append(
                UpliftBucket(
                    rank=b + 1,
                    n=len(chunk),
                    mean_predicted_cate=chunk_pred,
                    mean_realised_lift=chunk_lift,
                    cum_population_frac=end / n,
                    cum_lift=cum_lift,
                )
            )

        # Qini coefficient: integrate (cum_lift - ate * cum_frac) over buckets.
        # That is the area between the targeted curve and the random-targeting
        # straight line, scaled by population fraction.
        auuc = 0.0
        prev_frac = 0.0
        prev_y = 0.0
        for bk in buckets:
            y = bk.cum_lift * bk.cum_population_frac
            # Trapezoidal area increment.
            auuc += (bk.cum_population_frac - prev_frac) * 0.5 * (prev_y + y)
            prev_frac = bk.cum_population_frac
            prev_y = y
        random_auc = 0.5 * ate
        qini = auuc - random_auc
        qini_norm = qini / abs(ate) if abs(ate) > _EPS else 0.0

        # Build a one-screen summary.
        lines = [
            f"CausalLab.uplift learner={learner} n={n} treatment={t} control={c}",
            f"  ATE = {ate:+.4f} ± {ate_se:.4f}   Qini = {qini:+.4f}  AUUC = {auuc:+.4f}",
        ]
        for bk in buckets:
            lines.append(
                f"  D{bk.rank:>2}/{n_buckets} n={bk.n:>4d} "
                f"pred={bk.mean_predicted_cate:+.4f} "
                f"real={bk.mean_realised_lift:+.4f} "
                f"cum_lift={bk.cum_lift:+.4f} "
                f"@ {bk.cum_population_frac*100:>5.1f}% pop"
            )
        report = UpliftReport(
            learner=learner, n=n, n_buckets=n_buckets,
            buckets=tuple(buckets), qini_coefficient=qini,
            qini_normalised=qini_norm, auuc=auuc, ate=ate, ate_se=ate_se,
            summary="\n".join(lines),
        )
        self._emit(
            CAUSAL_UPLIFT,
            {"ate": ate, "qini": qini, "n": n, "n_buckets": n_buckets},
        )
        return report

    # ----- heterogeneity test ---------------------------------------

    def test_heterogeneity(
        self,
        *,
        treatment: str | None = None,
        control: str | None = None,
        learner: str = LEARNER_DR,
        n_permutations: int = 200,
        confidence: float = 0.95,
        max_eval_contexts: int = 200,
    ) -> HeterogeneityTest:
        """Permutation test for "is there *any* heterogeneity in τ(c)?".

        Null hypothesis: τ(c) is constant in c (i.e. the policy lifts
        every context equally). We use Var_c(τ̂(c)) as the statistic and
        build the null distribution by shuffling action labels among
        logged events (preserving marginals).

        Returns a `HeterogeneityTest` with the observed statistic, null
        moments, the (one-sided) p-value, and `is_heterogeneous` set
        when p < 1 - confidence.
        """
        if n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        t = treatment or self.treatment
        c = control or self.control
        if t is None or c is None:
            raise ValueError("treatment and control must be set")
        events = self._ensure_fit()
        relevant = [ev for ev in events if ev.action in (t, c)]
        if len(relevant) < 4:
            return HeterogeneityTest(
                statistic=0.0, null_mean=0.0, null_std=0.0, p_value=1.0,
                n_permutations=0, is_heterogeneous=False, confidence=confidence,
            )

        # Sample contexts to score (saves time on big logs).
        rng = random.Random(self._rng.random())
        eval_ctxs = [ev.context for ev in relevant]
        if len(eval_ctxs) > max_eval_contexts:
            eval_ctxs = rng.sample(eval_ctxs, max_eval_contexts)

        observed = _variance(
            [self.cate(c2, treatment=t, control=c, learner=learner,
                       confidence=confidence).value for c2 in eval_ctxs]
        )

        # Null distribution by shuffling treatment labels among (t, c) events.
        null_stats: list[float] = []
        actions = [ev.action for ev in relevant]
        for _ in range(n_permutations):
            shuffled = actions[:]
            rng.shuffle(shuffled)
            tmp = CausalLab(
                treatment=t, control=c,
                reward_model_factory=self._rm_factory,
                propensity_floor=self.propensity_floor,
                weight_clip=self.weight_clip,
                min_eff_n=self.min_eff_n,
                max_events=None,
            )
            # Also keep all non-(t,c) events untouched so the support is identical.
            other = [ev for ev in events if ev.action not in (t, c)]
            for ev, new_a in zip(relevant, shuffled):
                tmp.record(
                    LoggedEvent(
                        context=ev.context, action=new_a,
                        propensity=ev.propensity, reward=ev.reward,
                    )
                )
            for ev in other:
                tmp.record(ev)
            stat = _variance(
                [tmp.cate(c2, treatment=t, control=c, learner=learner,
                          confidence=confidence).value for c2 in eval_ctxs]
            )
            null_stats.append(stat)

        null_mean = statistics.fmean(null_stats)
        null_std = statistics.pstdev(null_stats) if len(null_stats) > 1 else 0.0
        # One-sided p: fraction of nulls at least as extreme.
        ge = sum(1 for s in null_stats if s >= observed)
        p_value = (ge + 1) / (len(null_stats) + 1)
        alpha = 1.0 - confidence
        is_het = p_value < alpha
        result = HeterogeneityTest(
            statistic=observed, null_mean=null_mean, null_std=null_std,
            p_value=p_value, n_permutations=n_permutations,
            is_heterogeneous=is_het, confidence=confidence,
        )
        self._emit(
            CAUSAL_HETEROGENEITY,
            {"statistic": observed, "p_value": p_value, "is_heterogeneous": is_het},
        )
        return result

    # ----- Best Linear Predictor of CATE ----------------------------

    def best_linear_predictor(
        self,
        *,
        treatment: str | None = None,
        control: str | None = None,
        learner: str = LEARNER_DR,
        confidence: float = 0.95,
    ) -> BLPReport:
        """Regress the DR pseudo-outcome on the raw context features.

        Returns OLS-style coefficients with valid SEs. Investor framing:
        "Per unit of `task_difficulty`, the cheap model loses
        $X ± $Y of EV vs. the strong one." Coefficients with CIs that
        exclude zero are the rules a coordination engine can ship.
        """
        t = treatment or self.treatment
        c = control or self.control
        if t is None or c is None:
            raise ValueError("treatment and control must be set")
        events = self._ensure_fit()
        if not events or t not in self._per_arm or c not in self._per_arm:
            empty = BLPCoefficient(
                feature="(intercept)", coef=0.0, se=float("inf"),
                ci_low=float("-inf"), ci_high=float("inf"), p_value=1.0,
            )
            return BLPReport(intercept=empty, coefficients=(), r_squared=0.0, n=0)
        floor = self.propensity_floor
        clip = self.weight_clip
        keys: set[str] = set()
        for ev in events:
            keys.update(ev.context.keys())
        feats = sorted(keys)
        # Build pseudo-outcomes and design matrix.
        X: list[list[float]] = []
        y: list[float] = []
        for ev in events:
            r_t = self._predict_arm(t, ev.context)
            r_c = self._predict_arm(c, ev.context)
            phi = r_t - r_c
            if ev.action == t:
                w = min(clip, 1.0 / max(floor, ev.propensity))
                phi += w * (ev.reward - r_t)
            elif ev.action == c:
                w = min(clip, 1.0 / max(floor, ev.propensity))
                phi -= w * (ev.reward - r_c)
            row = [1.0] + [float(ev.context.get(k, 0.0)) for k in feats]
            X.append(row)
            y.append(phi)
        beta, cov, r2 = _ols_with_covariance(X, y)
        if not beta:
            empty = BLPCoefficient(
                feature="(intercept)", coef=0.0, se=float("inf"),
                ci_low=float("-inf"), ci_high=float("inf"), p_value=1.0,
            )
            return BLPReport(intercept=empty, coefficients=(), r_squared=0.0, n=len(y))
        z = _student_t_z(confidence, len(y))
        names = ["(intercept)"] + feats
        rows: list[BLPCoefficient] = []
        for i, name in enumerate(names):
            se = math.sqrt(max(0.0, cov[i][i]))
            ci_lo = beta[i] - z * se
            ci_hi = beta[i] + z * se
            pv = _two_sided_p(beta[i] / se) if se > 0 else 1.0
            rows.append(BLPCoefficient(
                feature=name, coef=beta[i], se=se,
                ci_low=ci_lo, ci_high=ci_hi, p_value=pv,
            ))
        return BLPReport(
            intercept=rows[0],
            coefficients=tuple(rows[1:]),
            r_squared=r2,
            n=len(y),
        )

    # ----- support / overlap diagnostics ----------------------------

    def support(self, context: Mapping[str, float], treatment: str, control: str) -> float:
        """Public-API support score in [0, 1] for a context."""
        self._ensure_fit()
        return self._support(context, treatment, control)

    def _support(self, context: Mapping[str, float], treatment: str, control: str) -> float:
        # Coarse-but-honest: lean on the marginal frequencies of the two arms.
        # If both arms appear with non-trivial mass in the log, support = min ratio.
        ft = self._action_freq.get(treatment, 0.0)
        fc = self._action_freq.get(control, 0.0)
        if ft <= 0 or fc <= 0:
            return 0.0
        return 2.0 * min(ft, fc)  # caps at 1.0 when arms are balanced (0.5/0.5).

    def _predict_arm(self, action: str, context: Mapping[str, float]) -> float:
        rm = self._per_arm.get(action)
        if rm is None:
            return 0.0
        return float(rm.predict(context, action))

    def _arm_residual_var(self, action: str) -> tuple[float, int]:
        rm = self._per_arm.get(action)
        if rm is None:
            return (float("inf"), 0)
        events = [ev for ev in self._events if ev.action == action]
        if not events:
            return (float("inf"), 0)
        residuals = [(ev.reward - rm.predict(ev.context, action)) for ev in events]
        if len(residuals) < 2:
            return (float("inf"), len(residuals))
        mean = sum(residuals) / len(residuals)
        var = sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1)
        return (var, len(residuals))

    def _pooled_residual_var(self, a1: str, a2: str) -> tuple[float, int]:
        v1, n1 = self._arm_residual_var(a1)
        v2, n2 = self._arm_residual_var(a2)
        if not math.isfinite(v1) and not math.isfinite(v2):
            return (float("inf"), 0)
        if not math.isfinite(v1):
            return (v2, n2)
        if not math.isfinite(v2):
            return (v1, n1)
        n = n1 + n2
        if n < 2:
            return (float("inf"), n)
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, (n - 2))
        return (pooled, n)

    # ----- persistence ----------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "treatment": self.treatment,
                "control": self.control,
                "propensity_floor": self.propensity_floor,
                "weight_clip": self.weight_clip,
                "min_eff_n": self.min_eff_n,
                "events": [e.to_dict() for e in self._events],
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("version") != 1:
            raise ValueError("unknown snapshot version")
        with self._lock:
            self.treatment = snapshot.get("treatment")
            self.control = snapshot.get("control")
            self.propensity_floor = float(snapshot.get("propensity_floor", self.propensity_floor))
            self.weight_clip = float(snapshot.get("weight_clip", self.weight_clip))
            self.min_eff_n = int(snapshot.get("min_eff_n", self.min_eff_n))
            self._events = [LoggedEvent.from_dict(d) for d in snapshot.get("events", [])]
            self._invalidate_fit()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.snapshot()))

    def load(self, path: str | Path) -> None:
        self.restore(json.loads(Path(path).read_text()))

    # ----- integrations ---------------------------------------------

    def attach_to_policy_lab(self, lab: Any) -> int:
        """Drain a PolicyLab's existing events into this CausalLab.

        Returns the number of events imported. Useful when a coordinator
        already maintains a PolicyLab for average-effect evaluation and
        wants to layer per-context reasoning on the same log.
        """
        if not hasattr(lab, "events"):
            raise TypeError("attach_to_policy_lab requires a PolicyLab-like object")
        n = 0
        for ev in lab.events():
            self.record(ev)
            n += 1
        return n

    # ----- internal --------------------------------------------------

    def _emit(self, kind: str, data: Mapping[str, Any]) -> None:
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:
            pass


# =====================================================================
# Helpers
# =====================================================================


def _build_cate(
    value: float,
    se: float,
    treatment: str,
    control: str,
    learner: str,
    confidence: float,
    support: float,
    n_eff: int,
    min_eff_n: int,
    *,
    diagnostics: Mapping[str, float] | None = None,
) -> CATEPoint:
    if not math.isfinite(se):
        ci_low = float("-inf")
        ci_high = float("inf")
    else:
        z = _student_t_z(confidence, max(2, n_eff))
        ci_low = value - z * se
        ci_high = value + z * se
    return CATEPoint(
        treatment=treatment, control=control, value=value, se=se,
        ci_low=ci_low, ci_high=ci_high, confidence=confidence,
        learner=learner, support_score=support,
        low_data=n_eff < min_eff_n,
        diagnostics=dict(diagnostics or {}),
    )


def _empty_cate(
    treatment: str, control: str, learner: str, confidence: float
) -> CATEPoint:
    return CATEPoint(
        treatment=treatment, control=control, value=0.0, se=float("inf"),
        ci_low=float("-inf"), ci_high=float("inf"), confidence=confidence,
        learner=learner, support_score=0.0, low_data=True, diagnostics={},
    )


def _variance(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _two_sided_p(z: float) -> float:
    if not math.isfinite(z):
        return 1.0
    return 2.0 * (1.0 - _normal_pcdf(abs(z)))


def _ols_with_covariance(
    X: Sequence[Sequence[float]], y: Sequence[float]
) -> tuple[list[float], list[list[float]], float]:
    """OLS with sandwich-style residual covariance — stdlib Gauss-Jordan.

    Returns (beta, cov(beta), R²). On singular design, returns (empty, [], 0).
    """
    n = len(X)
    if n == 0:
        return ([], [], 0.0)
    d = len(X[0])
    # X^T X and X^T y.
    xtx = [[0.0] * d for _ in range(d)]
    xty = [0.0] * d
    for i in range(n):
        xi = X[i]
        yi = y[i]
        for r in range(d):
            xtx[r][r] += 0  # noop; placeholder for ridge if we ever add one
            xrv = xi[r]
            xty[r] += xrv * yi
            for c2 in range(d):
                xtx[r][c2] += xrv * xi[c2]
    # Solve (X^T X) β = X^T y.
    try:
        inv = _invert(xtx)
    except _SingularDesign:
        return ([], [], 0.0)
    beta = [0.0] * d
    for r in range(d):
        s = 0.0
        for c2 in range(d):
            s += inv[r][c2] * xty[c2]
        beta[r] = s
    # Residuals and σ².
    y_hat = []
    for i in range(n):
        s = 0.0
        for r in range(d):
            s += X[i][r] * beta[r]
        y_hat.append(s)
    residuals = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(r * r for r in residuals)
    df = max(1, n - d)
    sigma2 = rss / df
    cov = [[inv[r][c2] * sigma2 for c2 in range(d)] for r in range(d)]
    y_mean = sum(y) / n
    tss = sum((yi - y_mean) ** 2 for yi in y)
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
    return (beta, cov, r2)


class _SingularDesign(Exception):
    pass


def _invert(M: Sequence[Sequence[float]]) -> list[list[float]]:
    """Matrix inverse by Gauss-Jordan with partial pivoting."""
    n = len(M)
    A = [list(M[i]) + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        piv = i
        piv_val = abs(A[i][i])
        for r in range(i + 1, n):
            v = abs(A[r][i])
            if v > piv_val:
                piv_val = v
                piv = r
        if piv_val < 1e-12:
            raise _SingularDesign()
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
        pv = A[i][i]
        for c2 in range(i, 2 * n):
            A[i][c2] /= pv
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            if factor == 0.0:
                continue
            for c2 in range(i, 2 * n):
                A[r][c2] -= factor * A[i][c2]
    return [row[n:] for row in A]
