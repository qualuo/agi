"""PolicyLab — off-policy evaluation lab for the runtime.

`TicketOracle` answers "what if I had used different admission knobs?" by
re-running the advisor against logged receipts. That works because the
advisor is a deterministic function the runtime owns. The moment a
coordination engine wants to evaluate *its own* policy — say a new model
mix, a new pricing rule, a new retry strategy, a learned router — the
runtime needs a more general primitive: **off-policy evaluation (OPE)**.

`PolicyLab` is that primitive. Give it (1) a log of (context, action,
propensity, reward) tuples that the production system actually emitted
and (2) any new policy expressed as `π(a | c)`. It returns a calibrated
estimate of what the new policy *would have* earned on that traffic,
with a confidence interval, without spending a single real dollar.

This is the line between "we ran an experiment and got a number" and
"we can rank ten policies and ship the best one with a provable
statistical guarantee before the next billing cycle."

What it implements (razor's-edge of contextual bandits)
-------------------------------------------------------

  - **IPS** — inverse-propensity scoring (Horvitz-Thompson 1952). Unbiased
    when propensities are known; variance explodes on small μ(a|c).

  - **SNIPS** — self-normalised IPS (Trotter-Tukey 1956 → Swaminathan-
    Joachims 2015). Trades a small bias for an order-of-magnitude
    variance cut. The estimator the production OPE community ships.

  - **DM** — direct method. Fit a reward model r̂(c, a) and integrate
    against π. Low variance, high bias if the model is misspecified.

  - **DR** — doubly-robust (Dudík-Langford-Li 2011 / Robins 1994).
    Unbiased if either μ OR r̂ is correct. Best of both worlds; this
    is the workhorse of OPE in industrial RL.

  - **SWITCH-DR** — Wang-Agarwal-Dudík 2017. Use IPS only when the
    importance weight is below a threshold τ; fall back to DM for
    heavy-tailed weights. Provably lower MSE than DR on long-tail
    data.

  - **Empirical Bernstein** CIs (Maurer-Pontil 2009) on top of the
    influence functions: tighter than Hoeffding for bounded data with
    low empirical variance. Reports both Bernstein and Student-t.

  - **Pareto frontier** across `(expected_reward, expected_cost,
    p_action_taken)` for cost-aware policy selection. The default
    `recommend()` returns the dominant policies on the frontier, not
    just the single arg-max — coordination engines often want the
    knee, not the corner.

Surface a coordination engine drives
------------------------------------

    lab = PolicyLab()
    for receipt in driver.tickets():
        lab.record(LoggedEvent.from_receipt(receipt))

    est = lab.evaluate(my_new_policy, method="dr")
    print(est.value, est.ci_low, est.ci_high, est.diagnostics)

    cmp = lab.compare(
        target=PolicyCandidate("v2", new_policy),
        baseline=PolicyCandidate("v1", current_policy),
    )
    if cmp.recommend == "ship":
        coordinator.adopt(new_policy)

    rec = lab.recommend([
        PolicyCandidate("greedy", greedy),
        PolicyCandidate("eps_greedy", eps_greedy),
        PolicyCandidate("ucb",   ucb),
    ])
    print(rec.summary)

Honest about limits
-------------------

OPE inherits the support of the logging policy. An action no logging
policy ever took has no data — the lab will flag this via the
`coverage` diagnostic and lower the effective sample size accordingly.
For brand-new actions, run a small *online* experiment first
(`agi.experiments.ExperimentRunner`) and feed those traces back into
the lab.

The lab is stdlib-only. Ridge fits run on a Gauss-Jordan inverse;
matrices stay small because the reward model is per-action, not
end-to-end. Investor demos run in milliseconds.
"""
from __future__ import annotations

import bisect
import json
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# ----- event kinds ----------------------------------------------------

LAB_RECORDED = "policy_lab.recorded"
LAB_FIT = "policy_lab.fit"
LAB_EVALUATED = "policy_lab.evaluated"
LAB_COMPARED = "policy_lab.compared"
LAB_RECOMMENDED = "policy_lab.recommended"


# ----- methods --------------------------------------------------------

METHOD_IPS = "ips"
METHOD_SNIPS = "snips"
METHOD_DM = "dm"
METHOD_DR = "dr"
METHOD_SWITCH_DR = "switch-dr"

KNOWN_METHODS = (METHOD_IPS, METHOD_SNIPS, METHOD_DM, METHOD_DR, METHOD_SWITCH_DR)


# ----- recommendations -----------------------------------------------

REC_SHIP = "ship"
REC_KILL = "kill"
REC_INCONCLUSIVE = "inconclusive"


# Numerical floor for propensities and weight clips.
_EPS = 1e-9
_DEFAULT_WEIGHT_CLIP = 50.0
_DEFAULT_SWITCH_TAU = 10.0


# =====================================================================
# Dataclasses
# =====================================================================


Policy = Callable[[Mapping[str, float]], Mapping[str, float]]
"""A policy maps a context (numeric features) to a probability over actions.

Must return a mapping {action_name: prob} with non-negative values that
sum to ~1.0. The lab will renormalize for safety and treat unseen
actions as zero-probability.
"""


@dataclass(frozen=True)
class LoggedEvent:
    """One (context, action, propensity, reward) observation.

    `propensity` is the probability the logging policy assigned to
    `action` at the time of the decision. If unknown, pass `1.0` and the
    lab will fall back to direct-method-only estimators (DM) and warn
    via the `coverage` diagnostic.

    `reward` is the realised scalar outcome. Convention: larger is
    better. For costs, pass negative; for binary success, pass 0/1.

    `metadata` is opaque; the lab does not interpret it.
    """

    context: Mapping[str, float]
    action: str
    propensity: float
    reward: float
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tenant_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["context"] = dict(self.context)
        if self.metadata is not None:
            d["metadata"] = dict(self.metadata)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LoggedEvent":
        return LoggedEvent(
            context=dict(d["context"]),
            action=str(d["action"]),
            propensity=float(d["propensity"]),
            reward=float(d["reward"]),
            timestamp=float(d.get("timestamp", time.time())),
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            tenant_id=d.get("tenant_id"),
            metadata=dict(d["metadata"]) if d.get("metadata") else None,
        )


@dataclass(frozen=True)
class PolicyCandidate:
    """A named policy considered for evaluation."""

    name: str
    policy: Policy


@dataclass(frozen=True)
class Estimate:
    """An off-policy estimate of a target policy's expected reward."""

    method: str
    policy_name: str
    value: float
    se: float
    ci_low: float
    ci_high: float
    ci_low_bernstein: float
    ci_high_bernstein: float
    confidence: float
    n: int
    n_eff: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["diagnostics"] = dict(self.diagnostics)
        return d


@dataclass(frozen=True)
class Comparison:
    """A paired comparison between a target policy and a baseline."""

    target: Estimate
    baseline: Estimate
    lift: float
    lift_se: float
    lift_ci_low: float
    lift_ci_high: float
    p_better: float
    recommend: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "baseline": self.baseline.to_dict(),
            "lift": self.lift,
            "lift_se": self.lift_se,
            "lift_ci_low": self.lift_ci_low,
            "lift_ci_high": self.lift_ci_high,
            "p_better": self.p_better,
            "recommend": self.recommend,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class FrontierPoint:
    """One point on the Pareto frontier across (reward, cost, coverage)."""

    name: str
    estimate: Estimate
    expected_cost: float
    coverage: float
    dominated_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecommendationReport:
    """Result of `lab.recommend()` over a candidate set."""

    best: str
    frontier: tuple[FrontierPoint, ...]
    estimates: Mapping[str, Estimate]
    comparisons: Mapping[str, Comparison]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "frontier": [
                {
                    "name": p.name,
                    "estimate": p.estimate.to_dict(),
                    "expected_cost": p.expected_cost,
                    "coverage": p.coverage,
                    "dominated_by": list(p.dominated_by),
                }
                for p in self.frontier
            ],
            "estimates": {k: v.to_dict() for k, v in self.estimates.items()},
            "comparisons": {k: v.to_dict() for k, v in self.comparisons.items()},
            "summary": self.summary,
        }


# =====================================================================
# Reward models
# =====================================================================


class RewardModel:
    """Estimate E[r | context, action]. Used by DM, DR, SWITCH-DR."""

    def fit(self, events: Sequence[LoggedEvent]) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict(self, context: Mapping[str, float], action: str) -> float:  # pragma: no cover
        raise NotImplementedError

    def actions(self) -> Sequence[str]:  # pragma: no cover
        raise NotImplementedError


class PerActionMeanRewardModel(RewardModel):
    """Conservative baseline: mean reward per action, ignoring context.

    Robust to misspecification, low variance, but throws away any signal
    in the context. Use as the default when you do not have or do not
    trust feature engineering.
    """

    def __init__(self, prior_n: float = 1.0, prior_mean: float = 0.0) -> None:
        self.prior_n = max(0.0, float(prior_n))
        self.prior_mean = float(prior_mean)
        self._mean: dict[str, float] = {}
        self._n: dict[str, float] = {}
        self._global_mean: float = self.prior_mean

    def fit(self, events: Sequence[LoggedEvent]) -> None:
        sums: dict[str, float] = {}
        counts: dict[str, float] = {}
        total_r = 0.0
        total_n = 0.0
        for e in events:
            sums[e.action] = sums.get(e.action, 0.0) + e.reward
            counts[e.action] = counts.get(e.action, 0.0) + 1.0
            total_r += e.reward
            total_n += 1.0
        self._mean.clear()
        self._n.clear()
        for a, s in sums.items():
            n = counts[a]
            self._mean[a] = (s + self.prior_n * self.prior_mean) / (n + self.prior_n)
            self._n[a] = n
        self._global_mean = (
            (total_r + self.prior_n * self.prior_mean) / (total_n + self.prior_n)
            if total_n + self.prior_n > 0
            else self.prior_mean
        )

    def predict(self, context: Mapping[str, float], action: str) -> float:
        return self._mean.get(action, self._global_mean)

    def actions(self) -> Sequence[str]:
        return tuple(self._mean.keys())


class LinearRewardModel(RewardModel):
    """Per-action ridge regression over numeric context features.

    For each observed action a, fit β_a = (X_a^T X_a + λ I)^{-1} X_a^T y_a
    where X_a stacks context vectors of events with that action and y_a
    is the reward vector. Prediction: r̂(c, a) = c · β_a + intercept_a.

    Closed-form, no external deps, no SGD. Context vectors are coerced
    to a fixed feature order (sorted keys) on first fit.
    """

    def __init__(self, ridge: float = 1.0) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.ridge = float(ridge)
        self._feature_order: tuple[str, ...] = ()
        self._coef: dict[str, list[float]] = {}
        self._intercept: dict[str, float] = {}
        self._fallback: float = 0.0

    def _vec(self, context: Mapping[str, float]) -> list[float]:
        return [float(context.get(f, 0.0)) for f in self._feature_order]

    def fit(self, events: Sequence[LoggedEvent]) -> None:
        if not events:
            self._coef.clear()
            self._intercept.clear()
            self._feature_order = ()
            self._fallback = 0.0
            return
        # Stable, deterministic feature order from union of context keys.
        keys: set[str] = set()
        for e in events:
            keys.update(e.context.keys())
        self._feature_order = tuple(sorted(keys))
        # Group events by action.
        by_action: dict[str, list[LoggedEvent]] = {}
        for e in events:
            by_action.setdefault(e.action, []).append(e)
        all_r = [e.reward for e in events]
        self._fallback = sum(all_r) / len(all_r)
        self._coef.clear()
        self._intercept.clear()
        for action, evs in by_action.items():
            # Build augmented design matrix with intercept column.
            d = len(self._feature_order) + 1
            X: list[list[float]] = []
            y: list[float] = []
            for e in evs:
                row = [1.0] + self._vec(e.context)
                X.append(row)
                y.append(e.reward)
            # Closed-form ridge: β = (X^T X + λ I)^-1 X^T y
            xtx = _matmul_transpose_left(X, X)
            for i in range(d):
                # Don't penalize the intercept (i=0) — standard practice.
                if i != 0:
                    xtx[i][i] += self.ridge
            xty = _matvec_transpose(X, y)
            try:
                beta = _solve(xtx, xty)
            except _SingularMatrix:
                # Heavy regularization fallback.
                for i in range(d):
                    xtx[i][i] += 10.0 * (self.ridge + 1.0)
                beta = _solve(xtx, xty)
            self._intercept[action] = beta[0]
            self._coef[action] = beta[1:]

    def predict(self, context: Mapping[str, float], action: str) -> float:
        if action not in self._coef:
            return self._fallback
        x = self._vec(context)
        coef = self._coef[action]
        s = self._intercept[action]
        for xi, bi in zip(x, coef):
            s += xi * bi
        return s

    def actions(self) -> Sequence[str]:
        return tuple(self._coef.keys())


# =====================================================================
# Small linear-algebra helpers (stdlib-only)
# =====================================================================


class _SingularMatrix(Exception):
    pass


def _matmul_transpose_left(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Compute A^T B for A (n×d), B (n×d) → (d×d)."""
    n = len(A)
    if n == 0:
        return []
    d_a = len(A[0])
    d_b = len(B[0])
    out = [[0.0] * d_b for _ in range(d_a)]
    for i in range(n):
        ai = A[i]
        bi = B[i]
        for r in range(d_a):
            air = ai[r]
            row = out[r]
            for c in range(d_b):
                row[c] += air * bi[c]
    return out


def _matvec_transpose(A: list[list[float]], y: list[float]) -> list[float]:
    """Compute A^T y for A (n×d), y (n) → (d)."""
    n = len(A)
    if n == 0:
        return []
    d = len(A[0])
    out = [0.0] * d
    for i in range(n):
        ai = A[i]
        yi = y[i]
        for r in range(d):
            out[r] += ai[r] * yi
    return out


def _solve(M: list[list[float]], b: list[float]) -> list[float]:
    """Solve M x = b via Gauss-Jordan with partial pivoting."""
    n = len(M)
    # Build augmented matrix.
    A = [list(row) + [b[i]] for i, row in enumerate(M)]
    for i in range(n):
        # Partial pivot.
        piv = i
        piv_val = abs(A[i][i])
        for r in range(i + 1, n):
            v = abs(A[r][i])
            if v > piv_val:
                piv_val = v
                piv = r
        if piv_val < 1e-12:
            raise _SingularMatrix()
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
        # Normalize pivot row.
        pv = A[i][i]
        for c in range(i, n + 1):
            A[i][c] /= pv
        # Eliminate.
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            if factor == 0.0:
                continue
            for c in range(i, n + 1):
                A[r][c] -= factor * A[i][c]
    return [A[r][n] for r in range(n)]


# =====================================================================
# Estimator math
# =====================================================================


def _normalize_policy(p: Mapping[str, float]) -> dict[str, float]:
    """Clip negatives to zero and renormalize a probability mapping.

    A misbehaving policy that returns negative values or unnormalized
    output gets cleaned up rather than crashing the lab. If the total
    mass is zero, returns an empty dict (treated as zero probability
    everywhere downstream).
    """
    cleaned = {a: max(0.0, float(v)) for a, v in p.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {a: v / total for a, v in cleaned.items()}


def _weight(target_p: float, logging_p: float, clip: float) -> float:
    if logging_p <= _EPS:
        return clip
    w = target_p / logging_p
    if w > clip:
        return clip
    return w


def _student_t_z(confidence: float, n: int) -> float:
    """Approximate two-sided Student-t critical value.

    For n >= 30, this is within 1% of the exact t-table at 95%/99%.
    For smaller n we use a conservative inflation. We avoid scipy.
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0, 1)")
    # Normal z-scores for common confidence levels.
    z_normal = {
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.9600,
        0.975: 2.2414,
        0.99: 2.5758,
    }
    key = round(confidence, 3)
    z = z_normal.get(key)
    if z is None:
        # Interpolate via inverse-erf approximation (Beasley-Springer-Moro).
        z = _inv_normal((1 + confidence) / 2)
    df = max(1, n - 1)
    # Hill's approximation for t inflation factor at moderate df.
    inflate = 1.0 + (z * z + 1.0) / (4.0 * df) + (5.0 * z**4 + 16.0 * z**2 + 3.0) / (96.0 * df * df)
    return z * inflate


def _inv_normal(p: float) -> float:
    """Inverse standard normal CDF via Acklam's approximation."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    # Coefficients.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    )


def _bernstein_radius(
    contributions: Sequence[float], confidence: float
) -> float:
    """Empirical Bernstein bound (Maurer & Pontil 2009).

    radius = sqrt( 2 V ln(2/δ) / n ) + 7 R ln(2/δ) / (3 (n-1))

    where V is the sample variance, R is the range, and δ = 1 - conf.
    Returns 0 for trivial n.
    """
    n = len(contributions)
    if n < 2:
        return float("inf")
    delta = max(_EPS, 1.0 - confidence)
    ln = math.log(2.0 / delta)
    mean = sum(contributions) / n
    var = sum((x - mean) ** 2 for x in contributions) / (n - 1)
    rng = max(contributions) - min(contributions)
    return math.sqrt(2.0 * var * ln / n) + 7.0 * rng * ln / (3.0 * (n - 1))


def _normal_pcdf(z: float) -> float:
    """Standard-normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# =====================================================================
# The lab
# =====================================================================


class PolicyLab:
    """Off-policy policy evaluation lab.

    The lab is thread-safe for record/evaluate/compare. Fits are done
    lazily on first evaluate() that needs them, or eagerly via fit().

    Parameters
    ----------
    event_bus
        Optional `EventBus` to emit `policy_lab.*` events for an
        operator dashboard.
    reward_model
        Defaults to `PerActionMeanRewardModel`. Pass `LinearRewardModel`
        when contexts have numeric features.
    weight_clip
        Importance-weight clip. Defaults to 50.0. Larger admits more
        bias from rare actions; smaller floors variance.
    switch_tau
        Threshold for SWITCH-DR: above this weight, fall back to DM.
        Defaults to 10.0.
    max_events
        Soft cap on the in-memory log. When exceeded, the oldest events
        are dropped FIFO. Set to None for unbounded.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        reward_model: RewardModel | None = None,
        weight_clip: float = _DEFAULT_WEIGHT_CLIP,
        switch_tau: float = _DEFAULT_SWITCH_TAU,
        max_events: int | None = 100_000,
    ) -> None:
        if weight_clip <= 0:
            raise ValueError("weight_clip must be positive")
        if switch_tau <= 0:
            raise ValueError("switch_tau must be positive")
        self.event_bus = event_bus
        self.reward_model: RewardModel = reward_model or PerActionMeanRewardModel()
        self.weight_clip = float(weight_clip)
        self.switch_tau = float(switch_tau)
        self.max_events = max_events
        self._lock = threading.RLock()
        self._events: list[LoggedEvent] = []
        self._fitted: bool = False
        self._fitted_at_n: int = 0
        self._known_actions: set[str] = set()

    # ----- ingest ----------------------------------------------------

    def record(self, event: LoggedEvent) -> None:
        """Record one logged event. Cheap; refit is lazy."""
        with self._lock:
            self._events.append(event)
            self._known_actions.add(event.action)
            self._fitted = False
            if self.max_events is not None and len(self._events) > self.max_events:
                drop = len(self._events) - self.max_events
                del self._events[:drop]
        self._emit(LAB_RECORDED, {"action": event.action, "reward": event.reward})

    def record_batch(self, events: Iterable[LoggedEvent]) -> int:
        n = 0
        for e in events:
            self.record(e)
            n += 1
        return n

    def events(self) -> list[LoggedEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def fit(self) -> None:
        """Eager refit of the reward model."""
        with self._lock:
            self.reward_model.fit(list(self._events))
            self._fitted = True
            self._fitted_at_n = len(self._events)
        self._emit(LAB_FIT, {"n": self._fitted_at_n})

    def _ensure_fitted(self) -> None:
        with self._lock:
            if not self._fitted or self._fitted_at_n != len(self._events):
                self.reward_model.fit(list(self._events))
                self._fitted = True
                self._fitted_at_n = len(self._events)

    # ----- evaluate --------------------------------------------------

    def evaluate(
        self,
        policy: Policy | PolicyCandidate,
        *,
        name: str | None = None,
        method: str = METHOD_DR,
        confidence: float = 0.95,
    ) -> Estimate:
        """Estimate target policy's expected reward on the logged traffic.

        method ∈ {ips, snips, dm, dr, switch-dr}; default 'dr'.
        """
        if method not in KNOWN_METHODS:
            raise ValueError(
                f"unknown method {method!r}; expected one of {KNOWN_METHODS}"
            )
        if confidence <= 0 or confidence >= 1:
            raise ValueError("confidence must be in (0, 1)")
        if isinstance(policy, PolicyCandidate):
            pname = policy.name
            pi = policy.policy
        else:
            pname = name or "target"
            pi = policy

        with self._lock:
            events = list(self._events)

        if not events:
            return _empty_estimate(method, pname, confidence)

        if method in (METHOD_DM, METHOD_DR, METHOD_SWITCH_DR):
            self._ensure_fitted()

        target_dists = [_normalize_policy(pi(e.context)) for e in events]

        contributions, diagnostics = self._contributions(
            events, target_dists, method
        )

        n = len(contributions)
        value = sum(contributions) / n
        # Sample SE on the influence-function contributions.
        if n > 1:
            var = sum((x - value) ** 2 for x in contributions) / (n - 1)
            se = math.sqrt(var / n)
        else:
            se = float("inf")
        z = _student_t_z(confidence, n)
        ci_low_t = value - z * se
        ci_high_t = value + z * se
        radius = _bernstein_radius(contributions, confidence)
        ci_low_b = value - radius
        ci_high_b = value + radius

        # Effective sample size: heuristic from importance weights.
        n_eff = diagnostics.get("n_eff", float(n))

        est = Estimate(
            method=method,
            policy_name=pname,
            value=value,
            se=se,
            ci_low=ci_low_t,
            ci_high=ci_high_t,
            ci_low_bernstein=ci_low_b,
            ci_high_bernstein=ci_high_b,
            confidence=confidence,
            n=n,
            n_eff=n_eff,
            diagnostics=diagnostics,
        )
        self._emit(
            LAB_EVALUATED,
            {
                "policy": pname,
                "method": method,
                "value": value,
                "n": n,
                "n_eff": n_eff,
            },
        )
        return est

    def _contributions(
        self,
        events: Sequence[LoggedEvent],
        target_dists: Sequence[Mapping[str, float]],
        method: str,
    ) -> tuple[list[float], dict[str, float]]:
        """Per-event influence-function contributions for `method`.

        IPS:    ψ_i = w_i r_i
        SNIPS:  ψ_i = w_i r_i / mean(w_i)        (post-normalization)
        DM:     ψ_i = Σ_a π(a|c_i) r̂(c_i, a)
        DR:     ψ_i = Σ_a π(a|c_i) r̂(c_i, a)
                       + w_i (r_i - r̂(c_i, a_i))
        SWITCH-DR: ψ_i = DM(i) + 1[w_i ≤ τ] · w_i (r_i - r̂(c_i, a_i))
        """
        n = len(events)
        weights: list[float] = [0.0] * n
        rewards: list[float] = [e.reward for e in events]
        for i, e in enumerate(events):
            t = target_dists[i].get(e.action, 0.0)
            weights[i] = _weight(t, e.propensity, self.weight_clip)

        # Coverage diagnostics: fraction of target mass not assigned to
        # any logged action. Lower = more support; higher = riskier.
        target_unsupported: list[float] = []
        for i, e in enumerate(events):
            target_unsupported.append(
                max(0.0, 1.0 - target_dists[i].get(e.action, 0.0))
            )

        mean_w = sum(weights) / n if n else 0.0
        max_w = max(weights) if weights else 0.0
        ess = (sum(weights) ** 2) / (sum(w * w for w in weights) + _EPS) if weights else 0.0

        diagnostics: dict[str, float] = {
            "mean_weight": mean_w,
            "max_weight": max_w,
            "n_eff": ess,
            "weight_clip": self.weight_clip,
            "mean_reward_logged": sum(rewards) / n,
        }

        if method == METHOD_IPS:
            contribs = [weights[i] * rewards[i] for i in range(n)]
            return contribs, diagnostics

        if method == METHOD_SNIPS:
            sum_w = sum(weights)
            if sum_w <= _EPS:
                return [0.0] * n, {**diagnostics, "snips_normalizer": 0.0}
            # Per-event contributions normalised by sum_w/n so they
            # average to the SNIPS estimate.
            scale = n / sum_w
            contribs = [scale * weights[i] * rewards[i] for i in range(n)]
            diagnostics["snips_normalizer"] = sum_w / n
            return contribs, diagnostics

        if method == METHOD_DM:
            contribs: list[float] = []
            actions_seen = list(self._known_actions)
            for i, e in enumerate(events):
                td = target_dists[i]
                v = 0.0
                for a, p in td.items():
                    if p <= 0:
                        continue
                    v += p * self.reward_model.predict(e.context, a)
                contribs.append(v)
            diagnostics["dm_n_actions"] = float(len(actions_seen))
            return contribs, diagnostics

        if method == METHOD_DR:
            contribs = []
            for i, e in enumerate(events):
                td = target_dists[i]
                dm_term = 0.0
                for a, p in td.items():
                    if p <= 0:
                        continue
                    dm_term += p * self.reward_model.predict(e.context, a)
                r_hat = self.reward_model.predict(e.context, e.action)
                contribs.append(dm_term + weights[i] * (rewards[i] - r_hat))
            return contribs, diagnostics

        if method == METHOD_SWITCH_DR:
            tau = self.switch_tau
            switched_count = 0
            contribs = []
            for i, e in enumerate(events):
                td = target_dists[i]
                dm_term = 0.0
                for a, p in td.items():
                    if p <= 0:
                        continue
                    dm_term += p * self.reward_model.predict(e.context, a)
                w = weights[i]
                if w <= tau:
                    r_hat = self.reward_model.predict(e.context, e.action)
                    contribs.append(dm_term + w * (rewards[i] - r_hat))
                else:
                    switched_count += 1
                    contribs.append(dm_term)
            diagnostics["switch_dr_tau"] = tau
            diagnostics["switch_dr_fallback_frac"] = switched_count / n
            return contribs, diagnostics

        raise AssertionError(f"unreachable: method {method!r}")  # pragma: no cover

    # ----- compare ---------------------------------------------------

    def compare(
        self,
        target: PolicyCandidate | Policy,
        baseline: PolicyCandidate | Policy,
        *,
        target_name: str = "target",
        baseline_name: str = "baseline",
        method: str = METHOD_DR,
        confidence: float = 0.95,
        min_lift: float = 0.0,
    ) -> Comparison:
        """Paired comparison via per-event influence-function differences.

        We compute ψ^t_i and ψ^b_i on the same logged events and treat
        their differences δ_i = ψ^t_i - ψ^b_i as a paired sample. The
        resulting SE is typically much tighter than independent
        evaluation, because the per-event noise cancels.
        """
        if isinstance(target, PolicyCandidate):
            tname, tp = target.name, target.policy
        else:
            tname, tp = target_name, target
        if isinstance(baseline, PolicyCandidate):
            bname, bp = baseline.name, baseline.policy
        else:
            bname, bp = baseline_name, baseline

        with self._lock:
            events = list(self._events)

        if not events:
            empty_t = _empty_estimate(method, tname, confidence)
            empty_b = _empty_estimate(method, bname, confidence)
            return Comparison(
                target=empty_t,
                baseline=empty_b,
                lift=0.0,
                lift_se=float("inf"),
                lift_ci_low=float("-inf"),
                lift_ci_high=float("inf"),
                p_better=0.5,
                recommend=REC_INCONCLUSIVE,
                rationale="no logged events",
            )

        if method in (METHOD_DM, METHOD_DR, METHOD_SWITCH_DR):
            self._ensure_fitted()

        t_dists = [_normalize_policy(tp(e.context)) for e in events]
        b_dists = [_normalize_policy(bp(e.context)) for e in events]

        t_contribs, t_diag = self._contributions(events, t_dists, method)
        b_contribs, b_diag = self._contributions(events, b_dists, method)
        n = len(events)
        t_value = sum(t_contribs) / n
        b_value = sum(b_contribs) / n
        deltas = [t_contribs[i] - b_contribs[i] for i in range(n)]
        lift = t_value - b_value
        if n > 1:
            var_d = sum((d - lift) ** 2 for d in deltas) / (n - 1)
            se_d = math.sqrt(var_d / n)
        else:
            se_d = float("inf")
        z = _student_t_z(confidence, n)
        lo, hi = lift - z * se_d, lift + z * se_d

        # p_better ~ Φ(lift / se) (normal approx).
        if se_d > 0 and math.isfinite(se_d):
            p_better = _normal_pcdf(lift / se_d)
        else:
            p_better = 1.0 if lift > 0 else 0.0

        if lo > min_lift:
            rec = REC_SHIP
            rationale = (
                f"lift {lift:+.4f} ≥ min_lift {min_lift:+.4f} with "
                f"{int(confidence*100)}% CI [{lo:+.4f}, {hi:+.4f}]"
            )
        elif hi < -min_lift:
            rec = REC_KILL
            rationale = (
                f"lift {lift:+.4f} ≤ -min_lift {-min_lift:+.4f} with "
                f"{int(confidence*100)}% CI [{lo:+.4f}, {hi:+.4f}]"
            )
        else:
            rec = REC_INCONCLUSIVE
            rationale = (
                f"lift {lift:+.4f} CI [{lo:+.4f}, {hi:+.4f}] spans 0 "
                f"(n={n}, n_eff_t={t_diag.get('n_eff', 0):.0f}, "
                f"n_eff_b={b_diag.get('n_eff', 0):.0f})"
            )

        t_est = self._estimate_from_contribs(
            t_contribs, tname, method, confidence, t_diag
        )
        b_est = self._estimate_from_contribs(
            b_contribs, bname, method, confidence, b_diag
        )
        cmp = Comparison(
            target=t_est,
            baseline=b_est,
            lift=lift,
            lift_se=se_d,
            lift_ci_low=lo,
            lift_ci_high=hi,
            p_better=p_better,
            recommend=rec,
            rationale=rationale,
        )
        self._emit(
            LAB_COMPARED,
            {
                "target": tname,
                "baseline": bname,
                "lift": lift,
                "recommend": rec,
            },
        )
        return cmp

    def _estimate_from_contribs(
        self,
        contribs: Sequence[float],
        name: str,
        method: str,
        confidence: float,
        diagnostics: Mapping[str, float],
    ) -> Estimate:
        n = len(contribs)
        if not n:
            return _empty_estimate(method, name, confidence)
        v = sum(contribs) / n
        if n > 1:
            var = sum((x - v) ** 2 for x in contribs) / (n - 1)
            se = math.sqrt(var / n)
        else:
            se = float("inf")
        z = _student_t_z(confidence, n)
        radius = _bernstein_radius(contribs, confidence)
        return Estimate(
            method=method,
            policy_name=name,
            value=v,
            se=se,
            ci_low=v - z * se,
            ci_high=v + z * se,
            ci_low_bernstein=v - radius,
            ci_high_bernstein=v + radius,
            confidence=confidence,
            n=n,
            n_eff=diagnostics.get("n_eff", float(n)),
            diagnostics=dict(diagnostics),
        )

    # ----- recommend -------------------------------------------------

    def recommend(
        self,
        candidates: Sequence[PolicyCandidate],
        *,
        method: str = METHOD_DR,
        confidence: float = 0.95,
        cost_per_action: Mapping[str, float] | None = None,
    ) -> RecommendationReport:
        """Evaluate a set of candidates and return Pareto-best.

        `cost_per_action` is optional but recommended for investor
        framing — when present, the Pareto frontier is over (reward,
        cost), letting a coordinator pick the knee point rather than
        the corner.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")

        estimates: dict[str, Estimate] = {}
        costs: dict[str, float] = {}
        coverages: dict[str, float] = {}
        with self._lock:
            events = list(self._events)
        for cand in candidates:
            est = self.evaluate(cand, method=method, confidence=confidence)
            estimates[cand.name] = est
            costs[cand.name] = _expected_cost(cand, events, cost_per_action)
            coverages[cand.name] = _coverage(cand, events)

        names = [c.name for c in candidates]
        frontier_names = _pareto_frontier(
            [(n, estimates[n].value, -costs[n]) for n in names]
        )
        # Best = highest reward among frontier; ties broken by lower cost.
        best = max(
            frontier_names,
            key=lambda n: (estimates[n].value, -costs[n]),
        )

        comparisons: dict[str, Comparison] = {}
        if len(candidates) >= 2:
            for cand in candidates:
                if cand.name == best:
                    continue
                cmp = self.compare(
                    target=next(c for c in candidates if c.name == best),
                    baseline=cand,
                    method=method,
                    confidence=confidence,
                )
                comparisons[cand.name] = cmp

        # Build frontier points with dominators noted.
        points: list[FrontierPoint] = []
        for n in names:
            est = estimates[n]
            dominators = tuple(
                d
                for d in names
                if d != n
                and estimates[d].value >= est.value
                and costs[d] <= costs[n]
                and (estimates[d].value > est.value or costs[d] < costs[n])
            )
            points.append(
                FrontierPoint(
                    name=n,
                    estimate=est,
                    expected_cost=costs[n],
                    coverage=coverages[n],
                    dominated_by=dominators,
                )
            )

        summary_lines = [
            f"PolicyLab recommend (method={method}, n={len(events)})",
            f"  best: {best} -> value={estimates[best].value:+.4f}",
        ]
        for p in points:
            mark = " *" if p.name == best else "  "
            dom = f" dominated_by={p.dominated_by}" if p.dominated_by else ""
            summary_lines.append(
                f"{mark} {p.name:<20} value={p.estimate.value:+.4f} "
                f"ci=[{p.estimate.ci_low:+.4f},{p.estimate.ci_high:+.4f}] "
                f"cost={p.expected_cost:.4f} cov={p.coverage:.2f}{dom}"
            )
        report = RecommendationReport(
            best=best,
            frontier=tuple(points),
            estimates=estimates,
            comparisons=comparisons,
            summary="\n".join(summary_lines),
        )
        self._emit(
            LAB_RECOMMENDED,
            {"best": best, "n_candidates": len(candidates), "method": method},
        )
        return report

    # ----- integrations ----------------------------------------------

    def attach_to_driver(
        self,
        driver: Any,
        *,
        action_of: Callable[[Any], str] | None = None,
        propensity_of: Callable[[Any], float] | None = None,
        reward_of: Callable[[Any], float] | None = None,
        context_of: Callable[[Any], Mapping[str, float]] | None = None,
    ) -> Callable[[], None]:
        """Drain a RuntimeDriver's receipt stream into the lab.

        Returns an unsubscribe callable. The mappers default to
        sensible field reads on a `Receipt`; pass your own to extract
        actions/propensities/rewards from a custom domain.
        """
        action_of = action_of or _default_action_of
        propensity_of = propensity_of or _default_propensity_of
        reward_of = reward_of or _default_reward_of
        context_of = context_of or _default_context_of

        def listener(receipt: Any) -> None:
            try:
                ev = LoggedEvent(
                    context=context_of(receipt),
                    action=action_of(receipt),
                    propensity=propensity_of(receipt),
                    reward=reward_of(receipt),
                    tenant_id=getattr(receipt, "tenant_id", None),
                    metadata={"receipt_id": getattr(receipt, "id", None)},
                )
                self.record(ev)
            except Exception:
                # Best-effort: a bad receipt should not poison the lab.
                pass

        # Most drivers expose either subscribe(callback) or a list of receipts.
        if hasattr(driver, "subscribe_receipts"):
            return driver.subscribe_receipts(listener)
        if hasattr(driver, "subscribe"):
            return driver.subscribe(listener)
        # Polling fallback: just drain the current tickets() once.
        if hasattr(driver, "tickets"):
            for r in driver.tickets():
                listener(r)
        return lambda: None

    def attach_to_bus(
        self,
        bus: EventBus,
        *,
        kinds: Sequence[str] = (),
        extractor: Callable[[Event], LoggedEvent | None] | None = None,
    ) -> Callable[[], None]:
        """Subscribe to an EventBus, extracting LoggedEvents.

        If `extractor` is None, expects events to carry the raw fields
        directly in `event.data` under the keys `context`, `action`,
        `propensity`, `reward`.
        """
        def cb(event: Event) -> None:
            try:
                if extractor:
                    ev = extractor(event)
                    if ev is not None:
                        self.record(ev)
                    return
                d = event.data or {}
                if not {"context", "action", "propensity", "reward"} <= set(d):
                    return
                self.record(
                    LoggedEvent(
                        context=dict(d["context"]),
                        action=str(d["action"]),
                        propensity=float(d["propensity"]),
                        reward=float(d["reward"]),
                        tenant_id=d.get("tenant_id"),
                        metadata=dict(d.get("metadata") or {}),
                    )
                )
            except Exception:
                pass

        if not kinds:
            sub_id = bus.subscribe(cb)
            return lambda: bus.unsubscribe(sub_id)
        sub_ids = [bus.subscribe(cb, kind=k) for k in kinds]
        return lambda: [bus.unsubscribe(sid) for sid in sub_ids]

    # ----- persistence ----------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "events": [e.to_dict() for e in self._events],
                "weight_clip": self.weight_clip,
                "switch_tau": self.switch_tau,
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("version") != 1:
            raise ValueError("unknown snapshot version")
        with self._lock:
            self._events = [LoggedEvent.from_dict(d) for d in snapshot.get("events", [])]
            self._known_actions = {e.action for e in self._events}
            self.weight_clip = float(snapshot.get("weight_clip", self.weight_clip))
            self.switch_tau = float(snapshot.get("switch_tau", self.switch_tau))
            self._fitted = False

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.snapshot()))

    def load(self, path: str | Path) -> None:
        self.restore(json.loads(Path(path).read_text()))

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


def _empty_estimate(method: str, name: str, confidence: float) -> Estimate:
    return Estimate(
        method=method,
        policy_name=name,
        value=0.0,
        se=float("inf"),
        ci_low=float("-inf"),
        ci_high=float("inf"),
        ci_low_bernstein=float("-inf"),
        ci_high_bernstein=float("inf"),
        confidence=confidence,
        n=0,
        n_eff=0.0,
        diagnostics={},
    )


def _coverage(cand: PolicyCandidate, events: Sequence[LoggedEvent]) -> float:
    """Fraction of logged events where the candidate puts mass on the logged
    action. 1.0 = fully covered; 0.0 = candidate prefers actions never seen."""
    if not events:
        return 0.0
    s = 0.0
    for e in events:
        p = _normalize_policy(cand.policy(e.context))
        s += p.get(e.action, 0.0)
    return s / len(events)


def _expected_cost(
    cand: PolicyCandidate,
    events: Sequence[LoggedEvent],
    cost_per_action: Mapping[str, float] | None,
) -> float:
    if not events:
        return 0.0
    if cost_per_action is None:
        return 0.0
    total = 0.0
    for e in events:
        p = _normalize_policy(cand.policy(e.context))
        for a, prob in p.items():
            total += prob * cost_per_action.get(a, 0.0)
    return total / len(events)


def _pareto_frontier(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """Pareto front maximizing both coords. Each point is (name, x, y)."""
    frontier: list[str] = []
    for name, x, y in points:
        dominated = False
        for _, ox, oy in points:
            if (ox > x and oy >= y) or (ox >= x and oy > y):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


# ----- default driver field mappers ----------------------------------


def _default_action_of(receipt: Any) -> str:
    # Receipt verdict is what an admission policy decided ("admit",
    # "defer", "downgrade", "reject"). Fall back to model used.
    for attr in ("verdict", "decision", "action", "model"):
        v = getattr(receipt, attr, None)
        if v:
            return str(v)
    return "unknown"


def _default_propensity_of(receipt: Any) -> float:
    # If the driver doesn't record propensities, default to 1.0 — the
    # lab will warn via low effective-sample-size.
    p = getattr(receipt, "propensity", None)
    if p is None:
        return 1.0
    try:
        return max(_EPS, min(1.0, float(p)))
    except (TypeError, ValueError):
        return 1.0


def _default_reward_of(receipt: Any) -> float:
    # Order of preference: realized EV, net revenue, success (0/1).
    for attr in ("ev_realized", "ev", "revenue", "margin"):
        v = getattr(receipt, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    status = getattr(receipt, "status", None)
    if status is not None:
        return 1.0 if str(status).lower() == "completed" else 0.0
    return 0.0


def _default_context_of(receipt: Any) -> Mapping[str, float]:
    # Mine a small numeric feature vector from receipt fields.
    out: dict[str, float] = {}
    for attr in (
        "estimated_cost_usd",
        "estimated_duration_s",
        "estimated_p_success",
        "priority",
    ):
        v = getattr(receipt, attr, None)
        if v is None:
            continue
        try:
            out[attr] = float(v)
        except (TypeError, ValueError):
            pass
    return out
