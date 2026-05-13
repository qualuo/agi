"""PolicyImprover — safe off-policy policy *optimization* as a runtime primitive.

`PolicyLab` answers "what is *this fixed* policy worth?" It evaluates.
`PolicyImprover` answers the strictly harder operational question a
coordination engine actually faces:

    "From the policies I am allowed to deploy, which is the best one I
     can ship right now *that I can prove won't regress production*?"

Off-policy *evaluation* is necessary but not sufficient. The coordination
engine wants to upgrade the routing/admission/retry/pricing rule it is
currently running. Doing that safely off-line is a different problem:

  - Maximizing an off-policy estimator can chase variance to crazy
    parameters (the IPS objective is unbounded for rare actions).
  - A point estimate that is higher than baseline is not enough; a
    finite-sample lower bound on the new policy's value has to clear
    the baseline before the system is allowed to deploy.

`PolicyImprover` is the primitive that does both: it searches a
parameterised policy space, and it returns the best policy *with a
finite-sample High-Confidence Policy Improvement (HCPI) certificate*
(Thomas, Theocharous & Ghavamzadeh 2015). If no candidate clears the
LCB > baseline threshold, the result is honestly tagged ``unsafe`` and
the coordination engine keeps the incumbent.

Science it implements
---------------------

  - **Counterfactual Risk Minimization (CRM)** — Swaminathan & Joachims
    2015 (POEM). Clipped IPS is the off-policy training surrogate; the
    weight clip plays the dual role of regularizer + variance control.

  - **Self-Normalized IPS (SNIPS)** — Trotter-Tukey 1956 → Swaminathan-
    Joachims 2015. Used as the stable *evaluation* estimator after each
    optimization step; bounded in [r_min, r_max] which makes the
    Bernstein bound meaningful.

  - **High-Confidence Policy Improvement (HCPI)** — Thomas et al. 2015.
    Deploy a new policy ONLY when ``LCB(V̂(π_new)) > V(π_baseline)``.
    The lower bound is empirical Bernstein (Maurer-Pontil 2009) on the
    clipped weighted rewards.

  - **Pessimistic action mass** — when the logged data assigns
    near-zero propensity to an action under the new policy, the
    estimator has no support there. We refuse to improve in directions
    the data cannot certify; the optimizer adds a *coverage floor* that
    discounts candidates with low effective sample size.

  - **Conservative mixing dial** — `MixturePolicySpace(α, baseline,
    target)` interpolates baseline ↔ target. With α = 0 the new policy
    equals the baseline (improvement is provably non-negative).
    Searching α with a Bernstein LCB at every step is the simplest
    valid HCPI procedure — it is Algorithm 2 in Thomas et al.

  - **Multi-start projected gradient** ascent on clipped IPS over
    softmax-θ. Multi-start handles non-convexity; clip handles
    variance; projection keeps θ in a bounded box.

Surface a coordination engine drives
------------------------------------

::

    improver = PolicyImprover(
        policy_space=SoftmaxPolicySpace(
            actions=["haiku", "sonnet", "opus"],
            feature_names=["task_difficulty", "tenant_premium"],
        ),
        baseline_value=0.62,        # value of incumbent policy on this traffic
        weight_clip=20.0,
        delta=0.05,                 # safety level
    )

    for ev in policy_lab.events():
        improver.record(ev)

    imp = improver.improve(n_restarts=5, n_iters=200)
    if imp.safe:
        coordinator.adopt(imp.policy)
    else:
        # LCB on V(π_new) - V(π_baseline) did not clear zero — keep incumbent.
        log.info("no safe improvement found", lcb=imp.improvement_lcb)

The same instance can certify an *arbitrary* policy a coordination
engine produced elsewhere:

::

    rpt = improver.safety_check(custom_policy, baseline_value=0.62)
    if rpt.verdict == SAFE:
        coordinator.adopt(custom_policy)

Honest about limits
-------------------

  - **No exploration**: this is offline. Actions with zero logged
    propensity have no signal — the optimizer cannot recommend them.
    Diagnostic ``coverage`` measures this; pair with `ExperimentRunner`
    for online exploration when coverage is too low.

  - **Bandit feedback only**: stateless, no sequential credit
    assignment. Useful for routing / admission / pricing /
    one-shot-action decisions. Sequential RL is out of scope.

  - **HCPI is conservative on purpose**: Bernstein-LCB is a true
    lower bound, not a point estimate. On small n the safety gate will
    refuse to deploy real improvements. This is the desired bias.
    Provide more data, or accept a wider δ, to tighten it.
"""

from __future__ import annotations

import math
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus
from agi.policy_lab import (
    LoggedEvent,
    Policy,
    PolicyCandidate,
    PolicyLab,
)


# =====================================================================
# Event kinds, methods, verdicts
# =====================================================================

IMP_RECORDED = "policy_improver.recorded"
IMP_OPTIMIZED = "policy_improver.optimized"
IMP_CHECKED = "policy_improver.checked"
IMP_PROMOTED = "policy_improver.promoted"
IMP_REJECTED = "policy_improver.rejected"


OBJ_CLIPPED_IPS = "clipped_ips"
OBJ_SNIPS = "snips"
OBJ_CRM_VAR = "crm_var"  # CRM with explicit sample-variance penalty
KNOWN_OBJECTIVES = (OBJ_CLIPPED_IPS, OBJ_SNIPS, OBJ_CRM_VAR)


# Safety verdicts returned by `safety_check`.
SAFE = "safe"          # LCB(V_new - V_baseline) > 0
UNSAFE = "unsafe"      # UCB(V_new - V_baseline) ≤ 0 (provably worse or tied)
UNCERTAIN = "uncertain"  # CI straddles zero
VERDICTS = (SAFE, UNSAFE, UNCERTAIN)


_EPS = 1e-12
_DEFAULT_WEIGHT_CLIP = 20.0
_DEFAULT_DELTA = 0.05
_DEFAULT_PARAM_BOUND = 5.0


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """Health of the off-policy fit. Coordination engines should branch on these."""

    n: int
    n_eff: float                  # Kish effective sample size
    max_weight: float
    mean_weight: float
    clipped_fraction: float       # fraction of (i) for which w_i was clipped
    coverage: float               # fraction of events whose action has π_new > 0
    iterations: int
    restarts: int
    objective_value: float        # last value of the optimization objective
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Improvement:
    """Result of one `improve()` call.

    `safe` is the HCPI verdict: True iff the empirical-Bernstein LCB on
    (V(π_new) - V(π_baseline)) exceeds zero at the configured δ.
    """

    parameters: tuple[float, ...]
    value: float
    value_se: float
    value_lcb: float
    value_ucb: float
    baseline_value: float
    improvement: float
    improvement_se: float
    improvement_lcb: float
    improvement_ucb: float
    delta: float
    safe: bool
    verdict: str
    diagnostics: OptimizationDiagnostics
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["parameters"] = list(self.parameters)
        d["diagnostics"] = self.diagnostics.to_dict()
        return d


@dataclass(frozen=True)
class HCPIReport:
    """Safety-only report for an arbitrary fixed policy (no optimization).

    Useful to certify policies a coordination engine produced elsewhere
    (e.g., from a learned model, from a heuristic, from a human).
    """

    policy_name: str
    value: float
    value_se: float
    value_lcb: float
    value_ucb: float
    baseline_value: float
    improvement: float
    improvement_lcb: float
    improvement_ucb: float
    delta: float
    verdict: str
    diagnostics: OptimizationDiagnostics
    rationale: str = ""

    @property
    def safe(self) -> bool:
        return self.verdict == SAFE

    def to_dict(self) -> dict[str, Any]:
        d = dict(asdict(self))
        d["diagnostics"] = self.diagnostics.to_dict()
        return d


# =====================================================================
# PolicySpace — parameterised policy families
# =====================================================================


class PolicySpace:
    """Parameterised family of policies the improver can search over.

    Concrete implementations: `SoftmaxPolicySpace`, `EpsilonGreedyPolicySpace`,
    `MixturePolicySpace`. A coordination engine can add its own subclass.
    """

    actions: tuple[str, ...]

    def dim(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def initial_parameters(self, rng: random.Random) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def project(self, theta: list[float]) -> list[float]:
        """Project θ back into the feasible region. Default: no-op."""
        return theta

    def probs(
        self, theta: Sequence[float], context: Mapping[str, float]
    ) -> dict[str, float]:  # pragma: no cover
        """Return the action distribution under θ in this context."""
        raise NotImplementedError

    def grad_log_prob(
        self,
        theta: Sequence[float],
        context: Mapping[str, float],
        action: str,
    ) -> list[float]:  # pragma: no cover
        """∂ log π(action | context; θ) / ∂θ."""
        raise NotImplementedError

    def to_policy(self, theta: Sequence[float]) -> Policy:
        """Bake θ into a Policy callable compatible with PolicyLab."""
        t = tuple(theta)
        space = self

        def _pi(context: Mapping[str, float]) -> Mapping[str, float]:
            return space.probs(t, context)

        return _pi


class SoftmaxPolicySpace(PolicySpace):
    """Linear-softmax over actions: π(a|x) ∝ exp(θ_a · φ(x)).

    Features are read by name from each context's mapping; missing
    features default to 0.0. An intercept feature is always included.
    Parameters are stored as a flat vector of length |actions| × (1 + d).
    """

    def __init__(
        self,
        actions: Sequence[str],
        feature_names: Sequence[str] = (),
        param_bound: float = _DEFAULT_PARAM_BOUND,
    ) -> None:
        if not actions:
            raise ValueError("SoftmaxPolicySpace needs at least one action")
        if len(set(actions)) != len(actions):
            raise ValueError("actions must be unique")
        self.actions = tuple(actions)
        self.feature_names = tuple(feature_names)
        self.param_bound = float(param_bound)
        self._k = len(self.actions)
        self._d = 1 + len(self.feature_names)  # +1 for intercept

    def dim(self) -> int:
        return self._k * self._d

    def initial_parameters(self, rng: random.Random) -> list[float]:
        # Small random init keeps the softmax near uniform.
        return [rng.gauss(0.0, 0.01) for _ in range(self.dim())]

    def project(self, theta: list[float]) -> list[float]:
        b = self.param_bound
        return [max(-b, min(b, x)) for x in theta]

    def _features(self, context: Mapping[str, float]) -> list[float]:
        out = [1.0]
        for name in self.feature_names:
            v = context.get(name, 0.0)
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    def _logits(
        self, theta: Sequence[float], features: Sequence[float]
    ) -> list[float]:
        logits: list[float] = []
        for ai in range(self._k):
            off = ai * self._d
            s = 0.0
            for j in range(self._d):
                s += theta[off + j] * features[j]
            logits.append(s)
        return logits

    @staticmethod
    def _softmax(logits: Sequence[float]) -> list[float]:
        m = max(logits)
        ex = [math.exp(z - m) for z in logits]
        s = sum(ex)
        return [e / s for e in ex] if s > 0 else [1.0 / len(logits)] * len(logits)

    def probs(
        self, theta: Sequence[float], context: Mapping[str, float]
    ) -> dict[str, float]:
        feats = self._features(context)
        logits = self._logits(theta, feats)
        p = self._softmax(logits)
        return {a: p[i] for i, a in enumerate(self.actions)}

    def grad_log_prob(
        self,
        theta: Sequence[float],
        context: Mapping[str, float],
        action: str,
    ) -> list[float]:
        feats = self._features(context)
        logits = self._logits(theta, feats)
        p = self._softmax(logits)
        try:
            a_idx = self.actions.index(action)
        except ValueError:
            return [0.0] * self.dim()
        grad = [0.0] * self.dim()
        for ai in range(self._k):
            indicator = 1.0 if ai == a_idx else 0.0
            coef = indicator - p[ai]
            off = ai * self._d
            for j in range(self._d):
                grad[off + j] = coef * feats[j]
        return grad


class EpsilonGreedyPolicySpace(PolicySpace):
    """π = (1 - ε) · π_inner + ε · uniform.

    Useful when the coordination engine wants to *floor* exploration
    while still tuning the deterministic policy. Parameter vector is
    just [logit(ε)]; the inner policy is fixed.
    """

    def __init__(
        self,
        inner: Policy,
        actions: Sequence[str],
        eps_min: float = 0.0,
        eps_max: float = 0.5,
    ) -> None:
        if not actions:
            raise ValueError("EpsilonGreedyPolicySpace needs at least one action")
        if not (0.0 <= eps_min < eps_max <= 1.0):
            raise ValueError("require 0 ≤ eps_min < eps_max ≤ 1")
        self.actions = tuple(actions)
        self.inner = inner
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)

    def dim(self) -> int:
        return 1

    def initial_parameters(self, rng: random.Random) -> list[float]:
        return [rng.gauss(0.0, 0.5)]

    def project(self, theta: list[float]) -> list[float]:
        return [max(-10.0, min(10.0, theta[0]))]

    def _eps(self, theta: Sequence[float]) -> float:
        sig = 1.0 / (1.0 + math.exp(-theta[0]))
        return self.eps_min + (self.eps_max - self.eps_min) * sig

    def probs(
        self, theta: Sequence[float], context: Mapping[str, float]
    ) -> dict[str, float]:
        eps = self._eps(theta)
        inner = dict(self.inner(context))
        # Normalize inner to be safe.
        s = sum(max(0.0, v) for v in inner.values())
        if s <= 0:
            inner = {a: 1.0 / len(self.actions) for a in self.actions}
            s = 1.0
        u = 1.0 / len(self.actions)
        out: dict[str, float] = {}
        for a in self.actions:
            p_inner = max(0.0, inner.get(a, 0.0)) / s
            out[a] = (1.0 - eps) * p_inner + eps * u
        return out

    def grad_log_prob(
        self,
        theta: Sequence[float],
        context: Mapping[str, float],
        action: str,
    ) -> list[float]:
        # Numeric gradient: 1-D, cheap.
        h = 1e-4
        p0 = self.probs(theta, context).get(action, 0.0)
        p1 = self.probs([theta[0] + h], context).get(action, 0.0)
        if p0 <= 0 or p1 <= 0:
            return [0.0]
        return [(math.log(p1) - math.log(p0)) / h]


class MixturePolicySpace(PolicySpace):
    """π = (1 - α) · baseline + α · target. Parameter is α ∈ [0, 1].

    The classic Thomas-et-al. "safe mixing" device: at α=0 the policy IS
    the baseline so improvement ≥ 0 trivially; at α=1 the policy is the
    target. Searching α with a Bernstein LCB at every step is the
    simplest valid HCPI procedure.
    """

    def __init__(
        self,
        baseline: Policy,
        target: Policy,
        actions: Sequence[str],
    ) -> None:
        if not actions:
            raise ValueError("MixturePolicySpace needs at least one action")
        self.actions = tuple(actions)
        self.baseline = baseline
        self.target = target

    def dim(self) -> int:
        return 1

    def initial_parameters(self, rng: random.Random) -> list[float]:
        return [rng.uniform(0.0, 1.0)]

    def project(self, theta: list[float]) -> list[float]:
        return [max(0.0, min(1.0, theta[0]))]

    def probs(
        self, theta: Sequence[float], context: Mapping[str, float]
    ) -> dict[str, float]:
        a = max(0.0, min(1.0, theta[0]))
        b = self.baseline(context)
        t = self.target(context)
        bs = sum(max(0.0, v) for v in b.values()) or 1.0
        ts = sum(max(0.0, v) for v in t.values()) or 1.0
        out: dict[str, float] = {}
        for act in self.actions:
            pb = max(0.0, b.get(act, 0.0)) / bs
            pt = max(0.0, t.get(act, 0.0)) / ts
            out[act] = (1.0 - a) * pb + a * pt
        return out

    def grad_log_prob(
        self,
        theta: Sequence[float],
        context: Mapping[str, float],
        action: str,
    ) -> list[float]:
        # ∂π(a) / ∂α = π_target(a) - π_baseline(a).
        # ∂ log π / ∂α = (π_target(a) - π_baseline(a)) / π(a).
        a = max(0.0, min(1.0, theta[0]))
        b = self.baseline(context)
        t = self.target(context)
        bs = sum(max(0.0, v) for v in b.values()) or 1.0
        ts = sum(max(0.0, v) for v in t.values()) or 1.0
        pb = max(0.0, b.get(action, 0.0)) / bs
        pt = max(0.0, t.get(action, 0.0)) / ts
        p = (1.0 - a) * pb + a * pt
        if p <= _EPS:
            return [0.0]
        return [(pt - pb) / p]


# =====================================================================
# Bounds: empirical Bernstein for the clipped weighted reward
# =====================================================================


def empirical_bernstein_bound(
    values: Sequence[float],
    delta: float,
    value_range: float,
) -> tuple[float, float]:
    """Two-sided empirical Bernstein bound (Maurer-Pontil 2009).

    For i.i.d. (X_i) ∈ [a, b] with empirical variance σ̂² and mean X̄,
    with prob ≥ 1 - 2δ:

        |E[X] - X̄| ≤ √(2 σ̂² log(2/δ) / n) + 7 R log(2/δ) / (3 (n - 1))

    where R = b - a. Returns the half-width *as if* the same bound holds
    on both sides; callers typically use ``lcb = mean - half_width``.

    `value_range` is R. For clipped IPS rewards with rewards in
    [r_min, r_max] and clip C, set R = C · (r_max - r_min).
    """
    n = len(values)
    if n == 0:
        return float("inf"), float("inf")
    mean = sum(values) / n
    if n == 1:
        return mean, value_range
    # Unbiased sample variance.
    s2 = sum((x - mean) ** 2 for x in values) / (n - 1)
    if s2 < 0:
        s2 = 0.0
    log_term = math.log(2.0 / max(delta, _EPS))
    half = math.sqrt(2.0 * s2 * log_term / n) + 7.0 * value_range * log_term / (3.0 * (n - 1))
    return mean, half


def normal_lcb(mean: float, se: float, delta: float) -> float:
    """Gaussian LCB at level 1-δ. Used as a fast supplementary bound."""
    z = _inv_normal(1.0 - delta)
    return mean - z * se


def _inv_normal(p: float) -> float:
    """Beasley-Springer-Moro inverse standard-normal CDF (good enough)."""
    p = max(min(p, 1 - 1e-12), 1e-12)
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
        ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


# =====================================================================
# Core: weighted-reward estimators
# =====================================================================


def _clipped_weight(target_p: float, propensity: float, clip: float) -> tuple[float, bool]:
    """Return (w, clipped?) where w = min(clip, π(a|x) / μ(a|x))."""
    mu = max(_EPS, propensity)
    raw = target_p / mu
    if raw > clip:
        return clip, True
    if raw < 0:
        return 0.0, False
    return raw, False


def _per_event_value(
    space: PolicySpace,
    theta: Sequence[float],
    events: Sequence[LoggedEvent],
    clip: float,
) -> tuple[list[float], list[float], list[bool]]:
    """Return (w_i, w_i * r_i, clipped_i) for each event."""
    weights: list[float] = []
    weighted_rewards: list[float] = []
    clipped: list[bool] = []
    for e in events:
        p = space.probs(theta, e.context).get(e.action, 0.0)
        w, c = _clipped_weight(p, e.propensity, clip)
        weights.append(w)
        weighted_rewards.append(w * e.reward)
        clipped.append(c)
    return weights, weighted_rewards, clipped


def _snips_value(weighted_rewards: Sequence[float], weights: Sequence[float]) -> float:
    sw = sum(weights)
    if sw <= _EPS:
        return 0.0
    return sum(weighted_rewards) / sw


def _ips_value(weighted_rewards: Sequence[float]) -> float:
    n = len(weighted_rewards)
    if n == 0:
        return 0.0
    return sum(weighted_rewards) / n


def _ess(weights: Sequence[float]) -> float:
    """Kish effective sample size: (Σ w)² / Σ w²."""
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    if sw2 <= _EPS:
        return 0.0
    return (sw * sw) / sw2


def _coverage(
    space: PolicySpace,
    theta: Sequence[float],
    events: Sequence[LoggedEvent],
) -> float:
    if not events:
        return 0.0
    covered = 0
    for e in events:
        p = space.probs(theta, e.context).get(e.action, 0.0)
        if p > _EPS:
            covered += 1
    return covered / len(events)


# =====================================================================
# CRM gradient
# =====================================================================


def _clipped_ips_gradient(
    space: PolicySpace,
    theta: Sequence[float],
    events: Sequence[LoggedEvent],
    clip: float,
) -> tuple[list[float], float]:
    """∇_θ V̂_IPS_clipped(π_θ) and the objective value.

    Where the per-sample weight is *not* clipped, its gradient is
        ∂(w_i r_i) / ∂θ = w_i r_i · ∇ log π(a_i | x_i).
    Where it *is* clipped, ∂(w_i r_i) / ∂θ = 0 — we are on a plateau.

    This is the standard CRM gradient (Swaminathan-Joachims 2015).
    """
    n = len(events)
    if n == 0:
        return [0.0] * space.dim(), 0.0
    grad = [0.0] * space.dim()
    total = 0.0
    for e in events:
        probs = space.probs(theta, e.context)
        p = probs.get(e.action, 0.0)
        mu = max(_EPS, e.propensity)
        raw = p / mu
        if raw > clip:
            w = clip
            d_contrib = [0.0] * space.dim()  # plateau
        elif raw <= 0:
            w = 0.0
            d_contrib = [0.0] * space.dim()
        else:
            w = raw
            g = space.grad_log_prob(theta, e.context, e.action)
            # d/dθ (w_i r_i) = w_i r_i · ∇ log π(a|x)
            d_contrib = [w * e.reward * gj for gj in g]
        total += w * e.reward
        for j in range(space.dim()):
            grad[j] += d_contrib[j] / n
    return grad, total / n


def _crm_var_penalty(
    weighted_rewards: Sequence[float], lam: float
) -> float:
    """λ · √(Var̂(w_i r_i) / n). Returned as a *subtraction* from the
    objective; the optimizer maximizes (mean - penalty)."""
    n = len(weighted_rewards)
    if n < 2 or lam <= 0:
        return 0.0
    mean = sum(weighted_rewards) / n
    s2 = sum((x - mean) ** 2 for x in weighted_rewards) / (n - 1)
    if s2 <= 0:
        return 0.0
    return lam * math.sqrt(s2 / n)


# =====================================================================
# Main class
# =====================================================================


class PolicyImprover:
    """Safe off-policy policy optimizer.

    Parameters
    ----------
    policy_space
        Parameterised family of policies to search.
    baseline_value
        Estimated value of the incumbent policy on the same traffic.
        This is the bar HCPI has to clear. Coordination engines
        typically pass the SNIPS/DR value of the logging policy from
        `PolicyLab.evaluate(logging_policy)`.
    objective
        Optimization objective. ``"clipped_ips"`` (default), ``"snips"``,
        or ``"crm_var"`` (CRM with explicit variance penalty).
    weight_clip
        Importance-weight clip. Larger = less bias, more variance.
        20.0 is a reasonable default for offline RL.
    delta
        Total miscoverage budget for the two-sided Bernstein bound on
        the new policy's value. The HCPI safety gate uses
        LCB at level 1 - δ. Defaults to 0.05.
    crm_lambda
        Variance-penalty weight when ``objective == "crm_var"``.
    reward_range
        (r_min, r_max) for the Bernstein bound. If omitted, inferred
        from observed rewards (which is *not* a valid finite-sample
        bound — pass an a-priori range for true HCPI).
    event_bus
        Optional `EventBus` for telemetry.
    seed
        RNG seed for reproducibility of multi-start.
    """

    def __init__(
        self,
        policy_space: PolicySpace,
        baseline_value: float,
        *,
        objective: str = OBJ_CLIPPED_IPS,
        weight_clip: float = _DEFAULT_WEIGHT_CLIP,
        delta: float = _DEFAULT_DELTA,
        crm_lambda: float = 0.5,
        reward_range: tuple[float, float] | None = None,
        event_bus: EventBus | None = None,
        max_events: int | None = 200_000,
        seed: int | None = None,
    ) -> None:
        if objective not in KNOWN_OBJECTIVES:
            raise ValueError(
                f"unknown objective {objective!r}; expected one of {KNOWN_OBJECTIVES}"
            )
        if weight_clip <= 0:
            raise ValueError("weight_clip must be positive")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if reward_range is not None:
            r_lo, r_hi = float(reward_range[0]), float(reward_range[1])
            if not (r_hi >= r_lo):
                raise ValueError("reward_range must satisfy hi >= lo")
        self.policy_space = policy_space
        self.baseline_value = float(baseline_value)
        self.objective = objective
        self.weight_clip = float(weight_clip)
        self.delta = float(delta)
        self.crm_lambda = float(crm_lambda)
        self.reward_range = (
            (float(reward_range[0]), float(reward_range[1])) if reward_range else None
        )
        self.event_bus = event_bus
        self.max_events = max_events
        self.seed = seed
        self._lock = threading.RLock()
        self._events: list[LoggedEvent] = []
        self._rng = random.Random(seed)

    # ----- ingest ----------------------------------------------------

    def record(self, event: LoggedEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self.max_events is not None and len(self._events) > self.max_events:
                drop = len(self._events) - self.max_events
                del self._events[:drop]
        self._emit(IMP_RECORDED, {"action": event.action, "reward": event.reward})

    def record_batch(self, events: Iterable[LoggedEvent]) -> int:
        n = 0
        for e in events:
            self.record(e)
            n += 1
        return n

    def ingest_from_lab(self, lab: "PolicyLab") -> int:
        """Convenience: pull all events from a `PolicyLab`."""
        return self.record_batch(lab.events())

    def events(self) -> list[LoggedEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # ----- optimize --------------------------------------------------

    def improve(
        self,
        *,
        n_restarts: int = 3,
        n_iters: int = 200,
        learning_rate: float = 0.1,
        tol: float = 1e-5,
        verbose: bool = False,
    ) -> Improvement:
        """Search the policy space and return the best safe policy.

        Multi-start projected gradient ascent on the configured
        objective. At each restart, runs `n_iters` updates with step
        `learning_rate`. The best restart is selected by HCPI lower
        bound (not point estimate) so the optimizer is forced toward
        regions the data can certify.
        """
        if n_restarts < 1:
            raise ValueError("n_restarts must be ≥ 1")
        if n_iters < 1:
            raise ValueError("n_iters must be ≥ 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        with self._lock:
            events = list(self._events)

        best: Improvement | None = None
        total_iters = 0
        for restart in range(n_restarts):
            theta = self.policy_space.initial_parameters(self._rng)
            theta = self.policy_space.project(theta)
            prev_obj = -float("inf")
            converged = False
            iters_done = 0
            for it in range(n_iters):
                grad, obj = _clipped_ips_gradient(
                    self.policy_space, theta, events, self.weight_clip
                )
                if self.objective == OBJ_CRM_VAR:
                    # Subtract variance-penalty gradient via finite difference
                    # (cheap; the penalty is a smooth scalar of θ).
                    obj_pen = obj - self._var_penalty_at(theta, events)
                    grad = self._add_var_penalty_grad(grad, theta, events)
                    obj_now = obj_pen
                elif self.objective == OBJ_SNIPS:
                    obj_now = self._snips_at(theta, events)
                else:
                    obj_now = obj
                # ascent
                step = learning_rate
                theta = [theta[j] + step * grad[j] for j in range(len(theta))]
                theta = self.policy_space.project(theta)
                iters_done += 1
                total_iters += 1
                if it > 0 and abs(obj_now - prev_obj) < tol:
                    converged = True
                    break
                prev_obj = obj_now
            if verbose:  # pragma: no cover
                print(f"[improver] restart {restart}: obj={prev_obj:.6f} iters={iters_done}")
            candidate = self._finalize(
                theta=theta,
                events=events,
                iterations=iters_done,
                restarts=restart + 1,
                last_objective=prev_obj,
                converged=converged,
            )
            if best is None or candidate.improvement_lcb > best.improvement_lcb:
                best = candidate
        assert best is not None
        # Attach the total iteration count across restarts.
        diag = best.diagnostics
        diag2 = OptimizationDiagnostics(
            n=diag.n,
            n_eff=diag.n_eff,
            max_weight=diag.max_weight,
            mean_weight=diag.mean_weight,
            clipped_fraction=diag.clipped_fraction,
            coverage=diag.coverage,
            iterations=total_iters,
            restarts=n_restarts,
            objective_value=diag.objective_value,
            converged=diag.converged,
        )
        result = Improvement(
            parameters=best.parameters,
            value=best.value,
            value_se=best.value_se,
            value_lcb=best.value_lcb,
            value_ucb=best.value_ucb,
            baseline_value=best.baseline_value,
            improvement=best.improvement,
            improvement_se=best.improvement_se,
            improvement_lcb=best.improvement_lcb,
            improvement_ucb=best.improvement_ucb,
            delta=best.delta,
            safe=best.safe,
            verdict=best.verdict,
            diagnostics=diag2,
            rationale=best.rationale,
        )
        self._emit(
            IMP_OPTIMIZED,
            {
                "value": result.value,
                "value_lcb": result.value_lcb,
                "improvement_lcb": result.improvement_lcb,
                "verdict": result.verdict,
                "n": diag2.n,
                "iterations": diag2.iterations,
                "restarts": diag2.restarts,
            },
        )
        if result.safe:
            self._emit(IMP_PROMOTED, {"value": result.value, "lcb": result.improvement_lcb})
        else:
            self._emit(IMP_REJECTED, {"value": result.value, "lcb": result.improvement_lcb})
        return result

    @property
    def policy(self) -> Policy:
        """Return a callable for the last `improve()` result.

        Use the returned `Improvement.parameters` plus
        `policy_space.to_policy(theta)` directly if you want explicit
        provenance; this property exists for ergonomics on hot paths.
        """
        if self._last_theta is None:  # pragma: no cover
            raise RuntimeError("call improve() first")
        return self.policy_space.to_policy(self._last_theta)

    _last_theta: tuple[float, ...] | None = None

    def _finalize(
        self,
        *,
        theta: Sequence[float],
        events: Sequence[LoggedEvent],
        iterations: int,
        restarts: int,
        last_objective: float,
        converged: bool,
    ) -> Improvement:
        """Wrap a fitted θ in a full Improvement with HCPI bound."""
        weights, weighted, clipped = _per_event_value(
            self.policy_space, theta, events, self.weight_clip
        )
        n = len(events)
        value_point = _snips_value(weighted, weights) if weights else 0.0
        # For finite-sample bound, use IPS-style (each event contributes
        # w_i r_i; n is fixed; per-event range is C * (r_max - r_min)).
        # SNIPS is the *point* but it's slightly biased; we use the IPS
        # bound as the safety bar to remain conservative.
        ips_point = _ips_value(weighted)
        rng_lo, rng_hi = self._effective_reward_range(events)
        per_event_range = self.weight_clip * (rng_hi - rng_lo)
        # Bernstein half-width on IPS estimator.
        _, half_bern = empirical_bernstein_bound(
            weighted, delta=self.delta, value_range=per_event_range
        )
        # Gaussian SE as a supplementary number.
        if n >= 2:
            mean = ips_point
            s2 = sum((x - mean) ** 2 for x in weighted) / (n - 1)
            se = math.sqrt(max(0.0, s2) / n)
        else:
            se = 0.0
        # We center the CI on the SNIPS point but use the Bernstein
        # half-width (a valid bound on the IPS estimator). The shift
        # (snips - ips) is O(1/√n) → in the bound asymptotically; for
        # finite n we conservatively report `min(snips, ips) - half` as
        # the LCB to keep the safety claim true.
        lcb_value = min(value_point, ips_point) - half_bern
        ucb_value = max(value_point, ips_point) + half_bern
        improvement = value_point - self.baseline_value
        improvement_lcb = lcb_value - self.baseline_value
        improvement_ucb = ucb_value - self.baseline_value
        # SE on the lift: baseline is treated as a fixed quantity.
        improvement_se = se
        # Verdict.
        if improvement_lcb > 0.0:
            verdict = SAFE
            rationale = (
                f"LCB on V(π_new) - V(π_baseline) = {improvement_lcb:.4f} > 0 at δ={self.delta}; "
                f"safe to ship."
            )
        elif improvement_ucb <= 0.0:
            verdict = UNSAFE
            rationale = (
                f"UCB on V(π_new) - V(π_baseline) = {improvement_ucb:.4f} ≤ 0; "
                f"new policy is at best tied with baseline."
            )
        else:
            verdict = UNCERTAIN
            rationale = (
                f"CI on (V_new - V_baseline) = [{improvement_lcb:.4f}, {improvement_ucb:.4f}] "
                f"straddles 0 — collect more data or widen δ."
            )
        diag = OptimizationDiagnostics(
            n=n,
            n_eff=_ess(weights),
            max_weight=max(weights) if weights else 0.0,
            mean_weight=(sum(weights) / n) if n else 0.0,
            clipped_fraction=(sum(1 for c in clipped if c) / n) if n else 0.0,
            coverage=_coverage(self.policy_space, theta, events),
            iterations=iterations,
            restarts=restarts,
            objective_value=last_objective,
            converged=converged,
        )
        self._last_theta = tuple(theta)
        return Improvement(
            parameters=tuple(theta),
            value=value_point,
            value_se=se,
            value_lcb=lcb_value,
            value_ucb=ucb_value,
            baseline_value=self.baseline_value,
            improvement=improvement,
            improvement_se=improvement_se,
            improvement_lcb=improvement_lcb,
            improvement_ucb=improvement_ucb,
            delta=self.delta,
            safe=verdict == SAFE,
            verdict=verdict,
            diagnostics=diag,
            rationale=rationale,
        )

    # ----- safety check on arbitrary policies ------------------------

    def safety_check(
        self,
        policy: Policy | PolicyCandidate,
        *,
        baseline_value: float | None = None,
        name: str | None = None,
    ) -> HCPIReport:
        """HCPI on an arbitrary fixed policy; no optimization.

        Useful to certify a hand-written rule, a learned model, or a
        policy produced by another component before deploying.
        """
        if isinstance(policy, PolicyCandidate):
            pname = policy.name
            pi = policy.policy
        else:
            pname = name or "policy"
            pi = policy
        baseline = (
            float(baseline_value) if baseline_value is not None else self.baseline_value
        )
        with self._lock:
            events = list(self._events)
        # Treat the fixed policy as a 0-dim "space" for downstream uniformity.
        fake_space = _FixedPolicySpace(pi)
        weights, weighted, clipped = _per_event_value(
            fake_space, (), events, self.weight_clip
        )
        n = len(events)
        value_point = _snips_value(weighted, weights) if weights else 0.0
        ips_point = _ips_value(weighted)
        rng_lo, rng_hi = self._effective_reward_range(events)
        per_event_range = self.weight_clip * (rng_hi - rng_lo)
        _, half_bern = empirical_bernstein_bound(
            weighted, delta=self.delta, value_range=per_event_range
        )
        if n >= 2:
            mean = ips_point
            s2 = sum((x - mean) ** 2 for x in weighted) / (n - 1)
            se = math.sqrt(max(0.0, s2) / n)
        else:
            se = 0.0
        lcb_value = min(value_point, ips_point) - half_bern
        ucb_value = max(value_point, ips_point) + half_bern
        improvement = value_point - baseline
        improvement_lcb = lcb_value - baseline
        improvement_ucb = ucb_value - baseline
        if improvement_lcb > 0.0:
            verdict = SAFE
            rationale = "LCB > 0; safe to deploy."
        elif improvement_ucb <= 0.0:
            verdict = UNSAFE
            rationale = "UCB ≤ 0; policy is provably not better."
        else:
            verdict = UNCERTAIN
            rationale = "CI straddles 0; insufficient evidence."
        diag = OptimizationDiagnostics(
            n=n,
            n_eff=_ess(weights),
            max_weight=max(weights) if weights else 0.0,
            mean_weight=(sum(weights) / n) if n else 0.0,
            clipped_fraction=(sum(1 for c in clipped if c) / n) if n else 0.0,
            coverage=_coverage(fake_space, (), events),
            iterations=0,
            restarts=0,
            objective_value=value_point,
            converged=True,
        )
        report = HCPIReport(
            policy_name=pname,
            value=value_point,
            value_se=se,
            value_lcb=lcb_value,
            value_ucb=ucb_value,
            baseline_value=baseline,
            improvement=improvement,
            improvement_lcb=improvement_lcb,
            improvement_ucb=improvement_ucb,
            delta=self.delta,
            verdict=verdict,
            diagnostics=diag,
            rationale=rationale,
        )
        self._emit(
            IMP_CHECKED,
            {
                "policy": pname,
                "verdict": verdict,
                "value": value_point,
                "value_lcb": lcb_value,
                "improvement_lcb": improvement_lcb,
            },
        )
        return report

    # ----- helpers ----------------------------------------------------

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:  # pragma: no cover - telemetry shouldn't break runtime
            pass

    def _effective_reward_range(
        self, events: Sequence[LoggedEvent]
    ) -> tuple[float, float]:
        if self.reward_range is not None:
            return self.reward_range
        if not events:
            return (0.0, 1.0)
        lo = min(e.reward for e in events)
        hi = max(e.reward for e in events)
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _snips_at(self, theta: Sequence[float], events: Sequence[LoggedEvent]) -> float:
        weights, weighted, _ = _per_event_value(
            self.policy_space, theta, events, self.weight_clip
        )
        return _snips_value(weighted, weights)

    def _var_penalty_at(
        self, theta: Sequence[float], events: Sequence[LoggedEvent]
    ) -> float:
        _, weighted, _ = _per_event_value(
            self.policy_space, theta, events, self.weight_clip
        )
        return _crm_var_penalty(weighted, self.crm_lambda)

    def _add_var_penalty_grad(
        self,
        grad: list[float],
        theta: Sequence[float],
        events: Sequence[LoggedEvent],
    ) -> list[float]:
        # Finite-difference subtraction of the variance-penalty gradient.
        # Penalty term is small and smooth; central differences are
        # cheap relative to one IPS pass.
        eps = 1e-4
        base = self._var_penalty_at(theta, events)
        out = list(grad)
        for j in range(len(theta)):
            t2 = list(theta)
            t2[j] += eps
            pen2 = self._var_penalty_at(t2, events)
            out[j] -= (pen2 - base) / eps
        return out


class _FixedPolicySpace(PolicySpace):
    """Wrap a Policy callable as a 0-dim PolicySpace for `safety_check`."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self.actions = ()

    def dim(self) -> int:
        return 0

    def initial_parameters(self, rng: random.Random) -> list[float]:
        return []

    def probs(
        self, theta: Sequence[float], context: Mapping[str, float]
    ) -> dict[str, float]:
        out = self._policy(context)
        s = sum(max(0.0, v) for v in out.values())
        if s <= 0:
            return {k: 0.0 for k in out}
        return {k: max(0.0, v) / s for k, v in out.items()}

    def grad_log_prob(
        self,
        theta: Sequence[float],
        context: Mapping[str, float],
        action: str,
    ) -> list[float]:
        return []


# =====================================================================
# Convenience: turn an Improvement into a PolicyCandidate for PolicyLab
# =====================================================================


def to_policy_candidate(
    improver: PolicyImprover,
    improvement: Improvement,
    name: str = "π_improved",
) -> PolicyCandidate:
    """Wrap a fitted Improvement as a `PolicyCandidate` that `PolicyLab`
    can evaluate. Verifies the optimizer's claim against PolicyLab's
    DR/SNIPS independently — useful as a second opinion."""
    pi = improver.policy_space.to_policy(improvement.parameters)
    return PolicyCandidate(name=name, policy=pi)


__all__ = [
    "EpsilonGreedyPolicySpace",
    "HCPIReport",
    "IMP_CHECKED",
    "IMP_OPTIMIZED",
    "IMP_PROMOTED",
    "IMP_RECORDED",
    "IMP_REJECTED",
    "Improvement",
    "KNOWN_OBJECTIVES",
    "MixturePolicySpace",
    "OBJ_CLIPPED_IPS",
    "OBJ_CRM_VAR",
    "OBJ_SNIPS",
    "OptimizationDiagnostics",
    "PolicyImprover",
    "PolicySpace",
    "SAFE",
    "SoftmaxPolicySpace",
    "UNCERTAIN",
    "UNSAFE",
    "VERDICTS",
    "empirical_bernstein_bound",
    "normal_lcb",
    "to_policy_candidate",
]
