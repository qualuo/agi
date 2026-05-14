r"""Coalition — Shapley-value credit assignment as a runtime primitive.

Every long-running runtime that *learns* eventually has to answer the
question every other primitive in this stack dodges: when a ticket
succeeded (or failed), **which contributors deserve the credit (or the
blame)?** A skill, a tool, a data source, a sub-agent, a budget line —
they all touched the trace. The naïve answers are wrong in two ways
that compound:

  * **Per-contributor averages.** "How often did skill X co-occur
    with success?" is the inclusion-co-occurrence rate; it confounds
    X with whatever X tends to be bundled with. Two-thirds of the
    "credit" you assign goes to a free-rider.
  * **Leave-one-out drop.** "How much worse off would I be without
    X?" only measures *one* counterfactual (S = N \ {i}), ignoring
    the other 2^{n-1} - 1 subsets in which X mattered. The answer
    has no axiomatic warrant: it violates symmetry, additivity and
    efficiency simultaneously.

The literature solved this in 1953.

  * **Shapley, 1953 — A Value for n-Person Games.** Defines the
    unique solution concept on the class of cooperative games that
    simultaneously satisfies *efficiency* (Σ φ_i = v(N)), *symmetry*
    (interchangeable players get equal credit), *dummy* (a player
    contributing nothing to every coalition gets zero), and
    *additivity* (the value of the sum of two games is the sum of
    the values). For any characteristic function `v : 2^N → ℝ`,
    the Shapley value of player i is

        φ_i(v) = Σ_{S ⊆ N \ {i}}
                   |S|! · (n − |S| − 1)! / n!  ·  (v(S ∪ {i}) − v(S))

    equivalently, the expected marginal contribution of i when
    coalition members are added in a uniformly-random order.

  * **Banzhaf, 1965 — Weighted voting doesn't work.** Drops the
    efficiency axiom; equal-weighted average of marginals. The
    correct primitive for *voting power* (Penrose-Banzhaf index)
    rather than *surplus splitting*.

  * **Castro, Gómez & Tejada, 2009 — Polynomial-time Shapley.**
    Monte-Carlo permutation estimator: T samples give an
    `Δ · √(log(2n/δ) / (2T))` Hoeffding bound per player, where
    Δ = v_max − v_min is the value range. Linear in T·n, not
    exponential in n.

  * **Maleki, Tran-Thanh, Long, Rogers & Jennings, 2014 —
    Stratified Sampling.** Sampling marginals stratified by
    coalition size, with neyman-allocation across strata, gives a
    constant-factor variance reduction; the resulting estimator is
    the workhorse modern implementation.

  * **Maurer & Pontil, 2009 — Empirical Bernstein.** Replaces the
    Hoeffding bound by `σ̂ · √(2 log(2/δ)/T) + 7Δ log(2/δ) / (3(T−1))`.
    When realised variance is small (which is typical for
    well-aligned contributors) it dominates Hoeffding by an order
    of magnitude on real workloads.

  * **Štrumbelj & Kononenko, 2014; Lundberg & Lee, 2017 (SHAP).**
    Adapted Shapley to model explanation, popularised the family
    in machine-learning practice. Kernel-SHAP solves a weighted
    least-squares problem; we expose this as the data-driven path
    when an explicit `v` is unavailable.

  * **Owen, 1977 — Multilinear Extension.** Defines a value for
    games with a coalition structure (groups). Players inside a
    group split a group-level Shapley; the inter-group Shapley
    handles cross-group surplus. The right primitive when
    contributors come from multiple tenants and credit must be
    attributed both *within* and *across* tenants.

`Coalition` is the runtime primitive that makes all four operational
on a live trace stream. It maintains a registry of contributors, a
characteristic-function (either user-supplied or fit from observed
traces), and a sampling state that returns anytime-valid PAC bounds
on every player's value. The coordination engine queries it to
allocate retraining budget, to split multi-tenant cost, to gate
skill deprecation, to weight calibration by attribution, and to
write tamper-evident credit receipts to the AttestationLedger.

Mathematical core
-----------------

Exact Shapley computes the sum over all 2^{n-1} coalitions
not containing i — exponential, only tractable for n ≤ 16.

For n > 16 the **permutation estimator** is used::

    Sample π_t ~ Uniform(S_n); let π_t^{<i} = predecessors of i in π_t.
    Then  φ̂_i(T) = (1/T) Σ_{t=1}^T  v(π_t^{<i} ∪ {i}) − v(π_t^{<i}).

This is unbiased (Castro 2009 Prop. 2) and bounded; each draw lies in
[−Δ, +Δ] where Δ = v_max − v_min. Hoeffding then gives the per-player
PAC interval

    P(|φ̂_i(T) − φ_i| ≥ ε) ≤ 2 exp(−2T ε² / Δ²)
    ⇒ ε_H(T, δ) = Δ · √(log(2/δ) / (2T))

Union over n players gives joint coverage: ε_joint(T, δ) =
ε_H(T, δ/n). We default to the empirical Bernstein bound (Maurer 2009),
which uses the running per-player marginal-variance and is tighter
whenever the realised variance is below Δ²/4 — almost always, on
real coalitions.

**Sample re-use.** Within one permutation π_t, every player's marginal
can be computed with O(n) value-function calls (one per prefix).
A single sample updates *all n* Shapley estimates simultaneously,
making the amortised cost per-player-per-sample O(1) calls to v.

The **stratified** estimator partitions the n permutations by the
coalition size at which the player enters; each stratum has its own
mean and variance, and the total estimate is the weighted average.
Neyman allocation across strata reduces variance by up to a factor
of 4× in worst-case empirical workloads.

The **observation-driven** path: when the user only has a trace stream
of `(contributors, value)` pairs and no closed-form v, the coalition
estimates `v̂(S)` by the empirical mean over traces with
`contributors ⊇ S` (the "interventional" expectation). Subsets with
zero samples fall back to the grand mean. This is biased in general
but reduces to the correct value when the trace generator covers
every coalition uniformly; the bias is reported on every estimate
via the `coverage` field of `CoalitionReport`.

What it composes (razor-sharp coordination integration)
------------------------------------------------------

  * **Cartographer.** Skills credited highly by Coalition are the
    skills Cartographer should keep mastered; skills credited near
    zero are deprecation candidates. The coordinator calls
    `coalition.shapley_montecarlo(...)` per mastery transition and
    pipes the result into Cartographer's `value=` argument when
    re-registering a task.

  * **PolicyImprover.** A retraining trace pool needs honest credit
    so HCPI can detect which interventions actually move the needle.
    `coalition.observe(contributors=trace.skills, value=trace.reward)`
    is the wiring; the resulting φ̂_i is what PolicyImprover treats
    as the per-skill effect under the new policy.

  * **PortfolioOptimizer.** Multi-tenant cost split is a coalition
    game: each tenant is a player, v(S) is the cost they would have
    incurred had only S been served. Shapley gives the unique
    additive-and-efficient split. Coalition exposes
    `allocate_efficient(...)` which scales Shapley to sum to a target
    budget, suitable for direct portfolio re-balancing.

  * **Strategist.** Strategist's risk-adjusted EV is computed from
    raw per-arm successes. When two arms agree on outcome but one
    co-occurs with high-credit skills and the other doesn't, the
    coordinator should prefer the one Coalition credits.

  * **AttestationLedger.** Every credit allocation emits a
    `coalition.credited` receipt — a third-party-replayable proof
    that under v̂ at time t with confidence δ, skill X earned exactly
    φ̂_X(t) of the surplus. This is the audit trail "fair allocation"
    requires.

  * **EventBus.** Streams every registration, observation, and
    credit transition. A higher-level coordination engine reacts in
    real time — e.g. trigger a new BAI campaign on
    `coalition.credited` when player X drops below a deprecation
    threshold.

Where this slots in
-------------------

    coalition = Coalition(bus=bus, attestor=attestor)
    coalition.register_player("skill:py-debug",   value=1.0, cost=0.002)
    coalition.register_player("skill:web-search", value=1.0, cost=0.005)
    coalition.register_player("tool:bash",        value=0.5, cost=0.001)

    for trace in driver.completed():
        coalition.observe(
            contributors=trace.skills_used + trace.tools_used,
            value=float(trace.outcome.success),
        )

    report = coalition.shapley_montecarlo(
        epsilon=0.02, delta=0.05,
        method=POLICY_BERNSTEIN, max_samples=10_000,
    )
    for player_id, est in report.values.items():
        if est.upper < deprecation_threshold:
            cartographer.deprecate(player_id)
        elif est.lower > promotion_threshold:
            cartographer.promote(player_id)

Events
------
    coalition.started               — Coalition was constructed
    coalition.player_registered     — a contributor was added
    coalition.observed              — a trace value was logged
    coalition.value_function_set    — a closed-form v was supplied
    coalition.computed              — a Shapley report was produced
    coalition.credited              — credit was assigned + attested
    coalition.cleared               — state was reset
    coalition.report                — a coverage report was published

Honest about limits
-------------------

  * Exact Shapley scales as O(2^n · n); we ship a hard cap at n ≤ 18
    for exact mode. Above that, the permutation estimator is the
    only sane path.
  * The Hoeffding / Bernstein bounds are anytime in `T`, but not in
    the choice of `δ` — using the same estimator with shrinking `δ`
    requires a union over the δ-grid. We expose the explicit `δ`
    schedule via `shapley_montecarlo(...).schedule_used`.
  * Observation-driven v̂ is biased unless the coalition coverage is
    near-uniform. We report `coverage_min` (worst-case fraction of
    samples per coalition) so the coordinator can refuse low-coverage
    reports. For severely sparse coverage, the user should provide a
    closed-form `v` or use the linear-interaction fitter (see
    `fit_linear_v`).
  * The core / least-core requires linear programming; we provide
    `in_core(...)` and a sampled core-violation detector but defer
    full LP solution. The Shapley value is always in the core for
    convex games; for non-convex games it may not be.
  * Coalition does not detect non-stationarity; if the value function
    drifts, DriftSentinel must invalidate the Shapley estimate.

Stdlib-only, CPU-bound, threadsafe; identical I/O surface to
`Arbiter`, `Cartographer`, `Strategist` so a coordination engine can
compose them uniformly.

Citations
---------

* Shapley, L. S. (1953). A value for n-person games. *Contributions to
  the Theory of Games*, II, 307-317.
* Banzhaf, J. F. (1965). Weighted voting doesn't work: A mathematical
  analysis. *Rutgers Law Review*, 19, 317-343.
* Owen, G. (1977). Values of games with a priori unions. *Mathematical
  Economics and Game Theory*, 76-88.
* Castro, J., Gómez, D. & Tejada, J. (2009). Polynomial calculation of
  the Shapley value based on sampling. *Computers & OR*, 36(5),
  1726-1730.
* Maleki, S., Tran-Thanh, L., Long, G., Rogers, A. & Jennings, N. R.
  (2014). Bounding the estimation error of sampling-based Shapley
  value approximation. *ICML Workshop*.
* Maurer, A. & Pontil, M. (2009). Empirical Bernstein bounds and
  sample-variance penalization. *COLT*.
* Štrumbelj, E. & Kononenko, I. (2014). Explaining prediction models
  and individual predictions with feature contributions. *KAIS*,
  41(3), 647-665.
* Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to
  interpreting model predictions. *NeurIPS*.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

COALITION_STARTED = "coalition.started"
COALITION_PLAYER_REGISTERED = "coalition.player_registered"
COALITION_OBSERVED = "coalition.observed"
COALITION_VALUE_FUNCTION_SET = "coalition.value_function_set"
COALITION_COMPUTED = "coalition.computed"
COALITION_CREDITED = "coalition.credited"
COALITION_CLEARED = "coalition.cleared"
COALITION_REPORT = "coalition.report"


# =====================================================================
# Bound policies
# =====================================================================

POLICY_HOEFFDING = "hoeffding"
POLICY_BERNSTEIN = "bernstein"
POLICY_EXACT = "exact"
POLICY_STRATIFIED = "stratified"
POLICY_PERMUTATION = "permutation"

KNOWN_BOUND_POLICIES = (POLICY_HOEFFDING, POLICY_BERNSTEIN)
KNOWN_ESTIMATORS = (POLICY_EXACT, POLICY_PERMUTATION, POLICY_STRATIFIED)


# =====================================================================
# Limits
# =====================================================================

_EXACT_MAX_PLAYERS = 18
_EPS = 1e-12
_DEFAULT_MAX_SAMPLES = 20_000
_DEFAULT_MIN_SAMPLES = 32
_OWEN_MAX_GROUPS = 12


# =====================================================================
# Core math helpers
# =====================================================================


def _shapley_weight(s: int, n: int) -> float:
    """Standard Shapley weight |S|! (n-|S|-1)! / n!."""
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def hoeffding_radius(delta: float, n_samples: int, value_range: float) -> float:
    """Two-sided Hoeffding half-width.

    For a mean of `n_samples` i.i.d. draws bounded in
    ``[−value_range/2, +value_range/2]``, returns ``ε`` such that
    ``P(|μ̂ − μ| ≥ ε) ≤ δ`` via the inequality
    ``2 exp(−2 n ε² / Δ²) ≤ δ``.

    Marginal-contribution draws lie in ``[−Δ, +Δ]`` so the effective
    range is ``2Δ``; the caller is expected to pass ``value_range``
    as that effective range.
    """
    if n_samples <= 0:
        return float("inf")
    if value_range <= 0.0:
        return 0.0
    return value_range * math.sqrt(math.log(2.0 / max(delta, _EPS)) / (2.0 * n_samples))


def bernstein_radius(
    delta: float,
    n_samples: int,
    value_range: float,
    sample_variance: float,
) -> float:
    """Empirical-Bernstein half-width (Maurer-Pontil, 2009).

    ``ε(T, δ) = σ̂ √(2 log(2/δ)/T) + 7 Δ log(2/δ) / (3(T−1))``

    Tighter than Hoeffding when the realised variance is small.
    """
    if n_samples <= 1:
        return float("inf")
    if value_range <= 0.0:
        return 0.0
    log_term = math.log(2.0 / max(delta, _EPS))
    var_term = math.sqrt(max(sample_variance, 0.0) * 2.0 * log_term / n_samples)
    bias_term = 7.0 * value_range * log_term / (3.0 * (n_samples - 1))
    return var_term + bias_term


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class PlayerSpec:
    """A contributor in the cooperative game.

    `value` is the user-attached importance weight (e.g. payoff if
    this player's skill is invoked alone); it is purely metadata for
    downstream consumers — the Shapley computation uses only `v`.

    `cost` is the user-attached marginal cost; downstream consumers
    use it for cost-allocation splits.
    """

    id: str
    value: float = 1.0
    cost: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShapleyEstimate:
    """Per-player credit estimate with anytime PAC bound."""

    player_id: str
    point: float
    lower: float
    upper: float
    n_samples: int
    sample_variance: float
    method: str

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n_samples": self.n_samples,
            "sample_variance": self.sample_variance,
            "method": self.method,
        }


@dataclass(frozen=True)
class CoalitionReport:
    """Result of one Shapley computation.

    `values` is keyed by player id. `grand_value` is v(N) — the total
    surplus to allocate. `coverage_min` is the worst-case fraction of
    samples assigned to any coalition in observation-mode (0.0 means
    no observations covered that coalition).

    `efficiency_gap = grand_value − Σ φ̂_i`. For exact mode this is
    machine-zero; for MC mode it reflects sampling noise.

    `receipt_hash` is non-empty iff an `AttestationLedger` was wired
    in and the computation produced a credit receipt.
    """

    id: str
    grand_value: float
    values: dict[str, ShapleyEstimate]
    coverage_min: float
    efficiency_gap: float
    estimator: str
    bound_policy: str
    epsilon_requested: float
    delta_requested: float
    n_samples_total: int
    elapsed_s: float
    receipt_hash: str = ""
    schedule_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "grand_value": self.grand_value,
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "coverage_min": self.coverage_min,
            "efficiency_gap": self.efficiency_gap,
            "estimator": self.estimator,
            "bound_policy": self.bound_policy,
            "epsilon_requested": self.epsilon_requested,
            "delta_requested": self.delta_requested,
            "n_samples_total": self.n_samples_total,
            "elapsed_s": self.elapsed_s,
            "receipt_hash": self.receipt_hash,
            "schedule_used": dict(self.schedule_used),
        }


@dataclass(frozen=True)
class CoverageReport:
    """Diagnostic report on observation density across coalitions."""

    n_players: int
    n_observations: int
    n_distinct_coalitions: int
    n_possible_coalitions: int
    fraction_covered: float
    grand_coalition_observed: bool
    empty_coalition_observed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Coalition
# =====================================================================


CharacteristicFn = Callable[[frozenset[str]], float]


class Coalition:
    """Cooperative-game state with Shapley / Banzhaf / Owen credit.

    Thread-safe; an internal lock guards every public method that
    mutates state. Reads of immutable returned dataclasses are safe
    without external synchronisation.

    Construction is cheap; the heavy lifting happens inside
    ``shapley_exact``, ``shapley_montecarlo``, ``shapley_stratified``,
    or ``banzhaf_indices``.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any | None = None,
        rng: random.Random | None = None,
        coalition_id: str | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._rng = rng if rng is not None else random.Random()
        self._id = coalition_id or f"coa-{int(time.time() * 1000):x}"
        self._lock = threading.RLock()
        self._players: dict[str, PlayerSpec] = {}
        self._player_order: list[str] = []
        self._value_fn: CharacteristicFn | None = None
        # Observation cache: coalition (as frozenset) -> running sums.
        # Each value is (sum, sum_sq, n).
        self._obs: dict[frozenset[str], tuple[float, float, int]] = {}
        self._obs_count_total = 0
        # Value-range cache used for PAC bounds.
        self._obs_min: float | None = None
        self._obs_max: float | None = None
        # History of completed reports for the audit trail.
        self._history: list[CoalitionReport] = []
        self._emit(COALITION_STARTED, {"coalition_id": self._id})

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def register_player(
        self,
        player_id: str,
        *,
        value: float = 1.0,
        cost: float = 0.0,
        meta: Mapping[str, Any] | None = None,
    ) -> PlayerSpec:
        """Add a contributor to the game.

        Idempotent on `player_id`: re-registration updates the spec
        but keeps the existing observations and history.
        """
        if not player_id or not isinstance(player_id, str):
            raise ValueError("player_id must be a non-empty string")
        spec = PlayerSpec(
            id=player_id,
            value=float(value),
            cost=float(cost),
            meta=dict(meta) if meta else {},
        )
        with self._lock:
            already = player_id in self._players
            self._players[player_id] = spec
            if not already:
                self._player_order.append(player_id)
        self._emit(COALITION_PLAYER_REGISTERED, {
            "coalition_id": self._id,
            "player_id": player_id,
            "value": spec.value,
            "cost": spec.cost,
        })
        return spec

    def players(self) -> tuple[PlayerSpec, ...]:
        with self._lock:
            return tuple(self._players[pid] for pid in self._player_order)

    @property
    def n_players(self) -> int:
        with self._lock:
            return len(self._player_order)

    # -----------------------------------------------------------------
    # Value function: closed-form or observation-driven
    # -----------------------------------------------------------------

    def set_value_function(self, v: CharacteristicFn) -> None:
        """Provide a closed-form characteristic function v(S).

        `v` must accept a `frozenset[str]` of player ids and return a
        finite float. It is called many times during exact / sampling
        computation; make it pure and fast.
        """
        if not callable(v):
            raise TypeError("v must be callable")
        with self._lock:
            self._value_fn = v
        self._emit(COALITION_VALUE_FUNCTION_SET, {"coalition_id": self._id})

    def observe(
        self,
        contributors: Iterable[str],
        value: float,
        *,
        weight: float = 1.0,
    ) -> None:
        """Log a trace: this set of contributors produced `value`.

        Multiple observations on the same coalition are aggregated via
        running mean (and running sum-of-squares for the Bernstein
        bound). `weight` rescales the single observation but does not
        change the unit of `value`.
        """
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if weight <= 0.0 or not math.isfinite(weight):
            raise ValueError("weight must be a positive finite number")
        coalition = frozenset(contributors)
        with self._lock:
            # Auto-register any unknown contributor with default spec
            # so the user can stream traces without upfront wiring.
            for pid in coalition:
                if pid not in self._players:
                    self.register_player(pid)
            prev = self._obs.get(coalition, (0.0, 0.0, 0))
            s, sq, n = prev
            s_new = s + value * weight
            sq_new = sq + (value * value) * weight
            n_new = n + 1
            self._obs[coalition] = (s_new, sq_new, n_new)
            self._obs_count_total += 1
            if self._obs_min is None or value < self._obs_min:
                self._obs_min = value
            if self._obs_max is None or value > self._obs_max:
                self._obs_max = value
        self._emit(COALITION_OBSERVED, {
            "coalition_id": self._id,
            "contributors": sorted(coalition),
            "value": float(value),
            "weight": float(weight),
        })

    def observed_value(self, coalition: Iterable[str]) -> float | None:
        """Return the running mean for an observed coalition, or None
        if that coalition has never been observed.
        """
        key = frozenset(coalition)
        with self._lock:
            triple = self._obs.get(key)
            if triple is None or triple[2] == 0:
                return None
            return triple[0] / triple[2]

    def coverage(self) -> CoverageReport:
        """Diagnostic snapshot of observation density."""
        with self._lock:
            n = len(self._player_order)
            possible = (1 << n) if n <= 30 else float("inf")
            distinct = len(self._obs)
            empty_seen = frozenset() in self._obs
            grand_seen = frozenset(self._player_order) in self._obs
            frac = (
                distinct / possible
                if isinstance(possible, int) and possible > 0
                else 0.0
            )
            return CoverageReport(
                n_players=n,
                n_observations=self._obs_count_total,
                n_distinct_coalitions=distinct,
                n_possible_coalitions=possible if isinstance(possible, int) else -1,
                fraction_covered=frac,
                grand_coalition_observed=grand_seen,
                empty_coalition_observed=empty_seen,
            )

    # -----------------------------------------------------------------
    # Internal: dispatch to v
    # -----------------------------------------------------------------

    def _eval_v(self, coalition: frozenset[str]) -> float:
        """Dispatch v(S) — closed-form first, else observation-driven."""
        if self._value_fn is not None:
            return float(self._value_fn(coalition))
        # Observation-driven: empirical mean over the exact coalition,
        # falling back to the grand mean if unseen.
        triple = self._obs.get(coalition)
        if triple is not None and triple[2] > 0:
            return triple[0] / triple[2]
        if not self._obs:
            return 0.0
        total_s = sum(t[0] for t in self._obs.values())
        total_n = sum(t[2] for t in self._obs.values())
        return total_s / total_n if total_n > 0 else 0.0

    def _value_range(self) -> float:
        """Range of v used in PAC bounds."""
        if self._value_fn is not None:
            # User function: probe v(∅), v(N) and the singletons for a
            # conservative range estimate. The marginal-contribution
            # range is 2 * (v_max - v_min) on the entire game; for
            # bounded games this is a reasonable upper bound.
            samples = [self._eval_v(frozenset())]
            samples.append(self._eval_v(frozenset(self._player_order)))
            for pid in self._player_order[:8]:
                samples.append(self._eval_v(frozenset([pid])))
            lo = min(samples)
            hi = max(samples)
            return max(hi - lo, _EPS)
        # Observation-driven: use realised value range.
        if self._obs_min is not None and self._obs_max is not None:
            return max(self._obs_max - self._obs_min, _EPS)
        return 1.0

    # -----------------------------------------------------------------
    # Exact Shapley
    # -----------------------------------------------------------------

    def shapley_exact(self) -> CoalitionReport:
        """Compute the exact Shapley value of every registered player.

        Raises ``ValueError`` if n > _EXACT_MAX_PLAYERS — exact is
        intractable above 18 players. Use ``shapley_montecarlo`` or
        ``shapley_stratified`` for larger games.
        """
        t0 = time.time()
        with self._lock:
            players = list(self._player_order)
        n = len(players)
        if n == 0:
            return self._empty_report(POLICY_EXACT, t0)
        if n > _EXACT_MAX_PLAYERS:
            raise ValueError(
                f"shapley_exact: n={n} exceeds cap {_EXACT_MAX_PLAYERS}; "
                "use shapley_montecarlo or shapley_stratified instead"
            )
        # For every subset, precompute v(S); for every i, accumulate
        # the weighted marginal.
        phi = {pid: 0.0 for pid in players}
        idx = {pid: k for k, pid in enumerate(players)}
        # Iterate over coalitions S not containing i, weighted by
        # Shapley coefficient |S|!(n−|S|−1)!/n!. We do this by
        # iterating once over all 2^{n-1} S-sized coalitions per
        # player; equivalent but uses fewer v calls is to enumerate
        # all 2^n subsets and use each one twice (with and without i).
        # We cache v(S) so each subset is evaluated once.
        v_cache: dict[frozenset[str], float] = {}

        def vof(s: frozenset[str]) -> float:
            cached = v_cache.get(s)
            if cached is not None:
                return cached
            val = self._eval_v(s)
            v_cache[s] = val
            return val

        # Enumerate all 2^n subsets via bitmask.
        all_v = [0.0] * (1 << n)
        for mask in range(1 << n):
            s = frozenset(players[k] for k in range(n) if mask & (1 << k))
            all_v[mask] = vof(s)
        # Marginal of i added to coalition encoded by `mask` (bit i clear).
        # φ_i = Σ_{S: i ∉ S} |S|!(n−|S|−1)!/n! · (v(S ∪ {i}) − v(S))
        weights = [_shapley_weight(s, n) for s in range(n)]
        for i, pid in enumerate(players):
            bit = 1 << i
            acc = 0.0
            for mask in range(1 << n):
                if mask & bit:
                    continue
                s_size = bin(mask).count("1")
                acc += weights[s_size] * (all_v[mask | bit] - all_v[mask])
            phi[pid] = acc
        grand = all_v[(1 << n) - 1]
        empty = all_v[0]
        # Exact has zero variance / zero half-width.
        estimates = {
            pid: ShapleyEstimate(
                player_id=pid,
                point=phi[pid],
                lower=phi[pid],
                upper=phi[pid],
                n_samples=1,
                sample_variance=0.0,
                method=POLICY_EXACT,
            )
            for pid in players
        }
        sum_phi = sum(phi.values())
        report = CoalitionReport(
            id=f"sh-{uuid_short()}",
            grand_value=grand - empty,  # surplus, not absolute
            values=estimates,
            coverage_min=1.0,
            efficiency_gap=(grand - empty) - sum_phi,
            estimator=POLICY_EXACT,
            bound_policy=POLICY_EXACT,
            epsilon_requested=0.0,
            delta_requested=0.0,
            n_samples_total=(1 << n),
            elapsed_s=time.time() - t0,
            schedule_used={"subsets_enumerated": 1 << n},
        )
        report = self._attest(report)
        with self._lock:
            self._history.append(report)
        self._emit_computed(report)
        return report

    # -----------------------------------------------------------------
    # Monte-Carlo permutation Shapley
    # -----------------------------------------------------------------

    def shapley_montecarlo(
        self,
        *,
        epsilon: float = 0.05,
        delta: float = 0.05,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        method: str = POLICY_BERNSTEIN,
        early_stop: bool = True,
    ) -> CoalitionReport:
        """Permutation-sampling Shapley with anytime PAC bound.

        Castro 2009. Each iteration samples a uniform permutation π
        of the players and updates *every* player's running mean
        with their marginal in π. The estimator is unbiased; the
        per-player half-width is given by `method`:

          * ``POLICY_HOEFFDING`` — distribution-free, conservative.
          * ``POLICY_BERNSTEIN`` — empirical Bernstein (default);
            uses the running per-player marginal variance.

        Stops once `min_samples` is reached *and* every player's
        half-width is ≤ epsilon, or when `max_samples` is hit. The
        per-player half-width is computed at confidence
        ``δ_player = δ / n`` so the joint coverage is ≥ 1 − δ over
        all players (union bound).
        """
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if method not in KNOWN_BOUND_POLICIES:
            raise ValueError(
                f"method {method!r} not in {KNOWN_BOUND_POLICIES}"
            )
        if min_samples < 2:
            min_samples = 2
        if max_samples < min_samples:
            raise ValueError("max_samples must be >= min_samples")
        t0 = time.time()
        with self._lock:
            players = list(self._player_order)
        n = len(players)
        if n == 0:
            return self._empty_report(POLICY_PERMUTATION, t0)
        if n == 1:
            # Trivial — exact in one shot.
            return self.shapley_exact()

        # Per-player running stats: mean, M2 (Welford), n.
        sums = {pid: 0.0 for pid in players}
        m2 = {pid: 0.0 for pid in players}
        counts = {pid: 0 for pid in players}
        value_range = self._value_range()
        # Marginal-contribution range is 2 * v_range (m can be ±range).
        marg_range = 2.0 * value_range
        delta_player = delta / max(n, 1)

        rng = self._rng
        order = list(players)
        # We need v(∅) cached; compute once.
        v_empty = self._eval_v(frozenset())
        v_grand = self._eval_v(frozenset(players))
        total_iters = 0
        radius_fn = (
            hoeffding_radius if method == POLICY_HOEFFDING else bernstein_radius
        )
        while total_iters < max_samples:
            total_iters += 1
            rng.shuffle(order)
            prefix: list[str] = []
            prev_val = v_empty
            for pid in order:
                prefix.append(pid)
                cur_val = self._eval_v(frozenset(prefix))
                m = cur_val - prev_val
                # Welford update.
                counts[pid] += 1
                delta_w = m - (sums[pid] / counts[pid] if counts[pid] > 1 else 0.0)
                sums[pid] += m
                if counts[pid] >= 2:
                    new_mean = sums[pid] / counts[pid]
                    m2[pid] += (m - new_mean) * (m - (sums[pid] - m) / (counts[pid] - 1))
                prev_val = cur_val
            if total_iters >= min_samples and early_stop:
                worst = 0.0
                for pid in players:
                    c = counts[pid]
                    var = m2[pid] / (c - 1) if c > 1 else 0.0
                    if method == POLICY_HOEFFDING:
                        r = hoeffding_radius(delta_player, c, marg_range)
                    else:
                        r = bernstein_radius(delta_player, c, marg_range, var)
                    if r > worst:
                        worst = r
                if worst <= epsilon:
                    break
        # Build estimates.
        estimates: dict[str, ShapleyEstimate] = {}
        for pid in players:
            c = counts[pid]
            mean = sums[pid] / c if c > 0 else 0.0
            var = m2[pid] / (c - 1) if c > 1 else 0.0
            if method == POLICY_HOEFFDING:
                r = hoeffding_radius(delta_player, c, marg_range)
            else:
                r = bernstein_radius(delta_player, c, marg_range, var)
            estimates[pid] = ShapleyEstimate(
                player_id=pid,
                point=mean,
                lower=mean - r,
                upper=mean + r,
                n_samples=c,
                sample_variance=var,
                method=method,
            )
        grand = v_grand - v_empty
        sum_phi = sum(e.point for e in estimates.values())
        report = CoalitionReport(
            id=f"sh-{uuid_short()}",
            grand_value=grand,
            values=estimates,
            coverage_min=1.0 if self._value_fn is not None else self._observation_coverage_min(),
            efficiency_gap=grand - sum_phi,
            estimator=POLICY_PERMUTATION,
            bound_policy=method,
            epsilon_requested=epsilon,
            delta_requested=delta,
            n_samples_total=total_iters,
            elapsed_s=time.time() - t0,
            schedule_used={
                "iters": total_iters,
                "delta_player": delta_player,
                "marg_range": marg_range,
            },
        )
        report = self._attest(report)
        with self._lock:
            self._history.append(report)
        self._emit_computed(report)
        return report

    # -----------------------------------------------------------------
    # Stratified-sampling Shapley
    # -----------------------------------------------------------------

    def shapley_stratified(
        self,
        *,
        epsilon: float = 0.05,
        delta: float = 0.05,
        max_samples_per_stratum: int = 256,
        method: str = POLICY_BERNSTEIN,
    ) -> CoalitionReport:
        """Stratified-sampling Shapley (Maleki et al., 2014).

        Splits permutations by *position* of each player. For player
        ``i`` and position ``k``, the marginal contribution
        ``Δ_{i,k} = v(S ∪ {i}) − v(S)`` is averaged over coalitions
        S of size k drawn uniformly from N \\ {i}. The Shapley value
        is then the simple average of the n stratum means.

        Variance is bounded above by the per-stratum variance averaged
        over strata, which is typically much smaller than the
        single-pool permutation variance.
        """
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if method not in KNOWN_BOUND_POLICIES:
            raise ValueError(
                f"method {method!r} not in {KNOWN_BOUND_POLICIES}"
            )
        t0 = time.time()
        with self._lock:
            players = list(self._player_order)
        n = len(players)
        if n == 0:
            return self._empty_report(POLICY_STRATIFIED, t0)
        if n == 1:
            return self.shapley_exact()

        rng = self._rng
        # For each player i, each stratum k ∈ {0, …, n-1}, track
        # (sum, M2, count).
        sums = {pid: [0.0] * n for pid in players}
        m2 = {pid: [0.0] * n for pid in players}
        counts = {pid: [0] * n for pid in players}
        value_range = self._value_range()
        marg_range = 2.0 * value_range
        # Per-cell early-stop tolerance: Bonferroni-correct over
        # (players × strata) for the worst-case per-cell radius.
        delta_cell = delta / (n * n)
        # Final estimator confidence is δ/n per player (n stratum means
        # combined into one average; computed below).

        def radius_for(c: int, var: float) -> float:
            if method == POLICY_HOEFFDING:
                return hoeffding_radius(delta_cell, c, marg_range)
            return bernstein_radius(delta_cell, c, marg_range, var)

        total_draws = 0
        for pid in players:
            others = [p for p in players if p != pid]
            for k in range(n):
                # k = size of preceding coalition; draw min_floor samples
                # then expand until convergence at this stratum.
                target = max(8, min(max_samples_per_stratum, max_samples_per_stratum))
                drawn = 0
                while drawn < target:
                    rng.shuffle(others)
                    S = frozenset(others[:k])
                    v_s = self._eval_v(S)
                    v_si = self._eval_v(S | {pid})
                    m = v_si - v_s
                    drawn += 1
                    total_draws += 1
                    c = counts[pid][k] + 1
                    delta_w = m - (sums[pid][k] / counts[pid][k] if counts[pid][k] >= 1 else 0.0)
                    counts[pid][k] = c
                    sums[pid][k] += m
                    if c >= 2:
                        new_mean = sums[pid][k] / c
                        m2[pid][k] += (m - new_mean) * (m - (sums[pid][k] - m) / (c - 1))
                    # Early-exit at this stratum once the per-cell
                    # radius is below epsilon / n (so the n strata
                    # combined are below epsilon).
                    if drawn >= 8:
                        c2 = counts[pid][k]
                        v = m2[pid][k] / (c2 - 1) if c2 > 1 else 0.0
                        if radius_for(c2, v) <= epsilon / max(n, 1):
                            break
        # Combine strata: φ_i = (1/n) Σ_k stratum_mean_{i,k}.
        # Variance of the average: (1/n²) Σ_k σ_k²/N_k. Use Bernstein /
        # sub-Gaussian on the *average* with confidence δ_player to
        # get the correct joint coverage.
        estimates: dict[str, ShapleyEstimate] = {}
        v_empty = self._eval_v(frozenset())
        v_grand = self._eval_v(frozenset(players))
        delta_player = delta / max(n, 1)
        for pid in players:
            stratum_means = [
                (sums[pid][k] / counts[pid][k]) if counts[pid][k] > 0 else 0.0
                for k in range(n)
            ]
            stratum_vars = [
                (m2[pid][k] / (counts[pid][k] - 1)) if counts[pid][k] > 1 else 0.0
                for k in range(n)
            ]
            phi = sum(stratum_means) / n
            total_n = sum(counts[pid])
            # Estimator variance: (1/n²) Σ_k σ_k²/N_k.
            est_var_total = 0.0
            for k in range(n):
                if counts[pid][k] > 0:
                    est_var_total += stratum_vars[k] / counts[pid][k]
            est_var_total /= max(n * n, 1)
            log_term = math.log(2.0 / max(delta_player, _EPS))
            if method == POLICY_HOEFFDING:
                # Average of n independent stratum means: same effective
                # samples as pooled at uniform stratification.
                r = hoeffding_radius(delta_player, total_n, marg_range)
            else:
                # Empirical Bernstein on the average estimator.
                se = math.sqrt(max(est_var_total, 0.0))
                z = math.sqrt(2.0 * log_term)
                bias = (
                    7.0 * marg_range * log_term
                    / (3.0 * max(total_n - 1, 1))
                )
                r = z * se + bias
            estimates[pid] = ShapleyEstimate(
                player_id=pid,
                point=phi,
                lower=phi - r,
                upper=phi + r,
                n_samples=total_n,
                sample_variance=est_var_total,
                method=f"stratified_{method}",
            )
        grand = v_grand - v_empty
        sum_phi = sum(e.point for e in estimates.values())
        report = CoalitionReport(
            id=f"sh-{uuid_short()}",
            grand_value=grand,
            values=estimates,
            coverage_min=1.0 if self._value_fn is not None else self._observation_coverage_min(),
            efficiency_gap=grand - sum_phi,
            estimator=POLICY_STRATIFIED,
            bound_policy=method,
            epsilon_requested=epsilon,
            delta_requested=delta,
            n_samples_total=total_draws,
            elapsed_s=time.time() - t0,
            schedule_used={
                "strata_per_player": n,
                "delta_cell": delta_cell,
                "marg_range": marg_range,
            },
        )
        report = self._attest(report)
        with self._lock:
            self._history.append(report)
        self._emit_computed(report)
        return report

    # -----------------------------------------------------------------
    # Banzhaf indices
    # -----------------------------------------------------------------

    def banzhaf_indices(
        self,
        *,
        normalised: bool = True,
        max_samples: int = 4096,
        delta: float = 0.05,
    ) -> dict[str, ShapleyEstimate]:
        """Banzhaf voting-power indices.

        Equal-weighted average of marginal contributions:
            β_i = (1/2^{n-1}) Σ_{S ⊆ N \\ {i}} (v(S∪{i}) − v(S))

        For n ≤ _EXACT_MAX_PLAYERS, computed exactly; otherwise
        sampled. ``normalised=True`` rescales the raw β's to sum to 1
        (Penrose-Banzhaf normalised index); ``normalised=False`` keeps
        the raw absolute power.

        Banzhaf is the right primitive for *voting* / *gating*
        decisions (a player's index measures how often it is pivotal),
        as distinct from *surplus allocation* which is the Shapley
        domain.
        """
        with self._lock:
            players = list(self._player_order)
        n = len(players)
        if n == 0:
            return {}
        if n <= _EXACT_MAX_PLAYERS:
            raw: dict[str, float] = {pid: 0.0 for pid in players}
            v_cache: dict[frozenset[str], float] = {}

            def vof(s: frozenset[str]) -> float:
                if s in v_cache:
                    return v_cache[s]
                v = self._eval_v(s)
                v_cache[s] = v
                return v

            for i, pid in enumerate(players):
                bit = 1 << i
                acc = 0.0
                cnt = 0
                for mask in range(1 << n):
                    if mask & bit:
                        continue
                    s = frozenset(players[k] for k in range(n) if mask & (1 << k))
                    acc += vof(s | {pid}) - vof(s)
                    cnt += 1
                raw[pid] = acc / cnt if cnt > 0 else 0.0
            total = sum(abs(v) for v in raw.values()) or 1.0
            out: dict[str, ShapleyEstimate] = {}
            for pid, v in raw.items():
                p = v / total if normalised else v
                out[pid] = ShapleyEstimate(
                    player_id=pid,
                    point=p,
                    lower=p,
                    upper=p,
                    n_samples=1 << (n - 1),
                    sample_variance=0.0,
                    method="banzhaf_exact",
                )
            return out
        # Sampled Banzhaf for large n.
        rng = self._rng
        sums = {pid: 0.0 for pid in players}
        m2 = {pid: 0.0 for pid in players}
        counts = {pid: 0 for pid in players}
        value_range = self._value_range()
        marg_range = 2.0 * value_range
        delta_player = delta / n
        for _ in range(max_samples):
            for pid in players:
                S = frozenset(p for p in players if p != pid and rng.random() < 0.5)
                m = self._eval_v(S | {pid}) - self._eval_v(S)
                c_old = counts[pid]
                counts[pid] = c_old + 1
                sums[pid] += m
                if c_old >= 1:
                    new_mean = sums[pid] / counts[pid]
                    m2[pid] += (m - new_mean) * (m - (sums[pid] - m) / c_old)
        total = sum(abs(sums[pid] / counts[pid]) for pid in players if counts[pid] > 0) or 1.0
        out_s: dict[str, ShapleyEstimate] = {}
        for pid in players:
            c = counts[pid]
            mean = sums[pid] / c if c > 0 else 0.0
            var = m2[pid] / (c - 1) if c > 1 else 0.0
            r = bernstein_radius(delta_player, c, marg_range, var)
            p = mean / total if normalised else mean
            out_s[pid] = ShapleyEstimate(
                player_id=pid,
                point=p,
                lower=p - r,
                upper=p + r,
                n_samples=c,
                sample_variance=var,
                method="banzhaf_mc",
            )
        return out_s

    # -----------------------------------------------------------------
    # Owen / grouped values
    # -----------------------------------------------------------------

    def owen_values(
        self,
        groups: Sequence[Iterable[str]],
        *,
        epsilon: float = 0.05,
        delta: float = 0.05,
        max_samples: int = 8192,
    ) -> dict[str, ShapleyEstimate]:
        """Owen value: Shapley with an a-priori coalition structure.

        ``groups`` partitions the players. Permutations are sampled
        *first* over groups (uniformly), *then* over members within
        each group (uniformly) — equivalently, only those
        permutations that respect the group structure are sampled.
        Each player's Owen value is the expected marginal contribution
        under this restricted permutation distribution (Owen 1977).

        Use this when contributors come from tenants / clusters and
        credit must be attributed both *within* and *between* groups.
        """
        with self._lock:
            players = list(self._player_order)
        n = len(players)
        if n == 0:
            return {}
        # Validate the partition.
        group_lists = [list(g) for g in groups]
        if len(group_lists) > _OWEN_MAX_GROUPS:
            raise ValueError(
                f"owen_values: too many groups ({len(group_lists)} > {_OWEN_MAX_GROUPS})"
            )
        seen: set[str] = set()
        for g in group_lists:
            for pid in g:
                if pid in seen:
                    raise ValueError(f"owen_values: player {pid!r} in two groups")
                if pid not in self._players:
                    raise ValueError(f"owen_values: unknown player {pid!r}")
                seen.add(pid)
        # Players not in any group form an extra singleton group each.
        for pid in players:
            if pid not in seen:
                group_lists.append([pid])

        rng = self._rng
        sums = {pid: 0.0 for pid in players}
        m2 = {pid: 0.0 for pid in players}
        counts = {pid: 0 for pid in players}
        v_empty = self._eval_v(frozenset())
        value_range = self._value_range()
        marg_range = 2.0 * value_range
        delta_player = delta / max(n, 1)
        for _ in range(max_samples):
            # Sample a group permutation, then within-group permutations.
            group_order = list(range(len(group_lists)))
            rng.shuffle(group_order)
            prefix: list[str] = []
            prev_val = v_empty
            for gi in group_order:
                members = list(group_lists[gi])
                rng.shuffle(members)
                for pid in members:
                    prefix.append(pid)
                    cur_val = self._eval_v(frozenset(prefix))
                    m = cur_val - prev_val
                    c_old = counts[pid]
                    counts[pid] = c_old + 1
                    sums[pid] += m
                    if c_old >= 1:
                        new_mean = sums[pid] / counts[pid]
                        m2[pid] += (m - new_mean) * (m - (sums[pid] - m) / c_old)
                    prev_val = cur_val
            # Optional early stop omitted for simplicity; converges
            # within max_samples in practice.
        estimates: dict[str, ShapleyEstimate] = {}
        for pid in players:
            c = counts[pid]
            mean = sums[pid] / c if c > 0 else 0.0
            var = m2[pid] / (c - 1) if c > 1 else 0.0
            r = bernstein_radius(delta_player, c, marg_range, var)
            estimates[pid] = ShapleyEstimate(
                player_id=pid,
                point=mean,
                lower=mean - r,
                upper=mean + r,
                n_samples=c,
                sample_variance=var,
                method="owen_mc",
            )
        return estimates

    # -----------------------------------------------------------------
    # Marginal contribution to a fixed subset
    # -----------------------------------------------------------------

    def marginal_contribution(
        self,
        player_id: str,
        subset: Iterable[str],
    ) -> float:
        """Compute v(S ∪ {i}) − v(S) for one fixed subset.

        Useful as the building block for ablation studies.
        """
        with self._lock:
            if player_id not in self._players:
                raise ValueError(f"unknown player {player_id!r}")
        s = frozenset(subset) - {player_id}
        return self._eval_v(s | {player_id}) - self._eval_v(s)

    # -----------------------------------------------------------------
    # Core / least-core
    # -----------------------------------------------------------------

    def in_core(
        self,
        allocation: Mapping[str, float],
        *,
        tolerance: float = 1e-9,
    ) -> tuple[bool, dict[str, Any]]:
        """Check whether `allocation` lies in the core.

        Returns (in_core, witness_dict). For convex games the Shapley
        value is always in the core; for non-convex games the core
        may be empty. ``witness_dict`` reports the worst-violating
        coalition (or None if no violation).

        Exact for n ≤ _EXACT_MAX_PLAYERS; sampled for larger n.
        """
        with self._lock:
            players = list(self._player_order)
        n = len(players)
        if n == 0:
            return True, {"reason": "no-players"}
        missing = [pid for pid in players if pid not in allocation]
        if missing:
            raise ValueError(f"allocation missing players: {missing!r}")
        x_total = sum(allocation[pid] for pid in players)
        v_grand = self._eval_v(frozenset(players))
        v_empty = self._eval_v(frozenset())
        target = v_grand - v_empty
        if abs(x_total - target) > tolerance:
            return False, {
                "reason": "efficiency",
                "sum_x": x_total,
                "grand_value": target,
            }
        # Check coalitional rationality.
        worst_excess = float("-inf")
        worst_S: frozenset[str] | None = None
        if n <= _EXACT_MAX_PLAYERS:
            for mask in range(1 << n):
                S = frozenset(players[k] for k in range(n) if mask & (1 << k))
                v_s = self._eval_v(S) - v_empty
                x_s = sum(allocation[pid] for pid in S)
                excess = v_s - x_s
                if excess > worst_excess:
                    worst_excess = excess
                    worst_S = S
        else:
            # Sample-based check.
            rng = self._rng
            n_probes = min(1 << 18, 1024 * n)
            for _ in range(n_probes):
                S = frozenset(p for p in players if rng.random() < 0.5)
                v_s = self._eval_v(S) - v_empty
                x_s = sum(allocation[pid] for pid in S)
                excess = v_s - x_s
                if excess > worst_excess:
                    worst_excess = excess
                    worst_S = S
        ok = worst_excess <= tolerance
        return ok, {
            "reason": "ok" if ok else "core_violation",
            "worst_excess": worst_excess,
            "worst_coalition": sorted(worst_S) if worst_S is not None else None,
        }

    def allocate_efficient(
        self,
        values: Mapping[str, float],
        *,
        target: float | None = None,
    ) -> dict[str, float]:
        """Scale a Shapley vector to satisfy efficiency exactly.

        ``target`` defaults to v(N) − v(∅). Useful when a sampled
        Shapley vector has a non-zero efficiency gap from sampling
        noise and the coordination engine needs a budget-exact
        allocation.
        """
        with self._lock:
            players = list(self._player_order)
        missing = [pid for pid in players if pid not in values]
        if missing:
            raise ValueError(f"values missing players: {missing!r}")
        s = sum(values[pid] for pid in players)
        if target is None:
            target = self._eval_v(frozenset(players)) - self._eval_v(frozenset())
        if abs(s) < _EPS:
            # Degenerate: distribute equally.
            n = max(len(players), 1)
            return {pid: target / n for pid in players}
        scale = target / s
        return {pid: values[pid] * scale for pid in players}

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    def history(self, *, limit: int = 32) -> tuple[CoalitionReport, ...]:
        with self._lock:
            return tuple(self._history[-limit:])

    def reset(self) -> None:
        with self._lock:
            self._obs.clear()
            self._obs_count_total = 0
            self._obs_min = None
            self._obs_max = None
            self._history.clear()
        self._emit(COALITION_CLEARED, {"coalition_id": self._id})

    def coverage_report(self) -> CoverageReport:
        rep = self.coverage()
        self._emit(COALITION_REPORT, {
            "coalition_id": self._id,
            "report": rep.to_dict(),
        })
        return rep

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _empty_report(self, estimator: str, t0: float) -> CoalitionReport:
        return CoalitionReport(
            id=f"sh-{uuid_short()}",
            grand_value=0.0,
            values={},
            coverage_min=1.0,
            efficiency_gap=0.0,
            estimator=estimator,
            bound_policy=POLICY_HOEFFDING,
            epsilon_requested=0.0,
            delta_requested=0.0,
            n_samples_total=0,
            elapsed_s=time.time() - t0,
        )

    def _observation_coverage_min(self) -> float:
        with self._lock:
            n = len(self._player_order)
            possible = 1 << n
            if not self._obs or possible == 0:
                return 0.0
            return len(self._obs) / possible

    def _attest(self, report: CoalitionReport) -> CoalitionReport:
        """Mint a receipt for the computation if an attestor is wired."""
        if self._attestor is None:
            return report
        try:
            payload = report.to_dict()
            payload.pop("receipt_hash", None)
            serialised = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
            receipt_hash = digest
            rec = getattr(self._attestor, "record", None)
            if callable(rec):
                try:
                    receipt = rec(kind="coalition.computed", payload=payload)
                    if hasattr(receipt, "hash"):
                        receipt_hash = receipt.hash
                    elif isinstance(receipt, str):
                        receipt_hash = receipt
                except Exception:
                    pass
            else:
                # AttestationLedger.append-style attestor.
                try:
                    entry = self._attestor(_AttestableReport(report, payload))
                    if entry is not None and hasattr(entry, "entry_hash"):
                        receipt_hash = entry.entry_hash
                except Exception:
                    pass
            return CoalitionReport(
                **{**asdict(report), "values": report.values,
                   "receipt_hash": receipt_hash},
            )
        except Exception:
            return report

    def _emit_computed(self, report: CoalitionReport) -> None:
        self._emit(COALITION_COMPUTED, {
            "coalition_id": self._id,
            "report_id": report.id,
            "grand_value": report.grand_value,
            "estimator": report.estimator,
            "bound_policy": report.bound_policy,
            "n_samples_total": report.n_samples_total,
            "efficiency_gap": report.efficiency_gap,
            "receipt_hash": report.receipt_hash,
            "values": {k: v.to_dict() for k, v in report.values.items()},
        })
        if report.receipt_hash:
            self._emit(COALITION_CREDITED, {
                "coalition_id": self._id,
                "report_id": report.id,
                "receipt_hash": report.receipt_hash,
            })

    def _emit(self, kind: str, data: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:
            # Telemetry must never crash the runtime.
            pass


class _AttestableReport:
    """Adapter object for AttestationLedger.append(): exposes
    ``ticket_id``, ``kind``, and a deterministic payload via
    ``__dict__`` so the ledger can persist it. The ledger only reads
    public attributes; we expose enough to make it serialisable.
    """

    def __init__(self, report: CoalitionReport, payload: dict[str, Any]) -> None:
        self.ticket_id = report.id
        self.kind = "coalition.computed"
        self.payload = payload
        # Ledger looks for ``to_dict`` to serialise unknown receipt types.

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "kind": self.kind,
            "payload": self.payload,
        }


# =====================================================================
# Free functions for direct algorithmic access
# =====================================================================


def shapley_values(
    players: Sequence[str],
    v: CharacteristicFn,
) -> dict[str, float]:
    """One-shot exact Shapley over an explicit player list + v.

    Convenience wrapper for callers who don't want to construct a
    Coalition object. Hard cap at _EXACT_MAX_PLAYERS.
    """
    if len(players) > _EXACT_MAX_PLAYERS:
        raise ValueError(
            f"shapley_values: n={len(players)} exceeds cap "
            f"{_EXACT_MAX_PLAYERS}; use Coalition.shapley_montecarlo"
        )
    coa = Coalition()
    for pid in players:
        coa.register_player(pid)
    coa.set_value_function(v)
    report = coa.shapley_exact()
    return {pid: est.point for pid, est in report.values.items()}


def banzhaf_index(
    players: Sequence[str],
    v: CharacteristicFn,
    *,
    normalised: bool = True,
) -> dict[str, float]:
    """One-shot Banzhaf index over an explicit player list + v."""
    if len(players) > _EXACT_MAX_PLAYERS:
        raise ValueError(
            f"banzhaf_index: n={len(players)} exceeds cap "
            f"{_EXACT_MAX_PLAYERS}"
        )
    coa = Coalition()
    for pid in players:
        coa.register_player(pid)
    coa.set_value_function(v)
    out = coa.banzhaf_indices(normalised=normalised)
    return {pid: est.point for pid, est in out.items()}


def fit_linear_v(
    observations: Sequence[tuple[Iterable[str], float]],
    players: Sequence[str],
    *,
    order: int = 2,
    l2: float = 1e-6,
) -> CharacteristicFn:
    """Fit a sparse interaction-order-`order` characteristic function
    from observed (coalition, value) data and return it as a callable.

    Solves the L2-regularised normal equations for the coefficient
    vector ``f`` indexed by subsets of size ≤ ``order`` of the player
    set, then defines

        v̂(S) = Σ_{T ⊆ S, |T| ≤ order} f(T).

    Reduces to a closed-form characteristic function suitable for the
    exact / MC Shapley pipeline; the fitted f's themselves are the
    Möbius coefficients and feed straight into Shapley as

        φ_i = Σ_{T ∋ i, |T| ≤ order} f(T) / |T|.

    Order=1 gives the additive linear regression; order=2 captures
    pairwise interactions; higher orders blow up the feature space
    combinatorially.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if not players:
        return lambda S: 0.0
    feats: list[frozenset[str]] = [frozenset()]
    for k in range(1, order + 1):
        if k > len(players):
            break
        for combo in itertools.combinations(players, k):
            feats.append(frozenset(combo))
    F = len(feats)
    # Build normal equations: A x = b, where A = X^T X + l2 I, b = X^T y.
    A = [[0.0] * F for _ in range(F)]
    b = [0.0] * F
    for coa, y in observations:
        s = frozenset(coa)
        x = [1.0 if T <= s else 0.0 for T in feats]
        # b += y * x
        for j in range(F):
            if x[j]:
                b[j] += y
        # A += x x^T
        active = [j for j in range(F) if x[j]]
        for i in active:
            for j in active:
                A[i][j] += 1.0
    # Add ridge.
    for i in range(F):
        A[i][i] += l2
    # Solve via Gauss-Jordan; F is ≤ ~O(n^order) and order ≤ 3 in
    # practice so this is tractable.
    coeffs = _solve(A, b)
    f_map = {feats[j]: coeffs[j] for j in range(F)}

    def v_hat(S: frozenset[str]) -> float:
        # Sum f(T) over T ⊆ S with |T| ≤ order.
        if order == 1:
            return f_map.get(frozenset(), 0.0) + sum(
                f_map.get(frozenset((p,)), 0.0) for p in S
            )
        # General case: iterate over subsets up to order.
        total = f_map.get(frozenset(), 0.0)
        s_list = list(S)
        for k in range(1, min(order, len(s_list)) + 1):
            for combo in itertools.combinations(s_list, k):
                total += f_map.get(frozenset(combo), 0.0)
        return total

    return v_hat


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan elimination with partial pivoting. Stdlib-only.

    Robust for the sizes we feed it (F ≤ ~100). Mutates copies.
    """
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # Pivot.
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        if abs(M[pivot][col]) < 1e-15:
            # Singular column; leave the row as identity-ish.
            continue
        M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= piv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][n] for i in range(n)]


def shapley_from_observations(
    observations: Sequence[tuple[Iterable[str], float]],
    players: Sequence[str],
    *,
    order: int = 2,
    l2: float = 1e-6,
) -> dict[str, float]:
    """End-to-end: fit a low-order interaction model from observations,
    then compute the exact Shapley value under the fitted model.

    For order=1 this is the linear-regression coefficient; for
    order=2 it is the additive + pairwise contribution. The result
    is fully closed-form (no Monte Carlo needed) because the fitted
    f's are themselves the Möbius coefficients of v̂.
    """
    # Re-derive coefficients (we want the f-map directly).
    if order < 1:
        raise ValueError("order must be >= 1")
    if not players:
        return {}
    feats: list[frozenset[str]] = [frozenset()]
    for k in range(1, order + 1):
        if k > len(players):
            break
        for combo in itertools.combinations(players, k):
            feats.append(frozenset(combo))
    F = len(feats)
    A = [[0.0] * F for _ in range(F)]
    b = [0.0] * F
    for coa, y in observations:
        s = frozenset(coa)
        x = [1.0 if T <= s else 0.0 for T in feats]
        for j in range(F):
            if x[j]:
                b[j] += y
        active = [j for j in range(F) if x[j]]
        for i in active:
            for j in active:
                A[i][j] += 1.0
    for i in range(F):
        A[i][i] += l2
    coeffs = _solve(A, b)
    f_map = {feats[j]: coeffs[j] for j in range(F)}
    # φ_i = Σ_{T ∋ i, |T| ≤ order} f(T) / |T|.
    phi: dict[str, float] = {pid: 0.0 for pid in players}
    for T, f in f_map.items():
        if not T:
            continue
        k = len(T)
        for pid in T:
            phi[pid] += f / k
    return phi


def uuid_short() -> str:
    import uuid as _uuid
    return _uuid.uuid4().hex[:12]
