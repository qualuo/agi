"""Cartographer — zone-of-proximal-development curriculum kernel.

Every long-running runtime that *learns* eventually faces the meta-problem
the coordination engine can't dodge: of the thousands of tasks, skills,
prompts and benchmarks reachable from here, **which one should we attempt
next?** Without an answer, the runtime drifts: it overspends on tasks it
already masters, ignores tasks just beyond reach, and never discovers the
ones it could not yet attempt because their prerequisites are unfilled.

The literature names this problem in three places that converged on the
same shape:

  * **Vygotsky, 1934 — Zone of Proximal Development (ZPD).** Learning
    progresses fastest in tasks the learner cannot do alone but can do
    *with scaffolding*. The frontier of competence is the locus of
    growth; below it, no signal; above it, no progress.
  * **Oudeyer & Kaplan, 2007 — Intrinsic Motivation Systems / IAC.**
    The agent that maximises its own *learning progress* — the rate
    of decrease of prediction error — develops more general
    competencies than one driven by extrinsic reward alone. Practical
    operationalisation: track `LP_i = |μ̂_recent − μ̂_prev|` over a
    sliding window and pick the task with the largest LP.
  * **Graves et al., 2017 — Automated Curriculum Learning.** Frames
    curriculum as a non-stationary multi-armed bandit over tasks
    where reward is gain-in-competence; EXP3.S with a forgetting
    factor handles the non-stationarity.

`Cartographer` is the runtime primitive that closes this loop. It
maintains a Bayesian posterior over competence per task, derives an
anytime-valid Wilson confidence interval, computes a learning-progress
signal, and emits curriculum recommendations that the coordination
engine can route through the rest of the runtime: hand the frontier to
`Arbiter` for fixed-confidence BAI, ask `ExperimentDesigner` which
frontier item is most informative, gate mastery with `Strategist` and
`PolicyImprover`. The complement to *"how do I commit?"* (Arbiter) is
*"what should I commit to next?"* — that is Cartographer.

Mathematical core
-----------------

For each task ``i`` we maintain conjugate Beta-Binomial state ``(s_i,
n_i)`` (Bernoulli outcomes) or sufficient-statistics state ``(Σx, Σx²,
n_i)`` (Gaussian). The posterior mean uses a Jeffreys prior
``Beta(½, ½)`` by default — the uniform-noninformative choice that
matches the asymptotic frequentist coverage of the Wilson interval:

    μ̂_i = (s_i + ½) / (n_i + 1)

The **Wilson score interval** is the anytime confidence band:

    L_i, U_i = ( p̂ + z²/(2n) ± z · √( p̂(1−p̂)/n + z²/(4n²) ) ) / (1 + z²/n)

with ``p̂ = s_i / n_i`` and ``z = Φ⁻¹(1 − α/2)``. The Wilson interval
is the right primitive for *competence* because it is well-behaved at
the boundaries (n=0 → L=0, U=1; s=n → L < 1) and matches the score
test's asymptotic level. We expose `beta_lcb` / `beta_ucb` for users
who want the Bayesian Beta-quantile bound and `clopper_pearson_ci` for
the strictly-conservative exact bound.

The **learning-progress signal** is computed on the sliding ring buffer
of recent outcomes. Given the last ``W = window_recent + window_prior``
outcomes, the LP is

    LP_i = ( μ_recent − μ_prior )

i.e. the *signed* slope across the window (Oudeyer 2007 used absolute
value; we keep the sign so the coordinator can distinguish *gain* from
*regression*). Magnitudes are smoothed by an exponential weight so a
single bad outcome on a frontier task does not flip the recommendation.

The **frontier** is the set of tasks with ``U_i ≥ entry_threshold``
and ``L_i < mastery_threshold`` — the zone above noise-floor and below
mastered. Tasks with all prerequisites mastered enter the frontier;
tasks whose lower CB crosses ``mastery_threshold`` exit. A "fragile"
status names mastered tasks whose mean has subsequently dropped — the
hook DriftSentinel uses to demote a stale mastery.

The **budgeted recommendation** is the submodular knapsack

    maximise   Σ_i x_i · ( ω_LP · LP_i + ω_unc · σ_i ) · v_i
    s.t.       Σ_i x_i · c_i · n̂_i ≤ B,   x_i ∈ {0, 1}

where ``σ_i ≈ (U_i − L_i)/2`` is the posterior uncertainty, ``v_i`` is
the task value, and ``n̂_i`` is the expected pulls to advance one
posterior-half-width step. The objective is monotone-submodular under
diminishing returns of uncertainty reduction so the cost-greedy
heuristic (Sviridenko 2004) ships with a `(1 - 1/e)`-of-OPT guarantee
when items are individually small relative to the budget.

What it composes (razor-sharp coordination integration)
------------------------------------------------------

  * **Strategist.** Strategist asks "what should I do *for this
    ticket*?". Cartographer asks "what should I do *next overall*?".
    A typical loop: every N tickets the coordinator calls
    `cartographer.recommend(policy=POLICY_LP, k=8)` to update the
    active task set; Strategist then routes individual tickets
    within that set.

  * **Arbiter.** Once Cartographer has nominated a frontier subset,
    `arbiter.start_campaign(arms=frontier_ids, delta=δ)` runs BAI
    inside that subset to commit to one. Two complementary
    primitives: Cartographer says *which K are worth running*,
    Arbiter says *which of those K is best*.

  * **ExperimentDesigner.** The Bayesian dual: ExperimentDesigner
    chooses the design x* maximising EIG over a model;
    Cartographer chooses the task whose observation most reduces
    its own posterior variance. Calling
    `experiment_design.bald_score(...)` over the frontier and
    passing the result back as `extra_score` is supported via the
    `score_fn=` hook of `recommend`.

  * **CalibrationEngine.** The cartographer's predicted competence
    ``μ̂_i`` is exactly the kind of probability calibration the
    engine consumes. Pipe `cartographer.competence(i).mean` →
    CalibrationEngine before promoting a task to "mastered".

  * **DriftSentinel.** Subscribe DriftSentinel to a mastered task's
    reward stream; on drift the cartographer auto-demotes the task
    via `regress(task_id)` and the curriculum re-opens it.

  * **AttestationLedger.** Mastery transitions emit
    `cartographer.advanced` with a content-hash receipt — a
    third-party-replayable proof that a task was mastered with the
    observed n and the observed mean, at the observed time.

  * **PolicyImprover.** Before a mastery transition ships, the
    coordinator can call `PolicyImprover.safety_check(...)` over
    the task's induced policy; if HCPI rejects, Cartographer keeps
    the task on the frontier rather than promoting.

  * **EventBus.** Streams every observation, recommendation, and
    transition. The bus is how a coordination engine reacts in
    real time — e.g., spawn a backfill worker on
    `cartographer.advanced`, retrain the calibrator on
    `cartographer.frontier_changed`.

Where this slots in
-------------------

    cart = Cartographer(bus=bus, attestor=attestor)
    cart.register_task("two-digit-add", value=1.0, cost=0.002)
    cart.register_task("three-digit-add", value=2.0, cost=0.003,
                        prereqs=("two-digit-add",))
    cart.register_task("long-division", value=4.0, cost=0.010,
                        prereqs=("three-digit-add",))

    for outcome in driver.completed_for_task("two-digit-add"):
        cart.observe("two-digit-add", float(outcome.success))
    curriculum = cart.recommend(policy=POLICY_LP, k=4, budget=0.20)
    for item in curriculum.items:
        coordinator.queue(item.task_id, pulls=item.pulls)

Events
------
    cartographer.started            — a task was registered
    cartographer.observed           — an outcome was recorded
    cartographer.recommended        — a curriculum was emitted
    cartographer.advanced           — a task became mastered
    cartographer.regressed          — a mastered task dropped
    cartographer.frontier_changed   — frontier composition changed
    cartographer.cleared            — a task / the whole map was reset
    cartographer.report             — a coverage report was published

Honest about limits
-------------------

  * Wilson CI is asymptotic; for n_i < 5, prefer `clopper_pearson_ci`
    if a strictly-valid bound matters. We default to Wilson because
    its calibration on the regime that matters (5 ≤ n ≤ 5000) is
    materially better.
  * The submodular-knapsack approximation guarantee assumes per-item
    cost is small relative to budget. For lumpy cost regimes the
    coordinator should run `recommend(...).items` through its own
    feasibility check.
  * Learning-progress can hallucinate gains on a tiny window with a
    lucky streak. We require ``window_recent + window_prior ≤ n_i``
    before reporting LP; otherwise LP is undefined and reported as
    ``None``.
  * Prereq DAGs are validated for cycles at registration; a cycle
    raises `ValueError`. We do not attempt to resolve dependency
    *chains* — a task is locked iff *every direct* prereq is mastered.
    Transitive locking emerges naturally because mastery propagates
    on each `tick`.
  * Cartographer is **not** a planner. It does not synthesize tasks
    it has not been told about. Task generation lives upstream in
    `skillmine` / `goalc`; Cartographer schedules over a fixed pool.

Stdlib-only, CPU-bound, threadsafe; identical I/O surface to
`Arbiter`, `Deliberator`, `Strategist`, `ExperimentDesigner` so a
coordination engine can compose them uniformly.

Citations
---------

* Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher
  Psychological Processes.* Harvard University Press. [ZPD]
* Oudeyer, P.-Y., & Kaplan, F. (2007). What is intrinsic motivation? A
  typology of computational approaches. *Frontiers in Neurorobotics*,
  1, 6.
* Graves, A., et al. (2017). Automated Curriculum Learning for Neural
  Networks. *Proc. ICML*, 1311–1320.
* Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. *JASA*, 22(158), 209–212.
* Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or
  fiducial limits illustrated in the case of the binomial. *Biometrika*,
  26(4), 404–413.
* Sviridenko, M. (2004). A note on maximizing a submodular set
  function subject to a knapsack constraint. *Operations Research
  Letters*, 32(1), 41–43.
* Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
  Curriculum learning. *Proc. ICML*, 41–48.
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
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

CARTOGRAPHER_STARTED = "cartographer.started"
CARTOGRAPHER_OBSERVED = "cartographer.observed"
CARTOGRAPHER_RECOMMENDED = "cartographer.recommended"
CARTOGRAPHER_ADVANCED = "cartographer.advanced"
CARTOGRAPHER_REGRESSED = "cartographer.regressed"
CARTOGRAPHER_FRONTIER_CHANGED = "cartographer.frontier_changed"
CARTOGRAPHER_CLEARED = "cartographer.cleared"
CARTOGRAPHER_REPORT = "cartographer.report"


# =====================================================================
# Policies & statuses
# =====================================================================

POLICY_LP = "lp"
"""Greedy learning-progress: pick frontier tasks with the largest LP_i."""

POLICY_UCB = "ucb"
"""Wilson upper-confidence on (value × competence) with bonus."""

POLICY_INFOGAIN = "infogain"
"""Posterior-variance reduction (one-step) per unit cost."""

POLICY_THOMPSON = "thompson"
"""Beta-Thompson sampling: draw μ̃_i ~ Beta(α_0+s, β_0+n-s), rank by value × μ̃."""

POLICY_KNAPSACK = "knapsack"
"""Submodular cost-greedy knapsack maximising (LP + uncertainty) × value."""

POLICY_ROUND_ROBIN = "round_robin"
"""Anti-degeneracy fallback: cycles through the frontier — useful for cold-start."""

KNOWN_POLICIES = (
    POLICY_LP,
    POLICY_UCB,
    POLICY_INFOGAIN,
    POLICY_THOMPSON,
    POLICY_KNAPSACK,
    POLICY_ROUND_ROBIN,
)

STATUS_NOVICE = "novice"
"""Upper CB below entry threshold — task is below noise floor."""
STATUS_FRONTIER = "frontier"
"""Lower CB below mastery, upper CB above entry — the zone of proximal development."""
STATUS_MASTERED = "mastered"
"""Lower CB ≥ mastery threshold — committed."""
STATUS_LOCKED = "locked"
"""Has unmet prerequisites — cannot be sampled regardless of competence."""
STATUS_FRAGILE = "fragile"
"""Previously mastered, mean has since dropped — candidate for re-mastery."""

KNOWN_STATUSES = (
    STATUS_NOVICE,
    STATUS_FRONTIER,
    STATUS_MASTERED,
    STATUS_LOCKED,
    STATUS_FRAGILE,
)

REWARD_BERNOULLI = "bernoulli"
REWARD_GAUSSIAN = "gaussian"
KNOWN_REWARDS = (REWARD_BERNOULLI, REWARD_GAUSSIAN)


# =====================================================================
# Numerical defaults
# =====================================================================

_DEFAULT_PRIOR_ALPHA = 0.5
_DEFAULT_PRIOR_BETA = 0.5
_DEFAULT_ENTRY_THRESHOLD = 0.2
_DEFAULT_MASTERY_THRESHOLD = 0.8
_DEFAULT_WINDOW_RECENT = 12
_DEFAULT_WINDOW_PRIOR = 12
_DEFAULT_Z = 1.959963984540054   # z_{0.975}
_DEFAULT_GAUSSIAN_VAR = 1.0
_EPS = 1e-12


# =====================================================================
# Numerical primitives (module-level — composable with Strategist etc.)
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wilson_ci(
    successes: int,
    n: int,
    *,
    z: float = _DEFAULT_Z,
) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli mean.

    With ``n = 0`` returns the trivial ``(0.0, 1.0)``. With ``z = 1.96``
    the interval has asymptotic coverage 0.95. The Wilson form is the
    correct choice for *competence* tracking because it never goes
    below 0 or above 1, is well-defined for ``s ∈ {0, n}``, and
    matches the score test of the binomial proportion.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    return low, high


def clopper_pearson_ci(
    successes: int,
    n: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Exact Clopper–Pearson binomial CI via the F-distribution dual.

    Strict (overcoverage), but the right choice when ``n`` is small or
    the user wants a coverage *guarantee* rather than an asymptotic
    target. Uses the regularised-incomplete-Beta function via Newton
    inversion since stdlib does not ship beta-quantile.
    """
    if n <= 0:
        return 0.0, 1.0
    if successes <= 0:
        low = 0.0
    else:
        low = _beta_inv(alpha / 2.0, successes, n - successes + 1)
    if successes >= n:
        high = 1.0
    else:
        high = _beta_inv(1.0 - alpha / 2.0, successes + 1, n - successes)
    return low, high


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b) via Lentz's continued fraction.

    Numerically stable for the regime we use (a, b ≥ 1, x ∈ [0, 1]).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-12) -> float:
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
        if abs(delta - 1.0) < tol:
            return h
    return h


def _beta_inv(p: float, a: float, b: float) -> float:
    """Inverse regularised incomplete beta via bisection-with-Newton fallback.

    Robust and stdlib-only. The regime (a, b ≤ 1e5, p ∈ (0, 1)) is
    well-handled in O(40) iterations to 1e-9.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = _reg_inc_beta(mid, a, b)
        if v < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def beta_lcb(
    successes: int,
    n: int,
    *,
    delta: float = 0.05,
    alpha: float = _DEFAULT_PRIOR_ALPHA,
    beta: float = _DEFAULT_PRIOR_BETA,
) -> float:
    """Bayesian lower credible bound on a Beta-Binomial posterior.

    Returns ``Q_δ(Beta(α + s, β + n - s))``. Used as the *Bayesian*
    counterpart to `wilson_ci`'s lower endpoint when a posterior
    interpretation is the right one for the coordinator (e.g., "ship
    if Pr[μ ≥ τ] ≥ 1 − δ").
    """
    a = alpha + successes
    b = beta + max(0, n - successes)
    if a <= 0 or b <= 0:
        return 0.0
    return _beta_inv(delta, a, b)


def beta_ucb(
    successes: int,
    n: int,
    *,
    delta: float = 0.05,
    alpha: float = _DEFAULT_PRIOR_ALPHA,
    beta: float = _DEFAULT_PRIOR_BETA,
) -> float:
    """Bayesian upper credible bound; complement of `beta_lcb`."""
    a = alpha + successes
    b = beta + max(0, n - successes)
    if a <= 0 or b <= 0:
        return 1.0
    return _beta_inv(1.0 - delta, a, b)


def learning_progress(
    outcomes: Sequence[float],
    *,
    window_recent: int = _DEFAULT_WINDOW_RECENT,
    window_prior: int = _DEFAULT_WINDOW_PRIOR,
) -> float | None:
    """Oudeyer-style learning-progress signal on a sliding window.

    Returns the *signed* mean delta ``μ_recent − μ_prior``. Positive →
    competence improving (the desirable signal for ZPD); negative →
    regression; zero → stagnation. Returns ``None`` when there is
    insufficient history to form both windows.
    """
    n = len(outcomes)
    need = window_recent + window_prior
    if n < need:
        return None
    recent = outcomes[-window_recent:]
    prior = outcomes[-need:-window_recent]
    if not recent or not prior:
        return None
    return statistics.fmean(recent) - statistics.fmean(prior)


def _exp_smoothed(values: Sequence[float], alpha: float = 0.3) -> float:
    """Exponentially-smoothed mean (most-recent-weight = alpha)."""
    if not values:
        return 0.0
    out = values[0]
    for v in values[1:]:
        out = alpha * v + (1.0 - alpha) * out
    return out


def submodular_knapsack(
    items: Sequence[tuple[str, float, float]],
    budget: float,
) -> list[str]:
    """Cost-greedy submodular knapsack with Sviridenko's swap.

    Items are ``(id, gain, cost)`` with gain ≥ 0 and cost > 0. Returns
    the chosen ids. The returned set has objective value at least
    ``(1 - 1/e)`` × OPT *provided* every individual item's cost is
    small relative to the budget (Sviridenko 2004); otherwise the
    routine falls back to the better of the cost-greedy set and the
    single-item argmax — the standard `(1/2)(1 - 1/e)` guarantee.
    """
    if budget <= 0.0 or not items:
        return []
    valid = [(i, g, c) for (i, g, c) in items if g >= 0.0 and c > 0.0]
    # Cost-greedy.
    ordered = sorted(valid, key=lambda x: x[1] / x[2], reverse=True)
    chosen: list[str] = []
    spent = 0.0
    chosen_gain = 0.0
    for tid, gain, cost in ordered:
        if spent + cost <= budget + _EPS:
            chosen.append(tid)
            spent += cost
            chosen_gain += gain
    # Single-item argmax fallback (Sviridenko's swap).
    best_single = max(
        ((i, g) for (i, g, c) in valid if c <= budget + _EPS),
        default=None, key=lambda x: x[1],
    )
    if best_single is not None and best_single[1] > chosen_gain:
        return [best_single[0]]
    return chosen


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class Competence:
    """Per-task posterior snapshot at the moment of query."""

    task_id: str
    n: int
    mean: float            # smoothed posterior mean
    raw_mean: float        # s / n (or n=0 -> 0)
    lower: float           # CI lower (Wilson by default)
    upper: float           # CI upper
    variance: float        # posterior variance
    last_value: float | None
    last_seen_at: float
    learning_progress: float | None  # None when window not full
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    """Static configuration for a task in the curriculum."""

    id: str
    value: float = 1.0
    cost: float = 1.0
    prereqs: tuple[str, ...] = ()
    reward_model: str = REWARD_BERNOULLI
    sigma2: float = _DEFAULT_GAUSSIAN_VAR
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumItem:
    """One recommended sample-batch."""

    task_id: str
    pulls: int
    score: float
    cost: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Curriculum:
    """A coordinator-consumable recommendation bundle."""

    id: str
    policy: str
    k: int
    budget: float | None
    items: list[CurriculumItem]
    total_cost: float
    rationale: str
    issued_at: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [i.to_dict() if hasattr(i, "to_dict") else dict(i) for i in self.items]
        return d


@dataclass
class TransitionEvent:
    """A status change emitted by `tick()`."""

    task_id: str
    old_status: str
    new_status: str
    at: float
    n: int
    mean: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TickReport:
    """Returned by `tick()` — aggregates the transitions for this tick."""

    advanced: list[TransitionEvent]
    regressed: list[TransitionEvent]
    frontier_size: int
    mastered_size: int
    locked_size: int
    novice_size: int
    fragile_size: int
    at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "advanced": [e.to_dict() for e in self.advanced],
            "regressed": [e.to_dict() for e in self.regressed],
            "frontier_size": self.frontier_size,
            "mastered_size": self.mastered_size,
            "locked_size": self.locked_size,
            "novice_size": self.novice_size,
            "fragile_size": self.fragile_size,
            "at": self.at,
        }


@dataclass
class CoverageReport:
    """Calibration of the cartographer's own predictions.

    For each task, when the cartographer predicted ``μ̂`` at any
    point in time and the eventual *long-run* empirical mean came in
    at ``μ̄``, we measure ``μ̂ − μ̄`` and aggregate. A well-calibrated
    cartographer has mean error near zero and Brier ≤ a few percent.
    """

    n_tasks: int
    n_predictions: int
    mean_signed_error: float
    mean_abs_error: float
    brier_score: float
    n_by_status: dict[str, int]
    mean_n_by_status: dict[str, float]
    mean_lp_by_status: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Internal per-task state
# =====================================================================


@dataclass
class _TaskState:
    spec: TaskSpec
    n: int = 0
    s: float = 0.0           # successes (Bernoulli) or Σx (Gaussian)
    sum_sq: float = 0.0
    last_value: float | None = None
    last_seen_at: float = 0.0
    first_seen_at: float = 0.0
    ring: list[float] = field(default_factory=list)
    ring_capacity: int = 256
    status: str = STATUS_LOCKED
    last_status_change_at: float = 0.0
    mastered_at: float | None = None
    mastery_mean: float | None = None
    advance_count: int = 0
    regress_count: int = 0
    rr_cursor: int = 0       # for round-robin tiebreaking

    def record(self, value: float) -> None:
        self.n += 1
        self.s += value
        self.sum_sq += value * value
        self.last_value = value
        self.ring.append(value)
        if len(self.ring) > self.ring_capacity:
            self.ring = self.ring[-self.ring_capacity:]

    def raw_mean(self) -> float:
        return (self.s / self.n) if self.n > 0 else 0.0

    def posterior_mean(self, *, alpha: float, beta: float) -> float:
        return (self.s + alpha) / (self.n + alpha + beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "n": self.n,
            "s": self.s,
            "sum_sq": self.sum_sq,
            "last_value": self.last_value,
            "last_seen_at": self.last_seen_at,
            "first_seen_at": self.first_seen_at,
            "ring": list(self.ring),
            "status": self.status,
            "last_status_change_at": self.last_status_change_at,
            "mastered_at": self.mastered_at,
            "mastery_mean": self.mastery_mean,
            "advance_count": self.advance_count,
            "regress_count": self.regress_count,
        }


# =====================================================================
# Cartographer
# =====================================================================


class Cartographer:
    """Zone-of-proximal-development curriculum kernel.

    Thread-safe. Wire an `EventBus` to stream every observation,
    recommendation, and transition to the coordination engine. Wire
    an `attestor` (typically `RuntimeAttestor` over
    `AttestationLedger`) to produce tamper-evident mastery receipts.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any | None = None,
        entry_threshold: float = _DEFAULT_ENTRY_THRESHOLD,
        mastery_threshold: float = _DEFAULT_MASTERY_THRESHOLD,
        delta: float = 0.05,
        z: float = _DEFAULT_Z,
        prior_alpha: float = _DEFAULT_PRIOR_ALPHA,
        prior_beta: float = _DEFAULT_PRIOR_BETA,
        window_recent: int = _DEFAULT_WINDOW_RECENT,
        window_prior: int = _DEFAULT_WINDOW_PRIOR,
        smoothing_alpha: float = 0.3,
        regression_margin: float = 0.05,
    ) -> None:
        if not 0.0 < entry_threshold < mastery_threshold < 1.0:
            raise ValueError(
                "require 0 < entry_threshold < mastery_threshold < 1"
            )
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")
        if window_recent < 1 or window_prior < 1:
            raise ValueError("windows must be positive")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior pseudo-counts must be positive")
        self._bus = bus
        self._attestor = attestor
        self.entry_threshold = float(entry_threshold)
        self.mastery_threshold = float(mastery_threshold)
        self.delta = float(delta)
        self.z = float(z)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.window_recent = int(window_recent)
        self.window_prior = int(window_prior)
        self.smoothing_alpha = float(smoothing_alpha)
        self.regression_margin = float(regression_margin)
        self._lock = threading.RLock()
        self._tasks: dict[str, _TaskState] = {}
        self._truth: dict[str, float] = {}        # for coverage_report
        self._prediction_log: list[tuple[str, float, float]] = []
        self._frontier_signature: tuple[str, ...] = ()
        self._recommended_count = 0
        self._round_robin_cursor = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_task(
        self,
        task_id: str,
        *,
        value: float = 1.0,
        cost: float = 1.0,
        prereqs: Sequence[str] = (),
        reward_model: str = REWARD_BERNOULLI,
        sigma2: float = _DEFAULT_GAUSSIAN_VAR,
        tags: Sequence[str] = (),
    ) -> TaskSpec:
        """Add a task to the curriculum.

        Raises ``ValueError`` for invalid configuration (unknown
        reward model, non-positive cost, cyclic prereq graph). Idempotent
        on identical re-registration; raises on conflicting
        re-registration so the coordinator detects drift in task
        catalog vs. runtime.
        """
        task_id = str(task_id)
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if reward_model not in KNOWN_REWARDS:
            raise ValueError(f"unknown reward model: {reward_model!r}")
        if cost <= 0.0:
            raise ValueError("cost must be > 0")
        if value < 0.0:
            raise ValueError("value must be ≥ 0")
        if sigma2 <= 0.0:
            raise ValueError("sigma2 must be > 0")
        spec = TaskSpec(
            id=task_id,
            value=float(value),
            cost=float(cost),
            prereqs=tuple(str(p) for p in prereqs),
            reward_model=reward_model,
            sigma2=float(sigma2),
            tags=tuple(str(t) for t in tags),
        )
        with self._lock:
            if task_id in self._tasks:
                existing = self._tasks[task_id].spec
                if existing != spec:
                    raise ValueError(
                        f"task {task_id!r} already registered with "
                        f"different spec: {existing!r} vs {spec!r}"
                    )
                return existing
            self._tasks[task_id] = _TaskState(
                spec=spec,
                first_seen_at=time.time(),
                last_status_change_at=time.time(),
            )
            # Validate the prereq graph still has no cycles.
            self._validate_dag_locked()
            # Initial status.
            self._refresh_status_locked(task_id, emit=False)
            self._emit(CARTOGRAPHER_STARTED, {
                "task_id": task_id,
                "spec": spec.to_dict(),
            })
            return spec

    def unregister_task(self, task_id: str) -> bool:
        """Remove a task. Returns True if it was present."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            # Reject if other tasks depend on it.
            dependents = [
                tid for tid, st in self._tasks.items()
                if task_id in st.spec.prereqs and tid != task_id
            ]
            if dependents:
                raise ValueError(
                    f"cannot remove {task_id!r}: required by {dependents!r}"
                )
            del self._tasks[task_id]
            self._emit(CARTOGRAPHER_CLEARED, {"task_id": task_id})
            return True

    def _validate_dag_locked(self) -> None:
        """Raise on cyclic prereq graph."""
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}

        def dfs(tid: str, path: list[str]) -> None:
            color[tid] = GREY
            path.append(tid)
            for prereq in self._tasks[tid].spec.prereqs:
                if prereq not in self._tasks:
                    continue   # dangling — locks the dependent, not a cycle
                if color[prereq] == GREY:
                    cycle = " -> ".join(path[path.index(prereq):] + [prereq])
                    raise ValueError(f"cyclic prereq graph: {cycle}")
                if color[prereq] == WHITE:
                    dfs(prereq, path)
            path.pop()
            color[tid] = BLACK

        for tid in list(self._tasks):
            if color[tid] == WHITE:
                dfs(tid, [])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        task_id: str,
        outcome: float,
        *,
        at: float | None = None,
    ) -> Competence:
        """Record one outcome on a task; return updated `Competence`.

        For ``reward_model = REWARD_BERNOULLI`` the outcome must lie
        in ``[0, 1]`` (commonly 0 or 1). For ``REWARD_GAUSSIAN`` it
        is the realised real-valued reward. Triggers a `tick()` to
        propagate any status transitions, except no transition-event
        is emitted from inside `observe`; call `tick()` explicitly
        if the caller wants the transition bundle.
        """
        ts = at if at is not None else time.time()
        v = float(outcome)
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            st = self._tasks[task_id]
            if st.spec.reward_model == REWARD_BERNOULLI:
                if not 0.0 <= v <= 1.0:
                    raise ValueError(
                        "Bernoulli outcome must be in [0, 1]"
                    )
            st.record(v)
            st.last_seen_at = ts
            self._emit(CARTOGRAPHER_OBSERVED, {
                "task_id": task_id,
                "outcome": v,
                "n": st.n,
                "mean": st.raw_mean(),
                "ts": ts,
            })
            # Status transitions are deferred to tick() so a single
            # observation never silently advances or regresses a task.
            comp = self._competence_locked(task_id)
            # Predict-log for coverage_report.
            self._prediction_log.append((task_id, comp.mean, ts))
            if len(self._prediction_log) > 32_768:
                self._prediction_log = self._prediction_log[-32_768:]
            return comp

    def observe_many(
        self,
        task_id: str,
        outcomes: Iterable[float],
        *,
        at: float | None = None,
    ) -> Competence:
        comp = None
        for o in outcomes:
            comp = self.observe(task_id, o, at=at)
        if comp is None:
            raise ValueError("outcomes is empty")
        return comp

    def regress(self, task_id: str, *, rationale: str = "manual") -> None:
        """Manually demote a mastered task back to the frontier.

        Intended for the DriftSentinel integration: when drift on a
        mastered task is detected the coordinator calls
        ``cart.regress(task_id, rationale='drift')`` and the task
        re-enters the curriculum. The transition is logged and
        attested if an attestor is wired.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            st = self._tasks[task_id]
            if st.status != STATUS_MASTERED and st.status != STATUS_FRAGILE:
                return
            old = st.status
            st.status = STATUS_FRONTIER
            st.regress_count += 1
            st.last_status_change_at = time.time()
            self._emit(CARTOGRAPHER_REGRESSED, {
                "task_id": task_id,
                "old_status": old,
                "rationale": rationale,
                "n": st.n,
                "mean": st.raw_mean(),
            })
            self._frontier_signature = self._frontier_signature_locked()

    def record_truth(self, task_id: str, true_mean: float) -> None:
        """Record the ground-truth long-run mean for coverage analysis."""
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            self._truth[task_id] = float(true_mean)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        *,
        policy: str = POLICY_LP,
        k: int = 1,
        budget: float | None = None,
        score_fn: Callable[["Cartographer", str], float] | None = None,
        rng_seed: int | None = None,
    ) -> Curriculum:
        """Return a curriculum: list of `CurriculumItem` to execute.

        - ``policy=POLICY_LP``: argsort frontier by ``LP_i × value_i``;
          falls back to UCB on cold start (no LP yet).
        - ``policy=POLICY_UCB``: argsort by ``U_i × value_i`` (Wilson upper).
        - ``policy=POLICY_INFOGAIN``: argsort by one-step variance
          reduction per unit cost.
        - ``policy=POLICY_THOMPSON``: stochastic — sample Beta posteriors
          and rank by ``value × draw``; respects ``rng_seed`` for reproducibility.
        - ``policy=POLICY_KNAPSACK``: cost-greedy submodular maximisation
          with an explicit ``budget``. ``k`` caps the output.
        - ``policy=POLICY_ROUND_ROBIN``: a low-variance fallback that
          rotates through frontier tasks deterministically.

        `score_fn(cart, task_id)` returns an additional additive score
        per task — the hook that lets a coordination engine fold in
        ExperimentDesigner BALD or PolicyImprover safety margins.
        """
        if k < 0:
            raise ValueError("k must be ≥ 0")
        if policy not in KNOWN_POLICIES:
            raise ValueError(f"unknown policy: {policy!r}")
        if policy == POLICY_KNAPSACK and budget is None:
            raise ValueError("POLICY_KNAPSACK requires budget")
        with self._lock:
            # Advance the curriculum so the frontier reflects every
            # observation since the last tick. Transitions emit events.
            self._tick_locked()
            frontier_ids = [
                tid for tid, st in self._tasks.items()
                if st.status in (STATUS_FRONTIER, STATUS_FRAGILE)
            ]
            self._recommended_count += 1
            cur_id = f"cart-{uuid.uuid4().hex[:12]}"
            if not frontier_ids:
                cur = Curriculum(
                    id=cur_id, policy=policy, k=k, budget=budget,
                    items=[], total_cost=0.0,
                    rationale="empty frontier",
                    issued_at=time.time(),
                )
                self._emit(CARTOGRAPHER_RECOMMENDED, {
                    "id": cur_id, "policy": policy, "k": k,
                    "n_items": 0, "rationale": "empty frontier",
                })
                return cur
            items = self._dispatch_policy_locked(
                policy=policy, k=k, budget=budget,
                frontier=frontier_ids, score_fn=score_fn,
                rng_seed=rng_seed,
            )
            total_cost = sum(it.cost for it in items)
            cur = Curriculum(
                id=cur_id, policy=policy, k=k, budget=budget,
                items=items, total_cost=total_cost,
                rationale=f"{policy} over |F|={len(frontier_ids)}",
                issued_at=time.time(),
            )
            self._emit(CARTOGRAPHER_RECOMMENDED, {
                "id": cur_id, "policy": policy,
                "items": [it.to_dict() for it in items],
                "frontier_size": len(frontier_ids),
                "total_cost": total_cost,
            })
            return cur

    def _dispatch_policy_locked(
        self,
        *,
        policy: str,
        k: int,
        budget: float | None,
        frontier: list[str],
        score_fn: Callable[["Cartographer", str], float] | None,
        rng_seed: int | None,
    ) -> list[CurriculumItem]:
        # Round-robin: deterministic, ignores scores.
        if policy == POLICY_ROUND_ROBIN:
            n = len(frontier)
            out: list[CurriculumItem] = []
            for i in range(k):
                tid = frontier[(self._round_robin_cursor + i) % n]
                st = self._tasks[tid]
                out.append(CurriculumItem(
                    task_id=tid, pulls=1,
                    score=0.0, cost=st.spec.cost,
                    rationale="round-robin",
                ))
            self._round_robin_cursor = (self._round_robin_cursor + k) % max(1, n)
            return out

        scored: list[tuple[str, float, float]] = []
        for tid in frontier:
            st = self._tasks[tid]
            comp = self._competence_locked(tid)
            base = self._policy_score_locked(
                policy=policy, tid=tid, comp=comp, rng_seed=rng_seed,
            )
            extra = 0.0
            if score_fn is not None:
                try:
                    extra = float(score_fn(self, tid))
                except Exception:
                    extra = 0.0
            total = base + extra
            scored.append((tid, total, st.spec.cost))

        if policy == POLICY_KNAPSACK:
            # cost-greedy submodular knapsack
            ids = submodular_knapsack(scored, budget=budget or 0.0)
            limited = ids[:k] if k > 0 else ids
            order = {tid: scr for (tid, scr, _) in scored}
            return [
                CurriculumItem(
                    task_id=tid, pulls=1,
                    score=order[tid],
                    cost=self._tasks[tid].spec.cost,
                    rationale="knapsack",
                )
                for tid in limited
            ]

        # default: top-k by score
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k] if k > 0 else scored
        return [
            CurriculumItem(
                task_id=tid, pulls=1, score=scr,
                cost=self._tasks[tid].spec.cost,
                rationale=policy,
            )
            for (tid, scr, _) in top
        ]

    def _policy_score_locked(
        self,
        *,
        policy: str,
        tid: str,
        comp: Competence,
        rng_seed: int | None,
    ) -> float:
        st = self._tasks[tid]
        v = st.spec.value
        n = max(1, comp.n)
        if policy == POLICY_LP:
            lp = comp.learning_progress
            if lp is None:
                # Cold-start: fall back to UCB-style optimism.
                return v * comp.upper
            # Reward absolute LP — both improvement and regression are
            # signal that the task is informative.
            return v * abs(lp) + 0.05 * v * comp.upper
        if policy == POLICY_UCB:
            return v * comp.upper
        if policy == POLICY_INFOGAIN:
            # one-step posterior variance reduction (Bernoulli case):
            #   Δσ² ≈ σ² · 1 / (n + α + β + 1) per pull
            return (comp.variance / (n + self.prior_alpha + self.prior_beta + 1.0)) * v / st.spec.cost
        if policy == POLICY_THOMPSON:
            # Beta sample, optionally seeded for tests.
            a = self.prior_alpha + st.s
            b = self.prior_beta + max(0, st.n - st.s)
            if rng_seed is None:
                draw = _beta_random(a, b)
            else:
                seed_str = f"{tid}|{rng_seed}|{self._recommended_count}"
                u = _hash_to_unit(seed_str)
                draw = _beta_inv(u, a, b)
            return v * draw
        if policy == POLICY_KNAPSACK:
            lp = comp.learning_progress
            lp_term = abs(lp) if lp is not None else (comp.upper - comp.lower)
            unc = max(0.0, (comp.upper - comp.lower) * 0.5)
            return v * (lp_term + 0.5 * unc)
        return 0.0

    # ------------------------------------------------------------------
    # Status engine
    # ------------------------------------------------------------------

    def tick(self) -> TickReport:
        """Idempotent: recompute every task's status, emit transitions.

        Mastery and regression transitions emit events here (not from
        `observe`), so a coordination engine that wants to react in
        bulk to a batch of observations can publish all of them
        through `observe_many` and call `tick()` once.

        ``recommend(...)`` calls `tick()` internally; explicit calls
        are only needed when the caller wants the transition bundle
        between observations.
        """
        with self._lock:
            return self._tick_locked()

    def _tick_locked(self) -> TickReport:
        advanced: list[TransitionEvent] = []
        regressed: list[TransitionEvent] = []
        # Iterate over a topological-ish order — mastery of a prereq
        # in this tick may immediately unlock a dependent. We retry
        # until no further changes occur, but cap at K passes.
        for _ in range(max(1, len(self._tasks))):
            changed = False
            for tid, st in list(self._tasks.items()):
                old = st.status
                new = self._compute_status_locked(tid)
                if old == new:
                    continue
                ts = time.time()
                evt = TransitionEvent(
                    task_id=tid, old_status=old, new_status=new,
                    at=ts, n=st.n, mean=st.raw_mean(),
                    rationale=self._transition_rationale_locked(tid, old, new),
                )
                st.status = new
                st.last_status_change_at = ts
                if new == STATUS_MASTERED:
                    st.mastered_at = ts
                    st.mastery_mean = st.raw_mean()
                    st.advance_count += 1
                    advanced.append(evt)
                    self._emit_advance(tid, evt)
                elif old == STATUS_MASTERED and new != STATUS_MASTERED:
                    st.regress_count += 1
                    regressed.append(evt)
                    self._emit(CARTOGRAPHER_REGRESSED, evt.to_dict())
                changed = True
            if not changed:
                break
        new_sig = self._frontier_signature_locked()
        if new_sig != self._frontier_signature:
            self._frontier_signature = new_sig
            self._emit(CARTOGRAPHER_FRONTIER_CHANGED, {
                "signature": list(new_sig),
            })
        return TickReport(
            advanced=advanced,
            regressed=regressed,
            frontier_size=sum(
                1 for st in self._tasks.values()
                if st.status in (STATUS_FRONTIER, STATUS_FRAGILE)
            ),
            mastered_size=sum(
                1 for st in self._tasks.values()
                if st.status == STATUS_MASTERED
            ),
            locked_size=sum(
                1 for st in self._tasks.values()
                if st.status == STATUS_LOCKED
            ),
            novice_size=sum(
                1 for st in self._tasks.values()
                if st.status == STATUS_NOVICE
            ),
            fragile_size=sum(
                1 for st in self._tasks.values()
                if st.status == STATUS_FRAGILE
            ),
            at=time.time(),
        )

    def _emit_advance(self, tid: str, evt: TransitionEvent) -> None:
        payload = evt.to_dict()
        # Attestation receipt for mastery: a content hash that an
        # auditor can independently reproduce. If a `record()` method
        # is exposed by the attestor, we use it.
        receipt_hash = ""
        if self._attestor is not None:
            try:
                core = json.dumps(payload, sort_keys=True, default=str)
                receipt_hash = hashlib.sha256(core.encode("utf-8")).hexdigest()
                rec = getattr(self._attestor, "record", None)
                if callable(rec):
                    try:
                        receipt = rec(kind=CARTOGRAPHER_ADVANCED, payload=payload)
                        if hasattr(receipt, "hash"):
                            receipt_hash = receipt.hash
                        elif isinstance(receipt, str):
                            receipt_hash = receipt
                    except Exception:
                        pass
            except Exception:
                receipt_hash = ""
        payload["receipt_hash"] = receipt_hash
        self._emit(CARTOGRAPHER_ADVANCED, payload)

    def _refresh_status_locked(self, task_id: str, *, emit: bool) -> None:
        st = self._tasks[task_id]
        new = self._compute_status_locked(task_id)
        if new != st.status:
            old = st.status
            st.status = new
            st.last_status_change_at = time.time()
            if new == STATUS_MASTERED:
                st.mastered_at = time.time()
                st.mastery_mean = st.raw_mean()
                st.advance_count += 1
            if emit:
                evt_payload = {
                    "task_id": task_id, "old_status": old,
                    "new_status": new, "n": st.n,
                    "mean": st.raw_mean(),
                }
                if new == STATUS_MASTERED:
                    self._emit(CARTOGRAPHER_ADVANCED, evt_payload)
                elif old == STATUS_MASTERED:
                    self._emit(CARTOGRAPHER_REGRESSED, evt_payload)

    def _compute_status_locked(self, task_id: str) -> str:
        st = self._tasks[task_id]
        # Locked iff any prereq is registered and not mastered.
        for prereq in st.spec.prereqs:
            prereq_st = self._tasks.get(prereq)
            if prereq_st is None:
                # Unknown prereq → permanently locked (defensive).
                return STATUS_LOCKED
            if prereq_st.status != STATUS_MASTERED:
                return STATUS_LOCKED
        if st.n == 0:
            return STATUS_FRONTIER if not st.spec.prereqs else STATUS_FRONTIER
        low, high = self._ci_locked(task_id)
        if low >= self.mastery_threshold:
            return STATUS_MASTERED
        # Fragile: was mastered, mean dropped meaningfully.
        if st.mastered_at is not None and st.mastery_mean is not None:
            cur_mean = st.raw_mean()
            if cur_mean < st.mastery_mean - self.regression_margin:
                return STATUS_FRAGILE
        if high < self.entry_threshold:
            return STATUS_NOVICE
        return STATUS_FRONTIER

    def _transition_rationale_locked(self, tid: str, old: str, new: str) -> str:
        st = self._tasks[tid]
        low, high = self._ci_locked(tid)
        return (
            f"{old}->{new} | n={st.n} | mean={st.raw_mean():.3f} | "
            f"CI=[{low:.3f}, {high:.3f}]"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def competence(self, task_id: str) -> Competence:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            return self._competence_locked(task_id)

    def _competence_locked(self, task_id: str) -> Competence:
        st = self._tasks[task_id]
        low, high = self._ci_locked(task_id)
        raw = st.raw_mean()
        post = st.posterior_mean(alpha=self.prior_alpha, beta=self.prior_beta)
        # Smoothed posterior mean: emphasise recent observations so
        # the recommendation engine reacts to streaks.
        if st.ring:
            recent_mean = _exp_smoothed(st.ring[-min(64, len(st.ring)):], alpha=self.smoothing_alpha)
            mean = 0.5 * (post + recent_mean)
        else:
            mean = post
        # Variance.
        if st.spec.reward_model == REWARD_BERNOULLI:
            var = max(_EPS, mean * (1.0 - mean))
        else:
            n = max(1, st.n - 1)
            mu = raw
            var = max(_EPS, (st.sum_sq - 2 * mu * st.s + n * mu * mu) / n)
        lp = learning_progress(
            st.ring,
            window_recent=self.window_recent,
            window_prior=self.window_prior,
        )
        return Competence(
            task_id=task_id,
            n=st.n,
            mean=mean,
            raw_mean=raw,
            lower=low,
            upper=high,
            variance=var,
            last_value=st.last_value,
            last_seen_at=st.last_seen_at,
            learning_progress=lp,
            status=st.status,
        )

    def _ci_locked(self, task_id: str) -> tuple[float, float]:
        st = self._tasks[task_id]
        if st.spec.reward_model == REWARD_BERNOULLI:
            # Successes for Wilson must be integer when possible.
            n_int = st.n
            s_int = int(round(st.s))
            s_int = max(0, min(n_int, s_int))
            return wilson_ci(s_int, n_int, z=self.z)
        # Gaussian: Student-t CI on the mean.
        n = st.n
        if n == 0:
            return 0.0, 1.0
        mu = st.raw_mean()
        if n == 1:
            return mu - st.spec.sigma2, mu + st.spec.sigma2
        # Sample variance.
        ss = st.sum_sq
        var = max(_EPS, (ss - 2 * mu * st.s + n * mu * mu) / (n - 1))
        sd = math.sqrt(var / n)
        # z-approx (n typically large in the regime we care about).
        margin = self.z * sd
        return max(0.0, mu - margin), min(1.0, mu + margin)

    def frontier(self) -> list[Competence]:
        with self._lock:
            return [
                self._competence_locked(tid)
                for tid, st in self._tasks.items()
                if st.status in (STATUS_FRONTIER, STATUS_FRAGILE)
            ]

    def mastered(self) -> list[Competence]:
        with self._lock:
            return [
                self._competence_locked(tid)
                for tid, st in self._tasks.items()
                if st.status == STATUS_MASTERED
            ]

    def locked(self) -> list[Competence]:
        with self._lock:
            return [
                self._competence_locked(tid)
                for tid, st in self._tasks.items()
                if st.status == STATUS_LOCKED
            ]

    def novice(self) -> list[Competence]:
        with self._lock:
            return [
                self._competence_locked(tid)
                for tid, st in self._tasks.items()
                if st.status == STATUS_NOVICE
            ]

    def status(self, task_id: str) -> str:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            return self._tasks[task_id].status

    def spec(self, task_id: str) -> TaskSpec:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            return self._tasks[task_id].spec

    def task_ids(self) -> list[str]:
        with self._lock:
            return list(self._tasks)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        with self._lock:
            return task_id in self._tasks

    def _frontier_signature_locked(self) -> tuple[str, ...]:
        return tuple(sorted(
            tid for tid, st in self._tasks.items()
            if st.status in (STATUS_FRONTIER, STATUS_FRAGILE)
        ))

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------

    def coverage_report(self) -> CoverageReport:
        """Calibration of the cartographer's own predictions.

        Compares logged predictions (``μ̂`` at observation time) to
        truth values supplied by `record_truth` for the same task. The
        report does *not* require every task to have a truth — tasks
        without recorded truth simply do not contribute to the
        error aggregates.
        """
        with self._lock:
            by_status: dict[str, int] = {s: 0 for s in KNOWN_STATUSES}
            n_by_status: dict[str, list[int]] = {s: [] for s in KNOWN_STATUSES}
            lp_by_status: dict[str, list[float]] = {s: [] for s in KNOWN_STATUSES}
            for tid, st in self._tasks.items():
                by_status[st.status] = by_status.get(st.status, 0) + 1
                n_by_status[st.status].append(st.n)
                lp = learning_progress(
                    st.ring,
                    window_recent=self.window_recent,
                    window_prior=self.window_prior,
                )
                if lp is not None:
                    lp_by_status[st.status].append(lp)
            err_signed = 0.0
            err_abs = 0.0
            brier = 0.0
            kept = 0
            for tid, mu_hat, _ts in self._prediction_log:
                truth = self._truth.get(tid)
                if truth is None:
                    continue
                err_signed += (mu_hat - truth)
                err_abs += abs(mu_hat - truth)
                brier += (mu_hat - truth) ** 2
                kept += 1
            n_pred = kept
            return CoverageReport(
                n_tasks=len(self._tasks),
                n_predictions=n_pred,
                mean_signed_error=(err_signed / n_pred) if n_pred > 0 else 0.0,
                mean_abs_error=(err_abs / n_pred) if n_pred > 0 else 0.0,
                brier_score=(brier / n_pred) if n_pred > 0 else 0.0,
                n_by_status=by_status,
                mean_n_by_status={
                    s: (statistics.fmean(v) if v else 0.0)
                    for s, v in n_by_status.items()
                },
                mean_lp_by_status={
                    s: (statistics.fmean(v) if v else 0.0)
                    for s, v in lp_by_status.items()
                },
            )

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": {
                    "entry_threshold": self.entry_threshold,
                    "mastery_threshold": self.mastery_threshold,
                    "delta": self.delta,
                    "z": self.z,
                    "prior_alpha": self.prior_alpha,
                    "prior_beta": self.prior_beta,
                    "window_recent": self.window_recent,
                    "window_prior": self.window_prior,
                    "smoothing_alpha": self.smoothing_alpha,
                    "regression_margin": self.regression_margin,
                },
                "tasks": {tid: st.to_dict() for tid, st in self._tasks.items()},
                "truths": dict(self._truth),
                "round_robin_cursor": self._round_robin_cursor,
                "recommended_count": self._recommended_count,
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Replace state from a `snapshot()` dict. Idempotent."""
        cfg = snapshot.get("config", {})
        tasks = snapshot.get("tasks", {})
        truths = snapshot.get("truths", {})
        with self._lock:
            for k, v in cfg.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._tasks.clear()
            for tid, t in tasks.items():
                spec_d = t["spec"]
                spec = TaskSpec(
                    id=spec_d["id"],
                    value=spec_d.get("value", 1.0),
                    cost=spec_d.get("cost", 1.0),
                    prereqs=tuple(spec_d.get("prereqs", ())),
                    reward_model=spec_d.get("reward_model", REWARD_BERNOULLI),
                    sigma2=spec_d.get("sigma2", _DEFAULT_GAUSSIAN_VAR),
                    tags=tuple(spec_d.get("tags", ())),
                )
                st = _TaskState(
                    spec=spec,
                    n=int(t.get("n", 0)),
                    s=float(t.get("s", 0.0)),
                    sum_sq=float(t.get("sum_sq", 0.0)),
                    last_value=t.get("last_value"),
                    last_seen_at=float(t.get("last_seen_at", 0.0)),
                    first_seen_at=float(t.get("first_seen_at", 0.0)),
                    ring=list(t.get("ring", [])),
                    status=t.get("status", STATUS_LOCKED),
                    last_status_change_at=float(t.get("last_status_change_at", 0.0)),
                    mastered_at=t.get("mastered_at"),
                    mastery_mean=t.get("mastery_mean"),
                    advance_count=int(t.get("advance_count", 0)),
                    regress_count=int(t.get("regress_count", 0)),
                )
                self._tasks[tid] = st
            self._truth = {k: float(v) for k, v in truths.items()}
            self._round_robin_cursor = int(snapshot.get("round_robin_cursor", 0))
            self._recommended_count = int(snapshot.get("recommended_count", 0))
            # Validate DAG and recompute statuses cleanly.
            self._validate_dag_locked()
            for tid in list(self._tasks):
                self._refresh_status_locked(tid, emit=False)
            self._frontier_signature = self._frontier_signature_locked()

    def clear(self, task_id: str | None = None) -> None:
        """Reset a task's observed history, or wipe the curriculum entirely."""
        with self._lock:
            if task_id is None:
                self._tasks.clear()
                self._truth.clear()
                self._prediction_log.clear()
                self._frontier_signature = ()
                self._round_robin_cursor = 0
                self._emit(CARTOGRAPHER_CLEARED, {"scope": "all"})
                return
            if task_id not in self._tasks:
                raise KeyError(f"unknown task: {task_id!r}")
            st = self._tasks[task_id]
            st.n = 0
            st.s = 0.0
            st.sum_sq = 0.0
            st.last_value = None
            st.last_seen_at = 0.0
            st.ring = []
            st.status = STATUS_FRONTIER
            st.mastered_at = None
            st.mastery_mean = None
            self._refresh_status_locked(task_id, emit=False)
            self._emit(CARTOGRAPHER_CLEARED, {"task_id": task_id})

    # ------------------------------------------------------------------
    # Eventing
    # ------------------------------------------------------------------

    def _emit(self, kind: str, data: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:
            # Telemetry must never crash the runtime.
            pass


# =====================================================================
# Small helpers for Thompson sampling
# =====================================================================


def _hash_to_unit(s: str) -> float:
    """Deterministic [0,1) draw from a string seed (stdlib-only)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return (n / 2**64) * (1.0 - 1e-15) + 1e-16


def _beta_random(a: float, b: float) -> float:
    """Beta(a, b) via two Gamma draws using `random.gammavariate`.

    Stdlib `random` is sufficient for Thompson; tests that need
    determinism use the `rng_seed=` path that avoids it.
    """
    import random
    x = random.gammavariate(a, 1.0) if a > 0 else 0.0
    y = random.gammavariate(b, 1.0) if b > 0 else 0.0
    if x + y <= 0.0:
        return 0.5
    return x / (x + y)


# =====================================================================
# Convenience free functions (Strategist composition)
# =====================================================================


def frontier_subset(
    means: Mapping[str, float],
    counts: Mapping[str, int],
    *,
    entry: float = _DEFAULT_ENTRY_THRESHOLD,
    mastery: float = _DEFAULT_MASTERY_THRESHOLD,
    z: float = _DEFAULT_Z,
) -> list[str]:
    """Identify the frontier subset given empirical means and counts.

    A one-shot convenience for coordinators that already have
    statistics and just want the ZPD set. Equivalent to the Wilson
    test inside `Cartographer` but does not require constructing a
    full instance.
    """
    out: list[str] = []
    for tid, mean in means.items():
        n = max(0, int(counts.get(tid, 0)))
        s = int(round(mean * n)) if n > 0 else 0
        s = max(0, min(n, s))
        low, high = wilson_ci(s, n, z=z)
        if n == 0 or (low < mastery and high >= entry):
            out.append(tid)
    return sorted(out)


def expected_pulls_to_master(
    successes: int,
    n: int,
    *,
    mastery: float = _DEFAULT_MASTERY_THRESHOLD,
    z: float = _DEFAULT_Z,
    max_pulls: int = 10_000,
) -> int:
    """Asymptotic estimate of remaining pulls to reach Wilson-LCB ≥ mastery.

    Inverts the Wilson lower bound at the current empirical mean to
    estimate the smallest ``n′`` such that ``L(p̂, n′) ≥ mastery``.
    Returns ``max_pulls`` if mastery is unreachable at the current
    empirical mean (i.e. ``p̂ ≤ mastery``).

    Useful for **budget-aware coordinators** that need to estimate the
    cost of advancing a task before committing samples.
    """
    if n <= 0:
        return max_pulls
    p = successes / n
    if p <= mastery + _EPS:
        return max_pulls
    # Bisection on n′.
    lo, hi = n, max_pulls
    target_s = round(p * lo)
    target_lcb, _ = wilson_ci(int(target_s), lo, z=z)
    if target_lcb >= mastery:
        return 0
    while lo < hi:
        mid = (lo + hi) // 2
        s_mid = int(round(p * mid))
        low_mid, _ = wilson_ci(s_mid, mid, z=z)
        if low_mid >= mastery:
            hi = mid
        else:
            lo = mid + 1
    return max(0, lo - n)


def lp_signal(
    ring: Sequence[float],
    *,
    window_recent: int = _DEFAULT_WINDOW_RECENT,
    window_prior: int = _DEFAULT_WINDOW_PRIOR,
) -> float | None:
    """Public alias for `learning_progress`."""
    return learning_progress(
        ring, window_recent=window_recent, window_prior=window_prior,
    )


# =====================================================================
# Public re-exports
# =====================================================================


__all__ = [
    "CARTOGRAPHER_STARTED",
    "CARTOGRAPHER_OBSERVED",
    "CARTOGRAPHER_RECOMMENDED",
    "CARTOGRAPHER_ADVANCED",
    "CARTOGRAPHER_REGRESSED",
    "CARTOGRAPHER_FRONTIER_CHANGED",
    "CARTOGRAPHER_CLEARED",
    "CARTOGRAPHER_REPORT",
    "POLICY_LP",
    "POLICY_UCB",
    "POLICY_INFOGAIN",
    "POLICY_THOMPSON",
    "POLICY_KNAPSACK",
    "POLICY_ROUND_ROBIN",
    "KNOWN_POLICIES",
    "STATUS_NOVICE",
    "STATUS_FRONTIER",
    "STATUS_MASTERED",
    "STATUS_LOCKED",
    "STATUS_FRAGILE",
    "KNOWN_STATUSES",
    "REWARD_BERNOULLI",
    "REWARD_GAUSSIAN",
    "KNOWN_REWARDS",
    "Cartographer",
    "Competence",
    "TaskSpec",
    "CurriculumItem",
    "Curriculum",
    "TransitionEvent",
    "TickReport",
    "CoverageReport",
    "wilson_ci",
    "clopper_pearson_ci",
    "beta_lcb",
    "beta_ucb",
    "learning_progress",
    "lp_signal",
    "submodular_knapsack",
    "frontier_subset",
    "expected_pulls_to_master",
]
