r"""Budgeter — compute-optimal test-time inference allocation as a runtime primitive.

Where :mod:`agi.scaler` answers *"how much compute should I spend on
training?"*, ``Budgeter`` answers the dual question every coordination
engine has to settle at inference time:

    Given a task of estimated difficulty :math:`d`, a compute budget
    :math:`C`, and a roster of inference strategies — best-of-N sampling
    (Brown et al. 2024 *Large Language Monkeys*), self-consistency
    majority vote (Wang et al. 2022), verifier-guided reranking
    (Cobbe et al. 2021 *GSM-Verifier*), beam-of-thought / process-reward
    rerank (Lightman et al. 2024 *Let's Verify Step-by-Step*), tree- /
    MCTS-of-thoughts (Yao et al. 2023), and sequential "think longer"
    chain-of-thought (OpenAI o1 system card 2024; DeepSeek-R1 2025) —
    **what mix of strategies maximises pass rate at budget** :math:`C`?

This is the central question of the test-time-compute scaling
paradigm.  Snell, Lee, Xu, Kumar (2024) "Scaling LLM Test-Time
Compute Optimally Can Be More Effective Than Scaling Model
Parameters" shows that an *allocation-aware* test-time policy beats
naive best-of-N by up to **4× FLOP efficiency** on MATH and that the
optimal mix flips between parallel and sequential as difficulty
rises.  Without an allocation policy, every additional inference
dollar buys a stochastic, often-saturated improvement.  With one,
each dollar buys a predictable ΔP(pass), which is exactly the curve
investors price an inference business on.

``Budgeter`` is the **inference-resource primitive**.  Compose with
``Scaler`` (training compute) and ``Anticipator`` (sleep-time compute)
to span the three axes of compute spend, and ``Speculator`` for the
decoding-pass axis underneath them all.

Mathematical and algorithmic roots
----------------------------------

* **Brown, B., Juravsky, J., Ehrlich, R., et al. (2024) — "Large
  Language Monkeys: Scaling Inference Compute with Repeated
  Sampling" (arXiv:2407.21787).**  Empirically, pass@k for an LLM on
  fixed tasks follows the Bernoulli-superposition

  .. math::

     p_k = 1 - (1 - p_1)^{k}

  on the easy side and the Snell exponential-saturation form

  .. math::

     p_k = p_{\\infty} - (p_{\\infty} - p_1) \\exp(-k / \\tau)

  near the ceiling.  Implemented as :data:`STRAT_PARALLEL` with the
  fitted pair :math:`(p_1, p_\\infty)` and concentration :math:`\\tau`.

* **Wang, X., Wei, J., Schuurmans, D., et al. (2022) — "Self-
  Consistency Improves Chain-of-Thought Reasoning in Language Models"
  (arXiv:2203.11171).**  Majority-vote over :math:`k` independent
  samples; the pass curve is a *thresholded* binomial above the
  modal-correct rate.  Implemented as :data:`STRAT_MAJORITY` with the
  closed-form upper envelope of the de Moivre–Laplace approximation.

* **Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021) — "Training
  Verifiers to Solve Math Word Problems" (arXiv:2110.14168).**
  Verifier-guided rerank where the verifier has informativeness
  :math:`r \\in [0, 1]` (1.0 = oracle).  The pass curve at :math:`k`
  draws is the order-statistic max

  .. math::

     p_k = p_{\\infty} - (p_{\\infty} - p_1) (1 - r)^{k - 1}

  which collapses to parallel when :math:`r = 0` and to oracle-best
  when :math:`r = 1`.  Implemented as :data:`STRAT_VERIFIER`.

* **Lightman, H., Kosaraju, V., Burda, Y., et al. (2024) — "Let's
  Verify Step-by-Step" (arXiv:2305.20050).**  Process-reward beam
  search: the verifier scores partial traces so wasted rollouts get
  pruned early.  The pass curve at a beam of width :math:`w` and depth
  :math:`L` follows a *cumulative-survival* form that approaches the
  ORM ceiling faster per FLOP than naive rerank.  Implemented as
  :data:`STRAT_BEAM`.

* **Yao, S., Yu, D., Zhao, J., et al. (2023) — "Tree of Thoughts"
  (arXiv:2305.10601) and Snell et al. 2024 §4 MCTS-of-thoughts.**
  Tree search over partial CoT.  Pass curve modelled as a saturating
  power law on the number of rollouts.  Implemented as
  :data:`STRAT_TREE`.

* **OpenAI (2024) "Learning to Reason with LLMs" / o1 system card;
  DeepSeek-AI (2025) "DeepSeek-R1: Incentivizing Reasoning Capability
  in LLMs via Reinforcement Learning" (arXiv:2501.12948).**  Sequential
  "think longer" CoT.  Pass curve is a saturating exponential in
  thinking-token budget :math:`t`:

  .. math::

     p(t) = p_{\\infty} - (p_{\\infty} - p_1) \\exp(-t / \\tau_t).

  Implemented as :data:`STRAT_SEQUENTIAL`.

* **Snell, C., Lee, J., Xu, K., Kumar, A. (2024) — "Scaling LLM Test-
  Time Compute Optimally Can Be More Effective Than Scaling Model
  Parameters" (arXiv:2408.03314).**  The *compute-optimal* allocation
  is found by Lagrangian water-filling: at the optimum, the marginal
  ΔP/Δcost is equalised across all strategies that receive non-zero
  budget.  Implemented in :meth:`Budgeter.allocate`.

* **Karmarkar, N. (1984) / Dantzig, G. (1947) — convex-concave knapsack.**
  Since each strategy's pass curve is monotone concave in compute,
  the constrained-budget problem
  :math:`\\max \\sum p_i(c_i) \\text{ s.t. } \\sum c_i \\le C`
  is solved by water-filling on the inverse marginal-utility
  curves :math:`(p_i')^{-1}(\\lambda)`.

* **Bickel, P.J., Doksum, K.A. (1981) — order statistics for pass@k.**
  When pass labels are i.i.d. Bernoulli with rate :math:`p_1` per
  sample, the empirical estimator of :math:`p_k = 1 - (1 - p_1)^k` is
  the standard unbiased pass@k estimator (Chen et al. 2021,
  arXiv:2107.03374, eq. 1):

  .. math::

     \\widehat{p_k} = 1 - \\binom{n - c}{k} / \\binom{n}{k}

  for :math:`n` draws and :math:`c` successes.  Implemented in
  :func:`unbiased_pass_at_k`.

* **Maurer, A., Pontil, M. (2009) — "Empirical Bernstein Bounds and
  Sample-Variance Penalization"** and **Hoeffding (1963).**  The PAC
  certificate on allocation regret uses both the sample-variance-
  aware Bernstein LCB and the variance-free Hoeffding LCB on held-out
  pass rate at the chosen allocation.

* **Bauer, F., Pereyra, V. (1973) / Levenberg-Marquardt (1944, 1963).**
  Per-strategy curves are fit by Levenberg-Marquardt in log-curve
  space with finite-difference Jacobians (same machinery as
  :mod:`agi.scaler`) — Pure stdlib, no NumPy.

* **Efron, B. (1979).**  Bootstrap-percentile CIs on the per-strategy
  pass curve and on the chosen-allocation predicted pass rate.

Composes with
-------------

* :mod:`agi.scaler` — training-time compute scaling.  Together they
  answer "how much compute, where" across train/test.
* :mod:`agi.anticipator` — sleep-time compute.  Budgeter
  surfaces the *online* axis; Anticipator surfaces the *offline* axis.
* :mod:`agi.speculator` — decoding-pass compute.  Speculator buys
  cheaper tokens; Budgeter decides how many to draw.
* :mod:`agi.stepwiser` / :mod:`agi.verifier` — supply the
  verifier informativeness :math:`r` for :data:`STRAT_VERIFIER` and
  :data:`STRAT_BEAM`.
* :mod:`agi.searcher` — implements :data:`STRAT_TREE` / MCTS-of-
  thoughts as a concrete search algorithm.
* :mod:`agi.economist` / :mod:`agi.market` — turn ΔP into dollars per
  passed task, the headline unit-economic metric.
* :mod:`agi.strategist` / :mod:`agi.portfolio` — risk-adjusted
  selection across strategies given uncertainty bands.
* :mod:`agi.conformal` / :mod:`agi.calibration` — held-out coverage
  of the predicted pass rate.
* :mod:`agi.attest` / :mod:`agi.governance` — replay-verifiable
  fingerprint chain over every observation, fit, and allocation.

What this primitive ships
-------------------------

* :class:`Observation` — one ``(strategy, difficulty, compute_units,
  trials, successes)`` row.  ``compute_units`` is dimensionless
  (samples for parallel, thinking-tokens for sequential, etc. — the
  primitive is unit-agnostic; the *unit cost* in the strategy spec
  bridges to a common budget currency).
* :class:`StrategySpec` — the cost-per-unit and curve family.
* :class:`BudgeterConfig` — seed, bootstrap-B, holdout, ridge, tol.
* :class:`StrategyFit` — fitted parameters per strategy with stderr.
* :class:`Allocation` — chosen split + predicted pass + CI.
* :class:`ParetoPoint` — one point on the cost/pass frontier.
* :class:`BudgeterCertificate` — PAC LCB on regret vs oracle, in-
  range diagnostics, fingerprint chain.
* :class:`BudgeterReport` — bundle for the coordination engine.
* :class:`Budgeter` — the primitive.  ``observe → fit → allocate →
  pareto → certify → report``.

Pure stdlib.  Deterministic given seed.  Thread-safe.
``json.dumps(report.to_dict())`` round-trips.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus

# ---------------------------------------------------------------------------
# Strategy taxonomy
# ---------------------------------------------------------------------------

#: Independent best-of-N sampling with oracle verifier (or final-answer
#: comparison against a held-out reference).
STRAT_PARALLEL = "parallel"

#: Majority-vote / self-consistency over ``k`` independent samples.
STRAT_MAJORITY = "majority"

#: Best-of-N with a *learned* outcome reward model (ORM verifier).
STRAT_VERIFIER = "verifier"

#: Process-reward-guided beam / lookahead search.
STRAT_BEAM = "beam"

#: Tree- / MCTS-of-thoughts over partial reasoning paths.
STRAT_TREE = "tree"

#: Sequential "think longer" chain-of-thought; budget is in thinking-
#: tokens (or thinking-time), not in number of samples.
STRAT_SEQUENTIAL = "sequential"

KNOWN_STRATEGIES: tuple[str, ...] = (
    STRAT_PARALLEL,
    STRAT_MAJORITY,
    STRAT_VERIFIER,
    STRAT_BEAM,
    STRAT_TREE,
    STRAT_SEQUENTIAL,
)


# ---------------------------------------------------------------------------
# Event names emitted on the runtime EventBus
# ---------------------------------------------------------------------------

BUDGETER_STARTED = "budgeter.started"
BUDGETER_OBSERVED = "budgeter.observed"
BUDGETER_FIT = "budgeter.fit"
BUDGETER_EXTRAPOLATED = "budgeter.extrapolated"
BUDGETER_ALLOCATED = "budgeter.allocated"
BUDGETER_PARETO = "budgeter.pareto"
BUDGETER_CERTIFIED = "budgeter.certified"
BUDGETER_REPORTED = "budgeter.reported"
BUDGETER_RESET = "budgeter.reset"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgeterError(ValueError):
    """Base class for Budgeter-specific errors."""


class InvalidConfig(BudgeterError):
    """The :class:`BudgeterConfig` or a :class:`StrategySpec` is inconsistent."""


class InvalidObservation(BudgeterError):
    """The observation row violates a runtime invariant."""


class UnknownStrategy(BudgeterError):
    """A strategy name was not recognised."""


class NotFitted(BudgeterError):
    """Tried to allocate / extrapolate / certify before fitting."""


class FitFailed(BudgeterError):
    """Per-strategy LM fit did not converge."""


class InfeasibleBudget(BudgeterError):
    """The requested budget cannot fit even the smallest viable allocation."""


# ---------------------------------------------------------------------------
# Configuration and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySpec:
    """Per-strategy compute-curve specification.

    The pass curve is a function of dimensionless ``compute_units``;
    ``unit_cost`` bridges those to a common budget currency (FLOPs,
    dollars, wall-clock-tokens — the runtime never inspects it, only
    sums it).

    Attributes:
        name: one of :data:`KNOWN_STRATEGIES`.
        unit_cost: cost of one compute unit, in the budget currency.
        verifier_info: informativeness :math:`r \\in [0, 1]` of the
            verifier this strategy depends on.  ``0`` = no verifier,
            ``1`` = oracle.  Only :data:`STRAT_VERIFIER` and
            :data:`STRAT_BEAM` consume it; others ignore.
        min_units: floor on units; pass curve clipped to this domain.
        max_units: ceiling on units; allocator never exceeds it (so a
            saturating strategy doesn't soak the entire budget once
            its marginal gain is ~zero).
        metadata: free-form coordination payload.
    """
    name: str
    unit_cost: float = 1.0
    verifier_info: float = 0.0
    min_units: float = 1.0
    max_units: float = 1024.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in KNOWN_STRATEGIES:
            raise UnknownStrategy(
                f"strategy {self.name!r} not in {KNOWN_STRATEGIES!r}"
            )
        if not (self.unit_cost > 0.0 and math.isfinite(self.unit_cost)):
            raise InvalidConfig("unit_cost must be a positive finite float")
        if not (0.0 <= self.verifier_info <= 1.0):
            raise InvalidConfig("verifier_info must lie in [0, 1]")
        if not (0.0 < self.min_units <= self.max_units):
            raise InvalidConfig("require 0 < min_units <= max_units")
        if not math.isfinite(self.max_units):
            raise InvalidConfig("max_units must be finite")


@dataclass(frozen=True)
class BudgeterConfig:
    """Configuration for a :class:`Budgeter` instance.

    Attributes:
        seed: PRNG seed for bootstrap and tie-breaking.
        bootstrap_b: number of bootstrap resamples (``0`` disables CI).
        confidence: bootstrap-percentile coverage.
        holdout_fraction: fraction of observations held out per strategy
            for the PAC certificate.  ``0.0`` disables.
        max_iters: Levenberg-Marquardt iteration cap.
        tol: convergence tolerance.
        ridge: log-parameter L2 regulariser (numeric stability).
        difficulty_kernel_bandwidth: when no explicit difficulty bucket
            is set, rows within this distance are pooled into the same
            curve.  ``0`` disables difficulty modelling (single curve).
        allocator_grid: log-spaced number of grid points used by the
            water-filling allocator.
        oracle_ref_units: number of units the held-out oracle uses to
            estimate the per-strategy ceiling :math:`p_\\infty`.
    """
    seed: int = 0
    bootstrap_b: int = 100
    confidence: float = 0.95
    holdout_fraction: float = 0.2
    max_iters: int = 200
    tol: float = 1e-8
    ridge: float = 1e-6
    difficulty_kernel_bandwidth: float = 0.0
    allocator_grid: int = 80
    oracle_ref_units: float = 256.0

    def __post_init__(self) -> None:
        if self.bootstrap_b < 0:
            raise InvalidConfig("bootstrap_b must be >= 0")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must be in (0, 1)")
        if not (0.0 <= self.holdout_fraction < 1.0):
            raise InvalidConfig("holdout_fraction must be in [0, 1)")
        if self.max_iters <= 0:
            raise InvalidConfig("max_iters must be > 0")
        if self.tol <= 0.0:
            raise InvalidConfig("tol must be > 0")
        if self.ridge < 0.0:
            raise InvalidConfig("ridge must be >= 0")
        if self.difficulty_kernel_bandwidth < 0.0:
            raise InvalidConfig("difficulty_kernel_bandwidth must be >= 0")
        if self.allocator_grid < 8:
            raise InvalidConfig("allocator_grid must be >= 8")
        if not (self.oracle_ref_units > 0):
            raise InvalidConfig("oracle_ref_units must be > 0")


@dataclass(frozen=True)
class Observation:
    """One per-strategy result row.

    Attributes:
        strategy: one of :data:`KNOWN_STRATEGIES`.
        difficulty: scalar difficulty estimate (e.g. log-probability of
            single-sample failure under a calibrated baseline, or a
            curated difficulty bucket index in :math:`[0, 1]`).  Pooled
            across rows within the kernel bandwidth.
        compute_units: dimensionless units fed to the strategy
            (samples for parallel, tokens for sequential, rollouts for
            tree).  Must be ``> 0``.
        trials: number of independent task instances at this
            ``(strategy, difficulty, compute_units)`` cell.  ``>= 1``.
        successes: number of those trials that passed.  ``0 <=
            successes <= trials``.
        weight: row weight (e.g. importance sample weight).  Default 1.
        metadata: free-form coordination payload.
    """
    strategy: str
    difficulty: float
    compute_units: float
    trials: int
    successes: int
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in KNOWN_STRATEGIES:
            raise UnknownStrategy(f"strategy {self.strategy!r} unknown")
        if not (math.isfinite(self.difficulty)):
            raise InvalidObservation("difficulty must be finite")
        if not (self.compute_units > 0.0 and math.isfinite(self.compute_units)):
            raise InvalidObservation("compute_units must be positive finite")
        if self.trials <= 0:
            raise InvalidObservation("trials must be > 0")
        if not (0 <= self.successes <= self.trials):
            raise InvalidObservation("0 <= successes <= trials")
        if not (self.weight > 0.0 and math.isfinite(self.weight)):
            raise InvalidObservation("weight must be positive finite")


@dataclass(frozen=True)
class StrategyFit:
    """Fitted parameters for a single strategy.

    Curve families:

    * :data:`STRAT_PARALLEL` — :math:`p(k) = p_\\infty - (p_\\infty - p_1)
      \\exp(-k / \\tau)`.  Three params: ``p1, p_inf, tau``.
    * :data:`STRAT_MAJORITY` — :math:`p(k) = \\Phi((k - k_0) / s)`
      saturating to ``p_inf``; with floor ``p1``.  Four params:
      ``p1, p_inf, k0, s``.
    * :data:`STRAT_VERIFIER` — :math:`p(k) = p_\\infty - (p_\\infty -
      p_1) (1 - r)^{k - 1}`.  Three params: ``p1, p_inf, r``.
    * :data:`STRAT_BEAM` — :math:`p(w) = p_\\infty - (p_\\infty -
      p_1) (1 - r)^{w}` (Lightman ORM beam).  Three params:
      ``p1, p_inf, r``.
    * :data:`STRAT_TREE` — :math:`p(n) = p_\\infty - (p_\\infty - p_1)
      / (1 + n / n_0)^{\\gamma}` (saturating power-law in rollouts).
      Four params: ``p1, p_inf, n0, gamma``.
    * :data:`STRAT_SEQUENTIAL` — :math:`p(t) = p_\\infty - (p_\\infty -
      p_1) \\exp(-t / \\tau_t)`.  Three params: ``p1, p_inf, tau_t``.
    """
    strategy: str
    params: dict[str, float]
    n_in_sample: int
    n_held_out: int
    log_loss_in_sample: float
    log_loss_held_out: float | None
    iters: int
    converged: bool
    parameter_stderr: dict[str, float]


    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "params": dict(self.params),
            "n_in_sample": self.n_in_sample,
            "n_held_out": self.n_held_out,
            "log_loss_in_sample": self.log_loss_in_sample,
            "log_loss_held_out": self.log_loss_held_out,
            "iters": self.iters,
            "converged": self.converged,
            "parameter_stderr": dict(self.parameter_stderr),
        }


@dataclass(frozen=True)
class Allocation:
    """A single allocation across strategies.

    Attributes:
        budget: total compute spend in the budget currency.
        difficulty: difficulty the allocation is optimised for.
        per_strategy_units: ``{strategy: compute_units}`` to spend.
        per_strategy_cost: ``{strategy: cost}`` (units × unit_cost).
        per_strategy_pass: ``{strategy: predicted p(pass)}`` if the
            allocation were *exclusively* run on that strategy.
        predicted_pass: combined predicted pass rate under independent-
            strategy assumption (``1 - prod(1 - p_i)``).  Saturated at
            ``p_infinity`` of the best strategy.
        predicted_lower: bootstrap-percentile CI lower bound.
        predicted_upper: bootstrap-percentile CI upper bound.
        confidence: nominal coverage.
        active_strategies: which strategies received non-zero budget.
        spent: sum of ``per_strategy_cost``.  ``<= budget``.
    """
    budget: float
    difficulty: float
    per_strategy_units: dict[str, float]
    per_strategy_cost: dict[str, float]
    per_strategy_pass: dict[str, float]
    predicted_pass: float
    predicted_lower: float
    predicted_upper: float
    confidence: float
    active_strategies: tuple[str, ...]
    spent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "difficulty": self.difficulty,
            "per_strategy_units": dict(self.per_strategy_units),
            "per_strategy_cost": dict(self.per_strategy_cost),
            "per_strategy_pass": dict(self.per_strategy_pass),
            "predicted_pass": self.predicted_pass,
            "predicted_lower": self.predicted_lower,
            "predicted_upper": self.predicted_upper,
            "confidence": self.confidence,
            "active_strategies": list(self.active_strategies),
            "spent": self.spent,
        }


@dataclass(frozen=True)
class ParetoPoint:
    """One ``(cost, predicted_pass)`` point on the frontier."""
    budget: float
    spent: float
    predicted_pass: float
    predicted_lower: float
    predicted_upper: float
    active_strategies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "spent": self.spent,
            "predicted_pass": self.predicted_pass,
            "predicted_lower": self.predicted_lower,
            "predicted_upper": self.predicted_upper,
            "active_strategies": list(self.active_strategies),
        }


@dataclass(frozen=True)
class BudgeterCertificate:
    """Replay-verifiable certificate over the latest allocation.

    Carries the Hoeffding / empirical-Bernstein LCB on the held-out
    pass rate at the chosen allocation, the regret upper bound vs
    the per-budget oracle (best single strategy), in-range
    diagnostics, and the SHA-256 fingerprint chain over the entire
    observe / fit / allocate trace.
    """
    confidence: float
    n_held_out: int
    pass_held_out: float | None
    pass_lcb_hoeffding: float | None
    pass_lcb_bernstein: float | None
    regret_ucb: float | None
    oracle_strategy: str | None
    oracle_pass: float | None
    in_range_per_strategy: dict[str, tuple[float, float]]
    extrapolation_factor: float
    fingerprint_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "n_held_out": self.n_held_out,
            "pass_held_out": self.pass_held_out,
            "pass_lcb_hoeffding": self.pass_lcb_hoeffding,
            "pass_lcb_bernstein": self.pass_lcb_bernstein,
            "regret_ucb": self.regret_ucb,
            "oracle_strategy": self.oracle_strategy,
            "oracle_pass": self.oracle_pass,
            "in_range_per_strategy": {
                s: list(rg) for s, rg in self.in_range_per_strategy.items()
            },
            "extrapolation_factor": self.extrapolation_factor,
            "fingerprint_hash": self.fingerprint_hash,
        }


@dataclass(frozen=True)
class BudgeterReport:
    """A bundle of everything the coordinator needs to act on."""
    config: dict[str, Any]
    strategies: dict[str, Any]
    observations: int
    fits: dict[str, Any]
    allocation: dict[str, Any] | None
    certificate: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "strategies": dict(self.strategies),
            "observations": self.observations,
            "fits": dict(self.fits),
            "allocation": dict(self.allocation) if self.allocation else None,
            "certificate": dict(self.certificate) if self.certificate else None,
        }


# ---------------------------------------------------------------------------
# Pass-curve families
# ---------------------------------------------------------------------------


def _logistic(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _erf_approx(x: float) -> float:
    """Abramowitz-Stegun 7.1.26 approximation; pure stdlib."""
    sign = -1.0 if x < 0.0 else 1.0
    ax = abs(x)
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-ax * ax)
    return sign * y


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2.0)))


def _saturating_exp(units: float, p1: float, p_inf: float,
                    tau: float) -> float:
    """``p_inf - (p_inf - p1) * exp(-units / tau)``.

    Clipped so the result remains in ``[0, 1]``.  Defines
    parallel / sequential / verifier-equivalent strategies.
    """
    if tau <= 0.0:
        tau = 1e-9
    val = p_inf - (p_inf - p1) * math.exp(-units / tau)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _power_saturating(units: float, p1: float, p_inf: float,
                      n0: float, gamma: float) -> float:
    """``p_inf - (p_inf - p1) / (1 + units / n0)^gamma``.

    Tree-search saturating power law.
    """
    if n0 <= 0.0:
        n0 = 1e-9
    if gamma <= 0.0:
        gamma = 1e-9
    val = p_inf - (p_inf - p1) / math.pow(1.0 + units / n0, gamma)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _majority_curve(units: float, p1: float, p_inf: float,
                    k0: float, s: float) -> float:
    """Self-consistency curve: floor at ``p1`` then phase-transition.

    Uses a logistic CDF (smooth approximation to the de Moivre-Laplace
    threshold) saturating at ``p_inf``.
    """
    if s <= 0.0:
        s = 1e-3
    frac = _logistic((units - k0) / s)
    val = p1 + (p_inf - p1) * frac
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _verifier_curve(units: float, p1: float, p_inf: float,
                    r: float) -> float:
    """Verifier-rerank order-statistic curve.

    ``p(k) = p_inf - (p_inf - p1) (1 - r)^{k - 1}``, clipped to [0, 1].
    """
    if r < 0.0:
        r = 0.0
    if r >= 1.0:
        r = 1.0 - 1e-12
    val = p_inf - (p_inf - p1) * math.pow(1.0 - r, max(0.0, units - 1.0))
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    STRAT_PARALLEL: ("p1", "p_inf", "tau"),
    STRAT_SEQUENTIAL: ("p1", "p_inf", "tau_t"),
    STRAT_VERIFIER: ("p1", "p_inf", "r"),
    STRAT_BEAM: ("p1", "p_inf", "r"),
    STRAT_TREE: ("p1", "p_inf", "n0", "gamma"),
    STRAT_MAJORITY: ("p1", "p_inf", "k0", "s"),
}


# All strategies parameterise their curves on *unconstrained* reals
# using:
#   p1, p_inf:  logit-bounded to (0, 1).
#   tau, tau_t, n0, k0:  log-bounded to (0, +inf).
#   gamma, s:  log-bounded to (0, +inf).
#   r:  logit-bounded to (0, 1).
#
# A constraint ``p1 < p_inf`` is *softly* enforced by sorting
# unconstrained pairs at fit-evaluation time (``p_lo, p_hi``); the
# returned ``params`` dict reports ``min`` as ``p1`` and ``max`` as
# ``p_inf`` to match the curve formula.


def _from_logit(x: float) -> float:
    return _logistic(x)


def _from_log(x: float) -> float:
    return math.exp(x)


def _curve_eval(strategy: str, params: Sequence[float],
                units: float) -> float:
    if strategy == STRAT_PARALLEL or strategy == STRAT_SEQUENTIAL:
        pa, pb, lt = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        p1 = min(p_a, p_b)
        p_inf = max(p_a, p_b)
        tau = _from_log(lt)
        return _saturating_exp(units, p1, p_inf, tau)
    if strategy == STRAT_VERIFIER or strategy == STRAT_BEAM:
        pa, pb, lr = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        p1 = min(p_a, p_b)
        p_inf = max(p_a, p_b)
        r = _from_logit(lr)
        return _verifier_curve(units, p1, p_inf, r)
    if strategy == STRAT_TREE:
        pa, pb, ln0, lgamma = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        p1 = min(p_a, p_b)
        p_inf = max(p_a, p_b)
        n0 = _from_log(ln0)
        gamma = _from_log(lgamma)
        return _power_saturating(units, p1, p_inf, n0, gamma)
    if strategy == STRAT_MAJORITY:
        pa, pb, lk0, ls = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        p1 = min(p_a, p_b)
        p_inf = max(p_a, p_b)
        k0 = _from_log(lk0)
        s = _from_log(ls)
        return _majority_curve(units, p1, p_inf, k0, s)
    raise UnknownStrategy(strategy)


def _named_params(strategy: str, params: Sequence[float]) -> dict[str, float]:
    if strategy in (STRAT_PARALLEL, STRAT_SEQUENTIAL):
        pa, pb, lt = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        names = _PARAM_NAMES[strategy]
        return {
            names[0]: min(p_a, p_b),
            names[1]: max(p_a, p_b),
            names[2]: _from_log(lt),
        }
    if strategy in (STRAT_VERIFIER, STRAT_BEAM):
        pa, pb, lr = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        return {
            "p1": min(p_a, p_b),
            "p_inf": max(p_a, p_b),
            "r": _from_logit(lr),
        }
    if strategy == STRAT_TREE:
        pa, pb, ln0, lgamma = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        return {
            "p1": min(p_a, p_b),
            "p_inf": max(p_a, p_b),
            "n0": _from_log(ln0),
            "gamma": _from_log(lgamma),
        }
    if strategy == STRAT_MAJORITY:
        pa, pb, lk0, ls = params
        p_a = _from_logit(pa)
        p_b = _from_logit(pb)
        return {
            "p1": min(p_a, p_b),
            "p_inf": max(p_a, p_b),
            "k0": _from_log(lk0),
            "s": _from_log(ls),
        }
    raise UnknownStrategy(strategy)


# Initial parameter guess in unconstrained space, per strategy.
_INIT_GUESSES: dict[str, list[float]] = {
    STRAT_PARALLEL: [
        math.log(0.3 / 0.7),    # p1 ≈ 0.3
        math.log(0.85 / 0.15),  # p_inf ≈ 0.85
        math.log(16.0),         # tau
    ],
    STRAT_SEQUENTIAL: [
        math.log(0.3 / 0.7),
        math.log(0.85 / 0.15),
        math.log(64.0),
    ],
    STRAT_VERIFIER: [
        math.log(0.3 / 0.7),
        math.log(0.85 / 0.15),
        math.log(0.3 / 0.7),    # r ≈ 0.3
    ],
    STRAT_BEAM: [
        math.log(0.3 / 0.7),
        math.log(0.85 / 0.15),
        math.log(0.4 / 0.6),
    ],
    STRAT_TREE: [
        math.log(0.3 / 0.7),
        math.log(0.85 / 0.15),
        math.log(32.0),
        math.log(0.6),
    ],
    STRAT_MAJORITY: [
        math.log(0.3 / 0.7),
        math.log(0.85 / 0.15),
        math.log(5.0),
        math.log(2.0),
    ],
}


def _min_obs(strategy: str) -> int:
    return len(_INIT_GUESSES[strategy]) + 1


# ---------------------------------------------------------------------------
# Levenberg-Marquardt — shared with scaler's design but adapted to a
# *binomial* likelihood on (k, n) per row.  We fit in log-loss space
# on the negative-log-likelihood of the binomial; for stability we
# clip the predicted probability away from {0, 1}.
# ---------------------------------------------------------------------------


def _nll_one(strategy: str, params: Sequence[float],
             obs: Observation) -> float:
    p = _curve_eval(strategy, params, obs.compute_units)
    eps = 1e-9
    p = min(1.0 - eps, max(eps, p))
    return -obs.weight * (
        obs.successes * math.log(p)
        + (obs.trials - obs.successes) * math.log(1.0 - p)
    )


def _residuals(strategy: str, params: Sequence[float],
               obs: Sequence[Observation]
               ) -> tuple[list[float], list[float]]:
    """Residual vector for LM in *deviance* space.

    The binomial deviance contribution of one row is
    :math:`2 [k \\log(k / (n p)) + (n - k) \\log((n - k) / (n (1 - p)))]`.
    Its square root, signed by ``sign(k/n - p)``, is the deviance
    residual and is the LM target (Fisher 1922; McCullagh-Nelder 1989).
    """
    r: list[float] = []
    sw: list[float] = []
    eps = 1e-9
    for o in obs:
        p = _curve_eval(strategy, params, o.compute_units)
        p = min(1.0 - eps, max(eps, p))
        k = o.successes
        n = o.trials
        k_hat = k
        nk = n - k
        # Binomial deviance.
        if k_hat > 0:
            t1 = k_hat * math.log(k_hat / (n * p))
        else:
            t1 = 0.0
        if nk > 0:
            t2 = nk * math.log(nk / (n * (1.0 - p)))
        else:
            t2 = 0.0
        d = 2.0 * (t1 + t2)
        if d < 0.0:
            d = 0.0
        sign = 1.0 if (k / n) > p else -1.0
        r.append(sign * math.sqrt(d))
        sw.append(math.sqrt(o.weight))
    return r, sw


def _finite_diff_jacobian(strategy: str, params: Sequence[float],
                          obs: Sequence[Observation],
                          eps: float = 1e-5) -> list[list[float]]:
    p_len = len(params)
    rows: list[list[float]] = []
    base_r, _ = _residuals(strategy, params, obs)
    for idx, o in enumerate(obs):
        row = [0.0] * p_len
        for j in range(p_len):
            plus = list(params)
            minus = list(params)
            plus[j] = params[j] + eps
            minus[j] = params[j] - eps
            r_plus, _ = _residuals(strategy, plus, [o])
            r_minus, _ = _residuals(strategy, minus, [o])
            row[j] = (r_plus[0] - r_minus[0]) / (2.0 * eps)
        sw = math.sqrt(o.weight)
        rows.append([v * sw for v in row])
    return rows


def _matmul_at_b(a_rows: list[list[float]], b_rows: list[list[float]]
                 ) -> list[list[float]]:
    m = len(a_rows)
    n_a = len(a_rows[0]) if m else 0
    n_b = len(b_rows[0]) if b_rows else 0
    out = [[0.0] * n_b for _ in range(n_a)]
    for col in range(n_a):
        for row_b in range(len(b_rows)):
            v = a_rows[row_b][col]
            if v == 0.0:
                continue
            row_out = out[col]
            row_b_data = b_rows[row_b]
            for kk in range(n_b):
                row_out[kk] += v * row_b_data[kk]
    return out


def _matvec_at_b(a_rows: list[list[float]], b_vec: Sequence[float]
                 ) -> list[float]:
    m = len(a_rows)
    n_a = len(a_rows[0]) if m else 0
    out = [0.0] * n_a
    for row in range(m):
        v = b_vec[row]
        if v == 0.0:
            continue
        row_a = a_rows[row]
        for col in range(n_a):
            out[col] += row_a[col] * v
    return out


def _cholesky_solve(a: list[list[float]],
                    b: list[float]) -> list[float] | None:
    n = len(a)
    if n == 0:
        return []
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 0.0:
                    return None
                l[i][j] = math.sqrt(d)
            else:
                if l[j][j] == 0.0:
                    return None
                l[i][j] = (a[i][j] - s) / l[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(l[i][k] * y[k] for k in range(i))) / l[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(l[k][i] * x[k] for k in range(i + 1, n))) / l[i][i]
    return x


def _lm_fit(strategy: str, obs: Sequence[Observation],
            cfg: BudgeterConfig) -> tuple[list[float], int, bool]:
    if not obs:
        raise FitFailed("no observations to fit")
    params = list(_INIT_GUESSES[strategy])
    damping = 1e-2
    prev_rss = float("inf")
    converged = False
    iters = 0
    for it in range(cfg.max_iters):
        iters = it + 1
        r_raw, sw = _residuals(strategy, params, obs)
        r = [rv * w for rv, w in zip(r_raw, sw)]
        rss = sum(v * v for v in r)
        j_rows = _finite_diff_jacobian(strategy, params, obs)
        jtj = _matmul_at_b(j_rows, j_rows)
        jr = _matvec_at_b(j_rows, r)
        for i in range(len(params)):
            jtj[i][i] += damping + cfg.ridge
        step = _cholesky_solve(jtj, jr)
        if step is None:
            damping *= 10.0
            if damping > 1e12:
                break
            continue
        new_params = [params[i] - step[i] for i in range(len(params))]
        new_r_raw, new_sw = _residuals(strategy, new_params, obs)
        new_r = [rv * w for rv, w in zip(new_r_raw, new_sw)]
        new_rss = sum(v * v for v in new_r)
        if new_rss < rss:
            rel = sum(abs(s) for s in step) / max(1.0,
                                                  sum(abs(p) for p in params))
            params = new_params
            damping = max(damping / 3.0, 1e-12)
            if rel < cfg.tol and abs(prev_rss - new_rss) < cfg.tol * (rss + 1e-12):
                converged = True
                break
            prev_rss = new_rss
        else:
            damping = min(damping * 5.0, 1e12)
    return params, iters, converged


def _log_loss(strategy: str, params: Sequence[float],
              obs: Sequence[Observation]) -> float:
    if not obs:
        return 0.0
    total = 0.0
    w_total = 0.0
    eps = 1e-9
    for o in obs:
        p = _curve_eval(strategy, params, o.compute_units)
        p = min(1.0 - eps, max(eps, p))
        contrib = -(o.successes * math.log(p)
                    + (o.trials - o.successes) * math.log(1.0 - p))
        total += o.weight * contrib / o.trials
        w_total += o.weight
    return total / max(w_total, 1e-12)


def _stderr(strategy: str, params: Sequence[float],
            obs: Sequence[Observation], ridge: float) -> dict[str, float]:
    if not obs:
        return {n: 0.0 for n in _PARAM_NAMES[strategy]}
    j_rows = _finite_diff_jacobian(strategy, params, obs)
    r_raw, sw = _residuals(strategy, params, obs)
    r = [rv * w for rv, w in zip(r_raw, sw)]
    dof = max(1, len(obs) - len(params))
    sigma2 = sum(v * v for v in r) / dof
    jtj = _matmul_at_b(j_rows, j_rows)
    for i in range(len(params)):
        jtj[i][i] += ridge
    p_len = len(params)
    diag: list[float] = [float("inf")] * p_len
    for i in range(p_len):
        e = [0.0] * p_len
        e[i] = 1.0
        col = _cholesky_solve(jtj, e)
        if col is not None:
            diag[i] = col[i] * sigma2
    names = _PARAM_NAMES[strategy]
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        out[name] = math.sqrt(diag[i]) if math.isfinite(diag[i]) and diag[i] > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Unbiased pass@k estimator
# ---------------------------------------------------------------------------


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. (2021) — unbiased estimator of pass@k.

    :math:`1 - \\binom{n - c}{k} / \\binom{n}{k}` when ``n - c >= k``;
    ``1.0`` otherwise.  Pure Python, log-binom for numerical safety.

    Args:
        n: number of independent samples drawn.
        c: number of those samples that were correct.
        k: pass@k target.

    Returns:
        Estimated probability that *some* of ``k`` resampled draws are
        correct.

    Examples
    --------
    >>> abs(unbiased_pass_at_k(10, 3, 1) - 0.3) < 1e-9
    True
    >>> unbiased_pass_at_k(10, 0, 5) == 0.0
    True
    >>> unbiased_pass_at_k(5, 5, 3) == 1.0
    True
    """
    if not (0 <= c <= n and k >= 1):
        raise InvalidObservation(
            f"require 0 <= c <= n and k >= 1; got n={n}, c={c}, k={k}"
        )
    if n - c < k:
        return 1.0
    # log-binom(n-c, k) - log-binom(n, k) in stable form.
    log_num = 0.0
    log_den = 0.0
    for i in range(k):
        log_num += math.log((n - c - i) / (n - i))
    return 1.0 - math.exp(log_num)


# ---------------------------------------------------------------------------
# Allocator — Lagrangian water-filling
# ---------------------------------------------------------------------------


def _combine_independent(passes: Sequence[float]) -> float:
    """Combine independent per-strategy pass probabilities.

    Treats every active strategy as conditionally independent given
    the task; the combined pass is ``1 - prod(1 - p_i)``.  This is the
    standard ensemble bound (any-of-N).
    """
    prod = 1.0
    for p in passes:
        if p <= 0.0:
            continue
        prod *= (1.0 - p)
    return 1.0 - prod


def _allocate_water_filling(
    fits: Mapping[str, StrategyFit],
    specs: Mapping[str, StrategySpec],
    budget: float,
    cfg: BudgeterConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Lagrangian-style water-filling on a log-spaced compute grid.

    For each strategy ``s`` and unit count ``u`` on a log-spaced grid
    in ``[min_units, max_units]``, evaluate the cost ``c = u * unit_cost``
    and pass ``p = curve(u)``.  Then we solve

    .. math::

       \\max_{(u_s)_s} \\bigl[1 - \\prod_s (1 - p_s(u_s))\\bigr]
       \\text{ s.t. } \\sum_s c_s \\le C

    Because each ``(1 - p_s)`` is monotone non-increasing convex in
    ``u``, the log-objective decomposes per strategy and the optimum
    is a multiplicative water-filling:

    .. math::

       \\frac{d}{du} \\log(1 - p_s(u)) = -\\lambda \\cdot \\text{unit\\_cost}_s.

    We solve via bisection on :math:`\\lambda` between 0 and a large
    upper bound, picking, for each :math:`\\lambda`, the per-strategy
    ``u`` on the grid that maximises
    :math:`-\\log(1 - p_s(u)) - \\lambda \\cdot c_s`.  Returns
    ``({strategy: units}, {strategy: cost})``.
    """
    grid_size = cfg.allocator_grid
    eps = 1e-12

    grids: dict[str, list[tuple[float, float, float]]] = {}
    for name, fit in fits.items():
        spec = specs[name]
        params = _params_for_curve(fit)
        lo = math.log(max(eps, spec.min_units))
        hi = math.log(spec.max_units)
        units_grid: list[tuple[float, float, float]] = []
        # Always include "zero compute" via a leading sentinel (0 units, 0 cost,
        # 0 pass) so a strategy can drop out if its marginal is too weak.
        units_grid.append((0.0, 0.0, 0.0))
        for i in range(grid_size):
            t = i / (grid_size - 1)
            log_u = lo + t * (hi - lo)
            u = math.exp(log_u)
            p = _curve_eval(name, params, u)
            cost = u * spec.unit_cost
            units_grid.append((u, cost, p))
        grids[name] = units_grid

    def best_for_lambda(lam: float) -> tuple[dict[str, float],
                                              dict[str, float],
                                              float]:
        chosen_u: dict[str, float] = {}
        chosen_c: dict[str, float] = {}
        for name, grid in grids.items():
            best_score = -float("inf")
            best_u = 0.0
            best_c = 0.0
            for (u, c, p) in grid:
                log_miss = math.log(max(eps, 1.0 - p))
                # Maximise (-log_miss) - lam * c.
                score = (-log_miss) - lam * c
                if score > best_score:
                    best_score = score
                    best_u = u
                    best_c = c
            chosen_u[name] = best_u
            chosen_c[name] = best_c
        total = sum(chosen_c.values())
        return chosen_u, chosen_c, total

    # Find lambda such that total cost ≤ budget.
    lam_lo = 0.0
    # Upper bound: pick lam so the most expensive strategy gets 0 units.
    max_unit_cost = max(spec.unit_cost for spec in specs.values())
    lam_hi = max(1.0, 100.0 / max(1.0, budget) / max(eps, max_unit_cost))
    # Expand lam_hi until total cost at lam_hi <= budget.
    for _ in range(60):
        _, _, total_hi = best_for_lambda(lam_hi)
        if total_hi <= budget:
            break
        lam_hi *= 2.0
    # Bisection.
    u_chosen: dict[str, float] = {}
    c_chosen: dict[str, float] = {}
    for _ in range(80):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        u_chosen, c_chosen, total = best_for_lambda(lam_mid)
        if total > budget:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        if lam_hi - lam_lo < 1e-9 * max(1.0, lam_hi):
            break
    # Return the highest-spend lambda that still respects the budget.
    u_chosen, c_chosen, total = best_for_lambda(lam_hi)
    if total > budget:
        # Force budget compliance: prune the highest-cost strategy until
        # we're inside the envelope.
        items = sorted(c_chosen.items(), key=lambda kv: kv[1], reverse=True)
        for name, _ in items:
            if total <= budget:
                break
            u_chosen[name] = 0.0
            c_chosen[name] = 0.0
            total = sum(c_chosen.values())
    return u_chosen, c_chosen


def _params_for_curve(fit: StrategyFit) -> list[float]:
    """Recover the unconstrained parameter vector from a ``StrategyFit``.

    A ``StrategyFit`` stores the bounded form; the curve evaluator
    accepts the unconstrained form.  This re-encodes
    ``(p1, p_inf, …)`` -> unconstrained pair preserving the curve.
    """
    s = fit.strategy
    params = fit.params
    eps = 1e-9
    def _logit(p: float) -> float:
        p = min(1.0 - eps, max(eps, p))
        return math.log(p / (1.0 - p))
    if s in (STRAT_PARALLEL, STRAT_SEQUENTIAL):
        return [_logit(params["p1"]), _logit(params["p_inf"]),
                math.log(max(eps, params["tau" if s == STRAT_PARALLEL else "tau_t"]))]
    if s in (STRAT_VERIFIER, STRAT_BEAM):
        return [_logit(params["p1"]), _logit(params["p_inf"]),
                _logit(params["r"])]
    if s == STRAT_TREE:
        return [_logit(params["p1"]), _logit(params["p_inf"]),
                math.log(max(eps, params["n0"])),
                math.log(max(eps, params["gamma"]))]
    if s == STRAT_MAJORITY:
        return [_logit(params["p1"]), _logit(params["p_inf"]),
                math.log(max(eps, params["k0"])),
                math.log(max(eps, params["s"]))]
    raise UnknownStrategy(s)


# ---------------------------------------------------------------------------
# Budgeter primitive
# ---------------------------------------------------------------------------


class Budgeter:
    """Compute-optimal test-time inference allocation primitive.

    Lifecycle:

      1. :meth:`register_strategy` for each strategy the coordinator
         considers running (each carries a ``unit_cost`` in the common
         budget currency).
      2. :meth:`observe` rows of ``(strategy, difficulty, units, trials,
         successes)`` as inference runs complete.
      3. :meth:`fit` fits the per-strategy curve and caches it.
      4. :meth:`allocate` returns the budget-optimal split.
      5. :meth:`pareto` returns the cost/pass frontier.
      6. :meth:`certificate` bundles PAC LCBs, regret vs oracle, and the
         fingerprint chain.
      7. :meth:`report` returns a single :class:`BudgeterReport`.

    Thread-safe.  Deterministic given the seed.
    """

    def __init__(self, config: BudgeterConfig | None = None,
                 *, bus: EventBus | None = None) -> None:
        self.config = config or BudgeterConfig()
        self.bus = bus
        self._lock = threading.RLock()
        self._specs: dict[str, StrategySpec] = {}
        self._obs: list[Observation] = []
        self._fits: dict[str, StrategyFit] = {}
        self._fit_params: dict[str, list[float]] = {}
        self._last_allocation: Allocation | None = None
        self._fingerprint = hashlib.sha256()
        self._fingerprint.update(json.dumps({
            "version": 1,
            "seed": self.config.seed,
        }, sort_keys=True).encode())
        self._publish(BUDGETER_STARTED, {"seed": self.config.seed})

    # ---- event helpers ----
    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        payload = {**data, "ts": time.time()}
        self._fingerprint.update(json.dumps(
            {"kind": kind, "data": _stable(payload)}, sort_keys=True
        ).encode())
        if self.bus is not None:
            self.bus.publish(Event(kind=kind, data=payload))

    @property
    def fingerprint_hash(self) -> str:
        return self._fingerprint.hexdigest()

    # ---- strategy registration ----
    def register_strategy(self, spec: StrategySpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec
            self._fits.pop(spec.name, None)
            self._fit_params.pop(spec.name, None)

    @property
    def strategies(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._specs.keys()))

    def spec(self, name: str) -> StrategySpec:
        with self._lock:
            if name not in self._specs:
                raise UnknownStrategy(name)
            return self._specs[name]

    # ---- ingestion ----
    def observe(self, obs: Observation | Iterable[Observation]) -> int:
        with self._lock:
            if isinstance(obs, Observation):
                rows = [obs]
            else:
                rows = list(obs)
            for o in rows:
                if not isinstance(o, Observation):
                    raise InvalidObservation(
                        f"expected Observation, got {type(o).__name__}"
                    )
                if o.strategy not in self._specs:
                    raise UnknownStrategy(
                        f"observation refers to unregistered strategy "
                        f"{o.strategy!r}; call register_strategy() first"
                    )
                self._obs.append(o)
                self._publish(BUDGETER_OBSERVED, {
                    "strategy": o.strategy,
                    "difficulty": o.difficulty,
                    "compute_units": o.compute_units,
                    "trials": o.trials,
                    "successes": o.successes,
                    "weight": o.weight,
                })
            # Invalidate previous fits — re-fit on next call.
            self._fits.clear()
            self._fit_params.clear()
            self._last_allocation = None
            return len(self._obs)

    @property
    def observations(self) -> tuple[Observation, ...]:
        with self._lock:
            return tuple(self._obs)

    def reset(self) -> None:
        with self._lock:
            self._obs.clear()
            self._fits.clear()
            self._fit_params.clear()
            self._last_allocation = None
            self._publish(BUDGETER_RESET, {})

    # ---- per-strategy observation pooling ----
    def _rows_for(self, strategy: str,
                  difficulty: float | None) -> list[Observation]:
        rows = [o for o in self._obs if o.strategy == strategy]
        if difficulty is None or self.config.difficulty_kernel_bandwidth <= 0.0:
            return rows
        bw = self.config.difficulty_kernel_bandwidth
        return [o for o in rows if abs(o.difficulty - difficulty) <= bw]

    # ---- fitting ----
    def fit(self, *, difficulty: float | None = None,
            strategies: Iterable[str] | None = None
            ) -> dict[str, StrategyFit]:
        """Fit per-strategy curves and cache them.

        ``difficulty`` restricts the fit to rows within the kernel
        bandwidth of the given value.  ``strategies`` limits which
        strategies are fit (defaults to all registered).
        """
        with self._lock:
            if not self._specs:
                raise InvalidConfig("no strategies registered")
            names = (list(strategies) if strategies is not None
                     else list(self._specs.keys()))
            for n in names:
                if n not in self._specs:
                    raise UnknownStrategy(n)
            cfg = self.config
            results: dict[str, StrategyFit] = {}
            for n in names:
                rows = self._rows_for(n, difficulty)
                if len(rows) < _min_obs(n):
                    raise FitFailed(
                        f"strategy {n!r} needs >= {_min_obs(n)} observations "
                        f"to fit; have {len(rows)}"
                    )
                rng = random.Random(cfg.seed + _stable_hash(n))
                indices = list(range(len(rows)))
                rng.shuffle(indices)
                ho_n = int(round(cfg.holdout_fraction * len(rows)))
                ho_n = min(ho_n, max(0, len(rows) - _min_obs(n)))
                ho_set = set(indices[:ho_n])
                in_rows = [rows[i] for i in range(len(rows)) if i not in ho_set]
                ho_rows = [rows[i] for i in sorted(ho_set)]
                params, iters, converged = _lm_fit(n, in_rows, cfg)
                self._fit_params[n] = params
                ll_in = _log_loss(n, params, in_rows)
                ll_out = _log_loss(n, params, ho_rows) if ho_rows else None
                stderr = _stderr(n, params, in_rows, cfg.ridge)
                fit = StrategyFit(
                    strategy=n,
                    params=_named_params(n, params),
                    n_in_sample=len(in_rows),
                    n_held_out=len(ho_rows),
                    log_loss_in_sample=ll_in,
                    log_loss_held_out=ll_out,
                    iters=iters,
                    converged=converged,
                    parameter_stderr=stderr,
                )
                self._fits[n] = fit
                results[n] = fit
                self._publish(BUDGETER_FIT, {
                    "strategy": n,
                    "params": fit.params,
                    "n_in_sample": fit.n_in_sample,
                    "n_held_out": fit.n_held_out,
                    "log_loss_in_sample": fit.log_loss_in_sample,
                    "log_loss_held_out": fit.log_loss_held_out,
                    "converged": fit.converged,
                    "iters": fit.iters,
                })
            return results

    @property
    def fits(self) -> dict[str, StrategyFit]:
        with self._lock:
            return dict(self._fits)

    # ---- single-strategy extrapolation ----
    def extrapolate(self, strategy: str, compute_units: float
                    ) -> tuple[float, float, float]:
        """Predict ``(p, lower, upper)`` for one strategy at ``compute_units``.

        Bootstrap-percentile CI; degrades to point estimate when
        ``bootstrap_b == 0``.
        """
        with self._lock:
            if strategy not in self._fits:
                raise NotFitted(
                    f"strategy {strategy!r} not fit; call fit() first"
                )
            if not (compute_units > 0.0 and math.isfinite(compute_units)):
                raise InvalidObservation("compute_units must be positive finite")
            params = self._fit_params[strategy]
            point = _curve_eval(strategy, params, compute_units)
            if self.config.bootstrap_b == 0:
                self._publish(BUDGETER_EXTRAPOLATED, {
                    "strategy": strategy,
                    "compute_units": compute_units,
                    "p_point": point, "p_lower": point, "p_upper": point,
                })
                return point, point, point
            samples = self._bootstrap_curve(strategy, compute_units)
            samples.sort()
            alpha = (1.0 - self.config.confidence) / 2.0
            lo_idx = max(0, int(math.floor(alpha * len(samples))))
            hi_idx = min(len(samples) - 1,
                         int(math.ceil((1.0 - alpha) * len(samples))) - 1)
            lo = samples[lo_idx]
            hi = samples[hi_idx]
            self._publish(BUDGETER_EXTRAPOLATED, {
                "strategy": strategy,
                "compute_units": compute_units,
                "p_point": point, "p_lower": lo, "p_upper": hi,
            })
            return point, lo, hi

    def _bootstrap_curve(self, strategy: str,
                         compute_units: float) -> list[float]:
        """Case-resample bootstrap predictions for one strategy."""
        rows = [o for o in self._obs if o.strategy == strategy]
        if len(rows) < _min_obs(strategy):
            return [_curve_eval(strategy, self._fit_params[strategy],
                                compute_units)]
        rng = random.Random(self.config.seed + _stable_hash(strategy))
        preds: list[float] = []
        for _ in range(self.config.bootstrap_b):
            sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            try:
                p, _, _ = _lm_fit(strategy, sample, self.config)
            except FitFailed:
                continue
            preds.append(_curve_eval(strategy, p, compute_units))
        if not preds:
            preds.append(_curve_eval(strategy, self._fit_params[strategy],
                                     compute_units))
        return preds

    # ---- allocation ----
    def allocate(self, budget: float, *,
                 difficulty: float = 0.0,
                 strategies: Iterable[str] | None = None) -> Allocation:
        """Return the compute-optimal allocation at total budget ``budget``.

        Considers all strategies that have a fit.  Unfit strategies are
        silently skipped — register/observe/fit them first to include
        them in the allocation.  ``strategies`` further restricts the
        candidate set to a named subset.
        """
        with self._lock:
            if not self._specs:
                raise InvalidConfig("no strategies registered")
            if not (budget > 0.0 and math.isfinite(budget)):
                raise InvalidConfig("budget must be positive finite")
            if not self._fits:
                # Best-effort: only fit strategies that have observations.
                fittable = sorted({o.strategy for o in self._obs})
                if not fittable:
                    raise NotFitted(
                        "no observations recorded; call observe() and fit() "
                        "before allocate()"
                    )
                self.fit(difficulty=difficulty, strategies=fittable)
            candidates = list(self._fits.keys())
            if strategies is not None:
                wanted = set(strategies)
                for n in wanted:
                    if n not in self._fits:
                        raise NotFitted(
                            f"strategy {n!r} has no fit; call fit() first"
                        )
                candidates = [n for n in candidates if n in wanted]
            if not candidates:
                raise NotFitted("no fitted strategies available")
            fits_view = {n: self._fits[n] for n in candidates}
            specs_view = {n: self._specs[n] for n in candidates}
            # If we can't even afford the minimum on *one* strategy, infeasible.
            cheapest = min(specs_view[n].min_units * specs_view[n].unit_cost
                           for n in candidates)
            if budget < cheapest:
                raise InfeasibleBudget(
                    f"budget={budget} below cheapest single-strategy "
                    f"floor {cheapest:.4f}"
                )
            u_chosen, c_chosen = _allocate_water_filling(
                fits_view, specs_view, budget, self.config
            )
            per_pass: dict[str, float] = {}
            per_units: dict[str, float] = {}
            per_cost: dict[str, float] = {}
            active: list[str] = []
            for n in candidates:
                u = u_chosen.get(n, 0.0)
                per_units[n] = u
                per_cost[n] = c_chosen.get(n, 0.0)
                if u <= 0.0:
                    per_pass[n] = 0.0
                    continue
                params = self._fit_params[n]
                p = _curve_eval(n, params, u)
                per_pass[n] = p
                if p > 0.0:
                    active.append(n)
            point = _combine_independent(list(per_pass.values()))
            # Bootstrap the combined pass under uncertainty in the curves.
            lo = hi = point
            if self.config.bootstrap_b > 0:
                samples = self._bootstrap_allocation(u_chosen)
                samples.sort()
                alpha = (1.0 - self.config.confidence) / 2.0
                lo_idx = max(0, int(math.floor(alpha * len(samples))))
                hi_idx = min(len(samples) - 1,
                             int(math.ceil((1.0 - alpha) * len(samples))) - 1)
                lo = samples[lo_idx]
                hi = samples[hi_idx]
            spent = sum(per_cost.values())
            alloc = Allocation(
                budget=budget,
                difficulty=difficulty,
                per_strategy_units=dict(per_units),
                per_strategy_cost=dict(per_cost),
                per_strategy_pass=dict(per_pass),
                predicted_pass=point,
                predicted_lower=lo,
                predicted_upper=hi,
                confidence=self.config.confidence,
                active_strategies=tuple(sorted(active)),
                spent=spent,
            )
            self._last_allocation = alloc
            self._publish(BUDGETER_ALLOCATED, alloc.to_dict())
            return alloc

    def _bootstrap_allocation(self, units: Mapping[str, float]
                              ) -> list[float]:
        rng = random.Random(self.config.seed + 7919)
        bsamples: dict[str, list[float]] = {}
        for n in units:
            if n not in self._fit_params:
                continue
            rows = [o for o in self._obs if o.strategy == n]
            if len(rows) < _min_obs(n):
                bsamples[n] = [_curve_eval(n, self._fit_params[n],
                                            units.get(n, 0.0))]
                continue
            preds: list[float] = []
            for _ in range(self.config.bootstrap_b):
                sample = [rows[rng.randrange(len(rows))]
                          for _ in range(len(rows))]
                try:
                    p, _, _ = _lm_fit(n, sample, self.config)
                except FitFailed:
                    continue
                preds.append(_curve_eval(n, p, units.get(n, 0.0)))
            if not preds:
                preds.append(_curve_eval(n, self._fit_params[n],
                                          units.get(n, 0.0)))
            bsamples[n] = preds
        # Combine bootstrap draws across strategies (independent assumption).
        b = self.config.bootstrap_b
        out: list[float] = []
        names = sorted(bsamples.keys())
        for i in range(b):
            ps = []
            for n in names:
                draws = bsamples[n]
                ps.append(draws[i % len(draws)])
            out.append(_combine_independent(ps))
        return out

    # ---- pareto frontier ----
    def pareto(self, *, min_budget: float, max_budget: float,
               n_points: int = 16, difficulty: float = 0.0
               ) -> list[ParetoPoint]:
        """Return ``n_points`` log-spaced budget levels on the frontier."""
        with self._lock:
            if n_points < 2:
                raise InvalidConfig("n_points must be >= 2")
            if not (0.0 < min_budget < max_budget):
                raise InvalidConfig("require 0 < min_budget < max_budget")
            if not self._fits:
                self.fit(difficulty=difficulty)
            log_lo = math.log(min_budget)
            log_hi = math.log(max_budget)
            out: list[ParetoPoint] = []
            for i in range(n_points):
                t = i / (n_points - 1)
                budget = math.exp(log_lo + t * (log_hi - log_lo))
                try:
                    alloc = self.allocate(budget, difficulty=difficulty)
                except InfeasibleBudget:
                    continue
                out.append(ParetoPoint(
                    budget=budget,
                    spent=alloc.spent,
                    predicted_pass=alloc.predicted_pass,
                    predicted_lower=alloc.predicted_lower,
                    predicted_upper=alloc.predicted_upper,
                    active_strategies=alloc.active_strategies,
                ))
            self._publish(BUDGETER_PARETO, {
                "n_points": len(out),
                "min_budget": min_budget,
                "max_budget": max_budget,
                "difficulty": difficulty,
            })
            return out

    # ---- certificate ----
    def certificate(self, *, confidence: float | None = None
                    ) -> BudgeterCertificate:
        """Return a replay-verifiable certificate over the latest allocation."""
        with self._lock:
            if self._last_allocation is None:
                raise NotFitted("call allocate() before certificate()")
            cfg = self.config
            conf = confidence if confidence is not None else cfg.confidence
            if not (0.0 < conf < 1.0):
                raise InvalidConfig("confidence must lie in (0, 1)")
            alloc = self._last_allocation
            # Held-out pass: pool all held-out rows for the active strategies.
            ho_pass_num = 0.0
            ho_pass_den = 0.0
            sigma_sq_num = 0.0
            for n, fit in self._fits.items():
                if fit.n_held_out == 0:
                    continue
                rows = self._rows_for(n, alloc.difficulty)
                rng = random.Random(cfg.seed + _stable_hash(n))
                indices = list(range(len(rows)))
                rng.shuffle(indices)
                ho_n = int(round(cfg.holdout_fraction * len(rows)))
                ho_n = min(ho_n, max(0, len(rows) - _min_obs(n)))
                ho_set = set(indices[:ho_n])
                ho_rows = [rows[i] for i in sorted(ho_set)]
                params = self._fit_params[n]
                for o in ho_rows:
                    p_pred = _curve_eval(n, params, o.compute_units)
                    rate = o.successes / o.trials
                    diff = rate - p_pred
                    ho_pass_num += rate * o.trials
                    ho_pass_den += o.trials
                    sigma_sq_num += diff * diff * o.trials
            ho_rate: float | None = None
            ho_lcb_h: float | None = None
            ho_lcb_b: float | None = None
            if ho_pass_den > 0:
                ho_rate = ho_pass_num / ho_pass_den
                m = int(ho_pass_den)
                delta = 1.0 - conf
                hoeffding = math.sqrt(math.log(1.0 / delta) / (2.0 * m))
                # Empirical-Bernstein on the absolute deviation,
                # using deviation between predicted and observed.
                sigma2 = sigma_sq_num / max(1, m - 1)
                sigma = math.sqrt(max(sigma2, 0.0))
                bernstein = (
                    math.sqrt(2.0 * sigma * sigma * math.log(2.0 / delta) / m)
                    + 7.0 * math.log(2.0 / delta) / (3.0 * max(1, m - 1))
                )
                ho_lcb_h = max(0.0, ho_rate - hoeffding)
                ho_lcb_b = max(0.0, ho_rate - bernstein)
            # Oracle: best single-strategy allocation at the same total spend.
            oracle_name: str | None = None
            oracle_pass: float | None = None
            for n in self._fits:
                spec = self._specs[n]
                budget = alloc.spent
                if budget <= 0:
                    continue
                u = min(budget / spec.unit_cost, spec.max_units)
                if u < spec.min_units:
                    continue
                params = self._fit_params[n]
                p_oracle = _curve_eval(n, params, u)
                if oracle_pass is None or p_oracle > oracle_pass:
                    oracle_pass = p_oracle
                    oracle_name = n
            regret: float | None
            if oracle_pass is not None:
                regret = max(0.0, oracle_pass - alloc.predicted_pass)
            else:
                regret = None
            in_range: dict[str, tuple[float, float]] = {}
            for n in self._fits:
                rows = [o for o in self._obs if o.strategy == n]
                if not rows:
                    in_range[n] = (0.0, 0.0)
                    continue
                in_range[n] = (
                    min(o.compute_units for o in rows),
                    max(o.compute_units for o in rows),
                )
            extrap_factor = 1.0
            for n, (_lo, hi) in in_range.items():
                used = alloc.per_strategy_units.get(n, 0.0)
                if hi > 0:
                    factor = used / hi if used > 0 else 0.0
                    if factor > extrap_factor:
                        extrap_factor = factor
            cert = BudgeterCertificate(
                confidence=conf,
                n_held_out=int(ho_pass_den),
                pass_held_out=ho_rate,
                pass_lcb_hoeffding=ho_lcb_h,
                pass_lcb_bernstein=ho_lcb_b,
                regret_ucb=regret,
                oracle_strategy=oracle_name,
                oracle_pass=oracle_pass,
                in_range_per_strategy=in_range,
                extrapolation_factor=extrap_factor,
                fingerprint_hash=self.fingerprint_hash,
            )
            self._publish(BUDGETER_CERTIFIED, cert.to_dict())
            return cert

    # ---- recommend (sugar) ----
    def recommend(self, *, budget: float,
                  difficulty: float = 0.0) -> Allocation:
        """End-to-end: fit if needed, allocate, certify, return allocation."""
        with self._lock:
            if not self._fits:
                self.fit(difficulty=difficulty)
            return self.allocate(budget, difficulty=difficulty)

    # ---- report ----
    def report(self) -> BudgeterReport:
        with self._lock:
            cfg = {
                "seed": self.config.seed,
                "bootstrap_b": self.config.bootstrap_b,
                "confidence": self.config.confidence,
                "holdout_fraction": self.config.holdout_fraction,
                "max_iters": self.config.max_iters,
                "tol": self.config.tol,
                "ridge": self.config.ridge,
                "difficulty_kernel_bandwidth": self.config.difficulty_kernel_bandwidth,
                "allocator_grid": self.config.allocator_grid,
                "oracle_ref_units": self.config.oracle_ref_units,
            }
            strategies = {
                n: {
                    "unit_cost": s.unit_cost,
                    "verifier_info": s.verifier_info,
                    "min_units": s.min_units,
                    "max_units": s.max_units,
                }
                for n, s in self._specs.items()
            }
            fits = {n: f.to_dict() for n, f in self._fits.items()}
            alloc = self._last_allocation.to_dict() if self._last_allocation else None
            cert: dict[str, Any] | None
            try:
                cert = self.certificate().to_dict() if self._last_allocation else None
            except (NotFitted, InvalidConfig):
                cert = None
            out = BudgeterReport(
                config=cfg,
                strategies=strategies,
                observations=len(self._obs),
                fits=fits,
                allocation=alloc,
                certificate=cert,
            )
            self._publish(BUDGETER_REPORTED, {
                "observations": out.observations,
                "n_strategies": len(strategies),
                "n_fits": len(fits),
                "has_allocation": alloc is not None,
            })
            return out


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def default_parallel_spec(*, unit_cost: float = 1.0,
                          min_units: float = 1.0,
                          max_units: float = 256.0) -> StrategySpec:
    return StrategySpec(name=STRAT_PARALLEL,
                        unit_cost=unit_cost,
                        min_units=min_units,
                        max_units=max_units)


def default_majority_spec(*, unit_cost: float = 1.0,
                          min_units: float = 1.0,
                          max_units: float = 256.0) -> StrategySpec:
    return StrategySpec(name=STRAT_MAJORITY,
                        unit_cost=unit_cost,
                        min_units=min_units,
                        max_units=max_units)


def default_verifier_spec(*, unit_cost: float = 1.2,
                          verifier_info: float = 0.4,
                          min_units: float = 1.0,
                          max_units: float = 256.0) -> StrategySpec:
    return StrategySpec(name=STRAT_VERIFIER,
                        unit_cost=unit_cost,
                        verifier_info=verifier_info,
                        min_units=min_units,
                        max_units=max_units)


def default_beam_spec(*, unit_cost: float = 1.5,
                      verifier_info: float = 0.6,
                      min_units: float = 1.0,
                      max_units: float = 64.0) -> StrategySpec:
    return StrategySpec(name=STRAT_BEAM,
                        unit_cost=unit_cost,
                        verifier_info=verifier_info,
                        min_units=min_units,
                        max_units=max_units)


def default_tree_spec(*, unit_cost: float = 2.0,
                      min_units: float = 1.0,
                      max_units: float = 128.0) -> StrategySpec:
    return StrategySpec(name=STRAT_TREE,
                        unit_cost=unit_cost,
                        min_units=min_units,
                        max_units=max_units)


def default_sequential_spec(*, unit_cost: float = 0.05,
                            min_units: float = 16.0,
                            max_units: float = 4096.0) -> StrategySpec:
    """Sequential thinking — unit is *one thinking-token*; the unit_cost
    therefore is much smaller than for one full sample."""
    return StrategySpec(name=STRAT_SEQUENTIAL,
                        unit_cost=unit_cost,
                        min_units=min_units,
                        max_units=max_units)


def fresh_budgeter(*, seed: int = 0, **cfg: Any) -> Budgeter:
    """Construct a :class:`Budgeter` pre-loaded with sensible default specs.

    Returns a Budgeter with parallel + majority + verifier + beam + tree
    + sequential strategies registered.  The caller still owns
    :meth:`observe`, :meth:`fit`, :meth:`allocate`.
    """
    b = Budgeter(BudgeterConfig(seed=seed, **cfg))
    b.register_strategy(default_parallel_spec())
    b.register_strategy(default_majority_spec())
    b.register_strategy(default_verifier_spec())
    b.register_strategy(default_beam_spec())
    b.register_strategy(default_tree_spec())
    b.register_strategy(default_sequential_spec())
    return b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_hash(name: str) -> int:
    """Stable per-process hash of a strategy name.

    Python's builtin ``hash()`` for strings is randomised per process
    (``PYTHONHASHSEED``), which breaks cross-process determinism.  We
    derive a 16-bit integer from the first four bytes of SHA-256 of
    the name — same value across every Python process.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big")


def _stable(value: Any) -> Any:
    """Make a payload deterministic for the fingerprint chain."""
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value.keys()) if k != "ts"}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.12g}")
    return value


__all__ = [
    "STRAT_PARALLEL",
    "STRAT_MAJORITY",
    "STRAT_VERIFIER",
    "STRAT_BEAM",
    "STRAT_TREE",
    "STRAT_SEQUENTIAL",
    "KNOWN_STRATEGIES",
    "BUDGETER_STARTED",
    "BUDGETER_OBSERVED",
    "BUDGETER_FIT",
    "BUDGETER_EXTRAPOLATED",
    "BUDGETER_ALLOCATED",
    "BUDGETER_PARETO",
    "BUDGETER_CERTIFIED",
    "BUDGETER_REPORTED",
    "BUDGETER_RESET",
    "BudgeterError",
    "InvalidConfig",
    "InvalidObservation",
    "UnknownStrategy",
    "NotFitted",
    "FitFailed",
    "InfeasibleBudget",
    "StrategySpec",
    "BudgeterConfig",
    "Observation",
    "StrategyFit",
    "Allocation",
    "ParetoPoint",
    "BudgeterCertificate",
    "BudgeterReport",
    "Budgeter",
    "unbiased_pass_at_k",
    "default_parallel_spec",
    "default_majority_spec",
    "default_verifier_spec",
    "default_beam_spec",
    "default_tree_spec",
    "default_sequential_spec",
    "fresh_budgeter",
]
