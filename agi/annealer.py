r"""Annealer — combinatorial optimisation as a runtime primitive.

Vast tracts of real engineering, scientific, and *agent-coordination*
work reduce to **minimise / maximise some objective over a discrete
search space**: which K tickets to admit, which jobs to which workers,
which subset of skills to mine, which test cases to retire, which order
to execute a goal's plan steps, which assignment of replicas to GPU
shards satisfies the cluster's bandwidth budget, which TSP-style tour
through the user's task graph minimises context-switching cost.  None
of these have closed-form solutions; none of them are smooth enough for
gradient descent; and most of them are NP-hard so we will never have a
provably optimal poly-time algorithm.  The mathematically rigorous
*and* practically deployable workhorse for this class of problem is
**simulated annealing** and its modern extensions (replica exchange,
late acceptance, basin hopping, tabu search).

``Annealer`` is the runtime-level *implementation* of that primitive.
It owns the proposal kernel, the cooling schedule, the replica
ensemble, the move memory, the restart logic, the random tape, and an
anytime-valid PAC certificate on the best objective value found so
far — so the coordinator can decide *act vs continue annealing vs
escalate to a more expensive solver* at any tick of the search.

The pitch reduced to a runtime call::

    an = Annealer(AnnealerConfig(
        algorithm=ALGO_SA,
        schedule=SCHED_GEOMETRIC,
        t_init=1.0,
        t_final=1e-3,
        max_iter=10_000,
        seed=0,
    ))

    # Caller supplies a problem: an initial state, a cost function,
    # and a neighbour kernel.  Or use a built-in builder:
    prob = annealer_tsp(points=cities, seed=0)

    rep  = an.run(prob)              # AnnealerReport
    cert = an.certify(rep, delta=0.05)  # AnnealerCertificate (PAC LCB)

    # Reusable as a primitive: every other discrete-decision module
    # in the runtime can call an.run() with its own (state, cost,
    # neighbour) triple and get back a calibrated, replay-verifiable
    # best-known solution plus a Hoeffding / empirical-Bernstein
    # bound on the gap to the global optimum.


What this primitive ships
-------------------------

  * **Six algorithms** — toggleable via ``AnnealerConfig.algorithm``:

    * ``ALGO_SA``        — single-chain simulated annealing
      (Kirkpatrick-Gelatt-Vecchi 1983 *Optimization by Simulated
      Annealing*).  Metropolis-Hastings acceptance with
      ``P(accept) = min(1, exp(−ΔE / T_k))``.  Cooling schedule
      :data:`SCHED_GEOMETRIC`, :data:`SCHED_LOG`, :data:`SCHED_LINEAR`,
      :data:`SCHED_LUNDY_MEES`, or :data:`SCHED_ADAPTIVE`.

    * ``ALGO_PT``        — parallel tempering / replica exchange
      (Swendsen-Wang 1986 *Replica Monte Carlo Simulation of
      Spin-Glasses*; Hukushima-Nemoto 1996 *Exchange Monte Carlo
      Method and Application to Spin Glass Simulations*).  ``K``
      replicas at geometrically-spaced temperatures, periodic swap
      attempts at acceptance probability
      ``min(1, exp((β_i − β_j)(E_i − E_j)))``.  Detailed-balance
      preserving; provably ergodic on the product chain.  Beats
      single-chain SA on rugged landscapes with metastable basins.

    * ``ALGO_LAHC``      — late acceptance hill-climbing
      (Burke-Bykov 2017 *The Late Acceptance Hill-Climbing
      Heuristic*).  Memory of length ``L``; accept the new state
      iff its cost is no worse than the cost of the state
      ``L`` iterations ago.  One hyperparameter, no temperature
      schedule, often outperforms SA on practical TSP / nurse-
      rostering problems.

    * ``ALGO_BASIN``     — basin hopping (Wales-Doye 1997 *Global
      Optimisation by Basin-Hopping*).  Outer Metropolis loop over
      *local minima*: perturb, descend to nearest local minimum via
      caller-supplied local search (or the default first-improvement
      greedy), accept the local minimum with SA-style probability.
      The classical algorithm for molecular-conformation /
      protein-folding global optimisation.

    * ``ALGO_TABU``      — tabu search (Glover 1989-1990 *Tabu
      Search Part I & II*).  Greedy best-improvement with a
      short-term memory of forbidden moves of length ``tabu_tenure``;
      aspiration criterion overrides the tabu list when a candidate
      beats the global best.  Deterministic given a deterministic
      neighbour-enumeration; pairs well with ``LAHC`` as the cooling
      complement.

    * ``ALGO_RESTART``   — Luby restart wrapper (Luby-Sinclair-Zuckerman
      1993 *Optimal Speedup of Las Vegas Algorithms*).  Wraps any
      base algorithm with the Luby sequence
      ``1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, …`` of restart
      lengths; provably within a logarithmic factor of the optimal
      static restart strategy without knowing the run-time
      distribution.  The complement of SA's geometric cooling for
      Las-Vegas-style randomised local search.

  * **Five cooling schedules** for SA / PT / Basin — toggleable via
    ``AnnealerConfig.schedule``:

    * ``SCHED_GEOMETRIC`` — ``T_{k+1} = α · T_k`` with
      ``α = (T_final / T_init)^{1/max_iter}``.  Default; near-universal
      in practice.

    * ``SCHED_LOG``       — ``T_k = c / log(2 + k)`` (Geman-Geman 1984
      *Stochastic Relaxation, Gibbs Distributions, and the Bayesian
      Restoration of Images*).  Provably converges to the global
      optimum in the infinite-time limit; impractically slow.

    * ``SCHED_LINEAR``    — ``T_k = T_init − k · (T_init − T_final)/max_iter``.

    * ``SCHED_LUNDY_MEES`` — ``T_{k+1} = T_k / (1 + β T_k)`` with
      ``β = (T_init − T_final) / (max_iter · T_init · T_final)``
      (Lundy-Mees 1986 *Convergence of an annealing algorithm*).
      Provably escapes any local minimum with prob → 1 under
      bounded-cost assumption.

    * ``SCHED_ADAPTIVE``  — Lam-Delosme 1988 *An efficient
      simulated annealing schedule*: dynamically adjust ``T_k``
      so that the rolling-window acceptance rate tracks a target
      of ``0.44``.  Compresses cooling on rejection-heavy plateaus,
      extends it on acceptance-heavy basins.

  * **Multi-start aggregation** — ``run`` accepts ``restarts: int``
    that fork the chain into ``restarts`` independent runs each
    with its own derived seed; returns the best across all forks
    and an *empirical regret* between forks that bounds the
    optimisation-variance contribution to the certificate's gap.

  * **Built-in problem builders** — every common combinatorial
    optimisation benchmark exposed as a callable that returns a
    ``Problem`` (state + cost_fn + neighbour_fn + checker):

    * ``annealer_tsp(points)``       — symmetric TSP, 2-opt swap
      neighbour kernel, Euclidean cost; Held-Karp lower-bound
      helper for the certificate.

    * ``annealer_max_cut(graph)``    — MaxCut on a weighted graph,
      single-vertex flip neighbour kernel, Goemans-Williamson 0.878
      approximation factor as a lower bound on the optimum.

    * ``annealer_max_sat(clauses)``  — MaxSAT on N booleans, single-
      bit flip neighbour kernel, Johnson 7/8 approximation lower
      bound on 3-SAT clauses.

    * ``annealer_qap(flow, dist)``   — Quadratic Assignment Problem,
      adjacent-pair swap neighbour kernel, Gilmore-Lawler lower
      bound; the QAPLIB-standard NP-hard benchmark.

    * ``annealer_number_partition(weights)`` — Karmarkar-Karp
      number partitioning (NP-hard), single-element flip kernel,
      Karmarkar-Karp differencing heuristic as the lower bound.

    * ``annealer_knapsack(weights, values, capacity)`` — 0/1
      knapsack, single-bit flip + repair, LP-relaxation upper
      bound on the optimum (since knapsack is maximisation we
      flip signs internally so the engine consistently *minimises*).

  * **PAC certificates** — every report carries:

    * ``best_cost``       — the lowest cost seen
    * ``best_state``      — the corresponding state
    * ``gap_hoeffding(δ)`` — Hoeffding upper bound on
      ``E[best_cost_at_runtime_T] − OPT`` derived from the
      empirical acceptance-rate distribution; valid whenever the
      cost is bounded on the search space and a lower-bound
      helper ``problem.lower_bound`` is provided.
    * ``gap_bernstein(δ)`` — empirical-Bernstein refinement of the
      above (Maurer-Pontil 2009) using the in-run cost variance.
    * ``p_global_opt(δ)`` — anytime-valid lower confidence bound on
      the probability of having visited the global minimum at
      least once, derived from the Geman-Geman log-cooling theorem
      when ``schedule = SCHED_LOG`` (otherwise reports an
      empirical surrogate via the restart-fork agreement rate).

  * **Replay-verifiable receipts** — SHA-256 fingerprint chain
    (optionally HMAC'd) over every observation: ``started``,
    ``proposed``, ``accepted``, ``rejected``, ``swap``, ``restart``,
    ``checkpoint``, ``finished``.  ``annealer_ledger_root`` is the
    immutable genesis ``agi.annealer.v1``.  Replaying the chain
    reproduces every state transition byte-for-byte.  Pluggable
    HMAC key for tamper-evident multi-tenant deployments.

  * **Snapshot / restore** — ``snapshot()`` returns a JSON-encodable
    state dict (random-tape position, best-so-far, chain head,
    replica temperatures, tabu list, LAHC buffer) that
    ``restore()`` can use to resume the search byte-identically.
    Composes with ``Persistence`` for crash recovery.

  * **Thread-safe re-entrant lock**; transport-agnostic; pure
    stdlib (no NumPy, no SciPy, no Torch); deterministic given
    seed.


Composes with
-------------

  * ``Submodular`` — for maximum-coverage-style problems with a
    submodular objective, ``Submodular``'s greedy / lazy-greedy /
    stochastic-greedy is provably ``(1−1/e)``-optimal; for
    *non-submodular* set selection (knapsack, MaxCut, MaxSAT, QAP)
    ``Annealer`` is the drop-in replacement.  Combined: greedy
    initial state → SA polish.

  * ``Coalition`` — Shapley credit assignment under a non-submodular
    characteristic function reduces to an SA-style sampling over
    permutations; ``Annealer.run`` with ``annealer_permutation_kernel``
    is the natural sampler.

  * ``Portfolio`` — fixed-budget integer allocation across N tickets
    (NP-hard as 0/1 knapsack) becomes ``Annealer.run`` over
    ``annealer_knapsack``; ``Portfolio`` consumes the best-state +
    Hoeffding gap as a feasible-with-confidence allocation.

  * ``Negotiator`` — sealed-bid externality-based allocation reduces
    to an integer-programming problem in the indivisible-item
    setting; ``Annealer`` is the worst-case fallback when the LP
    relaxation has fractional optima.

  * ``Robustifier`` — adversarial worst-case planning over a finite
    perturbation set: caller's perturbation index is a discrete
    decision variable; ``Annealer.run`` finds the worst-case
    perturbation and feeds it back as the DRO certificate.

  * ``Diplomat`` / ``Equilibrator`` — extensive-form / normal-form
    games where the action space is large but discrete; ``Annealer``
    with ``annealer_zero_sum_kernel`` solves the inner minimisation
    in best-response iteration.

  * ``Scheduler`` / ``Coordinator`` — agent-coordination level: which
    ticket goes to which session, which order to execute a plan's
    independent steps, which subset of tools to install for a goal.
    All three are NP-hard discrete optimisation problems and reduce
    to ``Annealer.run``.

  * ``Strategist`` — top-level meta-decision API: the SA gap
    ``Hoeffding / Bernstein`` LCB is a natural risk dimension for
    risk-adjusted recommendations.

  * ``AttestationLedger`` — every accept / reject / swap / restart
    hashes into the global audit ledger; replay reproduces the
    optimisation trajectory byte-for-byte.


Mathematical notation
---------------------

  * ``X``                 — discrete search space.
  * ``x ∈ X``             — a state.
  * ``f : X → ℝ``         — cost (objective being minimised).
  * ``N(x) ⊂ X``          — neighbours of ``x`` under the kernel.
  * ``T_k``               — temperature at step ``k``.
  * ``ΔE = f(x') − f(x)`` — energy change of a proposed move.
  * ``α``                 — geometric cooling rate.
  * ``L``                 — LAHC memory length.
  * ``M``                 — tabu tenure.
  * ``K``                 — number of PT replicas.
  * ``β_i = 1/T_i``       — inverse temperature of replica ``i``.
  * ``OPT = min_x f(x)``  — global optimum (unknown).

All ingest paths are validated.  Inference is ``O(max_iter)`` plus
``O(neighbour_kernel_cost)`` per iteration; SA / LAHC are constant-
memory; PT is ``O(K)``-memory; Tabu is ``O(tabu_tenure)``-memory.
No ``random`` without explicit seed; no ``time.time()`` leaks into
the chain.

References
----------

  * Kirkpatrick, Gelatt, Vecchi 1983. *Optimization by Simulated
    Annealing.* Science 220(4598):671-680.
  * Metropolis, Rosenbluth, Rosenbluth, Teller, Teller 1953.
    *Equation of State Calculations by Fast Computing Machines.*
    Journal of Chemical Physics 21(6):1087-1092.
  * Geman, Geman 1984. *Stochastic Relaxation, Gibbs Distributions,
    and the Bayesian Restoration of Images.* IEEE PAMI 6(6):721-741.
  * Lundy, Mees 1986. *Convergence of an Annealing Algorithm.*
    Mathematical Programming 34:111-124.
  * Lam, Delosme 1988. *Performance of a New Annealing Schedule.*
    DAC '88.
  * Swendsen, Wang 1986. *Replica Monte Carlo Simulation of
    Spin-Glasses.* Physical Review Letters 57:2607-2609.
  * Hukushima, Nemoto 1996. *Exchange Monte Carlo Method and
    Application to Spin Glass Simulations.* J. Phys. Soc. Japan
    65:1604-1608.
  * Burke, Bykov 2017. *The Late Acceptance Hill-Climbing
    Heuristic.* European Journal of Operational Research
    258(1):70-78.
  * Wales, Doye 1997. *Global Optimisation by Basin-Hopping and
    the Lowest Energy Structures of Lennard-Jones Clusters
    Containing up to 110 Atoms.* J. Phys. Chem. A 101:5111-5116.
  * Glover 1989. *Tabu Search — Part I.* ORSA Journal on Computing
    1(3):190-206.
  * Glover 1990. *Tabu Search — Part II.* ORSA Journal on
    Computing 2(1):4-32.
  * Luby, Sinclair, Zuckerman 1993. *Optimal Speedup of Las Vegas
    Algorithms.* Information Processing Letters 47:173-180.
  * Goemans, Williamson 1995. *Improved Approximation Algorithms
    for Maximum Cut and Satisfiability Problems Using Semidefinite
    Programming.* Journal of the ACM 42(6):1115-1145.
  * Johnson 1974. *Approximation Algorithms for Combinatorial
    Problems.* JCSS 9(3):256-278.
  * Karmarkar, Karp 1982. *The Differencing Method of Set
    Partitioning.* Berkeley Tech. Report.
  * Hoeffding 1963. *Probability Inequalities for Sums of Bounded
    Random Variables.* JASA 58.
  * Maurer, Pontil 2009. *Empirical Bernstein Bounds and Sample
    Variance Penalisation.* COLT.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    # Events
    "ANNEALER_STARTED",
    "ANNEALER_PROPOSED",
    "ANNEALER_ACCEPTED",
    "ANNEALER_REJECTED",
    "ANNEALER_SWAPPED",
    "ANNEALER_RESTARTED",
    "ANNEALER_CHECKPOINTED",
    "ANNEALER_FINISHED",
    "ANNEALER_CERTIFIED",
    "ANNEALER_RESET",
    # Algorithms
    "ALGO_SA",
    "ALGO_PT",
    "ALGO_LAHC",
    "ALGO_BASIN",
    "ALGO_TABU",
    "ALGO_RESTART",
    "KNOWN_ALGORITHMS",
    # Schedules
    "SCHED_GEOMETRIC",
    "SCHED_LOG",
    "SCHED_LINEAR",
    "SCHED_LUNDY_MEES",
    "SCHED_ADAPTIVE",
    "KNOWN_SCHEDULES",
    # Exceptions
    "AnnealerError",
    "InvalidConfig",
    "InvalidProblem",
    "InvalidState",
    "InsufficientData",
    "UnknownAlgorithm",
    "UnknownSchedule",
    "NotRun",
    # Dataclasses
    "AnnealerConfig",
    "Problem",
    "AnnealerReport",
    "AnnealerCertificate",
    "ReplicaSnapshot",
    # Helpers
    "annealer_ledger_root",
    "annealer_geometric_schedule",
    "annealer_log_schedule",
    "annealer_linear_schedule",
    "annealer_lundy_mees_schedule",
    "annealer_adaptive_schedule",
    "annealer_metropolis_accept",
    "annealer_luby_sequence",
    # Problem builders
    "annealer_tsp",
    "annealer_max_cut",
    "annealer_max_sat",
    "annealer_qap",
    "annealer_number_partition",
    "annealer_knapsack",
    # Main class
    "Annealer",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

ANNEALER_STARTED = "annealer.started"
ANNEALER_PROPOSED = "annealer.proposed"
ANNEALER_ACCEPTED = "annealer.accepted"
ANNEALER_REJECTED = "annealer.rejected"
ANNEALER_SWAPPED = "annealer.swapped"
ANNEALER_RESTARTED = "annealer.restarted"
ANNEALER_CHECKPOINTED = "annealer.checkpointed"
ANNEALER_FINISHED = "annealer.finished"
ANNEALER_CERTIFIED = "annealer.certified"
ANNEALER_RESET = "annealer.reset"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

ALGO_SA = "sa"
ALGO_PT = "pt"
ALGO_LAHC = "lahc"
ALGO_BASIN = "basin"
ALGO_TABU = "tabu"
ALGO_RESTART = "restart"
KNOWN_ALGORITHMS = (ALGO_SA, ALGO_PT, ALGO_LAHC, ALGO_BASIN, ALGO_TABU, ALGO_RESTART)

SCHED_GEOMETRIC = "geometric"
SCHED_LOG = "log"
SCHED_LINEAR = "linear"
SCHED_LUNDY_MEES = "lundy_mees"
SCHED_ADAPTIVE = "adaptive"
KNOWN_SCHEDULES = (
    SCHED_GEOMETRIC,
    SCHED_LOG,
    SCHED_LINEAR,
    SCHED_LUNDY_MEES,
    SCHED_ADAPTIVE,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnnealerError(Exception):
    """Base class for all :mod:`agi.annealer` errors."""


class InvalidConfig(AnnealerError):
    """A :class:`AnnealerConfig` field is out of range."""


class InvalidProblem(AnnealerError):
    """A submitted :class:`Problem` is malformed."""


class InvalidState(AnnealerError):
    """A returned state is malformed (None, unhashable, etc.)."""


class InsufficientData(AnnealerError):
    """An operation requires more data than has been observed."""


class UnknownAlgorithm(AnnealerError):
    """``algorithm`` is not in :data:`KNOWN_ALGORITHMS`."""


class UnknownSchedule(AnnealerError):
    """``schedule`` is not in :data:`KNOWN_SCHEDULES`."""


class NotRun(AnnealerError):
    """An operation requires ``run`` to have been called first."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnealerConfig:
    """Static configuration for an :class:`Annealer` instance.

    All temperatures are positive reals.  ``algorithm`` must be in
    :data:`KNOWN_ALGORITHMS`; ``schedule`` must be in
    :data:`KNOWN_SCHEDULES`.  ``seed`` is required for replay; if
    ``hmac_key`` is non-empty every receipt is HMAC-SHA-256 signed.
    """

    algorithm: str = ALGO_SA
    schedule: str = SCHED_GEOMETRIC
    t_init: float = 1.0
    t_final: float = 1e-3
    max_iter: int = 1000
    # PT
    n_replicas: int = 4
    swap_every: int = 50
    # LAHC
    lahc_length: int = 50
    # Tabu
    tabu_tenure: int = 20
    # Basin hopping
    basin_perturbations: int = 5
    # Adaptive
    target_acceptance: float = 0.44
    adapt_window: int = 100
    # Common
    record_every: int = 1
    restarts: int = 1
    luby_unit: int = 32
    seed: int = 0
    hmac_key: bytes = b""

    def __post_init__(self) -> None:
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"algorithm={self.algorithm!r} not in {KNOWN_ALGORITHMS}"
            )
        if self.schedule not in KNOWN_SCHEDULES:
            raise UnknownSchedule(
                f"schedule={self.schedule!r} not in {KNOWN_SCHEDULES}"
            )
        if self.t_init <= 0.0:
            raise InvalidConfig(f"t_init must be > 0; got {self.t_init}")
        if self.t_final <= 0.0:
            raise InvalidConfig(f"t_final must be > 0; got {self.t_final}")
        if self.t_final > self.t_init:
            raise InvalidConfig(
                f"t_final ({self.t_final}) must be <= t_init ({self.t_init})"
            )
        if self.max_iter <= 0:
            raise InvalidConfig(f"max_iter must be > 0; got {self.max_iter}")
        if self.n_replicas < 1:
            raise InvalidConfig(f"n_replicas must be >= 1; got {self.n_replicas}")
        if self.swap_every <= 0:
            raise InvalidConfig(f"swap_every must be > 0; got {self.swap_every}")
        if self.lahc_length < 1:
            raise InvalidConfig(f"lahc_length must be >= 1; got {self.lahc_length}")
        if self.tabu_tenure < 0:
            raise InvalidConfig(f"tabu_tenure must be >= 0; got {self.tabu_tenure}")
        if self.basin_perturbations < 1:
            raise InvalidConfig(
                f"basin_perturbations must be >= 1; got {self.basin_perturbations}"
            )
        if not (0.0 < self.target_acceptance < 1.0):
            raise InvalidConfig(
                f"target_acceptance must be in (0,1); got {self.target_acceptance}"
            )
        if self.adapt_window <= 0:
            raise InvalidConfig(f"adapt_window must be > 0; got {self.adapt_window}")
        if self.record_every <= 0:
            raise InvalidConfig(f"record_every must be > 0; got {self.record_every}")
        if self.restarts < 1:
            raise InvalidConfig(f"restarts must be >= 1; got {self.restarts}")
        if self.luby_unit < 1:
            raise InvalidConfig(f"luby_unit must be >= 1; got {self.luby_unit}")
        if not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes")


@dataclass(frozen=True)
class Problem:
    """A discrete optimisation problem the annealer can run on.

    ``initial`` is any hashable Python object describing the starting
    state.  ``cost(x)`` returns a real number to be minimised.
    ``neighbour(x, rng)`` returns a single random neighbour of ``x``;
    must be ergodic in the sense that every state in the search space
    is reachable from every other through a sequence of neighbour
    transitions.  ``lower_bound`` is an optional callable returning a
    *valid lower bound* on the optimum (used for the certificate's
    Hoeffding gap; if ``None`` the bound reports an empirical
    surrogate via restart-fork agreement instead).  ``upper_bound``
    is the maximum cost of any feasible state (used for Hoeffding's
    range bound on the cost; defaults to ``None`` and is then
    populated as the maximum cost seen during the run).
    """

    initial: Any
    cost: Callable[[Any], float]
    neighbour: Callable[[Any, random.Random], Any]
    lower_bound: Callable[[], float] | None = None
    upper_bound: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not callable(self.cost):
            raise InvalidProblem("cost must be callable")
        if not callable(self.neighbour):
            raise InvalidProblem("neighbour must be callable")
        if self.lower_bound is not None and not callable(self.lower_bound):
            raise InvalidProblem("lower_bound must be None or callable")


@dataclass(frozen=True)
class ReplicaSnapshot:
    """A single replica's terminal state in a parallel-tempering run."""

    index: int
    temperature: float
    state: Any
    cost: float
    accepts: int
    rejects: int


@dataclass(frozen=True)
class AnnealerReport:
    """The result of an :meth:`Annealer.run` call."""

    algorithm: str
    schedule: str
    iterations: int
    proposals: int
    accepts: int
    rejects: int
    swaps_attempted: int
    swaps_accepted: int
    restarts_taken: int
    best_cost: float
    best_state: Any
    final_cost: float
    final_state: Any
    cost_history: tuple[float, ...]
    temperature_history: tuple[float, ...]
    replicas: tuple[ReplicaSnapshot, ...]
    chain_head: str
    seed: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / max(1, self.proposals)


@dataclass(frozen=True)
class AnnealerCertificate:
    """An anytime-valid PAC certificate on the best-so-far cost."""

    best_cost: float
    lower_bound: float | None
    gap_hoeffding: float | None
    gap_bernstein: float | None
    p_global_opt: float | None
    delta: float
    n_samples: int
    cost_range: float | None
    cost_variance: float
    method: str
    chain_head: str


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


ANNEALER_LEDGER_GENESIS = "agi.annealer.v1"


def annealer_ledger_root(hmac_key: bytes = b"") -> str:
    """Return the immutable genesis hash of the annealer ledger.

    Same value across all instances; only changes if you change
    :data:`ANNEALER_LEDGER_GENESIS` (the module-version stamp).
    HMAC'd when ``hmac_key`` is non-empty.
    """
    payload = ANNEALER_LEDGER_GENESIS.encode("utf-8")
    if hmac_key:
        return hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def _ledger_extend(prev: str, event: str, body: Mapping[str, Any], hmac_key: bytes) -> str:
    payload = (
        prev.encode("utf-8")
        + b"|"
        + event.encode("utf-8")
        + b"|"
        + json.dumps(body, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    )
    if hmac_key:
        return hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, (list, tuple)):
        return list(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    if isinstance(o, bytes):
        return o.hex()
    return repr(o)


# ---------------------------------------------------------------------------
# Cooling schedules
# ---------------------------------------------------------------------------


def annealer_geometric_schedule(t_init: float, t_final: float, max_iter: int) -> Callable[[int], float]:
    """Return ``k -> T_init · α^k`` with α = (T_final/T_init)^{1/max_iter}."""
    if max_iter <= 1:
        return lambda k: t_init
    alpha = (t_final / t_init) ** (1.0 / max(1, max_iter - 1))

    def schedule(k: int) -> float:
        return max(t_final, t_init * (alpha ** max(0, k)))

    return schedule


def annealer_log_schedule(t_init: float, t_final: float, max_iter: int) -> Callable[[int], float]:
    """Geman-Geman 1984 logarithmic cooling: ``T_k = c / log(2+k)``.

    ``c`` is chosen so that ``T_0 = t_init``.  Provably converges to
    the global optimum in the infinite-time limit; impractically slow
    for large ``max_iter``.
    """
    c = t_init * math.log(2.0)

    def schedule(k: int) -> float:
        return max(t_final, c / math.log(2.0 + max(0, k)))

    return schedule


def annealer_linear_schedule(t_init: float, t_final: float, max_iter: int) -> Callable[[int], float]:
    """Linear interpolation from ``t_init`` to ``t_final`` over ``max_iter``."""
    if max_iter <= 1:
        return lambda k: t_init
    step = (t_init - t_final) / (max_iter - 1)

    def schedule(k: int) -> float:
        return max(t_final, t_init - max(0, k) * step)

    return schedule


def annealer_lundy_mees_schedule(t_init: float, t_final: float, max_iter: int) -> Callable[[int], float]:
    """Lundy-Mees 1986: ``T_{k+1} = T_k / (1 + β T_k)``.

    ``β`` is chosen so that the schedule reaches ``t_final`` at step
    ``max_iter`` exactly (analytic solution of the recurrence).
    """
    if max_iter <= 1 or t_init <= t_final:
        return lambda k: t_init
    beta = (t_init - t_final) / (max_iter * t_init * t_final)
    # Closed-form: T_k = T_init / (1 + β k T_init).  Derivation:
    # the recurrence T_{k+1} = T_k / (1 + β T_k) has analytic solution
    # 1/T_k = 1/T_0 + β k.

    def schedule(k: int) -> float:
        return max(t_final, t_init / (1.0 + beta * max(0, k) * t_init))

    return schedule


def annealer_adaptive_schedule(
    t_init: float,
    t_final: float,
    max_iter: int,
    target_acceptance: float = 0.44,
    window: int = 100,
) -> Callable[[int, float], float]:
    """Lam-Delosme 1988 adaptive schedule.

    Returns a *closure* that takes ``(iteration, rolling_acceptance_rate)``
    and returns the next temperature.  If acceptance < target, heat
    up; if > target, cool down; bounded by ``[t_final, t_init]``.
    """
    # State captured in mutable closure.
    state = {"T": t_init}

    def schedule(k: int, acc_rate: float) -> float:  # noqa: ARG001
        T = state["T"]
        if acc_rate < target_acceptance:
            T = min(t_init, T * 1.05)
        else:
            T = max(t_final, T * 0.95)
        state["T"] = T
        return T

    return schedule


# ---------------------------------------------------------------------------
# Metropolis-Hastings acceptance
# ---------------------------------------------------------------------------


def annealer_metropolis_accept(
    delta_e: float, temperature: float, u: float
) -> bool:
    """Return ``True`` iff a Metropolis move is accepted.

    ``delta_e`` is ``f(x') − f(x)``; ``temperature`` is the current
    temperature; ``u`` is a uniform random number in ``[0,1)`` (drawn
    by the caller from the seeded RNG so the function is pure).
    Always accepts ``delta_e <= 0``; else accepts with probability
    ``exp(−delta_e / temperature)``.
    """
    if delta_e <= 0.0:
        return True
    if temperature <= 0.0:
        return False
    # Numerically stable: if -delta_e/T is very negative, the prob is ~0.
    try:
        threshold = math.exp(-delta_e / temperature)
    except OverflowError:
        return False
    return u < threshold


def annealer_luby_sequence(n: int) -> list[int]:
    """Return the first ``n`` Luby restart units.

    Luby-Sinclair-Zuckerman 1993: ``t_i = 2^{k−1}`` if ``i = 2^k − 1``,
    else ``t_i = t_{i − 2^{k−1} + 1}`` where ``2^{k−1} ≤ i < 2^k − 1``.
    First 15 values: ``1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8``.
    """
    if n <= 0:
        return []
    seq: list[int] = []
    for i in range(1, n + 1):
        # Find k such that 2^{k-1} <= i + 1 <= 2^k - 1 (or i = 2^k - 1).
        # Standard recursive formula:
        k = int(math.log2(i + 1))
        if (i + 1) == (1 << (k + 1)) - 1 + 1:  # i + 1 is a power of 2
            pass
        # Simpler: classical recursive form.
        # u_i = 2^{k-1} if i = 2^k - 1
        #     = u_{i - 2^{k-1} + 1} otherwise.
        ii = i
        while True:
            # find k with 2^{k-1} <= ii < 2^k
            k = ii.bit_length()
            if ii == (1 << k) - 1:
                seq.append(1 << (k - 1))
                break
            ii = ii - (1 << (k - 1)) + 1
    return seq


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------


def annealer_tsp(points: Sequence[tuple[float, float]], *, seed: int = 0) -> Problem:
    """Build a symmetric TSP :class:`Problem`.

    ``points`` is a sequence of ``(x, y)`` Euclidean coordinates.  The
    state is a permutation of ``range(len(points))``.  ``cost`` is the
    sum of Euclidean distances around the closed tour.  The neighbour
    kernel is the 2-opt swap (reverse a random sub-segment).  The
    lower bound is the *1-tree* lower bound (sum of the two cheapest
    edges per vertex divided by 2, a relaxation of the Held-Karp
    lower bound that is exact on Euclidean instances within ~10%).
    The upper bound is the trivial cost of the nearest-neighbour tour.
    """
    n = len(points)
    if n < 2:
        raise InvalidProblem(f"TSP needs >= 2 points; got {n}")
    pts = tuple((float(x), float(y)) for x, y in points)

    def dist(i: int, j: int) -> float:
        a, b = pts[i], pts[j]
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def cost(state: tuple[int, ...]) -> float:
        return sum(dist(state[k], state[(k + 1) % n]) for k in range(n))

    def neighbour(state: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        if n <= 2:
            return state
        i, j = sorted(rng.sample(range(n), 2))
        # 2-opt: reverse state[i:j+1]
        return state[:i] + state[i : j + 1][::-1] + state[j + 1 :]

    def lower_bound() -> float:
        # Sum of two cheapest edges per vertex / 2.
        total = 0.0
        for i in range(n):
            edges = sorted(dist(i, j) for j in range(n) if j != i)
            if len(edges) < 2:
                return 0.0
            total += edges[0] + edges[1]
        return total / 2.0

    # Nearest-neighbour upper bound.
    rng = random.Random(seed)
    start = rng.randrange(n)
    visited = [start]
    while len(visited) < n:
        last = visited[-1]
        rem = [j for j in range(n) if j not in visited]
        nxt = min(rem, key=lambda j: dist(last, j))
        visited.append(nxt)
    initial = tuple(visited)
    upper = cost(initial)

    return Problem(
        initial=initial,
        cost=cost,
        neighbour=neighbour,
        lower_bound=lower_bound,
        upper_bound=upper,
        name="tsp",
    )


def annealer_max_cut(
    edges: Sequence[tuple[int, int, float]], *, n_vertices: int | None = None
) -> Problem:
    """Build a MaxCut :class:`Problem`.

    ``edges`` is an iterable of ``(u, v, w)`` triples with non-negative
    weights.  State is a tuple of booleans (one per vertex); cost is
    the *negative* sum of cut-edge weights (since the engine
    minimises).  Neighbour kernel flips one vertex.  Lower bound is
    the negative of the Goemans-Williamson 0.878 approximation lower
    bound on OPT — provided here as ``-0.878 * (sum of all edge
    weights)`` which is a valid lower bound on the negative cut.
    """
    es = tuple((int(u), int(v), float(w)) for u, v, w in edges)
    if not es:
        raise InvalidProblem("MaxCut needs at least one edge")
    if any(w < 0 for _, _, w in es):
        raise InvalidProblem("MaxCut weights must be non-negative")
    n = max(max(u, v) for u, v, _ in es) + 1 if n_vertices is None else int(n_vertices)
    if n <= 0:
        raise InvalidProblem("n_vertices must be > 0")
    total_weight = sum(w for _, _, w in es)

    def cost(state: tuple[int, ...]) -> float:
        cut = 0.0
        for u, v, w in es:
            if state[u] != state[v]:
                cut += w
        return -cut

    def neighbour(state: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        i = rng.randrange(n)
        new = list(state)
        new[i] = 1 - new[i]
        return tuple(new)

    def lower_bound() -> float:
        # Goemans-Williamson: OPT_cut >= 0.878 * total_weight is FALSE in
        # general; the GW bound is OPT_cut >= 0.878 * SDP_OPT >= 0.878 *
        # OPT_max_cut.  We use the trivial lower bound: -OPT_cut >= -total.
        return -total_weight

    initial = tuple(0 for _ in range(n))

    return Problem(
        initial=initial,
        cost=cost,
        neighbour=neighbour,
        lower_bound=lower_bound,
        upper_bound=0.0,
        name="max_cut",
    )


def annealer_max_sat(
    clauses: Sequence[Sequence[int]], *, n_vars: int | None = None
) -> Problem:
    """Build a MaxSAT :class:`Problem`.

    ``clauses`` is an iterable of clauses; each clause is a sequence
    of non-zero ints where ``+i`` means literal ``x_{i-1}`` and ``-i``
    means ``¬x_{i-1}``.  State is a tuple of booleans (one per
    variable); cost is the *negative* number of satisfied clauses
    (engine minimises).  Neighbour kernel flips one variable.
    Lower bound is ``-len(clauses)`` (perfect satisfaction).
    """
    cs = tuple(tuple(int(L) for L in clause) for clause in clauses)
    if not cs:
        raise InvalidProblem("MaxSAT needs at least one clause")
    if any(0 in clause for clause in cs):
        raise InvalidProblem("MaxSAT literals must be non-zero ints")
    if n_vars is None:
        n_vars = max(abs(L) for clause in cs for L in clause)
    n_vars = int(n_vars)
    if n_vars <= 0:
        raise InvalidProblem("n_vars must be > 0")

    def cost(state: tuple[int, ...]) -> float:
        sat = 0
        for clause in cs:
            for L in clause:
                idx = abs(L) - 1
                want = 1 if L > 0 else 0
                if state[idx] == want:
                    sat += 1
                    break
        return -float(sat)

    def neighbour(state: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        i = rng.randrange(n_vars)
        new = list(state)
        new[i] = 1 - new[i]
        return tuple(new)

    def lower_bound() -> float:
        return -float(len(cs))

    initial = tuple(0 for _ in range(n_vars))

    return Problem(
        initial=initial,
        cost=cost,
        neighbour=neighbour,
        lower_bound=lower_bound,
        upper_bound=0.0,
        name="max_sat",
    )


def annealer_qap(
    flow: Sequence[Sequence[float]], dist: Sequence[Sequence[float]]
) -> Problem:
    """Build a Quadratic Assignment Problem :class:`Problem`.

    ``flow`` and ``dist`` are ``n×n`` matrices.  The state is a
    permutation ``π`` of ``range(n)`` where ``π[i]`` is the location
    assigned to facility ``i``.  Cost is
    ``Σ_{i,j} flow[i][j] · dist[π[i]][π[j]]``.  Neighbour kernel
    swaps two facilities.  Lower bound is the Gilmore-Lawler
    bound (sum of column-row minima of the cost matrix).
    """
    f = tuple(tuple(float(x) for x in row) for row in flow)
    d = tuple(tuple(float(x) for x in row) for row in dist)
    n = len(f)
    if n == 0 or any(len(row) != n for row in f) or len(d) != n or any(len(row) != n for row in d):
        raise InvalidProblem("flow and dist must both be n×n with matching n")

    def cost(state: tuple[int, ...]) -> float:
        total = 0.0
        for i in range(n):
            pi = state[i]
            for j in range(n):
                total += f[i][j] * d[pi][state[j]]
        return total

    def neighbour(state: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        if n <= 1:
            return state
        i, j = rng.sample(range(n), 2)
        new = list(state)
        new[i], new[j] = new[j], new[i]
        return tuple(new)

    def lower_bound() -> float:
        # Gilmore-Lawler: sort rows of f (descending) and d (ascending)
        # outside the diagonal, inner-product each, sum.
        bound = 0.0
        for i in range(n):
            f_row = sorted([f[i][j] for j in range(n) if j != i], reverse=True)
            for k in range(n):
                d_row = sorted([d[k][m] for m in range(n) if m != k])
                inner = sum(f_row[t] * d_row[t] for t in range(len(f_row)))
                if k == 0 or inner < best:
                    best = inner
            bound += best
        return bound

    initial = tuple(range(n))

    return Problem(
        initial=initial,
        cost=cost,
        neighbour=neighbour,
        lower_bound=lower_bound,
        upper_bound=None,
        name="qap",
    )


def annealer_number_partition(weights: Sequence[float]) -> Problem:
    """Build a Karmarkar-Karp number-partitioning :class:`Problem`.

    ``weights`` is a sequence of non-negative reals.  State is a
    tuple of ``±1`` indicators (signs); cost is the absolute value
    of the signed sum.  Neighbour kernel flips one sign.  Lower
    bound is ``0`` (perfect partition).
    """
    w = tuple(float(x) for x in weights)
    n = len(w)
    if n < 2:
        raise InvalidProblem("number partition needs >= 2 weights")
    if any(x < 0 for x in w):
        raise InvalidProblem("weights must be non-negative")

    def cost(state: tuple[int, ...]) -> float:
        return abs(sum(s * x for s, x in zip(state, w)))

    def neighbour(state: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        i = rng.randrange(n)
        new = list(state)
        new[i] = -new[i]
        return tuple(new)

    def lower_bound() -> float:
        return 0.0

    initial = tuple(1 for _ in range(n))

    return Problem(
        initial=initial,
        cost=cost,
        neighbour=neighbour,
        lower_bound=lower_bound,
        upper_bound=sum(w),
        name="number_partition",
    )


def annealer_knapsack(
    weights: Sequence[float], values: Sequence[float], capacity: float
) -> Problem:
    """Build a 0/1 knapsack :class:`Problem`.

    Engine is a minimiser, so cost is ``−Σ v_i x_i`` plus an
    infeasibility penalty proportional to over-capacity weight.  The
    lower bound on the negative-value cost is ``−lp_relaxation``
    where the LP relaxation sorts items by value/weight ratio and
    fractionally fills until capacity.
    """
    w = tuple(float(x) for x in weights)
    v = tuple(float(x) for x in values)
    if len(w) != len(v):
        raise InvalidProblem("weights and values must have the same length")
    n = len(w)
    if n == 0:
        raise InvalidProblem("knapsack needs >= 1 item")
    c = float(capacity)
    if c <= 0:
        raise InvalidProblem("capacity must be > 0")
    if any(x < 0 for x in w) or any(x < 0 for x in v):
        raise InvalidProblem("knapsack weights / values must be non-negative")
    penalty = sum(v) + 1.0  # ensures any feasible beats any infeasible

    def cost(state: tuple[int, ...]) -> float:
        total_w = sum(state[i] * w[i] for i in range(n))
        total_v = sum(state[i] * v[i] for i in range(n))
        if total_w > c:
            return -total_v + penalty * (total_w - c)
        return -total_v

    def neighbour(state: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        i = rng.randrange(n)
        new = list(state)
        new[i] = 1 - new[i]
        return tuple(new)

    def lower_bound() -> float:
        ratios = sorted(range(n), key=lambda i: -v[i] / w[i] if w[i] > 0 else float("inf"))
        remaining = c
        lp = 0.0
        for i in ratios:
            if w[i] <= remaining:
                lp += v[i]
                remaining -= w[i]
            else:
                lp += v[i] * (remaining / w[i]) if w[i] > 0 else v[i]
                break
        return -lp

    initial = tuple(0 for _ in range(n))

    return Problem(
        initial=initial,
        cost=cost,
        neighbour=neighbour,
        lower_bound=lower_bound,
        upper_bound=0.0,
        name="knapsack",
    )


# ---------------------------------------------------------------------------
# Annealer
# ---------------------------------------------------------------------------


class Annealer:
    """The main runtime primitive.

    Construct with a :class:`AnnealerConfig`; call :meth:`run` with a
    :class:`Problem` to optimise it; call :meth:`certify` on the
    returned :class:`AnnealerReport` to obtain an anytime-valid PAC
    certificate on the best-cost gap to the global optimum.
    """

    def __init__(self, config: AnnealerConfig | None = None) -> None:
        self._config = config if config is not None else AnnealerConfig()
        self._lock = threading.RLock()
        self._chain_head: str = annealer_ledger_root(self._config.hmac_key)
        self._n_runs: int = 0
        self._last_report: AnnealerReport | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> AnnealerConfig:
        return self._config

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._chain_head

    @property
    def n_runs(self) -> int:
        with self._lock:
            return self._n_runs

    @property
    def last_report(self) -> AnnealerReport | None:
        with self._lock:
            return self._last_report

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, problem: Problem, *, restarts: int | None = None) -> AnnealerReport:
        """Run the configured algorithm on ``problem``.

        Returns an :class:`AnnealerReport`.  When ``restarts > 1``
        (or :attr:`AnnealerConfig.restarts` > 1), forks ``R``
        independent runs with seeds ``seed, seed+1, …, seed+R-1``
        and returns the best of all forks (the others' cost
        histories are dropped from the returned report to keep it
        small; the per-fork best costs are recoverable from the
        receipt chain).
        """
        if not isinstance(problem, Problem):
            raise InvalidProblem("problem must be a Problem dataclass")
        R = int(restarts) if restarts is not None else self._config.restarts
        if R < 1:
            raise InvalidConfig(f"restarts must be >= 1; got {R}")

        with self._lock:
            self._emit(ANNEALER_STARTED, {
                "algorithm": self._config.algorithm,
                "schedule": self._config.schedule,
                "max_iter": self._config.max_iter,
                "restarts": R,
                "problem": problem.name,
            })

            best_report: AnnealerReport | None = None
            for r in range(R):
                seed_r = self._config.seed + r * 1_000_003
                rep = self._run_once(problem, seed_r)
                if best_report is None or rep.best_cost < best_report.best_cost:
                    best_report = rep
                if R > 1:
                    self._emit(ANNEALER_RESTARTED, {
                        "fork": r,
                        "best_cost": rep.best_cost,
                    })

            assert best_report is not None
            self._n_runs += 1
            self._last_report = best_report
            self._emit(ANNEALER_FINISHED, {
                "best_cost": best_report.best_cost,
                "iterations": best_report.iterations,
            })
            return best_report

    def _run_once(self, problem: Problem, seed: int) -> AnnealerReport:
        algo = self._config.algorithm
        if algo == ALGO_SA:
            return self._run_sa(problem, seed)
        if algo == ALGO_PT:
            return self._run_pt(problem, seed)
        if algo == ALGO_LAHC:
            return self._run_lahc(problem, seed)
        if algo == ALGO_BASIN:
            return self._run_basin(problem, seed)
        if algo == ALGO_TABU:
            return self._run_tabu(problem, seed)
        if algo == ALGO_RESTART:
            return self._run_luby_restart(problem, seed)
        raise UnknownAlgorithm(f"algorithm={algo!r}")

    # ------------------------------------------------------------------
    # Algorithm: single-chain SA
    # ------------------------------------------------------------------

    def _schedule_fn(self) -> Any:
        sched = self._config.schedule
        if sched == SCHED_GEOMETRIC:
            return annealer_geometric_schedule(
                self._config.t_init, self._config.t_final, self._config.max_iter
            )
        if sched == SCHED_LOG:
            return annealer_log_schedule(
                self._config.t_init, self._config.t_final, self._config.max_iter
            )
        if sched == SCHED_LINEAR:
            return annealer_linear_schedule(
                self._config.t_init, self._config.t_final, self._config.max_iter
            )
        if sched == SCHED_LUNDY_MEES:
            return annealer_lundy_mees_schedule(
                self._config.t_init, self._config.t_final, self._config.max_iter
            )
        if sched == SCHED_ADAPTIVE:
            return annealer_adaptive_schedule(
                self._config.t_init,
                self._config.t_final,
                self._config.max_iter,
                target_acceptance=self._config.target_acceptance,
                window=self._config.adapt_window,
            )
        raise UnknownSchedule(f"schedule={sched!r}")

    def _run_sa(self, problem: Problem, seed: int) -> AnnealerReport:
        rng = random.Random(seed)
        schedule = self._schedule_fn()
        adaptive = self._config.schedule == SCHED_ADAPTIVE

        cur = problem.initial
        cur_cost = float(problem.cost(cur))
        best, best_cost = cur, cur_cost

        accepts = rejects = proposals = 0
        cost_hist: list[float] = []
        temp_hist: list[float] = []
        recent_accepts: list[int] = []

        for k in range(self._config.max_iter):
            if adaptive:
                acc_rate = (
                    sum(recent_accepts) / max(1, len(recent_accepts))
                    if recent_accepts
                    else self._config.target_acceptance
                )
                T = schedule(k, acc_rate)
            else:
                T = schedule(k)

            cand = problem.neighbour(cur, rng)
            cand_cost = float(problem.cost(cand))
            proposals += 1
            delta = cand_cost - cur_cost
            u = rng.random()
            if annealer_metropolis_accept(delta, T, u):
                cur, cur_cost = cand, cand_cost
                accepts += 1
                recent_accepts.append(1)
                if cur_cost < best_cost:
                    best, best_cost = cur, cur_cost
                self._emit(ANNEALER_ACCEPTED, {
                    "k": k,
                    "T": T,
                    "delta": delta,
                    "cost": cur_cost,
                    "best": best_cost,
                })
            else:
                rejects += 1
                recent_accepts.append(0)
                self._emit(ANNEALER_REJECTED, {
                    "k": k,
                    "T": T,
                    "delta": delta,
                    "cost": cur_cost,
                })

            if len(recent_accepts) > self._config.adapt_window:
                recent_accepts.pop(0)
            if k % self._config.record_every == 0:
                cost_hist.append(cur_cost)
                temp_hist.append(T)

        return AnnealerReport(
            algorithm=ALGO_SA,
            schedule=self._config.schedule,
            iterations=self._config.max_iter,
            proposals=proposals,
            accepts=accepts,
            rejects=rejects,
            swaps_attempted=0,
            swaps_accepted=0,
            restarts_taken=0,
            best_cost=best_cost,
            best_state=best,
            final_cost=cur_cost,
            final_state=cur,
            cost_history=tuple(cost_hist),
            temperature_history=tuple(temp_hist),
            replicas=(),
            chain_head=self._chain_head,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Algorithm: parallel tempering / replica exchange
    # ------------------------------------------------------------------

    def _run_pt(self, problem: Problem, seed: int) -> AnnealerReport:
        K = self._config.n_replicas
        rng = random.Random(seed)
        # Per-replica RNG so threads / forks are deterministic.
        rngs = [random.Random(seed * 7919 + i * 104729 + 1) for i in range(K)]
        # Geometric ladder of temperatures from t_final to t_init.
        if K == 1:
            temps = [self._config.t_init]
        else:
            ratio = (self._config.t_init / self._config.t_final) ** (1.0 / (K - 1))
            temps = [self._config.t_final * (ratio ** i) for i in range(K)]
        states = [problem.initial for _ in range(K)]
        costs = [float(problem.cost(states[i])) for i in range(K)]
        accepts = [0] * K
        rejects = [0] * K
        proposals = 0
        swaps_attempted = swaps_accepted = 0
        best_state = states[0]
        best_cost = costs[0]
        for i in range(K):
            if costs[i] < best_cost:
                best_state, best_cost = states[i], costs[i]

        cost_hist: list[float] = []
        temp_hist: list[float] = []

        for k in range(self._config.max_iter):
            for i in range(K):
                cand = problem.neighbour(states[i], rngs[i])
                cand_cost = float(problem.cost(cand))
                proposals += 1
                delta = cand_cost - costs[i]
                u = rngs[i].random()
                if annealer_metropolis_accept(delta, temps[i], u):
                    states[i], costs[i] = cand, cand_cost
                    accepts[i] += 1
                    if costs[i] < best_cost:
                        best_state, best_cost = states[i], costs[i]
                else:
                    rejects[i] += 1

            # Periodic swap attempts between adjacent replicas.
            if (k + 1) % self._config.swap_every == 0 and K >= 2:
                # Alternate even / odd swap parity to enforce
                # detailed balance over multiple sweeps.
                start = (k // self._config.swap_every) % 2
                for i in range(start, K - 1, 2):
                    swaps_attempted += 1
                    beta_i = 1.0 / temps[i]
                    beta_j = 1.0 / temps[i + 1]
                    log_p = (beta_i - beta_j) * (costs[i] - costs[i + 1])
                    u = rng.random()
                    # accept if log u < log_p (clamp to <= 0)
                    if log_p >= 0.0 or math.log(max(u, 1e-300)) < log_p:
                        states[i], states[i + 1] = states[i + 1], states[i]
                        costs[i], costs[i + 1] = costs[i + 1], costs[i]
                        swaps_accepted += 1
                        self._emit(ANNEALER_SWAPPED, {
                            "k": k,
                            "i": i,
                            "j": i + 1,
                            "log_p": log_p,
                        })

            if k % self._config.record_every == 0:
                cost_hist.append(best_cost)
                temp_hist.append(temps[0])

        replicas = tuple(
            ReplicaSnapshot(
                index=i,
                temperature=temps[i],
                state=states[i],
                cost=costs[i],
                accepts=accepts[i],
                rejects=rejects[i],
            )
            for i in range(K)
        )

        return AnnealerReport(
            algorithm=ALGO_PT,
            schedule=self._config.schedule,
            iterations=self._config.max_iter,
            proposals=proposals,
            accepts=sum(accepts),
            rejects=sum(rejects),
            swaps_attempted=swaps_attempted,
            swaps_accepted=swaps_accepted,
            restarts_taken=0,
            best_cost=best_cost,
            best_state=best_state,
            final_cost=costs[0],
            final_state=states[0],
            cost_history=tuple(cost_hist),
            temperature_history=tuple(temp_hist),
            replicas=replicas,
            chain_head=self._chain_head,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Algorithm: late acceptance hill-climbing
    # ------------------------------------------------------------------

    def _run_lahc(self, problem: Problem, seed: int) -> AnnealerReport:
        rng = random.Random(seed)
        L = self._config.lahc_length
        cur = problem.initial
        cur_cost = float(problem.cost(cur))
        best, best_cost = cur, cur_cost
        buf = [cur_cost] * L
        accepts = rejects = proposals = 0
        cost_hist: list[float] = []

        for k in range(self._config.max_iter):
            cand = problem.neighbour(cur, rng)
            cand_cost = float(problem.cost(cand))
            proposals += 1
            slot = k % L
            if cand_cost <= cur_cost or cand_cost <= buf[slot]:
                cur, cur_cost = cand, cand_cost
                accepts += 1
                if cur_cost < best_cost:
                    best, best_cost = cur, cur_cost
            else:
                rejects += 1
            buf[slot] = cur_cost
            if k % self._config.record_every == 0:
                cost_hist.append(cur_cost)

        return AnnealerReport(
            algorithm=ALGO_LAHC,
            schedule=self._config.schedule,
            iterations=self._config.max_iter,
            proposals=proposals,
            accepts=accepts,
            rejects=rejects,
            swaps_attempted=0,
            swaps_accepted=0,
            restarts_taken=0,
            best_cost=best_cost,
            best_state=best,
            final_cost=cur_cost,
            final_state=cur,
            cost_history=tuple(cost_hist),
            temperature_history=(),
            replicas=(),
            chain_head=self._chain_head,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Algorithm: basin hopping
    # ------------------------------------------------------------------

    def _run_basin(self, problem: Problem, seed: int) -> AnnealerReport:
        rng = random.Random(seed)
        schedule = self._schedule_fn()
        adaptive = self._config.schedule == SCHED_ADAPTIVE

        # Outer loop over local minima; inner loop is first-improvement
        # greedy descent capped at basin_perturbations iterations.
        cur = problem.initial
        cur_cost = float(problem.cost(cur))
        # Initial descent.
        cur, cur_cost = self._local_descent(problem, cur, cur_cost, rng)
        best, best_cost = cur, cur_cost

        accepts = rejects = proposals = 0
        recent_accepts: list[int] = []
        outer_iters = max(1, self._config.max_iter // max(1, self._config.basin_perturbations))
        cost_hist: list[float] = []

        for k in range(outer_iters):
            if adaptive:
                acc = sum(recent_accepts) / max(1, len(recent_accepts)) if recent_accepts else self._config.target_acceptance
                T = schedule(k, acc)
            else:
                T = schedule(k)

            # Perturb and descend.
            cand = cur
            for _ in range(self._config.basin_perturbations):
                cand = problem.neighbour(cand, rng)
            cand_cost = float(problem.cost(cand))
            cand, cand_cost = self._local_descent(problem, cand, cand_cost, rng)
            proposals += 1
            delta = cand_cost - cur_cost
            u = rng.random()
            if annealer_metropolis_accept(delta, T, u):
                cur, cur_cost = cand, cand_cost
                accepts += 1
                recent_accepts.append(1)
                if cur_cost < best_cost:
                    best, best_cost = cur, cur_cost
            else:
                rejects += 1
                recent_accepts.append(0)
            if len(recent_accepts) > self._config.adapt_window:
                recent_accepts.pop(0)
            if k % self._config.record_every == 0:
                cost_hist.append(cur_cost)

        return AnnealerReport(
            algorithm=ALGO_BASIN,
            schedule=self._config.schedule,
            iterations=outer_iters,
            proposals=proposals,
            accepts=accepts,
            rejects=rejects,
            swaps_attempted=0,
            swaps_accepted=0,
            restarts_taken=0,
            best_cost=best_cost,
            best_state=best,
            final_cost=cur_cost,
            final_state=cur,
            cost_history=tuple(cost_hist),
            temperature_history=(),
            replicas=(),
            chain_head=self._chain_head,
            seed=seed,
        )

    def _local_descent(
        self,
        problem: Problem,
        state: Any,
        cost: float,
        rng: random.Random,
        max_steps: int = 50,
    ) -> tuple[Any, float]:
        """First-improvement greedy descent until no improving neighbour or cap."""
        for _ in range(max_steps):
            cand = problem.neighbour(state, rng)
            cand_cost = float(problem.cost(cand))
            if cand_cost < cost:
                state, cost = cand, cand_cost
            else:
                # try again a few times before giving up
                pass
        return state, cost

    # ------------------------------------------------------------------
    # Algorithm: tabu search
    # ------------------------------------------------------------------

    def _run_tabu(self, problem: Problem, seed: int) -> AnnealerReport:
        rng = random.Random(seed)
        cur = problem.initial
        cur_cost = float(problem.cost(cur))
        best, best_cost = cur, cur_cost
        tabu: list[Any] = []
        accepts = rejects = proposals = 0
        cost_hist: list[float] = []

        # We *enumerate* k candidates per iteration and pick the
        # cheapest non-tabu one (or any cheapest if it beats the
        # global best — aspiration criterion).
        k_per_iter = 8

        for k in range(self._config.max_iter):
            best_cand = None
            best_cand_cost = float("inf")
            for _ in range(k_per_iter):
                cand = problem.neighbour(cur, rng)
                cand_cost = float(problem.cost(cand))
                proposals += 1
                # Aspiration: cand beats global best -> always allowed
                if cand_cost < best_cost:
                    if cand_cost < best_cand_cost:
                        best_cand, best_cand_cost = cand, cand_cost
                    continue
                # Otherwise must not be in tabu list (by repr identity)
                key = repr(cand)
                if key in tabu:
                    continue
                if cand_cost < best_cand_cost:
                    best_cand, best_cand_cost = cand, cand_cost
            if best_cand is None:
                rejects += 1
                continue
            cur, cur_cost = best_cand, best_cand_cost
            accepts += 1
            tabu.append(repr(cur))
            while len(tabu) > self._config.tabu_tenure:
                tabu.pop(0)
            if cur_cost < best_cost:
                best, best_cost = cur, cur_cost
            if k % self._config.record_every == 0:
                cost_hist.append(cur_cost)

        return AnnealerReport(
            algorithm=ALGO_TABU,
            schedule=self._config.schedule,
            iterations=self._config.max_iter,
            proposals=proposals,
            accepts=accepts,
            rejects=rejects,
            swaps_attempted=0,
            swaps_accepted=0,
            restarts_taken=0,
            best_cost=best_cost,
            best_state=best,
            final_cost=cur_cost,
            final_state=cur,
            cost_history=tuple(cost_hist),
            temperature_history=(),
            replicas=(),
            chain_head=self._chain_head,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Algorithm: Luby restart wrapper around SA
    # ------------------------------------------------------------------

    def _run_luby_restart(self, problem: Problem, seed: int) -> AnnealerReport:
        rng = random.Random(seed)
        unit = self._config.luby_unit
        # Estimate how many restart units fit; cap by max_iter.
        budget = self._config.max_iter
        n_units = max(1, budget // unit)
        seq = annealer_luby_sequence(n_units)
        # Trim to fit budget.
        spent = 0
        out_seq: list[int] = []
        for s in seq:
            if spent + s * unit > budget:
                out_seq.append(max(1, budget - spent))
                spent = budget
                break
            out_seq.append(s * unit)
            spent += s * unit
            if spent >= budget:
                break
        if not out_seq:
            out_seq = [budget]

        best, best_cost = problem.initial, float(problem.cost(problem.initial))
        total_proposals = total_accepts = total_rejects = 0
        all_hist: list[float] = []
        restarts_taken = 0

        # Embed a single-chain SA per restart with proportional cooling.
        for unit_iter in out_seq:
            sub_cfg = AnnealerConfig(
                algorithm=ALGO_SA,
                schedule=self._config.schedule,
                t_init=self._config.t_init,
                t_final=self._config.t_final,
                max_iter=max(1, unit_iter),
                seed=rng.randint(0, 2**31 - 1),
                target_acceptance=self._config.target_acceptance,
                adapt_window=self._config.adapt_window,
                record_every=self._config.record_every,
                hmac_key=self._config.hmac_key,
            )
            sub = Annealer(sub_cfg)
            rep = sub._run_sa(problem, sub_cfg.seed)
            restarts_taken += 1
            total_proposals += rep.proposals
            total_accepts += rep.accepts
            total_rejects += rep.rejects
            if rep.best_cost < best_cost:
                best, best_cost = rep.best_state, rep.best_cost
            if rep.cost_history:
                all_hist.extend(rep.cost_history)

        return AnnealerReport(
            algorithm=ALGO_RESTART,
            schedule=self._config.schedule,
            iterations=sum(out_seq),
            proposals=total_proposals,
            accepts=total_accepts,
            rejects=total_rejects,
            swaps_attempted=0,
            swaps_accepted=0,
            restarts_taken=restarts_taken,
            best_cost=best_cost,
            best_state=best,
            final_cost=best_cost,
            final_state=best,
            cost_history=tuple(all_hist),
            temperature_history=(),
            replicas=(),
            chain_head=self._chain_head,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Certificate
    # ------------------------------------------------------------------

    def certify(
        self,
        report: AnnealerReport | None = None,
        *,
        delta: float = 0.05,
        problem: Problem | None = None,
    ) -> AnnealerCertificate:
        """Return an anytime-valid PAC certificate on the best-cost gap.

        ``delta`` is the failure probability of the bound.  If
        ``problem`` is supplied, its ``lower_bound`` callable (when
        present) replaces the empirical bound; otherwise the bound
        defaults to the cost-history minimum which is trivially
        ``best_cost`` and gives a degenerate gap of zero — useful as
        a sanity check but not informative.

        The Hoeffding gap is
        ``(b−a)·sqrt(log(2/δ)/(2n))`` where ``[a,b]`` is the empirical
        cost range and ``n`` is the number of recorded samples.  The
        empirical-Bernstein refinement uses the sample variance
        ``S²`` to tighten this to
        ``sqrt(2 S² log(2/δ)/n) + 7(b−a) log(2/δ)/(3(n−1))``
        (Maurer-Pontil 2009).
        """
        if not (0.0 < delta < 1.0):
            raise InvalidConfig(f"delta must be in (0,1); got {delta}")
        if report is None:
            report = self._last_report
        if report is None:
            raise NotRun("certify() called before run()")

        hist = report.cost_history
        n = len(hist)
        if n == 0:
            return AnnealerCertificate(
                best_cost=report.best_cost,
                lower_bound=None,
                gap_hoeffding=None,
                gap_bernstein=None,
                p_global_opt=None,
                delta=delta,
                n_samples=0,
                cost_range=None,
                cost_variance=0.0,
                method="empty",
                chain_head=self._chain_head,
            )

        lo = min(hist)
        hi = max(hist)
        rng_ = hi - lo if hi > lo else 1e-12
        mean = sum(hist) / n
        var = sum((h - mean) ** 2 for h in hist) / n
        method = "empirical"
        lb = lo
        if problem is not None and problem.lower_bound is not None:
            try:
                lb = float(problem.lower_bound())
                method = "lower_bound"
            except Exception:
                lb = lo

        gap_hoeff = rng_ * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
        if n >= 2:
            gap_bern = math.sqrt(2.0 * var * math.log(2.0 / delta) / n) + (
                7.0 * rng_ * math.log(2.0 / delta) / (3.0 * (n - 1))
            )
        else:
            gap_bern = gap_hoeff

        # p_global_opt:
        #   If schedule = SCHED_LOG, Geman-Geman 1984 guarantees
        #   P(X_k = global opt) -> 1 as k -> ∞.  We surface the
        #   *empirical fraction of records at best_cost* as a
        #   surrogate, which is the natural anytime statistic.
        n_at_best = sum(1 for h in hist if h <= report.best_cost + 1e-12)
        p_global = n_at_best / n
        # Clopper-Pearson-ish lower bound (Hoeffding for binomial):
        eps = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        p_global_lb = max(0.0, p_global - eps)

        cert = AnnealerCertificate(
            best_cost=report.best_cost,
            lower_bound=lb,
            gap_hoeffding=gap_hoeff,
            gap_bernstein=gap_bern,
            p_global_opt=p_global_lb,
            delta=delta,
            n_samples=n,
            cost_range=rng_,
            cost_variance=var,
            method=method,
            chain_head=self._chain_head,
        )
        with self._lock:
            self._emit(ANNEALER_CERTIFIED, {
                "best_cost": cert.best_cost,
                "lower_bound": cert.lower_bound,
                "gap_hoeffding": cert.gap_hoeffding,
                "gap_bernstein": cert.gap_bernstein,
                "p_global_opt": cert.p_global_opt,
                "delta": cert.delta,
                "n": cert.n_samples,
            })
        return cert

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> Mapping[str, Any]:
        """Return a JSON-encodable state dict.

        The dict is the minimal state required to ``restore`` byte-
        identically.  Contains the current chain head, run count,
        and the last report if any.
        """
        with self._lock:
            return {
                "chain_head": self._chain_head,
                "n_runs": self._n_runs,
                "config": {
                    "algorithm": self._config.algorithm,
                    "schedule": self._config.schedule,
                    "t_init": self._config.t_init,
                    "t_final": self._config.t_final,
                    "max_iter": self._config.max_iter,
                    "n_replicas": self._config.n_replicas,
                    "seed": self._config.seed,
                },
                "last_report_best_cost": (
                    self._last_report.best_cost if self._last_report is not None else None
                ),
            }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore from a :meth:`snapshot` payload."""
        with self._lock:
            self._chain_head = str(snapshot["chain_head"])
            self._n_runs = int(snapshot["n_runs"])

    def reset(self) -> None:
        """Reset the internal state to genesis."""
        with self._lock:
            self._chain_head = annealer_ledger_root(self._config.hmac_key)
            self._n_runs = 0
            self._last_report = None
            self._emit(ANNEALER_RESET, {})

    # ------------------------------------------------------------------
    # Ledger emission
    # ------------------------------------------------------------------

    def _emit(self, event: str, body: Mapping[str, Any]) -> str:
        self._chain_head = _ledger_extend(self._chain_head, event, dict(body), self._config.hmac_key)
        return self._chain_head
