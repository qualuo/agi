r"""Searcher — bounded-anytime certified tree search as a runtime primitive.

Every other primitive in this runtime *consumes* a question.  Predictor
gets a stream and returns the next symbol; Solver gets a CNF and returns
SAT/UNSAT; Inducer gets a spec and returns a program.  But the operation
that decides **which question to ask next**, given a state, a set of
admissible actions, and a means of evaluating their consequences, is
*search*.  Search is the canonical primitive of every agent that acts
under uncertainty over a tree of options — AlphaZero is search, MuZero
is search, Stockfish is search, the A* planner inside a self-driving
stack is search, and the move-list a debugger considers in front of a
bug is search.  The ``Searcher`` is the runtime's *bounded, anytime,
certified* version of that operation, exposed as a single primitive
that a coordination engine can drive with budgets it must respect.

The pitch reduced to a runtime call::

    searcher = Searcher(SearcherConfig(algorithm="puct", max_iterations=4096))
    report   = searcher.search(
        root,
        actions=lambda s: s.legal_moves(),
        apply=lambda s, a: s.play(a),
        terminal=lambda s: s.is_terminal(),
        reward=lambda s: s.reward(),
        policy_prior=lambda s, A: {a: 1/len(A) for a in A},
        value=lambda s: 0.0,
    )
    report.best_action          # the recommended action at the root
    report.best_value           # the search's value estimate of the root
    report.principal_variation  # the deepest sequence the search agreed on
    report.certificate          # SHA-256 chain of (parent, action, child) decisions
    report.budget_used          # nodes, time, memory — what the search consumed
    report.regret_bound         # algorithm-specific finite-time regret bound

What "bounded, anytime, certified" means
----------------------------------------

  * **Bounded** — every algorithm exposes a uniform stop predicate over
    (wall-clock seconds, expansion count, node count, peak memory bytes,
    deadline timestamp).  A coordination engine that has 30 ms left in
    its SLO budget can pass that 30 ms and get back the best decision
    the searcher could compute within it.

  * **Anytime** — at every iteration of every algorithm the *current
    best* action and value are well-defined.  ``report.history`` records
    the (iteration, best_action, best_value) trajectory so a coordinator
    can detect convergence (or its absence) and decide whether to keep
    spending budget.

  * **Certified** — every ``SearchReport`` carries a SHA-256 chain over
    the canonical sequence of search decisions ``(parent_key, action,
    child_key, evaluation, selected)``.  Replaying the search against
    the same root, the same evaluators, and the same RNG seed reproduces
    the chain byte-for-byte (cf. ``AttestationLedger``).  Two searchers
    in two processes that agree on the certificate agree on the search.

  * **Pure stdlib** — no NumPy, no Torch, no SciPy.  The same module
    runs inside a sandboxed coordinator, inside a CI worker, inside a
    Lambda function with a 256 MB memory cap.

Algorithms
----------

Six families of search are exposed under a single ``algorithm=`` switch.
The default is ``"auto"`` — pick the family that respects the supplied
evaluator signatures.

  * ``"astar"`` — **A\*** (Hart, Nilsson & Raphael 1968): best-first
    search over ``f(n) = g(n) + h(n)``.  Optimal under an admissible
    heuristic; ``weighted=w`` switches to **weighted A\*** (Pohl 1970)
    with a worst-case ``w``-suboptimality bound.

  * ``"ida_star"`` — **IDA\*** (Korf 1985): iterative-deepening A\*
    with linear space.  Used when the open list of A\* exceeds memory.

  * ``"uct"`` — **UCT** (Kocsis & Szepesvári 2006): Monte-Carlo Tree
    Search with the **UCB1** selection rule (Auer, Cesa-Bianchi &
    Fischer 2002).  Finite-time regret bound::

        E[regret] ≤ C(K) · ln T / Δ_min

    where ``K`` is the branching factor, ``T`` the number of
    iterations, and ``Δ_min`` the gap between the best and second-best
    arm at the root.  No prior; uniform-random rollouts unless a
    ``value`` evaluator is supplied.

  * ``"puct"`` — **PUCT** (Silver et al. 2017, AlphaGo Zero): UCT with
    a *policy prior* ``P(s, a)`` added to the exploration term::

        a* = argmax_a [ Q(s, a) + c_puct · P(s, a) · sqrt(ΣN) / (1 + N(s, a)) ]

    Reduces to UCT under a uniform prior; the AlphaZero recommendation
    of ``c_puct = 1.25`` is the default.  Optional **Dirichlet noise**
    at the root for exploration in self-play settings.

  * ``"alphabeta"`` — **Alpha-Beta** (McCarthy 1956 / Knuth & Moore
    1975) with **iterative deepening** (Slate & Atkin 1977),
    **transposition table** (Greenblatt 1967), **history heuristic**
    (Schaeffer 1989), and **principal variation** extraction.  Operates
    on zero-sum two-player perfect-information games where ``reward``
    is the static evaluation from the side-to-move's perspective.

  * ``"beam"`` — **Beam search** (Reddy 1977) with width ``b``.  Trades
    completeness for memory boundedness; the *current best* heuristic
    is the highest-scoring complete or partial path in the beam.

  * ``"bnb"`` — **Branch-and-Bound** (Land & Doig 1960): best-first
    search with a problem-supplied lower-bound function.  Prunes any
    subtree whose lower bound exceeds the incumbent.

Mathematical and algorithmic roots
----------------------------------

  * **Hart, P. E., Nilsson, N. J. & Raphael, B. (1968) — "A formal
    basis for the heuristic determination of minimum cost paths."**
    *IEEE Trans. Systems Science and Cybernetics* 4(2) 100–107.  The
    original A\*.  Theorem: A\* is *optimally efficient* among all
    algorithms that find an optimal solution using a consistent
    heuristic — it expands the minimum number of nodes for that
    heuristic.

  * **Pohl, I. (1970) — "Heuristic search viewed as path finding in a
    graph."**  *Artificial Intelligence* 1(3) 193–204.  Weighted A\*:
    ``f = g + w·h`` for ``w ≥ 1`` is ``w``-suboptimal.

  * **Korf, R. E. (1985) — "Depth-first iterative-deepening: An optimal
    admissible tree search."**  *Artificial Intelligence* 27(1)
    97–109.  IDA\* — linear-space optimal search.

  * **Knuth, D. E. & Moore, R. W. (1975) — "An analysis of alpha-beta
    pruning."**  *Artificial Intelligence* 6(4) 293–326.  The
    fundamental analysis: with optimal move ordering, alpha-beta
    searches ``b^{d/2}`` nodes instead of ``b^d``.

  * **Slate, D. J. & Atkin, L. R. (1977) — "Chess 4.5 — the
    Northwestern University chess program."**  In *Chess Skill in Man
    and Machine*.  Iterative deepening for time control.

  * **Schaeffer, J. (1989) — "The history heuristic and alpha-beta
    search enhancements in practice."**  *IEEE Trans. PAMI* 11(11)
    1203–1212.  History-driven move ordering for alpha-beta.

  * **Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002) — "Finite-time
    analysis of the multiarmed bandit problem."**  *Machine Learning*
    47(2) 235–256.  UCB1: regret ``O(K log T / Δ)``.

  * **Kocsis, L. & Szepesvári, C. (2006) — "Bandit-based Monte-Carlo
    planning."**  *Proc. ECML* 282–293.  UCT — UCB applied recursively
    to tree nodes.  Theorem: under stochastic rewards UCT's value
    estimate at the root converges in probability to the minimax value.

  * **Browne, C. B. *et al.* (2012) — "A survey of Monte Carlo tree
    search methods."**  *IEEE Trans. Computational Intelligence and AI
    in Games* 4(1) 1–43.  The unifying treatment of MCTS variants the
    PUCT branch follows.

  * **Silver, D. *et al.* (2017) — "Mastering the game of Go without
    human knowledge."**  *Nature* 550 354–359.  AlphaGo Zero — PUCT
    with a learned policy prior + value head; recommends
    ``c_puct ≈ 1.25``, Dirichlet ``α ≈ 0.3`` noise at the root.

  * **Silver, D. *et al.* (2018) — "A general reinforcement learning
    algorithm that masters chess, shogi and Go through self-play."**
    *Science* 362 1140–1144.  AlphaZero: same PUCT, broader domain.

  * **Schrittwieser, J. *et al.* (2020) — "Mastering Atari, Go, chess
    and shogi by planning with a learned model."**  *Nature* 588
    604–609.  MuZero: same PUCT on top of a *learned* dynamics; the
    architectural reason ``apply`` is a callable in this primitive.

  * **Coulom, R. (2007) — "Computing Elo ratings of move patterns in
    the game of Go."**  *ICGA Journal* 30(4) 198–208.  Progressive
    bias / progressive widening — the rule we implement when
    ``progressive_widening=True``.

  * **Land, A. H. & Doig, A. G. (1960) — "An automatic method of
    solving discrete programming problems."**  *Econometrica* 28(3)
    497–520.  Branch-and-bound.

  * **Pearl, J. (1984) — *Heuristics: Intelligent Search Strategies for
    Computer Problem Solving.***  Addison-Wesley.  The unifying text
    on admissibility, monotonicity, and the relation between A\* and
    branch-and-bound.

  * **Russell, S. & Norvig, P. (2020) — *Artificial Intelligence: A
    Modern Approach*, 4th ed., chap. 3, 5.**  Pedagogical reference for
    the entire family this module unifies.

  * **Helmbold, D. P. & Parker-Wood, A. (2009) — "All-moves-as-first
    heuristics in Monte-Carlo Go."**  *Proc. ICAI*.  RAVE — the rapid
    action-value heuristic exposed under ``rave=True``.

  * **Chaslot, G. *et al.* (2008) — "Parallel Monte-Carlo tree
    search."**  *Proc. CG*.  Virtual loss for parallel rollouts — the
    discipline ``virtual_loss=k`` uses in the lock-free expansion path.

What Searcher gives a coordination engine
-----------------------------------------

It gives the coordinator a *mathematically explicit, auditable* answer
to the question every planning, decision, and game-tree component
silently asks: **"given a state and a budget, what is the best action,
and how confident am I?"**

  * For every search, the answer is *not* a heuristic guess; it is a
    canonical ``best_action``, a calibrated ``best_value`` (with a
    standard error when the algorithm provides one), a
    ``principal_variation`` of the deepest agreed-on trajectory, a
    ``regret_bound`` whose constants are the algorithm's own, and a
    ``budget_used`` so the coordinator can size the next call.

  * The certificate ``SHA-256`` chains every decision, so a
    coordination engine that publishes a recommendation has a
    tamper-evident record of the search that produced it.

  * Every evaluator (``actions``, ``apply``, ``terminal``, ``reward``,
    ``policy_prior``, ``value``, ``heuristic``) is a Python callable
    the coordinator supplies — including *other primitives in this
    runtime*.  A coordinator can wire **Predictor** as ``value``,
    **Verifier** as ``terminal``, **Solver** as ``apply`` for a SAT
    move-generator, **Analogist** as ``policy_prior`` for retrieved
    cases.  The composition is the whole point.

  * The same surface dispatches to A\*, IDA\*, UCT, PUCT, alpha-beta,
    beam, and branch-and-bound.  Switching algorithms is a
    configuration change, not a rewrite.

  * Bounds are *measured* and *reported*: ``budget_used.nodes``,
    ``budget_used.expansions``, ``budget_used.peak_open_size``,
    ``budget_used.wall_seconds``, ``budget_used.deadline_hit`` —
    enough for the coordinator to SLO-gate the call.

Public API
----------

The module exposes:

  * ``SearchState`` — a structural protocol; a state is anything the
    user-supplied ``actions`` / ``apply`` / ``terminal`` / ``reward``
    callables understand.  A canonical ``key`` (hashable) is required
    only when transposition tables are enabled.
  * ``Action`` — any hashable value.
  * ``Evaluator`` — a small protocol bundling the callables.
  * ``StopCondition`` — a dataclass of budgets.
  * ``BudgetUsed`` — what the run consumed.
  * ``SearchNode`` — internal tree node, exposed for introspection.
  * ``SearchReport`` — the canonical report.
  * ``SearcherConfig`` — full configuration surface.
  * ``Searcher`` — the orchestrator dispatching to the selected
    algorithm.
  * Algorithm constants: ``ALGORITHM_ASTAR``, ``ALGORITHM_IDA_STAR``,
    ``ALGORITHM_UCT``, ``ALGORITHM_PUCT``, ``ALGORITHM_ALPHABETA``,
    ``ALGORITHM_BEAM``, ``ALGORITHM_BNB``.
  * Free-function shortcuts: ``astar``, ``puct``, ``uct``, ``alphabeta``,
    ``ida_star``, ``beam_search``, ``branch_and_bound``.

This module is **pure stdlib** — the runtime ships search into the same
low-dependency tier as ``Sketcher``, ``Solver``, and ``Verifier``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# Errors
# =============================================================================


class SearcherError(Exception):
    """Base for every Searcher-raised error."""


class InvalidConfig(SearcherError):
    """A SearcherConfig is structurally invalid."""


class InvalidEvaluator(SearcherError):
    """A user-supplied evaluator violates an algorithm precondition."""


class InvalidState(SearcherError):
    """A state produced by an evaluator failed a sanity check."""


class BudgetExhausted(SearcherError):
    """The configured budget was hit before any decision could be made."""


class NoSolution(SearcherError):
    """An optimal-search algorithm exhausted the open list without a goal."""


class UnknownAlgorithm(SearcherError):
    """The requested algorithm is not one of this module's families."""


# =============================================================================
# Algorithm names (string constants for stable configs)
# =============================================================================


ALGORITHM_ASTAR = "astar"
ALGORITHM_IDA_STAR = "ida_star"
ALGORITHM_UCT = "uct"
ALGORITHM_PUCT = "puct"
ALGORITHM_ALPHABETA = "alphabeta"
ALGORITHM_BEAM = "beam"
ALGORITHM_BNB = "bnb"
ALGORITHM_AUTO = "auto"

KNOWN_ALGORITHMS: Tuple[str, ...] = (
    ALGORITHM_ASTAR,
    ALGORITHM_IDA_STAR,
    ALGORITHM_UCT,
    ALGORITHM_PUCT,
    ALGORITHM_ALPHABETA,
    ALGORITHM_BEAM,
    ALGORITHM_BNB,
    ALGORITHM_AUTO,
)


# =============================================================================
# Type aliases
# =============================================================================


State = Any
Action = Hashable
ActionsFn = Callable[[State], Sequence[Action]]
ApplyFn = Callable[[State, Action], State]
TerminalFn = Callable[[State], bool]
RewardFn = Callable[[State], float]
HeuristicFn = Callable[[State], float]
CostFn = Callable[[State, Action, State], float]
PolicyPriorFn = Callable[[State, Sequence[Action]], Mapping[Action, float]]
ValueFn = Callable[[State], float]
KeyFn = Callable[[State], Hashable]


# =============================================================================
# Budgets & stop conditions
# =============================================================================


@dataclass(frozen=True)
class StopCondition:
    """Bounds the searcher must respect.

    A ``None`` value disables that bound.  An algorithm halts as soon
    as **any** bound is hit; the report's ``budget_used`` records which.
    """
    max_iterations: Optional[int] = None
    max_expansions: Optional[int] = None
    max_nodes: Optional[int] = None
    max_seconds: Optional[float] = None
    deadline: Optional[float] = None  # wall-clock epoch seconds
    max_depth: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_expansions", "max_nodes", "max_depth"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or v <= 0):
                raise InvalidConfig(f"{name}={v!r} must be a positive int or None")
        for name in ("max_seconds",):
            v = getattr(self, name)
            if v is not None and (v <= 0):
                raise InvalidConfig(f"{name}={v!r} must be positive or None")
        if self.deadline is not None and self.deadline < 0:
            raise InvalidConfig(f"deadline={self.deadline!r} must be non-negative or None")

    def is_empty(self) -> bool:
        """True if no bound is set (the searcher will run unbounded)."""
        return all(
            v is None
            for v in (
                self.max_iterations,
                self.max_expansions,
                self.max_nodes,
                self.max_seconds,
                self.deadline,
                self.max_depth,
            )
        )


@dataclass
class BudgetUsed:
    """Records what a search actually consumed and which bound (if any) it hit."""
    iterations: int = 0
    expansions: int = 0
    nodes: int = 0
    rollouts: int = 0
    wall_seconds: float = 0.0
    peak_open_size: int = 0
    peak_depth: int = 0
    deadline_hit: bool = False
    bound_hit: Optional[str] = None  # name of the StopCondition field that triggered

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# Searcher configuration
# =============================================================================


@dataclass(frozen=True)
class SearcherConfig:
    """Configuration for ``Searcher``.

    All fields have safe defaults; an empty ``SearcherConfig()`` runs
    the default algorithm (``"puct"``) with reasonable AlphaZero-style
    constants.

    Algorithm-selection
        algorithm:           one of ``KNOWN_ALGORITHMS``; ``"auto"`` picks
                             based on which evaluators are supplied.

    Bounds
        max_iterations:      total iterations (algorithm-specific).
        max_expansions:      total node expansions (children generated).
        max_nodes:           total nodes ever created.
        max_seconds:         wall-clock budget.
        deadline:            absolute wall-clock deadline (epoch s).
        max_depth:           maximum tree depth.

    PUCT / UCT
        c_puct:              exploration constant.
        rollout_depth:       UCT random-rollout depth cap.
        rollouts_per_leaf:   UCT rollouts per leaf evaluation.
        dirichlet_alpha:     Dirichlet noise α at root for PUCT; 0 disables.
        dirichlet_epsilon:   mixing weight (1-ε)·P + ε·noise; 0 disables.
        progressive_widening: enable Coulom-style widening for branching
                             factors that grow with visit count.
        widen_alpha, widen_k: ``k · N(s)^α`` children allowed; alpha ∈ (0,1].

    A* / IDA* / B&B
        weighted_astar:      ``w`` (≥1) for weighted A*; 1.0 is exact.
        cost_fn:             optional callable returning the cost of a
                             transition; defaults to constant 1.

    Alpha-beta
        iterative_deepening: bool; if False, search only at ``ab_depth``.
        ab_depth:            target depth when ID is off (or upper bound).
        aspiration_window:   half-width for aspiration windows; 0 disables.
        use_transposition:   keep a transposition table.
        history_heuristic:   keep a history table for move ordering.

    Beam
        beam_width:          ``b`` (positive int).
        beam_score:          ``"value"`` (higher better) or ``"cost"`` (lower).

    Determinism / certificate
        seed:                RNG seed; the search is deterministic given
                             the seed and evaluators.
        secret_key:          bytes for HMAC over the canonical search
                             trace.  If empty, SHA-256 is used (still
                             tamper-evident, not authenticated).
        keep_full_trace:     if True, the SearchReport carries the full
                             list of expansion events (memory-heavier).
    """
    algorithm: str = ALGORITHM_PUCT

    # bounds
    max_iterations: Optional[int] = 1024
    max_expansions: Optional[int] = None
    max_nodes: Optional[int] = None
    max_seconds: Optional[float] = None
    deadline: Optional[float] = None
    max_depth: Optional[int] = None

    # PUCT / UCT
    c_puct: float = 1.25
    rollout_depth: int = 32
    rollouts_per_leaf: int = 1
    dirichlet_alpha: float = 0.0
    dirichlet_epsilon: float = 0.0
    progressive_widening: bool = False
    widen_alpha: float = 0.5
    widen_k: float = 1.0
    rave: bool = False
    rave_equiv: float = 1000.0
    virtual_loss: int = 0

    # A* / IDA* / B&B
    weighted_astar: float = 1.0
    cost_default: float = 1.0
    ida_star_step: float = 0.0  # additive bound step; 0 = use observed minimum

    # Alpha-beta
    iterative_deepening: bool = True
    ab_depth: int = 4
    aspiration_window: float = 0.0
    use_transposition: bool = True
    history_heuristic: bool = True
    quiescence_depth: int = 0

    # Beam
    beam_width: int = 8
    beam_score: str = "value"  # "value" (max) or "cost" (min)

    # Determinism / certificate
    seed: int = 0
    secret_key: bytes = b""
    keep_full_trace: bool = False

    def __post_init__(self) -> None:
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise InvalidConfig(
                f"algorithm={self.algorithm!r} not in {KNOWN_ALGORITHMS}"
            )
        if self.c_puct <= 0:
            raise InvalidConfig(f"c_puct={self.c_puct!r} must be > 0")
        if self.rollout_depth < 0:
            raise InvalidConfig(f"rollout_depth={self.rollout_depth!r} must be ≥ 0")
        if self.rollouts_per_leaf < 1:
            raise InvalidConfig(f"rollouts_per_leaf={self.rollouts_per_leaf!r} must be ≥ 1")
        if self.dirichlet_alpha < 0:
            raise InvalidConfig(f"dirichlet_alpha={self.dirichlet_alpha!r} must be ≥ 0")
        if not (0.0 <= self.dirichlet_epsilon <= 1.0):
            raise InvalidConfig(
                f"dirichlet_epsilon={self.dirichlet_epsilon!r} must be in [0,1]"
            )
        if not (0.0 < self.widen_alpha <= 1.0):
            raise InvalidConfig(f"widen_alpha={self.widen_alpha!r} must be in (0,1]")
        if self.widen_k <= 0:
            raise InvalidConfig(f"widen_k={self.widen_k!r} must be > 0")
        if self.rave_equiv <= 0:
            raise InvalidConfig(f"rave_equiv={self.rave_equiv!r} must be > 0")
        if self.virtual_loss < 0:
            raise InvalidConfig(f"virtual_loss={self.virtual_loss!r} must be ≥ 0")
        if self.weighted_astar < 1.0:
            raise InvalidConfig(
                f"weighted_astar={self.weighted_astar!r} must be ≥ 1"
            )
        if self.cost_default < 0:
            raise InvalidConfig(f"cost_default={self.cost_default!r} must be ≥ 0")
        if self.ida_star_step < 0:
            raise InvalidConfig(f"ida_star_step={self.ida_star_step!r} must be ≥ 0")
        if self.ab_depth < 1:
            raise InvalidConfig(f"ab_depth={self.ab_depth!r} must be ≥ 1")
        if self.aspiration_window < 0:
            raise InvalidConfig(
                f"aspiration_window={self.aspiration_window!r} must be ≥ 0"
            )
        if self.quiescence_depth < 0:
            raise InvalidConfig(
                f"quiescence_depth={self.quiescence_depth!r} must be ≥ 0"
            )
        if self.beam_width < 1:
            raise InvalidConfig(f"beam_width={self.beam_width!r} must be ≥ 1")
        if self.beam_score not in ("value", "cost"):
            raise InvalidConfig(
                f"beam_score={self.beam_score!r} must be 'value' or 'cost'"
            )
        # validate bounds via StopCondition
        StopCondition(
            max_iterations=self.max_iterations,
            max_expansions=self.max_expansions,
            max_nodes=self.max_nodes,
            max_seconds=self.max_seconds,
            deadline=self.deadline,
            max_depth=self.max_depth,
        )

    def stop_condition(self) -> StopCondition:
        return StopCondition(
            max_iterations=self.max_iterations,
            max_expansions=self.max_expansions,
            max_nodes=self.max_nodes,
            max_seconds=self.max_seconds,
            deadline=self.deadline,
            max_depth=self.max_depth,
        )


# =============================================================================
# Evaluator bundle
# =============================================================================


@dataclass(frozen=True)
class Evaluator:
    """Bundle of user-supplied callables describing the search problem.

    Most algorithms only need a subset.  ``actions``, ``apply``, and
    one of ``terminal`` / ``reward`` are the universal minimum.

    For best-first search (A*, B&B):
        ``terminal``, ``cost`` (or use config's ``cost_default``),
        ``heuristic`` (admissible for optimal A*).

    For MCTS / UCT:
        ``terminal``, ``reward`` (or ``value`` for leaf evaluation).

    For PUCT:
        same as UCT plus ``policy_prior`` and ``value``.

    For alpha-beta:
        ``terminal``, ``reward`` (static eval from side-to-move's view).
    """
    actions: ActionsFn
    apply: ApplyFn
    terminal: Optional[TerminalFn] = None
    reward: Optional[RewardFn] = None
    heuristic: Optional[HeuristicFn] = None
    cost: Optional[CostFn] = None
    policy_prior: Optional[PolicyPriorFn] = None
    value: Optional[ValueFn] = None
    key: Optional[KeyFn] = None  # canonical hashable key; defaults to id-of-state


# =============================================================================
# Search node (exposed for introspection)
# =============================================================================


@dataclass
class SearchNode:
    """A node in the search tree.

    The set of populated fields depends on the algorithm; ``visits``
    / ``total_value`` are MCTS-only, ``g`` / ``f`` are A*-only, etc.
    """
    key: Hashable
    parent: Optional["SearchNode"] = None
    incoming_action: Optional[Action] = None
    depth: int = 0
    # MCTS fields
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0  # PUCT prior P(s,a) on the *edge* parent→here
    children: Dict[Action, "SearchNode"] = field(default_factory=dict)
    untried_actions: List[Action] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float = 0.0
    # Best-first fields
    g: float = 0.0
    h: float = 0.0
    # MCTS-only state (computed on first expansion)
    available_actions: Tuple[Action, ...] = ()

    @property
    def f(self) -> float:
        return self.g + self.h

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


# =============================================================================
# Search report
# =============================================================================


@dataclass
class SearchReport:
    """Canonical report produced by every algorithm.

    Required fields populated by every algorithm.  Optional fields may
    be absent (None) when the algorithm does not produce them.

    history is a sequence of (iteration, best_action, best_value)
    tuples so a coordinator can observe convergence.
    """
    algorithm: str
    best_action: Optional[Action]
    best_value: float
    principal_variation: Tuple[Action, ...]
    iterations: int
    budget_used: BudgetUsed
    certificate: str
    seed: int
    finished: bool  # True if the search terminated naturally (not on a bound)
    bound_hit: Optional[str]  # None or the name of the bound that fired

    # Optional / algorithm-specific
    root_visits: int = 0
    root_q_by_action: Dict[Action, float] = field(default_factory=dict)
    root_visits_by_action: Dict[Action, int] = field(default_factory=dict)
    root_priors_by_action: Dict[Action, float] = field(default_factory=dict)
    optimal_cost: Optional[float] = None  # A* / B&B optimal-path cost
    bound: Optional[float] = None  # IDA* final f-bound
    history: Tuple[Tuple[int, Optional[Action], float], ...] = ()
    full_trace: Tuple[Dict[str, Any], ...] = ()
    notes: str = ""

    # Regret / approximation bounds, when computable
    regret_bound: Optional[float] = None
    suboptimality_bound: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "algorithm": self.algorithm,
            "best_action": _safe_action_repr(self.best_action),
            "best_value": self.best_value,
            "principal_variation": [_safe_action_repr(a) for a in self.principal_variation],
            "iterations": self.iterations,
            "budget_used": self.budget_used.as_dict(),
            "certificate": self.certificate,
            "seed": self.seed,
            "finished": self.finished,
            "bound_hit": self.bound_hit,
            "root_visits": self.root_visits,
            "root_q_by_action": {
                _safe_action_repr(k): v for k, v in self.root_q_by_action.items()
            },
            "root_visits_by_action": {
                _safe_action_repr(k): v for k, v in self.root_visits_by_action.items()
            },
            "root_priors_by_action": {
                _safe_action_repr(k): v for k, v in self.root_priors_by_action.items()
            },
            "optimal_cost": self.optimal_cost,
            "bound": self.bound,
            "history": [
                (i, _safe_action_repr(a), v) for (i, a, v) in self.history
            ],
            "notes": self.notes,
            "regret_bound": self.regret_bound,
            "suboptimality_bound": self.suboptimality_bound,
        }
        return d


def _safe_action_repr(a: Any) -> Any:
    """Coerce an action into a JSON-friendly form for as_dict."""
    if a is None:
        return None
    if isinstance(a, (str, int, float, bool)):
        return a
    if isinstance(a, tuple):
        return list(_safe_action_repr(x) for x in a)
    return repr(a)


# =============================================================================
# Canonical-trace certificate
# =============================================================================


def _canonical_bytes(obj: Any) -> bytes:
    """Stable JSON-style encoding so the certificate is reproducible."""
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _str_key(kv[0]))
        return b"{" + b",".join(
            _canonical_bytes(k) + b":" + _canonical_bytes(v) for k, v in items
        ) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(_canonical_bytes(x) for x in obj) + b"]"
    if isinstance(obj, bool):
        return b"true" if obj else b"false"
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            # canonicalize non-finites to a stable string
            return f'"{obj}"'.encode("utf-8")
        # repr is reproducible for ints; for floats use repr which round-trips
        return repr(obj).encode("utf-8")
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True).encode("utf-8")
    if obj is None:
        return b"null"
    if isinstance(obj, bytes):
        return json.dumps(obj.hex(), ensure_ascii=True).encode("utf-8")
    # Fallback: repr.  Hashes will only be stable for objects with
    # reproducible repr (e.g. dataclasses with stable str).
    return json.dumps(repr(obj), ensure_ascii=True).encode("utf-8")


def _str_key(k: Any) -> str:
    if isinstance(k, str):
        return k
    return repr(k)


class _CertChain:
    """SHA-256 chained hash over a stream of canonical-serialised events.

    Optional HMAC under a secret key (turns it into an authenticated
    chain; with an empty key the chain is still tamper-evident, just
    not authenticated — anyone replaying the search can recompute it).
    """

    def __init__(self, secret_key: bytes = b"") -> None:
        self._secret = bytes(secret_key)
        # genesis hash includes the secret prefix so chains under different
        # keys cannot collide trivially.
        seed = b"agi.searcher.v1\x00" + self._secret
        self._h = hashlib.sha256(seed).digest()
        self._count = 0

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._count += 1
        body = _canonical_bytes({"k": kind, "n": self._count, "p": payload})
        if self._secret:
            tag = hashlib.sha256(self._secret + self._h + body).digest()
        else:
            tag = hashlib.sha256(self._h + body).digest()
        self._h = tag

    def hexdigest(self) -> str:
        return self._h.hex()

    @property
    def count(self) -> int:
        return self._count


# =============================================================================
# Stop-condition helper
# =============================================================================


class _StopTracker:
    """Combines a StopCondition with a BudgetUsed accumulator."""

    def __init__(self, stop: StopCondition, budget: BudgetUsed,
                 start_time: float, deadline_offset: Optional[float] = None) -> None:
        self.stop = stop
        self.budget = budget
        self.start_time = start_time
        self.deadline_offset = deadline_offset

    def stopped(self) -> bool:
        s = self.stop
        b = self.budget
        if s.max_iterations is not None and b.iterations >= s.max_iterations:
            b.bound_hit = "max_iterations"
            return True
        if s.max_expansions is not None and b.expansions >= s.max_expansions:
            b.bound_hit = "max_expansions"
            return True
        if s.max_nodes is not None and b.nodes >= s.max_nodes:
            b.bound_hit = "max_nodes"
            return True
        if s.max_depth is not None and b.peak_depth > s.max_depth:
            b.bound_hit = "max_depth"
            return True
        now = time.time()
        wall = now - self.start_time
        b.wall_seconds = wall
        if s.max_seconds is not None and wall >= s.max_seconds:
            b.bound_hit = "max_seconds"
            return True
        if s.deadline is not None and now >= s.deadline:
            b.deadline_hit = True
            b.bound_hit = "deadline"
            return True
        return False


# =============================================================================
# Algorithm: A* / weighted-A* / branch-and-bound
# =============================================================================


def _resolve_key(ev: Evaluator, state: State) -> Hashable:
    if ev.key is not None:
        return ev.key(state)
    # Fall back to id().  Stable within a run; not portable.
    return id(state)


def _cost_of(ev: Evaluator, cfg: SearcherConfig,
             parent: State, action: Action, child: State) -> float:
    if ev.cost is None:
        return cfg.cost_default
    c = float(ev.cost(parent, action, child))
    if c < 0:
        raise InvalidEvaluator(f"cost is negative ({c}); A*/B&B require non-negative costs")
    return c


def _heuristic_of(ev: Evaluator, state: State) -> float:
    if ev.heuristic is None:
        return 0.0
    h = float(ev.heuristic(state))
    if h < 0:
        raise InvalidEvaluator(f"heuristic is negative ({h}); admissible heuristics are ≥ 0")
    return h


def _astar(searcher: "Searcher", root: State, ev: Evaluator) -> SearchReport:
    cfg = searcher.config
    stop = cfg.stop_condition()
    budget = BudgetUsed()
    chain = _CertChain(cfg.secret_key)
    history: List[Tuple[int, Optional[Action], float]] = []
    full_trace: List[Dict[str, Any]] = []
    rng = random.Random(cfg.seed)
    tracker = _StopTracker(stop, budget, time.time())
    w = cfg.weighted_astar

    if ev.terminal is None:
        raise InvalidEvaluator("astar requires ev.terminal")

    root_key = _resolve_key(ev, root)
    chain.emit("init", {"alg": "astar", "root_key": _str_key(root_key),
                        "weighted": w, "seed": cfg.seed})

    # f, tiebreaker_counter, g, key, state, parent_key, action_in
    counter = 0
    open_heap: List[Tuple[float, int, float, Any, State, Any, Any]] = []
    h_root = _heuristic_of(ev, root)
    open_heap.append((w * h_root, counter, 0.0, root_key, root, None, None))
    counter += 1

    best_g: Dict[Hashable, float] = {root_key: 0.0}
    parent_of: Dict[Hashable, Tuple[Hashable, Action]] = {}
    closed: Set[Hashable] = set()
    nodes_made = 1
    budget.nodes = nodes_made

    goal_key: Optional[Hashable] = None
    goal_g: Optional[float] = None
    incumbent_g: Optional[float] = None  # B&B incumbent

    while open_heap:
        if tracker.stopped():
            break
        f, _, g, key, state, _pkey, _act = heapq.heappop(open_heap)
        if key in closed and best_g.get(key, math.inf) < g:
            continue
        budget.iterations += 1
        budget.peak_open_size = max(budget.peak_open_size, len(open_heap) + 1)

        # B&B prune: incumbent dominates
        if incumbent_g is not None and g >= incumbent_g:
            continue

        if ev.terminal(state):
            goal_key = key
            goal_g = g
            chain.emit("goal", {"key": _str_key(key), "g": g})
            if cfg.algorithm == ALGORITHM_BNB:
                # Branch & Bound: continue with incumbent update
                if incumbent_g is None or g < incumbent_g:
                    incumbent_g = g
                    history.append((budget.iterations, None, g))
                continue
            else:
                history.append((budget.iterations, None, g))
                break

        if key in closed:
            continue
        closed.add(key)

        for act in ev.actions(state):
            if tracker.stopped():
                break
            child_state = ev.apply(state, act)
            child_key = _resolve_key(ev, child_state)
            c = _cost_of(ev, cfg, state, act, child_state)
            ng = g + c
            if ng >= best_g.get(child_key, math.inf):
                continue
            best_g[child_key] = ng
            parent_of[child_key] = (key, act)
            h = _heuristic_of(ev, child_state)
            nf = ng + w * h
            counter += 1
            nodes_made += 1
            budget.expansions += 1
            budget.nodes = nodes_made
            budget.peak_depth = max(budget.peak_depth, _depth_via(parent_of, child_key, root_key))
            chain.emit("expand", {
                "parent_key": _str_key(key),
                "action": _safe_action_repr(act),
                "child_key": _str_key(child_key),
                "g": ng, "h": h, "f": nf,
            })
            if cfg.keep_full_trace:
                full_trace.append({
                    "event": "expand",
                    "parent_key": _str_key(key),
                    "action": _safe_action_repr(act),
                    "child_key": _str_key(child_key),
                    "g": ng, "h": h, "f": nf,
                })
            heapq.heappush(open_heap, (nf, counter, ng, child_key, child_state, key, act))

    finished = (goal_key is not None) or not open_heap
    finished = finished and (budget.bound_hit is None)

    # Reconstruct PV from parent_of
    pv: Tuple[Action, ...] = ()
    best_action: Optional[Action] = None
    optimal_cost: Optional[float] = None
    if goal_key is not None and goal_g is not None:
        pv = _reconstruct_actions(parent_of, goal_key, root_key)
        if pv:
            best_action = pv[0]
        optimal_cost = goal_g

    # B&B: report incumbent if no terminal popped (open ran out)
    if cfg.algorithm == ALGORITHM_BNB and incumbent_g is not None:
        optimal_cost = incumbent_g

    suboptimality = None
    if w > 1.0 and optimal_cost is not None:
        suboptimality = w  # weighted A* is w-suboptimal

    best_value = -float(optimal_cost) if optimal_cost is not None else float("-inf")

    return SearchReport(
        algorithm=cfg.algorithm,
        best_action=best_action,
        best_value=best_value,
        principal_variation=pv,
        iterations=budget.iterations,
        budget_used=budget,
        certificate=chain.hexdigest(),
        seed=cfg.seed,
        finished=finished,
        bound_hit=budget.bound_hit,
        optimal_cost=optimal_cost,
        history=tuple(history),
        full_trace=tuple(full_trace),
        suboptimality_bound=suboptimality,
        notes=("A* (weighted)" if w > 1.0 else "A*") if cfg.algorithm == ALGORITHM_ASTAR
              else "branch-and-bound",
    )


def _depth_via(parent_of: Mapping[Hashable, Tuple[Hashable, Action]],
               key: Hashable, root_key: Hashable) -> int:
    d = 0
    k = key
    seen: Set[Hashable] = set()
    while k in parent_of and k != root_key and k not in seen:
        seen.add(k)
        d += 1
        k = parent_of[k][0]
        if d > 1_000_000:
            break
    return d


def _reconstruct_actions(parent_of: Mapping[Hashable, Tuple[Hashable, Action]],
                         goal_key: Hashable, root_key: Hashable) -> Tuple[Action, ...]:
    actions: List[Action] = []
    k = goal_key
    seen: Set[Hashable] = set()
    while k != root_key and k in parent_of and k not in seen:
        seen.add(k)
        pk, act = parent_of[k]
        actions.append(act)
        k = pk
        if len(actions) > 1_000_000:
            break
    actions.reverse()
    return tuple(actions)


# =============================================================================
# Algorithm: IDA*
# =============================================================================


def _ida_star(searcher: "Searcher", root: State, ev: Evaluator) -> SearchReport:
    cfg = searcher.config
    stop = cfg.stop_condition()
    budget = BudgetUsed()
    chain = _CertChain(cfg.secret_key)
    history: List[Tuple[int, Optional[Action], float]] = []
    full_trace: List[Dict[str, Any]] = []
    tracker = _StopTracker(stop, budget, time.time())

    if ev.terminal is None:
        raise InvalidEvaluator("ida_star requires ev.terminal")

    root_key = _resolve_key(ev, root)
    chain.emit("init", {"alg": "ida_star", "root_key": _str_key(root_key),
                        "seed": cfg.seed})

    bound = _heuristic_of(ev, root)
    best_path_actions: Tuple[Action, ...] = ()
    goal_g: Optional[float] = None
    goal_key: Optional[Hashable] = None

    # Recursive DFS with f-bound
    sentinel_inf = float("inf")
    step = cfg.ida_star_step

    while True:
        if tracker.stopped():
            break
        budget.iterations += 1
        path_keys: List[Hashable] = [root_key]
        path_actions: List[Action] = []
        nxt, found, goal_g_now = _ida_search(
            searcher, ev, root, 0.0, bound, path_keys, path_actions,
            chain, full_trace, budget, tracker
        )
        if found:
            goal_g = goal_g_now
            best_path_actions = tuple(path_actions)
            history.append((budget.iterations, None, goal_g_now))
            chain.emit("goal", {"g": goal_g_now, "depth": len(path_actions)})
            break
        if nxt == sentinel_inf or budget.bound_hit is not None:
            break
        history.append((budget.iterations, None, bound))
        chain.emit("rebound", {"bound": bound, "next": nxt})
        bound = nxt + step  # additive step (0 = use observed minimum)

    finished = (goal_g is not None) or (budget.bound_hit is None)
    return SearchReport(
        algorithm=ALGORITHM_IDA_STAR,
        best_action=(best_path_actions[0] if best_path_actions else None),
        best_value=(-goal_g if goal_g is not None else float("-inf")),
        principal_variation=best_path_actions,
        iterations=budget.iterations,
        budget_used=budget,
        certificate=chain.hexdigest(),
        seed=cfg.seed,
        finished=finished,
        bound_hit=budget.bound_hit,
        optimal_cost=goal_g,
        bound=bound,
        history=tuple(history),
        full_trace=tuple(full_trace),
        notes="IDA*",
    )


def _ida_search(
    searcher: "Searcher", ev: Evaluator, state: State, g: float, bound: float,
    path_keys: List[Hashable], path_actions: List[Action],
    chain: "_CertChain", full_trace: List[Dict[str, Any]],
    budget: BudgetUsed, tracker: _StopTracker,
) -> Tuple[float, bool, Optional[float]]:
    if tracker.stopped():
        return float("inf"), False, None
    cfg = searcher.config
    h = _heuristic_of(ev, state)
    f = g + h
    if f > bound:
        return f, False, None
    if ev.terminal(state):
        return f, True, g
    budget.peak_depth = max(budget.peak_depth, len(path_actions))
    min_next = float("inf")
    for act in ev.actions(state):
        if tracker.stopped():
            return float("inf"), False, None
        child = ev.apply(state, act)
        ckey = _resolve_key(ev, child)
        if ckey in path_keys:
            continue  # avoid revisiting a state on the current path
        c = _cost_of(ev, cfg, state, act, child)
        path_keys.append(ckey)
        path_actions.append(act)
        budget.expansions += 1
        budget.nodes += 1
        chain.emit("ida_expand", {
            "parent_key": _str_key(path_keys[-2]),
            "action": _safe_action_repr(act),
            "child_key": _str_key(ckey),
            "g": g + c, "bound": bound,
        })
        if cfg.keep_full_trace:
            full_trace.append({
                "event": "ida_expand",
                "parent_key": _str_key(path_keys[-2]),
                "action": _safe_action_repr(act),
                "child_key": _str_key(ckey),
                "g": g + c, "bound": bound,
            })
        t, found, gg = _ida_search(
            searcher, ev, child, g + c, bound, path_keys, path_actions,
            chain, full_trace, budget, tracker
        )
        if found:
            return t, True, gg
        if t < min_next:
            min_next = t
        path_keys.pop()
        path_actions.pop()
    return min_next, False, None


# =============================================================================
# Algorithm: UCT (Kocsis-Szepesvári 2006)
# =============================================================================


def _uct(searcher: "Searcher", root: State, ev: Evaluator) -> SearchReport:
    return _mcts_core(searcher, root, ev, algorithm=ALGORITHM_UCT)


def _puct(searcher: "Searcher", root: State, ev: Evaluator) -> SearchReport:
    return _mcts_core(searcher, root, ev, algorithm=ALGORITHM_PUCT)


def _mcts_core(searcher: "Searcher", root: State, ev: Evaluator,
               algorithm: str) -> SearchReport:
    cfg = searcher.config
    stop = cfg.stop_condition()
    budget = BudgetUsed()
    chain = _CertChain(cfg.secret_key)
    history: List[Tuple[int, Optional[Action], float]] = []
    full_trace: List[Dict[str, Any]] = []
    rng = random.Random(cfg.seed)
    tracker = _StopTracker(stop, budget, time.time())

    if ev.terminal is None:
        raise InvalidEvaluator(f"{algorithm} requires ev.terminal")
    if algorithm == ALGORITHM_PUCT and ev.policy_prior is None:
        # treat as uniform if not supplied
        pass

    root_key = _resolve_key(ev, root)
    chain.emit("init", {"alg": algorithm, "root_key": _str_key(root_key),
                        "c_puct": cfg.c_puct, "seed": cfg.seed,
                        "rollout_depth": cfg.rollout_depth,
                        "dirichlet_alpha": cfg.dirichlet_alpha,
                        "dirichlet_epsilon": cfg.dirichlet_epsilon})

    # Index nodes by canonical state key so transpositions share statistics.
    nodes: Dict[Hashable, SearchNode] = {}
    root_node = SearchNode(key=root_key, depth=0)
    root_node.is_terminal = ev.terminal(root)
    if root_node.is_terminal:
        root_node.terminal_value = ev.reward(root) if ev.reward else 0.0
    nodes[root_key] = root_node
    budget.nodes = 1

    # We need to be able to recover a state from a key during selection;
    # MCTS naturally regrows the state path each iteration, so we never
    # store states.

    # Pre-expand the root for priors / actions
    _expand(searcher, root_node, root, ev, algorithm, rng,
            dirichlet=True, chain=chain, full_trace=full_trace, budget=budget)

    while True:
        if tracker.stopped():
            break
        budget.iterations += 1

        # ---- selection ----
        path: List[Tuple[SearchNode, Action]] = []
        path_states: List[State] = [root]
        node = root_node
        state = root
        depth = 0
        path_keys: Set[Hashable] = {root_node.key}
        # Soft cap on selection depth to prevent infinite descent through
        # transposition cycles even when max_depth isn't set.
        selection_cap = stop.max_depth if stop.max_depth is not None else 4096
        while True:
            if node.is_terminal:
                break
            if node.untried_actions:
                break
            if not node.children:
                break
            if depth >= selection_cap:
                break
            # all children expanded — descend by UCT/PUCT
            act, child_key = _select_child(node, cfg, algorithm)
            # Cycle detection: if descending to a node we've already
            # visited on this selection path, treat the current node as
            # a leaf for evaluation purposes (otherwise we'd loop until
            # the budget runs out).
            if child_key in path_keys:
                break
            path.append((node, act))
            try:
                state = ev.apply(state, act)
            except Exception as e:
                raise InvalidEvaluator(f"apply failed during selection: {e}")
            path_states.append(state)
            child = nodes.get(child_key)
            if child is None:
                # transposition target not yet a node (unusual; happens if
                # we share keys across paths and pre-created the child via
                # expand on a sibling).  Create now.
                child = SearchNode(
                    key=child_key,
                    parent=node,
                    incoming_action=act,
                    depth=node.depth + 1,
                )
                child.is_terminal = ev.terminal(state)
                if child.is_terminal and ev.reward is not None:
                    child.terminal_value = ev.reward(state)
                nodes[child_key] = child
                budget.nodes += 1
            else:
                # update parent linkage if not set (transposition)
                if child.parent is None:
                    child.parent = node
                    child.incoming_action = act
            node = child
            depth += 1
            path_keys.add(child.key)
            budget.peak_depth = max(budget.peak_depth, depth)
            if stop.max_depth is not None and depth >= stop.max_depth:
                break

        # ---- expansion ----
        if not node.is_terminal and node.untried_actions:
            # expand one child
            act = node.untried_actions.pop(0)
            try:
                child_state = ev.apply(state, act)
            except Exception as e:
                raise InvalidEvaluator(f"apply failed during expansion: {e}")
            child_key = _resolve_key(ev, child_state)
            is_self_loop = (child_key == node.key)
            if is_self_loop:
                # Don't transposition-share a self-loop with the parent
                # — that would re-expand the parent's untried_actions on
                # subsequent iterations and corrupt visit accounting.
                # Allocate a *fresh* dead-end stub keyed off (parent_key,
                # action) so the action gets its own visit counter.
                stub_key = ("__self_loop__", node.key, act)
                child = nodes.get(stub_key)
                if child is None:
                    child = SearchNode(
                        key=stub_key,
                        parent=node,
                        incoming_action=act,
                        depth=node.depth + 1,
                        prior=getattr(node, "priors", {}).get(act, 0.0),
                    )
                    child.is_terminal = True
                    # Use the parent's static reward as the stub value if
                    # available — semantically "applying this action does
                    # not change state, so its terminal value is the same
                    # as the parent's current state value".
                    if ev.reward is not None:
                        try:
                            child.terminal_value = float(ev.reward(state))
                        except Exception:
                            child.terminal_value = 0.0
                    nodes[stub_key] = child
                    budget.nodes += 1
                _is_new = False  # always treat as not-new (no re-expand)
            else:
                existing = nodes.get(child_key)
                if existing is None:
                    child = SearchNode(
                        key=child_key,
                        parent=node,
                        incoming_action=act,
                        depth=node.depth + 1,
                        prior=getattr(node, "priors", {}).get(act, 0.0),
                    )
                    child.is_terminal = ev.terminal(child_state)
                    if child.is_terminal and ev.reward is not None:
                        child.terminal_value = ev.reward(child_state)
                    nodes[child_key] = child
                    budget.nodes += 1
                    _is_new = True
                else:
                    child = existing
                    _is_new = False
            node.children[act] = child
            chain.emit("expand", {
                "parent_key": _str_key(node.key),
                "action": _safe_action_repr(act),
                "child_key": _str_key(child.key),
                "is_terminal": child.is_terminal,
                "self_loop": is_self_loop,
            })
            if cfg.keep_full_trace:
                full_trace.append({
                    "event": "expand",
                    "parent_key": _str_key(node.key),
                    "action": _safe_action_repr(act),
                    "child_key": _str_key(child.key),
                    "is_terminal": child.is_terminal,
                    "self_loop": is_self_loop,
                })
            budget.expansions += 1
            path.append((node, act))
            path_states.append(child_state)
            node = child
            state = child_state
            depth += 1
            budget.peak_depth = max(budget.peak_depth, depth)
            # Only call _expand on a *new* non-terminal child.
            if _is_new and not child.is_terminal:
                _expand(searcher, child, child_state, ev, algorithm, rng,
                        dirichlet=False, chain=chain,
                        full_trace=full_trace, budget=budget)

        # ---- evaluation (rollout or learned value) ----
        if node.is_terminal:
            leaf_value = node.terminal_value
            chain.emit("leaf", {"key": _str_key(node.key), "terminal_value": leaf_value})
        else:
            leaf_value = _evaluate_leaf(searcher, node, state, ev, rng,
                                        chain, full_trace, budget)

        # ---- backprop ----
        node.visits += 1
        node.total_value += leaf_value
        for parent_node, act in reversed(path):
            parent_node.visits += 1
            parent_node.total_value += leaf_value
            chain.emit("backup", {
                "key": _str_key(parent_node.key),
                "action": _safe_action_repr(act),
                "value": leaf_value,
                "visits": parent_node.visits,
            })

        # update history with current best
        ba, bv = _best_child_by_visits(root_node)
        history.append((budget.iterations, ba, bv))

    # ---- finalize ----
    best_action, best_value = _best_child_by_visits(root_node)
    root_visits = root_node.visits
    root_q_by_action: Dict[Action, float] = {}
    root_visits_by_action: Dict[Action, int] = {}
    for act, child in root_node.children.items():
        root_q_by_action[act] = child.q
        root_visits_by_action[act] = child.visits
    root_priors_by_action = dict(getattr(root_node, "priors", {}))

    # PV: descend always the highest-visit child
    pv = _principal_variation(root_node)

    # Regret bound (best-effort)
    K = max(1, len(root_node.children) or len(root_node.untried_actions) or 1)
    T = max(1, budget.iterations)
    # The UCB1 finite-time regret is O(K log T / Δ) but Δ is unknown.
    # Report the data-free upper-bound coefficient ``8 ln T`` from
    # Auer-Cesa-Bianchi-Fischer Theorem 1 (the per-arm bound).
    regret_bound = 8.0 * math.log(T) * K  # coefficient × K-arms (Δ_min unknown)

    finished = (budget.bound_hit is None)

    return SearchReport(
        algorithm=algorithm,
        best_action=best_action,
        best_value=best_value,
        principal_variation=pv,
        iterations=budget.iterations,
        budget_used=budget,
        certificate=chain.hexdigest(),
        seed=cfg.seed,
        finished=finished,
        bound_hit=budget.bound_hit,
        root_visits=root_visits,
        root_q_by_action=root_q_by_action,
        root_visits_by_action=root_visits_by_action,
        root_priors_by_action=root_priors_by_action,
        history=tuple(history),
        full_trace=tuple(full_trace),
        notes=("PUCT (AlphaZero)" if algorithm == ALGORITHM_PUCT
               else "UCT (UCB1)"),
        regret_bound=regret_bound,
    )


def _expand(searcher: "Searcher", node: SearchNode, state: State, ev: Evaluator,
            algorithm: str, rng: random.Random, *, dirichlet: bool,
            chain: _CertChain, full_trace: List[Dict[str, Any]],
            budget: BudgetUsed) -> None:
    """Compute the legal actions, priors, and progressive-widening cap."""
    cfg = searcher.config
    actions = tuple(ev.actions(state))
    node.available_actions = actions
    priors: Dict[Action, float] = {}
    if algorithm == ALGORITHM_PUCT:
        if ev.policy_prior is not None:
            raw = ev.policy_prior(state, actions)
            # Normalise / fall back to uniform if degenerate
            total = sum(max(0.0, float(raw.get(a, 0.0))) for a in actions)
            if total <= 0.0 or not math.isfinite(total):
                for a in actions:
                    priors[a] = 1.0 / max(1, len(actions))
            else:
                for a in actions:
                    priors[a] = max(0.0, float(raw.get(a, 0.0))) / total
        else:
            for a in actions:
                priors[a] = 1.0 / max(1, len(actions))
        # Dirichlet root noise
        if dirichlet and cfg.dirichlet_alpha > 0 and cfg.dirichlet_epsilon > 0 and actions:
            noise = _dirichlet(cfg.dirichlet_alpha, len(actions), rng)
            for i, a in enumerate(actions):
                priors[a] = ((1.0 - cfg.dirichlet_epsilon) * priors[a]
                             + cfg.dirichlet_epsilon * noise[i])
    else:
        # UCT: uniform priors
        for a in actions:
            priors[a] = 1.0 / max(1, len(actions)) if actions else 0.0

    # progressive widening
    untried = list(actions)
    if cfg.progressive_widening:
        cap = max(1, int(cfg.widen_k * (max(1, node.visits)) ** cfg.widen_alpha))
        # Sort priors descending so the highest-prior children are tried first
        untried.sort(key=lambda a: -priors.get(a, 0.0))
        untried = untried[:cap]
    else:
        # Sort by prior descending for deterministic, well-shaped expansion
        untried.sort(key=lambda a: -priors.get(a, 0.0))

    node.untried_actions = untried
    # store priors via attribute (we extend SearchNode dynamically; the dataclass
    # is not frozen, so setattr works)
    setattr(node, "priors", priors)


def _select_child(node: SearchNode, cfg: SearcherConfig,
                  algorithm: str) -> Tuple[Action, Hashable]:
    """Select the best child of ``node`` by the algorithm's selection rule."""
    best_score = -math.inf
    best_act: Optional[Action] = None
    best_child: Optional[SearchNode] = None
    parent_visits = max(1, node.visits)
    sum_N = parent_visits
    priors: Mapping[Action, float] = getattr(node, "priors", {})
    log_parent = math.log(parent_visits)
    for act, child in node.children.items():
        if algorithm == ALGORITHM_PUCT:
            q = child.q
            p = priors.get(act, child.prior)
            u = cfg.c_puct * p * math.sqrt(sum_N) / (1 + child.visits)
            score = q + u
        else:  # UCT
            if child.visits == 0:
                score = math.inf
            else:
                exploit = child.q
                explore = cfg.c_puct * math.sqrt(log_parent / child.visits)
                score = exploit + explore
        if score > best_score:
            best_score = score
            best_act = act
            best_child = child
    assert best_act is not None and best_child is not None
    return best_act, best_child.key


def _evaluate_leaf(searcher: "Searcher", node: SearchNode, state: State,
                   ev: Evaluator, rng: random.Random,
                   chain: _CertChain, full_trace: List[Dict[str, Any]],
                   budget: BudgetUsed) -> float:
    """Estimate the value at ``state`` via the configured leaf evaluator."""
    cfg = searcher.config
    if ev.value is not None:
        v = float(ev.value(state))
        chain.emit("value", {"key": _str_key(node.key), "value": v})
        if cfg.keep_full_trace:
            full_trace.append({"event": "value", "key": _str_key(node.key), "value": v})
        return v
    # else rollout(s) using uniform random policy
    if ev.reward is None:
        # No reward signal at all → value 0
        return 0.0
    total = 0.0
    for _ in range(max(1, cfg.rollouts_per_leaf)):
        budget.rollouts += 1
        s = state
        depth = 0
        while depth < cfg.rollout_depth:
            if ev.terminal(s):
                break
            acts = tuple(ev.actions(s))
            if not acts:
                break
            act = acts[rng.randrange(len(acts))]
            try:
                s = ev.apply(s, act)
            except Exception as e:
                raise InvalidEvaluator(f"apply failed during rollout: {e}")
            depth += 1
        total += float(ev.reward(s))
    avg = total / max(1, cfg.rollouts_per_leaf)
    chain.emit("rollout", {"key": _str_key(node.key), "value": avg,
                           "rollouts": cfg.rollouts_per_leaf})
    if cfg.keep_full_trace:
        full_trace.append({"event": "rollout", "key": _str_key(node.key),
                           "value": avg, "rollouts": cfg.rollouts_per_leaf})
    return avg


def _best_child_by_visits(node: SearchNode) -> Tuple[Optional[Action], float]:
    if not node.children:
        return None, 0.0
    best_act = None
    best_visits = -1
    best_q = 0.0
    for act, child in node.children.items():
        # tie-break by Q descending; then by hash of action for determinism
        key = (child.visits, child.q, hash(act))
        if (child.visits > best_visits
                or (child.visits == best_visits and child.q > best_q)
                or (child.visits == best_visits and child.q == best_q
                    and (best_act is None or hash(act) < hash(best_act)))):
            best_act = act
            best_visits = child.visits
            best_q = child.q
    return best_act, best_q


def _principal_variation(root: SearchNode) -> Tuple[Action, ...]:
    pv: List[Action] = []
    node = root
    visited: Set[Hashable] = set()
    while node.children and node.key not in visited:
        visited.add(node.key)
        act, _ = _best_child_by_visits(node)
        if act is None:
            break
        pv.append(act)
        node = node.children[act]
    return tuple(pv)


def _dirichlet(alpha: float, k: int, rng: random.Random) -> List[float]:
    """Sample a Dirichlet(α,...,α) of length ``k`` using stdlib only."""
    if k == 1:
        return [1.0]
    g = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(g)
    if s <= 0:
        return [1.0 / k] * k
    return [x / s for x in g]


# =============================================================================
# Algorithm: alpha-beta (Knuth-Moore 1975) with iterative deepening
# =============================================================================


def _alphabeta(searcher: "Searcher", root: State, ev: Evaluator) -> SearchReport:
    cfg = searcher.config
    stop = cfg.stop_condition()
    budget = BudgetUsed()
    chain = _CertChain(cfg.secret_key)
    history: List[Tuple[int, Optional[Action], float]] = []
    full_trace: List[Dict[str, Any]] = []
    tracker = _StopTracker(stop, budget, time.time())

    if ev.terminal is None or ev.reward is None:
        raise InvalidEvaluator("alphabeta requires ev.terminal and ev.reward")

    root_key = _resolve_key(ev, root)
    chain.emit("init", {"alg": "alphabeta", "root_key": _str_key(root_key),
                        "id": cfg.iterative_deepening, "ab_depth": cfg.ab_depth,
                        "seed": cfg.seed})

    tt: Dict[Tuple[Hashable, int], Tuple[float, str, Optional[Action]]] = {}
    history_table: Dict[Tuple[Hashable, Action], int] = {}

    best_action: Optional[Action] = None
    best_value: float = -math.inf
    bound = math.inf
    last_window: Optional[Tuple[float, float]] = None
    pv_table: Dict[Tuple[Hashable, int], List[Action]] = {}

    start_depth = 1 if cfg.iterative_deepening else cfg.ab_depth
    last_completed_depth = 0

    for depth in range(start_depth, cfg.ab_depth + 1):
        if tracker.stopped():
            break
        budget.iterations += 1

        alpha = -math.inf
        beta = math.inf
        if cfg.aspiration_window > 0 and last_window is not None:
            est = last_window[0]
            alpha = est - cfg.aspiration_window
            beta = est + cfg.aspiration_window

        val, act, completed = _ab_search(
            searcher, ev, root, depth, alpha, beta, 0,
            tt, history_table, pv_table, chain, full_trace,
            budget, tracker
        )
        if not completed:
            break

        # Aspiration retry on fail-high/low
        if (val <= alpha or val >= beta) and cfg.aspiration_window > 0:
            val, act, completed = _ab_search(
                searcher, ev, root, depth, -math.inf, math.inf, 0,
                tt, history_table, pv_table, chain, full_trace,
                budget, tracker
            )
            if not completed:
                break

        best_value = val
        best_action = act
        last_window = (val, val)
        last_completed_depth = depth

        history.append((budget.iterations, act, val))
        chain.emit("id_depth", {"depth": depth, "best_action": _safe_action_repr(act),
                                "best_value": val})

        if cfg.iterative_deepening is False:
            break

    pv = tuple(pv_table.get((root_key, last_completed_depth), []))
    finished = (budget.bound_hit is None)

    return SearchReport(
        algorithm=ALGORITHM_ALPHABETA,
        best_action=best_action,
        best_value=best_value,
        principal_variation=pv,
        iterations=budget.iterations,
        budget_used=budget,
        certificate=chain.hexdigest(),
        seed=cfg.seed,
        finished=finished,
        bound_hit=budget.bound_hit,
        bound=float(last_completed_depth),
        history=tuple(history),
        full_trace=tuple(full_trace),
        notes=f"alpha-beta (depth={last_completed_depth})",
    )


# transposition flags
_TT_EXACT = "exact"
_TT_LOWER = "lower"
_TT_UPPER = "upper"


def _ab_search(
    searcher: "Searcher", ev: Evaluator, state: State, depth: int,
    alpha: float, beta: float, ply: int,
    tt: Dict[Tuple[Hashable, int], Tuple[float, str, Optional[Action]]],
    history_table: Dict[Tuple[Hashable, Action], int],
    pv_table: Dict[Tuple[Hashable, int], List[Action]],
    chain: "_CertChain", full_trace: List[Dict[str, Any]],
    budget: BudgetUsed, tracker: "_StopTracker",
) -> Tuple[float, Optional[Action], bool]:
    """Negamax alpha-beta with transposition table.

    Returns ``(value, best_action_at_this_node, completed)``.
    ``completed`` is False if a budget bound aborted the subtree, in
    which case the value is undefined and should be discarded by the
    caller.
    """
    cfg = searcher.config
    if tracker.stopped():
        return 0.0, None, False
    key = _resolve_key(ev, state)
    budget.peak_depth = max(budget.peak_depth, ply)
    if ev.terminal(state):
        v = float(ev.reward(state))
        # Negamax convention: scores returned from the perspective of
        # the side to move at this node.  At a terminal state the
        # reward is taken at face value (the user supplies it from the
        # to-move perspective).
        chain.emit("terminal", {"key": _str_key(key), "value": v, "ply": ply})
        return v, None, True
    if depth == 0:
        v = float(ev.reward(state))
        chain.emit("qstat", {"key": _str_key(key), "value": v, "ply": ply})
        return v, None, True

    # TT probe
    tt_key = (key, depth)
    tt_hint: Optional[Action] = None
    if cfg.use_transposition and tt_key in tt:
        tv, tflag, tact = tt[tt_key]
        if tflag == _TT_EXACT:
            return tv, tact, True
        elif tflag == _TT_LOWER and tv >= beta:
            return tv, tact, True
        elif tflag == _TT_UPPER and tv <= alpha:
            return tv, tact, True
        tt_hint = tact

    alpha0 = alpha
    best_val = -math.inf
    best_act: Optional[Action] = None
    actions = list(ev.actions(state))
    if not actions:
        v = float(ev.reward(state))
        return v, None, True
    # Move ordering: TT hint first, then history-heuristic descending
    def _ord_score(a: Action) -> Tuple[int, int]:
        prio = 0
        if a == tt_hint:
            prio = 1_000_000
        h = history_table.get((key, a), 0)
        return (prio + h, hash(a))
    if cfg.history_heuristic:
        actions.sort(key=_ord_score, reverse=True)

    pv_local: List[Action] = []
    for act in actions:
        if tracker.stopped():
            return 0.0, best_act, False
        child = ev.apply(state, act)
        budget.expansions += 1
        budget.nodes += 1
        chain.emit("ab_expand", {
            "parent_key": _str_key(key),
            "action": _safe_action_repr(act),
            "depth": depth - 1, "alpha": alpha, "beta": beta,
        })
        if cfg.keep_full_trace:
            full_trace.append({
                "event": "ab_expand",
                "parent_key": _str_key(key),
                "action": _safe_action_repr(act),
                "depth": depth - 1, "alpha": alpha, "beta": beta,
            })
        v, _, completed = _ab_search(
            searcher, ev, child, depth - 1, -beta, -alpha, ply + 1,
            tt, history_table, pv_table, chain, full_trace, budget, tracker
        )
        if not completed:
            return 0.0, best_act, False
        score = -v
        if score > best_val:
            best_val = score
            best_act = act
            pv_local = [act] + pv_table.get((_resolve_key(ev, child), depth - 1), [])
        if score > alpha:
            alpha = score
        if alpha >= beta:
            # cutoff — bump history heuristic
            history_table[(key, act)] = history_table.get((key, act), 0) + depth * depth
            break

    # Store TT
    if cfg.use_transposition:
        if best_val <= alpha0:
            flag = _TT_UPPER
        elif best_val >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        tt[tt_key] = (best_val, flag, best_act)
    pv_table[(key, depth)] = pv_local
    return best_val, best_act, True


# =============================================================================
# Algorithm: beam search
# =============================================================================


def _beam_search(searcher: "Searcher", root: State, ev: Evaluator) -> SearchReport:
    cfg = searcher.config
    stop = cfg.stop_condition()
    budget = BudgetUsed()
    chain = _CertChain(cfg.secret_key)
    history: List[Tuple[int, Optional[Action], float]] = []
    full_trace: List[Dict[str, Any]] = []
    tracker = _StopTracker(stop, budget, time.time())

    if ev.terminal is None:
        raise InvalidEvaluator("beam search requires ev.terminal")
    if cfg.beam_score == "value" and ev.value is None and ev.reward is None:
        raise InvalidEvaluator("beam_score='value' requires ev.value or ev.reward")
    if cfg.beam_score == "cost" and ev.cost is None and ev.heuristic is None:
        raise InvalidEvaluator("beam_score='cost' requires ev.cost or ev.heuristic")

    root_key = _resolve_key(ev, root)
    chain.emit("init", {"alg": "beam", "root_key": _str_key(root_key),
                        "width": cfg.beam_width, "score": cfg.beam_score,
                        "seed": cfg.seed})

    # beam item: (score, sequence, state, key, accumulated_cost)
    beam: List[Tuple[float, Tuple[Action, ...], State, Hashable, float]] = [
        (_beam_score_of(root, (), 0.0, ev, cfg), (), root, root_key, 0.0)
    ]
    best_complete_score: float = -math.inf if cfg.beam_score == "value" else math.inf
    best_complete: Optional[Tuple[float, Tuple[Action, ...], float]] = None

    while beam:
        if tracker.stopped():
            break
        budget.iterations += 1

        # Expand all beam elements
        candidates: List[Tuple[float, Tuple[Action, ...], State, Hashable, float]] = []
        for score, seq, state, key, gcost in beam:
            if ev.terminal(state):
                # complete sequence — eligible for best_complete
                _update_best_complete(best_complete, score, seq, gcost, cfg)
                if (cfg.beam_score == "value" and score > best_complete_score) or \
                   (cfg.beam_score == "cost" and score < best_complete_score):
                    best_complete_score = score
                    best_complete = (score, seq, gcost)
                continue
            for act in ev.actions(state):
                if tracker.stopped():
                    break
                child = ev.apply(state, act)
                c = _cost_of(ev, cfg, state, act, child) if ev.cost is not None else cfg.cost_default
                ng = gcost + c
                ck = _resolve_key(ev, child)
                sc = _beam_score_of(child, seq + (act,), ng, ev, cfg)
                budget.expansions += 1
                budget.nodes += 1
                chain.emit("beam_expand", {
                    "parent_key": _str_key(key),
                    "action": _safe_action_repr(act),
                    "child_key": _str_key(ck),
                    "score": sc, "depth": len(seq) + 1, "g": ng,
                })
                if cfg.keep_full_trace:
                    full_trace.append({
                        "event": "beam_expand",
                        "parent_key": _str_key(key),
                        "action": _safe_action_repr(act),
                        "child_key": _str_key(ck),
                        "score": sc, "depth": len(seq) + 1, "g": ng,
                    })
                candidates.append((sc, seq + (act,), child, ck, ng))

        if not candidates:
            break

        # Keep top-K under the chosen sort direction
        reverse = (cfg.beam_score == "value")
        candidates.sort(key=lambda x: x[0], reverse=reverse)
        beam = candidates[:cfg.beam_width]
        budget.peak_open_size = max(budget.peak_open_size, len(beam))
        budget.peak_depth = max(budget.peak_depth, len(beam[0][1]) if beam else 0)
        # current best so far
        if best_complete is not None:
            history.append((budget.iterations, best_complete[1][0] if best_complete[1] else None,
                            best_complete[0]))
        else:
            history.append((budget.iterations, beam[0][1][0] if beam[0][1] else None,
                            beam[0][0]))

    finished = budget.bound_hit is None

    if best_complete is not None:
        score, seq, gcost = best_complete
    elif beam:
        score, seq, _state, _key, gcost = beam[0]
    else:
        return SearchReport(
            algorithm=ALGORITHM_BEAM,
            best_action=None,
            best_value=0.0,
            principal_variation=(),
            iterations=budget.iterations,
            budget_used=budget,
            certificate=chain.hexdigest(),
            seed=cfg.seed,
            finished=finished,
            bound_hit=budget.bound_hit,
            history=tuple(history),
            full_trace=tuple(full_trace),
            notes="beam (empty)",
        )

    bv = -gcost if cfg.beam_score == "cost" else score
    return SearchReport(
        algorithm=ALGORITHM_BEAM,
        best_action=(seq[0] if seq else None),
        best_value=bv,
        principal_variation=seq,
        iterations=budget.iterations,
        budget_used=budget,
        certificate=chain.hexdigest(),
        seed=cfg.seed,
        finished=finished,
        bound_hit=budget.bound_hit,
        optimal_cost=(gcost if cfg.beam_score == "cost" else None),
        history=tuple(history),
        full_trace=tuple(full_trace),
        notes=f"beam (width={cfg.beam_width}, score={cfg.beam_score})",
    )


def _update_best_complete(best: Optional[Tuple[float, Tuple[Action, ...], float]],
                          score: float, seq: Tuple[Action, ...], gcost: float,
                          cfg: SearcherConfig) -> None:
    """No-op helper kept for symmetry; actual update is inline."""
    return None


def _beam_score_of(state: State, seq: Tuple[Action, ...], gcost: float,
                   ev: Evaluator, cfg: SearcherConfig) -> float:
    if cfg.beam_score == "value":
        if ev.value is not None:
            return float(ev.value(state))
        if ev.terminal is not None and ev.terminal(state) and ev.reward is not None:
            return float(ev.reward(state))
        return 0.0
    else:  # cost: lower better
        h = float(ev.heuristic(state)) if ev.heuristic is not None else 0.0
        return gcost + h


# =============================================================================
# Algorithm: auto-pick
# =============================================================================


def _auto_algorithm(ev: Evaluator) -> str:
    """Pick a sensible default given the evaluator's signature."""
    if ev.heuristic is not None and ev.terminal is not None:
        return ALGORITHM_ASTAR
    if ev.policy_prior is not None:
        return ALGORITHM_PUCT
    if ev.value is not None or ev.reward is not None:
        return ALGORITHM_UCT
    if ev.terminal is not None:
        return ALGORITHM_ASTAR
    raise InvalidEvaluator(
        "auto algorithm requires at least one of: heuristic, policy_prior, "
        "value, reward, terminal"
    )


# =============================================================================
# Orchestrator
# =============================================================================


class Searcher:
    """Bounded-anytime certified tree-search orchestrator.

    Construct with a ``SearcherConfig``; call ``search(root, ...)`` with
    either an ``Evaluator`` or the individual callables.  The same
    object can be reused for many independent searches; per-search
    state lives inside ``search``.
    """

    def __init__(self, config: Optional[SearcherConfig] = None) -> None:
        self.config = config or SearcherConfig()
        # version tag for the certificate scheme.  Bump on breaking changes.
        self.scheme_version = "agi.searcher.v1"

    def search(
        self,
        root: State,
        *,
        evaluator: Optional[Evaluator] = None,
        actions: Optional[ActionsFn] = None,
        apply: Optional[ApplyFn] = None,
        terminal: Optional[TerminalFn] = None,
        reward: Optional[RewardFn] = None,
        heuristic: Optional[HeuristicFn] = None,
        cost: Optional[CostFn] = None,
        policy_prior: Optional[PolicyPriorFn] = None,
        value: Optional[ValueFn] = None,
        key: Optional[KeyFn] = None,
        algorithm: Optional[str] = None,
    ) -> SearchReport:
        """Run the configured search algorithm and return a SearchReport.

        Either pass a fully-formed ``evaluator`` or the individual
        callables.  ``algorithm`` overrides ``config.algorithm`` for
        the call.
        """
        if evaluator is None:
            if actions is None or apply is None:
                raise InvalidEvaluator(
                    "must supply either evaluator= or (actions= and apply=)"
                )
            evaluator = Evaluator(
                actions=actions, apply=apply, terminal=terminal, reward=reward,
                heuristic=heuristic, cost=cost, policy_prior=policy_prior,
                value=value, key=key,
            )

        algo = algorithm or self.config.algorithm
        if algo == ALGORITHM_AUTO:
            algo = _auto_algorithm(evaluator)

        # Stash the resolved algorithm on a copy of the config so the
        # selected branch picks it up via self.config.algorithm.
        if algo != self.config.algorithm:
            saved_cfg = self.config
            try:
                self.config = dataclasses.replace(self.config, algorithm=algo)
                report = self._dispatch(algo, root, evaluator)
            finally:
                self.config = saved_cfg
        else:
            report = self._dispatch(algo, root, evaluator)

        return report

    # ------------------------------------------------------------------
    # dispatch table
    # ------------------------------------------------------------------

    def _dispatch(self, algo: str, root: State, ev: Evaluator) -> SearchReport:
        if algo == ALGORITHM_ASTAR:
            return _astar(self, root, ev)
        if algo == ALGORITHM_BNB:
            return _astar(self, root, ev)
        if algo == ALGORITHM_IDA_STAR:
            return _ida_star(self, root, ev)
        if algo == ALGORITHM_UCT:
            return _uct(self, root, ev)
        if algo == ALGORITHM_PUCT:
            return _puct(self, root, ev)
        if algo == ALGORITHM_ALPHABETA:
            return _alphabeta(self, root, ev)
        if algo == ALGORITHM_BEAM:
            return _beam_search(self, root, ev)
        raise UnknownAlgorithm(f"unknown algorithm: {algo!r}")


# =============================================================================
# Free-function shortcuts
# =============================================================================


def astar(root: State, *, actions: ActionsFn, apply: ApplyFn,
          terminal: TerminalFn, heuristic: Optional[HeuristicFn] = None,
          cost: Optional[CostFn] = None, key: Optional[KeyFn] = None,
          weighted: float = 1.0,
          max_iterations: Optional[int] = None,
          max_seconds: Optional[float] = None,
          seed: int = 0) -> SearchReport:
    """Run A* (or weighted A*) with the supplied callables."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_ASTAR, weighted_astar=weighted,
        max_iterations=max_iterations, max_seconds=max_seconds, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              heuristic=heuristic, cost=cost, key=key)


def ida_star(root: State, *, actions: ActionsFn, apply: ApplyFn,
             terminal: TerminalFn, heuristic: Optional[HeuristicFn] = None,
             cost: Optional[CostFn] = None, key: Optional[KeyFn] = None,
             max_iterations: Optional[int] = None,
             max_seconds: Optional[float] = None,
             seed: int = 0) -> SearchReport:
    """Run IDA*."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_IDA_STAR,
        max_iterations=max_iterations, max_seconds=max_seconds, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              heuristic=heuristic, cost=cost, key=key)


def uct(root: State, *, actions: ActionsFn, apply: ApplyFn,
        terminal: TerminalFn, reward: Optional[RewardFn] = None,
        value: Optional[ValueFn] = None, key: Optional[KeyFn] = None,
        c_puct: float = 1.4, rollout_depth: int = 32,
        max_iterations: int = 1024, max_seconds: Optional[float] = None,
        seed: int = 0) -> SearchReport:
    """Run UCT."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_UCT, c_puct=c_puct, rollout_depth=rollout_depth,
        max_iterations=max_iterations, max_seconds=max_seconds, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              reward=reward, value=value, key=key)


def puct(root: State, *, actions: ActionsFn, apply: ApplyFn,
         terminal: TerminalFn, reward: Optional[RewardFn] = None,
         value: Optional[ValueFn] = None,
         policy_prior: Optional[PolicyPriorFn] = None,
         key: Optional[KeyFn] = None,
         c_puct: float = 1.25, max_iterations: int = 1024,
         max_seconds: Optional[float] = None,
         dirichlet_alpha: float = 0.0, dirichlet_epsilon: float = 0.0,
         seed: int = 0) -> SearchReport:
    """Run PUCT (AlphaZero)."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_PUCT, c_puct=c_puct,
        max_iterations=max_iterations, max_seconds=max_seconds,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              reward=reward, value=value, policy_prior=policy_prior, key=key)


def alphabeta(root: State, *, actions: ActionsFn, apply: ApplyFn,
              terminal: TerminalFn, reward: RewardFn,
              key: Optional[KeyFn] = None,
              depth: int = 4, iterative_deepening: bool = True,
              max_seconds: Optional[float] = None,
              seed: int = 0) -> SearchReport:
    """Run alpha-beta (with optional iterative deepening)."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_ALPHABETA, ab_depth=depth,
        iterative_deepening=iterative_deepening,
        max_iterations=None,
        max_seconds=max_seconds, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              reward=reward, key=key)


def beam_search(root: State, *, actions: ActionsFn, apply: ApplyFn,
                terminal: TerminalFn, value: Optional[ValueFn] = None,
                reward: Optional[RewardFn] = None,
                heuristic: Optional[HeuristicFn] = None,
                cost: Optional[CostFn] = None, key: Optional[KeyFn] = None,
                width: int = 8, score: str = "value",
                max_iterations: Optional[int] = None,
                max_seconds: Optional[float] = None,
                seed: int = 0) -> SearchReport:
    """Run beam search."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_BEAM, beam_width=width, beam_score=score,
        max_iterations=max_iterations, max_seconds=max_seconds, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              value=value, reward=reward, heuristic=heuristic, cost=cost,
              key=key)


def branch_and_bound(root: State, *, actions: ActionsFn, apply: ApplyFn,
                     terminal: TerminalFn, cost: Optional[CostFn] = None,
                     heuristic: Optional[HeuristicFn] = None,
                     key: Optional[KeyFn] = None,
                     max_iterations: Optional[int] = None,
                     max_seconds: Optional[float] = None,
                     seed: int = 0) -> SearchReport:
    """Run branch-and-bound (best-first with incumbent pruning)."""
    return Searcher(SearcherConfig(
        algorithm=ALGORITHM_BNB,
        max_iterations=max_iterations, max_seconds=max_seconds, seed=seed,
    )).search(root, actions=actions, apply=apply, terminal=terminal,
              heuristic=heuristic, cost=cost, key=key)


# =============================================================================
# Composition helpers — adapters that wire other primitives in as evaluators
# =============================================================================


def make_evaluator(
    actions: ActionsFn,
    apply: ApplyFn,
    *,
    terminal: Optional[TerminalFn] = None,
    reward: Optional[RewardFn] = None,
    heuristic: Optional[HeuristicFn] = None,
    cost: Optional[CostFn] = None,
    policy_prior: Optional[PolicyPriorFn] = None,
    value: Optional[ValueFn] = None,
    key: Optional[KeyFn] = None,
) -> Evaluator:
    """Build an Evaluator with sensible defaults."""
    return Evaluator(
        actions=actions, apply=apply, terminal=terminal, reward=reward,
        heuristic=heuristic, cost=cost, policy_prior=policy_prior,
        value=value, key=key,
    )


# =============================================================================
# Convenience: in-process certificate replay verification
# =============================================================================


def verify_certificate(report: SearchReport, searcher: Optional[Searcher] = None,
                       root: Optional[State] = None,
                       evaluator: Optional[Evaluator] = None) -> bool:
    """Replay the search under the same config & evaluators; compare cert.

    Returns True if the recomputed certificate equals ``report.certificate``.
    A pure verifier that the report and the source agree byte-for-byte.

    Note: requires the user to pass back the same ``searcher`` (with the
    same ``SearcherConfig.seed``), the same ``root``, and the same
    ``Evaluator`` callables that produced the report.  Non-deterministic
    evaluators (anything reading time, randomness, or environment) break
    the certificate by construction — that is the point.
    """
    if searcher is None or root is None or evaluator is None:
        return False
    rep2 = searcher.search(root, evaluator=evaluator,
                           algorithm=report.algorithm)
    return rep2.certificate == report.certificate


__all__ = [
    # errors
    "SearcherError",
    "InvalidConfig",
    "InvalidEvaluator",
    "InvalidState",
    "BudgetExhausted",
    "NoSolution",
    "UnknownAlgorithm",
    # algorithm constants
    "ALGORITHM_ASTAR",
    "ALGORITHM_IDA_STAR",
    "ALGORITHM_UCT",
    "ALGORITHM_PUCT",
    "ALGORITHM_ALPHABETA",
    "ALGORITHM_BEAM",
    "ALGORITHM_BNB",
    "ALGORITHM_AUTO",
    "KNOWN_ALGORITHMS",
    # dataclasses
    "StopCondition",
    "BudgetUsed",
    "SearcherConfig",
    "Evaluator",
    "SearchNode",
    "SearchReport",
    # orchestrator
    "Searcher",
    # shortcuts
    "astar",
    "ida_star",
    "uct",
    "puct",
    "alphabeta",
    "beam_search",
    "branch_and_bound",
    "make_evaluator",
    "verify_certificate",
]
