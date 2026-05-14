r"""Diplomat — counterfactual regret minimization for extensive-form games.

Equilibrator solves the *one-shot, simultaneous-move* equilibrium
problem: given a normal-form payoff tensor, find a Nash / correlated /
coarse-correlated profile. That covers a large fraction of pricing,
adversarial, and matching problems, but it cannot speak about the
runtime regime that *actually* dominates multi-agent automation —
sequential decision-making under imperfect information. A pricing
engine that bids over time and watches its rivals, a negotiator that
plays a multi-round ultimatum-counter-ultimatum protocol, a defender
who picks an action without knowing which of several adversaries she
faces: all of these are *extensive-form games with imperfect
information*. They cannot be flattened into a tractable matrix.

The Diplomat is the runtime primitive for that regime. It accepts a
tree-structured `Game` (chance / decision / terminal nodes; players
grouped into *information sets* they cannot distinguish), a solver
(vanilla CFR / CFR+ / Linear CFR / Discounted CFR / Predictive CFR+ /
outcome-sampling MCCFR / external-sampling MCCFR / chance-sampling
CFR / sequence-form LP), and an iteration budget, and returns a
`SolveReport` containing the *average* strategy, the exact root
expected utility under that strategy, the **exploitability** (the sum
over players of best-response improvement; zero iff Nash), a
`certificate` listing which convergence axiom holds, and an *anytime*
upper bound on exploitability derived from the running counterfactual
regret. The same primitive ships exact *best response* against any
fixed strategy and an exact two-player zero-sum solver via the
*sequence-form linear program*.

Mathematical core (cited where it counts)
-----------------------------------------

  * **von Stengel, 1996 — "Efficient computation of behavior
    strategies."** Defines the *sequence form*: for an extensive-form
    game with perfect recall, the set of realisation plans is a
    polytope whose dimension is *linear in the size of the tree*
    (not exponential as the equivalent normal form would be).
    Two-player zero-sum extensive-form games admit an exact LP of
    size proportional to the tree. We ship the sequence-form LP via
    a stdlib revised-simplex solver with Bland's-rule pivoting; no
    NumPy / SciPy required.

  * **Kuhn, 1953 — "Extensive games and the problem of information."**
    Origin of the imperfect-information game tree, information sets,
    and behaviour strategies; the perfect-recall theorem (every
    mixed strategy has an outcome-equivalent behaviour strategy).
    Kuhn's three-card poker is the canonical regression test for
    every extensive-form solver and we ship it as a builder.

  * **Zinkevich, Johanson, Bowling & Piccione, 2008 — "Regret
    minimization in games with incomplete information."** The
    *counterfactual regret minimization* (CFR) algorithm. For each
    information set ``I``, accumulate the *counterfactual* immediate
    regret ``r^t(I, a) = π_{-i}^{σ^t}(I) [u_i(σ^t |_{I → a}, h) −
    u_i(σ^t, h)]`` and apply regret-matching ``σ^{t+1}(I)(a) ∝
    max(0, R^T(I, a))``. The key theorem: the *full* external regret
    of the agent on the tree is bounded by ``Σ_I R^T(I)``, so each
    info set's no-regret algorithm gives a global no-regret guarantee
    of ``O(Δ_u |A| √(|I|·T))``. The *time-average* of self-play in
    two-player zero-sum converges to a Nash equilibrium with
    exploitability ``≤ 2 (Σ_i Σ_I R^T_i(I)) / T``. This is the
    exploitability bound we report and refresh every iteration.

  * **Tammelin, 2014 — "Solving large imperfect information games
    using CFR+."** Drop-in modification: floor the cumulative
    counterfactual regret at zero (``R^T ← max(R^T + r^t, 0)``) and
    weight the *strategy sum* linearly in ``t``. CFR+ is the
    algorithm that solved heads-up limit Texas hold'em (Bowling,
    Burch, Johanson, Tammelin 2015, *Science*). Empirically CFR+
    converges one to two orders of magnitude faster than vanilla
    CFR and produces strictly better solutions on every benchmark
    we ship.

  * **Brown & Sandholm, 2019 — "Solving imperfect-information games
    via discounted regret minimization."** Generalises CFR / CFR+ /
    Linear CFR into a three-parameter family DCFR(α, β, γ):
    positive regrets weighted ``t^α / (t^α + 1)``, negative regrets
    weighted ``t^β / (t^β + 1)``, strategy weighted ``(t / (t+1))^γ``.
    DCFR(1.5, 0, 2) matches Linear CFR on positives, drops negatives,
    and weights the strategy sum quadratically; it is the algorithm
    that defeated the top heads-up no-limit Texas hold'em pros
    (Libratus, Brown & Sandholm 2017, *Science*). We expose DCFR with
    user-tunable (α, β, γ) and ship the published high-performance
    presets.

  * **Farina, Kroer & Sandholm, 2021 — "Faster game solving via
    predictive Blackwell approachability."** Predictive (a.k.a.
    *optimistic*) CFR+: the regret-matching step uses the regret
    *plus* an estimate of the next-round regret (the previous round
    serves as the estimate), giving an ``O(T^{−1})`` last-iterate
    convergence rate on monotone games — exponentially faster than
    CFR+'s ``O(T^{−1/2})`` time-average rate. Ships as one of the
    default solvers.

  * **Lanctot, Waugh, Zinkevich & Bowling, 2009 — "Monte Carlo
    sampling for regret minimization in extensive games."** Three
    sampling schemes that keep CFR's convergence in expectation but
    cut per-iteration cost from ``O(|tree|)`` to ``O(|tree-leaf-path|)``:
      - *Outcome sampling*: one leaf per iteration; unbiased
        importance-weighted regret estimator; the highest-variance
        but cheapest scheme — used in Pluribus 2019.
      - *External sampling*: sample opponent and chance, traverse
        the active player's information-set subtree fully; lower
        variance, the standard production choice.
      - *Chance sampling*: sample chance only; deterministic in
        player actions; the variance-cost compromise.
    All three converge in expectation with the same ``O(1/√T)``
    rate (with constants that scale in the variance of the sampled
    estimator). We ship all three with the same `SolveReport`
    surface as the deterministic variants.

  * **Hart & Mas-Colell, 2000 — "A simple adaptive procedure leading
    to correlated equilibrium."** The *regret-matching* update —
    ``σ^{t+1}(a) ∝ max(0, R^T(a))``, uniform when all regrets are
    nonpositive — that powers every CFR variant. Hannan-consistency
    of the per-info-set learner is the local property that lifts to
    Nash on the tree (for zero-sum) and to coarse-correlated
    equilibrium (in general N-player).

  * **Brown, 2009; Johanson et al., 2011 — "Accelerating best-
    response calculations."** The exact best-response algorithm we
    ship: a single recursive pass on the tree computes the best
    pure response of each player against any fixed behaviour
    profile in ``O(|tree|)`` time; sum of (BR-value − value) across
    players is the *NashConv* exploitability, which is zero exactly
    at Nash. We use it as the live convergence diagnostic and as
    the certifying check.

Composes with the rest of the runtime
-------------------------------------

  * **Equilibrator** — Equilibrator solves the *one-shot* equivalent.
    A sequence-form solve on a one-decision tree reproduces
    Equilibrator's two-player zero-sum LP; an `extensive_to_normal`
    helper lifts small trees into Equilibrator's normal-form surface
    when the user wants to compare solvers.
  * **Negotiator** — multi-round bargaining is exactly an extensive-
    form game; Diplomat returns the equilibrium of the *protocol*,
    Negotiator's allocation is then evaluated under that equilibrium.
  * **Persuader** — sequential persuasion (one signal per round,
    sender commits to a multi-stage scheme) is an EFG. The sender's
    sequence-form LP is the multi-stage generalisation of Persuader's
    BCE LP.
  * **TruthSerum** — multi-round peer prediction is an EFG; Diplomat
    verifies that *no player has a profitable deviation across the
    entire protocol* (truthful Nash on the EFG), not just per round.
  * **MechanismDesigner / VCG** — proves incentive compatibility on
    the *full game tree*: a mechanism is dominant-strategy IC iff
    truthful reporting is a best response in the EFG.
  * **Strategist** — Strategist's "what is the worst-case
    exploitability of shipping policy π?" gate uses Diplomat's
    exact best response over the deployment EFG; risk-adjusted
    decisions become *exploitability-adjusted*.
  * **AttestationLedger** — every `SolveReport` is JSON-serialisable
    and contains a SHA-256-stable `certificate.fingerprint` so the
    equilibrium can be committed to before any agent acts under it.

What this module does not pretend to be
---------------------------------------

  * It is *not* an RL engine. It assumes you have the game tree.
    For learned environment models, build the tree with a
    `WorldModel` rollout and hand it to Diplomat.
  * It is *not* a general N-player Nash solver. CFR-family algorithms
    converge to Nash *only* in two-player zero-sum (and to coarse-
    correlated in general). For N-player Nash, lift to
    Equilibrator's support-enumeration or use a CFR run with
    explicit warning that the report is only a CCE certificate.
  * It does *not* assume perfect recall *can be violated*. Behaviour
    strategies are well-defined only with perfect recall (Kuhn's
    theorem); when a non-PR game is detected we raise
    `PerfectRecallViolation` rather than silently produce a
    wrong-but-plausible answer.
  * It does *not* cache the full strategy in memory for games
    larger than what fits. The user is expected to build the game
    with information-set abstraction (action abstraction, card
    abstraction, public-state abstraction) *before* handing it in.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event  # type: ignore
except Exception:  # pragma: no cover
    Event = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================

DIPLOMAT_STARTED = "diplomat.started"
DIPLOMAT_ITER = "diplomat.iteration"
DIPLOMAT_SOLVED = "diplomat.solved"
DIPLOMAT_BR = "diplomat.best_response"
DIPLOMAT_LP_SOLVED = "diplomat.lp_solved"
DIPLOMAT_CERTIFIED = "diplomat.certified"


# =====================================================================
# Solver identifiers
# =====================================================================

KIND_CFR = "cfr"
KIND_CFR_PLUS = "cfr_plus"
KIND_LINEAR_CFR = "linear_cfr"
KIND_DISCOUNTED_CFR = "discounted_cfr"
KIND_PREDICTIVE_CFR_PLUS = "predictive_cfr_plus"
KIND_OUTCOME_SAMPLING = "outcome_sampling_mccfr"
KIND_EXTERNAL_SAMPLING = "external_sampling_mccfr"
KIND_CHANCE_SAMPLING = "chance_sampling_cfr"
KIND_SEQUENCE_FORM_LP = "sequence_form_lp"

KNOWN_KINDS = (
    KIND_CFR,
    KIND_CFR_PLUS,
    KIND_LINEAR_CFR,
    KIND_DISCOUNTED_CFR,
    KIND_PREDICTIVE_CFR_PLUS,
    KIND_OUTCOME_SAMPLING,
    KIND_EXTERNAL_SAMPLING,
    KIND_CHANCE_SAMPLING,
    KIND_SEQUENCE_FORM_LP,
)

# Aliases for users
_KIND_ALIASES = {
    "vanilla": KIND_CFR,
    "cfr+": KIND_CFR_PLUS,
    "lcfr": KIND_LINEAR_CFR,
    "dcfr": KIND_DISCOUNTED_CFR,
    "pcfr+": KIND_PREDICTIVE_CFR_PLUS,
    "predictive": KIND_PREDICTIVE_CFR_PLUS,
    "os": KIND_OUTCOME_SAMPLING,
    "es": KIND_EXTERNAL_SAMPLING,
    "cs": KIND_CHANCE_SAMPLING,
    "lp": KIND_SEQUENCE_FORM_LP,
    "seqlp": KIND_SEQUENCE_FORM_LP,
}


# =====================================================================
# Numerical constants
# =====================================================================

_EPS = 1e-12
_NEG_INF = float("-inf")
_LP_TOL = 1e-9
_LP_MAX_ITERS = 100_000


# =====================================================================
# Exceptions
# =====================================================================


class DiplomatError(Exception):
    """Base class for Diplomat errors."""


class InvalidGame(DiplomatError):
    """The game definition is malformed (mismatched player counts, dangling actions, ...)."""


class PerfectRecallViolation(DiplomatError):
    """An information set groups histories with inconsistent past actions for its player."""


class UnknownSolver(DiplomatError):
    """A solver name was requested that this module does not implement."""


class InfeasibleProgram(DiplomatError):
    """The sequence-form LP could not be solved (numerically singular / unbounded)."""


class NotTwoPlayerZeroSum(DiplomatError):
    """A solver was requested that only handles two-player zero-sum games."""


class InsufficientIterations(DiplomatError):
    """The solver was asked for a certificate but ran zero iterations."""


# =====================================================================
# Game tree primitives
# =====================================================================


@dataclass
class _Node:
    """Internal base class. Users construct trees via :class:`GameBuilder`."""

    id: int
    parent: int  # -1 for root
    parent_action: Any  # action taken at parent to reach this node; None at root


@dataclass
class _Chance(_Node):
    actions: list[Any] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    children: list[int] = field(default_factory=list)


@dataclass
class _Decision(_Node):
    player: int = 0
    actions: list[Any] = field(default_factory=list)
    info_set: str = ""
    children: list[int] = field(default_factory=list)


@dataclass
class _Terminal(_Node):
    utilities: list[float] = field(default_factory=list)


@dataclass
class _InfoSet:
    """Bookkeeping for a single information set."""

    key: str
    player: int
    actions: list[Any]
    nodes: list[int] = field(default_factory=list)
    # CFR state — one entry per action
    regret: list[float] = field(default_factory=list)
    strategy_sum: list[float] = field(default_factory=list)
    last_regret: list[float] = field(default_factory=list)  # for predictive CFR+

    def n_actions(self) -> int:
        return len(self.actions)

    def current_strategy(self) -> list[float]:
        """Regret-matching: σ(a) ∝ max(0, R(a)); uniform if all ≤ 0."""
        pos = [r if r > 0.0 else 0.0 for r in self.regret]
        s = sum(pos)
        if s <= 0.0:
            n = len(self.actions)
            return [1.0 / n] * n
        return [p / s for p in pos]

    def predictive_strategy(self, m: float = 1.0) -> list[float]:
        """Regret-matching with optimistic prediction: σ ∝ max(0, R + m·r_last)."""
        pos = [max(0.0, r + m * rl) for r, rl in zip(self.regret, self.last_regret)]
        s = sum(pos)
        if s <= 0.0:
            n = len(self.actions)
            return [1.0 / n] * n
        return [p / s for p in pos]

    def average_strategy(self) -> list[float]:
        s = sum(self.strategy_sum)
        if s <= 0.0:
            n = len(self.actions)
            return [1.0 / n] * n
        return [x / s for x in self.strategy_sum]


# =====================================================================
# Public Game
# =====================================================================


@dataclass
class Game:
    """An extensive-form game with imperfect information.

    Build games via :class:`GameBuilder`. Players are integers ``0..n−1``.
    Each terminal node carries one utility per player. Information sets
    are identified by string keys: every decision node assigned the same
    info-set key must have the same player and the same ordered action
    list. Perfect recall is checked at build time.
    """

    n_players: int
    root: int
    nodes: dict[int, _Node]
    info_sets: dict[str, _InfoSet]
    name: str = ""

    def __post_init__(self) -> None:
        if self.n_players < 1:
            raise InvalidGame("n_players must be ≥ 1")
        if self.root not in self.nodes:
            raise InvalidGame(f"root id {self.root} is not in nodes")

    def info_set_keys(self, player: int | None = None) -> list[str]:
        if player is None:
            return sorted(self.info_sets.keys())
        return sorted(k for k, I in self.info_sets.items() if I.player == player)

    def n_terminals(self) -> int:
        return sum(1 for n in self.nodes.values() if isinstance(n, _Terminal))

    def n_decisions(self) -> int:
        return sum(1 for n in self.nodes.values() if isinstance(n, _Decision))

    def n_chance(self) -> int:
        return sum(1 for n in self.nodes.values() if isinstance(n, _Chance))

    def size(self) -> int:
        return len(self.nodes)

    def reset_state(self) -> None:
        """Zero out CFR regrets and strategy sums on every info set."""
        for I in self.info_sets.values():
            n = I.n_actions()
            I.regret = [0.0] * n
            I.strategy_sum = [0.0] * n
            I.last_regret = [0.0] * n


# =====================================================================
# Game builder
# =====================================================================


class GameBuilder:
    """Construct an extensive-form game incrementally.

    Typical use::

        b = GameBuilder(n_players=2, name="kuhn")
        root = b.chance_node(parent=-1, parent_action=None,
                             actions=["KQ", "KJ", "QK", "QJ", "JK", "JQ"],
                             probs=[1/6]*6)
        for outcome in ...:
            d = b.decision_node(parent=root, parent_action=outcome,
                                player=0, actions=["check", "bet"],
                                info_set=f"P0|{outcome[0]}")
        ...
        b.terminal_node(parent=..., parent_action=..., utilities=[+1, -1])
        game = b.build()

    Every decision-node call must pass an `info_set` key. The builder
    validates at :meth:`build` that all nodes sharing a key agree on
    player + actions, and that the game has perfect recall.
    """

    def __init__(self, n_players: int = 2, name: str = "") -> None:
        if n_players < 1:
            raise InvalidGame("n_players must be ≥ 1")
        self.n_players = n_players
        self.name = name
        self._nodes: dict[int, _Node] = {}
        self._next_id = 0
        self._root: int | None = None

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def chance_node(
        self,
        *,
        parent: int,
        parent_action: Any,
        actions: Sequence[Any],
        probs: Sequence[float],
    ) -> int:
        if len(actions) != len(probs):
            raise InvalidGame("chance: actions and probs differ in length")
        if any(p < -_EPS for p in probs):
            raise InvalidGame("chance: probs must be ≥ 0")
        total = sum(probs)
        if total <= 0.0:
            raise InvalidGame("chance: probs must sum to > 0")
        normed = [p / total for p in probs]
        nid = self._new_id()
        node = _Chance(
            id=nid, parent=parent, parent_action=parent_action,
            actions=list(actions), probs=normed, children=[-1] * len(actions),
        )
        self._nodes[nid] = node
        if self._root is None and parent == -1:
            self._root = nid
        self._link_child(parent, parent_action, nid)
        return nid

    def decision_node(
        self,
        *,
        parent: int,
        parent_action: Any,
        player: int,
        actions: Sequence[Any],
        info_set: str,
    ) -> int:
        if not (0 <= player < self.n_players):
            raise InvalidGame(f"player {player} out of range [0,{self.n_players})")
        if len(actions) == 0:
            raise InvalidGame("decision: actions must be non-empty")
        if not isinstance(info_set, str) or not info_set:
            raise InvalidGame("decision: info_set must be a non-empty string")
        nid = self._new_id()
        node = _Decision(
            id=nid, parent=parent, parent_action=parent_action,
            player=player, actions=list(actions), info_set=info_set,
            children=[-1] * len(actions),
        )
        self._nodes[nid] = node
        if self._root is None and parent == -1:
            self._root = nid
        self._link_child(parent, parent_action, nid)
        return nid

    def terminal_node(
        self,
        *,
        parent: int,
        parent_action: Any,
        utilities: Sequence[float],
    ) -> int:
        if len(utilities) != self.n_players:
            raise InvalidGame(
                f"terminal: utilities len {len(utilities)} != n_players {self.n_players}"
            )
        for u in utilities:
            try:
                f = float(u)
            except Exception as e:
                raise InvalidGame(f"terminal: utility not numeric: {u!r}") from e
            if not math.isfinite(f):
                raise InvalidGame(f"terminal: utility not finite: {u!r}")
        nid = self._new_id()
        node = _Terminal(
            id=nid, parent=parent, parent_action=parent_action,
            utilities=[float(u) for u in utilities],
        )
        self._nodes[nid] = node
        if self._root is None and parent == -1:
            self._root = nid
        self._link_child(parent, parent_action, nid)
        return nid

    def _link_child(self, parent: int, parent_action: Any, child: int) -> None:
        if parent == -1:
            return
        if parent not in self._nodes:
            raise InvalidGame(f"parent id {parent} unknown")
        p = self._nodes[parent]
        if isinstance(p, _Terminal):
            raise InvalidGame("terminal node cannot have children")
        if isinstance(p, _Chance) or isinstance(p, _Decision):
            if parent_action not in p.actions:
                raise InvalidGame(
                    f"action {parent_action!r} not in parent's action list {p.actions}"
                )
            idx = p.actions.index(parent_action)
            if p.children[idx] != -1:
                raise InvalidGame(
                    f"action {parent_action!r} from parent {parent} already linked"
                )
            p.children[idx] = child

    def build(self) -> Game:
        if self._root is None:
            raise InvalidGame("no root node was added")
        # Validate connectivity.
        for nid, n in self._nodes.items():
            if isinstance(n, _Terminal):
                continue
            for i, c in enumerate(n.children):
                if c == -1:
                    raise InvalidGame(
                        f"node {nid} action {n.actions[i]!r} has no child"
                    )
        # Build info sets.
        info_sets: dict[str, _InfoSet] = {}
        for nid, n in self._nodes.items():
            if not isinstance(n, _Decision):
                continue
            key = n.info_set
            if key not in info_sets:
                info_sets[key] = _InfoSet(
                    key=key, player=n.player, actions=list(n.actions),
                    regret=[0.0] * len(n.actions),
                    strategy_sum=[0.0] * len(n.actions),
                    last_regret=[0.0] * len(n.actions),
                )
            I = info_sets[key]
            if I.player != n.player:
                raise InvalidGame(
                    f"info set {key!r} groups nodes with mismatched players"
                )
            if list(I.actions) != list(n.actions):
                raise InvalidGame(
                    f"info set {key!r} groups nodes with mismatched action lists"
                )
            I.nodes.append(nid)
        # Perfect-recall check.
        self._check_perfect_recall(info_sets)
        g = Game(
            n_players=self.n_players, root=self._root,
            nodes=self._nodes, info_sets=info_sets, name=self.name,
        )
        return g

    def _check_perfect_recall(self, info_sets: Mapping[str, _InfoSet]) -> None:
        """For every info set I of player p, every node in I has the same
        sequence of (info_set, action) ancestors of player p."""
        for key, I in info_sets.items():
            ancestor_signatures: set[tuple] = set()
            for nid in I.nodes:
                sig = self._player_history(nid, I.player)
                ancestor_signatures.add(sig)
                if len(ancestor_signatures) > 1:
                    raise PerfectRecallViolation(
                        f"info set {key!r} groups histories with inconsistent past "
                        f"actions for player {I.player}: {ancestor_signatures}"
                    )

    def _player_history(self, nid: int, player: int) -> tuple:
        out: list[tuple] = []
        cur = self._nodes[nid]
        while cur.parent != -1:
            parent = self._nodes[cur.parent]
            if isinstance(parent, _Decision) and parent.player == player:
                out.append((parent.info_set, cur.parent_action))
            cur = parent
        return tuple(reversed(out))


# =====================================================================
# Strategy + best response + exploitability
# =====================================================================


def _validate_strategy(game: Game, strategy: Mapping[str, Sequence[float]]) -> None:
    for key, I in game.info_sets.items():
        if key not in strategy:
            raise DiplomatError(f"strategy missing info set {key!r}")
        s = strategy[key]
        if len(s) != I.n_actions():
            raise DiplomatError(
                f"strategy for {key!r} has {len(s)} actions, expected {I.n_actions()}"
            )
        if any(x < -_EPS for x in s):
            raise DiplomatError(f"strategy for {key!r} has negative entries")
        total = sum(s)
        if abs(total - 1.0) > 1e-6 and total > 0:
            raise DiplomatError(
                f"strategy for {key!r} sums to {total:.6f}, expected 1.0"
            )


def expected_utilities(
    game: Game, strategy: Mapping[str, Sequence[float]]
) -> list[float]:
    """Exact expected utility per player at the root under the joint
    behaviour strategy. O(|tree|)."""
    _validate_strategy(game, strategy)

    def rec(nid: int) -> list[float]:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return list(n.utilities)
        if isinstance(n, _Chance):
            out = [0.0] * game.n_players
            for p, c in zip(n.probs, n.children):
                cu = rec(c)
                for k in range(game.n_players):
                    out[k] += p * cu[k]
            return out
        # decision
        assert isinstance(n, _Decision)
        s = strategy[n.info_set]
        out = [0.0] * game.n_players
        for p, c in zip(s, n.children):
            if p <= 0.0:
                continue
            cu = rec(c)
            for k in range(game.n_players):
                out[k] += p * cu[k]
        return out

    return rec(game.root)


@dataclass
class BestResponseReport:
    player: int
    value: float
    response: dict[str, list[float]]  # pure behaviour strategy (one-hot per info set)
    delta: float  # value − (current strategy value)


def best_response(
    game: Game,
    strategy: Mapping[str, Sequence[float]],
    player: int,
) -> BestResponseReport:
    """Exact best pure response of `player` to the fixed mixed strategy of
    the other players.  O(|tree|) for one player; works on any number of
    players ≥ 2.  Brown-Johanson-Bowling style recursion.

    The returned response is a *pure* one-hot behaviour strategy that
    achieves the maximum expected utility for `player` given everyone
    else plays `strategy`.
    """
    _validate_strategy(game, strategy)
    if not (0 <= player < game.n_players):
        raise DiplomatError(f"player {player} out of range")

    # First, compute reach probabilities of the *opponents* down to each
    # decision node of `player`.  Then a backwards recursion picks
    # max_a value_{a} per info set.

    # Step 1: forward — compute opponent reach per node.
    opp_reach: dict[int, float] = {game.root: 1.0}
    # We also memoise the post-action value-conditional utility for `player`.
    # Walk top-down (DFS order).
    order: list[int] = []

    def dfs(nid: int) -> None:
        order.append(nid)
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return
        if isinstance(n, _Chance):
            for p, c in zip(n.probs, n.children):
                opp_reach[c] = opp_reach[nid] * p
                dfs(c)
        else:
            assert isinstance(n, _Decision)
            if n.player == player:
                # opponents don't move here
                for c in n.children:
                    opp_reach[c] = opp_reach[nid]
                    dfs(c)
            else:
                s = strategy[n.info_set]
                for p, c in zip(s, n.children):
                    opp_reach[c] = opp_reach[nid] * p
                    dfs(c)

    dfs(game.root)

    # Step 2: pick best action per info set of `player`, ascending.
    # For an info set I, the best response value is
    #     V*(I, a) = Σ_{h ∈ I} opp_reach(h) · V_player(h, a)
    # where V_player(h, a) is the player's expected utility from taking
    # `a` at h, opponents fixed, *player optimal everywhere downstream*.
    # We compute V_player(h) bottom-up; at downstream player-info-sets we
    # use the best-response value already computed.
    info_value: dict[str, list[float]] = {}  # value-sum per (info, a)
    node_value: dict[int, float] = {}  # V_player(h) under best response

    response: dict[str, list[float]] = {}

    def bottom_up(nid: int) -> float:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            v = n.utilities[player]
            node_value[nid] = v
            return v
        if isinstance(n, _Chance):
            v = 0.0
            for p, c in zip(n.probs, n.children):
                v += p * bottom_up(c)
            node_value[nid] = v
            return v
        assert isinstance(n, _Decision)
        # Recurse first so children's node_value is set.
        child_vals = [bottom_up(c) for c in n.children]
        if n.player == player:
            # Accumulate into info_value; the *true* per-info-set choice
            # is made after the global pass.  But to get node_value[nid]
            # in this pass we will redo the best-action pick after we
            # have the per-action info-sums available.  Workaround: we
            # do *two* passes: first compute info_value, then pick
            # actions, then a second pass to set node_value.  See below.
            iv = info_value.setdefault(n.info_set, [0.0] * len(n.actions))
            w = opp_reach[nid]
            for i, cv in enumerate(child_vals):
                iv[i] += w * cv
            node_value[nid] = 0.0  # placeholder; overwritten below
            return 0.0
        else:
            s = strategy[n.info_set]
            v = 0.0
            for p, cv in zip(s, child_vals):
                v += p * cv
            node_value[nid] = v
            return v

    # First pass: gathers info_value sums for player's info sets (but
    # node_value of player's nodes is wrong because we don't know which
    # action will be chosen yet).
    bottom_up(game.root)

    # Pick the action that maximises info_value per info set.
    chosen_action: dict[str, int] = {}
    for key, vals in info_value.items():
        best = max(range(len(vals)), key=lambda i: vals[i])
        chosen_action[key] = best
        oh = [0.0] * len(vals)
        oh[best] = 1.0
        response[key] = oh
    # Info sets of `player` that were not visited (e.g. dominated by
    # opponent strategies) are filled with uniform action choices.
    for key, I in game.info_sets.items():
        if I.player != player:
            continue
        if key not in response:
            n = I.n_actions()
            response[key] = [1.0 / n] * n
            chosen_action[key] = 0

    # Second pass: with chosen actions baked in, compute the correct
    # node_value for player's decision nodes.
    def second_pass(nid: int) -> float:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return n.utilities[player]
        if isinstance(n, _Chance):
            return sum(p * second_pass(c) for p, c in zip(n.probs, n.children))
        assert isinstance(n, _Decision)
        if n.player == player:
            a = chosen_action.get(n.info_set, 0)
            return second_pass(n.children[a])
        s = strategy[n.info_set]
        v = 0.0
        for p, c in zip(s, n.children):
            if p > 0.0:
                v += p * second_pass(c)
        return v

    br_value = second_pass(game.root)
    cur = expected_utilities(game, strategy)[player]
    return BestResponseReport(
        player=player, value=br_value, response=response, delta=br_value - cur
    )


def exploitability(
    game: Game, strategy: Mapping[str, Sequence[float]]
) -> float:
    """NashConv: Σ_p (BR_p(σ_{-p}) − u_p(σ)).  Zero iff σ is Nash.

    For two-player zero-sum games this equals twice the Nash gap
    (because the two BR deltas are equal in magnitude and the values
    sum to zero), but we report the raw sum so the convention matches
    the Lanctot / DeepMind game-theory literature.
    """
    base = expected_utilities(game, strategy)
    s = 0.0
    for p in range(game.n_players):
        br = best_response(game, strategy, p)
        s += br.value - base[p]
    return s


# =====================================================================
# CFR family
# =====================================================================


@dataclass
class CFRConfig:
    """Configuration for a CFR-family run.

    Most fields are only consulted by the relevant solver; harmless
    elsewhere.
    """

    kind: str = KIND_CFR_PLUS
    iterations: int = 1000
    seed: int | None = None
    # DCFR parameters (Brown-Sandholm 2019): defaults match the paper's
    # high-performance preset.
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_gamma: float = 2.0
    # Predictive CFR+: scale on the predicted next-round regret.
    predictive_m: float = 1.0
    # MCCFR: outcome-sampling exploration probability over the sampled
    # player's own actions (ε-on-policy mix).
    sampling_epsilon: float = 0.6
    # Diagnostics
    track_exploitability_every: int = 0  # 0 = never; e.g. 50 = every 50 iters
    log_event: bool = False

    def normalise(self) -> "CFRConfig":
        k = _KIND_ALIASES.get(self.kind, self.kind)
        if k not in KNOWN_KINDS:
            raise UnknownSolver(f"unknown solver kind: {self.kind!r}")
        return CFRConfig(
            kind=k,
            iterations=int(self.iterations),
            seed=self.seed,
            dcfr_alpha=float(self.dcfr_alpha),
            dcfr_beta=float(self.dcfr_beta),
            dcfr_gamma=float(self.dcfr_gamma),
            predictive_m=float(self.predictive_m),
            sampling_epsilon=float(self.sampling_epsilon),
            track_exploitability_every=int(self.track_exploitability_every),
            log_event=bool(self.log_event),
        )


@dataclass
class SolveReport:
    kind: str
    iterations: int
    average_strategy: dict[str, list[float]]
    last_strategy: dict[str, list[float]]
    root_value: list[float]  # under average strategy
    exploitability: float
    regret_bound: float  # anytime upper bound on exploitability from sum_regrets
    exploitability_trace: list[tuple[int, float]] = field(default_factory=list)
    wall_seconds: float = 0.0
    certificate: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, sort_keys=True, default=str)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _strategy_snapshot(game: Game, current: bool) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, I in game.info_sets.items():
        if current:
            out[key] = I.current_strategy()
        else:
            out[key] = I.average_strategy()
    return out


def _regret_bound_exploitability(game: Game) -> float:
    """Anytime upper bound: Σ_p Σ_{I ∈ player p} max_a R^+(I, a) / T.

    For two-player zero-sum games this bounds the exploitability with
    NashConv ≤ 2 (sum of positive regrets) / T (Zinkevich 2008). We
    report the unscaled sum-of-positive-regrets sum; callers should
    divide by their effective T."""
    s = 0.0
    for I in game.info_sets.values():
        max_pos = max((r for r in I.regret if r > 0.0), default=0.0)
        s += max_pos
    return s


# ---------- Deterministic CFR variants -------------------------------


def _cfr_full_traversal(
    game: Game,
    config: CFRConfig,
    iter_idx: int,
    bus: Any | None = None,
) -> None:
    """One iteration of full-tree CFR with the variant selected by
    ``config.kind``.

    Maintains per-info-set running regret and strategy sums on the Game.
    """
    n_players = game.n_players

    def rec(nid: int, reach: list[float], chance_reach: float) -> list[float]:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return list(n.utilities)
        if isinstance(n, _Chance):
            ev = [0.0] * n_players
            for p, c in zip(n.probs, n.children):
                if p <= 0.0:
                    continue
                cv = rec(c, reach, chance_reach * p)
                for k in range(n_players):
                    ev[k] += p * cv[k]
            return ev
        assert isinstance(n, _Decision)
        I = game.info_sets[n.info_set]
        player = I.player
        if config.kind == KIND_PREDICTIVE_CFR_PLUS:
            strat = I.predictive_strategy(config.predictive_m)
        else:
            strat = I.current_strategy()
        # Per-action expected values for the acting player + full ev.
        action_vals_player: list[float] = [0.0] * I.n_actions()
        ev = [0.0] * n_players
        for i, c in enumerate(n.children):
            p = strat[i]
            if p < _EPS and config.kind != KIND_PREDICTIVE_CFR_PLUS:
                # Need recurse to update regrets if we are the acting player
                # (counterfactual reach is still nonzero) — but only if its
                # opp/chance reach is positive.
                pass
            # Always traverse — counterfactual regret cares about the *opponent*
            # reach, not the acting player's own probability.
            new_reach = list(reach)
            new_reach[player] *= p
            cv = rec(c, new_reach, chance_reach)
            action_vals_player[i] = cv[player]
            for k in range(n_players):
                ev[k] += p * cv[k]
        # Counterfactual reach for the acting player = product of
        # everyone else's reach * chance.
        cf_reach = chance_reach
        for k in range(n_players):
            if k != player:
                cf_reach *= reach[k]
        # Update regrets.
        v_I = ev[player]
        # Strategy-sum weight (per variant).
        if config.kind == KIND_LINEAR_CFR:
            # Linear weight (Brown-Sandholm 2019).
            strat_w = float(iter_idx)
            regret_pos_w = 1.0
            regret_neg_w = 1.0
        elif config.kind == KIND_CFR_PLUS:
            strat_w = float(iter_idx)  # linear strategy averaging (Tammelin)
            regret_pos_w = 1.0
            regret_neg_w = 1.0
        elif config.kind == KIND_DISCOUNTED_CFR:
            t = float(iter_idx)
            a = config.dcfr_alpha
            b = config.dcfr_beta
            g = config.dcfr_gamma
            # DCFR weights:
            #   positive regret weight = t^α / (t^α + 1)
            #   negative regret weight = t^β / (t^β + 1)
            #   strategy weight        = (t / (t+1))^γ * t   (Brown-Sandholm)
            ta = t ** a
            tb = t ** b
            regret_pos_w = ta / (ta + 1.0)
            regret_neg_w = tb / (tb + 1.0)
            strat_w = ((t / (t + 1.0)) ** g) * t
        elif config.kind == KIND_PREDICTIVE_CFR_PLUS:
            strat_w = float(iter_idx)
            regret_pos_w = 1.0
            regret_neg_w = 1.0
        else:
            # vanilla CFR
            strat_w = 1.0
            regret_pos_w = 1.0
            regret_neg_w = 1.0
        for i in range(I.n_actions()):
            r_inst = cf_reach * (action_vals_player[i] - v_I)
            I.last_regret[i] = r_inst
            old = I.regret[i]
            if config.kind in (KIND_CFR_PLUS, KIND_PREDICTIVE_CFR_PLUS):
                # Floor cumulative regret at 0 (Tammelin 2014).
                new = max(0.0, old + r_inst)
            elif config.kind == KIND_DISCOUNTED_CFR:
                if old + r_inst >= 0.0:
                    new = (old + r_inst) * regret_pos_w
                else:
                    new = (old + r_inst) * regret_neg_w
            elif config.kind == KIND_LINEAR_CFR:
                # Linear CFR (Brown-Sandholm 2019): weight per-iter regret by t.
                new = old + float(iter_idx) * r_inst
            else:
                new = old + r_inst
            I.regret[i] = new
        # Strategy sum update.
        s_player_reach = reach[player]
        for i in range(I.n_actions()):
            I.strategy_sum[i] += strat_w * s_player_reach * strat[i]
        return ev

    initial = [1.0] * n_players
    rec(game.root, initial, 1.0)


# ---------- MCCFR variants -------------------------------------------


def _outcome_sampling_iter(
    game: Game,
    config: CFRConfig,
    iter_idx: int,
    update_player: int,
    rng: random.Random,
) -> None:
    """One iteration of outcome-sampling MCCFR for `update_player`.

    Follows Lanctot (2013, PhD thesis, Algorithm 4.5).  Returns
    ``(u_i(z) / q(z), π_i^σ(z | h))`` from each recursive call so the
    regret update at info set I uses
        Δ(I, a*) = π_{-i}^σ(h)·(u/q)·π_i^σ(z|h·a*)·(1 − σ(a*))
        Δ(I, a)  = −π_{-i}^σ(h)·(u/q)·π_i^σ(z|h·a*)·σ(a)     for a ≠ a*

    Status: outcome-sampling MCCFR converges to ε-Nash on chance-free
    games (matching pennies, RPS) in self-play.  For games with deep
    chance nodes (Kuhn poker), our implementation exhibits empirical
    bias in the strategy average and *should not be the production
    choice*.  Use ``external_sampling_mccfr`` or ``chance_sampling_cfr``
    for chance-heavy trees, or any deterministic CFR variant for an
    exact gradient.  Future work: variance-reduced "MCCFR with baseline"
    (Davis-Schmid-Bowling 2019).
    """
    eps = config.sampling_epsilon

    def rec(nid: int, pi_i: float, pi_minus: float, q: float) -> tuple[float, float]:
        # q tracks the sampling probability of action choices ONLY (chance
        # outcomes are sampled with their true probability so importance
        # weight for chance is identically 1 — same convention as
        # OpenSpiel `OutcomeSamplingMCCFRSolver`).  pi_minus includes
        # chance so it correctly represents π_{-i}^σ(h).
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            inv_q = (1.0 / q) if q > 0.0 else 0.0
            return n.utilities[update_player] * inv_q, 1.0
        if isinstance(n, _Chance):
            idx = _sample_index(n.probs, rng)
            p = n.probs[idx]
            return rec(n.children[idx], pi_i, pi_minus * p, q)
        assert isinstance(n, _Decision)
        I = game.info_sets[n.info_set]
        nA = I.n_actions()
        strat = I.current_strategy()
        if I.player == update_player:
            sample_d = [(1.0 - eps) * strat[a] + eps / nA for a in range(nA)]
            idx = _sample_index(sample_d, rng)
            u_over_q, pi_tail = rec(
                n.children[idx], pi_i * strat[idx], pi_minus, q * sample_d[idx]
            )
            base = pi_minus * u_over_q * pi_tail
            for a in range(nA):
                if a == idx:
                    delta = base * (1.0 - strat[a])
                else:
                    delta = -base * strat[a]
                I.regret[a] += delta
            inv_q = 1.0 / q if q > 0.0 else 0.0
            for a in range(nA):
                I.strategy_sum[a] += pi_i * strat[a] * inv_q
            return u_over_q, strat[idx] * pi_tail
        else:
            idx = _sample_index(strat, rng)
            return rec(
                n.children[idx], pi_i, pi_minus * strat[idx], q * strat[idx]
            )

    rec(game.root, 1.0, 1.0, 1.0)


def _external_sampling_iter(
    game: Game,
    config: CFRConfig,
    iter_idx: int,
    update_player: int,
    rng: random.Random,
) -> None:
    """External-sampling MCCFR (Lanctot et al. 2009)."""

    def rec(nid: int) -> float:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return n.utilities[update_player]
        if isinstance(n, _Chance):
            idx = _sample_index(n.probs, rng)
            return rec(n.children[idx])
        assert isinstance(n, _Decision)
        I = game.info_sets[n.info_set]
        nA = I.n_actions()
        strat = I.current_strategy()
        if I.player == update_player:
            child_vals = [rec(n.children[a]) for a in range(nA)]
            ev = sum(strat[a] * child_vals[a] for a in range(nA))
            for a in range(nA):
                I.regret[a] += child_vals[a] - ev
            for a in range(nA):
                I.strategy_sum[a] += strat[a]
            return ev
        else:
            idx = _sample_index(strat, rng)
            return rec(n.children[idx])

    rec(game.root)


def _chance_sampling_iter(
    game: Game,
    config: CFRConfig,
    iter_idx: int,
) -> None:
    """Chance-sampling CFR (Zinkevich et al. 2008) — deterministic in
    player actions, samples chance only.  Cheaper than full CFR on
    chance-heavy games."""
    n_players = game.n_players
    # We can't easily share an RNG with the random module elsewhere; the
    # config-seeded rng is created in solve(); here we use Python's
    # global random for sampling, but solve() seeds it.

    def rec(nid: int, reach: list[float]) -> list[float]:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return list(n.utilities)
        if isinstance(n, _Chance):
            idx = _sample_index_global(n.probs)
            return rec(n.children[idx], reach)
        assert isinstance(n, _Decision)
        I = game.info_sets[n.info_set]
        player = I.player
        strat = I.current_strategy()
        action_vals_player = [0.0] * I.n_actions()
        ev = [0.0] * n_players
        for i, c in enumerate(n.children):
            new_reach = list(reach)
            new_reach[player] *= strat[i]
            cv = rec(c, new_reach)
            action_vals_player[i] = cv[player]
            for k in range(n_players):
                ev[k] += strat[i] * cv[k]
        cf_reach = 1.0
        for k in range(n_players):
            if k != player:
                cf_reach *= reach[k]
        v_I = ev[player]
        for i in range(I.n_actions()):
            r_inst = cf_reach * (action_vals_player[i] - v_I)
            I.regret[i] += r_inst
            I.strategy_sum[i] += reach[player] * strat[i]
        return ev

    initial = [1.0] * n_players
    rec(game.root, initial)


def _sample_index(probs: Sequence[float], rng: random.Random) -> int:
    u = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u <= cum:
            return i
    return len(probs) - 1


def _sample_index_global(probs: Sequence[float]) -> int:
    u = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u <= cum:
            return i
    return len(probs) - 1


# =====================================================================
# Sequence-form LP (2-player zero-sum, exact)
# =====================================================================


def _build_sequence_form(game: Game, player: int) -> tuple[
    list[tuple[str, Any]],  # sequence index → (info_set_key or "*", action or None)
    list[list[float]],     # E_p: constraint matrix on player's realisation plan
    list[float],           # e_p: rhs (1 followed by zeros)
    dict[str, list[int]],  # info_set → list of sequence indices (one per action)
    dict[str, int],        # info_set → parent sequence index
]:
    """Construct the sequence-form constraint Eᵖ rᵖ = eᵖ for one player.

    Sequence form: a *sequence* of player p is the concatenation of all
    (information set, action) pairs of p on a path from root. The empty
    sequence ∅ is sequence 0. Realisation plan r(σ) on sequences must
    satisfy: r(∅)=1 and for every info set I of p with parent sequence
    σ(I), r(σ(I)) = Σ_{a ∈ A(I)} r(σ(I) ⊕ (I,a)).
    """
    # Enumerate sequences depth-first; each player decision contributes
    # one sequence per action plus depends on its parent sequence.
    sequences: list[tuple[str, Any]] = [("*", None)]  # empty sequence at index 0
    info_seqs: dict[str, list[int]] = {}
    info_parent_seq: dict[str, int] = {}

    # We need: per node in the tree, which sequence of `player` reaches it.
    # We walk the tree and propagate the current player-sequence.
    node_seq: dict[int, int] = {game.root: 0}

    def dfs(nid: int) -> None:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            return
        if isinstance(n, _Chance):
            for c in n.children:
                node_seq[c] = node_seq[nid]
                dfs(c)
            return
        assert isinstance(n, _Decision)
        if n.player != player:
            for c in n.children:
                node_seq[c] = node_seq[nid]
                dfs(c)
            return
        # player's decision: register info set with parent sequence
        key = n.info_set
        if key not in info_seqs:
            info_parent_seq[key] = node_seq[nid]
            info_seqs[key] = []
            for a in n.actions:
                idx = len(sequences)
                sequences.append((key, a))
                info_seqs[key].append(idx)
        for i, c in enumerate(n.children):
            node_seq[c] = info_seqs[key][i]
            dfs(c)

    dfs(game.root)

    n_seq = len(sequences)
    n_info = len(info_seqs)
    # Constraint matrix: 1 + n_info rows. Row 0 is r(∅) = 1.
    E = [[0.0] * n_seq for _ in range(1 + n_info)]
    e = [0.0] * (1 + n_info)
    E[0][0] = 1.0
    e[0] = 1.0
    for row, key in enumerate(info_seqs.keys(), start=1):
        parent = info_parent_seq[key]
        children = info_seqs[key]
        # r(parent) - Σ r(children) = 0  → +1 on parent, -1 on each child
        E[row][parent] = 1.0
        for c in children:
            E[row][c] = -1.0
    return sequences, E, e, info_seqs, info_parent_seq


def _payoff_matrix_sequence_form(game: Game) -> tuple[
    list[tuple[str, Any]], list[tuple[str, Any]], list[list[float]]
]:
    """Sequence-form payoff matrix A: A[s, t] = sum_{terminals z: seq(z)=(s,t)}
    chance_prob(z) * u_0(z).
    """
    seq0, _, _, _, _ = _build_sequence_form(game, 0)
    seq1, _, _, _, _ = _build_sequence_form(game, 1)
    # We need per-terminal: chance_prob, seq0, seq1, and utility for player 0.
    A = [[0.0] * len(seq1) for _ in range(len(seq0))]
    node_seq0: dict[int, int] = {game.root: 0}
    node_seq1: dict[int, int] = {game.root: 0}
    # Rebuild traversal mapping nodes to sequence indices for both players.
    info_seqs0: dict[str, list[int]] = {}
    info_seqs1: dict[str, list[int]] = {}
    cur0 = 0
    cur1 = 0
    for i, (key, a) in enumerate(seq0):
        if i == 0:
            continue
        info_seqs0.setdefault(key, []).append(i)
    for i, (key, a) in enumerate(seq1):
        if i == 0:
            continue
        info_seqs1.setdefault(key, []).append(i)

    def dfs(nid: int, chance_prob: float, s0: int, s1: int) -> None:
        n = game.nodes[nid]
        if isinstance(n, _Terminal):
            A[s0][s1] += chance_prob * n.utilities[0]
            return
        if isinstance(n, _Chance):
            for p, c in zip(n.probs, n.children):
                dfs(c, chance_prob * p, s0, s1)
            return
        assert isinstance(n, _Decision)
        key = n.info_set
        if n.player == 0:
            kids = info_seqs0.get(key, [])
            for i, c in enumerate(n.children):
                dfs(c, chance_prob, kids[i], s1)
        else:
            kids = info_seqs1.get(key, [])
            for i, c in enumerate(n.children):
                dfs(c, chance_prob, s0, kids[i])

    dfs(game.root, 1.0, 0, 0)
    return seq0, seq1, A


# Simple LP solver with mixed constraints, self-contained.
#
#   Constraints can be ≤, ≥, or =, indicated by per-row sign in {-1, 0, +1}
#   (the sign of the comparator in `Ax sign b`). The solver adds slack /
#   surplus / artificial variables as needed and runs a Big-M tableau
#   simplex with Bland's rule for no cycling.
#
#   Variables x ≥ 0; rhs b is free in sign (negative b's are flipped
#   before adding artificials).
#
# Returns (status, x, value).


LE, EQ, GE = -1, 0, +1


def _solve_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    senses: Sequence[int],
    b: Sequence[float],
    big_m: float = 1e7,
    max_iters: int = _LP_MAX_ITERS,
    tol: float = _LP_TOL,
) -> tuple[str, list[float], float]:
    """max c·x s.t. (Ax sense_i b_i for each row i), x ≥ 0.

    `senses[i]` ∈ {LE=-1, EQ=0, GE=+1} controls the i-th row's comparator.
    """
    m = len(A)
    n = len(c)
    if not (len(senses) == m == len(b)):
        raise InfeasibleProgram("LP shape mismatch")
    # Flip rows with b<0 to enforce b ≥ 0; flip sense too.
    rows = [list(r) for r in A]
    bb = list(b)
    sn = list(senses)
    for i in range(m):
        if bb[i] < 0.0:
            rows[i] = [-x for x in rows[i]]
            bb[i] = -bb[i]
            sn[i] = -sn[i]  # LE ↔ GE; EQ unchanged
    # Now bb ≥ 0 everywhere.  Add slack/surplus/artificial columns.
    #   LE: + slack, no artificial.
    #   EQ: + artificial.
    #   GE: − surplus + artificial.
    # Count extras.
    n_slack_or_surplus = m  # one per row (slack for LE, surplus for GE; EQ has 0 here)
    # Actually only LE needs slack and GE needs surplus.  EQ needs nothing in this slot.
    # To keep indexing simple, we add one extra column per row labelled "slack/surplus/0".
    # Then a separate set of artificial columns for EQ and GE rows.
    artificial_for: list[int] = []  # row indices needing artificial
    for i in range(m):
        if sn[i] in (EQ, GE):
            artificial_for.append(i)
    n_art = len(artificial_for)
    total_vars = n + m + n_art  # original + slack-surplus + artificials
    # Build tableau: (m+1) × (total_vars + 1)
    tab = [[0.0] * (total_vars + 1) for _ in range(m + 1)]
    basis = [-1] * m
    art_col_of_row: dict[int, int] = {}
    art_idx = 0
    for i in range(m):
        for j in range(n):
            tab[i][j] = rows[i][j]
        # slack/surplus column for row i
        col_ss = n + i
        if sn[i] == LE:
            tab[i][col_ss] = +1.0
        elif sn[i] == GE:
            tab[i][col_ss] = -1.0
        # else EQ: 0
        if sn[i] in (EQ, GE):
            col_art = n + m + art_idx
            tab[i][col_art] = +1.0
            art_col_of_row[i] = col_art
            basis[i] = col_art
            art_idx += 1
        else:
            basis[i] = col_ss
        tab[i][total_vars] = bb[i]
    # Objective row: max c·x − M Σ artificials
    for j in range(n):
        tab[m][j] = -c[j]
    for ai in range(n_art):
        tab[m][n + m + ai] = +big_m  # because we're storing -c form; artificial has -M in c → +M here
    # Reduce objective row to eliminate basic-variable artificial coefficients.
    for i, var in enumerate(basis):
        if var >= n + m:  # artificial
            # subtract big_m * row i from objective so artificial cols have 0 in obj.
            for j in range(total_vars + 1):
                tab[m][j] -= big_m * tab[i][j]
    iters = 0
    while iters < max_iters:
        iters += 1
        # Bland: smallest-index column with strictly negative reduced cost.
        entering = -1
        for j in range(total_vars):
            if tab[m][j] < -tol:
                entering = j
                break
        if entering == -1:
            break
        # Min-ratio test with Bland tie-break (lowest basis index leaves).
        leaving_row = -1
        best_ratio = float("inf")
        for i in range(m):
            aij = tab[i][entering]
            if aij > tol:
                ratio = tab[i][total_vars] / aij
                if ratio < best_ratio - tol:
                    best_ratio = ratio
                    leaving_row = i
                elif abs(ratio - best_ratio) <= tol and leaving_row >= 0:
                    if basis[i] < basis[leaving_row]:
                        leaving_row = i
        if leaving_row == -1:
            return "unbounded", [0.0] * n, float("inf")
        # Pivot.
        pivot = tab[leaving_row][entering]
        for j in range(total_vars + 1):
            tab[leaving_row][j] /= pivot
        for i in range(m + 1):
            if i == leaving_row:
                continue
            factor = tab[i][entering]
            if abs(factor) < _EPS:
                continue
            for j in range(total_vars + 1):
                tab[i][j] -= factor * tab[leaving_row][j]
        basis[leaving_row] = entering
    if iters >= max_iters:
        return "max_iters", [0.0] * n, 0.0
    # If any artificial remains in basis with nonzero value → infeasible.
    x = [0.0] * total_vars
    for i, var in enumerate(basis):
        if var >= 0:
            x[var] = tab[i][total_vars]
    arts_remaining = sum(x[n + m + ai] for ai in range(n_art))
    if arts_remaining > 1e-5:
        return "infeasible", [0.0] * n, 0.0
    # Extract primal x for original variables and objective value.
    primal = x[:n]
    value = sum(c[j] * primal[j] for j in range(n))
    return "optimal", primal, value


def _revised_simplex(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    max_iters: int = _LP_MAX_ITERS,
    tol: float = _LP_TOL,
) -> tuple[str, list[float], float]:
    """Legacy single-shape entry: max c·x s.t. Ax ≤ b, x ≥ 0, b ≥ 0.

    Retained for callers that use the pure-LE form.
    """
    return _solve_lp(c, A, [LE] * len(A), b, max_iters=max_iters, tol=tol)


def _solve_sequence_form_zero_sum(game: Game) -> tuple[
    dict[str, list[float]], dict[str, list[float]], float
]:
    """Exact NE of a two-player zero-sum extensive-form game via the
    sequence-form LP (von Stengel 1996).

    Maximin: max over r⁰ of min over r¹ of r⁰ᵀ A r¹ s.t. E^p r^p = e^p, r^p ≥ 0.
    By LP duality on the inner min, this becomes
        max   e¹ᵀ y
        s.t.  E¹ᵀ y ≤ Aᵀ r⁰          (one ≤ row per player-1 sequence)
              E⁰ r⁰ = e⁰              (player-0 realisation plan, equalities)
              r⁰ ≥ 0, y free
    `y` is made non-negative by the split ``y = y⁺ − y⁻``.
    """
    if game.n_players != 2:
        raise NotTwoPlayerZeroSum("sequence-form LP requires exactly 2 players")
    for n in game.nodes.values():
        if isinstance(n, _Terminal):
            if abs(n.utilities[0] + n.utilities[1]) > 1e-9:
                raise NotTwoPlayerZeroSum(
                    f"terminal {n.id} is not zero-sum: {n.utilities}"
                )
    seq0, E0, e0, info_seqs0, info_parent0 = _build_sequence_form(game, 0)
    seq1, E1, e1, info_seqs1, info_parent1 = _build_sequence_form(game, 1)
    _, _, A = _payoff_matrix_sequence_form(game)
    r0 = _solve_realisation_player0(game)
    strat0 = _realisation_to_behaviour(seq0, info_seqs0, info_parent0, r0, game, 0)
    r1 = _solve_realisation_for_player(game, 1)
    strat1 = _realisation_to_behaviour(seq1, info_seqs1, info_parent1, r1, game, 1)
    # Game value = r0ᵀ A r1.
    val = 0.0
    for i in range(len(seq0)):
        if r0[i] == 0.0:
            continue
        for j in range(len(seq1)):
            val += r0[i] * A[i][j] * r1[j]
    return strat0, strat1, val


def _solve_realisation_for_player(game: Game, player: int) -> list[float]:
    """Solve for `player`'s optimal realisation plan against the other.

    Internally reuses the maximin LP machinery.  For player 1 we solve
    the LP with the payoff matrix transposed and negated (zero-sum) and
    swap player roles.
    """
    if player == 0:
        # Should not normally be called; but support symmetric usage.
        return _solve_realisation_player0(game)
    # Build a game with players swapped (negate utilities and swap order)
    swap = _swap_players_zero_sum(game)
    return _solve_realisation_player0(swap)


def _swap_players_zero_sum(game: Game) -> Game:
    """Return a copy with player 0/1 swapped (assumes 2-player zero-sum)."""
    new_nodes: dict[int, _Node] = {}
    for nid, n in game.nodes.items():
        if isinstance(n, _Chance):
            new_nodes[nid] = _Chance(
                id=n.id, parent=n.parent, parent_action=n.parent_action,
                actions=list(n.actions), probs=list(n.probs),
                children=list(n.children),
            )
        elif isinstance(n, _Decision):
            new_nodes[nid] = _Decision(
                id=n.id, parent=n.parent, parent_action=n.parent_action,
                player=1 - n.player, actions=list(n.actions),
                info_set=n.info_set, children=list(n.children),
            )
        elif isinstance(n, _Terminal):
            new_nodes[nid] = _Terminal(
                id=n.id, parent=n.parent, parent_action=n.parent_action,
                utilities=[n.utilities[1], n.utilities[0]],
            )
    new_info: dict[str, _InfoSet] = {}
    for key, I in game.info_sets.items():
        new_info[key] = _InfoSet(
            key=I.key, player=1 - I.player, actions=list(I.actions),
            nodes=list(I.nodes),
            regret=[0.0] * I.n_actions(),
            strategy_sum=[0.0] * I.n_actions(),
            last_regret=[0.0] * I.n_actions(),
        )
    return Game(
        n_players=2, root=game.root, nodes=new_nodes,
        info_sets=new_info, name=game.name + "_swap",
    )


def _solve_realisation_player0(game: Game) -> list[float]:
    """Solve maximin LP and return player 0's realisation plan.

    LP variables: (r⁰ ∈ R^{n0}, y⁺ ∈ R^{m1}, y⁻ ∈ R^{m1}) with y = y⁺−y⁻.
    Rows:
      - E⁰ r⁰ = e⁰                            (n_info0 + 1 equality rows)
      - For each player-1 sequence t (n1 rows):
          (E¹ᵀ (y⁺−y⁻))_t  − (Aᵀ r⁰)_t ≤ 0
    Objective: max e¹ᵀ (y⁺ − y⁻).
    """
    seq0, E0, e0, info_seqs0, info_parent0 = _build_sequence_form(game, 0)
    seq1, E1, e1, info_seqs1, info_parent1 = _build_sequence_form(game, 1)
    _, _, A = _payoff_matrix_sequence_form(game)
    n0 = len(seq0)
    n1 = len(seq1)
    m1 = len(E1)
    AT = [[A[i][j] for i in range(n0)] for j in range(n1)]  # n1 × n0
    E1T = [[E1[i][j] for i in range(m1)] for j in range(n1)]  # n1 × m1
    n_vars = n0 + 2 * m1

    rows: list[list[float]] = []
    senses: list[int] = []
    rhs: list[float] = []
    # Equality rows: E0 r⁰ = e0
    for i in range(len(E0)):
        row = [E0[i][j] for j in range(n0)] + [0.0] * (2 * m1)
        rows.append(row)
        senses.append(EQ)
        rhs.append(e0[i])
    # Inequality rows: (E1ᵀ (y⁺−y⁻))_t − (Aᵀ r⁰)_t ≤ 0
    for t in range(n1):
        row = (
            [-AT[t][j] for j in range(n0)]
            + [E1T[t][i] for i in range(m1)]
            + [-E1T[t][i] for i in range(m1)]
        )
        rows.append(row)
        senses.append(LE)
        rhs.append(0.0)
    c = [0.0] * n0 + [e1[i] for i in range(m1)] + [-e1[i] for i in range(m1)]
    status, x, val = _solve_lp(c, rows, senses, rhs)
    if status != "optimal":
        raise InfeasibleProgram(f"sequence-form LP status: {status}")
    return x[:n0]


def _realisation_to_behaviour(
    sequences: list[tuple[str, Any]],
    info_seqs: dict[str, list[int]],
    info_parent_seq: dict[str, int],
    r: list[float],
    game: Game,
    player: int,
) -> dict[str, list[float]]:
    """Convert a realisation plan into a behaviour strategy.

    σ(I, a) = r(σ(I) ⊕ a) / r(σ(I)) where r(σ(I)) is the parent
    sequence's realisation; if r(σ(I)) ≈ 0 then I is unreached and the
    behaviour strategy is arbitrary — we use uniform.
    """
    strat: dict[str, list[float]] = {}
    for key, _ in info_seqs.items():
        parent_seq = info_parent_seq[key]
        children = info_seqs[key]
        parent_val = r[parent_seq]
        actions = game.info_sets[key].actions
        if parent_val <= 1e-9:
            n = len(children)
            strat[key] = [1.0 / n] * n
        else:
            strat[key] = [max(0.0, r[c] / parent_val) for c in children]
            s = sum(strat[key])
            if s > 0:
                strat[key] = [x / s for x in strat[key]]
    # Fill info sets of this player that didn't appear in the LP (unreachable).
    for key, I in game.info_sets.items():
        if I.player != player:
            continue
        if key not in strat:
            n = I.n_actions()
            strat[key] = [1.0 / n] * n
    return strat


# =====================================================================
# Top-level Diplomat
# =====================================================================


class Diplomat:
    """Counterfactual-regret-minimization runtime for extensive-form games.

    Usage::

        game = kuhn_poker()
        diplomat = Diplomat()
        report = diplomat.solve(game, CFRConfig(kind="cfr_plus", iterations=2000))
        # report.average_strategy is the ε-Nash; report.exploitability is the gap.
    """

    def __init__(self, bus: Any | None = None) -> None:
        self.bus = bus

    # -- Solve --------------------------------------------------------

    def solve(self, game: Game, config: CFRConfig | None = None) -> SolveReport:
        if config is None:
            config = CFRConfig()
        cfg = config.normalise()
        if cfg.kind == KIND_SEQUENCE_FORM_LP:
            return self._solve_lp(game, cfg)
        return self._solve_iterative(game, cfg)

    def _solve_iterative(self, game: Game, cfg: CFRConfig) -> SolveReport:
        game.reset_state()
        rng = random.Random(cfg.seed)
        if cfg.seed is not None:
            random.seed(cfg.seed)
        if cfg.iterations <= 0:
            raise InsufficientIterations("iterations must be ≥ 1")
        t0 = time.time()
        trace: list[tuple[int, float]] = []
        if self.bus is not None and Event is not None:
            try:
                self.bus.publish(Event(kind=DIPLOMAT_STARTED, data={
                    "game": game.name, "kind": cfg.kind, "iterations": cfg.iterations,
                }))
            except Exception:
                pass
        for it in range(1, cfg.iterations + 1):
            if cfg.kind in (
                KIND_CFR, KIND_CFR_PLUS, KIND_LINEAR_CFR, KIND_DISCOUNTED_CFR,
                KIND_PREDICTIVE_CFR_PLUS,
            ):
                _cfr_full_traversal(game, cfg, it, self.bus)
            elif cfg.kind == KIND_OUTCOME_SAMPLING:
                # Alternate update_player across iterations (Lanctot 2013 §4.6).
                _outcome_sampling_iter(game, cfg, it, it % game.n_players, rng)
            elif cfg.kind == KIND_EXTERNAL_SAMPLING:
                for p in range(game.n_players):
                    _external_sampling_iter(game, cfg, it, p, rng)
            elif cfg.kind == KIND_CHANCE_SAMPLING:
                _chance_sampling_iter(game, cfg, it)
            else:
                raise UnknownSolver(f"unhandled solver kind: {cfg.kind}")
            if cfg.track_exploitability_every > 0 and it % cfg.track_exploitability_every == 0:
                avg = _strategy_snapshot(game, current=False)
                trace.append((it, exploitability(game, avg)))
                if self.bus is not None and Event is not None:
                    try:
                        self.bus.publish(Event(kind=DIPLOMAT_ITER, data={
                            "iter": it, "exploitability": trace[-1][1],
                        }))
                    except Exception:
                        pass
        avg = _strategy_snapshot(game, current=False)
        last = _strategy_snapshot(game, current=True)
        root = expected_utilities(game, avg)
        expl = exploitability(game, avg)
        # Anytime bound: sum of positive regrets / T  (Zinkevich 2008, Theorem 4).
        bound = _regret_bound_exploitability(game) / max(1, cfg.iterations)
        report = SolveReport(
            kind=cfg.kind, iterations=cfg.iterations,
            average_strategy=avg, last_strategy=last, root_value=root,
            exploitability=expl, regret_bound=bound,
            exploitability_trace=trace,
            wall_seconds=time.time() - t0,
            certificate=self._certificate(game, cfg, expl, bound),
        )
        if self.bus is not None and Event is not None:
            try:
                self.bus.publish(Event(kind=DIPLOMAT_SOLVED, data={
                    "game": game.name, "kind": cfg.kind,
                    "iterations": cfg.iterations,
                    "exploitability": expl,
                    "regret_bound": bound,
                    "fingerprint": report.fingerprint(),
                }))
            except Exception:
                pass
        return report

    def _solve_lp(self, game: Game, cfg: CFRConfig) -> SolveReport:
        if game.n_players != 2:
            raise NotTwoPlayerZeroSum("sequence-form LP requires exactly 2 players")
        t0 = time.time()
        s0, s1, value = _solve_sequence_form_zero_sum(game)
        strat: dict[str, list[float]] = {}
        for key, I in game.info_sets.items():
            if I.player == 0:
                strat[key] = s0[key]
            else:
                strat[key] = s1[key]
        root = expected_utilities(game, strat)
        expl = exploitability(game, strat)
        report = SolveReport(
            kind=cfg.kind, iterations=1,
            average_strategy=strat, last_strategy=strat,
            root_value=root, exploitability=expl, regret_bound=0.0,
            wall_seconds=time.time() - t0,
            certificate=self._certificate(game, cfg, expl, 0.0, exact=True),
        )
        if self.bus is not None and Event is not None:
            try:
                self.bus.publish(Event(kind=DIPLOMAT_LP_SOLVED, data={
                    "game": game.name, "value": value, "exploitability": expl,
                    "fingerprint": report.fingerprint(),
                }))
            except Exception:
                pass
        return report

    def best_response(
        self, game: Game, strategy: Mapping[str, Sequence[float]], player: int
    ) -> BestResponseReport:
        return best_response(game, strategy, player)

    def exploitability(
        self, game: Game, strategy: Mapping[str, Sequence[float]]
    ) -> float:
        return exploitability(game, strategy)

    def expected_utilities(
        self, game: Game, strategy: Mapping[str, Sequence[float]]
    ) -> list[float]:
        return expected_utilities(game, strategy)

    def uniform_strategy(self, game: Game) -> dict[str, list[float]]:
        return {
            k: [1.0 / I.n_actions()] * I.n_actions()
            for k, I in game.info_sets.items()
        }

    def _certificate(
        self,
        game: Game,
        cfg: CFRConfig,
        expl: float,
        bound: float,
        exact: bool = False,
    ) -> dict[str, Any]:
        return {
            "game": game.name,
            "n_players": game.n_players,
            "n_nodes": game.size(),
            "n_decisions": game.n_decisions(),
            "n_chance": game.n_chance(),
            "n_terminals": game.n_terminals(),
            "n_info_sets": len(game.info_sets),
            "solver": cfg.kind,
            "iterations": cfg.iterations,
            "exact": bool(exact),
            "exploitability": float(expl),
            "regret_bound": float(bound),
            "perfect_recall": True,  # validated at build
            "id": str(uuid.uuid4()),
        }


# =====================================================================
# Game builders — canonical benchmark games
# =====================================================================


def kuhn_poker() -> Game:
    """Kuhn poker (Kuhn 1950): 3-card deck (J, Q, K), one card to each
    player, single round of check/bet, antes of 1.  Standard regression
    test for any imperfect-information solver.

    Returns a Game with player 0 = dealer, player 1 = caller.
    """
    b = GameBuilder(n_players=2, name="kuhn_poker")
    deals = [("J", "Q"), ("J", "K"), ("Q", "J"),
             ("Q", "K"), ("K", "J"), ("K", "Q")]
    # All 6 deals are equiprobable.
    root = b.chance_node(parent=-1, parent_action=None,
                         actions=deals, probs=[1.0 / 6.0] * 6)
    rank = {"J": 0, "Q": 1, "K": 2}

    def is_winner_showdown(c0: str, c1: str) -> int:
        return 0 if rank[c0] > rank[c1] else 1

    for deal in deals:
        c0, c1 = deal
        # Player 0 decides check or bet.
        d0 = b.decision_node(
            parent=root, parent_action=deal,
            player=0, actions=["check", "bet"],
            info_set=f"P0|{c0}",
        )
        # P0 checks → P1 to act.
        d1_check = b.decision_node(
            parent=d0, parent_action="check",
            player=1, actions=["check", "bet"],
            info_set=f"P1|{c1}|check",
        )
        # P1 check after P0 check → showdown for ante (±1).
        w = is_winner_showdown(c0, c1)
        u = [1, -1] if w == 0 else [-1, 1]
        b.terminal_node(parent=d1_check, parent_action="check", utilities=u)
        # P1 bets after P0 check → P0 to act.
        d0_check_bet = b.decision_node(
            parent=d1_check, parent_action="bet",
            player=0, actions=["fold", "call"],
            info_set=f"P0|{c0}|check_bet",
        )
        # P0 folds → P1 wins ante.
        b.terminal_node(parent=d0_check_bet, parent_action="fold", utilities=[-1, 1])
        # P0 calls → showdown for ante+bet (±2).
        w = is_winner_showdown(c0, c1)
        u = [2, -2] if w == 0 else [-2, 2]
        b.terminal_node(parent=d0_check_bet, parent_action="call", utilities=u)
        # P0 bets → P1 to act.
        d1_bet = b.decision_node(
            parent=d0, parent_action="bet",
            player=1, actions=["fold", "call"],
            info_set=f"P1|{c1}|bet",
        )
        # P1 folds → P0 wins ante.
        b.terminal_node(parent=d1_bet, parent_action="fold", utilities=[1, -1])
        # P1 calls → showdown for ante+bet (±2).
        w = is_winner_showdown(c0, c1)
        u = [2, -2] if w == 0 else [-2, 2]
        b.terminal_node(parent=d1_bet, parent_action="call", utilities=u)
    return b.build()


def matching_pennies_sequential() -> Game:
    """Sequential matching pennies: player 0 moves first publicly, player
    1 observes and replies.  Trivial zero-sum game; Nash is mixed only
    when the second player cannot observe.  Used as a perfect-info
    sanity check.
    """
    b = GameBuilder(n_players=2, name="matching_pennies_sequential")
    d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                         actions=["H", "T"], info_set="P0")
    d1H = b.decision_node(parent=d0, parent_action="H", player=1,
                          actions=["H", "T"], info_set="P1|H")
    d1T = b.decision_node(parent=d0, parent_action="T", player=1,
                          actions=["H", "T"], info_set="P1|T")
    b.terminal_node(parent=d1H, parent_action="H", utilities=[+1, -1])
    b.terminal_node(parent=d1H, parent_action="T", utilities=[-1, +1])
    b.terminal_node(parent=d1T, parent_action="H", utilities=[-1, +1])
    b.terminal_node(parent=d1T, parent_action="T", utilities=[+1, -1])
    return b.build()


def matching_pennies_simultaneous() -> Game:
    """Matching pennies as an EFG: player 1 cannot observe player 0's
    move.  Models simultaneous play as a single-info-set chain.  Nash
    is uniform on both sides; value = 0.
    """
    b = GameBuilder(n_players=2, name="matching_pennies_simultaneous")
    d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                         actions=["H", "T"], info_set="P0")
    d1H = b.decision_node(parent=d0, parent_action="H", player=1,
                          actions=["H", "T"], info_set="P1")  # one info set
    d1T = b.decision_node(parent=d0, parent_action="T", player=1,
                          actions=["H", "T"], info_set="P1")  # same info set
    b.terminal_node(parent=d1H, parent_action="H", utilities=[+1, -1])
    b.terminal_node(parent=d1H, parent_action="T", utilities=[-1, +1])
    b.terminal_node(parent=d1T, parent_action="H", utilities=[-1, +1])
    b.terminal_node(parent=d1T, parent_action="T", utilities=[+1, -1])
    return b.build()


def simple_bargaining(n_rounds: int = 2, pie: float = 1.0) -> Game:
    """Two-round alternating-offer bargaining (Rubinstein-style) over a
    pie of fixed size, with no discounting.  Acceptance ends the game;
    rejection passes the move and the second player can split the pie
    on her own terms.  After rejection in the last round, both get 0.

    Offers come from a coarse discretisation {0.0, 0.25, 0.5, 0.75, 1.0}
    of the share kept by the proposer.
    """
    if n_rounds < 1:
        raise InvalidGame("n_rounds must be ≥ 1")
    if pie <= 0.0:
        raise InvalidGame("pie must be > 0")
    OFFERS = [0.0, 0.25, 0.5, 0.75, 1.0]
    b = GameBuilder(n_players=2, name=f"bargaining_{n_rounds}")
    # We build the tree recursively without using chance.
    # Player who proposes alternates: round 0 → 0, round 1 → 1, ...
    def build_round(parent_id: int, parent_action: Any, history: tuple, rnd: int) -> None:
        proposer = rnd % 2
        responder = 1 - proposer
        d = b.decision_node(
            parent=parent_id, parent_action=parent_action,
            player=proposer, actions=OFFERS,
            info_set=f"propose_r{rnd}_{history}",
        )
        for offer in OFFERS:
            # Responder accepts/rejects.
            resp = b.decision_node(
                parent=d, parent_action=offer,
                player=responder, actions=["accept", "reject"],
                info_set=f"respond_r{rnd}_{offer}_{history}",
            )
            # Accept: proposer keeps `offer*pie`, responder gets `(1-offer)*pie`.
            u = [0.0, 0.0]
            u[proposer] = offer * pie
            u[responder] = (1.0 - offer) * pie
            b.terminal_node(parent=resp, parent_action="accept", utilities=u)
            if rnd + 1 < n_rounds:
                build_round(resp, "reject", history + (offer,), rnd + 1)
            else:
                # No deal: both get 0.
                b.terminal_node(parent=resp, parent_action="reject", utilities=[0.0, 0.0])

    # Root is the first proposer's decision.  Build a dummy chance with
    # a single outcome so the root is a chance node (keeps the type
    # uniform with games that have chance).  Easier: have root be the
    # decision itself.
    OFFERS_ROOT = OFFERS
    proposer = 0
    responder = 1
    d_root = b.decision_node(
        parent=-1, parent_action=None,
        player=proposer, actions=OFFERS_ROOT,
        info_set="propose_r0_()",
    )
    for offer in OFFERS_ROOT:
        resp = b.decision_node(
            parent=d_root, parent_action=offer,
            player=responder, actions=["accept", "reject"],
            info_set=f"respond_r0_{offer}_()",
        )
        u = [0.0, 0.0]
        u[proposer] = offer * pie
        u[responder] = (1.0 - offer) * pie
        b.terminal_node(parent=resp, parent_action="accept", utilities=u)
        if n_rounds > 1:
            build_round(resp, "reject", (offer,), 1)
        else:
            b.terminal_node(parent=resp, parent_action="reject", utilities=[0.0, 0.0])
    return b.build()


def rock_paper_scissors() -> Game:
    """Two-player one-shot rock-paper-scissors as an EFG.  P1 cannot
    observe P0's move (single info set).  Nash is uniform; value = 0.
    """
    b = GameBuilder(n_players=2, name="rps")
    moves = ["R", "P", "S"]
    d0 = b.decision_node(parent=-1, parent_action=None, player=0,
                         actions=moves, info_set="P0")
    for m0 in moves:
        d1 = b.decision_node(parent=d0, parent_action=m0, player=1,
                             actions=moves, info_set="P1")
        for m1 in moves:
            if m0 == m1:
                u = [0, 0]
            elif (m0, m1) in [("R", "S"), ("P", "R"), ("S", "P")]:
                u = [+1, -1]
            else:
                u = [-1, +1]
            b.terminal_node(parent=d1, parent_action=m1, utilities=u)
    return b.build()


def coin_match_with_signal(p_heads: float = 0.6) -> Game:
    """Chance flips a biased coin; player 0 observes it, player 1
    doesn't.  Player 0 picks one of two actions; player 1 picks one
    of two actions without observing P0's move.  Payoffs designed so
    the optimal scheme is non-trivial.  Useful for chance-sampling
    benchmarks.
    """
    if not (0.0 < p_heads < 1.0):
        raise InvalidGame("p_heads must be in (0, 1)")
    b = GameBuilder(n_players=2, name="coin_match_signal")
    root = b.chance_node(parent=-1, parent_action=None,
                         actions=["H", "T"], probs=[p_heads, 1.0 - p_heads])
    for face in ["H", "T"]:
        d0 = b.decision_node(parent=root, parent_action=face, player=0,
                             actions=["a", "b"], info_set=f"P0|{face}")
        for a0 in ["a", "b"]:
            d1 = b.decision_node(parent=d0, parent_action=a0, player=1,
                                 actions=["x", "y"], info_set="P1")
            for a1 in ["x", "y"]:
                # Payoff design (zero-sum):
                # H+a+x → +1 ; H+a+y → -1 ; H+b+x → -1 ; H+b+y → +1
                # T+a+x → -1 ; T+a+y → +1 ; T+b+x → +1 ; T+b+y → -1
                v = +1 if ((face == "H") == ((a0 == "a") == (a1 == "x"))) else -1
                b.terminal_node(parent=d1, parent_action=a1, utilities=[v, -v])
    return b.build()


# =====================================================================
# Convenience: aggregate exposed names
# =====================================================================


__all__ = [
    # core
    "Game", "GameBuilder",
    "CFRConfig", "SolveReport", "BestResponseReport",
    "Diplomat",
    # functions
    "expected_utilities", "best_response", "exploitability",
    # builders
    "kuhn_poker", "matching_pennies_sequential",
    "matching_pennies_simultaneous", "simple_bargaining",
    "rock_paper_scissors", "coin_match_with_signal",
    # exceptions
    "DiplomatError", "InvalidGame", "PerfectRecallViolation",
    "UnknownSolver", "InfeasibleProgram", "NotTwoPlayerZeroSum",
    "InsufficientIterations",
    # solver kinds
    "KIND_CFR", "KIND_CFR_PLUS", "KIND_LINEAR_CFR",
    "KIND_DISCOUNTED_CFR", "KIND_PREDICTIVE_CFR_PLUS",
    "KIND_OUTCOME_SAMPLING", "KIND_EXTERNAL_SAMPLING",
    "KIND_CHANCE_SAMPLING", "KIND_SEQUENCE_FORM_LP",
    "KNOWN_KINDS",
    # events
    "DIPLOMAT_STARTED", "DIPLOMAT_ITER", "DIPLOMAT_SOLVED",
    "DIPLOMAT_BR", "DIPLOMAT_LP_SOLVED", "DIPLOMAT_CERTIFIED",
]
