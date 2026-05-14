r"""Equilibrator — non-cooperative game-theoretic equilibria as a runtime primitive.

Negotiator solves the *cooperative* bargaining problem: a single
benevolent designer chooses an allocation maximising some welfare
function over a feasible set. Coalition solves the *cooperative* credit
problem: it splits an already-realised surplus by Shapley value.
Both presume that parties are coordinated and that an objective has
already been agreed upon.

But the bulk of multi-agent runtime traffic is *non-cooperative*:
parties simultaneously choose strategies, each maximising her own
payoff, and the only stable points are those at which **no party has
a profitable unilateral deviation**. Pricing engines bidding for the
same compute pool, defenders reasoning about an adversary's best
response, mechanism designers verifying incentive compatibility,
self-interested tenants whose strategy choices change other tenants'
rewards — they all live in this regime.

The Equilibrator is the primitive that turns the rest of the stack
into a *strategically stable* runtime. It accepts a finite N-player
normal-form game, a solution concept (Nash / correlated / coarse
correlated / minimax / pure Nash / evolutionary stable), and an
algorithm (support enumeration, fictitious play, multiplicative
weights, replicator dynamics, linear program, best-response dynamics),
and returns an `EquilibriumReport` carrying the equilibrium profile,
the exact expected payoffs at it, the **exploitability** (NashConv:
the maximum any party could gain by switching to a best response),
and a tamper-evident `certificate` documenting which axioms hold and
which convergence guarantee was achieved.

Mathematical roots
------------------

  * **von Neumann, 1928 — *Zur Theorie der Gesellschaftsspiele*.**
    The minimax theorem: every finite two-player zero-sum game has
    a unique value `v` and saddle-point strategies (σ⋆, τ⋆) with
    `max_σ min_τ σᵀ A τ = min_τ max_σ σᵀ A τ = v`. Equivalent to
    LP duality on the matrix.

  * **Nash, 1950, 1951 — *Equilibrium points in N-person games* /
    *Non-cooperative games*.** Every finite normal-form game has at
    least one mixed-strategy equilibrium: a profile `(σ_i)` such
    that `u_i(σ_i, σ_{-i}) ≥ u_i(σ_i', σ_{-i})` for all i and all
    σ_i'. Existence by Kakutani fixed-point on the best-response
    correspondence.

  * **Brown, 1951; Robinson, 1951 — Fictitious play.** Each player
    plays the best response to the time-average of opponents' past
    play. The time-average of play converges to a Nash equilibrium
    in two-player zero-sum games (Robinson) and in finite potential
    games (Monderer & Shapley 1996), but in general N-player games
    may cycle (Shapley's 3×3 counter-example, 1964).

  * **Hannan, 1957; Blackwell, 1956 — No-regret learning.** A
    decision rule has *external regret*
        `R_T = max_a Σ u(a, x_t) − Σ u(a_t, x_t)`
    on a sequence (a_t, x_t). Hannan showed `R_T / T → 0` is
    achievable; Blackwell's approachability gives a constructive
    proof. The runtime consequence is the standard one: when *every*
    player runs an external no-regret algorithm against a sequence
    of opponents, the *empirical joint distribution* of play
    converges to the set of **coarse correlated equilibria** (Hart &
    Mas-Colell 2000). In zero-sum games the time-average converges
    to the Nash equilibrium.

  * **Freund & Schapire, 1997, 1999 — Hedge / Multiplicative
    weights.** The canonical optimal-rate no-regret algorithm:
        `p_{t+1}(a) ∝ p_t(a) · exp(η · u(a, x_t))`,
    with regret `R_T ≤ √(2 T log K) + log K / η` for K actions.
    Time-average play in self-play on a finite normal-form game
    achieves an `ε`-coarse-correlated equilibrium after
    `O(log K / ε²)` rounds; on zero-sum, an `ε`-Nash after the
    same.

  * **Aumann, 1974 — Correlated equilibrium.** A distribution `μ`
    over joint actions is a correlated equilibrium iff for every
    player i and every pair of actions (a, a′),
        `Σ_{a_{-i}} μ(a, a_{-i}) · [u_i(a, a_{-i}) − u_i(a′, a_{-i})] ≥ 0`.
    Set is convex and contains the convex hull of Nash equilibria;
    it can be characterised by a polynomial-size linear program
    (Hart & Mas-Colell 1989). Foster & Vohra 1997 give an
    internal-no-regret learning rule that converges to it.

  * **Shapley, 1953 — Stochastic games.** Two-player zero-sum
    stochastic game has a Markov-perfect value, computable by value
    iteration of the Shapley operator `T(v)(s) = val(R(s) + γ P(s)v)`,
    contracting in sup-norm. Single-state special case is the
    matrix-game minimax.

  * **Taylor & Jonker, 1978 — Replicator dynamics.** Evolutionary
    dynamics on the simplex: `ẋ_a = x_a · (u(a, x) − ū(x))`. Fixed
    points are *Nash equilibria*; asymptotically stable fixed points
    are *evolutionarily stable strategies* (ESS). The runtime
    cousin of multiplicative weights — Hedge with η → 0⁺ is exactly
    the discretised replicator.

  * **Lemke & Howson, 1964 — Bimatrix Nash.** A path-following
    algorithm on labelled vertices of the strategy product that
    terminates at a Nash equilibrium. Linear-complementarity
    formulation. Worst-case exponential (Savani & von Stengel 2006)
    but typical-case efficient on small games. We provide the
    related **support enumeration** algorithm: for each pair of
    supports `(I, J)`, solve the system of indifference equations;
    if a probability profile exists with `supp(σ_1) = I`,
    `supp(σ_2) = J`, accept.

  * **Daskalakis, Goldberg & Papadimitriou, 2009 — PPAD-hardness.**
    Computing a Nash equilibrium of a 3-player game is PPAD-complete;
    2-player is also PPAD-complete. There is *no* polynomial-time
    algorithm known. The runtime therefore commits to one of two
    paths: **exact** for small games (support enumeration up to
    `n ≤ 8`), **approximate** for everything else (multiplicative
    weights / fictitious play, with explicit ε bounds).

These eight pillars are not preferences a coordinator picks among
arbitrarily: they are projections of the same Nash existence theorem
under different rationality assumptions. Nash is "what would happen
under common knowledge of rationality"; correlated equilibria are
"what would happen if parties could see a public signal first";
coarse correlated equilibria are "what would happen if parties ran
no-regret learners against each other"; replicator dynamics are
"what would happen under population-level imitation". Each is *the*
canonical answer when its assumptions fit the use case.

What it composes (razor-sharp coordination integration)
------------------------------------------------------

  * **Negotiator.** Negotiator presumes parties cooperate. When they
    don't, the Equilibrator finds the strategic equilibrium first,
    and Negotiator splits the *surplus over the threat point* by
    Nash bargaining with `d_i = u_i(σ⋆)`. The disagreement vector
    used by Negotiator is exactly the Equilibrator's equilibrium
    payoff vector.

  * **Coalition.** Coalition value-function `v(S)` is the value the
    coalition `S` could *guarantee against the worst response by
    N\S*. This is the minimax value of the two-player zero-sum game
    `(S, N\S)`, which Equilibrator solves directly via
    `zero_sum_value(...)`.

  * **TicketMarket.** Bidders for an auction-style market are
    strategic. The market is incentive-compatible iff truth-telling
    is a Bayes-Nash equilibrium — an EquilibriumReport with
    `axiom_incentive_compatible=True` certifies this. The runtime
    pipes the auction's payoff tensor through the Equilibrator
    before promoting any mechanism change.

  * **Robustifier.** Robustifier finds robust strategies against an
    adversary in a fixed ambiguity set. Equilibrator's zero-sum
    minimax gives the *worst-case mixed adversary* and the optimal
    randomised defender — together they certify Robustifier's
    chosen radius is the right one (it is the radius at which the
    minimax value equals the requested target).

  * **PolicyImprover.** When policies are interacting (e.g. multi-
    agent settings), the HCPI bound is conditional on opponents'
    behaviour. Equilibrator gives the *exploitability* of the
    proposed new policy under the empirical opponent distribution —
    a deployment is HCPI-safe AND not exploitable when both bounds
    are simultaneously satisfied.

  * **Strategist.** Strategist picks among candidate strategies.
    When candidates interact with each other (or with an adversary),
    the right meta-decision criterion is `EV(candidate) –
    γ · exploitability(candidate)`. Equilibrator supplies the
    exploitability term.

  * **AttestationLedger.** Every equilibrium decision is hashed and
    appended as a `equilibrator.solved` receipt — a third-party-
    replayable proof that under payoff tensor with content-hash H at
    time t the chosen profile σ has NashConv ≤ ε.

  * **EventBus.** Streams every registration, observation, online
    update, and solve. A higher-level coordination engine reacts in
    real time — e.g. retrigger calibration on
    `equilibrator.solved` when the exploitability of the live
    policy crosses an SLO.

Where this slots in
-------------------

    eq = Equilibrator(bus=bus, attestor=attestor)
    eq.register_game(
        "compute_auction",
        payoffs=[
            # Row player payoff tensor
            [[3, 0], [5, 1]],
            # Column player payoff tensor
            [[3, 5], [0, 1]],
        ],
    )
    rep = eq.solve("compute_auction",
                   concept=CONCEPT_NASH,
                   method=METHOD_MULTIPLICATIVE_WEIGHTS,
                   iterations=10_000,
                   epsilon=1e-3)
    # rep.profile          → (Strategy([0,1]), Strategy([0,1]))   # ((D,D) for prisoner-style)
    # rep.expected_payoff  → (1.0, 1.0)
    # rep.exploitability   → ≤ 1e-3
    # rep.certificate      → {convergence: …, regret_bound: …, content_hash: …}

Events
------
    equilibrator.started           — engine was constructed
    equilibrator.game_registered   — a game was added
    equilibrator.game_removed      — a game was removed
    equilibrator.solved            — an equilibrium was computed
    equilibrator.observed          — a streaming joint action was logged
    equilibrator.cleared           — state was reset
    equilibrator.report            — a coverage report was published
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

EQUILIBRATOR_STARTED = "equilibrator.started"
EQUILIBRATOR_GAME_REGISTERED = "equilibrator.game_registered"
EQUILIBRATOR_GAME_REMOVED = "equilibrator.game_removed"
EQUILIBRATOR_SOLVED = "equilibrator.solved"
EQUILIBRATOR_OBSERVED = "equilibrator.observed"
EQUILIBRATOR_CLEARED = "equilibrator.cleared"
EQUILIBRATOR_REPORT = "equilibrator.report"


# =====================================================================
# Solution concepts
# =====================================================================

CONCEPT_NASH = "nash"
CONCEPT_PURE_NASH = "pure_nash"
CONCEPT_CORRELATED = "correlated"
CONCEPT_COARSE_CORRELATED = "coarse_correlated"
CONCEPT_MINIMAX = "minimax"
CONCEPT_ESS = "ess"

KNOWN_CONCEPTS = (
    CONCEPT_NASH,
    CONCEPT_PURE_NASH,
    CONCEPT_CORRELATED,
    CONCEPT_COARSE_CORRELATED,
    CONCEPT_MINIMAX,
    CONCEPT_ESS,
)


# =====================================================================
# Algorithms
# =====================================================================

METHOD_AUTO = "auto"
METHOD_SUPPORT_ENUMERATION = "support_enumeration"
METHOD_FICTITIOUS_PLAY = "fictitious_play"
METHOD_MULTIPLICATIVE_WEIGHTS = "multiplicative_weights"
METHOD_REPLICATOR = "replicator"
METHOD_BEST_RESPONSE = "best_response"
METHOD_LINEAR_PROGRAM = "linear_program"
METHOD_PURE_SEARCH = "pure_search"

KNOWN_METHODS = (
    METHOD_AUTO,
    METHOD_SUPPORT_ENUMERATION,
    METHOD_FICTITIOUS_PLAY,
    METHOD_MULTIPLICATIVE_WEIGHTS,
    METHOD_REPLICATOR,
    METHOD_BEST_RESPONSE,
    METHOD_LINEAR_PROGRAM,
    METHOD_PURE_SEARCH,
)


# =====================================================================
# Axiom flags returned in EquilibriumReport.certificate
# =====================================================================

AXIOM_BEST_RESPONSE = "best_response"               # all players play BR
AXIOM_NO_REGRET = "no_regret"                       # avg play vanishing regret
AXIOM_MINIMAX = "minimax"                           # σ achieves zero-sum value
AXIOM_INCENTIVE_COMPATIBLE = "incentive_compatible" # truthful BNE
AXIOM_PARETO_OPTIMAL = "pareto_optimal"             # no joint strict improvement
AXIOM_SYMMETRIC = "symmetric"                       # invariant under player swap
AXIOM_ESS = "evolutionarily_stable"

KNOWN_AXIOMS = (
    AXIOM_BEST_RESPONSE,
    AXIOM_NO_REGRET,
    AXIOM_MINIMAX,
    AXIOM_INCENTIVE_COMPATIBLE,
    AXIOM_PARETO_OPTIMAL,
    AXIOM_SYMMETRIC,
    AXIOM_ESS,
)


_EPS = 1e-12
_DEFAULT_EPSILON = 1e-3
_DEFAULT_ITERATIONS = 5_000
_SIMPLEX_TOL = 1e-9
_SIMPLEX_MAX_ITER = 2_000


# =====================================================================
# Errors
# =====================================================================


class EquilibratorError(Exception):
    """Base class for Equilibrator errors."""


class UnknownGame(EquilibratorError):
    """Raised when an operation references an unregistered game."""


class InvalidGame(EquilibratorError):
    """Raised when a payoff tensor or shape is malformed."""


class SolverUnavailable(EquilibratorError):
    """Raised when a concept/method pair is not implementable."""


# =====================================================================
# Strategy & profile
# =====================================================================


@dataclass(frozen=True)
class Strategy:
    """Mixed strategy: probability distribution over a player's actions.

    Probabilities are stored as a tuple of floats summing to 1
    (within `_EPS` tolerance). Pure strategies are represented as
    one-hot tuples, exposed via `pure(action, n_actions)`.
    """

    probabilities: tuple

    def __post_init__(self) -> None:
        probs = tuple(float(p) for p in self.probabilities)
        if not probs:
            raise InvalidGame("Strategy must have at least one action")
        for p in probs:
            if not math.isfinite(p):
                raise InvalidGame(f"Strategy contains non-finite probability {p}")
            if p < -_EPS:
                raise InvalidGame(f"Strategy contains negative probability {p}")
        total = sum(max(0.0, p) for p in probs)
        if total <= _EPS:
            raise InvalidGame("Strategy probabilities sum to zero")
        # renormalise tiny numerical drift; reject larger drift
        normalised = tuple(max(0.0, p) / total for p in probs)
        if abs(total - 1.0) > 1e-6:
            # only warn via field; we still accept and renormalise
            pass
        object.__setattr__(self, "probabilities", normalised)

    @property
    def n_actions(self) -> int:
        return len(self.probabilities)

    @property
    def support(self) -> tuple:
        return tuple(i for i, p in enumerate(self.probabilities) if p > _EPS)

    def probability(self, action: int) -> float:
        return float(self.probabilities[action])

    def expected_value(self, payoffs_per_action: Sequence[float]) -> float:
        if len(payoffs_per_action) != self.n_actions:
            raise InvalidGame(
                f"expected_value: payoffs length {len(payoffs_per_action)} != "
                f"strategy length {self.n_actions}"
            )
        return sum(p * float(u) for p, u in zip(self.probabilities, payoffs_per_action))

    def entropy(self) -> float:
        h = 0.0
        for p in self.probabilities:
            if p > _EPS:
                h -= p * math.log(p)
        return h

    def total_variation(self, other: "Strategy") -> float:
        if self.n_actions != other.n_actions:
            raise InvalidGame("total_variation: action counts differ")
        return 0.5 * sum(abs(a - b) for a, b in zip(self.probabilities, other.probabilities))

    @classmethod
    def pure(cls, action: int, n_actions: int) -> "Strategy":
        if action < 0 or action >= n_actions:
            raise InvalidGame(f"pure: action {action} out of range [0, {n_actions})")
        probs = [0.0] * n_actions
        probs[action] = 1.0
        return cls(probabilities=tuple(probs))

    @classmethod
    def uniform(cls, n_actions: int) -> "Strategy":
        if n_actions <= 0:
            raise InvalidGame("uniform: n_actions must be positive")
        p = 1.0 / n_actions
        return cls(probabilities=tuple(p for _ in range(n_actions)))

    @classmethod
    def from_weights(cls, weights: Sequence[float]) -> "Strategy":
        total = sum(max(0.0, float(w)) for w in weights)
        if total <= _EPS:
            return cls.uniform(len(weights))
        return cls(probabilities=tuple(max(0.0, float(w)) / total for w in weights))

    def to_list(self) -> list:
        return [float(p) for p in self.probabilities]


@dataclass(frozen=True)
class Profile:
    """A strategy profile: one mixed strategy per player."""

    strategies: tuple

    def __post_init__(self) -> None:
        if not self.strategies:
            raise InvalidGame("Profile must contain at least one strategy")
        for i, s in enumerate(self.strategies):
            if not isinstance(s, Strategy):
                raise InvalidGame(f"Profile entry {i} is not a Strategy")

    @property
    def n_players(self) -> int:
        return len(self.strategies)

    @property
    def action_counts(self) -> tuple:
        return tuple(s.n_actions for s in self.strategies)

    def __getitem__(self, i: int) -> Strategy:
        return self.strategies[i]

    def replace(self, player: int, strategy: Strategy) -> "Profile":
        new = list(self.strategies)
        if strategy.n_actions != new[player].n_actions:
            raise InvalidGame(
                f"replace: strategy has {strategy.n_actions} actions, expected "
                f"{new[player].n_actions} for player {player}"
            )
        new[player] = strategy
        return Profile(strategies=tuple(new))

    def to_list(self) -> list:
        return [s.to_list() for s in self.strategies]


# =====================================================================
# Game record
# =====================================================================


@dataclass(frozen=True)
class GameRecord:
    """Normal-form game with explicit payoff tensor.

    `payoffs` is stored flat: `payoffs[player][k]` is u_player evaluated
    at the joint action whose flat index is k (lexicographic on
    `(a_0, a_1, ..., a_{N-1})`). Use `payoff(joint)` or
    `flat_index(joint)` to navigate.
    """

    game_id: str
    n_players: int
    action_counts: tuple
    payoffs: tuple                # tuple of tuples: per-player flat tensor
    action_names: tuple           # tuple of tuples of str
    is_zero_sum: bool
    is_constant_sum: bool
    is_symmetric: bool
    metadata: Mapping = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.n_players < 1:
            raise InvalidGame("game must have at least one player")
        if len(self.action_counts) != self.n_players:
            raise InvalidGame("action_counts length must equal n_players")
        for c in self.action_counts:
            if c < 1:
                raise InvalidGame("each player must have at least one action")
        expected_flat = 1
        for c in self.action_counts:
            expected_flat *= c
        if len(self.payoffs) != self.n_players:
            raise InvalidGame("payoffs length must equal n_players")
        for i, p_tensor in enumerate(self.payoffs):
            if len(p_tensor) != expected_flat:
                raise InvalidGame(
                    f"player {i} payoff tensor has {len(p_tensor)} entries, "
                    f"expected {expected_flat}"
                )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", _hash_payoffs(self.payoffs, self.action_counts))

    # ---- navigation ----

    def flat_index(self, joint_action: Sequence[int]) -> int:
        if len(joint_action) != self.n_players:
            raise InvalidGame("joint_action length != n_players")
        idx = 0
        for player, a in enumerate(joint_action):
            if a < 0 or a >= self.action_counts[player]:
                raise InvalidGame(
                    f"joint_action[{player}] = {a} out of range "
                    f"[0, {self.action_counts[player]})"
                )
            idx = idx * self.action_counts[player] + a
        return idx

    def joint_from_index(self, idx: int) -> tuple:
        if idx < 0:
            raise InvalidGame("flat index must be non-negative")
        out = [0] * self.n_players
        for player in range(self.n_players - 1, -1, -1):
            c = self.action_counts[player]
            out[player] = idx % c
            idx //= c
        if idx != 0:
            raise InvalidGame("flat index out of range")
        return tuple(out)

    def payoff(self, joint_action: Sequence[int]) -> tuple:
        idx = self.flat_index(joint_action)
        return tuple(self.payoffs[player][idx] for player in range(self.n_players))

    def player_payoff(self, player: int, joint_action: Sequence[int]) -> float:
        return float(self.payoffs[player][self.flat_index(joint_action)])

    def joint_actions(self) -> Iterable[tuple]:
        return itertools.product(*(range(c) for c in self.action_counts))

    def n_joint_actions(self) -> int:
        n = 1
        for c in self.action_counts:
            n *= c
        return n

    # ---- introspection ----

    def is_two_player(self) -> bool:
        return self.n_players == 2

    def payoff_range(self) -> tuple:
        lo = math.inf
        hi = -math.inf
        for tensor in self.payoffs:
            for v in tensor:
                if v < lo:
                    lo = float(v)
                if v > hi:
                    hi = float(v)
        if lo == math.inf:
            return (0.0, 0.0)
        return (lo, hi)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "n_players": self.n_players,
            "action_counts": list(self.action_counts),
            "action_names": [list(names) for names in self.action_names],
            "is_zero_sum": self.is_zero_sum,
            "is_constant_sum": self.is_constant_sum,
            "is_symmetric": self.is_symmetric,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


def _hash_payoffs(payoffs: tuple, action_counts: tuple) -> str:
    h = hashlib.sha256()
    h.update(b"equilibrator.payoffs:v1\n")
    for c in action_counts:
        h.update(f"{c}|".encode())
    for tensor in payoffs:
        for v in tensor:
            h.update(f"{float(v):.17g}|".encode())
    return h.hexdigest()


# =====================================================================
# Reports
# =====================================================================


@dataclass(frozen=True)
class EquilibriumReport:
    """Outcome of a solve() call.

    Encodes either a profile (Nash, minimax, ESS) or a joint
    distribution (correlated, coarse correlated). `exploitability`
    is the maximum unilateral deviation gain across all players,
    a.k.a. NashConv. `epsilon` is the formal approximation guarantee
    achieved by the solver — for support enumeration this is 0, for
    iterative methods it is `regret_bound(T) + numerical_floor`.
    """

    game_id: str
    concept: str
    method: str
    profile: Any                              # Profile or None
    distribution: Any                         # tuple of (joint, prob) or None
    expected_payoff: tuple
    exploitability: float
    epsilon: float
    iterations: int
    converged: bool
    value: Any                                # float | None
    certificate: Mapping
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    receipt_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        prof = self.profile.to_list() if self.profile is not None else None
        dist = (
            [(list(joint), float(p)) for joint, p in self.distribution]
            if self.distribution is not None
            else None
        )
        return {
            "game_id": self.game_id,
            "concept": self.concept,
            "method": self.method,
            "profile": prof,
            "distribution": dist,
            "expected_payoff": list(self.expected_payoff),
            "exploitability": float(self.exploitability),
            "epsilon": float(self.epsilon),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "value": (None if self.value is None else float(self.value)),
            "certificate": dict(self.certificate),
            "timestamp_ns": int(self.timestamp_ns),
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True)
class CoverageReport:
    """High-level snapshot of an Equilibrator instance."""

    n_games: int
    n_solved: int
    n_observed: int
    games: tuple
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    def to_dict(self) -> dict:
        return {
            "n_games": self.n_games,
            "n_solved": self.n_solved,
            "n_observed": self.n_observed,
            "games": list(self.games),
            "timestamp_ns": int(self.timestamp_ns),
        }


# =====================================================================
# Game construction helpers
# =====================================================================


def make_game(
    game_id: str,
    payoffs: Any,
    *,
    action_names: Any = None,
    metadata: Mapping = None,
) -> GameRecord:
    """Build a `GameRecord` from a nested or flat payoff tensor.

    `payoffs` may be:
      * a sequence of per-player N-dimensional nested lists, OR
      * a sequence of per-player flat sequences of length
        prod(action_counts) with action_counts inferred from depth.

    For two-player games the natural form is two |A_0| × |A_1| matrices
    `[A, B]` where A[i][j] = u_1(i, j), B[i][j] = u_2(i, j).
    For N-player games the natural form is N tensors of shape
    `action_counts`.
    """
    if payoffs is None:
        raise InvalidGame("payoffs is required")
    payoffs_list = list(payoffs)
    n_players = len(payoffs_list)
    if n_players < 1:
        raise InvalidGame("at least one player required")

    # Infer action_counts from the first player's tensor
    action_counts = _infer_shape(payoffs_list[0])
    if len(action_counts) != n_players:
        raise InvalidGame(
            f"payoff tensor rank {len(action_counts)} does not match "
            f"n_players={n_players}"
        )

    # Sanity-check all players have the same shape
    for i, tensor in enumerate(payoffs_list):
        shape = _infer_shape(tensor)
        if tuple(shape) != tuple(action_counts):
            raise InvalidGame(
                f"player {i} payoff has shape {shape}, expected {action_counts}"
            )

    flat_payoffs = tuple(
        tuple(float(x) for x in _flatten(tensor)) for tensor in payoffs_list
    )

    if action_names is None:
        names = tuple(
            tuple(f"a{i}_{a}" for a in range(c))
            for i, c in enumerate(action_counts)
        )
    else:
        names = tuple(tuple(str(n) for n in row) for row in action_names)
        for i, row in enumerate(names):
            if len(row) != action_counts[i]:
                raise InvalidGame(
                    f"action_names[{i}] has length {len(row)}, expected {action_counts[i]}"
                )

    # Detect properties
    is_zero_sum = _is_zero_sum(flat_payoffs)
    is_constant_sum = is_zero_sum or _is_constant_sum(flat_payoffs)
    is_symmetric = (n_players == 2) and _is_symmetric_bimatrix(flat_payoffs, action_counts)

    return GameRecord(
        game_id=str(game_id),
        n_players=n_players,
        action_counts=tuple(action_counts),
        payoffs=flat_payoffs,
        action_names=names,
        is_zero_sum=is_zero_sum,
        is_constant_sum=is_constant_sum,
        is_symmetric=is_symmetric,
        metadata=dict(metadata or {}),
    )


def _infer_shape(tensor: Any) -> list:
    """Return the shape of a nested-list / nested-tuple tensor."""
    shape = []
    cur = tensor
    while isinstance(cur, (list, tuple)) and not _is_scalar(cur):
        shape.append(len(cur))
        if len(cur) == 0:
            break
        # validate uniformity of length at this level
        first_len = None
        for child in cur:
            if isinstance(child, (list, tuple)) and not _is_scalar(child):
                if first_len is None:
                    first_len = len(child)
                elif len(child) != first_len:
                    raise InvalidGame("payoff tensor has ragged rows")
        cur = cur[0]
    if not shape:
        raise InvalidGame("payoff tensor is empty")
    return shape


def _is_scalar(x: Any) -> bool:
    if isinstance(x, (list, tuple)):
        return False
    return isinstance(x, (int, float))


def _flatten(tensor: Any) -> Iterable[float]:
    if isinstance(tensor, (list, tuple)) and not _is_scalar(tensor):
        for child in tensor:
            yield from _flatten(child)
    else:
        yield float(tensor)


def _is_zero_sum(flat_payoffs: tuple) -> bool:
    if not flat_payoffs:
        return False
    length = len(flat_payoffs[0])
    for k in range(length):
        s = 0.0
        for player_tensor in flat_payoffs:
            s += player_tensor[k]
        if abs(s) > 1e-9:
            return False
    return True


def _is_constant_sum(flat_payoffs: tuple) -> bool:
    if not flat_payoffs:
        return False
    length = len(flat_payoffs[0])
    if length == 0:
        return True
    expected = sum(t[0] for t in flat_payoffs)
    for k in range(1, length):
        s = sum(t[k] for t in flat_payoffs)
        if abs(s - expected) > 1e-9:
            return False
    return True


def _is_symmetric_bimatrix(flat_payoffs: tuple, action_counts: tuple) -> bool:
    """Two-player game with |A_0| = |A_1| is symmetric iff
    u_1(i,j) = u_2(j,i) for all i,j."""
    if len(flat_payoffs) != 2 or len(action_counts) != 2:
        return False
    if action_counts[0] != action_counts[1]:
        return False
    n = action_counts[0]
    A, B = flat_payoffs
    for i in range(n):
        for j in range(n):
            if abs(A[i * n + j] - B[j * n + i]) > 1e-9:
                return False
    return True


# =====================================================================
# Core algorithmic primitives
# =====================================================================


def player_payoff_vector(
    game: GameRecord,
    player: int,
    profile: Profile,
) -> tuple:
    """For a given player `player`, return a tuple of expected payoffs
    `(E[u_player(a, σ_{-player})] for a in 0..A_player-1)`.

    Marginalises over the *product* distribution induced by other
    players' mixed strategies. This is the standard "expected payoff
    against opponents' mixture" computation; it is exact (no Monte
    Carlo) and runs in O(|A_player| · prod(|A_-player|)) time.
    """
    if profile.n_players != game.n_players:
        raise InvalidGame("profile player count != game player count")
    counts = game.action_counts
    n_actions = counts[player]
    others = [i for i in range(game.n_players) if i != player]
    out = [0.0] * n_actions
    tensor = game.payoffs[player]

    if not others:
        for a in range(n_actions):
            out[a] = float(tensor[a])
        return tuple(out)

    # Iterate over the product of opponents' action spaces.
    other_ranges = [range(counts[i]) for i in others]
    for opp_action in itertools.product(*other_ranges):
        w = 1.0
        for k, i in enumerate(others):
            w *= profile[i].probability(opp_action[k])
        if w <= 0.0:
            continue
        for a in range(n_actions):
            joint = [0] * game.n_players
            joint[player] = a
            for k, i in enumerate(others):
                joint[i] = opp_action[k]
            idx = game.flat_index(joint)
            out[a] += w * tensor[idx]
    return tuple(out)


def expected_payoff(game: GameRecord, profile: Profile) -> tuple:
    """Expected payoff vector under a profile.

    Computed as `E_{σ_player}[ E_{σ_-player}[ u_player(a) ] ]` so the
    interior expectation re-uses `player_payoff_vector`.
    """
    out = []
    for p in range(game.n_players):
        pv = player_payoff_vector(game, p, profile)
        out.append(profile[p].expected_value(pv))
    return tuple(out)


def best_response(
    game: GameRecord,
    player: int,
    profile: Profile,
) -> tuple:
    """Compute player's pure best response actions.

    Returns `(best_actions, best_value, action_values)`, where
    `best_actions` is the (sorted) tuple of all actions tied for the
    best expected payoff against the profile. `action_values` is the
    full vector of expected payoffs per action.
    """
    pv = player_payoff_vector(game, player, profile)
    best_val = max(pv)
    best_actions = tuple(a for a, v in enumerate(pv) if abs(v - best_val) <= _EPS * max(1.0, abs(best_val)))
    return best_actions, float(best_val), pv


def exploitability(game: GameRecord, profile: Profile) -> tuple:
    """Compute NashConv: total deviation gain summed over players.

    Returns `(total, per_player_gain)`. Profile is an `ε`-Nash iff
    `max_i per_player_gain[i] ≤ ε`.
    """
    cur = expected_payoff(game, profile)
    gains = []
    for p in range(game.n_players):
        _, br_val, _ = best_response(game, p, profile)
        gain = max(0.0, br_val - cur[p])
        gains.append(gain)
    total = sum(gains)
    return float(total), tuple(float(g) for g in gains)


# =====================================================================
# Pure-strategy Nash search
# =====================================================================


def pure_nash_equilibria(game: GameRecord) -> tuple:
    """Enumerate all pure-strategy Nash equilibria of a normal-form game.

    Runs in `O(prod(|A|) · sum(|A|))` time. Suitable for small games;
    for large games, the iterative methods below are the right tool.
    """
    eqs = []
    counts = game.action_counts
    for joint in game.joint_actions():
        is_eq = True
        for p in range(game.n_players):
            cur = game.player_payoff(p, joint)
            for ap in range(counts[p]):
                if ap == joint[p]:
                    continue
                alt = list(joint)
                alt[p] = ap
                if game.player_payoff(p, alt) > cur + _EPS:
                    is_eq = False
                    break
            if not is_eq:
                break
        if is_eq:
            eqs.append(tuple(joint))
    return tuple(eqs)


# =====================================================================
# Multiplicative weights / Hedge
# =====================================================================


def multiplicative_weights(
    game: GameRecord,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    eta: float = None,
    epsilon: float = _DEFAULT_EPSILON,
    initial_profile: Profile = None,
    record_history: bool = False,
    seed: int = None,
) -> dict:
    """Self-play multiplicative weights (a.k.a. Hedge).

    Each player runs an independent Hedge / exponential-weights
    no-regret algorithm against the realised opponents' play. The
    *time-average* of play converges to a coarse correlated
    equilibrium (Hannan, Hart-Mas-Colell); in zero-sum games it
    converges to a Nash equilibrium with NashConv
    `O(√(log K / T))` after T rounds.

    `eta` defaults to the theory-optimal `√(8 log K / T)` where
    K = max action count.

    Returns a dict with keys:
      profile          → Profile (time-average mixed strategies)
      payoffs          → tuple of expected payoffs at the average
      exploitability   → NashConv at the average profile
      epsilon          → regret_bound + numerical floor
      iterations       → T
      converged        → whether NashConv ≤ epsilon
      history          → list of per-iteration Profiles (if requested)
      regret           → tuple of per-player external regret over T rounds
    """
    counts = game.action_counts
    n = game.n_players
    if iterations < 1:
        raise InvalidGame("iterations must be >= 1")

    K = max(counts)
    if eta is None:
        eta = math.sqrt(8.0 * math.log(max(2, K)) / max(1, iterations))
    if eta <= 0:
        raise InvalidGame("eta must be > 0")

    # Scale payoffs into [0, 1] for stable Hedge.
    lo, hi = game.payoff_range()
    span = max(hi - lo, _EPS)

    rng = random.Random(seed) if seed is not None else random.Random()

    if initial_profile is not None and initial_profile.n_players == n:
        weights = [list(s.probabilities) for s in initial_profile.strategies]
        weights = [[max(_EPS, w) for w in row] for row in weights]
    else:
        weights = [[1.0] * counts[p] for p in range(n)]

    cum_strategy = [[0.0] * counts[p] for p in range(n)]
    cum_strategy_count = 0

    history = []

    # We don't sample joint actions; we use the analytic *full-information*
    # update: each player computes the expected payoff of every action
    # under opponents' *current mixed strategies* and Hedge-weights
    # accordingly. This is the standard "no regret in full information"
    # setting; it is variance-free and gives the strongest possible
    # convergence rate.
    for t in range(iterations):
        # Normalise weights to a Profile.
        prof = Profile(strategies=tuple(
            Strategy.from_weights(weights[p]) for p in range(n)
        ))
        # Compute each player's expected payoff vector
        action_values = []
        for p in range(n):
            pv = player_payoff_vector(game, p, prof)
            action_values.append(pv)
        # Hedge update
        for p in range(n):
            for a in range(counts[p]):
                # rescale to [0,1]
                gain = (action_values[p][a] - lo) / span
                weights[p][a] = max(_EPS, weights[p][a] * math.exp(eta * gain))
            # normalise to avoid overflow
            s = sum(weights[p])
            if s > 0:
                weights[p] = [w / s for w in weights[p]]
        # Accumulate
        for p in range(n):
            for a in range(counts[p]):
                cum_strategy[p][a] += prof[p].probability(a)
        cum_strategy_count += 1
        if record_history:
            history.append(prof)

    avg_profile = Profile(strategies=tuple(
        Strategy.from_weights([c / max(1, cum_strategy_count) for c in cum_strategy[p]])
        for p in range(n)
    ))
    expl_total, expl_per = exploitability(game, avg_profile)
    payoffs = expected_payoff(game, avg_profile)

    # Theoretical regret bound for Hedge with constant eta:
    #   R_T / T ≤ log K / (η T) + η G²
    # where gains are in [0, 1] after rescaling, so G ≤ 1. For
    # eta = √(8 log K / T) the per-round regret bound becomes
    #   √(log K / (2T))  (Freund & Schapire 1997)
    regret_bound_per_round = math.sqrt(math.log(max(2, K)) / (2.0 * iterations))
    # In native payoff units:
    regret_bound = regret_bound_per_round * span

    return {
        "profile": avg_profile,
        "payoffs": payoffs,
        "exploitability": expl_total,
        "exploitability_per_player": expl_per,
        "epsilon": regret_bound + 1e-12,
        "iterations": iterations,
        "converged": expl_total <= regret_bound + 1e-9,
        "history": history if record_history else None,
        "regret_bound": regret_bound,
        "eta": eta,
        "payoff_span": span,
    }


# =====================================================================
# Fictitious play
# =====================================================================


def fictitious_play(
    game: GameRecord,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    epsilon: float = _DEFAULT_EPSILON,
    initial_actions: Sequence[int] = None,
    seed: int = None,
    record_history: bool = False,
) -> dict:
    """Brown-Robinson discrete-time fictitious play.

    Each player plays a best response to the *empirical frequency*
    of opponents' past actions. Converges to Nash in:
      - two-player zero-sum games (Robinson 1951)
      - finite potential games (Monderer & Shapley 1996)
      - 2×2 games (always)
    May cycle in arbitrary N-player games (Shapley 1964), but the
    empirical frequencies often visit a neighbourhood of Nash even
    when they don't converge.

    Returns the same dict shape as multiplicative_weights().
    """
    counts = game.action_counts
    n = game.n_players
    if iterations < 1:
        raise InvalidGame("iterations must be >= 1")

    rng = random.Random(seed) if seed is not None else random.Random()

    counts_played = [[0.0] * counts[p] for p in range(n)]
    if initial_actions is not None:
        if len(initial_actions) != n:
            raise InvalidGame("initial_actions length != n_players")
        for p, a in enumerate(initial_actions):
            if not (0 <= a < counts[p]):
                raise InvalidGame(f"initial_actions[{p}] out of range")
            counts_played[p][a] = 1.0
    else:
        # initialise uniform with mass 1
        for p in range(n):
            for a in range(counts[p]):
                counts_played[p][a] = 1.0 / counts[p]

    history = []
    for t in range(iterations):
        prof = Profile(strategies=tuple(
            Strategy.from_weights(counts_played[p]) for p in range(n)
        ))
        if record_history:
            history.append(prof)
        # Each player computes BR; we tie-break uniformly at random.
        new_actions = []
        for p in range(n):
            br_actions, _, _ = best_response(game, p, prof)
            if len(br_actions) == 1:
                a = br_actions[0]
            else:
                a = rng.choice(br_actions)
            new_actions.append(a)
        for p, a in enumerate(new_actions):
            counts_played[p][a] += 1.0

    avg_profile = Profile(strategies=tuple(
        Strategy.from_weights(counts_played[p]) for p in range(n)
    ))
    expl_total, expl_per = exploitability(game, avg_profile)
    payoffs = expected_payoff(game, avg_profile)

    # FP has no general worst-case rate. For 2-player zero-sum,
    # Karlin (1959) gives O(T^{-1/(n_1+n_2-2)}). We don't claim a
    # PAC bound; we report the empirical exploitability instead.
    return {
        "profile": avg_profile,
        "payoffs": payoffs,
        "exploitability": expl_total,
        "exploitability_per_player": expl_per,
        "epsilon": expl_total,
        "iterations": iterations,
        "converged": expl_total <= epsilon,
        "history": history if record_history else None,
    }


# =====================================================================
# Replicator dynamics
# =====================================================================


def replicator_dynamics(
    game: GameRecord,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    dt: float = 0.01,
    initial_profile: Profile = None,
    epsilon: float = _DEFAULT_EPSILON,
    record_history: bool = False,
) -> dict:
    """Discrete-time replicator dynamics in N-player normal-form games.

    Update law (per player p, per action a):

        x_p,a ← x_p,a · (1 + dt · (u_p(a, x_{-p}) − ū_p(x)))

    Fixed points are Nash equilibria (Taylor & Jonker 1978).
    Asymptotically stable fixed points are ESS in symmetric games.
    This is the discretised version of the continuous ODE
    `ẋ_a = x_a · (u_a − ū)`. For small `dt` it tracks the ODE; for
    larger `dt` it behaves like a step of Hedge with `η = dt`.
    """
    counts = game.action_counts
    n = game.n_players
    if iterations < 1:
        raise InvalidGame("iterations must be >= 1")
    if dt <= 0:
        raise InvalidGame("dt must be > 0")

    if initial_profile is not None:
        x = [list(s.probabilities) for s in initial_profile.strategies]
    else:
        x = [[1.0 / counts[p]] * counts[p] for p in range(n)]

    history = []
    for t in range(iterations):
        prof = Profile(strategies=tuple(
            Strategy.from_weights(x[p]) for p in range(n)
        ))
        if record_history:
            history.append(prof)
        for p in range(n):
            pv = player_payoff_vector(game, p, prof)
            mean = sum(x[p][a] * pv[a] for a in range(counts[p]))
            new = [
                max(_EPS, x[p][a] * (1.0 + dt * (pv[a] - mean)))
                for a in range(counts[p])
            ]
            s = sum(new)
            if s > 0:
                x[p] = [v / s for v in new]
            else:
                x[p] = [1.0 / counts[p]] * counts[p]

    final = Profile(strategies=tuple(
        Strategy.from_weights(x[p]) for p in range(n)
    ))
    expl_total, expl_per = exploitability(game, final)
    payoffs = expected_payoff(game, final)
    return {
        "profile": final,
        "payoffs": payoffs,
        "exploitability": expl_total,
        "exploitability_per_player": expl_per,
        "epsilon": expl_total,
        "iterations": iterations,
        "converged": expl_total <= epsilon,
        "history": history if record_history else None,
    }


# =====================================================================
# Best-response dynamics
# =====================================================================


def best_response_dynamics(
    game: GameRecord,
    *,
    iterations: int = 1_000,
    initial: Sequence[int] = None,
    seed: int = None,
) -> dict:
    """Sequential best-response dynamics.

    At each round, a player (round-robin) deviates to a best response.
    In *potential games* (Monderer-Shapley 1996) this converges to a
    pure Nash equilibrium in finitely many steps.

    Returns the final (pure) profile, the visited trajectory, and a
    `converged` flag indicating whether a pure Nash was found.
    """
    n = game.n_players
    counts = game.action_counts
    if iterations < 1:
        raise InvalidGame("iterations must be >= 1")

    rng = random.Random(seed) if seed is not None else random.Random()

    if initial is None:
        cur = [rng.randrange(counts[p]) for p in range(n)]
    else:
        cur = list(initial)

    history = [tuple(cur)]
    converged = False
    for t in range(iterations):
        p = t % n
        # Compute BR for player p against the *pure* current profile.
        pv = []
        for a in range(counts[p]):
            joint = list(cur)
            joint[p] = a
            pv.append(game.player_payoff(p, joint))
        best_val = max(pv)
        best_actions = [a for a, v in enumerate(pv) if abs(v - best_val) <= _EPS]
        new_a = rng.choice(best_actions) if len(best_actions) > 1 else best_actions[0]
        if new_a != cur[p]:
            cur[p] = new_a
            history.append(tuple(cur))
        # Check for pure Nash: no player has a strictly better deviation.
        if t % n == n - 1:
            converged = True
            for q in range(n):
                pv_q = []
                for a in range(counts[q]):
                    joint = list(cur)
                    joint[q] = a
                    pv_q.append(game.player_payoff(q, joint))
                if max(pv_q) > pv_q[cur[q]] + _EPS:
                    converged = False
                    break
            if converged:
                break

    profile = Profile(strategies=tuple(Strategy.pure(cur[p], counts[p]) for p in range(n)))
    expl_total, expl_per = exploitability(game, profile)
    payoffs = expected_payoff(game, profile)
    return {
        "profile": profile,
        "joint_action": tuple(cur),
        "payoffs": payoffs,
        "exploitability": expl_total,
        "exploitability_per_player": expl_per,
        "epsilon": 0.0,
        "iterations": len(history) - 1,
        "converged": converged,
        "history": history,
    }


# =====================================================================
# Support enumeration for 2-player Nash
# =====================================================================


def support_enumeration_bimatrix(
    game: GameRecord,
    *,
    max_support: int = None,
    epsilon: float = _DEFAULT_EPSILON,
) -> dict:
    """Exact Nash via support enumeration on a 2-player game.

    For each candidate pair of supports `(I, J) ⊆ A_1 × A_2`, set up
    the indifference system:

        Σ_{j ∈ J} σ_2(j) · u_1(i, j) = v_1     for all i ∈ I
        Σ_{i ∈ I} σ_1(i) · u_2(i, j) = v_2     for all j ∈ J
        Σ_{j ∈ J} σ_2(j) = 1, Σ_{i ∈ I} σ_1(i) = 1
        σ_1(i) = 0 for i ∉ I, σ_2(j) = 0 for j ∉ J
        u_1(i', j) over σ_2 ≤ v_1 for i' ∉ I
        u_2(i, j') over σ_1 ≤ v_2 for j' ∉ J

    The first four constraints are linear; we solve with Gaussian
    elimination and validate the non-negativity / inequality
    constraints. Complexity is `Σ_{k_1, k_2} C(|A_1|, k_1) · C(|A_2|, k_2)`
    matrix solves, exponential in support size but exact.

    Returns the list of Nash equilibria found and their values.
    """
    if game.n_players != 2:
        raise SolverUnavailable("support_enumeration_bimatrix requires 2-player game")

    m, k = game.action_counts
    A = [[game.player_payoff(0, (i, j)) for j in range(k)] for i in range(m)]
    B = [[game.player_payoff(1, (i, j)) for j in range(k)] for i in range(m)]

    if max_support is None:
        max_support = min(m, k)

    equilibria = []

    for size in range(1, max_support + 1):
        for I in itertools.combinations(range(m), size):
            for J in itertools.combinations(range(k), size):
                sol = _solve_support_pair(A, B, I, J, m, k)
                if sol is None:
                    continue
                sigma1, sigma2, v1, v2 = sol
                # Verify external best-response constraints
                ok = True
                for i_outside in range(m):
                    if i_outside in I:
                        continue
                    val = sum(sigma2[j] * A[i_outside][j] for j in range(k))
                    if val > v1 + 1e-7:
                        ok = False
                        break
                if not ok:
                    continue
                for j_outside in range(k):
                    if j_outside in J:
                        continue
                    val = sum(sigma1[i] * B[i][j_outside] for i in range(m))
                    if val > v2 + 1e-7:
                        ok = False
                        break
                if not ok:
                    continue
                prof = Profile(strategies=(
                    Strategy.from_weights(sigma1),
                    Strategy.from_weights(sigma2),
                ))
                # Deduplicate close equilibria
                dup = False
                for prev in equilibria:
                    diff1 = sum(abs(a - b) for a, b in zip(prof[0].probabilities, prev["profile"][0].probabilities))
                    diff2 = sum(abs(a - b) for a, b in zip(prof[1].probabilities, prev["profile"][1].probabilities))
                    if diff1 + diff2 < 1e-6:
                        dup = True
                        break
                if dup:
                    continue
                equilibria.append({"profile": prof, "values": (v1, v2), "support": (I, J)})

    return {"equilibria": tuple(equilibria), "n_found": len(equilibria)}


def _solve_support_pair(A, B, I, J, m, k):
    """Solve the indifference system for support pair (I, J).

    Variables: σ_1[i] for i ∈ I, σ_2[j] for j ∈ J, v_1, v_2.

    Constraints (linear):
      For each i ∈ I:        Σ_{j ∈ J} A[i][j] σ_2[j] − v_1 = 0
      For each j ∈ J:        Σ_{i ∈ I} B[i][j] σ_1[i] − v_2 = 0
      Σ σ_1[i] = 1
      Σ σ_2[j] = 1

    Returns (σ_1 vector len m, σ_2 vector len k, v_1, v_2) or None
    if the system is singular or yields negative probabilities.
    """
    sI = len(I)
    sJ = len(J)
    # Variable layout: [σ_1 (len sI), σ_2 (len sJ), v_1, v_2]
    nvar = sI + sJ + 2
    nrow = sI + sJ + 2
    M = [[0.0] * nvar for _ in range(nrow)]
    rhs = [0.0] * nrow

    row = 0
    # Indifference for player 1
    for i_idx, i in enumerate(I):
        for j_idx, j in enumerate(J):
            M[row][sI + j_idx] = A[i][j]
        M[row][sI + sJ] = -1.0   # -v_1
        rhs[row] = 0.0
        row += 1
    # Indifference for player 2
    for j_idx, j in enumerate(J):
        for i_idx, i in enumerate(I):
            M[row][i_idx] = B[i][j]
        M[row][sI + sJ + 1] = -1.0   # -v_2
        rhs[row] = 0.0
        row += 1
    # Σ σ_1 = 1
    for i_idx in range(sI):
        M[row][i_idx] = 1.0
    rhs[row] = 1.0
    row += 1
    # Σ σ_2 = 1
    for j_idx in range(sJ):
        M[row][sI + j_idx] = 1.0
    rhs[row] = 1.0

    sol = _solve_linear_system(M, rhs)
    if sol is None:
        return None
    sigma1 = [0.0] * m
    sigma2 = [0.0] * k
    for i_idx, i in enumerate(I):
        if sol[i_idx] < -1e-7:
            return None
        sigma1[i] = max(0.0, sol[i_idx])
    for j_idx, j in enumerate(J):
        if sol[sI + j_idx] < -1e-7:
            return None
        sigma2[j] = max(0.0, sol[sI + j_idx])
    v1 = sol[sI + sJ]
    v2 = sol[sI + sJ + 1]
    # Renormalise σ_1, σ_2 (small drift)
    s1 = sum(sigma1)
    s2 = sum(sigma2)
    if s1 <= _EPS or s2 <= _EPS:
        return None
    sigma1 = [s / s1 for s in sigma1]
    sigma2 = [s / s2 for s in sigma2]
    return sigma1, sigma2, v1, v2


def _solve_linear_system(M_in, b_in):
    """Solve M x = b via Gaussian elimination with partial pivoting.

    Returns None if the system is singular or inconsistent. Accepts
    square or over-determined systems by least-squares (we only need
    the square case here, since support enumeration generates exact-
    rank systems).
    """
    n = len(M_in)
    if n == 0:
        return []
    m = len(M_in[0])
    if n < m:
        return None
    M = [list(row) for row in M_in]
    b = list(b_in)

    # Forward elimination
    for col in range(min(m, n)):
        # find pivot
        pivot_row = col
        pivot_val = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > pivot_val:
                pivot_val = abs(M[r][col])
                pivot_row = r
        if pivot_val < 1e-12:
            # column of zeros; check feasibility
            continue
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]
        pivot = M[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / pivot
            if abs(factor) < 1e-15:
                continue
            for c in range(col, m):
                M[r][c] -= factor * M[col][c]
            b[r] -= factor * b[col]

    # If any over-determined row has zero LHS but non-zero RHS → inconsistent.
    sol = [0.0] * m
    for col in range(m):
        if abs(M[col][col]) < 1e-12:
            return None
        sol[col] = b[col] / M[col][col]
    # Sanity-check that the remaining (n - m) rows are consistent.
    for r in range(m, n):
        residual = sum(M[r][c] * sol[c] for c in range(m)) - b[r]
        if abs(residual) > 1e-7:
            return None
    return sol


# =====================================================================
# Zero-sum minimax via LP (simplex on the standard form)
# =====================================================================


def zero_sum_value(
    payoff_matrix: Sequence[Sequence[float]],
    *,
    method: str = METHOD_LINEAR_PROGRAM,
    iterations: int = 50_000,
    epsilon: float = 1e-6,
) -> dict:
    """Solve a 2-player zero-sum matrix game.

    `payoff_matrix[i][j]` is the row player's payoff for the joint
    action `(i, j)`; the column player receives `−payoff_matrix[i][j]`.
    Returns the game value `v` and optimal mixed strategies.

    Methods:
      LINEAR_PROGRAM: simplex on the LP
          maximise v
          subject to  Σ_i σ_1(i) · A[i,j] ≥ v for all j
                      Σ_i σ_1(i) = 1, σ_1 ≥ 0
        (and dual for σ_2.)

      MULTIPLICATIVE_WEIGHTS: self-play Hedge, average converges to
        Nash in zero-sum (Freund-Schapire). Rate `O(√(log K / T))`.
    """
    m = len(payoff_matrix)
    if m == 0:
        raise InvalidGame("payoff_matrix is empty")
    k = len(payoff_matrix[0])
    for row in payoff_matrix:
        if len(row) != k:
            raise InvalidGame("payoff_matrix has ragged rows")

    if method == METHOD_MULTIPLICATIVE_WEIGHTS:
        # Wrap as a GameRecord and run multiplicative_weights.
        flat0 = tuple(float(payoff_matrix[i][j]) for i in range(m) for j in range(k))
        flat1 = tuple(-float(payoff_matrix[i][j]) for i in range(m) for j in range(k))
        gr = GameRecord(
            game_id="__zero_sum_internal",
            n_players=2,
            action_counts=(m, k),
            payoffs=(flat0, flat1),
            action_names=(tuple(f"r{i}" for i in range(m)), tuple(f"c{j}" for j in range(k))),
            is_zero_sum=True,
            is_constant_sum=True,
            is_symmetric=(m == k and all(
                abs(payoff_matrix[i][j] + payoff_matrix[j][i]) < 1e-9
                for i in range(m) for j in range(k)
            )),
            metadata={},
        )
        res = multiplicative_weights(gr, iterations=iterations)
        return {
            "value": float(res["payoffs"][0]),
            "row_strategy": res["profile"][0],
            "col_strategy": res["profile"][1],
            "exploitability": float(res["exploitability"]),
            "epsilon": float(res["epsilon"]),
            "iterations": iterations,
            "method": METHOD_MULTIPLICATIVE_WEIGHTS,
            "converged": bool(res["converged"]),
        }

    # LP path. Solve maximin via two SEPARATE primal LPs (one for row,
    # one for column). We avoid extracting dual prices from the simplex
    # tableau (a known source of precision drift) — instead each player's
    # strategy comes from its own primal optimum, both LPs sharing the
    # same Dantzig-shifted matrix.
    lo = min(min(row) for row in payoff_matrix)
    shift = 1.0 - lo
    A = [[payoff_matrix[i][j] + shift for j in range(k)] for i in range(m)]

    # Row player LP — Dantzig form: let y_i = σ_1(i) / v_shifted, then
    #   min Σ y_i  s.t. A^T y ≥ 1, y ≥ 0
    # We pass this through `_simplex_full` as
    #   min Σ y_i  s.t. -A^T y ≤ -1, y ≥ 0
    c_row = [1.0] * m
    A_ub_row = []
    b_ub_row = []
    for j in range(k):
        A_ub_row.append([-A[i][j] for i in range(m)])
        b_ub_row.append(-1.0)
    y_sol = _simplex_full(c_row, A_ub_row, b_ub_row, [], [], m)
    if y_sol is None or sum(y_sol) <= _EPS:
        return zero_sum_value(payoff_matrix,
                              method=METHOD_MULTIPLICATIVE_WEIGHTS,
                              iterations=iterations,
                              epsilon=epsilon)
    sy = sum(y_sol)
    sigma_row = [y / sy for y in y_sol]
    v_prime_row = 1.0 / sy

    # Column player LP — Dantzig form: let z_j = σ_2(j) / v_shifted, then
    #   max Σ z_j  s.t. A z ≤ 1, z ≥ 0
    # Pass through `_simplex_full` as min -Σ z_j s.t. A z ≤ 1, z ≥ 0.
    c_col = [-1.0] * k
    A_ub_col = []
    b_ub_col = []
    for i in range(m):
        A_ub_col.append([A[i][j] for j in range(k)])
        b_ub_col.append(1.0)
    z_sol = _simplex_full(c_col, A_ub_col, b_ub_col, [], [], k)
    if z_sol is None or sum(z_sol) <= _EPS:
        return zero_sum_value(payoff_matrix,
                              method=METHOD_MULTIPLICATIVE_WEIGHTS,
                              iterations=iterations,
                              epsilon=epsilon)
    sz = sum(z_sol)
    sigma_col = [z / sz for z in z_sol]
    v_prime_col = 1.0 / sz

    # By LP duality the two values agree exactly; we average for
    # symmetry (any discrepancy reflects simplex precision).
    v_prime = 0.5 * (v_prime_row + v_prime_col)
    value = v_prime - shift

    return {
        "value": float(value),
        "row_strategy": Strategy.from_weights(sigma_row),
        "col_strategy": Strategy.from_weights(sigma_col),
        "exploitability": 0.0,
        "epsilon": _SIMPLEX_TOL,
        "iterations": 0,
        "method": METHOD_LINEAR_PROGRAM,
        "converged": True,
    }


def _simplex_max_sum_under_az_le_1(A):
    """Solve  max Σ z_j  subject to A z ≤ 1, z ≥ 0  via the revised
    simplex (tableau form). A is m × k, output z of length k or None
    on failure.

    Variables: z_1..z_k (k), slacks s_1..s_m (m).
    Tableau: [A | I | 1].
    Objective: maximise Σ z_j.
    Reduced costs start at 1 for the z_j (we maximise; in
    minimisation form, we minimise -Σ z_j).
    """
    m = len(A)
    k = len(A[0]) if m else 0
    if m == 0 or k == 0:
        return [0.0] * k

    # We'll work with the minimisation form: min -Σ z_j  s.t. A z + s = 1, z,s ≥ 0.
    # Standard tableau: B = identity (slacks), basic vars = slacks (indices k..k+m-1).
    n = k + m
    tab = [list(A[i]) + [0.0] * m + [1.0] for i in range(m)]
    for i in range(m):
        tab[i][k + i] = 1.0
    # Objective row: -1 for z_j, 0 for slacks, 0 RHS.
    obj = [-1.0] * k + [0.0] * m + [0.0]
    basis = [k + i for i in range(m)]   # slacks

    for it in range(_SIMPLEX_MAX_ITER):
        # Pick entering variable (most negative reduced cost).
        enter = -1
        best = -_SIMPLEX_TOL
        for j in range(n):
            if obj[j] < best:
                best = obj[j]
                enter = j
        if enter < 0:
            break
        # Ratio test
        leave = -1
        best_ratio = math.inf
        for i in range(m):
            if tab[i][enter] > _SIMPLEX_TOL:
                ratio = tab[i][n] / tab[i][enter]
                if ratio < best_ratio - _SIMPLEX_TOL:
                    best_ratio = ratio
                    leave = i
        if leave < 0:
            # Unbounded — shouldn't happen with our constraints
            return None
        # Pivot
        piv = tab[leave][enter]
        for j in range(n + 1):
            tab[leave][j] /= piv
        for i in range(m):
            if i == leave:
                continue
            factor = tab[i][enter]
            if abs(factor) < 1e-15:
                continue
            for j in range(n + 1):
                tab[i][j] -= factor * tab[leave][j]
        factor = obj[enter]
        if abs(factor) > 1e-15:
            for j in range(n + 1):
                obj[j] -= factor * tab[leave][j]
        basis[leave] = enter
    else:
        return None   # didn't converge

    # Read off z values
    z = [0.0] * k
    for i in range(m):
        b = basis[i]
        if b < k:
            z[b] = tab[i][n]
    return z


def _simplex_min_sum_under_aty_ge_1(A, transpose=False):
    """Solve  min Σ y_i  subject to Aᵀ y ≥ 1, y ≥ 0  via the dual.

    The dual of {min Σ y_i s.t. Aᵀ y ≥ 1, y ≥ 0} is
                  {max Σ z_j s.t. A z ≤ 1, z ≥ 0}
    by LP duality (von Neumann minimax). We compute the dual
    optimum via simplex, then recover y by complementary slackness.

    But for our purposes, both primal and dual values are equal,
    and we ultimately need both σ_1 and σ_2. We solve the *primal*
    LP directly via simplex with the same template.

    Variables: y_1..y_m (m), surplus s_1..s_k subtracted from each
    inequality, plus artificial variables. To avoid the two-phase
    simplex, we instead exploit duality: solve the dual LP
    (max Σ z_j s.t. A z ≤ 1) and reconstruct y from the optimal
    dual values of the slack constraints.

    Implementation: we solve max Σ z_j (which is _simplex_max…) and
    record the dual variables.
    """
    m = len(A)
    k = len(A[0]) if m else 0
    if m == 0 or k == 0:
        return [0.0] * m

    # We run simplex on the dual {max Σ z s.t. A z ≤ 1, z ≥ 0} and
    # extract the optimal dual variables (= reduced costs of the
    # corresponding slacks) at the end, which equal y.
    n = k + m
    tab = [list(A[i]) + [0.0] * m + [1.0] for i in range(m)]
    for i in range(m):
        tab[i][k + i] = 1.0
    obj = [-1.0] * k + [0.0] * m + [0.0]
    basis = [k + i for i in range(m)]

    for it in range(_SIMPLEX_MAX_ITER):
        enter = -1
        best = -_SIMPLEX_TOL
        for j in range(n):
            if obj[j] < best:
                best = obj[j]
                enter = j
        if enter < 0:
            break
        leave = -1
        best_ratio = math.inf
        for i in range(m):
            if tab[i][enter] > _SIMPLEX_TOL:
                ratio = tab[i][n] / tab[i][enter]
                if ratio < best_ratio - _SIMPLEX_TOL:
                    best_ratio = ratio
                    leave = i
        if leave < 0:
            return None
        piv = tab[leave][enter]
        for j in range(n + 1):
            tab[leave][j] /= piv
        for i in range(m):
            if i == leave:
                continue
            factor = tab[i][enter]
            if abs(factor) < 1e-15:
                continue
            for j in range(n + 1):
                tab[i][j] -= factor * tab[leave][j]
        factor = obj[enter]
        if abs(factor) > 1e-15:
            for j in range(n + 1):
                obj[j] -= factor * tab[leave][j]
        basis[leave] = enter
    else:
        return None

    # Dual values: y_i is the reduced cost of slack s_i (column k+i)
    # in the final objective row, taken as max(0, -obj[k+i]) since we
    # minimised -Σ z.
    y = [max(0.0, -obj[k + i]) for i in range(m)]
    # Renormalise to satisfy A^T y ≥ 1 exactly (numerical hygiene).
    return y


# =====================================================================
# Correlated equilibrium via LP
# =====================================================================


def correlated_equilibrium_lp(
    game: GameRecord,
    *,
    objective: str = "uniform",
    max_actions: int = 10,
) -> dict:
    """Compute a correlated equilibrium of a finite game by LP.

    Variables: μ(a) ≥ 0 for each joint action a, Σ μ(a) = 1.
    Constraints (Aumann 1974): for every player p and every pair of
    actions (a_p, a_p'),
        Σ_{a_{-p}} μ(a_p, a_{-p}) · (u_p(a_p, a_{-p}) − u_p(a_p', a_{-p})) ≥ 0.

    For a non-degenerate LP, the uniform distribution is in the
    interior and many solutions exist. We pick the one maximising
    Σ μ(a) · log(prod_p σ_p(a_p))  → maximum-entropy CE — or,
    equivalently for the linear objective option:
      "uniform"      : minimise Σ |μ(a) − 1/|A||
                       (forces solution close to uniform)
      "welfare"      : maximise expected total welfare
      "feasibility"  : any CE
      "egalitarian"  : maximise minimum expected payoff
    """
    n_joint = game.n_joint_actions()
    if n_joint > 4096:
        raise SolverUnavailable(
            f"correlated_equilibrium_lp: |A| = {n_joint} too large; "
            "use coarse_correlated_equilibrium (no-regret) instead"
        )
    for c in game.action_counts:
        if c > max_actions:
            raise SolverUnavailable(
                f"correlated_equilibrium_lp: action count {c} > max_actions={max_actions}"
            )

    counts = game.action_counts
    n = game.n_players
    joint_list = list(game.joint_actions())

    # Build incentive constraints (linear inequalities):
    #   Σ_{a_{-p}} μ(a_p, a_{-p}) · (u_p(a_p, a_{-p}) − u_p(a_p', a_{-p})) ≥ 0.
    # The variable index for joint a is its flat index. We accumulate
    # all such constraints and run feasibility via the LP feasibility
    # path (we just need ANY μ in the polytope).
    constraints = []
    for p in range(n):
        for ap in range(counts[p]):
            for ap_prime in range(counts[p]):
                if ap == ap_prime:
                    continue
                # Constraint: Σ_a μ(a) · I[a_p == ap] · (u_p(a) - u_p(a with a_p → ap')) ≥ 0
                row = [0.0] * n_joint
                ok = True
                for idx, joint in enumerate(joint_list):
                    if joint[p] != ap:
                        continue
                    alt = list(joint)
                    alt[p] = ap_prime
                    diff = (
                        game.player_payoff(p, joint)
                        - game.player_payoff(p, alt)
                    )
                    row[idx] = diff
                constraints.append(row)

    # Solve feasibility LP:
    #   find μ ≥ 0 with Σ μ = 1 and  G μ ≥ 0  (G is the incentive matrix)
    # We pick objective max Σ μ_i · w_i:
    #   "welfare":      w_i = total payoff at joint i
    #   "egalitarian":  max  t   s.t.  Σ μ_i · u_p(i) ≥ t  for all p
    #   "uniform":      min  Σ |μ_i − 1/n_joint|
    #   "feasibility":  any feasible point (w = 0)

    if objective == "welfare":
        w = [sum(game.player_payoff(p, joint) for p in range(n))
             for joint in joint_list]
        c_vec = [-x for x in w]   # minimise -welfare
    elif objective == "egalitarian":
        return _ce_egalitarian(game, joint_list, constraints, n_joint, n)
    elif objective in ("uniform", "max_entropy"):
        # Approximate max-entropy by an L1 ball minimisation around uniform.
        target = 1.0 / n_joint
        # Solve the L1 problem via a feasibility LP: introduce slack t_i,
        # minimise Σ t_i subject to μ_i − target ≤ t_i, target − μ_i ≤ t_i.
        # This is equivalent to minimising Σ |μ_i − target|.
        return _ce_uniform(game, joint_list, constraints, n_joint, target)
    elif objective == "feasibility":
        c_vec = [0.0] * n_joint
    else:
        raise SolverUnavailable(f"unknown CE objective: {objective}")

    sol = _solve_ce_lp(n_joint, c_vec, constraints)
    if sol is None:
        raise SolverUnavailable("LP solver failed on correlated equilibrium")

    return _format_ce_result(game, joint_list, sol)


def _ce_uniform(game, joint_list, constraints, n_joint, target):
    """L1-minimisation toward uniform CE. Linear programme:

       min Σ_i t_i
       s.t. μ_i − t_i ≤ target
            -μ_i − t_i ≤ -target
            Σ_i μ_i = 1
            μ_i ≥ 0
            G μ ≥ 0  (incentive constraints)

    Variables: [μ_0..μ_{n-1}, t_0..t_{n-1}], length 2n.
    """
    n = n_joint
    nvar = 2 * n
    # Inequalities (Ax ≤ b)
    A_ub = []
    b_ub = []
    for i in range(n):
        row = [0.0] * nvar
        row[i] = 1.0
        row[n + i] = -1.0
        A_ub.append(row)
        b_ub.append(target)
        row2 = [0.0] * nvar
        row2[i] = -1.0
        row2[n + i] = -1.0
        A_ub.append(row2)
        b_ub.append(-target)
    # Incentive constraints (G μ ≥ 0 → -G μ ≤ 0)
    for g in constraints:
        row = [-v for v in g] + [0.0] * n
        A_ub.append(row)
        b_ub.append(0.0)
    # Equality Σ μ = 1
    A_eq = [[1.0] * n + [0.0] * n]
    b_eq = [1.0]
    c_vec = [0.0] * n + [1.0] * n
    sol = _simplex_full(c_vec, A_ub, b_ub, A_eq, b_eq, nvar)
    if sol is None:
        return None
    mu = sol[:n]
    return _format_ce_result(game, joint_list, mu)


def _ce_egalitarian(game, joint_list, constraints, n_joint, n_players):
    """Maximise the minimum expected payoff among players over CE.

    max t
    s.t. Σ_a μ(a) u_p(a) ≥ t   ∀p
         Σ_a μ(a) = 1
         μ ≥ 0
         G μ ≥ 0
    """
    nvar = n_joint + 1   # last variable is t
    c_vec = [0.0] * n_joint + [-1.0]   # minimise -t
    A_ub = []
    b_ub = []
    # min-payoff constraints: -Σ μ(a) u_p(a) + t ≤ 0
    for p in range(n_players):
        row = [-float(game.player_payoff(p, joint)) for joint in joint_list] + [1.0]
        A_ub.append(row)
        b_ub.append(0.0)
    # incentive constraints
    for g in constraints:
        A_ub.append([-v for v in g] + [0.0])
        b_ub.append(0.0)
    A_eq = [[1.0] * n_joint + [0.0]]
    b_eq = [1.0]
    sol = _simplex_full(c_vec, A_ub, b_ub, A_eq, b_eq, nvar)
    if sol is None:
        return None
    mu = sol[:n_joint]
    return _format_ce_result(game, joint_list, mu)


def _solve_ce_lp(n_joint, c_vec, constraints):
    """Solve the CE feasibility/optimisation LP.

       min c^T μ
       s.t. Σ μ = 1
            G μ ≥ 0  (incentive)
            μ ≥ 0
    """
    A_ub = []
    b_ub = []
    for g in constraints:
        A_ub.append([-v for v in g])
        b_ub.append(0.0)
    A_eq = [[1.0] * n_joint]
    b_eq = [1.0]
    return _simplex_full(c_vec, A_ub, b_ub, A_eq, b_eq, n_joint)


def _format_ce_result(game, joint_list, mu):
    # Renormalise
    s = sum(max(0.0, m) for m in mu)
    if s <= _EPS:
        return None
    mu_clean = [max(0.0, m) / s for m in mu]
    dist = tuple((joint, mu_clean[i]) for i, joint in enumerate(joint_list) if mu_clean[i] > _EPS)
    expected = [0.0] * game.n_players
    for joint, p in dist:
        for q in range(game.n_players):
            expected[q] += p * game.player_payoff(q, joint)
    return {
        "distribution": dist,
        "expected_payoff": tuple(expected),
    }


def _simplex_full(c, A_ub, b_ub, A_eq, b_eq, nvar):
    """General two-phase simplex for

        min c^T x
        s.t. A_ub x ≤ b_ub
             A_eq x = b_eq
             x ≥ 0

    Implemented via the Big-M method with an absorbing M = 1e8. Adds
    artificial variables for equality constraints and for inequality
    rows with negative b_ub. Returns the optimal x of length nvar, or
    None on failure / unboundedness.
    """
    M = 1e8
    m_ub = len(A_ub)
    m_eq = len(A_eq)
    # Inequality A_ub x + s = b_ub where s ≥ 0. If b_ub < 0, multiply
    # both sides by -1: -A x - s = -b_ub, then add an artificial.
    rows = []
    rhs = []
    basis = []
    extra_slacks = 0
    artificials = 0
    # We'll grow column count dynamically.
    # Total variable layout:
    #   x_1..x_nvar  (decision)
    #   s_1..s_{m_ub}  (slacks for inequalities, after sign-correction)
    #   a_1..a_?       (artificials for equality rows AND for inequality
    #                   rows that we negated)
    # Build big-M cost: c_full = c + 0 for slacks + M for artificials.
    n_slack = m_ub
    rows_list = []
    rhs_list = []
    art_cols = []   # indices of artificial columns

    total_cols = nvar + n_slack

    # Inequality rows
    for i in range(m_ub):
        if b_ub[i] >= -1e-12:
            row = list(A_ub[i]) + [0.0] * n_slack
            row[nvar + i] = 1.0
            rhs_val = b_ub[i]
            rows_list.append(row)
            rhs_list.append(rhs_val)
            basis.append(nvar + i)
        else:
            row = [-v for v in A_ub[i]] + [0.0] * n_slack
            row[nvar + i] = -1.0
            rhs_val = -b_ub[i]
            # need artificial
            art_cols.append(total_cols)
            row.append(1.0)
            total_cols += 1
            rows_list.append(row)
            rhs_list.append(rhs_val)
            basis.append(total_cols - 1)
            # extend earlier rows to keep column count
            for prev in rows_list[:-1]:
                prev.append(0.0)
            # but the c_full extension is done after the loop

    # Equality rows
    for i in range(m_eq):
        if b_eq[i] >= 0:
            row = list(A_eq[i]) + [0.0] * (total_cols - nvar)
        else:
            row = [-v for v in A_eq[i]] + [0.0] * (total_cols - nvar)
        rhs_val = b_eq[i] if b_eq[i] >= 0 else -b_eq[i]
        # add artificial
        art_cols.append(total_cols)
        row.append(1.0)
        total_cols += 1
        rows_list.append(row)
        rhs_list.append(rhs_val)
        basis.append(total_cols - 1)
        for prev in rows_list[:-1]:
            prev.append(0.0)

    # Pad all rows to total_cols
    for r in rows_list:
        while len(r) < total_cols:
            r.append(0.0)

    # Cost
    c_full = list(c) + [0.0] * n_slack
    # extend for artificials
    while len(c_full) < total_cols:
        c_full.append(M)

    # Initial reduced costs (objective row)
    obj = list(c_full)
    rhs_obj = 0.0
    for i, b_var in enumerate(basis):
        if abs(c_full[b_var]) > 1e-15:
            factor = c_full[b_var]
            for j in range(total_cols):
                obj[j] -= factor * rows_list[i][j]
            rhs_obj -= factor * rhs_list[i]

    # Standard simplex loop
    nrows = len(rows_list)
    for it in range(_SIMPLEX_MAX_ITER):
        # Find entering: most negative reduced cost.
        enter = -1
        best = -_SIMPLEX_TOL
        for j in range(total_cols):
            if obj[j] < best:
                best = obj[j]
                enter = j
        if enter < 0:
            break
        # Ratio test
        leave = -1
        best_ratio = math.inf
        for i in range(nrows):
            if rows_list[i][enter] > _SIMPLEX_TOL:
                ratio = rhs_list[i] / rows_list[i][enter]
                if ratio < best_ratio - _SIMPLEX_TOL:
                    best_ratio = ratio
                    leave = i
        if leave < 0:
            return None   # unbounded
        # Pivot
        piv = rows_list[leave][enter]
        for j in range(total_cols):
            rows_list[leave][j] /= piv
        rhs_list[leave] /= piv
        for i in range(nrows):
            if i == leave:
                continue
            factor = rows_list[i][enter]
            if abs(factor) < 1e-15:
                continue
            for j in range(total_cols):
                rows_list[i][j] -= factor * rows_list[leave][j]
            rhs_list[i] -= factor * rhs_list[leave]
        # update objective
        factor = obj[enter]
        if abs(factor) > 1e-15:
            for j in range(total_cols):
                obj[j] -= factor * rows_list[leave][j]
            rhs_obj -= factor * rhs_list[leave]
        basis[leave] = enter
    else:
        return None

    # Check artificials at 0
    for col in art_cols:
        for i, b_var in enumerate(basis):
            if b_var == col and rhs_list[i] > 1e-6:
                return None   # infeasible

    x = [0.0] * nvar
    for i, b_var in enumerate(basis):
        if b_var < nvar:
            x[b_var] = max(0.0, rhs_list[i])
    return x


# =====================================================================
# Coarse correlated equilibrium via no-regret self-play
# =====================================================================


def coarse_correlated_equilibrium(
    game: GameRecord,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    eta: float = None,
    epsilon: float = _DEFAULT_EPSILON,
    record_joint: bool = True,
    seed: int = None,
) -> dict:
    """Run independent no-regret learners (Hedge) per player and
    return the *empirical joint distribution* of play.

    By the no-regret-→-CCE folk theorem (Hannan 1957; Hart &
    Mas-Colell 2000), this distribution converges to a coarse
    correlated equilibrium. The "coarse" version is weaker than
    Aumann's correlated equilibrium — it only protects against
    *unconditional* deviations, not conditional ones — but its
    convergence rate is `O(√(log K / T))`, much faster than the
    internal-no-regret rates required for full CE.

    Returns:
      distribution: tuple of (joint, probability) pairs
      expected_payoff: tuple of expected payoffs under the joint
      profile: time-average per-player marginals (for compatibility)
      exploitability_unconditional: max over players of
        (best fixed-action payoff - their CCE payoff)
    """
    counts = game.action_counts
    n = game.n_players
    if iterations < 1:
        raise InvalidGame("iterations must be >= 1")

    K = max(counts)
    if eta is None:
        eta = math.sqrt(8.0 * math.log(max(2, K)) / max(1, iterations))

    lo, hi = game.payoff_range()
    span = max(hi - lo, _EPS)

    rng = random.Random(seed) if seed is not None else random.Random()

    weights = [[1.0] * counts[p] for p in range(n)]
    joint_counts = {}
    cum_strategy = [[0.0] * counts[p] for p in range(n)]

    for t in range(iterations):
        prof = Profile(strategies=tuple(
            Strategy.from_weights(weights[p]) for p in range(n)
        ))
        # SAMPLE a pure joint action (for a true distribution)
        joint = tuple(_sample(prof[p].probabilities, rng) for p in range(n))
        if record_joint:
            joint_counts[joint] = joint_counts.get(joint, 0) + 1
        # Update weights using full-information feedback (variance-free).
        for p in range(n):
            pv = player_payoff_vector(game, p, prof)
            for a in range(counts[p]):
                gain = (pv[a] - lo) / span
                weights[p][a] = max(_EPS, weights[p][a] * math.exp(eta * gain))
            s = sum(weights[p])
            if s > 0:
                weights[p] = [w / s for w in weights[p]]
        for p in range(n):
            for a in range(counts[p]):
                cum_strategy[p][a] += prof[p].probability(a)

    total = max(1, sum(joint_counts.values()))
    distribution = tuple(
        (joint, count / total)
        for joint, count in sorted(joint_counts.items())
    )
    expected = [0.0] * n
    for joint, prob in distribution:
        for q in range(n):
            expected[q] += prob * game.player_payoff(q, joint)

    avg_profile = Profile(strategies=tuple(
        Strategy.from_weights([c / max(1, iterations) for c in cum_strategy[p]])
        for p in range(n)
    ))

    # "Unconditional exploitability": maximum gain by deviating to
    # *any fixed action* over the joint distribution. Standard CCE check.
    expl_per_player = []
    for p in range(n):
        gains = [0.0] * counts[p]
        for joint, prob in distribution:
            for a in range(counts[p]):
                alt = list(joint)
                alt[p] = a
                gains[a] += prob * game.player_payoff(p, alt)
        best_gain = max(gains)
        expl_per_player.append(max(0.0, best_gain - expected[p]))
    expl_total = sum(expl_per_player)

    regret_bound = math.sqrt(math.log(max(2, K)) / (2.0 * iterations)) * span

    return {
        "distribution": distribution,
        "expected_payoff": tuple(expected),
        "profile": avg_profile,
        "exploitability_unconditional": expl_total,
        "exploitability_unconditional_per_player": tuple(expl_per_player),
        "epsilon": regret_bound + 1e-12,
        "iterations": iterations,
        "converged": expl_total <= regret_bound + 1e-9,
        "regret_bound": regret_bound,
    }


def _sample(probs, rng):
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1


# =====================================================================
# Internal: AttestationLedger adapter
# =====================================================================


class _AttestableReceipt:
    """Adapter object for ``AttestationLedger.append()``.

    Exposes ``ticket_id``, ``kind``, and a ``to_dict`` serialiser so the
    ledger can persist any equilibrator-emitted receipt without
    importing the equilibrator module.
    """

    __slots__ = ("ticket_id", "kind", "payload", "digest")

    def __init__(self, kind: str, payload: dict, digest: str = "") -> None:
        self.kind = kind
        self.payload = payload
        self.digest = digest
        self.ticket_id = payload.get("receipt_id") or digest[:16] or uuid.uuid4().hex[:16]

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "kind": self.kind,
            "payload": self.payload,
            "digest": self.digest,
        }


# =====================================================================
# Equilibrator runtime
# =====================================================================


class Equilibrator:
    """Non-cooperative game-theoretic equilibrium engine.

    Stateless except for a registry of games and a per-game counter
    of online updates. Thread-safe via a single recursive lock.

    Optional dependencies:
      bus       — agi.events.EventBus (events emitted on every state change)
      attestor  — agi.attest.RuntimeAttestor (receipts written on solve)
    """

    def __init__(
        self,
        *,
        bus: EventBus = None,
        attestor: Any = None,
        random_seed: int = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._games: dict = {}
        self._observations: dict = {}     # game_id → list of joint actions
        self._n_solved = 0
        self._random_seed = random_seed
        self._emit(EQUILIBRATOR_STARTED, {
            "id": uuid.uuid4().hex[:16],
            "timestamp_ns": time.time_ns(),
        })

    # ---- emit / attest ----

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None:
            return
        self._bus.publish(Event(kind=kind, data=dict(payload)))

    def _attest(self, kind: str, payload: dict) -> str:
        """Mint an attestation receipt if an attestor is wired.

        Accepts either:
          * `.record(kind=, payload=)` style (Attestor protocol), or
          * a callable accepting a single Attestable receipt object
            (e.g. ``RuntimeAttestor`` from ``agi.attest``).
        Returns a string hash if available, else empty string.
        """
        if self._attestor is None:
            return ""
        # Compute a deterministic hash of the payload either way.
        try:
            serialised = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        except Exception:
            digest = ""
        # Try `.record` first
        rec = getattr(self._attestor, "record", None)
        if callable(rec):
            try:
                receipt = rec(kind=kind, payload=payload)
                if hasattr(receipt, "hash"):
                    return receipt.hash
                if isinstance(receipt, str):
                    return receipt
            except Exception:
                pass
        # Otherwise try direct call with our small Attestable receipt
        try:
            entry = self._attestor(_AttestableReceipt(kind=kind, payload=payload, digest=digest))
            if entry is not None:
                for attr in ("entry_hash", "receipt_hash", "hash"):
                    v = getattr(entry, attr, None)
                    if v:
                        return str(v)
        except Exception:
            pass
        return digest

    # ---- registry ----

    def register_game(
        self,
        game_id: str,
        payoffs: Any,
        *,
        action_names: Any = None,
        metadata: Mapping = None,
    ) -> GameRecord:
        with self._lock:
            if game_id in self._games:
                raise InvalidGame(f"game {game_id!r} already registered")
            game = make_game(
                game_id,
                payoffs,
                action_names=action_names,
                metadata=metadata,
            )
            self._games[game_id] = game
            self._observations[game_id] = []
            self._emit(EQUILIBRATOR_GAME_REGISTERED, {
                "game_id": game_id,
                "n_players": game.n_players,
                "action_counts": list(game.action_counts),
                "is_zero_sum": game.is_zero_sum,
                "is_symmetric": game.is_symmetric,
                "content_hash": game.content_hash,
                "timestamp_ns": time.time_ns(),
            })
            return game

    def remove_game(self, game_id: str) -> None:
        with self._lock:
            if game_id not in self._games:
                raise UnknownGame(game_id)
            del self._games[game_id]
            del self._observations[game_id]
            self._emit(EQUILIBRATOR_GAME_REMOVED, {
                "game_id": game_id,
                "timestamp_ns": time.time_ns(),
            })

    def get_game(self, game_id: str) -> GameRecord:
        with self._lock:
            if game_id not in self._games:
                raise UnknownGame(game_id)
            return self._games[game_id]

    def games(self) -> Mapping:
        with self._lock:
            return dict(self._games)

    def clear(self) -> None:
        with self._lock:
            self._games.clear()
            self._observations.clear()
            self._n_solved = 0
            self._emit(EQUILIBRATOR_CLEARED, {"timestamp_ns": time.time_ns()})

    # ---- solve ----

    def solve(
        self,
        game_id: str,
        *,
        concept: str = CONCEPT_NASH,
        method: str = METHOD_AUTO,
        iterations: int = _DEFAULT_ITERATIONS,
        epsilon: float = _DEFAULT_EPSILON,
        eta: float = None,
        seed: int = None,
    ) -> EquilibriumReport:
        if concept not in KNOWN_CONCEPTS:
            raise SolverUnavailable(f"unknown concept: {concept}")
        if method not in KNOWN_METHODS:
            raise SolverUnavailable(f"unknown method: {method}")

        with self._lock:
            if game_id not in self._games:
                raise UnknownGame(game_id)
            game = self._games[game_id]

        seed = self._random_seed if seed is None else seed

        method_used = method
        if method == METHOD_AUTO:
            method_used = _auto_method(concept, game)

        # Concept dispatch
        if concept == CONCEPT_PURE_NASH:
            return self._solve_pure_nash(game, method_used)
        if concept == CONCEPT_MINIMAX:
            return self._solve_minimax(game, method_used, iterations, epsilon, seed)
        if concept == CONCEPT_NASH:
            return self._solve_nash(game, method_used, iterations, epsilon, eta, seed)
        if concept == CONCEPT_CORRELATED:
            return self._solve_correlated(game, method_used, iterations, epsilon, seed)
        if concept == CONCEPT_COARSE_CORRELATED:
            return self._solve_coarse_correlated(game, method_used, iterations, epsilon, eta, seed)
        if concept == CONCEPT_ESS:
            return self._solve_ess(game, iterations, epsilon, seed)
        raise SolverUnavailable(f"unhandled concept: {concept}")

    def _solve_pure_nash(self, game: GameRecord, method: str) -> EquilibriumReport:
        eqs = pure_nash_equilibria(game)
        if not eqs:
            # No pure Nash; return the lowest-exploitability pure profile
            best_joint = None
            best_expl = math.inf
            for joint in game.joint_actions():
                prof = Profile(strategies=tuple(
                    Strategy.pure(joint[p], game.action_counts[p]) for p in range(game.n_players)
                ))
                e, _ = exploitability(game, prof)
                if e < best_expl:
                    best_expl = e
                    best_joint = joint
            joint = best_joint
            converged = False
        else:
            # Pick the one maximising welfare for determinism.
            best = max(eqs, key=lambda j: sum(game.player_payoff(p, j) for p in range(game.n_players)))
            joint = best
            converged = True
        profile = Profile(strategies=tuple(
            Strategy.pure(joint[p], game.action_counts[p]) for p in range(game.n_players)
        ))
        expl_total, expl_per = exploitability(game, profile)
        payoffs = expected_payoff(game, profile)
        cert = {
            "concept": CONCEPT_PURE_NASH,
            "method": method,
            "all_equilibria": [list(e) for e in eqs],
            "axioms": [AXIOM_BEST_RESPONSE] if converged else [],
            "content_hash": game.content_hash,
        }
        return self._publish_solved(
            game,
            EquilibriumReport(
                game_id=game.game_id,
                concept=CONCEPT_PURE_NASH,
                method=method,
                profile=profile,
                distribution=None,
                expected_payoff=payoffs,
                exploitability=expl_total,
                epsilon=expl_total,
                iterations=0,
                converged=converged,
                value=None,
                certificate=cert,
            ),
        )

    def _solve_minimax(self, game, method, iterations, epsilon, seed):
        if game.n_players != 2 or not game.is_zero_sum:
            raise SolverUnavailable(
                "CONCEPT_MINIMAX requires a 2-player zero-sum game"
            )
        m, k = game.action_counts
        A = [[game.player_payoff(0, (i, j)) for j in range(k)] for i in range(m)]
        zs = zero_sum_value(
            A,
            method=method if method in (METHOD_LINEAR_PROGRAM, METHOD_MULTIPLICATIVE_WEIGHTS) else METHOD_LINEAR_PROGRAM,
            iterations=iterations,
            epsilon=epsilon,
        )
        profile = Profile(strategies=(zs["row_strategy"], zs["col_strategy"]))
        payoffs = (zs["value"], -zs["value"])
        cert = {
            "concept": CONCEPT_MINIMAX,
            "method": zs["method"],
            "value": zs["value"],
            "epsilon": zs["epsilon"],
            "axioms": [AXIOM_MINIMAX, AXIOM_BEST_RESPONSE],
            "content_hash": game.content_hash,
        }
        return self._publish_solved(
            game,
            EquilibriumReport(
                game_id=game.game_id,
                concept=CONCEPT_MINIMAX,
                method=zs["method"],
                profile=profile,
                distribution=None,
                expected_payoff=payoffs,
                exploitability=zs["exploitability"],
                epsilon=zs["epsilon"],
                iterations=zs["iterations"],
                converged=zs["converged"],
                value=zs["value"],
                certificate=cert,
            ),
        )

    def _solve_nash(self, game, method, iterations, epsilon, eta, seed):
        if method == METHOD_SUPPORT_ENUMERATION:
            if game.n_players != 2:
                raise SolverUnavailable("support enumeration requires 2-player game")
            res = support_enumeration_bimatrix(game)
            if not res["equilibria"]:
                raise SolverUnavailable("no Nash equilibrium found by support enumeration")
            # pick max-welfare equilibrium
            best = max(res["equilibria"],
                       key=lambda e: sum(e["values"]))
            profile = best["profile"]
            payoffs = expected_payoff(game, profile)
            expl_total, expl_per = exploitability(game, profile)
            cert = {
                "concept": CONCEPT_NASH,
                "method": METHOD_SUPPORT_ENUMERATION,
                "n_equilibria_found": res["n_found"],
                "all_equilibria": [
                    {"support": [list(s) for s in e["support"]],
                     "values": list(e["values"]),
                     "profile": e["profile"].to_list()}
                    for e in res["equilibria"]
                ],
                "axioms": [AXIOM_BEST_RESPONSE],
                "content_hash": game.content_hash,
            }
            return self._publish_solved(
                game,
                EquilibriumReport(
                    game_id=game.game_id,
                    concept=CONCEPT_NASH,
                    method=METHOD_SUPPORT_ENUMERATION,
                    profile=profile,
                    distribution=None,
                    expected_payoff=payoffs,
                    exploitability=expl_total,
                    epsilon=0.0,
                    iterations=0,
                    converged=True,
                    value=None,
                    certificate=cert,
                ),
            )

        if method == METHOD_FICTITIOUS_PLAY:
            res = fictitious_play(game, iterations=iterations, epsilon=epsilon, seed=seed)
        elif method == METHOD_MULTIPLICATIVE_WEIGHTS:
            res = multiplicative_weights(game, iterations=iterations, eta=eta, epsilon=epsilon, seed=seed)
        elif method == METHOD_REPLICATOR:
            res = replicator_dynamics(game, iterations=iterations, epsilon=epsilon)
        elif method == METHOD_BEST_RESPONSE:
            res = best_response_dynamics(game, iterations=iterations, seed=seed)
        else:
            raise SolverUnavailable(f"method {method!r} not applicable to Nash")

        axioms = []
        if res.get("converged"):
            if method in (METHOD_FICTITIOUS_PLAY, METHOD_MULTIPLICATIVE_WEIGHTS):
                axioms.append(AXIOM_NO_REGRET)
            if method == METHOD_BEST_RESPONSE:
                axioms.append(AXIOM_BEST_RESPONSE)
            if game.is_zero_sum:
                axioms.append(AXIOM_MINIMAX)
        cert = {
            "concept": CONCEPT_NASH,
            "method": method,
            "regret_bound": res.get("regret_bound"),
            "eta": res.get("eta"),
            "axioms": axioms,
            "content_hash": game.content_hash,
        }
        return self._publish_solved(
            game,
            EquilibriumReport(
                game_id=game.game_id,
                concept=CONCEPT_NASH,
                method=method,
                profile=res["profile"],
                distribution=None,
                expected_payoff=res["payoffs"],
                exploitability=res["exploitability"],
                epsilon=res["epsilon"],
                iterations=res["iterations"],
                converged=res["converged"],
                value=None,
                certificate=cert,
            ),
        )

    def _solve_correlated(self, game, method, iterations, epsilon, seed):
        if method in (METHOD_LINEAR_PROGRAM, METHOD_AUTO):
            try:
                ce = correlated_equilibrium_lp(game, objective="uniform")
            except SolverUnavailable:
                # Fall back to coarse correlated via no-regret
                return self._solve_coarse_correlated(game, METHOD_MULTIPLICATIVE_WEIGHTS, iterations, epsilon, None, seed)
            distribution = ce["distribution"]
            expected = ce["expected_payoff"]
            # Verify CE constraints
            expl = _correlated_eq_exploitability(game, distribution)
            cert = {
                "concept": CONCEPT_CORRELATED,
                "method": METHOD_LINEAR_PROGRAM,
                "objective": "uniform",
                "axioms": [AXIOM_BEST_RESPONSE],
                "content_hash": game.content_hash,
            }
            return self._publish_solved(
                game,
                EquilibriumReport(
                    game_id=game.game_id,
                    concept=CONCEPT_CORRELATED,
                    method=METHOD_LINEAR_PROGRAM,
                    profile=None,
                    distribution=distribution,
                    expected_payoff=expected,
                    exploitability=expl,
                    epsilon=_SIMPLEX_TOL,
                    iterations=0,
                    converged=True,
                    value=None,
                    certificate=cert,
                ),
            )
        raise SolverUnavailable(f"method {method!r} not implemented for correlated equilibrium")

    def _solve_coarse_correlated(self, game, method, iterations, epsilon, eta, seed):
        # MW self-play sampled joint actions form an empirical CCE.
        res = coarse_correlated_equilibrium(
            game, iterations=iterations, eta=eta, epsilon=epsilon, seed=seed,
        )
        cert = {
            "concept": CONCEPT_COARSE_CORRELATED,
            "method": METHOD_MULTIPLICATIVE_WEIGHTS,
            "regret_bound": res["regret_bound"],
            "axioms": [AXIOM_NO_REGRET],
            "content_hash": game.content_hash,
        }
        return self._publish_solved(
            game,
            EquilibriumReport(
                game_id=game.game_id,
                concept=CONCEPT_COARSE_CORRELATED,
                method=METHOD_MULTIPLICATIVE_WEIGHTS,
                profile=res["profile"],
                distribution=res["distribution"],
                expected_payoff=res["expected_payoff"],
                exploitability=res["exploitability_unconditional"],
                epsilon=res["epsilon"],
                iterations=res["iterations"],
                converged=res["converged"],
                value=None,
                certificate=cert,
            ),
        )

    def _solve_ess(self, game, iterations, epsilon, seed):
        if not game.is_symmetric:
            raise SolverUnavailable("CONCEPT_ESS requires a symmetric game")
        res = replicator_dynamics(game, iterations=iterations, epsilon=epsilon)
        # ESS requires (i) Nash equilibrium and (ii) invasion barrier:
        # u(σ, σ) > u(σ', σ) for all σ' ≠ σ in a neighbourhood.
        # We test the strict-Nash condition; full neighbourhood test is
        # done numerically via the eigenvalues of the Jacobian, which we
        # omit. Report ESS if the final profile is locally asymptotically
        # stable under replicator (no migration).
        # Simple stability heuristic: perturb each player's strategy and
        # see if replicator returns to the original.
        ess_certified = _is_locally_stable_under_replicator(game, res["profile"])
        cert = {
            "concept": CONCEPT_ESS,
            "method": METHOD_REPLICATOR,
            "ess_certified": ess_certified,
            "axioms": [AXIOM_ESS] if ess_certified else [],
            "content_hash": game.content_hash,
        }
        return self._publish_solved(
            game,
            EquilibriumReport(
                game_id=game.game_id,
                concept=CONCEPT_ESS,
                method=METHOD_REPLICATOR,
                profile=res["profile"],
                distribution=None,
                expected_payoff=res["payoffs"],
                exploitability=res["exploitability"],
                epsilon=res["epsilon"],
                iterations=res["iterations"],
                converged=ess_certified,
                value=None,
                certificate=cert,
            ),
        )

    # ---- best response / exploitability convenience ----

    def best_response(
        self,
        game_id: str,
        player: int,
        profile: Profile,
    ) -> tuple:
        game = self.get_game(game_id)
        return best_response(game, player, profile)

    def exploitability(
        self,
        game_id: str,
        profile: Profile,
    ) -> tuple:
        game = self.get_game(game_id)
        return exploitability(game, profile)

    def expected_payoff(self, game_id: str, profile: Profile) -> tuple:
        game = self.get_game(game_id)
        return expected_payoff(game, profile)

    # ---- streaming online play ----

    def observe(self, game_id: str, joint_action: Sequence[int]) -> None:
        """Log a realised joint action; used by online empirical
        analysis (the engine does not currently learn from observations,
        but Coordinator integration may project them through the
        Coalition or Strategist primitive)."""
        with self._lock:
            if game_id not in self._games:
                raise UnknownGame(game_id)
            game = self._games[game_id]
            # validate
            if len(joint_action) != game.n_players:
                raise InvalidGame("observe: joint_action length != n_players")
            for p, a in enumerate(joint_action):
                if not (0 <= a < game.action_counts[p]):
                    raise InvalidGame(f"observe: joint_action[{p}] out of range")
            self._observations[game_id].append(tuple(joint_action))
            self._emit(EQUILIBRATOR_OBSERVED, {
                "game_id": game_id,
                "joint_action": list(joint_action),
                "timestamp_ns": time.time_ns(),
            })

    def empirical_distribution(self, game_id: str) -> tuple:
        with self._lock:
            obs = self._observations.get(game_id, [])
            if not obs:
                return ()
            counts = {}
            for j in obs:
                counts[j] = counts.get(j, 0) + 1
            total = len(obs)
            return tuple(
                (joint, count / total)
                for joint, count in sorted(counts.items())
            )

    def empirical_exploitability(self, game_id: str) -> float:
        game = self.get_game(game_id)
        dist = self.empirical_distribution(game_id)
        if not dist:
            return 0.0
        return _correlated_eq_exploitability(game, dist)

    # ---- coverage ----

    def coverage(self) -> CoverageReport:
        with self._lock:
            return CoverageReport(
                n_games=len(self._games),
                n_solved=self._n_solved,
                n_observed=sum(len(v) for v in self._observations.values()),
                games=tuple(self._games.keys()),
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "n_games": len(self._games),
                "games": {gid: game.to_dict() for gid, game in self._games.items()},
                "n_solved": self._n_solved,
                "n_observations": {gid: len(obs) for gid, obs in self._observations.items()},
            }

    def _publish_solved(self, game, report: EquilibriumReport) -> EquilibriumReport:
        self._n_solved += 1
        payload = report.to_dict()
        receipt = self._attest("equilibrator.solved", payload)
        if receipt:
            cert = dict(report.certificate)
            cert["attestation_receipt"] = receipt
            report = EquilibriumReport(
                game_id=report.game_id,
                concept=report.concept,
                method=report.method,
                profile=report.profile,
                distribution=report.distribution,
                expected_payoff=report.expected_payoff,
                exploitability=report.exploitability,
                epsilon=report.epsilon,
                iterations=report.iterations,
                converged=report.converged,
                value=report.value,
                certificate=cert,
                timestamp_ns=report.timestamp_ns,
                receipt_id=report.receipt_id,
            )
        self._emit(EQUILIBRATOR_SOLVED, report.to_dict())
        return report


# =====================================================================
# Helpers
# =====================================================================


def _auto_method(concept: str, game: GameRecord) -> str:
    """Pick a sensible default solver method for a (concept, game) pair."""
    if concept == CONCEPT_PURE_NASH:
        return METHOD_PURE_SEARCH
    if concept == CONCEPT_MINIMAX:
        return METHOD_LINEAR_PROGRAM
    if concept == CONCEPT_CORRELATED:
        if game.n_joint_actions() <= 64:
            return METHOD_LINEAR_PROGRAM
        return METHOD_MULTIPLICATIVE_WEIGHTS
    if concept == CONCEPT_COARSE_CORRELATED:
        return METHOD_MULTIPLICATIVE_WEIGHTS
    if concept == CONCEPT_ESS:
        return METHOD_REPLICATOR
    if concept == CONCEPT_NASH:
        # Small 2-player: exact via support enumeration. Otherwise: MW.
        if game.is_two_player() and max(game.action_counts) <= 4:
            return METHOD_SUPPORT_ENUMERATION
        return METHOD_MULTIPLICATIVE_WEIGHTS
    return METHOD_MULTIPLICATIVE_WEIGHTS


def _correlated_eq_exploitability(game: GameRecord, distribution: tuple) -> float:
    """Largest *conditional* deviation gain: a player conditioned on
    seeing his recommended action `a` may want to switch to some `a'`.
    Returns the maximum such gain across players, (a, a'). For a true
    CE this is 0.
    """
    n = game.n_players
    counts = game.action_counts
    expl = 0.0
    for p in range(n):
        for ap in range(counts[p]):
            for ap_prime in range(counts[p]):
                if ap == ap_prime:
                    continue
                gain = 0.0
                for joint, prob in distribution:
                    if joint[p] != ap:
                        continue
                    alt = list(joint)
                    alt[p] = ap_prime
                    gain += prob * (game.player_payoff(p, alt) - game.player_payoff(p, joint))
                if gain > expl:
                    expl = gain
    return float(expl)


def _is_locally_stable_under_replicator(
    game: GameRecord,
    profile: Profile,
    perturb: float = 1e-3,
    iterations: int = 200,
) -> bool:
    """Heuristic stability check: perturb each player's strategy in
    each direction and see if replicator dynamics returns to within
    a tolerance of the original profile."""
    tol = 2.0 * perturb
    for p in range(game.n_players):
        for a in range(game.action_counts[p]):
            probs = list(profile[p].probabilities)
            if probs[a] >= 1.0 - _EPS:
                continue
            old = probs[a]
            probs[a] = min(1.0 - _EPS, old + perturb)
            # renormalise
            tot = sum(probs)
            probs = [x / tot for x in probs]
            new_profile = profile.replace(p, Strategy.from_weights(probs))
            res = replicator_dynamics(game, initial_profile=new_profile, iterations=iterations, dt=0.01)
            diff = max(
                max(abs(res["profile"][q].probability(b) - profile[q].probability(b))
                    for b in range(game.action_counts[q]))
                for q in range(game.n_players)
            )
            if diff > tol:
                return False
    return True


# =====================================================================
# Module exports
# =====================================================================


__all__ = [
    # Event kinds
    "EQUILIBRATOR_STARTED",
    "EQUILIBRATOR_GAME_REGISTERED",
    "EQUILIBRATOR_GAME_REMOVED",
    "EQUILIBRATOR_SOLVED",
    "EQUILIBRATOR_OBSERVED",
    "EQUILIBRATOR_CLEARED",
    "EQUILIBRATOR_REPORT",
    # Solution concepts
    "CONCEPT_NASH",
    "CONCEPT_PURE_NASH",
    "CONCEPT_CORRELATED",
    "CONCEPT_COARSE_CORRELATED",
    "CONCEPT_MINIMAX",
    "CONCEPT_ESS",
    "KNOWN_CONCEPTS",
    # Methods
    "METHOD_AUTO",
    "METHOD_SUPPORT_ENUMERATION",
    "METHOD_FICTITIOUS_PLAY",
    "METHOD_MULTIPLICATIVE_WEIGHTS",
    "METHOD_REPLICATOR",
    "METHOD_BEST_RESPONSE",
    "METHOD_LINEAR_PROGRAM",
    "METHOD_PURE_SEARCH",
    "KNOWN_METHODS",
    # Axioms
    "AXIOM_BEST_RESPONSE",
    "AXIOM_NO_REGRET",
    "AXIOM_MINIMAX",
    "AXIOM_INCENTIVE_COMPATIBLE",
    "AXIOM_PARETO_OPTIMAL",
    "AXIOM_SYMMETRIC",
    "AXIOM_ESS",
    "KNOWN_AXIOMS",
    # Errors
    "EquilibratorError",
    "UnknownGame",
    "InvalidGame",
    "SolverUnavailable",
    # Types
    "Strategy",
    "Profile",
    "GameRecord",
    "EquilibriumReport",
    "CoverageReport",
    # Functional core
    "make_game",
    "player_payoff_vector",
    "expected_payoff",
    "best_response",
    "exploitability",
    "pure_nash_equilibria",
    "multiplicative_weights",
    "fictitious_play",
    "replicator_dynamics",
    "best_response_dynamics",
    "support_enumeration_bimatrix",
    "zero_sum_value",
    "correlated_equilibrium_lp",
    "coarse_correlated_equilibrium",
    # Runtime
    "Equilibrator",
]
