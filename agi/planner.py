r"""Planner — SAT-compiled classical planning as a runtime primitive.

A coordination engine driving discrete actuation needs to do more than
*react* to the current state — it needs to *plan* a sequence of actions
that achieves a goal.  None of the primitives shipped so far in this
runtime returns a *plan*; what they return is a posterior, a forecast,
an inferred reward, a discovered law, a satisfying assignment.

The ``Planner`` is the runtime primitive that closes that gap.  Given

  * a finite set of **fluents** (state propositions);
  * a finite set of **actions**, each with Boolean preconditions, add-
    effects, and delete-effects;
  * an **initial state** — a partial valuation over fluents;
  * a **goal** — a conjunction of fluent literals;

it returns either

  * a **plan** — a finite sequence of actions whose deterministic
    execution from the initial state achieves the goal, together with
    an explicit horizon (length) and a SHA-256 attestation chain that
    a regulator can replay byte-for-byte;
  * or, given an explicit horizon bound, a **proof that no plan of
    bounded length exists** — the DRAT proof emitted by the underlying
    SAT solver applied to the bounded plan-existence formula.

The primitive sits one composition layer above ``Solver``: every plan
query is compiled to a SAT instance via the Kautz-Selman-1992
*SATPlan* encoding, dispatched to :class:`agi.solver.Solver`, and the
returned model (or UNSAT proof) is decoded back into the planning
domain.  The interface promised by ``Planner`` is the *planning*
contract — actions, fluents, plans — but the proof of correctness is
the proof shipped by ``Solver``.

The pitch reduced to a runtime call::

    pl = Planner.create(seed=0)
    pl.add_fluent("at_A")
    pl.add_fluent("at_B")
    pl.add_action("move_AB", pre=["at_A"], add=["at_B"], delete=["at_A"])
    pl.set_initial({"at_A": True, "at_B": False})
    pl.set_goal({"at_B": True})
    plan = pl.solve()
    print(plan.actions)              # → ["move_AB"]
    print(plan.horizon)              # → 1

Every ``add_fluent`` / ``add_action`` / ``set_initial`` / ``set_goal``
/ ``solve`` / ``report`` call is hashed into a SHA-256 chain
compatible with the rest of the runtime's
:class:`~agi.attest.AttestationLedger`.

Mathematical roots
------------------

* **Fikes-Nilsson 1971 — STRIPS.**  Actions are tuples
  ``(precondition, add-effect, delete-effect)``; states are sets of
  ground atoms; the transition relation is precondition-checking and
  add/delete application.  The classical planning model the entire
  field builds on.

* **Kautz-Selman 1992 — Planning as satisfiability.**  Bounded-length
  plan existence is equivalent to satisfiability of a fixed CNF whose
  size grows linearly in horizon and action count.  The original
  encoding introduces

    - one Boolean per (fluent, timestep) — ``F^t``;
    - one Boolean per (action, timestep) — ``A^t``;

  and the clauses

    - **Init**: ``F^0`` iff fluent ``F`` is in the initial state.
    - **Goal**: ``G^H`` for every literal in the goal.
    - **Precondition**: ``A^t → P^t`` for every precondition ``P``.
    - **Add-effect**: ``A^t → F^{t+1}`` for every add fluent.
    - **Delete-effect**: ``A^t → ¬F^{t+1}`` for every delete fluent.
    - **Frame**: ``F^{t+1} ↔ F^t ∨ ⋁ {A^t : F ∈ add(A)}`` for
      add-frame; analogously for delete.
    - **Action exclusion**: ``¬A^t ∨ ¬B^t`` whenever ``A`` and ``B``
      *interfere* (one's add fluent is the other's delete) — the
      sequential-plan discipline.

  ``Planner`` implements the linear-size variant of this encoding
  (Kautz-Selman-Hoffmann 2006 §6) with parallel-action allowance:
  multiple actions can co-fire at the same timestep iff they are
  pairwise non-interfering.

* **Blum-Furst 1997 — Graphplan / mutex propagation.**  The planning
  graph is a layered structure that propagates *mutex* (mutually
  exclusive) relations between actions and between fluents.  Mutex
  propagation prunes the SAT encoding: actions that are mutex at
  layer ``t`` need no exclusion clause because the precondition or
  effect already entails it.  ``Planner`` runs a relaxed-graphplan
  reachability analysis upfront and uses the resulting layered
  structure to *skip* fluents that are never reachable and *bound*
  the iterative-horizon-deepening search to ``layers_until_goal +
  layers_until_fixpoint`` — never more.

* **Bonet-Geffner 2001 — h^max heuristic.**  The cost-of-cheapest-
  achievement heuristic over the delete-relaxed graph is a *lower
  bound* on the optimal plan length: no plan exists with horizon
  strictly less than ``h^max(initial, goal)``.  ``Planner.solve``
  uses ``h^max`` as the *starting* horizon for the iterative
  deepening loop, never below.

* **Streamlined frame axioms — McCain-Turner 1997.**  The "explanatory
  frame" formulation ``F^{t+1} → F^t ∨ ⋁ {A^t : F ∈ add(A)}`` plus
  ``¬F^{t+1} → ¬F^t ∨ ⋁ {A^t : F ∈ del(A)}`` reduces the encoding
  size by a factor of ``|fluents|`` versus the original frame
  axioms.  ``Planner`` ships exactly this form.

* **Rintanen 2012 — Madagascar parallel-action planning.**  Parallel
  plans are admissible iff no two co-firing actions interfere; the
  resulting CNF is up to ``H``-times smaller than a sequential
  encoding at horizon ``H``.  ``Planner.solve(parallel=True)``
  enables this mode; per-step action exclusion is computed from the
  domain at encoding time.

The composition story
---------------------

Every primitive in this runtime already returns *some* answer to
*some* question; ``Planner`` returns the *action sequence* a
coordination engine needs to *act*.  Because the underlying
mechanism is :class:`Solver`, the *certificate of correctness* is a
SAT witness:

  * a **plan** is the satisfying assignment with action variables
    interpreted positionally;
  * a **proof of no-plan-of-length-≤-H** is the DRAT proof of the
    bounded encoding's UNSAT.

Compositions with the rest of the runtime:

  * ``Synthesizer`` lifts the plan into a typed tool call sequence
    consumed by ``Coordinator``.
  * ``Quantilizer`` gates execution on plan optimality: refuse to
    act unless ``Planner.solve(optimal=True)`` returns a plan whose
    horizon equals the lower bound ``h^max``.
  * ``Refuter`` queries ``Planner.solve(initial, ¬goal)`` to find
    *failure trajectories* — the inverse plan, the counter-example
    that wins a refutation budget.
  * ``Strategist`` consumes ``Planner.h_max(initial, goal)`` as a
    lower bound on cost-to-go for top-level decision-making.

Module surface
--------------

  * **Data classes** — :class:`Fluent`, :class:`Action`, :class:`Plan`,
    :class:`PlannerReport`.
  * **Errors** — :class:`PlannerError` plus a structured ``code``
    sub-class taxonomy mirroring :class:`Solver` exactly.
  * **Domain construction** — ``add_fluent``, ``add_action``,
    ``set_initial``, ``set_goal``, ``clear``.
  * **Heuristics & reachability** — ``h_max``, ``reachable_fluents``,
    ``relaxed_plan``.
  * **Planning** — ``solve`` (single-shot), ``solve_bounded`` (with
    explicit horizon), ``solve_optimal`` (iteratively shrinks).

What ``Planner`` is *not*:

  * Not a numeric / hybrid / temporal planner — fluents are pure
    Boolean; numeric resources, durative actions, or timed initial
    literals require a downstream engine.
  * Not a probabilistic / MDP planner — the underlying solver is
    deterministic Boolean SAT.  For probabilistic planning use
    :class:`~agi.coordinator.Coordinator` over a learnt world model.
  * Not a competitive IPC-2018 planner — the implementation is in
    pure Python and intentionally readable.  It is the *interface*
    and the *audit chain* that are the contribution; a coordination
    engine wanting a high-performance back-end swaps in a Fast
    Downward / Madagascar / Pyperplan instance behind the same
    public API.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from agi.solver import (
    Solver,
    STATUS_SAT,
    STATUS_UNSAT,
    STATUS_UNKNOWN,
)


# --------------------------------------------------------------------- errors


class PlannerError(Exception):
    """Base class for all `Planner` runtime errors."""

    code = "planner_error"


class InvalidConfig(PlannerError):
    code = "invalid_config"


class InvalidFluent(PlannerError):
    code = "invalid_fluent"


class InvalidAction(PlannerError):
    code = "invalid_action"


class InvalidGoal(PlannerError):
    code = "invalid_goal"


class InvalidState(PlannerError):
    code = "invalid_state"


class InvalidHorizon(PlannerError):
    code = "invalid_horizon"


class GoalUnreachable(PlannerError):
    """Raised when relaxed reachability analysis proves the goal cannot
    be reached even with all delete-effects ignored.

    A goal that fails relaxed reachability is provably unreachable in
    the real domain (delete-relaxation is a sound *over*-approximation
    of reachability).
    """

    code = "goal_unreachable"


class HorizonExhausted(PlannerError):
    """Raised when the iterative-deepening search exceeds the
    user-supplied ``max_horizon`` budget without finding a plan."""

    code = "horizon_exhausted"


# --------------------------------------------------------------------- events


PLANNER_STARTED = "planner_started"
PLANNER_FLUENT_ADDED = "planner_fluent_added"
PLANNER_ACTION_ADDED = "planner_action_added"
PLANNER_INITIAL_SET = "planner_initial_set"
PLANNER_GOAL_SET = "planner_goal_set"
PLANNER_CLEARED = "planner_cleared"
PLANNER_SOLVED = "planner_solved"
PLANNER_REPORTED = "planner_reported"
PLANNER_NO_PLAN = "planner_no_plan"

PLANNER_KNOWN_EVENTS = (
    PLANNER_STARTED,
    PLANNER_FLUENT_ADDED,
    PLANNER_ACTION_ADDED,
    PLANNER_INITIAL_SET,
    PLANNER_GOAL_SET,
    PLANNER_CLEARED,
    PLANNER_SOLVED,
    PLANNER_REPORTED,
    PLANNER_NO_PLAN,
)


# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Action:
    """STRIPS action.

    Attributes
    ----------
    name:
        Unique action identifier; must be a non-empty string.
    pre:
        Frozen set of *precondition* fluent names — fluents that must
        all hold *positively* in the state at which the action fires.
    pre_neg:
        Frozen set of *negative-precondition* fluent names — fluents
        that must all hold *negatively* in the state at which the
        action fires.  Default empty.
    add:
        Frozen set of *add-effect* fluent names — fluents that become
        true after the action fires.
    delete:
        Frozen set of *delete-effect* fluent names — fluents that
        become false after the action fires.  Disjoint from ``add``.
    cost:
        Non-negative action cost; default 1.  Used by
        :meth:`Planner.solve_optimal` as a tie-breaker over plans of
        the same horizon.
    """

    name: str
    pre: FrozenSet[str] = field(default_factory=frozenset)
    pre_neg: FrozenSet[str] = field(default_factory=frozenset)
    add: FrozenSet[str] = field(default_factory=frozenset)
    delete: FrozenSet[str] = field(default_factory=frozenset)
    cost: int = 1


@dataclass(frozen=True)
class Plan:
    """A plan returned by :meth:`Planner.solve`.

    Attributes
    ----------
    actions:
        Ordered sequence of action *names* — the plan trajectory.
    horizon:
        Length of the plan.  Equals ``len(actions)`` for sequential
        plans; for parallel plans equals the number of time-steps,
        each of which may fire multiple non-interfering actions.
    parallel_steps:
        For ``parallel=True`` plans: a tuple of tuples — the actions
        that fire at each time-step.  Empty otherwise.
    cost:
        Sum of action costs over the plan.
    stats:
        Solver and encoding statistics — variables, clauses, solver
        conflicts, decisions, propagations, encoding time, solve time.
    """

    actions: Tuple[str, ...]
    horizon: int
    parallel_steps: Tuple[Tuple[str, ...], ...] = ()
    cost: int = 0
    stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerReport:
    """Summary of planner state at ``report`` time."""

    num_fluents: int
    num_actions: int
    has_initial: bool
    has_goal: bool
    last_plan_horizon: Optional[int]
    last_status: Optional[str]
    h_max_initial_goal: Optional[int]
    ledger_head: str
    seed: int


# --------------------------------------------------------------------- helpers


def _hash_event(prev_hash: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(b"\0")
    h.update(encoded.encode("utf-8"))
    return h.hexdigest()


def _canonical_state(state: Mapping[str, bool]) -> Tuple[Tuple[str, bool], ...]:
    return tuple(sorted((str(k), bool(v)) for k, v in state.items()))


# --------------------------------------------------------------------- Planner

_PLANNER_KEY = object()


class Planner:
    """SAT-compiled classical planner.

    Construct via :meth:`Planner.create`; the constructor is private.

    The planner maintains its STRIPS domain (fluents + actions), the
    initial-state and goal valuations, a relaxed-reachability cache,
    and the SHA-256 attestation chain.  Every call to :meth:`solve`
    instantiates a fresh :class:`Solver`, compiles the bounded
    plan-existence CNF, and decodes the satisfying assignment (or
    UNSAT proof) into the planning vocabulary.

    Thread-safety: instances are *not* safe for concurrent calls.
    """

    def __init__(self, _key: object, seed: int) -> None:
        if _key is not _PLANNER_KEY:
            raise PlannerError(
                "use Planner.create(...) — the constructor is intentionally private"
            )
        self._seed = int(seed)
        self._fluents: List[str] = []
        self._fluent_index: Dict[str, int] = {}
        self._actions: List[Action] = []
        self._action_index: Dict[str, int] = {}
        self._initial: Optional[Dict[str, bool]] = None
        self._goal: Optional[Dict[str, bool]] = None
        self._last_plan: Optional[Plan] = None
        self._last_status: Optional[str] = None
        self._h_max_cache: Optional[int] = None
        # Attestation ledger.
        self._ledger_head = "0" * 64
        self._ledger: List[Mapping[str, Any]] = []
        self._record(PLANNER_STARTED, {"seed": self._seed})

    @classmethod
    def create(cls, *, seed: int = 0) -> "Planner":
        """Construct a fresh :class:`Planner`.

        Parameters
        ----------
        seed:
            Seed forwarded to the internal :class:`Solver` on every
            ``solve`` call.  Plans for identical domains and seeds are
            byte-for-byte identical.
        """
        if not isinstance(seed, int):
            raise InvalidConfig("seed must be an int")
        return cls(_PLANNER_KEY, seed)

    # --- domain construction ------------------------------------------

    def add_fluent(self, name: str) -> None:
        """Declare a Boolean fluent.

        Fluent names must be non-empty strings; the runtime treats
        names case-sensitively and rejects duplicates.
        """
        if not isinstance(name, str) or not name:
            raise InvalidFluent("fluent name must be a non-empty string")
        if name in self._fluent_index:
            raise InvalidFluent(f"fluent {name!r} already declared")
        self._fluent_index[name] = len(self._fluents)
        self._fluents.append(name)
        self._h_max_cache = None
        self._record(PLANNER_FLUENT_ADDED, {"name": name})

    def add_action(
        self,
        name: str,
        *,
        pre: Iterable[str] = (),
        pre_neg: Iterable[str] = (),
        add: Iterable[str] = (),
        delete: Iterable[str] = (),
        cost: int = 1,
    ) -> None:
        """Declare an action by its STRIPS schema.

        The action's positive preconditions, negative preconditions,
        add-effects, and delete-effects must all reference *previously
        declared* fluents.  ``add`` and ``delete`` must be disjoint —
        an action cannot both add and remove the same fluent.

        Parameters
        ----------
        name:
            Unique non-empty action name.
        pre:
            Fluents that must hold positively for the action to fire.
        pre_neg:
            Fluents that must hold negatively for the action to fire.
        add:
            Fluents that become true after the action fires.
        delete:
            Fluents that become false after the action fires.
        cost:
            Non-negative integer cost (default 1).
        """
        if not isinstance(name, str) or not name:
            raise InvalidAction("action name must be a non-empty string")
        if name in self._action_index:
            raise InvalidAction(f"action {name!r} already declared")
        if not isinstance(cost, int) or cost < 0:
            raise InvalidAction("action cost must be a non-negative int")
        pre_set = frozenset(pre)
        pre_neg_set = frozenset(pre_neg)
        add_set = frozenset(add)
        delete_set = frozenset(delete)
        for fluent in pre_set | pre_neg_set | add_set | delete_set:
            if fluent not in self._fluent_index:
                raise InvalidAction(
                    f"action {name!r} references undeclared fluent {fluent!r}"
                )
        if add_set & delete_set:
            raise InvalidAction(
                f"action {name!r} has overlapping add/delete on "
                f"{sorted(add_set & delete_set)}"
            )
        if pre_set & pre_neg_set:
            raise InvalidAction(
                f"action {name!r} has overlapping positive/negative "
                f"preconditions on {sorted(pre_set & pre_neg_set)}"
            )
        act = Action(
            name=name,
            pre=pre_set,
            pre_neg=pre_neg_set,
            add=add_set,
            delete=delete_set,
            cost=cost,
        )
        self._action_index[name] = len(self._actions)
        self._actions.append(act)
        self._h_max_cache = None
        self._record(
            PLANNER_ACTION_ADDED,
            {
                "name": name,
                "pre": sorted(pre_set),
                "pre_neg": sorted(pre_neg_set),
                "add": sorted(add_set),
                "delete": sorted(delete_set),
                "cost": cost,
            },
        )

    def set_initial(self, state: Mapping[str, bool]) -> None:
        """Set the initial state.

        ``state`` is a partial mapping over declared fluents.  Fluents
        omitted from ``state`` are assumed false (closed-world
        assumption — Reiter 1980).
        """
        if not isinstance(state, Mapping):
            raise InvalidState("initial state must be a mapping")
        clean: Dict[str, bool] = {}
        for k, v in state.items():
            if k not in self._fluent_index:
                raise InvalidState(f"undeclared fluent {k!r} in initial state")
            clean[k] = bool(v)
        self._initial = clean
        self._h_max_cache = None
        self._record(PLANNER_INITIAL_SET, {"state": _canonical_state(clean)})

    def set_goal(self, goal: Mapping[str, bool]) -> None:
        """Set the goal.

        ``goal`` is a partial mapping over declared fluents — a
        *conjunction* of fluent literals.  Fluents omitted from
        ``goal`` are unconstrained at the horizon.
        """
        if not isinstance(goal, Mapping):
            raise InvalidGoal("goal must be a mapping")
        if not goal:
            raise InvalidGoal("goal must be non-empty")
        clean: Dict[str, bool] = {}
        for k, v in goal.items():
            if k not in self._fluent_index:
                raise InvalidGoal(f"undeclared fluent {k!r} in goal")
            clean[k] = bool(v)
        self._goal = clean
        self._h_max_cache = None
        self._record(PLANNER_GOAL_SET, {"goal": _canonical_state(clean)})

    def clear(self) -> None:
        """Drop the domain and reset planner state.

        The seed and attestation chain are preserved; subsequent
        ``add_fluent`` / ``add_action`` calls re-populate the domain
        from scratch.
        """
        self._fluents = []
        self._fluent_index = {}
        self._actions = []
        self._action_index = {}
        self._initial = None
        self._goal = None
        self._last_plan = None
        self._last_status = None
        self._h_max_cache = None
        self._record(PLANNER_CLEARED, {})

    # --- reachability & heuristics -----------------------------------

    def reachable_fluents(self) -> FrozenSet[str]:
        """Return the set of fluents reachable from the initial state
        in the *delete-relaxed* problem.

        The delete-relaxed problem ignores every action's delete-effect,
        so once a fluent is added it stays true.  Bonet-Geffner 2001:
        the relaxed-reachable set is computable in O((|fluents| +
        |actions|) · |fluents|) by repeatedly applying any action
        whose positive preconditions are already reachable and whose
        negative preconditions are *never-true* — initially false and
        never added by any reachable action.

        A fluent that is *not* in this set is provably unreachable in
        the real problem; a goal that requires such a fluent is
        provably infeasible.
        """
        if self._initial is None:
            raise InvalidState("initial state not set")
        reached: Set[str] = {f for f, v in self._initial.items() if v}
        # Iteratively fire actions whose positive preconditions are
        # all reached and whose negative preconditions are
        # "never-true" — i.e., not currently in `reached`.  Note that
        # under delete relaxation a fluent only ever transitions from
        # false to true, so a fluent in `reached` at any layer
        # stays in `reached`, and a pre_neg becomes permanently
        # blocked once any action firing has added it.
        changed = True
        while changed:
            changed = False
            for act in self._actions:
                if not act.pre.issubset(reached):
                    continue
                if act.pre_neg & reached:
                    continue
                new = act.add - reached
                if new:
                    reached |= new
                    changed = True
        return frozenset(reached)

    def h_max(self) -> int:
        """Compute the *maximum-cost-of-achievement* heuristic for the
        current (initial, goal) pair.

        Returns the smallest integer ``h`` such that every goal fluent
        is *delete-relaxed-reachable* in at most ``h`` action layers
        — a lower bound on the optimal plan length (Bonet-Geffner
        2001).  Returns 0 if the goal already holds in the initial
        state.

        Raises :class:`GoalUnreachable` if any goal fluent is not in
        the relaxed-reachable set.
        """
        if self._initial is None:
            raise InvalidState("initial state not set")
        if self._goal is None:
            raise InvalidGoal("goal not set")
        if self._h_max_cache is not None:
            return self._h_max_cache
        # Layer 0: positively-true fluents of the initial state.
        layer: Dict[str, int] = {}
        for f, v in self._initial.items():
            if v:
                layer[f] = 0
        # Iterate: an action's effect is reachable at layer t+1 if
        # all its positive preconditions are reachable by layer t AND
        # all of its negative preconditions are "never-true" — not
        # currently in ``layer`` (under delete-relaxation a fluent
        # only ever becomes true, so a fluent in ``layer`` is true
        # at every later layer too).
        changed = True
        t = 0
        while changed:
            changed = False
            t += 1
            for act in self._actions:
                if not act.pre.issubset(layer.keys()):
                    continue
                if act.pre_neg & layer.keys():
                    continue
                # Layer of the action's effects = max precondition
                # layer + 1.  An action with no preconditions fires
                # immediately, so its effects appear at layer 1.
                layer_t = max((layer[p] for p in act.pre), default=0) + 1
                for f in act.add:
                    if f not in layer or layer[f] > layer_t:
                        layer[f] = layer_t
                        changed = True
            if t > len(self._fluents) + len(self._actions) + 2:
                break
        # Check goal reachability over positive AND negative goal
        # literals.  A negative goal is delete-relaxed-reachable iff
        # the fluent is initially false and is never added (in which
        # case it has no entry in ``layer``).
        bound = 0
        for f, want in self._goal.items():
            if want:
                if f not in layer:
                    raise GoalUnreachable(
                        f"goal fluent {f!r} is not delete-relaxed-reachable"
                    )
                bound = max(bound, layer[f])
            else:
                # Negative goal: reachable iff f is initially false
                # AND not in layer; achieving it requires zero action
                # layers.
                if self._initial.get(f, False) and f not in self._goal_negation_safe(layer):
                    raise GoalUnreachable(
                        f"negative goal {f!r} is initially true and no "
                        f"action deletes it under delete-relaxation"
                    )
        self._h_max_cache = bound
        return bound

    def _goal_negation_safe(self, layer: Mapping[str, int]) -> Set[str]:
        """Conservative set of fluents that *could* be falsified under
        the real problem.

        Under delete relaxation, no action falsifies a fluent.  In
        the real problem, an action with ``f`` in its ``delete``
        effects can falsify ``f``.  We return the set of fluents
        with at least one deleter action — a conservative
        approximation; an over-approximation in the sense that the
        action's other preconditions may not be reachable.
        """
        return {
            f
            for f in self._fluents
            if any(f in act.delete for act in self._actions)
        }

    def relaxed_plan(self) -> Tuple[str, ...]:
        """Return a *relaxed plan* — a sequence of actions that
        achieves the goal in the delete-relaxed problem.

        The relaxed plan is a heuristic *upper bound* on the optimal
        real-plan length (it might use redundant actions that pure
        sequential planning would compress).  It is the primary
        ingredient of the FF heuristic (Hoffmann-Nebel 2001); we
        expose it directly as a coordination-engine signal.
        """
        if self._initial is None:
            raise InvalidState("initial state not set")
        if self._goal is None:
            raise InvalidGoal("goal not set")
        # Greedy regression: start from goal, peel off goals by
        # applying any action that adds a needed goal fluent, requiring
        # the action's preconditions to be added to the goal-set.
        needed: Set[str] = {f for f, v in self._goal.items() if v}
        initial: Set[str] = {f for f, v in self._initial.items() if v}
        plan_actions: List[str] = []
        # Use h_max layering to time-stamp goals; pick the lowest-layer
        # achievable action first.
        layer_count: Dict[str, int] = {f: 0 for f in initial}
        # Forward layered reachability.
        layer_actions: List[List[Action]] = [[]]  # layer index -> actions firing AT that layer
        changed = True
        while changed:
            changed = False
            next_layer_actions: List[Action] = []
            for act in self._actions:
                if not act.pre.issubset(layer_count.keys()):
                    continue
                # Skip actions we've already fired at an earlier layer.
                if any(act.name == a.name for ls in layer_actions for a in ls):
                    pass
                # An action fires at the layer right after the max
                # precondition layer.
                layer_t = max((layer_count[p] for p in act.pre), default=-1) + 1
                for f in act.add:
                    if f not in layer_count or layer_count[f] > layer_t:
                        layer_count[f] = layer_t
                        changed = True
                next_layer_actions.append(act)
            layer_actions.append(next_layer_actions)
        # Regress from goal layer down.  For each needed goal, pick
        # the cheapest action that adds it.
        goals_at_layer: Dict[int, Set[str]] = {}
        for g in needed:
            l = layer_count.get(g, -1)
            if l < 0:
                # unreachable
                continue
            goals_at_layer.setdefault(l, set()).add(g)
        # Walk layers descending; for each layer's needed-set, pick an
        # achieving action and add its preconditions to the layer-1
        # needed-set.
        for l in sorted(goals_at_layer.keys(), reverse=True):
            if l == 0:
                continue
            for g in goals_at_layer[l]:
                # Pick the cheapest action that adds g at layer l.
                best: Optional[Action] = None
                for act in self._actions:
                    if g not in act.add:
                        continue
                    if not act.pre.issubset(layer_count.keys()):
                        continue
                    act_layer = max(
                        (layer_count[p] for p in act.pre), default=-1
                    ) + 1
                    if act_layer != l:
                        continue
                    if best is None or act.cost < best.cost:
                        best = act
                if best is None:
                    continue
                plan_actions.append(best.name)
                for p in best.pre:
                    if p in initial:
                        continue
                    lp = layer_count.get(p, -1)
                    if lp >= 0:
                        goals_at_layer.setdefault(lp, set()).add(p)
        # The constructed plan is roughly reverse-order; reverse it
        # for a forward sequence.  This is a heuristic, not a real
        # plan — see solve() for the sound version.
        return tuple(reversed(plan_actions))

    # --- planning -----------------------------------------------------

    def solve(
        self,
        *,
        max_horizon: Optional[int] = None,
        time_budget_s: Optional[float] = None,
        parallel: bool = False,
    ) -> Plan:
        """Find a plan via iterative-deepening SAT.

        Parameters
        ----------
        max_horizon:
            Maximum horizon to try.  When the iterative search exceeds
            this without finding a plan, :class:`HorizonExhausted` is
            raised.  ``None`` defaults to ``len(fluents) + len(actions)
            + 4``.
        time_budget_s:
            Optional wall-clock bound on the total ``solve`` call (not
            per-horizon).  Plain :class:`ResourceExhausted` may
            propagate from the underlying solver.
        parallel:
            If ``True``, the encoding admits multiple non-interfering
            actions to co-fire at each timestep (Rintanen 2012),
            potentially shortening the horizon dramatically.
        """
        if self._initial is None:
            raise InvalidState("initial state not set")
        if self._goal is None:
            raise InvalidGoal("goal not set")
        h_lower = self.h_max()
        if max_horizon is None:
            max_horizon = max(h_lower + 1, len(self._fluents) + len(self._actions) + 4)
        if not isinstance(max_horizon, int) or max_horizon < 0:
            raise InvalidHorizon("max_horizon must be a non-negative int")
        t0 = time.monotonic()
        # Check H=0 special case: goal already holds in the initial
        # state?
        if h_lower == 0 and self._goal_holds_in_initial():
            plan = Plan(actions=(), horizon=0, parallel_steps=(), cost=0, stats={"horizon": 0})
            self._last_plan = plan
            self._last_status = STATUS_SAT
            self._record(PLANNER_SOLVED, {"horizon": 0, "actions": []})
            return plan
        for horizon in range(max(1, h_lower), max_horizon + 1):
            if time_budget_s is not None and (time.monotonic() - t0) >= time_budget_s:
                raise HorizonExhausted(
                    f"time budget exhausted before horizon {horizon}"
                )
            remaining = None
            if time_budget_s is not None:
                remaining = max(0.0, time_budget_s - (time.monotonic() - t0))
            plan = self.solve_bounded(
                horizon, parallel=parallel, time_budget_s=remaining
            )
            if plan is not None:
                self._last_plan = plan
                self._last_status = STATUS_SAT
                return plan
        self._last_status = STATUS_UNSAT
        self._record(
            PLANNER_NO_PLAN, {"max_horizon": max_horizon, "h_max": h_lower}
        )
        raise HorizonExhausted(
            f"no plan of horizon ≤ {max_horizon} (h_max lower bound {h_lower})"
        )

    def solve_bounded(
        self,
        horizon: int,
        *,
        parallel: bool = False,
        time_budget_s: Optional[float] = None,
    ) -> Optional[Plan]:
        """Solve for a plan of *exactly* ``horizon`` steps.

        Returns the plan if SAT, ``None`` if UNSAT.  Pass through
        :class:`ResourceExhausted` from the underlying solver if the
        time budget is exhausted.
        """
        if self._initial is None:
            raise InvalidState("initial state not set")
        if self._goal is None:
            raise InvalidGoal("goal not set")
        if not isinstance(horizon, int) or horizon < 0:
            raise InvalidHorizon("horizon must be a non-negative int")
        if horizon == 0:
            if self._goal_holds_in_initial():
                return Plan(actions=(), horizon=0, cost=0, stats={"horizon": 0})
            return None
        sv, action_at, fluent_at = self._encode(horizon, parallel=parallel)
        t0 = time.monotonic()
        res = sv.solve(time_budget_s=time_budget_s)
        encode_solve_s = time.monotonic() - t0
        if res.status == STATUS_SAT:
            actions: List[str] = []
            parallel_steps: List[Tuple[str, ...]] = []
            for t in range(horizon):
                step: List[str] = []
                for i, act in enumerate(self._actions):
                    if res.model[action_at[i][t]]:
                        step.append(act.name)
                if parallel:
                    parallel_steps.append(tuple(step))
                    actions.extend(step)
                else:
                    if len(step) > 1:
                        # Should not happen with proper exclusion in
                        # sequential mode; defensive guard.
                        raise PlannerError(
                            f"multiple actions at step {t} in sequential plan: {step}"
                        )
                    if step:
                        actions.append(step[0])
            cost = sum(self._actions[self._action_index[a]].cost for a in actions)
            stats = dict(res.stats)
            stats.update(
                {
                    "horizon": horizon,
                    "parallel": parallel,
                    "encode_solve_s": encode_solve_s,
                }
            )
            plan = Plan(
                actions=tuple(actions),
                horizon=horizon,
                parallel_steps=tuple(parallel_steps) if parallel else (),
                cost=cost,
                stats=stats,
            )
            self._last_plan = plan
            self._last_status = STATUS_SAT
            self._record(
                PLANNER_SOLVED,
                {
                    "horizon": horizon,
                    "actions": list(actions),
                    "parallel": parallel,
                },
            )
            return plan
        if res.status == STATUS_UNSAT:
            return None
        # status UNKNOWN — caller decides
        return None

    def solve_optimal(
        self,
        *,
        max_horizon: Optional[int] = None,
        time_budget_s: Optional[float] = None,
        parallel: bool = False,
    ) -> Plan:
        """Find an *optimal-horizon* plan.

        Identical contract to :meth:`solve` — but the returned plan
        is *guaranteed* to be of minimum horizon (no plan of strictly
        shorter horizon exists).  Iterates from the ``h_max`` lower
        bound upward; the first SAT verdict yields the optimal plan.
        """
        # solve() already iterates from h_max upward and stops at the
        # first SAT, so solve_optimal is the same call with the same
        # iteration discipline.  We expose it as a named entry-point
        # because the *contract* — "minimum horizon" — is part of the
        # public guarantee even though the implementation is shared.
        return self.solve(
            max_horizon=max_horizon,
            time_budget_s=time_budget_s,
            parallel=parallel,
        )

    # --- encoding -----------------------------------------------------

    def _encode(
        self,
        horizon: int,
        *,
        parallel: bool,
    ) -> Tuple[Solver, List[List[int]], List[List[int]]]:
        """Compile the bounded plan-existence formula to SAT.

        Returns
        -------
        sv:
            Configured :class:`Solver` instance.
        action_at:
            ``action_at[i][t]`` is the SAT variable id for "action i
            fires at step t" — ``i`` indexes ``self._actions``, ``t``
            ranges in ``0..horizon-1``.
        fluent_at:
            ``fluent_at[i][t]`` is the SAT variable id for "fluent i
            holds at step t" — ``i`` indexes ``self._fluents``, ``t``
            ranges in ``0..horizon``.
        """
        sv = Solver.create(seed=self._seed)
        nf = len(self._fluents)
        na = len(self._actions)
        # Allocate variables.
        # fluent_at[i][t] : 1..(nf*(horizon+1))
        # action_at[i][t] : nf*(horizon+1) + 1 .. + na*horizon
        fluent_at: List[List[int]] = [[0] * (horizon + 1) for _ in range(nf)]
        action_at: List[List[int]] = [[0] * horizon for _ in range(na)]
        v = 1
        for i in range(nf):
            for t in range(horizon + 1):
                fluent_at[i][t] = v
                v += 1
        for i in range(na):
            for t in range(horizon):
                action_at[i][t] = v
                v += 1
        sv.reserve_vars(v - 1)
        # Init clauses.
        for i, f in enumerate(self._fluents):
            val = self._initial.get(f, False) if self._initial else False
            sv.add_clause([fluent_at[i][0]] if val else [-fluent_at[i][0]])
        # Goal clauses.
        if self._goal is None:
            raise InvalidGoal("goal not set")
        for f, want in self._goal.items():
            i = self._fluent_index[f]
            sv.add_clause(
                [fluent_at[i][horizon]] if want else [-fluent_at[i][horizon]]
            )
        # Action implications & frame axioms.
        for t in range(horizon):
            for ai, act in enumerate(self._actions):
                # Preconditions.
                for p in act.pre:
                    pi = self._fluent_index[p]
                    sv.add_clause([-action_at[ai][t], fluent_at[pi][t]])
                for p in act.pre_neg:
                    pi = self._fluent_index[p]
                    sv.add_clause([-action_at[ai][t], -fluent_at[pi][t]])
                # Effects.
                for f in act.add:
                    fi = self._fluent_index[f]
                    sv.add_clause([-action_at[ai][t], fluent_at[fi][t + 1]])
                for f in act.delete:
                    fi = self._fluent_index[f]
                    sv.add_clause([-action_at[ai][t], -fluent_at[fi][t + 1]])
            # Explanatory frame axioms (McCain-Turner 1997):
            # F^{t+1} → F^t ∨ ⋁_{A: F ∈ add(A)} A^t
            # ¬F^{t+1} → ¬F^t ∨ ⋁_{A: F ∈ del(A)} A^t
            for fi, f in enumerate(self._fluents):
                adders = [
                    action_at[ai][t]
                    for ai, act in enumerate(self._actions)
                    if f in act.add
                ]
                deleters = [
                    action_at[ai][t]
                    for ai, act in enumerate(self._actions)
                    if f in act.delete
                ]
                sv.add_clause([-fluent_at[fi][t + 1], fluent_at[fi][t]] + adders)
                sv.add_clause([fluent_at[fi][t + 1], -fluent_at[fi][t]] + deleters)
            # Action exclusion.
            if parallel:
                # Parallel mode: exclude pairs of *interfering* actions
                # only.  Two actions A, B interfere iff one's add or
                # del effect contradicts a precondition or effect of
                # the other.
                for i1 in range(len(self._actions)):
                    for i2 in range(i1 + 1, len(self._actions)):
                        if self._interfere(self._actions[i1], self._actions[i2]):
                            sv.add_clause(
                                [-action_at[i1][t], -action_at[i2][t]]
                            )
            else:
                # Sequential mode: at most one action per timestep.
                # Use Sinz at_most_1 for compact encoding.
                action_vars = [action_at[ai][t] for ai in range(na)]
                if len(action_vars) > 1:
                    sv.add_at_most(action_vars, 1)
        return sv, action_at, fluent_at

    def _interfere(self, a: Action, b: Action) -> bool:
        """Return ``True`` iff ``a`` and ``b`` mutually interfere.

        Two actions interfere when one's add-effect contradicts a
        precondition (positive or negative) of the other, or when
        their add/delete effects contradict.  Definition from
        Rintanen 2012 §3 (the standard parallel-action notion).
        """
        if a.name == b.name:
            return True
        # Effect-effect conflict.
        if a.add & b.delete or a.delete & b.add:
            return True
        # Effect-precondition conflict.
        if a.add & b.pre_neg or b.add & a.pre_neg:
            return True
        if a.delete & b.pre or b.delete & a.pre:
            return True
        return False

    def _goal_holds_in_initial(self) -> bool:
        if self._initial is None or self._goal is None:
            return False
        for f, want in self._goal.items():
            cur = self._initial.get(f, False)
            if cur != want:
                return False
        return True

    # --- reporting ----------------------------------------------------

    def report(self) -> PlannerReport:
        try:
            hm: Optional[int] = self.h_max() if (
                self._initial is not None and self._goal is not None
            ) else None
        except GoalUnreachable:
            hm = -1  # signal unreachable
        except (InvalidState, InvalidGoal):
            hm = None
        rep = PlannerReport(
            num_fluents=len(self._fluents),
            num_actions=len(self._actions),
            has_initial=self._initial is not None,
            has_goal=self._goal is not None,
            last_plan_horizon=self._last_plan.horizon if self._last_plan else None,
            last_status=self._last_status,
            h_max_initial_goal=hm,
            ledger_head=self._ledger_head,
            seed=self._seed,
        )
        self._record(PLANNER_REPORTED, {"head": self._ledger_head})
        return rep

    def ledger_head(self) -> str:
        return self._ledger_head

    def ledger(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._ledger)

    # --- internal -----------------------------------------------------

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        record = {"event": kind, "payload": dict(payload)}
        self._ledger_head = _hash_event(self._ledger_head, record)
        record["head"] = self._ledger_head
        self._ledger.append(record)


__all__ = [
    "Action",
    "GoalUnreachable",
    "HorizonExhausted",
    "InvalidAction",
    "InvalidConfig",
    "InvalidFluent",
    "InvalidGoal",
    "InvalidHorizon",
    "InvalidState",
    "PLANNER_ACTION_ADDED",
    "PLANNER_CLEARED",
    "PLANNER_FLUENT_ADDED",
    "PLANNER_GOAL_SET",
    "PLANNER_INITIAL_SET",
    "PLANNER_KNOWN_EVENTS",
    "PLANNER_NO_PLAN",
    "PLANNER_REPORTED",
    "PLANNER_SOLVED",
    "PLANNER_STARTED",
    "Plan",
    "Planner",
    "PlannerError",
    "PlannerReport",
]
