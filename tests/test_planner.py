"""Tests for ``agi.planner`` — SAT-compiled classical planner."""

from __future__ import annotations

import pytest

from agi.planner import (
    Action,
    GoalUnreachable,
    HorizonExhausted,
    InvalidAction,
    InvalidConfig,
    InvalidFluent,
    InvalidGoal,
    InvalidHorizon,
    InvalidState,
    PLANNER_ACTION_ADDED,
    PLANNER_FLUENT_ADDED,
    PLANNER_GOAL_SET,
    PLANNER_INITIAL_SET,
    PLANNER_KNOWN_EVENTS,
    PLANNER_SOLVED,
    PLANNER_STARTED,
    Plan,
    Planner,
    PlannerError,
    PlannerReport,
)


# --------------------------------------------------------------------- domain construction


def test_construct_planner() -> None:
    pl = Planner.create(seed=0)
    rep = pl.report()
    assert rep.num_fluents == 0
    assert rep.num_actions == 0
    assert rep.last_status is None


def test_private_constructor() -> None:
    with pytest.raises(PlannerError):
        Planner(object(), 0)  # type: ignore[arg-type]


def test_invalid_seed_rejected() -> None:
    with pytest.raises(InvalidConfig):
        Planner.create(seed=1.5)  # type: ignore[arg-type]


def test_fluent_validation() -> None:
    pl = Planner.create()
    with pytest.raises(InvalidFluent):
        pl.add_fluent("")
    with pytest.raises(InvalidFluent):
        pl.add_fluent(0)  # type: ignore[arg-type]
    pl.add_fluent("at_A")
    with pytest.raises(InvalidFluent):
        pl.add_fluent("at_A")  # duplicate


def test_action_references_undeclared_fluent() -> None:
    pl = Planner.create()
    pl.add_fluent("at_A")
    with pytest.raises(InvalidAction):
        pl.add_action("move", pre=["undeclared"], add=["at_A"])


def test_action_overlapping_effects() -> None:
    pl = Planner.create()
    pl.add_fluent("at_A")
    with pytest.raises(InvalidAction):
        pl.add_action("bad", add=["at_A"], delete=["at_A"])


def test_action_overlapping_pre() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    with pytest.raises(InvalidAction):
        pl.add_action("bad", pre=["p"], pre_neg=["p"])


def test_action_validation() -> None:
    pl = Planner.create()
    pl.add_fluent("at_A")
    with pytest.raises(InvalidAction):
        pl.add_action("", pre=["at_A"])
    pl.add_action("move", pre=["at_A"])
    with pytest.raises(InvalidAction):
        pl.add_action("move")  # duplicate
    with pytest.raises(InvalidAction):
        pl.add_action("neg_cost", cost=-1)


def test_initial_state_validation() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    with pytest.raises(InvalidState):
        pl.set_initial("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(InvalidState):
        pl.set_initial({"undeclared": True})


def test_goal_validation() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    with pytest.raises(InvalidGoal):
        pl.set_goal({})
    with pytest.raises(InvalidGoal):
        pl.set_goal({"undeclared": True})
    with pytest.raises(InvalidGoal):
        pl.set_goal("not a mapping")  # type: ignore[arg-type]


def test_clear_resets_domain() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.add_action("a", pre=["p"])
    pl.set_initial({"p": True})
    pl.clear()
    rep = pl.report()
    assert rep.num_fluents == 0
    assert rep.num_actions == 0
    assert rep.has_initial is False
    assert rep.has_goal is False


# --------------------------------------------------------------------- trivial plans


def test_one_step_plan() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("at_A")
    pl.add_fluent("at_B")
    pl.add_action("move_AB", pre=["at_A"], add=["at_B"], delete=["at_A"])
    pl.set_initial({"at_A": True, "at_B": False})
    pl.set_goal({"at_B": True})
    plan = pl.solve()
    assert plan.actions == ("move_AB",)
    assert plan.horizon == 1
    assert plan.cost == 1


def test_zero_step_plan_when_goal_holds() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    pl.set_goal({"p": True})
    plan = pl.solve()
    assert plan.horizon == 0
    assert plan.actions == ()


def test_multi_step_plan() -> None:
    pl = Planner.create(seed=0)
    for loc in "ABCD":
        pl.add_fluent(f"at_{loc}")
        pl.add_fluent(f"v_{loc}")
    for X in "ABCD":
        for Y in "ABCD":
            if X != Y:
                pl.add_action(
                    f"go_{X}_{Y}",
                    pre=[f"at_{X}"],
                    add=[f"at_{Y}", f"v_{Y}"],
                    delete=[f"at_{X}"],
                )
    pl.set_initial({"at_A": True, "v_A": True})
    pl.set_goal({"v_A": True, "v_B": True, "v_C": True, "v_D": True})
    plan = pl.solve()
    # Must visit B, C, D in some order, starting from A.
    assert plan.horizon == 3
    visited = {"A"}
    for a in plan.actions:
        _, _, Y = a.split("_")
        visited.add(Y)
    assert visited == {"A", "B", "C", "D"}


def test_solve_optimal_returns_minimum_horizon() -> None:
    pl = Planner.create(seed=0)
    for loc in "ABCD":
        pl.add_fluent(f"at_{loc}")
    for X in "ABCD":
        for Y in "ABCD":
            if X != Y:
                pl.add_action(
                    f"go_{X}_{Y}",
                    pre=[f"at_{X}"],
                    add=[f"at_{Y}"],
                    delete=[f"at_{X}"],
                )
    pl.set_initial({"at_A": True})
    pl.set_goal({"at_C": True})
    plan = pl.solve_optimal()
    # Optimal is 1 step (direct go_A_C).
    assert plan.horizon == 1


# --------------------------------------------------------------------- bounded


def test_solve_bounded_returns_none_when_unsat() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("a")
    pl.add_fluent("b")
    pl.add_action("a_to_b", pre=["a"], add=["b"], delete=["a"])
    pl.set_initial({"a": True})
    pl.set_goal({"b": True})
    plan_h0 = pl.solve_bounded(0)
    assert plan_h0 is None  # b not in initial


def test_solve_bounded_validation() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    pl.set_goal({"p": True})
    with pytest.raises(InvalidHorizon):
        pl.solve_bounded(-1)


def test_solve_bounded_horizon_zero_when_goal_holds() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    pl.set_goal({"p": True})
    plan = pl.solve_bounded(0)
    assert plan is not None
    assert plan.horizon == 0
    assert plan.actions == ()


# --------------------------------------------------------------------- reachability


def test_reachable_fluents() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.add_fluent("q")
    pl.add_fluent("r")
    pl.add_action("p_to_q", pre=["p"], add=["q"])
    pl.add_action("q_to_r", pre=["q"], add=["r"])
    pl.set_initial({"p": True})
    pl.set_goal({"r": True})
    reach = pl.reachable_fluents()
    assert reach == frozenset({"p", "q", "r"})


def test_unreachable_goal_raises() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.add_fluent("q")
    pl.add_fluent("r")
    # No action produces r.
    pl.add_action("p_to_q", pre=["p"], add=["q"])
    pl.set_initial({"p": True})
    pl.set_goal({"r": True})
    with pytest.raises(GoalUnreachable):
        pl.h_max()
    with pytest.raises(GoalUnreachable):
        pl.solve()


def test_h_max_increases_with_depth() -> None:
    pl = Planner.create()
    for i in range(5):
        pl.add_fluent(f"l_{i}")
    for i in range(4):
        pl.add_action(
            f"step_{i}", pre=[f"l_{i}"], add=[f"l_{i+1}"], delete=[f"l_{i}"]
        )
    pl.set_initial({"l_0": True})
    pl.set_goal({"l_4": True})
    assert pl.h_max() == 4  # 4 distinct action layers required


def test_h_max_initial_satisfies_goal() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    pl.set_goal({"p": True})
    assert pl.h_max() == 0


def test_relaxed_plan_returns_some_sequence() -> None:
    pl = Planner.create()
    pl.add_fluent("a")
    pl.add_fluent("b")
    pl.add_action("a_to_b", pre=["a"], add=["b"])
    pl.set_initial({"a": True})
    pl.set_goal({"b": True})
    rp = pl.relaxed_plan()
    assert "a_to_b" in rp


def test_h_max_validation() -> None:
    pl = Planner.create()
    with pytest.raises(InvalidState):
        pl.h_max()
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    with pytest.raises(InvalidGoal):
        pl.h_max()


def test_reachable_fluents_no_initial_raises() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    with pytest.raises(InvalidState):
        pl.reachable_fluents()


# --------------------------------------------------------------------- parallel planning


def test_parallel_plan_concurrent_actions() -> None:
    pl = Planner.create(seed=0)
    # Two non-interfering actions can co-fire under parallel mode.
    pl.add_fluent("a_done")
    pl.add_fluent("b_done")
    pl.add_action("do_a", add=["a_done"])
    pl.add_action("do_b", add=["b_done"])
    pl.set_initial({})
    pl.set_goal({"a_done": True, "b_done": True})
    plan = pl.solve(parallel=True)
    # Parallel mode lets both fire at step 0 → horizon 1
    assert plan.horizon == 1
    assert plan.parallel_steps == (("do_a", "do_b"),) or plan.parallel_steps == (
        ("do_b", "do_a"),
    )


def test_sequential_plan_forces_ordering() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("a_done")
    pl.add_fluent("b_done")
    pl.add_action("do_a", add=["a_done"])
    pl.add_action("do_b", add=["b_done"])
    pl.set_initial({})
    pl.set_goal({"a_done": True, "b_done": True})
    plan = pl.solve(parallel=False)
    # Sequential: horizon ≥ 2.
    assert plan.horizon == 2
    assert set(plan.actions) == {"do_a", "do_b"}


def test_parallel_interference_serializes() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("p")
    # Two actions interfere: one adds, one deletes p.
    pl.add_action("set_p", add=["p"])
    pl.add_action("clear_p", delete=["p"])
    pl.set_initial({"p": False})
    pl.set_goal({"p": True})
    plan = pl.solve(parallel=True)
    # interfere ⇒ cannot co-fire ⇒ but we only need set_p.
    assert "set_p" in plan.actions


# --------------------------------------------------------------------- pre_neg


def test_negative_preconditions() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("locked")
    pl.add_fluent("open")
    # Open the door iff it's not locked.
    pl.add_action("open_door", pre_neg=["locked"], add=["open"])
    pl.set_initial({"locked": False})
    pl.set_goal({"open": True})
    plan = pl.solve()
    assert plan.actions == ("open_door",)


def test_negative_precondition_blocks() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("locked")
    pl.add_fluent("open")
    pl.add_action("open_door", pre_neg=["locked"], add=["open"])
    pl.set_initial({"locked": True})
    pl.set_goal({"open": True})
    # No way to unset locked → goal unreachable in relaxed sense
    with pytest.raises(GoalUnreachable):
        pl.solve()


# --------------------------------------------------------------------- attestation


def test_attestation_ledger_chains() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("p")
    pl.add_action("do_p", add=["p"])
    pl.set_initial({})
    pl.set_goal({"p": True})
    pl.solve()
    head = pl.ledger_head()
    assert isinstance(head, str) and len(head) == 64
    events = [rec["event"] for rec in pl.ledger()]
    assert PLANNER_STARTED in events
    assert PLANNER_FLUENT_ADDED in events
    assert PLANNER_ACTION_ADDED in events
    assert PLANNER_INITIAL_SET in events
    assert PLANNER_GOAL_SET in events
    assert PLANNER_SOLVED in events
    for rec in pl.ledger():
        assert rec["event"] in PLANNER_KNOWN_EVENTS


def test_seed_determinism() -> None:
    def run(seed: int) -> Plan:
        pl = Planner.create(seed=seed)
        for loc in "ABCD":
            pl.add_fluent(f"at_{loc}")
        for X in "ABCD":
            for Y in "ABCD":
                if X != Y:
                    pl.add_action(
                        f"go_{X}_{Y}",
                        pre=[f"at_{X}"],
                        add=[f"at_{Y}"],
                        delete=[f"at_{X}"],
                    )
        pl.set_initial({"at_A": True})
        pl.set_goal({"at_D": True})
        return pl.solve()

    p1 = run(7)
    p2 = run(7)
    assert p1.actions == p2.actions


# --------------------------------------------------------------------- report


def test_report_fields() -> None:
    pl = Planner.create(seed=42)
    pl.add_fluent("p")
    pl.add_action("do_p", add=["p"])
    pl.set_initial({})
    pl.set_goal({"p": True})
    pl.solve()
    rep = pl.report()
    assert isinstance(rep, PlannerReport)
    assert rep.num_fluents == 1
    assert rep.num_actions == 1
    assert rep.has_initial
    assert rep.has_goal
    assert rep.last_status == "sat"
    assert rep.last_plan_horizon == 1
    assert rep.seed == 42
    assert rep.h_max_initial_goal == 1


def test_report_without_solve() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    pl.set_goal({"p": True})
    rep = pl.report()
    assert rep.last_status is None
    assert rep.last_plan_horizon is None


def test_h_max_cache_invalidates() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.add_fluent("q")
    pl.add_action("p_to_q", pre=["p"], add=["q"])
    pl.set_initial({"p": True})
    pl.set_goal({"q": True})
    h1 = pl.h_max()
    # Add another fluent and refresh goal.
    pl.add_fluent("r")
    pl.add_action("q_to_r", pre=["q"], add=["r"])
    pl.set_goal({"r": True})
    h2 = pl.h_max()
    assert h2 > h1


# --------------------------------------------------------------------- horizon exhaustion


def test_horizon_exhausted_when_no_plan_within_bound() -> None:
    pl = Planner.create(seed=0)
    # Construct a domain where the only way to reach the goal is a
    # long chain, but cap horizon below it.
    n = 5
    for i in range(n):
        pl.add_fluent(f"f_{i}")
    for i in range(n - 1):
        pl.add_action(
            f"step_{i}", pre=[f"f_{i}"], add=[f"f_{i+1}"], delete=[f"f_{i}"]
        )
    pl.set_initial({"f_0": True})
    pl.set_goal({f"f_{n-1}": True})
    with pytest.raises(HorizonExhausted):
        pl.solve(max_horizon=1)


def test_max_horizon_validation() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.set_initial({"p": True})
    pl.set_goal({"p": True})
    with pytest.raises(InvalidHorizon):
        pl.solve(max_horizon=-1)


# --------------------------------------------------------------------- action interference


def test_interference_diagnostic() -> None:
    pl = Planner.create()
    pl.add_fluent("p")
    pl.add_fluent("q")
    a = Action(name="A", add=frozenset({"p"}))
    b = Action(name="B", delete=frozenset({"p"}))
    c = Action(name="C", add=frozenset({"q"}))
    assert pl._interfere(a, b)  # add/delete conflict
    assert not pl._interfere(a, c)  # disjoint effects
    # Effect vs pre_neg
    d = Action(name="D", pre_neg=frozenset({"p"}))
    assert pl._interfere(a, d)


# --------------------------------------------------------------------- larger example


def test_blocks_world_swap() -> None:
    pl = Planner.create(seed=0)
    blocks = "ABC"
    fluents = []
    for b in blocks:
        fluents.append(f"on_{b}_table")
        fluents.append(f"clear_{b}")
    for x in blocks:
        for y in blocks:
            if x != y:
                fluents.append(f"on_{x}{y}")
    for f in fluents:
        pl.add_fluent(f)

    def put_from_table(X, Y):
        pl.add_action(
            f"put_{X}_on_{Y}",
            pre=[f"clear_{X}", f"on_{X}_table", f"clear_{Y}"],
            add=[f"on_{X}{Y}"],
            delete=[f"on_{X}_table", f"clear_{Y}"],
        )

    def move_block(X, Y, Z):
        pl.add_action(
            f"move_{X}_{Y}_to_{Z}",
            pre=[f"clear_{X}", f"on_{X}{Y}", f"clear_{Z}"],
            add=[f"on_{X}{Z}", f"clear_{Y}"],
            delete=[f"on_{X}{Y}", f"clear_{Z}"],
        )

    def unstack(X, Y):
        pl.add_action(
            f"unstack_{X}_{Y}",
            pre=[f"clear_{X}", f"on_{X}{Y}"],
            add=[f"on_{X}_table", f"clear_{Y}"],
            delete=[f"on_{X}{Y}"],
        )

    for X in blocks:
        for Y in blocks:
            if X != Y:
                put_from_table(X, Y)
                unstack(X, Y)
                for Z in blocks:
                    if Z != X and Z != Y:
                        move_block(X, Y, Z)

    # Stack: A on B, C on table; goal: C on A.  Requires multi-step
    # rearrangement because A is not clear.
    pl.set_initial(
        {"on_AB": True, "on_C_table": True, "on_B_table": True, "clear_A": True, "clear_C": True}
    )
    pl.set_goal({"on_CA": True})
    plan = pl.solve()
    # C is on table, A is clear, so we can just put C on A in 1 step.
    assert plan.horizon == 1
    assert "put_C_on_A" in plan.actions


def test_blocks_world_unstack_required() -> None:
    pl = Planner.create(seed=0)
    blocks = "ABC"
    fluents = []
    for b in blocks:
        fluents.append(f"on_{b}_table")
        fluents.append(f"clear_{b}")
    for x in blocks:
        for y in blocks:
            if x != y:
                fluents.append(f"on_{x}{y}")
    for f in fluents:
        pl.add_fluent(f)

    def put_from_table(X, Y):
        pl.add_action(
            f"put_{X}_on_{Y}",
            pre=[f"clear_{X}", f"on_{X}_table", f"clear_{Y}"],
            add=[f"on_{X}{Y}"],
            delete=[f"on_{X}_table", f"clear_{Y}"],
        )

    def unstack(X, Y):
        pl.add_action(
            f"unstack_{X}_{Y}",
            pre=[f"clear_{X}", f"on_{X}{Y}"],
            add=[f"on_{X}_table", f"clear_{Y}"],
            delete=[f"on_{X}{Y}"],
        )

    for X in blocks:
        for Y in blocks:
            if X != Y:
                put_from_table(X, Y)
                unstack(X, Y)

    # A on B, C on table.  Goal: B on A — needs to unstack first.
    pl.set_initial(
        {"on_AB": True, "on_C_table": True, "on_B_table": True, "clear_A": True, "clear_C": True}
    )
    pl.set_goal({"on_BA": True})
    plan = pl.solve()
    assert plan.horizon == 2
    assert plan.actions[0] == "unstack_A_B"
    assert plan.actions[1] == "put_B_on_A"


# --------------------------------------------------------------------- composition


def test_solver_certificate_in_plan_stats() -> None:
    pl = Planner.create(seed=0)
    pl.add_fluent("p")
    pl.add_action("do_p", add=["p"])
    pl.set_initial({})
    pl.set_goal({"p": True})
    plan = pl.solve()
    assert "conflicts" in plan.stats
    assert "decisions" in plan.stats
    assert plan.stats["horizon"] == 1
