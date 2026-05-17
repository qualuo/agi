"""Planner demo — SAT-compiled classical planning.

Run with ``python -m examples.planner_demo``.  Demonstrates how the
``Planner`` primitive sits one composition layer above ``Solver`` and
returns a plan plus a SAT-derived certificate of correctness.
"""

from __future__ import annotations

from agi.planner import GoalUnreachable, HorizonExhausted, Planner


def demo_one_step() -> None:
    print("--- 1. One-step plan ---")
    pl = Planner.create(seed=0)
    pl.add_fluent("at_A")
    pl.add_fluent("at_B")
    pl.add_action("move_AB", pre=["at_A"], add=["at_B"], delete=["at_A"])
    pl.set_initial({"at_A": True, "at_B": False})
    pl.set_goal({"at_B": True})
    plan = pl.solve()
    print(f"  plan={plan.actions}  horizon={plan.horizon}  cost={plan.cost}")
    print(f"  h_max lower bound = {pl.h_max()}")


def demo_tour() -> None:
    print("\n--- 2. Multi-step tour ---")
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
    print(f"  plan={plan.actions}")
    print(f"  horizon={plan.horizon}  cost={plan.cost}")


def demo_blocks_unstack() -> None:
    print("\n--- 3. Blocks world: unstack-then-restack ---")
    pl = Planner.create(seed=0)
    blocks = "ABC"
    for b in blocks:
        pl.add_fluent(f"on_{b}_table")
        pl.add_fluent(f"clear_{b}")
    for x in blocks:
        for y in blocks:
            if x != y:
                pl.add_fluent(f"on_{x}{y}")

    def put_from_table(X: str, Y: str) -> None:
        pl.add_action(
            f"put_{X}_on_{Y}",
            pre=[f"clear_{X}", f"on_{X}_table", f"clear_{Y}"],
            add=[f"on_{X}{Y}"],
            delete=[f"on_{X}_table", f"clear_{Y}"],
        )

    def unstack(X: str, Y: str) -> None:
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

    pl.set_initial(
        {
            "on_AB": True,
            "on_C_table": True,
            "on_B_table": True,
            "clear_A": True,
            "clear_C": True,
        }
    )
    pl.set_goal({"on_BA": True})
    plan = pl.solve()
    print(f"  initial: A on B, C on table")
    print(f"  goal:    B on A")
    for i, a in enumerate(plan.actions, 1):
        print(f"   step {i}: {a}")


def demo_parallel() -> None:
    print("\n--- 4. Parallel-action mode ---")
    pl = Planner.create(seed=0)
    for n in "abcd":
        pl.add_fluent(f"{n}_done")
        pl.add_action(f"do_{n}", add=[f"{n}_done"])
    pl.set_initial({})
    pl.set_goal({f"{n}_done": True for n in "abcd"})
    seq = pl.solve(parallel=False)
    par = pl.solve(parallel=True)
    print(f"  sequential horizon: {seq.horizon}  parallel horizon: {par.horizon}")
    print(f"  parallel steps: {par.parallel_steps}")


def demo_unreachable_goal() -> None:
    print("\n--- 5. Provably-unreachable goal ---")
    pl = Planner.create(seed=0)
    pl.add_fluent("p")
    pl.add_fluent("q")
    pl.add_fluent("r")
    pl.add_action("p_to_q", pre=["p"], add=["q"])
    pl.set_initial({"p": True})
    pl.set_goal({"r": True})
    try:
        pl.h_max()
    except GoalUnreachable as e:
        print(f"  caught: {e}")


def demo_horizon_exhausted() -> None:
    print("\n--- 6. Bounded-horizon UNSAT ---")
    pl = Planner.create(seed=0)
    n = 5
    for i in range(n):
        pl.add_fluent(f"f_{i}")
    for i in range(n - 1):
        pl.add_action(
            f"step_{i}", pre=[f"f_{i}"], add=[f"f_{i+1}"], delete=[f"f_{i}"]
        )
    pl.set_initial({"f_0": True})
    pl.set_goal({f"f_{n-1}": True})
    plan_short = pl.solve_bounded(2)
    print(f"  horizon=2 plan? {plan_short}")  # UNSAT
    plan_long = pl.solve_bounded(4)
    print(f"  horizon=4 plan? {plan_long.actions}  (horizon={plan_long.horizon})")


def demo_attestation() -> None:
    print("\n--- 7. Attestation ledger ---")
    pl = Planner.create(seed=42)
    pl.add_fluent("p")
    pl.add_action("do_p", add=["p"])
    pl.set_initial({})
    pl.set_goal({"p": True})
    pl.solve()
    rep = pl.report()
    print(f"  num_fluents={rep.num_fluents} num_actions={rep.num_actions}")
    print(f"  h_max={rep.h_max_initial_goal}  last_horizon={rep.last_plan_horizon}")
    print(f"  ledger head = {rep.ledger_head}")
    print(f"  ledger length = {len(pl.ledger())} events")


def main() -> None:
    demo_one_step()
    demo_tour()
    demo_blocks_unstack()
    demo_parallel()
    demo_unreachable_goal()
    demo_horizon_exhausted()
    demo_attestation()


if __name__ == "__main__":
    main()
