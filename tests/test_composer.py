"""Tests for the Composer runtime primitive."""
from __future__ import annotations

import math
import pytest

from agi.composer import (
    ASTAR,
    BUDGET_EXHAUSTED,
    Certificate,
    Composer,
    ComposerError,
    ComposerReport,
    DIJKSTRA,
    ExecutionFailure,
    H_GOAL_COUNT,
    H_LANDMARK,
    H_ZERO,
    IDA_STAR,
    ILL_TYPED,
    INDEPENDENT,
    INFEASIBLE,
    InvalidGoal,
    InvalidOperator,
    InvalidPredicate,
    KNOWN_ALGORITHMS,
    KNOWN_HEURISTICS,
    KNOWN_REGIMES,
    Operator,
    Outcome,
    Plan,
    PlanStep,
    PlanningFailure,
    Predicate,
    REGRESSION,
    SOLVED,
    TypeCon,
    TypeVar,
    UnificationError,
    UnknownAlgorithm,
    UnknownHeuristic,
    UnknownOperator,
    UnknownRegime,
    WORST_CASE,
    apply_subst,
    clopper_pearson_lower,
    clopper_pearson_upper,
    composer_from_spec,
    empirical_bernstein_lower,
    free_vars,
    fresh_renaming,
    hoeffding_lower,
    hoeffding_upper,
    kl_bernoulli,
    kl_bernoulli_lower_inverse,
    kl_bernoulli_upper_inverse,
    pac_bayes_catoni,
    parse_predicate,
    parse_type,
    strongly_connected_components,
    topological_sort,
    unify,
)


# =====================================================================
# Type system
# =====================================================================


def test_parse_type_base():
    t = parse_type("Int")
    assert isinstance(t, TypeCon)
    assert t.name == "Int"
    assert t.args == ()


def test_parse_type_variable():
    t = parse_type("?a")
    assert isinstance(t, TypeVar)
    assert t.name == "a"


def test_parse_type_parameterised():
    t = parse_type("List<Int>")
    assert isinstance(t, TypeCon)
    assert t.name == "List"
    assert len(t.args) == 1
    assert t.args[0].name == "Int"


def test_parse_type_nested():
    t = parse_type("Map<Str, List<Int>>")
    assert t.name == "Map"
    assert len(t.args) == 2
    assert t.args[1].name == "List"
    assert t.args[1].args[0].name == "Int"


def test_parse_type_idempotent():
    t = parse_type("List<?a>")
    t2 = parse_type(t)
    assert t == t2


def test_parse_type_malformed():
    from agi.composer import TypeError_
    with pytest.raises(TypeError_):
        parse_type("")
    with pytest.raises(TypeError_):
        parse_type("List<Int")
    with pytest.raises(TypeError_):
        parse_type("?")


def test_unify_constants():
    s = unify(parse_type("Int"), parse_type("Int"))
    assert s == {}


def test_unify_variable():
    s = unify(parse_type("?a"), parse_type("Int"))
    assert apply_subst(parse_type("?a"), s) == parse_type("Int")


def test_unify_conflict():
    with pytest.raises(UnificationError):
        unify(parse_type("Int"), parse_type("Str"))
    with pytest.raises(UnificationError):
        unify(parse_type("List<Int>"), parse_type("List<Str>"))


def test_unify_occurs_check():
    with pytest.raises(UnificationError):
        unify(parse_type("?a"), parse_type("List<?a>"))


def test_unify_compound():
    s = unify(parse_type("List<?a>"), parse_type("List<Int>"))
    assert apply_subst(parse_type("?a"), s) == parse_type("Int")


def test_free_vars():
    assert free_vars(parse_type("Int")) == set()
    assert free_vars(parse_type("?a")) == {"a"}
    assert free_vars(parse_type("Map<?a, List<?b>>")) == {"a", "b"}


def test_fresh_renaming():
    counter = [0]
    t1 = fresh_renaming(parse_type("List<?a>"), counter)
    t2 = fresh_renaming(parse_type("List<?a>"), counter)
    assert free_vars(t1) != free_vars(t2)


# =====================================================================
# Predicates
# =====================================================================


def test_parse_predicate_zero_args():
    p = parse_predicate("Ready")
    assert p.name == "Ready"
    assert p.args == ()
    assert p.is_ground


def test_parse_predicate_with_args():
    p = parse_predicate("on(a, b)")
    assert p.name == "on"
    assert p.args == ("a", "b")
    assert p.is_ground


def test_parse_predicate_with_variable():
    p = parse_predicate("on(?x, b)")
    assert p.args == ("?x", "b")
    assert not p.is_ground
    assert p.variables() == ["x"]


def test_parse_predicate_numeric_args():
    p = parse_predicate("eq(1, 2)")
    assert p.args == (1, 2)


def test_predicate_substitute():
    p = parse_predicate("on(?x, b)")
    q = p.substitute({"x": "a"})
    assert q.args == ("a", "b")
    assert q.is_ground


def test_invalid_predicate():
    with pytest.raises(InvalidPredicate):
        parse_predicate("on(a, b")


# =====================================================================
# Operator registration
# =====================================================================


def test_register_operator_basic():
    c = Composer()
    op = c.register_operator(
        "noop",
        params=[("x", "Int")],
        pre=[],
        add=[],
        cost=0.0,
    )
    assert op.name == "noop"
    assert "noop" in c.operators()


def test_register_operator_typed_param_string():
    c = Composer()
    op = c.register_operator(
        "trim",
        params=["s:Str"],
        pre=["raw(?s)"],
        add=["trimmed(?s)"],
    )
    assert op.parameter_names() == ["s"]


def test_register_operator_duplicate_raises():
    c = Composer()
    c.register_operator("foo")
    with pytest.raises(InvalidOperator):
        c.register_operator("foo")


def test_register_operator_undeclared_variable():
    c = Composer()
    with pytest.raises(InvalidOperator):
        c.register_operator(
            "bad",
            params=[("x", "Int")],
            pre=["use(?y)"],   # ?y not declared
            add=[],
        )


def test_register_operator_negative_cost():
    c = Composer()
    with pytest.raises(InvalidOperator):
        c.register_operator("bad", cost=-1.0)


def test_register_operator_with_reliability_and_strength():
    c = Composer()
    op = c.register_operator("r", reliability=0.9, prior_strength=10.0)
    assert math.isclose(op.alpha + op.beta, 10.0, abs_tol=1e-6)
    assert math.isclose(op.reliability_mean(), 0.9, abs_tol=1e-6)


def test_register_operator_with_alpha_beta_direct():
    c = Composer()
    op = c.register_operator("r", alpha=4.0, beta=1.0)
    assert op.alpha == 4.0 and op.beta == 1.0


def test_register_operator_mixed_priors_raises():
    c = Composer()
    with pytest.raises(InvalidOperator):
        c.register_operator("r", reliability=0.9, alpha=4.0)


def test_register_operator_param_dup_raises():
    c = Composer()
    with pytest.raises(InvalidOperator):
        c.register_operator("d", params=[("x", "Int"), ("x", "Str")])


# =====================================================================
# Axiom store
# =====================================================================


def test_add_axiom():
    c = Composer()
    c.add_axiom("ready(server1)")
    assert any(
        a.name == "ready" and a.args == ("server1",) for a in c.axioms()
    )


def test_add_axiom_with_variable_raises():
    c = Composer()
    with pytest.raises(InvalidPredicate):
        c.add_axiom("ready(?x)")


# =====================================================================
# Planning: A*
# =====================================================================


def _basic_pipeline(c: Composer) -> None:
    c.register_operator(
        "parse",
        params=[("q", "Str")],
        pre=["raw(?q)"],
        add=["parsed(?q)"],
        cost=0.01,
        reliability=0.98,
    )
    c.register_operator(
        "analyze",
        params=[("q", "Str")],
        pre=["parsed(?q)"],
        add=["analyzed(?q)"],
        cost=0.05,
        reliability=0.97,
    )
    c.register_operator(
        "respond",
        params=[("q", "Str")],
        pre=["analyzed(?q)"],
        add=["response(?q)"],
        cost=0.001,
        reliability=0.99,
    )


def test_astar_finds_three_step_plan():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(hello)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    assert plan.verdict == SOLVED
    assert plan.length == 3
    assert plan.goal_bindings == {"x": "hello"}
    assert plan.steps[0].op_name == "parse"
    assert plan.steps[-1].op_name == "respond"


def test_astar_plan_cost_and_reliability():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(x)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    expected_cost = 0.01 + 0.05 + 0.001
    assert math.isclose(plan.cost, expected_cost, abs_tol=1e-9)
    expected_rel = 0.98 * 0.97 * 0.99
    assert math.isclose(plan.reliability_mean, expected_rel, abs_tol=1e-6)


def test_astar_infeasible():
    c = Composer()
    _basic_pipeline(c)
    # No 'raw' axiom, so 'parsed' never becomes derivable.
    plan = c.synthesize(initial=[], post=["response(?x)"])
    assert plan.verdict == INFEASIBLE
    assert plan.length == 0


def test_astar_already_satisfied():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("response(yes)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    assert plan.verdict == SOLVED
    assert plan.length == 0  # nothing to do
    assert plan.goal_bindings == {"x": "yes"}


def test_astar_no_operators_raises():
    c = Composer()
    with pytest.raises(PlanningFailure):
        c.synthesize(initial=[], post=["response(?x)"])


def test_astar_budget_exhausted():
    c = Composer()
    c.register_operator(
        "step",
        params=[("x", "Str")],
        pre=["a(?x)"],
        add=["b(?x)"],
    )
    c.register_operator(
        "step2",
        params=[("y", "Str")],
        pre=["b(?y)"],
        add=["a(?y)"],
    )
    c.add_axiom("a(s)")
    plan = c.synthesize(initial=[], post=["c(?x)"], budget=10)
    assert plan.verdict in (BUDGET_EXHAUSTED, INFEASIBLE)


def test_astar_unknown_algorithm_raises():
    c = Composer()
    c.register_operator("noop")
    with pytest.raises(UnknownAlgorithm):
        c.synthesize(initial=[], post=[], algorithm="bogo")


def test_astar_unknown_heuristic_raises():
    c = Composer()
    c.register_operator("noop")
    with pytest.raises(UnknownHeuristic):
        c.synthesize(initial=[], post=[], heuristic="oracle")


# =====================================================================
# Planning: all algorithms agree
# =====================================================================


@pytest.mark.parametrize("algo", sorted(KNOWN_ALGORITHMS))
def test_all_algorithms_find_a_solution(algo):
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"], algorithm=algo)
    assert plan.verdict == SOLVED
    assert plan.length == 3


@pytest.mark.parametrize("heur", sorted(KNOWN_HEURISTICS))
def test_all_heuristics_solve(heur):
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"], heuristic=heur)
    assert plan.verdict == SOLVED
    assert plan.length == 3


# =====================================================================
# Certificate
# =====================================================================


def test_verify_zero_observations_widens_interval():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    cert = c.verify(plan)
    assert cert.reliability_lower == 0.0
    assert cert.reliability_upper == 1.0


def test_verify_observations_narrow_interval():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    for _ in range(100):
        c.observe("parse", True)
        c.observe("analyze", True)
        c.observe("respond", True)
    cert = c.verify(plan, alpha=0.05)
    assert cert.reliability_lower > 0.85
    assert cert.reliability_upper == 1.0  # closed form for k = n


def test_verify_independent_vs_worst_case():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    for _ in range(200):
        c.observe("parse", True)
        c.observe("analyze", True)
        c.observe("respond", True)
    ind = c.verify(plan, regime=INDEPENDENT)
    wc = c.verify(plan, regime=WORST_CASE)
    # Worst-case (union bound) is never tighter than independent product
    # for a multi-step plan with imperfect operators.  When all ops are
    # perfect (k = n, CP_lo = α^{1/n}), independent rel_lo = α^{n/(n_ops)}
    # which can be very close to the union bound; we assert weakly.
    assert wc.reliability_lower <= ind.reliability_lower + 1e-9


def test_verify_unknown_regime_raises():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    with pytest.raises(UnknownRegime):
        c.verify(plan, regime="invented")


def test_verify_unknown_bound_raises():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    with pytest.raises(ComposerError):
        c.verify(plan, bound="invented")


def test_verify_kl_inv_bound_tighter_or_equal_to_hoeffding():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    for _ in range(50):
        c.observe("parse", True)
        c.observe("analyze", True)
        c.observe("respond", True)
    h = c.verify(plan, bound="hoeffding")
    k = c.verify(plan, bound="kl_inv")
    # KL is at least as tight as Hoeffding (Garivier-Cappé)
    assert k.reliability_lower >= h.reliability_lower - 1e-6


def test_verify_infeasible_returns_zero_bound():
    c = Composer()
    _basic_pipeline(c)
    plan = c.synthesize(initial=[], post=["response(?x)"])
    assert plan.verdict == INFEASIBLE
    cert = c.verify(plan)
    assert cert.reliability_lower == 0.0
    assert cert.reliability_upper == 0.0


# =====================================================================
# Execution
# =====================================================================


def test_execute_full_success():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])

    def exec_(op, bindings):
        return True, f"ok-{op}"

    out = c.execute(plan, exec_)
    assert out.succeeded
    assert out.steps_run == 3
    assert all(o["output"].startswith("ok-") for o in out.outputs)


def test_execute_failure_stops_by_default():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])

    def exec_(op, bindings):
        return op != "analyze", "out"

    out = c.execute(plan, exec_)
    assert not out.succeeded
    assert out.steps_run == 2
    assert "analyze" in out.error


def test_execute_failure_continues_when_requested():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])

    def exec_(op, bindings):
        return op != "analyze", "out"

    out = c.execute(plan, exec_, stop_on_failure=False)
    assert not out.succeeded
    assert out.steps_run == 3


def test_execute_observes_reliability():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])

    # Snapshot priors.
    parse_before = c.operators()["parse"]
    respond_before = c.operators()["respond"]

    def exec_(op, bindings):
        return op != "respond", "out"

    c.execute(plan, exec_, stop_on_failure=False)
    parse_op = c.operators()["parse"]
    respond_op = c.operators()["respond"]
    # parse succeeded → alpha increases by 1; beta unchanged.
    assert parse_op.alpha == parse_before.alpha + 1.0
    assert parse_op.beta == parse_before.beta
    # respond failed → beta increases by 1; alpha unchanged.
    assert respond_op.alpha == respond_before.alpha
    assert respond_op.beta == respond_before.beta + 1.0


def test_execute_infeasible_plan_raises():
    c = Composer()
    _basic_pipeline(c)
    plan = c.synthesize(initial=[], post=["response(?x)"])
    assert plan.verdict == INFEASIBLE
    with pytest.raises(ExecutionFailure):
        c.execute(plan, lambda *_a: (True, None))


# =====================================================================
# Observation
# =====================================================================


def test_observe_updates_posterior_monotonically():
    c = Composer()
    c.register_operator("opx")
    before = c.operators()["opx"]
    c.observe("opx", True)
    after_s = c.operators()["opx"]
    assert after_s.alpha == before.alpha + 1.0
    c.observe("opx", False)
    after_f = c.operators()["opx"]
    assert after_f.beta == before.beta + 1.0


def test_observe_unknown_op_raises():
    c = Composer()
    with pytest.raises(UnknownOperator):
        c.observe("nope", True)


# =====================================================================
# Type-check (post-plan)
# =====================================================================


def test_typecheck_succeeds_for_consistent_plan():
    c = Composer()
    # parse(s:Str) → trimmed(?s); answer(s:Str) consumes trimmed.
    # Both use Str → type-check passes.
    c.register_operator(
        "parse",
        params=[("s", "Str")],
        pre=["raw(?s)"],
        add=["trimmed(?s)"],
    )
    c.register_operator(
        "answer",
        params=[("s", "Str")],
        pre=["trimmed(?s)"],
        add=["response(?s)"],
    )
    c.add_axiom("raw(hello)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    subst = c.typecheck_plan(plan)
    assert isinstance(subst, dict)


def test_typecheck_detects_conflict():
    c = Composer()
    c.register_operator(
        "lookup",
        params=[("q", "Query")],
        pre=["stored(?q)"],
        add=["result(?q)"],
    )
    c.register_operator(
        "answer",
        params=[("r", "Result")],
        pre=["result(?r)"],
        add=["delivered(?r)"],
    )
    c.add_axiom("stored(q1)")
    plan = c.synthesize(initial=[], post=["delivered(?x)"])
    with pytest.raises(UnificationError):
        c.typecheck_plan(plan)


# =====================================================================
# Numerical helpers
# =====================================================================


def test_clopper_pearson_lower_zero_successes():
    assert clopper_pearson_lower(0, 10, 0.05) == 0.0


def test_clopper_pearson_lower_all_successes_closed_form():
    val = clopper_pearson_lower(10, 10, 0.05)
    expected = math.pow(0.025, 1.0 / 10)
    assert math.isclose(val, expected, abs_tol=1e-6)


def test_clopper_pearson_upper_all_successes():
    assert clopper_pearson_upper(10, 10, 0.05) == 1.0


def test_clopper_pearson_lower_monotone_in_k():
    lo_low = clopper_pearson_lower(3, 10, 0.05)
    lo_high = clopper_pearson_lower(8, 10, 0.05)
    assert lo_low < lo_high


def test_hoeffding_bounds_contain_phat():
    p_hat = 0.7
    n = 50
    k = int(p_hat * n)
    lo = hoeffding_lower(k, n, 0.05)
    hi = hoeffding_upper(k, n, 0.05)
    assert lo < p_hat < hi


def test_empirical_bernstein_lower_within_unit():
    val = empirical_bernstein_lower(50, 100, 0.05)
    assert 0.0 <= val <= 1.0


def test_kl_bernoulli_self_zero():
    assert math.isclose(kl_bernoulli(0.3, 0.3), 0.0, abs_tol=1e-9)


def test_kl_inverse_brackets_phat():
    p_hat = 0.7
    n = 50
    lo = kl_bernoulli_lower_inverse(p_hat, n, 0.05)
    hi = kl_bernoulli_upper_inverse(p_hat, n, 0.05)
    assert lo < p_hat < hi


def test_pac_bayes_catoni_decays_with_n():
    a = pac_bayes_catoni(kl_div=0.1, n=10, alpha=0.05)
    b = pac_bayes_catoni(kl_div=0.1, n=10_000, alpha=0.05)
    assert b > a


# =====================================================================
# Graph helpers
# =====================================================================


def test_sccs_single_node():
    sccs = strongly_connected_components(["a"], [])
    assert sccs == [["a"]]


def test_sccs_cycle():
    sccs = strongly_connected_components(
        ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]
    )
    assert len(sccs) == 1
    assert set(sccs[0]) == {"a", "b", "c"}


def test_sccs_dag():
    sccs = strongly_connected_components(
        ["a", "b", "c"], [("a", "b"), ("b", "c")]
    )
    assert len(sccs) == 3


def test_topological_sort_dag():
    order = topological_sort(["a", "b", "c"], [("a", "b"), ("b", "c")])
    # Stable, deterministic order
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_sort_cycle_raises():
    with pytest.raises(ComposerError):
        topological_sort(
            ["a", "b"], [("a", "b"), ("b", "a")]
        )


# =====================================================================
# Report
# =====================================================================


def test_report_basic():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    for _ in range(20):
        c.observe("parse", True)
    rep = c.report()
    assert rep.operator_count == 3
    assert rep.axiom_count == 1
    assert rep.plan_count == 1
    parse_stat = [s for s in rep.operator_stats if s["name"] == "parse"][0]
    assert parse_stat["observations"] == 20
    assert parse_stat["successes"] == 20


def test_report_sccs_for_linear_pipeline():
    c = Composer()
    _basic_pipeline(c)
    rep = c.report()
    # Three operators in a linear chain are three single-node SCCs.
    assert all(len(c) == 1 for c in rep.sccs)
    assert rep.cycles == ()


def test_report_detects_cycle_in_operator_graph():
    c = Composer()
    c.register_operator("a", params=[("x", "T")], pre=["p(?x)"], add=["q(?x)"])
    c.register_operator("b", params=[("x", "T")], pre=["q(?x)"], add=["p(?x)"])
    rep = c.report()
    assert len(rep.cycles) >= 1


# =====================================================================
# Fingerprint determinism + tamper-evident chain
# =====================================================================


def test_fingerprint_changes_on_registration():
    c = Composer()
    f0 = c.fingerprint
    c.register_operator("a")
    f1 = c.fingerprint
    assert f0 != f1


def test_fingerprint_chain_deterministic():
    def make(clock_seq):
        it = iter(clock_seq)
        c = Composer(clock=lambda: next(it))
        c.register_operator("a", reliability=0.9, prior_strength=4.0)
        c.register_operator("b", reliability=0.95, prior_strength=4.0)
        c.add_axiom("x")
        c.observe("a", True)
        c.observe("a", False)
        return c.fingerprint
    f1 = make([0.0] * 20)
    f2 = make([0.0] * 20)
    assert f1 == f2


def test_fingerprint_diverges_on_observation():
    c1 = Composer(clock=lambda: 0.0)
    c2 = Composer(clock=lambda: 0.0)
    c1.register_operator("a")
    c2.register_operator("a")
    c1.observe("a", True)
    c2.observe("a", False)
    assert c1.fingerprint != c2.fingerprint


def test_events_recorded():
    c = Composer(clock=lambda: 0.0)
    c.register_operator("a")
    c.observe("a", True)
    events = c.events()
    assert events[0]["kind"] == "composer.started"
    kinds = [e["kind"] for e in events]
    assert "composer.operator_registered" in kinds
    assert "composer.observed" in kinds


# =====================================================================
# Clear
# =====================================================================


def test_clear_resets_registry():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    assert plan.verdict == SOLVED
    c.clear()
    assert c.operators() == {}
    assert c.axioms() == frozenset()


# =====================================================================
# from_spec helper
# =====================================================================


def test_composer_from_spec():
    spec = [
        {
            "name": "parse",
            "params": [("q", "Str")],
            "pre": ["raw(?q)"],
            "add": ["parsed(?q)"],
            "cost": 0.01,
            "reliability": 0.98,
            "prior_strength": 4.0,
        },
        {
            "name": "respond",
            "params": [("q", "Str")],
            "pre": ["parsed(?q)"],
            "add": ["response(?q)"],
            "cost": 0.001,
        },
    ]
    c = composer_from_spec(spec)
    assert "parse" in c.operators()
    assert "respond" in c.operators()


# =====================================================================
# Larger end-to-end scenario
# =====================================================================


def test_end_to_end_pipeline_with_attestation_ledger_payload():
    c = Composer()
    c.register_operator(
        "ingest",
        params=[("d", "Doc")],
        pre=["fresh(?d)"],
        add=["ingested(?d)"],
        cost=0.005,
        reliability=0.99,
        prior_strength=20.0,
    )
    c.register_operator(
        "embed",
        params=[("d", "Doc")],
        pre=["ingested(?d)"],
        add=["embedded(?d)"],
        cost=0.02,
        reliability=0.995,
        prior_strength=20.0,
    )
    c.register_operator(
        "retrieve",
        params=[("d", "Doc")],
        pre=["embedded(?d)"],
        add=["retrieved(?d)"],
        cost=0.001,
        reliability=0.999,
        prior_strength=50.0,
    )
    c.register_operator(
        "summarise",
        params=[("d", "Doc")],
        pre=["retrieved(?d)"],
        add=["summarised(?d)"],
        cost=0.05,
        reliability=0.97,
        prior_strength=20.0,
    )
    c.add_axiom("fresh(doc1)")
    plan = c.synthesize(initial=[], post=["summarised(?d)"])
    assert plan.verdict == SOLVED
    assert plan.length == 4
    cert = c.verify(plan, alpha=0.05, regime=INDEPENDENT)
    # Lower bound should reflect prior strength: with prior strengths
    # 20–50 per operator, four-step Bonferroni gives a non-trivial LCB.
    assert cert.reliability_lower > 0.3
    # Fingerprint is hex 64.
    assert len(cert.fingerprint) == 64
    assert all(ch in "0123456789abcdef" for ch in cert.fingerprint)


def test_plan_executes_against_typed_executor():
    c = Composer()
    c.register_operator(
        "extract",
        params=[("u", "Url")],
        pre=["pending(?u)"],
        add=["extracted(?u)"],
        reliability=0.95,
        prior_strength=10.0,
    )
    c.register_operator(
        "classify",
        params=[("u", "Url")],
        pre=["extracted(?u)"],
        add=["classified(?u)"],
        reliability=0.97,
        prior_strength=10.0,
    )
    c.add_axiom("pending(https_example_com)")
    plan = c.synthesize(initial=[], post=["classified(?u)"])

    executor_calls: list = []

    def exec_(op, bindings):
        executor_calls.append((op, bindings))
        return True, {"op": op, "args": bindings}

    out = c.execute(plan, exec_)
    assert out.succeeded
    assert len(executor_calls) == 2
    assert executor_calls[0][0] == "extract"
    assert executor_calls[1][0] == "classify"


def test_jsonable_plan_and_certificate():
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    plan = c.synthesize(initial=[], post=["response(?x)"])
    j = plan.to_jsonable()
    assert j["verdict"] == SOLVED
    assert j["length"] == 3
    cert = c.verify(plan)
    cj = cert.to_jsonable()
    assert cj["regime"] == INDEPENDENT


def test_dijkstra_admits_zero_heuristic_optimality():
    # Dijkstra uses h=0 internally; A* with h_landmark should at worst
    # be tied with Dijkstra under uniform operator cost.
    c = Composer()
    _basic_pipeline(c)
    c.add_axiom("raw(z)")
    p_dij = c.synthesize(
        initial=[], post=["response(?x)"], algorithm=DIJKSTRA
    )
    p_ast = c.synthesize(initial=[], post=["response(?x)"], algorithm=ASTAR)
    assert math.isclose(p_dij.cost, p_ast.cost, abs_tol=1e-9)
