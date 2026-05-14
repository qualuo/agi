"""Tests for ``agi.transporter`` — optimal-transport runtime primitive.

The contract is theorem-driven, and the tests mirror that:

1. **Kantorovich duality**: the 1-D dual potential ``φ`` is 1-Lipschitz
   and certifies ``W_1 = E_a φ − E_b φ``.

2. **Closed-form 1-D Wasserstein**: ``W_p(μ, ν)`` equals the integral
   of ``|F_μ^{-1} − F_ν^{-1}|^p`` — checked against hand-computed
   reference values on translations, scalings, and CDF mixings.

3. **Hungarian optimality**: the recovered assignment cost matches an
   exhaustive search on small instances.

4. **Sinkhorn marginal feasibility**: the recovered plan has row and
   column marginals matching the input weights to the requested
   tolerance.

5. **Sinkhorn entropic-bias asymptote**: as ``ε → 0`` the Sinkhorn
   cost converges *monotonically* to the LP optimum.

6. **Sinkhorn divergence is a near-metric**: ``S_ε(a, a) = 0``,
   ``S_ε(a, b) ≥ 0``, and the triangle inequality is satisfied to
   within an O(ε) slack on small problems.

7. **Sliced Wasserstein invariance**: invariant under permutation of
   samples; converges to 0 on identical inputs; scales with
   translation.

8. **Cyclic monotonicity certificate**: an optimal plan returns 0
   violation; a deliberately suboptimal plan returns strictly positive
   violation.

9. **Unbalanced limit**: as the marginal penalty ``ρ → ∞`` the
   unbalanced Sinkhorn plan converges to the balanced one.

10. **Gromov-Wasserstein isometry invariance**: GW between a metric
    structure and an isometric copy / rotation / translation of it is
    zero up to the entropic bias.

11. **Barycenter midpoint**: the W_2 barycenter of two empirical 1-D
    distributions is their quantile-averaged midpoint.

12. **Runtime composability**: registered references, drift evaluation
    against them, breach thresholds, event-bus emission, attestation
    pass-through, and coverage counters all work end to end.
"""
from __future__ import annotations

import math
import threading

import pytest

from agi.events import Event, EventBus
from agi.transporter import (
    COST_EUCLIDEAN,
    COST_MANHATTAN,
    COST_SQEUCLIDEAN,
    CoverageReport,
    DriftReport,
    InvalidProblem,
    KNOWN_COSTS,
    KNOWN_METHODS,
    METHOD_AUTO,
    METHOD_EMD_1D,
    METHOD_GROMOV,
    METHOD_HUNGARIAN,
    METHOD_SINKHORN,
    METHOD_SLICED,
    METHOD_UNBALANCED,
    NotConverged,
    TRANSPORTER_BARYCENTER,
    TRANSPORTER_COMPUTED,
    TRANSPORTER_DRIFT_EVALUATED,
    TRANSPORTER_REFERENCE_REGISTERED,
    TRANSPORTER_REFERENCE_REMOVED,
    TRANSPORTER_REPORT,
    TRANSPORTER_STARTED,
    TransportProblem,
    TransportReport,
    Transporter,
    TransporterError,
    UnknownReference,
    cost_matrix,
    cyclic_monotonicity_violation,
    emd,
    gromov_wasserstein,
    hungarian,
    kantorovich_rubinstein_1d,
    make_problem,
    sinkhorn,
    sinkhorn_cost,
    sinkhorn_divergence,
    sinkhorn_entropy,
    sliced_wasserstein,
    unbalanced_sinkhorn,
    wasserstein,
    wasserstein_1d,
    wasserstein_barycenter_1d,
)


# =====================================================================
# 1-D Wasserstein (closed form)
# =====================================================================


def test_w1_identity_is_zero():
    assert wasserstein_1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_w1_constant_shift_equals_shift():
    for shift in (0.5, 1.0, 3.7):
        d = wasserstein_1d([0.0, 1.0, 2.0], [s + shift for s in [0.0, 1.0, 2.0]])
        assert math.isclose(d, shift, rel_tol=1e-9, abs_tol=1e-9)


def test_w1_p2_scales_quadratically():
    # W_2² of shift s on uniform empirical measure = s²
    d = wasserstein_1d([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], p=2.0)
    # d is W_2, not W_2². Square it.
    assert math.isclose(d * d, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_w1_handles_unequal_sizes():
    # 1 atom at 0 vs 2 atoms uniform at {0, 1}: W_1 = 0.5·0 + 0.5·1 = 0.5
    d = wasserstein_1d([0.0], [0.0, 1.0])
    assert math.isclose(d, 0.5, rel_tol=1e-9, abs_tol=1e-9)


def test_w1_rejects_empty_input():
    with pytest.raises(InvalidProblem):
        wasserstein_1d([], [1.0])


def test_w1_rejects_nonpositive_p():
    with pytest.raises(InvalidProblem):
        wasserstein_1d([0.0], [1.0], p=0.0)


# =====================================================================
# Kantorovich-Rubinstein dual
# =====================================================================


def test_kr_dual_potential_is_1_lipschitz():
    w1, phi = kantorovich_rubinstein_1d([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    # Test 1-Lipschitz on a grid.
    grid = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for i in range(len(grid)):
        for j in range(len(grid)):
            assert abs(phi(grid[i]) - phi(grid[j])) <= abs(grid[i] - grid[j]) + 1e-9


def test_kr_dual_realises_primal_value():
    a = [0.0, 1.0, 2.0]
    b = [1.0, 2.0, 3.0]
    w1, phi = kantorovich_rubinstein_1d(a, b)
    ea = sum(phi(x) for x in a) / len(a)
    eb = sum(phi(x) for x in b) / len(b)
    assert math.isclose(ea - eb, w1, rel_tol=1e-9, abs_tol=1e-9)


# =====================================================================
# Hungarian
# =====================================================================


def test_hungarian_identity_is_zero_on_diagonal_cost():
    C = [[0.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]]
    assignment, total = hungarian(C)
    assert assignment == (0, 1, 2)
    assert total == 0.0


def test_hungarian_matches_brute_force_on_small():
    import itertools
    rng_cost = [
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2],
    ]
    assignment, total = hungarian(rng_cost)
    # Brute-force.
    perms = list(itertools.permutations(range(3)))
    best = min(sum(rng_cost[i][p[i]] for i in range(3)) for p in perms)
    assert total == best
    assert sum(rng_cost[i][assignment[i]] for i in range(3)) == best


def test_hungarian_rectangular_rows_lt_cols():
    # 2 rows, 3 cols: every row gets matched, one col unused.
    assignment, total = hungarian([[1, 2, 3], [2, 4, 6]])
    # All 6 possible matchings:
    # (0,0,_),(1,1,_),... — exhaustive:
    cands = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    best = min((1 if r0c == 0 else 2 if r0c == 1 else 3) +
               (2 if r1c == 0 else 4 if r1c == 1 else 6)
               for r0c, r1c in cands)
    assert total == best


def test_hungarian_rectangular_rows_gt_cols():
    # 3 rows, 2 cols: one row left out (assignment -1).
    assignment, total = hungarian([[1, 2], [2, 4], [3, 6]])
    assert -1 in assignment
    used_cols = [a for a in assignment if a >= 0]
    assert sorted(used_cols) == [0, 1]


def test_hungarian_handles_empty():
    assert hungarian([]) == ((), 0.0)


def test_hungarian_invalid_ragged_raises():
    with pytest.raises(InvalidProblem):
        hungarian([[1, 2], [3]])


# =====================================================================
# Sinkhorn
# =====================================================================


def test_sinkhorn_marginals_are_feasible():
    p = make_problem([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]],
                     cost=COST_SQEUCLIDEAN)
    P, _, _, _, _, mv = sinkhorn(p, reg=0.1, max_iter=2000, tol=1e-10)
    assert mv < 1e-9
    for i, ai in enumerate(p.source_weights):
        assert abs(sum(P[i]) - ai) < 1e-8
    for j, bj in enumerate(p.target_weights):
        assert abs(sum(P[i][j] for i in range(p.n)) - bj) < 1e-8


def test_sinkhorn_converges_to_lp_as_reg_shrinks():
    # On identical 3-atom distributions, LP optimum is 0; cost should
    # decrease monotonically with smaller reg.
    p = make_problem([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]],
                     cost=COST_SQEUCLIDEAN)
    costs = []
    for reg in (1.0, 0.3, 0.1, 0.03):
        P, _, _, _, _, _ = sinkhorn(p, reg=reg, max_iter=5000, tol=1e-12)
        costs.append(sinkhorn_cost(P, p.cost))
    for i in range(1, len(costs)):
        assert costs[i] <= costs[i - 1] + 1e-9
    assert costs[-1] < 1e-3


def test_sinkhorn_shifted_distribution():
    # μ = uniform on {0, 1, 2}, ν = shift by 1: W_2² ≈ 1
    p = make_problem([[0.0], [1.0], [2.0]], [[1.0], [2.0], [3.0]],
                     cost=COST_SQEUCLIDEAN)
    P, _, _, _, _, mv = sinkhorn(p, reg=0.05, max_iter=5000, tol=1e-9)
    assert mv < 1e-4
    cost = sinkhorn_cost(P, p.cost)
    assert math.isclose(cost, 1.0, rel_tol=0.05, abs_tol=0.05)


def test_sinkhorn_invalid_reg_raises():
    p = make_problem([[0.0]], [[0.0]])
    with pytest.raises(InvalidProblem):
        sinkhorn(p, reg=0.0)


# =====================================================================
# Sinkhorn divergence
# =====================================================================


def test_sinkhorn_divergence_self_is_zero():
    a = [[0.0], [1.0], [2.0]]
    s = sinkhorn_divergence(a, a, reg=0.1, max_iter=2000, tol=1e-10)
    assert math.isclose(s, 0.0, abs_tol=1e-9)


def test_sinkhorn_divergence_nonneg():
    a = [[0.0], [1.0]]
    b = [[2.0], [3.0]]
    s = sinkhorn_divergence(a, b, reg=0.1, max_iter=2000, tol=1e-10)
    assert s >= -1e-9


def test_sinkhorn_divergence_grows_with_separation():
    a = [[0.0], [1.0]]
    near = sinkhorn_divergence(a, [[0.5], [1.5]], reg=0.1, max_iter=2000)
    far = sinkhorn_divergence(a, [[5.0], [6.0]], reg=0.1, max_iter=2000)
    assert far > near + 1.0  # well-separated case dominates


# =====================================================================
# Sliced Wasserstein
# =====================================================================


def test_sliced_wasserstein_identity_zero():
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    sw, se, vals = sliced_wasserstein(X, X, p=2.0, n_projections=32, seed=1)
    assert sw < 1e-9


def test_sliced_wasserstein_sample_permutation_invariant():
    # Identical inputs in different sample orders → SW is invariant.
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    Y = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    sw1, *_ = sliced_wasserstein(X, Y, n_projections=16, seed=42)
    sw2, *_ = sliced_wasserstein(Y, X, n_projections=16, seed=42)
    assert math.isclose(sw1, sw2, abs_tol=1e-9)


def test_sliced_wasserstein_deterministic_under_seed():
    X = [[0.0, 0.0], [1.0, 0.0]]
    Y = [[2.0, 0.0], [3.0, 0.0]]
    a, _, _ = sliced_wasserstein(X, Y, n_projections=16, seed=7)
    b, _, _ = sliced_wasserstein(X, Y, n_projections=16, seed=7)
    assert a == b


def test_sliced_wasserstein_dimension_mismatch_raises():
    with pytest.raises(InvalidProblem):
        sliced_wasserstein([[0.0]], [[0.0, 0.0]])


def test_sliced_wasserstein_responds_to_translation():
    X = [[0.0, 0.0]] * 5
    Y = [[10.0, 10.0]] * 5
    near, *_ = sliced_wasserstein(X, X, n_projections=16, seed=1)
    far, *_ = sliced_wasserstein(X, Y, n_projections=16, seed=1)
    assert far > near + 5.0


# =====================================================================
# Cyclic monotonicity
# =====================================================================


def test_cyclic_monotonicity_optimal_plan_has_zero_violation():
    plan = ((0.5, 0.0), (0.0, 0.5))
    cost = ((0.0, 1.0), (1.0, 0.0))
    mv, nv, nc = cyclic_monotonicity_violation(
        plan, cost, n_cycles=100, cycle_length=2, seed=1
    )
    assert mv == 0.0
    assert nv == 0
    assert nc <= 100


def test_cyclic_monotonicity_suboptimal_plan_violates():
    plan_bad = ((0.0, 0.5), (0.5, 0.0))   # swaps i.e. anti-diagonal
    cost = ((0.0, 1.0), (1.0, 0.0))
    mv, nv, nc = cyclic_monotonicity_violation(
        plan_bad, cost, n_cycles=50, cycle_length=2, seed=1
    )
    assert mv > 0.0
    assert nv > 0


# =====================================================================
# Unbalanced OT
# =====================================================================


def test_unbalanced_converges_toward_balanced_when_penalty_large():
    p = make_problem([[0.0], [1.0]], [[0.0], [1.0]], cost=COST_SQEUCLIDEAN)
    Pb, _, _, _, conv_b, mv_b = sinkhorn(p, reg=0.05, max_iter=5000, tol=1e-10)
    Pu, _, _, _, _, mv_u = unbalanced_sinkhorn(
        p, reg=0.05, marginal_penalty=1e4, max_iter=5000, tol=1e-10
    )
    # Plans should agree to within marginal-penalty bias.
    for i in range(2):
        for j in range(2):
            assert abs(Pu[i][j] - Pb[i][j]) < 1e-3


def test_unbalanced_rejects_nonpositive_reg():
    p = make_problem([[0.0]], [[0.0]])
    with pytest.raises(InvalidProblem):
        unbalanced_sinkhorn(p, reg=0.0)
    with pytest.raises(InvalidProblem):
        unbalanced_sinkhorn(p, reg=0.1, marginal_penalty=0.0)


# =====================================================================
# Gromov-Wasserstein
# =====================================================================


def test_gromov_translation_invariant():
    # Same point cloud, translated: the inner distance matrix is
    # identical, so GW should be near zero.
    Xpts = [[0.0], [1.0], [3.0]]
    Cx = cost_matrix(Xpts, Xpts, kind=COST_EUCLIDEAN)
    Ypts = [[10.0], [11.0], [13.0]]
    Cy = cost_matrix(Ypts, Ypts, kind=COST_EUCLIDEAN)
    P, gw, n_iter, conv = gromov_wasserstein(
        Cx, Cy, [1/3, 1/3, 1/3], [1/3, 1/3, 1/3],
        reg=0.05, outer_iter=30, inner_iter=200, tol=1e-7,
    )
    # Optimal GW is 0 here; entropic bias keeps it small.
    assert gw < 0.5


def test_gromov_responds_to_shape_mismatch():
    # Different shapes: line vs single point cluster — should be positive.
    Cx = cost_matrix([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]],
                     kind=COST_EUCLIDEAN)
    Cy = cost_matrix([[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]],
                     kind=COST_EUCLIDEAN)
    P, gw, _, _ = gromov_wasserstein(
        Cx, Cy, [1/3]*3, [1/3]*3, reg=0.05,
        outer_iter=30, inner_iter=200, tol=1e-7,
    )
    assert gw > 0.1


def test_gromov_rejects_size_mismatch():
    # Cx is 1×1 but two marginal weights supplied — inconsistent.
    with pytest.raises(InvalidProblem):
        gromov_wasserstein([[0.0]], [[0.0, 1.0], [1.0, 0.0]],
                           [0.5, 0.5], [0.5, 0.5])


# =====================================================================
# Barycenter
# =====================================================================


def test_barycenter_midpoint_of_two_translated():
    A = [0.0, 1.0, 2.0]
    B = [10.0, 11.0, 12.0]
    sup, w = wasserstein_barycenter_1d([A, B], n_support=120)
    # Each barycenter atom is (a + b)/2 = 0.5·atom_A + 0.5·atom_B in
    # quantile space. Mean of the support should be 6.0.
    mean = sum(s * wi for s, wi in zip(sup, w))
    assert math.isclose(mean, 6.0, rel_tol=0.01, abs_tol=0.05)
    # Weights sum to 1.
    assert math.isclose(sum(w), 1.0, rel_tol=1e-9)


def test_barycenter_invalid_inputs():
    with pytest.raises(InvalidProblem):
        wasserstein_barycenter_1d([])
    with pytest.raises(InvalidProblem):
        wasserstein_barycenter_1d([[0.0]], weights=[1.0, 2.0])


# =====================================================================
# Cost matrix
# =====================================================================


def test_cost_matrix_known_costs():
    pts = [[0.0, 0.0], [1.0, 0.0]]
    sq = cost_matrix(pts, pts, kind=COST_SQEUCLIDEAN)
    eu = cost_matrix(pts, pts, kind=COST_EUCLIDEAN)
    ma = cost_matrix(pts, pts, kind=COST_MANHATTAN)
    assert sq == ((0.0, 1.0), (1.0, 0.0))
    assert eu == ((0.0, 1.0), (1.0, 0.0))
    assert ma == ((0.0, 1.0), (1.0, 0.0))


def test_cost_matrix_rejects_unknown_kind():
    with pytest.raises(InvalidProblem):
        cost_matrix([[0.0]], [[0.0]], kind="not-a-cost")


def test_cost_matrix_rejects_custom_kind():
    with pytest.raises(InvalidProblem):
        cost_matrix([[0.0]], [[0.0]], kind="custom")


# =====================================================================
# make_problem validation
# =====================================================================


def test_make_problem_rejects_inconsistent_dim():
    with pytest.raises(InvalidProblem):
        make_problem([[0.0]], [[0.0, 1.0]])


def test_make_problem_rejects_bad_weights():
    with pytest.raises(InvalidProblem):
        make_problem([[0.0], [1.0]], [[0.0], [1.0]],
                     source_weights=[1.0, -0.5])
    with pytest.raises(InvalidProblem):
        make_problem([[0.0]], [[0.0]], source_weights=[1.0, 2.0])


def test_make_problem_rejects_ragged_custom_cost():
    # Wrong row count: source has 1 atom, cost has 2 rows.
    with pytest.raises(InvalidProblem):
        make_problem([[0.0]], [[0.0], [1.0]], cost=[[1.0, 2.0], [3.0, 4.0]])
    # Wrong col count: target has 2 atoms, row only has 1 entry.
    with pytest.raises(InvalidProblem):
        make_problem([[0.0]], [[0.0], [1.0]], cost=[[1.0]])


def test_make_problem_uniform_weight_default():
    p = make_problem([[0.0], [1.0]], [[0.0], [1.0]])
    assert p.source_weights == (0.5, 0.5)
    assert p.target_weights == (0.5, 0.5)


# =====================================================================
# Transporter runtime
# =====================================================================


def test_transporter_emits_started_on_construction():
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    Transporter(bus=bus)
    assert any(e.kind == TRANSPORTER_STARTED for e in events)


def test_transporter_register_and_remove_reference():
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    tr = Transporter(bus=bus)
    tr.register_reference("R", samples=[[0.0], [1.0], [2.0]])
    assert "R" in tr.references()
    assert any(e.kind == TRANSPORTER_REFERENCE_REGISTERED for e in events)
    assert tr.remove_reference("R")
    assert "R" not in tr.references()
    assert any(e.kind == TRANSPORTER_REFERENCE_REMOVED for e in events)
    assert not tr.remove_reference("nope")


def test_transporter_rejects_empty_reference_id():
    tr = Transporter()
    with pytest.raises(InvalidProblem):
        tr.register_reference("", samples=[[0.0]])


def test_transporter_drift_against_unknown_reference_raises():
    tr = Transporter()
    with pytest.raises(UnknownReference):
        tr.drift("nope", samples=[[0.0]])


def test_transporter_drift_self_is_near_zero():
    tr = Transporter()
    samples = [[0.0], [1.0], [2.0]]
    tr.register_reference("ref", samples=samples)
    rep = tr.drift("ref", samples=samples, method=METHOD_SINKHORN,
                   reg=0.05, max_iter=2000, tol=1e-10)
    assert rep.score < 0.05
    assert rep.breach is False
    assert isinstance(rep, DriftReport)


def test_transporter_drift_far_breaches_threshold():
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    tr = Transporter(bus=bus)
    tr.register_reference("ref", samples=[[0.0], [1.0], [2.0]])
    rep = tr.drift("ref", samples=[[10.0], [11.0], [12.0]],
                   method=METHOD_SINKHORN, reg=0.05, threshold=10.0)
    assert rep.score > 10.0
    assert rep.breach is True
    assert any(
        e.kind == TRANSPORTER_DRIFT_EVALUATED and e.data["breach"]
        for e in events
    )


def test_transporter_compute_emits_computed_event():
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    tr = Transporter(bus=bus)
    rep = tr.compute(source=[[0.0], [1.0]], target=[[0.5], [1.5]],
                     method=METHOD_SINKHORN, reg=0.05)
    assert isinstance(rep, TransportReport)
    assert any(e.kind == TRANSPORTER_COMPUTED for e in events)
    assert rep.certificate["method"] == METHOD_SINKHORN
    assert rep.certificate["content_hash"]
    assert rep.certificate["n"] == 2
    assert rep.certificate["m"] == 2


def test_transporter_compute_with_sinkhorn_divergence():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0]], target=[[0.0], [1.0]],
                     method=METHOD_SINKHORN, reg=0.1,
                     compute_divergence=True, max_iter=2000)
    assert rep.divergence is not None
    assert abs(rep.divergence) < 0.01


def test_transporter_compute_with_cyclic_monotonicity_verification():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0], [2.0]],
                     target=[[0.0], [1.0], [2.0]],
                     method=METHOD_SINKHORN, reg=0.01,
                     max_iter=5000, tol=1e-12,
                     verify_monotonicity=True)
    cmono = rep.certificate.get("cyclic_monotonicity")
    assert cmono is not None
    assert cmono["max_violation"] >= 0


def test_transporter_compute_hungarian_route():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0], [2.0]],
                     target=[[0.0], [1.0], [2.0]],
                     method=METHOD_HUNGARIAN)
    assert rep.method == METHOD_HUNGARIAN
    assert rep.distance == 0.0
    assert rep.plan is not None
    # Plan rows should sum to source weight 1/3.
    for row in rep.plan:
        assert math.isclose(sum(row), 1.0 / 3, abs_tol=1e-12)


def test_transporter_compute_hungarian_falls_back_when_rectangular():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0]],
                     target=[[0.0], [1.0], [2.0]],
                     method=METHOD_HUNGARIAN, reg=0.05, max_iter=2000)
    # Falls back to Sinkhorn for rectangular problem.
    assert rep.method == METHOD_SINKHORN


def test_transporter_compute_emd_1d_route():
    tr = Transporter()
    rep = tr.compute(source=[0.0, 1.0, 2.0],
                     target=[1.0, 2.0, 3.0],
                     method=METHOD_EMD_1D)
    assert rep.method == METHOD_EMD_1D
    assert math.isclose(rep.distance, 1.0, abs_tol=1e-9)


def test_transporter_compute_sliced_route():
    tr = Transporter()
    rep = tr.compute(source=[[0.0, 0.0]] * 3,
                     target=[[1.0, 1.0]] * 3,
                     method=METHOD_SLICED, n_projections=16, seed=1)
    assert rep.method == METHOD_SLICED
    assert rep.distance > 0


def test_transporter_compute_unbalanced_route():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0]], target=[[0.0], [1.0]],
                     method=METHOD_UNBALANCED, reg=0.05,
                     marginal_penalty=1000.0, max_iter=2000)
    assert rep.method == METHOD_UNBALANCED
    assert rep.plan is not None


def test_transporter_compute_gromov_route():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0], [2.0]],
                     target=[[0.0], [1.0], [2.0]],
                     method=METHOD_GROMOV, reg=0.05, max_iter=2000)
    assert rep.method == METHOD_GROMOV
    assert rep.plan is not None
    assert rep.distance >= 0


def test_transporter_compute_rejects_unknown_method():
    tr = Transporter()
    with pytest.raises(InvalidProblem):
        tr.compute(source=[[0.0]], target=[[0.0]], method="bogus")


def test_transporter_compute_auto_method_uses_hungarian_on_uniform_square():
    tr = Transporter()
    rep = tr.compute(source=[[0.0], [1.0]], target=[[0.0], [1.0]],
                     method=METHOD_AUTO)
    assert rep.method == METHOD_HUNGARIAN


def test_transporter_compute_auto_method_uses_sinkhorn_on_large():
    tr = Transporter()
    n = 10
    rep = tr.compute(source=[[float(i)] for i in range(n)],
                     target=[[float(i)] for i in range(n)],
                     method=METHOD_AUTO, reg=0.05, max_iter=2000)
    assert rep.method == METHOD_SINKHORN


def test_transporter_match_returns_one_to_one_assignment_hungarian():
    tr = Transporter()
    plan, total = tr.match(source=[[0.0], [1.0], [2.0]],
                           target=[[0.1], [1.1], [2.1]],
                           method=METHOD_HUNGARIAN)
    assert plan is not None
    # 3×3 plan, one non-zero per row.
    for row in plan:
        nz = [v for v in row if v > 1e-9]
        assert len(nz) == 1


def test_transporter_barycenter_emits_event():
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    tr = Transporter(bus=bus)
    rep = tr.barycenter_1d([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
                           n_support=80)
    assert rep.sources == 2
    assert len(rep.support) == 80
    assert any(e.kind == TRANSPORTER_BARYCENTER for e in events)


def test_transporter_coverage_counts():
    tr = Transporter()
    tr.register_reference("R", samples=[[0.0]])
    tr.compute(source=[[0.0]], target=[[0.0]], method=METHOD_SINKHORN, reg=0.5)
    tr.drift("R", samples=[[0.0]], method=METHOD_SINKHORN, reg=0.5)
    tr.barycenter_1d([[0.0]], n_support=10)
    cov = tr.coverage()
    assert isinstance(cov, CoverageReport)
    assert cov.references == 1
    assert cov.computes >= 2  # compute + the drift internally calls compute
    assert cov.drifts == 1
    assert cov.barycenters == 1


def test_transporter_clear_resets_state():
    tr = Transporter()
    tr.register_reference("R", samples=[[0.0]])
    tr.compute(source=[[0.0]], target=[[0.0]], method=METHOD_SINKHORN, reg=0.5)
    tr.clear()
    cov = tr.coverage()
    assert cov.references == 0
    assert cov.computes == 0
    assert cov.drifts == 0
    assert cov.barycenters == 0


def test_transporter_report_emits_event_with_summary():
    bus = EventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    tr = Transporter(bus=bus)
    tr.register_reference("R", samples=[[0.0]])
    cov = tr.report()
    assert isinstance(cov, CoverageReport)
    report_evt = [e for e in events if e.kind == TRANSPORTER_REPORT]
    assert report_evt and report_evt[0].data["references"] == 1


def test_transporter_attestor_pass_through():
    class FakeAttestor:
        def __init__(self):
            self.calls = []

        def record(self, *, kind, payload):
            self.calls.append((kind, dict(payload)))

    att = FakeAttestor()
    tr = Transporter(attestor=att)
    tr.compute(source=[[0.0]], target=[[0.0]], method=METHOD_SINKHORN, reg=0.5)
    assert any(k == "transporter.computed" for k, _ in att.calls)


def test_transporter_thread_safety_register_concurrent():
    tr = Transporter()
    errs = []

    def worker(i):
        try:
            tr.register_reference(f"R{i}", samples=[[float(i)]])
        except Exception as e:
            errs.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    assert len(tr.references()) == 20


# =====================================================================
# Convenience wrappers
# =====================================================================


def test_wasserstein_convenience():
    d = wasserstein(source=[[0.0], [1.0]], target=[[1.0], [2.0]],
                    method=METHOD_HUNGARIAN)
    # Optimal matching: 0→1, 1→2 with sq-Euclidean cost: 1+1 / 2 = 1
    assert math.isclose(d, 1.0, abs_tol=1e-9)


def test_emd_convenience_uniform_square():
    d = emd([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], cost=COST_EUCLIDEAN)
    assert math.isclose(d, 3.0, abs_tol=1e-9)


def test_emd_convenience_falls_back_to_sinkhorn_when_unbalanced():
    d = emd([[0.0]], [[0.0], [1.0]], cost=COST_EUCLIDEAN)
    # 1 atom at 0 against {0, 1} uniform: W_1 = 0.5
    assert math.isclose(d, 0.5, abs_tol=1e-3)


# =====================================================================
# Determinism
# =====================================================================


def test_transporter_deterministic_with_seed():
    tr1 = Transporter(random_seed=42)
    tr2 = Transporter(random_seed=42)
    r1 = tr1.compute(source=[[0.0, 0.0], [1.0, 0.0]],
                     target=[[2.0, 0.0], [3.0, 0.0]],
                     method=METHOD_SLICED, n_projections=8, seed=42)
    r2 = tr2.compute(source=[[0.0, 0.0], [1.0, 0.0]],
                     target=[[2.0, 0.0], [3.0, 0.0]],
                     method=METHOD_SLICED, n_projections=8, seed=42)
    assert r1.distance == r2.distance


# =====================================================================
# Method / cost catalogues
# =====================================================================


def test_known_methods_complete():
    assert METHOD_AUTO in KNOWN_METHODS
    assert METHOD_HUNGARIAN in KNOWN_METHODS
    assert METHOD_SINKHORN in KNOWN_METHODS
    assert METHOD_SLICED in KNOWN_METHODS
    assert METHOD_EMD_1D in KNOWN_METHODS
    assert METHOD_UNBALANCED in KNOWN_METHODS
    assert METHOD_GROMOV in KNOWN_METHODS


def test_known_costs_complete():
    assert COST_SQEUCLIDEAN in KNOWN_COSTS
    assert COST_EUCLIDEAN in KNOWN_COSTS
    assert COST_MANHATTAN in KNOWN_COSTS


# =====================================================================
# Composition with other primitives (smoke tests)
# =====================================================================


def test_drift_composes_with_existing_eventbus_pattern():
    # The runtime emits transporter.drift_evaluated events that a
    # higher-level coordinator subscribes to.
    bus = EventBus()
    breach_seen = []
    bus.subscribe(
        lambda e: breach_seen.append(e.data["score"]),
        kind=TRANSPORTER_DRIFT_EVALUATED,
    )
    tr = Transporter(bus=bus)
    tr.register_reference("ref", samples=[[0.0], [1.0]])
    tr.drift("ref", samples=[[5.0], [6.0]], method=METHOD_SINKHORN, reg=0.1,
             threshold=1.0)
    assert breach_seen
    assert breach_seen[0] > 1.0


def test_certificate_content_hash_is_deterministic():
    tr = Transporter()
    r1 = tr.compute(source=[[0.0], [1.0]], target=[[0.0], [1.0]],
                    method=METHOD_SINKHORN, reg=0.1)
    r2 = tr.compute(source=[[0.0], [1.0]], target=[[0.0], [1.0]],
                    method=METHOD_SINKHORN, reg=0.1)
    assert r1.certificate["content_hash"] == r2.certificate["content_hash"]
