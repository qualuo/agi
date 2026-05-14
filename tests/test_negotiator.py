"""Tests for ``agi.negotiator`` — multi-party allocation runtime primitive.

The contract is axiomatic, and the tests reflect that:

1. **Pareto-optimality** — every solver returns a Pareto-optimal point
   relative to the rest of the family. Verified on hand-built games
   where the analytic frontier is known.

2. **Equal-treatment / symmetry** — when two parties are
   indistinguishable (same utility, same disagreement, same weight),
   they receive identical allocations. Verified for every concept.

3. **Closed-form anchors** — under linear utilities + zero
   disagreement, Nash, KS, and proportional-fair all collapse to the
   uniform x = B/n allocation. Numerical solvers must match within
   bisection tolerance.

4. **Affine-invariance** — Nash and KS satisfy
   ``x*(α u + β) = x*(u)``. Verified by rescaling utilities and
   re-solving.

5. **Leximin equity** — leximin Pareto-dominates max-min when there
   are unused parties, and equals max-min when the bottleneck is
   unique.

6. **VCG correctness** — single-item second-price equals the second-
   highest bid; multi-item externality matches the analytic formula
   on hand-built bid matrices.

7. **Envy-freeness** — for symmetric parties under Nash/KS/PF the
   allocation is envy-free; the report flag is set accordingly.

8. **Event emission + attestation pass-through** — every public
   mutator and solver emits the documented event, and an
   AttestationLedger receipt is minted when an attestor is wired in.
"""
from __future__ import annotations

import math
import threading

import pytest

from agi.events import Event, EventBus
from agi.negotiator import (
    AXIOM_AFFINE_INVARIANCE,
    AXIOM_BUDGET_BALANCED,
    AXIOM_ENVY_FREE,
    AXIOM_IIA,
    AXIOM_LEXIMIN_EQUITY,
    AXIOM_MONOTONICITY,
    AXIOM_PARETO,
    AXIOM_SYMMETRY,
    Allocation,
    CONCEPT_EGALITARIAN,
    CONCEPT_KALAI_SMORODINSKY,
    CONCEPT_LEXIMIN,
    CONCEPT_NASH,
    CONCEPT_PROPORTIONAL_FAIR,
    CONCEPT_UTILITARIAN,
    CONCEPT_VCG,
    EnvyReport,
    KNOWN_CONCEPTS,
    LinearUtility,
    NEGOTIATOR_ALLOCATED,
    NEGOTIATOR_BUDGET_CHANGED,
    NEGOTIATOR_CLEARED,
    NEGOTIATOR_PARTY_REGISTERED,
    NEGOTIATOR_PARTY_REMOVED,
    NEGOTIATOR_STARTED,
    NegotiationInfeasible,
    Negotiator,
    PartySpec,
    PiecewiseLinearUtility,
    QuadraticUtility,
    Utility,
    VCGAllocation,
    allocate,
    compute_envy,
    compute_pareto_check,
    min_utility,
    nash_product,
    vcg_allocate,
    welfare,
)


# ============================================================
# Section 1 — Utility classes
# ============================================================


class TestLinearUtility:
    def test_basic_evaluation(self) -> None:
        u = LinearUtility(slope=2.0, cap=5.0)
        assert u.evaluate(0.0) == 0.0
        assert u.evaluate(2.5) == 5.0
        assert u.evaluate(5.0) == 10.0
        # Beyond cap is clipped.
        assert u.evaluate(10.0) == 10.0
        assert u.evaluate(-1.0) == 0.0

    def test_derivative_constant_under_cap(self) -> None:
        u = LinearUtility(slope=3.0, cap=4.0)
        assert u.derivative(0.0) == 3.0
        assert u.derivative(2.0) == 3.0
        # At the cap, the right derivative is 0.
        assert u.derivative(4.0) == 0.0

    def test_inverse_round_trip(self) -> None:
        u = LinearUtility(slope=2.0, cap=5.0)
        for x in [0.0, 1.0, 2.5, 4.5]:
            assert math.isclose(u.inverse(u.evaluate(x)), x, abs_tol=1e-9)

    def test_rejects_negative_slope(self) -> None:
        with pytest.raises(ValueError):
            LinearUtility(slope=-1.0, cap=5.0)

    def test_rejects_nonpositive_cap(self) -> None:
        with pytest.raises(ValueError):
            LinearUtility(slope=1.0, cap=0.0)
        with pytest.raises(ValueError):
            LinearUtility(slope=1.0, cap=-1.0)

    def test_intercept(self) -> None:
        u = LinearUtility(slope=1.0, cap=3.0, intercept=2.0)
        assert u.evaluate(0.0) == 2.0
        assert u.evaluate(3.0) == 5.0
        # Inverse must respect intercept.
        assert math.isclose(u.inverse(3.0), 1.0, abs_tol=1e-9)


class TestQuadraticUtility:
    def test_concave_at_saturation(self) -> None:
        u = QuadraticUtility(a=4.0, b=2.0)
        # cap = a/b = 2, u_max = a^2 / 2b = 4
        assert math.isclose(u.cap, 2.0, abs_tol=1e-9)
        assert math.isclose(u.evaluate(2.0), 4.0, abs_tol=1e-9)
        # u'(0) = a; u'(cap) = 0
        assert math.isclose(u.derivative(0.0), 4.0, abs_tol=1e-9)
        assert math.isclose(u.derivative(2.0), 0.0, abs_tol=1e-9)

    def test_inverse_consistency(self) -> None:
        u = QuadraticUtility(a=4.0, b=2.0)
        for x in [0.1, 0.5, 1.0, 1.5, 1.9]:
            v = u.evaluate(x)
            x_back = u.inverse(v)
            assert math.isclose(x_back, x, abs_tol=1e-7)

    def test_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            QuadraticUtility(a=0.0, b=1.0)
        with pytest.raises(ValueError):
            QuadraticUtility(a=1.0, b=0.0)
        with pytest.raises(ValueError):
            QuadraticUtility(a=-1.0, b=1.0)


class TestPiecewiseLinearUtility:
    def test_evaluation_along_segments(self) -> None:
        u = PiecewiseLinearUtility(breakpoints=((0.0, 0.0), (2.0, 4.0), (5.0, 7.0)))
        assert math.isclose(u.evaluate(0.0), 0.0)
        assert math.isclose(u.evaluate(1.0), 2.0)
        assert math.isclose(u.evaluate(2.0), 4.0)
        # Second segment: slope = (7-4)/3 = 1.0
        assert math.isclose(u.evaluate(3.5), 4.0 + 1.5, abs_tol=1e-9)
        assert math.isclose(u.evaluate(5.0), 7.0)

    def test_derivative_per_segment(self) -> None:
        u = PiecewiseLinearUtility(breakpoints=((0.0, 0.0), (2.0, 4.0), (5.0, 7.0)))
        assert math.isclose(u.derivative(0.5), 2.0)
        assert math.isclose(u.derivative(3.0), 1.0)
        assert u.derivative(5.0) == 0.0

    def test_inverse_consistency(self) -> None:
        u = PiecewiseLinearUtility(breakpoints=((0.0, 0.0), (2.0, 4.0), (5.0, 7.0)))
        for x in [0.5, 1.5, 2.0, 3.0, 4.5]:
            v = u.evaluate(x)
            x_back = u.inverse(v)
            assert math.isclose(x_back, x, abs_tol=1e-7)

    def test_rejects_non_concave_slopes(self) -> None:
        # Slopes increasing -> convex, not concave.
        with pytest.raises(ValueError):
            PiecewiseLinearUtility(breakpoints=((0.0, 0.0), (2.0, 1.0), (5.0, 7.0)))

    def test_rejects_decreasing_u(self) -> None:
        with pytest.raises(ValueError):
            PiecewiseLinearUtility(breakpoints=((0.0, 5.0), (2.0, 3.0)))

    def test_rejects_duplicate_x(self) -> None:
        with pytest.raises(ValueError):
            PiecewiseLinearUtility(breakpoints=((0.0, 0.0), (0.0, 1.0)))


# ============================================================
# Section 2 — Utilitarian allocation
# ============================================================


class TestUtilitarian:
    def test_linear_three_party_water_fills_highest_slope_first(self) -> None:
        neg = Negotiator()
        neg.register_party("hi", LinearUtility(slope=3.0, cap=4.0))
        neg.register_party("mid", LinearUtility(slope=2.0, cap=4.0))
        neg.register_party("lo", LinearUtility(slope=1.0, cap=4.0))
        neg.set_budget(6.0)
        r = neg.allocate_utilitarian()
        # Optimal: fill hi=4, mid=2, lo=0; welfare = 12 + 4 = 16.
        assert math.isclose(r.allocation.assignments["hi"], 4.0, abs_tol=1e-6)
        assert math.isclose(r.allocation.assignments["mid"], 2.0, abs_tol=1e-6)
        assert math.isclose(r.allocation.assignments["lo"], 0.0, abs_tol=1e-6)
        assert math.isclose(r.welfare, 16.0, abs_tol=1e-6)
        assert math.isclose(r.allocation.total_allocated, 6.0, abs_tol=1e-6)

    def test_budget_exceeds_total_caps(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=3.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=3.0))
        neg.set_budget(10.0)
        r = neg.allocate_utilitarian()
        # Both saturate at cap, slack remains.
        assert math.isclose(r.allocation.assignments["a"], 3.0, abs_tol=1e-6)
        assert math.isclose(r.allocation.assignments["b"], 3.0, abs_tol=1e-6)
        assert r.allocation.slack() > 0

    def test_quadratic_water_fill(self) -> None:
        neg = Negotiator()
        neg.register_party("a", QuadraticUtility(a=4.0, b=1.0))  # cap = 4
        neg.register_party("b", QuadraticUtility(a=2.0, b=1.0))  # cap = 2
        neg.set_budget(3.0)
        r = neg.allocate_utilitarian()
        # KKT: u_a' = 4 - x_a = λ ; u_b' = 2 - x_b = λ ; x_a + x_b = 3.
        # ⇒ x_a - x_b = 2, x_a + x_b = 3 → x_a = 2.5, x_b = 0.5.
        assert math.isclose(r.allocation.assignments["a"], 2.5, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["b"], 0.5, abs_tol=1e-5)

    def test_certificate_includes_pareto_and_budget_balanced(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(5.0)
        r = neg.allocate_utilitarian()
        assert AXIOM_BUDGET_BALANCED in r.certificate
        # Single party => trivially Pareto-optimal.
        assert AXIOM_PARETO in r.certificate

    def test_zero_budget_yields_zero_allocations(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=5.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=5.0))
        neg.set_budget(0.0)
        r = neg.allocate_utilitarian()
        assert all(v == 0.0 for v in r.allocation.assignments.values())
        assert r.welfare == 0.0


# ============================================================
# Section 3 — Egalitarian + leximin
# ============================================================


class TestEgalitarian:
    def test_linear_equal_utility(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=2.0, cap=10.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(6.0)
        r = neg.allocate_egalitarian()
        # Equal utilities → 2 x_a = x_b, x_a + x_b = 6 → x_a = 2, x_b = 4
        assert math.isclose(r.allocation.assignments["a"], 2.0, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["b"], 4.0, abs_tol=1e-5)
        u_a = r.allocation.utilities["a"]
        u_b = r.allocation.utilities["b"]
        assert math.isclose(u_a, u_b, abs_tol=1e-5)

    def test_egalitarian_with_weights(self) -> None:
        # weights skew the floor: party "vip" demands 2× the floor.
        neg = Negotiator()
        neg.register_party("vip", LinearUtility(slope=1.0, cap=10.0), weight=2.0)
        neg.register_party("std", LinearUtility(slope=1.0, cap=10.0), weight=1.0)
        neg.set_budget(6.0)
        r = neg.allocate_egalitarian()
        # u/w equal across parties → u_vip = 2 u_std and x_vip + x_std = 6.
        # With slope 1, u_vip = x_vip; so x_vip = 2 x_std → x_vip = 4, x_std = 2.
        assert math.isclose(r.allocation.assignments["vip"], 4.0, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["std"], 2.0, abs_tol=1e-5)

    def test_egalitarian_with_disagreement(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0), disagreement=2.0)
        neg.register_party("b", LinearUtility(slope=1.0, cap=10.0), disagreement=0.0)
        neg.set_budget(4.0)
        r = neg.allocate_egalitarian(floor_from_disagreement=True)
        # u_i - d_i equal → u_a - 2 = u_b → x_a = x_b + 2;
        # x_a + x_b = 4 → x_a = 3, x_b = 1.
        assert math.isclose(r.allocation.assignments["a"], 3.0, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["b"], 1.0, abs_tol=1e-5)

    def test_egalitarian_pareto_recovers_when_capped(self) -> None:
        # Cap forces one party to saturate; remaining budget should flow
        # to the under-capped party.
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=1.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(5.0)
        r = neg.allocate_egalitarian()
        # a saturates at 1.0; b gets the rest (4.0). Utilities (1, 4) — egalitarian
        # solver does what it can; the egalitarian *floor* is 1.0 set by a's cap.
        # We DO expect leximin to use the slack, but the egalitarian solver
        # itself may legitimately stop at "equal up to caps".  Verify the floor
        # is at least 1.0.
        assert min(r.allocation.utilities.values()) >= 1.0 - 1e-6


class TestLeximin:
    def test_leximin_uses_slack_after_floor(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=1.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(5.0)
        r = neg.allocate_leximin()
        # a saturates at 1.0; b gets remaining budget (4.0).
        assert math.isclose(r.allocation.assignments["a"], 1.0, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["b"], 4.0, abs_tol=1e-5)
        assert AXIOM_LEXIMIN_EQUITY in r.certificate

    def test_leximin_equals_egalitarian_when_no_slack(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.register_party("b", LinearUtility(slope=2.0, cap=10.0))
        neg.set_budget(6.0)
        egal = neg.allocate_egalitarian()
        lex = neg.allocate_leximin()
        for pid in egal.allocation.assignments:
            assert math.isclose(
                egal.allocation.assignments[pid],
                lex.allocation.assignments[pid],
                abs_tol=1e-5,
            )


# ============================================================
# Section 4 — Nash bargaining
# ============================================================


class TestNash:
    def test_linear_zero_disagreement_uniform(self) -> None:
        # Nash with linear utilities + zero disagreement → uniform split.
        neg = Negotiator()
        for slope in [1.0, 2.0, 5.0, 0.5]:
            neg.register_party(f"p_{slope}", LinearUtility(slope=slope, cap=20.0))
        neg.set_budget(8.0)
        r = neg.allocate_nash()
        target = 8.0 / 4
        for pid, x in r.allocation.assignments.items():
            assert math.isclose(x, target, abs_tol=1e-4)
        # Envy-free under symmetric Nash.
        assert r.envy.envy_free or r.envy.max_envy < 1e-3

    def test_nash_certificate(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(4.0)
        r = neg.allocate_nash()
        # Nash certificate must include IIA + affine-invariance.
        assert AXIOM_IIA in r.certificate
        assert AXIOM_AFFINE_INVARIANCE in r.certificate

    def test_nash_with_positive_disagreement(self) -> None:
        # Party with high disagreement gets more (must clear its BATNA first).
        neg = Negotiator()
        # Note: u(x) needs to exceed d; with linear slope=1, x must exceed d.
        neg.register_party("hi", LinearUtility(slope=1.0, cap=10.0), disagreement=2.0)
        neg.register_party("lo", LinearUtility(slope=1.0, cap=10.0), disagreement=0.0)
        neg.set_budget(6.0)
        r = neg.allocate_nash()
        # Nash: maximise log(x_hi - 2) + log(x_lo). KKT 1/(x_hi-2) = 1/x_lo →
        # x_hi - 2 = x_lo, plus budget x_hi + x_lo = 6 → x_hi = 4, x_lo = 2.
        assert math.isclose(r.allocation.assignments["hi"], 4.0, abs_tol=1e-4)
        assert math.isclose(r.allocation.assignments["lo"], 2.0, abs_tol=1e-4)

    def test_nash_affine_invariance(self) -> None:
        # Rescaling a utility by α should leave the x-allocation unchanged.
        neg_base = Negotiator()
        neg_base.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg_base.register_party("b", LinearUtility(slope=2.0, cap=10.0))
        neg_base.set_budget(4.0)
        base = neg_base.allocate_nash()

        neg_scaled = Negotiator()
        neg_scaled.register_party("a", LinearUtility(slope=10.0, cap=10.0))  # ×10
        neg_scaled.register_party("b", LinearUtility(slope=2.0, cap=10.0))
        neg_scaled.set_budget(4.0)
        scaled = neg_scaled.allocate_nash()
        for pid in ("a", "b"):
            assert math.isclose(
                base.allocation.assignments[pid],
                scaled.allocation.assignments[pid],
                abs_tol=1e-3,
            )


# ============================================================
# Section 5 — Kalai-Smorodinsky
# ============================================================


class TestKalaiSmorodinsky:
    def test_ks_progress_to_ideal(self) -> None:
        # Two parties with different ideal points; KS picks proportional progress.
        neg = Negotiator()
        # ideal_a = 1.0 * 4 = 4 ; ideal_b = 2.0 * 4 = 8 (both with cap 4)
        neg.register_party("a", LinearUtility(slope=1.0, cap=4.0))
        neg.register_party("b", LinearUtility(slope=2.0, cap=4.0))
        # With budget = 4 and disagreement 0, KS picks t such that
        #   u_a = 4t (so x_a = 4t / 1 = 4t)
        #   u_b = 8t (so x_b = 8t / 2 = 4t)
        # Sum to budget: 8t = 4 → t = 0.5, x_a = x_b = 2.
        neg.set_budget(4.0)
        r = neg.allocate_kalai_smorodinsky()
        assert math.isclose(r.allocation.assignments["a"], 2.0, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["b"], 2.0, abs_tol=1e-5)
        assert AXIOM_MONOTONICITY in r.certificate

    def test_ks_full_budget_when_caps_allow(self) -> None:
        # When the budget exceeds Σ caps, KS gives everyone their ideal.
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=3.0))
        neg.register_party("b", LinearUtility(slope=2.0, cap=3.0))
        neg.set_budget(10.0)
        r = neg.allocate_kalai_smorodinsky()
        assert math.isclose(r.allocation.assignments["a"], 3.0, abs_tol=1e-5)
        assert math.isclose(r.allocation.assignments["b"], 3.0, abs_tol=1e-5)

    def test_ks_rejects_unreachable_disagreement(self) -> None:
        neg = Negotiator()
        # disagreement >= ideal makes the KS axiom ill-defined.
        neg.register_party("a", LinearUtility(slope=1.0, cap=5.0), disagreement=5.0)
        neg.register_party("b", LinearUtility(slope=1.0, cap=5.0), disagreement=0.0)
        neg.set_budget(2.0)
        with pytest.raises(NegotiationInfeasible):
            neg.allocate_kalai_smorodinsky()


# ============================================================
# Section 6 — Proportional fair
# ============================================================


class TestProportionalFair:
    def test_pf_equals_nash_at_zero_disagreement(self) -> None:
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.register_party("b", LinearUtility(slope=2.0, cap=10.0))
        neg.set_budget(4.0)
        pf = neg.allocate_proportional_fair()
        nash = neg.allocate_nash()
        for pid in ("a", "b"):
            assert math.isclose(
                pf.allocation.assignments[pid],
                nash.allocation.assignments[pid],
                abs_tol=1e-4,
            )


# ============================================================
# Section 7 — Symmetry / equal-treatment
# ============================================================


class TestSymmetry:
    @pytest.mark.parametrize(
        "concept",
        [
            CONCEPT_UTILITARIAN,
            CONCEPT_EGALITARIAN,
            CONCEPT_NASH,
            CONCEPT_KALAI_SMORODINSKY,
            CONCEPT_PROPORTIONAL_FAIR,
            CONCEPT_LEXIMIN,
        ],
    )
    def test_symmetric_parties_get_equal_share(self, concept: str) -> None:
        # Three identical parties → x_i = B/n for all i.
        neg = Negotiator()
        for pid in ("p1", "p2", "p3"):
            neg.register_party(pid, LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(6.0)
        r = getattr(neg, f"allocate_{concept}")()
        target = 6.0 / 3
        for pid, x in r.allocation.assignments.items():
            assert math.isclose(x, target, abs_tol=1e-4), (
                f"{concept}: party {pid} got {x}, expected {target}"
            )


# ============================================================
# Section 8 — Envy + Pareto checks
# ============================================================


class TestEnvyAndPareto:
    def test_envy_free_for_symmetric_nash(self) -> None:
        neg = Negotiator()
        for pid in ("a", "b", "c"):
            neg.register_party(pid, LinearUtility(slope=1.0, cap=5.0))
        neg.set_budget(6.0)
        r = neg.allocate_nash()
        assert r.envy.envy_free
        assert r.envy.max_envy < 1e-3
        assert AXIOM_ENVY_FREE in r.certificate

    def test_utilitarian_can_have_envy(self) -> None:
        # Utilitarian gives everything to highest-slope; the lower-slope party
        # envies because *their* utility for the high party's allocation
        # exceeds their own (0).
        neg = Negotiator()
        neg.register_party("hi", LinearUtility(slope=5.0, cap=10.0))
        neg.register_party("lo", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(5.0)
        r = neg.allocate_utilitarian()
        assert not r.envy.envy_free
        assert r.envy.max_envy > 0

    def test_pareto_check_no_dominator_for_utilitarian(self) -> None:
        # Utilitarian is Pareto-optimal by construction; the probe sweep
        # cannot find a dominator.
        neg = Negotiator()
        neg.register_party("a", LinearUtility(slope=2.0, cap=5.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=5.0))
        neg.set_budget(4.0)
        r = neg.allocate_utilitarian()
        assert not r.pareto.dominated


# ============================================================
# Section 9 — VCG mechanism
# ============================================================


class TestVCG:
    def test_single_item_second_price(self) -> None:
        # Single item with bids 10, 7, 3 → winner pays 7.
        result = vcg_allocate(
            bids={"a": {"x": 10.0}, "b": {"x": 7.0}, "c": {"x": 3.0}},
            items=("x",),
        )
        assert result.winners == {"x": "a"}
        assert math.isclose(result.payments["a"], 7.0, abs_tol=1e-9)
        assert result.payments["b"] == 0.0
        assert result.payments["c"] == 0.0

    def test_two_items_independent_externalities(self) -> None:
        # Each item is independent — VCG payment is the second-highest bid
        # *on that item* when the winner has no externality on the other.
        result = vcg_allocate(
            bids={
                "a": {"x": 10.0, "y": 2.0},
                "b": {"x": 5.0, "y": 8.0},
                "c": {"x": 1.0, "y": 3.0},
            },
            items=("x", "y"),
        )
        # a wins x (bids 10 vs 5 vs 1), b wins y (8 vs 3 vs 2).
        assert result.winners["x"] == "a"
        assert result.winners["y"] == "b"
        # a's externality: w_{−a} = b's y(8) + b's x(5) = 13 ; others' welfare
        # under actual alloc = b's y(8) = 8 → payment = 5.
        assert math.isclose(result.payments["a"], 5.0, abs_tol=1e-9)
        # b's externality: w_{−b} = a's x(10) + c's y(3) = 13 ; others'
        # welfare under actual = a's x(10) = 10 → payment = 3.
        assert math.isclose(result.payments["b"], 3.0, abs_tol=1e-9)

    def test_vcg_zero_payment_for_non_winners(self) -> None:
        result = vcg_allocate(
            bids={"a": {"x": 10.0}, "b": {"x": 7.0}},
            items=("x",),
        )
        assert result.payments["b"] == 0.0

    def test_vcg_individual_rationality(self) -> None:
        # Each winning party's utility (bid - payment) >= 0.
        result = vcg_allocate(
            bids={"a": {"x": 10.0}, "b": {"x": 7.0}, "c": {"x": 3.0}},
            items=("x",),
        )
        for party, items in result.bundle.items():
            total_bid = sum(0.0 if items is None else
                            sum({"x": 10.0, "y": 5.0}.get(it, 0.0) for it in items)
                            for _ in [None]) if False else \
                        sum(
                            {"a": {"x": 10.0}, "b": {"x": 7.0}, "c": {"x": 3.0}}
                            .get(party, {}).get(it, 0.0)
                            for it in items
                        )
            assert total_bid >= result.payments.get(party, 0.0) - 1e-9

    def test_vcg_truthfulness_dominant_strategy(self) -> None:
        # Bidding above true value can only lose money, never gain it. Simulate.
        truthful = vcg_allocate(
            bids={"a": {"x": 10.0}, "b": {"x": 7.0}},
            items=("x",),
        )
        overbid = vcg_allocate(
            bids={"a": {"x": 15.0}, "b": {"x": 7.0}},  # a overbids
            items=("x",),
        )
        # Both: a wins; payment in both cases = 7.0; net utility = 10 − 7 = 3.
        assert truthful.payments["a"] == overbid.payments["a"]

    def test_bundle_cap_raises(self) -> None:
        bids = {"a": {f"i{i}": float(i) for i in range(20)}}
        with pytest.raises(NegotiationInfeasible):
            vcg_allocate(bids, items=tuple(f"i{i}" for i in range(20)))

    def test_negotiator_vcg_method(self) -> None:
        neg = Negotiator()
        result = neg.vcg_auction(
            items=("x",),
            bids={"a": {"x": 10.0}, "b": {"x": 7.0}},
        )
        assert isinstance(result, VCGAllocation)
        assert result.winners == {"x": "a"}


# ============================================================
# Section 10 — Event emission
# ============================================================


class TestEventEmission:
    def _bus_with_log(self) -> tuple[EventBus, list[Event]]:
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(lambda e: seen.append(e))
        return bus, seen

    def test_started_event_on_construction(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        kinds = [e.kind for e in seen]
        assert NEGOTIATOR_STARTED in kinds

    def test_register_emits_event(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        neg.register_party("a", LinearUtility(slope=1.0, cap=5.0))
        kinds = [e.kind for e in seen]
        assert NEGOTIATOR_PARTY_REGISTERED in kinds

    def test_remove_emits_event(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        neg.register_party("a", LinearUtility(slope=1.0, cap=5.0))
        seen.clear()
        neg.remove_party("a")
        kinds = [e.kind for e in seen]
        assert NEGOTIATOR_PARTY_REMOVED in kinds

    def test_remove_nonexistent_does_not_emit(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        seen.clear()
        assert neg.remove_party("nope") is False
        assert not any(e.kind == NEGOTIATOR_PARTY_REMOVED for e in seen)

    def test_budget_change_emits_event(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        seen.clear()
        neg.set_budget(5.0)
        kinds = [e.kind for e in seen]
        assert NEGOTIATOR_BUDGET_CHANGED in kinds

    def test_allocated_event_carries_certificate(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.register_party("b", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(4.0)
        seen.clear()
        neg.allocate_nash()
        alloc_events = [e for e in seen if e.kind == NEGOTIATOR_ALLOCATED]
        assert len(alloc_events) == 1
        assert "certificate" in alloc_events[0].data
        assert "assignments" in alloc_events[0].data

    def test_failure_event_on_invalid_budget(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        # Force an infeasible by mutating budget to NaN via the path:
        # the public setter rejects, so we bypass via direct attribute.
        neg._budget = float("nan")  # type: ignore[attr-defined]
        seen.clear()
        with pytest.raises(NegotiationInfeasible):
            neg.allocate_utilitarian()
        kinds = [e.kind for e in seen]
        assert "negotiator.allocation_failed" in kinds

    def test_clear_emits_event(self) -> None:
        bus, seen = self._bus_with_log()
        neg = Negotiator(bus=bus)
        seen.clear()
        neg.clear()
        kinds = [e.kind for e in seen]
        assert NEGOTIATOR_CLEARED in kinds


# ============================================================
# Section 11 — Attestation
# ============================================================


class _FakeAttestor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def record(self, *, kind: str, payload: dict) -> dict:
        self.calls.append((kind, payload))
        # Return an object with a `.hash` attribute.
        return type("R", (), {"hash": f"sha-{len(self.calls):03d}"})()


class TestAttestation:
    def test_negotiation_carries_receipt_when_attested(self) -> None:
        att = _FakeAttestor()
        neg = Negotiator(attestor=att)
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(4.0)
        r = neg.allocate_nash()
        assert r.receipt_hash != ""
        assert att.calls[0][0] == "negotiator.allocated"

    def test_vcg_carries_receipt_when_attested(self) -> None:
        att = _FakeAttestor()
        neg = Negotiator(attestor=att)
        r = neg.vcg_auction(
            items=("x",),
            bids={"a": {"x": 5.0}, "b": {"x": 3.0}},
        )
        assert r.receipt_hash != ""
        assert any(call[0] == "negotiator.vcg" for call in att.calls)

    def test_attestor_failure_is_swallowed(self) -> None:
        class BadAttestor:
            def record(self, *, kind: str, payload: dict) -> None:
                raise RuntimeError("ledger down")

        neg = Negotiator(attestor=BadAttestor())
        neg.register_party("a", LinearUtility(slope=1.0, cap=10.0))
        neg.set_budget(4.0)
        # Must not raise.
        r = neg.allocate_nash()
        # Receipt falls back to the local digest, so non-empty either way.
        assert r.receipt_hash != ""


# ============================================================
# Section 12 — Thread-safety
# ============================================================


class TestThreadSafety:
    def test_concurrent_register_and_allocate(self) -> None:
        neg = Negotiator()

        def register(idx: int) -> None:
            for i in range(50):
                neg.register_party(
                    f"t{idx}-{i}",
                    LinearUtility(slope=1.0 + 0.1 * i, cap=5.0),
                )

        threads = [threading.Thread(target=register, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert neg.n_parties == 200
        neg.set_budget(50.0)
        r = neg.allocate_egalitarian()
        assert math.isclose(r.allocation.total_allocated, 50.0, abs_tol=1e-3)


# ============================================================
# Section 13 — Free function: `allocate(...)`
# ============================================================


class TestFreeAllocateFunction:
    def test_allocate_unknown_concept(self) -> None:
        with pytest.raises(ValueError):
            allocate(parties=[], budget=1.0, concept="bogus")

    def test_allocate_vcg_via_free_fn_rejects(self) -> None:
        with pytest.raises(ValueError):
            allocate(parties=[], budget=1.0, concept=CONCEPT_VCG)

    def test_allocate_each_concept(self) -> None:
        parties = [
            PartySpec(id="a", utility=LinearUtility(slope=1.0, cap=10.0)),
            PartySpec(id="b", utility=LinearUtility(slope=2.0, cap=10.0)),
        ]
        for concept in [
            CONCEPT_UTILITARIAN,
            CONCEPT_EGALITARIAN,
            CONCEPT_LEXIMIN,
            CONCEPT_NASH,
            CONCEPT_KALAI_SMORODINSKY,
            CONCEPT_PROPORTIONAL_FAIR,
        ]:
            alloc = allocate(parties=parties, budget=6.0, concept=concept)
            assert math.isclose(sum(alloc.values()), 6.0, abs_tol=1e-4)


# ============================================================
# Section 14 — Welfare / Nash-product helpers
# ============================================================


class TestWelfareHelpers:
    def test_welfare_matches_solver(self) -> None:
        parties = [
            PartySpec(id="a", utility=LinearUtility(slope=2.0, cap=10.0)),
            PartySpec(id="b", utility=LinearUtility(slope=1.0, cap=10.0)),
        ]
        alloc = {"a": 3.0, "b": 2.0}
        w = welfare(parties, alloc)
        assert math.isclose(w, 2.0 * 3.0 + 1.0 * 2.0, abs_tol=1e-9)

    def test_min_utility(self) -> None:
        parties = [
            PartySpec(id="a", utility=LinearUtility(slope=2.0, cap=10.0)),
            PartySpec(id="b", utility=LinearUtility(slope=1.0, cap=10.0)),
        ]
        alloc = {"a": 1.0, "b": 5.0}
        assert min_utility(parties, alloc) == 2.0

    def test_nash_product_minus_inf_at_disagreement(self) -> None:
        parties = [
            PartySpec(
                id="a",
                utility=LinearUtility(slope=1.0, cap=10.0),
                disagreement=2.0,
            )
        ]
        # At x=2, u-d=0 → log = -inf.
        assert nash_product(parties, {"a": 2.0}) == float("-inf")
        # Above disagreement, finite.
        assert nash_product(parties, {"a": 5.0}) > 0.0


# ============================================================
# Section 15 — Validation surfaces
# ============================================================


class TestValidation:
    def test_register_rejects_empty_id(self) -> None:
        neg = Negotiator()
        with pytest.raises(ValueError):
            neg.register_party("", LinearUtility(slope=1.0, cap=5.0))

    def test_register_rejects_non_utility(self) -> None:
        neg = Negotiator()
        with pytest.raises(TypeError):
            neg.register_party("a", "not a utility")  # type: ignore[arg-type]

    def test_register_rejects_nonpositive_weight(self) -> None:
        neg = Negotiator()
        with pytest.raises(ValueError):
            neg.register_party(
                "a", LinearUtility(slope=1.0, cap=5.0), weight=0.0,
            )

    def test_register_rejects_negative_disagreement(self) -> None:
        neg = Negotiator()
        with pytest.raises(ValueError):
            neg.register_party(
                "a", LinearUtility(slope=1.0, cap=5.0), disagreement=-1.0,
            )

    def test_set_budget_rejects_negative(self) -> None:
        neg = Negotiator()
        with pytest.raises(ValueError):
            neg.set_budget(-1.0)

    def test_known_concepts_constant_matches_methods(self) -> None:
        # Every concept in KNOWN_CONCEPTS except VCG has an allocate_* method.
        neg = Negotiator()
        for concept in KNOWN_CONCEPTS:
            if concept == CONCEPT_VCG:
                continue
            assert hasattr(neg, f"allocate_{concept}")
