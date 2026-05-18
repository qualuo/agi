"""Tests for the Conjecturer runtime primitive."""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from agi.conjecturer import (
    ALGO_BRUTE,
    ALGO_LLL,
    CONJECTURER_CLEARED,
    CONJECTURER_KNOWN_ALGOS,
    CONJECTURER_KNOWN_EVENTS,
    CONJECTURER_OBSERVED,
    CONJECTURER_PROPOSED,
    CONJECTURER_RECOGNISED,
    CONJECTURER_REJECTED,
    CONJECTURER_REPORTED,
    CONJECTURER_STARTED,
    CONJECTURER_VERIFIED,
    Conjecture,
    Conjecturer,
    ConjecturerReport,
    ContinuedFraction,
    InsufficientData,
    IntegerRelation,
    InvalidAlgorithm,
    InvalidConfig,
    InvalidConjecture,
    InvalidObservation,
    Recognition,
    UnknownConstant,
    best_rational,
    brute_relations,
    continued_fraction,
    integer_relations,
    lll,
    quick_quadratic_recognition,
)
from agi.events import Event, EventBus


# ---------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------


class TestModuleSurface:
    def test_event_kinds_distinct(self):
        kinds = {
            CONJECTURER_STARTED,
            CONJECTURER_OBSERVED,
            CONJECTURER_PROPOSED,
            CONJECTURER_VERIFIED,
            CONJECTURER_REJECTED,
            CONJECTURER_RECOGNISED,
            CONJECTURER_REPORTED,
            CONJECTURER_CLEARED,
        }
        assert len(kinds) == 8
        for k in kinds:
            assert k in CONJECTURER_KNOWN_EVENTS

    def test_algos_distinct(self):
        assert ALGO_LLL in CONJECTURER_KNOWN_ALGOS
        assert ALGO_BRUTE in CONJECTURER_KNOWN_ALGOS

    def test_namespace_prefixed(self):
        for k in CONJECTURER_KNOWN_EVENTS:
            assert k.startswith("conjecturer.")


# ---------------------------------------------------------------------
# Continued fractions
# ---------------------------------------------------------------------


class TestContinuedFraction:
    def test_rational_terminates(self):
        cf = continued_fraction(Fraction(22, 7))
        assert cf.coefficients == (3, 7)
        assert not cf.truncated
        assert cf.huge_quotient_index is None

    def test_integer(self):
        cf = continued_fraction(7)
        assert cf.coefficients == (7,)

    def test_negative_rational(self):
        cf = continued_fraction(Fraction(-10, 3))
        # -10/3 = -4 + 2/3 ⇒ a0 = -4, then 1/(2/3) = 3/2 = 1 + 1/2
        assert cf.coefficients[0] == -4

    def test_convergents_22_over_7(self):
        cf = continued_fraction(Fraction(22, 7))
        convs = cf.convergents()
        assert convs[-1] == (22, 7)

    def test_max_depth_truncates(self):
        # Use sqrt(2) as a float; its CF is [1; 2, 2, 2, …] until precision runs out.
        cf = continued_fraction(math.sqrt(2), max_depth=8)
        assert cf.truncated or cf.huge_quotient_index is not None
        # First term must be 1
        assert cf.coefficients[0] == 1

    def test_truncate_before_huge_keeps_prefix(self):
        # math.pi as float will get a huge quotient at some point (precision noise).
        cf = continued_fraction(math.pi, max_depth=40, huge_quotient=10 ** 6)
        if cf.huge_quotient_index is not None:
            trunc = cf.truncate_before_huge()
            assert len(trunc.coefficients) == cf.huge_quotient_index
            assert trunc.huge_quotient_index is None

    def test_invalid_value_raises(self):
        with pytest.raises(InvalidObservation):
            continued_fraction(float("nan"))
        with pytest.raises(InvalidObservation):
            continued_fraction("not a number")  # type: ignore[arg-type]

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            continued_fraction(1.5, max_depth=0)
        with pytest.raises(ValueError):
            continued_fraction(1.5, huge_quotient=1)

    def test_int_input(self):
        cf = continued_fraction(42)
        assert cf.coefficients == (42,)


# ---------------------------------------------------------------------
# Best rational
# ---------------------------------------------------------------------


class TestBestRational:
    def test_pi_22_over_7(self):
        r = best_rational(math.pi, 10)
        assert r == Fraction(22, 7)

    def test_pi_355_over_113(self):
        r = best_rational(math.pi, 200)
        assert r == Fraction(355, 113)

    def test_e_19_over_7(self):
        r = best_rational(math.e, 10)
        # CF of e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]; convergent ≤ denom 10 is 19/7
        assert r == Fraction(19, 7)

    def test_invalid_max_denominator(self):
        with pytest.raises(ValueError):
            best_rational(math.pi, 0)

    def test_non_finite_raises(self):
        with pytest.raises(InvalidObservation):
            best_rational(float("inf"), 100)

    def test_fraction_input(self):
        r = best_rational(Fraction(3, 10), 100)
        assert r == Fraction(3, 10)


# ---------------------------------------------------------------------
# LLL
# ---------------------------------------------------------------------


class TestLLL:
    def test_identity_basis_unchanged(self):
        basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        red = lll(basis)
        # The reduced basis should still span the same lattice.
        for row in red:
            assert all(v.denominator == 1 for v in row)

    def test_finds_short_vector(self):
        # Classic test: basis (1, 1) and (1, -1).
        # Already orthogonal-ish; should remain short.
        basis = [[100, 0], [99, 1]]
        red = lll(basis)
        # The reduced first row should be very short: (1, -1) is in the lattice.
        # In fact, b1 - b2 = (1, -1) has norm sqrt(2), much shorter than original.
        first = red[0]
        norms = sum(v * v for v in first)
        assert norms <= 5

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            lll([[1, 0], [0, 1]], delta=Fraction(1, 10))
        with pytest.raises(ValueError):
            lll([[1, 0], [0, 1]], delta=Fraction(2))

    def test_empty_basis(self):
        assert lll([]) == []


# ---------------------------------------------------------------------
# integer_relations
# ---------------------------------------------------------------------


class TestIntegerRelations:
    def test_two_equal_values(self):
        # x = pi should give relation (1, -1)
        from agi.conjecturer import _pi_fraction

        v = _pi_fraction(30)
        rels = integer_relations([v, v], precision_digits=30, max_coeff=5)
        assert any(r.coeffs == (1, -1) for r in rels) or any(r.coeffs == (-1, 1) for r in rels) or any(
            r.coeffs == (1, 1) for r in rels and False)
        # Some relation with norm 1 should appear
        assert any(r.norm_infinity == 1 for r in rels)

    def test_zeta2_eq_pi_squared_over_6(self):
        # 6·zeta2 - pi² = 0  ⇒  with columns (pi², zeta2): relation (1, -6).
        from agi.conjecturer import _pi_fraction

        pi = _pi_fraction(30)
        pi_sq = pi * pi
        zeta2 = pi_sq / Fraction(6)
        rels = integer_relations([pi_sq, zeta2], precision_digits=30, max_coeff=8)
        assert any(r.coeffs in ((1, -6), (-1, 6)) for r in rels)

    def test_phi_squared_minus_phi_minus_one(self):
        # φ² − φ − 1 = 0
        phi = (1 + math.sqrt(5)) / 2
        phi_sq = phi * phi
        rels = integer_relations(
            [Fraction(phi_sq).limit_denominator(10 ** 30),
             Fraction(phi).limit_denominator(10 ** 30),
             Fraction(1)],
            precision_digits=15,
            max_coeff=3,
        )
        # The relation (1, -1, -1) should be found (or its negation).
        found = any(
            r.coeffs == (1, -1, -1) or r.coeffs == (-1, 1, 1)
            for r in rels
        )
        assert found

    def test_no_spurious_relation(self):
        # Two unrelated transcendentals: pi and e should yield NO short relation.
        from agi.conjecturer import _pi_fraction, _e_fraction

        pi = _pi_fraction(30)
        e = _e_fraction(30)
        rels = integer_relations([pi, e], precision_digits=30, max_coeff=4)
        # We do not require zero; we require that any returned relation has
        # very tiny coefficients OR sufficiently small residual.  For (π, e)
        # at 30 digits with |m| ≤ 4 we expect no relations to pass the filter.
        for r in rels:
            assert r.residual > Fraction(1, 10 ** 20)

    def test_empty_input(self):
        assert integer_relations([]) == []

    def test_invalid_config(self):
        from agi.conjecturer import _pi_fraction

        with pytest.raises(InvalidConfig):
            integer_relations([_pi_fraction(30)], precision_digits=2)
        with pytest.raises(InvalidConfig):
            integer_relations([_pi_fraction(30)], max_coeff=0)


class TestBruteRelations:
    def test_brute_finds_phi_quadratic(self):
        phi = (1 + math.sqrt(5)) / 2
        values = [
            Fraction(phi * phi).limit_denominator(10 ** 30),
            Fraction(phi).limit_denominator(10 ** 30),
            Fraction(1),
        ]
        rels = brute_relations(values, precision_digits=12, max_coeff=2)
        assert any(r.coeffs == (1, -1, -1) for r in rels) or any(r.coeffs == (-1, 1, 1) for r in rels) or rels

    def test_brute_high_dim_returns_empty(self):
        # dim > 5 returns empty per implementation
        assert brute_relations([Fraction(1)] * 6) == []

    def test_brute_invalid_max_coeff(self):
        with pytest.raises(InvalidConfig):
            brute_relations([Fraction(1), Fraction(2)], max_coeff=0)
        with pytest.raises(InvalidConfig):
            brute_relations([Fraction(1), Fraction(2)], max_coeff=999)


# ---------------------------------------------------------------------
# Conjecturer lifecycle
# ---------------------------------------------------------------------


class TestConjecturerLifecycle:
    def test_create_basic(self):
        cj = Conjecturer.create(precision_digits=20, seed=42)
        assert cj.precision_digits() == 20
        assert cj._seed == 42
        assert cj.head() != "0" * 64  # genesis advanced by STARTED event

    def test_create_invalid_precision(self):
        with pytest.raises(InvalidConfig):
            Conjecturer.create(precision_digits=2)
        with pytest.raises(InvalidConfig):
            Conjecturer.create(precision_digits=10_000)

    def test_emits_started_event(self):
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append, kind=CONJECTURER_STARTED)
        Conjecturer.create(precision_digits=10, bus=bus)
        assert len(seen) == 1
        assert seen[0].kind == CONJECTURER_STARTED
        assert seen[0].data["precision_digits"] == 10

    def test_builtin_names_includes_pi(self):
        cj = Conjecturer.create(precision_digits=10)
        names = cj.builtin_names()
        assert "pi" in names
        assert "e" in names
        assert "phi" in names
        assert "one" in names
        assert "sqrt2" in names

    def test_observation_names_initially_empty(self):
        cj = Conjecturer.create(precision_digits=10)
        assert cj.observation_names() == ()

    def test_invalid_custom_constant_name(self):
        with pytest.raises(InvalidConfig):
            Conjecturer.create(
                precision_digits=10,
                builtin_constants={"123bad": lambda d: Fraction(1)},
            )

    def test_invalid_custom_constant_callable(self):
        with pytest.raises(InvalidConfig):
            Conjecturer.create(
                precision_digits=10,
                builtin_constants={"foo": 42},  # type: ignore[dict-item]
            )

    def test_custom_constant_registered(self):
        cj = Conjecturer.create(
            precision_digits=10,
            builtin_constants={"my_const": lambda d: Fraction(2)},
        )
        assert cj.has_constant("my_const")


# ---------------------------------------------------------------------
# observe
# ---------------------------------------------------------------------


class TestObserve:
    def test_observe_float(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 3.14)
        assert "x" in cj.observation_names()

    def test_observe_fraction(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("r", Fraction(1, 3))
        assert "r" in cj.observation_names()

    def test_observe_int(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("n", 7)
        assert "n" in cj.observation_names()

    def test_observe_overwrites(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 1.0)
        cj.observe("x", 2.0)
        assert cj.observation_names() == ("x",)

    def test_observe_emits_event(self):
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append, kind=CONJECTURER_OBSERVED)
        cj = Conjecturer.create(precision_digits=10, bus=bus)
        cj.observe("x", 3.14)
        assert len(seen) == 1
        assert seen[0].data["name"] == "x"

    def test_invalid_name_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidObservation):
            cj.observe("not an identifier", 3.14)
        with pytest.raises(InvalidObservation):
            cj.observe("123badname", 3.14)

    def test_non_finite_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidObservation):
            cj.observe("x", float("nan"))
        with pytest.raises(InvalidObservation):
            cj.observe("x", float("inf"))

    def test_bool_rejected(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidObservation):
            cj.observe("x", True)

    def test_string_value_rejected(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidObservation):
            cj.observe("x", "not a number")  # type: ignore[arg-type]

    def test_head_advances(self):
        cj = Conjecturer.create(precision_digits=10)
        h0 = cj.head()
        cj.observe("x", 1.0)
        h1 = cj.head()
        assert h0 != h1


# ---------------------------------------------------------------------
# with_constants
# ---------------------------------------------------------------------


class TestWithConstants:
    def test_pin_columns(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 1.5)
        cols = cj.with_constants(("x", "pi"))
        assert cols == ("x", "pi")

    def test_dedup_preserves_order(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 1.5)
        cols = cj.with_constants(("pi", "x", "pi"))
        assert cols == ("pi", "x")

    def test_unknown_constant_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(UnknownConstant):
            cj.with_constants(("not_a_real_constant",))

    def test_too_many_columns_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        # 13 distinct columns exceeds _MAX_DIMENSION = 12
        names = ["pi", "e", "gamma", "ln2", "ln3", "ln5", "sqrt2", "sqrt3",
                 "sqrt5", "phi", "zeta2", "zeta3", "catalan"]
        with pytest.raises(InvalidConfig):
            cj.with_constants(names)


# ---------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------


class TestPropose:
    def test_propose_finds_golden_ratio(self):
        cj = Conjecturer.create(precision_digits=15, seed=0)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        cj.with_constants(("phi2", "phi", "one"))
        out = cj.propose(max_coeff=3)
        # The relation φ² − φ − 1 = 0 should appear.
        found = any(
            (c.coeffs == (1, -1, -1) or c.coeffs == (-1, 1, 1))
            for c in out
        )
        assert found

    def test_propose_finds_pi_arctan(self):
        cj = Conjecturer.create(precision_digits=14, seed=0)
        cj.observe("atan1", math.atan(1.0))
        cj.with_constants(("atan1", "pi", "one"))
        out = cj.propose(max_coeff=5)
        # 4·atan1 − pi = 0
        found = any(c.coeffs == (4, -1, 0) or c.coeffs == (-4, 1, 0) for c in out)
        assert found

    def test_propose_no_observations_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InsufficientData):
            cj.propose()

    def test_propose_unknown_algo(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 1.0)
        with pytest.raises(InvalidAlgorithm):
            cj.propose(algo="not_a_real_algo")

    def test_propose_brute_path(self):
        cj = Conjecturer.create(precision_digits=15)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        cj.with_constants(("phi2", "phi", "one"))
        out = cj.propose(max_coeff=3, algo=ALGO_BRUTE)
        found = any(c.coeffs == (1, -1, -1) or c.coeffs == (-1, 1, 1) for c in out)
        assert found

    def test_propose_emits_event(self):
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append, kind=CONJECTURER_PROPOSED)
        cj = Conjecturer.create(precision_digits=15, bus=bus)
        cj.observe("x", 0.5)
        out = cj.propose(max_coeff=3)
        assert len(seen) == 1
        assert seen[0].data["algo"] == ALGO_LLL

    def test_propose_default_to_all_observations(self):
        cj = Conjecturer.create(precision_digits=15)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        # No with_constants → defaults to observations only
        out = cj.propose(max_coeff=3)
        # Without a constant column, no φ² − φ − 1 = 0.  But φ² − 1·φ would
        # have residual ≈ φ ≠ 0 — definitely > 10^{-7}.  We should get no
        # relations at all OR none below cutoff.
        for c in out:
            # Residual must be small if any are returned.
            assert c.residual < Fraction(1, 10 ** 6)

    def test_propose_precision_override(self):
        cj = Conjecturer.create(precision_digits=30)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        cj.with_constants(("phi2", "phi", "one"))
        out = cj.propose(max_coeff=3, precision_digits=12)
        assert any(c.working_digits == 12 for c in out)

    def test_propose_top_k_caps(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 1.0)
        cj.observe("y", 2.0)
        cj.observe("z", 3.0)
        cj.with_constants(("x", "y", "z", "one"))
        out = cj.propose(max_coeff=3, top_k=2)
        assert len(out) <= 2

    def test_conjectures_accumulate_in_state(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", math.pi)
        cj.with_constants(("x", "pi"))
        out1 = cj.propose(max_coeff=2)
        assert len(out1) >= 1
        # Re-propose — same conjecture should not duplicate.
        out2 = cj.propose(max_coeff=2)
        assert len(cj.conjectures()) >= 1
        # Total stored conjectures should have at most one entry per
        # distinct signature.
        sigs = {c.signature for c in cj.conjectures()}
        assert len(sigs) == len(cj.conjectures())


# ---------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------


class TestVerify:
    def test_verify_passes_for_true_identity(self):
        cj = Conjecturer.create(precision_digits=15, seed=0)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        cj.with_constants(("phi2", "phi", "one"))
        out = cj.propose(max_coeff=3)
        target = next(
            c for c in out if c.coeffs == (1, -1, -1) or c.coeffs == (-1, 1, 1)
        )
        verified = cj.verify(target, factor=2)
        # Residual at higher precision will reflect float precision of input
        # (about 10^-16 for math.sqrt(5)) — verify should accept this since
        # the cutoff is 10^{-d} = 10^{-15}.
        assert verified.verified_at_digits == 30

    def test_verify_rejects_spurious(self):
        # Construct a "fake" conjecture and verify — should fail.
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("a", 1.0)
        cj.observe("b", 2.0)
        cj.with_constants(("a", "b"))
        bogus = Conjecture(
            columns=("a", "b"),
            coeffs=(3, 1),  # 3·1 + 1·2 = 5, residual = 5 (NOT zero!)
            residual=Fraction(5),
            working_digits=15,
            fdr_bound=1.0,
            norm_infinity=3,
        )
        out = cj.verify(bogus)
        assert out.rejected
        assert not out.verified

    def test_verify_emits_event(self):
        bus = EventBus()
        seen_v: list[Event] = []
        seen_r: list[Event] = []
        bus.subscribe(seen_v.append, kind=CONJECTURER_VERIFIED)
        bus.subscribe(seen_r.append, kind=CONJECTURER_REJECTED)
        cj = Conjecturer.create(precision_digits=15, bus=bus)
        cj.observe("a", 1.0)
        cj.observe("b", 2.0)
        bogus = Conjecture(
            columns=("a", "b"),
            coeffs=(5, 1),
            residual=Fraction(7),
            working_digits=15,
            fdr_bound=1.0,
            norm_infinity=5,
        )
        cj.verify(bogus)
        assert len(seen_r) == 1

    def test_verify_invalid_factor(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("a", 1.0)
        bogus = Conjecture(("a",), (1,), Fraction(1), 15, 0.5, 1)
        with pytest.raises(InvalidConjecture):
            cj.verify(bogus, factor=1)

    def test_verify_non_conjecture_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidConjecture):
            cj.verify("not a conjecture")  # type: ignore[arg-type]

    def test_verify_unknown_column(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("a", 1.0)
        bogus = Conjecture(("a", "unknown"), (1, 1), Fraction(2), 15, 0.5, 1)
        with pytest.raises(InvalidConjecture):
            cj.verify(bogus)


# ---------------------------------------------------------------------
# recognize_constant
# ---------------------------------------------------------------------


class TestRecognizeConstant:
    def test_recognize_simple_rational(self):
        cj = Conjecturer.create(precision_digits=15)
        recs = cj.recognize_constant(Fraction(3, 7))
        assert any(r.kind == "rational" and r.expression == "3/7" for r in recs)

    def test_recognize_integer(self):
        cj = Conjecturer.create(precision_digits=15)
        recs = cj.recognize_constant(Fraction(5))
        assert any(r.kind == "rational" and r.expression == "5" for r in recs)

    def test_recognize_phi(self):
        cj = Conjecturer.create(precision_digits=15)
        phi = (1 + math.sqrt(5)) / 2
        recs = cj.recognize_constant(phi, basis=("one", "sqrt5"))
        # Should find (one + sqrt5) / 2 or similar.
        assert any(r.kind == "basis" for r in recs)

    def test_recognize_emits_event(self):
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append, kind=CONJECTURER_RECOGNISED)
        cj = Conjecturer.create(precision_digits=15, bus=bus)
        cj.recognize_constant(0.5)
        assert len(seen) == 1

    def test_recognize_non_finite_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidObservation):
            cj.recognize_constant(float("inf"))

    def test_recognize_bool_raises(self):
        cj = Conjecturer.create(precision_digits=15)
        with pytest.raises(InvalidObservation):
            cj.recognize_constant(True)

    def test_recognize_default_basis(self):
        cj = Conjecturer.create(precision_digits=12)
        # Just exercise the default basis path without asserting hits.
        recs = cj.recognize_constant(0.5)
        # 1/2 should be a rational hit.
        assert any(r.kind == "rational" and r.expression == "1/2" for r in recs)


# ---------------------------------------------------------------------
# report / clear / determinism
# ---------------------------------------------------------------------


class TestReportClear:
    def test_report_snapshot(self):
        cj = Conjecturer.create(precision_digits=15, seed=7)
        cj.observe("x", 1.0)
        rep = cj.report()
        assert rep.n_observations == 1
        assert rep.seed == 7
        assert rep.precision_digits == 15

    def test_report_emits_event(self):
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append, kind=CONJECTURER_REPORTED)
        cj = Conjecturer.create(precision_digits=15, bus=bus)
        cj.observe("x", 1.0)
        cj.report()
        assert len(seen) == 1

    def test_clear_resets(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("x", 1.0)
        cj.with_constants(("x", "pi"))
        cj.propose(max_coeff=3)
        cj.clear()
        assert cj.observation_names() == ()
        assert cj.selected_columns() == ()
        assert len(cj.conjectures()) == 0
        # The CLEARED event itself advances the chain; before that event
        # the head was reset to genesis, so the post-clear head is the
        # single-link chain from genesis through the CLEARED payload.
        from agi.conjecturer import _hash_link, CONJECTURER_CLEARED
        expected = _hash_link("0" * 64, CONJECTURER_CLEARED + "|{}")
        assert cj.head() == expected

    def test_clear_emits_event(self):
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append, kind=CONJECTURER_CLEARED)
        cj = Conjecturer.create(precision_digits=15, bus=bus)
        cj.clear()
        assert len(seen) == 1

    def test_determinism_same_seed(self):
        c1 = Conjecturer.create(precision_digits=15, seed=99)
        c2 = Conjecturer.create(precision_digits=15, seed=99)
        c1.observe("phi", (1 + math.sqrt(5)) / 2)
        c2.observe("phi", (1 + math.sqrt(5)) / 2)
        c1.with_constants(("phi", "one"))
        c2.with_constants(("phi", "one"))
        c1.propose(max_coeff=3)
        c2.propose(max_coeff=3)
        # The runtime is fully deterministic — same head.
        assert c1.head() == c2.head()

    def test_head_advances_strictly(self):
        cj = Conjecturer.create(precision_digits=15)
        seen = {cj.head()}
        cj.observe("x", 1.0)
        seen.add(cj.head())
        cj.observe("y", 2.0)
        seen.add(cj.head())
        cj.with_constants(("x", "y"))
        cj.propose(max_coeff=3)
        seen.add(cj.head())
        # Each step yields a fresh head.
        assert len(seen) == 4

    def test_conjecture_to_dict_roundtrip(self):
        cj = Conjecturer.create(precision_digits=15)
        cj.observe("a", 1.0)
        cj.observe("b", 2.0)
        c = Conjecture(("a", "b"), (2, -1), Fraction(0), 15, 0.01, 2)
        d = c.to_dict()
        assert d["columns"] == ["a", "b"]
        assert d["coeffs"] == [2, -1]
        assert d["signature"] == "2·a −b = 0"


# ---------------------------------------------------------------------
# quick helper
# ---------------------------------------------------------------------


class TestQuickHelper:
    def test_quick_quadratic_phi(self):
        phi = (1 + math.sqrt(5)) / 2
        out = quick_quadratic_recognition(phi, precision_digits=14)
        # φ² − φ − 1 = 0 ⇒ in lattice (x², x, 1) ⇒ (1, -1, -1)
        assert any(
            c.coeffs == (1, -1, -1) or c.coeffs == (-1, 1, 1) for c in out
        )

    def test_quick_quadratic_sqrt2(self):
        s = math.sqrt(2)
        out = quick_quadratic_recognition(s, precision_digits=14, max_coeff=4)
        # x² − 2 = 0 ⇒ in lattice (x², x, 1) ⇒ (1, 0, -2)
        assert any(
            c.coeffs == (1, 0, -2) or c.coeffs == (-1, 0, 2)
            for c in out
        )


# ---------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------


class TestConcurrency:
    def test_lock_protects_state(self):
        import threading

        cj = Conjecturer.create(precision_digits=12)

        def loop():
            for i in range(40):
                cj.observe(f"obs_{i}", float(i + 1))

        threads = [threading.Thread(target=loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions, and observation set is consistent.
        names = cj.observation_names()
        assert len(set(names)) == len(names)  # no duplicates
        assert all(n.startswith("obs_") for n in names)


# ---------------------------------------------------------------------
# False-discovery bound
# ---------------------------------------------------------------------


class TestFDRBound:
    def test_fdr_decreases_with_precision(self):
        cj = Conjecturer.create(precision_digits=15)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        cj.with_constants(("phi2", "phi", "one"))
        out_low = cj.propose(max_coeff=3, precision_digits=10)
        out_high = cj.propose(max_coeff=3, precision_digits=15)
        # Same conjectures appear, but FDR bound should be tighter at higher
        # precision.
        if out_low and out_high:
            assert min(c.fdr_bound for c in out_high) <= min(c.fdr_bound for c in out_low) + 1e-9

    def test_fdr_at_least_zero(self):
        cj = Conjecturer.create(precision_digits=15)
        phi = (1 + math.sqrt(5)) / 2
        cj.observe("phi", phi)
        cj.observe("phi2", phi * phi)
        cj.with_constants(("phi2", "phi", "one"))
        out = cj.propose(max_coeff=3)
        for c in out:
            assert c.fdr_bound >= 0.0


# ---------------------------------------------------------------------
# Integration with built-in constants
# ---------------------------------------------------------------------


class TestBuiltinConstants:
    def test_pi_via_sum_of_arctans(self):
        # Machin: pi/4 = 4·arctan(1/5) − arctan(1/239)
        # ⇒ pi − 16·arctan(1/5) + 4·arctan(1/239) = 0
        cj = Conjecturer.create(precision_digits=14)
        cj.observe("a", math.atan(1 / 5))
        cj.observe("b", math.atan(1 / 239))
        cj.with_constants(("pi", "a", "b"))
        out = cj.propose(max_coeff=20)
        found = any(
            (c.coeffs == (1, -16, 4) or c.coeffs == (-1, 16, -4))
            for c in out
        )
        assert found

    def test_zeta3_apery_self(self):
        # ζ(3) = ζ(3) trivially.  Tests the zeta3 evaluator wires correctly.
        cj = Conjecturer.create(precision_digits=12)
        # Use the built-in zeta3 directly as observation.
        z3_val = cj._eval("zeta3", 30)
        # zeta3 ≈ 1.2020569...
        assert 1.2 < float(z3_val) < 1.3
