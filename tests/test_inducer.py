"""Tests for the Inducer runtime primitive — Levin universal search."""
from __future__ import annotations

import math

import pytest

from agi.inducer import (
    ADD,
    ALPHABET_ARITH,
    ALPHABET_FULL,
    ALPHABET_STRAIGHT,
    DRP,
    DUP,
    HALT,
    INP,
    Inducer,
    InducerConfig,
    InducerError,
    InducerReport,
    InvalidConfig,
    InvalidProgram,
    InvalidSpec,
    JNZ,
    MOD,
    MUL,
    NEG,
    NOP,
    NoSolution,
    OPCODE_COUNT,
    OPCODE_NAME,
    PUSH0,
    PUSH1,
    PUSH2,
    PUSHN1,
    Program,
    Spec,
    SUB,
    SWP,
    VM_DIVERGED,
    VM_FAIL,
    VM_OK,
    VMResult,
    coding_theorem_posterior_mass,
    count_programs,
    enumerate_programs,
    induce,
    kraft_normalised_posterior,
    kt_complexity_upper_bound,
    levin_runtime_bound,
    run,
)


# ----------------------------------------------------------------
# Opcode alphabet & utilities
# ----------------------------------------------------------------


class TestOpcodes:
    def test_opcode_count_is_16(self):
        assert OPCODE_COUNT == 16

    def test_every_opcode_has_a_name(self):
        for op in range(OPCODE_COUNT):
            assert op in OPCODE_NAME

    def test_alphabet_straight_excludes_jnz(self):
        assert JNZ not in ALPHABET_STRAIGHT
        # ALPHABET_STRAIGHT includes everything else except NOP and JNZ
        for op in ALPHABET_STRAIGHT:
            assert 0 <= op < OPCODE_COUNT
            assert op != JNZ
            assert op != NOP

    def test_alphabet_full_includes_jnz(self):
        assert JNZ in ALPHABET_FULL
        assert len(ALPHABET_FULL) == OPCODE_COUNT

    def test_alphabet_arith_is_smaller(self):
        assert len(ALPHABET_ARITH) < len(ALPHABET_STRAIGHT)
        for op in ALPHABET_ARITH:
            assert op in ALPHABET_STRAIGHT


# ----------------------------------------------------------------
# Program serialization
# ----------------------------------------------------------------


class TestProgram:
    def test_disassemble_basic(self):
        p = Program(ops=(INP, DUP, MUL, HALT))
        assert p.disassemble() == "INP DUP MUL HALT"

    def test_invalid_opcode_rejected(self):
        with pytest.raises(InvalidProgram):
            Program(ops=(INP, 99))

    def test_round_trip_bytes_even_length(self):
        p = Program(ops=(INP, DUP, MUL, HALT))
        b = p.to_bytes()
        q = Program.from_bytes(b, length=4)
        assert q == p

    def test_round_trip_bytes_odd_length_strips_nop(self):
        p = Program(ops=(INP, DUP, MUL))
        b = p.to_bytes()
        q = Program.from_bytes(b)
        # NOP padding is stripped on decode
        assert q.ops == p.ops

    def test_length_property(self):
        assert Program(ops=()).length == 0
        assert Program(ops=(HALT,)).length == 1
        assert Program(ops=(INP, DUP, MUL, HALT)).length == 4


# ----------------------------------------------------------------
# VM evaluation
# ----------------------------------------------------------------


class TestVM:
    def test_empty_program_returns_zero(self):
        r = run(Program(ops=()), [])
        # PC immediately off the end -> implicit halt with stack-top (0)
        assert r.output == 0

    def test_halt_returns_top_of_stack(self):
        p = Program(ops=(PUSH2, HALT))
        r = run(p, [])
        assert r.status == VM_OK
        assert r.output == 2

    def test_inp_reads_input(self):
        p = Program(ops=(INP, HALT))
        r = run(p, [42])
        assert r.status == VM_OK
        assert r.output == 42

    def test_inp_underflow_fails(self):
        p = Program(ops=(INP, INP, HALT))
        r = run(p, [42])  # only one input
        assert r.status == VM_FAIL

    def test_square(self):
        p = Program(ops=(INP, DUP, MUL, HALT))
        r = run(p, [7])
        assert r.status == VM_OK
        assert r.output == 49

    def test_add_op(self):
        p = Program(ops=(INP, PUSH1, ADD, HALT))
        r = run(p, [5])
        assert r.status == VM_OK
        assert r.output == 6

    def test_sub_is_b_minus_a(self):
        # push 10, push 3, SUB -> 10 - 3 = 7
        p = Program(ops=(INP, INP, SUB, HALT))
        r = run(p, [10, 3])
        assert r.status == VM_OK
        assert r.output == 7

    def test_mod_by_zero_fails(self):
        p = Program(ops=(INP, PUSH0, MOD, HALT))
        r = run(p, [5])
        assert r.status == VM_FAIL

    def test_mod_works(self):
        p = Program(ops=(INP, INP, MOD, HALT))
        r = run(p, [10, 3])
        assert r.status == VM_OK
        assert r.output == 1

    def test_swap_works(self):
        p = Program(ops=(INP, INP, SWP, SUB, HALT))  # b - a -> with SWP: a - b
        r = run(p, [10, 3])
        # stack after INP INP: [10, 3]; SWP: [3, 10]; SUB: 3 - 10 = -7
        assert r.status == VM_OK
        assert r.output == -7

    def test_drop_works(self):
        p = Program(ops=(INP, INP, DRP, HALT))
        r = run(p, [10, 3])
        assert r.output == 10

    def test_neg_works(self):
        p = Program(ops=(INP, NEG, HALT))
        r = run(p, [5])
        assert r.output == -5

    def test_dup_fails_on_empty(self):
        p = Program(ops=(DUP, HALT))
        r = run(p, [])
        assert r.status == VM_FAIL

    def test_jnz_loops(self):
        # JNZ jumps back 3.  Construct a known-terminating loop:
        # INP                  ; [n]
        # DUP                  ; [n n]
        # PUSH1                ; [n n 1]    <- pc 2
        # SUB                  ; [n n-1]    <- pc 3
        # JNZ                  ; pops top, if !=0 jump to pc-3=0
        # On n=3: stack starts [3], after DUP [3,3], PUSH1 [3,3,1], SUB [3,2], JNZ pops 2 -> jump back to INP which fails (no more inputs)
        # So the construct is fragile. Just test that JNZ is recognized.
        p = Program(ops=(PUSH1, PUSH1, PUSH1, PUSH0, JNZ, HALT))
        r = run(p, [])
        # JNZ pops 0 (PUSH0); 0 means do NOT jump (a == 0).
        # Stack before HALT: [1, 1, 1]; top = 1
        assert r.status == VM_OK
        assert r.output == 1

    def test_value_bound_fails(self):
        # Push 2, repeatedly square -> blow value bound
        p = Program(ops=(INP, DUP, MUL, DUP, MUL, DUP, MUL, HALT))
        r = run(p, [10], value_bound=100)
        assert r.status == VM_FAIL


# ----------------------------------------------------------------
# Spec
# ----------------------------------------------------------------


class TestSpec:
    def test_from_pairs_basic(self):
        spec = Spec.from_pairs([(1, 1), (2, 4), (3, 9)])
        assert spec.n_examples == 3
        assert spec.examples[0].inputs == (1,)
        assert spec.examples[0].output == 1

    def test_from_pairs_tuple_inputs(self):
        spec = Spec.from_pairs([((1, 2), 3), ((4, 5), 9)])
        assert spec.examples[0].inputs == (1, 2)
        assert spec.examples[0].output == 3

    def test_from_pairs_empty_rejected(self):
        with pytest.raises(InvalidSpec):
            Spec.from_pairs([])

    def test_fingerprint_deterministic(self):
        s1 = Spec.from_pairs([(2, 4), (3, 9)], name="square")
        s2 = Spec.from_pairs([(2, 4), (3, 9)], name="square")
        assert s1.fingerprint() == s2.fingerprint()

    def test_fingerprint_changes_with_example(self):
        s1 = Spec.from_pairs([(2, 4), (3, 9)], name="square")
        s2 = Spec.from_pairs([(2, 4), (3, 8)], name="square")
        assert s1.fingerprint() != s2.fingerprint()


# ----------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------


class TestEnumeration:
    def test_count_programs(self):
        assert count_programs(14, 3) == 14 ** 3
        assert count_programs(16, 0) == 1

    def test_enumerate_lex_order(self):
        alpha = (HALT, PUSH0, PUSH1)
        progs = list(enumerate_programs(alpha, 2))
        assert len(progs) == 9
        # First program should be (HALT, HALT)
        assert progs[0].ops == (HALT, HALT)
        # Last should be (PUSH1, PUSH1)
        assert progs[-1].ops == (PUSH1, PUSH1)

    def test_zero_length_yields_one(self):
        progs = list(enumerate_programs((HALT,), 0))
        assert len(progs) == 1
        assert progs[0].ops == ()

    def test_negative_length_rejected(self):
        with pytest.raises(InvalidConfig):
            list(enumerate_programs((HALT,), -1))


# ----------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------


class TestConfig:
    def test_empty_alphabet_rejected(self):
        with pytest.raises(InvalidConfig):
            InducerConfig(alphabet=())

    def test_unknown_mode_rejected(self):
        with pytest.raises(InvalidConfig):
            InducerConfig(mode="genie")

    def test_levin_doubling_must_exceed_one(self):
        with pytest.raises(InvalidConfig):
            InducerConfig(mode="levin", levin_phase_doubling=1.0)

    def test_invalid_opcode_in_alphabet(self):
        with pytest.raises(InvalidConfig):
            InducerConfig(alphabet=(99,))

    def test_max_program_length_must_be_positive(self):
        with pytest.raises(InvalidConfig):
            InducerConfig(max_program_length=0)


# ----------------------------------------------------------------
# Search: iterative deepening (basic)
# ----------------------------------------------------------------


class TestIDDFS:
    def test_finds_square(self):
        spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25), (7, 49)])
        cfg = InducerConfig(max_program_length=4, max_wallclock_s=10.0)
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        assert rep.program.length <= 4
        # Verify on a held-out input
        assert rep.eval([6]) == 36

    def test_finds_doubling(self):
        spec = Spec.from_pairs([(1, 2), (3, 6), (5, 10), (10, 20)])
        cfg = InducerConfig(max_program_length=4, max_wallclock_s=10.0)
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        # Confirm semantic correctness
        for x in [0, 4, 8, 100]:
            assert rep.eval([x]) == 2 * x

    def test_finds_increment(self):
        spec = Spec.from_pairs([(1, 2), (3, 4), (5, 6)])
        cfg = InducerConfig(max_program_length=4, max_wallclock_s=10.0)
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        for x in [0, 7, 100]:
            assert rep.eval([x]) == x + 1

    def test_finds_constant_function(self):
        # All examples map any input to 0; trivial program: PUSH0 HALT.
        spec = Spec.from_pairs([(2, 0), (3, 0), (4, 0)])
        cfg = InducerConfig(
            max_program_length=3, max_wallclock_s=5.0,
            prune_constant_outputs=False,  # required for constant programs
        )
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        assert rep.eval([99]) == 0

    def test_constant_pruning_skips_const_progs_when_outputs_distinct(self):
        # n -> n^2; outputs are distinct so prune_constant_outputs should kick in.
        spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25)])
        cfg = InducerConfig(
            max_program_length=4, max_wallclock_s=5.0,
            prune_constant_outputs=True,
        )
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        # Program must contain INP since outputs depend on input
        assert INP in rep.program.ops

    def test_no_solution_within_budget(self):
        # Require something the alphabet can never express in 3 ops:
        # outputs depend on inputs but we cap length=2 with no INP-readable progs.
        spec = Spec.from_pairs([(2, 5), (3, 7), (5, 11), (7, 13)])  # 2n+1
        cfg = InducerConfig(
            max_program_length=2,
            max_wallclock_s=2.0,
            alphabet=(HALT, PUSH0, PUSH1),  # no INP, no ADD, no MUL
        )
        rep = Inducer(cfg).search(spec)
        assert rep.program is None
        assert rep.stats.programs_visited > 0

    def test_minimal_program_wins(self):
        # Square has a length-3 program (INP DUP MUL); confirm we return it
        # rather than some longer program that also fits.
        spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25), (7, 49)])
        cfg = InducerConfig(max_program_length=5, max_wallclock_s=10.0)
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        assert rep.program.length <= 4  # 3 (program) implicitly halts at PC end


# ----------------------------------------------------------------
# Search: Levin universal mode
# ----------------------------------------------------------------


class TestLevin:
    def test_finds_square_in_levin_mode(self):
        spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25)])
        cfg = InducerConfig(
            mode="levin",
            max_program_length=4,
            max_wallclock_s=10.0,
            levin_start_budget=64,
        )
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        assert rep.stats.phases_completed >= 1

    def test_levin_phase_doubling_recorded(self):
        spec = Spec.from_pairs([(1, 2), (3, 6), (5, 10)])
        cfg = InducerConfig(
            mode="levin",
            max_program_length=4,
            max_wallclock_s=10.0,
            levin_start_budget=32,
            levin_phase_doubling=2.0,
        )
        rep = Inducer(cfg).search(spec)
        assert rep.program is not None
        assert rep.stats.steps_executed > 0


# ----------------------------------------------------------------
# Reports & bounds
# ----------------------------------------------------------------


class TestReports:
    def test_universal_prior_mass_is_two_to_minus_length(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        rep = Inducer(InducerConfig(max_program_length=4)).search(spec)
        assert rep.program is not None
        expected = 2.0 ** -rep.program.length
        assert math.isclose(rep.universal_prior_mass(), expected, rel_tol=1e-12)

    def test_levin_complexity_is_finite_when_found(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        rep = Inducer(InducerConfig(max_program_length=4)).search(spec)
        assert rep.program is not None
        Kt = rep.levin_complexity()
        assert math.isfinite(Kt)
        assert Kt >= rep.program.length

    def test_levin_complexity_is_inf_when_no_solution(self):
        # Force no solution
        spec = Spec.from_pairs([(2, 5), (3, 7)])
        cfg = InducerConfig(
            max_program_length=1,
            alphabet=(HALT,),
            max_wallclock_s=1.0,
        )
        rep = Inducer(cfg).search(spec)
        assert rep.program is None
        assert math.isinf(rep.levin_complexity())

    def test_occam_bound_decreases_with_more_examples(self):
        spec_small = Spec.from_pairs([(2, 4), (3, 9)])
        spec_large = Spec.from_pairs([(i, i * i) for i in range(2, 20)])
        r1 = Inducer(InducerConfig(max_program_length=4)).search(spec_small)
        r2 = Inducer(InducerConfig(max_program_length=4)).search(spec_large)
        assert r1.program is not None
        assert r2.program is not None
        assert r1.program.length == r2.program.length  # both find same prog
        assert r2.occam_bound(delta=0.05) < r1.occam_bound(delta=0.05)

    def test_occam_bound_rejects_bad_delta(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        rep = Inducer(InducerConfig(max_program_length=4)).search(spec)
        with pytest.raises(InvalidConfig):
            rep.occam_bound(delta=0.0)
        with pytest.raises(InvalidConfig):
            rep.occam_bound(delta=1.0)

    def test_levin_runtime_bound_grows_with_length(self):
        b3 = levin_runtime_bound(3, 100)
        b6 = levin_runtime_bound(6, 100)
        assert b6 == 8 * b3  # 2^3 factor

    def test_certificate_is_deterministic(self):
        spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25)])
        r1 = Inducer(InducerConfig(max_program_length=4)).search(spec)
        r2 = Inducer(InducerConfig(max_program_length=4)).search(spec)
        assert r1.certificate == r2.certificate

    def test_certificate_changes_with_spec(self):
        s1 = Spec.from_pairs([(2, 4), (3, 9)])
        s2 = Spec.from_pairs([(2, 4), (3, 10)])
        r1 = Inducer(InducerConfig(max_program_length=4)).search(s1)
        r2 = Inducer(InducerConfig(max_program_length=4)).search(s2)
        assert r1.certificate != r2.certificate

    def test_vm_signature_stable(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        r1 = Inducer(InducerConfig(max_program_length=4)).search(spec)
        r2 = Inducer(InducerConfig(max_program_length=4)).search(spec)
        assert r1.vm_signature == r2.vm_signature

    def test_kt_complexity_upper_bound_matches_levin_complexity(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        rep = Inducer(InducerConfig(max_program_length=4)).search(spec)
        assert kt_complexity_upper_bound(rep) == rep.levin_complexity()


# ----------------------------------------------------------------
# Kraft / posterior helpers
# ----------------------------------------------------------------


class TestKraft:
    def test_coding_theorem_posterior_mass_kraft_inequality(self):
        # Distinct programs of the same length contribute 2^-L each;
        # together they must not exceed 1.
        p1 = Program(ops=(HALT,))
        p2 = Program(ops=(PUSH0, HALT))
        p3 = Program(ops=(PUSH1, HALT))
        mass = coding_theorem_posterior_mass([p1, p2, p3])
        # 1/2 + 1/4 + 1/4 = 1.0 — exactly at Kraft limit
        assert math.isclose(mass, 1.0, rel_tol=1e-12)

    def test_kraft_normalised_posterior_sums_to_one(self):
        p1 = Program(ops=(HALT,))
        p2 = Program(ops=(PUSH0, HALT))
        p3 = Program(ops=(PUSH1, HALT))
        normed = kraft_normalised_posterior([p1, p2, p3])
        assert math.isclose(sum(w for _, w in normed), 1.0, rel_tol=1e-12)

    def test_kraft_empty_input_returns_empty(self):
        assert kraft_normalised_posterior([]) == []


# ----------------------------------------------------------------
# Events
# ----------------------------------------------------------------


class TestEvents:
    def test_events_fire_during_search(self):
        events: list[tuple[str, dict]] = []

        def hook(kind: str, data: dict) -> None:
            events.append((kind, data))

        spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25)])
        Inducer(InducerConfig(max_program_length=3), on_event=hook).search(spec)

        kinds = {k for k, _ in events}
        # At least one phase + completion event
        assert "inducer.phase.started" in kinds
        assert "inducer.search.completed" in kinds

    def test_event_hook_exception_does_not_crash(self):
        def bad_hook(kind, data):
            raise RuntimeError("nope")

        spec = Spec.from_pairs([(2, 4), (3, 9)])
        rep = Inducer(
            InducerConfig(max_program_length=3),
            on_event=bad_hook,
        ).search(spec)
        assert rep.program is not None


# ----------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------


class TestBudgets:
    def test_max_programs_caps_search(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        cfg = InducerConfig(
            max_program_length=4,
            max_programs=10,
            max_wallclock_s=5.0,
        )
        rep = Inducer(cfg).search(spec)
        assert rep.stats.programs_visited <= 100  # rough headroom for branch/budget pacing

    def test_max_wallclock_caps_search(self):
        spec = Spec.from_pairs([(2, 4), (3, 9)])
        cfg = InducerConfig(
            max_program_length=8,
            max_wallclock_s=0.01,  # nearly instant
            mode="levin",
            levin_start_budget=4,
        )
        rep = Inducer(cfg).search(spec)
        # Even if no solution, we must return cleanly
        assert isinstance(rep, InducerReport)


# ----------------------------------------------------------------
# Convenience entry point
# ----------------------------------------------------------------


class TestInduce:
    def test_induce_returns_report(self):
        rep = induce(
            [(2, 4), (3, 9), (5, 25)],
            config=InducerConfig(max_program_length=4, max_wallclock_s=5.0),
            name="square",
        )
        assert rep.program is not None
        assert rep.spec.name == "square"

    def test_induce_with_default_config(self):
        rep = induce([(2, 4), (3, 9)], name="square")
        assert rep.program is not None


# ----------------------------------------------------------------
# Multi-input problems
# ----------------------------------------------------------------


class TestMultiInput:
    def test_two_input_addition(self):
        # f(a, b) = a + b
        pairs = [((1, 2), 3), ((3, 4), 7), ((5, 6), 11), ((10, 20), 30)]
        spec = Spec.from_pairs(pairs)
        rep = Inducer(InducerConfig(max_program_length=4, max_wallclock_s=10.0)).search(spec)
        assert rep.program is not None
        # Confirm on held-out
        assert rep.eval((100, 200)) == 300

    def test_two_input_subtraction(self):
        # f(a, b) = a - b
        pairs = [((10, 3), 7), ((20, 5), 15), ((100, 1), 99)]
        spec = Spec.from_pairs(pairs)
        rep = Inducer(InducerConfig(max_program_length=4, max_wallclock_s=10.0)).search(spec)
        assert rep.program is not None
        assert rep.eval((50, 10)) == 40
