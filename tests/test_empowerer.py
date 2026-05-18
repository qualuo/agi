"""Tests for the Empowerer primitive — empowerment / intrinsic motivation."""
from __future__ import annotations

import json
import math
import random

import pytest

from agi.empowerer import (
    EMPOWERER_CERTIFIED,
    EMPOWERER_FITTED,
    EMPOWERER_OBSERVED,
    EMPOWERER_REPORTED,
    EMPOWERER_RESET,
    EMPOWERER_REWARDED,
    EMPOWERER_SHIELDED,
    EMPOWERER_SKILLS_DISCOVERED,
    EMPOWERER_SOLVED,
    EMPOWERER_STARTED,
    EMPOWERER_VARIATIONAL,
    EST_BLAHUT_ARIMOTO,
    EST_VARIATIONAL_DV,
    EST_VARIATIONAL_INFONCE,
    EST_VARIATIONAL_NWJ,
    BlowupGuard,
    Empowerer,
    EmpowererConfig,
    EmpowererError,
    InsufficientData,
    InvalidAction,
    InvalidConfig,
    InvalidHorizon,
    InvalidState,
    InvalidTransition,
    REWARD_DELTA_EMPOWERMENT,
    REWARD_STATE_EMPOWERMENT,
    REWARD_TRANSITION_SURPRISE,
    UnknownEstimator,
    UnknownRewardMode,
    blahut_arimoto_capacity,
    diayn_intrinsic_reward,
    donsker_varadhan_lower_bound,
    empowerer_ledger_root,
    infonce_lower_bound,
    n_step_kernel,
    nwj_lower_bound,
    paninski_empowerment_bound,
)


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


class TestBlahutArimoto:
    def test_perfect_binary_channel_is_one_bit(self):
        # I(A;Y) for the identity channel between binary alphabets = 1 bit.
        channel = [[1.0, 0.0], [0.0, 1.0]]
        cap, alpha, _, conv = blahut_arimoto_capacity(channel)
        assert cap == pytest.approx(1.0, abs=1e-6)
        assert conv
        # Uniform capacity-achieving distribution.
        assert alpha[0] == pytest.approx(0.5, abs=1e-4)
        assert alpha[1] == pytest.approx(0.5, abs=1e-4)

    def test_useless_channel_is_zero_bits(self):
        # Both inputs map to the same output → zero capacity.
        channel = [[1.0, 0.0], [1.0, 0.0]]
        cap, _, _, _ = blahut_arimoto_capacity(channel)
        assert cap == pytest.approx(0.0, abs=1e-9)

    def test_bsc_half_capacity(self):
        # Binary symmetric channel with crossover p=0.1:
        #   C = 1 - H_b(0.1) ≈ 0.531 bits.
        p = 0.1
        channel = [[1.0 - p, p], [p, 1.0 - p]]
        cap, _, _, _ = blahut_arimoto_capacity(channel)
        h_b = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        assert cap == pytest.approx(1.0 - h_b, abs=1e-4)

    def test_warm_start_converges_immediately(self):
        channel = [[1.0, 0.0], [0.0, 1.0]]
        warm = [0.5, 0.5]
        cap, _, iters, conv = blahut_arimoto_capacity(channel, warm_start=warm)
        assert conv
        assert iters <= 2  # warm-started at the optimum.
        assert cap == pytest.approx(1.0, abs=1e-6)

    def test_capacity_is_non_negative(self):
        rng = random.Random(0)
        for _ in range(10):
            # Random row-stochastic 3x3 channel.
            rows = []
            for _ in range(3):
                xs = [rng.random() for _ in range(3)]
                s = sum(xs)
                rows.append([x / s for x in xs])
            cap, _, _, _ = blahut_arimoto_capacity(rows)
            assert cap >= -1e-9

    def test_empty_channel_is_zero(self):
        cap, alpha, _, conv = blahut_arimoto_capacity([])
        assert cap == 0.0
        assert alpha == []
        assert conv


class TestNStepKernel:
    def test_two_step_unrolls_correctly(self):
        # deterministic 2-state, 2-action: a=0 keeps state, a=1 flips.
        one_step = [
            [[1.0, 0.0], [0.0, 1.0]],   # from s=0
            [[0.0, 1.0], [1.0, 0.0]],   # from s=1
        ]
        seqs, rows = n_step_kernel(one_step, state=0, horizon=2)
        # 2 actions * 2 actions = 4 sequences
        assert len(seqs) == 4
        # (0,0) → 0; (0,1) → 1; (1,0) → 0 (flipped then back? no, 1 then a=0 keeps) → 1; (1,1) → 0
        m = dict(zip(seqs, rows))
        assert m[(0, 0)] == [1.0, 0.0]
        assert m[(0, 1)] == [0.0, 1.0]
        assert m[(1, 0)] == [0.0, 1.0]   # flip to 1 then stay
        assert m[(1, 1)] == [1.0, 0.0]   # flip then flip back

    def test_blowup_guard(self):
        # |A|^horizon = 2^20 way over the cap.
        one_step = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        with pytest.raises(BlowupGuard):
            n_step_kernel(one_step, state=0, horizon=20, max_action_seqs=10)


class TestInfoNCEBound:
    def test_uniform_scores_yield_zero(self):
        # Equal scores everywhere → no contrast → 0 bits.
        scores = [[0.0, 0.0], [0.0, 0.0]]
        b = infonce_lower_bound(scores)
        assert b == pytest.approx(0.0, abs=1e-9)

    def test_diagonal_dominance_gives_positive_bound(self):
        # Diagonal much higher than off-diagonal.
        scores = [[10.0, 0.0], [0.0, 10.0]]
        b = infonce_lower_bound(scores)
        assert b > 0.0
        # Bound is at most log2(K) = 1 bit for K=2.
        assert b <= 1.0 + 1e-6

    def test_one_sample_is_zero(self):
        # Cannot contrast against negatives.
        b = infonce_lower_bound([[5.0]])
        assert b == 0.0


class TestNWJBound:
    def test_nonnegative_typical_case(self):
        # Positives all 0.0, negatives all 0.0 → e^{0-1} = 1/e per neg.
        b = nwj_lower_bound([0.0, 0.0], [0.0, 0.0])
        expected = (0.0 - math.exp(-1.0)) / math.log(2.0)
        assert b == pytest.approx(expected, abs=1e-9)


class TestDVBound:
    def test_zero_for_identical_distributions(self):
        b = donsker_varadhan_lower_bound([0.0, 0.0], [0.0, 0.0])
        # E[f]=0, log(E[e^f])=log(1)=0 → 0 bits.
        assert b == pytest.approx(0.0, abs=1e-9)


class TestDiaynReward:
    def test_log_ratio(self):
        # q(z|s)=0.5, p(z)=0.25 → log2(0.5/0.25) = 1 bit.
        r = diayn_intrinsic_reward(math.log(0.5), math.log(0.25))
        assert r == pytest.approx(1.0, abs=1e-9)


class TestPaninskiBound:
    def test_zero_samples_is_infinite(self):
        assert math.isinf(paninski_empowerment_bound(4, 4, 0, 0.95))

    def test_more_samples_tighter_bound(self):
        b_small = paninski_empowerment_bound(4, 4, 10, 0.95)
        b_large = paninski_empowerment_bound(4, 4, 1000, 0.95)
        assert b_large < b_small


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default(self):
        cfg = EmpowererConfig()
        assert cfg.estimator == EST_BLAHUT_ARIMOTO
        assert cfg.horizon == 1

    def test_invalid_estimator(self):
        with pytest.raises(UnknownEstimator):
            EmpowererConfig(estimator="not_a_real_estimator")

    def test_invalid_reward_mode(self):
        with pytest.raises(UnknownRewardMode):
            EmpowererConfig(reward_mode="nope")

    def test_invalid_dim(self):
        with pytest.raises(InvalidConfig):
            EmpowererConfig(dim_state=0)
        with pytest.raises(InvalidConfig):
            EmpowererConfig(dim_action=-1)

    def test_invalid_horizon(self):
        with pytest.raises(InvalidConfig):
            EmpowererConfig(horizon=0)

    def test_invalid_confidence(self):
        with pytest.raises(InvalidConfig):
            EmpowererConfig(confidence=1.0)
        with pytest.raises(InvalidConfig):
            EmpowererConfig(confidence=0.0)

    def test_invalid_safety_estimator(self):
        with pytest.raises(InvalidConfig):
            EmpowererConfig(safety_estimator="bogus")

    def test_nan_inf_rejected(self):
        with pytest.raises(InvalidConfig):
            EmpowererConfig(ba_tol=float("nan"))
        with pytest.raises(InvalidConfig):
            EmpowererConfig(safety_margin=float("inf"))


# ---------------------------------------------------------------------------
# Construction and ledger
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_chain_starts_after_genesis(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        # After STARTED block.
        assert em.chain_head != empowerer_ledger_root()

    def test_observer_called(self):
        events = []
        em = Empowerer(
            EmpowererConfig(dim_state=2, dim_action=2),
            observer=lambda k, p: events.append((k, p)),
        )
        em.observe_transition(0, 0, 1)
        kinds = [e[0] for e in events]
        assert EMPOWERER_STARTED in kinds
        assert EMPOWERER_OBSERVED in kinds

    def test_observer_exceptions_are_swallowed(self):
        def bad_observer(k, p):
            raise RuntimeError("observer is broken")

        em = Empowerer(
            EmpowererConfig(dim_state=2, dim_action=2),
            observer=bad_observer,
        )
        # Should not raise.
        em.observe_transition(0, 0, 1)

    def test_chain_advances_with_each_observation(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        h0 = em.chain_head
        em.observe_transition(0, 0, 1)
        h1 = em.chain_head
        em.observe_transition(1, 1, 0)
        h2 = em.chain_head
        assert h0 != h1 != h2

    def test_hmac_key_changes_chain(self):
        em_unkey = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        em_keyed = Empowerer(
            EmpowererConfig(dim_state=2, dim_action=2, hmac_key=b"secret")
        )
        em_unkey.observe_transition(0, 0, 1)
        em_keyed.observe_transition(0, 0, 1)
        # The exact head differs because the seed digests do.
        assert em_unkey.chain_head != em_keyed.chain_head


# ---------------------------------------------------------------------------
# Ingest validation
# ---------------------------------------------------------------------------


class TestValidation:
    def setup_method(self):
        self.em = Empowerer(EmpowererConfig(dim_state=3, dim_action=2))

    def test_invalid_state(self):
        with pytest.raises(InvalidState):
            self.em.observe_transition(5, 0, 0)
        with pytest.raises(InvalidState):
            self.em.observe_transition(0, 0, -1)
        with pytest.raises(InvalidState):
            self.em.observe_transition("a", 0, 0)  # type: ignore[arg-type]

    def test_invalid_action(self):
        with pytest.raises(InvalidAction):
            self.em.observe_transition(0, 7, 0)
        with pytest.raises(InvalidAction):
            self.em.observe_transition(0, -1, 0)

    def test_invalid_horizon(self):
        with pytest.raises(InvalidHorizon):
            self.em.n_step_empowerment(0, 0)
        with pytest.raises(InvalidHorizon):
            self.em.n_step_empowerment(0, -3)

    def test_batch_ingest_validates(self):
        with pytest.raises(InvalidTransition):
            self.em.fit_transitions([(0, 0)])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Empowerment computation
# ---------------------------------------------------------------------------


class TestEmpowerment:
    def test_perfectly_controllable_two_state(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2, laplace_alpha=0.0))
        for _ in range(50):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        r = em.empowerment(0)
        assert r.empowerment_bits == pytest.approx(1.0, abs=1e-4)
        assert r.converged
        assert r.optimal_action in (0, 1)

    def test_no_control_zero_bits(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2, laplace_alpha=0.0))
        for _ in range(50):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 0)  # same outcome regardless of action
        r = em.empowerment(0)
        assert r.empowerment_bits == pytest.approx(0.0, abs=1e-6)

    def test_three_action_full_control_log2_3(self):
        em = Empowerer(EmpowererConfig(dim_state=3, dim_action=3, laplace_alpha=0.0))
        for _ in range(100):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
            em.observe_transition(0, 2, 2)
        r = em.empowerment(0)
        assert r.empowerment_bits == pytest.approx(math.log2(3.0), abs=1e-4)

    def test_n_step_empowerment_aggregates(self):
        # |A|=2, |S|=4 cyclic.  2-step gives 1 bit (3 sequences map to 3 states
        # but only 2 distinct outcomes: +/- 2 and stay).
        em = Empowerer(EmpowererConfig(dim_state=4, dim_action=2, horizon=2, laplace_alpha=0.0))
        for s in range(4):
            for _ in range(20):
                em.observe_transition(s, 0, (s - 1) % 4)
                em.observe_transition(s, 1, (s + 1) % 4)
        e1 = em.n_step_empowerment(0, 1)
        e2 = em.n_step_empowerment(0, 2)
        # 1-step controllability is binary (E≈1), 2-step doesn't improve here.
        assert e1 == pytest.approx(1.0, abs=1e-3)
        assert e2 >= e1 - 1e-6

    def test_horizon_one_uses_one_step_kernel(self):
        em = Empowerer(EmpowererConfig(dim_state=3, dim_action=2, horizon=1, laplace_alpha=0.0))
        for _ in range(20):
            em.observe_transition(0, 0, 1)
            em.observe_transition(0, 1, 2)
        r = em.empowerment(0)
        # 2 reachable states with equal-probability optimal policy → 1 bit.
        assert r.empowerment_bits == pytest.approx(1.0, abs=1e-4)

    def test_warm_start_reuses_alpha(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2, ba_warm_start=True, laplace_alpha=0.0))
        for _ in range(50):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        em.empowerment(0)
        r = em.empowerment(0)
        # On second call, warm start should converge faster.
        assert r.iterations <= 3

    def test_blowup_guard_raises(self):
        em = Empowerer(
            EmpowererConfig(
                dim_state=4,
                dim_action=8,
                horizon=10,
                max_action_seqs=100,
            )
        )
        em.observe_transition(0, 0, 0)
        with pytest.raises(BlowupGuard):
            em.empowerment(0)


# ---------------------------------------------------------------------------
# Variational empowerment
# ---------------------------------------------------------------------------


class TestVariational:
    def test_lower_bound_below_capacity(self):
        em = Empowerer(
            EmpowererConfig(
                dim_state=2,
                dim_action=2,
                estimator=EST_VARIATIONAL_INFONCE,
                variational_samples=32,
                laplace_alpha=0.0,
            )
        )
        # Perfectly controllable: true empowerment = 1 bit.
        for _ in range(200):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        # Burn-in: train the decoder.
        for _ in range(50):
            em.variational_empowerment(0)
        vr = em.variational_empowerment(0)
        # InfoNCE bound is at most log2 K.
        cap = math.log2(32)
        assert vr.lower_bound_bits <= cap + 1e-6
        # And at most the true 1 bit (within sample noise).
        assert vr.lower_bound_bits <= 1.0 + 0.5

    def test_nwj_estimator(self):
        em = Empowerer(
            EmpowererConfig(
                dim_state=2,
                dim_action=2,
                estimator=EST_VARIATIONAL_NWJ,
                variational_samples=16,
                laplace_alpha=0.0,
            )
        )
        for _ in range(50):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        vr = em.variational_empowerment(0)
        assert vr.estimator == EST_VARIATIONAL_NWJ

    def test_dv_estimator(self):
        em = Empowerer(
            EmpowererConfig(
                dim_state=2,
                dim_action=2,
                estimator=EST_VARIATIONAL_DV,
                variational_samples=16,
                laplace_alpha=0.0,
            )
        )
        for _ in range(50):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        vr = em.variational_empowerment(0)
        assert vr.estimator == EST_VARIATIONAL_DV

    def test_hoeffding_half_width_decreases(self):
        em = Empowerer(
            EmpowererConfig(
                dim_state=2,
                dim_action=2,
                estimator=EST_VARIATIONAL_INFONCE,
            )
        )
        em.observe_transition(0, 0, 1)
        vr_small = em.variational_empowerment(0, n_samples=8)
        vr_large = em.variational_empowerment(0, n_samples=128)
        assert vr_large.hoeffding_half_width < vr_small.hoeffding_half_width


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------


class TestSkills:
    def test_discovery_smoke(self):
        em = Empowerer(EmpowererConfig(dim_state=6, dim_action=3))
        rng = random.Random(42)
        for _ in range(300):
            s = rng.randrange(6)
            a = rng.randrange(3)
            sp = rng.randrange(6)
            em.observe_transition(s, a, sp)
        sk = em.skill_discovery(n_skills=3, steps=20)
        assert sk.n_skills == 3
        # 3 skills × |S|=6 normalised distribution.
        for row in sk.skill_state_dist:
            assert len(row) == 6
            assert sum(row) == pytest.approx(1.0, abs=1e-6)
            for v in row:
                assert v >= 0.0
        # Entropy should be at most log2 K = log2 3 ≈ 1.585.
        assert sk.skill_entropy <= math.log2(3) + 1e-3

    def test_discovery_requires_observations(self):
        em = Empowerer(EmpowererConfig(dim_state=4, dim_action=2))
        with pytest.raises(InsufficientData):
            em.skill_discovery(n_skills=2)

    def test_too_few_skills_rejected(self):
        em = Empowerer(EmpowererConfig(dim_state=3, dim_action=2))
        em.observe_transition(0, 0, 1)
        with pytest.raises(InvalidConfig):
            em.skill_discovery(n_skills=1)


# ---------------------------------------------------------------------------
# Safety / shielding
# ---------------------------------------------------------------------------


class TestShielding:
    def test_admissibility_filters_low_empowerment_successors(self):
        # 3-state world.  From state 0:
        #   action 0 → state 1 (controllable)
        #   action 1 → state 2 (trap: only self-loops)
        em = Empowerer(EmpowererConfig(dim_state=3, dim_action=2, horizon=1, laplace_alpha=0.0))
        for _ in range(50):
            em.observe_transition(0, 0, 1)
            em.observe_transition(0, 1, 2)
            # state 1 is fully controllable.
            em.observe_transition(1, 0, 0)
            em.observe_transition(1, 1, 2)
            # state 2 is a trap.
            em.observe_transition(2, 0, 2)
            em.observe_transition(2, 1, 2)
        shield = em.safe_actions(0, candidates=(0, 1), margin=0.5)
        # Action 0 leads to a state with empowerment, action 1 to a trap.
        assert 0 in shield.admissible
        assert 1 not in shield.admissible

    def test_min_safety_estimator(self):
        em = Empowerer(
            EmpowererConfig(
                dim_state=3,
                dim_action=2,
                horizon=1,
                laplace_alpha=0.0,
                safety_estimator="min",
                safety_margin=0.0,
            )
        )
        for _ in range(20):
            em.observe_transition(0, 0, 1)
            em.observe_transition(0, 1, 2)
            em.observe_transition(1, 0, 0)
            em.observe_transition(1, 1, 0)
            em.observe_transition(2, 0, 2)
            em.observe_transition(2, 1, 2)
        shield = em.safe_actions(0, candidates=(0, 1), margin=0.0)
        # Trap (state 2) is filtered.
        assert 1 not in shield.admissible


# ---------------------------------------------------------------------------
# Intrinsic reward
# ---------------------------------------------------------------------------


class TestIntrinsicReward:
    def setup_method(self):
        self.em = Empowerer(
            EmpowererConfig(
                dim_state=2,
                dim_action=2,
                horizon=1,
                laplace_alpha=0.0,
                reward_mode=REWARD_STATE_EMPOWERMENT,
            )
        )
        for _ in range(50):
            self.em.observe_transition(0, 0, 0)
            self.em.observe_transition(0, 1, 1)
            self.em.observe_transition(1, 0, 0)
            self.em.observe_transition(1, 1, 1)

    def test_state_empowerment_mode(self):
        r = self.em.intrinsic_reward(0, 0, 1, mode=REWARD_STATE_EMPOWERMENT)
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_delta_empowerment_mode(self):
        # Both states have equal empowerment, so delta = 0.
        r = self.em.intrinsic_reward(0, 0, 1, mode=REWARD_DELTA_EMPOWERMENT)
        assert abs(r) < 1e-3

    def test_transition_surprise_mode(self):
        # In perfect 2-state world the action-conditional transition is
        # deterministic (≈1.0), so −log p ≈ 0.
        r = self.em.intrinsic_reward(0, 1, 1, mode=REWARD_TRANSITION_SURPRISE)
        assert r >= 0.0
        assert r < 1e-3

    def test_unknown_mode_rejected(self):
        with pytest.raises(UnknownRewardMode):
            self.em.intrinsic_reward(0, 0, 1, mode="bogus")

    def test_intrinsic_scale(self):
        em2 = Empowerer(
            EmpowererConfig(
                dim_state=2,
                dim_action=2,
                horizon=1,
                laplace_alpha=0.0,
                reward_mode=REWARD_STATE_EMPOWERMENT,
                intrinsic_scale=5.0,
            )
        )
        for _ in range(50):
            em2.observe_transition(0, 0, 0)
            em2.observe_transition(0, 1, 1)
        r = em2.intrinsic_reward(0, 0, 0)
        assert r == pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Landscape & bottlenecks
# ---------------------------------------------------------------------------


class TestLandscape:
    def test_landscape_covers_visited_states(self):
        em = Empowerer(EmpowererConfig(dim_state=4, dim_action=2, laplace_alpha=0.0))
        em.observe_transition(0, 0, 0)
        em.observe_transition(0, 1, 1)
        em.observe_transition(2, 0, 3)
        em.observe_transition(2, 1, 2)
        land = em.landscape()
        assert set(land.keys()) >= {0, 2}
        for v in land.values():
            assert v >= 0.0

    def test_bottleneck_finds_lowest(self):
        em = Empowerer(EmpowererConfig(dim_state=3, dim_action=2, horizon=1, laplace_alpha=0.0))
        for _ in range(20):
            # state 0 fully controllable; state 1 is a trap.
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
            em.observe_transition(1, 0, 1)
            em.observe_transition(1, 1, 1)
        bts = em.bottleneck_states(top_k=1)
        # state 1 has zero empowerment.
        assert bts == [1]


# ---------------------------------------------------------------------------
# Reports & certificates
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_aggregates(self):
        em = Empowerer(EmpowererConfig(dim_state=3, dim_action=2, laplace_alpha=0.0))
        em.observe_transition(0, 0, 0)
        em.observe_transition(0, 1, 1)
        em.observe_transition(2, 0, 2)
        rep = em.report()
        assert rep.total_transitions == 3
        assert rep.distinct_states == 2  # states 0 and 2 visited
        assert rep.mean_state_empowerment >= 0.0
        assert 0.0 <= rep.state_coverage_fraction <= 1.0

    def test_certificate_holds_when_sampled(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2, laplace_alpha=0.0))
        for _ in range(100):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        cert = em.certify(0)
        assert cert.holds
        assert cert.lower_bound_bits <= cert.empowerment_bits <= cert.upper_bound_bits
        assert cert.confidence > 0

    def test_certificate_does_not_hold_when_unsampled(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        cert = em.certify(0)
        assert not cert.holds
        assert math.isinf(cert.upper_bound_bits) or cert.upper_bound_bits > 1e3


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_round_trip_preserves_empowerment(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2, laplace_alpha=0.0))
        for _ in range(50):
            em.observe_transition(0, 0, 0)
            em.observe_transition(0, 1, 1)
        original = em.empowerment(0).empowerment_bits
        snap = em.snapshot()
        em2 = Empowerer(EmpowererConfig(dim_state=2, dim_action=2, laplace_alpha=0.0))
        em2.restore(snap)
        restored = em2.empowerment(0).empowerment_bits
        assert restored == pytest.approx(original, abs=1e-9)

    def test_round_trip_preserves_chain_head(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        em.observe_transition(0, 0, 1)
        em.observe_transition(0, 1, 0)
        snap = em.snapshot()
        em2 = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        em2.restore(snap)
        assert em2.chain_head == em.chain_head

    def test_snapshot_is_json_serialisable(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        em.observe_transition(0, 0, 1)
        snap = em.snapshot()
        # rng_state is a tuple of tuples — JSON serialiser sees lists on
        # round-trip.  Verify the snapshot is still consumable.
        round_tripped = json.loads(json.dumps(snap, default=list))
        em2 = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        em2.restore(round_tripped)
        assert em2.total_transitions == em.total_transitions

    def test_snapshot_version_mismatch_rejected(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        snap = em.snapshot()
        snap["version"] = "wrong"
        em2 = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        with pytest.raises(InvalidConfig):
            em2.restore(snap)

    def test_snapshot_dim_mismatch_rejected(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        snap = em.snapshot()
        em2 = Empowerer(EmpowererConfig(dim_state=3, dim_action=2))
        with pytest.raises(InvalidConfig):
            em2.restore(snap)


class TestReset:
    def test_reset_clears_counts(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        em.observe_transition(0, 0, 1)
        assert em.total_transitions == 1
        em.reset()
        assert em.total_transitions == 0

    def test_reset_extends_chain(self):
        em = Empowerer(EmpowererConfig(dim_state=2, dim_action=2))
        head_before = em.chain_head
        em.reset()
        assert em.chain_head != head_before


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_runtime_lifecycle(self):
        """The exact lifecycle a coordinator would drive."""
        em = Empowerer(
            EmpowererConfig(
                dim_state=4,
                dim_action=2,
                horizon=2,
                reward_mode=REWARD_DELTA_EMPOWERMENT,
                confidence=0.95,
                safety_margin=0.1,
                laplace_alpha=0.5,
            )
        )
        # 1) Ingest a stream.
        rng = random.Random(7)
        for _ in range(500):
            s = rng.randrange(4)
            a = rng.randrange(2)
            # Slightly controllable: action shifts state.
            sp = (s + a) % 4 if rng.random() < 0.8 else rng.randrange(4)
            em.observe_transition(s, a, sp)
        # 2) Read scalar empowerment.
        r0 = em.empowerment(0)
        assert r0.empowerment_bits >= 0
        # 3) Get intrinsic reward.
        ir = em.intrinsic_reward(0, 0, 1)
        assert isinstance(ir, float)
        # 4) Safe-action filter.
        shield = em.safe_actions(0)
        assert shield.state == 0
        assert all(0 <= a < 2 for a in shield.admissible)
        # 5) Skill discovery.
        sk = em.skill_discovery(n_skills=3, steps=15)
        assert sk.n_skills == 3
        # 6) Certify.
        cert = em.certify(0)
        assert cert.lower_bound_bits <= cert.upper_bound_bits
        # 7) Report.
        rep = em.report()
        assert rep.total_transitions == 500
        # 8) Snapshot and restore.
        snap = em.snapshot()
        em2 = Empowerer(
            EmpowererConfig(
                dim_state=4,
                dim_action=2,
                horizon=2,
                reward_mode=REWARD_DELTA_EMPOWERMENT,
                confidence=0.95,
                safety_margin=0.1,
                laplace_alpha=0.5,
            )
        )
        em2.restore(snap)
        r0_b = em2.empowerment(0)
        assert r0_b.empowerment_bits == pytest.approx(r0.empowerment_bits, abs=1e-6)
