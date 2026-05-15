"""Tests for the Hedger runtime primitive — universal online learning."""
from __future__ import annotations

import math
import random
import pytest

from agi.hedger import (
    ADAHEDGE,
    ANYTIME,
    BERNSTEIN,
    BOA,
    ExpertCertificate,
    FTPL,
    FTRL_ENTROPY,
    FTRL_L2,
    GenericConfigError,
    HEDGE,
    HEDGER_OBSERVED,
    HEDGER_PREDICTED,
    HEDGER_REPORT,
    HEDGER_SELECTED,
    HEDGER_STARTED,
    HOEFFDING,
    Hedger,
    HedgerConfig,
    HedgerError,
    HedgerReport,
    InsufficientData,
    InvalidExperts,
    InvalidLearningRate,
    InvalidLoss,
    InvalidLossRange,
    InvalidPrior,
    KNOWN_ALGORITHMS,
    KNOWN_BOUND_METHODS,
    KNOWN_EVENTS,
    ML_PROD,
    NORMAL_HEDGE,
    OMD_ENTROPY,
    Prediction,
    RegretCertificate,
    Round,
    SQUINT,
    Selection,
    UnknownAlgorithm,
    UnknownBoundMethod,
    adahedge_regret_bound,
    anytime_lcb,
    anytime_ucb,
    boa_regret_bound,
    empirical_bernstein_lcb,
    empirical_bernstein_ucb,
    hedge_minimax_eta,
    hedge_minimax_regret,
    hedge_regret_bound,
    hoeffding_lcb,
    hoeffding_ucb,
    kl_divergence,
    ml_prod_regret_bound,
    normal_hedge_regret_bound,
    pac_bayes_regret_bound,
    squint_regret_bound,
)


# =====================================================================
# Configuration / validation
# =====================================================================


class TestValidation:
    def test_empty_experts_rejected(self):
        with pytest.raises(InvalidExperts):
            Hedger.create([])

    def test_duplicate_experts_rejected(self):
        with pytest.raises(InvalidExperts):
            Hedger.create(["a", "a", "b"])

    def test_unhashable_expert_rejected(self):
        with pytest.raises(InvalidExperts):
            Hedger.create([{"a"}])  # set is unhashable

    def test_unknown_algorithm_rejected(self):
        with pytest.raises(UnknownAlgorithm):
            Hedger.create(["a", "b"], algorithm="not_a_real_algo")

    def test_bad_eta_rejected(self):
        with pytest.raises(InvalidLearningRate):
            Hedger.create(["a", "b"], eta=0.0)
        with pytest.raises(InvalidLearningRate):
            Hedger.create(["a", "b"], eta=-1.0)
        with pytest.raises(InvalidLearningRate):
            Hedger.create(["a", "b"], eta=float("nan"))
        with pytest.raises(InvalidLearningRate):
            Hedger.create(["a", "b"], eta=float("inf"))

    def test_bad_loss_range_rejected(self):
        with pytest.raises(InvalidLossRange):
            Hedger.create(["a", "b"], loss_lower=1.0, loss_upper=0.0)
        with pytest.raises(InvalidLossRange):
            Hedger.create(["a", "b"], loss_lower=0.0, loss_upper=0.0)
        with pytest.raises(InvalidLossRange):
            Hedger.create(["a", "b"], loss_lower=float("nan"), loss_upper=1.0)

    def test_bad_prior_rejected(self):
        with pytest.raises(InvalidPrior):
            Hedger.create(["a", "b"], prior={"a": 1.0, "c": 0.0})  # wrong support
        with pytest.raises(InvalidPrior):
            Hedger.create(["a", "b"], prior={"a": -1.0, "b": 2.0})
        with pytest.raises(InvalidPrior):
            Hedger.create(["a", "b"], prior={"a": 0.0, "b": 0.0})  # zero mass

    def test_bad_horizon_rejected(self):
        with pytest.raises(GenericConfigError):
            Hedger.create(["a", "b"], horizon=0)
        with pytest.raises(GenericConfigError):
            Hedger.create(["a", "b"], horizon=-1)

    def test_bad_ftpl_draws_rejected(self):
        with pytest.raises(GenericConfigError):
            Hedger.create(["a", "b"], ftpl_draws=0)

    def test_loss_missing_expert(self):
        h = Hedger.create(["a", "b", "c"])
        with pytest.raises(InvalidLoss):
            h.observe({"a": 0.5, "b": 0.5})

    def test_loss_nan_rejected(self):
        h = Hedger.create(["a", "b"])
        with pytest.raises(InvalidLoss):
            h.observe({"a": float("nan"), "b": 0.5})

    def test_loss_clipping(self):
        h = Hedger.create(["a", "b"], loss_lower=0.0, loss_upper=1.0)
        # Out-of-range loss is clipped, not rejected.
        rd = h.observe({"a": 2.0, "b": -1.0})
        assert rd.clipped_losses["a"] == 1.0
        assert rd.clipped_losses["b"] == 0.0


# =====================================================================
# Closed-form regret bounds
# =====================================================================


class TestClosedFormBounds:
    def test_hedge_minimax_regret_monotone_T(self):
        N = 5
        prev = 0.0
        for T in [1, 10, 100, 1000, 10_000]:
            r = hedge_minimax_regret(T, N)
            assert r > prev
            prev = r

    def test_hedge_minimax_regret_monotone_N(self):
        T = 100
        prev = 0.0
        for N in [2, 5, 10, 100]:
            r = hedge_minimax_regret(T, N)
            assert r > prev
            prev = r

    def test_hedge_minimax_eta_closed_form(self):
        T, N = 100, 5
        eta = hedge_minimax_eta(T, N)
        # Verify: this η minimises η T / 8 + log N / η.
        # f'(η) = T/8 - log N / η² = 0 → η = √(8 log N / T).
        expected = math.sqrt(8.0 * math.log(N) / T)
        assert math.isclose(eta, expected, rel_tol=1.0e-9)

    def test_hedge_minimax_eta_minimises_bound(self):
        T, N = 1000, 10
        eta_star = hedge_minimax_eta(T, N)
        r_star = hedge_regret_bound(T, N, eta_star)
        for off in [0.1, 0.5, 2.0, 10.0]:
            r_off = hedge_regret_bound(T, N, eta_star * off)
            assert r_star <= r_off + 1.0e-9

    def test_adahedge_regret_nonneg(self):
        for V_T, N in [(0.0, 5), (10.0, 10), (1000.0, 100)]:
            assert adahedge_regret_bound(V_T, N) >= 0.0

    def test_normal_hedge_regret_monotone(self):
        T = 1000
        N = 50
        prev = 0.0
        for rank in [1, 2, 5, 10]:
            r = normal_hedge_regret_bound(T, N, rank=rank)
            assert r > prev
            prev = r

    def test_squint_bound_monotone_in_K(self):
        # As K → N, the bound goes to 0 (we're competing with the whole pool).
        V_T = 10.0
        N = 100
        b1 = squint_regret_bound(V_T, N, K=1)
        bn = squint_regret_bound(V_T, N, K=N)
        assert b1 > bn
        assert bn == 0.0

    def test_ml_prod_bound_matches_paper_constants(self):
        V_T = 100.0
        N = 10
        b = ml_prod_regret_bound(V_T, N)
        # log N = log 10 ≈ 2.303; √(8 · 100 · 2.303) ≈ 42.93; + 5·2.303 ≈ 11.51
        expected = math.sqrt(8.0 * V_T * math.log(N)) + 5.0 * math.log(N)
        assert math.isclose(b, expected, rel_tol=1.0e-9)

    def test_boa_bound_smaller_than_mlprod_on_high_variance(self):
        V_T = 1000.0
        N = 10
        # BOA has √(2 V_T log N), ML-Prod has √(8 V_T log N) — BOA always tighter
        assert boa_regret_bound(V_T, N) < ml_prod_regret_bound(V_T, N)

    def test_pac_bayes_zero_at_uniform(self):
        prior = {"a": 0.5, "b": 0.5}
        # Posterior = prior → KL = 0 → bound = 0.
        assert pac_bayes_regret_bound(100, prior, prior) == 0.0

    def test_pac_bayes_increasing_with_kl(self):
        prior = {"a": 0.5, "b": 0.5}
        post_close = {"a": 0.55, "b": 0.45}
        post_far = {"a": 0.95, "b": 0.05}
        assert (pac_bayes_regret_bound(100, prior, post_far)
                > pac_bayes_regret_bound(100, prior, post_close))


# =====================================================================
# Finite-sample bound helpers
# =====================================================================


class TestFiniteSampleBounds:
    def test_hoeffding_lcb_ucb_order(self):
        l = hoeffding_lcb(0.5, 100, delta=0.05)
        u = hoeffding_ucb(0.5, 100, delta=0.05)
        assert l < 0.5 < u

    def test_hoeffding_tighter_with_more_data(self):
        l1 = hoeffding_lcb(0.5, 10, delta=0.05)
        l2 = hoeffding_lcb(0.5, 1000, delta=0.05)
        assert l1 < l2 < 0.5

    def test_bernstein_tighter_than_hoeffding_lowvar(self):
        # When variance is small, Bernstein should be tighter.
        n = 1000
        mean = 0.5
        var = 0.001
        bl = empirical_bernstein_lcb(mean, var, n, delta=0.05)
        hl = hoeffding_lcb(mean, n, delta=0.05)
        # Bernstein gets the variance-driven sharpening; Hoeffding uses range.
        assert bl > hl

    def test_anytime_wider_than_hoeffding(self):
        # Anytime bound pays a √log log n factor for time-uniformity.
        n = 100
        mean = 0.5
        al = anytime_lcb(mean, n, delta=0.05)
        hl = hoeffding_lcb(mean, n, delta=0.05)
        assert al < hl

    def test_insufficient_data(self):
        with pytest.raises(InsufficientData):
            hoeffding_lcb(0.5, 0, delta=0.05)
        with pytest.raises(InsufficientData):
            empirical_bernstein_lcb(0.5, 0.1, 1, delta=0.05)


# =====================================================================
# KL divergence
# =====================================================================


class TestKL:
    def test_kl_self_is_zero(self):
        p = {"a": 0.3, "b": 0.7}
        assert math.isclose(kl_divergence(p, p), 0.0, abs_tol=1.0e-12)

    def test_kl_nonneg(self):
        p = {"a": 0.3, "b": 0.7}
        q = {"a": 0.5, "b": 0.5}
        assert kl_divergence(p, q) > 0.0

    def test_kl_mismatched_support(self):
        with pytest.raises(HedgerError):
            kl_divergence({"a": 1.0}, {"b": 1.0})


# =====================================================================
# Hedge: basic dynamics
# =====================================================================


class TestHedgeBasic:
    def test_uniform_initial_weights(self):
        h = Hedger.create(["a", "b", "c", "d"])
        pred = h.predict()
        for w in pred.weights.values():
            assert math.isclose(w, 0.25)

    def test_weights_sum_to_one(self):
        for algo in [HEDGE, ADAHEDGE, NORMAL_HEDGE, SQUINT, ML_PROD,
                     FTRL_ENTROPY, FTRL_L2, OMD_ENTROPY, BOA]:
            h = Hedger.create(["a", "b", "c"], algorithm=algo, eta=0.5)
            for _ in range(5):
                rng = random.Random(42)
                losses = {e: rng.random() for e in h.experts}
                h.observe(losses)
                pred = h.predict()
                total = sum(pred.weights.values())
                assert math.isclose(total, 1.0, abs_tol=1.0e-6), \
                    f"algorithm {algo}: weights sum to {total}"

    def test_hedge_favours_low_loss_expert(self):
        h = Hedger.create(["good", "bad"], algorithm=HEDGE, eta=1.0)
        # Round after round: "good" always 0, "bad" always 1.
        for _ in range(20):
            h.observe({"good": 0.0, "bad": 1.0})
        pred = h.predict()
        assert pred.weights["good"] > pred.weights["bad"]
        assert pred.weights["good"] > 0.99

    def test_hedge_with_horizon_uses_minimax_eta(self):
        h = Hedger.create(["a", "b", "c"], algorithm=HEDGE, horizon=100)
        eta = h._current_eta()
        expected = hedge_minimax_eta(100, 3)
        assert math.isclose(eta, expected, rel_tol=1.0e-9)

    def test_hedge_regret_bound_at_minimax(self):
        T = 200
        rng = random.Random(0)
        h = Hedger.create(["a", "b", "c"], algorithm=HEDGE,
                           horizon=T, seed=0)
        # IID Bernoulli losses, all means = 0.5.
        for _ in range(T):
            losses = {e: 1.0 if rng.random() < 0.5 else 0.0
                       for e in h.experts}
            h.observe(losses)
        rep = h.report()
        # Realised regret must be within the minimax bound.
        assert rep.realised_regret_so_far <= rep.regret_certificate.first_order_bound \
               + 1.0e-9


# =====================================================================
# AdaHedge: adaptive learning rate
# =====================================================================


class TestAdaHedge:
    def test_adahedge_mixability_gap_nonneg(self):
        h = Hedger.create(["a", "b", "c"], algorithm=ADAHEDGE)
        for t in range(50):
            rng = random.Random(t)
            losses = {e: rng.random() for e in h.experts}
            rd = h.observe(losses)
            assert rd.delta_mixability_gap >= -1.0e-12

    def test_adahedge_bound_finite(self):
        T = 200
        rng = random.Random(1)
        h = Hedger.create(["a", "b", "c"], algorithm=ADAHEDGE)
        for _ in range(T):
            losses = {e: rng.random() for e in h.experts}
            h.observe(losses)
        rep = h.report()
        assert math.isfinite(rep.regret_certificate.first_order_bound)
        assert rep.realised_regret_so_far <= rep.regret_certificate.first_order_bound \
               + 1.0e-6

    def test_adahedge_beats_uniform_on_clear_winner(self):
        T = 200
        h = Hedger.create(["good", "bad1", "bad2"], algorithm=ADAHEDGE)
        for _ in range(T):
            h.observe({"good": 0.0, "bad1": 1.0, "bad2": 1.0})
        rep = h.report()
        # Cumulative loss of AdaHedge should be ≪ T/3 (uniform reference).
        assert rep.cumulative_weighted_loss < T / 3.0


# =====================================================================
# NormalHedge: parameter-free anytime
# =====================================================================


class TestNormalHedge:
    def test_normal_hedge_weights_nonneg_sum_to_one(self):
        h = Hedger.create(["a", "b", "c", "d"], algorithm=NORMAL_HEDGE)
        for t in range(50):
            rng = random.Random(t)
            h.observe({e: rng.random() for e in h.experts})
            pred = h.predict()
            for w in pred.weights.values():
                assert w >= -1.0e-9
            assert math.isclose(sum(pred.weights.values()), 1.0, abs_tol=1.0e-6)

    def test_normal_hedge_no_eta_needed(self):
        h = Hedger.create(["a", "b"], algorithm=NORMAL_HEDGE)
        eta = h._current_eta()
        assert math.isnan(eta)

    def test_normal_hedge_concentrates_on_winner(self):
        h = Hedger.create(["good", "bad"], algorithm=NORMAL_HEDGE)
        for _ in range(100):
            h.observe({"good": 0.0, "bad": 1.0})
        pred = h.predict()
        assert pred.weights["good"] > pred.weights["bad"]


# =====================================================================
# Squint
# =====================================================================


class TestSquint:
    def test_squint_concentrates_on_winner(self):
        h = Hedger.create(["good", "bad"], algorithm=SQUINT)
        for _ in range(50):
            h.observe({"good": 0.0, "bad": 1.0})
        pred = h.predict()
        assert pred.weights["good"] > pred.weights["bad"]


# =====================================================================
# ML-Prod
# =====================================================================


class TestMLProd:
    def test_ml_prod_weights_consistent(self):
        h = Hedger.create(["a", "b", "c"], algorithm=ML_PROD)
        for t in range(30):
            rng = random.Random(t)
            h.observe({e: rng.random() for e in h.experts})
        pred = h.predict()
        assert math.isclose(sum(pred.weights.values()), 1.0, abs_tol=1.0e-6)

    def test_ml_prod_finite_regret(self):
        h = Hedger.create(["a", "b", "c"], algorithm=ML_PROD)
        for t in range(100):
            rng = random.Random(t)
            h.observe({e: rng.random() for e in h.experts})
        rep = h.report()
        assert math.isfinite(rep.regret_certificate.first_order_bound)


# =====================================================================
# FTRL variants
# =====================================================================


class TestFTRL:
    def test_ftrl_entropy_matches_hedge(self):
        h_a = Hedger.create(["a", "b", "c"], algorithm=HEDGE, eta=0.5)
        h_b = Hedger.create(["a", "b", "c"], algorithm=FTRL_ENTROPY, eta=0.5)
        rng = random.Random(7)
        for _ in range(20):
            losses = {e: rng.random() for e in h_a.experts}
            h_a.observe(losses)
            h_b.observe(losses)
        pa = h_a.predict().weights
        pb = h_b.predict().weights
        for e in pa:
            assert math.isclose(pa[e], pb[e], abs_tol=1.0e-9)

    def test_omd_entropy_matches_hedge(self):
        h_a = Hedger.create(["a", "b"], algorithm=HEDGE, eta=1.0)
        h_b = Hedger.create(["a", "b"], algorithm=OMD_ENTROPY, eta=1.0)
        rng = random.Random(11)
        for _ in range(15):
            losses = {e: rng.random() for e in h_a.experts}
            h_a.observe(losses)
            h_b.observe(losses)
        pa = h_a.predict().weights
        pb = h_b.predict().weights
        for e in pa:
            assert math.isclose(pa[e], pb[e], abs_tol=1.0e-9)

    def test_ftrl_l2_projection_to_simplex(self):
        h = Hedger.create(["a", "b", "c"], algorithm=FTRL_L2, eta=0.1)
        for t in range(20):
            rng = random.Random(t)
            h.observe({e: rng.random() for e in h.experts})
            pred = h.predict()
            for w in pred.weights.values():
                assert -1.0e-9 <= w <= 1.0 + 1.0e-9
            assert math.isclose(sum(pred.weights.values()), 1.0, abs_tol=1.0e-6)


# =====================================================================
# FTPL
# =====================================================================


class TestFTPL:
    def test_ftpl_deterministic_given_seed(self):
        h_a = Hedger.create(["a", "b", "c"], algorithm=FTPL, eta=1.0, seed=42)
        h_b = Hedger.create(["a", "b", "c"], algorithm=FTPL, eta=1.0, seed=42)
        for t in range(10):
            losses = {"a": 0.1 * t, "b": 0.5, "c": 1.0 - 0.1 * t}
            h_a.observe(losses)
            h_b.observe(losses)
        assert h_a.predict().weights == h_b.predict().weights

    def test_ftpl_concentrates_on_winner(self):
        h = Hedger.create(["good", "bad"], algorithm=FTPL,
                           eta=1.0, seed=0, ftpl_draws=256)
        for _ in range(50):
            h.observe({"good": 0.0, "bad": 1.0})
        pred = h.predict()
        assert pred.weights["good"] > pred.weights["bad"]


# =====================================================================
# BOA
# =====================================================================


class TestBOA:
    def test_boa_weights_sum_to_one(self):
        h = Hedger.create(["a", "b", "c"], algorithm=BOA)
        for t in range(30):
            rng = random.Random(t)
            h.observe({e: rng.random() for e in h.experts})
            pred = h.predict()
            assert math.isclose(sum(pred.weights.values()), 1.0, abs_tol=1.0e-6)

    def test_boa_per_expert_adaptivity(self):
        # An expert with low variance should get amplified weight quicker.
        h = Hedger.create(["stable_low", "noisy"], algorithm=BOA)
        rng = random.Random(0)
        for _ in range(50):
            h.observe({
                "stable_low": 0.1,
                "noisy": 0.5 + 0.4 * (rng.random() - 0.5),
            })
        pred = h.predict()
        assert pred.weights["stable_low"] > pred.weights["noisy"]


# =====================================================================
# Selection (sampling)
# =====================================================================


class TestSelection:
    def test_selection_returns_known_expert(self):
        h = Hedger.create(["a", "b", "c"])
        sel = h.select()
        assert sel.expert in {"a", "b", "c"}

    def test_selection_replay_deterministic(self):
        h_a = Hedger.create(["a", "b"], algorithm=HEDGE, eta=1.0, seed=99)
        h_b = Hedger.create(["a", "b"], algorithm=HEDGE, eta=1.0, seed=99)
        seqs_a = [h_a.select().expert for _ in range(20)]
        seqs_b = [h_b.select().expert for _ in range(20)]
        assert seqs_a == seqs_b

    def test_selection_fingerprint_chain(self):
        h = Hedger.create(["a", "b"])
        s1 = h.select()
        s2 = h.select()
        # Each selection chains parent fingerprint.
        assert s1.fingerprint != s2.fingerprint

    def test_selection_after_observe_updates_distribution(self):
        h = Hedger.create(["good", "bad"], algorithm=HEDGE, eta=10.0, seed=0)
        for _ in range(20):
            h.observe({"good": 0.0, "bad": 1.0})
        # After enough rounds 'good' should be picked overwhelmingly.
        picks = [h.select().expert for _ in range(100)]
        good_frac = sum(1 for p in picks if p == "good") / len(picks)
        assert good_frac > 0.95


# =====================================================================
# Reports + receipts
# =====================================================================


class TestReports:
    def test_report_after_observe(self):
        h = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5)
        h.observe({"a": 0.0, "b": 1.0})
        rep = h.report()
        assert isinstance(rep, HedgerReport)
        assert rep.T == 1
        assert rep.cumulative_loss_by_expert["a"] == 0.0
        assert rep.cumulative_loss_by_expert["b"] == 1.0

    def test_report_includes_regret_certificate(self):
        h = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5, horizon=100)
        h.observe({"a": 0.2, "b": 0.8})
        rep = h.report()
        rc = rep.regret_certificate
        assert isinstance(rc, RegretCertificate)
        assert rc.T == 1
        assert rc.first_order_bound > 0.0
        assert rc.best_expert == "a"

    def test_per_expert_certificate_progression(self):
        # Use enough samples so the Maurer-Pontil range term shrinks
        # relative to the variance-dominated term.
        h = Hedger.create(["a", "b"], algorithm=HEDGE)
        for _ in range(500):
            h.observe({"a": 0.3, "b": 0.7})
        cert = h.per_expert_certificate("a", delta=0.05)
        assert isinstance(cert, ExpertCertificate)
        assert cert.n == 500
        assert math.isclose(cert.mean_loss, 0.3, abs_tol=1.0e-9)
        assert cert.hoeffding_lcb < cert.mean_loss < cert.hoeffding_ucb
        assert cert.anytime_lcb < cert.mean_loss < cert.anytime_ucb
        assert cert.bernstein_lcb is not None
        # Bernstein on near-constant loss should be tight at n=500.
        assert cert.bernstein_ucb - cert.bernstein_lcb < 0.1
        # And Bernstein should be tighter than Hoeffding when variance ≈ 0.
        assert (cert.bernstein_ucb - cert.bernstein_lcb
                < cert.hoeffding_ucb - cert.hoeffding_lcb)

    def test_per_expert_certificate_unknown_expert(self):
        h = Hedger.create(["a", "b"])
        h.observe({"a": 0.5, "b": 0.5})
        with pytest.raises(InvalidLoss):
            h.per_expert_certificate("nonexistent")

    def test_per_expert_certificate_zero_rounds(self):
        h = Hedger.create(["a", "b"])
        with pytest.raises(InsufficientData):
            h.per_expert_certificate("a")


# =====================================================================
# Sleeping experts (specialists)
# =====================================================================


class TestSleepingExperts:
    def test_sleeping_expert_loss_unchanged(self):
        h = Hedger.create(["a", "b", "c"], algorithm=HEDGE, eta=0.5)
        h.observe_partial({"a": 0.2, "b": 0.5}, sleeping={"c"})
        rep = h.report()
        # 'c' wasn't observed → cumulative loss = 0.
        assert rep.cumulative_loss_by_expert["c"] == 0.0
        assert rep.cumulative_loss_by_expert["a"] == 0.2
        assert rep.cumulative_loss_by_expert["b"] == 0.5

    def test_sleeping_unknown_expert_rejected(self):
        h = Hedger.create(["a", "b"])
        with pytest.raises(InvalidLoss):
            h.observe_partial({"a": 0.5, "b": 0.5}, sleeping={"unknown"})

    def test_all_sleeping_rejected(self):
        h = Hedger.create(["a", "b"])
        with pytest.raises(InvalidLoss):
            h.observe_partial({}, sleeping={"a", "b"})


# =====================================================================
# State management: clear / snapshot / restore
# =====================================================================


class TestStateManagement:
    def test_clear_resets_state(self):
        h = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5)
        for _ in range(5):
            h.observe({"a": 0.2, "b": 0.8})
        assert h.T == 5
        h.clear()
        assert h.T == 0
        pred = h.predict()
        for w in pred.weights.values():
            assert math.isclose(w, 0.5)

    def test_snapshot_restore_roundtrip(self):
        h1 = Hedger.create(["a", "b", "c"], algorithm=HEDGE, eta=0.5, seed=7)
        for t in range(10):
            rng = random.Random(t)
            h1.observe({e: rng.random() for e in h1.experts})
        snap = h1.snapshot()

        h2 = Hedger.create(["a", "b", "c"], algorithm=HEDGE, eta=0.5, seed=7)
        h2.restore(snap)

        # Both should now produce identical weights and certs.
        p1 = h1.predict().weights
        p2 = h2.predict().weights
        for e in p1:
            assert math.isclose(p1[e], p2[e], abs_tol=1.0e-9)
        r1 = h1.regret_certificate()
        r2 = h2.regret_certificate()
        assert math.isclose(r1.first_order_bound, r2.first_order_bound)


# =====================================================================
# Replay determinism / fingerprint chain
# =====================================================================


class TestReplay:
    def test_observe_fingerprint_changes_each_round(self):
        h = Hedger.create(["a", "b"])
        fps = set()
        for _ in range(10):
            rd = h.observe({"a": 0.3, "b": 0.7})
            fps.add(rd.fingerprint)
        assert len(fps) == 10

    def test_identical_history_same_fingerprint(self):
        h1 = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5, seed=3)
        h2 = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5, seed=3)
        losses_seq = [{"a": 0.1, "b": 0.9},
                       {"a": 0.2, "b": 0.8},
                       {"a": 0.3, "b": 0.7}]
        for L in losses_seq:
            h1.observe(L)
            h2.observe(L)
        # Same config + same observations → byte-identical fingerprint chain.
        assert h1.report().fingerprint == h2.report().fingerprint
        assert h1._rounds[-1].fingerprint == h2._rounds[-1].fingerprint

    def test_divergent_history_different_fingerprint(self):
        h1 = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5, seed=3)
        h2 = Hedger.create(["a", "b"], algorithm=HEDGE, eta=0.5, seed=3)
        h1.observe({"a": 0.1, "b": 0.9})
        h2.observe({"a": 0.9, "b": 0.1})
        assert h1.report().fingerprint != h2.report().fingerprint


# =====================================================================
# Theoretical guarantee: regret upper bound is honoured
# =====================================================================


class TestTheoreticalGuarantees:
    @pytest.mark.parametrize("algorithm",
                              [HEDGE, ADAHEDGE, NORMAL_HEDGE, ML_PROD,
                               FTRL_L2, BOA])
    def test_realised_regret_within_bound_iid(self, algorithm):
        T = 200
        N = 4
        experts = [f"e{i}" for i in range(N)]
        means = [0.2, 0.4, 0.5, 0.6]
        h = Hedger.create(experts, algorithm=algorithm,
                           eta=hedge_minimax_eta(T, N) if algorithm == HEDGE else None,
                           horizon=T,
                           seed=0)
        rng = random.Random(0)
        for _ in range(T):
            losses = {experts[i]: 1.0 if rng.random() < means[i] else 0.0
                       for i in range(N)}
            h.observe(losses)
        rep = h.report()
        rc = rep.regret_certificate
        # The realised regret must respect the closed-form bound (a fortiori
        # plus a safety margin).
        assert rc.realised_regret_so_far <= rc.first_order_bound + 1.0e-6

    def test_hedge_log_n_T_growth(self):
        # Empirical regret should grow no faster than √(T log N).
        N = 5
        experts = [f"e{i}" for i in range(N)]
        T_small = 100
        T_large = 1000

        def run(T):
            h = Hedger.create(experts, algorithm=HEDGE,
                               horizon=T, seed=0)
            rng = random.Random(0)
            means = [0.3, 0.35, 0.4, 0.45, 0.5]
            for _ in range(T):
                h.observe({experts[i]: 1.0 if rng.random() < means[i] else 0.0
                            for i in range(N)})
            return h.report().realised_regret_so_far

        # Ratio should be bounded — generous factor.
        r_s = run(T_small)
        r_l = run(T_large)
        # √(T log N) growth → r_l / r_s ≤ √10 · (small constant).
        if r_s > 0.5:
            assert r_l / r_s < 10.0


# =====================================================================
# Composition: PAC-Bayes against a non-uniform prior
# =====================================================================


class TestPACBayesComposition:
    def test_pac_bayes_with_informative_prior(self):
        prior = {"a": 0.7, "b": 0.15, "c": 0.15}
        h = Hedger.create(["a", "b", "c"], algorithm=HEDGE,
                           eta=0.5, prior=prior, seed=0)
        for _ in range(20):
            h.observe({"a": 0.1, "b": 0.9, "c": 0.9})
        pred = h.predict()
        # Prior already favoured 'a'; combined with consistent low loss, 'a'
        # should dominate.
        assert pred.weights["a"] > 0.5

    def test_pac_bayes_bound_finite(self):
        prior = {"a": 0.4, "b": 0.6}
        h = Hedger.create(["a", "b"], algorithm=HEDGE,
                           eta=0.5, prior=prior, seed=0)
        for _ in range(15):
            h.observe({"a": 0.3, "b": 0.7})
        rep = h.report()
        assert math.isfinite(rep.regret_certificate.pac_bayes_bound_uniform)


# =====================================================================
# Coordination engine smoke test
# =====================================================================


class TestCoordinationSmoke:
    def test_meta_decision_over_primitives(self):
        """Smoke test: a Hedger composes recommendations from three
        primitives (named "bandit", "bayesopt", "thompson"); after T
        rounds the cumulative loss should track the best primitive."""
        random.seed(0)
        h = Hedger.create(["bandit", "bayesopt", "thompson"],
                           algorithm=ADAHEDGE, seed=0)
        # Simulated per-primitive losses: bayesopt is the best.
        means = {"bandit": 0.4, "bayesopt": 0.2, "thompson": 0.5}
        rng = random.Random(123)
        T = 500
        for _ in range(T):
            losses = {k: 1.0 if rng.random() < means[k] else 0.0
                       for k in means}
            h.observe(losses)
        rep = h.report()
        assert rep.regret_certificate.best_expert == "bayesopt"
        # The realised cumulative weighted loss should be much closer to
        # bayesopt's cumulative loss than to thompson's.
        cum_best = rep.cumulative_loss_by_expert["bayesopt"]
        cum_alg = rep.cumulative_weighted_loss
        excess_regret = cum_alg - cum_best
        # With T=500, ADaHedge should keep regret bounded.
        assert excess_regret < rep.regret_certificate.first_order_bound + 1.0e-6
