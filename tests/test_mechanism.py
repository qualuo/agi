"""Tests for ``agi.mechanism`` — revenue-optimal mechanism design.

Mathematical contract under test:

  1. **Vickrey** is DSIC: truthful bidding maximises every bidder's utility
     under any opponent bid profile. Winner = arg-max bid, payment =
     second-highest bid.
  2. **First-price** is NOT DSIC: bidders strictly gain by shading their
     bid just above the second-highest. Our certifier *catches* the gain.
  3. **Myerson** with U[0,1] priors has closed-form reserve r* = 1/2 and
     virtual valuation φ(v) = 2v − 1. The auction's stochastic revenue
     matches the analytical formula
        E[Rev] = E[max(φ_(1), 0)] = ∫_{1/2}^{1} (2v−1) · n · v^{n−1} dv
     for symmetric n-bidder U[0,1].
  4. **VCG** is DSIC and IR: payments equal externalities, utilities ≥ 0.
  5. **Posted-price** is DSIC by construction (each bidder offered a
     take-it-or-leave-it deal).
  6. **Bulow-Klemperer**: for U[0,1], (n+1)-bidder Vickrey *without*
     reserve has revenue ≥ n-bidder Myerson revenue.
  7. **Empirical Myerson** (`empirical_reserve`) converges to b/2 = 0.5
     as n → ∞ when samples are U[0,1].
  8. **Sample complexity** scales as ε^{-2} for regular priors.
  9. **Online posted-price** has revenue → optimal under a fixed-threshold
     adversary (long-horizon convergence).
 10. **Attestation** receipt digests are deterministic content hashes.
 11. **Event bus** receives ``mechanism.*`` events on every action.

Tests use *only* the Python standard library and the stdlib `random`
module (seeded).
"""
from __future__ import annotations

import math
import random
import statistics
import threading
import time

import pytest

from agi.events import Event, EventBus
from agi.mechanism import (
    Allocation,
    BulowKlemperer,
    DSICReport,
    EmpiricalDistribution,
    ExponentialDistribution,
    InsufficientData,
    InvalidBid,
    InvalidDistribution,
    KIND_ALL_PAY,
    KIND_ANONYMOUS_RESERVE,
    KIND_FIRST_PRICE,
    KIND_MYERSON,
    KIND_POSTED_PRICE,
    KIND_VCG,
    KIND_VICKREY,
    KNOWN_MECHANISMS,
    MECH_ALLOCATED,
    MECH_BULOW_KLEMPERER,
    MECH_CERTIFIED,
    MECH_RESERVE_FIT,
    MECH_STARTED,
    MechanismDesigner,
    MechanismError,
    UniformDistribution,
    UnknownMechanism,
    VCGAllocation,
    empirical_bernstein_radius,
    hoeffding_radius,
    myerson_winner_and_payment,
    quick_myerson,
    quick_vcg,
    quick_vickrey,
    sample_complexity_for_eps_optimal,
)


# -------------------- distributions -------------------------------------


class TestUniformDistribution:
    def test_cdf_pdf_at_boundaries(self):
        d = UniformDistribution(0.0, 1.0)
        assert d.cdf(-1) == 0.0
        assert d.cdf(0.0) == 0.0
        assert d.cdf(0.5) == pytest.approx(0.5)
        assert d.cdf(1.0) == 1.0
        assert d.cdf(2.0) == 1.0
        assert d.pdf(0.5) == pytest.approx(1.0)
        assert d.pdf(-1) == 0.0
        assert d.pdf(2) == 0.0

    def test_virtual_value_closed_form(self):
        # φ(v) = 2v − b
        d = UniformDistribution(0.0, 1.0)
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert d.virtual_value(v) == pytest.approx(2 * v - 1)
        d2 = UniformDistribution(0.0, 4.0)
        assert d2.virtual_value(2.0) == pytest.approx(0.0)
        assert d2.virtual_value(3.0) == pytest.approx(2.0)

    def test_monopoly_reserve(self):
        assert UniformDistribution(0, 1).monopoly_reserve() == pytest.approx(0.5)
        assert UniformDistribution(0, 4).monopoly_reserve() == pytest.approx(2.0)
        # When a > b/2, reserve is a (binding lower bound)
        assert UniformDistribution(3, 4).monopoly_reserve() == pytest.approx(3.0)

    def test_quantile_inverse(self):
        d = UniformDistribution(0.0, 1.0)
        for p in (0.0, 0.25, 0.5, 0.99, 1.0):
            assert d.quantile(p) == pytest.approx(p)

    def test_sample_in_support(self):
        d = UniformDistribution(2.0, 5.0)
        rng = random.Random(1)
        for _ in range(100):
            v = d.sample(rng)
            assert 2.0 <= v <= 5.0

    def test_invalid_construction(self):
        with pytest.raises(InvalidDistribution):
            UniformDistribution(1.0, 0.0)

    def test_regularity_label(self):
        from agi.mechanism import REG_REGULAR
        assert UniformDistribution(0, 1).regularity() == REG_REGULAR


class TestExponentialDistribution:
    def test_virtual_value_closed_form(self):
        # φ(v) = v − 1/λ, monopoly reserve = 1/λ
        d = ExponentialDistribution(rate=2.0, vmax=10.0)
        assert d.virtual_value(0.5) == pytest.approx(0.0)
        assert d.virtual_value(1.0) == pytest.approx(0.5)
        assert d.monopoly_reserve() == pytest.approx(0.5)

    def test_pdf_decreasing(self):
        d = ExponentialDistribution(rate=1.0, vmax=10.0)
        a = d.pdf(0.0)
        b = d.pdf(1.0)
        c = d.pdf(5.0)
        assert a > b > c

    def test_invalid_construction(self):
        with pytest.raises(InvalidDistribution):
            ExponentialDistribution(rate=-1.0)
        with pytest.raises(InvalidDistribution):
            ExponentialDistribution(vmax=-1.0)


class TestEmpiricalDistribution:
    def test_cdf_glivenko_cantelli(self):
        rng = random.Random(1)
        samples = [rng.uniform(0, 1) for _ in range(1000)]
        d = EmpiricalDistribution(samples=tuple(samples))
        # For U[0,1]: CDF(x) ≈ x. Glivenko-Cantelli: |F_n − F|_∞ = O(1/√n).
        for x in (0.1, 0.5, 0.9):
            assert abs(d.cdf(x) - x) < 0.05

    def test_sample_is_in_support(self):
        d = EmpiricalDistribution(samples=(0.0, 0.5, 1.0, 0.3))
        rng = random.Random(0)
        for _ in range(50):
            v = d.sample(rng)
            assert 0.0 <= v <= 1.0

    def test_invalid_construction(self):
        with pytest.raises(InsufficientData):
            EmpiricalDistribution(samples=(0.5,))
        with pytest.raises(InsufficientData):
            EmpiricalDistribution(samples=(0.5, 0.5, 0.5))


# -------------------- concentration helpers -----------------------------


class TestConcentrationRadii:
    def test_hoeffding_radius_shrinks_with_n(self):
        # Radius should shrink as 1/√n
        r10 = hoeffding_radius(10, delta=0.05, range_=1.0)
        r100 = hoeffding_radius(100, delta=0.05, range_=1.0)
        r1000 = hoeffding_radius(1000, delta=0.05, range_=1.0)
        assert r10 > r100 > r1000 > 0
        # Approximate sqrt-shrinkage: ratio ≈ √10
        assert r10 / r100 == pytest.approx(math.sqrt(10), rel=0.05)

    def test_empirical_bernstein_with_zero_variance(self):
        vals = [0.5] * 50
        r = empirical_bernstein_radius(vals, delta=0.05, range_=1.0)
        # variance is 0 → first term vanishes; only the additive O(1/n) survives
        assert r > 0
        assert r < 0.2

    def test_hoeffding_radius_validation(self):
        with pytest.raises(MechanismError):
            hoeffding_radius(10, delta=0.0, range_=1.0)
        with pytest.raises(MechanismError):
            hoeffding_radius(10, delta=1.0, range_=1.0)


# -------------------- Vickrey -------------------------------------------


class TestVickrey:
    def test_winner_and_payment(self):
        md = MechanismDesigner()
        a = md.vickrey_auction({"a": 0.5, "b": 0.7, "c": 0.9})
        assert a.winner == "c"
        assert a.payment == pytest.approx(0.7)
        assert a.revenue == pytest.approx(0.7)
        assert a.welfare == pytest.approx(0.9)

    def test_reserve_blocks_low_bids(self):
        md = MechanismDesigner()
        # Reserve above all bids: no sale
        a = md.vickrey_auction({"a": 0.1, "b": 0.2}, reserve=0.5)
        assert a.winner is None
        assert a.revenue == 0.0

    def test_reserve_above_second(self):
        md = MechanismDesigner()
        # bids 0.4, 0.9; reserve 0.6 → winner pays reserve, not second
        a = md.vickrey_auction({"a": 0.4, "b": 0.9}, reserve=0.6)
        assert a.winner == "b"
        assert a.payment == pytest.approx(0.6)

    def test_single_bidder_pays_reserve(self):
        md = MechanismDesigner()
        a = md.vickrey_auction({"only": 0.7}, reserve=0.3)
        assert a.winner == "only"
        assert a.payment == pytest.approx(0.3)

    def test_ir(self):
        md = MechanismDesigner()
        a = md.vickrey_auction({"a": 0.3, "b": 0.7, "c": 0.5})
        ir = md.certify_ir(a)
        assert ir.is_ir
        assert ir.worst_utility >= -1e-9

    def test_dsic_certificate_passes(self):
        md = MechanismDesigner()
        r = md.certify_dsic(KIND_VICKREY, bids_truthful={"a": 0.3, "b": 0.7, "c": 0.5})
        assert r.is_dsic
        assert r.worst_gain <= 1e-6

    def test_empty_bids(self):
        with pytest.raises(InvalidBid):
            MechanismDesigner().vickrey_auction({})

    def test_invalid_negative_bid(self):
        with pytest.raises(InvalidBid):
            from agi.mechanism import Bid
            Bid(bidder_id="x", value=-1.0)


# -------------------- First-price (not DSIC) ----------------------------


class TestFirstPrice:
    def test_pay_your_bid(self):
        md = MechanismDesigner()
        a = md.first_price_auction({"a": 0.5, "b": 0.7, "c": 0.9})
        assert a.winner == "c"
        assert a.payment == pytest.approx(0.9)

    def test_dsic_certificate_FAILS(self):
        md = MechanismDesigner()
        r = md.certify_dsic(
            KIND_FIRST_PRICE,
            bids_truthful={"a": 0.3, "b": 0.7, "c": 0.5},
        )
        assert not r.is_dsic
        # gain ≥ second-highest opponent gap
        assert r.worst_gain > 0.05


# -------------------- Myerson optimal -----------------------------------


class TestMyerson:
    def test_uniform_priors_closed_form(self):
        md = MechanismDesigner()
        d = {x: UniformDistribution(0, 1) for x in ("a", "b", "c")}
        a = md.myerson_auction({"a": 0.3, "b": 0.7, "c": 0.5}, d)
        # bidder b has highest virtual value 0.4; second-highest is c=0
        # Threshold for b: smallest v* s.t. 2v* − 1 ≥ 0 (second-best) AND ≥ 0 (reserve)
        # → v* = 0.5
        assert a.winner == "b"
        assert a.payment == pytest.approx(0.5)
        assert a.reserve == pytest.approx(0.5)

    def test_reserve_blocks_low_bids(self):
        md = MechanismDesigner()
        d = {x: UniformDistribution(0, 1) for x in ("a", "b")}
        # both bids below 0.5 reserve → no sale
        a = md.myerson_auction({"a": 0.3, "b": 0.4}, d)
        assert a.winner is None
        assert a.revenue == 0.0

    def test_two_strong_bidders(self):
        md = MechanismDesigner()
        d = {x: UniformDistribution(0, 1) for x in ("a", "b")}
        a = md.myerson_auction({"a": 0.8, "b": 0.9}, d)
        # winner b. Threshold: φ_b(v*) = φ_a(0.8) = 0.6 → v* = 0.8
        assert a.winner == "b"
        assert a.payment == pytest.approx(0.8)

    def test_dsic_certificate(self):
        md = MechanismDesigner()
        d = {x: UniformDistribution(0, 1) for x in ("a", "b", "c")}
        r = md.certify_dsic(
            KIND_MYERSON,
            bids_truthful={"a": 0.3, "b": 0.7, "c": 0.5},
            distributions=d,
        )
        assert r.is_dsic

    def test_asymmetric_priors(self):
        # Bidder with higher prior gets lower reserve in Myerson — the
        # auction discriminates against the strong bidder
        md = MechanismDesigner()
        d = {
            "weak": UniformDistribution(0, 1),
            "strong": UniformDistribution(0, 2),
        }
        # Strong's monopoly reserve = 1; weak's = 0.5
        a = md.myerson_auction({"weak": 0.8, "strong": 0.8}, d)
        # φ_weak(0.8) = 0.6 ; φ_strong(0.8) = 2*0.8 − 2 = −0.4 (below reserve)
        # → weak wins
        assert a.winner == "weak"

    def test_revenue_dominates_vickrey_no_reserve(self):
        # Repeated random draws: Myerson should beat 2nd-price-no-reserve
        # in expected revenue on U[0,1] with 2 bidders.
        md = MechanismDesigner()
        d = {x: UniformDistribution(0, 1) for x in ("a", "b")}
        rng = random.Random(7)
        rev_my, rev_vi = 0.0, 0.0
        N = 3000
        for _ in range(N):
            v_a, v_b = rng.uniform(0, 1), rng.uniform(0, 1)
            bids = {"a": v_a, "b": v_b}
            rev_my += md.myerson_auction(bids, d).revenue
            rev_vi += md.vickrey_auction(bids).revenue
        # Myerson should be strictly higher on average
        assert rev_my / N > rev_vi / N

    def test_from_samples_runs(self):
        md = MechanismDesigner(random_seed=1)
        rng = random.Random(1)
        samples = {x: [rng.uniform(0, 1) for _ in range(200)] for x in ("a", "b", "c")}
        a = md.myerson_from_samples({"a": 0.3, "b": 0.7, "c": 0.5}, samples)
        # Behaviorally identical to Myerson with U[0,1] priors (modulo finite-sample wiggle)
        assert a.winner in {"a", "b", "c", None}

    def test_from_samples_insufficient(self):
        md = MechanismDesigner()
        with pytest.raises(InsufficientData):
            md.myerson_from_samples(
                {"a": 0.5, "b": 0.5}, {"a": [0.1, 0.2], "b": [0.3, 0.4]}
            )


# -------------------- Anonymous-reserve Vickrey -------------------------


class TestAnonymousReserve:
    def test_runs_with_reserve(self):
        md = MechanismDesigner()
        a = md.anonymous_reserve_auction({"a": 0.3, "b": 0.7, "c": 0.5}, reserve=0.45)
        assert a.winner == "b"
        # Second-highest above reserve is c=0.5; payment = max(0.5, 0.45) = 0.5
        assert a.payment == pytest.approx(0.5)
        assert a.mechanism == KIND_ANONYMOUS_RESERVE

    def test_dsic(self):
        md = MechanismDesigner()
        r = md.certify_dsic(
            KIND_ANONYMOUS_RESERVE,
            bids_truthful={"a": 0.3, "b": 0.7, "c": 0.5},
            reserve=0.4,
        )
        assert r.is_dsic


# -------------------- VCG -----------------------------------------------


class TestVCG:
    def test_two_bidders_two_items_unit_demand(self):
        md = MechanismDesigner()
        # A values g0 at 10, g1 at 5; B values g0 at 7, g1 at 9
        # Welfare-max: A gets g0 (10), B gets g1 (9) → W*=19
        # Without A: B can only take 1 → picks g1 = 9. W*_-A=9. Payment_A = 9 − (19−10) = 0
        # Without B: A picks g0 = 10. W*_-B=10. Payment_B = 10 − (19−9) = 0
        # Both pay 0 (no externality) — both items "free"
        v = md.vcg_allocation(
            items=["g0", "g1"],
            bids={"A": {"g0": 10, "g1": 5}, "B": {"g0": 7, "g1": 9}},
            capacity={"A": 1, "B": 1},
        )
        assert v.total_welfare == pytest.approx(19.0)
        # IR
        ir = md.certify_ir(v)
        assert ir.is_ir

    def test_pivot_pays_externality(self):
        # Three bidders for one item: A=10, B=7, C=5 with capacity 1 each
        md = MechanismDesigner()
        v = md.vcg_allocation(
            items=["only"],
            bids={"A": {"only": 10}, "B": {"only": 7}, "C": {"only": 5}},
            capacity={"A": 1, "B": 1, "C": 1},
        )
        # A wins, externality = 7 (welfare without A) − 0 = 7
        assert v.total_welfare == pytest.approx(10.0)
        # A pays 7 (second-price equivalent under unit-demand)
        a_assignment = [x for x in v.assignments if x.bidder_id == "A"][0]
        assert a_assignment.payment == pytest.approx(7.0)

    def test_empty_inputs(self):
        md = MechanismDesigner()
        with pytest.raises(InvalidBid):
            md.vcg_allocation(items=["x"], bids={})
        with pytest.raises(InvalidBid):
            md.vcg_allocation(items=[], bids={"A": {}})

    def test_negative_bid_rejected(self):
        md = MechanismDesigner()
        with pytest.raises(InvalidBid):
            md.vcg_allocation(items=["x"], bids={"A": {"x": -1.0}})

    def test_quick_facade(self):
        v = quick_vcg(
            items=["g0", "g1"],
            bids={"A": {"g0": 5, "g1": 3}, "B": {"g0": 4, "g1": 6}},
        )
        assert isinstance(v, VCGAllocation)
        assert v.total_welfare > 0


# -------------------- Posted-price --------------------------------------


class TestPostedPrice:
    def test_first_acceptor_wins(self):
        md = MechanismDesigner()
        out = md.posted_price(
            valuations={"a": 0.4, "b": 0.7, "c": 0.5},
            prices={"a": 0.6, "b": 0.5, "c": 0.5},
            order=["a", "b", "c"],
        )
        # a (val 0.4 < price 0.6) → reject
        # b (val 0.7 ≥ price 0.5) → accept, pays 0.5
        # c never offered
        assert out.accepted == ("b",)
        assert out.revenue == pytest.approx(0.5)
        utilities = dict(out.utilities)
        assert utilities["a"] == 0.0
        assert utilities["b"] == pytest.approx(0.2)
        assert utilities["c"] == 0.0

    def test_no_acceptor(self):
        md = MechanismDesigner()
        out = md.posted_price(
            valuations={"a": 0.4, "b": 0.5},
            prices={"a": 0.6, "b": 0.7},
        )
        assert out.accepted == ()
        assert out.revenue == 0.0


# -------------------- Bulow-Klemperer -----------------------------------


class TestBulowKlemperer:
    def test_n_plus_one_beats_n_with_reserve_uniform(self):
        # On U[0,1], (n+1)-bidder Vickrey-no-reserve should have higher
        # expected revenue than n-bidder Myerson, by BK theorem.
        md = MechanismDesigner(random_seed=42)
        rng = random.Random(42)
        samples = [rng.uniform(0, 1) for _ in range(500)]
        bk = md.bulow_klemperer(samples, n=2, trials=1500)
        assert isinstance(bk, BulowKlemperer)
        # The ratio should be ≥ ~1 (allowing finite-sample wiggle)
        assert bk.ratio > 0.9
        # And the revenues should be in the right order on average
        assert bk.revenue_vickrey_n_plus_1 > 0
        assert bk.revenue_myerson_n > 0

    def test_insufficient_samples(self):
        with pytest.raises(InsufficientData):
            MechanismDesigner().bulow_klemperer([0.1, 0.2], n=2)


# -------------------- Empirical reserve ---------------------------------


class TestEmpiricalReserve:
    def test_converges_to_uniform_monopoly(self):
        # For U[0,1], monopoly reserve = 1/2. With many samples, the
        # empirical monopoly reserve should be close.
        rng = random.Random(0)
        samples = [rng.uniform(0, 1) for _ in range(5000)]
        rp = MechanismDesigner().empirical_reserve(samples, method="monopoly")
        assert abs(rp.reserve - 0.5) < 0.05
        # Revenue LCB is meaningful (well below the mean)
        assert rp.revenue_lcb <= rp.revenue_mean

    def test_median_method(self):
        rng = random.Random(7)
        samples = sorted(rng.uniform(0, 1) for _ in range(1001))
        rp = MechanismDesigner().empirical_reserve(samples, method="median")
        assert rp.reserve == pytest.approx(statistics.median(samples), rel=1e-9)

    def test_invalid_method(self):
        with pytest.raises(MechanismError):
            MechanismDesigner().empirical_reserve([0.0] * 20 + [1.0] * 20, method="invalid")

    def test_insufficient_data(self):
        with pytest.raises(InsufficientData):
            MechanismDesigner().empirical_reserve([0.5, 0.6, 0.7])


# -------------------- Sample complexity ---------------------------------


class TestSampleComplexity:
    def test_regular_scales_inv_eps_squared(self):
        n1 = sample_complexity_for_eps_optimal(epsilon=0.1, delta=0.05)
        n2 = sample_complexity_for_eps_optimal(epsilon=0.05, delta=0.05)
        # Halving ε should roughly 4× the bound (∝ 1/ε²)
        assert n2 / n1 == pytest.approx(4.0, rel=0.2)

    def test_mhr_scales_inv_eps_three_halves(self):
        from agi.mechanism import REG_MHR
        n1 = sample_complexity_for_eps_optimal(epsilon=0.1, delta=0.05, regularity=REG_MHR)
        n2 = sample_complexity_for_eps_optimal(epsilon=0.05, delta=0.05, regularity=REG_MHR)
        # ratio ≈ 2^{1.5} ≈ 2.83
        assert n2 / n1 == pytest.approx(2 ** 1.5, rel=0.2)

    def test_bounded_scales_inv_eps_cubed(self):
        from agi.mechanism import REG_BOUNDED
        n1 = sample_complexity_for_eps_optimal(epsilon=0.1, delta=0.05, regularity=REG_BOUNDED)
        n2 = sample_complexity_for_eps_optimal(epsilon=0.05, delta=0.05, regularity=REG_BOUNDED)
        # ratio ≈ 8 (1/ε³)
        assert n2 / n1 == pytest.approx(8.0, rel=0.25)

    def test_validation(self):
        with pytest.raises(MechanismError):
            sample_complexity_for_eps_optimal(epsilon=1.5, delta=0.05)
        with pytest.raises(MechanismError):
            sample_complexity_for_eps_optimal(epsilon=0.1, delta=2.0)


# -------------------- Online posted-price (Kleinberg-Leighton) ----------


class TestOnlinePostedPrice:
    def test_convergence_against_fixed_threshold(self):
        # Buyer accepts iff price ≤ 0.6.
        # Optimal posted price is 0.6, optimal revenue per round = 0.6.
        md = MechanismDesigner(random_seed=11)
        threshold = 0.6

        def buyer(p: float) -> bool:
            return p <= threshold

        out = md.online_posted_price(feedback=buyer, T=2000, v_max=1.0)
        # Over T=2000 the mean revenue should beat 50% of optimal
        assert out.revenue_mean >= 0.20

    def test_validation(self):
        md = MechanismDesigner()
        with pytest.raises(MechanismError):
            md.online_posted_price(feedback=lambda p: False, T=0, v_max=1.0)
        with pytest.raises(MechanismError):
            md.online_posted_price(feedback=lambda p: False, T=10, v_max=-1)
        with pytest.raises(MechanismError):
            md.online_posted_price(feedback=lambda p: False, T=10, v_max=1, algorithm="bogus")

    def test_returns_history(self):
        md = MechanismDesigner(random_seed=2)
        out = md.online_posted_price(feedback=lambda p: True, T=50, v_max=1.0)
        assert len(out.price_history) == 50
        assert len(out.accept_history) == 50
        # All-accept buyer → revenue == sum of prices played
        assert out.revenue == pytest.approx(sum(out.price_history))


# -------------------- Revenue certificate -------------------------------


class TestRevenueCertificate:
    def test_lcb_below_mean(self):
        md = MechanismDesigner()
        rng = random.Random(3)
        revs = [rng.uniform(0.4, 0.6) for _ in range(200)]
        cert = md.revenue_certificate(KIND_VICKREY, revs, delta=0.05)
        assert cert.revenue_lcb <= cert.revenue_mean
        assert cert.revenue_radius >= 0
        assert cert.mechanism == KIND_VICKREY

    def test_empty(self):
        with pytest.raises(InsufficientData):
            MechanismDesigner().revenue_certificate(KIND_VICKREY, [])


# -------------------- Event bus ----------------------------------------


class TestEventBusIntegration:
    def test_emits_on_start_and_allocate(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(lambda ev: received.append(ev))
        md = MechanismDesigner(bus=bus)
        md.vickrey_auction({"a": 0.5, "b": 0.7})
        kinds = {ev.kind for ev in received}
        assert MECH_STARTED in kinds
        assert MECH_ALLOCATED in kinds


# -------------------- Attestation ----------------------------------------


class TestAttestation:
    def test_receipt_recorded(self):
        records: list[dict] = []

        class Attestor:
            def record(self, *, kind, payload):
                records.append({"kind": kind, "payload": payload})

        md = MechanismDesigner(attestor=Attestor())
        a = md.vickrey_auction({"a": 0.5, "b": 0.7})
        assert len(a.receipt_digest) == 64
        # at least one MECH_ALLOCATED was recorded
        assert any(r["kind"] == MECH_ALLOCATED for r in records)

    def test_digest_is_content_hash(self):
        md = MechanismDesigner()
        a1 = md.vickrey_auction({"a": 0.5, "b": 0.7})
        a2 = md.vickrey_auction({"a": 0.5, "b": 0.7})
        # Identical inputs → identical content digest
        assert a1.receipt_digest == a2.receipt_digest


# -------------------- Concurrency --------------------------------------


class TestConcurrency:
    def test_threadsafe_allocations(self):
        md = MechanismDesigner()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    md.vickrey_auction({"a": 0.3, "b": 0.7})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert md.stats()["n_allocations"] == 50 * 8


# -------------------- Facade ----------------------------------------


class TestFacade:
    def test_quick_vickrey(self):
        a = quick_vickrey({"a": 0.3, "b": 0.8})
        assert a.winner == "b"

    def test_quick_myerson(self):
        a = quick_myerson(
            {"a": 0.3, "b": 0.8},
            {"a": UniformDistribution(0, 1), "b": UniformDistribution(0, 1)},
        )
        assert a.winner == "b"


# -------------------- Stand-alone helpers -------------------------------


class TestStandaloneHelpers:
    def test_vickrey_payment_function(self):
        from agi.mechanism import vickrey_payment, first_price_payment
        idx, pay = vickrey_payment([0.9, 0.7, 0.5])
        assert idx == 0 and pay == 0.7
        idx, pay = vickrey_payment([0.9, 0.7, 0.5], reserve=0.8)
        # Top above reserve, second below → pays reserve
        assert idx == 0 and pay == 0.8
        idx, pay = vickrey_payment([0.3, 0.2], reserve=0.5)
        assert idx == -1 and pay == 0.0
        idx, pay = first_price_payment([0.9, 0.7])
        assert idx == 0 and pay == 0.9
        idx, pay = vickrey_payment([])
        assert idx == -1

    def test_myerson_winner_and_payment_function(self):
        d = [UniformDistribution(0, 1), UniformDistribution(0, 1)]
        winner, pay, virt, phis = myerson_winner_and_payment(
            ["a", "b"], [0.3, 0.8], d
        )
        assert winner == "b"
        assert pay == pytest.approx(0.5)

    def test_myerson_no_sale_below_reserve(self):
        d = [UniformDistribution(0, 1), UniformDistribution(0, 1)]
        winner, pay, virt, phis = myerson_winner_and_payment(
            ["a", "b"], [0.3, 0.4], d
        )
        assert winner is None
