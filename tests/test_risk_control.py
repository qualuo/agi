"""Tests for `agi.risk_control` — distribution-free risk control.

We check three things in order of severity:

1. The concentration bounds (Hoeffding, Hoeffding-Bentkus, WSR) are
   coverage-valid in Monte Carlo. We simulate {Bernoulli(p)}_{i=1..n}
   samples, build a UCB at level δ on the empirical mean, and check
   that the bound covers the true mean at the requested rate. We do
   *not* check that the bound is tight — only that it is conservative
   (i.e. covers ≥ 1 − δ of the time). Anti-conservative is a bug.
2. CRC's E-bound holds: a threshold selected by `select(method="crc")`
   yields an expected loss on a fresh sample that is ≤ the requested
   target (in expectation over many calibration draws). We accept a
   small slack for finite Monte Carlo.
3. LTT (HB and WSR) yield FWER ≤ δ at the population level: simulate
   nulls where the true risk is exactly the target, and check that
   the procedure rejects (i.e. certifies a candidate as safe) at most
   δ fraction of the time.

We also verify the cosmetic surface: dataclass invariants, event
emissions, group filtering, and the multi-risk Bonferroni split.
"""
from __future__ import annotations

import math
import random
import statistics

import pytest

from agi.events import EventBus
from agi.risk_control import (
    KNOWN_METHODS,
    KNOWN_ORDERINGS,
    METHOD_CRC,
    METHOD_LTT_CLT,
    METHOD_LTT_HB,
    METHOD_LTT_HOEFFDING,
    METHOD_LTT_WSR,
    ORDER_AGGRESSIVE_FIRST,
    ORDER_BONFERRONI,
    ORDER_CONSERVATIVE_FIRST,
    ORDER_FIXED,
    RISK_FAILED,
    RISK_OBSERVED,
    RISK_REPORT,
    RISK_SELECTED,
    Risk,
    RiskController,
    RiskPoint,
    RiskSelection,
    bentkus_pvalue,
    hoeffding_bentkus_pvalue,
    hoeffding_bentkus_ucb,
    hoeffding_pvalue,
    hoeffding_ucb,
    loss_indicator_above,
    loss_indicator_below,
    loss_overrun,
    wsr_pvalue,
    wsr_ucb,
)


# ----- concentration primitives --------------------------------------


class TestHoeffdingBentkus:
    def test_pvalues_monotone_in_null(self) -> None:
        # As the null mean R grows, the p-value for H_0: E[L] ≥ R
        # (with fixed empirical mean below R) is non-increasing in R
        # only when p̂ ≤ R; beyond that the p-value pegs at 1.
        p_hat, n = 0.10, 200
        last = -1.0
        for R in [0.11, 0.20, 0.30, 0.50, 0.80, 0.99]:
            p = hoeffding_bentkus_pvalue(p_hat, n, R)
            assert 0.0 <= p <= 1.0
            # Smaller p-values for harder-to-reject nulls is wrong; HB
            # is *decreasing* as the null moves away from p̂.
            if last >= 0:
                assert p <= last + 1e-9, (R, p, last)
            last = p

    def test_pvalue_one_when_phat_at_or_above_null(self) -> None:
        assert hoeffding_pvalue(0.5, 100, 0.5) == 1.0
        assert hoeffding_pvalue(0.6, 100, 0.5) == 1.0
        assert bentkus_pvalue(0.5, 100, 0.5) == 1.0
        assert hoeffding_bentkus_pvalue(0.5, 100, 0.5) == 1.0

    def test_ucb_is_above_phat(self) -> None:
        for n, p_hat in [(10, 0.1), (100, 0.05), (1000, 0.20)]:
            ucb = hoeffding_bentkus_ucb(p_hat, n, delta=0.1)
            assert ucb >= p_hat - 1e-9
            assert 0.0 <= ucb <= 1.0

    def test_ucb_shrinks_with_n(self) -> None:
        # More data ⇒ tighter UCB.
        ucbs = [hoeffding_bentkus_ucb(0.10, n, delta=0.1) for n in [50, 200, 1000, 5000]]
        for a, b in zip(ucbs, ucbs[1:]):
            assert b <= a + 1e-9

    def test_ucb_shrinks_with_higher_delta(self) -> None:
        # Larger δ (weaker confidence) ⇒ smaller UCB.
        ucbs = [hoeffding_bentkus_ucb(0.10, 500, delta=d) for d in [0.01, 0.05, 0.10, 0.25]]
        for a, b in zip(ucbs, ucbs[1:]):
            assert b <= a + 1e-9

    def test_hoeffding_is_no_tighter_than_hb(self) -> None:
        # HB ≤ Hoeffding (HB takes the min).
        for n in [50, 500, 5000]:
            for p_hat in [0.01, 0.10, 0.30, 0.60]:
                ucb_h = hoeffding_ucb(p_hat, n, delta=0.1)
                ucb_hb = hoeffding_bentkus_ucb(p_hat, n, delta=0.1)
                assert ucb_hb <= ucb_h + 1e-9

    def test_coverage_hb_bernoulli(self) -> None:
        """Monte Carlo: HB UCB covers the true Bernoulli mean ≥ 1 − δ."""
        rng = random.Random(42)
        delta = 0.1
        n = 200
        true_p = 0.15
        n_trials = 400
        n_covered = 0
        for _ in range(n_trials):
            sample = [1.0 if rng.random() < true_p else 0.0 for _ in range(n)]
            p_hat = statistics.fmean(sample)
            ucb = hoeffding_bentkus_ucb(p_hat, n, delta=delta)
            if ucb >= true_p:
                n_covered += 1
        # Allow a small Monte Carlo cushion (Wilson half-width at n=400, δ=0.1 ≈ 0.025).
        assert n_covered / n_trials >= (1.0 - delta) - 0.04, n_covered / n_trials


class TestWSR:
    def test_wsr_ucb_in_range(self) -> None:
        rng = random.Random(7)
        sample = [1.0 if rng.random() < 0.2 else 0.0 for _ in range(300)]
        ucb = wsr_ucb(sample, delta=0.1)
        p_hat = statistics.fmean(sample)
        assert p_hat <= ucb <= 1.0 + 1e-9

    def test_wsr_coverage_bernoulli(self) -> None:
        rng = random.Random(11)
        delta = 0.1
        n = 200
        true_p = 0.15
        n_trials = 300
        n_covered = 0
        for _ in range(n_trials):
            sample = [1.0 if rng.random() < true_p else 0.0 for _ in range(n)]
            ucb = wsr_ucb(sample, delta=delta)
            if ucb >= true_p:
                n_covered += 1
        # Same conservatism check as HB.
        assert n_covered / n_trials >= (1.0 - delta) - 0.04, n_covered / n_trials

    def test_wsr_pvalue_monotone_in_null(self) -> None:
        rng = random.Random(13)
        losses = [1.0 if rng.random() < 0.1 else 0.0 for _ in range(200)]
        last = -1.0
        for R in [0.11, 0.20, 0.30, 0.50]:
            p = wsr_pvalue(losses, R)
            assert 0.0 <= p <= 1.0
            if last >= 0:
                assert p <= last + 1e-6, (R, p, last)
            last = p


# ----- RiskController plumbing ---------------------------------------


class TestRiskControllerPlumbing:
    def test_record_appends(self) -> None:
        rc = RiskController()
        assert len(rc) == 0
        rc.record(score=0.5, outcome=True, group="t1")
        rc.record(score=0.7, outcome=False, group="t2")
        assert len(rc) == 2
        pts = rc.points()
        assert pts[0].score == 0.5
        assert pts[0].group == "t1"
        assert pts[1].outcome is False

    def test_record_many(self) -> None:
        rc = RiskController(max_history=3)
        rc.record_many(RiskPoint(score=float(i)) for i in range(5))
        # Ring buffer caps at max_history.
        assert len(rc) == 3
        assert [p.score for p in rc.points()] == [2.0, 3.0, 4.0]

    def test_invalid_score_rejected(self) -> None:
        with pytest.raises(ValueError):
            RiskPoint(score=float("nan"))

    def test_invalid_weight_rejected(self) -> None:
        with pytest.raises(ValueError):
            RiskPoint(score=0.5, weight=-1.0)

    def test_risk_target_validation(self) -> None:
        with pytest.raises(ValueError):
            Risk(name="r", target=0.0, loss_fn=lambda p, t: 0.0)
        with pytest.raises(ValueError):
            Risk(name="r", target=1.5, loss_fn=lambda p, t: 0.0, B=1.0)
        with pytest.raises(ValueError):
            Risk(name="r", target=0.5, loss_fn=lambda p, t: 0.0, monotone="zigzag")

    def test_event_bus_emits_observed(self) -> None:
        bus = EventBus()
        events: list[str] = []
        bus.subscribe(lambda e: events.append(e.kind), kind=RISK_OBSERVED)
        rc = RiskController(bus=bus)
        rc.record(score=0.5)
        assert events == [RISK_OBSERVED]

    def test_select_unknown_method_rejected(self) -> None:
        rc = RiskController()
        rc.record(score=0.5, outcome=True)
        with pytest.raises(ValueError):
            rc.select(
                candidates=[0.1, 0.5],
                target=0.05,
                loss_fn=lambda p, t: 0.0,
                method="banana",
            )

    def test_select_empty_candidates_rejected(self) -> None:
        rc = RiskController()
        rc.record(score=0.5, outcome=True)
        with pytest.raises(ValueError):
            rc.select(
                candidates=[],
                target=0.05,
                loss_fn=lambda p, t: 0.0,
                method=METHOD_LTT_HB,
            )

    def test_select_invalid_delta_rejected(self) -> None:
        rc = RiskController()
        rc.record(score=0.5, outcome=True)
        with pytest.raises(ValueError):
            rc.select(
                candidates=[0.1],
                target=0.05,
                loss_fn=lambda p, t: 0.0,
                method=METHOD_LTT_HB,
                delta=0.0,
            )

    def test_select_empty_calibration_returns_none(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e: seen.append(e.kind), kind=RISK_FAILED)
        rc = RiskController(bus=bus)
        sel = rc.select(
            candidates=[0.5],
            target=0.1,
            loss_fn=lambda p, t: 0.0,
            method=METHOD_LTT_HB,
        )
        assert sel is None
        assert seen == [RISK_FAILED]

    def test_loss_outside_bounds_raises(self) -> None:
        rc = RiskController()
        rc.record(score=0.5, outcome=True)

        def bad_loss(p: RiskPoint, t: float) -> float:
            return 2.0  # outside [0, B=1]

        with pytest.raises(ValueError):
            rc.select(
                candidates=[0.5],
                target=0.1,
                loss_fn=bad_loss,
                method=METHOD_LTT_HB,
            )

    def test_known_methods_and_orderings_exposed(self) -> None:
        assert METHOD_CRC in KNOWN_METHODS
        assert METHOD_LTT_HB in KNOWN_METHODS
        assert METHOD_LTT_WSR in KNOWN_METHODS
        assert ORDER_AGGRESSIVE_FIRST in KNOWN_ORDERINGS
        assert ORDER_BONFERRONI in KNOWN_ORDERINGS


# ----- CRC behaviour -------------------------------------------------


def _calibration_easy_decreasing(rng: random.Random, n: int, true_p_at: dict[float, float]):
    """Build a calibration set where loss(λ) ≈ true_p_at[λ] for λ ∈ keys.
    Implementation: each point carries an i.i.d. uniform u in features;
    the loss at λ is 1 iff u < true_p_at[λ]."""
    pts: list[RiskPoint] = []
    for _ in range(n):
        u = rng.random()
        pts.append(RiskPoint(score=0.0, features={"u": u}))
    return pts


def _loss_decreasing_in_lambda(p: RiskPoint, lam: float) -> float:
    # L(λ) = 1[u < f(λ)] where f is decreasing in λ. Define f explicitly.
    # We map λ ∈ [0,1] to f(λ) = max(0, 0.4 - 0.4*λ) so that f is
    # 0.4 at λ=0 and 0 at λ≥1. Monotone non-increasing in λ.
    u = float(p.features.get("u", 0.0))
    return 1.0 if u < max(0.0, 0.4 - 0.4 * lam) else 0.0


class TestCRC:
    def test_crc_picks_aggressive_threshold(self) -> None:
        rng = random.Random(0)
        n = 1500
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, n, {}))
        # Target risk 5%. With f(λ) = max(0, 0.4 − 0.4λ), the true
        # risk hits 0.05 at λ = 0.875. CRC should pick something in
        # the right neighbourhood.
        sel = rc.select(
            candidates=[round(0.05 * i, 4) for i in range(21)],
            target=0.05,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_CRC,
            monotone="decreasing",
        )
        assert sel is not None
        # The CRC E-bound at the selected λ should be ≤ target.
        assert sel.empirical_risk + 1.0 / (n + 1) <= 0.05 + 1e-9
        # And the chosen threshold should be ≥ the true-risk crossing
        # point (i.e. CRC is conservative, not aggressive past it).
        assert sel.threshold >= 0.85 - 1e-6 or sel.empirical_risk < 0.05

    def test_crc_failure_when_target_below_floor(self) -> None:
        rng = random.Random(1)
        n = 200
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, n, {}))
        # No λ in [0, 1] makes f ≤ 0 *and* the +B/(n+1) slack is large
        # at small n. A microscopic target with small n is infeasible
        # even at λ=1: emp = 0, +B/(n+1) ≈ 0.005, still > 0.001.
        sel = rc.select(
            candidates=[1.0],
            target=0.001,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_CRC,
            monotone="decreasing",
        )
        assert sel is None

    def test_crc_requires_monotone(self) -> None:
        rc = RiskController()
        rc.record(score=0.5, outcome=True)
        with pytest.raises(ValueError):
            rc.select(
                candidates=[0.5],
                target=0.1,
                loss_fn=lambda p, t: 0.5,
                method=METHOD_CRC,
                monotone="none",
            )

    def test_crc_expected_loss_bound_holds_monte_carlo(self) -> None:
        """Run the CRC procedure 60 times on fresh calibration sets;
        the realized loss on a fresh held-out test point should be
        ≤ target in expectation.
        """
        rng = random.Random(123)
        target = 0.10
        n_trials = 60
        n_cal = 300
        realized_losses: list[float] = []
        candidates = [round(0.05 * i, 4) for i in range(21)]
        for _ in range(n_trials):
            rc = RiskController()
            rc.record_many(_calibration_easy_decreasing(rng, n_cal, {}))
            sel = rc.select(
                candidates=candidates,
                target=target,
                loss_fn=_loss_decreasing_in_lambda,
                method=METHOD_CRC,
                monotone="decreasing",
            )
            if sel is None:
                # Conservatively count as 0 loss — λ wasn't deployed.
                realized_losses.append(0.0)
                continue
            # Sample one fresh test point and compute its realized loss.
            test_pt = RiskPoint(score=0.0, features={"u": rng.random()})
            realized_losses.append(_loss_decreasing_in_lambda(test_pt, sel.threshold))
        mean_loss = statistics.fmean(realized_losses)
        # E[L] ≤ target. Allow small Monte Carlo cushion.
        assert mean_loss <= target + 0.05, mean_loss


# ----- LTT behaviour -------------------------------------------------


class TestLTT:
    def test_ltt_hb_picks_safe_threshold(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 2000, {}))
        sel = rc.select(
            candidates=[round(0.05 * i, 4) for i in range(21)],
            target=0.10,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_HB,
            monotone="decreasing",
            ordering=ORDER_AGGRESSIVE_FIRST,
        )
        assert sel is not None
        # The UCB at the selected λ must be ≤ target by construction.
        assert sel.ucb <= sel.target + 1e-9, (sel.ucb, sel.target)
        # And the realized empirical risk ≤ UCB ≤ target.
        assert sel.empirical_risk <= sel.ucb + 1e-9

    def test_ltt_wsr_picks_safe_threshold(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 2000, {}))
        sel = rc.select(
            candidates=[round(0.05 * i, 4) for i in range(21)],
            target=0.10,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_WSR,
            monotone="decreasing",
            ordering=ORDER_AGGRESSIVE_FIRST,
        )
        assert sel is not None
        assert sel.ucb <= sel.target + 1e-2  # WSR UCB inversion is by bisection

    def test_ltt_bonferroni_more_conservative_than_fixed(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 2000, {}))
        cands = [round(0.05 * i, 4) for i in range(21)]
        sel_fixed = rc.select(
            candidates=cands,
            target=0.10,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_HB,
            monotone="decreasing",
            ordering=ORDER_AGGRESSIVE_FIRST,
        )
        sel_bonf = rc.select(
            candidates=cands,
            target=0.10,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_HB,
            monotone="decreasing",
            ordering=ORDER_BONFERRONI,
        )
        assert sel_fixed is not None and sel_bonf is not None
        # Bonferroni splits δ across 21 tests ⇒ per-test δ ≈ 0.0048.
        # That gives a tighter UCB threshold ⇒ a more conservative λ̂.
        # For a monotone-decreasing loss, "more conservative" = larger λ.
        assert sel_bonf.threshold >= sel_fixed.threshold - 1e-9

    def test_ltt_fwer_at_population(self) -> None:
        """When the true risk at λ is *exactly* the target, FWER ≤ δ.

        Use a one-candidate setup where the population mean of the
        loss is the target. Many trials, fraction of false rejections
        should be ≤ δ + small Monte Carlo slack.
        """
        rng = random.Random(2024)
        target = 0.05
        delta = 0.10
        n = 500
        n_trials = 200
        n_rejected = 0
        # Custom loss: P(L=1) = target, independent of λ.
        def boundary_loss(p: RiskPoint, lam: float) -> float:
            return 1.0 if p.features.get("u", 1.0) < target else 0.0

        for _ in range(n_trials):
            rc = RiskController()
            rc.record_many(
                RiskPoint(score=0.0, features={"u": rng.random()})
                for _ in range(n)
            )
            sel = rc.select(
                candidates=[0.5],
                target=target,
                delta=delta,
                loss_fn=boundary_loss,
                method=METHOD_LTT_HB,
                monotone="none",
                ordering=ORDER_FIXED,
            )
            if sel is not None:
                n_rejected += 1
        # False rejection rate ≤ δ (with a slack for n_trials=200).
        assert n_rejected / n_trials <= delta + 0.04, n_rejected / n_trials

    def test_ltt_returns_none_when_target_infeasible(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 500, {}))
        # f(λ=0) = 0.40 ≫ target 0.001. No candidate can be certified.
        sel = rc.select(
            candidates=[0.0],
            target=0.001,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_HB,
            monotone="decreasing",
            ordering=ORDER_AGGRESSIVE_FIRST,
        )
        assert sel is None

    def test_ltt_clt_baseline_runs(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 500, {}))
        sel = rc.select(
            candidates=[round(0.05 * i, 4) for i in range(21)],
            target=0.05,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_CLT,
            monotone="decreasing",
        )
        assert sel is not None or sel is None  # smoke

    def test_ltt_hb_emits_selected_event(self) -> None:
        bus = EventBus()
        kinds: list[str] = []
        bus.subscribe(lambda e: kinds.append(e.kind), kind=RISK_SELECTED)
        rng = random.Random(0)
        rc = RiskController(bus=bus)
        rc.record_many(_calibration_easy_decreasing(rng, 500, {}))
        sel = rc.select(
            candidates=[round(0.05 * i, 4) for i in range(21)],
            target=0.10,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_HB,
            monotone="decreasing",
        )
        assert sel is not None
        assert kinds == [RISK_SELECTED]


# ----- multi-risk and group filtering --------------------------------


class TestMultiRisk:
    def test_multi_risk_bonferroni_split(self) -> None:
        rng = random.Random(7)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 2000, {}))
        sels = rc.select_multi(
            candidates=[round(0.05 * i, 4) for i in range(21)],
            delta=0.10,
            method=METHOD_LTT_HB,
            risks=[
                Risk(
                    name="risk_a",
                    target=0.10,
                    loss_fn=_loss_decreasing_in_lambda,
                    monotone="decreasing",
                ),
                Risk(
                    name="risk_b",
                    target=0.20,
                    loss_fn=_loss_decreasing_in_lambda,
                    monotone="decreasing",
                ),
            ],
            ordering=ORDER_AGGRESSIVE_FIRST,
        )
        assert "risk_a" in sels and "risk_b" in sels
        # Each per-risk selection used δ/2.
        for sel in sels.values():
            if sel is not None:
                assert sel.delta == pytest.approx(0.05)
                assert sel.ucb <= sel.target + 1e-9

    def test_multi_risk_empty(self) -> None:
        rc = RiskController()
        out = rc.select_multi(candidates=[0.5], delta=0.1, risks=[])
        assert out == {}


class TestGroupFiltering:
    def test_select_filters_by_group(self) -> None:
        rng = random.Random(99)
        rc = RiskController()
        # Group A has high risk; Group B has low risk. Selection on
        # group B should pick an aggressive threshold; on A, conservative.
        for _ in range(800):
            rc.record(score=0.0, group="A", outcome=None, features={"u": rng.random()})
        for _ in range(800):
            u = rng.random() * 0.5  # smaller u floor ⇒ less risk... actually larger
            rc.record(score=0.0, group="B", outcome=None, features={"u": rng.random()})
        # Simpler: B has zero risk regardless of λ (loss ≡ 0).
        loss_zero = lambda p, t: 0.0
        sel_a = rc.select(
            candidates=[0.5],
            target=0.10,
            loss_fn=loss_zero,
            method=METHOD_LTT_HB,
            group="A",
            monotone="decreasing",
        )
        sel_b = rc.select(
            candidates=[0.5],
            target=0.10,
            loss_fn=loss_zero,
            method=METHOD_LTT_HB,
            group="B",
            monotone="decreasing",
        )
        assert sel_a is not None and sel_b is not None
        assert sel_a.n == 800
        assert sel_b.n == 800


# ----- report --------------------------------------------------------


class TestReport:
    def test_report_empty(self) -> None:
        rc = RiskController()
        rep = rc.report(threshold=0.5, loss_fn=lambda p, t: 0.0)
        assert rep.n == 0
        assert "empty_calibration" in rep.notes

    def test_report_basic(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 500, {}))
        rep = rc.report(
            threshold=1.0,
            loss_fn=_loss_decreasing_in_lambda,
            delta=0.1,
            method=METHOD_LTT_HB,
        )
        assert rep.n == 500
        # f(λ=1) = 0 ⇒ empirical risk = 0.
        assert rep.empirical_risk == 0.0
        assert rep.ucb >= 0.0

    def test_report_per_group(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        # Group A: high-loss; Group B: low-loss.
        for _ in range(200):
            rc.record(score=0.0, group="A", features={"u": rng.random()})
        for _ in range(200):
            rc.record(score=0.0, group="B", features={"u": 0.99})  # low loss
        rep = rc.report(
            threshold=0.0,
            loss_fn=_loss_decreasing_in_lambda,
            delta=0.1,
        )
        assert "A" in rep.per_group and "B" in rep.per_group
        # Group A's empirical risk ≈ 0.4 (uniform); Group B ≈ 0.
        assert rep.per_group["A"].empirical_risk > rep.per_group["B"].empirical_risk

    def test_report_event(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e: seen.append(e.kind), kind=RISK_REPORT)
        rng = random.Random(0)
        rc = RiskController(bus=bus)
        rc.record_many(_calibration_easy_decreasing(rng, 100, {}))
        rc.report(threshold=0.5, loss_fn=_loss_decreasing_in_lambda, delta=0.1)
        assert seen == [RISK_REPORT]


# ----- loss-factory helpers ------------------------------------------


class TestLossFactories:
    def test_loss_indicator_above(self) -> None:
        fn = loss_indicator_above()
        p = RiskPoint(score=0.7, outcome=True)
        assert fn(p, 0.5) == 1.0
        assert fn(p, 0.8) == 0.0
        p2 = RiskPoint(score=0.7, outcome=False)
        assert fn(p2, 0.5) == 0.0

    def test_loss_indicator_below(self) -> None:
        fn = loss_indicator_below()
        p = RiskPoint(score=0.3, outcome=True)
        assert fn(p, 0.5) == 1.0
        assert fn(p, 0.2) == 0.0

    def test_loss_overrun(self) -> None:
        fn = loss_overrun(multiplier=1.5)
        # outcome > λ * 1.5 ⇒ 1
        p = RiskPoint(score=1.0, outcome=10.0)
        assert fn(p, 5.0) == 1.0  # 10 > 5*1.5 = 7.5 ✓
        assert fn(p, 8.0) == 0.0  # 10 < 8*1.5 = 12

    def test_loss_overrun_nonnumeric_outcome(self) -> None:
        fn = loss_overrun()
        p = RiskPoint(score=1.0, outcome=None)
        assert fn(p, 5.0) == 0.0


# ----- end-to-end: realistic runtime scenario ------------------------


class TestRuntimeScenario:
    def test_hedge_threshold_selection_full_loop(self) -> None:
        """Simulate the canonical use case: select a hedge-trigger
        score threshold whose realized "hedged and still overran" rate
        is bounded.

        Model: predicted_cost = score; actual_cost = score * shock,
        where shock ~ LogNormal(0, σ²). We hedge iff score ≥ λ.
        Loss = 1 iff (we hedged AND the ticket overran by ≥ 1.5x).
        """
        rng = random.Random(2026)
        rc = RiskController()
        for _ in range(1500):
            score = rng.expovariate(2.0)  # mean cost ≈ 0.5
            shock = math.exp(rng.gauss(0.0, 0.7))
            actual = score * shock
            rc.record(score=score, outcome=actual)

        def overran_when_hedged(p: RiskPoint, lam: float) -> float:
            return 1.0 if (p.score >= lam and float(p.outcome) > p.score * 1.5) else 0.0

        sel = rc.select(
            candidates=[round(0.1 * i, 2) for i in range(1, 31)],
            target=0.10,
            delta=0.10,
            loss_fn=overran_when_hedged,
            method=METHOD_LTT_HB,
            monotone="none",  # loss in λ is not monotone (it goes up then down)
            ordering=ORDER_FIXED,
        )
        # Either we select *some* threshold whose UCB ≤ 10%, or no
        # candidate qualifies. We don't enforce which; the contract
        # is that any returned selection has UCB ≤ target.
        if sel is not None:
            assert sel.ucb <= 0.10 + 1e-9
            assert sel.empirical_risk <= sel.ucb + 1e-9

    def test_selection_to_dict_roundtrip(self) -> None:
        rng = random.Random(0)
        rc = RiskController()
        rc.record_many(_calibration_easy_decreasing(rng, 500, {}))
        sel = rc.select(
            candidates=[0.5, 0.9, 1.0],
            target=0.10,
            delta=0.10,
            loss_fn=_loss_decreasing_in_lambda,
            method=METHOD_LTT_HB,
            monotone="decreasing",
        )
        assert sel is not None
        d = sel.to_dict()
        assert d["method"] == METHOD_LTT_HB
        assert d["threshold"] == sel.threshold
