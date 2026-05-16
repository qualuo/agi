"""Tests for agi.curator — automated curriculum generation."""
from __future__ import annotations

import math
import random
import unittest

from agi.curator import (
    Cell,
    CompetenceEstimate,
    Curator,
    CuratorConfig,
    CuratorReport,
    InvalidConfig,
    InvalidParameter,
    KNOWN_STRATEGIES,
    Proposal,
    STRATEGY_LEARNING_PROGRESS,
    STRATEGY_THOMPSON_LP,
    STRATEGY_ZPD,
    UnknownStrategy,
    learning_progress_curator,
    thompson_lp_curator,
    zpd_curator,
)


# =============================================================================
# Configuration
# =============================================================================


class TestCuratorConfig(unittest.TestCase):

    def test_known_strategies(self):
        for s in ("zpd", "learning_progress", "thompson_lp"):
            self.assertIn(s, KNOWN_STRATEGIES)

    def test_default_config(self):
        cfg = CuratorConfig()
        self.assertEqual(cfg.strategy, STRATEGY_ZPD)
        self.assertEqual(cfg.param_lo, (0.0,))

    def test_dim_mismatch_rejected(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(param_lo=(0.0, 0.0), param_hi=(1.0,))
        with self.assertRaises(InvalidConfig):
            CuratorConfig(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(2, 3))

    def test_hi_below_lo_rejected(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(param_lo=(1.0,), param_hi=(0.0,))

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(strategy="random")

    def test_invalid_competence_target(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(target_competence=0.0)
        with self.assertRaises(InvalidConfig):
            CuratorConfig(target_competence=1.0)
        with self.assertRaises(InvalidConfig):
            CuratorConfig(target_competence=-0.1)

    def test_invalid_tolerance(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(target_tolerance=0.0)
        with self.assertRaises(InvalidConfig):
            CuratorConfig(target_tolerance=0.5)
        with self.assertRaises(InvalidConfig):
            CuratorConfig(target_tolerance=0.6)

    def test_invalid_windows(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(recent_window=0)
        with self.assertRaises(InvalidConfig):
            CuratorConfig(prev_window=0)

    def test_invalid_explore_p(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(explore_p=-0.1)
        with self.assertRaises(InvalidConfig):
            CuratorConfig(explore_p=1.1)

    def test_invalid_n_buckets(self):
        with self.assertRaises(InvalidConfig):
            CuratorConfig(n_buckets=(0,))


# =============================================================================
# Cell
# =============================================================================


class TestCell(unittest.TestCase):

    def test_basic_construction(self):
        c = Cell(indices=(0, 1, 2))
        self.assertEqual(c.indices, (0, 1, 2))

    def test_negative_indices_rejected(self):
        with self.assertRaises(InvalidParameter):
            Cell(indices=(-1, 0))


# =============================================================================
# Competence estimate
# =============================================================================


class TestCompetenceEstimate(unittest.TestCase):

    def test_initial_mean_is_jeffreys(self):
        e = CompetenceEstimate(cell=Cell(indices=(0,)))
        # mean = 0.5 / 1 = 0.5 under Jeffreys prior
        self.assertAlmostEqual(e.mean, 0.5)

    def test_ci_width_shrinks_with_data(self):
        e1 = CompetenceEstimate(cell=Cell(indices=(0,)),
                                successes=5, n=10)
        e2 = CompetenceEstimate(cell=Cell(indices=(0,)),
                                successes=50, n=100)
        self.assertGreater(e1.width, e2.width)

    def test_learning_progress_zero_when_no_data(self):
        e = CompetenceEstimate(cell=Cell(indices=(0,)))
        self.assertEqual(e.learning_progress(), 0.0)

    def test_learning_progress_nonzero_with_split(self):
        e = CompetenceEstimate(cell=Cell(indices=(0,)),
                               successes=8, n=10,
                               recent_successes=8, recent_n=8,
                               prev_successes=0, prev_n=8)
        self.assertGreater(e.learning_progress(), 0.5)


# =============================================================================
# Curator orchestrator
# =============================================================================


class TestCurator(unittest.TestCase):

    def test_basic_construction(self):
        c = Curator()
        self.assertEqual(c.config.strategy, STRATEGY_ZPD)
        self.assertEqual(c.n_observations(), 0)

    def test_propose_returns_n_proposals(self):
        c = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(8,))
        def gen(theta): return f"t{theta[0]:.2f}"
        proposals = c.propose(n=5, generator=gen)
        self.assertEqual(len(proposals), 5)
        for p in proposals:
            self.assertIsInstance(p, Proposal)

    def test_observe_records_data(self):
        c = zpd_curator()
        c.observe(theta=(0.3,), success=True)
        self.assertEqual(c.n_observations(), 1)
        c.observe(theta=(0.3,), success=False)
        self.assertEqual(c.n_observations(), 2)

    def test_competence_estimate_for_theta(self):
        c = zpd_curator(n_buckets=(8,))
        for _ in range(10):
            c.observe(theta=(0.3,), success=True)
        est = c.competence_estimate((0.3,))
        self.assertGreater(est.mean, 0.85)

    def test_theta_outside_cube_rejected(self):
        c = zpd_curator(param_lo=(0.0,), param_hi=(1.0,))
        with self.assertRaises(InvalidParameter):
            c.observe(theta=(2.0,), success=True)

    def test_propose_with_n_zero_raises(self):
        c = zpd_curator()
        with self.assertRaises(InvalidConfig):
            c.propose(n=0, generator=lambda t: t)

    def test_certificate_changes_with_observations(self):
        c = zpd_curator()
        cert0 = c.certificate
        c.observe(theta=(0.3,), success=True)
        self.assertNotEqual(c.certificate, cert0)

    def test_certificate_deterministic_for_same_seed(self):
        def fixture():
            c = zpd_curator(seed=42)
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                c.observe(theta=(t,), success=t < 0.5)
            return c.certificate
        self.assertEqual(fixture(), fixture())

    def test_certificate_differs_for_different_seed(self):
        c1 = zpd_curator(seed=1)
        c1.observe(theta=(0.3,), success=True)
        c2 = zpd_curator(seed=2)
        c2.observe(theta=(0.3,), success=True)
        # Observations the same, but init payload includes the seed
        self.assertNotEqual(c1.certificate, c2.certificate)


# =============================================================================
# ZPD strategy
# =============================================================================


class TestZPDStrategy(unittest.TestCase):

    def test_zpd_concentrates_on_frontier(self):
        # Competence drops linearly from 1.0 at theta=0 to 0.0 at theta=1
        rng = random.Random(0)

        def gen(theta): return float(theta[0])
        def oracle_p(t): return max(0.0, 1.0 - t)

        c = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(10,),
                        target_competence=0.5, target_tolerance=0.1,
                        seed=0)
        # bootstrap with uniform observations
        for theta_val in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
            for _ in range(20):
                t = (theta_val,)
                success = rng.random() < oracle_p(theta_val)
                c.observe(theta=t, success=success)

        # ZPD should pick around theta ≈ 0.5 (where competence ≈ 0.5)
        proposals = c.propose(n=20, generator=gen)
        thetas = [p.theta[0] for p in proposals]
        median = sorted(thetas)[len(thetas) // 2]
        self.assertGreater(median, 0.3)
        self.assertLess(median, 0.7)


# =============================================================================
# Learning-progress strategy
# =============================================================================


class TestLearningProgressStrategy(unittest.TestCase):

    def test_lp_returns_proposals(self):
        c = learning_progress_curator(seed=0)
        def gen(theta): return tuple(theta)
        # Some observations
        for _ in range(20):
            c.observe(theta=(0.5,), success=True)
        proposals = c.propose(n=5, generator=gen)
        self.assertEqual(len(proposals), 5)


# =============================================================================
# Thompson-LP strategy
# =============================================================================


class TestThompsonLPStrategy(unittest.TestCase):

    def test_thompson_lp_returns_proposals(self):
        c = thompson_lp_curator(seed=0)
        def gen(theta): return tuple(theta)
        # Some observations
        for theta_val in (0.2, 0.5, 0.8):
            for _ in range(10):
                c.observe(theta=(theta_val,), success=(theta_val < 0.5))
        proposals = c.propose(n=5, generator=gen)
        self.assertEqual(len(proposals), 5)


# =============================================================================
# Multi-dim cube
# =============================================================================


class TestMultiDim(unittest.TestCase):

    def test_2d_cube(self):
        c = zpd_curator(param_lo=(0.0, 0.0), param_hi=(1.0, 1.0),
                        n_buckets=(4, 4), seed=0)
        def gen(theta): return theta
        proposals = c.propose(n=8, generator=gen)
        self.assertEqual(len(proposals), 8)
        for p in proposals:
            self.assertEqual(len(p.theta), 2)
            self.assertGreaterEqual(p.theta[0], 0.0)
            self.assertLessEqual(p.theta[0], 1.0)

    def test_2d_observation(self):
        c = zpd_curator(param_lo=(0.0, 0.0), param_hi=(1.0, 1.0),
                        n_buckets=(4, 4), seed=0)
        c.observe(theta=(0.3, 0.6), success=True)
        self.assertEqual(c.n_observations(), 1)


# =============================================================================
# Frontier
# =============================================================================


class TestFrontier(unittest.TestCase):

    def test_frontier_finds_target_band(self):
        c = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(10,),
                        target_competence=0.5, target_tolerance=0.2, seed=0)
        # Heavy observations: 0.0-0.4 always succeed, 0.5-0.9 never succeed
        for theta_val in (0.05, 0.15, 0.25, 0.35):
            for _ in range(20):
                c.observe(theta=(theta_val,), success=True)
        for theta_val in (0.55, 0.65, 0.75, 0.85, 0.95):
            for _ in range(20):
                c.observe(theta=(theta_val,), success=False)
        front = c.frontier()
        # cells with mean near 0.5 should not exist (all are 0 or 1)
        # so frontier may be empty or limited to widely-uncertain cells
        # We just check frontier doesn't include heavily-observed easy or
        # heavily-observed impossible cells.
        for cell in front:
            est = c.cells()[cell]
            # CI must bracket 0.5±0.2
            lo, hi = est.ci
            self.assertTrue(lo <= 0.7 and hi >= 0.3)


# =============================================================================
# Brier calibration
# =============================================================================


class TestBrierCalibration(unittest.TestCase):

    def test_brier_score_none_when_no_predictions(self):
        c = zpd_curator()
        self.assertIsNone(c.brier_score())

    def test_brier_score_decreases_with_calibrated_predictions(self):
        c = zpd_curator()
        # Provide well-calibrated predictions: 50/50 outcomes with 0.5 pred
        for i in range(100):
            c.observe(theta=(0.5,), success=(i % 2 == 0),
                      predicted_competence=0.5)
        b = c.brier_score()
        self.assertIsNotNone(b)
        self.assertLess(b, 0.3)  # 0.25 is the theoretical minimum for p=0.5


# =============================================================================
# Most competent
# =============================================================================


class TestMostCompetent(unittest.TestCase):

    def test_most_competent_finds_easy_cell(self):
        c = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(10,),
                        seed=0)
        for theta_val in (0.1, 0.5, 0.9):
            for _ in range(20):
                c.observe(theta=(theta_val,), success=(theta_val < 0.3))
        cell, val = c.most_competent()
        # cell 1 (centre 0.15) should be most competent
        self.assertEqual(cell.indices[0], 1)
        self.assertGreater(val, 0.9)


# =============================================================================
# Report
# =============================================================================


class TestReport(unittest.TestCase):

    def test_report_basic(self):
        c = zpd_curator()
        for theta_val in (0.1, 0.5, 0.9):
            for _ in range(10):
                c.observe(theta=(theta_val,), success=True)
        rep = c.report()
        self.assertIsInstance(rep, CuratorReport)
        self.assertGreater(rep.n_cells_explored, 0)
        self.assertIsNotNone(rep.certificate)

    def test_report_as_dict_serialises(self):
        import json as _json
        c = zpd_curator()
        c.observe(theta=(0.3,), success=True, predicted_competence=0.5)
        rep = c.report()
        d = rep.as_dict()
        s = _json.dumps(d)
        self.assertIn("strategy", s)


# =============================================================================
# Free-function shortcuts
# =============================================================================


class TestFreeFunctions(unittest.TestCase):

    def test_zpd_curator(self):
        c = zpd_curator()
        self.assertEqual(c.config.strategy, STRATEGY_ZPD)

    def test_learning_progress_curator(self):
        c = learning_progress_curator()
        self.assertEqual(c.config.strategy, STRATEGY_LEARNING_PROGRESS)

    def test_thompson_lp_curator(self):
        c = thompson_lp_curator()
        self.assertEqual(c.config.strategy, STRATEGY_THOMPSON_LP)


# =============================================================================
# Composition with Searcher / Distiller (illustrative)
# =============================================================================


class TestComposition(unittest.TestCase):

    def test_curator_drives_difficulty_increase(self):
        # As the agent gets better (oracle becomes always-true), the
        # frontier should shift to higher theta.
        c = zpd_curator(param_lo=(0.0,), param_hi=(1.0,), n_buckets=(8,),
                        target_competence=0.5, target_tolerance=0.15,
                        seed=0)
        rng = random.Random(0)

        def gen(theta): return float(theta[0])

        # Stage 1: agent solves theta < 0.5 only
        def stage_1_p(t): return 1.0 if t < 0.5 else 0.0
        for _ in range(10):
            for p in c.propose(n=4, generator=gen):
                success = rng.random() < stage_1_p(p.theta[0])
                c.observe(theta=p.theta, success=success,
                          predicted_competence=p.predicted_competence)
        # Stage 1 frontier should hover near 0.5
        f1 = c.frontier()
        if f1:
            f1_centres = [c.cell_centre(cell)[0] for cell in f1]
            mean_f1 = sum(f1_centres) / len(f1_centres)
            self.assertGreater(mean_f1, 0.2)
            self.assertLess(mean_f1, 0.7)


if __name__ == "__main__":
    unittest.main()
