"""Tests for agi.pareto."""
from __future__ import annotations

import math
import random
import unittest

from agi.pareto import (
    KNOWN_SCALARISATIONS,
    KNOWN_SENSES,
    PARETO_CERTIFIED,
    PARETO_FRONTIER,
    PARETO_OBSERVED,
    PARETO_REPORTED,
    PARETO_STARTED,
    SCALAR_ASF,
    SCALAR_AUG_TCHEBYCHEFF,
    SCALAR_PBI,
    SCALAR_TCHEBYCHEFF,
    SCALAR_WEIGHTED_SUM,
    SENSE_MAX,
    SENSE_MIN,
    Candidate,
    EHVIReport,
    FrontierReport,
    HypervolumeReport,
    InsufficientData,
    InvalidCandidate,
    InvalidConfig,
    InvalidQuery,
    MetricsReport,
    Pareto,
    ParetoConfig,
    ParetoError,
    ProgressCertificate,
    ScalarisationReport,
    UnknownCandidate,
    achievement_scalarising_function,
    augmented_tchebycheff,
    crowding_distance,
    das_dennis_weights,
    dominates,
    ehvi_2d,
    ehvi_monte_carlo,
    empirical_bernstein_half_width,
    generational_distance,
    hoeffding_half_width,
    hrms_half_width,
    hypervolume,
    hypervolume_2d,
    hypervolume_3d,
    inverted_generational_distance,
    ledger_root,
    maximum_spread,
    non_dominated_sort,
    pbi_scalarisation,
    spacing,
    tchebycheff,
    uniform_simplex_weights,
    weighted_sum,
)


# ---------------------------------------------------------------------------
# Dominance + non-dominated sort
# ---------------------------------------------------------------------------


class DominanceTests(unittest.TestCase):
    def test_strict_dominance(self) -> None:
        self.assertTrue(dominates((0.1, 0.1), (0.2, 0.2)))
        self.assertFalse(dominates((0.1, 0.3), (0.2, 0.2)))
        self.assertFalse(dominates((0.2, 0.2), (0.1, 0.1)))

    def test_weak_dominance_is_not_strict(self) -> None:
        # equal vectors do not dominate each other
        self.assertFalse(dominates((0.1, 0.1), (0.1, 0.1)))

    def test_partial_equal_dominance(self) -> None:
        # equal on one obj, strictly better on the other → dominates
        self.assertTrue(dominates((0.1, 0.2), (0.1, 0.3)))
        self.assertTrue(dominates((0.2, 0.1), (0.3, 0.1)))

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ParetoError):
            dominates((0.1,), (0.1, 0.2))


class NonDominatedSortTests(unittest.TestCase):
    def test_single_front(self) -> None:
        pts = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]
        fronts = non_dominated_sort(pts)
        self.assertEqual(len(fronts), 1)
        self.assertEqual(set(fronts[0]), {0, 1, 2})

    def test_layered_fronts(self) -> None:
        pts = [
            (0.0, 1.0),  # rank 0
            (0.5, 0.5),  # rank 0
            (1.0, 0.0),  # rank 0
            (0.6, 0.6),  # rank 1 (dominated by 0.5,0.5)
            (0.7, 0.7),  # rank 2 (dominated by 0.6,0.6)
            (0.8, 0.8),  # rank 3
        ]
        fronts = non_dominated_sort(pts)
        self.assertEqual(set(fronts[0]), {0, 1, 2})
        self.assertEqual(set(fronts[1]), {3})
        self.assertEqual(set(fronts[2]), {4})
        self.assertEqual(set(fronts[3]), {5})

    def test_empty_input(self) -> None:
        self.assertEqual(non_dominated_sort([]), [])

    def test_feasibility_first(self) -> None:
        # second point is infeasible — feasibility-first dominance
        # places the feasible point above regardless of objective values.
        pts = [(0.5, 0.5), (0.1, 0.1)]
        viols = [0.0, 1.0]  # second is infeasible
        fronts = non_dominated_sort(pts, viols)
        self.assertIn(0, fronts[0])
        self.assertNotIn(1, fronts[0])

    def test_crowding_boundary_inf(self) -> None:
        pts = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        crowd = crowding_distance(pts)
        # boundary points (sorted-min and sorted-max in any obj) → inf
        self.assertTrue(math.isinf(crowd[0]))
        self.assertTrue(math.isinf(crowd[2]))
        # The middle point is interior in both objectives
        self.assertFalse(math.isinf(crowd[1]))
        self.assertGreater(crowd[1], 0.0)

    def test_crowding_single_point(self) -> None:
        crowd = crowding_distance([(0.0, 0.0)])
        self.assertEqual(len(crowd), 1)


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


class Hypervolume2DTests(unittest.TestCase):
    def test_three_point_sweep(self) -> None:
        hv = hypervolume_2d(
            [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)], (1.0, 1.0)
        )
        self.assertAlmostEqual(hv, 0.37, places=10)

    def test_single_point(self) -> None:
        hv = hypervolume_2d([(0.5, 0.5)], (1.0, 1.0))
        self.assertAlmostEqual(hv, 0.25, places=10)

    def test_empty_front(self) -> None:
        self.assertEqual(hypervolume_2d([], (1.0, 1.0)), 0.0)

    def test_dominated_point_does_not_increase_hv(self) -> None:
        hv_a = hypervolume_2d([(0.5, 0.5)], (1.0, 1.0))
        hv_b = hypervolume_2d([(0.5, 0.5), (0.7, 0.7)], (1.0, 1.0))
        self.assertAlmostEqual(hv_a, hv_b, places=10)

    def test_points_outside_reference_skipped(self) -> None:
        hv = hypervolume_2d([(2.0, 2.0), (0.5, 0.5)], (1.0, 1.0))
        self.assertAlmostEqual(hv, 0.25, places=10)


class Hypervolume3DTests(unittest.TestCase):
    def test_single_3d_point(self) -> None:
        hv = hypervolume_3d([(0.5, 0.5, 0.5)], (1.0, 1.0, 1.0))
        self.assertAlmostEqual(hv, 0.125, places=10)

    def test_dispatch_via_hypervolume(self) -> None:
        hv, algo = hypervolume([(0.5, 0.5, 0.5)], (1.0, 1.0, 1.0))
        self.assertAlmostEqual(hv, 0.125, places=10)
        self.assertEqual(algo, "exact_3d_slicing")

    def test_three_3d_points_agrees_with_wfg(self) -> None:
        # Force WFG path by faking 4D embedding; sanity check.
        pts3 = [(0.2, 0.8, 0.5), (0.5, 0.5, 0.5), (0.8, 0.2, 0.5)]
        hv3 = hypervolume_3d(pts3, (1.0, 1.0, 1.0))
        # All three points share z=0.5 so contribute a single slab of
        # height 0.5 with 2D HV = 0.37 → 0.185.
        self.assertAlmostEqual(hv3, 0.185, places=10)


class HypervolumeDispatchTests(unittest.TestCase):
    def test_2d_dispatch(self) -> None:
        hv, algo = hypervolume([(0.5, 0.5)], (1.0, 1.0))
        self.assertAlmostEqual(hv, 0.25, places=10)
        self.assertEqual(algo, "exact_2d_sweep")

    def test_1d(self) -> None:
        hv, algo = hypervolume([(0.5,), (0.3,)], (1.0,))
        self.assertAlmostEqual(hv, 0.7, places=10)
        self.assertEqual(algo, "1d_min")

    def test_4d_wfg(self) -> None:
        pts = [(0.2, 0.8, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)]
        hv, algo = hypervolume(pts, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(algo, "wfg_4d")
        self.assertGreater(hv, 0.0)

    def test_inclusion_exclusion_for_6d(self) -> None:
        pts = [tuple(0.5 for _ in range(6))]
        hv, algo = hypervolume(pts, tuple(1.0 for _ in range(6)))
        self.assertEqual(algo, "inclusion_exclusion")
        self.assertAlmostEqual(hv, 0.5 ** 6, places=10)


# ---------------------------------------------------------------------------
# EHVI
# ---------------------------------------------------------------------------


class EHVI2DTests(unittest.TestCase):
    def test_ehvi_non_negative(self) -> None:
        pts = [(0.5, 0.5)]
        ehvi = ehvi_2d(pts, (1.0, 1.0), (0.3, 0.3), (0.1, 0.1))
        self.assertGreaterEqual(ehvi, 0.0)

    def test_zero_sigma_collapses_to_improvement(self) -> None:
        # When sigma → 0, EHVI → max(0, HV(new ∪ front) − HV(front))
        pts = [(0.5, 0.5)]
        m = (0.3, 0.3)
        improvement = (
            hypervolume_2d([(0.5, 0.5), m], (1.0, 1.0))
            - hypervolume_2d(pts, (1.0, 1.0))
        )
        ehvi = ehvi_2d(pts, (1.0, 1.0), m, (1e-9, 1e-9))
        self.assertAlmostEqual(ehvi, improvement, places=4)

    def test_dominated_candidate_has_lower_ehvi(self) -> None:
        pts = [(0.5, 0.5)]
        # m1 dominates the front, m2 is dominated by it.
        e1 = ehvi_2d(pts, (1.0, 1.0), (0.3, 0.3), (0.05, 0.05))
        e2 = ehvi_2d(pts, (1.0, 1.0), (0.7, 0.7), (0.05, 0.05))
        self.assertGreater(e1, e2)

    def test_shape_validation(self) -> None:
        with self.assertRaises(ParetoError):
            ehvi_2d([(0.5,)], (1.0, 1.0), (0.3, 0.3), (0.1, 0.1))


class EHVIMonteCarloTests(unittest.TestCase):
    def test_2d_agrees_with_closed_form(self) -> None:
        pts = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]
        ref = (1.0, 1.0)
        closed = ehvi_2d(pts, ref, (0.4, 0.4), (0.05, 0.05))
        mc, se = ehvi_monte_carlo(pts, ref, (0.4, 0.4), (0.05, 0.05),
                                  n_samples=2048, seed=0)
        self.assertAlmostEqual(closed, mc, delta=4 * (se + 1e-3))


# ---------------------------------------------------------------------------
# Scalarisations
# ---------------------------------------------------------------------------


class ScalarisationsTests(unittest.TestCase):
    def test_weighted_sum_recovers_argmin(self) -> None:
        self.assertAlmostEqual(weighted_sum((0.5, 0.5), (1.0, 1.0)), 1.0)
        self.assertAlmostEqual(weighted_sum((0.0, 1.0), (1.0, 0.0)), 0.0)

    def test_tchebycheff_max_of_weighted(self) -> None:
        self.assertAlmostEqual(
            tchebycheff((0.2, 0.8), (1.0, 1.0), (0.0, 0.0)), 0.8
        )

    def test_augmented_breaks_weak_pareto_degeneracy(self) -> None:
        # Two points with the same Tchebycheff value but different
        # weighted-sum values — augmented Tchebycheff prefers the
        # weighted-sum-smaller one.
        v1 = augmented_tchebycheff((0.5, 0.0), (1.0, 1.0), (0.0, 0.0))
        v2 = augmented_tchebycheff((0.5, 0.5), (1.0, 1.0), (0.0, 0.0))
        self.assertLess(v1, v2)

    def test_asf_min_at_reference(self) -> None:
        # ASF achieves its min at the reference point itself.
        v_at = achievement_scalarising_function((0.5, 0.5), (0.5, 0.5))
        v_away = achievement_scalarising_function((0.7, 0.7), (0.5, 0.5))
        self.assertLess(v_at, v_away)

    def test_pbi_d2_zero_on_ray(self) -> None:
        # A point exactly on the weight ray has d2 = 0; the PBI value
        # equals d1.
        v = pbi_scalarisation((0.5, 0.5), (1.0, 1.0), (0.0, 0.0), theta=5.0)
        # d1 = (0.5+0.5)/sqrt(2) = 1/√2  ≈ 0.7071
        self.assertAlmostEqual(v, math.sqrt(0.5), places=10)


# ---------------------------------------------------------------------------
# Weight grids
# ---------------------------------------------------------------------------


class WeightGridTests(unittest.TestCase):
    def test_das_dennis_count(self) -> None:
        # C(p+M-1, M-1)
        # for M=3, p=4 → C(6, 2) = 15
        w = das_dennis_weights(3, 4)
        self.assertEqual(len(w), 15)
        for v in w:
            self.assertAlmostEqual(sum(v), 1.0, places=10)

    def test_uniform_simplex_sums_to_one(self) -> None:
        ws = uniform_simplex_weights(50, 4, seed=0)
        for w in ws:
            self.assertAlmostEqual(sum(w), 1.0, places=10)


# ---------------------------------------------------------------------------
# Convergence + diversity metrics
# ---------------------------------------------------------------------------


class MetricsTests(unittest.TestCase):
    def test_igd_zero_on_exact_match(self) -> None:
        ref = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        self.assertAlmostEqual(
            inverted_generational_distance(ref, ref), 0.0, places=10
        )

    def test_gd_higher_for_worse_front(self) -> None:
        ref = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        good = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        bad = [(0.6, 1.0), (0.5, 1.2), (1.2, 0.6)]
        self.assertAlmostEqual(generational_distance(good, ref), 0.0, places=10)
        self.assertGreater(generational_distance(bad, ref), 0.0)

    def test_spacing_zero_when_uniform(self) -> None:
        front = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        # neighbour distances are equal across the spaced front
        self.assertAlmostEqual(spacing(front), 0.0, places=10)

    def test_max_spread(self) -> None:
        front = [(0.0, 1.0), (1.0, 0.0)]
        # diagonal length √2
        self.assertAlmostEqual(maximum_spread(front), math.sqrt(2.0), places=10)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


class StatisticalHelpersTests(unittest.TestCase):
    def test_hrms_shrinks_with_n(self) -> None:
        self.assertGreater(hrms_half_width(10), hrms_half_width(100))

    def test_empirical_bernstein_scales_with_variance(self) -> None:
        self.assertGreater(
            empirical_bernstein_half_width(100, 0.25),
            empirical_bernstein_half_width(100, 0.0),
        )

    def test_hoeffding(self) -> None:
        self.assertGreater(hoeffding_half_width(10), hoeffding_half_width(1000))


# ---------------------------------------------------------------------------
# Pareto class
# ---------------------------------------------------------------------------


class ParetoConfigTests(unittest.TestCase):
    def test_invalid_sense_raises(self) -> None:
        with self.assertRaises(InvalidConfig):
            ParetoConfig(senses=("bogus",))

    def test_reference_shape_mismatch_raises(self) -> None:
        with self.assertRaises(InvalidConfig):
            ParetoConfig(senses=("min", "min"), reference=(1.0,))

    def test_confidence_bound(self) -> None:
        with self.assertRaises(InvalidConfig):
            ParetoConfig(senses=("min",), confidence=0.4)

    def test_hmac_key_type(self) -> None:
        with self.assertRaises(InvalidConfig):
            ParetoConfig(senses=("min",), hmac_key="not bytes")  # type: ignore[arg-type]


class ParetoObservationTests(unittest.TestCase):
    def _make(self) -> Pareto:
        return Pareto(ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0)))

    def test_observe_and_get(self) -> None:
        pf = self._make()
        pf.observe("c1", (0.5, 0.5))
        c = pf.get("c1")
        self.assertEqual(c.cost, (0.5, 0.5))
        self.assertEqual(len(pf), 1)

    def test_invalid_cost_length(self) -> None:
        pf = self._make()
        with self.assertRaises(InvalidCandidate):
            pf.observe("c1", (0.5,))

    def test_invalid_cost_nan(self) -> None:
        pf = self._make()
        with self.assertRaises(InvalidCandidate):
            pf.observe("c1", (0.5, float("nan")))

    def test_remove_unknown_raises(self) -> None:
        pf = self._make()
        with self.assertRaises(UnknownCandidate):
            pf.remove("c1")

    def test_get_unknown_raises(self) -> None:
        pf = self._make()
        with self.assertRaises(UnknownCandidate):
            pf.get("c1")

    def test_clear(self) -> None:
        pf = self._make()
        pf.observe("c1", (0.5, 0.5))
        pf.clear()
        self.assertEqual(len(pf), 0)

    def test_event_publishing(self) -> None:
        events: list[tuple[str, dict]] = []
        pf = Pareto(
            ParetoConfig(senses=("min", "min")),
            publisher=lambda k, p: events.append((k, p)),
        )
        pf.observe("c1", (0.5, 0.5))
        kinds = [k for k, _ in events]
        self.assertIn(PARETO_STARTED, kinds)
        self.assertIn(PARETO_OBSERVED, kinds)


class ParetoFrontierTests(unittest.TestCase):
    def _populate(self) -> Pareto:
        pf = Pareto(ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0)))
        pf.observe("a", (0.2, 0.8))
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (0.8, 0.2))
        pf.observe("d", (0.6, 0.7))  # dominated by b
        return pf

    def test_front_excludes_dominated(self) -> None:
        pf = self._populate()
        fr = pf.frontier()
        self.assertNotIn("d", fr.candidates)
        self.assertEqual(set(fr.candidates), {"a", "b", "c"})

    def test_costs_returned_in_user_coordinates(self) -> None:
        pf = self._populate()
        fr = pf.frontier()
        cost_for_b = next(
            c for cid, c in zip(fr.candidates, fr.costs) if cid == "b"
        )
        self.assertEqual(cost_for_b, (0.5, 0.5))

    def test_rank_layer_indexing(self) -> None:
        pf = self._populate()
        fr1 = pf.frontier(rank=1)
        self.assertIn("d", fr1.candidates)

    def test_invalid_rank_raises(self) -> None:
        pf = self._populate()
        with self.assertRaises(InvalidQuery):
            pf.frontier(rank=99)

    def test_max_sense_is_handled(self) -> None:
        # Maximisation: bigger is better; the user supplies positive
        # values they want to maximise.
        pf = Pareto(ParetoConfig(senses=("max", "max"), reference=(0.0, 0.0)))
        pf.observe("a", (0.2, 0.8))
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (0.8, 0.2))
        pf.observe("d", (0.4, 0.4))  # dominated by b
        fr = pf.frontier()
        self.assertEqual(set(fr.candidates), {"a", "b", "c"})

    def test_mixed_sense(self) -> None:
        pf = Pareto(ParetoConfig(senses=("min", "max")))
        pf.observe("a", (0.1, 0.9))  # great on both
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (0.9, 0.1))
        pf.observe("d", (0.2, 0.5))  # dominated by a? a=(0.1, 0.9): 0.1<0.2 AND 0.9>0.5 → a dominates d
        fr = pf.frontier()
        self.assertNotIn("d", fr.candidates)

    def test_insufficient_data_raises(self) -> None:
        pf = Pareto(ParetoConfig(senses=("min",)))
        with self.assertRaises(InsufficientData):
            pf.frontier()


class ParetoHypervolumeTests(unittest.TestCase):
    def test_hv_matches_manual_2d(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0))
        )
        pf.observe("a", (0.2, 0.8))
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (0.8, 0.2))
        rep = pf.hypervolume()
        self.assertAlmostEqual(rep.hypervolume, 0.37, places=10)

    def test_hv_strictly_increases_with_non_dominated_addition(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0))
        )
        pf.observe("a", (0.5, 0.5))
        rep1 = pf.hypervolume()
        pf.observe("b", (0.2, 0.7))
        rep2 = pf.hypervolume()
        self.assertGreater(rep2.hypervolume, rep1.hypervolume)

    def test_hv_invariant_under_dominated_addition(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0))
        )
        pf.observe("a", (0.5, 0.5))
        rep1 = pf.hypervolume()
        pf.observe("b", (0.7, 0.7))
        rep2 = pf.hypervolume()
        self.assertAlmostEqual(rep1.hypervolume, rep2.hypervolume, places=10)

    def test_default_reference_inferred(self) -> None:
        # Reference omitted in config and call — should infer.
        pf = Pareto(ParetoConfig(senses=("min", "min")))
        pf.observe("a", (0.2, 0.8))
        pf.observe("b", (0.5, 0.5))
        rep = pf.hypervolume()
        self.assertGreater(rep.hypervolume, 0.0)

    def test_constraint_violation_excludes_from_hv(self) -> None:
        pf = Pareto(
            ParetoConfig(
                senses=("min", "min"),
                reference=(1.0, 1.0),
                constraint_dim=1,
            )
        )
        pf.observe("a", (0.5, 0.5), constraints=(0.0,))
        pf.observe("b", (0.1, 0.1), constraints=(1.0,))  # infeasible
        rep = pf.hypervolume()
        # Only feasible 'a' contributes; HV = (1-0.5)*(1-0.5)=0.25
        self.assertAlmostEqual(rep.hypervolume, 0.25, places=10)

    def test_hv_3d(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min", "min"), reference=(1.0, 1.0, 1.0))
        )
        pf.observe("a", (0.5, 0.5, 0.5))
        rep = pf.hypervolume()
        self.assertAlmostEqual(rep.hypervolume, 0.125, places=10)


class ParetoEHVITests(unittest.TestCase):
    def test_ehvi_non_negative(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0))
        )
        pf.observe("a", (0.5, 0.5))
        rep = pf.expected_hypervolume_improvement((0.3, 0.3), (0.1, 0.1))
        self.assertGreaterEqual(rep.ehvi, 0.0)
        self.assertEqual(rep.algorithm, "closed_form_2d")

    def test_ehvi_shape_mismatch(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0))
        )
        with self.assertRaises(InvalidQuery):
            pf.expected_hypervolume_improvement((0.3,), (0.1, 0.1))

    def test_ehvi_3d_mc(self) -> None:
        pf = Pareto(
            ParetoConfig(senses=("min", "min", "min"), reference=(1.0, 1.0, 1.0))
        )
        pf.observe("a", (0.5, 0.5, 0.5))
        rep = pf.expected_hypervolume_improvement(
            (0.3, 0.3, 0.3), (0.05, 0.05, 0.05), n_samples=512
        )
        self.assertGreater(rep.ehvi, 0.0)
        self.assertEqual(rep.algorithm, "monte_carlo_3d")


class ParetoScalarisationTests(unittest.TestCase):
    def _pf(self) -> Pareto:
        pf = Pareto(ParetoConfig(senses=("min", "min")))
        pf.observe("a", (0.0, 1.0))
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (1.0, 0.0))
        return pf

    def test_weighted_sum_recovers_endpoint(self) -> None:
        pf = self._pf()
        # weight (1, 0) prefers small first objective → "a"
        sc = pf.scalarise(SCALAR_WEIGHTED_SUM, (1.0, 0.0))
        self.assertEqual(sc.argmin_candidate, "a")
        sc = pf.scalarise(SCALAR_WEIGHTED_SUM, (0.0, 1.0))
        self.assertEqual(sc.argmin_candidate, "c")

    def test_tchebycheff_recovers_middle(self) -> None:
        pf = self._pf()
        sc = pf.scalarise(SCALAR_TCHEBYCHEFF, (0.5, 0.5))
        # Tchebycheff with equal weights and ideal-shifted prefers
        # balanced 'b'
        self.assertEqual(sc.argmin_candidate, "b")

    def test_unknown_scalarisation_raises(self) -> None:
        pf = self._pf()
        with self.assertRaises(InvalidQuery):
            pf.scalarise("foo", (0.5, 0.5))

    def test_sweep_returns_one_per_grid_vertex(self) -> None:
        pf = self._pf()
        out = pf.sweep(method=SCALAR_TCHEBYCHEFF, p=3)
        # C(p+M-1, M-1) = C(4,1) = 4
        self.assertEqual(len(out), 4)


class ParetoMetricsTests(unittest.TestCase):
    def test_metrics_without_reference_set(self) -> None:
        pf = Pareto(ParetoConfig(senses=("min", "min")))
        pf.observe("a", (0.0, 1.0))
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (1.0, 0.0))
        rep = pf.metrics()
        self.assertIsNone(rep.igd)
        self.assertIsNone(rep.gd)
        self.assertAlmostEqual(rep.spacing, 0.0, places=10)
        self.assertAlmostEqual(rep.max_spread, math.sqrt(2.0), places=10)

    def test_metrics_with_reference_set(self) -> None:
        pf = Pareto(ParetoConfig(senses=("min", "min")))
        pf.observe("a", (0.0, 1.0))
        pf.observe("b", (0.5, 0.5))
        pf.observe("c", (1.0, 0.0))
        rep = pf.metrics(reference_set=[(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
        self.assertAlmostEqual(rep.igd, 0.0, places=10)
        self.assertAlmostEqual(rep.gd, 0.0, places=10)


class ParetoProgressTests(unittest.TestCase):
    def test_progress_certificate_converges_on_flat_history(self) -> None:
        pf = Pareto(ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0)))
        pf.observe("a", (0.5, 0.5))
        for _ in range(8):
            pf.hypervolume()
        cert = pf.certify_progress(epsilon=1e-3)
        self.assertTrue(cert.converged)
        self.assertGreaterEqual(cert.delta_hv, 0.0)


class ParetoSnapshotTests(unittest.TestCase):
    def test_roundtrip_chain_head(self) -> None:
        pf1 = Pareto(ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0)))
        pf1.observe("a", (0.5, 0.5))
        pf1.observe("b", (0.3, 0.7))
        snap = pf1.snapshot()
        pf2 = Pareto(ParetoConfig(senses=("min", "min"), reference=(1.0, 1.0)))
        pf2.restore(snap)
        self.assertEqual(pf1.chain_head, pf2.chain_head)
        self.assertEqual(pf1.frontier().candidates, pf2.frontier().candidates)


class ParetoChainTests(unittest.TestCase):
    def test_chain_advances_per_observation(self) -> None:
        pf = Pareto(ParetoConfig(senses=("min", "min")))
        h0 = pf.chain_head
        pf.observe("a", (0.5, 0.5))
        h1 = pf.chain_head
        pf.observe("b", (0.3, 0.7))
        h2 = pf.chain_head
        self.assertNotEqual(h0, h1)
        self.assertNotEqual(h1, h2)

    def test_chain_deterministic_under_identical_history(self) -> None:
        cfg = ParetoConfig(senses=("min", "min"))
        pf1 = Pareto(cfg)
        pf2 = Pareto(cfg)
        for cid, c in [("a", (0.5, 0.5)), ("b", (0.3, 0.7)), ("c", (0.7, 0.3))]:
            pf1.observe(cid, c)
            pf2.observe(cid, c)
        self.assertEqual(pf1.chain_head, pf2.chain_head)

    def test_hmac_key_changes_chain(self) -> None:
        cfg_a = ParetoConfig(senses=("min", "min"), hmac_key=b"k1")
        cfg_b = ParetoConfig(senses=("min", "min"), hmac_key=b"k2")
        pf1 = Pareto(cfg_a)
        pf2 = Pareto(cfg_b)
        pf1.observe("a", (0.5, 0.5))
        pf2.observe("a", (0.5, 0.5))
        self.assertNotEqual(pf1.chain_head, pf2.chain_head)


class ParetoStressTests(unittest.TestCase):
    def test_random_2d_population(self) -> None:
        rng = random.Random(0)
        pf = Pareto(ParetoConfig(senses=("min", "min"), reference=(2.0, 2.0)))
        for i in range(50):
            pf.observe(f"c{i}", (rng.random(), rng.random()))
        fr = pf.frontier()
        # Every point on the front strictly dominates none of its
        # peers.
        for i, ci in enumerate(fr.candidates):
            for j, cj in enumerate(fr.candidates):
                if i == j:
                    continue
                self.assertFalse(dominates(fr.costs[i], fr.costs[j]))

    def test_dominated_layers_partition_population(self) -> None:
        rng = random.Random(1)
        pf = Pareto(ParetoConfig(senses=("min", "min")))
        for i in range(30):
            pf.observe(f"c{i}", (rng.random(), rng.random()))
        layers = pf.rank_layers()
        seen: set[str] = set()
        for layer in layers:
            for cid in layer:
                self.assertNotIn(cid, seen)
                seen.add(cid)
        self.assertEqual(len(seen), 30)


if __name__ == "__main__":
    unittest.main()
