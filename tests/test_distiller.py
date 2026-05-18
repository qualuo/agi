"""Tests for agi.distiller — amortized policy/value distillation."""
from __future__ import annotations

import math
import random
import unittest

from agi.distiller import (
    Demonstration,
    Distiller,
    DistillerConfig,
    DistillerReport,
    EnsembleModel,
    InvalidConfig,
    InvalidDemonstration,
    KNNModel,
    KNOWN_MODELS,
    LinearModel,
    LocallyWeightedModel,
    MODEL_ENSEMBLE,
    MODEL_KNN,
    MODEL_LINEAR,
    MODEL_LOCALLY_WEIGHTED,
    MODEL_UCB_TABLE,
    NotFitted,
    ReservoirBuffer,
    UCBTableModel,
    UnknownModel,
    ensemble_distiller,
    expert_iteration_step,
    knn_distiller,
    linear_distiller,
    locally_weighted_distiller,
    ucb_table_distiller,
)


# =============================================================================
# Configuration
# =============================================================================


class TestDistillerConfig(unittest.TestCase):

    def test_known_models(self):
        for m in ("knn", "linear", "locally_weighted", "ucb_table", "ensemble"):
            self.assertIn(m, KNOWN_MODELS)

    def test_default_config_is_linear(self):
        cfg = DistillerConfig()
        self.assertEqual(cfg.model, MODEL_LINEAR)
        self.assertEqual(cfg.n_features, 4096)

    def test_invalid_model(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(model="totally-not-a-model")

    def test_invalid_n_features(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(n_features=4)

    def test_invalid_knn_k(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(knn_k=0)

    def test_invalid_bandwidth(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(local_bandwidth=0)
        with self.assertRaises(InvalidConfig):
            DistillerConfig(local_bandwidth=-1)

    def test_invalid_lrs(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(lr_policy=0)
        with self.assertRaises(InvalidConfig):
            DistillerConfig(lr_value=-1)

    def test_invalid_huber(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(value_huber_delta=-0.1)

    def test_invalid_buffer_capacity(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(buffer_capacity=0)

    def test_invalid_holdout(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(eval_holdout_fraction=-0.1)
        with self.assertRaises(InvalidConfig):
            DistillerConfig(eval_holdout_fraction=1.0)
        with self.assertRaises(InvalidConfig):
            DistillerConfig(eval_holdout_fraction=1.5)

    def test_invalid_min_improvement(self):
        with self.assertRaises(InvalidConfig):
            DistillerConfig(min_improvement=-0.1)


# =============================================================================
# Demonstration
# =============================================================================


class TestDemonstration(unittest.TestCase):

    def test_basic_construction(self):
        d = Demonstration(state="s", action_distribution={"a": 1, "b": 2},
                          value=0.5)
        self.assertEqual(d.state, "s")
        self.assertEqual(d.value, 0.5)

    def test_empty_distribution_rejected(self):
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution={}, value=0.0)

    def test_negative_weight_in_distribution_rejected(self):
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution={"a": -1.0},
                          value=0.0)

    def test_nonfinite_value_rejected(self):
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution={"a": 1},
                          value=float("nan"))
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution={"a": 1},
                          value=float("inf"))

    def test_negative_weight_rejected(self):
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution={"a": 1},
                          value=0.0, weight=-0.1)

    def test_nonnumeric_weight_rejected(self):
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution={"a": "high"},
                          value=0.0)

    def test_mapping_required(self):
        with self.assertRaises(InvalidDemonstration):
            Demonstration(state="s", action_distribution=[("a", 1)],
                          value=0.0)


# =============================================================================
# Reservoir buffer
# =============================================================================


class TestReservoirBuffer(unittest.TestCase):

    def test_basic_fill(self):
        r = ReservoirBuffer(capacity=3, seed=0)
        for i in range(3):
            r.add(Demonstration(state=i, action_distribution={"a": 1}, value=0.0))
        self.assertEqual(len(r), 3)

    def test_overflow_keeps_capacity(self):
        r = ReservoirBuffer(capacity=3, seed=0)
        for i in range(10):
            r.add(Demonstration(state=i, action_distribution={"a": 1}, value=0.0))
        self.assertEqual(len(r), 3)
        self.assertEqual(r.total_seen, 10)

    def test_invalid_capacity(self):
        with self.assertRaises(InvalidConfig):
            ReservoirBuffer(capacity=0)

    def test_uniform_distribution_over_long_run(self):
        # Vitter R: each item should be present with probability N/T
        # Roughly check by counting visits over many trials.
        capacity = 5
        stream_len = 50
        trials = 1000
        counts = [0] * stream_len
        for trial in range(trials):
            r = ReservoirBuffer(capacity=capacity, seed=trial)
            for i in range(stream_len):
                r.add(Demonstration(state=i, action_distribution={"a": 1},
                                    value=0.0))
            for d in r.items():
                counts[d.state] += 1
        # expected count per item ≈ trials * N/T = 100
        expected = trials * capacity / stream_len
        # Loose bound: within 40% of expectation (Bernoulli noise)
        for c in counts:
            self.assertGreater(c, expected * 0.6)
            self.assertLess(c, expected * 1.4)


# =============================================================================
# Individual models
# =============================================================================


class TestKNNModel(unittest.TestCase):

    def test_not_fitted_raises(self):
        m = KNNModel(k=1)
        with self.assertRaises(NotFitted):
            m.policy("x", ["a"])

    def test_basic_policy_value(self):
        m = KNNModel(k=1, n_features=64, seed=0)
        from agi.distiller import _default_featurizer
        demos = [
            Demonstration(state="A", action_distribution={"go": 9, "stop": 1},
                          value=2.0),
            Demonstration(state="B", action_distribution={"go": 1, "stop": 9},
                          value=-2.0),
        ]
        m.fit(demos, _default_featurizer)
        p = m.policy("A", ["go", "stop"])
        self.assertGreater(p["go"], p["stop"])
        self.assertAlmostEqual(sum(p.values()), 1.0, places=4)

    def test_empty_demos_returns_uniform(self):
        m = KNNModel(k=1, n_features=64, seed=0)
        from agi.distiller import _default_featurizer
        m.fit([], _default_featurizer)
        p = m.policy("X", ["a", "b"])
        self.assertAlmostEqual(p["a"], 0.5, places=4)


class TestLinearModel(unittest.TestCase):

    def test_basic_softmax(self):
        m = LinearModel(n_features=64, seed=0, epochs=50)
        from agi.distiller import _default_featurizer
        demos = [
            Demonstration(state="A", action_distribution={"go": 19, "stop": 1},
                          value=2.0),
            Demonstration(state="B", action_distribution={"go": 1, "stop": 19},
                          value=-2.0),
        ]
        m.fit(demos, _default_featurizer)
        p = m.policy("A", ["go", "stop"])
        self.assertGreater(p["go"], 0.7)
        p = m.policy("B", ["go", "stop"])
        self.assertGreater(p["stop"], 0.7)

    def test_value_head_learns(self):
        m = LinearModel(n_features=64, seed=0, epochs=200)
        from agi.distiller import _default_featurizer
        demos = [
            Demonstration(state="A", action_distribution={"go": 1}, value=3.0),
            Demonstration(state="B", action_distribution={"go": 1}, value=-3.0),
        ]
        m.fit(demos, _default_featurizer)
        self.assertGreater(m.value("A"), m.value("B"))


class TestLocallyWeightedModel(unittest.TestCase):

    def test_lwr_basic(self):
        m = LocallyWeightedModel(n_features=64, bandwidth=2.0, seed=0)
        from agi.distiller import _default_featurizer
        demos = [
            Demonstration(state="A", action_distribution={"go": 1}, value=1.0),
            Demonstration(state="B", action_distribution={"stop": 1}, value=-1.0),
        ]
        m.fit(demos, _default_featurizer)
        p = m.policy("A", ["go", "stop"])
        # weight ≈ 1 on A → go dominates
        self.assertGreater(p["go"], p["stop"])


class TestUCBTableModel(unittest.TestCase):

    def test_table_exact_recall(self):
        m = UCBTableModel()
        from agi.distiller import _default_featurizer
        demos = [
            Demonstration(state="A", action_distribution={"go": 9, "stop": 1},
                          value=2.0),
            Demonstration(state="A", action_distribution={"go": 1, "stop": 0},
                          value=2.5),
        ]
        m.fit(demos, _default_featurizer)
        p = m.policy("A", ["go", "stop"])
        # Aggregation: go received 9+1=10 / total=11, stop=1/11
        self.assertGreater(p["go"], 0.8)

    def test_table_unknown_state_uniform(self):
        m = UCBTableModel()
        from agi.distiller import _default_featurizer
        m.fit([Demonstration(state="A", action_distribution={"go": 1},
                             value=0.0)], _default_featurizer)
        p = m.policy("X", ["a", "b", "c"])
        self.assertAlmostEqual(p["a"], 1/3, places=4)


class TestEnsembleModel(unittest.TestCase):

    def test_empty_components_rejected(self):
        with self.assertRaises(InvalidConfig):
            EnsembleModel(components=[])

    def test_pool(self):
        from agi.distiller import _default_featurizer
        knn = KNNModel(k=1, n_features=64, seed=0)
        ucb = UCBTableModel()
        ens = EnsembleModel([knn, ucb])
        demos = [
            Demonstration(state="A", action_distribution={"go": 9, "stop": 1},
                          value=2.0),
        ]
        ens.fit(demos, _default_featurizer)
        p = ens.policy("A", ["go", "stop"])
        self.assertGreater(p["go"], p["stop"])


# =============================================================================
# Distiller orchestrator
# =============================================================================


class TestDistiller(unittest.TestCase):

    def test_default_distiller(self):
        d = Distiller()
        self.assertEqual(d.config.model, MODEL_LINEAR)
        self.assertEqual(len(d), 0)
        self.assertFalse(d.is_fitted)

    def test_observe_and_buffer(self):
        d = Distiller(DistillerConfig(buffer_capacity=2))
        d.observe(state="A", action_distribution={"a": 1}, value=0.5)
        self.assertEqual(len(d), 1)
        d.observe(state="B", action_distribution={"a": 1}, value=0.3)
        self.assertEqual(len(d), 2)
        d.observe(state="C", action_distribution={"a": 1}, value=0.1)
        self.assertEqual(len(d), 2)  # capacity hit

    def test_fit_below_minimum_raises(self):
        d = Distiller(DistillerConfig(min_fit_demonstrations=10))
        d.observe(state="A", action_distribution={"a": 1}, value=0.0)
        with self.assertRaises(NotFitted):
            d.fit()

    def test_first_fit_deploys(self):
        d = linear_distiller(n_features=64, seed=0)
        for i in range(5):
            d.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                      value=float(i))
        rep = d.fit()
        self.assertTrue(rep.deployed)
        self.assertTrue(d.is_fitted)

    def test_fit_records_history(self):
        d = linear_distiller(n_features=64, seed=0)
        for i in range(5):
            d.observe(state=f"s{i}", action_distribution={"a": 1},
                      value=float(i))
        d.fit()
        d.fit()
        self.assertEqual(len(d.history), 2)

    def test_value_unfit_returns_zero(self):
        d = linear_distiller(n_features=64)
        self.assertEqual(d.value("anything"), 0.0)

    def test_policy_unfit_returns_uniform(self):
        d = linear_distiller(n_features=64)
        p = d.policy("x", ["a", "b", "c"])
        self.assertAlmostEqual(p["a"], 1.0/3, places=4)

    def test_certificate_present_and_changes(self):
        d = linear_distiller(n_features=64, seed=0)
        cert0 = d.certificate
        d.observe(state="A", action_distribution={"a": 1}, value=0.0)
        cert1 = d.certificate
        self.assertNotEqual(cert0, cert1)

    def test_as_policy_prior_callable(self):
        d = linear_distiller(n_features=64, seed=0)
        for i in range(5):
            d.observe(state=f"s{i}",
                      action_distribution={"a": 9, "b": 1}, value=0.0)
        d.fit()
        pp = d.as_policy_prior()
        result = pp("s0", ["a", "b"])
        self.assertIn("a", result)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=4)

    def test_as_value_callable(self):
        d = linear_distiller(n_features=64, seed=0)
        for i in range(5):
            d.observe(state=f"s{i}", action_distribution={"a": 1},
                      value=float(i))
        d.fit()
        v = d.as_value()
        self.assertIsInstance(v("s0"), float)


# =============================================================================
# Eval gating
# =============================================================================


class TestEvalGating(unittest.TestCase):

    def test_gate_rejects_when_improvement_is_negligible(self):
        cfg = DistillerConfig(model=MODEL_LINEAR, n_features=64, seed=0,
                              eval_holdout_fraction=0.2, min_improvement=10.0)
        d = Distiller(cfg)
        for i in range(20):
            d.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                      value=0.0)
        rep1 = d.fit()
        # First fit deploys (no incumbent yet).
        self.assertTrue(rep1.deployed)
        # Second fit on same data unlikely to improve by 10.0
        rep2 = d.fit()
        self.assertFalse(rep2.deployed)

    def test_history_eval_metrics_recorded(self):
        d = linear_distiller(n_features=64, seed=0)
        for i in range(20):
            d.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                      value=float(i))
        rep = d.fit()
        self.assertIsInstance(rep.policy_train_cross_entropy, float)
        self.assertIsInstance(rep.value_train_mse, float)


# =============================================================================
# Calibration
# =============================================================================


class TestCalibration(unittest.TestCase):

    def test_temperature_calibration_optional(self):
        d = Distiller(DistillerConfig(
            model=MODEL_LINEAR, n_features=64, seed=0,
            temperature_calibration=True,
        ))
        for i in range(20):
            d.observe(state=f"s{i}", action_distribution={"a": 9, "b": 1},
                      value=0.0)
        rep = d.fit()
        self.assertGreater(rep.temperature, 0.0)

    def test_isotonic_calibration_optional(self):
        d = Distiller(DistillerConfig(
            model=MODEL_LINEAR, n_features=64, seed=0,
            isotonic_value_calibration=True,
        ))
        for i in range(30):
            d.observe(state=f"s{i}", action_distribution={"a": 1},
                      value=float(i))
        rep = d.fit()
        self.assertIsInstance(rep.isotonic_breakpoints, tuple)


# =============================================================================
# Free functions
# =============================================================================


class TestFreeFunctions(unittest.TestCase):

    def test_knn_distiller(self):
        d = knn_distiller(k=3, n_features=64, seed=0)
        self.assertEqual(d.config.model, MODEL_KNN)

    def test_linear_distiller(self):
        d = linear_distiller(n_features=64, seed=0)
        self.assertEqual(d.config.model, MODEL_LINEAR)

    def test_locally_weighted_distiller(self):
        d = locally_weighted_distiller(n_features=64, seed=0)
        self.assertEqual(d.config.model, MODEL_LOCALLY_WEIGHTED)

    def test_ucb_table_distiller(self):
        d = ucb_table_distiller(seed=0)
        self.assertEqual(d.config.model, MODEL_UCB_TABLE)

    def test_ensemble_distiller(self):
        d = ensemble_distiller(n_features=64, seed=0)
        self.assertEqual(d.config.model, MODEL_ENSEMBLE)


# =============================================================================
# Featurizer
# =============================================================================


class TestFeaturizer(unittest.TestCase):

    def test_custom_featurizer_used(self):
        def feat(s):
            return {"_": float(s)}
        d = linear_distiller(n_features=64, seed=0, featurizer=feat)
        for i in range(10):
            d.observe(state=float(i),
                      action_distribution={"go": 1 if i > 4 else 0,
                                            "stop": 1 if i <= 4 else 0},
                      value=float(i))
        d.fit()
        # The custom featurizer should let the model distinguish small/large i
        p_small = d.policy(0.0, ["go", "stop"])
        p_large = d.policy(9.0, ["go", "stop"])
        self.assertGreater(p_large["go"], p_small["go"])


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_certificate(self):
        d1 = linear_distiller(n_features=64, seed=0)
        d2 = linear_distiller(n_features=64, seed=0)
        for i in range(10):
            d1.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                       value=float(i))
            d2.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                       value=float(i))
        r1 = d1.fit()
        r2 = d2.fit()
        self.assertEqual(r1.certificate, r2.certificate)

    def test_different_seed_different_certificate(self):
        d1 = linear_distiller(n_features=64, seed=0)
        d2 = linear_distiller(n_features=64, seed=1)
        for i in range(10):
            d1.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                       value=float(i))
            d2.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                       value=float(i))
        r1 = d1.fit()
        r2 = d2.fit()
        self.assertNotEqual(r1.certificate, r2.certificate)


# =============================================================================
# Composition with Searcher (ExIt loop)
# =============================================================================


class TestExpertIteration(unittest.TestCase):

    def test_exit_step_collects_demonstrations(self):
        from agi.searcher import Searcher, SearcherConfig, ALGORITHM_PUCT
        size = 3
        goal = (2, 2)

        def acts(_s): return ["N", "S", "E", "W"]
        def app(s, a):
            x, y = s
            if a == "N": y -= 1
            elif a == "S": y += 1
            elif a == "E": x += 1
            elif a == "W": x -= 1
            if not (0 <= x < size and 0 <= y < size):
                return s
            return (x, y)
        def term(s): return s == goal
        def rew(s): return 10.0 if s == goal else -1.0
        def feat(s): return {"x": float(s[0]), "y": float(s[1]), "_b": 1.0}

        distiller = linear_distiller(n_features=64, seed=0, featurizer=feat)
        searcher = Searcher(SearcherConfig(algorithm=ALGORITHM_PUCT,
                                            max_iterations=50, seed=0))

        def teacher(state, prior, value):
            rep = searcher.search(state, actions=acts, apply=app, terminal=term,
                                  reward=rew, key=lambda s: s,
                                  policy_prior=prior, value=value)
            return rep.best_action, rep.best_value, rep.root_visits_by_action

        n = expert_iteration_step(
            distiller,
            teacher_search=teacher,
            root=(0, 0),
            n_episodes=1,
            transition=app,
            is_terminal=term,
            max_steps=10,
        )
        self.assertGreater(n, 0)
        self.assertGreater(len(distiller), 0)


# =============================================================================
# Robustness
# =============================================================================


class TestRobustness(unittest.TestCase):

    def test_handles_single_demo(self):
        d = linear_distiller(n_features=64, seed=0)
        d.observe(state="A", action_distribution={"a": 1, "b": 1}, value=0.0)
        # holdout fraction default 0.2 → eval = 0, train = 1
        rep = d.fit()
        self.assertIsInstance(rep, DistillerReport)

    def test_buffer_overflow_keeps_signal(self):
        d = Distiller(DistillerConfig(
            model=MODEL_UCB_TABLE, buffer_capacity=10, seed=0,
        ))
        for i in range(100):
            d.observe(state="A", action_distribution={"go": 1}, value=1.0)
            d.observe(state="B", action_distribution={"stop": 1}, value=-1.0)
        # buffer holds 10 items via reservoir sampling
        self.assertEqual(len(d), 10)
        rep = d.fit()
        self.assertTrue(rep.deployed)

    def test_huber_robust_loss_finite(self):
        # With an extreme outlier and Huber loss, the value head stays
        # finite (clipped per coordinate) — the fit doesn't NaN/inf.
        d = Distiller(DistillerConfig(
            model=MODEL_LINEAR, n_features=64, seed=0,
            value_huber_delta=1.0, lr_value=0.5,
        ))
        for i in range(20):
            d.observe(state=f"s{i}",
                      action_distribution={"a": 1}, value=0.5)
        d.observe(state="OUTLIER",
                  action_distribution={"a": 1}, value=1e9)
        rep = d.fit()
        v = d.value("OUTLIER")
        self.assertTrue(math.isfinite(v))
        self.assertTrue(math.isfinite(rep.value_train_mse))


# =============================================================================
# Report
# =============================================================================


class TestReport(unittest.TestCase):

    def test_report_as_dict(self):
        d = linear_distiller(n_features=64, seed=0)
        for i in range(5):
            d.observe(state=f"s{i}", action_distribution={"a": 1, "b": 1},
                      value=float(i))
        rep = d.fit()
        d2 = rep.as_dict()
        self.assertIn("model", d2)
        self.assertIn("certificate", d2)
        self.assertIn("deployed", d2)


if __name__ == "__main__":
    unittest.main()
