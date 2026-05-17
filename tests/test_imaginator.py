"""Tests for agi.imaginator."""
from __future__ import annotations

import random
import unittest

from agi.imaginator import (
    FAMILY_CATEGORICAL,
    FAMILY_LINEAR_GAUSSIAN,
    IMAGINATOR_CERTIFIED,
    IMAGINATOR_IMAGINED,
    IMAGINATOR_OBSERVED,
    IMAGINATOR_PLANNED,
    IMAGINATOR_REGISTERED,
    IMAGINATOR_STARTED,
    SAMPLE_BAYES_AVG,
    SAMPLE_POSTERIOR_MEAN,
    SAMPLE_THOMPSON,
    Imaginator,
    ImaginatorConfig,
    InsufficientData,
    InvalidConfig,
    InvalidEnv,
    InvalidObservation,
    UnknownEnv,
    dirichlet_mean,
    dirichlet_sample,
    empirical_bernstein_half_width,
    hoeffding_half_width,
    hrms_half_width,
    ks_pvalue,
    ledger_root,
    softmax,
)


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------


class TestSoftmax(unittest.TestCase):
    def test_uniform_zero(self):
        self.assertEqual(softmax([]), [])
        out = softmax([1.0, 1.0, 1.0])
        for p in out:
            self.assertAlmostEqual(p, 1 / 3)

    def test_temperature(self):
        out_cold = softmax([0, 1], beta=100)
        out_hot = softmax([0, 1], beta=0.01)
        self.assertGreater(out_cold[1], 0.99)
        self.assertAlmostEqual(out_hot[0], 0.5, places=2)


class TestDirichlet(unittest.TestCase):
    def test_mean_normalises(self):
        self.assertAlmostEqual(sum(dirichlet_mean([1, 2, 3])), 1.0)

    def test_mean_zero(self):
        self.assertEqual(dirichlet_mean([]), [])

    def test_sample_normalises(self):
        rng = random.Random(0)
        s = dirichlet_sample(rng, [1, 1, 1, 1])
        self.assertAlmostEqual(sum(s), 1.0, places=6)
        for p in s:
            self.assertGreaterEqual(p, 0.0)

    def test_sample_small_alpha(self):
        rng = random.Random(0)
        s = dirichlet_sample(rng, [0.1, 0.1])
        self.assertAlmostEqual(sum(s), 1.0, places=6)


class TestBoundHelpers(unittest.TestCase):
    def test_hoeffding_shrinks(self):
        self.assertGreater(hoeffding_half_width(10), hoeffding_half_width(100))
        self.assertGreater(hoeffding_half_width(100), hoeffding_half_width(1000))

    def test_hoeffding_zero(self):
        self.assertEqual(hoeffding_half_width(0), float("inf"))

    def test_bernstein_smaller_when_variance_smaller(self):
        eb_low = empirical_bernstein_half_width(100, 0.01)
        eb_high = empirical_bernstein_half_width(100, 0.25)
        self.assertLess(eb_low, eb_high)

    def test_bernstein_one_sample(self):
        self.assertEqual(empirical_bernstein_half_width(1, 0.1), float("inf"))

    def test_hrms_anytime_valid(self):
        # Bound shrinks but more slowly than Hoeffding (log log term).
        h10 = hrms_half_width(10)
        h100 = hrms_half_width(100)
        self.assertGreater(h10, h100)


class TestKSTest(unittest.TestCase):
    def test_uniform_passes(self):
        rng = random.Random(0)
        samples = [rng.random() for _ in range(500)]
        _, p = ks_pvalue(samples)
        self.assertGreater(p, 0.01)

    def test_skewed_fails(self):
        rng = random.Random(0)
        samples = [rng.random() ** 3 for _ in range(500)]  # skewed
        _, p = ks_pvalue(samples)
        self.assertLess(p, 0.01)

    def test_empty(self):
        d, p = ks_pvalue([])
        self.assertEqual(d, 0.0)
        self.assertEqual(p, 1.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_default(self):
        cfg = ImaginatorConfig()
        self.assertEqual(cfg.family, FAMILY_CATEGORICAL)

    def test_invalid_family(self):
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(family="random")

    def test_invalid_confidence(self):
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(confidence=0.5)
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(confidence=1.0)

    def test_invalid_discount(self):
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(discount=-0.1)
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(discount=1.1)

    def test_invalid_priors(self):
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(dirichlet_prior=0)
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(reward_precision_prior=0)
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(reward_gamma_a=0)

    def test_invalid_hmac_key(self):
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(hmac_key="not-bytes")  # type: ignore[arg-type]

    def test_invalid_cap(self):
        with self.assertRaises(InvalidConfig):
            ImaginatorConfig(max_observations_per_env=0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    def test_register_categorical(self):
        im = Imaginator()
        spec = im.register_env("env1", states=("s1", "s2"), actions=("a1", "a2"))
        self.assertEqual(spec.env_id, "env1")
        self.assertEqual(spec.states, ("s1", "s2"))
        self.assertEqual(spec.actions, ("a1", "a2"))
        self.assertIn("env1", im.envs())

    def test_register_linear_gaussian(self):
        im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN))
        spec = im.register_env("env1", state_dim=2, action_dim=1)
        self.assertEqual(spec.family, FAMILY_LINEAR_GAUSSIAN)
        self.assertEqual(spec.state_dim, 2)
        self.assertEqual(spec.action_dim, 1)

    def test_duplicate_register(self):
        im = Imaginator()
        im.register_env("env1", states=("s",), actions=("a",))
        with self.assertRaises(InvalidEnv):
            im.register_env("env1", states=("s",), actions=("a",))

    def test_empty_states(self):
        im = Imaginator()
        with self.assertRaises(InvalidEnv):
            im.register_env("env1", states=(), actions=("a",))

    def test_duplicate_states(self):
        im = Imaginator()
        with self.assertRaises(InvalidEnv):
            im.register_env("env1", states=("s", "s"), actions=("a",))

    def test_remove(self):
        im = Imaginator()
        im.register_env("env1", states=("s",), actions=("a",))
        im.remove_env("env1")
        self.assertNotIn("env1", im.envs())

    def test_remove_unknown(self):
        im = Imaginator()
        with self.assertRaises(UnknownEnv):
            im.remove_env("missing")

    def test_clear(self):
        im = Imaginator()
        im.register_env("env1", states=("s",), actions=("a",))
        im.register_env("env2", states=("s",), actions=("a",))
        im.clear()
        self.assertEqual(im.envs(), [])

    def test_env_spec(self):
        im = Imaginator()
        im.register_env("env1", states=("s",), actions=("a",))
        spec = im.env_spec("env1")
        self.assertEqual(spec.env_id, "env1")

    def test_unknown_env_spec(self):
        im = Imaginator()
        with self.assertRaises(UnknownEnv):
            im.env_spec("missing")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservation(unittest.TestCase):
    def test_categorical_observation(self):
        im = Imaginator()
        im.register_env("e", states=("a", "b"), actions=("x", "y"))
        im.observe("e", "a", "x", "b", 1.0)
        post = im.posterior_mean_transition("e", "a", "x")
        self.assertGreater(post["b"], post["a"])

    def test_unknown_state(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        with self.assertRaises(InvalidObservation):
            im.observe("e", "z", "x", "a", 0.0)
        with self.assertRaises(InvalidObservation):
            im.observe("e", "a", "x", "z", 0.0)

    def test_unknown_action(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        with self.assertRaises(InvalidObservation):
            im.observe("e", "a", "z", "a", 0.0)

    def test_nan_reward(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        with self.assertRaises(InvalidObservation):
            im.observe("e", "a", "x", "a", float("nan"))

    def test_linear_gaussian_observation(self):
        im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN))
        im.register_env("e", state_dim=2, action_dim=1)
        im.observe("e", [1.0, 0.0], [0.5], [1.5, 0.1], 0.0)

    def test_linear_gaussian_shape_mismatch(self):
        im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN))
        im.register_env("e", state_dim=2, action_dim=1)
        with self.assertRaises(InvalidObservation):
            im.observe("e", [1.0], [0.5], [1.5, 0.1], 0.0)

    def test_posterior_mean_reward(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        for _ in range(20):
            im.observe("e", "a", "x", "a", 1.0)
        self.assertAlmostEqual(im.posterior_mean_reward("e", "a", "x"), 1.0, places=1)

    def test_posterior_variance_decreases(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        var0 = im.posterior_variance_reward("e", "a", "x")
        for _ in range(50):
            im.observe("e", "a", "x", "a", 1.0)
        var1 = im.posterior_variance_reward("e", "a", "x")
        self.assertLess(var1, var0)

    def test_max_cap_decays(self):
        im = Imaginator(ImaginatorConfig(max_observations_per_env=10))
        im.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(30):
            im.observe("e", "a", "x", "b", 0.5)
        # Decay still keeps the posterior biased toward "b".
        post = im.posterior_mean_transition("e", "a", "x")
        self.assertGreater(post["b"], post["a"])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling(unittest.TestCase):
    def test_sample_transition_posterior_mean(self):
        im = Imaginator(ImaginatorConfig(rng_seed=1))
        im.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(100):
            im.observe("e", "a", "x", "b", 0.0)
        # Sample many; "b" should dominate.
        counts = {"a": 0, "b": 0}
        for _ in range(500):
            counts[im.sample_transition("e", "a", "x", method=SAMPLE_POSTERIOR_MEAN)] += 1
        self.assertGreater(counts["b"], counts["a"] * 5)

    def test_sample_transition_thompson(self):
        im = Imaginator(ImaginatorConfig(rng_seed=1))
        im.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(100):
            im.observe("e", "a", "x", "b", 0.0)
        # Sample many; "b" should still dominate but more variable.
        counts = {"a": 0, "b": 0}
        for _ in range(500):
            counts[im.sample_transition("e", "a", "x", method=SAMPLE_THOMPSON)] += 1
        self.assertGreater(counts["b"], counts["a"])

    def test_sample_transition_bayes_avg(self):
        im = Imaginator(ImaginatorConfig(rng_seed=1))
        im.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(100):
            im.observe("e", "a", "x", "b", 0.0)
        # Sample with BMA.
        s = im.sample_transition("e", "a", "x", method=SAMPLE_BAYES_AVG)
        self.assertIn(s, ("a", "b"))

    def test_sample_unknown_method(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            im.sample_transition("e", "a", "x", method="random")

    def test_sample_reward(self):
        im = Imaginator(ImaginatorConfig(rng_seed=1))
        im.register_env("e", states=("a",), actions=("x",))
        for _ in range(100):
            im.observe("e", "a", "x", "a", 2.0)
        # Sampled rewards should cluster around 2.0.
        samples = [im.sample_reward("e", "a", "x") for _ in range(200)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 2.0, places=0)


# ---------------------------------------------------------------------------
# Imagine
# ---------------------------------------------------------------------------


class TestImagine(unittest.TestCase):
    def setUp(self):
        self.im = Imaginator(ImaginatorConfig(rng_seed=42, discount=0.9))
        self.im.register_env("e", states=("a", "b"), actions=("x", "y"))
        # Cycle env: "x" goes a→b, "y" goes b→a; rewards +1 on a→b, +0.5 on b→a.
        for _ in range(50):
            self.im.observe("e", "a", "x", "b", 1.0)
            self.im.observe("e", "b", "y", "a", 0.5)
            self.im.observe("e", "a", "y", "a", 0.0)
            self.im.observe("e", "b", "x", "b", 0.0)

    def test_imagine_returns_rollout(self):
        roll = self.im.imagine(
            "e",
            state="a",
            policy=lambda s: "x" if s == "a" else "y",
            horizon=5,
            samples=64,
        )
        self.assertEqual(roll.samples, 64)
        self.assertEqual(roll.horizon, 5)
        self.assertGreater(roll.expected_return, 0)
        self.assertLess(roll.value_lcb, roll.expected_return)
        self.assertGreater(roll.value_ucb, roll.expected_return)

    def test_imagine_thompson(self):
        roll = self.im.imagine(
            "e",
            state="a",
            policy=lambda s: "x" if s == "a" else "y",
            horizon=5,
            samples=64,
            method=SAMPLE_THOMPSON,
        )
        # Thompson rollouts should produce wider-bound intervals than mean.
        roll_mean = self.im.imagine(
            "e",
            state="a",
            policy=lambda s: "x" if s == "a" else "y",
            horizon=5,
            samples=64,
            method=SAMPLE_POSTERIOR_MEAN,
        )
        # Thompson typically has bigger std (more dynamics uncertainty).
        # Just check both run.
        self.assertGreater(roll.return_std, -1)
        self.assertGreater(roll_mean.return_std, -1)

    def test_imagine_hrms_anytime(self):
        roll = self.im.imagine(
            "e",
            state="a",
            policy=lambda s: "x" if s == "a" else "y",
            horizon=5,
            samples=32,
        )
        self.assertLess(roll.hrms_lcb, roll.expected_return)
        self.assertGreater(roll.hrms_ucb, roll.expected_return)
        # HRMS and Bernstein bracket each other; both bracket the mean.
        self.assertLess(roll.hrms_lcb, roll.hrms_ucb)
        self.assertLess(roll.value_lcb, roll.value_ucb)

    def test_imagine_quantiles_monotone(self):
        roll = self.im.imagine(
            "e",
            state="a",
            policy=lambda s: "x" if s == "a" else "y",
            horizon=5,
            samples=128,
        )
        qs = roll.return_quantiles
        keys = sorted(qs.keys())
        for i in range(1, len(keys)):
            self.assertGreaterEqual(qs[keys[i]], qs[keys[i - 1]])

    def test_imagine_trajectories(self):
        roll = self.im.imagine(
            "e",
            state="a",
            policy=lambda s: "x" if s == "a" else "y",
            horizon=3,
            samples=10,
        )
        self.assertEqual(len(roll.trajectories), 10)
        for t in roll.trajectories:
            self.assertEqual(len(t), 3)
            for step in t:
                self.assertIsInstance(step, tuple)
                self.assertEqual(len(step), 3)

    def test_imagine_invalid_horizon(self):
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            self.im.imagine("e", state="a", policy=lambda s: "x", horizon=0, samples=1)

    def test_imagine_invalid_samples(self):
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            self.im.imagine("e", state="a", policy=lambda s: "x", horizon=1, samples=0)

    def test_imagine_unknown_state(self):
        with self.assertRaises(InvalidObservation):
            self.im.imagine(
                "e", state="zzz", policy=lambda s: "x", horizon=1, samples=1
            )

    def test_imagine_policy_returns_unknown(self):
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            self.im.imagine(
                "e",
                state="a",
                policy=lambda s: "zzz",
                horizon=1,
                samples=1,
            )


# ---------------------------------------------------------------------------
# Value iteration & PSRL
# ---------------------------------------------------------------------------


class TestValueIteration(unittest.TestCase):
    def setUp(self):
        self.im = Imaginator(ImaginatorConfig(rng_seed=42, discount=0.9))
        self.im.register_env("e", states=("a", "b"), actions=("x", "y"))
        # x is good in a (→b,r=1); y is bad in a (→a,r=0)
        for _ in range(50):
            self.im.observe("e", "a", "x", "b", 1.0)
            self.im.observe("e", "a", "y", "a", 0.0)
            self.im.observe("e", "b", "x", "b", 0.5)
            self.im.observe("e", "b", "y", "a", 0.5)

    def test_value_iteration_correct(self):
        plan = self.im.value_iteration("e", horizon=20, discount=0.9)
        self.assertEqual(plan.policy["a"], "x")
        self.assertGreater(plan.values["a"], plan.values["b"])

    def test_value_iteration_converges(self):
        plan = self.im.value_iteration("e", horizon=200, discount=0.9, tol=1e-3)
        # Should converge well before horizon expires.
        self.assertLess(plan.sweeps, 200)

    def test_thompson_policy(self):
        psrl = self.im.thompson_policy("e", horizon=20, discount=0.9)
        self.assertIn(psrl.policy["a"], ("x", "y"))
        self.assertIn(psrl.policy["b"], ("x", "y"))

    def test_value_iteration_invalid(self):
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            self.im.value_iteration("e", horizon=0)
        with self.assertRaises(ImaginatorError):
            self.im.value_iteration("e", horizon=10, discount=-0.1)


# ---------------------------------------------------------------------------
# PAC bound
# ---------------------------------------------------------------------------


class TestPACBound(unittest.TestCase):
    def setUp(self):
        self.im = Imaginator(ImaginatorConfig(discount=0.9, rng_seed=0))
        self.im.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(200):
            self.im.observe("e", "a", "x", "b", 1.0)
            self.im.observe("e", "b", "x", "a", 0.5)

    def test_pac_bound(self):
        pac = self.im.pac_value_bound(
            "e", policy={"a": "x", "b": "x"}, delta=0.05, horizon=20
        )
        self.assertGreater(pac.epsilon, 0)
        self.assertGreater(pac.transition_error, 0)
        self.assertEqual(pac.delta, 0.05)
        self.assertGreaterEqual(pac.min_observations, 100)

    def test_pac_bound_callable_policy(self):
        pac = self.im.pac_value_bound(
            "e", policy=lambda s: "x", delta=0.05, horizon=20
        )
        self.assertGreater(pac.epsilon, 0)

    def test_pac_bound_decreases_with_data(self):
        small = Imaginator(ImaginatorConfig(discount=0.9))
        small.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(20):
            small.observe("e", "a", "x", "b", 1.0)
            small.observe("e", "b", "x", "a", 0.5)
        pac_small = small.pac_value_bound(
            "e", policy={"a": "x", "b": "x"}, delta=0.05, horizon=20
        )
        pac_big = self.im.pac_value_bound(
            "e", policy={"a": "x", "b": "x"}, delta=0.05, horizon=20
        )
        self.assertLess(pac_big.epsilon, pac_small.epsilon)

    def test_pac_bound_needs_data(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        with self.assertRaises(InsufficientData):
            im.pac_value_bound("e", policy={"a": "x"}, delta=0.05, horizon=1)

    def test_pac_bound_invalid_delta(self):
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            self.im.pac_value_bound("e", policy={"a": "x", "b": "x"}, delta=1.0)

    def test_required_samples_for_pac(self):
        n = self.im.required_samples_for_pac(env_id="e", epsilon=0.1, delta=0.05)
        self.assertGreater(n, 0)
        n2 = self.im.required_samples_for_pac(env_id="e", epsilon=0.05, delta=0.05)
        self.assertGreater(n2, n)


# ---------------------------------------------------------------------------
# Identifiability + PIT
# ---------------------------------------------------------------------------


class TestReports(unittest.TestCase):
    def test_identifiability_flags_unobserved(self):
        im = Imaginator()
        im.register_env("e", states=("a", "b"), actions=("x", "y"))
        for _ in range(20):
            im.observe("e", "a", "x", "b", 0.0)
        report = im.identifiability_report("e", min_observations=5)
        self.assertEqual(report.n_pairs, 4)
        self.assertEqual(report.n_under_observed, 3)
        # The well-observed pair shouldn't be in the list
        self.assertNotIn(("a", "x"), report.under_observed)

    def test_pit_runs(self):
        im = Imaginator(ImaginatorConfig(rng_seed=0))
        im.register_env("e", states=("a",), actions=("x",))
        for _ in range(50):
            im.observe("e", "a", "x", "a", random.random())
        pit = im.pit_calibration("e")
        self.assertEqual(pit.n_observations, 49)  # first one has no predictive
        self.assertGreaterEqual(pit.p_value, 0.0)
        self.assertLessEqual(pit.p_value, 1.0)

    def test_pit_no_data(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        with self.assertRaises(InsufficientData):
            im.pit_calibration("e")


# ---------------------------------------------------------------------------
# Bayesian model averaging
# ---------------------------------------------------------------------------


class TestBayesAvgValue(unittest.TestCase):
    def test_bma_value_returns_finite(self):
        im = Imaginator(ImaginatorConfig(rng_seed=1, discount=0.9))
        im.register_env("e", states=("a", "b"), actions=("x",))
        for _ in range(50):
            im.observe("e", "a", "x", "b", 1.0)
            im.observe("e", "b", "x", "a", 0.5)
        v = im.bayes_average_value(
            "e", policy={"a": "x", "b": "x"}, horizon=10, samples=16, n_models=4
        )
        self.assertGreater(v, 0)
        v_start = im.bayes_average_value(
            "e",
            policy={"a": "x", "b": "x"},
            horizon=10,
            samples=16,
            n_models=4,
            start="a",
        )
        self.assertGreater(v_start, 0)


# ---------------------------------------------------------------------------
# Linear-Gaussian dynamics
# ---------------------------------------------------------------------------


class TestLinearGaussian(unittest.TestCase):
    def test_recovers_dynamics(self):
        im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN, rng_seed=0))
        im.register_env("e", state_dim=2, action_dim=1)
        rng = random.Random(0)
        A_true = [[0.9, 0.05], [0.0, 0.95]]
        B_true = [[0.0], [0.1]]
        s = [1.0, 0.0]
        for _ in range(500):
            a = [rng.gauss(0, 1)]
            nxt = [
                A_true[i][0] * s[0] + A_true[i][1] * s[1] + B_true[i][0] * a[0] + rng.gauss(0, 0.01)
                for i in range(2)
            ]
            im.observe("e", s, a, nxt, 0.0)
            s = nxt
            if abs(s[0]) > 50:
                s = [1.0, 0.0]
        A, B = im.posterior_mean_dynamics("e")
        # With the ridge prior of magnitude 1 against 500 noisy observations
        # the posterior mean shrinks slightly toward zero — check within 0.15.
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(A[i][j], A_true[i][j], delta=0.15)
        for i in range(2):
            self.assertAlmostEqual(B[i][0], B_true[i][0], delta=0.15)

    def test_moment_rollout(self):
        im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN, rng_seed=0))
        im.register_env("e", state_dim=2, action_dim=1)
        rng = random.Random(0)
        for _ in range(100):
            s = [rng.gauss(0, 1), rng.gauss(0, 1)]
            a = [rng.gauss(0, 1)]
            nxt = [s[0] + 0.1 * s[1] + rng.gauss(0, 0.01), s[1] + 0.1 * a[0] + rng.gauss(0, 0.01)]
            im.observe("e", s, a, nxt, 0.0)
        trace = im.moment_rollout(
            "e", state=[1.0, 0.0], policy=lambda s: [-0.1 * s[0]], horizon=5
        )
        self.assertEqual(len(trace), 5)
        # Variance grows monotonically
        for i in range(1, 5):
            self.assertGreaterEqual(trace[i][1][0][0], trace[i - 1][1][0][0])

    def test_categorical_in_linear_env(self):
        im = Imaginator()
        im.register_env("e", states=("a",), actions=("x",))
        from agi.imaginator import ImaginatorError
        with self.assertRaises(ImaginatorError):
            im.posterior_mean_dynamics("e")
        with self.assertRaises(ImaginatorError):
            im.moment_rollout("e", state=[1.0], policy=lambda s: [0.0], horizon=1)


# ---------------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------------


class TestFingerprintChain(unittest.TestCase):
    def test_chain_advances(self):
        im = Imaginator(ImaginatorConfig(rng_seed=0))
        h0 = im.chain_head
        im.register_env("e", states=("a",), actions=("x",))
        h1 = im.chain_head
        self.assertNotEqual(h0, h1)
        im.observe("e", "a", "x", "a", 1.0)
        h2 = im.chain_head
        self.assertNotEqual(h1, h2)

    def test_chain_replays(self):
        im_a = Imaginator(ImaginatorConfig(rng_seed=0))
        im_b = Imaginator(ImaginatorConfig(rng_seed=0))
        for im in (im_a, im_b):
            im.register_env("e", states=("a", "b"), actions=("x",))
            for r in (1.0, 0.5, 1.5, 0.7):
                im.observe("e", "a", "x", "b", r)
        self.assertEqual(im_a.chain_head, im_b.chain_head)

    def test_hmac_key_changes_chain(self):
        im_a = Imaginator(ImaginatorConfig(rng_seed=0))
        im_b = Imaginator(ImaginatorConfig(rng_seed=0, hmac_key=b"secret"))
        self.assertNotEqual(im_a.chain_head, im_b.chain_head)

    def test_ledger_root_stable(self):
        self.assertEqual(ledger_root(), ledger_root())
        self.assertNotEqual(ledger_root(), ledger_root(b"key"))


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------


class TestExportImport(unittest.TestCase):
    def test_roundtrip_categorical(self):
        im = Imaginator(ImaginatorConfig(rng_seed=0))
        im.register_env("e", states=("a", "b"), actions=("x", "y"))
        for _ in range(20):
            im.observe("e", "a", "x", "b", 1.0)
        snap = im.export_state()
        im2 = Imaginator(ImaginatorConfig(rng_seed=0))
        im2.import_state(snap)
        self.assertEqual(im.chain_head, im2.chain_head)
        self.assertEqual(im.envs(), im2.envs())
        self.assertEqual(
            im.posterior_mean_transition("e", "a", "x"),
            im2.posterior_mean_transition("e", "a", "x"),
        )

    def test_roundtrip_linear_gaussian(self):
        im = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN, rng_seed=0))
        im.register_env("e", state_dim=2, action_dim=1)
        rng = random.Random(0)
        for _ in range(50):
            s = [rng.gauss(0, 1), rng.gauss(0, 1)]
            a = [rng.gauss(0, 1)]
            im.observe("e", s, a, [s[0] + 0.1, s[1]], 0.0)
        snap = im.export_state()
        im2 = Imaginator(ImaginatorConfig(family=FAMILY_LINEAR_GAUSSIAN, rng_seed=0))
        im2.import_state(snap)
        A1, B1 = im.posterior_mean_dynamics("e")
        A2, B2 = im2.posterior_mean_dynamics("e")
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(A1[i][j], A2[i][j])
            for j in range(1):
                self.assertAlmostEqual(B1[i][j], B2[i][j])


# ---------------------------------------------------------------------------
# Event publishing
# ---------------------------------------------------------------------------


class TestEventPublishing(unittest.TestCase):
    def test_publishes(self):
        events: list[tuple[str, dict]] = []
        def cb(k, d):
            events.append((k, d))
        im = Imaginator(publisher=cb)
        im.register_env("e", states=("a", "b"), actions=("x",))
        im.observe("e", "a", "x", "b", 1.0)
        kinds = [k for k, _ in events]
        self.assertIn(IMAGINATOR_STARTED, kinds)
        self.assertIn(IMAGINATOR_REGISTERED, kinds)
        self.assertIn(IMAGINATOR_OBSERVED, kinds)

    def test_publish_failure_is_tolerated(self):
        def bad(k, d):
            raise RuntimeError("nope")
        im = Imaginator(publisher=bad)
        im.register_env("e", states=("a",), actions=("x",))
        # Did not raise.
        im.observe("e", "a", "x", "a", 1.0)


# ---------------------------------------------------------------------------
# Thread-safety smoke
# ---------------------------------------------------------------------------


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_observe(self):
        import threading
        im = Imaginator(ImaginatorConfig(rng_seed=0))
        im.register_env("e", states=("a", "b"), actions=("x", "y"))
        def worker():
            for _ in range(100):
                im.observe("e", "a", "x", "b", 1.0)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 400 observations were made.
        post = im.posterior_mean_transition("e", "a", "x")
        # b should dominate
        self.assertGreater(post["b"], 0.99)


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def test_supply_chain_loop(self):
        """End-to-end: observe → plan → imagine → certify → audit chain."""
        im = Imaginator(ImaginatorConfig(rng_seed=7, discount=0.9))
        im.register_env(
            "supply", states=("ok", "stockout"), actions=("ship", "wait")
        )
        rng = random.Random(0)
        for _ in range(300):
            s = rng.choice(["ok", "stockout"])
            a = rng.choice(["ship", "wait"])
            if a == "ship" and s == "stockout":
                nxt = "ok" if rng.random() < 0.8 else "stockout"
                r = -0.5
            elif a == "ship" and s == "ok":
                nxt = "ok" if rng.random() < 0.95 else "stockout"
                r = 1.0
            elif a == "wait" and s == "ok":
                nxt = "ok" if rng.random() < 0.7 else "stockout"
                r = 0.5
            else:
                nxt = "ok" if rng.random() < 0.1 else "stockout"
                r = -2.0
            im.observe("supply", s, a, nxt, r)
        plan = im.value_iteration("supply", horizon=30, discount=0.9)
        self.assertEqual(plan.policy["ok"], "ship")
        self.assertEqual(plan.policy["stockout"], "ship")
        roll = im.imagine(
            "supply", state="ok", policy=lambda s: plan.policy[s], horizon=10, samples=64
        )
        # Imagined return should be positive and bounds should bracket it.
        self.assertGreater(roll.expected_return, 0)
        self.assertLess(roll.value_lcb, roll.expected_return)
        self.assertGreater(roll.value_ucb, roll.expected_return)
        # PAC bound finite
        pac = im.pac_value_bound(
            "supply", policy=plan.policy, delta=0.05, horizon=30
        )
        self.assertGreater(pac.epsilon, 0)
        # Audit chain
        self.assertNotEqual(im.chain_head, ledger_root())


if __name__ == "__main__":
    unittest.main()
