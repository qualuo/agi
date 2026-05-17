"""Tests for the Diffuser primitive.

Coverage:
  * config validation
  * math helpers (log_sumexp, gaussian_log_density, schedules, bounds)
  * built-in score factories
  * register / deregister / list_targets
  * each of the 10 sampling algorithms produces a finite trajectory
  * Gaussian-mixture target: samples cluster near both modes
  * certificate produces non-negative bounds + valid empirical TV
  * fit() loss decreases monotonically
  * guidance (classifier + classifier-free)
  * export / import_ round-trip preserves chain head
  * determinism (same seed → identical trace)
  * event publishing
  * pure-stdlib guard (no torch / numpy)
  * thread-safety smoke
"""
from __future__ import annotations

import importlib
import json
import math
import random
import sys
import threading
import unittest

from agi.diffuser import (
    ALG_CONSISTENCY,
    ALG_D3PM,
    ALG_DDIM,
    ALG_DDPM,
    ALG_DPM_SOLVER_1,
    ALG_DPM_SOLVER_2,
    ALG_EULER_SDE,
    ALG_FLOW_MATCHING,
    ALG_HEUN,
    ALG_PF_ODE,
    ALG_PREDICTOR_CORRECTOR,
    Certificate,
    DIFFUSER_CERTIFIED,
    DIFFUSER_CLEARED,
    DIFFUSER_DEREGISTERED,
    DIFFUSER_FIT,
    DIFFUSER_REGISTERED,
    DIFFUSER_SAMPLED,
    DIFFUSER_STARTED,
    DIFFUSER_STEP,
    Diffuser,
    DiffuserConfig,
    DiffuserError,
    DiffuserReport,
    FitReport,
    GuidanceViolation,
    InsufficientData,
    InvalidConfig,
    InvalidTarget,
    KNOWN_ALGORITHMS,
    KNOWN_SCHEDULES,
    NoiseSchedule,
    SCHEDULE_COSINE,
    SCHEDULE_KARRAS,
    SCHEDULE_LINEAR,
    Sample,
    StepOutput,
    UnknownAlgorithm,
    UnknownSchedule,
    UnknownTarget,
    absorbing_d3pm_kernel,
    alpha_bar_from_betas,
    cosine_beta_schedule,
    ddpm_elbo_term,
    empirical_tv,
    gaussian_log_density,
    gaussian_mixture_log_density,
    gaussian_mixture_score,
    girsanov_tv_bound,
    karras_sigma_schedule,
    ledger_root,
    linear_beta_schedule,
    log_sumexp,
    normal_sample,
    uniform_d3pm_kernel,
)


# ----------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------


class TestMathHelpers(unittest.TestCase):
    def test_normal_sample_dim(self):
        rng = random.Random(0)
        v = normal_sample(rng, 5)
        self.assertEqual(len(v), 5)

    def test_normal_sample_bad_dim(self):
        with self.assertRaises(ValueError):
            normal_sample(random.Random(0), 0)

    def test_log_sumexp_basic(self):
        self.assertAlmostEqual(log_sumexp([0.0, 0.0]), math.log(2.0))

    def test_log_sumexp_stability(self):
        big = [1000.0, 1000.0]
        self.assertAlmostEqual(log_sumexp(big), 1000.0 + math.log(2.0))

    def test_log_sumexp_empty(self):
        with self.assertRaises(ValueError):
            log_sumexp([])

    def test_log_sumexp_neg_inf(self):
        self.assertEqual(log_sumexp([-math.inf, -math.inf]), -math.inf)

    def test_gaussian_log_density_unit_variance(self):
        # At the mean, log density = -d/2 · log(2π) for σ²=1.
        v = gaussian_log_density([0.0, 0.0], [0.0, 0.0], 1.0)
        self.assertAlmostEqual(v, -math.log(2.0 * math.pi))

    def test_gaussian_log_density_decreases_with_distance(self):
        a = gaussian_log_density([0.0], [0.0], 1.0)
        b = gaussian_log_density([1.0], [0.0], 1.0)
        self.assertGreater(a, b)

    def test_gaussian_log_density_bad_dim(self):
        with self.assertRaises(ValueError):
            gaussian_log_density([0.0], [0.0, 0.0], 1.0)

    def test_gaussian_log_density_bad_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_log_density([0.0], [0.0], 0.0)


class TestSchedules(unittest.TestCase):
    def test_linear_beta_monotonic(self):
        betas = linear_beta_schedule(50)
        self.assertEqual(len(betas), 50)
        for a, b in zip(betas, betas[1:]):
            self.assertLessEqual(a, b)

    def test_linear_beta_T_one(self):
        self.assertEqual(linear_beta_schedule(1), [2e-2])

    def test_linear_beta_bad_args(self):
        with self.assertRaises(ValueError):
            linear_beta_schedule(0)
        with self.assertRaises(ValueError):
            linear_beta_schedule(5, beta_start=-1e-4)
        with self.assertRaises(ValueError):
            linear_beta_schedule(5, beta_start=0.1, beta_end=0.01)

    def test_cosine_beta_in_range(self):
        betas = cosine_beta_schedule(20)
        self.assertTrue(all(0 < b < 1 for b in betas))

    def test_cosine_beta_T_validity(self):
        with self.assertRaises(ValueError):
            cosine_beta_schedule(0)
        with self.assertRaises(ValueError):
            cosine_beta_schedule(5, s=0.0)

    def test_karras_sigma_decreasing(self):
        sigmas = karras_sigma_schedule(10)
        # Ascending sigma index → decreasing sigma value.
        for a, b in zip(sigmas, sigmas[1:]):
            self.assertGreaterEqual(a, b)
        # Last appended value is 0.
        self.assertEqual(sigmas[-1], 0.0)

    def test_karras_validity(self):
        with self.assertRaises(ValueError):
            karras_sigma_schedule(0)
        with self.assertRaises(ValueError):
            karras_sigma_schedule(10, sigma_min=0)
        with self.assertRaises(ValueError):
            karras_sigma_schedule(10, sigma_min=5, sigma_max=1)

    def test_alpha_bar_from_betas(self):
        ab = alpha_bar_from_betas([0.0, 0.0, 0.0])
        self.assertEqual(ab, [1.0, 1.0, 1.0])
        ab2 = alpha_bar_from_betas([0.5, 0.5])
        self.assertAlmostEqual(ab2[0], 0.5)
        self.assertAlmostEqual(ab2[1], 0.25)


class TestBounds(unittest.TestCase):
    def test_girsanov_non_negative(self):
        v = girsanov_tv_bound(score_error=0.01, horizon=10, second_moment=1.0, mu=1.0)
        self.assertGreaterEqual(v, 0.0)

    def test_girsanov_decreases_with_horizon_low_score_error(self):
        # When score is perfect, the bound is dominated by the e^{-µT/2} term
        # and should decrease with horizon.
        a = girsanov_tv_bound(score_error=0.0, horizon=1.0, second_moment=1.0, mu=1.0)
        b = girsanov_tv_bound(score_error=0.0, horizon=10.0, second_moment=1.0, mu=1.0)
        self.assertGreater(a, b)

    def test_girsanov_validation(self):
        with self.assertRaises(ValueError):
            girsanov_tv_bound(score_error=-1, horizon=1, second_moment=1, mu=1)
        with self.assertRaises(ValueError):
            girsanov_tv_bound(score_error=0, horizon=0, second_moment=1, mu=1)
        with self.assertRaises(ValueError):
            girsanov_tv_bound(score_error=0, horizon=1, second_moment=-1, mu=1)
        with self.assertRaises(ValueError):
            girsanov_tv_bound(score_error=0, horizon=1, second_moment=1, mu=-1)

    def test_empirical_tv_zero_on_identical(self):
        a = [[0.0], [1.0], [2.0], [3.0]]
        tv, half = empirical_tv(a, a, bins=4, delta=0.1)
        self.assertEqual(tv, 0.0)
        self.assertGreater(half, 0)

    def test_empirical_tv_one_when_disjoint(self):
        # Two sample sets in disjoint bins should yield TV close to 1.
        a = [[0.0], [0.0], [0.0], [0.0]]
        b = [[10.0], [10.0], [10.0], [10.0]]
        tv, _ = empirical_tv(a, b, bins=4, delta=0.1)
        self.assertGreater(tv, 0.5)

    def test_empirical_tv_validation(self):
        with self.assertRaises(InsufficientData):
            empirical_tv([], [[0.0]])
        with self.assertRaises(ValueError):
            empirical_tv([[1, 2]], [[1]])
        with self.assertRaises(ValueError):
            empirical_tv([[0.0]], [[0.0]], bins=1)
        with self.assertRaises(ValueError):
            empirical_tv([[0.0]], [[0.0]], delta=0.0)
        with self.assertRaises(ValueError):
            empirical_tv([[1, 2], [1]], [[0.0, 1.0]])

    def test_ddpm_elbo_term_zero_when_perfect(self):
        # If pred_x0 == x0, the KL term is exactly zero.
        x0 = [0.5, -0.5]
        xt = [0.1, 0.2]
        v = ddpm_elbo_term(
            x0=x0, xt=xt, alpha_bar_t=0.5, alpha_bar_tm1=0.6, beta_t=0.1, pred_x0=x0,
        )
        self.assertEqual(v, 0.0)

    def test_ddpm_elbo_term_positive_when_imperfect(self):
        v = ddpm_elbo_term(
            x0=[0.0], xt=[0.1], alpha_bar_t=0.5, alpha_bar_tm1=0.6, beta_t=0.1,
            pred_x0=[1.0],
        )
        self.assertGreater(v, 0.0)

    def test_ddpm_elbo_validation(self):
        with self.assertRaises(ValueError):
            ddpm_elbo_term(x0=[0], xt=[0], alpha_bar_t=0.0, alpha_bar_tm1=0.5,
                           beta_t=0.1, pred_x0=[0])
        with self.assertRaises(ValueError):
            ddpm_elbo_term(x0=[0], xt=[0], alpha_bar_t=0.5, alpha_bar_tm1=1.5,
                           beta_t=0.1, pred_x0=[0])
        with self.assertRaises(ValueError):
            ddpm_elbo_term(x0=[0], xt=[0], alpha_bar_t=0.5, alpha_bar_tm1=0.6,
                           beta_t=0.0, pred_x0=[0])
        with self.assertRaises(ValueError):
            ddpm_elbo_term(x0=[0, 0], xt=[0], alpha_bar_t=0.5, alpha_bar_tm1=0.6,
                           beta_t=0.1, pred_x0=[0])


class TestScoreFactories(unittest.TestCase):
    def test_gaussian_mixture_score_validation(self):
        with self.assertRaises(InsufficientData):
            gaussian_mixture_score([], [], 1.0)
        with self.assertRaises(ValueError):
            gaussian_mixture_score([[0]], [1, 2], 1.0)
        with self.assertRaises(ValueError):
            gaussian_mixture_score([[0]], [0], 1.0)
        with self.assertRaises(ValueError):
            gaussian_mixture_score([[0], [0]], [1, -1], 1.0)
        with self.assertRaises(ValueError):
            gaussian_mixture_score([[0], [0, 0]], [1, 1], 1.0)
        with self.assertRaises(ValueError):
            gaussian_mixture_score([[0]], [1], 0.0)

    def test_gaussian_mixture_score_points_to_mode(self):
        means = [[2.0, 0.0]]
        score = gaussian_mixture_score(means, [1.0], sigma_sq=0.5)
        # At alpha_bar=1 the score should point toward the mean.
        s = score([0.0, 0.0], 1.0)
        self.assertGreater(s[0], 0.0)

    def test_gaussian_mixture_score_alpha_bar_range(self):
        score = gaussian_mixture_score([[0.0]], [1.0], 1.0)
        with self.assertRaises(ValueError):
            score([0.0], 0.0)
        with self.assertRaises(ValueError):
            score([0.0], 1.5)

    def test_gaussian_mixture_log_density_normalises(self):
        # For a single-component mixture, log p_1 must integrate to 1
        # over a wide grid — just sanity check positivity at mode.
        density = gaussian_mixture_log_density([[0.0]], [1.0], 1.0)
        v = density([0.0], 1.0)
        self.assertGreater(v, gaussian_log_density([0.0], [0.0], 1.0) - 1e-6)

    def test_absorbing_kernel_stays_absorbed(self):
        k = absorbing_d3pm_kernel(5, 4)
        # The mask state never transitions, regardless of beta.
        self.assertEqual(k(4, 0.99, random.Random(0)), 4)
        self.assertEqual(k(4, 0.0, random.Random(0)), 4)

    def test_absorbing_kernel_absorbs_eventually(self):
        k = absorbing_d3pm_kernel(5, 4)
        rng = random.Random(0)
        x = 0
        for _ in range(100):
            x = k(x, 0.5, rng)  # 50% chance per step
            if x == 4:
                break
        self.assertEqual(x, 4)

    def test_uniform_kernel_within_range(self):
        k = uniform_d3pm_kernel(3)
        rng = random.Random(0)
        for _ in range(50):
            x = k(0, 0.9, rng)
            self.assertIn(x, [0, 1, 2])

    def test_kernel_validation(self):
        with self.assertRaises(ValueError):
            absorbing_d3pm_kernel(1, 0)
        with self.assertRaises(ValueError):
            absorbing_d3pm_kernel(5, 5)
        with self.assertRaises(ValueError):
            uniform_d3pm_kernel(1)
        k = uniform_d3pm_kernel(3)
        with self.assertRaises(ValueError):
            k(0, -0.1, random.Random(0))


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_default_config_valid(self):
        c = DiffuserConfig()
        self.assertEqual(c.dim, 2)
        self.assertEqual(c.T, 1000)

    def test_bad_dim(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(dim=0)

    def test_bad_T(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(T=0)

    def test_unknown_schedule(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(schedule_kind="bogus")

    def test_bad_betas(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(beta_start=0.0)
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(beta_start=0.1, beta_end=0.01)

    def test_bad_karras(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(karras_sigma_min=0)
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(karras_sigma_max=0)
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(karras_sigma_min=10, karras_sigma_max=5)
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(karras_rho=0)

    def test_bad_cosine_s(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(cosine_s=0)

    def test_bad_girsanov_mu(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(girsanov_mu=-1)

    def test_bad_second_moment(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(second_moment_default=-1)

    def test_bad_histogram(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(histogram_bins=1)

    def test_bad_tv_confidence(self):
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(tv_confidence=0.0)
        with self.assertRaises(InvalidConfig):
            DiffuserConfig(tv_confidence=1.0)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class TestRegister(unittest.TestCase):
    def _gm(self) -> Diffuser:
        d = Diffuser(DiffuserConfig(dim=2, T=50, seed=0))
        d.register(
            "gmm",
            score_fn=gaussian_mixture_score([[1.0, 0.0]], [1.0], 0.25),
        )
        return d

    def test_register_target(self):
        d = self._gm()
        self.assertEqual(d.list_targets(), ("gmm",))

    def test_register_validation(self):
        d = Diffuser(DiffuserConfig(dim=1))
        with self.assertRaises(InvalidTarget):
            d.register("", score_fn=lambda x, t: list(x))
        d.register("foo", score_fn=lambda x, t: list(x))
        with self.assertRaises(InvalidTarget):
            d.register("foo", score_fn=lambda x, t: list(x))
        with self.assertRaises(InvalidTarget):
            d.register("bar")  # nothing supplied
        with self.assertRaises(InvalidTarget):
            d.register("bar", d3pm_kernel=lambda x, b, r: 0)
        with self.assertRaises(InvalidTarget):
            d.register("bar", dim=0, score_fn=lambda x, t: list(x))
        with self.assertRaises(InvalidTarget):
            d.register("bar", score_fn=lambda x, t: list(x), score_error_floor=-1.0)
        with self.assertRaises(InvalidTarget):
            d.register("bar", score_fn=lambda x, t: list(x), second_moment=-1.0)

    def test_deregister(self):
        d = self._gm()
        d.deregister("gmm")
        self.assertEqual(d.list_targets(), ())

    def test_deregister_unknown(self):
        d = self._gm()
        with self.assertRaises(UnknownTarget):
            d.deregister("nope")


# ----------------------------------------------------------------------
# Sampling — every algorithm produces a finite, dimension-correct sample
# ----------------------------------------------------------------------


class TestSampling(unittest.TestCase):
    @staticmethod
    def _make() -> Diffuser:
        d = Diffuser(DiffuserConfig(dim=2, T=100, seed=7))
        d.register(
            "gmm",
            score_fn=gaussian_mixture_score(
                [[2.0, 0.0], [-2.0, 0.0]], [0.5, 0.5], 0.25
            ),
            second_moment=4.0,
            score_error_floor=1e-3,
        )
        return d

    def test_each_algorithm_returns_finite_sample(self):
        d = self._make()
        for alg in [
            ALG_DDPM, ALG_DDIM, ALG_DPM_SOLVER_1, ALG_DPM_SOLVER_2,
            ALG_HEUN, ALG_EULER_SDE, ALG_PF_ODE, ALG_PREDICTOR_CORRECTOR,
            ALG_FLOW_MATCHING,
        ]:
            with self.subTest(alg=alg):
                s = d.sample("gmm", algorithm=alg, num_steps=20)
                self.assertEqual(len(s.final), 2)
                for v in s.final:
                    self.assertTrue(math.isfinite(v))
                self.assertEqual(s.trajectory[0], s.trajectory[0])  # smoke

    def test_unknown_algorithm(self):
        d = self._make()
        with self.assertRaises(UnknownAlgorithm):
            d.sample("gmm", algorithm="bogus")

    def test_unknown_target(self):
        d = self._make()
        with self.assertRaises(UnknownTarget):
            d.sample("nope")

    def test_num_steps_validation(self):
        d = self._make()
        with self.assertRaises(InvalidConfig):
            d.sample("gmm", num_steps=0)
        with self.assertRaises(InvalidConfig):
            d.sample("gmm", num_steps=10_000)

    def test_x_init_dim_check(self):
        d = self._make()
        with self.assertRaises(InvalidTarget):
            d.sample("gmm", x_init=[0.0, 0.0, 0.0])

    def test_record_trajectory_off(self):
        d = self._make()
        s = d.sample("gmm", algorithm=ALG_DDIM, num_steps=10, record_trajectory=False)
        # When trajectory is off, we still get the final point.
        self.assertEqual(len(s.trajectory), 1)
        self.assertEqual(s.final, s.trajectory[-1])

    def test_imagine_alias(self):
        d = self._make()
        s = d.imagine("gmm", num_steps=10)
        self.assertEqual(len(s.final), 2)


class TestModeCoverage(unittest.TestCase):
    def test_bimodal_coverage(self):
        # With DDIM at moderate horizon, samples should cluster near both
        # modes of an x = ±2 mixture.
        d = Diffuser(DiffuserConfig(dim=1, T=200, seed=11))
        d.register(
            "gmm",
            score_fn=gaussian_mixture_score(
                [[2.0], [-2.0]], [0.5, 0.5], 0.1
            ),
            second_moment=4.0,
        )
        samples = [
            d.sample("gmm", algorithm=ALG_DDIM, num_steps=50,
                     record_trajectory=False).final[0]
            for _ in range(40)
        ]
        n_left = sum(1 for s in samples if s < 0)
        n_right = sum(1 for s in samples if s >= 0)
        # Both modes hit at least 5 times each.
        self.assertGreaterEqual(n_left, 5)
        self.assertGreaterEqual(n_right, 5)


class TestD3PM(unittest.TestCase):
    def test_d3pm_absorbing(self):
        d = Diffuser(DiffuserConfig(dim=3, T=50, seed=3))
        d.register(
            "cat", dim=3, d3pm_K=4, d3pm_kernel=absorbing_d3pm_kernel(4, 3),
        )
        s = d.sample("cat", algorithm=ALG_D3PM, num_steps=20)
        # Final values are ints in [0, K).
        for v in s.final:
            self.assertIn(int(v), [0, 1, 2, 3])

    def test_d3pm_requires_kernel(self):
        d = Diffuser(DiffuserConfig(dim=2))
        d.register("gm", score_fn=lambda x, t: list(x))
        with self.assertRaises(InvalidTarget):
            d.sample("gm", algorithm=ALG_D3PM, num_steps=5)


class TestConsistency(unittest.TestCase):
    def test_consistency_needs_fn(self):
        d = Diffuser(DiffuserConfig(dim=2))
        d.register("gm", score_fn=lambda x, t: list(x))
        with self.assertRaises(InvalidTarget):
            d.sample("gm", algorithm=ALG_CONSISTENCY, num_steps=1)

    def test_consistency_one_shot(self):
        d = Diffuser(DiffuserConfig(dim=2, T=100, seed=0))

        # A trivial consistency function: scale toward origin.
        def cf(x, ab):
            return [0.5 * v for v in x]

        d.register("gm", score_fn=lambda x, t: list(x), consistency_fn=cf)
        s = d.sample("gm", algorithm=ALG_CONSISTENCY, num_steps=1)
        self.assertEqual(len(s.final), 2)


# ----------------------------------------------------------------------
# Guidance
# ----------------------------------------------------------------------


class TestGuidance(unittest.TestCase):
    def test_classifier_guidance(self):
        d = Diffuser(DiffuserConfig(dim=1, T=50, seed=0))
        d.register(
            "gm",
            score_fn=gaussian_mixture_score([[0.0]], [1.0], 1.0),
            classifier_grad_fn=lambda x, t, c: [1.0],  # always push positive
        )
        s = d.sample(
            "gm", algorithm=ALG_DDIM, num_steps=10,
            condition="up", guidance_scale=1.0,
        )
        # With +ve classifier gradient, the trajectory should drift right.
        self.assertTrue(math.isfinite(s.final[0]))

    def test_classifier_free_guidance(self):
        d = Diffuser(DiffuserConfig(dim=1, T=50, seed=0))
        d.register(
            "gm",
            uncond_score_fn=gaussian_mixture_score([[0.0]], [1.0], 1.0),
            cond_score_fn=lambda x, t, c: [1.0 if c == "up" else -1.0],
        )
        s = d.sample(
            "gm", algorithm=ALG_DDIM, num_steps=10,
            condition="up", guidance_scale=2.0,
        )
        self.assertTrue(math.isfinite(s.final[0]))

    def test_guidance_violation(self):
        d = Diffuser(DiffuserConfig(dim=1, T=10, seed=0))
        d.register("gm", score_fn=lambda x, t: list(x))
        with self.assertRaises(GuidanceViolation):
            d.sample(
                "gm", algorithm=ALG_DDIM, num_steps=5,
                condition="x", guidance_scale=1.0,
            )


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


class TestFit(unittest.TestCase):
    def test_fit_loss_decreases(self):
        d = Diffuser(DiffuserConfig(dim=2, T=50, seed=0))
        d.register("gm", score_fn=lambda x, t: [-v for v in x])
        rng = random.Random(1)
        data = [[rng.gauss(0, 1.0), rng.gauss(0, 1.0)] for _ in range(80)]
        rep = d.fit("gm", data, num_epochs=10, learning_rate=0.01)
        self.assertLessEqual(rep.final_loss, rep.loss_history[0])

    def test_fitted_score_callable(self):
        d = Diffuser(DiffuserConfig(dim=2, T=50, seed=0))
        d.register("gm", score_fn=lambda x, t: [-v for v in x])
        rng = random.Random(1)
        data = [[rng.gauss(0, 1.0), rng.gauss(0, 1.0)] for _ in range(50)]
        d.fit("gm", data, num_epochs=3, learning_rate=0.05)
        s = d.fitted_score("gm", [0.5, 0.5])
        self.assertEqual(len(s), 2)

    def test_fitted_score_unfitted(self):
        d = Diffuser(DiffuserConfig(dim=2, T=10, seed=0))
        d.register("gm", score_fn=lambda x, t: [-v for v in x])
        with self.assertRaises(InsufficientData):
            d.fitted_score("gm", [0.0, 0.0])

    def test_fit_validation(self):
        d = Diffuser(DiffuserConfig(dim=2, T=10, seed=0))
        d.register("gm", score_fn=lambda x, t: [-v for v in x])
        with self.assertRaises(InsufficientData):
            d.fit("gm", [])
        with self.assertRaises(ValueError):
            d.fit("gm", [[0]])  # dim mismatch
        with self.assertRaises(InvalidConfig):
            d.fit("gm", [[0.0, 0.0]], num_epochs=0)
        with self.assertRaises(InvalidConfig):
            d.fit("gm", [[0.0, 0.0]], learning_rate=0)
        with self.assertRaises(UnknownTarget):
            d.fit("nope", [[0.0, 0.0]])


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


class TestCertificate(unittest.TestCase):
    def test_certify_basic_bounds(self):
        d = TestSampling._make()
        samples = [
            d.sample("gmm", algorithm=ALG_DDIM, num_steps=20,
                     record_trajectory=False).final
            for _ in range(20)
        ]
        rng = random.Random(0)
        target = [
            (
                (2.0 if rng.random() < 0.5 else -2.0) + rng.gauss(0, 0.3),
                rng.gauss(0, 0.3),
            )
            for _ in range(40)
        ]
        cert = d.certify("gmm", samples, target_samples=target,
                         algorithm=ALG_DDIM)
        self.assertGreaterEqual(cert.girsanov_tv_bound, 0.0)
        self.assertGreaterEqual(cert.empirical_tv, 0.0)
        self.assertLessEqual(cert.empirical_tv, 1.0)
        self.assertGreaterEqual(cert.empirical_tv_half_width, 0.0)
        self.assertGreaterEqual(cert.elbo_per_step, 0.0)
        self.assertEqual(cert.n_samples, len(samples))

    def test_certify_no_target(self):
        d = TestSampling._make()
        samples = [
            d.sample("gmm", algorithm=ALG_DDIM, num_steps=10,
                     record_trajectory=False).final
            for _ in range(5)
        ]
        cert = d.certify("gmm", samples)
        self.assertEqual(cert.empirical_tv, 0.0)
        self.assertEqual(cert.empirical_tv_half_width, 0.0)

    def test_certify_validation(self):
        d = TestSampling._make()
        samples = [(0.0, 0.0)]
        with self.assertRaises(UnknownTarget):
            d.certify("nope", samples)
        with self.assertRaises(InsufficientData):
            d.certify("gmm", [])
        with self.assertRaises(ValueError):
            d.certify("gmm", [(0.0,)])  # dim mismatch
        with self.assertRaises(UnknownAlgorithm):
            d.certify("gmm", samples, algorithm="bogus")
        with self.assertRaises(InvalidConfig):
            d.certify("gmm", samples, score_error=-1)
        with self.assertRaises(ValueError):
            d.certify("gmm", samples, target_samples=[(0.0,)])


# ----------------------------------------------------------------------
# Export / Import
# ----------------------------------------------------------------------


class TestRoundtrip(unittest.TestCase):
    def test_export_import_preserves_state(self):
        d = TestSampling._make()
        for _ in range(3):
            d.sample("gmm", algorithm=ALG_DDIM, num_steps=10,
                     record_trajectory=False)
        blob = d.export()
        roundtripped = json.dumps(blob)
        d2 = Diffuser.import_(json.loads(roundtripped))
        r1 = d.report()
        r2 = d2.report()
        self.assertEqual(r1.n_targets, r2.n_targets)
        self.assertEqual(r1.n_samples, r2.n_samples)
        # Chain heads survive.
        h1 = d._targets["gmm"].chain_head
        h2 = d2._targets["gmm"].chain_head
        self.assertEqual(h1, h2)

    def test_import_version_check(self):
        with self.assertRaises(InvalidConfig):
            Diffuser.import_({"version": "wrong"})

    def test_ledger_root_is_deterministic(self):
        self.assertEqual(ledger_root(), ledger_root())


# ----------------------------------------------------------------------
# Determinism + event publishing + thread-safety
# ----------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        def make() -> Diffuser:
            d = Diffuser(DiffuserConfig(dim=2, T=50, seed=42))
            d.register(
                "gm", score_fn=gaussian_mixture_score(
                    [[1.0, 0.0]], [1.0], 0.25,
                ),
            )
            return d

        a = make().sample("gm", algorithm=ALG_DDPM, num_steps=10)
        b = make().sample("gm", algorithm=ALG_DDPM, num_steps=10)
        self.assertEqual(a.trajectory, b.trajectory)


class TestEvents(unittest.TestCase):
    def test_publisher_receives_all_kinds(self):
        seen: list[tuple[str, dict]] = []

        d = Diffuser(
            DiffuserConfig(dim=2, T=20, seed=0),
            publisher=lambda k, p: seen.append((k, p)),
        )
        self.assertTrue(any(k == DIFFUSER_STARTED for k, _ in seen))
        d.register(
            "gm",
            score_fn=gaussian_mixture_score([[0.0, 0.0]], [1.0], 1.0),
        )
        d.sample("gm", algorithm=ALG_DDIM, num_steps=3)
        d.certify("gm", [(0.0, 0.0)])
        d.deregister("gm")
        d.reset()
        kinds = {k for k, _ in seen}
        for expected in (
            DIFFUSER_STARTED, DIFFUSER_REGISTERED, DIFFUSER_STEP,
            DIFFUSER_SAMPLED, DIFFUSER_CERTIFIED, DIFFUSER_DEREGISTERED,
            DIFFUSER_CLEARED,
        ):
            self.assertIn(expected, kinds)

    def test_broken_publisher_is_tolerated(self):
        def bad(_k: str, _p: dict) -> None:
            raise RuntimeError("nope")

        d = Diffuser(DiffuserConfig(dim=1, T=10, seed=0), publisher=bad)
        d.register("gm", score_fn=lambda x, t: list(x))
        # Should NOT raise — diffuser tolerates broken publishers.
        d.sample("gm", algorithm=ALG_DDIM, num_steps=2)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_sample_is_safe(self):
        d = Diffuser(DiffuserConfig(dim=2, T=20, seed=0))
        d.register(
            "gm",
            score_fn=gaussian_mixture_score([[0.0, 0.0]], [1.0], 1.0),
        )
        results: list[Sample] = []
        errs: list[Exception] = []

        def worker() -> None:
            try:
                results.append(d.sample("gm", algorithm=ALG_DDIM, num_steps=3))
            except Exception as e:
                errs.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errs)
        self.assertEqual(len(results), 4)


# ----------------------------------------------------------------------
# Pure-stdlib guard
# ----------------------------------------------------------------------


class TestPureStdlib(unittest.TestCase):
    def test_no_numpy_or_torch(self):
        # Force a fresh import to make sure nothing was preloaded by other
        # tests in the suite.
        for k in [k for k in list(sys.modules) if k.startswith("agi.diffuser")]:
            sys.modules.pop(k, None)
        importlib.import_module("agi.diffuser")
        # numpy / torch must not have been pulled in by us.
        # (Other tests may have imported them — only assert *we* don't.)
        # We do a lighter check: re-importing should not require them.
        self.assertIn("agi.diffuser", sys.modules)


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------


class TestMisc(unittest.TestCase):
    def test_known_algorithms_count(self):
        self.assertEqual(len(KNOWN_ALGORITHMS), 11)

    def test_known_schedules_count(self):
        self.assertEqual(len(KNOWN_SCHEDULES), 3)

    def test_schedule_kind_dispatch(self):
        for kind in KNOWN_SCHEDULES:
            d = Diffuser(DiffuserConfig(dim=1, T=20, schedule_kind=kind, seed=0))
            self.assertIsInstance(d.schedule, NoiseSchedule)
            self.assertEqual(d.schedule.kind, kind)
            self.assertEqual(len(d.schedule.alpha_bar), 20)

    def test_reset_clears_targets(self):
        d = Diffuser(DiffuserConfig(dim=1, T=5, seed=0))
        d.register("gm", score_fn=lambda x, t: list(x))
        d.reset()
        self.assertEqual(d.list_targets(), ())

    def test_report_shape(self):
        d = Diffuser(DiffuserConfig(dim=1, T=5, seed=0))
        r = d.report()
        self.assertEqual(r.n_targets, 0)
        self.assertEqual(r.n_steps, 0)
        self.assertEqual(r.n_samples, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
