"""Tests for `agi.bayesopt` — Bayesian optimisation as a runtime primitive.

The tests cover four layers, in order of generality:

  1. Numerical primitives — Cholesky, Φ / Φ⁻¹, kernels, gradients.
  2. GP regression — posterior shape and convergence.
  3. Acquisition functions — closed-form sanity at standard limits.
  4. BayesOpt main class — convergence on toy oracles, batch behaviour,
     replay determinism, regret bounds, mixed-domain handling, errors.

Every probabilistic test is seeded so the suite is deterministic.
"""
from __future__ import annotations

import math
import random

import pytest

from agi.bayesopt import (
    ACQ_EI,
    ACQ_KG,
    ACQ_PI,
    ACQ_THOMPSON,
    ACQ_UCB,
    BAYESOPT_OBSERVED,
    BAYESOPT_REPORT,
    BAYESOPT_STARTED,
    BAYESOPT_SUGGESTED,
    BayesOpt,
    BayesOptConfig,
    BayesOptError,
    CategoricalDim,
    ContinuousBox,
    GPPosterior,
    InvalidDomain,
    InvalidObservation,
    KERNEL_MATERN32,
    KERNEL_MATERN52,
    KERNEL_RBF,
    KNOWN_ACQUISITIONS,
    KNOWN_EVENTS,
    KNOWN_KERNELS,
    Kernel,
    MAXIMISE,
    MINIMISE,
    MixedDomain,
    Observation,
    Suggestion,
    UnknownAcquisition,
    UnknownKernel,
    acq_ei,
    acq_pi,
    acq_thompson_value,
    acq_ucb,
    gp_log_marginal_likelihood,
    gp_predict,
    learn_hyperparameters,
    make_kernel,
    maximise,
    minimise,
    optimise_acquisition,
)
from agi.bayesopt import _cholesky, _GPFit, _halton_point, _phi, _phi_inv, _phi_pdf


# ---------------------------------------------------------------------
# 1.  Numerical primitives
# ---------------------------------------------------------------------


class TestNumericalPrimitives:

    def test_phi_phi_inv_round_trip(self):
        for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            assert abs(_phi(_phi_inv(p)) - p) < 1e-9

    def test_phi_endpoints(self):
        assert _phi(-10.0) == pytest.approx(0.0, abs=1e-10)
        assert _phi(0.0) == pytest.approx(0.5, abs=1e-10)
        assert _phi(10.0) == pytest.approx(1.0, abs=1e-10)

    def test_phi_pdf_integrates_around_1(self):
        # Riemann sum over [-6, 6]; rough check.
        total = sum(_phi_pdf(x * 0.01) * 0.01 for x in range(-600, 601))
        assert abs(total - 1.0) < 1e-3

    def test_cholesky_identity(self):
        I = [[1.0, 0.0], [0.0, 1.0]]
        L = _cholesky(I)
        assert L[0][0] == pytest.approx(1.0)
        assert L[1][1] == pytest.approx(1.0)
        assert L[0][1] == 0.0

    def test_cholesky_reconstructs(self):
        # SPD matrix
        A = [[4.0, 2.0, 1.0],
             [2.0, 3.0, 0.5],
             [1.0, 0.5, 2.0]]
        L = _cholesky(A)
        # Verify L Lᵀ ≈ A
        for i in range(3):
            for j in range(3):
                s = sum(L[i][k] * L[j][k] for k in range(3))
                assert abs(s - A[i][j]) < 1e-9

    def test_halton_in_unit_cube(self):
        for i in range(1, 20):
            u = _halton_point(i, 3)
            assert len(u) == 3
            for ui in u:
                assert 0.0 <= ui < 1.0

    def test_halton_distinct(self):
        points = {tuple(_halton_point(i, 2)) for i in range(1, 50)}
        assert len(points) == 49  # all distinct in the unit square


# ---------------------------------------------------------------------
# 2.  Kernels
# ---------------------------------------------------------------------


class TestKernels:

    def test_known_kernel_set(self):
        assert KNOWN_KERNELS == {KERNEL_RBF, KERNEL_MATERN52, KERNEL_MATERN32}

    @pytest.mark.parametrize("name", list(KNOWN_KERNELS))
    def test_kernel_self_value_equals_signal_var(self, name):
        k = make_kernel(name, dim=2, lengthscale=0.5, signal_var=1.7)
        assert k.value([0.3, 0.4], [0.3, 0.4]) == pytest.approx(1.7, rel=1e-12)

    @pytest.mark.parametrize("name", list(KNOWN_KERNELS))
    def test_kernel_symmetric(self, name):
        k = make_kernel(name, dim=2, lengthscale=0.3)
        x, y = [0.1, 0.2], [0.7, 0.4]
        assert k.value(x, y) == pytest.approx(k.value(y, x))

    @pytest.mark.parametrize("name", list(KNOWN_KERNELS))
    def test_kernel_decays_with_distance(self, name):
        k = make_kernel(name, dim=1, lengthscale=0.1)
        near = k.value([0.5], [0.55])
        far = k.value([0.5], [0.9])
        assert near > far > 0.0

    @pytest.mark.parametrize("name", list(KNOWN_KERNELS))
    def test_kernel_gradient_numerical(self, name):
        k = make_kernel(name, dim=2, lengthscale=0.4, signal_var=1.2)
        x = [0.31, 0.62]
        y = [0.55, 0.18]
        analytic = k.grad_x(x, y)
        eps = 1e-6
        for d in range(2):
            xp = x[:]
            xm = x[:]
            xp[d] += eps
            xm[d] -= eps
            num = (k.value(xp, y) - k.value(xm, y)) / (2.0 * eps)
            assert abs(analytic[d] - num) < 1e-4

    def test_kernel_gradient_zero_at_same_point(self):
        for name in KNOWN_KERNELS:
            k = make_kernel(name, dim=2)
            g = k.grad_x([0.4, 0.7], [0.4, 0.7])
            assert all(abs(gi) < 1e-9 for gi in g)

    def test_kernel_gram_psd(self):
        k = make_kernel(KERNEL_RBF, dim=1, lengthscale=0.2)
        X = [[0.1], [0.3], [0.5], [0.9]]
        K = k.gram(X)
        # Smallest eigenvalue ≥ 0 → Cholesky succeeds with no jitter.
        L = _cholesky([row[:] for row in K])
        assert all(L[i][i] > 0.0 for i in range(len(L)))

    def test_unknown_kernel_raises(self):
        with pytest.raises(UnknownKernel):
            make_kernel("bogus", dim=2)

    def test_kernel_constructor_validates(self):
        with pytest.raises(BayesOptError):
            Kernel(name=KERNEL_RBF, lengthscales=(0.0,), signal_var=1.0)
        with pytest.raises(BayesOptError):
            Kernel(name=KERNEL_RBF, lengthscales=(0.5,), signal_var=-1.0)


# ---------------------------------------------------------------------
# 3.  GP regression
# ---------------------------------------------------------------------


class TestGP:

    def test_predict_known_point_returns_observation(self):
        k = make_kernel(KERNEL_RBF, dim=1, lengthscale=0.2)
        X = [[0.2], [0.5], [0.8]]
        y = [1.0, -0.5, 0.7]
        mu, var = gp_predict(kernel=k, X=X, y=y, x_star=[0.5], noise_var=1e-8)
        assert mu == pytest.approx(-0.5, abs=1e-3)
        assert var < 1e-2

    def test_predict_far_returns_prior(self):
        k = make_kernel(KERNEL_RBF, dim=1, lengthscale=0.1)
        X = [[0.5]]
        y = [1.0]
        mu, var = gp_predict(kernel=k, X=X, y=y, x_star=[10.0])
        # Far from data → posterior reverts to data mean (1.0 here).
        assert mu == pytest.approx(1.0, abs=1e-3)
        assert var > 0.5

    def test_log_marginal_likelihood_finite(self):
        k = make_kernel(KERNEL_MATERN52, dim=1, lengthscale=0.2)
        X = [[0.1], [0.3], [0.5], [0.7]]
        y = [0.1, 0.3, 0.2, 0.5]
        ll = gp_log_marginal_likelihood(kernel=k, X=X, y=y)
        assert math.isfinite(ll)

    def test_learn_hyperparameters_improves_ll(self):
        # Generate a sample from a known-lengthscale GP-like function.
        rng = random.Random(11)
        X = [[rng.uniform(0.0, 1.0)] for _ in range(10)]
        y = [math.sin(6.0 * x[0]) for x in X]
        k_bad = make_kernel(KERNEL_MATERN52, dim=1, lengthscale=0.01)
        ll_before = gp_log_marginal_likelihood(kernel=k_bad, X=X, y=y)
        k_opt = learn_hyperparameters(kernel=k_bad, X=X, y=y)
        ll_after = gp_log_marginal_likelihood(kernel=k_opt, X=X, y=y)
        assert ll_after >= ll_before - 1e-6
        # Lengthscale should have moved toward something reasonable.
        assert 0.05 < k_opt.lengthscales[0] < 5.0

    def test_predict_grad_matches_finite_diff(self):
        k = make_kernel(KERNEL_MATERN52, dim=2, lengthscale=0.3)
        X = [[0.2, 0.3], [0.5, 0.7], [0.9, 0.1]]
        y = [0.5, -0.2, 0.8]
        fit = _GPFit(kernel=k, X=X, y=y, noise_var=1e-4)
        x = [0.4, 0.55]
        mu, var, gmu, gvar = fit.predict_grad(x)
        eps = 1e-5
        for d in range(2):
            xp = x[:]
            xm = x[:]
            xp[d] += eps
            xm[d] -= eps
            mp, vp = fit.predict(xp)
            mm, vm = fit.predict(xm)
            num_gmu = (mp - mm) / (2.0 * eps)
            num_gvar = (vp - vm) / (2.0 * eps)
            assert abs(gmu[d] - num_gmu) < 1e-3
            assert abs(gvar[d] - num_gvar) < 1e-3

    def test_gp_var_nonneg(self):
        k = make_kernel(KERNEL_RBF, dim=1, lengthscale=0.2)
        X = [[0.2], [0.4], [0.6]]
        y = [1.0, 2.0, 0.5]
        fit = _GPFit(kernel=k, X=X, y=y, noise_var=1e-4)
        for grid in (i / 50.0 for i in range(51)):
            mu, var = fit.predict([grid])
            assert var >= 0.0
            assert math.isfinite(mu)


# ---------------------------------------------------------------------
# 4.  Acquisitions — closed-form checks
# ---------------------------------------------------------------------


class TestAcquisitions:

    def test_known_acquisition_set(self):
        assert KNOWN_ACQUISITIONS == {ACQ_UCB, ACQ_EI, ACQ_PI, ACQ_THOMPSON, ACQ_KG}

    def test_ucb_increases_with_std(self):
        a1 = acq_ucb(mu=0.5, std=0.1, beta_t=2.0, direction=MAXIMISE)
        a2 = acq_ucb(mu=0.5, std=0.5, beta_t=2.0, direction=MAXIMISE)
        assert a2 > a1

    def test_ucb_minimisation_flips_sign(self):
        a_max = acq_ucb(mu=0.5, std=0.1, beta_t=2.0, direction=MAXIMISE)
        a_min = acq_ucb(mu=0.5, std=0.1, beta_t=2.0, direction=MINIMISE)
        assert a_max != a_min
        # LCB = -μ + √β σ for minimisation
        assert a_min == pytest.approx(-0.5 + math.sqrt(2.0) * 0.1)

    def test_ei_zero_when_well_below_incumbent(self):
        # Maximisation: μ << incumbent and σ tiny ⇒ near zero EI.
        ei = acq_ei(mu=-5.0, std=1e-9, incumbent=0.0, direction=MAXIMISE)
        assert ei == pytest.approx(0.0, abs=1e-6)

    def test_ei_at_incumbent_equals_sigma_phi0(self):
        # If μ = incumbent, σ > 0, EI = σ φ(0) = σ / √(2π)
        sigma = 0.4
        ei = acq_ei(mu=1.0, std=sigma, incumbent=1.0, direction=MAXIMISE)
        assert ei == pytest.approx(sigma / math.sqrt(2.0 * math.pi))

    def test_ei_directions_consistent(self):
        # Maximising f and minimising -f should give the same EI.
        ei_max = acq_ei(mu=1.0, std=0.3, incumbent=0.5, direction=MAXIMISE)
        ei_min = acq_ei(mu=-1.0, std=0.3, incumbent=-0.5, direction=MINIMISE)
        assert ei_max == pytest.approx(ei_min, rel=1e-9)

    def test_pi_monotone_in_diff(self):
        a1 = acq_pi(mu=0.1, std=0.2, incumbent=0.5, direction=MAXIMISE)
        a2 = acq_pi(mu=0.7, std=0.2, incumbent=0.5, direction=MAXIMISE)
        assert a2 > a1

    def test_thompson_value_consistent(self):
        v_max = acq_thompson_value(mu=0.5, std=0.1, sample_z=1.0, direction=MAXIMISE)
        v_min = acq_thompson_value(mu=0.5, std=0.1, sample_z=1.0, direction=MINIMISE)
        # MAXIMISE: μ + z σ ; MINIMISE: -(μ + z σ)
        assert v_max == pytest.approx(0.5 + 0.1)
        assert v_min == pytest.approx(-(0.5 + 0.1))


# ---------------------------------------------------------------------
# 5.  Domain validation
# ---------------------------------------------------------------------


class TestDomain:

    def test_continuous_box_dim(self):
        b = ContinuousBox(low=(0.0, -1.0), high=(1.0, 2.0))
        assert b.dim == 2

    def test_continuous_box_invalid(self):
        with pytest.raises(InvalidDomain):
            ContinuousBox(low=(1.0,), high=(0.0,))
        with pytest.raises(InvalidDomain):
            ContinuousBox(low=(), high=())
        with pytest.raises(InvalidDomain):
            ContinuousBox(low=(0.0,), high=(1.0, 2.0))

    def test_continuous_box_clip_sample_contains(self):
        b = ContinuousBox(low=(0.0,), high=(1.0,))
        rng = random.Random(1)
        for _ in range(50):
            x = b.sample(rng)
            assert b.contains(x)
        out = b.clip([2.0])
        assert out == [1.0]
        in_ = b.clip([-0.5])
        assert in_ == [0.0]

    def test_categorical_dim_empty_invalid(self):
        with pytest.raises(InvalidDomain):
            CategoricalDim(name="x", values=())

    def test_mixed_domain_dim(self):
        m = MixedDomain(
            cont=ContinuousBox(low=(0.0,), high=(1.0,)),
            cats=(CategoricalDim(name="k", values=("a", "b", "c")),),
        )
        assert m.dim == 2
        rng = random.Random(0)
        for _ in range(10):
            x = m.sample(rng)
            assert 0.0 <= x[0] <= 1.0
            assert 0.0 <= x[1] <= 2.0
            assert x[1] == int(x[1])  # categorical encoded as integer-valued float

    def test_mixed_domain_clip_categorical(self):
        m = MixedDomain(
            cont=None,
            cats=(CategoricalDim(name="k", values=("a", "b", "c")),),
        )
        assert m.clip([10.0]) == [2.0]
        assert m.clip([-5.0]) == [0.0]
        assert m.clip([1.4]) == [1.0]


# ---------------------------------------------------------------------
# 6.  BayesOpt main class
# ---------------------------------------------------------------------


class TestBayesOpt:

    def test_starts_with_no_observations(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        assert bo.n_observations == 0
        assert bo.best() is None
        assert bo.regret_bound() is None
        assert bo.cumulative_regret_bound() is None

    def test_cold_start_uses_halton(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        s1 = bo.suggest()
        s2 = bo.suggest()
        assert s1.x != s2.x
        assert s1.rationale.startswith("cold-start")
        assert s2.rationale.startswith("cold-start")

    def test_minimisation_quadratic_converges(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(-2.0,), high=(2.0,)),
            config=BayesOptConfig(
                direction=MINIMISE, acquisition=ACQ_EI, seed=7,
                noise_var=1e-4,
            ),
        )
        for _ in range(25):
            sug = bo.suggest()
            y = (sug.x[0] - 0.7) ** 2
            bo.observe(sug.x, y)
        best = bo.best()
        assert best is not None
        assert abs(best.x[0] - 0.7) < 0.1
        assert best.y < 0.05

    def test_maximisation_2d_converges(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0, 0.0), high=(1.0, 1.0)),
            config=BayesOptConfig(
                direction=MAXIMISE, acquisition=ACQ_UCB, seed=3,
                noise_var=1e-4,
            ),
        )
        for _ in range(30):
            sug = bo.suggest()
            y = 1.0 - ((sug.x[0] - 0.3) ** 2 + (sug.x[1] - 0.6) ** 2)
            bo.observe(sug.x, y)
        best = bo.best()
        assert best is not None
        assert best.y > 0.95

    @pytest.mark.parametrize("acq",
                             [ACQ_UCB, ACQ_EI, ACQ_PI, ACQ_THOMPSON])
    def test_all_acquisitions_converge_on_1d(self, acq):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(
                direction=MINIMISE, acquisition=acq, seed=5,
                noise_var=1e-4,
            ),
        )
        for _ in range(20):
            sug = bo.suggest()
            bo.observe(sug.x, (sug.x[0] - 0.4) ** 2)
        best = bo.best()
        assert best is not None
        assert best.y < 0.05

    def test_predict_returns_gpposterior(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=2, noise_var=1e-4),
        )
        for x, y in [(0.1, 0.5), (0.5, 1.0), (0.9, 0.2)]:
            bo.observe((x,), y)
        post = bo.predict((0.5,))
        assert isinstance(post, GPPosterior)
        assert post.mean == pytest.approx(1.0, abs=0.1)
        lo, hi = post.credible(0.95)
        assert lo <= post.mean <= hi

    def test_credible_interval_levels(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        bo.observe((0.5,), 1.0)
        bo.observe((0.7,), 2.0)
        lo50, hi50 = bo.credible_interval((0.6,), level=0.5)
        lo95, hi95 = bo.credible_interval((0.6,), level=0.95)
        assert hi95 - lo95 > hi50 - lo50

    def test_observe_invalid_dim_raises(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0, 0.0), high=(1.0, 1.0)),
            config=BayesOptConfig(seed=0),
        )
        with pytest.raises(InvalidObservation):
            bo.observe((0.5,), 1.0)

    def test_observe_nonfinite_y_raises(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        with pytest.raises(InvalidObservation):
            bo.observe((0.5,), float("nan"))

    def test_replay_determinism(self):
        bo1 = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=42, acquisition=ACQ_EI),
        )
        bo2 = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=42, acquisition=ACQ_EI),
        )
        for _ in range(10):
            s1 = bo1.suggest()
            s2 = bo2.suggest()
            assert s1.x == s2.x
            bo1.observe(s1.x, (s1.x[0] - 0.3) ** 2)
            bo2.observe(s2.x, (s2.x[0] - 0.3) ** 2)
        assert bo1.fingerprint() == bo2.fingerprint()

    def test_fingerprint_changes_with_data(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        fp0 = bo.fingerprint()
        bo.observe((0.5,), 1.0)
        fp1 = bo.fingerprint()
        assert fp0 != fp1

    def test_clear_resets_state(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        bo.observe((0.5,), 1.0)
        bo.observe((0.7,), 2.0)
        bo.clear()
        assert bo.n_observations == 0
        assert bo.best() is None

    def test_batch_distinct(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0, 0.0), high=(1.0, 1.0)),
            config=BayesOptConfig(
                direction=MAXIMISE, acquisition=ACQ_UCB, seed=11,
            ),
        )
        # Seed with a single observation so the GP is informative.
        bo.observe((0.5, 0.5), 0.0)
        bo.observe((0.4, 0.6), 0.1)
        bo.observe((0.6, 0.4), 0.1)
        batch = bo.suggest_batch(4)
        xs = [s.x for s in batch]
        assert len(set(xs)) == 4

    def test_regret_bound_decreases_with_observations(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0, noise_var=1e-4),
        )
        # Cover the space densely.
        for i in range(20):
            x = i / 19.0
            bo.observe((x,), (x - 0.5) ** 2)
        # Now the max posterior std should be small.
        assert bo.regret_bound() < 1.0

    def test_report_fields(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        for x, y in [(0.1, 1.0), (0.5, 0.0), (0.9, 1.0)]:
            bo.observe((x,), y)
        r = bo.report()
        assert r.n_observations == 3
        assert r.best_x is not None
        assert r.kernel_name == KERNEL_MATERN52
        assert r.acquisition == ACQ_EI
        assert r.fingerprint == bo.fingerprint()
        assert r.beta_t > 0.0

    def test_events_emitted(self):
        events: list[tuple[str, dict]] = []

        def sink(t, p):
            events.append((t, dict(p)))

        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0, learn_hypers_every=0),
            event_sink=sink,
        )
        bo.suggest()
        bo.observe((0.5,), 1.0)
        bo.report()
        types = {e[0] for e in events}
        assert BAYESOPT_STARTED in types
        assert BAYESOPT_SUGGESTED in types
        assert BAYESOPT_OBSERVED in types
        assert BAYESOPT_REPORT in types

    def test_event_set_complete(self):
        # Spot-check the public event set is self-consistent.
        assert {BAYESOPT_STARTED, BAYESOPT_SUGGESTED,
                BAYESOPT_OBSERVED, BAYESOPT_REPORT} <= KNOWN_EVENTS

    def test_invalid_config_raises(self):
        with pytest.raises(BayesOptError):
            BayesOpt(
                domain=ContinuousBox(low=(0.0,), high=(1.0,)),
                config=BayesOptConfig(direction="zigzag"),
            )
        with pytest.raises(UnknownAcquisition):
            BayesOpt(
                domain=ContinuousBox(low=(0.0,), high=(1.0,)),
                config=BayesOptConfig(acquisition="bogus"),
            )
        with pytest.raises(UnknownKernel):
            BayesOpt(
                domain=ContinuousBox(low=(0.0,), high=(1.0,)),
                config=BayesOptConfig(kernel="bogus"),
            )

    def test_minimise_driver(self):
        report = minimise(
            f=lambda x: (x[0] - 0.5) ** 2 + (x[1] + 0.3) ** 2,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            n_steps=25,
            n_seed=99,
        )
        assert report.best_y is not None
        assert report.best_y < 0.05

    def test_maximise_driver(self):
        report = maximise(
            f=lambda x: -((x[0] - 0.2) ** 2 + (x[1] - 0.4) ** 2),
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            n_steps=20,
            n_seed=99,
        )
        assert report.best_y is not None
        assert report.best_y > -0.1


class TestMixedDomain:

    def test_mixed_continuous_categorical(self):
        cont = ContinuousBox(low=(0.0,), high=(1.0,))
        cats = (CategoricalDim(name="k", values=("a", "b", "c")),)
        domain = MixedDomain(cont=cont, cats=cats)

        def f(x):
            target_cont = 0.65
            target_cat = 1.0  # 'b'
            return -((x[0] - target_cont) ** 2 + (x[1] - target_cat) ** 2)

        bo = BayesOpt(
            domain=domain,
            config=BayesOptConfig(
                direction=MAXIMISE, acquisition=ACQ_EI, seed=23,
            ),
        )
        for _ in range(35):
            sug = bo.suggest()
            bo.observe(sug.x, f(sug.x))
        best = bo.best()
        assert best is not None
        assert int(round(best.x[1])) == 1  # picked category 'b'
        assert abs(best.x[0] - 0.65) < 0.15


class TestRegretBoundsMonotone:

    def test_cumulative_regret_bound_grows_sublinearly(self):
        """Cumulative regret bound √(C₁ T β_T γ_T) should grow sublinearly
        in T for the GP-UCB setting on a 1D RBF kernel."""
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(
                direction=MAXIMISE, acquisition=ACQ_UCB,
                seed=17, kernel=KERNEL_RBF, noise_var=1e-3,
                learn_hypers_every=0,
            ),
        )
        bounds: list[float] = []
        for i in range(30):
            sug = bo.suggest()
            y = 1.0 - (sug.x[0] - 0.4) ** 2
            bo.observe(sug.x, y)
            cr = bo.cumulative_regret_bound()
            if cr is not None:
                bounds.append(cr)
        # Bound should be monotonically non-decreasing in T.
        for a, b in zip(bounds, bounds[1:]):
            assert b >= a - 1e-9
        # And R_T / T should shrink over time (sublinear regret).
        ratio_early = bounds[2] / 3
        ratio_late = bounds[-1] / len(bounds)
        assert ratio_late <= ratio_early + 1e-9


class TestAcquisitionOptimisation:

    def test_optimise_acquisition_returns_in_box(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(seed=0),
        )
        for x, y in [(0.2, 0.3), (0.5, 0.9), (0.7, 0.5)]:
            bo.observe((x,), y)
        fit = bo._fit()
        x_star, acq = optimise_acquisition(
            fit, ContinuousBox(low=(0.0,), high=(1.0,)),
            acquisition=ACQ_EI, beta_t=2.0, incumbent=0.9,
            direction=MAXIMISE, rng=random.Random(0),
        )
        assert 0.0 <= x_star[0] <= 1.0
        assert math.isfinite(acq)

    def test_thompson_uses_halton_grid(self):
        bo = BayesOpt(
            domain=ContinuousBox(low=(0.0,), high=(1.0,)),
            config=BayesOptConfig(acquisition=ACQ_THOMPSON, seed=1),
        )
        for x, y in [(0.3, 0.5), (0.6, 1.0)]:
            bo.observe((x,), y)
        fit = bo._fit()
        x_star, acq = optimise_acquisition(
            fit, ContinuousBox(low=(0.0,), high=(1.0,)),
            acquisition=ACQ_THOMPSON, beta_t=2.0, incumbent=1.0,
            direction=MAXIMISE, rng=random.Random(2), thompson_grid=64,
        )
        assert 0.0 <= x_star[0] <= 1.0
        assert math.isfinite(acq)
