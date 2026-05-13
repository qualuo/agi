"""Deterministic tests for agi.experiment_design.

Where we have closed-form ground truth (discrete EIG, BALD bounds,
D-optimal designs with known optima), the tests pin to it. Where the
estimator is intrinsically Monte Carlo (nested-MC EIG, Thompson Top-K),
the tests seed the RNG and check that the estimator converges to the
closed-form reference within tolerance.
"""

import math
import random
import pytest

from agi.experiment_design import (
    BALDResult,
    BayesianBatchPlanner,
    DesignRequest,
    DOptimalDesigner,
    ExperimentCandidate,
    ExperimentDesigner,
    NestedMCEig,
    bald_score,
    eig_discrete,
    eig_nested_mc,
    entropy,
    js_divergence,
    kl_divergence,
    knowledge_gradient,
    predictive_distribution,
    thompson_top_k,
)


# ---------------------------------------------------------------------------
# Entropy & divergence sanity.
# ---------------------------------------------------------------------------


class TestEntropyDivergence:
    def test_entropy_uniform_is_log_k(self):
        for k in (2, 3, 4, 7, 16):
            assert entropy([1.0 / k] * k) == pytest.approx(math.log(k), abs=1e-12)

    def test_entropy_zero_for_point_mass(self):
        assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)
        assert entropy([0.0, 1.0]) == pytest.approx(0.0, abs=1e-12)

    def test_entropy_in_bits(self):
        # Fair coin = 1 bit.
        assert entropy([0.5, 0.5], base=2) == pytest.approx(1.0, abs=1e-12)
        # Fair 4-way = 2 bits.
        assert entropy([0.25] * 4, base=2) == pytest.approx(2.0, abs=1e-12)

    def test_kl_zero_iff_equal(self):
        p = [0.2, 0.3, 0.5]
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
        q = [0.3, 0.3, 0.4]
        assert kl_divergence(p, q) > 0.0

    def test_kl_infinite_when_q_zero_p_positive(self):
        assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")

    def test_kl_finite_when_p_zero_q_zero(self):
        # 0 * log(0/0) := 0 by convention.
        assert math.isfinite(kl_divergence([1.0, 0.0], [0.5, 0.5]))

    def test_js_symmetric_and_bounded(self):
        p, q = [0.7, 0.3], [0.1, 0.9]
        a = js_divergence(p, q)
        b = js_divergence(q, p)
        assert a == pytest.approx(b, abs=1e-12)
        assert 0.0 <= a <= math.log(2) + 1e-12

    def test_js_zero_iff_equal(self):
        p = [0.4, 0.6]
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Discrete EIG — closed-form reference.
# ---------------------------------------------------------------------------


class TestDiscreteEIG:
    def test_zero_when_likelihood_independent_of_theta(self):
        # Two θ values, but they produce the same outcome distribution.
        prior = [0.5, 0.5]
        likelihood = [[0.3, 0.7], [0.3, 0.7]]
        assert eig_discrete(prior, likelihood) == pytest.approx(0.0, abs=1e-12)

    def test_max_when_outcome_identifies_theta(self):
        # Two θ values produce deterministic, disjoint outcomes.
        # Then EIG = H[prior] = log 2 nats.
        prior = [0.5, 0.5]
        likelihood = [[1.0, 0.0], [0.0, 1.0]]
        assert eig_discrete(prior, likelihood) == pytest.approx(math.log(2), abs=1e-12)

    def test_max_with_skewed_prior(self):
        # Same fully-informative likelihood, but skewed prior.
        # EIG = H[prior] for fully informative experiments.
        prior = [0.2, 0.8]
        likelihood = [[1.0, 0.0], [0.0, 1.0]]
        eig = eig_discrete(prior, likelihood)
        ref = -(0.2 * math.log(0.2) + 0.8 * math.log(0.8))
        assert eig == pytest.approx(ref, abs=1e-12)

    def test_predictive_distribution_normalised(self):
        prior = [0.3, 0.7]
        likelihood = [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]
        py = predictive_distribution(prior, likelihood)
        assert sum(py) == pytest.approx(1.0, abs=1e-12)
        # Manual check.
        assert py[0] == pytest.approx(0.3 * 0.2 + 0.7 * 0.6)
        assert py[1] == pytest.approx(0.3 * 0.3 + 0.7 * 0.3)
        assert py[2] == pytest.approx(0.3 * 0.5 + 0.7 * 0.1)

    def test_non_negativity_random_likelihoods(self):
        rng = random.Random(42)
        for _ in range(50):
            k = rng.randint(2, 6)
            y = rng.randint(2, 8)
            prior = [rng.random() for _ in range(k)]
            z = sum(prior)
            prior = [p / z for p in prior]
            likelihood = []
            for _ in range(k):
                row = [rng.random() for _ in range(y)]
                rz = sum(row)
                likelihood.append([v / rz for v in row])
            assert eig_discrete(prior, likelihood) >= -1e-15

    def test_eig_in_bits_vs_nats(self):
        prior = [0.5, 0.5]
        likelihood = [[0.9, 0.1], [0.1, 0.9]]
        nats = eig_discrete(prior, likelihood, base=math.e)
        bits = eig_discrete(prior, likelihood, base=2)
        assert bits == pytest.approx(nats / math.log(2), abs=1e-12)


# ---------------------------------------------------------------------------
# Nested MC EIG — converges to closed form on a tractable case.
# ---------------------------------------------------------------------------


class TestNestedMC:
    def test_converges_to_discrete_reference(self):
        # Two-θ, two-y model. Reference EIG: closed form via eig_discrete.
        prior_probs = [0.5, 0.5]
        likelihood = [[0.9, 0.1], [0.2, 0.8]]
        ref = eig_discrete(prior_probs, likelihood)

        rng = random.Random(2026)

        def prior_sampler(r):
            return 0 if r.random() < prior_probs[0] else 1

        def lik_sampler(theta, r):
            row = likelihood[theta]
            return 0 if r.random() < row[0] else 1

        def log_lik(y, theta):
            return math.log(likelihood[theta][y])

        out = eig_nested_mc(
            prior_sampler, lik_sampler, log_lik, n_outer=2000, n_inner=500, rng=rng
        )
        # MC tolerance: a few SEM of the outer-loop estimator + small bias.
        assert isinstance(out, NestedMCEig)
        assert out.eig == pytest.approx(ref, abs=4.0 * out.stderr + 0.02)
        assert out.bias_bound >= 0.0
        assert out.n_outer == 2000
        assert out.n_inner == 500

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            eig_nested_mc(
                lambda r: 0,
                lambda t, r: 0,
                lambda y, t: 0.0,
                n_outer=1,
                n_inner=10,
            )
        with pytest.raises(ValueError):
            eig_nested_mc(
                lambda r: 0,
                lambda t, r: 0,
                lambda y, t: 0.0,
                n_outer=10,
                n_inner=0,
            )


# ---------------------------------------------------------------------------
# BALD — epistemic / aleatoric decomposition.
# ---------------------------------------------------------------------------


class TestBALD:
    def test_zero_when_committee_agrees(self):
        # All members predict identically → only aleatoric uncertainty.
        committee = [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]]
        res = bald_score(committee)
        assert isinstance(res, BALDResult)
        assert res.bald == pytest.approx(0.0, abs=1e-12)
        assert res.epistemic == pytest.approx(res.bald, abs=1e-15)
        assert res.aleatoric == pytest.approx(entropy([0.6, 0.4]), abs=1e-12)
        assert res.predictive_entropy == pytest.approx(res.aleatoric, abs=1e-12)

    def test_max_when_committee_disagrees_with_certainty(self):
        # Two members, each certain about a different class. Mean is uniform.
        committee = [[1.0, 0.0], [0.0, 1.0]]
        res = bald_score(committee)
        assert res.aleatoric == pytest.approx(0.0, abs=1e-12)
        assert res.predictive_entropy == pytest.approx(math.log(2), abs=1e-12)
        assert res.bald == pytest.approx(math.log(2), abs=1e-12)

    def test_bald_bounded_above(self):
        rng = random.Random(7)
        for _ in range(40):
            m = rng.randint(2, 6)
            y = rng.randint(2, 5)
            committee = []
            for _ in range(m):
                row = [rng.random() for _ in range(y)]
                z = sum(row)
                committee.append([v / z for v in row])
            res = bald_score(committee)
            assert -1e-12 <= res.bald <= math.log(min(m, y)) + 1e-10

    def test_inconsistent_lengths_raise(self):
        with pytest.raises(ValueError):
            bald_score([[0.5, 0.5], [0.3, 0.3, 0.4]])
        with pytest.raises(ValueError):
            bald_score([])


# ---------------------------------------------------------------------------
# Knowledge gradient.
# ---------------------------------------------------------------------------


class TestKnowledgeGradient:
    def test_kg_zero_for_dominant_arm_with_no_uncertainty(self):
        # Arm 0 is clearly best and has zero posterior variance —
        # learning anything about any arm cannot change argmax.
        kg = knowledge_gradient(means=[10.0, 1.0, 0.5], stds=[0.0, 0.0, 0.0], obs_noise_std=[1.0, 1.0, 1.0])
        for v in kg:
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_kg_largest_for_close_arms_with_high_variance(self):
        # Two close arms with high posterior variance; the wider-σ arm
        # should have the higher KG because sampling it most changes the
        # posterior on argmax.
        kg = knowledge_gradient(
            means=[1.0, 0.95], stds=[0.3, 0.3], obs_noise_std=[0.1, 0.1]
        )
        # Both KG should be positive (both could flip argmax).
        assert all(v >= 0.0 for v in kg)
        assert sum(kg) > 0.0

    def test_kg_independent_of_argmax_dominance_when_noise_huge(self):
        # If observation noise dwarfs prior variance, no update — KG ~ 0.
        kg = knowledge_gradient(
            means=[1.0, 0.99], stds=[0.1, 0.1], obs_noise_std=[100.0, 100.0]
        )
        for v in kg:
            assert v < 1e-3

    def test_kg_3_arms_symmetric(self):
        # Three identical arms: KG should be equal across them.
        kg = knowledge_gradient(
            means=[1.0, 1.0, 1.0], stds=[1.0, 1.0, 1.0], obs_noise_std=[1.0, 1.0, 1.0]
        )
        assert kg[0] == pytest.approx(kg[1], abs=1e-9)
        assert kg[1] == pytest.approx(kg[2], abs=1e-9)
        assert kg[0] > 0.0


# ---------------------------------------------------------------------------
# Thompson Top-K.
# ---------------------------------------------------------------------------


class TestThompsonTopK:
    def test_picks_top_k_when_means_dominate(self):
        # Arms with near-deterministic posteriors: top-k should be a
        # function of the means.
        means = [1.0, 5.0, 3.0, 9.0, 7.0]
        samplers = [
            (lambda m=m: (lambda r: r.gauss(m, 0.001)))() for m in means
        ]
        rng = random.Random(1)
        top3 = thompson_top_k(samplers, k=3, rng=rng)
        # Highest three means are arms 3, 4, 1 (values 9, 7, 5).
        assert sorted(top3) == [1, 3, 4]

    def test_full_set_when_k_equals_n(self):
        samplers = [(lambda r: r.random()) for _ in range(4)]
        out = thompson_top_k(samplers, k=4, rng=random.Random(0))
        assert sorted(out) == [0, 1, 2, 3]

    def test_rejects_k_too_big(self):
        with pytest.raises(ValueError):
            thompson_top_k([(lambda r: r.random())], k=2)


# ---------------------------------------------------------------------------
# BayesianBatchPlanner — submodular greedy.
# ---------------------------------------------------------------------------


class TestBatchPlanner:
    def test_picks_top_eig_for_equal_cost_size_k(self):
        cands = [
            ExperimentCandidate(id=f"t{i}", eig=eig)
            for i, eig in enumerate([0.1, 0.4, 0.05, 0.3, 0.2])
        ]
        planner = BayesianBatchPlanner()
        plan = planner.plan(cands, k=3)
        assert sorted(plan.selected) == ["t1", "t3", "t4"]
        assert plan.total_eig == pytest.approx(0.4 + 0.3 + 0.2)
        assert plan.eig_per == [0.4, 0.3, 0.2]
        assert plan.total_cost == pytest.approx(3.0)

    def test_knapsack_uses_density(self):
        # Tickets:
        #   A: eig=1.0, cost=1.0  → density 1.0
        #   B: eig=2.0, cost=3.0  → density 0.67
        #   C: eig=1.4, cost=1.0  → density 1.4
        # Budget=3: greedy density picks C, A → eig 2.4, cost 2.
        # Then B would push to cost 5 (over budget). So final {A, C}.
        cands = [
            ExperimentCandidate(id="A", eig=1.0, cost=1.0),
            ExperimentCandidate(id="B", eig=2.0, cost=3.0),
            ExperimentCandidate(id="C", eig=1.4, cost=1.0),
        ]
        plan = BayesianBatchPlanner().plan(cands, budget=3.0)
        assert sorted(plan.selected) == ["A", "C"]
        assert plan.total_cost == pytest.approx(2.0)
        assert plan.total_eig == pytest.approx(2.4)

    def test_correlated_marginal_yields_diminishing_returns(self):
        # Three correlated candidates; each subsequent one is worth half
        # as much given any previously selected one.
        def marginal(selected, cand):
            return cand.eig / (1 << len(selected))

        cands = [
            ExperimentCandidate(id="X", eig=1.0),
            ExperimentCandidate(id="Y", eig=1.0),
            ExperimentCandidate(id="Z", eig=1.0),
        ]
        plan = BayesianBatchPlanner(eig_marginal=marginal).plan(cands, k=3)
        # First pick at gain 1.0, second at 0.5, third at 0.25.
        assert plan.eig_per[0] == pytest.approx(1.0)
        assert plan.eig_per[1] == pytest.approx(0.5)
        assert plan.eig_per[2] == pytest.approx(0.25)
        assert plan.total_eig == pytest.approx(1.75)
        # Lazy greedy should still call marginal more than the strict
        # minimum (N + k = 6) but well under naive N·k = 9.
        assert plan.n_evaluations >= 3

    def test_skips_non_monotone_candidates(self):
        # A candidate that turns negative after the first pick must
        # not be selected (planner would otherwise destroy the
        # monotonicity guarantee).
        def marginal(selected, cand):
            if not selected:
                return cand.eig
            if cand.id == "B":
                return -1.0
            return cand.eig * 0.5

        cands = [
            ExperimentCandidate(id="A", eig=1.0),
            ExperimentCandidate(id="B", eig=0.9),
            ExperimentCandidate(id="C", eig=0.8),
        ]
        plan = BayesianBatchPlanner(eig_marginal=marginal).plan(cands, k=3)
        # A picked first (highest); B excluded (negative marginal);
        # C added at 0.4. Some inputs may run out of budget on B and
        # never accept it.
        assert "B" not in plan.selected
        assert "A" in plan.selected
        assert "C" in plan.selected

    def test_empty_pool(self):
        plan = BayesianBatchPlanner().plan([], k=3)
        assert plan.selected == []
        assert plan.total_eig == 0.0

    def test_requires_constraint(self):
        with pytest.raises(ValueError):
            BayesianBatchPlanner().plan(
                [ExperimentCandidate(id="a", eig=1.0)]
            )

    def test_stops_on_zero_marginal_gain(self):
        # All later candidates have zero marginal gain — planner should
        # stop early instead of padding the batch with zeros.
        def marginal(selected, cand):
            return 1.0 if not selected else 0.0

        cands = [ExperimentCandidate(id=f"t{i}", eig=1.0) for i in range(5)]
        plan = BayesianBatchPlanner(eig_marginal=marginal).plan(cands, k=5)
        assert len(plan.selected) == 1


# ---------------------------------------------------------------------------
# D-optimal designer — known optimum on a small problem.
# ---------------------------------------------------------------------------


class TestDOptimal:
    def test_recovers_known_optimum_2d(self):
        # 2D candidate set: rows [1, x_i] for x in {-1, -0.5, 0, 0.5, 1}.
        # The D-optimal 2-design is {-1, 1} (extremes maximise the
        # variance of x → maximise det of X^T X).
        rows = [[1.0, x] for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        rng = random.Random(0)
        designer = DOptimalDesigner(rows, criterion="D", rng=rng)
        res = designer.select(k=2)
        # Selected x-values should be the extremes.
        xs = sorted(rows[i][1] for i in res.selected)
        assert xs == [-1.0, 1.0]
        assert res.criterion == "D"

    def test_a_optimal_returns_smaller_trace_than_random(self):
        rows = [
            [1.0, math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)]
            for i in range(12)
        ]
        designer = DOptimalDesigner(rows, criterion="A", rng=random.Random(3))
        res = designer.select(k=6)
        # Trace of the A-criterion at the optimum should be < trace at
        # a sequential subset.
        seq_designer = DOptimalDesigner(rows, criterion="A", rng=random.Random(3))
        seq_score = seq_designer._score(seq_designer._info_matrix([0, 1, 2, 3, 4, 5]))
        assert res.criterion_value <= seq_score + 1e-9

    def test_e_optimal_returns_smaller_max_eigenvalue_than_random(self):
        rows = [
            [1.0, math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)]
            for i in range(8)
        ]
        designer = DOptimalDesigner(rows, criterion="E", rng=random.Random(4))
        res = designer.select(k=4)
        seq_score = designer._score(designer._info_matrix([0, 1, 2, 3]))
        assert res.criterion_value <= seq_score + 1e-9

    def test_rejects_k_below_p(self):
        rows = [[1.0, x] for x in (-1.0, 0.0, 1.0)]
        designer = DOptimalDesigner(rows)
        with pytest.raises(ValueError):
            designer.select(k=1)  # k < p

    def test_rejects_k_above_n(self):
        rows = [[1.0, x] for x in (-1.0, 0.0, 1.0)]
        designer = DOptimalDesigner(rows)
        with pytest.raises(ValueError):
            designer.select(k=4)

    def test_rejects_bad_criterion(self):
        with pytest.raises(ValueError):
            DOptimalDesigner([[1.0, 0.0]], criterion="W")

    def test_rejects_inconsistent_pool(self):
        with pytest.raises(ValueError):
            DOptimalDesigner([[1.0, 0.0], [1.0, 0.0, 0.0]])

    def test_rejects_empty_pool(self):
        with pytest.raises(ValueError):
            DOptimalDesigner([])


# ---------------------------------------------------------------------------
# Top-level ExperimentDesigner surface.
# ---------------------------------------------------------------------------


class TestDesignerSurface:
    def test_designer_routes_to_planner(self):
        cands = [
            ExperimentCandidate(id="a", eig=0.5, cost=1.0),
            ExperimentCandidate(id="b", eig=0.3, cost=1.0),
            ExperimentCandidate(id="c", eig=0.7, cost=1.0),
        ]
        designer = ExperimentDesigner()
        resp = designer.design(DesignRequest(candidates=cands, k=2))
        assert sorted(resp.plan.selected) == ["a", "c"]
        assert resp.plan.total_eig == pytest.approx(1.2)
        assert resp.eig_per_dollar == pytest.approx(1.2 / 2.0)
        assert resp.binding_constraint == "k"

    def test_designer_filters_by_min_eig_per(self):
        cands = [
            ExperimentCandidate(id="a", eig=0.01),
            ExperimentCandidate(id="b", eig=0.5),
            ExperimentCandidate(id="c", eig=0.001),
        ]
        designer = ExperimentDesigner()
        resp = designer.design(
            DesignRequest(candidates=cands, k=3, min_eig_per=0.05)
        )
        assert resp.plan.selected == ["b"]

    def test_designer_binding_constraint_budget(self):
        cands = [
            ExperimentCandidate(id="a", eig=2.0, cost=3.0),
            ExperimentCandidate(id="b", eig=1.0, cost=3.0),
        ]
        designer = ExperimentDesigner()
        resp = designer.design(DesignRequest(candidates=cands, budget=3.0))
        assert resp.plan.selected == ["a"]
        assert resp.binding_constraint == "budget"

    def test_designer_empty_pool(self):
        designer = ExperimentDesigner()
        resp = designer.design(DesignRequest(candidates=[], k=3))
        assert resp.plan.selected == []
        assert resp.eig_per_dollar == 0.0
        assert resp.binding_constraint == "min_eig_per"

    def test_designer_score_bald(self):
        res = ExperimentDesigner.score_bald([[1.0, 0.0], [0.0, 1.0]])
        assert res.bald == pytest.approx(math.log(2), abs=1e-12)

    def test_designer_score_kg(self):
        kg = ExperimentDesigner.score_knowledge_gradient(
            means=[1.0, 1.0], stds=[1.0, 1.0], obs_noise_std=[1.0, 1.0]
        )
        assert len(kg) == 2
        assert kg[0] > 0.0
        assert kg[1] > 0.0

    def test_designer_correlated_routing(self):
        # When the caller supplies a correlated callback but doesn't set
        # `correlated=True`, the planner uses the independent default.
        def m(selected, cand):
            return cand.eig if not selected else cand.eig * 0.1

        cands = [
            ExperimentCandidate(id="a", eig=1.0),
            ExperimentCandidate(id="b", eig=1.0),
        ]
        designer = ExperimentDesigner(eig_marginal=m)
        resp_indep = designer.design(DesignRequest(candidates=cands, k=2))
        # Without correlated=True, both get full EIG.
        assert resp_indep.plan.total_eig == pytest.approx(2.0)
        resp_corr = designer.design(
            DesignRequest(candidates=cands, k=2, correlated=True)
        )
        # With correlated=True, the second is discounted.
        assert resp_corr.plan.total_eig == pytest.approx(1.1)
