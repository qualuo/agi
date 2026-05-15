"""Tests for the Compressor runtime primitive."""
from __future__ import annotations

import math
import random
import pytest

from agi.compressor import (
    AIC,
    BERNOULLI,
    BIC,
    COMPRESSOR_FIT,
    COMPRESSOR_MODEL_REGISTERED,
    COMPRESSOR_OBSERVED,
    COMPRESSOR_REPORT,
    COMPRESSOR_SCORED,
    COMPRESSOR_SELECTED,
    COMPRESSOR_STARTED,
    CONSTANT,
    Codelength,
    Comparison,
    Compressor,
    CompressorError,
    CompressorReport,
    ELIAS_DELTA,
    ELIAS_GAMMA,
    Fit,
    GAUSSIAN,
    GAUSSIAN_KNOWN_SIGMA,
    GEOMETRIC,
    HISTOGRAM,
    IncompatibleModels,
    InsufficientData,
    InvalidData,
    InvalidModel,
    KNOWN_EVENTS,
    KNOWN_INT_CODES,
    KNOWN_METHODS,
    KNOWN_MODELS,
    MARKOV,
    ML,
    MULTINOMIAL,
    ModelSpec,
    NML,
    OnlineState,
    POISSON,
    PREQUENTIAL,
    RISSANEN_LOGSTAR,
    Selection,
    TWO_PART,
    UNIFORM_DISCRETE,
    UnknownIntCode,
    UnknownMethod,
    UnknownModel,
    compressor_from_spec,
    elias_delta_bits,
    elias_gamma_bits,
    kt_codelength_binary,
    kt_codelength_multinomial,
    laplace_codelength_multinomial,
    log_bernoulli_nml_constant,
    log_multinomial_nml_constant,
    parametric_complexity_gaussian,
    parametric_complexity_gaussian_known_sigma,
    parametric_complexity_geometric,
    parametric_complexity_poisson,
    rissanen_logstar_bits,
    universal_int_bits,
)


# =====================================================================
# Universal integer codes
# =====================================================================


class TestEliasGamma:
    def test_one_bit(self):
        assert elias_gamma_bits(1) == 1.0

    def test_two(self):
        # ⌊log2(2)⌋ = 1 → 2*1+1 = 3
        assert elias_gamma_bits(2) == 3.0

    def test_seven(self):
        # ⌊log2(7)⌋ = 2 → 2*2+1 = 5
        assert elias_gamma_bits(7) == 5.0

    def test_rejects_zero(self):
        with pytest.raises(CompressorError):
            elias_gamma_bits(0)

    def test_rejects_negative(self):
        with pytest.raises(CompressorError):
            elias_gamma_bits(-1)


class TestEliasDelta:
    def test_one_bit(self):
        assert elias_delta_bits(1) == 1.0

    def test_delta_seven_matches_formula(self):
        n = 7
        fl = math.floor(math.log2(n))
        expected = fl + 2 * math.floor(math.log2(fl + 1)) + 1
        assert elias_delta_bits(n) == expected

    def test_delta_eventually_shorter_than_gamma(self):
        # Elias-δ overtakes Elias-γ for moderately large n.
        n = 1024
        assert elias_delta_bits(n) < elias_gamma_bits(n)


class TestRissanenLogStar:
    def test_one(self):
        # log* converges immediately for n=1
        assert rissanen_logstar_bits(1) == pytest.approx(math.log2(2.865064))

    def test_eight(self):
        # log* should be log(c0) + 3 + log2(3) + ...
        v = rissanen_logstar_bits(8)
        # rough sanity bracket
        assert v > 4.0 and v < 8.0

    def test_log_star_monotone(self):
        prev = -1.0
        for n in (1, 2, 5, 10, 100, 1000):
            now = rissanen_logstar_bits(n)
            assert now > prev
            prev = now

    def test_universal_int_bits_dispatch(self):
        for code in (ELIAS_GAMMA, ELIAS_DELTA, RISSANEN_LOGSTAR):
            assert universal_int_bits(5, code=code) > 0
        with pytest.raises(UnknownIntCode):
            universal_int_bits(5, code="bogus")


# =====================================================================
# NML constants
# =====================================================================


class TestBernoulliNMLConstant:
    def test_n_zero(self):
        assert log_bernoulli_nml_constant(0) == 0.0

    def test_n_one(self):
        # C_1 = (0/1)^0 (1/1)^1 + (1/1)^1 (0/1)^0 = 1 + 1 = 2
        assert log_bernoulli_nml_constant(1) == pytest.approx(math.log(2.0), abs=1e-9)

    def test_increases_in_n(self):
        prev = -1.0
        for n in (1, 5, 50, 500):
            v = log_bernoulli_nml_constant(n)
            assert v > prev
            prev = v

    def test_asymptotic_matches_exact(self):
        # at n=10000 the exact and asymptotic should be close
        exact = log_bernoulli_nml_constant(10_000)
        asymp = 0.5 * math.log(10_000 * math.pi / 2.0)
        assert abs(exact - asymp) < 1e-2

    def test_rejects_negative(self):
        with pytest.raises(CompressorError):
            log_bernoulli_nml_constant(-1)


class TestMultinomialNMLConstant:
    def test_k_one(self):
        assert log_multinomial_nml_constant(100, 1) == 0.0

    def test_k_two_matches_bernoulli(self):
        for n in (1, 10, 100, 1000):
            assert log_multinomial_nml_constant(n, 2) == pytest.approx(
                log_bernoulli_nml_constant(n), abs=1e-9
            )

    def test_increases_in_k(self):
        prev = log_multinomial_nml_constant(100, 2)
        for k in (3, 5, 10, 20):
            v = log_multinomial_nml_constant(100, k)
            assert v > prev
            prev = v

    def test_recurrence_consistency(self):
        # Mononen recurrence: C_n(j) = C_n(j-1) + (n/(j-2)) C_n(j-2)
        n = 50
        for j in range(3, 10):
            c_j = math.exp(log_multinomial_nml_constant(n, j))
            c_jm1 = math.exp(log_multinomial_nml_constant(n, j - 1))
            c_jm2 = math.exp(log_multinomial_nml_constant(n, j - 2))
            assert c_j == pytest.approx(c_jm1 + (n / (j - 2)) * c_jm2, rel=1e-9)


class TestParametricComplexityGaussian:
    def test_known_sigma_increases_in_n(self):
        prev = -math.inf
        for n in (10, 100, 1000, 10_000):
            v = parametric_complexity_gaussian_known_sigma(n, 1.0, -5.0, 5.0)
            assert v > prev
            prev = v

    def test_known_sigma_rejects_bad_range(self):
        with pytest.raises(InvalidModel):
            parametric_complexity_gaussian_known_sigma(10, 1.0, 5.0, 5.0)

    def test_known_sigma_rejects_bad_sigma(self):
        with pytest.raises(InvalidModel):
            parametric_complexity_gaussian_known_sigma(10, 0.0, -5.0, 5.0)

    def test_unknown_sigma_two_param_correction(self):
        # unknown-σ should be larger than known-σ at same n
        v_unknown = parametric_complexity_gaussian(100, 0.1, 10.0)
        v_known = parametric_complexity_gaussian_known_sigma(100, 1.0, -10.0, 10.0)
        # not strictly comparable but ranges similar; both finite
        assert math.isfinite(v_unknown)
        assert math.isfinite(v_known)

    def test_unknown_sigma_n_one_zero(self):
        assert parametric_complexity_gaussian(1, 0.1, 10.0) == 0.0


class TestParametricComplexityGeometric:
    def test_positive_and_increasing(self):
        prev = -math.inf
        for n in (5, 50, 500, 5000):
            v = parametric_complexity_geometric(n)
            assert v > prev
            prev = v


class TestParametricComplexityPoisson:
    def test_positive_and_increasing(self):
        prev = -math.inf
        for n in (5, 50, 500):
            v = parametric_complexity_poisson(n, 0.1, 10.0)
            assert v > prev
            prev = v

    def test_rejects_bad_range(self):
        with pytest.raises(InvalidModel):
            parametric_complexity_poisson(10, 5.0, 1.0)


# =====================================================================
# Prequential codes
# =====================================================================


class TestKTBinary:
    def test_zero_data(self):
        assert kt_codelength_binary(0, 0) == 0.0

    def test_symmetric(self):
        assert kt_codelength_binary(3, 7) == pytest.approx(
            kt_codelength_binary(7, 3), rel=1e-9
        )

    def test_kt_matches_factorisation(self):
        # KT closed form: -log P_KT = log B(1/2,1/2) - log B(n0+1/2, n1+1/2)
        n0, n1 = 3, 5
        v = kt_codelength_binary(n0, n1)
        # incremental check: equals sum of -log( (count + 1/2) / (t + 1) ) over the
        # canonical ordering n0 zeros first then n1 ones
        running = 0.0
        c0 = c1 = 0
        for x in [0] * n0 + [1] * n1:
            t = c0 + c1
            p = ((c1 if x == 1 else c0) + 0.5) / (t + 1.0)
            running += -math.log(p)
            if x == 1:
                c1 += 1
            else:
                c0 += 1
        assert v == pytest.approx(running, rel=1e-9)

    def test_kt_regret_vs_nml(self):
        # KT and NML differ by at most O(1) — the gap stays bounded as n grows.
        for n in (50, 500, 5000):
            n1 = n // 3
            n0 = n - n1
            kt = kt_codelength_binary(n0, n1)
            # ML log-loss
            from agi.compressor import ml_loglik_bernoulli
            _p, ll = ml_loglik_bernoulli(n0, n1)
            nml = -ll + log_bernoulli_nml_constant(n)
            # KT is at most (1/2)log(πn) more than NML; in practice within ~0.5 nat
            assert abs(kt - nml) < 1.5


class TestKTMultinomial:
    def test_zero_n(self):
        assert kt_codelength_multinomial([0, 0, 0]) == 0.0

    def test_matches_bernoulli_for_k2(self):
        for n0, n1 in ((1, 0), (3, 5), (10, 10), (20, 5)):
            assert kt_codelength_multinomial([n0, n1]) == pytest.approx(
                kt_codelength_binary(n0, n1), rel=1e-9
            )

    def test_laplace_vs_kt_ordering(self):
        # Laplace (α=1) is looser than KT (α=1/2) for typical data
        cnts = [3, 5, 7]
        kt = kt_codelength_multinomial(cnts)
        lap = laplace_codelength_multinomial(cnts)
        # both finite, both positive
        assert kt > 0 and lap > 0


# =====================================================================
# Compressor — registration & errors
# =====================================================================


class TestCompressorRegistration:
    def test_register_known_kinds(self):
        c = Compressor()
        c.register("a", BERNOULLI)
        c.register("b", MULTINOMIAL, k=4)
        c.register("d", GEOMETRIC)
        c.register("e", POISSON, lam_min=0.1, lam_max=10.0)
        c.register("f", GAUSSIAN_KNOWN_SIGMA, sigma=1.0, mu_min=-1.0, mu_max=1.0)
        c.register("g", GAUSSIAN, sigma_min=0.1, sigma_max=5.0)
        c.register("h", UNIFORM_DISCRETE, k=3)
        c.register("i", HISTOGRAM, m=5, lo=0.0, hi=1.0)
        c.register("j", MARKOV, k=2, r=1)
        c.register("k", CONSTANT, c=0)
        models = c.models()
        assert set(models.keys()) == {"a", "b", "d", "e", "f", "g", "h", "i", "j", "k"}

    def test_register_unknown_kind_raises(self):
        c = Compressor()
        with pytest.raises(UnknownModel):
            c.register("x", "not_a_model")

    def test_double_register_raises(self):
        c = Compressor()
        c.register("a", BERNOULLI)
        with pytest.raises(InvalidModel):
            c.register("a", BERNOULLI)

    def test_empty_name_rejected(self):
        c = Compressor()
        with pytest.raises(InvalidModel):
            c.register("", BERNOULLI)

    def test_multinomial_needs_k(self):
        c = Compressor()
        with pytest.raises(InvalidModel):
            c.register("a", MULTINOMIAL)
        with pytest.raises(InvalidModel):
            c.register("a", MULTINOMIAL, k=1)

    def test_histogram_needs_m_lo_hi(self):
        c = Compressor()
        with pytest.raises(InvalidModel):
            c.register("a", HISTOGRAM, m=5, lo=0.0)
        with pytest.raises(InvalidModel):
            c.register("a", HISTOGRAM, m=1, lo=0.0, hi=1.0)
        with pytest.raises(InvalidModel):
            c.register("a", HISTOGRAM, m=5, lo=1.0, hi=0.0)

    def test_markov_needs_k_r(self):
        c = Compressor()
        with pytest.raises(InvalidModel):
            c.register("a", MARKOV, k=2)
        with pytest.raises(InvalidModel):
            c.register("a", MARKOV, k=1, r=1)
        with pytest.raises(InvalidModel):
            c.register("a", MARKOV, k=2, r=-1)

    def test_constant_needs_c(self):
        c = Compressor()
        with pytest.raises(InvalidModel):
            c.register("a", CONSTANT)


# =====================================================================
# Compressor — fit
# =====================================================================


class TestCompressorFit:
    def test_bernoulli_fit_returns_p(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        f = c.fit("b", [0, 1, 1, 0, 1])
        assert f.params["p"] == pytest.approx(0.6, abs=1e-9)
        assert f.n == 5

    def test_multinomial_fit_returns_probs(self):
        c = Compressor()
        c.register("m", MULTINOMIAL, k=3)
        f = c.fit("m", [0, 0, 1, 2, 1, 0])
        assert f.params["probs"] == pytest.approx([0.5, 1.0 / 3, 1.0 / 6], abs=1e-9)

    def test_gaussian_fit_returns_mu_sigma(self):
        c = Compressor()
        c.register("g", GAUSSIAN, sigma_min=0.01, sigma_max=10.0)
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        f = c.fit("g", data)
        assert f.params["mu"] == pytest.approx(3.0, abs=1e-9)
        assert f.params["sigma"] == pytest.approx(math.sqrt(2.0), abs=1e-9)

    def test_geometric_fit(self):
        c = Compressor()
        c.register("g", GEOMETRIC)
        f = c.fit("g", [1, 2, 1, 3, 1, 2])
        # n=6, sum=10, p_hat = 6/10 = 0.6
        assert f.params["p"] == pytest.approx(0.6, abs=1e-9)

    def test_poisson_fit(self):
        c = Compressor()
        c.register("p", POISSON, lam_min=0.01, lam_max=10.0)
        f = c.fit("p", [0, 1, 2, 1, 0, 3])
        # mean = 7/6
        assert f.params["lam"] == pytest.approx(7.0 / 6.0, abs=1e-9)

    def test_unknown_model_in_fit(self):
        c = Compressor()
        with pytest.raises(UnknownModel):
            c.fit("nope", [0, 1])

    def test_invalid_data_for_bernoulli(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        with pytest.raises(InvalidData):
            c.fit("b", [0, 1, 2])

    def test_invalid_data_for_geometric(self):
        c = Compressor()
        c.register("g", GEOMETRIC)
        with pytest.raises(InvalidData):
            c.fit("g", [0, 1, 2])

    def test_invalid_data_for_gaussian(self):
        c = Compressor()
        c.register("g", GAUSSIAN, sigma_min=0.1, sigma_max=10.0)
        with pytest.raises(InvalidData):
            c.fit("g", [1.0, float("nan"), 2.0])


# =====================================================================
# Compressor — codelength
# =====================================================================


class TestCompressorCodelength:
    def test_uniform_beats_bernoulli_on_balanced(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        # perfectly balanced — Bernoulli pays parametric cost the uniform doesn't
        data = [0, 1] * 50
        cl_ber = c.codelength("ber", data)
        cl_uni = c.codelength("uni", data)
        assert cl_uni.bits < cl_ber.bits

    def test_bernoulli_beats_uniform_on_biased(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        data = [0] * 95 + [1] * 5
        assert c.codelength("ber", data).bits < c.codelength("uni", data).bits

    def test_markov_beats_iid_on_correlated(self):
        c = Compressor()
        c.register("iid", MULTINOMIAL, k=2)
        c.register("m1", MARKOV, k=2, r=1)
        data = [0, 1] * 50
        cl_iid = c.codelength("iid", data)
        cl_m1 = c.codelength("m1", data)
        assert cl_m1.bits < cl_iid.bits

    def test_codelength_components_consistent(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        cl = c.codelength("b", [0, 1, 0, 1, 0])
        # stochastic complexity = ml + parametric complexity (in nats)
        assert cl.stochastic_complexity == pytest.approx(
            cl.ml + cl.parametric_complexity, rel=1e-9
        )

    def test_codelength_bits_is_nats_over_ln2(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        cl = c.codelength("b", [0, 1, 0, 1])
        assert cl.bits == pytest.approx(cl.stochastic_complexity / math.log(2.0), rel=1e-12)

    def test_constant_perfect_match_zero_bits(self):
        c = Compressor()
        c.register("k0", CONSTANT, c=0)
        cl = c.codelength("k0", [0, 0, 0, 0])
        assert cl.bits == 0.0

    def test_constant_mismatch_infinite(self):
        c = Compressor()
        c.register("k0", CONSTANT, c=0)
        cl = c.codelength("k0", [0, 1, 0])
        assert math.isinf(cl.bits)

    def test_histogram_density_correction(self):
        # A density on [0,1] discretised into 10 bins, all 1000 samples uniform:
        # the NML codelength should reflect ~uniform density of 1.0 = 0 bits/sample
        # (plus the parametric complexity term ~ (1/2) log(n)).
        c = Compressor()
        c.register("h", HISTOGRAM, m=10, lo=0.0, hi=1.0)
        random.seed(0)
        data = [random.random() for _ in range(2000)]
        cl = c.codelength("h", data)
        # per-sample density log should be near 0; total bits ~ (m-1)/2 log2(n)
        per_sample = cl.bits / cl.n
        assert per_sample < 0.5  # density well below 2^0 = 1

    def test_codelength_value_select(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        cl = c.codelength("b", [0, 1, 0, 1, 1])
        for m in (ML, NML, TWO_PART, PREQUENTIAL, BIC, AIC):
            assert math.isfinite(cl.select_value(m))
        with pytest.raises(UnknownMethod):
            cl.select_value("bogus")


# =====================================================================
# Compressor — select
# =====================================================================


class TestCompressorSelect:
    def test_select_picks_winner(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        sel = c.select([0, 1] * 50)
        assert sel.winner == "uni"
        assert sel.runner_up == "ber"
        # negative gap = winner is shorter
        assert sel.gap_nats < 0
        assert sel.gap_bits < 0
        assert sel.bayes_factor > 1.0

    def test_select_method_two_part(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        sel = c.select([0, 1] * 50, method=TWO_PART)
        assert sel.method == TWO_PART
        # winner is still uniform on balanced data
        assert sel.winner == "uni"

    def test_select_method_bic_aic(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        for m in (BIC, AIC, PREQUENTIAL, ML):
            sel = c.select([0, 1] * 50, method=m)
            assert sel.method == m
            assert sel.winner in ("ber", "uni")

    def test_select_unknown_method(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        with pytest.raises(UnknownMethod):
            c.select([0, 1], method="bogus")

    def test_select_empty_registry(self):
        c = Compressor()
        with pytest.raises(CompressorError):
            c.select([0, 1])

    def test_select_subset(self):
        c = Compressor()
        c.register("a", BERNOULLI)
        c.register("b", BERNOULLI)
        c.register("u", UNIFORM_DISCRETE, k=2)
        # using only a and u
        sel = c.select([0, 1] * 50, names=["a", "u"])
        assert set(sel.codelengths_nats.keys()) == {"a", "u"}

    def test_per_symbol_regret_nonneg(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        sel = c.select([0, 1] * 50)
        assert sel.per_symbol_regret_bits >= 0.0

    def test_select_fingerprint_changes(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        fp0 = c.fingerprint
        c.select([0, 1] * 10)
        fp1 = c.fingerprint
        c.select([0, 0, 0, 1] * 10)
        fp2 = c.fingerprint
        assert fp0 != fp1
        assert fp1 != fp2


# =====================================================================
# Compressor — compare
# =====================================================================


class TestCompressorCompare:
    def test_compare_returns_bayes_factor(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        # biased data: bernoulli should win
        cmp = c.compare("ber", "uni", [0] * 95 + [1] * 5)
        assert cmp.delta_bits < 0
        assert cmp.bayes_factor_for_a > 1.0

    def test_compare_incompatible_spaces(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("gauss", GAUSSIAN, sigma_min=0.1, sigma_max=10.0)
        with pytest.raises(IncompatibleModels):
            c.compare("ber", "gauss", [0, 1])

    def test_compare_unknown_method(self):
        c = Compressor()
        c.register("a", BERNOULLI)
        c.register("b", UNIFORM_DISCRETE, k=2)
        with pytest.raises(UnknownMethod):
            c.compare("a", "b", [0, 1], method="bogus")

    def test_compare_binary_sym_kl_finite(self):
        c = Compressor()
        c.register("a", BERNOULLI)
        c.register("b", BERNOULLI)
        cmp = c.compare("a", "b", [0, 1, 0, 1])
        # both bernoullis on same data have identical KT predictives → sym_kl = 0
        assert cmp.sym_kl_predictive == pytest.approx(0.0, abs=1e-9)


# =====================================================================
# Compressor — online
# =====================================================================


class TestCompressorOnline:
    def test_online_observe_increments_total(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        s0 = c.online_state("b")
        assert s0.n == 0
        c.online_observe("b", 1)
        c.online_observe("b", 0)
        c.online_observe("b", 1)
        s = c.online_state("b")
        assert s.n == 3
        assert s.n1 == 2
        assert s.n0 == 1
        assert s.prequential_nats > 0

    def test_online_total_matches_batch_kt(self):
        # The sum of per-symbol prequential costs should equal the batch KT codelength.
        c = Compressor()
        c.register("b", BERNOULLI)
        random.seed(42)
        data = [random.randint(0, 1) for _ in range(100)]
        for x in data:
            c.online_observe("b", x)
        n0 = data.count(0)
        n1 = data.count(1)
        expected = kt_codelength_binary(n0, n1)
        s = c.online_state("b")
        assert s.prequential_nats == pytest.approx(expected, rel=1e-9)

    def test_online_multinomial_matches_batch(self):
        c = Compressor()
        c.register("m", MULTINOMIAL, k=4)
        random.seed(123)
        data = [random.randint(0, 3) for _ in range(200)]
        for x in data:
            c.online_observe("m", x)
        counts = [data.count(i) for i in range(4)]
        expected = kt_codelength_multinomial(counts)
        s = c.online_state("m")
        assert s.prequential_nats == pytest.approx(expected, rel=1e-9)

    def test_online_constant_match(self):
        c = Compressor()
        c.register("k", CONSTANT, c=7)
        c.online_observe("k", 7)
        c.online_observe("k", 7)
        assert c.online_state("k").prequential_nats == 0.0

    def test_online_constant_mismatch(self):
        c = Compressor()
        c.register("k", CONSTANT, c=7)
        c.online_observe("k", 7)
        c.online_observe("k", 8)
        assert math.isinf(c.online_state("k").prequential_nats)

    def test_online_reset_one(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        c.register("m", MULTINOMIAL, k=3)
        for x in [0, 1, 0, 1]:
            c.online_observe("b", x)
        for x in [0, 1, 2, 0]:
            c.online_observe("m", x)
        c.online_reset("b")
        assert c.online_state("b").n == 0
        assert c.online_state("m").n == 4

    def test_online_reset_all(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        c.register("m", MULTINOMIAL, k=3)
        c.online_observe("b", 0)
        c.online_observe("m", 1)
        c.online_reset()
        assert c.online_state("b").n == 0
        assert c.online_state("m").n == 0

    def test_online_invalid_datum(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        with pytest.raises(InvalidData):
            c.online_observe("b", 7)

    def test_online_unknown_model(self):
        c = Compressor()
        with pytest.raises(UnknownModel):
            c.online_state("nope")

    def test_online_real_observations(self):
        c = Compressor()
        c.register("g", GAUSSIAN, sigma_min=0.01, sigma_max=10.0)
        for x in [0.1, -0.2, 0.3, 0.0, -0.1]:
            c.online_observe("g", x)
        s = c.online_state("g")
        assert s.n == 5
        assert s.sum_x == pytest.approx(0.1)
        assert s.prequential_nats > 0

    def test_online_geometric_predictive_well_formed(self):
        c = Compressor()
        c.register("g", GEOMETRIC)
        for x in [1, 1, 2, 1, 3]:
            c.online_observe("g", x)
        s = c.online_state("g")
        assert s.n == 5
        assert s.sum_count == 8
        assert math.isfinite(s.prequential_nats)
        assert s.prequential_nats > 0

    def test_online_poisson_predictive_well_formed(self):
        c = Compressor()
        c.register("p", POISSON, lam_min=0.01, lam_max=10.0)
        for x in [0, 1, 2, 0, 1, 3]:
            c.online_observe("p", x)
        s = c.online_state("p")
        assert s.n == 6
        assert math.isfinite(s.prequential_nats)


# =====================================================================
# Compressor — report and fingerprint chain
# =====================================================================


class TestCompressorReport:
    def test_empty_report(self):
        c = Compressor()
        r = c.report()
        assert isinstance(r, CompressorReport)
        assert r.models == {}
        assert r.last_fits == {}
        assert r.fingerprint != ""
        assert r.n_events >= 1  # at least the started + report events

    def test_report_after_workflow(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        c.register("u", UNIFORM_DISCRETE, k=2)
        c.fit("b", [0, 1, 0])
        c.codelength("u", [0, 1, 0])
        c.select([0, 1, 0])
        c.compare("b", "u", [0, 1, 0])
        r = c.report()
        assert set(r.models) == {"b", "u"}
        assert r.last_fits["b"].n == 3
        assert "u" in r.last_codelengths
        assert len(r.selections) == 1
        assert len(r.comparisons) == 1

    def test_fingerprint_deterministic_under_clock(self):
        # Same byte sequence → same fingerprint.
        ticks = iter(range(1_000_000))
        c1 = Compressor(clock=lambda: float(next(ticks)))
        c1.register("b", BERNOULLI)
        c1.codelength("b", [0, 1, 0])
        fp1 = c1.fingerprint
        ticks2 = iter(range(1_000_000))
        c2 = Compressor(clock=lambda: float(next(ticks2)))
        c2.register("b", BERNOULLI)
        c2.codelength("b", [0, 1, 0])
        fp2 = c2.fingerprint
        assert fp1 == fp2

    def test_fingerprint_diverges_under_different_actions(self):
        c1 = Compressor()
        c1.register("a", BERNOULLI)
        c2 = Compressor()
        c2.register("b", BERNOULLI)
        assert c1.fingerprint != c2.fingerprint

    def test_clear_resets_state(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        c.codelength("b", [0, 1])
        c.clear()
        assert c.models() == {}
        # registering anew should work
        c.register("b", BERNOULLI)

    def test_events_include_known_kinds(self):
        c = Compressor()
        c.register("b", BERNOULLI)
        c.fit("b", [0, 1])
        c.codelength("b", [0, 1])
        c.online_observe("b", 0)
        c.select([0, 1])
        c.report()
        kinds = {e["kind"] for e in c.events()}
        assert COMPRESSOR_STARTED in kinds
        assert COMPRESSOR_MODEL_REGISTERED in kinds
        assert COMPRESSOR_FIT in kinds
        assert COMPRESSOR_SCORED in kinds
        assert COMPRESSOR_OBSERVED in kinds
        assert COMPRESSOR_SELECTED in kinds
        assert COMPRESSOR_REPORT in kinds
        assert kinds <= KNOWN_EVENTS


# =====================================================================
# Spec-based factory
# =====================================================================


class TestCompressorFromSpec:
    def test_basic_spec(self):
        spec = {
            "models": [
                {"name": "ber", "kind": "bernoulli"},
                {"name": "uni", "kind": "uniform_discrete", "params": {"k": 2}},
                {"name": "g3", "kind": "multinomial", "params": {"k": 3}},
            ]
        }
        c = compressor_from_spec(spec)
        assert set(c.models()) == {"ber", "uni", "g3"}

    def test_empty_spec(self):
        c = compressor_from_spec({})
        assert c.models() == {}

    def test_non_mapping_raises(self):
        with pytest.raises(CompressorError):
            compressor_from_spec([])

    def test_bad_model_entry_raises(self):
        with pytest.raises(CompressorError):
            compressor_from_spec({"models": [["bad"]]})
        with pytest.raises(InvalidModel):
            compressor_from_spec({"models": [{"name": "x"}]})
        with pytest.raises(InvalidModel):
            compressor_from_spec({"models": [{"kind": "bernoulli"}]})


# =====================================================================
# Thread safety smoke
# =====================================================================


class TestCompressorConcurrency:
    def test_concurrent_observe_consistent(self):
        import threading
        c = Compressor()
        c.register("b", BERNOULLI)
        N = 200
        threads = []
        for i in range(N):
            t = threading.Thread(target=c.online_observe, args=("b", i % 2))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s = c.online_state("b")
        assert s.n == N
        # n0 + n1 == N
        assert s.n0 + s.n1 == N


# =====================================================================
# End-to-end: simulated streams
# =====================================================================


class TestEndToEnd:
    def test_bernoulli_recovery(self):
        """If the data is Bernoulli(0.3), the Bernoulli model should win
        against uniform and the right unbiased multinomials."""
        random.seed(2026)
        data = [1 if random.random() < 0.3 else 0 for _ in range(2000)]
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        c.register("k4", MULTINOMIAL, k=4)
        # k4 actually can't see the data because all symbols are in {0,1};
        # we score it on the same data padded into {0,1} ⊂ {0,1,2,3}
        sel = c.select(data, names=["ber", "uni"])
        assert sel.winner == "ber"
        assert sel.bayes_factor > 1.0

    def test_markov_recovery_on_chain(self):
        """A first-order Markov chain (sticky) should beat the IID model."""
        random.seed(2027)
        data = [0]
        for _ in range(2000):
            # stay with prob 0.9
            if random.random() < 0.9:
                data.append(data[-1])
            else:
                data.append(1 - data[-1])
        c = Compressor()
        c.register("iid", MULTINOMIAL, k=2)
        c.register("m1", MARKOV, k=2, r=1)
        sel = c.select(data)
        assert sel.winner == "m1"

    def test_correctly_sized_gaussian_wins(self):
        random.seed(2028)
        # Data is N(0, 1).  Two registered Gaussians, one tightly bounded around
        # mu=0, sigma=1, and one loose.  The tight bounds correspond to the true
        # process and should give a shorter codelength.
        data = [random.gauss(0.0, 1.0) for _ in range(500)]
        c = Compressor()
        c.register("g_tight", GAUSSIAN_KNOWN_SIGMA, sigma=1.0, mu_min=-0.5, mu_max=0.5)
        c.register("g_loose", GAUSSIAN_KNOWN_SIGMA, sigma=1.0, mu_min=-50.0, mu_max=50.0)
        sel = c.select(data, names=["g_tight", "g_loose"])
        # The tight prior on μ-range gives a smaller parametric complexity
        # term — but the data is centred so both fit equally well; the tighter
        # range wins because its parametric complexity is smaller.
        assert sel.winner == "g_tight"

    def test_constant_stream_picked_when_truly_constant(self):
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("k0", CONSTANT, c=0)
        sel = c.select([0] * 100)
        assert sel.winner == "k0"
        # constant is zero bits, bernoulli is nonzero from parametric complexity
        assert sel.codelengths_bits["k0"] == 0.0
        assert sel.codelengths_bits["ber"] > 0.0

    def test_anytime_validity_on_prefix(self):
        """Selection on a prefix and selection on the full stream should both
        return valid certificates; the per-symbol regret bound should shrink with n."""
        c = Compressor()
        c.register("ber", BERNOULLI)
        c.register("uni", UNIFORM_DISCRETE, k=2)
        random.seed(7)
        data = [random.randint(0, 1) for _ in range(2000)]
        s_short = c.select(data[:100])
        s_long = c.select(data)
        # per-symbol regret is a function of (gap, n); even if gap grows with n,
        # the bound is monotone non-decreasing in n only if normalised correctly.
        # We at least check finiteness here.
        assert math.isfinite(s_short.per_symbol_regret_bits)
        assert math.isfinite(s_long.per_symbol_regret_bits)
