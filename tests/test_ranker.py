"""Tests for `agi.ranker` — paired-comparison ranking as a runtime primitive."""

from __future__ import annotations

import math
import random

import pytest

from agi.events import EventBus
from agi.ranker import (
    BRADLEY_TERRY_MAP,
    BRADLEY_TERRY_MM,
    ELO,
    GAUGE_FIX_FIRST,
    GAUGE_ZERO_SUM,
    GLICKO,
    GLICKO2,
    KNOWN_ALGORITHMS,
    PLACKETT_LUCE_MM,
    RANKER_FITTED,
    RANKER_OBSERVED,
    RANKER_REPORT,
    RANKER_STARTED,
    THURSTONE_MM,
    TRUE_SKILL,
    InvalidObservation,
    Ranker,
    RankerError,
    UnknownAlgorithm,
    UnknownItem,
    bradley_terry_fit,
    elo_run,
    empirical_bernstein_half_width,
    hoeffding_half_width,
    hox_sample_complexity,
    hrm_anytime_half_width,
    logistic_log,
    phi,
    phi_pdf,
    plackett_luce_fit,
    rank_correlation_kendall,
    rank_correlation_spearman,
    sigmoid,
    strongly_connected_components,
    trueskill_run,
)


# =====================================================================
# Numerical helpers
# =====================================================================


def test_sigmoid_centred_at_half():
    assert abs(sigmoid(0.0) - 0.5) < 1e-12


def test_sigmoid_monotone():
    xs = [-5, -1, 0, 0.5, 1, 3]
    ys = [sigmoid(x) for x in xs]
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))


def test_sigmoid_extremes_stable():
    # No NaN / no overflow at large |x|.
    assert sigmoid(-1000) >= 0.0 and math.isfinite(sigmoid(-1000))
    assert sigmoid(1000) <= 1.0 and math.isfinite(sigmoid(1000))
    assert sigmoid(-1000) < 1e-100


def test_sigmoid_symmetry():
    for x in [0.1, 1.0, 5.0, 10.0]:
        assert abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-12


def test_logistic_log_consistent_with_log_sigmoid():
    for x in [-3.0, -0.5, 0.0, 0.7, 4.0]:
        assert abs(logistic_log(x) - math.log(sigmoid(x))) < 1e-10


def test_phi_at_zero_is_half():
    assert abs(phi(0.0) - 0.5) < 1e-12


def test_phi_symmetry():
    for x in [0.1, 1.0, 2.0]:
        assert abs(phi(x) + phi(-x) - 1.0) < 1e-12


def test_phi_pdf_centered():
    assert phi_pdf(0.0) > phi_pdf(1.0) > phi_pdf(2.0) > 0


def test_hoeffding_half_width_decreases_with_n():
    h1 = hoeffding_half_width(10, 0.05)
    h2 = hoeffding_half_width(1000, 0.05)
    assert h1 > h2 > 0


def test_hoeffding_half_width_zero_for_zero_n():
    assert math.isinf(hoeffding_half_width(0, 0.05))


def test_empirical_bernstein_smaller_than_hoeffding_at_zero_var():
    n = 1000
    delta = 0.05
    eb = empirical_bernstein_half_width(n, 0.0, delta)
    ho = hoeffding_half_width(n, delta)
    assert eb < ho


def test_empirical_bernstein_monotone_in_variance():
    eb_low = empirical_bernstein_half_width(500, 0.01, 0.05)
    eb_high = empirical_bernstein_half_width(500, 0.25, 0.05)
    assert eb_low < eb_high


def test_hrm_anytime_widens_with_anytime_factor():
    eb = empirical_bernstein_half_width(500, 0.1, 0.05)
    hrm = hrm_anytime_half_width(500, 0.1, 0.05)
    assert hrm > eb        # time-uniform pays an extra log-log factor


# =====================================================================
# Strongly-connected components (Tarjan 1972)
# =====================================================================


def test_scc_single_cycle():
    sccs = strongly_connected_components(3, [(0, 1), (1, 2), (2, 0)])
    assert sccs == [[0, 1, 2]]


def test_scc_disconnected():
    sccs = strongly_connected_components(4, [(0, 1), (1, 0), (2, 3), (3, 2)])
    assert len(sccs) == 2
    assert sorted([sorted(c) for c in sccs]) == [[0, 1], [2, 3]]


def test_scc_chain_no_cycle():
    sccs = strongly_connected_components(3, [(0, 1), (1, 2)])
    # Each vertex is its own SCC in a DAG.
    assert sum(len(c) for c in sccs) == 3
    assert all(len(c) == 1 for c in sccs)


def test_scc_isolated_vertex():
    sccs = strongly_connected_components(3, [(0, 1), (1, 0)])
    sizes = sorted(len(c) for c in sccs)
    assert sizes == [1, 2]


def test_scc_large_random():
    rng = random.Random(0)
    K = 60
    edges = []
    for _ in range(300):
        u, v = rng.sample(range(K), 2)
        edges.append((u, v))
    sccs = strongly_connected_components(K, edges)
    # Every vertex appears exactly once.
    flat = sorted(v for c in sccs for v in c)
    assert flat == list(range(K))


# =====================================================================
# Constructor validation
# =====================================================================


def test_constructor_basic():
    r = Ranker(items=["a", "b", "c"])
    assert r.n_items == 3
    assert r.algorithm == BRADLEY_TERRY_MM


@pytest.mark.parametrize("algo", sorted(KNOWN_ALGORITHMS))
def test_constructor_accepts_all_algorithms(algo):
    r = Ranker(items=["a", "b"], algorithm=algo)
    assert r.algorithm == algo


def test_constructor_rejects_empty_items():
    with pytest.raises(RankerError):
        Ranker(items=[])


def test_constructor_rejects_duplicate_items():
    with pytest.raises(RankerError):
        Ranker(items=["a", "a", "b"])


def test_constructor_rejects_unknown_algorithm():
    with pytest.raises(UnknownAlgorithm):
        Ranker(items=["a", "b"], algorithm="not_a_thing")


def test_constructor_rejects_bad_gauge():
    with pytest.raises(RankerError):
        Ranker(items=["a", "b"], gauge="weird")


def test_constructor_rejects_negative_lam():
    with pytest.raises(RankerError):
        Ranker(items=["a", "b"], lam=-1.0)


def test_constructor_rejects_negative_elo_k():
    with pytest.raises(RankerError):
        Ranker(items=["a", "b"], elo_k=-1.0)


def test_constructor_rejects_bad_draw_prob():
    with pytest.raises(RankerError):
        Ranker(items=["a", "b"], draw_prob=1.5)


def test_constructor_rejects_bad_item_name():
    with pytest.raises(RankerError):
        Ranker(items=["a", "", "b"])


# =====================================================================
# Observation validation
# =====================================================================


def test_observe_unknown_winner_raises():
    r = Ranker(items=["a", "b"])
    with pytest.raises(UnknownItem):
        r.observe_pair("nope", "a")


def test_observe_unknown_loser_raises():
    r = Ranker(items=["a", "b"])
    with pytest.raises(UnknownItem):
        r.observe_pair("a", "nope")


def test_observe_self_raises():
    r = Ranker(items=["a", "b"])
    with pytest.raises(InvalidObservation):
        r.observe_pair("a", "a")


def test_observe_zero_weight_raises():
    r = Ranker(items=["a", "b"])
    with pytest.raises(InvalidObservation):
        r.observe_pair("a", "b", weight=0)


def test_observe_ranking_too_short_raises():
    r = Ranker(items=["a", "b", "c"])
    with pytest.raises(InvalidObservation):
        r.observe_ranking(["a"])


def test_observe_ranking_duplicate_raises():
    r = Ranker(items=["a", "b", "c"])
    with pytest.raises(InvalidObservation):
        r.observe_ranking(["a", "a", "b"])


def test_observe_ranking_unknown_item_raises():
    r = Ranker(items=["a", "b", "c"])
    with pytest.raises(UnknownItem):
        r.observe_ranking(["a", "b", "z"])


def test_observe_score_rejects_nan():
    r = Ranker(items=["a", "b"])
    with pytest.raises(InvalidObservation):
        r.observe_score("a", float("nan"))


# =====================================================================
# Recovery: every algorithm finds the right order from synthetic BT data
# =====================================================================


def _simulate_pairs(skills, n, seed=0):
    rng = random.Random(seed)
    items = list(skills.keys())
    pairs = []
    for _ in range(n):
        a, b = rng.sample(items, 2)
        p = sigmoid(skills[a] - skills[b])
        if rng.random() < p:
            pairs.append((a, b))
        else:
            pairs.append((b, a))
    return pairs


@pytest.mark.parametrize("algo", [
    BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP, PLACKETT_LUCE_MM,
    THURSTONE_MM, GLICKO, GLICKO2, TRUE_SKILL,
])
def test_recovery_order_matches_truth(algo):
    skills = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    pairs = _simulate_pairs(skills, 3000, seed=0)
    r = Ranker(items=list(skills.keys()), algorithm=algo)
    for w, l in pairs:
        r.observe_pair(w, l)
    assert r.rank() == ["D", "C", "B", "A"]


def test_recovery_order_elo_at_high_volume():
    # Elo with default K-factor needs more matches; lower K-factor helps.
    skills = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    pairs = _simulate_pairs(skills, 8000, seed=1)
    r = Ranker(items=list(skills.keys()), algorithm=ELO, elo_k=8.0)
    for w, l in pairs:
        r.observe_pair(w, l)
    # Final ranking must agree on top and bottom.
    rk = r.rank()
    assert rk[0] == "D" and rk[-1] == "A"


# =====================================================================
# Bradley-Terry math
# =====================================================================


def test_bt_mm_fixed_gauge_zero_at_first():
    r = Ranker(items=["A", "B", "C"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(100):
        r.observe_pair("A", "B")
    for _ in range(100):
        r.observe_pair("B", "C")
    r.fit()
    assert abs(r.rate("A").mean) < 1e-9     # gauge anchors first to 0


def test_bt_mm_zero_sum_gauge_sums_to_zero():
    r = Ranker(items=["A", "B", "C"], algorithm=BRADLEY_TERRY_MM,
               gauge=GAUGE_ZERO_SUM)
    for w, l in _simulate_pairs({"A": 0, "B": 1, "C": -1}, 500, seed=2):
        r.observe_pair(w, l)
    s = sum(r.rate(n).mean for n in ["A", "B", "C"])
    assert abs(s) < 1e-6


def test_bt_map_handles_disconnected_graph():
    # Graph has two components: {A,B} and {C,D}.
    r = Ranker(items=["A", "B", "C", "D"], algorithm=BRADLEY_TERRY_MAP, lam=1.0)
    for _ in range(20):
        r.observe_pair("A", "B")
    for _ in range(20):
        r.observe_pair("C", "D")
    r.fit()
    # MAP must still produce finite ratings (ridge prior keeps it identifiable).
    for n in ["A", "B", "C", "D"]:
        assert math.isfinite(r.rate(n).mean)


def test_bt_mm_loglikelihood_increases_with_fit():
    skills = {"A": 0.0, "B": 1.5}
    pairs = _simulate_pairs(skills, 300, seed=3)
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM, auto_fit=False)
    for w, l in pairs:
        r.observe_pair(w, l)
    # Before fitting, log-lik is at the zero-vector point.
    from agi.ranker import _bt_log_likelihood
    ll0 = _bt_log_likelihood(
        2, [0.0, 0.0], r._pair_wins, r._pair_counts,
    )
    r.fit()
    assert r._last_log_likelihood > ll0


def test_bt_fisher_se_finite_and_positive():
    skills = {"A": 0.0, "B": 1.5, "C": 0.5}
    pairs = _simulate_pairs(skills, 500, seed=4)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    for n in skills:
        rt = r.rate(n)
        assert math.isfinite(rt.stderr)
        assert rt.stderr >= 0.0


# =====================================================================
# Plackett-Luce
# =====================================================================


def test_pl_handles_full_rankings():
    rankings = []
    rng = random.Random(0)
    items = ["A", "B", "C", "D"]
    truth = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}

    def _sample_ranking():
        # Sample without replacement under PL: at each step, sample
        # proportional to exp(θ_i) from the survivors.
        survivors = list(items)
        out = []
        for _ in range(len(items)):
            weights = [math.exp(truth[i]) for i in survivors]
            s = sum(weights)
            r = rng.uniform(0, s)
            c = 0.0
            for k, w in enumerate(weights):
                c += w
                if r <= c:
                    out.append(survivors.pop(k))
                    break
        return out

    for _ in range(300):
        rankings.append(_sample_ranking())
    r = Ranker(items=items, algorithm=PLACKETT_LUCE_MM)
    for rk in rankings:
        r.observe_ranking(rk)
    assert r.rank() == ["D", "C", "B", "A"]


def test_pl_with_partial_top_k_rankings():
    # Only observe top-2 of each 4-permutation.
    rankings = []
    rng = random.Random(0)
    items = ["A", "B", "C", "D"]
    truth = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    for _ in range(400):
        survivors = list(items)
        out = []
        for _ in range(2):       # only top-2
            weights = [math.exp(truth[i]) for i in survivors]
            s = sum(weights)
            r = rng.uniform(0, s)
            c = 0.0
            for k, w in enumerate(weights):
                c += w
                if r <= c:
                    out.append(survivors.pop(k))
                    break
        rankings.append(out)
    r = Ranker(items=items, algorithm=PLACKETT_LUCE_MM)
    for rk in rankings:
        r.observe_ranking(rk)
    # D should still emerge as the top, A at the bottom.
    rk = r.rank()
    assert rk[0] == "D"


# =====================================================================
# Thurstone-Mosteller
# =====================================================================


def test_thurstone_recovers_order():
    skills = {"A": 0.0, "B": 0.8, "C": 1.6, "D": 2.4}
    pairs = _simulate_pairs(skills, 3000, seed=7)
    r = Ranker(items=list(skills.keys()), algorithm=THURSTONE_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    assert r.rank() == ["D", "C", "B", "A"]


# =====================================================================
# Elo / Glicko / Glicko-2 / TrueSkill
# =====================================================================


def test_elo_winner_rating_goes_up():
    r = Ranker(items=["A", "B"], algorithm=ELO)
    initial_a = r.rate("A").mean
    r.observe_pair("A", "B")
    assert r.rate("A").mean > initial_a
    assert r.rate("B").mean < initial_a


def test_elo_default_rating_is_1500():
    r = Ranker(items=["A", "B"], algorithm=ELO)
    assert abs(r.rate("A").mean - 1500.0) < 1e-9


def test_elo_zero_sum_property():
    r = Ranker(items=["A", "B"], algorithm=ELO)
    s0 = r.rate("A").mean + r.rate("B").mean
    r.observe_pair("A", "B")
    s1 = r.rate("A").mean + r.rate("B").mean
    assert abs(s0 - s1) < 1e-9


def test_glicko_shrinks_uncertainty_with_play():
    r = Ranker(items=["A", "B"], algorithm=GLICKO)
    initial_phi = r.rate("A").stderr
    for _ in range(10):
        r.observe_pair("A", "B")
    assert r.rate("A").stderr < initial_phi


def test_glicko2_volatility_field_changes():
    r = Ranker(items=["A", "B"], algorithm=GLICKO2)
    initial = r._items[0].sigma
    for _ in range(20):
        r.observe_pair("A", "B")
        r.observe_pair("B", "A")
    final = r._items[0].sigma
    # Volatility should evolve (up or down depending on noise).
    assert final != initial


def test_trueskill_uncertainty_shrinks_with_play():
    r = Ranker(items=["A", "B"], algorithm=TRUE_SKILL)
    sigma0 = r.rate("A").stderr
    for _ in range(50):
        r.observe_pair("A", "B")
    assert r.rate("A").stderr < sigma0


def test_trueskill_winner_mean_increases():
    r = Ranker(items=["A", "B"], algorithm=TRUE_SKILL)
    mu_a0 = r.rate("A").mean
    r.observe_pair("A", "B")
    assert r.rate("A").mean > mu_a0
    assert r.rate("B").mean < mu_a0


def test_trueskill_draw_pulls_means_together():
    r = Ranker(items=["A", "B"], algorithm=TRUE_SKILL, draw_prob=0.3)
    # First open up a gap.
    for _ in range(10):
        r.observe_pair("A", "B")
    gap_before = abs(r.rate("A").mean - r.rate("B").mean)
    for _ in range(20):
        r.observe_pair("A", "B", draw=True)
    gap_after = abs(r.rate("A").mean - r.rate("B").mean)
    assert gap_after < gap_before


# =====================================================================
# Win-probability + CI
# =====================================================================


def test_predict_win_prob_at_equal_skill_is_half():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    # No observations: skills equal → P = 0.5.
    p = r.predict_win_prob("A", "B")
    assert abs(p - 0.5) < 1e-9


def test_predict_win_prob_dominant_above_half():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(20):
        r.observe_pair("A", "B")
    assert r.predict_win_prob("A", "B") > 0.9


def test_win_probability_ci_is_significant_for_strong_winner():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(100):
        r.observe_pair("A", "B")
    cp = r.compare("A", "B")
    assert cp.is_significant
    assert cp.ci_low > 0.5
    assert cp.n_direct == 100


def test_win_probability_ci_not_significant_for_50_50():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(50):
        r.observe_pair("A", "B")
        r.observe_pair("B", "A")
    cp = r.compare("A", "B")
    assert not cp.is_significant


def test_win_probability_ci_anytime_wider():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(50):
        r.observe_pair("A", "B")
        r.observe_pair("B", "A")
    eb = r.win_probability_ci("A", "B", anytime=False)
    hrm = r.win_probability_ci("A", "B", anytime=True)
    assert hrm.ci_half_width >= eb.ci_half_width


def test_win_probability_ci_model_fallback_when_no_direct_pairs():
    r = Ranker(items=["A", "B", "C"], algorithm=BRADLEY_TERRY_MM)
    # Only AB and BC observations; no direct AC.
    for _ in range(80):
        r.observe_pair("A", "B")
    for _ in range(80):
        r.observe_pair("B", "C")
    cp = r.compare("A", "C")
    assert cp.n_direct == 0
    assert cp.method == "model_delta"
    assert 0.0 <= cp.ci_low <= cp.mean_win_prob <= cp.ci_high <= 1.0


# =====================================================================
# Top-K with PAC certificate
# =====================================================================


def test_top_k_returns_expected_items():
    skills = {"A": 0.0, "B": 0.5, "C": 1.5, "D": 3.0}
    pairs = _simulate_pairs(skills, 2000, seed=10)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    dec = r.top_k(2)
    assert dec.items == ["D", "C"]


def test_top_k_pac_certified_with_clear_gap():
    skills = {"A": 0.0, "B": 0.1, "C": 2.5, "D": 3.0}
    pairs = _simulate_pairs(skills, 4000, seed=11)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    dec = r.top_k(2)
    # Gap between B and C is huge → certification should land.
    assert dec.pac_certified


def test_top_k_not_certified_with_tight_gap():
    skills = {"A": 0.0, "B": 0.01, "C": 0.02, "D": 0.03}
    pairs = _simulate_pairs(skills, 200, seed=12)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    dec = r.top_k(2)
    assert not dec.pac_certified


def test_top_k_validates_k():
    r = Ranker(items=["A", "B", "C"])
    with pytest.raises(RankerError):
        r.top_k(0)
    with pytest.raises(RankerError):
        r.top_k(10)


# =====================================================================
# Report — diagnostics, identifiability, gauge round-trip
# =====================================================================


def test_report_carries_full_diagnostics():
    skills = {"A": 0.0, "B": 1.0, "C": 2.0}
    pairs = _simulate_pairs(skills, 500, seed=13)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    rep = r.report()
    assert rep.identifiable
    assert rep.scc_size == 3
    assert rep.scc_count >= 1
    assert rep.isolated_items == []
    assert rep.rank_order == ["C", "B", "A"]
    assert rep.algorithm == BRADLEY_TERRY_MM
    assert rep.pseudo_r2 > 0.0
    assert rep.converged
    assert rep.fingerprint.startswith("sha256:")


def test_report_detects_non_identifiable_graph():
    # No comparisons at all → no edges, every vertex is its own SCC.
    r = Ranker(items=["A", "B", "C"], algorithm=BRADLEY_TERRY_MM)
    rep = r.report()
    assert not rep.identifiable
    assert rep.scc_size == 1
    assert sorted(rep.isolated_items) == ["B", "C"]


def test_report_sample_complexity_nonzero():
    skills = {"A": 0.0, "B": 1.0, "C": 2.0}
    pairs = _simulate_pairs(skills, 200, seed=14)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    rep = r.report()
    assert rep.sample_complexity_to_topk_99 > 0


def test_report_to_dict_roundtrips_json():
    import json as _json
    skills = {"A": 0.0, "B": 1.0}
    pairs = _simulate_pairs(skills, 50, seed=15)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    rep = r.report().to_dict()
    s = _json.dumps(rep, default=str)
    assert "rank_order" in s and "fingerprint" in s


# =====================================================================
# Fingerprint = replay-determinism contract
# =====================================================================


def test_fingerprint_changes_with_observation():
    r = Ranker(items=["A", "B"])
    fp0 = r.fingerprint
    r.observe_pair("A", "B")
    fp1 = r.fingerprint
    assert fp0 != fp1


def test_fingerprint_includes_algorithm():
    r1 = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    r2 = Ranker(items=["A", "B"], algorithm=TRUE_SKILL)
    assert r1.fingerprint != r2.fingerprint


def test_fingerprint_includes_items_order():
    r1 = Ranker(items=["A", "B"])
    r2 = Ranker(items=["B", "A"])
    assert r1.fingerprint != r2.fingerprint


def test_fingerprint_stable_across_identical_inputs():
    r1 = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    r2 = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(20):
        r1.observe_pair("A", "B")
        r2.observe_pair("A", "B")
    assert r1.fingerprint == r2.fingerprint


def test_fingerprint_changes_with_draw():
    r1 = Ranker(items=["A", "B"])
    r2 = Ranker(items=["A", "B"])
    r1.observe_pair("A", "B", draw=False)
    r2.observe_pair("A", "B", draw=True)
    assert r1.fingerprint != r2.fingerprint


# =====================================================================
# state() / from_state()
# =====================================================================


def test_state_roundtrip_preserves_ratings():
    skills = {"A": 0.0, "B": 1.0, "C": 2.0}
    pairs = _simulate_pairs(skills, 500, seed=17)
    r = Ranker(items=list(skills.keys()), algorithm=BRADLEY_TERRY_MM)
    for w, l in pairs:
        r.observe_pair(w, l)
    state = r.state()
    r2 = Ranker.from_state(state)
    r2.fit()
    for n in skills:
        assert abs(r.rate(n).mean - r2.rate(n).mean) < 1e-9


def test_state_roundtrip_preserves_trueskill():
    r = Ranker(items=["A", "B", "C"], algorithm=TRUE_SKILL)
    for _ in range(40):
        r.observe_pair("A", "B")
        r.observe_pair("B", "C")
    state = r.state()
    r2 = Ranker.from_state(state)
    assert abs(r.rate("A").mean - r2.rate("A").mean) < 1e-12
    assert abs(r.rate("A").stderr - r2.rate("A").stderr) < 1e-12


def test_state_is_json_serializable():
    import json as _json
    r = Ranker(items=["A", "B"])
    for _ in range(10):
        r.observe_pair("A", "B")
    s = _json.dumps(r.state())
    state = _json.loads(s)
    r2 = Ranker.from_state(state)
    assert r2.rank() == r.rank()


# =====================================================================
# Forget, clear, weight
# =====================================================================


def test_forget_decays_counts():
    r = Ranker(items=["A", "B", "C"])
    for _ in range(100):
        r.observe_pair("A", "B")
    n_before = r._pair_counts[r._idx("A")][r._idx("B")]
    r.forget("A", halflife=10.0)
    n_after = r._pair_counts[r._idx("A")][r._idx("B")]
    assert n_after < n_before


def test_forget_invalid_halflife_raises():
    r = Ranker(items=["A", "B"])
    with pytest.raises(RankerError):
        r.forget("A", halflife=0)


def test_clear_wipes_observations():
    r = Ranker(items=["A", "B"])
    for _ in range(20):
        r.observe_pair("A", "B")
    r.clear()
    assert r.n_observations == 0
    assert r.rate("A").mean == 0.0 or abs(r.rate("A").mean) < 1e-9
    assert r._pair_counts[0][1] == 0.0


def test_weighted_observation_scales_count():
    r = Ranker(items=["A", "B"])
    r.observe_pair("A", "B", weight=3.0)
    assert r._pair_counts[r._idx("A")][r._idx("B")] == 3.0


def test_draw_observation_splits_credit():
    r = Ranker(items=["A", "B"])
    r.observe_pair("A", "B", draw=True)
    assert r._pair_wins[r._idx("A")][r._idx("B")] == 0.5
    assert r._pair_wins[r._idx("B")][r._idx("A")] == 0.5


# =====================================================================
# Events
# =====================================================================


def test_events_published_on_observation():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda e: seen.append(e.kind))
    r = Ranker(items=["A", "B"], bus=bus)
    r.observe_pair("A", "B")
    r.observe_pair("B", "A")
    r.report()
    kinds = set(seen)
    assert RANKER_STARTED in kinds
    assert RANKER_OBSERVED in kinds
    assert RANKER_REPORT in kinds


def test_events_published_with_session_id():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda e: seen.append(e))
    r = Ranker(items=["A", "B"], bus=bus, session_id="sid-42")
    r.observe_pair("A", "B")
    sids = {e.session_id for e in seen if e.session_id is not None}
    assert sids == {"sid-42"}


def test_event_subscriber_failure_does_not_break_ranker():
    bus = EventBus()
    def boom(_e):
        raise RuntimeError("listener bug")
    bus.subscribe(boom)
    r = Ranker(items=["A", "B"], bus=bus)
    # Should not raise.
    r.observe_pair("A", "B")
    assert r.n_observations == 1


# =====================================================================
# Convenience module-level fits
# =====================================================================


def test_bradley_terry_fit_convenience():
    pairs = _simulate_pairs({"A": 0.0, "B": 1.5}, 200, seed=18)
    theta = bradley_terry_fit(items=["A", "B"], pairs=pairs)
    assert theta["B"] > theta["A"]
    assert abs(theta["A"]) < 1e-9       # gauge fixes first to 0


def test_plackett_luce_fit_convenience():
    rng = random.Random(0)
    items = ["A", "B", "C"]
    truth = {"A": 0.0, "B": 1.0, "C": 2.0}
    rankings = []
    for _ in range(200):
        survivors = list(items)
        order = []
        for _ in range(len(items)):
            ws = [math.exp(truth[s]) for s in survivors]
            s = sum(ws)
            r = rng.uniform(0, s)
            c = 0.0
            for k, w in enumerate(ws):
                c += w
                if r <= c:
                    order.append(survivors.pop(k))
                    break
        rankings.append(order)
    theta = plackett_luce_fit(items=items, rankings=rankings)
    assert theta["C"] > theta["B"] > theta["A"]


def test_elo_run_convenience():
    pairs = _simulate_pairs({"A": 0.0, "B": 2.0}, 100, seed=19)
    res = elo_run(items=["A", "B"], pairs=pairs)
    assert res["B"] > res["A"]


def test_trueskill_run_convenience():
    pairs = _simulate_pairs({"A": 0.0, "B": 2.0}, 100, seed=20)
    res = trueskill_run(items=["A", "B"], pairs=pairs)
    assert res["B"][0] > res["A"][0]
    assert res["A"][1] > 0  # σ_A > 0


# =====================================================================
# Sample-complexity helper
# =====================================================================


def test_hox_sample_complexity_decreases_with_gap():
    n_small_gap = hox_sample_complexity(k=10, gap=0.05)
    n_big_gap = hox_sample_complexity(k=10, gap=0.5)
    assert n_small_gap > n_big_gap


def test_hox_sample_complexity_scales_with_k():
    n_few = hox_sample_complexity(k=5, gap=0.1)
    n_many = hox_sample_complexity(k=50, gap=0.1)
    assert n_many > n_few


def test_hox_sample_complexity_zero_gap_is_huge():
    assert hox_sample_complexity(k=5, gap=0.0) >= 10 ** 8


# =====================================================================
# Rank-correlation helpers
# =====================================================================


def test_kendall_perfect():
    assert rank_correlation_kendall(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_kendall_reversed():
    assert rank_correlation_kendall(["A", "B", "C", "D"], ["D", "C", "B", "A"]) == -1.0


def test_kendall_one_swap():
    tau = rank_correlation_kendall(["A", "B", "C"], ["A", "C", "B"])
    assert 0.0 < tau < 1.0


def test_spearman_perfect():
    assert abs(rank_correlation_spearman(["A", "B", "C"], ["A", "B", "C"]) - 1.0) < 1e-12


def test_spearman_reversed():
    assert abs(
        rank_correlation_spearman(["A", "B", "C", "D"], ["D", "C", "B", "A"]) + 1.0
    ) < 1e-12


def test_rank_correlation_rejects_different_items():
    with pytest.raises(ValueError):
        rank_correlation_kendall(["A", "B"], ["A", "C"])
    with pytest.raises(ValueError):
        rank_correlation_spearman(["A", "B"], ["A", "B", "C"])


# =====================================================================
# Composition with other primitives (smoke tests)
# =====================================================================


def test_compose_with_attestation_ledger():
    # The Ranker fingerprint should hash into the attestation chain.
    from agi.attest import AttestationLedger

    r = Ranker(items=["A", "B"])
    for _ in range(10):
        r.observe_pair("A", "B")
    ledger = AttestationLedger()
    entry = ledger.append({
        "ticket_id": "ranker-001",
        "ranker_fp": r.fingerprint,
        "winner": r.rank()[0],
    })
    assert entry.entry_hash
    ok, reason = ledger.verify()
    assert ok, reason


def test_compose_with_event_bus_history():
    bus = EventBus()
    r = Ranker(items=["A", "B"], bus=bus)
    r.observe_pair("A", "B")
    r.report()
    # Histories should include both observed and report kinds.
    kinds = {e.kind for e in bus.history()}
    assert RANKER_OBSERVED in kinds
    assert RANKER_REPORT in kinds


# =====================================================================
# Auto-fit, batch ingestion mode
# =====================================================================


def test_auto_fit_off_defers_fit():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM, auto_fit=False)
    for _ in range(100):
        r.observe_pair("A", "B")
    assert r._last_iterations == 0    # never fit yet
    r.fit()
    assert r._last_iterations > 0


def test_auto_fit_on_fits_every_observe():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM, auto_fit=True)
    r.observe_pair("A", "B")
    iter1 = r._last_iterations
    r.observe_pair("A", "B")
    # Both fits produced at least one iteration.
    assert iter1 > 0 and r._last_iterations > 0


def test_auto_fit_no_op_for_online_algos():
    r = Ranker(items=["A", "B"], algorithm=ELO, auto_fit=True)
    r.observe_pair("A", "B")
    # Online algos never set _last_iterations.
    assert r._last_iterations == 0


# =====================================================================
# Edge cases — single item, single observation, all draws
# =====================================================================


def test_single_item_smoke():
    r = Ranker(items=["only_one"])
    assert r.rank() == ["only_one"]
    rep = r.report()
    assert rep.scc_size == 1
    assert rep.scc_count == 1


def test_only_draws_gives_50_50_estimate():
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(50):
        r.observe_pair("A", "B", draw=True)
    # Equal wins → P(A>B) ≈ 0.5.
    assert abs(r.predict_win_prob("A", "B") - 0.5) < 0.05


def test_observe_score_records_extra():
    r = Ranker(items=["A", "B"])
    r.observe_score("A", 3.5)
    r.observe_score("A", 4.5)
    assert r._items[r._idx("A")].extra["scores"] == 8.0
    assert r._items[r._idx("A")].extra["score_count"] == 2.0


# =====================================================================
# Multi-item BT correctness against a hand-worked closed form
# =====================================================================


def test_bt_two_items_match_closed_form():
    """For a 2-item BT model with W wins and L losses, the MLE is
    π_A / π_B = W / L, i.e., θ_A − θ_B = log(W/L).
    """
    r = Ranker(items=["A", "B"], algorithm=BRADLEY_TERRY_MM, gauge=GAUGE_FIX_FIRST)
    for _ in range(60):
        r.observe_pair("B", "A")        # B wins 60, A wins 20
    for _ in range(20):
        r.observe_pair("A", "B")
    r.fit()
    expected = math.log(60.0 / 20.0)
    actual = r.rate("B").mean - r.rate("A").mean
    assert abs(actual - expected) < 1e-6


def test_bt_three_items_circular_data():
    """Cyclic data A>B 30, B>C 30, C>A 30: tests numerical stability
    when no Condorcet winner exists.  BT-MM should still converge to a
    cyclic-symmetric near-uniform solution.
    """
    r = Ranker(items=["A", "B", "C"], algorithm=BRADLEY_TERRY_MM)
    for _ in range(30):
        r.observe_pair("A", "B")
        r.observe_pair("B", "C")
        r.observe_pair("C", "A")
    rep = r.report()
    assert rep.converged
    means = [r.rate(n).mean for n in ["A", "B", "C"]]
    # Range of means is small under perfect rock-paper-scissors symmetry.
    assert max(means) - min(means) < 0.05
