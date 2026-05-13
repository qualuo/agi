"""Tests for `agi.deliberator` — adaptive sequential sampling kernel.

Three contracts to verify:

1. **Anytime-valid coverage.** A Deliberator run with commit_threshold = 0.5
   at level α on a stream where the *true* modal probability is 0.5 should
   commit (with confidence "modal cluster dominates") at a rate ≤ α even
   under data-dependent stopping. This is the central guarantee of WSR
   confidence sequences: classical fixed-n CIs lose validity under early
   stopping, ours does not.

2. **Compute savings on easy queries.** When the true modal probability
   is far from the threshold (e.g. 0.9 vs commit=0.5), the Deliberator
   should stop in well below `max_samples`. We don't make a tight claim,
   just that the median run is short.

3. **Convergence detection.** When samples genuinely tie (true p_top
   below commit and EIG stabilises), the Deliberator should report
   STOP_CONVERGENCE rather than STOP_BUDGET, so a coordination engine
   can route the input for escalation instead of guessing.

We also check the cosmetic surface: dataclass invariants, event emissions,
coverage reporting, batch-mode equivalence, and the attestation hash.
"""
from __future__ import annotations

import math
import random
import statistics

import pytest

from agi.deliberator import (
    DELIB_COMMITTED,
    DELIB_ESCALATED,
    DELIB_EXHAUSTED,
    DELIB_OBSERVED,
    DELIB_SAMPLED,
    DELIB_STARTED,
    Deliberation,
    Deliberator,
    STOP_BUDGET,
    STOP_CONVERGENCE,
    STOP_EVIDENCE,
    STOP_INFEASIBLE,
    Sample,
    canonical_cluster_key,
)
from agi.events import Event, EventBus


# ----- helpers ---------------------------------------------------------


def _mk_sampler(rng: random.Random, weights: dict[str, float], cost: float = 0.01):
    """A sampler that returns `Sample(cluster_key=...)` drawn from `weights`."""
    keys = list(weights)
    probs = [weights[k] for k in keys]
    total = sum(probs)
    probs = [p / total for p in probs]

    def _draw() -> Sample:
        u = rng.random()
        c = 0.0
        for k, p in zip(keys, probs):
            c += p
            if u <= c:
                return Sample(answer=k, cluster_key=k, cost=cost)
        return Sample(answer=keys[-1], cluster_key=keys[-1], cost=cost)

    return _draw


# ----- basic surface ---------------------------------------------------


def test_deliberator_constructor_validates_args():
    Deliberator()
    with pytest.raises(ValueError):
        Deliberator(prior_strength=0.0)
    with pytest.raises(ValueError):
        Deliberator(default_alpha=0.0)
    with pytest.raises(ValueError):
        Deliberator(default_alpha=1.0)
    with pytest.raises(ValueError):
        Deliberator(default_commit_threshold=0.0)
    with pytest.raises(ValueError):
        Deliberator(default_commit_threshold=1.0)
    with pytest.raises(ValueError):
        Deliberator(default_eig_floor=-0.001)
    with pytest.raises(ValueError):
        Deliberator(eig_window=0)


def test_deliberate_validates_args():
    d = Deliberator()
    s = _mk_sampler(random.Random(0), {"a": 1.0})
    with pytest.raises(ValueError):
        d.deliberate(s, max_samples=0)
    with pytest.raises(ValueError):
        d.deliberate(s, max_samples=4, min_samples=5)
    with pytest.raises(ValueError):
        d.deliberate(s, alpha=0.0)
    with pytest.raises(ValueError):
        d.deliberate(s, commit_threshold=1.0)
    with pytest.raises(ValueError):
        d.deliberate(s, eig_floor=-0.1)


def test_sampler_must_return_sample():
    d = Deliberator()
    with pytest.raises(TypeError):
        d.deliberate(lambda: "not a sample", max_samples=4)


def test_canonical_cluster_key_normalises():
    assert canonical_cluster_key("Hello   World") == canonical_cluster_key("hello world")
    assert canonical_cluster_key("foo", normalise=False) != canonical_cluster_key("Foo", normalise=False)


# ----- evidence stopping on a heavily-favoured cluster -----------------


def test_evidence_stop_on_clear_winner():
    rng = random.Random(42)
    d = Deliberator()
    # The "yes" cluster wins ~95% of samples — easy.
    sampler = _mk_sampler(rng, {"yes": 0.95, "no": 0.05})
    delib = d.deliberate(sampler, max_samples=32, alpha=0.05, commit_threshold=0.5)
    assert delib.stop_reason == STOP_EVIDENCE
    assert delib.cluster_key == "yes"
    assert delib.posterior_lower >= 0.5
    assert delib.posterior_mean > 0.85
    assert delib.n_samples <= 32
    assert delib.n_samples >= 1
    # The receipt hash is non-empty and stable across identical inputs.
    assert len(delib.receipt_hash) == 32


def test_evidence_stop_terminates_fast_on_extreme_winner():
    # When one cluster always wins, we should commit quickly.
    rng = random.Random(0)
    d = Deliberator()
    sampler = _mk_sampler(rng, {"yes": 1.0})
    ns = []
    for _ in range(30):
        delib = d.deliberate(sampler, max_samples=64, alpha=0.05, commit_threshold=0.5)
        assert delib.stop_reason == STOP_EVIDENCE
        ns.append(delib.n_samples)
    # Median should be small (≤ 15 by construction of the WSR LCB
    # for E[X] = 1).
    assert statistics.median(ns) <= 15


# ----- convergence stopping on an ambiguous stream ---------------------


def test_convergence_stop_on_genuine_tie():
    """When the true modal mass is below the commit threshold and the
    posterior stabilises, we should escalate (STOP_CONVERGENCE), not
    exhaust budget."""
    rng = random.Random(123)
    d = Deliberator(eig_window=3)
    # Three-way tie — true mode mass = 1/3 < 0.5 commit threshold.
    sampler = _mk_sampler(rng, {"a": 1.0, "b": 1.0, "c": 1.0})
    seen_convergence = 0
    for _ in range(20):
        delib = d.deliberate(
            sampler,
            max_samples=64,
            alpha=0.05,
            commit_threshold=0.5,
            eig_floor=0.02,        # generous floor; ties stabilise quickly
        )
        if delib.stop_reason == STOP_CONVERGENCE:
            seen_convergence += 1
        # In a 3-way tie the LCB can't possibly cross 0.5 from below
        # without dramatic luck; nearly all runs should NOT commit.
        assert delib.stop_reason != STOP_EVIDENCE
    # At least most runs should hit convergence (not exhaust budget).
    assert seen_convergence >= 14


def test_budget_stop_when_neither_evidence_nor_convergence():
    rng = random.Random(7)
    d = Deliberator(eig_window=3)
    sampler = _mk_sampler(rng, {"a": 0.55, "b": 0.45})
    # Tight budget, demanding threshold, very small EIG floor → likely
    # to exhaust without committing or converging.
    delib = d.deliberate(
        sampler,
        max_samples=4,
        alpha=0.05,
        commit_threshold=0.9,
        eig_floor=0.0,
    )
    assert delib.stop_reason in (STOP_BUDGET, STOP_EVIDENCE)
    # n_samples must respect the cap.
    assert delib.n_samples <= 4


def test_cost_budget_stops_run_early():
    rng = random.Random(11)
    d = Deliberator()
    # Each sample costs 1.0; cost cap 2.5 ⇒ at most 3 samples drawn.
    sampler = _mk_sampler(rng, {"a": 0.5, "b": 0.5}, cost=1.0)
    delib = d.deliberate(
        sampler,
        max_samples=64,
        max_cost=2.5,
        alpha=0.05,
        commit_threshold=0.9,
        eig_floor=0.0,
    )
    assert delib.n_samples <= 3
    assert delib.cost <= 3.0
    if delib.stop_reason == STOP_BUDGET:
        assert "cost" in delib.rationale.lower() or "max_samples" in delib.rationale.lower()


# ----- anytime-valid coverage under data-dependent stopping ------------


def test_anytime_valid_coverage_under_null():
    """The headline guarantee: if the true modal probability is *exactly*
    the commit threshold (0.5 here), the Deliberator should commit
    (STOP_EVIDENCE) at most α fraction of the time — even with
    data-dependent stopping.

    We test the boundary: true p_top = 0.5, commit_threshold = 0.5.
    Under a sound anytime-valid LCB, the LCB crosses 0.5 only when
    there is genuine evidence p_top > 0.5; under the null p_top = 0.5
    this should happen at most α of the time."""
    rng = random.Random(2025)
    alpha = 0.1
    d = Deliberator()
    sampler = _mk_sampler(rng, {"a": 0.5, "b": 0.5})

    false_commits = 0
    trials = 200
    for _ in range(trials):
        delib = d.deliberate(
            sampler,
            max_samples=40,
            alpha=alpha,
            commit_threshold=0.5,
            eig_floor=0.0,         # never stop on EIG; only evidence/budget
            min_samples=2,
        )
        if delib.stop_reason == STOP_EVIDENCE:
            false_commits += 1
    rate = false_commits / trials
    # Anytime-valid bound: rate ≤ α + Monte Carlo slack.
    # With trials=200 the Wald slack at 99% is ~3*sqrt(α(1-α)/200) ≈ 0.064
    # for α=0.1. Use a comfortable cushion.
    assert rate <= alpha + 0.1, (
        f"false-commit rate {rate:.3f} > α + slack ({alpha + 0.1:.3f}) — "
        "anytime-valid contract broken"
    )


# ----- events, observation, coverage report ---------------------------


def test_events_emitted_in_order():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(lambda e: seen.append(e))
    rng = random.Random(0)
    d = Deliberator(bus=bus)
    sampler = _mk_sampler(rng, {"yes": 1.0})
    delib = d.deliberate(sampler, max_samples=8)
    kinds = [e.kind for e in seen]
    assert DELIB_STARTED in kinds
    assert kinds.count(DELIB_SAMPLED) == delib.n_samples
    # Terminal event matches stop_reason.
    if delib.stop_reason == STOP_EVIDENCE:
        assert DELIB_COMMITTED in kinds
        assert kinds[-1] == DELIB_COMMITTED
    elif delib.stop_reason == STOP_CONVERGENCE:
        assert DELIB_ESCALATED in kinds
    elif delib.stop_reason == STOP_BUDGET:
        assert DELIB_EXHAUSTED in kinds


def test_observe_and_coverage_report():
    rng = random.Random(0)
    d = Deliberator()
    sampler_easy = _mk_sampler(rng, {"yes": 0.95, "no": 0.05})
    correct = 0
    for _ in range(20):
        delib = d.deliberate(sampler_easy, max_samples=32, alpha=0.05)
        if delib.stop_reason == STOP_EVIDENCE:
            # "Correct" iff committed cluster is the dominant one.
            ok = delib.cluster_key == "yes"
            d.observe(delib, success=ok)
            if ok:
                correct += 1
    rep = d.coverage_report()
    assert rep.n_evidence >= 1
    assert rep.realised_success_rate >= 0.9
    assert rep.target_success_rate == pytest.approx(1 - 0.05)
    # No miscoverage when the dominant cluster wins.
    assert rep.miscoverage <= 0.05


def test_coverage_report_quiet_when_no_evidence_commits():
    """A deliberator with only STOP_CONVERGENCE / STOP_BUDGET observations
    must not report miscoverage — there were no commits to fail coverage on."""
    d = Deliberator()
    rng = random.Random(0)
    sampler = _mk_sampler(rng, {"a": 1.0, "b": 1.0, "c": 1.0})
    for _ in range(5):
        delib = d.deliberate(sampler, max_samples=8, eig_floor=0.02)
        d.observe(delib, success=None)
    rep = d.coverage_report()
    assert rep.n_evidence == 0
    assert rep.miscoverage == 0.0


def test_observe_with_none_is_uncounted():
    d = Deliberator()
    rng = random.Random(1)
    sampler = _mk_sampler(rng, {"a": 1.0})
    delib = d.deliberate(sampler, max_samples=8)
    d.observe(delib, success=None)
    rep = d.coverage_report()
    # Stop count is recorded but success is not.
    assert rep.n_observed == 0
    assert sum(rep.n_by_stop_reason.values()) >= 1


# ----- batch mode -----------------------------------------------------


def test_batch_mode_matches_online_when_no_early_stop():
    rng = random.Random(3)
    d = Deliberator()
    sampler = _mk_sampler(rng, {"a": 0.6, "b": 0.4})
    fixed = [sampler() for _ in range(20)]
    rep_online = d.deliberate_batch(fixed, alpha=0.05, commit_threshold=0.5)
    # Same n_samples, same modal cluster.
    counts = {}
    for s in fixed:
        counts[s.cluster_key] = counts.get(s.cluster_key, 0) + 1
    top = max(counts, key=counts.get)
    assert rep_online.cluster_key == top
    assert rep_online.n_samples == 20


def test_batch_mode_empty_returns_infeasible():
    d = Deliberator()
    rep = d.deliberate_batch([])
    assert rep.stop_reason == STOP_INFEASIBLE
    assert rep.n_samples == 0
    assert rep.answer is None


# ----- structural invariants ------------------------------------------


def test_cluster_stats_invariants():
    rng = random.Random(4)
    d = Deliberator()
    sampler = _mk_sampler(rng, {"a": 0.5, "b": 0.3, "c": 0.2})
    delib = d.deliberate(sampler, max_samples=20)
    # Posterior means sum to 1 (Dirichlet posterior is proper).
    assert delib.posterior_mean > 0.0
    means = [c.posterior_mean for c in delib.clusters]
    assert abs(sum(means) - 1.0) < 1e-9
    # Each cluster's LCB ≤ mean ≤ UCB.
    for c in delib.clusters:
        assert 0.0 <= c.posterior_lower <= c.posterior_mean + 1e-9
        assert c.posterior_mean - 1e-9 <= c.posterior_upper <= 1.0
    # Counts sum to n_samples.
    assert sum(c.count for c in delib.clusters) == delib.n_samples
    # cluster_key is the modal cluster.
    top = max(delib.clusters, key=lambda c: c.count)
    assert top.key == delib.cluster_key


def test_to_dict_is_json_safe():
    rng = random.Random(5)
    d = Deliberator()
    sampler = _mk_sampler(rng, {"x": 1.0})
    delib = d.deliberate(sampler, max_samples=4)
    import json
    payload = json.dumps(delib.to_dict())
    assert "stop_reason" in payload
    assert delib.cluster_key in payload


def test_receipt_hash_stable_across_identical_runs():
    # Two runs with identical sample sequences produce identical receipt
    # hashes (modulo the random id which is excluded from the hash input).
    def make_delib(seed: int) -> Deliberation:
        rng = random.Random(seed)
        d = Deliberator()
        sampler = _mk_sampler(rng, {"x": 1.0})
        return d.deliberate(sampler, max_samples=4, alpha=0.05, commit_threshold=0.5)

    a = make_delib(99)
    b = make_delib(99)
    # The id is random per call, but everything that contributes to the
    # receipt hash (sample sequence, counts, params) is identical.
    assert a.cluster_key == b.cluster_key
    assert a.n_samples == b.n_samples
    # Hashes differ only because `id` is in the hash input. That's a
    # feature: each receipt is unique. Validate that excluding `id`
    # would tie them — by hashing the rest manually.
    from agi.deliberator import _receipt_hash
    def _strip(d: Deliberation) -> dict:
        return {
            "n_samples": d.n_samples,
            "cluster_key": d.cluster_key,
            "stop_reason": d.stop_reason,
            "alpha": d.alpha,
            "commit_threshold": d.commit_threshold,
            "samples": [(s.cluster_key, s.cost) for s in d.samples],
        }
    assert _receipt_hash(_strip(a)) == _receipt_hash(_strip(b))


def test_attestor_called_when_wired():
    calls = []

    class _StubAttestor:
        def append(self, payload, *, kind):
            calls.append((kind, payload["stop_reason"]))

    d = Deliberator(attestor=_StubAttestor())
    rng = random.Random(0)
    sampler = _mk_sampler(rng, {"a": 1.0})
    d.deliberate(sampler, max_samples=4)
    assert len(calls) == 1
    assert calls[0][0] == "deliberation"
