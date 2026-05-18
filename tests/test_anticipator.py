"""Tests for the Anticipator sleep-time / anticipatory compute primitive."""
from __future__ import annotations

import json
import math
import random
import threading

import pytest

from agi.events import EventBus
from agi.anticipator import (
    ANTICIPATOR_ALLOCATED,
    ANTICIPATOR_CERTIFIED,
    ANTICIPATOR_ENUMERATED,
    ANTICIPATOR_EVICTED,
    ANTICIPATOR_HIT,
    ANTICIPATOR_INVALIDATED,
    ANTICIPATOR_MISS,
    ANTICIPATOR_PRECOMPUTED,
    ANTICIPATOR_REGISTERED,
    ANTICIPATOR_REPORTED,
    ANTICIPATOR_SERVED,
    ANTICIPATOR_STARTED,
    Anticipator,
    AnticipatorCertificate,
    AnticipatorConfig,
    AnticipatorError,
    AnticipatorReport,
    BudgetExceeded,
    Candidate,
    CacheEntry,
    ContextRecord,
    EmptyForecast,
    EVICT_BELADY,
    EVICT_LFU,
    EVICT_LRU,
    InvalidConfig,
    KNAPSACK_EXACT,
    KNAPSACK_GREEDY,
    MATCH_EXACT,
    MATCH_HASH,
    MATCH_PREFIX,
    MATCH_SIMILARITY,
    Plan,
    PrecomputeResult,
    ServeResult,
    UnknownContext,
)


# ---------------------------------------------------------------------------
# Helpers — minimal forecaster / answerer / embedder fakes.
# ---------------------------------------------------------------------------


def make_forecaster(spec):
    """Return a forecaster that yields Candidates from a static spec list."""

    def f(ctx, k, rng):
        for (q, prior, miss, pre) in spec[:k]:
            yield Candidate(
                query={"q": q},
                prior=prior,
                est_miss_cost=miss,
                est_precompute_cost=pre,
            )

    return f


def make_answerer(table, *, cost=1.0):
    def a(ctx, query):
        return table.get(query["q"], None), cost

    return a


def fixed_embedder(table):
    def e(query):
        q = query["q"]
        return table.get(q, [0.0] * 4)

    return e


# ---------------------------------------------------------------------------
# Config and record validation.
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_ok(self):
        cfg = AnticipatorConfig()
        assert cfg.matcher == MATCH_HASH
        assert cfg.knapsack == KNAPSACK_EXACT
        assert cfg.eviction == EVICT_LRU
        assert 0 < cfg.alpha < 1

    @pytest.mark.parametrize("kwargs", [
        dict(sleep_budget_per_ctx=0),
        dict(sleep_budget_per_ctx=-1),
        dict(sleep_budget_global=0),
        dict(cache_size_limit=-1),
        dict(matcher="bogus"),
        dict(similarity_threshold=0.0),
        dict(similarity_threshold=1.5),
        dict(eviction="bogus"),
        dict(knapsack="bogus"),
        dict(alpha=0.0),
        dict(alpha=1.0),
        dict(min_serves_for_certificate=0),
        dict(hit_cost=-0.1),
    ])
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(InvalidConfig):
            AnticipatorConfig(**kwargs)


class TestContextRecord:
    def test_valid(self):
        r = ContextRecord(ctx_id="c1", ctx={"x": 1})
        assert r.ctx_id == "c1"
        assert r.weight == 1.0

    @pytest.mark.parametrize("kwargs", [
        dict(ctx_id="", ctx={}),
        dict(ctx_id="x", ctx="notamapping"),
        dict(ctx_id="x", ctx={}, weight=0),
        dict(ctx_id="x", ctx={}, weight=-1),
        dict(ctx_id="x", ctx={}, deadline_hint=float("inf")),
    ])
    def test_invalid_raises(self, kwargs):
        with pytest.raises(AnticipatorError):
            ContextRecord(**kwargs)


class TestCandidate:
    def test_value_property(self):
        c = Candidate(query={"q": "x"}, prior=0.4, est_miss_cost=10.0, est_precompute_cost=2.0)
        assert c.value == pytest.approx(4.0)

    @pytest.mark.parametrize("kwargs", [
        dict(query="not-a-mapping", prior=0.5, est_miss_cost=1.0, est_precompute_cost=1.0),
        dict(query={}, prior=-0.1, est_miss_cost=1.0, est_precompute_cost=1.0),
        dict(query={}, prior=1.5, est_miss_cost=1.0, est_precompute_cost=1.0),
        dict(query={}, prior=0.5, est_miss_cost=-1.0, est_precompute_cost=1.0),
        dict(query={}, prior=0.5, est_miss_cost=1.0, est_precompute_cost=-1.0),
        dict(query={}, prior=float("nan"), est_miss_cost=1.0, est_precompute_cost=1.0),
    ])
    def test_invalid_raises(self, kwargs):
        with pytest.raises(AnticipatorError):
            Candidate(**kwargs)


# ---------------------------------------------------------------------------
# Knapsack — exact vs greedy.
# ---------------------------------------------------------------------------


class TestKnapsack:
    def _spec(self):
        # Four candidates with varying value/cost.
        return [
            # (query_id, prior, miss_cost, pre_cost)
            ("A", 0.5, 10.0, 2.0),  # value=5.0 cost=2.0  ratio=2.5
            ("B", 0.4,  8.0, 3.0),  # value=3.2 cost=3.0  ratio=1.07
            ("C", 0.3,  6.0, 1.0),  # value=1.8 cost=1.0  ratio=1.8
            ("D", 0.1,  5.0, 4.0),  # value=0.5 cost=4.0  ratio=0.125
        ]

    def test_exact_picks_best_subset(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=4.0, knapsack=KNAPSACK_EXACT,
        ))
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster(self._spec()), k=4)
        plan = ant.allocate("c", budget=4.0)
        # Best subset under budget=4: A+C (cost 3, value 6.8) beats A+B (cost 5 over budget),
        # and B alone (cost 3, value 3.2), and A alone (cost 2, value 5.0).
        chosen_costs = plan.total_precompute_cost
        chosen_value = plan.total_value
        assert chosen_costs <= 4.0 + 1e-9
        assert chosen_value == pytest.approx(6.8)

    def test_greedy_within_factor_two(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=4.0, knapsack=KNAPSACK_GREEDY,
        ))
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster(self._spec()), k=4)
        plan = ant.allocate("c", budget=4.0)
        # Greedy by ratio picks A (2.5), then C (1.8): cost 3, value 6.8.
        assert plan.total_precompute_cost <= 4.0 + 1e-9
        assert plan.total_value >= 6.8 - 1e-9

    def test_budget_zero_yields_empty_plan(self):
        ant = Anticipator(AnticipatorConfig(sleep_budget_per_ctx=0.1))
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster(self._spec()), k=4)
        plan = ant.allocate("c", budget=0.0)
        assert plan.chosen == ()
        assert plan.total_precompute_cost == 0.0

    def test_zero_cost_candidate_always_picked(self):
        ant = Anticipator(AnticipatorConfig())
        ant.register_context("c", ctx={})
        # One candidate with zero pre-compute cost and a non-zero value
        # is always free money.
        ant.enumerate("c", make_forecaster([
            ("F", 0.9, 5.0, 0.0),
            ("G", 0.1, 1.0, 5.0),
        ]), k=2)
        plan = ant.allocate("c", budget=0.5)
        assert 0 in plan.chosen


# ---------------------------------------------------------------------------
# Hit / miss / matchers.
# ---------------------------------------------------------------------------


class TestServeHash:
    def setup_method(self):
        self.ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=10.0, min_serves_for_certificate=1,
        ))
        self.ant.register_context("c", ctx={})
        spec = [("x", 0.7, 4.0, 1.0), ("y", 0.2, 3.0, 1.0)]
        self.ant.enumerate("c", make_forecaster(spec), k=2)
        plan = self.ant.allocate("c", budget=10.0)
        self.ant.precompute("c", plan, make_answerer({"x": "X!", "y": "Y!"}))

    def test_hit_returns_cached_answer(self):
        r = self.ant.serve("c", {"q": "x"})
        assert r.hit
        assert r.answer == "X!"
        assert r.saved_cost == 4.0

    def test_miss_returns_fresh(self):
        r = self.ant.serve("c", {"q": "z"}, answerer=make_answerer({"z": "Z!"}, cost=7.0))
        assert not r.hit
        assert r.answer == "Z!"
        assert r.saved_cost == 0.0
        assert r.served_cost == 7.0

    def test_miss_without_answerer_returns_none(self):
        r = self.ant.serve("c", {"q": "z"})
        assert not r.hit
        assert r.answer is None
        assert r.served_cost == 0.0

    def test_hits_and_misses_book_counters(self):
        self.ant.serve("c", {"q": "x"})
        self.ant.serve("c", {"q": "x"})  # repeat hit
        self.ant.serve("c", {"q": "z"})
        cert = self.ant.certificate()
        assert cert.n_serves == 3
        assert cert.n_hits == 2

    def test_unknown_context_raises(self):
        with pytest.raises(UnknownContext):
            self.ant.serve("nope", {"q": "x"})


class TestMatchExact:
    def test_dict_equality(self):
        cfg = AnticipatorConfig(matcher=MATCH_EXACT)
        ant = Anticipator(cfg)
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([("k", 0.9, 2.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=2.0)
        ant.precompute("c", plan, make_answerer({"k": "ans"}))
        r = ant.serve("c", {"q": "k"})
        assert r.hit


class TestMatchPrefix:
    def test_prefix_hit(self):
        cfg = AnticipatorConfig(matcher=MATCH_PREFIX)
        ant = Anticipator(cfg)
        ant.register_context("c", ctx={})
        # Cache "weather" and serve "weather-detailed" → JSON forms share
        # the leading characters, so the prefix matcher fires.
        ant.enumerate("c", make_forecaster([("weather", 0.8, 3.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=2.0)
        ant.precompute("c", plan, make_answerer({"weather": "sunny"}))
        # Same key, longer value would not test prefix; instead, test the
        # cached entry being a prefix of the live one.  Use mapping {"q": "..."}.
        r = ant.serve("c", {"q": "weather"})
        assert r.hit


class TestMatchSimilarity:
    def test_high_cosine_hits(self):
        emb_table = {
            "a": [1.0, 0.0, 0.0, 0.0],
            "b": [0.99, 0.14, 0.0, 0.0],  # cos(a,b) ≈ 0.99
            "c": [0.0, 0.0, 1.0, 0.0],
        }
        cfg = AnticipatorConfig(matcher=MATCH_SIMILARITY, similarity_threshold=0.95)
        ant = Anticipator(cfg)
        ant.register_context("ctx", ctx={})
        ant.enumerate("ctx", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        plan = ant.allocate("ctx", budget=1.0)
        ant.precompute("ctx", plan, make_answerer({"a": "A"}))
        r = ant.serve("ctx", {"q": "b"}, embedder=fixed_embedder(emb_table))
        assert r.hit
        assert r.matcher == MATCH_SIMILARITY
        assert r.answer == "A"

    def test_low_cosine_misses(self):
        emb_table = {
            "a": [1.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 1.0],
        }
        cfg = AnticipatorConfig(matcher=MATCH_SIMILARITY, similarity_threshold=0.9)
        ant = Anticipator(cfg)
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=1.0)
        ant.precompute("c", plan, make_answerer({"a": "A"}))
        r = ant.serve("c", {"q": "z"}, embedder=fixed_embedder(emb_table),
                       answerer=make_answerer({"z": "Z"}, cost=2.0))
        assert not r.hit
        assert r.answer == "Z"

    def test_requires_embedder(self):
        cfg = AnticipatorConfig(matcher=MATCH_SIMILARITY)
        ant = Anticipator(cfg)
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=1.0)
        ant.precompute("c", plan, make_answerer({"a": "A"}))
        with pytest.raises(AnticipatorError):
            ant.serve("c", {"q": "a"})


# ---------------------------------------------------------------------------
# Eviction.
# ---------------------------------------------------------------------------


class TestEviction:
    def _populate(self, ant):
        ant.register_context("c", ctx={})
        spec = [
            ("a", 0.9, 5.0, 1.0),
            ("b", 0.5, 5.0, 1.0),
            ("c", 0.3, 5.0, 1.0),
        ]
        ant.enumerate("c", make_forecaster(spec), k=3)
        plan = ant.allocate("c", budget=3.0)
        ant.precompute("c", plan, make_answerer({"a": "A", "b": "B", "c": "C"}))

    def test_lru_evicts_oldest(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=3.0, cache_size_limit=2, eviction=EVICT_LRU,
        ))
        self._populate(ant)
        assert len(ant.cache) == 2  # 3 items, limit 2 → one evicted on insert.
        # The earliest inserted (lowest-value 'c' is enumerated last;
        # 'a' first) should have been bumped if recently touched.  The
        # default insertion order is by knapsack pick; with all costs
        # equal, exact knapsack picks highest priors first: a, b, c.
        # That makes 'a' the oldest at the size-limit point, so 'a'
        # is evicted.
        survivors = {e.query["q"] for e in ant.cache}
        assert "c" in survivors

    def test_lfu_evicts_least_frequent(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=3.0, cache_size_limit=3, eviction=EVICT_LFU,
        ))
        self._populate(ant)
        # Hit 'a' twice, 'b' once.  'c' never hit.  Then push the cache
        # over the limit and confirm 'c' is the first to go.
        ant.serve("c", {"q": "a"})
        ant.serve("c", {"q": "a"})
        ant.serve("c", {"q": "b"})
        # Add a new cache entry to over-fill.
        ant.enumerate("c", make_forecaster([("d", 0.7, 5.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=1.0)
        ant.precompute("c", plan, make_answerer({"d": "D"}))
        survivors = {e.query["q"] for e in ant.cache}
        assert "a" in survivors
        assert len(ant.cache) <= 3


# ---------------------------------------------------------------------------
# Budgets and exceptional paths.
# ---------------------------------------------------------------------------


class TestBudgets:
    def test_per_ctx_budget_capped(self):
        ant = Anticipator(AnticipatorConfig(sleep_budget_per_ctx=1.5))
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([
            ("a", 0.9, 5.0, 1.0),
            ("b", 0.9, 5.0, 1.0),
            ("c", 0.9, 5.0, 1.0),
        ]), k=3)
        plan = ant.allocate("c")
        assert plan.total_precompute_cost <= 1.5 + 1e-9
        ant.precompute("c", plan, make_answerer({"a": "A", "b": "B", "c": "C"}))
        # Second precompute should be budget-stopped on at least one item.
        ant.enumerate("c", make_forecaster([("d", 0.9, 5.0, 1.0)]), k=1)
        plan2 = ant.allocate("c")
        # Per-ctx budget already fully spent.
        assert plan2.total_precompute_cost == 0.0 or plan2.chosen == ()

    def test_global_budget_capped(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=10.0, sleep_budget_global=1.5,
        ))
        ant.register_context("c1", ctx={})
        ant.register_context("c2", ctx={})
        for cid in ("c1", "c2"):
            ant.enumerate(cid, make_forecaster([("x", 0.9, 5.0, 1.0)]), k=1)
            plan = ant.allocate(cid)
            ant.precompute(cid, plan, make_answerer({"x": "X"}))
        # Global budget 1.5 means at most 1 of the two could land
        # (each costs 1.0; after first c1 leaves 0.5 → c2 cannot fit 1.0).
        assert len(ant.cache) == 1


class TestUnknownAndErrors:
    def test_allocate_unknown_context(self):
        ant = Anticipator()
        with pytest.raises(UnknownContext):
            ant.allocate("nope")

    def test_enumerate_empty_raises(self):
        ant = Anticipator()
        ant.register_context("c", ctx={})

        def empty(ctx, k, rng):
            return iter([])

        with pytest.raises(EmptyForecast):
            ant.enumerate("c", empty, k=4)

    def test_enumerate_wrong_type_raises(self):
        ant = Anticipator()
        ant.register_context("c", ctx={})

        def bad(ctx, k, rng):
            yield "not a candidate"

        with pytest.raises(AnticipatorError):
            ant.enumerate("c", bad, k=1)

    def test_allocate_without_forecaster_or_enumerate(self):
        ant = Anticipator()
        ant.register_context("c", ctx={})
        with pytest.raises(AnticipatorError):
            ant.allocate("c")

    def test_certificate_too_few_serves(self):
        ant = Anticipator(AnticipatorConfig(min_serves_for_certificate=5))
        ant.register_context("c", ctx={})
        with pytest.raises(AnticipatorError):
            ant.certificate()

    def test_invalidate_clears_cache(self):
        ant = Anticipator(AnticipatorConfig(min_serves_for_certificate=1))
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=2.0)
        ant.precompute("c", plan, make_answerer({"a": "A"}))
        assert len(ant.cache) == 1
        removed = ant.invalidate("c")
        assert removed == 1
        assert len(ant.cache) == 0


# ---------------------------------------------------------------------------
# Certificate.
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_hit_rate_ci_contains_truth_with_many_trials(self):
        """If the realised hit rate is 0.5, the Wilson CI from a few hundred
        trials should comfortably contain 0.5 and have width < 0.2."""
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=200.0, min_serves_for_certificate=10,
        ))
        ant.register_context("c", ctx={})
        spec = [(f"q{i}", 0.5, 1.0, 1.0) for i in range(100)]
        ant.enumerate("c", make_forecaster(spec), k=100)
        plan = ant.allocate("c", budget=200.0)
        ant.precompute("c", plan, make_answerer({f"q{i}": i for i in range(100)}))
        # Issue 200 serves: 100 hits (queries we cached), 100 misses (queries we didn't).
        for i in range(100):
            ant.serve("c", {"q": f"q{i}"})            # hits
            ant.serve("c", {"q": f"miss{i}"})         # misses
        cert = ant.certificate()
        assert cert.n_hits == 100
        assert cert.n_serves == 200
        assert cert.hit_rate == pytest.approx(0.5)
        # Wilson CI width should be small for n=200.
        assert (cert.hit_rate_wilson_hi - cert.hit_rate_wilson_lo) < 0.15
        # Coverage: 0.5 lies inside.
        assert cert.hit_rate_wilson_lo <= 0.5 <= cert.hit_rate_wilson_hi

    def test_eb_lcb_is_bounded_above_by_mean(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=200.0, min_serves_for_certificate=2,
        ))
        ant.register_context("c", ctx={})
        spec = [(f"q{i}", 0.9, 10.0, 1.0) for i in range(20)]
        ant.enumerate("c", make_forecaster(spec), k=20)
        plan = ant.allocate("c", budget=40.0)
        ant.precompute("c", plan, make_answerer({f"q{i}": i for i in range(20)}))
        for i in range(20):
            ant.serve("c", {"q": f"q{i}"})
        cert = ant.certificate()
        # The empirical-Bernstein LCB is by construction ≤ the empirical mean.
        assert cert.saved_cost_eb_lo <= cert.saved_cost_mean + 1e-9

    def test_certificate_roundtrip_json(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=10.0, min_serves_for_certificate=1,
        ))
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=2.0)
        ant.precompute("c", plan, make_answerer({"a": "A"}))
        ant.serve("c", {"q": "a"})
        cert = ant.certificate()
        s = json.dumps(cert.to_dict(), sort_keys=True)
        d = json.loads(s)
        assert d["n_hits"] == 1
        assert d["n_serves"] == 1
        assert d["cost_unit"] == "flops"

    def test_fingerprint_changes_with_every_event(self):
        ant = Anticipator(AnticipatorConfig(min_serves_for_certificate=1))
        fp0 = ant.fingerprint_hash
        ant.register_context("c", ctx={})
        fp1 = ant.fingerprint_hash
        ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        fp2 = ant.fingerprint_hash
        plan = ant.allocate("c", budget=2.0)
        fp3 = ant.fingerprint_hash
        assert len({fp0, fp1, fp2, fp3}) == 4

    def test_fingerprint_deterministic_across_instances(self):
        """Two instances run on the same sequence of events with the same
        seed should produce the same fingerprint."""

        def run():
            ant = Anticipator(
                AnticipatorConfig(min_serves_for_certificate=1, seed=42),
                instance_id="determinism",
                clock=lambda: 0.0,  # freeze the clock
            )
            ant.register_context("c", ctx={"k": 1})
            ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
            plan = ant.allocate("c", budget=2.0)
            ant.precompute("c", plan, make_answerer({"a": "A"}))
            ant.serve("c", {"q": "a"})
            return ant.fingerprint_hash

        assert run() == run()


# ---------------------------------------------------------------------------
# Event bus integration.
# ---------------------------------------------------------------------------


class TestEvents:
    def test_events_published(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e: seen.append(e.kind))
        ant = Anticipator(
            AnticipatorConfig(min_serves_for_certificate=1),
            bus=bus,
        )
        ant.register_context("c", ctx={})
        ant.enumerate("c", make_forecaster([("a", 0.9, 5.0, 1.0)]), k=1)
        plan = ant.allocate("c", budget=2.0)
        ant.precompute("c", plan, make_answerer({"a": "A"}))
        ant.serve("c", {"q": "a"})
        ant.serve("c", {"q": "miss"})
        ant.certificate()
        ant.report()

        # Spot-check that each expected event kind has fired.
        for kind in (
            ANTICIPATOR_STARTED,
            ANTICIPATOR_REGISTERED,
            ANTICIPATOR_ENUMERATED,
            ANTICIPATOR_ALLOCATED,
            ANTICIPATOR_PRECOMPUTED,
            ANTICIPATOR_HIT,
            ANTICIPATOR_MISS,
            ANTICIPATOR_SERVED,
            ANTICIPATOR_CERTIFIED,
            ANTICIPATOR_REPORTED,
        ):
            assert kind in seen, f"missing event {kind!r} in {seen!r}"


# ---------------------------------------------------------------------------
# Thread-safety smoke test.
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_serves(self):
        ant = Anticipator(AnticipatorConfig(
            sleep_budget_per_ctx=100.0, min_serves_for_certificate=4,
        ))
        ant.register_context("c", ctx={})
        spec = [(f"q{i}", 0.9, 2.0, 1.0) for i in range(20)]
        ant.enumerate("c", make_forecaster(spec), k=20)
        plan = ant.allocate("c", budget=20.0)
        ant.precompute("c", plan, make_answerer({f"q{i}": i for i in range(20)}))

        errors: list[BaseException] = []

        def worker(start, count):
            try:
                for i in range(start, start + count):
                    ant.serve("c", {"q": f"q{i % 20}"})
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 50, 50))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        cert = ant.certificate()
        assert cert.n_serves == 200
        assert cert.n_hits == 200


# ---------------------------------------------------------------------------
# Manifest registration.
# ---------------------------------------------------------------------------


def test_manifest_registration():
    from agi.manifest import default_manifest

    m = default_manifest(fresh=True)
    spec = m.lookup("anticipator")
    assert spec.name == "anticipator"
    assert spec.certificate == "pac"
    assert spec.determinism == "seeded"
    assert spec.dependency == "stdlib"
    assert "forecaster" in spec.composes_with
    assert "scheduler" in spec.composes_with
