"""Tests for agi.speculator — speculative execution as a runtime primitive."""
from __future__ import annotations

import math
import random
import unittest

from agi.speculator import (
    ALG_EAGLE,
    ALG_GREEDY,
    ALG_LEVIATHAN,
    ALG_LOOKAHEAD,
    ALG_MEDUSA_TREE,
    ALG_SELF_SPEC,
    ALG_SPEC_SAMPLING,
    InvalidConfig,
    InvalidDraft,
    InvalidTarget,
    KNOWN_ALGORITHMS,
    Speculator,
    SpeculatorConfig,
    SpeculatorReport,
    StepOutput,
    eagle_speculator,
    greedy_speculator,
    leviathan_speculator,
    lookahead_speculator,
    medusa_speculator,
    self_spec_speculator,
    speculative_sampling_speculator,
)


# =============================================================================
# Fixture helpers
# =============================================================================


def _peaky_target(k):
    """Target favors 'A' with p=0.7."""
    def _t(state, draft_tokens):
        return [("A", {"A": 0.7, "B": 0.2, "C": 0.1})
                for _ in range(len(draft_tokens) + 1)]
    return _t


def _matching_draft(seed=0):
    rng = random.Random(seed)
    def _d(state):
        return [("A", {"A": 0.5, "B": 0.3, "C": 0.2}) for _ in range(4)]
    return _d


def _random_draft(seed=0):
    rng = random.Random(seed)
    def _d(state):
        out = []
        for _ in range(4):
            u = rng.random()
            if u < 0.5: t = "A"
            elif u < 0.8: t = "B"
            else: t = "C"
            out.append((t, {"A": 0.5, "B": 0.3, "C": 0.2}))
        return out
    return _d


def _make_dist_uniform(tokens):
    p = 1.0 / len(tokens)
    return {t: p for t in tokens}


# =============================================================================
# Configuration
# =============================================================================


class TestConfig(unittest.TestCase):

    def test_known_algorithms(self):
        for alg in ("speculative_sampling", "leviathan_decoding", "greedy",
                    "medusa_tree", "self_spec_early_exit", "eagle",
                    "lookahead"):
            self.assertIn(alg, KNOWN_ALGORITHMS)

    def test_default_config_is_spec_sampling(self):
        cfg = SpeculatorConfig()
        self.assertEqual(cfg.algorithm, ALG_SPEC_SAMPLING)
        self.assertGreater(cfg.k_draft, 0)

    def test_invalid_algorithm(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(algorithm="not_real")

    def test_invalid_k_draft(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(k_draft=0)

    def test_invalid_costs(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(draft_cost=0)
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(target_cost=-1)

    def test_invalid_alpha(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(alpha=0)
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(alpha=1.5)

    def test_invalid_fail_on(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(equivalence_fail_on="ignore")

    def test_invalid_eagle_alpha(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(eagle_alpha=-0.1)
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(eagle_alpha=1.5)

    def test_invalid_skip_fraction(self):
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(skip_fraction=0.0)
        with self.assertRaises(InvalidConfig):
            SpeculatorConfig(skip_fraction=1.0)


# =============================================================================
# Single step semantics
# =============================================================================


class TestStep(unittest.TestCase):

    def test_step_emits_at_least_one_token(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        out = s.step(0, draft=_random_draft(0), target=_peaky_target(4))
        self.assertGreaterEqual(len(out.tokens), 1)

    def test_step_at_most_k_plus_one_tokens(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        out = s.step(0, draft=_random_draft(0), target=_peaky_target(4))
        self.assertLessEqual(len(out.tokens), 5)

    def test_step_accept_count_in_range(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        out = s.step(0, draft=_random_draft(0), target=_peaky_target(4))
        self.assertGreaterEqual(out.accept_count, 0)
        self.assertLessEqual(out.accept_count, out.n_proposed)

    def test_bonus_xor_correction(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        for i in range(50):
            out = s.step(i, draft=_random_draft(i), target=_peaky_target(4))
            # Either we got a bonus (all accepted) or a correction (one rejected).
            self.assertTrue(out.bonus_token_included or out.correction_was_sampled)

    def test_draft_empty_raises(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        def empty_draft(state):
            return []
        with self.assertRaises(InvalidDraft):
            s.step(0, draft=empty_draft, target=_peaky_target(4))

    def test_target_wrong_length_raises(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        def bad_target(state, draft_tokens):
            # Returns wrong number of verifications.
            return [("A", {"A": 1.0})]
        with self.assertRaises(InvalidTarget):
            s.step(0, draft=_random_draft(0), target=bad_target)

    def test_draft_zero_distribution_raises(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        def bad_draft(state):
            return [("A", {"A": 0.0, "B": 0.0})]
        with self.assertRaises(InvalidDraft):
            s.step(0, draft=bad_draft, target=_peaky_target(4))

    def test_draft_negative_probability_raises(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        def bad_draft(state):
            return [("A", {"A": -0.1, "B": 1.0})]
        with self.assertRaises(InvalidDraft):
            s.step(0, draft=bad_draft, target=_peaky_target(4))


# =============================================================================
# Acceptance + speedup statistics
# =============================================================================


class TestStatistics(unittest.TestCase):

    def test_accept_rate_high_when_draft_matches_target(self):
        # Draft equal to target — should accept ~always.
        def matching_draft(state):
            return [("A", {"A": 0.99, "B": 0.005, "C": 0.005})
                    for _ in range(4)]
        def matching_target(state, draft_tokens):
            return [("A", {"A": 0.99, "B": 0.005, "C": 0.005})
                    for _ in range(len(draft_tokens) + 1)]
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        for i in range(50):
            s.step(i, draft=matching_draft, target=matching_target)
        r = s.report()
        self.assertGreater(r.empirical_acceptance_rate, 0.9)
        self.assertGreater(r.expected_tokens_per_target_call, 4.0)

    def test_accept_rate_low_when_draft_disagrees(self):
        # Draft always picks 'C', target favours 'A' heavily.
        def bad_draft(state):
            return [("C", {"A": 0.05, "B": 0.05, "C": 0.9}) for _ in range(4)]
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        for i in range(80):
            s.step(i, draft=bad_draft, target=_peaky_target(4))
        r = s.report()
        # Acceptance rate should be much lower.
        self.assertLess(r.empirical_acceptance_rate, 0.5)

    def test_speedup_at_least_one(self):
        # Speedup should always be ≥ 1 in expectation — every step at least
        # emits 1 token.
        s = speculative_sampling_speculator(k_draft=4, seed=0,
                                            draft_cost=0.0001)
        for i in range(50):
            s.step(i, draft=_random_draft(i), target=_peaky_target(4))
        r = s.report()
        # With near-zero draft cost, speedup >= 1.0 always.
        self.assertGreaterEqual(r.empirical_speedup, 1.0 - 1e-6)

    def test_lcb_bounds_in_unit_interval(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        for i in range(80):
            s.step(i, draft=_random_draft(i), target=_peaky_target(4))
        r = s.report()
        self.assertGreaterEqual(r.empirical_acceptance_rate_lcb_hoeffding, 0.0)
        self.assertGreaterEqual(r.empirical_acceptance_rate_lcb_bernstein, 0.0)
        self.assertGreaterEqual(r.empirical_acceptance_rate_lcb_anytime, 0.0)
        self.assertLessEqual(r.empirical_acceptance_rate_ucb_hoeffding, 1.0)
        self.assertLessEqual(r.empirical_acceptance_rate_lcb_hoeffding,
                             r.empirical_acceptance_rate)
        self.assertGreaterEqual(r.empirical_acceptance_rate_ucb_hoeffding,
                                r.empirical_acceptance_rate)


# =============================================================================
# Greedy algorithm
# =============================================================================


class TestGreedy(unittest.TestCase):

    def test_greedy_accepts_when_argmax_matches(self):
        # Draft always picks 'A'; target argmax is 'A'.
        def good_draft(state):
            return [("A", {"A": 1.0, "B": 0.0}) for _ in range(4)]
        s = greedy_speculator(k_draft=4, seed=0)
        for i in range(20):
            s.step(i, draft=good_draft, target=_peaky_target(4))
        r = s.report()
        self.assertEqual(r.empirical_acceptance_rate, 1.0)

    def test_greedy_rejects_when_argmax_disagrees(self):
        # Draft picks 'C', target argmax is 'A'.
        def bad_draft(state):
            return [("C", {"A": 0.1, "C": 0.9}) for _ in range(4)]
        s = greedy_speculator(k_draft=4, seed=0)
        for i in range(20):
            out = s.step(i, draft=bad_draft, target=_peaky_target(4))
            # Should accept 0 and emit the argmax 'A'.
            self.assertEqual(out.accept_count, 0)
            self.assertEqual(out.tokens[0], "A")


# =============================================================================
# Replay determinism + fingerprint
# =============================================================================


class TestDeterminism(unittest.TestCase):

    def _run(self, seed=0):
        s = speculative_sampling_speculator(k_draft=4, seed=seed)
        # Use deterministic draft fn (state -> always same).
        def d(state):
            return [("A", {"A": 0.5, "B": 0.5}), ("B", {"A": 0.5, "B": 0.5}),
                    ("A", {"A": 0.5, "B": 0.5}), ("A", {"A": 0.5, "B": 0.5})]
        for _ in range(30):
            s.step(0, draft=d, target=_peaky_target(4))
        return s.fingerprint()

    def test_replay_determinism(self):
        fp1 = self._run(seed=42)
        fp2 = self._run(seed=42)
        self.assertEqual(fp1, fp2)

    def test_different_seed_changes_fingerprint(self):
        fp1 = self._run(seed=1)
        fp2 = self._run(seed=2)
        self.assertNotEqual(fp1, fp2)


# =============================================================================
# Report contents
# =============================================================================


class TestReport(unittest.TestCase):

    def test_report_fields(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        for i in range(20):
            s.step(i, draft=_random_draft(i), target=_peaky_target(4))
        r = s.report()
        self.assertIsInstance(r, SpeculatorReport)
        self.assertEqual(r.algorithm, ALG_SPEC_SAMPLING)
        self.assertGreater(r.n_steps, 0)
        self.assertGreater(r.n_proposed_total, 0)
        self.assertGreaterEqual(r.n_accepted_total, 0)
        self.assertEqual(len(r.fingerprint_hash), 64)
        self.assertGreater(r.chain_length, 0)
        self.assertGreaterEqual(r.elapsed_seconds, 0.0)

    def test_step_output_fields(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        out = s.step(0, draft=_random_draft(0), target=_peaky_target(4))
        self.assertIsInstance(out, StepOutput)
        self.assertGreater(len(out.tokens), 0)
        self.assertGreaterEqual(out.elapsed_seconds, 0.0)


# =============================================================================
# Reset
# =============================================================================


class TestReset(unittest.TestCase):

    def test_reset_clears_statistics(self):
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        for i in range(10):
            s.step(i, draft=_random_draft(i), target=_peaky_target(4))
        self.assertGreater(s.state()["n_steps"], 0)
        s.reset()
        self.assertEqual(s.state()["n_steps"], 0)


# =============================================================================
# All convenience constructors
# =============================================================================


class TestConstructors(unittest.TestCase):

    def test_each_constructor_works(self):
        for ctor, kw in [
            (speculative_sampling_speculator, {}),
            (leviathan_speculator, {}),
            (greedy_speculator, {}),
            (medusa_speculator, {}),
            (eagle_speculator, {}),
            (lookahead_speculator, {}),
            (self_spec_speculator, {}),
        ]:
            s = ctor(k_draft=2, seed=0, **kw)
            self.assertIsInstance(s, Speculator)
            # Each should be able to take one step.
            out = s.step(0, draft=_random_draft(0), target=_peaky_target(2))
            self.assertGreater(len(out.tokens), 0)


# =============================================================================
# Output-equivalence sanity (empirical Chi-squared on emitted marginal)
# =============================================================================


class TestEquivalence(unittest.TestCase):

    def test_empirical_marginal_matches_target(self):
        # When draft = target AND the draft actually samples from that
        # distribution, the emitted-token marginal should match target.
        # (Speculative sampling's equivalence guarantee requires that
        # the proposed token be sampled from the reported distribution.)
        target_dist = {"A": 0.6, "B": 0.3, "C": 0.1}
        rng_draft = random.Random(12345)

        def _sample(d):
            u = rng_draft.random()
            acc = 0.0
            for k, v in sorted(d.items()):
                acc += v
                if u <= acc:
                    return k
            return list(sorted(d.keys()))[-1]

        def draft(state):
            return [(_sample(target_dist), target_dist) for _ in range(4)]

        def target_fn(state, draft_tokens):
            return [("A", target_dist)
                    for _ in range(len(draft_tokens) + 1)]
        s = speculative_sampling_speculator(k_draft=4, seed=0)
        counts = {"A": 0, "B": 0, "C": 0}
        for i in range(500):
            out = s.step(i, draft=draft, target=target_fn)
            for tok in out.tokens:
                counts[tok] += 1
        total = sum(counts.values())
        # Marginal should be near (0.6, 0.3, 0.1) within 5% (loose CI).
        self.assertLess(abs(counts["A"] / total - 0.6), 0.06)
        self.assertLess(abs(counts["B"] / total - 0.3), 0.06)
        self.assertLess(abs(counts["C"] / total - 0.1), 0.05)


# =============================================================================
# Thread safety
# =============================================================================


class TestThreadSafety(unittest.TestCase):

    def test_concurrent_steps(self):
        import threading
        s = speculative_sampling_speculator(k_draft=4, seed=0)

        def worker(n):
            for i in range(n):
                s.step(i, draft=_random_draft(i), target=_peaky_target(4))

        threads = [threading.Thread(target=worker, args=(25,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(s.state()["n_steps"], 100)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
