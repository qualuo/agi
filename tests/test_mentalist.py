"""Tests for agi.mentalist — Bayesian theory-of-mind primitive."""
from __future__ import annotations

import math
import random
import unittest

from agi.mentalist import (
    InsufficientData,
    InvalidAgent,
    InvalidConfig,
    InvalidObservation,
    KNOWN_PREDICT_METHODS,
    MENTALIST_INFERRED,
    MENTALIST_OBSERVED,
    MENTALIST_PREDICTED,
    MENTALIST_REGISTERED,
    Mentalist,
    MentalistConfig,
    MentalReport,
    PACBayesBound,
    PREDICT_BAYES_AVG,
    PREDICT_MAP,
    PREDICT_SOFTMAX,
    PREDICT_THOMPSON,
    UnknownAgent,
    boltzmann_policy,
    clopper_pearson_ci,
    dirichlet_mean,
    hoeffding_half_width,
    kl_divergence,
    ledger_root,
    max_ent_irl,
    softmax,
)


class TestHelpers(unittest.TestCase):
    def test_softmax_uniform_at_zero_beta(self) -> None:
        out = softmax([1.0, 5.0, -3.0], beta=0.0)
        self.assertEqual(len(out), 3)
        for p in out:
            self.assertAlmostEqual(p, 1.0 / 3.0)

    def test_softmax_normalises(self) -> None:
        out = softmax([1.0, 2.0, 3.0], beta=1.5)
        self.assertAlmostEqual(sum(out), 1.0, places=10)
        # Monotone in scores at positive beta.
        self.assertLess(out[0], out[1])
        self.assertLess(out[1], out[2])

    def test_softmax_empty(self) -> None:
        self.assertEqual(softmax([]), [])

    def test_boltzmann_policy(self) -> None:
        dist = boltzmann_policy({"a": 1.0, "b": 2.0}, beta=2.0)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=10)
        self.assertGreater(dist["b"], dist["a"])

    def test_kl_divergence(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=10)
        q = {"a": 0.9, "b": 0.1}
        self.assertGreater(kl_divergence(p, q), 0.0)
        # Hard zero -> inf.
        self.assertEqual(kl_divergence({"a": 1.0}, {"a": 0.0, "b": 1.0}), math.inf)

    def test_dirichlet_mean(self) -> None:
        m = dirichlet_mean({"a": 2.0, "b": 8.0})
        self.assertAlmostEqual(m["a"], 0.2, places=10)
        self.assertAlmostEqual(m["b"], 0.8, places=10)

    def test_hoeffding_half_width(self) -> None:
        h_small = hoeffding_half_width(100)
        h_big = hoeffding_half_width(10000)
        self.assertGreater(h_small, h_big)
        self.assertEqual(hoeffding_half_width(0), math.inf)

    def test_clopper_pearson_edges(self) -> None:
        # 0/n: lower bound exact zero.
        lo, hi = clopper_pearson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        # n/n: upper bound exact one.
        lo, hi = clopper_pearson_ci(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 1.0)
        # 0 trials: full interval.
        self.assertEqual(clopper_pearson_ci(0, 0), (0.0, 1.0))

    def test_clopper_pearson_coverage(self) -> None:
        # Known textbook value: 5/10 @ 95% conf ≈ [0.187, 0.813].
        lo, hi = clopper_pearson_ci(5, 10, conf=0.95)
        self.assertAlmostEqual(lo, 0.187, places=2)
        self.assertAlmostEqual(hi, 0.813, places=2)

    def test_clopper_pearson_invalid(self) -> None:
        with self.assertRaises(InvalidConfig):
            clopper_pearson_ci(5, 3)
        with self.assertRaises(InvalidConfig):
            clopper_pearson_ci(-1, 10)
        with self.assertRaises(InvalidConfig):
            clopper_pearson_ci(1, 10, conf=1.0)

    def test_ledger_root_is_deterministic(self) -> None:
        self.assertEqual(ledger_root(), ledger_root())
        self.assertEqual(len(ledger_root()), 64)


class TestMentalistConfig(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        MentalistConfig()

    def test_invalid_prior_alpha(self) -> None:
        with self.assertRaises(InvalidConfig):
            MentalistConfig(prior_alpha=0.0)

    def test_invalid_confidence(self) -> None:
        with self.assertRaises(InvalidConfig):
            MentalistConfig(confidence=1.5)

    def test_invalid_pac_bayes_delta(self) -> None:
        with self.assertRaises(InvalidConfig):
            MentalistConfig(pac_bayes_delta=0.0)

    def test_invalid_irl_lr(self) -> None:
        with self.assertRaises(InvalidConfig):
            MentalistConfig(irl_lr=-1.0)

    def test_invalid_irl_l2(self) -> None:
        with self.assertRaises(InvalidConfig):
            MentalistConfig(irl_l2=-1e-3)


class TestRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Mentalist()

    def test_register_simple_agent(self) -> None:
        spec = self.m.register_agent(
            "alice",
            states=["s1", "s2"],
            actions=["a1", "a2"],
            outcomes=["win", "lose"],
        )
        self.assertEqual(spec.agent_id, "alice")
        self.assertIn("alice", self.m.known_agents())

    def test_double_register_raises(self) -> None:
        self.m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])
        with self.assertRaises(InvalidAgent):
            self.m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])

    def test_empty_schema_raises(self) -> None:
        with self.assertRaises(InvalidAgent):
            self.m.register_agent("a", states=[], actions=["x"], outcomes=["w"])
        with self.assertRaises(InvalidAgent):
            self.m.register_agent("b", states=["s"], actions=[], outcomes=["w"])
        with self.assertRaises(InvalidAgent):
            self.m.register_agent("c", states=["s"], actions=["x"], outcomes=[])

    def test_duplicate_labels_raises(self) -> None:
        with self.assertRaises(InvalidAgent):
            self.m.register_agent(
                "a", states=["s", "s"], actions=["x"], outcomes=["w"]
            )

    def test_remove_agent(self) -> None:
        self.m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])
        self.m.remove_agent("a")
        self.assertNotIn("a", self.m.known_agents())
        with self.assertRaises(UnknownAgent):
            self.m.remove_agent("a")

    def test_max_agents(self) -> None:
        cfg = MentalistConfig(max_agents=2)
        m = Mentalist(cfg)
        m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])
        m.register_agent("b", states=["s"], actions=["x"], outcomes=["w"])
        with self.assertRaises(InvalidConfig):
            m.register_agent("c", states=["s"], actions=["x"], outcomes=["w"])

    def test_prior_utility_validated(self) -> None:
        with self.assertRaises(InvalidAgent):
            self.m.register_agent(
                "a",
                states=["s"],
                actions=["x"],
                outcomes=["w"],
                prior_utility={"unknown_outcome": 1.0},
            )


class TestObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Mentalist()
        self.m.register_agent(
            "alice",
            states=["low", "high"],
            actions=["pass", "bid"],
            outcomes=["win", "lose"],
        )

    def test_basic_observe(self) -> None:
        self.m.observe("alice", state="low", action="pass", reward=0.0, outcome="lose")
        self.assertEqual(self.m.observation_count, 1)

    def test_unknown_state_raises(self) -> None:
        with self.assertRaises(InvalidObservation):
            self.m.observe("alice", state="???", action="pass", reward=0.0)

    def test_unknown_action_raises(self) -> None:
        with self.assertRaises(InvalidObservation):
            self.m.observe("alice", state="low", action="???", reward=0.0)

    def test_unknown_outcome_raises(self) -> None:
        with self.assertRaises(InvalidObservation):
            self.m.observe("alice", state="low", action="pass", reward=0.0, outcome="???")

    def test_unknown_agent_raises(self) -> None:
        with self.assertRaises(UnknownAgent):
            self.m.observe("nobody", state="low", action="pass", reward=0.0)

    def test_observe_batch(self) -> None:
        n = self.m.observe_batch(
            "alice",
            [
                ("low", "pass", 0.0, "lose"),
                ("high", "bid", 1.0, "win"),
                ("high", "bid", 1.0, "win"),
            ],
        )
        self.assertEqual(n, 3)
        self.assertEqual(self.m.observation_count, 3)

    def test_chain_head_changes_on_observation(self) -> None:
        head0 = self.m.chain_head
        self.m.observe("alice", state="low", action="pass", reward=0.0, outcome="lose")
        head1 = self.m.chain_head
        self.assertNotEqual(head0, head1)


class TestPrediction(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Mentalist(MentalistConfig(rng_seed=42))
        self.m.register_agent(
            "alice",
            states=["low", "high"],
            actions=["pass", "bid"],
            outcomes=["win", "lose"],
        )

    def test_predict_all_methods_normalise(self) -> None:
        for _ in range(8):
            self.m.observe("alice", state="low", action="pass", reward=0.0, outcome="lose")
            self.m.observe("alice", state="high", action="bid", reward=1.0, outcome="win")
        for method in KNOWN_PREDICT_METHODS:
            dist = self.m.predict("alice", state="high", method=method)
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
            for p in dist.values():
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0 + 1e-9)

    def test_predict_unknown_method_raises(self) -> None:
        with self.assertRaises(InvalidConfig):
            self.m.predict("alice", state="low", method="???")

    def test_predict_unknown_state_raises(self) -> None:
        with self.assertRaises(InvalidObservation):
            self.m.predict("alice", state="???")

    def test_predict_unknown_agent_raises(self) -> None:
        with self.assertRaises(UnknownAgent):
            self.m.predict("nobody", state="low")

    def test_predict_learns_a_useful_signal(self) -> None:
        # Agent strongly prefers bidding when high, passing when low.  A
        # few exploratory observations of the opposite action are needed
        # so MaxEnt IRL can break the symmetry between win/lose utility.
        for _ in range(30):
            self.m.observe("alice", state="low", action="pass", reward=1.0, outcome="win")
            self.m.observe("alice", state="high", action="bid", reward=1.0, outcome="win")
        for _ in range(3):
            self.m.observe("alice", state="low", action="bid", reward=-1.0, outcome="lose")
            self.m.observe("alice", state="high", action="pass", reward=-1.0, outcome="lose")
        # Force the IRL refit.
        self.m.infer_desire("alice", force=True)
        # After many observations the softmax policy should *prefer* bid at high.
        dist = self.m.predict("alice", state="high", method=PREDICT_SOFTMAX)
        self.assertGreater(dist["bid"], dist["pass"])
        # And prefer pass at low.
        dist_low = self.m.predict("alice", state="low", method=PREDICT_SOFTMAX)
        self.assertGreater(dist_low["pass"], dist_low["bid"])

    def test_predict_map_returns_onehot(self) -> None:
        for _ in range(4):
            self.m.observe("alice", state="low", action="pass", reward=0.0, outcome="lose")
        self.m.infer_desire("alice", force=True)
        dist = self.m.predict("alice", state="low", method=PREDICT_MAP)
        self.assertEqual(sum(1 for p in dist.values() if p > 0.0), 1)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=10)


class TestInverseRL(unittest.TestCase):
    def test_irl_infers_positive_utility_for_winning_outcome(self) -> None:
        m = Mentalist()
        m.register_agent(
            "alice",
            states=["s"],
            actions=["a", "b"],
            outcomes=["win", "lose"],
        )
        # Alice strongly prefers action 'a', which usually yields 'win'.
        # MaxEnt IRL needs an *asymmetric* action distribution to recover
        # utility; equal counts produce zero gradient by construction.
        for _ in range(30):
            m.observe("alice", state="s", action="a", reward=1.0, outcome="win")
        for _ in range(2):
            m.observe("alice", state="s", action="a", reward=-1.0, outcome="lose")
        for _ in range(2):
            m.observe("alice", state="s", action="b", reward=1.0, outcome="win")
        for _ in range(8):
            m.observe("alice", state="s", action="b", reward=-1.0, outcome="lose")
        desires = m.infer_desire("alice", force=True)
        self.assertGreater(desires["win"], desires["lose"])

    def test_irl_insufficient_data_raises(self) -> None:
        m = Mentalist()
        m.register_agent(
            "alice", states=["s"], actions=["a"], outcomes=["w", "l"]
        )
        with self.assertRaises(InsufficientData):
            m.infer_desire("alice")

    def test_max_ent_irl_centered_at_zero_mean(self) -> None:
        # Agent overwhelmingly prefers action 'a' (which mostly produces 'w');
        # the asymmetry in action counts is what makes MaxEnt identifiable.
        sao = {
            ("s", "a", "w"): 30,
            ("s", "a", "l"): 2,
            ("s", "b", "w"): 1,
            ("s", "b", "l"): 4,
        }
        theta, _hist = max_ent_irl(
            states=("s",),
            actions=("a", "b"),
            outcomes=("w", "l"),
            sao_counts=sao,
            beta=1.0,
            max_iters=400,
        )
        mean = sum(theta.values()) / len(theta)
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertGreater(theta["w"], theta["l"])


class TestExpectedUtility(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Mentalist()
        self.m.register_agent(
            "alice",
            states=["s"],
            actions=["a", "b"],
            outcomes=["win", "lose"],
        )
        # Asymmetric action choices: prefers 'a' overwhelmingly.
        for _ in range(30):
            self.m.observe("alice", state="s", action="a", reward=1.0, outcome="win")
        for _ in range(2):
            self.m.observe("alice", state="s", action="a", reward=-1.0, outcome="lose")
        for _ in range(2):
            self.m.observe("alice", state="s", action="b", reward=1.0, outcome="win")
        for _ in range(8):
            self.m.observe("alice", state="s", action="b", reward=-1.0, outcome="lose")
        self.m.infer_desire("alice", force=True)

    def test_expected_utility_orders_actions(self) -> None:
        eu = self.m.expected_utility("alice", state="s")
        self.assertGreater(eu["a"], eu["b"])

    def test_state_distribution_marginalised(self) -> None:
        dist = self.m.state_distribution("alice")
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=10)


class TestConfidence(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Mentalist()
        self.m.register_agent(
            "alice",
            states=["s1", "s2"],
            actions=["a", "b"],
            outcomes=["win", "lose"],
        )

    def test_confidence_after_observations(self) -> None:
        for _ in range(40):
            self.m.observe("alice", state="s1", action="a", reward=1.0, outcome="win")
        lo, hi = self.m.confidence("alice", state="s1", action="a")
        self.assertGreater(lo, 0.7)
        self.assertEqual(hi, 1.0)

    def test_confidence_marginal_across_states(self) -> None:
        for _ in range(10):
            self.m.observe("alice", state="s1", action="a", reward=1.0, outcome="win")
            self.m.observe("alice", state="s2", action="a", reward=0.0, outcome="lose")
        lo, hi = self.m.confidence("alice", action="a")  # marginal
        self.assertLess(lo, 0.7)
        self.assertGreater(hi, 0.3)


class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Mentalist(MentalistConfig(rng_seed=2024))
        self.m.register_agent(
            "alice",
            states=["a", "b"],
            actions=["x", "y"],
            outcomes=["win", "lose"],
        )
        for _ in range(20):
            self.m.observe("alice", state="a", action="x", reward=1.0, outcome="win")
            self.m.observe("alice", state="b", action="y", reward=1.0, outcome="win")
        self.m.infer_desire("alice", force=True)

    def test_simulate_returns_correct_length(self) -> None:
        traj = self.m.simulate("alice", start_state="a", horizon=5)
        self.assertEqual(len(traj), 5)
        for s, action in traj:
            self.assertIn(s, ("a", "b"))
            self.assertIn(action, ("x", "y"))

    def test_simulate_with_transition_callable(self) -> None:
        traj = self.m.simulate(
            "alice",
            start_state="a",
            horizon=4,
            transition=lambda s, a: "b" if s == "a" else "a",
            rng_seed=0,
        )
        # State alternates a, b, a, b.
        self.assertEqual([s for s, _ in traj], ["a", "b", "a", "b"])

    def test_simulate_invalid_horizon(self) -> None:
        with self.assertRaises(InvalidConfig):
            self.m.simulate("alice", start_state="a", horizon=0)


class TestNestedToM(unittest.TestCase):
    def test_nested_belief_uses_observer_evidence(self) -> None:
        m = Mentalist()
        m.register_agent(
            "alice",
            states=["s"],
            actions=["x", "y"],
            outcomes=["win", "lose"],
        )
        m.register_agent(
            "bob",
            states=["s"],
            actions=["x", "y"],
            outcomes=["win", "lose"],
        )
        # Bob has *seen* Alice pick y in s with positive reward.
        for _ in range(20):
            m.observe("bob", state="s", action="y", reward=1.0, outcome="win")
        # Bob's prediction of Alice should now favour y at state s.
        dist = m.nested_belief(observer="bob", target="alice", state="s")
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
        self.assertGreater(dist["y"], dist["x"])

    def test_nested_belief_disjoint_schema_raises(self) -> None:
        m = Mentalist()
        m.register_agent(
            "alice", states=["a"], actions=["x"], outcomes=["w"]
        )
        m.register_agent(
            "bob", states=["b"], actions=["y"], outcomes=["L"]
        )
        with self.assertRaises(InvalidObservation):
            m.nested_belief(observer="bob", target="alice", state="b")


class TestPACBayes(unittest.TestCase):
    def test_pac_bayes_bound_is_positive(self) -> None:
        m = Mentalist()
        m.register_agent(
            "alice",
            states=["s"],
            actions=["x", "y"],
            outcomes=["win", "lose"],
        )
        for _ in range(20):
            m.observe("alice", state="s", action="x", reward=1.0, outcome="win")
        bound = m.pac_bayes_bound("alice")
        self.assertIsInstance(bound, PACBayesBound)
        self.assertEqual(bound.delta, 0.05)
        self.assertGreater(bound.n, 0)
        self.assertGreaterEqual(bound.upper_bound, bound.empirical_log_loss - 1e-9)
        self.assertGreaterEqual(bound.kl_to_prior, 0.0)


class TestIdentifiability(unittest.TestCase):
    def test_identifiability_unique_when_distinguished(self) -> None:
        m = Mentalist()
        m.register_agent(
            "a", states=["s"], actions=["x", "y"], outcomes=["w", "l"]
        )
        m.observe("a", state="s", action="x", reward=1.0, outcome="w")
        m.observe("a", state="s", action="y", reward=0.0, outcome="l")
        report = m.identifiability("a")
        self.assertTrue(report.is_unique)

    def test_identifiability_block_when_indistinguishable(self) -> None:
        m = Mentalist()
        m.register_agent(
            "a", states=["s"], actions=["x"], outcomes=["w", "l"]
        )
        # Equal counts -> identical empirical p(o | s, a) signatures.
        m.observe("a", state="s", action="x", reward=1.0, outcome="w")
        m.observe("a", state="s", action="x", reward=0.0, outcome="l")
        report = m.identifiability("a")
        # The signatures are different (w sees 1/2, l sees 1/2) - they're
        # actually identical in this construction, so should be merged.
        # Note: equality is on the per-(s,a) conditional, so w has 0.5
        # and l also has 0.5 → SAME signature → merged.
        self.assertFalse(report.is_unique)


class TestReport(unittest.TestCase):
    def test_report_contains_all_fields(self) -> None:
        m = Mentalist()
        m.register_agent(
            "alice",
            states=["s"],
            actions=["x", "y"],
            outcomes=["w", "l"],
        )
        for _ in range(10):
            m.observe("alice", state="s", action="x", reward=1.0, outcome="w")
            m.observe("alice", state="s", action="y", reward=0.0, outcome="l")
        m.infer_desire("alice", force=True)
        r = m.report("alice")
        self.assertIsInstance(r, MentalReport)
        self.assertEqual(r.agent_id, "alice")
        self.assertEqual(r.n_observations, 20)
        self.assertAlmostEqual(sum(r.action_distribution.values()), 1.0, places=6)
        self.assertIn("x", r.confidence_intervals)
        self.assertEqual(len(r.certificate), 64)

    def test_report_unknown_agent(self) -> None:
        m = Mentalist()
        with self.assertRaises(UnknownAgent):
            m.report("nobody")


class TestExportImport(unittest.TestCase):
    def test_roundtrip(self) -> None:
        m1 = Mentalist(MentalistConfig(rng_seed=7))
        m1.register_agent(
            "alice",
            states=["s"],
            actions=["x", "y"],
            outcomes=["w", "l"],
        )
        for _ in range(10):
            m1.observe("alice", state="s", action="x", reward=1.0, outcome="w")
        snap = m1.export_state()
        m2 = Mentalist(MentalistConfig(rng_seed=7))
        m2.import_state(snap)
        d1 = m1.predict("alice", state="s", method=PREDICT_SOFTMAX)
        d2 = m2.predict("alice", state="s", method=PREDICT_SOFTMAX)
        for k in d1:
            self.assertAlmostEqual(d1[k], d2[k], places=10)

    def test_unsupported_version_raises(self) -> None:
        m = Mentalist()
        with self.assertRaises(InvalidConfig):
            m.import_state({"version": 999})

    def test_clear(self) -> None:
        m = Mentalist()
        m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])
        m.observe("a", state="s", action="x", reward=1.0, outcome="w")
        m.clear()
        self.assertEqual(m.known_agents(), [])
        self.assertEqual(m.observation_count, 0)
        self.assertEqual(m.chain_head, ledger_root())


class TestEventPublisher(unittest.TestCase):
    def test_events_emitted(self) -> None:
        events: list[tuple[str, dict]] = []
        m = Mentalist(publisher=lambda kind, data: events.append((kind, data)))
        m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])
        m.observe("a", state="s", action="x", reward=1.0, outcome="w")
        m.predict("a", state="s")
        kinds = {e[0] for e in events}
        self.assertIn(MENTALIST_REGISTERED, kinds)
        self.assertIn(MENTALIST_OBSERVED, kinds)
        self.assertIn(MENTALIST_PREDICTED, kinds)

    def test_publisher_failures_dont_crash(self) -> None:
        def bad_pub(_k: str, _d: dict) -> None:
            raise RuntimeError("boom")
        m = Mentalist(publisher=bad_pub)
        # Must not raise.
        m.register_agent("a", states=["s"], actions=["x"], outcomes=["w"])
        m.observe("a", state="s", action="x", reward=1.0, outcome="w")


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_chain(self) -> None:
        def play(seed: int) -> str:
            m = Mentalist(MentalistConfig(rng_seed=seed))
            m.register_agent("a", states=["s"], actions=["x", "y"], outcomes=["w", "l"])
            for _ in range(5):
                m.observe("a", state="s", action="x", reward=1.0, outcome="w")
                m.observe("a", state="s", action="y", reward=0.0, outcome="l")
            return m.chain_head
        # Same sequence -> same chain.
        self.assertEqual(play(42), play(42))


class TestPureStdlib(unittest.TestCase):
    """Mentalist must not depend on NumPy/Torch/SciPy."""

    def test_module_does_not_import_numpy(self) -> None:
        import agi.mentalist as mod
        with open(mod.__file__) as f:
            src = f.read()
        for forbidden in ("import numpy", "from numpy", "import torch", "import scipy"):
            self.assertNotIn(forbidden, src)


class TestIntegration(unittest.TestCase):
    """Smoke test the full workflow a coordination engine would use."""

    def test_full_workflow(self) -> None:
        m = Mentalist(MentalistConfig(rng_seed=99))
        m.register_agent(
            "trader",
            states=["bull", "bear", "flat"],
            actions=["buy", "sell", "hold"],
            outcomes=["profit", "loss", "even"],
        )
        rng = random.Random(0)
        # Synthetic trader: buys in bull, sells in bear, holds in flat.
        for _ in range(60):
            s = rng.choice(["bull", "bear", "flat"])
            if s == "bull":
                a, r, o = "buy", 1.0, "profit"
            elif s == "bear":
                a, r, o = "sell", 1.0, "profit"
            else:
                a, r, o = "hold", 0.0, "even"
            m.observe("trader", state=s, action=a, reward=r, outcome=o)
        m.infer_desire("trader", force=True)
        # Predictions should be tightly peaked on the right action in each state.
        for s, expected in (("bull", "buy"), ("bear", "sell"), ("flat", "hold")):
            dist = m.predict("trader", state=s, method=PREDICT_SOFTMAX)
            best = max(dist, key=lambda a: dist[a])
            self.assertEqual(best, expected, f"state={s} -> got {best} expected {expected}")
        # Capability CI should be high & tight on the chosen action.
        lo, hi = m.confidence("trader", action="buy", state="bull")
        self.assertGreater(lo, 0.5)
        # Report should bundle everything.
        r = m.report("trader")
        self.assertEqual(r.n_observations, 60)
        self.assertEqual(len(r.certificate), 64)


if __name__ == "__main__":
    unittest.main()
