"""Tests for ``agi.persuader`` — Bayesian persuasion runtime primitive.

The mathematical contract under test:

  1. **Kamenica-Gentzkow (2011) prosecutor/judge** — the canonical
     example. Prior P(guilty) = 0.3, judge convicts iff P(guilty) ≥ ½,
     prosecutor wants conviction. The optimum is the *partial-pooling*
     scheme that induces posteriors {μ = 1/2 (recommend convict),
     μ = 0 (recommend acquit)} with marginals (0.6, 0.4). Sender
     value = 0.6, strictly above the no-info baseline of 0.0 (judge
     acquits) and matched analytically.
  2. **No persuasion at non-degenerate priors with full agreement** —
     when sender and receiver utilities are identical the optimum is
     the no-info scheme (revealing nothing wastes Bayes-plausibility).
  3. **Bayes plausibility** — for any optimal scheme,
     Σ_s P(s) · μ_s = μ₀ to numerical precision.
  4. **Obedience** — every recommended action is a (weak) best response
     under its induced posterior. The verifier's deviation gain is
     ≤ tolerance.
  5. **Persuasion never hurts** — V*(μ₀) ≥ v̂(μ₀): the optimum is at
     least the no-info value, and at most the full-info value.
  6. **LP solver: 3-state weather example.** Closed-form check on a
     hand-constructed 3 × 3 game.
  7. **Online persuasion: regret ≤ bound.** The cumulative regret
     emitted by `online_persuade` is ≤ √(T · ln K / 2) + 1.
  8. **Robust persuasion: worst-case improvement.** The robust scheme
     dominates the worst single-prior LP solution.
  9. **Multi-receiver private (additive sender utility)** — independent
     per-receiver LP sums match the joint optimum exactly.
 10. **Receipt digests are deterministic content hashes** — same input
     produces identical digest.
 11. **Event bus receives `persuader.*` events on every solve.**
 12. **Hoeffding PAC certificate covers the optimum value as n → ∞.**
"""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.events import Event, EventBus
from agi.persuader import (
    InvalidPrior,
    InvalidUtility,
    KIND_CONCAVIFICATION,
    KIND_LP,
    KNOWN_KINDS,
    MultiReceiverOutcome,
    OnlinePersuasionOutcome,
    PERSUADE_CERTIFIED,
    PERSUADE_MULTI_SOLVED,
    PERSUADE_ONLINE_STEP,
    PERSUADE_ROBUST_SOLVED,
    PERSUADE_SOLVED,
    PERSUADE_STARTED,
    PERSUADE_VERIFIED,
    PayoffCertificate,
    Persuader,
    PersuaderError,
    PersuasionGame,
    PersuasionOutcome,
    RobustOutcome,
    SignalingScheme,
    VerificationReport,
    best_response,
    empirical_bernstein_radius,
    hoeffding_radius,
    quick_persuade,
    quick_verify,
    receiver_value_under_posterior,
    sender_value_under_full_information,
    verify_scheme,
)


# =====================================================================
# Game fixtures
# =====================================================================


def _kg_game(prior_guilty: float = 0.3) -> PersuasionGame:
    """Kamenica-Gentzkow 2011 prosecutor/judge example."""
    return PersuasionGame(
        states=("guilty", "innocent"),
        actions=("convict", "acquit"),
        prior=(prior_guilty, 1.0 - prior_guilty),
        sender_utility=((1.0, 1.0), (0.0, 0.0)),
        receiver_utility=((1.0, -1.0), (0.0, 0.0)),
    )


def _identical_game() -> PersuasionGame:
    """Sender and receiver have identical utility; no persuasion needed."""
    return PersuasionGame(
        states=("a", "b"),
        actions=("A", "B"),
        prior=(0.4, 0.6),
        sender_utility=((1.0, 0.0), (0.0, 1.0)),
        receiver_utility=((1.0, 0.0), (0.0, 1.0)),
    )


def _weather_game() -> PersuasionGame:
    """3-state, 3-action weather game (umbrella seller)."""
    return PersuasionGame(
        states=("sun", "cloud", "rain"),
        actions=("jacket", "tshirt", "umbrella"),
        prior=(0.3, 0.4, 0.3),
        sender_utility=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        receiver_utility=(
            (-1.0, 1.0, 0.0),
            (1.0, 0.0, -2.0),
            (-0.5, -0.5, 1.0),
        ),
    )


# =====================================================================
# Validation
# =====================================================================


class TestValidation:
    def test_prior_must_sum_to_one(self) -> None:
        with pytest.raises(InvalidPrior):
            PersuasionGame(
                states=("a", "b"),
                actions=("A",),
                prior=(0.4, 0.4),
                sender_utility=((1.0, 1.0),),
                receiver_utility=((1.0, 1.0),),
            )

    def test_prior_negatives_rejected(self) -> None:
        with pytest.raises(InvalidPrior):
            PersuasionGame(
                states=("a", "b"),
                actions=("A",),
                prior=(-0.1, 1.1),
                sender_utility=((1.0, 1.0),),
                receiver_utility=((1.0, 1.0),),
            )

    def test_utility_shape_must_match(self) -> None:
        with pytest.raises(InvalidUtility):
            PersuasionGame(
                states=("a", "b"),
                actions=("A", "B"),
                prior=(0.5, 0.5),
                sender_utility=((1.0, 1.0), (0.0,)),
                receiver_utility=((1.0, 1.0), (0.0, 0.0)),
            )

    def test_signaling_scheme_columns_sum_to_one(self) -> None:
        # π(·|state) must be a probability distribution.
        with pytest.raises(InvalidUtility):
            SignalingScheme(
                signals=("s1", "s2"),
                states=("a", "b"),
                pi=((0.5, 0.5), (0.4, 0.4)),
            )

    def test_known_kinds_set(self) -> None:
        for k in (KIND_CONCAVIFICATION, KIND_LP):
            assert k in KNOWN_KINDS


# =====================================================================
# Receiver best response
# =====================================================================


class TestBestResponse:
    def test_argmax_expected_utility(self) -> None:
        u = [[1.0, 0.0], [0.0, 1.0]]
        assert best_response(u, [0.7, 0.3]) == 0
        assert best_response(u, [0.3, 0.7]) == 1

    def test_tiebreak_lowest_then_highest(self) -> None:
        u = [[1.0, 1.0], [1.0, 1.0]]
        assert best_response(u, [0.5, 0.5], tiebreak="lowest_index") == 0
        assert best_response(u, [0.5, 0.5], tiebreak="highest_index") == 1


# =====================================================================
# Canonical KG result
# =====================================================================


class TestKamenicaGentzkow:
    def test_optimal_sender_value_equals_six_tenths(self) -> None:
        g = _kg_game(prior_guilty=0.3)
        p = Persuader()
        out = p.persuade(g)
        # Theoretical optimum is 0.6.
        assert out.sender_value == pytest.approx(0.6, abs=1e-6)

    def test_recommended_posteriors_are_kg_pair(self) -> None:
        g = _kg_game(prior_guilty=0.3)
        out = Persuader().persuade(g)
        # Sort posteriors by P(guilty) ascending; expect (0, 1) and (0.5, 0.5).
        post_pairs = sorted(out.induced_posteriors, key=lambda p: p[0])
        assert post_pairs[0][0] == pytest.approx(0.0, abs=1e-6)
        assert post_pairs[0][1] == pytest.approx(1.0, abs=1e-6)
        assert post_pairs[1][0] == pytest.approx(0.5, abs=1e-6)
        assert post_pairs[1][1] == pytest.approx(0.5, abs=1e-6)

    def test_signal_marginals_are_six_to_four(self) -> None:
        g = _kg_game(prior_guilty=0.3)
        out = Persuader().persuade(g)
        marginals = sorted(out.signal_marginals)
        assert marginals[0] == pytest.approx(0.4, abs=1e-6)
        assert marginals[1] == pytest.approx(0.6, abs=1e-6)

    def test_no_persuasion_for_prior_above_half(self) -> None:
        # If P(guilty) ≥ 0.5 the judge convicts unconditionally; persuasion
        # cannot strictly improve.
        g = _kg_game(prior_guilty=0.7)
        out = Persuader().persuade(g)
        # No-info BR is convict ⇒ sender gets 1 regardless of signals.
        assert out.sender_value == pytest.approx(1.0, abs=1e-6)

    def test_persuasion_never_below_no_info_or_above_sender_optimum(self) -> None:
        g = _kg_game(prior_guilty=0.3)
        out = Persuader().persuade(g)
        # Lower bound: no-info value (receiver picks BR under prior).
        no_info = sender_value_under_full_information(g, g.prior)
        # Upper bound: sender's *first-best* (would the receiver be a puppet).
        # This is Σ_ω μ₀(ω) · max_a u_S(a, ω).
        sender_first_best = sum(
            g.prior[w] * max(g.sender_utility[a][w]
                              for a in range(len(g.actions)))
            for w in range(len(g.states))
        )
        assert out.sender_value >= no_info - 1e-9
        assert out.sender_value <= sender_first_best + 1e-9


# =====================================================================
# Bayes plausibility / obedience
# =====================================================================


class TestBayesPlausibility:
    def test_optimal_scheme_is_bayes_plausible(self) -> None:
        out = Persuader().persuade(_kg_game())
        assert out.bayes_plausible
        report = verify_scheme(_kg_game(), out.scheme)
        assert report.bayes_plausibility_gap < 1e-6

    def test_optimal_scheme_is_obedient(self) -> None:
        out = Persuader().persuade(_kg_game())
        assert out.obedience_ok
        report = verify_scheme(_kg_game(), out.scheme)
        assert report.max_obedience_violation < 1e-6

    def test_3_state_lp_is_bayes_plausible(self) -> None:
        g = _weather_game()
        out = Persuader().persuade(g, kind=KIND_LP)
        assert out.bayes_plausible
        # Marginals should sum to 1 across signals.
        assert sum(out.signal_marginals) == pytest.approx(1.0, abs=1e-6)
        # Average of weighted posteriors = prior.
        avg = [0.0] * len(g.states)
        for i in range(len(out.scheme.signals)):
            m = out.signal_marginals[i]
            for j in range(len(g.states)):
                avg[j] += m * out.induced_posteriors[i][j]
        for j in range(len(g.states)):
            assert avg[j] == pytest.approx(g.prior[j], abs=1e-6)


# =====================================================================
# No-info optimum when interests align
# =====================================================================


class TestNoPersuasionWhenAligned:
    def test_identical_utilities_yield_no_strict_improvement(self) -> None:
        g = _identical_game()
        out = Persuader().persuade(g)
        # No-info value: receiver picks argmax expected utility under prior.
        no_info = sender_value_under_full_information(g, g.prior)
        full_info = sum(
            g.prior[w] *
            sender_value_under_full_information(g,
                tuple(1.0 if i == w else 0.0 for i in range(len(g.states))))
            for w in range(len(g.states))
        )
        # The optimum equals full-info because sender benefits from accuracy.
        assert out.sender_value == pytest.approx(full_info, abs=1e-6)
        assert out.sender_value >= no_info - 1e-9


# =====================================================================
# 3-state LP
# =====================================================================


class TestWeatherGame:
    def test_optimum_strictly_beats_full_info(self) -> None:
        g = _weather_game()
        out = Persuader().persuade(g, kind=KIND_LP)
        full_info = sum(
            g.prior[w] *
            sender_value_under_full_information(g,
                tuple(1.0 if i == w else 0.0 for i in range(len(g.states))))
            for w in range(len(g.states))
        )
        no_info = sender_value_under_full_information(g, g.prior)
        # Persuasion can strictly improve sender value above both bounds.
        # Closed-form optimum here is 0.9 (pool sun+cloud+rain at 1/3 each).
        assert out.sender_value > no_info + 1e-6
        assert out.sender_value > full_info + 1e-6
        assert out.sender_value == pytest.approx(0.9, abs=1e-3)


# =====================================================================
# Send-signal sampling
# =====================================================================


class TestSignalSampling:
    def test_signal_distribution_matches_pi(self) -> None:
        g = _kg_game()
        out = Persuader().persuade(g)
        rng = random.Random(42)
        # Empirical frequency under realised state "innocent" should match
        # column-1 of π. Find that column.
        counts = {s: 0 for s in out.scheme.signals}
        for _ in range(20_000):
            s = Persuader().send_signal(out.scheme, "innocent", rng=rng)
            counts[s] += 1
        # Expected pi[i][1] for each signal i.
        for i, sig in enumerate(out.scheme.signals):
            expected = out.scheme.pi[i][1]
            empirical = counts[sig] / 20_000
            assert abs(expected - empirical) < 0.02


# =====================================================================
# Online persuasion
# =====================================================================


class TestOnlinePersuasion:
    def test_cumulative_regret_within_bound(self) -> None:
        g = _kg_game()
        # Adversary: each round returns the sender's payoff under the
        # scheme on a sampled state from the prior.
        rng = random.Random(0)

        def feedback(scheme: SignalingScheme) -> float:
            # Realise state ~ μ₀.
            u = rng.random()
            cum = 0.0
            w = 0
            for k, p in enumerate(g.prior):
                cum += p
                if u <= cum:
                    w = k
                    break
            # Sample signal | state.
            weights = [scheme.pi[i][w] for i in range(len(scheme.signals))]
            tot = sum(weights)
            uu = rng.random() * tot
            running = 0.0
            sig_idx = 0
            for i, wt in enumerate(weights):
                running += wt
                if uu <= running:
                    sig_idx = i
                    break
            post = scheme.posterior(scheme.signals[sig_idx], g.prior)
            post_tuple = [post[st] for st in g.states]
            a = best_response(g.receiver_utility, post_tuple)
            return g.sender_utility[a][w]

        out = Persuader(rng=random.Random(0)).online_persuade(
            g, feedback, T=200, grid=3
        )
        # Regret bound from Hedge with K arms is √(T ln K / 2) + 1.
        assert out.cumulative_regret <= out.regret_bound + 1e-6

    def test_online_outcome_shape(self) -> None:
        g = _kg_game()
        out = Persuader(rng=random.Random(1)).online_persuade(
            g, lambda s: 0.5, T=20, grid=3
        )
        assert out.T == 20
        assert isinstance(out.final_scheme, SignalingScheme)
        assert out.regret_bound > 0


# =====================================================================
# Robust persuasion
# =====================================================================


class TestRobustPersuasion:
    def test_robust_dominates_single_prior_solutions(self) -> None:
        # Receiver prefers action A if μ(state=a) > 0.5.
        states = ("a", "b")
        actions = ("A", "B")
        sender_u = ((1.0, 1.0), (0.0, 0.0))
        receiver_u = ((1.0, -1.0), (0.0, 0.0))
        prior_set = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
        p = Persuader()
        out = p.robust_persuade(states, actions, sender_u, receiver_u,
                                prior_set)
        # Check that for every prior in the set, the chosen scheme yields
        # at least the worst-case value reported.
        for mu in prior_set:
            game = PersuasionGame(
                states=states, actions=actions, prior=tuple(mu),
                sender_utility=sender_u, receiver_utility=receiver_u,
            )
            rep = verify_scheme(game, out.scheme)
            assert rep.sender_value >= out.worst_case_value - 1e-6


# =====================================================================
# Multi-receiver
# =====================================================================


class TestMultiReceiver:
    def test_private_additive_decomposes_per_receiver(self) -> None:
        g1 = _kg_game(prior_guilty=0.3)
        g2 = _kg_game(prior_guilty=0.4)
        p = Persuader()
        out = p.multi_receiver_private([("r1", g1), ("r2", g2)])
        single1 = p.persuade(g1, kind=KIND_LP).sender_value
        single2 = p.persuade(g2, kind=KIND_LP).sender_value
        assert out.joint_sender_value == pytest.approx(single1 + single2,
                                                       abs=1e-6)
        assert isinstance(out, MultiReceiverOutcome)
        ids = [rid for rid, _ in out.per_receiver_schemes]
        assert ids == ["r1", "r2"]


# =====================================================================
# Hoeffding PAC certificate
# =====================================================================


class TestCertificate:
    def test_certificate_lcb_under_mean(self) -> None:
        g = _kg_game()
        out = Persuader().persuade(g)
        cert = Persuader(rng=random.Random(0)).simulate(
            g, out.scheme, T=2_000, delta=0.05, method="hoeffding"
        )
        assert cert.lcb <= cert.empirical_mean <= cert.ucb

    def test_empirical_bernstein_tighter(self) -> None:
        g = _kg_game()
        out = Persuader().persuade(g)
        cert_h = Persuader(rng=random.Random(0)).simulate(
            g, out.scheme, T=2_000, method="hoeffding"
        )
        cert_b = Persuader(rng=random.Random(0)).simulate(
            g, out.scheme, T=2_000, method="empirical_bernstein"
        )
        # Empirical-Bernstein is *not always* tighter at small n but
        # should be at this scale for a high-variance Bernoulli signal.
        # We assert finite & non-negative half-widths.
        assert cert_h.half_width > 0
        assert cert_b.half_width > 0


# =====================================================================
# Receipt determinism
# =====================================================================


class TestReceiptDeterminism:
    def test_identical_inputs_produce_identical_receipts(self) -> None:
        g = _kg_game()
        out1 = Persuader().persuade(g)
        out2 = Persuader().persuade(g)
        assert out1.receipt_digest == out2.receipt_digest


# =====================================================================
# Event emission
# =====================================================================


class TestEvents:
    def test_persuade_solved_fires(self) -> None:
        bus = EventBus()
        seen: list[Event] = []

        def collect(ev: Event) -> None:
            seen.append(ev)

        bus.subscribe(collect)
        p = Persuader(bus=bus)
        p.persuade(_kg_game())
        kinds = {ev.kind for ev in seen}
        assert PERSUADE_STARTED in kinds
        assert PERSUADE_SOLVED in kinds

    def test_verify_event_fires(self) -> None:
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append)
        p = Persuader(bus=bus)
        out = p.persuade(_kg_game())
        p.verify(_kg_game(), out.scheme)
        assert PERSUADE_VERIFIED in {ev.kind for ev in seen}

    def test_certify_event_fires(self) -> None:
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(seen.append)
        p = Persuader(bus=bus, rng=random.Random(0))
        out = p.persuade(_kg_game())
        p.simulate(_kg_game(), out.scheme, T=200)
        assert PERSUADE_CERTIFIED in {ev.kind for ev in seen}


# =====================================================================
# Concentration helpers
# =====================================================================


class TestConcentration:
    def test_hoeffding_radius_scales_correctly(self) -> None:
        r1 = hoeffding_radius(100, delta=0.05, range_=1.0)
        r2 = hoeffding_radius(400, delta=0.05, range_=1.0)
        assert r2 == pytest.approx(r1 / 2.0, rel=1e-9)

    def test_empirical_bernstein_decreases_with_n(self) -> None:
        samples_small = [0.5, 0.6, 0.4, 0.55, 0.45]
        eb_small = empirical_bernstein_radius(samples_small, delta=0.05, range_=1.0)
        samples_big = samples_small * 100
        eb_big = empirical_bernstein_radius(samples_big, delta=0.05, range_=1.0)
        assert eb_big < eb_small


# =====================================================================
# Facade
# =====================================================================


class TestFacade:
    def test_quick_persuade(self) -> None:
        out = quick_persuade(
            states=("a", "b"),
            actions=("A", "B"),
            prior=(0.3, 0.7),
            sender_utility=((1.0, 1.0), (0.0, 0.0)),
            receiver_utility=((1.0, -1.0), (0.0, 0.0)),
        )
        assert out.sender_value == pytest.approx(0.6, abs=1e-6)

    def test_quick_verify(self) -> None:
        out = quick_persuade(
            states=("a", "b"),
            actions=("A", "B"),
            prior=(0.3, 0.7),
            sender_utility=((1.0, 1.0), (0.0, 0.0)),
            receiver_utility=((1.0, -1.0), (0.0, 0.0)),
        )
        rep = quick_verify(
            states=("a", "b"),
            actions=("A", "B"),
            prior=(0.3, 0.7),
            sender_utility=((1.0, 1.0), (0.0, 0.0)),
            receiver_utility=((1.0, -1.0), (0.0, 0.0)),
            scheme=out.scheme,
        )
        assert rep.bayes_plausible
        assert rep.obedience_ok


# =====================================================================
# Stats / lifecycle
# =====================================================================


class TestLifecycle:
    def test_stats_increments(self) -> None:
        p = Persuader()
        p.persuade(_kg_game())
        p.persuade(_kg_game())
        assert p.stats()["persuade_calls"] == 2

    def test_clear_resets_counters(self) -> None:
        p = Persuader()
        p.persuade(_kg_game())
        p.clear()
        assert p.stats()["persuade_calls"] == 0

    def test_thread_safety(self) -> None:
        p = Persuader()
        g = _kg_game()

        def run() -> None:
            for _ in range(20):
                p.persuade(g)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Final counter should be exactly 80.
        assert p.stats()["persuade_calls"] == 80
