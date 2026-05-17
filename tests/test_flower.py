"""Tests for ``agi.flower`` — Generative Flow Networks as a runtime primitive."""
from __future__ import annotations

import math
import random
import threading

import pytest

from agi.flower import (
    FLOWER_CERTIFIED,
    FLOWER_CLEARED,
    FLOWER_OBSERVED,
    FLOWER_REGISTERED,
    FLOWER_REMOVED,
    FLOWER_SAMPLED,
    FLOWER_STARTED,
    FLOWER_TRAINED,
    KNOWN_LOSSES,
    LOSS_DETAILED_BALANCE,
    LOSS_FLOW_MATCHING,
    LOSS_SUBTRAJECTORY_BALANCE,
    LOSS_TRAJECTORY_BALANCE,
    EnvSpec,
    Flower,
    FlowerConfig,
    InsufficientData,
    InvalidConfig,
    InvalidEnv,
    InvalidTrajectory,
    ModeCoverageReport,
    PITCalibrationReport,
    SampleBatch,
    TrainReport,
    Trajectory,
    UnknownEnv,
    empirical_bernstein_half_width,
    hoeffding_half_width,
    hrms_half_width,
    ks_pvalue,
    ledger_root,
    logsumexp,
    softmax,
    total_variation,
)


# ---------------------------------------------------------------------------
# Tiny reward-shaped DAGs used across tests
# ---------------------------------------------------------------------------


def chain_env(depth: int = 4, reward_at_end: float = 5.0):
    """Linear chain of length depth; reward at the terminal."""

    def succ(s):
        if s >= depth:
            return []
        return [("a", s + 1)]

    def term(s):
        return s == depth

    def rew(s):
        return reward_at_end if s == depth else 1e-12

    return dict(initial=0, successors=succ, terminal=term, reward=rew)


def binary_tree_env(depth: int = 3, rewards: dict[str, float] | None = None):
    """Binary tree of given depth.  ``rewards`` maps leaf string → R."""
    default_rewards = rewards or {}

    def succ(s):
        if len(s) >= depth:
            return []
        return [("L", s + "L"), ("R", s + "R")]

    def term(s):
        return len(s) == depth

    def rew(s):
        return default_rewards.get(s, 1.0)

    return dict(initial="", successors=succ, terminal=term, reward=rew)


def grid_env(rows: int = 3, cols: int = 3):
    """Grid where successors go right or down; reward = row * col."""

    def succ(s):
        r, c = s
        out = []
        if r + 1 < rows:
            out.append(("D", (r + 1, c)))
        if c + 1 < cols:
            out.append(("R", (r, c + 1)))
        return out

    def term(s):
        r, c = s
        return r == rows - 1 and c == cols - 1

    def rew(s):
        r, c = s
        # Make corner rewards differ.
        return float((r + 1) * (c + 1))

    return dict(initial=(0, 0), successors=succ, terminal=term, reward=rew)


def collect_events(events_list):
    def publisher(kind, payload):
        events_list.append((kind, dict(payload)))

    return publisher


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config_is_valid(self):
        cfg = FlowerConfig()
        assert cfg.loss == LOSS_TRAJECTORY_BALANCE
        assert 0.5 < cfg.confidence < 1.0

    def test_known_losses(self):
        assert LOSS_FLOW_MATCHING in KNOWN_LOSSES
        assert LOSS_DETAILED_BALANCE in KNOWN_LOSSES
        assert LOSS_TRAJECTORY_BALANCE in KNOWN_LOSSES
        assert LOSS_SUBTRAJECTORY_BALANCE in KNOWN_LOSSES
        assert len(KNOWN_LOSSES) == 4

    def test_unknown_loss_rejected(self):
        with pytest.raises(InvalidConfig, match="unknown loss"):
            FlowerConfig(loss="nonsense")

    def test_invalid_confidence_rejected(self):
        with pytest.raises(InvalidConfig, match="confidence"):
            FlowerConfig(confidence=0.0)
        with pytest.raises(InvalidConfig, match="confidence"):
            FlowerConfig(confidence=1.0)

    def test_invalid_learning_rate_rejected(self):
        with pytest.raises(InvalidConfig, match="learning_rate"):
            FlowerConfig(learning_rate=0)
        with pytest.raises(InvalidConfig, match="learning_rate"):
            FlowerConfig(learning_rate=-1.0)

    def test_invalid_logz_lr_rejected(self):
        with pytest.raises(InvalidConfig, match="learning_rate_logz"):
            FlowerConfig(learning_rate_logz=-1.0)

    def test_invalid_subtb_lambda_rejected(self):
        with pytest.raises(InvalidConfig, match="subtb_lambda"):
            FlowerConfig(subtb_lambda=0.0)
        with pytest.raises(InvalidConfig, match="subtb_lambda"):
            FlowerConfig(subtb_lambda=1.5)

    def test_invalid_epsilon_rejected(self):
        with pytest.raises(InvalidConfig, match="epsilon"):
            FlowerConfig(epsilon_exploration=-0.1)
        with pytest.raises(InvalidConfig, match="epsilon"):
            FlowerConfig(epsilon_exploration=1.1)

    def test_invalid_replay_rejected(self):
        with pytest.raises(InvalidConfig, match="replay_capacity"):
            FlowerConfig(replay_capacity=-1)
        with pytest.raises(InvalidConfig, match="replay_min_fill"):
            FlowerConfig(replay_min_fill=-1)

    def test_invalid_traj_len_rejected(self):
        with pytest.raises(InvalidConfig, match="max_trajectory_length"):
            FlowerConfig(max_trajectory_length=0)

    def test_invalid_reward_floor_rejected(self):
        with pytest.raises(InvalidConfig, match="reward_floor"):
            FlowerConfig(reward_floor=0)
        with pytest.raises(InvalidConfig, match="reward_floor"):
            FlowerConfig(reward_floor=-1.0)

    def test_invalid_grad_clip_rejected(self):
        with pytest.raises(InvalidConfig, match="grad_clip"):
            FlowerConfig(grad_clip=0)

    def test_invalid_max_envs_rejected(self):
        with pytest.raises(InvalidConfig, match="max_envs"):
            FlowerConfig(max_envs=0)
        with pytest.raises(InvalidConfig, match="max_envs"):
            FlowerConfig(max_envs=-1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_env_returns_spec(self):
        f = Flower()
        env = chain_env()
        spec = f.register_env("chain", **env)
        assert isinstance(spec, EnvSpec)
        assert spec.env == "chain"
        assert spec.initial == 0
        assert "chain" in f.envs()

    def test_register_with_loss_override(self):
        f = Flower(FlowerConfig(loss=LOSS_TRAJECTORY_BALANCE))
        env = chain_env()
        f.register_env("chain", loss=LOSS_DETAILED_BALANCE, **env)
        spec = f.env_spec("chain")
        assert spec.loss == LOSS_DETAILED_BALANCE

    def test_register_unknown_loss_rejected(self):
        f = Flower()
        env = chain_env()
        with pytest.raises(InvalidEnv, match="unknown loss"):
            f.register_env("chain", loss="bogus", **env)

    def test_register_empty_name_rejected(self):
        f = Flower()
        env = chain_env()
        with pytest.raises(InvalidEnv):
            f.register_env("", **env)

    def test_double_register_rejected(self):
        f = Flower()
        env = chain_env()
        f.register_env("chain", **env)
        with pytest.raises(InvalidEnv, match="already registered"):
            f.register_env("chain", **env)

    def test_remove_env(self):
        f = Flower()
        env = chain_env()
        f.register_env("chain", **env)
        f.remove_env("chain")
        assert "chain" not in f.envs()

    def test_remove_unknown_env(self):
        f = Flower()
        with pytest.raises(UnknownEnv):
            f.remove_env("missing")

    def test_max_envs_enforced(self):
        f = Flower(FlowerConfig(max_envs=2))
        env = chain_env()
        f.register_env("a", **env)
        f.register_env("b", **env)
        with pytest.raises(InvalidEnv, match="max_envs"):
            f.register_env("c", **env)

    def test_clear_resets_state(self):
        f = Flower()
        env = chain_env()
        f.register_env("chain", **env)
        head_before = f.chain_head
        f.clear()
        assert f.envs() == []
        # Chain head moves after clear.
        assert f.chain_head != head_before

    def test_env_spec_unknown(self):
        f = Flower()
        with pytest.raises(UnknownEnv):
            f.env_spec("missing")


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_softmax_uniform_when_all_equal(self):
        result = softmax([0.0, 0.0, 0.0])
        for r in result:
            assert math.isclose(r, 1 / 3)

    def test_softmax_temperature(self):
        # High beta → sharper, low beta → flatter.
        s_hi = softmax([1.0, 2.0], beta=10.0)
        s_lo = softmax([1.0, 2.0], beta=0.01)
        assert s_hi[1] > s_lo[1]
        assert math.isclose(sum(s_hi), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(s_lo), 1.0, abs_tol=1e-9)

    def test_softmax_empty(self):
        assert softmax([]) == []

    def test_softmax_handles_inf_floor(self):
        result = softmax([-1e9, -1e9])
        # Both zero exps after re-centering; gets uniform fallback.
        for r in result:
            assert math.isclose(r, 0.5)

    def test_logsumexp_empty(self):
        assert logsumexp([]) == float("-inf")

    def test_logsumexp_known(self):
        # log(e^0 + e^1 + e^2) = log(1 + e + e^2)
        v = logsumexp([0.0, 1.0, 2.0])
        assert math.isclose(v, math.log(1 + math.e + math.e**2), rel_tol=1e-9)

    def test_logsumexp_inf(self):
        assert logsumexp([float("-inf"), float("-inf")]) == float("-inf")

    def test_hoeffding_decreases_with_n(self):
        a = hoeffding_half_width(10)
        b = hoeffding_half_width(100)
        c = hoeffding_half_width(1000)
        assert a > b > c

    def test_hoeffding_zero_n(self):
        assert hoeffding_half_width(0) == float("inf")

    def test_empirical_bernstein_zero_var(self):
        # Pure log term scales when variance = 0.
        v = empirical_bernstein_half_width(100, variance=0.0)
        assert v > 0
        assert math.isfinite(v)

    def test_empirical_bernstein_n_le_1(self):
        assert empirical_bernstein_half_width(1, 0.5) == float("inf")
        assert empirical_bernstein_half_width(0, 0.5) == float("inf")

    def test_hrms_decreases_with_n(self):
        a = hrms_half_width(10)
        b = hrms_half_width(100)
        c = hrms_half_width(1000)
        assert a > b > c

    def test_hrms_n_le_1(self):
        assert hrms_half_width(0) == float("inf")
        assert hrms_half_width(1) == float("inf")

    def test_ks_pvalue_uniform_passes(self):
        rng = random.Random(0)
        samples = [rng.random() for _ in range(500)]
        d, p = ks_pvalue(samples)
        assert d >= 0
        assert p > 0.05  # uniform should pass

    def test_ks_pvalue_skewed_fails(self):
        rng = random.Random(0)
        # All small → concentrated on 0, should fail Uniform(0, 1).
        samples = [rng.random() * 0.1 for _ in range(500)]
        _, p = ks_pvalue(samples)
        assert p < 0.05

    def test_ks_pvalue_empty(self):
        d, p = ks_pvalue([])
        assert d == 0.0
        assert p == 1.0

    def test_total_variation_identity(self):
        p = {"a": 0.5, "b": 0.5}
        assert total_variation(p, p) == 0

    def test_total_variation_disjoint(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        assert math.isclose(total_variation(p, q), 1.0)


# ---------------------------------------------------------------------------
# Trajectory enumeration via observe()
# ---------------------------------------------------------------------------


class TestObserve:
    def test_observe_simple_trajectory(self):
        f = Flower()
        f.register_env("chain", **chain_env(depth=3))
        traj = f.observe("chain", states=(0, 1, 2, 3), actions=("a", "a", "a"))
        assert traj.terminal
        assert traj.reward == 5.0
        assert traj.states == (0, 1, 2, 3)

    def test_observe_with_explicit_reward(self):
        f = Flower()
        f.register_env("chain", **chain_env(depth=2))
        traj = f.observe(
            "chain", states=(0, 1, 2), actions=("a", "a"), reward=7.5
        )
        assert traj.reward == 7.5

    def test_observe_invalid_states_length(self):
        f = Flower()
        f.register_env("chain", **chain_env(depth=2))
        with pytest.raises(InvalidTrajectory):
            f.observe("chain", states=(), actions=())

    def test_observe_action_length_mismatch(self):
        f = Flower()
        f.register_env("chain", **chain_env(depth=2))
        with pytest.raises(InvalidTrajectory, match="len\\(actions\\)"):
            f.observe("chain", states=(0, 1, 2), actions=("a",))

    def test_observe_unknown_env(self):
        f = Flower()
        with pytest.raises(UnknownEnv):
            f.observe("missing", states=(0,), actions=())

    def test_observe_records_visits(self):
        f = Flower()
        f.register_env("chain", **chain_env(depth=2))
        f.observe("chain", states=(0, 1, 2), actions=("a", "a"))
        f.observe("chain", states=(0, 1, 2), actions=("a", "a"))
        # Inspect via identifiability — edge counts ≥ 2.
        rep = f.identifiability("chain", top_k=5)
        counts = {(s, a): c for s, a, c in rep.under_sampled_edges}
        assert counts[(0, "a")] == 2
        assert counts[(1, "a")] == 2

    def test_observe_floors_low_reward(self):
        f = Flower(FlowerConfig(reward_floor=0.001))
        f.register_env("chain", **chain_env(depth=2))
        traj = f.observe(
            "chain", states=(0, 1, 2), actions=("a", "a"), reward=0.0
        )
        assert traj.reward == 0.001


# ---------------------------------------------------------------------------
# Training — convergence on small DAGs
# ---------------------------------------------------------------------------


class TestTraining:
    def _train_until(self, f, env, *, target_loss=0.01, max_iters=400, batch=16):
        last_loss = float("inf")
        for _ in range(max_iters):
            r = f.train_step(env, n_trajectories=batch)
            last_loss = r.loss_value
            if last_loss < target_loss:
                return r
        return r

    def test_trajectory_balance_converges_on_chain(self):
        f = Flower(FlowerConfig(loss=LOSS_TRAJECTORY_BALANCE, learning_rate=0.1, rng_seed=1))
        f.register_env("c", **chain_env(depth=3, reward_at_end=4.0))
        rep = self._train_until(f, "c", target_loss=0.005, max_iters=300)
        assert rep.loss_value < 0.05
        # log Z should approach log(4) ≈ 1.386
        assert abs(rep.logZ_estimate - math.log(4.0)) < 0.15

    def test_detailed_balance_converges_on_chain(self):
        f = Flower(FlowerConfig(loss=LOSS_DETAILED_BALANCE, learning_rate=0.1, rng_seed=2))
        f.register_env("c", **chain_env(depth=3, reward_at_end=4.0))
        rep = self._train_until(f, "c", target_loss=0.05, max_iters=200)
        assert rep.loss_value < 0.05

    def test_flow_matching_converges_on_chain(self):
        f = Flower(FlowerConfig(loss=LOSS_FLOW_MATCHING, learning_rate=0.05, rng_seed=3))
        f.register_env("c", **chain_env(depth=3, reward_at_end=4.0))
        rep = self._train_until(f, "c", target_loss=0.1, max_iters=300, batch=16)
        # FM converges more slowly; lenient bound.
        assert rep.loss_value < 0.5

    def test_subtb_converges_on_chain(self):
        f = Flower(FlowerConfig(loss=LOSS_SUBTRAJECTORY_BALANCE, learning_rate=0.05, rng_seed=4))
        f.register_env("c", **chain_env(depth=3, reward_at_end=4.0))
        rep = self._train_until(f, "c", target_loss=0.5, max_iters=200, batch=8)
        # SubTB has many residuals → loss scale differs.
        assert rep.loss_value < 1.0

    def test_train_step_returns_train_report(self):
        f = Flower(FlowerConfig(rng_seed=5))
        f.register_env("c", **chain_env())
        rep = f.train_step("c", n_trajectories=4)
        assert isinstance(rep, TrainReport)
        assert rep.env == "c"
        assert rep.n_trajectories >= 4  # may include replay
        assert rep.fingerprint != ""

    def test_train_step_zero_trajectories(self):
        f = Flower()
        f.register_env("c", **chain_env())
        with pytest.raises(InvalidTrajectory):
            f.train_step("c", n_trajectories=0)

    def test_train_step_unknown_loss(self):
        f = Flower()
        f.register_env("c", **chain_env())
        with pytest.raises(InvalidConfig, match="unknown loss"):
            f.train_step("c", n_trajectories=4, loss="bogus")

    def test_grad_norm_decreases_after_convergence(self):
        f = Flower(FlowerConfig(rng_seed=6, learning_rate=0.1))
        f.register_env("c", **chain_env(depth=2))
        early = f.train_step("c", n_trajectories=8)
        for _ in range(200):
            f.train_step("c", n_trajectories=8)
        late = f.train_step("c", n_trajectories=8)
        assert late.grad_norm <= early.grad_norm + 0.1

    def test_logz_estimate_converges_to_partition(self):
        # Build a 2-leaf DAG with R(LL)=2, R(RR)=6 ⇒ Z = 8.
        f = Flower(FlowerConfig(loss=LOSS_TRAJECTORY_BALANCE, learning_rate=0.05, rng_seed=7))
        env = binary_tree_env(depth=2, rewards={"LL": 2.0, "LR": 1.0, "RL": 1.0, "RR": 4.0})
        f.register_env("tree", **env)
        for _ in range(400):
            r = f.train_step("tree", n_trajectories=16, epsilon=0.1)
        # Z = 2 + 1 + 1 + 4 = 8 ⇒ log Z = log(8) ≈ 2.079
        assert abs(r.logZ_estimate - math.log(8.0)) < 0.2


# ---------------------------------------------------------------------------
# Sampling — outputs + bounds
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_returns_batch(self):
        f = Flower(FlowerConfig(rng_seed=10))
        f.register_env("c", **chain_env())
        batch = f.sample("c", n=5)
        assert isinstance(batch, SampleBatch)
        assert batch.n == 5
        assert len(batch.trajectories) == 5
        assert len(batch.terminals) == 5
        assert len(batch.rewards) == 5

    def test_sample_zero_rejected(self):
        f = Flower()
        f.register_env("c", **chain_env())
        with pytest.raises(InvalidTrajectory):
            f.sample("c", n=0)

    def test_sample_lcb_bounds_mean(self):
        f = Flower(FlowerConfig(rng_seed=11))
        f.register_env("c", **chain_env(reward_at_end=2.0))
        batch = f.sample("c", n=30)
        assert batch.mean_reward_lcb <= batch.mean_reward
        assert batch.mean_reward_hrms_lcb <= batch.mean_reward

    def test_sample_entropy_zero_when_unique(self):
        f = Flower(FlowerConfig(rng_seed=12))
        f.register_env("c", **chain_env(depth=2))  # only one terminal
        batch = f.sample("c", n=10)
        assert batch.unique_terminals == 1
        assert batch.forward_entropy == 0

    def test_sample_diversity_on_branching(self):
        f = Flower(FlowerConfig(rng_seed=13))
        f.register_env("tree", **binary_tree_env(depth=2, rewards={"LL": 1.0, "LR": 1.0, "RL": 1.0, "RR": 1.0}))
        batch = f.sample("tree", n=50)
        assert batch.unique_terminals >= 2
        assert batch.forward_entropy > 0

    def test_sample_temperature_zero_is_greedy(self):
        f = Flower(FlowerConfig(rng_seed=14))
        env = binary_tree_env(depth=2, rewards={"LL": 100.0, "LR": 0.01, "RL": 0.01, "RR": 0.01})
        f.register_env("tree", **env)
        for _ in range(200):
            f.train_step("tree", n_trajectories=8)
        # After training, temperature=0 → deterministic argmax → all LL.
        batch = f.sample("tree", n=20, temperature=0.0, epsilon=0.0)
        terms = set(batch.terminals)
        assert terms == {"LL"}

    def test_sample_records_visits_by_default(self):
        f = Flower(FlowerConfig(rng_seed=15))
        f.register_env("c", **chain_env())
        f.sample("c", n=5, record=True)
        rep = f.identifiability("c", top_k=5)
        # Some edge had > 0 visits.
        assert any(c > 0 for _, _, c in rep.under_sampled_edges)

    def test_sample_no_record_skips_visits(self):
        f = Flower(FlowerConfig(rng_seed=16))
        f.register_env("c", **chain_env())
        f.sample("c", n=5, record=False)
        rep = f.identifiability("c", top_k=5)
        # No visits recorded.
        assert all(c == 0 for _, _, c in rep.under_sampled_edges)

    def test_sample_distribution_proportional_to_reward(self):
        """Acceptance test: GFlowNet samples ~ proportional to reward."""
        f = Flower(FlowerConfig(loss=LOSS_TRAJECTORY_BALANCE, learning_rate=0.05, rng_seed=17))
        # Two strong modes
        env = binary_tree_env(
            depth=3,
            rewards={
                "LLL": 8.0,
                "LLR": 1.0,
                "LRL": 1.0,
                "LRR": 1.0,
                "RLL": 1.0,
                "RLR": 1.0,
                "RRL": 1.0,
                "RRR": 4.0,
            },
        )
        f.register_env("tree", **env)
        for _ in range(300):
            f.train_step("tree", n_trajectories=16, epsilon=0.1)
        batch = f.sample("tree", n=600, temperature=1.0, epsilon=0.0)
        counts: dict[str, int] = {}
        for t in batch.terminals:
            counts[t] = counts.get(t, 0) + 1
        # LLL should be sampled most, RRR second.
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        top = ranked[0][0]
        second = ranked[1][0]
        assert top == "LLL"
        assert second == "RRR"
        # And the proportion of LLL ≈ 8/18 = 0.44 within ±0.2 tolerance.
        assert counts["LLL"] / 600 > 0.25


# ---------------------------------------------------------------------------
# Mode coverage + identifiability + PIT calibration
# ---------------------------------------------------------------------------


class TestModeCoverage:
    def test_mode_coverage_with_no_terminals_raises(self):
        f = Flower()
        f.register_env("c", **chain_env())
        # Register but never sample — raise.
        # We need to bypass sample()'s own observation; use a private hack:
        # Force the env state to have empty terminal counts AND skip the
        # internal sample by overriding n_samples to a value where
        # _sample_one is called but no terminal is reached.  Easiest path:
        # use an env where no path terminates at all.

        def bad_succ(_):
            return [("loop", 0)]

        def never_term(_):
            return False

        def zero_rew(_):
            return 1.0

        f.register_env(
            "stuck",
            initial=0,
            successors=bad_succ,
            terminal=never_term,
            reward=zero_rew,
        )
        with pytest.raises(InsufficientData):
            f.mode_coverage("stuck", n_samples=2, top_k=1)

    def test_mode_coverage_returns_report(self):
        f = Flower(FlowerConfig(rng_seed=20))
        f.register_env("c", **chain_env())
        cov = f.mode_coverage("c", n_samples=20, top_k=1)
        assert isinstance(cov, ModeCoverageReport)
        assert cov.modes_found == 1
        assert cov.top_k_recovered == (1, 1)
        assert 0 <= cov.tv_to_target <= 1
        assert 0 <= cov.mode_coverage_lcb <= 1

    def test_mode_coverage_tv_low_after_training(self):
        f = Flower(FlowerConfig(loss=LOSS_TRAJECTORY_BALANCE, learning_rate=0.05, rng_seed=21))
        env = binary_tree_env(
            depth=2, rewards={"LL": 4.0, "LR": 1.0, "RL": 1.0, "RR": 2.0}
        )
        f.register_env("tree", **env)
        for _ in range(300):
            f.train_step("tree", n_trajectories=16, epsilon=0.1)
        cov = f.mode_coverage("tree", n_samples=500, top_k=2)
        # After training, TV should be small.
        assert cov.tv_to_target < 0.20
        # Both top modes recovered.
        assert cov.top_k_recovered == (2, 2)

    def test_mode_coverage_zero_n_rejected(self):
        f = Flower()
        f.register_env("c", **chain_env())
        with pytest.raises(InvalidTrajectory):
            f.mode_coverage("c", n_samples=0, top_k=1)
        with pytest.raises(InvalidTrajectory):
            f.mode_coverage("c", n_samples=10, top_k=0)

    def test_mode_coverage_unknown_env(self):
        f = Flower()
        with pytest.raises(UnknownEnv):
            f.mode_coverage("missing", n_samples=1, top_k=1)

    def test_identifiability_under_sampled_edges(self):
        f = Flower(FlowerConfig(rng_seed=22))
        env = binary_tree_env(depth=2, rewards={"LL": 1.0, "LR": 1.0, "RL": 1.0, "RR": 1.0})
        f.register_env("tree", **env)
        # Sample only a few — most edges still rare.
        f.sample("tree", n=5)
        rep = f.identifiability("tree", top_k=3)
        assert len(rep.under_sampled_edges) <= 3
        # No unreachable modes expected once edges are exhaustively covered.
        # Saturated should be empty under low visit count.
        assert rep.saturated_edges == ()

    def test_identifiability_unreachable_modes(self):
        f = Flower()
        # Make tree but only first branch ever explored by manually
        # setting epsilon=0 + start logits dominating L early.
        env = binary_tree_env(depth=2, rewards={"LL": 1.0, "LR": 1.0, "RL": 1.0, "RR": 1.0})
        f.register_env("tree", **env)
        # Don't sample anything → all modes unreachable so far.
        rep = f.identifiability("tree", top_k=10)
        # All 4 modes positive reward + 0 visits.
        assert set(rep.unreachable_modes) == {"LL", "LR", "RL", "RR"}

    def test_pit_calibration_requires_samples(self):
        f = Flower()
        f.register_env("c", **chain_env())
        with pytest.raises(InsufficientData):
            f.pit_calibration("c")

    def test_pit_calibration_returns_report(self):
        f = Flower(FlowerConfig(rng_seed=23))
        f.register_env("c", **chain_env())
        f.sample("c", n=10)
        rep = f.pit_calibration("c")
        assert isinstance(rep, PITCalibrationReport)
        assert rep.n >= 10
        assert 0 <= rep.p_value <= 1


# ---------------------------------------------------------------------------
# Top-K extraction
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_returns_sorted_distinct_terminals(self):
        f = Flower(FlowerConfig(rng_seed=30))
        env = binary_tree_env(depth=2, rewards={"LL": 5.0, "LR": 3.0, "RL": 1.0, "RR": 2.0})
        f.register_env("tree", **env)
        f.sample("tree", n=200)
        topk = f.top_k("tree", k=3)
        rewards = [r for _, r, _ in topk]
        assert rewards == sorted(rewards, reverse=True)
        assert rewards[0] == 5.0

    def test_top_k_empty_when_no_terminals_yet(self):
        f = Flower()
        f.register_env("c", **chain_env())
        # No sampling done.
        assert f.top_k("c", k=10) == []

    def test_top_k_invalid_k(self):
        f = Flower()
        f.register_env("c", **chain_env())
        with pytest.raises(InvalidTrajectory):
            f.top_k("c", k=0)


# ---------------------------------------------------------------------------
# Events + chain head
# ---------------------------------------------------------------------------


class TestEvents:
    def test_started_event_fires(self):
        events: list[tuple[str, dict]] = []
        Flower(publisher=collect_events(events))
        kinds = {k for k, _ in events}
        assert FLOWER_STARTED in kinds

    def test_registered_event_fires(self):
        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.register_env("c", **chain_env())
        kinds = [k for k, _ in events]
        assert FLOWER_REGISTERED in kinds

    def test_train_event_fires(self):
        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.register_env("c", **chain_env())
        f.train_step("c", n_trajectories=2)
        kinds = [k for k, _ in events]
        assert FLOWER_TRAINED in kinds

    def test_sample_event_fires(self):
        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.register_env("c", **chain_env())
        f.sample("c", n=1)
        kinds = [k for k, _ in events]
        assert FLOWER_SAMPLED in kinds

    def test_certified_event_fires(self):
        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.register_env("c", **chain_env())
        f.sample("c", n=1)
        f.mode_coverage("c", n_samples=4, top_k=1)
        kinds = [k for k, _ in events]
        assert FLOWER_CERTIFIED in kinds

    def test_remove_event_fires(self):
        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.register_env("c", **chain_env())
        f.remove_env("c")
        kinds = [k for k, _ in events]
        assert FLOWER_REMOVED in kinds

    def test_cleared_event_fires(self):
        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.clear()
        kinds = [k for k, _ in events]
        assert FLOWER_CLEARED in kinds

    def test_publisher_exception_swallowed(self):
        def bad(_kind, _payload):
            raise RuntimeError("nope")

        # Should not propagate.
        f = Flower(publisher=bad)
        f.register_env("c", **chain_env())
        f.sample("c", n=1)


# ---------------------------------------------------------------------------
# Attestation chain
# ---------------------------------------------------------------------------


class TestAttestation:
    def test_chain_head_advances_on_register(self):
        f = Flower()
        before = f.chain_head
        f.register_env("c", **chain_env())
        after = f.chain_head
        assert before != after

    def test_chain_head_advances_on_train(self):
        f = Flower()
        f.register_env("c", **chain_env())
        before = f.chain_head
        f.train_step("c", n_trajectories=1)
        after = f.chain_head
        assert before != after

    def test_chain_head_advances_on_sample(self):
        f = Flower()
        f.register_env("c", **chain_env())
        before = f.chain_head
        f.sample("c", n=1)
        after = f.chain_head
        assert before != after

    def test_hmac_root_differs_per_key(self):
        a = ledger_root(b"key-a")
        b = ledger_root(b"key-b")
        c = ledger_root(None)
        assert a != b
        assert a != c
        assert b != c

    def test_reproducible_under_same_seed(self):
        # Two flowers with same seed see same chain progression on
        # identical inputs.
        def build():
            f = Flower(FlowerConfig(rng_seed=99))
            f.register_env("c", **chain_env())
            for _ in range(3):
                f.train_step("c", n_trajectories=2)
            return f.chain_head

        assert build() == build()

    def test_different_seed_yields_different_chain(self):
        def build(seed):
            f = Flower(FlowerConfig(rng_seed=seed))
            f.register_env(
                "t",
                **binary_tree_env(
                    depth=2, rewards={"LL": 1.0, "LR": 1.0, "RL": 1.0, "RR": 1.0}
                ),
            )
            for _ in range(3):
                f.train_step("t", n_trajectories=2)
            return f.chain_head

        assert build(1) != build(2)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class TestReplay:
    def test_replay_capacity_bounded(self):
        f = Flower(FlowerConfig(replay_capacity=3, rng_seed=40))
        f.register_env("c", **chain_env())
        for _ in range(20):
            f.train_step("c", n_trajectories=2)
        st = f._envs["c"]
        assert len(st.replay) <= 3

    def test_replay_zero_skips(self):
        f = Flower(FlowerConfig(replay_capacity=0, rng_seed=41))
        f.register_env("c", **chain_env())
        for _ in range(5):
            f.train_step("c", n_trajectories=2)
        st = f._envs["c"]
        assert st.replay == []


# ---------------------------------------------------------------------------
# Threadsafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_train_does_not_corrupt(self):
        f = Flower(FlowerConfig(rng_seed=50, learning_rate=0.05))
        f.register_env("c", **chain_env())

        def worker():
            for _ in range(20):
                f.train_step("c", n_trajectories=4)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Final logZ should still be a finite number.
        rep = f.train_step("c", n_trajectories=4)
        assert math.isfinite(rep.logZ_estimate)

    def test_concurrent_sample(self):
        f = Flower(FlowerConfig(rng_seed=51))
        f.register_env("c", **chain_env())
        batches: list[SampleBatch] = []

        def worker():
            batches.append(f.sample("c", n=5))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(batches) == 4
        for b in batches:
            assert b.n == 5


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_returns_serialisable_dict(self):
        f = Flower(FlowerConfig(rng_seed=60))
        f.register_env("c", **chain_env())
        for _ in range(3):
            f.train_step("c", n_trajectories=2)
        snap = f.snapshot("c")
        assert "logZ" in snap
        assert "theta" in snap
        assert "edge_counts" in snap
        assert "terminal_counts" in snap
        assert snap["env"] == "c"
        # All keys serialisable as strings.
        for k in snap["theta"]:
            assert isinstance(k, str)


# ---------------------------------------------------------------------------
# Grid acceptance test: GFlowNet on a grid DAG
# ---------------------------------------------------------------------------


class TestGridAcceptance:
    def test_grid_corner_reward_recovered(self):
        f = Flower(FlowerConfig(loss=LOSS_TRAJECTORY_BALANCE, learning_rate=0.05, rng_seed=80))
        f.register_env("grid", **grid_env(rows=3, cols=3))
        for _ in range(300):
            f.train_step("grid", n_trajectories=16, epsilon=0.1)
        cov = f.mode_coverage("grid", n_samples=500, top_k=1)
        # Only one terminal in this grid; coverage probability ≈ 1.
        assert cov.mode_coverage_lcb > 0.85
        # And top mode recovered.
        assert cov.top_k_recovered == (1, 1)


# ---------------------------------------------------------------------------
# Coordination engine integration sanity
# ---------------------------------------------------------------------------


class TestCoordinationContract:
    def test_event_payloads_are_jsonable(self):
        """Coordination engines hand-off via JSON over a queue.
        Every event payload must be JSON-serialisable.
        """
        import json

        events: list[tuple[str, dict]] = []
        f = Flower(publisher=collect_events(events))
        f.register_env("c", **chain_env())
        f.train_step("c", n_trajectories=2)
        batch = f.sample("c", n=3)
        f.mode_coverage("c", n_samples=4, top_k=1)
        for kind, payload in events:
            assert isinstance(kind, str)
            # We can JSON-dump as long as terminal states are
            # primitive-like; chain envs use int states which is fine.
            json.dumps(payload)

    def test_fingerprint_is_hex_sha256(self):
        f = Flower()
        f.register_env("c", **chain_env())
        rep = f.train_step("c", n_trajectories=2)
        assert len(rep.fingerprint) == 64
        int(rep.fingerprint, 16)  # parses as hex
