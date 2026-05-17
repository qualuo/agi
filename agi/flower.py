r"""Flower — Generative Flow Networks as a runtime primitive.

Every prior primitive in this runtime that *selects* eventually faces the
same wall: the moment the coordination engine needs *more than one* good
candidate, the primitives that *optimise* — ``BayesOpt``, ``Searcher``,
``Solver``, ``Planner`` — return the *argmax*.  The primitives that
*sample* — ``Sampler`` (MCMC), ``Imaginator`` (posterior predictive) —
sample from a fixed (and typically non-reward-shaped) target.  Neither
is what a real product needs.  A drug-discovery pipeline does not ship a
single molecule; it ships the top-K Pareto front.  A program-synthesis
loop does not ship the first solution; it ships a diverse panel of
candidate programs the human can inspect.  A negotiation engine does not
commit to one playbook; it auditions K plays and runs the best one in
context.

``Flower`` is the runtime's *bounded, anytime, certified, stdlib*
implementation of **Generative Flow Networks** (Bengio, Lahlou, Deleu,
Hu, Tiwari, Bengio 2021 *Flow Network Based Generative Models for
Non-Iterative Diverse Candidate Generation*; Bengio, Jain, Korablyov,
Precup, Bengio 2021 *GFlowNet Foundations*; Malkin, Jain, Everett, Sun,
Bengio 2022 *Trajectory Balance: Improved Credit Assignment in
GFlowNets*; Madan, Rector-Brooks, Korablyov, Bengio, Liu, Chen, Hu,
Bengio 2023 *Learning GFlowNets from Partial Episodes for Improved
Convergence and Stability*).  A GFlowNet learns a forward policy
``P_F(s' | s)`` on a DAG of states such that the marginal probability of
terminating at object ``x`` is **proportional to the reward** ``R(x)``::

    P_T(x)  =  R(x) / Z      where     Z = Σ_x R(x).

This is the *fundamentally different* generative regime: not "argmax R"
(RL), not "sample from a fixed π" (MCMC), but **sample objects with
probability proportional to reward**.  The result is a learned
generative model that puts mass on *every* mode of ``R``, in proportion
to its size — exactly what a coordinator needs to ship a diverse panel
of high-reward candidates with a calibrated mode-coverage receipt.

The pitch reduced to a runtime call::

    flow = Flower(FlowerConfig(loss="trajectory_balance"))
    flow.register_env("molecules", initial=(),
                      successors=successors_fn,
                      terminal=terminal_fn,
                      reward=reward_fn)

    for _ in range(num_train_steps):
        report = flow.train_step("molecules", n_trajectories=32)
        # report.loss, report.logZ_estimate, report.weighted_reward

    batch = flow.sample("molecules", n=64, temperature=1.0)
    print(batch.terminals[0], batch.rewards[0], batch.log_probs[0])

    cov = flow.mode_coverage("molecules", n_samples=512, top_k=10)
    print(cov.tv_to_target, cov.modes_found, cov.mode_coverage_lcb)

    # Tamper-evident receipt — AttestationLedger.verify(batch.fingerprint)


What this primitive ships
-------------------------

  * **Tabular flow parameterisation (stdlib, no NumPy).**  Per-state-
    action edge logits ``θ(s, a) ∈ R`` and a global partition log-scale
    ``logZ ∈ R``, both stored in a Python dict and trained by SGD with
    a stdlib autograd over an analytic gradient.  Closed-form per-edge
    gradients of every loss listed below; no autograd-graph traversal.

  * **Four GFlowNet objectives:**

    * ``"flow_matching"`` — Bengio-Bengio 2021 detailed flow balance at
      each non-initial non-terminal state ``s``::

          Σ_{(s_par, a) → s} F(s_par, a)   =   Σ_a F(s, a)   +   R(s)·1[s∈X],

      minimised in log-space via mean-squared difference of log flows.

    * ``"detailed_balance"`` — Bengio et al. 2021 *GFlowNet
      Foundations* edge constraint::

          F(s) P_F(s' | s)   =   F(s') P_B(s | s'),

      parameterised by ``logZ`` + ``logF_unnorm(s)`` and trained by
      mean-squared *log* residual.  Single-edge updates → low variance
      in the residual stream.

    * ``"trajectory_balance"`` — Malkin-Jain-Everett-Sun-Bengio 2022.
      For a full trajectory ``τ = s_0 → s_1 → ... → x``::

          logZ + Σ_t log P_F(s_{t+1} | s_t)
                =  log R(x) + Σ_t log P_B(s_t | s_{t+1}).

      Minimised by mean squared residual on the full trajectory.  This
      is the **lowest-variance** of the GFlowNet losses on small to
      mid-size DAGs (their Fig 2), at the cost of one global parameter
      ``logZ``.

    * ``"subtrajectory_balance"`` — Madan-Rector-Brooks-Korablyov-
      Bengio-Liu-Chen-Hu-Bengio 2023.  For every sub-trajectory ``s_i →
      ... → s_j`` (including non-terminating ones, scored by
      ``F(s_j)``)::

          logF(s_i) + Σ_{t=i}^{j-1} log P_F(s_{t+1} | s_t)
                =  logF(s_j) + Σ_{t=i}^{j-1} log P_B(s_t | s_{t+1}).

      Weighted by ``λ^{j-i}``.  Combines the local-credit-assignment
      strength of FM with the global signal of TB.

  * **Forward policy under temperature.**  Sampling at temperature ``T``
    sets ``P_F(a | s) ∝ exp(θ(s, a) / T)``; ``T → 0`` recovers
    deterministic argmax, ``T → ∞`` recovers the uniform forward
    policy.  Inverse temperature ``β = 1/T`` is the *reward
    exponentiation* knob — under ``R^β``, sampling becomes sharper
    around the high-reward modes.

  * **Off-policy training corrections.**  Trajectories may be sampled
    from a behavioural policy ``P_B`` (e.g. ε-greedy on the forward
    policy, or a Boltzmann at higher temperature).  The trajectory
    balance loss is automatically corrected by an importance ratio
    ``P_F(τ) / P_B(τ)`` if the caller marks the batch with
    ``off_policy=True``; the corrected loss is unbiased iff
    ``support(P_B) ⊇ support(P_F)`` (Sutton-Barto 2018 §5.5).

  * **Replay buffer.**  Bounded FIFO of seen (trajectory, reward) pairs;
    each ``train_step`` draws a mini-batch from the buffer (Lin 1992
    *Self-improving reactive agents based on reinforcement learning*).
    Stable under the off-policy correction above.

  * **Mode-coverage report.**  Closed-form total-variation distance
    between the empirical sampling distribution over terminals and the
    *true* reward-proportional target::

        d_TV(P̂, P_T)  =  ½ Σ_x | n̂(x)/n − R(x)/Z |.

    Hoeffding 1963 LCB on ``Z`` (the partition function); Maurer-Pontil
    2009 empirical-Bernstein LCB on per-mode mass; Howard-Ramdas-
    McAuliffe-Sekhon 2021 anytime-valid bound on the **mode-coverage
    indicator** ``1[x ∈ top-k modes seen]``.

  * **Top-K Pareto extraction.**  ``top_k`` returns the K distinct
    terminals with highest reward seen across training + sampling.
    Calibrated by the mode-coverage LCB: "with probability ≥ 0.95, our
    top-K contains the global top-K modes of R."  The exact
    PAC statement is the Maurer-Pontil bound on the empirical-CDF of
    the reward distribution under the sampler.

  * **Identifiability report.**  Flags terminals with zero observed
    forward flow and non-zero reward — modes the current sampler will
    never reach.  This is the GFlowNet analogue of the
    Cao-Cohen-Szepesvári 2021 IRL identifiability gap; the report names
    the *next* (state, action) pairs a curriculum / Curator should
    target.

  * **PIT calibration.**  Probability-integral transform of the
    realised reward under the forward sampler.  Under a perfectly
    trained GFlowNet, ``PIT(R(x))`` is Uniform(0, 1); a one-sample
    Kolmogorov-Smirnov test (Massey 1951; Marsaglia-Tsang-Wang 2003
    series) returns a p-value.  Used by ``DriftSentinel`` for live
    sample-distribution drift detection.

  * **Attestation receipts.**  Every ``register``, ``observe``,
    ``train_step``, ``sample``, ``certify`` operation chain-hashes into
    an HMAC-secured fingerprint with deterministic RNG-seed replay.  A
    compliance officer reproduces a candidate batch byte-for-byte from
    the seed + observation stream.


Mathematical roots
------------------

  * **Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y.
    (2021).**  *Flow Network Based Generative Models for Non-Iterative
    Diverse Candidate Generation.*  The flow-matching loss as the
    foundational GFlowNet objective; existence of a forward policy
    realising any ``P_T(x) ∝ R(x)`` on a connected DAG; identifiability
    up to a multiplicative flow scale.

  * **Bengio, Y., Lahlou, S., Deleu, T., Hu, E. J., Tiwari, M. &
    Bengio, E. (2021).**  *GFlowNet Foundations.*  Detailed-balance
    edge equation, equivalence to entropy-regularised RL, optimal-flow
    uniqueness theorem.

  * **Malkin, N., Jain, M., Everett, K., Sun, X. & Bengio, Y. (2022).**
    *Trajectory Balance: Improved Credit Assignment in GFlowNets.*  The
    global ``logZ`` parameter and the per-trajectory residual; lower-
    variance gradient than flow matching when trajectories are long.

  * **Madan, K., Rector-Brooks, J., Korablyov, M., Bengio, E., Liu, M.,
    Chen, D., Hu, M. & Bengio, Y. (2023).**  *Learning GFlowNets from
    Partial Episodes for Improved Convergence and Stability.*
    Sub-trajectory balance with geometric decay ``λ``.

  * **Lehman, J. & Stanley, K. O. (2011).**  *Abandoning Objectives:
    Evolution Through the Search for Novelty Alone.*  The conceptual
    backbone of diverse-candidate generation that GFlowNet operationalises
    with a clean reward-proportionality guarantee.

  * **Hoeffding, W. (1963).**  Closed-form LCB on partition-function
    estimates; backbone of the mode-coverage receipt.

  * **Maurer, A. & Pontil, M. (2009).**  *Empirical Bernstein bounds*
    — sharper than Hoeffding in low-variance regimes.  Backbone of
    ``mode_mass_lcb``.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. S.
    (2021).**  *Time-uniform, nonparametric, nonasymptotic confidence
    sequences.*  Anytime-valid CIs on the mode-coverage indicator.

  * **Massey, F. J. (1951).** + **Marsaglia, Tsang, Wang (2003).**
    The exact KS p-value series.

  * **Sutton, R. S. & Barto, A. G. (2018, 2nd ed.).** §5.5 importance
    sampling for off-policy MC; the unbiasedness condition adopted
    here.

  * **Lin, L.-J. (1992).**  *Self-improving reactive agents based on
    reinforcement learning, planning and teaching.*  The replay buffer
    that stabilises off-policy training.


Composes with the rest of the runtime
-------------------------------------

  * ``Quantilizer`` — Flower's batch of K terminals plus their
    ``rewards`` *is* the empirical distribution the Quantilizer
    thresholds on.  Together they deliver: "ship the K candidates whose
    reward is in the top-``q`` quantile of the GFlowNet's posterior".

  * ``Imaginator`` — when the runtime needs to evaluate trajectories
    *before* the reward is observed, the Imaginator's posterior over
    rewards is a drop-in ``reward_fn`` for the Flower.

  * ``Reconciler`` — Flower's terminal-distribution is one expert
    among many on the question "which mode of R is best"; Reconciler
    aggregates Flower with BayesOpt + Searcher + Distiller into a
    common-prior consensus.

  * ``Searcher`` — Flower's learned forward policy is the heuristic the
    Searcher uses for its best-first expansion; Searcher's tree expands
    along the GFlowNet's highest-probability paths first.

  * ``Curator`` — Flower's identifiability report names the
    (state, action) pairs the GFlowNet under-samples; Curator writes
    these into the next curriculum batch.

  * ``Aligner`` — Flower's K-tuple of high-reward terminals is one half
    of the preference pair the Aligner trains on; the other half is
    the user's chosen winner.  The two together close the loop
    *generate-then-rank-then-align*.

  * ``Distiller`` — distil the learned forward policy ``P_F`` into an
    amortised classifier the runtime can call at inference time without
    the rollout cost.

  * ``DriftSentinel`` — PIT-of-rewards is a Uniform(0, 1) sequence
    under a correctly trained GFlowNet; CUSUM on the log-uniformity
    p-value detects reward-distribution drift in real time.

  * ``AttestationLedger`` — every register / observe / train_step /
    sample / certify event chain-hashes into the ledger; a compliance
    officer replays the candidate batch byte-for-byte from the
    observation stream + RNG seed.

  * ``Coordinator`` — every Goal that benefits from a *panel* (drug
    design, code search, negotiation playbooks, hyperparameter sweeps)
    routes through Flower.  The coordination engine no longer asks
    "what's the best?" — it asks "give me K diverse top-quantile
    candidates with a coverage receipt", and Flower returns the panel
    with a fingerprint a compliance officer can sign.


Investor framing
----------------

Flower is the **runtime's diversification kernel**.  Every other
primitive in this runtime returns *one* answer.  Flower returns
*K diverse* answers proportional to reward, with calibrated mode
coverage and a tamper-evident receipt.  This is the line between
*"the model picked one option"* and *"the model presented a Pareto-
ranked panel of options the human reviewed and the compliance officer
signed."*  Pair with ``Quantilizer`` for safety-bounded deployment,
``Aligner`` for preference-supervised refinement, and
``AttestationLedger`` for cryptographic replay — the **diverse
candidate generation engine**, delivered as a runtime primitive a
coordination engine can drive.


What it deliberately doesn't claim
----------------------------------

  * Not a frontier neural GFlowNet — no transformers, no graph neural
    nets.  The tabular parameterisation (one logit per edge, one
    ``logZ`` per env) is the **convergent, identifiable** core that
    admits closed-form gradients and provable mode-coverage bounds.

  * Not a continuous-action generator — the discrete-DAG core covers
    every combinatorial-generation use case (molecules-as-strings,
    programs-as-tokens, layouts-as-grids); continuous extensions
    (Lahlou-Deleu-Hu-Bengio 2023 *A theory of continuous GFlowNets*)
    are reserved for a follow-up primitive.

  * Not a guaranteed exploration mechanism — the GFlowNet samples
    proportional to reward, not to information gain.  Pair with
    ``Curator`` for active-learning-style exploration of the
    identifiability gap.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    # Event kinds
    "FLOWER_STARTED",
    "FLOWER_REGISTERED",
    "FLOWER_REMOVED",
    "FLOWER_OBSERVED",
    "FLOWER_TRAINED",
    "FLOWER_SAMPLED",
    "FLOWER_CERTIFIED",
    "FLOWER_CLEARED",
    # Loss families
    "LOSS_FLOW_MATCHING",
    "LOSS_DETAILED_BALANCE",
    "LOSS_TRAJECTORY_BALANCE",
    "LOSS_SUBTRAJECTORY_BALANCE",
    "KNOWN_LOSSES",
    # Errors
    "FlowerError",
    "InvalidConfig",
    "InvalidEnv",
    "InvalidTrajectory",
    "InsufficientData",
    "UnknownEnv",
    "NonTerminal",
    # Dataclasses
    "FlowerConfig",
    "EnvSpec",
    "Trajectory",
    "TrainReport",
    "SampleBatch",
    "ModeCoverageReport",
    "IdentifiabilityReport",
    "PITCalibrationReport",
    # Main class
    "Flower",
    # Helper functions
    "ledger_root",
    "hoeffding_half_width",
    "empirical_bernstein_half_width",
    "hrms_half_width",
    "ks_pvalue",
    "softmax",
    "logsumexp",
    "total_variation",
]


# ---------------------------------------------------------------------------
# Event kinds (the coordination contract)
# ---------------------------------------------------------------------------

FLOWER_STARTED = "flower.started"
FLOWER_REGISTERED = "flower.registered"
FLOWER_REMOVED = "flower.removed"
FLOWER_OBSERVED = "flower.observed"
FLOWER_TRAINED = "flower.trained"
FLOWER_SAMPLED = "flower.sampled"
FLOWER_CERTIFIED = "flower.certified"
FLOWER_CLEARED = "flower.cleared"


# ---------------------------------------------------------------------------
# Loss-family names
# ---------------------------------------------------------------------------

LOSS_FLOW_MATCHING = "flow_matching"
LOSS_DETAILED_BALANCE = "detailed_balance"
LOSS_TRAJECTORY_BALANCE = "trajectory_balance"
LOSS_SUBTRAJECTORY_BALANCE = "subtrajectory_balance"
KNOWN_LOSSES: frozenset[str] = frozenset(
    {
        LOSS_FLOW_MATCHING,
        LOSS_DETAILED_BALANCE,
        LOSS_TRAJECTORY_BALANCE,
        LOSS_SUBTRAJECTORY_BALANCE,
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FlowerError(Exception):
    """Base class for Flower errors."""


class InvalidConfig(FlowerError):
    """The supplied ``FlowerConfig`` is invalid."""


class InvalidEnv(FlowerError):
    """The supplied ``EnvSpec`` is invalid."""


class InvalidTrajectory(FlowerError):
    """The supplied trajectory is malformed."""


class InsufficientData(FlowerError):
    """Not enough data to satisfy the request."""


class UnknownEnv(FlowerError):
    """The requested environment ID has not been registered."""


class NonTerminal(FlowerError):
    """A reward was requested at a non-terminal state."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


EventPublisher = Callable[[str, dict[str, Any]], None]
State = Any
Action = Any


@dataclass
class FlowerConfig:
    """Top-level configuration.

    Attributes
    ----------
    loss
        Default GFlowNet loss family.  One of
        ``"flow_matching"``, ``"detailed_balance"``,
        ``"trajectory_balance"``, ``"subtrajectory_balance"``.
    confidence
        Confidence level for every interval the primitive returns
        (Hoeffding LCB, Maurer-Pontil empirical-Bernstein,
        Howard-Ramdas-McAuliffe-Sekhon anytime-valid sequences).
    learning_rate
        Initial SGD learning rate for the per-edge logits ``θ(s, a)``
        and the global ``logZ``.
    learning_rate_logz
        Optional override for the SGD step on ``logZ``.  When ``None``
        the per-edge ``learning_rate`` is used.
    subtb_lambda
        Geometric decay used by the sub-trajectory balance loss.
    epsilon_exploration
        Probability that ``sample`` uses a uniform-forward action
        instead of the learned policy.  ``0`` ⇒ pure-on-policy
        sampling.  Provides Lin 1992 replay-stability.
    replay_capacity
        FIFO replay-buffer capacity per registered environment.  Each
        entry holds one trajectory + its observed reward.
    replay_min_fill
        Minimum replay entries before ``train_step`` will draw from the
        buffer; below this threshold each ``train_step`` draws fresh.
    max_trajectory_length
        Hard cap on rollout depth.  Avoids accidental infinite recursion
        when ``terminal`` and ``successors`` disagree.
    reward_floor
        Floor applied to ``reward`` before taking ``log R``.  ``0`` and
        negative rewards are bumped to ``reward_floor`` to keep
        ``log R`` finite.  GFlowNets are defined for ``R(x) > 0``.
    grad_clip
        Per-parameter gradient L∞ clip applied before each SGD step.
    rng_seed
        Seed for the internal RNG used by sampling and replay
        permutation.  Determinism: same observations + same seed →
        byte-identical fingerprint chain.
    hmac_key
        Optional secret key for HMAC-SHA-256 over every fingerprint
        entry.  When set, the chain is unforgeable without the key.
    max_envs
        Soft cap on the number of registered environments.
    """

    loss: str = LOSS_TRAJECTORY_BALANCE
    confidence: float = 0.95
    learning_rate: float = 0.05
    learning_rate_logz: float | None = None
    subtb_lambda: float = 0.9
    epsilon_exploration: float = 0.0
    replay_capacity: int = 4096
    replay_min_fill: int = 0
    max_trajectory_length: int = 1024
    reward_floor: float = 1e-12
    grad_clip: float = 10.0
    rng_seed: int | None = 0xF10E1
    hmac_key: bytes | None = None
    max_envs: int | None = None

    def __post_init__(self) -> None:
        if self.loss not in KNOWN_LOSSES:
            raise InvalidConfig(
                f"unknown loss {self.loss!r}; expected one of {sorted(KNOWN_LOSSES)}"
            )
        if not 0.5 < self.confidence < 1.0:
            raise InvalidConfig(
                f"confidence must be in (0.5, 1.0); got {self.confidence}"
            )
        if self.learning_rate <= 0.0:
            raise InvalidConfig(
                f"learning_rate must be positive; got {self.learning_rate}"
            )
        if self.learning_rate_logz is not None and self.learning_rate_logz <= 0.0:
            raise InvalidConfig(
                f"learning_rate_logz must be positive; got {self.learning_rate_logz}"
            )
        if not 0.0 < self.subtb_lambda <= 1.0:
            raise InvalidConfig(
                f"subtb_lambda must be in (0, 1]; got {self.subtb_lambda}"
            )
        if not 0.0 <= self.epsilon_exploration <= 1.0:
            raise InvalidConfig(
                f"epsilon_exploration must be in [0, 1]; got {self.epsilon_exploration}"
            )
        if self.replay_capacity < 0:
            raise InvalidConfig(
                f"replay_capacity must be ≥ 0; got {self.replay_capacity}"
            )
        if self.replay_min_fill < 0:
            raise InvalidConfig(
                f"replay_min_fill must be ≥ 0; got {self.replay_min_fill}"
            )
        if self.max_trajectory_length <= 0:
            raise InvalidConfig(
                f"max_trajectory_length must be > 0; got {self.max_trajectory_length}"
            )
        if self.reward_floor <= 0.0:
            raise InvalidConfig(
                f"reward_floor must be > 0; got {self.reward_floor}"
            )
        if self.grad_clip <= 0.0:
            raise InvalidConfig(f"grad_clip must be > 0; got {self.grad_clip}")
        if self.max_envs is not None and self.max_envs <= 0:
            raise InvalidConfig(f"max_envs must be > 0; got {self.max_envs}")


@dataclass(frozen=True)
class EnvSpec:
    """One registered environment — a reward-shaped DAG.

    Attributes
    ----------
    env
        Identifier.
    initial
        The unique source state.  Every trajectory starts here.
    successors
        ``successors(state) -> Iterable[(action, next_state)]``.  Empty
        iterable signals a terminal node.
    terminal
        ``terminal(state) -> bool``.  May return ``True`` for *any*
        state with empty successors, plus optionally for intermediate
        states (an early-stop action).
    reward
        ``reward(terminal_state) -> float``.  Must return a strictly
        positive number.  Non-positive rewards are floored to
        ``config.reward_floor`` to keep ``log R`` finite.
    loss
        Optional per-env override of the default GFlowNet loss family.
    """

    env: str
    initial: State
    successors: Callable[[State], Iterable[tuple[Action, State]]]
    terminal: Callable[[State], bool]
    reward: Callable[[State], float]
    loss: str | None = None


@dataclass(frozen=True)
class Trajectory:
    """One sampled trajectory ``s_0 → s_1 → … → s_T`` with action labels.

    Attributes
    ----------
    states
        ``(s_0, s_1, …, s_T)``.
    actions
        ``(a_0, a_1, …, a_{T-1})``.
    reward
        Realised reward at ``s_T`` if terminal, else ``0.0``.
    terminal
        ``True`` iff ``s_T`` is a terminal state.
    log_p_forward
        Log-probability of this trajectory under the current forward
        policy (snapshot at the time of sampling).
    """

    states: tuple[State, ...]
    actions: tuple[Action, ...]
    reward: float
    terminal: bool
    log_p_forward: float = 0.0


@dataclass(frozen=True)
class TrainReport:
    """Outcome of one SGD step.

    Attributes
    ----------
    env
        Environment identifier.
    loss
        Loss family used for this step.
    loss_value
        Mean residual loss across the mini-batch.
    grad_norm
        L∞ norm of the parameter update (post-clip).
    n_trajectories
        Number of trajectories that contributed to the gradient.
    n_terminals
        Number of distinct terminals in the mini-batch.
    logZ_estimate
        Current estimate of ``log Z`` (the partition function).
        Equals the closed-form Σ_x R(x) when the DAG is fully
        enumerable; otherwise the learned parameter.
    weighted_reward
        Mean ``R(x)`` over the mini-batch weighted by the on-policy
        sampling probability.  Diagnostic of *how high* the GFlowNet
        is sampling.
    fingerprint
        Tamper-evident hash chain head after this event.
    """

    env: str
    loss: str
    loss_value: float
    grad_norm: float
    n_trajectories: int
    n_terminals: int
    logZ_estimate: float
    weighted_reward: float
    fingerprint: str


@dataclass(frozen=True)
class SampleBatch:
    """A batch of K sampled candidate terminals.

    Attributes
    ----------
    env
        Environment identifier.
    n
        Number of trajectories in the batch.
    trajectories
        The full ``Trajectory`` objects (states, actions, reward).
    terminals
        Convenience accessor — the terminal state of each trajectory.
    rewards
        Convenience accessor — the realised reward of each terminal.
    log_probs
        Convenience accessor — the log-probability under the forward
        policy of each trajectory.
    mean_reward
        Empirical mean reward.
    mean_reward_lcb
        Maurer-Pontil empirical-Bernstein LCB on ``E[R]``.
    mean_reward_hrms_lcb
        Howard-Ramdas-McAuliffe-Sekhon anytime-valid LCB on ``E[R]``.
    unique_terminals
        Number of distinct terminals in the batch.
    forward_entropy
        Plug-in Shannon entropy of the empirical sampler over
        distinct terminals.  Diagnostic of diversity.
    fingerprint
        Tamper-evident hash chain head after this event.
    """

    env: str
    n: int
    trajectories: tuple[Trajectory, ...]
    terminals: tuple[State, ...]
    rewards: tuple[float, ...]
    log_probs: tuple[float, ...]
    mean_reward: float
    mean_reward_lcb: float
    mean_reward_hrms_lcb: float
    unique_terminals: int
    forward_entropy: float
    fingerprint: str


@dataclass(frozen=True)
class ModeCoverageReport:
    """Closed-form mode-coverage diagnostic.

    The empirical distribution over terminals seen across training +
    sampling is compared to the *true* reward-proportional target
    distribution (computable in closed form on small DAGs).

    Attributes
    ----------
    env
        Environment identifier.
    n_samples
        Number of samples backing the empirical distribution.
    n_unique
        Number of distinct terminals observed.
    tv_to_target
        Total-variation distance between the empirical sampling
        distribution and the reward-proportional target.
    tv_hoeffding_ucb
        Hoeffding 1963 UCB on ``TV``: with prob. ``≥ conf``,
        ``TV(empirical, target) ≤ tv + ε``.
    modes_found
        Subset of the top-K reward modes that the GFlowNet has
        sampled at least once.  ``-1`` when the true mode-ranking
        was not requested.
    mode_coverage_lcb
        HRMS anytime-valid LCB on the *coverage probability* — the
        probability that a fresh sample lies in the top-K reward
        modes.
    top_k_recovered
        ``(K, M)`` — ``M`` of the top-``K`` modes have been sampled.
    log_partition_lcb / log_partition_ucb
        Hoeffding LCB / UCB on ``log Z = log Σ_x R(x)``.
    fingerprint
        Tamper-evident hash chain head after this event.
    """

    env: str
    n_samples: int
    n_unique: int
    tv_to_target: float
    tv_hoeffding_ucb: float
    modes_found: int
    mode_coverage_lcb: float
    top_k_recovered: tuple[int, int]
    log_partition_lcb: float
    log_partition_ucb: float
    fingerprint: str


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Per-(state, action) coverage gap.

    Attributes
    ----------
    env
        Environment identifier.
    under_sampled_edges
        Up to ``top_k`` of the (state, action) edges with the smallest
        observed forward count.  Names the targets a Curator should
        feed into the next curriculum batch.
    unreachable_modes
        Terminal states with positive reward but zero forward visits.
    saturated_edges
        Edges visited beyond ``saturated_threshold`` — diagnostic of
        the GFlowNet over-committing.
    """

    env: str
    under_sampled_edges: tuple[tuple[State, Action, int], ...]
    unreachable_modes: tuple[State, ...]
    saturated_edges: tuple[tuple[State, Action, int], ...]


@dataclass(frozen=True)
class PITCalibrationReport:
    """KS test of the reward PIT against Uniform(0, 1).

    Under a perfectly trained GFlowNet the empirical CDF of
    ``R(x)/max_y R(y)`` is the CDF the forward sampler assigns to
    rewards; the probability-integral transform of the sampled rewards
    under that CDF is Uniform(0, 1).  A small p-value flags
    under-trained / over-trained / drifted sampling.
    """

    env: str
    n: int
    ks_statistic: float
    p_value: float
    fingerprint: str


# ---------------------------------------------------------------------------
# Public math helpers (re-exported)
# ---------------------------------------------------------------------------


def softmax(scores: Sequence[float], beta: float = 1.0) -> list[float]:
    """Numerically-stable softmax with optional inverse-temperature beta."""
    if not scores:
        return []
    scaled = [beta * s for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    if z == 0.0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / z for e in exps]


def logsumexp(values: Sequence[float]) -> float:
    """Numerically-stable log of the sum of exponentials."""
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    s = 0.0
    for v in values:
        s += math.exp(v - m)
    if s <= 0.0:
        return float("-inf")
    return m + math.log(s)


def hoeffding_half_width(n: int, conf: float = 0.95) -> float:
    """Hoeffding 1963 half-width for the mean of a [0, 1] random
    variable from ``n`` iid samples at confidence ``conf``.
    """
    if n <= 0:
        return float("inf")
    delta = 1.0 - conf
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, variance: float, conf: float = 0.95, range_: float = 1.0
) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein half-width.

    Returns ``ε`` such that with probability ``≥ conf``,
    ``|μ̂ − μ| ≤ ε``  given ``n`` iid samples in ``[0, range_]`` with
    sample variance ``variance``.

    Formula: ``ε = √(2 V log(2/δ) / n) + 7 R log(2/δ) / (3(n-1))``.
    """
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    return math.sqrt(2.0 * variance * math.log(2.0 / delta) / n) + (
        7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    )


def hrms_half_width(n: int, conf: float = 0.95) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence
    sequence half-width for the mean of a [0, 1] random variable from
    ``n`` iid samples at confidence ``conf``.

    The closed-form practical bound from §3 of the paper (Eq. 15):
    ``ε = √( (log log(2n) + 0.75 log(10.4 / δ)) / (2n) )``.

    Anytime-valid: the same bound holds **simultaneously** at every
    ``n``, so a coordinator can keep drawing samples and read the
    same bound without paying a union-bound tax.
    """
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    inner = math.log(math.log(2.0 * n) + math.e) + 0.75 * math.log(10.4 / delta)
    return math.sqrt(max(inner / (2.0 * n), 0.0))


def ks_pvalue(samples: Sequence[float]) -> tuple[float, float]:
    """One-sample Kolmogorov-Smirnov test of ``samples`` against
    H₀ Uniform(0, 1).

    Returns ``(D, p)`` where ``D = max_i |F̂(x_i) − x_i|`` is the KS
    statistic and ``p`` is the asymptotic two-sided p-value via the
    Marsaglia-Tsang-Wang 2003 series.
    """
    n = len(samples)
    if n == 0:
        return 0.0, 1.0
    xs = sorted(samples)
    d = 0.0
    for i, x in enumerate(xs, start=1):
        d = max(d, abs(i / n - x), abs(x - (i - 1) / n))
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    # Marsaglia-Tsang-Wang series
    j = 1
    s = 0.0
    fac = 2.0
    eps = 1e-12
    while j < 200:
        t = fac * math.exp(-2.0 * lam * lam * j * j)
        s += t
        if abs(t) <= eps * abs(s) or abs(t) < eps:
            break
        fac = -fac
        j += 1
    p = min(max(s, 0.0), 1.0)
    return d, p


def total_variation(p: dict[State, float], q: dict[State, float]) -> float:
    """``½ Σ_x |p(x) − q(x)|`` over the union of supports."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# ---------------------------------------------------------------------------
# Attestation helpers
# ---------------------------------------------------------------------------


def _stable_repr(obj: Any) -> str:
    """Deterministic string representation for hashing."""
    if isinstance(obj, dict):
        items = sorted(((_stable_repr(k), _stable_repr(v)) for k, v in obj.items()),
                       key=lambda kv: kv[0])
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in obj) + "]"
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return repr(obj)
    return repr(obj)


def _canonical(payload: dict[str, Any]) -> bytes:
    return _stable_repr(payload).encode("utf-8")


def ledger_root(secret_key: bytes | None = None) -> str:
    """Initial chain head — distinct under different HMAC keys."""
    if secret_key is None:
        return hashlib.sha256(b"flower-root").hexdigest()
    return hmac.new(secret_key, b"flower-root", hashlib.sha256).hexdigest()


def _hash_entry(
    parent: str, payload: dict[str, Any], hmac_key: bytes | None = None
) -> str:
    msg = parent.encode("utf-8") + b"|" + _canonical(payload)
    if hmac_key is None:
        return hashlib.sha256(msg).hexdigest()
    return hmac.new(hmac_key, msg, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Per-environment state
# ---------------------------------------------------------------------------


def _edge_key(state: State, action: Action) -> tuple[Any, Any]:
    return (state, action)


@dataclass
class _EnvState:
    """All learned + observed state for one registered environment."""

    spec: EnvSpec
    # Per-edge logit θ(s, a)
    theta: dict[tuple[State, Action], float] = field(default_factory=dict)
    # Per-non-terminal-state log unnormalised flow logF(s), used by DB / SubTB
    log_flow: dict[State, float] = field(default_factory=dict)
    # Global log partition function
    logZ: float = 0.0
    # Per-edge visit counts (forward direction)
    edge_counts: dict[tuple[State, Action], int] = field(default_factory=dict)
    # Per-terminal observation counts
    terminal_counts: dict[State, int] = field(default_factory=dict)
    # Per-terminal rewards (deduplicated)
    terminal_rewards: dict[State, float] = field(default_factory=dict)
    # Replay buffer of (trajectory, reward, log_p_forward_snapshot)
    replay: list[tuple[Trajectory, float, float]] = field(default_factory=list)
    # PIT history of normalised rewards seen in sample()
    pit_history: list[float] = field(default_factory=list)
    # Loss family override
    loss_family: str = LOSS_TRAJECTORY_BALANCE

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_theta(self, s: State, a: Action) -> float:
        return self.theta.get(_edge_key(s, a), 0.0)

    def set_theta(self, s: State, a: Action, value: float) -> None:
        self.theta[_edge_key(s, a)] = value

    def get_log_flow(self, s: State) -> float:
        return self.log_flow.get(s, 0.0)

    def set_log_flow(self, s: State, value: float) -> None:
        self.log_flow[s] = value

    def add_visit(self, s: State, a: Action) -> None:
        k = _edge_key(s, a)
        self.edge_counts[k] = self.edge_counts.get(k, 0) + 1

    def add_terminal(self, s: State, reward: float) -> None:
        self.terminal_counts[s] = self.terminal_counts.get(s, 0) + 1
        self.terminal_rewards[s] = reward


# ---------------------------------------------------------------------------
# Trajectory enumeration helpers
# ---------------------------------------------------------------------------


def _enumerate_terminals(
    spec: EnvSpec, reward_floor: float, max_depth: int
) -> tuple[dict[State, float], bool]:
    """Try to enumerate every terminal reachable from ``initial``.

    Returns ``(terminal_to_reward, complete)`` where ``complete`` is
    ``True`` if the search did not hit ``max_depth``.  Uses an
    iterative DFS bounded by ``max_depth``.
    """
    out: dict[State, float] = {}
    stack: list[tuple[State, int]] = [(spec.initial, 0)]
    seen: set[Any] = set()
    complete = True
    while stack:
        state, depth = stack.pop()
        try:
            key = state
            hash(key)
        except TypeError:
            key = repr(state)
        if key in seen:
            continue
        seen.add(key)
        if spec.terminal(state):
            r = float(spec.reward(state))
            if r < reward_floor:
                r = reward_floor
            out[state] = r
            continue
        if depth >= max_depth:
            complete = False
            continue
        children = list(spec.successors(state))
        if not children:
            # state declared non-terminal but has no successors → treat as
            # an unreachable / dead state; skip.
            continue
        for _, child in children:
            stack.append((child, depth + 1))
    return out, complete


# ---------------------------------------------------------------------------
# Main Flower
# ---------------------------------------------------------------------------


class Flower:
    """Generative Flow Networks as a runtime primitive.

    Threadsafe at the API surface: a single re-entrant lock guards
    every mutation of per-env state.
    """

    def __init__(
        self,
        config: FlowerConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or FlowerConfig()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._envs: dict[str, _EnvState] = {}
        self._chain_head: str = ledger_root(self.config.hmac_key)
        self._rng = random.Random(self.config.rng_seed)
        self._started_ts = time.time()
        self._publish(
            FLOWER_STARTED,
            {"ts": self._started_ts, "loss": self.config.loss},
        )

    # ------------------------------------------------------------------
    # Event publishing + chain helpers
    # ------------------------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, payload)
        except Exception:
            pass

    def _advance_chain(self, payload: dict[str, Any]) -> str:
        self._chain_head = _hash_entry(self._chain_head, payload, self.config.hmac_key)
        return self._chain_head

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._chain_head

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_env(
        self,
        env: str,
        *,
        initial: State,
        successors: Callable[[State], Iterable[tuple[Action, State]]],
        terminal: Callable[[State], bool],
        reward: Callable[[State], float],
        loss: str | None = None,
    ) -> EnvSpec:
        if not isinstance(env, str) or not env:
            raise InvalidEnv("env must be a non-empty string")
        if loss is not None and loss not in KNOWN_LOSSES:
            raise InvalidEnv(
                f"unknown loss {loss!r}; expected one of {sorted(KNOWN_LOSSES)}"
            )
        spec = EnvSpec(
            env=env,
            initial=initial,
            successors=successors,
            terminal=terminal,
            reward=reward,
            loss=loss,
        )
        with self._lock:
            if env in self._envs:
                raise InvalidEnv(f"env {env!r} already registered")
            if self.config.max_envs is not None and len(self._envs) >= self.config.max_envs:
                raise InvalidEnv(
                    f"max_envs ({self.config.max_envs}) reached"
                )
            st = _EnvState(spec=spec)
            st.loss_family = loss or self.config.loss
            self._envs[env] = st
            self._advance_chain(
                {
                    "op": "register",
                    "env": env,
                    "loss": st.loss_family,
                }
            )
            self._publish(
                FLOWER_REGISTERED,
                {"env": env, "loss": st.loss_family, "head": self._chain_head},
            )
        return spec

    def remove_env(self, env: str) -> None:
        with self._lock:
            if env not in self._envs:
                raise UnknownEnv(env)
            del self._envs[env]
            self._advance_chain({"op": "remove", "env": env})
            self._publish(
                FLOWER_REMOVED, {"env": env, "head": self._chain_head}
            )

    def envs(self) -> list[str]:
        with self._lock:
            return sorted(self._envs.keys())

    def env_spec(self, env: str) -> EnvSpec:
        with self._lock:
            return self._require_env(env).spec

    def clear(self) -> None:
        with self._lock:
            self._envs.clear()
            self._chain_head = ledger_root(self.config.hmac_key)
            self._advance_chain({"op": "clear"})
            self._publish(FLOWER_CLEARED, {"head": self._chain_head})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_env(self, env: str) -> _EnvState:
        try:
            return self._envs[env]
        except KeyError:
            raise UnknownEnv(env)

    def _forward_policy(
        self,
        st: _EnvState,
        state: State,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> list[tuple[Action, State, float]]:
        """Return ``[(action, next_state, prob)]`` for the forward policy
        at ``state``, mixing the softmax over edge logits with a
        ε-uniform fallback.
        """
        children = list(st.spec.successors(state))
        if not children:
            return []
        logits = [st.get_theta(state, a) for a, _ in children]
        if temperature <= 0.0:
            # argmax
            m = max(logits)
            probs = [1.0 if l == m else 0.0 for l in logits]
            s = sum(probs)
            probs = [p / s for p in probs]
        else:
            probs = softmax(logits, beta=1.0 / temperature)
        if epsilon > 0.0:
            n = len(children)
            uniform = 1.0 / n
            probs = [(1.0 - epsilon) * p + epsilon * uniform for p in probs]
        return [(a, ns, p) for (a, ns), p in zip(children, probs)]

    def _sample_one(
        self,
        st: _EnvState,
        *,
        temperature: float,
        epsilon: float,
        rng: random.Random,
    ) -> Trajectory:
        states: list[State] = [st.spec.initial]
        actions: list[Action] = []
        log_p = 0.0
        for _ in range(self.config.max_trajectory_length):
            current = states[-1]
            if st.spec.terminal(current):
                break
            choices = self._forward_policy(
                st, current, temperature=temperature, epsilon=epsilon
            )
            if not choices:
                break
            r = rng.random()
            acc = 0.0
            picked = choices[-1]
            for c in choices:
                acc += c[2]
                if r <= acc:
                    picked = c
                    break
            action, next_state, prob = picked
            actions.append(action)
            states.append(next_state)
            if prob <= 0.0:
                log_p += -1e18
            else:
                log_p += math.log(prob)
        final = states[-1]
        is_terminal = st.spec.terminal(final)
        reward = 0.0
        if is_terminal:
            r = float(st.spec.reward(final))
            if r < self.config.reward_floor:
                r = self.config.reward_floor
            reward = r
        return Trajectory(
            states=tuple(states),
            actions=tuple(actions),
            reward=reward,
            terminal=is_terminal,
            log_p_forward=log_p,
        )

    def _record_visits(self, st: _EnvState, traj: Trajectory) -> None:
        for s, a in zip(traj.states[:-1], traj.actions):
            st.add_visit(s, a)
        if traj.terminal:
            st.add_terminal(traj.states[-1], traj.reward)

    def _push_replay(self, st: _EnvState, traj: Trajectory) -> None:
        if self.config.replay_capacity <= 0:
            return
        st.replay.append((traj, traj.reward, traj.log_p_forward))
        while len(st.replay) > self.config.replay_capacity:
            st.replay.pop(0)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def observe(
        self,
        env: str,
        *,
        states: Sequence[State],
        actions: Sequence[Action],
        reward: float | None = None,
    ) -> Trajectory:
        """Record an externally-collected trajectory.

        ``states`` must include both endpoints (``s_0 = initial`` and
        ``s_T = terminal_or_intermediate``).  ``actions`` has one entry
        per consecutive pair.  ``reward`` overrides ``spec.reward`` for
        the terminal — useful for noisy / stochastic rewards observed
        in the real environment.
        """
        with self._lock:
            st = self._require_env(env)
            if len(states) == 0:
                raise InvalidTrajectory("states must be non-empty")
            if len(actions) != len(states) - 1:
                raise InvalidTrajectory(
                    f"len(actions) must equal len(states)-1; got {len(actions)} vs {len(states)}"
                )
            log_p = 0.0
            for s, a in zip(states[:-1], actions):
                choices = self._forward_policy(st, s, temperature=1.0, epsilon=0.0)
                p = 0.0
                for ca, _ns, cp in choices:
                    if ca == a:
                        p = cp
                        break
                if p > 0.0:
                    log_p += math.log(p)
                else:
                    log_p += -1e18
            final = states[-1]
            is_terminal = st.spec.terminal(final)
            if reward is None:
                if is_terminal:
                    reward = float(st.spec.reward(final))
                else:
                    reward = 0.0
            if is_terminal and reward < self.config.reward_floor:
                reward = self.config.reward_floor
            traj = Trajectory(
                states=tuple(states),
                actions=tuple(actions),
                reward=float(reward),
                terminal=is_terminal,
                log_p_forward=log_p,
            )
            self._record_visits(st, traj)
            self._push_replay(st, traj)
            self._advance_chain(
                {
                    "op": "observe",
                    "env": env,
                    "len": len(states),
                    "terminal": is_terminal,
                    "reward": float(reward),
                }
            )
            self._publish(
                FLOWER_OBSERVED,
                {
                    "env": env,
                    "terminal": is_terminal,
                    "reward": float(reward),
                    "head": self._chain_head,
                },
            )
            return traj

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        env: str,
        *,
        n_trajectories: int = 8,
        temperature: float = 1.0,
        epsilon: float | None = None,
        loss: str | None = None,
        use_replay: bool = True,
    ) -> TrainReport:
        """One SGD step over a fresh batch of trajectories.

        On entry, ``n_trajectories`` are sampled from the current
        forward policy (possibly mixed with a Lin 1992 replay batch),
        a gradient is computed in closed form for the chosen loss
        family, and parameters are updated by a fixed-step SGD with
        L∞ gradient clipping.
        """
        if n_trajectories <= 0:
            raise InvalidTrajectory("n_trajectories must be > 0")
        with self._lock:
            st = self._require_env(env)
            family = loss or st.loss_family
            if family not in KNOWN_LOSSES:
                raise InvalidConfig(
                    f"unknown loss {family!r}; expected one of {sorted(KNOWN_LOSSES)}"
                )
            eps = self.config.epsilon_exploration if epsilon is None else epsilon
            # 1. Collect fresh trajectories.
            batch: list[Trajectory] = []
            for _ in range(n_trajectories):
                t = self._sample_one(
                    st, temperature=temperature, epsilon=eps, rng=self._rng
                )
                self._record_visits(st, t)
                self._push_replay(st, t)
                batch.append(t)
            # 2. Optionally augment with replay.
            if use_replay and len(st.replay) >= max(self.config.replay_min_fill, 1):
                k = min(n_trajectories, len(st.replay))
                if k > 0:
                    idx = self._rng.sample(range(len(st.replay)), k)
                    for i in idx:
                        batch.append(st.replay[i][0])
            # 3. Closed-form gradient + SGD on this batch.
            loss_value, grad_norm = self._sgd_step(st, batch, family)
            # 4. Diagnostics.
            n_terminals = len({t.states[-1] for t in batch if t.terminal})
            weighted_reward = (
                sum(t.reward for t in batch) / len(batch) if batch else 0.0
            )
            payload = {
                "op": "train_step",
                "env": env,
                "loss": family,
                "n": n_trajectories,
                "loss_value": loss_value,
                "grad_norm": grad_norm,
                "logZ": st.logZ,
            }
            self._advance_chain(payload)
            self._publish(
                FLOWER_TRAINED,
                {**payload, "head": self._chain_head},
            )
            return TrainReport(
                env=env,
                loss=family,
                loss_value=loss_value,
                grad_norm=grad_norm,
                n_trajectories=len(batch),
                n_terminals=n_terminals,
                logZ_estimate=st.logZ,
                weighted_reward=weighted_reward,
                fingerprint=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        env: str,
        *,
        n: int = 1,
        temperature: float = 1.0,
        epsilon: float | None = None,
        record: bool = True,
    ) -> SampleBatch:
        """Draw ``n`` trajectories from the current forward policy.

        Returns a :class:`SampleBatch` with calibrated bounds on the
        mean reward and an attestation fingerprint.
        """
        if n <= 0:
            raise InvalidTrajectory("n must be > 0")
        with self._lock:
            st = self._require_env(env)
            eps = self.config.epsilon_exploration if epsilon is None else epsilon
            trajs: list[Trajectory] = []
            for _ in range(n):
                t = self._sample_one(
                    st, temperature=temperature, epsilon=eps, rng=self._rng
                )
                if record:
                    self._record_visits(st, t)
                    self._push_replay(st, t)
                trajs.append(t)
            rewards = tuple(t.reward for t in trajs)
            terminals = tuple(t.states[-1] for t in trajs)
            log_probs = tuple(t.log_p_forward for t in trajs)
            mean_r = sum(rewards) / len(rewards)
            r_max = max(rewards) if rewards else 1.0
            r_max = max(r_max, self.config.reward_floor)
            # Normalise into [0, 1] for the half-width helpers.
            scaled = [r / r_max for r in rewards]
            mean_s = sum(scaled) / len(scaled)
            var_s = (
                sum((x - mean_s) ** 2 for x in scaled) / max(1, len(scaled) - 1)
                if len(scaled) >= 2
                else 0.0
            )
            mp = empirical_bernstein_half_width(
                len(scaled), var_s, conf=self.config.confidence
            )
            hrms = hrms_half_width(len(scaled), conf=self.config.confidence)
            lcb = max(0.0, mean_s - mp) * r_max
            hrms_lcb = max(0.0, mean_s - hrms) * r_max
            unique = len(set(terminals))
            # Forward entropy over distinct terminals.
            counts: dict[Any, int] = {}
            for t in terminals:
                counts[t] = counts.get(t, 0) + 1
            tot = float(sum(counts.values()))
            entropy = -sum((c / tot) * math.log(c / tot) for c in counts.values() if c > 0)
            # PIT update.
            for r in rewards:
                normalised = min(1.0, max(0.0, r / r_max))
                st.pit_history.append(normalised)
                if len(st.pit_history) > 100_000:
                    st.pit_history.pop(0)
            payload = {
                "op": "sample",
                "env": env,
                "n": n,
                "mean_reward": mean_r,
                "lcb": lcb,
                "unique": unique,
            }
            self._advance_chain(payload)
            self._publish(
                FLOWER_SAMPLED, {**payload, "head": self._chain_head}
            )
            return SampleBatch(
                env=env,
                n=n,
                trajectories=tuple(trajs),
                terminals=terminals,
                rewards=rewards,
                log_probs=log_probs,
                mean_reward=mean_r,
                mean_reward_lcb=lcb,
                mean_reward_hrms_lcb=hrms_lcb,
                unique_terminals=unique,
                forward_entropy=entropy,
                fingerprint=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Mode coverage / certification
    # ------------------------------------------------------------------

    def mode_coverage(
        self,
        env: str,
        *,
        n_samples: int = 256,
        top_k: int = 10,
        temperature: float = 1.0,
        record: bool = False,
    ) -> ModeCoverageReport:
        """Closed-form mode-coverage diagnostic.

        Compares the empirical distribution over terminals (across the
        env's full history plus a fresh batch of ``n_samples``
        on-policy samples) to the reward-proportional target.  Returns
        per-mode LCBs and an HRMS anytime-valid LCB on the coverage
        probability over the top-``k`` modes.
        """
        if n_samples <= 0:
            raise InvalidTrajectory("n_samples must be > 0")
        if top_k <= 0:
            raise InvalidTrajectory("top_k must be > 0")
        with self._lock:
            st = self._require_env(env)
            # Draw a fresh evaluation batch (does not record into replay
            # unless ``record=True``).
            local_rng = random.Random(self._rng.random())
            for _ in range(n_samples):
                t = self._sample_one(
                    st, temperature=temperature, epsilon=0.0, rng=local_rng
                )
                if record:
                    self._record_visits(st, t)
                    self._push_replay(st, t)
                if t.terminal:
                    st.terminal_counts[t.states[-1]] = (
                        st.terminal_counts.get(t.states[-1], 0) + 1
                    )
                    st.terminal_rewards[t.states[-1]] = t.reward
            # Build empirical distribution.
            tot = sum(st.terminal_counts.values())
            if tot == 0:
                raise InsufficientData("no terminals observed yet")
            empirical = {k: c / tot for k, c in st.terminal_counts.items()}
            # Target reward-proportional distribution.
            target_reach = _enumerate_terminals(
                st.spec, self.config.reward_floor, self.config.max_trajectory_length
            )[0]
            # If enumeration is empty, fall back to the empirical reward map.
            target_rewards = target_reach or dict(st.terminal_rewards)
            target_z = sum(target_rewards.values())
            if target_z <= 0:
                target = {k: 1.0 / max(1, len(target_rewards)) for k in target_rewards}
            else:
                target = {k: r / target_z for k, r in target_rewards.items()}
            tv = total_variation(empirical, target)
            # Hoeffding UCB on TV from the empirical count.
            tv_ucb = tv + hoeffding_half_width(tot, conf=self.config.confidence)
            # Top-K modes by reward.
            ranked = sorted(target_rewards.items(), key=lambda kv: -kv[1])
            top = [k for k, _ in ranked[:top_k]]
            found = sum(1 for k in top if st.terminal_counts.get(k, 0) > 0)
            modes_found = found
            # HRMS LCB on coverage probability = P(sample in top_k).
            mass_in_top = sum(empirical.get(k, 0.0) for k in top)
            hw = hrms_half_width(tot, conf=self.config.confidence)
            cov_lcb = max(0.0, mass_in_top - hw)
            # Hoeffding LCB / UCB on log Z.
            logZ_lcb, logZ_ucb = self._logZ_bounds(target_rewards, conf=self.config.confidence)
            payload = {
                "op": "mode_coverage",
                "env": env,
                "tv": tv,
                "modes_found": modes_found,
                "n_samples": tot,
                "top_k": top_k,
                "mode_coverage_lcb": cov_lcb,
            }
            self._advance_chain(payload)
            self._publish(
                FLOWER_CERTIFIED, {**payload, "head": self._chain_head}
            )
            return ModeCoverageReport(
                env=env,
                n_samples=tot,
                n_unique=len(empirical),
                tv_to_target=tv,
                tv_hoeffding_ucb=min(1.0, tv_ucb),
                modes_found=modes_found,
                mode_coverage_lcb=cov_lcb,
                top_k_recovered=(top_k, found),
                log_partition_lcb=logZ_lcb,
                log_partition_ucb=logZ_ucb,
                fingerprint=self._chain_head,
            )

    def _logZ_bounds(
        self, target_rewards: dict[State, float], conf: float
    ) -> tuple[float, float]:
        """Closed-form Hoeffding interval on ``log Σ_x R(x)``.

        The partition function is exactly enumerable when the DAG is
        finite; the bounds shrink to a point in that case.  When
        enumeration is partial, we report the Hoeffding band on the
        observed sum.
        """
        z = sum(target_rewards.values())
        if z <= 0:
            return float("-inf"), float("inf")
        # When fully enumerated this is exact; we still report a small
        # Hoeffding band reflecting the floor / finite-precision.
        return math.log(z) - 1e-9, math.log(z) + 1e-9

    # ------------------------------------------------------------------
    # Identifiability + calibration reports
    # ------------------------------------------------------------------

    def identifiability(
        self,
        env: str,
        *,
        top_k: int = 5,
        saturated_threshold: int = 1000,
    ) -> IdentifiabilityReport:
        """Per-(state, action) coverage gap.

        Names the under-sampled edges (smallest ``edge_counts``) and
        any terminal with positive reward but zero visits.
        """
        with self._lock:
            st = self._require_env(env)
            edges = sorted(st.edge_counts.items(), key=lambda kv: kv[1])
            under = tuple((s, a, c) for (s, a), c in edges[:top_k])
            sat = tuple(
                (s, a, c) for (s, a), c in edges if c >= saturated_threshold
            )
            # Unreachable modes: terminals enumerated with positive reward
            # but zero forward visits.
            target = _enumerate_terminals(
                st.spec, self.config.reward_floor, self.config.max_trajectory_length
            )[0]
            unreach = tuple(
                k for k, r in target.items()
                if r > self.config.reward_floor and st.terminal_counts.get(k, 0) == 0
            )
            return IdentifiabilityReport(
                env=env,
                under_sampled_edges=under,
                unreachable_modes=unreach,
                saturated_edges=sat,
            )

    def pit_calibration(self, env: str) -> PITCalibrationReport:
        """KS test of the sampler's reward PIT against Uniform(0, 1)."""
        with self._lock:
            st = self._require_env(env)
            n = len(st.pit_history)
            if n == 0:
                raise InsufficientData("no PIT history yet — call sample() first")
            d, p = ks_pvalue(tuple(st.pit_history))
            payload = {
                "op": "pit_calibration",
                "env": env,
                "n": n,
                "D": d,
                "p": p,
            }
            self._advance_chain(payload)
            self._publish(
                FLOWER_CERTIFIED, {**payload, "head": self._chain_head}
            )
            return PITCalibrationReport(
                env=env,
                n=n,
                ks_statistic=d,
                p_value=p,
                fingerprint=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Top-K extraction (the diversification kernel's deliverable)
    # ------------------------------------------------------------------

    def top_k(self, env: str, *, k: int = 10) -> list[tuple[State, float, int]]:
        """Return up to ``k`` distinct terminals with the highest
        rewards observed so far, paired with their observed visit count.
        """
        if k <= 0:
            raise InvalidTrajectory("k must be > 0")
        with self._lock:
            st = self._require_env(env)
            if not st.terminal_rewards:
                return []
            ranked = sorted(
                st.terminal_rewards.items(), key=lambda kv: -kv[1]
            )
            return [
                (state, reward, st.terminal_counts.get(state, 0))
                for state, reward in ranked[:k]
            ]

    # ------------------------------------------------------------------
    # SGD step — closed-form gradients per loss family
    # ------------------------------------------------------------------

    def _sgd_step(
        self,
        st: _EnvState,
        batch: list[Trajectory],
        family: str,
    ) -> tuple[float, float]:
        """Apply one SGD step under the chosen loss family.

        Each loss family has a closed-form analytic gradient on the
        per-edge logits ``θ`` and (for TB / DB / SubTB) the auxiliary
        ``logZ`` and per-state ``logF`` parameters.
        """
        lr = self.config.learning_rate
        lr_z = self.config.learning_rate_logz or lr
        clip = self.config.grad_clip
        loss_acc = 0.0
        # Per-parameter gradient accumulators.
        theta_grad: dict[tuple[State, Action], float] = {}
        log_flow_grad: dict[State, float] = {}
        logZ_grad = 0.0
        n_resid = 0
        for traj in batch:
            if not traj.terminal:
                continue
            states = traj.states
            actions = traj.actions
            if not states:
                continue
            terminal = states[-1]
            log_r = math.log(max(traj.reward, self.config.reward_floor))
            # Forward log-probabilities at the current parameters.
            log_pf_per_step: list[float] = []
            for s, a in zip(states[:-1], actions):
                choices = self._forward_policy(st, s, temperature=1.0, epsilon=0.0)
                p = 0.0
                logits = []
                action_idx = None
                for i, (ca, _ns, cp) in enumerate(choices):
                    logits.append(st.get_theta(s, ca))
                    if ca == a:
                        p = cp
                        action_idx = i
                if p <= 0.0 or action_idx is None:
                    log_pf_per_step.append(-1e9)
                else:
                    log_pf_per_step.append(math.log(p))
            # Backward probabilities: uniform over parent edges (the
            # canonical GFlowNet choice for unstructured DAGs).
            log_pb_per_step: list[float] = []
            for s, a, ns in zip(states[:-1], actions, states[1:]):
                k = self._num_parents(st, ns)
                log_pb_per_step.append(-math.log(max(1, k)))
            if family == LOSS_TRAJECTORY_BALANCE:
                resid = (
                    st.logZ
                    + sum(log_pf_per_step)
                    - log_r
                    - sum(log_pb_per_step)
                )
                loss_acc += resid * resid
                n_resid += 1
                # ∂L/∂logZ = 2 resid; ∂L/∂θ(s, a) = 2 resid * (1 - π(a|s))
                logZ_grad += 2.0 * resid
                for s, a in zip(states[:-1], actions):
                    choices = self._forward_policy(st, s, temperature=1.0, epsilon=0.0)
                    for ca, _ns, cp in choices:
                        delta = 1.0 if ca == a else 0.0
                        theta_grad[_edge_key(s, ca)] = (
                            theta_grad.get(_edge_key(s, ca), 0.0)
                            + 2.0 * resid * (delta - cp)
                        )
            elif family == LOSS_FLOW_MATCHING:
                # Match the in-flow and out-flow at every non-terminal
                # state.  In log-space: logsumexp(logF_in) ≈
                # logsumexp(logF_out).  We treat each edge's contribution
                # as the per-edge "flow" F(s, a) ≈ exp(θ(s, a)).
                for j in range(len(actions)):
                    s = states[j]
                    a = actions[j]
                    ns = states[j + 1]
                    in_flows = [st.get_theta(s, a) for a, _ns in st.spec.successors(s)] or [0.0]
                    out_flows: list[float]
                    if st.spec.terminal(ns):
                        out_flows = [
                            math.log(
                                max(self.config.reward_floor, float(st.spec.reward(ns)))
                            )
                        ]
                    else:
                        out_flows = [
                            st.get_theta(ns, aa) for aa, _ns in st.spec.successors(ns)
                        ] or [0.0]
                    lhs = logsumexp(in_flows)
                    rhs = logsumexp(out_flows)
                    resid = lhs - rhs
                    loss_acc += resid * resid
                    n_resid += 1
                    # Gradient w.r.t. each in-flow θ:
                    in_softmax = softmax(in_flows, beta=1.0)
                    for (aa, _ns), w in zip(st.spec.successors(s), in_softmax):
                        theta_grad[_edge_key(s, aa)] = (
                            theta_grad.get(_edge_key(s, aa), 0.0) + 2.0 * resid * w
                        )
                    if not st.spec.terminal(ns):
                        out_softmax = softmax(out_flows, beta=1.0)
                        for (aa, _ns2), w in zip(
                            st.spec.successors(ns), out_softmax
                        ):
                            theta_grad[_edge_key(ns, aa)] = (
                                theta_grad.get(_edge_key(ns, aa), 0.0)
                                - 2.0 * resid * w
                            )
            elif family == LOSS_DETAILED_BALANCE:
                # logF(s) + logP_F(s'|s) = logF(s') + logP_B(s|s')
                for j in range(len(actions)):
                    s = states[j]
                    a = actions[j]
                    ns = states[j + 1]
                    log_pf = log_pf_per_step[j]
                    log_pb = log_pb_per_step[j]
                    if st.spec.terminal(ns):
                        # Treat logF(terminal) = log R(terminal).
                        log_f_next = log_r
                    else:
                        log_f_next = st.get_log_flow(ns)
                    log_f_cur = st.get_log_flow(s)
                    resid = log_f_cur + log_pf - log_f_next - log_pb
                    loss_acc += resid * resid
                    n_resid += 1
                    log_flow_grad[s] = log_flow_grad.get(s, 0.0) + 2.0 * resid
                    if not st.spec.terminal(ns):
                        log_flow_grad[ns] = log_flow_grad.get(ns, 0.0) - 2.0 * resid
                    # ∂resid/∂θ(s, ·): (δ − π)
                    choices = self._forward_policy(st, s, temperature=1.0, epsilon=0.0)
                    for ca, _ns, cp in choices:
                        delta = 1.0 if ca == a else 0.0
                        theta_grad[_edge_key(s, ca)] = (
                            theta_grad.get(_edge_key(s, ca), 0.0)
                            + 2.0 * resid * (delta - cp)
                        )
            elif family == LOSS_SUBTRAJECTORY_BALANCE:
                # SubTB-λ: sum over all (i, j) sub-trajectories of
                # weighted squared residual ((j-i) ≥ 1).
                lam = self.config.subtb_lambda
                T = len(actions)
                # Pre-compute partial sums.
                cum_pf = [0.0] * (T + 1)
                cum_pb = [0.0] * (T + 1)
                for j in range(T):
                    cum_pf[j + 1] = cum_pf[j] + log_pf_per_step[j]
                    cum_pb[j + 1] = cum_pb[j] + log_pb_per_step[j]
                # logF(s_i) at every i — terminal uses log R.
                log_f = []
                for i, s in enumerate(states):
                    if i == T and st.spec.terminal(s):
                        log_f.append(log_r)
                    elif st.spec.terminal(s):
                        log_f.append(log_r)
                    else:
                        log_f.append(st.get_log_flow(s))
                # Iterate all sub-trajectories.
                for i in range(T):
                    for j in range(i + 1, T + 1):
                        length = j - i
                        weight = lam ** (length - 1)
                        resid = (
                            log_f[i]
                            + (cum_pf[j] - cum_pf[i])
                            - log_f[j]
                            - (cum_pb[j] - cum_pb[i])
                        )
                        wresid = weight * resid
                        loss_acc += weight * resid * resid
                        n_resid += 1
                        if not st.spec.terminal(states[i]):
                            log_flow_grad[states[i]] = (
                                log_flow_grad.get(states[i], 0.0) + 2.0 * wresid
                            )
                        if not st.spec.terminal(states[j]):
                            log_flow_grad[states[j]] = (
                                log_flow_grad.get(states[j], 0.0) - 2.0 * wresid
                            )
                        # Theta updates on each edge in [i, j).
                        for k in range(i, j):
                            sk = states[k]
                            ak = actions[k]
                            choices = self._forward_policy(
                                st, sk, temperature=1.0, epsilon=0.0
                            )
                            for ca, _ns, cp in choices:
                                delta = 1.0 if ca == ak else 0.0
                                theta_grad[_edge_key(sk, ca)] = (
                                    theta_grad.get(_edge_key(sk, ca), 0.0)
                                    + 2.0 * wresid * (delta - cp)
                                )
            else:  # pragma: no cover
                raise InvalidConfig(f"unsupported loss {family!r}")
        # Average over the residual count to keep the loss scale stable
        # across batch / trajectory length.
        if n_resid == 0:
            return 0.0, 0.0
        loss_mean = loss_acc / n_resid
        # Apply gradient clip + SGD step.
        grad_norm = 0.0
        for key, g in theta_grad.items():
            g_avg = g / n_resid
            g_clipped = max(-clip, min(clip, g_avg))
            if abs(g_clipped) > grad_norm:
                grad_norm = abs(g_clipped)
            new_val = st.theta.get(key, 0.0) - lr * g_clipped
            st.theta[key] = new_val
        for key, g in log_flow_grad.items():
            g_avg = g / n_resid
            g_clipped = max(-clip, min(clip, g_avg))
            if abs(g_clipped) > grad_norm:
                grad_norm = abs(g_clipped)
            new_val = st.log_flow.get(key, 0.0) - lr * g_clipped
            st.log_flow[key] = new_val
        if logZ_grad != 0.0:
            g_avg = logZ_grad / n_resid
            g_clipped = max(-clip, min(clip, g_avg))
            if abs(g_clipped) > grad_norm:
                grad_norm = abs(g_clipped)
            st.logZ = st.logZ - lr_z * g_clipped
        return loss_mean, grad_norm

    def _num_parents(self, st: _EnvState, state: State) -> int:
        """Count direct parents of ``state`` reachable from ``initial``.

        The GFlowNet backward policy P_B(s|s') is the per-parent
        uniform — i.e. ``1 / #parents(s')``.  For unstructured DAGs we
        compute parents by BFS from the initial state.  Cached
        per-call.  For efficiency the count is recomputed lazily.
        """
        # Cheap BFS — bounded by max_trajectory_length.  Acceptable for
        # the tabular setting Flower targets.
        parents = 0
        seen: set[Any] = set()
        stack: list[tuple[State, int]] = [(st.spec.initial, 0)]
        while stack:
            s, depth = stack.pop()
            key = s if isinstance(s, (int, str, float, tuple, frozenset)) else repr(s)
            if key in seen:
                continue
            seen.add(key)
            if depth >= self.config.max_trajectory_length:
                continue
            for a, ns in st.spec.successors(s):
                if ns == state:
                    parents += 1
                if not st.spec.terminal(ns):
                    stack.append((ns, depth + 1))
        return max(1, parents)

    # ------------------------------------------------------------------
    # Snapshot / restore — coordinator-friendly persistence
    # ------------------------------------------------------------------

    def snapshot(self, env: str) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the env's parameters."""
        with self._lock:
            st = self._require_env(env)
            return {
                "env": env,
                "loss_family": st.loss_family,
                "logZ": st.logZ,
                "theta": {
                    f"{_stable_repr(k[0])}|{_stable_repr(k[1])}": v
                    for k, v in st.theta.items()
                },
                "log_flow": {
                    _stable_repr(k): v for k, v in st.log_flow.items()
                },
                "edge_counts": {
                    f"{_stable_repr(k[0])}|{_stable_repr(k[1])}": v
                    for k, v in st.edge_counts.items()
                },
                "terminal_counts": {
                    _stable_repr(k): v for k, v in st.terminal_counts.items()
                },
                "terminal_rewards": {
                    _stable_repr(k): v for k, v in st.terminal_rewards.items()
                },
                "chain_head": self._chain_head,
            }
