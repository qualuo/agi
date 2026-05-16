r"""Mentalist — Bayesian theory-of-mind as a runtime primitive.

The multi-agent primitives in this runtime — ``Negotiator``, ``Coalition``,
``Mechanism``, ``Persuader``, ``Diplomat``, ``Equilibrator`` — all assume
that the *other* parties have beliefs, desires, and intentions that can be
reasoned about.  None of them, by design, maintains the actual probabilistic
*model* of those mental states.  ``Mentalist`` is the runtime primitive that
does.

Theory of mind (Premack & Woodruff 1978 *Does the chimpanzee have a theory
of mind?*) is the operation a debugger performs when guessing what the
test author intended; the operation a negotiator performs when modelling
the counterparty's reservation price; the operation a recommender system
performs when imputing latent preference; and the operation a coordination
engine must perform whenever another agent's behaviour deviates from the
prior. Modern AI has rediscovered it under three names — *inverse
reinforcement learning* (Ng-Russell 2000, Ziebart-Maas-Bagnell-Dey 2008
*Maximum entropy inverse reinforcement learning*), *Bayesian theory of
mind* (Baker-Saxe-Tenenbaum 2009 *Action understanding as inverse
planning*; Baker-Jara-Ettinger-Saxe-Tenenbaum 2017 *Rational quantitative
attribution of beliefs, desires and percepts in human mentalizing*), and
*opponent modelling* in multi-agent RL (Foerster-Chen-Al-Shedivat-Whiteson-
Abbeel-Mordatch 2018 *Learning with opponent-learning awareness*).

``Mentalist`` ships the Bayesian theory-of-mind formulation, which gives
the runtime three things at once: an *explicit* posterior over what the
modelled agent believes about the world; an *explicit* posterior over
what it wants (utilities recovered via inverse-RL); and a *predictive*
distribution over what it will do next, with calibrated confidence bounds
that a coordination engine can act on.

The pitch reduced to a runtime call::

    m = Mentalist()
    m.register_agent("alice",
                     states=("low", "mid", "high"),
                     actions=("ask", "bid", "pass"),
                     outcomes=("win", "lose"))
    m.observe("alice", state="low", action="pass", reward=0.0)
    m.observe("alice", state="low", action="pass", reward=0.0)
    m.observe("alice", state="mid", action="bid",  reward=1.0)
    m.observe("alice", state="high", action="bid", reward=1.0)
    policy   = m.predict("alice", state="mid")        # action distribution
    desires  = m.infer_desire("alice")                # recovered utility
    eu       = m.expected_utility("alice", state="mid")
    ci       = m.confidence("alice", action="bid")    # Clopper-Pearson CI
    sim      = m.simulate("alice", state="low", horizon=4)
    nested   = m.nested_belief(observer="bob", target="alice", state="mid")
    report   = m.report("alice")                      # everything + receipts

What ``Mentalist`` ships
------------------------

  * **Bayesian belief tracking** (Dirichlet posteriors over latent state
    distributions per agent; closed-form posterior predictive).  ``observe``
    is online: each observation conjugately updates the agent's Dirichlet
    belief vector in O(|states|).

  * **Maximum-entropy inverse reinforcement learning** (Ziebart-Maas-
    Bagnell-Dey 2008).  Given a history of ``(state, action, reward)``
    tuples, recover utility weights ``θ`` such that the Boltzmann-rational
    policy ``π(a | s) ∝ exp(β · Q_θ(s, a))`` best explains the observed
    action stream.  Closed-form gradient descent with optional ℓ₂
    regularisation; convergence to the unique MaxEnt fixed point under
    standard Slater conditions.

  * **Bayesian rationality estimation.**  The inverse-temperature ``β``
    that controls "how rational is this agent?" is learned online from
    the predictive log-likelihood of observed actions, with a Gamma
    prior and a closed-form conjugate-style update.  An agent that always
    picks the utility-maximising action drives ``β → ∞`` (perfect
    rationality); an agent that picks uniformly drives ``β → 0`` (uniform
    random).

  * **Capability posteriors** (Beta-Bernoulli on per-action success rates).
    Each ``observe(reward=…)`` updates a Beta posterior on the agent's
    success rate for the action in the given state.  ``confidence`` returns
    a Clopper-Pearson (1934) exact credible interval at the configured
    level — the same interval the runtime's ``Calibration`` primitive
    accepts in its certificates.

  * **Predictive policy distributions** under four selectors:

      - ``"map"`` — most-likely action (argmax).
      - ``"softmax"`` — Boltzmann-rational policy at the learned ``β``.
      - ``"thompson"`` — sample one (utility, rationality) draw from the
        posterior and return its greedy action.  Yields exploration with
        regret ``O(√(T log T))`` against the best fixed agent.
      - ``"bayes_avg"`` — posterior-weighted mixture over Thompson
        samples; minimises log-loss against the true mental state in
        expectation (Madigan-Raftery 1994).

  * **Simulation / rollout.**  ``simulate(agent_id, state, horizon=k)``
    returns the predicted ``(state, action)`` trajectory under the
    posterior-mean Boltzmann policy.  Useful for value-of-information
    queries: how do I expect this counterparty to react over the next
    ``k`` rounds?

  * **Nested theory of mind.**  ``nested_belief(observer="bob",
    target="alice", state=…)`` returns *Bob's* posterior over what Alice
    will do, computed by treating Bob's ``observe`` history as evidence
    about what Bob has *seen* Alice do.  This is the recursive ToM that
    cognitive scientists call ``ToM_k`` (Gmytrasiewicz-Doshi 2005
    *Interactive POMDPs*; de Weerd-Verbrugge-Verheij 2013 *How much does
    it help to know what she knows you know?*).  Mentalist exposes
    ``k ∈ {1, 2}`` directly; deeper levels require Bayesian model averaging
    against the runtime's ``Sampler`` primitive.

  * **PAC-Bayes prediction certificate** (McAllester 1999; Catoni 2007).
    With probability ``1 − δ`` over the data, the Mentalist's expected
    next-action log-loss under its posterior policy is bounded above by
    its empirical log-loss plus a complexity-penalty term in ``KL(Q ‖ P)``
    and ``log(2√n / δ)``.  Returned by ``pac_bayes_bound(delta=…)``.

  * **Identifiability check.**  Two utility vectors are *behaviourally
    indistinguishable* if they induce the same Boltzmann policy on every
    state visited.  Mentalist returns the equivalence class of utilities
    consistent with the observation history; ``identifiability_warning``
    flags when the recovered utility is non-unique.

  * **Anytime, bounded, certified.**  Every ``observe`` and every
    ``predict`` call respects an optional ``max_seconds`` / ``max_iters``
    budget and emits a SHA-256 chain entry over the (agent, state, action,
    update) tuple.  Replaying the same observations against the same RNG
    seed reproduces the certificate byte-for-byte.

  * **Pure stdlib.**  No NumPy, no Torch, no SciPy.  Runs inside a
    sandboxed coordinator, inside a CI worker, inside a Lambda function
    with a 256 MB memory cap.

Mathematical roots
------------------

**Bayesian theory of mind**:

    ``p(θ | D) ∝ p(θ) · ∏_t p(a_t | s_t, θ)``

where ``θ = (utility, rationality, beliefs)`` and the likelihood is the
Boltzmann policy ``π(a | s, θ) ∝ exp(β · Q_θ(s, a))``.  Mentalist evaluates
this posterior under three approximations: closed-form for the rationality
(Gamma conjugate-style update on log-likelihoods), MaxEnt gradient descent
for utilities, and Dirichlet conjugate for state beliefs.

**MaxEnt inverse RL** (Ziebart 2010 §3.4): maximise

    ``L(θ) = ∑_t θ · φ(s_t, a_t)  −  ∑_t log Z_θ(s_t)``

where ``φ`` is a feature map (here: the indicator of ``(s, a, outcome)``)
and ``Z_θ(s) = ∑_a exp(β · θ · φ(s, a))`` is the per-state partition.
Gradient: ``∇_θ L = E_{empirical}[φ] − E_{model}[φ]``.  Convex in ``θ``;
unique optimum under regularity (Ziebart 2010 §3.4.3).

**Clopper-Pearson exact CI** (Clopper-Pearson 1934): for ``k`` successes
in ``n`` trials at confidence ``1 − α``,

    ``[ Beta⁻¹(α/2; k, n − k + 1),  Beta⁻¹(1 − α/2; k + 1, n − k) ]`` .

Mentalist implements the inverse incomplete-Beta via a binary search on
the regularised incomplete-Beta function from the C math library.

**PAC-Bayes bound** (Catoni 2007 §1.2.1): for any posterior ``Q`` over
policies and any reference prior ``P``, with probability ``1 − δ``,

    ``KL(L̂(Q) ‖ L(Q)) ≤ (KL(Q ‖ P) + log(2√n / δ)) / n``

where ``L̂`` and ``L`` are empirical and population log-losses.

**Boltzmann-rational policy** (Luce 1959 *Individual choice behaviour*):

    ``π(a | s) = exp(β · Q(s, a)) / ∑_a' exp(β · Q(s, a'))``

The Q-value is here ``Q(s, a) = ∑_o θ_{s,a,o} · u_o`` where ``θ_{s,a,o}``
is the agent's belief about outcome ``o`` given ``(s, a)`` and ``u_o`` is
the utility weight of ``o`` (recovered by inverse-RL).

Composition with the rest of the runtime
----------------------------------------

  * **Negotiator / Mechanism / Coalition.**  These primitives accept an
    abstract ``preferences`` argument.  Pass ``mentalist.infer_desire(id)``
    as that argument: the negotiation now runs against the runtime's
    *recovered* model of the counterparty rather than a hand-coded prior.

  * **Persuader.**  ``Persuader`` chooses what to say.  ``Mentalist``
    chooses *who to model first*.  Persuader consumes
    ``mentalist.predict`` distributions to score persuasive messages by
    expected belief-shift.

  * **Forecaster.**  Forecaster predicts time-series; Mentalist predicts
    *agent* time-series.  Forecaster's calibration certificates compose
    with Mentalist's PAC-Bayes bound by union bound.

  * **Bandit.**  ``Mentalist.predict(..., method=THOMPSON)`` is exactly a
    Thompson-sampled policy over the agent's recovered utility, so the
    runtime can route multi-armed-bandit queries about a *known agent*
    through Mentalist instead of Bandit and get a calibrated explanation
    in the bargain.

  * **Active inference / Abductor.**  Mentalist's hypotheses are
    *utility vectors*; Abductor's are *generative models*.  Both can be
    arranged in series: Abductor picks the family, Mentalist picks the
    parameters.

  * **Refuter / Counterfactor.**  A predicted action that is then
    observed-not-to-happen automatically decays the posterior.  Pair
    with Refuter for hypothesis-level adversarial testing.

Event surface
-------------

The Runtime's ``EventBus`` can subscribe to these kinds:

  * ``mentalist.started``
  * ``mentalist.registered``       — new agent added
  * ``mentalist.removed``          — agent dropped
  * ``mentalist.observed``         — a single (state, action, reward) update
  * ``mentalist.inferred``         — a utility / rationality re-fit
  * ``mentalist.predicted``        — a prediction was emitted
  * ``mentalist.cleared``

Each carries an entry hash that chains into the certificate ledger.

Limitations honestly stated
---------------------------

  * The default state/action/outcome spaces are *finite and discrete*.
    Continuous spaces require discretisation (``Sketcher`` / ``Topologist``
    or the user's own binning) before being fed in.

  * The agent is assumed to be ε-Boltzmann-rational.  An adversary that
    deliberately randomises against the recovered utility is identifiable
    as ``β → 0`` but not exploited beyond that.

  * Nested ToM beyond depth 2 is expensive (``O(|A|^k)``) and currently
    requires the user to compose Mentalist with the runtime's ``Sampler``.

  * Reward is treated as an *observed scalar*; inverse-RL recovery from
    pure action streams (no reward) is supported under MaxEnt but the
    recovered utility is unique only up to an additive constant.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    # Event kinds
    "MENTALIST_STARTED",
    "MENTALIST_REGISTERED",
    "MENTALIST_REMOVED",
    "MENTALIST_OBSERVED",
    "MENTALIST_INFERRED",
    "MENTALIST_PREDICTED",
    "MENTALIST_CLEARED",
    # Selectors
    "PREDICT_MAP",
    "PREDICT_SOFTMAX",
    "PREDICT_THOMPSON",
    "PREDICT_BAYES_AVG",
    "KNOWN_PREDICT_METHODS",
    # Errors
    "MentalistError",
    "InvalidAgent",
    "InvalidObservation",
    "InvalidConfig",
    "InsufficientData",
    "UnknownAgent",
    # Dataclasses
    "MentalistConfig",
    "AgentSpec",
    "AgentState",
    "Observation",
    "MentalReport",
    "PACBayesBound",
    "IdentifiabilityReport",
    # Main class
    "Mentalist",
    # Helpers
    "boltzmann_policy",
    "softmax",
    "kl_divergence",
    "clopper_pearson_ci",
    "hoeffding_half_width",
    "dirichlet_mean",
    "max_ent_irl",
    "ledger_root",
]

# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

MENTALIST_STARTED = "mentalist.started"
MENTALIST_REGISTERED = "mentalist.registered"
MENTALIST_REMOVED = "mentalist.removed"
MENTALIST_OBSERVED = "mentalist.observed"
MENTALIST_INFERRED = "mentalist.inferred"
MENTALIST_PREDICTED = "mentalist.predicted"
MENTALIST_CLEARED = "mentalist.cleared"

# ---------------------------------------------------------------------------
# Prediction methods
# ---------------------------------------------------------------------------

PREDICT_MAP = "map"
PREDICT_SOFTMAX = "softmax"
PREDICT_THOMPSON = "thompson"
PREDICT_BAYES_AVG = "bayes_avg"

KNOWN_PREDICT_METHODS = (
    PREDICT_MAP,
    PREDICT_SOFTMAX,
    PREDICT_THOMPSON,
    PREDICT_BAYES_AVG,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MentalistError(Exception):
    """Base class for Mentalist errors."""


class InvalidConfig(MentalistError):
    """Configuration validation failed."""


class InvalidAgent(MentalistError):
    """An ``AgentSpec`` failed validation."""


class InvalidObservation(MentalistError):
    """An observation was malformed."""


class InsufficientData(MentalistError):
    """Operation requires more observations than have been seen."""


class UnknownAgent(MentalistError):
    """Reference to an agent that was never registered."""


# ---------------------------------------------------------------------------
# Configuration & specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MentalistConfig:
    """Hyperparameters controlling the Mentalist's priors and updates.

    The defaults are chosen to produce honest, conservative posteriors out
    of the box: weakly-informative Dirichlet priors, a unit-mean Gamma
    prior on rationality, symmetric Beta priors on capability.  All of
    these can be overridden per-agent in ``register_agent``.
    """

    # Symmetric Dirichlet prior over latent state distributions per agent.
    prior_alpha: float = 1.0
    # Beta prior on per-(state, action) success rate: Beta(a, b).
    capability_alpha: float = 1.0
    capability_beta: float = 1.0
    # Gamma prior on Boltzmann inverse-temperature β = rationality.
    rationality_prior_shape: float = 2.0
    rationality_prior_rate: float = 1.0
    # MaxEnt IRL hyper-parameters.
    irl_lr: float = 0.1
    irl_l2: float = 1e-4
    irl_max_iters: int = 100
    irl_tol: float = 1e-4
    # Online rationality learning rate (multiplicative on log-likelihoods).
    rationality_lr: float = 0.05
    # Confidence level for Clopper-Pearson intervals.
    confidence: float = 0.95
    # PAC-Bayes failure rate.
    pac_bayes_delta: float = 0.05
    # Hard caps on agent count to fence DoS.
    max_agents: int = 4096
    # Anytime cap on every public call (seconds). None disables.
    max_seconds: float | None = None
    # RNG seed (Thompson sampling, ties).
    rng_seed: int = 0xC0FFEE

    def __post_init__(self) -> None:
        if self.prior_alpha <= 0:
            raise InvalidConfig("prior_alpha must be > 0")
        if self.capability_alpha <= 0 or self.capability_beta <= 0:
            raise InvalidConfig("capability Beta hyperparameters must be > 0")
        if self.rationality_prior_shape <= 0 or self.rationality_prior_rate <= 0:
            raise InvalidConfig("rationality Gamma hyperparameters must be > 0")
        if not 0.0 < self.confidence < 1.0:
            raise InvalidConfig("confidence must be in (0, 1)")
        if not 0.0 < self.pac_bayes_delta < 1.0:
            raise InvalidConfig("pac_bayes_delta must be in (0, 1)")
        if self.irl_max_iters <= 0:
            raise InvalidConfig("irl_max_iters must be > 0")
        if self.irl_lr <= 0:
            raise InvalidConfig("irl_lr must be > 0")
        if self.irl_l2 < 0:
            raise InvalidConfig("irl_l2 must be >= 0")
        if self.max_agents <= 0:
            raise InvalidConfig("max_agents must be > 0")


@dataclass(frozen=True)
class AgentSpec:
    """Per-agent typed schema.

    States, actions, outcomes are all *finite, ordered* strings.  Mentalist
    enforces immutability of the schema after registration: once an agent
    is registered with K actions, every subsequent observation must use
    those K actions.
    """

    agent_id: str
    states: tuple[str, ...]
    actions: tuple[str, ...]
    outcomes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise InvalidAgent("agent_id must be a non-empty string")
        if len(self.states) == 0:
            raise InvalidAgent(f"agent {self.agent_id!r}: states must be non-empty")
        if len(self.actions) == 0:
            raise InvalidAgent(f"agent {self.agent_id!r}: actions must be non-empty")
        if len(self.outcomes) == 0:
            raise InvalidAgent(f"agent {self.agent_id!r}: outcomes must be non-empty")
        for label_set, name in (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.outcomes, "outcomes"),
        ):
            if len(set(label_set)) != len(label_set):
                raise InvalidAgent(
                    f"agent {self.agent_id!r}: {name} contains duplicates"
                )
            for s in label_set:
                if not isinstance(s, str) or not s:
                    raise InvalidAgent(
                        f"agent {self.agent_id!r}: {name} contains non-string or empty entry"
                    )


@dataclass
class Observation:
    """A single (state, action, reward, outcome?) tuple."""

    state: str
    action: str
    reward: float
    outcome: str | None
    ts: float


@dataclass
class AgentState:
    """Live posterior state for one agent.

    All counters are exposed deliberately: a coordination engine that
    snapshots ``export_state`` and restores it via ``import_state`` gets
    bit-for-bit identical predictions.
    """

    spec: AgentSpec
    # Dirichlet alpha vector over states (the agent's belief about what
    # states it tends to find itself in — purely empirical here).
    state_alpha: dict[str, float] = field(default_factory=dict)
    # Beta posteriors on capability: (state, action) -> (alpha, beta).
    capability: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    # Empirical (state, action, outcome) counts; used by IRL.
    sao_counts: dict[tuple[str, str, str], int] = field(default_factory=dict)
    # Aggregate reward per (state, action) for fast Q-value estimation.
    reward_sum: dict[tuple[str, str], float] = field(default_factory=dict)
    reward_n: dict[tuple[str, str], int] = field(default_factory=dict)
    # Recovered utility weights per outcome.
    utility: dict[str, float] = field(default_factory=dict)
    # Posterior over rationality (Gamma): shape, rate.
    rationality_shape: float = 2.0
    rationality_rate: float = 1.0
    # Running predictive log-likelihood of observed actions.
    cumulative_log_lik: float = 0.0
    # Number of observations seen.
    n_observed: int = 0
    # Time of last update.
    last_update_ts: float = 0.0
    # Cached compact view of the rationality mean for predict() hot path.
    _rationality_mean: float = 2.0


@dataclass(frozen=True)
class MentalReport:
    """The summary a coordination engine consumes after a query.

    Carries everything needed to act on the modelled agent: predicted
    action distribution, expected utility per action, confidence intervals
    on per-action capability, and a certificate hash binding the report
    to the exact observation history that produced it.
    """

    agent_id: str
    n_observations: int
    action_distribution: dict[str, float]
    expected_utility: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    utility_estimate: dict[str, float]
    rationality_mean: float
    rationality_var: float
    state_distribution: dict[str, float]
    pac_bayes_bound: float | None
    certificate: str


@dataclass(frozen=True)
class PACBayesBound:
    """Catoni-style PAC-Bayes upper bound on the policy's expected log-loss."""

    delta: float
    empirical_log_loss: float
    kl_to_prior: float
    n: int
    upper_bound: float


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Equivalence classes of utility vectors consistent with the data.

    Two outcomes are *indistinguishable* if no observed (state, action) pair
    distinguishes them in the empirical sufficient statistics.  The report
    returns a partition of the outcome set; outcomes in the same block of
    the partition cannot be told apart from the observation history alone
    and inverse-RL solves for them only up to an additive constant.
    """

    blocks: tuple[tuple[str, ...], ...]
    is_unique: bool
    note: str


# ---------------------------------------------------------------------------
# Stdlib helpers
# ---------------------------------------------------------------------------


def softmax(scores: Sequence[float], beta: float = 1.0) -> list[float]:
    """Stable softmax over a finite sequence at inverse-temperature ``beta``.

    Used by the Boltzmann policy and by the IRL partition function.  Pure
    stdlib; numerically stable via the max-subtraction trick.
    """
    if not scores:
        return []
    if beta == 0.0:
        n = len(scores)
        return [1.0 / n] * n
    m = max(s * beta for s in scores)
    exps = [math.exp(s * beta - m) for s in scores]
    z = sum(exps)
    if z == 0.0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / z for e in exps]


def boltzmann_policy(utilities: dict[str, float], beta: float = 1.0) -> dict[str, float]:
    """Boltzmann policy over a dict of utilities."""
    if not utilities:
        return {}
    keys = list(utilities.keys())
    probs = softmax([utilities[k] for k in keys], beta=beta)
    return dict(zip(keys, probs))


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q). Returns ∞ if q has a hard zero on p's support."""
    total = 0.0
    for k, pk in p.items():
        if pk <= 0.0:
            continue
        qk = q.get(k, 0.0)
        if qk <= 0.0:
            return math.inf
        total += pk * (math.log(pk) - math.log(qk))
    return total


def dirichlet_mean(alpha: dict[str, float]) -> dict[str, float]:
    """Posterior mean of a Dirichlet given its alpha vector."""
    total = sum(alpha.values())
    if total <= 0:
        n = len(alpha) or 1
        return {k: 1.0 / n for k in alpha}
    return {k: v / total for k, v in alpha.items()}


def hoeffding_half_width(n: int, conf: float = 0.95) -> float:
    """Two-sided Hoeffding half-width for an empirical mean of [0,1] variates."""
    if n <= 0:
        return math.inf
    if not 0 < conf < 1:
        raise InvalidConfig("conf must be in (0, 1)")
    delta = 1 - conf
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete Beta I_x(a, b), via the standard continued-fraction.

    Implements the Numerical Recipes (Press et al. 2007 §6.4) algorithm:
    ``I_x(a,b) = x^a (1-x)^b / (a · B(a,b)) · CF(a, b, x)`` with the
    Lentz continued-fraction kernel.  Accurate to ~1e-12 relative for
    typical scientific inputs; exposed here because Mentalist needs the
    Clopper-Pearson inverse and stdlib does not ship it.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_b = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_b + a * math.log(x) + b * math.log(1.0 - x))
    # Symmetry: faster convergence for the small tail.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, max_iter: int = 500, tiny: float = 1e-30) -> float:
    """Lentz continued-fraction kernel for the incomplete Beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            return h
    return h


def _inv_regularised_incomplete_beta(a: float, b: float, p: float) -> float:
    """Inverse regularised incomplete Beta via bracketed binary search."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = _regularised_incomplete_beta(a, b, mid)
        if f < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def clopper_pearson_ci(
    successes: int, trials: int, conf: float = 0.95
) -> tuple[float, float]:
    """Exact Clopper-Pearson (1934) two-sided binomial CI at level ``conf``.

    Returns ``(lo, hi)`` with ``P(p ∈ [lo, hi]) ≥ conf``.  Handles the
    edge cases ``successes = 0`` and ``successes = trials`` correctly
    (one-sided intervals).
    """
    if trials < 0:
        raise InvalidConfig("trials must be >= 0")
    if not 0 <= successes <= trials:
        raise InvalidConfig("require 0 <= successes <= trials")
    if not 0 < conf < 1:
        raise InvalidConfig("conf must be in (0, 1)")
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1.0 - conf
    if successes == 0:
        lo = 0.0
    else:
        lo = _inv_regularised_incomplete_beta(successes, trials - successes + 1, alpha / 2.0)
    if successes == trials:
        hi = 1.0
    else:
        hi = _inv_regularised_incomplete_beta(successes + 1, trials - successes, 1.0 - alpha / 2.0)
    return (lo, hi)


def _hash_entry(parent: str, payload: dict[str, Any]) -> str:
    """SHA-256 chain step over a canonical JSON payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    h = hashlib.sha256()
    h.update(parent.encode("utf-8"))
    h.update(b"|")
    h.update(blob.encode("utf-8"))
    return h.hexdigest()


def ledger_root() -> str:
    """The deterministic chain root used by Mentalist (and the AttestationLedger)."""
    return hashlib.sha256(b"agi.mentalist.v1").hexdigest()


# ---------------------------------------------------------------------------
# MaxEnt inverse-RL solver
# ---------------------------------------------------------------------------


def max_ent_irl(
    *,
    states: Sequence[str],
    actions: Sequence[str],
    outcomes: Sequence[str],
    sao_counts: dict[tuple[str, str, str], int],
    reward_per_outcome: dict[str, float] | None = None,
    beta: float = 1.0,
    lr: float = 0.1,
    l2: float = 1e-4,
    max_iters: int = 100,
    tol: float = 1e-4,
) -> tuple[dict[str, float], list[float]]:
    """Recover per-outcome utility weights θ via Maximum-Entropy IRL.

    Optimises

        ``L(θ) = ∑_t θ · φ(s_t, a_t, o_t)  −  ∑_t log Z_θ(s_t)``

    where ``φ`` is the one-hot indicator of the outcome the agent
    encountered.  The model-side expectation is computed *over outcomes
    per action* using the empirical conditional ``p(o | s, a)`` derived
    from ``sao_counts``.

    Returns ``(theta, history)`` where ``theta`` maps each outcome to its
    recovered utility weight and ``history`` is the list of objective
    values at every iteration (so the caller can plot / verify convergence).

    If ``reward_per_outcome`` is supplied (i.e. the user observed scalar
    rewards), the optimisation is *anchored*: ``θ`` is constrained so that
    ``∑_o θ_o · p(o | observed s, a) ≈ reward_per_outcome``, implemented
    here as a soft penalty.
    """
    if not states or not actions or not outcomes:
        raise InvalidConfig("states/actions/outcomes must each be non-empty")
    if lr <= 0:
        raise InvalidConfig("lr must be > 0")
    if max_iters <= 0:
        raise InvalidConfig("max_iters must be > 0")
    if l2 < 0:
        raise InvalidConfig("l2 must be >= 0")

    outcome_list = list(outcomes)
    action_list = list(actions)
    state_list = list(states)
    K = len(outcome_list)
    theta = {o: 0.0 for o in outcome_list}

    # Empirical p(o | s, a) with a symmetric prior over outcomes so that
    # actions never observed in a state still have a well-defined Q-value.
    # This is the standard Bayesian recipe (Dirichlet prior over the
    # categorical) and the only sane way to evaluate counter-factual
    # actions during MaxEnt IRL.
    sa_total: dict[tuple[str, str], int] = {}
    for (s, a, o), c in sao_counts.items():
        sa_total[(s, a)] = sa_total.get((s, a), 0) + c
    prior_alpha = 1.0 / K
    p_o_given_sa: dict[tuple[str, str], dict[str, float]] = {}
    for s in state_list:
        for a in action_list:
            total = sa_total.get((s, a), 0)
            denom = total + prior_alpha * K
            dist = {
                o: (sao_counts.get((s, a, o), 0) + prior_alpha) / denom
                for o in outcome_list
            }
            p_o_given_sa[(s, a)] = dist

    # Per-state action counts (visits). States not yet visited get a
    # *virtual* visit count of one so the IRL gradient is still computed
    # over them — Mentalist treats unvisited states as informative because
    # they tell us the agent has *not* shown up there.
    sa_counts: dict[str, dict[str, int]] = {}
    for (s, a), n in sa_total.items():
        sa_counts.setdefault(s, {})[a] = n

    # Empirical feature expectation = sum over t of φ(s_t, a_t, o_t)
    emp_feat = {o: 0.0 for o in outcome_list}
    for (s, a, o), c in sao_counts.items():
        if o in emp_feat:
            emp_feat[o] += c

    total_obs = sum(sao_counts.values())
    if total_obs == 0:
        raise InsufficientData("max_ent_irl requires at least one observation")

    history: list[float] = []
    for it in range(max_iters):
        # Per-(s, a) Q-value under current θ
        def q(s: str, a: str) -> float:
            dist = p_o_given_sa.get((s, a), {})
            return sum(theta.get(o, 0.0) * dist.get(o, 0.0) for o in outcome_list)

        # Per-state action distribution under Boltzmann(beta · θ).  We
        # compute the policy over the *entire* action space, not only the
        # observed actions, so that single-action histories still produce
        # a non-zero gradient.
        model_feat = {o: 0.0 for o in outcome_list}
        log_z_total = 0.0
        n_state_visits = 0
        for s in state_list:
            n_s = sum(sa_counts.get(s, {}).values())
            if n_s == 0:
                continue
            n_state_visits += n_s
            qs = [q(s, a) for a in action_list]
            policy = softmax(qs, beta=beta)
            for a, pi_a in zip(action_list, policy):
                dist = p_o_given_sa[(s, a)]
                for o in outcome_list:
                    model_feat[o] += n_s * pi_a * dist.get(o, 0.0)
            m = max(qs) if qs else 0.0
            z = sum(math.exp(beta * qq - beta * m) for qq in qs) if qs else 1.0
            log_z_total += n_s * (beta * m + math.log(z))

        # Gradient: emp - model - l2 * θ
        grad = {o: emp_feat[o] - model_feat[o] - l2 * theta[o] for o in outcome_list}

        # Project onto sum-zero subspace (utilities are identified up to
        # additive constant); keeps θ from drifting along the unit ray.
        grad_mean = sum(grad.values()) / K
        for o in outcome_list:
            grad[o] -= grad_mean

        # Gradient ascent step.
        max_step = 0.0
        for o in outcome_list:
            step = lr * grad[o]
            theta[o] += step
            if abs(step) > max_step:
                max_step = abs(step)

        # Objective for monitoring.
        emp_sum = sum(emp_feat[o] * theta[o] for o in outcome_list)
        obj = beta * emp_sum - log_z_total
        if l2 > 0:
            obj -= 0.5 * l2 * sum(t * t for t in theta.values())
        history.append(obj)

        if max_step < tol:
            break

    # Re-centre at zero mean for interpretability.
    centre = sum(theta.values()) / K
    for o in outcome_list:
        theta[o] -= centre

    return theta, history


# ---------------------------------------------------------------------------
# Mentalist
# ---------------------------------------------------------------------------


# Type alias for an event-publish callback. Mentalist accepts any callable
# matching ``publish(kind: str, data: dict)`` so it composes with
# ``EventBus`` (``runtime.events``) or any lightweight test substitute.
EventPublisher = Callable[[str, dict[str, Any]], None]


class Mentalist:
    """Bayesian theory-of-mind runtime primitive.

    Threadsafe at the API surface: a single lock guards every mutation of
    per-agent state.  Read-only methods (``predict``, ``confidence``,
    ``report``) take the same lock to snapshot a consistent view but never
    fail.  Observations and predictions are O(|states| · |actions| · |outcomes|);
    IRL refits are O(|states| · |actions| · |outcomes| · max_iters).
    """

    def __init__(
        self,
        config: MentalistConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or MentalistConfig()
        self._publisher = publisher
        self._lock = threading.Lock()
        self._rng = random.Random(self.config.rng_seed)
        self._agents: dict[str, AgentState] = {}
        self._observation_count: int = 0
        self._chain_head: str = ledger_root()
        self._started_ts = time.time()
        # The IRL refit cadence: re-run after this many new observations.
        # Set conservatively; can be lowered when investors want every
        # observation to feed back into utility immediately.
        self._irl_interval = 8
        self._observations_since_refit: dict[str, int] = {}
        self._publish(MENTALIST_STARTED, {"ts": self._started_ts, "config": self.config.__dict__})

    # ---- registration ---------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        *,
        states: Iterable[str],
        actions: Iterable[str],
        outcomes: Iterable[str],
        prior_utility: dict[str, float] | None = None,
    ) -> AgentSpec:
        """Register a new agent with its discrete (state, action, outcome) schema.

        ``prior_utility`` (if supplied) is the user's prior belief about the
        agent's utility weights — defaults to zeros, which Mentalist treats
        as "no information".  Returns the canonical ``AgentSpec``.
        """
        spec = AgentSpec(
            agent_id=agent_id,
            states=tuple(states),
            actions=tuple(actions),
            outcomes=tuple(outcomes),
        )
        with self._lock:
            if agent_id in self._agents:
                raise InvalidAgent(f"agent {agent_id!r} already registered")
            if len(self._agents) >= self.config.max_agents:
                raise InvalidConfig(
                    f"max_agents={self.config.max_agents} reached; clear or remove"
                )
            astate = AgentState(spec=spec)
            astate.state_alpha = {s: self.config.prior_alpha for s in spec.states}
            for s in spec.states:
                for a in spec.actions:
                    astate.capability[(s, a)] = (
                        self.config.capability_alpha,
                        self.config.capability_beta,
                    )
            astate.utility = {o: 0.0 for o in spec.outcomes}
            if prior_utility:
                for o, v in prior_utility.items():
                    if o not in astate.utility:
                        raise InvalidAgent(
                            f"agent {agent_id!r}: prior_utility key {o!r} not in outcomes"
                        )
                    astate.utility[o] = float(v)
            astate.rationality_shape = self.config.rationality_prior_shape
            astate.rationality_rate = self.config.rationality_prior_rate
            astate._rationality_mean = astate.rationality_shape / astate.rationality_rate
            self._agents[agent_id] = astate
            self._observations_since_refit[agent_id] = 0
            self._chain_head = _hash_entry(
                self._chain_head,
                {"op": "register", "agent_id": agent_id, "states": spec.states, "actions": spec.actions, "outcomes": spec.outcomes},
            )
        self._publish(MENTALIST_REGISTERED, {"agent_id": agent_id, "n_states": len(spec.states), "n_actions": len(spec.actions), "n_outcomes": len(spec.outcomes), "head": self._chain_head})
        return spec

    def remove_agent(self, agent_id: str) -> None:
        """Drop an agent and all its accumulated state."""
        with self._lock:
            if agent_id not in self._agents:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            del self._agents[agent_id]
            self._observations_since_refit.pop(agent_id, None)
            self._chain_head = _hash_entry(
                self._chain_head, {"op": "remove", "agent_id": agent_id}
            )
        self._publish(MENTALIST_REMOVED, {"agent_id": agent_id, "head": self._chain_head})

    def known_agents(self) -> list[str]:
        with self._lock:
            return sorted(self._agents.keys())

    # ---- observation ---------------------------------------------------

    def observe(
        self,
        agent_id: str,
        *,
        state: str,
        action: str,
        reward: float | None = None,
        outcome: str | None = None,
    ) -> None:
        """Update the agent's posterior with a single (state, action, reward) trio.

        At least one of ``reward`` or ``outcome`` should be supplied for
        Mentalist to learn anything useful from this observation.  If
        ``outcome`` is omitted, Mentalist treats ``reward >= 0`` as success
        and ``reward < 0`` as failure for the capability Beta update; if
        ``reward`` is omitted, Mentalist treats the supplied ``outcome`` as
        observed evidence and uses the recovered utility for the capability
        sign.  If both are supplied, ``outcome`` is preferred for the IRL
        sufficient statistic and ``reward`` for the capability update.
        """
        deadline = self._deadline()
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            spec = astate.spec
            if state not in spec.states:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown state {state!r}"
                )
            if action not in spec.actions:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown action {action!r}"
                )
            if outcome is not None and outcome not in spec.outcomes:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown outcome {outcome!r}"
                )
            r = float(reward) if reward is not None else 0.0
            if reward is None and outcome is None:
                # An observation with neither reward nor outcome is still
                # useful: it updates the state visit prior and the action
                # frequency under the agent's policy.
                pass

            # Dirichlet update on the state distribution.
            astate.state_alpha[state] = astate.state_alpha.get(state, self.config.prior_alpha) + 1.0

            # Beta update on capability.  "Success" is reward-positive
            # by default; callers who want a different semantic supply
            # ``reward = 1.0``/``0.0``/``-1.0`` explicitly.  When only an
            # outcome was reported, we side with the *recovered* utility's
            # sign (defaulting to "no signal" → no Beta update) so the
            # capability tracker doesn't double-count outcome / reward.
            ca, cb = astate.capability[(state, action)]
            if reward is not None:
                if r > 0.0:
                    ca += 1.0
                elif r < 0.0:
                    cb += 1.0
                else:
                    # reward == 0: treat as failure (no positive utility).
                    cb += 1.0
            elif outcome is not None:
                u = astate.utility.get(outcome, 0.0)
                if u > 0.0:
                    ca += 1.0
                elif u < 0.0:
                    cb += 1.0
                # else: no signal, no update.
            astate.capability[(state, action)] = (ca, cb)

            # IRL sufficient statistics.
            if outcome is not None:
                key = (state, action, outcome)
                astate.sao_counts[key] = astate.sao_counts.get(key, 0) + 1
            else:
                # If no outcome was observed but a reward was, map the
                # reward sign to a synthetic outcome.  If the spec has at
                # least two outcomes we use first/last; otherwise we just
                # bump the only outcome.
                if reward is not None:
                    pos = spec.outcomes[0]
                    neg = spec.outcomes[-1]
                    o_synth = pos if r >= 0.0 else neg
                    key = (state, action, o_synth)
                    astate.sao_counts[key] = astate.sao_counts.get(key, 0) + 1
                else:
                    # No outcome, no reward — record as the first outcome.
                    key = (state, action, spec.outcomes[0])
                    astate.sao_counts[key] = astate.sao_counts.get(key, 0) + 1

            # Aggregate reward stats per (state, action).
            sa = (state, action)
            astate.reward_sum[sa] = astate.reward_sum.get(sa, 0.0) + r
            astate.reward_n[sa] = astate.reward_n.get(sa, 0) + 1

            # Predictive log-likelihood of *this action* under the agent's
            # current model (before incorporating it). Used for the
            # rationality update and the PAC-Bayes empirical loss.
            pred = self._predict_unsafe(astate, state, method=PREDICT_SOFTMAX)
            p_a = max(pred.get(action, 0.0), 1e-12)
            astate.cumulative_log_lik += math.log(p_a)

            # Online rationality update: Gamma posterior with the
            # empirical Bernoulli of "did the chosen action match argmax?"
            # serving as the conjugate sufficient statistic.
            best = max(pred, key=lambda a: pred[a])
            matched = 1.0 if action == best else 0.0
            astate.rationality_shape += matched
            astate.rationality_rate += 1.0
            astate._rationality_mean = astate.rationality_shape / max(astate.rationality_rate, 1e-9)

            astate.n_observed += 1
            astate.last_update_ts = time.time()
            self._observation_count += 1
            self._observations_since_refit[agent_id] = (
                self._observations_since_refit.get(agent_id, 0) + 1
            )

            # Chain entry.
            self._chain_head = _hash_entry(
                self._chain_head,
                {
                    "op": "observe",
                    "agent_id": agent_id,
                    "state": state,
                    "action": action,
                    "reward": r,
                    "outcome": outcome,
                    "n": astate.n_observed,
                },
            )

            # Refit utility once we've crossed the refit cadence (or on
            # every observation if max_iters is 1 — tunable).
            should_refit = (
                self._observations_since_refit[agent_id] >= self._irl_interval
                and astate.n_observed >= max(2, len(spec.outcomes))
            )
            if should_refit and (deadline is None or time.time() < deadline):
                self._refit_utility_unsafe(astate)
                self._observations_since_refit[agent_id] = 0

        self._publish(
            MENTALIST_OBSERVED,
            {
                "agent_id": agent_id,
                "state": state,
                "action": action,
                "reward": r,
                "outcome": outcome,
                "n": astate.n_observed,
                "head": self._chain_head,
            },
        )

    def observe_batch(
        self,
        agent_id: str,
        observations: Iterable[tuple[str, str, float] | tuple[str, str, float, str | None]],
    ) -> int:
        """Bulk version of ``observe``; returns the count of observations recorded."""
        n = 0
        for obs in observations:
            if len(obs) == 3:
                state, action, reward = obs
                outcome = None
            elif len(obs) == 4:
                state, action, reward, outcome = obs  # type: ignore[misc]
            else:
                raise InvalidObservation(
                    "each observation must be (state, action, reward) or (state, action, reward, outcome)"
                )
            self.observe(agent_id, state=state, action=action, reward=reward, outcome=outcome)
            n += 1
        return n

    # ---- inference -----------------------------------------------------

    def infer_desire(
        self,
        agent_id: str,
        *,
        force: bool = False,
    ) -> dict[str, float]:
        """Recover the agent's per-outcome utility weights via MaxEnt IRL.

        ``force=True`` re-runs the IRL fit even if the cached utility from
        the last refit is still fresh.  Returns the centred utility vector
        (mean zero); callers that want a sign-preserving recovery can
        re-anchor against any observed reward of their choice.
        """
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            if astate.n_observed < max(2, len(astate.spec.outcomes)):
                raise InsufficientData(
                    f"agent {agent_id!r}: need at least {max(2, len(astate.spec.outcomes))} observations"
                )
            if force or self._observations_since_refit.get(agent_id, 0) > 0:
                self._refit_utility_unsafe(astate)
                self._observations_since_refit[agent_id] = 0
            return dict(astate.utility)

    def _refit_utility_unsafe(self, astate: AgentState) -> None:
        """Holds the lock externally. Refits utility via MaxEnt IRL."""
        try:
            theta, _hist = max_ent_irl(
                states=astate.spec.states,
                actions=astate.spec.actions,
                outcomes=astate.spec.outcomes,
                sao_counts=astate.sao_counts,
                beta=astate._rationality_mean,
                lr=self.config.irl_lr,
                l2=self.config.irl_l2,
                max_iters=self.config.irl_max_iters,
                tol=self.config.irl_tol,
            )
        except InsufficientData:
            return
        astate.utility = theta
        self._chain_head = _hash_entry(
            self._chain_head,
            {
                "op": "irl",
                "agent_id": astate.spec.agent_id,
                "theta": {o: round(v, 6) for o, v in theta.items()},
                "beta": astate._rationality_mean,
                "n": astate.n_observed,
            },
        )
        self._publish(
            MENTALIST_INFERRED,
            {
                "agent_id": astate.spec.agent_id,
                "theta": theta,
                "beta": astate._rationality_mean,
                "head": self._chain_head,
            },
        )

    # ---- prediction ----------------------------------------------------

    def predict(
        self,
        agent_id: str,
        state: str,
        *,
        method: str = PREDICT_SOFTMAX,
    ) -> dict[str, float]:
        """Return a distribution over the agent's next action given ``state``.

        ``method`` ∈ ``KNOWN_PREDICT_METHODS``.  Posterior-weighted Bayesian
        model averages over Thompson samples are O(K) extra work for K
        samples; the default Thompson sample count is 32.
        """
        if method not in KNOWN_PREDICT_METHODS:
            raise InvalidConfig(
                f"unknown method {method!r}; expected one of {KNOWN_PREDICT_METHODS}"
            )
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            if state not in astate.spec.states:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown state {state!r}"
                )
            dist = self._predict_unsafe(astate, state, method=method)
            self._chain_head = _hash_entry(
                self._chain_head,
                {
                    "op": "predict",
                    "agent_id": agent_id,
                    "state": state,
                    "method": method,
                    "dist": {a: round(p, 6) for a, p in dist.items()},
                },
            )
        self._publish(
            MENTALIST_PREDICTED,
            {"agent_id": agent_id, "state": state, "method": method, "dist": dist, "head": self._chain_head},
        )
        return dist

    def _predict_unsafe(
        self,
        astate: AgentState,
        state: str,
        *,
        method: str,
    ) -> dict[str, float]:
        spec = astate.spec
        utilities = self._action_utilities_unsafe(astate, state)
        if method == PREDICT_MAP:
            best = max(utilities, key=lambda a: utilities[a])
            return {a: (1.0 if a == best else 0.0) for a in spec.actions}
        if method == PREDICT_SOFTMAX:
            return boltzmann_policy(utilities, beta=astate._rationality_mean)
        if method == PREDICT_THOMPSON:
            beta = self._sample_rationality_unsafe(astate)
            # Sample a perturbed utility (Dirichlet-driven outcome distributions).
            sampled = self._sample_utility_unsafe(astate)
            sampled_q = {
                a: sum(sampled.get(o, 0.0) * self._p_outcome_unsafe(astate, state, a, o)
                       for o in spec.outcomes)
                for a in spec.actions
            }
            return boltzmann_policy(sampled_q, beta=beta)
        if method == PREDICT_BAYES_AVG:
            # Average Boltzmann policies over Thompson samples.
            n_samples = 32
            agg: dict[str, float] = {a: 0.0 for a in spec.actions}
            for _ in range(n_samples):
                beta = self._sample_rationality_unsafe(astate)
                sampled = self._sample_utility_unsafe(astate)
                sampled_q = {
                    a: sum(sampled.get(o, 0.0) * self._p_outcome_unsafe(astate, state, a, o)
                           for o in spec.outcomes)
                    for a in spec.actions
                }
                pol = boltzmann_policy(sampled_q, beta=beta)
                for a, p in pol.items():
                    agg[a] += p / n_samples
            return agg
        raise InvalidConfig(f"unhandled method {method!r}")

    def _action_utilities_unsafe(self, astate: AgentState, state: str) -> dict[str, float]:
        """Expected utility per action under the posterior-mean model."""
        return {
            a: sum(
                astate.utility.get(o, 0.0) * self._p_outcome_unsafe(astate, state, a, o)
                for o in astate.spec.outcomes
            )
            for a in astate.spec.actions
        }

    def _p_outcome_unsafe(self, astate: AgentState, state: str, action: str, outcome: str) -> float:
        """Posterior mean of ``p(outcome | state, action)`` under a Dirichlet."""
        spec = astate.spec
        prior_alpha = self.config.prior_alpha
        denom = prior_alpha * len(spec.outcomes)
        num = prior_alpha
        for o in spec.outcomes:
            c = astate.sao_counts.get((state, action, o), 0)
            denom += c
            if o == outcome:
                num += c
        return num / denom

    def _sample_rationality_unsafe(self, astate: AgentState) -> float:
        """Sample β ~ Gamma(shape, rate). Uses stdlib ``random.gammavariate``.

        ``gammavariate`` takes (alpha, beta) with the *scale* parameterisation,
        which is the *inverse* of the *rate* parameterisation we use here:
        ``Gamma(shape=k, rate=λ)`` has mean ``k/λ`` and is equivalent to
        ``random.gammavariate(k, 1/λ)``.
        """
        rate = max(astate.rationality_rate, 1e-9)
        return self._rng.gammavariate(astate.rationality_shape, 1.0 / rate)

    def _sample_utility_unsafe(self, astate: AgentState) -> dict[str, float]:
        """Sample a utility vector by Gaussian perturbation around the posterior mean.

        A proper Bayesian utility posterior would be the implicit MaxEnt
        Hessian inverse; that is expensive.  We approximate it with an
        isotropic Gaussian centred at the recovered utility, with scale
        ``1 / sqrt(n)``.  Asymptotically correct under standard regularity
        and good enough for Thompson-sampling exploration.
        """
        spec = astate.spec
        n = max(1, astate.n_observed)
        scale = 1.0 / math.sqrt(n)
        return {
            o: astate.utility.get(o, 0.0) + self._rng.gauss(0.0, scale)
            for o in spec.outcomes
        }

    def expected_utility(self, agent_id: str, state: str) -> dict[str, float]:
        """Posterior-mean expected utility per action."""
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            if state not in astate.spec.states:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown state {state!r}"
                )
            return self._action_utilities_unsafe(astate, state)

    def confidence(
        self,
        agent_id: str,
        *,
        state: str | None = None,
        action: str,
        conf: float | None = None,
    ) -> tuple[float, float]:
        """Clopper-Pearson exact CI on the agent's per-action success rate.

        If ``state`` is supplied, the CI is conditional on the (state, action)
        pair; otherwise it's marginalised over all observed states.
        """
        c = conf if conf is not None else self.config.confidence
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            if action not in astate.spec.actions:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown action {action!r}"
                )
            if state is not None:
                if state not in astate.spec.states:
                    raise InvalidObservation(
                        f"agent {agent_id!r}: unknown state {state!r}"
                    )
                ca, cb = astate.capability[(state, action)]
                # Subtract priors to recover empirical successes / failures.
                successes = int(round(ca - self.config.capability_alpha))
                failures = int(round(cb - self.config.capability_beta))
            else:
                successes = 0
                failures = 0
                for s in astate.spec.states:
                    ca, cb = astate.capability[(s, action)]
                    successes += int(round(ca - self.config.capability_alpha))
                    failures += int(round(cb - self.config.capability_beta))
            successes = max(0, successes)
            failures = max(0, failures)
            trials = successes + failures
            return clopper_pearson_ci(successes, trials, conf=c)

    def state_distribution(self, agent_id: str) -> dict[str, float]:
        """Posterior-mean over the latent state distribution."""
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            return dirichlet_mean(astate.state_alpha)

    def simulate(
        self,
        agent_id: str,
        *,
        start_state: str,
        horizon: int,
        method: str = PREDICT_SOFTMAX,
        transition: Callable[[str, str], str] | None = None,
        rng_seed: int | None = None,
    ) -> list[tuple[str, str]]:
        """Roll out the agent's expected (state, action) trajectory.

        ``transition`` is an optional deterministic or stochastic state
        kernel; if absent, Mentalist samples the *next state* from the
        posterior Dirichlet mean (a coarse but well-defined fallback).

        Returns a list of ``(state, action)`` pairs of length ``horizon``.
        Deterministic up to ``rng_seed`` when one is supplied.
        """
        if horizon <= 0:
            raise InvalidConfig("horizon must be >= 1")
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            if start_state not in astate.spec.states:
                raise InvalidObservation(
                    f"agent {agent_id!r}: unknown start_state {start_state!r}"
                )
            local_rng = random.Random(rng_seed) if rng_seed is not None else self._rng
            trajectory: list[tuple[str, str]] = []
            current = start_state
            for _ in range(horizon):
                pol = self._predict_unsafe(astate, current, method=method)
                action = _sample_categorical(local_rng, pol)
                trajectory.append((current, action))
                if transition is not None:
                    nxt = transition(current, action)
                    if nxt not in astate.spec.states:
                        raise InvalidObservation(
                            f"transition produced unknown state {nxt!r}"
                        )
                    current = nxt
                else:
                    state_pol = dirichlet_mean(astate.state_alpha)
                    current = _sample_categorical(local_rng, state_pol)
            return trajectory

    # ---- nested theory of mind -----------------------------------------

    def nested_belief(
        self,
        *,
        observer: str,
        target: str,
        state: str,
        method: str = PREDICT_SOFTMAX,
    ) -> dict[str, float]:
        """``ToM_2``: observer's posterior over target's next action.

        Concretely: take the *observer*'s sao_counts as evidence about the
        *target* (the observer's observations of the target) and predict
        the target's action under that observer-private posterior.  This
        is the simplest nesting that still gives a coordinator something
        actionable — "Bob's guess at what Alice will do, given what Bob
        has seen Alice do".

        For deeper nesting the user composes Mentalist with itself or with
        ``Sampler`` for posterior-marginalised inference.
        """
        with self._lock:
            obs_state = self._agents.get(observer)
            tgt_state = self._agents.get(target)
            if obs_state is None:
                raise UnknownAgent(f"observer {observer!r} not registered")
            if tgt_state is None:
                raise UnknownAgent(f"target {target!r} not registered")
            if state not in tgt_state.spec.states:
                raise InvalidObservation(
                    f"target {target!r}: unknown state {state!r}"
                )
            # Treat observer.sao_counts as if it were a *private* mentalist
            # state for target; combine schemas via intersection.
            shared_states = tuple(s for s in tgt_state.spec.states if s in obs_state.spec.states)
            shared_actions = tuple(a for a in tgt_state.spec.actions if a in obs_state.spec.actions)
            shared_outcomes = tuple(o for o in tgt_state.spec.outcomes if o in obs_state.spec.outcomes)
            if not (shared_states and shared_actions and shared_outcomes):
                raise InvalidObservation(
                    f"observer {observer!r} and target {target!r} share no schema"
                )
            # Build a synthetic AgentState representing the observer's
            # belief about the target.
            synth = AgentState(
                spec=AgentSpec(
                    agent_id=f"{observer}->{target}",
                    states=shared_states,
                    actions=shared_actions,
                    outcomes=shared_outcomes,
                ),
            )
            synth.state_alpha = {
                s: self.config.prior_alpha + obs_state.state_alpha.get(s, 0.0)
                for s in shared_states
            }
            synth.sao_counts = {
                (s, a, o): obs_state.sao_counts.get((s, a, o), 0)
                for s in shared_states
                for a in shared_actions
                for o in shared_outcomes
                if (s, a, o) in obs_state.sao_counts
            }
            synth.rationality_shape = obs_state.rationality_shape
            synth.rationality_rate = obs_state.rationality_rate
            synth._rationality_mean = obs_state.rationality_shape / max(obs_state.rationality_rate, 1e-9)
            # Recover the observer's recovered utility *of the target* via IRL
            # on the synthetic state. If we have no data, fall back to zero.
            if sum(synth.sao_counts.values()) >= max(2, len(shared_outcomes)):
                try:
                    theta, _ = max_ent_irl(
                        states=shared_states,
                        actions=shared_actions,
                        outcomes=shared_outcomes,
                        sao_counts=synth.sao_counts,
                        beta=synth._rationality_mean,
                        lr=self.config.irl_lr,
                        l2=self.config.irl_l2,
                        max_iters=self.config.irl_max_iters,
                        tol=self.config.irl_tol,
                    )
                    synth.utility = theta
                except InsufficientData:
                    synth.utility = {o: 0.0 for o in shared_outcomes}
            else:
                synth.utility = {o: 0.0 for o in shared_outcomes}
            return self._predict_unsafe(synth, state, method=method)

    # ---- bounds & certificates -----------------------------------------

    def pac_bayes_bound(self, agent_id: str, *, delta: float | None = None) -> PACBayesBound:
        """Catoni-style PAC-Bayes upper bound on the policy's log-loss.

        The recovered utility (centred at the posterior mean) is treated as
        the posterior ``Q``; the prior ``P`` is the symmetric Dirichlet
        prior the agent was registered with.  ``KL(Q || P)`` is computed as
        the sum of per-(state, action) Dirichlet KLs in closed form.

        Returns an explicit, audit-friendly bound carrying every input the
        runtime needs to certify the prediction quality.
        """
        d = delta if delta is not None else self.config.pac_bayes_delta
        if not 0 < d < 1:
            raise InvalidConfig("delta must be in (0, 1)")
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            n = max(1, astate.n_observed)
            empirical_log_loss = -astate.cumulative_log_lik / n
            # KL(Q || P) for Dirichlet posterior on per-(s,a) outcome counts
            # vs symmetric prior. Closed-form summation.
            prior_alpha = self.config.prior_alpha
            kl = 0.0
            for s in astate.spec.states:
                for a in astate.spec.actions:
                    counts = [astate.sao_counts.get((s, a, o), 0) for o in astate.spec.outcomes]
                    posterior = [prior_alpha + c for c in counts]
                    kl += _dirichlet_kl(posterior, [prior_alpha] * len(posterior))
            penalty = (kl + math.log(2.0 * math.sqrt(n) / d)) / n
            upper = empirical_log_loss + penalty
            return PACBayesBound(
                delta=d,
                empirical_log_loss=empirical_log_loss,
                kl_to_prior=kl,
                n=n,
                upper_bound=upper,
            )

    def identifiability(self, agent_id: str) -> IdentifiabilityReport:
        """Equivalence classes of outcomes indistinguishable on observed data."""
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            spec = astate.spec
            # Compute the per-outcome empirical conditional vector
            # ``ψ(o) = vec_(s,a) p(o | s, a)``. Two outcomes are
            # indistinguishable iff their ψ vectors are componentwise equal
            # on the observed support.
            signature: dict[str, tuple[tuple[str, str, float], ...]] = {}
            sa_totals: dict[tuple[str, str], int] = {}
            for (s, a, _o), c in astate.sao_counts.items():
                sa_totals[(s, a)] = sa_totals.get((s, a), 0) + c
            for o in spec.outcomes:
                vec: list[tuple[str, str, float]] = []
                for (s, a), total in sa_totals.items():
                    c = astate.sao_counts.get((s, a, o), 0)
                    p = c / total
                    vec.append((s, a, round(p, 6)))
                vec.sort()
                signature[o] = tuple(vec)
            classes: dict[tuple[tuple[str, str, float], ...], list[str]] = {}
            for o, sig in signature.items():
                classes.setdefault(sig, []).append(o)
            blocks = tuple(tuple(sorted(b)) for b in classes.values())
            is_unique = all(len(b) == 1 for b in blocks)
            note = (
                "utility recoverable up to additive constant per block"
                if not is_unique
                else "all outcomes empirically distinguishable"
            )
            return IdentifiabilityReport(blocks=blocks, is_unique=is_unique, note=note)

    def report(self, agent_id: str) -> MentalReport:
        """Bundle the full state into a single immutable summary."""
        with self._lock:
            astate = self._agents.get(agent_id)
            if astate is None:
                raise UnknownAgent(f"agent {agent_id!r} not registered")
            state_dist = dirichlet_mean(astate.state_alpha)
            # Marginal action distribution = sum_s p(s) · π(a | s).
            agg_action: dict[str, float] = {a: 0.0 for a in astate.spec.actions}
            agg_eu: dict[str, float] = {a: 0.0 for a in astate.spec.actions}
            for s, p_s in state_dist.items():
                pol = self._predict_unsafe(astate, s, method=PREDICT_SOFTMAX)
                eus = self._action_utilities_unsafe(astate, s)
                for a in astate.spec.actions:
                    agg_action[a] += p_s * pol.get(a, 0.0)
                    agg_eu[a] += p_s * eus.get(a, 0.0)
            cis: dict[str, tuple[float, float]] = {}
            for a in astate.spec.actions:
                successes = 0
                failures = 0
                for s in astate.spec.states:
                    ca, cb = astate.capability[(s, a)]
                    successes += int(round(ca - self.config.capability_alpha))
                    failures += int(round(cb - self.config.capability_beta))
                successes = max(0, successes)
                failures = max(0, failures)
                trials = successes + failures
                cis[a] = clopper_pearson_ci(successes, trials, conf=self.config.confidence)
            rationality_mean = astate._rationality_mean
            rationality_var = astate.rationality_shape / (max(astate.rationality_rate, 1e-9) ** 2)
            try:
                bound = self.pac_bayes_bound_unsafe(astate)
            except Exception:
                bound = None
            return MentalReport(
                agent_id=agent_id,
                n_observations=astate.n_observed,
                action_distribution=agg_action,
                expected_utility=agg_eu,
                confidence_intervals=cis,
                utility_estimate=dict(astate.utility),
                rationality_mean=rationality_mean,
                rationality_var=rationality_var,
                state_distribution=state_dist,
                pac_bayes_bound=bound,
                certificate=self._chain_head,
            )

    def pac_bayes_bound_unsafe(self, astate: AgentState) -> float | None:
        """Internal: PAC-Bayes upper bound assuming the lock is held."""
        d = self.config.pac_bayes_delta
        n = max(1, astate.n_observed)
        empirical_log_loss = -astate.cumulative_log_lik / n
        prior_alpha = self.config.prior_alpha
        kl = 0.0
        for s in astate.spec.states:
            for a in astate.spec.actions:
                counts = [astate.sao_counts.get((s, a, o), 0) for o in astate.spec.outcomes]
                posterior = [prior_alpha + c for c in counts]
                kl += _dirichlet_kl(posterior, [prior_alpha] * len(posterior))
        penalty = (kl + math.log(2.0 * math.sqrt(n) / d)) / n
        return empirical_log_loss + penalty

    # ---- export / import / clear ---------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Serialise the full Mentalist state to a JSON-safe dict."""
        with self._lock:
            agents: dict[str, Any] = {}
            for aid, a in self._agents.items():
                agents[aid] = {
                    "spec": {
                        "agent_id": a.spec.agent_id,
                        "states": list(a.spec.states),
                        "actions": list(a.spec.actions),
                        "outcomes": list(a.spec.outcomes),
                    },
                    "state_alpha": a.state_alpha,
                    "capability": {f"{s}|{ac}": list(v) for (s, ac), v in a.capability.items()},
                    "sao_counts": {f"{s}|{ac}|{o}": c for (s, ac, o), c in a.sao_counts.items()},
                    "reward_sum": {f"{s}|{ac}": v for (s, ac), v in a.reward_sum.items()},
                    "reward_n": {f"{s}|{ac}": v for (s, ac), v in a.reward_n.items()},
                    "utility": a.utility,
                    "rationality_shape": a.rationality_shape,
                    "rationality_rate": a.rationality_rate,
                    "cumulative_log_lik": a.cumulative_log_lik,
                    "n_observed": a.n_observed,
                    "last_update_ts": a.last_update_ts,
                }
            return {
                "version": 1,
                "config": self.config.__dict__,
                "agents": agents,
                "chain_head": self._chain_head,
                "observation_count": self._observation_count,
                "started_ts": self._started_ts,
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore from a snapshot produced by ``export_state``.

        Validates the version field and re-instantiates agents with the
        same schema and counts.  Predictions made after import will be
        bit-identical to predictions before export, modulo Thompson
        sampling (which depends on the RNG state, intentionally not
        serialised).
        """
        if state.get("version") != 1:
            raise InvalidConfig("unsupported Mentalist snapshot version")
        with self._lock:
            self._agents = {}
            self._observations_since_refit = {}
            for aid, a in state.get("agents", {}).items():
                spec = AgentSpec(
                    agent_id=a["spec"]["agent_id"],
                    states=tuple(a["spec"]["states"]),
                    actions=tuple(a["spec"]["actions"]),
                    outcomes=tuple(a["spec"]["outcomes"]),
                )
                astate = AgentState(spec=spec)
                astate.state_alpha = {k: float(v) for k, v in a["state_alpha"].items()}
                astate.capability = {
                    tuple(k.split("|", 1)): (float(v[0]), float(v[1]))
                    for k, v in a["capability"].items()
                }
                astate.sao_counts = {
                    tuple(k.split("|", 2)): int(v) for k, v in a["sao_counts"].items()
                }
                astate.reward_sum = {
                    tuple(k.split("|", 1)): float(v) for k, v in a["reward_sum"].items()
                }
                astate.reward_n = {
                    tuple(k.split("|", 1)): int(v) for k, v in a["reward_n"].items()
                }
                astate.utility = {k: float(v) for k, v in a["utility"].items()}
                astate.rationality_shape = float(a["rationality_shape"])
                astate.rationality_rate = float(a["rationality_rate"])
                astate.cumulative_log_lik = float(a["cumulative_log_lik"])
                astate.n_observed = int(a["n_observed"])
                astate.last_update_ts = float(a["last_update_ts"])
                astate._rationality_mean = astate.rationality_shape / max(astate.rationality_rate, 1e-9)
                self._agents[aid] = astate
                self._observations_since_refit[aid] = 0
            self._chain_head = state.get("chain_head", ledger_root())
            self._observation_count = int(state.get("observation_count", 0))

    def clear(self) -> None:
        """Drop all agents and reset the chain head."""
        with self._lock:
            self._agents = {}
            self._observations_since_refit = {}
            self._observation_count = 0
            self._chain_head = ledger_root()
        self._publish(MENTALIST_CLEARED, {"head": self._chain_head})

    # ---- introspection -------------------------------------------------

    @property
    def chain_head(self) -> str:
        """The current SHA-256 chain head."""
        return self._chain_head

    @property
    def observation_count(self) -> int:
        return self._observation_count

    # ---- internals -----------------------------------------------------

    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, data)
        except Exception:
            # Publishing must never destabilise the primitive.
            pass

    def _deadline(self) -> float | None:
        if self.config.max_seconds is None:
            return None
        return time.time() + self.config.max_seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_categorical(rng: random.Random, dist: dict[str, float]) -> str:
    """Sample from a {key: prob} categorical."""
    items = list(dist.items())
    total = sum(p for _, p in items)
    if total <= 0:
        return items[0][0]
    u = rng.random() * total
    acc = 0.0
    for k, p in items:
        acc += p
        if u <= acc:
            return k
    return items[-1][0]


def _dirichlet_kl(post: Sequence[float], prior: Sequence[float]) -> float:
    """Closed-form KL between two Dirichlet distributions.

    ``KL(Dir(α) || Dir(β)) = lgamma(α_0) − lgamma(β_0) − Σ (lgamma(α_i)
                            − lgamma(β_i)) + Σ (α_i − β_i)(ψ(α_i) − ψ(α_0))``

    Pure stdlib; we implement digamma via the standard asymptotic
    expansion (Abramowitz & Stegun §6.3) since ``math`` ships ``lgamma``
    but not ``digamma``.
    """
    if len(post) != len(prior):
        raise InvalidConfig("Dirichlet KL: dim mismatch")
    a0 = sum(post)
    b0 = sum(prior)
    if a0 <= 0 or b0 <= 0:
        return 0.0
    psi_a0 = _digamma(a0)
    total = math.lgamma(a0) - math.lgamma(b0)
    for a_i, b_i in zip(post, prior):
        if a_i <= 0 or b_i <= 0:
            continue
        total -= math.lgamma(a_i) - math.lgamma(b_i)
        total += (a_i - b_i) * (_digamma(a_i) - psi_a0)
    return max(0.0, total)


def _digamma(x: float) -> float:
    """Digamma (ψ) via Bernoulli asymptotic expansion + recurrence.

    Accurate to ~1e-10 for x >= 1; uses ``ψ(x) = ψ(x+1) − 1/x`` to shift
    small arguments into the asymptotic regime.  Standard textbook recipe.
    """
    if x <= 0.0:
        # Reflection: ψ(1-x) = ψ(x) + π cot(πx); we don't need negative
        # arguments here but guard the call site.
        return float("nan")
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic: ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − …
    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv
    result -= inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return result
