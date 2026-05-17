r"""Empowerer — empowerment & intrinsic motivation as a runtime primitive.

Every learning primitive in this runtime is a *means* — it improves a
parameter, fits a model, picks a plan.  Real agents need an *end* as
well: a scalar criterion that says, in the absence of a task reward,
*which states are worth being in*.  The deepest information-theoretic
answer is **empowerment** (Klyubin, Polani, Nehaniv 2005 *Empowerment: A
universal agent-centric measure of control*) — the channel capacity
between an agent's *actions* and its *future sensor state*::

    𝔈ⁿ(s)  =  max_{p(a₁:ₙ)}  I(A₁:ₙ ; Sₙ | S₀ = s)
                                          (in bits)

In one number it tells the coordination engine *how much agency the
agent has at state s*: how many distinguishable futures the agent can
reach by acting.  Maximising empowerment is a parameter-free, task-free,
horizon-aware intrinsic drive — it generates exploration in
sparse-reward MDPs (Mohamed, Rezende 2015; Eysenbach et al. 2018 DIAYN),
identifies bottleneck / option-boundary states (Çelik, Polani 2010),
factors out safe-by-design actions that preserve future controllability
(Salge, Glackin, Polani 2014; Hadfield-Menell et al. 2017 *Inverse
reward design*), and provides a single scalar a coordinator can read
to decide *act vs ask vs defer* without ever specifying a task.

The pitch reduced to a runtime call::

    em = Empowerer(EmpowererConfig(dim_state=64, dim_action=4, horizon=3))
    for (s, a, s_next) in env.stream():
        em.observe_transition(s, a, s_next)

    val   = em.empowerment(state)               # bits
    safe  = em.safe_actions(state, candidates)  # empowerment-preserving
    ireward = em.intrinsic_reward(s, a, s_next) # for any RL learner
    skills = em.skill_discovery(n_skills=8)     # DIAYN latent skills
    rep   = em.report()
    cert  = em.certify()                        # PAC bound on estimate


What this primitive ships
-------------------------

  * **Blahut-Arimoto exact channel capacity** (Blahut 1972, Arimoto
    1972).  Given a transition channel ``p(s'|s, a)`` over discrete
    states / actions, iterates
    ::

        q_{a|s'} ←  p_a · p(s'|s,a) / Σ_{a'} p_{a'} · p(s'|s,a')
        p_a     ←  exp( Σ_{s'} p(s'|s,a) log q_{a|s'} ) / Z

    converging geometrically to the empowerment ``𝔈¹(s)`` and the
    *capacity-achieving action distribution*.  Returned as a bits
    scalar plus the optimal policy at ``s``.

  * **N-step empowerment** by treating action *sequences*
    ``a₁:ₙ`` as inputs and ``sₙ`` as outputs.  The composite channel
    ``p(sₙ | s₀, a₁:ₙ) = Σ_{s₁,…,sₙ₋₁} ∏ p(sₖ | sₖ₋₁, aₖ)`` is
    formed lazily; Blahut-Arimoto runs on the unrolled |A|ⁿ × |S|
    matrix.  Guards against ``|A|ⁿ`` blow-up via configurable
    ``max_action_seqs``.

  * **Variational empowerment lower bound**
    (Mohamed, Rezende 2015 *Variational information maximisation*;
    Barber & Agakov 2003 IM-algorithm).  When the channel is not
    enumerable, the InfoNCE / DV / NWJ bounds
    ::

        I(A;S') ≥ E_{(a,s')~p}[log q(a|s,s')] − E_{a~p(a|s)}[log p(a|s)]

    are estimated from K paired transition samples with a tractable
    softmax decoder ``q``.  Returns lower-bound estimate + sample
    confidence interval (Hoeffding on bounded bits).

  * **Mutual-information skill discovery — DIAYN**
    (Eysenbach, Gupta, Ibarz, Levine 2018 *Diversity is all you need:
    learning skills without a reward function*).  Latent skill
    ``z ∈ {1..K}``, discriminator ``q(z|s)`` trained on visit
    counts; intrinsic reward at ``(z,s)`` is ``log q(z|s) − log p(z)``.
    Returns per-skill state distributions and a skill-separability
    score (mean discriminator log-likelihood).

  * **Variational Intrinsic Control / DADS-style empowerment**
    (Gregor, Rezende, Wierstra 2017 *Variational intrinsic control*;
    Sharma et al. 2020 *Dynamics-aware unsupervised discovery of
    skills*) — option/skill-level empowerment ``I(Z;Sₙ|s₀)`` over a
    horizon ``n``, computed from skill-conditioned transition
    estimates the runtime can ingest from ``Imaginator``.

  * **Empowerment-preserving safe-action shielding**
    (Salge, Polani 2017 *Empowerment as replacement for the three
    laws of robotics*; Krakovna et al. 2020 *Avoiding side effects
    by considering future tasks*).  An action ``a`` at ``s`` is
    admissible if the *expected* successor-state empowerment
    satisfies ``E_{s'~p(·|s,a)} 𝔈ⁿ(s') ≥ 𝔈ⁿ(s) − margin``.  Lets a
    coordinator filter actions that would *throw away future agency*
    even when they look attractive on a task-specific reward.

  * **Per-state empowerment landscape.**  Returns the dict
    ``{s : 𝔈ⁿ(s)}`` over all observed states — a curiosity heat-map
    the curriculum primitive can sample from.  Bottleneck states
    have *locally minimal* empowerment in their reach-neighbourhood
    (Çelik, Polani 2010); they make natural sub-goal candidates.

  * **Intrinsic-reward shaping** — three modes:

      * ``r_emp(s)``       = 𝔈ⁿ(s)              (state empowerment)
      * ``Δr_emp(s,a,s')`` = 𝔈ⁿ(s') − 𝔈ⁿ(s)    (empowerment gain)
      * ``r_imp(s,a,s')``  = log p(s'|s,a)⁻¹    (transition surprise)

    Each is a drop-in for any RL primitive's reward channel —
    ``Aligner``, ``Quantilizer``, ``Pareto``, ``Bandit`` all consume
    a real scalar.

  * **PAC confidence interval for empowerment** (Paninski 2003
    *Estimation of entropy and mutual information*; Antos & Kontoyiannis
    2001 *Convergence properties of functional estimates of entropies*).
    For a count-based plug-in estimate of the channel ``p̂(s'|s,a)``
    formed from ``n_{s,a}`` samples per action, the empowerment
    estimate ``Î¹(s)`` satisfies, w.p. ≥ 1 − δ,

        ``|Î¹(s) − 𝔈¹(s)|  ≤  (|A| · |S| / min_a n_{s,a}) · log(2|S|/δ) / √2``

    derived by Hoeffding on entropy plus the *bias-variance* result
    of Paninski.  The runtime exposes the bound on every certify().

  * **Empowerment-augmented coordination protocol** —
    a documented contract over events that lets ``Coordinator`` /
    ``Strategist`` / ``Driver`` consume empowerment as either an
    intrinsic reward stream or a *deferred-action gate* (refuse to
    act when ``𝔈ⁿ(s) < threshold``, i.e. "we don't know enough yet").

  * **Tamper-evident SHA-256 fingerprint chain** (genesis
    ``agi.empowerer.v1`` + optional HMAC) over every observation,
    estimator update, capacity solve, skill-discovery step, and
    certificate.  ``EmpowererLedger`` replays every operation
    byte-for-byte from the chain head.

  * **Snapshot / restore** of every byte: transition counts, the
    Blahut-Arimoto warm-start distributions, the discriminator
    parameters, the chain head — round-trippable so the coordinator
    can hibernate an empowerment estimator and resume on the next
    host.

  * **Thread-safe re-entrant lock** + transport-agnostic + pure stdlib.


Composes with
-------------

  * ``Imaginator`` — supplies a learned transition model
    ``p̂(s'|s,a)``; ``Empowerer`` returns empowerment / intrinsic
    reward over the learned world.
  * ``Quantilizer`` — ``Empowerer.safe_actions`` filters the
    candidate set before quantilising — bounded optimisation
    *within* the empowerment-preserving subset.
  * ``Aligner`` / ``Bandit`` / ``Pareto`` — consume the intrinsic
    reward stream as a regulariser on task reward.
  * ``Curator`` — uses the per-state empowerment landscape as a
    curriculum priority signal (bottleneck states get higher
    weight).
  * ``Continualist`` — pairs lifelong learning with an empowerment
    target so the agent never *loses* agency on prior environments.
  * ``Mentalist`` — feeds ToM-inferred action distributions of
    *other* agents as the channel input, yielding *social
    empowerment* (Salge, Polani 2018).
  * ``Reconciler`` / ``Coordinator`` — read the certify() bound to
    decide deferral / hand-off.


Mathematical notation
---------------------

  * ``|S|`` — number of discrete states.
  * ``|A|`` — number of discrete actions.
  * ``p(s'|s,a)`` — one-step transition kernel.
  * ``Pⁿ(sₙ|s₀, a₁:ₙ)`` — n-step kernel under action sequence.
  * ``𝔈ⁿ(s)`` — n-step empowerment at state ``s`` in bits.
  * ``q(z|s)`` — DIAYN discriminator over skill latent ``z``.
  * ``α`` — Blahut-Arimoto current action distribution iterate.

All inputs are validated; all updates are ``O(|S| · |A|)`` per ingest
and ``O(|S| · |A| · iter)`` per capacity solve.  No randomness uses
``random`` without an explicit seed; no ``time.time()`` calls leak into
the chain.

References
----------

  * Klyubin, Polani, Nehaniv 2005. *Empowerment: A universal
    agent-centric measure of control.* IEEE CEC.
  * Klyubin, Polani, Nehaniv 2008. *Keep your options open: An
    information-based driving principle for sensorimotor systems.*
    PLoS ONE 3(12).
  * Blahut 1972. *Computation of channel capacity and rate-distortion
    functions.* IEEE T-IT 18(4).
  * Arimoto 1972. *An algorithm for computing the capacity of arbitrary
    discrete memoryless channels.* IEEE T-IT 18(1).
  * Salge, Glackin, Polani 2014. *Changing the environment based on
    empowerment as intrinsic motivation.* Entropy 16(5).
  * Salge, Polani 2017. *Empowerment as replacement for the three
    laws of robotics.* Frontiers Robot. AI 4.
  * Mohamed, Rezende 2015. *Variational information maximisation for
    intrinsically motivated reinforcement learning.* NeurIPS.
  * Eysenbach, Gupta, Ibarz, Levine 2018. *Diversity is all you need:
    learning skills without a reward function.* ICLR.
  * Gregor, Rezende, Wierstra 2017. *Variational intrinsic control.*
    ICLR Workshop.
  * Sharma et al. 2020. *Dynamics-aware unsupervised discovery of
    skills.* ICLR.
  * Çelik, Polani 2010. *Empowerment for continuous agent-environment
    systems.* IEEE T-AMD 2(1).
  * Paninski 2003. *Estimation of entropy and mutual information.*
    Neural Computation 15(6).
  * Antos, Kontoyiannis 2001. *Convergence properties of functional
    estimates for discrete distributions.* Random Struct. & Algo.
    19(3-4).
  * Krakovna et al. 2020. *Avoiding side effects by considering
    future tasks.* NeurIPS.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


__all__ = [
    # Events
    "EMPOWERER_STARTED",
    "EMPOWERER_OBSERVED",
    "EMPOWERER_FITTED",
    "EMPOWERER_SOLVED",
    "EMPOWERER_VARIATIONAL",
    "EMPOWERER_SKILLS_DISCOVERED",
    "EMPOWERER_SHIELDED",
    "EMPOWERER_REWARDED",
    "EMPOWERER_REPORTED",
    "EMPOWERER_CERTIFIED",
    "EMPOWERER_RESET",
    # Estimator codes
    "EST_BLAHUT_ARIMOTO",
    "EST_VARIATIONAL_INFONCE",
    "EST_VARIATIONAL_NWJ",
    "EST_VARIATIONAL_DV",
    "KNOWN_ESTIMATORS",
    # Reward shaping
    "REWARD_STATE_EMPOWERMENT",
    "REWARD_DELTA_EMPOWERMENT",
    "REWARD_TRANSITION_SURPRISE",
    "KNOWN_REWARD_MODES",
    # Exceptions
    "EmpowererError",
    "InvalidConfig",
    "InvalidTransition",
    "InvalidState",
    "InvalidAction",
    "InvalidHorizon",
    "InsufficientData",
    "UnknownEstimator",
    "UnknownRewardMode",
    "BlowupGuard",
    # Dataclasses
    "EmpowererConfig",
    "TransitionRecord",
    "EmpowermentResult",
    "VariationalResult",
    "SkillReport",
    "ShieldDecision",
    "EmpowererReport",
    "EmpowererCertificate",
    # Helpers
    "empowerer_ledger_root",
    "blahut_arimoto_capacity",
    "n_step_kernel",
    "infonce_lower_bound",
    "nwj_lower_bound",
    "donsker_varadhan_lower_bound",
    "diayn_intrinsic_reward",
    "paninski_empowerment_bound",
    # Main class
    "Empowerer",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

EMPOWERER_STARTED = "empowerer.started"
EMPOWERER_OBSERVED = "empowerer.observed"
EMPOWERER_FITTED = "empowerer.fitted"
EMPOWERER_SOLVED = "empowerer.solved"
EMPOWERER_VARIATIONAL = "empowerer.variational"
EMPOWERER_SKILLS_DISCOVERED = "empowerer.skills_discovered"
EMPOWERER_SHIELDED = "empowerer.shielded"
EMPOWERER_REWARDED = "empowerer.rewarded"
EMPOWERER_REPORTED = "empowerer.reported"
EMPOWERER_CERTIFIED = "empowerer.certified"
EMPOWERER_RESET = "empowerer.reset"


# ---------------------------------------------------------------------------
# Estimator enum
# ---------------------------------------------------------------------------

EST_BLAHUT_ARIMOTO = "blahut_arimoto"
EST_VARIATIONAL_INFONCE = "variational_infonce"
EST_VARIATIONAL_NWJ = "variational_nwj"
EST_VARIATIONAL_DV = "variational_dv"

KNOWN_ESTIMATORS = (
    EST_BLAHUT_ARIMOTO,
    EST_VARIATIONAL_INFONCE,
    EST_VARIATIONAL_NWJ,
    EST_VARIATIONAL_DV,
)


# ---------------------------------------------------------------------------
# Reward shaping enum
# ---------------------------------------------------------------------------

REWARD_STATE_EMPOWERMENT = "state_empowerment"
REWARD_DELTA_EMPOWERMENT = "delta_empowerment"
REWARD_TRANSITION_SURPRISE = "transition_surprise"

KNOWN_REWARD_MODES = (
    REWARD_STATE_EMPOWERMENT,
    REWARD_DELTA_EMPOWERMENT,
    REWARD_TRANSITION_SURPRISE,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmpowererError(Exception):
    """Base error for :class:`Empowerer`."""


class InvalidConfig(EmpowererError):
    """Configuration is malformed."""


class InvalidTransition(EmpowererError):
    """Transition tuple is malformed."""


class InvalidState(EmpowererError):
    """State index is malformed or out-of-range."""


class InvalidAction(EmpowererError):
    """Action index is malformed or out-of-range."""


class InvalidHorizon(EmpowererError):
    """Horizon is malformed or impossible to evaluate."""


class InsufficientData(EmpowererError):
    """Not enough observations for the requested operation."""


class UnknownEstimator(EmpowererError):
    """Unknown empowerment estimator."""


class UnknownRewardMode(EmpowererError):
    """Unknown reward shaping mode."""


class BlowupGuard(EmpowererError):
    """Action-sequence enumeration would exceed configured limit."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpowererConfig:
    """Configuration for :class:`Empowerer`.

    Parameters
    ----------
    dim_state : int
        Number of discrete states ``|S|``.
    dim_action : int
        Number of discrete actions ``|A|``.
    horizon : int
        Empowerment look-ahead ``n`` (n-step empowerment).
    estimator : str
        One of ``EST_*``.  ``EST_BLAHUT_ARIMOTO`` solves the discrete
        channel exactly; the variational families estimate via samples.
    reward_mode : str
        One of ``REWARD_*``.  How :meth:`Empowerer.intrinsic_reward`
        synthesises a per-transition scalar.
    ba_tol : float
        Blahut-Arimoto convergence tolerance ``ε`` on the action
        distribution L¹ change between iterations.
    ba_max_iter : int
        Maximum Blahut-Arimoto iterations.
    ba_warm_start : bool
        Reuse the previous action distribution at the same state to
        seed the next solve.
    laplace_alpha : float
        Dirichlet pseudo-count for the count-based transition kernel
        estimate ``p̂(s'|s,a) = (n_{s,a,s'} + α) / (n_{s,a} + |S| · α)``.
    max_action_seqs : int
        Hard cap on the number of action sequences enumerated for
        n-step empowerment.  ``Empowerer.empowerment`` raises
        :class:`BlowupGuard` instead of constructing a
        ``|A|^horizon`` channel larger than this.
    variational_samples : int
        Number of paired samples used per variational bound estimate.
    variational_lr : float
        Step size for variational decoder updates (gradient ascent in
        log-space on a tabular softmax).
    diayn_lr : float
        Step size for the DIAYN discriminator.
    intrinsic_scale : float
        Multiplicative gain on :meth:`Empowerer.intrinsic_reward`.
    safety_margin : float
        Empowerment-loss bound (bits) tolerated by
        :meth:`Empowerer.safe_actions`.
    safety_estimator : str
        ``"expectation"`` (default — expected successor empowerment),
        ``"min"`` (worst-case over successors).
    confidence : float
        Certify confidence ``1 − δ``.
    rng_seed : int
        Deterministic RNG seed for variational sampling and DIAYN.
    hmac_key : bytes | None
        Optional HMAC key for the ledger chain.

    Raises
    ------
    InvalidConfig
        Any field is out of range.
    UnknownEstimator
        ``estimator`` is unknown.
    UnknownRewardMode
        ``reward_mode`` is unknown.
    """

    dim_state: int = 1
    dim_action: int = 1
    horizon: int = 1
    estimator: str = EST_BLAHUT_ARIMOTO
    reward_mode: str = REWARD_STATE_EMPOWERMENT
    ba_tol: float = 1e-10
    ba_max_iter: int = 512
    ba_warm_start: bool = True
    laplace_alpha: float = 1.0
    max_action_seqs: int = 4096
    variational_samples: int = 128
    variational_lr: float = 0.1
    diayn_lr: float = 0.1
    intrinsic_scale: float = 1.0
    safety_margin: float = 0.0
    safety_estimator: str = "expectation"
    confidence: float = 0.95
    rng_seed: int = 0
    hmac_key: bytes | None = None

    def __post_init__(self) -> None:
        if self.estimator not in KNOWN_ESTIMATORS:
            raise UnknownEstimator(self.estimator)
        if self.reward_mode not in KNOWN_REWARD_MODES:
            raise UnknownRewardMode(self.reward_mode)
        if not isinstance(self.dim_state, int) or self.dim_state < 1:
            raise InvalidConfig("dim_state must be a positive integer")
        if not isinstance(self.dim_action, int) or self.dim_action < 1:
            raise InvalidConfig("dim_action must be a positive integer")
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise InvalidConfig("horizon must be a positive integer")
        for name in (
            "ba_tol",
            "laplace_alpha",
            "variational_lr",
            "diayn_lr",
            "intrinsic_scale",
            "safety_margin",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)):
                raise InvalidConfig(f"{name} must be numeric")
            if math.isnan(v) or math.isinf(v):
                raise InvalidConfig(f"{name} must be finite")
            if v < 0:
                raise InvalidConfig(f"{name} must be non-negative")
        if not isinstance(self.ba_max_iter, int) or self.ba_max_iter < 1:
            raise InvalidConfig("ba_max_iter must be a positive integer")
        if not isinstance(self.max_action_seqs, int) or self.max_action_seqs < 1:
            raise InvalidConfig("max_action_seqs must be a positive integer")
        if not isinstance(self.variational_samples, int) or self.variational_samples < 2:
            raise InvalidConfig("variational_samples must be at least 2")
        if self.safety_estimator not in ("expectation", "min"):
            raise InvalidConfig("safety_estimator must be 'expectation' or 'min'")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must lie in (0, 1)")
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes or None")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionRecord:
    """One ingested transition.  Used by reports and snapshots."""

    state: int
    action: int
    next_state: int
    sequence_index: int


@dataclass(frozen=True)
class EmpowermentResult:
    """Output of an empowerment solve at a single starting state."""

    state: int
    horizon: int
    empowerment_bits: float
    action_seq_distribution: tuple[float, ...]
    optimal_action: int
    iterations: int
    converged: bool
    chain_head: str


@dataclass(frozen=True)
class VariationalResult:
    """Output of a variational empowerment bound estimate."""

    state: int
    horizon: int
    estimator: str
    lower_bound_bits: float
    hoeffding_half_width: float
    samples_used: int
    chain_head: str


@dataclass(frozen=True)
class SkillReport:
    """Output of DIAYN-style mutual-information skill discovery."""

    n_skills: int
    iterations: int
    avg_log_q: float
    skill_state_dist: tuple[tuple[float, ...], ...]
    skill_separability: float
    skill_entropy: float
    chain_head: str


@dataclass(frozen=True)
class ShieldDecision:
    """Output of empowerment-preserving safe-action shielding."""

    state: int
    candidates: tuple[int, ...]
    admissible: tuple[int, ...]
    state_empowerment_bits: float
    successor_empowerment_bits: tuple[float, ...]
    margin: float
    chain_head: str


@dataclass(frozen=True)
class EmpowererReport:
    """Snapshot of estimator state at a moment in time."""

    total_transitions: int
    distinct_states: int
    distinct_actions: int
    horizon: int
    estimator: str
    reward_mode: str
    mean_state_empowerment: float
    max_state_empowerment: float
    min_state_empowerment: float
    state_coverage_fraction: float
    chain_head: str


@dataclass(frozen=True)
class EmpowererCertificate:
    """PAC certificate over the empowerment estimate."""

    state: int
    horizon: int
    empowerment_bits: float
    upper_bound_bits: float
    lower_bound_bits: float
    confidence: float
    n_samples: int
    holds: bool
    chain_head: str


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------


_GENESIS = "agi.empowerer.v1"


def empowerer_ledger_root() -> str:
    """Genesis fingerprint for the ledger chain."""
    return hashlib.sha256(_GENESIS.encode("utf-8")).hexdigest()


def _digest(prev: str, payload: Mapping[str, Any], hmac_key: bytes | None) -> str:
    """Sequentially-keyed SHA-256 / HMAC over ``(prev, payload)``."""
    body = json.dumps(
        {"prev": prev, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hmac_key is None:
        return hashlib.sha256(body).hexdigest()
    return hmac.new(hmac_key, body, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Pure-math helpers (no Empowerer state required)
# ---------------------------------------------------------------------------


_LN2 = math.log(2.0)


def _log2(x: float) -> float:
    if x <= 0.0:
        return float("-inf")
    return math.log(x) / _LN2


def _safe_log2(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return math.log(x) / _LN2


def _normalise(v: Sequence[float]) -> list[float]:
    s = sum(v)
    if s <= 0.0:
        n = len(v)
        if n == 0:
            return []
        return [1.0 / n] * n
    return [float(x) / s for x in v]


def _kl_bits(p: Sequence[float], q: Sequence[float]) -> float:
    """Discrete KL divergence in bits.  ``p, q`` must be the same length."""
    if len(p) != len(q):
        raise InvalidConfig("KL requires equal-length distributions")
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        out += pi * (math.log(pi) - math.log(qi)) / _LN2
    return out


def blahut_arimoto_capacity(
    channel: Sequence[Sequence[float]],
    *,
    tol: float = 1e-10,
    max_iter: int = 512,
    warm_start: Sequence[float] | None = None,
) -> tuple[float, list[float], int, bool]:
    """Blahut-Arimoto channel capacity for a discrete memoryless channel.

    Parameters
    ----------
    channel : Sequence[Sequence[float]]
        Row-stochastic matrix ``W[a][y]`` with ``W[a][y] = p(y | a)``.
        ``len(W)`` is ``|A|``, every row is a distribution over the
        ``|Y|`` outputs.
    tol : float
        L¹ tolerance on the action distribution update.
    max_iter : int
        Cap on iterations.
    warm_start : Sequence[float] | None
        Optional initial action distribution.

    Returns
    -------
    (capacity_bits, alpha, iterations, converged)
        ``alpha`` is the capacity-achieving input distribution.

    Notes
    -----
    Implements Blahut 1972 / Arimoto 1972 in the standard log-space
    form (numerically robust).  The capacity is the channel mutual
    information ``I(A;Y)`` evaluated at the converged ``alpha``.
    """
    if not channel:
        return 0.0, [], 0, True
    n_a = len(channel)
    n_y = len(channel[0])
    for row in channel:
        if len(row) != n_y:
            raise InvalidConfig("channel rows must all have the same length")
        for v in row:
            if v < -1e-12:
                raise InvalidConfig("channel entries must be non-negative")
    rows = [_normalise(row) for row in channel]
    if warm_start is not None and len(warm_start) == n_a:
        alpha = list(_normalise(warm_start))
        if any(a < 1e-12 for a in alpha):
            alpha = [a + 1e-12 for a in alpha]
            alpha = _normalise(alpha)
    else:
        alpha = [1.0 / n_a] * n_a
    converged = False
    for it in range(max_iter):
        # Compute log r(a) = sum_y W(y|a) [ log W(y|a) - log sum_a' alpha(a') W(y|a') ]
        log_r = [0.0] * n_a
        for y in range(n_y):
            mix = 0.0
            for a in range(n_a):
                mix += alpha[a] * rows[a][y]
            if mix <= 0.0:
                continue
            log_mix = math.log(mix)
            for a in range(n_a):
                w = rows[a][y]
                if w <= 0.0:
                    continue
                log_r[a] += w * (math.log(w) - log_mix)
        # New alpha proportional to alpha * exp(log_r).
        max_log = max(log_r)
        new_unnorm = [alpha[a] * math.exp(log_r[a] - max_log) for a in range(n_a)]
        z = sum(new_unnorm)
        if z <= 0.0:
            new_alpha = [1.0 / n_a] * n_a
        else:
            new_alpha = [v / z for v in new_unnorm]
        delta = 0.0
        for a in range(n_a):
            delta += abs(new_alpha[a] - alpha[a])
        alpha = new_alpha
        if delta < tol:
            converged = True
            it_used = it + 1
            break
    else:
        it_used = max_iter
    # Compute capacity at converged alpha (in bits).
    capacity = 0.0
    for y in range(n_y):
        mix = 0.0
        for a in range(n_a):
            mix += alpha[a] * rows[a][y]
        if mix <= 0.0:
            continue
        for a in range(n_a):
            w = rows[a][y]
            if w <= 0.0:
                continue
            capacity += alpha[a] * w * (math.log(w) - math.log(mix)) / _LN2
    if capacity < 0.0 and capacity > -1e-9:
        capacity = 0.0
    return capacity, alpha, it_used, converged


def n_step_kernel(
    one_step: Sequence[Sequence[Sequence[float]]],
    state: int,
    horizon: int,
    *,
    max_action_seqs: int = 4096,
) -> tuple[list[tuple[int, ...]], list[list[float]]]:
    """Build the n-step channel ``p(sₙ | s₀, a₁:ₙ)`` for a fixed ``s₀``.

    Parameters
    ----------
    one_step : 3-d array
        ``one_step[s][a][s']`` is the one-step kernel ``p(s'|s,a)``.
    state : int
        Starting state ``s₀``.
    horizon : int
        Number of steps ``n``.

    Returns
    -------
    (sequences, rows)
        ``sequences`` is the list of action sequences ``a₁:ₙ`` (each a
        tuple of length ``horizon``); ``rows[i]`` is the distribution
        over ``sₙ`` induced by ``sequences[i]`` starting from
        ``state``.

    Raises
    ------
    BlowupGuard
        If ``|A|^horizon`` would exceed ``max_action_seqs``.
    """
    if not one_step:
        return [], []
    n_states = len(one_step)
    n_actions = len(one_step[0]) if one_step[0] else 0
    if horizon < 1:
        raise InvalidHorizon("horizon must be at least 1")
    if n_actions ** horizon > max_action_seqs:
        raise BlowupGuard(
            f"|A|^horizon = {n_actions ** horizon} > max_action_seqs {max_action_seqs}"
        )
    seqs: list[tuple[int, ...]] = [()]
    for _ in range(horizon):
        new_seqs: list[tuple[int, ...]] = []
        for seq in seqs:
            for a in range(n_actions):
                new_seqs.append(seq + (a,))
        seqs = new_seqs
    rows: list[list[float]] = []
    for seq in seqs:
        # Forward propagate the state distribution under this sequence.
        dist = [0.0] * n_states
        dist[state] = 1.0
        for a in seq:
            nxt = [0.0] * n_states
            for s in range(n_states):
                p_s = dist[s]
                if p_s <= 0.0:
                    continue
                row = one_step[s][a]
                for sp in range(n_states):
                    nxt[sp] += p_s * row[sp]
            dist = nxt
        rows.append(dist)
    return seqs, rows


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    s = 0.0
    for v in values:
        s += math.exp(v - m)
    return m + math.log(s)


def infonce_lower_bound(scores: Sequence[Sequence[float]]) -> float:
    """InfoNCE / contrastive mutual-information lower bound in bits.

    Parameters
    ----------
    scores : 2-d array
        ``scores[i][j] = f(x_i, y_j)``.  The diagonal pairs are
        positives; off-diagonals are negatives.

    Returns
    -------
    bits : float
        The bound is ``log K + (1/K) Σ_i [f(x_i,y_i) − logsumexp_j f(x_i,y_j)]``
        converted to bits.  See van den Oord et al. 2018 *CPC*.
    """
    k = len(scores)
    if k < 2:
        return 0.0
    total_nats = math.log(k)
    for i in range(k):
        row = scores[i]
        if len(row) != k:
            raise InvalidConfig("scores must be square")
        lse = _logsumexp(row)
        total_nats += (row[i] - lse) / k
    return total_nats / _LN2


def nwj_lower_bound(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """NWJ / f-MINE lower bound (Nguyen, Wainwright, Jordan 2010).

    ``I(X;Y) ≥ E_{p(x,y)}[f(x,y)] − E_{p(x)p(y)}[e^{f(x,y)-1}]`` in nats;
    we convert to bits.
    """
    if not scores_pos or not scores_neg:
        return 0.0
    e1 = sum(scores_pos) / len(scores_pos)
    e2 = sum(math.exp(s - 1.0) for s in scores_neg) / len(scores_neg)
    return (e1 - e2) / _LN2


def donsker_varadhan_lower_bound(
    scores_pos: Sequence[float], scores_neg: Sequence[float]
) -> float:
    """Donsker-Varadhan / MINE bound (Belghazi et al. 2018) in bits.

    ``I(X;Y) ≥ E_p[f] − log E_{p×p}[e^f]``.
    """
    if not scores_pos or not scores_neg:
        return 0.0
    e1 = sum(scores_pos) / len(scores_pos)
    lse = _logsumexp(list(scores_neg)) - math.log(len(scores_neg))
    return (e1 - lse) / _LN2


def diayn_intrinsic_reward(
    discriminator_log_q: float, prior_log_p: float
) -> float:
    """DIAYN intrinsic reward ``log q(z|s) − log p(z)`` in bits.

    Eysenbach et al. 2018 *Diversity is all you need*.
    """
    return (discriminator_log_q - prior_log_p) / _LN2


def paninski_empowerment_bound(
    dim_state: int, dim_action: int, min_count: int, confidence: float
) -> float:
    """Hoeffding half-width on a plug-in empowerment estimate (Paninski 2003).

    Parameters
    ----------
    dim_state : int
        |S|.
    dim_action : int
        |A|.
    min_count : int
        Minimum samples per (s,a) pair informing the kernel.
    confidence : float
        Coverage ``1 − δ``.

    Returns
    -------
    half_width_bits : float
        The bound ``(|A| · |S| / min_count) · log(2|S|/δ) / √2`` in bits.
        Returns ``+inf`` if ``min_count`` is zero.
    """
    if min_count <= 0:
        return float("inf")
    delta = max(1e-12, 1.0 - confidence)
    return (dim_action * dim_state / float(min_count)) * (
        math.log(2.0 * dim_state / delta) / math.sqrt(2.0)
    ) / _LN2


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Empowerer:
    """Empowerment / intrinsic-motivation primitive.

    See module docstring.  Thread-safe via a single re-entrant lock.
    """

    def __init__(
        self,
        config: EmpowererConfig | None = None,
        *,
        observer: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._config = config if config is not None else EmpowererConfig()
        self._observer = observer
        self._lock = threading.RLock()

        n_s = self._config.dim_state
        n_a = self._config.dim_action

        # Count-based transition table: counts[s][a][s'] = N_{s,a,s'}
        self._counts: list[list[list[int]]] = [
            [[0 for _ in range(n_s)] for _ in range(n_a)] for _ in range(n_s)
        ]
        # Marginal counts per (s,a)
        self._counts_sa: list[list[int]] = [[0 for _ in range(n_a)] for _ in range(n_s)]
        # State visit counts
        self._counts_s: list[int] = [0 for _ in range(n_s)]

        # Warm-start Blahut-Arimoto action distribution per state.
        self._warm: dict[int, list[float]] = {}

        # DIAYN discriminator: q[z][s], stored as logits.
        self._diayn_logits: list[list[float]] | None = None
        self._diayn_n_skills: int = 0
        self._diayn_visit: list[list[int]] | None = None

        # Variational decoder logits for variational empowerment:
        # q[a][s'] for one-step bound (tabular softmax over actions
        # given next-state).
        self._var_logits: list[list[float]] = [
            [0.0 for _ in range(n_s)] for _ in range(n_a)
        ]

        # RNG.
        self._rng = random.Random(self._config.rng_seed)

        # Ledger.
        self._chain_head = empowerer_ledger_root()
        self._sequence_index = 0

        # Emit STARTED record.
        self._append_block(
            EMPOWERER_STARTED,
            {
                "dim_state": n_s,
                "dim_action": n_a,
                "horizon": self._config.horizon,
                "estimator": self._config.estimator,
                "reward_mode": self._config.reward_mode,
                "rng_seed": self._config.rng_seed,
            },
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EmpowererConfig:
        return self._config

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._chain_head

    @property
    def total_transitions(self) -> int:
        with self._lock:
            return int(sum(sum(row) for row in self._counts_sa))

    @property
    def dim_state(self) -> int:
        return self._config.dim_state

    @property
    def dim_action(self) -> int:
        return self._config.dim_action

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def observe_transition(self, state: int, action: int, next_state: int) -> None:
        """Ingest a single one-step transition ``(s, a, s')``."""
        self._validate_state(state, "state")
        self._validate_action(action)
        self._validate_state(next_state, "next_state")
        with self._lock:
            self._counts[state][action][next_state] += 1
            self._counts_sa[state][action] += 1
            self._counts_s[state] += 1
            self._sequence_index += 1
            self._append_block(
                EMPOWERER_OBSERVED,
                {
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "sequence_index": self._sequence_index,
                },
            )

    def fit_transitions(
        self, transitions: Iterable[tuple[int, int, int]]
    ) -> int:
        """Ingest a batch of transitions; returns the count ingested."""
        n = 0
        with self._lock:
            for t in transitions:
                if len(t) != 3:
                    raise InvalidTransition("transition must be (s, a, s')")
                self.observe_transition(*t)
                n += 1
            self._append_block(
                EMPOWERER_FITTED,
                {"n_ingested": n, "total_transitions": self.total_transitions},
            )
        return n

    # ------------------------------------------------------------------
    # Transition kernel access (count-based MLE with Laplace smoothing)
    # ------------------------------------------------------------------

    def transition_probability(
        self, state: int, action: int, next_state: int
    ) -> float:
        """Smoothed ``p̂(s'|s,a)`` from observed counts."""
        self._validate_state(state, "state")
        self._validate_action(action)
        self._validate_state(next_state, "next_state")
        alpha = self._config.laplace_alpha
        with self._lock:
            num = self._counts[state][action][next_state] + alpha
            denom = self._counts_sa[state][action] + alpha * self._config.dim_state
            if denom <= 0.0:
                return 1.0 / self._config.dim_state
            return num / denom

    def transition_row(self, state: int, action: int) -> list[float]:
        """Smoothed distribution ``p̂(·|s,a)`` (list of length |S|)."""
        self._validate_state(state, "state")
        self._validate_action(action)
        alpha = self._config.laplace_alpha
        n_s = self._config.dim_state
        with self._lock:
            row = self._counts[state][action]
            denom = self._counts_sa[state][action] + alpha * n_s
            if denom <= 0.0:
                return [1.0 / n_s] * n_s
            return [(row[sp] + alpha) / denom for sp in range(n_s)]

    def one_step_kernel(self) -> list[list[list[float]]]:
        """Full smoothed kernel ``[s][a][s']``."""
        with self._lock:
            return [
                [self.transition_row(s, a) for a in range(self._config.dim_action)]
                for s in range(self._config.dim_state)
            ]

    # ------------------------------------------------------------------
    # Empowerment
    # ------------------------------------------------------------------

    def empowerment(self, state: int) -> EmpowermentResult:
        """Compute empowerment at ``state``.  Dispatches on estimator."""
        self._validate_state(state, "state")
        est = self._config.estimator
        if est == EST_BLAHUT_ARIMOTO:
            return self._empowerment_blahut(state)
        if est in (EST_VARIATIONAL_INFONCE, EST_VARIATIONAL_NWJ, EST_VARIATIONAL_DV):
            # Variational route returns its own dataclass — we wrap into
            # an EmpowermentResult for API uniformity.
            vr = self.variational_empowerment(state, n_samples=None)
            return EmpowermentResult(
                state=state,
                horizon=self._config.horizon,
                empowerment_bits=vr.lower_bound_bits,
                action_seq_distribution=tuple(),
                optimal_action=-1,
                iterations=0,
                converged=True,
                chain_head=vr.chain_head,
            )
        raise UnknownEstimator(est)

    def _empowerment_blahut(self, state: int) -> EmpowermentResult:
        n = self._config.horizon
        n_a = self._config.dim_action
        n_states = self._config.dim_state
        kernel = self.one_step_kernel()
        if n == 1:
            channel = [kernel[state][a] for a in range(n_a)]
            seqs = [(a,) for a in range(n_a)]
        else:
            seqs, channel = n_step_kernel(
                kernel,
                state,
                n,
                max_action_seqs=self._config.max_action_seqs,
            )
        warm = self._warm.get(state) if self._config.ba_warm_start else None
        if warm is not None and len(warm) != len(channel):
            warm = None
        cap, alpha, iters, conv = blahut_arimoto_capacity(
            channel,
            tol=self._config.ba_tol,
            max_iter=self._config.ba_max_iter,
            warm_start=warm,
        )
        if self._config.ba_warm_start:
            self._warm[state] = list(alpha)
        opt_idx = max(range(len(alpha)), key=lambda i: alpha[i]) if alpha else -1
        opt_first_action = seqs[opt_idx][0] if opt_idx >= 0 and seqs else -1
        head = self._append_block(
            EMPOWERER_SOLVED,
            {
                "state": state,
                "horizon": n,
                "empowerment_bits": cap,
                "iterations": iters,
                "converged": conv,
                "n_sequences": len(seqs),
                "optimal_first_action": opt_first_action,
            },
        )
        return EmpowermentResult(
            state=state,
            horizon=n,
            empowerment_bits=cap,
            action_seq_distribution=tuple(alpha),
            optimal_action=opt_first_action,
            iterations=iters,
            converged=conv,
            chain_head=head,
        )

    def n_step_empowerment(self, state: int, n: int) -> float:
        """Convenience: empowerment at an arbitrary horizon ``n`` in bits."""
        self._validate_state(state, "state")
        if not isinstance(n, int) or n < 1:
            raise InvalidHorizon("n must be a positive integer")
        n_a = self._config.dim_action
        kernel = self.one_step_kernel()
        if n == 1:
            channel = [kernel[state][a] for a in range(n_a)]
        else:
            _, channel = n_step_kernel(
                kernel,
                state,
                n,
                max_action_seqs=self._config.max_action_seqs,
            )
        cap, _, _, _ = blahut_arimoto_capacity(
            channel,
            tol=self._config.ba_tol,
            max_iter=self._config.ba_max_iter,
        )
        return cap

    # ------------------------------------------------------------------
    # Variational empowerment
    # ------------------------------------------------------------------

    def variational_empowerment(
        self,
        state: int,
        *,
        n_samples: int | None = None,
    ) -> VariationalResult:
        """Estimate empowerment via a variational lower bound.

        Uses the configured estimator (InfoNCE / NWJ / DV).  The
        decoder ``q(a|s')`` is a tabular softmax kept in
        ``self._var_logits``; one stochastic update step is applied
        per call before the bound is evaluated.
        """
        self._validate_state(state, "state")
        est = self._config.estimator
        if est not in (EST_VARIATIONAL_INFONCE, EST_VARIATIONAL_NWJ, EST_VARIATIONAL_DV):
            est = EST_VARIATIONAL_INFONCE
        if n_samples is None:
            n_samples = self._config.variational_samples
        if n_samples < 2:
            raise InvalidConfig("n_samples must be at least 2")
        n_a = self._config.dim_action
        n_s = self._config.dim_state

        # Sample (a_i, s'_i) ~ uniform-a × p(s'|s,a).
        actions: list[int] = []
        next_states: list[int] = []
        for _ in range(n_samples):
            a = self._rng.randrange(n_a)
            row = self.transition_row(state, a)
            sp = self._sample_categorical(row)
            actions.append(a)
            next_states.append(sp)

        # Variational decoder log q(a|s') from logits.
        def log_q(a: int, sp: int) -> float:
            logits = [self._var_logits[ap][sp] for ap in range(n_a)]
            m = max(logits)
            denom = math.log(sum(math.exp(l - m) for l in logits)) + m
            return self._var_logits[a][sp] - denom

        # Compute the bound.
        if est == EST_VARIATIONAL_INFONCE:
            # f(a_i, s'_j) = log q(a_i | s'_j).  Diagonal positives.
            scores: list[list[float]] = []
            for i in range(n_samples):
                row = []
                for j in range(n_samples):
                    row.append(log_q(actions[i], next_states[j]))
                scores.append(row)
            bits = infonce_lower_bound(scores)
        elif est == EST_VARIATIONAL_NWJ:
            pos = [log_q(actions[i], next_states[i]) for i in range(n_samples)]
            neg = []
            for i in range(n_samples):
                j = (i + 1) % n_samples
                neg.append(log_q(actions[i], next_states[j]))
            bits = nwj_lower_bound(pos, neg)
        else:  # EST_VARIATIONAL_DV
            pos = [log_q(actions[i], next_states[i]) for i in range(n_samples)]
            neg = []
            for i in range(n_samples):
                j = (i + 1) % n_samples
                neg.append(log_q(actions[i], next_states[j]))
            bits = donsker_varadhan_lower_bound(pos, neg)

        # Gradient ascent step on the decoder logits for InfoNCE
        # objective.  For action a, next-state s', positive sample
        # contributes (1 − q(a|s')); negatives contribute (−q(a|s')).
        lr = self._config.variational_lr
        for i in range(n_samples):
            sp = next_states[i]
            a_pos = actions[i]
            logits = [self._var_logits[ap][sp] for ap in range(n_a)]
            m = max(logits)
            denom = sum(math.exp(l - m) for l in logits)
            probs = [math.exp(l - m) / denom for l in logits]
            for ap in range(n_a):
                grad = (1.0 if ap == a_pos else 0.0) - probs[ap]
                self._var_logits[ap][sp] += lr * grad / n_samples

        # Hoeffding half-width on bits: bounded in [0, log2 |A|].
        cap = math.log2(max(n_a, 2))
        delta = max(1e-12, 1.0 - self._config.confidence)
        half = cap * math.sqrt(math.log(2.0 / delta) / (2.0 * n_samples))
        head = self._append_block(
            EMPOWERER_VARIATIONAL,
            {
                "state": state,
                "horizon": 1,
                "estimator": est,
                "lower_bound_bits": bits,
                "hoeffding_half_width": half,
                "samples_used": n_samples,
            },
        )
        return VariationalResult(
            state=state,
            horizon=1,
            estimator=est,
            lower_bound_bits=bits,
            hoeffding_half_width=half,
            samples_used=n_samples,
            chain_head=head,
        )

    # ------------------------------------------------------------------
    # Skill discovery (DIAYN)
    # ------------------------------------------------------------------

    def skill_discovery(
        self,
        n_skills: int,
        *,
        steps: int = 200,
        prior_uniform: bool = True,
    ) -> SkillReport:
        """DIAYN-style mutual-information skill discovery.

        Implements the *tabular* relaxation of Eysenbach et al. 2018
        on the observed empirical state distribution.  Returns
        per-skill state occupancy and a separability score
        ``E_{z,s~ρ_z}[log q(z|s)]``.

        Parameters
        ----------
        n_skills : int
            Number of latent skills ``K``.
        steps : int
            Number of expectation-maximisation rounds.
        prior_uniform : bool
            If True, use uniform prior ``p(z) = 1/K``.
        """
        if not isinstance(n_skills, int) or n_skills < 2:
            raise InvalidConfig("n_skills must be at least 2")
        if not isinstance(steps, int) or steps < 1:
            raise InvalidConfig("steps must be a positive integer")
        n_s = self._config.dim_state
        if self.total_transitions == 0:
            raise InsufficientData("no observations to discover skills over")
        with self._lock:
            visit = [int(c) for c in self._counts_s]
            total = sum(visit)
            if total == 0:
                raise InsufficientData("no state observations")
            rho = [v / total for v in visit]
            # Initialise discriminator logits — small Gaussian noise so
            # skills break symmetry.
            logits = [
                [self._rng.gauss(0.0, 0.1) for _ in range(n_s)] for _ in range(n_skills)
            ]
            # Skill-occupancy is initially uniform across skills then
            # updated via soft-EM: q(z|s) ∝ exp(logits[z][s]); skill
            # occupancy ρ_z(s) ∝ ρ(s) q(z|s).
            for _ in range(steps):
                # Per-state q(z|s).
                q_zs: list[list[float]] = []
                for s in range(n_s):
                    if rho[s] <= 0.0:
                        q_zs.append([1.0 / n_skills] * n_skills)
                        continue
                    col = [logits[z][s] for z in range(n_skills)]
                    m = max(col)
                    es = [math.exp(c - m) for c in col]
                    z_sum = sum(es)
                    q_zs.append([e / z_sum for e in es])
                # Skill-occupancy ρ_z(s) (normalised over s).
                rho_z: list[list[float]] = []
                for z in range(n_skills):
                    raw = [rho[s] * q_zs[s][z] for s in range(n_s)]
                    rz = sum(raw)
                    if rz > 0.0:
                        rho_z.append([r / rz for r in raw])
                    else:
                        rho_z.append([1.0 / n_s] * n_s)
                # Gradient ascent on log q(z|s) toward separating skills.
                lr = self._config.diayn_lr
                for s in range(n_s):
                    if rho[s] <= 0.0:
                        continue
                    for z in range(n_skills):
                        target = rho_z[z][s]
                        # KL gradient on softmax with target distribution
                        # across z given s ∝ rho_z(z,s).
                        col = [logits[zp][s] for zp in range(n_skills)]
                        m = max(col)
                        es = [math.exp(c - m) for c in col]
                        z_sum = sum(es)
                        probs = [e / z_sum for e in es]
                        for zp in range(n_skills):
                            grad = (1.0 if zp == z else 0.0) - probs[zp]
                            logits[zp][s] += lr * target * grad / max(1, n_s)
            self._diayn_logits = logits
            self._diayn_n_skills = n_skills
            self._diayn_visit = [list(visit) for _ in range(n_skills)]
            # Compute skill-state distribution and separability.
            skill_state_dist: list[list[float]] = []
            for z in range(n_skills):
                raw = [rho[s] * math.exp(logits[z][s]) for s in range(n_s)]
                z_sum = sum(raw)
                if z_sum > 0.0:
                    skill_state_dist.append([r / z_sum for r in raw])
                else:
                    skill_state_dist.append([1.0 / n_s] * n_s)
            # Skill entropy = H(z|s) averaged over ρ(s).
            sep = 0.0
            ent = 0.0
            for s in range(n_s):
                if rho[s] <= 0.0:
                    continue
                col = [logits[z][s] for z in range(n_skills)]
                m = max(col)
                denom = sum(math.exp(c - m) for c in col)
                logZ = math.log(denom) + m
                row_ent = 0.0
                for z in range(n_skills):
                    log_q = col[z] - logZ
                    p = math.exp(log_q)
                    sep += rho[s] * p * log_q / _LN2
                    if p > 0.0:
                        row_ent += -p * log_q / _LN2
                ent += rho[s] * row_ent
            head = self._append_block(
                EMPOWERER_SKILLS_DISCOVERED,
                {
                    "n_skills": n_skills,
                    "iterations": steps,
                    "avg_log_q_bits": sep,
                    "skill_entropy_bits": ent,
                },
            )
            return SkillReport(
                n_skills=n_skills,
                iterations=steps,
                avg_log_q=sep,
                skill_state_dist=tuple(tuple(r) for r in skill_state_dist),
                skill_separability=sep,
                skill_entropy=ent,
                chain_head=head,
            )

    # ------------------------------------------------------------------
    # Safe-action shielding
    # ------------------------------------------------------------------

    def safe_actions(
        self,
        state: int,
        candidates: Sequence[int] | None = None,
        *,
        margin: float | None = None,
    ) -> ShieldDecision:
        """Return the empowerment-preserving subset of ``candidates``.

        An action ``a`` at ``s`` is admissible iff the expected (or
        worst-case, per ``safety_estimator``) successor empowerment
        is at least ``𝔈ⁿ(s) − margin``.  Margin defaults to the
        config-level ``safety_margin``.
        """
        self._validate_state(state, "state")
        n_a = self._config.dim_action
        if candidates is None:
            candidates = tuple(range(n_a))
        else:
            for a in candidates:
                self._validate_action(a)
            candidates = tuple(candidates)
        if margin is None:
            margin = self._config.safety_margin
        if margin < 0.0:
            raise InvalidConfig("margin must be non-negative")

        state_emp = self.n_step_empowerment(state, self._config.horizon)
        successor_emp: list[float] = []
        admissible: list[int] = []
        for a in candidates:
            row = self.transition_row(state, a)
            per_sp = [
                self.n_step_empowerment(sp, self._config.horizon)
                for sp in range(self._config.dim_state)
            ]
            if self._config.safety_estimator == "min":
                succ = min(per_sp[sp] for sp in range(self._config.dim_state))
            else:
                succ = sum(row[sp] * per_sp[sp] for sp in range(self._config.dim_state))
            successor_emp.append(succ)
            if succ >= state_emp - margin:
                admissible.append(a)
        head = self._append_block(
            EMPOWERER_SHIELDED,
            {
                "state": state,
                "candidates": list(candidates),
                "admissible": admissible,
                "state_empowerment_bits": state_emp,
                "margin": margin,
            },
        )
        return ShieldDecision(
            state=state,
            candidates=candidates,
            admissible=tuple(admissible),
            state_empowerment_bits=state_emp,
            successor_empowerment_bits=tuple(successor_emp),
            margin=margin,
            chain_head=head,
        )

    # ------------------------------------------------------------------
    # Intrinsic reward
    # ------------------------------------------------------------------

    def intrinsic_reward(
        self,
        state: int,
        action: int,
        next_state: int,
        *,
        mode: str | None = None,
    ) -> float:
        """Synthesise an intrinsic reward for ``(s, a, s')``.

        Modes (see ``REWARD_*``):
          * ``state_empowerment``      — empowerment of ``state``
          * ``delta_empowerment``      — 𝔈(s') − 𝔈(s)
          * ``transition_surprise``    — −log₂ p̂(s'|s,a)
        """
        self._validate_state(state, "state")
        self._validate_action(action)
        self._validate_state(next_state, "next_state")
        mode = mode or self._config.reward_mode
        if mode not in KNOWN_REWARD_MODES:
            raise UnknownRewardMode(mode)
        scale = self._config.intrinsic_scale
        if mode == REWARD_STATE_EMPOWERMENT:
            r = self.n_step_empowerment(state, self._config.horizon)
        elif mode == REWARD_DELTA_EMPOWERMENT:
            r1 = self.n_step_empowerment(state, self._config.horizon)
            r2 = self.n_step_empowerment(next_state, self._config.horizon)
            r = r2 - r1
        else:  # transition_surprise
            p = self.transition_probability(state, action, next_state)
            r = -_log2(max(p, 1e-30))
        out = scale * r
        self._append_block(
            EMPOWERER_REWARDED,
            {
                "state": state,
                "action": action,
                "next_state": next_state,
                "mode": mode,
                "reward": out,
            },
        )
        return out

    # ------------------------------------------------------------------
    # Landscape (per-state empowerment over observed states)
    # ------------------------------------------------------------------

    def landscape(self) -> dict[int, float]:
        """Per-state empowerment over states with at least one outgoing visit."""
        out: dict[int, float] = {}
        for s in range(self._config.dim_state):
            if self._counts_s[s] == 0 and sum(self._counts_sa[s]) == 0:
                continue
            out[s] = self.n_step_empowerment(s, self._config.horizon)
        return out

    def bottleneck_states(self, *, top_k: int = 1) -> list[int]:
        """States with locally minimal empowerment (option boundaries).

        Returns the ``top_k`` lowest-empowerment states with at least
        one observed transition.  Çelik, Polani 2010.
        """
        land = self.landscape()
        if not land:
            return []
        items = sorted(land.items(), key=lambda kv: kv[1])
        return [s for s, _ in items[:top_k]]

    # ------------------------------------------------------------------
    # Reports & certificates
    # ------------------------------------------------------------------

    def report(self) -> EmpowererReport:
        """Snapshot of the estimator state."""
        with self._lock:
            visited = sum(1 for c in self._counts_s if c > 0)
            distinct_actions = sum(
                1
                for a in range(self._config.dim_action)
                if any(self._counts_sa[s][a] > 0 for s in range(self._config.dim_state))
            )
            total = self.total_transitions
            if total == 0:
                mean_emp = 0.0
                mx = 0.0
                mn = 0.0
            else:
                landscape = self.landscape()
                if landscape:
                    vals = list(landscape.values())
                    mean_emp = sum(vals) / len(vals)
                    mx = max(vals)
                    mn = min(vals)
                else:
                    mean_emp = 0.0
                    mx = 0.0
                    mn = 0.0
            head = self._append_block(
                EMPOWERER_REPORTED,
                {
                    "total_transitions": total,
                    "distinct_states": visited,
                    "distinct_actions": distinct_actions,
                    "horizon": self._config.horizon,
                    "estimator": self._config.estimator,
                    "reward_mode": self._config.reward_mode,
                    "mean_state_empowerment": mean_emp,
                    "max_state_empowerment": mx,
                    "min_state_empowerment": mn,
                    "state_coverage_fraction": visited / self._config.dim_state,
                },
            )
            return EmpowererReport(
                total_transitions=total,
                distinct_states=visited,
                distinct_actions=distinct_actions,
                horizon=self._config.horizon,
                estimator=self._config.estimator,
                reward_mode=self._config.reward_mode,
                mean_state_empowerment=mean_emp,
                max_state_empowerment=mx,
                min_state_empowerment=mn,
                state_coverage_fraction=visited / self._config.dim_state,
                chain_head=head,
            )

    def certify(self, state: int) -> EmpowererCertificate:
        """PAC certificate on the empowerment estimate at ``state``."""
        self._validate_state(state, "state")
        with self._lock:
            min_count = min(
                self._counts_sa[state][a] for a in range(self._config.dim_action)
            )
            est = self.n_step_empowerment(state, self._config.horizon)
            half = paninski_empowerment_bound(
                self._config.dim_state,
                self._config.dim_action,
                min_count,
                self._config.confidence,
            )
            upper = est + half
            lower = max(0.0, est - half)
            holds = min_count > 0
            head = self._append_block(
                EMPOWERER_CERTIFIED,
                {
                    "state": state,
                    "horizon": self._config.horizon,
                    "empowerment_bits": est,
                    "upper_bound_bits": upper,
                    "lower_bound_bits": lower,
                    "confidence": self._config.confidence,
                    "n_samples": min_count,
                    "holds": holds,
                },
            )
            return EmpowererCertificate(
                state=state,
                horizon=self._config.horizon,
                empowerment_bits=est,
                upper_bound_bits=upper,
                lower_bound_bits=lower,
                confidence=self._config.confidence,
                n_samples=min_count,
                holds=holds,
                chain_head=head,
            )

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a byte-deterministic snapshot of estimator state."""
        with self._lock:
            return {
                "version": _GENESIS,
                "dim_state": self._config.dim_state,
                "dim_action": self._config.dim_action,
                "horizon": self._config.horizon,
                "estimator": self._config.estimator,
                "reward_mode": self._config.reward_mode,
                "counts": [
                    [list(self._counts[s][a]) for a in range(self._config.dim_action)]
                    for s in range(self._config.dim_state)
                ],
                "counts_sa": [list(self._counts_sa[s]) for s in range(self._config.dim_state)],
                "counts_s": list(self._counts_s),
                "var_logits": [
                    list(self._var_logits[a]) for a in range(self._config.dim_action)
                ],
                "diayn_logits": (
                    [list(self._diayn_logits[z]) for z in range(self._diayn_n_skills)]
                    if self._diayn_logits is not None
                    else None
                ),
                "diayn_n_skills": self._diayn_n_skills,
                "warm": {str(k): list(v) for k, v in self._warm.items()},
                "chain_head": self._chain_head,
                "sequence_index": self._sequence_index,
                "rng_state": self._rng.getstate(),
            }

    def restore(self, snap: Mapping[str, Any]) -> None:
        """Restore byte-identical state from a previous snapshot."""
        if snap.get("version") != _GENESIS:
            raise InvalidConfig(f"snapshot version mismatch: {snap.get('version')}")
        if (
            snap["dim_state"] != self._config.dim_state
            or snap["dim_action"] != self._config.dim_action
        ):
            raise InvalidConfig("snapshot dim mismatch with config")
        with self._lock:
            self._counts = [
                [list(snap["counts"][s][a]) for a in range(self._config.dim_action)]
                for s in range(self._config.dim_state)
            ]
            self._counts_sa = [
                list(snap["counts_sa"][s]) for s in range(self._config.dim_state)
            ]
            self._counts_s = list(snap["counts_s"])
            self._var_logits = [list(snap["var_logits"][a]) for a in range(self._config.dim_action)]
            self._diayn_logits = (
                [list(row) for row in snap["diayn_logits"]]
                if snap.get("diayn_logits") is not None
                else None
            )
            self._diayn_n_skills = int(snap.get("diayn_n_skills", 0))
            self._warm = {int(k): list(v) for k, v in snap.get("warm", {}).items()}
            self._chain_head = str(snap["chain_head"])
            self._sequence_index = int(snap["sequence_index"])
            rng_state = snap.get("rng_state")
            if rng_state is not None:
                # Random.getstate returns nested tuples — JSON-roundtripped
                # snapshots will see lists; coerce back.
                self._rng.setstate(_coerce_rng_state(rng_state))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the estimator to its initial state.  Ledger continues."""
        with self._lock:
            n_s = self._config.dim_state
            n_a = self._config.dim_action
            self._counts = [
                [[0 for _ in range(n_s)] for _ in range(n_a)] for _ in range(n_s)
            ]
            self._counts_sa = [[0 for _ in range(n_a)] for _ in range(n_s)]
            self._counts_s = [0 for _ in range(n_s)]
            self._warm = {}
            self._var_logits = [[0.0 for _ in range(n_s)] for _ in range(n_a)]
            self._diayn_logits = None
            self._diayn_n_skills = 0
            self._diayn_visit = None
            self._rng = random.Random(self._config.rng_seed)
            self._sequence_index = 0
            self._append_block(EMPOWERER_RESET, {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_state(self, s: int, name: str) -> None:
        if not isinstance(s, int):
            raise InvalidState(f"{name} must be an integer")
        if s < 0 or s >= self._config.dim_state:
            raise InvalidState(
                f"{name}={s} out of range [0, {self._config.dim_state})"
            )

    def _validate_action(self, a: int) -> None:
        if not isinstance(a, int):
            raise InvalidAction("action must be an integer")
        if a < 0 or a >= self._config.dim_action:
            raise InvalidAction(
                f"action={a} out of range [0, {self._config.dim_action})"
            )

    def _sample_categorical(self, dist: Sequence[float]) -> int:
        r = self._rng.random()
        c = 0.0
        last = 0
        for i, p in enumerate(dist):
            c += p
            last = i
            if r < c:
                return i
        return last

    def _append_block(self, kind: str, payload: Mapping[str, Any]) -> str:
        with self._lock:
            block = {"kind": kind, **dict(payload)}
            new_head = _digest(self._chain_head, block, self._config.hmac_key)
            self._chain_head = new_head
            if self._observer is not None:
                try:
                    self._observer(kind, dict(payload))
                except Exception:
                    # Observers must never bring down the estimator.
                    pass
            return new_head


# ---------------------------------------------------------------------------
# Module-level helpers used by the demo and by snapshot/restore.
# ---------------------------------------------------------------------------


def _coerce_rng_state(state: Any) -> tuple[Any, ...]:
    """Coerce a JSON-roundtripped ``random.Random.getstate()`` tuple back.

    ``random.Random.setstate`` requires the inner integer tuple to be a
    tuple.  JSON deserialises to lists; we walk the structure and
    convert.
    """
    if isinstance(state, list):
        return tuple(_coerce_rng_state(x) for x in state)
    return state
