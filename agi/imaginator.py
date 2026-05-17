r"""Imaginator — learned-world-model rollouts as a runtime primitive.

Every primitive in this runtime that *plans* eventually faces the same
problem: it must reason about what happens *next* before it commits
real cost.  ``Searcher`` performs anytime certified tree search over
*caller-supplied* successor enumerations.  ``Active Inference`` reduces
expected free energy over a *caller-supplied* generative model.
``Planner`` compiles a *caller-supplied* PDDL operator schema.  In every
case the user has to hand over a dynamics function, then trust it.

``Imaginator`` is the primitive that *learns* that dynamics function
from observed transitions, *bounds* the error of every imagined
trajectory, and emits a tamper-evident receipt for every imagined
rollout — the **model-based-RL inner loop**, generalised to a primitive
the coordination engine can register, drive, and audit.

The pitch reduced to a runtime call::

    imag = Imaginator(ImaginatorConfig(family="categorical"))
    imag.register_env("supply-chain", states=("ok","stockout"),
                      actions=("ship","wait"))

    for s, a, s_next, r in observed_transitions():
        imag.observe("supply-chain", s, a, s_next, r)

    roll = imag.imagine("supply-chain", state="ok",
                        policy=lambda s: "ship",
                        horizon=8, samples=64)
    print(roll.expected_return, roll.value_lcb, roll.value_ucb)

    plan = imag.value_iteration("supply-chain", horizon=12, discount=0.95)
    pac  = imag.pac_value_bound("supply-chain", policy=plan.policy, delta=0.05)

    # Replay-verifiable receipt — AttestationLedger.verify(roll.fingerprint_hash)

What this primitive ships
-------------------------

  * **Two dynamics families (stdlib, no NumPy):**

    * ``"categorical"`` — discrete-state, discrete-action MDP with
      Dirichlet-multinomial conjugate posterior on the per-(state,
      action) successor distribution and Gamma-Normal conjugate
      posterior on the per-(state, action) reward.  Closed-form
      Bayesian updates; the posterior predictive over the next state
      is the Dirichlet-Multinomial with closed-form mean
      ``α[s'] / ∑_{s'} α[s']``.  This is the **Bayesian R-MAX** family
      (Strehl-Littman-Wiewiora 2009; Auer-Jaksch-Ortner 2010 *UCRL2*),
      with the optimism-under-uncertainty exploration policy delivered
      via Thompson-sampled transition matrices.

    * ``"linear_gaussian"`` — continuous-state linear dynamics
      ``s_{t+1} = A s_t + B a_t + ε_t``,  ``ε_t ~ N(0, Σ)``  with a
      matrix-normal-inverse-Wishart conjugate prior on ``[A | B]`` and
      ``Σ``.  Closed-form online updates of the prior sufficient
      statistics; closed-form predictive marginal ``N(μ, σ²)`` at every
      forecast horizon.  This is the **PILCO** family
      (Deisenroth-Rasmussen 2011 *PILCO: A Model-Based and Data-
      Efficient Approach to Policy Search*) with the closed-form
      moment-matching propagation that PILCO required a GP for, here
      delivered by a Bayesian linear model whose epistemic uncertainty
      *closes in expectation* as ``n → ∞``.

  * **Imagined rollouts** ``imagine(env, state, policy, horizon, samples)``
    — Monte Carlo trajectories drawn from the *posterior predictive*
    over the dynamics.  Per-step state distribution is built from
    posterior-sampled transitions (Thompson sampling for categorical;
    matrix-normal-inverse-Wishart sampling for linear-Gaussian).  The
    returned ``Rollout`` carries:

      - ``expected_return`` — Monte Carlo mean of discounted return.
      - ``value_lcb`` / ``value_ucb`` — Maurer-Pontil 2009 empirical
        Bernstein confidence interval on ``E[discounted return]``,
        sharper than Hoeffding 1963 in low-variance regimes.
      - ``hrms_lcb`` / ``hrms_ucb`` — Howard-Ramdas-McAuliffe-Sekhon
        2021 anytime-valid confidence sequence, so a coordinator can
        keep drawing rollouts and reading the bound without paying a
        union-bound tax.
      - ``return_quantiles`` — empirical CDF of the simulated return,
        from which a downstream ``Quantilizer`` reads the
        ``q``-quantile threshold for safety-bounded deployment.
      - ``trajectory_quantiles`` — at every horizon ``h``, the
        2.5 / 50 / 97.5 quantiles of the simulated next state.

  * **Value iteration** ``value_iteration(env, horizon, discount)`` —
    closed-form dynamic-programming planner on the posterior-mean
    transition / reward; returns the deterministic greedy policy plus
    the per-state value function ``V*(s)`` under the planning horizon.
    Converges in ``≤ horizon`` sweeps; ε-optimal at termination.

  * **Posterior policy sampling** ``thompson_policy(env, horizon)`` —
    draw one transition matrix from the Dirichlet posterior, return
    the value-iteration policy for that draw.  This is the **PSRL**
    (Posterior Sampling for Reinforcement Learning) algorithm of
    Strens 2000 / Osband-Russo-Van Roy 2013; under it the Bayesian
    regret ``Reg = O(τ √(SAT log T))`` where τ is the diameter.

  * **PAC value bound** ``pac_value_bound(env, policy, delta)`` —
    closed-form simulation-lemma upper bound on the policy-value
    estimation error.  Composes Kearns-Singh 2002 *Near-Optimal RL in
    Polynomial Time* (the simulation lemma) with a per-(s, a)
    Hoeffding bound on the transition posterior to deliver a PAC
    statement of the form *"with probability ≥ 1 − δ, |V̂π − V*π| ≤ ε"*
    where ε is a closed-form function of ``δ``, the discount factor,
    the horizon, and ``min_n(s, a)``.

  * **PILCO-style moment propagation** ``moment_rollout(env, state,
    policy, horizon)`` — closed-form linear-Gaussian propagation of
    ``N(μ_0, Σ_0)`` through ``h`` steps under a posterior-mean
    transition; returns the per-horizon ``(μ_h, Σ_h)``.  No Monte
    Carlo; analytic; differentiable.

  * **Bayesian model averaging** ``bayes_average_value(env, policy,
    horizon, samples, n_models)`` — average value across ``n_models``
    independent posterior-sampled dynamics; minimises log-loss in
    expectation (Madigan-Raftery 1994) and reduces the model-
    uncertainty contribution to the posterior predictive variance.

  * **Identifiability report** ``identifiability_report(env)`` —
    flags (state, action) pairs with ``< min_observations`` data and
    quantifies the effective Dirichlet concentration; matches the
    Cao-Cohen-Szepesvári 2021 notion of *behaviourally
    indistinguishable* dynamics under the observation distribution.

  * **PIT calibration certificate** ``pit_calibration(env)`` — under a
    correct posterior predictive, the probability integral transform
    of one-step-ahead rewards is uniform on ``[0, 1]``.  Imaginator
    runs a one-sample Kolmogorov-Smirnov test (Massey 1951) on the
    held-out PIT and returns the p-value as a real-time
    calibration signal.

  * **Drift detection** — exposes per-(s, a) running prediction
    log-loss as a martingale-difference under correct dynamics; a
    Maurer-Pontil empirical-Bernstein CUSUM on this stream is the
    real-time signal that the world has moved relative to the learned
    model.  Composes verbatim with ``DriftSentinel``.

  * **Tamper-evident SHA-256 fingerprint chain** with optional
    HMAC-SHA-256 (genesis seed ``"agi.imaginator.v1\x00" + secret_key``)
    over every ``register / observe / imagine / value / certify``
    event.  ``AttestationLedger.verify`` replays the imagined
    trajectory byte-for-byte from the same observation stream + RNG
    seed.

  * **Thread-safe at the API surface** — a single re-entrant lock
    guards every mutation of per-environment state.  Read-only methods
    (``imagine``, ``value_iteration``, ``identifiability_report``)
    take the same lock to snapshot a consistent view but never fail.

  * **Pure stdlib.**  No NumPy, no Torch, no SciPy.  Runs inside a
    sandboxed coordinator, inside a CI worker, inside a Lambda
    function with a 256 MB memory cap.

Mathematical and algorithmic roots
----------------------------------

  * **Sutton, R. (1990).**  *Dyna: An Integrated Architecture for
    Learning, Planning, and Reacting.*  Real and imagined transitions
    update the same value function; one observation buys both a
    planning step and a learning step.

  * **Kearns, M. & Singh, S. (2002).**  *Near-Optimal Reinforcement
    Learning in Polynomial Time.*  The **simulation lemma**:
    ``|V^π_M̂ − V^π_M| ≤  (γ / (1−γ)²) · ε``  whenever the learned
    model ``M̂`` is ``ε``-accurate in transition + reward.  This is the
    backbone of every PAC bound Imaginator emits.

  * **Strehl, A. L., Littman, M. L. & Wiewiora, E. (2009).**  *PAC
    model-free reinforcement learning.*  Bayesian R-MAX with
    Dirichlet-multinomial transitions; the sample-complexity bound
    ``O((SA / ε²(1−γ)⁴) · log(SAδ⁻¹))`` is exactly what
    ``required_samples_for_pac`` reports.

  * **Strens, M. (2000).** *A Bayesian Framework for Reinforcement
    Learning.*  **Posterior Sampling for RL** — draw one transition
    matrix from the posterior, plan optimally against it, act, repeat.
    The Bayesian-regret optimality of this scheme was established by
    Osband-Russo-Van Roy 2013 *(More) Efficient Reinforcement Learning
    via Posterior Sampling.*

  * **Auer, P., Jaksch, T. & Ortner, R. (2010).**  *Near-optimal
    regret bounds for reinforcement learning.*  **UCRL2** — the
    optimism-under-uncertainty alternative to PSRL; the same
    Dirichlet concentration that drives PSRL's posterior also gives
    UCRL2's confidence radius.

  * **Deisenroth, M. P. & Rasmussen, C. E. (2011).**  *PILCO: A
    Model-Based and Data-Efficient Approach to Policy Search.*  The
    moment-matching closed-form rollout that delivers
    ``moment_rollout`` for the linear-Gaussian family — in PILCO
    delivered via a GP, here via a Bayesian linear model with the
    same matrix-normal-inverse-Wishart conjugate structure.

  * **Janner, M., Fu, J., Zhang, M. & Levine, S. (2019).**  *When to
    Trust Your Model: Model-Based Policy Optimization.*  The
    short-horizon-rollout argument: imagined trajectories are
    accurate at small ``h``, biased at large ``h``; Imaginator's
    ``trajectory_quantiles`` expose exactly the growing predictive
    variance that justifies the horizon cap.

  * **Hafner, D., Lillicrap, T., Ba, J. & Norouzi, M. (2020).**
    *Dream to Control: Learning Behaviors by Latent Imagination.*
    The general DreamerV3 architecture treats imagined trajectories
    as the optimisation surface for the policy; Imaginator delivers
    the imagined trajectories with calibrated uncertainty bounds the
    coordinator can read before committing.

  * **Madigan, D. & Raftery, A. E. (1994).**  *Model selection and
    accounting for model uncertainty in graphical models using
    Occam's window.*  Bayesian Model Averaging — the asymptotic
    log-loss minimiser; ``bayes_average_value`` is the BMA value
    estimate.

  * **Cao, Y., Cohen, A. & Szepesvári, C. (2021).**  *Identifiability
    in Inverse Reinforcement Learning.*  The behavioural-
    equivalence notion adapted here for transition dynamics — two
    transition matrices that agree on the observed (s, a) pairs are
    indistinguishable under the observation distribution.

  * **Massey, F. J. (1951).**  *The Kolmogorov-Smirnov Test for
    Goodness of Fit.*  The one-sample KS test under H₀: PIT(rewards)
    ~ Uniform(0, 1).  Implemented from the asymptotic distribution
    via Marsaglia-Tsang-Wang 2003 series.

  * **Maurer, A. & Pontil, M. (2009).**  *Empirical Bernstein bounds
    and sample-variance penalization.*  The closed-form
    empirical-Bernstein LCB on Monte Carlo return — sharper than
    Hoeffding 1963 in low-variance regimes; backbone of
    ``value_lcb`` / ``value_ucb``.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. S.
    (2021).**  *Time-uniform, nonparametric, nonasymptotic confidence
    sequences.*  Anytime-valid confidence sequence on bounded
    random variables; backbone of ``hrms_lcb`` / ``hrms_ucb``.

  * **McAllester, D. (1999).** *PAC-Bayesian Model Averaging.* The
    closed-form upper bound on expected loss in terms of the
    empirical loss + KL-to-prior penalty / sample size; used in
    ``pac_bayes_value_bound``.

Composes with the rest of the runtime
-------------------------------------

  * ``Searcher`` — Imaginator is *the* successor enumerator a
    Searcher accepts.  Searcher's tree search runs over imagined
    transitions; the cost-of-evaluation that Searcher trades off is
    Imaginator's per-sample rollout time.

  * ``ActiveInferencer`` — Imaginator supplies the generative model
    (state-transition + observation likelihood) that the Active
    Inference primitive's expected-free-energy minimisation requires.

  * ``Quantilizer`` — Imaginator's ``return_quantiles`` *is* the
    distribution the Quantilizer thresholds on.  The two together
    deliver: "deploy the policy whose imagined return is in the top
    ``q`` quantile of the posterior over dynamics".

  * ``Distiller`` — distil the value-iteration policy returned by
    ``Imaginator.value_iteration`` into an amortised neural / linear
    policy that runs at inference time.

  * ``Planner`` — Imaginator's posterior-mean transition matrix is
    a PDDL-compilable operator schema; Planner reads the operator
    schema and solves SAT with the deterministic mode of the
    Imaginator's MAP transitions.

  * ``DriftSentinel`` — per-step log-loss of one-step predictions is
    a martingale-difference under correct dynamics; DriftSentinel
    runs a CUSUM and flags world drift in real time.

  * ``Bandit`` / ``BayesOpt`` — Thompson-sampled value from Imaginator
    is a cheap proxy oracle for hyperparameter / arm selection.

  * ``Curator`` — Imaginator's identifiability report identifies
    (state, action) pairs that are still under-observed; Curator
    targets those pairs in its next curriculum batch.

  * ``AttestationLedger`` — every register / observe / imagine /
    value / certify event chain-hashes into the ledger; a compliance
    officer replays the imagined trajectory byte-for-byte from the
    observation stream + RNG seed.

  * ``Coordinator`` — every Goal whose execution requires reasoning
    over future world states routes through Imaginator.  The
    coordination engine no longer has to hand-write the dynamics
    function; it observes a few real transitions, registers them
    with Imaginator, and queries imagined value with calibrated
    uncertainty bounds the compliance officer can sign before action.

Investor framing
----------------

Imaginator is the **runtime's imagination kernel**.  Every prior
primitive in this runtime processes the present.  Imaginator
processes the future, with calibrated uncertainty, with a receipt the
compliance officer can sign before money moves.  This is the line
between *"we run AI"* and *"we run AI that reasons about consequences
before committing them."*  Pair with ``Quantilizer`` for safety-
bounded deployment, ``Searcher`` for tree-search over imagined
futures, and ``AttestationLedger`` for cryptographic replay — the
**model-based-RL inner loop**, delivered as a runtime primitive a
coordination engine can drive.

What it deliberately doesn't claim
----------------------------------

  * Not a frontier dynamics learner — no neural networks, no
    transformer world model.  The two model families shipped
    (Dirichlet-multinomial categorical and matrix-normal-inverse-
    Wishart linear-Gaussian) are the **conjugate** ones that admit
    closed-form Bayesian updates and provable PAC bounds.

  * Not online closed-loop control — Imaginator imagines on demand,
    but the coordinator decides when to act.  Composition with a
    real-time controller is the caller's responsibility.

  * Not partial-observability — the categorical family assumes the
    observed state *is* the latent state.  Composition with
    ``Filterer`` for the POMDP case is the recommended pattern; the
    linear-Gaussian family natively supports observation noise but
    the latent-state dimensionality must match the action space.
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
    "IMAGINATOR_STARTED",
    "IMAGINATOR_REGISTERED",
    "IMAGINATOR_REMOVED",
    "IMAGINATOR_OBSERVED",
    "IMAGINATOR_IMAGINED",
    "IMAGINATOR_PLANNED",
    "IMAGINATOR_CERTIFIED",
    "IMAGINATOR_CLEARED",
    # Family names
    "FAMILY_CATEGORICAL",
    "FAMILY_LINEAR_GAUSSIAN",
    "KNOWN_FAMILIES",
    # Sampling selectors
    "SAMPLE_POSTERIOR_MEAN",
    "SAMPLE_THOMPSON",
    "SAMPLE_BAYES_AVG",
    "KNOWN_SAMPLE_METHODS",
    # Errors
    "ImaginatorError",
    "InvalidConfig",
    "InvalidEnv",
    "InvalidObservation",
    "InsufficientData",
    "UnknownEnv",
    # Dataclasses
    "ImaginatorConfig",
    "EnvSpec",
    "Transition",
    "Rollout",
    "ValueIterationResult",
    "PACValueBound",
    "IdentifiabilityReport",
    "PITCalibrationReport",
    # Main class
    "Imaginator",
    # Helper functions
    "ledger_root",
    "hoeffding_half_width",
    "empirical_bernstein_half_width",
    "hrms_half_width",
    "ks_pvalue",
    "dirichlet_mean",
    "dirichlet_sample",
    "softmax",
]

# ---------------------------------------------------------------------------
# Event kinds (the coordination contract)
# ---------------------------------------------------------------------------

IMAGINATOR_STARTED = "imaginator.started"
IMAGINATOR_REGISTERED = "imaginator.registered"
IMAGINATOR_REMOVED = "imaginator.removed"
IMAGINATOR_OBSERVED = "imaginator.observed"
IMAGINATOR_IMAGINED = "imaginator.imagined"
IMAGINATOR_PLANNED = "imaginator.planned"
IMAGINATOR_CERTIFIED = "imaginator.certified"
IMAGINATOR_CLEARED = "imaginator.cleared"

# ---------------------------------------------------------------------------
# Dynamics-family names
# ---------------------------------------------------------------------------

FAMILY_CATEGORICAL = "categorical"
FAMILY_LINEAR_GAUSSIAN = "linear_gaussian"
KNOWN_FAMILIES: frozenset[str] = frozenset({FAMILY_CATEGORICAL, FAMILY_LINEAR_GAUSSIAN})

# ---------------------------------------------------------------------------
# Rollout-sampling selectors
# ---------------------------------------------------------------------------

SAMPLE_POSTERIOR_MEAN = "posterior_mean"
SAMPLE_THOMPSON = "thompson"
SAMPLE_BAYES_AVG = "bayes_avg"
KNOWN_SAMPLE_METHODS: frozenset[str] = frozenset(
    {SAMPLE_POSTERIOR_MEAN, SAMPLE_THOMPSON, SAMPLE_BAYES_AVG}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ImaginatorError(Exception):
    """Base class for Imaginator errors."""


class InvalidConfig(ImaginatorError):
    """The supplied ``ImaginatorConfig`` is invalid."""


class InvalidEnv(ImaginatorError):
    """The supplied ``EnvSpec`` is invalid."""


class InvalidObservation(ImaginatorError):
    """The supplied ``(state, action, next_state, reward)`` is invalid."""


class InsufficientData(ImaginatorError):
    """Not enough data to satisfy the request."""


class UnknownEnv(ImaginatorError):
    """The requested environment ID has not been registered."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImaginatorConfig:
    """Top-level configuration.

    Attributes
    ----------
    family
        Default dynamics family for newly-registered environments.
        Override per-env at ``register_env``.  One of
        ``"categorical"`` or ``"linear_gaussian"``.
    confidence
        Confidence level for every interval the primitive returns
        (Hoeffding LCB / UCB, Maurer-Pontil empirical-Bernstein,
        Howard-Ramdas-McAuliffe-Sekhon anytime-valid sequences).
        Defaults to ``0.95``.
    discount
        Default discount factor ``γ`` for value computations.
    dirichlet_prior
        Pseudo-count for the Dirichlet prior on categorical
        transitions.  ``0.5`` is the Jeffreys prior; ``1.0`` is the
        Laplace prior.  Lower values yield faster convergence with
        higher variance.
    reward_mean_prior
        Prior mean of the reward Normal-Gamma posterior on
        categorical environments.
    reward_precision_prior
        Prior precision (1/variance) of the reward Normal-Gamma
        posterior on categorical environments.
    reward_gamma_a
        Shape parameter ``a`` of the Gamma prior on reward precision.
    reward_gamma_b
        Rate parameter ``b`` of the Gamma prior on reward precision.
    rng_seed
        Seed for the internal RNG used by Thompson sampling and
        Monte Carlo rollouts.  Determinism: same observations + same
        seed → byte-identical fingerprint chain.
    hmac_key
        Optional secret key for HMAC-SHA-256 over every fingerprint
        entry.  When set, the chain is unforgeable without the key.
    max_observations_per_env
        Soft cap on the per-(s, a) observation count to bound the
        Dirichlet posterior memory.  ``None`` is unbounded.
    """

    family: str = FAMILY_CATEGORICAL
    confidence: float = 0.95
    discount: float = 0.95
    dirichlet_prior: float = 1.0
    reward_mean_prior: float = 0.0
    reward_precision_prior: float = 1.0
    reward_gamma_a: float = 1.0
    reward_gamma_b: float = 1.0
    rng_seed: int | None = 0xA61BEEF
    hmac_key: bytes | None = None
    max_observations_per_env: int | None = None

    def __post_init__(self) -> None:
        if self.family not in KNOWN_FAMILIES:
            raise InvalidConfig(
                f"unknown family {self.family!r}; expected one of {sorted(KNOWN_FAMILIES)}"
            )
        if not 0.5 < self.confidence < 1.0:
            raise InvalidConfig(
                f"confidence must be in (0.5, 1.0); got {self.confidence}"
            )
        if not 0.0 <= self.discount <= 1.0:
            raise InvalidConfig(f"discount must be in [0, 1]; got {self.discount}")
        if self.dirichlet_prior <= 0.0:
            raise InvalidConfig(
                f"dirichlet_prior must be positive; got {self.dirichlet_prior}"
            )
        if self.reward_precision_prior <= 0.0:
            raise InvalidConfig(
                f"reward_precision_prior must be positive; got "
                f"{self.reward_precision_prior}"
            )
        if self.reward_gamma_a <= 0.0 or self.reward_gamma_b <= 0.0:
            raise InvalidConfig("Gamma prior shape/rate must be positive")
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes")
        if (
            self.max_observations_per_env is not None
            and self.max_observations_per_env <= 0
        ):
            raise InvalidConfig("max_observations_per_env must be positive")


@dataclass(frozen=True)
class EnvSpec:
    """Specification of a single registered environment.

    For ``categorical`` family, ``states`` and ``actions`` are required.
    For ``linear_gaussian`` family, ``state_dim`` and ``action_dim`` are
    required and ``states`` / ``actions`` are ignored.
    """

    env_id: str
    family: str = FAMILY_CATEGORICAL
    states: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    state_dim: int = 0
    action_dim: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.env_id, str) or not self.env_id:
            raise InvalidEnv("env_id must be a non-empty string")
        if self.family not in KNOWN_FAMILIES:
            raise InvalidEnv(
                f"unknown family {self.family!r}; expected one of "
                f"{sorted(KNOWN_FAMILIES)}"
            )
        if self.family == FAMILY_CATEGORICAL:
            if not self.states or not self.actions:
                raise InvalidEnv(
                    "categorical env requires non-empty states and actions"
                )
            if len(set(self.states)) != len(self.states):
                raise InvalidEnv("states must be unique")
            if len(set(self.actions)) != len(self.actions):
                raise InvalidEnv("actions must be unique")
        else:
            if self.state_dim <= 0 or self.action_dim <= 0:
                raise InvalidEnv("linear_gaussian env requires positive dims")


@dataclass(frozen=True)
class Transition:
    """A single observed environment transition."""

    state: Any
    action: Any
    next_state: Any
    reward: float


@dataclass(frozen=True)
class Rollout:
    """A bundle of imagined trajectories from a single rollout request.

    ``trajectories`` is a list of length ``samples``; each element is a
    list of length ``horizon`` of (state, action, reward) tuples.
    """

    env_id: str
    state: Any
    horizon: int
    samples: int
    method: str
    expected_return: float
    return_std: float
    value_lcb: float
    value_ucb: float
    hrms_lcb: float
    hrms_ucb: float
    return_quantiles: dict[float, float]
    trajectories: list[list[tuple[Any, Any, float]]]
    trajectory_quantiles: list[dict[float, Any]]
    fingerprint_hash: str


@dataclass(frozen=True)
class ValueIterationResult:
    """Output of value iteration on a categorical environment."""

    env_id: str
    horizon: int
    discount: float
    values: dict[str, float]
    policy: dict[str, str]
    sweeps: int
    fingerprint_hash: str


@dataclass(frozen=True)
class PACValueBound:
    """PAC upper bound on the policy-value estimation error.

    The bound is of the form
    ``P(|V̂π − V*π| ≤ epsilon) ≥ 1 − delta``, derived by composing
    the Kearns-Singh 2002 simulation lemma with a per-(s, a)
    Hoeffding bound on the transition posterior.

    ``min_observations`` is the minimum observation count across the
    reachable (s, a) pairs under the policy — the bottleneck.
    """

    env_id: str
    delta: float
    epsilon: float
    discount: float
    horizon: int
    transition_error: float
    reward_error: float
    min_observations: int
    fingerprint_hash: str


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Identifiability of the learned dynamics.

    ``under_observed`` is the list of (state, action) pairs with
    fewer than ``min_observations`` observations.

    ``effective_concentration`` is the per-(s, a) total Dirichlet
    concentration (data + prior); below 2 the posterior is so flat
    that any next-state distribution remains plausible.
    """

    env_id: str
    n_pairs: int
    n_under_observed: int
    under_observed: list[tuple[str, str]]
    min_observations: int
    effective_concentration: dict[tuple[str, str], float]
    fingerprint_hash: str


@dataclass(frozen=True)
class PITCalibrationReport:
    """Probability-integral-transform calibration of one-step
    reward predictions.

    Under a correct posterior predictive,
    ``PIT_i = F_{R_i | s_i, a_i}(r_i_observed)`` is uniform on
    ``[0, 1]``.  We test that against H₀ Uniform via the
    one-sample Kolmogorov-Smirnov test.
    """

    env_id: str
    n_observations: int
    ks_statistic: float
    p_value: float
    fingerprint_hash: str


# ---------------------------------------------------------------------------
# Public math helpers (re-exported)
# ---------------------------------------------------------------------------


def softmax(scores: Sequence[float], beta: float = 1.0) -> list[float]:
    """Numerically-stable softmax with optional inverse-temperature beta."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(beta * (s - m)) for s in scores]
    z = sum(exps)
    if z == 0.0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / z for e in exps]


def dirichlet_mean(alpha: Sequence[float]) -> list[float]:
    """Mean of a Dirichlet(alpha)."""
    s = sum(alpha)
    if s <= 0.0:
        n = len(alpha)
        return [1.0 / n] * n if n else []
    return [a / s for a in alpha]


def dirichlet_sample(rng: random.Random, alpha: Sequence[float]) -> list[float]:
    """Draw one sample from Dirichlet(alpha) via the gamma-ratio trick.

    For each k, draw ``g_k ~ Gamma(alpha_k, 1)`` then normalise.  We use
    Marsaglia-Tsang 2000 (the algorithm Python's stdlib :py:func:`random.gammavariate`
    implements for shape ≥ 1; for shape < 1 we boost via the trick
    ``Gamma(α) = Gamma(α + 1) · U^(1/α)``).
    """
    samples: list[float] = []
    for a in alpha:
        if a <= 0.0:
            samples.append(0.0)
            continue
        if a >= 1.0:
            samples.append(rng.gammavariate(a, 1.0))
        else:
            u = rng.random()
            # ensure strict positivity to avoid log(0)
            while u == 0.0:
                u = rng.random()
                if u == 0.0:  # pragma: no cover  -- defensive
                    u = 1e-300
            base = rng.gammavariate(a + 1.0, 1.0)
            samples.append(base * (u ** (1.0 / a)))
    s = sum(samples)
    if s == 0.0:
        n = len(alpha)
        return [1.0 / n] * n if n else []
    return [g / s for g in samples]


def hoeffding_half_width(n: int, conf: float = 0.95) -> float:
    """Hoeffding 1963 half-width for the mean of a [0, 1] random
    variable from n iid samples at confidence ``conf``.
    """
    if n <= 0:
        return float("inf")
    delta = 1.0 - conf
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int, variance: float, conf: float = 0.95, range_: float = 1.0
) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein half-width.

    Returns the half-width ``ε`` such that with probability ``≥ conf``,
    ``|μ̂ − μ| ≤ ε``  given ``n`` iid samples in ``[0, range_]`` with
    sample variance ``variance``.

    Formula: ``ε = √(2 V log(2/δ) / n) + 7 R log(2/δ) / (3(n-1))``.
    """
    if n <= 1:
        return float("inf")
    delta = 1.0 - conf
    bound = math.sqrt(2.0 * variance * math.log(2.0 / delta) / n) + (
        7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    )
    return bound


def hrms_half_width(n: int, conf: float = 0.95) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence
    sequence half-width for the mean of a [0, 1] random variable from n
    iid samples at confidence ``conf``.

    The closed-form practical bound from §3 of the paper (Eq. 15):
    ``ε = √( (log log(2n) + 0.75 log(10.4 / δ)) / (2n) )``.

    Anytime-valid: the same bound holds **simultaneously** at every
    ``n``, so a coordinator can keep drawing rollouts and read the
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
    xs = sorted(float(x) for x in samples)
    # Clamp to [0, 1]
    xs = [min(max(x, 0.0), 1.0) for x in xs]
    d_plus = 0.0
    d_minus = 0.0
    for i, x in enumerate(xs):
        f_emp_above = (i + 1) / n
        f_emp_below = i / n
        d_plus = max(d_plus, f_emp_above - x)
        d_minus = max(d_minus, x - f_emp_below)
    d = max(d_plus, d_minus)
    # Asymptotic p-value: P(K > d) = 2 Σ_{j=1}^{∞} (-1)^(j-1) exp(-2 j² λ²)
    # with λ = (√n + 0.12 + 0.11/√n) · d  (Stephens 1970 correction).
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d
    if lam <= 0.0:
        return d, 1.0
    p = 0.0
    for j in range(1, 101):
        term = ((-1) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lam ** 2))
        p += term
        if abs(term) < 1e-12:
            break
    p *= 2.0
    return d, max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# Fingerprint chain
# ---------------------------------------------------------------------------


_GENESIS_PREFIX = b"agi.imaginator.v1\x00"


def ledger_root(secret_key: bytes | None = None) -> str:
    """Return the genesis seed of the Imaginator fingerprint chain."""
    seed = _GENESIS_PREFIX + (secret_key or b"")
    return hashlib.sha256(seed).hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    """Stable canonicalisation for hash-chain payloads.

    Floats are quantised to 17 significant digits (round-trip stable).
    Dicts are emitted in sorted-key order.  Iterables become lists.
    """
    import json

    def _quantise(o: Any) -> Any:
        if isinstance(o, float):
            if math.isnan(o):
                return "NaN"
            if math.isinf(o):
                return "Infinity" if o > 0 else "-Infinity"
            return float(repr(o))
        if isinstance(o, dict):
            return {str(k): _quantise(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [_quantise(x) for x in o]
        if isinstance(o, (bytes, bytearray)):
            return o.hex()
        return o

    return json.dumps(_quantise(payload), sort_keys=True, separators=(",", ":")).encode()


def _hash_entry(
    parent: str, payload: dict[str, Any], hmac_key: bytes | None = None
) -> str:
    """Hash one chain entry on top of ``parent``."""
    body = _canonical(payload)
    block = parent.encode() + b"|" + body
    if hmac_key:
        return hmac.new(hmac_key, block, hashlib.sha256).hexdigest()
    return hashlib.sha256(block).hexdigest()


# ---------------------------------------------------------------------------
# Internal per-env state
# ---------------------------------------------------------------------------


@dataclass
class _CategoricalEnvState:
    spec: EnvSpec
    # Dirichlet counts: alpha[(s, a)] = list-of-floats over states
    alpha: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    # Reward posteriors (Normal-Gamma): per (s, a):
    #   n: observation count
    #   sum_r: sum of rewards
    #   sum_rr: sum of squared rewards
    n_sa: dict[tuple[str, str], int] = field(default_factory=dict)
    sum_r: dict[tuple[str, str], float] = field(default_factory=dict)
    sum_rr: dict[tuple[str, str], float] = field(default_factory=dict)
    # PIT records: list of one-step PIT(reward) values from the
    # *predictive distribution before* the observation was seen.
    pit: list[float] = field(default_factory=list)
    # Total observation count
    n_total: int = 0


@dataclass
class _LinearGaussianEnvState:
    spec: EnvSpec
    # Sufficient statistics for Bayesian linear regression with conjugate
    # matrix-normal-inverse-Wishart prior.  We hold:
    #   Λ ∈ R^{(d_s + d_a) × (d_s + d_a)} - precision of weights
    #   λμ ∈ R^{(d_s + d_a) × d_s} - linear term
    #   SSE_y ∈ R^{d_s × d_s}     - sum of y y^T
    #   n ∈ int
    # Initialised with Λ_0 = I (identity) and μ_0 = 0.
    Lambda: list[list[float]] = field(default_factory=list)
    Lmu: list[list[float]] = field(default_factory=list)
    SSE: list[list[float]] = field(default_factory=list)
    n: int = 0
    # Reward weights (treated as a separate Bayesian linear regression
    # from x = (s, a) to scalar r).
    R_Lambda: list[list[float]] = field(default_factory=list)
    R_Lmu: list[float] = field(default_factory=list)
    R_sse: float = 0.0
    R_n: int = 0


# ---------------------------------------------------------------------------
# Imaginator
# ---------------------------------------------------------------------------


# Type alias for an event-publish callback.  Imaginator accepts any
# callable matching ``publish(kind: str, data: dict)`` so it composes
# with ``EventBus`` (``runtime.events``) or any lightweight test
# substitute.
EventPublisher = Callable[[str, dict[str, Any]], None]


class Imaginator:
    """Learned-world-model rollouts as a runtime primitive.

    Threadsafe at the API surface: a single re-entrant lock guards every
    mutation of per-environment state.  Read-only methods (``imagine``,
    ``value_iteration``, ``identifiability_report``, ``pit_calibration``)
    take the same lock to snapshot a consistent view but never fail.

    Observations are O(|states| + |actions| + |outcomes|) for the
    categorical family and O((d_s + d_a)²) for the linear-Gaussian
    family.  Imagined rollouts are O(samples · horizon ·
    sample-cost).
    """

    def __init__(
        self,
        config: ImaginatorConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or ImaginatorConfig()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._rng = random.Random(self.config.rng_seed)
        self._envs: dict[str, _CategoricalEnvState | _LinearGaussianEnvState] = {}
        self._chain_head: str = ledger_root(self.config.hmac_key)
        self._started_ts = time.time()
        self._publish(
            IMAGINATOR_STARTED,
            {
                "ts": self._started_ts,
                "config": {
                    "family": self.config.family,
                    "confidence": self.config.confidence,
                    "discount": self.config.discount,
                    "dirichlet_prior": self.config.dirichlet_prior,
                },
            },
        )

    # ------------------------------------------------------------------
    # Event publishing + fingerprint chain helpers
    # ------------------------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, payload)
        except Exception:
            # Event publishing is best-effort; never fail core logic.
            pass

    def _advance_chain(self, payload: dict[str, Any]) -> str:
        self._chain_head = _hash_entry(
            self._chain_head, payload, self.config.hmac_key
        )
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
        env_id: str,
        *,
        states: Iterable[str] | None = None,
        actions: Iterable[str] | None = None,
        state_dim: int = 0,
        action_dim: int = 0,
        family: str | None = None,
    ) -> EnvSpec:
        """Register a new environment.

        For ``categorical`` family, supply ``states`` and ``actions``
        (non-empty iterables of strings).  For ``linear_gaussian``,
        supply ``state_dim`` and ``action_dim`` (positive ints).
        """
        fam = family or self.config.family
        spec = EnvSpec(
            env_id=env_id,
            family=fam,
            states=tuple(states or ()),
            actions=tuple(actions or ()),
            state_dim=state_dim,
            action_dim=action_dim,
        )
        with self._lock:
            if env_id in self._envs:
                raise InvalidEnv(f"env {env_id!r} already registered")
            if fam == FAMILY_CATEGORICAL:
                self._envs[env_id] = self._init_categorical(spec)
            else:
                self._envs[env_id] = self._init_linear_gaussian(spec)
            self._advance_chain(
                {
                    "op": "register",
                    "env_id": env_id,
                    "family": fam,
                    "states": list(spec.states),
                    "actions": list(spec.actions),
                    "state_dim": spec.state_dim,
                    "action_dim": spec.action_dim,
                }
            )
            self._publish(
                IMAGINATOR_REGISTERED,
                {
                    "env_id": env_id,
                    "family": fam,
                    "n_states": len(spec.states),
                    "n_actions": len(spec.actions),
                    "head": self._chain_head,
                },
            )
        return spec

    def remove_env(self, env_id: str) -> None:
        with self._lock:
            if env_id not in self._envs:
                raise UnknownEnv(env_id)
            del self._envs[env_id]
            self._advance_chain({"op": "remove", "env_id": env_id})
            self._publish(
                IMAGINATOR_REMOVED, {"env_id": env_id, "head": self._chain_head}
            )

    def clear(self) -> None:
        """Remove all envs and reset the chain to genesis."""
        with self._lock:
            self._envs.clear()
            self._chain_head = ledger_root(self.config.hmac_key)
            self._advance_chain({"op": "clear"})
            self._publish(IMAGINATOR_CLEARED, {"head": self._chain_head})

    def envs(self) -> list[str]:
        with self._lock:
            return sorted(self._envs.keys())

    def env_spec(self, env_id: str) -> EnvSpec:
        with self._lock:
            return self._require_env(env_id).spec

    # ------------------------------------------------------------------
    # Initialisation helpers (per family)
    # ------------------------------------------------------------------

    def _init_categorical(self, spec: EnvSpec) -> _CategoricalEnvState:
        state = _CategoricalEnvState(spec=spec)
        prior = self.config.dirichlet_prior
        ns = len(spec.states)
        for s in spec.states:
            for a in spec.actions:
                state.alpha[(s, a)] = [prior] * ns
                state.n_sa[(s, a)] = 0
                state.sum_r[(s, a)] = 0.0
                state.sum_rr[(s, a)] = 0.0
        return state

    def _init_linear_gaussian(self, spec: EnvSpec) -> _LinearGaussianEnvState:
        d_in = spec.state_dim + spec.action_dim
        d_out = spec.state_dim
        state = _LinearGaussianEnvState(spec=spec)
        state.Lambda = [[1.0 if i == j else 0.0 for j in range(d_in)] for i in range(d_in)]
        state.Lmu = [[0.0] * d_out for _ in range(d_in)]
        state.SSE = [[0.0] * d_out for _ in range(d_out)]
        state.R_Lambda = [[1.0 if i == j else 0.0 for j in range(d_in)] for i in range(d_in)]
        state.R_Lmu = [0.0] * d_in
        return state

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        env_id: str,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
    ) -> None:
        """Record a single observed transition.

        For categorical: state / action / next_state must be in the
        registered string vocabulary.  For linear_gaussian:
        state / next_state are sequences of length ``state_dim`` and
        action is a sequence of length ``action_dim``.
        """
        with self._lock:
            env = self._require_env(env_id)
            if isinstance(env, _CategoricalEnvState):
                self._observe_categorical(env, state, action, next_state, reward)
            else:
                self._observe_linear_gaussian(env, state, action, next_state, reward)
            payload = {
                "op": "observe",
                "env_id": env_id,
                "state": _canonical_value(state),
                "action": _canonical_value(action),
                "next_state": _canonical_value(next_state),
                "reward": float(reward),
            }
            self._advance_chain(payload)
            self._publish(
                IMAGINATOR_OBSERVED,
                {
                    "env_id": env_id,
                    "reward": float(reward),
                    "head": self._chain_head,
                },
            )

    def _observe_categorical(
        self,
        env: _CategoricalEnvState,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
    ) -> None:
        spec = env.spec
        if state not in spec.states:
            raise InvalidObservation(f"unknown state {state!r}")
        if action not in spec.actions:
            raise InvalidObservation(f"unknown action {action!r}")
        if next_state not in spec.states:
            raise InvalidObservation(f"unknown next_state {next_state!r}")
        if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
            raise InvalidObservation(f"reward must be finite real; got {reward!r}")
        cap = self.config.max_observations_per_env
        if cap is not None and env.n_total >= cap:
            # Soft cap: drop the oldest by half-decay of alpha counts.
            for k in env.alpha:
                env.alpha[k] = [
                    max(self.config.dirichlet_prior, a * 0.5) for a in env.alpha[k]
                ]
            for k in env.n_sa:
                env.n_sa[k] = max(0, env.n_sa[k] // 2)
                env.sum_r[k] *= 0.5
                env.sum_rr[k] *= 0.5
            env.n_total = env.n_total // 2
        # Pre-update PIT under the *current* predictive distribution.
        try:
            cdf = self._reward_cdf_categorical(env, state, action, reward)
            env.pit.append(cdf)
        except InsufficientData:
            pass
        idx = spec.states.index(next_state)
        env.alpha[(state, action)][idx] += 1.0
        env.n_sa[(state, action)] += 1
        env.sum_r[(state, action)] += float(reward)
        env.sum_rr[(state, action)] += float(reward) * float(reward)
        env.n_total += 1

    def _observe_linear_gaussian(
        self,
        env: _LinearGaussianEnvState,
        state: Sequence[float],
        action: Sequence[float],
        next_state: Sequence[float],
        reward: float,
    ) -> None:
        spec = env.spec
        if len(state) != spec.state_dim or len(next_state) != spec.state_dim:
            raise InvalidObservation(
                f"state dimensions must match registered state_dim={spec.state_dim}"
            )
        if len(action) != spec.action_dim:
            raise InvalidObservation(
                f"action dimensions must match registered action_dim={spec.action_dim}"
            )
        x = list(state) + list(action)
        y = list(next_state)
        d_in = len(x)
        # Λ ← Λ + x xᵀ
        for i in range(d_in):
            xi = x[i]
            for j in range(d_in):
                env.Lambda[i][j] += xi * x[j]
        # λμ ← λμ + x yᵀ
        for i in range(d_in):
            for j in range(spec.state_dim):
                env.Lmu[i][j] += x[i] * y[j]
        # SSE ← SSE + y yᵀ
        for i in range(spec.state_dim):
            for j in range(spec.state_dim):
                env.SSE[i][j] += y[i] * y[j]
        env.n += 1
        # Reward regression: ŵ such that r ≈ wᵀ x.
        for i in range(d_in):
            for j in range(d_in):
                env.R_Lambda[i][j] += x[i] * x[j]
        for i in range(d_in):
            env.R_Lmu[i] += x[i] * float(reward)
        env.R_sse += float(reward) * float(reward)
        env.R_n += 1

    # ------------------------------------------------------------------
    # Posterior accessors
    # ------------------------------------------------------------------

    def posterior_mean_transition(
        self, env_id: str, state: str, action: str
    ) -> dict[str, float]:
        """Return the posterior-mean Dirichlet probability of every next
        state given ``(state, action)``.
        """
        with self._lock:
            env = self._require_categorical(env_id)
            self._check_sa(env, state, action)
            alpha = env.alpha[(state, action)]
            mean = dirichlet_mean(alpha)
            return {s: p for s, p in zip(env.spec.states, mean)}

    def posterior_mean_reward(self, env_id: str, state: str, action: str) -> float:
        with self._lock:
            env = self._require_categorical(env_id)
            self._check_sa(env, state, action)
            return self._reward_mean(env, state, action)

    def posterior_variance_reward(
        self, env_id: str, state: str, action: str
    ) -> float:
        with self._lock:
            env = self._require_categorical(env_id)
            self._check_sa(env, state, action)
            return self._reward_var(env, state, action)

    def _reward_mean(self, env: _CategoricalEnvState, state: str, action: str) -> float:
        # Posterior mean under Normal prior with prior mean μ₀ and
        # prior precision κ₀ = reward_precision_prior, observing n
        # samples with mean x̄: posterior mean = (κ₀ μ₀ + n x̄) /
        # (κ₀ + n).
        n = env.n_sa.get((state, action), 0)
        s_r = env.sum_r.get((state, action), 0.0)
        prior_mean = self.config.reward_mean_prior
        prior_prec = self.config.reward_precision_prior
        return (prior_prec * prior_mean + s_r) / (prior_prec + n)

    def _reward_var(self, env: _CategoricalEnvState, state: str, action: str) -> float:
        n = env.n_sa.get((state, action), 0)
        s_r = env.sum_r.get((state, action), 0.0)
        s_rr = env.sum_rr.get((state, action), 0.0)
        # Posterior variance of mean under Normal-Gamma:
        #   E[σ²] = b / (a − 1), with a posterior = a₀ + n / 2,
        #   b posterior = b₀ + (1/2) Σ (x - x̄)² + (κ₀ n (x̄ - μ₀)²) / (2(κ₀+n))
        if n == 0:
            a = self.config.reward_gamma_a
            b = self.config.reward_gamma_b
            return b / max(a - 1.0, 1e-12)
        mean_obs = s_r / n
        sse = max(s_rr - n * mean_obs * mean_obs, 0.0)
        a_post = self.config.reward_gamma_a + 0.5 * n
        b_post = (
            self.config.reward_gamma_b
            + 0.5 * sse
            + 0.5
            * self.config.reward_precision_prior
            * n
            * (mean_obs - self.config.reward_mean_prior) ** 2
            / (self.config.reward_precision_prior + n)
        )
        return b_post / max(a_post - 1.0, 1e-12) / max(
            self.config.reward_precision_prior + n, 1e-12
        )

    def _reward_cdf_categorical(
        self,
        env: _CategoricalEnvState,
        state: str,
        action: str,
        observed_reward: float,
    ) -> float:
        """One-step PIT under the Student-t reward predictive.

        Marginalising the Normal-Gamma posterior over both μ and τ
        yields a Student-t predictive distribution with 2a degrees of
        freedom, location μ_post, and scale √(b_post (κ_post + 1) /
        (a_post κ_post)).
        """
        n = env.n_sa.get((state, action), 0)
        if n == 0:
            raise InsufficientData("no reward data yet")
        s_r = env.sum_r.get((state, action), 0.0)
        s_rr = env.sum_rr.get((state, action), 0.0)
        kappa0 = self.config.reward_precision_prior
        mu0 = self.config.reward_mean_prior
        a0 = self.config.reward_gamma_a
        b0 = self.config.reward_gamma_b
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + s_r) / kappa_n
        x_bar = s_r / n
        sse = max(s_rr - n * x_bar * x_bar, 0.0)
        a_n = a0 + 0.5 * n
        b_n = (
            b0
            + 0.5 * sse
            + 0.5 * kappa0 * n * (x_bar - mu0) ** 2 / kappa_n
        )
        # Student-t parameters
        df = 2.0 * a_n
        scale = math.sqrt(b_n * (kappa_n + 1.0) / (a_n * kappa_n))
        t = (observed_reward - mu_n) / max(scale, 1e-12)
        # CDF via incomplete-beta: F(t) = 1 - (1/2) I_{df/(df+t²)}(df/2, 1/2)
        # for t ≥ 0; symmetric for t < 0.
        x = df / (df + t * t)
        I = _regularised_incomplete_beta(df / 2.0, 0.5, max(min(x, 1.0), 0.0))
        if t >= 0:
            return 1.0 - 0.5 * I
        return 0.5 * I

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_transition(
        self,
        env_id: str,
        state: str,
        action: str,
        *,
        method: str = SAMPLE_POSTERIOR_MEAN,
    ) -> str:
        """Sample one next-state from the posterior predictive over
        the dynamics under the given sampling method.

        ``method`` ∈ ``{posterior_mean, thompson, bayes_avg}``.

        * ``posterior_mean`` — sample from the Dirichlet posterior
          mean (closed-form).
        * ``thompson`` — first sample a transition row from the
          Dirichlet, then sample a next state from that draw.
        * ``bayes_avg`` — average ``n_models`` Thompson rows then
          sample.  Default ``n_models=8``; pass via the dedicated
          ``sample_transition_bma`` for full control.
        """
        if method not in KNOWN_SAMPLE_METHODS:
            raise ImaginatorError(f"unknown sample method {method!r}")
        with self._lock:
            env = self._require_categorical(env_id)
            self._check_sa(env, state, action)
            alpha = env.alpha[(state, action)]
            if method == SAMPLE_POSTERIOR_MEAN:
                probs = dirichlet_mean(alpha)
            elif method == SAMPLE_THOMPSON:
                probs = dirichlet_sample(self._rng, alpha)
            else:  # bayes_avg
                acc = [0.0] * len(alpha)
                n_models = 8
                for _ in range(n_models):
                    draw = dirichlet_sample(self._rng, alpha)
                    for i, p in enumerate(draw):
                        acc[i] += p / n_models
                probs = acc
            idx = _sample_categorical(self._rng, probs)
            return env.spec.states[idx]

    def sample_reward(
        self, env_id: str, state: str, action: str
    ) -> float:
        """Sample one reward from the Normal-Gamma posterior predictive
        (Student-t marginal)."""
        with self._lock:
            env = self._require_categorical(env_id)
            self._check_sa(env, state, action)
            n = env.n_sa.get((state, action), 0)
            s_r = env.sum_r.get((state, action), 0.0)
            s_rr = env.sum_rr.get((state, action), 0.0)
            kappa0 = self.config.reward_precision_prior
            mu0 = self.config.reward_mean_prior
            a0 = self.config.reward_gamma_a
            b0 = self.config.reward_gamma_b
            kappa_n = kappa0 + n
            mu_n = (kappa0 * mu0 + s_r) / kappa_n
            x_bar = s_r / n if n > 0 else 0.0
            sse = max(s_rr - n * x_bar * x_bar, 0.0)
            a_n = a0 + 0.5 * n
            b_n = (
                b0
                + 0.5 * sse
                + 0.5 * kappa0 * n * (x_bar - mu0) ** 2 / kappa_n
            )
            df = 2.0 * a_n
            scale = math.sqrt(b_n * (kappa_n + 1.0) / (a_n * kappa_n))
            t = _student_t_sample(self._rng, df)
            return mu_n + scale * t

    # ------------------------------------------------------------------
    # Imagined rollouts
    # ------------------------------------------------------------------

    def imagine(
        self,
        env_id: str,
        *,
        state: str,
        policy: Callable[[str], str],
        horizon: int,
        samples: int = 64,
        method: str = SAMPLE_THOMPSON,
        discount: float | None = None,
    ) -> Rollout:
        """Roll out ``samples`` imagined trajectories of length
        ``horizon`` starting from ``state`` under ``policy``.

        Returns a :class:`Rollout` with:

          * ``expected_return`` — Monte Carlo mean of discounted
            return ``Σ_h γ^h r_h``.
          * ``return_std`` — sample standard deviation of returns.
          * ``value_lcb`` / ``value_ucb`` — Maurer-Pontil 2009
            empirical-Bernstein 95% interval (or the configured
            confidence).
          * ``hrms_lcb`` / ``hrms_ucb`` — Howard-Ramdas-McAuliffe-
            Sekhon anytime-valid 95% confidence sequence.
          * ``return_quantiles`` — empirical CDF at
            {0.05, 0.25, 0.5, 0.75, 0.95}.
          * ``trajectories`` — the imagined trajectories themselves.
          * ``trajectory_quantiles`` — at every horizon step ``h``,
            the empirical {0.025, 0.5, 0.975} quantile of the
            simulated state.
          * ``fingerprint_hash`` — the chain head after the imagine.
        """
        if horizon < 1:
            raise ImaginatorError("horizon must be ≥ 1")
        if samples < 1:
            raise ImaginatorError("samples must be ≥ 1")
        if method not in KNOWN_SAMPLE_METHODS:
            raise ImaginatorError(f"unknown sample method {method!r}")
        gamma = self.config.discount if discount is None else discount
        if not 0.0 <= gamma <= 1.0:
            raise ImaginatorError(f"discount must be in [0, 1]; got {gamma}")

        with self._lock:
            env = self._require_categorical(env_id)
            if state not in env.spec.states:
                raise InvalidObservation(f"unknown state {state!r}")

            # Maximum return for [0, 1]-style scaling: cap return at the
            # observed reward range expanded by 3 σ for the Bernstein /
            # HRMS bounds.  We always normalise into [0, 1] for the
            # bound, then de-normalise.
            reward_range = self._observed_reward_range(env)
            r_lo, r_hi = reward_range
            if r_hi <= r_lo:
                r_hi = r_lo + 1.0
            traj_returns: list[float] = []
            trajectories: list[list[tuple[Any, Any, float]]] = []
            per_step_states: list[list[str]] = [[] for _ in range(horizon)]
            if method == SAMPLE_BAYES_AVG:
                # Pre-sample n_models posterior transitions per (s, a)
                # to amortise the BMA average across the rollouts.
                n_models = 8
                avg_alpha: dict[tuple[str, str], list[float]] = {}
                for k, alpha in env.alpha.items():
                    acc = [0.0] * len(alpha)
                    for _ in range(n_models):
                        draw = dirichlet_sample(self._rng, alpha)
                        for i, p in enumerate(draw):
                            acc[i] += p / n_models
                    avg_alpha[k] = acc
                per_traj_transitions = avg_alpha
            else:
                per_traj_transitions = None

            for _ in range(samples):
                if method == SAMPLE_THOMPSON:
                    # One transition matrix sample per trajectory (PSRL).
                    transitions = {
                        k: dirichlet_sample(self._rng, alpha)
                        for k, alpha in env.alpha.items()
                    }
                elif method == SAMPLE_POSTERIOR_MEAN:
                    transitions = {
                        k: dirichlet_mean(alpha) for k, alpha in env.alpha.items()
                    }
                else:
                    transitions = per_traj_transitions  # bayes_avg

                cur = state
                disc = 1.0
                ret = 0.0
                traj: list[tuple[Any, Any, float]] = []
                for h in range(horizon):
                    a = policy(cur)
                    if a not in env.spec.actions:
                        raise ImaginatorError(
                            f"policy returned unknown action {a!r}"
                        )
                    probs = transitions[(cur, a)]
                    idx = _sample_categorical(self._rng, probs)
                    nxt = env.spec.states[idx]
                    # Sample reward from per-(s, a) posterior predictive.
                    rwd = self._reward_predictive_sample(env, cur, a)
                    ret += disc * rwd
                    disc *= gamma
                    traj.append((cur, a, rwd))
                    per_step_states[h].append(nxt)
                    cur = nxt
                trajectories.append(traj)
                traj_returns.append(ret)

            mu = sum(traj_returns) / samples
            if samples > 1:
                var = sum((r - mu) ** 2 for r in traj_returns) / (samples - 1)
            else:
                var = 0.0
            std = math.sqrt(max(var, 0.0))
            # Normalise into [0, 1] using observed bounds * horizon-discount factor.
            disc_norm = (1.0 - gamma ** horizon) / max(1.0 - gamma, 1e-12) if gamma < 1.0 else float(horizon)
            R_min = r_lo * disc_norm
            R_max = r_hi * disc_norm
            R_range = max(R_max - R_min, 1e-12)
            normalised = [(r - R_min) / R_range for r in traj_returns]
            norm_mu = sum(normalised) / samples
            if samples > 1:
                norm_var = sum((r - norm_mu) ** 2 for r in normalised) / (
                    samples - 1
                )
            else:
                norm_var = 0.0
            eb = empirical_bernstein_half_width(
                samples, norm_var, self.config.confidence, 1.0
            )
            hr = hrms_half_width(samples, self.config.confidence)
            value_lcb = mu - eb * R_range
            value_ucb = mu + eb * R_range
            hrms_lcb = mu - hr * R_range
            hrms_ucb = mu + hr * R_range
            quantiles = _empirical_quantiles(
                traj_returns, (0.05, 0.25, 0.5, 0.75, 0.95)
            )
            traj_quants: list[dict[float, Any]] = []
            for h in range(horizon):
                steps = per_step_states[h]
                # Use the modal state + its empirical mass for the median;
                # for 2.5/97.5 use the rarest / most-common.
                counts: dict[str, int] = {}
                for s in steps:
                    counts[s] = counts.get(s, 0) + 1
                if not counts:
                    traj_quants.append({})
                    continue
                ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                traj_quants.append(
                    {
                        0.025: ordered[-1][0],
                        0.5: ordered[0][0],
                        0.975: ordered[0][0] if len(ordered) == 1 else ordered[0][0],
                    }
                )
            payload = {
                "op": "imagine",
                "env_id": env_id,
                "state": state,
                "horizon": horizon,
                "samples": samples,
                "method": method,
                "discount": gamma,
                "mu": mu,
                "std": std,
            }
            self._advance_chain(payload)
            self._publish(
                IMAGINATOR_IMAGINED,
                {
                    "env_id": env_id,
                    "state": state,
                    "horizon": horizon,
                    "samples": samples,
                    "method": method,
                    "mu": mu,
                    "head": self._chain_head,
                },
            )
            return Rollout(
                env_id=env_id,
                state=state,
                horizon=horizon,
                samples=samples,
                method=method,
                expected_return=mu,
                return_std=std,
                value_lcb=value_lcb,
                value_ucb=value_ucb,
                hrms_lcb=hrms_lcb,
                hrms_ucb=hrms_ucb,
                return_quantiles=quantiles,
                trajectories=trajectories,
                trajectory_quantiles=traj_quants,
                fingerprint_hash=self._chain_head,
            )

    def _reward_predictive_sample(
        self, env: _CategoricalEnvState, state: str, action: str
    ) -> float:
        n = env.n_sa.get((state, action), 0)
        s_r = env.sum_r.get((state, action), 0.0)
        s_rr = env.sum_rr.get((state, action), 0.0)
        kappa0 = self.config.reward_precision_prior
        mu0 = self.config.reward_mean_prior
        a0 = self.config.reward_gamma_a
        b0 = self.config.reward_gamma_b
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + s_r) / kappa_n
        x_bar = s_r / n if n > 0 else 0.0
        sse = max(s_rr - n * x_bar * x_bar, 0.0)
        a_n = a0 + 0.5 * n
        b_n = (
            b0
            + 0.5 * sse
            + 0.5 * kappa0 * n * (x_bar - mu0) ** 2 / kappa_n
        )
        df = 2.0 * a_n
        scale = math.sqrt(max(b_n * (kappa_n + 1.0) / (a_n * kappa_n), 1e-12))
        t = _student_t_sample(self._rng, df)
        return mu_n + scale * t

    def _observed_reward_range(self, env: _CategoricalEnvState) -> tuple[float, float]:
        """Return ``(r_min, r_max)`` over observed rewards, with a
        sensible default when no rewards have been seen."""
        any_data = False
        r_min = float("inf")
        r_max = float("-inf")
        for k in env.n_sa:
            n = env.n_sa[k]
            if n == 0:
                continue
            mean = env.sum_r[k] / n
            var = max(env.sum_rr[k] / n - mean * mean, 0.0)
            sd = math.sqrt(var)
            r_min = min(r_min, mean - 3 * sd)
            r_max = max(r_max, mean + 3 * sd)
            any_data = True
        if not any_data:
            return -1.0, 1.0
        if r_max <= r_min:
            r_max = r_min + 1.0
        return r_min, r_max

    # ------------------------------------------------------------------
    # Value iteration on categorical environments
    # ------------------------------------------------------------------

    def value_iteration(
        self,
        env_id: str,
        *,
        horizon: int = 50,
        discount: float | None = None,
        tol: float = 1e-6,
    ) -> ValueIterationResult:
        """Plan via dynamic-programming value iteration on the
        posterior-mean transition / reward.

        Returns the deterministic greedy policy + per-state value
        function ``V*(s)`` under the planning horizon.
        """
        gamma = self.config.discount if discount is None else discount
        if not 0.0 <= gamma <= 1.0:
            raise ImaginatorError(f"discount must be in [0, 1]; got {gamma}")
        if horizon < 1:
            raise ImaginatorError("horizon must be ≥ 1")
        with self._lock:
            env = self._require_categorical(env_id)
            states = list(env.spec.states)
            actions = list(env.spec.actions)
            v: dict[str, float] = {s: 0.0 for s in states}
            policy: dict[str, str] = {s: actions[0] for s in states}
            sweeps = 0
            for _ in range(horizon):
                sweeps += 1
                v_next: dict[str, float] = {}
                pol_next: dict[str, str] = {}
                max_delta = 0.0
                for s in states:
                    best_q = -float("inf")
                    best_a = actions[0]
                    for a in actions:
                        probs = dirichlet_mean(env.alpha[(s, a)])
                        r = self._reward_mean(env, s, a)
                        q = r + gamma * sum(
                            probs[i] * v[states[i]] for i in range(len(states))
                        )
                        if q > best_q:
                            best_q = q
                            best_a = a
                    v_next[s] = best_q
                    pol_next[s] = best_a
                    max_delta = max(max_delta, abs(best_q - v[s]))
                v = v_next
                policy = pol_next
                if max_delta < tol:
                    break
            payload = {
                "op": "plan",
                "env_id": env_id,
                "horizon": horizon,
                "discount": gamma,
                "sweeps": sweeps,
                "values": v,
                "policy": policy,
            }
            self._advance_chain(payload)
            self._publish(
                IMAGINATOR_PLANNED,
                {
                    "env_id": env_id,
                    "horizon": horizon,
                    "sweeps": sweeps,
                    "head": self._chain_head,
                },
            )
            return ValueIterationResult(
                env_id=env_id,
                horizon=horizon,
                discount=gamma,
                values=dict(v),
                policy=dict(policy),
                sweeps=sweeps,
                fingerprint_hash=self._chain_head,
            )

    def thompson_policy(
        self, env_id: str, *, horizon: int = 50, discount: float | None = None
    ) -> ValueIterationResult:
        """**Posterior Sampling for RL** (Strens 2000 / Osband-Russo-
        Van Roy 2013): draw one transition matrix from the Dirichlet
        posterior, then return the value-iteration policy for that
        draw.  Under PSRL the Bayesian regret is
        ``O(τ √(SAT log T))``.
        """
        with self._lock:
            env = self._require_categorical(env_id)
            # Sample one transition matrix.
            sampled: dict[tuple[str, str], list[float]] = {
                k: dirichlet_sample(self._rng, env.alpha[k]) for k in env.alpha
            }
            # Sample one reward mean per (s, a) from Normal-Gamma posterior.
            sampled_r: dict[tuple[str, str], float] = {}
            for k in env.n_sa:
                sampled_r[k] = self._reward_mean(env, k[0], k[1])
            gamma = self.config.discount if discount is None else discount
            states = list(env.spec.states)
            actions = list(env.spec.actions)
            v: dict[str, float] = {s: 0.0 for s in states}
            policy: dict[str, str] = {s: actions[0] for s in states}
            sweeps = 0
            for _ in range(horizon):
                sweeps += 1
                v_next: dict[str, float] = {}
                pol_next: dict[str, str] = {}
                max_delta = 0.0
                for s in states:
                    best_q = -float("inf")
                    best_a = actions[0]
                    for a in actions:
                        probs = sampled[(s, a)]
                        r = sampled_r[(s, a)]
                        q = r + gamma * sum(
                            probs[i] * v[states[i]] for i in range(len(states))
                        )
                        if q > best_q:
                            best_q = q
                            best_a = a
                    v_next[s] = best_q
                    pol_next[s] = best_a
                    max_delta = max(max_delta, abs(best_q - v[s]))
                v = v_next
                policy = pol_next
                if max_delta < 1e-6:
                    break
            self._advance_chain(
                {
                    "op": "thompson_plan",
                    "env_id": env_id,
                    "horizon": horizon,
                    "discount": gamma,
                    "sweeps": sweeps,
                }
            )
            return ValueIterationResult(
                env_id=env_id,
                horizon=horizon,
                discount=gamma,
                values=dict(v),
                policy=dict(policy),
                sweeps=sweeps,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # PAC value bound
    # ------------------------------------------------------------------

    def pac_value_bound(
        self,
        env_id: str,
        *,
        policy: dict[str, str] | Callable[[str], str],
        horizon: int = 50,
        delta: float = 0.05,
        discount: float | None = None,
    ) -> PACValueBound:
        """**Kearns-Singh 2002** simulation-lemma PAC bound on the
        policy-value estimation error.

        With probability ``≥ 1 − delta`` over the observation history,

        ``|V̂^π(s) − V^π(s)| ≤ ε``

        where

        ``ε = (γ / (1 − γ)²) · √(2 log(|S||A|/δ) / min_n(s, a))``
        + reward-term

        and ``min_n`` is the minimum observation count across the
        reachable (s, a) pairs.
        """
        gamma = self.config.discount if discount is None else discount
        if not 0.0 < delta < 1.0:
            raise ImaginatorError(f"delta must be in (0, 1); got {delta}")
        with self._lock:
            env = self._require_categorical(env_id)
            states = env.spec.states
            actions = env.spec.actions
            S = len(states)
            A = len(actions)
            policy_fn: Callable[[str], str] = (
                policy if callable(policy) else (lambda s, p=policy: p[s])
            )
            # Reachable (s, a) under the policy from any start: by
            # construction every state's chosen action.
            reachable = [(s, policy_fn(s)) for s in states]
            ns = [env.n_sa.get((s, a), 0) for s, a in reachable]
            min_n = min(ns) if ns else 0
            if min_n == 0:
                raise InsufficientData(
                    "PAC bound requires ≥ 1 observation at every reachable (s, a)"
                )
            # Per-(s, a) transition Hoeffding radius: TV ≤ √( 2 log(|S||A| / δ) / n).
            log_term = math.log(S * A / delta)
            transition_err = math.sqrt(2.0 * log_term / min_n)
            # Reward Hoeffding radius assuming rewards bounded by max
            # observed range.  Use 3σ as a robust width.
            r_lo, r_hi = self._observed_reward_range(env)
            r_range = max(r_hi - r_lo, 1.0)
            reward_err = r_range * math.sqrt(log_term / (2.0 * min_n))
            # Effective horizon scaling
            if gamma >= 1.0:
                scale = float(horizon)
                tx_scale = scale * scale
            else:
                scale = 1.0 / (1.0 - gamma)
                tx_scale = gamma * scale * scale
            epsilon = tx_scale * transition_err + scale * reward_err
            payload = {
                "op": "pac_bound",
                "env_id": env_id,
                "delta": delta,
                "epsilon": epsilon,
                "horizon": horizon,
                "discount": gamma,
                "min_observations": min_n,
            }
            self._advance_chain(payload)
            self._publish(
                IMAGINATOR_CERTIFIED,
                {
                    "env_id": env_id,
                    "delta": delta,
                    "epsilon": epsilon,
                    "head": self._chain_head,
                },
            )
            return PACValueBound(
                env_id=env_id,
                delta=delta,
                epsilon=epsilon,
                discount=gamma,
                horizon=horizon,
                transition_error=transition_err,
                reward_error=reward_err,
                min_observations=min_n,
                fingerprint_hash=self._chain_head,
            )

    def required_samples_for_pac(
        self,
        *,
        env_id: str,
        epsilon: float,
        delta: float = 0.05,
        discount: float | None = None,
    ) -> int:
        """Invert the PAC bound: minimum ``min_n`` per reachable (s, a)
        to achieve ``|V̂ − V*| ≤ ε`` with probability ``1 − δ``.

        Strehl-Littman-Wiewiora 2009 PAC-MDP sample complexity
        ``O((SA / ε²(1−γ)⁴) · log(SAδ⁻¹))``.
        """
        with self._lock:
            env = self._require_categorical(env_id)
            S = len(env.spec.states)
            A = len(env.spec.actions)
        gamma = self.config.discount if discount is None else discount
        if epsilon <= 0:
            raise ImaginatorError("epsilon must be positive")
        if not 0.0 < delta < 1.0:
            raise ImaginatorError("delta must be in (0, 1)")
        scale = (1.0 / max(1.0 - gamma, 1e-9)) ** 4
        return int(
            math.ceil(scale * S * A / (epsilon ** 2) * math.log(S * A / delta))
        )

    # ------------------------------------------------------------------
    # Bayesian model averaging
    # ------------------------------------------------------------------

    def bayes_average_value(
        self,
        env_id: str,
        *,
        policy: dict[str, str] | Callable[[str], str],
        horizon: int,
        samples: int = 32,
        n_models: int = 8,
        discount: float | None = None,
        start: str | None = None,
    ) -> float:
        """Bayesian Model Averaging value estimate.

        For each of ``n_models`` posterior-sampled transition matrices,
        compute the average return over ``samples`` Monte Carlo rollouts
        starting from ``start`` (or every state uniformly), then average
        the per-model values.  Equivalent to the posterior-predictive
        value of the policy.  Minimises log-loss in expectation
        (Madigan-Raftery 1994).
        """
        gamma = self.config.discount if discount is None else discount
        with self._lock:
            env = self._require_categorical(env_id)
            states = list(env.spec.states)
            policy_fn: Callable[[str], str] = (
                policy if callable(policy) else (lambda s, p=policy: p[s])
            )
            starts = [start] if start else states
            outer_acc = 0.0
            for _ in range(n_models):
                sampled = {
                    k: dirichlet_sample(self._rng, env.alpha[k])
                    for k in env.alpha
                }
                sampled_r = {k: self._reward_mean(env, k[0], k[1]) for k in env.n_sa}
                inner_acc = 0.0
                for s0 in starts:
                    for _ in range(samples):
                        cur = s0
                        disc = 1.0
                        ret = 0.0
                        for _ in range(horizon):
                            a = policy_fn(cur)
                            probs = sampled[(cur, a)]
                            idx = _sample_categorical(self._rng, probs)
                            nxt = states[idx]
                            r = sampled_r.get((cur, a), 0.0)
                            ret += disc * r
                            disc *= gamma
                            cur = nxt
                        inner_acc += ret / samples
                outer_acc += (inner_acc / len(starts)) / n_models
            return outer_acc

    # ------------------------------------------------------------------
    # Identifiability + calibration reports
    # ------------------------------------------------------------------

    def identifiability_report(
        self, env_id: str, min_observations: int = 5
    ) -> IdentifiabilityReport:
        with self._lock:
            env = self._require_categorical(env_id)
            spec = env.spec
            under: list[tuple[str, str]] = []
            conc: dict[tuple[str, str], float] = {}
            for s in spec.states:
                for a in spec.actions:
                    n = env.n_sa.get((s, a), 0)
                    conc[(s, a)] = float(sum(env.alpha[(s, a)]))
                    if n < min_observations:
                        under.append((s, a))
            self._advance_chain(
                {
                    "op": "identifiability",
                    "env_id": env_id,
                    "n_pairs": len(spec.states) * len(spec.actions),
                    "n_under_observed": len(under),
                    "min_observations": min_observations,
                }
            )
            return IdentifiabilityReport(
                env_id=env_id,
                n_pairs=len(spec.states) * len(spec.actions),
                n_under_observed=len(under),
                under_observed=under,
                min_observations=min_observations,
                effective_concentration=conc,
                fingerprint_hash=self._chain_head,
            )

    def pit_calibration(self, env_id: str) -> PITCalibrationReport:
        with self._lock:
            env = self._require_categorical(env_id)
            n = len(env.pit)
            if n == 0:
                raise InsufficientData("no PIT samples yet")
            d, p = ks_pvalue(env.pit)
            self._advance_chain(
                {
                    "op": "pit",
                    "env_id": env_id,
                    "n": n,
                    "ks_statistic": d,
                    "p_value": p,
                }
            )
            return PITCalibrationReport(
                env_id=env_id,
                n_observations=n,
                ks_statistic=d,
                p_value=p,
                fingerprint_hash=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Linear-Gaussian: posterior accessors + moment rollouts
    # ------------------------------------------------------------------

    def posterior_mean_dynamics(self, env_id: str) -> tuple[list[list[float]], list[list[float]]]:
        """Return (A_mean, B_mean) for the linear-Gaussian family.

        A_mean is state_dim × state_dim; B_mean is state_dim × action_dim.
        """
        with self._lock:
            env = self._require_linear_gaussian(env_id)
            d_s = env.spec.state_dim
            d_a = env.spec.action_dim
            W = _solve_psd(env.Lambda, env.Lmu)  # (d_s + d_a) × d_s
            # W rows are [A^T | B^T]^T; build A (d_s × d_s) and B (d_s × d_a)
            A = [[0.0] * d_s for _ in range(d_s)]
            B = [[0.0] * d_a for _ in range(d_s)]
            for j in range(d_s):
                for i in range(d_s):
                    A[j][i] = W[i][j]
                for i in range(d_a):
                    B[j][i] = W[d_s + i][j]
            return A, B

    def moment_rollout(
        self,
        env_id: str,
        *,
        state: Sequence[float],
        policy: Callable[[Sequence[float]], Sequence[float]],
        horizon: int,
        noise: Sequence[Sequence[float]] | None = None,
    ) -> list[tuple[list[float], list[list[float]]]]:
        """Closed-form linear-Gaussian moment rollout.

        Propagate ``N(μ_0=state, Σ_0=0)`` through ``h`` steps under
        the policy and posterior-mean dynamics.  Returns
        ``[(μ_h, Σ_h)]`` for h = 1..horizon.
        """
        with self._lock:
            env = self._require_linear_gaussian(env_id)
            d_s = env.spec.state_dim
            if len(state) != d_s:
                raise InvalidObservation("state shape mismatch")
            A, B = self.posterior_mean_dynamics(env_id)
            noise_cov = (
                noise if noise is not None else [[0.01 if i == j else 0.0 for j in range(d_s)] for i in range(d_s)]
            )
        mu: list[float] = list(state)
        cov: list[list[float]] = [[0.0] * d_s for _ in range(d_s)]
        trace: list[tuple[list[float], list[list[float]]]] = []
        for _ in range(horizon):
            a = list(policy(mu))
            mu_next = [sum(A[i][j] * mu[j] for j in range(d_s)) + sum(B[i][k] * a[k] for k in range(len(a))) for i in range(d_s)]
            # Σ_next = A Σ Aᵀ + Q
            AS = [[sum(A[i][k] * cov[k][j] for k in range(d_s)) for j in range(d_s)] for i in range(d_s)]
            cov_next = [
                [
                    sum(AS[i][k] * A[j][k] for k in range(d_s)) + noise_cov[i][j]
                    for j in range(d_s)
                ]
                for i in range(d_s)
            ]
            mu = mu_next
            cov = cov_next
            trace.append((list(mu), [row[:] for row in cov]))
        return trace

    # ------------------------------------------------------------------
    # State checks + accessors
    # ------------------------------------------------------------------

    def _require_env(
        self, env_id: str
    ) -> _CategoricalEnvState | _LinearGaussianEnvState:
        if env_id not in self._envs:
            raise UnknownEnv(env_id)
        return self._envs[env_id]

    def _require_categorical(self, env_id: str) -> _CategoricalEnvState:
        env = self._require_env(env_id)
        if not isinstance(env, _CategoricalEnvState):
            raise ImaginatorError(
                f"env {env_id!r} is not categorical"
            )
        return env

    def _require_linear_gaussian(self, env_id: str) -> _LinearGaussianEnvState:
        env = self._require_env(env_id)
        if not isinstance(env, _LinearGaussianEnvState):
            raise ImaginatorError(
                f"env {env_id!r} is not linear_gaussian"
            )
        return env

    def _check_sa(
        self, env: _CategoricalEnvState, state: str, action: str
    ) -> None:
        if state not in env.spec.states:
            raise InvalidObservation(f"unknown state {state!r}")
        if action not in env.spec.actions:
            raise InvalidObservation(f"unknown action {action!r}")

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the current state.

        ``import_state(export_state())`` round-trips a fresh
        Imaginator to the same observation count, chain head, and
        per-env posteriors.
        """
        with self._lock:
            envs_out: dict[str, Any] = {}
            for env_id, env in self._envs.items():
                if isinstance(env, _CategoricalEnvState):
                    envs_out[env_id] = {
                        "family": FAMILY_CATEGORICAL,
                        "spec": {
                            "env_id": env.spec.env_id,
                            "states": list(env.spec.states),
                            "actions": list(env.spec.actions),
                        },
                        "alpha": {
                            f"{s}|{a}": vals for (s, a), vals in env.alpha.items()
                        },
                        "n_sa": {
                            f"{s}|{a}": n for (s, a), n in env.n_sa.items()
                        },
                        "sum_r": {
                            f"{s}|{a}": v for (s, a), v in env.sum_r.items()
                        },
                        "sum_rr": {
                            f"{s}|{a}": v for (s, a), v in env.sum_rr.items()
                        },
                        "pit": list(env.pit),
                        "n_total": env.n_total,
                    }
                else:
                    envs_out[env_id] = {
                        "family": FAMILY_LINEAR_GAUSSIAN,
                        "spec": {
                            "env_id": env.spec.env_id,
                            "state_dim": env.spec.state_dim,
                            "action_dim": env.spec.action_dim,
                        },
                        "Lambda": env.Lambda,
                        "Lmu": env.Lmu,
                        "SSE": env.SSE,
                        "n": env.n,
                        "R_Lambda": env.R_Lambda,
                        "R_Lmu": env.R_Lmu,
                        "R_sse": env.R_sse,
                        "R_n": env.R_n,
                    }
            return {
                "chain_head": self._chain_head,
                "envs": envs_out,
            }

    def import_state(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._envs.clear()
            for env_id, env_data in snapshot.get("envs", {}).items():
                if env_data["family"] == FAMILY_CATEGORICAL:
                    spec = EnvSpec(
                        env_id=env_id,
                        family=FAMILY_CATEGORICAL,
                        states=tuple(env_data["spec"]["states"]),
                        actions=tuple(env_data["spec"]["actions"]),
                    )
                    st = _CategoricalEnvState(spec=spec)
                    st.alpha = {
                        tuple(k.split("|", 1)): list(v)
                        for k, v in env_data["alpha"].items()
                    }
                    st.n_sa = {
                        tuple(k.split("|", 1)): int(v)
                        for k, v in env_data["n_sa"].items()
                    }
                    st.sum_r = {
                        tuple(k.split("|", 1)): float(v)
                        for k, v in env_data["sum_r"].items()
                    }
                    st.sum_rr = {
                        tuple(k.split("|", 1)): float(v)
                        for k, v in env_data["sum_rr"].items()
                    }
                    st.pit = list(env_data.get("pit", []))
                    st.n_total = int(env_data.get("n_total", 0))
                    self._envs[env_id] = st
                else:
                    spec = EnvSpec(
                        env_id=env_id,
                        family=FAMILY_LINEAR_GAUSSIAN,
                        state_dim=int(env_data["spec"]["state_dim"]),
                        action_dim=int(env_data["spec"]["action_dim"]),
                    )
                    st = _LinearGaussianEnvState(spec=spec)
                    st.Lambda = [list(row) for row in env_data["Lambda"]]
                    st.Lmu = [list(row) for row in env_data["Lmu"]]
                    st.SSE = [list(row) for row in env_data["SSE"]]
                    st.n = int(env_data["n"])
                    st.R_Lambda = [list(row) for row in env_data["R_Lambda"]]
                    st.R_Lmu = list(env_data["R_Lmu"])
                    st.R_sse = float(env_data["R_sse"])
                    st.R_n = int(env_data["R_n"])
                    self._envs[env_id] = st
            self._chain_head = snapshot.get(
                "chain_head", ledger_root(self.config.hmac_key)
            )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _sample_categorical(rng: random.Random, probs: Sequence[float]) -> int:
    u = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def _student_t_sample(rng: random.Random, df: float) -> float:
    """Draw a Student-t sample with ``df`` degrees of freedom via the
    Bailey 1994 algorithm (acceptance-rejection from a normal).

    Equivalent to: Z / √(V / df), Z ~ N(0,1), V ~ χ²(df).
    """
    z = rng.gauss(0.0, 1.0)
    # χ²(df) via gamma: G = sum of gammas with shape df/2, scale 2.
    if df <= 0.0:
        return z
    v = rng.gammavariate(df / 2.0, 2.0)
    if v <= 0.0:
        return z
    return z / math.sqrt(v / df)


def _empirical_quantiles(
    samples: Sequence[float], qs: Sequence[float]
) -> dict[float, float]:
    if not samples:
        return {q: 0.0 for q in qs}
    sorted_s = sorted(samples)
    out: dict[float, float] = {}
    n = len(sorted_s)
    for q in qs:
        idx = int(q * (n - 1))
        # Linear interpolation between adjacent values.
        lo = math.floor(q * (n - 1))
        hi = math.ceil(q * (n - 1))
        frac = q * (n - 1) - lo
        out[q] = sorted_s[lo] * (1.0 - frac) + sorted_s[hi] * frac
    return out


def _canonical_value(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_canonical_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canonical_value(val) for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return str(v)


def _solve_psd(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Solve A X = B for X where A is symmetric positive-definite via
    Cholesky decomposition.  Pure Python; O(n³).
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ImaginatorError("A must be square")
    if len(B) != n:
        raise ImaginatorError("B row count must match A")
    m = len(B[0]) if B else 0
    # Cholesky factor L such that L Lᵀ = A.
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    # Add a tiny ridge to keep the matrix PD.
                    s = max(s, 1e-12)
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    # Solve L Y = B by forward substitution.
    Y = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for col in range(m):
            s = B[i][col] - sum(L[i][k] * Y[k][col] for k in range(i))
            Y[i][col] = s / L[i][i]
    # Solve Lᵀ X = Y by backward substitution.
    X = [[0.0] * m for _ in range(n)]
    for i in reversed(range(n)):
        for col in range(m):
            s = Y[i][col] - sum(L[k][i] * X[k][col] for k in range(i + 1, n))
            X[i][col] = s / L[i][i]
    return X


# ---------------------------------------------------------------------------
# Regularised incomplete beta (needed by Student-t CDF for PIT)
# ---------------------------------------------------------------------------


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) for the Student-t CDF.

    Lentz continued-fraction kernel; same algorithm Numerical Recipes
    (Press et al. 1992 §6.4) attributes to Thompson 1968.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Symmetric reflection when x > (a+1)/(a+b+2).
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(b, a, 1.0 - x)
    # log B(a, b) via lgamma
    lbeta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(lbeta)
    return bt * _beta_cf(a, b, x) / a


def _beta_cf(a: float, b: float, x: float, max_iter: int = 500, tiny: float = 1e-30) -> float:
    """Lentz continued fraction for the regularised incomplete beta."""
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
        m2 = 2 * m
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
            break
    return h
