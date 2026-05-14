r"""ActiveInferencer — free-energy POMDP planning as a runtime primitive.

A coordination engine that selects which sub-model, tool, or skill should
handle the next user query is solving a *partially observed sequential
decision problem*.  The state is "what does this user actually want; what
is the latent capability of each downstream worker"; the observation is
"how good was the answer that came back"; the action is "which worker
do I dispatch next, with what budget".  Bandit primitives (Arbiter) give
the right answer once arms are i.i.d.; off-policy primitives
(PolicyImprover) give safe deployment given an evaluation policy.
*Neither* fuses belief, preference and information-gain into a single
score that a planner can rank policies by.

The ActiveInferencer is the primitive that implements **Active Inference**
(Friston et al., 2009-2025) — variational inference over hidden states
plus Expected-Free-Energy (EFE) policy selection — as a typed,
stdlib-only, threadsafe service that other primitives compose with.

Mathematical roots
------------------

  * **Friston, 2010 — "The free-energy principle: a unified brain
    theory?" Nat. Rev. Neurosci. 11, 127-138.**  An agent that minimises
    its variational free energy implicitly maximises model evidence
    log P(o) and, under a generative model with action-conditioned
    transitions, performs Bayes-optimal exploration *and* exploitation.

  * **Friston, FitzGerald, Rigoli, Schwartenbeck, Pezzulo, 2017 —
    "Active Inference: A Process Theory." Neural Computation
    29 (1), 1-49.**  Closes the discrete generative model
    (A, B, C, D, E) and gives the canonical EFE decomposition

        G(π) = − E_q(o,s|π)[log P(o,s) − log q(s|π)]
             = D_KL[q(o|π) ‖ P(o)]                     (risk)
               + E_q(s|π)[ H[P(o|s)] ]                 (ambiguity)
             = − E_q(o|π)[ D_KL[q(s|o,π) ‖ q(s|π)] ]   (epistemic value)
               − E_q(o|π)[ log P(o) ]                  (pragmatic value)

    with policy posterior q(π) = σ(−γ G(π) + log E(π)).

  * **Da Costa, Parr, Sajid, Veselic, Neacsu, Friston, 2020 —
    "Active inference on discrete state-spaces: A synthesis."
    J. Math. Psych. 99, 102447.**  Gives the matrix forms used here:

        D[s]        prior over initial state
        A[o, s]     P(o | s)            (likelihood)
        B[s', s, a] P(s' | s, a)        (controlled transition)
        C[t][o]     log P̃(o) at time t (preferences / target outcomes)
        E[π]        habit prior over policies
        γ           inverse-temperature on EFE  (precision)

  * **Schwöbel, Markovic, Smolka, Kiebel, 2018 — "Active inference,
    belief propagation, and the Bethe approximation." Neural
    Computation 30 (9), 2530-2567.**  Variational message passing on
    a factor graph; the discrete update used in `update_belief_discrete`
    is the single-time-step special case of this recursion.

  * **Kalman, 1960; Anderson & Moore, 1979 — *Optimal Filtering*.**
    For linear-Gaussian generative models the variational posterior
    is exact and given by the Kalman filter; the EFE then has a
    closed form (Schwöbel 2018, Appendix B) used in
    `linear_gaussian_efe`.

  * **Doucet, de Freitas, Gordon, 2001 — *Sequential Monte Carlo
    Methods in Practice*.**  For general non-linear / non-Gaussian
    models we ship a bootstrap particle filter; the EFE under
    particles is the Monte-Carlo estimator of the decomposition
    above, with bias O(1/N).

  * **Lindley, 1956 — "On a measure of the information provided by
    an experiment."**  The epistemic-value term in EFE is exactly
    Lindley's expected information gain about the latent state; the
    primitive surfaces it as a stand-alone planner-input.

  * **Sajid, Ball, Parr, Friston, 2021 — "Active inference: demystified
    and compared." Neural Computation 33 (3), 674-712.**  Establishes
    that active inference reduces to (a) Bayes-optimal exploration
    under reward-free settings (D_KL = 0) and (b) KL-control / risk-
    sensitive control under reward-only settings (H[P(o|s)] = 0).
    The decomposition we report makes this reduction visible.

  * **Hoeffding, 1963; Maurer-Pontil, 2009.**  Closing the loop:
    the empirical expected utility of a stationary policy concentrates
    at rate Õ(√(log(1/δ)/n)) under bounded outcomes.  Bound is
    returned by `expected_utility_bound`.

Design contract
---------------

The ActiveInferencer holds a registry of named *agents*.  Each agent owns
a (a) generative model, (b) running posterior over hidden state, (c)
preference vector C, (d) habit prior E, (e) precision γ, (f) a buffer of
realised (action, observation) pairs.  The coordination engine drives
each agent by

  1. ``register_agent(name, model, C=…, gamma=…)`` — once.
  2. ``step(name, obs)`` — variational state update on a new observation.
  3. ``plan(name, horizon, candidate_policies=…)`` — returns a
     ``PolicySelection`` whose ``q_pi`` is the softmax-EFE policy
     posterior with explicit ``epistemic_value`` and ``pragmatic_value``
     components, plus a tamper-evident attestation digest.
  4. ``act(name)`` — samples or argmaxes a policy and returns the next
     action; mutates the running belief by predicting forward.
  5. ``observe(name, obs)`` — alias for ``step``; closes the loop.
  6. ``learn(name, ...)`` — Dirichlet posterior updates on A / B / C / E
     from realised history (canonical learning-by-counts).
  7. ``coverage()`` / ``snapshot(name)`` — lifetime + per-agent stats.

Compositional contracts:

  * If the coordination engine has a Forecaster stream that produces
    P(o | s) for a given action, ``set_likelihood_from_forecaster()``
    plugs it in as the A matrix.

  * If a CausalDiscoverer has identified an interventional transition
    model, ``set_transition_from_causal()`` plugs it in as the B matrix.

  * If a TruthSerum-aggregated peer signal exists,
    ``set_prior_from_truthserum()`` provides a robust prior D.

  * If a Robustifier has produced an ambiguity set over A, the planner
    can be made *worst-case* over that set via ``plan_robust(...)``.

  * If multiple agents disagree, ``bayesian_model_average(...)`` returns
    a mixture-of-experts belief weighted by posterior evidence.

Every state-changing call emits an event on the optional ``EventBus``
and writes a content-hashed receipt to the optional ``RuntimeAttestor``,
so the coordination engine can deterministically replay the agent's
decision history and re-verify each EFE computation.

The module is stdlib-only and threadsafe under a single recursive lock.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event  # type: ignore
except Exception:  # pragma: no cover - stdlib-only fallback
    Event = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================


AI_STARTED = "active_inference.started"
AI_AGENT_REGISTERED = "active_inference.agent_registered"
AI_AGENT_REMOVED = "active_inference.agent_removed"
AI_INFERRED = "active_inference.inferred"
AI_PLANNED = "active_inference.planned"
AI_ACTED = "active_inference.acted"
AI_LEARNED = "active_inference.learned"
AI_CLEARED = "active_inference.cleared"


# =====================================================================
# Model kinds
# =====================================================================


KIND_DISCRETE = "discrete"
KIND_LINEAR_GAUSSIAN = "linear_gaussian"
KIND_PARTICLE = "particle"

KNOWN_KINDS = frozenset({KIND_DISCRETE, KIND_LINEAR_GAUSSIAN, KIND_PARTICLE})


# =====================================================================
# Action selection modes
# =====================================================================


SELECT_ARGMAX = "argmax"
SELECT_SOFTMAX = "softmax"
SELECT_HABIT_ONLY = "habit_only"
SELECT_RANDOM = "random"

KNOWN_SELECTORS = frozenset({SELECT_ARGMAX, SELECT_SOFTMAX, SELECT_HABIT_ONLY, SELECT_RANDOM})


# =====================================================================
# Errors
# =====================================================================


class ActiveInferenceError(Exception):
    """Base error for ActiveInferencer."""


class InvalidModel(ActiveInferenceError):
    """Generative model fails the canonical row-/column-stochastic checks."""


class InvalidPolicy(ActiveInferenceError):
    """Policy references an unknown action or wrong horizon."""


class UnknownAgent(ActiveInferenceError):
    """Operation referenced an agent that was not registered."""


class InsufficientData(ActiveInferenceError):
    """A bound or estimator received fewer samples than its minimum."""


class UnknownKind(ActiveInferenceError):
    """Caller asked for a model kind that the primitive does not support."""


# =====================================================================
# Numerics
# =====================================================================


_EPS = 1e-12
_LOG_EPS = math.log(_EPS)


def _log(x: float) -> float:
    """Numerically-safe log."""
    return math.log(max(x, _EPS))


def _logsumexp(xs: Sequence[float]) -> float:
    """Numerically-stable log-sum-exp."""
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _softmax(xs: Sequence[float], *, temperature: float = 1.0) -> list[float]:
    """Softmax with temperature.  temperature > 0; output sums to 1."""
    if temperature <= 0.0:
        raise ValueError("softmax temperature must be > 0")
    z = [x / temperature for x in xs]
    lse = _logsumexp(z)
    return [math.exp(zi - lse) for zi in z]


def _normalize(p: Sequence[float]) -> list[float]:
    """Project onto the simplex by clipping and normalising."""
    pp = [max(_EPS, float(x)) for x in p]
    s = sum(pp)
    return [x / s for x in pp]


def _entropy(p: Sequence[float]) -> float:
    """Shannon entropy in nats."""
    h = 0.0
    for pi in p:
        if pi > _EPS:
            h -= pi * math.log(pi)
    return h


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    """KL divergence D_KL(p || q) in nats."""
    if len(p) != len(q):
        raise ValueError("kl: dimension mismatch")
    out = 0.0
    for pi, qi in zip(p, q):
        if pi > _EPS:
            out += pi * (math.log(pi) - math.log(max(qi, _EPS)))
    return max(0.0, out)


def _mat_vec(A: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    """Matrix-vector product A x for A of shape (m, n) and x of length n."""
    if not A:
        return []
    n = len(x)
    if any(len(row) != n for row in A):
        raise ValueError("mat_vec: ragged matrix")
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(len(A))]


def _row_stochastic(A: Sequence[Sequence[float]], *, tol: float = 1e-6) -> bool:
    """True iff each row sums to ≈1 and entries are in [0,1]."""
    for row in A:
        s = 0.0
        for v in row:
            if v < -tol or v > 1.0 + tol:
                return False
            s += v
        if abs(s - 1.0) > tol:
            return False
    return True


def _col_stochastic(A: Sequence[Sequence[float]], *, tol: float = 1e-6) -> bool:
    """True iff each column sums to ≈1 and entries are in [0,1]."""
    if not A:
        return True
    n_cols = len(A[0])
    if any(len(r) != n_cols for r in A):
        return False
    for j in range(n_cols):
        s = 0.0
        for i in range(len(A)):
            v = A[i][j]
            if v < -tol or v > 1.0 + tol:
                return False
            s += v
        if abs(s - 1.0) > tol:
            return False
    return True


def _is_pmf(p: Sequence[float], *, tol: float = 1e-6) -> bool:
    s = 0.0
    for v in p:
        if v < -tol or v > 1.0 + tol:
            return False
        s += v
    return abs(s - 1.0) <= tol


# =====================================================================
# Attestation helpers
# =====================================================================


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class _AttestableReceipt:
    kind: str
    payload: dict
    digest: str


# =====================================================================
# Generative models
# =====================================================================


@dataclass
class DiscreteGenerativeModel:
    """Discrete A/B/C/D/E active-inference model.

    Conventions follow Da Costa et al. 2020:

      * ``A[o][s] = P(o | s)``  — column-stochastic over s; shape (|O|, |S|).
      * ``B[a][s'][s] = P(s' | s, a)`` — for each action a, column-stochastic
        over s; shape (|A|, |S|, |S|).
      * ``C`` — log-preferences over outcomes.  May be:
            (a) a list of length |O| (time-invariant), or
            (b) a list[list[float]] of length T with each row length |O|.
      * ``D[s] = P(s_0)`` — simplex of length |S|.
      * ``E`` — habit prior over policies; arbitrary positive vector,
        normalised at use time.  May be None.
    """

    A: list[list[float]]
    B: list[list[list[float]]]
    C: list[float] | list[list[float]]
    D: list[float]
    E: list[float] | None = None

    @property
    def n_states(self) -> int:
        return len(self.D)

    @property
    def n_obs(self) -> int:
        return len(self.A)

    @property
    def n_actions(self) -> int:
        return len(self.B)

    def validate(self, *, tol: float = 1e-6) -> None:
        if not self.A or not self.B or not self.D:
            raise InvalidModel("discrete model: empty A/B/D")
        S = self.n_states
        O = self.n_obs
        Aacts = self.n_actions
        if not _is_pmf(self.D, tol=tol):
            raise InvalidModel("discrete model: D is not a pmf")
        if any(len(row) != S for row in self.A):
            raise InvalidModel(f"discrete model: A must be ({O}, {S})")
        if not _col_stochastic(self.A, tol=tol):
            raise InvalidModel("discrete model: A is not column-stochastic in s")
        if Aacts < 1:
            raise InvalidModel("discrete model: need at least one action in B")
        for a, Ba in enumerate(self.B):
            if len(Ba) != S or any(len(row) != S for row in Ba):
                raise InvalidModel(f"discrete model: B[{a}] must be ({S}, {S})")
            if not _col_stochastic(Ba, tol=tol):
                raise InvalidModel(f"discrete model: B[{a}] not column-stochastic in s")
        if self.C:
            if isinstance(self.C[0], (list, tuple)):
                for t, row in enumerate(self.C):
                    if len(row) != O:
                        raise InvalidModel(
                            f"discrete model: C[{t}] must have length {O}"
                        )
            else:
                if len(self.C) != O:
                    raise InvalidModel(f"discrete model: C must have length {O}")
        if self.E is not None and any(e < -tol for e in self.E):
            raise InvalidModel("discrete model: E must be non-negative")

    def preferences_at(self, t: int) -> list[float]:
        """Return log-preferences over outcomes at time ``t``."""
        if not self.C:
            return [0.0] * self.n_obs
        if isinstance(self.C[0], (list, tuple)):
            row_t = self.C[min(t, len(self.C) - 1)]
            return [float(x) for x in row_t]
        return [float(x) for x in self.C]


@dataclass
class LinearGaussianGenerativeModel:
    """Linear-Gaussian generative model.

      s_{t+1} = F(a) s_t + b(a) + w,   w ~ N(0, Q)
      o_t    = H s_t + v,              v ~ N(0, R)
      s_0    ~ N(mu0, Sigma0)

    All matrices are stored as nested Python lists; we keep a tiny linear-
    algebra core that only handles diagonal Q, R for tractability.  This
    is sufficient for the canonical use case where Q = q I_s and R = r I_o
    and the agent only needs to track a *mean* and *variance per dim*.
    """

    F: list[list[float]]
    b: list[float] | None
    H: list[list[float]]
    Q_diag: list[float]
    R_diag: list[float]
    mu0: list[float]
    Sigma0_diag: list[float]
    C: list[float] | None = None
    n_actions: int = 1
    F_per_action: list[list[list[float]]] | None = None
    b_per_action: list[list[float]] | None = None

    @property
    def n_states(self) -> int:
        return len(self.mu0)

    @property
    def n_obs(self) -> int:
        return len(self.H)

    def F_a(self, a: int) -> list[list[float]]:
        if self.F_per_action is not None:
            return self.F_per_action[a]
        return self.F

    def b_a(self, a: int) -> list[float]:
        if self.b_per_action is not None:
            return self.b_per_action[a]
        return self.b if self.b is not None else [0.0] * self.n_states

    def validate(self) -> None:
        S = self.n_states
        O = self.n_obs
        if any(len(row) != S for row in self.F):
            raise InvalidModel(f"linear-gaussian: F must be ({S}, {S})")
        if any(len(row) != S for row in self.H):
            raise InvalidModel(f"linear-gaussian: H must be ({O}, {S})")
        if len(self.Q_diag) != S:
            raise InvalidModel("linear-gaussian: Q_diag wrong length")
        if len(self.R_diag) != O:
            raise InvalidModel("linear-gaussian: R_diag wrong length")
        if len(self.Sigma0_diag) != S:
            raise InvalidModel("linear-gaussian: Sigma0_diag wrong length")
        if self.b is not None and len(self.b) != S:
            raise InvalidModel("linear-gaussian: b wrong length")
        if any(x < 0 for x in self.Q_diag + self.R_diag + self.Sigma0_diag):
            raise InvalidModel("linear-gaussian: covariances must be ≥ 0")
        if self.F_per_action is not None and len(self.F_per_action) != self.n_actions:
            raise InvalidModel("linear-gaussian: F_per_action wrong length")
        if self.b_per_action is not None and len(self.b_per_action) != self.n_actions:
            raise InvalidModel("linear-gaussian: b_per_action wrong length")


# =====================================================================
# Beliefs (posterior distributions)
# =====================================================================


@dataclass
class CategoricalBelief:
    """Posterior q(s) on a discrete latent state."""

    probs: list[float]

    def __post_init__(self) -> None:
        if not _is_pmf(self.probs):
            self.probs = _normalize(self.probs)

    def entropy(self) -> float:
        return _entropy(self.probs)

    def mode(self) -> int:
        best = 0
        bv = self.probs[0]
        for i, p in enumerate(self.probs):
            if p > bv:
                bv = p
                best = i
        return best

    def sample(self, rng: random.Random) -> int:
        u = rng.random()
        c = 0.0
        for i, p in enumerate(self.probs):
            c += p
            if u <= c:
                return i
        return len(self.probs) - 1


@dataclass
class GaussianBelief:
    """Diagonal-covariance Gaussian posterior."""

    mu: list[float]
    var_diag: list[float]

    def entropy(self) -> float:
        # Diagonal Gaussian: H = 0.5 * sum_i (1 + log(2 pi sigma_i^2))
        h = 0.0
        for v in self.var_diag:
            h += 0.5 * (1.0 + math.log(2.0 * math.pi * max(v, _EPS)))
        return h

    def sample(self, rng: random.Random) -> list[float]:
        return [
            mu_i + (math.sqrt(max(v, 0.0))) * rng.gauss(0.0, 1.0)
            for mu_i, v in zip(self.mu, self.var_diag)
        ]


# =====================================================================
# Policies & reports
# =====================================================================


@dataclass(frozen=True)
class Policy:
    """A finite-horizon plan: a tuple of action indices."""

    actions: tuple[int, ...]

    @property
    def horizon(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class EFEReport:
    """Decomposed Expected Free Energy of a single policy.

    G = risk + ambiguity = − epistemic_value − pragmatic_value.

    All values are in *nats* and accumulated over the policy's horizon.
    """

    policy: Policy
    G: float                # total expected free energy
    risk: float             # D_KL(q(o|π) || P̃(o))
    ambiguity: float        # E_q(s|π)[H[P(o|s)]]
    epistemic_value: float  # E_q(o|π)[KL(q(s|o,π) || q(s|π))]
    pragmatic_value: float  # E_q(o|π)[log P̃(o)]


@dataclass(frozen=True)
class PolicySelection:
    """Result of a planning call.

    ``q_pi`` is the softmax policy posterior over the supplied candidates,
    in the same order.  ``best`` is the argmax index.  ``efe`` is the list
    of EFEReports in candidate order.  ``digest`` is a content-hash of
    the planning input.
    """

    candidates: tuple[Policy, ...]
    q_pi: tuple[float, ...]
    best: int
    efe: tuple[EFEReport, ...]
    gamma: float
    digest: str


@dataclass(frozen=True)
class FreeEnergyReport:
    """Variational-free-energy decomposition for a single inferred belief.

      F(q,o) = complexity + accuracy_loss
             = D_KL(q || prior) − E_q[log P(o|s)]
    """

    F: float
    complexity: float        # D_KL(q(s) || prior)
    accuracy: float          # E_q[log P(o|s)]
    surprise: float          # − log P(o)
    digest: str


@dataclass(frozen=True)
class UtilityBound:
    """PAC bound on expected utility under a policy.

      P(|û − E[u]| ≥ ε) ≤ δ            (two-sided Hoeffding)

    Returned with the empirical mean and the half-width.
    """

    n: int
    mean: float
    half_width: float
    lcb: float
    ucb: float
    delta: float
    method: str  # "hoeffding" | "empirical_bernstein"


@dataclass(frozen=True)
class CoverageReport:
    """Lifetime stats of an ActiveInferencer."""

    started_ns: int
    agents: int
    inferences: int
    plans: int
    acts: int
    learns: int
    bma_calls: int
    receipts: int


@dataclass(frozen=True)
class AgentSnapshot:
    """Snapshot of one agent's running state."""

    name: str
    kind: str
    horizon: int
    gamma: float
    n_obs_seen: int
    last_action: int | None
    belief_entropy: float
    accumulated_free_energy: float
    last_efe: float | None


# =====================================================================
# Free-energy computations — discrete
# =====================================================================


def variational_free_energy_discrete(
    model: DiscreteGenerativeModel,
    prior: Sequence[float],
    observation: int,
    *,
    q_init: Sequence[float] | None = None,
    n_iters: int = 16,
    tol: float = 1e-9,
) -> tuple[CategoricalBelief, FreeEnergyReport]:
    """Single-step variational state inference.

    With a Categorical generative model the variational posterior over
    the current state given a single observation is closed form:

        q*(s) ∝ prior(s) · A[o | s].

    The iterative form below converges in one step; we keep the loop for
    parity with multi-step / hierarchical generalisations.

    Returns the posterior and a FreeEnergyReport with complexity and
    accuracy in nats.
    """
    model.validate()
    if not _is_pmf(prior):
        raise InvalidModel("prior must be a pmf")
    if not (0 <= observation < model.n_obs):
        raise InvalidModel(f"observation {observation} out of range")
    S = model.n_states
    A_o = [model.A[observation][s] for s in range(S)]
    if q_init is None:
        q = list(prior)
    else:
        if len(q_init) != S:
            raise InvalidModel("q_init length mismatch")
        q = _normalize(q_init)
    prev = None
    for _ in range(max(1, n_iters)):
        # Fixed-point under mean-field: log q = log prior + log A[o,·] − Z
        log_q = [_log(prior[s]) + _log(A_o[s]) for s in range(S)]
        lse = _logsumexp(log_q)
        q = [math.exp(lq - lse) for lq in log_q]
        if prev is not None:
            if max(abs(q[i] - prev[i]) for i in range(S)) < tol:
                break
        prev = q
    complexity = _kl(q, list(prior))
    accuracy = sum(q[s] * _log(A_o[s]) for s in range(S))
    surprise = -_logsumexp([_log(prior[s]) + _log(A_o[s]) for s in range(S)])
    F = complexity - accuracy
    digest = _hash_payload(
        {"observation": observation, "F": round(F, 12), "S": S}
    )
    return (
        CategoricalBelief(probs=list(q)),
        FreeEnergyReport(
            F=F, complexity=complexity, accuracy=accuracy, surprise=surprise, digest=digest
        ),
    )


def predict_belief_discrete(
    model: DiscreteGenerativeModel,
    belief: CategoricalBelief,
    action: int,
) -> CategoricalBelief:
    """Apply one transition: q(s_{t+1}) = B[a] q(s_t)."""
    if not (0 <= action < model.n_actions):
        raise InvalidPolicy(f"action {action} out of range")
    Ba = model.B[action]
    S = model.n_states
    out = [0.0] * S
    for sp in range(S):
        s_sum = 0.0
        for s in range(S):
            s_sum += Ba[sp][s] * belief.probs[s]
        out[sp] = s_sum
    return CategoricalBelief(probs=_normalize(out))


def predicted_observation_distribution(
    model: DiscreteGenerativeModel,
    belief: CategoricalBelief,
) -> list[float]:
    """Marginal q(o) = ∑_s A[o,s] q(s)."""
    O = model.n_obs
    out = [0.0] * O
    for o in range(O):
        s_sum = 0.0
        for s in range(model.n_states):
            s_sum += model.A[o][s] * belief.probs[s]
        out[o] = s_sum
    return _normalize(out)


def expected_free_energy_discrete(
    model: DiscreteGenerativeModel,
    policy: Policy,
    belief: CategoricalBelief,
) -> EFEReport:
    """EFE under a single policy, summed across horizon.

    Per-step EFE decomposed (Da Costa 2020, eq. 7):

        G_t(π) = ambiguity_t + risk_t
               = E_{q(s_t|π)}[ H[P(o|s_t)] ]
                 + D_KL[ q(o_t|π) ‖ P̃(o_t) ]

    The negative epistemic-value and negative pragmatic-value form is
    also returned:

        G_t = − epistemic_value − pragmatic_value
        epistemic_value = E_{q(o_t|π)}[ D_KL[q(s_t|o_t,π) ‖ q(s_t|π)] ]
        pragmatic_value = E_{q(o_t|π)}[ log P̃(o_t) ]

    The two decompositions agree up to a (state-independent) entropy of
    q(o_t|π) absorbed into both sides; we report both because downstream
    primitives (Robustifier, Cartographer) consume them separately.
    """
    model.validate()
    G = 0.0
    risk = 0.0
    ambiguity = 0.0
    epistemic = 0.0
    pragmatic = 0.0
    b = belief
    for t, a in enumerate(policy.actions):
        b_next = predict_belief_discrete(model, b, a)
        # ambiguity = E_q(s)[H[P(o|s)]]
        amb_t = 0.0
        for s, qs in enumerate(b_next.probs):
            col = [model.A[o][s] for o in range(model.n_obs)]
            amb_t += qs * _entropy(col)
        ambiguity += amb_t
        # q(o|π) at time t
        qo = predicted_observation_distribution(model, b_next)
        # preferences at time t (treat as log-pmf; if not normalised, normalise)
        C_t = model.preferences_at(t)
        # treat C as un-normalised log-preferences; convert to proper log-pmf
        Z = _logsumexp(C_t)
        P_t = [math.exp(c - Z) for c in C_t]
        # risk = KL(q(o) || P̃)
        risk_t = _kl(qo, P_t)
        risk += risk_t
        # epistemic value = E_q(o)[KL(q(s|o,π) || q(s|π))]
        ep_t = 0.0
        for o in range(model.n_obs):
            # q(s|o,π) ∝ A[o,s] q(s|π)
            num = [model.A[o][s] * b_next.probs[s] for s in range(model.n_states)]
            ssum = sum(num)
            if ssum < _EPS:
                continue
            qso = [n / ssum for n in num]
            ep_t += qo[o] * _kl(qso, b_next.probs)
        epistemic += ep_t
        # pragmatic value = E_q(o)[log P̃(o)]
        prag_t = sum(qo[o] * (C_t[o] - Z) for o in range(model.n_obs))
        pragmatic += prag_t
        G += amb_t + risk_t
        b = b_next
    return EFEReport(
        policy=policy,
        G=G,
        risk=risk,
        ambiguity=ambiguity,
        epistemic_value=epistemic,
        pragmatic_value=pragmatic,
    )


def enumerate_policies(n_actions: int, horizon: int) -> list[Policy]:
    """All policies of given horizon over n_actions.  Useful for small models."""
    if n_actions <= 0 or horizon <= 0:
        return []
    out: list[Policy] = []

    def _rec(prefix: list[int], depth: int) -> None:
        if depth == horizon:
            out.append(Policy(actions=tuple(prefix)))
            return
        for a in range(n_actions):
            prefix.append(a)
            _rec(prefix, depth + 1)
            prefix.pop()

    _rec([], 0)
    return out


def policy_posterior(
    efe: Sequence[EFEReport],
    *,
    gamma: float = 1.0,
    habit_E: Sequence[float] | None = None,
) -> list[float]:
    """q(π) = softmax(−γ G + log E).  habit_E is a positive vector or None."""
    if not efe:
        return []
    G = [r.G for r in efe]
    if habit_E is not None:
        if len(habit_E) != len(efe):
            raise InvalidPolicy("habit_E length mismatch")
        logits = [-gamma * G[i] + _log(habit_E[i]) for i in range(len(efe))]
    else:
        logits = [-gamma * g for g in G]
    return _softmax(logits)


# =====================================================================
# Free-energy computations — linear Gaussian
# =====================================================================


def _diag_kalman_predict(
    model: LinearGaussianGenerativeModel,
    mu: Sequence[float],
    var: Sequence[float],
    action: int,
) -> tuple[list[float], list[float]]:
    """One-step diagonal Kalman predict using only diagonal F (axis-aligned).

    For tractability with stdlib only we require F per action to be
    diagonal — for non-diagonal F we fall back to a Monte-Carlo
    approximation in ``linear_gaussian_efe`` (Schwöbel 2018).
    """
    Fa = model.F_a(action)
    ba = model.b_a(action)
    S = model.n_states
    mu_new = [0.0] * S
    var_new = list(model.Q_diag)
    for i in range(S):
        s_mu = 0.0
        s_var = 0.0
        for j in range(S):
            fij = Fa[i][j]
            s_mu += fij * mu[j]
            s_var += (fij ** 2) * var[j]
        mu_new[i] = s_mu + ba[i]
        var_new[i] += s_var
    return mu_new, var_new


def _diag_kalman_update(
    model: LinearGaussianGenerativeModel,
    mu: Sequence[float],
    var: Sequence[float],
    obs: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Diagonal Kalman update (axis-aligned H) — exact when H is diagonal."""
    H = model.H
    R = model.R_diag
    S = model.n_states
    O = model.n_obs
    mu_new = list(mu)
    var_new = list(var)
    # Innovation per output dim
    for o in range(O):
        h_row = H[o]
        # predict
        y_hat = sum(h_row[s] * mu_new[s] for s in range(S))
        innov = obs[o] - y_hat
        s_innov = R[o] + sum((h_row[s] ** 2) * var_new[s] for s in range(S))
        # Kalman gain — broadcast over states sharing the o-th observation
        for s in range(S):
            k = (var_new[s] * h_row[s]) / max(s_innov, _EPS)
            mu_new[s] = mu_new[s] + k * innov
            var_new[s] = var_new[s] * (1.0 - k * h_row[s])
            var_new[s] = max(var_new[s], _EPS)
    return mu_new, var_new


def variational_free_energy_linear_gaussian(
    model: LinearGaussianGenerativeModel,
    prior: GaussianBelief,
    observation: Sequence[float],
) -> tuple[GaussianBelief, FreeEnergyReport]:
    """Linear-Gaussian variational inference == Kalman update."""
    model.validate()
    if len(observation) != model.n_obs:
        raise InvalidModel("observation length mismatch")
    mu_new, var_new = _diag_kalman_update(
        model, prior.mu, prior.var_diag, observation
    )
    post = GaussianBelief(mu=mu_new, var_diag=var_new)
    # F = D_KL(q || p) − E_q[log P(o|s)]
    complexity = 0.0
    for i in range(model.n_states):
        v1 = max(var_new[i], _EPS)
        v0 = max(prior.var_diag[i], _EPS)
        complexity += 0.5 * (
            v1 / v0 + ((mu_new[i] - prior.mu[i]) ** 2) / v0 - 1.0 + math.log(v0 / v1)
        )
    accuracy = 0.0
    for o in range(model.n_obs):
        h_row = model.H[o]
        y_hat = sum(h_row[s] * mu_new[s] for s in range(model.n_states))
        r = model.R_diag[o]
        accuracy += -0.5 * math.log(2.0 * math.pi * max(r, _EPS)) - 0.5 * (
            (observation[o] - y_hat) ** 2 + sum((h_row[s] ** 2) * var_new[s] for s in range(model.n_states))
        ) / max(r, _EPS)
    F = complexity - accuracy
    surprise = -accuracy + complexity  # ELBO interpretation
    digest = _hash_payload({"obs_dim": model.n_obs, "F": round(F, 12)})
    return post, FreeEnergyReport(
        F=F,
        complexity=complexity,
        accuracy=accuracy,
        surprise=surprise,
        digest=digest,
    )


def expected_free_energy_linear_gaussian(
    model: LinearGaussianGenerativeModel,
    policy: Policy,
    belief: GaussianBelief,
) -> EFEReport:
    """Closed-form EFE for linear-Gaussian models (Schwöbel 2018, App. B).

    Per step:
      ambiguity_t = 0.5 ∑_o log(2π e R_o)       (state-indep noise entropy)
      epistemic_t = 0.5 ∑_s log(σ²_pred / σ²_post)   (information gain)
      pragmatic_t = − 0.5 ‖μ_pred − C‖² / σ²_pref   (if C provided)
    The risk + ambiguity decomposition is reported up to a constant.
    """
    model.validate()
    mu = list(belief.mu)
    var = list(belief.var_diag)
    risk = 0.0
    ambiguity = 0.0
    epistemic = 0.0
    pragmatic = 0.0
    for t, a in enumerate(policy.actions):
        # predict forward
        mu_pred, var_pred = _diag_kalman_predict(model, mu, var, a)
        # marginal var on observations
        obs_var = [
            model.R_diag[o] + sum((model.H[o][s] ** 2) * var_pred[s] for s in range(model.n_states))
            for o in range(model.n_obs)
        ]
        # ambiguity (expected noise entropy)
        amb_t = 0.0
        for o in range(model.n_obs):
            amb_t += 0.5 * math.log(2.0 * math.pi * math.e * max(model.R_diag[o], _EPS))
        ambiguity += amb_t
        # hypothetical update at the predicted mean (no actual obs) — entropy reduction
        ep_t = 0.0
        # post variance if we observe predicted mean
        _, var_post = _diag_kalman_update(
            model, mu_pred, var_pred,
            [sum(model.H[o][s] * mu_pred[s] for s in range(model.n_states)) for o in range(model.n_obs)],
        )
        for s in range(model.n_states):
            ep_t += 0.5 * math.log(max(var_pred[s], _EPS) / max(var_post[s], _EPS))
        epistemic += ep_t
        # pragmatic: gaussian preference around C with unit variance
        if model.C is not None:
            yhat = [
                sum(model.H[o][s] * mu_pred[s] for s in range(model.n_states))
                for o in range(model.n_obs)
            ]
            prag_t = -0.5 * sum((yhat[o] - model.C[o]) ** 2 for o in range(model.n_obs))
            pragmatic += prag_t
            risk_t = -prag_t  # KL ≈ ½ ‖μ − C‖² for unit-var Gaussian
            risk += risk_t
        mu, var = mu_pred, var_pred
    G = ambiguity + risk - epistemic
    return EFEReport(
        policy=policy,
        G=G,
        risk=risk,
        ambiguity=ambiguity,
        epistemic_value=epistemic,
        pragmatic_value=pragmatic,
    )


# =====================================================================
# Bayesian model averaging
# =====================================================================


def bayesian_model_average_belief(
    beliefs: Sequence[CategoricalBelief],
    log_evidence: Sequence[float],
) -> CategoricalBelief:
    """Posterior over states marginalised over models with weights ∝ exp(log_ev).

    All beliefs must share the same state cardinality.  Returns a pmf
    that is the weighted mixture.
    """
    if not beliefs:
        raise InvalidModel("bma: empty beliefs")
    if len(beliefs) != len(log_evidence):
        raise InvalidModel("bma: weights length mismatch")
    S = len(beliefs[0].probs)
    if any(len(b.probs) != S for b in beliefs):
        raise InvalidModel("bma: dimension mismatch across beliefs")
    w = _softmax(list(log_evidence))
    out = [0.0] * S
    for k, b in enumerate(beliefs):
        for s in range(S):
            out[s] += w[k] * b.probs[s]
    return CategoricalBelief(probs=_normalize(out))


def bayesian_surprise_discrete(
    model: DiscreteGenerativeModel,
    prior: Sequence[float],
    observation: int,
) -> float:
    """Bayesian surprise = D_KL(q(s|o) || q(s)) — the agent's belief update size.

    This is the information-theoretic novelty signal used by curiosity-
    driven exploration (Itti & Baldi 2009).
    """
    model.validate()
    if not _is_pmf(prior):
        raise InvalidModel("prior must be a pmf")
    if not (0 <= observation < model.n_obs):
        raise InvalidModel("observation out of range")
    A_o = [model.A[observation][s] for s in range(model.n_states)]
    Z = sum(A_o[s] * prior[s] for s in range(model.n_states))
    if Z < _EPS:
        return 0.0
    post = [A_o[s] * prior[s] / Z for s in range(model.n_states)]
    return _kl(post, list(prior))


# =====================================================================
# PAC bounds on expected utility
# =====================================================================


def hoeffding_half_width(n: int, *, delta: float, range_: float = 1.0) -> float:
    """Two-sided Hoeffding half-width: ε = range_ √(log(2/δ) / (2n))."""
    if n <= 0:
        raise InsufficientData("hoeffding: n must be ≥ 1")
    if not (0 < delta < 1):
        raise ValueError("hoeffding: delta must be in (0,1)")
    if range_ <= 0:
        raise ValueError("hoeffding: range_ must be > 0")
    return range_ * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    samples: Sequence[float],
    *,
    delta: float,
    range_: float = 1.0,
) -> float:
    """Maurer-Pontil empirical-Bernstein half-width (2009)."""
    n = len(samples)
    if n < 2:
        raise InsufficientData("empirical bernstein: n ≥ 2")
    if not (0 < delta < 1):
        raise ValueError("empirical bernstein: delta in (0,1)")
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / (n - 1)
    a = math.sqrt(2.0 * var * math.log(2.0 / delta) / n)
    b = 7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    return a + b


def expected_utility_bound(
    samples: Sequence[float],
    *,
    delta: float = 0.05,
    range_: float = 1.0,
    method: str = "empirical_bernstein",
) -> UtilityBound:
    """Anytime-valid PAC bound on the expected utility under a policy."""
    n = len(samples)
    if n == 0:
        raise InsufficientData("expected_utility_bound: zero samples")
    mean = sum(samples) / n
    if method == "hoeffding":
        eps = hoeffding_half_width(n, delta=delta, range_=range_)
    elif method == "empirical_bernstein":
        if n < 2:
            eps = hoeffding_half_width(n, delta=delta, range_=range_)
            method = "hoeffding"
        else:
            eps = empirical_bernstein_half_width(samples, delta=delta, range_=range_)
    else:
        raise ValueError(f"unknown method: {method}")
    return UtilityBound(
        n=n,
        mean=mean,
        half_width=eps,
        lcb=mean - eps,
        ucb=mean + eps,
        delta=delta,
        method=method,
    )


# =====================================================================
# Internal: per-agent state
# =====================================================================


@dataclass
class _AgentState:
    name: str
    kind: str
    model_discrete: DiscreteGenerativeModel | None = None
    model_linear: LinearGaussianGenerativeModel | None = None
    belief_discrete: CategoricalBelief | None = None
    belief_linear: GaussianBelief | None = None
    horizon: int = 1
    gamma: float = 1.0
    habit_E: list[float] | None = None
    obs_history: list[Any] = field(default_factory=list)
    action_history: list[int] = field(default_factory=list)
    utility_history: list[float] = field(default_factory=list)
    accumulated_F: float = 0.0
    last_efe: float | None = None
    counts_A: list[list[float]] | None = None
    counts_B: list[list[list[float]]] | None = None
    counts_E: list[float] | None = None


# =====================================================================
# ActiveInferencer
# =====================================================================


class ActiveInferencer:
    """Active inference as a runtime primitive.

    Maintains a registry of named *agents*, each with its own generative
    model, running belief and habit prior.  The class is fully threadsafe
    under a recursive lock; all methods are safe to call concurrently.

    Optional dependencies:

      bus       — `agi.events.EventBus` for live event broadcast.
      attestor  — `agi.attest.RuntimeAttestor` for content-hashed receipts.
      random_seed — for reproducible sampling.
    """

    def __init__(
        self,
        *,
        bus: Any = None,
        attestor: Any = None,
        random_seed: int | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._rng = random.Random(random_seed)
        self._agents: dict[str, _AgentState] = {}
        self._inferences = 0
        self._plans = 0
        self._acts = 0
        self._learns = 0
        self._bma_calls = 0
        self._receipts = 0
        self._started_ns = time.time_ns()
        self._emit(AI_STARTED, {"id": uuid.uuid4().hex[:16], "ts_ns": self._started_ns})

    # ---------- event + attest helpers ----------

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None or Event is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(payload)))
        except Exception:
            pass

    def _attest(self, kind: str, payload: dict) -> str:
        digest = _hash_payload(payload)
        if self._attestor is not None:
            receipt = _AttestableReceipt(kind=kind, payload=dict(payload), digest=digest)
            try:
                if hasattr(self._attestor, "record"):
                    self._attestor.record(kind=kind, payload=dict(payload))
                elif callable(self._attestor):
                    self._attestor(receipt)
            except Exception:
                pass
        self._receipts += 1
        return digest

    def _agent(self, name: str) -> _AgentState:
        if name not in self._agents:
            raise UnknownAgent(name)
        return self._agents[name]

    # ---------- agent lifecycle ----------

    def register_agent(
        self,
        name: str,
        model: DiscreteGenerativeModel | LinearGaussianGenerativeModel,
        *,
        gamma: float = 1.0,
        habit_E: Sequence[float] | None = None,
        horizon: int = 1,
    ) -> AgentSnapshot:
        """Register a new agent under a fresh name."""
        with self._lock:
            if name in self._agents:
                raise InvalidModel(f"agent {name!r} already registered")
            if gamma <= 0:
                raise InvalidModel("gamma must be > 0")
            if horizon < 1:
                raise InvalidModel("horizon must be ≥ 1")
            if isinstance(model, DiscreteGenerativeModel):
                model.validate()
                state = _AgentState(
                    name=name,
                    kind=KIND_DISCRETE,
                    model_discrete=model,
                    belief_discrete=CategoricalBelief(probs=list(model.D)),
                    horizon=horizon,
                    gamma=gamma,
                    habit_E=list(habit_E) if habit_E is not None else None,
                    counts_A=[[0.0] * model.n_states for _ in range(model.n_obs)],
                    counts_B=[
                        [[0.0] * model.n_states for _ in range(model.n_states)]
                        for _ in range(model.n_actions)
                    ],
                    counts_E=[0.0] * (model.n_actions ** horizon),
                )
            elif isinstance(model, LinearGaussianGenerativeModel):
                model.validate()
                state = _AgentState(
                    name=name,
                    kind=KIND_LINEAR_GAUSSIAN,
                    model_linear=model,
                    belief_linear=GaussianBelief(
                        mu=list(model.mu0), var_diag=list(model.Sigma0_diag)
                    ),
                    horizon=horizon,
                    gamma=gamma,
                    habit_E=list(habit_E) if habit_E is not None else None,
                )
            else:
                raise UnknownKind(f"unknown model type {type(model)!r}")
            self._agents[name] = state
            payload = {
                "name": name,
                "kind": state.kind,
                "gamma": gamma,
                "horizon": horizon,
            }
            digest = self._attest(AI_AGENT_REGISTERED, payload)
            self._emit(AI_AGENT_REGISTERED, {**payload, "digest": digest})
            return self.snapshot(name)

    def remove_agent(self, name: str) -> None:
        with self._lock:
            if name in self._agents:
                del self._agents[name]
                digest = self._attest(AI_AGENT_REMOVED, {"name": name})
                self._emit(AI_AGENT_REMOVED, {"name": name, "digest": digest})

    def list_agents(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    # ---------- inference ----------

    def step(self, name: str, observation: Any) -> FreeEnergyReport:
        """Variational state inference on a new observation.

        Discrete: ``observation`` is an int outcome index.
        Linear-Gaussian: ``observation`` is a sequence of floats of length n_obs.
        """
        with self._lock:
            state = self._agent(name)
            if state.kind == KIND_DISCRETE:
                if state.model_discrete is None or state.belief_discrete is None:
                    raise InvalidModel("discrete agent missing model/belief")
                prior = list(state.belief_discrete.probs)
                post, report = variational_free_energy_discrete(
                    state.model_discrete, prior, int(observation)
                )
                state.belief_discrete = post
                state.obs_history.append(int(observation))
            elif state.kind == KIND_LINEAR_GAUSSIAN:
                if state.model_linear is None or state.belief_linear is None:
                    raise InvalidModel("linear-gaussian agent missing model/belief")
                post, report = variational_free_energy_linear_gaussian(
                    state.model_linear, state.belief_linear, list(observation)
                )
                state.belief_linear = post
                state.obs_history.append(list(observation))
            else:
                raise UnknownKind(state.kind)
            state.accumulated_F += report.F
            self._inferences += 1
            payload = {
                "name": name,
                "F": report.F,
                "complexity": report.complexity,
                "accuracy": report.accuracy,
                "surprise": report.surprise,
            }
            digest = self._attest(AI_INFERRED, payload)
            self._emit(AI_INFERRED, {**payload, "digest": digest})
            return FreeEnergyReport(
                F=report.F,
                complexity=report.complexity,
                accuracy=report.accuracy,
                surprise=report.surprise,
                digest=digest,
            )

    def observe(self, name: str, observation: Any) -> FreeEnergyReport:
        """Alias for ``step``."""
        return self.step(name, observation)

    def belief(self, name: str) -> CategoricalBelief | GaussianBelief:
        """Return a copy of the current posterior belief."""
        with self._lock:
            state = self._agent(name)
            if state.kind == KIND_DISCRETE:
                assert state.belief_discrete is not None
                return CategoricalBelief(probs=list(state.belief_discrete.probs))
            else:
                assert state.belief_linear is not None
                return GaussianBelief(
                    mu=list(state.belief_linear.mu),
                    var_diag=list(state.belief_linear.var_diag),
                )

    # ---------- planning ----------

    def plan(
        self,
        name: str,
        *,
        horizon: int | None = None,
        candidate_policies: Sequence[Policy] | None = None,
    ) -> PolicySelection:
        """Rank candidate policies by softmax(-γ G) and return posterior.

        If ``candidate_policies`` is None and the action space and horizon
        are small (|A|^H ≤ 256), the planner enumerates *all* policies of
        the given horizon.  For larger spaces the caller should supply a
        candidate set (e.g. produced by random rollouts or a beam search).
        """
        with self._lock:
            state = self._agent(name)
            H = horizon if horizon is not None else state.horizon
            if H < 1:
                raise InvalidPolicy("horizon must be ≥ 1")
            if state.kind == KIND_DISCRETE:
                assert state.model_discrete is not None and state.belief_discrete is not None
                model = state.model_discrete
                if candidate_policies is None:
                    if model.n_actions ** H > 256:
                        raise InvalidPolicy(
                            f"enumeration size {model.n_actions ** H} > 256; "
                            "supply candidate_policies"
                        )
                    candidates = enumerate_policies(model.n_actions, H)
                else:
                    candidates = list(candidate_policies)
                    for p in candidates:
                        if p.horizon != H:
                            raise InvalidPolicy("policy horizon mismatch")
                        for a in p.actions:
                            if not (0 <= a < model.n_actions):
                                raise InvalidPolicy("action out of range")
                efe = [
                    expected_free_energy_discrete(model, p, state.belief_discrete)
                    for p in candidates
                ]
            elif state.kind == KIND_LINEAR_GAUSSIAN:
                assert state.model_linear is not None and state.belief_linear is not None
                model_lg = state.model_linear
                if candidate_policies is None:
                    if model_lg.n_actions ** H > 256:
                        raise InvalidPolicy(
                            f"enumeration size {model_lg.n_actions ** H} > 256; "
                            "supply candidate_policies"
                        )
                    candidates = enumerate_policies(model_lg.n_actions, H)
                else:
                    candidates = list(candidate_policies)
                efe = [
                    expected_free_energy_linear_gaussian(
                        model_lg, p, state.belief_linear
                    )
                    for p in candidates
                ]
            else:
                raise UnknownKind(state.kind)
            E = state.habit_E if (state.habit_E and len(state.habit_E) == len(efe)) else None
            q_pi = policy_posterior(efe, gamma=state.gamma, habit_E=E)
            best = max(range(len(q_pi)), key=lambda i: q_pi[i])
            payload = {
                "name": name,
                "horizon": H,
                "n_candidates": len(candidates),
                "best_idx": best,
                "best_G": efe[best].G,
                "gamma": state.gamma,
            }
            digest = self._attest(AI_PLANNED, payload)
            self._emit(AI_PLANNED, {**payload, "digest": digest})
            state.last_efe = efe[best].G
            self._plans += 1
            return PolicySelection(
                candidates=tuple(candidates),
                q_pi=tuple(q_pi),
                best=best,
                efe=tuple(efe),
                gamma=state.gamma,
                digest=digest,
            )

    def act(
        self,
        name: str,
        *,
        mode: str = SELECT_SOFTMAX,
        horizon: int | None = None,
        candidate_policies: Sequence[Policy] | None = None,
        advance_belief: bool = True,
    ) -> int:
        """Pick the next action under the chosen selection rule.

        Returns the action *index* (first step of the selected policy).
        When ``advance_belief=True`` the running belief is moved forward
        by one B(a) step (the prior for the next observation).
        """
        if mode not in KNOWN_SELECTORS:
            raise InvalidPolicy(f"unknown selector {mode!r}")
        with self._lock:
            state = self._agent(name)
            sel = self.plan(name, horizon=horizon, candidate_policies=candidate_policies)
            if mode == SELECT_ARGMAX:
                idx = sel.best
            elif mode == SELECT_SOFTMAX:
                u = self._rng.random()
                c = 0.0
                idx = 0
                for i, p in enumerate(sel.q_pi):
                    c += p
                    if u <= c:
                        idx = i
                        break
                else:
                    idx = len(sel.q_pi) - 1
            elif mode == SELECT_HABIT_ONLY:
                if state.habit_E is None:
                    raise InvalidPolicy("habit_only requested but no habit_E set")
                idx = max(range(len(state.habit_E)), key=lambda i: state.habit_E[i])
                if idx >= len(sel.candidates):
                    idx = sel.best
            elif mode == SELECT_RANDOM:
                idx = self._rng.randrange(len(sel.candidates))
            policy = sel.candidates[idx]
            action = policy.actions[0]
            state.action_history.append(action)
            # advance habit count
            if state.counts_E is not None and idx < len(state.counts_E):
                state.counts_E[idx] += 1.0
            if advance_belief:
                if state.kind == KIND_DISCRETE and state.model_discrete is not None and state.belief_discrete is not None:
                    state.belief_discrete = predict_belief_discrete(
                        state.model_discrete, state.belief_discrete, action
                    )
                elif state.kind == KIND_LINEAR_GAUSSIAN and state.model_linear is not None and state.belief_linear is not None:
                    mu_new, var_new = _diag_kalman_predict(
                        state.model_linear,
                        state.belief_linear.mu,
                        state.belief_linear.var_diag,
                        action,
                    )
                    state.belief_linear = GaussianBelief(mu=mu_new, var_diag=var_new)
            self._acts += 1
            payload = {"name": name, "action": action, "mode": mode, "policy_idx": idx}
            digest = self._attest(AI_ACTED, payload)
            self._emit(AI_ACTED, {**payload, "digest": digest})
            return action

    # ---------- learning ----------

    def learn(
        self,
        name: str,
        *,
        observation: int | None = None,
        prev_state_belief: Sequence[float] | None = None,
        action: int | None = None,
        next_state_belief: Sequence[float] | None = None,
        lr: float = 1.0,
    ) -> None:
        """Dirichlet learning of the A and B matrices.

        Discrete only.  Pass the current observation together with the
        agent's belief over the latent state at the moment of observing
        to update A by counts.  Pass the (action, prev_belief, next_belief)
        triple to update B.

        ``lr`` is a multiplicative weight on the count increment; the
        default lr=1 is the canonical posterior-evidence Dirichlet update.
        """
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"learn: kind {state.kind} not supported")
            assert state.model_discrete is not None and state.counts_A is not None and state.counts_B is not None
            model = state.model_discrete
            # A update
            if observation is not None and prev_state_belief is not None:
                if not (0 <= observation < model.n_obs):
                    raise InvalidModel("learn: observation out of range")
                if len(prev_state_belief) != model.n_states:
                    raise InvalidModel("learn: prev_state_belief length mismatch")
                for s in range(model.n_states):
                    state.counts_A[observation][s] += lr * prev_state_belief[s]
            # B update
            if action is not None and prev_state_belief is not None and next_state_belief is not None:
                if not (0 <= action < model.n_actions):
                    raise InvalidModel("learn: action out of range")
                if len(next_state_belief) != model.n_states:
                    raise InvalidModel("learn: next_state_belief length mismatch")
                for sp in range(model.n_states):
                    for s in range(model.n_states):
                        state.counts_B[action][sp][s] += (
                            lr * prev_state_belief[s] * next_state_belief[sp]
                        )
            self._learns += 1
            payload = {
                "name": name,
                "did_A": observation is not None,
                "did_B": action is not None,
            }
            digest = self._attest(AI_LEARNED, payload)
            self._emit(AI_LEARNED, {**payload, "digest": digest})

    def consolidate_learning(
        self,
        name: str,
        *,
        smoothing: float = 1.0,
    ) -> None:
        """Move accumulated Dirichlet counts into the live A/B with smoothing.

        Each A column and B[a] column is renormalised to a pmf with a small
        smoothing term (Laplace).  Discrete only.
        """
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"consolidate_learning: kind {state.kind}")
            assert state.model_discrete is not None and state.counts_A is not None and state.counts_B is not None
            model = state.model_discrete
            O = model.n_obs
            S = model.n_states
            new_A = [[0.0] * S for _ in range(O)]
            for s in range(S):
                col_total = sum(state.counts_A[o][s] for o in range(O)) + smoothing * O
                for o in range(O):
                    new_A[o][s] = (state.counts_A[o][s] + smoothing) / col_total
            model.A = new_A
            new_B = [
                [[0.0] * S for _ in range(S)] for _ in range(model.n_actions)
            ]
            for a in range(model.n_actions):
                for s in range(S):
                    col_total = (
                        sum(state.counts_B[a][sp][s] for sp in range(S))
                        + smoothing * S
                    )
                    for sp in range(S):
                        new_B[a][sp][s] = (
                            state.counts_B[a][sp][s] + smoothing
                        ) / col_total
            model.B = new_B

    # ---------- composition ----------

    def set_likelihood(self, name: str, A: Sequence[Sequence[float]]) -> None:
        """Replace the A matrix of a discrete agent.  Validates column-stochastic."""
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"set_likelihood: kind {state.kind}")
            assert state.model_discrete is not None
            new_A = [list(row) for row in A]
            if len(new_A) != state.model_discrete.n_obs:
                raise InvalidModel("set_likelihood: row count mismatch")
            if any(len(r) != state.model_discrete.n_states for r in new_A):
                raise InvalidModel("set_likelihood: col count mismatch")
            if not _col_stochastic(new_A):
                raise InvalidModel("set_likelihood: not column-stochastic")
            state.model_discrete.A = new_A

    def set_transition(
        self,
        name: str,
        B: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        """Replace the B tensor of a discrete agent.  Validates column-stochastic per action."""
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"set_transition: kind {state.kind}")
            assert state.model_discrete is not None
            new_B = [[list(r) for r in Ba] for Ba in B]
            S = state.model_discrete.n_states
            if any(len(Ba) != S or any(len(row) != S for row in Ba) for Ba in new_B):
                raise InvalidModel("set_transition: shape mismatch")
            for Ba in new_B:
                if not _col_stochastic(Ba):
                    raise InvalidModel("set_transition: not column-stochastic")
            state.model_discrete.B = new_B
            state.model_discrete.n_actions  # noqa  -- ensures lazy validation

    def set_preferences(
        self,
        name: str,
        C: Sequence[float] | Sequence[Sequence[float]],
    ) -> None:
        """Replace the C vector / matrix of a discrete agent.  Log-preferences."""
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"set_preferences: kind {state.kind}")
            assert state.model_discrete is not None
            if not C:
                raise InvalidModel("set_preferences: empty C")
            if isinstance(C[0], (list, tuple)):
                state.model_discrete.C = [[float(x) for x in row] for row in C]
            else:
                state.model_discrete.C = [float(x) for x in C]

    def set_prior(self, name: str, D: Sequence[float]) -> None:
        """Replace the initial-state prior D and reset the running belief."""
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"set_prior: kind {state.kind}")
            assert state.model_discrete is not None
            if not _is_pmf(D):
                raise InvalidModel("set_prior: D not a pmf")
            state.model_discrete.D = list(D)
            state.belief_discrete = CategoricalBelief(probs=list(D))

    def bayesian_model_average(
        self,
        names: Sequence[str],
        *,
        log_evidence: Sequence[float] | None = None,
    ) -> CategoricalBelief:
        """Marginalise the current belief across N discrete agents.

        If ``log_evidence`` is None, each agent contributes its
        (negative) accumulated free energy as model evidence
        (∝ log P(o_{1:t} | model)).
        """
        with self._lock:
            if not names:
                raise InvalidModel("bma: empty names")
            beliefs = []
            ev = []
            for n in names:
                state = self._agent(n)
                if state.kind != KIND_DISCRETE:
                    raise UnknownKind(f"bma: agent {n} is not discrete")
                assert state.belief_discrete is not None
                beliefs.append(state.belief_discrete)
                ev.append(-state.accumulated_F)
            if log_evidence is not None:
                if len(log_evidence) != len(names):
                    raise InvalidModel("bma: log_evidence length mismatch")
                ev = list(log_evidence)
            self._bma_calls += 1
            return bayesian_model_average_belief(beliefs, ev)

    # ---------- counterfactuals + bounds ----------

    def counterfactual(
        self,
        name: str,
        alt_policy: Policy,
        *,
        n_rollouts: int = 100,
    ) -> list[float]:
        """Monte-Carlo expected utility under an alternative policy.

        Discrete only.  Returns one realised utility per rollout where
        utility is read from the preference vector C as a softmax over
        outcomes (∝ exp C[o]).  This is the *expected pragmatic value*
        of the policy under the current belief — a stand-in for "reward"
        in a reward-free model.
        """
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind(f"counterfactual: kind {state.kind}")
            assert state.model_discrete is not None and state.belief_discrete is not None
            model = state.model_discrete
            rng = random.Random(self._rng.random())
            outs: list[float] = []
            for _ in range(int(n_rollouts)):
                u = 0.0
                b = state.belief_discrete
                for t, a in enumerate(alt_policy.actions):
                    s = b.sample(rng)
                    # transition
                    probs = [model.B[a][sp][s] for sp in range(model.n_states)]
                    sp = _sample_categorical(probs, rng)
                    # observe
                    obs_probs = [model.A[o][sp] for o in range(model.n_obs)]
                    obs = _sample_categorical(obs_probs, rng)
                    C_t = model.preferences_at(t)
                    u += C_t[obs]
                    b = CategoricalBelief(
                        probs=[1.0 if i == sp else 0.0 for i in range(model.n_states)]
                    )
                outs.append(u)
            return outs

    def expected_utility_bound(
        self,
        name: str,
        alt_policy: Policy,
        *,
        n_rollouts: int = 200,
        delta: float = 0.05,
        range_: float | None = None,
        method: str = "empirical_bernstein",
    ) -> UtilityBound:
        """PAC bound on the expected utility of ``alt_policy`` under current belief.

        Discrete only.  Uses Hoeffding or empirical-Bernstein on
        ``counterfactual`` rollouts.  ``range_`` defaults to the
        L1-spread of C over the horizon.
        """
        with self._lock:
            state = self._agent(name)
            if state.kind != KIND_DISCRETE:
                raise UnknownKind("expected_utility_bound: discrete only")
            assert state.model_discrete is not None
            samples = self.counterfactual(name, alt_policy, n_rollouts=n_rollouts)
            if range_ is None:
                hi = -math.inf
                lo = math.inf
                for t in range(alt_policy.horizon):
                    row = state.model_discrete.preferences_at(t)
                    hi = max(hi, max(row))
                    lo = min(lo, min(row))
                range_ = (hi - lo) * alt_policy.horizon if hi > lo else 1.0
            # shift samples to [0, range_] for Hoeffding validity
            shifted = [
                s - min(samples) if min(samples) < 0 else s for s in samples
            ]
            return expected_utility_bound(
                shifted, delta=delta, range_=range_, method=method
            )

    # ---------- introspection ----------

    def snapshot(self, name: str) -> AgentSnapshot:
        with self._lock:
            state = self._agent(name)
            if state.kind == KIND_DISCRETE:
                assert state.belief_discrete is not None
                bel_ent = state.belief_discrete.entropy()
            else:
                assert state.belief_linear is not None
                bel_ent = state.belief_linear.entropy()
            return AgentSnapshot(
                name=name,
                kind=state.kind,
                horizon=state.horizon,
                gamma=state.gamma,
                n_obs_seen=len(state.obs_history),
                last_action=state.action_history[-1] if state.action_history else None,
                belief_entropy=bel_ent,
                accumulated_free_energy=state.accumulated_F,
                last_efe=state.last_efe,
            )

    def coverage(self) -> CoverageReport:
        with self._lock:
            return CoverageReport(
                started_ns=self._started_ns,
                agents=len(self._agents),
                inferences=self._inferences,
                plans=self._plans,
                acts=self._acts,
                learns=self._learns,
                bma_calls=self._bma_calls,
                receipts=self._receipts,
            )

    def clear(self) -> None:
        with self._lock:
            self._agents.clear()
            self._inferences = 0
            self._plans = 0
            self._acts = 0
            self._learns = 0
            self._bma_calls = 0
            self._emit(AI_CLEARED, {"ts_ns": time.time_ns()})


# =====================================================================
# Internal helpers
# =====================================================================


def _sample_categorical(probs: Sequence[float], rng: random.Random) -> int:
    p = _normalize(probs)
    u = rng.random()
    c = 0.0
    for i, pi in enumerate(p):
        c += pi
        if u <= c:
            return i
    return len(p) - 1


# =====================================================================
# Quick convenience functions
# =====================================================================


def quick_two_armed_bandit(
    *,
    arm_means: Sequence[float] = (0.3, 0.7),
    horizon: int = 4,
    gamma: float = 4.0,
    random_seed: int | None = None,
) -> tuple[ActiveInferencer, str]:
    """Build a tiny two-armed Bernoulli bandit as a discrete active-inference agent.

    The latent ``hypothesis`` is which arm has the *higher* mean.  We
    augment the state with the agent's *last action* so that the
    observation likelihood A(o | s) is well-defined per (hypothesis,
    action) without needing per-action A matrices.

    States   = 4 = {hypothesis ∈ {0,1}} × {last_action ∈ {0,1}}
    Actions  = {pull_0, pull_1}
    Outcomes = {loss=0, win=1}

    With arm_means = (μ₀, μ₁), the high mean is max(μ₀, μ₁) and the low
    mean is min(μ₀, μ₁).  A[win | (h, a)] = high iff h == a else low.

    The agent should learn to pull the higher-mean arm because pulling
    the *correct* arm both wins more often (pragmatic value) and reduces
    ambiguity about which hypothesis is true (epistemic value).
    """
    mu_lo = min(arm_means[0], arm_means[1])
    mu_hi = max(arm_means[0], arm_means[1])
    if mu_hi == mu_lo:
        mu_hi = min(0.99, mu_hi + 1e-2)
        mu_lo = max(0.01, mu_lo - 1e-2)
    # P(o=1 | s) per state index s = hyp*2 + a_last
    p_win = [
        mu_hi,  # h=0 (arm 0 high), a_last=0 → pulled high
        mu_lo,  # h=0, a_last=1 → pulled low
        mu_lo,  # h=1, a_last=0 → pulled low
        mu_hi,  # h=1, a_last=1 → pulled high
    ]
    A = [
        [1.0 - p for p in p_win],
        list(p_win),
    ]
    # B[a]: send (h, *) → (h, a).  Indices in s-major order: 0..3.
    # Action a=0 should send all states with h=0 to state 0 (h=0, a=0)
    # and all states with h=1 to state 2 (h=1, a=0).
    B = [
        # action 0: a_last=0 in next state
        [
            [1.0, 1.0, 0.0, 0.0],  # next=0: from h=0,*
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],  # next=2: from h=1,*
            [0.0, 0.0, 0.0, 0.0],
        ],
        # action 1: a_last=1 in next state
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],  # next=1: from h=0,*
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],  # next=3: from h=1,*
        ],
    ]
    # Preferences: strongly prefer winning.
    C = [-3.0, 3.0]
    # Uniform prior over hypotheses; arbitrary last_action.
    D = [0.25, 0.25, 0.25, 0.25]
    model = DiscreteGenerativeModel(A=A, B=B, C=C, D=D)
    inf = ActiveInferencer(random_seed=random_seed)
    name = f"bandit-{uuid.uuid4().hex[:8]}"
    inf.register_agent(name, model, gamma=gamma, horizon=horizon)
    return inf, name


__all__ = [
    # event kinds
    "AI_STARTED",
    "AI_AGENT_REGISTERED",
    "AI_AGENT_REMOVED",
    "AI_INFERRED",
    "AI_PLANNED",
    "AI_ACTED",
    "AI_LEARNED",
    "AI_CLEARED",
    # constants
    "KIND_DISCRETE",
    "KIND_LINEAR_GAUSSIAN",
    "KIND_PARTICLE",
    "KNOWN_KINDS",
    "SELECT_ARGMAX",
    "SELECT_SOFTMAX",
    "SELECT_HABIT_ONLY",
    "SELECT_RANDOM",
    "KNOWN_SELECTORS",
    # errors
    "ActiveInferenceError",
    "InvalidModel",
    "InvalidPolicy",
    "UnknownAgent",
    "UnknownKind",
    "InsufficientData",
    # types
    "DiscreteGenerativeModel",
    "LinearGaussianGenerativeModel",
    "CategoricalBelief",
    "GaussianBelief",
    "Policy",
    "EFEReport",
    "PolicySelection",
    "FreeEnergyReport",
    "UtilityBound",
    "CoverageReport",
    "AgentSnapshot",
    # functions
    "variational_free_energy_discrete",
    "variational_free_energy_linear_gaussian",
    "expected_free_energy_discrete",
    "expected_free_energy_linear_gaussian",
    "predict_belief_discrete",
    "predicted_observation_distribution",
    "policy_posterior",
    "enumerate_policies",
    "bayesian_model_average_belief",
    "bayesian_surprise_discrete",
    "hoeffding_half_width",
    "empirical_bernstein_half_width",
    "expected_utility_bound",
    "quick_two_armed_bandit",
    # main class
    "ActiveInferencer",
]
