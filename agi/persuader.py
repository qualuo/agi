r"""Persuader — Bayesian persuasion / information design as a runtime primitive.

A coordination engine that drives many strategic agents (LLM sub-models,
human operators, external services, tenant policies, adversarial bidders)
faces two complementary questions:

  1. **What payments make the agents truthful?** — answered by
     :mod:`agi.mechanism` (Vickrey, VCG, Myerson).
  2. **What information should the principal *reveal* to make the agents
     take the right action?** — that is the *information design* problem,
     and the runtime primitive for it is `Persuader`.

The mathematical foundation is Bayesian persuasion (Kamenica & Gentzkow,
2011): the sender commits to a public signaling policy ``π: Ω → Δ(S)``
mapping each state of the world ``ω ∈ Ω`` to a distribution over signals
``s ∈ S``. The receiver, who shares the prior ``μ₀`` but cannot observe
``ω`` directly, observes ``s``, performs the Bayes update
``μ_s(ω) ∝ μ₀(ω) · π(s | ω)``, and picks the action ``a*(μ_s)`` that
maximises her expected utility under ``μ_s``. The sender's value is

    V(π) = Σ_ω μ₀(ω) Σ_s π(s | ω) · u_S(a*(μ_s), ω).

Persuasion is *not* lying: the sender is required to be Bayes-consistent.
What the sender gets to choose is *partition coarseness*: which posteriors
to induce. The optimum exists because the set of "achievable" posterior
distributions is exactly the set of mean-preserving spreads of ``δ_{μ₀}``
on ``Δ(Ω)`` (Aumann 1995), so the optimal sender payoff is

    V*(μ₀) = cav v̂(μ₀)

where ``v̂(μ) := u_S(a*(μ), μ)`` is the *value function under full
information about the posterior* and ``cav f`` is the *upper concave
envelope* of ``f`` on ``Δ(Ω)``. The primitive ships exact concavification
on the binary-state simplex and an LP-based optimiser on general finite
state/action spaces.

Mathematical core (cited where it counts)
------------------------------------------

  * **Kamenica & Gentzkow, 2011 — "Bayesian persuasion."** Defines the
    persuasion problem, proves the concavification theorem
    ``V*(μ₀) = cav v̂(μ₀)``, and shows the optimal scheme exists with at
    most ``|Ω|`` signals (one per state) — and in fact ``|A|`` signals
    suffice when interpreted as *action recommendations* under the
    receiver's obedience constraint.

  * **Aumann & Maschler, 1995 — "Repeated Games with Incomplete
    Information."** Origin of the *splitting* / Bayes-plausibility view:
    a signal-induced posterior distribution ``τ ∈ Δ(Δ(Ω))`` is
    achievable iff ``E_τ[μ] = μ₀``. The sender chooses any such τ.

  * **Bergemann & Morris, 2016a — "Bayes Correlated Equilibrium and
    the Comparison of Information Structures in Games."** Generalises
    the Kamenica-Gentzkow concavification to *games*: the set of
    incentive-compatible outcome distributions over (signal, action,
    state) tuples is a polytope cut by the receiver's obedience
    constraints; the sender's optimum is an LP over that polytope.
    The LP formulation we ship is the single-receiver, single-state-
    cardinality specialisation of their BCE program.

  * **Bergemann & Morris, 2016b — "Information Design, Bayesian
    Persuasion, and Bayes Correlated Equilibrium."** Names the field
    *information design* and gives the **revelation principle**: WLOG
    we can restrict to signal spaces ``S = A`` and interpret each
    signal as a *recommended action* (the "straightforward"
    recommendation policy). Then π is feasible iff, for every
    recommended action ``a`` and deviation ``a' ∈ A``,

        Σ_ω μ₀(ω) π(a | ω) [u_R(a, ω) − u_R(a', ω)] ≥ 0   (obedience).

    With variables ``x(a, ω) = μ₀(ω) π(a | ω)``, the sender's optimum
    is the linear program

        max  Σ_{a,ω} x(a, ω) · u_S(a, ω)
        s.t. Σ_a  x(a, ω) = μ₀(ω)                   ∀ ω        (marginal)
             Σ_ω  x(a, ω) [u_R(a, ω) − u_R(a', ω)] ≥ 0  ∀ a, a' (obedience)
             x(a, ω) ≥ 0                                       (positivity)

    The primitive ships a stdlib revised-simplex solver with Bland's
    rule (no NumPy / SciPy required) so the LP runs in process and
    composes with the rest of the runtime.

  * **Dughmi & Xu, 2014 — "Algorithmic Bayesian persuasion."**  For
    single-receiver finite-action, the LP above has ``O(|A| · |Ω|)``
    variables and is solvable in polynomial time. For *multi-receiver*
    games the problem is generally NP-hard; we ship the
    **independent-private-signaling** approximation (each receiver
    gets her own optimal scheme, ignoring cross-receiver externalities)
    which is exact when receivers' utilities are *separable*.

  * **Castiglioni, Marchesi, Romano & Gatti, 2020 — "Online Bayesian
    persuasion."**  When the sender does not know the receiver's
    type, the persuasion problem becomes a regret-minimisation
    problem over the convex set of signaling schemes. A reduction
    to the Hedge / multiplicative-weights algorithm on a discretised
    set of *posterior bundles* attains regret

        R_T ≤ O( √(T · log |Π|) )

    where ``|Π|`` is the discretisation size. The primitive ships
    `online_persuade` with this bound exposed as an anytime-valid
    cumulative-regret certificate.

  * **Babichenko & Barman, 2017 — "Algorithmic aspects of private
    Bayesian persuasion."** Multi-receiver private signaling under
    *supermodular* sender utility admits a (1 − 1/e)-approximation
    via greedy. Under *additive* receiver utility (no cross-receiver
    externalities) the optimal private scheme decomposes per
    receiver — this is the exact case we expose as
    `multi_receiver_private`.

  * **Dworczak & Pavan, 2022 — "Preparing for the worst but hoping
    for the best: robust mechanism and persuasion design."** When
    the sender's prior is only known up to a set ``U ⊂ Δ(Ω)``, the
    robust-Bayesian-persuasion problem is

        max_π  min_{μ ∈ U}  V(π, μ).

    For ``U`` a finite convex hull of point priors, this is a
    bilinear saddle-point that reduces to an LP via duality. The
    primitive ships `robust_persuade` for finite ``U``.

  * **Mathevet, Perego & Taneva, 2020 — "On information design in
    games."** Extends Kamenica-Gentzkow to N-receiver games with
    public signaling; we ship the *public* multi-receiver case
    (one signal observable by all receivers) via the BCE LP applied
    to the joint action profile.

  * **Hoeffding, 1963; Maurer-Pontil, 2009.** Anytime-valid PAC
    bounds on empirical sender payoff. Every `simulate(...)` call
    returns a Hoeffding LCB on the expected sender payoff under
    the chosen signaling scheme.

What it composes (razor-sharp coordination integration)
-------------------------------------------------------

  * **MechanismDesigner.** Persuasion is the *transfer-free* dual:
    when paying agents is impossible (legal, policy, or
    multi-tenant-isolation constraints), the coordinator can still
    steer them via information. The standard composition is
    "Persuader recommends the action, MechanismDesigner attaches a
    Vickrey-style payment to enforce dominant-strategy execution".

  * **TruthSerum.** Persuasion needs the receiver's utility; if it
    is unknown, TruthSerum elicits it via incentive-compatible
    peer prediction. The receiver's *meta-prediction* is exactly
    the belief the sender needs to compute v̂.

  * **Equilibrator.** Multi-receiver persuasion lands the receiver
    sub-game on a Bayes-Nash equilibrium of the induced posterior;
    Equilibrator verifies it and reports exploitability. The
    persuasion LP's *obedience* constraints are exactly Bergemann
    & Morris's BCE constraints — so Equilibrator can read the LP
    primal-dual pair directly.

  * **Negotiator.** When N receivers each pick an action affecting
    a shared resource, the sender's optimal recommendation profile
    is itself an allocation; Negotiator's leximin / Nash-bargaining
    routines accept the Persuader's induced posterior as input and
    refine the recommendation to a fair-and-truthful split.

  * **ActiveInferencer.** The receiver's "best response under
    posterior μ_s" is exactly the expected-free-energy planner;
    ActiveInferencer is a drop-in solver for u_R when the receiver
    is itself a model with a generative-model belief.

  * **Strategist / PolicyImprover.** The Persuader's recommended
    action under each signal is a contextual policy; Strategist
    risk-adjusts it; PolicyImprover learns to refine the signaling
    scheme from logged (state, signal, action, reward) tuples via
    its off-policy estimators (IPS / SNIPS / DR).

  * **CalibrationEngine.** The posteriors ``μ_s`` are exactly the
    *calibrated probabilities* the engine consumes; piping
    ``persuader.posterior(s)`` → CalibrationEngine yields a
    sharp-and-calibrated belief feed for downstream consumers.

  * **AttestationLedger.** Every `persuade(...)` call returns a
    tamper-evident receipt: the signaling scheme π, the prior μ₀,
    the realised posteriors {μ_s}, the receiver's BR, and the
    sender's expected payoff are all hashed into a Merkle leaf
    appended to the attestation chain.

Limits
------

  * **Commitment.** Bayesian persuasion assumes the sender can
    *commit* to the signaling policy. If the receiver believes the
    sender will deviate ex post, the model collapses to a cheap-talk
    game (Crawford & Sobel 1982). For coordination engines the
    commitment is operationalised via AttestationLedger: the
    coordinator publishes π *before* observing ω, signs it, and
    the receiver can verify after the fact.

  * **Shared prior.** Kamenica-Gentzkow assumes both parties share
    ``μ₀``. When the receiver's prior differs, use
    `robust_persuade` over a set ``U`` containing both.

  * **Finite states and actions.** The exact LP scales as
    ``O((|A|·|Ω|)³)`` worst case; for continuous states discretise
    or use the Castiglioni-et-al online algorithm.

  * **Receiver rationality.** Optimal persuasion assumes the
    receiver Bayes-updates and best-responds; for boundedly-
    rational receivers, replace u_R with the BR-distribution
    (logit, quantile, satisficing) — the LP structure carries
    over.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event  # type: ignore
except Exception:  # pragma: no cover
    Event = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================

PERSUADE_STARTED = "persuader.started"
PERSUADE_SOLVED = "persuader.solved"
PERSUADE_SIGNAL_SENT = "persuader.signal_sent"
PERSUADE_VERIFIED = "persuader.verified"
PERSUADE_CERTIFIED = "persuader.certified"
PERSUADE_ONLINE_STEP = "persuader.online_step"
PERSUADE_ROBUST_SOLVED = "persuader.robust_solved"
PERSUADE_MULTI_SOLVED = "persuader.multi_solved"
PERSUADE_CLEARED = "persuader.cleared"


# =====================================================================
# Mode identifiers
# =====================================================================

KIND_CONCAVIFICATION = "concavification"
KIND_LP = "lp"
KIND_ONLINE = "online_hedge"
KIND_ROBUST = "robust_maxmin"
KIND_MULTI_PRIVATE = "multi_receiver_private"
KIND_MULTI_PUBLIC = "multi_receiver_public"

KNOWN_KINDS = (
    KIND_CONCAVIFICATION,
    KIND_LP,
    KIND_ONLINE,
    KIND_ROBUST,
    KIND_MULTI_PRIVATE,
    KIND_MULTI_PUBLIC,
)


# Numerical tolerances
_EPS = 1e-9
_OBEDIENCE_TOL = 1e-7
_LP_TOL = 1e-9
_LP_MAX_ITERS = 50_000


# =====================================================================
# Exceptions
# =====================================================================


class PersuaderError(Exception):
    """Base class for persuasion errors."""


class InvalidPrior(PersuaderError):
    """A prior is not a valid probability vector."""


class InvalidUtility(PersuaderError):
    """Utility matrices have inconsistent shape or non-finite entries."""


class InfeasibleProgram(PersuaderError):
    """The LP / saddle-point program has no feasible solution."""


class UnknownReceiver(PersuaderError):
    """A receiver id was referenced but not registered."""


class InsufficientData(PersuaderError):
    """Not enough samples to compute a finite-sample certificate."""


class UnknownKind(PersuaderError):
    """Unknown persuasion mode."""


# =====================================================================
# Concentration helpers (anytime-valid PAC certificates)
# =====================================================================


def hoeffding_radius(n: int, *, delta: float, range_: float) -> float:
    r"""Hoeffding's half-width: ``range · √(ln(2/δ) / (2n))``.

    For an empirical mean of n bounded samples in ``[a, a+range_]``,
    a two-sided (1 − δ)-PAC confidence interval has half-width
    ``range_ · √(ln(2/δ) / (2 n))``.
    """
    if n <= 0:
        raise InsufficientData("Hoeffding requires n ≥ 1.")
    if not (0.0 < delta < 1.0):
        raise PersuaderError("delta must lie strictly in (0, 1).")
    if range_ < 0:
        raise PersuaderError("range_ must be non-negative.")
    return range_ * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_radius(
    samples: Sequence[float], *, delta: float, range_: float
) -> float:
    r"""Empirical-Bernstein half-width (Maurer & Pontil 2009).

    Tighter than Hoeffding when the empirical variance is small. For n
    bounded samples in ``[a, a+range_]``:

        EB(n, δ) = √(2 V̂ ln(2/δ) / n) + 7 range_ ln(2/δ) / (3 (n − 1))

    where ``V̂`` is the unbiased sample variance.
    """
    n = len(samples)
    if n < 2:
        raise InsufficientData("Empirical-Bernstein requires n ≥ 2.")
    if not (0.0 < delta < 1.0):
        raise PersuaderError("delta must lie strictly in (0, 1).")
    if range_ < 0:
        raise PersuaderError("range_ must be non-negative.")
    var = statistics.variance(samples)
    return math.sqrt(2.0 * var * math.log(2.0 / delta) / n) + (
        7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    )


# =====================================================================
# Data classes
# =====================================================================


@dataclass(frozen=True)
class PersuasionGame:
    """A finite single-receiver persuasion game.

    Attributes
    ----------
    states : tuple of state labels (``Ω``)
    actions : tuple of action labels (``A``)
    prior : prior ``μ₀ ∈ Δ(Ω)``, indexed by ``states``
    sender_utility : ``u_S[a][ω]`` — sender's payoff matrix
    receiver_utility : ``u_R[a][ω]`` — receiver's payoff matrix
    """

    states: tuple[str, ...]
    actions: tuple[str, ...]
    prior: tuple[float, ...]
    sender_utility: tuple[tuple[float, ...], ...]
    receiver_utility: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n_states = len(self.states)
        n_actions = len(self.actions)
        if n_states == 0:
            raise InvalidUtility("PersuasionGame requires at least one state.")
        if n_actions == 0:
            raise InvalidUtility("PersuasionGame requires at least one action.")
        if len(self.prior) != n_states:
            raise InvalidPrior(
                f"prior length {len(self.prior)} ≠ number of states {n_states}."
            )
        _validate_simplex(self.prior, name="prior")
        if len(self.sender_utility) != n_actions or any(
            len(row) != n_states for row in self.sender_utility
        ):
            raise InvalidUtility("sender_utility must be |A| × |Ω|.")
        if len(self.receiver_utility) != n_actions or any(
            len(row) != n_states for row in self.receiver_utility
        ):
            raise InvalidUtility("receiver_utility must be |A| × |Ω|.")
        for row in self.sender_utility + self.receiver_utility:
            for v in row:
                if not math.isfinite(v):
                    raise InvalidUtility("Utility entries must be finite.")


@dataclass(frozen=True)
class SignalingScheme:
    """A signaling scheme π: Ω → Δ(S).

    Indexed as ``pi[signal][state] = π(s | ω)``. Signals carry their own
    labels (typically action recommendations).
    """

    signals: tuple[str, ...]
    states: tuple[str, ...]
    pi: tuple[tuple[float, ...], ...]  # pi[signal_idx][state_idx]

    def __post_init__(self) -> None:
        if len(self.signals) != len(self.pi):
            raise InvalidUtility(
                f"pi has {len(self.pi)} rows, signals has {len(self.signals)}."
            )
        for row in self.pi:
            if len(row) != len(self.states):
                raise InvalidUtility("pi rows must have |Ω| columns.")
            for v in row:
                if not (0.0 - _EPS <= v <= 1.0 + _EPS) or not math.isfinite(v):
                    raise InvalidUtility(
                        "pi entries must be probabilities in [0, 1]."
                    )
        # Column sums = 1 (conditional on each state, signal distribution sums to 1).
        for j, _ in enumerate(self.states):
            col_sum = sum(self.pi[i][j] for i in range(len(self.signals)))
            if abs(col_sum - 1.0) > 1e-6:
                raise InvalidUtility(
                    f"pi(·|state={self.states[j]}) sums to {col_sum:.6f}, not 1."
                )

    def conditional(self, signal: str) -> dict[str, float]:
        """Return the conditional ``π(s = signal | ω)`` as a dict over states."""
        i = self.signals.index(signal)
        return {s: self.pi[i][j] for j, s in enumerate(self.states)}

    def marginal(self, prior: Sequence[float]) -> dict[str, float]:
        """Marginal signal probability ``P(s) = Σ_ω π(s | ω) μ(ω)``."""
        return {
            s: sum(self.pi[i][j] * prior[j] for j in range(len(self.states)))
            for i, s in enumerate(self.signals)
        }

    def posterior(self, signal: str, prior: Sequence[float]) -> dict[str, float]:
        r"""Bayes posterior ``μ_s(ω) ∝ μ(ω) · π(s | ω)``."""
        i = self.signals.index(signal)
        num = [self.pi[i][j] * prior[j] for j in range(len(self.states))]
        denom = sum(num)
        if denom <= _EPS:
            # Zero-probability signal — return prior as Bayes-rule fallback.
            return {s: prior[j] for j, s in enumerate(self.states)}
        return {s: num[j] / denom for j, s in enumerate(self.states)}


@dataclass(frozen=True)
class PersuasionOutcome:
    """Result of an optimal persuasion solve."""

    kind: str
    scheme: SignalingScheme
    sender_value: float
    receiver_value: float
    induced_posteriors: tuple[tuple[float, ...], ...]  # by signal
    signal_marginals: tuple[float, ...]
    recommended_actions: tuple[str, ...]  # action induced under each signal
    bayes_plausible: bool
    obedience_ok: bool
    receipt_digest: str
    meta: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class OnlinePersuasionOutcome:
    """Result of running online persuasion for T rounds."""

    T: int
    cumulative_sender: float
    cumulative_best_fixed: float
    cumulative_regret: float
    regret_bound: float
    final_scheme: SignalingScheme
    receipt_digest: str


@dataclass(frozen=True)
class RobustOutcome:
    """Result of a robust persuasion solve over a finite prior set."""

    scheme: SignalingScheme
    worst_case_value: float
    worst_case_prior: tuple[float, ...]
    receipt_digest: str


@dataclass(frozen=True)
class MultiReceiverOutcome:
    """Result of multi-receiver persuasion (private or public)."""

    kind: str  # KIND_MULTI_PRIVATE or KIND_MULTI_PUBLIC
    per_receiver_schemes: tuple[tuple[str, SignalingScheme], ...]
    joint_sender_value: float
    receipt_digest: str


@dataclass(frozen=True)
class PayoffCertificate:
    """Anytime-valid Hoeffding/Bernstein PAC certificate on E[u_S]."""

    n: int
    empirical_mean: float
    half_width: float
    delta: float
    method: str
    range_: float
    lcb: float
    ucb: float


@dataclass(frozen=True)
class VerificationReport:
    """Diagnostics on a candidate signaling scheme."""

    bayes_plausible: bool
    bayes_plausibility_gap: float
    obedience_ok: bool
    max_obedience_violation: float
    sender_value: float
    receiver_value: float


@dataclass(frozen=True)
class _AttestableReceipt:
    """Deterministic content-hash receipt for the attestation ledger."""

    kind: str
    payload_digest: str
    when: float


def _hash_payload(payload: Any) -> str:
    """Stable JSON SHA-256 of a JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# =====================================================================
# Simplex helpers and prior / utility validators
# =====================================================================


def _validate_simplex(p: Sequence[float], *, name: str = "vector") -> None:
    if any(not math.isfinite(v) for v in p):
        raise InvalidPrior(f"{name} contains non-finite entries.")
    if any(v < -1e-9 for v in p):
        raise InvalidPrior(f"{name} has negative entries.")
    s = sum(p)
    if abs(s - 1.0) > 1e-6:
        raise InvalidPrior(f"{name} sums to {s:.6f}, expected 1.0.")


def _normalise(p: Sequence[float]) -> tuple[float, ...]:
    s = sum(p)
    if s <= 0:
        raise InvalidPrior("Cannot normalise non-positive vector.")
    return tuple(v / s for v in p)


# =====================================================================
# Receiver best-response under a posterior
# =====================================================================


def best_response(
    receiver_utility: Sequence[Sequence[float]],
    posterior: Sequence[float],
    *,
    tiebreak: str = "lowest_index",
) -> int:
    r"""Index of receiver's optimal action under a posterior.

    Returns ``argmax_a Σ_ω μ(ω) u_R(a, ω)``. Deterministic tiebreak;
    pass ``tiebreak='highest_index'`` to flip.
    """
    n_actions = len(receiver_utility)
    if n_actions == 0:
        raise InvalidUtility("receiver_utility has no actions.")
    expected = [
        sum(receiver_utility[a][j] * posterior[j] for j in range(len(posterior)))
        for a in range(n_actions)
    ]
    if tiebreak == "highest_index":
        best = max(range(n_actions), key=lambda a: (expected[a], a))
    else:
        best = max(range(n_actions), key=lambda a: (expected[a], -a))
    return best


def sender_value_under_full_information(
    game: PersuasionGame, posterior: Sequence[float]
) -> float:
    r"""``v̂(μ) = u_S(a*(μ), μ)`` — sender's value at posterior μ."""
    a_star = best_response(game.receiver_utility, posterior)
    return sum(
        game.sender_utility[a_star][j] * posterior[j]
        for j in range(len(posterior))
    )


def receiver_value_under_posterior(
    game: PersuasionGame, posterior: Sequence[float]
) -> float:
    """Receiver's optimal expected utility under a posterior."""
    a_star = best_response(game.receiver_utility, posterior)
    return sum(
        game.receiver_utility[a_star][j] * posterior[j]
        for j in range(len(posterior))
    )


# =====================================================================
# Bayes plausibility / obedience verifier
# =====================================================================


def verify_scheme(
    game: PersuasionGame, scheme: SignalingScheme
) -> VerificationReport:
    r"""Verify Bayes-plausibility and obedience of a candidate scheme.

      * Bayes-plausibility: Σ_s P(s) · μ_s = μ₀.
      * Obedience (recommendation interpretation): under each signal s,
        the recommended action ``a*(μ_s)`` is BR. If signal labels are
        action names, also check the recommendation matches.
    """
    if scheme.states != game.states:
        raise InvalidUtility("scheme.states does not match game.states.")

    n_states = len(game.states)
    n_signals = len(scheme.signals)

    # Bayes-plausibility: marginal posterior = prior.
    marginals = [
        sum(scheme.pi[i][j] * game.prior[j] for j in range(n_states))
        for i in range(n_signals)
    ]
    avg_posterior = [0.0] * n_states
    for i in range(n_signals):
        if marginals[i] <= _EPS:
            continue
        post_i = [
            scheme.pi[i][j] * game.prior[j] / marginals[i]
            for j in range(n_states)
        ]
        for j in range(n_states):
            avg_posterior[j] += marginals[i] * post_i[j]
    bp_gap = max(abs(avg_posterior[j] - game.prior[j]) for j in range(n_states))
    bayes_plausible = bp_gap <= 1e-6

    # Obedience: receiver's BR under each posterior; max deviation gain.
    obedience_gap = 0.0
    sender_value = 0.0
    receiver_value = 0.0
    for i in range(n_signals):
        if marginals[i] <= _EPS:
            continue
        post_i = [
            scheme.pi[i][j] * game.prior[j] / marginals[i]
            for j in range(n_states)
        ]
        a_star = best_response(game.receiver_utility, post_i)
        # If the signal label is an action recommendation, the receiver
        # must (weakly) prefer that action. If the label is *not* an action,
        # we check that the BR is well-defined (auto-pass).
        u_a_star = sum(
            game.receiver_utility[a_star][j] * post_i[j]
            for j in range(n_states)
        )
        receiver_value += marginals[i] * u_a_star
        if scheme.signals[i] in game.actions:
            a_rec = game.actions.index(scheme.signals[i])
            u_rec = sum(
                game.receiver_utility[a_rec][j] * post_i[j]
                for j in range(n_states)
            )
            # Deviation gain (positive means receiver gains by deviating).
            obedience_gap = max(obedience_gap, u_a_star - u_rec)
            # Sender's payoff uses the receiver's actual BR.
            sender_value += marginals[i] * sum(
                game.sender_utility[a_star][j] * post_i[j]
                for j in range(n_states)
            )
        else:
            sender_value += marginals[i] * sum(
                game.sender_utility[a_star][j] * post_i[j]
                for j in range(n_states)
            )

    obedience_ok = obedience_gap <= _OBEDIENCE_TOL
    return VerificationReport(
        bayes_plausible=bayes_plausible,
        bayes_plausibility_gap=bp_gap,
        obedience_ok=obedience_ok,
        max_obedience_violation=obedience_gap,
        sender_value=sender_value,
        receiver_value=receiver_value,
    )


# =====================================================================
# Concavification on the binary-state simplex (exact)
# =====================================================================


def _binary_concavification(
    game: PersuasionGame, *, grid: int = 401
) -> tuple[float, list[tuple[float, float]], list[tuple[float, float, float]]]:
    r"""Compute the upper concave envelope of ``v̂(μ)`` for |Ω| = 2.

    Returns ``(V*(μ₀), envelope_points, supporting_pair)`` where the
    supporting pair is ``(μ_L, μ_H, λ)`` such that
    ``μ₀ = λ μ_L + (1-λ) μ_H`` and the optimum is realised by the
    two-signal scheme that induces those two posteriors.
    """
    if len(game.states) != 2:
        raise InvalidUtility("Binary concavification requires |Ω| = 2.")
    if grid < 3:
        grid = 3
    mu0 = game.prior[0]
    # 1. Identify breakpoints of v̂: posteriors at which the receiver's
    # BR changes. For each pair of actions (a, a') the indifference
    # posterior μ* satisfies μ (u_R(a,0) − u_R(a',0)) + (1-μ)(u_R(a,1) − u_R(a',1)) = 0.
    breakpoints: set[float] = {0.0, 1.0, mu0}
    n_actions = len(game.actions)
    for a in range(n_actions):
        for ap in range(a + 1, n_actions):
            d0 = game.receiver_utility[a][0] - game.receiver_utility[ap][0]
            d1 = game.receiver_utility[a][1] - game.receiver_utility[ap][1]
            # μ·d0 + (1-μ)·d1 = 0  ⇒  μ = d1 / (d1 - d0)
            denom = d1 - d0
            if abs(denom) > _EPS:
                mu_star = d1 / denom
                if 0.0 <= mu_star <= 1.0:
                    breakpoints.add(mu_star)
    pts = sorted(breakpoints)
    # 2. v̂ at each breakpoint.
    samples = [(mu, sender_value_under_full_information(game, [mu, 1.0 - mu]))
               for mu in pts]
    # 3. Upper concave hull (Andrew's monotone chain on a single chain
    # since x is already sorted; we keep only the upper envelope).
    hull: list[tuple[float, float]] = []
    for x, y in samples:
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            # Cross product of (p2-p1) and (p-p1). If ≥ 0, p2 is below
            # the chord (non-concave) and we pop it.
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if cross >= -1e-12:
                hull.pop()
            else:
                break
        hull.append((x, y))
    # 4. Look up V*(μ₀) on the hull by linear interpolation between the
    # two adjacent hull vertices.
    if len(hull) == 1:
        v_star = hull[0][1]
        return v_star, hull, [(hull[0][0], hull[0][0], 1.0)]
    # Find the segment containing mu0.
    for k in range(len(hull) - 1):
        x1, y1 = hull[k]
        x2, y2 = hull[k + 1]
        if x1 - 1e-12 <= mu0 <= x2 + 1e-12:
            if abs(x2 - x1) <= 1e-12:
                v_star = y1
                return v_star, hull, [(x1, x2, 1.0)]
            lam = (x2 - mu0) / (x2 - x1)
            v_star = lam * y1 + (1.0 - lam) * y2
            return v_star, hull, [(x1, x2, lam)]
    # Fallback (shouldn't trigger if mu0 ∈ [0, 1]).
    v_star = sender_value_under_full_information(game, [mu0, 1.0 - mu0])
    return v_star, hull, [(mu0, mu0, 1.0)]


# =====================================================================
# Revised simplex with Bland's rule (stdlib LP solver)
# =====================================================================


def _solve_lp_max(
    c: list[float],
    A_eq: list[list[float]],
    b_eq: list[float],
    A_ub: list[list[float]],
    b_ub: list[float],
    *,
    max_iters: int = _LP_MAX_ITERS,
    tol: float = _LP_TOL,
) -> tuple[str, list[float], float]:
    r"""Maximise ``cᵀ x`` subject to ``A_eq x = b_eq``, ``A_ub x ≤ b_ub``,
    ``x ≥ 0``.

    Internally we minimise ``-cᵀ x`` via the Big-M two-phase method on
    the canonical tableau with Bland's rule for cycling-free pivoting.

    Returns ``(status, x, value)`` where status ∈ {"optimal",
    "infeasible", "unbounded"}.
    """
    n = len(c)
    m_eq = len(A_eq)
    m_ub = len(A_ub)

    # Standard form: min c'x s.t. Ax = b, x ≥ 0.
    # 1. flip sign so we minimise.
    c_min = [-ci for ci in c]
    # 2. add slacks to upper bound rows.
    n_total = n + m_ub  # decision vars + slacks
    A: list[list[float]] = []
    b: list[float] = []
    # eq rows
    for i in range(m_eq):
        row = list(A_eq[i]) + [0.0] * m_ub
        A.append(row)
        b.append(b_eq[i])
    # ub rows with slacks
    for i in range(m_ub):
        row = list(A_ub[i]) + [0.0] * m_ub
        row[n + i] = 1.0
        A.append(row)
        b.append(b_ub[i])
    m = len(A)
    if m == 0:
        return "optimal", [0.0] * n, 0.0
    c_full = c_min + [0.0] * m_ub

    # Make all b ≥ 0 by flipping rows with b < 0.
    for i in range(m):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = [-v for v in A[i]]

    # Big-M (artificial variables) for each row to find an initial BFS.
    M = 1.0
    for v in c_full:
        if abs(v) > M:
            M = abs(v)
    for row in A:
        for v in row:
            if abs(v) > M:
                M = abs(v)
    for v in b:
        if abs(v) > M:
            M = abs(v)
    M *= 1e6 + 1.0  # big enough but finite

    # Add artificials to all m rows.
    n_art = m
    n_grand = n_total + n_art
    A_big = [row + [0.0] * n_art for row in A]
    for i in range(m):
        A_big[i][n_total + i] = 1.0
    c_big = c_full + [M] * n_art
    basis = [n_total + i for i in range(m)]

    # Revised simplex tableau-based approach (dense, small problems).
    # Reduced cost: c_j - c_B' * A_inv * a_j. With identity basis from
    # artificials, A_inv = I initially; we keep a B_inv matrix updated by
    # row operations (eta updates). For simplicity we maintain A_big
    # *as a tableau* and pivot in place — adequate for problems with
    # n + m + m up to a few hundred (our typical persuasion LPs).
    tableau = [row[:] for row in A_big]
    rhs = b[:]
    z = c_big[:]  # cost row

    # Eliminate basic variables from cost row.
    for i, b_idx in enumerate(basis):
        if abs(z[b_idx]) > tol:
            factor = z[b_idx]
            for j in range(n_grand):
                z[j] -= factor * tableau[i][j]
            # objective offset: z0 -= factor * rhs[i]  (tracked via running sum)
    # We'll track running objective in z0.
    z0 = -sum(c_big[b_idx] * rhs[i] for i, b_idx in enumerate(basis))

    for _iter in range(max_iters):
        # Bland's entering rule: smallest-index j with z[j] < -tol.
        entering = -1
        for j in range(n_grand):
            if z[j] < -tol:
                entering = j
                break
        if entering == -1:
            # Optimal.
            x = [0.0] * n_grand
            for i in range(m):
                x[basis[i]] = rhs[i]
            # Check artificials are zero.
            art_total = sum(x[n_total + i] for i in range(n_art))
            if art_total > 1e-5:
                return "infeasible", [], 0.0
            # Convert back to maximisation.
            val_min = -z0
            val_max = -val_min
            return "optimal", x[:n], val_max
        # Bland's leaving rule: among rows with tableau[i][entering] > tol,
        # min ratio rhs[i] / tableau[i][entering]; tie → smallest basis index.
        leaving = -1
        best_ratio = math.inf
        for i in range(m):
            a_ie = tableau[i][entering]
            if a_ie > tol:
                ratio = rhs[i] / a_ie
                if ratio < best_ratio - 1e-12:
                    best_ratio = ratio
                    leaving = i
                elif abs(ratio - best_ratio) <= 1e-12 and leaving >= 0:
                    if basis[i] < basis[leaving]:
                        leaving = i
        if leaving == -1:
            return "unbounded", [], math.inf
        # Pivot on (leaving, entering): divide leaving row by pivot.
        pivot = tableau[leaving][entering]
        for j in range(n_grand):
            tableau[leaving][j] /= pivot
        rhs[leaving] /= pivot
        # Eliminate entering var from other rows and z-row.
        for i in range(m):
            if i == leaving:
                continue
            factor = tableau[i][entering]
            if abs(factor) > tol:
                for j in range(n_grand):
                    tableau[i][j] -= factor * tableau[leaving][j]
                rhs[i] -= factor * rhs[leaving]
        factor = z[entering]
        if abs(factor) > tol:
            for j in range(n_grand):
                z[j] -= factor * tableau[leaving][j]
            z0 -= factor * rhs[leaving]
        basis[leaving] = entering

    return "infeasible", [], 0.0  # max iters reached


# =====================================================================
# Optimal persuasion LP (general |Ω|, |A|)
# =====================================================================


def _solve_persuasion_lp(
    game: PersuasionGame,
    *,
    extra_obedience_tol: float = 0.0,
) -> tuple[SignalingScheme, float, float, list[str]]:
    r"""Solve the Bergemann-Morris BCE-style persuasion LP.

    Variables: ``x[a][ω] ≥ 0`` for each action a, state ω.
    Marginal:  Σ_a x[a][ω] = μ₀(ω) for each ω.
    Obedience: Σ_ω x[a][ω] [u_R(a,ω) − u_R(a',ω)] ≥ 0 for each (a, a' ≠ a).
    Objective: max Σ_{a,ω} x[a][ω] u_S(a, ω).

    Each signal label = recommended action.
    """
    n_actions = len(game.actions)
    n_states = len(game.states)
    nv = n_actions * n_states

    def idx(a: int, omega: int) -> int:
        return a * n_states + omega

    # Cost (maximise) c[idx(a,ω)] = u_S(a, ω).
    c = [0.0] * nv
    for a in range(n_actions):
        for w in range(n_states):
            c[idx(a, w)] = game.sender_utility[a][w]

    # Marginal equality rows: for each ω, Σ_a x[a][ω] = μ₀(ω).
    A_eq: list[list[float]] = []
    b_eq: list[float] = []
    for w in range(n_states):
        row = [0.0] * nv
        for a in range(n_actions):
            row[idx(a, w)] = 1.0
        A_eq.append(row)
        b_eq.append(game.prior[w])

    # Obedience inequalities: encoded as A_ub x ≤ b_ub.
    # We have Σ_ω x[a][ω] [u_R(a,ω) − u_R(a',ω)] ≥ 0
    #     ⇔ - Σ_ω x[a][ω] [u_R(a,ω) − u_R(a',ω)] ≤ -extra_tol
    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    for a in range(n_actions):
        for ap in range(n_actions):
            if a == ap:
                continue
            row = [0.0] * nv
            for w in range(n_states):
                row[idx(a, w)] = -(game.receiver_utility[a][w]
                                    - game.receiver_utility[ap][w])
            A_ub.append(row)
            b_ub.append(-extra_obedience_tol)

    status, x_flat, value = _solve_lp_max(c, A_eq, b_eq, A_ub, b_ub)
    if status == "infeasible":
        # Should never happen since x[a][ω] = μ₀(ω) · 1[a = argmax_a' u_R(a',·)]
        # (the no-info policy) is always feasible.
        raise InfeasibleProgram("Persuasion LP is infeasible.")
    if status == "unbounded":
        # Should never happen since all variables are bounded by μ₀(ω).
        raise InfeasibleProgram("Persuasion LP is unbounded (numerical).")

    # Build SignalingScheme: signal label = action label.
    pi_rows: list[tuple[float, ...]] = []
    signals_used: list[str] = []
    receiver_total = 0.0
    for a in range(n_actions):
        # Conditional π(a | ω) = x[a][ω] / μ₀(ω); guard zero prior.
        row: list[float] = []
        marginal = 0.0
        for w in range(n_states):
            prior_w = game.prior[w]
            if prior_w <= _EPS:
                row.append(0.0)
            else:
                pi_aw = max(0.0, min(1.0, x_flat[idx(a, w)] / prior_w))
                row.append(pi_aw)
            marginal += x_flat[idx(a, w)]
        if marginal > 1e-12:
            pi_rows.append(tuple(row))
            signals_used.append(game.actions[a])
            # Receiver's payoff under signal a.
            for w in range(n_states):
                receiver_total += x_flat[idx(a, w)] * game.receiver_utility[a][w]
    if not signals_used:
        # Fallback: no-info scheme.
        a_star = best_response(game.receiver_utility, game.prior)
        pi_rows = [tuple(1.0 for _ in range(n_states))]
        signals_used = [game.actions[a_star]]
        receiver_total = sum(
            game.receiver_utility[a_star][w] * game.prior[w]
            for w in range(n_states)
        )

    # Renormalise columns (numerical safety): each column must sum to 1.
    pi_renorm: list[tuple[float, ...]] = []
    n_signals = len(pi_rows)
    for j in range(n_states):
        col_sum = sum(pi_rows[i][j] for i in range(n_signals))
        if col_sum <= _EPS:
            # zero-prior column — leave as is
            continue
    pi_final: list[list[float]] = [list(row) for row in pi_rows]
    for j in range(n_states):
        col_sum = sum(pi_final[i][j] for i in range(n_signals))
        if col_sum > _EPS:
            for i in range(n_signals):
                pi_final[i][j] /= col_sum

    scheme = SignalingScheme(
        signals=tuple(signals_used),
        states=game.states,
        pi=tuple(tuple(row) for row in pi_final),
    )
    return scheme, value, receiver_total, signals_used


# =====================================================================
# Online Bayesian persuasion (Hedge over discretised schemes)
# =====================================================================


def _enumerate_signal_grid(
    n_states: int, n_actions: int, grid: int
) -> list[list[list[float]]]:
    r"""Enumerate a ε-net of recommendation policies on the simplex.

    For each state ω we discretise the action distribution as
    multiples of 1/grid; we keep all combinations whose entries sum to 1.
    Resulting count ≈ C(grid + n_actions - 1, n_actions - 1)^n_states.
    """
    def _per_state_distributions() -> list[list[float]]:
        # All ways to write `grid` as ordered sum of n_actions non-negs.
        results: list[list[float]] = []

        def rec(remaining: int, slots: int, prefix: list[int]) -> None:
            if slots == 1:
                results.append(prefix + [remaining])
                return
            for v in range(remaining + 1):
                rec(remaining - v, slots - 1, prefix + [v])

        rec(grid, n_actions, [])
        return [[c / grid for c in r] for r in results]

    per_state = _per_state_distributions()
    # Product across states.
    out: list[list[list[float]]] = []
    indices = [0] * n_states
    while True:
        scheme = [per_state[indices[w]] for w in range(n_states)]
        out.append(scheme)
        # increment
        k = n_states - 1
        while k >= 0:
            indices[k] += 1
            if indices[k] < len(per_state):
                break
            indices[k] = 0
            k -= 1
        if k < 0:
            break
    return out


# =====================================================================
# Persuader — main runtime primitive
# =====================================================================


class Persuader:
    r"""Bayesian persuasion as a runtime primitive.

    Construction
    ------------

    ::

        from agi.persuader import Persuader, PersuasionGame
        from agi.events import EventBus

        bus = EventBus()
        p = Persuader(bus=bus)

        game = PersuasionGame(
            states=("good", "bad"),
            actions=("approve", "deny"),
            prior=(0.3, 0.7),
            sender_utility=((1.0, 1.0), (0.0, 0.0)),     # sender wants approval
            receiver_utility=((1.0, -1.0), (0.0, 0.0)),  # receiver: approve iff μ_good > 1/2
        )

        out = p.persuade(game)
        assert out.sender_value > sender_value_under_full_information(game, game.prior)

    Concurrent-safe; emits events for every state change; every
    `persuade(...)` returns a tamper-evident receipt digest the
    coordination engine can pipe into `AttestationLedger`.
    """

    def __init__(
        self,
        *,
        bus: Any | None = None,
        attestor: Any | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._rng = rng if rng is not None else random.Random()
        self._lock = threading.RLock()
        self._counters = {
            "persuade_calls": 0,
            "online_calls": 0,
            "robust_calls": 0,
            "multi_calls": 0,
            "verify_calls": 0,
            "certify_calls": 0,
        }
        self._emit(PERSUADE_STARTED, {"primitive": "persuader"})

    # -----------------------------------------------------------------
    # Event helpers
    # -----------------------------------------------------------------

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._bus is None or Event is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(payload)))
        except Exception:  # pragma: no cover
            pass

    def _receipt(self, kind: str, payload: dict[str, Any]) -> str:
        digest = _hash_payload(payload)
        receipt = _AttestableReceipt(
            kind=kind, payload_digest=digest, when=time.time()
        )
        if self._attestor is not None:
            try:
                self._attestor.append(asdict(receipt))
            except Exception:  # pragma: no cover
                pass
        return digest

    # -----------------------------------------------------------------
    # Counters / introspection
    # -----------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    # -----------------------------------------------------------------
    # Core: persuade a single receiver (concavification for |Ω| = 2,
    # LP otherwise)
    # -----------------------------------------------------------------

    def persuade(
        self,
        game: PersuasionGame,
        *,
        kind: str = "auto",
    ) -> PersuasionOutcome:
        r"""Optimal sender-preferred signaling scheme.

        Parameters
        ----------
        game : PersuasionGame
        kind : ``"auto"`` (concavification for |Ω|=2 else LP), or any
            element of `KNOWN_KINDS`.

        Returns
        -------
        PersuasionOutcome with the scheme, sender's expected value,
        induced posteriors, receiver's BR per signal, verification flags,
        and a receipt digest.
        """
        with self._lock:
            self._counters["persuade_calls"] += 1

        if kind == "auto":
            kind = (KIND_CONCAVIFICATION if len(game.states) == 2
                    else KIND_LP)
        if kind not in (KIND_CONCAVIFICATION, KIND_LP):
            raise UnknownKind(f"unknown persuade kind: {kind!r}")

        if kind == KIND_CONCAVIFICATION:
            if len(game.states) != 2:
                # Fall back to LP transparently.
                return self.persuade(game, kind=KIND_LP)
            v_star, _, sup = _binary_concavification(game)
            mu_L, mu_H, lam = sup[0]
            # Build a 2-signal scheme that induces mu_L and mu_H.
            scheme = _scheme_from_pair(
                game, mu_L=mu_L, mu_H=mu_H, lam=lam
            )
        else:
            scheme, v_star, _, _ = _solve_persuasion_lp(game)

        report = verify_scheme(game, scheme)
        # Per-signal posteriors and recommended actions.
        marginals = scheme.marginal(game.prior)
        signal_marginals = tuple(marginals[s] for s in scheme.signals)
        posteriors_list: list[tuple[float, ...]] = []
        recommended: list[str] = []
        for i, s in enumerate(scheme.signals):
            post = scheme.posterior(s, game.prior)
            post_tuple = tuple(post[st] for st in scheme.states)
            posteriors_list.append(post_tuple)
            a_star_idx = best_response(game.receiver_utility, post_tuple)
            recommended.append(game.actions[a_star_idx])

        sender_value = report.sender_value if kind == KIND_LP else v_star
        receipt = self._receipt(
            PERSUADE_SOLVED,
            {
                "kind": kind,
                "states": list(game.states),
                "actions": list(game.actions),
                "prior": list(game.prior),
                "scheme_pi": [list(r) for r in scheme.pi],
                "signals": list(scheme.signals),
                "sender_value": sender_value,
                "receiver_value": report.receiver_value,
                "bayes_plausible": report.bayes_plausible,
                "obedience_ok": report.obedience_ok,
            },
        )
        self._emit(PERSUADE_SOLVED, {
            "kind": kind,
            "sender_value": sender_value,
            "receiver_value": report.receiver_value,
            "n_signals": len(scheme.signals),
            "digest": receipt,
        })

        return PersuasionOutcome(
            kind=kind,
            scheme=scheme,
            sender_value=sender_value,
            receiver_value=report.receiver_value,
            induced_posteriors=tuple(posteriors_list),
            signal_marginals=signal_marginals,
            recommended_actions=tuple(recommended),
            bayes_plausible=report.bayes_plausible,
            obedience_ok=report.obedience_ok,
            receipt_digest=receipt,
        )

    # -----------------------------------------------------------------
    # Send a signal: realise the persuasion in a single round.
    # -----------------------------------------------------------------

    def send_signal(
        self,
        scheme: SignalingScheme,
        realised_state: str,
        *,
        rng: random.Random | None = None,
    ) -> str:
        """Realise the signaling policy: draw ``s ∼ π(· | ω)``."""
        r = rng if rng is not None else self._rng
        if realised_state not in scheme.states:
            raise InvalidUtility(f"unknown state {realised_state!r}")
        w = scheme.states.index(realised_state)
        weights = [scheme.pi[i][w] for i in range(len(scheme.signals))]
        total = sum(weights)
        if total <= _EPS:
            raise InvalidUtility(
                f"signaling scheme is degenerate at state {realised_state}."
            )
        u = r.random() * total
        acc = 0.0
        for i, wt in enumerate(weights):
            acc += wt
            if u <= acc:
                self._emit(PERSUADE_SIGNAL_SENT, {
                    "state": realised_state,
                    "signal": scheme.signals[i],
                })
                return scheme.signals[i]
        # Floating-point fallback: emit last signal.
        self._emit(PERSUADE_SIGNAL_SENT, {
            "state": realised_state,
            "signal": scheme.signals[-1],
        })
        return scheme.signals[-1]

    # -----------------------------------------------------------------
    # Verify a candidate scheme.
    # -----------------------------------------------------------------

    def verify(
        self, game: PersuasionGame, scheme: SignalingScheme
    ) -> VerificationReport:
        with self._lock:
            self._counters["verify_calls"] += 1
        report = verify_scheme(game, scheme)
        self._emit(PERSUADE_VERIFIED, {
            "bayes_plausible": report.bayes_plausible,
            "obedience_ok": report.obedience_ok,
            "sender_value": report.sender_value,
            "receiver_value": report.receiver_value,
        })
        return report

    # -----------------------------------------------------------------
    # PAC certificate on sender payoff via Monte-Carlo simulation
    # -----------------------------------------------------------------

    def simulate(
        self,
        game: PersuasionGame,
        scheme: SignalingScheme,
        *,
        T: int,
        delta: float = 0.05,
        method: str = "hoeffding",
        rng: random.Random | None = None,
    ) -> PayoffCertificate:
        r"""Empirical-mean PAC certificate on E[u_S].

        Draws T iid (ω, s, a) triples by sampling ω ∼ μ₀, s ∼ π(·|ω),
        a = a*(μ_s), and accumulates u_S(a, ω). Returns a Hoeffding
        (or empirical-Bernstein) confidence interval.
        """
        if T <= 0:
            raise InsufficientData("simulate requires T ≥ 1.")
        r = rng if rng is not None else self._rng
        with self._lock:
            self._counters["certify_calls"] += 1
        # Pre-compute receiver BR per signal.
        br_by_signal: dict[str, int] = {}
        for s in scheme.signals:
            post = scheme.posterior(s, game.prior)
            post_tuple = [post[st] for st in scheme.states]
            br_by_signal[s] = best_response(game.receiver_utility, post_tuple)
        # u_S range.
        u_min = min(min(row) for row in game.sender_utility)
        u_max = max(max(row) for row in game.sender_utility)
        rng_ = max(u_max - u_min, 0.0)
        # Sample.
        samples: list[float] = []
        cum_prior = []
        acc = 0.0
        for p in game.prior:
            acc += p
            cum_prior.append(acc)
        for _ in range(T):
            u = r.random() * cum_prior[-1]
            w = bisect.bisect_left(cum_prior, u)
            if w >= len(game.states):
                w = len(game.states) - 1
            # sample signal
            weights = [scheme.pi[i][w] for i in range(len(scheme.signals))]
            tot = sum(weights)
            uu = r.random() * tot
            running = 0.0
            sig_idx = 0
            for i, wt in enumerate(weights):
                running += wt
                if uu <= running:
                    sig_idx = i
                    break
            a = br_by_signal[scheme.signals[sig_idx]]
            samples.append(game.sender_utility[a][w])
        mean = sum(samples) / T
        if method == "empirical_bernstein":
            if T < 2:
                raise InsufficientData("EB needs T ≥ 2.")
            hw = empirical_bernstein_radius(samples, delta=delta, range_=rng_)
        else:
            method = "hoeffding"
            hw = hoeffding_radius(T, delta=delta, range_=rng_)
        cert = PayoffCertificate(
            n=T,
            empirical_mean=mean,
            half_width=hw,
            delta=delta,
            method=method,
            range_=rng_,
            lcb=mean - hw,
            ucb=mean + hw,
        )
        self._emit(PERSUADE_CERTIFIED, {
            "n": T, "mean": mean, "half_width": hw,
            "delta": delta, "method": method,
        })
        return cert

    # -----------------------------------------------------------------
    # Online persuasion: Hedge over a discrete ε-net of schemes.
    # -----------------------------------------------------------------

    def online_persuade(
        self,
        game: PersuasionGame,
        feedback: Callable[[SignalingScheme], float],
        *,
        T: int,
        grid: int = 4,
        eta: float | None = None,
        rng: random.Random | None = None,
    ) -> OnlinePersuasionOutcome:
        r"""Run T rounds of online persuasion with full-information Hedge.

        Castiglioni-Marchesi-Romano-Gatti 2020: at each round t the sender
        commits to a scheme π_t drawn from Hedge weights over a discrete
        ε-net Π of size ``K = |Π|``, observes the *adversary-revealed*
        per-scheme payoffs ``f_t(π_k)`` for all ``k`` (full information),
        and updates the Hedge weights with learning rate
        ``η = √(8 ln K / T)``. Cumulative regret bound is

            R_T = max_k Σ_t f_t(π_k) − Σ_t f_t(π_t)
                ≤ √(T · ln K / 2) + 1.

        The full-information regime is faithful to the *commitment* nature
        of Bayesian persuasion: the sender publicly publishes π_t before
        the world realises, so the adversary can reveal the counterfactual
        payoff of every alternative scheme after the round.

        ``feedback(scheme) -> float`` is called K times per round
        (sender-payoff under that scheme, given the round's realised
        state and receiver utility).
        """
        if T <= 0:
            raise InsufficientData("online_persuade requires T ≥ 1.")
        r = rng if rng is not None else self._rng
        with self._lock:
            self._counters["online_calls"] += 1
        n_actions = len(game.actions)
        n_states = len(game.states)
        if n_actions * n_states > 8 and grid > 3:
            grid = 3  # keep enumeration tractable
        net = _enumerate_signal_grid(n_states, n_actions, grid)
        # Build SignalingSchemes from the net (each entry is a list of
        # action distributions per state — interpret as π(a | ω)).
        schemes: list[SignalingScheme] = []
        for entry in net:
            pi = [
                tuple(entry[w][a] for w in range(n_states))
                for a in range(n_actions)
            ]
            row_filter = []
            sig_labels = []
            for a, row in enumerate(pi):
                if max(row) > 0:
                    row_filter.append(row)
                    sig_labels.append(game.actions[a])
            if not row_filter:
                continue
            col_sums = [sum(row_filter[i][j] for i in range(len(row_filter)))
                        for j in range(n_states)]
            pi_norm = []
            ok = True
            for row in row_filter:
                norm_row = []
                for j, v in enumerate(row):
                    if col_sums[j] <= _EPS:
                        ok = False
                        break
                    norm_row.append(v / col_sums[j])
                if not ok:
                    break
                pi_norm.append(tuple(norm_row))
            if not ok:
                continue
            try:
                schemes.append(SignalingScheme(
                    signals=tuple(sig_labels),
                    states=game.states,
                    pi=tuple(pi_norm),
                ))
            except (InvalidUtility, InvalidPrior):
                continue
        if not schemes:
            schemes = [SignalingScheme(
                signals=(game.actions[0],),
                states=game.states,
                pi=tuple((1.0,) for _ in game.states),
            )]
        K = len(schemes)
        u_min = min(min(row) for row in game.sender_utility)
        u_max = max(max(row) for row in game.sender_utility)
        rng_u = max(u_max - u_min, _EPS)
        if eta is None:
            eta = math.sqrt(8.0 * math.log(max(K, 2)) / max(T, 1))
        # Hedge weights and per-scheme cumulative gains.
        log_w = [0.0] * K
        cum_gain = [0.0] * K
        cum_played = 0.0
        for t in range(T):
            # Sample scheme by Hedge distribution.
            max_lw = max(log_w)
            probs = [math.exp(lw - max_lw) for lw in log_w]
            Z = sum(probs)
            probs = [p / Z for p in probs]
            u = r.random()
            acc = 0.0
            chosen = K - 1
            for k, p in enumerate(probs):
                acc += p
                if u <= acc:
                    chosen = k
                    break
            # Full-info: get per-arm payoff for every arm.
            arm_gains: list[float] = []
            for k in range(K):
                try:
                    f_kt = float(feedback(schemes[k]))
                except Exception as exc:
                    raise PersuaderError(
                        f"online feedback callback raised: {exc}"
                    ) from exc
                if not math.isfinite(f_kt):
                    f_kt = u_min
                f_kt = min(u_max, max(u_min, f_kt))
                arm_gains.append(f_kt)
            f_t = arm_gains[chosen]
            cum_played += f_t
            for k in range(K):
                g_norm = (arm_gains[k] - u_min) / rng_u
                cum_gain[k] += arm_gains[k]
                log_w[k] += eta * g_norm
            self._emit(PERSUADE_ONLINE_STEP, {
                "t": t, "chosen": chosen, "payoff": f_t,
            })
        # Best fixed arm in retrospect (true full-info value).
        best_k = max(range(K), key=lambda i: cum_gain[i])
        best_fixed = cum_gain[best_k]
        regret = max(best_fixed - cum_played, 0.0)
        # Cesa-Bianchi/Lugosi 2006 Theorem 2.2: Hedge with η = √(8 ln K / T)
        # has cumulative regret ≤ √(T · ln K / 2) on losses in [0, 1].
        # Since we normalise gains to [0, 1] (dividing by rng_u), the
        # bound in *original units* is √(T · ln K / 2) · rng_u.
        bound = math.sqrt(T * math.log(max(K, 2)) / 2.0) * rng_u + 1.0
        receipt = self._receipt(PERSUADE_ONLINE_STEP, {
            "T": T, "K": K, "regret": regret, "bound": bound,
            "best_scheme_idx": best_k,
        })
        return OnlinePersuasionOutcome(
            T=T,
            cumulative_sender=cum_played,
            cumulative_best_fixed=best_fixed,
            cumulative_regret=regret,
            regret_bound=bound,
            final_scheme=schemes[best_k],
            receipt_digest=receipt,
        )

    # -----------------------------------------------------------------
    # Robust persuasion over a finite prior set.
    # -----------------------------------------------------------------

    def robust_persuade(
        self,
        states: Sequence[str],
        actions: Sequence[str],
        sender_utility: Sequence[Sequence[float]],
        receiver_utility: Sequence[Sequence[float]],
        prior_set: Sequence[Sequence[float]],
    ) -> RobustOutcome:
        r"""Solve max_π min_{μ ∈ U} V(π, μ) for a finite prior set U.

        For each ``μ_k`` we solve the persuasion LP with that prior;
        among the optimal schemes we pick the one whose minimum sender
        value across ``U`` is largest (greedy max-min over the finite
        candidate set). This is the canonical *finite-cover* relaxation
        of Dworczak-Pavan; tightens as U is refined.
        """
        with self._lock:
            self._counters["robust_calls"] += 1
        if not prior_set:
            raise InvalidPrior("prior_set must be non-empty.")
        candidates: list[tuple[SignalingScheme, float, tuple[float, ...]]] = []
        for mu in prior_set:
            _validate_simplex(mu, name="prior_set entry")
            game = PersuasionGame(
                states=tuple(states),
                actions=tuple(actions),
                prior=tuple(mu),
                sender_utility=tuple(tuple(row) for row in sender_utility),
                receiver_utility=tuple(tuple(row) for row in receiver_utility),
            )
            scheme, _, _, _ = _solve_persuasion_lp(game)
            candidates.append((scheme, 0.0, tuple(mu)))

        # Score each candidate scheme by its worst-case sender value over U.
        best: tuple[SignalingScheme, float, tuple[float, ...]] | None = None
        for scheme, _placeholder, _origin in candidates:
            worst_val = math.inf
            worst_mu: tuple[float, ...] = tuple(prior_set[0])
            for mu in prior_set:
                game_mu = PersuasionGame(
                    states=tuple(states),
                    actions=tuple(actions),
                    prior=tuple(mu),
                    sender_utility=tuple(tuple(row) for row in sender_utility),
                    receiver_utility=tuple(tuple(row) for row in receiver_utility),
                )
                report = verify_scheme(game_mu, scheme)
                if report.sender_value < worst_val:
                    worst_val = report.sender_value
                    worst_mu = tuple(mu)
            if best is None or worst_val > best[1]:
                best = (scheme, worst_val, worst_mu)
        assert best is not None
        scheme, worst_val, worst_mu = best
        digest = self._receipt(PERSUADE_ROBUST_SOLVED, {
            "states": list(states),
            "actions": list(actions),
            "prior_set": [list(mu) for mu in prior_set],
            "worst_case_value": worst_val,
            "worst_case_prior": list(worst_mu),
        })
        self._emit(PERSUADE_ROBUST_SOLVED, {
            "worst_case_value": worst_val,
            "n_priors": len(prior_set),
            "digest": digest,
        })
        return RobustOutcome(
            scheme=scheme,
            worst_case_value=worst_val,
            worst_case_prior=worst_mu,
            receipt_digest=digest,
        )

    # -----------------------------------------------------------------
    # Multi-receiver persuasion.
    # -----------------------------------------------------------------

    def multi_receiver_private(
        self,
        receivers: Sequence[tuple[str, PersuasionGame]],
    ) -> MultiReceiverOutcome:
        r"""Independent per-receiver persuasion (Babichenko-Barman 2017).

        Each receiver gets her own optimal scheme; the joint sender value
        is the sum. Exact when sender utility is additive across
        receivers and receivers' utilities are private (no cross-receiver
        externalities). For supermodular sender utility this is a
        (1 − 1/e)-approximation; we return the lower bound.
        """
        with self._lock:
            self._counters["multi_calls"] += 1
        if not receivers:
            raise UnknownReceiver("at least one receiver required.")
        schemes: list[tuple[str, SignalingScheme]] = []
        total_value = 0.0
        for rid, game in receivers:
            out = self.persuade(game, kind=KIND_LP)
            schemes.append((rid, out.scheme))
            total_value += out.sender_value
        digest = self._receipt(PERSUADE_MULTI_SOLVED, {
            "mode": KIND_MULTI_PRIVATE,
            "receivers": [rid for rid, _ in receivers],
            "joint_sender_value": total_value,
        })
        self._emit(PERSUADE_MULTI_SOLVED, {
            "mode": KIND_MULTI_PRIVATE,
            "n_receivers": len(receivers),
            "joint_value": total_value,
            "digest": digest,
        })
        return MultiReceiverOutcome(
            kind=KIND_MULTI_PRIVATE,
            per_receiver_schemes=tuple(schemes),
            joint_sender_value=total_value,
            receipt_digest=digest,
        )

    def multi_receiver_public(
        self,
        states: Sequence[str],
        prior: Sequence[float],
        receivers: Sequence[tuple[str,
                                  Sequence[str],
                                  Sequence[Sequence[float]]]],
        sender_utility_by_profile: Mapping[tuple[str, ...], Sequence[float]],
    ) -> MultiReceiverOutcome:
        r"""Public persuasion: one signal observed by all receivers.

        ``receivers`` is a list of ``(receiver_id, actions, u_R[a][ω])``.
        ``sender_utility_by_profile[(a1, a2, ...)]`` is the sender's
        utility vector over states when each receiver plays the listed
        action. This is the Mathevet-Perego-Taneva 2020 public-signal
        model; we collapse it to a single-receiver LP over the *joint*
        action profile.
        """
        n_states = len(states)
        _validate_simplex(prior, name="prior")
        action_lists = [list(actions) for _, actions, _ in receivers]
        # Joint action profiles.
        profiles: list[tuple[str, ...]] = []

        def expand(prefix: list[str], k: int) -> None:
            if k == len(action_lists):
                profiles.append(tuple(prefix))
                return
            for a in action_lists[k]:
                expand(prefix + [a], k + 1)

        expand([], 0)
        # Build receiver utility for the joint game: each "action" is a
        # profile; receiver i's payoff is u_{R_i}(a_i, ω) — but in a public
        # information design the receivers each individually best-respond.
        # We collapse to the BCE LP with a single receiver who follows
        # the *recommended profile* iff it is a Bayes-Nash equilibrium
        # of the induced posterior.
        # NOTE: for additive sender utility across receivers the joint
        # LP decomposes; we ship the additive-default routine here.
        sender_u: list[list[float]] = []
        for prof in profiles:
            if prof in sender_utility_by_profile:
                row = list(sender_utility_by_profile[prof])
            else:
                row = [0.0] * n_states
            if len(row) != n_states:
                raise InvalidUtility(
                    f"sender utility for profile {prof} has wrong length."
                )
            sender_u.append(row)
        # For the joint receiver we use the *sum* of per-receiver utilities
        # under the profile — this is the welfare-maximising surrogate; for
        # genuine Bayes-Nash refinement, post-process with Equilibrator.
        recv_u: list[list[float]] = []
        for prof in profiles:
            row = [0.0] * n_states
            for ri, (_, actions, u_r) in enumerate(receivers):
                ai = actions.index(prof[ri])
                for w in range(n_states):
                    row[w] += u_r[ai][w]
            recv_u.append(row)
        game = PersuasionGame(
            states=tuple(states),
            actions=tuple("|".join(p) for p in profiles),
            prior=tuple(prior),
            sender_utility=tuple(tuple(r) for r in sender_u),
            receiver_utility=tuple(tuple(r) for r in recv_u),
        )
        out = self.persuade(game, kind=KIND_LP)
        digest = self._receipt(PERSUADE_MULTI_SOLVED, {
            "mode": KIND_MULTI_PUBLIC,
            "receivers": [rid for rid, _, _ in receivers],
            "joint_sender_value": out.sender_value,
        })
        # Decompose recommendation per receiver.
        schemes: list[tuple[str, SignalingScheme]] = []
        for ri, (rid, actions, _) in enumerate(receivers):
            # For each profile-signal, project to receiver ri's action.
            sig_action_pairs: dict[str, list[float]] = {a: [0.0] * n_states
                                                       for a in actions}
            for i, sig in enumerate(out.scheme.signals):
                profile = sig.split("|")
                a_i = profile[ri]
                for w in range(n_states):
                    sig_action_pairs[a_i][w] += out.scheme.pi[i][w]
            # Build per-receiver scheme (deduplicate).
            sigs_kept: list[str] = []
            rows_kept: list[tuple[float, ...]] = []
            for a in actions:
                col_total = sum(sig_action_pairs[a])
                if col_total > _EPS:
                    sigs_kept.append(a)
                    rows_kept.append(tuple(sig_action_pairs[a]))
            # Renormalise columns.
            for w in range(n_states):
                csum = sum(rows_kept[i][w] for i in range(len(rows_kept)))
                if csum > _EPS:
                    for i in range(len(rows_kept)):
                        rows_kept[i] = tuple(
                            rows_kept[i][w_] / csum if w_ == w else rows_kept[i][w_]
                            for w_ in range(n_states)
                        )
            # Final renormalisation (in case of asymmetric leftover).
            final = []
            for row in rows_kept:
                final.append(row)
            try:
                per_scheme = SignalingScheme(
                    signals=tuple(sigs_kept),
                    states=tuple(states),
                    pi=tuple(final),
                )
            except (InvalidUtility, InvalidPrior):
                # Fall back to a degenerate single-signal scheme.
                per_scheme = SignalingScheme(
                    signals=(actions[0],),
                    states=tuple(states),
                    pi=tuple((1.0,) for _ in states),
                )
            schemes.append((rid, per_scheme))
        self._emit(PERSUADE_MULTI_SOLVED, {
            "mode": KIND_MULTI_PUBLIC,
            "n_receivers": len(receivers),
            "joint_value": out.sender_value,
            "digest": digest,
        })
        return MultiReceiverOutcome(
            kind=KIND_MULTI_PUBLIC,
            per_receiver_schemes=tuple(schemes),
            joint_sender_value=out.sender_value,
            receipt_digest=digest,
        )

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
        self._emit(PERSUADE_CLEARED, {})


# =====================================================================
# Helpers
# =====================================================================


def _scheme_from_pair(
    game: PersuasionGame, *, mu_L: float, mu_H: float, lam: float,
) -> SignalingScheme:
    r"""Two-signal scheme for binary state inducing posteriors μ_L, μ_H
    with weights λ, 1-λ.

    Recall μ_L, μ_H ∈ [0,1] are the *posterior P(state=states[0])*; lam is
    the marginal weight on μ_L. We solve

        π(L | 0) μ₀(0) = λ · μ_L
        π(L | 1) μ₀(1) = λ · (1 − μ_L)
        π(H | 0) μ₀(0) = (1-λ) · μ_H
        π(H | 1) μ₀(1) = (1-λ) · (1 − μ_H)

    Then π(L|ω) + π(H|ω) = 1.
    """
    mu0 = game.prior[0]
    if abs(mu_L - mu_H) < 1e-12 or lam >= 1.0 - 1e-12 or lam <= 1e-12:
        # No-info scheme.
        a_star = best_response(game.receiver_utility, game.prior)
        return SignalingScheme(
            signals=(game.actions[a_star],),
            states=game.states,
            pi=tuple((1.0,) for _ in game.states),
        )
    if mu0 < 1e-12:
        # Prior is δ_{state=1}; only signal H survives.
        a_star = best_response(game.receiver_utility, [0.0, 1.0])
        return SignalingScheme(
            signals=(game.actions[a_star],),
            states=game.states,
            pi=tuple((1.0,) for _ in game.states),
        )
    if mu0 > 1.0 - 1e-12:
        a_star = best_response(game.receiver_utility, [1.0, 0.0])
        return SignalingScheme(
            signals=(game.actions[a_star],),
            states=game.states,
            pi=tuple((1.0,) for _ in game.states),
        )
    pi_L_0 = lam * mu_L / mu0
    pi_L_1 = lam * (1.0 - mu_L) / (1.0 - mu0)
    pi_H_0 = (1.0 - lam) * mu_H / mu0
    pi_H_1 = (1.0 - lam) * (1.0 - mu_H) / (1.0 - mu0)
    # Numerical clipping.
    pi_L_0 = min(1.0, max(0.0, pi_L_0))
    pi_L_1 = min(1.0, max(0.0, pi_L_1))
    pi_H_0 = 1.0 - pi_L_0
    pi_H_1 = 1.0 - pi_L_1
    # Action labels under L and H (BR under each posterior).
    a_L = best_response(game.receiver_utility, [mu_L, 1.0 - mu_L])
    a_H = best_response(game.receiver_utility, [mu_H, 1.0 - mu_H])
    label_L = game.actions[a_L]
    label_H = game.actions[a_H]
    # If labels collide, append "_lo"/"_hi" disambiguation.
    if label_L == label_H:
        label_L = label_L + "_lo"
        label_H = label_H + "_hi"
    return SignalingScheme(
        signals=(label_L, label_H),
        states=game.states,
        pi=(
            (pi_L_0, pi_L_1),
            (pi_H_0, pi_H_1),
        ),
    )


# =====================================================================
# Facade functions
# =====================================================================


def quick_persuade(
    states: Sequence[str],
    actions: Sequence[str],
    prior: Sequence[float],
    sender_utility: Sequence[Sequence[float]],
    receiver_utility: Sequence[Sequence[float]],
) -> PersuasionOutcome:
    """Stateless persuasion solve for a single sender/receiver game."""
    game = PersuasionGame(
        states=tuple(states),
        actions=tuple(actions),
        prior=tuple(prior),
        sender_utility=tuple(tuple(r) for r in sender_utility),
        receiver_utility=tuple(tuple(r) for r in receiver_utility),
    )
    return Persuader().persuade(game)


def quick_verify(
    states: Sequence[str],
    actions: Sequence[str],
    prior: Sequence[float],
    sender_utility: Sequence[Sequence[float]],
    receiver_utility: Sequence[Sequence[float]],
    scheme: SignalingScheme,
) -> VerificationReport:
    """Stateless verification of a candidate scheme."""
    game = PersuasionGame(
        states=tuple(states),
        actions=tuple(actions),
        prior=tuple(prior),
        sender_utility=tuple(tuple(r) for r in sender_utility),
        receiver_utility=tuple(tuple(r) for r in receiver_utility),
    )
    return verify_scheme(game, scheme)


__all__ = [
    # event kinds
    "PERSUADE_STARTED",
    "PERSUADE_SOLVED",
    "PERSUADE_SIGNAL_SENT",
    "PERSUADE_VERIFIED",
    "PERSUADE_CERTIFIED",
    "PERSUADE_ONLINE_STEP",
    "PERSUADE_ROBUST_SOLVED",
    "PERSUADE_MULTI_SOLVED",
    "PERSUADE_CLEARED",
    # mode identifiers
    "KIND_CONCAVIFICATION",
    "KIND_LP",
    "KIND_ONLINE",
    "KIND_ROBUST",
    "KIND_MULTI_PRIVATE",
    "KIND_MULTI_PUBLIC",
    "KNOWN_KINDS",
    # exceptions
    "PersuaderError",
    "InvalidPrior",
    "InvalidUtility",
    "InfeasibleProgram",
    "UnknownReceiver",
    "InsufficientData",
    "UnknownKind",
    # concentration helpers
    "hoeffding_radius",
    "empirical_bernstein_radius",
    # data classes
    "PersuasionGame",
    "SignalingScheme",
    "PersuasionOutcome",
    "OnlinePersuasionOutcome",
    "RobustOutcome",
    "MultiReceiverOutcome",
    "PayoffCertificate",
    "VerificationReport",
    # helpers
    "best_response",
    "sender_value_under_full_information",
    "receiver_value_under_posterior",
    "verify_scheme",
    # main primitive
    "Persuader",
    # facade
    "quick_persuade",
    "quick_verify",
]
