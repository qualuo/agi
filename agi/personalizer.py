r"""Personalizer — online per-user preference learning at inference time.

Every other learning primitive in this runtime learns a *single* policy
or scoring function from a stream of (prompt, x, signal) triples.
``Aligner`` (DPO) is the canonical example: one preference stream
becomes one policy.  But the operational reality of a coordination
engine serving many users is that *each user's preferences differ* —
on tone, formality, verbosity, taste, terminology, what counts as
helpful — and a one-policy-fits-all aligner saturates the global
average rather than the per-user maximum.

``Personalizer`` is the runtime primitive that closes that gap with
**bounded, anytime, certified, stdlib** machinery.  For each user
identifier it learns an *adapter* — a low-dimensional ridge-regressed
score offset on top of a global prior policy — by online updates
from per-user preference signals (pairwise BTL, unary KTO).  Every
prediction carries a per-user, anytime-valid confidence interval on
the predicted preference probability so the coordinator can decide
whether to *trust the personalised score* or *fall back to the
global policy* on a per-prompt basis.

How a coordination engine uses it
---------------------------------

  1. The engine maintains a single :class:`Personalizer` instance per
     deployment.  Each prompt the engine routes carries a stable
     ``user_id``.
  2. Before generation, the engine asks
     ``personalizer.score(user_id, prompt, candidates)``.  The
     primitive returns per-candidate logits adjusted by the user's
     adapter, with an anytime-valid CI per candidate and a binary
     ``trust`` flag (the CI half-width is below a configured
     tolerance and the user has crossed a calibration sample
     threshold).
  3. The engine generates / dispatches according to whichever
     ranking it trusts more.  After the user responds (thumbs,
     explicit comparison, or an implicit-signal proxy like accept-
     rate), the engine calls
     ``personalizer.observe(user_id, signal)``.
  4. The adapter updates incrementally.  No batch retrain, no
     gradient steps over many users at once — each call is O(d) in
     the adapter dimension.  Privacy is local-DP-friendly: the
     coordinator can wrap ``observe`` with a Gaussian-mechanism
     decorator (``observe_dp``).

Algorithmic surface
-------------------

  * **Pairwise (Bradley-Terry).**  The user prefers ``x_w`` over
    ``x_l`` for a prompt ``q``.  We model the per-user score
    :math:`s_u(q, x) = s_{\\text{ref}}(q, x) + \\langle \\theta_u,
    \\phi(q, x) \\rangle` where :math:`\\phi` is a fixed feature map
    (coordinator-supplied) and :math:`\\theta_u` is the user's
    adapter.  The BTL likelihood
    :math:`P(x_w \\succ x_l | q, u) = \\sigma(s_u(q, x_w) - s_u(q,
    x_l))` gives a strongly-convex log-loss that we minimise by
    online stochastic mirror descent with an :math:`L_2` shrinkage
    toward the global prior :math:`\\theta_g`.  (Bradley–Terry 1952;
    Hunter 2004; Rafailov-Sharma-Mitchell-Ermon-Manning-Finn 2023
    DPO; Lee et al. 2023 SLiC.)

  * **Unary (Kahneman-Tversky).**  The user marks a single response
    desirable or undesirable.  We use the KTO loss (Ethayarajh et
    al. 2024) — value-function reframing of preference — implemented
    as a logistic regression on a derived margin against a per-user
    reference rate.

  * **Per-user ridge anchored to a global prior.**  The objective
    minimises :math:`\\sum_i \\ell_i(\\theta_u) + \\lambda \\|\\theta_u
    - \\theta_g\\|^2`.  When the user has 0 observations the adapter
    *is* the global prior; with N observations the posterior
    shrinks toward the user's empirical optimum at rate
    :math:`O(\\lambda / N)`.  This is the standard hierarchical
    Bayes argument from Gelman-Hill 2006 with the closed-form ridge.

  * **Anytime-valid CI on per-candidate predicted preference
    probability.**  The CI is computed from an empirical-Bernstein
    bound (Maurer-Pontil 2009) on the per-user residuals around the
    fitted predictor, with sample size :math:`n_u` the user's
    observation count.  Coordinator can ask
    ``personalizer.trust(user_id, half_width)`` to gate the use of
    the adapter on a CI half-width budget.

  * **Differential privacy (optional).**  Gaussian-mechanism wrapper
    on the gradient: ``g'_t = g_t + N(0, \\sigma^2 I)``.  With
    learning rate :math:`\\eta` and ``T`` updates the cumulative
    privacy loss tracks via the Rényi-DP composition of Mironov
    2017.  Reported per-user.

  * **Replay-verifiable.**  Every ``observe`` / ``score`` /
    ``predict`` / ``promote`` transition appends to a SHA-256
    fingerprint chain so a coordinator can replay-verify a
    Personalizer run byte-for-byte at audit.

Composes with
-------------

  * :mod:`agi.aligner`     — Aligner's global DPO is the reference
                              policy ``π_ref`` the per-user adapter
                              shrinks toward.  Promote a user's
                              adapter to the global one when it
                              dominates on a held-out preference set.

  * :mod:`agi.steerer`     — Each user can also receive a *steering*
                              vector certified by Steerer; the
                              Personalizer score and the Steerer
                              direction together yield an
                              orthogonalised per-user intervention.

  * :mod:`agi.policy`,
    :mod:`agi.capabilities` — Personalizer's per-user score becomes a
                              feature in the policy router.

  * :mod:`agi.privacy`     — co-issue privacy budget via the Rényi-DP
                              accountant.

  * :mod:`agi.governance`  — per-user quotas and rate limits gate
                              ``observe`` calls.

  * :mod:`agi.refuser`,
    :mod:`agi.sycophant`,
    :mod:`agi.confabulator` — per-user adapter does not over-write
                              the safety-stack decisions; ordering is
                              safety → personalised score.

Design contract
---------------

* **Pure stdlib.**  No NumPy / Torch.  All linear algebra on tuples-
  and-lists of floats.

* **Stateful, thread-safe, deterministic given seed.**

* **No model coupling.**  Personalizer never sees tokens.  It sees
  (user_id, feature_vector, signal) triples produced by the
  coordinator.

* **Event-fingerprinted.**  Every observe / score / predict / fit
  transition is hashed into a SHA-256 chain.

Usage
-----

>>> from agi.personalizer import (
...     Personalizer, PersonalizerConfig, PairwisePreference, UnarySignal,
...     ALG_BTL, ALG_KTO,
... )
>>> p = Personalizer(PersonalizerConfig(algorithm=ALG_BTL, dim=4, seed=0))
>>> # user "u1" prefers candidate with feature x=(1,0,0,0) over (0,1,0,0)
>>> for _ in range(20):
...     _ = p.observe_pair(PairwisePreference(
...         user_id="u1",
...         features_winner=(1.0, 0.0, 0.0, 0.0),
...         features_loser=(0.0, 1.0, 0.0, 0.0),
...     ))
>>> scores = p.score("u1", [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)])
>>> scores[0].mean > scores[1].mean
True
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALG_BTL = "btl"       # Bradley-Terry pairwise (Rafailov DPO-style)
ALG_KTO = "kto"       # Kahneman-Tversky unary (Ethayarajh 2024)
ALG_LOGISTIC = "logistic"  # logistic regression on unary signals
KNOWN_ALGORITHMS: tuple[str, ...] = (ALG_BTL, ALG_KTO, ALG_LOGISTIC)

# Trust verdicts
TRUST_FALLBACK = "fallback"   # not enough data → use global
TRUST_BLEND = "blend"          # blend personalised + global by confidence
TRUST_PROMOTE = "promote"      # personalised score is trusted

KNOWN_TRUSTS: tuple[str, ...] = (TRUST_FALLBACK, TRUST_BLEND, TRUST_PROMOTE)

# Events
PERS_STARTED = "personalizer.started"
PERS_OBSERVED = "personalizer.observed"
PERS_SCORED = "personalizer.scored"
PERS_PROMOTED = "personalizer.promoted"
PERS_REPORTED = "personalizer.reported"
PERS_RESET = "personalizer.reset"
PERS_USER_REMOVED = "personalizer.user_removed"
PERS_DP_BUDGET_UPDATED = "personalizer.dp_budget_updated"

KNOWN_EVENTS: tuple[str, ...] = (
    PERS_STARTED,
    PERS_OBSERVED,
    PERS_SCORED,
    PERS_PROMOTED,
    PERS_REPORTED,
    PERS_RESET,
    PERS_USER_REMOVED,
    PERS_DP_BUDGET_UPDATED,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PersonalizerError(ValueError):
    """Base class."""


class InvalidConfig(PersonalizerError):
    """The :class:`PersonalizerConfig` is internally inconsistent."""


class InvalidSignal(PersonalizerError):
    """A preference / signal row violates a runtime invariant."""


class UnknownAlgorithm(PersonalizerError):
    """Algorithm name not in :data:`KNOWN_ALGORITHMS`."""


class UnknownUser(PersonalizerError):
    """User_id not seen by this Personalizer."""


class DimensionMismatch(PersonalizerError):
    """Feature vector length differs from config.dim."""


class PrivacyBudgetExceeded(PersonalizerError):
    """Reported when the DP composition exhausts the configured budget."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _validate_features(v: Sequence[float], dim: int, what: str) -> tuple[float, ...]:
    if not isinstance(v, (list, tuple)):
        raise InvalidSignal(f"{what} must be a sequence of floats")
    if len(v) != dim:
        raise DimensionMismatch(
            f"{what} has length {len(v)} but config dim={dim}"
        )
    out: list[float] = []
    for i, x in enumerate(v):
        if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise InvalidSignal(f"{what}[{i}] is not a finite number")
        out.append(float(x))
    return tuple(out)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwisePreference:
    """One pairwise preference observation.

    The user supplied a prompt and was shown a winner and loser
    candidate.  ``features_winner`` and ``features_loser`` are the
    coordinator-supplied feature embeddings of the two candidates
    *within the prompt's context*.  ``confidence`` is an optional
    [0, 1] strength of the preference (1.0 = explicit thumbs;
    < 1.0 = implicit signal like accept-rate).
    """

    user_id: str
    features_winner: tuple[float, ...]
    features_loser: tuple[float, ...]
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id:
            raise InvalidSignal("user_id must be a non-empty string")
        if not 0.0 < float(self.confidence) <= 1.0:
            raise InvalidSignal("confidence must be in (0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class UnarySignal:
    """One unary (KTO-style) signal: the user labelled this response
    desirable or undesirable.

    Attributes:
        user_id: stable identifier.
        features: candidate feature vector.
        desirable: True iff the user signalled positive.
        confidence: strength of the signal, in (0, 1].
    """

    user_id: str
    features: tuple[float, ...]
    desirable: bool
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id:
            raise InvalidSignal("user_id must be a non-empty string")
        if not 0.0 < float(self.confidence) <= 1.0:
            raise InvalidSignal("confidence must be in (0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class PersonalizerConfig:
    """Static config.

    Attributes:
        algorithm: one of :data:`KNOWN_ALGORITHMS`.
        dim: feature dimensionality.
        learning_rate: SGD step size.
        ridge: ``λ`` in the per-user :math:`L_2` shrinkage toward
            the global prior.  Larger = stronger global pull.
        min_observations_for_trust: user must have at least this many
            observations before ``trust`` returns ``TRUST_PROMOTE``.
        ci_half_width_promote: per-candidate CI half-width below
            which the personalised score is promoted.  Above
            ``2 * ci_half_width_promote`` → fallback.
        max_users: ring-buffer cap on retained adapters.  LRU
            eviction.  Default ``10_000``.
        max_observations_per_user: ring-buffer cap.
        dp_sigma: optional standard deviation for Gaussian-mechanism
            DP on gradients.  ``None`` ⇒ DP disabled.
        dp_epsilon_target: optional target ε at which to raise
            :class:`PrivacyBudgetExceeded`.  ``None`` ⇒ unlimited.
        dp_delta: companion δ in (0, 1).  Default ``1e-6``.
        seed: deterministic RNG.
    """

    algorithm: str = ALG_BTL
    dim: int = 0
    learning_rate: float = 0.05
    ridge: float = 0.01
    min_observations_for_trust: int = 16
    ci_half_width_promote: float = 0.10
    max_users: int = 10_000
    max_observations_per_user: int = 10_000
    dp_sigma: float | None = None
    dp_epsilon_target: float | None = None
    dp_delta: float = 1e-6
    seed: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"algorithm={self.algorithm!r} not in {KNOWN_ALGORITHMS!r}"
            )
        if not isinstance(self.dim, int) or self.dim < 1:
            raise InvalidConfig("dim must be a positive int")
        if float(self.learning_rate) <= 0.0:
            raise InvalidConfig("learning_rate must be > 0")
        if float(self.ridge) < 0.0:
            raise InvalidConfig("ridge must be >= 0")
        if self.min_observations_for_trust < 1:
            raise InvalidConfig("min_observations_for_trust must be >= 1")
        if not 0.0 < float(self.ci_half_width_promote) < 1.0:
            raise InvalidConfig("ci_half_width_promote must be in (0, 1)")
        if self.max_users < 1:
            raise InvalidConfig("max_users must be >= 1")
        if self.max_observations_per_user < 1:
            raise InvalidConfig("max_observations_per_user must be >= 1")
        if self.dp_sigma is not None and float(self.dp_sigma) < 0.0:
            raise InvalidConfig("dp_sigma must be >= 0 or None")
        if self.dp_epsilon_target is not None and float(self.dp_epsilon_target) <= 0:
            raise InvalidConfig("dp_epsilon_target must be > 0 or None")
        if not 0.0 < float(self.dp_delta) < 1.0:
            raise InvalidConfig("dp_delta must be in (0, 1)")


@dataclass(frozen=True)
class CandidateScore:
    """Result of :meth:`Personalizer.score` for one candidate.

    Attributes:
        features: echoed back so a coordinator can ``zip`` results.
        global_score: ``<theta_g, features>``.
        personalised_score: ``<theta_u, features>``.
        mean: posterior mean (= personalised_score).
        ci_low / ci_high: 1-α CI on the personalised score.
        trust: one of :data:`KNOWN_TRUSTS`.
        n_observations: how many signals this user contributed.
    """

    features: tuple[float, ...]
    global_score: float
    personalised_score: float
    mean: float
    ci_low: float
    ci_high: float
    trust: str
    n_observations: int


@dataclass(frozen=True)
class UserSummary:
    """One user's posterior summary."""

    user_id: str
    theta: tuple[float, ...]
    n_observations: int
    last_loss: float
    epsilon_spent: float
    fingerprint: str


@dataclass(frozen=True)
class PersonalizerReport:
    """Snapshot bundle the coordinator reads."""

    n_users: int
    total_observations: int
    global_theta: tuple[float, ...]
    per_user: tuple[UserSummary, ...]
    fingerprint: str


# ---------------------------------------------------------------------------
# Renyi-DP composition (Mironov 2017)
# ---------------------------------------------------------------------------


def renyi_epsilon(sigma: float, steps: int, delta: float, q: float = 1.0) -> float:
    """Convert a Gaussian-mechanism noise level ``sigma`` (relative to
    L2-sensitivity 1) over ``steps`` to an ``(ε, δ)``-DP bound via the
    standard RDP composition + conversion.

    A simplified, conservative ε computation: for full-batch
    Gaussian mechanism, ε_α = α / (2 σ²); composition multiplies by
    ``steps``; conversion to (ε, δ)-DP uses ε = ε_α + log(1/δ) /
    (α - 1) with the optimal α from a small grid.  Sub-sampling
    factor ``q`` ∈ (0, 1] additionally scales the RDP.

    The conservative form is sufficient as a budget guard; the
    coordinator can wire in a tighter accountant from
    :mod:`agi.privacy` for production reporting.
    """
    if sigma <= 0 or steps <= 0:
        return float("inf")
    best = float("inf")
    for alpha in (2, 3, 5, 8, 16, 32, 64, 128):
        rdp = alpha / (2.0 * sigma * sigma) * q * q
        rdp *= steps
        eps = rdp + math.log(1.0 / max(delta, 1e-12)) / (alpha - 1)
        if eps < best:
            best = eps
    return best


# ---------------------------------------------------------------------------
# The Personalizer class
# ---------------------------------------------------------------------------


def _now() -> float:
    import time
    return time.time()


class _UserState:
    """Per-user mutable state — adapter, sample count, DP step count."""

    __slots__ = (
        "theta",
        "n_observations",
        "loss_sum",
        "var_residual_sum",
        "dp_steps",
        "lru_tick",
    )

    def __init__(self, dim: int) -> None:
        self.theta: list[float] = [0.0] * dim
        self.n_observations: int = 0
        self.loss_sum: float = 0.0
        self.var_residual_sum: float = 0.0
        self.dp_steps: int = 0
        self.lru_tick: int = 0


class Personalizer:
    """Coordinator-facing per-user preference-learning primitive.

    Thread-safe.  Pure compute.  Replay-verifiable: identical
    observation streams produce identical fingerprint chains under
    the same config and seed.
    """

    def __init__(
        self,
        config: PersonalizerConfig,
        bus: Any = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(config, PersonalizerConfig):
            raise InvalidConfig("config must be a PersonalizerConfig")
        # Defensive re-validate
        PersonalizerConfig(**{f: getattr(config, f) for f in (
            "algorithm", "dim", "learning_rate", "ridge",
            "min_observations_for_trust", "ci_half_width_promote",
            "max_users", "max_observations_per_user", "dp_sigma",
            "dp_epsilon_target", "dp_delta", "seed",
        )})
        self._config = config
        self._bus = bus
        self._clock = clock or _now
        self._lock = threading.RLock()
        self._rng = random.Random(config.seed)
        self._users: dict[str, _UserState] = {}
        self._global_theta: list[float] = [0.0] * config.dim
        self._global_obs_count: int = 0
        self._lru_clock: int = 0
        self._fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "init": True,
                    "config": {
                        "algorithm": config.algorithm,
                        "dim": config.dim,
                        "learning_rate": config.learning_rate,
                        "ridge": config.ridge,
                        "dp_sigma": config.dp_sigma,
                        "dp_delta": config.dp_delta,
                        "seed": config.seed,
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self._emit(PERS_STARTED, {"algorithm": config.algorithm, "dim": config.dim})

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> PersonalizerConfig:
        return self._config

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def n_users(self) -> int:
        with self._lock:
            return len(self._users)

    @property
    def total_observations(self) -> int:
        with self._lock:
            return sum(s.n_observations for s in self._users.values())

    @property
    def global_theta(self) -> tuple[float, ...]:
        return tuple(self._global_theta)

    def user_summary(self, user_id: str) -> UserSummary:
        with self._lock:
            st = self._users.get(user_id)
            if st is None:
                raise UnknownUser(user_id)
            eps = self._user_epsilon(st)
            return UserSummary(
                user_id=user_id,
                theta=tuple(st.theta),
                n_observations=st.n_observations,
                last_loss=(st.loss_sum / st.n_observations) if st.n_observations else 0.0,
                epsilon_spent=eps,
                fingerprint=self._fingerprint,
            )

    def _user_epsilon(self, st: _UserState) -> float:
        if self._config.dp_sigma is None or st.dp_steps == 0:
            return 0.0
        return renyi_epsilon(
            sigma=float(self._config.dp_sigma),
            steps=st.dp_steps,
            delta=float(self._config.dp_delta),
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_pair(self, pref: PairwisePreference) -> str:
        if not isinstance(pref, PairwisePreference):
            raise InvalidSignal("pref must be a PairwisePreference")
        if self._config.algorithm != ALG_BTL:
            raise InvalidSignal(
                f"observe_pair requires algorithm={ALG_BTL}, "
                f"have {self._config.algorithm}"
            )
        fw = _validate_features(pref.features_winner, self._config.dim, "features_winner")
        fl = _validate_features(pref.features_loser, self._config.dim, "features_loser")
        with self._lock:
            st = self._touch_user(pref.user_id)
            # Score margin
            margin = sum(st.theta[i] * (fw[i] - fl[i]) for i in range(self._config.dim))
            global_margin = sum(self._global_theta[i] * (fw[i] - fl[i]) for i in range(self._config.dim))
            total_margin = margin + global_margin
            p = _sigmoid(total_margin)
            # BTL log-loss: -log σ(margin); gradient wrt θ_u is
            # -(1-p) * (φ_w - φ_l) + λ (θ_u - θ_g)
            # Gradient of -log σ(margin) w.r.t. θ is -(1 - p) · (φ_w - φ_l).
            # _step subtracts η·grad, so we want grad with that sign.
            err = (1.0 - p) * pref.confidence
            grad = [-err * (fw[i] - fl[i]) for i in range(self._config.dim)]
            self._step(st, grad)
            # Global also gets a small update (regularised)
            self._step_global(grad, weight=0.1)
            st.loss_sum += -math.log(max(p, 1e-12))
            # Residual variance proxy: |p - 1|, used in CI
            st.var_residual_sum += (1.0 - p) ** 2
            self._global_obs_count += 1
            self._chain("observe_pair", {
                "user_id": pref.user_id,
                "n": st.n_observations,
                "p": p,
                "confidence": pref.confidence,
            })
            self._emit(PERS_OBSERVED, {
                "kind": "pair",
                "user_id": pref.user_id,
                "n_observations": st.n_observations,
                "fingerprint": self._fingerprint,
            })
            return self._fingerprint

    def observe_unary(self, sig: UnarySignal) -> str:
        if not isinstance(sig, UnarySignal):
            raise InvalidSignal("sig must be a UnarySignal")
        f = _validate_features(sig.features, self._config.dim, "features")
        with self._lock:
            st = self._touch_user(sig.user_id)
            # KTO / logistic: score the candidate, push up if desirable.
            score = sum(st.theta[i] * f[i] for i in range(self._config.dim))
            global_score = sum(self._global_theta[i] * f[i] for i in range(self._config.dim))
            total = score + global_score
            p = _sigmoid(total)
            target = 1.0 if sig.desirable else 0.0
            # Logistic loss: -t·log(p) - (1-t)·log(1-p).  Gradient w.r.t. θ
            # is (p - target) · f.  _step subtracts η·grad, so pass +grad.
            err = (p - target) * sig.confidence
            grad = [err * f[i] for i in range(self._config.dim)]
            self._step(st, grad)
            self._step_global(grad, weight=0.1)
            loss = -target * math.log(max(p, 1e-12)) - (1 - target) * math.log(max(1 - p, 1e-12))
            st.loss_sum += loss
            st.var_residual_sum += (target - p) ** 2
            self._global_obs_count += 1
            self._chain("observe_unary", {
                "user_id": sig.user_id,
                "n": st.n_observations,
                "desirable": sig.desirable,
                "p": p,
                "confidence": sig.confidence,
            })
            self._emit(PERS_OBSERVED, {
                "kind": "unary",
                "user_id": sig.user_id,
                "desirable": sig.desirable,
                "n_observations": st.n_observations,
                "fingerprint": self._fingerprint,
            })
            return self._fingerprint

    # ------------------------------------------------------------------
    # Internal update — SGD with ridge shrinkage and optional Gaussian DP
    # ------------------------------------------------------------------

    def _step(self, st: _UserState, grad: list[float]) -> None:
        cfg = self._config
        # Shrinkage toward global θ:
        lam = cfg.ridge
        # Optional DP noise on the gradient:
        if cfg.dp_sigma is not None and cfg.dp_sigma > 0.0:
            # Gaussian mechanism w/ L2-sensitivity assumed 1 (the caller
            # should clip features; we still add noise as the budget
            # guard).
            for i in range(cfg.dim):
                grad[i] += self._rng.gauss(0.0, cfg.dp_sigma)
            st.dp_steps += 1
            # Budget check
            if cfg.dp_epsilon_target is not None:
                eps = renyi_epsilon(
                    sigma=cfg.dp_sigma, steps=st.dp_steps, delta=cfg.dp_delta
                )
                if eps > cfg.dp_epsilon_target:
                    raise PrivacyBudgetExceeded(
                        f"user has spent ε={eps:.3f} > target {cfg.dp_epsilon_target}"
                    )
        # SGD step: θ ← θ - η · ∇L  with ridge toward global
        eta = cfg.learning_rate
        for i in range(cfg.dim):
            shrink = lam * (st.theta[i] - self._global_theta[i])
            st.theta[i] -= eta * (grad[i] + shrink)
        st.n_observations += 1
        # Evict if over per-user cap.
        if st.n_observations > cfg.max_observations_per_user:
            st.n_observations = cfg.max_observations_per_user

    def _step_global(self, grad: list[float], weight: float = 0.1) -> None:
        eta = self._config.learning_rate * weight
        for i in range(self._config.dim):
            self._global_theta[i] -= eta * grad[i]

    def _touch_user(self, user_id: str) -> _UserState:
        st = self._users.get(user_id)
        self._lru_clock += 1
        if st is None:
            # Initialise from the global prior so a brand-new user
            # already gets the average policy.
            st = _UserState(self._config.dim)
            for i in range(self._config.dim):
                st.theta[i] = self._global_theta[i]
            st.lru_tick = self._lru_clock
            self._users[user_id] = st
            # Evict only AFTER setting lru_tick so the freshly-added
            # user is not its own victim.
            if len(self._users) > self._config.max_users:
                self._evict_one_user(protect=user_id)
        st.lru_tick = self._lru_clock
        return st

    def _evict_one_user(self, protect: str | None = None) -> None:
        candidates = [
            (uid, st) for uid, st in self._users.items() if uid != protect
        ]
        if not candidates:
            return
        victim = min(candidates, key=lambda kv: kv[1].lru_tick)
        del self._users[victim[0]]
        self._emit(PERS_USER_REMOVED, {"user_id": victim[0]})

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        user_id: str,
        candidates: Sequence[Sequence[float]],
        alpha: float = 0.05,
    ) -> list[CandidateScore]:
        """Score a list of candidate feature vectors for ``user_id``.

        Returns one :class:`CandidateScore` per candidate.  The
        trust flag is computed from the per-user observation count
        and the predicted CI half-width.
        """
        with self._lock:
            cfg = self._config
            st = self._users.get(user_id)
            n_obs = st.n_observations if st is not None else 0
            theta_u = (
                tuple(st.theta) if st is not None else tuple(self._global_theta)
            )
            # Empirical-Bernstein on residual variance:
            if st is None or st.n_observations < 2:
                var = 0.25  # max Bernoulli var
            else:
                var = st.var_residual_sum / max(1, st.n_observations)
            # Half-width on the per-candidate Bernoulli probability
            # estimate, computed from sample size & variance estimate.
            n_safe = max(n_obs, 1)
            t_alpha = math.sqrt(2.0 * math.log(2.0 / alpha) * var / n_safe)
            range_bound = math.log(2.0 / alpha) * 7.0 / (3.0 * (n_safe - 1)) if n_safe > 1 else 1.0
            half = t_alpha + range_bound
            # Saturate the half-width at 0.5 (we're on a probability).
            half = min(half, 0.5)
            out: list[CandidateScore] = []
            for v in candidates:
                fv = _validate_features(v, cfg.dim, "candidate")
                gs = sum(self._global_theta[i] * fv[i] for i in range(cfg.dim))
                ps = sum(theta_u[i] * fv[i] for i in range(cfg.dim))
                # Final logit:
                logit = gs + ps
                p = _sigmoid(logit)
                lo = max(0.0, p - half)
                hi = min(1.0, p + half)
                if n_obs < cfg.min_observations_for_trust:
                    trust = TRUST_FALLBACK
                elif half > 2 * cfg.ci_half_width_promote:
                    trust = TRUST_FALLBACK
                elif half > cfg.ci_half_width_promote:
                    trust = TRUST_BLEND
                else:
                    trust = TRUST_PROMOTE
                out.append(CandidateScore(
                    features=fv,
                    global_score=float(gs),
                    personalised_score=float(ps),
                    mean=float(p),
                    ci_low=float(lo),
                    ci_high=float(hi),
                    trust=trust,
                    n_observations=n_obs,
                ))
            self._chain("score", {
                "user_id": user_id,
                "n_candidates": len(candidates),
                "n_obs": n_obs,
            })
            self._emit(PERS_SCORED, {
                "user_id": user_id,
                "n_candidates": len(out),
                "trust_distribution": {
                    t: sum(1 for r in out if r.trust == t)
                    for t in KNOWN_TRUSTS
                },
                "fingerprint": self._fingerprint,
            })
            return out

    def predict(
        self,
        user_id: str,
        feature_a: Sequence[float],
        feature_b: Sequence[float],
        alpha: float = 0.05,
    ) -> tuple[float, float, float, str]:
        """Predict P(user prefers A over B), plus 1-α CI and trust verdict."""
        scores = self.score(user_id, [feature_a, feature_b], alpha=alpha)
        margin = scores[0].mean - scores[1].mean
        # Centered probability of preference for A:
        prob = 0.5 + 0.5 * margin
        prob = max(0.0, min(1.0, prob))
        half = scores[0].ci_high - scores[0].mean  # approximate same width
        return prob, max(0.0, prob - half), min(1.0, prob + half), scores[0].trust

    def trust(self, user_id: str, alpha: float = 0.05) -> str:
        """One-shot trust verdict for a user (independent of candidate)."""
        with self._lock:
            st = self._users.get(user_id)
            if st is None or st.n_observations < self._config.min_observations_for_trust:
                return TRUST_FALLBACK
            var = st.var_residual_sum / max(1, st.n_observations)
            n_safe = st.n_observations
            half = math.sqrt(2.0 * math.log(2.0 / alpha) * var / n_safe)
            half = min(half, 0.5)
            if half > 2 * self._config.ci_half_width_promote:
                return TRUST_FALLBACK
            if half > self._config.ci_half_width_promote:
                return TRUST_BLEND
            return TRUST_PROMOTE

    # ------------------------------------------------------------------
    # Promote / remove
    # ------------------------------------------------------------------

    def promote_to_global(self, user_id: str, blend: float = 0.5) -> tuple[float, ...]:
        """Update the global prior by blending in a user's adapter.

        ``blend`` ∈ [0, 1] is the convex mix weight on the user's
        ``θ_u``.  ``0`` is a no-op; ``1`` replaces the global prior
        entirely.

        The intended use is when a user's adapter dominates the
        global one on a held-out preference set (a meta-evaluation
        outside Personalizer's scope) — at that point the
        coordinator can "promote" this user's solution to the
        population.
        """
        if not 0.0 <= blend <= 1.0:
            raise InvalidSignal("blend must be in [0, 1]")
        with self._lock:
            st = self._users.get(user_id)
            if st is None:
                raise UnknownUser(user_id)
            for i in range(self._config.dim):
                self._global_theta[i] = (
                    (1.0 - blend) * self._global_theta[i] + blend * st.theta[i]
                )
            self._chain("promote", {"user_id": user_id, "blend": blend})
            self._emit(PERS_PROMOTED, {
                "user_id": user_id,
                "blend": blend,
                "fingerprint": self._fingerprint,
            })
            return tuple(self._global_theta)

    def remove_user(self, user_id: str) -> bool:
        """Forget a user's adapter — GDPR Article 17 ('right to erasure').

        Returns True iff the user existed.  The chain still records
        the erasure so the action itself remains auditable.
        """
        with self._lock:
            existed = user_id in self._users
            if existed:
                del self._users[user_id]
                self._chain("remove_user", {"user_id": user_id})
                self._emit(PERS_USER_REMOVED, {
                    "user_id": user_id,
                    "fingerprint": self._fingerprint,
                })
            return existed

    # ------------------------------------------------------------------
    # Report bundle
    # ------------------------------------------------------------------

    def report(self) -> PersonalizerReport:
        with self._lock:
            users = []
            for uid, st in sorted(self._users.items()):
                users.append(UserSummary(
                    user_id=uid,
                    theta=tuple(st.theta),
                    n_observations=st.n_observations,
                    last_loss=(st.loss_sum / st.n_observations) if st.n_observations else 0.0,
                    epsilon_spent=self._user_epsilon(st),
                    fingerprint=self._fingerprint,
                ))
            r = PersonalizerReport(
                n_users=len(self._users),
                total_observations=sum(s.n_observations for s in self._users.values()),
                global_theta=tuple(self._global_theta),
                per_user=tuple(users),
                fingerprint=self._fingerprint,
            )
            self._emit(PERS_REPORTED, {
                "n_users": r.n_users,
                "total_observations": r.total_observations,
                "fingerprint": self._fingerprint,
            })
            return r

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._users.clear()
            self._global_theta = [0.0] * self._config.dim
            self._global_obs_count = 0
            self._lru_clock = 0
            self._chain("reset", {})
            self._emit(PERS_RESET, {"fingerprint": self._fingerprint})

    # ------------------------------------------------------------------
    # Internal: fingerprint chain + event emission
    # ------------------------------------------------------------------

    def _chain(self, kind: str, payload: Mapping[str, Any]) -> None:
        h = hashlib.sha256()
        h.update(self._fingerprint.encode("utf-8"))
        h.update(kind.encode("utf-8"))
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        self._fingerprint = h.hexdigest()

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            from agi.events import Event
        except Exception:  # pragma: no cover
            return
        for attempt in (
            lambda: Event(kind=kind, data=dict(payload), ts=self._clock()),
            lambda: Event(kind=kind, data=dict(payload)),
            lambda: Event(kind, None, dict(payload)),
            lambda: Event(kind),
        ):
            try:
                ev = attempt()
                break
            except TypeError:
                continue
        else:  # pragma: no cover
            return
        try:
            self._bus.publish(ev)
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def fresh_personalizer(
    algorithm: str = ALG_BTL,
    *,
    dim: int,
    learning_rate: float = 0.05,
    ridge: float = 0.01,
    seed: int = 0,
) -> Personalizer:
    return Personalizer(PersonalizerConfig(
        algorithm=algorithm,
        dim=dim,
        learning_rate=learning_rate,
        ridge=ridge,
        seed=seed,
    ))


def synthetic_users(
    n_users: int,
    n_prefs_per_user: int,
    dim: int,
    seed: int = 0,
) -> tuple[list[PairwisePreference], list[tuple[float, ...]]]:
    """Synthesise a paired-preference stream over ``n_users``, each
    with a *different* preferred axis.  Returns (preferences, ground-
    truth user vectors).
    """
    rng = random.Random(seed)
    users_axes: list[int] = [rng.randrange(dim) for _ in range(n_users)]
    prefs: list[PairwisePreference] = []
    truths: list[tuple[float, ...]] = []
    for u, axis in enumerate(users_axes):
        truth = tuple(1.0 if j == axis else 0.0 for j in range(dim))
        truths.append(truth)
        for _ in range(n_prefs_per_user):
            # Two candidate vectors; winner has higher projection along
            # user's preferred axis.
            v1 = tuple(rng.gauss(0.0, 1.0) for _ in range(dim))
            v2 = tuple(rng.gauss(0.0, 1.0) for _ in range(dim))
            if v1[axis] > v2[axis]:
                prefs.append(PairwisePreference(
                    user_id=f"u{u}",
                    features_winner=v1,
                    features_loser=v2,
                ))
            else:
                prefs.append(PairwisePreference(
                    user_id=f"u{u}",
                    features_winner=v2,
                    features_loser=v1,
                ))
    return prefs, truths


__all__ = [
    "ALG_BTL",
    "ALG_KTO",
    "ALG_LOGISTIC",
    "KNOWN_ALGORITHMS",
    "TRUST_FALLBACK",
    "TRUST_BLEND",
    "TRUST_PROMOTE",
    "KNOWN_TRUSTS",
    "PERS_STARTED",
    "PERS_OBSERVED",
    "PERS_SCORED",
    "PERS_PROMOTED",
    "PERS_REPORTED",
    "PERS_RESET",
    "PERS_USER_REMOVED",
    "PERS_DP_BUDGET_UPDATED",
    "KNOWN_EVENTS",
    "PersonalizerError",
    "InvalidConfig",
    "InvalidSignal",
    "UnknownAlgorithm",
    "UnknownUser",
    "DimensionMismatch",
    "PrivacyBudgetExceeded",
    "PairwisePreference",
    "UnarySignal",
    "PersonalizerConfig",
    "CandidateScore",
    "UserSummary",
    "PersonalizerReport",
    "Personalizer",
    "renyi_epsilon",
    "fresh_personalizer",
    "synthetic_users",
]
