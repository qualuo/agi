r"""Continualist — continual / lifelong learning as a runtime primitive.

Every other learning primitive in this runtime assumes the data
distribution is i.i.d. for the duration of a fit.  Real agents see a
*non-stationary* stream of tasks: a coding session yields to a research
session yields to a finance session, and the agent that was good at the
first one is now expected to be good at all three.  A naive ``Aligner``
or ``Distiller`` updated on the new task overwrites the parameters the
old task depended on — the catastrophic-forgetting phenomenon
McCloskey-Cohen 1989 named almost forty years ago and that the
state-of-the-art only now treats as first-class.

``Continualist`` is the runtime's *bounded, anytime, certified, stdlib*
continual-learning kernel.  It tracks a parameter vector ``θ ∈ R^d``,
ingests per-task gradient/loss/accuracy signals from any downstream
learner, and emits the regularisation term, replay batch, and
plasticity-stability certificate the coordination engine needs to decide
when to keep adapting and when to lock-in a skill.

The pitch reduced to a runtime call::

    cl = Continualist(ContinualistConfig(method="online_ewc", dim=64))
    cl.register_task("coding")
    for step in range(N):
        grad = compute_grad(...)
        cl.update("coding", grad=grad, theta=theta, loss=loss, accuracy=acc)
        # Continualist contributes a regulariser the trainer can add:
        reg, reg_grad = cl.regulariser(theta)
    cl.commit_task("coding", final_theta=theta)

    cl.register_task("research")
    ...                                      # train as above, never forget
    rep = cl.report()                        # BWT, FWT, AvgAcc, forgetting
    cert = cl.certify()                      # PAC-Bayes continual-risk bound


What this primitive ships
-------------------------

  * **Online Elastic Weight Consolidation** (Schwarz et al. 2018
    *Progress & Compress*, building on Kirkpatrick et al. 2017
    *Overcoming catastrophic forgetting in neural networks*).
    Maintains a running Fisher-information estimate ``F`` and an
    anchor ``θ*``; the regulariser is ``(λ/2) Σ_i F_i (θ_i − θ*_i)²``.
    Online: after every committed task, ``F ← γ·F + F_new`` and
    ``θ* ← θ_final``.  γ ∈ [0,1] is the decay; γ = 0 is task-EWC,
    γ = 1 is fully accumulated.

  * **Synaptic Intelligence** (Zenke, Poole, Ganguli 2017
    *Continual learning through synaptic intelligence*).  Tracks the
    path integral ``ω_i = − Σ_t g_i^t · Δθ_i^t`` of per-parameter
    contribution to loss reduction, then folds it into an importance
    ``Ω_i = ω_i / ((Δθ_i^task)² + ξ)``.  Regulariser:
    ``(c/2) Σ_i Ω_i (θ_i − θ*_i)²``.

  * **Memory Aware Synapses** (Aljundi, Babiloni, Elhoseiny, Rohrbach,
    Tuytelaars 2018 *Memory aware synapses: Learning what (not) to
    forget*).  Importance is the L2 norm of the gradient of the
    *output* magnitude with respect to each parameter, accumulated
    over an unlabelled probe stream — unsupervised, so a coordinator
    can refresh importance from new domains without re-labelling.

  * **Averaged Gradient Episodic Memory (A-GEM)** (Chaudhry, Ranzato,
    Rohrbach, Elhoseiny 2018 *Efficient lifelong learning with A-GEM*).
    Per-step projection: if the new gradient ``g`` and the averaged
    replay gradient ``g_ref`` have negative inner product, project
    ``g ← g − ((g·g_ref)/(g_ref·g_ref)) g_ref``.  Cheap, no per-task
    constraint set, beats GEM on most benchmarks.

  * **Reservoir-sample replay buffer** (Vitter 1985, Algorithm R).
    Uniform random sample of the lifetime stream in O(1) per item,
    with deterministic seeding for replay.  Optional class-balanced
    reservoir (Chrysakis-Moens 2020) when item class labels are
    provided.

  * **Learning-without-Forgetting distillation residual** (Li, Hoiem
    2017 *Learning without forgetting*).  Cross-entropy between the
    *current* policy logits and a *frozen old-task* policy logits on
    new-task inputs.  No old data needed, just the frozen anchor head.

  * **Bayesian Online Change-Point Detection** (Adams, MacKay 2007
    *Bayesian online changepoint detection*).  When the coordinator
    doesn't tell us task boundaries, the run-length posterior over a
    Normal-Gamma stream gives an anytime-valid change probability.
    The agent self-segments its experience into tasks.

  * **Continual evaluation metrics** (Lopez-Paz, Ranzato 2017
    *Gradient episodic memory for continual learning*).  Backward
    Transfer ``BWT = (1/(T−1)) Σ_{i<T} (R_{T,i} − R_{i,i})``, Forward
    Transfer ``FWT = (1/(T−1)) Σ_{i>1} (R_{i−1,i} − R̄_i)``, Average
    Accuracy ``A = (1/T) Σ_i R_{T,i}``, Forgetting ``F = (1/(T−1))
    Σ_{i<T} max_{t≤T} (R_{t,i} − R_{T,i})``.

  * **PAC-Bayes continual-risk certificate** (Pentina, Lampert 2014
    *A PAC-Bayesian bound for lifelong learning*; McAllester 1999
    base bound).  Given ``T`` tasks, each with empirical risk
    ``R̂_t`` and ``n_t`` samples, the expected risk on a new task
    drawn from the same environment satisfies, w.p. ≥ 1 − δ,

        ``E[R] ≤ (1/T) Σ_t R̂_t + sqrt((KL(Q‖P) + ln(2T/δ)) / (2 n))``

    where ``KL(Q‖P)`` is computed from the EWC posterior covariance.
    The runtime exposes the bound and its tightening over time.

  * **Plasticity-stability certificate.**  Watson-Crick style
    monotone-trade-off receipt: if every committed task's accuracy
    ``A_t,t`` stays above ``A_min`` *and* every previous-task
    accuracy ``A_T,t`` stays above ``A_max − ε``, the certificate
    fires and the coordinator can advertise "no skill loss".

  * **Tamper-evident SHA-256 fingerprint chain** (genesis
    ``agi.continualist.v1`` + optional HMAC) over every register /
    update / commit / boundary / report / certify event.
    ``AttestationLedger`` replays every continual update byte-for-byte
    from the observation stream.

  * **Snapshot / restore.**  ``snapshot()`` and ``restore()``
    round-trip a byte-identical chain head + parameter anchors +
    importance vector + replay buffer + change-point posterior, so a
    coordination engine can hibernate a lifelong learner, ship it to
    another host, and resume on the next task without loss.

  * **Thread-safe re-entrant lock** + transport-agnostic + pure stdlib.

Composes with
-------------

  * ``Drift`` — feed change-point detections from ``Drift`` into
    ``Continualist.task_boundary`` to auto-segment the stream when
    the coordinator does not know task labels.
  * ``Curator`` — generates the next task; ``Continualist`` tracks
    per-task accuracy over time and feeds forgetting back into
    curriculum selection (high-forgetting tasks should be revisited).
  * ``Distiller`` — LwF distillation uses ``Distiller`` to fit the
    frozen old-task surrogate over which the residual is computed.
  * ``Conformal`` — wraps the PAC-Bayes continual-risk bound with a
    second-layer conformal certificate over held-out tasks.
  * ``Pareto`` — exposes the *plasticity (new accuracy) vs stability
    (old accuracy)* trade-off as a 2-objective frontier the coordinator
    can scalarise.
  * ``Aligner`` — every Aligner step can call
    ``Continualist.project_gradient`` (A-GEM) to refuse a preference
    update that would damage prior preferences.
  * ``Speculator`` — when speculative decoding is on, the per-token
    accept/reject signal can be ingested as the continual-learning
    learning signal.
  * ``Coordinator`` — the headline runtime API: every long-running
    coordinator instantiates one Continualist per skill axis and
    reads its certificate to decide *adapt vs lock-in*.

Notation
--------

  * ``θ`` — parameter vector, dimension ``d``.
  * ``θ_t*`` — anchor (committed parameters after task t).
  * ``F_i`` — Fisher importance for parameter i (Online EWC).
  * ``Ω_i`` — importance score (SI / MAS).
  * ``R_{t,s}`` — accuracy on task ``s`` after training task ``t``.
  * ``g`` — current gradient (current task).
  * ``g_ref`` — averaged replay gradient (old tasks).

All inputs are validated; all updates are O(d) or O(d + B) for replay
batch ``B``.  No randomness uses ``random`` without an explicit seed; no
``time.time()`` calls leak into the chain.

References
----------

  * McCloskey, Cohen 1989. *Catastrophic interference in connectionist
    networks: The sequential learning problem.* Psychology of
    Learning and Motivation 24:109-165.
  * Kirkpatrick et al. 2017. *Overcoming catastrophic forgetting in
    neural networks.* PNAS 114(13):3521-3526.
  * Schwarz et al. 2018. *Progress & Compress: A scalable framework
    for continual learning.* ICML.
  * Zenke, Poole, Ganguli 2017. *Continual learning through synaptic
    intelligence.* ICML.
  * Aljundi et al. 2018. *Memory aware synapses: Learning what (not)
    to forget.* ECCV.
  * Chaudhry et al. 2018. *Efficient lifelong learning with A-GEM.*
    ICLR 2019.
  * Vitter 1985. *Random sampling with a reservoir.* ACM TOMS 11(1).
  * Li, Hoiem 2017. *Learning without forgetting.* TPAMI 40(12).
  * Adams, MacKay 2007. *Bayesian online changepoint detection.*
    arXiv:0710.3742.
  * Lopez-Paz, Ranzato 2017. *Gradient episodic memory for continual
    learning.* NeurIPS.
  * Pentina, Lampert 2014. *A PAC-Bayesian bound for lifelong
    learning.* ICML.
  * McAllester 1999. *Some PAC-Bayesian theorems.* COLT.
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
    "CONTINUALIST_STARTED",
    "CONTINUALIST_REGISTERED",
    "CONTINUALIST_UPDATED",
    "CONTINUALIST_PROJECTED",
    "CONTINUALIST_COMMITTED",
    "CONTINUALIST_BOUNDARY",
    "CONTINUALIST_REPLAY_PUSHED",
    "CONTINUALIST_REPORTED",
    "CONTINUALIST_CERTIFIED",
    "CONTINUALIST_RESET",
    # Method codes
    "METHOD_ONLINE_EWC",
    "METHOD_SI",
    "METHOD_MAS",
    "METHOD_AGEM",
    "METHOD_REPLAY",
    "METHOD_LWF",
    "METHOD_NONE",
    "KNOWN_METHODS",
    # Replay strategies
    "REPLAY_RESERVOIR",
    "REPLAY_RING",
    "REPLAY_BALANCED",
    "KNOWN_REPLAY_STRATEGIES",
    # Exceptions
    "ContinualistError",
    "InvalidConfig",
    "InvalidGradient",
    "InvalidTheta",
    "InvalidTask",
    "InsufficientData",
    "UnknownMethod",
    "UnknownReplayStrategy",
    "UnknownTask",
    # Dataclasses
    "ContinualistConfig",
    "TaskRecord",
    "UpdateOutput",
    "ProjectionOutput",
    "ReplayItem",
    "Boundary",
    "ContinualistReport",
    "ContinualistCertificate",
    # Helpers
    "continualist_ledger_root",
    "pac_bayes_continual_bound",
    "backward_transfer",
    "forward_transfer",
    "average_accuracy",
    "forgetting_metric",
    # Main class
    "Continualist",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

CONTINUALIST_STARTED = "continualist.started"
CONTINUALIST_REGISTERED = "continualist.registered"
CONTINUALIST_UPDATED = "continualist.updated"
CONTINUALIST_PROJECTED = "continualist.projected"
CONTINUALIST_COMMITTED = "continualist.committed"
CONTINUALIST_BOUNDARY = "continualist.boundary"
CONTINUALIST_REPLAY_PUSHED = "continualist.replay_pushed"
CONTINUALIST_REPORTED = "continualist.reported"
CONTINUALIST_CERTIFIED = "continualist.certified"
CONTINUALIST_RESET = "continualist.reset"


# ---------------------------------------------------------------------------
# Method / replay strategy enums
# ---------------------------------------------------------------------------

METHOD_ONLINE_EWC = "online_ewc"
METHOD_SI = "si"
METHOD_MAS = "mas"
METHOD_AGEM = "agem"
METHOD_REPLAY = "replay"
METHOD_LWF = "lwf"
METHOD_NONE = "none"

KNOWN_METHODS = (
    METHOD_ONLINE_EWC,
    METHOD_SI,
    METHOD_MAS,
    METHOD_AGEM,
    METHOD_REPLAY,
    METHOD_LWF,
    METHOD_NONE,
)

REPLAY_RESERVOIR = "reservoir"
REPLAY_RING = "ring"
REPLAY_BALANCED = "balanced"

KNOWN_REPLAY_STRATEGIES = (REPLAY_RESERVOIR, REPLAY_RING, REPLAY_BALANCED)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContinualistError(Exception):
    """Base error for Continualist."""


class InvalidConfig(ContinualistError):
    """Configuration is malformed."""


class InvalidGradient(ContinualistError):
    """Gradient vector is malformed."""


class InvalidTheta(ContinualistError):
    """Parameter vector is malformed."""


class InvalidTask(ContinualistError):
    """Task identifier is malformed."""


class InsufficientData(ContinualistError):
    """Not enough committed tasks / observations for the requested operation."""


class UnknownMethod(ContinualistError):
    """Unknown continual-learning method."""


class UnknownReplayStrategy(ContinualistError):
    """Unknown replay-buffer strategy."""


class UnknownTask(ContinualistError):
    """Task id not registered."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinualistConfig:
    """Configuration for :class:`Continualist`.

    Parameters
    ----------
    method : str
        One of ``METHOD_*``. Drives which regulariser is applied.
    dim : int
        Parameter dimension ``d``.  All gradients / θ vectors must
        have this length.
    ewc_lambda : float
        EWC regularisation strength λ (Kirkpatrick 2017).
    fisher_decay : float
        Online-EWC accumulator decay γ ∈ [0, 1] (Schwarz 2018).
    si_c : float
        Synaptic Intelligence strength c (Zenke 2017).
    si_xi : float
        Synaptic Intelligence dampening ξ (avoids division blow-up).
    mas_lambda : float
        MAS regularisation strength.
    agem_eps : float
        A-GEM projection epsilon (Chaudhry 2018) — denominator floor.
    replay_capacity : int
        Maximum replay buffer size B.
    replay_strategy : str
        One of ``REPLAY_*``.
    boundary_detection : bool
        If True, BOCD self-segments the stream when no explicit task
        commit is given.
    boundary_hazard : float
        Constant hazard rate ``H ∈ (0, 1)`` for BOCD.  Lower = rarer
        change-points.
    boundary_threshold : float
        Run-length-0 posterior probability threshold above which a
        boundary is fired (default 0.5).
    confidence : float
        PAC-Bayes / accuracy-bound confidence ``1 − δ``.
    plasticity_min : float
        Minimum acceptable per-task fresh accuracy for the
        plasticity-stability certificate.
    stability_eps : float
        Maximum acceptable drop in any previous-task accuracy for the
        plasticity-stability certificate.
    seed : int
        Reservoir-sampling and projection-jitter RNG seed.
    hmac_key : bytes | None
        Optional HMAC key for the ledger chain (signs every block).
    """

    method: str = METHOD_ONLINE_EWC
    dim: int = 1
    ewc_lambda: float = 1.0
    fisher_decay: float = 0.95
    si_c: float = 0.1
    si_xi: float = 1e-3
    mas_lambda: float = 1.0
    agem_eps: float = 1e-12
    replay_capacity: int = 256
    replay_strategy: str = REPLAY_RESERVOIR
    boundary_detection: bool = False
    boundary_hazard: float = 1e-2
    boundary_threshold: float = 0.5
    confidence: float = 0.95
    plasticity_min: float = 0.5
    stability_eps: float = 0.05
    seed: int = 0
    hmac_key: bytes | None = None

    def __post_init__(self) -> None:
        if self.method not in KNOWN_METHODS:
            raise UnknownMethod(self.method)
        if self.replay_strategy not in KNOWN_REPLAY_STRATEGIES:
            raise UnknownReplayStrategy(self.replay_strategy)
        if not isinstance(self.dim, int) or self.dim < 1:
            raise InvalidConfig("dim must be a positive integer")
        for name in (
            "ewc_lambda",
            "si_c",
            "si_xi",
            "mas_lambda",
            "agem_eps",
            "boundary_hazard",
            "boundary_threshold",
            "plasticity_min",
            "stability_eps",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise InvalidConfig(f"{name} must be finite")
            if v < 0:
                raise InvalidConfig(f"{name} must be non-negative")
        if not (0.0 <= self.fisher_decay <= 1.0):
            raise InvalidConfig("fisher_decay must lie in [0, 1]")
        if not (0.0 < self.boundary_hazard < 1.0):
            raise InvalidConfig("boundary_hazard must lie in (0, 1)")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must lie in (0, 1)")
        if not isinstance(self.replay_capacity, int) or self.replay_capacity < 1:
            raise InvalidConfig("replay_capacity must be a positive integer")
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes or None")


# ---------------------------------------------------------------------------
# Records / outputs
# ---------------------------------------------------------------------------


@dataclass
class TaskRecord:
    """Live record of a single task over the agent's lifetime."""

    task_id: str
    registered_at_step: int
    committed: bool = False
    n_updates: int = 0
    # Accuracy *on this task* observed during training.
    train_accuracy: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    # Per-task held-out accuracy at every commit point.
    # accuracy_matrix[task_index_committed][self] is the relevant
    # entry; we store the full row here, per training-task index.
    accuracy_at_commit: dict[int, float] = field(default_factory=dict)
    final_accuracy: float | None = None
    initial_accuracy: float | None = None
    final_loss: float | None = None
    final_theta: tuple[float, ...] | None = None


@dataclass(frozen=True)
class UpdateOutput:
    """Result of a single :meth:`Continualist.update`."""

    step: int
    task_id: str
    regulariser: float
    regulariser_grad: tuple[float, ...]
    head: str
    boundary_probability: float


@dataclass(frozen=True)
class ProjectionOutput:
    """Result of :meth:`Continualist.project_gradient`."""

    projected: tuple[float, ...]
    was_projected: bool
    inner_product: float
    head: str


@dataclass
class ReplayItem:
    """One reservoir-sampled item from the lifetime stream."""

    task_id: str
    step: int
    gradient: tuple[float, ...]
    loss: float
    label: str | None = None


@dataclass(frozen=True)
class Boundary:
    """Detected task boundary (BOCD or explicit)."""

    step: int
    probability: float
    explicit: bool
    head: str


@dataclass(frozen=True)
class ContinualistReport:
    """Continual evaluation metrics over committed tasks."""

    n_tasks: int
    n_steps: int
    backward_transfer: float
    forward_transfer: float
    average_accuracy: float
    forgetting: float
    plasticity: float
    n_boundaries: int
    n_projections_applied: int
    replay_size: int
    head: str


@dataclass(frozen=True)
class ContinualistCertificate:
    """PAC-Bayes + plasticity-stability certificate."""

    n_tasks: int
    empirical_mean_risk: float
    pac_bayes_bound: float
    kl_complexity: float
    confidence: float
    plasticity_ok: bool
    stability_ok: bool
    min_fresh_accuracy: float
    max_forget_gap: float
    head: str


# ---------------------------------------------------------------------------
# Ledger chain
# ---------------------------------------------------------------------------


_GENESIS_PREFIX = b"agi.continualist.v1\x00"


def continualist_ledger_root(secret_key: bytes | None = None) -> str:
    """Genesis chain head for a Continualist ledger."""
    seed = _GENESIS_PREFIX + (secret_key or b"")
    return hashlib.sha256(seed).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    def _q(o: Any) -> Any:
        if isinstance(o, float):
            if math.isnan(o):
                return "NaN"
            if math.isinf(o):
                return "Infinity" if o > 0 else "-Infinity"
            return float(repr(o))
        if isinstance(o, dict):
            return {str(k): _q(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [_q(x) for x in o]
        if isinstance(o, (bytes, bytearray)):
            return o.hex()
        return o

    return json.dumps(_q(payload), sort_keys=True, separators=(",", ":")).encode()


def _hash_entry(parent: str, payload: Mapping[str, Any], hmac_key: bytes | None = None) -> str:
    body = _canonical(payload)
    block = parent.encode() + b"|" + body
    if hmac_key:
        return hmac.new(hmac_key, block, hashlib.sha256).hexdigest()
    return hashlib.sha256(block).hexdigest()


# ---------------------------------------------------------------------------
# Pure-function metrics (Lopez-Paz & Ranzato 2017)
# ---------------------------------------------------------------------------


def backward_transfer(R: Sequence[Sequence[float]]) -> float:
    """``BWT = (1/(T−1)) Σ_{i<T} (R_{T,i} − R_{i,i})``."""
    T = len(R)
    if T < 2:
        return 0.0
    last = R[T - 1]
    total = 0.0
    for i in range(T - 1):
        if i >= len(last) or i >= len(R[i]):
            continue
        total += float(last[i]) - float(R[i][i])
    return total / (T - 1)


def forward_transfer(R: Sequence[Sequence[float]], baseline: Sequence[float]) -> float:
    """``FWT = (1/(T−1)) Σ_{i>1} (R_{i−1,i} − R̄_i)``.

    ``baseline[i]`` = ``R̄_i`` = accuracy a randomly initialised
    model would get on task i without having seen it.
    """
    T = len(R)
    if T < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(1, T):
        if i >= len(R[i - 1]):
            continue
        if i >= len(baseline):
            continue
        total += float(R[i - 1][i]) - float(baseline[i])
        count += 1
    if count == 0:
        return 0.0
    return total / count


def average_accuracy(R: Sequence[Sequence[float]]) -> float:
    """``A = (1/T) Σ_i R_{T,i}``."""
    T = len(R)
    if T == 0:
        return 0.0
    last = R[T - 1]
    if not last:
        return 0.0
    return sum(float(x) for x in last[:T]) / min(T, len(last))


def forgetting_metric(R: Sequence[Sequence[float]]) -> float:
    """``F = (1/(T−1)) Σ_{i<T} max_{t≤T} (R_{t,i} − R_{T,i})``."""
    T = len(R)
    if T < 2:
        return 0.0
    last = R[T - 1]
    total = 0.0
    count = 0
    for i in range(T - 1):
        if i >= len(last):
            continue
        best = float("-inf")
        for t in range(T):
            if i < len(R[t]):
                best = max(best, float(R[t][i]))
        if best == float("-inf"):
            continue
        total += best - float(last[i])
        count += 1
    if count == 0:
        return 0.0
    return total / count


def pac_bayes_continual_bound(
    *,
    empirical_mean_risk: float,
    kl_complexity: float,
    n_tasks: int,
    n_samples_per_task: int,
    confidence: float,
) -> float:
    """Pentina-Lampert 2014 lifelong-learning PAC-Bayes bound.

    Given ``T`` tasks of size ``n`` each, the expected risk on a new
    task from the same environment is bounded above (w.p. ≥ 1 − δ) by

        ``R̄ + sqrt((KL(Q‖P) + ln(2T/δ)) / (2 n))``.
    """
    if n_tasks <= 0 or n_samples_per_task <= 0:
        return float("inf")
    delta = 1.0 - confidence
    if delta <= 0.0:
        return float("inf")
    if kl_complexity < 0.0 or math.isnan(kl_complexity) or math.isinf(kl_complexity):
        kl_complexity = 0.0
    num = kl_complexity + math.log(max(2.0 * n_tasks / delta, math.e))
    half = math.sqrt(num / (2.0 * max(n_samples_per_task, 1)))
    return float(empirical_mean_risk) + half


# ---------------------------------------------------------------------------
# Vector helpers (pure stdlib; never depend on numpy)
# ---------------------------------------------------------------------------


def _check_vec(v: Sequence[float], dim: int, label: str) -> tuple[float, ...]:
    if not hasattr(v, "__len__"):
        raise InvalidGradient(f"{label} must be a sequence")
    if len(v) != dim:
        raise InvalidGradient(f"{label} must have length {dim}, got {len(v)}")
    out: list[float] = []
    for x in v:
        if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
            raise InvalidGradient(f"{label} entries must be finite")
        out.append(float(x))
    return tuple(out)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _l2(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


# ---------------------------------------------------------------------------
# Bayesian Online Change-Point Detection (Adams & MacKay 2007)
# ---------------------------------------------------------------------------


class _BOCD:
    """Run-length posterior over a streaming Gaussian observation.

    Constant hazard ``H``; conjugate Normal-Gamma prior on (μ, τ).
    Tracks per-run-length predictive sufficient statistics, prunes to
    a maximum run-length window so memory is bounded.
    """

    def __init__(
        self,
        *,
        hazard: float,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 256,
    ) -> None:
        self.hazard = float(hazard)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.max_run_length = int(max_run_length)
        # Per-run-length statistics — list parallel to run-length 0..R.
        self._mu: list[float] = [self.mu0]
        self._kappa: list[float] = [self.kappa0]
        self._alpha: list[float] = [self.alpha0]
        self._beta: list[float] = [self.beta0]
        self._log_prob: list[float] = [0.0]  # log p(r=0) = 0

    def step(self, x: float) -> float:
        """Ingest ``x``; return change-recently posterior signal.

        Specifically, returns the posterior mass on *short* run-lengths
        (``P(r_t < n/2 | x_{1:t})`` where ``n`` is the truncated
        support).  After a long stable run the run-length posterior
        peaks at the true run length and this signal sits near 0;
        after a true change-point the mass collapses to small run
        lengths and this signal jumps toward 1.

        Returning ``P(r_t = 0)`` alone is uninformative — the marginal
        always equals the hazard ``H`` (Adams & MacKay 2007 §3.1
        eq. 3): the prior probability of *any* change is structural,
        independent of the data.  The discriminative signal lives in
        the shape of the full run-length posterior.
        """
        x = float(x)
        # Student-T predictive likelihood per run-length.
        log_pred: list[float] = []
        for mu, kappa, alpha, beta in zip(self._mu, self._kappa, self._alpha, self._beta):
            df = 2.0 * alpha
            scale = math.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
            log_pred.append(_log_student_t(x, mu, df, scale))
        log_h = math.log(self.hazard)
        log_1mh = math.log(1.0 - self.hazard)
        # Posterior at r=0 (change): logsumexp(prior + likelihood + log_h).
        terms = [lp + lh + log_h for lp, lh in zip(self._log_prob, log_pred)]
        log_change = _log_sum_exp(terms)
        # Posterior at r>0 (growth): elementwise prior + likelihood + log_1mh.
        log_growth = [lp + lh + log_1mh for lp, lh in zip(self._log_prob, log_pred)]
        new_log_prob = [log_change] + log_growth
        # Normalise.
        norm = _log_sum_exp(new_log_prob)
        new_log_prob = [v - norm for v in new_log_prob]
        # Update sufficient statistics: new[0] = prior; new[r+1] = posterior update of r.
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]
        for mu, kappa, alpha, beta in zip(self._mu, self._kappa, self._alpha, self._beta):
            new_kappa.append(kappa + 1.0)
            new_mu.append((kappa * mu + x) / (kappa + 1.0))
            new_alpha.append(alpha + 0.5)
            new_beta.append(beta + (kappa * (x - mu) ** 2) / (2.0 * (kappa + 1.0)))
        # Prune.
        if len(new_log_prob) > self.max_run_length:
            new_log_prob = new_log_prob[: self.max_run_length]
            new_mu = new_mu[: self.max_run_length]
            new_kappa = new_kappa[: self.max_run_length]
            new_alpha = new_alpha[: self.max_run_length]
            new_beta = new_beta[: self.max_run_length]
            # Re-normalise after prune.
            n = _log_sum_exp(new_log_prob)
            new_log_prob = [v - n for v in new_log_prob]
        self._log_prob = new_log_prob
        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta
        # Discriminative signal: cumulative posterior mass on short
        # run-lengths.  Below 4 observations the signal isn't reliable.
        n = len(new_log_prob)
        if n < 4:
            return 0.0
        cutoff = max(1, n // 2)
        log_mass = _log_sum_exp(new_log_prob[:cutoff])
        return math.exp(log_mass)

    def reset(self) -> None:
        self._mu = [self.mu0]
        self._kappa = [self.kappa0]
        self._alpha = [self.alpha0]
        self._beta = [self.beta0]
        self._log_prob = [0.0]


def _log_sum_exp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def _log_student_t(x: float, mu: float, df: float, scale: float) -> float:
    """Log density of Student-T(mu, df, scale)."""
    if scale <= 0.0:
        scale = 1e-12
    z = (x - mu) / scale
    log_norm = math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
    log_norm -= 0.5 * math.log(df * math.pi)
    log_norm -= math.log(scale)
    return log_norm - 0.5 * (df + 1.0) * math.log(1.0 + z * z / df)


# ---------------------------------------------------------------------------
# The Continualist
# ---------------------------------------------------------------------------


EventPublisher = Callable[[str, dict[str, Any]], None]


class Continualist:
    """Continual / lifelong learning as a runtime primitive.

    Thread-safe at the API surface: a single re-entrant lock guards
    every mutation.
    """

    def __init__(
        self,
        config: ContinualistConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or ContinualistConfig()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._rng = random.Random(self.config.seed)
        # State.
        self._step: int = 0
        self._tasks: dict[str, TaskRecord] = {}
        self._task_order: list[str] = []
        self._committed_tasks: list[str] = []
        # Importance / anchor state.
        d = self.config.dim
        self._anchor: list[float] = [0.0] * d
        self._fisher: list[float] = [0.0] * d  # Online-EWC
        self._si_omega: list[float] = [0.0] * d  # Synaptic Intelligence path integral
        self._si_importance: list[float] = [0.0] * d  # Committed SI importance
        self._mas_importance: list[float] = [0.0] * d  # MAS importance
        self._task_start_theta: list[float] = [0.0] * d
        self._last_theta: list[float] = [0.0] * d
        # Per-step accumulators (for current task).
        self._task_grad_sq_sum: list[float] = [0.0] * d  # for Fisher fit
        self._task_grad_sq_count: int = 0
        # Replay buffer.
        self._replay: list[ReplayItem] = []
        self._stream_seen: int = 0  # for reservoir sampling
        self._replay_label_counts: dict[str, int] = {}
        # Accuracy matrix (R[t][s] = accuracy on task s after training task t).
        # We store it as a list of dicts keyed by task index.
        self._R: list[list[float]] = []
        # Detected boundaries.
        self._boundaries: list[Boundary] = []
        self._bocd = _BOCD(
            hazard=self.config.boundary_hazard,
            max_run_length=max(64, self.config.replay_capacity),
        )
        # A-GEM stats.
        self._n_projections: int = 0
        # Chain.
        self._chain_head: str = continualist_ledger_root(self.config.hmac_key)
        self._publish_and_chain(
            CONTINUALIST_STARTED,
            {
                "method": self.config.method,
                "dim": self.config.dim,
                "replay_strategy": self.config.replay_strategy,
                "boundary_detection": self.config.boundary_detection,
                "confidence": self.config.confidence,
            },
        )

    # ------------------------------------------------------------------
    # Ledger helpers
    # ------------------------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, payload)
        except Exception:
            pass

    def _advance_chain(self, payload: Mapping[str, Any]) -> str:
        self._chain_head = _hash_entry(self._chain_head, payload, self.config.hmac_key)
        return self._chain_head

    def _publish_and_chain(self, kind: str, payload: dict[str, Any]) -> str:
        head = self._advance_chain({"k": kind, **payload})
        self._publish(kind, {**payload, "head": head})
        return head

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._chain_head

    @property
    def step(self) -> int:
        with self._lock:
            return self._step

    @property
    def n_tasks(self) -> int:
        with self._lock:
            return len(self._tasks)

    @property
    def n_committed(self) -> int:
        with self._lock:
            return len(self._committed_tasks)

    # ------------------------------------------------------------------
    # Task registration / boundaries
    # ------------------------------------------------------------------

    def register_task(self, task_id: str) -> TaskRecord:
        """Begin a new task — sets the SI / EWC anchor to current θ."""
        with self._lock:
            if not isinstance(task_id, str) or not task_id:
                raise InvalidTask("task_id must be a non-empty string")
            if task_id in self._tasks:
                raise InvalidTask(f"task already registered: {task_id}")
            rec = TaskRecord(task_id=task_id, registered_at_step=self._step)
            self._tasks[task_id] = rec
            self._task_order.append(task_id)
            self._task_start_theta = list(self._last_theta)
            # Reset SI per-task accumulator.
            self._si_omega = [0.0] * self.config.dim
            # Reset Fisher accumulator for new task (per-task block of online EWC).
            self._task_grad_sq_sum = [0.0] * self.config.dim
            self._task_grad_sq_count = 0
            self._publish_and_chain(
                CONTINUALIST_REGISTERED,
                {"task_id": task_id, "step": self._step},
            )
            return rec

    def task_boundary(self, *, explicit: bool = True) -> Boundary:
        """Record an explicit task boundary (independent of register).

        Use this when downstream code does *not* commit a task but
        wants the change recorded for replay / certificate.
        """
        with self._lock:
            b = Boundary(
                step=self._step,
                probability=1.0,
                explicit=bool(explicit),
                head=self._chain_head,
            )
            self._boundaries.append(b)
            self._publish_and_chain(
                CONTINUALIST_BOUNDARY,
                {"step": b.step, "probability": 1.0, "explicit": bool(explicit)},
            )
            return Boundary(
                step=b.step,
                probability=b.probability,
                explicit=b.explicit,
                head=self._chain_head,
            )

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(
        self,
        task_id: str,
        *,
        grad: Sequence[float],
        theta: Sequence[float],
        loss: float,
        accuracy: float | None = None,
        label: str | None = None,
        delta_theta: Sequence[float] | None = None,
        boundary_signal: float | None = None,
    ) -> UpdateOutput:
        """Ingest one training step from the downstream learner.

        Parameters
        ----------
        task_id : str
            Identifier of the *current* task.  Must have been
            previously registered.
        grad : Sequence[float]
            Gradient at the *previous* θ (i.e. the gradient just
            applied), used for SI path-integral and reservoir.
        theta : Sequence[float]
            θ *after* the step.
        loss : float
            Scalar training loss the step yielded.
        accuracy : float | None
            Optional [0, 1] accuracy on the current task.
        label : str | None
            Optional per-item class label for balanced reservoir.
        delta_theta : Sequence[float] | None
            Optional Δθ for SI; if absent we compute ``theta − last_theta``.
        boundary_signal : float | None
            Optional scalar to feed BOCD.  If absent and
            ``boundary_detection`` is True, we feed the loss.

        Returns
        -------
        UpdateOutput
            Per-step receipt including the regulariser value, its
            gradient (so the trainer can add it on the next step),
            the new chain head, and the BOCD boundary probability.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise UnknownTask(task_id)
            rec = self._tasks[task_id]
            d = self.config.dim
            g = _check_vec(grad, d, "grad")
            theta_v = _check_vec(theta, d, "theta")
            if not isinstance(loss, (int, float)) or math.isnan(loss) or math.isinf(loss):
                raise InvalidGradient("loss must be finite")
            loss_f = float(loss)
            if accuracy is not None:
                if not isinstance(accuracy, (int, float)) or math.isnan(accuracy) or math.isinf(accuracy):
                    raise InvalidGradient("accuracy must be finite")
                if not (0.0 <= float(accuracy) <= 1.0):
                    raise InvalidGradient("accuracy must lie in [0, 1]")
            self._step += 1
            rec.n_updates += 1
            rec.train_loss.append(loss_f)
            if accuracy is not None:
                rec.train_accuracy.append(float(accuracy))
                if rec.initial_accuracy is None:
                    rec.initial_accuracy = float(accuracy)
                rec.final_accuracy = float(accuracy)
            rec.final_loss = loss_f
            # Compute Δθ.
            if delta_theta is None:
                dtheta_v = tuple(theta_v[i] - self._last_theta[i] for i in range(d))
            else:
                dtheta_v = _check_vec(delta_theta, d, "delta_theta")
            # SI path integral: ω_i ← ω_i − g_i · Δθ_i.
            for i in range(d):
                self._si_omega[i] -= g[i] * dtheta_v[i]
            # Online EWC Fisher accumulator (gradient-squared average).
            for i in range(d):
                self._task_grad_sq_sum[i] += g[i] * g[i]
            self._task_grad_sq_count += 1
            # Compute regulariser at *new* theta.
            reg_value, reg_grad = self._regulariser(theta_v)
            # Replay push.
            self._replay_push(task_id, g, loss_f, label=label)
            # Update last theta.
            self._last_theta = list(theta_v)
            # Boundary detection.
            bp = 0.0
            if self.config.boundary_detection:
                signal = loss_f if boundary_signal is None else float(boundary_signal)
                if math.isnan(signal) or math.isinf(signal):
                    signal = 0.0
                bp = self._bocd.step(signal)
                if bp >= self.config.boundary_threshold:
                    b = Boundary(
                        step=self._step,
                        probability=bp,
                        explicit=False,
                        head=self._chain_head,
                    )
                    self._boundaries.append(b)
                    self._publish_and_chain(
                        CONTINUALIST_BOUNDARY,
                        {"step": self._step, "probability": bp, "explicit": False},
                    )
                    # Reset the change-point posterior after firing.
                    self._bocd.reset()
            head = self._publish_and_chain(
                CONTINUALIST_UPDATED,
                {
                    "task_id": task_id,
                    "step": self._step,
                    "loss": loss_f,
                    "accuracy": float(accuracy) if accuracy is not None else None,
                    "regulariser": float(reg_value),
                    "boundary_probability": float(bp),
                },
            )
            return UpdateOutput(
                step=self._step,
                task_id=task_id,
                regulariser=float(reg_value),
                regulariser_grad=reg_grad,
                head=head,
                boundary_probability=float(bp),
            )

    # ------------------------------------------------------------------
    # Regulariser (EWC / SI / MAS / replay / lwf / none)
    # ------------------------------------------------------------------

    def regulariser(self, theta: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        """Public regulariser surface — value and its gradient at θ."""
        with self._lock:
            theta_v = _check_vec(theta, self.config.dim, "theta")
            return self._regulariser(theta_v)

    def _regulariser(self, theta: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        method = self.config.method
        d = self.config.dim
        if method == METHOD_NONE or method == METHOD_AGEM or method == METHOD_REPLAY or method == METHOD_LWF:
            return 0.0, tuple([0.0] * d)
        if method == METHOD_ONLINE_EWC:
            return self._reg_ewc(theta)
        if method == METHOD_SI:
            return self._reg_si(theta)
        if method == METHOD_MAS:
            return self._reg_mas(theta)
        raise UnknownMethod(method)

    def _reg_ewc(self, theta: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        lam = self.config.ewc_lambda
        v = 0.0
        g = [0.0] * self.config.dim
        for i, (ti, ai, fi) in enumerate(zip(theta, self._anchor, self._fisher)):
            d_i = ti - ai
            v += 0.5 * lam * fi * d_i * d_i
            g[i] = lam * fi * d_i
        return v, tuple(g)

    def _reg_si(self, theta: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        c = self.config.si_c
        v = 0.0
        g = [0.0] * self.config.dim
        for i, (ti, ai, oi) in enumerate(zip(theta, self._anchor, self._si_importance)):
            d_i = ti - ai
            v += 0.5 * c * oi * d_i * d_i
            g[i] = c * oi * d_i
        return v, tuple(g)

    def _reg_mas(self, theta: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        lam = self.config.mas_lambda
        v = 0.0
        g = [0.0] * self.config.dim
        for i, (ti, ai, oi) in enumerate(zip(theta, self._anchor, self._mas_importance)):
            d_i = ti - ai
            v += 0.5 * lam * oi * d_i * d_i
            g[i] = lam * oi * d_i
        return v, tuple(g)

    # ------------------------------------------------------------------
    # Importance refresh (MAS — unsupervised)
    # ------------------------------------------------------------------

    def refresh_mas(self, output_grads: Iterable[Sequence[float]]) -> None:
        """Accumulate MAS importance from |∂‖output‖²/∂θ| samples.

        The caller passes an iterable of per-sample gradient vectors of
        ``‖model(x)‖²`` w.r.t. θ.  We average their absolute values.
        """
        with self._lock:
            d = self.config.dim
            acc = [0.0] * d
            n = 0
            for og in output_grads:
                v = _check_vec(og, d, "output_grad")
                for i in range(d):
                    acc[i] += abs(v[i])
                n += 1
            if n == 0:
                raise InsufficientData("output_grads must contain at least one sample")
            new = [acc[i] / n for i in range(d)]
            # MAS importance accumulates across tasks.
            for i in range(d):
                self._mas_importance[i] += new[i]
            self._publish_and_chain(
                CONTINUALIST_UPDATED,
                {"event": "mas_refresh", "step": self._step, "n": n},
            )

    # ------------------------------------------------------------------
    # A-GEM gradient projection
    # ------------------------------------------------------------------

    def project_gradient(self, grad: Sequence[float]) -> ProjectionOutput:
        """A-GEM (Chaudhry 2018) project the new-task gradient against
        the averaged replay gradient ``g_ref``.

        If ``g · g_ref < 0`` we subtract the dominant offending
        component so the projected gradient is the closest one to
        ``g`` that does not increase the average old-task loss.
        """
        with self._lock:
            d = self.config.dim
            g = _check_vec(grad, d, "grad")
            if not self._replay:
                return ProjectionOutput(
                    projected=g,
                    was_projected=False,
                    inner_product=0.0,
                    head=self._chain_head,
                )
            # g_ref = mean(replay.gradient).
            g_ref = [0.0] * d
            for item in self._replay:
                for i in range(d):
                    g_ref[i] += item.gradient[i]
            n = len(self._replay)
            for i in range(d):
                g_ref[i] /= n
            ip = _dot(g, g_ref)
            if ip >= 0.0:
                head = self._publish_and_chain(
                    CONTINUALIST_PROJECTED,
                    {"step": self._step, "projected": False, "inner_product": float(ip)},
                )
                return ProjectionOutput(
                    projected=g,
                    was_projected=False,
                    inner_product=float(ip),
                    head=head,
                )
            denom = _dot(g_ref, g_ref) + self.config.agem_eps
            alpha = ip / denom  # alpha < 0 here
            projected = tuple(g[i] - alpha * g_ref[i] for i in range(d))
            self._n_projections += 1
            head = self._publish_and_chain(
                CONTINUALIST_PROJECTED,
                {
                    "step": self._step,
                    "projected": True,
                    "inner_product": float(ip),
                    "alpha": float(alpha),
                },
            )
            return ProjectionOutput(
                projected=projected,
                was_projected=True,
                inner_product=float(ip),
                head=head,
            )

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def _replay_push(
        self,
        task_id: str,
        grad: Sequence[float],
        loss: float,
        *,
        label: str | None,
    ) -> None:
        item = ReplayItem(
            task_id=task_id,
            step=self._step,
            gradient=tuple(float(x) for x in grad),
            loss=float(loss),
            label=label,
        )
        strat = self.config.replay_strategy
        cap = self.config.replay_capacity
        self._stream_seen += 1
        if strat == REPLAY_RING:
            self._replay.append(item)
            if len(self._replay) > cap:
                self._replay.pop(0)
        elif strat == REPLAY_RESERVOIR:
            # Vitter Algorithm R: keep first B; thereafter swap in with
            # probability cap / stream_seen.
            if len(self._replay) < cap:
                self._replay.append(item)
            else:
                j = self._rng.randrange(self._stream_seen)
                if j < cap:
                    self._replay[j] = item
        elif strat == REPLAY_BALANCED:
            # Class-balanced reservoir (Chrysakis-Moens 2020).
            key = label or "__unlabeled__"
            self._replay_label_counts[key] = self._replay_label_counts.get(key, 0) + 1
            ck = self._replay_label_counts[key]
            # Per-class reservoir of size cap / K with K live classes.
            live = max(1, len(self._replay_label_counts))
            per_class = max(1, cap // live)
            class_items = [(i, it) for i, it in enumerate(self._replay) if (it.label or "__unlabeled__") == key]
            if len(class_items) < per_class:
                self._replay.append(item)
            else:
                j = self._rng.randrange(ck)
                if j < per_class:
                    idx, _ = class_items[j]
                    self._replay[idx] = item
            if len(self._replay) > cap:
                self._replay.pop(0)
        else:  # pragma: no cover - guarded by config
            raise UnknownReplayStrategy(strat)
        self._publish(
            CONTINUALIST_REPLAY_PUSHED,
            {
                "task_id": task_id,
                "step": self._step,
                "size": len(self._replay),
                "stream_seen": self._stream_seen,
            },
        )

    def replay_sample(self, k: int) -> tuple[ReplayItem, ...]:
        """Uniform sample of *k* replay items (or fewer if buffer < k)."""
        with self._lock:
            if not isinstance(k, int) or k < 0:
                raise InvalidGradient("k must be a non-negative integer")
            if not self._replay:
                return tuple()
            n = min(k, len(self._replay))
            idxs = self._rng.sample(range(len(self._replay)), n)
            return tuple(self._replay[i] for i in idxs)

    def replay_size(self) -> int:
        with self._lock:
            return len(self._replay)

    # ------------------------------------------------------------------
    # Commit task
    # ------------------------------------------------------------------

    def commit_task(
        self,
        task_id: str,
        *,
        final_theta: Sequence[float],
        accuracies: Mapping[str, float] | None = None,
    ) -> ContinualistReport:
        """Mark a task as completed and update importance / anchor.

        Parameters
        ----------
        task_id : str
            The task to commit.  Must be currently registered.
        final_theta : Sequence[float]
            θ at the end of training the task — becomes the new
            EWC / SI anchor.
        accuracies : Mapping[str, float] | None
            Held-out accuracy on *every* previously-committed task
            (including this one).  Required for BWT/FWT/forgetting
            metrics.  ``accuracies[task_id]`` is mandatory.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise UnknownTask(task_id)
            rec = self._tasks[task_id]
            if rec.committed:
                raise InvalidTask(f"task already committed: {task_id}")
            d = self.config.dim
            theta_v = _check_vec(final_theta, d, "final_theta")
            # 1. Online-EWC Fisher update.
            if self._task_grad_sq_count > 0:
                avg = [
                    self._task_grad_sq_sum[i] / self._task_grad_sq_count for i in range(d)
                ]
            else:
                avg = [0.0] * d
            gamma = self.config.fisher_decay
            new_fisher = [gamma * self._fisher[i] + avg[i] for i in range(d)]
            # 2. SI importance update.
            new_si_importance = list(self._si_importance)
            for i in range(d):
                delta = theta_v[i] - self._task_start_theta[i]
                new_si_importance[i] += self._si_omega[i] / (delta * delta + self.config.si_xi)
            # 3. Reset SI path integral; anchor.
            self._fisher = new_fisher
            self._si_importance = new_si_importance
            self._si_omega = [0.0] * d
            self._anchor = list(theta_v)
            self._last_theta = list(theta_v)
            self._task_start_theta = list(theta_v)
            self._task_grad_sq_sum = [0.0] * d
            self._task_grad_sq_count = 0
            # 4. Mark committed.
            rec.committed = True
            rec.final_theta = tuple(theta_v)
            self._committed_tasks.append(task_id)
            # 5. Accuracy matrix update.
            t_idx = len(self._committed_tasks) - 1
            row: list[float] = []
            accs = dict(accuracies) if accuracies else {}
            # Mandate accuracy for *this* task.
            if task_id not in accs:
                if rec.final_accuracy is not None:
                    accs[task_id] = rec.final_accuracy
                else:
                    accs[task_id] = 0.0
            for s_id in self._committed_tasks:
                a = accs.get(s_id)
                if a is None:
                    # Carry-forward last known accuracy.
                    prev = self._tasks[s_id].accuracy_at_commit
                    if prev:
                        a = prev[max(prev)]
                    else:
                        a = 0.0
                row.append(float(a))
                self._tasks[s_id].accuracy_at_commit[t_idx] = float(a)
            self._R.append(row)
            self._publish_and_chain(
                CONTINUALIST_COMMITTED,
                {
                    "task_id": task_id,
                    "step": self._step,
                    "final_accuracy": rec.final_accuracy,
                    "row": [float(x) for x in row],
                    "n_committed": len(self._committed_tasks),
                },
            )
            return self.report()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> ContinualistReport:
        with self._lock:
            n_tasks = len(self._committed_tasks)
            bwt = backward_transfer(self._R) if n_tasks >= 2 else 0.0
            # FWT baseline: per-task *initial* accuracy when first registered.
            baseline = [
                (self._tasks[t].initial_accuracy if self._tasks[t].initial_accuracy is not None else 0.0)
                for t in self._committed_tasks
            ]
            fwt = forward_transfer(self._R, baseline) if n_tasks >= 2 else 0.0
            avg = average_accuracy(self._R) if n_tasks >= 1 else 0.0
            forget = forgetting_metric(self._R) if n_tasks >= 2 else 0.0
            # Plasticity := mean(final - initial) per task.
            plasticity_terms: list[float] = []
            for t in self._committed_tasks:
                rec = self._tasks[t]
                if rec.initial_accuracy is not None and rec.final_accuracy is not None:
                    plasticity_terms.append(rec.final_accuracy - rec.initial_accuracy)
            plasticity = (
                sum(plasticity_terms) / len(plasticity_terms) if plasticity_terms else 0.0
            )
            payload = {
                "n_tasks": n_tasks,
                "n_steps": self._step,
                "backward_transfer": float(bwt),
                "forward_transfer": float(fwt),
                "average_accuracy": float(avg),
                "forgetting": float(forget),
                "plasticity": float(plasticity),
                "n_boundaries": len(self._boundaries),
                "n_projections_applied": self._n_projections,
                "replay_size": len(self._replay),
            }
            head = self._publish_and_chain(CONTINUALIST_REPORTED, payload)
            return ContinualistReport(head=head, **payload)

    # ------------------------------------------------------------------
    # Certificate
    # ------------------------------------------------------------------

    def certify(
        self,
        *,
        n_samples_per_task: int | None = None,
    ) -> ContinualistCertificate:
        """PAC-Bayes continual-risk + plasticity-stability certificate.

        ``n_samples_per_task`` defaults to mean(rec.n_updates) over
        committed tasks.
        """
        with self._lock:
            T = len(self._committed_tasks)
            if T < 1:
                raise InsufficientData("certify requires at least one committed task")
            # Empirical mean risk = 1 − avg accuracy on each task at commit.
            risks: list[float] = []
            for i, tid in enumerate(self._committed_tasks):
                a = self._R[i][i] if i < len(self._R) and i < len(self._R[i]) else 0.0
                risks.append(1.0 - float(a))
            mean_risk = sum(risks) / max(T, 1)
            # KL complexity: For Online-EWC, the posterior precision is
            # diag(λ F + 1) over a unit-prior; KL ≈ 0.5 Σ_i [log(1 + λ F_i)
            # − λF_i/(1+λF_i) + (θ*_i)² · 1/(1+λF_i)].
            kl = 0.0
            if self.config.method == METHOD_ONLINE_EWC:
                lam = self.config.ewc_lambda
                for fi, ai in zip(self._fisher, self._anchor):
                    pi = 1.0 + lam * fi
                    if pi <= 0.0:
                        continue
                    kl += 0.5 * (math.log(pi) - (lam * fi) / pi + (ai * ai) / pi)
            elif self.config.method == METHOD_SI:
                c = self.config.si_c
                for oi, ai in zip(self._si_importance, self._anchor):
                    pi = 1.0 + c * abs(oi)
                    if pi <= 0.0:
                        continue
                    kl += 0.5 * (math.log(pi) - (c * abs(oi)) / pi + (ai * ai) / pi)
            elif self.config.method == METHOD_MAS:
                lam = self.config.mas_lambda
                for oi, ai in zip(self._mas_importance, self._anchor):
                    pi = 1.0 + lam * oi
                    if pi <= 0.0:
                        continue
                    kl += 0.5 * (math.log(pi) - (lam * oi) / pi + (ai * ai) / pi)
            # Default n: mean per-task updates.
            if n_samples_per_task is None:
                ns = [self._tasks[t].n_updates for t in self._committed_tasks]
                n = max(1, sum(ns) // max(len(ns), 1))
            else:
                if not isinstance(n_samples_per_task, int) or n_samples_per_task < 1:
                    raise InvalidGradient("n_samples_per_task must be a positive int")
                n = n_samples_per_task
            bound = pac_bayes_continual_bound(
                empirical_mean_risk=mean_risk,
                kl_complexity=kl,
                n_tasks=T,
                n_samples_per_task=n,
                confidence=self.config.confidence,
            )
            # Plasticity-stability.
            fresh_accs = [
                (self._tasks[t].final_accuracy if self._tasks[t].final_accuracy is not None else 0.0)
                for t in self._committed_tasks
            ]
            min_fresh = min(fresh_accs) if fresh_accs else 0.0
            # Forget gap per old task = best - current.
            max_gap = 0.0
            if T >= 2:
                last_row = self._R[-1]
                for i in range(T - 1):
                    if i >= len(last_row):
                        continue
                    best = max(self._R[t][i] for t in range(T) if i < len(self._R[t]))
                    gap = best - last_row[i]
                    if gap > max_gap:
                        max_gap = gap
            plasticity_ok = min_fresh >= self.config.plasticity_min
            stability_ok = max_gap <= self.config.stability_eps
            payload = {
                "n_tasks": T,
                "empirical_mean_risk": float(mean_risk),
                "pac_bayes_bound": float(bound),
                "kl_complexity": float(kl),
                "confidence": float(self.config.confidence),
                "plasticity_ok": bool(plasticity_ok),
                "stability_ok": bool(stability_ok),
                "min_fresh_accuracy": float(min_fresh),
                "max_forget_gap": float(max_gap),
            }
            head = self._publish_and_chain(CONTINUALIST_CERTIFIED, payload)
            return ContinualistCertificate(head=head, **payload)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "v": 1,
                "config": _config_to_dict(self.config),
                "step": self._step,
                "task_order": list(self._task_order),
                "committed_tasks": list(self._committed_tasks),
                "tasks": {tid: _task_to_dict(rec) for tid, rec in self._tasks.items()},
                "anchor": list(self._anchor),
                "fisher": list(self._fisher),
                "si_omega": list(self._si_omega),
                "si_importance": list(self._si_importance),
                "mas_importance": list(self._mas_importance),
                "task_start_theta": list(self._task_start_theta),
                "last_theta": list(self._last_theta),
                "task_grad_sq_sum": list(self._task_grad_sq_sum),
                "task_grad_sq_count": self._task_grad_sq_count,
                "replay": [
                    {
                        "task_id": it.task_id,
                        "step": it.step,
                        "gradient": list(it.gradient),
                        "loss": it.loss,
                        "label": it.label,
                    }
                    for it in self._replay
                ],
                "stream_seen": self._stream_seen,
                "replay_label_counts": dict(self._replay_label_counts),
                "R": [list(r) for r in self._R],
                "boundaries": [
                    {
                        "step": b.step,
                        "probability": b.probability,
                        "explicit": b.explicit,
                        "head": b.head,
                    }
                    for b in self._boundaries
                ],
                "n_projections": self._n_projections,
                "chain_head": self._chain_head,
                "rng_state": list(self._rng.getstate()[1]),
            }

    def restore(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            if state.get("v") != 1:
                raise InvalidConfig("snapshot version mismatch")
            cfg = _config_from_dict(state["config"])
            self.config = cfg
            self._step = int(state["step"])
            self._task_order = list(state["task_order"])
            self._committed_tasks = list(state["committed_tasks"])
            self._tasks = {tid: _task_from_dict(d) for tid, d in state["tasks"].items()}
            self._anchor = [float(x) for x in state["anchor"]]
            self._fisher = [float(x) for x in state["fisher"]]
            self._si_omega = [float(x) for x in state["si_omega"]]
            self._si_importance = [float(x) for x in state["si_importance"]]
            self._mas_importance = [float(x) for x in state["mas_importance"]]
            self._task_start_theta = [float(x) for x in state["task_start_theta"]]
            self._last_theta = [float(x) for x in state["last_theta"]]
            self._task_grad_sq_sum = [float(x) for x in state["task_grad_sq_sum"]]
            self._task_grad_sq_count = int(state["task_grad_sq_count"])
            self._replay = [
                ReplayItem(
                    task_id=it["task_id"],
                    step=int(it["step"]),
                    gradient=tuple(float(x) for x in it["gradient"]),
                    loss=float(it["loss"]),
                    label=it.get("label"),
                )
                for it in state["replay"]
            ]
            self._stream_seen = int(state["stream_seen"])
            self._replay_label_counts = dict(state["replay_label_counts"])
            self._R = [[float(x) for x in row] for row in state["R"]]
            self._boundaries = [
                Boundary(
                    step=int(b["step"]),
                    probability=float(b["probability"]),
                    explicit=bool(b["explicit"]),
                    head=str(b["head"]),
                )
                for b in state["boundaries"]
            ]
            self._n_projections = int(state["n_projections"])
            self._chain_head = str(state["chain_head"])
            rng_state = tuple(state["rng_state"])
            # Tail token Python adds is included; restore as full state.
            try:
                self._rng.setstate((3, tuple(int(x) for x in rng_state), None))
            except (TypeError, ValueError):
                self._rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            d = self.config.dim
            self._step = 0
            self._tasks.clear()
            self._task_order.clear()
            self._committed_tasks.clear()
            self._anchor = [0.0] * d
            self._fisher = [0.0] * d
            self._si_omega = [0.0] * d
            self._si_importance = [0.0] * d
            self._mas_importance = [0.0] * d
            self._task_start_theta = [0.0] * d
            self._last_theta = [0.0] * d
            self._task_grad_sq_sum = [0.0] * d
            self._task_grad_sq_count = 0
            self._replay.clear()
            self._stream_seen = 0
            self._replay_label_counts.clear()
            self._R.clear()
            self._boundaries.clear()
            self._n_projections = 0
            self._bocd.reset()
            self._chain_head = continualist_ledger_root(self.config.hmac_key)
            self._publish(CONTINUALIST_RESET, {"step": 0})


# ---------------------------------------------------------------------------
# Snapshot serialisation helpers
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: ContinualistConfig) -> dict[str, Any]:
    out = {
        "method": cfg.method,
        "dim": cfg.dim,
        "ewc_lambda": cfg.ewc_lambda,
        "fisher_decay": cfg.fisher_decay,
        "si_c": cfg.si_c,
        "si_xi": cfg.si_xi,
        "mas_lambda": cfg.mas_lambda,
        "agem_eps": cfg.agem_eps,
        "replay_capacity": cfg.replay_capacity,
        "replay_strategy": cfg.replay_strategy,
        "boundary_detection": cfg.boundary_detection,
        "boundary_hazard": cfg.boundary_hazard,
        "boundary_threshold": cfg.boundary_threshold,
        "confidence": cfg.confidence,
        "plasticity_min": cfg.plasticity_min,
        "stability_eps": cfg.stability_eps,
        "seed": cfg.seed,
        "hmac_key": cfg.hmac_key.hex() if cfg.hmac_key else None,
    }
    return out


def _config_from_dict(d: Mapping[str, Any]) -> ContinualistConfig:
    hk = d.get("hmac_key")
    if isinstance(hk, str):
        hk = bytes.fromhex(hk)
    return ContinualistConfig(
        method=d["method"],
        dim=int(d["dim"]),
        ewc_lambda=float(d["ewc_lambda"]),
        fisher_decay=float(d["fisher_decay"]),
        si_c=float(d["si_c"]),
        si_xi=float(d["si_xi"]),
        mas_lambda=float(d["mas_lambda"]),
        agem_eps=float(d["agem_eps"]),
        replay_capacity=int(d["replay_capacity"]),
        replay_strategy=d["replay_strategy"],
        boundary_detection=bool(d["boundary_detection"]),
        boundary_hazard=float(d["boundary_hazard"]),
        boundary_threshold=float(d["boundary_threshold"]),
        confidence=float(d["confidence"]),
        plasticity_min=float(d["plasticity_min"]),
        stability_eps=float(d["stability_eps"]),
        seed=int(d["seed"]),
        hmac_key=hk,
    )


def _task_to_dict(rec: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": rec.task_id,
        "registered_at_step": rec.registered_at_step,
        "committed": rec.committed,
        "n_updates": rec.n_updates,
        "train_accuracy": list(rec.train_accuracy),
        "train_loss": list(rec.train_loss),
        "accuracy_at_commit": {str(k): float(v) for k, v in rec.accuracy_at_commit.items()},
        "final_accuracy": rec.final_accuracy,
        "initial_accuracy": rec.initial_accuracy,
        "final_loss": rec.final_loss,
        "final_theta": list(rec.final_theta) if rec.final_theta is not None else None,
    }


def _task_from_dict(d: Mapping[str, Any]) -> TaskRecord:
    return TaskRecord(
        task_id=str(d["task_id"]),
        registered_at_step=int(d["registered_at_step"]),
        committed=bool(d["committed"]),
        n_updates=int(d["n_updates"]),
        train_accuracy=[float(x) for x in d["train_accuracy"]],
        train_loss=[float(x) for x in d["train_loss"]],
        accuracy_at_commit={int(k): float(v) for k, v in d["accuracy_at_commit"].items()},
        final_accuracy=d.get("final_accuracy"),
        initial_accuracy=d.get("initial_accuracy"),
        final_loss=d.get("final_loss"),
        final_theta=(tuple(float(x) for x in d["final_theta"]) if d.get("final_theta") is not None else None),
    )
