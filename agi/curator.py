r"""Curator — automated curriculum *generation* as a runtime primitive.

Every long-running runtime that learns eventually faces a problem that
neither ``Cartographer`` nor ``Arbiter`` can answer alone.  Cartographer
selects *from a given pool* of tasks the one with the highest expected
learning progress.  Arbiter commits to the best of *a finite arm set*
with a fixed-confidence bound.  But the problem upstream of both is:
**where does the pool come from?**  AlphaZero is not AlphaZero because
it picks well from a fixed library of board positions — it is AlphaZero
because **self-play generates the positions**, *at a difficulty just
beyond current capability*, *forever*.  The same is true of every
self-improving system: a curriculum is built, not given.

``Curator`` is the runtime primitive that builds it.  Given a
parameterised *task generator* (a function from a difficulty vector
:math:`θ ∈ Θ` to a concrete task) and a *competence oracle* (a function
that runs the agent against a task and returns a 0/1 success), the
``Curator`` maintains an online estimate of the agent's competence
across :math:`Θ` and proposes new tasks drawn from the **frontier of
proximal development** (Vygotsky 1934, Oudeyer & Kaplan 2007): tasks
the agent solves with probability that is neither too low (no signal)
nor too high (no progress), and whose *learning progress* (rate of
decrease of error) is empirically the highest.

The pitch reduced to a runtime call::

    def task_gen(theta: tuple[float, ...]) -> str:
        # interpret theta as a difficulty knob (rule complexity, depth, ...)
        return generate_problem(theta)

    def oracle(task: str) -> bool:
        return run_agent(task) == solve(task)

    cur = Curator(CuratorConfig(
        param_lo=(1.0,), param_hi=(10.0,),
        target_competence=0.6, batch_size=8,
    ))

    for round in range(100):
        proposals = cur.propose(generator=task_gen, oracle=oracle, n=8)
        # `proposals` is a list of (theta, task, predicted_competence) triples
        # in the ZPD: the agent's expected success rate is near `target_competence`.
        for theta, task, _p in proposals:
            success = oracle(task)
            cur.observe(theta=theta, success=success)

    # After enough observations, the Curator's `competence_estimate(theta)`
    # is a calibrated probability across the parameter cube.
    cur.report()       # canonical report with certificate

What this primitive ships
-------------------------

  * **Three generation strategies** (selectable via ``strategy=``):

    * ``"zpd"`` — *zone of proximal development*: sample :math:`θ`
      such that the posterior on competence is closest to
      ``target_competence``.  Uses the same Beta-Binomial conjugate
      posterior as ``Cartographer``.
    * ``"learning_progress"`` — *Oudeyer-Kaplan IAC*: track recent
      vs. older competence, sample :math:`θ` proportional to
      :math:`|\hat μ_{recent} - \hat μ_{prev}|`.
    * ``"thompson_lp"`` — *Thompson sampling over learning progress*:
      draw :math:`θ` with probability proportional to a sampled LP
      from a Dirichlet posterior (Russo et al. 2018).

  * **Difficulty featurisation** — :math:`Θ` is mapped to a small set
    of discrete *cells* via uniform quantisation (default) or a
    user-supplied bucketer.  This keeps the posterior over competence
    tractable without committing to a parametric model.

  * **Difficulty inference** — given a stream of (:math:`θ`, success)
    observations, the Curator fits a *monotone* competence curve
    along each dimension via *isotonic regression* (Brunk et al.
    1972).  Plays well with the common case where increasing the
    difficulty knob monotonically decreases success rate.

  * **Goldilocks frontier** — the **ZPD frontier set** is the set of
    cells whose competence credible interval brackets
    ``target_competence``; the Curator's proposal distribution is
    *uniform-over-frontier* unless ``learning_progress`` or
    ``thompson_lp`` is selected.

  * **Bootstrap from seeds** — without any observations, the Curator
    samples uniformly across the parameter cube.

  * **Calibration check** — ``Curator.brier_score()`` reports the
    Brier score of the predicted-vs-realised success rate over a
    sliding window; if it exceeds a threshold the Curator widens its
    proposal distribution to recover diversity (auto-temperature).

  * **Certificate chain** — SHA-256 over the canonical sequence of
    (proposal, observation) events, replay-verifiable byte-for-byte.

Mathematical and algorithmic roots
----------------------------------

  * **Vygotsky, L. S. (1934) — *Mind in Society: The Development of
    Higher Psychological Processes.***  The original *zone of
    proximal development* — the locus of growth is what the learner
    cannot do alone but can do with scaffolding.

  * **Oudeyer, P.-Y. & Kaplan, F. (2007) — "What is intrinsic
    motivation? A typology of computational approaches."**
    *Frontiers in Neurorobotics* 1 6.  IAC: maximise the rate of
    decrease of prediction error as an intrinsic-reward signal.

  * **Schmidhuber, J. (1991) — "A possibility for implementing
    curiosity and boredom in model-building neural controllers."**
    The information-gain rendering of the same idea.

  * **Graves, A. *et al.* (2017) — "Automated curriculum learning
    for neural networks."**  *Proc. ICML.*  EXP3.S over tasks where
    reward is *gain in competence*; the multi-armed-bandit reduction.

  * **Florensa, C. *et al.* (2018) — "Automatic goal generation for
    reinforcement learning agents."**  *Proc. ICML.*  GoalGAN — a
    generative model of *goals* in the success-probability sweet
    spot.  Curator's ``zpd`` strategy is the discretised analogue.

  * **Wang, R. *et al.* (2019) — "POET: Endlessly generating
    increasingly complex and diverse learning environments and their
    solutions."**  *arXiv:1901.01753.*  Open-ended evolution of
    environment + agent pairs; the architectural inspiration for
    ``learning_progress``.

  * **Dennis, M. *et al.* (2020) — "Emergent complexity and zero-shot
    transfer via unsupervised environment design."**  *Proc.
    NeurIPS.*  Regret as the curriculum signal — closely related to
    Cartographer's competence-gap; the basis of the
    ``thompson_lp`` strategy.

  * **Brunk, H. D. *et al.* (1972) — *Statistical Inference under
    Order Restrictions.***  Isotonic regression via pool-adjacent-
    violators — the difficulty curve fitter.

  * **Wilson, E. B. (1927) — "Probable inference, the law of
    succession, and statistical inference."**  *JASA.*  Score
    interval used as the competence credible band.

  * **Russo, D. *et al.* (2018) — *A Tutorial on Thompson
    Sampling*, FnT in ML 11(1).  Theoretical justification for
    ``thompson_lp``.

  * **Gneiting, T. & Raftery, A. E. (2007) — "Strictly proper
    scoring rules, prediction, and estimation."**  *JASA* 102(477).
    Brier-score calibration check.

What Curator gives a coordination engine
----------------------------------------

It gives the coordinator the **task-generation half** of the closed
self-improvement loop.  ``Searcher`` solves a given task.
``Distiller`` compiles solved tasks into a fast student.  ``Curator``
**proposes new tasks at the ZPD frontier** so the cycle never runs out
of training signal.  Composed with ``Cartographer`` (which picks
*from* the proposals), the runtime has the complete:

    "where do new tasks come from?" — Curator
    "which of the proposed tasks should I attempt next?" — Cartographer
    "given this task, what's the answer?" — Searcher
    "compile the answer into a callable student" — Distiller
    "use the student as the prior for the next Searcher call" — closed.

This is the AlphaGo Zero / MuZero loop, rendered as four in-process
runtime primitives, all stdlib, all certificate-producing.

Public API
----------

The module exposes:

  * ``Cell`` — frozen discretised parameter cell (a tuple of bucket
    indices).
  * ``Proposal`` — frozen ``(theta, task, predicted_competence)``
    triple.
  * ``CompetenceEstimate`` — a Beta-Binomial cell estimate with
    Wilson CI and an optional isotonic-monotone projection.
  * ``CuratorConfig`` / ``CuratorReport`` — configuration and the
    canonical report.
  * ``Curator`` — the orchestrator.
  * Strategy constants: ``STRATEGY_ZPD``, ``STRATEGY_LEARNING_PROGRESS``,
    ``STRATEGY_THOMPSON_LP``.
  * Free-function shortcuts: ``zpd_curator``,
    ``learning_progress_curator``, ``thompson_lp_curator``.

This module is **pure stdlib** — the runtime ships curriculum
generation into the same low-dependency tier as ``Searcher``,
``Distiller``, and ``Cartographer``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# =============================================================================
# Errors
# =============================================================================


class CuratorError(Exception):
    """Base for every Curator-raised error."""


class InvalidConfig(CuratorError):
    """A CuratorConfig is structurally invalid."""


class InvalidParameter(CuratorError):
    """A theta vector is out of the configured parameter cube."""


class InsufficientData(CuratorError):
    """A query requires more observations than have been recorded."""


class UnknownStrategy(CuratorError):
    """The requested strategy is not one of this module's strategies."""


# =============================================================================
# Strategy constants
# =============================================================================


STRATEGY_ZPD = "zpd"
STRATEGY_LEARNING_PROGRESS = "learning_progress"
STRATEGY_THOMPSON_LP = "thompson_lp"

KNOWN_STRATEGIES: Tuple[str, ...] = (
    STRATEGY_ZPD,
    STRATEGY_LEARNING_PROGRESS,
    STRATEGY_THOMPSON_LP,
)


# =============================================================================
# Cells & parameter cube
# =============================================================================


@dataclass(frozen=True)
class Cell:
    """A discretised parameter-cube cell — tuple of bucket indices."""
    indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        for i in self.indices:
            if not isinstance(i, int) or i < 0:
                raise InvalidParameter(
                    f"cell indices must be non-negative ints; got {self.indices!r}"
                )


def _quantise(theta: Sequence[float], lo: Sequence[float],
              hi: Sequence[float], n_buckets: Sequence[int]) -> Cell:
    """Map theta ∈ [lo, hi]^d to a Cell of bucket indices."""
    if len(theta) != len(lo) or len(theta) != len(hi) or len(theta) != len(n_buckets):
        raise InvalidParameter(
            f"theta dim {len(theta)} != lo/hi/n_buckets dim "
            f"({len(lo)}/{len(hi)}/{len(n_buckets)})"
        )
    idx = []
    for t, l, h, n in zip(theta, lo, hi, n_buckets):
        if t < l or t > h:
            raise InvalidParameter(
                f"theta={theta!r} outside cube [{lo!r}, {hi!r}]"
            )
        if h == l:
            idx.append(0)
        else:
            # Map to [0, n) with right-inclusive top
            j = int((t - l) / (h - l) * n)
            if j >= n:
                j = n - 1
            idx.append(j)
    return Cell(indices=tuple(idx))


def _cell_centre(cell: Cell, lo: Sequence[float], hi: Sequence[float],
                 n_buckets: Sequence[int]) -> Tuple[float, ...]:
    """Centre point of the cell in continuous parameter space."""
    out = []
    for j, l, h, n in zip(cell.indices, lo, hi, n_buckets):
        if n == 0:
            out.append(l)
            continue
        width = (h - l) / n
        out.append(l + (j + 0.5) * width)
    return tuple(out)


def _enumerate_cells(n_buckets: Sequence[int]) -> Iterable[Cell]:
    """Iterate over every cell in the parameter cube."""
    def rec(prefix: Tuple[int, ...], dims: Sequence[int]) -> Iterable[Cell]:
        if not dims:
            yield Cell(indices=prefix)
            return
        for j in range(dims[0]):
            yield from rec(prefix + (j,), dims[1:])
    yield from rec((), tuple(n_buckets))


# =============================================================================
# Wilson score CI (same as Cartographer)
# =============================================================================


def _wilson_ci(s: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    # z = 1.96 for alpha=0.05
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _normal_quantile(1 - alpha / 2)
    p = s / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF via the Acklam approximation."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# =============================================================================
# Competence estimate
# =============================================================================


@dataclass
class CompetenceEstimate:
    """Per-cell competence estimate with Wilson CI.

    Uses Beta-Binomial (Jeffreys prior Beta(½, ½)) so the posterior
    mean equals the Wilson score interval centre asymptotically.
    """
    cell: Cell
    successes: int = 0
    n: int = 0
    recent_successes: int = 0
    recent_n: int = 0
    prev_successes: int = 0
    prev_n: int = 0

    @property
    def mean(self) -> float:
        """Posterior mean under Jeffreys prior."""
        return (self.successes + 0.5) / (self.n + 1)

    @property
    def ci(self) -> Tuple[float, float]:
        return _wilson_ci(self.successes, max(1, self.n))

    @property
    def width(self) -> float:
        lo, hi = self.ci
        return hi - lo

    def learning_progress(self) -> float:
        """|μ_recent − μ_prev|: Oudeyer-Kaplan learning-progress signal."""
        if self.recent_n == 0 or self.prev_n == 0:
            return 0.0
        mu_r = (self.recent_successes + 0.5) / (self.recent_n + 1)
        mu_p = (self.prev_successes + 0.5) / (self.prev_n + 1)
        return abs(mu_r - mu_p)


# =============================================================================
# Proposal
# =============================================================================


@dataclass(frozen=True)
class Proposal:
    """A single curriculum proposal."""
    theta: Tuple[float, ...]
    task: Any
    cell: Cell
    predicted_competence: float
    strategy: str


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class CuratorConfig:
    """Configuration for ``Curator``.

    Parameter cube
        param_lo / param_hi:   inclusive bounds of theta cube.
        n_buckets:             discretisation per dimension; default 8 per dim.

    Strategy
        strategy:              one of ``KNOWN_STRATEGIES``.
        target_competence:     ZPD target probability of success.
        target_tolerance:      tolerance band around target for ZPD.

    Memory / learning-progress
        recent_window:         number of recent observations per cell
                               for LP estimation.
        prev_window:           number of prior observations to compare
                               recent against.

    Sampling
        explore_p:             ε-greedy mixture: explore uniformly with
                               probability ``explore_p``.

    Calibration
        brier_window:          number of recent (predicted, realised)
                               pairs to score.

    Determinism / certificate
        seed:                  RNG seed.
        secret_key:            HMAC key for certificate; empty → SHA-256.
    """
    param_lo: Tuple[float, ...] = (0.0,)
    param_hi: Tuple[float, ...] = (1.0,)
    n_buckets: Tuple[int, ...] = (8,)

    strategy: str = STRATEGY_ZPD
    target_competence: float = 0.6
    target_tolerance: float = 0.2

    recent_window: int = 8
    prev_window: int = 8

    explore_p: float = 0.1

    brier_window: int = 64

    seed: int = 0
    secret_key: bytes = b""

    def __post_init__(self) -> None:
        if len(self.param_lo) != len(self.param_hi):
            raise InvalidConfig(
                f"param_lo and param_hi must have the same dim "
                f"({len(self.param_lo)} vs {len(self.param_hi)})"
            )
        if len(self.n_buckets) != len(self.param_lo):
            raise InvalidConfig(
                f"n_buckets dim ({len(self.n_buckets)}) must match "
                f"param dim ({len(self.param_lo)})"
            )
        for l, h in zip(self.param_lo, self.param_hi):
            if h < l:
                raise InvalidConfig(
                    f"param_hi must be ≥ param_lo; got [{l}, {h}]"
                )
        for n in self.n_buckets:
            if n < 1:
                raise InvalidConfig(f"n_buckets entry must be ≥ 1; got {n}")
        if self.strategy not in KNOWN_STRATEGIES:
            raise InvalidConfig(
                f"strategy={self.strategy!r} not in {KNOWN_STRATEGIES}"
            )
        if not (0.0 < self.target_competence < 1.0):
            raise InvalidConfig(
                f"target_competence must be in (0, 1); got {self.target_competence}"
            )
        if not (0.0 < self.target_tolerance < 0.5):
            raise InvalidConfig(
                f"target_tolerance must be in (0, 0.5); got {self.target_tolerance}"
            )
        if self.recent_window < 1:
            raise InvalidConfig(f"recent_window must be ≥ 1; got {self.recent_window}")
        if self.prev_window < 1:
            raise InvalidConfig(f"prev_window must be ≥ 1; got {self.prev_window}")
        if not (0.0 <= self.explore_p <= 1.0):
            raise InvalidConfig(f"explore_p must be in [0,1]; got {self.explore_p}")
        if self.brier_window < 1:
            raise InvalidConfig(f"brier_window must be ≥ 1; got {self.brier_window}")


# =============================================================================
# Report
# =============================================================================


@dataclass
class CuratorReport:
    """Canonical report from a Curator after a round of proposals / observations."""
    strategy: str
    n_observations: int
    n_cells_explored: int
    n_cells_total: int
    n_cells_in_frontier: int
    mean_competence: float
    mean_ci_width: float
    brier_score: Optional[float]
    most_competent_cell: Optional[Tuple[int, ...]]
    most_competent_value: Optional[float]
    target_competence: float
    target_tolerance: float
    seed: int
    certificate: str
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# Certificate chain
# =============================================================================


def _canonical_bytes(obj: Any) -> bytes:
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return b"{" + b",".join(
            _canonical_bytes(k) + b":" + _canonical_bytes(v) for k, v in items
        ) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(_canonical_bytes(x) for x in obj) + b"]"
    if isinstance(obj, bool):
        return b"true" if obj else b"false"
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return f'"{obj}"'.encode("utf-8")
        return repr(obj).encode("utf-8")
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True).encode("utf-8")
    if obj is None:
        return b"null"
    if isinstance(obj, bytes):
        return json.dumps(obj.hex(), ensure_ascii=True).encode("utf-8")
    return json.dumps(repr(obj), ensure_ascii=True).encode("utf-8")


class _CertChain:

    def __init__(self, secret_key: bytes = b"") -> None:
        self._secret = bytes(secret_key)
        seed = b"agi.curator.v1\x00" + self._secret
        self._h = hashlib.sha256(seed).digest()
        self._count = 0

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._count += 1
        body = _canonical_bytes({"k": kind, "n": self._count, "p": payload})
        if self._secret:
            tag = hmac.new(self._secret, self._h + body, hashlib.sha256).digest()
        else:
            tag = hashlib.sha256(self._h + body).digest()
        self._h = tag

    def hexdigest(self) -> str:
        return self._h.hex()

    @property
    def count(self) -> int:
        return self._count


# =============================================================================
# Orchestrator
# =============================================================================


class Curator:
    """Automated curriculum *generation* as a runtime primitive."""

    def __init__(self, config: Optional[CuratorConfig] = None) -> None:
        self.config = config or CuratorConfig()
        self._rng = random.Random(self.config.seed)
        self._cells: Dict[Cell, CompetenceEstimate] = {}
        self._recent_obs: Dict[Cell, List[bool]] = {}  # FIFO per cell
        self._chain = _CertChain(self.config.secret_key)
        self._chain.emit("init", {
            "strategy": self.config.strategy,
            "param_lo": list(self.config.param_lo),
            "param_hi": list(self.config.param_hi),
            "n_buckets": list(self.config.n_buckets),
            "target_competence": self.config.target_competence,
            "target_tolerance": self.config.target_tolerance,
            "seed": self.config.seed,
        })
        # Calibration tracking
        self._brier_buf: List[Tuple[float, bool]] = []

    # ------------------------------------------------------------------
    # cell access
    # ------------------------------------------------------------------

    def _cell_for(self, theta: Sequence[float]) -> Cell:
        return _quantise(theta, self.config.param_lo, self.config.param_hi,
                         self.config.n_buckets)

    def _est_for(self, cell: Cell) -> CompetenceEstimate:
        est = self._cells.get(cell)
        if est is None:
            est = CompetenceEstimate(cell=cell)
            self._cells[cell] = est
        return est

    def cell_centre(self, cell: Cell) -> Tuple[float, ...]:
        return _cell_centre(cell, self.config.param_lo,
                            self.config.param_hi, self.config.n_buckets)

    # ------------------------------------------------------------------
    # observation intake
    # ------------------------------------------------------------------

    def observe(self, *, theta: Sequence[float], success: bool,
                predicted_competence: Optional[float] = None) -> None:
        """Record a (theta, success) outcome.

        If ``predicted_competence`` is supplied (as from a Proposal),
        the (predicted, realised) pair is added to the Brier window.
        """
        cell = self._cell_for(tuple(theta))
        est = self._est_for(cell)
        est.n += 1
        if success:
            est.successes += 1
        # rotate FIFO
        buf = self._recent_obs.setdefault(cell, [])
        buf.append(bool(success))
        max_buf = self.config.recent_window + self.config.prev_window
        if len(buf) > max_buf:
            buf.pop(0)
        # split recent/prev for LP
        rec = self.config.recent_window
        prv = self.config.prev_window
        recent = buf[-rec:]
        prev = buf[-(rec + prv):-rec] if len(buf) > rec else []
        est.recent_n = len(recent)
        est.recent_successes = sum(recent)
        est.prev_n = len(prev)
        est.prev_successes = sum(prev)

        # certificate
        self._chain.emit("observe", {
            "cell": list(cell.indices),
            "theta": list(map(float, theta)),
            "success": bool(success),
        })

        # Brier tracking
        if predicted_competence is not None:
            self._brier_buf.append((float(predicted_competence), bool(success)))
            if len(self._brier_buf) > self.config.brier_window:
                self._brier_buf.pop(0)

    # ------------------------------------------------------------------
    # proposal
    # ------------------------------------------------------------------

    def propose(self, *, n: int,
                generator: Callable[[Tuple[float, ...]], Any],
                oracle: Optional[Callable[[Any], bool]] = None,
                ) -> List[Proposal]:
        """Propose ``n`` new tasks drawn from the configured strategy.

        ``generator`` is called on the centre point of each proposed cell
        to produce the concrete task object.  ``oracle`` is unused here
        (kept on the API for future per-proposal eager evaluation).
        """
        if n < 1:
            raise InvalidConfig("n must be ≥ 1")
        cells = self._pick_cells(n)
        out: List[Proposal] = []
        for cell in cells:
            theta = self.cell_centre(cell)
            task = generator(theta)
            est = self._cells.get(cell)
            pred = est.mean if est is not None else self.config.target_competence
            prop = Proposal(theta=theta, task=task, cell=cell,
                            predicted_competence=pred,
                            strategy=self.config.strategy)
            out.append(prop)
            self._chain.emit("propose", {
                "cell": list(cell.indices),
                "theta": list(map(float, theta)),
                "predicted_competence": pred,
                "strategy": self.config.strategy,
            })
        return out

    def _pick_cells(self, n: int) -> List[Cell]:
        """Pick n cells according to the configured strategy."""
        all_cells = list(_enumerate_cells(self.config.n_buckets))
        if not all_cells:
            return []
        out: List[Cell] = []
        for _ in range(n):
            # ε-greedy: explore uniformly
            if self._rng.random() < self.config.explore_p or not self._cells:
                out.append(all_cells[self._rng.randrange(len(all_cells))])
                continue
            if self.config.strategy == STRATEGY_ZPD:
                out.append(self._pick_zpd(all_cells))
            elif self.config.strategy == STRATEGY_LEARNING_PROGRESS:
                out.append(self._pick_lp(all_cells))
            elif self.config.strategy == STRATEGY_THOMPSON_LP:
                out.append(self._pick_thompson_lp(all_cells))
            else:
                raise UnknownStrategy(
                    f"unknown strategy: {self.config.strategy!r}"
                )
        return out

    def _pick_zpd(self, all_cells: Sequence[Cell]) -> Cell:
        """Pick the cell whose competence estimate is closest to target."""
        tgt = self.config.target_competence
        tol = self.config.target_tolerance
        # Score = -|μ̂ - target| + CI-width bonus for under-sampled cells
        best_score = -math.inf
        best_cell: Cell = all_cells[0]
        unseen_bonus = tol * 2.0  # prefer un-observed cells over far-from-target
        for cell in all_cells:
            est = self._cells.get(cell)
            if est is None or est.n == 0:
                score = -abs(0.5 - tgt) + unseen_bonus
            else:
                score = -abs(est.mean - tgt) + 0.5 * est.width
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell

    def _pick_lp(self, all_cells: Sequence[Cell]) -> Cell:
        """Pick proportional to learning-progress magnitude."""
        weights: List[float] = []
        for cell in all_cells:
            est = self._cells.get(cell)
            if est is None or est.recent_n == 0:
                weights.append(1.0)  # baseline weight for unseen cells
            else:
                # Add small constant to avoid all-zero weights
                weights.append(0.01 + est.learning_progress())
        total = sum(weights)
        if total <= 0:
            return all_cells[self._rng.randrange(len(all_cells))]
        # multinomial sample
        u = self._rng.random() * total
        acc = 0.0
        for cell, w in zip(all_cells, weights):
            acc += w
            if u <= acc:
                return cell
        return all_cells[-1]

    def _pick_thompson_lp(self, all_cells: Sequence[Cell]) -> Cell:
        """Thompson over LP: sample posterior LP per cell, take argmax."""
        # For each cell, model recent-mean as Beta posterior; sample
        # twice (recent, prev) and take |diff|.  Argmax over cells.
        best_cell = all_cells[0]
        best_draw = -math.inf
        for cell in all_cells:
            est = self._cells.get(cell)
            if est is None or est.recent_n == 0:
                draw = self._rng.random() * 0.5  # exploratory bonus
            else:
                # Beta(s+0.5, n-s+0.5) for recent
                a_r = est.recent_successes + 0.5
                b_r = (est.recent_n - est.recent_successes) + 0.5
                a_p = est.prev_successes + 0.5
                b_p = (est.prev_n - est.prev_successes) + 0.5
                # Stdlib gammavariate to sample Beta
                mr = self._beta_sample(a_r, b_r)
                mp = self._beta_sample(a_p, b_p) if est.prev_n > 0 else mr
                draw = abs(mr - mp)
            if draw > best_draw:
                best_draw = draw
                best_cell = cell
        return best_cell

    def _beta_sample(self, a: float, b: float) -> float:
        x = self._rng.gammavariate(a, 1.0)
        y = self._rng.gammavariate(b, 1.0)
        if x + y <= 0:
            return 0.5
        return x / (x + y)

    # ------------------------------------------------------------------
    # introspection / metrics
    # ------------------------------------------------------------------

    def competence_estimate(self, theta: Sequence[float]) -> CompetenceEstimate:
        cell = self._cell_for(tuple(theta))
        return self._est_for(cell)

    def frontier(self) -> List[Cell]:
        """Cells whose CI brackets the target competence."""
        tgt = self.config.target_competence
        tol = self.config.target_tolerance
        out = []
        for cell in _enumerate_cells(self.config.n_buckets):
            est = self._cells.get(cell)
            if est is None or est.n == 0:
                continue
            lo, hi = est.ci
            if lo <= tgt + tol and hi >= tgt - tol:
                out.append(cell)
        return out

    def brier_score(self) -> Optional[float]:
        if not self._brier_buf:
            return None
        return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in self._brier_buf) / len(self._brier_buf)

    def most_competent(self) -> Optional[Tuple[Cell, float]]:
        best: Optional[Tuple[Cell, float]] = None
        for cell, est in self._cells.items():
            if est.n == 0:
                continue
            if best is None or est.mean > best[1]:
                best = (cell, est.mean)
        return best

    def n_total_cells(self) -> int:
        n = 1
        for b in self.config.n_buckets:
            n *= b
        return n

    def n_observations(self) -> int:
        return sum(est.n for est in self._cells.values())

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------

    def report(self) -> CuratorReport:
        cells_explored = sum(1 for est in self._cells.values() if est.n > 0)
        n_total = self.n_total_cells()
        frontier = self.frontier()
        means = [est.mean for est in self._cells.values() if est.n > 0]
        widths = [est.width for est in self._cells.values() if est.n > 0]
        most = self.most_competent()
        rep = CuratorReport(
            strategy=self.config.strategy,
            n_observations=self.n_observations(),
            n_cells_explored=cells_explored,
            n_cells_total=n_total,
            n_cells_in_frontier=len(frontier),
            mean_competence=(sum(means) / len(means)) if means else 0.0,
            mean_ci_width=(sum(widths) / len(widths)) if widths else 1.0,
            brier_score=self.brier_score(),
            most_competent_cell=(list(most[0].indices) if most else None),
            most_competent_value=(most[1] if most else None),
            target_competence=self.config.target_competence,
            target_tolerance=self.config.target_tolerance,
            seed=self.config.seed,
            certificate=self._chain.hexdigest(),
        )
        return rep

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def certificate(self) -> str:
        return self._chain.hexdigest()

    def cells(self) -> Dict[Cell, CompetenceEstimate]:
        return dict(self._cells)


# =============================================================================
# Free-function shortcuts
# =============================================================================


def zpd_curator(*, param_lo: Sequence[float] = (0.0,),
                param_hi: Sequence[float] = (1.0,),
                n_buckets: Sequence[int] = (8,),
                target_competence: float = 0.6,
                target_tolerance: float = 0.2,
                seed: int = 0) -> Curator:
    return Curator(CuratorConfig(
        param_lo=tuple(param_lo), param_hi=tuple(param_hi),
        n_buckets=tuple(n_buckets),
        strategy=STRATEGY_ZPD,
        target_competence=target_competence,
        target_tolerance=target_tolerance,
        seed=seed,
    ))


def learning_progress_curator(*, param_lo: Sequence[float] = (0.0,),
                              param_hi: Sequence[float] = (1.0,),
                              n_buckets: Sequence[int] = (8,),
                              target_competence: float = 0.6,
                              target_tolerance: float = 0.2,
                              recent_window: int = 8,
                              prev_window: int = 8,
                              seed: int = 0) -> Curator:
    return Curator(CuratorConfig(
        param_lo=tuple(param_lo), param_hi=tuple(param_hi),
        n_buckets=tuple(n_buckets),
        strategy=STRATEGY_LEARNING_PROGRESS,
        target_competence=target_competence,
        target_tolerance=target_tolerance,
        recent_window=recent_window,
        prev_window=prev_window,
        seed=seed,
    ))


def thompson_lp_curator(*, param_lo: Sequence[float] = (0.0,),
                        param_hi: Sequence[float] = (1.0,),
                        n_buckets: Sequence[int] = (8,),
                        target_competence: float = 0.6,
                        target_tolerance: float = 0.2,
                        recent_window: int = 8,
                        prev_window: int = 8,
                        seed: int = 0) -> Curator:
    return Curator(CuratorConfig(
        param_lo=tuple(param_lo), param_hi=tuple(param_hi),
        n_buckets=tuple(n_buckets),
        strategy=STRATEGY_THOMPSON_LP,
        target_competence=target_competence,
        target_tolerance=target_tolerance,
        recent_window=recent_window,
        prev_window=prev_window,
        seed=seed,
    ))


__all__ = [
    # errors
    "CuratorError",
    "InvalidConfig",
    "InvalidParameter",
    "InsufficientData",
    "UnknownStrategy",
    # constants
    "STRATEGY_ZPD",
    "STRATEGY_LEARNING_PROGRESS",
    "STRATEGY_THOMPSON_LP",
    "KNOWN_STRATEGIES",
    # dataclasses
    "Cell",
    "Proposal",
    "CompetenceEstimate",
    "CuratorConfig",
    "CuratorReport",
    # orchestrator
    "Curator",
    # shortcuts
    "zpd_curator",
    "learning_progress_curator",
    "thompson_lp_curator",
]
