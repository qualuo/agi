r"""Stepwiser — process-reward modelling as a runtime primitive.

The new scaling law in 2024-2026 is **inference-time compute**: spending
more tokens at test time (best-of-N, beam search, tree-of-thoughts,
self-consistency) buys problem-solving capability that no amount of
extra pre-training can.  The *unit operation* underneath every modern
flavour of that idea is a **process reward model (PRM)** — a verifier
that scores a reasoning *trajectory step by step* rather than judging
only the final answer.  PRMs are why OpenAI's o1 / o3 line, Google's
"Let's Verify Step by Step" agenda (Lightman et al. 2023), DeepMind's
process-supervised math reasoners (Uesato et al. 2022), and a wave of
open replications (Math-Shepherd, ReST-MCTS, Process-Aware-DPO) all
beat outcome-only verifiers on hard reasoning at fixed compute.

``Stepwiser`` is the runtime-level *implementation* of that primitive.
It owns the verifier model, the reward-aggregation policy, and the
search procedures (best-of-N, stepwise beam, branch-and-prune) that
convert a stream of candidate trajectories into a single accepted
trajectory with a calibrated, anytime-valid certificate.

The pitch reduced to a runtime call::

    sw = Stepwiser(StepwiserConfig(
        model=MODEL_PRM, aggregator=AGG_MIN, k_beam=4, seed=0))

    # Train: stream labelled trajectories (per-step labels are the
    # gold standard; outcome-only is supported and auto-propagated
    # back via Math-Shepherd MC rollouts).
    for traj in labelled_stream:
        sw.observe(traj)
    sw.fit()

    # Score: one trajectory or many — calibrated probabilities,
    # per-step shaping rewards, abort/continue decisions.
    score = sw.score(trajectory)
    chosen, runner_ups = sw.best_of_n(candidates)
    finals = sw.beam_search(root, expand=expand_fn, terminal=is_terminal)

    rep  = sw.report()
    cert = sw.certify()      # PAC bound on PRM accuracy + selection gap


What this primitive ships
-------------------------

  * **Reward families** — toggleable via ``StepwiserConfig.model``:

    * ``MODEL_ORM`` — outcome reward model (Cobbe et al. 2021
      *Training verifiers to solve math word problems*).  Logistic
      regression on hashed n-gram features of the *final answer*;
      maximum-likelihood with L2 and momentum, ECE-calibrated via
      Platt scaling on a held-out fold.  Cheap, high-variance.

    * ``MODEL_PRM`` — process reward model (Lightman et al. 2023
      *Let's verify step by step*; Uesato et al. 2022 *Solving math
      word problems with process- and outcome-based feedback*).
      Same featurizer, but classifier is *per-step*: features are
      the cumulative prefix's hashed n-grams, target is the gold
      per-step correctness label.  Trained as a single shared model
      with step-position as a categorical feature.  Returns a
      sequence of calibrated probabilities, one per step.

    * ``MODEL_VALUE`` — value-style verifier (Math-Shepherd, Wang
      et al. 2024).  Per-step *target* is the empirical Monte-Carlo
      probability that a random completion of the prefix reaches a
      correct outcome — that is, the prefix's *value under the
      generator*.  When per-step gold is unavailable but the
      generator can roll out continuations, this lets us synthesise
      step labels from outcome labels at the cost of K rollouts per
      prefix.  Implemented as Math-Shepherd's *hard-estimation*
      variant: ``ŷ_t = 1{ ∃ rollout from prefix_t that succeeds }``.
      Soft estimation (mean of rollout outcomes) is available via
      ``shepherd_estimation = "soft"``.

  * **Aggregators** — toggleable via ``StepwiserConfig.aggregator``:

    * ``AGG_MIN``      — ``score(τ) = min_t p_t``         (Lightman 2023)
    * ``AGG_MEAN``     — ``score(τ) = mean_t p_t``        (consensus)
    * ``AGG_PROD``     — ``score(τ) = Π_t p_t``           (chain rule)
    * ``AGG_LAST``     — ``score(τ) = p_T``               (outcome only)
    * ``AGG_LOGSUMEXP``— ``score(τ) = LSE_t(β·log p_t)/β``(soft-min)

    All five are exposed as standalone helpers ``stepwiser_*_aggregate``
    so they can be reused outside the class.

  * **Selection procedures**:

    * **Best-of-N rerank** — Cobbe 2021 / Lightman 2023 / Brown 2024
      *Large Language Monkeys: Scaling Inference Compute with Repeated
      Sampling*.  Given ``N`` candidate trajectories, return the one
      with the highest aggregated score.  Returns the winner, a
      ranked list of runner-ups, and an *empirical PRM-selection
      gap* — the margin between rank-1 and rank-2 score, an
      indicator of selection confidence.

    * **Stepwise beam search** — Yao et al. 2023 *Tree of Thoughts*;
      Xie et al. 2023 *Self-Evaluation Guided Beam Search*.  At each
      depth, expand all live beams via ``expand(beam)`` (caller-
      supplied), score each one-step continuation, retain top-``B``
      by PRM aggregated score, recurse.  Terminates when every beam
      satisfies the caller's ``is_terminal``.  ``branch_factor`` and
      ``max_depth`` guard runaway expansion.

    * **Stepwise quantilization** — Snell et al. 2024 *Scaling LLM
      test-time compute optimally*.  Only retain steps whose
      PRM score exceeds the *q-th quantile* of historical scores.
      Behaves as an anti-tail-risk filter: agents never pursue
      reasoning steps the PRM rates worse than ``q`` of training
      data.  Compatible with ``Quantilizer``-style guarantees.

    * **MCTS-PRM rollouts** — ReST-MCTS-style backups (Zhang et al.
      2024).  Optional helper ``stepwiser_mcts_backup`` runs a
      caller-supplied expansion / simulation oracle and produces
      PRM-weighted Q-value backups; integrates with ``Searcher``.

  * **Step-level shaping rewards** — potential-based shaping (Ng,
    Harada, Russell 1999).  Given per-step PRM scores ``V(s_t)``,
    emits ``r̃_t = γ · V(s_{t+1}) − V(s_t)``.  Policy-invariance
    guaranteed: any RL primitive trained on shaped reward converges
    to the same optimal policy as the unshaped reward, but with
    lower variance.  Drop-in for ``Aligner`` / ``Pareto`` / RL
    learners that consume scalar rewards.

  * **Probability calibration** — Platt scaling + isotonic regression.
    Inference-time PRM outputs are passed through a logistic
    transform fit on a held-out fold (Platt 1999), then an isotonic
    monotone fit (Zadrozny & Elkan 2002).  Returns the expected
    calibration error (ECE) on the held-out fold; aborts deployment
    if ``ECE > config.max_ece``.  Calibration is essential for the
    quantilization gate and for the certify() bound.

  * **Confidence intervals** — Hoeffding (1963) and empirical-
    Bernstein (Maurer & Pontil 2009) bounds on PRM accuracy.  For
    selection, returns an *anytime-valid* lower confidence bound on
    ``P(selected = best)`` derived from the binomial concentration
    around the per-trajectory pairwise win-rate.  No assumption on
    underlying distribution beyond bounded ``[0,1]`` scores.

  * **Drift detection** — two-sample Kolmogorov-Smirnov on the
    distribution of *prefix-feature counts* between training and
    streaming inputs.  Surfaces "the kind of reasoning we're
    verifying now is unlike anything we trained on", a strong
    indicator that the calibration bound no longer holds.

  * **Skill mining** — surfaces the K most predictive step-prefix
    patterns from the trained PRM weights (top-magnitude features
    in the per-step classifier).  These are *interpretable
    artefacts* a coordinator can promote into ``Skillmine`` /
    ``Skills``.

  * **Replay-verifiable receipts** — SHA-256 fingerprint chain
    (optionally HMAC'd) over every observation, fit, score, search,
    calibration, and certify call.  ``stepwiser_ledger_root`` is the
    immutable genesis.  Replaying the chain reproduces every state
    transition byte-for-byte.

  * **Thread-safe re-entrant lock**; transport-agnostic;
    pure stdlib (no NumPy, no Torch); deterministic given seed.


Composes with
-------------

  * ``Verifier`` — outcome-only verifier; ``Stepwiser`` *generalises*
    it to per-step.  ``Verifier`` becomes the supervision signal
    for ``Math-Shepherd`` style training when per-step labels are
    unavailable.
  * ``Speculator`` — token-level acceleration with provable
    equivalence.  ``Stepwiser`` operates one layer up (steps, not
    tokens) and provides the *acceptance signal* for plan-level
    speculation: the cheap draft executor proposes a plan,
    ``Stepwiser`` accepts/rejects per step.
  * ``Searcher`` / ``Sketcher`` — consume PRM scores as the heuristic
    in best-first / beam search.  ``stepwiser_mcts_backup`` is a
    PRM-aware backup for ``Searcher``'s rollout policy.
  * ``Quantilizer`` — wraps ``Stepwiser.quantilize_step`` to refuse
    out-of-distribution reasoning prefixes.
  * ``Aligner`` — shaped reward channel for RL fine-tuning;
    ``stepwiser_potential_shaping`` is a drop-in for ``Aligner``'s
    auxiliary reward.
  * ``Empowerer`` — empowerment + PRM = "verify each step both
    *correct* (PRM) and *agency-preserving* (Empowerer)" — a useful
    double-check for safety-critical deployments.
  * ``Coordinator`` / ``Driver`` — read ``Stepwiser.certify()`` to
    decide *act vs ask vs branch vs defer*.  The selection-gap
    bound is the natural deferral signal: when rank-1 and rank-2
    are within ``2δ``, the coordinator escalates to a more
    expensive verifier rather than committing.


Mathematical notation
---------------------

  * ``τ = (s_1, …, s_T)``       — a reasoning trajectory; each ``s_t``
    is a step (free-form text).
  * ``y_t ∈ {0,1}``              — gold per-step correctness label.
  * ``p_t = P(y_t = 1 | s_{1:t})`` — PRM prediction at step ``t``.
  * ``A(τ) ∈ ℝ``                 — aggregated score under
    :class:`StepwiserConfig.aggregator`.
  * ``Δ(τ) = A(τ_*) − A(τ_{(2)})`` — selection gap.
  * ``γ ∈ [0,1)``                — shaping discount.

All ingest paths are validated.  Per-step inference is
``O(|features|)``; training is ``O(n_examples · epochs · |features|)``;
calibration is ``O(n_holdout · log n_holdout)``.  No ``random`` without
explicit seed; no ``time.time()`` leaks into the chain.

References
----------

  * Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek,
    Hilton, Nakano, Hesse, Schulman 2021. *Training verifiers to
    solve math word problems.* arXiv:2110.14168.
  * Uesato, Kushman, Kumar, Song, Siegel, Wang, Creswell, Irving,
    Higgins 2022. *Solving math word problems with process- and
    outcome-based feedback.* arXiv:2211.14275.
  * Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike,
    Schulman, Sutskever, Cobbe 2023. *Let's verify step by step.*
    arXiv:2305.20050.
  * Wang, Li, Shao, Xu, Dai, Li, Wang, Zhu 2024. *Math-Shepherd:
    Verify and reinforce LLMs step-by-step without human
    annotations.* ACL 2024.
  * Zhang, Wu, Yao, Yang, Yan, Zhang, Liu, Wei, Wang, Wang 2024.
    *ReST-MCTS\*: LLM self-training via process reward guided tree
    search.* NeurIPS.
  * Brown, Juravsky, Ehrlich, Clark, Le, Rä, Mirhoseini 2024.
    *Large language monkeys: Scaling inference compute with
    repeated sampling.* arXiv:2407.21787.
  * Snell, Lee, Xu, Kumar 2024. *Scaling LLM test-time compute
    optimally can be more effective than scaling model parameters.*
    arXiv:2408.03314.
  * Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan 2023. *Tree
    of thoughts: Deliberate problem solving with large language
    models.* NeurIPS.
  * Xie, Kawaguchi, Zhao, Zhao, Wei, Kan, Zhang, He 2023.
    *Self-evaluation guided beam search for reasoning.* NeurIPS.
  * Ng, Harada, Russell 1999. *Policy invariance under reward
    transformations: Theory and application to reward shaping.*
    ICML.
  * Platt 1999. *Probabilistic outputs for support vector machines
    and comparisons to regularized likelihood methods.* Adv. Large
    Margin Classifiers.
  * Zadrozny, Elkan 2002. *Transforming classifier scores into
    accurate multiclass probability estimates.* KDD.
  * Hoeffding 1963. *Probability inequalities for sums of bounded
    random variables.* JASA 58.
  * Maurer, Pontil 2009. *Empirical Bernstein bounds and sample
    variance penalisation.* COLT.
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
    "STEPWISER_STARTED",
    "STEPWISER_OBSERVED",
    "STEPWISER_FITTED",
    "STEPWISER_CALIBRATED",
    "STEPWISER_SCORED",
    "STEPWISER_SELECTED",
    "STEPWISER_SEARCHED",
    "STEPWISER_SHAPED",
    "STEPWISER_QUANTILIZED",
    "STEPWISER_DRIFT_DETECTED",
    "STEPWISER_REPORTED",
    "STEPWISER_CERTIFIED",
    "STEPWISER_RESET",
    # Model codes
    "MODEL_ORM",
    "MODEL_PRM",
    "MODEL_VALUE",
    "KNOWN_MODELS",
    # Aggregators
    "AGG_MIN",
    "AGG_MEAN",
    "AGG_PROD",
    "AGG_LAST",
    "AGG_LOGSUMEXP",
    "KNOWN_AGGREGATORS",
    # Calibrators
    "CAL_NONE",
    "CAL_PLATT",
    "CAL_ISOTONIC",
    "KNOWN_CALIBRATORS",
    # Shepherd
    "SHEPHERD_HARD",
    "SHEPHERD_SOFT",
    "KNOWN_SHEPHERD",
    # Exceptions
    "StepwiserError",
    "InvalidConfig",
    "InvalidTrajectory",
    "InvalidStep",
    "InsufficientData",
    "UnknownModel",
    "UnknownAggregator",
    "UnknownCalibrator",
    "NotFitted",
    "DriftDetected",
    # Dataclasses
    "StepwiserConfig",
    "TrajectoryRecord",
    "StepScore",
    "TrajectoryScore",
    "SelectionResult",
    "BeamResult",
    "CalibrationResult",
    "DriftResult",
    "StepwiserReport",
    "StepwiserCertificate",
    # Helpers
    "stepwiser_ledger_root",
    "stepwiser_min_aggregate",
    "stepwiser_mean_aggregate",
    "stepwiser_prod_aggregate",
    "stepwiser_last_aggregate",
    "stepwiser_logsumexp_aggregate",
    "stepwiser_potential_shaping",
    "stepwiser_mcts_backup",
    "hash_step_features",
    "platt_fit",
    "platt_apply",
    "isotonic_fit",
    "isotonic_apply",
    "expected_calibration_error",
    "ks_two_sample",
    # Main class
    "Stepwiser",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

STEPWISER_STARTED = "stepwiser.started"
STEPWISER_OBSERVED = "stepwiser.observed"
STEPWISER_FITTED = "stepwiser.fitted"
STEPWISER_CALIBRATED = "stepwiser.calibrated"
STEPWISER_SCORED = "stepwiser.scored"
STEPWISER_SELECTED = "stepwiser.selected"
STEPWISER_SEARCHED = "stepwiser.searched"
STEPWISER_SHAPED = "stepwiser.shaped"
STEPWISER_QUANTILIZED = "stepwiser.quantilized"
STEPWISER_DRIFT_DETECTED = "stepwiser.drift_detected"
STEPWISER_REPORTED = "stepwiser.reported"
STEPWISER_CERTIFIED = "stepwiser.certified"
STEPWISER_RESET = "stepwiser.reset"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

MODEL_ORM = "orm"
MODEL_PRM = "prm"
MODEL_VALUE = "value"
KNOWN_MODELS = (MODEL_ORM, MODEL_PRM, MODEL_VALUE)

AGG_MIN = "min"
AGG_MEAN = "mean"
AGG_PROD = "prod"
AGG_LAST = "last"
AGG_LOGSUMEXP = "logsumexp"
KNOWN_AGGREGATORS = (AGG_MIN, AGG_MEAN, AGG_PROD, AGG_LAST, AGG_LOGSUMEXP)

CAL_NONE = "none"
CAL_PLATT = "platt"
CAL_ISOTONIC = "isotonic"
KNOWN_CALIBRATORS = (CAL_NONE, CAL_PLATT, CAL_ISOTONIC)

SHEPHERD_HARD = "hard"
SHEPHERD_SOFT = "soft"
KNOWN_SHEPHERD = (SHEPHERD_HARD, SHEPHERD_SOFT)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StepwiserError(Exception):
    """Base class for all :mod:`agi.stepwiser` errors."""


class InvalidConfig(StepwiserError):
    """A :class:`StepwiserConfig` field is out of range."""


class InvalidTrajectory(StepwiserError):
    """A submitted trajectory is malformed."""


class InvalidStep(StepwiserError):
    """A step is malformed (not a string, empty, etc.)."""


class InsufficientData(StepwiserError):
    """An operation requires more data than has been observed."""


class UnknownModel(StepwiserError):
    """``model`` is not in :data:`KNOWN_MODELS`."""


class UnknownAggregator(StepwiserError):
    """``aggregator`` is not in :data:`KNOWN_AGGREGATORS`."""


class UnknownCalibrator(StepwiserError):
    """``calibrator`` is not in :data:`KNOWN_CALIBRATORS`."""


class NotFitted(StepwiserError):
    """:meth:`Stepwiser.score` called before :meth:`Stepwiser.fit`."""


class DriftDetected(StepwiserError):
    """Two-sample KS rejected the null at ``config.drift_alpha``."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepwiserConfig:
    """Configuration for :class:`Stepwiser`.

    Parameters
    ----------
    model : str
        One of :data:`KNOWN_MODELS`.  ``MODEL_ORM`` for outcome-only,
        ``MODEL_PRM`` for per-step gold labels, ``MODEL_VALUE`` for
        Math-Shepherd MC rollouts.
    aggregator : str
        How :meth:`Stepwiser.score` combines per-step probabilities
        into a trajectory score.  One of :data:`KNOWN_AGGREGATORS`.
    calibrator : str
        One of :data:`KNOWN_CALIBRATORS`.  Calibration is applied
        after :meth:`Stepwiser.fit` on the held-out fold.
    shepherd_estimation : str
        ``"hard"`` (Math-Shepherd default) or ``"soft"`` MC.
    feature_dim : int
        Hashed-feature dimension.  Larger reduces collisions; smaller
        is faster.  Power-of-two recommended.
    ngram_n : int
        Maximum n-gram length on token-split steps.  Inclusive of
        unigrams up to ``ngram_n``.
    include_step_index : bool
        Append a categorical "step index" feature.  Lets the model
        condition on position.
    learning_rate : float
        SGD step on logistic regression.  L2 regularised.
    l2 : float
        L2 weight decay.
    epochs : int
        Number of full passes over the training data.
    momentum : float
        Heavy-ball momentum coefficient.
    holdout_fraction : float
        Fraction of training data held out for calibration / ECE.
    max_ece : float
        Hard ceiling on expected calibration error.  Exceeded ECE
        raises :class:`InvalidConfig` at calibration time.
    k_beam : int
        Beam width for :meth:`Stepwiser.beam_search`.
    branch_factor : int
        Maximum continuations expanded per beam per depth.
    max_depth : int
        Hard cap on beam-search depth.
    quantile : float
        Threshold ``q ∈ [0,1]`` for :meth:`Stepwiser.quantilize_step`.
    discount : float
        Shaping discount ``γ`` in
        :func:`stepwiser_potential_shaping`.
    logsumexp_beta : float
        Inverse temperature for ``AGG_LOGSUMEXP``.
    drift_alpha : float
        Significance level for the KS drift test.
    confidence : float
        ``1 − δ`` used by :meth:`Stepwiser.certify`.
    rng_seed : int
        Deterministic RNG seed for SGD shuffling and rollouts.
    hmac_key : bytes | None
        Optional HMAC key for the ledger chain.
    """

    model: str = MODEL_PRM
    aggregator: str = AGG_MIN
    calibrator: str = CAL_PLATT
    shepherd_estimation: str = SHEPHERD_HARD
    feature_dim: int = 4096
    ngram_n: int = 2
    include_step_index: bool = True
    learning_rate: float = 0.1
    l2: float = 1e-3
    epochs: int = 16
    momentum: float = 0.9
    holdout_fraction: float = 0.2
    max_ece: float = 0.25
    k_beam: int = 4
    branch_factor: int = 4
    max_depth: int = 32
    quantile: float = 0.5
    discount: float = 0.99
    logsumexp_beta: float = 4.0
    drift_alpha: float = 0.05
    confidence: float = 0.95
    rng_seed: int = 0
    hmac_key: bytes | None = None

    def __post_init__(self) -> None:
        if self.model not in KNOWN_MODELS:
            raise UnknownModel(self.model)
        if self.aggregator not in KNOWN_AGGREGATORS:
            raise UnknownAggregator(self.aggregator)
        if self.calibrator not in KNOWN_CALIBRATORS:
            raise UnknownCalibrator(self.calibrator)
        if self.shepherd_estimation not in KNOWN_SHEPHERD:
            raise InvalidConfig(
                f"shepherd_estimation must be one of {KNOWN_SHEPHERD}"
            )
        if not isinstance(self.feature_dim, int) or self.feature_dim < 8:
            raise InvalidConfig("feature_dim must be int ≥ 8")
        if not isinstance(self.ngram_n, int) or self.ngram_n < 1:
            raise InvalidConfig("ngram_n must be int ≥ 1")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise InvalidConfig("learning_rate must be > 0 and finite")
        if self.l2 < 0.0 or not math.isfinite(self.l2):
            raise InvalidConfig("l2 must be ≥ 0 and finite")
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise InvalidConfig("epochs must be int ≥ 1")
        if not (0.0 <= self.momentum < 1.0):
            raise InvalidConfig("momentum must be in [0, 1)")
        if not (0.0 < self.holdout_fraction < 1.0):
            raise InvalidConfig("holdout_fraction must be in (0, 1)")
        if not (0.0 <= self.max_ece <= 1.0):
            raise InvalidConfig("max_ece must be in [0, 1]")
        if not isinstance(self.k_beam, int) or self.k_beam < 1:
            raise InvalidConfig("k_beam must be int ≥ 1")
        if not isinstance(self.branch_factor, int) or self.branch_factor < 1:
            raise InvalidConfig("branch_factor must be int ≥ 1")
        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise InvalidConfig("max_depth must be int ≥ 1")
        if not (0.0 <= self.quantile <= 1.0):
            raise InvalidConfig("quantile must be in [0, 1]")
        if not (0.0 <= self.discount <= 1.0):
            raise InvalidConfig("discount must be in [0, 1]")
        if self.logsumexp_beta <= 0.0:
            raise InvalidConfig("logsumexp_beta must be > 0")
        if not (0.0 < self.drift_alpha < 1.0):
            raise InvalidConfig("drift_alpha must be in (0, 1)")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must be in (0, 1)")
        if not isinstance(self.rng_seed, int):
            raise InvalidConfig("rng_seed must be int")
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes or None")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryRecord:
    """A single labelled trajectory used for training."""

    steps: tuple[str, ...]
    step_labels: tuple[int, ...] | None = None  # per-step gold (PRM)
    outcome: int | None = None                  # 0/1 final correctness (ORM)
    weight: float = 1.0


@dataclass(frozen=True)
class StepScore:
    """Per-step PRM output."""

    step_index: int
    raw_logit: float
    raw_prob: float       # sigmoid(raw_logit)
    calibrated_prob: float


@dataclass(frozen=True)
class TrajectoryScore:
    """Aggregated trajectory score."""

    aggregated: float
    aggregator: str
    per_step: tuple[StepScore, ...]


@dataclass(frozen=True)
class SelectionResult:
    """Output of :meth:`Stepwiser.best_of_n`."""

    chosen_index: int
    chosen_score: TrajectoryScore
    ranked: tuple[tuple[int, float], ...]
    selection_gap: float  # rank-1 minus rank-2 aggregated score
    selection_gap_lcb: float  # anytime-valid LCB on the gap


@dataclass(frozen=True)
class BeamResult:
    """Output of :meth:`Stepwiser.beam_search`."""

    beams: tuple[tuple[str, ...], ...]
    scores: tuple[float, ...]
    depth: int
    expansions: int


@dataclass(frozen=True)
class CalibrationResult:
    """Output of :meth:`Stepwiser.fit` calibration phase."""

    method: str
    ece_before: float
    ece_after: float
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class DriftResult:
    """Output of :meth:`Stepwiser.drift`."""

    ks_statistic: float
    threshold: float
    rejected: bool
    n_train: int
    n_stream: int


@dataclass(frozen=True)
class StepwiserReport:
    """Self-describing report dataclass."""

    config: StepwiserConfig
    n_trajectories_observed: int
    n_steps_observed: int
    fitted: bool
    calibrated: bool
    train_accuracy: float
    holdout_accuracy: float
    holdout_ece: float
    n_scored: int
    n_selected: int
    n_beam_searched: int
    last_selection_gap: float
    last_aggregator: str
    chain_head: str


@dataclass(frozen=True)
class StepwiserCertificate:
    """PAC certificate from :meth:`Stepwiser.certify`."""

    confidence: float
    n_holdout: int
    holdout_accuracy: float
    hoeffding_half_width: float
    bernstein_half_width: float
    accuracy_lcb: float
    last_selection_gap: float
    last_selection_gap_lcb: float
    chain_head: str
    fingerprint_hash: str


# ---------------------------------------------------------------------------
# Helpers (free functions)
# ---------------------------------------------------------------------------


GENESIS = "agi.stepwiser.v1"


def stepwiser_ledger_root() -> str:
    """Return the genesis fingerprint for the Stepwiser chain."""
    return hashlib.sha256(GENESIS.encode("utf-8")).hexdigest()


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, bytes):
        return o.hex()
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def _digest(prev: str, payload: Mapping[str, Any], hmac_key: bytes | None) -> str:
    body = _canonical_bytes({"prev": prev, "payload": payload})
    if hmac_key is None:
        return hashlib.sha256(body).hexdigest()
    return hmac.new(hmac_key, body, hashlib.sha256).hexdigest()


def _stable_token_hash(token: str, dim: int) -> int:
    """Stable, deterministic hash into ``[0, dim)``.  No Python ``hash``."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % dim


def hash_step_features(
    steps: Sequence[str],
    *,
    feature_dim: int,
    ngram_n: int = 2,
    step_index: int | None = None,
    include_step_index: bool = True,
) -> dict[int, float]:
    """Hashed-feature map for the prefix ``steps[: step_index+1]``.

    Tokenisation: whitespace + punctuation strip.  N-gram lengths 1..n.
    Optional categorical step-index feature appended.

    Returns a sparse ``{feature_id: count}`` dict.
    """
    if feature_dim < 8:
        raise InvalidConfig("feature_dim must be ≥ 8")
    if ngram_n < 1:
        raise InvalidConfig("ngram_n must be ≥ 1")
    feats: dict[int, float] = {}
    # Bias feature
    feats[0] = 1.0
    # Tokenize the whole prefix.
    tokens: list[str] = []
    for s in steps:
        if not isinstance(s, str):
            raise InvalidStep("steps must be str")
        toks = _simple_tokenize(s)
        tokens.extend(toks)
        tokens.append("<|step|>")
    for n in range(1, ngram_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            fid = 1 + _stable_token_hash(f"{n}:{ngram}", feature_dim - 2)
            feats[fid] = feats.get(fid, 0.0) + 1.0
    if include_step_index and step_index is not None:
        # Map step-index to a dedicated slot in [feature_dim-1].
        fid = feature_dim - 1
        feats[fid] = feats.get(fid, 0.0) + float(step_index + 1)
    return feats


def _simple_tokenize(s: str) -> list[str]:
    if not s:
        return []
    out: list[str] = []
    cur: list[str] = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch.lower())
        else:
            if cur:
                out.append("".join(cur))
                cur = []
            if not ch.isspace():
                out.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _sigmoid(z: float) -> float:
    # Numerically stable.
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _dot_sparse(weights: list[float], feats: Mapping[int, float]) -> float:
    out = 0.0
    for fid, v in feats.items():
        if 0 <= fid < len(weights):
            out += weights[fid] * v
    return out


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# Aggregators --------------------------------------------------------


def stepwiser_min_aggregate(probs: Sequence[float]) -> float:
    """``min_t p_t`` — bottleneck-step PRM score."""
    if not probs:
        return 0.0
    return min(probs)


def stepwiser_mean_aggregate(probs: Sequence[float]) -> float:
    """``mean_t p_t``."""
    if not probs:
        return 0.0
    return sum(probs) / len(probs)


def stepwiser_prod_aggregate(probs: Sequence[float]) -> float:
    """``Π_t p_t`` — chain-rule joint probability of stepwise correctness."""
    if not probs:
        return 0.0
    acc = 1.0
    for p in probs:
        acc *= max(min(p, 1.0), 0.0)
    return acc


def stepwiser_last_aggregate(probs: Sequence[float]) -> float:
    """``p_T`` — outcome-only proxy."""
    if not probs:
        return 0.0
    return probs[-1]


def stepwiser_logsumexp_aggregate(
    probs: Sequence[float], *, beta: float = 4.0
) -> float:
    """Soft-min via ``-LSE(-β log p) / β``.

    As ``β → ∞`` recovers ``stepwiser_min_aggregate``.  Differentiable
    everywhere on ``(0, 1]``; values clamped to a tiny epsilon to
    avoid log(0).
    """
    if not probs:
        return 0.0
    if beta <= 0.0:
        raise InvalidConfig("beta must be > 0")
    eps = 1e-12
    neg_log = [-math.log(max(p, eps)) for p in probs]
    # Soft-max of (-log p) → soft-min of log p.
    m = max(beta * x for x in neg_log)
    s = sum(math.exp(beta * x - m) for x in neg_log)
    smax = (m + math.log(s)) / beta
    return math.exp(-smax)


# Calibration --------------------------------------------------------


def platt_fit(
    logits: Sequence[float], labels: Sequence[int],
    *, max_iter: int = 100, lr: float = 0.1,
) -> tuple[float, float]:
    """Platt scaling fit: ``y ≈ σ(A·z + B)``.

    Newton-style gradient descent on log-likelihood.  Returns ``(A, B)``.
    """
    if len(logits) != len(labels):
        raise InvalidConfig("platt_fit: length mismatch")
    if not logits:
        return 1.0, 0.0
    A, B = 1.0, 0.0
    for _ in range(max_iter):
        gA, gB = 0.0, 0.0
        for z, y in zip(logits, labels):
            p = _sigmoid(A * z + B)
            err = p - float(y)
            gA += err * z
            gB += err
        A -= lr * gA / len(logits)
        B -= lr * gB / len(logits)
    return A, B


def platt_apply(logits: Sequence[float], A: float, B: float) -> list[float]:
    """Apply Platt parameters to a sequence of logits."""
    return [_sigmoid(A * z + B) for z in logits]


def isotonic_fit(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pool-adjacent-violators isotonic regression.

    Returns ``(thresholds, values)`` so that
    ``isotonic_apply(s, thresholds, values)`` is monotone in ``s``.
    """
    if len(scores) != len(labels):
        raise InvalidConfig("isotonic_fit: length mismatch")
    if not scores:
        return tuple(), tuple()
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(labels[i]) for i in order]
    # Initialise blocks.
    block_x = list(xs)
    block_y = list(ys)
    block_w = [1.0] * len(xs)
    # Pool adjacent violators.
    i = 0
    while i < len(block_y) - 1:
        if block_y[i] > block_y[i + 1]:
            # Merge.
            w = block_w[i] + block_w[i + 1]
            y_merged = (block_y[i] * block_w[i] + block_y[i + 1] * block_w[i + 1]) / w
            x_merged = (block_x[i] * block_w[i] + block_x[i + 1] * block_w[i + 1]) / w
            block_y[i] = y_merged
            block_x[i] = x_merged
            block_w[i] = w
            del block_y[i + 1]
            del block_x[i + 1]
            del block_w[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return tuple(block_x), tuple(block_y)


def isotonic_apply(
    s: float, thresholds: Sequence[float], values: Sequence[float]
) -> float:
    """Apply isotonic regression: piecewise-constant interpolation."""
    if not thresholds:
        return s
    if s <= thresholds[0]:
        return values[0]
    if s >= thresholds[-1]:
        return values[-1]
    # Binary search.
    lo, hi = 0, len(thresholds) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if thresholds[mid] <= s:
            lo = mid
        else:
            hi = mid
    return values[lo]


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> float:
    """Standard ECE (Guo et al. 2017).  Uniform-width bins."""
    if len(probs) != len(labels):
        raise InvalidConfig("ECE: length mismatch")
    if not probs:
        return 0.0
    n = len(probs)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((p, int(y)))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        mean_p = sum(x[0] for x in b) / len(b)
        mean_y = sum(x[1] for x in b) / len(b)
        ece += (len(b) / n) * abs(mean_p - mean_y)
    return ece


def ks_two_sample(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic.

    Returns sup_x |F_a(x) - F_b(x)| in [0,1].
    """
    if not a or not b:
        return 0.0
    xs = sorted(set(list(a) + list(b)))
    sa = sorted(a)
    sb = sorted(b)
    ka, kb = 0, 0
    na, nb = len(sa), len(sb)
    best = 0.0
    for x in xs:
        while ka < na and sa[ka] <= x:
            ka += 1
        while kb < nb and sb[kb] <= x:
            kb += 1
        diff = abs(ka / na - kb / nb)
        if diff > best:
            best = diff
    return best


# Reward shaping ----------------------------------------------------


def stepwiser_potential_shaping(
    values: Sequence[float], *, discount: float = 0.99
) -> list[float]:
    """Potential-based shaping (Ng-Harada-Russell 1999).

    Given per-step values ``V(s_t)`` (e.g. PRM probabilities), returns
    shaped rewards ``r̃_t = γ V(s_{t+1}) − V(s_t)``.  Last step uses
    terminal potential of zero.
    """
    if not (0.0 <= discount <= 1.0):
        raise InvalidConfig("discount must be in [0, 1]")
    n = len(values)
    out = [0.0] * n
    for t in range(n):
        next_v = values[t + 1] if t + 1 < n else 0.0
        out[t] = discount * next_v - values[t]
    return out


# MCTS backup -------------------------------------------------------


def stepwiser_mcts_backup(
    leaf_values: Sequence[float], visit_counts: Sequence[int],
    *, alpha: float = 0.5,
) -> list[float]:
    """ReST-MCTS-style PRM-weighted Q-value backup.

    Q̂_i = α · V_i + (1−α) · sum(n_j V_j) / sum(n_j)

    Blends per-leaf PRM value with visit-weighted children mean.  Used
    inside ``Searcher`` rollouts when leaves are evaluated by PRM.
    """
    if len(leaf_values) != len(visit_counts):
        raise InvalidConfig("backup: length mismatch")
    if not (0.0 <= alpha <= 1.0):
        raise InvalidConfig("alpha must be in [0, 1]")
    if not leaf_values:
        return []
    total = sum(visit_counts)
    if total <= 0:
        weighted = sum(leaf_values) / len(leaf_values)
    else:
        weighted = sum(v * n for v, n in zip(leaf_values, visit_counts)) / total
    return [alpha * v + (1.0 - alpha) * weighted for v in leaf_values]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def _hoeffding_half_width(n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def _empirical_variance(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _bernstein_half_width(n: int, variance: float, alpha: float,
                          range_: float = 1.0) -> float:
    if n <= 1:
        return range_
    return (math.sqrt(2.0 * variance * math.log(3.0 / alpha) / n)
            + 3.0 * range_ * math.log(3.0 / alpha) / n)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Stepwiser:
    """Process-reward modelling, calibration, and verifier-guided search.

    See module docstring for the conceptual pitch and references.
    """

    def __init__(self, config: StepwiserConfig | None = None) -> None:
        self._config = config or StepwiserConfig()
        self._lock = threading.RLock()
        self._rng = random.Random(self._config.rng_seed)

        # Storage of observed trajectories (immutable copies).
        self._trajectories: list[TrajectoryRecord] = []

        # Training-derived state.
        self._weights: list[float] = [0.0] * self._config.feature_dim
        self._velocity: list[float] = [0.0] * self._config.feature_dim
        self._fitted = False
        self._train_accuracy = 0.0
        self._holdout_accuracy = 0.0
        self._holdout_ece_before = 0.0
        self._holdout_ece_after = 0.0
        self._holdout_logits: list[float] = []
        self._holdout_labels: list[int] = []

        # Calibration state.
        self._calibrated = False
        self._platt_A = 1.0
        self._platt_B = 0.0
        self._isotonic_thresh: tuple[float, ...] = tuple()
        self._isotonic_vals: tuple[float, ...] = tuple()

        # Counters.
        self._n_scored = 0
        self._n_selected = 0
        self._n_beam_searched = 0
        self._last_gap = 0.0
        self._last_gap_lcb = 0.0
        self._last_aggregator = self._config.aggregator

        # Training-distribution features (for drift detection).
        self._train_feature_density: list[float] = []

        # Ledger chain.
        self._chain_head = stepwiser_ledger_root()
        self._chain_history: list[tuple[str, str]] = []  # (kind, head)

        self._append("started", {
            "config_hash": self._config_hash(),
        })

    # -- Config / introspection -----------------------------------------

    @property
    def config(self) -> StepwiserConfig:
        return self._config

    @property
    def chain_head(self) -> str:
        return self._chain_head

    def _config_hash(self) -> str:
        return hashlib.sha256(
            _canonical_bytes({
                "model": self._config.model,
                "aggregator": self._config.aggregator,
                "calibrator": self._config.calibrator,
                "feature_dim": self._config.feature_dim,
                "ngram_n": self._config.ngram_n,
                "learning_rate": self._config.learning_rate,
                "l2": self._config.l2,
                "epochs": self._config.epochs,
                "momentum": self._config.momentum,
                "holdout_fraction": self._config.holdout_fraction,
                "rng_seed": self._config.rng_seed,
                "include_step_index": self._config.include_step_index,
            })
        ).hexdigest()

    def _append(self, kind: str, payload: Mapping[str, Any]) -> str:
        h = _digest(self._chain_head, payload, self._config.hmac_key)
        self._chain_head = h
        self._chain_history.append((kind, h))
        return h

    # -- Ingest ---------------------------------------------------------

    def observe(self, trajectory: TrajectoryRecord) -> None:
        """Append a labelled trajectory to the training pool."""
        with self._lock:
            self._validate_trajectory(trajectory)
            self._trajectories.append(trajectory)
            self._append("observed", {
                "n_steps": len(trajectory.steps),
                "has_step_labels": trajectory.step_labels is not None,
                "outcome": trajectory.outcome,
                "weight": trajectory.weight,
            })

    def observe_many(self, trajectories: Iterable[TrajectoryRecord]) -> int:
        n = 0
        for t in trajectories:
            self.observe(t)
            n += 1
        return n

    def _validate_trajectory(self, t: TrajectoryRecord) -> None:
        if not isinstance(t, TrajectoryRecord):
            raise InvalidTrajectory("must be TrajectoryRecord")
        if not t.steps:
            raise InvalidTrajectory("trajectory has zero steps")
        for s in t.steps:
            if not isinstance(s, str):
                raise InvalidStep("steps must be str")
        if t.step_labels is not None:
            if len(t.step_labels) != len(t.steps):
                raise InvalidTrajectory("step_labels length mismatch")
            for y in t.step_labels:
                if y not in (0, 1):
                    raise InvalidTrajectory("step_labels must be 0/1")
        if t.outcome is not None and t.outcome not in (0, 1):
            raise InvalidTrajectory("outcome must be 0/1")
        if t.weight <= 0.0 or not math.isfinite(t.weight):
            raise InvalidTrajectory("weight must be > 0 and finite")
        if self._config.model == MODEL_PRM and t.step_labels is None:
            # PRM training requires per-step labels unless outcome can
            # be back-propagated; we allow it but mark for outcome-fallback.
            if t.outcome is None:
                raise InvalidTrajectory(
                    "MODEL_PRM requires step_labels or outcome (Math-Shepherd)"
                )

    # -- Training -------------------------------------------------------

    def fit(self) -> StepwiserReport:
        """Fit the verifier on the observed trajectories."""
        with self._lock:
            if not self._trajectories:
                raise InsufficientData("no trajectories observed yet")
            examples = self._build_training_set()
            if not examples:
                raise InsufficientData("training set is empty post-build")
            train, holdout = self._split_train_holdout(examples)
            self._train_sgd(train)
            self._evaluate(train, holdout)
            self._update_train_feature_density(examples)
            self._fitted = True
            self._append("fitted", {
                "n_train": len(train),
                "n_holdout": len(holdout),
                "train_accuracy": self._train_accuracy,
                "holdout_accuracy": self._holdout_accuracy,
            })
            if self._config.calibrator != CAL_NONE:
                self._calibrate()
            return self.report()

    def _build_training_set(self) -> list[tuple[Mapping[int, float], int, float]]:
        """Convert trajectories to (features, label, weight) tuples."""
        out: list[tuple[Mapping[int, float], int, float]] = []
        cfg = self._config
        for t in self._trajectories:
            if cfg.model == MODEL_ORM:
                # One example per trajectory: features over all steps,
                # label = outcome.
                if t.outcome is None:
                    if t.step_labels is None:
                        continue
                    # Treat last step label as outcome.
                    y = int(t.step_labels[-1])
                else:
                    y = int(t.outcome)
                feats = hash_step_features(
                    t.steps,
                    feature_dim=cfg.feature_dim,
                    ngram_n=cfg.ngram_n,
                    step_index=len(t.steps) - 1,
                    include_step_index=cfg.include_step_index,
                )
                out.append((feats, y, t.weight))
            elif cfg.model == MODEL_PRM:
                # One example per (trajectory, step).  If per-step
                # labels exist, use them; else Math-Shepherd propagation:
                # all prefixes get the outcome label.  (Hard estimation.)
                if t.step_labels is not None:
                    labels = t.step_labels
                else:
                    labels = tuple([int(t.outcome)] * len(t.steps))
                for i in range(len(t.steps)):
                    feats = hash_step_features(
                        t.steps[: i + 1],
                        feature_dim=cfg.feature_dim,
                        ngram_n=cfg.ngram_n,
                        step_index=i,
                        include_step_index=cfg.include_step_index,
                    )
                    out.append((feats, int(labels[i]), t.weight))
            elif cfg.model == MODEL_VALUE:
                # Value-style: per-prefix label is the trajectory outcome
                # (soft variant scales the label by the empirical mean).
                if t.outcome is None:
                    if t.step_labels is None:
                        continue
                    y_traj = int(t.step_labels[-1])
                else:
                    y_traj = int(t.outcome)
                for i in range(len(t.steps)):
                    feats = hash_step_features(
                        t.steps[: i + 1],
                        feature_dim=cfg.feature_dim,
                        ngram_n=cfg.ngram_n,
                        step_index=i,
                        include_step_index=cfg.include_step_index,
                    )
                    out.append((feats, y_traj, t.weight))
            else:  # pragma: no cover
                raise UnknownModel(cfg.model)
        return out

    def _split_train_holdout(
        self, examples: list[tuple[Mapping[int, float], int, float]],
    ) -> tuple[list, list]:
        rng = random.Random(self._config.rng_seed ^ 0xA5A5)
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        n_holdout = max(1, int(self._config.holdout_fraction * len(idx)))
        # Always reserve at least one example for training.
        n_holdout = min(n_holdout, len(idx) - 1) if len(idx) > 1 else 0
        holdout_idx = set(idx[:n_holdout])
        train = [examples[i] for i in range(len(examples)) if i not in holdout_idx]
        holdout = [examples[i] for i in range(len(examples)) if i in holdout_idx]
        return train, holdout

    def _train_sgd(self, train: list[tuple[Mapping[int, float], int, float]]) -> None:
        cfg = self._config
        dim = cfg.feature_dim
        self._weights = [0.0] * dim
        self._velocity = [0.0] * dim
        if not train:
            return
        rng = random.Random(cfg.rng_seed ^ 0x1234)
        order = list(range(len(train)))
        for _ in range(cfg.epochs):
            rng.shuffle(order)
            for i in order:
                feats, y, w = train[i]
                z = _dot_sparse(self._weights, feats)
                p = _sigmoid(z)
                err = (p - float(y)) * w
                lr = cfg.learning_rate
                mu = cfg.momentum
                # Sparse update.
                for fid, v in feats.items():
                    if 0 <= fid < dim:
                        grad = err * v + cfg.l2 * self._weights[fid]
                        self._velocity[fid] = mu * self._velocity[fid] - lr * grad
                        self._weights[fid] += self._velocity[fid]

    def _evaluate(self, train: list, holdout: list) -> None:
        self._train_accuracy = self._accuracy(train)
        if holdout:
            self._holdout_accuracy = self._accuracy(holdout)
            self._holdout_logits = [
                _dot_sparse(self._weights, f) for f, _, _ in holdout
            ]
            self._holdout_labels = [int(y) for _, y, _ in holdout]
            probs = [_sigmoid(z) for z in self._holdout_logits]
            self._holdout_ece_before = expected_calibration_error(
                probs, self._holdout_labels
            )
            self._holdout_ece_after = self._holdout_ece_before
        else:
            self._holdout_accuracy = self._train_accuracy
            self._holdout_logits = []
            self._holdout_labels = []
            self._holdout_ece_before = 0.0
            self._holdout_ece_after = 0.0

    def _accuracy(self, examples: list) -> float:
        if not examples:
            return 0.0
        correct = 0.0
        total = 0.0
        for feats, y, w in examples:
            z = _dot_sparse(self._weights, feats)
            yhat = 1 if z >= 0 else 0
            correct += w if yhat == int(y) else 0.0
            total += w
        return correct / total if total > 0 else 0.0

    def _update_train_feature_density(self, examples: list) -> None:
        # A scalar per training trajectory summarising its full-trajectory
        # feature density (sum of feature counts divided by ngram_n).
        # Computed at full-trajectory length to match drift()'s stream
        # computation (both ingest complete trajectories there).
        cfg = self._config
        denom = max(1, cfg.ngram_n)
        self._train_feature_density = []
        for t in self._trajectories:
            feats = hash_step_features(
                t.steps,
                feature_dim=cfg.feature_dim,
                ngram_n=cfg.ngram_n,
                step_index=len(t.steps) - 1,
                include_step_index=cfg.include_step_index,
            )
            self._train_feature_density.append(sum(feats.values()) / denom)

    # -- Calibration ----------------------------------------------------

    def _calibrate(self) -> CalibrationResult:
        cfg = self._config
        if not self._holdout_labels:
            self._calibrated = True
            result = CalibrationResult(
                method=cfg.calibrator,
                ece_before=self._holdout_ece_before,
                ece_after=self._holdout_ece_after,
                parameters=tuple(),
            )
            self._append("calibrated", {
                "method": cfg.calibrator,
                "ece_before": self._holdout_ece_before,
                "ece_after": self._holdout_ece_after,
            })
            return result
        if cfg.calibrator == CAL_PLATT:
            self._platt_A, self._platt_B = platt_fit(
                self._holdout_logits, self._holdout_labels
            )
            probs_cal = platt_apply(
                self._holdout_logits, self._platt_A, self._platt_B
            )
            params = (self._platt_A, self._platt_B)
        elif cfg.calibrator == CAL_ISOTONIC:
            probs_raw = [_sigmoid(z) for z in self._holdout_logits]
            self._isotonic_thresh, self._isotonic_vals = isotonic_fit(
                probs_raw, self._holdout_labels
            )
            probs_cal = [
                isotonic_apply(p, self._isotonic_thresh, self._isotonic_vals)
                for p in probs_raw
            ]
            params = (float(len(self._isotonic_thresh)),)
        elif cfg.calibrator == CAL_NONE:
            probs_cal = [_sigmoid(z) for z in self._holdout_logits]
            params = tuple()
        else:  # pragma: no cover
            raise UnknownCalibrator(cfg.calibrator)
        self._holdout_ece_after = expected_calibration_error(
            probs_cal, self._holdout_labels
        )
        if self._holdout_ece_after > cfg.max_ece:
            raise InvalidConfig(
                f"ECE {self._holdout_ece_after:.3f} exceeds max_ece "
                f"{cfg.max_ece:.3f}; refusing to deploy calibrator"
            )
        self._calibrated = True
        self._append("calibrated", {
            "method": cfg.calibrator,
            "ece_before": self._holdout_ece_before,
            "ece_after": self._holdout_ece_after,
        })
        return CalibrationResult(
            method=cfg.calibrator,
            ece_before=self._holdout_ece_before,
            ece_after=self._holdout_ece_after,
            parameters=tuple(params),
        )

    def _apply_calibration(self, logits: Sequence[float]) -> list[float]:
        cfg = self._config
        if not self._calibrated or cfg.calibrator == CAL_NONE:
            return [_sigmoid(z) for z in logits]
        if cfg.calibrator == CAL_PLATT:
            return platt_apply(list(logits), self._platt_A, self._platt_B)
        if cfg.calibrator == CAL_ISOTONIC:
            raws = [_sigmoid(z) for z in logits]
            return [
                isotonic_apply(p, self._isotonic_thresh, self._isotonic_vals)
                for p in raws
            ]
        return [_sigmoid(z) for z in logits]  # pragma: no cover

    # -- Scoring --------------------------------------------------------

    def score(self, steps: Sequence[str]) -> TrajectoryScore:
        """Score a (possibly partial) reasoning trajectory."""
        with self._lock:
            if not self._fitted:
                raise NotFitted("call fit() before score()")
            if not steps:
                raise InvalidTrajectory("score requires at least one step")
            cfg = self._config
            logits: list[float] = []
            for i in range(len(steps)):
                feats = hash_step_features(
                    steps[: i + 1],
                    feature_dim=cfg.feature_dim,
                    ngram_n=cfg.ngram_n,
                    step_index=i,
                    include_step_index=cfg.include_step_index,
                )
                logits.append(_dot_sparse(self._weights, feats))
            raws = [_sigmoid(z) for z in logits]
            cals = self._apply_calibration(logits)
            per_step = tuple(
                StepScore(
                    step_index=i,
                    raw_logit=logits[i],
                    raw_prob=raws[i],
                    calibrated_prob=cals[i],
                )
                for i in range(len(steps))
            )
            agg = self._aggregate(cals)
            self._n_scored += 1
            self._last_aggregator = cfg.aggregator
            self._append("scored", {
                "n_steps": len(steps),
                "aggregated": agg,
                "aggregator": cfg.aggregator,
            })
            return TrajectoryScore(
                aggregated=agg,
                aggregator=cfg.aggregator,
                per_step=per_step,
            )

    def _aggregate(self, probs: Sequence[float]) -> float:
        cfg = self._config
        if cfg.aggregator == AGG_MIN:
            return stepwiser_min_aggregate(probs)
        if cfg.aggregator == AGG_MEAN:
            return stepwiser_mean_aggregate(probs)
        if cfg.aggregator == AGG_PROD:
            return stepwiser_prod_aggregate(probs)
        if cfg.aggregator == AGG_LAST:
            return stepwiser_last_aggregate(probs)
        if cfg.aggregator == AGG_LOGSUMEXP:
            return stepwiser_logsumexp_aggregate(probs, beta=cfg.logsumexp_beta)
        raise UnknownAggregator(cfg.aggregator)  # pragma: no cover

    # -- Selection ------------------------------------------------------

    def best_of_n(
        self, candidates: Sequence[Sequence[str]],
    ) -> SelectionResult:
        """Return the highest-scoring candidate trajectory."""
        with self._lock:
            if not candidates:
                raise InvalidTrajectory("candidates must be non-empty")
            scored: list[tuple[int, TrajectoryScore]] = []
            for i, c in enumerate(candidates):
                scored.append((i, self.score(c)))
            ranked = sorted(scored, key=lambda kv: kv[1].aggregated, reverse=True)
            chosen = ranked[0]
            gap = (ranked[0][1].aggregated - ranked[1][1].aggregated) \
                if len(ranked) >= 2 else ranked[0][1].aggregated
            # Anytime-valid LCB: Hoeffding with N = #candidates.
            half = _hoeffding_half_width(
                max(len(candidates), 1), 1.0 - self._config.confidence
            )
            gap_lcb = max(0.0, gap - 2.0 * half)
            self._n_selected += 1
            self._last_gap = gap
            self._last_gap_lcb = gap_lcb
            ranked_repr = tuple(
                (idx, sc.aggregated) for idx, sc in ranked
            )
            self._append("selected", {
                "chosen_index": chosen[0],
                "selection_gap": gap,
                "selection_gap_lcb": gap_lcb,
                "n_candidates": len(candidates),
            })
            return SelectionResult(
                chosen_index=chosen[0],
                chosen_score=chosen[1],
                ranked=ranked_repr,
                selection_gap=gap,
                selection_gap_lcb=gap_lcb,
            )

    # -- Beam search ----------------------------------------------------

    def beam_search(
        self,
        root: Sequence[str],
        *,
        expand: Callable[[tuple[str, ...]], Sequence[str]],
        terminal: Callable[[tuple[str, ...]], bool],
        k_beam: int | None = None,
        branch_factor: int | None = None,
        max_depth: int | None = None,
    ) -> BeamResult:
        """Stepwise beam search guided by PRM aggregated score.

        ``expand(beam)`` returns up to ``branch_factor`` candidate
        next steps (strings).  ``terminal(beam)`` returns ``True``
        when the beam is finished.  At every depth, the top ``k_beam``
        beams by aggregated score are retained.
        """
        with self._lock:
            if not self._fitted:
                raise NotFitted("call fit() before beam_search()")
            cfg = self._config
            kb = k_beam if k_beam is not None else cfg.k_beam
            bf = branch_factor if branch_factor is not None else cfg.branch_factor
            md = max_depth if max_depth is not None else cfg.max_depth
            if kb < 1 or bf < 1 or md < 1:
                raise InvalidConfig("k_beam/branch_factor/max_depth must be ≥ 1")
            beams: list[tuple[tuple[str, ...], float]] = [
                (tuple(root), self.score(root).aggregated if root else 0.0)
            ]
            expansions = 0
            depth = 0
            while depth < md:
                if all(terminal(b[0]) for b in beams):
                    break
                next_beams: list[tuple[tuple[str, ...], float]] = []
                for beam, _ in beams:
                    if terminal(beam):
                        next_beams.append((beam, self._score_or_zero(beam)))
                        continue
                    cands = list(expand(beam))[:bf]
                    expansions += 1
                    for nxt in cands:
                        if not isinstance(nxt, str):
                            raise InvalidStep("expand must yield str")
                        new = beam + (nxt,)
                        sc = self.score(new).aggregated
                        next_beams.append((new, sc))
                if not next_beams:
                    break
                next_beams.sort(key=lambda kv: kv[1], reverse=True)
                beams = next_beams[:kb]
                depth += 1
            self._n_beam_searched += 1
            sorted_final = sorted(beams, key=lambda kv: kv[1], reverse=True)
            self._append("searched", {
                "depth": depth,
                "expansions": expansions,
                "k_beam": kb,
                "branch_factor": bf,
                "best": sorted_final[0][1] if sorted_final else 0.0,
            })
            return BeamResult(
                beams=tuple(b[0] for b in sorted_final),
                scores=tuple(b[1] for b in sorted_final),
                depth=depth,
                expansions=expansions,
            )

    def _score_or_zero(self, beam: tuple[str, ...]) -> float:
        if not beam:
            return 0.0
        return self.score(beam).aggregated

    # -- Quantilization -------------------------------------------------

    def quantilize_step(
        self, prefix: Sequence[str], candidates: Sequence[str],
    ) -> tuple[str | None, list[tuple[str, float]]]:
        """Filter step candidates by the quantile-thresholded PRM score.

        Returns the *highest-scoring* candidate whose extended-prefix
        PRM aggregated score exceeds the ``q``-quantile of the
        training-distribution aggregated scores (``None`` if all reject).
        Also returns the per-candidate score list for transparency.
        """
        with self._lock:
            if not self._fitted:
                raise NotFitted("call fit() before quantilize_step()")
            if not candidates:
                raise InvalidStep("candidates must be non-empty")
            q = self._config.quantile
            ref = self._reference_distribution_scores()
            if ref:
                ref_sorted = sorted(ref)
                threshold_idx = max(0, min(int(q * len(ref_sorted)), len(ref_sorted) - 1))
                threshold = ref_sorted[threshold_idx]
            else:
                threshold = q
            scored: list[tuple[str, float]] = []
            for c in candidates:
                if not isinstance(c, str):
                    raise InvalidStep("candidates must be str")
                ext = tuple(prefix) + (c,)
                s = self.score(ext).aggregated
                scored.append((c, s))
            scored_sorted = sorted(scored, key=lambda kv: kv[1], reverse=True)
            chosen: str | None = None
            for c, s in scored_sorted:
                if s >= threshold:
                    chosen = c
                    break
            self._append("quantilized", {
                "n_candidates": len(candidates),
                "threshold": threshold,
                "quantile": q,
                "chosen": chosen is not None,
            })
            return chosen, scored

    def _reference_distribution_scores(self) -> list[float]:
        """Return a small representative set of aggregated training scores."""
        if not self._trajectories:
            return []
        out: list[float] = []
        for t in self._trajectories[: min(64, len(self._trajectories))]:
            try:
                out.append(self.score(t.steps).aggregated)
            except (NotFitted, InvalidTrajectory):
                continue
        return out

    # -- Reward shaping -------------------------------------------------

    def shape(self, steps: Sequence[str]) -> list[float]:
        """Return potential-based shaped rewards for the trajectory."""
        with self._lock:
            sc = self.score(steps)
            vals = [ss.calibrated_prob for ss in sc.per_step]
            shaped = stepwiser_potential_shaping(
                vals, discount=self._config.discount
            )
            self._append("shaped", {
                "n_steps": len(steps),
                "sum_shaped": sum(shaped),
            })
            return shaped

    # -- Drift detection ------------------------------------------------

    def drift(self, stream_steps: Sequence[Sequence[str]]) -> DriftResult:
        """Two-sample KS on feature densities (training vs stream)."""
        with self._lock:
            if not self._fitted:
                raise NotFitted("call fit() before drift()")
            if not stream_steps:
                raise InvalidTrajectory("stream must be non-empty")
            cfg = self._config
            denom = max(1, cfg.ngram_n)
            stream_density: list[float] = []
            for steps in stream_steps:
                feats = hash_step_features(
                    steps,
                    feature_dim=cfg.feature_dim,
                    ngram_n=cfg.ngram_n,
                    step_index=len(steps) - 1,
                    include_step_index=cfg.include_step_index,
                )
                stream_density.append(sum(feats.values()) / denom)
            ks = ks_two_sample(self._train_feature_density, stream_density)
            # Standard KS critical value (two-sample) approximation
            # c(α) = sqrt(-0.5 * ln(α/2)).
            ca = math.sqrt(-0.5 * math.log(cfg.drift_alpha / 2.0))
            n1 = len(self._train_feature_density)
            n2 = len(stream_density)
            threshold = ca * math.sqrt((n1 + n2) / max(1, n1 * n2))
            rejected = ks > threshold
            self._append("drift_detected" if rejected else "drift_checked", {
                "ks": ks,
                "threshold": threshold,
                "rejected": rejected,
                "n_train": n1,
                "n_stream": n2,
            })
            return DriftResult(
                ks_statistic=ks,
                threshold=threshold,
                rejected=rejected,
                n_train=n1,
                n_stream=n2,
            )

    # -- Reporting / certification --------------------------------------

    def report(self) -> StepwiserReport:
        with self._lock:
            n_steps = sum(len(t.steps) for t in self._trajectories)
            rep = StepwiserReport(
                config=self._config,
                n_trajectories_observed=len(self._trajectories),
                n_steps_observed=n_steps,
                fitted=self._fitted,
                calibrated=self._calibrated,
                train_accuracy=self._train_accuracy,
                holdout_accuracy=self._holdout_accuracy,
                holdout_ece=self._holdout_ece_after,
                n_scored=self._n_scored,
                n_selected=self._n_selected,
                n_beam_searched=self._n_beam_searched,
                last_selection_gap=self._last_gap,
                last_aggregator=self._last_aggregator,
                chain_head=self._chain_head,
            )
            self._append("reported", {
                "n_observed": len(self._trajectories),
                "fitted": self._fitted,
            })
            return rep

    def certify(self) -> StepwiserCertificate:
        with self._lock:
            if not self._fitted:
                raise NotFitted("call fit() before certify()")
            alpha = 1.0 - self._config.confidence
            n = len(self._holdout_labels)
            hoeff = _hoeffding_half_width(n, alpha) if n else 1.0
            indicators = []
            for z, y in zip(self._holdout_logits, self._holdout_labels):
                indicators.append(1.0 if (1 if z >= 0 else 0) == int(y) else 0.0)
            var = _empirical_variance(indicators)
            bern = _bernstein_half_width(n, var, alpha) if n else 1.0
            half = min(hoeff, bern)
            lcb = max(0.0, self._holdout_accuracy - half)
            fingerprint_payload = {
                "config_hash": self._config_hash(),
                "n_trajectories": len(self._trajectories),
                "holdout_accuracy": self._holdout_accuracy,
                "last_gap": self._last_gap,
                "chain_head": self._chain_head,
            }
            fingerprint = hashlib.sha256(
                _canonical_bytes(fingerprint_payload)
            ).hexdigest()
            self._append("certified", {
                "confidence": self._config.confidence,
                "holdout_accuracy": self._holdout_accuracy,
                "lcb": lcb,
                "fingerprint": fingerprint,
            })
            return StepwiserCertificate(
                confidence=self._config.confidence,
                n_holdout=n,
                holdout_accuracy=self._holdout_accuracy,
                hoeffding_half_width=hoeff,
                bernstein_half_width=bern,
                accuracy_lcb=lcb,
                last_selection_gap=self._last_gap,
                last_selection_gap_lcb=self._last_gap_lcb,
                chain_head=self._chain_head,
                fingerprint_hash=fingerprint,
            )

    # -- Skill mining ---------------------------------------------------

    def top_features(self, k: int = 16) -> list[tuple[int, float]]:
        """Return the top-``k`` weight indices by |weight|.

        A coordinator can promote the implied n-gram patterns into the
        skill library (since the hash is deterministic, equivalent
        prefixes activate the same slot).
        """
        with self._lock:
            if not self._fitted:
                raise NotFitted("call fit() before top_features()")
            ranked = sorted(
                enumerate(self._weights),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            return ranked[: max(0, k)]

    # -- Snapshot / restore ---------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "weights": list(self._weights),
                "velocity": list(self._velocity),
                "fitted": self._fitted,
                "train_accuracy": self._train_accuracy,
                "holdout_accuracy": self._holdout_accuracy,
                "holdout_logits": list(self._holdout_logits),
                "holdout_labels": list(self._holdout_labels),
                "holdout_ece_before": self._holdout_ece_before,
                "holdout_ece_after": self._holdout_ece_after,
                "calibrated": self._calibrated,
                "platt_A": self._platt_A,
                "platt_B": self._platt_B,
                "isotonic_thresh": list(self._isotonic_thresh),
                "isotonic_vals": list(self._isotonic_vals),
                "n_scored": self._n_scored,
                "n_selected": self._n_selected,
                "n_beam_searched": self._n_beam_searched,
                "last_gap": self._last_gap,
                "last_gap_lcb": self._last_gap_lcb,
                "last_aggregator": self._last_aggregator,
                "chain_head": self._chain_head,
                "train_feature_density": list(self._train_feature_density),
                "trajectories": [
                    {
                        "steps": list(t.steps),
                        "step_labels": list(t.step_labels)
                        if t.step_labels is not None else None,
                        "outcome": t.outcome,
                        "weight": t.weight,
                    }
                    for t in self._trajectories
                ],
            }

    def restore(self, snap: Mapping[str, Any]) -> None:
        with self._lock:
            self._weights = list(snap["weights"])
            self._velocity = list(snap["velocity"])
            self._fitted = bool(snap["fitted"])
            self._train_accuracy = float(snap["train_accuracy"])
            self._holdout_accuracy = float(snap["holdout_accuracy"])
            self._holdout_logits = list(snap["holdout_logits"])
            self._holdout_labels = list(snap["holdout_labels"])
            self._holdout_ece_before = float(snap["holdout_ece_before"])
            self._holdout_ece_after = float(snap["holdout_ece_after"])
            self._calibrated = bool(snap["calibrated"])
            self._platt_A = float(snap["platt_A"])
            self._platt_B = float(snap["platt_B"])
            self._isotonic_thresh = tuple(snap["isotonic_thresh"])
            self._isotonic_vals = tuple(snap["isotonic_vals"])
            self._n_scored = int(snap["n_scored"])
            self._n_selected = int(snap["n_selected"])
            self._n_beam_searched = int(snap["n_beam_searched"])
            self._last_gap = float(snap["last_gap"])
            self._last_gap_lcb = float(snap["last_gap_lcb"])
            self._last_aggregator = str(snap["last_aggregator"])
            self._chain_head = str(snap["chain_head"])
            self._train_feature_density = list(snap["train_feature_density"])
            self._trajectories = []
            for d in snap["trajectories"]:
                self._trajectories.append(TrajectoryRecord(
                    steps=tuple(d["steps"]),
                    step_labels=tuple(d["step_labels"])
                    if d["step_labels"] is not None else None,
                    outcome=d["outcome"],
                    weight=float(d["weight"]),
                ))

    # -- Reset ----------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._trajectories = []
            self._weights = [0.0] * self._config.feature_dim
            self._velocity = [0.0] * self._config.feature_dim
            self._fitted = False
            self._calibrated = False
            self._train_accuracy = 0.0
            self._holdout_accuracy = 0.0
            self._holdout_ece_before = 0.0
            self._holdout_ece_after = 0.0
            self._holdout_logits = []
            self._holdout_labels = []
            self._platt_A = 1.0
            self._platt_B = 0.0
            self._isotonic_thresh = tuple()
            self._isotonic_vals = tuple()
            self._n_scored = 0
            self._n_selected = 0
            self._n_beam_searched = 0
            self._last_gap = 0.0
            self._last_gap_lcb = 0.0
            self._train_feature_density = []
            self._chain_head = stepwiser_ledger_root()
            self._chain_history = []
            self._append("reset", {})
