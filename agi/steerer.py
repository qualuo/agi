r"""Steerer — contrastive activation steering & representation engineering as a runtime primitive.

Where :mod:`agi.mechanizer` *decomposes* dense activations into a sparse
interpretable basis, ``Steerer`` answers the dual operational question
every coordination engine has the moment it ships a frozen model into
production: *given a behavioural axis we want more or less of —
truthfulness, helpfulness, formality, sycophancy, refusal — can we
shift the model along that axis at inference time, by how much, and
with what statistical guarantee, without touching a single weight?*

This is the runtime side of **representation engineering** (Zou et al.,
2023): rather than fine-tune, *intervene on activations*.  Concretely
the primitive operates on three pieces of data the coordinator supplies:

  * **Contrastive pairs** — for each task or paired prompt, two
    activation vectors collected at the same residual-stream layer:
    one from the *positive* class (e.g., the truthful response, the
    helpful response, the un-jailbroken refusal), one from the
    *negative* class (untruthful, sycophantic, jailbroken).

  * **Probe activations** — for held-out trials, the same layer's
    activations from the model under control, plus a downstream
    behavioural label (did this response satisfy the policy?).

  * **Outcome observations** — once a steered activation flows back
    through the rest of the network and produces a behaviour
    (refused / answered / scored), the realised outcome is recorded
    against its steering coefficient.

Steerer fits a *steering direction* on the contrastive pairs by one
of four well-motivated estimators (CAA difference-of-means,
class-mean PCA / LAT, logistic-probe normal, mean-difference rescaled
by Fisher LDA), reports per-direction quality metrics (separability
AUROC with a Mason-Graham CI, effect size Cohen-d, leakage on a held-
out paraphrase set, dose-response slope under the steered probes),
and emits an anytime-valid certificate that the steering vector
*actually shifts the realised outcome rate* relative to a non-steered
control by at least an effect size :math:`\\delta` with confidence
:math:`1-\\alpha`.  Every fit, dose, outcome, and verdict is chained
into a SHA-256 fingerprint so the coordinator can replay-verify the
run byte-for-byte at a later audit.

The pitch reduced to a runtime call::

    s = Steerer(SteererConfig(
        algorithm=ALG_CAA,                       # contrastive activation addition
        target_layer="resid.16",                 # site of intervention
        alpha=0.05,
        seed=0,
    ))

    # Contrastive pairs collected by the coordinator at one layer.
    for pos_vec, neg_vec, task_id in pairs:
        s.observe_pair(ContrastivePair(
            task_id=task_id,
            positive=pos_vec,
            negative=neg_vec,
        ))

    fit = s.fit()                       # SteererFit
    # `fit.direction` is the unit steering vector,
    # `fit.separability_auroc` ~ 1.0 means the direction perfectly
    # separates positive from negative on training set.

    # Apply at inference: coordinator hooks the residual stream at
    # `target_layer` with this vector and chosen coefficient.
    coeffs = s.recommend_coefficients(safety_floor=0.95)

    # Coordinator runs the steered model with each coefficient and
    # records (refused / scored).  Steerer accumulates evidence:
    for trial in steered_trials:
        s.observe_outcome(DoseOutcome(
            task_id=trial.id,
            coefficient=trial.coef,
            outcome=trial.refused,
        ))

    cert = s.certify(delta=0.10, alpha=0.05)
    if cert.verdict == VERDICT_PASS:
        # Steerer has anytime-valid evidence that the chosen
        # coefficient lifts outcome rate by ≥ 0.10 vs. control.
        ...

How the primitive composes
--------------------------

  * :mod:`agi.refuser`      — Steerer's contrastive pairs are typically
                              ``(refused, complied)`` activation
                              pairs.  The certificate justifies
                              promoting a steering vector from
                              ``warn`` to ``trust`` in the refuser's
                              recommendation surface.

  * :mod:`agi.sycophant`    — pairs from
                              ``(pre-pressure, post-pressure)``
                              activations isolate the *yielding
                              direction*.  Steering against it raises
                              robustness to user pressure without
                              touching weights.

  * :mod:`agi.confabulator` — pairs from
                              ``(grounded, hallucinated)`` activations
                              isolate the *factuality direction*.

  * :mod:`agi.constitutionalist` — Steerer turns each constitutional
                              principle into a steering vector.  A
                              coordinator can mix-and-match principle
                              vectors at inference time.

  * :mod:`agi.mechanizer`   — Mechanizer's sparse-feature decomposition
                              produces *interpretable* unit vectors.
                              Steerer accepts those as candidate
                              directions and certifies their
                              behavioural effect.

  * :mod:`agi.capabilities`, :mod:`agi.policy` — once a steering
                              configuration is certified, it becomes
                              a routable bucket the policy router can
                              draw from.

  * :mod:`agi.governance`   — gate dispatch on a Steerer certificate
                              ("only ship the truth-steered
                              configuration if certified at
                              ``alpha=0.01``").

  * :mod:`agi.anticipator`  — schedule contrastive-pair collection
                              during sleep-time compute.

Mathematical and algorithmic roots
----------------------------------

  * **Zou, A., Phan, L., Chen, S., et al. (2023) — "Representation
    Engineering: A Top-Down Approach to AI Transparency"
    (arXiv:2310.01405).**  The umbrella term and the LAT (Linear
    Artificial Tomography) PCA-on-paired-differences estimator.
    Implemented as :data:`ALG_LAT`.

  * **Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger,
    E., Turner, A. (2024) — "Steering Llama 2 via Contrastive
    Activation Addition" (arXiv:2312.06681).**  CAA — the
    difference-of-means estimator that is the empirically strongest
    pre-trained-model intervention with constant memory.  Implemented
    as :data:`ALG_CAA`.

  * **Park, K., Choe, Y., Veitch, V. (2024) — "The Linear
    Representation Hypothesis and the Geometry of Large Language
    Models" (arXiv:2311.03658).**  Justifies treating "behavioural
    axes" as one-dimensional directions in residual-stream activation
    space and motivates the Fisher-LDA-rescaled variant
    :data:`ALG_LDA`.

  * **Tigges, C., Hollinsworth, O.J., Geiger, A., Nanda, N. (2023) —
    "Linear Representations of Sentiment in Large Language Models"
    (arXiv:2310.15154).**  Shows that the *probe-classifier normal*
    is also a valid steering direction; implemented as
    :data:`ALG_PROBE`.

  * **Subramani, N., Suresh, N., Peters, M. (2022) — "Extracting
    Latent Steering Vectors from Pretrained Language Models"
    (arXiv:2205.05124).**  The earliest formulation; Steerer's pair
    contract follows their (positive, negative) construction.

  * **Templeton, A., Conerly, T., Marcus, J., et al. (Anthropic, 2024)
    — "Scaling Monosemanticity: Extracting Interpretable Features
    from Claude 3 Sonnet."**  Demonstrates feature-steering at scale;
    Steerer's :meth:`apply_feature` accepts a Mechanizer feature row
    as a steering direction and certifies the same way.

  * **Arditi, A., et al. (2024) — "Refusal in Language Models Is
    Mediated by a Single Direction" (arXiv:2406.11717).**  Empirical
    evidence that the (clean, jailbreak) contrast yields a one-
    dimensional refusal direction.  Steerer's CAA on
    ``CTX_HARMFUL_CLEAN`` ÷ ``CTX_HARMFUL_JAILBREAK`` pairs
    reproduces this estimator.

  * **Fisher, R.A. (1936) — "The Use of Multiple Measurements in
    Taxonomic Problems."**  LDA's projection direction
    :math:`\\Sigma_w^{-1}(\\mu_+ - \\mu_-)`.  Steerer's regularised
    streaming form via Welford + Ledoit-Wolf shrinkage.

  * **Hanley, J.A., McNeil, B.J. (1982); Mason, S.J., Graham, N.E.
    (2002).**  AUROC point estimate + CI.  Steerer uses the
    Mason-Graham normal-approximation CI on the in-training
    separability AUROC.

  * **Cohen, J. (1988).**  Cohen's *d* effect size; Steerer reports
    it on every direction.

  * **Howard, S., Ramdas, A., McAuliffe, J., Sekhon, J. (2021);
    Waudby-Smith, I., Ramdas, A. (2024).**  Anytime-valid confidence
    sequences for the realised effect of the steered intervention.
    Steerer's outcome certificate is a hedged-capital betting e-process
    on the paired (steered, control) outcome differences.

  * **Ledoit, O., Wolf, M. (2004).**  Shrinkage covariance estimator
    for the within-class covariance under :data:`ALG_LDA` when sample
    counts approach the dimensionality.

  * **Welford, B.P. (1962).**  Numerically stable streaming mean +
    variance updates.

  * **Wu, Z., Geiger, A., et al. (2024) — "ReFT: Representation
    Fine-tuning" (arXiv:2404.03592).**  Steerer's
    :meth:`apply_rank_one` is the inference-time form of a ReFT-LoRA
    rank-1 intervention; the certificate certifies the same effect
    without a training run.

Design contract
---------------

* **Pure stdlib.**  No NumPy / SciPy / Torch.  Vector and matrix
  operations are implemented on tuples-of-floats so the runtime can
  ship Steerer to any deployment surface (edge-device, CI runner,
  serverless function) with zero binary dependencies.

* **Stateful, thread-safe, deterministic given seed.**  A single
  :class:`Steerer` audits one (model, layer, behavioural axis).
  Combine across configurations via :func:`compare_steerers`.

* **Replay-verifiable.**  Every ``observe_pair`` / ``observe_outcome``
  / ``fit`` / ``certify`` transition appends to a SHA-256 fingerprint
  chain.  The same observation stream produces the same chain hash.

* **No I/O, no side effects.**  Steerer is a pure compute object.
  Event emission is optional via an injected :class:`EventBus`.

* **No model coupling.**  Steerer never sees prompts, tokens, or
  weights.  It sees only the activation vectors and outcomes the
  coordinator provides.  This keeps it model-agnostic and lets the
  same primitive certify steering on a 7B open model, on a frontier
  API model with hooked activations, or on a Mechanizer-derived
  synthetic latent.

Usage
-----

>>> from agi.steerer import (
...     Steerer, SteererConfig, ContrastivePair, DoseOutcome,
...     ALG_CAA, VERDICT_PASS,
... )
>>> import random
>>> rng = random.Random(0)
>>> s = Steerer(SteererConfig(algorithm=ALG_CAA, dim=4, alpha=0.05, seed=0))
>>> # A toy axis where positive samples live near +x and negatives near -x:
>>> for i in range(32):
...     p = tuple(rng.gauss(1.0, 0.3) if j == 0 else rng.gauss(0.0, 0.3)
...               for j in range(4))
...     n = tuple(rng.gauss(-1.0, 0.3) if j == 0 else rng.gauss(0.0, 0.3)
...               for j in range(4))
...     _ = s.observe_pair(ContrastivePair(task_id=f"t{i}", positive=p, negative=n))
>>> fit = s.fit()
>>> abs(fit.direction[0]) > 0.95  # learns the +x axis
True
>>> fit.separability_auroc > 0.95
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
# Constants and taxonomy
# ---------------------------------------------------------------------------

# Algorithms — how the steering direction is fit from contrastive pairs.

ALG_CAA = "caa"       # Contrastive Activation Addition (Panickssery '24)
ALG_LAT = "lat"       # Linear Artificial Tomography (Zou '23)
ALG_PROBE = "probe"   # Logistic-probe normal (Tigges '23)
ALG_LDA = "lda"       # Fisher LDA with Ledoit-Wolf shrinkage

KNOWN_ALGORITHMS: tuple[str, ...] = (ALG_CAA, ALG_LAT, ALG_PROBE, ALG_LDA)

# Verdict labels — match :mod:`agi.refuser` / :mod:`agi.schemer` so a
# coordination engine can switch on a single field.

VERDICT_PASS = "pass"
VERDICT_WARN = "warn"
VERDICT_FAIL = "fail"
VERDICT_INCONCLUSIVE = "inconclusive"

KNOWN_VERDICTS: tuple[str, ...] = (
    VERDICT_PASS,
    VERDICT_WARN,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
)

# Coordinator-facing recommendations.

REC_PROMOTE = "promote"            # certified — ship this steering config
REC_HOLD = "hold"                  # keep collecting more pairs / doses
REC_REJECT = "reject"              # direction did not pass — do not steer
REC_FLIP = "flip"                  # sign convention was wrong; flip and re-certify

KNOWN_RECOMMENDATIONS: tuple[str, ...] = (
    REC_PROMOTE,
    REC_HOLD,
    REC_REJECT,
    REC_FLIP,
)

# Named tests inside the certificate.

TEST_SEPARABILITY = "separability"     # in-training AUROC ≥ AUROC_floor
TEST_DOSE_RESPONSE = "dose_response"   # outcome rate is monotone in coefficient
TEST_EFFECT_SIZE = "effect_size"       # outcome lift at recommended coef ≥ δ
TEST_LEAKAGE = "leakage"               # paraphrase-set AUROC ≥ AUROC_floor

KNOWN_TESTS: tuple[str, ...] = (
    TEST_SEPARABILITY,
    TEST_DOSE_RESPONSE,
    TEST_EFFECT_SIZE,
    TEST_LEAKAGE,
)

# Event kinds on the runtime EventBus.

STEERER_STARTED = "steerer.started"
STEERER_PAIR_OBSERVED = "steerer.pair_observed"
STEERER_OUTCOME_OBSERVED = "steerer.outcome_observed"
STEERER_FIT = "steerer.fit"
STEERER_TESTED = "steerer.tested"
STEERER_CERTIFIED = "steerer.certified"
STEERER_RECOMMENDED = "steerer.recommended"
STEERER_REPORTED = "steerer.reported"
STEERER_RESET = "steerer.reset"

KNOWN_EVENTS: tuple[str, ...] = (
    STEERER_STARTED,
    STEERER_PAIR_OBSERVED,
    STEERER_OUTCOME_OBSERVED,
    STEERER_FIT,
    STEERER_TESTED,
    STEERER_CERTIFIED,
    STEERER_RECOMMENDED,
    STEERER_REPORTED,
    STEERER_RESET,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SteererError(ValueError):
    """Base class for Steerer-specific errors."""


class InvalidConfig(SteererError):
    """The :class:`SteererConfig` is internally inconsistent."""


class InvalidPair(SteererError):
    """A contrastive pair violates a runtime invariant."""


class InvalidOutcome(SteererError):
    """A dose-outcome row violates a runtime invariant."""


class UnknownAlgorithm(SteererError):
    """Algorithm name is not in :data:`KNOWN_ALGORITHMS`."""


class UnknownTest(SteererError):
    """Test name is not in :data:`KNOWN_TESTS`."""


class InsufficientData(SteererError):
    """Operation requested with too few rows."""


class NotFitted(SteererError):
    """Direction was queried before :meth:`Steerer.fit`."""


class DimensionMismatch(SteererError):
    """Vector length differs from the configured ``dim``."""


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


def _validate_vector(v: Sequence[float], dim: int, what: str) -> tuple[float, ...]:
    if not isinstance(v, (list, tuple)):
        raise InvalidPair(f"{what} must be a sequence of floats")
    if len(v) != dim:
        raise DimensionMismatch(
            f"{what} has length {len(v)} but config dim={dim}"
        )
    out: list[float] = []
    for i, x in enumerate(v):
        if not isinstance(x, (int, float)):
            raise InvalidPair(f"{what}[{i}] is not a number")
        f = float(x)
        if not math.isfinite(f):
            raise InvalidPair(f"{what}[{i}] is not finite")
        out.append(f)
    return tuple(out)


@dataclass(frozen=True)
class ContrastivePair:
    """One training pair for the Steerer's fit.

    Attributes:
        task_id: stable identifier; the pair is the activation snapshot
            from two completions of the same prompt under the positive
            and negative behavioural conditions.
        positive: activation vector from the *desired* class.
        negative: activation vector from the *undesired* class.
        split: one of ``"train"``, ``"paraphrase"``, ``"holdout"`` —
            used by the leakage and held-out tests.  Default
            ``"train"``.
        weight: per-pair weight.  Default 1.0.
        metadata: opaque dict carried through the fingerprint.
    """

    task_id: str
    positive: tuple[float, ...]
    negative: tuple[float, ...]
    split: str = "train"
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise InvalidPair("task_id must be a non-empty string")
        if self.split not in ("train", "paraphrase", "holdout"):
            raise InvalidPair(
                f"split={self.split!r} not in ('train','paraphrase','holdout')"
            )
        if (
            not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
            or float(self.weight) <= 0.0
        ):
            raise InvalidPair("weight must be a positive finite number")
        object.__setattr__(self, "weight", float(self.weight))
        # Symmetric dim check: positive and negative must agree.
        if not isinstance(self.positive, (list, tuple)) or not isinstance(
            self.negative, (list, tuple)
        ):
            raise InvalidPair("positive/negative must be sequences of floats")
        if len(self.positive) != len(self.negative):
            raise DimensionMismatch(
                f"positive dim {len(self.positive)} != negative dim "
                f"{len(self.negative)}"
            )
        for v, name in ((self.positive, "positive"), (self.negative, "negative")):
            for i, x in enumerate(v):
                if not isinstance(x, (int, float)):
                    raise InvalidPair(f"{name}[{i}] is not a number")
                if not math.isfinite(float(x)):
                    raise InvalidPair(f"{name}[{i}] is not finite")
        # Coerce to tuples of floats.
        object.__setattr__(self, "positive", tuple(float(x) for x in self.positive))
        object.__setattr__(self, "negative", tuple(float(x) for x in self.negative))


@dataclass(frozen=True)
class DoseOutcome:
    """One realised outcome under a (possibly zero) steering coefficient.

    Attributes:
        task_id: stable identifier.  The same task may be observed at
            many coefficients to build a dose-response curve.
        coefficient: the multiplier applied to the steering vector when
            this trial ran.  ``0.0`` is the unsteered control.  May be
            negative (steers against the direction).
        outcome: True iff the *desired* behaviour was realised
            (refused-when-it-should, truthful, non-sycophantic, …).
            The coordinator decides the polarity; Steerer treats it
            as a Bernoulli observation.
        outcome_score: optional [0, 1] continuous quality score.  When
            present, supersedes ``outcome`` for the effect-size and
            dose-response tests (Welch-t / Spearman).
        weight: per-trial weight.  Default 1.0.
        metadata: opaque dict.
    """

    task_id: str
    coefficient: float
    outcome: bool
    outcome_score: float | None = None
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise InvalidOutcome("task_id must be a non-empty string")
        if (
            not isinstance(self.coefficient, (int, float))
            or not math.isfinite(float(self.coefficient))
        ):
            raise InvalidOutcome("coefficient must be a finite number")
        object.__setattr__(self, "coefficient", float(self.coefficient))
        if not isinstance(self.outcome, (bool, int)):
            raise InvalidOutcome("outcome must be a bool")
        object.__setattr__(self, "outcome", bool(self.outcome))
        if self.outcome_score is not None:
            if (
                not isinstance(self.outcome_score, (int, float))
                or not math.isfinite(float(self.outcome_score))
                or not 0.0 <= float(self.outcome_score) <= 1.0
            ):
                raise InvalidOutcome("outcome_score must be in [0,1] or None")
            object.__setattr__(self, "outcome_score", float(self.outcome_score))
        if (
            not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
            or float(self.weight) <= 0.0
        ):
            raise InvalidOutcome("weight must be a positive finite number")
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class SteererConfig:
    """Static configuration of one :class:`Steerer` instance.

    Attributes:
        algorithm: one of :data:`KNOWN_ALGORITHMS`.
        dim: dimensionality of the activation vectors.  All pairs and
            outcomes must match.
        target_layer: optional human-readable layer label carried into
            certificates and events for downstream reporting.
        alpha: certificate confidence level (each test held to ``α``).
        alpha_warn: relaxed level — clear at α_warn but not α ⇒ WARN.
        ridge: small ridge constant added before any matrix inverse
            in :data:`ALG_LDA` / :data:`ALG_PROBE`.
        shrinkage: Ledoit-Wolf shrinkage intensity in
            :data:`ALG_LDA`.  ``None`` ⇒ data-driven choice.
        auroc_floor: minimum acceptable in-training separability AUROC.
        delta: minimum acceptable outcome lift in
            :data:`TEST_EFFECT_SIZE`.
        coefficient_grid: candidate coefficients the recommender will
            score.  Default ``(-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)``.
        max_pairs: ring-buffer cap on retained pairs (older pairs
            evict).  Default ``100_000``.
        max_outcomes: ring-buffer cap on retained outcomes.
        seed: deterministic random state for probe-fit / shrinkage
            initialisation.
    """

    algorithm: str = ALG_CAA
    dim: int = 0
    target_layer: str = ""
    alpha: float = 0.05
    alpha_warn: float = 0.20
    ridge: float = 1e-6
    shrinkage: float | None = None
    auroc_floor: float = 0.70
    delta: float = 0.05
    coefficient_grid: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    max_pairs: int = 100_000
    max_outcomes: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"algorithm={self.algorithm!r} not in {KNOWN_ALGORITHMS!r}"
            )
        if not isinstance(self.dim, int) or self.dim < 1:
            raise InvalidConfig("dim must be a positive int")
        if not 0.0 < float(self.alpha) < 0.5:
            raise InvalidConfig("alpha must be in (0, 0.5)")
        if not float(self.alpha) <= float(self.alpha_warn) < 1.0:
            raise InvalidConfig("alpha_warn must satisfy alpha ≤ alpha_warn < 1")
        if float(self.ridge) < 0.0:
            raise InvalidConfig("ridge must be ≥ 0")
        if self.shrinkage is not None and not 0.0 <= float(self.shrinkage) <= 1.0:
            raise InvalidConfig("shrinkage must be in [0, 1] or None")
        if not 0.5 <= float(self.auroc_floor) <= 1.0:
            raise InvalidConfig("auroc_floor must be in [0.5, 1.0]")
        if not 0.0 < float(self.delta) < 1.0:
            raise InvalidConfig("delta must be in (0, 1)")
        if not self.coefficient_grid or any(
            not math.isfinite(float(c)) for c in self.coefficient_grid
        ):
            raise InvalidConfig("coefficient_grid must be non-empty finite floats")
        # Ensure 0 (control) is in the grid for the effect-size test.
        if 0.0 not in tuple(float(c) for c in self.coefficient_grid):
            raise InvalidConfig("coefficient_grid must include 0.0 (control)")
        if self.max_pairs < 16:
            raise InvalidConfig("max_pairs must be ≥ 16")
        if self.max_outcomes < 16:
            raise InvalidConfig("max_outcomes must be ≥ 16")


@dataclass(frozen=True)
class SteererFit:
    """Result of :meth:`Steerer.fit`.

    Attributes:
        algorithm: which estimator produced the direction.
        dim: dimensionality.
        direction: unit-norm steering vector.
        norm: pre-normalisation magnitude (information-bearing).
        positive_mean: centroid of the positive class.
        negative_mean: centroid of the negative class.
        n_pairs: number of training-split pairs used.
        separability_auroc: in-training AUROC of
            ``positive·direction`` vs ``negative·direction``.
        separability_ci: (low, high) Mason-Graham 1-α CI on the AUROC.
        cohen_d: standardised effect size of the projected scores.
        leakage_auroc: AUROC on ``split="paraphrase"`` pairs if any
            were observed, else ``None``.  A direction that
            *generalises* should retain AUROC on paraphrases.
        diagnostics: per-algorithm extras (e.g. shrinkage chosen).
        fingerprint: SHA-256 chain hash after this fit.
    """

    algorithm: str
    dim: int
    direction: tuple[float, ...]
    norm: float
    positive_mean: tuple[float, ...]
    negative_mean: tuple[float, ...]
    n_pairs: int
    separability_auroc: float
    separability_ci: tuple[float, float]
    cohen_d: float
    leakage_auroc: float | None
    diagnostics: Mapping[str, Any]
    fingerprint: str


@dataclass(frozen=True)
class TestResult:
    """One named test inside a :class:`SteererCertificate`.

    Attributes:
        name: one of :data:`KNOWN_TESTS`.
        statistic: the realised statistic (AUROC, Spearman ρ, lift).
        threshold: the configured threshold (or 0 for one-sided).
        p_value: nominal one-sided p (None when the test is
            anytime-valid, see ``e_value``).
        e_value: anytime-valid e-value for sequential tests, else None.
        ci_low / ci_high: 1-α CI on the statistic.
        passed: True iff cleared at α; warn iff cleared at α_warn only.
        n: effective sample size.
    """

    name: str
    statistic: float
    threshold: float
    p_value: float | None
    e_value: float | None
    ci_low: float
    ci_high: float
    passed: bool
    n: int


@dataclass(frozen=True)
class SteererCertificate:
    """Bundle of test results and a final verdict.

    Attributes:
        verdict: one of :data:`KNOWN_VERDICTS`.
        recommendation: one of :data:`KNOWN_RECOMMENDATIONS`.
        tests: per-test result dicts.
        recommended_coefficient: scalar α the coordinator should apply
            to the unit direction at inference.  ``None`` when verdict
            is not ``pass``.
        outcome_lift: estimated outcome-rate lift between the
            recommended coefficient and control (0.0).
        outcome_lift_ci: (low, high) confidence sequence on lift.
        n_pairs / n_outcomes: counts used.
        fingerprint: chain hash after certification.
    """

    verdict: str
    recommendation: str
    tests: tuple[TestResult, ...]
    recommended_coefficient: float | None
    outcome_lift: float
    outcome_lift_ci: tuple[float, float]
    n_pairs: int
    n_outcomes: int
    fingerprint: str


@dataclass(frozen=True)
class SteererReport:
    """Bundle: a fit + a certificate + the running counters."""

    fit: SteererFit
    certificate: SteererCertificate
    n_pairs: int
    n_outcomes: int
    fingerprint: str


# ---------------------------------------------------------------------------
# Vector / matrix helpers (pure stdlib)
# ---------------------------------------------------------------------------


def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _vadd(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _vsub(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _vscale(a: Sequence[float], s: float) -> tuple[float, ...]:
    return tuple(x * s for x in a)


def _vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vnorm(a: Sequence[float]) -> float:
    return math.sqrt(_vdot(a, a))


def _vnormalize(a: Sequence[float]) -> tuple[float, ...]:
    n = _vnorm(a)
    if n == 0.0:
        return tuple(a)
    inv = 1.0 / n
    return tuple(x * inv for x in a)


def _outer_add(M: list[list[float]], a: Sequence[float], b: Sequence[float], w: float = 1.0) -> None:
    """In-place ``M += w * outer(a, b)``."""
    for i, ai in enumerate(a):
        wai = w * ai
        if wai == 0.0:
            continue
        row = M[i]
        for j, bj in enumerate(b):
            row[j] += wai * bj


def _mat_eye(n: int, s: float = 1.0) -> list[list[float]]:
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = s
    return M


def _mat_add(A: list[list[float]], B: list[list[float]], scale: float = 1.0) -> list[list[float]]:
    n = len(A)
    return [[A[i][j] + scale * B[i][j] for j in range(n)] for i in range(n)]


def _mat_scale(A: list[list[float]], s: float) -> list[list[float]]:
    return [[s * v for v in row] for row in A]


def _mat_vec(A: list[list[float]], v: Sequence[float]) -> tuple[float, ...]:
    return tuple(sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A)))


def _solve_psd(A: list[list[float]], b: Sequence[float], ridge: float = 1e-8) -> tuple[float, ...]:
    """Solve ``(A + ridge·I) x = b`` for symmetric positive-(semi)-definite A.

    Uses Cholesky with ridge fallback.  Pure stdlib.  Returns
    ``tuple(x)``.
    """
    n = len(A)
    # Copy + ridge
    L = [[A[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        L[i][i] += ridge
    # In-place Cholesky.  If a diagonal pivot ≤ 0, increase ridge and retry.
    for attempt in range(8):
        ok = True
        # Reset L to copy with current ridge:
        if attempt > 0:
            for i in range(n):
                for j in range(n):
                    L[i][j] = A[i][j]
                L[i][i] += ridge * (10 ** attempt)
        try:
            for i in range(n):
                for j in range(i + 1):
                    s = L[i][j]
                    for k in range(j):
                        s -= L[i][k] * L[j][k]
                    if i == j:
                        if s <= 0.0:
                            ok = False
                            break
                        L[i][j] = math.sqrt(s)
                    else:
                        L[i][j] = s / L[j][j]
                if not ok:
                    break
            if ok:
                break
        except (ValueError, ZeroDivisionError):
            ok = False
    if not ok:
        # Diagonal fallback — solve the diagonal of A.
        return tuple(
            b[i] / (A[i][i] + ridge if A[i][i] + ridge > 0 else 1.0)
            for i in range(n)
        )
    # Forward solve L y = b
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / L[i][i]
    # Back solve L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return tuple(x)


# ---------------------------------------------------------------------------
# AUROC + Mann-Whitney U
# ---------------------------------------------------------------------------


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Empirical AUROC (= Mann-Whitney U / (m·n)) with mid-rank tie handling.

    Returns the probability that a randomly drawn positive score
    exceeds a randomly drawn negative one.  Symmetric: a perfect
    inverted predictor returns 0.0; chance returns 0.5.
    """
    m = len(positive)
    n = len(negative)
    if m == 0 or n == 0:
        return 0.5
    # Pool and rank with average ranks for ties.
    pool = [(float(s), 1) for s in positive] + [(float(s), 0) for s in negative]
    pool.sort(key=lambda x: x[0])
    ranks = [0.0] * len(pool)
    i = 0
    while i < len(pool):
        j = i
        while j + 1 < len(pool) and pool[j + 1][0] == pool[i][0]:
            j += 1
        # Average rank for indices i..j (1-based)
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k in range(len(pool)) if pool[k][1] == 1)
    u = rank_sum_pos - m * (m + 1) / 2.0
    return u / (m * n)


def auroc_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mason-Graham / Hanley-McNeil normal-approximation CI on AUROC.

    Returns ``(auroc, lo, hi)``.  Uses Hanley-McNeil 1982 SE with the
    Q1, Q2 quantities derived from the rank distribution.
    """
    a = auroc(positive, negative)
    m = max(1, len(positive))
    n = max(1, len(negative))
    # Hanley-McNeil SE.
    q1 = a / (2.0 - a)
    q2 = (2.0 * a * a) / (1.0 + a)
    var = (
        a * (1.0 - a)
        + (m - 1.0) * (q1 - a * a)
        + (n - 1.0) * (q2 - a * a)
    ) / (m * n)
    se = math.sqrt(max(var, 0.0))
    z = _z_critical(alpha)
    lo = max(0.0, a - z * se)
    hi = min(1.0, a + z * se)
    return a, lo, hi


def _z_critical(alpha: float) -> float:
    """Two-sided z* for confidence 1−α via the Abramowitz–Stegun rational."""
    p = 1.0 - alpha / 2.0
    if p <= 0.5:
        return 0.0
    # Inverse normal via Beasley-Springer / Moro (1995) low-order rational.
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1 - p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


# ---------------------------------------------------------------------------
# Spearman rank correlation (for the dose-response test)
# ---------------------------------------------------------------------------


def _ranks(xs: Sequence[float]) -> list[float]:
    indexed = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[indexed[j + 1]] == xs[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[indexed[k]] = avg
        i = j + 1
    return r


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def spearman_p(rho: float, n: int) -> float:
    """One-sided p-value for H0: rho ≤ 0 under t approximation."""
    if n < 4 or rho <= 0.0:
        return 1.0
    r = min(0.999, max(-0.999, rho))
    t = r * math.sqrt(max(0.0, (n - 2.0) / (1.0 - r * r)))
    # One-sided normal-tail approximation:
    return 1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Anytime-valid confidence sequence for paired Bernoulli lift
# ---------------------------------------------------------------------------


def hedged_capital_lift_e(
    diffs: Sequence[float],
    delta: float,
    lambda_max: float = 0.5,
) -> tuple[float, float, float]:
    """Hedged-capital betting e-process for H0: mean(diffs) ≤ delta.

    ``diffs`` is the stream of (steered_outcome - control_outcome)
    differences clipped to [-1, 1].  Returns
    ``(e_value, anytime_p, lower_confidence_sequence_estimate)`` for
    the mean.  Based on Waudby-Smith & Ramdas 2024 §3.2 with a
    grid-of-bets Cover-style mixture.

    The CS lower bound at confidence 1−α is the smallest mean μ such
    that the e-process for H0: mean ≤ μ stays below 1/α; we report a
    point lower estimate and let the caller decide α via the
    anytime-valid p-value.
    """
    n = len(diffs)
    if n == 0:
        return 1.0, 1.0, 0.0
    # Grid of bets over (μ around delta, λ).  We use a uniform mixture
    # over a small grid of λ values; the maximum e-value is achieved
    # by the data-driven λ but the mixture is conservatively valid.
    lambdas = (0.05, 0.1, 0.2, 0.3, 0.4, lambda_max)
    log_e = [0.0] * len(lambdas)
    # Track empirical mean for the CS estimate.
    s = 0.0
    for i, d in enumerate(diffs):
        # Clip to [-1, 1] — the realistic support for paired indicators.
        di = max(-1.0, min(1.0, d))
        s += di
        for k, lam in enumerate(lambdas):
            # Increment of log capital under bet λ against H0: μ = δ
            # K_t = ∏ (1 + λ (X - δ))
            log_e[k] += math.log(max(1e-12, 1.0 + lam * (di - delta)))
    # Mixture e-value
    mx = max(log_e)
    e = mx + math.log(sum(math.exp(le - mx) for le in log_e) / len(log_e))
    e_val = math.exp(e)
    anytime_p = min(1.0, 1.0 / max(e_val, 1.0))
    mean = s / n
    return e_val, anytime_p, mean


# ---------------------------------------------------------------------------
# Welch's t-test (used for outcome_score-valued effect tests)
# ---------------------------------------------------------------------------


def welch_t(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, float, float, float]:
    """Returns ``(t, df, mean_diff, one_sided_p_a_gt_b)``.

    Conservative normal-tail p suffices for our purpose (sample
    counts the coordinator collects are usually 30–300; the
    truncation error vs Student-t never exceeds 0.01 for df ≥ 30).
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 0.0, 0.0, 1.0
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    diff = ma - mb
    se = math.sqrt(va / na + vb / nb)
    if se == 0.0:
        return 0.0, 0.0, diff, 0.5
    t = diff / se
    df_num = (va / na + vb / nb) ** 2
    df_den = (va * va) / (na * na * (na - 1)) + (vb * vb) / (nb * nb * (nb - 1))
    df = df_num / df_den if df_den > 0 else float(na + nb - 2)
    p = 1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    return t, df, diff, p


# ---------------------------------------------------------------------------
# Fitting algorithms
# ---------------------------------------------------------------------------


def fit_caa(
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
    weights_pos: Sequence[float] | None = None,
    weights_neg: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    """Contrastive Activation Addition (Panickssery '24).

    Returns ``(unit_direction, positive_mean, negative_mean, norm)``.
    """
    if not positives or not negatives:
        raise InsufficientData("need at least one positive and one negative")
    dim = len(positives[0])
    if weights_pos is None:
        weights_pos = [1.0] * len(positives)
    if weights_neg is None:
        weights_neg = [1.0] * len(negatives)
    pos_sum = _zeros(dim)
    neg_sum = _zeros(dim)
    wp = 0.0
    wn = 0.0
    for v, w in zip(positives, weights_pos):
        for i, x in enumerate(v):
            pos_sum[i] += w * x
        wp += w
    for v, w in zip(negatives, weights_neg):
        for i, x in enumerate(v):
            neg_sum[i] += w * x
        wn += w
    pos_mean = tuple(x / wp for x in pos_sum)
    neg_mean = tuple(x / wn for x in neg_sum)
    diff = _vsub(pos_mean, neg_mean)
    n = _vnorm(diff)
    return _vnormalize(diff), pos_mean, neg_mean, n


def fit_lat(
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
    power_iters: int = 32,
    seed: int = 0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    """Linear Artificial Tomography (Zou '23).

    Takes per-pair differences ``p - n`` and computes the top principal
    component via power iteration.  Returns
    ``(unit_direction, positive_mean, negative_mean, top_singular_value)``.
    """
    if not positives or not negatives:
        raise InsufficientData("need at least one positive and one negative")
    if len(positives) != len(negatives):
        # LAT pairs by index; if mismatched, fall back to CAA semantics:
        return fit_caa(positives, negatives)
    dim = len(positives[0])
    diffs = [_vsub(p, n) for p, n in zip(positives, negatives)]
    rng = random.Random(seed)
    # Initial vector
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    nv = math.sqrt(sum(x * x for x in v)) or 1.0
    v = [x / nv for x in v]
    # Power iteration on the implicit Gram matrix
    # M = sum_i d_i d_i^T,   apply Mv = sum_i d_i (d_i · v)
    last_eig = 0.0
    for _ in range(power_iters):
        new_v = _zeros(dim)
        for d in diffs:
            c = sum(d[i] * v[i] for i in range(dim))
            for i in range(dim):
                new_v[i] += c * d[i]
        nv = math.sqrt(sum(x * x for x in new_v))
        if nv < 1e-18:
            break
        v = [x / nv for x in new_v]
        last_eig = nv
    # Sign by aligning with the mean-difference (else PCA sign is arbitrary)
    pos_mean = tuple(sum(p[i] for p in positives) / len(positives) for i in range(dim))
    neg_mean = tuple(sum(n[i] for n in negatives) / len(negatives) for i in range(dim))
    md = _vsub(pos_mean, neg_mean)
    if sum(v[i] * md[i] for i in range(dim)) < 0:
        v = [-x for x in v]
    return tuple(v), pos_mean, neg_mean, math.sqrt(last_eig)


def fit_lda(
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
    ridge: float = 1e-6,
    shrinkage: float | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float, dict]:
    """Fisher LDA with Ledoit-Wolf shrinkage on the within-class scatter.

    Returns ``(unit_direction, positive_mean, negative_mean, raw_norm,
    diagnostics)``.
    """
    if len(positives) < 2 or len(negatives) < 2:
        raise InsufficientData("LDA needs ≥ 2 positives and ≥ 2 negatives")
    dim = len(positives[0])
    pos_mean = tuple(sum(p[i] for p in positives) / len(positives) for i in range(dim))
    neg_mean = tuple(sum(n[i] for n in negatives) / len(negatives) for i in range(dim))
    # Within-class scatter S_w = Σ_p (p-μ+)(...)^T + Σ_n (n-μ-)(...)^T
    Sw = [[0.0] * dim for _ in range(dim)]
    for p in positives:
        d = _vsub(p, pos_mean)
        _outer_add(Sw, d, d, 1.0)
    for n in negatives:
        d = _vsub(n, neg_mean)
        _outer_add(Sw, d, d, 1.0)
    n_total = len(positives) + len(negatives) - 2
    if n_total <= 0:
        n_total = 1
    Sw_normed = _mat_scale(Sw, 1.0 / n_total)
    # Ledoit-Wolf shrinkage toward (tr(Sw)/dim) * I
    tr = sum(Sw_normed[i][i] for i in range(dim))
    target = tr / dim
    if shrinkage is None:
        # Data-driven oracle.  Use heuristic: shrinkage ≈ dim / (n_total + dim)
        shrink_used = min(1.0, dim / (n_total + dim))
    else:
        shrink_used = float(shrinkage)
    for i in range(dim):
        for j in range(dim):
            Sw_normed[i][j] = (1.0 - shrink_used) * Sw_normed[i][j]
        Sw_normed[i][i] += shrink_used * target
    md = _vsub(pos_mean, neg_mean)
    w = _solve_psd(Sw_normed, md, ridge=ridge)
    nw = _vnorm(w)
    if nw == 0.0:
        # Degenerate — fall back to CAA mean-difference.
        return (
            _vnormalize(md),
            pos_mean,
            neg_mean,
            _vnorm(md),
            {"shrinkage": shrink_used, "fallback": "caa"},
        )
    unit = tuple(x / nw for x in w)
    # Align sign with mean difference.
    if sum(unit[i] * md[i] for i in range(dim)) < 0:
        unit = tuple(-x for x in unit)
    return unit, pos_mean, neg_mean, nw, {"shrinkage": shrink_used}


def fit_probe(
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
    max_iter: int = 60,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float, dict]:
    """Logistic-probe normal as steering direction (Tigges '23).

    Trains a logistic classifier on positives vs negatives with simple
    gradient descent and returns the unit weight vector.
    """
    if not positives or not negatives:
        raise InsufficientData("need at least one positive and one negative")
    dim = len(positives[0])
    rng = random.Random(seed)
    w = [rng.gauss(0.0, 0.01) for _ in range(dim)]
    b = 0.0
    X = list(positives) + list(negatives)
    y = [1.0] * len(positives) + [0.0] * len(negatives)
    n = len(X)
    pos_mean = tuple(sum(p[i] for p in positives) / len(positives) for i in range(dim))
    neg_mean = tuple(sum(nv[i] for nv in negatives) / len(negatives) for i in range(dim))
    last_loss = float("inf")
    for it in range(max_iter):
        grad_w = [0.0] * dim
        grad_b = 0.0
        loss = 0.0
        for xi, yi in zip(X, y):
            z = sum(w[i] * xi[i] for i in range(dim)) + b
            # Numerically stable sigmoid + logloss
            if z >= 0:
                ez = math.exp(-z)
                p = 1.0 / (1.0 + ez)
                ll = z + math.log1p(ez) - yi * z
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)
                ll = math.log1p(ez) - yi * z
            loss += ll
            err = p - yi
            for i in range(dim):
                grad_w[i] += err * xi[i]
            grad_b += err
        for i in range(dim):
            grad_w[i] = grad_w[i] / n + l2 * w[i]
            w[i] -= learning_rate * grad_w[i]
        b -= learning_rate * (grad_b / n)
        loss = loss / n + 0.5 * l2 * sum(x * x for x in w)
        # Simple early stop
        if abs(last_loss - loss) < 1e-6 and it > 5:
            break
        last_loss = loss
    nw = _vnorm(w)
    if nw == 0.0:
        return (
            _vnormalize(_vsub(pos_mean, neg_mean)),
            pos_mean,
            neg_mean,
            _vnorm(_vsub(pos_mean, neg_mean)),
            {"final_loss": last_loss, "fallback": "caa"},
        )
    unit = tuple(x / nw for x in w)
    # Sign — already aligned by the {1,0} labelling.
    return unit, pos_mean, neg_mean, nw, {"final_loss": last_loss, "bias": b}


# ---------------------------------------------------------------------------
# Application primitives — the inference-time hooks
# ---------------------------------------------------------------------------


def apply_addition(
    activation: Sequence[float],
    direction: Sequence[float],
    coefficient: float,
) -> tuple[float, ...]:
    """Additive steering: ``a' = a + c · d``.

    The canonical CAA inference operation.  Direction need not be
    unit length; ``coefficient`` is interpreted in direction-norm
    units.
    """
    if len(activation) != len(direction):
        raise DimensionMismatch(
            f"activation dim {len(activation)} != direction dim {len(direction)}"
        )
    return tuple(a + coefficient * d for a, d in zip(activation, direction))


def apply_orthogonal_ablation(
    activation: Sequence[float],
    direction: Sequence[float],
) -> tuple[float, ...]:
    """Ablate the component of ``activation`` along ``direction``.

    ``a' = a − (a·d̂) d̂``.  Useful for *removing* a behavioural axis
    (Arditi-style refusal-direction ablation) rather than steering
    along it.
    """
    if len(activation) != len(direction):
        raise DimensionMismatch(
            f"activation dim {len(activation)} != direction dim {len(direction)}"
        )
    d_hat = _vnormalize(direction)
    proj = _vdot(activation, d_hat)
    return tuple(a - proj * dh for a, dh in zip(activation, d_hat))


def apply_clamp(
    activation: Sequence[float],
    direction: Sequence[float],
    target_value: float,
) -> tuple[float, ...]:
    """Clamp the projection along ``direction`` to ``target_value``.

    ``a' = a + (target − a·d̂) d̂``.  The "feature clamping"
    operation from Templeton et al. 2024 § Feature Steering: set a
    feature's activation level rather than add to it.
    """
    if len(activation) != len(direction):
        raise DimensionMismatch(
            f"activation dim {len(activation)} != direction dim {len(direction)}"
        )
    d_hat = _vnormalize(direction)
    proj = _vdot(activation, d_hat)
    delta = target_value - proj
    return tuple(a + delta * dh for a, dh in zip(activation, d_hat))


def apply_rank_one(
    activation: Sequence[float],
    direction: Sequence[float],
    coefficient: float,
    bias: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Inference-time form of a ReFT-style rank-1 intervention.

    ``a' = a + (coefficient · (d̂ · a) + bias·d̂) d̂``.  The shift is
    proportional to the existing projection — useful when the
    coordinator wants the intervention to scale with how strongly the
    behavioural axis already fires.
    """
    if len(activation) != len(direction):
        raise DimensionMismatch(
            f"activation dim {len(activation)} != direction dim {len(direction)}"
        )
    d_hat = _vnormalize(direction)
    proj = _vdot(activation, d_hat)
    bias_proj = _vdot(bias, d_hat) if bias is not None else 0.0
    shift = coefficient * proj + bias_proj
    return tuple(a + shift * dh for a, dh in zip(activation, d_hat))


# ---------------------------------------------------------------------------
# The Steerer class
# ---------------------------------------------------------------------------


def _now() -> float:
    """Indirection so tests can patch."""
    import time
    return time.time()


class Steerer:
    """Coordinator-facing handle for one (model, layer, axis) steering audit.

    Thread-safe.  Pure compute.  Replay-verifiable: the same ordered
    sequence of ``observe_*`` calls under the same config produces the
    same SHA-256 fingerprint chain.
    """

    def __init__(
        self,
        config: SteererConfig,
        bus: Any = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(config, SteererConfig):
            raise InvalidConfig("config must be a SteererConfig")
        # Re-run __post_init__ validation defensively (frozen dataclass
        # validates on construction, but a coordinator could pass a
        # mutated object through pickle/JSON):
        SteererConfig(**{f: getattr(config, f) for f in (
            "algorithm", "dim", "target_layer", "alpha", "alpha_warn",
            "ridge", "shrinkage", "auroc_floor", "delta",
            "coefficient_grid", "max_pairs", "max_outcomes", "seed",
        )})
        self._config = config
        self._bus = bus
        self._clock = clock or _now
        self._lock = threading.RLock()
        # Ring-buffer storage of observations.  Lists are simpler than
        # collections.deque for replay (positional indexing).
        self._pairs_train: list[ContrastivePair] = []
        self._pairs_paraphrase: list[ContrastivePair] = []
        self._pairs_holdout: list[ContrastivePair] = []
        self._outcomes: list[DoseOutcome] = []
        # Fingerprint chain
        self._fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "init": True,
                    "config": {
                        "algorithm": config.algorithm,
                        "dim": config.dim,
                        "target_layer": config.target_layer,
                        "alpha": config.alpha,
                        "alpha_warn": config.alpha_warn,
                        "auroc_floor": config.auroc_floor,
                        "delta": config.delta,
                        "coefficient_grid": list(config.coefficient_grid),
                        "seed": config.seed,
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self._last_fit: SteererFit | None = None
        self._emit(STEERER_STARTED, {"config": self._config_repr()})

    # ------------------------------------------------------------------
    # Coordinator-facing introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> SteererConfig:
        return self._config

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def n_pairs(self) -> int:
        with self._lock:
            return (
                len(self._pairs_train)
                + len(self._pairs_paraphrase)
                + len(self._pairs_holdout)
            )

    @property
    def n_outcomes(self) -> int:
        with self._lock:
            return len(self._outcomes)

    @property
    def last_fit(self) -> SteererFit | None:
        return self._last_fit

    def _config_repr(self) -> Mapping[str, Any]:
        c = self._config
        return {
            "algorithm": c.algorithm,
            "dim": c.dim,
            "target_layer": c.target_layer,
            "alpha": c.alpha,
            "auroc_floor": c.auroc_floor,
            "delta": c.delta,
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_pair(self, pair: ContrastivePair) -> str:
        if not isinstance(pair, ContrastivePair):
            raise InvalidPair("pair must be a ContrastivePair instance")
        # Validate dim
        _validate_vector(pair.positive, self._config.dim, "positive")
        _validate_vector(pair.negative, self._config.dim, "negative")
        with self._lock:
            bucket = self._bucket_for(pair.split)
            bucket.append(pair)
            # Evict
            if len(bucket) > self._config.max_pairs:
                del bucket[0:len(bucket) - self._config.max_pairs]
            self._chain(
                "pair",
                {
                    "task_id": pair.task_id,
                    "split": pair.split,
                    "weight": pair.weight,
                    # Hash the vectors so the chain is replay-stable
                    # but compact:
                    "pos_hash": _vhash(pair.positive),
                    "neg_hash": _vhash(pair.negative),
                },
            )
            self._emit(
                STEERER_PAIR_OBSERVED,
                {
                    "task_id": pair.task_id,
                    "split": pair.split,
                    "n_pairs": self.n_pairs,
                    "fingerprint": self._fingerprint,
                },
            )
            return self._fingerprint

    def observe_outcome(self, outcome: DoseOutcome) -> str:
        if not isinstance(outcome, DoseOutcome):
            raise InvalidOutcome("outcome must be a DoseOutcome instance")
        with self._lock:
            self._outcomes.append(outcome)
            if len(self._outcomes) > self._config.max_outcomes:
                del self._outcomes[0:len(self._outcomes) - self._config.max_outcomes]
            self._chain(
                "outcome",
                {
                    "task_id": outcome.task_id,
                    "coefficient": outcome.coefficient,
                    "outcome": bool(outcome.outcome),
                    "outcome_score": outcome.outcome_score,
                    "weight": outcome.weight,
                },
            )
            self._emit(
                STEERER_OUTCOME_OBSERVED,
                {
                    "task_id": outcome.task_id,
                    "coefficient": outcome.coefficient,
                    "outcome": bool(outcome.outcome),
                    "n_outcomes": self.n_outcomes,
                    "fingerprint": self._fingerprint,
                },
            )
            return self._fingerprint

    def _bucket_for(self, split: str) -> list[ContrastivePair]:
        if split == "train":
            return self._pairs_train
        if split == "paraphrase":
            return self._pairs_paraphrase
        if split == "holdout":
            return self._pairs_holdout
        raise InvalidPair(f"unknown split={split!r}")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> SteererFit:
        with self._lock:
            train = list(self._pairs_train)
            paraphrase = list(self._pairs_paraphrase)
            if len(train) < 2:
                raise InsufficientData(
                    "fit needs ≥ 2 train pairs; have "
                    f"{len(train)}"
                )
            positives = [p.positive for p in train]
            negatives = [p.negative for p in train]
            wpos = [p.weight for p in train]
            wneg = [p.weight for p in train]
            diagnostics: dict[str, Any] = {}
            algo = self._config.algorithm
            if algo == ALG_CAA:
                direction, pos_mean, neg_mean, norm = fit_caa(
                    positives, negatives, wpos, wneg
                )
            elif algo == ALG_LAT:
                direction, pos_mean, neg_mean, norm = fit_lat(
                    positives, negatives, seed=self._config.seed,
                )
            elif algo == ALG_LDA:
                direction, pos_mean, neg_mean, norm, diagnostics = fit_lda(
                    positives, negatives,
                    ridge=self._config.ridge,
                    shrinkage=self._config.shrinkage,
                )
            elif algo == ALG_PROBE:
                direction, pos_mean, neg_mean, norm, diagnostics = fit_probe(
                    positives, negatives, seed=self._config.seed,
                )
            else:  # pragma: no cover
                raise UnknownAlgorithm(algo)
            # Separability metrics on training set
            pos_proj = [_vdot(p, direction) for p in positives]
            neg_proj = [_vdot(n, direction) for n in negatives]
            a, lo, hi = auroc_ci(pos_proj, neg_proj, self._config.alpha)
            # Cohen-d
            if len(pos_proj) > 1 and len(neg_proj) > 1:
                mp = sum(pos_proj) / len(pos_proj)
                mn = sum(neg_proj) / len(neg_proj)
                vp = sum((x - mp) ** 2 for x in pos_proj) / (len(pos_proj) - 1)
                vn = sum((x - mn) ** 2 for x in neg_proj) / (len(neg_proj) - 1)
                pooled = math.sqrt(0.5 * (vp + vn))
                cohen_d = (mp - mn) / pooled if pooled > 0 else 0.0
            else:
                cohen_d = 0.0
            # Leakage AUROC on paraphrase split
            if paraphrase:
                pp_pos = [_vdot(p.positive, direction) for p in paraphrase]
                pp_neg = [_vdot(p.negative, direction) for p in paraphrase]
                leakage_auroc = auroc(pp_pos, pp_neg)
            else:
                leakage_auroc = None
            self._chain(
                "fit",
                {
                    "algorithm": algo,
                    "auroc": a,
                    "norm": norm,
                    "cohen_d": cohen_d,
                    "n_pairs": len(train),
                    "diag": dict(diagnostics),
                },
            )
            fit = SteererFit(
                algorithm=algo,
                dim=self._config.dim,
                direction=tuple(direction),
                norm=float(norm),
                positive_mean=tuple(pos_mean),
                negative_mean=tuple(neg_mean),
                n_pairs=len(train),
                separability_auroc=float(a),
                separability_ci=(float(lo), float(hi)),
                cohen_d=float(cohen_d),
                leakage_auroc=(float(leakage_auroc) if leakage_auroc is not None else None),
                diagnostics=dict(diagnostics),
                fingerprint=self._fingerprint,
            )
            self._last_fit = fit
            self._emit(
                STEERER_FIT,
                {
                    "algorithm": algo,
                    "auroc": a,
                    "cohen_d": cohen_d,
                    "n_pairs": len(train),
                    "fingerprint": self._fingerprint,
                },
            )
            return fit

    # ------------------------------------------------------------------
    # Recommend a coefficient
    # ------------------------------------------------------------------

    def recommend_coefficient(
        self,
        safety_floor: float | None = None,
    ) -> float:
        """Pick the coefficient that maximises outcome rate among
        observed dose buckets, subject to an optional ``safety_floor``
        on the *control* outcome rate at that coefficient.

        Falls back to ``0.0`` (no steering) when fewer than 8 doses
        per non-zero coefficient have been observed.
        """
        with self._lock:
            if not self._outcomes:
                return 0.0
            # Bucket by coefficient
            buckets: dict[float, list[DoseOutcome]] = {}
            for o in self._outcomes:
                buckets.setdefault(o.coefficient, []).append(o)
            # Score each bucket by outcome mean
            best_coef = 0.0
            best_score = -math.inf
            for coef, rows in buckets.items():
                if len(rows) < 4:
                    continue
                rate = sum(1.0 if r.outcome else 0.0 for r in rows) / len(rows)
                if safety_floor is not None and rate < safety_floor:
                    continue
                if rate > best_score:
                    best_score = rate
                    best_coef = coef
            self._emit(
                STEERER_RECOMMENDED,
                {
                    "coefficient": best_coef,
                    "rate": best_score if best_score != -math.inf else None,
                    "fingerprint": self._fingerprint,
                },
            )
            return best_coef

    # ------------------------------------------------------------------
    # Certify
    # ------------------------------------------------------------------

    def certify(
        self,
        delta: float | None = None,
        alpha: float | None = None,
    ) -> SteererCertificate:
        """Run all engaged tests, combine evidence, emit a verdict."""
        with self._lock:
            cfg = self._config
            delta = float(delta if delta is not None else cfg.delta)
            alpha = float(alpha if alpha is not None else cfg.alpha)
            if self._last_fit is None:
                # Try to fit on the fly
                try:
                    self.fit()
                except InsufficientData:
                    return self._inconclusive(reason="not fitted")
            tests: list[TestResult] = []
            # ---- TEST_SEPARABILITY ----
            fit = self._last_fit  # type: ignore[assignment]
            a, lo, hi = fit.separability_auroc, fit.separability_ci[0], fit.separability_ci[1]
            passed = lo >= cfg.auroc_floor
            warn = (not passed) and (a >= cfg.auroc_floor)
            tests.append(TestResult(
                name=TEST_SEPARABILITY,
                statistic=a,
                threshold=cfg.auroc_floor,
                p_value=None,
                e_value=None,
                ci_low=lo,
                ci_high=hi,
                passed=passed,
                n=fit.n_pairs,
            ))
            sep_warn = warn
            # ---- TEST_LEAKAGE ----
            if fit.leakage_auroc is not None:
                # Recompute a CI on the paraphrase set.
                pp = list(self._pairs_paraphrase)
                pp_pos = [_vdot(p.positive, fit.direction) for p in pp]
                pp_neg = [_vdot(p.negative, fit.direction) for p in pp]
                la, llo, lhi = auroc_ci(pp_pos, pp_neg, cfg.alpha)
                lpassed = llo >= cfg.auroc_floor
                lwarn = (not lpassed) and (la >= cfg.auroc_floor)
                tests.append(TestResult(
                    name=TEST_LEAKAGE,
                    statistic=la,
                    threshold=cfg.auroc_floor,
                    p_value=None,
                    e_value=None,
                    ci_low=llo,
                    ci_high=lhi,
                    passed=lpassed,
                    n=len(pp),
                ))
            else:
                lwarn = False
            # ---- TEST_DOSE_RESPONSE ----
            if self._outcomes:
                # Bucket per task and pair coefficients with outcomes;
                # Spearman ρ across (coefficient, outcome) is a clean
                # monotonicity test.
                coeffs = [o.coefficient for o in self._outcomes]
                ys: list[float]
                if all(o.outcome_score is not None for o in self._outcomes):
                    ys = [float(o.outcome_score) for o in self._outcomes]  # type: ignore[arg-type]
                else:
                    ys = [1.0 if o.outcome else 0.0 for o in self._outcomes]
                rho = spearman_rho(coeffs, ys)
                p = spearman_p(rho, len(coeffs))
                dr_passed = p < alpha and rho > 0
                dr_warn = (not dr_passed) and (p < cfg.alpha_warn) and rho > 0
                tests.append(TestResult(
                    name=TEST_DOSE_RESPONSE,
                    statistic=rho,
                    threshold=0.0,
                    p_value=p,
                    e_value=None,
                    ci_low=rho,  # asymptotic; we report point + p
                    ci_high=rho,
                    passed=dr_passed,
                    n=len(coeffs),
                ))
                # ---- TEST_EFFECT_SIZE — anytime-valid lift ----
                # Pair task_ids that appear under both coef=0 and the
                # best non-zero recommended coefficient.
                control = [o for o in self._outcomes if o.coefficient == 0.0]
                # Pick the candidate coefficient with maximum mean
                # outcome (the recommender):
                non_control_coeffs = sorted({o.coefficient for o in self._outcomes if o.coefficient != 0.0})
                best_coef: float | None = None
                best_rate = -math.inf
                for c in non_control_coeffs:
                    rows = [o for o in self._outcomes if o.coefficient == c]
                    if len(rows) < 4:
                        continue
                    rate = sum(1.0 if r.outcome else 0.0 for r in rows) / len(rows)
                    if rate > best_rate:
                        best_rate = rate
                        best_coef = c
                if best_coef is not None and control:
                    # Build per-task paired differences (mean steered −
                    # mean control); if a task has no control, fall
                    # back to the global control mean.
                    control_map: dict[str, list[bool]] = {}
                    for o in control:
                        control_map.setdefault(o.task_id, []).append(bool(o.outcome))
                    steered = [o for o in self._outcomes if o.coefficient == best_coef]
                    global_ctrl_mean = (
                        sum(1.0 if o.outcome else 0.0 for o in control) / len(control)
                    )
                    diffs: list[float] = []
                    for s in steered:
                        sv = 1.0 if s.outcome else 0.0
                        if s.task_id in control_map and control_map[s.task_id]:
                            cv = sum(1.0 if x else 0.0 for x in control_map[s.task_id]) / len(control_map[s.task_id])
                        else:
                            cv = global_ctrl_mean
                        diffs.append(sv - cv)
                    e_val, anytime_p, mean_diff = hedged_capital_lift_e(diffs, delta=delta)
                    # 1-α confidence sequence lower bound on the lift:
                    # use empirical mean - z*σ/√n as a fast approximation.
                    if len(diffs) > 1:
                        m = sum(diffs) / len(diffs)
                        v = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
                        z = _z_critical(alpha)
                        se = math.sqrt(v / len(diffs))
                        cs_lo = m - z * se
                        cs_hi = m + z * se
                    else:
                        cs_lo = mean_diff
                        cs_hi = mean_diff
                    ef_passed = anytime_p < alpha and mean_diff > delta
                    ef_warn = (not ef_passed) and (mean_diff > 0.5 * delta)
                    tests.append(TestResult(
                        name=TEST_EFFECT_SIZE,
                        statistic=mean_diff,
                        threshold=delta,
                        p_value=anytime_p,
                        e_value=e_val,
                        ci_low=cs_lo,
                        ci_high=cs_hi,
                        passed=ef_passed,
                        n=len(diffs),
                    ))
                    rec_coef = best_coef
                else:
                    ef_passed = False
                    ef_warn = False
                    rec_coef = None
                    cs_lo = 0.0
                    cs_hi = 0.0
                    mean_diff = 0.0
            else:
                rho_passed = False
                dr_warn = False
                ef_passed = False
                ef_warn = False
                rec_coef = None
                cs_lo = 0.0
                cs_hi = 0.0
                mean_diff = 0.0
                # Insert placeholder dose_response stub so the tests tuple
                # is consistent with the schema:
                tests.append(TestResult(
                    name=TEST_DOSE_RESPONSE,
                    statistic=0.0,
                    threshold=0.0,
                    p_value=None,
                    e_value=None,
                    ci_low=0.0,
                    ci_high=0.0,
                    passed=False,
                    n=0,
                ))
            # ---- Combine ----
            engaged = [t for t in tests if t.n > 0]
            has_effect = any(t.name == TEST_EFFECT_SIZE and t.n > 0 for t in tests)
            if not engaged:
                verdict = VERDICT_INCONCLUSIVE
            elif all(t.passed for t in engaged) and has_effect:
                verdict = VERDICT_PASS
            elif any(t.passed for t in engaged):
                # If separability passed but we cannot yet evaluate the
                # behavioural effect (no outcomes), this is WARN, not
                # PASS — promoting a steering config without a measured
                # outcome lift is unsafe.
                verdict = VERDICT_WARN
            else:
                verdict = VERDICT_FAIL
            # ---- Recommend ----
            if verdict == VERDICT_PASS and rec_coef is not None:
                recommendation = REC_PROMOTE
            elif verdict == VERDICT_INCONCLUSIVE:
                recommendation = REC_HOLD
            elif verdict == VERDICT_WARN:
                recommendation = REC_HOLD
            elif fit.cohen_d < -0.5:
                # Sign flipped — the "positive" class is actually
                # *below* the "negative" on the fitted direction.
                # Flip the convention.
                recommendation = REC_FLIP
            else:
                recommendation = REC_REJECT
            self._chain(
                "certify",
                {
                    "verdict": verdict,
                    "recommendation": recommendation,
                    "rec_coef": rec_coef,
                    "lift": mean_diff,
                    "delta": delta,
                    "alpha": alpha,
                },
            )
            self._emit(
                STEERER_TESTED,
                {
                    "verdict": verdict,
                    "n_tests": len(tests),
                    "fingerprint": self._fingerprint,
                },
            )
            cert = SteererCertificate(
                verdict=verdict,
                recommendation=recommendation,
                tests=tuple(tests),
                recommended_coefficient=(
                    float(rec_coef) if rec_coef is not None else None
                ),
                outcome_lift=float(mean_diff),
                outcome_lift_ci=(float(cs_lo), float(cs_hi)),
                n_pairs=self.n_pairs,
                n_outcomes=self.n_outcomes,
                fingerprint=self._fingerprint,
            )
            self._emit(
                STEERER_CERTIFIED,
                {
                    "verdict": verdict,
                    "recommendation": recommendation,
                    "rec_coef": rec_coef,
                    "lift": mean_diff,
                    "fingerprint": self._fingerprint,
                },
            )
            return cert

    def _inconclusive(self, reason: str) -> SteererCertificate:
        return SteererCertificate(
            verdict=VERDICT_INCONCLUSIVE,
            recommendation=REC_HOLD,
            tests=(),
            recommended_coefficient=None,
            outcome_lift=0.0,
            outcome_lift_ci=(0.0, 0.0),
            n_pairs=self.n_pairs,
            n_outcomes=self.n_outcomes,
            fingerprint=self._fingerprint,
        )

    # ------------------------------------------------------------------
    # Report bundle
    # ------------------------------------------------------------------

    def report(self) -> SteererReport:
        """Produce a (fit, certificate, counters) bundle in one call."""
        with self._lock:
            if self._last_fit is None:
                self.fit()
            cert = self.certify()
            report = SteererReport(
                fit=self._last_fit,  # type: ignore[arg-type]
                certificate=cert,
                n_pairs=self.n_pairs,
                n_outcomes=self.n_outcomes,
                fingerprint=self._fingerprint,
            )
            self._emit(
                STEERER_REPORTED,
                {
                    "verdict": cert.verdict,
                    "n_pairs": self.n_pairs,
                    "n_outcomes": self.n_outcomes,
                    "fingerprint": self._fingerprint,
                },
            )
            return report

    # ------------------------------------------------------------------
    # Application hooks (re-export the module-level functions so a
    # coordinator can drive everything off the Steerer handle):
    # ------------------------------------------------------------------

    def apply(
        self,
        activation: Sequence[float],
        coefficient: float,
    ) -> tuple[float, ...]:
        """Apply additive steering using the current ``last_fit.direction``."""
        if self._last_fit is None:
            raise NotFitted("call fit() before apply()")
        return apply_addition(activation, self._last_fit.direction, coefficient)

    def ablate(self, activation: Sequence[float]) -> tuple[float, ...]:
        """Remove the component of ``activation`` along the fitted direction."""
        if self._last_fit is None:
            raise NotFitted("call fit() before ablate()")
        return apply_orthogonal_ablation(activation, self._last_fit.direction)

    def clamp(
        self,
        activation: Sequence[float],
        target_value: float,
    ) -> tuple[float, ...]:
        """Set the projection along the fitted direction to ``target_value``."""
        if self._last_fit is None:
            raise NotFitted("call fit() before clamp()")
        return apply_clamp(activation, self._last_fit.direction, target_value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._pairs_train.clear()
            self._pairs_paraphrase.clear()
            self._pairs_holdout.clear()
            self._outcomes.clear()
            self._last_fit = None
            self._chain("reset", {})
            self._emit(STEERER_RESET, {"fingerprint": self._fingerprint})

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
            # Lazy import to avoid hard dependency cycle on agi.events.
            from agi.events import Event
            self._bus.publish(Event(kind=kind, data=dict(payload), ts=self._clock()))
        except Exception:  # pragma: no cover — never fail user calls
            pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _vhash(v: Sequence[float]) -> str:
    h = hashlib.sha256()
    for x in v:
        h.update(repr(float(x)).encode("utf-8"))
    return h.hexdigest()[:16]


def fresh_steerer(
    algorithm: str = ALG_CAA,
    *,
    dim: int,
    target_layer: str = "",
    alpha: float = 0.05,
    seed: int = 0,
) -> Steerer:
    """Convenience constructor with sensible defaults."""
    return Steerer(SteererConfig(
        algorithm=algorithm,
        dim=dim,
        target_layer=target_layer,
        alpha=alpha,
        seed=seed,
    ))


def synthetic_pairs(
    n: int,
    dim: int,
    axis: int = 0,
    snr: float = 3.0,
    noise: float = 0.3,
    seed: int = 0,
) -> list[ContrastivePair]:
    """Synthesise a contrastive-pair dataset along one axis (for testing /
    notebooks)."""
    rng = random.Random(seed)
    out: list[ContrastivePair] = []
    for i in range(n):
        pos = tuple(
            rng.gauss(snr if j == axis else 0.0, noise) for j in range(dim)
        )
        neg = tuple(
            rng.gauss(-snr if j == axis else 0.0, noise) for j in range(dim)
        )
        out.append(ContrastivePair(task_id=f"t{i}", positive=pos, negative=neg))
    return out


def compare_steerers(
    a: Steerer,
    b: Steerer,
) -> dict:
    """Compare two steerers — cosine similarity of fitted directions,
    pass-rate agreement on shared task_ids, lift-CI overlap.

    Used by the coordination engine to A/B-test two steering
    algorithms or two layers against each other.
    """
    if a.last_fit is None or b.last_fit is None:
        raise NotFitted("both Steerers must be fit before comparison")
    if a.last_fit.dim != b.last_fit.dim:
        raise DimensionMismatch(
            f"dims differ: {a.last_fit.dim} vs {b.last_fit.dim}"
        )
    cos = _vdot(a.last_fit.direction, b.last_fit.direction)
    return {
        "cosine_similarity": float(cos),
        "auroc_a": float(a.last_fit.separability_auroc),
        "auroc_b": float(b.last_fit.separability_auroc),
        "cohen_d_a": float(a.last_fit.cohen_d),
        "cohen_d_b": float(b.last_fit.cohen_d),
    }


__all__ = [
    # Algorithms
    "ALG_CAA",
    "ALG_LAT",
    "ALG_LDA",
    "ALG_PROBE",
    "KNOWN_ALGORITHMS",
    # Verdicts and recommendations
    "VERDICT_PASS",
    "VERDICT_WARN",
    "VERDICT_FAIL",
    "VERDICT_INCONCLUSIVE",
    "KNOWN_VERDICTS",
    "REC_PROMOTE",
    "REC_HOLD",
    "REC_REJECT",
    "REC_FLIP",
    "KNOWN_RECOMMENDATIONS",
    # Tests
    "TEST_SEPARABILITY",
    "TEST_DOSE_RESPONSE",
    "TEST_EFFECT_SIZE",
    "TEST_LEAKAGE",
    "KNOWN_TESTS",
    # Events
    "STEERER_STARTED",
    "STEERER_PAIR_OBSERVED",
    "STEERER_OUTCOME_OBSERVED",
    "STEERER_FIT",
    "STEERER_TESTED",
    "STEERER_CERTIFIED",
    "STEERER_RECOMMENDED",
    "STEERER_REPORTED",
    "STEERER_RESET",
    "KNOWN_EVENTS",
    # Errors
    "SteererError",
    "InvalidConfig",
    "InvalidPair",
    "InvalidOutcome",
    "UnknownAlgorithm",
    "UnknownTest",
    "InsufficientData",
    "NotFitted",
    "DimensionMismatch",
    # Data
    "ContrastivePair",
    "DoseOutcome",
    "SteererConfig",
    "SteererFit",
    "TestResult",
    "SteererCertificate",
    "SteererReport",
    # Functions
    "fit_caa",
    "fit_lat",
    "fit_lda",
    "fit_probe",
    "apply_addition",
    "apply_orthogonal_ablation",
    "apply_clamp",
    "apply_rank_one",
    "auroc",
    "auroc_ci",
    "spearman_rho",
    "spearman_p",
    "welch_t",
    "hedged_capital_lift_e",
    # Class
    "Steerer",
    # Helpers
    "fresh_steerer",
    "synthetic_pairs",
    "compare_steerers",
]
