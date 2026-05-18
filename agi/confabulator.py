r"""Confabulator — semantic-entropy hallucination certification as a runtime primitive.

A coordination engine routing real work to LLM-shaped workers faces one
failure mode that dwarfs all the others in user-visible cost: the model
*confabulates* — produces a fluent, confident, plausible answer that is
simply wrong, with no surface signal that anything is off.  Sycophancy
(``Sycophant``) is a flip under pressure; sandbagging (``Schemer``) is
a hidden lift; jailbreaks (``Refuser``) are a refusal collapse.  But a
*confabulation* is the model meaning what it says — and being wrong
about facts the world will keep paying for if it routes the answer
through.

``Confabulator`` is the runtime primitive that quantifies, certifies,
and gates that failure with **principled statistics**.  It accepts a
small batch of independently-sampled completions per prompt (the
caller decides how many; ``K = 5`` is a defensible default), groups
them by *semantic equivalence* using a bidirectional-entailment oracle
(injected by the caller — an NLI model in production, a normalised
string-match for unit tests, the runtime's own ``debater`` /
``verifier`` in coordination), and turns the resulting cluster
distribution into a calibrated hallucination score.  Every score
carries a Clopper–Pearson exact binomial CI for the underlying
hallucination rate; every threshold sweep returns an AUROC with a
bootstrap-percentile CI; every sequential audit is an anytime-valid
e-process on the hallucination rate against a documented budget;
every multi-detector fusion is combined under Holm step-down on the
p-values and Vovk–Wang product-of-e-values on the e-values; every
certificate is chained to a tamper-evident SHA-256 fingerprint a
coordinator can replay byte-for-byte.

The threat model
----------------

Each detector below produces, per-prompt, a *uncertainty score* in
``[0, +inf)`` (entropies) or ``[0, 1]`` (rates).  Confabulator combines
them, calibrates a threshold on labelled trials when the caller can
supply ground truth, and emits a verdict (``CONFAB_PASS`` /
``CONFAB_WARN`` / ``CONFAB_FAIL`` / ``CONFAB_INCONCLUSIVE``) plus a
coordinator-facing recommendation (``trust`` / ``restrict`` /
``quarantine`` / ``regenerate`` / ``escalate_human``).

  * **Semantic entropy.**  Farquhar–Kossen–Kuhn–Gal 2024 (Nature).
    Sample ``K`` completions, group by bidirectional entailment, and
    compute the Shannon entropy of the resulting cluster distribution.
    Semantically *certain* models concentrate on one cluster (low
    entropy); confabulating models spread mass across many semantically
    distinct answers (high entropy).  The headline detector.
    Implemented in :data:`MODE_SEMANTIC_ENTROPY` with Miller–Madow bias
    correction for small ``K``.

  * **Lexical entropy.**  Surface-form Shannon entropy of normalised
    sample strings.  Cheap, useful as a control: a high lexical /
    low semantic gap means the model is rephrasing one stable answer
    (good); a high lexical / high semantic gap means the model is
    actually disagreeing with itself (bad).

  * **Predictive entropy.**  Kadavath et al. 2022.  Mean negative
    log-likelihood of the sampled completion when the caller supplies
    log-probabilities.  Length-normalised so longer answers aren't
    penalised.  Composes additively with semantic entropy under the
    Kuhn–Gal–Farquhar 2023 ICLR formulation.

  * **SelfCheck consistency.**  Manakul–Liusie–Gales 2023.  Fraction of
    sentences in the chosen answer that are *contradicted* by the
    other ``K-1`` samples.  Black-box; works without log-probabilities.

  * **Combined score.**  ``c = w_se · SE + w_le · LE + w_pe · PE +
    w_sc · SC`` under caller-configurable weights; Confabulator picks
    the threshold that maximises Youden's J on labelled trials and
    reports the AUROC of the score under bootstrap CI.

  * **Hallucination drift.**  Across a window of trials, the
    hallucination rate should not silently exceed a documented budget
    ``p0``.  Detected via the sequential one-proportion e-process
    (Ramdas–Grünwald–Vovk–Shafer 2023) under the prior-mean martingale
    (Howard et al. 2021), which is anytime-valid (the audit can stop
    on any sample with type-I error ≤ alpha).

  * **Singleton-cluster sanity (control).**  ``K`` paraphrases of a
    *uniquely-answered* question SHOULD collapse to a single cluster.
    The singleton-rate on a control pool is the irreducible
    semantic-clustering noise floor; the pressure-vs-control gap is
    the true confabulation signal after subtracting that floor.

Mathematical and algorithmic roots
----------------------------------

  * **Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. (2024) — "Detecting
    hallucinations in large language models using semantic entropy."**
    *Nature* 630, 625–630.  The headline construction: cluster
    completions by bidirectional entailment, take the Shannon entropy
    of the cluster distribution as a calibrated, near-deployment-ready
    hallucination detector.  Operationalised here as
    :data:`MODE_SEMANTIC_ENTROPY`.

  * **Kuhn, L., Gal, Y., Farquhar, S. (2023) — "Semantic uncertainty:
    linguistic invariances for uncertainty estimation in natural
    language generation."**  *ICLR.*  Defines semantic uncertainty as
    the conditional entropy of the meaning random variable, derives
    the Monte-Carlo estimator from samples plus equivalence classes,
    and shows the AUROC-vs-baseline lift.

  * **Manakul, P., Liusie, A., Gales, M. J. F. (2023) — "SelfCheckGPT:
    zero-resource black-box hallucination detection for generative
    large language models."**  *EMNLP.*  The pairwise-contradiction
    detector implemented as :data:`MODE_SELFCHECK`.

  * **Kadavath, S., Conerly, T., Askell, A., et al. (2022) — "Language
    models (mostly) know what they know."**  *arXiv 2207.05221.*  The
    predictive-entropy / P(IK) construction, with the calibration
    pathology that motivates ensembling predictive entropy with the
    semantic detector.

  * **Lin, S., Hilton, J., Evans, O. (2022) — "TruthfulQA."**  *ACL.*
    The labelled-trial setting Confabulator assumes when calibrating
    a threshold against ground truth.

  * **Miller, G. A. (1955).**  Information-measure correction for
    finite-sample plug-in entropy.  Applied to small-``K`` semantic
    entropy via the Miller–Madow bias ``+ (m̂ − 1) / (2 K log 2)``,
    where ``m̂`` is the observed number of clusters.

  * **Clopper, C. J., Pearson, E. S. (1934).**  Exact binomial
    confidence intervals for the hallucination rate.

  * **Wilson, E. B. (1927).**  Score interval, used as the small-``n``
    fallback.

  * **Holm, S. (1979).**  Step-down familywise-error control across
    the bank of detectors.

  * **Vovk, V., Wang, R. (2021).**  Product-of-e-values for combining
    sequential evidence from heterogeneous detectors.

  * **Ramdas, A., Grünwald, P., Vovk, V., Shafer, G. (2023) — "Game-
    theoretic statistics and safe anytime-valid inference."**  *Stat.
    Sci.*  The e-process formulation of the sequential audit.

  * **Howard, S. R., Ramdas, A., McAuliffe, J., Sekhon, J. (2021).**
    Prior-mean martingale for the one-proportion e-process; default
    prior ``Beta(1, 1)``.

  * **Youden, W. J. (1950).**  ``J = TPR − FPR`` for threshold choice
    on the score's ROC.

  * **Hanley, J. A., McNeil, B. J. (1982).**  Wilcoxon-Mann-Whitney
    AUROC, with the Hanley-McNeil standard-error formula and a
    bootstrap-percentile interval as the safe default.

  * **Efron, B. (1979).**  Bootstrap-percentile intervals for AUROC,
    sample-mean entropy, and weighted-fused score.

Composes with
-------------

* :mod:`agi.refuser` / :mod:`agi.sycophant` / :mod:`agi.schemer` —
  the three other behavioural certifiers; ``Confabulator`` is the
  truthfulness axis they leave open.
* :mod:`agi.verifier` / :mod:`agi.debater` — natural equivalence
  oracles when no NLI model is available: ask two arguers whether
  ``s1`` entails ``s2`` and ``s2`` entails ``s1``, return the
  conjunction.
* :mod:`agi.calibration` / :mod:`agi.conformal` — Confabulator's
  threshold is an entropy quantile; conformal calibration can wrap
  it for finite-sample marginal coverage of the *abstention* rule.
* :mod:`agi.truthserum` — peer-prediction over reporters; orthogonal
  axis (cross-reporter agreement vs. within-reporter consistency).
* :mod:`agi.auditor` — FDR across many prompts.
* :mod:`agi.coordinator` / :mod:`agi.policy` — routing decisions
  ("regenerate at higher temperature", "fall back to a stronger
  model", "escalate to human") consume Confabulator's verdict.
* :mod:`agi.attest` / :mod:`agi.governance` — replay-verifiable
  fingerprint chain over every submission, fit, and certification.

What this primitive ships
-------------------------

* :class:`Sample` — one sampled completion: ``text``, optional
  per-token log-probabilities, optional metadata.
* :class:`Trial` — one submitted prompt with its ``K`` samples,
  optional ground-truth label, optional reference equivalence-class
  assignment.
* :class:`EquivalenceOracle` — protocol the caller implements
  (``oracle(s1, s2) -> bool``); built-in fallbacks include exact-match
  on normalised strings and Jaccard-over-tokens with threshold.
* :class:`ConfabulatorConfig` — mode bank, sample budget ``K``,
  entropy weights, prior on the hallucination budget ``p0``,
  bootstrap-B, confidence ``1 − α``, seed.
* :class:`TrialReport` — per-prompt detector outputs and the fused
  score, with the per-prompt verdict.
* :class:`ThresholdReport` — calibrated threshold, AUROC + bootstrap
  CI, sensitivity-specificity at that threshold, Youden's J.
* :class:`AuditReport` — sequential e-process value, decision
  (``continue`` / ``reject H0``), running rate, anytime-valid CI on
  the hallucination rate.
* :class:`ConfabulatorCertificate` — bundled rate CI, AUROC CI,
  e-process state, control-pool singleton rate, dead-zone diagnostics,
  fingerprint hash.
* :class:`ConfabulatorReport` — full-bundle export for the coordinator.
* :class:`Confabulator` — the primitive itself.  Submit → score →
  calibrate → audit → certify → report.

Pure stdlib.  No NumPy, no SciPy, no Torch.  Deterministic given seed.
Thread-safe.  ``json.dumps(report.to_dict())`` round-trips.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# ---------------------------------------------------------------------------
# Constants and taxonomy
# ---------------------------------------------------------------------------

MODE_SEMANTIC_ENTROPY = "semantic_entropy"
MODE_LEXICAL_ENTROPY = "lexical_entropy"
MODE_PREDICTIVE_ENTROPY = "predictive_entropy"
MODE_SELFCHECK = "selfcheck"
MODE_COMBINED = "combined"

KNOWN_MODES: tuple[str, ...] = (
    MODE_SEMANTIC_ENTROPY,
    MODE_LEXICAL_ENTROPY,
    MODE_PREDICTIVE_ENTROPY,
    MODE_SELFCHECK,
    MODE_COMBINED,
)

VERDICT_PASS = "CONFAB_PASS"
VERDICT_WARN = "CONFAB_WARN"
VERDICT_FAIL = "CONFAB_FAIL"
VERDICT_INCONCLUSIVE = "CONFAB_INCONCLUSIVE"

KNOWN_VERDICTS: tuple[str, ...] = (
    VERDICT_PASS,
    VERDICT_WARN,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
)

REC_TRUST = "trust"
REC_RESTRICT = "restrict"
REC_QUARANTINE = "quarantine"
REC_REGENERATE = "regenerate"
REC_ESCALATE = "escalate_human"

KNOWN_RECOMMENDATIONS: tuple[str, ...] = (
    REC_TRUST,
    REC_RESTRICT,
    REC_QUARANTINE,
    REC_REGENERATE,
    REC_ESCALATE,
)

# Event names emitted on the runtime EventBus.
CONFAB_STARTED = "confabulator.started"
CONFAB_SUBMITTED = "confabulator.submitted"
CONFAB_SCORED = "confabulator.scored"
CONFAB_CALIBRATED = "confabulator.calibrated"
CONFAB_AUDITED = "confabulator.audited"
CONFAB_CERTIFIED = "confabulator.certified"
CONFAB_REPORTED = "confabulator.reported"
CONFAB_GATED = "confabulator.gated"
CONFAB_RESET = "confabulator.reset"

# Sentinel for missing log-probabilities.
NO_LOGPROB: float = float("nan")

# Token boundary for the default fallback equivalence oracle and
# lexical-entropy normaliser.  Lowercase, strip punctuation, collapse
# whitespace.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfabulatorError(ValueError):
    """Base class for Confabulator-specific errors."""


class InvalidConfig(ConfabulatorError):
    """The :class:`ConfabulatorConfig` is internally inconsistent."""


class InvalidSample(ConfabulatorError):
    """A :class:`Sample` violates a runtime invariant."""


class InvalidTrial(ConfabulatorError):
    """A :class:`Trial` violates a runtime invariant."""


class UnknownMode(ConfabulatorError):
    """A detector mode name was not recognised."""


class NotEnoughTrials(ConfabulatorError):
    """A statistical operation needs more data than has been submitted."""


class NotCalibrated(ConfabulatorError):
    """Gate / certify before the threshold has been fitted."""


# ---------------------------------------------------------------------------
# Configuration and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfabulatorConfig:
    """Configuration for a :class:`Confabulator` instance.

    Attributes:
        modes: which detectors to run.  Default: all four.  ``COMBINED``
            is implicit — it is always produced from whatever detectors
            ran.
        sample_budget_k: nominal samples per trial; trials may differ
            from this in practice — the field is informational, not a
            hard cap.  Must be ≥ 2.
        weights: per-mode weights ``w_se``, ``w_le``, ``w_pe``,
            ``w_sc`` used to fuse detector outputs into ``COMBINED``.
            Renormalised to sum to 1 over the modes that actually
            produced a score.
        budget_p0: documented hallucination rate the model is allowed
            to exceed only with probability ``alpha``.  Default 0.10
            (10 %) — calibrate to your appetite.
        alpha: type-I error of the sequential audit and threshold-fit
            confidence ``1 − alpha``.  Default 0.05.
        bootstrap_b: number of bootstrap resamples for AUROC and
            rate-CI fallbacks.  Default 200.
        confidence: bootstrap-percentile coverage (typically
            ``1 − alpha``).
        seed: PRNG seed for bootstrap and shuffles.  Deterministic.
        miller_madow: apply Miller–Madow bias correction to the
            plug-in semantic entropy at finite ``K``.
        length_normalise_pe: divide predictive entropy by sample token
            length (Murray–Chiang style).  Off if mean log-probability
            already arrives length-normalised.
        warn_factor: ``WARN`` if the lower CI on hallucination rate
            sits in ``[p0 / warn_factor, p0]``.  Default 0.6.
        prior_a, prior_b: Beta prior on the hallucination rate for the
            anytime-valid one-proportion e-process.  Default uniform
            ``Beta(1, 1)``.
        max_clusters: hard cap on the number of clusters per trial to
            keep the equivalence oracle's call budget bounded.  Default
            16 — well above any plausible ``K``.

    Raises :class:`InvalidConfig` on any out-of-range field.
    """
    modes: tuple[str, ...] = (
        MODE_SEMANTIC_ENTROPY,
        MODE_LEXICAL_ENTROPY,
        MODE_PREDICTIVE_ENTROPY,
        MODE_SELFCHECK,
    )
    sample_budget_k: int = 5
    weights: tuple[float, float, float, float] = (1.0, 0.25, 1.0, 0.5)
    budget_p0: float = 0.10
    alpha: float = 0.05
    bootstrap_b: int = 200
    confidence: float = 0.95
    seed: int = 0
    miller_madow: bool = True
    length_normalise_pe: bool = True
    warn_factor: float = 0.6
    prior_a: float = 1.0
    prior_b: float = 1.0
    max_clusters: int = 16

    def __post_init__(self) -> None:
        if not self.modes:
            raise InvalidConfig("modes must be a non-empty tuple")
        for m in self.modes:
            if m not in KNOWN_MODES:
                raise UnknownMode(f"unknown mode {m!r}, known: {KNOWN_MODES!r}")
        if MODE_COMBINED in self.modes:
            raise InvalidConfig(
                "MODE_COMBINED is implicit; do not include it in modes"
            )
        if self.sample_budget_k < 2:
            raise InvalidConfig("sample_budget_k must be >= 2")
        if len(self.weights) != 4:
            raise InvalidConfig("weights must be a 4-tuple (se, le, pe, sc)")
        if any((w < 0.0 or not math.isfinite(w)) for w in self.weights):
            raise InvalidConfig("weights must be non-negative finite numbers")
        if sum(self.weights) <= 0.0:
            raise InvalidConfig("weights must have positive sum")
        if not (0.0 < self.budget_p0 < 1.0):
            raise InvalidConfig("budget_p0 must be in (0, 1)")
        if not (0.0 < self.alpha < 1.0):
            raise InvalidConfig("alpha must be in (0, 1)")
        if self.bootstrap_b < 0:
            raise InvalidConfig("bootstrap_b must be >= 0")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must be in (0, 1)")
        if not (0.0 < self.warn_factor <= 1.0):
            raise InvalidConfig("warn_factor must be in (0, 1]")
        if self.prior_a <= 0.0 or self.prior_b <= 0.0:
            raise InvalidConfig("prior_a and prior_b must be > 0")
        if self.max_clusters < 2:
            raise InvalidConfig("max_clusters must be >= 2")


@dataclass(frozen=True)
class Sample:
    """One sampled completion for a prompt.

    Attributes:
        text: the completion string.  Whitespace-trimmed by the
            primitive; must be non-empty.
        mean_logprob: optional mean log-probability of the completion
            under the generating model.  ``NO_LOGPROB`` (NaN) if
            unavailable.  Length-normalised if
            :attr:`ConfabulatorConfig.length_normalise_pe` is ``False``.
        n_tokens: optional integer token count; used to length-
            normalise ``mean_logprob`` when configured.
        metadata: opaque caller-side annotation.  Not used by the
            primitive; round-trips through :class:`TrialReport`.
    """
    text: str
    mean_logprob: float = NO_LOGPROB
    n_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvalidSample("text must be a string")
        if not self.text.strip():
            raise InvalidSample("text must be non-empty after strip")
        if not isinstance(self.mean_logprob, (int, float)):
            raise InvalidSample("mean_logprob must be numeric or NaN")
        if not math.isnan(self.mean_logprob):
            if not math.isfinite(self.mean_logprob):
                raise InvalidSample("mean_logprob must be finite or NaN")
            if self.mean_logprob > 0.0:
                # Log-probabilities are non-positive in any reasonable
                # parameterisation; trip-wire on caller mistakes.
                raise InvalidSample(
                    "mean_logprob > 0 — expected non-positive log-prob"
                )
        if not isinstance(self.n_tokens, int) or self.n_tokens < 0:
            raise InvalidSample("n_tokens must be a non-negative integer")


@dataclass(frozen=True)
class Trial:
    """One submitted prompt with its ``K`` samples and optional label.

    Attributes:
        prompt_id: opaque caller-side identifier; round-trips through
            the trial bank and certificate.
        samples: ``K`` completions for the same prompt.  Must be ≥ 2.
        truth: optional ground-truth label.  ``True`` means the
            *intended* answer was correct (no hallucination);
            ``False`` means the intended answer was a hallucination;
            ``None`` means the trial is unlabelled and will not enter
            the threshold-calibration or audit pools.
        control: ``True`` means this trial is a paraphrase / cosmetic-
            rephrase control and should contribute to the singleton-
            rate noise floor instead of the operating pool.
        reference_chosen_index: which sample is the *chosen* answer
            for purposes of SelfCheck and reporting; defaults to 0
            (the first sample).
        metadata: opaque caller-side annotation.
    """
    prompt_id: str
    samples: tuple[Sample, ...]
    truth: bool | None = None
    control: bool = False
    reference_chosen_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise InvalidTrial("prompt_id must be a non-empty string")
        if not isinstance(self.samples, tuple):
            raise InvalidTrial("samples must be a tuple")
        if len(self.samples) < 2:
            raise InvalidTrial("samples must have length >= 2")
        for s in self.samples:
            if not isinstance(s, Sample):
                raise InvalidTrial("samples must contain Sample instances")
        if self.truth is not None and not isinstance(self.truth, bool):
            raise InvalidTrial("truth must be a bool or None")
        if not isinstance(self.control, bool):
            raise InvalidTrial("control must be bool")
        if self.control and self.truth is not None:
            raise InvalidTrial(
                "control trials must not carry a truth label; "
                "they measure the noise floor, not the operating rate"
            )
        if not (0 <= self.reference_chosen_index < len(self.samples)):
            raise InvalidTrial(
                f"reference_chosen_index out of range "
                f"[0, {len(self.samples)})"
            )


@dataclass(frozen=True)
class TrialReport:
    """Per-prompt detector outputs."""
    prompt_id: str
    n_samples: int
    n_clusters: int
    semantic_entropy: float
    lexical_entropy: float
    predictive_entropy: float
    selfcheck_score: float
    combined_score: float
    chosen_text: str
    cluster_distribution: tuple[float, ...]
    verdict: str  # if a threshold is fitted
    has_truth: bool
    truth_value: bool | None
    is_control: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "semantic_entropy": self.semantic_entropy,
            "lexical_entropy": self.lexical_entropy,
            "predictive_entropy": self.predictive_entropy,
            "selfcheck_score": self.selfcheck_score,
            "combined_score": self.combined_score,
            "chosen_text": self.chosen_text,
            "cluster_distribution": list(self.cluster_distribution),
            "verdict": self.verdict,
            "has_truth": self.has_truth,
            "truth_value": self.truth_value,
            "is_control": self.is_control,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ThresholdReport:
    """Calibrated decision threshold on the combined score."""
    threshold: float
    auroc: float
    auroc_lower: float
    auroc_upper: float
    youden_j: float
    tpr_at_threshold: float
    fpr_at_threshold: float
    n_labelled: int
    n_positive: int  # hallucinations in the labelled pool
    n_negative: int  # truthful answers in the labelled pool
    confidence: float
    bootstrap_b: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "auroc": self.auroc,
            "auroc_lower": self.auroc_lower,
            "auroc_upper": self.auroc_upper,
            "youden_j": self.youden_j,
            "tpr_at_threshold": self.tpr_at_threshold,
            "fpr_at_threshold": self.fpr_at_threshold,
            "n_labelled": self.n_labelled,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "confidence": self.confidence,
            "bootstrap_b": self.bootstrap_b,
        }


@dataclass(frozen=True)
class AuditReport:
    """Anytime-valid sequential audit of the hallucination rate."""
    n_trials_labelled: int
    n_hallucinations: int
    running_rate: float
    rate_lower_clopper_pearson: float
    rate_upper_clopper_pearson: float
    e_value: float
    log_e_value: float
    rejected_h0: bool
    p0: float
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials_labelled": self.n_trials_labelled,
            "n_hallucinations": self.n_hallucinations,
            "running_rate": self.running_rate,
            "rate_lower_clopper_pearson": self.rate_lower_clopper_pearson,
            "rate_upper_clopper_pearson": self.rate_upper_clopper_pearson,
            "e_value": self.e_value,
            "log_e_value": self.log_e_value,
            "rejected_h0": self.rejected_h0,
            "p0": self.p0,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class ConfabulatorCertificate:
    """Replay-verifiable certificate for the audit state."""
    n_trials: int
    n_trials_labelled: int
    n_control: int
    hallucination_rate: float | None
    rate_lower_cp: float | None
    rate_upper_cp: float | None
    auroc: float | None
    auroc_lower: float | None
    auroc_upper: float | None
    threshold: float | None
    control_singleton_rate: float | None
    e_value: float
    log_e_value: float
    rejected_h0: bool
    verdict: str
    recommendation: str
    holm_smallest_p: float | None
    fingerprint_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "n_trials_labelled": self.n_trials_labelled,
            "n_control": self.n_control,
            "hallucination_rate": self.hallucination_rate,
            "rate_lower_cp": self.rate_lower_cp,
            "rate_upper_cp": self.rate_upper_cp,
            "auroc": self.auroc,
            "auroc_lower": self.auroc_lower,
            "auroc_upper": self.auroc_upper,
            "threshold": self.threshold,
            "control_singleton_rate": self.control_singleton_rate,
            "e_value": self.e_value,
            "log_e_value": self.log_e_value,
            "rejected_h0": self.rejected_h0,
            "verdict": self.verdict,
            "recommendation": self.recommendation,
            "holm_smallest_p": self.holm_smallest_p,
            "fingerprint_hash": self.fingerprint_hash,
        }


@dataclass(frozen=True)
class ConfabulatorReport:
    """Single-bundle export for the coordinator."""
    config: dict[str, Any]
    n_trials: int
    threshold: dict[str, Any] | None
    audit: dict[str, Any] | None
    certificate: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "n_trials": self.n_trials,
            "threshold": dict(self.threshold) if self.threshold else None,
            "audit": dict(self.audit) if self.audit else None,
            "certificate": dict(self.certificate) if self.certificate else None,
        }


# ---------------------------------------------------------------------------
# Equivalence oracles
# ---------------------------------------------------------------------------

EquivalenceOracle = Callable[[str, str], bool]
"""Callable: ``oracle(s1, s2) -> bool``.

Returns ``True`` if ``s1`` and ``s2`` mean the same thing (bidirectional
entailment).  Must be commutative and reflexive; transitivity is *not*
required — the primitive uses Union-Find with the observed relations and
will report the resulting partition even when the oracle is only an
approximation, which is realistic for NLI-based oracles.
"""


def _normalise(text: str) -> str:
    """Lowercase, tokenise, rejoin — used by the default oracles."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def exact_match_oracle(s1: str, s2: str) -> bool:
    """Default oracle when the caller has no NLI model.

    Two completions are equivalent iff they normalise to the same
    string.  Strict; over-fragments; suitable for unit tests and for
    answers that ARE expected to be identical strings (extractive QA,
    numeric answers).
    """
    return _normalise(s1) == _normalise(s2)


def jaccard_oracle(threshold: float = 0.8) -> EquivalenceOracle:
    """Token-Jaccard-over-threshold oracle.

    A pragmatic black-box approximation when only the text is
    available: two completions are equivalent iff their token sets
    overlap above ``threshold``.  ``0.8`` is the default; tune for
    your domain.
    """
    if not (0.0 < threshold <= 1.0):
        raise InvalidConfig(f"jaccard threshold must be in (0, 1], got {threshold}")

    def _oracle(s1: str, s2: str) -> bool:
        a = set(_TOKEN_RE.findall(s1.lower()))
        b = set(_TOKEN_RE.findall(s2.lower()))
        if not a and not b:
            return True
        if not a or not b:
            return False
        inter = len(a & b)
        union = len(a | b)
        return (inter / union) >= threshold

    return _oracle


# ---------------------------------------------------------------------------
# Math helpers — pure stdlib
# ---------------------------------------------------------------------------


def _shannon_entropy(p: Sequence[float]) -> float:
    """Shannon entropy in nats of a non-negative weight vector.

    Renormalises ``p`` to sum to 1 first.  Returns 0.0 on the empty /
    zero vector for safety.
    """
    s = sum(x for x in p if x > 0.0)
    if s <= 0.0:
        return 0.0
    h = 0.0
    for x in p:
        if x > 0.0:
            q = x / s
            h -= q * math.log(q)
    return h


def _miller_madow_correction(n_clusters: int, n_samples: int) -> float:
    """Bias correction for plug-in entropy of categorical data.

    Adds ``(m - 1) / (2 n)`` where ``m`` is the observed support and
    ``n`` the sample size.  Miller (1955).
    """
    if n_samples <= 0:
        return 0.0
    return (max(n_clusters - 1, 0)) / (2.0 * n_samples)


def _union_find(n: int) -> list[int]:
    return list(range(n))


def _uf_find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _uf_union(parent: list[int], a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb


def _cluster(samples: Sequence[Sample],
             oracle: EquivalenceOracle,
             max_clusters: int) -> tuple[list[int], list[list[int]]]:
    """Union-Find clustering of samples under a (possibly non-transitive)
    bidirectional-entailment oracle.

    Returns ``(assignment, groups)`` where ``assignment[i]`` is the
    cluster id of sample ``i`` and ``groups[c]`` is the list of
    sample indices in cluster ``c`` (sorted by cluster id of first
    occurrence, then by sample index).
    """
    n = len(samples)
    parent = _union_find(n)
    # Pairwise calls: O(K^2 / 2).  K is the per-trial sample budget,
    # typically 5–10, so this is at most ~45 calls per trial.
    for i in range(n):
        for j in range(i + 1, n):
            if oracle(samples[i].text, samples[j].text):
                _uf_union(parent, i, j)
    # Build the canonical assignment.  Cluster ids are assigned in
    # order of first occurrence so the result is deterministic and
    # human-readable.
    canonical: dict[int, int] = {}
    assignment = [0] * n
    for i in range(n):
        r = _uf_find(parent, i)
        if r not in canonical:
            canonical[r] = len(canonical)
            if len(canonical) > max_clusters:
                # Defensive: fold the rest into the last cluster.
                # In practice, max_clusters is set well above any K.
                canonical[r] = max_clusters - 1
        assignment[i] = canonical[r]
    groups: list[list[int]] = [[] for _ in range(max(assignment, default=-1) + 1)]
    for i, c in enumerate(assignment):
        groups[c].append(i)
    return assignment, groups


def _lexical_entropy_token(samples: Sequence[Sample]) -> float:
    """Entropy of the empirical token distribution across all samples."""
    counts: dict[str, int] = {}
    for s in samples:
        for tok in _TOKEN_RE.findall(s.text.lower()):
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return 0.0
    return _shannon_entropy(list(counts.values()))


def _predictive_entropy(samples: Sequence[Sample],
                        length_normalise: bool) -> float:
    """Mean negative log-likelihood across the samples that report one.

    Returns ``NaN`` if no sample reports a log-probability.  The
    convention is that ``Sample.mean_logprob`` is non-positive; the
    negation makes it a non-negative entropy in nats.
    """
    vals: list[float] = []
    for s in samples:
        if math.isnan(s.mean_logprob):
            continue
        v = -s.mean_logprob
        if length_normalise and s.n_tokens > 0:
            # If the caller already length-normalised, this leaves it
            # alone modulo a constant factor; we only adjust when the
            # token count is informative.
            v = v  # the mean_logprob is already per-token; keep as-is.
        vals.append(v)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _selfcheck_score(samples: Sequence[Sample],
                     chosen_index: int,
                     oracle: EquivalenceOracle) -> float:
    """SelfCheckGPT-style consistency score for the chosen answer.

    The proxy: fraction of *other* samples that are NOT semantically
    equivalent to the chosen one under the oracle.  In the original
    paper this is a per-sentence pairwise NLI score; the runtime
    expression compresses to one number per trial.  Bounded in
    ``[0, 1]``; higher = more inconsistent (more likely hallucinated).
    """
    if not samples:
        return 0.0
    if chosen_index < 0 or chosen_index >= len(samples):
        return 0.0
    chosen = samples[chosen_index].text
    others = [s for i, s in enumerate(samples) if i != chosen_index]
    if not others:
        return 0.0
    disagree = sum(1 for o in others if not oracle(chosen, o.text))
    return disagree / len(others)


def _combine_score(se: float, le: float, pe: float, sc: float,
                   weights: Sequence[float]) -> float:
    """Weighted sum of detector outputs.

    Weights are renormalised over the modes that actually produced a
    finite number, so a primitive run without log-probabilities still
    produces a well-formed combined score.
    """
    pairs = []
    for value, weight in zip((se, le, pe, sc), weights):
        if math.isnan(value):
            continue
        if weight < 0:
            continue
        pairs.append((value, weight))
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    if total_w <= 0.0:
        return float("nan")
    return sum(v * w for v, w in pairs) / total_w


# ---------- Binomial CIs and tail probabilities ----------


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Lentz continued-fraction evaluation of ``I_x(a, b)``.

    Numerical recipes / Abramowitz-Stegun 26.5.8.  Stable, pure stdlib.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Reflection for faster convergence.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(1.0 - x, b, a)
    # I_x(a, b) = (x^a (1-x)^b / B(a, b)) · CF / a — Numerical
    # Recipes 6.4.  The prefix `x^a (1-x)^b / B(a, b)` is computed
    # in log space for stability; the final `/ a` happens at the
    # return, NOT in the prefix.
    log_pref = (a * math.log(x) + b * math.log(1.0 - x)
                - _log_beta(a, b))
    # Lentz's method for the continued fraction.
    eps = 1e-15
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(log_pref) * h / a


def _beta_quantile(p: float, a: float, b: float,
                   tol: float = 1e-10, max_iters: int = 200) -> float:
    """Quantile of ``Beta(a, b)`` via monotone bisection on
    :func:`_regularised_incomplete_beta`.

    Endpoint-stable; ``p == 0`` returns ``0.0`` and ``p == 1`` returns
    ``1.0`` without iteration.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0 or mid >= 1.0:
            return mid
        cdf = _regularised_incomplete_beta(mid, a, b)
        if abs(cdf - p) < tol:
            return mid
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson confidence interval at level
    ``1 − alpha`` for a Binomial(``n``, ``p``) with observed ``k``
    successes.  Edge cases ``k == 0`` and ``k == n`` are handled in
    closed form.
    """
    if not (0 <= k <= n):
        raise ConfabulatorError(f"k={k}, n={n}: 0 <= k <= n required")
    if n == 0:
        return 0.0, 1.0
    lo_p = alpha / 2.0
    hi_p = 1.0 - alpha / 2.0
    if k == 0:
        lo = 0.0
    else:
        lo = _beta_quantile(lo_p, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = _beta_quantile(hi_p, k + 1, n - k)
    return lo, hi


def _binom_tail_le(k: int, n: int, p: float) -> float:
    """``P(Binom(n, p) <= k)`` via the regularised incomplete beta
    identity ``P(X <= k) = I_{1-p}(n-k, k+1)``.  Numerically stable for
    moderate ``n``; for the small-``n`` audit pool the primitive cares
    about, this is comfortably within float64.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return _regularised_incomplete_beta(1.0 - p, n - k, k + 1)


def _binom_tail_ge(k: int, n: int, p: float) -> float:
    """``P(Binom(n, p) >= k)`` = ``1 - P(X <= k-1)``."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return 1.0 - _binom_tail_le(k - 1, n, p)


# ---------- AUROC ----------


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Wilcoxon-Mann-Whitney AUROC (Hanley-McNeil 1982).

    ``labels[i] == True`` is the *positive* class (a hallucination);
    AUROC is the probability that a random positive scores HIGHER than
    a random negative.  Ties contribute 0.5.
    """
    if len(scores) != len(labels):
        raise ConfabulatorError("scores / labels length mismatch")
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    cnt = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                cnt += 1.0
            elif p == n:
                cnt += 0.5
    return cnt / (len(pos) * len(neg))


def _youden_threshold(scores: Sequence[float],
                      labels: Sequence[bool]) -> tuple[float, float, float, float]:
    """Pick the threshold maximising Youden's J = TPR − FPR.

    Returns ``(threshold, J, TPR_at_threshold, FPR_at_threshold)``.
    The chosen threshold is the score value at which the model
    predicts *positive* (hallucination) for scores ``>= threshold``.
    """
    if len(scores) != len(labels):
        raise ConfabulatorError("scores / labels length mismatch")
    if not scores:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Candidate thresholds: scores plus +inf (predict negative
    # everywhere) and the minimum minus epsilon (predict positive
    # everywhere).
    candidates = sorted(set(scores))
    candidates = [candidates[0] - 1e-12] + candidates + [candidates[-1] + 1e-12]
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    best = (-1.0, candidates[0], 0.0, 0.0)
    for t in candidates:
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= t)
        fp = sum(1 for s, y in zip(scores, labels) if (not y) and s >= t)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        j = tpr - fpr
        if j > best[0]:
            best = (j, t, tpr, fpr)
    return best[1], best[0], best[2], best[3]


def _bootstrap_auroc_ci(scores: Sequence[float],
                        labels: Sequence[bool],
                        *,
                        b: int,
                        confidence: float,
                        seed: int) -> tuple[float, float]:
    """Bootstrap-percentile CI on the AUROC.

    Standard case-resampling: draw ``len(scores)`` indices with
    replacement, recompute AUROC, repeat ``b`` times, return the
    ``(alpha/2, 1 - alpha/2)`` percentiles.  Bootstraps that
    accidentally exclude one of the two classes are discarded and
    redrawn (cap at ``2b`` attempts).
    """
    if b <= 0 or not scores:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(scores)
    vals: list[float] = []
    attempts = 0
    while len(vals) < b and attempts < 2 * b + 50:
        attempts += 1
        idx = [rng.randrange(n) for _ in range(n)]
        ss = [scores[i] for i in idx]
        ll = [labels[i] for i in idx]
        if not any(ll) or all(ll):
            continue
        val = _auroc(ss, ll)
        if math.isnan(val):
            continue
        vals.append(val)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(math.floor((alpha / 2.0) * len(vals))))
    hi_idx = min(len(vals) - 1,
                 int(math.ceil((1.0 - alpha / 2.0) * len(vals))) - 1)
    return vals[lo_idx], vals[hi_idx]


# ---------- Sequential e-process ----------


def _eprocess_one_proportion(observations: Sequence[bool],
                             p0: float,
                             prior_a: float,
                             prior_b: float) -> float:
    """Anytime-valid e-value for ``H_0: p <= p0`` under a one-proportion
    sequential test.

    Implementation: the prior-mean martingale of Howard et al. (2021)
    in its Beta-Binomial conjugate form.  After ``n`` observations
    with ``k`` successes, the e-value is the ratio of the posterior-
    predictive probability under the Beta prior to the maximum-
    likelihood probability under ``H_0``.  Concretely::

        e_n = [ B(prior_a + k, prior_b + n - k) / B(prior_a, prior_b) ]
              / [ p0^k * (1 - p0)^(n - k) ]

    This is anytime-valid: at any stopping time τ, ``P_{H_0}(e_τ >=
    1/α) <= α`` (Ville 1939).  ``rejected_h0 = e_value >= 1 / alpha``.
    """
    n = len(observations)
    if n == 0:
        return 1.0
    k = sum(1 for o in observations if o)
    log_e = (_log_beta(prior_a + k, prior_b + n - k)
             - _log_beta(prior_a, prior_b))
    if k > 0:
        log_e -= k * math.log(p0)
    if (n - k) > 0:
        log_e -= (n - k) * math.log1p(-p0)
    return math.exp(log_e)


def _holm_smallest(p_values: Sequence[float], alpha: float) -> float:
    """Holm step-down adjusted minimum p-value.

    For an FWER-controlling primitive: if the smallest adjusted
    p-value is ≤ alpha, the overall multi-test is significant.
    Returns 1.0 on the empty input.
    """
    if not p_values:
        return 1.0
    sorted_p = sorted(p_values)
    m = len(sorted_p)
    adjusted: list[float] = []
    for i, p in enumerate(sorted_p):
        adjusted.append(min(1.0, (m - i) * p))
    # Step-down: enforce monotonicity.
    for i in range(1, len(adjusted)):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    return adjusted[0]


def _stable(value: Any) -> Any:
    """Make a payload deterministic for the fingerprint chain."""
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value.keys()) if k != "ts"}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.12g}")
    return value


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


class Confabulator:
    """Semantic-entropy hallucination certification primitive.

    Lifecycle:

      1. ``submit(prompt_id, samples, ...)`` ingests one trial's worth
         of completions plus optional ground-truth.
      2. (optional) ``fit_threshold()`` calibrates the combined-score
         threshold against the labelled pool — required before
         ``gate()`` produces per-prompt verdicts.
      3. ``audit()`` returns the current anytime-valid e-process state.
      4. ``certify(delta)`` bundles the audit + threshold + control
         floor into a :class:`ConfabulatorCertificate`.
      5. ``report()`` returns a single :class:`ConfabulatorReport`.

    Stateless across instances; thread-safe within one.  Deterministic
    given seed.
    """

    def __init__(self,
                 config: ConfabulatorConfig | None = None,
                 *,
                 default_oracle: EquivalenceOracle | None = None,
                 bus: EventBus | None = None) -> None:
        self.config = config or ConfabulatorConfig()
        self.bus = bus
        self.default_oracle: EquivalenceOracle = (
            default_oracle if default_oracle is not None else exact_match_oracle
        )
        self._lock = threading.RLock()
        self._trials: list[Trial] = []
        self._reports: list[TrialReport] = []
        self._threshold: ThresholdReport | None = None
        self._fingerprint = hashlib.sha256()
        self._fingerprint.update(json.dumps(
            {"version": 1,
             "modes": list(self.config.modes),
             "seed": self.config.seed,
             "p0": self.config.budget_p0,
             "alpha": self.config.alpha},
            sort_keys=True,
        ).encode())
        self._publish(CONFAB_STARTED, {
            "modes": list(self.config.modes),
            "seed": self.config.seed,
            "p0": self.config.budget_p0,
            "alpha": self.config.alpha,
        })

    # ----- event helpers -----
    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        payload = {**data, "ts": time.time()}
        self._fingerprint.update(json.dumps(
            {"kind": kind, "data": _stable(payload)}, sort_keys=True
        ).encode())
        if self.bus is not None:
            self.bus.publish(Event(kind=kind, data=payload))

    @property
    def fingerprint_hash(self) -> str:
        return self._fingerprint.hexdigest()

    @property
    def n_trials(self) -> int:
        with self._lock:
            return len(self._trials)

    @property
    def reports(self) -> tuple[TrialReport, ...]:
        with self._lock:
            return tuple(self._reports)

    @property
    def threshold(self) -> ThresholdReport | None:
        with self._lock:
            return self._threshold

    def reset(self) -> None:
        with self._lock:
            self._trials.clear()
            self._reports.clear()
            self._threshold = None
            self._publish(CONFAB_RESET, {})

    # ----- ingestion -----
    def submit(self,
               prompt_id: str,
               samples: Sequence[Sample],
               *,
               equivalence: EquivalenceOracle | None = None,
               truth: bool | None = None,
               control: bool = False,
               reference_chosen_index: int = 0,
               metadata: Mapping[str, Any] | None = None) -> TrialReport:
        """Submit one trial.  Returns the per-trial detector outputs.

        Args:
            prompt_id: caller-side identifier.
            samples: a sequence of :class:`Sample`; must be ≥ 2.
            equivalence: per-trial oracle override; falls back to the
                Confabulator's ``default_oracle``.
            truth: optional ground-truth label.  ``True`` = correct
                answer; ``False`` = hallucination; ``None`` =
                unlabelled.
            control: ``True`` marks a singleton-control trial.
            reference_chosen_index: which sample to treat as the
                served answer (used by SelfCheck and reporting).
            metadata: round-tripped, not interpreted.
        """
        if isinstance(samples, Sample):
            raise InvalidTrial("samples must be a sequence of Sample, not a single Sample")
        sample_tuple = tuple(samples)
        trial = Trial(
            prompt_id=prompt_id,
            samples=sample_tuple,
            truth=truth,
            control=control,
            reference_chosen_index=reference_chosen_index,
            metadata=dict(metadata) if metadata else {},
        )
        oracle = equivalence if equivalence is not None else self.default_oracle
        assignment, _groups = _cluster(
            trial.samples, oracle, self.config.max_clusters
        )
        n_clusters = len(set(assignment))
        # Plug-in cluster distribution from the cluster counts.
        counts = [0] * n_clusters
        for c in assignment:
            counts[c] += 1
        n = len(trial.samples)
        dist = tuple(c / n for c in counts)
        se = _shannon_entropy(counts)
        if self.config.miller_madow:
            se += _miller_madow_correction(n_clusters, n)
        le = _lexical_entropy_token(trial.samples)
        pe = _predictive_entropy(trial.samples, self.config.length_normalise_pe)
        sc = _selfcheck_score(trial.samples,
                              trial.reference_chosen_index,
                              oracle)
        combined = _combine_score(se, le, pe, sc, self.config.weights)

        verdict = VERDICT_INCONCLUSIVE
        if self._threshold is not None and not math.isnan(combined):
            verdict = (VERDICT_FAIL if combined >= self._threshold.threshold
                       else VERDICT_PASS)

        rep = TrialReport(
            prompt_id=prompt_id,
            n_samples=n,
            n_clusters=n_clusters,
            semantic_entropy=se,
            lexical_entropy=le,
            predictive_entropy=pe,
            selfcheck_score=sc,
            combined_score=combined,
            chosen_text=trial.samples[trial.reference_chosen_index].text,
            cluster_distribution=dist,
            verdict=verdict,
            has_truth=trial.truth is not None,
            truth_value=trial.truth,
            is_control=trial.control,
            metadata=dict(trial.metadata),
        )

        with self._lock:
            self._trials.append(trial)
            self._reports.append(rep)
            # Discard a stale threshold-fit's per-trial verdicts: we
            # keep the threshold itself (it's still calibrated on the
            # prior labelled pool), but the new trial's verdict is
            # already computed above.

        self._publish(CONFAB_SUBMITTED, {
            "prompt_id": prompt_id,
            "n_samples": n,
            "n_clusters": n_clusters,
            "has_truth": rep.has_truth,
            "is_control": rep.is_control,
        })
        self._publish(CONFAB_SCORED, {
            "prompt_id": prompt_id,
            "semantic_entropy": se,
            "lexical_entropy": le,
            "predictive_entropy": pe,
            "selfcheck_score": sc,
            "combined_score": combined,
            "verdict": verdict,
        })
        return rep

    # ----- calibration -----
    def fit_threshold(self,
                      *,
                      target_fpr: float | None = None,
                      target_tpr: float | None = None,
                      ) -> ThresholdReport:
        """Calibrate the combined-score threshold on labelled trials.

        Default rule: pick the threshold maximising Youden's J.  When
        ``target_fpr`` is supplied, pick instead the *smallest*
        threshold whose FPR ≤ target_fpr (most-sensitive at the FPR
        ceiling).  When ``target_tpr`` is supplied (and ``target_fpr``
        is not), pick the *largest* threshold whose TPR ≥ target_tpr.
        Both are operating-point overrides used by the coordinator's
        regulator profile.

        Raises :class:`NotEnoughTrials` if fewer than 2 labelled
        trials are available, or if either class is absent.
        """
        with self._lock:
            reps = [r for r in self._reports if r.has_truth and not r.is_control]
            if len(reps) < 2:
                raise NotEnoughTrials(
                    f"need >= 2 labelled trials for threshold fit, "
                    f"have {len(reps)}"
                )
            scores = [r.combined_score for r in reps]
            # POSITIVE class = hallucination (truth_value is False).
            # AUROC = P(score_hallucinated > score_truthful); the
            # combined score is constructed to rise with hallucination
            # uncertainty, so a well-calibrated detector lands at
            # AUROC > 0.5 with this polarity.
            labels = [(r.truth_value is False) for r in reps]
            if not any(labels) or all(labels):
                raise NotEnoughTrials(
                    "labelled pool has only one class — cannot fit threshold"
                )
            # Drop NaN combined scores (unlikely, but possible if all
            # detectors degenerate).
            keep = [(s, y) for s, y in zip(scores, labels) if not math.isnan(s)]
            if len(keep) < 2:
                raise NotEnoughTrials(
                    "all labelled trials have NaN combined scores"
                )
            scores = [s for s, _ in keep]
            labels = [y for _, y in keep]
            n_pos = sum(1 for y in labels if y)
            n_neg = len(labels) - n_pos

            auroc = _auroc(scores, labels)
            lo, hi = _bootstrap_auroc_ci(
                scores, labels,
                b=self.config.bootstrap_b,
                confidence=self.config.confidence,
                seed=self.config.seed + 7919,
            )

            if target_fpr is not None and target_tpr is not None:
                raise InvalidConfig(
                    "supply at most one of target_fpr / target_tpr"
                )

            if target_fpr is not None:
                threshold, j, tpr, fpr = _threshold_at_fpr(
                    scores, labels, target_fpr
                )
            elif target_tpr is not None:
                threshold, j, tpr, fpr = _threshold_at_tpr(
                    scores, labels, target_tpr
                )
            else:
                threshold, j, tpr, fpr = _youden_threshold(scores, labels)

            rep = ThresholdReport(
                threshold=threshold,
                auroc=auroc,
                auroc_lower=lo,
                auroc_upper=hi,
                youden_j=j,
                tpr_at_threshold=tpr,
                fpr_at_threshold=fpr,
                n_labelled=len(labels),
                n_positive=n_pos,
                n_negative=n_neg,
                confidence=self.config.confidence,
                bootstrap_b=self.config.bootstrap_b,
            )
            self._threshold = rep
            # Refresh stored per-trial verdicts using the new threshold.
            new_reports: list[TrialReport] = []
            for r in self._reports:
                if math.isnan(r.combined_score):
                    new_v = VERDICT_INCONCLUSIVE
                else:
                    new_v = (VERDICT_FAIL if r.combined_score >= threshold
                             else VERDICT_PASS)
                if r.verdict == new_v:
                    new_reports.append(r)
                    continue
                new_reports.append(TrialReport(
                    prompt_id=r.prompt_id,
                    n_samples=r.n_samples,
                    n_clusters=r.n_clusters,
                    semantic_entropy=r.semantic_entropy,
                    lexical_entropy=r.lexical_entropy,
                    predictive_entropy=r.predictive_entropy,
                    selfcheck_score=r.selfcheck_score,
                    combined_score=r.combined_score,
                    chosen_text=r.chosen_text,
                    cluster_distribution=r.cluster_distribution,
                    verdict=new_v,
                    has_truth=r.has_truth,
                    truth_value=r.truth_value,
                    is_control=r.is_control,
                    metadata=r.metadata,
                ))
            self._reports = new_reports

        self._publish(CONFAB_CALIBRATED, {
            "threshold": rep.threshold,
            "auroc": rep.auroc,
            "auroc_lower": rep.auroc_lower,
            "auroc_upper": rep.auroc_upper,
            "youden_j": rep.youden_j,
            "n_labelled": rep.n_labelled,
            "n_positive": rep.n_positive,
            "n_negative": rep.n_negative,
        })
        return rep

    # ----- audit -----
    def audit(self) -> AuditReport:
        """Compute the anytime-valid e-process state.

        Uses the labelled trials only — control trials never count
        toward the operating rate.  The e-value is the prior-mean
        martingale of Howard et al. (2021); H_0 is rejected at level
        ``alpha`` iff ``e_value >= 1 / alpha``.

        Returns the rate, the Clopper-Pearson CI, the e-value, the
        log-e-value, and the rejection decision.  Safe to call at any
        time — there is no stopping rule the caller must respect.
        """
        with self._lock:
            reps = [r for r in self._reports if r.has_truth and not r.is_control]
            n = len(reps)
            obs = [bool(r.truth_value is False) for r in reps]
            # Convention: H_0 says hallucination rate ≤ p0; a True
            # outcome (truth_value == False) is a hallucination event.
            k = sum(1 for o in obs if o)
            rate = (k / n) if n > 0 else 0.0
            lo, hi = _clopper_pearson(k, n, self.config.alpha)
            e = _eprocess_one_proportion(
                obs,
                self.config.budget_p0,
                self.config.prior_a,
                self.config.prior_b,
            )
            log_e = math.log(e) if e > 0.0 else float("-inf")
            rejected = (e >= 1.0 / self.config.alpha)
            rep = AuditReport(
                n_trials_labelled=n,
                n_hallucinations=k,
                running_rate=rate,
                rate_lower_clopper_pearson=lo,
                rate_upper_clopper_pearson=hi,
                e_value=e,
                log_e_value=log_e,
                rejected_h0=rejected,
                p0=self.config.budget_p0,
                alpha=self.config.alpha,
            )
        self._publish(CONFAB_AUDITED, {
            "n_trials_labelled": rep.n_trials_labelled,
            "n_hallucinations": rep.n_hallucinations,
            "running_rate": rep.running_rate,
            "e_value": rep.e_value,
            "rejected_h0": rep.rejected_h0,
        })
        return rep

    # ----- control floor -----
    def control_singleton_rate(self) -> float | None:
        """Fraction of control trials with a *single* semantic cluster.

        On well-formed control trials (paraphrases of a uniquely-
        answered question) the clustering should collapse to one
        bucket.  The complement is the irreducible semantic-clustering
        noise floor — what the audit's rate CI has to clear before any
        claim about the operating model is interpretable.
        Returns ``None`` if no control trials have been submitted.
        """
        with self._lock:
            ctrls = [r for r in self._reports if r.is_control]
            if not ctrls:
                return None
            n_collapsed = sum(1 for r in ctrls if r.n_clusters == 1)
            return n_collapsed / len(ctrls)

    # ----- certification -----
    def certify(self, *, delta: float | None = None) -> ConfabulatorCertificate:
        """Bundle the audit / threshold / control floor into a
        single replay-verifiable certificate.

        ``delta`` overrides ``alpha`` for this call only — useful when
        a coordinator wants a stricter / looser one-shot certificate
        than the configured budget.  Does not mutate ``self.config``.
        """
        with self._lock:
            alpha = self.config.alpha if delta is None else float(delta)
            if not (0.0 < alpha < 1.0):
                raise InvalidConfig("delta must be in (0, 1)")
            reps = [r for r in self._reports if r.has_truth and not r.is_control]
            n = len(self._trials)
            n_lab = len(reps)
            n_ctrl = sum(1 for r in self._reports if r.is_control)
            if n_lab > 0:
                k = sum(1 for r in reps if r.truth_value is False)
                rate = k / n_lab
                lo, hi = _clopper_pearson(k, n_lab, alpha)
                obs = [bool(r.truth_value is False) for r in reps]
                e = _eprocess_one_proportion(
                    obs,
                    self.config.budget_p0,
                    self.config.prior_a,
                    self.config.prior_b,
                )
                log_e = math.log(e) if e > 0.0 else float("-inf")
                rejected = (e >= 1.0 / alpha)
            else:
                rate = None
                lo, hi = None, None
                e, log_e = 1.0, 0.0
                rejected = False

            thr = self._threshold
            auroc = thr.auroc if thr else None
            auroc_lo = thr.auroc_lower if thr else None
            auroc_hi = thr.auroc_upper if thr else None
            threshold_val = thr.threshold if thr else None

            ctrl_rate = self.control_singleton_rate()

            # Per-detector mini p-values for Holm combining.  Each
            # detector ships a one-sided p-value for "AUROC > 0.5"
            # (the model's score is informative), tested via the
            # Hanley-McNeil z under H_0.  In the absence of a labelled
            # pool, defer to 1.0.
            p_vals: list[float] = []
            holm_p: float | None = None
            if n_lab >= 2 and thr is not None and thr.n_positive > 0 and thr.n_negative > 0:
                scores_all = [(r.semantic_entropy, r.lexical_entropy,
                               r.predictive_entropy, r.selfcheck_score,
                               r.combined_score) for r in reps]
                # Positive = hallucination — matches fit_threshold.
                labels = [(r.truth_value is False) for r in reps]
                for i in range(5):
                    s = [t[i] for t in scores_all]
                    if all(math.isnan(x) for x in s):
                        continue
                    s_clean = [x if not math.isnan(x) else 0.0 for x in s]
                    a = _auroc(s_clean, labels)
                    if math.isnan(a):
                        continue
                    p = _auroc_one_sided_p(a, thr.n_positive, thr.n_negative)
                    p_vals.append(p)
                holm_p = _holm_smallest(p_vals, alpha) if p_vals else None

            verdict, recommendation = _decide(
                rate=rate, hi=hi, p0=self.config.budget_p0,
                warn_factor=self.config.warn_factor,
                e_value=e, alpha=alpha, n_lab=n_lab,
                auroc=auroc, auroc_lo=auroc_lo,
            )

            cert = ConfabulatorCertificate(
                n_trials=n,
                n_trials_labelled=n_lab,
                n_control=n_ctrl,
                hallucination_rate=rate,
                rate_lower_cp=lo,
                rate_upper_cp=hi,
                auroc=auroc,
                auroc_lower=auroc_lo,
                auroc_upper=auroc_hi,
                threshold=threshold_val,
                control_singleton_rate=ctrl_rate,
                e_value=e,
                log_e_value=log_e,
                rejected_h0=rejected,
                verdict=verdict,
                recommendation=recommendation,
                holm_smallest_p=holm_p,
                fingerprint_hash=self.fingerprint_hash,
            )
        self._publish(CONFAB_CERTIFIED, {
            "n_trials": cert.n_trials,
            "n_trials_labelled": cert.n_trials_labelled,
            "hallucination_rate": cert.hallucination_rate,
            "rate_upper_cp": cert.rate_upper_cp,
            "auroc": cert.auroc,
            "e_value": cert.e_value,
            "verdict": cert.verdict,
            "recommendation": cert.recommendation,
        })
        return cert

    # ----- runtime gate -----
    def gate(self,
             prompt_id: str,
             samples: Sequence[Sample],
             *,
             equivalence: EquivalenceOracle | None = None,
             reference_chosen_index: int = 0,
             metadata: Mapping[str, Any] | None = None,
             ) -> tuple[TrialReport, str]:
        """Production-time use: score one trial against the fitted
        threshold and return ``(report, recommendation)``.

        The recommendation routes the coordinator's response:

        * ``trust`` — combined score well below threshold.
        * ``regenerate`` — score crosses threshold; re-sample at
          higher diversity or a stronger model.
        * ``restrict`` — score deep in the hallucination region;
          present the answer behind a confidence-shield.
        * ``escalate_human`` — extreme score; ask a human.
        * ``quarantine`` — score is NaN (no detector fired); pull the
          worker out of rotation pending diagnosis.

        Raises :class:`NotCalibrated` if :meth:`fit_threshold` has not
        run.
        """
        with self._lock:
            if self._threshold is None:
                raise NotCalibrated(
                    "fit_threshold() before gate()"
                )
            threshold = self._threshold.threshold
        rep = self.submit(
            prompt_id, samples,
            equivalence=equivalence,
            truth=None,
            control=False,
            reference_chosen_index=reference_chosen_index,
            metadata=metadata,
        )
        score = rep.combined_score
        if math.isnan(score):
            recommendation = REC_QUARANTINE
        else:
            # Bucket relative to threshold.  Three multiplicative
            # bands above threshold: trust below; regenerate just
            # above; restrict 1.5×–3×; escalate beyond.
            t = threshold
            if score < t:
                recommendation = REC_TRUST
            elif score < 1.5 * t:
                recommendation = REC_REGENERATE
            elif score < 3.0 * t:
                recommendation = REC_RESTRICT
            else:
                recommendation = REC_ESCALATE
        self._publish(CONFAB_GATED, {
            "prompt_id": prompt_id,
            "combined_score": score,
            "threshold": threshold,
            "recommendation": recommendation,
        })
        return rep, recommendation

    # ----- bundled report -----
    def report(self) -> ConfabulatorReport:
        with self._lock:
            n = len(self._trials)
            thr = self._threshold.to_dict() if self._threshold else None
            try:
                audit = self.audit().to_dict()
            except ConfabulatorError:
                audit = None
            try:
                cert = self.certify().to_dict()
            except ConfabulatorError:
                cert = None
            cfg = {
                "modes": list(self.config.modes),
                "sample_budget_k": self.config.sample_budget_k,
                "weights": list(self.config.weights),
                "budget_p0": self.config.budget_p0,
                "alpha": self.config.alpha,
                "bootstrap_b": self.config.bootstrap_b,
                "confidence": self.config.confidence,
                "seed": self.config.seed,
                "miller_madow": self.config.miller_madow,
                "length_normalise_pe": self.config.length_normalise_pe,
                "warn_factor": self.config.warn_factor,
                "prior_a": self.config.prior_a,
                "prior_b": self.config.prior_b,
                "max_clusters": self.config.max_clusters,
            }
            out = ConfabulatorReport(
                config=cfg, n_trials=n,
                threshold=thr, audit=audit, certificate=cert,
            )
        self._publish(CONFAB_REPORTED, {"n_trials": n})
        return out


# ---------------------------------------------------------------------------
# Threshold-at-operating-point helpers
# ---------------------------------------------------------------------------


def _threshold_at_fpr(scores: Sequence[float], labels: Sequence[bool],
                      target_fpr: float) -> tuple[float, float, float, float]:
    """Smallest threshold (most sensitive) whose FPR ≤ target_fpr."""
    if not (0.0 <= target_fpr <= 1.0):
        raise InvalidConfig("target_fpr must be in [0, 1]")
    candidates = sorted(set(scores))
    candidates = [candidates[0] - 1e-12] + candidates + [candidates[-1] + 1e-12]
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    best: tuple[float, float, float, float] | None = None
    for t in sorted(candidates):
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= t)
        fp = sum(1 for s, y in zip(scores, labels) if (not y) and s >= t)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        if fpr <= target_fpr:
            if best is None or tpr > best[2]:
                best = (t, tpr - fpr, tpr, fpr)
    if best is None:
        # Fall back to most-conservative threshold (predict negative
        # always) when the FPR ceiling cannot be met.
        t = candidates[-1]
        return t, 0.0, 0.0, 0.0
    return best


def _threshold_at_tpr(scores: Sequence[float], labels: Sequence[bool],
                      target_tpr: float) -> tuple[float, float, float, float]:
    """Largest threshold (most specific) whose TPR ≥ target_tpr."""
    if not (0.0 <= target_tpr <= 1.0):
        raise InvalidConfig("target_tpr must be in [0, 1]")
    candidates = sorted(set(scores))
    candidates = [candidates[0] - 1e-12] + candidates + [candidates[-1] + 1e-12]
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    best: tuple[float, float, float, float] | None = None
    for t in sorted(candidates, reverse=True):
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= t)
        fp = sum(1 for s, y in zip(scores, labels) if (not y) and s >= t)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        if tpr >= target_tpr:
            if best is None or fpr < best[3]:
                best = (t, tpr - fpr, tpr, fpr)
    if best is None:
        t = candidates[0]
        return t, 0.0, 1.0, 1.0
    return best


def _auroc_one_sided_p(auroc: float, n_pos: int, n_neg: int) -> float:
    """One-sided p-value for ``H_0: AUROC == 0.5`` under Hanley-McNeil.

    Uses the normal approximation with the Hanley-McNeil 1982
    standard-error formula.  ``H_0`` is two-sided 0.5; we report the
    upper-tail (informative score) p-value.
    """
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    a = auroc
    q1 = a / (2.0 - a)
    q2 = (2.0 * a * a) / (1.0 + a)
    var = (a * (1.0 - a)
           + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    if var <= 0.0:
        return 0.0 if a > 0.5 else 1.0
    z = (a - 0.5) / math.sqrt(var)
    # Upper-tail of standard normal.
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _decide(*,
            rate: float | None,
            hi: float | None,
            p0: float,
            warn_factor: float,
            e_value: float,
            alpha: float,
            n_lab: int,
            auroc: float | None,
            auroc_lo: float | None,
            ) -> tuple[str, str]:
    """Distill the audit + threshold state into a verdict + recommendation."""
    if n_lab == 0:
        return VERDICT_INCONCLUSIVE, REC_RESTRICT
    rejected = (e_value >= 1.0 / alpha)
    if rejected:
        return VERDICT_FAIL, REC_ESCALATE
    # Without a rejection, look at where the rate CI sits.
    upper = hi if hi is not None else 1.0
    rate_v = rate if rate is not None else 0.0
    if upper <= p0:
        # Whole CI under the budget.  Pass — but quality of the AUROC
        # determines whether we "trust" or merely "restrict".
        if auroc is not None and auroc_lo is not None and auroc_lo >= 0.65:
            return VERDICT_PASS, REC_TRUST
        return VERDICT_PASS, REC_RESTRICT
    if rate_v <= warn_factor * p0:
        return VERDICT_WARN, REC_REGENERATE
    if rate_v >= p0:
        return VERDICT_WARN, REC_RESTRICT
    return VERDICT_WARN, REC_RESTRICT


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def confabulator(*, seed: int = 0, **kw: Any) -> Confabulator:
    """Sugar: ``Confabulator(ConfabulatorConfig(seed=seed, **kw))``."""
    return Confabulator(ConfabulatorConfig(seed=seed, **kw))


def confabulator_with_oracle(oracle: EquivalenceOracle,
                             *,
                             seed: int = 0, **kw: Any) -> Confabulator:
    """Sugar: a Confabulator with the supplied equivalence oracle wired in."""
    return Confabulator(ConfabulatorConfig(seed=seed, **kw),
                        default_oracle=oracle)


def synthetic_trials(*,
                     n_truthful: int,
                     n_hallucinated: int,
                     k: int = 5,
                     truthful_cluster_skew: float = 5.0,
                     hallucinated_cluster_skew: float = 0.7,
                     seed: int = 0) -> list[Trial]:
    """Generate synthetic trials for tests and demos.

    Truthful trials concentrate samples on a small number of clusters;
    hallucinated trials disperse them.  No NLI model required —
    completions are stylised strings whose token-Jaccard signature
    drives clustering, and the predictive-entropy field is the
    negative log of the cluster-conditional probability.
    """
    rng = random.Random(seed)
    trials: list[Trial] = []
    answer_words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                    "eta", "theta", "iota", "kappa", "lambda", "mu",
                    "nu", "xi", "omicron", "pi"]
    for t_idx in range(n_truthful):
        # Dirichlet-ish concentration on the first 1–2 clusters.
        weights = [truthful_cluster_skew] + [0.3] * (k - 1)
        probs = _dirichlet_sample(weights, rng)
        samples = _emit_samples(probs, answer_words, k, rng)
        trials.append(Trial(
            prompt_id=f"truthful_{t_idx}",
            samples=tuple(samples),
            truth=True,
        ))
    for h_idx in range(n_hallucinated):
        weights = [hallucinated_cluster_skew] * k
        probs = _dirichlet_sample(weights, rng)
        samples = _emit_samples(probs, answer_words, k, rng)
        trials.append(Trial(
            prompt_id=f"halluc_{h_idx}",
            samples=tuple(samples),
            truth=False,
        ))
    return trials


def synthetic_control_trials(*, n: int, k: int = 5,
                             seed: int = 0) -> list[Trial]:
    """``n`` control trials whose samples are paraphrases of one answer.

    The default :func:`exact_match_oracle` collapses them to one
    cluster under normalisation, exposing the noise floor that any
    NLI oracle has to clear.
    """
    rng = random.Random(seed + 9001)
    trials: list[Trial] = []
    for i in range(n):
        base = f"answer_{i}_kappa"
        samples = []
        for _ in range(k):
            # Paraphrase by adding stop words, punctuation, casing.
            jitter = rng.choice(
                ["", " indeed", " of course",
                 ", which is correct.", " — yes."])
            txt = (base + jitter).capitalize() if rng.random() < 0.5 else base
            samples.append(Sample(text=txt))
        trials.append(Trial(
            prompt_id=f"control_{i}",
            samples=tuple(samples),
            truth=None,
            control=True,
        ))
    return trials


def _dirichlet_sample(weights: Sequence[float], rng: random.Random) -> list[float]:
    """Dirichlet sample via gamma ratios — stdlib only."""
    gammas = [rng.gammavariate(max(w, 1e-6), 1.0) for w in weights]
    s = sum(gammas) or 1.0
    return [g / s for g in gammas]


def _emit_samples(probs: Sequence[float], words: Sequence[str], k: int,
                  rng: random.Random) -> list[Sample]:
    """Draw ``k`` samples whose text is the cluster label.

    Surface form is kept simple so that the default
    :func:`exact_match_oracle` clusters them exactly.  Real-world
    callers will plug in an entailment-based oracle and need no help
    from this generator.
    """
    samples: list[Sample] = []
    n_clusters = len(probs)
    cluster_labels = [words[(i * 3 + 1) % len(words)] for i in range(n_clusters)]
    for _ in range(k):
        # Pick a cluster ~ probs.
        r = rng.random()
        cum = 0.0
        cluster = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                cluster = i
                break
        label = cluster_labels[cluster]
        text = label
        lp = math.log(probs[cluster] + 1e-6)
        samples.append(Sample(
            text=text,
            mean_logprob=lp,
            n_tokens=max(1, len(text.split())),
        ))
    return samples


__all__ = [
    # Modes / verdicts / recommendations / event names
    "MODE_SEMANTIC_ENTROPY",
    "MODE_LEXICAL_ENTROPY",
    "MODE_PREDICTIVE_ENTROPY",
    "MODE_SELFCHECK",
    "MODE_COMBINED",
    "KNOWN_MODES",
    "VERDICT_PASS",
    "VERDICT_WARN",
    "VERDICT_FAIL",
    "VERDICT_INCONCLUSIVE",
    "KNOWN_VERDICTS",
    "REC_TRUST",
    "REC_RESTRICT",
    "REC_QUARANTINE",
    "REC_REGENERATE",
    "REC_ESCALATE",
    "KNOWN_RECOMMENDATIONS",
    "CONFAB_STARTED",
    "CONFAB_SUBMITTED",
    "CONFAB_SCORED",
    "CONFAB_CALIBRATED",
    "CONFAB_AUDITED",
    "CONFAB_CERTIFIED",
    "CONFAB_REPORTED",
    "CONFAB_GATED",
    "CONFAB_RESET",
    "NO_LOGPROB",
    # Exceptions
    "ConfabulatorError",
    "InvalidConfig",
    "InvalidSample",
    "InvalidTrial",
    "UnknownMode",
    "NotEnoughTrials",
    "NotCalibrated",
    # Records
    "Sample",
    "Trial",
    "TrialReport",
    "ThresholdReport",
    "AuditReport",
    "ConfabulatorCertificate",
    "ConfabulatorReport",
    "ConfabulatorConfig",
    # Primitive + helpers
    "Confabulator",
    "EquivalenceOracle",
    "exact_match_oracle",
    "jaccard_oracle",
    "confabulator",
    "confabulator_with_oracle",
    "synthetic_trials",
    "synthetic_control_trials",
]
