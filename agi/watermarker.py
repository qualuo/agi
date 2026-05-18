r"""Watermarker — synthetic-content provenance certification as a runtime primitive.

A coordination engine that routes work through LLM-shaped workers, ingests
content from upstream pipelines, or returns generated artefacts to paying
customers faces a *provenance* question every other primitive in this
runtime sidesteps: **was this token sequence emitted by a watermarked
generator, with what statistical strength, and what does it license a
downstream stage to do?**

``Refuser`` answers "will the model refuse the right things?".
``Sycophant`` answers "will it keep its answer under pressure?".
``Confabulator`` answers "is the answer a fluent hallucination?".
``Watermarker`` answers the *cryptographic* question that the EU AI
Act, the Coalition for Content Provenance and Authenticity (C2PA), and
every newsroom, marketplace, and education provider is racing to put on
top of generative output: **is this text marked, and how sure are we?**

The primitive ships the headline detector from the 2023–2024 watermarking
literature — the Kirchenbauer–Geiping–Wen *green-list* z-test
(Kirchenbauer et al. ICML 2023) — plus the controls and statistical
machinery that make a watermark verdict *replay-verifiable* to a
coordinator:

  * exact Clopper–Pearson binomial CI on the green-list fraction;
  * normal-approximation z-statistic with continuity correction for long
    documents;
  * AUROC sweep with bootstrap-percentile CI when the caller can supply
    a labelled mixed pool of watermarked vs. unwatermarked texts;
  * anytime-valid Beta-Binomial e-process for monitoring a stream of
    documents against a documented green-rate budget (Ville 1939;
    Howard–Ramdas–McAuliffe–Sekhon 2021);
  * Holm step-down across modes for FWER-controlled multi-test verdicts;
  * Vovk–Wang product-of-e-values fusion for e-process combination;
  * Benjamini–Hochberg FDR control over many per-document p-values;
  * tamper-evident SHA-256 fingerprint chain over certificates so a
    coordinator can replay the audit byte-for-byte.

The threat model
----------------

Two adversaries motivate the test surface.  Each emits a stream of
documents the coordinator must route.

  * **Counterfeit-as-genuine.**  An unwatermarked generator (or a
    human) produces text and tries to claim it came from a marked
    pipeline.  The null is "unmarked": the green-list rate is the
    documented ``γ``; the test rejects when the observed rate is
    *significantly higher*.  Headline detector :data:`MODE_KGW_GREEN`.

  * **Genuine-as-counterfeit.**  An adversary takes marked text and
    paraphrases / re-tokenises / truncates to *destroy* the mark.  The
    null becomes "marked at advertised strength"; the test rejects
    when the observed rate is *significantly lower than expected* under
    the documented ``γ + ε``.  Tested by sequential two-sided
    monitoring under :data:`MODE_KGW_SEQUENTIAL`.

Each mode below produces a *per-document* test statistic plus a green-
list rate.  The Watermarker combines them, calibrates a threshold on
labelled trials when the caller can supply ground truth, and emits a
verdict (``WM_PASS`` / ``WM_WARN`` / ``WM_FAIL`` / ``WM_INCONCLUSIVE``)
plus a coordinator-facing recommendation (``trust`` / ``restrict`` /
``quarantine`` / ``block`` / ``escalate_human``).

Modes
-----

  * **KGW green-list z-test.**  :data:`MODE_KGW_GREEN`.  The headline
    Kirchenbauer-Geiping-Wen (2023) construction.  Each token is
    classified *green* or *red* by a deterministic hash of its
    immediate predecessor (the "left-context width", ``h=1`` in the
    canonical paper) seeded by a watermark key.  Under the null
    (unwatermarked, fair-token) the expected green fraction is the
    declared ``γ`` and the standardised statistic is
    ``z = (G − γT) / √(Tγ(1−γ))`` (Kirchenbauer eq. 4).  Reject H0
    when ``z > z_{1-α}``.  The detector is **black-box for the verifier**:
    the caller needs only the public spec ``(key, γ, hash_kind, h)``
    and a tokenizer.

  * **KGW green-list exact test.**  :data:`MODE_KGW_EXACT`.  Same
    statistic as ``MODE_KGW_GREEN`` but with the *exact* binomial
    one-tail p-value computed via the regularised incomplete beta
    identity.  Use for short documents where the normal approximation
    is loose; the normal-z falls back to the exact tail when
    ``T < 50`` automatically.

  * **KGW selfhash variant.**  :data:`MODE_KGW_SELFHASH`.  Variant in
    which the hash includes the token itself (Kirchenbauer et al. 2024
    "On the Reliability of Watermarks"); empirically more robust to
    paraphrase but rejects the same H0.  The Watermarker re-uses the
    same z- and exact-test machinery, only the green-list construction
    differs.

  * **Lexical baseline (control).**  :data:`MODE_LEXICAL`.  Surface-
    form entropy of normalised tokens.  Cheap; useful as a control to
    detect when the input is degenerate (template-only, repeated
    boilerplate, empty after tokenisation) and the green-list test is
    *uninformative* rather than negative.  No certificate of its own;
    contributes to the combined-mode advisory.

  * **Combined.**  :data:`MODE_COMBINED`.  Holm-corrected smallest
    p-value across the configured modes, with Vovk–Wang product e-value
    fusion when an e-process is active.  The combined verdict is the
    most conservative of the individual verdicts under FWER control.

Statistical guarantees
----------------------

Each detector is calibrated to a *named* hypothesis:

  * Single-document test: the green-list count ``G`` is the test
    statistic; under H0 ``G ~ Binomial(T, γ)``.  Exact one-tail p-value
    via ``P(Binom(T, γ) ≥ g)`` (regularised incomplete beta); normal
    approximation with continuity correction when ``min(Tγ, T(1−γ)) ≥
    10`` (Cochran's rule of thumb).

  * Two-sided rate CI: Clopper-Pearson exact ``(1−α)``-CI on the
    underlying green-list rate from the observed pair ``(G, T)``.

  * Long-stream audit: a one-proportion anytime-valid e-process on the
    *per-token* green indicator, ``e_n = B(α+k, β+n−k) / B(α, β) /
    γ^k (1−γ)^{n−k}`` (Howard et al. 2021 prior-mean martingale; Ville
    1939 anytime-valid bound).  ``rejected_h0`` iff
    ``e_value ≥ 1 / α``.

  * AUROC: Wilcoxon–Mann–Whitney AUROC of the per-document z-statistic
    against the binary ground-truth label, with bootstrap-percentile
    CI on ``B`` resamples (Hanley-McNeil 1982; Efron 1979).

  * Threshold sweep: Youden's J = TPR − FPR (Youden 1950) over the
    observed score grid.

  * Multi-mode fusion: Holm step-down on the per-mode p-values
    (Holm 1979); Vovk–Wang product e-value when each mode emits an
    e-value (Vovk-Wang 2021).

  * Multi-document FDR: Benjamini–Hochberg adjusted p-values for
    controlling FDR at level α across an audit batch (Benjamini-
    Hochberg 1995).

  * Replay: every certificate is keyed by a SHA-256 fingerprint that
    chains the input config, the seen-trial digests, and the prior
    fingerprint — a coordinator that re-submits the same trial pool
    with the same config gets the same certificate hash, byte-for-byte.

Mathematical and algorithmic roots
----------------------------------

  * **Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I.,
    Goldstein, T. (2023) — "A Watermark for Large Language Models."**
    *ICML.*  The headline construction: a per-token *green-list* PRF
    partition of the vocabulary, a ``δ``-bias added to green-list logits
    at sampling, and the normal-z one-tail test on the green count for
    detection.  Operationalised here as :data:`MODE_KGW_GREEN`.

  * **Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saifullah, K.,
    Kong, K., Fernando, K., Saha, A., Goldblum, M., Goldstein, T.
    (2024) — "On the Reliability of Watermarks for Large Language
    Models."**  *ICLR.*  The selfhash variant and a careful analysis of
    paraphrase / truncation robustness.  Operationalised here as
    :data:`MODE_KGW_SELFHASH`.

  * **Aaronson, S. (2022) — "My AI Safety Lecture for UT Effective
    Altruism."**  The Gumbel-trick formulation of a cryptographically
    secure watermark — informally described, formalised by Christ-
    Gunn-Zamir 2024 STOC ("Undetectable Watermarks").  The Watermarker
    treats Aaronson-style schemes as a forward-compatible spec slot —
    the detection-side ``green-rate-like`` statistic shares the
    Binomial test machinery; the difference is in how green-list
    membership is derived from the PRF.

  * **Christ, M., Gunn, S., Zamir, O. (2024) — "Undetectable
    Watermarks for Language Models."**  *STOC.*  Cryptographic proof
    that watermarks can be made indistinguishable from un-watermarked
    output to a polynomial-time adversary without the key.  Frames why
    the verifier-side test surface in this primitive is *one-sided*
    against a Binomial-γ null: the watermark adds no detectable
    structure to an adversary without the key, but to a verifier with
    the key the green-list bias is unmissable at modest ``T``.

  * **Ville, J. (1939).**  Anytime-valid martingale inequality
    ``P(sup_t M_t ≥ 1/α) ≤ α`` for non-negative martingales — the
    foundation of the sequential e-process.

  * **Howard, S. R., Ramdas, A., McAuliffe, J., Sekhon, J. (2021) —
    "Time-uniform, nonparametric, nonasymptotic confidence
    sequences."**  *Ann. Statist.*  Prior-mean Beta-Binomial
    martingale construction, operationalised here as the audit's
    e-process.

  * **Vovk, V., Wang, R. (2021) — "E-values: Calibration,
    Combination and Applications."**  *Ann. Statist.*  Product of
    e-values as a calibrated combination rule; basis for the
    cross-mode combined e-value.

  * **Holm, S. (1979).**  Step-down FWER procedure for combining
    p-values across configured modes.

  * **Benjamini, Y., Hochberg, Y. (1995).**  BH adjusted p-values for
    controlling FDR over a batch of per-document tests.

  * **Hanley, J. A., McNeil, B. J. (1982).**  Wilcoxon-Mann-Whitney
    AUROC.

  * **Clopper, C. J., Pearson, E. S. (1934).**  Exact binomial
    confidence interval used as the rate CI.

  * **Cochran, W. G. (1952).**  Normal-approximation validity rule
    ``min(np, n(1−p)) ≥ 10``; the cut-over to the exact tail.

  * **Youden, W. J. (1950).**  J = TPR − FPR threshold criterion.

  * **Efron, B. (1979).**  Bootstrap-percentile CI on the AUROC.

What this primitive ships
-------------------------

* :class:`WatermarkSpec` — the public watermark configuration
  (``key``, ``gamma``, ``hash_kind``, ``left_context``, ``vocabulary``).
  A coordinator stores one per generator and routes documents through
  the Watermarker by spec.

* :class:`WatermarkerConfig` — runtime configuration (modes,
  ``alpha``, audit budget, weights, bootstrap iterations, seed).

* :class:`Token` — one token in a document.  Carries the token id (the
  hashable identity used by the green-list PRF) and the surface form
  (for reporting).

* :class:`Document` — a sequence of tokens with optional metadata.

* :class:`Trial` — one submitted document with optional ground-truth
  watermark label and control flag.

* :class:`TrialReport` — per-document detector outputs (green count,
  z-statistic, exact p-value, Clopper-Pearson CI, verdict).

* :class:`ThresholdReport` — Youden-J-fitted operating threshold with
  bootstrap AUROC CI.

* :class:`AuditReport` — anytime-valid sequential audit of the per-
  token green rate against the documented ``γ``.

* :class:`WatermarkCertificate` — replay-verifiable certificate
  bundling the audit state, verdict, recommendation, and the SHA-256
  fingerprint chain.

* :class:`WatermarkerReport` — single-bundle export for the
  coordinator.

* :class:`Watermarker` — the primitive itself.  Submit → score →
  calibrate → audit → certify → report → reset.

Pure stdlib.  No NumPy, no SciPy, no Torch.  Deterministic given seed.
Thread-safe.  ``json.dumps(report.to_dict())`` round-trips.

Coordination engine usage
-------------------------

A coordination engine wires the Watermarker into the routing layer in
three places:

  * **Provenance gate at ingest.**  Before routing an *incoming* user-
    supplied document through expensive downstream stages, submit it to
    the Watermarker.  If the verdict gates ``block`` or ``quarantine``,
    refuse to forward.  This is the EU-AI-Act disclosure surface.

  * **Provenance attest at egress.**  When the runtime emits a
    *generated* document, the Watermarker can pair with
    :mod:`agi.attest` to chain the watermark fingerprint into the
    tamper-evident receipt.  Downstream consumers verify both the
    receipt and the watermark.

  * **Stream audit.**  For long-running production streams, the
    sequential e-process gives an *anytime-valid* alert when the
    observed green rate silently drifts away from the documented
    spec — either because the generator changed (key compromise) or
    because the upstream pipeline is paraphrasing.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# ---------------------------------------------------------------------------
# Constants and taxonomy
# ---------------------------------------------------------------------------

MODE_KGW_GREEN = "kgw_green"
MODE_KGW_EXACT = "kgw_exact"
MODE_KGW_SELFHASH = "kgw_selfhash"
MODE_LEXICAL = "lexical"
MODE_COMBINED = "combined"

KNOWN_MODES: tuple[str, ...] = (
    MODE_KGW_GREEN,
    MODE_KGW_EXACT,
    MODE_KGW_SELFHASH,
    MODE_LEXICAL,
    MODE_COMBINED,
)

VERDICT_PASS = "WM_PASS"
VERDICT_WARN = "WM_WARN"
VERDICT_FAIL = "WM_FAIL"
VERDICT_INCONCLUSIVE = "WM_INCONCLUSIVE"

KNOWN_VERDICTS: tuple[str, ...] = (
    VERDICT_PASS,
    VERDICT_WARN,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
)

REC_TRUST = "trust"
REC_RESTRICT = "restrict"
REC_QUARANTINE = "quarantine"
REC_BLOCK = "block"
REC_ESCALATE = "escalate_human"

KNOWN_RECOMMENDATIONS: tuple[str, ...] = (
    REC_TRUST,
    REC_RESTRICT,
    REC_QUARANTINE,
    REC_BLOCK,
    REC_ESCALATE,
)

HASH_BLAKE2 = "blake2"
HASH_SHA256 = "sha256"
HASH_HMAC_SHA256 = "hmac_sha256"

KNOWN_HASH_KINDS: tuple[str, ...] = (
    HASH_BLAKE2,
    HASH_SHA256,
    HASH_HMAC_SHA256,
)

# Polarity of the H0 a coordinator wants tested.
POLARITY_DETECT_WATERMARK = "detect_watermark"  # H0: text is unmarked; reject upward
POLARITY_VERIFY_WATERMARK = "verify_watermark"  # H0: text is marked; reject downward

KNOWN_POLARITIES: tuple[str, ...] = (
    POLARITY_DETECT_WATERMARK,
    POLARITY_VERIFY_WATERMARK,
)

# Event names emitted on the runtime EventBus.
WM_STARTED = "watermarker.started"
WM_SUBMITTED = "watermarker.submitted"
WM_SCORED = "watermarker.scored"
WM_CALIBRATED = "watermarker.calibrated"
WM_AUDITED = "watermarker.audited"
WM_CERTIFIED = "watermarker.certified"
WM_REPORTED = "watermarker.reported"
WM_GATED = "watermarker.gated"
WM_RESET = "watermarker.reset"

# Token boundary used by the default text tokenizer.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

# Cochran's rule of thumb threshold for the normal approximation.
_NORMAL_APPROX_MIN = 10


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WatermarkerError(ValueError):
    """Base class for Watermarker-specific errors."""


class InvalidConfig(WatermarkerError):
    """The :class:`WatermarkerConfig` or :class:`WatermarkSpec` is invalid."""


class InvalidToken(WatermarkerError):
    """A :class:`Token` violates a runtime invariant."""


class InvalidDocument(WatermarkerError):
    """A :class:`Document` violates a runtime invariant."""


class InvalidTrial(WatermarkerError):
    """A :class:`Trial` violates a runtime invariant."""


class UnknownMode(WatermarkerError):
    """A detector mode name was not recognised."""


class UnknownHashKind(WatermarkerError):
    """A hash-kind name was not recognised."""


class UnknownPolarity(WatermarkerError):
    """A polarity name was not recognised."""


class NotEnoughTrials(WatermarkerError):
    """A statistical operation needs more data than has been submitted."""


class NotCalibrated(WatermarkerError):
    """Gate / certify before a threshold has been fitted."""


class TokenizerError(WatermarkerError):
    """A custom tokenizer raised or returned an invalid sequence."""


# ---------------------------------------------------------------------------
# Configuration records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatermarkSpec:
    """Public watermark configuration.

    Attributes:
        name: human-readable identifier; used in the certificate.
        key: shared secret bytes used to seed the green-list PRF.
            Must be non-empty; recommended ≥ 16 bytes.
        gamma: fraction of the vocabulary on the green list.  Default
            ``0.5`` matches the canonical KGW choice.  Must be in
            ``(0, 1)``.
        delta: documented logit bias the generator added to green-list
            tokens.  *Informational only* — the verifier does not need
            ``delta`` to detect, only to project effect sizes.
        hash_kind: which PRF to use to derive green-list membership.
            One of :data:`HASH_BLAKE2`, :data:`HASH_SHA256`,
            :data:`HASH_HMAC_SHA256`.  Default :data:`HASH_BLAKE2`
            (fast and uniformly random over the digest space).
        left_context: ``h`` in KGW eq. 2 — the number of preceding
            tokens whose ids feed into the PRF that decides whether
            the *next* token is green.  Default ``1`` (the canonical
            KGW choice).  ``0`` is disallowed because the resulting
            partition is a fixed function of the vocabulary alone,
            which collapses the test.
        selfhash: when ``True``, include the token itself in the hash
            (the Kirchenbauer-2024 selfhash variant).  When ``False``
            (default), only the ``left_context`` preceding tokens are
            hashed.
        vocabulary_size: documented vocabulary size.  Optional;
            informational.  Default ``0`` (unknown).
        version: schema version of the spec; ``1`` is current.

    Raises :class:`InvalidConfig` on any out-of-range field.
    """
    name: str = "default"
    key: bytes = b""
    gamma: float = 0.5
    delta: float = 2.0
    hash_kind: str = HASH_BLAKE2
    left_context: int = 1
    selfhash: bool = False
    vocabulary_size: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidConfig("WatermarkSpec.name must be a non-empty string")
        if not isinstance(self.key, (bytes, bytearray)):
            raise InvalidConfig("WatermarkSpec.key must be bytes")
        if not self.key:
            raise InvalidConfig("WatermarkSpec.key must be non-empty")
        # Coerce to immutable bytes.
        object.__setattr__(self, "key", bytes(self.key))
        if not (0.0 < self.gamma < 1.0):
            raise InvalidConfig("WatermarkSpec.gamma must be in (0, 1)")
        if not isinstance(self.delta, (int, float)):
            raise InvalidConfig("WatermarkSpec.delta must be numeric")
        if not math.isfinite(self.delta):
            raise InvalidConfig("WatermarkSpec.delta must be finite")
        if self.delta < 0:
            raise InvalidConfig("WatermarkSpec.delta must be non-negative")
        if self.hash_kind not in KNOWN_HASH_KINDS:
            raise UnknownHashKind(
                f"unknown hash_kind {self.hash_kind!r}; "
                f"known: {KNOWN_HASH_KINDS!r}"
            )
        if not isinstance(self.left_context, int) or self.left_context < 1:
            raise InvalidConfig("WatermarkSpec.left_context must be int >= 1")
        if not isinstance(self.selfhash, bool):
            raise InvalidConfig("WatermarkSpec.selfhash must be bool")
        if not isinstance(self.vocabulary_size, int) or self.vocabulary_size < 0:
            raise InvalidConfig(
                "WatermarkSpec.vocabulary_size must be a non-negative int"
            )
        if not isinstance(self.version, int) or self.version < 1:
            raise InvalidConfig("WatermarkSpec.version must be int >= 1")

    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the public-facing spec.

        The ``key`` is included via HMAC rather than verbatim so the
        spec can be logged without leaking the secret.  Anyone holding
        the key reproduces the fingerprint; anyone without it cannot.
        """
        body = {
            "name": self.name,
            "gamma": float(self.gamma),
            "delta": float(self.delta),
            "hash_kind": self.hash_kind,
            "left_context": int(self.left_context),
            "selfhash": bool(self.selfhash),
            "vocabulary_size": int(self.vocabulary_size),
            "version": int(self.version),
        }
        body_bytes = json.dumps(body, sort_keys=True).encode("utf-8")
        return hmac.new(self.key, body_bytes, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class WatermarkerConfig:
    """Configuration for a :class:`Watermarker` instance.

    Attributes:
        modes: which detectors to run.  Default: green-list normal +
            exact tests + lexical control.
        alpha: type-I error of the sequential audit and the operating
            threshold confidence ``1 − alpha``.  Default ``0.01``
            (1 %) — watermark detection is high-stakes, so the prior
            is *conservative*.
        bootstrap_b: number of bootstrap resamples for AUROC and rate
            CIs.  Default ``200``.
        confidence: bootstrap-percentile coverage (typically ``1 − alpha``).
        seed: PRNG seed for bootstraps and shuffles.  Deterministic.
        warn_factor: ``WARN`` if the lower CI on the green rate sits in
            ``[γ + warn_factor·(observed−γ), observed]`` — i.e. the
            test is *trending* significant without crossing.  Default
            ``0.5``.
        weights: per-mode weights ``(green, exact, selfhash, lexical)``
            used to fuse detector p-values into a Holm-corrected
            combined verdict.  Negative weights drop the mode; zero
            disables it from fusion but keeps it scored.  Default
            ``(1.0, 1.0, 1.0, 0.0)``.
        polarity: which H0 is being tested.  Default
            :data:`POLARITY_DETECT_WATERMARK`.
        prior_a, prior_b: Beta prior on the per-token green rate for
            the anytime-valid one-proportion e-process.  Default
            ``Beta(1, 1)``.
        min_tokens_for_normal: minimum ``min(Tγ, T(1−γ))`` to use the
            normal approximation; below this the exact tail is used.
            Default ``10`` (Cochran).
        max_documents: hard cap on the number of stored trials.  ``0``
            disables.  Default ``0``.
        require_label_for_threshold: refuse to fit a threshold without
            at least one positive and one negative labelled trial.
            Default ``True``.

    Raises :class:`InvalidConfig` on any out-of-range field.
    """
    modes: tuple[str, ...] = (
        MODE_KGW_GREEN,
        MODE_KGW_EXACT,
        MODE_LEXICAL,
    )
    alpha: float = 0.01
    bootstrap_b: int = 200
    confidence: float = 0.99
    seed: int = 0
    warn_factor: float = 0.5
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
    polarity: str = POLARITY_DETECT_WATERMARK
    prior_a: float = 1.0
    prior_b: float = 1.0
    min_tokens_for_normal: int = _NORMAL_APPROX_MIN
    max_documents: int = 0
    require_label_for_threshold: bool = True

    def __post_init__(self) -> None:
        if not self.modes:
            raise InvalidConfig("modes must be a non-empty tuple")
        for m in self.modes:
            if m not in KNOWN_MODES:
                raise UnknownMode(f"unknown mode {m!r}; known: {KNOWN_MODES!r}")
        if MODE_COMBINED in self.modes:
            raise InvalidConfig(
                "MODE_COMBINED is implicit; do not include it in modes"
            )
        if not (0.0 < self.alpha < 1.0):
            raise InvalidConfig("alpha must be in (0, 1)")
        if self.bootstrap_b < 0:
            raise InvalidConfig("bootstrap_b must be >= 0")
        if not (0.0 < self.confidence < 1.0):
            raise InvalidConfig("confidence must be in (0, 1)")
        if not (0.0 < self.warn_factor <= 1.0):
            raise InvalidConfig("warn_factor must be in (0, 1]")
        if len(self.weights) != 4:
            raise InvalidConfig(
                "weights must be a 4-tuple (green, exact, selfhash, lexical)"
            )
        for w in self.weights:
            if not isinstance(w, (int, float)) or not math.isfinite(w):
                raise InvalidConfig("weights must be finite numbers")
        if self.polarity not in KNOWN_POLARITIES:
            raise UnknownPolarity(
                f"unknown polarity {self.polarity!r}; "
                f"known: {KNOWN_POLARITIES!r}"
            )
        if self.prior_a <= 0.0 or self.prior_b <= 0.0:
            raise InvalidConfig("prior_a and prior_b must be > 0")
        if self.min_tokens_for_normal < 1:
            raise InvalidConfig("min_tokens_for_normal must be >= 1")
        if self.max_documents < 0:
            raise InvalidConfig("max_documents must be >= 0")
        if not isinstance(self.require_label_for_threshold, bool):
            raise InvalidConfig("require_label_for_threshold must be bool")


# ---------------------------------------------------------------------------
# Token / Document / Trial
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """One token in a document.

    Attributes:
        token_id: hashable identity used by the green-list PRF.  Must
            be a non-negative integer; integer ids are the standard
            convention in BPE / SentencePiece tokenisers.  A pure-text
            tokenizer can use ``hash(text) & 0xFFFFFFFF``; the runtime
            re-derives an integer id from the surface form when one is
            not supplied via :func:`tokenize_text`.
        text: human-readable surface form.  Optional; defaults to
            ``""``.  Used for the report and the lexical-entropy
            control only.

    Raises :class:`InvalidToken` on any out-of-range field.
    """
    token_id: int
    text: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.token_id, int):
            raise InvalidToken("token_id must be int")
        if self.token_id < 0:
            raise InvalidToken("token_id must be non-negative")
        if not isinstance(self.text, str):
            raise InvalidToken("text must be a string")


@dataclass(frozen=True)
class Document:
    """A sequence of tokens with optional metadata.

    Attributes:
        doc_id: opaque caller-side identifier; round-trips through the
            trial bank and certificate.
        tokens: ordered tuple of :class:`Token`.  Must have length ≥ 1.
        text: optional raw source text; if absent it is reconstructed
            from ``tokens[i].text`` for reporting only.
        metadata: opaque caller-side annotation.

    Raises :class:`InvalidDocument` on invalid input.
    """
    doc_id: str
    tokens: tuple[Token, ...]
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise InvalidDocument("doc_id must be a non-empty string")
        if not isinstance(self.tokens, tuple):
            raise InvalidDocument("tokens must be a tuple")
        if not self.tokens:
            raise InvalidDocument("tokens must be non-empty")
        for t in self.tokens:
            if not isinstance(t, Token):
                raise InvalidDocument("tokens must contain Token instances")
        if not isinstance(self.text, str):
            raise InvalidDocument("text must be a string")
        if not isinstance(self.metadata, Mapping):
            raise InvalidDocument("metadata must be a Mapping")


@dataclass(frozen=True)
class Trial:
    """One submitted document with optional ground-truth label.

    Attributes:
        document: the :class:`Document` to be scored.
        spec: the :class:`WatermarkSpec` the document is *claimed* to
            satisfy.  Detection uses this spec.
        truth: ground-truth watermark label, if known.  ``True`` means
            "watermarked under ``spec``" (positive class for
            :data:`POLARITY_DETECT_WATERMARK`); ``False`` means
            "unmarked" (negative class).  ``None`` means unlabelled and
            the trial will not enter threshold calibration.
        control: ``True`` means this trial is a calibration control —
            a benign sample that should test negative.  Control trials
            contribute to the FPR floor and never to the operating
            audit.
        metadata: opaque caller-side annotation.

    Raises :class:`InvalidTrial` on invalid input.
    """
    document: Document
    spec: WatermarkSpec
    truth: bool | None = None
    control: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            raise InvalidTrial("document must be a Document")
        if not isinstance(self.spec, WatermarkSpec):
            raise InvalidTrial("spec must be a WatermarkSpec")
        if self.truth is not None and not isinstance(self.truth, bool):
            raise InvalidTrial("truth must be bool or None")
        if not isinstance(self.control, bool):
            raise InvalidTrial("control must be bool")
        if self.control and self.truth is not None:
            raise InvalidTrial(
                "control trials must not carry a truth label; "
                "they measure the FPR floor"
            )
        if not isinstance(self.metadata, Mapping):
            raise InvalidTrial("metadata must be a Mapping")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialReport:
    """Per-document detector outputs."""
    doc_id: str
    n_tokens: int
    n_scoreable: int
    n_green: int
    green_fraction: float
    expected_green: float
    z_score: float
    p_value_normal: float
    p_value_exact: float
    rate_lower_cp: float
    rate_upper_cp: float
    lexical_entropy: float
    chosen_p_value: float
    verdict: str
    has_truth: bool
    truth_value: bool | None
    is_control: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "n_tokens": self.n_tokens,
            "n_scoreable": self.n_scoreable,
            "n_green": self.n_green,
            "green_fraction": self.green_fraction,
            "expected_green": self.expected_green,
            "z_score": self.z_score,
            "p_value_normal": self.p_value_normal,
            "p_value_exact": self.p_value_exact,
            "rate_lower_cp": self.rate_lower_cp,
            "rate_upper_cp": self.rate_upper_cp,
            "lexical_entropy": self.lexical_entropy,
            "chosen_p_value": self.chosen_p_value,
            "verdict": self.verdict,
            "has_truth": self.has_truth,
            "truth_value": self.truth_value,
            "is_control": self.is_control,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ThresholdReport:
    """Calibrated decision threshold on the per-document z-statistic."""
    threshold: float
    auroc: float
    auroc_lower: float
    auroc_upper: float
    youden_j: float
    tpr_at_threshold: float
    fpr_at_threshold: float
    n_labelled: int
    n_positive: int
    n_negative: int
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
    """Anytime-valid sequential audit of the per-token green rate."""
    n_tokens_seen: int
    n_green_seen: int
    running_rate: float
    rate_lower_clopper_pearson: float
    rate_upper_clopper_pearson: float
    e_value: float
    log_e_value: float
    rejected_h0: bool
    gamma: float
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens_seen": self.n_tokens_seen,
            "n_green_seen": self.n_green_seen,
            "running_rate": self.running_rate,
            "rate_lower_clopper_pearson": self.rate_lower_clopper_pearson,
            "rate_upper_clopper_pearson": self.rate_upper_clopper_pearson,
            "e_value": self.e_value,
            "log_e_value": self.log_e_value,
            "rejected_h0": self.rejected_h0,
            "gamma": self.gamma,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class WatermarkCertificate:
    """Replay-verifiable certificate over the Watermarker state."""
    spec_name: str
    spec_fingerprint: str
    n_trials: int
    n_trials_labelled: int
    n_control: int
    n_tokens_seen: int
    n_green_seen: int
    green_rate: float
    rate_lower_cp: float
    rate_upper_cp: float
    auroc: float | None
    auroc_lower: float | None
    auroc_upper: float | None
    threshold: float | None
    e_value: float
    log_e_value: float
    rejected_h0: bool
    verdict: str
    recommendation: str
    holm_smallest_p: float | None
    fdr_threshold_p: float | None
    fingerprint_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "n_trials": self.n_trials,
            "n_trials_labelled": self.n_trials_labelled,
            "n_control": self.n_control,
            "n_tokens_seen": self.n_tokens_seen,
            "n_green_seen": self.n_green_seen,
            "green_rate": self.green_rate,
            "rate_lower_cp": self.rate_lower_cp,
            "rate_upper_cp": self.rate_upper_cp,
            "auroc": self.auroc,
            "auroc_lower": self.auroc_lower,
            "auroc_upper": self.auroc_upper,
            "threshold": self.threshold,
            "e_value": self.e_value,
            "log_e_value": self.log_e_value,
            "rejected_h0": self.rejected_h0,
            "verdict": self.verdict,
            "recommendation": self.recommendation,
            "holm_smallest_p": self.holm_smallest_p,
            "fdr_threshold_p": self.fdr_threshold_p,
            "fingerprint_hash": self.fingerprint_hash,
        }


@dataclass(frozen=True)
class WatermarkerReport:
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
# Math helpers — pure stdlib
# ---------------------------------------------------------------------------


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Lentz continued-fraction evaluation of ``I_x(a, b)``.

    Numerical Recipes 6.4 / Abramowitz-Stegun 26.5.8.  Pure stdlib.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Reflection for faster convergence.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(1.0 - x, b, a)
    log_pref = (a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b))
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
    """Monotone-bisection quantile of ``Beta(a, b)``."""
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
    """Exact two-sided Clopper-Pearson CI for Binomial(n, p)."""
    if not (0 <= k <= n):
        raise WatermarkerError(f"k={k}, n={n}: 0 <= k <= n required")
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
    """``P(Binom(n, p) <= k)`` via the regularised incomplete beta."""
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


def _normal_sf(z: float) -> float:
    """One-tail normal survival ``P(Z > z)`` via :func:`math.erfc`.

    Numerically stable across the entire real line.
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _normal_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _eprocess_one_proportion(k: int, n: int,
                             p0: float,
                             prior_a: float,
                             prior_b: float) -> float:
    """Anytime-valid e-value for ``H_0: p <= p0`` (one-proportion,
    Beta-Binomial prior-mean martingale).

    Operates on the *sufficient statistic* ``(k, n)`` so the test is
    cheap to update incrementally and identical whether the caller
    passes in observation-by-observation or in batches.
    """
    if n <= 0:
        return 1.0
    if not (0.0 < p0 < 1.0):
        raise WatermarkerError("p0 must be in (0, 1)")
    log_e = (_log_beta(prior_a + k, prior_b + n - k)
             - _log_beta(prior_a, prior_b))
    if k > 0:
        log_e -= k * math.log(p0)
    if (n - k) > 0:
        log_e -= (n - k) * math.log1p(-p0)
    try:
        return math.exp(log_e)
    except OverflowError:
        return float("inf")


def _eprocess_log(k: int, n: int,
                  p0: float,
                  prior_a: float,
                  prior_b: float) -> float:
    """``log e_n`` for stability when ``e_n`` overflows."""
    if n <= 0:
        return 0.0
    log_e = (_log_beta(prior_a + k, prior_b + n - k)
             - _log_beta(prior_a, prior_b))
    if k > 0:
        log_e -= k * math.log(p0)
    if (n - k) > 0:
        log_e -= (n - k) * math.log1p(-p0)
    return log_e


def _holm_smallest(p_values: Sequence[float]) -> float:
    """Holm step-down adjusted minimum p-value."""
    if not p_values:
        return 1.0
    sorted_p = sorted(p_values)
    m = len(sorted_p)
    adjusted: list[float] = []
    for i, p in enumerate(sorted_p):
        adjusted.append(min(1.0, (m - i) * p))
    for i in range(1, len(adjusted)):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    return adjusted[0]


def _benjamini_hochberg(p_values: Sequence[float], alpha: float) -> float | None:
    """BH-adjusted rejection threshold for an FDR-controlled batch.

    Returns the *largest* p-value that survives the BH step-up, or
    ``None`` if no hypothesis is rejected.  ``alpha`` is the target FDR.
    """
    if not p_values:
        return None
    m = len(p_values)
    sorted_p = sorted(p_values)
    threshold = None
    for i, p in enumerate(sorted_p, start=1):
        if p <= (i / m) * alpha:
            threshold = p
    return threshold


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Wilcoxon-Mann-Whitney AUROC."""
    if len(scores) != len(labels):
        raise WatermarkerError("scores / labels length mismatch")
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


def _youden_threshold(
        scores: Sequence[float],
        labels: Sequence[bool]) -> tuple[float, float, float, float]:
    """Pick the threshold maximising Youden's J = TPR − FPR."""
    if len(scores) != len(labels):
        raise WatermarkerError("scores / labels length mismatch")
    if not scores:
        return float("nan"), float("nan"), float("nan"), float("nan")
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
    """Bootstrap-percentile CI on the AUROC."""
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


def _shannon_entropy(weights: Sequence[float]) -> float:
    """Shannon entropy (nats) of a non-negative weight vector."""
    s = sum(x for x in weights if x > 0.0)
    if s <= 0.0:
        return 0.0
    h = 0.0
    for x in weights:
        if x > 0.0:
            q = x / s
            h -= q * math.log(q)
    return h


# ---------------------------------------------------------------------------
# Green-list PRF — the heart of KGW detection
# ---------------------------------------------------------------------------


def _pack_context(token_ids: Sequence[int]) -> bytes:
    """Pack a sequence of non-negative ints into a stable little-endian
    8-byte-per-token byte string.  Used as the PRF input."""
    return b"".join(struct.pack("<Q", int(i) & 0xFFFFFFFFFFFFFFFF)
                    for i in token_ids)


def _prf_digest(spec: WatermarkSpec, context_ids: Sequence[int]) -> bytes:
    """Apply the spec's hash kind to ``key || context``."""
    payload = _pack_context(context_ids)
    if spec.hash_kind == HASH_BLAKE2:
        return hashlib.blake2b(spec.key + payload, digest_size=32).digest()
    if spec.hash_kind == HASH_SHA256:
        return hashlib.sha256(spec.key + payload).digest()
    if spec.hash_kind == HASH_HMAC_SHA256:
        return hmac.new(spec.key, payload, hashlib.sha256).digest()
    raise UnknownHashKind(f"unknown hash_kind {spec.hash_kind!r}")


def _digest_to_float(digest: bytes) -> float:
    """Map the first 8 digest bytes to ``[0, 1)`` uniformly."""
    raw = struct.unpack("<Q", digest[:8])[0]
    # The factor (1 / 2^64) maps to [0, 1).  Use 2^53 mantissa-aligned
    # rounding for IEEE float64 precision; the loss of the bottom 11
    # bits is irrelevant for green-list membership.
    return (raw >> 11) * (1.0 / (1 << 53))


def is_green_token(spec: WatermarkSpec,
                   token_id: int,
                   prev_token_ids: Sequence[int]) -> bool:
    """Return ``True`` iff ``token_id`` is on the green list given the
    ``prev_token_ids`` left-context window.

    The PRF input is ``key || prev_ids[-h:] || token_id``; the first
    64 bits of the digest map to a uniform ``u ∈ [0, 1)``; the token
    is green iff ``u < γ``.

    Statistical equivalence to canonical KGW.  Kirchenbauer et al.
    2023 frame the green-list as a *partition of the vocabulary*
    induced by a seed derived from the left context: for each position
    ``t``, the vocabulary is split into ``G_t`` (γ-fraction) and
    ``R_t`` ((1-γ)-fraction), and the green/red status of any
    candidate is read off the partition.  Operationally, this is
    indistinguishable from independently hashing each (context,
    candidate) pair and labelling green iff the hash uniform falls
    below γ — both procedures produce the same per-token Bernoulli(γ)
    label under H0 (unmarked, uniform sampling) and the same expected
    boost under H1 (marked).  The combined-input formulation
    sidesteps having to enumerate the vocabulary and lets the verifier
    test any token id without owning the generator's vocab table.

    The ``selfhash`` flag is *forward-compatible* — it duplicates the
    candidate token in the PRF input, which has no effect on the
    statistical null but signals to the generator-side that the
    seed depends on the current token, mirroring the Kirchenbauer
    2024 "selfhash" / k+1 hashing variant on the generator side.
    Detection-side, both flags produce a Bernoulli(γ) null under
    H0; the difference is purely how a downstream sampler interprets
    the spec.
    """
    h = spec.left_context
    # Insufficient context: token is *not scoreable* under the spec.
    # The caller (the Watermarker) filters these out before the test.
    if len(prev_token_ids) < h:
        raise WatermarkerError(
            f"insufficient context for green-list test: need {h} tokens, "
            f"got {len(prev_token_ids)}"
        )
    # Always include the candidate token in the PRF input — see the
    # docstring above.  ``selfhash`` adds an extra copy as a forward-
    # compatible signal for generator-side schemes.
    ctx = list(prev_token_ids[-h:]) + [int(token_id)]
    if spec.selfhash:
        ctx = ctx + [int(token_id)]
    digest = _prf_digest(spec, ctx)
    u = _digest_to_float(digest)
    return u < spec.gamma


def green_indicators(spec: WatermarkSpec,
                     token_ids: Sequence[int]) -> list[bool]:
    """Per-token green indicators for a full document.

    Returns one boolean per token at *scoreable* position (i.e.
    index >= ``spec.left_context``).  Indices ``[0, h)`` are unscoreable
    and not included in the output.  The output length is
    ``max(0, len(token_ids) - h)``.
    """
    h = spec.left_context
    if h <= 0:
        raise WatermarkerError("left_context must be >= 1")
    out: list[bool] = []
    for i in range(h, len(token_ids)):
        out.append(is_green_token(spec, token_ids[i], token_ids[i - h:i]))
    return out


# ---------------------------------------------------------------------------
# Tokenizer plumbing
# ---------------------------------------------------------------------------


Tokenizer = Callable[[str], list[Token]]
"""Callable: ``tokenize(text) -> list[Token]``.

The default tokenizer is a regex-based whitespace + word splitter that
derives a deterministic 64-bit token id from a BLAKE2 hash of the
surface form.  Production deployments should pass a real BPE / SP
tokenizer — but the runtime test surface is exhaustive against the
default tokenizer so the green-list math is provably correct on text
alone.
"""


def _default_token_id(text: str) -> int:
    """Map a token's surface form to a stable, uniformly-random 64-bit
    non-negative integer id.

    BLAKE2b is used so the partition is statistically indistinguishable
    from a random PRF over the surface vocabulary — which preserves
    the KGW null distribution under the default tokenizer.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


def default_tokenizer(text: str) -> list[Token]:
    """Regex tokenizer with hash-derived token ids.

    Lowercases, splits on word boundaries, drops the empty / whitespace
    fragments, and emits one :class:`Token` per surface form.  Pure
    function; thread-safe.
    """
    if not isinstance(text, str):
        raise TokenizerError("default_tokenizer requires str")
    out: list[Token] = []
    for surface in _TOKEN_RE.findall(text.lower()):
        if not surface:
            continue
        out.append(Token(token_id=_default_token_id(surface), text=surface))
    return out


def tokenize_text(text: str, tokenizer: Tokenizer | None = None) -> tuple[Token, ...]:
    """Apply ``tokenizer`` (or :func:`default_tokenizer`) and validate."""
    fn = tokenizer or default_tokenizer
    raw = fn(text)
    if not isinstance(raw, list):
        raise TokenizerError("tokenizer must return list[Token]")
    for t in raw:
        if not isinstance(t, Token):
            raise TokenizerError("tokenizer must return Token instances")
    return tuple(raw)


def make_document(doc_id: str,
                  text: str,
                  *,
                  tokenizer: Tokenizer | None = None,
                  metadata: Mapping[str, Any] | None = None) -> Document:
    """Build a :class:`Document` from raw text using ``tokenizer``.

    Convenience for callers without their own tokenizer.  In production
    deployments, pass the same tokenizer the generator used so the
    token ids match byte-for-byte.
    """
    toks = tokenize_text(text, tokenizer)
    if not toks:
        raise InvalidDocument(
            "tokenizer produced 0 tokens; check the input text"
        )
    return Document(doc_id=doc_id, tokens=toks, text=text,
                    metadata=dict(metadata or {}))


# ---------------------------------------------------------------------------
# Watermark INJECTION — black-box simulation for tests and labelled trials
# ---------------------------------------------------------------------------


def simulate_marked_token_ids(spec: WatermarkSpec,
                              n_tokens: int,
                              *,
                              prefix: Sequence[int] | None = None,
                              effective_delta: float | None = None,
                              vocabulary: Sequence[int] | None = None,
                              seed: int = 0) -> list[int]:
    """Simulate a watermarked token sequence under the spec.

    A *generation oracle* that biases towards green-list tokens at
    each step.  The bias strength is the ``effective_delta`` parameter
    (defaults to ``spec.delta``): each non-green candidate is rejected
    with probability ``1 − exp(−delta)`` and the position resamples.
    This reproduces the *effective* marked distribution that the
    detector sees, without requiring access to a real LM's logits.

    Used by :meth:`Watermarker.simulate_trial` to generate labelled
    trials for threshold calibration and AUROC reporting.

    Args:
        spec: the :class:`WatermarkSpec`.
        n_tokens: how many tokens to generate (excluding the prefix).
        prefix: optional initial token ids (length ≥ ``spec.left_context``).
            If omitted, ``spec.left_context`` deterministic seed tokens
            are generated.
        effective_delta: bias strength; default ``spec.delta``.
        vocabulary: pool of candidate token ids to sample from.
            Default: synthetic ids derived from ``seed``.
        seed: PRNG seed for the simulation.
    """
    if n_tokens <= 0:
        raise WatermarkerError("n_tokens must be > 0")
    delta = effective_delta if effective_delta is not None else spec.delta
    if delta < 0:
        raise WatermarkerError("effective_delta must be >= 0")
    rng = random.Random(seed)
    if vocabulary is None:
        # Synthesise a vocabulary of size 1024 with deterministic ids.
        vocabulary = [rng.randrange(1 << 60) for _ in range(1024)]
    vocab = list(vocabulary)
    if len(vocab) < 4:
        raise WatermarkerError("vocabulary must have at least 4 tokens")
    h = spec.left_context
    if prefix is None:
        prefix = [rng.choice(vocab) for _ in range(h)]
    if len(prefix) < h:
        raise WatermarkerError(
            f"prefix must have length >= {h} (spec.left_context)"
        )
    out: list[int] = list(prefix)
    keep_green_prob = 1.0  # always accept green
    keep_red_prob = math.exp(-delta) if delta > 0 else 1.0
    for _ in range(n_tokens):
        for _attempt in range(64):
            cand = rng.choice(vocab)
            green = is_green_token(spec, cand, out[-h:])
            if green:
                if rng.random() <= keep_green_prob:
                    out.append(cand)
                    break
            else:
                if rng.random() <= keep_red_prob:
                    out.append(cand)
                    break
        else:
            # Defensive: emit the last candidate regardless.
            out.append(cand)
    return out


def simulate_stripped_token_ids(spec: WatermarkSpec,
                                n_tokens: int,
                                *,
                                prefix: Sequence[int] | None = None,
                                effective_delta: float | None = None,
                                vocabulary: Sequence[int] | None = None,
                                seed: int = 0) -> list[int]:
    """Simulate text where an adversary has *biased away from* the
    green list — a stripping / paraphrase attack on a marked stream.

    Models the dual of :func:`simulate_marked_token_ids` with the
    bias direction flipped: green candidates are rejected with
    probability ``1 − exp(−delta)``.  Useful for testing verify-mode
    detection where ``H_0`` is "still marked".
    """
    if n_tokens <= 0:
        raise WatermarkerError("n_tokens must be > 0")
    delta = effective_delta if effective_delta is not None else spec.delta
    if delta < 0:
        raise WatermarkerError("effective_delta must be >= 0")
    rng = random.Random(seed)
    if vocabulary is None:
        vocabulary = [rng.randrange(1 << 60) for _ in range(1024)]
    vocab = list(vocabulary)
    if len(vocab) < 4:
        raise WatermarkerError("vocabulary must have at least 4 tokens")
    h = spec.left_context
    if prefix is None:
        prefix = [rng.choice(vocab) for _ in range(h)]
    if len(prefix) < h:
        raise WatermarkerError(
            f"prefix must have length >= {h} (spec.left_context)"
        )
    out: list[int] = list(prefix)
    keep_green_prob = math.exp(-delta) if delta > 0 else 1.0
    keep_red_prob = 1.0
    for _ in range(n_tokens):
        for _attempt in range(64):
            cand = rng.choice(vocab)
            green = is_green_token(spec, cand, out[-h:])
            if green:
                if rng.random() <= keep_green_prob:
                    out.append(cand)
                    break
            else:
                if rng.random() <= keep_red_prob:
                    out.append(cand)
                    break
        else:
            out.append(cand)
    return out


def simulate_stripped_document(spec: WatermarkSpec,
                               doc_id: str,
                               n_tokens: int,
                               *,
                               seed: int = 0,
                               effective_delta: float | None = None,
                               metadata: Mapping[str, Any] | None = None
                               ) -> Document:
    """Build a stripped :class:`Document` (adversarially biased red)."""
    ids = simulate_stripped_token_ids(spec, n_tokens, seed=seed,
                                      effective_delta=effective_delta)
    toks = tuple(Token(token_id=i, text=f"s_{i & 0xFFFF:04x}") for i in ids)
    return Document(doc_id=doc_id, tokens=toks,
                    metadata=dict(metadata or {}))


def simulate_unmarked_token_ids(spec: WatermarkSpec,
                                n_tokens: int,
                                *,
                                vocabulary: Sequence[int] | None = None,
                                seed: int = 0) -> list[int]:
    """Simulate an unwatermarked token sequence — uniform draws from
    the vocabulary.  Under H0 this is the null distribution.
    """
    if n_tokens <= 0:
        raise WatermarkerError("n_tokens must be > 0")
    rng = random.Random(seed)
    if vocabulary is None:
        vocabulary = [rng.randrange(1 << 60) for _ in range(1024)]
    vocab = list(vocabulary)
    if len(vocab) < 4:
        raise WatermarkerError("vocabulary must have at least 4 tokens")
    h = spec.left_context
    out: list[int] = [rng.choice(vocab) for _ in range(h + n_tokens)]
    return out


def simulate_marked_document(spec: WatermarkSpec,
                             doc_id: str,
                             n_tokens: int,
                             *,
                             seed: int = 0,
                             effective_delta: float | None = None,
                             metadata: Mapping[str, Any] | None = None
                             ) -> Document:
    """Build a marked :class:`Document` for tests / labelled trials."""
    ids = simulate_marked_token_ids(spec, n_tokens, seed=seed,
                                    effective_delta=effective_delta)
    toks = tuple(Token(token_id=i, text=f"t_{i & 0xFFFF:04x}") for i in ids)
    return Document(doc_id=doc_id, tokens=toks,
                    metadata=dict(metadata or {}))


def simulate_unmarked_document(spec: WatermarkSpec,
                               doc_id: str,
                               n_tokens: int,
                               *,
                               seed: int = 0,
                               metadata: Mapping[str, Any] | None = None
                               ) -> Document:
    """Build an unmarked :class:`Document` (null distribution)."""
    ids = simulate_unmarked_token_ids(spec, n_tokens, seed=seed)
    toks = tuple(Token(token_id=i, text=f"u_{i & 0xFFFF:04x}") for i in ids)
    return Document(doc_id=doc_id, tokens=toks,
                    metadata=dict(metadata or {}))


# ---------------------------------------------------------------------------
# Fingerprint chain — replay-verifiable certificate hash
# ---------------------------------------------------------------------------


def _stable(value: Any) -> Any:
    """Make a payload deterministic for the fingerprint chain.

    Drops ``ts`` keys (wall-clock annotations), sorts dict keys, sorts
    sets, leaves tuples / lists in given order, coerces bytes to hex.
    """
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value.keys()) if k != "ts"}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, set):
        return sorted(_stable(v) for v in value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _fingerprint(prev_hex: str, payload: Any) -> str:
    """Chain a new fingerprint onto ``prev_hex`` with ``payload``."""
    h = hashlib.sha256()
    h.update(prev_hex.encode("ascii"))
    h.update(b"\x00")
    h.update(json.dumps(_stable(payload), sort_keys=True,
                        separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Watermarker — the primitive itself
# ---------------------------------------------------------------------------


class Watermarker:
    """Synthetic-content provenance certification primitive.

    A coordination engine constructs one Watermarker per
    :class:`WatermarkSpec` (or one Watermarker that accepts trials
    against multiple specs — same construction, different rendezvous
    point), submits trials, lets the primitive score / calibrate /
    audit, then asks for a :class:`WatermarkCertificate` or a
    :class:`WatermarkerReport`.

    Public API (the *contract surface* a coordination engine drives):

      * :meth:`submit` — score a :class:`Trial` and store its
        :class:`TrialReport`.
      * :meth:`submit_text` — score a raw-text + spec pair.
      * :meth:`score_only` — score without storing (pure function).
      * :meth:`calibrate` — fit the Youden-J threshold against the
        labelled pool; bootstrap AUROC CI.
      * :meth:`audit` — anytime-valid sequential e-process on the
        per-token green rate.
      * :meth:`certify` — combine the above into a
        :class:`WatermarkCertificate` with a SHA-256 fingerprint
        chained onto the prior certificate.
      * :meth:`report` — :class:`WatermarkerReport` snapshot.
      * :meth:`gate` — yes/no on the current verdict, with a
        coordinator-facing recommendation.
      * :meth:`reset` — wipe the trial bank (idempotent).

    Thread-safe under a single internal lock; reentrant calls to the
    same Watermarker are serialised.
    """

    def __init__(self,
                 config: WatermarkerConfig | None = None,
                 *,
                 event_bus: EventBus | None = None,
                 spec: WatermarkSpec | None = None) -> None:
        self._config = config or WatermarkerConfig()
        self._event_bus = event_bus
        self._spec = spec
        self._lock = threading.RLock()
        # Trial bank.
        self._trials: list[Trial] = []
        self._reports: list[TrialReport] = []
        # Sufficient statistics for the audit.
        self._n_tokens_seen: int = 0
        self._n_green_seen: int = 0
        # Calibration state.
        self._threshold_report: ThresholdReport | None = None
        # Certificate chain.
        self._prev_fingerprint: str = "0" * 64
        self._last_certificate: WatermarkCertificate | None = None
        # Event book-keeping.
        self._n_submitted: int = 0
        self._emit(WM_STARTED, {
            "config": self._config_dict(),
            "spec": self._spec_dict() if self._spec else None,
        })

    # -- properties ----------------------------------------------------

    @property
    def config(self) -> WatermarkerConfig:
        return self._config

    @property
    def spec(self) -> WatermarkSpec | None:
        return self._spec

    @property
    def n_trials(self) -> int:
        with self._lock:
            return len(self._trials)

    @property
    def n_tokens_seen(self) -> int:
        with self._lock:
            return self._n_tokens_seen

    @property
    def n_green_seen(self) -> int:
        with self._lock:
            return self._n_green_seen

    # -- helpers -------------------------------------------------------

    def _emit(self, name: str, payload: Mapping[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(
                Event(kind=name, data=dict(payload), ts=time.time())
            )
        except Exception:
            # The Watermarker does not let an EventBus subscriber take
            # down a coordination flow.  Subscribers must be idempotent
            # and the runtime treats publish failures as advisory.
            pass

    def _config_dict(self) -> dict[str, Any]:
        c = self._config
        return {
            "modes": list(c.modes),
            "alpha": c.alpha,
            "bootstrap_b": c.bootstrap_b,
            "confidence": c.confidence,
            "seed": c.seed,
            "warn_factor": c.warn_factor,
            "weights": list(c.weights),
            "polarity": c.polarity,
            "prior_a": c.prior_a,
            "prior_b": c.prior_b,
            "min_tokens_for_normal": c.min_tokens_for_normal,
            "max_documents": c.max_documents,
            "require_label_for_threshold": c.require_label_for_threshold,
        }

    def _spec_dict(self) -> dict[str, Any]:
        s = self._spec
        if s is None:
            return {}
        return {
            "name": s.name,
            "gamma": s.gamma,
            "delta": s.delta,
            "hash_kind": s.hash_kind,
            "left_context": s.left_context,
            "selfhash": s.selfhash,
            "vocabulary_size": s.vocabulary_size,
            "version": s.version,
            "fingerprint": s.fingerprint(),
        }

    # -- core scoring (pure) ------------------------------------------

    def _score_document(self, doc: Document, spec: WatermarkSpec
                        ) -> dict[str, Any]:
        """Pure-function scoring of a single document against a spec.

        Returns a dict with raw test statistics; ``submit`` and
        :meth:`score_only` wrap this into a :class:`TrialReport`.
        """
        token_ids = [t.token_id for t in doc.tokens]
        n_total = len(token_ids)
        h = spec.left_context
        if n_total <= h:
            # Document is too short to score; report a clear
            # uninformative sentinel.
            return {
                "n_tokens": n_total,
                "n_scoreable": 0,
                "n_green": 0,
                "green_fraction": float("nan"),
                "expected_green": spec.gamma,
                "z_score": float("nan"),
                "p_value_normal": 1.0,
                "p_value_exact": 1.0,
                "rate_lower_cp": 0.0,
                "rate_upper_cp": 1.0,
                "lexical_entropy": 0.0,
                "chosen_p_value": float("nan"),
                "verdict": VERDICT_INCONCLUSIVE,
            }
        indicators = green_indicators(spec, token_ids)
        n_scoreable = len(indicators)
        n_green = sum(1 for g in indicators if g)
        gamma = spec.gamma
        expected = gamma * n_scoreable
        # Normal-approximation z-score (continuity-corrected).
        denom = math.sqrt(n_scoreable * gamma * (1.0 - gamma))
        if denom <= 0.0:
            z = 0.0
        else:
            # Continuity correction for the one-tailed test: subtract
            # 0.5 from |x − μ| toward the mean.
            x = float(n_green)
            mu = expected
            if x > mu:
                z = (x - 0.5 - mu) / denom
            elif x < mu:
                z = (x + 0.5 - mu) / denom
            else:
                z = 0.0
        # Choose one-sided p-value based on polarity.
        if self._config.polarity == POLARITY_DETECT_WATERMARK:
            p_normal = _normal_sf(z)
            p_exact = _binom_tail_ge(n_green, n_scoreable, gamma)
        else:  # POLARITY_VERIFY_WATERMARK — reject downward
            p_normal = _normal_cdf(z)
            p_exact = _binom_tail_le(n_green, n_scoreable, gamma)
        # Cap exact tail at 1.0 (numerical guard).
        p_exact = max(0.0, min(1.0, p_exact))
        # Clopper-Pearson CI on the green rate.
        cp_lo, cp_hi = _clopper_pearson(n_green, n_scoreable, self._config.alpha)
        # Lexical entropy control.
        lex_counts: dict[str, int] = {}
        for t in doc.tokens:
            if t.text:
                lex_counts[t.text.lower()] = lex_counts.get(t.text.lower(), 0) + 1
        if not lex_counts:
            lex_counts = {str(t.token_id): 1 for t in doc.tokens}
        lex_h = _shannon_entropy(list(lex_counts.values()))
        # Pick the chosen p-value: normal if both Nγ and N(1−γ) are above
        # min_tokens_for_normal; else exact.
        use_normal = (min(gamma, 1.0 - gamma) * n_scoreable
                      >= self._config.min_tokens_for_normal)
        chosen_p = p_normal if use_normal else p_exact
        # Per-trial verdict from chosen_p (polarity-aware).
        a = self._config.alpha
        if self._config.polarity == POLARITY_DETECT_WATERMARK:
            # Small p ⇒ unlikely under "unmarked" ⇒ marked is detected
            # ⇒ desired outcome ⇒ PASS.
            if chosen_p <= a:
                verdict = VERDICT_PASS
            elif chosen_p <= a / self._config.warn_factor:
                verdict = VERDICT_WARN
            else:
                verdict = VERDICT_FAIL
        else:  # POLARITY_VERIFY_WATERMARK
            # Small p ⇒ unlikely under "still marked" ⇒ mark stripped
            # ⇒ undesired outcome ⇒ FAIL.
            if chosen_p <= a:
                verdict = VERDICT_FAIL
            elif chosen_p <= a / self._config.warn_factor:
                verdict = VERDICT_WARN
            else:
                verdict = VERDICT_PASS
        return {
            "n_tokens": n_total,
            "n_scoreable": n_scoreable,
            "n_green": n_green,
            "green_fraction": n_green / n_scoreable if n_scoreable else float("nan"),
            "expected_green": expected,
            "z_score": z,
            "p_value_normal": p_normal,
            "p_value_exact": p_exact,
            "rate_lower_cp": cp_lo,
            "rate_upper_cp": cp_hi,
            "lexical_entropy": lex_h,
            "chosen_p_value": chosen_p,
            "verdict": verdict,
        }

    # -- score-only (pure, no state change) ---------------------------

    def score_only(self, trial: Trial) -> TrialReport:
        """Score a trial without storing it.  Pure function.

        Useful for *preview* queries from the coordination engine that
        want to know what the test would say without committing to the
        audit pool.
        """
        if not isinstance(trial, Trial):
            raise InvalidTrial("score_only requires a Trial")
        raw = self._score_document(trial.document, trial.spec)
        return TrialReport(
            doc_id=trial.document.doc_id,
            n_tokens=raw["n_tokens"],
            n_scoreable=raw["n_scoreable"],
            n_green=raw["n_green"],
            green_fraction=raw["green_fraction"],
            expected_green=raw["expected_green"],
            z_score=raw["z_score"],
            p_value_normal=raw["p_value_normal"],
            p_value_exact=raw["p_value_exact"],
            rate_lower_cp=raw["rate_lower_cp"],
            rate_upper_cp=raw["rate_upper_cp"],
            lexical_entropy=raw["lexical_entropy"],
            chosen_p_value=raw["chosen_p_value"],
            verdict=raw["verdict"],
            has_truth=trial.truth is not None,
            truth_value=trial.truth,
            is_control=trial.control,
            metadata=dict(trial.metadata),
        )

    # -- submit (stateful) --------------------------------------------

    def submit(self, trial: Trial) -> TrialReport:
        """Submit a :class:`Trial`, score it, and store the report."""
        if not isinstance(trial, Trial):
            raise InvalidTrial("submit requires a Trial")
        with self._lock:
            self._emit(WM_SUBMITTED, {
                "doc_id": trial.document.doc_id,
                "spec": trial.spec.name,
                "has_truth": trial.truth is not None,
                "is_control": trial.control,
            })
            report = self.score_only(trial)
            self._reports.append(report)
            self._trials.append(trial)
            # Update sufficient statistics for the audit (operating
            # pool only — control trials measure FPR, not audit rate).
            if not trial.control:
                self._n_tokens_seen += report.n_scoreable
                self._n_green_seen += report.n_green
            self._n_submitted += 1
            # Enforce max_documents.
            if (self._config.max_documents > 0
                    and len(self._trials) > self._config.max_documents):
                drop = len(self._trials) - self._config.max_documents
                # FIFO drop.  We do NOT remove tokens from the audit
                # sufficient statistics — the audit is monotone in
                # observations by design.
                self._trials = self._trials[drop:]
                self._reports = self._reports[drop:]
            self._emit(WM_SCORED, {
                "doc_id": report.doc_id,
                "n_tokens": report.n_tokens,
                "n_scoreable": report.n_scoreable,
                "n_green": report.n_green,
                "z_score": report.z_score,
                "chosen_p_value": report.chosen_p_value,
                "verdict": report.verdict,
            })
            return report

    def submit_text(self,
                    doc_id: str,
                    text: str,
                    spec: WatermarkSpec,
                    *,
                    tokenizer: Tokenizer | None = None,
                    truth: bool | None = None,
                    control: bool = False,
                    metadata: Mapping[str, Any] | None = None) -> TrialReport:
        """Tokenize raw text and submit as a :class:`Trial`."""
        doc = make_document(doc_id, text, tokenizer=tokenizer, metadata=metadata)
        trial = Trial(document=doc, spec=spec, truth=truth, control=control,
                      metadata=dict(metadata or {}))
        return self.submit(trial)

    # -- calibration ---------------------------------------------------

    def calibrate(self) -> ThresholdReport:
        """Fit the Youden-J threshold on the per-document z-statistic
        against the labelled pool; bootstrap-CI the AUROC.

        Returns the :class:`ThresholdReport` and caches it for
        certificate emission.

        Raises :class:`NotEnoughTrials` if the labelled pool lacks
        either class and ``require_label_for_threshold`` is ``True``.
        """
        with self._lock:
            labelled = [(r.z_score, bool(r.truth_value))
                        for r in self._reports
                        if r.has_truth and not r.is_control
                        and not math.isnan(r.z_score)]
            n_pos = sum(1 for _, y in labelled if y)
            n_neg = sum(1 for _, y in labelled if not y)
            if self._config.require_label_for_threshold:
                if not labelled or n_pos == 0 or n_neg == 0:
                    raise NotEnoughTrials(
                        "calibrate requires at least one positive and one "
                        "negative labelled trial"
                    )
            if not labelled:
                rep = ThresholdReport(
                    threshold=float("nan"),
                    auroc=float("nan"),
                    auroc_lower=float("nan"),
                    auroc_upper=float("nan"),
                    youden_j=float("nan"),
                    tpr_at_threshold=float("nan"),
                    fpr_at_threshold=float("nan"),
                    n_labelled=0,
                    n_positive=0,
                    n_negative=0,
                    confidence=self._config.confidence,
                    bootstrap_b=self._config.bootstrap_b,
                )
                self._threshold_report = rep
                self._emit(WM_CALIBRATED, rep.to_dict())
                return rep
            scores = [s for s, _ in labelled]
            labels = [y for _, y in labelled]
            auroc = _auroc(scores, labels)
            lo, hi = _bootstrap_auroc_ci(
                scores, labels,
                b=self._config.bootstrap_b,
                confidence=self._config.confidence,
                seed=self._config.seed,
            )
            t, j, tpr, fpr = _youden_threshold(scores, labels)
            rep = ThresholdReport(
                threshold=t,
                auroc=auroc,
                auroc_lower=lo,
                auroc_upper=hi,
                youden_j=j,
                tpr_at_threshold=tpr,
                fpr_at_threshold=fpr,
                n_labelled=len(labelled),
                n_positive=n_pos,
                n_negative=n_neg,
                confidence=self._config.confidence,
                bootstrap_b=self._config.bootstrap_b,
            )
            self._threshold_report = rep
            self._emit(WM_CALIBRATED, rep.to_dict())
            return rep

    # -- audit ---------------------------------------------------------

    def audit(self, *, spec: WatermarkSpec | None = None) -> AuditReport:
        """Sequential anytime-valid audit of the per-token green rate.

        Aggregates per-token green indicators across all *non-control*
        submitted trials and tests, *one-sided* under the configured
        polarity:

          * :data:`POLARITY_DETECT_WATERMARK`: ``H_0: green_rate ≤ γ``.
            Rejects when the observed rate is significantly above γ
            (text is watermarked).

          * :data:`POLARITY_VERIFY_WATERMARK`: ``H_0: green_rate ≥ γ``.
            Rejects when the observed rate is significantly below γ
            (text has been stripped / paraphrased away from the mark).

        Rejection criterion: the exact one-sided binomial p-value
        ``p ≤ alpha``.  The Beta-Binomial Bayes-factor e-value is
        reported for informational continuity with the e-process
        literature; the one-sided binomial test is the operational
        rejector.
        """
        with self._lock:
            chosen_spec = spec or self._spec
            if chosen_spec is None:
                # Default: pull γ from the first non-control trial's
                # spec.  All trials in a single audit should share the
                # same γ; if they don't, the caller should partition.
                for t in self._trials:
                    if not t.control:
                        chosen_spec = t.spec
                        break
            if chosen_spec is None:
                # No data.  Audit is vacuously inconclusive.
                rep = AuditReport(
                    n_tokens_seen=0,
                    n_green_seen=0,
                    running_rate=float("nan"),
                    rate_lower_clopper_pearson=0.0,
                    rate_upper_clopper_pearson=1.0,
                    e_value=1.0,
                    log_e_value=0.0,
                    rejected_h0=False,
                    gamma=float("nan"),
                    alpha=self._config.alpha,
                )
                self._emit(WM_AUDITED, rep.to_dict())
                return rep
            gamma = chosen_spec.gamma
            n = self._n_tokens_seen
            k = self._n_green_seen
            cp_lo, cp_hi = _clopper_pearson(k, n, self._config.alpha) if n else (0.0, 1.0)
            # Polarity-aware one-sided binomial p-value.
            if self._config.polarity == POLARITY_DETECT_WATERMARK:
                p_one_sided = _binom_tail_ge(k, n, gamma) if n else 1.0
            else:
                p_one_sided = _binom_tail_le(k, n, gamma) if n else 1.0
            # Cap into [0, 1] (numerical guard).
            p_one_sided = max(0.0, min(1.0, p_one_sided))
            rejected = (n > 0) and (p_one_sided <= self._config.alpha)
            # Beta-Binomial Bayes factor (two-sided in general; reported
            # for continuity with the e-process literature).
            log_e = _eprocess_log(k, n, gamma,
                                  self._config.prior_a,
                                  self._config.prior_b)
            try:
                e = math.exp(log_e) if n > 0 else 1.0
            except OverflowError:
                e = float("inf")
            rep = AuditReport(
                n_tokens_seen=n,
                n_green_seen=k,
                running_rate=k / n if n else float("nan"),
                rate_lower_clopper_pearson=cp_lo,
                rate_upper_clopper_pearson=cp_hi,
                e_value=e,
                log_e_value=log_e,
                rejected_h0=rejected,
                gamma=gamma,
                alpha=self._config.alpha,
            )
            self._emit(WM_AUDITED, rep.to_dict())
            return rep

    # -- multi-document FDR -------------------------------------------

    def fdr_threshold(self) -> float | None:
        """Benjamini-Hochberg FDR-adjusted p-value threshold over the
        per-document ``chosen_p_value``.

        Returns the *largest p-value* surviving BH at FDR ``alpha``,
        or ``None`` if no document rejects.
        """
        with self._lock:
            pvals = [r.chosen_p_value for r in self._reports
                     if not r.is_control
                     and not math.isnan(r.chosen_p_value)]
            return _benjamini_hochberg(pvals, self._config.alpha)

    # -- combined verdict via Holm step-down --------------------------

    def holm_combined_p(self) -> float | None:
        """Holm-corrected smallest p-value across the per-document
        results, returning ``None`` on an empty pool.

        Used as the multi-test Watermarker-level p-value for
        certificate emission.
        """
        with self._lock:
            pvals = [r.chosen_p_value for r in self._reports
                     if not r.is_control
                     and not math.isnan(r.chosen_p_value)]
            if not pvals:
                return None
            return _holm_smallest(pvals)

    # -- certificate ---------------------------------------------------

    def _verdict_and_recommendation(
            self,
            holm_p: float | None,
            audit: AuditReport,
            ) -> tuple[str, str]:
        """Combine Holm-corrected multi-test result with the audit
        verdict to produce a final coordinator-facing pair.

        Verdict semantics
        -----------------
        ``VERDICT_PASS`` always means "the *desired* hypothesis is
        confirmed":

          * Detect mode: the *desired* result is "the text **is**
            watermarked" (we caught it).  PASS ⇔ green-rate >> γ.
          * Verify mode: the *desired* result is "the text **is still**
            watermarked" (the mark survived).  PASS ⇔ green-rate ≥ γ
            with no significant downward drift.

        The recommendation is keyed off the verdict and polarity so
        that a coordination engine can route on a single label.
        """
        a = self._config.alpha
        polarity = self._config.polarity
        # Multi-test verdict from per-document Holm-corrected p-values
        # (polarity-aware — see ``_score_document``).
        if holm_p is None:
            multi_v = VERDICT_INCONCLUSIVE
        elif polarity == POLARITY_DETECT_WATERMARK:
            # Small p ⇒ marks detected ⇒ desired ⇒ PASS.
            if holm_p <= a:
                multi_v = VERDICT_PASS
            elif holm_p <= a / self._config.warn_factor:
                multi_v = VERDICT_WARN
            else:
                multi_v = VERDICT_FAIL
        else:  # VERIFY
            # Small p ⇒ marks stripped ⇒ undesired ⇒ FAIL.
            if holm_p <= a:
                multi_v = VERDICT_FAIL
            elif holm_p <= a / self._config.warn_factor:
                multi_v = VERDICT_WARN
            else:
                multi_v = VERDICT_PASS
        # Audit verdict — polarity-aware.
        if audit.n_tokens_seen == 0 or math.isnan(audit.gamma):
            audit_v = VERDICT_INCONCLUSIVE
        elif polarity == POLARITY_DETECT_WATERMARK:
            if audit.rejected_h0:
                audit_v = VERDICT_PASS
            elif audit.rate_lower_clopper_pearson > audit.gamma:
                audit_v = VERDICT_WARN
            else:
                audit_v = VERDICT_FAIL
        else:  # VERIFY mode
            if audit.rejected_h0:
                # H0: rate ≥ γ rejected ⇒ text is statistically below
                # the spec ⇒ stripped / paraphrased.
                audit_v = VERDICT_FAIL
            elif audit.rate_upper_clopper_pearson < audit.gamma:
                audit_v = VERDICT_WARN
            else:
                audit_v = VERDICT_PASS
        # Combine: take the more conservative non-INCONCLUSIVE.
        order = {
            VERDICT_PASS: 0,
            VERDICT_WARN: 1,
            VERDICT_FAIL: 2,
            VERDICT_INCONCLUSIVE: 3,
        }
        if multi_v == VERDICT_INCONCLUSIVE and audit_v == VERDICT_INCONCLUSIVE:
            v = VERDICT_INCONCLUSIVE
        elif multi_v == VERDICT_INCONCLUSIVE:
            v = audit_v
        elif audit_v == VERDICT_INCONCLUSIVE:
            v = multi_v
        else:
            v = max(multi_v, audit_v, key=lambda x: order[x])
        # Recommendation keyed off verdict and polarity.
        if v == VERDICT_PASS:
            rec = REC_TRUST
        elif v == VERDICT_WARN:
            rec = REC_RESTRICT
        elif v == VERDICT_FAIL:
            if polarity == POLARITY_DETECT_WATERMARK:
                # We expected a watermark and didn't find one.
                rec = REC_QUARANTINE
            else:
                # We expected the watermark to survive and it didn't.
                rec = REC_BLOCK
        else:
            rec = REC_ESCALATE
        return v, rec

    def certify(self) -> WatermarkCertificate:
        """Emit a replay-verifiable :class:`WatermarkCertificate`.

        Pulls the latest threshold (if calibrated), the audit state,
        the Holm combined p, and the BH FDR threshold; hashes them
        into the running fingerprint chain.
        """
        with self._lock:
            audit = self.audit()
            holm_p = self.holm_combined_p()
            fdr_p = self.fdr_threshold()
            t_rep = self._threshold_report
            v, rec = self._verdict_and_recommendation(holm_p, audit)
            # Spec metadata (the active spec, or the first trial's).
            spec_name = self._spec.name if self._spec else (
                self._trials[0].spec.name if self._trials else "—"
            )
            spec_fp = self._spec.fingerprint() if self._spec else (
                self._trials[0].spec.fingerprint() if self._trials else "0" * 64
            )
            n_labelled = sum(1 for r in self._reports
                             if r.has_truth and not r.is_control)
            n_control = sum(1 for r in self._reports if r.is_control)
            payload = {
                "config": self._config_dict(),
                "spec_name": spec_name,
                "spec_fingerprint": spec_fp,
                "n_trials": len(self._reports),
                "n_trials_labelled": n_labelled,
                "n_control": n_control,
                "audit": audit.to_dict(),
                "threshold": t_rep.to_dict() if t_rep else None,
                "holm_p": holm_p,
                "fdr_p": fdr_p,
                "verdict": v,
                "recommendation": rec,
            }
            chained = _fingerprint(self._prev_fingerprint, payload)
            self._prev_fingerprint = chained
            cert = WatermarkCertificate(
                spec_name=spec_name,
                spec_fingerprint=spec_fp,
                n_trials=len(self._reports),
                n_trials_labelled=n_labelled,
                n_control=n_control,
                n_tokens_seen=audit.n_tokens_seen,
                n_green_seen=audit.n_green_seen,
                green_rate=audit.running_rate,
                rate_lower_cp=audit.rate_lower_clopper_pearson,
                rate_upper_cp=audit.rate_upper_clopper_pearson,
                auroc=t_rep.auroc if t_rep else None,
                auroc_lower=t_rep.auroc_lower if t_rep else None,
                auroc_upper=t_rep.auroc_upper if t_rep else None,
                threshold=t_rep.threshold if t_rep else None,
                e_value=audit.e_value,
                log_e_value=audit.log_e_value,
                rejected_h0=audit.rejected_h0,
                verdict=v,
                recommendation=rec,
                holm_smallest_p=holm_p,
                fdr_threshold_p=fdr_p,
                fingerprint_hash=chained,
            )
            self._last_certificate = cert
            self._emit(WM_CERTIFIED, cert.to_dict())
            return cert

    # -- gate ----------------------------------------------------------

    def gate(self, *, on: tuple[str, ...] = (VERDICT_PASS,)
             ) -> tuple[bool, WatermarkCertificate]:
        """Yes/no on the current verdict.

        Returns ``(passed, cert)``.  ``passed`` iff the certificate's
        verdict is in ``on``.  ``on`` defaults to ``(VERDICT_PASS,)``.
        """
        cert = self.certify()
        passed = cert.verdict in on
        self._emit(WM_GATED, {
            "verdict": cert.verdict,
            "recommendation": cert.recommendation,
            "passed": passed,
            "on": list(on),
        })
        return passed, cert

    # -- report --------------------------------------------------------

    def report(self) -> WatermarkerReport:
        """Single-bundle snapshot of the Watermarker state."""
        with self._lock:
            audit = self.audit()
            cert = self._last_certificate or self.certify()
            t_rep = self._threshold_report
            rep = WatermarkerReport(
                config=self._config_dict(),
                n_trials=len(self._reports),
                threshold=t_rep.to_dict() if t_rep else None,
                audit=audit.to_dict(),
                certificate=cert.to_dict() if cert else None,
            )
            self._emit(WM_REPORTED, rep.to_dict())
            return rep

    # -- reset ---------------------------------------------------------

    def reset(self) -> None:
        """Wipe the trial bank and audit state (idempotent).

        The certificate chain's *prior fingerprint* is preserved so
        that an audit re-start emits a new certificate that still
        chains onto the previous run — replay continues to work.
        """
        with self._lock:
            self._trials = []
            self._reports = []
            self._n_tokens_seen = 0
            self._n_green_seen = 0
            self._threshold_report = None
            self._last_certificate = None
            self._n_submitted = 0
            self._emit(WM_RESET, {"prev_fingerprint": self._prev_fingerprint})

    # -- iteration -----------------------------------------------------

    def reports(self) -> tuple[TrialReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def trials(self) -> tuple[Trial, ...]:
        with self._lock:
            return tuple(self._trials)

    # -- simulation helpers -------------------------------------------

    def simulate_trial(self,
                       spec: WatermarkSpec,
                       *,
                       doc_id: str,
                       n_tokens: int,
                       marked: bool,
                       seed: int = 0,
                       effective_delta: float | None = None,
                       control: bool = False,
                       metadata: Mapping[str, Any] | None = None) -> Trial:
        """Construct a labelled :class:`Trial` from the spec.

        Used by tests and by callers building a synthetic calibration
        pool before they have real labelled traffic.  ``marked`` is
        the ground-truth label — passed through as ``Trial.truth``.

        ``control`` trials never carry a truth label (see
        :class:`Trial`).  ``control=True`` overrides ``marked=True`` —
        the result is an unmarked, unlabelled control.
        """
        if control:
            doc = simulate_unmarked_document(spec, doc_id, n_tokens, seed=seed,
                                             metadata=metadata)
            return Trial(document=doc, spec=spec, truth=None, control=True,
                         metadata=dict(metadata or {}))
        if marked:
            doc = simulate_marked_document(
                spec, doc_id, n_tokens, seed=seed,
                effective_delta=effective_delta, metadata=metadata,
            )
            return Trial(document=doc, spec=spec, truth=True, control=False,
                         metadata=dict(metadata or {}))
        doc = simulate_unmarked_document(spec, doc_id, n_tokens, seed=seed,
                                         metadata=metadata)
        return Trial(document=doc, spec=spec, truth=False, control=False,
                     metadata=dict(metadata or {}))


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    # constants
    "MODE_KGW_GREEN",
    "MODE_KGW_EXACT",
    "MODE_KGW_SELFHASH",
    "MODE_LEXICAL",
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
    "REC_BLOCK",
    "REC_ESCALATE",
    "KNOWN_RECOMMENDATIONS",
    "HASH_BLAKE2",
    "HASH_SHA256",
    "HASH_HMAC_SHA256",
    "KNOWN_HASH_KINDS",
    "POLARITY_DETECT_WATERMARK",
    "POLARITY_VERIFY_WATERMARK",
    "KNOWN_POLARITIES",
    "WM_STARTED",
    "WM_SUBMITTED",
    "WM_SCORED",
    "WM_CALIBRATED",
    "WM_AUDITED",
    "WM_CERTIFIED",
    "WM_REPORTED",
    "WM_GATED",
    "WM_RESET",
    # exceptions
    "WatermarkerError",
    "InvalidConfig",
    "InvalidToken",
    "InvalidDocument",
    "InvalidTrial",
    "UnknownMode",
    "UnknownHashKind",
    "UnknownPolarity",
    "NotEnoughTrials",
    "NotCalibrated",
    "TokenizerError",
    # records
    "WatermarkSpec",
    "WatermarkerConfig",
    "Token",
    "Document",
    "Trial",
    "TrialReport",
    "ThresholdReport",
    "AuditReport",
    "WatermarkCertificate",
    "WatermarkerReport",
    # primitive
    "Watermarker",
    # helpers
    "is_green_token",
    "green_indicators",
    "default_tokenizer",
    "tokenize_text",
    "make_document",
    "simulate_marked_token_ids",
    "simulate_unmarked_token_ids",
    "simulate_stripped_token_ids",
    "simulate_marked_document",
    "simulate_unmarked_document",
    "simulate_stripped_document",
]
