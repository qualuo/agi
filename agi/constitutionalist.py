r"""Constitutionalist — Constitutional AI / RLAIF as a runtime primitive.

Constitutional AI (Bai et al. 2022, Anthropic) is the only published recipe
that has trained a frontier-deployed chatbot to be both helpful and harmless
*without* RLHF on human-written harmlessness labels.  It does so by giving
the model a short list of natural-language *principles* (the
"constitution") and running two loops:

1.  **SL-CAI** — a model critiques and revises its own outputs against the
    constitution.  A small SFT pass on the revised pairs locks the behaviour
    in.
2.  **RL-CAI** — a preference model trained on AI-labelled comparisons (RLAIF)
    drives a downstream RL loop.

This module is the *runtime-primitive* expression of that recipe.  It is the
deterministic critique-revise loop expressed as a coordination-engine
primitive: pure stdlib, replay-verifiable, with per-principle PAC certificates.
It can be invoked at *inference time* as a guardrail (refuse / revise / accept)
and at *training time* to mine preference pairs for downstream alignment
(emit ``(rejected, chosen)`` for the :mod:`agi.aligner` primitive).

What earns this primitive a slot in the catalog
-----------------------------------------------

* It is the **only** primitive that takes a *natural-language* policy (a list
  of principles) and produces a per-principle, anytime-valid PAC certificate
  of the *realised* violation rate.  Coordinators handing off to it can prove
  a population-level compliance bound to an external auditor.

* It is **deterministic** given a seed and an idempotent critic/reviser: same
  text in, same critique-and-revise trajectory out, same fingerprint out.

* It is **complementary** to the safety primitives:

  - :mod:`agi.schemer` detects sandbagging / strategic deception *across*
    interactions; ``Constitutionalist`` enforces policy *per* interaction.
  - :mod:`agi.aligner` trains preference models on collected
    ``(rejected, chosen)`` pairs; ``Constitutionalist`` mines those pairs
    from the critique-revise loop.
  - :mod:`agi.debater` runs multi-agent debate to produce a more honest
    verdict on the same text.  ``Constitutionalist`` exposes a debate-as-
    critic mode for this composition.
  - :mod:`agi.governance` enforces hard refuse / quota policy at the
    runtime boundary; ``Constitutionalist`` is the soft, scored gate that
    feeds governance evidence.

Coordination-engine view
------------------------

The coordination engine treats ``Constitutionalist`` exactly like
``Anticipator`` or ``Schemer``: it injects a ``Critic`` and ``Reviser``,
registers items, calls a method that returns a typed result with a
``fingerprint`` field, subscribes to the EventBus for streaming evidence,
and asks for a PAC certificate at any time.  Two coordination modes are
supported out of the box:

* **Inline gate** — every model output flows through ``judge``; outputs
  whose worst-principle score falls below ``violation_threshold`` are
  fed to ``revise`` (bounded iteration); the result of the gate is
  ``accept`` / ``revise(N)`` / ``refuse`` with a stable rationale.
* **Best-of-N** — generate ``N`` revisions in parallel under different
  reviser seeds, judge each, keep the best by aggregate score.  The
  primitive exposes the deterministic selection rule and the per-step
  receipts.

Mathematical and algorithmic roots
----------------------------------

* **Bai, Y. et al. (2022) — "Constitutional AI: Harmlessness from AI
  Feedback" (arXiv:2212.08073).**  Establishes the critique-revise-AI-
  feedback recipe.  This primitive is the runtime expression of the
  SL-CAI inner loop, with the RLAIF outer loop delegated to
  :mod:`agi.aligner`.

* **Wang, T. et al. (2023) — "Aligning Large Language Models with Human:
  A Survey" / OpenAI RLHF, Anthropic CAI.**  Critique-revise as a
  generic alignment kernel.

* **Wilson, E.B. (1927) and Hoeffding, W. (1963) and Maurer & Pontil (2009).**
  Per-principle violation-rate confidence intervals (Wilson 2-sided,
  Hoeffding 1-sided LCB, empirical-Bernstein LCB).

* **Vovk, V. & Wang, R. (2021) — "E-values: Calibration, combination, and
  applications."**  Multi-principle joint confidence via Holm step-down
  on per-principle Bernoulli p-values (FWER controlled at ``alpha``).

* **Merkle, R. (1979).**  SHA-256 chain over every event yields the
  replay-verifiable receipt.

Composes with
-------------

* :mod:`agi.aligner`         — consume mined ``(rejected, chosen)`` pairs.
* :mod:`agi.schemer`         — cross-interaction deception detection.
* :mod:`agi.debater`         — debate-as-critic.
* :mod:`agi.governance`      — hard refuse / quota policy at the boundary.
* :mod:`agi.attest`          — append the certificate to the audit ledger.
* :mod:`agi.deliberator`     — escalate ambiguous critiques.
* :mod:`agi.attributor`      — per-principle data attribution on adapted
  models trained on the mined pairs.
* :mod:`agi.coordinator`     — the typical caller; ``judge`` and ``revise``
  are the hot paths.

What this primitive ships
-------------------------

* :class:`Principle`            — one natural-language constitutional rule.
* :class:`Constitution`         — ordered, hashed collection of principles.
* :class:`PrincipleScore`       — one ``(score, violated, rationale)``.
* :class:`Critique`             — full per-principle scoring of one text.
* :class:`RevisionStep`         — one step in the critique-revise loop.
* :class:`Revision`             — trajectory of revision steps.
* :class:`Verdict`              — accept / revise / refuse decision.
* :class:`ConstitutionalistConfig` — thresholds, aggregator, iteration budget.
* :class:`ConstitutionalistCertificate` — per-principle PAC bounds + fingerprint.
* :class:`ConstitutionalistReport`     — end-of-window audit.
* :class:`Constitutionalist`    — the primitive.

Pure stdlib.  No NumPy.  Deterministic given seed.  Thread-safe.
``json.dumps(report.to_dict())`` round-trips.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus

__all__ = [
    # Errors
    "ConstitutionalistError",
    "InvalidConfig",
    "InvalidConstitution",
    "InvalidCritique",
    "InvalidRevision",
    "UnknownItem",
    # Constants — verdict actions
    "ACTION_ACCEPT",
    "ACTION_REVISE",
    "ACTION_REFUSE",
    "KNOWN_ACTIONS",
    # Constants — severity
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "SEVERITY_VIOLATION",
    "SEVERITY_CRITICAL",
    "KNOWN_SEVERITIES",
    # Constants — aggregator
    "AGG_WORST",
    "AGG_WEIGHTED_MEAN",
    "AGG_WEIGHTED_GEOMETRIC",
    "AGG_SOFT_MIN",
    "KNOWN_AGGREGATORS",
    # Constants — stop rules
    "STOP_THRESHOLD",
    "STOP_NONINCREASING",
    "STOP_MAX_ITER",
    "KNOWN_STOP_RULES",
    # Events
    "CONSTITUTIONALIST_STARTED",
    "CONSTITUTIONALIST_REGISTERED",
    "CONSTITUTIONALIST_JUDGED",
    "CONSTITUTIONALIST_REVISED",
    "CONSTITUTIONALIST_ACCEPTED",
    "CONSTITUTIONALIST_REFUSED",
    "CONSTITUTIONALIST_BESTOF",
    "CONSTITUTIONALIST_CERTIFIED",
    "CONSTITUTIONALIST_REPORTED",
    "CONSTITUTIONALIST_RESET",
    # Records
    "Principle",
    "Constitution",
    "PrincipleScore",
    "Critique",
    "RevisionStep",
    "Revision",
    "Verdict",
    "ConstitutionalistConfig",
    "ConstitutionalistCertificate",
    "PrincipleCertificate",
    "ConstitutionalistReport",
    # Main
    "Constitutionalist",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConstitutionalistError(ValueError):
    """Base class for Constitutionalist errors."""


class InvalidConfig(ConstitutionalistError):
    """The :class:`ConstitutionalistConfig` is internally inconsistent."""


class InvalidConstitution(ConstitutionalistError):
    """The constitution is empty, malformed, or has duplicate principle ids."""


class InvalidCritique(ConstitutionalistError):
    """A critic returned a malformed critique."""


class InvalidRevision(ConstitutionalistError):
    """A reviser returned a malformed revision."""


class UnknownItem(ConstitutionalistError):
    """An item id was not registered."""


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

ACTION_ACCEPT = "accept"
ACTION_REVISE = "revise"
ACTION_REFUSE = "refuse"

KNOWN_ACTIONS: tuple[str, ...] = (ACTION_ACCEPT, ACTION_REVISE, ACTION_REFUSE)

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_VIOLATION = "violation"
SEVERITY_CRITICAL = "critical"

KNOWN_SEVERITIES: tuple[str, ...] = (
    SEVERITY_INFO, SEVERITY_WARN, SEVERITY_VIOLATION, SEVERITY_CRITICAL,
)

# Per-severity numerical ordering used by the aggregator and gate rules.
# Higher means stricter.
_SEVERITY_ORDER: dict[str, int] = {
    SEVERITY_INFO: 0,
    SEVERITY_WARN: 1,
    SEVERITY_VIOLATION: 2,
    SEVERITY_CRITICAL: 3,
}

AGG_WORST = "worst"
AGG_WEIGHTED_MEAN = "weighted-mean"
AGG_WEIGHTED_GEOMETRIC = "weighted-geometric"
AGG_SOFT_MIN = "soft-min"

KNOWN_AGGREGATORS: tuple[str, ...] = (
    AGG_WORST, AGG_WEIGHTED_MEAN, AGG_WEIGHTED_GEOMETRIC, AGG_SOFT_MIN,
)

STOP_THRESHOLD = "threshold"
STOP_NONINCREASING = "non-increasing"
STOP_MAX_ITER = "max-iter"

KNOWN_STOP_RULES: tuple[str, ...] = (
    STOP_THRESHOLD, STOP_NONINCREASING, STOP_MAX_ITER,
)


# Event names emitted onto the runtime EventBus.
CONSTITUTIONALIST_STARTED = "constitutionalist.started"
CONSTITUTIONALIST_REGISTERED = "constitutionalist.registered"
CONSTITUTIONALIST_JUDGED = "constitutionalist.judged"
CONSTITUTIONALIST_REVISED = "constitutionalist.revised"
CONSTITUTIONALIST_ACCEPTED = "constitutionalist.accepted"
CONSTITUTIONALIST_REFUSED = "constitutionalist.refused"
CONSTITUTIONALIST_BESTOF = "constitutionalist.bestof"
CONSTITUTIONALIST_CERTIFIED = "constitutionalist.certified"
CONSTITUTIONALIST_REPORTED = "constitutionalist.reported"
CONSTITUTIONALIST_RESET = "constitutionalist.reset"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principle:
    """One natural-language constitutional rule.

    Attributes:
        principle_id: stable opaque identifier.
        statement: natural-language statement of the principle.  Passed to
            the critic verbatim; callers should normalise (strip, casefold
            if desired) before constructing.
        severity: one of :data:`KNOWN_SEVERITIES`.  Determines the gate
            rule: a ``SEVERITY_CRITICAL`` failure overrides the aggregate
            and forces ``ACTION_REFUSE``.
        weight: per-principle weight (default ``1.0``).  Multiplies the
            principle's contribution to the aggregate score.  A weight of
            zero leaves the principle scored but excluded from the
            aggregate; useful for "informational" principles you want to
            log without gating on.
        threshold: minimum acceptable per-principle score in ``[0, 1]``.
            Defaults to ``0.5``.  Used by the iteration stop rule.
        metadata: opaque dict carried through fingerprinting.
    """

    principle_id: str
    statement: str
    severity: str = SEVERITY_VIOLATION
    weight: float = 1.0
    threshold: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.principle_id, str) or not self.principle_id:
            raise InvalidConstitution(
                "principle_id must be a non-empty string"
            )
        if not isinstance(self.statement, str) or not self.statement:
            raise InvalidConstitution("statement must be a non-empty string")
        if self.severity not in KNOWN_SEVERITIES:
            raise InvalidConstitution(
                f"severity must be one of {KNOWN_SEVERITIES!r}"
            )
        if (not isinstance(self.weight, (int, float))
                or not math.isfinite(float(self.weight))
                or float(self.weight) < 0):
            raise InvalidConstitution(
                "weight must be a non-negative finite number"
            )
        if (not isinstance(self.threshold, (int, float))
                or not math.isfinite(float(self.threshold))
                or not 0.0 <= float(self.threshold) <= 1.0):
            raise InvalidConstitution(
                "threshold must be a finite number in [0, 1]"
            )

    @property
    def is_critical(self) -> bool:
        return self.severity == SEVERITY_CRITICAL


@dataclass(frozen=True)
class Constitution:
    """Ordered list of principles, hashed for fingerprinting.

    Identity is structural: two constitutions with the same principle ids,
    statements, weights, thresholds and severities have the same
    :attr:`constitution_hash`.
    """

    principles: tuple[Principle, ...]
    name: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidConstitution("constitution name must be non-empty")
        if not isinstance(self.principles, tuple):
            object.__setattr__(self, "principles", tuple(self.principles))
        if not self.principles:
            raise InvalidConstitution("constitution must have >= 1 principle")
        seen: set[str] = set()
        for p in self.principles:
            if not isinstance(p, Principle):
                raise InvalidConstitution(
                    f"expected Principle, got {type(p).__name__}"
                )
            if p.principle_id in seen:
                raise InvalidConstitution(
                    f"duplicate principle_id: {p.principle_id!r}"
                )
            seen.add(p.principle_id)

    @classmethod
    def from_dicts(cls,
                   rows: Sequence[Mapping[str, Any]],
                   *,
                   name: str = "default") -> "Constitution":
        ps = tuple(
            Principle(
                principle_id=str(r["principle_id"]),
                statement=str(r["statement"]),
                severity=str(r.get("severity", SEVERITY_VIOLATION)),
                weight=float(r.get("weight", 1.0)),
                threshold=float(r.get("threshold", 0.5)),
                metadata=dict(r.get("metadata", {})),
            )
            for r in rows
        )
        return cls(principles=ps, name=name)

    @property
    def constitution_hash(self) -> str:
        return _hash_value({
            "name": self.name,
            "principles": [
                {
                    "id": p.principle_id,
                    "statement": p.statement,
                    "severity": p.severity,
                    "weight": float(p.weight),
                    "threshold": float(p.threshold),
                }
                for p in self.principles
            ],
        })

    def get(self, principle_id: str) -> Principle:
        for p in self.principles:
            if p.principle_id == principle_id:
                return p
        raise InvalidConstitution(
            f"no such principle_id: {principle_id!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "constitution_hash": self.constitution_hash,
            "principles": [
                {
                    "principle_id": p.principle_id,
                    "statement": p.statement,
                    "severity": p.severity,
                    "weight": float(p.weight),
                    "threshold": float(p.threshold),
                    "metadata": dict(p.metadata),
                }
                for p in self.principles
            ],
        }


@dataclass(frozen=True)
class PrincipleScore:
    """One ``(principle, score, violated, rationale)`` cell.

    Attributes:
        principle_id: which principle this score is about.
        score: in ``[0, 1]``.  Higher is "more compliant".
        violated: convenience flag ``score < principle.threshold``.  Set
            by :class:`Constitutionalist`; critics need not populate it.
        rationale: optional natural-language rationale from the critic.
    """

    principle_id: str
    score: float
    violated: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        if (not isinstance(self.score, (int, float))
                or not math.isfinite(float(self.score))
                or not 0.0 <= float(self.score) <= 1.0):
            raise InvalidCritique(
                f"score must be a finite number in [0, 1]; got {self.score!r}"
            )
        if not isinstance(self.rationale, str):
            raise InvalidCritique("rationale must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle_id": self.principle_id,
            "score": float(self.score),
            "violated": bool(self.violated),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class Critique:
    """Per-principle scoring of one text.

    Attributes:
        item_id: registered item identifier.
        text: the text that was scored.
        scores: tuple of per-principle scores, in constitution order.
        aggregate_score: pre-computed aggregate score under the configured
            aggregator.
        worst_principle: the principle id with the lowest score (ties
            broken lexicographically for determinism).
        worst_score: the lowest per-principle score.
        violations: tuple of principle ids whose score fell below the
            principle's threshold.
        critical_violations: subset of ``violations`` of severity CRITICAL.
        text_hash: SHA-256 of the canonical text.
        constitution_hash: SHA-256 of the active constitution.
        seed: the local seed used to invoke the critic.
        ts: clock time the critique was produced.
    """

    item_id: str
    text: str
    scores: tuple[PrincipleScore, ...]
    aggregate_score: float
    worst_principle: str
    worst_score: float
    violations: tuple[str, ...]
    critical_violations: tuple[str, ...]
    text_hash: str
    constitution_hash: str
    seed: int
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text_hash": self.text_hash,
            "constitution_hash": self.constitution_hash,
            "aggregate_score": float(self.aggregate_score),
            "worst_principle": self.worst_principle,
            "worst_score": float(self.worst_score),
            "violations": list(self.violations),
            "critical_violations": list(self.critical_violations),
            "scores": [s.to_dict() for s in self.scores],
            "seed": int(self.seed),
            "ts": float(self.ts),
        }


@dataclass(frozen=True)
class RevisionStep:
    """One step in the critique-revise loop.

    Attributes:
        step: 0-indexed iteration number.
        input_text: the text input to this step.
        critique: critique of ``input_text`` (the *pre*-revision critique).
        revised_text: revised text emitted by the reviser.  Empty if the
            stop rule fired before this step revised anything.
        improved: True iff aggregate_score went up vs. previous step.
        elapsed_ms: wall-clock time spent on this step.
        seed: local seed used.
    """

    step: int
    input_text: str
    critique: Critique
    revised_text: str
    improved: bool
    elapsed_ms: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "input_hash": _hash_text(self.input_text),
            "revised_hash": _hash_text(self.revised_text),
            "revised_text": self.revised_text,
            "critique": self.critique.to_dict(),
            "improved": bool(self.improved),
            "elapsed_ms": float(self.elapsed_ms),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class Revision:
    """Trajectory of a critique-revise loop on one item.

    Attributes:
        item_id: registered item identifier.
        original_text: input text the loop was entered on.
        final_text: the text the loop exited on.  Equals
            ``original_text`` iff no revision was emitted.
        steps: ordered tuple of :class:`RevisionStep`.
        final_critique: critique of ``final_text``.  Always present, even
            if no revision happened (then equals ``steps[0].critique``).
        converged: True iff the loop stopped because the stop rule was
            satisfied (i.e. all principles met threshold) rather than
            hitting the iteration budget.
        stop_reason: one of :data:`KNOWN_STOP_RULES`.
        total_elapsed_ms: sum of step elapsed times.
    """

    item_id: str
    original_text: str
    final_text: str
    steps: tuple[RevisionStep, ...]
    final_critique: Critique
    converged: bool
    stop_reason: str
    total_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "original_hash": _hash_text(self.original_text),
            "final_hash": _hash_text(self.final_text),
            "final_text": self.final_text,
            "steps": [s.to_dict() for s in self.steps],
            "final_critique": self.final_critique.to_dict(),
            "converged": bool(self.converged),
            "stop_reason": self.stop_reason,
            "total_elapsed_ms": float(self.total_elapsed_ms),
        }


@dataclass(frozen=True)
class Verdict:
    """Action-bearing gate decision.

    Attributes:
        item_id: registered item identifier.
        action: one of :data:`KNOWN_ACTIONS`.
        text: the text to use downstream — original (accept), revised
            (revise), or empty (refuse).
        revision: the underlying revision trajectory.  ``None`` only when
            the gate refused without entering the loop (e.g. critical
            violation on first critique with ``refuse_on_critical=True``).
        rationale: short, deterministic, machine-readable summary of why
            the verdict was produced.
        fingerprint: SHA-256 of the Merkle chain up to this verdict.
    """

    item_id: str
    action: str
    text: str
    revision: Revision | None
    rationale: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "action": self.action,
            "text_hash": _hash_text(self.text),
            "text": self.text,
            "revision": self.revision.to_dict() if self.revision else None,
            "rationale": self.rationale,
            "fingerprint": self.fingerprint,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstitutionalistConfig:
    """Configuration for a :class:`Constitutionalist` instance.

    Gate
    ----

    Attributes:
        violation_threshold: outputs whose aggregate score falls below
            this trigger the critique-revise loop.  In ``[0, 1]``.
            Default ``0.5``.
        accept_threshold: outputs whose aggregate score meets or exceeds
            this are accepted without revision.  In ``[0, 1]``.  Default
            ``0.85``.  ``accept_threshold >= violation_threshold`` is
            enforced; values in between map to ``ACTION_REVISE``.
        refuse_on_critical: if True, any non-zero count of CRITICAL
            severity violations forces ``ACTION_REFUSE`` regardless of
            aggregate score.  Default True.
        refuse_after_iters: if the loop has spent ``max_iters`` and the
            aggregate is still below ``violation_threshold``, refuse.
            Default True.

    Iteration
    ---------

    Attributes:
        max_iters: hard upper bound on critique-revise iterations per
            item.  Default 3 (matches the Bai et al. 2022 schedule).
        require_strict_improvement: if True, stop the loop the moment a
            revision step does not increase aggregate_score.  Default
            True (prevents diverging revisers from making things worse).

    Aggregation
    -----------

    Attributes:
        aggregator: one of :data:`KNOWN_AGGREGATORS`.
        soft_min_temperature: temperature for :data:`AGG_SOFT_MIN` (a
            negative log-sum-exp of negative scores).  Lower → harder
            min.  Default ``0.05``.

    Certificate
    -----------

    Attributes:
        alpha: confidence level for per-principle violation-rate intervals
            (default ``0.05`` ⇒ 95% CIs).
        min_items_for_certificate: refuse to certify until at least this
            many items have been judged.
        joint_correction: if True, the joint per-principle Wilson upper
            confidence bound on violation rate is Holm-corrected for the
            number of principles tested.  Default True.

    Reproducibility
    ---------------

    Attributes:
        seed: master RNG seed.  Per-item local seeds are derived as
            ``sha256(seed | item_id | constitution_hash)``.
    """

    violation_threshold: float = 0.5
    accept_threshold: float = 0.85
    refuse_on_critical: bool = True
    refuse_after_iters: bool = True

    max_iters: int = 3
    require_strict_improvement: bool = True

    aggregator: str = AGG_WEIGHTED_GEOMETRIC
    soft_min_temperature: float = 0.05

    alpha: float = 0.05
    min_items_for_certificate: int = 8
    joint_correction: bool = True

    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.violation_threshold) <= 1.0:
            raise InvalidConfig("violation_threshold must lie in [0, 1]")
        if not 0.0 <= float(self.accept_threshold) <= 1.0:
            raise InvalidConfig("accept_threshold must lie in [0, 1]")
        if float(self.accept_threshold) < float(self.violation_threshold):
            raise InvalidConfig(
                "accept_threshold must be >= violation_threshold"
            )
        if (not isinstance(self.max_iters, int)) or self.max_iters < 0:
            raise InvalidConfig("max_iters must be a non-negative int")
        if self.aggregator not in KNOWN_AGGREGATORS:
            raise InvalidConfig(
                f"aggregator must be one of {KNOWN_AGGREGATORS!r}"
            )
        if (not isinstance(self.soft_min_temperature, (int, float))
                or not math.isfinite(float(self.soft_min_temperature))
                or float(self.soft_min_temperature) <= 0):
            raise InvalidConfig("soft_min_temperature must be > 0")
        if not 0.0 < float(self.alpha) < 1.0:
            raise InvalidConfig("alpha must lie in (0, 1)")
        if (not isinstance(self.min_items_for_certificate, int)
                or self.min_items_for_certificate < 1):
            raise InvalidConfig(
                "min_items_for_certificate must be >= 1"
            )


@dataclass(frozen=True)
class PrincipleCertificate:
    """Per-principle realised-violation certificate.

    Attributes:
        principle_id: the principle.
        n_items: total items the principle was applied to.
        n_violations: items whose ``score < threshold``.
        violation_rate: ``n_violations / n_items``.
        wilson_lo, wilson_hi: two-sided Wilson interval on the rate.
        hoeffding_hi: one-sided Hoeffding *upper* confidence bound — the
            worst-case violation rate at confidence ``1 - alpha``.
        score_mean: mean per-principle score across items.
        score_eb_lo: Maurer-Pontil empirical-Bernstein *lower* confidence
            bound on the mean per-principle score.
        adjusted_alpha: per-principle alpha after Holm correction (==
            alpha when ``joint_correction = False``).
    """

    principle_id: str
    n_items: int
    n_violations: int
    violation_rate: float
    wilson_lo: float
    wilson_hi: float
    hoeffding_hi: float
    score_mean: float
    score_eb_lo: float
    adjusted_alpha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstitutionalistCertificate:
    """Replay-verifiable, anytime-valid certificate of compliance.

    Attributes:
        n_items: number of items judged.
        constitution_hash: hash of the active constitution.
        principles: per-principle certificates, in constitution order.
        worst_violation_rate: max over principles.
        worst_principle: principle id that attained ``worst_violation_rate``.
        aggregate_mean: mean aggregate score across items.
        aggregate_eb_lo: empirical-Bernstein LCB on the mean aggregate.
        accept_count: items the gate accepted.
        revise_count: items the gate sent through revision.
        refuse_count: items the gate refused.
        critical_count: items with at least one CRITICAL violation.
        alpha: configured confidence level.
        joint_correction: whether Holm correction was applied.
        fingerprint: SHA-256 of the Merkle chain.
    """

    n_items: int
    constitution_hash: str
    principles: tuple[PrincipleCertificate, ...]
    worst_violation_rate: float
    worst_principle: str
    aggregate_mean: float
    aggregate_eb_lo: float
    accept_count: int
    revise_count: int
    refuse_count: int
    critical_count: int
    alpha: float
    joint_correction: bool
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["principles"] = [p.to_dict() for p in self.principles]
        return d


@dataclass(frozen=True)
class ConstitutionalistReport:
    """End-of-window audit of the primitive."""

    items: int
    revisions_run: int
    certificate: ConstitutionalistCertificate

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["certificate"] = self.certificate.to_dict()
        return d


# ---------------------------------------------------------------------------
# Hashing / canonicalisation helpers (shared idiom across primitives)
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.12g}")
    if isinstance(value, bool):
        return bool(value)
    return value


def _hash_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True).encode()
    ).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable(value: Any) -> Any:
    """Strip non-determinism for the Merkle chain (drop volatile keys)."""
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value.keys()) if k != "ts"}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.12g}")
    if isinstance(value, bool):
        return bool(value)
    return value


def _local_seed(master_seed: int, *parts: str) -> int:
    h = hashlib.sha256()
    h.update(str(master_seed).encode())
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8"))
    return int(h.hexdigest(), 16) % (2 ** 31)


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def _aggregate(scores: Sequence[float],
               weights: Sequence[float],
               method: str,
               *,
               temperature: float = 0.05) -> float:
    """Combine per-principle scores into a single aggregate in ``[0, 1]``.

    Principles with zero weight are excluded from the aggregate (they are
    still surfaced as their own per-principle records).
    """
    pairs = [(float(s), float(w)) for s, w in zip(scores, weights) if w > 0]
    if not pairs:
        return 1.0
    if method == AGG_WORST:
        return min(s for s, _ in pairs)
    if method == AGG_WEIGHTED_MEAN:
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            return 1.0
        return sum(s * w for s, w in pairs) / total_w
    if method == AGG_WEIGHTED_GEOMETRIC:
        # Bai-style soft minimum: weighted geometric mean.  Floors at 0.
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            return 1.0
        log_acc = 0.0
        for s, w in pairs:
            # Clamp to a tiny floor so log is finite; zero on any principle
            # collapses the geometric mean toward zero, which is the desired
            # "worst principle dominates" behaviour without setting it to
            # exactly zero (which would force the loop into NaN territory).
            log_acc += w * math.log(max(s, 1e-12))
        return math.exp(log_acc / total_w)
    if method == AGG_SOFT_MIN:
        # -T * log( sum_i w_i * exp(-s_i / T) / sum_i w_i ).  As T -> 0,
        # approaches min(scores).  As T -> inf, approaches weighted mean.
        T = float(temperature)
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            return 1.0
        m = min(s for s, _ in pairs)
        # Subtract m for numerical stability.
        acc = sum(w * math.exp(-(s - m) / T) for s, w in pairs)
        # acc / total_w in (0, 1] of e^(-(s-m)/T) terms.
        # soft_min = m - T * log( acc / total_w ) ;  log(acc / total_w) <= 0
        sm = m + T * math.log(total_w) - T * math.log(max(acc, 1e-300))
        # Numerical clamp into [0, 1].
        return max(0.0, min(1.0, sm))
    raise InvalidConfig(f"unknown aggregator: {method!r}")


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def _z_two_sided(alpha: float) -> float:
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        p = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _wilson_ci(k: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = _z_two_sided(alpha)
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hoeffding_ucb_rate(k: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    eps = math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return min(1.0, k / n + eps)


def _empirical_bernstein_lcb(samples: Sequence[float],
                             alpha: float,
                             bound_b: float) -> float:
    n = len(samples)
    if n <= 1:
        return 0.0
    mean = sum(samples) / n
    var = sum((float(x) - mean) ** 2 for x in samples) / (n - 1)
    log_term = math.log(2.0 / alpha)
    bern_term = math.sqrt(2.0 * var * log_term / n)
    hoeff_term = 7.0 * float(bound_b) * log_term / (3.0 * (n - 1))
    return max(0.0, mean - bern_term - hoeff_term)


def _holm_alphas(alpha: float, k: int) -> list[float]:
    """Holm step-down alphas: i-th smallest p-value compared against
    ``alpha / (k - i + 1)``.  Returns the per-rank alpha thresholds in
    descending rank order (most permissive first)."""
    if k <= 0:
        return []
    return [alpha / max(1, k - i) for i in range(k)]


# ---------------------------------------------------------------------------
# Critic / Reviser protocols (structural)
# ---------------------------------------------------------------------------

# A ``Critic`` is any callable
#     critic(text: str, constitution: Constitution, *, rng: random.Random)
#         -> Iterable[PrincipleScore | dict | tuple]
# returning per-principle scores in any order.  Missing principles get a
# score of 1.0 (compliant by default) with an empty rationale.  Extra
# principle ids are ignored.
Critic = Callable[..., Iterable[Any]]

# A ``Reviser`` is any callable
#     reviser(text: str, critique: Critique, *, rng: random.Random) -> str
# returning a revised text.  May return the same text if it cannot improve.
Reviser = Callable[..., str]


def _coerce_scores(raw: Iterable[Any],
                   constitution: Constitution) -> dict[str, PrincipleScore]:
    """Coerce a critic's output into a ``{principle_id: PrincipleScore}`` map.

    Accepts:
        * an iterable of :class:`PrincipleScore`,
        * an iterable of dicts with keys ``principle_id``, ``score``,
          optional ``rationale``,
        * an iterable of ``(principle_id, score)`` or
          ``(principle_id, score, rationale)`` tuples.
    """
    out: dict[str, PrincipleScore] = {}
    valid_ids = {p.principle_id for p in constitution.principles}
    for item in raw:
        if isinstance(item, PrincipleScore):
            ps = item
        elif isinstance(item, Mapping):
            pid = str(item.get("principle_id", ""))
            if not pid:
                raise InvalidCritique(
                    "critic dict must include principle_id"
                )
            score = item.get("score")
            ps = PrincipleScore(
                principle_id=pid,
                score=float(score) if score is not None else 1.0,
                rationale=str(item.get("rationale", "")),
            )
        elif isinstance(item, (list, tuple)) and len(item) in (2, 3):
            pid = str(item[0])
            score = float(item[1])
            rationale = str(item[2]) if len(item) == 3 else ""
            ps = PrincipleScore(
                principle_id=pid, score=score, rationale=rationale,
            )
        else:
            raise InvalidCritique(
                f"critic returned unrecognised item: {type(item).__name__}"
            )
        if ps.principle_id not in valid_ids:
            # Silently ignore unknown principles — keeps critics forwards-
            # compatible with shrinking constitutions.
            continue
        if ps.principle_id in out:
            # Last write wins for an idempotent feel.
            pass
        out[ps.principle_id] = ps
    return out


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


class Constitutionalist:
    """Runtime-primitive entry point for Constitutional AI / RLAIF.

    Lifecycle::

        con = Constitutionalist(
            ConstitutionalistConfig(...),
            constitution=Constitution(principles=(...)),
        )
        verdict = con.gate("turn:42", text="...", critic=critic, reviser=reviser)
        cert = con.certificate()
        report = con.report()
    """

    def __init__(self,
                 config: ConstitutionalistConfig | None = None,
                 *,
                 constitution: Constitution,
                 bus: EventBus | None = None,
                 instance_id: str | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        if not isinstance(constitution, Constitution):
            raise InvalidConstitution(
                f"constitution must be a Constitution, got {type(constitution).__name__}"
            )
        self.config = config or ConstitutionalistConfig()
        self.constitution = constitution
        self.bus = bus
        self.instance_id = instance_id or ""
        self._clock = clock or time.time
        self._lock = threading.RLock()

        # Item bookkeeping.
        self._items: dict[str, str] = {}  # item_id -> original text
        self._critiques: list[Critique] = []
        self._revisions: list[Revision] = []
        self._verdicts: list[Verdict] = []

        # Per-principle running stats.
        # principle_id -> list[float] of scores; the matching violation count
        # is recomputed lazily from these.
        self._scores: dict[str, list[float]] = {
            p.principle_id: [] for p in constitution.principles
        }
        self._aggregates: list[float] = []

        # Gate counters.
        self._accept = 0
        self._revise = 0
        self._refuse = 0
        self._critical = 0

        # Merkle fingerprint.
        self._fingerprint = hashlib.sha256()
        self._fingerprint.update(json.dumps(
            {
                "version": 1,
                "instance_id": self.instance_id,
                "config": _stable({
                    "violation_threshold": self.config.violation_threshold,
                    "accept_threshold": self.config.accept_threshold,
                    "refuse_on_critical": self.config.refuse_on_critical,
                    "refuse_after_iters": self.config.refuse_after_iters,
                    "max_iters": self.config.max_iters,
                    "require_strict_improvement":
                        self.config.require_strict_improvement,
                    "aggregator": self.config.aggregator,
                    "soft_min_temperature": self.config.soft_min_temperature,
                    "alpha": self.config.alpha,
                    "min_items_for_certificate":
                        self.config.min_items_for_certificate,
                    "joint_correction": self.config.joint_correction,
                    "seed": self.config.seed,
                }),
                "constitution_hash": self.constitution.constitution_hash,
            },
            sort_keys=True,
        ).encode())

        self._publish(CONSTITUTIONALIST_STARTED, {
            "instance_id": self.instance_id,
            "constitution_hash": self.constitution.constitution_hash,
            "n_principles": len(constitution.principles),
            "aggregator": self.config.aggregator,
        })

    # ----- event helpers -----
    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        payload = {**data, "ts": self._clock()}
        self._fingerprint.update(json.dumps(
            {"kind": kind, "data": _stable(payload)}, sort_keys=True,
        ).encode())
        if self.bus is not None:
            self.bus.publish(Event(kind=kind, data=payload))

    @property
    def fingerprint_hash(self) -> str:
        return self._fingerprint.hexdigest()

    @property
    def critiques(self) -> tuple[Critique, ...]:
        with self._lock:
            return tuple(self._critiques)

    @property
    def revisions(self) -> tuple[Revision, ...]:
        with self._lock:
            return tuple(self._revisions)

    @property
    def verdicts(self) -> tuple[Verdict, ...]:
        with self._lock:
            return tuple(self._verdicts)

    # ----- registration -----
    def register_item(self, item_id: str, text: str) -> str:
        """Register an item.  Idempotent on ``item_id``; later calls
        overwrite the recorded ``text``.  Returns the item_id.
        """
        if not isinstance(item_id, str) or not item_id:
            raise ConstitutionalistError("item_id must be a non-empty string")
        if not isinstance(text, str):
            raise ConstitutionalistError("text must be a string")
        with self._lock:
            self._items[item_id] = text
            self._publish(CONSTITUTIONALIST_REGISTERED, {
                "item_id": item_id,
                "text_hash": _hash_text(text),
            })
            return item_id

    # ----- judge -----
    def judge(self,
              item_id: str,
              *,
              text: str | None = None,
              critic: Critic) -> Critique:
        """Score ``text`` (or the registered text for ``item_id``) against
        the constitution.  Returns a :class:`Critique`.

        Per-item RNG seed is derived from ``(config.seed, item_id,
        constitution_hash)`` so calling ``judge`` twice with the same
        ``(item_id, text)`` yields the same critique (modulo critic
        determinism).
        """
        with self._lock:
            if text is None:
                t = self._items.get(item_id)
                if t is None:
                    raise UnknownItem(
                        f"item_id={item_id!r} not registered and no text given"
                    )
            else:
                t = text
                self._items.setdefault(item_id, t)
            return self._judge_unlocked(item_id, t, critic=critic)

    def _judge_unlocked(self,
                        item_id: str,
                        text: str,
                        *,
                        critic: Critic) -> Critique:
        seed = _local_seed(
            self.config.seed,
            item_id,
            self.constitution.constitution_hash,
            _hash_text(text),
        )
        rng = random.Random(seed)
        try:
            raw = list(critic(text, self.constitution, rng=rng))
        except Exception as exc:
            raise InvalidCritique(f"critic raised: {exc!r}") from exc
        coerced = _coerce_scores(raw, self.constitution)

        scores: list[PrincipleScore] = []
        violations: list[str] = []
        critical: list[str] = []
        score_vals: list[float] = []
        weight_vals: list[float] = []
        worst_pid = ""
        worst_score = float("inf")

        for p in self.constitution.principles:
            ps = coerced.get(p.principle_id)
            if ps is None:
                # Missing → assume compliant by default.
                ps = PrincipleScore(
                    principle_id=p.principle_id,
                    score=1.0,
                    rationale="",
                )
            violated = float(ps.score) < float(p.threshold) - 1e-12
            if violated:
                violations.append(p.principle_id)
                if p.is_critical:
                    critical.append(p.principle_id)
            scores.append(PrincipleScore(
                principle_id=ps.principle_id,
                score=float(ps.score),
                violated=bool(violated),
                rationale=ps.rationale,
            ))
            score_vals.append(float(ps.score))
            weight_vals.append(float(p.weight))
            # Lex-stable tie break.
            if (float(ps.score) < worst_score
                    or (float(ps.score) == worst_score
                        and p.principle_id < worst_pid)):
                worst_score = float(ps.score)
                worst_pid = p.principle_id

        agg = _aggregate(
            score_vals, weight_vals, self.config.aggregator,
            temperature=self.config.soft_min_temperature,
        )
        text_hash = _hash_text(text)
        critique = Critique(
            item_id=item_id,
            text=text,
            scores=tuple(scores),
            aggregate_score=float(agg),
            worst_principle=worst_pid,
            worst_score=float(worst_score) if worst_score != float("inf") else 1.0,
            violations=tuple(violations),
            critical_violations=tuple(critical),
            text_hash=text_hash,
            constitution_hash=self.constitution.constitution_hash,
            seed=seed,
            ts=self._clock(),
        )

        # Accumulate per-principle stats.
        for p in self.constitution.principles:
            sv = next(s.score for s in critique.scores
                      if s.principle_id == p.principle_id)
            self._scores[p.principle_id].append(float(sv))
        self._aggregates.append(float(critique.aggregate_score))
        self._critiques.append(critique)

        self._publish(CONSTITUTIONALIST_JUDGED, {
            "item_id": item_id,
            "text_hash": text_hash,
            "aggregate_score": float(critique.aggregate_score),
            "worst_principle": worst_pid,
            "worst_score": float(critique.worst_score),
            "violations": list(violations),
            "critical_violations": list(critical),
            "constitution_hash": self.constitution.constitution_hash,
        })
        return critique

    # ----- revise -----
    def revise(self,
               item_id: str,
               *,
               text: str | None = None,
               critic: Critic,
               reviser: Reviser,
               max_iters: int | None = None) -> Revision:
        """Run the critique-revise loop on ``item_id``/``text``.

        Returns a :class:`Revision` whose ``final_text`` is the highest-
        scoring text observed across the trajectory (the loop is greedy
        — it accepts a revision only when ``aggregate_score`` strictly
        increases, unless ``require_strict_improvement`` is False).
        """
        with self._lock:
            if text is None:
                t = self._items.get(item_id)
                if t is None:
                    raise UnknownItem(
                        f"item_id={item_id!r} not registered and no text given"
                    )
            else:
                t = text
                self._items.setdefault(item_id, t)

            iters = self.config.max_iters if max_iters is None else int(max_iters)
            if iters < 0:
                raise ConstitutionalistError("max_iters must be >= 0")

            t0_total = self._clock()
            current_text = t
            current_critique = self._judge_unlocked(
                item_id, current_text, critic=critic,
            )
            steps: list[RevisionStep] = []
            stop_reason = STOP_MAX_ITER
            converged = False

            # Stop immediately if already above accept_threshold.
            if current_critique.aggregate_score >= self.config.accept_threshold:
                stop_reason = STOP_THRESHOLD
                converged = True

            best_text = current_text
            best_critique = current_critique
            best_score = current_critique.aggregate_score

            for step_idx in range(iters):
                if converged:
                    break
                step_seed = _local_seed(
                    self.config.seed,
                    item_id,
                    self.constitution.constitution_hash,
                    "step",
                    str(step_idx),
                )
                step_rng = random.Random(step_seed)
                t0 = self._clock()
                try:
                    revised = reviser(current_text, current_critique, rng=step_rng)
                except Exception as exc:
                    raise InvalidRevision(f"reviser raised: {exc!r}") from exc
                if not isinstance(revised, str):
                    raise InvalidRevision(
                        f"reviser must return str, got {type(revised).__name__}"
                    )

                # If reviser punted (returned same text), stop.
                if revised == current_text:
                    elapsed = (self._clock() - t0) * 1000.0
                    step = RevisionStep(
                        step=step_idx,
                        input_text=current_text,
                        critique=current_critique,
                        revised_text="",
                        improved=False,
                        elapsed_ms=elapsed,
                        seed=step_seed,
                    )
                    steps.append(step)
                    stop_reason = STOP_NONINCREASING
                    self._publish(CONSTITUTIONALIST_REVISED, step.to_dict())
                    break

                # Judge the revision.
                revised_critique = self._judge_unlocked(
                    item_id, revised, critic=critic,
                )
                improved = (revised_critique.aggregate_score
                            > current_critique.aggregate_score + 1e-12)
                elapsed = (self._clock() - t0) * 1000.0
                step = RevisionStep(
                    step=step_idx,
                    input_text=current_text,
                    critique=current_critique,
                    revised_text=revised,
                    improved=improved,
                    elapsed_ms=elapsed,
                    seed=step_seed,
                )
                steps.append(step)
                self._publish(CONSTITUTIONALIST_REVISED, step.to_dict())

                # Track best-so-far for the final_text.
                if revised_critique.aggregate_score > best_score:
                    best_text = revised
                    best_critique = revised_critique
                    best_score = revised_critique.aggregate_score

                if improved:
                    current_text = revised
                    current_critique = revised_critique
                    if current_critique.aggregate_score >= self.config.accept_threshold:
                        stop_reason = STOP_THRESHOLD
                        converged = True
                        break
                else:
                    if self.config.require_strict_improvement:
                        stop_reason = STOP_NONINCREASING
                        break
                    # Non-strict mode: keep iterating with the new text.
                    current_text = revised
                    current_critique = revised_critique

            total_elapsed = (self._clock() - t0_total) * 1000.0
            revision = Revision(
                item_id=item_id,
                original_text=t,
                final_text=best_text,
                steps=tuple(steps),
                final_critique=best_critique,
                converged=converged,
                stop_reason=stop_reason,
                total_elapsed_ms=total_elapsed,
            )
            self._revisions.append(revision)
            return revision

    # ----- best-of-N -----
    def bestof(self,
               item_id: str,
               *,
               text: str | None = None,
               n: int,
               critic: Critic,
               reviser: Reviser) -> Revision:
        """Generate ``n`` independent revision trajectories (each with its
        own seed) and return the one whose ``final_critique`` has the
        highest aggregate score.  Deterministic tie-break: earliest
        completed revision wins.
        """
        if not isinstance(n, int) or n < 1:
            raise ConstitutionalistError("n must be a positive int")
        with self._lock:
            if text is None:
                t = self._items.get(item_id)
                if t is None:
                    raise UnknownItem(
                        f"item_id={item_id!r} not registered and no text given"
                    )
            else:
                t = text
                self._items.setdefault(item_id, t)

            best: Revision | None = None
            for k in range(n):
                # Inject a deterministic per-branch seed by salting item_id.
                branch_id = f"{item_id}#bestof:{k}"
                self._items[branch_id] = t
                rev = self.revise(
                    branch_id, text=t, critic=critic, reviser=reviser,
                )
                if (best is None
                        or rev.final_critique.aggregate_score
                            > best.final_critique.aggregate_score + 1e-12):
                    best = rev

            # Re-attribute the winning revision to the parent ``item_id``
            # so the caller's bookkeeping is clean.  We do this by
            # appending a fresh Revision record under ``item_id`` whose
            # trajectory matches the winner; the per-branch records are
            # preserved for full auditability.
            winning = Revision(
                item_id=item_id,
                original_text=best.original_text,
                final_text=best.final_text,
                steps=best.steps,
                final_critique=Critique(
                    item_id=item_id,
                    text=best.final_critique.text,
                    scores=best.final_critique.scores,
                    aggregate_score=best.final_critique.aggregate_score,
                    worst_principle=best.final_critique.worst_principle,
                    worst_score=best.final_critique.worst_score,
                    violations=best.final_critique.violations,
                    critical_violations=best.final_critique.critical_violations,
                    text_hash=best.final_critique.text_hash,
                    constitution_hash=best.final_critique.constitution_hash,
                    seed=best.final_critique.seed,
                    ts=best.final_critique.ts,
                ),
                converged=best.converged,
                stop_reason=best.stop_reason,
                total_elapsed_ms=best.total_elapsed_ms,
            )
            self._revisions.append(winning)
            self._publish(CONSTITUTIONALIST_BESTOF, {
                "item_id": item_id,
                "n": n,
                "winning_aggregate": float(winning.final_critique.aggregate_score),
                "winning_final_hash": _hash_text(winning.final_text),
                "winning_stop_reason": winning.stop_reason,
            })
            return winning

    # ----- gate (the inline-policy entry point) -----
    def gate(self,
             item_id: str,
             *,
             text: str | None = None,
             critic: Critic,
             reviser: Reviser | None = None,
             max_iters: int | None = None) -> Verdict:
        """One-call gate: judge → (optionally) revise → accept/revise/refuse.

        Returns a :class:`Verdict` carrying the post-gate text and a
        machine-readable rationale.  A reviser is required iff the
        initial aggregate score falls between ``violation_threshold``
        and ``accept_threshold``.
        """
        with self._lock:
            if text is None:
                t = self._items.get(item_id)
                if t is None:
                    raise UnknownItem(
                        f"item_id={item_id!r} not registered and no text given"
                    )
            else:
                t = text
                self._items.setdefault(item_id, t)

            critique = self._judge_unlocked(item_id, t, critic=critic)
            crit_violations = critique.critical_violations

            # Critical-severity violation: hard refuse without revision.
            if self.config.refuse_on_critical and crit_violations:
                self._critical += 1
                self._refuse += 1
                rationale = (
                    f"refuse:critical_violation:{','.join(crit_violations)}"
                )
                verdict = Verdict(
                    item_id=item_id,
                    action=ACTION_REFUSE,
                    text="",
                    revision=None,
                    rationale=rationale,
                    fingerprint=self.fingerprint_hash,
                )
                self._verdicts.append(verdict)
                self._publish(CONSTITUTIONALIST_REFUSED, {
                    "item_id": item_id,
                    "rationale": rationale,
                    "critical_violations": list(crit_violations),
                    "aggregate_score": float(critique.aggregate_score),
                })
                return verdict

            # Already above accept_threshold: accept untouched.
            if critique.aggregate_score >= self.config.accept_threshold:
                self._accept += 1
                rationale = (
                    f"accept:above_threshold:aggregate={critique.aggregate_score:.4g}"
                )
                verdict = Verdict(
                    item_id=item_id,
                    action=ACTION_ACCEPT,
                    text=t,
                    revision=None,
                    rationale=rationale,
                    fingerprint=self.fingerprint_hash,
                )
                self._verdicts.append(verdict)
                self._publish(CONSTITUTIONALIST_ACCEPTED, {
                    "item_id": item_id,
                    "rationale": rationale,
                    "aggregate_score": float(critique.aggregate_score),
                    "worst_principle": critique.worst_principle,
                })
                return verdict

            # Below accept_threshold: need a reviser.
            if reviser is None:
                # Reviser absent: refuse if below violation threshold,
                # accept-with-warning otherwise.
                if critique.aggregate_score < self.config.violation_threshold:
                    self._refuse += 1
                    rationale = (
                        f"refuse:below_violation_threshold:aggregate="
                        f"{critique.aggregate_score:.4g}"
                    )
                    action = ACTION_REFUSE
                    out_text = ""
                else:
                    self._accept += 1
                    rationale = (
                        f"accept:warn:aggregate={critique.aggregate_score:.4g}"
                    )
                    action = ACTION_ACCEPT
                    out_text = t
                verdict = Verdict(
                    item_id=item_id,
                    action=action,
                    text=out_text,
                    revision=None,
                    rationale=rationale,
                    fingerprint=self.fingerprint_hash,
                )
                self._verdicts.append(verdict)
                if action == ACTION_REFUSE:
                    self._publish(CONSTITUTIONALIST_REFUSED, {
                        "item_id": item_id,
                        "rationale": rationale,
                        "aggregate_score": float(critique.aggregate_score),
                    })
                else:
                    self._publish(CONSTITUTIONALIST_ACCEPTED, {
                        "item_id": item_id,
                        "rationale": rationale,
                        "aggregate_score": float(critique.aggregate_score),
                        "warning": True,
                    })
                return verdict

            # Run the loop.  judge() inside revise() will re-append the
            # initial critique we already produced.  Skip the duplicate by
            # routing directly through the loop body.
            revision = self.revise(
                item_id, text=t, critic=critic, reviser=reviser,
                max_iters=max_iters,
            )
            final_agg = revision.final_critique.aggregate_score
            final_critical = revision.final_critique.critical_violations

            if self.config.refuse_on_critical and final_critical:
                self._critical += 1
                self._refuse += 1
                rationale = (
                    f"refuse:revised_critical_violation:"
                    f"{','.join(final_critical)}"
                )
                verdict = Verdict(
                    item_id=item_id,
                    action=ACTION_REFUSE,
                    text="",
                    revision=revision,
                    rationale=rationale,
                    fingerprint=self.fingerprint_hash,
                )
            elif final_agg >= self.config.accept_threshold:
                self._revise += 1
                rationale = (
                    f"revise:converged:aggregate={final_agg:.4g}"
                )
                verdict = Verdict(
                    item_id=item_id,
                    action=ACTION_REVISE,
                    text=revision.final_text,
                    revision=revision,
                    rationale=rationale,
                    fingerprint=self.fingerprint_hash,
                )
            elif final_agg >= self.config.violation_threshold:
                self._revise += 1
                rationale = (
                    f"revise:partial:aggregate={final_agg:.4g}"
                )
                verdict = Verdict(
                    item_id=item_id,
                    action=ACTION_REVISE,
                    text=revision.final_text,
                    revision=revision,
                    rationale=rationale,
                    fingerprint=self.fingerprint_hash,
                )
            else:
                # Iteration budget spent and still below violation_threshold.
                if self.config.refuse_after_iters:
                    self._refuse += 1
                    rationale = (
                        f"refuse:iter_budget_exhausted:aggregate={final_agg:.4g}"
                    )
                    verdict = Verdict(
                        item_id=item_id,
                        action=ACTION_REFUSE,
                        text="",
                        revision=revision,
                        rationale=rationale,
                        fingerprint=self.fingerprint_hash,
                    )
                else:
                    self._revise += 1
                    rationale = (
                        f"revise:no_refuse_policy:aggregate={final_agg:.4g}"
                    )
                    verdict = Verdict(
                        item_id=item_id,
                        action=ACTION_REVISE,
                        text=revision.final_text,
                        revision=revision,
                        rationale=rationale,
                        fingerprint=self.fingerprint_hash,
                    )
            self._verdicts.append(verdict)
            if verdict.action == ACTION_REFUSE:
                self._publish(CONSTITUTIONALIST_REFUSED, {
                    "item_id": item_id,
                    "rationale": verdict.rationale,
                    "aggregate_score": float(final_agg),
                })
            elif verdict.action == ACTION_REVISE:
                self._publish(CONSTITUTIONALIST_ACCEPTED, {
                    "item_id": item_id,
                    "rationale": verdict.rationale,
                    "aggregate_score": float(final_agg),
                    "revised": True,
                })
            else:
                self._publish(CONSTITUTIONALIST_ACCEPTED, {
                    "item_id": item_id,
                    "rationale": verdict.rationale,
                    "aggregate_score": float(final_agg),
                })
            return verdict

    # ----- preference mining -----
    def mine_preferences(self) -> tuple[dict[str, Any], ...]:
        """Mine ``(rejected, chosen)`` pairs from completed revisions.

        For each revision whose ``final_text`` improved over the original
        we emit one pair per accepted intermediate step.  Suitable
        directly as DPO / KTO / IPO training data for :mod:`agi.aligner`.
        """
        with self._lock:
            out: list[dict[str, Any]] = []
            for rev in self._revisions:
                # Only emit pairs where final beat original strictly.
                # We compare against the *first* judged critique, which is
                # the original-text critique.
                if not rev.steps:
                    continue
                # Original-text critique is steps[0].critique.
                original_agg = rev.steps[0].critique.aggregate_score
                final_agg = rev.final_critique.aggregate_score
                if final_agg <= original_agg + 1e-12:
                    continue
                out.append({
                    "item_id": rev.item_id,
                    "rejected": rev.original_text,
                    "chosen": rev.final_text,
                    "rejected_score": float(original_agg),
                    "chosen_score": float(final_agg),
                    "constitution_hash": self.constitution.constitution_hash,
                    "rejected_hash": _hash_text(rev.original_text),
                    "chosen_hash": _hash_text(rev.final_text),
                    "violations_rejected": list(rev.steps[0].critique.violations),
                    "violations_final": list(rev.final_critique.violations),
                    "stop_reason": rev.stop_reason,
                })
            return tuple(out)

    # ----- certificate / report -----
    def certificate(self) -> ConstitutionalistCertificate:
        """Per-principle, joint-corrected PAC certificate.

        Raises :class:`ConstitutionalistError` if fewer than
        ``min_items_for_certificate`` items have been judged.
        """
        with self._lock:
            # Items judged at least once (could be > 1 due to revisions).
            n = len(self._critiques)
            if n < self.config.min_items_for_certificate:
                raise ConstitutionalistError(
                    f"need at least {self.config.min_items_for_certificate}"
                    f" judged items; have {n}"
                )

            # Per-principle violation counts.  We use *first-judged* per
            # item_id so a revision loop doesn't bias the rate downward.
            # NB: we keep order across revisions; for the certificate we
            # collapse to per-item_id earliest critique.
            earliest: dict[str, Critique] = {}
            for c in self._critiques:
                if c.item_id not in earliest:
                    earliest[c.item_id] = c
            items: tuple[Critique, ...] = tuple(earliest.values())
            n_items = len(items)
            if n_items < self.config.min_items_for_certificate:
                raise ConstitutionalistError(
                    f"need at least {self.config.min_items_for_certificate}"
                    f" distinct items; have {n_items}"
                )

            principles = self.constitution.principles
            k_principles = len(principles)
            alphas: list[float]
            if self.config.joint_correction:
                alphas = _holm_alphas(self.config.alpha, k_principles)
            else:
                alphas = [self.config.alpha] * k_principles

            per_principle: list[PrincipleCertificate] = []
            worst_rate = 0.0
            worst_pid = principles[0].principle_id

            # Sort principles by ascending violation count so Holm gives
            # tighter alpha to the principles closer to compliance.
            # Returns are in original constitution order so the report is
            # readable.
            counts: dict[str, tuple[int, list[float]]] = {}
            for p in principles:
                ss = list(self._scores.get(p.principle_id, []))
                # Subset to first-judged-per-item by index alignment:
                # _scores accumulates *every* judge call.  We approximate
                # the per-item earliest score by taking the first score
                # observed in the order critiques were appended.
                first_scores: dict[str, float] = {}
                for c in self._critiques:
                    if c.item_id in first_scores:
                        continue
                    s = next(
                        (s.score for s in c.scores if s.principle_id == p.principle_id),
                        1.0,
                    )
                    first_scores[c.item_id] = float(s)
                vs = list(first_scores.values())
                viol = sum(1 for s in vs if s < p.threshold - 1e-12)
                counts[p.principle_id] = (viol, vs)

            ranked = sorted(
                principles,
                key=lambda p: (counts[p.principle_id][0], p.principle_id),
            )
            principle_to_rank: dict[str, int] = {
                p.principle_id: i for i, p in enumerate(ranked)
            }

            for p in principles:
                viol, vs = counts[p.principle_id]
                local_n = len(vs)
                rank = principle_to_rank[p.principle_id]
                a_local = alphas[rank] if alphas else self.config.alpha
                wlo, whi = _wilson_ci(viol, local_n, a_local)
                hub = _hoeffding_ucb_rate(viol, local_n, a_local)
                mean = sum(vs) / local_n if local_n else 1.0
                eb_lo = _empirical_bernstein_lcb(vs, a_local, 1.0)
                cert = PrincipleCertificate(
                    principle_id=p.principle_id,
                    n_items=int(local_n),
                    n_violations=int(viol),
                    violation_rate=float(viol / local_n) if local_n else 0.0,
                    wilson_lo=float(wlo),
                    wilson_hi=float(whi),
                    hoeffding_hi=float(hub),
                    score_mean=float(mean),
                    score_eb_lo=float(eb_lo),
                    adjusted_alpha=float(a_local),
                )
                per_principle.append(cert)
                rate = cert.violation_rate
                if rate > worst_rate or (rate == worst_rate
                                          and p.principle_id < worst_pid):
                    worst_rate = rate
                    worst_pid = p.principle_id

            # Aggregate stats.
            aggs = [c.aggregate_score for c in items]
            agg_mean = sum(aggs) / len(aggs)
            agg_eb_lo = _empirical_bernstein_lcb(
                aggs, self.config.alpha, 1.0,
            )

            cert = ConstitutionalistCertificate(
                n_items=int(n_items),
                constitution_hash=self.constitution.constitution_hash,
                principles=tuple(per_principle),
                worst_violation_rate=float(worst_rate),
                worst_principle=worst_pid,
                aggregate_mean=float(agg_mean),
                aggregate_eb_lo=float(agg_eb_lo),
                accept_count=int(self._accept),
                revise_count=int(self._revise),
                refuse_count=int(self._refuse),
                critical_count=int(self._critical),
                alpha=float(self.config.alpha),
                joint_correction=bool(self.config.joint_correction),
                fingerprint=self.fingerprint_hash,
            )
            self._publish(CONSTITUTIONALIST_CERTIFIED, {
                "n_items": cert.n_items,
                "worst_violation_rate": cert.worst_violation_rate,
                "worst_principle": cert.worst_principle,
                "aggregate_mean": cert.aggregate_mean,
                "aggregate_eb_lo": cert.aggregate_eb_lo,
                "accept_count": cert.accept_count,
                "revise_count": cert.revise_count,
                "refuse_count": cert.refuse_count,
                "critical_count": cert.critical_count,
                "alpha": cert.alpha,
                "fingerprint": cert.fingerprint,
            })
            return cert

    def report(self) -> ConstitutionalistReport:
        with self._lock:
            cert = self.certificate()
            rep = ConstitutionalistReport(
                items=int(len(set(c.item_id for c in self._critiques))),
                revisions_run=int(len(self._revisions)),
                certificate=cert,
            )
            self._publish(CONSTITUTIONALIST_REPORTED, {
                "items": rep.items,
                "revisions_run": rep.revisions_run,
            })
            return rep

    def reset(self) -> None:
        """Drop all per-item state; preserve config + constitution.

        The fingerprint chain *continues* across the reset so the
        operation is itself audit-visible.
        """
        with self._lock:
            self._items.clear()
            self._critiques.clear()
            self._revisions.clear()
            self._verdicts.clear()
            self._scores = {
                p.principle_id: [] for p in self.constitution.principles
            }
            self._aggregates.clear()
            self._accept = 0
            self._revise = 0
            self._refuse = 0
            self._critical = 0
            self._publish(CONSTITUTIONALIST_RESET, {})
