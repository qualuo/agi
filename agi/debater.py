r"""Debater — multi-agent debate as a runtime primitive.

The hard part of an agent runtime is not generating an answer.  It is
**deciding whether to trust an answer** that no single component is
strong enough to verify on its own.  A coordinator that can only ask
*one* expensive expert and accept its word can never exceed that
expert's calibration; the moment the expert is wrong the runtime is
wrong with it.  Two centuries of social-choice theory, half a century
of interactive-proof theory, and the last decade of AI-safety research
all converge on the same answer: **adversarial debate between symmetric
agents, refereed by a weaker but honest judge, is a strictly more
trustworthy verifier than any single agent**.

``Debater`` is the runtime-level implementation of that primitive.
It owns the round structure, the cross-examination kernel, the judge
loop, the persuasion-aware scoring, the equilibrium check, the
multi-judge jury aggregation, the anytime-valid stopping rule on win
rates across many debates, the calibration recaliber, and a
replay-verifiable hash-chained transcript so every decision is
re-derivable and tamper-evident.

The pitch reduced to a runtime call::

    d = Debater(DebaterConfig(
        protocol=PROTOCOL_TWO_PLAYER,
        max_depth=4,
        judge_accuracy=0.7,
        seed=0,
    ))

    rep  = d.run(spec)              # DebateReport
    cert = d.certify(rep, delta=0.05)  # DebaterCertificate (PAC LCB)

    # As a runtime primitive: any module that asks "is this answer
    # correct?" delegates to d.run() with two competing claims +
    # a judge callable + (optionally) a cross-examination kernel
    # and gets back a winner, an anytime-valid PAC certificate on
    # the probability that the winner's claim is true, an empirical
    # Nash check on the debate game's bimatrix payoff, and a
    # tamper-evident SHA-256 transcript.


What this primitive ships
-------------------------

  * **Six debate protocols** — toggleable via ``DebaterConfig.protocol``:

    * ``PROTOCOL_TWO_PLAYER``     — sequential two-player debate
      (Irving-Christiano-Amodei 2018 *AI Safety via Debate*).  Two
      symmetric debaters argue for opposite claims; the judge sees
      the full transcript at the end and selects a winner.  Bounded
      recursion via ``max_depth``: any claim can be challenged once
      and the challenge recursively debated up to the depth cap.
      The base case at depth ``0`` is the judge's direct verdict.

    * ``PROTOCOL_CROSS_EXAM``     — cross-examination debate
      (Barnes-Christiano 2020 *Debate update: Obfuscated arguments
      problem*).  In addition to the standard argue / counter-argue
      moves, each side can ``cross_examine`` a specific sub-claim
      that the opponent must commit to before the debate continues.
      Defeats the obfuscated-arguments attack on plain two-player
      debate.

    * ``PROTOCOL_DOUBLY_EFFICIENT`` — doubly-efficient debate
      (Brown-Cohen-Irving-Piliouras 2023 *Scalable AI Safety via
      Doubly-Efficient Debate*).  A bounded-depth tree where the
      judge only needs to verify a single leaf in polynomial time
      while the debaters reason over an exponential search.  Truth
      is preserved at each layer by a verifiable-step contract.

    * ``PROTOCOL_MARKET_MAKER``   — market-maker debate (Hubinger
      2020 *AI safety via market making*).  A single AI argues
      against an internal prediction-market subagent that posts
      counter-evidence whenever its price-implied probability
      crosses the AI's claim.  Equivalent to two-player debate when
      both sides reduce to the same probability-elicitation
      objective; cheaper at runtime when the second debater is
      well-modelled by a market subagent.

    * ``PROTOCOL_JURY``           — Condorcet jury aggregation
      (Condorcet 1785; Grofman-Owen-Feld 1983; Boland 1989).  ``M``
      independent judges each adjudicate the same debate;
      majority-vote wins.  Provably amplifies any per-judge accuracy
      ``> 0.5`` to ``→ 1`` exponentially in ``M`` with a finite-
      sample Hoeffding correction.

    * ``PROTOCOL_PERSUASION_AWARE`` — persuasion-aware debate
      (Khan-Hughes-Valentine-Ruis-Sachan-Radhakrishnan-Grefenstette-
      Bowman-Rocktäschel-Perez 2024 *Debating with More Persuasive
      LLMs Leads to More Truthful Answers*).  Each round's
      persuasion delta (judge's posterior shift) is decomposed into
      a *truthful* component (matched by verifiable evidence) and a
      *manipulative* component (un-verifiable rhetorical gain); the
      manipulative component is penalised on the win-rate score.

  * **Three judge models** — toggleable via ``DebaterConfig.judge_model``:

    * ``JUDGE_CALIBRATED``        — caller-supplied judge with a
      calibrated per-claim accuracy parameter ``judge_accuracy``;
      Debater treats the judge as a Bernoulli oracle and bounds
      win-rate variance with Hoeffding / empirical Bernstein.

    * ``JUDGE_PERSUASION_MODEL``  — caller-supplied judge plus a
      caller-supplied persuasion model (e.g. Mentalist).  The
      debate scores the judge's posterior shift after each round
      and applies the Khan-2024 truthful/manipulative decomposition.

    * ``JUDGE_JURY``              — ``M`` independent judges with
      per-judge accuracies.  Aggregates via majority vote (default)
      or Mansoor-aggregation (weighted log-odds).  Returns the
      majority winner plus the Condorcet PAC bound.

  * **Strategy spaces** — every debate exposes a finite ``policy``
    set (e.g. ``"truthful"`` / ``"obfuscate"`` / ``"abstain"``) per
    side; ``Debater.empirical_payoff`` returns the bimatrix of
    win-rates over a Monte-Carlo sample of debates and
    ``Debater.nash_check`` runs support-enumeration on the bimatrix
    to find a pure or mixed Nash; the *exploitability gap*
    ``NashConv = max_dev(payoff_dev) − payoff_current`` is reported
    as a primary certificate.

  * **Anytime-valid stopping** — when a coordinator runs many
    debates over a stream of queries, ``Debater.anytime_certify``
    returns a Howard-Ramdas-McAuliffe-Sekhon (HRMS) confidence
    sequence on the cumulative truth-win-rate so the coordinator
    can stop sampling under any data-dependent stopping rule with
    valid finite-sample coverage.

  * **PAC certificates** — every report carries:

    * ``winner``               — the winning claim
    * ``win_prob_hat``         — empirical win-rate over judges/Monte-Carlo
    * ``hoeffding_lcb(δ)``     — Hoeffding LCB on truth-win-rate
    * ``bernstein_lcb(δ)``     — Maurer-Pontil empirical-Bernstein LCB
    * ``condorcet_lcb(δ)``     — Boland-bound LCB for jury protocols
    * ``persuasion_penalty``   — Khan-2024 manipulative-gain penalty
    * ``nash_conv``            — exploitability gap of the play profile
    * ``calibration_ece``      — judge's empirical calibration error

  * **Replay-verifiable transcripts** — SHA-256 fingerprint chain
    (optionally HMAC'd) over every observation: ``opened``,
    ``argued``, ``countered``, ``cross_examined``, ``conceded``,
    ``judge_polled``, ``verdict``, ``calibrated``, ``closed``.
    ``debater_ledger_root`` is the immutable genesis
    ``agi.debater.v1``.  Replaying the chain reproduces every
    transcript byte-for-byte.

  * **Snapshot / restore** — ``snapshot()`` returns a JSON-encodable
    dict (RNG state, per-debate transcripts, jury votes, calibration
    bins) that ``restore()`` can use to resume execution
    byte-identically.

  * **Thread-safe re-entrant lock**; transport-agnostic; pure
    stdlib (no NumPy, no SciPy, no Torch); deterministic given seed.


Composes with
-------------

  * ``Mentalist`` — supplies the *judge belief model*.  Mentalist's
    posterior over the judge's hidden state becomes the persuasion-
    aware scoring's calibration prior.

  * ``TruthSerum`` — drops in as an alternative judge when no
    ground-truth label exists.  Bayesian Truth Serum / Correlated
    Agreement gives an incentive-compatible scoring signal for
    debate winners.

  * ``Reconciler`` — Aumann's agreement theorem terminates the
    debate as soon as both debaters' posteriors agree on a
    sufficiently small KL ball; Reconciler's iteration is the
    natural convergence test for the debate's claim.

  * ``Equilibrator`` — the strategy-space Nash check delegates to
    Equilibrator's support-enumeration / multiplicative-weights
    solver when the policy space exceeds the built-in 2-3 strategy
    enumeration.

  * ``Arbiter`` — when many debates are run to identify the best
    of several candidate answers, Arbiter's PAC-best-arm stopping
    rule consumes the per-debate win-rates and tells the
    coordinator when to halt with (ε, δ)-PAC confidence.

  * ``Auditor`` — when N parallel debates over a batch of claims
    each emit a per-debate p-value of truth, Auditor's BH / e-BH
    controls the discovery rate at level α.

  * ``Strategist`` — debate's win-rate LCB and persuasion penalty
    feed Strategist's risk-adjusted recommendation; the *NashConv*
    exploitability gap is a strategic-stability risk dimension.

  * ``AttestationLedger`` — every transcript line is chain-hashed;
    third parties can replay-verify the full debate.

  * ``Coordinator`` — every Goal whose acceptance requires verified
    correctness of an unconstrained generation routes the
    candidate answer through Debater before the verdict is fed
    back to ``AutonomousLoop``.


Mathematical notation
---------------------

  * ``T``                 — number of rounds in a single debate.
  * ``d``                 — recursion depth (1 = no recursion).
  * ``M``                 — number of judges in a jury.
  * ``p``                 — per-judge probability of selecting truth.
  * ``W_n``               — empirical truth-win-rate after ``n`` debates.
  * ``δ``                 — confidence parameter (failure prob).
  * ``ΔBel_t``            — judge posterior shift in round ``t``.
  * ``π``                 — debate-game mixed strategy.
  * ``NashConv``          — exploitability of profile ``π``.

All ingest paths are validated.  Inference is ``O(T · d · M)`` per
debate plus ``O(K)`` for a ``K``-strategy Nash check.  No
``random`` without explicit seed; no ``time.time()`` leaks into the
chain.


References
----------

  * Condorcet 1785. *Essai sur l'application de l'analyse à la
    probabilité des décisions rendues à la pluralité des voix.*
  * Boland 1989. *Majority Systems and the Condorcet Jury Theorem.*
    The Statistician 38:181-189.
  * Grofman, Owen, Feld 1983. *Thirteen theorems in search of the
    truth.* Theory and Decision 15:261-278.
  * Goldwasser, Micali, Rackoff 1985. *The Knowledge Complexity of
    Interactive Proof-Systems.* STOC.
  * Babai 1985. *Trading group theory for randomness.* STOC.
  * Aumann 1976. *Agreeing to Disagree.* Annals of Statistics 4:1236-9.
  * Irving, Christiano, Amodei 2018. *AI Safety via Debate.* arXiv.
  * Barnes, Christiano 2020. *Debate update: Obfuscated arguments
    problem.* AI Alignment Forum.
  * Hubinger 2020. *AI safety via market making.* AI Alignment Forum.
  * Brown-Cohen, Irving, Piliouras 2023. *Scalable AI Safety via
    Doubly-Efficient Debate.* arXiv:2311.14125.
  * Khan, Hughes, Valentine, Ruis, Sachan, Radhakrishnan,
    Grefenstette, Bowman, Rocktäschel, Perez 2024. *Debating with
    More Persuasive LLMs Leads to More Truthful Answers.* ICML.
  * Hoeffding 1963. *Probability Inequalities for Sums of Bounded
    Random Variables.* JASA 58:13-30.
  * Maurer, Pontil 2009. *Empirical Bernstein Bounds and Sample
    Variance Penalisation.* COLT.
  * Howard, Ramdas, McAuliffe, Sekhon 2021. *Time-uniform,
    nonparametric, nonasymptotic confidence sequences.* Annals of
    Statistics 49:1055-80.
  * Nash 1950. *Equilibrium Points in N-Person Games.* PNAS
    36:48-49.
  * Lemke, Howson 1964. *Equilibrium Points of Bimatrix Games.*
    SIAM Journal 12:413-423.
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
    "DEBATER_STARTED",
    "DEBATER_OPENED",
    "DEBATER_ARGUED",
    "DEBATER_COUNTERED",
    "DEBATER_CROSS_EXAMINED",
    "DEBATER_CONCEDED",
    "DEBATER_JUDGE_POLLED",
    "DEBATER_VERDICT",
    "DEBATER_CALIBRATED",
    "DEBATER_CLOSED",
    "DEBATER_CERTIFIED",
    "DEBATER_RESET",
    # Protocols
    "PROTOCOL_TWO_PLAYER",
    "PROTOCOL_CROSS_EXAM",
    "PROTOCOL_DOUBLY_EFFICIENT",
    "PROTOCOL_MARKET_MAKER",
    "PROTOCOL_JURY",
    "PROTOCOL_PERSUASION_AWARE",
    "KNOWN_PROTOCOLS",
    # Judge models
    "JUDGE_CALIBRATED",
    "JUDGE_PERSUASION_MODEL",
    "JUDGE_JURY",
    "KNOWN_JUDGE_MODELS",
    # Aggregation
    "AGG_MAJORITY",
    "AGG_WEIGHTED_LOG_ODDS",
    "AGG_UNANIMITY",
    "KNOWN_AGGREGATIONS",
    # Move kinds (for transcripts)
    "MOVE_OPEN",
    "MOVE_ARGUE",
    "MOVE_COUNTER",
    "MOVE_CROSS_EXAMINE",
    "MOVE_ANSWER",
    "MOVE_CONCEDE",
    "MOVE_VERDICT",
    "KNOWN_MOVES",
    # Side labels
    "SIDE_A",
    "SIDE_B",
    "SIDE_TIE",
    # Exceptions
    "DebaterError",
    "InvalidConfig",
    "InvalidSpec",
    "InvalidMove",
    "InvalidTranscript",
    "InsufficientData",
    "UnknownProtocol",
    "UnknownJudgeModel",
    "UnknownAggregation",
    "NotRun",
    # Dataclasses
    "DebaterConfig",
    "Argument",
    "DebateMove",
    "DebateSpec",
    "JuryConfig",
    "DebateReport",
    "DebaterCertificate",
    "AnytimeCertificate",
    "PayoffMatrix",
    "NashResult",
    "CalibrationReport",
    # Helpers
    "debater_ledger_root",
    "debater_hoeffding_lcb",
    "debater_bernstein_lcb",
    "debater_condorcet_lcb",
    "debater_hrms_radius",
    "debater_jury_majority",
    "debater_jury_log_odds",
    "debater_payoff_nash_2x2",
    "debater_support_enumeration",
    "debater_bayes_posterior_shift",
    "debater_persuasion_decomposition",
    "debater_calibration_ece",
    # Defaults / factories
    "make_constant_debater",
    "make_calibrated_judge",
    # Main class
    "Debater",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

DEBATER_STARTED = "debater.started"
DEBATER_OPENED = "debater.opened"
DEBATER_ARGUED = "debater.argued"
DEBATER_COUNTERED = "debater.countered"
DEBATER_CROSS_EXAMINED = "debater.cross_examined"
DEBATER_CONCEDED = "debater.conceded"
DEBATER_JUDGE_POLLED = "debater.judge_polled"
DEBATER_VERDICT = "debater.verdict"
DEBATER_CALIBRATED = "debater.calibrated"
DEBATER_CLOSED = "debater.closed"
DEBATER_CERTIFIED = "debater.certified"
DEBATER_RESET = "debater.reset"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

PROTOCOL_TWO_PLAYER = "two_player"
PROTOCOL_CROSS_EXAM = "cross_exam"
PROTOCOL_DOUBLY_EFFICIENT = "doubly_efficient"
PROTOCOL_MARKET_MAKER = "market_maker"
PROTOCOL_JURY = "jury"
PROTOCOL_PERSUASION_AWARE = "persuasion_aware"
KNOWN_PROTOCOLS = (
    PROTOCOL_TWO_PLAYER,
    PROTOCOL_CROSS_EXAM,
    PROTOCOL_DOUBLY_EFFICIENT,
    PROTOCOL_MARKET_MAKER,
    PROTOCOL_JURY,
    PROTOCOL_PERSUASION_AWARE,
)

JUDGE_CALIBRATED = "calibrated"
JUDGE_PERSUASION_MODEL = "persuasion_model"
JUDGE_JURY = "jury"
KNOWN_JUDGE_MODELS = (JUDGE_CALIBRATED, JUDGE_PERSUASION_MODEL, JUDGE_JURY)

AGG_MAJORITY = "majority"
AGG_WEIGHTED_LOG_ODDS = "weighted_log_odds"
AGG_UNANIMITY = "unanimity"
KNOWN_AGGREGATIONS = (AGG_MAJORITY, AGG_WEIGHTED_LOG_ODDS, AGG_UNANIMITY)

MOVE_OPEN = "open"
MOVE_ARGUE = "argue"
MOVE_COUNTER = "counter"
MOVE_CROSS_EXAMINE = "cross_examine"
MOVE_ANSWER = "answer"
MOVE_CONCEDE = "concede"
MOVE_VERDICT = "verdict"
KNOWN_MOVES = (
    MOVE_OPEN,
    MOVE_ARGUE,
    MOVE_COUNTER,
    MOVE_CROSS_EXAMINE,
    MOVE_ANSWER,
    MOVE_CONCEDE,
    MOVE_VERDICT,
)

SIDE_A = "A"
SIDE_B = "B"
SIDE_TIE = "tie"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DebaterError(Exception):
    """Base class for all :mod:`agi.debater` errors."""


class InvalidConfig(DebaterError):
    """A :class:`DebaterConfig` field is out of range."""


class InvalidSpec(DebaterError):
    """A :class:`DebateSpec` is malformed."""


class InvalidMove(DebaterError):
    """A debater returned a malformed :class:`DebateMove`."""


class InvalidTranscript(DebaterError):
    """A transcript fails consistency checks (chain-hash, ordering)."""


class InsufficientData(DebaterError):
    """An operation requires more debates/judges than have been observed."""


class UnknownProtocol(DebaterError):
    """``protocol`` is not in :data:`KNOWN_PROTOCOLS`."""


class UnknownJudgeModel(DebaterError):
    """``judge_model`` is not in :data:`KNOWN_JUDGE_MODELS`."""


class UnknownAggregation(DebaterError):
    """``aggregation`` is not in :data:`KNOWN_AGGREGATIONS`."""


class NotRun(DebaterError):
    """An operation requires ``run`` to have been called first."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebaterConfig:
    """Static configuration for a :class:`Debater` instance.

    All probabilities are in ``(0, 1)``; ``judge_accuracy`` must be
    strictly greater than ``0.5`` for the Condorcet bound to be
    informative.  ``protocol`` must be in :data:`KNOWN_PROTOCOLS`;
    ``judge_model`` must be in :data:`KNOWN_JUDGE_MODELS`.  ``seed``
    is required for replay; if ``hmac_key`` is non-empty every
    receipt is HMAC-SHA-256 signed.
    """

    protocol: str = PROTOCOL_TWO_PLAYER
    judge_model: str = JUDGE_CALIBRATED
    aggregation: str = AGG_MAJORITY
    max_rounds: int = 8
    max_depth: int = 2
    judge_accuracy: float = 0.7
    persuasion_penalty_weight: float = 1.0
    truthful_evidence_threshold: float = 0.1
    concede_threshold: float = 0.05
    confidence: float = 0.95
    record_every: int = 1
    seed: int = 0
    hmac_key: bytes = b""

    def __post_init__(self) -> None:
        if self.protocol not in KNOWN_PROTOCOLS:
            raise UnknownProtocol(
                f"protocol={self.protocol!r} not in {KNOWN_PROTOCOLS}"
            )
        if self.judge_model not in KNOWN_JUDGE_MODELS:
            raise UnknownJudgeModel(
                f"judge_model={self.judge_model!r} not in {KNOWN_JUDGE_MODELS}"
            )
        if self.aggregation not in KNOWN_AGGREGATIONS:
            raise UnknownAggregation(
                f"aggregation={self.aggregation!r} not in {KNOWN_AGGREGATIONS}"
            )
        if self.max_rounds <= 0:
            raise InvalidConfig(f"max_rounds must be > 0; got {self.max_rounds}")
        if self.max_depth <= 0:
            raise InvalidConfig(f"max_depth must be > 0; got {self.max_depth}")
        if not (0.0 < self.judge_accuracy < 1.0):
            raise InvalidConfig(
                f"judge_accuracy must be in (0, 1); got {self.judge_accuracy}"
            )
        if self.persuasion_penalty_weight < 0.0:
            raise InvalidConfig(
                f"persuasion_penalty_weight must be >= 0; got {self.persuasion_penalty_weight}"
            )
        if not (0.0 <= self.truthful_evidence_threshold <= 1.0):
            raise InvalidConfig(
                f"truthful_evidence_threshold must be in [0, 1]; got {self.truthful_evidence_threshold}"
            )
        if not (0.0 <= self.concede_threshold <= 1.0):
            raise InvalidConfig(
                f"concede_threshold must be in [0, 1]; got {self.concede_threshold}"
            )
        if not (0.5 < self.confidence < 1.0):
            raise InvalidConfig(
                f"confidence must be in (0.5, 1); got {self.confidence}"
            )
        if self.record_every <= 0:
            raise InvalidConfig(f"record_every must be > 0; got {self.record_every}")
        if not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key must be bytes")


@dataclass(frozen=True)
class Argument:
    """A debater's claim with optional verifiable evidence weight.

    ``evidence`` is in ``[0, 1]``: ``1.0`` means a fully verifiable
    citation (a hash, a proof step, a code execution receipt) that
    the judge can check directly; ``0.0`` means pure rhetoric.  Used
    by the persuasion-aware scoring to split posterior shift into
    truthful vs manipulative components.

    ``meta`` is an arbitrary caller-supplied dict carried byte-for-
    byte through the transcript chain.
    """

    side: str
    text: str
    evidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in (SIDE_A, SIDE_B):
            raise InvalidMove(f"side must be {SIDE_A!r} or {SIDE_B!r}; got {self.side!r}")
        if not isinstance(self.text, str):
            raise InvalidMove("Argument.text must be a string")
        if not (0.0 <= self.evidence <= 1.0):
            raise InvalidMove(
                f"Argument.evidence must be in [0, 1]; got {self.evidence}"
            )


@dataclass(frozen=True)
class DebateMove:
    """A single move in a debate transcript.

    ``kind`` is one of :data:`KNOWN_MOVES`.  ``argument`` is required
    for argue/counter/answer; ``target_index`` is required for
    cross_examine (the transcript index being challenged); the rest
    are optional.
    """

    kind: str
    side: str
    round_index: int
    argument: Argument | None = None
    target_index: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_MOVES:
            raise InvalidMove(f"kind={self.kind!r} not in {KNOWN_MOVES}")
        if self.side not in (SIDE_A, SIDE_B, SIDE_TIE):
            raise InvalidMove(f"side={self.side!r} not in {{A,B,tie}}")
        if self.round_index < 0:
            raise InvalidMove(f"round_index must be >= 0; got {self.round_index}")
        if self.kind in (MOVE_ARGUE, MOVE_COUNTER, MOVE_ANSWER) and self.argument is None:
            raise InvalidMove(f"{self.kind} requires an Argument")
        if self.kind == MOVE_CROSS_EXAMINE and self.target_index is None:
            raise InvalidMove(f"{MOVE_CROSS_EXAMINE} requires a target_index")
        if self.target_index is not None and self.target_index < 0:
            raise InvalidMove(f"target_index must be >= 0; got {self.target_index}")


# A debater is a callable: (DebateSpec, transcript_so_far, side) -> DebateMove
Debater_fn = Callable[["DebateSpec", Sequence[DebateMove], str], DebateMove]

# A judge is a callable: (DebateSpec, transcript) -> dict {side: probability of winning}
Judge_fn = Callable[["DebateSpec", Sequence[DebateMove]], Mapping[str, float]]

# A persuasion model is a callable: (DebateSpec, transcript_so_far) -> dict {side: judge posterior}
Persuasion_fn = Callable[["DebateSpec", Sequence[DebateMove]], Mapping[str, float]]


@dataclass(frozen=True)
class DebateSpec:
    """The complete description of one debate.

    ``question`` is the topic.  ``claim_a`` and ``claim_b`` are the
    two competing positions.  ``debater_a`` and ``debater_b`` are
    move-emitting callables.  ``judge`` is the verdict-emitting
    callable.  ``persuasion_model`` is optional and required only
    for :data:`PROTOCOL_PERSUASION_AWARE`.  ``ground_truth`` is
    optional and used only for calibration / ECE; never consulted
    by the debate itself.  ``strategy_space`` is the per-side
    discrete strategy set used by ``Debater.empirical_payoff`` and
    ``Debater.nash_check``; defaults to ``("truthful", "obfuscate")``.

    ``judges_for_jury`` is required for :data:`PROTOCOL_JURY`: a
    sequence of (judge_callable, judge_accuracy) tuples that
    replace the singleton ``judge``.
    """

    question: str
    claim_a: str
    claim_b: str
    debater_a: Debater_fn
    debater_b: Debater_fn
    judge: Judge_fn
    persuasion_model: Persuasion_fn | None = None
    judges_for_jury: tuple[tuple[Judge_fn, float], ...] = field(default_factory=tuple)
    ground_truth: str | None = None
    strategy_space: tuple[str, ...] = ("truthful", "obfuscate")
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question:
            raise InvalidSpec("question must be non-empty")
        if not self.claim_a or not self.claim_b:
            raise InvalidSpec("claim_a and claim_b must be non-empty")
        if self.claim_a == self.claim_b:
            raise InvalidSpec("claim_a and claim_b must differ")
        if not callable(self.debater_a) or not callable(self.debater_b):
            raise InvalidSpec("debater_a and debater_b must be callables")
        if not callable(self.judge):
            raise InvalidSpec("judge must be callable")
        if self.persuasion_model is not None and not callable(self.persuasion_model):
            raise InvalidSpec("persuasion_model must be callable when given")
        if self.ground_truth is not None and self.ground_truth not in (SIDE_A, SIDE_B):
            raise InvalidSpec(
                f"ground_truth must be {SIDE_A!r} or {SIDE_B!r}; got {self.ground_truth!r}"
            )
        if len(set(self.strategy_space)) != len(self.strategy_space):
            raise InvalidSpec("strategy_space entries must be unique")
        if len(self.strategy_space) < 2:
            raise InvalidSpec("strategy_space must have >= 2 strategies")
        for j, acc in self.judges_for_jury:
            if not callable(j):
                raise InvalidSpec("each jury judge must be callable")
            if not (0.0 < acc < 1.0):
                raise InvalidSpec(
                    f"each judge accuracy must be in (0, 1); got {acc}"
                )


@dataclass(frozen=True)
class JuryConfig:
    """Per-debate jury configuration override.

    Used to pass through aggregation method (majority / weighted-log-
    odds / unanimity) and a tie-break preference into the jury
    protocol without mutating :class:`DebaterConfig`.
    """

    aggregation: str = AGG_MAJORITY
    tie_break: str = SIDE_TIE

    def __post_init__(self) -> None:
        if self.aggregation not in KNOWN_AGGREGATIONS:
            raise UnknownAggregation(
                f"aggregation={self.aggregation!r} not in {KNOWN_AGGREGATIONS}"
            )
        if self.tie_break not in (SIDE_A, SIDE_B, SIDE_TIE):
            raise InvalidConfig(
                f"tie_break must be {SIDE_A!r}, {SIDE_B!r}, or {SIDE_TIE!r}; got {self.tie_break!r}"
            )


@dataclass(frozen=True)
class CalibrationReport:
    """Empirical-calibration report for a judge over a stream of debates.

    ``ece`` is the binned Expected Calibration Error (Naeini-Cooper-
    Hauskrecht 2015); ``bins`` are the bin edges, ``acc_per_bin``
    the empirical accuracy per bin, ``conf_per_bin`` the mean
    predicted probability per bin, ``n_per_bin`` the count per bin.
    """

    n: int
    n_bins: int
    ece: float
    bins: tuple[float, ...]
    acc_per_bin: tuple[float, ...]
    conf_per_bin: tuple[float, ...]
    n_per_bin: tuple[int, ...]


@dataclass(frozen=True)
class PayoffMatrix:
    """The bimatrix payoff over each side's strategy_space.

    ``matrix_a[i][j]`` is side A's empirical win-rate when A plays
    strategy ``i`` and B plays strategy ``j``.  ``matrix_b`` is the
    complementary B payoff.  ``samples_per_cell`` is the Monte-Carlo
    count.
    """

    strategies: tuple[str, ...]
    matrix_a: tuple[tuple[float, ...], ...]
    matrix_b: tuple[tuple[float, ...], ...]
    samples_per_cell: int


@dataclass(frozen=True)
class NashResult:
    """A Nash equilibrium of the debate game.

    ``pi_a`` and ``pi_b`` are mixed strategies over each side's
    strategy_space.  ``value_a`` and ``value_b`` are the per-side
    payoffs under the profile.  ``nash_conv`` is the *exploitability*
    gap: ``max(max_dev_a − value_a, max_dev_b − value_b)``; zero
    iff the profile is a Nash equilibrium.  ``support_a`` and
    ``support_b`` are the strategy supports used.  ``method`` is
    the solver tag.
    """

    strategies: tuple[str, ...]
    pi_a: tuple[float, ...]
    pi_b: tuple[float, ...]
    value_a: float
    value_b: float
    nash_conv: float
    support_a: tuple[str, ...]
    support_b: tuple[str, ...]
    method: str


@dataclass(frozen=True)
class DebateReport:
    """The result of a single :meth:`Debater.run` call.

    ``winner`` is in ``{SIDE_A, SIDE_B, SIDE_TIE}``.  ``win_prob_hat``
    is the (Monte-Carlo or jury-aggregated) probability that the
    winner is correct.  ``transcript`` is the full move sequence
    (replay-verifiable).  ``persuasion_trace`` is the per-round
    judge posterior shift (only populated for persuasion-aware
    protocols).  ``chain_head`` is the SHA-256 hash of the
    transcript chain.  ``protocol`` echoes ``DebaterConfig.protocol``.
    """

    winner: str
    win_prob_hat: float
    transcript: tuple[DebateMove, ...]
    persuasion_trace: tuple[tuple[float, float], ...]
    truthful_components: tuple[float, ...]
    manipulative_components: tuple[float, ...]
    judge_votes: tuple[tuple[str, float], ...]
    rounds_used: int
    converged: bool
    protocol: str
    chain_head: str
    spec_question: str


@dataclass(frozen=True)
class DebaterCertificate:
    """Anytime-valid PAC certificate over the win-rate of the truth side.

    ``win_prob_hat`` is the empirical estimate; ``hoeffding_lcb`` is
    the Hoeffding LCB on the underlying win probability;
    ``bernstein_lcb`` is the empirical-Bernstein (Maurer-Pontil 2009)
    refinement; ``condorcet_lcb`` is the Boland-bound LCB
    specifically for jury aggregations; ``persuasion_penalty`` is
    the Khan-2024 manipulative-component sum; ``nash_conv`` is the
    exploitability of the debaters' realised profile (0 = Nash);
    ``calibration_ece`` is the judge's binned ECE if calibration
    data was logged.
    """

    n: int
    delta: float
    win_prob_hat: float
    hoeffding_lcb: float
    bernstein_lcb: float
    condorcet_lcb: float
    persuasion_penalty: float
    nash_conv: float
    calibration_ece: float | None
    protocol: str
    chain_head: str


@dataclass(frozen=True)
class AnytimeCertificate:
    """Howard-Ramdas-McAuliffe-Sekhon (HRMS) anytime-valid confidence sequence.

    Valid at every ``n`` for *any* data-dependent stopping rule
    (Ville's inequality).  ``radius`` is the symmetric half-width.
    """

    n: int
    delta: float
    win_prob_hat: float
    lcb: float
    ucb: float
    radius: float


# ---------------------------------------------------------------------------
# Helper functions (pure)
# ---------------------------------------------------------------------------


_GENESIS_PREFIX = b"agi.debater.v1\x00"


def debater_ledger_root(secret_key: bytes | None = None) -> str:
    """Return the deterministic genesis hash for the Debater chain."""
    seed = _GENESIS_PREFIX + (secret_key or b"")
    return hashlib.sha256(seed).hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    """Canonical JSON encoding (sorted keys, repr floats)."""

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
        if isinstance(o, Argument):
            return _q({
                "side": o.side, "text": o.text, "evidence": o.evidence, "meta": dict(o.meta)
            })
        if isinstance(o, DebateMove):
            base = {
                "kind": o.kind, "side": o.side, "round_index": o.round_index,
                "meta": dict(o.meta),
            }
            if o.argument is not None:
                base["argument"] = o.argument
            if o.target_index is not None:
                base["target_index"] = o.target_index
            return _q(base)
        return o

    return json.dumps(_q(payload), sort_keys=True, separators=(",", ":")).encode()


def _hash_entry(parent: str, payload: dict[str, Any], hmac_key: bytes | None = None) -> str:
    body = _canonical(payload)
    block = parent.encode() + b"|" + body
    if hmac_key:
        return hmac.new(hmac_key, block, hashlib.sha256).hexdigest()
    return hashlib.sha256(block).hexdigest()


def debater_hoeffding_lcb(p_hat: float, n: int, delta: float) -> float:
    """One-sided Hoeffding lower confidence bound on a Bernoulli mean.

    ``LCB = p_hat − sqrt(ln(1/δ) / (2n))``, clipped to ``[0, 1]``.
    Requires ``n >= 1`` and ``0 < δ < 1``.  Distribution-free.
    """
    if n < 1:
        raise InsufficientData("Hoeffding LCB requires n >= 1")
    if not (0.0 < delta < 1.0):
        raise InvalidConfig(f"delta must be in (0, 1); got {delta}")
    if not (0.0 <= p_hat <= 1.0):
        raise InvalidConfig(f"p_hat must be in [0, 1]; got {p_hat}")
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return max(0.0, p_hat - radius)


def debater_bernstein_lcb(
    p_hat: float, var_hat: float, n: int, delta: float, *, b: float = 1.0
) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein LCB on a bounded mean.

    Tighter than Hoeffding whenever the empirical variance is small.
    ``LCB = p_hat − sqrt(2 var_hat ln(2/δ) / n) − 7 b ln(2/δ) / (3(n − 1))``.
    """
    if n < 2:
        raise InsufficientData("Empirical-Bernstein LCB requires n >= 2")
    if not (0.0 < delta < 1.0):
        raise InvalidConfig(f"delta must be in (0, 1); got {delta}")
    if var_hat < 0:
        raise InvalidConfig("var_hat must be non-negative")
    if b <= 0:
        raise InvalidConfig("b must be positive")
    term_var = math.sqrt(2.0 * var_hat * math.log(2.0 / delta) / n)
    term_b = 7.0 * b * math.log(2.0 / delta) / (3.0 * (n - 1))
    return max(0.0, p_hat - term_var - term_b)


def debater_condorcet_lcb(p: float, m: int, delta: float) -> float:
    """Boland 1989 LCB on the probability of majority correctness.

    For ``M`` iid Bernoulli judges with per-judge accuracy ``p``, the
    probability that the majority is correct is

        ``P(maj correct) = sum_{k > M/2} C(M, k) p^k (1−p)^{M−k}``.

    This function returns ``LCB = exact_p − sqrt(ln(1/δ) / (2M))``,
    where the Hoeffding radius bounds the deviation between empirical
    and true per-judge accuracy in the worst case.  Provably tight
    for ``p > 0.5``; uninformative (0) for ``p <= 0.5``.
    """
    if m < 1:
        raise InsufficientData("Condorcet LCB requires M >= 1")
    if not (0.0 < p < 1.0):
        raise InvalidConfig(f"p must be in (0, 1); got {p}")
    if not (0.0 < delta < 1.0):
        raise InvalidConfig(f"delta must be in (0, 1); got {delta}")
    # exact majority correctness
    # use log space to avoid overflow at large M
    threshold = m // 2 + 1
    total = 0.0
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    # log-binomial via log-gamma
    for k in range(threshold, m + 1):
        log_bin = (
            math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
        )
        log_term = log_bin + k * log_p + (m - k) * log_q
        total += math.exp(log_term)
    # finite-sample Hoeffding correction for uncertainty in p
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * m))
    return max(0.0, min(1.0, total - radius))


def debater_hrms_radius(n: int, delta: float) -> float:
    """HRMS-style anytime-valid radius for a [0, 1] mean.

    A simple closed-form radius taken from Howard-Ramdas-McAuliffe-
    Sekhon (Annals of Statistics 2021), specialised to a constant
    ``[0, 1]`` envelope:

        ``r_n = sqrt( (2 / n) * (log(log(2n) / δ) + log(2)) )``.

    Valid at every ``n >= 2`` simultaneously: Ville's inequality
    guarantees coverage under any stopping rule.
    """
    if n < 2:
        raise InsufficientData("HRMS radius requires n >= 2")
    if not (0.0 < delta < 1.0):
        raise InvalidConfig(f"delta must be in (0, 1); got {delta}")
    inner = max(1.0, math.log(2.0 * n))
    return math.sqrt((2.0 / n) * (math.log(inner / delta) + math.log(2.0)))


def debater_jury_majority(
    votes: Sequence[tuple[str, float]], tie_break: str = SIDE_TIE
) -> str:
    """Majority verdict over ``votes`` (side, confidence) tuples.

    Returns ``SIDE_A``, ``SIDE_B``, or ``tie_break`` on a tie.  Each
    vote contributes 1 regardless of confidence (use
    :func:`debater_jury_log_odds` for confidence-weighted aggregation).
    """
    a = sum(1 for side, _ in votes if side == SIDE_A)
    b = sum(1 for side, _ in votes if side == SIDE_B)
    if a > b:
        return SIDE_A
    if b > a:
        return SIDE_B
    return tie_break


def debater_jury_log_odds(
    votes: Sequence[tuple[str, float]],
    weights: Sequence[float] | None = None,
    tie_break: str = SIDE_TIE,
    *,
    eps: float = 1e-9,
) -> str:
    """Weighted log-odds verdict (Grofman-Owen-Feld 1983 optimal aggregation).

    Each judge ``i`` casts a vote with confidence ``c_i`` interpreted as
    ``P(correct)``; the optimal aggregation under independence is

        ``argmax_{side} Σ_i w_i · log(c_i / (1 − c_i)) · 1[vote_i = side]``.
    """
    if weights is None:
        weights = [1.0] * len(votes)
    if len(weights) != len(votes):
        raise InvalidConfig("len(weights) must equal len(votes)")
    score_a = 0.0
    score_b = 0.0
    for (side, conf), w in zip(votes, weights):
        c = min(1.0 - eps, max(eps, conf))
        odds = math.log(c / (1.0 - c))
        if side == SIDE_A:
            score_a += w * odds
        elif side == SIDE_B:
            score_b += w * odds
    if score_a > score_b:
        return SIDE_A
    if score_b > score_a:
        return SIDE_B
    return tie_break


def debater_bayes_posterior_shift(
    prior: Mapping[str, float], posterior: Mapping[str, float]
) -> float:
    """Total-variation distance between two pmfs over the same support.

    Used as the "persuasion delta" between two consecutive judge
    posteriors.  Defined as ``½ Σ_x |p(x) − q(x)|``; in ``[0, 1]``.
    """
    keys = set(prior) | set(posterior)
    return 0.5 * sum(abs(prior.get(k, 0.0) - posterior.get(k, 0.0)) for k in keys)


def debater_persuasion_decomposition(
    delta: float, evidence: float, *, threshold: float = 0.1
) -> tuple[float, float]:
    """Decompose a posterior shift into truthful and manipulative components.

    Khan-Hughes 2024 §3.2: a shift backed by verifiable evidence
    ``e >= threshold`` is *truthful*; the residual is *manipulative*.

    Returns ``(truthful_delta, manipulative_delta)`` with
    ``truthful_delta + manipulative_delta == delta``.
    """
    if delta < 0:
        raise InvalidConfig("delta must be non-negative")
    if not (0.0 <= evidence <= 1.0):
        raise InvalidConfig("evidence must be in [0, 1]")
    if evidence >= threshold:
        truthful = delta * evidence
        manipulative = delta * (1.0 - evidence)
    else:
        truthful = 0.0
        manipulative = delta
    return truthful, manipulative


def debater_calibration_ece(
    confidences: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> CalibrationReport:
    """Binned Expected Calibration Error (Naeini-Cooper-Hauskrecht 2015).

    ``outcomes[i] in {0, 1}`` is the realised correctness; the
    estimator partitions the confidence axis into ``n_bins`` equal-
    width buckets, averages accuracy / confidence per bucket, and
    returns the weighted-by-bin-count L1 distance.
    """
    if len(confidences) != len(outcomes):
        raise InvalidConfig("len(confidences) must equal len(outcomes)")
    if n_bins < 1:
        raise InvalidConfig("n_bins must be >= 1")
    n = len(confidences)
    if n == 0:
        raise InsufficientData("ECE requires at least one observation")
    bins = tuple(i / n_bins for i in range(n_bins + 1))
    acc_per_bin: list[float] = []
    conf_per_bin: list[float] = []
    n_per_bin: list[int] = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin: list[int] = []
        for i, c in enumerate(confidences):
            if (c > lo or (b == 0 and c >= lo)) and c <= hi:
                in_bin.append(i)
        cnt = len(in_bin)
        n_per_bin.append(cnt)
        if cnt == 0:
            acc_per_bin.append(0.0)
            conf_per_bin.append(0.0)
            continue
        acc = sum(outcomes[i] for i in in_bin) / cnt
        conf = sum(confidences[i] for i in in_bin) / cnt
        acc_per_bin.append(acc)
        conf_per_bin.append(conf)
        ece += (cnt / n) * abs(acc - conf)
    return CalibrationReport(
        n=n,
        n_bins=n_bins,
        ece=ece,
        bins=bins,
        acc_per_bin=tuple(acc_per_bin),
        conf_per_bin=tuple(conf_per_bin),
        n_per_bin=tuple(n_per_bin),
    )


def debater_payoff_nash_2x2(
    a_payoff: Sequence[Sequence[float]],
    b_payoff: Sequence[Sequence[float]],
) -> NashResult:
    """Closed-form mixed-Nash for a 2×2 bimatrix (Lemke-Howson 1964).

    For zero-sum or non-zero-sum 2×2 games, returns the mixed
    equilibrium via the standard indifference condition.  Falls back
    to a pure-strategy Nash when one exists.
    """
    a = [list(row) for row in a_payoff]
    b = [list(row) for row in b_payoff]
    if len(a) != 2 or len(b) != 2 or any(len(r) != 2 for r in a) or any(len(r) != 2 for r in b):
        raise InvalidConfig("payoff matrices must be 2x2")
    strategies = ("s0", "s1")
    # Look for pure-strategy Nash equilibria first
    pures: list[tuple[int, int]] = []
    for i in range(2):
        for j in range(2):
            a_dev = max(a[k][j] for k in range(2))
            b_dev = max(b[i][k] for k in range(2))
            if a[i][j] >= a_dev - 1e-12 and b[i][j] >= b_dev - 1e-12:
                pures.append((i, j))
    if pures:
        i, j = pures[0]
        pi_a = (1.0, 0.0) if i == 0 else (0.0, 1.0)
        pi_b = (1.0, 0.0) if j == 0 else (0.0, 1.0)
        return NashResult(
            strategies=strategies,
            pi_a=pi_a, pi_b=pi_b,
            value_a=a[i][j], value_b=b[i][j],
            nash_conv=0.0,
            support_a=(strategies[i],), support_b=(strategies[j],),
            method="pure",
        )
    # Mixed: solve indifference for opponent
    # For A: q makes A indifferent: a[0][0] q + a[0][1] (1-q) = a[1][0] q + a[1][1] (1-q)
    # => (a[0][0] - a[0][1] - a[1][0] + a[1][1]) q = a[1][1] - a[0][1]
    denom_q = (a[0][0] - a[0][1] - a[1][0] + a[1][1])
    denom_p = (b[0][0] - b[1][0] - b[0][1] + b[1][1])
    if abs(denom_q) < 1e-12 or abs(denom_p) < 1e-12:
        # Degenerate; fall back to support enumeration
        return debater_support_enumeration(a_payoff, b_payoff, strategies=strategies)
    q = (a[1][1] - a[0][1]) / denom_q
    p = (b[1][1] - b[1][0]) / denom_p
    p = min(1.0, max(0.0, p))
    q = min(1.0, max(0.0, q))
    value_a = a[0][0] * p * q + a[0][1] * p * (1 - q) + a[1][0] * (1 - p) * q + a[1][1] * (1 - p) * (1 - q)
    value_b = b[0][0] * p * q + b[0][1] * p * (1 - q) + b[1][0] * (1 - p) * q + b[1][1] * (1 - p) * (1 - q)
    # NashConv: best-response gap
    dev_a = max(
        a[0][0] * q + a[0][1] * (1 - q),
        a[1][0] * q + a[1][1] * (1 - q),
    )
    dev_b = max(
        b[0][0] * p + b[1][0] * (1 - p),
        b[0][1] * p + b[1][1] * (1 - p),
    )
    nash_conv = max(dev_a - value_a, dev_b - value_b)
    return NashResult(
        strategies=strategies,
        pi_a=(p, 1.0 - p),
        pi_b=(q, 1.0 - q),
        value_a=value_a,
        value_b=value_b,
        nash_conv=max(0.0, nash_conv),
        support_a=strategies,
        support_b=strategies,
        method="mixed_2x2",
    )


def debater_support_enumeration(
    a_payoff: Sequence[Sequence[float]],
    b_payoff: Sequence[Sequence[float]],
    *,
    strategies: Sequence[str] | None = None,
) -> NashResult:
    """Support enumeration Nash for an n×n bimatrix (Audet-Hansen 2001).

    Enumerates non-empty supports up to size ``min(n, n)`` and tries
    to solve the indifference system; returns the first Nash found,
    falling back to the best-response (NashConv-minimising) profile
    of the supports that admit a solution.

    Complexity: ``O(2^n · n^3)``; designed for ``n <= 6``.
    """
    a = [list(row) for row in a_payoff]
    b = [list(row) for row in b_payoff]
    n = len(a)
    m = len(a[0]) if a else 0
    if any(len(r) != m for r in a) or any(len(r) != m for r in b):
        raise InvalidConfig("payoff matrices must be rectangular")
    if len(b) != n:
        raise InvalidConfig("b_payoff must have same #rows as a_payoff")
    if strategies is None:
        strategies = tuple(f"s{i}" for i in range(max(n, m)))
    # Map supports → mixed strategies via indifference
    def _solve_support(support_a: list[int], support_b: list[int]) -> tuple[list[float], list[float]] | None:
        # Side A is indifferent across support_a:  for all i, j in support_a:
        #   Σ_k b[i][k] q[k] = Σ_k b[j][k] q[k]  (A's indifference is over B's strategy q)
        # Wait — A's payoff under q is Σ_k a[i][k] q[k].  Indifference: same across i in support_a.
        # Build linear system on q.
        sa = sorted(set(support_a))
        sb = sorted(set(support_b))
        ka = len(sa)
        kb = len(sb)
        if ka == 0 or kb == 0:
            return None
        # Solve for q on support_b: |support_b| unknowns, |support_a|-1 indifference eqns + 1 normalisation
        # Each indifference eqn: a[sa[0]][k] q[k] − a[sa[i]][k] q[k] = 0 for k in sb
        rows: list[list[float]] = []
        rhs: list[float] = []
        for i in range(1, ka):
            row = [a[sa[0]][k] - a[sa[i]][k] for k in sb]
            rows.append(row)
            rhs.append(0.0)
        # Normalisation: Σ_k q[k] = 1
        rows.append([1.0] * kb)
        rhs.append(1.0)
        # Stack: must have rows >= kb
        if len(rows) < kb:
            # Underdetermined; pick the simplex centroid as a hint and check
            q = [1.0 / kb] * kb
        else:
            # Solve via Gauss-Jordan, accepting first feasible non-negative solution
            q = _gauss_solve(rows, rhs)
            if q is None:
                return None
        if any(qq < -1e-9 for qq in q):
            return None
        q = [max(0.0, x) for x in q]
        s = sum(q)
        if s <= 0:
            return None
        q = [x / s for x in q]
        # Same for p
        rows2: list[list[float]] = []
        rhs2: list[float] = []
        for j in range(1, kb):
            row = [b[i][sb[0]] - b[i][sb[j]] for i in sa]
            rows2.append(row)
            rhs2.append(0.0)
        rows2.append([1.0] * ka)
        rhs2.append(1.0)
        if len(rows2) < ka:
            p = [1.0 / ka] * ka
        else:
            p = _gauss_solve(rows2, rhs2)
            if p is None:
                return None
        if any(pp < -1e-9 for pp in p):
            return None
        p = [max(0.0, x) for x in p]
        sp = sum(p)
        if sp <= 0:
            return None
        p = [x / sp for x in p]
        # Embed in full vectors
        pi_a = [0.0] * n
        for i, idx in enumerate(sa):
            pi_a[idx] = p[i]
        pi_b = [0.0] * m
        for j, idx in enumerate(sb):
            pi_b[idx] = q[j]
        # Verify off-support best-response: every i ∉ support_a, payoff <= indifferent value
        v_a_in = sum(a[sa[0]][k] * pi_b[k] for k in range(m))
        for i in range(n):
            if i in sa:
                continue
            if sum(a[i][k] * pi_b[k] for k in range(m)) > v_a_in + 1e-7:
                return None
        v_b_in = sum(b[sa[0]][k] * pi_a[sa[0]] for k in range(m))  # placeholder
        v_b_in = sum(b[i_row][sb[0]] * pi_a[i_row] for i_row in range(n))
        for j in range(m):
            if j in sb:
                continue
            if sum(b[i_row][j] * pi_a[i_row] for i_row in range(n)) > v_b_in + 1e-7:
                return None
        return pi_a, pi_b

    best: tuple[list[float], list[float], float, float, float, list[int], list[int]] | None = None
    # Enumerate supports up to size 3 (sufficient for most coordination-game uses)
    max_supp = min(3, n, m)
    for size_a in range(1, max_supp + 1):
        for size_b in range(1, max_supp + 1):
            for sa in _subsets(n, size_a):
                for sb in _subsets(m, size_b):
                    sol = _solve_support(list(sa), list(sb))
                    if sol is None:
                        continue
                    pi_a, pi_b = sol
                    v_a = sum(a[i][k] * pi_a[i] * pi_b[k] for i in range(n) for k in range(m))
                    v_b = sum(b[i][k] * pi_a[i] * pi_b[k] for i in range(n) for k in range(m))
                    dev_a = max(sum(a[i][k] * pi_b[k] for k in range(m)) for i in range(n))
                    dev_b = max(sum(b[i][k] * pi_a[i] for i in range(n)) for k in range(m))
                    nash_conv = max(dev_a - v_a, dev_b - v_b)
                    if best is None or nash_conv < best[4]:
                        best = (pi_a, pi_b, v_a, v_b, max(0.0, nash_conv), list(sa), list(sb))
    if best is None:
        # Pure uniform mix fallback
        pi_a = [1.0 / n] * n
        pi_b = [1.0 / m] * m
        v_a = sum(a[i][k] * pi_a[i] * pi_b[k] for i in range(n) for k in range(m))
        v_b = sum(b[i][k] * pi_a[i] * pi_b[k] for i in range(n) for k in range(m))
        dev_a = max(sum(a[i][k] * pi_b[k] for k in range(m)) for i in range(n))
        dev_b = max(sum(b[i][k] * pi_a[i] for i in range(n)) for k in range(m))
        return NashResult(
            strategies=tuple(strategies[: max(n, m)]),
            pi_a=tuple(pi_a), pi_b=tuple(pi_b),
            value_a=v_a, value_b=v_b,
            nash_conv=max(0.0, max(dev_a - v_a, dev_b - v_b)),
            support_a=tuple(strategies[: n]),
            support_b=tuple(strategies[: m]),
            method="uniform_fallback",
        )
    pi_a, pi_b, v_a, v_b, nash_conv, sa, sb = best
    return NashResult(
        strategies=tuple(strategies[: max(n, m)]),
        pi_a=tuple(pi_a), pi_b=tuple(pi_b),
        value_a=v_a, value_b=v_b,
        nash_conv=nash_conv,
        support_a=tuple(strategies[i] for i in sa),
        support_b=tuple(strategies[j] for j in sb),
        method="support_enumeration",
    )


def _subsets(n: int, k: int) -> Iterable[tuple[int, ...]]:
    """Enumerate ``k``-subsets of ``range(n)`` lexicographically."""
    if k == 0:
        yield ()
        return
    if k > n:
        return
    # iterative combinations
    indices = list(range(k))
    while True:
        yield tuple(indices)
        for i in range(k - 1, -1, -1):
            if indices[i] != i + n - k:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1


def _gauss_solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b via Gauss-Jordan with partial pivoting.

    Returns ``None`` if the system is singular or rank-deficient.
    For overdetermined systems, uses the leading-square minor and
    verifies the residual is small.
    """
    m = len(A)
    if m == 0:
        return None
    n = len(A[0])
    if any(len(row) != n for row in A) or len(b) != m:
        return None
    # Compress to square if overdetermined: keep first n independent rows
    aug = [row + [bi] for row, bi in zip(A, b)]
    # Gauss-Jordan
    row = 0
    pivots: list[int] = []
    for col in range(n):
        # find pivot
        pivot = -1
        best_val = 1e-12
        for r in range(row, m):
            v = abs(aug[r][col])
            if v > best_val:
                best_val = v
                pivot = r
        if pivot < 0:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        # normalise pivot row
        pv = aug[row][col]
        aug[row] = [v / pv for v in aug[row]]
        # eliminate
        for r in range(m):
            if r == row:
                continue
            factor = aug[r][col]
            if abs(factor) > 0:
                aug[r] = [aug[r][i] - factor * aug[row][i] for i in range(n + 1)]
        pivots.append(col)
        row += 1
        if row == m:
            break
    if len(pivots) < n:
        return None
    x = [0.0] * n
    for r, col in enumerate(pivots):
        x[col] = aug[r][n]
    # Check residual on any leftover rows
    for r in range(len(pivots), m):
        residual = sum(A[r][i] * x[i] for i in range(n)) - b[r]
        if abs(residual) > 1e-6:
            return None
    return x


# ---------------------------------------------------------------------------
# Default debaters and judges (for testing / demos / fallbacks)
# ---------------------------------------------------------------------------


def make_constant_debater(
    arguments_per_round: Sequence[Argument],
) -> Debater_fn:
    """Wrap a pre-canned argument sequence into a Debater callable.

    Useful for deterministic tests; cycles through ``arguments_per_round``.
    """
    args = list(arguments_per_round)
    if not args:
        raise InvalidConfig("arguments_per_round must be non-empty")

    def _fn(spec: DebateSpec, transcript: Sequence[DebateMove], side: str) -> DebateMove:
        idx = sum(1 for m in transcript if m.side == side and m.kind in (MOVE_ARGUE, MOVE_COUNTER, MOVE_ANSWER))
        arg = args[idx % len(args)]
        kind = MOVE_ARGUE if idx == 0 else MOVE_COUNTER
        round_index = idx
        # Construct argument with the called side baked in (the user may have left side=A by mistake)
        bound = Argument(side=side, text=arg.text, evidence=arg.evidence, meta=dict(arg.meta))
        return DebateMove(kind=kind, side=side, round_index=round_index, argument=bound)

    return _fn


def make_calibrated_judge(
    p_truth: float,
    truth_side: str,
    *,
    seed: int = 0,
) -> Judge_fn:
    """A Bernoulli judge: outputs ``truth_side`` with probability ``p_truth``.

    Pure / stateless: the verdict is a deterministic function of the
    transcript + ``seed``, so two judges with identical ``seed`` and
    identical transcripts always return the same verdict.  Across a
    population of seeds the per-judge accuracy approaches ``p_truth``.
    """
    if truth_side not in (SIDE_A, SIDE_B):
        raise InvalidSpec(f"truth_side must be {SIDE_A!r} or {SIDE_B!r}")
    if not (0.0 < p_truth < 1.0):
        raise InvalidConfig("p_truth must be in (0, 1)")
    seed_bytes = int(seed).to_bytes(8, "big", signed=False) if seed >= 0 else \
        (-int(seed)).to_bytes(8, "big", signed=False) + b"\x01"

    def _fn(spec: DebateSpec, transcript: Sequence[DebateMove]) -> Mapping[str, float]:
        h = hashlib.sha256()
        h.update(seed_bytes)
        # Include the spec question + claims so identical transcripts on
        # different questions still diverge.
        h.update(_canonical({
            "q": spec.question, "ca": spec.claim_a, "cb": spec.claim_b,
        }))
        for m in transcript:
            h.update(_canonical({"kind": m.kind, "side": m.side, "round": m.round_index,
                                  "arg": m.argument if m.argument else "",
                                  "tgt": m.target_index if m.target_index is not None else -1,
                                  "meta": dict(m.meta)}))
        digest = h.digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        pick_truth = u < p_truth
        if pick_truth:
            winner = truth_side
        else:
            winner = SIDE_B if truth_side == SIDE_A else SIDE_A
        return {winner: 1.0, (SIDE_B if winner == SIDE_A else SIDE_A): 0.0}

    return _fn


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


EventPublisher = Callable[[str, dict[str, Any]], None]


@dataclass
class _State:
    """Mutable internal state used by :class:`Debater`."""

    rng: random.Random
    reports: list[DebateReport] = field(default_factory=list)
    judge_confidences: list[float] = field(default_factory=list)
    judge_outcomes: list[int] = field(default_factory=list)  # 1 if winner == ground_truth
    chain_head: str = ""


# ---------------------------------------------------------------------------
# Debater
# ---------------------------------------------------------------------------


class Debater:
    """Multi-agent debate as a runtime primitive.

    Threadsafe at the API surface: a single re-entrant lock guards
    every mutation of the report log and chain head.
    """

    def __init__(
        self,
        config: DebaterConfig | None = None,
        *,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config or DebaterConfig()
        self._publisher = publisher
        self._lock = threading.RLock()
        head = debater_ledger_root(self.config.hmac_key if self.config.hmac_key else None)
        self._state = _State(rng=random.Random(self.config.seed), chain_head=head)
        self._publish(DEBATER_STARTED, {
            "protocol": self.config.protocol,
            "judge_model": self.config.judge_model,
            "head": head,
        })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chain_head(self) -> str:
        with self._lock:
            return self._state.chain_head

    @property
    def n_reports(self) -> int:
        with self._lock:
            return len(self._state.reports)

    # ------------------------------------------------------------------
    # Publishing + chain
    # ------------------------------------------------------------------

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher(kind, payload)
        except Exception:
            # Publisher exceptions never break the primitive
            pass

    def _advance(self, payload: dict[str, Any]) -> str:
        key = self.config.hmac_key if self.config.hmac_key else None
        self._state.chain_head = _hash_entry(self._state.chain_head, payload, key)
        return self._state.chain_head

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self, spec: DebateSpec, *, depth: int | None = None) -> DebateReport:
        """Execute one debate following ``self.config.protocol``."""
        if not isinstance(spec, DebateSpec):
            raise InvalidSpec("spec must be a DebateSpec")
        d = depth if depth is not None else self.config.max_depth
        if d <= 0:
            raise InvalidConfig("depth must be > 0")
        with self._lock:
            self._publish(DEBATER_OPENED, {
                "protocol": self.config.protocol, "question": spec.question,
                "claim_a": spec.claim_a, "claim_b": spec.claim_b, "depth": d,
            })
            head = self._advance({"op": "open", "question": spec.question,
                                  "claim_a": spec.claim_a, "claim_b": spec.claim_b,
                                  "protocol": self.config.protocol})
            if self.config.protocol == PROTOCOL_TWO_PLAYER:
                report = self._run_two_player(spec, d)
            elif self.config.protocol == PROTOCOL_CROSS_EXAM:
                report = self._run_cross_exam(spec, d)
            elif self.config.protocol == PROTOCOL_DOUBLY_EFFICIENT:
                report = self._run_doubly_efficient(spec, d)
            elif self.config.protocol == PROTOCOL_MARKET_MAKER:
                report = self._run_market_maker(spec, d)
            elif self.config.protocol == PROTOCOL_JURY:
                report = self._run_jury(spec, d)
            elif self.config.protocol == PROTOCOL_PERSUASION_AWARE:
                report = self._run_persuasion_aware(spec, d)
            else:
                raise UnknownProtocol(f"protocol={self.config.protocol!r}")
            self._state.reports.append(report)
            # Calibration tracking
            if spec.ground_truth is not None and report.winner in (SIDE_A, SIDE_B):
                self._state.judge_confidences.append(report.win_prob_hat)
                self._state.judge_outcomes.append(
                    1 if report.winner == spec.ground_truth else 0
                )
            head = self._advance({"op": "close", "winner": report.winner,
                                  "n_rounds": report.rounds_used, "head": report.chain_head})
            self._publish(DEBATER_CLOSED, {
                "winner": report.winner, "rounds": report.rounds_used,
                "head": self._state.chain_head,
            })
            return report

    # ------------------------------------------------------------------
    # Protocol: two-player debate (Irving 2018)
    # ------------------------------------------------------------------

    def _run_two_player(self, spec: DebateSpec, depth: int) -> DebateReport:
        transcript: list[DebateMove] = []
        local_head = self._state.chain_head
        rounds_used = 0
        converged = False
        for r in range(self.config.max_rounds):
            # A argues
            move_a = self._invoke_debater(spec.debater_a, spec, transcript, SIDE_A)
            self._validate_move(move_a, SIDE_A, r)
            transcript.append(move_a)
            local_head = _hash_entry(local_head, {"op": "argue", "move": move_a},
                                      self.config.hmac_key if self.config.hmac_key else None)
            self._publish(DEBATER_ARGUED, {"side": SIDE_A, "round": r, "head": local_head})
            # B counters
            move_b = self._invoke_debater(spec.debater_b, spec, transcript, SIDE_B)
            self._validate_move(move_b, SIDE_B, r)
            transcript.append(move_b)
            local_head = _hash_entry(local_head, {"op": "counter", "move": move_b},
                                      self.config.hmac_key if self.config.hmac_key else None)
            self._publish(DEBATER_COUNTERED, {"side": SIDE_B, "round": r, "head": local_head})
            rounds_used = r + 1
            # Concede check: if either side's argument has effectively conceded (very low confidence),
            # short-circuit.  Default: never concede unless meta marks it.
            if (move_a.meta.get("concede") is True) or (move_b.meta.get("concede") is True):
                converged = True
                break
        # Judge
        verdict = self._poll_judge(spec.judge, spec, transcript)
        winner = self._winner_from_verdict(verdict)
        win_prob = float(verdict.get(winner, 0.5))
        verdict_move = DebateMove(kind=MOVE_VERDICT, side=winner, round_index=rounds_used)
        transcript.append(verdict_move)
        local_head = _hash_entry(local_head, {"op": "verdict", "move": verdict_move},
                                  self.config.hmac_key if self.config.hmac_key else None)
        self._publish(DEBATER_VERDICT, {"winner": winner, "head": local_head})
        return DebateReport(
            winner=winner,
            win_prob_hat=win_prob,
            transcript=tuple(transcript),
            persuasion_trace=(),
            truthful_components=(),
            manipulative_components=(),
            judge_votes=((winner, win_prob),),
            rounds_used=rounds_used,
            converged=converged,
            protocol=PROTOCOL_TWO_PLAYER,
            chain_head=local_head,
            spec_question=spec.question,
        )

    # ------------------------------------------------------------------
    # Protocol: cross-examination debate (Barnes-Christiano 2020)
    # ------------------------------------------------------------------

    def _run_cross_exam(self, spec: DebateSpec, depth: int) -> DebateReport:
        transcript: list[DebateMove] = []
        local_head = self._state.chain_head
        rounds_used = 0
        for r in range(self.config.max_rounds):
            move_a = self._invoke_debater(spec.debater_a, spec, transcript, SIDE_A)
            self._validate_move(move_a, SIDE_A, r)
            transcript.append(move_a)
            local_head = _hash_entry(local_head, {"op": move_a.kind, "move": move_a},
                                      self.config.hmac_key if self.config.hmac_key else None)
            if move_a.kind == MOVE_CROSS_EXAMINE:
                self._publish(DEBATER_CROSS_EXAMINED, {
                    "side": SIDE_A, "round": r, "target": move_a.target_index, "head": local_head,
                })
                # Force B to answer the cross-examination
                answer_b = self._invoke_debater(spec.debater_b, spec, transcript, SIDE_B)
                self._validate_move(answer_b, SIDE_B, r)
                transcript.append(answer_b)
                local_head = _hash_entry(local_head, {"op": answer_b.kind, "move": answer_b},
                                          self.config.hmac_key if self.config.hmac_key else None)
            else:
                self._publish(DEBATER_ARGUED, {"side": SIDE_A, "round": r, "head": local_head})
            move_b = self._invoke_debater(spec.debater_b, spec, transcript, SIDE_B)
            self._validate_move(move_b, SIDE_B, r)
            transcript.append(move_b)
            local_head = _hash_entry(local_head, {"op": move_b.kind, "move": move_b},
                                      self.config.hmac_key if self.config.hmac_key else None)
            if move_b.kind == MOVE_CROSS_EXAMINE:
                self._publish(DEBATER_CROSS_EXAMINED, {
                    "side": SIDE_B, "round": r, "target": move_b.target_index, "head": local_head,
                })
                answer_a = self._invoke_debater(spec.debater_a, spec, transcript, SIDE_A)
                self._validate_move(answer_a, SIDE_A, r)
                transcript.append(answer_a)
                local_head = _hash_entry(local_head, {"op": answer_a.kind, "move": answer_a},
                                          self.config.hmac_key if self.config.hmac_key else None)
            else:
                self._publish(DEBATER_COUNTERED, {"side": SIDE_B, "round": r, "head": local_head})
            rounds_used = r + 1
            if (move_a.meta.get("concede") is True) or (move_b.meta.get("concede") is True):
                break
        verdict = self._poll_judge(spec.judge, spec, transcript)
        winner = self._winner_from_verdict(verdict)
        win_prob = float(verdict.get(winner, 0.5))
        verdict_move = DebateMove(kind=MOVE_VERDICT, side=winner, round_index=rounds_used)
        transcript.append(verdict_move)
        local_head = _hash_entry(local_head, {"op": "verdict", "move": verdict_move},
                                  self.config.hmac_key if self.config.hmac_key else None)
        return DebateReport(
            winner=winner, win_prob_hat=win_prob,
            transcript=tuple(transcript),
            persuasion_trace=(), truthful_components=(), manipulative_components=(),
            judge_votes=((winner, win_prob),),
            rounds_used=rounds_used, converged=False,
            protocol=PROTOCOL_CROSS_EXAM,
            chain_head=local_head, spec_question=spec.question,
        )

    # ------------------------------------------------------------------
    # Protocol: doubly-efficient debate (Brown-Cohen-Irving-Piliouras 2023)
    # ------------------------------------------------------------------

    def _run_doubly_efficient(self, spec: DebateSpec, depth: int) -> DebateReport:
        # Run a bounded-depth two-player debate, then at the end ask the judge
        # to verify ONLY a single move (sampled uniformly).  This is the
        # essence of the doubly-efficient protocol: provers reason over an
        # exponential tree; verifier checks one leaf in poly time.
        base = self._run_two_player(spec, depth)
        if not base.transcript:
            return base
        # Sample a single move (excluding the verdict) uniformly
        movable = [m for m in base.transcript if m.kind != MOVE_VERDICT]
        if not movable:
            return base
        sampled = self._state.rng.choice(movable)
        # Build a one-move-subtranscript and re-poll the judge on it
        sub_transcript = [sampled]
        sub_verdict = self._poll_judge(spec.judge, spec, sub_transcript)
        sub_winner = self._winner_from_verdict(sub_verdict)
        # Final winner: agree with base only if the verifier confirms the same side
        if sub_winner == base.winner:
            confirmed = True
            win_prob = max(base.win_prob_hat, float(sub_verdict.get(sub_winner, 0.5)))
        else:
            confirmed = False
            win_prob = min(base.win_prob_hat, 0.5)
        # New transcript with marker
        transcript_list = list(base.transcript)
        marker = DebateMove(
            kind=MOVE_VERDICT,
            side=base.winner if confirmed else SIDE_TIE,
            round_index=base.rounds_used,
            meta={"doubly_efficient_confirmed": confirmed, "sampled_round": sampled.round_index},
        )
        transcript_list.append(marker)
        return DebateReport(
            winner=base.winner if confirmed else SIDE_TIE,
            win_prob_hat=win_prob,
            transcript=tuple(transcript_list),
            persuasion_trace=base.persuasion_trace,
            truthful_components=base.truthful_components,
            manipulative_components=base.manipulative_components,
            judge_votes=base.judge_votes,
            rounds_used=base.rounds_used,
            converged=confirmed,
            protocol=PROTOCOL_DOUBLY_EFFICIENT,
            chain_head=base.chain_head,
            spec_question=spec.question,
        )

    # ------------------------------------------------------------------
    # Protocol: market-maker debate (Hubinger 2020)
    # ------------------------------------------------------------------

    def _run_market_maker(self, spec: DebateSpec, depth: int) -> DebateReport:
        transcript: list[DebateMove] = []
        local_head = self._state.chain_head
        # Market price: judge's posterior probability of side A being correct.
        # Each round, A argues; B (the market maker) updates its quote toward
        # the judge's posterior of A's argument.
        price = 0.5
        for r in range(self.config.max_rounds):
            move_a = self._invoke_debater(spec.debater_a, spec, transcript, SIDE_A)
            self._validate_move(move_a, SIDE_A, r)
            transcript.append(move_a)
            local_head = _hash_entry(local_head, {"op": "argue", "move": move_a},
                                      self.config.hmac_key if self.config.hmac_key else None)
            # Poll judge on transcript-so-far → new posterior
            verdict = self._poll_judge(spec.judge, spec, transcript)
            judge_p_a = float(verdict.get(SIDE_A, 0.5))
            price = 0.5 * price + 0.5 * judge_p_a  # exponential smoothing toward judge
            # Market maker generates counter-evidence whenever |price − 0.5| < concede_threshold
            if abs(price - 0.5) > 1.0 - self.config.concede_threshold:
                # market has converged; stop
                break
            move_b = self._invoke_debater(spec.debater_b, spec, transcript, SIDE_B)
            self._validate_move(move_b, SIDE_B, r)
            transcript.append(move_b)
            local_head = _hash_entry(local_head, {"op": "counter", "move": move_b},
                                      self.config.hmac_key if self.config.hmac_key else None)
        winner = SIDE_A if price > 0.5 else (SIDE_B if price < 0.5 else SIDE_TIE)
        win_prob = max(price, 1.0 - price)
        verdict_move = DebateMove(
            kind=MOVE_VERDICT, side=winner, round_index=len(transcript) // 2,
            meta={"market_price": price},
        )
        transcript.append(verdict_move)
        local_head = _hash_entry(local_head, {"op": "verdict", "move": verdict_move},
                                  self.config.hmac_key if self.config.hmac_key else None)
        return DebateReport(
            winner=winner, win_prob_hat=win_prob,
            transcript=tuple(transcript),
            persuasion_trace=(), truthful_components=(), manipulative_components=(),
            judge_votes=((winner, win_prob),),
            rounds_used=len(transcript) // 2, converged=True,
            protocol=PROTOCOL_MARKET_MAKER,
            chain_head=local_head, spec_question=spec.question,
        )

    # ------------------------------------------------------------------
    # Protocol: Condorcet jury (Boland 1989)
    # ------------------------------------------------------------------

    def _run_jury(self, spec: DebateSpec, depth: int) -> DebateReport:
        if not spec.judges_for_jury:
            raise InvalidSpec("PROTOCOL_JURY requires spec.judges_for_jury non-empty")
        # First run the underlying two-player debate to produce the transcript
        # (the jury votes on the same transcript).  Use the *first* judge as
        # the canonical singleton during transcript generation; per-judge
        # opinion is collected once the transcript closes.
        single_judge = spec.judges_for_jury[0][0]
        base_spec = DebateSpec(
            question=spec.question, claim_a=spec.claim_a, claim_b=spec.claim_b,
            debater_a=spec.debater_a, debater_b=spec.debater_b,
            judge=single_judge,
            persuasion_model=spec.persuasion_model,
            ground_truth=spec.ground_truth,
            strategy_space=spec.strategy_space, meta=dict(spec.meta),
        )
        base = self._run_two_player(base_spec, depth)
        # Collect jury votes
        transcript_without_verdict = tuple(m for m in base.transcript if m.kind != MOVE_VERDICT)
        votes: list[tuple[str, float]] = []
        weights: list[float] = []
        for judge_fn, judge_acc in spec.judges_for_jury:
            v = self._poll_judge(judge_fn, spec, transcript_without_verdict)
            w = self._winner_from_verdict(v)
            conf = float(v.get(w, 0.5))
            votes.append((w, conf))
            weights.append(judge_acc)
        # Aggregate
        if self.config.aggregation == AGG_MAJORITY:
            winner = debater_jury_majority(votes)
        elif self.config.aggregation == AGG_WEIGHTED_LOG_ODDS:
            winner = debater_jury_log_odds(votes, weights)
        elif self.config.aggregation == AGG_UNANIMITY:
            sides = {v[0] for v in votes}
            winner = next(iter(sides)) if len(sides) == 1 else SIDE_TIE
        else:
            raise UnknownAggregation(f"aggregation={self.config.aggregation!r}")
        # Win probability: fraction of judges agreeing with winner
        if winner == SIDE_TIE:
            agree = 0.5
        else:
            agree = sum(1 for s, _ in votes if s == winner) / len(votes)
        verdict_move = DebateMove(
            kind=MOVE_VERDICT, side=winner, round_index=base.rounds_used,
            meta={"jury_size": len(votes), "agreement": agree},
        )
        new_transcript = list(transcript_without_verdict)
        new_transcript.append(verdict_move)
        return DebateReport(
            winner=winner, win_prob_hat=agree,
            transcript=tuple(new_transcript),
            persuasion_trace=(), truthful_components=(), manipulative_components=(),
            judge_votes=tuple(votes),
            rounds_used=base.rounds_used, converged=True,
            protocol=PROTOCOL_JURY,
            chain_head=base.chain_head, spec_question=spec.question,
        )

    # ------------------------------------------------------------------
    # Protocol: persuasion-aware debate (Khan-Hughes 2024)
    # ------------------------------------------------------------------

    def _run_persuasion_aware(self, spec: DebateSpec, depth: int) -> DebateReport:
        if spec.persuasion_model is None:
            raise InvalidSpec("PROTOCOL_PERSUASION_AWARE requires spec.persuasion_model")
        transcript: list[DebateMove] = []
        local_head = self._state.chain_head
        persuasion_trace: list[tuple[float, float]] = []
        truthful_components: list[float] = []
        manipulative_components: list[float] = []
        # Initial belief
        prior = spec.persuasion_model(spec, tuple(transcript))
        rounds_used = 0
        for r in range(self.config.max_rounds):
            move_a = self._invoke_debater(spec.debater_a, spec, transcript, SIDE_A)
            self._validate_move(move_a, SIDE_A, r)
            transcript.append(move_a)
            post_a = spec.persuasion_model(spec, tuple(transcript))
            delta_a = debater_bayes_posterior_shift(prior, post_a)
            t_a, m_a = debater_persuasion_decomposition(
                delta_a,
                move_a.argument.evidence if move_a.argument else 0.0,
                threshold=self.config.truthful_evidence_threshold,
            )
            persuasion_trace.append((delta_a, 0.0))
            truthful_components.append(t_a)
            manipulative_components.append(m_a)
            prior = dict(post_a)
            local_head = _hash_entry(local_head, {"op": "argue", "move": move_a,
                                                    "delta": delta_a, "t": t_a, "m": m_a},
                                      self.config.hmac_key if self.config.hmac_key else None)
            self._publish(DEBATER_ARGUED, {
                "side": SIDE_A, "round": r, "delta": delta_a,
                "truthful": t_a, "manipulative": m_a, "head": local_head,
            })

            move_b = self._invoke_debater(spec.debater_b, spec, transcript, SIDE_B)
            self._validate_move(move_b, SIDE_B, r)
            transcript.append(move_b)
            post_b = spec.persuasion_model(spec, tuple(transcript))
            delta_b = debater_bayes_posterior_shift(prior, post_b)
            t_b, m_b = debater_persuasion_decomposition(
                delta_b,
                move_b.argument.evidence if move_b.argument else 0.0,
                threshold=self.config.truthful_evidence_threshold,
            )
            persuasion_trace[-1] = (delta_a, delta_b)
            truthful_components.append(t_b)
            manipulative_components.append(m_b)
            prior = dict(post_b)
            local_head = _hash_entry(local_head, {"op": "counter", "move": move_b,
                                                    "delta": delta_b, "t": t_b, "m": m_b},
                                      self.config.hmac_key if self.config.hmac_key else None)
            self._publish(DEBATER_COUNTERED, {
                "side": SIDE_B, "round": r, "delta": delta_b,
                "truthful": t_b, "manipulative": m_b, "head": local_head,
            })
            rounds_used = r + 1
        # Final judge with manipulative-component penalty
        verdict = self._poll_judge(spec.judge, spec, transcript)
        raw_p_a = float(verdict.get(SIDE_A, 0.5))
        raw_p_b = float(verdict.get(SIDE_B, 1.0 - raw_p_a))
        manip_a = sum(m for m, mv in zip(manipulative_components, _interleave_sides(rounds_used)) if mv == SIDE_A)
        manip_b = sum(m for m, mv in zip(manipulative_components, _interleave_sides(rounds_used)) if mv == SIDE_B)
        pen = self.config.persuasion_penalty_weight
        adj_a = raw_p_a - pen * manip_a
        adj_b = raw_p_b - pen * manip_b
        # Renormalise into a probability over A vs B
        adj_a = max(0.0, adj_a)
        adj_b = max(0.0, adj_b)
        if adj_a + adj_b <= 0.0:
            adj_a, adj_b = 0.5, 0.5
        else:
            s = adj_a + adj_b
            adj_a /= s
            adj_b /= s
        winner = SIDE_A if adj_a > adj_b else (SIDE_B if adj_b > adj_a else SIDE_TIE)
        win_prob = max(adj_a, adj_b)
        verdict_move = DebateMove(
            kind=MOVE_VERDICT, side=winner, round_index=rounds_used,
            meta={"penalty_applied": True, "manip_a": manip_a, "manip_b": manip_b},
        )
        transcript.append(verdict_move)
        local_head = _hash_entry(local_head, {"op": "verdict", "move": verdict_move},
                                  self.config.hmac_key if self.config.hmac_key else None)
        return DebateReport(
            winner=winner, win_prob_hat=win_prob,
            transcript=tuple(transcript),
            persuasion_trace=tuple(persuasion_trace),
            truthful_components=tuple(truthful_components),
            manipulative_components=tuple(manipulative_components),
            judge_votes=((winner, win_prob),),
            rounds_used=rounds_used, converged=False,
            protocol=PROTOCOL_PERSUASION_AWARE,
            chain_head=local_head, spec_question=spec.question,
        )

    # ------------------------------------------------------------------
    # Move/verdict utilities
    # ------------------------------------------------------------------

    def _invoke_debater(
        self,
        fn: Debater_fn,
        spec: DebateSpec,
        transcript: Sequence[DebateMove],
        side: str,
    ) -> DebateMove:
        try:
            move = fn(spec, tuple(transcript), side)
        except DebaterError:
            raise
        except Exception as exc:
            raise InvalidMove(f"debater for side {side!r} raised: {exc!r}") from exc
        if not isinstance(move, DebateMove):
            raise InvalidMove(f"debater for side {side!r} must return a DebateMove; got {type(move).__name__}")
        return move

    def _validate_move(self, move: DebateMove, expected_side: str, round_idx: int) -> None:
        if move.side != expected_side:
            raise InvalidMove(
                f"move.side={move.side!r} but expected {expected_side!r}"
            )
        if move.round_index < 0:
            raise InvalidMove(f"move.round_index must be >= 0; got {move.round_index}")
        if move.kind == MOVE_CROSS_EXAMINE and move.target_index is None:
            raise InvalidMove(f"{MOVE_CROSS_EXAMINE} requires target_index")

    def _poll_judge(
        self, fn: Judge_fn, spec: DebateSpec, transcript: Sequence[DebateMove],
    ) -> Mapping[str, float]:
        try:
            verdict = fn(spec, tuple(transcript))
        except DebaterError:
            raise
        except Exception as exc:
            raise InvalidMove(f"judge raised: {exc!r}") from exc
        if not isinstance(verdict, Mapping):
            raise InvalidMove(f"judge must return a Mapping; got {type(verdict).__name__}")
        # Normalise: take the {A: p, B: q} interpretation; map unknown keys to side
        a = float(verdict.get(SIDE_A, 0.0))
        b = float(verdict.get(SIDE_B, 0.0))
        if a < 0 or b < 0:
            raise InvalidMove(f"judge probabilities must be non-negative; got A={a}, B={b}")
        if a + b <= 0.0:
            return {SIDE_A: 0.5, SIDE_B: 0.5}
        s = a + b
        norm = {SIDE_A: a / s, SIDE_B: b / s}
        self._publish(DEBATER_JUDGE_POLLED, {"verdict": norm})
        return norm

    def _winner_from_verdict(self, verdict: Mapping[str, float]) -> str:
        a = verdict.get(SIDE_A, 0.0)
        b = verdict.get(SIDE_B, 0.0)
        if a > b:
            return SIDE_A
        if b > a:
            return SIDE_B
        return SIDE_TIE

    # ------------------------------------------------------------------
    # Multi-debate / Monte-Carlo / Nash
    # ------------------------------------------------------------------

    def empirical_payoff(
        self,
        spec: DebateSpec,
        *,
        debaters_by_strategy: Mapping[str, Mapping[str, Debater_fn]],
        samples_per_cell: int = 16,
    ) -> PayoffMatrix:
        """Estimate the bimatrix payoff over ``spec.strategy_space``.

        ``debaters_by_strategy[side][strategy]`` returns a Debater_fn
        to use when ``side`` plays ``strategy``.  For each cell
        ``(i, j)``, runs ``samples_per_cell`` Monte-Carlo debates with
        A playing strategy ``i`` and B playing strategy ``j``; payoff
        for A is the empirical fraction of A-wins; payoff for B is
        ``1 − payoff_A`` (zero-sum) when ``ground_truth`` is absent,
        or the per-side accuracy when ``ground_truth`` is supplied.
        """
        if samples_per_cell < 1:
            raise InvalidConfig("samples_per_cell must be >= 1")
        strategies = spec.strategy_space
        n_strat = len(strategies)
        a_mat: list[list[float]] = [[0.0] * n_strat for _ in range(n_strat)]
        b_mat: list[list[float]] = [[0.0] * n_strat for _ in range(n_strat)]
        for i, sa in enumerate(strategies):
            for j, sb in enumerate(strategies):
                if sa not in debaters_by_strategy.get(SIDE_A, {}):
                    raise InvalidSpec(f"missing A debater for strategy {sa!r}")
                if sb not in debaters_by_strategy.get(SIDE_B, {}):
                    raise InvalidSpec(f"missing B debater for strategy {sb!r}")
                wins_a = 0
                wins_b = 0
                for _ in range(samples_per_cell):
                    inner_spec = DebateSpec(
                        question=spec.question, claim_a=spec.claim_a, claim_b=spec.claim_b,
                        debater_a=debaters_by_strategy[SIDE_A][sa],
                        debater_b=debaters_by_strategy[SIDE_B][sb],
                        judge=spec.judge,
                        persuasion_model=spec.persuasion_model,
                        judges_for_jury=spec.judges_for_jury,
                        ground_truth=spec.ground_truth,
                        strategy_space=spec.strategy_space,
                        meta=dict(spec.meta),
                    )
                    report = self.run(inner_spec)
                    if report.winner == SIDE_A:
                        wins_a += 1
                    elif report.winner == SIDE_B:
                        wins_b += 1
                a_mat[i][j] = wins_a / samples_per_cell
                b_mat[i][j] = wins_b / samples_per_cell
        return PayoffMatrix(
            strategies=strategies,
            matrix_a=tuple(tuple(row) for row in a_mat),
            matrix_b=tuple(tuple(row) for row in b_mat),
            samples_per_cell=samples_per_cell,
        )

    def nash_check(self, payoff: PayoffMatrix) -> NashResult:
        """Run support enumeration on ``payoff`` to find a Nash equilibrium."""
        n = len(payoff.strategies)
        if n == 2:
            return debater_payoff_nash_2x2(payoff.matrix_a, payoff.matrix_b)
        return debater_support_enumeration(
            payoff.matrix_a, payoff.matrix_b, strategies=payoff.strategies,
        )

    # ------------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------------

    def certify(
        self,
        report: DebateReport | None = None,
        *,
        delta: float | None = None,
    ) -> DebaterCertificate:
        """PAC certificate over the accumulated win-rate of the truth side.

        If ``report`` is given, the certificate is restricted to that
        single debate.  Otherwise it aggregates across every debate
        ``run`` so far.
        """
        delta = delta if delta is not None else (1.0 - self.config.confidence)
        if not (0.0 < delta < 1.0):
            raise InvalidConfig("delta must be in (0, 1)")
        with self._lock:
            if report is not None:
                rows = [report]
            else:
                rows = list(self._state.reports)
            if not rows:
                raise NotRun("certify requires at least one report")
            n = len(rows)
            wins = [1.0 if r.winner != SIDE_TIE else 0.5 for r in rows]
            p_hat = sum(wins) / n
            if n >= 2:
                mean = p_hat
                var = sum((w - mean) ** 2 for w in wins) / max(1, n - 1)
            else:
                var = 0.25
            hoeff = debater_hoeffding_lcb(p_hat, n, delta)
            try:
                bern = debater_bernstein_lcb(p_hat, var, n, delta)
            except InsufficientData:
                bern = hoeff
            # Condorcet only meaningful if any report used jury aggregation
            if any(r.protocol == PROTOCOL_JURY for r in rows):
                avg_p = max(
                    0.51,
                    sum(r.win_prob_hat for r in rows if r.protocol == PROTOCOL_JURY)
                    / max(1, sum(1 for r in rows if r.protocol == PROTOCOL_JURY)),
                )
                m_judges = max(1, max(len(r.judge_votes) for r in rows if r.protocol == PROTOCOL_JURY))
                cond = debater_condorcet_lcb(avg_p, m_judges, delta)
            else:
                cond = hoeff
            # Persuasion penalty: total manipulative components observed
            persuasion_pen = sum(sum(r.manipulative_components) for r in rows)
            # Calibration ECE (across the cumulative state, not just `rows`)
            if self._state.judge_confidences:
                ece_report = debater_calibration_ece(
                    self._state.judge_confidences, self._state.judge_outcomes
                )
                ece = ece_report.ece
            else:
                ece = None
            cert = DebaterCertificate(
                n=n, delta=delta,
                win_prob_hat=p_hat,
                hoeffding_lcb=hoeff, bernstein_lcb=bern,
                condorcet_lcb=cond,
                persuasion_penalty=persuasion_pen,
                nash_conv=0.0,  # filled by an explicit nash_check pipeline
                calibration_ece=ece,
                protocol=self.config.protocol,
                chain_head=self._state.chain_head,
            )
            self._publish(DEBATER_CERTIFIED, {
                "n": n, "delta": delta, "p_hat": p_hat,
                "hoeffding": hoeff, "bernstein": bern,
                "condorcet": cond,
                "head": self._state.chain_head,
            })
            return cert

    def anytime_certify(self, delta: float | None = None) -> AnytimeCertificate:
        """HRMS anytime-valid confidence sequence over the cumulative win-rate."""
        delta = delta if delta is not None else (1.0 - self.config.confidence)
        if not (0.0 < delta < 1.0):
            raise InvalidConfig("delta must be in (0, 1)")
        with self._lock:
            n = len(self._state.reports)
            if n < 2:
                raise InsufficientData("anytime_certify requires >= 2 reports")
            wins = [1.0 if r.winner != SIDE_TIE else 0.5 for r in self._state.reports]
            p_hat = sum(wins) / n
            radius = debater_hrms_radius(n, delta)
            return AnytimeCertificate(
                n=n, delta=delta, win_prob_hat=p_hat,
                lcb=max(0.0, p_hat - radius), ucb=min(1.0, p_hat + radius),
                radius=radius,
            )

    def calibration(self, n_bins: int = 10) -> CalibrationReport:
        """ECE report over the judge's predicted vs realised correctness."""
        with self._lock:
            if not self._state.judge_confidences:
                raise InsufficientData("calibration requires reports with ground_truth")
            return debater_calibration_ece(
                self._state.judge_confidences, self._state.judge_outcomes, n_bins
            )

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-encodable snapshot of debater state."""
        with self._lock:
            return {
                "config": {
                    "protocol": self.config.protocol,
                    "judge_model": self.config.judge_model,
                    "aggregation": self.config.aggregation,
                    "max_rounds": self.config.max_rounds,
                    "max_depth": self.config.max_depth,
                    "judge_accuracy": self.config.judge_accuracy,
                    "persuasion_penalty_weight": self.config.persuasion_penalty_weight,
                    "truthful_evidence_threshold": self.config.truthful_evidence_threshold,
                    "concede_threshold": self.config.concede_threshold,
                    "confidence": self.config.confidence,
                    "record_every": self.config.record_every,
                    "seed": self.config.seed,
                    # hmac_key is sensitive; never serialised
                },
                "rng_state": _rng_state_to_json(self._state.rng),
                "chain_head": self._state.chain_head,
                "n_reports": len(self._state.reports),
                "judge_confidences": list(self._state.judge_confidences),
                "judge_outcomes": list(self._state.judge_outcomes),
            }

    def restore(self, snap: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`snapshot`."""
        with self._lock:
            if "chain_head" not in snap or "rng_state" not in snap:
                raise InvalidTranscript("snapshot is missing required fields")
            self._state.chain_head = str(snap["chain_head"])
            self._state.rng = _rng_state_from_json(snap["rng_state"])
            self._state.judge_confidences = list(snap.get("judge_confidences", []))
            self._state.judge_outcomes = list(snap.get("judge_outcomes", []))
            self._publish(DEBATER_RESET, {"head": self._state.chain_head})

    def reset(self) -> None:
        """Wipe per-instance state and re-seed."""
        with self._lock:
            self._state = _State(
                rng=random.Random(self.config.seed),
                chain_head=debater_ledger_root(
                    self.config.hmac_key if self.config.hmac_key else None
                ),
            )
            self._publish(DEBATER_RESET, {"head": self._state.chain_head})


# ---------------------------------------------------------------------------
# Helpers used by the persuasion-aware protocol
# ---------------------------------------------------------------------------


def _interleave_sides(rounds_used: int) -> list[str]:
    """Construct the A,B,A,B,... side list for ``2*rounds_used`` moves."""
    out: list[str] = []
    for _ in range(rounds_used):
        out.append(SIDE_A)
        out.append(SIDE_B)
    return out


def _rng_state_to_json(rng: random.Random) -> str:
    s = rng.getstate()
    # state is (version, internal_state_tuple, gauss_next)
    version, internal, gauss = s
    return json.dumps({"v": version, "s": list(internal), "g": gauss})


def _rng_state_from_json(blob: str) -> random.Random:
    obj = json.loads(blob)
    r = random.Random()
    r.setstate((int(obj["v"]), tuple(int(x) for x in obj["s"]), obj["g"]))
    return r
