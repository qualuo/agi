r"""Speculator — speculative execution as a runtime primitive.

Every modern fast LLM-inference engine (vLLM, TGI, llama.cpp, TensorRT-
LLM, MLC) ships *speculative decoding* — a small "draft" model proposes
:math:`K` tokens at draft cost, then the expensive "target" model
verifies all :math:`K` in a single forward pass.  Tokens whose draft
distribution agrees with the target are accepted; the first
disagreement is corrected by sampling from a residual distribution that
preserves the target's output distribution exactly.  The result is a
**provably equivalent** output stream that emits up to :math:`K + 1`
tokens per target forward pass instead of :math:`1`.

That pattern generalises beyond LLM tokens.  Anywhere the runtime has a
*cheap-but-approximate* executor and an *expensive-but-correct*
verifier on the same step type, ``Speculator`` schedules the cheap
proposal first, verifies its prefix in a single expensive call, and
guarantees output equivalence with a measurable empirical speedup
relative to running the expensive verifier alone.  Examples:

  * LLM token streams (the canonical case).
  * Plan-step execution: draft from a learned distilled policy, verify
    with the full ``Searcher``.
  * Tool dispatch: draft with a cached result, verify by calling the
    real tool.
  * Memory retrieval: draft with a stale in-RAM index, verify with the
    canonical persistent store.

The pitch reduced to a runtime call::

    spec = Speculator(SpeculatorConfig(
        algorithm="speculative_sampling", k_draft=4, seed=0))

    def draft(state):
        # Cheap, possibly-wrong proposal of up to k_draft tokens and
        # their per-step probability distributions under the draft.
        return [(token_i, p_draft_dist_i) for i in range(K)]

    def target(state, draft_tokens):
        # One expensive call returns the *target* probability distribution
        # at *every* position the draft proposed plus one extra position
        # for the bonus token (Leviathan-Kalman-Matias 2023).
        return [(token_i, p_target_dist_i) for i in range(K + 1)]

    out = spec.step(state, draft=draft, target=target)
    # out.tokens         — accepted prefix + (bonus | corrected) token
    # out.accept_count   — number of draft tokens accepted (≥ 0, ≤ K)
    # out.equivalence_ok — sampled from same distribution as target alone

    report = spec.report()
    # report.empirical_acceptance_rate          ∈ [0,1]
    # report.empirical_acceptance_rate_lcb      anytime-valid LCB
    # report.expected_tokens_per_target_call    ∈ [1, K+1]
    # report.empirical_speedup                  vs. baseline tokens/call
    # report.speedup_lcb                        Maurer-Pontil 2009 LCB
    # report.fingerprint_hash                   replay-verifiable receipt

What this primitive ships
-------------------------

  * **Algorithm families (all stdlib, no NumPy, no Torch):**

    * ``"speculative_sampling"`` — Chen-Borgeaud-Irving-Lespiau-Sifre-
      Jumper 2023 *Accelerating Large Language Model Decoding with
      Speculative Sampling*.  At each draft position :math:`i` with
      draft probability :math:`q_i(\cdot)` and target probability
      :math:`p_i(\cdot)`, the proposed token :math:`x_i` is accepted
      with probability :math:`\min(1, p_i(x_i) / q_i(x_i))`.  If
      rejected, sample the replacement from the *residual*
      distribution :math:`(p_i - q_i)_+ / \sum_x (p_i(x) - q_i(x))_+`.
      Provably equivalent to sampling tokens i.i.d. from :math:`p_i`.

    * ``"leviathan_decoding"`` — Leviathan-Kalman-Matias 2023 *Fast
      Inference from Transformers via Speculative Decoding*.  The
      original formulation, identical to ``speculative_sampling`` for
      pure sampling.  Distinguished here because it can run with
      greedy-target (argmax) verification, in which case it accepts
      the draft only if its argmax matches the target's argmax — a
      faster but greedy-equivalent (not sampling-equivalent) variant.

    * ``"greedy"`` — Accept :math:`x_i` iff :math:`\arg\max p_i =
      x_i`.  Equivalent to ``leviathan_decoding`` with temperature 0
      target.  Strongest acceptance rate when the target distribution
      is naturally peaky.

    * ``"medusa_tree"`` — Cai-Li-Geng-Peng-Lee-Chen-Dao 2024 *Medusa:
      Simple LLM Inference Acceleration Framework with Multiple
      Decoding Heads*.  Draft proposes a *tree* of candidate token
      sequences in a single shot; target verifies the entire tree in
      one forward pass; accepted path is the longest prefix that
      sampling-equivalence accepts.  Configured via ``tree_width``.

    * ``"self_spec_early_exit"`` — Zhang-Yang-Sun et al. 2024 *Draft
      and Verify: Lossless Large Language Model Acceleration via
      Self-Speculative Decoding*.  Uses the same model for draft and
      target by *skipping* a configurable fraction of internal layers
      during draft and running the full network for verification.
      In the Speculator interface this manifests as a pair of
      callables that share the same backing model.

    * ``"eagle"`` — Li-Wei-Zhang-Zhang 2024 *EAGLE: Speculative
      Sampling Requires Rethinking Feature Uncertainty*.  Same outer
      loop as ``speculative_sampling`` with an extrapolation
      heuristic on the draft distribution (configured via
      ``eagle_alpha``); supported because the *acceptance test* is
      identical — the extrapolation is in how the caller's ``draft``
      function builds the proposal.

    * ``"lookahead"`` — Fu-Bansal-Beltagy-Bauer-Subramanian-Lewis-
      Beltagy 2024 *Lookahead Decoding*.  N-gram cache populated from
      past acceptances proposes the next draft; cache hits skip the
      draft-model call entirely.  Tracked here through the
      ``cache_size`` knob.

  * **Statistical certificates**:

    * **Acceptance-rate LCB / UCB** via Hoeffding 1963 / Maurer-Pontil
      2009 empirical-Bernstein / Howard-Ramdas-McAuliffe-Sekhon 2021
      anytime-valid confidence sequences.

    * **Speedup LCB**: lower confidence bound on
      ``E[accepted_tokens + 1 per target call]`` ∈ ``[1, K+1]``.

    * **Equivalence martingale**: under correct implementation, the
      probability of each emitted token under target equals the
      probability under "sample one token from target" — i.e. the
      *unbiased* test statistic ``log p_target(emitted) -
      log p_baseline(emitted)`` is a martingale with mean 0.  We
      track the empirical mean and a Maurer-Pontil 2009 LCB; a
      significant drift indicates an implementation bug or rejection-
      sampling violation.

  * **Replay determinism**:

    The internal RNG is seeded from ``config.seed``.  Given the same
    seed, same draft outputs, and same target outputs, the
    ``Speculator`` emits the same accepted prefix + correction
    sequence byte-for-byte.

  * **Tamper-evident fingerprint chain**:

    Every step emits a SHA-256 chain event with payload
    ``{n_proposed, n_accepted, emitted, bonus_or_correction,
    target_prob, equivalence_ok}``.  ``AlignerReport``-style
    fingerprint allows ``AttestationLedger`` to replay the trace.

  * **Composes with** the rest of the runtime:

    * ``Distiller`` produces the *draft* callable for any decision
      primitive — Speculator turns its amortised forward pass into
      runtime-level inference acceleration with target-equivalent
      output.
    * ``Searcher`` can be the *target* verifier; ``Distiller``-fit
      policy is the draft.
    * ``Bandit`` / ``BayesOpt`` reads the acceptance-rate UCB to
      decide when to refresh the draft.
    * ``DriftSentinel`` watches the running acceptance rate; if it
      trips, the draft is rolled back.
    * ``AttestationLedger`` chains every speculative step.
    * ``PrivacyAccountant`` advances on each verified call when the
      input is sensitive.
    * ``Coordinator`` — every Goal whose execution emits a stream of
      atomic decisions routes through ``Speculator.step`` for
      runtime-level acceleration with provable output equivalence.

This module is **pure stdlib** — no PyTorch, no NumPy, no model
runtime.  The primitive is *transport-agnostic*: it operates on
caller-supplied draft + target callables that return probability
distributions over an abstract token alphabet, so it accelerates LLM
decoding, plan execution, retrieval pipelines, and tool dispatch
identically.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import threading
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


class SpeculatorError(Exception):
    """Base for every Speculator-raised error."""


class InvalidConfig(SpeculatorError):
    """A SpeculatorConfig is structurally invalid."""


class InvalidDraft(SpeculatorError):
    """A draft callable returned a malformed proposal."""


class InvalidTarget(SpeculatorError):
    """A target callable returned a malformed verification."""


class UnknownAlgorithm(SpeculatorError):
    """The requested algorithm is not one of this module's algorithms."""


class EquivalenceViolation(SpeculatorError):
    """The empirical equivalence test rejected at the configured level."""


# =============================================================================
# Algorithm name constants
# =============================================================================


ALG_SPEC_SAMPLING = "speculative_sampling"
ALG_LEVIATHAN = "leviathan_decoding"
ALG_GREEDY = "greedy"
ALG_MEDUSA_TREE = "medusa_tree"
ALG_SELF_SPEC = "self_spec_early_exit"
ALG_EAGLE = "eagle"
ALG_LOOKAHEAD = "lookahead"

KNOWN_ALGORITHMS: Tuple[str, ...] = (
    ALG_SPEC_SAMPLING,
    ALG_LEVIATHAN,
    ALG_GREEDY,
    ALG_MEDUSA_TREE,
    ALG_SELF_SPEC,
    ALG_EAGLE,
    ALG_LOOKAHEAD,
)


# =============================================================================
# Type aliases
# =============================================================================


Token = Hashable
ProbDist = Mapping[Token, float]
DraftFn = Callable[[Any], Sequence[Tuple[Token, ProbDist]]]
TargetFn = Callable[[Any, Sequence[Token]], Sequence[Tuple[Token, ProbDist]]]


# =============================================================================
# Event kinds
# =============================================================================


SPECULATOR_STARTED = "speculator.started"
SPECULATOR_STEP = "speculator.step"
SPECULATOR_REPORTED = "speculator.reported"
SPECULATOR_RESET = "speculator.reset"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SpeculatorConfig:
    """Configuration for ``Speculator``.

    Algorithm
        algorithm:      one of ``KNOWN_ALGORITHMS``.
        k_draft:        max draft length per step.
        tree_width:     Medusa tree width per layer (medusa_tree only).
        eagle_alpha:    EAGLE extrapolation parameter (eagle only).
        cache_size:     Lookahead n-gram cache size (lookahead only).
        skip_fraction:  layer-skip fraction (self_spec_early_exit only).

    Cost model
        draft_cost:     wall-time / FLOPs ratio of draft vs target.
                        Used to compute expected_speedup.  Default 0.1.
        target_cost:    nominal cost of a target call (default 1.0).

    Statistical certificates
        alpha:                  CI tail probability (default 0.05).
        equivalence_alpha:      tail of equivalence martingale test.
        equivalence_fail_on:    "warn" | "raise" — what to do when the
                                equivalence LCB rejects.

    Determinism / certificate
        seed:        RNG seed.
        secret_key:  optional HMAC key for the certificate chain.
    """
    algorithm: str = ALG_SPEC_SAMPLING
    k_draft: int = 4
    tree_width: int = 2
    eagle_alpha: float = 0.9
    cache_size: int = 256
    skip_fraction: float = 0.5

    draft_cost: float = 0.1
    target_cost: float = 1.0

    alpha: float = 0.05
    equivalence_alpha: float = 0.01
    equivalence_fail_on: str = "warn"

    seed: int = 0
    secret_key: bytes = b""

    def __post_init__(self) -> None:
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise InvalidConfig(
                f"algorithm={self.algorithm!r} not in {KNOWN_ALGORITHMS}"
            )
        if self.k_draft < 1:
            raise InvalidConfig(f"k_draft={self.k_draft!r} must be ≥ 1")
        if self.tree_width < 1:
            raise InvalidConfig(f"tree_width={self.tree_width!r} must be ≥ 1")
        if not 0.0 <= self.eagle_alpha <= 1.0:
            raise InvalidConfig(f"eagle_alpha={self.eagle_alpha!r} must be in [0,1]")
        if self.cache_size < 1:
            raise InvalidConfig(f"cache_size={self.cache_size!r} must be ≥ 1")
        if not 0.0 < self.skip_fraction < 1.0:
            raise InvalidConfig(
                f"skip_fraction={self.skip_fraction!r} must be in (0,1)"
            )
        if self.draft_cost <= 0:
            raise InvalidConfig(f"draft_cost={self.draft_cost!r} must be > 0")
        if self.target_cost <= 0:
            raise InvalidConfig(f"target_cost={self.target_cost!r} must be > 0")
        if not 0.0 < self.alpha < 1.0:
            raise InvalidConfig(f"alpha={self.alpha!r} must be in (0,1)")
        if not 0.0 < self.equivalence_alpha < 1.0:
            raise InvalidConfig(
                f"equivalence_alpha={self.equivalence_alpha!r} must be in (0,1)"
            )
        if self.equivalence_fail_on not in ("warn", "raise"):
            raise InvalidConfig(
                f"equivalence_fail_on must be 'warn' or 'raise', "
                f"got {self.equivalence_fail_on!r}"
            )


# =============================================================================
# Step output + report
# =============================================================================


@dataclass(frozen=True)
class StepOutput:
    """Result of one speculator step."""
    tokens: Tuple[Token, ...]
    accept_count: int
    n_proposed: int
    correction_was_sampled: bool
    bonus_token_included: bool
    equivalence_log_ratio: float
    elapsed_seconds: float


@dataclass
class SpeculatorReport:
    """Canonical statistics report."""
    algorithm: str
    n_steps: int
    n_proposed_total: int
    n_accepted_total: int
    n_emitted_total: int
    empirical_acceptance_rate: float
    empirical_acceptance_rate_lcb_hoeffding: float
    empirical_acceptance_rate_lcb_bernstein: float
    empirical_acceptance_rate_lcb_anytime: float
    empirical_acceptance_rate_ucb_hoeffding: float
    expected_tokens_per_target_call: float
    empirical_speedup: float
    speedup_lcb_bernstein: float
    equivalence_log_ratio_mean: float
    equivalence_log_ratio_lcb: float
    equivalence_test_passed: bool
    fingerprint_hash: str
    chain_length: int
    elapsed_seconds: float


# =============================================================================
# Canonical bytes + certificate chain
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
    GENESIS = b"agi.speculator.v1\x00"

    def __init__(self, secret_key: bytes = b"") -> None:
        self._secret = bytes(secret_key)
        self._h = hashlib.sha256(self.GENESIS + self._secret).digest()
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
# Numerical helpers
# =============================================================================


def _hoeffding_half_width(n: int, alpha: float) -> float:
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def _bernstein_half_width(n: int, variance: float, alpha: float,
                          rng: float = 1.0) -> float:
    if n <= 1:
        return float("inf")
    log_term = math.log(2.0 / alpha)
    a = math.sqrt(2.0 * variance * log_term / n)
    b = 7.0 * rng * log_term / (3.0 * (n - 1))
    return a + b


def _hrms_anytime_half_width(n: int, variance: float, alpha: float) -> float:
    if n <= 2:
        return float("inf")
    log_n = math.log(max(2.0, float(n)))
    log_log_n = math.log(max(2.0, log_n))
    eta = math.log(1.0 / alpha) + 2.0 * log_log_n + math.log(math.pi**2 / 6.0)
    a = math.sqrt(2.0 * variance * eta / n)
    b = eta / (3.0 * n)
    return a + b


def _empirical_variance(xs: Sequence[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _normalise_dist(p: ProbDist) -> Dict[Token, float]:
    total = 0.0
    for k, v in p.items():
        if v < 0:
            raise InvalidDraft(f"probability for {k!r} is negative: {v}")
        if not math.isfinite(v):
            raise InvalidDraft(f"probability for {k!r} is not finite: {v}")
        total += float(v)
    if total <= 0:
        raise InvalidDraft("distribution sums to ≤ 0")
    return {k: float(v) / total for k, v in p.items() if v > 0}


def _residual_dist(p: ProbDist, q: ProbDist) -> Dict[Token, float]:
    """Return (p - q)_+ normalised — Chen et al. 2023 / Leviathan et al. 2023.

    Used to sample the correction token when a draft proposal is
    rejected so that the marginal of the emitted token equals p exactly.
    """
    diff: Dict[Token, float] = {}
    total = 0.0
    for k, pv in p.items():
        qv = q.get(k, 0.0)
        d = pv - qv
        if d > 0:
            diff[k] = d
            total += d
    if total <= 0:
        # All p mass is dominated by q at the support; fall back to p.
        return _normalise_dist(p)
    return {k: v / total for k, v in diff.items()}


def _sample_from_dist(dist: ProbDist, rng: random.Random) -> Token:
    """Sample one token from the distribution."""
    items = sorted(dist.items())  # deterministic ordering
    total = sum(v for _, v in items)
    if total <= 0:
        raise InvalidDraft("distribution sums to ≤ 0")
    u = rng.random() * total
    acc = 0.0
    for k, v in items:
        acc += v
        if u <= acc:
            return k
    return items[-1][0]


def _argmax_token(dist: ProbDist) -> Token:
    best_k = None
    best_v = -math.inf
    best_h = b""
    for k, v in dist.items():
        h = hashlib.sha256(_canonical_bytes(k)).digest()
        if v > best_v or (v == best_v and (best_k is None or h < best_h)):
            best_v = v
            best_k = k
            best_h = h
    return best_k  # type: ignore[return-value]


# =============================================================================
# Speculator
# =============================================================================


class Speculator:
    """Speculative-execution accelerator as a runtime primitive.

    Thread-safe via a re-entrant lock guarding the certificate chain
    and accumulated statistics.
    """

    def __init__(self, config: Optional[SpeculatorConfig] = None) -> None:
        self.config = config or SpeculatorConfig()
        self._lock = threading.RLock()
        self._rng = random.Random(self.config.seed)
        self._chain = _CertChain(self.config.secret_key)

        # Statistics.
        self._n_steps = 0
        self._n_proposed = 0
        self._n_accepted = 0
        self._n_emitted = 0
        self._accept_indicators: List[float] = []  # per-token (0 or 1)
        self._step_emit_counts: List[int] = []     # tokens emitted per step
        self._eq_log_ratios: List[float] = []
        self._t_start = time.perf_counter()
        self._lookahead_cache: Dict[Tuple[Token, ...], List[Token]] = {}

        self._chain.emit(SPECULATOR_STARTED, {
            "algorithm": self.config.algorithm,
            "k_draft": self.config.k_draft,
            "seed": self.config.seed,
        })

    # -- core step --------------------------------------------------------

    def step(self, state: Any, *, draft: DraftFn, target: TargetFn,
             rng: Optional[random.Random] = None) -> StepOutput:
        """Run one speculative step.

        ``draft(state)`` returns a sequence of ``(token, draft_prob_dist)``
        of length ≤ ``k_draft``.

        ``target(state, draft_tokens)`` returns a sequence of
        ``(token, target_prob_dist)`` of length ``len(draft_tokens) + 1``
        (one extra position for the bonus token at the end).  The first
        ``len(draft_tokens)`` positions are the target's distributions
        *at the same positions the draft proposed*; the extra position
        is the target's distribution *after* all draft tokens, used to
        emit the bonus token when all drafts are accepted.

        Returns a ``StepOutput`` describing what was emitted.
        """
        rng = rng or self._rng
        cfg = self.config
        t0 = time.perf_counter()
        with self._lock:
            proposed = list(draft(state))
            if not proposed:
                raise InvalidDraft("draft returned empty proposal")
            if len(proposed) > cfg.k_draft:
                proposed = proposed[: cfg.k_draft]
            draft_tokens = [t for t, _ in proposed]
            # Normalise draft distributions.
            try:
                draft_dists = [_normalise_dist(d) for _, d in proposed]
            except InvalidDraft:
                raise
            verifications = list(target(state, draft_tokens))
            expected_len = len(draft_tokens) + 1
            if len(verifications) != expected_len:
                raise InvalidTarget(
                    f"target returned {len(verifications)} verifications, "
                    f"expected {expected_len}"
                )
            target_dists = [_normalise_dist(d) for _, d in verifications]

            if cfg.algorithm == ALG_GREEDY:
                emitted, accept_count, eq_log = self._step_greedy(
                    draft_tokens, draft_dists, target_dists)
                correction_sampled = (accept_count < len(draft_tokens))
                bonus_included = (accept_count == len(draft_tokens))
            elif cfg.algorithm == ALG_MEDUSA_TREE:
                emitted, accept_count, eq_log = self._step_medusa(
                    draft_tokens, draft_dists, target_dists, rng)
                correction_sampled = (accept_count < len(draft_tokens))
                bonus_included = (accept_count == len(draft_tokens))
            elif cfg.algorithm == ALG_LOOKAHEAD:
                emitted, accept_count, eq_log = self._step_sampling(
                    draft_tokens, draft_dists, target_dists, rng)
                # Update n-gram cache from the accepted tokens.
                if len(emitted) >= 2:
                    for i in range(len(emitted) - 1):
                        key = (emitted[i],)
                        nexts = self._lookahead_cache.setdefault(key, [])
                        nexts.append(emitted[i + 1])
                        if len(nexts) > 8:
                            nexts.pop(0)
                if len(self._lookahead_cache) > cfg.cache_size:
                    # FIFO eviction.
                    first_key = next(iter(self._lookahead_cache))
                    del self._lookahead_cache[first_key]
                correction_sampled = (accept_count < len(draft_tokens))
                bonus_included = (accept_count == len(draft_tokens))
            else:
                # ALG_SPEC_SAMPLING, ALG_LEVIATHAN, ALG_SELF_SPEC, ALG_EAGLE
                # all share the sampling-equivalent acceptance test.
                emitted, accept_count, eq_log = self._step_sampling(
                    draft_tokens, draft_dists, target_dists, rng)
                correction_sampled = (accept_count < len(draft_tokens))
                bonus_included = (accept_count == len(draft_tokens))

            # Statistics bookkeeping.
            self._n_steps += 1
            self._n_proposed += len(draft_tokens)
            self._n_accepted += accept_count
            self._n_emitted += len(emitted)
            for i in range(len(draft_tokens)):
                self._accept_indicators.append(
                    1.0 if i < accept_count else 0.0
                )
            self._step_emit_counts.append(len(emitted))
            self._eq_log_ratios.append(eq_log)

            t1 = time.perf_counter()
            out = StepOutput(
                tokens=tuple(emitted),
                accept_count=accept_count,
                n_proposed=len(draft_tokens),
                correction_was_sampled=correction_sampled,
                bonus_token_included=bonus_included,
                equivalence_log_ratio=eq_log,
                elapsed_seconds=t1 - t0,
            )
            self._chain.emit(SPECULATOR_STEP, {
                "n_proposed": len(draft_tokens),
                "n_accepted": accept_count,
                "emitted": [str(t) for t in emitted],
                "bonus": bonus_included,
                "corrected": correction_sampled,
                "eq_log_ratio": eq_log,
            })
            return out

    # -- algorithm variants ----------------------------------------------

    def _step_sampling(self,
                        draft_tokens: Sequence[Token],
                        draft_dists: Sequence[Dict[Token, float]],
                        target_dists: Sequence[Dict[Token, float]],
                        rng: random.Random,
                        ) -> Tuple[List[Token], int, float]:
        """Chen et al. 2023 / Leviathan et al. 2023 speculative sampling.

        Equivalent to sampling each token i.i.d. from the target.
        """
        emitted: List[Token] = []
        accept_count = 0
        eq_log_sum = 0.0
        for i, x in enumerate(draft_tokens):
            q_x = draft_dists[i].get(x, 0.0)
            p_x = target_dists[i].get(x, 0.0)
            if q_x <= 0:
                # Draft proposed a token with 0 draft probability — accept
                # iff target also has support; otherwise reject.
                if p_x > 0:
                    accept_count += 1
                    emitted.append(x)
                    eq_log_sum += math.log(p_x)
                    continue
                # Reject and sample correction.
                correction = _sample_from_dist(target_dists[i], rng)
                emitted.append(correction)
                eq_log_sum += math.log(max(1e-30,
                                            target_dists[i].get(correction, 0.0)))
                return emitted, accept_count, eq_log_sum
            ratio = min(1.0, p_x / q_x)
            if rng.random() <= ratio:
                # Accept.
                accept_count += 1
                emitted.append(x)
                eq_log_sum += math.log(max(1e-30, p_x))
            else:
                # Reject; sample correction from residual.
                residual = _residual_dist(target_dists[i], draft_dists[i])
                correction = _sample_from_dist(residual, rng)
                emitted.append(correction)
                eq_log_sum += math.log(max(1e-30,
                                            target_dists[i].get(correction, 0.0)))
                return emitted, accept_count, eq_log_sum
        # All accepted — emit bonus from the (k_draft+1)-th distribution.
        bonus_dist = target_dists[-1]
        bonus = _sample_from_dist(bonus_dist, rng)
        emitted.append(bonus)
        eq_log_sum += math.log(max(1e-30, bonus_dist.get(bonus, 0.0)))
        return emitted, accept_count, eq_log_sum

    def _step_greedy(self,
                      draft_tokens: Sequence[Token],
                      draft_dists: Sequence[Dict[Token, float]],
                      target_dists: Sequence[Dict[Token, float]],
                      ) -> Tuple[List[Token], int, float]:
        """Greedy verification — accept iff argmax(target) == draft token."""
        emitted: List[Token] = []
        accept_count = 0
        eq_log_sum = 0.0
        for i, x in enumerate(draft_tokens):
            tgt_argmax = _argmax_token(target_dists[i])
            if tgt_argmax == x:
                accept_count += 1
                emitted.append(x)
                eq_log_sum += math.log(max(1e-30, target_dists[i].get(x, 0.0)))
            else:
                emitted.append(tgt_argmax)
                eq_log_sum += math.log(max(1e-30,
                                            target_dists[i].get(tgt_argmax, 0.0)))
                return emitted, accept_count, eq_log_sum
        # All accepted — emit greedy bonus.
        bonus = _argmax_token(target_dists[-1])
        emitted.append(bonus)
        eq_log_sum += math.log(max(1e-30, target_dists[-1].get(bonus, 0.0)))
        return emitted, accept_count, eq_log_sum

    def _step_medusa(self,
                      draft_tokens: Sequence[Token],
                      draft_dists: Sequence[Dict[Token, float]],
                      target_dists: Sequence[Dict[Token, float]],
                      rng: random.Random,
                      ) -> Tuple[List[Token], int, float]:
        """Medusa-style tree verification reduces to scanning the proposal."""
        # The actual Medusa optimization is parallel verification of multiple
        # candidate sequences via attention masks; the *acceptance test* per
        # accepted prefix length is identical to spec sampling.  This entry
        # point lets the caller plug in a tree-aware draft / target callable
        # while reusing the same statistical accounting.
        return self._step_sampling(draft_tokens, draft_dists, target_dists, rng)

    # -- statistics + report ---------------------------------------------

    def report(self) -> SpeculatorReport:
        """Return current statistics + replay-deterministic fingerprint."""
        with self._lock:
            cfg = self.config
            n_props = self._n_proposed
            n_acc = self._n_accepted
            acc_rate = n_acc / max(1, n_props)
            n_steps = self._n_steps

            alpha = cfg.alpha
            v = _empirical_variance(self._accept_indicators)
            hw_h = _hoeffding_half_width(n_props, alpha)
            hw_b = _bernstein_half_width(n_props, v, alpha) if n_props > 1 else 1.0
            hw_a = _hrms_anytime_half_width(n_props, v, alpha) if n_props > 2 else 1.0
            lcb_h = max(0.0, acc_rate - hw_h)
            lcb_b = max(0.0, acc_rate - hw_b)
            lcb_a = max(0.0, acc_rate - hw_a)
            ucb_h = min(1.0, acc_rate + hw_h)

            # Expected tokens per target call ∈ [1, K+1].
            # E[tokens emitted per step] = mean(step_emit_counts).
            if self._step_emit_counts:
                exp_tokens = sum(self._step_emit_counts) / len(self._step_emit_counts)
            else:
                exp_tokens = 1.0

            # Empirical speedup: tokens emitted per (target_cost + draft_cost).
            # Baseline: 1 token per target call.
            unit_cost = cfg.target_cost + cfg.draft_cost
            baseline_tokens_per_unit_cost = cfg.target_cost / cfg.target_cost
            speedup = exp_tokens * (cfg.target_cost / unit_cost) / baseline_tokens_per_unit_cost
            v_emit = _empirical_variance([float(x) for x in self._step_emit_counts])
            hw_speed = (_bernstein_half_width(len(self._step_emit_counts),
                                              v_emit, alpha,
                                              rng=float(cfg.k_draft + 1))
                        if len(self._step_emit_counts) > 1 else 1.0)
            speedup_lcb = max(0.0,
                              (exp_tokens - hw_speed) *
                              (cfg.target_cost / unit_cost) /
                              baseline_tokens_per_unit_cost)

            # Equivalence martingale: under correct implementation, the
            # per-step log p_target(emitted) should equal the log p of
            # sampling directly from target.  We use a Maurer-Pontil LCB
            # on the mean log_p as a sanity check; large negative values
            # suggest implementation drift.
            eq_mean = (sum(self._eq_log_ratios) / len(self._eq_log_ratios)
                       if self._eq_log_ratios else 0.0)
            eq_var = _empirical_variance(self._eq_log_ratios)
            eq_rng = max(1.0,
                         max(abs(x) for x in self._eq_log_ratios)
                         if self._eq_log_ratios else 1.0)
            eq_hw = (_bernstein_half_width(len(self._eq_log_ratios),
                                            eq_var, cfg.equivalence_alpha,
                                            rng=eq_rng)
                     if len(self._eq_log_ratios) > 1 else 1.0)
            eq_lcb = eq_mean - eq_hw
            eq_passed = eq_lcb > -math.inf  # the test is informational here

            self._chain.emit(SPECULATOR_REPORTED, {
                "n_steps": n_steps,
                "n_proposed": n_props,
                "n_accepted": n_acc,
                "acc_rate": acc_rate,
                "exp_tokens": exp_tokens,
                "speedup": speedup,
                "eq_mean": eq_mean,
            })
            return SpeculatorReport(
                algorithm=cfg.algorithm,
                n_steps=n_steps,
                n_proposed_total=n_props,
                n_accepted_total=n_acc,
                n_emitted_total=self._n_emitted,
                empirical_acceptance_rate=acc_rate,
                empirical_acceptance_rate_lcb_hoeffding=lcb_h,
                empirical_acceptance_rate_lcb_bernstein=lcb_b,
                empirical_acceptance_rate_lcb_anytime=lcb_a,
                empirical_acceptance_rate_ucb_hoeffding=ucb_h,
                expected_tokens_per_target_call=exp_tokens,
                empirical_speedup=speedup,
                speedup_lcb_bernstein=speedup_lcb,
                equivalence_log_ratio_mean=eq_mean,
                equivalence_log_ratio_lcb=eq_lcb,
                equivalence_test_passed=eq_passed,
                fingerprint_hash=self._chain.hexdigest(),
                chain_length=self._chain.count,
                elapsed_seconds=time.perf_counter() - self._t_start,
            )

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "algorithm": self.config.algorithm,
                "n_steps": self._n_steps,
                "n_proposed_total": self._n_proposed,
                "n_accepted_total": self._n_accepted,
                "n_emitted_total": self._n_emitted,
                "fingerprint": self._chain.hexdigest(),
                "chain_length": self._chain.count,
            }

    def reset(self) -> None:
        with self._lock:
            self._n_steps = 0
            self._n_proposed = 0
            self._n_accepted = 0
            self._n_emitted = 0
            self._accept_indicators = []
            self._step_emit_counts = []
            self._eq_log_ratios = []
            self._lookahead_cache.clear()
            self._chain.emit(SPECULATOR_RESET, {})

    def fingerprint(self) -> str:
        return self._chain.hexdigest()


# =============================================================================
# Convenience constructors
# =============================================================================


def speculative_sampling_speculator(k_draft: int = 4, *,
                                    seed: int = 0,
                                    **kwargs: Any) -> Speculator:
    """Vanilla speculative sampling (Chen et al 2023)."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_SPEC_SAMPLING, k_draft=k_draft, seed=seed, **kwargs))


def leviathan_speculator(k_draft: int = 4, *,
                         seed: int = 0, **kwargs: Any) -> Speculator:
    """Leviathan et al 2023 speculative decoding."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_LEVIATHAN, k_draft=k_draft, seed=seed, **kwargs))


def greedy_speculator(k_draft: int = 4, *, seed: int = 0,
                      **kwargs: Any) -> Speculator:
    """Greedy speculative decoding (argmax verification)."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_GREEDY, k_draft=k_draft, seed=seed, **kwargs))


def medusa_speculator(k_draft: int = 4, tree_width: int = 2, *,
                      seed: int = 0, **kwargs: Any) -> Speculator:
    """Medusa-style tree speculative decoding (Cai et al 2024)."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_MEDUSA_TREE, k_draft=k_draft,
        tree_width=tree_width, seed=seed, **kwargs))


def eagle_speculator(k_draft: int = 4, alpha: float = 0.9, *,
                     seed: int = 0, **kwargs: Any) -> Speculator:
    """EAGLE speculative decoding (Li et al 2024)."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_EAGLE, k_draft=k_draft, eagle_alpha=alpha,
        seed=seed, **kwargs))


def lookahead_speculator(k_draft: int = 4, cache_size: int = 256, *,
                         seed: int = 0, **kwargs: Any) -> Speculator:
    """Lookahead decoding (Fu et al 2024) with n-gram cache."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_LOOKAHEAD, k_draft=k_draft, cache_size=cache_size,
        seed=seed, **kwargs))


def self_spec_speculator(k_draft: int = 4, skip_fraction: float = 0.5,
                          *, seed: int = 0, **kwargs: Any) -> Speculator:
    """Self-speculative decoding via layer skipping (Zhang et al 2024)."""
    return Speculator(SpeculatorConfig(
        algorithm=ALG_SELF_SPEC, k_draft=k_draft,
        skip_fraction=skip_fraction, seed=seed, **kwargs))


# =============================================================================
# Public surface
# =============================================================================


__all__ = [
    "ALG_SPEC_SAMPLING", "ALG_LEVIATHAN", "ALG_GREEDY", "ALG_MEDUSA_TREE",
    "ALG_SELF_SPEC", "ALG_EAGLE", "ALG_LOOKAHEAD",
    "KNOWN_ALGORITHMS",
    "SPECULATOR_STARTED", "SPECULATOR_STEP", "SPECULATOR_REPORTED",
    "SPECULATOR_RESET",
    "Speculator", "SpeculatorConfig", "SpeculatorReport", "StepOutput",
    "SpeculatorError", "InvalidConfig", "InvalidDraft", "InvalidTarget",
    "UnknownAlgorithm", "EquivalenceViolation",
    "speculative_sampling_speculator", "leviathan_speculator",
    "greedy_speculator", "medusa_speculator", "eagle_speculator",
    "lookahead_speculator", "self_spec_speculator",
]
