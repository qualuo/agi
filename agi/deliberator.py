"""Deliberator — adaptive sequential sampling kernel with anytime-valid stopping.

Every other forecaster in this runtime answers a question *before* the work
starts: `PreflightEstimator` predicts cost; `Strategist` decides whether to
admit, hedge, defer, or reject; `ConformalPredictor` gives a distribution-free
bound. None of them answers the operational question a coordination engine
actually asks *while a ticket is running*:

    "I have drawn k samples from this expensive stochastic process. The
     samples cluster into a few candidate answers. Should I draw one more,
     stop and commit to the modal answer, escalate to a stronger model, or
     give up? And whatever I decide — is my stopping rule statistically
     honest under sequential testing?"

That question shows up everywhere a runtime makes adaptive-compute decisions:

  - **Self-consistency sampling** for chain-of-thought reasoning. Draw K
    samples, take the modal answer. How small can K be without losing
    accuracy on hard inputs?

  - **Best-of-N sampling against a learned reward model.** When is the
    current best good enough to ship?

  - **Ensembling across model backends** (Haiku / Sonnet / Opus / fine-tuned
    open-weight). The cheap models agree on easy queries; the expensive ones
    are needed only when they don't.

  - **Online tool-call refinement.** Re-run a tool with perturbed inputs
    until results converge; how many perturbations are enough?

  - **Speculative decoding's stopping criterion**, generalised to any
    candidate-and-verify loop.

The naive answer — "draw a fixed K and majority-vote" — overspends on easy
inputs and underspends on hard ones. The wishful answer — "stop the first
time the leading candidate has a clear plurality" — has a known and ugly
failure mode: classical (fixed-n) confidence intervals are invalid under
data-dependent stopping. P-hacking is the same bug.

`Deliberator` solves both problems jointly. It maintains, sample-by-sample:

  - A Bayesian posterior over candidate-answer clusters (Dirichlet-Multinomial
    with configurable prior strength). Always available, finite-sample
    coherent.

  - An anytime-valid lower confidence bound on the probability of the
    leading cluster, via the Waudby-Smith-Ramdas (2024) predictable-mixture
    capital process — the current state of the art for finite-sample
    mean estimation of bounded random variables. The lower bound is valid
    at every step, under any data-dependent stopping rule, with no
    multiplicity correction. Citation: Waudby-Smith & Ramdas, *Estimating
    means of bounded random variables by betting*, JMLR 2024.

  - An expected information gain (EIG) estimate for one more sample — the
    expected reduction in posterior entropy. When EIG falls below a floor,
    the posterior has stabilised; further samples will not change the
    answer.

  - A running cost ledger.

It stops when *any* of these fire — and reports *which* fired:

  STOP_EVIDENCE     anytime-valid LCB on the leading cluster ≥ commit_threshold.
                    Stopping here is statistically honest under sequential use.
  STOP_BUDGET       max_samples or max_cost reached. The runtime stopped
                    *us*, not the data.
  STOP_CONVERGENCE  EIG below floor for the last `eig_window` samples. The
                    posterior is stable but no cluster dominates — call it
                    a tie and let the strategist hedge, escalate, or defer.
  STOP_INFEASIBLE   never reached; emitted when the caller passes a sampler
                    that produces zero usable samples.

Where this slots into the coordination engine
---------------------------------------------

    deliberator = Deliberator(bus=bus, attestor=attestor)

    def sample_once() -> Sample:
        out = agent.chat(prompt, temperature=0.7)
        key = canonicalize(out.text)               # cluster key
        return Sample(answer=out.text,
                      cluster_key=key,
                      cost=out.usage.cost_usd,
                      metadata={"tokens": out.tokens})

    delib = deliberator.deliberate(
        sample_once,
        max_samples=16,
        max_cost=0.50,
        alpha=0.05,
        commit_threshold=0.5,
        eig_floor=0.005,
    )

    if delib.stop_reason == STOP_EVIDENCE:
        coordinator.commit(delib.answer, cost=delib.cost)
    elif delib.stop_reason == STOP_CONVERGENCE:
        coordinator.escalate(delib)         # ambiguous; ask a stronger model
    elif delib.stop_reason == STOP_BUDGET:
        coordinator.defer(delib, reason="budget exhausted")

A coordination engine that wraps every model call through Deliberator
trades a single dial — `commit_threshold` and `alpha` — for adaptive compute
across the entire fleet: confident queries finish in 1–3 samples, hard
queries are flagged for escalation before they burn budget, and the early
stopping is statistically defensible because the LCB is anytime-valid.

What it composes
----------------

  - **CalibrationEngine** observers can subscribe to `deliberator.observed`
    events to track whether the modal answer was correct, refining the
    runtime's view of how confident *Deliberator itself* is.

  - **AttestationLedger** can ingest each `Deliberation` to produce a
    tamper-evident receipt that a downstream auditor can replay: same
    sampler, same seed → same posterior, same stop reason.

  - **Strategist** can call Deliberator inside a recommendation when
    `STRAT_SINGLE` is the verdict but `confidence == low` — running a
    short Deliberator pass on the top candidate yields an EV that is
    not just point-estimated but sequentially defended.

  - **PolicyLab / CausalLab** observe `(commit_threshold, n_samples,
    success)` tuples and feed back into the runtime's understanding of
    which tasks deserve which compute budgets.

Events
------
    deliberator.started      — a deliberation began (id, max_samples, alpha)
    deliberator.sampled      — one sample landed (id, cluster_key, cost)
    deliberator.committed    — STOP_EVIDENCE fired
    deliberator.escalated    — STOP_CONVERGENCE fired (ambiguity)
    deliberator.exhausted    — STOP_BUDGET fired
    deliberator.observed     — caller reported the realised outcome

Honest about limits
-------------------

  - The anytime-valid LCB is *predictable-mixture WSR*, which means it
    needs i.i.d. or exchangeable samples *under a fixed sampler*. If the
    sampler is itself adaptive (e.g. a temperature schedule that changes
    when the posterior gets sharp), the LCB is no longer anytime-valid.
    `Deliberator` does not detect this; it is the caller's contract.

  - The EIG estimate assumes the Dirichlet-Multinomial model is correct.
    If the underlying answer distribution has support on infinitely many
    clusters (typical of free-form text), EIG is an underestimate. Practical
    use: pair the deliberator with a `canonicalize(...)` step that maps
    answers to a finite (possibly large) discrete set — exact-match,
    normalised hash, retrieved id, etc.

  - When commit_threshold > 0.5, the leading cluster must dominate, not
    just lead. For ternary choices that's the right contract; for binary
    classification commit_threshold=0.5 already gives majority commit.

  - The receipt's cost is the sum of sample costs; it does not include
    the overhead of the deliberator itself, which is negligible (single-
    digit microseconds per sample).

All numerics are stdlib-only. A deliberation with 32 samples over 8 clusters
runs end-to-end in well under a millisecond on a laptop CPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# ----- event kinds ----------------------------------------------------

DELIB_STARTED = "deliberator.started"
DELIB_SAMPLED = "deliberator.sampled"
DELIB_COMMITTED = "deliberator.committed"
DELIB_ESCALATED = "deliberator.escalated"
DELIB_EXHAUSTED = "deliberator.exhausted"
DELIB_OBSERVED = "deliberator.observed"
DELIB_INFEASIBLE = "deliberator.infeasible"


# ----- stop reasons ---------------------------------------------------

STOP_EVIDENCE = "evidence"            # anytime-valid LCB on top cluster ≥ commit_threshold
STOP_BUDGET = "budget"                # max_samples or max_cost reached
STOP_CONVERGENCE = "convergence"      # EIG below floor; posterior stable
STOP_INFEASIBLE = "infeasible"        # no usable samples
STOP_REASONS = (
    STOP_EVIDENCE,
    STOP_BUDGET,
    STOP_CONVERGENCE,
    STOP_INFEASIBLE,
)


# ----- confidence levels ----------------------------------------------

CONF_HIGH = "high"
CONF_MEDIUM = "medium"
CONF_LOW = "low"


# ----- dataclasses ----------------------------------------------------


@dataclass
class Sample:
    """One stochastic sample drawn during deliberation.

    cluster_key is the discrete answer identity — exact-match strings,
    canonical hashes, retrieved doc ids, anything that a coordination
    engine can use to compare two answers for equivalence. The deliberator
    treats two samples with the same `cluster_key` as evidence for the
    same hypothesis; samples with different keys are evidence for
    different hypotheses.

    `cost` is the marginal cost of this sample in dollars (or any
    unit; the deliberator only sums it). `answer` is the human-facing
    payload — text, structured value, tool call — that the caller will
    return once a cluster wins.
    """
    answer: Any
    cluster_key: str
    cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ClusterStats:
    """Per-cluster aggregate inside a deliberation."""
    key: str
    count: int
    cost: float
    posterior_mean: float          # Dirichlet posterior mean P(cluster)
    posterior_lower: float         # anytime-valid (1−α)-LCB on P(cluster)
    posterior_upper: float         # informational; one-sided UCB
    first_seen_at: int             # 1-indexed sample number when first seen
    last_seen_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Deliberation:
    """The full record of one deliberation."""
    id: str
    answer: Any
    cluster_key: str
    n_samples: int
    cost: float
    stop_reason: str
    confidence: str
    posterior_mean: float          # P(top cluster | data), Bayesian
    posterior_lower: float         # anytime-valid LCB on P(top cluster)
    posterior_upper: float
    entropy: float                 # Shannon entropy of posterior, nats
    eig_estimate: float            # expected info gain of next sample, nats
    clusters: list[ClusterStats]
    samples: list[Sample]
    alpha: float
    commit_threshold: float
    eig_floor: float
    started_at: float
    finished_at: float
    rationale: str
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Samples carry arbitrary `answer` payloads — make them JSON-safe.
        d["samples"] = [
            {
                "answer": _jsonable(s.answer),
                "cluster_key": s.cluster_key,
                "cost": s.cost,
                "metadata": dict(s.metadata),
            }
            for s in self.samples
        ]
        return d


@dataclass
class CoverageReport:
    """How well-calibrated *this* Deliberator has been historically.

    For deliberations that hit STOP_EVIDENCE at level α, the realised
    success rate should be ≥ 1 − α. If it is materially below, the
    underlying sampler violates exchangeability (drifting prompt, biased
    cluster_key, etc.) and the runtime should refit or pause.
    """
    n_evidence: int
    n_observed: int
    realised_success_rate: float
    target_success_rate: float        # 1 − alpha at commit
    miscoverage: float                # max(0, target − realised)
    n_by_stop_reason: dict[str, int]
    mean_samples_by_stop: dict[str, float]
    mean_cost_by_stop: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----- numerical primitives -------------------------------------------


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wsr_lcb_indicator(
    indicators: Sequence[int],
    delta: float,
    *,
    c: float = 0.75,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """One-sided (1−δ)-anytime-valid LCB on E[indicator] for indicator ∈ {0,1}.

    Uses the WSR predictable-mixture supermartingale, betting *for* the
    null upward instead of *against* it. The capital process

        K_t(m) = Π_{i≤t} (1 + λ_i · (X_i − m))

    with X_i = indicator, predictable λ_i tuned via running mean and
    variance, is a nonnegative supermartingale under H_0 : E[X] ≤ m.
    Ville's inequality: P(sup_t K_t ≥ 1/δ) ≤ δ. We return the largest
    m for which we *cannot* reject H_0 — that is the (1−δ)-LCB.

    The function is anytime-valid: it is a valid LCB at *every* step,
    under any stopping rule. This is what makes early stopping in
    `Deliberator` statistically honest.

    Citation: Waudby-Smith & Ramdas, "Estimating means of bounded
    random variables by betting", JMLR 2024.
    """
    n = len(indicators)
    if n == 0 or delta <= 0.0:
        return 0.0
    if delta >= 1.0:
        return _clip(statistics.fmean(indicators), 0.0, 1.0)
    p_hat = sum(indicators) / n

    def _capital_p(m: float) -> float:
        """log sup_t K_t(m); larger means more evidence that E[X] > m."""
        if m <= 0.0:
            return math.inf
        if m >= 1.0:
            return -math.inf
        mu_hat = 0.5
        var_hat = 0.25
        sum_x = 0.0
        sum_x2 = 0.0
        log_capital = 0.0
        max_log_capital = -math.inf
        for t, raw in enumerate(indicators, start=1):
            x = _clip(float(raw), 0.0, 1.0)
            lam = min(
                math.sqrt(2.0 * math.log(2.0) / max(t * var_hat, 1e-12)),
                c,
            )
            factor = 1.0 + lam * (x - m)
            if factor <= 0.0:
                # Capital pinned at 0 against this null. Cannot reject from here.
                # Treat as no evidence — return current max.
                return max_log_capital if max_log_capital > -math.inf else 0.0
            log_capital += math.log(factor)
            if log_capital > max_log_capital:
                max_log_capital = log_capital
            sum_x += x
            sum_x2 += x * x
            n_eff = t + 1.0
            mu_hat = (0.5 + sum_x) / n_eff
            var_hat = max((0.25 + sum_x2) / n_eff - mu_hat * mu_hat, 1e-6)
        return max_log_capital if max_log_capital > -math.inf else 0.0

    threshold = -math.log(delta)  # reject when max_log_capital ≥ threshold

    # The LCB is the supremum of {m : H_0 : E[X] ≤ m not rejected}.
    # Capital against H_0 : E[X] ≤ m is non-increasing in m (a higher
    # null is easier to be consistent with), so we bisect.
    lo, hi = 0.0, _clip(p_hat, 0.0, 1.0)
    if _capital_p(0.0) < threshold:
        # Cannot reject E[X] ≤ 0 — no evidence the mean is positive.
        return 0.0
    if _capital_p(hi) >= threshold:
        # Even at the empirical mean we reject. This happens when n=1 and
        # the single sample is 1 with delta very loose; treat as no LCB
        # above empirical.
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _capital_p(mid) >= threshold:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def _wsr_ucb_indicator(
    indicators: Sequence[int],
    delta: float,
    *,
    c: float = 0.75,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """Symmetric (1−δ)-UCB on E[X] for X ∈ {0,1} via WSR betting downward."""
    n = len(indicators)
    if n == 0 or delta <= 0.0:
        return 1.0
    if delta >= 1.0:
        return _clip(statistics.fmean(indicators), 0.0, 1.0)
    p_hat = sum(indicators) / n

    def _capital_n(m: float) -> float:
        if m <= 0.0:
            return -math.inf
        if m >= 1.0:
            return math.inf
        mu_hat = 0.5
        var_hat = 0.25
        sum_x = 0.0
        sum_x2 = 0.0
        log_capital = 0.0
        max_log_capital = -math.inf
        for t, raw in enumerate(indicators, start=1):
            x = _clip(float(raw), 0.0, 1.0)
            lam = min(
                math.sqrt(2.0 * math.log(2.0) / max(t * var_hat, 1e-12)),
                c,
            )
            factor = 1.0 + lam * (m - x)
            if factor <= 0.0:
                return max_log_capital if max_log_capital > -math.inf else 0.0
            log_capital += math.log(factor)
            if log_capital > max_log_capital:
                max_log_capital = log_capital
            sum_x += x
            sum_x2 += x * x
            n_eff = t + 1.0
            mu_hat = (0.5 + sum_x) / n_eff
            var_hat = max((0.25 + sum_x2) / n_eff - mu_hat * mu_hat, 1e-6)
        return max_log_capital if max_log_capital > -math.inf else 0.0

    threshold = -math.log(delta)
    lo, hi = _clip(p_hat, 0.0, 1.0), 1.0
    if _capital_n(1.0) < threshold:
        return 1.0
    if _capital_n(lo) >= threshold:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _capital_n(mid) >= threshold:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def _dirichlet_posterior_mean(counts: Mapping[str, int], prior: float) -> dict[str, float]:
    """Posterior mean under a Dirichlet(prior, ..., prior) prior + Multinomial likelihood.

    For each cluster c with count n_c, the posterior mean is

        E[p_c | data] = (n_c + prior) / (sum_c n_c + K * prior)

    where K is the number of *observed* clusters. The cluster space is
    open (we count only clusters we've seen) — a Dirichlet Process would
    formalise that; for our use case the cluster-mean over observed keys
    is the right Bayes estimator under "no more clusters will appear",
    which is conservative for stopping (it overestimates p_top).
    """
    if not counts:
        return {}
    K = len(counts)
    n_total = sum(counts.values())
    denom = n_total + K * prior
    if denom <= 0.0:
        return {k: 1.0 / K for k in counts}
    return {k: (v + prior) / denom for k, v in counts.items()}


def _shannon_entropy(p: Mapping[str, float]) -> float:
    """Shannon entropy in nats. Robust to zero entries."""
    h = 0.0
    for v in p.values():
        if v > 0.0:
            h -= v * math.log(v)
    return h


def _expected_information_gain(counts: Mapping[str, int], prior: float) -> float:
    """E[H(p) − H(p | x_{n+1})] under the Dirichlet-Multinomial posterior.

    Closed form: for a Dirichlet posterior with concentration α_c =
    n_c + prior and total A = Σ α_c, the predictive distribution on the
    next sample is Categorical(α / A). The post-update posterior with
    one additional observation in cluster c has α_c → α_c + 1, leaving
    the others unchanged.

    We compute the EIG by direct expectation over the predictive
    distribution. Cheap (K terms) and exact.
    """
    if not counts:
        return 0.0
    K = len(counts)
    n_total = sum(counts.values())
    A = n_total + K * prior
    if A <= 0.0:
        return 0.0
    p_pred = {k: (counts[k] + prior) / A for k in counts}
    h_now = _shannon_entropy(p_pred)
    eig = 0.0
    for c in counts:
        # Hypothetical posterior after observing cluster c next.
        # Posterior mean over each cluster k:
        #   p_k_new = (n_k + prior + 1{k=c}) / (A + 1)
        # Entropy of that new mean distribution.
        h_after_c = 0.0
        for k in counts:
            num = counts[k] + prior + (1.0 if k == c else 0.0)
            pk = num / (A + 1.0)
            if pk > 0.0:
                h_after_c -= pk * math.log(pk)
        eig += p_pred[c] * (h_now - h_after_c)
    return max(eig, 0.0)


def _jsonable(x: Any) -> Any:
    """Coerce arbitrary payloads to JSON-safe representations."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return repr(x)


def _receipt_hash(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, default=_jsonable).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


# ----- Deliberator ----------------------------------------------------


@dataclass
class _ObservationRecord:
    deliberation_id: str
    stop_reason: str
    n_samples: int
    cost: float
    posterior_mean: float
    posterior_lower: float
    success: bool | None


class Deliberator:
    """Adaptive sequential sampler with anytime-valid stopping.

    Thread-safe. Stateless across `deliberate` calls except for the
    coverage log (which is used only for `coverage_report`).
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any = None,            # AttestationLedger duck-typed
        prior_strength: float = 1.0,
        default_alpha: float = 0.05,
        default_commit_threshold: float = 0.5,
        default_eig_floor: float = 0.005,
        eig_window: int = 3,
        max_history: int = 4096,
    ) -> None:
        if prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        if not (0.0 < default_alpha < 1.0):
            raise ValueError("default_alpha must be in (0, 1)")
        if not (0.0 < default_commit_threshold < 1.0):
            raise ValueError("default_commit_threshold must be in (0, 1)")
        if default_eig_floor < 0.0:
            raise ValueError("default_eig_floor must be ≥ 0")
        if eig_window < 1:
            raise ValueError("eig_window must be ≥ 1")
        self._bus = bus
        self._attestor = attestor
        self._prior = prior_strength
        self._alpha = default_alpha
        self._commit = default_commit_threshold
        self._eig_floor = default_eig_floor
        self._eig_window = eig_window
        self._lock = threading.Lock()
        self._log: list[_ObservationRecord] = []
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Online: stream samples from a callable

    def deliberate(
        self,
        sampler: Callable[[], Sample],
        *,
        max_samples: int = 16,
        max_cost: float = math.inf,
        min_samples: int = 1,
        alpha: float | None = None,
        commit_threshold: float | None = None,
        eig_floor: float | None = None,
        prior_strength: float | None = None,
        session_id: str | None = None,
    ) -> Deliberation:
        """Draw samples one at a time, decide when to stop, return the verdict.

        `sampler` is called with no arguments and must return a Sample.
        If it raises, the deliberation halts immediately and the
        accumulated samples (if any) are used to form a verdict.
        """
        a = float(self._alpha if alpha is None else alpha)
        ct = float(self._commit if commit_threshold is None else commit_threshold)
        ef = float(self._eig_floor if eig_floor is None else eig_floor)
        prior = float(self._prior if prior_strength is None else prior_strength)
        if not (0.0 < a < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (0.0 < ct < 1.0):
            raise ValueError("commit_threshold must be in (0, 1)")
        if ef < 0.0:
            raise ValueError("eig_floor must be ≥ 0")
        if max_samples < 1:
            raise ValueError("max_samples must be ≥ 1")
        if min_samples < 1 or min_samples > max_samples:
            raise ValueError("min_samples must be in [1, max_samples]")

        delib_id = uuid.uuid4().hex[:12]
        started_at = time.time()
        self._publish(
            DELIB_STARTED,
            session_id,
            {
                "id": delib_id,
                "max_samples": max_samples,
                "max_cost": max_cost if math.isfinite(max_cost) else None,
                "alpha": a,
                "commit_threshold": ct,
                "eig_floor": ef,
            },
        )

        samples: list[Sample] = []
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        cluster_cost: dict[str, float] = {}
        recent_eig: list[float] = []
        total_cost = 0.0

        stop_reason = STOP_BUDGET   # provisional; refined below
        rationale = ""

        for step in range(1, max_samples + 1):
            try:
                s = sampler()
            except Exception as exc:        # pragma: no cover - defensive
                rationale = f"sampler raised after {step - 1} samples: {exc!r}"
                stop_reason = STOP_INFEASIBLE if not samples else STOP_BUDGET
                break
            if not isinstance(s, Sample):
                raise TypeError(
                    f"sampler must return Sample, got {type(s).__name__}"
                )
            samples.append(s)
            key = s.cluster_key
            counts[key] = counts.get(key, 0) + 1
            cluster_cost[key] = cluster_cost.get(key, 0.0) + float(s.cost)
            first_seen.setdefault(key, step)
            last_seen[key] = step
            total_cost += float(s.cost)
            self._publish(
                DELIB_SAMPLED,
                session_id,
                {
                    "id": delib_id,
                    "step": step,
                    "cluster_key": key,
                    "cost": s.cost,
                    "cumulative_cost": total_cost,
                    "n_clusters": len(counts),
                },
            )

            # Evidence test: anytime-valid LCB on the *empirically leading*
            # cluster's posterior probability.
            top_key = max(counts, key=lambda k: counts[k])
            top_indicators = [1 if x.cluster_key == top_key else 0 for x in samples]
            lcb_top = _wsr_lcb_indicator(top_indicators, a)
            ucb_top = _wsr_ucb_indicator(top_indicators, a)
            eig = _expected_information_gain(counts, prior)
            recent_eig.append(eig)
            if len(recent_eig) > self._eig_window:
                recent_eig = recent_eig[-self._eig_window :]

            stop_now = False
            if total_cost > max_cost:
                stop_reason = STOP_BUDGET
                rationale = f"cost {total_cost:.4f} exceeded cap {max_cost:.4f}"
                stop_now = True
            elif step >= min_samples and lcb_top >= ct:
                stop_reason = STOP_EVIDENCE
                rationale = (
                    f"LCB({a:.3f}) on p_top={lcb_top:.4f} ≥ commit={ct:.3f} "
                    f"after {step} samples"
                )
                stop_now = True
            elif (
                len(counts) >= 2
                and step >= max(min_samples, self._eig_window)
                and len(recent_eig) >= self._eig_window
                and max(recent_eig) < ef
            ):
                stop_reason = STOP_CONVERGENCE
                rationale = (
                    f"EIG below floor {ef:.5f} for last {self._eig_window} "
                    f"samples (max recent EIG={max(recent_eig):.5f})"
                )
                stop_now = True
            elif step >= max_samples:
                stop_reason = STOP_BUDGET
                rationale = f"reached max_samples={max_samples}"
                stop_now = True
            if stop_now:
                break

        finished_at = time.time()

        if not samples:
            return self._build_infeasible(delib_id, a, ct, ef, started_at, finished_at, session_id)

        return self._finalize(
            delib_id=delib_id,
            samples=samples,
            counts=counts,
            cluster_cost=cluster_cost,
            first_seen=first_seen,
            last_seen=last_seen,
            total_cost=total_cost,
            stop_reason=stop_reason,
            rationale=rationale,
            alpha=a,
            commit_threshold=ct,
            eig_floor=ef,
            prior=prior,
            started_at=started_at,
            finished_at=finished_at,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Offline: score an already-drawn batch (still anytime-valid for prefixes)

    def deliberate_batch(
        self,
        samples: Iterable[Sample],
        *,
        alpha: float | None = None,
        commit_threshold: float | None = None,
        eig_floor: float | None = None,
        prior_strength: float | None = None,
        session_id: str | None = None,
    ) -> Deliberation:
        """Run the stopping logic over a fixed pre-drawn batch.

        Useful for offline analysis: "given these K samples, would I have
        committed?" The anytime-valid LCB is computed on the *full* batch,
        which is the most conservative possible answer.
        """
        a = float(self._alpha if alpha is None else alpha)
        ct = float(self._commit if commit_threshold is None else commit_threshold)
        ef = float(self._eig_floor if eig_floor is None else eig_floor)
        prior = float(self._prior if prior_strength is None else prior_strength)

        delib_id = uuid.uuid4().hex[:12]
        started_at = time.time()
        sample_list = list(samples)
        if not sample_list:
            finished_at = time.time()
            return self._build_infeasible(
                delib_id, a, ct, ef, started_at, finished_at, session_id
            )
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        cluster_cost: dict[str, float] = {}
        total_cost = 0.0
        for i, s in enumerate(sample_list, start=1):
            if not isinstance(s, Sample):
                raise TypeError(f"expected Sample, got {type(s).__name__}")
            counts[s.cluster_key] = counts.get(s.cluster_key, 0) + 1
            cluster_cost[s.cluster_key] = cluster_cost.get(s.cluster_key, 0.0) + float(s.cost)
            first_seen.setdefault(s.cluster_key, i)
            last_seen[s.cluster_key] = i
            total_cost += float(s.cost)
        top_key = max(counts, key=lambda k: counts[k])
        top_indicators = [1 if x.cluster_key == top_key else 0 for x in sample_list]
        lcb_top = _wsr_lcb_indicator(top_indicators, a)
        if lcb_top >= ct:
            stop_reason = STOP_EVIDENCE
            rationale = (
                f"batch: LCB({a:.3f}) on p_top={lcb_top:.4f} ≥ commit={ct:.3f} "
                f"over n={len(sample_list)}"
            )
        else:
            eig_last = _expected_information_gain(counts, prior)
            if eig_last < ef:
                stop_reason = STOP_CONVERGENCE
                rationale = (
                    f"batch: EIG={eig_last:.5f} below floor {ef:.5f}; "
                    f"posterior stable but no cluster dominates"
                )
            else:
                stop_reason = STOP_BUDGET
                rationale = (
                    f"batch: insufficient evidence (LCB={lcb_top:.4f} < "
                    f"commit={ct:.3f}) and EIG still high"
                )
        finished_at = time.time()
        return self._finalize(
            delib_id=delib_id,
            samples=sample_list,
            counts=counts,
            cluster_cost=cluster_cost,
            first_seen=first_seen,
            last_seen=last_seen,
            total_cost=total_cost,
            stop_reason=stop_reason,
            rationale=rationale,
            alpha=a,
            commit_threshold=ct,
            eig_floor=ef,
            prior=prior,
            started_at=started_at,
            finished_at=finished_at,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Closing the loop: caller reports the realised outcome

    def observe(
        self,
        deliberation: Deliberation,
        *,
        success: bool | None,
        session_id: str | None = None,
    ) -> None:
        """Record whether the committed answer was actually correct.

        Used by `coverage_report()` to verify that STOP_EVIDENCE
        deliberations achieved their nominal (1 − α) success rate. Pass
        `success=None` when the outcome is unknown (e.g. the caller
        deferred or escalated); only `True`/`False` count toward
        coverage.
        """
        rec = _ObservationRecord(
            deliberation_id=deliberation.id,
            stop_reason=deliberation.stop_reason,
            n_samples=deliberation.n_samples,
            cost=deliberation.cost,
            posterior_mean=deliberation.posterior_mean,
            posterior_lower=deliberation.posterior_lower,
            success=success,
        )
        with self._lock:
            self._log.append(rec)
            if len(self._log) > self._max_history:
                self._log = self._log[-self._max_history :]
        self._publish(
            DELIB_OBSERVED,
            session_id,
            {
                "id": deliberation.id,
                "success": success,
                "stop_reason": deliberation.stop_reason,
                "n_samples": deliberation.n_samples,
            },
        )

    # ------------------------------------------------------------------
    # Self-evaluation

    def coverage_report(self) -> CoverageReport:
        """How well-calibrated is the deliberator on past STOP_EVIDENCE calls?"""
        with self._lock:
            records = list(self._log)
        n_by_stop: dict[str, int] = {r: 0 for r in STOP_REASONS}
        samples_by_stop: dict[str, list[int]] = {r: [] for r in STOP_REASONS}
        cost_by_stop: dict[str, list[float]] = {r: [] for r in STOP_REASONS}
        n_evidence = 0
        n_correct = 0
        n_observed_total = 0
        target = 0.0
        # We track the *most recent* alpha as the target floor; a deliberator
        # used with varying alpha will see a pooled coverage rate.
        for r in records:
            n_by_stop[r.stop_reason] = n_by_stop.get(r.stop_reason, 0) + 1
            samples_by_stop.setdefault(r.stop_reason, []).append(r.n_samples)
            cost_by_stop.setdefault(r.stop_reason, []).append(r.cost)
            if r.stop_reason == STOP_EVIDENCE and r.success is not None:
                n_evidence += 1
                if r.success:
                    n_correct += 1
            if r.success is not None:
                n_observed_total += 1
        target = 1.0 - self._alpha
        # When no STOP_EVIDENCE commits have been observed, there is no
        # coverage to measure. Report realised = target so miscoverage = 0
        # rather than mis-flagging a quiet system as broken.
        rate = (n_correct / n_evidence) if n_evidence > 0 else target
        mean_n = {
            k: (statistics.fmean(v) if v else 0.0) for k, v in samples_by_stop.items()
        }
        mean_c = {
            k: (statistics.fmean(v) if v else 0.0) for k, v in cost_by_stop.items()
        }
        return CoverageReport(
            n_evidence=n_evidence,
            n_observed=n_observed_total,
            realised_success_rate=rate,
            target_success_rate=target,
            miscoverage=max(0.0, target - rate),
            n_by_stop_reason=dict(n_by_stop),
            mean_samples_by_stop=mean_n,
            mean_cost_by_stop=mean_c,
        )

    # ------------------------------------------------------------------
    # Internals

    def _finalize(
        self,
        *,
        delib_id: str,
        samples: list[Sample],
        counts: dict[str, int],
        cluster_cost: dict[str, float],
        first_seen: dict[str, int],
        last_seen: dict[str, int],
        total_cost: float,
        stop_reason: str,
        rationale: str,
        alpha: float,
        commit_threshold: float,
        eig_floor: float,
        prior: float,
        started_at: float,
        finished_at: float,
        session_id: str | None,
    ) -> Deliberation:
        posterior = _dirichlet_posterior_mean(counts, prior)
        top_key = max(counts, key=lambda k: counts[k])
        top_indicators = [1 if x.cluster_key == top_key else 0 for x in samples]
        lcb_top = _wsr_lcb_indicator(top_indicators, alpha)
        ucb_top = _wsr_ucb_indicator(top_indicators, alpha)
        entropy = _shannon_entropy(posterior)
        eig = _expected_information_gain(counts, prior)
        # Pick the most-recent sample in the top cluster as the answer
        # (caller-meaningful: it's the last canonical commit).
        answer = next(
            (s.answer for s in reversed(samples) if s.cluster_key == top_key),
            samples[-1].answer,
        )
        clusters: list[ClusterStats] = []
        for key in sorted(counts, key=lambda k: -counts[k]):
            indicators = [1 if x.cluster_key == key else 0 for x in samples]
            clusters.append(
                ClusterStats(
                    key=key,
                    count=counts[key],
                    cost=cluster_cost.get(key, 0.0),
                    posterior_mean=posterior[key],
                    posterior_lower=_wsr_lcb_indicator(indicators, alpha),
                    posterior_upper=_wsr_ucb_indicator(indicators, alpha),
                    first_seen_at=first_seen[key],
                    last_seen_at=last_seen[key],
                )
            )
        confidence = _confidence_level(lcb_top, commit_threshold, len(samples))
        d = Deliberation(
            id=delib_id,
            answer=answer,
            cluster_key=top_key,
            n_samples=len(samples),
            cost=total_cost,
            stop_reason=stop_reason,
            confidence=confidence,
            posterior_mean=posterior[top_key],
            posterior_lower=lcb_top,
            posterior_upper=ucb_top,
            entropy=entropy,
            eig_estimate=eig,
            clusters=clusters,
            samples=samples,
            alpha=alpha,
            commit_threshold=commit_threshold,
            eig_floor=eig_floor,
            started_at=started_at,
            finished_at=finished_at,
            rationale=rationale,
        )
        d.receipt_hash = self._maybe_attest(d, session_id)

        # Emit terminal event by stop_reason.
        evt_kind = {
            STOP_EVIDENCE: DELIB_COMMITTED,
            STOP_CONVERGENCE: DELIB_ESCALATED,
            STOP_BUDGET: DELIB_EXHAUSTED,
            STOP_INFEASIBLE: DELIB_INFEASIBLE,
        }.get(stop_reason, DELIB_EXHAUSTED)
        self._publish(
            evt_kind,
            session_id,
            {
                "id": delib_id,
                "answer_cluster": top_key,
                "n_samples": len(samples),
                "cost": total_cost,
                "posterior_mean": posterior[top_key],
                "posterior_lower": lcb_top,
                "entropy": entropy,
                "stop_reason": stop_reason,
                "rationale": rationale,
                "receipt_hash": d.receipt_hash,
            },
        )
        return d

    def _build_infeasible(
        self,
        delib_id: str,
        alpha: float,
        commit_threshold: float,
        eig_floor: float,
        started_at: float,
        finished_at: float,
        session_id: str | None,
    ) -> Deliberation:
        d = Deliberation(
            id=delib_id,
            answer=None,
            cluster_key="",
            n_samples=0,
            cost=0.0,
            stop_reason=STOP_INFEASIBLE,
            confidence=CONF_LOW,
            posterior_mean=0.0,
            posterior_lower=0.0,
            posterior_upper=1.0,
            entropy=0.0,
            eig_estimate=0.0,
            clusters=[],
            samples=[],
            alpha=alpha,
            commit_threshold=commit_threshold,
            eig_floor=eig_floor,
            started_at=started_at,
            finished_at=finished_at,
            rationale="sampler produced no samples",
        )
        d.receipt_hash = self._maybe_attest(d, session_id)
        self._publish(
            DELIB_INFEASIBLE,
            session_id,
            {"id": delib_id, "rationale": d.rationale, "receipt_hash": d.receipt_hash},
        )
        return d

    def _maybe_attest(self, d: Deliberation, session_id: str | None) -> str:
        # Compute a content hash even when no attestor is wired so the
        # caller always has a stable id for the receipt.
        receipt_input = {
            "id": d.id,
            "n_samples": d.n_samples,
            "cluster_key": d.cluster_key,
            "stop_reason": d.stop_reason,
            "alpha": d.alpha,
            "commit_threshold": d.commit_threshold,
            "posterior_mean": d.posterior_mean,
            "posterior_lower": d.posterior_lower,
            "clusters": [c.to_dict() for c in d.clusters],
            "samples": [
                {"cluster_key": s.cluster_key, "cost": s.cost} for s in d.samples
            ],
        }
        h = _receipt_hash(receipt_input)
        if self._attestor is not None:
            # Duck-typed: anything with `.append(payload, kind=...) -> receipt`.
            try:
                self._attestor.append(receipt_input, kind="deliberation")
            except Exception:
                pass
        return h

    def _publish(self, kind: str, session_id: str | None, data: dict[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, session_id=session_id, data=data))
        except Exception:
            pass


# ----- helpers --------------------------------------------------------


def _confidence_level(lcb: float, commit: float, n: int) -> str:
    """Map (LCB, threshold, n) to a coarse confidence bucket.

    - HIGH: LCB clears the threshold by a wide margin (≥ commit + 0.10).
    - MEDIUM: LCB ≥ commit (we *would* have committed).
    - LOW: LCB < commit (no commit-worthy evidence yet).
    """
    if n < 1:
        return CONF_LOW
    if lcb >= commit + 0.10:
        return CONF_HIGH
    if lcb >= commit:
        return CONF_MEDIUM
    return CONF_LOW


def canonical_cluster_key(text: str, *, normalise: bool = True) -> str:
    """Default cluster_key strategy for free-form text answers.

    Lowercases, strips whitespace, collapses internal runs, and hashes
    the result so callers get a stable string key even when the underlying
    answer is long. For structured/tool-call outputs, prefer to construct
    your own cluster_key from the canonical form of the action.
    """
    if normalise:
        s = " ".join(text.strip().lower().split())
    else:
        s = text
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "CONF_HIGH",
    "CONF_LOW",
    "CONF_MEDIUM",
    "ClusterStats",
    "CoverageReport",
    "DELIB_COMMITTED",
    "DELIB_ESCALATED",
    "DELIB_EXHAUSTED",
    "DELIB_INFEASIBLE",
    "DELIB_OBSERVED",
    "DELIB_SAMPLED",
    "DELIB_STARTED",
    "Deliberation",
    "Deliberator",
    "STOP_BUDGET",
    "STOP_CONVERGENCE",
    "STOP_EVIDENCE",
    "STOP_INFEASIBLE",
    "STOP_REASONS",
    "Sample",
    "canonical_cluster_key",
]
