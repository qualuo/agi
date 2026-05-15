r"""Hedger — universal prediction with experts / online learning with
provable regret bounds, as a runtime primitive.

The rest of the runtime's decision primitives — ``Bandit``, ``BayesOpt``,
``Arbiter``, ``PolicyImprover``, ``Strategist``, ``Forecaster``,
``Quantilizer`` — each *choose* an action under their own modelling
assumption (stochastic reward, smooth surrogate, fixed-confidence,
HCPI, …).  Every primitive is right under its own assumption and
*wrong* outside it.  In production no single assumption holds: data
non-stationarity, adversarial perturbation, model misspecification,
horizon uncertainty, and the simple fact that the coordination engine
has *several* candidate primitives competing for the same decision all
break "one primitive, one assumption".

``Hedger`` is the runtime primitive that solves the meta-decision: it
takes a *fixed pool of experts* (each expert being any other primitive,
any model version, any prompt, any decision rule whatsoever) and an
*incoming stream of losses* (the realised cost of each expert's
recommendation, revealed after the round), and returns at every round
a distribution over experts whose cumulative loss tracks the best
expert in hindsight up to a *vanishing per-round regret* — without
ever knowing in advance which expert will turn out to be best, without
needing the losses to be stationary, and without making any
distributional assumption on the loss sequence.

This is the *foundational online-learning* primitive.  It is the
universal aggregator that lets the coordination engine combine every
other primitive's output into a single decision whose worst-case
regret is *provably bounded* by ``O(√(T log N))`` against the best
fixed expert, ``O(√(T log N) + KL(prior, posterior))`` against any
fixed distribution over experts ("PAC-Bayesian aggregation"), and even
``O(√(T (V/L_T*) log N))`` second-order against the best expert when
the losses are well-behaved — all with anytime-valid finite-sample
certificates and tamper-evident receipts.

The pitch reduced to a runtime call::

  hedger = Hedger.create(experts=["bandit", "bayesopt", "thompson"],
                         algorithm=ADAHEDGE, seed=0)
  for t in range(T):
      dist = hedger.predict()              # distribution over experts
      sel = hedger.select()                # sample an expert (hashed)
      ...                                  # downstream code consults sel
      hedger.observe({"bandit": l_t_bandit,
                      "bayesopt": l_t_bayesopt,
                      "thompson": l_t_thompson})
  report = hedger.report()                  # regret bounds + receipts

Every ``observe`` returns a per-round receipt carrying a closed-form
*upper bound* on the cumulative regret of the algorithm against the
best fixed expert, the *anytime* Howard-Ramdas-McAuliffe-Sekhon 2021
confidence sequence on every expert's cumulative loss, the *realised*
KL of the current weights from the prior, and a SHA-256 fingerprint
chain hashing the entire trace into ``AttestationLedger``.

Algorithms shipped
------------------

**Hedge / EWA / Multiplicative Weights** (Vovk 1990 *Aggregating
strategies*; Littlestone-Warmuth 1994 *The Weighted Majority
Algorithm*; Freund-Schapire 1997 *A decision-theoretic generalisation
of on-line learning and an application to boosting*).  Per-expert
weight ``w_t(i) ∝ exp(-η L_{t-1}(i))`` for losses bounded in ``[0, 1]``
and a learning rate ``η > 0``.  The fundamental regret theorem (Vovk
1990; Cesa-Bianchi-Lugosi 2006 Theorem 2.2):

    ``R_T ≤ η T / 8 + log N / η``,

minimised at ``η* = √(8 log N / T)`` to give ``R_T ≤ √(T log N / 2)``.

**AdaHedge** (de Rooij-van Erven-Grünwald-Koolen 2014 *Follow the
Leader if you can, hedge if you must*).  Parameter-free adaptive
learning rate

    ``η_t = log N / Δ_{t-1}``,    ``Δ_t = Δ_{t-1} + δ_t``,

where ``δ_t`` is the *mixability gap* — the realised one-step loss
of Hedge minus the η-weighted mix of expert losses.  Theorem
(de Rooij et al. 2014):

    ``R_T ≤ 2 √(V_T log N) + 16 log N / 3``,

with ``V_T = sum_t δ_t = O(L_T*)`` second-order constant.  AdaHedge
*never loses* against the best fixed η-Hedge — pointwise — and gains
``O(L_T*)`` regret on "easy" sequences where one expert dominates.

**NormalHedge** (Chaudhuri-Freund-Hsu 2009 *A parameter-free hedging
algorithm*).  Anytime, parameter-free, with per-expert regret

    ``R_T(i) ≤ √(2 T (ln(d_T(i) + 1) + ln N))``,

where ``d_T(i)`` is the rank of expert ``i`` from the best. Crucially,
*no learning rate to set* and bounds hold for every ``T`` simultaneously.

**Squint** (Koolen-van Erven 2015 *Second-order quantile methods for
experts and combinatorial games*).  Improper-prior aggregation
algorithm with *quantile-second-order* regret

    ``R_T(K) ≤ O(√(V_T (KL(u_K ‖ π) + log log T))) + log T``,

where ``u_K`` is uniform on the best-K experts and ``V_T = Σ_t (l_t(i*)
− m_t)²`` is a second-order variance of the loss sequence. Squint is
optimal on "K-quantile" benchmarks, dominates AdaHedge whenever a small
top-K of experts is consistently good, and is *adaptive* to both the
KL of the comparator and the second-order variance.

**ML-Prod / Prod** (Cesa-Bianchi-Mansour-Stoltz 2007 *Improved second-
order bounds for prediction with expert advice*).  Polynomial-weighted
update ``w_t(i) ∝ Π_s (1 - η (l_s(i) - 〈l_s, w_s〉))`` giving
second-order regret

    ``R_T ≤ √(8 V_T ln N) + 5 ln N``,

with ``V_T = Σ_t (l_t(i*) - L_t/N)²`` variance against the running mean.

**Follow the Regularized Leader (FTRL)** (Shalev-Shwartz 2007;
Hazan 2019 *Introduction to Online Convex Optimization* §5).  Generic
update

    ``w_{t+1} = argmin_{w ∈ Δ_N} ⟨L_t, w⟩ + R(w) / η``,

with ``R(w)`` an η-strongly-convex regulariser on the simplex.

  * **FTRL-Entropy** = Hedge — the entropic regulariser
    ``R(w) = Σ w_i log w_i`` recovers Hedge / EWA exactly.

  * **FTRL-L2** = projected online gradient descent — the squared-
    Euclidean regulariser ``R(w) = ½ ‖w‖²`` gives Zinkevich 2003
    OGD with regret ``O(√T)``.

Both ship as deterministic, replay-verifiable, hash-chained selectors.

**Follow the Perturbed Leader (FTPL)** (Hannan 1957 *Approximation
to Bayes risk in repeated play*; Kalai-Vempala 2005 *Efficient
algorithms for online decision problems*).  Add IID exponential
perturbations to the cumulative losses and play the argmin.  Regret
``O(√(T log N))`` with the geometric mean property that FTPL is the
*only* family of expert algorithms that works on *combinatorial*
action spaces without an inner LP — a property the coordination engine
needs when the experts are themselves combinatorial primitives.

**Online Mirror Descent (OMD)** (Beck-Teboulle 2003 *Mirror descent
and nonlinear projected subgradient methods for convex
optimization*).  Generic mirror descent on the probability simplex.
Entropic mirror map → Hedge.  Quadratic mirror map → OGD.  Ships
with both maps and bisection-solved Bregman-projection onto the
simplex.

**Specialist / sleeping experts** (Freund-Schapire-Singer-Warmuth
1997 *Using and combining predictors that specialize*).  Some experts
abstain on some rounds; the per-round regret bound becomes a *per-
specialist* regret bound on the rounds where the expert was active.

**Best-of-Both adaptive aggregator (BOA)** (Wintenberger 2017
*Optimal learning with Bernstein online aggregation*).  Combines a
first-order ``√(T log N)`` bound with a second-order ``√(V_T log N)``
bound *simultaneously* — no learning-rate tuning, no horizon
knowledge.

**Doubling trick** (Cesa-Bianchi-Lugosi 2006 §2.3).  Wraps any
horizon-aware algorithm into an anytime variant at the cost of a
constant blow-up of the regret bound by ``√2 / (√2 − 1) ≈ 3.41``.

Anytime certificates
--------------------

Every Hedger emits a ``HedgerReport`` carrying

  * **First-order regret upper bound** in closed form (algorithm-
    specific): Vovk 1990, AdaHedge, NormalHedge, Squint, ML-Prod,
    Wintenberger 2017 — all derived from the respective theorems.

  * **PAC-Bayes regret bound** for any reference distribution ``π``:
    ``R_T(π) ≤ √(T (KL(π ‖ uniform) + log N) / 2)`` (McAllester 1999;
    Catoni 2007 generalisation).

  * **Anytime confidence sequences** on every expert's mean loss via
    Howard-Ramdas-McAuliffe-Sekhon 2021 (mixture-of-supermartingales,
    sub-Gaussian, time-uniform).  Coordinator may stop at any data-
    dependent time and the certificates still hold.

  * **Realised KL** ``KL(w_t ‖ π_0)`` — exact, in nats — relative to
    the prior.  Bounds how far the algorithm has moved from the safe
    starting point.

  * **Empirical Bernstein** (Maurer-Pontil 2009) on per-expert mean
    loss — sharper than Hoeffding when realised variance is small.

  * **Hoeffding** (Hoeffding 1963) on the same — distribution-free,
    finite-sample LCB / UCB.

  * **Tamper-evident fingerprint** — SHA-256 chain over (experts,
    algorithm, eta, seed, every observation, every selection) so an
    external auditor can replay every weight update byte-for-byte.

Composition with the rest of the runtime
----------------------------------------

  * **Bandit / BayesOpt / Arbiter / Strategist** — register each as
    an expert.  Hedger sums their per-round regrets and provides a
    decision whose cumulative loss tracks the *best primitive in
    hindsight*.  This is the universal "meta-bandit" the coordination
    engine wires its decision channels into.

  * **Forecaster** — hedge a panel of probabilistic forecasters under
    a proper scoring rule (log loss, Brier, CRPS).  Vovk 1990's
    aggregating algorithm gives the log-loss case ``R_T ≤ log N``
    (no √T term) — the "constant regret" universal predictor.

  * **PolicyImprover** — the PAC-Bayes regret bound *is* an HCPI-style
    safety gate.  Coordinator can refuse to switch experts unless the
    Hedger's regret bound is below a safety threshold.

  * **Quantilizer** — quantilize over the Hedger's weight distribution
    to bound KL drift from a safe expert baseline.

  * **DriftSentinel** — the mixability gap ``δ_t`` is a martingale
    drift signal under the null "no expert is consistently better";
    a CUSUM on ``δ_t`` detects regime change.

  * **Refuter** — refute claims about expert dominance via the per-
    expert Howard-Ramdas-McAuliffe-Sekhon 2021 anytime confidence
    sequence.

  * **AttestationLedger** — every ``predict`` / ``select`` / ``observe``
    chains into the ledger.

  * **Coordinator** — the natural target.  Every Goal whose execution
    picks among candidate primitives, model versions, prompts, or
    tools is routed through ``Hedger.select()``.  The coordinator
    *learns at runtime* which primitive to call in each situation,
    with bounded regret.

  * **Composer** — a Plan-level Hedger lets the coordinator hedge over
    several candidate Plans with composed reliability bounds; the
    Hedger's KL bound from the prior plan distribution sets the
    safety constant in Composer's PAC certificate.

Numerical conventions
---------------------

  * **Pure stdlib.**  No NumPy.  No SciPy.  Linear-time per-round
    updates.  Log-space weight maintenance avoids overflow.  Inverse-Φ
    via Beasley-Springer-Moro 1995 for Gaussian computations.

  * **Deterministic given seed.**  Every random draw goes through a
    single ``random.Random(seed)`` shared by the round.  Replay
    recovers an identical decision sequence and identical fingerprint.

  * **JSON-canonical event payloads.**  Replay-deterministic
    fingerprint over canonicalised events.

  * **Type discipline.**  Experts are hashable (typically strings,
    ints, or tuples).  Losses are floats; the algorithm clips to a
    coordinator-supplied range ``[lower, upper]`` (default ``[0, 1]``)
    for finite-sample bounds.  Weights are floats in ``[0, 1]`` summing
    to 1 (validated within ``_PROB_TOL``).

References
----------

  * **Vovk, V. (1990)** *Aggregating strategies*. COLT 1990. The
    originating universal aggregator.

  * **Littlestone, N. & Warmuth, M. K. (1994)** *The Weighted
    Majority Algorithm*. Information and Computation 108(2).

  * **Freund, Y. & Schapire, R. E. (1997)** *A decision-theoretic
    generalization of on-line learning and an application to
    boosting*. JCSS 55(1).

  * **Cesa-Bianchi, N. & Lugosi, G. (2006)** *Prediction, Learning,
    and Games*. Cambridge. The standard reference.

  * **Cesa-Bianchi, N., Mansour, Y. & Stoltz, G. (2007)** *Improved
    second-order bounds for prediction with expert advice*. ML
    66(2-3).  ML-Prod.

  * **Chaudhuri, K., Freund, Y. & Hsu, D. (2009)** *A parameter-free
    hedging algorithm*. NIPS 2009. NormalHedge.

  * **Hazan, E. (2019)** *Introduction to Online Convex
    Optimization*. 2nd edition. FnTML.

  * **Kalai, A. & Vempala, S. (2005)** *Efficient algorithms for
    online decision problems*. JCSS 71(3). FTPL.

  * **Koolen, W. M. & van Erven, T. (2015)** *Second-order quantile
    methods for experts and combinatorial games*. COLT 2015. Squint.

  * **de Rooij, S., van Erven, T., Grünwald, P. D. & Koolen, W. M.
    (2014)** *Follow the leader if you can, hedge if you must*.
    JMLR 15.  AdaHedge.

  * **Wintenberger, O. (2017)** *Optimal learning with Bernstein
    online aggregation*. Machine Learning 106(1). BOA.

  * **Hoeffding, W. (1963)** *Probability inequalities for sums of
    bounded random variables*. JASA 58.

  * **Maurer, A. & Pontil, M. (2009)** *Empirical Bernstein Bounds
    and Sample Variance Penalization*. COLT 2009.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021)**
    *Time-uniform, nonparametric, nonasymptotic confidence
    sequences*. Annals of Statistics.

  * **McAllester, D. (1999)** *PAC-Bayesian model averaging*.
    COLT 1999.

  * **Catoni, O. (2007)** *PAC-Bayesian Supervised Classification*.
    IMS Lecture Notes.

Author's contract
-----------------

The Hedger primitive returns *one* of these on every call:

  1. A distribution over experts that, in cumulative loss, tracks the
     best fixed expert up to a vanishing per-round regret, accompanied
     by the closed-form regret bound, the realised KL from the prior,
     and a tamper-evident fingerprint.

  2. A diagnostic: the input losses were out of range, an expert was
     missing from the loss map, or the algorithm encountered a
     numerical edge case — coordinator should re-supply a valid loss
     vector or pick a different algorithm.

The Hedger *never* claims an expert is the best — it claims that *no
fixed expert is doing better than ours by more than the regret bound*.
That is the entire universal-learning contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence


# =====================================================================
# Constants
# =====================================================================

# Algorithm names.
HEDGE = "hedge"                        # Vovk 1990 / Freund-Schapire 1997
ADAHEDGE = "adahedge"                  # de Rooij et al. 2014
NORMAL_HEDGE = "normal_hedge"          # Chaudhuri-Freund-Hsu 2009
SQUINT = "squint"                      # Koolen-van Erven 2015 (improper-prior)
ML_PROD = "ml_prod"                    # Cesa-Bianchi-Mansour-Stoltz 2007
FTRL_ENTROPY = "ftrl_entropy"          # = Hedge
FTRL_L2 = "ftrl_l2"                    # = projected OGD
FTPL = "ftpl"                          # Hannan / Kalai-Vempala
OMD_ENTROPY = "omd_entropy"            # Online Mirror Descent w/ entropic map
BOA = "boa"                            # Wintenberger 2017 Bernstein OA

KNOWN_ALGORITHMS = frozenset({
    HEDGE, ADAHEDGE, NORMAL_HEDGE, SQUINT, ML_PROD,
    FTRL_ENTROPY, FTRL_L2, FTPL, OMD_ENTROPY, BOA,
})

# Bound methods.
HOEFFDING = "hoeffding"
BERNSTEIN = "bernstein"
ANYTIME = "anytime"

KNOWN_BOUND_METHODS = frozenset({HOEFFDING, BERNSTEIN, ANYTIME})

# Numerical guards.
_PROB_TOL = 1.0e-9
_EPS = 1.0e-15
_INF = float("inf")
_LN2 = math.log(2.0)

# Hedge / AdaHedge constants.
_ADAHEDGE_INIT_DELTA = 1.0e-12         # avoid div-by-zero on round 1
_BETA_MAX_FTPL = 1.0e6                 # FTPL perturbation scale cap

# Genesis fingerprint.
_GENESIS = hashlib.sha256(b"hedger.v1.genesis").hexdigest()

# Events emitted on the runtime EventBus.
HEDGER_STARTED = "hedger.started"
HEDGER_PREDICTED = "hedger.predicted"
HEDGER_SELECTED = "hedger.selected"
HEDGER_OBSERVED = "hedger.observed"
HEDGER_REPORT = "hedger.report"
HEDGER_CLEARED = "hedger.cleared"

KNOWN_EVENTS = frozenset({
    HEDGER_STARTED,
    HEDGER_PREDICTED,
    HEDGER_SELECTED,
    HEDGER_OBSERVED,
    HEDGER_REPORT,
    HEDGER_CLEARED,
})


# =====================================================================
# Exceptions
# =====================================================================


class HedgerError(ValueError):
    """Base class for Hedger-domain errors."""


class UnknownAlgorithm(HedgerError):
    """Algorithm name is not in KNOWN_ALGORITHMS."""


class UnknownBoundMethod(HedgerError):
    """Bound method is not in KNOWN_BOUND_METHODS."""


class InvalidExperts(HedgerError):
    """Experts list is empty, has duplicates, or contains unhashable items."""


class InvalidLearningRate(HedgerError):
    """η is not in (0, ∞)."""


class InvalidPrior(HedgerError):
    """Prior is malformed (wrong support, negative, doesn't sum to 1)."""


class InvalidLoss(HedgerError):
    """Loss vector is malformed (missing experts, NaN, out of range)."""


class InvalidLossRange(HedgerError):
    """Loss range [lower, upper] is malformed."""


class InsufficientData(HedgerError):
    """Too few observations for the requested bound."""


class GenericConfigError(HedgerError):
    """Catch-all for misconfigured Hedger state."""


# =====================================================================
# Numerical helpers
# =====================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_log(x: float) -> float:
    return math.log(max(x, _EPS))


def _logsumexp(xs: Sequence[float]) -> float:
    if not xs:
        return -_INF
    m = max(xs)
    if m == -_INF:
        return -_INF
    s = 0.0
    for x in xs:
        s += math.exp(x - m)
    return m + math.log(s)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_json_default, allow_nan=False)


def _json_default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=lambda x: _canonical_json(x))
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(f"non-JSON-serialisable: {type(o).__name__}")


def _hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(_canonical_json(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _expert_key(e: Any) -> str:
    return hashlib.sha256(_canonical_json(e).encode("utf-8")).hexdigest()


def _sorted_pairs(d: Mapping[Any, float]) -> list:
    return sorted(((_expert_key(k), float(v)) for k, v in d.items()),
                  key=lambda kv: kv[0])


# =====================================================================
# Validation
# =====================================================================


def _validate_experts(experts: Sequence[Hashable]) -> tuple:
    if not isinstance(experts, (list, tuple)):
        raise InvalidExperts(
            f"experts must be a list/tuple; got {type(experts).__name__}")
    if len(experts) == 0:
        raise InvalidExperts("experts list is empty")
    seen: set = set()
    out: list = []
    for e in experts:
        try:
            hash(e)
        except TypeError as exc:
            raise InvalidExperts(f"expert {e!r} is not hashable") from exc
        if e in seen:
            raise InvalidExperts(f"expert {e!r} is duplicated")
        seen.add(e)
        out.append(e)
    return tuple(out)


def _validate_prior(prior: Mapping[Any, float] | None,
                    experts: Sequence[Hashable]) -> dict:
    if prior is None:
        n = len(experts)
        return {e: 1.0 / n for e in experts}
    if not isinstance(prior, Mapping):
        raise InvalidPrior(
            f"prior must be a Mapping; got {type(prior).__name__}")
    if set(prior.keys()) != set(experts):
        raise InvalidPrior(
            "prior support must equal experts list")
    out: dict = {}
    total = 0.0
    for e, p in prior.items():
        if not isinstance(p, (int, float)):
            raise InvalidPrior(f"prior[{e!r}] = {p!r} is not numeric")
        pv = float(p)
        if math.isnan(pv) or pv < 0.0:
            raise InvalidPrior(f"prior[{e!r}] = {pv} is negative or NaN")
        out[e] = pv
        total += pv
    if total <= 0.0:
        raise InvalidPrior("prior total mass is zero")
    if abs(total - 1.0) > _PROB_TOL:
        for k in out:
            out[k] /= total
    return out


def _validate_eta(eta: float) -> float:
    if not isinstance(eta, (int, float)):
        raise InvalidLearningRate(f"eta must be numeric; got {type(eta).__name__}")
    ev = float(eta)
    if math.isnan(ev) or ev <= 0.0 or math.isinf(ev):
        raise InvalidLearningRate(f"eta must be in (0, ∞); got {ev}")
    return ev


def _validate_loss_range(lower: float, upper: float) -> tuple[float, float]:
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        raise InvalidLossRange("loss range must be numeric")
    lv, uv = float(lower), float(upper)
    if math.isnan(lv) or math.isnan(uv):
        raise InvalidLossRange("loss range must not be NaN")
    if uv <= lv:
        raise InvalidLossRange(f"upper {uv} must be > lower {lv}")
    return lv, uv


def _validate_loss(losses: Mapping[Any, float],
                   experts: Sequence[Hashable],
                   lower: float, upper: float) -> dict:
    if not isinstance(losses, Mapping):
        raise InvalidLoss(
            f"losses must be a Mapping; got {type(losses).__name__}")
    out: dict = {}
    for e in experts:
        if e not in losses:
            raise InvalidLoss(f"losses[{e!r}] missing")
        l = losses[e]
        if not isinstance(l, (int, float)):
            raise InvalidLoss(f"losses[{e!r}] = {l!r} is not numeric")
        lv = float(l)
        if math.isnan(lv):
            raise InvalidLoss(f"losses[{e!r}] is NaN")
        # We do not raise on out-of-range losses; we clip and surface
        # the realised clip in the round receipt. This is friendlier to
        # the coordinator that may have approximate loss estimates.
        out[e] = lv
    return out


def _validate_delta(delta: float) -> float:
    if not isinstance(delta, (int, float)):
        raise HedgerError(f"delta must be numeric; got {type(delta).__name__}")
    dv = float(delta)
    if math.isnan(dv) or not (0.0 < dv < 1.0):
        raise HedgerError(f"delta must be in (0, 1); got {dv}")
    return dv


def _validate_bound_method(method: str) -> str:
    if method not in KNOWN_BOUND_METHODS:
        raise UnknownBoundMethod(
            f"bound method {method!r} not in {sorted(KNOWN_BOUND_METHODS)}")
    return method


def _validate_algorithm(algorithm: str) -> str:
    if algorithm not in KNOWN_ALGORITHMS:
        raise UnknownAlgorithm(
            f"algorithm {algorithm!r} not in {sorted(KNOWN_ALGORITHMS)}")
    return algorithm


# =====================================================================
# Finite-sample bound helpers
# =====================================================================


def hoeffding_lcb(mean: float, n: int, *, delta: float,
                  lower: float = 0.0, upper: float = 1.0) -> float:
    r"""Hoeffding 1963 LCB on a bounded mean.

    For ``X_i ∈ [lower, upper]``, with probability ≥ 1 − δ::

        E[X] ≥ mean − (upper − lower) √(log(1/δ) / (2 n)).
    """
    if n < 1:
        raise InsufficientData(f"hoeffding_lcb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    lower, upper = _validate_loss_range(lower, upper)
    half = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return mean - half


def hoeffding_ucb(mean: float, n: int, *, delta: float,
                  lower: float = 0.0, upper: float = 1.0) -> float:
    if n < 1:
        raise InsufficientData(f"hoeffding_ucb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    lower, upper = _validate_loss_range(lower, upper)
    half = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return mean + half


def empirical_bernstein_lcb(mean: float, var: float, n: int, *,
                            delta: float, lower: float = 0.0,
                            upper: float = 1.0) -> float:
    r"""Maurer-Pontil 2009 empirical-Bernstein LCB.

    For ``X_i ∈ [lower, upper]``, with probability ≥ 1 − δ::

        E[X] ≥ mean − √(2 σ̂² log(2/δ)/n) − 7(upper − lower) log(2/δ)/(3(n−1)).
    """
    if n < 2:
        raise InsufficientData(
            f"empirical_bernstein_lcb requires n ≥ 2; got {n}")
    delta = _validate_delta(delta)
    lower, upper = _validate_loss_range(lower, upper)
    if var < 0.0:
        raise HedgerError(f"var must be ≥ 0; got {var}")
    L = math.log(2.0 / delta)
    half_var = math.sqrt(2.0 * var * L / n)
    half_range = 7.0 * (upper - lower) * L / (3.0 * (n - 1))
    return mean - half_var - half_range


def empirical_bernstein_ucb(mean: float, var: float, n: int, *,
                            delta: float, lower: float = 0.0,
                            upper: float = 1.0) -> float:
    if n < 2:
        raise InsufficientData(
            f"empirical_bernstein_ucb requires n ≥ 2; got {n}")
    delta = _validate_delta(delta)
    lower, upper = _validate_loss_range(lower, upper)
    if var < 0.0:
        raise HedgerError(f"var must be ≥ 0; got {var}")
    L = math.log(2.0 / delta)
    half_var = math.sqrt(2.0 * var * L / n)
    half_range = 7.0 * (upper - lower) * L / (3.0 * (n - 1))
    return mean + half_var + half_range


def anytime_lcb(mean: float, n: int, *, delta: float,
                lower: float = 0.0, upper: float = 1.0) -> float:
    r"""Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid LCB.

    Mixture-of-supermartingales bound; valid simultaneously for every
    ``n ≥ 1``.  Half-width::

        ψ_n = √( (1 + 1/n) log(2 √(n+1) / δ) / (2 n) ) · (upper - lower).
    """
    if n < 1:
        raise InsufficientData(f"anytime_lcb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    lower, upper = _validate_loss_range(lower, upper)
    L = math.log(2.0 * math.sqrt(n + 1.0) / delta)
    half = (upper - lower) * math.sqrt((1.0 + 1.0 / n) * L / (2.0 * n))
    return mean - half


def anytime_ucb(mean: float, n: int, *, delta: float,
                lower: float = 0.0, upper: float = 1.0) -> float:
    if n < 1:
        raise InsufficientData(f"anytime_ucb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    lower, upper = _validate_loss_range(lower, upper)
    L = math.log(2.0 * math.sqrt(n + 1.0) / delta)
    half = (upper - lower) * math.sqrt((1.0 + 1.0 / n) * L / (2.0 * n))
    return mean + half


# =====================================================================
# Closed-form regret bounds
# =====================================================================


def hedge_regret_bound(T: int, N: int, eta: float, *,
                       loss_range: float = 1.0) -> float:
    r"""Vovk 1990 / Cesa-Bianchi-Lugosi 2006 Theorem 2.2 Hedge regret
    upper bound::

        R_T ≤ η T (loss_range)² / 8 + log N / η.

    Minimised at ``η* = √(8 log N / T) / loss_range`` to give the
    classical ``loss_range · √(T log N / 2)`` bound.
    """
    if T < 0 or N < 1:
        raise HedgerError(f"need T ≥ 0, N ≥ 1; got T={T}, N={N}")
    eta = _validate_eta(eta)
    return eta * T * loss_range * loss_range / 8.0 + math.log(N) / eta


def hedge_minimax_eta(T: int, N: int, *, loss_range: float = 1.0) -> float:
    r"""Optimal η for Hedge minimising the Vovk 1990 bound::

        η* = √(8 log N / T) / loss_range.

    For T = 0 returns 1.0 (degenerate; caller should use the uniform
    distribution).
    """
    if T < 0 or N < 1:
        raise HedgerError(f"need T ≥ 0, N ≥ 1; got T={T}, N={N}")
    if T == 0 or N == 1:
        return 1.0
    return math.sqrt(8.0 * math.log(N) / T) / max(loss_range, _EPS)


def hedge_minimax_regret(T: int, N: int, *, loss_range: float = 1.0) -> float:
    r"""Closed-form minimax regret bound for Hedge::

        R_T ≤ loss_range · √(T log N / 2).
    """
    if T <= 0 or N <= 1:
        return 0.0
    return loss_range * math.sqrt(T * math.log(N) / 2.0)


def adahedge_regret_bound(V_T: float, N: int, *,
                          loss_range: float = 1.0) -> float:
    r"""de Rooij et al. 2014 Theorem 8 AdaHedge regret bound::

        R_T ≤ 2 √(V_T log N) + 16 log N (loss_range) / 3.

    ``V_T = sum_t δ_t`` is the AdaHedge cumulative mixability gap,
    bounded above by the cumulative loss of the best expert times the
    loss range.
    """
    if V_T < 0.0 or N < 1:
        raise HedgerError(f"need V_T ≥ 0, N ≥ 1; got V_T={V_T}, N={N}")
    if N == 1:
        return 0.0
    return (2.0 * math.sqrt(V_T * math.log(N))
            + 16.0 * math.log(N) * loss_range / 3.0)


def normal_hedge_regret_bound(T: int, N: int, rank: int = 1, *,
                              loss_range: float = 1.0) -> float:
    r"""Chaudhuri-Freund-Hsu 2009 NormalHedge per-rank regret bound::

        R_T(rank) ≤ loss_range · √(2 T (log(rank+1) + log N)).

    Holds for every ``T`` simultaneously (anytime) by the time-uniform
    construction.  ``rank=1`` gives the standard bound against the
    best fixed expert.
    """
    if T < 0 or N < 1 or rank < 1:
        raise HedgerError("need T ≥ 0, N ≥ 1, rank ≥ 1")
    if N == 1 or T == 0:
        return 0.0
    return loss_range * math.sqrt(2.0 * T * (math.log(rank + 1.0)
                                              + math.log(N)))


def squint_regret_bound(V_T: float, N: int, K: int = 1, *,
                        loss_range: float = 1.0) -> float:
    r"""Koolen-van Erven 2015 Squint K-quantile second-order bound::

        R_T(K) ≤ loss_range · √(2 V_T (log(N/K) + log log T))
                 + loss_range · log(N/K).

    Approximated with the log-log-T term replaced by a constant 2.0
    for finite T to keep the bound monotone in K.  ``V_T`` is the
    cumulative second-order variance of the comparator's loss.
    """
    if V_T < 0.0 or N < 1 or K < 1 or K > N:
        raise HedgerError("need V_T ≥ 0, 1 ≤ K ≤ N")
    if K == N:
        return 0.0
    log_nk = math.log(N / K)
    return loss_range * math.sqrt(2.0 * V_T * (log_nk + 2.0)) \
           + loss_range * log_nk


def ml_prod_regret_bound(V_T: float, N: int, *,
                         loss_range: float = 1.0) -> float:
    r"""Cesa-Bianchi-Mansour-Stoltz 2007 ML-Prod second-order regret::

        R_T ≤ loss_range · √(8 V_T log N) + 5 loss_range · log N.
    """
    if V_T < 0.0 or N < 1:
        raise HedgerError("need V_T ≥ 0, N ≥ 1")
    if N == 1:
        return 0.0
    return loss_range * math.sqrt(8.0 * V_T * math.log(N)) \
           + 5.0 * loss_range * math.log(N)


def boa_regret_bound(V_T: float, N: int, *,
                     loss_range: float = 1.0) -> float:
    r"""Wintenberger 2017 BOA second-order regret bound (Corollary 1)::

        R_T ≤ loss_range · √(2 V_T (1 + log N))
              + 2 loss_range · (1 + log N).

    The ``(1 + log N)`` term arises from the PAC-Bayesian comparator
    being uniform over a worst-case singleton.  Tighter ``log N``-only
    forms appear in special cases (no-improper-prior, fixed-T) but
    this is the universal bound used for the second-order
    ``√(V_T log N)`` rate that distinguishes BOA from first-order
    Hedge.
    """
    if V_T < 0.0 or N < 1:
        raise HedgerError("need V_T ≥ 0, N ≥ 1")
    if N == 1:
        return 0.0
    lp = 1.0 + math.log(N)
    return loss_range * math.sqrt(2.0 * V_T * lp) \
           + 2.0 * loss_range * lp


def pac_bayes_regret_bound(T: int, prior: Mapping[Any, float],
                           posterior: Mapping[Any, float], *,
                           loss_range: float = 1.0) -> float:
    r"""PAC-Bayes (McAllester 1999; Catoni 2007) regret bound against
    any reference distribution::

        R_T(posterior, prior) ≤ loss_range · √(T (KL(posterior ‖ prior)
                                                  + log(1/δ)) / 2).

    Returns the bound at the canonical δ → 0 sub-Gaussian rate; the
    coordinator may add ``log(1/δ)`` for a specific confidence level.
    """
    if T < 0:
        raise HedgerError(f"T must be ≥ 0; got {T}")
    p = _validate_prior(prior, list(posterior.keys()))
    kl = 0.0
    for e, q in posterior.items():
        if q > 0.0:
            kl += q * math.log(q / max(p[e], _EPS))
    return loss_range * math.sqrt(max(T * kl / 2.0, 0.0))


def kl_divergence(p: Mapping[Any, float], q: Mapping[Any, float]) -> float:
    r"""KL(p ‖ q) over a common support; in nats."""
    if set(p.keys()) != set(q.keys()):
        raise HedgerError("KL requires identical support")
    kl = 0.0
    for k in p:
        pv = float(p[k])
        qv = float(q[k])
        if pv > 0.0:
            kl += pv * math.log(pv / max(qv, _EPS))
    return kl


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class Prediction:
    """A weight distribution over experts at round t."""
    round: int
    weights: dict
    realised_kl_from_prior: float
    entropy: float
    fingerprint: str


@dataclass(frozen=True)
class Selection:
    """A single sampled expert with replay-deterministic fingerprint."""
    round: int
    expert: Any
    weight: float
    seed: int
    fingerprint: str
    parent_fingerprint: str
    timestamp: float


@dataclass(frozen=True)
class Round:
    """One observation step: losses revealed, weights updated."""
    round: int
    losses: dict
    clipped_losses: dict
    weighted_loss: float                  # 〈l, w_t〉
    min_loss: float
    max_loss: float
    mean_loss: float
    realised_eta: float
    delta_mixability_gap: float
    cumulative_mixability_gap: float
    cumulative_loss_by_expert: dict
    cumulative_weighted_loss: float
    best_expert: Any
    best_cumulative_loss: float
    fingerprint: str
    parent_fingerprint: str
    timestamp: float


@dataclass(frozen=True)
class ExpertCertificate:
    """Finite-sample certificates on one expert's mean loss."""
    expert: Any
    n: int
    mean_loss: float
    variance: float
    hoeffding_lcb: float
    hoeffding_ucb: float
    bernstein_lcb: float | None
    bernstein_ucb: float | None
    anytime_lcb: float
    anytime_ucb: float
    delta: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RegretCertificate:
    """Closed-form regret bound applicable at the current round."""
    algorithm: str
    T: int
    N: int
    loss_range: float
    eta: float | None
    first_order_bound: float
    second_order_bound: float | None
    pac_bayes_bound_uniform: float
    realised_regret_so_far: float        # cumulative weighted loss minus best
    best_expert: Any
    best_cumulative_loss: float


@dataclass(frozen=True)
class HedgerReport:
    algorithm: str
    T: int
    N: int
    eta: float | None
    loss_range: tuple
    cumulative_loss_by_expert: dict
    cumulative_weighted_loss: float
    realised_regret_so_far: float
    cumulative_mixability_gap: float
    last_prediction: Prediction | None
    last_selection: Selection | None
    last_round: Round | None
    regret_certificate: RegretCertificate
    fingerprint: str
    genesis: str
    config: dict


# =====================================================================
# AdaHedge η solver
# =====================================================================


def _adahedge_eta(cumulative_delta: float, N: int) -> float:
    r"""AdaHedge learning rate update::

        η_{t} = log N / max(Δ_{t-1}, ε).
    """
    if N <= 1:
        return 1.0
    if cumulative_delta <= _ADAHEDGE_INIT_DELTA:
        cumulative_delta = _ADAHEDGE_INIT_DELTA
    return math.log(N) / cumulative_delta


def _adahedge_mixability_gap(losses: Sequence[float],
                             weights: Sequence[float],
                             eta: float) -> float:
    r"""Per-round mixability gap

        δ_t = 〈l_t, w_t〉 - M_{η_t}(l_t, w_t),

    with the mix-loss ``M_η(l, w) = −(1/η) log Σ w_i exp(−η l_i)`` —
    the η-exponentially-weighted-average. ``δ_t`` is a measure of how
    far the experts disagreed on this round; AdaHedge feeds Σ_t δ_t
    back into the next learning rate.

    The gap is non-negative for losses bounded in [0, 1] and any η,
    by Jensen's inequality applied to the convex function
    ``φ(l) = exp(−η l)``.
    """
    if eta <= 0.0:
        return 0.0
    inner_loss = sum(w * l for w, l in zip(weights, losses))
    # Mix loss in log-space for stability.
    log_terms = []
    for w, l in zip(weights, losses):
        if w > 0.0:
            log_terms.append(math.log(w) - eta * l)
    if not log_terms:
        return 0.0
    log_z = _logsumexp(log_terms)
    mix_loss = -log_z / eta
    gap = inner_loss - mix_loss
    # Numerical guard: gap ≥ 0.
    if gap < 0.0 and gap > -1.0e-9:
        gap = 0.0
    return max(gap, 0.0)


# =====================================================================
# Per-algorithm weight updates
# =====================================================================


def _hedge_weights(log_prior: Mapping[Any, float],
                   cum_losses: Mapping[Any, float],
                   eta: float) -> dict:
    r"""Hedge / EWA weights in log-space::

        w_t(i) ∝ prior(i) · exp(−η L_{t-1}(i)).
    """
    experts = list(log_prior.keys())
    log_unnorm = {e: log_prior[e] - eta * cum_losses[e] for e in experts}
    log_z = _logsumexp(list(log_unnorm.values()))
    return {e: math.exp(log_unnorm[e] - log_z) for e in experts}


def _normal_hedge_weights(regrets: Mapping[Any, float],
                          c: float) -> dict:
    r"""NormalHedge weights (Chaudhuri-Freund-Hsu 2009)::

        w_t(i) ∝ ((R_{t-1}(i))_+ / c) · exp((R_{t-1}(i))_+² / (2 c)),

    with ``c`` solved by bisection to satisfy

        (1/N) Σ_i exp((R_{t-1}(i))_+² / (2 c)) = e.

    If all regrets are non-positive, weights default to uniform.
    """
    pos_regrets = {e: max(r, 0.0) for e, r in regrets.items()}
    if max(pos_regrets.values(), default=0.0) <= 0.0:
        n = len(regrets)
        return {e: 1.0 / n for e in regrets}
    # Bisection on c ∈ [c_lo, c_hi] for the NormalHedge potential.
    n = len(regrets)
    target = math.e

    def potential(c_val: float) -> float:
        s = 0.0
        for r in pos_regrets.values():
            if r > 0.0:
                s += math.exp(r * r / (2.0 * c_val))
            else:
                s += 1.0
        return s / n - target

    r_max = max(pos_regrets.values())
    c_lo = max(r_max * r_max / (2.0 * 50.0), 1.0e-12)     # potential > e
    c_hi = max(r_max * r_max * 100.0, 1.0)
    f_lo = potential(c_lo)
    f_hi = potential(c_hi)
    # Ensure bracketing.
    for _ in range(64):
        if f_lo * f_hi <= 0.0:
            break
        if f_hi > 0.0:
            c_hi *= 2.0
            f_hi = potential(c_hi)
        else:
            c_lo *= 0.5
            f_lo = potential(c_lo)
    for _ in range(128):
        c_mid = 0.5 * (c_lo + c_hi)
        f_mid = potential(c_mid)
        if abs(f_mid) < 1.0e-10 or (c_hi - c_lo) < 1.0e-12:
            break
        if f_mid > 0.0:
            c_lo = c_mid
        else:
            c_hi = c_mid
    c_val = 0.5 * (c_lo + c_hi)
    unnorm = {e: (max(r, 0.0) / c_val) * math.exp(r * r / (2.0 * c_val))
              if r > 0.0 else 0.0
              for e, r in regrets.items()}
    total = sum(unnorm.values())
    if total <= 0.0:
        return {e: 1.0 / n for e in regrets}
    return {e: u / total for e, u in unnorm.items()}


def _squint_integral(R: float, V: float) -> float:
    r"""Closed-form Squint integral

        I(R, V) = ∫_0^{1/2} η exp(η R − η² V) dη.

    Computed in log-space (subtracting the maximum of the exponent on
    [0, 1/2]) for numerical stability.  Falls back to direct Simpson
    quadrature when V is tiny.  Always strictly positive.
    """
    if V > 1.0e-12:
        # Completing the square: η R − η² V = −V (η − μ)² + R²/(4V).
        mu = R / (2.0 * V)
        sqrt_V = math.sqrt(V)
        # Pull max of the exponent out for stability.  Max on [0, 1/2]
        # is at η = clip(μ, 0, 1/2).
        mu_clip = _clip(mu, 0.0, 0.5)
        log_max = mu_clip * R - mu_clip * mu_clip * V
        # Now I(R, V) = exp(log_max) · ∫_0^{1/2} η exp(g(η)) dη with
        # g(η) = η R − η² V − log_max ≤ 0.  Use 64-pt Simpson's rule.
        n = 64
        if n % 2 == 1:
            n += 1
        h = 0.5 / n
        total = 0.0
        for i in range(n + 1):
            eta = i * h
            g = eta * R - eta * eta * V - log_max
            f = eta * math.exp(g)
            if i == 0 or i == n:
                total += f
            elif i % 2 == 1:
                total += 4.0 * f
            else:
                total += 2.0 * f
        total *= h / 3.0
        return max(total, _EPS) * math.exp(log_max)
    # V ≈ 0 → ∫_0^{1/2} η exp(η R) dη — closed form.
    if abs(R) < 1.0e-9:
        return 1.0 / 8.0   # ∫_0^{1/2} η dη
    # ∫_0^{1/2} η e^{η R} dη = [(η R − 1)/R²] e^{η R} from 0 to 1/2.
    a = 0.5
    val = ((a * R - 1.0) * math.exp(a * R) + 1.0) / (R * R)
    return max(val, _EPS)


def _squint_weights(prior: Mapping[Any, float],
                    cum_regrets: Mapping[Any, float],
                    cum_variances: Mapping[Any, float]) -> dict:
    r"""Squint improper-prior weights (Koolen-van Erven 2015)::

        w_t(i) ∝ π(i) · ∫_0^{1/2} η exp(η R_{t-1}(i) − η² V_{t-1}(i)) dη.

    The integral is computed by 64-point Simpson's rule with a
    log-max stabilisation that handles arbitrarily large positive R
    without overflow.  On round 1 (V = 0 for all experts) the integral
    closed-form is used.

    Returns the renormalised posterior over experts.
    """
    unnorm: dict = {}
    for e in prior:
        R = cum_regrets[e]
        V = cum_variances[e]
        unnorm[e] = prior[e] * _squint_integral(R, V)
    total = sum(unnorm.values())
    if total <= 0.0:
        n = len(prior)
        return {e: 1.0 / n for e in prior}
    return {e: u / total for e, u in unnorm.items()}


def _ml_prod_weights(log_w: Mapping[Any, float]) -> dict:
    r"""ML-Prod normalisation: ``w_t(i) = exp(log_w[i]) / Σ_j exp(log_w[j])``."""
    log_z = _logsumexp(list(log_w.values()))
    return {e: math.exp(lw - log_z) for e, lw in log_w.items()}


def _ftrl_l2_weights(prior: Mapping[Any, float],
                     cum_losses: Mapping[Any, float],
                     eta: float) -> dict:
    r"""FTRL with L2 regulariser = projected OGD::

        w = argmin_{w ∈ Δ_N} η ⟨L, w⟩ + ½ ‖w − prior‖²
          = Project_{Δ_N}(prior − η L).

    Projection onto the simplex via the Wang-Carreira-Perpinan 2013
    algorithm (linear-time, exact).
    """
    experts = list(prior.keys())
    v = [prior[e] - eta * cum_losses[e] for e in experts]
    # Project v onto the probability simplex.
    u = sorted(v, reverse=True)
    cssv = 0.0
    rho = 0
    for i, u_i in enumerate(u, start=1):
        cssv += u_i
        if u_i - (cssv - 1.0) / i > 0.0:
            rho = i
    if rho == 0:
        # All entries negative; fall back to argmin only.
        i_min = min(range(len(experts)), key=lambda i: cum_losses[experts[i]])
        return {e: (1.0 if i == i_min else 0.0) for i, e in enumerate(experts)}
    cssv_rho = sum(u[:rho])
    theta = (cssv_rho - 1.0) / rho
    return {e: max(v[i] - theta, 0.0) for i, e in enumerate(experts)}


def _ftpl_weights(prior: Mapping[Any, float],
                  cum_losses: Mapping[Any, float],
                  eta: float,
                  seed: int,
                  draws: int = 64) -> dict:
    r"""Follow-the-Perturbed-Leader weights estimated by Monte-Carlo::

        w_t(i) = Pr_{Z ~ Exp(1)^N} [i = argmin_j (L(j) − Z_j / η)].

    With ``Z_j`` IID standard exponentials.  Closed form for the
    weights doesn't exist in general; we estimate by ``draws`` IID
    perturbations.  Deterministic given seed.

    Kalai-Vempala 2005 prove ``O(√(T log N))`` regret for this scheme
    with the exponential distribution and ``eta = √(log N / T)``.
    """
    rng = random.Random(seed)
    experts = list(prior.keys())
    counts = {e: 0 for e in experts}
    for _ in range(draws):
        best_e = None
        best_score = _INF
        for e in experts:
            # Use −log(uniform) to draw Exp(1).
            u = max(rng.random(), _EPS)
            z = -math.log(u)
            # Prior tilt: place log(prior[e])/eta into the perturbation budget.
            score = cum_losses[e] - z / eta \
                    - _safe_log(prior[e]) / eta
            if score < best_score:
                best_score = score
                best_e = e
        counts[best_e] += 1
    return {e: counts[e] / draws for e in experts}


def _omd_entropy_weights(prior: Mapping[Any, float],
                         cum_losses: Mapping[Any, float],
                         eta: float) -> dict:
    r"""OMD with entropic mirror map.  Coincides with Hedge / EWA."""
    log_prior = {e: _safe_log(prior[e]) for e in prior}
    return _hedge_weights(log_prior, cum_losses, eta)


def _boa_weights(prior: Mapping[Any, float],
                 cum_regrets: Mapping[Any, float],
                 cum_variances: Mapping[Any, float]) -> dict:
    r"""Wintenberger 2017 BOA (Bernstein Online Aggregation) weights::

        w_t(i) ∝ π(i) · exp(η_i · R_{t-1}(i) − η_i² · V_{t-1}(i)),
        η_i = 1 / (2 (loss_range)) · 1 / (1 + log(1 + V_{t-1}(i))).

    Per-expert learning rate adapts to the expert's loss variance with
    the Bernstein-style decay schedule from the paper.  The corrective
    ``− η² V`` term in the exponent is what gives BOA its second-order
    ``√(V_T log N)`` regret rather than first-order ``√(T log N)``.

    Reference: Wintenberger 2017 *Optimal learning with Bernstein
    online aggregation*, Algorithm 1.
    """
    unnorm: dict = {}
    for e, p in prior.items():
        v = cum_variances[e]
        # Wintenberger 2017 Theorem 1's bounded-loss schedule:
        # eta_i = 1 / (2 (1 + log(1 + V))).
        eta_e = 1.0 / (2.0 * (1.0 + math.log1p(max(v, 0.0))))
        # Stabilised exp argument with the corrective Bernstein term.
        arg = eta_e * cum_regrets[e] - eta_e * eta_e * v
        unnorm[e] = p * math.exp(min(arg, 400.0))
    total = sum(unnorm.values())
    if total <= 0.0:
        n = len(prior)
        return {e: 1.0 / n for e in prior}
    return {e: u / total for e, u in unnorm.items()}


# =====================================================================
# Main Hedger class
# =====================================================================


@dataclass(frozen=True)
class HedgerConfig:
    experts: tuple
    algorithm: str
    eta: float | None
    loss_lower: float
    loss_upper: float
    prior: tuple
    seed: int
    bound_delta: float
    ftpl_draws: int
    horizon: int | None


class Hedger:
    r"""Universal prediction-with-experts / online learning primitive.

    Construction
    ------------

    ``Hedger.create(experts, algorithm=HEDGE, ...)`` with:

      * ``experts``: a non-empty sequence of hashable expert names.

      * ``algorithm``: one of ``KNOWN_ALGORITHMS``.

      * ``eta``: learning rate for fixed-η algorithms (``HEDGE``,
        ``FTRL_ENTROPY``, ``FTRL_L2``, ``FTPL``, ``OMD_ENTROPY``,
        ``ML_PROD``).  ``None`` selects the Vovk-1990 minimax-optimal
        ``η* = √(8 log N / T) / loss_range`` if a horizon is supplied,
        else ``η = 1 / loss_range``.

      * ``loss_lower`` / ``loss_upper``: the assumed range of per-round
        per-expert losses.  Defaults to ``[0, 1]``.  Losses outside
        are clipped (and the realised clip surfaced in the round
        receipt).

      * ``prior``: optional non-uniform prior over experts (defaults
        to uniform).  Used by every algorithm; sets the "safety
        baseline" relative to which KL drift is measured.

      * ``seed``: deterministic seed for FTPL and any stochastic
        ``select`` calls.

      * ``bound_delta``: confidence level for the finite-sample
        per-expert LCB / UCB certificates (default 0.05).

      * ``ftpl_draws``: number of Monte-Carlo draws for FTPL weight
        estimation (default 64).

      * ``horizon``: optional known total round count ``T``.  When
        supplied, ``HEDGE`` uses the Vovk-1990 minimax-optimal
        learning rate; otherwise AdaHedge or NormalHedge are
        recommended for the unknown-T regime.

    Public API
    ----------

      * ``predict() -> Prediction``: current weight distribution over
        experts plus realised KL from the prior.

      * ``select() -> Selection``: sample a single expert from the
        current weights using the seeded RNG; returns a Selection
        receipt with a tamper-evident fingerprint.

      * ``observe(losses: Mapping[expert, loss]) -> Round``: feed in
        the realised loss of every expert this round.  Updates the
        cumulative losses, the AdaHedge mixability gap, the variance
        terms, and the weight distribution.

      * ``observe_partial(losses, sleeping) -> Round``: same but with
        a ``sleeping`` set of experts that abstained (specialists,
        Freund-Schapire-Singer-Warmuth 1997).  The cumulative loss
        of sleeping experts is left unchanged; the weights are
        renormalised over the active subset.

      * ``per_expert_certificate(expert, delta=None) -> ExpertCertificate``:
        Hoeffding / empirical-Bernstein / anytime-confidence-sequence
        LCB and UCB on the expert's mean loss.

      * ``regret_certificate() -> RegretCertificate``: closed-form
        algorithm-specific bound (first-order ``√(T log N)`` plus the
        second-order ``√(V_T log N)`` when available).

      * ``report() -> HedgerReport``: aggregated state, the regret
        certificate, the per-round receipts, and the fingerprint chain.

      * ``clear() -> None``: reset the cumulative state, preserving
        the configuration.

      * ``snapshot() / restore(snapshot)``: replay-deterministic
        serialisation.

    Properties
    ----------

      * Thread-safe via an internal re-entrant lock.

      * Replay-deterministic given the seed.

      * Pure stdlib; no NumPy dependency.

      * Composes with ``AttestationLedger`` (fingerprint chain),
        ``Quantilizer`` (safety-bounded versions of weight
        distributions), and every decision primitive as an expert.
    """

    def __init__(self, config: HedgerConfig,
                 bus: Any = None):
        self._config = config
        self._lock = threading.RLock()
        self._bus = bus

        self._experts: tuple = config.experts
        self._N: int = len(self._experts)
        self._algorithm: str = config.algorithm
        self._eta_fixed: float | None = config.eta
        self._lower: float = config.loss_lower
        self._upper: float = config.loss_upper
        self._loss_range: float = config.loss_upper - config.loss_lower
        self._seed: int = config.seed
        self._bound_delta: float = config.bound_delta
        self._ftpl_draws: int = config.ftpl_draws
        self._horizon: int | None = config.horizon

        prior_dict = dict(config.prior) if config.prior else {}
        self._prior: dict = _validate_prior(prior_dict or None, self._experts)

        # State
        self._T: int = 0
        self._cum_losses: dict = {e: 0.0 for e in self._experts}
        self._sum_loss_sq: dict = {e: 0.0 for e in self._experts}
        # For ML-Prod / Squint: per-expert variance against the running mean.
        self._cum_variance: dict = {e: 0.0 for e in self._experts}
        self._cum_regret: dict = {e: 0.0 for e in self._experts}
        self._cumulative_weighted_loss: float = 0.0
        self._cum_mixability_gap: float = _ADAHEDGE_INIT_DELTA
        # For NormalHedge: regret per expert at round t.
        # (re-use _cum_regret above)
        # For FTRL_L2: log-prior is irrelevant; need prior itself.
        # ML-Prod log-weights (initialised at log prior).
        self._mlprod_log_w: dict = {e: _safe_log(self._prior[e])
                                     for e in self._experts}
        self._n_rounds_active: dict = {e: 0 for e in self._experts}

        # RNG.
        self._rng: random.Random = random.Random(self._seed)

        # Receipts.
        self._predictions: list[Prediction] = []
        self._selections: list[Selection] = []
        self._rounds: list[Round] = []

        # Fingerprint chain.
        self._genesis: str = _GENESIS
        self._fingerprint: str = self._genesis

        # Sanity: emit the genesis event.
        self._emit(HEDGER_STARTED, {
            "experts": [str(_expert_key(e)) for e in self._experts],
            "algorithm": self._algorithm,
            "eta": self._eta_fixed,
            "loss_range": [self._lower, self._upper],
            "prior": _sorted_pairs(self._prior),
            "seed": self._seed,
            "horizon": self._horizon,
        })

    # ----- factory ---------------------------------------------------

    @classmethod
    def create(cls,
               experts: Sequence[Hashable],
               *,
               algorithm: str = HEDGE,
               eta: float | None = None,
               loss_lower: float = 0.0,
               loss_upper: float = 1.0,
               prior: Mapping[Any, float] | None = None,
               seed: int = 0,
               bound_delta: float = 0.05,
               ftpl_draws: int = 64,
               horizon: int | None = None,
               bus: Any = None,
               ) -> "Hedger":
        experts_t = _validate_experts(experts)
        algorithm = _validate_algorithm(algorithm)
        loss_lower, loss_upper = _validate_loss_range(loss_lower, loss_upper)
        prior_d = _validate_prior(prior, experts_t)
        if eta is not None:
            _validate_eta(eta)
        if not isinstance(seed, int):
            raise GenericConfigError("seed must be int")
        _validate_delta(bound_delta)
        if not isinstance(ftpl_draws, int) or ftpl_draws < 1:
            raise GenericConfigError("ftpl_draws must be a positive int")
        if horizon is not None and (not isinstance(horizon, int) or horizon < 1):
            raise GenericConfigError("horizon must be a positive int or None")

        config = HedgerConfig(
            experts=experts_t,
            algorithm=algorithm,
            eta=eta,
            loss_lower=loss_lower,
            loss_upper=loss_upper,
            prior=tuple(sorted(prior_d.items(),
                               key=lambda kv: _expert_key(kv[0]))),
            seed=seed,
            bound_delta=bound_delta,
            ftpl_draws=ftpl_draws,
            horizon=horizon,
        )
        return cls(config, bus=bus)

    # ----- private: event emission ----------------------------------

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(kind, dict(payload))  # type: ignore[attr-defined]
        except Exception:
            # The hedger's correctness must never depend on the bus.
            pass

    # ----- private: η selection -------------------------------------

    def _current_eta(self) -> float:
        """Return the learning rate used at the current round."""
        if self._algorithm in (HEDGE, FTRL_ENTROPY, FTRL_L2, FTPL,
                                OMD_ENTROPY, ML_PROD):
            if self._eta_fixed is not None:
                return self._eta_fixed
            if self._horizon is not None and self._N > 1:
                return hedge_minimax_eta(self._horizon, self._N,
                                          loss_range=self._loss_range)
            # No horizon, no fixed η → use 1 / loss_range as a default.
            return 1.0 / max(self._loss_range, _EPS)
        if self._algorithm == ADAHEDGE:
            return _adahedge_eta(self._cum_mixability_gap, self._N)
        # NORMAL_HEDGE, SQUINT, BOA do not use a global η.
        return float("nan")

    # ----- public: current weights ----------------------------------

    def predict(self) -> Prediction:
        with self._lock:
            return self._predict_locked()

    def _predict_locked(self) -> Prediction:
        weights = self._compute_weights()
        kl = kl_divergence(weights, self._prior)
        H = 0.0
        for w in weights.values():
            if w > 0.0:
                H -= w * math.log(w)
        fp = _hash({
            "kind": "predict",
            "T": self._T,
            "weights": _sorted_pairs(weights),
            "parent": self._fingerprint,
        })
        pred = Prediction(
            round=self._T,
            weights=dict(weights),
            realised_kl_from_prior=kl,
            entropy=H,
            fingerprint=fp,
        )
        self._predictions.append(pred)
        self._emit(HEDGER_PREDICTED, {
            "round": self._T,
            "weights": _sorted_pairs(weights),
            "kl": kl,
            "entropy": H,
            "fingerprint": fp,
        })
        return pred

    def _compute_weights(self) -> dict:
        """Algorithm-dispatched current weights."""
        algo = self._algorithm
        if algo in (HEDGE, FTRL_ENTROPY, OMD_ENTROPY):
            eta = self._current_eta()
            log_prior = {e: _safe_log(self._prior[e]) for e in self._experts}
            return _hedge_weights(log_prior, self._cum_losses, eta)
        if algo == ADAHEDGE:
            eta = self._current_eta()
            log_prior = {e: _safe_log(self._prior[e]) for e in self._experts}
            return _hedge_weights(log_prior, self._cum_losses, eta)
        if algo == NORMAL_HEDGE:
            return _normal_hedge_weights(self._cum_regret, 1.0)
        if algo == SQUINT:
            return _squint_weights(self._prior, self._cum_regret,
                                    self._cum_variance)
        if algo == ML_PROD:
            return _ml_prod_weights(self._mlprod_log_w)
        if algo == FTRL_L2:
            eta = self._current_eta()
            return _ftrl_l2_weights(self._prior, self._cum_losses, eta)
        if algo == FTPL:
            eta = self._current_eta()
            # FTPL uses a per-call deterministic seed derived from the
            # round + the initial seed to ensure replay determinism.
            ftpl_seed = (self._seed * 1_000_003) ^ (self._T * 65537)
            return _ftpl_weights(self._prior, self._cum_losses, eta,
                                  seed=ftpl_seed,
                                  draws=self._ftpl_draws)
        if algo == BOA:
            return _boa_weights(self._prior, self._cum_regret,
                                 self._cum_variance)
        raise UnknownAlgorithm(f"weights for algorithm {algo!r} not implemented")

    # ----- public: stochastic selection -----------------------------

    def select(self, *, seed: int | None = None) -> Selection:
        with self._lock:
            return self._select_locked(seed=seed)

    def _select_locked(self, *, seed: int | None) -> Selection:
        pred = self._predict_locked()
        # Use a per-select seed for replay determinism.
        if seed is None:
            # Mix the round number into the global RNG.
            rng = self._rng
        else:
            rng = random.Random(seed)
        u = rng.random()
        cum = 0.0
        chosen = None
        for e in self._experts:
            cum += pred.weights[e]
            if u <= cum:
                chosen = e
                break
        if chosen is None:
            chosen = self._experts[-1]
        fp = _hash({
            "kind": "select",
            "T": self._T,
            "expert": _expert_key(chosen),
            "u": u,
            "parent": pred.fingerprint,
        })
        sel = Selection(
            round=self._T,
            expert=chosen,
            weight=pred.weights[chosen],
            seed=seed if seed is not None else self._seed,
            fingerprint=fp,
            parent_fingerprint=pred.fingerprint,
            timestamp=time.time(),
        )
        self._selections.append(sel)
        self._fingerprint = fp
        self._emit(HEDGER_SELECTED, {
            "round": self._T,
            "expert": str(_expert_key(chosen)),
            "weight": pred.weights[chosen],
            "fingerprint": fp,
        })
        return sel

    # ----- public: observe (the central update) ---------------------

    def observe(self, losses: Mapping[Hashable, float]) -> Round:
        with self._lock:
            return self._observe_locked(losses, sleeping=frozenset())

    def observe_partial(self, losses: Mapping[Hashable, float],
                        sleeping: Iterable[Hashable]) -> Round:
        with self._lock:
            return self._observe_locked(losses, sleeping=frozenset(sleeping))

    def _observe_locked(self, raw_losses: Mapping[Hashable, float],
                        sleeping: frozenset) -> Round:
        # Validate sleeping subset.
        for e in sleeping:
            if e not in set(self._experts):
                raise InvalidLoss(f"sleeping expert {e!r} not registered")
        # Active set = experts \ sleeping.
        active = tuple(e for e in self._experts if e not in sleeping)
        # Validate losses for active experts.
        if not active:
            raise InvalidLoss("all experts are sleeping; nothing to observe")
        # Construct a loss vector only on active experts; ignore sleeping
        # entries (a sleeping expert may or may not appear in the loss map).
        active_losses = {}
        for e in active:
            if e not in raw_losses:
                raise InvalidLoss(f"losses[{e!r}] missing")
            l = raw_losses[e]
            if not isinstance(l, (int, float)):
                raise InvalidLoss(f"losses[{e!r}] = {l!r} is not numeric")
            lv = float(l)
            if math.isnan(lv):
                raise InvalidLoss(f"losses[{e!r}] is NaN")
            active_losses[e] = lv
        # Clip into [lower, upper] for finite-sample bounds.
        clipped: dict = {}
        for e, l in active_losses.items():
            clipped[e] = _clip(l, self._lower, self._upper)

        # Current weights (renormalised over active experts).
        pred = self._predict_locked()
        if sleeping:
            mass = sum(pred.weights[e] for e in active)
            if mass <= 0.0:
                # Defensive: all active mass collapsed — re-uniform.
                weights = {e: 1.0 / len(active) for e in active}
            else:
                weights = {e: pred.weights[e] / mass for e in active}
        else:
            weights = pred.weights

        # 〈l, w〉
        inner = sum(weights[e] * clipped[e] for e in active)

        # AdaHedge mixability gap.
        eta_now = self._current_eta()
        if self._algorithm == ADAHEDGE and not math.isnan(eta_now):
            delta = _adahedge_mixability_gap(
                losses=[clipped[e] for e in active],
                weights=[weights[e] for e in active],
                eta=eta_now)
        elif self._algorithm in (HEDGE, FTRL_ENTROPY, OMD_ENTROPY) \
                and not math.isnan(eta_now):
            delta = _adahedge_mixability_gap(
                losses=[clipped[e] for e in active],
                weights=[weights[e] for e in active],
                eta=eta_now)
        else:
            # NormalHedge / Squint / BOA / FTPL / FTRL-L2 don't maintain a
            # global mixability gap.  We still track 〈l, w〉 for diagnostics.
            delta = 0.0

        # Update cumulative variance against the inner-weighted mean
        # (used by ML-Prod, Squint, BOA).
        for e in active:
            r = inner - clipped[e]               # per-round regret of expert e against algo
            self._cum_regret[e] += r
            # Squint / BOA second-order variance is (l_t(i) − 〈l, w〉)².
            self._cum_variance[e] += (clipped[e] - inner) ** 2
            self._sum_loss_sq[e] += clipped[e] ** 2
            self._cum_losses[e] += clipped[e]
            self._n_rounds_active[e] += 1

        # ML-Prod update.
        if self._algorithm == ML_PROD:
            # eta per expert: eta_e = min(1/2, sqrt(log N / V_T(i)))
            for e in active:
                v_e = max(self._cum_variance[e], 1.0)
                eta_e = min(0.5,
                             math.sqrt(math.log(max(self._N, 2)) / v_e))
                r = inner - clipped[e]
                # ML-Prod log-update:
                #   log w_{t+1}(i) = log w_t(i) + log(1 + eta_e (l_t - 〈l, w〉))
                arg = 1.0 + eta_e * r
                if arg <= 0.0:
                    arg = _EPS
                self._mlprod_log_w[e] += math.log(arg)

        # Cumulative metrics.
        self._cum_mixability_gap += delta
        self._cumulative_weighted_loss += inner
        self._T += 1

        # Best expert and its cumulative loss.
        best_e = min(self._experts,
                     key=lambda e: self._cum_losses[e])
        best_loss = self._cum_losses[best_e]

        fp = _hash({
            "kind": "observe",
            "T": self._T,
            "losses": _sorted_pairs(clipped),
            "inner": inner,
            "delta": delta,
            "parent": self._fingerprint,
        })

        rd = Round(
            round=self._T,
            losses=dict(active_losses),
            clipped_losses=dict(clipped),
            weighted_loss=inner,
            min_loss=min(clipped.values()),
            max_loss=max(clipped.values()),
            mean_loss=sum(clipped.values()) / len(clipped),
            realised_eta=eta_now,
            delta_mixability_gap=delta,
            cumulative_mixability_gap=self._cum_mixability_gap,
            cumulative_loss_by_expert=dict(self._cum_losses),
            cumulative_weighted_loss=self._cumulative_weighted_loss,
            best_expert=best_e,
            best_cumulative_loss=best_loss,
            fingerprint=fp,
            parent_fingerprint=self._fingerprint,
            timestamp=time.time(),
        )
        self._rounds.append(rd)
        self._fingerprint = fp
        self._emit(HEDGER_OBSERVED, {
            "round": self._T,
            "inner_loss": inner,
            "delta": delta,
            "best_expert": str(_expert_key(best_e)),
            "best_cumulative_loss": best_loss,
            "fingerprint": fp,
        })
        return rd

    # ----- public: per-expert certificates --------------------------

    def per_expert_certificate(self, expert: Hashable, *,
                                delta: float | None = None) -> ExpertCertificate:
        with self._lock:
            if expert not in set(self._experts):
                raise InvalidLoss(f"unknown expert {expert!r}")
            d = self._bound_delta if delta is None else _validate_delta(delta)
            n = self._n_rounds_active[expert]
            if n < 1:
                raise InsufficientData(
                    f"per_expert_certificate requires n ≥ 1 observations; got 0")
            cum = self._cum_losses[expert]
            sum_sq = self._sum_loss_sq[expert]
            mean = cum / n
            var = max((sum_sq / n) - mean * mean, 0.0)
            hl = hoeffding_lcb(mean, n, delta=d,
                                lower=self._lower, upper=self._upper)
            hu = hoeffding_ucb(mean, n, delta=d,
                                lower=self._lower, upper=self._upper)
            al = anytime_lcb(mean, n, delta=d,
                              lower=self._lower, upper=self._upper)
            au = anytime_ucb(mean, n, delta=d,
                              lower=self._lower, upper=self._upper)
            bl: float | None = None
            bu: float | None = None
            if n >= 2:
                bl = empirical_bernstein_lcb(mean, var, n, delta=d,
                                              lower=self._lower,
                                              upper=self._upper)
                bu = empirical_bernstein_ucb(mean, var, n, delta=d,
                                              lower=self._lower,
                                              upper=self._upper)
            return ExpertCertificate(
                expert=expert,
                n=n,
                mean_loss=mean,
                variance=var,
                hoeffding_lcb=hl,
                hoeffding_ucb=hu,
                bernstein_lcb=bl,
                bernstein_ucb=bu,
                anytime_lcb=al,
                anytime_ucb=au,
                delta=d,
                lower=self._lower,
                upper=self._upper,
            )

    # ----- public: closed-form regret bound -------------------------

    def regret_certificate(self) -> RegretCertificate:
        with self._lock:
            return self._regret_certificate_locked()

    def _regret_certificate_locked(self) -> RegretCertificate:
        algo = self._algorithm
        N = self._N
        T = self._T
        L = self._loss_range
        # Best expert in hindsight.
        if T == 0:
            best_e = self._experts[0]
            best_cum = 0.0
            realised = 0.0
        else:
            best_e = min(self._experts, key=lambda e: self._cum_losses[e])
            best_cum = self._cum_losses[best_e]
            realised = self._cumulative_weighted_loss - best_cum

        eta_now = self._current_eta()

        if algo in (HEDGE, FTRL_ENTROPY, OMD_ENTROPY):
            first = (hedge_minimax_regret(T, N, loss_range=L)
                     if self._eta_fixed is None and self._horizon is None
                     else hedge_regret_bound(T, N,
                                              eta=eta_now if not math.isnan(eta_now)
                                              else 1.0,
                                              loss_range=L))
            second = None
        elif algo == ADAHEDGE:
            first = adahedge_regret_bound(self._cum_mixability_gap, N,
                                            loss_range=L)
            second = first
        elif algo == NORMAL_HEDGE:
            first = normal_hedge_regret_bound(T, N, rank=1, loss_range=L)
            second = None
        elif algo == SQUINT:
            V_T = self._cum_variance[best_e]
            first = squint_regret_bound(V_T, N, K=1, loss_range=L)
            second = first
        elif algo == ML_PROD:
            V_T = self._cum_variance[best_e]
            first = ml_prod_regret_bound(V_T, N, loss_range=L)
            second = first
        elif algo == FTRL_L2:
            # Zinkevich 2003: R_T ≤ D √T (with D = 1 for the simplex).
            first = math.sqrt(max(T, 0)) * L
            second = None
        elif algo == FTPL:
            # Kalai-Vempala 2005: R_T ≤ O(√(T log N)) with exponential perturbations.
            first = hedge_minimax_regret(T, N, loss_range=L) * 2.0
            second = None
        elif algo == BOA:
            V_T = self._cum_variance[best_e]
            first = boa_regret_bound(V_T, N, loss_range=L)
            second = first
        else:
            raise UnknownAlgorithm(f"regret cert for algorithm {algo!r} not implemented")

        # PAC-Bayes against the uniform prior (= log N at η=optimal).
        # Use the current weights as posterior.
        pb = 0.0
        if T > 0 and N > 1:
            try:
                weights = self._compute_weights()
                pb = pac_bayes_regret_bound(T, self._prior, weights,
                                             loss_range=L)
            except Exception:
                pb = 0.0

        return RegretCertificate(
            algorithm=algo,
            T=T,
            N=N,
            loss_range=L,
            eta=None if math.isnan(eta_now) else eta_now,
            first_order_bound=first,
            second_order_bound=second,
            pac_bayes_bound_uniform=pb,
            realised_regret_so_far=realised,
            best_expert=best_e,
            best_cumulative_loss=best_cum,
        )

    # ----- public: aggregated report --------------------------------

    def report(self) -> HedgerReport:
        with self._lock:
            rc = self._regret_certificate_locked()
            fp = _hash({
                "kind": "report",
                "T": self._T,
                "cum_losses": _sorted_pairs(self._cum_losses),
                "cum_weighted_loss": self._cumulative_weighted_loss,
                "parent": self._fingerprint,
            })
            self._emit(HEDGER_REPORT, {
                "T": self._T,
                "best_expert": str(_expert_key(rc.best_expert)),
                "first_order_bound": rc.first_order_bound,
                "realised_regret_so_far": rc.realised_regret_so_far,
                "fingerprint": fp,
            })
            return HedgerReport(
                algorithm=self._algorithm,
                T=self._T,
                N=self._N,
                eta=rc.eta,
                loss_range=(self._lower, self._upper),
                cumulative_loss_by_expert=dict(self._cum_losses),
                cumulative_weighted_loss=self._cumulative_weighted_loss,
                realised_regret_so_far=rc.realised_regret_so_far,
                cumulative_mixability_gap=self._cum_mixability_gap,
                last_prediction=self._predictions[-1] if self._predictions else None,
                last_selection=self._selections[-1] if self._selections else None,
                last_round=self._rounds[-1] if self._rounds else None,
                regret_certificate=rc,
                fingerprint=fp,
                genesis=self._genesis,
                config={
                    "algorithm": self._algorithm,
                    "N": self._N,
                    "eta_fixed": self._eta_fixed,
                    "loss_range": [self._lower, self._upper],
                    "seed": self._seed,
                    "bound_delta": self._bound_delta,
                    "horizon": self._horizon,
                    "ftpl_draws": self._ftpl_draws,
                },
            )

    # ----- public: state management ---------------------------------

    def clear(self) -> None:
        with self._lock:
            self._T = 0
            self._cum_losses = {e: 0.0 for e in self._experts}
            self._sum_loss_sq = {e: 0.0 for e in self._experts}
            self._cum_variance = {e: 0.0 for e in self._experts}
            self._cum_regret = {e: 0.0 for e in self._experts}
            self._cumulative_weighted_loss = 0.0
            self._cum_mixability_gap = _ADAHEDGE_INIT_DELTA
            self._mlprod_log_w = {e: _safe_log(self._prior[e])
                                    for e in self._experts}
            self._n_rounds_active = {e: 0 for e in self._experts}
            self._rng = random.Random(self._seed)
            self._predictions.clear()
            self._selections.clear()
            self._rounds.clear()
            self._fingerprint = self._genesis
            self._emit(HEDGER_CLEARED, {"reset_to": self._genesis})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "config": asdict(self._config),
                "T": self._T,
                "cum_losses": dict(self._cum_losses),
                "sum_loss_sq": dict(self._sum_loss_sq),
                "cum_variance": dict(self._cum_variance),
                "cum_regret": dict(self._cum_regret),
                "cumulative_weighted_loss": self._cumulative_weighted_loss,
                "cum_mixability_gap": self._cum_mixability_gap,
                "mlprod_log_w": dict(self._mlprod_log_w),
                "n_rounds_active": dict(self._n_rounds_active),
                "fingerprint": self._fingerprint,
            }

    def restore(self, snap: Mapping[str, Any]) -> None:
        with self._lock:
            self._T = int(snap["T"])
            self._cum_losses = dict(snap["cum_losses"])
            self._sum_loss_sq = dict(snap["sum_loss_sq"])
            self._cum_variance = dict(snap["cum_variance"])
            self._cum_regret = dict(snap["cum_regret"])
            self._cumulative_weighted_loss = float(snap["cumulative_weighted_loss"])
            self._cum_mixability_gap = float(snap["cum_mixability_gap"])
            self._mlprod_log_w = dict(snap["mlprod_log_w"])
            self._n_rounds_active = dict(snap["n_rounds_active"])
            self._fingerprint = str(snap["fingerprint"])

    # ----- introspection --------------------------------------------

    @property
    def experts(self) -> tuple:
        return self._experts

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def T(self) -> int:
        return self._T

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def cumulative_losses(self) -> dict:
        return dict(self._cum_losses)

    @property
    def best_expert(self) -> Any:
        if self._T == 0:
            return self._experts[0]
        return min(self._experts, key=lambda e: self._cum_losses[e])


__all__ = [
    # constants
    "HEDGE", "ADAHEDGE", "NORMAL_HEDGE", "SQUINT", "ML_PROD",
    "FTRL_ENTROPY", "FTRL_L2", "FTPL", "OMD_ENTROPY", "BOA",
    "KNOWN_ALGORITHMS",
    "HOEFFDING", "BERNSTEIN", "ANYTIME", "KNOWN_BOUND_METHODS",
    "HEDGER_STARTED", "HEDGER_PREDICTED", "HEDGER_SELECTED",
    "HEDGER_OBSERVED", "HEDGER_REPORT", "HEDGER_CLEARED",
    "KNOWN_EVENTS",
    # exceptions
    "HedgerError", "UnknownAlgorithm", "UnknownBoundMethod",
    "InvalidExperts", "InvalidLearningRate", "InvalidPrior",
    "InvalidLoss", "InvalidLossRange", "InsufficientData",
    "GenericConfigError",
    # dataclasses
    "Prediction", "Selection", "Round", "ExpertCertificate",
    "RegretCertificate", "HedgerReport", "HedgerConfig",
    # main class
    "Hedger",
    # public functions
    "hedge_regret_bound", "hedge_minimax_eta", "hedge_minimax_regret",
    "adahedge_regret_bound", "normal_hedge_regret_bound",
    "squint_regret_bound", "ml_prod_regret_bound", "boa_regret_bound",
    "pac_bayes_regret_bound", "kl_divergence",
    "hoeffding_lcb", "hoeffding_ucb",
    "empirical_bernstein_lcb", "empirical_bernstein_ucb",
    "anytime_lcb", "anytime_ucb",
]
