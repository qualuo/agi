r"""Quantilizer — safety-bounded optimisation as a runtime primitive.

Every other decision primitive in this runtime — `Bandit`, `BayesOpt`,
`Arbiter`, `PolicyImprover`, `Persuader`, `Strategist` — answers the
question *"what is the best action?"*.  Each delivers an excellent
answer when the proxy utility being optimised is the true utility.
None of them defend against the case the proxy is *wrong* — the
classical Goodhart pathology in which optimising a proxy past the
threshold at which it correlates with the truth amplifies hidden
costs without bound (Manheim & Garrabrant 2018 — *Categorizing
Variants of Goodhart's Law*).

That gap is what ``Quantilizer`` fills.

A *q-quantilizer* (Taylor 2016 — *Quantilizers: A Safer Alternative
to Maximizers for Limited Optimization*) is the optimiser that, given
a *base distribution* ``b`` over actions and a *proxy utility* ``U``,
returns the distribution that samples uniformly from the top-``q``-
quantile of ``b`` ranked by ``U``.  When ``q = 1`` it is the base
distribution itself.  When ``q → 0`` it is the argmax of ``U`` —
exactly the Goodhart-vulnerable optimiser.  In between it interpolates
along a precisely characterised safety / performance frontier:

  * **KL bound** (Taylor 2016, Theorem 1):
        ``KL( quantilizer_q ‖ b ) ≤ log(1 / q)``.

    Tight, achieved when the top-``q``-quantile of ``b`` is a single
    point mass.  Hand a quantilizer to a coordinator and the
    coordinator can prove a runtime-level KL budget on how far the
    chosen policy can drift from the safe base — the cleanest
    information-theoretic guard against Goodhart drift.

  * **Cost-amplification bound** (Taylor 2016, Theorem 2):
        If a *hidden cost* ``c`` satisfies ``E_b[c] ≤ C``, then
        ``E_{q-quant}[c] ≤ C / q``.

    Whatever unmodelled risk a black-box action might carry is
    bounded by ``1 / q`` times its base-distribution average.  The
    coordinator chooses ``q`` to trade expected utility against this
    multiplicative worst-case downside; ``q = 0.01`` admits at most a
    100× amplification, ``q = 0.1`` at most 10×, ``q = 1`` no
    amplification at all.

  * **Total variation bound**:
        ``TV( quantilizer_q ‖ b ) ≤ 1 − q``.

    Direct from coupling.  Composes with f-divergence inequalities
    (Pinsker, Bretagnolle-Huber, le Cam) to give a half-dozen
    distance-from-base certificates in one call.

The pitch reduced to a runtime call:

  * ``select(base, utility, q)`` returns the action sampled from the
    q-quantilizer plus a ``Selection`` carrying the KL bound, the TV
    bound, the cost-amplification factor, the realised quantile
    threshold, the exact probability under base ``b``, and a
    tamper-evident receipt fingerprint hashable into
    `AttestationLedger`.

  * ``quantilize_discrete(base, utility, q)`` returns the exact
    discrete quantilizer distribution over a finite support.

  * ``quantilize_samples(samples, utility, q)`` is the empirical
    analogue: estimate the (1-q)-quantile of ``U`` under ``b`` from a
    sample, return both the chosen subset and a Massart-DKW
    finite-sample bound on the quantile-estimation error.

  * ``soft_quantilize(base, utility, kl_budget)`` is the smooth dual:
    return the Boltzmann distribution ``b(a) exp(β U(a)) / Z`` with
    ``β`` solved (by bisection) to land exactly on a coordinator-
    supplied KL budget — the continuous analogue of the hard
    quantilizer that composes with any utility, including unbounded
    or noisy ones.

  * ``expected_utility_lcb(δ)`` returns a finite-sample LCB on the
    expected utility of the quantilizer's action under the *true*
    utility, with confidence ``1 − δ`` — via Hoeffding 1963
    distribution-free, Maurer-Pontil 2009 empirical-Bernstein, or
    Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence
    sequence.

  * ``cost_ucb(δ)`` returns the matching UCB on hidden cost via the
    Taylor-2016 amplification bound applied to the empirical
    base-distribution cost estimate.

  * ``report()`` aggregates every selection, the KL/TV/cost certificates,
    every observed utility for finite-sample bookkeeping, and a
    SHA-256 hash chain (genesis ``quantilizer.v1.genesis``) so an
    external auditor can replay every safety budget byte-for-byte.

Algorithms shipped
------------------

**Hard quantilization** (Taylor 2016)

  * **Exact discrete quantilization**.  Given a finite-support base
    ``b`` and a utility ``U: supp(b) → ℝ``, sort the support by
    ``U`` (with deterministic stable tie-breaking on a SHA-256 of
    the action's canonical JSON), accumulate base mass from the top
    until it first equals or exceeds ``q``, and renormalise.  Handles
    the boundary atom by mixing: at the ``(1-q)``-quantile the
    fractional remaining mass receives a partial weight so the total
    output mass is exactly ``q`` *before* the ``1/q`` rescale.  The
    KL bound ``log(1/q)`` is exact in the worst case; for
    less-degenerate utilities the realised KL is smaller and is
    returned alongside the bound.

  * **Sample-based quantilization** (Hoeffding 1963 quantile
    bound).  Sample ``n`` actions from ``b``, compute their proxy
    utilities, retain the top ``⌈qn⌉`` and resample uniformly.  The
    empirical quantilizer is a uniform draw over the retained set.
    Massart-DKW 1990 gives the deviation of the empirical CDF from
    the true CDF: ``P(sup |F̂_n − F| > ε) ≤ 2 exp(-2 n ε²)``, which
    inverts to a finite-sample ``ε`` band on the realised
    ``(1-q)``-quantile that bounds *both* the realised mass passed
    through the safety filter *and* the deviation from the
    population quantile.

**Smooth (Boltzmann / Gibbs) quantilization** (Garrabrant et al.
2021 *Quantilizers and exploration-exploitation in deception
robustness*; the continuous extension implicitly used in safe-RL
soft-Q-learning, Haarnoja et al. 2017)

Given a KL budget ``B``, define

    ``π_β(a) = b(a) exp(β U(a)) / Z(β)``,  ``Z(β) = E_b[exp(β U)]``.

The KL of ``π_β`` from ``b`` is monotone increasing in ``β`` (it is
the derivative of the cumulant ``log Z`` with respect to ``β``
times ``β`` minus ``log Z`` — convex in ``β``), so a unique ``β*``
satisfies ``KL(π_{β*} ‖ b) = B``, found by bisection on ``β ∈ [0,
β_max]``.  This continuous-temperature dual of the hard quantilizer
composes with any utility (including unbounded ones, by clipping or
normalising first) and is the natural choice when the base
distribution is given as a sampler rather than an explicit pmf.

**K-quantilization** (the discrete-action specialisation)

For combinatorial action spaces, ``top_K(base, utility, K)`` returns
the K actions with the largest ``U`` weighted by ``b``, with a
deterministic break-tie.  The KL bound is ``log(N / K)`` when ``b``
is uniform over N atoms and tighter for skewed ``b``.  Composes with
`Submodular` for diversity-aware quantilization of subset selection.

**Conditional / contextual quantilization**

``quantilize_conditional(base_by_context, utility_by_context, q,
context)`` returns the q-quantilizer of the *conditional* base
distribution given a context.  Composes with `Bandit.LinUCB` and
`BayesOpt` to bound the KL of any contextual policy from a safe
prior contextual policy.

**Quantile estimation with PAC guarantees**

  * **Massart-DKW finite-sample band**.  ``ε_n = √(log(2/δ) / (2n))``;
    every empirical quantile ``q̂`` is within ``ε_n`` of the true
    quantile with probability ``1 − δ`` simultaneously over all
    quantiles.

  * **Hoeffding LCB on expected utility under the quantilizer**.
    With ``U`` bounded in ``[0, 1]`` and ``n`` samples through the
    quantilizer, ``Ê_q U − √(log(1/δ) / (2 n_q)) ≤ E_q U``.

  * **Maurer-Pontil 2009 empirical-Bernstein LCB**.  Sharper when the
    observed variance ``σ̂²`` is small:
    ``Ê − √(2 σ̂² log(2/δ)/n_q) − 7 (b−a) log(2/δ) / (3(n_q-1))``.

  * **Anytime confidence sequence** (Howard, Ramdas, McAuliffe &
    Sekhon 2021).  Mixture-of-supermartingales construction.  The
    bound holds *simultaneously for every n ≥ 1* — the coordinator
    may stop sampling at any data-dependent time and the certificate
    still holds.

Composition with the rest of the runtime
----------------------------------------

  * **Bandit / BayesOpt / Arbiter** — wrap the inner-loop selection
    in ``Quantilizer.select(base=algorithm_distribution, utility=
    estimated_reward, q=q)`` to bound KL from a safe baseline.  This
    is the *exploration safety budget* that makes any bandit run
    Goodhart-robust at the cost of ``1/q`` regret amplification.

  * **PolicyImprover** — the natural safe-improvement step: instead
    of CRM-optimising the policy, quantilize against the *current*
    deployed policy.  The KL bound ``log(1/q)`` becomes the safety
    constant in the HCPI safety gate.

  * **Persuader** — the q-quantilizer over signal schemes bounds the
    information design's deviation from a truthful disclosure
    baseline.

  * **Strategist** — quantilize over recommendations to provide a
    risk-adjusted, KL-bounded meta-decision.

  * **Refuter** — Refuter's adversarial search becomes a quantilizer
    over the search space when the falsification budget needs to be
    bounded against false negatives via the cost-amplification
    inequality.

  * **PrivacyAccountant** — quantilization itself is post-processing
    of ``b`` and does not consume privacy budget when ``b`` is
    public; if ``b`` is the output of a noisy mechanism, the privacy
    guarantee on the quantilizer is the same as on ``b``.

  * **Sampler** — when the base distribution is given by an MCMC
    chain, ``quantilize_samples`` consumes a draw from the chain
    and returns a quantilized draw with the empirical quantile
    bound; combined with the Sampler's PSRF / ESS diagnostics, the
    quantilizer's sample-based bound is valid only on the converged
    chain.

  * **DriftSentinel** — a sudden change in the realised quantile
    threshold is a drift signal on the base distribution.

  * **AttestationLedger** — every ``Selection`` chain-hashes into the
    ledger, including the cryptographic commit to the base
    distribution, the proxy utility, the chosen ``q``, and the seed.

  * **Coordinator** — every Goal whose execution chooses among
    candidate plans, prompts, models or tools can be safety-budgeted
    by routing the candidate distribution through `Quantilizer`
    before action.

Anytime safety certificates
---------------------------

Every Quantilizer emits a `QuantilizerReport` carrying

  * **KL bound** ``log(1/q)`` — exact, distribution-free, tight in the
    worst case.

  * **Cost-amplification factor** ``1/q`` — Taylor 2016 Theorem 2.

  * **TV bound** ``1 − q`` — from coupling.

  * **Pinsker / Bretagnolle-Huber / Le Cam derived bounds** on every
    convex divergence the coordinator might ask for.

  * **Realised KL** — exact when the discrete quantilizer is used,
    estimated with Maurer-Pontil empirical-Bernstein when the
    sample-based quantilizer is used.

  * **Massart-DKW quantile-estimation error** in nats and probability.

  * **Hoeffding / Maurer-Pontil / Howard-Ramdas-McAuliffe-Sekhon
    finite-sample LCB on the expected utility under the quantilizer**.

  * **Tamper-evident fingerprint** — SHA-256 hash chain over
    (base, utility, q, seed, every selection) ensuring replay
    determinism and compatibility with `AttestationLedger`.

Numerical conventions
---------------------

  * **Pure stdlib.**  No NumPy.  No SciPy.  No SMT solver.  Linear-
    time exact discrete quantilization.  Bisection for the smooth
    quantilizer's temperature.  Beasley-Springer-Moro inverse-Φ for
    Gaussian computations.  ``math.lgamma`` for any incomplete-beta
    quantile inversion.

  * **Deterministic given seed.**  Every random draw goes through
    one ``random.Random(seed)`` shared by the selection.  Replay
    recovers an identical chosen action and identical fingerprint.

  * **JSON-canonical event payloads.**  Replay-deterministic
    fingerprint over canonicalised events.

  * **Type discipline.**  Actions are hashable (typically strings,
    ints, or tuples).  Probabilities are floats in ``[0, 1]`` summing
    to 1 (validated within ``_PROB_TOL``).  Utilities are floats
    (any range; clipped to ``[0, 1]`` only for tight Hoeffding bounds
    when the user opts in).

References
----------

  * **Taylor, J. (2016)** *Quantilizers: A Safer Alternative to
    Maximizers for Limited Optimization*.  AAAI-16 AI Ethics
    workshop.  The originating safety contract.

  * **Manheim, D. & Garrabrant, S. (2018)** *Categorizing Variants
    of Goodhart's Law*.  Establishes the failure modes the
    quantilizer defends against.

  * **Hoeffding, W. (1963)** *Probability inequalities for sums of
    bounded random variables*.  Finite-sample LCB.

  * **Massart, P. (1990)** *The tight constant in the Dvoretzky-
    Kiefer-Wolfowitz inequality*.  Quantile-estimation band.

  * **Maurer, A. & Pontil, M. (2009)** *Empirical Bernstein Bounds
    and Sample Variance Penalization*.  Sharper LCB.

  * **Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021)**
    *Time-uniform, nonparametric, nonasymptotic confidence
    sequences*.  Anytime-valid mixture-of-supermartingales bound.

  * **Bretagnolle, J. & Huber, C. (1979)** *Estimation des densités
    : risque minimax*.  TV ≤ √(1 − exp(−KL)) ≤ √(KL).

  * **Pinsker, M. S. (1964)** *Information and Information Stability
    of Random Variables and Processes*.  TV ≤ √(KL / 2).

  * **Haarnoja, T., Tang, H., Abbeel, P. & Levine, S. (2017)**
    *Reinforcement Learning with Deep Energy-Based Policies*.  The
    soft-Q-learning Boltzmann-policy parallel of the smooth
    quantilizer.

Author's contract
-----------------

The Quantilizer primitive returns *one* of these on every selection:

  1. A chosen action sampled from the q-quantilizer of a coordinator-
     supplied base distribution against a coordinator-supplied proxy
     utility, accompanied by the exact KL/TV/cost bounds, the
     realised quantile threshold, and a tamper-evident fingerprint.

  2. A diagnostic: the input was malformed, infeasible, or the
     quantile budget would land outside its support — coordinator
     should pick a less aggressive ``q``.

The quantilizer *never* claims a best action — only a *safer-than-
argmax* one.  That is the entire safety contract.
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

# Quantilization algorithm names.
HARD = "hard"                          # Taylor 2016 exact quantilizer
SOFT = "soft"                          # Boltzmann / Gibbs continuous quantilizer
TOP_K = "top_k"                        # K-best with deterministic tie-break
SAMPLE = "sample"                      # empirical quantilizer from samples

KNOWN_ALGORITHMS = frozenset({HARD, SOFT, TOP_K, SAMPLE})

# Lower-confidence-bound methods.
HOEFFDING = "hoeffding"                # Hoeffding 1963
BERNSTEIN = "bernstein"                # Maurer-Pontil 2009 empirical Bernstein
ANYTIME = "anytime"                    # Howard-Ramdas-McAuliffe-Sekhon 2021
DKW = "dkw"                            # Massart 1990 Dvoretzky-Kiefer-Wolfowitz

KNOWN_LCB_METHODS = frozenset({HOEFFDING, BERNSTEIN, ANYTIME, DKW})

# Soft-quantilizer temperature search bounds.
_BETA_MIN = 0.0
_BETA_MAX = 1.0e6
_BETA_BISECT_TOL = 1.0e-10
_BETA_MAX_ITER = 256

# Probability normalisation tolerance.
_PROB_TOL = 1.0e-9

# Small constants.
_EPS = 1.0e-15
_LN2 = math.log(2.0)
_INF = float("inf")

# Genesis fingerprint.
_GENESIS = hashlib.sha256(b"quantilizer.v1.genesis").hexdigest()

# Events emitted on the runtime EventBus.
QUANTILIZER_STARTED = "quantilizer.started"
QUANTILIZER_SELECTED = "quantilizer.selected"
QUANTILIZER_QUANTILIZED = "quantilizer.quantilized"
QUANTILIZER_OBSERVED = "quantilizer.observed"
QUANTILIZER_REPORT = "quantilizer.report"
QUANTILIZER_CLEARED = "quantilizer.cleared"

KNOWN_EVENTS = frozenset({
    QUANTILIZER_STARTED,
    QUANTILIZER_SELECTED,
    QUANTILIZER_QUANTILIZED,
    QUANTILIZER_OBSERVED,
    QUANTILIZER_REPORT,
    QUANTILIZER_CLEARED,
})


# =====================================================================
# Exceptions
# =====================================================================


class QuantilizerError(ValueError):
    """Base class for Quantilizer-domain errors."""


class InvalidQuantile(QuantilizerError):
    """q is not in (0, 1]."""


class InvalidDistribution(QuantilizerError):
    """Base distribution is malformed (negative mass, doesn't sum to 1,
    has a key that's not also in the utility map, etc.)."""


class InvalidUtility(QuantilizerError):
    """Utility is malformed (non-numeric, NaN, has a key not in base)."""


class InvalidSamples(QuantilizerError):
    """Sample list is malformed (empty, weights wrong, NaN utilities)."""


class UnknownAlgorithm(QuantilizerError):
    """Algorithm name is not in KNOWN_ALGORITHMS."""


class UnknownLCBMethod(QuantilizerError):
    """LCB method name is not in KNOWN_LCB_METHODS."""


class InsufficientData(QuantilizerError):
    """Too few observations for the requested bound."""


class BudgetInfeasible(QuantilizerError):
    """A KL budget is larger than ``log(N)``, the maximum achievable KL
    of any distribution on an N-atom support against a uniform base."""


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
    """JSON canonicalisation: sort keys, no whitespace, NaN-safe."""
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
    """Stable SHA-256 hex digest over JSON-canonicalised parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(_canonical_json(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _action_key(a: Any) -> str:
    """Tie-break key for actions: SHA-256 over the canonical JSON."""
    return hashlib.sha256(_canonical_json(a).encode("utf-8")).hexdigest()


# =====================================================================
# Validation
# =====================================================================


def _validate_quantile(q: float) -> float:
    if not isinstance(q, (int, float)):
        raise InvalidQuantile(f"q must be numeric; got {type(q).__name__}")
    qv = float(q)
    if math.isnan(qv) or not (0.0 < qv <= 1.0):
        raise InvalidQuantile(f"q must be in (0, 1]; got {qv}")
    return qv


def _validate_kl_budget(B: float) -> float:
    if not isinstance(B, (int, float)):
        raise QuantilizerError(f"KL budget must be numeric; got {type(B).__name__}")
    bv = float(B)
    if math.isnan(bv) or bv < 0.0:
        raise QuantilizerError(f"KL budget must be ≥ 0; got {bv}")
    return bv


def _validate_distribution(base: Mapping[Any, float]) -> dict:
    if not isinstance(base, Mapping):
        raise InvalidDistribution(
            f"base must be a Mapping; got {type(base).__name__}")
    if len(base) == 0:
        raise InvalidDistribution("base distribution is empty")
    out: dict = {}
    total = 0.0
    for a, p in base.items():
        if not isinstance(p, (int, float)):
            raise InvalidDistribution(
                f"base[{a!r}] = {p!r} is not numeric")
        pv = float(p)
        if math.isnan(pv) or pv < 0.0:
            raise InvalidDistribution(
                f"base[{a!r}] = {pv} is negative or NaN")
        out[a] = pv
        total += pv
    if total <= 0.0:
        raise InvalidDistribution("base distribution has zero total mass")
    if abs(total - 1.0) > _PROB_TOL:
        # accept up-to-tolerance, renormalise
        for k in out:
            out[k] /= total
    return out


def _validate_utility(utility: Mapping[Any, float],
                      support: Iterable[Any]) -> dict:
    if not isinstance(utility, Mapping):
        raise InvalidUtility(
            f"utility must be a Mapping; got {type(utility).__name__}")
    out: dict = {}
    for a in support:
        if a not in utility:
            raise InvalidUtility(
                f"utility[{a!r}] missing; base support requires it")
        u = utility[a]
        if not isinstance(u, (int, float)):
            raise InvalidUtility(
                f"utility[{a!r}] = {u!r} is not numeric")
        uv = float(u)
        if math.isnan(uv):
            raise InvalidUtility(f"utility[{a!r}] is NaN")
        out[a] = uv
    return out


def _validate_lcb_method(method: str) -> str:
    if method not in KNOWN_LCB_METHODS:
        raise UnknownLCBMethod(
            f"LCB method {method!r} not in {sorted(KNOWN_LCB_METHODS)}")
    return method


def _validate_delta(delta: float) -> float:
    if not isinstance(delta, (int, float)):
        raise QuantilizerError(f"delta must be numeric; got {type(delta).__name__}")
    dv = float(delta)
    if math.isnan(dv) or not (0.0 < dv < 1.0):
        raise QuantilizerError(f"delta must be in (0, 1); got {dv}")
    return dv


# =====================================================================
# Information-theoretic bound helpers (closed form)
# =====================================================================


def kl_bound_from_quantile(q: float) -> float:
    r"""Worst-case KL divergence of a q-quantilizer from its base.

    ``KL( quantilizer_q ‖ b ) ≤ log(1/q)``  (Taylor 2016, Theorem 1).

    Tight when the top-q-quantile of b is a single atom of mass q.
    Returns the bound in nats.
    """
    q = _validate_quantile(q)
    return -math.log(q)


def cost_amplification(q: float) -> float:
    r"""Worst-case hidden-cost amplification factor.

    If ``E_b[c] ≤ C`` for any non-negative hidden cost c, then
    ``E_{q-quant}[c] ≤ C / q``  (Taylor 2016, Theorem 2).
    """
    q = _validate_quantile(q)
    return 1.0 / q


def tv_bound_from_quantile(q: float) -> float:
    r"""Worst-case total-variation distance of a q-quantilizer from its base.

    ``TV( quantilizer_q ‖ b ) ≤ 1 − q``.

    Direct from coupling: the quantilizer agrees with the base on the
    top-q-quantile, so they couple on a mass-q event.
    """
    q = _validate_quantile(q)
    return 1.0 - q


def pinsker_tv_from_kl(kl: float) -> float:
    r"""Pinsker's inequality: ``TV(p, q) ≤ √(KL(p ‖ q) / 2)``."""
    if kl < 0.0:
        raise QuantilizerError(f"KL must be ≥ 0; got {kl}")
    return math.sqrt(kl / 2.0)


def bretagnolle_huber_tv_from_kl(kl: float) -> float:
    r"""Bretagnolle-Huber 1979: ``TV(p, q) ≤ √(1 − exp(−KL))``.

    Tighter than Pinsker for large KL (≥ 2/3).
    """
    if kl < 0.0:
        raise QuantilizerError(f"KL must be ≥ 0; got {kl}")
    return math.sqrt(1.0 - math.exp(-kl))


def le_cam_overlap_from_tv(tv: float) -> float:
    r"""Le Cam: the overlap (Hellinger-like) ``≥ 1 − TV``."""
    if not 0.0 <= tv <= 1.0:
        raise QuantilizerError(f"TV must be in [0, 1]; got {tv}")
    return 1.0 - tv


def kl_kl_bernoulli(p: float, q: float) -> float:
    r"""KL(Ber(p) ‖ Ber(q))."""
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise QuantilizerError("p and q must be in [0, 1]")
    if q in (0.0, 1.0) and p != q:
        return _INF
    if p == 0.0:
        return -math.log(1.0 - q) if q < 1.0 else _INF
    if p == 1.0:
        return -math.log(q) if q > 0.0 else _INF
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


# =====================================================================
# Inverse-Phi (Beasley-Springer-Moro 1995) for Gaussian probabilities
# =====================================================================


def _inverse_phi(p: float) -> float:
    """Beasley-Springer-Moro 1995 inverse-Φ. Returns Z such that Φ(Z) = p.

    Accurate to roughly 1e-9 over (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise QuantilizerError(f"inverse_phi domain (0, 1); got {p}")
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = ((((-25.44106049637) * r + 41.39119773534) * r +
                -18.61500062529) * r + 2.50662823884) * y
        den = ((((3.13082909833) * r + -21.06224101826) * r +
                23.08336743743) * r + -8.47351093090) * r + 1.0
        return num / den
    r = p if y < 0.0 else 1.0 - p
    r = math.log(-math.log(r))
    z = (0.3374754822726147 + r * (0.9761690190917186 + r * (0.1607979714918209
        + r * (0.0276438810333863 + r * (0.0038405729373609 + r * (
        0.0003951896511919 + r * (0.0000321767881768 + r * (
        0.0000002888167364 + r * 0.0000003960315187))))))))
    return -z if y < 0.0 else z


# =====================================================================
# Finite-sample bounds
# =====================================================================


def hoeffding_lcb(mean: float, n: int, *, delta: float,
                  lower: float = 0.0, upper: float = 1.0) -> float:
    r"""Hoeffding 1963 LCB on a bounded mean.

    For X_i ∈ [lower, upper], with probability ≥ 1 - delta:
        E[X] ≥ mean - (upper - lower) √(log(1/δ) / (2n)).
    """
    if n < 1:
        raise InsufficientData(f"hoeffding_lcb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    if upper < lower:
        raise QuantilizerError("upper < lower")
    half = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return mean - half


def hoeffding_ucb(mean: float, n: int, *, delta: float,
                  lower: float = 0.0, upper: float = 1.0) -> float:
    """Hoeffding 1963 UCB; analogous to ``hoeffding_lcb``."""
    if n < 1:
        raise InsufficientData(f"hoeffding_ucb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    if upper < lower:
        raise QuantilizerError("upper < lower")
    half = (upper - lower) * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return mean + half


def empirical_bernstein_lcb(mean: float, var: float, n: int, *,
                            delta: float, lower: float = 0.0,
                            upper: float = 1.0) -> float:
    r"""Maurer-Pontil 2009 empirical-Bernstein LCB.

    For X_i ∈ [lower, upper], with probability ≥ 1 − δ:
        E[X] ≥ mean − √(2 σ̂² log(2/δ)/n) − 7(upper−lower) log(2/δ)/(3(n−1)).
    """
    if n < 2:
        raise InsufficientData(
            f"empirical_bernstein_lcb requires n ≥ 2; got {n}")
    delta = _validate_delta(delta)
    if upper < lower:
        raise QuantilizerError("upper < lower")
    if var < 0.0:
        raise QuantilizerError(f"var must be ≥ 0; got {var}")
    L = math.log(2.0 / delta)
    half_var = math.sqrt(2.0 * var * L / n)
    half_range = 7.0 * (upper - lower) * L / (3.0 * (n - 1))
    return mean - half_var - half_range


def empirical_bernstein_ucb(mean: float, var: float, n: int, *,
                            delta: float, lower: float = 0.0,
                            upper: float = 1.0) -> float:
    """Maurer-Pontil 2009 empirical-Bernstein UCB."""
    if n < 2:
        raise InsufficientData(
            f"empirical_bernstein_ucb requires n ≥ 2; got {n}")
    delta = _validate_delta(delta)
    if var < 0.0:
        raise QuantilizerError(f"var must be ≥ 0; got {var}")
    L = math.log(2.0 / delta)
    half_var = math.sqrt(2.0 * var * L / n)
    half_range = 7.0 * (upper - lower) * L / (3.0 * (n - 1))
    return mean + half_var + half_range


def anytime_lcb(mean: float, n: int, *, delta: float,
                lower: float = 0.0, upper: float = 1.0) -> float:
    r"""Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid LCB.

    Mixture-of-supermartingales bound; valid simultaneously for every
    n ≥ 1 (the coordinator may stop adaptively without invalidating
    the bound).  The half-width is:

        ψ_n = √( (1 + 1/n) log(2 √(n+1) / δ) / (2 n) ) · (upper - lower)

    a sub-Gaussian time-uniform tightening of the Hoeffding bound by
    a factor only ``√log log n`` larger than the fixed-time half-width
    in the leading constant.
    """
    if n < 1:
        raise InsufficientData(f"anytime_lcb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    if upper < lower:
        raise QuantilizerError("upper < lower")
    L = math.log(2.0 * math.sqrt(n + 1.0) / delta)
    half = (upper - lower) * math.sqrt((1.0 + 1.0 / n) * L / (2.0 * n))
    return mean - half


def anytime_ucb(mean: float, n: int, *, delta: float,
                lower: float = 0.0, upper: float = 1.0) -> float:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid UCB."""
    if n < 1:
        raise InsufficientData(f"anytime_ucb requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    if upper < lower:
        raise QuantilizerError("upper < lower")
    L = math.log(2.0 * math.sqrt(n + 1.0) / delta)
    half = (upper - lower) * math.sqrt((1.0 + 1.0 / n) * L / (2.0 * n))
    return mean + half


def dkw_band(n: int, *, delta: float) -> float:
    r"""Massart 1990 Dvoretzky-Kiefer-Wolfowitz finite-sample CDF band.

    With probability ≥ 1 − δ, ``sup_x |F̂_n(x) − F(x)| ≤ ε``, where
    ``ε = √(log(2/δ) / (2n))``.  This is the tight constant.  Inverts
    to a quantile band: every empirical quantile is within ε (in
    probability mass) of the population quantile simultaneously.
    """
    if n < 1:
        raise InsufficientData(f"dkw_band requires n ≥ 1; got {n}")
    delta = _validate_delta(delta)
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def quantile_lcb_dkw(samples: Sequence[float], q: float, *,
                      delta: float) -> tuple[float, float]:
    r"""Massart-DKW finite-sample LCB and UCB on the (1-q)-quantile.

    Returns ``(lcb_quantile, ucb_quantile)`` such that with probability
    ≥ 1 − δ the true (1 − q)-quantile of the underlying distribution
    lies in ``[lcb_quantile, ucb_quantile]``.
    """
    if not samples:
        raise InsufficientData("quantile_lcb_dkw requires at least one sample")
    delta = _validate_delta(delta)
    q = _validate_quantile(q)
    sorted_samples = sorted(float(s) for s in samples)
    n = len(sorted_samples)
    eps = dkw_band(n, delta=delta)
    # The empirical (1 - q)-quantile is the sample at rank ⌈(1-q)n⌉.
    # The DKW band ε shifts the rank by ±εn.
    target = 1.0 - q
    lo_rank_frac = max(0.0, target - eps)
    hi_rank_frac = min(1.0, target + eps)
    lo_idx = int(math.floor(lo_rank_frac * n))
    hi_idx = int(math.ceil(hi_rank_frac * n)) - 1
    lo_idx = _clip(lo_idx, 0, n - 1)
    hi_idx = _clip(hi_idx, 0, n - 1)
    return (sorted_samples[lo_idx], sorted_samples[hi_idx])


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class QuantilizedDistribution:
    """A finite-support distribution after quantilization.

    ``probs`` is a dict mapping each (kept) action to its quantilizer
    probability.  ``threshold`` is the realised (1 − q)-quantile of the
    proxy utility under the base distribution.  ``realised_kl`` is the
    exact KL of the quantilizer from the base (≤ ``kl_bound``).
    """
    q: float
    algorithm: str
    probs: dict
    base_probs: dict
    utilities: dict
    threshold: float
    realised_kl: float
    kl_bound: float
    tv_bound: float
    realised_tv: float
    cost_amplification: float
    n_support: int
    n_kept: int
    fingerprint: str

    def support(self) -> tuple:
        return tuple(self.probs.keys())

    def expected_utility(self) -> float:
        return sum(p * self.utilities[a] for a, p in self.probs.items())

    def expected_utility_base(self) -> float:
        return sum(p * self.utilities[a] for a, p in self.base_probs.items())

    def utility_lift(self) -> float:
        """Δ = E_q[U] − E_b[U] — the quantilizer's *proxy* lift over base."""
        return self.expected_utility() - self.expected_utility_base()


@dataclass(frozen=True)
class Selection:
    """A single quantilized action draw with safety certificate."""
    action: Any
    q: float
    algorithm: str
    probability_under_base: float
    probability_under_quantilizer: float
    utility: float
    threshold: float
    kl_bound: float
    tv_bound: float
    realised_kl: float
    cost_amplification: float
    seed: int
    fingerprint: str
    parent_fingerprint: str
    timestamp: float


@dataclass(frozen=True)
class QuantileEstimate:
    """A finite-sample estimate of the (1 − q)-quantile of U under b."""
    q: float
    estimate: float
    lcb: float
    ucb: float
    dkw_band: float
    n_samples: int
    delta: float


@dataclass(frozen=True)
class UtilityBound:
    """Finite-sample LCB/UCB on the expected utility under the quantilizer."""
    method: str
    mean: float
    lcb: float
    ucb: float
    n: int
    delta: float
    variance: float
    lower: float
    upper: float


@dataclass(frozen=True)
class CostBound:
    """Cost UCB via Taylor-2016 amplification of a base-distribution UCB."""
    base_cost_ucb: float
    amplification: float
    quantilizer_cost_ucb: float
    n_base_samples: int
    delta: float
    method: str


@dataclass(frozen=True)
class Observation:
    """One realised (action, utility) pair logged for finite-sample stats."""
    action: Any
    utility: float
    came_from_quantilizer: bool
    timestamp: float


@dataclass(frozen=True)
class QuantilizerReport:
    n_selections: int
    n_observations: int
    n_quantilized: int
    cumulative_kl_bound: float
    last_selection: Selection | None
    last_quantilized: QuantilizedDistribution | None
    fingerprint: str
    genesis: str
    config: dict


# =====================================================================
# Discrete (exact) hard quantilizer
# =====================================================================


def quantilize_discrete(base: Mapping[Any, float],
                        utility: Mapping[Any, float], q: float) -> QuantilizedDistribution:
    r"""Exact discrete q-quantilizer (Taylor 2016).

    Algorithm:
      1. Validate inputs; renormalise base if needed.
      2. Sort the support by *decreasing* utility, breaking ties on the
         SHA-256 of the canonical-JSON of each action (deterministic).
      3. Walk the sorted list accumulating base mass.  Atoms that fit
         entirely under the mass budget ``q`` receive renormalised
         weight ``base[a] / q``.  The single boundary atom (if any)
         receives a fractional weight that makes the kept mass exactly
         q before rescale.
      4. Compute the realised KL and TV against the base.

    Returns a ``QuantilizedDistribution`` carrying every safety
    certificate the coordinator might ask for.
    """
    q = _validate_quantile(q)
    base = _validate_distribution(base)
    u = _validate_utility(utility, base.keys())

    # Sort by decreasing utility, with deterministic tie-break on the
    # SHA-256 of the canonical-JSON of the action.
    support = list(base.keys())
    support.sort(key=lambda a: (-u[a], _action_key(a)))

    # Walk the sorted list accumulating base mass until we reach q.
    cum = 0.0
    kept: dict = {}
    threshold = float("nan")
    for a in support:
        b_a = base[a]
        if cum + b_a <= q + _PROB_TOL:
            kept[a] = b_a
            cum += b_a
            threshold = u[a]
            if cum >= q - _PROB_TOL:
                # Filled exactly (within tolerance) — stop.
                break
        else:
            # Boundary atom: partial mass.
            partial = q - cum
            if partial > _PROB_TOL:
                kept[a] = partial
                cum += partial
                threshold = u[a]
            break

    # Renormalise to sum to 1 (the kept mass is q before rescale).
    total_kept = sum(kept.values())
    if total_kept <= 0.0:
        raise InvalidDistribution(
            "after quantilization, retained mass is zero — base or q malformed")
    probs = {a: p / total_kept for a, p in kept.items()}

    # Realised KL = sum_a probs[a] log(probs[a] / base[a]).
    realised_kl = 0.0
    for a, p in probs.items():
        if p > 0.0:
            realised_kl += p * math.log(p / base[a])

    # Realised TV = (1/2) sum_a |probs[a] - base[a]|  (over the full support).
    realised_tv = 0.0
    full_support = set(base.keys())
    for a in full_support:
        realised_tv += abs(probs.get(a, 0.0) - base[a])
    realised_tv *= 0.5

    fp = _hash({
        "kind": "quantilize_discrete",
        "algorithm": HARD,
        "q": q,
        "base": _sorted_pairs(base),
        "utility": _sorted_pairs(u),
        "probs": _sorted_pairs(probs),
        "threshold": threshold,
    })

    return QuantilizedDistribution(
        q=q,
        algorithm=HARD,
        probs=probs,
        base_probs=dict(base),
        utilities=dict(u),
        threshold=float(threshold),
        realised_kl=realised_kl,
        kl_bound=-math.log(q),
        tv_bound=1.0 - q,
        realised_tv=realised_tv,
        cost_amplification=1.0 / q,
        n_support=len(base),
        n_kept=len(probs),
        fingerprint=fp,
    )


# =====================================================================
# K-quantilizer (top-K with renormalisation)
# =====================================================================


def quantilize_top_k(base: Mapping[Any, float],
                      utility: Mapping[Any, float], K: int) -> QuantilizedDistribution:
    """Top-K quantilizer: keep the K base atoms with largest utility.

    The KL bound is ``log(W_total / W_K)`` where ``W_total = 1`` and
    ``W_K`` is the sum of base probabilities of the K retained atoms.

    Equivalent to the hard quantilizer with q = W_K (the mass of the
    top-K under base).
    """
    if K < 1:
        raise QuantilizerError(f"top-K requires K ≥ 1; got {K}")
    base = _validate_distribution(base)
    u = _validate_utility(utility, base.keys())
    if K >= len(base):
        K = len(base)

    support = list(base.keys())
    support.sort(key=lambda a: (-u[a], _action_key(a)))
    kept_actions = support[:K]
    kept_mass = {a: base[a] for a in kept_actions}
    total_kept = sum(kept_mass.values())
    if total_kept <= 0.0:
        raise InvalidDistribution("top-K kept mass is zero")
    probs = {a: p / total_kept for a, p in kept_mass.items()}

    realised_kl = sum(p * math.log(p / base[a]) for a, p in probs.items() if p > 0.0)
    realised_tv = 0.0
    for a in base:
        realised_tv += abs(probs.get(a, 0.0) - base[a])
    realised_tv *= 0.5
    threshold = u[kept_actions[-1]]

    fp = _hash({
        "kind": "quantilize_top_k",
        "algorithm": TOP_K,
        "K": K,
        "base": _sorted_pairs(base),
        "utility": _sorted_pairs(u),
        "probs": _sorted_pairs(probs),
        "threshold": threshold,
    })

    return QuantilizedDistribution(
        q=total_kept,
        algorithm=TOP_K,
        probs=probs,
        base_probs=dict(base),
        utilities=dict(u),
        threshold=float(threshold),
        realised_kl=realised_kl,
        kl_bound=-math.log(total_kept) if total_kept > 0.0 else _INF,
        tv_bound=1.0 - total_kept,
        realised_tv=realised_tv,
        cost_amplification=1.0 / total_kept if total_kept > 0.0 else _INF,
        n_support=len(base),
        n_kept=len(probs),
        fingerprint=fp,
    )


def _sorted_pairs(d: Mapping[Any, float]) -> list:
    """Stable canonical representation of a dict of (action, value) pairs."""
    return sorted(((str(_action_key(k)), float(v)) for k, v in d.items()),
                  key=lambda kv: kv[0])


# =====================================================================
# Smooth (Boltzmann / Gibbs) quantilizer
# =====================================================================


def _boltzmann_distribution(base: Mapping[Any, float],
                            utility: Mapping[Any, float],
                            beta: float) -> tuple[dict, float]:
    r"""Build ``π_β(a) ∝ b(a) exp(β U(a))``; return (π, log Z).

    Computed in log-space; numerically stable for any finite β.
    """
    log_b = {a: _safe_log(p) for a, p in base.items()}
    log_unnorm = {a: log_b[a] + beta * utility[a] for a in base}
    log_z = _logsumexp(list(log_unnorm.values()))
    probs = {a: math.exp(lu - log_z) for a, lu in log_unnorm.items()}
    return probs, log_z


def _boltzmann_kl(base: Mapping[Any, float],
                   utility: Mapping[Any, float],
                   beta: float) -> tuple[float, dict, float]:
    r"""KL(π_β ‖ b) = β · E_{π_β}[U] − log Z(β) − β · E_b[U].

    Returns (kl, probs, log_z).  Pure stdlib, numerically stable.
    """
    probs, log_z = _boltzmann_distribution(base, utility, beta)
    eu_base = sum(base[a] * utility[a] for a in base)
    log_zb = math.log(sum(base[a] for a in base))  # = 0 for normalised base
    # E_{π_β}[β U] - log Z(β) - β E_b[U] - log Z_b
    # Equivalent canonical form:
    e_pi = sum(probs[a] * utility[a] for a in base)
    kl = beta * e_pi - log_z + log_zb - 0.0  # log_zb = 0 because base normalised
    # Numerical guard: KL ≥ 0
    if kl < 0.0 and kl > -1e-9:
        kl = 0.0
    if kl < 0.0:
        # This indicates a numerical issue; recompute exactly.
        kl = 0.0
        for a, p in probs.items():
            if p > 0.0:
                kl += p * math.log(p / base[a])
    return kl, probs, log_z


def soft_quantilize(base: Mapping[Any, float],
                    utility: Mapping[Any, float], *,
                    kl_budget: float) -> QuantilizedDistribution:
    r"""Smooth (Boltzmann) quantilizer with a coordinator-supplied KL budget.

    Returns the distribution ``π_β(a) ∝ b(a) exp(β U(a))`` with β
    chosen by bisection so that ``KL(π_β ‖ b) = kl_budget``.

    Limit β → 0: π_β → b (kl_budget = 0).
    Limit β → ∞: π_β → argmax-supported distribution (kl_budget = log N
    in the worst case for an N-atom uniform base).
    """
    base = _validate_distribution(base)
    u = _validate_utility(utility, base.keys())
    B = _validate_kl_budget(kl_budget)

    # Maximum achievable KL: put all mass on the argmax atom.
    sorted_actions = sorted(base.keys(),
                             key=lambda a: (-u[a], _action_key(a)))
    arg_a = sorted_actions[0]
    max_kl = -math.log(max(base[arg_a], _EPS))
    if B > max_kl + 1.0e-9:
        raise BudgetInfeasible(
            f"KL budget {B:.6g} exceeds max achievable {max_kl:.6g} "
            f"on this support")

    if B <= 0.0:
        # Degenerate: return base itself.
        probs = dict(base)
        realised_kl = 0.0
        realised_tv = 0.0
        threshold = u[arg_a]
        fp = _hash({"kind": "soft_quantilize", "kl_budget": 0.0,
                    "base": _sorted_pairs(base),
                    "utility": _sorted_pairs(u)})
        return QuantilizedDistribution(
            q=1.0,
            algorithm=SOFT,
            probs=probs,
            base_probs=dict(base),
            utilities=dict(u),
            threshold=float(threshold),
            realised_kl=realised_kl,
            kl_bound=0.0,
            tv_bound=0.0,
            realised_tv=realised_tv,
            cost_amplification=1.0,
            n_support=len(base),
            n_kept=len(probs),
            fingerprint=fp,
        )

    # Check that utility is non-constant.
    u_values = list(u.values())
    u_range = max(u_values) - min(u_values)
    if u_range <= 1.0e-12:
        # Utility is constant; any β returns base itself.
        raise BudgetInfeasible(
            "utility is constant on the support; "
            "soft-quantilizer cannot achieve positive KL")

    # Bisection on β.
    lo, hi = _BETA_MIN, 1.0
    # Expand hi until KL(π_hi) ≥ B.
    for _ in range(64):
        kl_hi, _, _ = _boltzmann_kl(base, u, hi)
        if kl_hi >= B:
            break
        hi *= 2.0
        if hi > _BETA_MAX:
            hi = _BETA_MAX
            break

    # Bisect.
    for _ in range(_BETA_MAX_ITER):
        mid = 0.5 * (lo + hi)
        kl_mid, _, _ = _boltzmann_kl(base, u, mid)
        if kl_mid > B:
            hi = mid
        else:
            lo = mid
        if hi - lo < _BETA_BISECT_TOL:
            break

    beta_star = 0.5 * (lo + hi)
    kl_final, probs, log_z = _boltzmann_kl(base, u, beta_star)

    realised_tv = 0.0
    for a in base:
        realised_tv += abs(probs[a] - base[a])
    realised_tv *= 0.5

    # Effective q via KL = log(1/q) ⇒ q = exp(-KL).
    eff_q = math.exp(-kl_final) if kl_final < 50.0 else _EPS

    threshold = u[arg_a]
    fp = _hash({
        "kind": "soft_quantilize",
        "algorithm": SOFT,
        "kl_budget": B,
        "kl_final": kl_final,
        "beta": beta_star,
        "base": _sorted_pairs(base),
        "utility": _sorted_pairs(u),
        "probs": _sorted_pairs(probs),
    })

    return QuantilizedDistribution(
        q=eff_q,
        algorithm=SOFT,
        probs=probs,
        base_probs=dict(base),
        utilities=dict(u),
        threshold=float(threshold),
        realised_kl=kl_final,
        kl_bound=B,
        tv_bound=1.0 - eff_q,
        realised_tv=realised_tv,
        cost_amplification=1.0 / eff_q if eff_q > 0.0 else _INF,
        n_support=len(base),
        n_kept=len(probs),
        fingerprint=fp,
    )


def soft_quantilize_with_beta(base: Mapping[Any, float],
                              utility: Mapping[Any, float], *,
                              beta: float) -> QuantilizedDistribution:
    """Boltzmann distribution at a given inverse-temperature β (no bisection)."""
    base = _validate_distribution(base)
    u = _validate_utility(utility, base.keys())
    if not isinstance(beta, (int, float)) or math.isnan(beta) or beta < 0.0:
        raise QuantilizerError(f"beta must be ≥ 0; got {beta!r}")
    beta = float(beta)

    kl_final, probs, log_z = _boltzmann_kl(base, u, beta)

    sorted_actions = sorted(base.keys(),
                             key=lambda a: (-u[a], _action_key(a)))
    arg_a = sorted_actions[0]
    threshold = u[arg_a]
    realised_tv = 0.0
    for a in base:
        realised_tv += abs(probs[a] - base[a])
    realised_tv *= 0.5
    eff_q = math.exp(-kl_final) if kl_final < 50.0 else _EPS
    fp = _hash({
        "kind": "soft_quantilize_beta",
        "algorithm": SOFT,
        "beta": beta,
        "kl_final": kl_final,
        "base": _sorted_pairs(base),
        "utility": _sorted_pairs(u),
        "probs": _sorted_pairs(probs),
    })

    return QuantilizedDistribution(
        q=eff_q,
        algorithm=SOFT,
        probs=probs,
        base_probs=dict(base),
        utilities=dict(u),
        threshold=float(threshold),
        realised_kl=kl_final,
        kl_bound=kl_final,
        tv_bound=1.0 - eff_q,
        realised_tv=realised_tv,
        cost_amplification=1.0 / eff_q if eff_q > 0.0 else _INF,
        n_support=len(base),
        n_kept=len(probs),
        fingerprint=fp,
    )


# =====================================================================
# Sample-based quantilizer
# =====================================================================


@dataclass(frozen=True)
class SampleQuantilization:
    """Empirical quantilization from samples + DKW band on the quantile."""
    q: float
    algorithm: str
    kept: list
    kept_utilities: list
    n_total: int
    n_kept: int
    threshold: float
    quantile_lcb: float
    quantile_ucb: float
    dkw_band: float
    delta: float
    realised_q: float
    kl_bound: float
    tv_bound: float
    cost_amplification: float
    fingerprint: str


def quantilize_samples(samples: Sequence[Any],
                        utility: Callable[[Any], float],
                        q: float, *,
                        delta: float = 0.05) -> SampleQuantilization:
    r"""Empirical q-quantilizer over a sample drawn from the base.

    Algorithm:
      1. Compute ``u_i = utility(s_i)`` for every sample.
      2. Sort by decreasing utility (deterministic tie-break on
         SHA-256(canonical_json(s_i))).
      3. Retain the top ``k = ⌈qn⌉`` samples; the empirical
         quantilizer is uniform over the retained set.
      4. Return the Massart-DKW band ``ε = √(log(2/δ)/(2n))`` on the
         (1-q)-quantile of U under the (unknown) base.

    Even with no knowledge of the base distribution beyond the samples,
    the DKW band gives a distribution-free finite-sample certificate
    on the realised quantile threshold.
    """
    if not samples:
        raise InvalidSamples("quantilize_samples requires at least one sample")
    q = _validate_quantile(q)
    delta = _validate_delta(delta)
    n = len(samples)

    utilities = []
    for s in samples:
        u = utility(s)
        if not isinstance(u, (int, float)) or math.isnan(float(u)):
            raise InvalidUtility(f"utility({s!r}) returned non-numeric {u!r}")
        utilities.append(float(u))

    indexed = list(enumerate(samples))
    indexed.sort(key=lambda iv: (-utilities[iv[0]], _action_key(iv[1])))

    k = max(1, int(math.ceil(q * n)))
    kept_idx = [iv[0] for iv in indexed[:k]]
    kept = [samples[i] for i in kept_idx]
    kept_u = [utilities[i] for i in kept_idx]
    threshold = kept_u[-1]
    realised_q = k / n

    sorted_u = sorted(utilities)
    lcb_q, ucb_q = quantile_lcb_dkw(sorted_u, q, delta=delta)
    eps = dkw_band(n, delta=delta)

    fp = _hash({
        "kind": "quantilize_samples",
        "algorithm": SAMPLE,
        "q": q,
        "n": n,
        "k": k,
        "threshold": threshold,
        "kept_utilities": kept_u,
        "delta": delta,
    })

    return SampleQuantilization(
        q=q,
        algorithm=SAMPLE,
        kept=kept,
        kept_utilities=kept_u,
        n_total=n,
        n_kept=k,
        threshold=threshold,
        quantile_lcb=lcb_q,
        quantile_ucb=ucb_q,
        dkw_band=eps,
        delta=delta,
        realised_q=realised_q,
        kl_bound=-math.log(q),
        tv_bound=1.0 - q,
        cost_amplification=1.0 / q,
        fingerprint=fp,
    )


# =====================================================================
# Sampling from a quantilized distribution
# =====================================================================


def sample_from_distribution(dist: QuantilizedDistribution,
                              *, seed: int) -> tuple[Any, float]:
    """Sample a single action from a QuantilizedDistribution.

    Returns ``(action, probability_under_dist)``.  Deterministic given
    seed.
    """
    rng = random.Random(int(seed))
    pairs = sorted(dist.probs.items(),
                   key=lambda ap: _action_key(ap[0]))
    actions = [a for a, _ in pairs]
    weights = [p for _, p in pairs]
    cum = []
    s = 0.0
    for w in weights:
        s += w
        cum.append(s)
    if s <= 0.0:
        raise InvalidDistribution("quantilized distribution has no mass")
    # Renormalise on the fly: u ~ U(0, s).
    u = rng.random() * s
    for i, c in enumerate(cum):
        if u <= c:
            return actions[i], weights[i] / s
    return actions[-1], weights[-1] / s


# =====================================================================
# The Quantilizer class
# =====================================================================


class Quantilizer:
    r"""Stateful quantilizer with audit ledger.

    Holds:
      * a *default* quantile ``q`` (overrideable per call);
      * a list of every Selection, every quantilized distribution, and
        every observation;
      * a SHA-256 fingerprint chain over every event (genesis
        ``quantilizer.v1.genesis``);
      * a thread-safe re-entrant lock for concurrent access.

    The class is intentionally side-effect-free: ``select`` and
    ``quantilize`` are deterministic given inputs and seed.  ``observe``
    advances finite-sample statistics so anytime LCBs / UCBs can be
    queried as data accrues.
    """

    def __init__(self, *,
                 q: float = 0.1,
                 default_lcb_method: str = HOEFFDING,
                 utility_lower: float = 0.0,
                 utility_upper: float = 1.0,
                 default_delta: float = 0.05,
                 default_seed: int | None = None,
                 sink: Callable[[str, Mapping[str, Any]], None] | None = None) -> None:
        self._q = _validate_quantile(q)
        if default_lcb_method not in KNOWN_LCB_METHODS:
            raise UnknownLCBMethod(
                f"default_lcb_method {default_lcb_method!r} unknown")
        self._default_lcb = default_lcb_method
        if utility_upper < utility_lower:
            raise QuantilizerError("utility_upper < utility_lower")
        self._u_lo = float(utility_lower)
        self._u_hi = float(utility_upper)
        self._default_delta = _validate_delta(default_delta)
        self._default_seed = default_seed
        self._sink = sink

        self._lock = threading.RLock()
        self._selections: list[Selection] = []
        self._quantizations: list[QuantilizedDistribution] = []
        self._observations: list[Observation] = []
        # Finite-sample stats per action (and overall).
        self._n_quant = 0
        self._sum_u_quant = 0.0
        self._sum_u2_quant = 0.0
        self._n_base = 0
        self._sum_u_base = 0.0
        self._sum_u2_base = 0.0
        self._fingerprint = _GENESIS
        self._emit(QUANTILIZER_STARTED, {
            "q": self._q,
            "default_lcb_method": self._default_lcb,
            "default_delta": self._default_delta,
            "utility_range": [self._u_lo, self._u_hi],
            "genesis": _GENESIS,
        })

    # ------------------------------------------------------------------
    # Quantilization
    # ------------------------------------------------------------------

    def quantilize(self, base: Mapping[Any, float],
                   utility: Mapping[Any, float], *,
                   q: float | None = None,
                   algorithm: str = HARD,
                   K: int | None = None,
                   kl_budget: float | None = None,
                   beta: float | None = None) -> QuantilizedDistribution:
        """Build a quantilized distribution and record it in the ledger.

        ``algorithm`` selects between HARD (default Taylor 2016 exact),
        TOP_K (requires ``K``), and SOFT (requires ``kl_budget`` or
        ``beta``).
        """
        if algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"algorithm {algorithm!r} not in {sorted(KNOWN_ALGORITHMS)}")
        qv = self._q if q is None else _validate_quantile(q)

        with self._lock:
            if algorithm == HARD:
                d = quantilize_discrete(base, utility, qv)
            elif algorithm == TOP_K:
                if K is None or K < 1:
                    raise QuantilizerError(
                        f"top_k requires K ≥ 1; got {K!r}")
                d = quantilize_top_k(base, utility, K)
            elif algorithm == SOFT:
                if kl_budget is not None:
                    d = soft_quantilize(base, utility, kl_budget=kl_budget)
                elif beta is not None:
                    d = soft_quantilize_with_beta(base, utility, beta=beta)
                else:
                    raise QuantilizerError(
                        "soft algorithm requires kl_budget or beta")
            else:  # SAMPLE handled by ``quantilize_samples``
                raise QuantilizerError(
                    "algorithm SAMPLE requires Quantilizer.from_samples(...)")
            self._quantizations.append(d)
            self._emit(QUANTILIZER_QUANTILIZED, {
                "algorithm": algorithm,
                "q": d.q,
                "kl_bound": d.kl_bound,
                "realised_kl": d.realised_kl,
                "fingerprint": d.fingerprint,
            })
            return d

    def from_samples(self, samples: Sequence[Any],
                     utility: Callable[[Any], float], *,
                     q: float | None = None,
                     delta: float | None = None) -> SampleQuantilization:
        """Empirical sample-based quantilizer with DKW quantile band."""
        qv = self._q if q is None else _validate_quantile(q)
        dv = self._default_delta if delta is None else _validate_delta(delta)
        with self._lock:
            sq = quantilize_samples(samples, utility, qv, delta=dv)
            self._emit(QUANTILIZER_QUANTILIZED, {
                "algorithm": SAMPLE,
                "q": sq.q,
                "n_total": sq.n_total,
                "n_kept": sq.n_kept,
                "threshold": sq.threshold,
                "dkw_band": sq.dkw_band,
                "fingerprint": sq.fingerprint,
            })
            return sq

    # ------------------------------------------------------------------
    # Selection (sample)
    # ------------------------------------------------------------------

    def select(self, base: Mapping[Any, float],
               utility: Mapping[Any, float], *,
               q: float | None = None,
               algorithm: str = HARD,
               K: int | None = None,
               kl_budget: float | None = None,
               beta: float | None = None,
               seed: int | None = None) -> Selection:
        """Quantilize and draw a single action; record in the ledger.

        Deterministic given (base, utility, q, algorithm, seed).  The
        returned ``Selection`` carries all safety bounds, the realised
        KL of the distribution from which the action was drawn, and a
        SHA-256 fingerprint chaining into the running ledger.
        """
        with self._lock:
            d = self.quantilize(base, utility, q=q, algorithm=algorithm,
                                K=K, kl_budget=kl_budget, beta=beta)
            s = seed if seed is not None else (
                self._default_seed if self._default_seed is not None
                else random.randint(0, 2**31 - 1))
            action, p_quant = sample_from_distribution(d, seed=s)
            p_base = float(d.base_probs.get(action, 0.0))
            u = float(d.utilities[action])

            parent = self._fingerprint
            fp = _hash({
                "kind": "select",
                "parent": parent,
                "dist_fingerprint": d.fingerprint,
                "action": _action_key(action),
                "seed": s,
                "p_quant": p_quant,
                "p_base": p_base,
                "utility": u,
            })
            self._fingerprint = fp

            sel = Selection(
                action=action,
                q=d.q,
                algorithm=d.algorithm,
                probability_under_base=p_base,
                probability_under_quantilizer=p_quant,
                utility=u,
                threshold=d.threshold,
                kl_bound=d.kl_bound,
                tv_bound=d.tv_bound,
                realised_kl=d.realised_kl,
                cost_amplification=d.cost_amplification,
                seed=int(s),
                fingerprint=fp,
                parent_fingerprint=parent,
                timestamp=time.time(),
            )
            self._selections.append(sel)
            self._emit(QUANTILIZER_SELECTED, {
                "action_key": _action_key(action),
                "q": d.q,
                "algorithm": d.algorithm,
                "kl_bound": d.kl_bound,
                "realised_kl": d.realised_kl,
                "cost_amplification": d.cost_amplification,
                "p_base": p_base,
                "p_quant": p_quant,
                "utility": u,
                "fingerprint": fp,
            })
            return sel

    # ------------------------------------------------------------------
    # Observation: record realised utility outcomes
    # ------------------------------------------------------------------

    def observe(self, action: Any, utility: float, *,
                came_from_quantilizer: bool = True) -> None:
        """Record a realised (action, utility) pair.

        Updates running mean / variance estimates so anytime LCBs /
        UCBs can be queried.  Set ``came_from_quantilizer=False`` to
        log a base-policy outcome (used for the cost-UCB calculation).
        """
        if not isinstance(utility, (int, float)) or math.isnan(float(utility)):
            raise InvalidUtility(f"observe utility must be numeric; got {utility!r}")
        uv = float(utility)
        with self._lock:
            self._observations.append(Observation(
                action=action,
                utility=uv,
                came_from_quantilizer=came_from_quantilizer,
                timestamp=time.time(),
            ))
            if came_from_quantilizer:
                self._n_quant += 1
                self._sum_u_quant += uv
                self._sum_u2_quant += uv * uv
            else:
                self._n_base += 1
                self._sum_u_base += uv
                self._sum_u2_base += uv * uv
            parent = self._fingerprint
            fp = _hash({
                "kind": "observe",
                "parent": parent,
                "action": _action_key(action),
                "utility": uv,
                "came_from_quantilizer": came_from_quantilizer,
            })
            self._fingerprint = fp
            self._emit(QUANTILIZER_OBSERVED, {
                "action_key": _action_key(action),
                "utility": uv,
                "came_from_quantilizer": came_from_quantilizer,
                "n_quant": self._n_quant,
                "n_base": self._n_base,
                "fingerprint": fp,
            })

    # ------------------------------------------------------------------
    # Finite-sample bounds on expected utility under the quantilizer
    # ------------------------------------------------------------------

    def expected_utility_lcb(self, *, delta: float | None = None,
                              method: str | None = None,
                              source: str = "quantilizer") -> UtilityBound:
        """Finite-sample LCB on the expected utility under the quantilizer.

        ``source`` is "quantilizer" (use observations from quantilizer
        draws) or "base" (use base-policy observations).  Method:
        Hoeffding / Bernstein / anytime.
        """
        m = method if method is not None else self._default_lcb
        if m not in KNOWN_LCB_METHODS or m == DKW:
            raise UnknownLCBMethod(
                f"utility LCB method {m!r} unsupported")
        dv = self._default_delta if delta is None else _validate_delta(delta)

        with self._lock:
            if source == "quantilizer":
                n = self._n_quant
                s = self._sum_u_quant
                s2 = self._sum_u2_quant
            elif source == "base":
                n = self._n_base
                s = self._sum_u_base
                s2 = self._sum_u2_base
            else:
                raise QuantilizerError(
                    f"source must be 'quantilizer' or 'base'; got {source!r}")
            if n < 1:
                raise InsufficientData(
                    f"no observations on source={source!r}")
            mean = s / n
            var = max(0.0, s2 / n - mean * mean) if n >= 2 else 0.0
            # unbiased sample variance
            if n >= 2:
                var = var * n / (n - 1)
            if m == HOEFFDING:
                lcb = hoeffding_lcb(mean, n, delta=dv,
                                    lower=self._u_lo, upper=self._u_hi)
                ucb = hoeffding_ucb(mean, n, delta=dv,
                                    lower=self._u_lo, upper=self._u_hi)
            elif m == BERNSTEIN:
                if n < 2:
                    raise InsufficientData("Bernstein requires n ≥ 2")
                lcb = empirical_bernstein_lcb(mean, var, n, delta=dv,
                                              lower=self._u_lo,
                                              upper=self._u_hi)
                ucb = empirical_bernstein_ucb(mean, var, n, delta=dv,
                                              lower=self._u_lo,
                                              upper=self._u_hi)
            elif m == ANYTIME:
                lcb = anytime_lcb(mean, n, delta=dv,
                                  lower=self._u_lo, upper=self._u_hi)
                ucb = anytime_ucb(mean, n, delta=dv,
                                  lower=self._u_lo, upper=self._u_hi)
            else:
                raise UnknownLCBMethod(f"method {m!r} unsupported here")
            return UtilityBound(
                method=m,
                mean=mean,
                lcb=lcb,
                ucb=ucb,
                n=n,
                delta=dv,
                variance=var,
                lower=self._u_lo,
                upper=self._u_hi,
            )

    def cost_ucb(self, base_cost_mean_ucb: float, *,
                 q: float | None = None) -> CostBound:
        r"""Hidden-cost UCB via Taylor-2016 amplification.

        Given a coordinator-supplied UCB ``C`` on ``E_b[c]`` for some
        non-negative hidden cost, return the matching UCB on
        ``E_q[c]`` (the quantilizer's expected hidden cost).

        Theorem (Taylor 2016): ``E_q[c] ≤ E_b[c] / q``, so
        ``UCB(E_q[c]) = UCB(E_b[c]) / q``.
        """
        qv = self._q if q is None else _validate_quantile(q)
        if not isinstance(base_cost_mean_ucb, (int, float)):
            raise QuantilizerError("base cost UCB must be numeric")
        if base_cost_mean_ucb < 0.0:
            raise QuantilizerError("base cost UCB must be ≥ 0")
        amp = 1.0 / qv
        return CostBound(
            base_cost_ucb=float(base_cost_mean_ucb),
            amplification=amp,
            quantilizer_cost_ucb=base_cost_mean_ucb * amp,
            n_base_samples=self._n_base,
            delta=self._default_delta,
            method="taylor2016",
        )

    def quantile_estimate(self, q: float | None = None, *,
                           delta: float | None = None) -> QuantileEstimate:
        """Empirical (1 − q)-quantile of utility under base, with DKW band.

        Uses the base-source observations.  Raises if none recorded.
        """
        qv = self._q if q is None else _validate_quantile(q)
        dv = self._default_delta if delta is None else _validate_delta(delta)
        with self._lock:
            base_utilities = [o.utility for o in self._observations
                              if not o.came_from_quantilizer]
            if not base_utilities:
                raise InsufficientData(
                    "no base-source observations to estimate quantile")
            sorted_u = sorted(base_utilities)
            n = len(sorted_u)
            target = 1.0 - qv
            idx = int(math.ceil(target * n)) - 1
            idx = _clip(idx, 0, n - 1)
            estimate = sorted_u[idx]
            lcb, ucb = quantile_lcb_dkw(sorted_u, qv, delta=dv)
            eps = dkw_band(n, delta=dv)
            return QuantileEstimate(
                q=qv,
                estimate=estimate,
                lcb=lcb,
                ucb=ucb,
                dkw_band=eps,
                n_samples=n,
                delta=dv,
            )

    # ------------------------------------------------------------------
    # Information-theoretic bound aggregators
    # ------------------------------------------------------------------

    def divergence_certificates(self, q: float | None = None) -> dict:
        """All divergence bounds for a quantilizer at the given q."""
        qv = self._q if q is None else _validate_quantile(q)
        kl = kl_bound_from_quantile(qv)
        tv = tv_bound_from_quantile(qv)
        return {
            "kl_nats": kl,
            "kl_bits": kl / _LN2,
            "tv": tv,
            "cost_amplification": cost_amplification(qv),
            "pinsker_tv_from_kl": pinsker_tv_from_kl(kl),
            "bretagnolle_huber_tv_from_kl": bretagnolle_huber_tv_from_kl(kl),
            "le_cam_overlap": le_cam_overlap_from_tv(tv),
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> QuantilizerReport:
        with self._lock:
            cumulative_kl = sum(q.realised_kl for q in self._quantizations)
            r = QuantilizerReport(
                n_selections=len(self._selections),
                n_observations=len(self._observations),
                n_quantilized=len(self._quantizations),
                cumulative_kl_bound=cumulative_kl,
                last_selection=self._selections[-1] if self._selections else None,
                last_quantilized=self._quantizations[-1] if self._quantizations else None,
                fingerprint=self._fingerprint,
                genesis=_GENESIS,
                config={
                    "q": self._q,
                    "default_lcb_method": self._default_lcb,
                    "default_delta": self._default_delta,
                    "utility_range": [self._u_lo, self._u_hi],
                },
            )
            self._emit(QUANTILIZER_REPORT, {
                "n_selections": r.n_selections,
                "n_observations": r.n_observations,
                "n_quantilized": r.n_quantilized,
                "fingerprint": r.fingerprint,
            })
            return r

    def clear(self) -> None:
        """Reset everything; fingerprint returns to genesis."""
        with self._lock:
            self._selections.clear()
            self._quantizations.clear()
            self._observations.clear()
            self._n_quant = self._n_base = 0
            self._sum_u_quant = self._sum_u2_quant = 0.0
            self._sum_u_base = self._sum_u2_base = 0.0
            self._fingerprint = _GENESIS
            self._emit(QUANTILIZER_CLEARED, {"genesis": _GENESIS})

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def q(self) -> float:
        return self._q

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def selections(self) -> tuple[Selection, ...]:
        return tuple(self._selections)

    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def quantizations(self) -> tuple[QuantilizedDistribution, ...]:
        return tuple(self._quantizations)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._sink is None:
            return
        try:
            self._sink(kind, payload)
        except Exception:
            # Sinks must not break the runtime.
            pass


# =====================================================================
# Spec factory
# =====================================================================


def quantilizer_from_spec(spec: Mapping[str, Any]) -> Quantilizer:
    """Build a ``Quantilizer`` from a JSON-able spec.

    Expected shape::

        {
          "q": 0.05,
          "default_lcb_method": "hoeffding",
          "default_delta": 0.05,
          "utility_lower": 0.0,
          "utility_upper": 1.0,
          "default_seed": 17
        }
    """
    if not isinstance(spec, Mapping):
        raise QuantilizerError(
            f"spec must be a mapping; got {type(spec).__name__}")
    q = float(spec.get("q", 0.1))
    method = spec.get("default_lcb_method", HOEFFDING)
    delta = float(spec.get("default_delta", 0.05))
    lo = float(spec.get("utility_lower", 0.0))
    hi = float(spec.get("utility_upper", 1.0))
    seed = spec.get("default_seed")
    if seed is not None:
        seed = int(seed)
    return Quantilizer(
        q=q,
        default_lcb_method=method,
        default_delta=delta,
        utility_lower=lo,
        utility_upper=hi,
        default_seed=seed,
    )


# =====================================================================
# Composition adapters (lightweight; do not import sibling modules)
# =====================================================================


def quantilize_bandit_distribution(arm_probs: Mapping[Any, float],
                                    expected_rewards: Mapping[Any, float],
                                    q: float) -> QuantilizedDistribution:
    """Compose with a Bandit's mixed policy.

    A bandit's per-step policy ``π_t(a)`` (e.g. Thompson, EXP3 weights)
    composed with the bandit's posterior mean reward ``μ̂_a`` becomes a
    quantilized policy with KL bound ``log(1/q)`` from the bandit's
    own policy — a *safe exploration* wrapper.
    """
    return quantilize_discrete(arm_probs, expected_rewards, q)


def quantilize_policy_improvement(deployed_policy: Mapping[Any, float],
                                   proposed_score: Mapping[Any, float],
                                   *, kl_budget: float) -> QuantilizedDistribution:
    """KL-bounded safe policy improvement step.

    Given a currently-deployed policy and a per-action quality score
    (e.g. CRM-estimated reward), return the soft-quantilizer that lies
    on the KL-budget frontier from ``deployed_policy``.
    """
    return soft_quantilize(deployed_policy, proposed_score, kl_budget=kl_budget)
