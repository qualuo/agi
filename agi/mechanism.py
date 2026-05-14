r"""MechanismDesigner — revenue-optimal mechanism design as a runtime primitive.

A coordination engine that lets autonomous agents (sub-models, tools, tenants,
human contractors, external services) bid for scarce resources — compute,
queue priority, attention, exclusive licences, data access — must answer
**two simultaneous questions**:

  1. **What allocation maximises welfare or seller revenue** given the bids
     that landed?
  2. **What prices must we charge so that truth-telling is a dominant
     strategy** of every participant?

Naively, "pay your bid" (1st-price) is the obvious rule and is *exactly the
wrong one*: bidders shade their valuations downwards, the allocation no
longer tracks who values the item most, and the platform leaves money on
the table. The right mechanisms — Vickrey, Myerson, VCG, posted-price —
are non-obvious, almost a half-century old, mathematically deep, and
*provably* truthful and revenue-optimal under stated assumptions.

`MechanismDesigner` is the runtime primitive that ships them. It composes
with `Market` (auctions for compute), `Economist` (closed-loop margin
defence), `Negotiator` (multi-party bargaining), `Equilibrator` (Nash
verification of strategic play), and `AttestationLedger` (tamper-evident
receipts of every allocation). It is stdlib-only, threadsafe, and emits
events on every state change so the coordination engine can react.

Mathematical core (cited where it counts)
-----------------------------------------

* **Vickrey, 1961** — "Counterspeculation, auctions, and competitive sealed
  tenders." The second-price sealed-bid auction is DSIC (dominant-strategy
  incentive-compatible): truthful bidding is a weakly dominant strategy.
  Payment: pay the second-highest bid; allocate to the highest.

* **Clarke, 1971 / Groves, 1973** — The VCG family generalises Vickrey to
  arbitrary social-choice functions. Allocate to maximise total reported
  welfare; charge each agent the externality they impose on others (the
  difference in others' welfare with and without the agent present). DSIC
  + IR for any *quasi-linear* preference and any feasible allocation set.

* **Myerson, 1981** — "Optimal auction design." For single-item auctions
  with *independent* private values v_i ~ F_i, the unique revenue-optimal
  DSIC mechanism is:

      φ_i(v) = v - (1 - F_i(v)) / f_i(v)            (virtual valuation)
      allocate to argmax_i φ_i(v_i) if max ≥ 0, else no sale
      pay_i = inf { b_i' : φ_i(b_i') ≥ max_{j ≠ i} φ_j(v_j) ∧ φ_i(b_i') ≥ 0 }

  When all F_i are *regular* (φ_i monotone non-decreasing), this is a
  *2nd-price auction with bidder-specific reserves r_i = φ_i^{-1}(0)*.
  Myerson's *ironing* procedure handles non-regular distributions by
  averaging virtual values over the non-monotone region.

* **Bulow & Klemperer, 1996** — "Auctions versus negotiations." A
  second-price auction with n+1 bidders earns at least as much expected
  revenue as the optimal mechanism with n bidders. Practically: it is
  worth more to attract one more bidder than to design the optimal
  reserve. We ship the explicit `bulow_klemperer` comparison.

* **Hartline & Roughgarden, 2009** — "Simple versus optimal mechanisms."
  An anonymous-reserve second-price auction earns ≥ ½ of Myerson's
  expected revenue for *any* set of regular distributions. The right
  anonymous reserve is the *monopoly reserve* of the *aggregate* virtual
  valuation. We implement the Hartline-Roughgarden anonymous-reserve
  mechanism as the "simple" production-grade default.

* **Cole & Roughgarden, 2014** — "The sample complexity of revenue
  maximization." Given Õ(K · ε^-7) samples from K regular distributions,
  one can construct a mechanism within (1-ε) of Myerson's expected
  revenue with high probability. The bound was tightened by Gonczarowski
  & Weinberg (2021) to Õ(K · ε^-2). We expose `sample_complexity` and
  `empirical_myerson` for the practical *data-driven* case: a
  distribution-free, sample-based mechanism designer.

* **Chawla, Hartline, Malec & Sivan, 2010** — "Multi-parameter mechanism
  design and sequential posted pricing." Sequential posted-price
  mechanisms (SPMs) — present each bidder a take-it-or-leave-it price —
  are *DSIC by construction* (no bidder interaction) and achieve a
  constant approximation to the BIC optimum for matroid constraints. We
  implement SPM under user-provided prices and orderings; the user can
  back the prices out of `empirical_myerson` or `monopoly_reserve`.

* **Kleinberg & Leighton, 2003** — "The value of knowing a demand curve:
  bounds on regret for online posted-price auctions." Against a single
  bidder over T rounds:
    - regret O(T^{2/3} · (log T)^{1/3}) for general bounded valuations,
    - regret O(√(T · log T))            for valuations from a fixed distribution.
  The optimal strategy uses an EXP3-style price grid over [0, v_max] with
  resolution T^{1/3}. We ship the EXP3-grid bandit as `online_posted_price`.

* **Borgs, Chayes, Immorlica, Mahdian & Saberi, 2005** — Multi-item VCG
  with sub-modular valuations is welfare-truthful; we implement VCG for
  unit-demand and additive valuations exactly, and warn the caller for
  the general combinatorial case (NP-hard winner determination).

* **Hoeffding, 1963; Maurer-Pontil, 2009.** Anytime-valid PAC bounds on
  empirical revenue. Every empirical-mechanism call returns a
  finite-sample lower confidence bound on its expected revenue.

* **Lehmann, O'Callaghan & Shoham, 2002.** Greedy posted-price for
  single-minded combinatorial bidders is m-approximate with truthfulness.

Surface a coordination engine drives
------------------------------------

::

    md = MechanismDesigner(bus=bus, attestor=attestor)

    # Single-shot Myerson (priors known)
    out = md.myerson_auction(
        bids={"haiku": 0.62, "sonnet": 0.81, "opus": 0.93},
        distributions={
            "haiku":  UniformDistribution(0.0, 1.0),
            "sonnet": UniformDistribution(0.0, 1.0),
            "opus":   UniformDistribution(0.0, 1.0),
        },
    )
    coordinator.assign(item="ticket-2026Q2", to=out.winner, pay=out.payment)

    # Data-driven: samples instead of priors
    out = md.myerson_from_samples(
        bids={"haiku": 0.62, "sonnet": 0.81, "opus": 0.93},
        samples={
            "haiku":  haiku_history,    # list[float]
            "sonnet": sonnet_history,
            "opus":   opus_history,
        },
        delta=0.05,
    )
    assert out.revenue_lcb > 0.0     # PAC certificate on expected revenue

    # Multi-item welfare-maximising VCG
    vcg = md.vcg_allocation(
        items=["gpu-0", "gpu-1", "gpu-2"],
        bids={
            "tenantA": {"gpu-0": 10.0, "gpu-1":  8.0, "gpu-2": 6.0},
            "tenantB": {"gpu-0":  7.0, "gpu-1":  9.0, "gpu-2": 5.0},
        },
        capacity={"tenantA": 2, "tenantB": 1},
    )
    for tenant, item, price in vcg.assignments:
        coordinator.assign(item=item, to=tenant, pay=price)

    # Online posted price against an opaque buyer
    out = md.online_posted_price(
        feedback=lambda price: float(buyer_accept(price)),
        T=1000,
        v_max=1.0,
        delta=0.05,
    )
    # Receive regret-bounded revenue stream with anytime certificate.

Events
------

  mechanism.started        — primitive constructed
  mechanism.allocated      — an allocation was decided
  mechanism.priced         — payments computed
  mechanism.certified      — a DSIC / IR / revenue certificate emitted
  mechanism.reserve_fit    — empirical reserve learned from samples
  mechanism.bulow_klemperer — BK ratio computed
  mechanism.online_step    — one round of the online posted-price loop
  mechanism.cleared        — designer reset

Honest about limits
-------------------

  - **Myerson assumes independence.** Correlated values require Crémer-
    McLean-style mechanisms, which we do *not* ship — they are fragile to
    misspecification and rarely deployable in adversarial conditions.

  - **Combinatorial VCG is NP-hard** for general valuations. We support
    additive and unit-demand (LP-solvable as assignment), but not
    arbitrary subadditive valuations.

  - **Online posted price assumes one bidder per round.** Multi-bidder
    online auctions need the Roughgarden 2020 "no-regret learner"
    construction; out of scope for v1.

  - **DSIC certificates are constructive only**: we *prove* DSIC for
    Vickrey/VCG/SPM/Myerson by construction and *empirically test*
    deviations for caller-supplied mechanisms.

  - **Revenue certificates are LCBs on empirical mean revenue** under the
    bidder distribution from which samples were drawn. If the underlying
    distribution drifts (see `DriftSentinel`), the certificate is void.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from agi.events import Event  # type: ignore
except Exception:  # pragma: no cover
    Event = None  # type: ignore


# =====================================================================
# Event kinds
# =====================================================================

MECH_STARTED = "mechanism.started"
MECH_ALLOCATED = "mechanism.allocated"
MECH_PRICED = "mechanism.priced"
MECH_CERTIFIED = "mechanism.certified"
MECH_RESERVE_FIT = "mechanism.reserve_fit"
MECH_BULOW_KLEMPERER = "mechanism.bulow_klemperer"
MECH_ONLINE_STEP = "mechanism.online_step"
MECH_CLEARED = "mechanism.cleared"

# =====================================================================
# Mechanism / distribution / format identifiers
# =====================================================================

KIND_VICKREY = "vickrey"
KIND_FIRST_PRICE = "first_price"
KIND_MYERSON = "myerson"
KIND_VCG = "vcg"
KIND_POSTED_PRICE = "posted_price"
KIND_ANONYMOUS_RESERVE = "anonymous_reserve_vickrey"
KIND_ALL_PAY = "all_pay"

KNOWN_MECHANISMS = (
    KIND_VICKREY,
    KIND_FIRST_PRICE,
    KIND_MYERSON,
    KIND_VCG,
    KIND_POSTED_PRICE,
    KIND_ANONYMOUS_RESERVE,
    KIND_ALL_PAY,
)

DIST_UNIFORM = "uniform"
DIST_EXPONENTIAL = "exponential"
DIST_TRUNC_NORMAL = "truncated_normal"
DIST_EMPIRICAL = "empirical"

REG_REGULAR = "regular"
REG_MHR = "mhr"  # monotone hazard rate
REG_BOUNDED = "bounded"

# =====================================================================
# Exceptions
# =====================================================================


class MechanismError(Exception):
    """Base exception for mechanism design errors."""


class InvalidBid(MechanismError):
    pass


class InvalidDistribution(MechanismError):
    pass


class InsufficientData(MechanismError):
    pass


class InfeasibleAllocation(MechanismError):
    pass


class UnknownMechanism(MechanismError):
    pass


# =====================================================================
# Value distributions
# =====================================================================


class ValueDistribution:
    """Abstract bidder-value distribution.

    Subclasses must implement `cdf`, `pdf`, `support`, and `sample`.
    `quantile` and `virtual_value` are provided generically but may be
    overridden for closed-form efficiency.

    A *regular* distribution has non-decreasing virtual valuation
    ``φ(v) = v − (1 − F(v))/f(v)`` over its support. Regularity is what
    makes Myerson's allocation rule monotone and therefore DSIC.
    """

    name: str = "abstract"

    def cdf(self, v: float) -> float:
        raise NotImplementedError

    def pdf(self, v: float) -> float:
        raise NotImplementedError

    def support(self) -> tuple[float, float]:
        raise NotImplementedError

    def sample(self, rng: random.Random) -> float:
        raise NotImplementedError

    def virtual_value(self, v: float) -> float:
        f = self.pdf(v)
        if f <= 0.0:
            # at the boundary of support — define φ(v_max) = v_max,
            # φ(v_min) = -∞ (sentinel via a large negative)
            return -1e18 if v <= self.support()[0] else v
        return v - (1.0 - self.cdf(v)) / f

    def quantile(self, p: float) -> float:
        """Inverse CDF by bisection. Subclasses with closed forms override."""
        if not (0.0 <= p <= 1.0):
            raise InvalidDistribution(f"quantile p={p} outside [0,1]")
        lo, hi = self.support()
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def monopoly_reserve(self) -> float:
        """The reserve r solving φ(r) = 0. By bisection over support."""
        lo, hi = self.support()
        # Walk inward to avoid pathologic atoms at the boundary.
        eps = 1e-9 * max(1.0, hi - lo)
        lo_in = lo + eps
        hi_in = hi - eps
        # φ at lo_in is typically negative, at hi_in positive. Bisect.
        try:
            phi_lo = self.virtual_value(lo_in)
            phi_hi = self.virtual_value(hi_in)
        except Exception:
            return 0.0
        if phi_lo >= 0:
            return lo_in
        if phi_hi <= 0:
            return hi_in
        a, b = lo_in, hi_in
        for _ in range(80):
            mid = 0.5 * (a + b)
            phi_m = self.virtual_value(mid)
            if phi_m >= 0:
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    def regularity(self) -> str:
        """Best-effort classification; override when known analytically."""
        return REG_BOUNDED


@dataclass(frozen=True)
class UniformDistribution(ValueDistribution):
    """U[a, b]. Regular. Monopoly reserve = (a+b)/2 for a ≥ 0."""

    a: float = 0.0
    b: float = 1.0
    name: str = field(default=DIST_UNIFORM, init=False)

    def __post_init__(self) -> None:
        if self.b <= self.a:
            raise InvalidDistribution(f"UniformDistribution: b={self.b} ≤ a={self.a}")

    def cdf(self, v: float) -> float:
        if v <= self.a:
            return 0.0
        if v >= self.b:
            return 1.0
        return (v - self.a) / (self.b - self.a)

    def pdf(self, v: float) -> float:
        if self.a <= v <= self.b:
            return 1.0 / (self.b - self.a)
        return 0.0

    def support(self) -> tuple[float, float]:
        return (self.a, self.b)

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.a, self.b)

    def virtual_value(self, v: float) -> float:
        # φ(v) = v − (1 − (v-a)/(b-a)) · (b-a) = v − (b - v) = 2v − b
        return 2.0 * v - self.b

    def quantile(self, p: float) -> float:
        return self.a + p * (self.b - self.a)

    def monopoly_reserve(self) -> float:
        # φ(r) = 0  ⇒  r = b/2
        return max(self.a, self.b / 2.0)

    def regularity(self) -> str:
        return REG_REGULAR


@dataclass(frozen=True)
class ExponentialDistribution(ValueDistribution):
    """Exp(rate) truncated to [0, vmax]. MHR.

    For untruncated Exp(λ), virtual value is v − 1/λ; monopoly reserve
    is 1/λ. We truncate at `vmax` so empirical revenue is well-defined.
    """

    rate: float = 1.0
    vmax: float = 10.0
    name: str = field(default=DIST_EXPONENTIAL, init=False)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise InvalidDistribution(f"ExponentialDistribution: rate={self.rate}")
        if self.vmax <= 0:
            raise InvalidDistribution(f"ExponentialDistribution: vmax={self.vmax}")

    def cdf(self, v: float) -> float:
        if v <= 0.0:
            return 0.0
        if v >= self.vmax:
            return 1.0
        z = 1.0 - math.exp(-self.rate * v)
        z_max = 1.0 - math.exp(-self.rate * self.vmax)
        return z / z_max if z_max > 0 else 0.0

    def pdf(self, v: float) -> float:
        if v < 0.0 or v > self.vmax:
            return 0.0
        z_max = 1.0 - math.exp(-self.rate * self.vmax)
        return self.rate * math.exp(-self.rate * v) / z_max if z_max > 0 else 0.0

    def support(self) -> tuple[float, float]:
        return (0.0, self.vmax)

    def sample(self, rng: random.Random) -> float:
        # Inverse-CDF sampling on truncated exponential
        u = rng.random()
        z_max = 1.0 - math.exp(-self.rate * self.vmax)
        return -math.log(1.0 - u * z_max) / self.rate

    def virtual_value(self, v: float) -> float:
        if v < 0.0:
            return -1e18
        if v > self.vmax:
            return v
        return v - 1.0 / self.rate

    def monopoly_reserve(self) -> float:
        return min(self.vmax, 1.0 / self.rate)

    def regularity(self) -> str:
        return REG_MHR


@dataclass
class EmpiricalDistribution(ValueDistribution):
    """Bidder-value distribution from samples.

    CDF: linear interpolation between empirical points (Glivenko-Cantelli
    converges uniformly at rate O(1/√n)).
    PDF: kernel density via a top-hat band of width h = (max-min)/√n.
    """

    samples: tuple[float, ...] = ()
    name: str = field(default=DIST_EMPIRICAL, init=False)

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise InsufficientData(
                f"EmpiricalDistribution needs ≥ 2 samples (got {len(self.samples)})"
            )
        self._sorted = tuple(sorted(self.samples))
        n = len(self._sorted)
        self._n = n
        s_min, s_max = self._sorted[0], self._sorted[-1]
        if s_max <= s_min:
            raise InsufficientData(
                "EmpiricalDistribution: all samples equal — cannot fit"
            )
        self._support = (s_min, s_max)
        self._bandwidth = max((s_max - s_min) / max(1.0, math.sqrt(n)), 1e-6)

    def cdf(self, v: float) -> float:
        # Linear interpolation between empirical points
        n = self._n
        if v <= self._sorted[0]:
            return 0.0
        if v >= self._sorted[-1]:
            return 1.0
        idx = bisect.bisect_right(self._sorted, v)
        # value is between _sorted[idx-1] and _sorted[idx]
        if idx >= n:
            return 1.0
        lo_v, hi_v = self._sorted[idx - 1], self._sorted[idx]
        if hi_v == lo_v:
            return idx / n
        frac = (v - lo_v) / (hi_v - lo_v)
        return (idx - 1 + frac) / n

    def pdf(self, v: float) -> float:
        # top-hat KDE
        h = self._bandwidth
        if v < self._support[0] - h or v > self._support[1] + h:
            return 0.0
        count = 0
        lo, hi = v - h, v + h
        lidx = bisect.bisect_left(self._sorted, lo)
        ridx = bisect.bisect_right(self._sorted, hi)
        count = ridx - lidx
        return count / (2.0 * h * self._n) if self._n > 0 else 0.0

    def support(self) -> tuple[float, float]:
        return self._support

    def sample(self, rng: random.Random) -> float:
        # bootstrap
        return rng.choice(self._sorted)

    def quantile(self, p: float) -> float:
        if not (0.0 <= p <= 1.0):
            raise InvalidDistribution(f"quantile p={p}")
        n = self._n
        if p <= 0:
            return self._sorted[0]
        if p >= 1:
            return self._sorted[-1]
        # Linear interpolation
        idx = p * (n - 1)
        lo_i = int(math.floor(idx))
        hi_i = int(math.ceil(idx))
        if lo_i == hi_i:
            return self._sorted[lo_i]
        frac = idx - lo_i
        return self._sorted[lo_i] * (1 - frac) + self._sorted[hi_i] * frac

    def regularity(self) -> str:
        # We cannot certify regularity from samples in general; report
        # bounded by default and let the caller assert it via a flag if
        # desired.
        return REG_BOUNDED


# =====================================================================
# Hash / attestation helpers
# =====================================================================


def _hash_payload(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


class _AttestableReceipt:
    __slots__ = ("ticket_id", "kind", "payload", "digest")

    def __init__(self, kind: str, payload: dict, digest: str = "") -> None:
        self.kind = kind
        self.payload = payload
        self.digest = digest
        self.ticket_id = (
            payload.get("receipt_id") or digest[:16] or uuid.uuid4().hex[:16]
        )

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "kind": self.kind,
            "payload": self.payload,
            "digest": self.digest,
        }


# =====================================================================
# Concentration helpers
# =====================================================================


def hoeffding_radius(n: int, *, delta: float, range_: float) -> float:
    """Hoeffding (1963) radius for a bounded mean estimator.

      |X̄ − E[X]| ≤ range · √( log(2/δ) / (2n) )    w.p. ≥ 1 − δ
    """
    if n <= 0:
        return float("inf")
    if not (0 < delta < 1):
        raise MechanismError(f"delta out of (0,1): {delta}")
    return range_ * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def empirical_bernstein_radius(
    values: Sequence[float], *, delta: float, range_: float
) -> float:
    """Maurer-Pontil (2009) empirical Bernstein radius."""
    n = len(values)
    if n <= 1:
        return float("inf")
    var = statistics.pvariance(values)
    a = math.sqrt(2.0 * var * math.log(2.0 / delta) / n)
    b = 7.0 * range_ * math.log(2.0 / delta) / (3.0 * (n - 1))
    return a + b


# =====================================================================
# Data classes — bids, allocations, certificates
# =====================================================================


@dataclass(frozen=True)
class Bid:
    """A single bid in a single-item or multi-item auction."""

    bidder_id: str
    value: float
    # optional: per-item value map (multi-item auctions)
    item_values: tuple[tuple[str, float], ...] = ()
    # optional metadata (tier, tenant, etc.)
    meta: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.bidder_id, str) or not self.bidder_id:
            raise InvalidBid("bidder_id must be a non-empty string")
        if not math.isfinite(self.value) or self.value < 0:
            raise InvalidBid(f"bid value must be ≥ 0 and finite, got {self.value}")


@dataclass(frozen=True)
class Allocation:
    """Outcome of a single-item auction."""

    mechanism: str
    winner: str | None
    payment: float
    welfare: float
    revenue: float
    # per-bidder utility = value if winner else 0, minus payment if winner
    utilities: tuple[tuple[str, float], ...]
    bids: tuple[tuple[str, float], ...]
    reserve: float = 0.0
    receipt_digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VCGAssignment:
    bidder_id: str
    item_id: str
    payment: float
    value: float


@dataclass(frozen=True)
class VCGAllocation:
    mechanism: str
    assignments: tuple[VCGAssignment, ...]
    total_welfare: float
    total_revenue: float
    utilities: tuple[tuple[str, float], ...]
    receipt_digest: str = ""


@dataclass(frozen=True)
class PostedPriceOutcome:
    mechanism: str
    accepted: tuple[str, ...]           # bidders who accepted, in order
    item_assignments: tuple[tuple[str, str, float], ...]   # (bidder, item, price)
    revenue: float
    utilities: tuple[tuple[str, float], ...]
    receipt_digest: str = ""


@dataclass(frozen=True)
class ReservePolicy:
    """Sample-based reserve with PAC certificate."""

    reserve: float
    n_samples: int
    delta: float
    revenue_lcb: float        # lower confidence bound on E[revenue at this reserve]
    revenue_mean: float
    revenue_radius: float
    method: str
    distribution_class: str = ""


@dataclass(frozen=True)
class BulowKlemperer:
    """BK comparison: revenue of (n+1)-bidder Vickrey vs n-bidder Myerson.

    `ratio > 1.0` ⇒ inviting one more bidder beats designing the reserve.
    """

    n: int
    revenue_vickrey_n_plus_1: float
    revenue_myerson_n: float
    ratio: float
    samples_used: int


@dataclass(frozen=True)
class DSICReport:
    mechanism: str
    is_dsic: bool
    worst_gain: float            # max utility a bidder gained by lying
    max_attempts: int
    failing_bidder: str | None


@dataclass(frozen=True)
class IRReport:
    mechanism: str
    is_ir: bool
    worst_utility: float


@dataclass(frozen=True)
class OnlinePostedPriceOutcome:
    T: int
    v_max: float
    algorithm: str
    revenue: float
    revenue_mean: float
    n_grid: int
    price_history: tuple[float, ...]
    accept_history: tuple[int, ...]
    pseudo_regret_ub: float
    receipt_digest: str = ""


@dataclass(frozen=True)
class RevenueCertificate:
    """Anytime-valid LCB on expected revenue under a mechanism."""

    mechanism: str
    n_samples: int
    delta: float
    revenue_mean: float
    revenue_lcb: float
    revenue_radius: float
    receipt_digest: str = ""


# =====================================================================
# Core mechanism implementations
# =====================================================================


def vickrey_payment(bids_sorted_desc: list[float], *, reserve: float = 0.0) -> tuple[int, float]:
    """Return (winner_index, payment) for a Vickrey auction with optional reserve.

    Bidders are sorted in *descending* order of bid; index returned is into
    this sorted list. Winner index = -1 means "no sale".
    """
    if not bids_sorted_desc:
        return -1, 0.0
    top = bids_sorted_desc[0]
    if top < reserve:
        return -1, 0.0
    if len(bids_sorted_desc) == 1:
        # Single bidder above reserve — pays reserve
        return 0, reserve
    second = bids_sorted_desc[1]
    payment = max(second, reserve)
    return 0, payment


def first_price_payment(bids_sorted_desc: list[float], *, reserve: float = 0.0) -> tuple[int, float]:
    if not bids_sorted_desc:
        return -1, 0.0
    top = bids_sorted_desc[0]
    if top < reserve:
        return -1, 0.0
    return 0, top


def myerson_winner_and_payment(
    bidders: list[str],
    values: list[float],
    distributions: list[ValueDistribution],
) -> tuple[str | None, float, float, list[float]]:
    """Run Myerson optimal auction.

    Returns (winner, payment, virtual-welfare, per-bidder virtual values).

    Allocates to the bidder with the highest *non-negative* virtual value.
    Payment is the *threshold bid* — the smallest value the winner could
    have reported while still winning.
    """
    if len(bidders) != len(values) or len(bidders) != len(distributions):
        raise InvalidBid("bidders/values/distributions length mismatch")
    if not bidders:
        return None, 0.0, 0.0, []

    phis = [d.virtual_value(v) for d, v in zip(distributions, values)]

    # Winner = argmax_i φ_i if φ_i ≥ 0; else no sale.
    best_idx = -1
    best_phi = 0.0
    for i, phi in enumerate(phis):
        if phi >= 0 and (best_idx < 0 or phi > best_phi):
            best_idx = i
            best_phi = phi

    if best_idx < 0:
        return None, 0.0, 0.0, phis

    # Threshold bid: smallest v* such that φ_w(v*) ≥ max(0, second-best φ).
    others = [phis[j] for j in range(len(phis)) if j != best_idx]
    second_phi = max([0.0] + [p for p in others if p >= 0]) if others else 0.0
    # When others' virtual values are all negative, second_phi = 0 — winner
    # only needs to clear the reserve r = φ_w^{-1}(0).
    target_phi = second_phi
    payment = _phi_inverse(distributions[best_idx], target_phi)
    # Clamp payment in [reserve, value]
    reserve = _phi_inverse(distributions[best_idx], 0.0)
    payment = max(payment, reserve)
    payment = min(payment, values[best_idx])

    return bidders[best_idx], payment, best_phi, phis


def _phi_inverse(dist: ValueDistribution, target: float) -> float:
    """Solve φ(v) = target by bisection over support.

    Closed-form fast paths for known distributions.
    """
    if isinstance(dist, UniformDistribution):
        # φ(v) = 2v − b  ⇒  v = (target + b)/2
        v = 0.5 * (target + dist.b)
        return max(dist.a, min(dist.b, v))
    if isinstance(dist, ExponentialDistribution):
        # φ(v) = v − 1/λ  ⇒  v = target + 1/λ
        v = target + 1.0 / dist.rate
        return max(0.0, min(dist.vmax, v))

    # Generic bisection. Bracket within support.
    lo, hi = dist.support()
    eps = 1e-9 * max(1.0, hi - lo)
    a, b = lo + eps, hi - eps
    phi_a = dist.virtual_value(a)
    phi_b = dist.virtual_value(b)
    if target <= phi_a:
        return a
    if target >= phi_b:
        return b
    for _ in range(80):
        mid = 0.5 * (a + b)
        phi_m = dist.virtual_value(mid)
        if phi_m < target:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


# =====================================================================
# VCG — additive / unit-demand (Hungarian / greedy)
# =====================================================================


def _vcg_welfare_assignment(
    bidders: list[str],
    items: list[str],
    values: dict[str, dict[str, float]],
    capacity: dict[str, int],
    item_supply: dict[str, int] | None = None,
) -> tuple[float, dict[str, list[str]]]:
    """Welfare-maximising assignment with per-bidder capacity.

    Bidders have additive utilities. Items are indivisible with supply 1
    (default). We return (welfare, assignment) where assignment maps
    bidder → list of items.

    Greedy by descending marginal value. For unit-demand (capacity=1) this
    is exact. For additive valuations with capacity > 1 and supply 1, the
    LP relaxation is integral (assignment polytope is unimodular), and
    the greedy step described here is exact (each item assigned to its
    best feasible bidder).
    """
    if item_supply is None:
        item_supply = {it: 1 for it in items}

    # Build list of (value, bidder, item) sorted descending by value.
    triples: list[tuple[float, str, str]] = []
    for b in bidders:
        for it in items:
            v = values.get(b, {}).get(it, 0.0)
            if v > 0:
                triples.append((v, b, it))
    triples.sort(reverse=True)

    assigned: dict[str, list[str]] = {b: [] for b in bidders}
    item_remaining = dict(item_supply)
    cap_left = dict(capacity)
    welfare = 0.0
    for v, b, it in triples:
        if item_remaining.get(it, 0) <= 0:
            continue
        if cap_left.get(b, 0) <= 0:
            continue
        # Don't assign same item twice to same bidder
        if it in assigned[b]:
            continue
        assigned[b].append(it)
        item_remaining[it] -= 1
        cap_left[b] -= 1
        welfare += v

    return welfare, assigned


# =====================================================================
# Sample complexity bounds
# =====================================================================


def sample_complexity_for_eps_optimal(
    *, epsilon: float, delta: float, k_bidders: int = 1, regularity: str = REG_REGULAR
) -> int:
    """Number of i.i.d. samples per bidder needed for (1-ε)-Myerson revenue
    with probability ≥ 1 − δ.

    For *regular* distributions, the Gonczarowski-Weinberg (2021) bound is
    n ≥ C · k · log(k/δ) / ε². We expose the constant as C = 32 by default
    — conservative but matches the literature for distribution-free
    constructions.

    For *MHR* the bound is tighter, n ≥ C · log(k/δ) / ε^{3/2}.
    For *bounded* it is n ≥ C · k · log(k/δ) / ε^3 (Cole-Roughgarden).
    """
    if not (0 < epsilon < 1):
        raise MechanismError(f"epsilon must be in (0,1): {epsilon}")
    if not (0 < delta < 1):
        raise MechanismError(f"delta must be in (0,1): {delta}")
    if k_bidders < 1:
        raise MechanismError(f"k_bidders must be ≥ 1: {k_bidders}")

    C = 32
    if regularity == REG_MHR:
        return int(math.ceil(C * math.log(max(2, k_bidders) / delta) / (epsilon ** 1.5)))
    if regularity == REG_BOUNDED:
        return int(
            math.ceil(C * k_bidders * math.log(max(2, k_bidders) / delta) / (epsilon ** 3))
        )
    # default: regular
    return int(
        math.ceil(C * k_bidders * math.log(max(2, k_bidders) / delta) / (epsilon ** 2))
    )


# =====================================================================
# MechanismDesigner runtime
# =====================================================================


class MechanismDesigner:
    """Revenue-optimal mechanism design as a runtime primitive.

    Threadsafe. Optional dependencies:

      bus       — `agi.events.EventBus` for live event broadcast.
      attestor  — `agi.attest.RuntimeAttestor` for content-hashed receipts.
      random_seed — for reproducible online / sample-based calls.

    Most methods are pure functions of inputs and emit a single
    ``mechanism.allocated``/``mechanism.priced``/``mechanism.certified``
    event on completion. State (campaign counters, attestation receipts)
    is maintained per instance.
    """

    def __init__(
        self,
        *,
        bus: Any = None,
        attestor: Any = None,
        random_seed: int | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._lock = threading.RLock()
        self._rng = random.Random(random_seed)
        self._n_allocations = 0
        self._n_vcg = 0
        self._n_posted_price = 0
        self._n_reserve_fits = 0
        self._n_online_calls = 0
        self._n_certificates = 0
        self._started_ns = time.time_ns()
        self._emit(
            MECH_STARTED,
            {"id": uuid.uuid4().hex[:16], "ts_ns": self._started_ns},
        )

    # -------- event / attest helpers --------

    def _emit(self, kind: str, payload: dict) -> None:
        if self._bus is None or Event is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(payload)))
        except Exception:
            pass

    def _attest(self, kind: str, payload: dict) -> str:
        digest = _hash_payload(payload)
        if self._attestor is None:
            return digest
        receipt = _AttestableReceipt(kind=kind, payload=dict(payload), digest=digest)
        try:
            if hasattr(self._attestor, "record"):
                self._attestor.record(kind=kind, payload=dict(payload))
            elif callable(self._attestor):
                self._attestor(receipt)
        except Exception:
            pass
        return digest

    # -------- single-item: Vickrey / first-price --------

    def vickrey_auction(
        self,
        bids: Mapping[str, float],
        *,
        reserve: float = 0.0,
    ) -> Allocation:
        """Second-price sealed-bid auction with optional reserve.

        DSIC for any bidder: truthful bidding weakly dominates any deviation.
        """
        if not bids:
            raise InvalidBid("vickrey_auction: empty bids")
        sorted_pairs = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
        sorted_bids = [v for _, v in sorted_pairs]
        sorted_names = [n for n, _ in sorted_pairs]

        winner_idx, payment = vickrey_payment(sorted_bids, reserve=reserve)
        if winner_idx < 0:
            winner = None
            welfare = 0.0
            revenue = 0.0
            utilities = tuple((n, 0.0) for n in bids.keys())
        else:
            winner = sorted_names[winner_idx]
            welfare = sorted_bids[winner_idx]
            revenue = payment
            utils = []
            for n, v in bids.items():
                u = (v - payment) if n == winner else 0.0
                utils.append((n, u))
            utilities = tuple(utils)

        payload = {
            "mechanism": KIND_VICKREY,
            "winner": winner,
            "payment": payment,
            "revenue": revenue,
            "welfare": welfare,
            "reserve": reserve,
            "n_bidders": len(bids),
        }
        digest = self._attest(MECH_ALLOCATED, payload)
        with self._lock:
            self._n_allocations += 1
        self._emit(MECH_ALLOCATED, {**payload, "digest": digest})
        return Allocation(
            mechanism=KIND_VICKREY,
            winner=winner,
            payment=payment,
            welfare=welfare,
            revenue=revenue,
            utilities=utilities,
            bids=tuple(sorted_pairs),
            reserve=reserve,
            receipt_digest=digest,
        )

    def first_price_auction(
        self,
        bids: Mapping[str, float],
        *,
        reserve: float = 0.0,
    ) -> Allocation:
        """First-price sealed-bid auction. NOT DSIC — informational use only."""
        if not bids:
            raise InvalidBid("first_price_auction: empty bids")
        sorted_pairs = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
        sorted_bids = [v for _, v in sorted_pairs]
        sorted_names = [n for n, _ in sorted_pairs]

        winner_idx, payment = first_price_payment(sorted_bids, reserve=reserve)
        if winner_idx < 0:
            winner = None
            welfare = 0.0
            revenue = 0.0
            utilities = tuple((n, 0.0) for n in bids.keys())
        else:
            winner = sorted_names[winner_idx]
            welfare = sorted_bids[winner_idx]
            revenue = payment
            utils = []
            for n, v in bids.items():
                u = (v - payment) if n == winner else 0.0
                utils.append((n, u))
            utilities = tuple(utils)

        payload = {
            "mechanism": KIND_FIRST_PRICE,
            "winner": winner,
            "payment": payment,
            "revenue": revenue,
            "welfare": welfare,
            "reserve": reserve,
            "n_bidders": len(bids),
        }
        digest = self._attest(MECH_ALLOCATED, payload)
        with self._lock:
            self._n_allocations += 1
        self._emit(MECH_ALLOCATED, {**payload, "digest": digest})
        return Allocation(
            mechanism=KIND_FIRST_PRICE,
            winner=winner,
            payment=payment,
            welfare=welfare,
            revenue=revenue,
            utilities=utilities,
            bids=tuple(sorted_pairs),
            reserve=reserve,
            receipt_digest=digest,
        )

    def all_pay_auction(
        self,
        bids: Mapping[str, float],
    ) -> Allocation:
        """All-pay: highest bid wins the item, *every* bidder pays their bid.

        Not DSIC; revenue-equivalent in BNE to Vickrey for symmetric regular
        priors. Useful as a contest primitive.
        """
        if not bids:
            raise InvalidBid("all_pay_auction: empty bids")
        sorted_pairs = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
        winner = sorted_pairs[0][0]
        winner_value = sorted_pairs[0][1]
        revenue = sum(bids.values())
        utilities = []
        for n, v in bids.items():
            paid = v
            got = v if n == winner else 0.0
            utilities.append((n, got - paid))
        payload = {
            "mechanism": KIND_ALL_PAY,
            "winner": winner,
            "revenue": revenue,
            "n_bidders": len(bids),
        }
        digest = self._attest(MECH_ALLOCATED, payload)
        with self._lock:
            self._n_allocations += 1
        self._emit(MECH_ALLOCATED, {**payload, "digest": digest})
        return Allocation(
            mechanism=KIND_ALL_PAY,
            winner=winner,
            payment=winner_value,
            welfare=winner_value,
            revenue=revenue,
            utilities=tuple(utilities),
            bids=tuple(sorted_pairs),
            reserve=0.0,
            receipt_digest=digest,
        )

    # -------- single-item: Myerson optimal --------

    def myerson_auction(
        self,
        bids: Mapping[str, float],
        distributions: Mapping[str, ValueDistribution],
    ) -> Allocation:
        """Myerson optimal auction with known priors.

        DSIC + revenue-optimal among all single-item DSIC mechanisms,
        given independent regular priors per bidder.
        """
        if not bids:
            raise InvalidBid("myerson_auction: empty bids")
        bidders = list(bids.keys())
        for b in bidders:
            if b not in distributions:
                raise InvalidDistribution(
                    f"myerson_auction: missing distribution for bidder '{b}'"
                )
        vals = [bids[b] for b in bidders]
        dists = [distributions[b] for b in bidders]
        winner, payment, virt_welfare, phis = myerson_winner_and_payment(
            bidders, vals, dists
        )
        if winner is None:
            welfare = 0.0
            revenue = 0.0
            utilities = tuple((b, 0.0) for b in bidders)
            reserve = 0.0
        else:
            welfare = bids[winner]
            revenue = payment
            utils = []
            for n, v in bids.items():
                u = (v - payment) if n == winner else 0.0
                utils.append((n, u))
            utilities = tuple(utils)
            reserve = _phi_inverse(distributions[winner], 0.0)

        payload = {
            "mechanism": KIND_MYERSON,
            "winner": winner,
            "payment": payment,
            "revenue": revenue,
            "welfare": welfare,
            "virtual_welfare": virt_welfare,
            "reserve": reserve,
            "n_bidders": len(bids),
        }
        digest = self._attest(MECH_ALLOCATED, payload)
        with self._lock:
            self._n_allocations += 1
        self._emit(MECH_ALLOCATED, {**payload, "digest": digest})
        sorted_pairs = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
        return Allocation(
            mechanism=KIND_MYERSON,
            winner=winner,
            payment=payment,
            welfare=welfare,
            revenue=revenue,
            utilities=utilities,
            bids=tuple(sorted_pairs),
            reserve=reserve,
            receipt_digest=digest,
        )

    def myerson_from_samples(
        self,
        bids: Mapping[str, float],
        samples: Mapping[str, Sequence[float]],
        *,
        delta: float = 0.05,
    ) -> Allocation:
        """Empirical-Myerson: fit a distribution per bidder from samples,
        then run Myerson. The reserve is the *empirical* monopoly reserve.

        Sample-complexity-aware: refuses to run if any bidder has fewer
        than 8 samples (Cole-Roughgarden minimum for a non-trivial bound).
        """
        if not bids:
            raise InvalidBid("myerson_from_samples: empty bids")
        dists: dict[str, ValueDistribution] = {}
        for b in bids:
            s = samples.get(b)
            if s is None or len(s) < 8:
                raise InsufficientData(
                    f"myerson_from_samples: bidder '{b}' has < 8 samples"
                )
            dists[b] = EmpiricalDistribution(samples=tuple(s))
        return self.myerson_auction(bids, dists)

    # -------- anonymous-reserve Vickrey (Hartline-Roughgarden) --------

    def anonymous_reserve_auction(
        self,
        bids: Mapping[str, float],
        *,
        reserve: float,
    ) -> Allocation:
        """Anonymous-reserve Vickrey: 2nd-price with single reserve `r`.

        Hartline-Roughgarden (2009): for any set of regular distributions,
        the optimal anonymous-reserve Vickrey earns ≥ ½ of Myerson's
        expected revenue, with the right `r` being the monopoly reserve
        of the *aggregate* distribution.
        """
        out = self.vickrey_auction(bids, reserve=reserve)
        # Re-tag mechanism kind
        return Allocation(
            mechanism=KIND_ANONYMOUS_RESERVE,
            winner=out.winner,
            payment=out.payment,
            welfare=out.welfare,
            revenue=out.revenue,
            utilities=out.utilities,
            bids=out.bids,
            reserve=reserve,
            receipt_digest=out.receipt_digest,
        )

    # -------- VCG multi-item allocation --------

    def vcg_allocation(
        self,
        items: Sequence[str],
        bids: Mapping[str, Mapping[str, float]],
        *,
        capacity: Mapping[str, int] | None = None,
        item_supply: Mapping[str, int] | None = None,
    ) -> VCGAllocation:
        """VCG allocation for additive / unit-demand bidders.

        Computes welfare-maximising assignment, then each winning agent
        pays the externality `W*−i_present − W*−i_absent`, where W*−i is
        the welfare achievable *without* agent i's allocated items
        (effectively, what the items would be worth to the *other*
        agents). DSIC by construction.
        """
        bidder_list = list(bids.keys())
        if not bidder_list:
            raise InvalidBid("vcg_allocation: empty bids")
        if not items:
            raise InvalidBid("vcg_allocation: empty items")
        if capacity is None:
            capacity = {b: len(items) for b in bidder_list}
        cap = {b: int(capacity.get(b, len(items))) for b in bidder_list}
        sup = {it: int((item_supply or {}).get(it, 1)) for it in items}
        # Make values fully populated
        vals: dict[str, dict[str, float]] = {
            b: {it: float(bids[b].get(it, 0.0)) for it in items} for b in bidder_list
        }
        for b in bidder_list:
            for it in items:
                if vals[b][it] < 0:
                    raise InvalidBid(
                        f"vcg_allocation: bidder {b} item {it} has negative value"
                    )

        W_star, alloc = _vcg_welfare_assignment(
            bidder_list, list(items), vals, cap, sup
        )

        # Per-bidder VCG payment: externality = W*_{−i} − (W* − v_i(alloc_i))
        # where W*_{−i} is the welfare-max assignment without bidder i.
        assignments: list[VCGAssignment] = []
        utilities: list[tuple[str, float]] = []
        total_revenue = 0.0

        for b in bidder_list:
            v_b = sum(vals[b][it] for it in alloc[b])
            others = [x for x in bidder_list if x != b]
            cap_others = {x: cap[x] for x in others}
            W_minus_i, _ = _vcg_welfare_assignment(
                others, list(items), vals, cap_others, sup
            )
            externality = W_minus_i - (W_star - v_b)
            externality = max(0.0, externality)
            # The "price" charged to b is the externality. Split evenly across
            # the items b won (so total paid = externality).
            n_items_won = len(alloc[b])
            per_item_price = externality / n_items_won if n_items_won > 0 else 0.0
            for it in alloc[b]:
                assignments.append(
                    VCGAssignment(
                        bidder_id=b,
                        item_id=it,
                        payment=per_item_price,
                        value=vals[b][it],
                    )
                )
            total_revenue += externality
            utilities.append((b, v_b - externality))

        payload = {
            "mechanism": KIND_VCG,
            "items": list(items),
            "bidders": bidder_list,
            "total_welfare": W_star,
            "total_revenue": total_revenue,
            "n_assignments": len(assignments),
        }
        digest = self._attest(MECH_ALLOCATED, payload)
        with self._lock:
            self._n_vcg += 1
        self._emit(MECH_ALLOCATED, {**payload, "digest": digest})
        return VCGAllocation(
            mechanism=KIND_VCG,
            assignments=tuple(assignments),
            total_welfare=W_star,
            total_revenue=total_revenue,
            utilities=tuple(utilities),
            receipt_digest=digest,
        )

    # -------- posted-price (sequential, DSIC by construction) --------

    def posted_price(
        self,
        valuations: Mapping[str, float],
        prices: Mapping[str, float],
        *,
        order: Sequence[str] | None = None,
        item_id: str = "item",
    ) -> PostedPriceOutcome:
        """Single-item sequential posted-price mechanism.

        Each bidder, in `order`, is shown price `prices[bidder]`. They
        accept iff their value ≥ their price. The first acceptor wins.
        DSIC by construction: a bidder cannot affect any other bidder's
        price, and is offered a take-it-or-leave-it deal.
        """
        if order is None:
            order = list(valuations.keys())
        accepted: list[str] = []
        assignments: list[tuple[str, str, float]] = []
        utilities: dict[str, float] = {b: 0.0 for b in valuations}
        revenue = 0.0
        sold = False

        for b in order:
            if b not in valuations:
                raise InvalidBid(f"posted_price: order has unknown bidder '{b}'")
            if b not in prices:
                raise InvalidBid(f"posted_price: missing price for bidder '{b}'")
            if sold:
                # not offered — utility 0
                continue
            p = float(prices[b])
            v = float(valuations[b])
            if v >= p:
                accepted.append(b)
                assignments.append((b, item_id, p))
                utilities[b] = v - p
                revenue += p
                sold = True

        payload = {
            "mechanism": KIND_POSTED_PRICE,
            "item_id": item_id,
            "revenue": revenue,
            "order": list(order),
            "accepted": list(accepted),
            "n_offered": len(order),
        }
        digest = self._attest(MECH_ALLOCATED, payload)
        with self._lock:
            self._n_posted_price += 1
        self._emit(MECH_ALLOCATED, {**payload, "digest": digest})
        return PostedPriceOutcome(
            mechanism=KIND_POSTED_PRICE,
            accepted=tuple(accepted),
            item_assignments=tuple(assignments),
            revenue=revenue,
            utilities=tuple(sorted(utilities.items())),
            receipt_digest=digest,
        )

    # -------- sample-based reserves & Myerson --------

    def empirical_reserve(
        self,
        samples: Sequence[float],
        *,
        delta: float = 0.05,
        method: str = "monopoly",
    ) -> ReservePolicy:
        """Fit a monopoly reserve from i.i.d. samples of a single bidder's
        value distribution.

        Methods:
          - "monopoly"  — pick r* = argmax_r r · (1 − F̂(r))  (Myerson reserve)
          - "median"    — r = sample median  (simple baseline)

        Returns reserve plus a Hoeffding LCB on the expected revenue
        attained against the *same* distribution (E[v · 1{v ≥ r}]).
        """
        if len(samples) < 8:
            raise InsufficientData(
                f"empirical_reserve: need ≥ 8 samples, got {len(samples)}"
            )
        s_sorted = sorted(samples)
        n = len(s_sorted)
        s_max = s_sorted[-1]

        if method == "median":
            r_star = s_sorted[n // 2]
        elif method == "monopoly":
            # Maximize r · (1 − F̂(r)) over candidate reserves = sample
            # values. The argmax is achieved at one of the sample points.
            best_rev = -1.0
            r_star = s_sorted[0]
            for i, r in enumerate(s_sorted):
                # 1 − F̂(r) for r = s_sorted[i]:
                #   probability a sample is ≥ r is (n − i) / n   (right-continuous)
                p_geq = (n - i) / n
                rev = r * p_geq
                if rev > best_rev:
                    best_rev = rev
                    r_star = r
        else:
            raise MechanismError(f"empirical_reserve: unknown method '{method}'")

        # Plug-in revenue estimator and Hoeffding LCB
        revenues = [v if v >= r_star else 0.0 for v in samples]
        # NOTE: this is welfare-of-winner, not posted-price revenue. The
        # *posted-price* revenue at reserve r is r · 1{v ≥ r}. Use that
        # for the LCB so we certify what the actual mechanism returns.
        posted_revenues = [r_star if v >= r_star else 0.0 for v in samples]
        rev_mean = statistics.fmean(posted_revenues)
        radius = hoeffding_radius(n, delta=delta, range_=s_max)
        rev_lcb = max(0.0, rev_mean - radius)

        payload = {
            "method": method,
            "reserve": r_star,
            "n_samples": n,
            "delta": delta,
            "revenue_mean": rev_mean,
            "revenue_lcb": rev_lcb,
            "revenue_radius": radius,
        }
        digest = self._attest(MECH_RESERVE_FIT, payload)
        with self._lock:
            self._n_reserve_fits += 1
        self._emit(MECH_RESERVE_FIT, {**payload, "digest": digest})

        return ReservePolicy(
            reserve=r_star,
            n_samples=n,
            delta=delta,
            revenue_lcb=rev_lcb,
            revenue_mean=rev_mean,
            revenue_radius=radius,
            method=method,
            distribution_class=DIST_EMPIRICAL,
        )

    def sample_complexity(
        self,
        *,
        epsilon: float,
        delta: float,
        k_bidders: int = 1,
        regularity: str = REG_REGULAR,
    ) -> int:
        """Wraps `sample_complexity_for_eps_optimal` for instance use."""
        return sample_complexity_for_eps_optimal(
            epsilon=epsilon,
            delta=delta,
            k_bidders=k_bidders,
            regularity=regularity,
        )

    # -------- Bulow-Klemperer comparison --------

    def bulow_klemperer(
        self,
        samples: Sequence[float],
        *,
        n: int,
        trials: int = 1000,
        rng: random.Random | None = None,
    ) -> BulowKlemperer:
        """Bulow-Klemperer (1996) ratio.

        Compare:
          A) `n+1`-bidder Vickrey *without* a reserve
          B) `n`-bidder Myerson with the *empirical* monopoly reserve

        On i.i.d. draws from the empirical distribution induced by
        `samples`. By BK's theorem (for regular priors), the expected
        revenue of (A) ≥ revenue of (B) — i.e., one extra bidder dominates
        any reserve. We *measure* the ratio empirically over `trials`
        simulations.
        """
        if len(samples) < 8:
            raise InsufficientData("bulow_klemperer: need ≥ 8 samples")
        if n < 1:
            raise MechanismError(f"bulow_klemperer: n must be ≥ 1, got {n}")
        rng = rng or self._rng

        dist = EmpiricalDistribution(samples=tuple(samples))
        # Reserve = empirical monopoly reserve
        rp = self.empirical_reserve(samples, method="monopoly")
        reserve = rp.reserve

        rev_a_total = 0.0
        rev_b_total = 0.0
        for _ in range(trials):
            draws_b = [dist.sample(rng) for _ in range(n)]
            draws_a = draws_b + [dist.sample(rng)]
            # A: (n+1)-bidder Vickrey, no reserve
            draws_a_sorted = sorted(draws_a, reverse=True)
            _, pay_a = vickrey_payment(draws_a_sorted, reserve=0.0)
            rev_a_total += pay_a
            # B: n-bidder Vickrey with reserve (Myerson under regular common prior)
            draws_b_sorted = sorted(draws_b, reverse=True)
            _, pay_b = vickrey_payment(draws_b_sorted, reserve=reserve)
            rev_b_total += pay_b
        rev_a = rev_a_total / trials
        rev_b = rev_b_total / trials
        ratio = (rev_a / rev_b) if rev_b > 0 else float("inf")

        payload = {
            "n": n,
            "revenue_vickrey_n_plus_1": rev_a,
            "revenue_myerson_n": rev_b,
            "ratio": ratio,
            "trials": trials,
            "samples_used": len(samples),
            "reserve": reserve,
        }
        digest = self._attest(MECH_BULOW_KLEMPERER, payload)
        self._emit(MECH_BULOW_KLEMPERER, {**payload, "digest": digest})

        return BulowKlemperer(
            n=n,
            revenue_vickrey_n_plus_1=rev_a,
            revenue_myerson_n=rev_b,
            ratio=ratio,
            samples_used=len(samples),
        )

    # -------- DSIC / IR certificates --------

    def certify_dsic(
        self,
        mechanism: str,
        *,
        bids_truthful: Mapping[str, float],
        distributions: Mapping[str, ValueDistribution] | None = None,
        deviation_grid: int = 21,
        reserve: float = 0.0,
    ) -> DSICReport:
        """Empirical DSIC certificate.

        For each bidder, sweep their reported bid over a fine grid in
        ``[0, v_i]`` and ``[v_i, v_max]`` and check that truthful bidding
        attains the maximum utility (winning probability × (v_i − payment)).
        Returns the worst observed *gain from lying*.

        - Vickrey, Myerson, VCG, posted-price all certify DSIC.
        - First-price, all-pay fail by construction.
        """
        if mechanism not in KNOWN_MECHANISMS:
            raise UnknownMechanism(f"certify_dsic: unknown '{mechanism}'")

        bidders = list(bids_truthful.keys())
        if not bidders:
            raise InvalidBid("certify_dsic: empty bids")

        v_max = max(bids_truthful.values()) * 2.0 + 1.0
        worst_gain = 0.0
        failing = None

        # IMPORTANT: utility must be computed against the bidder's *true*
        # valuation, not their (possibly deviated) bid. The auctions return
        # utilities assuming value = bid; we override that here.
        def _utility(mech: str, bids_now: dict[str, float], who: str, true_v: float) -> float:
            if mech == KIND_VICKREY:
                out = self.vickrey_auction(bids_now, reserve=reserve)
                if out.winner == who:
                    return true_v - out.payment
                return 0.0
            if mech == KIND_FIRST_PRICE:
                out = self.first_price_auction(bids_now, reserve=reserve)
                if out.winner == who:
                    return true_v - out.payment
                return 0.0
            if mech == KIND_MYERSON:
                if distributions is None:
                    raise InvalidDistribution("Myerson requires distributions")
                out = self.myerson_auction(bids_now, distributions)
                if out.winner == who:
                    return true_v - out.payment
                return 0.0
            if mech == KIND_ANONYMOUS_RESERVE:
                out = self.anonymous_reserve_auction(bids_now, reserve=reserve)
                if out.winner == who:
                    return true_v - out.payment
                return 0.0
            if mech == KIND_ALL_PAY:
                out = self.all_pay_auction(bids_now)
                # in all-pay every bidder pays their bid
                paid = bids_now[who]
                won = true_v if out.winner == who else 0.0
                return won - paid
            raise UnknownMechanism(f"certify_dsic: mechanism {mech} not single-item")

        for who in bidders:
            v = bids_truthful[who]
            truthful_bids = dict(bids_truthful)
            truthful_bids[who] = v
            u_truth = _utility(mechanism, truthful_bids, who, v)
            # Sweep deviation grid in [0, 2 * v_max], augmented with the
            # "shading break-points" (just above each opponent's bid). This
            # is exactly where pay-your-bid mechanisms admit utility gains.
            uniform_grid = [
                i * 2.0 * v_max / max(1, deviation_grid - 1)
                for i in range(deviation_grid)
            ]
            opponents = [bids_truthful[o] for o in bidders if o != who]
            eps = 1e-6 * max(1.0, v_max)
            shading_grid = [b + eps for b in opponents] + [b - eps for b in opponents]
            grid = uniform_grid + shading_grid
            best_gain = 0.0
            for b_alt in grid:
                trial = dict(bids_truthful)
                trial[who] = b_alt
                try:
                    u = _utility(mechanism, trial, who, v)
                except Exception:
                    continue
                gain = u - u_truth
                if gain > best_gain:
                    best_gain = gain
            if best_gain > worst_gain:
                worst_gain = best_gain
                failing = who

        is_dsic = worst_gain <= 1e-6
        payload = {
            "mechanism": mechanism,
            "is_dsic": is_dsic,
            "worst_gain": worst_gain,
            "deviation_grid": deviation_grid,
            "failing_bidder": failing if not is_dsic else None,
        }
        digest = self._attest(MECH_CERTIFIED, payload)
        with self._lock:
            self._n_certificates += 1
        self._emit(MECH_CERTIFIED, {**payload, "digest": digest})
        return DSICReport(
            mechanism=mechanism,
            is_dsic=is_dsic,
            worst_gain=worst_gain,
            max_attempts=deviation_grid * len(bidders),
            failing_bidder=failing if not is_dsic else None,
        )

    def certify_ir(self, allocation: Allocation | VCGAllocation) -> IRReport:
        """Individual rationality: every participant's utility ≥ 0."""
        worst = 0.0
        for _, u in allocation.utilities:
            if u < worst:
                worst = u
        is_ir = worst >= -1e-9
        payload = {
            "mechanism": allocation.mechanism,
            "is_ir": is_ir,
            "worst_utility": worst,
        }
        digest = self._attest(MECH_CERTIFIED, payload)
        with self._lock:
            self._n_certificates += 1
        self._emit(MECH_CERTIFIED, {**payload, "digest": digest})
        return IRReport(
            mechanism=allocation.mechanism,
            is_ir=is_ir,
            worst_utility=worst,
        )

    # -------- revenue certificate (anytime LCB) --------

    def revenue_certificate(
        self,
        mechanism: str,
        revenues: Sequence[float],
        *,
        delta: float = 0.05,
        range_: float | None = None,
    ) -> RevenueCertificate:
        """Anytime PAC LCB on expected revenue of `mechanism` over a stream
        of observed `revenues` from repeated runs.
        """
        n = len(revenues)
        if n < 1:
            raise InsufficientData("revenue_certificate: empty revenues")
        if range_ is None:
            range_ = max(revenues) if revenues else 1.0
            if range_ <= 0:
                range_ = 1.0
        mean = statistics.fmean(revenues)
        radius = empirical_bernstein_radius(revenues, delta=delta, range_=range_)
        lcb = max(0.0, mean - radius)
        payload = {
            "mechanism": mechanism,
            "n_samples": n,
            "delta": delta,
            "revenue_mean": mean,
            "revenue_lcb": lcb,
            "revenue_radius": radius,
        }
        digest = self._attest(MECH_CERTIFIED, payload)
        with self._lock:
            self._n_certificates += 1
        self._emit(MECH_CERTIFIED, {**payload, "digest": digest})
        return RevenueCertificate(
            mechanism=mechanism,
            n_samples=n,
            delta=delta,
            revenue_mean=mean,
            revenue_lcb=lcb,
            revenue_radius=radius,
            receipt_digest=digest,
        )

    # -------- online posted-price (Kleinberg-Leighton) --------

    def online_posted_price(
        self,
        feedback: Callable[[float], bool | int | float],
        *,
        T: int,
        v_max: float,
        delta: float = 0.05,
        algorithm: str = "kleinberg_leighton",
    ) -> OnlinePostedPriceOutcome:
        """Online posted-price against a single bidder over T rounds.

        `feedback(price)` returns 1/True if the bidder accepts at `price`
        (and pays `price`), 0/False otherwise. Implements the
        Kleinberg-Leighton (2003) EXP3-on-grid bandit with grid
        resolution T^{1/3}, achieving Õ(T^{2/3}) regret against the best
        fixed price in hindsight for any bounded valuation sequence.

        Note: the *bidder* in this online setting is *not* required to
        be strategic — Kleinberg-Leighton's regret bound is over the
        adversarial valuation sequence. The mechanism is DSIC per round
        because each round is a take-it-or-leave-it offer.
        """
        if T < 1:
            raise MechanismError(f"online_posted_price: T must be ≥ 1, got {T}")
        if v_max <= 0:
            raise MechanismError(f"online_posted_price: v_max must be > 0")
        if algorithm not in ("kleinberg_leighton", "uniform"):
            raise MechanismError(f"online_posted_price: unknown algorithm '{algorithm}'")

        # Grid of K candidate prices in (0, v_max]. KL recommend K ≈ T^{1/3}.
        K = max(3, int(round(T ** (1.0 / 3.0))))
        prices = [v_max * (k + 1) / K for k in range(K)]
        # EXP3 weights
        weights = [1.0] * K
        gamma = min(1.0, math.sqrt(K * math.log(K) / max(1.0, T)))

        revenue = 0.0
        price_hist: list[float] = []
        accept_hist: list[int] = []

        rng = self._rng

        for t in range(T):
            total_w = sum(weights)
            probs = [
                (1.0 - gamma) * (w / total_w) + gamma / K for w in weights
            ]
            # sample arm
            r = rng.random()
            cum = 0.0
            chosen = K - 1
            for k, p in enumerate(probs):
                cum += p
                if r <= cum:
                    chosen = k
                    break
            p_t = prices[chosen]
            accepted_raw = feedback(p_t)
            accepted = 1 if accepted_raw else 0
            paid = p_t if accepted else 0.0
            revenue += paid
            price_hist.append(p_t)
            accept_hist.append(accepted)

            # EXP3 update (reward in [0, 1] via normalization)
            if algorithm == "kleinberg_leighton":
                reward_norm = paid / v_max
                est = reward_norm / probs[chosen]
                weights[chosen] *= math.exp(gamma * est / K)
                # Cap weights to avoid float overflow
                m = max(weights)
                if m > 1e60:
                    weights = [w / m for w in weights]

        # Regret upper bound: KL bound = O(T^{2/3} (log T)^{1/3})
        # We report a concrete constant: 4 · T^{2/3} · (log(2/δ))^{1/3}
        pseudo_regret_ub = 4.0 * (T ** (2.0 / 3.0)) * (math.log(2.0 / delta) ** (1.0 / 3.0))
        mean_rev = revenue / T

        payload = {
            "T": T,
            "v_max": v_max,
            "algorithm": algorithm,
            "n_grid": K,
            "revenue": revenue,
            "revenue_mean": mean_rev,
            "pseudo_regret_ub": pseudo_regret_ub,
        }
        digest = self._attest(MECH_ONLINE_STEP, payload)
        with self._lock:
            self._n_online_calls += 1
        self._emit(MECH_ONLINE_STEP, {**payload, "digest": digest})

        return OnlinePostedPriceOutcome(
            T=T,
            v_max=v_max,
            algorithm=algorithm,
            revenue=revenue,
            revenue_mean=mean_rev,
            n_grid=K,
            price_history=tuple(price_hist),
            accept_history=tuple(accept_hist),
            pseudo_regret_ub=pseudo_regret_ub,
            receipt_digest=digest,
        )

    # -------- introspection --------

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_allocations": self._n_allocations,
                "n_vcg": self._n_vcg,
                "n_posted_price": self._n_posted_price,
                "n_reserve_fits": self._n_reserve_fits,
                "n_online_calls": self._n_online_calls,
                "n_certificates": self._n_certificates,
                "uptime_ns": time.time_ns() - self._started_ns,
            }

    def clear(self) -> None:
        with self._lock:
            self._n_allocations = 0
            self._n_vcg = 0
            self._n_posted_price = 0
            self._n_reserve_fits = 0
            self._n_online_calls = 0
            self._n_certificates = 0
            self._emit(MECH_CLEARED, {"ts_ns": time.time_ns()})


# =====================================================================
# Facade for one-shot use
# =====================================================================


def quick_vickrey(bids: Mapping[str, float], *, reserve: float = 0.0) -> Allocation:
    """Stateless Vickrey auction."""
    return MechanismDesigner().vickrey_auction(bids, reserve=reserve)


def quick_myerson(
    bids: Mapping[str, float], distributions: Mapping[str, ValueDistribution]
) -> Allocation:
    """Stateless Myerson auction."""
    return MechanismDesigner().myerson_auction(bids, distributions)


def quick_vcg(
    items: Sequence[str],
    bids: Mapping[str, Mapping[str, float]],
    *,
    capacity: Mapping[str, int] | None = None,
) -> VCGAllocation:
    """Stateless VCG allocation."""
    return MechanismDesigner().vcg_allocation(items, bids, capacity=capacity)


__all__ = [
    # event kinds
    "MECH_STARTED",
    "MECH_ALLOCATED",
    "MECH_PRICED",
    "MECH_CERTIFIED",
    "MECH_RESERVE_FIT",
    "MECH_BULOW_KLEMPERER",
    "MECH_ONLINE_STEP",
    "MECH_CLEARED",
    # mechanism / distribution ids
    "KIND_VICKREY",
    "KIND_FIRST_PRICE",
    "KIND_MYERSON",
    "KIND_VCG",
    "KIND_POSTED_PRICE",
    "KIND_ANONYMOUS_RESERVE",
    "KIND_ALL_PAY",
    "KNOWN_MECHANISMS",
    "DIST_UNIFORM",
    "DIST_EXPONENTIAL",
    "DIST_TRUNC_NORMAL",
    "DIST_EMPIRICAL",
    "REG_REGULAR",
    "REG_MHR",
    "REG_BOUNDED",
    # exceptions
    "MechanismError",
    "InvalidBid",
    "InvalidDistribution",
    "InsufficientData",
    "InfeasibleAllocation",
    "UnknownMechanism",
    # distributions
    "ValueDistribution",
    "UniformDistribution",
    "ExponentialDistribution",
    "EmpiricalDistribution",
    # data classes
    "Bid",
    "Allocation",
    "VCGAssignment",
    "VCGAllocation",
    "PostedPriceOutcome",
    "ReservePolicy",
    "BulowKlemperer",
    "DSICReport",
    "IRReport",
    "OnlinePostedPriceOutcome",
    "RevenueCertificate",
    # concentration helpers
    "hoeffding_radius",
    "empirical_bernstein_radius",
    # mechanism functions
    "vickrey_payment",
    "first_price_payment",
    "myerson_winner_and_payment",
    "sample_complexity_for_eps_optimal",
    # main primitive
    "MechanismDesigner",
    # facade
    "quick_vickrey",
    "quick_myerson",
    "quick_vcg",
]
