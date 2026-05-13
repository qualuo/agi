"""RiskController — distribution-free, finite-sample risk control for
runtime operating decisions.

Every coordination engine eventually has to set a *threshold*: at what
predicted-cost do we hedge a ticket? at what model-confidence do we
abstain? at what tenant-credit do we throttle? Conventional answers
("pick λ to maximize F1 on a holdout", "use the empirical 95th
percentile") give you a *point estimate*. They give you no guarantee
that, under the next batch of inputs, the resulting risk — refund rate,
error rate, false-abstain rate — will stay below the level you
promised an SLO for. A coordination engine that promises a tenant
"refund rate ≤ 1%" cannot ship a threshold whose realized refund rate
is 1% on average; with finite calibration, it has to ship a threshold
whose realized refund rate is provably ≤ 1% with high probability.

`RiskController` solves that problem. It implements two of the
strongest recent results in distribution-free uncertainty
quantification:

Conformal Risk Control — Angelopoulos-Bates-Fisch-Lei-Schuster 2023
    For losses L_i(λ) that are *monotone non-increasing* in λ (the
    canonical case: as a hedge-trigger threshold rises, the realized
    hedge-and-still-fail rate falls), pick the smallest λ from a
    user-supplied grid such that

        R̂_n(λ) := (Σ_i L_i(λ) + B) / (n + 1)  ≤  α

    where B is a known upper bound on L. This rule guarantees

        E[ L_{n+1}(λ̂) ]  ≤  α

    in *expectation over the next test point* (i.e. the bound is on
    the marginal risk of the deployed system). The only assumption is
    exchangeability between calibration and test. No distributional
    form, no parametric family, no asymptotic regime — finite-sample
    tight.

Learn-Then-Test — Bates-Angelopoulos-Lei-Malik-Jordan 2021
    For losses that are *not* monotone in λ — operating curves with
    multiple optima, calibrated abstention thresholds where the loss
    jumps non-monotonically, risk surfaces controlled by a vector λ
    — CRC's monotone-step trick does not apply. LTT treats each
    candidate λ ∈ Λ as a hypothesis H_λ : R(λ) > α and *tests them
    multiply* with FWER controlled at δ. Any candidate whose p-value
    falls below δ (after correction) is provably safe at the
    requested level. The runtime then deploys the most aggressive
    such candidate.

    Sequential testing — `fixed_sequence` — orders the candidates
    from most-aggressive to most-conservative and tests them in
    order, stopping at the first failure. Under that ordering FWER
    holds *without any multiplicity correction*, so the procedure is
    uniformly more powerful than Bonferroni. It is the right test
    family for any one-dimensional operating threshold.

    Bonferroni — `bonferroni` — tests each candidate at level δ/|Λ|.
    Required when there is no defensible monotone ordering, e.g. for
    a small vector λ ∈ ℝ^k with no natural sweep direction.

The p-values come from one of three concentration bounds, all
implemented exactly:

Hoeffding-Bentkus  (Bates 2021)
    The minimum of the Hoeffding-tail bound and Bentkus' binomial
    inequality. Bentkus is uniformly tighter than Hoeffding once
    n·R is moderate; Hoeffding wins near R≈0. The min of the two is
    the universal default.

WSR — Waudby-Smith-Ramdas 2024
    Predictable-mixture supermartingale bound. For bounded losses
    L ∈ [0, B], the running capital process

        K_t(m) = Π_{i≤t} (1 + λ_i · (L_i − m) / B)

    with predictable λ_i tuned from past variance is a non-negative
    supermartingale under H_0 : E[L] ≤ m. Ville's inequality gives
    a finite-sample anytime-valid p-value. WSR is the *current
    state of the art* for finite-sample mean estimation of bounded
    random variables — narrower than Hoeffding-Bentkus on average,
    especially for small means. Citation: Waudby-Smith & Ramdas,
    JMLR 2024, "Estimating means of bounded random variables by
    betting".

CLT (Wald)
    Plug-in Gaussian upper bound. Asymptotic only — included for
    sanity comparisons and as a deliberately *unsafe* baseline you
    can swap in when you have lots of data and want a tighter (but
    only asymptotically valid) bound.

Where this slots into the coordination engine
---------------------------------------------

    rc = RiskController(bus=bus)

    # Drain receipts into the calibration set. Each point carries the
    # score the threshold acts on plus the realized outcome.
    for ticket in driver.completed():
        rc.record(
            score=ticket.predicted_cost_usd,
            outcome=ticket.actual_cost_usd,
            group=ticket.tenant_id,
        )

    # Pick the most aggressive hedge threshold whose UCB on
    # "hedged-and-still-overran" rate is ≤ 1%, at 90% confidence.
    sel = rc.select(
        candidates=sorted([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]),
        target=0.01,
        delta=0.10,
        loss_fn=lambda p, threshold: 1.0 if (p.score >= threshold and p.outcome > p.score * 1.5) else 0.0,
        method="ltt_hb",
        ordering="from_aggressive",
    )
    if sel is None:
        coordinator.suspend_hedging("calibration insufficient")
    else:
        coordinator.set_hedge_threshold(sel.threshold)

    # Multi-risk: bound *both* refund rate AND abstention rate
    # simultaneously with a shared FWER δ.
    sel = rc.select_multi(
        candidates=...,
        risks=[
            Risk(name="refund_rate", target=0.01, loss_fn=...),
            Risk(name="abstain_rate", target=0.05, loss_fn=...),
        ],
        delta=0.10,
        method="ltt_hb",
    )

Events
    risk.observed   — one (score, outcome) recorded
    risk.fit        — calibration window refit (n samples used)
    risk.selected   — a threshold was selected (target, delta, method, λ̂)
    risk.failed     — no threshold met the constraint
    risk.report     — periodic empirical-risk report

Honest about limits
-------------------

Exchangeability is the assumption. Under heavy regime change the
calibration set ages out of distribution and the UCB bound becomes
silently optimistic. The runtime should pair `RiskController` with
`ConformalPredictor.update_adaptive` (or a drift detector) so it
re-selects when drift is flagged.

Bonferroni is loose by definition — if you pass 1000 candidates
without a natural ordering, expect the selected threshold to be
conservative. Prefer fixed-sequence with a defensibly monotone
ordering whenever the loss is monotone in the candidate parameter.

CRC's E-bound is on *expected* loss over the next test point. It
does *not* give a high-probability bound on the empirical loss over
the next batch. For high-probability batch control, use LTT with HB
or WSR.

All numerics are stdlib-only. Concentration UCBs invert exactly to
machine precision via bisection on closed-form tails.
"""
from __future__ import annotations

import bisect
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

RISK_OBSERVED = "risk.observed"
RISK_FIT = "risk.fit"
RISK_SELECTED = "risk.selected"
RISK_FAILED = "risk.failed"
RISK_REPORT = "risk.report"


# ----- selection methods ---------------------------------------------

METHOD_CRC = "crc"                # Conformal Risk Control (monotone)
METHOD_LTT_HB = "ltt_hb"          # Learn-Then-Test, Hoeffding-Bentkus p-value
METHOD_LTT_WSR = "ltt_wsr"        # Learn-Then-Test, WSR p-value
METHOD_LTT_HOEFFDING = "ltt_hoeffding"  # LTT with pure Hoeffding (looser)
METHOD_LTT_CLT = "ltt_clt"        # LTT with CLT/Wald (asymptotic, unsafe; baseline)

KNOWN_METHODS = (
    METHOD_CRC,
    METHOD_LTT_HB,
    METHOD_LTT_WSR,
    METHOD_LTT_HOEFFDING,
    METHOD_LTT_CLT,
)


# ----- candidate orderings -------------------------------------------

ORDER_AGGRESSIVE_FIRST = "from_aggressive"   # try lowest-loss-side first
ORDER_CONSERVATIVE_FIRST = "from_conservative"
ORDER_FIXED = "fixed"                         # use whatever order caller passed
ORDER_BONFERRONI = "bonferroni"               # ignore order, correct for |Λ|

KNOWN_ORDERINGS = (
    ORDER_AGGRESSIVE_FIRST,
    ORDER_CONSERVATIVE_FIRST,
    ORDER_FIXED,
    ORDER_BONFERRONI,
)


# ----- internal numerical primitives ---------------------------------


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _bern_kl(p: float, q: float) -> float:
    """Bernoulli KL divergence d(p || q). p, q ∈ [0,1].

    Convention: d(0||q) = -log(1-q), d(1||q) = -log(q), d(p||0)=+inf
    if p>0, d(p||1)=+inf if p<1. Returns +inf when undefined finitely.
    """
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        return math.inf
    if p == 0.0:
        return -math.log(1.0 - q) if q < 1.0 else math.inf
    if p == 1.0:
        return -math.log(q) if q > 0.0 else math.inf
    if q <= 0.0 or q >= 1.0:
        return math.inf
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _log_binom_coeff(n: int, k: int) -> float:
    """log(C(n,k)) via lgamma. Exact for any n, k in valid range."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for X ~ Binomial(n, p). Stable for n up to ~50000."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    # Sum the smaller tail for numerical stability, then complement if needed.
    use_upper = k > n * p
    if use_upper:
        total = 0.0
        for i in range(k + 1, n + 1):
            total += math.exp(_log_binom_coeff(n, i) + i * math.log(p) + (n - i) * math.log(1.0 - p))
        return _clip(1.0 - total, 0.0, 1.0)
    total = 0.0
    for i in range(0, k + 1):
        total += math.exp(_log_binom_coeff(n, i) + i * math.log(p) + (n - i) * math.log(1.0 - p))
    return _clip(total, 0.0, 1.0)


def hoeffding_pvalue(p_hat: float, n: int, null_mean: float) -> float:
    """Hoeffding's inequality as a one-sided p-value for H_0 : E[L] ≥ null_mean.

    Specifically, with L_i ∈ [0,1] and the empirical mean p_hat, the
    Chernoff-Hoeffding bound gives

        P( p_hat ≤ p̂ | E[L] = R )  ≤  exp(-n · d(p̂ || R))   when p̂ ≤ R

    where d is the Bernoulli KL. We return that as our p-value for
    rejecting "R ≥ null_mean" in favor of "R < null_mean". p̂ > R is
    not evidence against H_0; the p-value is 1.
    """
    if n <= 0:
        return 1.0
    if p_hat >= null_mean:
        return 1.0
    return math.exp(-n * _bern_kl(p_hat, null_mean))


def bentkus_pvalue(p_hat: float, n: int, null_mean: float) -> float:
    """Bentkus' inequality (2004) as a one-sided p-value for H_0 : E[L] ≥ R.

    For L_i ∈ [0,1] and X = Σ L_i,

        P( X ≤ k | E[L] = R )  ≤  e · P( Bin(n, R) ≤ k )

    We use k = ⌊n · p̂⌋. Uniformly sharper than Hoeffding once
    n·R is moderate. Citation: Bentkus 2004, "On Hoeffding's
    inequalities", Annals of Probability.
    """
    if n <= 0:
        return 1.0
    if p_hat >= null_mean:
        return 1.0
    k = int(math.floor(n * p_hat))
    return min(1.0, math.e * _binomial_cdf(k, n, null_mean))


def hoeffding_bentkus_pvalue(p_hat: float, n: int, null_mean: float) -> float:
    """The Hoeffding-Bentkus p-value: min of the two bounds.

    This is the universal default for distribution-free risk control
    on bounded losses. Bates-Angelopoulos-Lei-Malik-Jordan 2021,
    "Distribution-Free, Risk-Controlling Prediction Sets" (Algorithm 1).
    """
    return min(
        hoeffding_pvalue(p_hat, n, null_mean),
        bentkus_pvalue(p_hat, n, null_mean),
    )


def _invert_upper(
    p_value_fn: Callable[[float], float],
    delta: float,
    lo: float,
    hi: float,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> float:
    """Bisection: find the supremum of {R ∈ [lo, hi] : p_value_fn(R) > δ}.

    p_value_fn(R) is the p-value for H_0 : E[L] ≥ R, so it is *non-
    increasing* in R (under a fixed p̂ < R). We want the largest R
    whose null we still *cannot* reject; that R is our UCB.
    """
    if p_value_fn(lo) <= delta:
        return lo
    if p_value_fn(hi) > delta:
        return hi
    left, right = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        if p_value_fn(mid) > delta:
            left = mid
        else:
            right = mid
        if right - left < tol:
            break
    return left


def hoeffding_bentkus_ucb(p_hat: float, n: int, delta: float) -> float:
    """One-sided (1−δ)-UCB on the mean of [0,1] losses.

    The largest R for which the Hoeffding-Bentkus p-value
    exceeds δ. By the duality between tests and confidence sets,
    P(E[L] ≤ this UCB) ≥ 1 − δ.
    """
    if n <= 0 or delta <= 0.0:
        return 1.0
    if delta >= 1.0:
        return _clip(p_hat, 0.0, 1.0)
    p_hat = _clip(p_hat, 0.0, 1.0)
    return _invert_upper(
        lambda R: hoeffding_bentkus_pvalue(p_hat, n, R),
        delta,
        lo=p_hat,
        hi=1.0,
    )


def hoeffding_ucb(p_hat: float, n: int, delta: float) -> float:
    """One-sided (1−δ)-UCB on the mean of [0,1] losses via pure Hoeffding.

    Closed form (no inversion needed):
        UCB = p̂ + sqrt( log(1/δ) / (2n) ) ∧ 1
    But the KL form is tighter and equally cheap.
    """
    if n <= 0 or delta <= 0.0:
        return 1.0
    if delta >= 1.0:
        return _clip(p_hat, 0.0, 1.0)
    p_hat = _clip(p_hat, 0.0, 1.0)
    return _invert_upper(
        lambda R: hoeffding_pvalue(p_hat, n, R),
        delta,
        lo=p_hat,
        hi=1.0,
    )


# ----- WSR (Waudby-Smith-Ramdas) bound -------------------------------


def wsr_pvalue(
    losses: Sequence[float],
    null_mean: float,
    *,
    B: float = 1.0,
    c: float = 0.75,
) -> float:
    """WSR predictable-mixture supermartingale p-value for H_0 : E[L] ≥ R.

    L_i ∈ [0, B]. Running mean and variance estimates set λ_t in a
    predictable way (uses only past samples). The capital process

        K_t(R) = Π_{i≤t} (1 + λ_i · (R − L_i / B))

    is a nonnegative supermartingale under H_0 : E[L]/B ≥ R/B.
    Ville's inequality gives sup_t K_t(R) ≥ 1/δ ⇒ reject. We return
    1 / sup_t K_t(R) (clipped to [0,1]) as the anytime-valid p-value.

    The predictable λ_t is the classical Waudby-Smith-Ramdas choice:

        λ_t = min( sqrt(2 · log(1/δ_target) / (t · σ̂²_{t−1})),
                   c / B )

    For p-value computation we want the result *not* to depend on δ
    (otherwise the p-value's definition becomes circular). The
    canonical workaround: fix the log term to log(2), giving the
    "betting" parametrization. Practical and well-behaved; this is
    what the WSR reference implementation does.

    Citation: Waudby-Smith & Ramdas, "Estimating means of bounded
    random variables by betting", JMLR 2024.
    """
    if not losses or B <= 0.0:
        return 1.0
    R = _clip(null_mean / B, 0.0, 1.0)
    if R <= 0.0:
        return 0.0
    # Running mean and variance with Laplace-style smoothing (matches
    # the reference implementation; avoids λ_1 being undefined).
    mu_hat = 0.5
    var_hat = 0.25
    sum_x = 0.0
    sum_x2 = 0.0
    log_capital = 0.0
    max_log_capital = 0.0
    for t, raw_loss in enumerate(losses, start=1):
        x = _clip(raw_loss / B, 0.0, 1.0)
        # Predictable λ uses statistics through t−1.
        lam = min(math.sqrt(2.0 * math.log(2.0) / max(t * var_hat, 1e-12)), c)
        # Capital factor for H_0 : E[X] ≥ R, betting against the null.
        factor = 1.0 + lam * (R - x)
        if factor <= 0.0:
            # Capital pinned at 0 ⇒ null cannot be rejected from here.
            return 1.0
        log_capital += math.log(factor)
        if log_capital > max_log_capital:
            max_log_capital = log_capital
        # Update predictable statistics with x_t for next iteration.
        sum_x += x
        sum_x2 += x * x
        n_eff = t + 1.0
        mu_hat = (0.5 + sum_x) / n_eff
        var_hat = max((0.25 + sum_x2) / n_eff - mu_hat * mu_hat, 1e-6)
    # Ville's inequality: P(sup K_t ≥ 1/δ) ≤ δ.
    p = math.exp(-max_log_capital)
    return _clip(p, 0.0, 1.0)


def wsr_ucb(
    losses: Sequence[float],
    delta: float,
    *,
    B: float = 1.0,
    c: float = 0.75,
    tol: float = 1e-6,
) -> float:
    """One-sided (1−δ)-UCB on E[L] via the WSR martingale.

    Inverts the WSR p-value over the null mean. Returns a value in
    [empirical_mean(losses), B].
    """
    if not losses:
        return B
    p_hat = statistics.fmean(losses)
    if delta <= 0.0:
        return B
    if delta >= 1.0:
        return _clip(p_hat, 0.0, B)
    return _invert_upper(
        lambda R: wsr_pvalue(losses, R, B=B, c=c),
        delta,
        lo=_clip(p_hat, 0.0, B),
        hi=B,
        tol=tol,
    )


def clt_ucb(losses: Sequence[float], delta: float, *, B: float = 1.0) -> float:
    """Asymptotic Wald UCB. Included as a deliberately *unsafe*
    baseline — useful to compare against to show how much tightness
    you give up for finite-sample validity.

    For δ ≤ 0.5 (the only useful regime) z = sqrt(2 · erfc⁻¹(2δ)).
    We approximate erfc⁻¹ via Newton on math.erfc — stdlib only.
    """
    n = len(losses)
    if n == 0:
        return B
    if delta <= 0.0:
        return B
    if delta >= 1.0:
        return _clip(statistics.fmean(losses), 0.0, B)
    p_hat = statistics.fmean(losses)
    if n == 1:
        var = 0.0
    else:
        var = statistics.pvariance(losses)
    se = math.sqrt(max(var, 0.0) / n)
    z = _z_one_sided(delta)
    return _clip(p_hat + z * se, 0.0, B)


def _z_one_sided(delta: float) -> float:
    """One-sided normal critical value: Φ⁻¹(1−δ). Bisection on math.erf."""
    if delta <= 0.0:
        return float("inf")
    if delta >= 1.0:
        return float("-inf")
    target = 1.0 - 2.0 * delta  # Φ(z) = 1 − δ  ⇒  erf(z/√2) = 1 − 2δ
    # erf is monotone increasing; bisect for z in [-10, 10].
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2.0)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----- dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class RiskPoint:
    """One calibration observation.

    `score` is the decision-relevant signal the threshold acts on
    (predicted cost, model confidence, anomaly score, etc.). `outcome`
    is whatever realized observable lets the user compute losses
    (actual cost, true label, refund-or-not). `features` is optional
    side data the loss function may need. `group` lets Mondrian-style
    selection partition the calibration set per tenant / model / task.
    """
    score: float
    outcome: Any = None
    features: Mapping[str, Any] = field(default_factory=dict)
    group: str = ""
    ts: float = field(default_factory=time.time)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(f"weight must be finite ≥ 0, got {self.weight}")
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")


LossFn = Callable[[RiskPoint, float], float]
"""Type alias for loss functions: (point, candidate_threshold) -> loss ∈ [0, B]."""


@dataclass(frozen=True)
class Risk:
    """One risk to control. `loss_fn(point, λ)` returns a non-negative
    loss bounded by `B`; `target` is the level we want E[L] ≤ target.
    `name` is for diagnostics; the runtime uses it in events.

    `monotone` is a property of the loss-as-a-function-of-λ — set
    `decreasing` if L(λ) is non-increasing in λ (the canonical case
    for a hedge-trigger threshold), `increasing` if non-decreasing.
    CRC requires one of these orderings; LTT does not.
    """
    name: str
    target: float
    loss_fn: LossFn
    B: float = 1.0
    monotone: str = "decreasing"  # one of: "decreasing", "increasing", "none"

    def __post_init__(self) -> None:
        if self.B <= 0.0:
            raise ValueError(f"B must be positive, got {self.B}")
        if not 0.0 < self.target <= self.B:
            raise ValueError(f"target must be in (0, B], got {self.target}")
        if self.monotone not in ("decreasing", "increasing", "none"):
            raise ValueError(f"monotone must be decreasing|increasing|none, got {self.monotone!r}")


@dataclass(frozen=True)
class RiskSelection:
    """The output of a selection call.

    `threshold` is the chosen λ̂. `empirical_risk` is the calibration-set
    mean of the loss at that λ̂. `ucb` is the corresponding upper
    confidence bound (under the method used). `n` is the calibration
    size that the certificate rests on; `delta` is the FWER level used.
    `risk_name` identifies which risk this selection bounds (for
    multi-risk calls; otherwise "default").
    """
    threshold: float
    method: str
    risk_name: str
    target: float
    delta: float
    empirical_risk: float
    ucb: float
    n: int
    candidates_tested: int
    group: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskReport:
    """Empirical risk diagnostics over a held-out (or in-sample) slice."""
    n: int
    threshold: float
    empirical_risk: float
    ucb: float
    method: str
    target: float
    delta: float
    drift_detected: bool = False
    per_group: dict[str, "GroupRisk"] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_group"] = {k: asdict(v) for k, v in self.per_group.items()}
        return d


@dataclass(frozen=True)
class GroupRisk:
    group: str
    n: int
    empirical_risk: float
    ucb: float


# ----- core selection algorithms -------------------------------------


def _losses_at(points: Sequence[RiskPoint], risk: Risk, threshold: float) -> list[float]:
    out = []
    for p in points:
        l = float(risk.loss_fn(p, threshold))
        if not math.isfinite(l):
            raise ValueError(f"loss_fn returned non-finite value: {l}")
        if l < 0.0 or l > risk.B:
            raise ValueError(
                f"loss_fn returned {l} outside [0, {risk.B}] at threshold {threshold}; "
                f"the bound B={risk.B} is wrong or the loss is not bounded."
            )
        out.append(l)
    return out


def _select_crc(
    points: Sequence[RiskPoint],
    risk: Risk,
    candidates: Sequence[float],
    target: float,
) -> tuple[float | None, dict[str, Any]]:
    """Conformal Risk Control. Requires monotone loss in λ.

    With L_i monotone non-increasing in λ:
      pick smallest λ such that (Σ L_i(λ) + B) / (n + 1) ≤ target.
    With L_i monotone non-decreasing in λ:
      pick largest λ such that (Σ L_i(λ) + B) / (n + 1) ≤ target.

    Returns (λ̂, diagnostics) or (None, diagnostics) if no candidate is safe.
    """
    n = len(points)
    if n == 0:
        return None, {"reason": "empty_calibration"}
    if risk.monotone not in ("decreasing", "increasing"):
        raise ValueError("CRC requires risk.monotone in {decreasing, increasing}")
    # CRC sweeps from conservative to aggressive and stops at the boundary.
    # For monotone-decreasing L(λ), the conservative side is large λ.
    if risk.monotone == "decreasing":
        sweep = sorted(candidates, reverse=True)
    else:
        sweep = sorted(candidates)
    last_safe: float | None = None
    plus_B = risk.B / (n + 1)
    tested = []
    for lam in sweep:
        losses = _losses_at(points, risk, lam)
        emp = sum(losses) / (n + 1) + plus_B
        tested.append((lam, emp))
        if emp <= target:
            last_safe = lam
        else:
            break  # past the boundary; monotone ⇒ no need to keep sweeping
    return last_safe, {
        "method": METHOD_CRC,
        "n": n,
        "B": risk.B,
        "sweep_len": len(tested),
        "boundary_at": tested[-1] if tested else None,
    }


def _ltt_pvalue(
    losses: Sequence[float],
    null_mean: float,
    *,
    method: str,
    B: float = 1.0,
) -> float:
    """Dispatch to the requested p-value family. Losses must be in [0, B];
    Hoeffding-Bentkus and Hoeffding need them normalized into [0,1] by
    dividing by B — we do that here."""
    n = len(losses)
    if n == 0 or B <= 0.0:
        return 1.0
    p_hat_norm = statistics.fmean(losses) / B
    null_norm = null_mean / B
    if method == METHOD_LTT_HB:
        return hoeffding_bentkus_pvalue(p_hat_norm, n, null_norm)
    if method == METHOD_LTT_HOEFFDING:
        return hoeffding_pvalue(p_hat_norm, n, null_norm)
    if method == METHOD_LTT_WSR:
        return wsr_pvalue(losses, null_mean, B=B)
    if method == METHOD_LTT_CLT:
        # Symmetric two-tail isn't quite right; we want the one-sided
        # upper tail. Translate via the normal CDF.
        var = statistics.pvariance(losses) if n > 1 else 0.0
        se = math.sqrt(max(var, 0.0) / n)
        if se <= 0.0:
            return 0.0 if p_hat_norm * B < null_mean else 1.0
        z = (null_mean - p_hat_norm * B) / se
        # one-sided upper-tail p = P(Z ≥ z) under H_0
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    raise ValueError(f"unknown LTT method {method!r}")


def _select_ltt(
    points: Sequence[RiskPoint],
    risk: Risk,
    candidates: Sequence[float],
    target: float,
    delta: float,
    method: str,
    ordering: str,
) -> tuple[float | None, dict[str, Any]]:
    """Learn-Then-Test: multiplicity-corrected sequential testing.

    Sequential ordering (`from_aggressive` / `from_conservative` /
    `fixed`) → reject in order, stop at first non-rejection,
    return the last rejected λ. FWER controlled at δ without
    correction (fixed-sequence procedure).

    Bonferroni → test each candidate at δ/|Λ|, return the most
    aggressive rejected candidate.

    "Aggressive" here means *the side that maximizes loss* in the
    canonical convention; we pick the most aggressive λ whose null
    is rejected and report it as λ̂.
    """
    n = len(points)
    if n == 0:
        return None, {"reason": "empty_calibration"}
    if not candidates:
        return None, {"reason": "no_candidates"}
    # Fixed-sequence testing requires the safe-side candidates first
    # and the unsafe-side last — we reject (= certify safe) as a
    # prefix and stop at the first non-rejection. The *last*
    # rejected candidate is what we deploy. For ORDER_AGGRESSIVE_FIRST
    # the caller wants the *most aggressive* certified λ, so we sweep
    # conservative → aggressive and return the last success.
    # ORDER_CONSERVATIVE_FIRST sweeps aggressive → conservative; the
    # last success is the most conservative certified λ.
    if ordering == ORDER_AGGRESSIVE_FIRST:
        if risk.monotone == "decreasing":
            sweep = sorted(candidates, reverse=True)  # conservative (large) first
        elif risk.monotone == "increasing":
            sweep = sorted(candidates)               # conservative (small) first
        else:
            sweep = list(candidates)                  # caller supplied
    elif ordering == ORDER_CONSERVATIVE_FIRST:
        if risk.monotone == "decreasing":
            sweep = sorted(candidates)               # aggressive (small) first
        elif risk.monotone == "increasing":
            sweep = sorted(candidates, reverse=True) # aggressive (large) first
        else:
            sweep = list(candidates)
    elif ordering == ORDER_FIXED:
        sweep = list(candidates)
    elif ordering == ORDER_BONFERRONI:
        sweep = list(candidates)
    else:
        raise ValueError(f"unknown ordering {ordering!r}")

    delta_per = delta / max(len(sweep), 1) if ordering == ORDER_BONFERRONI else delta
    last_safe: float | None = None
    last_p = math.nan
    last_emp = math.nan
    n_rejected = 0
    n_tested = 0
    for lam in sweep:
        losses = _losses_at(points, risk, lam)
        emp = statistics.fmean(losses) if losses else 0.0
        p = _ltt_pvalue(losses, target, method=method, B=risk.B)
        n_tested += 1
        if p <= delta_per:
            last_safe = lam
            last_p = p
            last_emp = emp
            n_rejected += 1
        else:
            if ordering != ORDER_BONFERRONI:
                # Fixed-sequence: stop at first non-rejection.
                break
            # Bonferroni: keep testing; no early stop.
    return last_safe, {
        "method": method,
        "ordering": ordering,
        "n": n,
        "B": risk.B,
        "delta_per_test": delta_per,
        "n_tested": n_tested,
        "n_rejected": n_rejected,
        "last_pvalue": last_p,
        "last_empirical_risk": last_emp,
    }


# ----- main controller -----------------------------------------------


class RiskController:
    """Distribution-free risk control for runtime operating points.

    Drain receipts into the calibration set with `record(...)`. Call
    `select(...)` to pick a threshold whose realized risk is provably
    bounded; `select_multi(...)` for several risks at once; `report(...)`
    for empirical-risk diagnostics on a holdout.

    Thread-safe: a single `RLock` guards the calibration buffer. All
    selections operate on a snapshot of the buffer taken under the
    lock, so concurrent `record()` calls cannot perturb a running
    selection.
    """

    def __init__(
        self,
        *,
        max_history: int = 10000,
        bus: EventBus | None = None,
        drift_window: int = 200,
        drift_threshold: float = 0.05,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.max_history = int(max_history)
        self.bus = bus
        self.drift_window = int(drift_window)
        self.drift_threshold = float(drift_threshold)
        self._lock = threading.RLock()
        self._points: list[RiskPoint] = []

    # ----- recording -------------------------------------------------

    def record(
        self,
        *,
        score: float,
        outcome: Any = None,
        features: Mapping[str, Any] | None = None,
        group: str = "",
        weight: float = 1.0,
    ) -> None:
        point = RiskPoint(
            score=float(score),
            outcome=outcome,
            features=dict(features or {}),
            group=group,
            weight=float(weight),
        )
        with self._lock:
            self._points.append(point)
            if len(self._points) > self.max_history:
                self._points = self._points[-self.max_history :]
        if self.bus is not None:
            self.bus.publish(Event(
                kind=RISK_OBSERVED,
                data={
                    "group": group,
                    "n_total": len(self._points),
                },
            ))

    def record_many(self, points: Iterable[RiskPoint]) -> None:
        with self._lock:
            for p in points:
                self._points.append(p)
            if len(self._points) > self.max_history:
                self._points = self._points[-self.max_history :]

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)

    def points(self) -> tuple[RiskPoint, ...]:
        with self._lock:
            return tuple(self._points)

    def clear(self) -> None:
        with self._lock:
            self._points.clear()

    # ----- selection -------------------------------------------------

    def select(
        self,
        *,
        candidates: Sequence[float],
        target: float,
        loss_fn: LossFn,
        method: str = METHOD_LTT_HB,
        delta: float = 0.10,
        B: float = 1.0,
        monotone: str = "decreasing",
        ordering: str = ORDER_AGGRESSIVE_FIRST,
        group: str = "",
        risk_name: str = "default",
    ) -> RiskSelection | None:
        """Pick a threshold whose risk is provably ≤ target at level δ.

        method=crc          Conformal Risk Control (E-bound; monotone losses).
        method=ltt_hb       LTT + Hoeffding-Bentkus (default; robust, sharp).
        method=ltt_wsr      LTT + Waudby-Smith-Ramdas (sharpest for [0,B]; finite-sample).
        method=ltt_hoeffding LTT + pure Hoeffding (looser; provided for comparison).
        method=ltt_clt      LTT + Wald/CLT (asymptotic ONLY; unsafe baseline).

        ordering controls how multiplicity is handled when method
        is an LTT variant: fixed-sequence (no correction, fastest)
        or Bonferroni (correct for |Λ| tests).

        Returns the selection or None when no candidate was certified.
        """
        risk = Risk(name=risk_name, target=target, loss_fn=loss_fn, B=B, monotone=monotone)
        return self._select_one(
            risk=risk,
            candidates=candidates,
            method=method,
            delta=delta,
            ordering=ordering,
            group=group,
        )

    def select_multi(
        self,
        *,
        candidates: Sequence[float],
        risks: Sequence[Risk],
        delta: float = 0.10,
        method: str = METHOD_LTT_HB,
        ordering: str = ORDER_AGGRESSIVE_FIRST,
        group: str = "",
    ) -> dict[str, RiskSelection | None]:
        """Bound several risks simultaneously with shared FWER δ.

        Bonferroni-splits δ across the |risks| separate risk targets.
        Each risk is then certified independently at level δ/|risks|.
        The most aggressive threshold *common* to all certified per-risk
        selections is the one to deploy. Returns the per-risk selection
        map; callers reconcile (typically by taking the max λ for
        monotone-decreasing losses).
        """
        if not risks:
            return {}
        delta_per = delta / len(risks)
        out: dict[str, RiskSelection | None] = {}
        for r in risks:
            sel = self._select_one(
                risk=r,
                candidates=candidates,
                method=method,
                delta=delta_per,
                ordering=ordering,
                group=group,
            )
            out[r.name] = sel
        return out

    def _select_one(
        self,
        *,
        risk: Risk,
        candidates: Sequence[float],
        method: str,
        delta: float,
        ordering: str,
        group: str,
    ) -> RiskSelection | None:
        if method not in KNOWN_METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {KNOWN_METHODS}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0,1), got {delta}")
        if not 0.0 < risk.target <= risk.B:
            raise ValueError(f"target must be in (0, B], got {risk.target}")
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if ordering not in KNOWN_ORDERINGS:
            raise ValueError(f"unknown ordering {ordering!r}")
        # Snapshot calibration set under the lock.
        with self._lock:
            pts = [p for p in self._points if (group == "" or p.group == group)]
        if not pts:
            if self.bus is not None:
                self.bus.publish(Event(
                    kind=RISK_FAILED,
                    data={"reason": "empty_calibration", "risk": risk.name, "group": group},
                ))
            return None
        if method == METHOD_CRC:
            lam, diag = _select_crc(pts, risk, candidates, risk.target)
        else:
            lam, diag = _select_ltt(pts, risk, candidates, risk.target, delta, method, ordering)
        if lam is None:
            if self.bus is not None:
                self.bus.publish(Event(
                    kind=RISK_FAILED,
                    data={
                        "reason": "no_certifiable_candidate",
                        "risk": risk.name,
                        "target": risk.target,
                        "delta": delta,
                        "method": method,
                        "n": len(pts),
                        "group": group,
                        **diag,
                    },
                ))
            return None
        # Compute empirical risk and UCB at the chosen threshold.
        losses_at_pick = _losses_at(pts, risk, lam)
        emp = statistics.fmean(losses_at_pick)
        ucb = _ucb_at(losses_at_pick, delta, method=method, B=risk.B)
        sel = RiskSelection(
            threshold=float(lam),
            method=method,
            risk_name=risk.name,
            target=risk.target,
            delta=delta,
            empirical_risk=emp,
            ucb=ucb,
            n=len(pts),
            candidates_tested=int(diag.get("n_tested", diag.get("sweep_len", len(candidates)))),
            group=group,
            diagnostics=diag,
        )
        if self.bus is not None:
            self.bus.publish(Event(
                kind=RISK_SELECTED,
                data={
                    "risk": risk.name,
                    "threshold": float(lam),
                    "target": risk.target,
                    "delta": delta,
                    "ucb": ucb,
                    "empirical_risk": emp,
                    "method": method,
                    "n": len(pts),
                    "group": group,
                },
            ))
        return sel

    # ----- diagnostics ----------------------------------------------

    def report(
        self,
        *,
        threshold: float,
        loss_fn: LossFn,
        B: float = 1.0,
        delta: float = 0.10,
        method: str = METHOD_LTT_HB,
        group: str = "",
        monotone: str = "decreasing",
    ) -> RiskReport:
        """Empirical-risk and group-conditional report at a fixed threshold.

        Useful as a steady-state monitor: re-run on a sliding window of
        recent receipts and compare against the previously certified
        UCB. A jump in the empirical risk past the UCB is a strong
        drift signal.
        """
        risk = Risk(name="report", target=B, loss_fn=loss_fn, B=B, monotone=monotone)
        with self._lock:
            all_pts = list(self._points)
            sel_pts = [p for p in all_pts if (group == "" or p.group == group)]
        if not sel_pts:
            return RiskReport(
                n=0,
                threshold=float(threshold),
                empirical_risk=0.0,
                ucb=B,
                method=method,
                target=risk.target,
                delta=delta,
                drift_detected=False,
                notes=("empty_calibration",),
            )
        losses = _losses_at(sel_pts, risk, threshold)
        emp = statistics.fmean(losses)
        ucb = _ucb_at(losses, delta, method=method, B=B)
        # Drift detection: compare empirical risk on the recent tail
        # vs. the head; if the tail mean exceeds head mean by more
        # than `drift_threshold` (in absolute terms, normalized by B),
        # raise the flag.
        drift = False
        if len(losses) >= 2 * self.drift_window:
            head = losses[: len(losses) // 2]
            tail = losses[-self.drift_window :]
            if statistics.fmean(tail) - statistics.fmean(head) > self.drift_threshold * B:
                drift = True
        per_group: dict[str, GroupRisk] = {}
        if group == "":
            groups = sorted({p.group for p in sel_pts if p.group})
            for g in groups:
                g_pts = [p for p in sel_pts if p.group == g]
                g_loss = _losses_at(g_pts, risk, threshold)
                if not g_loss:
                    continue
                per_group[g] = GroupRisk(
                    group=g,
                    n=len(g_loss),
                    empirical_risk=statistics.fmean(g_loss),
                    ucb=_ucb_at(g_loss, delta, method=method, B=B),
                )
        report = RiskReport(
            n=len(losses),
            threshold=float(threshold),
            empirical_risk=emp,
            ucb=ucb,
            method=method,
            target=risk.target,
            delta=delta,
            drift_detected=drift,
            per_group=per_group,
        )
        if self.bus is not None:
            self.bus.publish(Event(
                kind=RISK_REPORT,
                data={
                    "threshold": float(threshold),
                    "n": len(losses),
                    "empirical_risk": emp,
                    "ucb": ucb,
                    "drift_detected": drift,
                    "method": method,
                    "group": group,
                },
            ))
        return report


# ----- UCB dispatch (for diagnostics) --------------------------------


def _ucb_at(
    losses: Sequence[float],
    delta: float,
    *,
    method: str,
    B: float = 1.0,
) -> float:
    """One-sided (1−δ) UCB on E[L] using the named method.

    Returns a value in [empirical_mean, B]. For CRC selections we
    report a Hoeffding-Bentkus UCB (the CRC E-bound is on the
    *expected* loss of the selected λ, not a high-prob bound on
    the empirical loss of the next batch — so we expose a separate
    high-prob UCB for monitoring).
    """
    if not losses or B <= 0.0:
        return B
    p_hat = statistics.fmean(losses) / B
    n = len(losses)
    if method == METHOD_LTT_WSR:
        return wsr_ucb(losses, delta, B=B)
    if method == METHOD_LTT_HOEFFDING:
        return B * hoeffding_ucb(p_hat, n, delta)
    if method == METHOD_LTT_CLT:
        return clt_ucb(losses, delta, B=B)
    # Default — HB also serves CRC for the high-prob UCB.
    return B * hoeffding_bentkus_ucb(p_hat, n, delta)


# ----- common loss-function factories --------------------------------

def loss_indicator_above(
    score_key: str = "score",
    outcome_key: str | None = None,
    outcome_truthy: bool = True,
) -> LossFn:
    """L(p, λ) = 1 if p.score ≥ λ AND p.outcome is truthy, else 0.

    Canonical "abstention" risk: λ is an abstain-above threshold; you
    abstain on point p iff score(p) ≥ λ; the loss is 1 iff the model
    abstained on what would have been a correct call.
    """
    def _fn(p: RiskPoint, threshold: float) -> float:
        s = p.score if score_key == "score" else float(p.features.get(score_key, 0.0))
        if outcome_key is None:
            out = bool(p.outcome) if outcome_truthy else not bool(p.outcome)
        else:
            out = bool(p.features.get(outcome_key, False))
            if not outcome_truthy:
                out = not out
        return 1.0 if (s >= threshold and out) else 0.0
    return _fn


def loss_indicator_below(
    score_key: str = "score",
    outcome_key: str | None = None,
    outcome_truthy: bool = True,
) -> LossFn:
    """L(p, λ) = 1 if p.score < λ AND p.outcome truthy, else 0.

    Canonical "missed-coverage" risk: λ is a hedge-trigger; below it
    you do *not* hedge; you regret iff you didn't hedge a ticket that
    later failed.
    """
    def _fn(p: RiskPoint, threshold: float) -> float:
        s = p.score if score_key == "score" else float(p.features.get(score_key, 0.0))
        if outcome_key is None:
            out = bool(p.outcome) if outcome_truthy else not bool(p.outcome)
        else:
            out = bool(p.features.get(outcome_key, False))
            if not outcome_truthy:
                out = not out
        return 1.0 if (s < threshold and out) else 0.0
    return _fn


def loss_overrun(
    multiplier: float = 1.5,
) -> LossFn:
    """L(p, λ) = 1 if p.outcome > λ * multiplier else 0. Canonical
    "predicted-cost overran by ≥`multiplier`×" risk for preflight
    estimation. Monotone non-increasing in λ.
    """
    def _fn(p: RiskPoint, threshold: float) -> float:
        try:
            y = float(p.outcome)
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if y > threshold * multiplier else 0.0
    return _fn
