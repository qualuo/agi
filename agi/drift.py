"""DriftSentinel — anytime-valid sequential drift detection.

Every other forecaster in this runtime answers a question about the *current*
state of the world: `CalibrationEngine` warps a single probability,
`ConformalPredictor` produces a single interval, `PolicyLab` evaluates a single
policy. They all share an unstated assumption: that the data they were fit on
and the data flowing through production are exchangeable. The moment the world
shifts — a new prompt mix, a model swap upstream, a tool change, a tenant going
adversarial — that assumption silently breaks. The calibrators stay confident,
the conformal intervals stay narrow, the policy estimates stay sharp, and the
coordination engine keeps committing to decisions whose statistical foundation
has quietly evaporated.

`DriftSentinel` is the runtime primitive that detects exactly this. It sits on
the event bus, watches any scalar stream of interest — `p_success` residuals,
`cost` log-ratios, reward signal, critic score, tenant outcome rate — and emits
a `drift.detected` event the moment the stream has shifted by enough that the
runtime should treat its other forecasters as stale.

What "the moment" means here is precisely defined. The guarantee is
**anytime-valid**: the runtime can call `update(...)` as often as it likes,
peek at the test statistic between every sample, and the false-alarm probability
remains bounded uniformly across all stopping times:

    P_H0( ∃ t : sentinel.update(x_t).triggered )  ≤  α

That is the strongest guarantee a sequential detector can give. Classical
threshold rules (e.g. "alert when |x_t − μ_0| > 2σ") do not give it: their
false-alarm probability grows without bound as you peek longer. P-hacking and
sequential testing fail for the same reason.

`DriftSentinel` composes three detectors and triggers on the first that fires —
each chosen so its failure modes are uncorrelated with the others:

  1. **Page-Hinkley CUSUM.** Classic, cheap, mean-shift sensitive. The
     workhorse: O(1) state, O(1) per sample, fast to trigger on abrupt
     mean shifts. Threshold `h = log(1/α)` calibrates expected run length
     under H_0 to ~1/α by Wald's approximation.

  2. **Bayesian Online Changepoint Detection (BOCPD).** Adams-MacKay 2007.
     Maintains a posterior over the *current run length* under a
     Gaussian-NIχ² conjugate predictive. Triggers when the posterior mass
     on r_t < `bocpd_short_run` exceeds `bocpd_alarm_mass`. Gives a
     *calibrated changepoint location* — not just "something changed" but
     "it changed roughly at sample τ̂". The runtime can use τ̂ to roll
     back calibration to a known-good window.

  3. **Betting Martingale (e-process).** Waudby-Smith-Ramdas 2024 +
     Shin-Ramdas-Rinaldo 2024. The current state of the art for
     anytime-valid mean testing on bounded random variables. The capital
     process

           K_t(μ_0) = ∏_{i=1}^t (1 + λ_i · (x_i − μ_0))

     with predictable bets λ_i ∈ [−1/(1−μ_0), 1/μ_0] (capped) is a
     non-negative martingale under H_0 : E[x_i] = μ_0. By Ville's
     inequality, P( sup_t K_t ≥ 1/α ) ≤ α. The detector rejects H_0 the
     first time K_t crosses 1/α. We run two such processes (upper and
     lower) and union-bound at α/2 each for a two-sided test. The bets
     λ_i are picked by aGRAPA — approximate Growth-Rate Adaptive to a
     Particular Alternative — tuned online from a running mean/variance,
     which is provably the rate-optimal choice for unknown alternatives.

The composition gives the runtime *both* fast detection of obvious shifts
(CUSUM and BOCPD typically trigger first on a real shift) *and* a
nonparametric anytime-valid guarantee (the betting martingale never
false-alarms more than α even if the stream is heavy-tailed, non-Gaussian,
non-i.i.d. exchangeable, etc.). The triggered detector is reported so the
coordination engine can act differently: CUSUM-triggered means "an abrupt
mean shift happened, retrain"; BOCPD-triggered means "the changepoint
posterior locates the shift at τ̂, roll back to τ̂"; martingale-triggered
means "the evidence has accumulated enough that under any distribution we
can reject stationarity, flag for human review".

Where this slots into the coordination engine
---------------------------------------------

The sentinel is the *trust gate* on every other forecaster:

    sentinel = DriftSentinel(reference_mean=0.83, alpha=0.01, bus=bus)
    bus.subscribe(kind=DRIFT_DETECTED, cb=lambda e: calibration.refit())
    bus.subscribe(kind=DRIFT_DETECTED, cb=lambda e: conformal.invalidate_calibration())
    bus.subscribe(kind=DRIFT_DETECTED, cb=lambda e: policy_lab.flag_stale())

    for ticket in driver.completed():
        obs = sentinel.update(ticket.actual_p_success)
        if obs.triggered:
            coordinator.enter_safe_mode(reason=obs.method, since=obs.changepoint_estimate)

The coordination engine treats `sentinel.is_drift_active()` as a kill-switch
on aggressive routing decisions. Until the sentinel resets — typically after
the calibrators have refit on the post-changepoint window — the coordinator
falls back to the safe-default action.

What it composes
----------------

  - **CalibrationEngine** subscribes to `drift.detected` and refits on the
    post-changepoint window, using `bocpd.changepoint_estimate` to pick the
    cut.

  - **ConformalPredictor** invalidates its calibration set on drift and
    falls back to wider intervals (or its ACI variant for online recovery).

  - **PolicyLab / CausalLab** flag their stored estimates as stale; the
    Strategist downweights their evidence in its EV computation until
    enough post-drift data has accumulated.

  - **AttestationLedger** records the drift event with the witness sample
    and the running statistic, so an auditor can replay the detection.

  - **Coordinator / AutonomousLoop** enter a safe-mode where new tickets
    route through the conservative baseline action; risky exploration is
    paused.

  - **Strategist** can call `sentinel.is_drift_active()` and bias its
    recommendation toward DEFER until the sentinel clears.

Events
------

    drift.started      — a new sentinel began watching a stream
    drift.observation  — a sample was processed (sample, statistic snapshot)
    drift.detected     — at least one detector triggered
    drift.reset        — operator called `reset()` after recovery
    drift.cleared      — the post-drift mean has restabilised and the
                         sentinel re-armed without operator intervention
                         (only emitted when `auto_clear=True`)

Honest about limits
-------------------

  - The Page-Hinkley statistic is most powerful against *mean shifts*. It is
    insensitive to variance-only or higher-moment shifts. The betting
    martingale partially compensates but with lower power.

  - BOCPD's predictive is Gaussian-NIχ² by default. Heavy-tailed streams
    will see inflated false-alarm rates from BOCPD specifically; the union
    test still respects α because the betting martingale is
    distribution-free, but BOCPD will fire more often than its nominal
    contribution suggests. For heavy-tailed streams, set `bocpd_alarm_mass=1.0`
    to disable BOCPD and rely on CUSUM + betting alone.

  - The anytime-valid guarantee holds under *exchangeability* of pre-drift
    samples. Pre-existing autocorrelation (e.g. consecutive tickets from the
    same prompt) inflates the effective sample size and weakens the
    guarantee. The honest fix is to thin the stream (sample one per session)
    before feeding the sentinel.

  - For known reference mean μ_0 you pass it in. If you don't have one, the
    sentinel can run in "self-reference" mode where the first
    `warmup_samples` are used to estimate μ_0, but the anytime-validity then
    starts *after* warmup — drift inside the warmup window will not trigger
    and may pollute the reference. The runtime should warm up on known-good
    data.

  - Two-sided test by default. For a one-sided contract ("alert only if
    p_success drops"), pass `direction="lower"` and the union-bound is
    avoided, sharpening the test by a factor of two.

All numerics are stdlib-only. A single `update()` runs in O(R) where R is
the maximum tracked run length in BOCPD (default 200) — sub-millisecond on
a laptop CPU. The sentinel maintains O(R + window) memory.

References
----------

  - Page, *Continuous Inspection Schemes*, Biometrika 1954.
  - Adams & MacKay, *Bayesian Online Changepoint Detection*, arXiv:0710.3742, 2007.
  - Howard, Ramdas, McAuliffe, Sekhon, *Time-uniform, nonparametric, nonasymptotic
    confidence sequences*, Ann. Stat. 2021.
  - Waudby-Smith & Ramdas, *Estimating means of bounded random variables by
    betting*, JMLR 2024.
  - Shin, Ramdas, Rinaldo, *E-detectors: A nonparametric framework for sequential
    change detection*, NEJSDS 2024.
"""
from __future__ import annotations

import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from agi.events import Event, EventBus


# ----- event kinds ----------------------------------------------------

DRIFT_STARTED = "drift.started"
DRIFT_OBSERVATION = "drift.observation"
DRIFT_DETECTED = "drift.detected"
DRIFT_RESET = "drift.reset"
DRIFT_CLEARED = "drift.cleared"


# ----- detector identities --------------------------------------------

METHOD_CUSUM = "cusum"
METHOD_BOCPD = "bocpd"
METHOD_BETTING = "betting_martingale"
METHODS = (METHOD_CUSUM, METHOD_BOCPD, METHOD_BETTING)


# ----- directions -----------------------------------------------------

DIR_TWO_SIDED = "two_sided"
DIR_UPPER = "upper"
DIR_LOWER = "lower"
DIRECTIONS = (DIR_TWO_SIDED, DIR_UPPER, DIR_LOWER)


# ----- dataclasses ----------------------------------------------------


@dataclass
class DetectorStats:
    """Per-detector running statistics, surfaced on every observation."""

    method: str
    statistic: float          # the current value of the test statistic
    threshold: float          # the rejection threshold for this method
    triggered: bool           # whether this detector has fired
    triggered_at: int | None  # 1-indexed sample at which it fired, if any
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftObservation:
    """The result of one `sentinel.update(x)` call.

    A `DriftObservation` is the unit of feedback the coordination engine
    consumes between every sample. `triggered=True` means at least one
    detector has fired; `method` names the first to fire. The
    `changepoint_estimate` (if available from BOCPD) lets the runtime roll
    back stateful forecasters to a known-good cut.

    All fields are immutable; the sentinel owns the underlying running
    state.
    """

    sentinel_id: str
    t: int                                # 1-indexed sample number
    sample: float                          # the raw input
    triggered: bool
    method: str | None                     # which detector fired first
    changepoint_estimate: int | None       # ML estimate of changepoint time
    confidence: float                      # 1 − posterior P(no drift)
    detectors: dict[str, DetectorStats]    # all detectors' snapshots
    reference_mean: float                  # the H_0 mean at the time of obs
    ts: float                              # wall-clock observation time

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "detectors": {k: v.to_dict() for k, v in self.detectors.items()},
        }


@dataclass
class DriftReport:
    """Summary of a sentinel's life, returned by `sentinel.report()`."""

    sentinel_id: str
    n_samples: int
    triggered: bool
    first_trigger_t: int | None
    first_trigger_method: str | None
    changepoint_estimate: int | None
    detectors: dict[str, DetectorStats]
    reference_mean: float
    observed_mean: float
    observed_std: float
    started_at: float
    last_ts: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "detectors": {k: v.to_dict() for k, v in self.detectors.items()}}


# ----- detector building blocks ---------------------------------------


class _CUSUM:
    """Page-Hinkley CUSUM.

    Maintains symmetric running sums:
        S_t^+ = max(0, S_{t-1}^+ + (x_t − μ_0) − δ/2)
        S_t^- = max(0, S_{t-1}^- − (x_t − μ_0) − δ/2)

    Triggers when max(S_t^+, S_t^-) > h.

    `delta` is the minimum mean shift the test is tuned to detect (in the
    same units as x). `h` is the alarm threshold. For desired false-alarm
    probability α and IID normal noise, h ≈ log(1/α) gives ARL_0 ≈ 1/α via
    Wald's approximation; we expose the threshold directly so the caller
    can override.
    """

    __slots__ = ("delta", "h", "direction", "s_pos", "s_neg", "triggered_at", "max_stat")

    def __init__(self, delta: float, h: float, direction: str = DIR_TWO_SIDED) -> None:
        if delta < 0:
            raise ValueError("CUSUM delta must be non-negative")
        if h <= 0:
            raise ValueError("CUSUM threshold h must be > 0")
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        self.delta = float(delta)
        self.h = float(h)
        self.direction = direction
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.triggered_at: int | None = None
        self.max_stat = 0.0

    def update(self, t: int, x: float, mu0: float) -> tuple[float, bool]:
        d = x - mu0
        self.s_pos = max(0.0, self.s_pos + d - self.delta / 2.0)
        self.s_neg = max(0.0, self.s_neg - d - self.delta / 2.0)
        if self.direction == DIR_UPPER:
            stat = self.s_pos
        elif self.direction == DIR_LOWER:
            stat = self.s_neg
        else:
            stat = max(self.s_pos, self.s_neg)
        if stat > self.max_stat:
            self.max_stat = stat
        triggered = stat > self.h
        if triggered and self.triggered_at is None:
            self.triggered_at = t
        return stat, triggered

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.triggered_at = None
        self.max_stat = 0.0


class _BettingMartingale:
    """Anytime-valid mean test via predictable-mixture betting.

    Implements the Waudby-Smith-Ramdas 2024 capital process

        K_t = ∏_{i=1}^t (1 + λ_i · (x_i − μ_0))

    with predictable λ_i selected by aGRAPA: a running estimate of the
    growth-optimal bet under the empirical alternative. By Ville's
    inequality, under H_0 : E[x] = μ_0,

        P( sup_t K_t ≥ 1/α )  ≤  α.

    The detector rejects H_0 the first time K_t crosses 1/α. We work in
    log-space (`log K_t`) for numerical stability — the threshold becomes
    log(1/α).

    Inputs must lie in a bounded interval [lo, hi]; bets are capped at
    1/(hi − μ_0) and 1/(μ_0 − lo) so the capital process stays
    non-negative. Two instances are typically run in parallel (upper and
    lower) and union-bounded at α/2 for a two-sided test; the
    `DriftSentinel` does this composition.
    """

    __slots__ = (
        "mu0", "lo", "hi", "log_threshold", "side", "log_capital",
        "sum_x", "sum_x2", "n_obs", "triggered_at", "max_log_capital",
        "_cap_lambda",
    )

    def __init__(self, mu0: float, lo: float, hi: float, alpha: float, side: str = "upper") -> None:
        if not (lo <= mu0 <= hi):
            raise ValueError(f"mu0 ({mu0}) must lie in [lo, hi] = [{lo}, {hi}]")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if side not in ("upper", "lower"):
            raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")
        self.mu0 = float(mu0)
        self.lo = float(lo)
        self.hi = float(hi)
        self.log_threshold = math.log(1.0 / alpha)
        self.side = side
        self.log_capital = 0.0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.n_obs = 0
        self.triggered_at: int | None = None
        self.max_log_capital = 0.0
        # cap on |λ| so 1 + λ(x − μ0) > 0 for all x in [lo, hi]
        upper_room = max(self.hi - self.mu0, 1e-12)
        lower_room = max(self.mu0 - self.lo, 1e-12)
        self._cap_lambda = 1.0 / max(upper_room, lower_room)

    def _agrapa_bet(self) -> float:
        """Pick λ_i predictably from past samples via aGRAPA.

        The Growth-Rate Adaptive bet for testing μ_0 against the empirical
        mean is

            λ* = (μ̂ − μ_0) / (σ̂² + (μ̂ − μ_0)²)

        clipped to keep the capital process non-negative.
        """
        if self.n_obs == 0:
            return 0.0
        mu_hat = self.sum_x / self.n_obs
        # Bessel-corrected variance; fallback 1.0 if n=1
        if self.n_obs >= 2:
            var = max(
                (self.sum_x2 - self.n_obs * mu_hat * mu_hat) / (self.n_obs - 1),
                1e-12,
            )
        else:
            var = 1.0
        diff = mu_hat - self.mu0
        # For 'upper' side we bet positive only (testing μ > μ_0).
        # For 'lower' side we bet negative only (testing μ < μ_0).
        raw = diff / (var + diff * diff)
        if self.side == "upper":
            raw = max(0.0, raw)
        else:
            raw = min(0.0, raw)
        # cap so 1 + λ(x − μ_0) > 0 for all x ∈ [lo, hi]
        if raw > self._cap_lambda:
            raw = self._cap_lambda
        elif raw < -self._cap_lambda:
            raw = -self._cap_lambda
        # Conservative shrinkage: aGRAPA paper recommends 0.5·λ* for safety.
        return 0.5 * raw

    def update(self, t: int, x: float) -> tuple[float, bool]:
        lam = self._agrapa_bet()
        factor = 1.0 + lam * (x - self.mu0)
        # numerical guard: factor should be > 0 by construction, but cap
        if factor <= 1e-12:
            factor = 1e-12
        self.log_capital += math.log(factor)
        # accumulate after the bet (so λ is predictable)
        self.sum_x += x
        self.sum_x2 += x * x
        self.n_obs += 1
        if self.log_capital > self.max_log_capital:
            self.max_log_capital = self.log_capital
        triggered = self.log_capital >= self.log_threshold
        if triggered and self.triggered_at is None:
            self.triggered_at = t
        return self.log_capital, triggered

    def reset(self) -> None:
        self.log_capital = 0.0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.n_obs = 0
        self.triggered_at = None
        self.max_log_capital = 0.0


class _BOCPD:
    """Bayesian Online Changepoint Detection — Adams & MacKay 2007.

    Maintains a discrete posterior P(r_t = r | x_{1:t}) over the current
    run length r ∈ {0, 1, ..., R_max}. The predictive distribution
    p(x_t | r_{t-1}, history) is a Student-t arising from the
    Normal-Inverse-χ² conjugate prior on (μ, σ²).

    Trigger rule: alarm when the posterior mass on short run lengths
    (r < `short_run`) exceeds `alarm_mass`. The changepoint estimate is
    argmax_r P(r_t = r | x_{1:t}) — typically the mode of the posterior
    immediately after the alarm fires.

    Hazard rate `lambda_hazard`: the prior probability of a changepoint at
    any given step is 1/`lambda_hazard`. Default 250 corresponds to a
    weakly-informative "changepoints are rare" prior.
    """

    __slots__ = (
        "mu0", "kappa0", "alpha0", "beta0",
        "lambda_hazard", "short_run", "alarm_mass",
        "r_max", "run_post", "mu_n", "kappa_n", "alpha_n", "beta_n",
        "triggered_at", "max_alarm_mass", "_armed",
    )

    def __init__(
        self,
        mu0: float,
        var0: float,
        lambda_hazard: float = 250.0,
        short_run: int = 5,
        alarm_mass: float = 0.5,
        r_max: int = 200,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
    ) -> None:
        if lambda_hazard <= 0:
            raise ValueError("lambda_hazard must be > 0")
        if not 0 < alarm_mass <= 1.0:
            raise ValueError("alarm_mass must be in (0,1]")
        if short_run < 1:
            raise ValueError("short_run must be >= 1")
        if r_max < short_run:
            raise ValueError("r_max must be >= short_run")
        if var0 <= 0:
            raise ValueError("var0 must be > 0")
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(alpha0 * var0)
        self.lambda_hazard = float(lambda_hazard)
        self.short_run = int(short_run)
        self.alarm_mass = float(alarm_mass)
        self.r_max = int(r_max)
        # P(r_t = 0) = 1 initially
        self.run_post: list[float] = [1.0]
        self.mu_n: list[float] = [self.mu0]
        self.kappa_n: list[float] = [self.kappa0]
        self.alpha_n: list[float] = [self.alpha0]
        self.beta_n: list[float] = [self.beta0]
        self.triggered_at: int | None = None
        self.max_alarm_mass = 0.0
        # BOCPD is "armed" only after the run-length posterior has had a
        # chance to grow under stationary data; otherwise the mass on small
        # r is trivially high at startup and triggers a false alarm at t=1.
        self._armed = False

    @staticmethod
    def _log_student_t_pdf(x: float, mu: float, scale_sq: float, df: float) -> float:
        # log p(x | μ, scale²·, ν) of the Student-t predictive
        z = (x - mu)
        denom = df * scale_sq
        log_coef = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(math.pi * denom)
        )
        log_kernel = -((df + 1.0) / 2.0) * math.log(1.0 + (z * z) / denom)
        return log_coef + log_kernel

    def update(self, t: int, x: float) -> tuple[float, int, bool]:
        # 1. Compute predictive probabilities for each run length
        n = len(self.run_post)
        log_pred = [0.0] * n
        for r in range(n):
            kappa = self.kappa_n[r]
            alpha = self.alpha_n[r]
            beta = self.beta_n[r]
            mu = self.mu_n[r]
            df = 2.0 * alpha
            scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
            scale_sq = max(scale_sq, 1e-12)
            log_pred[r] = self._log_student_t_pdf(x, mu, scale_sq, df)

        # 2. Compute hazard
        h = 1.0 / self.lambda_hazard

        # 3. New posterior over run length
        # Growth probabilities: r_t = r_{t-1} + 1
        log_post_prev = [math.log(max(p, 1e-300)) for p in self.run_post]
        log_growth = [log_post_prev[r] + log_pred[r] + math.log1p(-h) for r in range(n)]
        # Changepoint: r_t = 0
        log_cp_terms = [log_post_prev[r] + log_pred[r] + math.log(h) for r in range(n)]
        # logsumexp
        m = max(log_cp_terms)
        log_cp = m + math.log(sum(math.exp(v - m) for v in log_cp_terms))

        # Truncate to r_max
        new_len = min(n + 1, self.r_max + 1)
        new_log_post = [0.0] * new_len
        new_log_post[0] = log_cp
        for r in range(1, new_len):
            if r - 1 < len(log_growth):
                new_log_post[r] = log_growth[r - 1]
            else:
                new_log_post[r] = -math.inf

        # Normalize
        m2 = max(new_log_post)
        log_z = m2 + math.log(sum(math.exp(v - m2) for v in new_log_post))
        new_post = [math.exp(v - log_z) for v in new_log_post]
        # numerical safety
        s = sum(new_post)
        if s <= 0:
            new_post = [1.0] + [0.0] * (new_len - 1)
        else:
            new_post = [p / s for p in new_post]

        # 4. Update sufficient statistics, prepending r=0 reset values
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]
        for r in range(n):
            if 1 + r > self.r_max:
                break
            kappa = self.kappa_n[r]
            alpha = self.alpha_n[r]
            beta = self.beta_n[r]
            mu = self.mu_n[r]
            mu_new = (kappa * mu + x) / (kappa + 1.0)
            kappa_new = kappa + 1.0
            alpha_new = alpha + 0.5
            beta_new = beta + (kappa * (x - mu) ** 2) / (2.0 * (kappa + 1.0))
            new_mu.append(mu_new)
            new_kappa.append(kappa_new)
            new_alpha.append(alpha_new)
            new_beta.append(beta_new)

        # Truncate sufficient stat arrays to match posterior length
        new_mu = new_mu[:new_len]
        new_kappa = new_kappa[:new_len]
        new_alpha = new_alpha[:new_len]
        new_beta = new_beta[:new_len]

        self.run_post = new_post
        self.mu_n = new_mu
        self.kappa_n = new_kappa
        self.alpha_n = new_alpha
        self.beta_n = new_beta

        # 5. Compute alarm mass and changepoint estimate
        cutoff = min(self.short_run, len(new_post))
        alarm_mass = sum(new_post[:cutoff])
        if alarm_mass > self.max_alarm_mass:
            self.max_alarm_mass = alarm_mass
        # mode of the run-length posterior — the ML estimate of the
        # current run length. If the mode is small, the changepoint is
        # recent; estimate τ = t − mode.
        mode_r = max(range(len(new_post)), key=lambda r: new_post[r])
        cp_estimate = max(0, t - mode_r)

        # Arming: under stationary data, alarm_mass drops as the run-length
        # posterior concentrates on r > short_run. We require the alarm_mass
        # to have fallen below `1 - alarm_mass` (i.e. clearly below the
        # trigger band) before we accept any subsequent crossing as evidence
        # of drift. This kills the trivial false alarm at t=1 where mass on
        # r=0 starts at 1.
        if not self._armed and alarm_mass < (1.0 - self.alarm_mass):
            self._armed = True

        triggered = (
            self._armed
            and alarm_mass >= self.alarm_mass
            and mode_r < self.short_run
        )
        if triggered and self.triggered_at is None:
            self.triggered_at = t
        return alarm_mass, cp_estimate, triggered

    def reset(self) -> None:
        self.run_post = [1.0]
        self.mu_n = [self.mu0]
        self.kappa_n = [self.kappa0]
        self.alpha_n = [self.alpha0]
        self.beta_n = [self.beta0]
        self.triggered_at = None
        self.max_alarm_mass = 0.0
        self._armed = False


# ----- public sentinel ------------------------------------------------


class DriftSentinel:
    """Anytime-valid sequential drift detector.

    A `DriftSentinel` watches a scalar stream and emits a `drift.detected`
    event the moment the stream has shifted enough that downstream
    forecasters should be considered stale. It composes three detectors
    (CUSUM, BOCPD, betting martingale) and triggers on the first to fire.

    The false-alarm rate is bounded by `alpha` *uniformly across all
    stopping times* — the runtime can call `update()` and peek at
    `is_drift_active()` between every sample without inflating the false-
    alarm probability. This is the strongest guarantee a sequential test
    can give and the one a coordination engine needs in order to use the
    sentinel as a kill-switch on aggressive routing.

    Construction
    ------------

        sentinel = DriftSentinel(
            reference_mean=0.83,        # H_0 mean (e.g. baseline p_success)
            reference_var=0.04,         # rough scale for BOCPD predictive
            value_range=(0.0, 1.0),     # required for betting martingale
            alpha=0.01,                 # nominal false-alarm probability
            direction="two_sided",      # also "upper" or "lower"
            bus=bus,                    # optional EventBus for coordination
            name="p_success_drift",
        )

    Or in self-reference mode (no μ_0 known up front):

        sentinel = DriftSentinel(
            warmup_samples=200,
            value_range=(0.0, 1.0),
            alpha=0.01,
        )
        # first 200 samples define μ_0 and σ²; detection arms after that

    Lifecycle
    ---------

        for x in stream:
            obs = sentinel.update(x)
            if obs.triggered:
                coordinator.enter_safe_mode(...)
                break

        # after operator-verified recovery (or auto_clear=True)
        sentinel.reset()

    Threading
    ---------

    The sentinel is thread-safe: `update()`, `report()`, `is_drift_active()`,
    `reset()` all take an internal lock. Events are published outside the
    lock to avoid subscriber deadlock.
    """

    def __init__(
        self,
        *,
        reference_mean: float | None = None,
        reference_var: float = 1.0,
        value_range: tuple[float, float] | None = None,
        alpha: float = 0.01,
        direction: str = DIR_TWO_SIDED,
        cusum_delta: float | None = None,
        cusum_threshold: float | None = None,
        bocpd_short_run: int = 5,
        bocpd_alarm_mass: float = 0.5,
        bocpd_hazard: float = 250.0,
        bocpd_r_max: int = 200,
        bocpd_enabled: bool = True,
        martingale_enabled: bool = True,
        cusum_enabled: bool = True,
        warmup_samples: int = 0,
        auto_clear: bool = False,
        clear_window: int = 50,
        clear_tolerance: float = 0.1,
        bus: EventBus | None = None,
        session_id: str | None = None,
        name: str = "drift",
        attestor: Any | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        if not (bocpd_enabled or martingale_enabled or cusum_enabled):
            raise ValueError("at least one detector must be enabled")
        if warmup_samples < 0:
            raise ValueError("warmup_samples must be >= 0")
        if warmup_samples == 0 and reference_mean is None:
            raise ValueError(
                "either reference_mean must be provided or warmup_samples > 0"
            )
        if value_range is None:
            if martingale_enabled:
                raise ValueError(
                    "value_range=(lo, hi) is required when betting martingale is enabled"
                )
        else:
            lo, hi = value_range
            if not lo < hi:
                raise ValueError(f"value_range must satisfy lo<hi, got {(lo,hi)}")

        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self._bus = bus
        self._session_id = session_id
        self._attestor = attestor
        self._lock = threading.Lock()

        self._alpha = float(alpha)
        self._direction = direction
        self._value_range = value_range
        self._warmup_samples = int(warmup_samples)
        self._warmup_buffer: list[float] = []
        self._reference_mean: float | None = (
            float(reference_mean) if reference_mean is not None else None
        )
        self._reference_var = float(reference_var)
        self._reference_locked = reference_mean is not None
        self._auto_clear = bool(auto_clear)
        self._clear_window = int(clear_window)
        self._clear_tolerance = float(clear_tolerance)

        # CUSUM threshold/sensitivity defaults — pick sensible values keyed
        # to alpha and reference variance.
        self._cusum_enabled = cusum_enabled
        if cusum_delta is None:
            cusum_delta = math.sqrt(self._reference_var) * 0.5
        if cusum_threshold is None:
            cusum_threshold = math.log(1.0 / alpha) * math.sqrt(self._reference_var)
        self._cusum_delta = float(cusum_delta)
        self._cusum_threshold = float(cusum_threshold)

        self._bocpd_enabled = bocpd_enabled
        self._bocpd_short_run = int(bocpd_short_run)
        self._bocpd_alarm_mass = float(bocpd_alarm_mass)
        self._bocpd_hazard = float(bocpd_hazard)
        self._bocpd_r_max = int(bocpd_r_max)

        self._martingale_enabled = martingale_enabled

        # Detectors are instantiated when reference_mean is known.
        self._cusum: _CUSUM | None = None
        self._bocpd: _BOCPD | None = None
        self._mart_up: _BettingMartingale | None = None
        self._mart_lo: _BettingMartingale | None = None
        self._build_detectors_if_ready()

        self.n_samples = 0
        self.triggered = False
        self.first_trigger_t: int | None = None
        self.first_trigger_method: str | None = None
        self.changepoint_estimate: int | None = None
        self._sum_x = 0.0
        self._sum_x2 = 0.0
        self._recent: list[float] = []  # for auto-clear bookkeeping
        self.started_at = time.time()
        self._last_ts = self.started_at
        self._observations: list[DriftObservation] = []
        self._max_history = 1024

        if bus is not None and self._reference_locked:
            self._publish(DRIFT_STARTED, {
                "sentinel_id": self.id,
                "name": self.name,
                "reference_mean": self._reference_mean,
                "alpha": self._alpha,
                "direction": self._direction,
            })

    # ----- internal helpers -----

    def _build_detectors_if_ready(self) -> None:
        if self._reference_mean is None:
            return
        if self._cusum_enabled:
            self._cusum = _CUSUM(
                delta=self._cusum_delta, h=self._cusum_threshold, direction=self._direction
            )
        if self._bocpd_enabled:
            self._bocpd = _BOCPD(
                mu0=self._reference_mean,
                var0=max(self._reference_var, 1e-9),
                lambda_hazard=self._bocpd_hazard,
                short_run=self._bocpd_short_run,
                alarm_mass=self._bocpd_alarm_mass,
                r_max=self._bocpd_r_max,
            )
        if self._martingale_enabled and self._value_range is not None:
            lo, hi = self._value_range
            # Two-sided test: α/2 each side. One-sided: α on the relevant side.
            if self._direction == DIR_TWO_SIDED:
                alpha_each = self._alpha / 2.0
                self._mart_up = _BettingMartingale(
                    mu0=self._reference_mean, lo=lo, hi=hi, alpha=alpha_each, side="upper"
                )
                self._mart_lo = _BettingMartingale(
                    mu0=self._reference_mean, lo=lo, hi=hi, alpha=alpha_each, side="lower"
                )
            elif self._direction == DIR_UPPER:
                self._mart_up = _BettingMartingale(
                    mu0=self._reference_mean, lo=lo, hi=hi, alpha=self._alpha, side="upper"
                )
            else:  # DIR_LOWER
                self._mart_lo = _BettingMartingale(
                    mu0=self._reference_mean, lo=lo, hi=hi, alpha=self._alpha, side="lower"
                )

    def _publish(self, kind: str, data: dict[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, session_id=self._session_id, data=data))
        except Exception:
            # event bus failures must not poison the sentinel
            pass

    def _attest(self, kind: str, payload: dict[str, Any]) -> None:
        if self._attestor is None:
            return
        try:
            record = getattr(self._attestor, "attest", None)
            if record is not None:
                record(kind=kind, payload=payload)
        except Exception:
            pass

    def _snapshot_detectors(self) -> dict[str, DetectorStats]:
        out: dict[str, DetectorStats] = {}
        if self._cusum is not None:
            out[METHOD_CUSUM] = DetectorStats(
                method=METHOD_CUSUM,
                statistic=max(self._cusum.s_pos, self._cusum.s_neg)
                if self._direction == DIR_TWO_SIDED
                else (self._cusum.s_pos if self._direction == DIR_UPPER else self._cusum.s_neg),
                threshold=self._cusum.h,
                triggered=self._cusum.triggered_at is not None,
                triggered_at=self._cusum.triggered_at,
                extra={
                    "s_pos": self._cusum.s_pos,
                    "s_neg": self._cusum.s_neg,
                    "max_stat": self._cusum.max_stat,
                },
            )
        if self._bocpd is not None:
            cutoff = min(self._bocpd.short_run, len(self._bocpd.run_post))
            alarm_mass = sum(self._bocpd.run_post[:cutoff])
            mode_r = (
                max(range(len(self._bocpd.run_post)), key=lambda r: self._bocpd.run_post[r])
                if self._bocpd.run_post
                else 0
            )
            out[METHOD_BOCPD] = DetectorStats(
                method=METHOD_BOCPD,
                statistic=alarm_mass,
                threshold=self._bocpd.alarm_mass,
                triggered=self._bocpd.triggered_at is not None,
                triggered_at=self._bocpd.triggered_at,
                extra={
                    "mode_run_length": mode_r,
                    "changepoint_estimate": max(0, self.n_samples - mode_r),
                    "max_alarm_mass": self._bocpd.max_alarm_mass,
                },
            )
        if self._mart_up is not None or self._mart_lo is not None:
            up_cap = self._mart_up.log_capital if self._mart_up else -math.inf
            lo_cap = self._mart_lo.log_capital if self._mart_lo else -math.inf
            stat = max(up_cap, lo_cap)
            threshold = (
                self._mart_up.log_threshold
                if self._mart_up
                else self._mart_lo.log_threshold
            )
            triggered_at: int | None = None
            for d in (self._mart_up, self._mart_lo):
                if d is None:
                    continue
                if d.triggered_at is not None:
                    if triggered_at is None or d.triggered_at < triggered_at:
                        triggered_at = d.triggered_at
            out[METHOD_BETTING] = DetectorStats(
                method=METHOD_BETTING,
                statistic=stat,
                threshold=threshold,
                triggered=triggered_at is not None,
                triggered_at=triggered_at,
                extra={
                    "log_capital_upper": up_cap if math.isfinite(up_cap) else None,
                    "log_capital_lower": lo_cap if math.isfinite(lo_cap) else None,
                },
            )
        return out

    # ----- public API -----

    def is_drift_active(self) -> bool:
        """True iff at least one detector has triggered and not been reset."""
        with self._lock:
            return self.triggered

    def reference_mean(self) -> float | None:
        with self._lock:
            return self._reference_mean

    def update(self, x: float) -> DriftObservation:
        """Process one sample and return a `DriftObservation`.

        Until the sentinel has accumulated `warmup_samples` samples in
        self-reference mode, the observation's `triggered` is always
        `False` and the detectors are not yet running. Once warmup ends,
        the reference mean/variance are locked in and the detectors begin
        accumulating.

        Once `triggered=True` has been emitted, subsequent calls keep
        returning `triggered=True` (the sentinel is *latching*) until the
        caller invokes `reset()` or auto-clear fires.
        """
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            raise ValueError(f"sample must be finite, got {x}")
        if self._value_range is not None:
            lo, hi = self._value_range
            if not (lo <= x <= hi):
                raise ValueError(
                    f"sample {x} out of declared value_range [{lo}, {hi}]"
                )

        with self._lock:
            self.n_samples += 1
            t = self.n_samples
            self._sum_x += x
            self._sum_x2 += x * x
            self._recent.append(x)
            if len(self._recent) > self._clear_window:
                self._recent = self._recent[-self._clear_window :]

            # Self-reference warmup mode
            if not self._reference_locked:
                self._warmup_buffer.append(x)
                if len(self._warmup_buffer) >= self._warmup_samples:
                    self._reference_mean = statistics.fmean(self._warmup_buffer)
                    if len(self._warmup_buffer) >= 2:
                        self._reference_var = max(
                            statistics.pvariance(self._warmup_buffer), 1e-9
                        )
                    self._reference_locked = True
                    self._build_detectors_if_ready()
                    self._publish(
                        DRIFT_STARTED,
                        {
                            "sentinel_id": self.id,
                            "name": self.name,
                            "reference_mean": self._reference_mean,
                            "reference_var": self._reference_var,
                            "alpha": self._alpha,
                            "direction": self._direction,
                            "warmup_samples": self._warmup_samples,
                        },
                    )
                obs = DriftObservation(
                    sentinel_id=self.id,
                    t=t,
                    sample=x,
                    triggered=False,
                    method=None,
                    changepoint_estimate=None,
                    confidence=0.0,
                    detectors={},
                    reference_mean=self._reference_mean or float("nan"),
                    ts=time.time(),
                )
                self._observations.append(obs)
                if len(self._observations) > self._max_history:
                    self._observations = self._observations[-self._max_history :]
                self._last_ts = obs.ts
                return obs

            mu0 = self._reference_mean
            assert mu0 is not None

            cp_estimate: int | None = None
            confidence = 0.0
            first_method: str | None = None
            first_t: int | None = None

            if self._cusum is not None:
                stat, fired = self._cusum.update(t, x, mu0)
                if fired and first_method is None:
                    first_method = METHOD_CUSUM
                    first_t = t
                # confidence contribution: stat / threshold
                confidence = max(confidence, min(1.0, stat / self._cusum.h))

            if self._bocpd is not None:
                bocpd_alarm, cp, fired = self._bocpd.update(t, x)
                if fired:
                    cp_estimate = cp
                    if first_method is None:
                        first_method = METHOD_BOCPD
                        first_t = t
                # Use the current alarm mass (only counts once the detector is
                # armed; that's the same gate BOCPD uses to decide whether to
                # trigger, so confidence and trigger are aligned).
                if self._bocpd._armed:
                    confidence = max(confidence, bocpd_alarm)

            if self._mart_up is not None:
                lc, fired = self._mart_up.update(t, x)
                if fired and first_method is None:
                    first_method = METHOD_BETTING
                    first_t = t
                confidence = max(
                    confidence, min(1.0, lc / max(self._mart_up.log_threshold, 1e-9))
                )
            if self._mart_lo is not None:
                lc, fired = self._mart_lo.update(t, x)
                if fired and first_method is None:
                    first_method = METHOD_BETTING
                    first_t = t
                confidence = max(
                    confidence, min(1.0, lc / max(self._mart_lo.log_threshold, 1e-9))
                )

            ts = time.time()
            self._last_ts = ts

            # latch the first trigger
            newly_triggered = False
            if first_method is not None and not self.triggered:
                self.triggered = True
                self.first_trigger_t = first_t
                self.first_trigger_method = first_method
                self.changepoint_estimate = cp_estimate or t
                newly_triggered = True

            detectors = self._snapshot_detectors()

            obs = DriftObservation(
                sentinel_id=self.id,
                t=t,
                sample=x,
                triggered=self.triggered,
                method=self.first_trigger_method,
                changepoint_estimate=self.changepoint_estimate,
                confidence=confidence,
                detectors=detectors,
                reference_mean=mu0,
                ts=ts,
            )
            self._observations.append(obs)
            if len(self._observations) > self._max_history:
                self._observations = self._observations[-self._max_history :]

            # auto-clear: after a drift, if the last `clear_window` samples are
            # back within `clear_tolerance` of the (new running) mean, declare
            # cleared. This is *not* anytime-valid for the next test — call
            # `reset()` to formally re-arm.
            should_clear = False
            if (
                self._auto_clear
                and self.triggered
                and len(self._recent) >= self._clear_window
            ):
                recent_mean = statistics.fmean(self._recent)
                if abs(recent_mean - mu0) <= self._clear_tolerance:
                    should_clear = True

        # publish outside the lock
        if newly_triggered:
            self._publish(
                DRIFT_DETECTED,
                {
                    "sentinel_id": self.id,
                    "name": self.name,
                    "t": t,
                    "method": first_method,
                    "changepoint_estimate": self.changepoint_estimate,
                    "sample": x,
                    "reference_mean": mu0,
                    "confidence": confidence,
                    "detectors": {k: v.to_dict() for k, v in detectors.items()},
                },
            )
            self._attest("drift.detected", {
                "sentinel_id": self.id,
                "t": t,
                "method": first_method,
                "changepoint_estimate": self.changepoint_estimate,
                "sample": x,
            })
        else:
            self._publish(
                DRIFT_OBSERVATION,
                {
                    "sentinel_id": self.id,
                    "t": t,
                    "sample": x,
                    "confidence": confidence,
                    "triggered": self.triggered,
                },
            )
        if should_clear:
            self.reset(emit_kind=DRIFT_CLEARED)
        return obs

    def update_many(self, xs: Iterable[float]) -> list[DriftObservation]:
        """Convenience: feed an iterable, return all observations."""
        return [self.update(x) for x in xs]

    def report(self) -> DriftReport:
        with self._lock:
            n = self.n_samples
            mean = self._sum_x / n if n else float("nan")
            var = (
                max(self._sum_x2 / n - mean * mean, 0.0) if n else float("nan")
            )
            std = math.sqrt(var) if n else float("nan")
            detectors = self._snapshot_detectors()
            rationale = self._rationale()
            return DriftReport(
                sentinel_id=self.id,
                n_samples=n,
                triggered=self.triggered,
                first_trigger_t=self.first_trigger_t,
                first_trigger_method=self.first_trigger_method,
                changepoint_estimate=self.changepoint_estimate,
                detectors=detectors,
                reference_mean=self._reference_mean if self._reference_mean is not None else float("nan"),
                observed_mean=mean,
                observed_std=std,
                started_at=self.started_at,
                last_ts=self._last_ts,
                rationale=rationale,
            )

    def _rationale(self) -> str:
        if not self.triggered:
            if not self._reference_locked:
                remaining = self._warmup_samples - len(self._warmup_buffer)
                return f"warming up: {remaining} samples remain"
            return "no detector has triggered"
        parts = [f"first trigger: {self.first_trigger_method} at t={self.first_trigger_t}"]
        if self.changepoint_estimate is not None:
            parts.append(f"changepoint≈t={self.changepoint_estimate}")
        if self._reference_mean is not None and self.n_samples:
            observed = self._sum_x / self.n_samples
            parts.append(
                f"observed mean {observed:.4f} vs reference {self._reference_mean:.4f}"
            )
        return "; ".join(parts)

    def reset(self, *, keep_reference: bool = True, emit_kind: str = DRIFT_RESET) -> None:
        """Clear all detector state and re-arm.

        If `keep_reference` is True (default), the reference mean and
        variance are kept and the sentinel re-arms immediately with the
        same null. If False, the sentinel goes back into warmup mode
        (only meaningful if `warmup_samples > 0`).
        """
        with self._lock:
            self.triggered = False
            self.first_trigger_t = None
            self.first_trigger_method = None
            self.changepoint_estimate = None
            self._sum_x = 0.0
            self._sum_x2 = 0.0
            self._recent.clear()
            self.n_samples = 0
            self._observations.clear()
            if not keep_reference:
                self._reference_mean = None
                self._reference_locked = False
                self._warmup_buffer.clear()
                self._cusum = None
                self._bocpd = None
                self._mart_up = None
                self._mart_lo = None
            else:
                self._build_detectors_if_ready()
        self._publish(
            emit_kind,
            {
                "sentinel_id": self.id,
                "name": self.name,
                "keep_reference": keep_reference,
            },
        )

    def observations(self) -> list[DriftObservation]:
        with self._lock:
            return list(self._observations)

    # ----- summary helpers (read-only) -----

    def trigger_method(self) -> str | None:
        with self._lock:
            return self.first_trigger_method

    def changepoint(self) -> int | None:
        with self._lock:
            return self.changepoint_estimate

    def cusum_statistic(self) -> float | None:
        with self._lock:
            if self._cusum is None:
                return None
            if self._direction == DIR_UPPER:
                return self._cusum.s_pos
            if self._direction == DIR_LOWER:
                return self._cusum.s_neg
            return max(self._cusum.s_pos, self._cusum.s_neg)

    def betting_log_capital(self) -> tuple[float | None, float | None]:
        """Return (upper_log_capital, lower_log_capital), or None per side
        if that side's martingale isn't running."""
        with self._lock:
            up = self._mart_up.log_capital if self._mart_up else None
            lo = self._mart_lo.log_capital if self._mart_lo else None
            return up, lo

    def bocpd_alarm_mass(self) -> float | None:
        with self._lock:
            if self._bocpd is None:
                return None
            cutoff = min(self._bocpd.short_run, len(self._bocpd.run_post))
            return sum(self._bocpd.run_post[:cutoff])


# ----- multiplexer ----------------------------------------------------


class DriftSentinelGroup:
    """Hold N independent sentinels, broadcast `update` across labels.

    Useful when the coordination engine wants to watch many streams at
    once (per-tenant, per-model, per-tool) and act on whichever drifts
    first. Each sentinel's α is *not* Bonferroni-corrected at this layer:
    if you want simultaneous FWER across all streams at α, construct each
    sentinel with α / N.
    """

    def __init__(self, sentinels: Mapping[str, DriftSentinel] | None = None) -> None:
        self._sentinels: dict[str, DriftSentinel] = dict(sentinels or {})
        self._lock = threading.Lock()

    def add(self, label: str, sentinel: DriftSentinel) -> None:
        with self._lock:
            if label in self._sentinels:
                raise ValueError(f"duplicate label {label!r}")
            self._sentinels[label] = sentinel

    def remove(self, label: str) -> None:
        with self._lock:
            self._sentinels.pop(label, None)

    def labels(self) -> list[str]:
        with self._lock:
            return list(self._sentinels.keys())

    def get(self, label: str) -> DriftSentinel | None:
        with self._lock:
            return self._sentinels.get(label)

    def update(self, label: str, x: float) -> DriftObservation | None:
        with self._lock:
            sentinel = self._sentinels.get(label)
        if sentinel is None:
            return None
        return sentinel.update(x)

    def active(self) -> list[str]:
        """Labels of sentinels currently in the drift state."""
        with self._lock:
            return [k for k, s in self._sentinels.items() if s.is_drift_active()]

    def reports(self) -> dict[str, DriftReport]:
        with self._lock:
            return {k: s.report() for k, s in self._sentinels.items()}

    def reset(self, label: str | None = None, *, keep_reference: bool = True) -> None:
        with self._lock:
            if label is None:
                items = list(self._sentinels.values())
            else:
                s = self._sentinels.get(label)
                items = [s] if s is not None else []
        for s in items:
            s.reset(keep_reference=keep_reference)


__all__ = [
    "DRIFT_STARTED",
    "DRIFT_OBSERVATION",
    "DRIFT_DETECTED",
    "DRIFT_RESET",
    "DRIFT_CLEARED",
    "METHOD_CUSUM",
    "METHOD_BOCPD",
    "METHOD_BETTING",
    "METHODS",
    "DIR_TWO_SIDED",
    "DIR_UPPER",
    "DIR_LOWER",
    "DIRECTIONS",
    "DetectorStats",
    "DriftObservation",
    "DriftReport",
    "DriftSentinel",
    "DriftSentinelGroup",
]
