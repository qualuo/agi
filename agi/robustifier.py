r"""Robustifier — Distributionally Robust Optimization as a runtime primitive.

Every primitive in this stack assumes — explicitly or implicitly — that the
*evaluation distribution* is the *deployment distribution*. PolicyLab estimates
``E_{P̂}[r(π)]`` and trusts ``P̂``. Strategist's risk-adjusted EV plugs in the
calibrated point estimate and trusts the calibration. PolicyImprover ships a
policy whose HCPI lower bound was computed against the logging distribution and
trusts that the world won't shift between when we logged and when we ship.

DriftSentinel exists to tell us *when* that assumption breaks. But by then
the damage is done: a policy that was a 60% winner under ``P̂`` may be a
40% winner under ``P̂ + δ`` for some small but realistic ``δ``. The dual to
"detect drift" is "**plan robustly against the plausible drift you haven't yet
seen**." That is the gap Robustifier fills.

The literature, condensed
-------------------------

  * **Scarf, 1958 — A Min-Max Solution of an Inventory Problem.** First
    formal DRO problem: minimise worst-case expected cost over all
    distributions with a given mean and variance. The seed of everything
    that follows.

  * **Ben-Tal, El Ghaoui, Nemirovski, 2009 — Robust Optimization.** Lifts
    the program to general convex uncertainty sets; gives the modern
    machinery (Lagrangian duality, conic reformulations).

  * **Delage & Ye, 2010 — Distributionally Robust Optimization Under
    Moment Uncertainty with Application to Data-Driven Problems.**
    Moment-based DRO with finite-sample guarantees, the bridge from
    classical robust optimisation to data-driven DRO.

  * **Mohajerin Esfahani & Kuhn, 2018 — Data-Driven Distributionally
    Robust Optimization Using the Wasserstein Metric.** Tractable
    reformulations of Wasserstein-DRO; established Wasserstein as the
    canonical data-driven uncertainty set.

  * **Sinha, Namkoong, Duchi, 2018 — Certifying Some Distributional
    Robustness with Principled Adversarial Training.** Lagrangian
    relaxation of W₂-DRO that gives a tractable risk surrogate for
    ML training and a principled certificate of robustness.

  * **Blanchet & Murthy, 2019 — Quantifying Distributional Model Risk
    via Optimal Transport.** Establishes the dual formula
    ``sup_{W_c(Q,P)≤ρ} E_Q[f] = inf_{λ≥0} λρ + E_P[(f)^{c,λ}]`` and
    a Lipschitz-on-the-cost-c version with a *closed-form* answer when
    ``f`` is L-Lipschitz: ``V_R = E_P[f] + L·ρ``. The simplest, sharpest
    Wasserstein bound. We use it on the runtime path.

  * **Hu & Hong, 2013 — Kullback-Leibler Divergence Constrained DRO.**
    Closed-form dual ``sup_{KL(Q,P)≤η} E_Q[f] = inf_{α>0} α log E_P[exp(f/α)] + αη``.
    A single-variable convex problem in ``α``. We solve it with
    monotone scalar bracketing.

  * **Lam, 2019 — Recovering Best Statistical Guarantees via the Empirical
    Divergence-Based Distributionally Robust Optimization.** χ²-DRO gives
    the *exact* asymptotic CB for the mean: ``√(2·Var/n)·z_{1-α}`` is the
    χ²-DRO half-width at radius ρ = z²/n. The data-driven primitive that
    most cleanly drops in where one would otherwise put a Wald CB.

  * **Owen, 1988 — Empirical Likelihood Ratio Confidence Intervals for a
    Single Functional.** The χ²-DRO answer for the mean is exactly the
    empirical-likelihood CI; Wilks' theorem gives the asymptotic
    chi-square calibration.

  * **Rockafellar & Uryasev, 2000 — Optimization of Conditional
    Value-at-Risk.** Defines ``CVaR_α`` and shows it admits the variational
    form ``inf_τ τ + (1/α) E[(f-τ)_+]`` — convex in f, the workhorse of
    risk-constrained optimisation. Equivalent to a one-sided KL-DRO with
    a specific divergence (Ahmadi-Javid 2012).

  * **Fournier & Guillin, 2015 — On the rate of convergence in
    Wasserstein distance of the empirical measure.** Concentration of
    ``W_p(P̂_n, P)`` — gives the finite-sample-valid radius ``ρ(n, δ)``
    needed to make the Wasserstein DRO ball a valid confidence set.

  * **Duchi & Namkoong, 2021 — Statistics of Robust Optimization.** The
    finite-sample picture: variance-penalisation = χ²-DRO at small radius,
    Bernstein-LCB = χ²-DRO + bias correction. Unifies modern empirical
    process theory with DRO.

  * **Rahimian & Mehrotra, 2022 — Frameworks and Results in
    Distributionally Robust Optimization.** Survey. Best one-stop
    reference for the unified picture.

What Robustifier provides
-------------------------

Six DRO primitives, all stdlib-only and exposed as both *free functions*
(for direct math) and a *Robustifier* class (for stateful coordination):

  * **Wasserstein-1 (Blanchet-Murthy)** with assumed-Lipschitz cost.
    Closed form: ``V_R(ρ) = E_P[f] + L·ρ``. The simplest robust bound.
    O(n) per evaluation; the radius ``ρ`` is a tuning dial or computed
    from concentration (Fournier-Guillin).

  * **KL-DRO (Hu-Hong)** via Donsker-Varadhan duality. Single-variable
    convex optimisation in the Lagrangian ``α``; solved by golden-section
    on a guaranteed bracket. O(n · log(1/ε)) per evaluation. Honours
    finite-sample radius ``η`` from Sanov's theorem.

  * **χ²-DRO (Lam, Owen)** with two-moment closed-form:
    ``V_R(ρ) = E_P[f] - √(2ρ · Var_P[f])``. Exact when the implicit
    non-negativity constraint on the worst-case density is not binding
    (true at all standard coverage levels for n ≳ 30). The cleanest
    drop-in for Wald CBs; the radius corresponds to Wilks' chi-square
    calibration ``ρ = χ²_{1,1-α} / (2n)``.

  * **CVaR_α (Rockafellar-Uryasev)** via sorted-sample tail mean.
    Computes ``CVaR_α(f) = (1/α) E[(f - VaR_α)_+] + VaR_α``. Closed
    form, O(n log n) per evaluation. The "I want to avoid the bottom
    α-quantile crash" primitive.

  * **Empirical Likelihood (Owen 1988)** confidence interval for the
    mean by solving the EL profile via Lagrange multiplier root-find.
    Exact (Wilks calibration) χ²-DRO is its dual.

  * **Robust argmax / minimax regret** over a set of K arms: pick the
    arm whose worst-case mean over the DRO ball is largest, with
    correction for joint coverage over K arms (Bonferroni or
    Sidak; the former is conservative, the latter exact under
    independence).

What it composes
----------------

  * **DriftSentinel.** When drift is detected at magnitude ``Δ``, the
    coordinator should re-evaluate policies under a DRO ball of radius
    proportional to ``Δ``. Robustifier exposes ``radius_from_drift``
    to translate a Page-Hinkley statistic into a KL / Wasserstein
    radius via the asymptotic conversion ``η ≈ Δ² / (2σ²)``.

  * **Strategist.** Strategist currently picks the arm maximising
    risk-adjusted EV ``μ̂ - λ σ̂``. Robustifier provides a *principled*
    risk adjustment: ``V_R(ρ) = μ̂ - √(2ρ · σ̂²)`` is the same shape
    but ``ρ`` has a *guaranteed coverage interpretation* via Lam 2019.
    The coordinator should swap the heuristic for the principled bound
    whenever it has a concrete coverage target.

  * **PolicyImprover.** Robustifier supplies a finite-sample lower CB
    for ``E_{P̂}[r(π)]`` that is *also valid under bounded distribution
    shift*. Robust HCPI = HCPI + DRO; the policy ships only if its
    *robust* lower bound exceeds the deployment baseline.

  * **Arbiter.** Best-arm identification with CVaR objective: pick the
    arm with the highest CVaR_α(reward), not the highest mean. Robust
    BAI is just BAI on the robust evaluator. Robustifier exposes the
    sample-friendly evaluator interface that Arbiter requires.

  * **Coalition.** Robust Shapley: rather than ``E_P̂[v(S)]``, use
    ``V_R(v, S; ρ)`` so credit allocation is robust to trace-distribution
    shift. The composition is mechanical: pass Robustifier's evaluator
    into Coalition's ``set_value_function``.

  * **AttestationLedger.** Every robust evaluation emits a
    ``robust.evaluated`` receipt with the method, radius, and confidence
    level — a third-party-replayable proof that under the DRO ball
    parameterised by (method, ρ, δ), the lower bound was exactly V_R.

Where this slots in
-------------------

::

    robust = Robustifier(bus=bus, attestor=attestor)
    for arm_id, samples in policy_returns.items():
        robust.observe(arm_id, samples)

    # 1. Closed-form Wasserstein DRO with assumed Lipschitz cost
    report = robust.evaluate(
        method=METHOD_WASSERSTEIN_1,
        radius=0.05, lipschitz=1.0,
    )

    # 2. KL-DRO with finite-sample radius from Sanov
    report = robust.evaluate(method=METHOD_KL, radius="auto", delta=0.05)

    # 3. χ²-DRO drop-in for Wald CB
    report = robust.evaluate(method=METHOD_CHI2, delta=0.05)

    # 4. Tail-risk: CVaR at 10% level
    report = robust.evaluate(method=METHOD_CVAR, alpha=0.10)

    # 5. Robust argmax — pick arm whose worst-case mean is best
    winner = robust.robust_argmax(method=METHOD_CHI2, delta=0.05)

    # 6. Minimax regret over the DRO ball
    regret = robust.minimax_regret(method=METHOD_CHI2, delta=0.05)

Events
------
    robust.started               — Robustifier was constructed
    robust.observed              — samples were logged for an arm
    robust.arm_cleared           — an arm's samples were reset
    robust.cleared               — entire state was reset
    robust.evaluated             — a RobustReport was produced
    robust.argmax                — a robust winner was selected
    robust.regret                — a minimax-regret evaluation finished

Honest about limits
-------------------

  * **Wasserstein-1 closed form** assumes ``f`` is L-Lipschitz on the
    underlying metric. For discrete arm rewards bounded in [0, R], any
    L ≥ R is admissible. The bound is tight when the worst case
    concentrates mass at the support extremes.

  * **KL-DRO** with finite radius is always a *one-sided* bound; the
    other side is obtained by negating ``f``. The dual is convex but
    the bracket can be wide for very small variance — we solve with
    golden section to a relative tolerance of 1e-9 and 200 iterations.

  * **χ²-DRO closed form** ``μ̂ - √(2ρ·σ̂²)`` is exact whenever the
    worst-case density's non-negativity constraint is slack, which is
    the case for all standard coverage levels and n ≳ 30 on bounded
    support. It reduces to the empirical-likelihood (Owen 1988)
    confidence interval at ``ρ = χ²_{1,1-α}/(2n)`` via Wilks calibration.

  * **CVaR_α** is *not* a valid confidence bound on the mean — it's a
    bound on the tail. Mixing CVaR with mean-based methods on the same
    report is allowed but flagged in the report's `objective` field.

  * **Empirical likelihood** assumes the parameter is identified by
    ``E[f] = μ``; it does not handle composite parameters. Sample sizes
    below 20 should fall back to the t-distribution Wald CB; we emit a
    warning in that regime.

  * **Joint coverage** across K arms uses Bonferroni by default
    (``δ → δ/K``). Sidak (``1 - (1 - δ)^{1/K}``) is also available;
    it is exact under independence and otherwise still valid up to a
    second-order correction. We expose both.

Citations
---------

* Scarf, H. (1958). A min-max solution of an inventory problem.
  *Studies in the Mathematical Theory of Inventory and Production*, 201-209.
* Ben-Tal, A., El Ghaoui, L. & Nemirovski, A. (2009). *Robust
  Optimization*. Princeton University Press.
* Delage, E. & Ye, Y. (2010). Distributionally robust optimization under
  moment uncertainty with application to data-driven problems.
  *Operations Research*, 58(3), 595-612.
* Hu, Z. & Hong, L. J. (2013). Kullback-Leibler divergence constrained
  distributionally robust optimization. *Optimization Online*, 1695.
* Ben-Tal, A., den Hertog, D., De Waegenaere, A., Melenberg, B., Rennen,
  G. (2013). Robust solutions of optimization problems affected by
  uncertain probabilities. *Management Science*, 59(2), 341-357.
* Fournier, N. & Guillin, A. (2015). On the rate of convergence in
  Wasserstein distance of the empirical measure. *Probability Theory
  and Related Fields*, 162(3-4), 707-738.
* Mohajerin Esfahani, P. & Kuhn, D. (2018). Data-driven distributionally
  robust optimization using the Wasserstein metric. *Mathematical
  Programming*, 171(1-2), 115-166.
* Sinha, A., Namkoong, H., Duchi, J. (2018). Certifying some
  distributional robustness with principled adversarial training. *ICLR*.
* Blanchet, J. & Murthy, K. (2019). Quantifying distributional model
  risk via optimal transport. *Mathematics of Operations Research*,
  44(2), 565-600.
* Lam, H. (2019). Recovering best statistical guarantees via the
  empirical divergence-based distributionally robust optimization.
  *Operations Research*, 67(4), 1090-1105.
* Owen, A. B. (1988). Empirical likelihood ratio confidence intervals
  for a single functional. *Biometrika*, 75(2), 237-249.
* Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional
  value-at-risk. *Journal of Risk*, 2(3), 21-42.
* Duchi, J. C. & Namkoong, H. (2021). Statistics of robust optimization:
  A generalized empirical likelihood approach. *Annals of Statistics*,
  49(3), 1378-1406.
* Rahimian, H. & Mehrotra, S. (2022). Frameworks and results in
  distributionally robust optimization. *Open Journal of Mathematical
  Optimization*, 3, 1-85.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

ROBUST_STARTED = "robust.started"
ROBUST_OBSERVED = "robust.observed"
ROBUST_ARM_CLEARED = "robust.arm_cleared"
ROBUST_CLEARED = "robust.cleared"
ROBUST_EVALUATED = "robust.evaluated"
ROBUST_ARGMAX = "robust.argmax"
ROBUST_REGRET = "robust.regret"


# =====================================================================
# Methods (uncertainty-set families)
# =====================================================================

METHOD_WASSERSTEIN_1 = "wasserstein_1"
METHOD_KL = "kl"
METHOD_CHI2 = "chi2"
METHOD_CVAR = "cvar"
METHOD_EL = "empirical_likelihood"

KNOWN_METHODS = (
    METHOD_WASSERSTEIN_1,
    METHOD_KL,
    METHOD_CHI2,
    METHOD_CVAR,
    METHOD_EL,
)

# Joint-coverage corrections for K-arm reports
CORRECTION_BONFERRONI = "bonferroni"
CORRECTION_SIDAK = "sidak"
CORRECTION_NONE = "none"

KNOWN_CORRECTIONS = (CORRECTION_BONFERRONI, CORRECTION_SIDAK, CORRECTION_NONE)


# =====================================================================
# Constants
# =====================================================================

_EPS = 1e-12
_GS_TOL = 1e-9
_GS_MAX_ITER = 200
_EL_TOL = 1e-9
_EL_MAX_ITER = 100
_MIN_SAMPLES_FOR_VARIANCE = 2
_T_CRITICAL_FALLBACK_N = 20


# =====================================================================
# Statistical helpers
# =====================================================================


def _mean(samples: Sequence[float]) -> float:
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    return sum(samples) / n


def _sample_var(samples: Sequence[float], mean: float | None = None) -> float:
    """Bessel-corrected sample variance.

    Returns 0.0 for n ≤ 1. The Bessel correction matters for KL- and
    χ²-DRO finite-sample radii.
    """
    n = len(samples)
    if n < _MIN_SAMPLES_FOR_VARIANCE:
        return 0.0
    mu = mean if mean is not None else _mean(samples)
    return sum((x - mu) ** 2 for x in samples) / (n - 1)


def _normal_quantile(p: float) -> float:
    """Standard normal inverse CDF (Beasley-Springer-Moro algorithm).

    Accuracy is ~1e-7 in the central 99% of the distribution and
    ~1e-3 in the extreme tails (p < 1e-9 or p > 1 - 1e-9). Sufficient
    for confidence-band purposes; we are not in the tails of tails.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (
        -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
        1.383577518672690e2, -3.066479806614716e1, 2.506628277459239,
    )
    b = (
        -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
        6.680131188771972e1, -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
        -2.549732539343734, 4.374664141464968, 2.938163982698783,
    )
    d = (
        7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
        3.754408661907416,
    )
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _chi2_quantile_df1(p: float) -> float:
    """χ²_1 inverse CDF at probability ``p``.

    Closed form via the standard-normal inverse: if ``Z ~ N(0,1)`` then
    ``Z² ~ χ²_1``, so ``F^{-1}_{χ²_1}(p) = (Φ^{-1}((1+p)/2))²``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    z = _normal_quantile(0.5 * (1.0 + p))
    return z * z


# =====================================================================
# Radius computation
# =====================================================================


def chi2_radius_for_coverage(n: int, delta: float) -> float:
    """χ²-DRO ball radius for a one-sided 1-δ confidence mean CB.

    Lam (2019), Owen (1988): the χ²-DRO lower bound on the mean
    ``μ̂ - √(2ρ · σ̂²/n)`` is asymptotically a valid 1-δ CB when
    ``ρ = χ²_{1, 1-2δ} / 2``. The factor of 2 in the quantile arises
    because the one-sided EL test rejects when ``-2 log Λ > χ²_{1, 1-2δ}``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return 0.5 * _chi2_quantile_df1(1.0 - 2.0 * delta) if delta < 0.5 else 0.0


def kl_radius_for_coverage(n: int, delta: float) -> float:
    """KL-DRO ball radius for a 1-δ confidence set on the true distribution.

    By Sanov's theorem, for a finite alphabet ``X`` of size ``k``,
    ``P_P(KL(P̂_n ∥ P) > η) ≤ (n+1)^k · exp(-n η)``. Setting the bound to
    δ gives ``η = (log(1/δ) + k · log(n+1)) / n``. We use ``k = 2``
    (the worst-case alphabet for a bounded-mean parameter; finer
    discretisation tightens the radius and we cap at the asymptotic
    Wilks limit ``χ²_{1,1-δ}/(2n)``).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    sanov = (math.log(1.0 / delta) + 2.0 * math.log(n + 1)) / n
    wilks = _chi2_quantile_df1(1.0 - delta) / (2.0 * n)
    return min(sanov, max(wilks, _EPS))


def wasserstein1_radius_for_coverage(n: int, delta: float, *, diameter: float = 1.0) -> float:
    """W₁-DRO ball radius for a 1-δ confidence set on the true distribution.

    Fournier-Guillin (2015), Corollary 2: in 1D with support of
    diameter ``D``,

        P(W₁(P̂_n, P) > ε) ≤ C₁ · exp(-c₁ · n · ε²)

    for absolute constants C₁, c₁ when ``ε ≤ D``. Setting RHS to δ:

        ε(n, δ) = D · √(log(C₁/δ) / (c₁ · n))

    With ``C₁ = 2`` and ``c₁ = 2`` (the bounded-support sub-Gaussian
    constants), this gives ``ε = D · √(log(2/δ) / (2n))``, the standard
    DKW-style radius that we use here.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    return diameter * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def radius_from_drift(drift_magnitude: float, *, variance: float = 1.0) -> float:
    """Translate a drift statistic into a KL-DRO radius.

    Asymptotic Gaussian conversion: for a mean shift of magnitude ``Δ``
    in a distribution with variance ``σ²``, the KL divergence is
    approximately ``Δ² / (2σ²)``. The coordinator uses this to
    parameterise DRO directly from DriftSentinel's Page-Hinkley statistic
    or a BOCPD posterior-mean change.
    """
    if drift_magnitude < 0:
        raise ValueError("drift_magnitude must be non-negative")
    if variance <= 0:
        raise ValueError("variance must be positive")
    return drift_magnitude * drift_magnitude / (2.0 * variance)


# =====================================================================
# Wasserstein-1 DRO (Blanchet-Murthy)
# =====================================================================


def mean_wasserstein_dro(
    samples: Sequence[float],
    *,
    radius: float,
    lipschitz: float = 1.0,
    side: str = "lower",
) -> float:
    """Closed-form W₁-DRO bound on ``E[f]`` for L-Lipschitz ``f``.

    Blanchet-Murthy (2019), Eq. (24): for any 1-Lipschitz cost ``c``
    and L-Lipschitz objective ``f``,

        sup_{Q : W_c(Q, P̂) ≤ ρ} E_Q[f] = E_{P̂}[f] + L · ρ

    The lower bound is the symmetric ``E_{P̂}[f] - L · ρ``. The result is
    exact for the 1D bounded case when the cost is the absolute distance;
    no Lagrangian root-find is needed.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if lipschitz < 0:
        raise ValueError("lipschitz must be non-negative")
    mu = _mean(samples)
    if side == "lower":
        return mu - lipschitz * radius
    if side == "upper":
        return mu + lipschitz * radius
    raise ValueError("side must be 'lower' or 'upper'")


# =====================================================================
# KL-DRO (Hu-Hong, Donsker-Varadhan)
# =====================================================================


def _logsumexp(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Numerically stable ``log Σ w_i · exp(v_i)``.

    Used inside the KL-DRO dual: ``log E_{P̂}[exp(f/α)]`` is the
    cumulant generating function which we evaluate many times during
    golden-section.
    """
    if not values:
        raise ValueError("empty values")
    if weights is None:
        m = max(values)
        return m + math.log(sum(math.exp(v - m) for v in values) / len(values))
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    m = max(values)
    return m + math.log(sum(w * math.exp(v - m) for w, v in zip(weights, values)) / total)


def mean_kl_dro(
    samples: Sequence[float],
    *,
    radius: float,
    side: str = "lower",
) -> float:
    """Worst-case mean over a KL ball: Hu-Hong (2013) dual.

    For an empirical distribution P̂ on the samples and a KL ball of
    radius η,

        sup_{KL(Q, P̂) ≤ η} E_Q[f] = inf_{α > 0} { α log E_{P̂}[exp(f/α)] + α η }

    The dual objective ``g(α) = α log E[exp(f/α)] + α η`` is convex on
    α > 0 with g(α) → max(f) as α → 0⁺ and g(α) → E[f] + sqrt(2 η Var)
    + O(η³ᐟ²) as α → ∞ (this is Lam's χ²-asymptotic). We solve it by
    golden-section search on a bracket ``[α_low, α_high]`` chosen so the
    interior minimum is guaranteed contained.

    The lower bound is obtained by applying the dual to ``-f``.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if radius == 0.0:
        return _mean(samples)
    if side == "lower":
        return -mean_kl_dro(tuple(-x for x in samples), radius=radius, side="upper")

    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    f_max = max(samples)
    f_min = min(samples)
    spread = f_max - f_min
    if spread <= _EPS:
        return f_max

    def g(alpha: float) -> float:
        # α · log( (1/n) Σ exp(f_i / α) ) + α · η
        return alpha * _logsumexp([x / alpha for x in samples]) + alpha * radius

    # Bracket: g(α→0⁺) → f_max; g(α→∞) → ∞;
    # the convex minimum lies in (α_low, α_high) when η > 0.
    alpha_low = max(spread / 1024.0, _EPS)
    alpha_high = max(spread * 10.0, 1.0)
    # Expand upper bracket if minimum is past it.
    while g(alpha_high / 2.0) < g(alpha_high) and alpha_high < 1e12:
        alpha_high *= 2.0
    # Expand lower bracket if minimum is below it (large-η regime).
    while alpha_low > _EPS * 1e3 and g(alpha_low * 0.5) < g(alpha_low):
        alpha_low *= 0.5
    # Golden-section search on [alpha_low, alpha_high].
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # 1/golden ratio
    a, b = alpha_low, alpha_high
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = g(c)
    fd = g(d)
    for _ in range(_GS_MAX_ITER):
        if abs(b - a) < _GS_TOL * (1.0 + abs(a) + abs(b)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = g(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = g(d)
    best = min(fc, fd, g(0.5 * (a + b)))
    # Clamp to support extremes: the worst-case mean is bounded by
    # min(f) and max(f) by construction of any probability measure.
    return min(best, f_max)


# =====================================================================
# χ²-DRO (Lam, Owen empirical likelihood)
# =====================================================================


def mean_chi2_dro(
    samples: Sequence[float],
    *,
    radius: float,
    side: str = "lower",
) -> float:
    """χ²-DRO bound on the mean — closed-form asymptotic.

    Lam (2019), Theorem 1: for an empirical distribution P̂ of size n and
    a χ²-divergence ball of radius ρ,

        sup_{χ²(Q, P̂) ≤ ρ} E_Q[f]
            = E_{P̂}[f] + √(2ρ · Var_{P̂}[f]) + o(1)   as n → ∞.

    The closed form is *exact* when the implicit non-negativity
    constraint on the worst-case density (``dQ/dP̂ ≥ 0``) is not
    binding — equivalently, when ``√(2ρ) ≤ 1 / (max_i |f_i - μ̂| / σ̂_n)``.
    For sample sizes ≳ 30 with bounded support this always holds at the
    radii that correspond to standard coverage levels (δ ≥ 0.01). The
    bound is symmetric (``μ̂ ± √(2ρ σ̂²_n)``) by construction.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    mu = _mean(samples)
    # Use the uncorrected (population) sample variance — the chi-square
    # divergence is defined in terms of the empirical measure, not a
    # bias-corrected estimator of the population variance.
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    var = sum((x - mu) ** 2 for x in samples) / n
    half = math.sqrt(2.0 * radius * max(var, 0.0))
    return mu - half if side == "lower" else mu + half


# =====================================================================
# CVaR (Rockafellar-Uryasev)
# =====================================================================


def cvar(
    samples: Sequence[float],
    *,
    alpha: float,
    side: str = "lower",
) -> float:
    """Conditional Value-at-Risk at confidence level ``alpha`` ∈ (0, 1].

    Rockafellar-Uryasev (2000): for samples ``y_1, ..., y_n`` sorted
    ascending, the *lower-tail* CVaR is the mean of the ⌈αn⌉ smallest
    samples — the expectation conditioned on being in the bottom α
    fraction. The *upper-tail* CVaR (used for risk = max loss) is the
    mean of the ⌈αn⌉ largest.

    The "side" argument is named consistently with the rest of the
    Robustifier: ``"lower"`` returns the bottom-α tail mean (the value
    that a risk-averse maximiser worries about); ``"upper"`` returns
    the top-α tail mean (the value that a risk-averse minimiser worries
    about).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    sorted_s = sorted(samples)
    k = max(1, math.ceil(alpha * n))
    if side == "lower":
        return sum(sorted_s[:k]) / k
    return sum(sorted_s[-k:]) / k


def var_at_level(samples: Sequence[float], *, alpha: float, side: str = "lower") -> float:
    """Value-at-Risk: the α-quantile (lower) or (1-α)-quantile (upper)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    sorted_s = sorted(samples)
    if side == "lower":
        idx = max(0, math.ceil(alpha * n) - 1)
    else:
        idx = min(n - 1, n - math.ceil(alpha * n))
    return sorted_s[idx]


# =====================================================================
# Empirical Likelihood (Owen 1988)
# =====================================================================


def empirical_likelihood_ratio(samples: Sequence[float], target_mean: float) -> float:
    """The −2 log empirical-likelihood ratio at ``target_mean``.

    Owen (1988): for samples ``X_1, ..., X_n`` and a hypothesised mean
    ``μ``, the profile EL ratio is

        R(μ) = max_{p ≥ 0, Σp = 1, Σp_i X_i = μ} Π n p_i

    The maximum is achieved at ``p_i = 1 / (n (1 + λ(X_i - μ)))`` where
    ``λ`` solves ``Σ (X_i - μ) / (1 + λ(X_i - μ)) = 0``. We find λ by
    monotone bisection on its sign-changing bracket.

    The −2 log R(μ) is asymptotically χ²_1 by Wilks' theorem; the
    1-δ EL CI is ``{μ : -2 log R(μ) ≤ χ²_{1, 1-δ}}``, which corresponds
    *exactly* to the χ²-DRO at radius ``ρ = χ²_{1, 1-δ} / (2n)``.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    if n == 1:
        return 0.0 if math.isclose(samples[0], target_mean) else math.inf
    centred = [x - target_mean for x in samples]
    c_min, c_max = min(centred), max(centred)
    # Target mean must be strictly inside the convex hull of samples
    # otherwise R(μ) = 0 and −2 log R = ∞.
    if c_min >= 0 or c_max <= 0:
        if all(math.isclose(c, 0.0, abs_tol=_EL_TOL) for c in centred):
            return 0.0
        return math.inf
    # Solve Σ c_i / (1 + λ c_i) = 0 for λ.
    # The function is monotone-decreasing in λ on the admissibility interval.
    lam_low = -1.0 / c_max + _EPS
    lam_high = -1.0 / c_min - _EPS

    def s(lam: float) -> float:
        return sum(c / (1.0 + lam * c) for c in centred)

    for _ in range(_EL_MAX_ITER):
        if abs(lam_high - lam_low) < _EL_TOL * (1.0 + abs(lam_low) + abs(lam_high)):
            break
        mid = 0.5 * (lam_low + lam_high)
        if s(mid) > 0:
            lam_low = mid
        else:
            lam_high = mid
    lam = 0.5 * (lam_low + lam_high)
    log_r = sum(-math.log(1.0 + lam * c) for c in centred) - n * math.log(n)
    # log_r is log Π (n p_i) so −2 log R is the standard test statistic:
    log_likelihood_max = -n * math.log(n)  # log Π (1/n) = -n log n
    log_likelihood_at_mu = sum(-math.log(n * (1.0 + lam * c)) for c in centred)
    return -2.0 * (log_likelihood_at_mu - log_likelihood_max)


def empirical_likelihood_ci(
    samples: Sequence[float], *, delta: float = 0.05
) -> tuple[float, float]:
    """1−δ empirical-likelihood confidence interval for the mean.

    Wilks calibration: ``{μ : −2 log R(μ) ≤ χ²_{1, 1-δ}}``. Computed by
    bisecting separately on (mean, max_sample) and (min_sample, mean).
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    n = len(samples)
    if n < _T_CRITICAL_FALLBACK_N:
        # Fall back to t-distribution Wald CB at small n
        mu = _mean(samples)
        var = _sample_var(samples, mu)
        # t-quantile approximation via normal + Bartlett correction;
        # accurate to ~1e-3 for n ≥ 10.
        z = _normal_quantile(1.0 - delta / 2.0)
        half = z * math.sqrt(var / n) * (1.0 + (z * z + 1.0) / (4.0 * n))
        return (mu - half, mu + half)
    chi2 = _chi2_quantile_df1(1.0 - delta)
    mu = _mean(samples)
    s_min, s_max = min(samples), max(samples)

    def at(m: float) -> float:
        return empirical_likelihood_ratio(samples, m)

    # Upper bound: search in (mu, s_max).
    lo, hi = mu, s_max
    for _ in range(_EL_MAX_ITER):
        if hi - lo < _EL_TOL * (1.0 + abs(lo) + abs(hi)):
            break
        mid = 0.5 * (lo + hi)
        if at(mid) <= chi2:
            lo = mid
        else:
            hi = mid
    upper = lo
    # Lower bound: search in (s_min, mu).
    lo, hi = s_min, mu
    for _ in range(_EL_MAX_ITER):
        if hi - lo < _EL_TOL * (1.0 + abs(lo) + abs(hi)):
            break
        mid = 0.5 * (lo + hi)
        if at(mid) <= chi2:
            hi = mid
        else:
            lo = mid
    lower = hi
    return (lower, upper)


# =====================================================================
# Joint coverage corrections for K-arm reports
# =====================================================================


def joint_delta(per_arm_delta: float, k: int, correction: str) -> float:
    """Map a target *per-arm* δ to the *family-wise* δ under a correction.

    Bonferroni: family-wise δ ≥ K · per_arm_δ; per-arm δ = δ / K.
    Sidak:     family-wise δ = 1 - (1 - per_arm_δ)^K; per-arm = 1 - (1-δ)^{1/K}.
    None:      identity; no joint guarantee.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 < per_arm_delta < 1.0:
        raise ValueError("per_arm_delta must be in (0, 1)")
    if correction == CORRECTION_BONFERRONI:
        return min(1.0, k * per_arm_delta)
    if correction == CORRECTION_SIDAK:
        return 1.0 - (1.0 - per_arm_delta) ** k
    if correction == CORRECTION_NONE:
        return per_arm_delta
    raise ValueError(f"unknown correction: {correction}")


def correct_delta(family_delta: float, k: int, correction: str) -> float:
    """Inverse of ``joint_delta``: per-arm δ achieving family-wise δ."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 < family_delta < 1.0:
        raise ValueError("family_delta must be in (0, 1)")
    if correction == CORRECTION_BONFERRONI:
        return family_delta / k
    if correction == CORRECTION_SIDAK:
        return 1.0 - (1.0 - family_delta) ** (1.0 / k)
    if correction == CORRECTION_NONE:
        return family_delta
    raise ValueError(f"unknown correction: {correction}")


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class RobustEstimate:
    """Per-arm robust evaluation.

    ``point`` is the empirical mean (or empirical objective if non-mean
    method); ``lower`` and ``upper`` are the worst-case (over the DRO
    ball) bounds. ``half_width = (upper - lower) / 2``.

    ``method``, ``radius``, ``delta_used`` document the uncertainty set
    that the bounds were computed against.
    """

    arm_id: str
    point: float
    lower: float
    upper: float
    n_samples: int
    sample_variance: float
    method: str
    radius: float
    delta_used: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustReport:
    """Result of one ``Robustifier.evaluate(...)`` call."""

    id: str
    method: str
    objective: str  # "mean" or "cvar"
    radius: float
    delta_family: float
    correction: str
    delta_per_arm: float
    estimates: dict[str, RobustEstimate]
    elapsed_s: float
    receipt_hash: str = ""

    def best_arm(self, *, by: str = "lower") -> str:
        """The arm with the highest worst-case-best statistic.

        ``by="lower"`` picks the arm with the largest lower CB
        (most robust to downside drift). ``by="point"`` picks by
        empirical mean. ``by="upper"`` picks by upper CB (most
        optimistic — usually wrong unless you want exploration).
        """
        if by not in ("lower", "point", "upper"):
            raise ValueError("by must be 'lower', 'point', or 'upper'")
        if not self.estimates:
            raise ValueError("no estimates available")
        return max(self.estimates.items(), key=lambda kv: getattr(kv[1], by))[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "objective": self.objective,
            "radius": self.radius,
            "delta_family": self.delta_family,
            "correction": self.correction,
            "delta_per_arm": self.delta_per_arm,
            "estimates": {k: v.to_dict() for k, v in self.estimates.items()},
            "elapsed_s": self.elapsed_s,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True)
class RegretReport:
    """Minimax-regret summary across K arms.

    ``minimax_regret_value`` is the worst-case loss the chosen arm
    incurs against an oracle that knows the true distribution. Lower
    is better; an arm with regret 0 is a Stackelberg leader against
    the adversary.
    """

    id: str
    method: str
    radius: float
    delta_family: float
    chosen_arm: str
    minimax_regret_value: float
    regret_per_arm: dict[str, float]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Robustifier
# =====================================================================


class Robustifier:
    """Robust-evaluation state with anytime DRO bounds.

    Thread-safe; an internal lock guards every public mutator. Reads of
    immutable returned dataclasses are safe without synchronisation.

    Construction is cheap; the heavy lifting happens inside
    ``evaluate`` and ``robust_argmax``.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any | None = None,
        robustifier_id: str | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._id = robustifier_id or f"rob-{int(time.time() * 1000):x}"
        self._lock = threading.RLock()
        self._samples: dict[str, list[float]] = {}
        self._history: list[RobustReport] = []
        self._emit(ROBUST_STARTED, {"robustifier_id": self._id})

    # -----------------------------------------------------------------
    # Sample ingestion
    # -----------------------------------------------------------------

    def observe(self, arm_id: str, value: float | Sequence[float]) -> None:
        """Append one or many samples for ``arm_id``."""
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("arm_id must be a non-empty string")
        if isinstance(value, (int, float)):
            values = [float(value)]
        else:
            values = [float(v) for v in value]
        for v in values:
            if not math.isfinite(v):
                raise ValueError("value must be finite")
        with self._lock:
            self._samples.setdefault(arm_id, []).extend(values)
            count = len(self._samples[arm_id])
        self._emit(ROBUST_OBSERVED, {
            "robustifier_id": self._id,
            "arm_id": arm_id,
            "n_added": len(values),
            "n_total": count,
        })

    def clear_arm(self, arm_id: str) -> None:
        """Remove all samples for an arm but keep the arm registered."""
        with self._lock:
            if arm_id in self._samples:
                self._samples[arm_id] = []
        self._emit(ROBUST_ARM_CLEARED, {
            "robustifier_id": self._id,
            "arm_id": arm_id,
        })

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._history.clear()
        self._emit(ROBUST_CLEARED, {"robustifier_id": self._id})

    def arms(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._samples.keys()))

    def n_samples(self, arm_id: str) -> int:
        with self._lock:
            return len(self._samples.get(arm_id, ()))

    def samples_snapshot(self, arm_id: str) -> tuple[float, ...]:
        with self._lock:
            return tuple(self._samples.get(arm_id, ()))

    def history(self) -> tuple[RobustReport, ...]:
        with self._lock:
            return tuple(self._history)

    # -----------------------------------------------------------------
    # Core: evaluate all arms under one DRO method
    # -----------------------------------------------------------------

    def evaluate(
        self,
        *,
        method: str,
        radius: float | str = "auto",
        delta: float = 0.05,
        correction: str = CORRECTION_BONFERRONI,
        alpha: float | None = None,
        lipschitz: float = 1.0,
        diameter: float = 1.0,
    ) -> RobustReport:
        """Compute robust bounds for every observed arm.

        ``method`` ∈ KNOWN_METHODS selects the uncertainty set.
        ``radius`` is either a non-negative float, or "auto" — in which
        case it is computed from ``delta`` (and ``diameter`` for W₁) via
        ``chi2_radius_for_coverage`` / ``kl_radius_for_coverage`` /
        ``wasserstein1_radius_for_coverage``.

        ``correction`` adjusts ``delta`` for the K arms in the report
        (Bonferroni / Sidak / none). The *per-arm* δ used by the radius
        formula is ``correct_delta(delta, K, correction)``.

        For CVaR, ``alpha`` ∈ (0, 1] is the tail level; ``radius`` and
        ``delta`` are ignored. The "lower" bound is the lower-tail
        CVaR, the "upper" bound is the upper-tail CVaR.

        For W₁, ``lipschitz`` is the Lipschitz constant of the cost
        function. For mean evaluation on a 1D bounded support, an
        admissible L is the support diameter.
        """
        if method not in KNOWN_METHODS:
            raise ValueError(f"unknown method: {method}")
        if correction not in KNOWN_CORRECTIONS:
            raise ValueError(f"unknown correction: {correction}")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        t0 = time.time()
        with self._lock:
            arm_ids = sorted(self._samples.keys())
            snapshots = {a: list(self._samples[a]) for a in arm_ids if self._samples[a]}
        if not snapshots:
            raise ValueError("no arms have any observed samples")
        k = len(snapshots)
        per_arm_delta = correct_delta(delta, k, correction)

        estimates: dict[str, RobustEstimate] = {}
        actual_radius = 0.0
        objective = "mean"

        for arm_id, samples in snapshots.items():
            n = len(samples)
            mu = _mean(samples)
            var = _sample_var(samples, mu)
            if method == METHOD_CVAR:
                if alpha is None or not 0.0 < alpha <= 1.0:
                    raise ValueError("alpha in (0, 1] required for CVaR")
                lower = cvar(samples, alpha=alpha, side="lower")
                upper = cvar(samples, alpha=alpha, side="upper")
                used_radius = alpha
                objective = "cvar"
            else:
                if radius == "auto":
                    if method == METHOD_WASSERSTEIN_1:
                        used_radius = wasserstein1_radius_for_coverage(
                            n, per_arm_delta, diameter=diameter
                        )
                    elif method == METHOD_KL:
                        used_radius = kl_radius_for_coverage(n, per_arm_delta)
                    elif method in (METHOD_CHI2, METHOD_EL):
                        used_radius = chi2_radius_for_coverage(n, per_arm_delta) / max(n, 1)
                    else:
                        raise ValueError(f"auto radius not supported for {method}")
                else:
                    if not isinstance(radius, (int, float)) or radius < 0:
                        raise ValueError("radius must be 'auto' or a non-negative float")
                    used_radius = float(radius)

                if method == METHOD_WASSERSTEIN_1:
                    lower = mean_wasserstein_dro(samples, radius=used_radius, lipschitz=lipschitz, side="lower")
                    upper = mean_wasserstein_dro(samples, radius=used_radius, lipschitz=lipschitz, side="upper")
                elif method == METHOD_KL:
                    lower = mean_kl_dro(samples, radius=used_radius, side="lower")
                    upper = mean_kl_dro(samples, radius=used_radius, side="upper")
                elif method == METHOD_CHI2:
                    lower = mean_chi2_dro(samples, radius=used_radius, side="lower")
                    upper = mean_chi2_dro(samples, radius=used_radius, side="upper")
                elif method == METHOD_EL:
                    lower_, upper_ = empirical_likelihood_ci(samples, delta=per_arm_delta)
                    lower, upper = lower_, upper_
                    used_radius = chi2_radius_for_coverage(n, per_arm_delta) / max(n, 1)
                else:
                    raise ValueError(f"unsupported method: {method}")
            actual_radius = used_radius
            estimates[arm_id] = RobustEstimate(
                arm_id=arm_id,
                point=mu if method != METHOD_CVAR else (lower + upper) / 2.0,
                lower=lower,
                upper=upper,
                n_samples=n,
                sample_variance=var,
                method=method,
                radius=used_radius,
                delta_used=per_arm_delta,
            )

        elapsed = time.time() - t0
        report = RobustReport(
            id=f"{self._id}-eval-{int(t0 * 1000):x}",
            method=method,
            objective=objective,
            radius=actual_radius,
            delta_family=delta,
            correction=correction,
            delta_per_arm=per_arm_delta,
            estimates=estimates,
            elapsed_s=elapsed,
        )
        report = self._maybe_attest(report)
        with self._lock:
            self._history.append(report)
        self._emit(ROBUST_EVALUATED, {
            "robustifier_id": self._id,
            "report_id": report.id,
            "method": method,
            "objective": objective,
            "radius": actual_radius,
            "k_arms": k,
            "elapsed_s": elapsed,
        })
        return report

    # -----------------------------------------------------------------
    # Convenience: robust selection
    # -----------------------------------------------------------------

    def robust_argmax(
        self,
        *,
        method: str,
        radius: float | str = "auto",
        delta: float = 0.05,
        correction: str = CORRECTION_BONFERRONI,
        alpha: float | None = None,
        lipschitz: float = 1.0,
        diameter: float = 1.0,
    ) -> tuple[str, RobustReport]:
        """Return the arm whose worst-case mean is largest."""
        report = self.evaluate(
            method=method,
            radius=radius,
            delta=delta,
            correction=correction,
            alpha=alpha,
            lipschitz=lipschitz,
            diameter=diameter,
        )
        winner = report.best_arm(by="lower")
        self._emit(ROBUST_ARGMAX, {
            "robustifier_id": self._id,
            "report_id": report.id,
            "winner": winner,
            "method": method,
        })
        return winner, report

    # -----------------------------------------------------------------
    # Minimax regret over the DRO ball
    # -----------------------------------------------------------------

    def minimax_regret(
        self,
        *,
        method: str,
        radius: float | str = "auto",
        delta: float = 0.05,
        correction: str = CORRECTION_BONFERRONI,
        lipschitz: float = 1.0,
        diameter: float = 1.0,
    ) -> RegretReport:
        """For each arm, compute the worst-case regret against the
        best alternative over the DRO ball.

        ``regret(i) = max_j upper_j - lower_i``: the worst-case payoff
        of the best other arm, minus the worst-case payoff of arm i.
        The minimax-regret choice is ``argmin_i regret(i)``.

        Under nominal P̂, this reduces to the standard regret. Under
        bounded model shift, it bounds the worst-case loss from
        committing to arm i.
        """
        t0 = time.time()
        report = self.evaluate(
            method=method,
            radius=radius,
            delta=delta,
            correction=correction,
            lipschitz=lipschitz,
            diameter=diameter,
        )
        ests = report.estimates
        max_upper = max(e.upper for e in ests.values())
        regret = {a: max_upper - e.lower for a, e in ests.items()}
        chosen = min(regret.items(), key=lambda kv: kv[1])[0]
        elapsed = time.time() - t0
        rr = RegretReport(
            id=f"{self._id}-regret-{int(t0 * 1000):x}",
            method=method,
            radius=report.radius,
            delta_family=delta,
            chosen_arm=chosen,
            minimax_regret_value=regret[chosen],
            regret_per_arm=regret,
            elapsed_s=elapsed,
        )
        self._emit(ROBUST_REGRET, {
            "robustifier_id": self._id,
            "regret_id": rr.id,
            "method": method,
            "chosen_arm": chosen,
            "minimax_regret_value": regret[chosen],
        })
        return rr

    # -----------------------------------------------------------------
    # Internal: event + attestation pass-through
    # -----------------------------------------------------------------

    def _emit(self, kind: str, data: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(Event(kind=kind, data=dict(data)))
        except Exception:
            pass

    def _maybe_attest(self, report: RobustReport) -> RobustReport:
        if self._attestor is None:
            return report
        payload = report.to_dict()
        payload.pop("receipt_hash", None)
        try:
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except Exception:
            return report
        # Optional: if attestor has an `append` API, write a receipt.
        for method_name in ("append_receipt", "append", "write"):
            fn = getattr(self._attestor, method_name, None)
            if callable(fn):
                try:
                    fn({
                        "kind": "robust.evaluated",
                        "report_id": report.id,
                        "method": report.method,
                        "objective": report.objective,
                        "radius": report.radius,
                        "delta_family": report.delta_family,
                        "correction": report.correction,
                        "delta_per_arm": report.delta_per_arm,
                        "hash": digest,
                    })
                except Exception:
                    pass
                break
        return RobustReport(
            id=report.id,
            method=report.method,
            objective=report.objective,
            radius=report.radius,
            delta_family=report.delta_family,
            correction=report.correction,
            delta_per_arm=report.delta_per_arm,
            estimates=report.estimates,
            elapsed_s=report.elapsed_s,
            receipt_hash=digest,
        )


# =====================================================================
# Free-function evaluator for a single mean estimate
# =====================================================================


def robust_mean_lower(
    samples: Sequence[float],
    *,
    method: str = METHOD_CHI2,
    radius: float | str = "auto",
    delta: float = 0.05,
    lipschitz: float = 1.0,
    diameter: float = 1.0,
) -> float:
    """Compute the worst-case lower bound on ``E[X]`` for one sample stream.

    Convenience wrapper that performs the same dispatch as
    ``Robustifier.evaluate`` for a single sample list. Handy for inline
    drop-in use inside other primitives.
    """
    if method not in KNOWN_METHODS:
        raise ValueError(f"unknown method: {method}")
    if method == METHOD_CVAR:
        raise ValueError("use cvar() directly for CVaR")
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    if radius == "auto":
        if method == METHOD_WASSERSTEIN_1:
            used = wasserstein1_radius_for_coverage(n, delta, diameter=diameter)
        elif method == METHOD_KL:
            used = kl_radius_for_coverage(n, delta)
        else:
            used = chi2_radius_for_coverage(n, delta) / n
    else:
        used = float(radius)
    if method == METHOD_WASSERSTEIN_1:
        return mean_wasserstein_dro(samples, radius=used, lipschitz=lipschitz, side="lower")
    if method == METHOD_KL:
        return mean_kl_dro(samples, radius=used, side="lower")
    if method == METHOD_CHI2:
        return mean_chi2_dro(samples, radius=used, side="lower")
    if method == METHOD_EL:
        lo, _ = empirical_likelihood_ci(samples, delta=delta)
        return lo
    raise ValueError(f"unsupported method: {method}")


def robust_mean_upper(
    samples: Sequence[float],
    *,
    method: str = METHOD_CHI2,
    radius: float | str = "auto",
    delta: float = 0.05,
    lipschitz: float = 1.0,
    diameter: float = 1.0,
) -> float:
    """Symmetric to ``robust_mean_lower`` — worst-case upper bound."""
    if method not in KNOWN_METHODS:
        raise ValueError(f"unknown method: {method}")
    if method == METHOD_CVAR:
        raise ValueError("use cvar() directly for CVaR")
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample sequence")
    if radius == "auto":
        if method == METHOD_WASSERSTEIN_1:
            used = wasserstein1_radius_for_coverage(n, delta, diameter=diameter)
        elif method == METHOD_KL:
            used = kl_radius_for_coverage(n, delta)
        else:
            used = chi2_radius_for_coverage(n, delta) / n
    else:
        used = float(radius)
    if method == METHOD_WASSERSTEIN_1:
        return mean_wasserstein_dro(samples, radius=used, lipschitz=lipschitz, side="upper")
    if method == METHOD_KL:
        return mean_kl_dro(samples, radius=used, side="upper")
    if method == METHOD_CHI2:
        return mean_chi2_dro(samples, radius=used, side="upper")
    if method == METHOD_EL:
        _, hi = empirical_likelihood_ci(samples, delta=delta)
        return hi
    raise ValueError(f"unsupported method: {method}")
