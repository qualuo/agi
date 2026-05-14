r"""Auditor — multiple-hypothesis testing with FDR / FWER control as a runtime primitive.

A coordination engine running thousands of simultaneous statistical tests
— drift detectors, A/B experiments, calibration audits, Arbiter K-arm
reports, ExperimentDesigner posteriors, Coalition Shapley CBs — faces a
problem every primitive in this stack ignores: **multiplicity**. With m
independent tests at level α, the probability of *at least one* false
discovery is ``1 - (1 - α)^m``, which already exceeds 50% for m ≥ 14 at
α = 0.05. The runtime cannot publish "drift detected on N tenants this
morning" without correcting for the fact that it ran N tests.

Deliberator (single hypothesis, anytime-valid) is the *sequential* dual.
Auditor is the *multiple* dual: take many test outcomes and decide which
are real with provable joint error guarantees.

The literature, condensed
-------------------------

**Family-wise error rate (FWER)** controls ``P(any false rejection)``.
Strong control: this holds under every configuration of nulls.

  * **Bonferroni, 1936 — Teoria statistica delle classi.** Reject if
    ``p_i ≤ α/m``. Trivially conservative under any dependence.

  * **Šidák, 1967 — Rectangular confidence regions for the means of
    multivariate normal distributions.** Reject if
    ``p_i ≤ 1 - (1-α)^{1/m}``. Exact under independence; valid under
    positive orthant dependence; strictly tighter than Bonferroni.

  * **Holm, 1979 — A simple sequentially rejective multiple test
    procedure.** Sort p-values. Reject the i-th smallest if all
    smaller p-values are rejected and ``p_(i) ≤ α/(m-i+1)``. Strong
    FWER control under arbitrary dependence; uniformly more powerful
    than Bonferroni.

  * **Hochberg, 1988 — A sharper Bonferroni procedure for multiple
    tests of significance.** A step-up version of Holm, slightly more
    powerful, valid under PRDS (positive regression dependence).

**False discovery rate (FDR)** controls
``E[#false rejections / max(#rejections, 1)]`` — the *expected fraction*
of rejections that are wrong. Vastly more powerful than FWER when many
real signals exist. The modern multiple-testing workhorse.

  * **Benjamini & Hochberg, 1995 — Controlling the false discovery
    rate: a practical and powerful approach to multiple testing.**
    Sort. Find largest ``k`` with ``p_(k) ≤ k·α/m``. Reject all
    ``H_(1),...,H_(k)``. Controls FDR ≤ ``π_0·α`` under independence
    (Benjamini-Yekutieli 2001 extends to PRDS).

  * **Benjamini & Yekutieli, 2001 — The control of the false discovery
    rate in multiple testing under dependency.** Replace threshold by
    ``i·α/(m·L_m)`` where ``L_m = Σ_{k=1}^m 1/k`` is the m-th harmonic
    number. Controls FDR under *arbitrary* dependence.

  * **Storey, 2002 — A direct approach to false discovery rates.**
    Estimate ``π_0 = #{p_i > λ} / (m(1-λ))`` and run BH at adjusted
    level ``α/π̂_0``. Uniformly tighter than BH when π_0 < 1.

  * **Storey, Taylor, Siegmund, 2004 — Strong control, conservative
    point estimation and simultaneous conservative consistency.**
    Formal proof that ``q(p_(i)) = min_{j≥i} m·π̂_0·p_(j)/j`` controls
    FDR pointwise.

**Online FDR** — tests arrive one at a time, decide each *before*
seeing future p-values. Foundational for streaming runtimes.

  * **Foster & Stine, 2008 — α-investing rules for streamwise false
    discovery rate control.** Maintain a "wealth" budget; spending it
    on tests, earning it back on rejections. Controls a variant called
    mFDR = E[V]/E[R+η].

  * **Aharoni & Rosset, 2014 — Generalised α-investing: definitions,
    optimality results and application to public databases.** Wider
    family of α-investing rules with anytime FDR-like guarantees.

  * **Javanmard & Montanari, 2018 — Online rules for control of false
    discovery rate and false discovery exceedance.** Introduce LORD
    (Levels based on Recent Discovery). Strong online FDR control.

  * **Ramdas, Yang, Wainwright, Jordan, 2017 — Online control of the
    false discovery rate with decaying memory.** LORD-3, with weights
    ``γ_k = c/(k·(log max(k,2))^2)``, decaying memory that retains
    asymptotic power.

  * **Ramdas, Zrnic, Wainwright, Jordan, 2018 — SAFFRON: an
    adaptive algorithm for online control of the false discovery
    rate.** SAFFRON uses a candidate threshold λ ∈ (0,1); only
    "interesting" tests (p ≤ λ) drain wealth. Strictly more powerful
    than LORD-3 when most nulls have p ≫ 0.

  * **Tian & Ramdas, 2019 — ADDIS: an adaptive discarding algorithm
    for online FDR control with conservative nulls.** Discards tests
    with p > τ entirely; refunds wealth. Best-in-class power for
    online FDR.

**E-values and post-hoc valid inference.** The newest frontier —
hypothesis tests that compose without paying the multiplicity tax.

  * **Vovk & Wang, 2021 — E-values: calibration, combination, and
    applications.** E-values e ≥ 0 with ``E_{H_0}[e] ≤ 1``. Their
    multiplicative combination yields valid e-values under any
    dependence. The Bayes-factor analogue of the p-value.

  * **Wang & Ramdas, 2022 — False discovery rate control with
    e-values.** **e-BH**: sort e-values descending; reject the top-k
    where ``k·e_(k)/m ≥ 1/α``. Controls FDR ≤ α under *arbitrary*
    dependence — strictly more general than Benjamini-Yekutieli.

  * **Ramdas, Grünwald, Vovk, Shafer, 2023 — Game-theoretic
    statistics and safe anytime-valid inference.** The unifying
    framework: e-values, p-values, and anytime-valid CIs as a
    coherent betting calculus.

**Global combiners** — single combined p-value from m local tests,
useful for meta-analysis.

  * **Fisher, 1925 — Statistical methods for research workers.**
    ``-2 Σ log p_i ~ χ²_{2m}`` under independence and H₀.

  * **Stouffer, 1949 — The American Soldier.** Combine via inverse
    normal: ``Z = (Σ w_i Φ^{-1}(1-p_i)) / √(Σ w_i²)``.

  * **Simes, 1986 — An improved Bonferroni procedure for multiple
    tests of significance.** ``p_Simes = min_i m·p_(i)/i``. The
    threshold that BH is named after.

  * **Wilson, 2019 — The harmonic mean p-value for combining
    dependent tests.** HMP works under arbitrary dependence among
    tests, unlike Fisher and Stouffer.

What Auditor provides
---------------------

A *single* class wires every procedure above to one consistent surface:

::

    auditor = Auditor(bus=bus, attestor=attestor)
    for test_id, p in stream_of_results:
        auditor.observe(test_id, p_value=p)

    # Batch (offline) decision after seeing all m tests
    report = auditor.decide(method="bh", alpha=0.05)
    report = auditor.decide(method="by", alpha=0.05)              # PRDS-free
    report = auditor.decide(method="holm", alpha=0.05)            # FWER
    report = auditor.decide(method="storey", alpha=0.05)          # adaptive BH

    # E-value version (arbitrary dependence)
    auditor.observe(test_id, e_value=ev)
    report = auditor.decide(method="ebh", alpha=0.05)

    # Online — decision per arrival
    is_reject = auditor.decide_online(test_id, p_value=p, method="lord", alpha=0.05)

    # Combiners
    pf = auditor.combine(method="fisher")
    ps = auditor.combine(method="stouffer", weights=...)
    pa = auditor.combine(method="harmonic")

Where this slots in
-------------------

  * **DriftSentinel.** Tenant-level drift tests produce a p-value per
    tenant per epoch. With 10k tenants, BH at α=0.05 ensures at most
    5% of declared "drifting" tenants are false alarms — the floor of
    a usable multi-tenant drift dashboard.

  * **ExperimentRunner.** Many concurrent A/B tests with shared
    infrastructure. Each test emits a p-value at completion; LORD
    controls FDR over the whole stream so business owners can trust
    the "winners" without preregistering a single test.

  * **Arbiter.** When the same arms are evaluated by multiple
    Arbiters (one per region, per tenant), Holm controls FWER over the
    joint family. With e-values, e-BH composes the per-Arbiter
    confidence reports under arbitrary dependence.

  * **CausalLab / CausalDiscoverer.** A causal discovery sweep tests
    edge presence for every variable pair (m = O(p²) tests). BH or
    BY controls the *edge* FDR, which directly trades off graph
    sparsity against missed-edge probability.

  * **ExperimentDesigner.** Bayesian posterior tail probabilities
    feed Auditor as e-values; combining many small experiments via
    e-BH gives an honest joint guarantee at the run level.

  * **Strategist.** Risk-adjusted EV across K tickets ranks the
    candidates; Auditor decides *which* of the K candidates are
    statistically distinct from the baseline at family-wise level α.

  * **AttestationLedger.** Every batch decision emits an audit
    receipt: the test ids, the method, α, the rejection set, and a
    SHA-256 of the report. Third-party-replayable proof of who got
    flagged on what evidence.

Events
------

::

    audit.started               Auditor was constructed
    audit.observed              a test result was logged
    audit.decided               a batch decision was rendered
    audit.online_decided        an online decision was rendered
    audit.budget_updated        online wealth was updated
    audit.combined              a global p-value was combined
    audit.test_cleared          a test's evidence was reset
    audit.cleared               all state was reset

Honest about limits
-------------------

  * **BH and Storey** assume independence or PRDS among test
    statistics. For arbitrary dependence use **BY** (loss of a
    log m factor of power) or **e-BH** (no loss when e-values are
    available).

  * **Online procedures (LORD, SAFFRON, ADDIS)** control *anytime*
    FDR if the user picks the weight sequence γ correctly. We default
    to γ_k ∝ 1/(k · log²(max(k,2))), the LORD-3 recommendation.

  * **Storey's π̂_0** is unbiased when ``λ`` is large enough that
    the conditional distribution of p-values above λ is uniform under
    H₀ for the truly null hypotheses. We default to λ = 0.5; the
    estimator is clipped to (0, 1].

  * **HMP combiner** is valid up to a Landau correction; we use the
    asymptotic correction ``A_k ≈ exp(γ + log(log k))`` (γ = Euler
    constant) which is accurate for k ≥ 10. For k < 10, we fall back
    to the Bonferroni-conservative ``HMP·k``.

  * **E-values** require the user to supply a valid e-statistic; we
    do not verify ``E_{H_0}[e] ≤ 1``. Garbage in, garbage out.

Citations
---------

* Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo
  delle probabilità. *Pubblicazioni del R Istituto Superiore di Scienze
  Economiche e Commerciali di Firenze*, 8, 3-62.
* Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
  Oliver and Boyd.
* Šidák, Z. (1967). Rectangular confidence regions for the means of
  multivariate normal distributions. *JASA*, 62, 626-633.
* Holm, S. (1979). A simple sequentially rejective multiple test
  procedure. *Scandinavian Journal of Statistics*, 6, 65-70.
* Simes, R. J. (1986). An improved Bonferroni procedure for multiple
  tests of significance. *Biometrika*, 73, 751-754.
* Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple
  tests of significance. *Biometrika*, 75, 800-802.
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
  rate. *JRSS-B*, 57, 289-300.
* Benjamini, Y. & Yekutieli, D. (2001). The control of the false
  discovery rate in multiple testing under dependency. *Annals of
  Statistics*, 29(4), 1165-1188.
* Storey, J. D. (2002). A direct approach to false discovery rates.
  *JRSS-B*, 64(3), 479-498.
* Storey, J. D. & Tibshirani, R. (2003). Statistical significance for
  genomewide studies. *PNAS*, 100, 9440-9445.
* Storey, J. D., Taylor, J. E., Siegmund, D. (2004). Strong control,
  conservative point estimation and simultaneous conservative
  consistency of false discovery rates. *JRSS-B*, 66(1), 187-205.
* Foster, D. P. & Stine, R. A. (2008). α-investing: a procedure for
  sequential control of expected false discoveries. *JRSS-B*, 70(2),
  429-444.
* Aharoni, E. & Rosset, S. (2014). Generalised α-investing.
  *JRSS-B*, 76(4), 771-794.
* Javanmard, A. & Montanari, A. (2018). Online rules for control of
  false discovery rate and false discovery exceedance. *Annals of
  Statistics*, 46(2), 526-554.
* Ramdas, A., Yang, F., Wainwright, M. J., Jordan, M. I. (2017).
  Online control of the false discovery rate with decaying memory.
  *NeurIPS*.
* Ramdas, A., Zrnic, T., Wainwright, M. J., Jordan, M. I. (2018).
  SAFFRON: an adaptive algorithm for online control of the false
  discovery rate. *ICML*.
* Tian, J. & Ramdas, A. (2019). ADDIS: an adaptive discarding
  algorithm for online FDR control with conservative nulls. *NeurIPS*.
* Wilson, D. J. (2019). The harmonic mean p-value for combining
  dependent tests. *PNAS*, 116(4), 1195-1200.
* Vovk, V. & Wang, R. (2021). E-values: calibration, combination, and
  applications. *Annals of Statistics*, 49(3), 1736-1754.
* Wang, R. & Ramdas, A. (2022). False discovery rate control with
  e-values. *JRSS-B*, 84(3), 822-852.
* Ramdas, A., Grünwald, P., Vovk, V., Shafer, G. (2023).
  Game-theoretic statistics and safe anytime-valid inference.
  *Statistical Science*, 38(4), 576-601.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds
# =====================================================================

AUDIT_STARTED = "audit.started"
AUDIT_OBSERVED = "audit.observed"
AUDIT_DECIDED = "audit.decided"
AUDIT_ONLINE_DECIDED = "audit.online_decided"
AUDIT_BUDGET_UPDATED = "audit.budget_updated"
AUDIT_COMBINED = "audit.combined"
AUDIT_TEST_CLEARED = "audit.test_cleared"
AUDIT_CLEARED = "audit.cleared"


# =====================================================================
# Methods
# =====================================================================

# Offline / batch procedures
METHOD_BH = "bh"  # Benjamini-Hochberg 1995 (FDR, independence/PRDS)
METHOD_BY = "by"  # Benjamini-Yekutieli 2001 (FDR, arbitrary dependence)
METHOD_HOLM = "holm"  # Holm 1979 (FWER, any dependence)
METHOD_HOCHBERG = "hochberg"  # Hochberg 1988 (FWER, PRDS)
METHOD_BONFERRONI = "bonferroni"  # 1936 (FWER, any dependence)
METHOD_SIDAK = "sidak"  # 1967 (FWER, independence)
METHOD_STOREY = "storey"  # Storey 2002 (adaptive FDR)
METHOD_EBH = "ebh"  # Wang-Ramdas 2022 (FDR with e-values)

# Online procedures
METHOD_LORD = "lord"  # LORD-3 (online FDR)
METHOD_SAFFRON = "saffron"  # adaptive online FDR
METHOD_ADDIS = "addis"  # ADDIS (adaptive + discarding)
METHOD_ALPHA_INVEST = "alpha_invest"  # Foster-Stine alpha-investing (mFDR)

# Combiners
COMBINE_FISHER = "fisher"
COMBINE_STOUFFER = "stouffer"
COMBINE_SIMES = "simes"
COMBINE_HARMONIC = "harmonic"
COMBINE_BONFERRONI = "bonferroni"

KNOWN_OFFLINE_METHODS = (
    METHOD_BH,
    METHOD_BY,
    METHOD_HOLM,
    METHOD_HOCHBERG,
    METHOD_BONFERRONI,
    METHOD_SIDAK,
    METHOD_STOREY,
    METHOD_EBH,
)

KNOWN_ONLINE_METHODS = (
    METHOD_LORD,
    METHOD_SAFFRON,
    METHOD_ADDIS,
    METHOD_ALPHA_INVEST,
)

KNOWN_COMBINERS = (
    COMBINE_FISHER,
    COMBINE_STOUFFER,
    COMBINE_SIMES,
    COMBINE_HARMONIC,
    COMBINE_BONFERRONI,
)

KNOWN_METHODS = KNOWN_OFFLINE_METHODS + KNOWN_ONLINE_METHODS


# =====================================================================
# Constants
# =====================================================================

_EPS = 1e-15
_EULER_GAMMA = 0.5772156649015329
_NORMAL_QUANTILE_TOL = 1e-12


# =====================================================================
# Utility math
# =====================================================================


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _validate_p(p: float) -> None:
    if not math.isfinite(p):
        raise ValueError("p-value must be finite")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p-value must be in [0, 1]")


def _validate_e(e: float) -> None:
    if not math.isfinite(e):
        raise ValueError("e-value must be finite")
    if e < 0:
        raise ValueError("e-value must be non-negative")


def _normal_quantile(p: float) -> float:
    """Beasley-Springer-Moro Φ⁻¹(p). ~1e-7 accuracy on the central 99%."""
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
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _normal_cdf(z: float) -> float:
    """Φ(z) via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _chi2_sf_2m(t: float, m: int) -> float:
    """Survival function of χ²_{2m} at point t.

    Closed form: ``P(χ²_{2m} > t) = e^{-t/2} · Σ_{k=0}^{m-1} (t/2)^k / k!``.

    Used by Fisher's combiner: ``-2 Σ log p_i ~ χ²_{2m}`` under H₀ and
    independence.
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    if m <= 0:
        raise ValueError("m must be positive")
    half = 0.5 * t
    # Sum the truncated Poisson series.
    s = 0.0
    log_term = -half  # log of e^{-half}
    log_fact = 0.0
    for k in range(m):
        if k > 0:
            log_fact += math.log(k)
        log_pk = k * math.log(half) - log_fact + log_term if half > 0 else (log_term if k == 0 else -math.inf)
        if log_pk == -math.inf:
            continue
        s += math.exp(log_pk)
    return min(1.0, max(0.0, s))


# =====================================================================
# Section 1 — Offline FDR procedures
# =====================================================================


def bh_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg (1995) step-up FDR control.

    Sort p-values ascending. Find the largest ``k`` such that
    ``p_(k) ≤ k·α/m``. Reject the ``k`` smallest hypotheses.

    FDR ≤ ``π_0 · α`` under independence or PRDS.
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    k_max = 0
    for rank, (_idx, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            k_max = rank
    reject = [False] * m
    for rank, (idx, _p) in enumerate(indexed, start=1):
        if rank <= k_max:
            reject[idx] = True
    return reject


def by_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Benjamini-Yekutieli (2001) — BH with a harmonic-number correction.

    Replaces threshold ``i·α/m`` with ``i·α/(m·L_m)`` where
    ``L_m = Σ_{k=1}^m 1/k``. Controls FDR ≤ α under *arbitrary*
    dependence (the price is a factor of ``log m`` in power).
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    L_m = sum(1.0 / k for k in range(1, m + 1))
    return bh_rejections(p_values, alpha / L_m)


def holm_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Holm (1979) step-down FWER control.

    Sort ascending. Walk up: at rank ``i``, reject iff
    ``p_(i) ≤ α/(m - i + 1)`` *and* all smaller p-values were rejected.
    First failure stops the procedure. Strong FWER control under any
    dependence.
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    reject = [False] * m
    for rank, (idx, p) in enumerate(indexed, start=1):
        if p <= alpha / (m - rank + 1):
            reject[idx] = True
        else:
            break
    return reject


def hochberg_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Hochberg (1988) step-up FWER control.

    Sort ascending. Walk *down*: largest ``i`` with
    ``p_(i) ≤ α/(m - i + 1)`` triggers rejection of all smaller ranks.
    Valid under PRDS; uniformly more powerful than Holm under PRDS.
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    k_max = 0
    for rank, (_idx, p) in enumerate(indexed, start=1):
        if p <= alpha / (m - rank + 1):
            k_max = rank
    reject = [False] * m
    for rank, (idx, _p) in enumerate(indexed, start=1):
        if rank <= k_max:
            reject[idx] = True
    return reject


def bonferroni_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Bonferroni (1936): reject iff ``p_i ≤ α/m``. FWER ≤ α."""
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    thr = alpha / m
    return [p <= thr for p in p_values]


def sidak_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    """Šidák (1967): reject iff ``p_i ≤ 1 - (1-α)^{1/m}``.

    Exact FWER = α under independence; valid under positive orthant
    dependence. Strictly tighter than Bonferroni for m ≥ 2.
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    thr = 1.0 - (1.0 - alpha) ** (1.0 / m)
    return [p <= thr for p in p_values]


def storey_pi0(p_values: Sequence[float], lam: float = 0.5) -> float:
    """Storey (2002) estimator of the null proportion π_0.

    π̂_0(λ) = #{p_i > λ} / (m(1-λ)). Clipped to (0, 1].

    Intuition: under H₀, p-values are Uniform(0,1); the right tail
    ``[λ, 1]`` should contain ``π_0 · m · (1-λ)`` nulls plus a small
    contribution from alternatives. Inverting gives the estimator.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    m = len(p_values)
    if m == 0:
        return 1.0
    for p in p_values:
        _validate_p(p)
    above = sum(1 for p in p_values if p > lam)
    pi0 = above / (m * (1.0 - lam))
    return min(1.0, max(1.0 / m, pi0))


def storey_rejections(
    p_values: Sequence[float], alpha: float, lam: float = 0.5
) -> list[bool]:
    """Storey (2002) adaptive BH — run BH at level ``α/π̂_0(λ)``.

    Asymptotically tighter than BH when π_0 < 1. Reduces to BH when
    π̂_0 = 1.
    """
    _validate_alpha(alpha)
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    pi0 = storey_pi0(p_values, lam)
    return bh_rejections(p_values, min(alpha / pi0, 1.0 - _EPS))


def q_values(p_values: Sequence[float], pi0: float | None = None) -> list[float]:
    """Storey-Tibshirani (2003) q-values.

    ``q(p_(i)) = min_{j ≥ i} m·π̂_0·p_(j) / j`` (the *pointwise* FDR
    estimate at the rejection threshold p_(i)). Returns one q-value per
    input p-value in input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    pi0 = storey_pi0(p_values) if pi0 is None else pi0
    if not 0.0 < pi0 <= 1.0:
        raise ValueError("pi0 must be in (0, 1]")
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    sorted_p = [kv[1] for kv in indexed]
    sorted_q = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q_i = pi0 * m * sorted_p[i] / rank
        running_min = min(running_min, q_i)
        sorted_q[i] = running_min
    out = [0.0] * m
    for sorted_idx, (orig_idx, _p) in enumerate(indexed):
        out[orig_idx] = sorted_q[sorted_idx]
    return out


# =====================================================================
# Section 2 — E-value FDR (Wang-Ramdas 2022)
# =====================================================================


def e_value_bh_rejections(e_values: Sequence[float], alpha: float) -> list[bool]:
    """e-BH (Wang-Ramdas 2022): FDR ≤ α under *arbitrary* dependence.

    Sort e-values *descending*: ``e_(1) ≥ e_(2) ≥ ... ≥ e_(m)``.
    Find the largest ``k`` such that ``k · e_(k) / m ≥ 1/α``.
    Reject the top-k tests.

    Equivalently: reject tests with ``e_i ≥ m / (k·α)`` for the
    computed k. Validity requires only ``E_{H_0}[e] ≤ 1``.
    """
    _validate_alpha(alpha)
    m = len(e_values)
    if m == 0:
        return []
    for e in e_values:
        _validate_e(e)
    indexed = sorted(enumerate(e_values), key=lambda kv: kv[1], reverse=True)
    k_max = 0
    for rank, (_idx, e) in enumerate(indexed, start=1):
        if rank * e / m >= 1.0 / alpha:
            k_max = rank
    reject = [False] * m
    for rank, (idx, _e) in enumerate(indexed, start=1):
        if rank <= k_max:
            reject[idx] = True
    return reject


# =====================================================================
# Section 3 — Online FDR procedures
# =====================================================================


_BASEL = math.pi * math.pi / 6.0  # Σ_{k=1}^∞ 1/k² = π²/6


def _gamma_normalised(k: int) -> float:
    """Normalised inverse-square weight: γ_k = 6/(π²·k²).

    Σ_{k=1}^∞ γ_k = 1 (Basel problem), γ_k non-increasing in k,
    γ_1 = 6/π² ≈ 0.608. This is the canonical LORD weight family
    that makes ``α_t ≤ α`` hold *for every t and every history* and
    therefore guarantees the FDR ≤ α bound stated in
    Ramdas-Yang-Wainwright-Jordan 2017.
    """
    if k <= 0:
        return 0.0
    return 1.0 / (k * k * _BASEL)


def lord_weights(t: int, *, kind: str = "lord3") -> list[float]:
    """Default normalised weight sequence γ_k for LORD / SAFFRON / ADDIS.

    γ_k = 6/(π²·k²) so Σ_{k≥1} γ_k = 1 (Basel sum). γ_k is positive
    and non-increasing. This is the cleanest defensible choice; it
    makes the per-step level

        α_t = γ_t · W_0 + (α - W_0) · Σ_{j∈R(t-1)} γ_{t-τ_j}

    bounded by α for every history (Ramdas-Yang-Wainwright-Jordan 2017,
    Lemma 2). The ``kind`` argument is retained for backward compat
    but currently selects the same family.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if kind not in ("lord3", "saffron"):
        raise ValueError("kind must be 'lord3' or 'saffron'")
    return [_gamma_normalised(k) for k in range(1, t + 1)]


def lord_online_decisions(
    p_values: Sequence[float],
    alpha: float,
    *,
    w0: float | None = None,
    gamma_kind: str = "lord3",
) -> list[bool]:
    """LORD-3 online FDR (Ramdas-Yang-Wainwright-Jordan 2017).

    At step ``t`` test ``t`` is rejected iff ``p_t ≤ α_t`` where

        α_t = γ_t · W_0 + (α - W_0) · Σ_{j ∈ R(t-1)} γ_{t - τ_j}

    with ``γ`` a non-negative non-increasing sequence summing to ≤ 1,
    ``W_0`` initial wealth, ``τ_j`` the j-th rejection time.

    Strong FDR control under independence (and certain dependency
    conditions in subsequent work). Power retained asymptotically.
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    w0_use = 0.5 * alpha if w0 is None else float(w0)
    if not 0.0 <= w0_use < alpha:
        raise ValueError("W_0 must be in [0, alpha)")
    rejection_times: list[int] = []
    decisions: list[bool] = []
    for t_idx, p in enumerate(p_values, start=1):
        alpha_t = _gamma_normalised(t_idx) * w0_use
        for tau in rejection_times:
            k = t_idx - tau
            if k >= 1:
                alpha_t += (alpha - w0_use) * _gamma_normalised(k)
        alpha_t = min(alpha_t, alpha)
        is_reject = p <= alpha_t
        decisions.append(is_reject)
        if is_reject:
            rejection_times.append(t_idx)
    return decisions


def saffron_online_decisions(
    p_values: Sequence[float],
    alpha: float,
    *,
    w0: float | None = None,
    lam: float = 0.5,
) -> list[bool]:
    """SAFFRON (Ramdas et al. 2018) adaptive online FDR.

    Uses a candidate threshold λ ∈ (0,1). "Interesting" tests (p ≤ λ)
    are *candidates*; only candidates spend wealth. At step ``t``:

        α_t = min(λ, (1 - λ) · [ γ_{t-C(t-1)} · W_0
            + (α - W_0)·Σ_{j∈R(t-1), j≥τ_1} γ_{t-τ_j-C(t-1,τ_j)} + ...])

    where ``C(t)`` counts candidates by time t. We use a simplified
    correct implementation: per-step α_t shrinks linearly with the
    candidate count.

    Strong FDR control; uniformly more powerful than LORD-3 when most
    nulls have p ≫ λ.
    """
    _validate_alpha(alpha)
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    w0_use = 0.5 * alpha if w0 is None else float(w0)
    if not 0.0 <= w0_use < alpha:
        raise ValueError("W_0 must be in [0, alpha)")
    candidates_until: list[int] = []  # times where p_t <= lam
    rejection_times: list[int] = []
    decisions: list[bool] = []
    for t_idx, p in enumerate(p_values, start=1):
        c_t_minus_1 = len(candidates_until)
        idx0 = t_idx - c_t_minus_1
        alpha_t = (1.0 - lam) * _gamma_normalised(idx0) * w0_use if idx0 >= 1 else 0.0
        for j, tau in enumerate(rejection_times):
            c_btw = sum(1 for c in candidates_until if c > tau)
            kk = t_idx - tau - c_btw
            if kk >= 1:
                alpha_t += (1.0 - lam) * (alpha - w0_use) * _gamma_normalised(kk)
        alpha_t = min(alpha_t, lam, alpha)
        is_reject = p <= alpha_t
        decisions.append(is_reject)
        if p <= lam:
            candidates_until.append(t_idx)
        if is_reject:
            rejection_times.append(t_idx)
    return decisions


def addis_online_decisions(
    p_values: Sequence[float],
    alpha: float,
    *,
    w0: float | None = None,
    lam: float = 0.5,
    tau: float = 0.5,
) -> list[bool]:
    """ADDIS (Tian-Ramdas 2019) — adaptive + discarding.

    Builds on SAFFRON: tests with ``p > τ`` are *discarded*, refunding
    wealth, while ``p ≤ λ`` are *candidates*. We require ``λ ≤ τ``.
    Best-in-class power when most nulls have ``p`` close to 1.
    """
    _validate_alpha(alpha)
    if not 0.0 < lam < 1.0 or not 0.0 < tau < 1.0:
        raise ValueError("lam, tau must be in (0, 1)")
    if lam > tau:
        raise ValueError("require lam <= tau")
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    w0_use = 0.5 * alpha if w0 is None else float(w0)
    if not 0.0 <= w0_use < alpha:
        raise ValueError("W_0 must be in [0, alpha)")
    candidates: list[int] = []
    discarded: set[int] = set()
    rejection_times: list[int] = []
    decisions: list[bool] = []
    for t_idx, p in enumerate(p_values, start=1):
        considered = t_idx - len(discarded)
        candidate_count = len(candidates)
        idx0 = considered - candidate_count
        alpha_t = (tau - lam) * _gamma_normalised(idx0) * w0_use if idx0 >= 1 else 0.0
        for tau_j in rejection_times:
            cands_after = sum(1 for c in candidates if c > tau_j)
            disc_after = sum(1 for d in discarded if d > tau_j)
            kk = (t_idx - tau_j) - cands_after - disc_after
            if kk >= 1:
                alpha_t += (tau - lam) * (alpha - w0_use) * _gamma_normalised(kk)
        alpha_t = min(alpha_t, lam, alpha)
        is_reject = p <= alpha_t
        decisions.append(is_reject)
        if p > tau:
            discarded.add(t_idx)
        elif p <= lam:
            candidates.append(t_idx)
        if is_reject:
            rejection_times.append(t_idx)
    return decisions


def alpha_invest_online_decisions(
    p_values: Sequence[float],
    alpha: float,
    *,
    w0: float | None = None,
    payout: float | None = None,
) -> list[bool]:
    """Alpha-investing (Foster-Stine 2008) — controls mFDR.

    Maintain "wealth" ``W``. At step ``t``, spend ``α_t = W/(1+W)`` (a
    rule that admits arbitrary positive α_t ≤ W as well; we use the
    optimal-for-many-nulls choice). On reject: ``W += ω - α_t/(1-α_t)``.
    On accept: ``W -= α_t/(1-α_t)``.

    Controls ``mFDR = E[V]/E[R+1/(1-α)]`` ≤ α.
    """
    _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        _validate_p(p)
    w0_use = 0.5 * alpha if w0 is None else float(w0)
    payout_use = alpha if payout is None else float(payout)
    if w0_use < 0 or w0_use >= 1.0:
        raise ValueError("W_0 must be in [0, 1)")
    if payout_use <= 0 or payout_use >= 1.0:
        raise ValueError("payout must be in (0, 1)")
    W = w0_use
    decisions: list[bool] = []
    for p in p_values:
        alpha_t = min(W / (1.0 + W), 1.0 - _EPS)
        if alpha_t <= 0:
            decisions.append(False)
            continue
        is_reject = p <= alpha_t
        decisions.append(is_reject)
        cost = alpha_t / max(1.0 - alpha_t, _EPS)
        if is_reject:
            W += payout_use - cost
        else:
            W -= cost
        if W < 0:
            W = 0.0
    return decisions


# =====================================================================
# Section 4 — Global combiners
# =====================================================================


def combine_fisher(p_values: Sequence[float]) -> float:
    """Fisher (1925) combined p-value: ``P(χ²_{2m} > -2 Σ log p_i)``.

    Valid under independence and H₀. Highly sensitive to ``min(p_i)``
    when one or two small p-values dominate the sum.
    """
    m = len(p_values)
    if m == 0:
        raise ValueError("empty p-value sequence")
    for p in p_values:
        _validate_p(p)
    stat = -2.0 * sum(math.log(max(p, _EPS)) for p in p_values)
    return _chi2_sf_2m(stat, m)


def combine_stouffer(
    p_values: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    """Stouffer (1949) inverse-normal combiner.

    ``Z = (Σ w_i · Φ^{-1}(1 - p_i)) / √(Σ w_i²)``, combined p =
    1 - Φ(Z). Equal weights → square root scaling.
    """
    m = len(p_values)
    if m == 0:
        raise ValueError("empty p-value sequence")
    for p in p_values:
        _validate_p(p)
    if weights is None:
        weights = [1.0] * m
    if len(weights) != m:
        raise ValueError("weights and p_values must have same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("weights must sum to positive")
    z = sum(
        w * _normal_quantile(min(1.0 - _EPS, max(_EPS, 1.0 - p)))
        for w, p in zip(weights, p_values)
    )
    z /= math.sqrt(sum(w * w for w in weights))
    return 1.0 - _normal_cdf(z)


def combine_simes(p_values: Sequence[float]) -> float:
    """Simes (1986) combiner: ``min_i m·p_(i)/i``.

    Valid under independence and PRDS. The exact analogue of BH at
    the global-null hypothesis. Always ≤ Bonferroni global p.
    """
    m = len(p_values)
    if m == 0:
        raise ValueError("empty p-value sequence")
    for p in p_values:
        _validate_p(p)
    sorted_p = sorted(p_values)
    return min(1.0, min(m * sorted_p[i] / (i + 1) for i in range(m)))


def combine_harmonic(p_values: Sequence[float]) -> float:
    """Harmonic mean p-value (Wilson 2019) — robust to dependence.

    HMP = ``m / Σ_i 1/p_i`` (with equal weights). The combined
    p-value is ``HMP · A_m`` where ``A_m = exp(γ + log log m)`` is the
    asymptotic Landau correction (γ = Euler-Mascheroni constant).

    For m < 10 we fall back to the Bonferroni-conservative
    ``HMP · m`` to guarantee validity at small m.
    """
    m = len(p_values)
    if m == 0:
        raise ValueError("empty p-value sequence")
    for p in p_values:
        _validate_p(p)
    safe_p = [max(p, _EPS) for p in p_values]
    hmp = m / sum(1.0 / p for p in safe_p)
    if m < 10:
        return min(1.0, hmp * m)
    a_m = math.exp(_EULER_GAMMA + math.log(math.log(m)))
    return min(1.0, hmp * a_m)


def combine_bonferroni(p_values: Sequence[float]) -> float:
    """Bonferroni global p: ``min(1, m·min_i p_i)``."""
    m = len(p_values)
    if m == 0:
        raise ValueError("empty p-value sequence")
    for p in p_values:
        _validate_p(p)
    return min(1.0, m * min(p_values))


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class TestRecord:
    """One observed test.

    Either ``p_value`` *or* ``e_value`` is set (the other is ``None``).
    ``metadata`` is opaque user data passed through to receipts.
    """

    __test__ = False  # pytest: not a test class

    test_id: str
    p_value: float | None
    e_value: float | None
    ts: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditDecision:
    """Per-test decision inside an AuditReport."""

    test_id: str
    rejected: bool
    p_value: float | None
    e_value: float | None
    q_value: float | None
    rank: int  # 1-based rank by ascending p (or descending e)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReport:
    """The result of one batch ``Auditor.decide(...)`` call."""

    id: str
    method: str
    alpha: float
    n_tests: int
    n_rejected: int
    pi0_estimate: float | None
    decisions: dict[str, AuditDecision]
    elapsed_s: float
    receipt_hash: str = ""

    def rejected_ids(self) -> tuple[str, ...]:
        return tuple(sorted(t for t, d in self.decisions.items() if d.rejected))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "n_rejected": self.n_rejected,
            "pi0_estimate": self.pi0_estimate,
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "elapsed_s": self.elapsed_s,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True)
class CombinedReport:
    """Result of one ``Auditor.combine(...)`` call."""

    id: str
    method: str
    n_tests: int
    combined_p: float
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Auditor — stateful primitive
# =====================================================================


class Auditor:
    """Stateful multi-hypothesis controller.

    Thread-safe. Designed to be wired into the runtime's event bus so
    that every drift / experiment / arbiter / calibration test can
    write a single ``observe(test_id, ...)`` call and have FDR / FWER
    controlled across the whole multiverse of simultaneous decisions.

    Online state (LORD / SAFFRON / ADDIS / α-invest) is maintained
    incrementally; offline state (BH / BY / Holm / Storey / e-BH) is
    computed fresh on every call to ``decide()``.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        attestor: Any | None = None,
        auditor_id: str | None = None,
    ) -> None:
        self._bus = bus
        self._attestor = attestor
        self._id = auditor_id or f"aud-{int(time.time() * 1000):x}"
        self._lock = threading.RLock()
        # Observed tests keyed by id; order of first observation preserved.
        self._tests: dict[str, TestRecord] = {}
        self._order: list[str] = []
        # Per-method online state.
        self._online_state: dict[str, dict[str, Any]] = {}
        self._reports: list[AuditReport] = []
        self._emit(AUDIT_STARTED, {"auditor_id": self._id})

    # -----------------------------------------------------------------
    # Sample ingestion
    # -----------------------------------------------------------------

    def observe(
        self,
        test_id: str,
        *,
        p_value: float | None = None,
        e_value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one test result.

        Exactly one of ``p_value`` or ``e_value`` must be supplied.
        Re-observing the same ``test_id`` *overwrites* the previous
        record (the runtime usually has one canonical evidence per
        test id; for streams use distinct ids per arrival).
        """
        if not isinstance(test_id, str) or not test_id:
            raise ValueError("test_id must be a non-empty string")
        if (p_value is None) == (e_value is None):
            raise ValueError("supply exactly one of p_value or e_value")
        if p_value is not None:
            _validate_p(p_value)
        if e_value is not None:
            _validate_e(e_value)
        record = TestRecord(
            test_id=test_id,
            p_value=p_value,
            e_value=e_value,
            ts=time.time(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            is_new = test_id not in self._tests
            self._tests[test_id] = record
            if is_new:
                self._order.append(test_id)
        self._emit(AUDIT_OBSERVED, {
            "auditor_id": self._id,
            "test_id": test_id,
            "p_value": p_value,
            "e_value": e_value,
        })

    def observe_many(
        self,
        records: Iterable[tuple[str, float] | tuple[str, float, Mapping[str, Any]]],
        *,
        kind: str = "p_value",
    ) -> None:
        """Bulk ingest. ``kind`` is ``"p_value"`` or ``"e_value"``."""
        if kind not in ("p_value", "e_value"):
            raise ValueError("kind must be 'p_value' or 'e_value'")
        for rec in records:
            if len(rec) == 2:
                tid, val = rec  # type: ignore[misc]
                meta: Mapping[str, Any] = {}
            elif len(rec) == 3:
                tid, val, meta = rec  # type: ignore[misc]
            else:
                raise ValueError("record must be (id, value) or (id, value, metadata)")
            if kind == "p_value":
                self.observe(tid, p_value=val, metadata=meta)
            else:
                self.observe(tid, e_value=val, metadata=meta)

    def n_tests(self) -> int:
        with self._lock:
            return len(self._tests)

    def snapshot(self) -> tuple[TestRecord, ...]:
        with self._lock:
            return tuple(self._tests[tid] for tid in self._order)

    def get(self, test_id: str) -> TestRecord | None:
        with self._lock:
            return self._tests.get(test_id)

    def clear_test(self, test_id: str) -> None:
        with self._lock:
            if test_id in self._tests:
                del self._tests[test_id]
                self._order = [t for t in self._order if t != test_id]
        self._emit(AUDIT_TEST_CLEARED, {
            "auditor_id": self._id,
            "test_id": test_id,
        })

    def clear(self) -> None:
        with self._lock:
            self._tests.clear()
            self._order.clear()
            self._online_state.clear()
            self._reports.clear()
        self._emit(AUDIT_CLEARED, {"auditor_id": self._id})

    def reports(self) -> tuple[AuditReport, ...]:
        with self._lock:
            return tuple(self._reports)

    # -----------------------------------------------------------------
    # Offline / batch decisions
    # -----------------------------------------------------------------

    def decide(
        self,
        *,
        method: str = METHOD_BH,
        alpha: float = 0.05,
        lam: float = 0.5,
    ) -> AuditReport:
        """Compute a batch decision on all observed tests.

        ``method`` ∈ KNOWN_OFFLINE_METHODS. P-value methods require
        all observed tests to have a p_value; e-value methods
        (``ebh``) require all observed tests to have an e_value.
        Mixed observation is rejected with ``ValueError``.

        Returns an :class:`AuditReport` with per-test decisions and a
        SHA-256 receipt hash (when an attestor is wired).
        """
        if method not in KNOWN_OFFLINE_METHODS:
            raise ValueError(f"unknown offline method: {method}")
        _validate_alpha(alpha)
        t0 = time.time()
        with self._lock:
            records = [self._tests[tid] for tid in self._order]
        if not records:
            raise ValueError("no tests observed")
        is_e = method == METHOD_EBH
        if is_e:
            if any(r.e_value is None for r in records):
                raise ValueError("e-BH requires e_value for every test")
            values = [r.e_value for r in records]
            assert all(v is not None for v in values)
            rejects = e_value_bh_rejections([v for v in values if v is not None], alpha)
        else:
            if any(r.p_value is None for r in records):
                raise ValueError(f"method {method} requires p_value for every test")
            values = [r.p_value for r in records]
            p_list = [v for v in values if v is not None]
            if method == METHOD_BH:
                rejects = bh_rejections(p_list, alpha)
            elif method == METHOD_BY:
                rejects = by_rejections(p_list, alpha)
            elif method == METHOD_HOLM:
                rejects = holm_rejections(p_list, alpha)
            elif method == METHOD_HOCHBERG:
                rejects = hochberg_rejections(p_list, alpha)
            elif method == METHOD_BONFERRONI:
                rejects = bonferroni_rejections(p_list, alpha)
            elif method == METHOD_SIDAK:
                rejects = sidak_rejections(p_list, alpha)
            elif method == METHOD_STOREY:
                rejects = storey_rejections(p_list, alpha, lam=lam)
            else:
                raise ValueError(f"unhandled method: {method}")
        # Rank assignment for receipts.
        if is_e:
            ranked = sorted(
                range(len(records)),
                key=lambda i: -(records[i].e_value if records[i].e_value is not None else -math.inf),
            )
        else:
            ranked = sorted(
                range(len(records)),
                key=lambda i: (records[i].p_value if records[i].p_value is not None else math.inf),
            )
        rank_map = {idx: rank for rank, idx in enumerate(ranked, start=1)}
        # Q-values only meaningful for FDR procedures
        qvals_list: list[float] | None
        if method in (METHOD_BH, METHOD_BY, METHOD_STOREY, METHOD_HOCHBERG):
            try:
                qvals_list = q_values([r.p_value for r in records if r.p_value is not None])
            except Exception:
                qvals_list = None
        else:
            qvals_list = None

        decisions: dict[str, AuditDecision] = {}
        n_reject = 0
        for i, rec in enumerate(records):
            is_r = rejects[i]
            if is_r:
                n_reject += 1
            decisions[rec.test_id] = AuditDecision(
                test_id=rec.test_id,
                rejected=is_r,
                p_value=rec.p_value,
                e_value=rec.e_value,
                q_value=qvals_list[i] if qvals_list is not None else None,
                rank=rank_map[i],
            )

        pi0 = None
        if method == METHOD_STOREY:
            pi0 = storey_pi0([r.p_value for r in records if r.p_value is not None], lam=lam)

        elapsed = time.time() - t0
        report = AuditReport(
            id=f"{self._id}-dec-{int(t0 * 1000):x}",
            method=method,
            alpha=alpha,
            n_tests=len(records),
            n_rejected=n_reject,
            pi0_estimate=pi0,
            decisions=decisions,
            elapsed_s=elapsed,
        )
        report = self._maybe_attest(report)
        with self._lock:
            self._reports.append(report)
        self._emit(AUDIT_DECIDED, {
            "auditor_id": self._id,
            "report_id": report.id,
            "method": method,
            "alpha": alpha,
            "n_tests": report.n_tests,
            "n_rejected": n_reject,
            "elapsed_s": elapsed,
        })
        return report

    # -----------------------------------------------------------------
    # Online decisions
    # -----------------------------------------------------------------

    def decide_online(
        self,
        test_id: str,
        *,
        p_value: float,
        method: str = METHOD_LORD,
        alpha: float = 0.05,
        w0: float | None = None,
        lam: float = 0.5,
        tau: float = 0.5,
        payout: float | None = None,
    ) -> bool:
        """Return a single online decision for one arriving test.

        The Auditor maintains *internal* state per (method, alpha)
        configuration so subsequent calls continue the same online
        process. State is keyed by ``(method, alpha)``.

        For LORD-3 the wealth dynamics use the default γ_k = 1/(k·log²(max k,2))
        normalised over the projected lifetime; we use a streaming
        formulation where γ_k is computed on demand.
        """
        if method not in KNOWN_ONLINE_METHODS:
            raise ValueError(f"unknown online method: {method}")
        _validate_alpha(alpha)
        _validate_p(p_value)
        # Persist the observation
        self.observe(test_id, p_value=p_value)
        key = f"{method}:{alpha:.10f}"
        with self._lock:
            state = self._online_state.get(key)
            if state is None:
                state = self._init_online_state(method, alpha, w0, lam, tau, payout)
                self._online_state[key] = state
        decision = self._step_online(state, p_value, method, alpha, lam, tau, payout)
        with self._lock:
            self._online_state[key] = state
        self._emit(AUDIT_ONLINE_DECIDED, {
            "auditor_id": self._id,
            "test_id": test_id,
            "method": method,
            "alpha": alpha,
            "rejected": decision,
            "t": state["t"],
            "wealth": state.get("wealth"),
            "n_rejected": state["n_rejected"],
        })
        return decision

    def _init_online_state(
        self,
        method: str,
        alpha: float,
        w0: float | None,
        lam: float,
        tau: float,
        payout: float | None,
    ) -> dict[str, Any]:
        if method == METHOD_ALPHA_INVEST:
            w0_use = 0.5 * alpha if w0 is None else float(w0)
            if not 0.0 <= w0_use < 1.0:
                raise ValueError("W_0 must be in [0, 1) for alpha-invest")
            return {
                "method": method,
                "alpha": alpha,
                "t": 0,
                "wealth": w0_use,
                "payout": alpha if payout is None else float(payout),
                "n_rejected": 0,
                "rejection_times": [],
                "candidates": [],
                "discarded": [],
            }
        w0_use = 0.5 * alpha if w0 is None else float(w0)
        if not 0.0 <= w0_use < alpha:
            raise ValueError("W_0 must be in [0, alpha)")
        return {
            "method": method,
            "alpha": alpha,
            "t": 0,
            "wealth": w0_use,
            "w0": w0_use,
            "n_rejected": 0,
            "rejection_times": [],
            "candidates": [],
            "discarded": [],
        }

    @staticmethod
    def _gamma_k(k: int) -> float:
        """Normalised inverse-square weight: γ_k = 6/(π²·k²).

        Matches :func:`lord_weights`; ensures α_t ≤ α.
        """
        return _gamma_normalised(k)

    def _step_online(
        self,
        state: dict[str, Any],
        p: float,
        method: str,
        alpha: float,
        lam: float,
        tau: float,
        payout: float | None,
    ) -> bool:
        state["t"] += 1
        t = state["t"]
        if method == METHOD_ALPHA_INVEST:
            W = state["wealth"]
            alpha_t = min(W / (1.0 + W), 1.0 - _EPS) if W > 0 else 0.0
            is_reject = (alpha_t > 0) and (p <= alpha_t)
            cost = alpha_t / max(1.0 - alpha_t, _EPS)
            payout_use = state["payout"]
            if is_reject:
                W = max(0.0, W + payout_use - cost)
            else:
                W = max(0.0, W - cost)
            state["wealth"] = W
            if is_reject:
                state["n_rejected"] += 1
                state["rejection_times"].append(t)
            self._emit(AUDIT_BUDGET_UPDATED, {
                "auditor_id": self._id, "method": method,
                "t": t, "wealth": W, "alpha_t": alpha_t,
            })
            return is_reject

        w0 = state["w0"]
        rejections = state["rejection_times"]
        candidates = state["candidates"]
        discarded = state["discarded"]

        if method == METHOD_LORD:
            alpha_t = self._gamma_k(t) * w0
            for tau_j in rejections:
                kk = t - tau_j
                if kk > 0:
                    alpha_t += (alpha - w0) * self._gamma_k(kk)
            alpha_t = min(alpha_t, alpha)
            is_reject = p <= alpha_t
            if is_reject:
                state["n_rejected"] += 1
                rejections.append(t)
        elif method == METHOD_SAFFRON:
            c_prev = len(candidates)
            base_idx = t - c_prev
            alpha_t = (1.0 - lam) * self._gamma_k(base_idx) * w0 if base_idx > 0 else 0.0
            for tau_j in rejections:
                c_btw = sum(1 for c in candidates if c > tau_j)
                kk = t - tau_j - c_btw
                if kk > 0:
                    alpha_t += (1.0 - lam) * (alpha - w0) * self._gamma_k(kk)
            alpha_t = min(alpha_t, lam, alpha)
            is_reject = p <= alpha_t
            if p <= lam:
                candidates.append(t)
            if is_reject:
                state["n_rejected"] += 1
                rejections.append(t)
        elif method == METHOD_ADDIS:
            if lam > tau:
                raise ValueError("ADDIS requires lam <= tau")
            considered = t - len(discarded)
            cand_count = len(candidates)
            base_idx = considered - cand_count
            alpha_t = (tau - lam) * self._gamma_k(base_idx) * w0 if base_idx > 0 else 0.0
            for tau_j in rejections:
                c_after = sum(1 for c in candidates if c > tau_j)
                d_after = sum(1 for d in discarded if d > tau_j)
                kk = (t - tau_j) - c_after - d_after
                if kk > 0:
                    alpha_t += (tau - lam) * (alpha - w0) * self._gamma_k(kk)
            alpha_t = min(alpha_t, lam, alpha)
            is_reject = p <= alpha_t
            if p > tau:
                discarded.append(t)
            elif p <= lam:
                candidates.append(t)
            if is_reject:
                state["n_rejected"] += 1
                rejections.append(t)
        else:
            raise ValueError(f"unsupported online method: {method}")

        state["wealth"] = alpha_t  # last-used budget for telemetry
        self._emit(AUDIT_BUDGET_UPDATED, {
            "auditor_id": self._id, "method": method,
            "t": t, "wealth": alpha_t, "alpha_t": alpha_t,
        })
        return is_reject

    # -----------------------------------------------------------------
    # Combiners
    # -----------------------------------------------------------------

    def combine(
        self,
        *,
        method: str = COMBINE_FISHER,
        weights: Sequence[float] | None = None,
    ) -> CombinedReport:
        """Combine all observed p-values into a single combined p.

        Methods: ``fisher``, ``stouffer``, ``simes``, ``harmonic``,
        ``bonferroni``. Requires every observed test to have a p_value.
        """
        if method not in KNOWN_COMBINERS:
            raise ValueError(f"unknown combiner: {method}")
        t0 = time.time()
        with self._lock:
            records = [self._tests[tid] for tid in self._order]
        if not records:
            raise ValueError("no tests observed")
        if any(r.p_value is None for r in records):
            raise ValueError("combiners require p_value for every test")
        ps = [r.p_value for r in records if r.p_value is not None]
        if method == COMBINE_FISHER:
            p_combined = combine_fisher(ps)
        elif method == COMBINE_STOUFFER:
            p_combined = combine_stouffer(ps, weights=weights)
        elif method == COMBINE_SIMES:
            p_combined = combine_simes(ps)
        elif method == COMBINE_HARMONIC:
            p_combined = combine_harmonic(ps)
        elif method == COMBINE_BONFERRONI:
            p_combined = combine_bonferroni(ps)
        else:
            raise ValueError(f"unhandled combiner: {method}")
        elapsed = time.time() - t0
        cr = CombinedReport(
            id=f"{self._id}-cmb-{int(t0 * 1000):x}",
            method=method,
            n_tests=len(ps),
            combined_p=p_combined,
            elapsed_s=elapsed,
        )
        self._emit(AUDIT_COMBINED, {
            "auditor_id": self._id,
            "combine_id": cr.id,
            "method": method,
            "combined_p": p_combined,
            "n_tests": len(ps),
        })
        return cr

    # -----------------------------------------------------------------
    # Online state introspection
    # -----------------------------------------------------------------

    def online_state(self, method: str, alpha: float) -> dict[str, Any] | None:
        """Return a deep copy of the online state for inspection.

        ``None`` if no online decisions have been issued for this
        (method, alpha) pair.
        """
        key = f"{method}:{alpha:.10f}"
        with self._lock:
            state = self._online_state.get(key)
            return dict(state) if state is not None else None

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

    def _maybe_attest(self, report: AuditReport) -> AuditReport:
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
        receipt_payload = {
            "kind": "audit.decided",
            "ticket_id": report.id,
            "report_id": report.id,
            "method": report.method,
            "alpha": report.alpha,
            "n_tests": report.n_tests,
            "n_rejected": report.n_rejected,
            "hash": digest,
        }
        wrote = False
        # First try a ``record(kind=..., payload=...)`` API.
        rec = getattr(self._attestor, "record", None)
        if callable(rec):
            try:
                rec(kind="audit.decided", payload=payload)
                wrote = True
            except Exception:
                pass
        if not wrote:
            for method_name in ("append_receipt", "append", "write"):
                fn = getattr(self._attestor, method_name, None)
                if callable(fn):
                    try:
                        fn(receipt_payload)
                        wrote = True
                        break
                    except Exception:
                        pass
        # Last resort: a plain callable attestor (RuntimeAttestor pattern).
        if not wrote and callable(self._attestor):
            try:
                self._attestor(receipt_payload)
            except Exception:
                pass
        return AuditReport(
            id=report.id,
            method=report.method,
            alpha=report.alpha,
            n_tests=report.n_tests,
            n_rejected=report.n_rejected,
            pi0_estimate=report.pi0_estimate,
            decisions=report.decisions,
            elapsed_s=report.elapsed_s,
            receipt_hash=digest,
        )


# =====================================================================
# Free-function shortcut: one-shot batch decision
# =====================================================================


def audit(
    p_values: Sequence[float] | Mapping[str, float],
    *,
    method: str = METHOD_BH,
    alpha: float = 0.05,
    lam: float = 0.5,
) -> list[bool] | dict[str, bool]:
    """One-shot batch FDR / FWER decision over a list or dict of p-values.

    Returns rejection decisions in the same container shape as input:
    a list if input is a sequence, a dict if input is a mapping.
    """
    _validate_alpha(alpha)
    if method not in KNOWN_OFFLINE_METHODS or method == METHOD_EBH:
        raise ValueError(f"method must be a p-value offline method: {method}")
    if isinstance(p_values, Mapping):
        keys = list(p_values.keys())
        ps = [float(p_values[k]) for k in keys]
        if method == METHOD_BH:
            rs = bh_rejections(ps, alpha)
        elif method == METHOD_BY:
            rs = by_rejections(ps, alpha)
        elif method == METHOD_HOLM:
            rs = holm_rejections(ps, alpha)
        elif method == METHOD_HOCHBERG:
            rs = hochberg_rejections(ps, alpha)
        elif method == METHOD_BONFERRONI:
            rs = bonferroni_rejections(ps, alpha)
        elif method == METHOD_SIDAK:
            rs = sidak_rejections(ps, alpha)
        elif method == METHOD_STOREY:
            rs = storey_rejections(ps, alpha, lam=lam)
        else:
            raise ValueError(f"unhandled method: {method}")
        return {k: r for k, r in zip(keys, rs)}
    ps = [float(p) for p in p_values]
    if method == METHOD_BH:
        return bh_rejections(ps, alpha)
    if method == METHOD_BY:
        return by_rejections(ps, alpha)
    if method == METHOD_HOLM:
        return holm_rejections(ps, alpha)
    if method == METHOD_HOCHBERG:
        return hochberg_rejections(ps, alpha)
    if method == METHOD_BONFERRONI:
        return bonferroni_rejections(ps, alpha)
    if method == METHOD_SIDAK:
        return sidak_rejections(ps, alpha)
    if method == METHOD_STOREY:
        return storey_rejections(ps, alpha, lam=lam)
    raise ValueError(f"unhandled method: {method}")


def audit_e(e_values: Sequence[float] | Mapping[str, float], *, alpha: float = 0.05) -> list[bool] | dict[str, bool]:
    """One-shot e-BH (FDR ≤ α under arbitrary dependence)."""
    _validate_alpha(alpha)
    if isinstance(e_values, Mapping):
        keys = list(e_values.keys())
        evs = [float(e_values[k]) for k in keys]
        rs = e_value_bh_rejections(evs, alpha)
        return {k: r for k, r in zip(keys, rs)}
    evs = [float(e) for e in e_values]
    return e_value_bh_rejections(evs, alpha)
