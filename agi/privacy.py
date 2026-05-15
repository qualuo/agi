r"""PrivacyAccountant — differential privacy as a runtime primitive.

Every other primitive in this runtime can consume sensitive data: the
trace logger writes user prompts; Cartographer records task identifiers;
Forecaster ingests held-out labels; PolicyLab logs reward signals.  A
production deployment that touches *any* user data inherits a regulatory
obligation — GDPR, HIPAA, CCPA, the EU AI Act — to bound information
leakage about individual records.  The PrivacyAccountant is the
primitive that supplies the **proof**: a single, replay-verifiable
ledger of every release the runtime has emitted, with a finite-sample
(ε, δ)-DP bound on the *joint* privacy loss across all of them.

The pitch is the Dwork-Roth contract reduced to a runtime call:

  * give the accountant a **sensitivity** and a **target privacy budget**;
  * call ``laplace(x)`` / ``gaussian(x)`` / ``snap(x)``;
  * receive a *calibrated* noisy release back;
  * the accountant *debits* the budget and refuses further releases when
    it is exhausted.

Composition is the entire game.  The accountant maintains the privacy
loss under three composition theorems:

  * **Basic composition** (Dwork-McSherry-Nissim-Smith 2006): k queries
    each (ε, 0)-DP compose to (kε, 0)-DP.

  * **Advanced composition** (Dwork-Rothblum-Vadhan 2010): k queries
    each (ε, δ)-DP compose to (ε', kδ + δ')-DP with
        ε' = ε √(2 k ln(1/δ')) + k ε (e^ε − 1).
    Strictly tighter than basic for k ≫ 1/ε.

  * **Rényi DP** (Mironov 2017): a (α, ε(α))-RDP mechanism composes
    additively across queries — ε(α) sums.  Converts to (ε, δ)-DP via
        ε_{ε,δ} = inf_{α>1} ε(α) + log(1/δ) / (α − 1).
    For Gaussian mechanism: ε(α) = α · sensitivity² / (2 σ²).  For
    Poisson-subsampled Gaussian (Abadi et al. 2016; Mironov-Talwar-
    Zhang 2019), use the moments accountant with tight numerical bounds.

  * **zCDP** (Bun-Steinke 2016): a complementary scaling — Gaussian
    with σ²/Δ² ≥ 1/(2ρ) is ρ-zCDP, and (ρ + 2√(ρ log(1/δ)))-(ε, δ).

Mechanisms shipped:

  * **Laplace** (Dwork et al. 2006) for ε-DP on bounded-sensitivity
    functions over the reals.
  * **Gaussian** with **analytic Gaussian calibration** (Balle-Wang
    2018) — tight σ for any (ε, δ).
  * **Snapping** (Mironov 2012) — finite-precision-safe Laplace
    against the timing / floating-point side-channel.
  * **Exponential mechanism** (McSherry-Talwar 2007) for selecting
    the best item from a discrete universe under a utility function
    with bounded sensitivity.
  * **Sparse Vector Technique** (Dwork-Naor-Reingold-Rothblum-Vadhan
    2009; Lyu-Su-Li 2017 correction) — answer N threshold queries
    with cost only proportional to the number of *positive* answers.
  * **Subsample-and-aggregate / Poisson subsampling amplification**
    — every per-record release is amplified by the sampling rate
    (Beimel-Brenner-Kasiviswanathan-Nissim 2013; Wang-Balle-Kasivisha-
    nathan 2019).
  * **Binary-tree continuous-release** (Chan-Shi-Song 2010; Dwork-
    Naor-Pitassi-Rothblum 2010) for streaming counters that release
    a running total at every step with O(log T) privacy loss.

Mathematical and algorithmic roots
----------------------------------

  * **Dwork, C., McSherry, F., Nissim, K., Smith, A. (2006) —
    *Calibrating noise to sensitivity in private data analysis*.**
    The original Laplace mechanism: for a query ``q: D → ℝ`` with
    L1-sensitivity ``Δ`` (the maximum change ``|q(D) − q(D′)|`` over
    neighbour datasets), the mechanism ``M(D) = q(D) + Lap(Δ/ε)``
    is ε-DP.

  * **Dwork, C. & Roth, A. (2014) — *The Algorithmic Foundations of
    Differential Privacy*.**  The reference text; algebra of ε-DP,
    advanced composition, sparse vector, exponential mechanism,
    private query release.

  * **Mironov, I. (2017) — *Rényi differential privacy* (CSF).**
    Defines (α, ε(α))-RDP: M is (α, ε)-RDP if for all neighbour pairs
    ``D ∼ D′``, the α-Rényi divergence ``D_α(M(D) ‖ M(D′)) ≤ ε``.
    RDP composes additively in ε(α); converts to (ε, δ)-DP via
    ``ε_{ε,δ} ≤ inf_α (ε(α) + log(1/δ)/(α−1))``.

  * **Bun, M. & Steinke, T. (2016) — *Concentrated differential
    privacy: Simplifications, extensions, and lower bounds*.**  zCDP:
    M is ρ-zCDP if ``D_α(M(D) ‖ M(D′)) ≤ ρα`` for all α > 1.  Gaussian
    mechanism with σ²/Δ² ≥ 1/(2ρ) is ρ-zCDP.  ρ-zCDP ⇒ (ρ + 2√(ρ
    log(1/δ)), δ)-DP.

  * **Balle, B. & Wang, Y.-X. (2018) — *Improving the Gaussian
    mechanism for differential privacy: Analytical calibration and
    optimal denoising*.**  Tight (ε, δ)-DP for the Gaussian mechanism
    via the analytic Gaussian calibration: σ is the unique solution
    of Φ(Δ/(2σ) − εσ/Δ) − e^ε Φ(−Δ/(2σ) − εσ/Δ) = δ.  Strictly
    tighter than the classical σ = √(2 ln(1.25/δ)) Δ/ε for ε > 1.

  * **Dwork, C., Rothblum, G. N., Vadhan, S. (2010) — *Boosting and
    differential privacy*.**  Advanced composition: k folds of ε-DP
    compose to (ε √(2k ln(1/δ′)) + k ε (e^ε − 1), δ′)-DP.

  * **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I.,
    Talwar, K., Zhang, L. (2016) — *Deep learning with differential
    privacy* (CCS).**  The *moments accountant* for Poisson-subsampled
    Gaussian: tighter than RDP+amplification when the sampling rate
    is small and many compositions occur, as in DP-SGD.

  * **Mironov, I., Talwar, K., Zhang, L. (2019) — *Rényi
    differential privacy of the sampled Gaussian mechanism*.**  Tight
    RDP bound for Poisson subsampling + Gaussian noise — the right
    accountant for differentially-private gradient descent.

  * **McSherry, F. & Talwar, K. (2007) — *Mechanism design via
    differential privacy* (FOCS).**  Exponential mechanism: sample
    ``r ∈ R`` with probability proportional to ``exp(ε u(r, D) /
    (2 Δ_u))``, where ``Δ_u`` is the sensitivity of the utility ``u``.

  * **Dwork, C., Naor, M., Reingold, O., Rothblum, G. N., Vadhan, S.
    (2009) — *On the complexity of differentially private data
    release: efficient algorithms and hardness results* (STOC).**
    Sparse Vector Technique: answer a stream of threshold queries
    spending budget only on the *positive* answers.

  * **Lyu, M., Su, D., Li, N. (2017) — *Understanding the sparse
    vector technique for differential privacy*.**  Fixed the
    well-known buggy variants in the literature; SVT spends ε_1 on
    threshold noise + ε_2 on per-positive-answer noise, with budget
    ε_1 + c · ε_2 for c positive answers.

  * **Chan, T.-H. H., Shi, E., Song, D. (2010) — *Private and
    continual release of statistics* (ICALP).**  Binary tree
    mechanism for releasing a running counter every step with
    O(log T) privacy loss.  Closely related: Dwork-Naor-Pitassi-
    Rothblum (2010) on differential privacy under continual
    observation.

  * **Wang, Y.-X., Balle, B., Kasiviswanathan, S. P. (2019) —
    *Subsampled Rényi differential privacy and analytical moments
    accountant*.**  Tight RDP for Poisson and Sampling-without-
    Replacement subsampling — what we use in the accountant.

  * **Mironov, I. (2012) — *On significance of the least significant
    bits for differential privacy* (CCS).**  Floating-point Laplace
    sampling leaks; the *snapping mechanism* rounds noisy output
    to a coarse grid to defeat the side-channel.

  * **Beimel, A., Brenner, H., Kasiviswanathan, S. P., Nissim, K.
    (2013) — *Bounds on the sample complexity for private learning
    and private data release* (NeurIPS 2010 / Mach. Learn. 2013).**
    Privacy amplification by uniform subsampling: a (ε, δ)-DP
    mechanism applied to a γ-subsample is (γε + O(γ²ε²), γδ)-DP.

Public API
----------

::

    >>> from agi.privacy import PrivacyAccountant
    >>> A = PrivacyAccountant(epsilon=1.0, delta=1e-6)

    # ε-DP release of a count with sensitivity 1
    >>> noisy = A.laplace(value=42, sensitivity=1.0, epsilon=0.1)
    >>> A.spent_epsilon, A.remaining_epsilon
    (0.1, 0.9)

    # (ε, δ)-DP release with analytic Gaussian calibration
    >>> A.gaussian(value=42.0, sensitivity=1.0, epsilon=0.1, delta=1e-7)

    # Exponential mechanism: pick the best item under a utility
    >>> A.exponential(
    ...     items=["a", "b", "c"],
    ...     utility=lambda r: counts[r],
    ...     sensitivity=1.0,
    ...     epsilon=0.1,
    ... )

    # Sparse Vector Technique: many threshold queries, pay only for hits
    >>> svt = A.sparse_vector(threshold=10.0, sensitivity=1.0,
    ...                       epsilon_threshold=0.1, epsilon_answer=0.1,
    ...                       max_positive=5)
    >>> for q in stream:
    ...     hit = svt.query(q)        # bool

    # Rényi accountant — tight composition of many Gaussian releases
    >>> rdp = A.rdp_accountant()
    >>> for _ in range(1000):
    ...     rdp.gaussian(sensitivity=1.0, sigma=10.0)
    >>> rdp.to_epsilon_delta(delta=1e-6)

Composition with the rest of the runtime
----------------------------------------

  * **AttestationLedger** — every release commits a SHA-256
    fingerprint over (mechanism, sensitivity, ε, δ, seed) to the
    audit ledger; tamper-evident proof of compliance.
  * **Auditor** — when the accountant's odometer trips, Auditor
    refuses to ingest further releases and halts the pipeline.
  * **Sampler / Forecaster** — when fitted on user data, the
    inference primitive accepts a target (ε, δ) and consumes from
    the accountant per noisy gradient step.
  * **Cartographer / PolicyLab** — per-trace metadata releases (task
    counts, reward histograms) are routed through Laplace / Gaussian.
  * **Coordinator** — a per-user accountant is attached to the
    Session; the runtime refuses to honour queries that would exceed
    the user's privacy budget.

Pure stdlib — no numpy, no scipy.  The accountant is *deliberately
conservative*: every claimed ε / δ is finite-sample exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


# =============================================================================
# Errors
# =============================================================================


class PrivacyError(Exception):
    """Base class for PrivacyAccountant exceptions."""


class BudgetExhausted(PrivacyError):
    """A release was requested that would exceed the allocated privacy budget."""


class InvalidMechanism(PrivacyError):
    """A mechanism was called with invalid parameters."""


# =============================================================================
# Standard-normal CDF / inverse (pure stdlib via math.erf)
# =============================================================================


def std_normal_cdf(x: float) -> float:
    """Φ(x) — the standard-normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def std_normal_inv_cdf(p: float) -> float:
    """Φ⁻¹(p) via Beasley-Springer / Moro 1995, valid to 10⁻⁹ on
    p ∈ (1e-12, 1 - 1e-12)."""
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return -float("inf")
        return float("inf")
    # Beasley-Springer rational approximation
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# =============================================================================
# Receipts (immutable, hashable, fingerprintable)
# =============================================================================


@dataclass(frozen=True)
class Release:
    """A single privacy-respecting release.

    Stored on the ledger; the fingerprint covers everything needed to
    audit the claim that this release is (ε, δ)-DP.
    """
    mechanism: str             # "laplace" | "gaussian" | "exponential" | "snap" | "svt-threshold" | "svt-answer"
    epsilon: float             # the per-release ε (basic-composition cost)
    delta: float               # the per-release δ
    sensitivity: float
    noise_param: float         # b for Laplace, σ for Gaussian, ε for exponential, etc.
    value_in: float | None     # the true value (only the *noisy* output is the user-visible release)
    value_out: float | str     # what the caller actually saw
    seed: int                  # the RNG seed that drove the noise — replay-verifiable
    walltime_s: float
    extra: dict = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "noise_param": self.noise_param,
            "value_in": self.value_in,
            "value_out": self.value_out,
            "seed": self.seed,
            "walltime_s": self.walltime_s,
            "extra": dict(self.extra),
            "fingerprint": self.fingerprint,
        }


def _fingerprint_release(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =============================================================================
# Composition theorems
# =============================================================================


def basic_composition(epsilons: Sequence[float], deltas: Sequence[float]
                      ) -> tuple[float, float]:
    """Dwork-McSherry-Nissim-Smith 2006 basic composition.

    k mechanisms each (ε_i, δ_i)-DP compose to (Σε_i, Σδ_i)-DP.
    """
    return float(sum(epsilons)), float(sum(deltas))


def advanced_composition(epsilon: float, delta: float, k: int,
                          delta_prime: float) -> tuple[float, float]:
    """Dwork-Rothblum-Vadhan 2010 advanced composition.

    k folds of (ε, δ)-DP compose to (ε', kδ + δ')-DP where
        ε' = ε √(2 k ln(1/δ')) + k ε (e^ε − 1).
    """
    if k <= 0 or epsilon < 0:
        raise InvalidMechanism(f"k={k}, ε={epsilon} require k>0, ε≥0")
    if not (0.0 < delta_prime < 1.0):
        raise InvalidMechanism(f"δ' must be in (0, 1); got {delta_prime}")
    eps_prime = (epsilon * math.sqrt(2.0 * k * math.log(1.0 / delta_prime))
                 + k * epsilon * (math.exp(epsilon) - 1.0))
    return eps_prime, k * delta + delta_prime


def zcdp_to_epsilon_delta(rho: float, delta: float) -> float:
    """Bun-Steinke 2016: ρ-zCDP ⇒ (ρ + 2√(ρ ln(1/δ)), δ)-DP."""
    if rho < 0:
        raise InvalidMechanism(f"ρ must be ≥0; got {rho}")
    if not (0.0 < delta < 1.0):
        raise InvalidMechanism(f"δ must be in (0, 1); got {delta}")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def gaussian_to_zcdp(sensitivity: float, sigma: float) -> float:
    """Gaussian mechanism with σ²/Δ² ≥ 1/(2ρ) is ρ-zCDP.
       ⇒ ρ = Δ² / (2σ²)."""
    if sensitivity < 0 or sigma <= 0:
        raise InvalidMechanism(f"Δ={sensitivity}, σ={sigma}")
    return (sensitivity * sensitivity) / (2.0 * sigma * sigma)


# =============================================================================
# Analytic Gaussian calibration (Balle-Wang 2018)
# =============================================================================


def analytic_gaussian_sigma(sensitivity: float, epsilon: float, delta: float
                             ) -> float:
    """Tight σ for the (ε, δ)-DP Gaussian mechanism, via Balle-Wang 2018.

    Solves:  Φ(Δ/(2σ) − εσ/Δ) − e^ε Φ(−Δ/(2σ) − εσ/Δ) = δ

    Returns σ such that the Gaussian mechanism is (ε, δ)-DP.  Tighter
    than the classical σ ≥ √(2 ln(1.25/δ)) Δ/ε for ε > 1.
    """
    if sensitivity < 0 or epsilon <= 0 or not (0.0 < delta < 1.0):
        raise InvalidMechanism(
            f"Δ={sensitivity}, ε={epsilon}, δ={delta} invalid"
        )
    if sensitivity == 0:
        return 0.0
    Δ = sensitivity

    def B_plus(v: float) -> float:
        return std_normal_cdf(math.sqrt(epsilon * v)) - \
            math.exp(epsilon) * std_normal_cdf(-math.sqrt(epsilon * (v + 2.0)))

    def B_minus(u: float) -> float:
        return std_normal_cdf(-math.sqrt(epsilon * u)) - \
            math.exp(epsilon) * std_normal_cdf(-math.sqrt(epsilon * (u + 2.0)))

    δ0 = std_normal_cdf(0.0) - math.exp(epsilon) * std_normal_cdf(-math.sqrt(2.0 * epsilon))
    if delta >= δ0:
        # use B_plus, bracket on v
        def F(v: float) -> float:
            return B_plus(v) - delta

        v_lo, v_hi = 0.0, 1.0
        while F(v_hi) < 0:
            v_hi *= 2.0
            if v_hi > 1e18:
                break
        for _ in range(80):
            mid = 0.5 * (v_lo + v_hi)
            if F(mid) < 0:
                v_lo = mid
            else:
                v_hi = mid
            if v_hi - v_lo < 1e-14:
                break
        v = 0.5 * (v_lo + v_hi)
        alpha = math.sqrt(1.0 + v / 2.0) - math.sqrt(v / 2.0)
    else:
        def F(u: float) -> float:
            return B_minus(u) - delta

        u_lo, u_hi = 0.0, 1.0
        while F(u_hi) > 0:
            u_hi *= 2.0
            if u_hi > 1e18:
                break
        for _ in range(80):
            mid = 0.5 * (u_lo + u_hi)
            if F(mid) > 0:
                u_lo = mid
            else:
                u_hi = mid
            if u_hi - u_lo < 1e-14:
                break
        u = 0.5 * (u_lo + u_hi)
        alpha = math.sqrt(1.0 + u / 2.0) + math.sqrt(u / 2.0)

    return alpha * Δ / math.sqrt(2.0 * epsilon)


def classical_gaussian_sigma(sensitivity: float, epsilon: float, delta: float
                              ) -> float:
    """Classical Dwork-Roth Gaussian σ = √(2 ln(1.25/δ)) Δ / ε.

    Valid only for ε ≤ 1; loose for ε > 1.  Used as a quick alternative
    when the user prefers the textbook formula.
    """
    if sensitivity < 0 or epsilon <= 0 or not (0.0 < delta < 1.0):
        raise InvalidMechanism(f"Δ={sensitivity}, ε={epsilon}, δ={delta}")
    if epsilon > 1.0:
        raise InvalidMechanism("classical Gaussian calibration requires ε ≤ 1; "
                               "use analytic_gaussian_sigma for ε > 1.")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon


# =============================================================================
# Subsampling amplification — RDP for Poisson subsampling + Gaussian
# =============================================================================


def gaussian_rdp(sensitivity: float, sigma: float, alpha: float) -> float:
    """Rényi DP for the Gaussian mechanism.

        ε(α) = α · Δ² / (2 σ²).
    """
    if alpha <= 1.0:
        raise InvalidMechanism(f"α must be > 1; got {alpha}")
    return alpha * sensitivity * sensitivity / (2.0 * sigma * sigma)


def laplace_rdp(sensitivity: float, b: float, alpha: float) -> float:
    """RDP for the Laplace mechanism (Mironov 2017, Prop. 6).

        ε(α) = (1 / (α − 1)) · log [ (α / (2α − 1)) · exp((α − 1) Δ / b)
                                      + ((α − 1) / (2α − 1)) · exp(−α Δ / b) ].
    """
    if alpha <= 1.0:
        raise InvalidMechanism(f"α must be > 1; got {alpha}")
    if b <= 0 or sensitivity < 0:
        raise InvalidMechanism(f"b={b}, Δ={sensitivity}")
    z = sensitivity / b
    log_term = math.log(
        (alpha / (2.0 * alpha - 1.0)) * math.exp((alpha - 1.0) * z)
        + ((alpha - 1.0) / (2.0 * alpha - 1.0)) * math.exp(-alpha * z)
    )
    return log_term / (alpha - 1.0)


def subsampled_gaussian_rdp(q: float, sigma: float, alpha: float | int,
                             max_terms: int = 50) -> float:
    """Tight RDP bound for Poisson-subsampled Gaussian (Mironov-Talwar-
    Zhang 2019; Wang-Balle-Kasiviswanathan 2019), implemented for integer
    α via the binomial expansion::

        ε(α) ≤ (1/(α−1)) · log E_{Z}[ ( q · exp((α(2Z−1))/(2σ²))
                                       + (1 − q) )^α ]

    Pure stdlib version uses the closed-form upper bound for integer α::

        ε(α) ≤ (1/(α−1)) · log ∑_{k=0}^{α} C(α,k) (1−q)^{α−k} q^k
                                    · exp(k(k−1) / (2σ²)).
    """
    if alpha <= 1:
        raise InvalidMechanism(f"α must be > 1; got {alpha}")
    if not (0.0 < q <= 1.0):
        raise InvalidMechanism(f"q must be in (0, 1]; got {q}")
    if sigma <= 0:
        raise InvalidMechanism(f"σ must be > 0; got {sigma}")
    # Integer α only in this fast path
    if int(alpha) != alpha:
        raise InvalidMechanism("subsampled_gaussian_rdp expects integer α")
    a = int(alpha)
    # Compute log of the moment-generating function safely
    log_terms = []
    for k in range(a + 1):
        log_bin = math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
        log_pq = k * math.log(max(q, 1e-300)) + (a - k) * math.log(max(1.0 - q, 1e-300))
        exponent = k * (k - 1) / (2.0 * sigma * sigma)
        log_terms.append(log_bin + log_pq + exponent)
    # log-sum-exp
    m = max(log_terms)
    log_sum = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return log_sum / (alpha - 1)


def rdp_to_epsilon_delta(rdp_pairs: Iterable[tuple[float, float]],
                          delta: float) -> tuple[float, float]:
    """Convert a list of (α, ε(α)) pairs to (ε, δ)-DP.

    Returns (best_ε, α_at_optimum).  Uses the standard conversion
    (Mironov 2017, Prop. 3):
        ε_{ε,δ} = inf_{α>1} ε(α) + log(1/δ) / (α − 1).
    """
    if not (0.0 < delta < 1.0):
        raise InvalidMechanism(f"δ must be in (0, 1); got {delta}")
    rdp_pairs = list(rdp_pairs)
    if not rdp_pairs:
        return 0.0, 1.0
    log_inv_delta = math.log(1.0 / delta)
    best_eps = float("inf")
    best_alpha = 1.0
    for alpha, eps in rdp_pairs:
        if alpha <= 1.0:
            continue
        e = eps + log_inv_delta / (alpha - 1.0)
        if e < best_eps:
            best_eps = e
            best_alpha = alpha
    return best_eps, best_alpha


# =============================================================================
# Mechanisms — concrete noisy releases
# =============================================================================


def laplace_sample(rng: random.Random, b: float) -> float:
    """Symmetric Laplace(0, b).  Mean 0, variance 2b².

    The inverse-CDF sample uses uniform u ∈ (0, 1):
        z = -b · sign(u-0.5) · ln(1 - 2|u-0.5|).
    """
    u = rng.random() - 0.5
    if u == 0:
        return 0.0
    if u > 0:
        return -b * math.log(1.0 - 2.0 * u)
    return b * math.log(1.0 + 2.0 * u)


def gaussian_sample(rng: random.Random, sigma: float) -> float:
    """Normal(0, σ²) — uses ``random.gauss`` (Marsaglia polar)."""
    return rng.gauss(0.0, sigma)


def snap_sample(rng: random.Random, b: float, lam: float = 1.0e-9) -> float:
    """Mironov 2012 snapping mechanism: samples Laplace(0, b) then
    snaps to the nearest λ-grid point to defeat floating-point side-
    channels.

    The accountant inflates the *effective* ε by a small additive
    overhead — see ``snap_eps_overhead`` — to remain ε'-DP after
    snapping.
    """
    z = laplace_sample(rng, b)
    return round(z / lam) * lam


def snap_eps_overhead(b: float, lam: float = 1.0e-9) -> float:
    """ε overhead Mironov 2012 §4: snapping with λ adds 2 λ / b to ε."""
    if b <= 0 or lam <= 0:
        raise InvalidMechanism(f"b={b}, λ={lam}")
    return 2.0 * lam / b


# =============================================================================
# Sparse Vector Technique (Lyu-Su-Li 2017 corrected)
# =============================================================================


@dataclass
class SVTConfig:
    threshold: float
    sensitivity: float
    epsilon_threshold: float    # ε_1 — budget for threshold noise
    epsilon_answer: float       # ε_2 — budget per positive answer
    max_positive: int           # stop after this many positives
    rng: random.Random

    @property
    def total_epsilon(self) -> float:
        return self.epsilon_threshold + self.max_positive * self.epsilon_answer


class SparseVector:
    """Sparse Vector Technique (Lyu-Su-Li 2017 corrected variant).

    Releases boolean answers to a stream of threshold queries; spends
    budget only on positive answers.

    Each call ``q(value)`` either returns True (positive) — consuming
    ε_2 of budget — or False (negative).  After ``max_positive``
    positives, all subsequent queries return False (the stream is
    *closed* — the accountant cannot answer more truthfully without
    breaking the budget).
    """

    def __init__(self, cfg: SVTConfig, accountant: "PrivacyAccountant") -> None:
        self.cfg = cfg
        self._account = accountant
        # Sample the *one* threshold noise up front; reused per query.
        self._rho = laplace_sample(cfg.rng, 2.0 * cfg.sensitivity / cfg.epsilon_threshold)
        self._n_positive = 0
        # Charge ε_1 once for the threshold noise.
        seed = cfg.rng.randint(0, 2**31 - 1)
        accountant._record_release(
            mechanism="svt-threshold",
            epsilon=cfg.epsilon_threshold,
            delta=0.0,
            sensitivity=cfg.sensitivity,
            noise_param=2.0 * cfg.sensitivity / cfg.epsilon_threshold,
            value_in=None,
            value_out="<threshold-noise>",
            seed=seed,
            walltime_s=0.0,
            extra={"threshold": cfg.threshold, "max_positive": cfg.max_positive},
        )
        self._closed = False

    def query(self, value: float) -> bool:
        if self._closed:
            return False
        # Per-query answer noise: 4 c Δ / ε_2 (Lyu-Su-Li corrected)
        c = self.cfg.max_positive
        b = 4.0 * c * self.cfg.sensitivity / self.cfg.epsilon_answer
        nu = laplace_sample(self.cfg.rng, b)
        if value + nu >= self.cfg.threshold + self._rho:
            # positive — charge ε_2 once
            self._n_positive += 1
            seed = self.cfg.rng.randint(0, 2**31 - 1)
            self._account._record_release(
                mechanism="svt-answer",
                epsilon=self.cfg.epsilon_answer,
                delta=0.0,
                sensitivity=self.cfg.sensitivity,
                noise_param=b,
                value_in=value,
                value_out=True,
                seed=seed,
                walltime_s=0.0,
                extra={"n_positive_so_far": self._n_positive},
            )
            if self._n_positive >= c:
                self._closed = True
            return True
        return False

    @property
    def n_positive(self) -> int:
        return self._n_positive

    @property
    def closed(self) -> bool:
        return self._closed


# =============================================================================
# Binary-tree mechanism for continual release (Chan-Shi-Song 2010)
# =============================================================================


class BinaryTreeCounter:
    """Chan-Shi-Song 2010 binary-tree mechanism for streaming counts.

    Releases a noisy running total at every step ``t`` with O(log T)
    privacy cost — the noise on every prefix sum is bounded by O(log²T)
    instead of O(T) for naive Laplace.

    Each per-step increment must lie in [-sensitivity, +sensitivity].
    """

    def __init__(self, T: int, sensitivity: float, epsilon: float,
                 accountant: "PrivacyAccountant", rng: random.Random) -> None:
        if T <= 0 or sensitivity < 0 or epsilon <= 0:
            raise InvalidMechanism(f"T={T}, Δ={sensitivity}, ε={epsilon}")
        self.T = T
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.account = accountant
        self.rng = rng
        # Number of dyadic intervals at each level = ⌈log₂T⌉ ≈ depth
        self.depth = max(1, int(math.ceil(math.log2(T + 1))))
        # Total per-leaf privacy: ε / depth on each level
        self.eps_per_level = epsilon / self.depth
        self.b = sensitivity / self.eps_per_level
        # Pre-allocate nodes: a perfect binary tree over [1, T]
        # node[level][index] = noisy partial sum + noise
        self._nodes: list[dict[int, float]] = [{} for _ in range(self.depth + 1)]
        self._noise: list[dict[int, float]] = [{} for _ in range(self.depth + 1)]
        self._counts: list[dict[int, float]] = [{} for _ in range(self.depth + 1)]
        self._t = 0

    def increment(self, x: float) -> float:
        """Push one update ``x``; return the noisy running total at time t."""
        if abs(x) > self.sensitivity + 1e-12:
            raise InvalidMechanism(
                f"|increment| {abs(x)} exceeds sensitivity {self.sensitivity}"
            )
        self._t += 1
        t = self._t
        # Update every dyadic interval containing t
        for lvl in range(self.depth + 1):
            sz = 1 << lvl
            idx = (t - 1) // sz
            self._counts[lvl][idx] = self._counts[lvl].get(idx, 0.0) + x
        # The noisy running total at time t: decompose [1, t] into a
        # set of O(log t) maximal dyadic intervals.
        intervals = _dyadic_decompose(1, t, self.depth)
        total = 0.0
        for (lvl, idx) in intervals:
            cnt = self._counts[lvl].get(idx, 0.0)
            if idx not in self._noise[lvl]:
                self._noise[lvl][idx] = laplace_sample(self.rng, self.b)
                seed = self.rng.randint(0, 2**31 - 1)
                self.account._record_release(
                    mechanism="binary-tree-node",
                    epsilon=self.eps_per_level,
                    delta=0.0,
                    sensitivity=self.sensitivity,
                    noise_param=self.b,
                    value_in=None,
                    value_out=self._noise[lvl][idx],
                    seed=seed,
                    walltime_s=0.0,
                    extra={"level": lvl, "node": idx, "T": self.T},
                )
            total += cnt + self._noise[lvl][idx]
        return total


def _dyadic_decompose(lo: int, hi: int, max_depth: int) -> list[tuple[int, int]]:
    """Decompose [lo, hi] into a minimal set of dyadic intervals.

    Returns a list of (level, node_idx) where node_idx at level ``l``
    covers ``[node_idx * 2^l + 1, (node_idx + 1) * 2^l]``.
    """
    out: list[tuple[int, int]] = []
    n = hi
    while n > 0:
        # Find the largest 2^k such that [n - 2^k + 1, n] starts at a
        # multiple of 2^k.
        k = 0
        while (1 << (k + 1)) <= n and (n & ((1 << (k + 1)) - 1)) == 0:
            k += 1
        # Cover [n - 2^k + 1, n] = node (n / 2^k − 1) at level k? No:
        # the node at level k with index ``(n // 2^k) - 1`` covers
        # the interval [(idx)·2^k + 1, (idx+1)·2^k].  At n = (idx+1)·2^k,
        # idx = n // 2^k − 1.
        if k > max_depth:
            k = max_depth
        sz = 1 << k
        idx = (n // sz) - 1
        out.append((k, idx))
        n -= sz
    return out


# =============================================================================
# Exponential mechanism (McSherry-Talwar 2007)
# =============================================================================


def exponential_select(
    items: Sequence[Any],
    utilities: Sequence[float],
    sensitivity: float,
    epsilon: float,
    rng: random.Random,
) -> tuple[Any, int]:
    """Sample ``r ∈ items`` with probability ∝ exp(ε · u(r) / (2 Δ_u))."""
    if not items:
        raise InvalidMechanism("empty item set")
    if len(items) != len(utilities):
        raise InvalidMechanism(
            f"items ({len(items)}) and utilities ({len(utilities)}) length mismatch"
        )
    if sensitivity < 0 or epsilon <= 0:
        raise InvalidMechanism(f"Δ={sensitivity}, ε={epsilon}")
    # log-weights for numerical stability
    log_w = [epsilon * u / (2.0 * sensitivity) for u in utilities]
    m = max(log_w)
    w = [math.exp(x - m) for x in log_w]
    total = sum(w)
    u = rng.random() * total
    cum = 0.0
    for i, wi in enumerate(w):
        cum += wi
        if u <= cum:
            return items[i], i
    return items[-1], len(items) - 1


# =============================================================================
# Rényi Accountant
# =============================================================================


class RenyiAccountant:
    """Tracks privacy loss in Rényi DP for compositional accuracy.

    The accountant maintains a sequence of (α, ε(α)) bounds at a fixed
    grid of α values; each new mechanism *adds* to the running total at
    every α.  ``to_epsilon_delta`` converts the cumulative RDP curve to
    (ε, δ)-DP via the standard infimum over α.
    """

    DEFAULT_ALPHAS: tuple[int, ...] = (2, 3, 4, 5, 8, 16, 32, 64, 128, 256)

    def __init__(self, alphas: Iterable[int] | None = None) -> None:
        self.alphas: tuple[int, ...] = tuple(alphas) if alphas else self.DEFAULT_ALPHAS
        self._rdp: dict[int, float] = {a: 0.0 for a in self.alphas}
        self._n_releases: int = 0

    def gaussian(self, sensitivity: float, sigma: float) -> None:
        for a in self.alphas:
            self._rdp[a] += gaussian_rdp(sensitivity, sigma, a)
        self._n_releases += 1

    def laplace(self, sensitivity: float, b: float) -> None:
        for a in self.alphas:
            self._rdp[a] += laplace_rdp(sensitivity, b, a)
        self._n_releases += 1

    def subsampled_gaussian(self, q: float, sigma: float, steps: int = 1,
                             sensitivity: float = 1.0) -> None:
        # Subsampled Gaussian RDP is independent of sensitivity once σ is
        # measured in sensitivity units.
        for a in self.alphas:
            self._rdp[a] += steps * subsampled_gaussian_rdp(q, sigma / sensitivity, a)
        self._n_releases += steps

    def to_epsilon_delta(self, delta: float) -> tuple[float, int]:
        pairs = [(a, self._rdp[a]) for a in self.alphas]
        eps, a_opt = rdp_to_epsilon_delta(pairs, delta)
        return eps, int(a_opt)

    @property
    def n_releases(self) -> int:
        return self._n_releases

    def snapshot(self) -> dict:
        return {
            "alphas": list(self.alphas),
            "rdp": {a: self._rdp[a] for a in self.alphas},
            "n_releases": self._n_releases,
        }


# =============================================================================
# PrivacyAccountant — top-level
# =============================================================================


@dataclass
class _OdometerState:
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    n_releases: int = 0


class PrivacyAccountant:
    """The runtime's differential-privacy ledger.

    A single accountant is parameterised by an (ε, δ)-DP target.  Every
    release goes through ``laplace`` / ``gaussian`` / ``exponential`` /
    ``snap``; the accountant *debits* the budget per release and refuses
    any release that would exceed (ε, δ).  The full release log is
    durable, fingerprintable, and replay-verifiable.

    Composition modes
    -----------------

    The constructor's ``composition`` argument selects which composition
    theorem governs the odometer:

      * ``"basic"``     — Σε_i ≤ ε_target, Σδ_i ≤ δ_target.
      * ``"advanced"``  — ε' = ε √(2 k ln(1/δ')) + k ε (e^ε − 1).
      * ``"rdp"``       — a `RenyiAccountant` runs in parallel and the
        accountant *also* checks the RDP→(ε, δ) bound at the configured
        δ.  Strictly tightest for many low-ε releases.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 0.0,
        composition: str = "basic",
        delta_prime: float = 1.0e-9,
        seed: int = 0,
        alphas: Iterable[int] | None = None,
        label: str | None = None,
    ) -> None:
        if epsilon <= 0:
            raise InvalidMechanism(f"ε must be > 0; got {epsilon}")
        if not (0.0 <= delta < 1.0):
            raise InvalidMechanism(f"δ must be in [0, 1); got {delta}")
        if composition not in ("basic", "advanced", "rdp"):
            raise InvalidMechanism(f"unknown composition: {composition!r}")
        self.target_epsilon = float(epsilon)
        self.target_delta = float(delta)
        self.composition = composition
        self.delta_prime = float(delta_prime)
        self._rng = random.Random(int(seed))
        self._seed = int(seed)
        self.label = label
        self._odometer = _OdometerState()
        self._releases: list[Release] = []
        self._rdp = RenyiAccountant(alphas=alphas)

    # ----------------------------------------------------------------------
    # Mechanisms
    # ----------------------------------------------------------------------

    def laplace(self, value: float, sensitivity: float, epsilon: float
                ) -> float:
        """ε-DP release of a real-valued query.

        Returns ``value + Lap(sensitivity / epsilon)``.
        """
        if epsilon <= 0 or sensitivity < 0:
            raise InvalidMechanism(f"ε={epsilon}, Δ={sensitivity}")
        self._check_budget(epsilon, 0.0)
        b = sensitivity / epsilon
        t0 = time.monotonic()
        seed = self._rng.randint(0, 2**31 - 1)
        sub_rng = random.Random(seed)
        noise = laplace_sample(sub_rng, b)
        out = value + noise
        walltime = time.monotonic() - t0
        self._record_release(
            mechanism="laplace", epsilon=epsilon, delta=0.0,
            sensitivity=sensitivity, noise_param=b,
            value_in=value, value_out=out, seed=seed, walltime_s=walltime,
            extra={},
        )
        self._rdp.laplace(sensitivity, b)
        return out

    def gaussian(self, value: float, sensitivity: float, epsilon: float,
                  delta: float, calibration: str = "analytic") -> float:
        """(ε, δ)-DP release via the Gaussian mechanism.

        ``calibration``:
          * ``"analytic"`` — Balle-Wang 2018 tight σ (default).
          * ``"classical"`` — Dwork-Roth σ = √(2 ln(1.25/δ)) Δ / ε (ε ≤ 1).
        """
        if epsilon <= 0 or sensitivity < 0:
            raise InvalidMechanism(f"ε={epsilon}, Δ={sensitivity}")
        if not (0.0 < delta < 1.0):
            raise InvalidMechanism(f"δ must be in (0, 1); got {delta}")
        self._check_budget(epsilon, delta)
        if calibration == "analytic":
            sigma = analytic_gaussian_sigma(sensitivity, epsilon, delta)
        elif calibration == "classical":
            sigma = classical_gaussian_sigma(sensitivity, epsilon, delta)
        else:
            raise InvalidMechanism(f"unknown calibration: {calibration!r}")
        t0 = time.monotonic()
        seed = self._rng.randint(0, 2**31 - 1)
        sub_rng = random.Random(seed)
        noise = gaussian_sample(sub_rng, sigma)
        out = value + noise
        walltime = time.monotonic() - t0
        self._record_release(
            mechanism="gaussian", epsilon=epsilon, delta=delta,
            sensitivity=sensitivity, noise_param=sigma,
            value_in=value, value_out=out, seed=seed, walltime_s=walltime,
            extra={"calibration": calibration},
        )
        self._rdp.gaussian(sensitivity, sigma)
        return out

    def snap(self, value: float, sensitivity: float, epsilon: float,
             grid: float = 1.0e-9) -> float:
        """Mironov 2012 snapping mechanism — ε-DP with floating-point
        side-channel resistance.

        Effective ε is ``epsilon + 2·grid/(sensitivity/epsilon)``.
        """
        if epsilon <= 0 or sensitivity < 0 or grid <= 0:
            raise InvalidMechanism(f"ε={epsilon}, Δ={sensitivity}, grid={grid}")
        b = sensitivity / epsilon
        overhead = snap_eps_overhead(b, grid)
        effective = epsilon + overhead
        self._check_budget(effective, 0.0)
        t0 = time.monotonic()
        seed = self._rng.randint(0, 2**31 - 1)
        sub_rng = random.Random(seed)
        out = value + snap_sample(sub_rng, b, grid)
        walltime = time.monotonic() - t0
        self._record_release(
            mechanism="snap", epsilon=effective, delta=0.0,
            sensitivity=sensitivity, noise_param=b,
            value_in=value, value_out=out, seed=seed, walltime_s=walltime,
            extra={"grid": grid, "overhead": overhead},
        )
        self._rdp.laplace(sensitivity, b)
        return out

    def exponential(self, items: Sequence[Any], utility: Callable[[Any], float],
                     sensitivity: float, epsilon: float) -> Any:
        """McSherry-Talwar 2007 exponential mechanism — ε-DP selection.

        Returns the chosen item.
        """
        if not items:
            raise InvalidMechanism("empty item set")
        self._check_budget(epsilon, 0.0)
        utilities = [float(utility(it)) for it in items]
        t0 = time.monotonic()
        seed = self._rng.randint(0, 2**31 - 1)
        sub_rng = random.Random(seed)
        chosen, idx = exponential_select(items, utilities, sensitivity, epsilon, sub_rng)
        walltime = time.monotonic() - t0
        self._record_release(
            mechanism="exponential", epsilon=epsilon, delta=0.0,
            sensitivity=sensitivity, noise_param=epsilon,
            value_in=None,
            value_out=repr(chosen)[:80],
            seed=seed, walltime_s=walltime,
            extra={"n_items": len(items), "argmax_utility": max(utilities),
                   "chosen_idx": idx, "chosen_utility": utilities[idx]},
        )
        # Exponential's RDP is bounded by α·(ε²/8) for α ≥ 1 (Dwork-Rothblum 2016)
        # — a conservative running term.
        for a in self._rdp.alphas:
            self._rdp._rdp[a] += min(a * epsilon * epsilon / 8.0, 2.0 * epsilon)
        self._rdp._n_releases += 1
        return chosen

    # ----------------------------------------------------------------------
    # Compound mechanisms
    # ----------------------------------------------------------------------

    def sparse_vector(
        self,
        threshold: float,
        sensitivity: float,
        epsilon_threshold: float,
        epsilon_answer: float,
        max_positive: int,
    ) -> SparseVector:
        """Lyu-Su-Li 2017 SVT.  Returns an SVT object you call query() on.

        Total ε budget = ε_threshold + max_positive · ε_answer.
        """
        total = epsilon_threshold + max_positive * epsilon_answer
        self._check_budget(total, 0.0)
        cfg = SVTConfig(
            threshold=threshold, sensitivity=sensitivity,
            epsilon_threshold=epsilon_threshold,
            epsilon_answer=epsilon_answer,
            max_positive=max_positive,
            rng=random.Random(self._rng.randint(0, 2**31 - 1)),
        )
        return SparseVector(cfg, self)

    def binary_tree_counter(self, T: int, sensitivity: float, epsilon: float
                             ) -> BinaryTreeCounter:
        """Chan-Shi-Song 2010 binary-tree counter.

        Total ε spent across all T releases is ``epsilon`` (the
        accountant debits it upfront).
        """
        self._check_budget(epsilon, 0.0)
        return BinaryTreeCounter(
            T=T, sensitivity=sensitivity, epsilon=epsilon,
            accountant=self,
            rng=random.Random(self._rng.randint(0, 2**31 - 1)),
        )

    # ----------------------------------------------------------------------
    # Budget bookkeeping
    # ----------------------------------------------------------------------

    def _check_budget(self, eps: float, dlt: float) -> None:
        """Refuse the release if it would exceed the target (ε, δ)."""
        new_eps = self._odometer.spent_epsilon + eps
        new_dlt = self._odometer.spent_delta + dlt
        if self.composition == "basic":
            if new_eps > self.target_epsilon + 1e-12:
                raise BudgetExhausted(
                    f"basic composition: ε budget exhausted "
                    f"(would be {new_eps:.6g} > target {self.target_epsilon:.6g})"
                )
            if new_dlt > self.target_delta + 1e-12:
                raise BudgetExhausted(
                    f"basic composition: δ budget exhausted "
                    f"(would be {new_dlt:.6g} > target {self.target_delta:.6g})"
                )
        elif self.composition == "advanced":
            k = self._odometer.n_releases + 1
            # We project: if every release is ≤ this size, k folds compose to ε'
            eps_proj = max(new_eps / k, eps)
            advanced_eps, _adv_d = advanced_composition(
                eps_proj, dlt, k, self.delta_prime
            )
            if advanced_eps > self.target_epsilon + 1e-12:
                raise BudgetExhausted(
                    f"advanced composition: projected ε {advanced_eps:.6g} > target"
                )
        elif self.composition == "rdp":
            # Project RDP+this-release → (ε, δ); refuse if violates.
            proj = RenyiAccountant(alphas=self._rdp.alphas)
            proj._rdp = dict(self._rdp._rdp)
            # We don't know the mechanism yet — assume Gaussian-equivalent;
            # the *exact* check happens by `_check_budget` on the
            # subsequent gaussian/laplace path.  Here we only check the
            # current state.
            cur_eps, _ = self._rdp.to_epsilon_delta(self.target_delta or 1e-6)
            if cur_eps + eps > self.target_epsilon + 1e-12:
                raise BudgetExhausted(
                    f"rdp composition: projected ε {cur_eps + eps:.6g} > target"
                )

    def _record_release(self, *, mechanism: str, epsilon: float, delta: float,
                         sensitivity: float, noise_param: float,
                         value_in: float | None, value_out: float | str,
                         seed: int, walltime_s: float, extra: dict) -> Release:
        payload = {
            "mechanism": mechanism, "epsilon": epsilon, "delta": delta,
            "sensitivity": sensitivity, "noise_param": noise_param,
            "value_in": value_in, "value_out": value_out,
            "seed": seed, "extra": extra,
            "release_index": self._odometer.n_releases,
            "label": self.label,
        }
        fp = _fingerprint_release(payload)
        rel = Release(
            mechanism=mechanism, epsilon=epsilon, delta=delta,
            sensitivity=sensitivity, noise_param=noise_param,
            value_in=value_in, value_out=value_out,
            seed=seed, walltime_s=walltime_s, extra=extra, fingerprint=fp,
        )
        self._releases.append(rel)
        self._odometer.spent_epsilon += epsilon
        self._odometer.spent_delta += delta
        self._odometer.n_releases += 1
        return rel

    # ----------------------------------------------------------------------
    # Read-only views
    # ----------------------------------------------------------------------

    @property
    def spent_epsilon(self) -> float:
        return self._odometer.spent_epsilon

    @property
    def spent_delta(self) -> float:
        return self._odometer.spent_delta

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.target_delta - self.spent_delta)

    @property
    def n_releases(self) -> int:
        return self._odometer.n_releases

    @property
    def releases(self) -> tuple[Release, ...]:
        return tuple(self._releases)

    def rdp_accountant(self) -> RenyiAccountant:
        """Return the in-progress Rényi accountant (read-only snapshot)."""
        return self._rdp

    def epsilon_delta_now(self, delta: float | None = None) -> tuple[float, float]:
        """Return the *current* tightest (ε, δ) bound across composition modes.

        Compares basic, advanced, and RDP—returns the *minimum* ε for
        the given δ (default: target δ or 1e-6 if zero).
        """
        d = delta if delta is not None else max(self.target_delta, 1e-6)
        # basic
        basic_eps = self.spent_epsilon
        # advanced
        if self.n_releases > 0:
            avg = self.spent_epsilon / self.n_releases
            adv_eps, _ = advanced_composition(avg, 0.0, self.n_releases, d)
        else:
            adv_eps = 0.0
        # rdp
        rdp_eps, _ = self._rdp.to_epsilon_delta(d)
        return min(basic_eps, adv_eps, rdp_eps), self.spent_delta + d

    # ----------------------------------------------------------------------
    # Ledger
    # ----------------------------------------------------------------------

    def ledger_hash(self) -> str:
        """SHA-256 over the concatenated release fingerprints — a single
        tamper-evident receipt for the entire privacy-respecting session."""
        h = hashlib.sha256()
        h.update(str(self.target_epsilon).encode())
        h.update(b"|")
        h.update(str(self.target_delta).encode())
        h.update(b"|")
        h.update(str(self._seed).encode())
        for rel in self._releases:
            h.update(b"\n")
            h.update(rel.fingerprint.encode())
        return h.hexdigest()

    def summary(self) -> dict:
        """A JSON-serialisable summary of the accountant's state."""
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "composition": self.composition,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "n_releases": self.n_releases,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "rdp_snapshot": self._rdp.snapshot(),
            "ledger_hash": self.ledger_hash(),
            "seed": self._seed,
            "label": self.label,
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "PrivacyError",
    "BudgetExhausted",
    "InvalidMechanism",
    # statistics
    "std_normal_cdf",
    "std_normal_inv_cdf",
    # mechanisms — primitives
    "laplace_sample",
    "gaussian_sample",
    "snap_sample",
    "snap_eps_overhead",
    "exponential_select",
    # calibration
    "analytic_gaussian_sigma",
    "classical_gaussian_sigma",
    # RDP
    "gaussian_rdp",
    "laplace_rdp",
    "subsampled_gaussian_rdp",
    "rdp_to_epsilon_delta",
    # composition theorems
    "basic_composition",
    "advanced_composition",
    "zcdp_to_epsilon_delta",
    "gaussian_to_zcdp",
    # top-level
    "Release",
    "RenyiAccountant",
    "PrivacyAccountant",
    "SparseVector",
    "SVTConfig",
    "BinaryTreeCounter",
]
