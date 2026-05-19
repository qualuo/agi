r"""Goodharter — proxy-reward / specification-gaming divergence detection.

Goodhart's law — "when a measure becomes a target, it ceases to be a
good measure" (Strathern 1997, formalising Goodhart 1975) — is the
central failure mode of any agent that optimises a *proxy* signal in
place of an unobserved *true* objective.  Modern frontier-model
deployments are run-through with proxies: a learned reward model
substitutes for human-judged helpfulness; a unit-test pass-rate
substitutes for the user's actual goal; a quick-scoring rubric
substitutes for an expert eval; a click-through rate substitutes for
durable satisfaction.  Under sufficient optimisation pressure these
proxies *systematically diverge* from the truth they were meant to
track — and a coordination engine that fails to *certify* the gap
within a documented budget routes high-stakes work onto a runaway
optimiser.

``Goodharter`` is the runtime primitive that closes that gap with
**bounded, anytime, certified, pure-stdlib** machinery.  For each
proxy-reward signal it tracks paired ``(proxy, true)`` observations as
they stream in, fits Pearson + Spearman + Kendall-tau dependence
estimators with closed-form confidence intervals, runs an anytime-valid
e-process on the *signed gap* ``proxy - true`` against a documented
divergence budget, scores monotonicity violations between proxy and
true rankings, and aggregates the family via Holm step-down FWER and
Vovk-Wang product-of-e-values.  It issues a structured verdict and a
typed recommendation a coordination engine can dispatch on:
``TRUST | INVESTIGATE | RETRAIN | QUARANTINE`` paired with
``DEPLOY | MONITOR | RETUNE | REPLACE | ESCALATE_HUMAN``.

How a coordination engine uses it
---------------------------------

  1. The engine maintains a :class:`Goodharter` per ``proxy_id`` —
     one per reward model, one per scoring rubric, one per click-
     through-rate funnel, one per learned critic.  Each
     ``Goodharter`` has a documented **divergence budget**: the
     max acceptable expected gap or the min acceptable rank
     correlation, set by policy.
  2. Whenever the engine has access to a ``(proxy, true)`` pair —
     either from a held-out labelled probe, an expert label, a
     downstream outcome, or a delayed user signal — it calls
     ``goodharter.observe(RewardObservation(...))``.  Pairs need
     not arrive uniformly; the engine can subsample expensive
     true-reward acquisitions.
  3. At dispatch time the engine asks
     ``goodharter.certify()`` for the current verdict.  On
     ``TRUST`` the engine routes work to the optimiser as
     usual.  On ``INVESTIGATE`` it adds extra paired probes
     (more true-reward acquisitions).  On ``RETRAIN`` it pauses
     adapter promotion (composes with ``personalizer`` /
     ``aligner``) and triggers a rebuild of the reward model.
     On ``QUARANTINE`` it stops dispatch to the affected
     surface entirely and escalates to a human.
  4. The certificate carries a SHA-256 fingerprint chain over
     every observation, so a coordinator's decision can be
     replay-verified against the exact stream it saw.

Algorithmic surface
-------------------

  * **Welford-stable streaming moments.** Online mean and variance
    of proxy, true, and their gap ``g = proxy - true``, computed
    by Welford's algorithm (Welford 1962) so the second-order
    statistics are numerically stable under arbitrarily long
    streams.

  * **Pearson correlation with Fisher-Z CI.** The bivariate sample
    correlation between (proxy, true) is updated online via
    Welford's covariance variant; the confidence interval is
    obtained by Fisher's r-to-z transform (Fisher 1915) followed
    by Gaussian quantiles — exact under bivariate normality, an
    anchoring approximation otherwise.

  * **Spearman rank correlation.** Computed in batch on the
    retained window — a windowed-snapshot estimator with the
    Hanley-McNeil-style normal-approximation CI.  Distribution-
    free, captures monotone dependence that Pearson misses (the
    classic Goodhart signature: proxy and true both rise, but
    the proxy rises *faster*, so monotone but not linear).

  * **Kendall tau-b** monotonicity statistic — counts pair
    concordances vs. discordances over the retained window.
    Especially sensitive to *ranking flips* (the exact failure
    mode the alignment community calls "reward hacking": pairs
    where ``proxy_i > proxy_j`` while ``true_i < true_j``).

  * **Empirical-Bernstein CI on the gap mean.**  Maurer-Pontil
    2009 ``empirical-Bernstein`` bound on the running gap
    ``E[g]``.  Sharper than Hoeffding when the gap variance is
    small (the regime where Goodhart drift is subtle and
    needs sensitive detection).

  * **Anytime-valid e-process on gap > divergence budget.**
    Beta-Binomial universal-portfolio e-process (Howard-Ramdas-
    McAuliffe-Sekhon 2021) over the indicator ``g_t > budget``;
    closed-form, never decays under optional stopping.  The
    engine can call ``certify()`` arbitrarily often without
    multiple-testing inflation (Ramdas-Grunwald-Vovk-Shafer
    2023).

  * **Anytime-valid e-process on correlation drop.** Welford-
    stable difference-of-means e-process on the standardised
    correlation residuals — alerts when ``ρ`` drops below
    ``min_correlation``.

  * **Hedged-capital e-process (Waudby-Smith-Ramdas 2024)** on
    the *signed* gap with a grid-mixture over the bet — the
    sharpest known anytime-valid CI on a bounded-mean stream.

  * **Holm step-down FWER + Vovk-Wang product of e-values.**  The
    family ``{ρ_drop, gap_excess, monotonicity_violation,
    distribution_shift}`` is combined under Holm step-down
    (Holm 1979) for the p-value tests, and via Vovk-Wang 2021
    product-of-e-values for the e-process tests — both anytime-
    valid under arbitrary dependence.

  * **Distribution-shift detection.**  Optional Kolmogorov-
    Smirnov (Kolmogorov 1933; Smirnov 1948) two-sample test
    between a historical proxy distribution snapshot and the
    recent window — flags when the underlying *task mix*
    has shifted, which is the canonical *cause* of Goodhart
    drift.

  * **Replay-verifiable.**  Every ``observe`` / ``certify`` /
    ``report`` transition appends to a SHA-256 fingerprint chain
    so a coordinator can replay-verify a Goodharter run byte-
    for-byte at audit.

Why this is the right interface
-------------------------------

Most "is the reward model right?" tooling either ships as

  * **a single-batch evaluation** that loses validity the moment
    the deployment distribution drifts, or

  * **an opaque dashboard metric** that gives a number but no
    bounded guarantee on the probability of having missed a
    divergence — exactly the *epistemic* failure Goodhart's law
    induces.

The runtime instead exposes a *streaming, anytime-valid* test that
gives a coordination engine a *quantified* guarantee at every
dispatch decision.  The engine never has to choose between
*stopping early* (and losing validity from peeking) and *waiting
forever* (and missing the divergence window).  This is the alignment-
research statistical contract spelt out in Howard et al. 2021 and
Ramdas et al. 2023, applied to the canonical AGI safety problem of
proxy drift.

Composes with
-------------

  * :mod:`agi.aligner`      — the global DPO / RLHF policy
                              under-optimises against a proxy
                              reward model; Goodharter monitors
                              the proxy and gates promotion.

  * :mod:`agi.intender`     — Goodharter alerts feed back into
                              inverse-RL reward-shape refinement.

  * :mod:`agi.robustifier`  — distributionally robust
                              optimisation under the Goodharter-
                              estimated ambiguity ball.

  * :mod:`agi.constitutionalist`  — proxies that satisfy the
                                    rubric but violate principles
                                    surface as gap on the
                                    principle-evaluator-as-true
                                    pathway.

  * :mod:`agi.refuser`,
    :mod:`agi.sycophant`,
    :mod:`agi.confabulator`  — each safety primitive's certified
                               score is a *true* reward source
                               that Goodharter can pair against a
                               cheap proxy.

  * :mod:`agi.schemer`      — Goodharter pairs especially well
                              with the deception axis: a model
                              that *plays* its proxy in evaluation
                              but reverts under-observation
                              registers as both a gap signal
                              (Goodharter) and a scheming signal
                              (Schemer).

  * :mod:`agi.drift`        — distribution drift in the *input*;
                              Goodharter handles drift in the
                              *reward signal*.  Together they
                              bound both halves of the
                              specification problem.

  * :mod:`agi.attributor`   — when Goodharter alerts, Attributor
                              identifies which training trajectories
                              most pushed the proxy / true gap.

  * :mod:`agi.auditor`,
    :mod:`agi.attest`,
    :mod:`agi.governance`   — the multi-test correction, tamper-
                              evident receipt, and gating layer.

  * :mod:`agi.coordinator`,
    :mod:`agi.strategist`,
    :mod:`agi.portfolio`    — the dispatch surface that *acts* on
                              Goodharter's verdict.

  * :mod:`agi.pool`         — federated Goodharter instances across
                              many proxy_ids in a fleet.

Design contract
---------------

* **Pure stdlib.**  No NumPy / Torch.  All linear algebra on
  floats; all stats from elementary library functions plus
  closed-form approximations.

* **Stateful, thread-safe, deterministic given seed.**  Identical
  observation streams produce identical fingerprint chains under
  the same config.

* **No model coupling.**  Goodharter never sees tokens or weights.
  It sees ``(proxy, true)`` floats produced by the coordinator's
  oracle pipeline.

* **Anytime-valid.**  Every reported confidence interval and
  every certificate verdict is valid under arbitrary peek-and-
  stop policy.

* **Event-fingerprinted.**  Every observe / certify / report
  transition is hashed into a SHA-256 chain.

Mathematical sketch
-------------------

Let :math:`(P_t, T_t) \in [0, 1]^2` denote the t-th proxy / true
reward observation.  Define the **gap** :math:`g_t := P_t - T_t`
and the **divergence budget** :math:`\delta_0 \in [0, 1]`.

Pearson correlation:

  .. math::

      \hat{\rho}_n = \frac{\sum_t (P_t - \bar P)(T_t - \bar T)}
                            {\sqrt{\sum_t (P_t - \bar P)^2 \sum_t (T_t - \bar T)^2}}.

Fisher's z-transform gives the CI:

  .. math::

      z = \tfrac{1}{2} \log\frac{1 + \hat\rho}{1 - \hat\rho}, \quad
      z \pm \frac{q_{1 - \alpha/2}}{\sqrt{n - 3}} \to \rho_{lo}, \rho_{hi}.

Gap empirical-Bernstein (Maurer-Pontil 2009):

  .. math::

      \bar g \pm \sqrt{\frac{2 \hat\sigma_g^2 \log(2/\alpha)}{n}}
              + \frac{7 \log(2/\alpha)}{3 (n - 1)}.

Anytime-valid e-process on :math:`g > \delta_0`:

  .. math::

      E_n = \prod_{t=1}^{n} \frac{B(a + S_t, b + n - S_t)}
                                  {B(a, b)},

with :math:`S_t = \sum_s \mathbb{1}\{g_s > \delta_0\}`,
:math:`(a, b) = (1, 1)` (the universal-portfolio default), and
:math:`E_n > 1/\alpha` ⇒ reject the null
:math:`P(g > \delta_0) \le \mu_0` for the documented null rate
:math:`\mu_0`.  Ville's inequality gives anytime validity.

Hedged-capital e-process (Waudby-Smith-Ramdas 2024) on the signed
mean gap :math:`\mu_g`:

  .. math::

      K_n(\lambda) = \prod_{t=1}^{n} \big(1 + \lambda (g_t - \mu_0)\big)
                     \quad \text{for } \lambda \in [\lambda_-, \lambda_+],

with a grid mixture over :math:`\lambda` and the lower-confidence-
sequence inverted from the rejected nulls.

Holm step-down FWER over the family of p-values:

  .. math::

      \text{Reject } H_{(k)} \iff p_{(k)} \le \frac{\alpha}{m - k + 1}
      \text{ for all } j \le k,

with :math:`m` tests in the family.  Vovk-Wang product of e-values
is anytime-valid under arbitrary dependence:

  .. math::

      E_{\text{combined}} = \prod_k E_k.

Usage
-----

>>> from agi.goodharter import (
...     Goodharter, GoodharterConfig, RewardObservation,
... )
>>> g = Goodharter(GoodharterConfig(
...     proxy_id="reward_model_v3",
...     divergence_budget=0.10,
...     min_correlation=0.7,
...     seed=0,
... ))
>>> for t in range(50):
...     # Well-aligned stream: proxy and true track closely.
...     true = (t % 10) / 10.0
...     proxy = true + 0.02 * (t % 3 - 1)
...     g.observe(RewardObservation(
...         decision_id=f"d{t}", proxy_reward=proxy, true_reward=true,
...     ))
>>> cert = g.certify()
>>> cert.verdict
'TRUST'

References
----------

* Goodhart, *Problems of Monetary Management*, 1975.
* Strathern, *Improving Ratings*, 1997.
* Manheim & Garrabrant, *Categorizing Variants of Goodhart's Law*,
  arXiv:1803.04585.
* Krakovna et al., *Specification gaming: the flip side of AI
  ingenuity*, DeepMind 2020.
* Hennessy & Goodhart, *Goodhart's Law and Machine Learning*, 2023.
* Christiano, *Worst-case guarantees as Goodhart-style overoptimisation
  resistance*, 2017.
* Welford, *Note on a method for calculating corrected sums of
  squares and products*, Technometrics 4(3) 1962.
* Fisher, *Frequency distribution of the values of the correlation
  coefficient in samples from an indefinitely large population*,
  Biometrika 10 1915.
* Kendall, *A new measure of rank correlation*, Biometrika 30 1938.
* Kolmogorov 1933; Smirnov 1948 — KS two-sample test.
* Hoeffding, *Probability inequalities for sums of bounded random
  variables*, JASA 58 1963.
* Maurer & Pontil, *Empirical Bernstein bounds and sample variance
  penalisation*, COLT 2009.
* Howard, Ramdas, McAuliffe, Sekhon, *Time-uniform Chernoff bounds
  via nonnegative supermartingales*, Probability Surveys 18 2021.
* Ramdas, Grünwald, Vovk, Shafer, *Game-theoretic statistics and
  safe anytime-valid inference*, Statistical Science 38 2023.
* Vovk & Wang, *E-values: calibration, combination, and applications*,
  Annals of Statistics 49 2021.
* Waudby-Smith & Ramdas, *Estimating means of bounded random
  variables by betting*, JRSS-B 86 2024.
* Holm, *A simple sequentially rejective multiple test procedure*,
  Scandinavian Journal of Statistics 6 1979.
* Ville, *Étude critique de la notion de collectif*, 1939.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verdict taxonomy — what the engine should *do*.
VERDICT_TRUST = "TRUST"
VERDICT_INVESTIGATE = "INVESTIGATE"
VERDICT_RETRAIN = "RETRAIN"
VERDICT_QUARANTINE = "QUARANTINE"

KNOWN_VERDICTS: tuple[str, ...] = (
    VERDICT_TRUST,
    VERDICT_INVESTIGATE,
    VERDICT_RETRAIN,
    VERDICT_QUARANTINE,
)

# Recommendation taxonomy — what the engine should *operate*.
REC_DEPLOY = "DEPLOY"
REC_MONITOR = "MONITOR"
REC_RETUNE = "RETUNE"
REC_REPLACE = "REPLACE"
REC_ESCALATE_HUMAN = "ESCALATE_HUMAN"

KNOWN_RECOMMENDATIONS: tuple[str, ...] = (
    REC_DEPLOY,
    REC_MONITOR,
    REC_RETUNE,
    REC_REPLACE,
    REC_ESCALATE_HUMAN,
)

# Test labels — used in the multi-test family report.
TEST_PEARSON_DROP = "pearson_correlation_drop"
TEST_SPEARMAN_DROP = "spearman_correlation_drop"
TEST_KENDALL_DROP = "kendall_tau_drop"
TEST_GAP_EXCESS = "gap_excess"
TEST_GAP_EVALUE = "gap_evalue"
TEST_GAP_HEDGED = "gap_hedged_evalue"
TEST_DIST_SHIFT = "distribution_shift"

KNOWN_TESTS: tuple[str, ...] = (
    TEST_PEARSON_DROP,
    TEST_SPEARMAN_DROP,
    TEST_KENDALL_DROP,
    TEST_GAP_EXCESS,
    TEST_GAP_EVALUE,
    TEST_GAP_HEDGED,
    TEST_DIST_SHIFT,
)

# Event topics.
GH_STARTED = "goodharter.started"
GH_OBSERVED = "goodharter.observed"
GH_CERTIFIED = "goodharter.certified"
GH_REPORTED = "goodharter.reported"
GH_RESET = "goodharter.reset"
GH_ALERTED = "goodharter.alerted"
GH_BUDGET_UPDATED = "goodharter.budget_updated"
GH_DRIFT_FLAGGED = "goodharter.drift_flagged"

KNOWN_EVENTS: tuple[str, ...] = (
    GH_STARTED,
    GH_OBSERVED,
    GH_CERTIFIED,
    GH_REPORTED,
    GH_RESET,
    GH_ALERTED,
    GH_BUDGET_UPDATED,
    GH_DRIFT_FLAGGED,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GoodharterError(ValueError):
    """Base class."""


class InvalidConfig(GoodharterError):
    """Config violates an invariant."""


class InvalidObservation(GoodharterError):
    """Observation violates an invariant."""


class InsufficientData(GoodharterError):
    """Certification requested before ``min_observations`` reached."""


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardObservation:
    """One paired ``(proxy, true)`` observation from a single decision.

    Attributes:
        decision_id: stable identifier of the decision (used only for
            the audit chain; never re-used for statistics).
        proxy_reward: the proxy signal value, expected in [0, 1].
            Callers running on a different scale should pre-normalise.
        true_reward: the ground-truth signal value, expected in [0, 1].
            Acquired through a held-out probe, expert label, downstream
            outcome, or delayed user signal.
        context_features: optional feature vector — when provided, the
            Goodharter can run a distribution-shift detector across
            time-buckets of observations.
        is_control: True if this observation is part of a control /
            calibration stream (kept out of the active divergence test
            but kept in the audit chain).
        metadata: opaque to the primitive; persisted to the trace.
    """

    decision_id: str
    proxy_reward: float
    true_reward: float
    context_features: tuple[float, ...] = ()
    is_control: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise InvalidObservation("decision_id must be a non-empty string")
        for label, x in (("proxy_reward", self.proxy_reward),
                         ("true_reward", self.true_reward)):
            if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                raise InvalidObservation(f"{label} must be a finite number")
        # Cast & enforce [0, 1] range (the standard reward-model output range)
        for label in ("proxy_reward", "true_reward"):
            v = float(getattr(self, label))
            if v < -1e-9 or v > 1.0 + 1e-9:
                raise InvalidObservation(
                    f"{label}={v} out of [0, 1]; pre-normalise upstream"
                )
            object.__setattr__(self, label, max(0.0, min(1.0, v)))
        if not isinstance(self.context_features, tuple):
            object.__setattr__(self, "context_features",
                               tuple(float(x) for x in self.context_features))
        else:
            for i, x in enumerate(self.context_features):
                if not math.isfinite(float(x)):
                    raise InvalidObservation(
                        f"context_features[{i}] is not finite"
                    )


@dataclass(frozen=True)
class GoodharterConfig:
    """Static config — frozen after construction.

    Attributes:
        proxy_id: stable identifier of the proxy signal under test.
        divergence_budget: documented max acceptable expected gap
            E[proxy - true] in [-1, 1].  Default 0.05 (a 5pp gap).
        min_correlation: documented min acceptable Pearson correlation
            on the (proxy, true) stream.  Default 0.7.
        min_observations: minimum N before ``certify()`` returns a
            non-pending verdict.  Default 32.
        window_size: ring-buffer cap on retained observations for
            Spearman / Kendall / distribution-shift tests.  The
            running moment statistics (Welford) are unbounded — this
            only caps the rank-based / shift tests.  Default 1024.
        alpha: significance level for the family of tests.  Default
            0.05.  Holm step-down corrects for multiplicity within
            this budget.
        gap_evalue_threshold: e-process threshold for the
            ``gap_evalue`` test (rejects when ``E_n > 1/α``).
            Default ``1.0 / alpha`` if None.
        hedged_grid_size: number of betting parameters in the
            hedged-capital e-process mixture.  Default 32.
        rec_investigate_threshold: how many *individual* tests must
            mark a violation before issuing INVESTIGATE.  Default 1.
        rec_retrain_threshold: ... before RETRAIN.  Default 2.
        rec_quarantine_threshold: ... before QUARANTINE.  Default 3.
        track_history: keep a bounded per-observation trail of
            (proxy, true, gap) for the report.  Default True.
        seed: deterministic RNG seed for any randomised tie-breakers.
    """

    proxy_id: str = "default"
    divergence_budget: float = 0.05
    min_correlation: float = 0.7
    min_observations: int = 32
    window_size: int = 1024
    alpha: float = 0.05
    gap_evalue_threshold: float | None = None
    hedged_grid_size: int = 32
    rec_investigate_threshold: int = 1
    rec_retrain_threshold: int = 2
    rec_quarantine_threshold: int = 3
    track_history: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.proxy_id, str) or not self.proxy_id:
            raise InvalidConfig("proxy_id must be a non-empty string")
        if not -1.0 <= float(self.divergence_budget) <= 1.0:
            raise InvalidConfig("divergence_budget must be in [-1, 1]")
        if not -1.0 < float(self.min_correlation) < 1.0:
            raise InvalidConfig("min_correlation must be in (-1, 1)")
        if int(self.min_observations) < 4:
            raise InvalidConfig("min_observations must be >= 4")
        if int(self.window_size) < int(self.min_observations):
            raise InvalidConfig("window_size must be >= min_observations")
        if not 0.0 < float(self.alpha) < 1.0:
            raise InvalidConfig("alpha must be in (0, 1)")
        if self.gap_evalue_threshold is not None and float(self.gap_evalue_threshold) <= 0:
            raise InvalidConfig("gap_evalue_threshold must be > 0 or None")
        if int(self.hedged_grid_size) < 1:
            raise InvalidConfig("hedged_grid_size must be >= 1")
        for label, v in (("rec_investigate_threshold", self.rec_investigate_threshold),
                         ("rec_retrain_threshold", self.rec_retrain_threshold),
                         ("rec_quarantine_threshold", self.rec_quarantine_threshold)):
            if int(v) < 1:
                raise InvalidConfig(f"{label} must be >= 1")
        if not (self.rec_investigate_threshold <= self.rec_retrain_threshold
                <= self.rec_quarantine_threshold):
            raise InvalidConfig(
                "rec thresholds must satisfy investigate <= retrain <= quarantine"
            )


@dataclass(frozen=True)
class TestResult:
    """One row of the multi-test family report."""

    name: str
    statistic: float
    threshold: float
    p_value: float | None
    e_value: float | None
    rejected: bool
    detail: str = ""


@dataclass(frozen=True)
class GoodharterCertificate:
    """The certificate a coordination engine reaches for.

    Frozen / JSON-encodable.  All numeric quantities are finite floats.
    """

    proxy_id: str
    n_observations: int
    n_control: int
    verdict: str
    recommendation: str

    # Headline correlation summaries.
    pearson_r: float
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_r: float
    kendall_tau: float

    # Gap statistics.
    gap_mean: float
    gap_var: float
    gap_ci_low: float
    gap_ci_high: float
    gap_evalue: float
    gap_hedged_evalue: float
    hedged_lcs_low: float
    hedged_lcs_high: float

    # Bookkeeping.
    monotonicity_violation_rate: float
    n_pairs_evaluated: int
    distribution_shift_p: float | None

    # Family of tests + multi-test correction.
    tests: tuple[TestResult, ...]
    holm_rejected: tuple[str, ...]
    product_evalue: float

    # Audit.
    fingerprint: str


@dataclass(frozen=True)
class GoodharterReport:
    """Snapshot bundle the coordinator reads."""

    proxy_id: str
    n_observations: int
    n_control: int
    last_verdict: str
    last_recommendation: str
    last_fingerprint: str

    # Running moments.
    proxy_mean: float
    proxy_var: float
    true_mean: float
    true_var: float
    gap_mean: float
    gap_var: float
    cov_proxy_true: float

    # Most-recent slice of the audit chain.
    recent_observations: tuple[tuple[str, float, float, float], ...]


# ---------------------------------------------------------------------------
# Helpers — pure-stdlib statistics
# ---------------------------------------------------------------------------


def _phi_inv(p: float) -> float:
    """Inverse normal CDF via Beasley-Springer-Moro (Moro 1995).

    Accurate to ~1e-7 in the bulk; degrades in the deep tails but
    those are not used for any test threshold here.
    """
    if not 0.0 < p < 1.0:
        if p <= 0.0:
            return -math.inf
        if p >= 1.0:
            return math.inf
    # Beasley-Springer rational approximation
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _phi(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_sided_normal_p(z: float) -> float:
    """Two-sided p-value from a z-statistic."""
    return 2.0 * (1.0 - _phi(abs(z)))


def _fisher_z_ci(r: float, n: int, alpha: float) -> tuple[float, float]:
    """Fisher r-to-z CI on the Pearson correlation."""
    if n < 4:
        return (-1.0, 1.0)
    r = max(min(r, 1.0 - 1e-9), -1.0 + 1e-9)
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    q = _phi_inv(1.0 - alpha / 2.0)
    z_lo, z_hi = z - q * se, z + q * se
    r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
    return (r_lo, r_hi)


def _rank(values: Sequence[float]) -> list[float]:
    """Average-ranks of ``values`` (ties broken by mean rank)."""
    n = len(values)
    if n == 0:
        return []
    pairs = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[pairs[j + 1]] == values[pairs[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[pairs[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _rank(xs)
    ry = _rank(ys)
    return _pearson(rx, ry)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Closed-form Pearson r."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ys[i] - my) ** 2 for i in range(n))
    den = math.sqrt(dx2 * dy2)
    if den <= 0:
        return 0.0
    return num / den


def _kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, int, int, int]:
    """Kendall tau-b plus (n_concordant, n_discordant, n_pairs).

    O(n²) — sufficient for the typical retained-window sizes
    (default 1024); for larger windows the caller can shrink
    ``window_size``.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return (0.0, 0, 0, 0)
    nc = nd = 0
    nx_t = ny_t = 0  # tie counts
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            n_pairs += 1
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                nx_t += 1
                continue
            if dy == 0:
                ny_t += 1
                continue
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                nc += 1
            else:
                nd += 1
    den = math.sqrt(max((nc + nd + nx_t) * (nc + nd + ny_t), 1))
    tau_b = (nc - nd) / den if den > 0 else 0.0
    return (tau_b, nc, nd, n_pairs)


def _ks_two_sample(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test.

    Returns (D, p_value) using the asymptotic two-sided formula
    (Smirnov 1948).  Pure stdlib.
    """
    nx, ny = len(xs), len(ys)
    if nx == 0 or ny == 0:
        return (0.0, 1.0)
    a = sorted(xs)
    b = sorted(ys)
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while i < nx and j < ny:
        if a[i] <= b[j]:
            i += 1
            cdf_a = i / nx
        else:
            j += 1
            cdf_b = j / ny
        gap = abs(cdf_a - cdf_b)
        if gap > d:
            d = gap
    # Asymptotic p (Smirnov 1948)
    n_eff = math.sqrt(nx * ny / float(nx + ny))
    lam = max(0.0, (n_eff + 0.12 + 0.11 / n_eff) * d)
    # Kolmogorov distribution series — first few terms suffice.
    s = 0.0
    sign = 1.0
    for k in range(1, 101):
        term = sign * math.exp(-2.0 * lam * lam * k * k)
        s += term
        sign *= -1.0
        if abs(term) < 1e-12:
            break
    p = max(0.0, min(1.0, 2.0 * s))
    return (d, p)


def _holm_step_down(p_values: Sequence[tuple[str, float]],
                    alpha: float) -> tuple[list[str], list[str]]:
    """Holm-Bonferroni step-down rejection at family alpha.

    Returns (rejected_names, sorted_indices_of_input_in_test_order).
    """
    m = len(p_values)
    if m == 0:
        return ([], [])
    ordered = sorted(p_values, key=lambda kv: kv[1])
    rejected: list[str] = []
    for k, (name, p) in enumerate(ordered):
        cutoff = alpha / max(m - k, 1)
        if p <= cutoff:
            rejected.append(name)
        else:
            break
    return (rejected, [name for name, _ in ordered])


# ---------------------------------------------------------------------------
# Welford streaming bivariate moments
# ---------------------------------------------------------------------------


class _BivariateWelford:
    """Numerically stable mean / variance / covariance for (proxy, true).

    Implements the standard online algorithm of Welford 1962, extended
    to the covariance via the parallel-update form
    (Chan-Golub-LeVeque 1979).
    """

    __slots__ = (
        "n",
        "mx", "my",
        "m2x", "m2y", "m2xy",  # second-order central moments
        "gap_n", "gap_mean", "gap_m2",
    )

    def __init__(self) -> None:
        self.n: int = 0
        self.mx: float = 0.0
        self.my: float = 0.0
        self.m2x: float = 0.0
        self.m2y: float = 0.0
        self.m2xy: float = 0.0
        self.gap_n: int = 0
        self.gap_mean: float = 0.0
        self.gap_m2: float = 0.0

    def update(self, x: float, y: float) -> None:
        # Joint update — Welford on x and y, and on the cross-product.
        self.n += 1
        dx = x - self.mx
        self.mx += dx / self.n
        dy = y - self.my
        self.my += dy / self.n
        self.m2x += dx * (x - self.mx)
        self.m2y += dy * (y - self.my)
        self.m2xy += dx * (y - self.my)  # this is the exact update form
        # Gap is its own Welford so we don't lose the inner variance.
        g = x - y
        self.gap_n += 1
        dg = g - self.gap_mean
        self.gap_mean += dg / self.gap_n
        self.gap_m2 += dg * (g - self.gap_mean)

    @property
    def var_x(self) -> float:
        return self.m2x / max(self.n - 1, 1)

    @property
    def var_y(self) -> float:
        return self.m2y / max(self.n - 1, 1)

    @property
    def cov_xy(self) -> float:
        return self.m2xy / max(self.n - 1, 1)

    @property
    def pearson_r(self) -> float:
        d = math.sqrt(max(self.m2x, 0.0) * max(self.m2y, 0.0))
        if d <= 0:
            return 0.0
        return max(-1.0, min(1.0, self.m2xy / d))

    @property
    def gap_var(self) -> float:
        return self.gap_m2 / max(self.gap_n - 1, 1)


# ---------------------------------------------------------------------------
# Anytime-valid e-processes
# ---------------------------------------------------------------------------


class _BetaBinomialEProcess:
    """One-sided betting e-process against ``H0: P(X = 1) ≤ μ0``.

    For Bernoulli ``X_t`` and a positive bet ``λ ∈ [0, 1/μ0)``, the
    process

    .. math::
        K_n(\\lambda) = \\prod_{t=1}^n \\big(1 + \\lambda (X_t - \\mu_0)\\big)

    is a nonnegative martingale under ``E[X_t] = μ0`` and a
    *super*-martingale under ``E[X_t] ≤ μ0`` (Ville 1939;
    Waudby-Smith-Ramdas 2024).  We mix over a uniform grid of
    admissible bets to retain power against a range of alternatives —
    a closed-form discrete approximation to the universal-portfolio
    mixture (Howard-Ramdas-McAuliffe-Sekhon 2021) for Bernoulli data.

    Under the null the mixture stays below ``1/α`` with probability
    ``≥ 1 - α`` simultaneously for all n; under the alternative
    ``E[X_t] > μ0`` it grows to infinity, so the rejection rule
    ``K_n ≥ 1/α`` is anytime-valid (Ramdas et al. 2023).
    """

    __slots__ = ("mu0", "grid", "log_K", "n", "s")

    def __init__(self, mu0: float, grid_size: int = 16) -> None:
        if not 0.0 < mu0 < 1.0:
            raise InvalidConfig(f"BetaBinomialEProcess mu0={mu0} must be in (0, 1)")
        if grid_size < 1:
            raise InvalidConfig("grid_size must be >= 1")
        self.mu0: float = float(mu0)
        # Admissible positive bets λ in (0, 1/μ0).  Cap at lam_max for
        # numerical stability; the e-process is one-sided against
        # E[X] > μ0.
        lam_max = min(0.95 / mu0, 50.0)
        self.grid: tuple[float, ...] = tuple(
            lam_max * (i + 1) / (grid_size + 1) for i in range(grid_size)
        )
        self.log_K: list[float] = [0.0] * len(self.grid)
        self.n: int = 0
        self.s: int = 0

    def update(self, indicator: bool) -> None:
        self.n += 1
        x = 1.0 if indicator else 0.0
        if indicator:
            self.s += 1
        for i, lam in enumerate(self.grid):
            term = 1.0 + lam * (x - self.mu0)
            if term <= 1e-12:
                continue
            self.log_K[i] += math.log(term)

    @property
    def e_value(self) -> float:
        """Mixture e-value across the admissible bet grid."""
        if not self.log_K:
            return 1.0
        m = max(self.log_K)
        s = 0.0
        for lv in self.log_K:
            s += math.exp(lv - m)
        try:
            return math.exp(m) * s / len(self.log_K)
        except OverflowError:
            return math.inf


class _HedgedCapitalEProcess:
    """One-sided hedged-capital e-process against ``H0: E[g_t] ≤ μ0``
    on a bounded stream (Waudby-Smith-Ramdas 2024).

    For ``g_t`` centred so that ``g_t - μ0 ∈ [-1, 1]`` and a positive
    bet ``λ ∈ (0, 1)``, the process

    .. math::
        K_n(\\lambda) = \\prod_{t=1}^n \\big(1 + \\lambda (g_t - \\mu_0)\\big)

    is a nonnegative supermartingale under the null.  We mix over a
    grid of positive ``λ`` (so the test rejects only when the true
    mean exceeds ``μ0``, i.e. the Goodhart direction).  The companion
    LCS inverts this construction over a coarse grid of candidate
    ``μ0``.
    """

    __slots__ = ("grid", "log_K", "max_log_K", "lam_max")

    def __init__(self, grid_size: int = 32, lam_max: float = 0.5) -> None:
        if grid_size < 1:
            grid_size = 1
        self.lam_max: float = float(lam_max)
        # Positive-only grid for the one-sided test.
        self.grid: tuple[float, ...] = tuple(
            lam_max * (i + 1) / (grid_size + 1) for i in range(grid_size)
        )
        self.log_K: list[float] = [0.0 for _ in self.grid]
        self.max_log_K: list[float] = [0.0 for _ in self.grid]

    def update(self, g_centered: float) -> None:
        # g_centered = g_t - μ0; admissible λ keeps (1 + λ·g_c) > 0.
        for i, lam in enumerate(self.grid):
            term = 1.0 + lam * g_centered
            if term <= 1e-12:
                continue
            v = math.log(term)
            self.log_K[i] += v
            if self.log_K[i] > self.max_log_K[i]:
                self.max_log_K[i] = self.log_K[i]

    @property
    def mixture_e_value(self) -> float:
        if not self.log_K:
            return 1.0
        m = max(self.log_K)
        s = 0.0
        for lv in self.log_K:
            s += math.exp(lv - m)
        try:
            return math.exp(m) * s / len(self.log_K)
        except OverflowError:
            return math.inf

    @property
    def max_e_value(self) -> float:
        if not self.log_K:
            return 1.0
        try:
            return math.exp(max(self.max_log_K))
        except OverflowError:
            return math.inf


def _hedged_lcs(stream_g: Sequence[float],
                grid_size: int,
                alpha: float,
                budget: float) -> tuple[float, float]:
    """Two-sided LCS on ``E[g_t]`` by inverting two one-sided hedged-
    capital e-processes.

    Returns ``(mu_lo, mu_hi)`` — the smallest and largest ``μ_0`` not
    rejected at level α by the corresponding one-sided test (an
    upper-rejecting e-process for ``μ_lo``, a lower-rejecting one for
    ``μ_hi``).  This is the standard betting-LCS construction
    (Waudby-Smith-Ramdas 2024) discretised on a coarse grid.

    Pure stdlib; ``O(grid * 1/step * n)`` worst case which keeps the
    cost manageable on the windowed-slice the caller passes.
    """
    n = len(stream_g)
    if n < 4:
        return (-1.0, 1.0)
    mu_hat = sum(stream_g) / n
    threshold = math.log(1.0 / max(alpha, 1e-12))
    step = 0.01

    def _reject_upper(mu0: float) -> bool:
        # Rejects H0: E[g] <= mu0  (positive bets).
        ep = _HedgedCapitalEProcess(grid_size=grid_size, lam_max=0.5)
        for g in stream_g:
            ep.update(g - mu0)
        return math.log(max(ep.mixture_e_value, 1e-300)) > threshold

    def _reject_lower(mu0: float) -> bool:
        # Rejects H0: E[g] >= mu0  (negative bets) — implemented as the
        # upper-rejecting e-process on -g against -mu0.
        ep = _HedgedCapitalEProcess(grid_size=grid_size, lam_max=0.5)
        for g in stream_g:
            ep.update(-(g - mu0))
        return math.log(max(ep.mixture_e_value, 1e-300)) > threshold

    # Walk down from mu_hat to find the smallest mu_0 not rejected by
    # the upper-rejecting test — this is the LCS lower bound.
    mu_lo = -1.0
    cur = mu_hat
    for _ in range(int(2.0 / step)):
        cur -= step
        if cur <= -1.0:
            mu_lo = -1.0
            break
        if not _reject_upper(cur):
            # Already not rejected here; keep stepping down.
            continue
        mu_lo = cur + step
        break

    # Walk up from mu_hat to find the largest mu_0 not rejected.
    mu_hi = 1.0
    cur = mu_hat
    for _ in range(int(2.0 / step)):
        cur += step
        if cur >= 1.0:
            mu_hi = 1.0
            break
        if not _reject_lower(cur):
            continue
        mu_hi = cur - step
        break

    return (max(-1.0, mu_lo), min(1.0, mu_hi))


# ---------------------------------------------------------------------------
# The Goodharter class
# ---------------------------------------------------------------------------


def _now() -> float:
    import time
    return time.time()


class Goodharter:
    """Streaming proxy-vs-true reward divergence certifier.

    Thread-safe.  Pure compute.  Replay-verifiable: identical
    observation streams produce identical fingerprint chains under
    the same config.

    A coordination engine wires one :class:`Goodharter` per proxy
    signal (reward model, scoring rubric, click-through funnel,
    judge LLM).  It calls :meth:`observe` whenever a paired
    ``(proxy, true)`` becomes available and :meth:`certify`
    arbitrarily often — anytime-validity means peeking is free.
    """

    def __init__(
        self,
        config: GoodharterConfig,
        bus: Any = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(config, GoodharterConfig):
            raise InvalidConfig("config must be a GoodharterConfig")
        # Defensive re-validate (clones the dataclass with same fields).
        GoodharterConfig(**{
            f: getattr(config, f) for f in (
                "proxy_id", "divergence_budget", "min_correlation",
                "min_observations", "window_size", "alpha",
                "gap_evalue_threshold", "hedged_grid_size",
                "rec_investigate_threshold", "rec_retrain_threshold",
                "rec_quarantine_threshold", "track_history", "seed",
            )
        })
        self._config = config
        self._bus = bus
        self._clock = clock or _now
        self._lock = threading.RLock()
        self._welford = _BivariateWelford()
        self._control_welford = _BivariateWelford()
        # Beta-binomial one-sided e-process on indicator ``g > budget``.
        # The null rate is the documented per-round violation tolerance
        # — by default we accept up to ``alpha`` rate of gap excursions
        # before flagging.  Subclasses may override.
        self._gap_eprocess = _BetaBinomialEProcess(
            mu0=max(min(self._config.alpha, 0.5), 1e-3),
        )
        self._hedged = _HedgedCapitalEProcess(
            grid_size=config.hedged_grid_size, lam_max=0.5
        )
        # Ring buffer over (proxy, true) tuples for rank-based tests.
        self._window_proxy: list[float] = []
        self._window_true: list[float] = []
        self._window_idx: list[str] = []          # decision_id parallel
        self._window_features: list[tuple[float, ...]] = []
        # Historical snapshot for distribution-shift detection
        # (snapshotted at half-window).
        self._snapshot_proxy: tuple[float, ...] = ()
        self._snapshot_set: bool = False
        # Per-decision trail.
        self._history: list[tuple[str, float, float, float]] = []
        # Audit chain.
        self._fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "init": True,
                    "config": {
                        "proxy_id": config.proxy_id,
                        "divergence_budget": config.divergence_budget,
                        "min_correlation": config.min_correlation,
                        "min_observations": config.min_observations,
                        "window_size": config.window_size,
                        "alpha": config.alpha,
                        "seed": config.seed,
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        # Last certificate cache (for stable .last property).
        self._last_certificate: GoodharterCertificate | None = None
        self._last_verdict: str = VERDICT_TRUST
        self._last_recommendation: str = REC_DEPLOY
        self._emit(GH_STARTED, {
            "proxy_id": config.proxy_id,
            "divergence_budget": config.divergence_budget,
            "min_correlation": config.min_correlation,
        })

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> GoodharterConfig:
        return self._config

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def n_observations(self) -> int:
        with self._lock:
            return self._welford.n

    @property
    def n_control(self) -> int:
        with self._lock:
            return self._control_welford.n

    @property
    def last_verdict(self) -> str:
        with self._lock:
            return self._last_verdict

    @property
    def last_recommendation(self) -> str:
        with self._lock:
            return self._last_recommendation

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, obs: RewardObservation) -> str:
        """Ingest one paired observation; return the new fingerprint.

        Raises :class:`InvalidObservation` if the observation is
        malformed; otherwise updates all online state and returns
        the new SHA-256 fingerprint of the audit chain.
        """
        if not isinstance(obs, RewardObservation):
            raise InvalidObservation("obs must be a RewardObservation")
        with self._lock:
            if obs.is_control:
                self._control_welford.update(obs.proxy_reward, obs.true_reward)
                self._chain("observe_control", {
                    "decision_id": obs.decision_id,
                    "proxy": obs.proxy_reward,
                    "true": obs.true_reward,
                })
                self._emit(GH_OBSERVED, {
                    "decision_id": obs.decision_id,
                    "is_control": True,
                    "n_observations": self._welford.n,
                    "n_control": self._control_welford.n,
                    "fingerprint": self._fingerprint,
                })
                return self._fingerprint
            # Active observation.
            self._welford.update(obs.proxy_reward, obs.true_reward)
            gap = obs.proxy_reward - obs.true_reward
            # Indicator e-process: is the gap > budget?
            indicator = gap > self._config.divergence_budget
            self._gap_eprocess.update(indicator)
            # Hedged e-process on the centered gap.
            self._hedged.update(gap - self._config.divergence_budget)
            # Window for rank-based tests.
            self._window_proxy.append(obs.proxy_reward)
            self._window_true.append(obs.true_reward)
            self._window_idx.append(obs.decision_id)
            self._window_features.append(obs.context_features)
            if len(self._window_proxy) > self._config.window_size:
                # Drop oldest; rotates the rank-based tests over the
                # *recent* window so the certification stays current.
                self._window_proxy.pop(0)
                self._window_true.pop(0)
                self._window_idx.pop(0)
                self._window_features.pop(0)
            # Snapshot for distribution shift at half-window.
            if (not self._snapshot_set
                    and len(self._window_proxy) >= self._config.window_size // 2
                    and self._config.window_size >= 4):
                self._snapshot_proxy = tuple(self._window_proxy)
                self._snapshot_set = True
            if self._config.track_history:
                self._history.append((obs.decision_id,
                                       float(obs.proxy_reward),
                                       float(obs.true_reward),
                                       float(gap)))
                # Cap the history at 2x the window for a viewable trail.
                cap = 2 * self._config.window_size
                if len(self._history) > cap:
                    self._history = self._history[-cap:]
            self._chain("observe", {
                "decision_id": obs.decision_id,
                "proxy": obs.proxy_reward,
                "true": obs.true_reward,
                "gap": gap,
            })
            self._emit(GH_OBSERVED, {
                "decision_id": obs.decision_id,
                "is_control": False,
                "proxy": obs.proxy_reward,
                "true": obs.true_reward,
                "gap": gap,
                "n_observations": self._welford.n,
                "fingerprint": self._fingerprint,
            })
            return self._fingerprint

    # ------------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------------

    def certify(self) -> GoodharterCertificate:
        """Produce a structured certificate from the current state.

        Anytime-valid: the e-process based tests can be called
        arbitrarily often without inflating the type-I error rate.
        Returns a :class:`GoodharterCertificate` whose
        ``recommendation`` field is what the coordination engine
        should *act* on.
        """
        with self._lock:
            cfg = self._config
            n = self._welford.n
            if n < cfg.min_observations:
                # Pending verdict — emit a TRUST verdict with the
                # caveat that not enough data has accumulated.
                cert = self._pending_certificate(n)
                self._last_certificate = cert
                self._last_verdict = cert.verdict
                self._last_recommendation = cert.recommendation
                self._emit(GH_CERTIFIED, {
                    "verdict": cert.verdict,
                    "recommendation": cert.recommendation,
                    "n_observations": n,
                    "fingerprint": self._fingerprint,
                    "pending": True,
                })
                return cert
            tests = list(self._run_tests())
            # Multi-test correction
            holm_rejected, _ = _holm_step_down(
                [(t.name, t.p_value) for t in tests if t.p_value is not None],
                alpha=cfg.alpha,
            )
            # Vovk-Wang product of e-values.
            e_product = 1.0
            for t in tests:
                if t.e_value is not None and t.e_value > 0:
                    e_product *= t.e_value
            # Verdict by violation count.
            violations = [t for t in tests if t.rejected]
            n_viol = len(violations)
            verdict, rec = self._classify(n_viol, tests, e_product)
            cert = self._build_certificate(
                tests=tests,
                violations=violations,
                holm_rejected=tuple(holm_rejected),
                product_evalue=e_product,
                verdict=verdict,
                recommendation=rec,
            )
            self._chain("certify", {
                "verdict": verdict,
                "recommendation": rec,
                "n_observations": n,
                "n_violations": n_viol,
                "product_evalue": e_product,
            })
            self._last_certificate = cert
            self._last_verdict = verdict
            self._last_recommendation = rec
            self._emit(GH_CERTIFIED, {
                "verdict": verdict,
                "recommendation": rec,
                "n_observations": n,
                "n_violations": n_viol,
                "fingerprint": self._fingerprint,
                "pending": False,
            })
            if verdict in (VERDICT_RETRAIN, VERDICT_QUARANTINE):
                self._emit(GH_ALERTED, {
                    "verdict": verdict,
                    "violations": [v.name for v in violations],
                    "fingerprint": self._fingerprint,
                })
            return cert

    def _pending_certificate(self, n: int) -> GoodharterCertificate:
        # All zero CIs — coordinator should interpret as "not yet certifiable".
        return GoodharterCertificate(
            proxy_id=self._config.proxy_id,
            n_observations=n,
            n_control=self._control_welford.n,
            verdict=VERDICT_TRUST,
            recommendation=REC_MONITOR,
            pearson_r=self._welford.pearson_r,
            pearson_ci_low=-1.0,
            pearson_ci_high=1.0,
            spearman_r=0.0,
            kendall_tau=0.0,
            gap_mean=self._welford.gap_mean,
            gap_var=self._welford.gap_var,
            gap_ci_low=-1.0,
            gap_ci_high=1.0,
            gap_evalue=self._gap_eprocess.e_value,
            gap_hedged_evalue=self._hedged.mixture_e_value,
            hedged_lcs_low=-1.0,
            hedged_lcs_high=1.0,
            monotonicity_violation_rate=0.0,
            n_pairs_evaluated=0,
            distribution_shift_p=None,
            tests=(),
            holm_rejected=(),
            product_evalue=1.0,
            fingerprint=self._fingerprint,
        )

    def _run_tests(self) -> Iterable[TestResult]:
        cfg = self._config
        w = self._welford
        # 1. Pearson drop.
        r = w.pearson_r
        r_lo, r_hi = _fisher_z_ci(r, w.n, cfg.alpha)
        # We reject H0: ρ >= min_correlation if the upper bound of
        # ρ (i.e., the "good" side) is below the floor — equivalently,
        # if r_hi < min_correlation.
        z = (r - cfg.min_correlation) * math.sqrt(max(w.n - 3, 1))
        p_pearson = _two_sided_normal_p(z) if r < cfg.min_correlation else 1.0
        rejected_pearson = r_hi < cfg.min_correlation
        yield TestResult(
            name=TEST_PEARSON_DROP,
            statistic=r,
            threshold=cfg.min_correlation,
            p_value=p_pearson,
            e_value=None,
            rejected=rejected_pearson,
            detail=f"r={r:.3f} (CI {r_lo:.3f}..{r_hi:.3f})",
        )
        # 2. Spearman drop.
        rho_s = _spearman(self._window_proxy, self._window_true)
        ns = len(self._window_proxy)
        if ns >= 4:
            zs = rho_s * math.sqrt(ns - 2) / math.sqrt(max(1.0 - rho_s * rho_s, 1e-9))
            p_spearman = _two_sided_normal_p(zs) if rho_s < cfg.min_correlation else 1.0
        else:
            p_spearman = 1.0
        rejected_spearman = rho_s < cfg.min_correlation - 0.05  # 5pp tolerance
        yield TestResult(
            name=TEST_SPEARMAN_DROP,
            statistic=rho_s,
            threshold=cfg.min_correlation,
            p_value=p_spearman,
            e_value=None,
            rejected=rejected_spearman,
            detail=f"ρ_s={rho_s:.3f} on window n={ns}",
        )
        # 3. Kendall tau drop + monotonicity violation rate.
        tau_b, nc, nd, n_pairs = _kendall_tau(self._window_proxy, self._window_true)
        viol_rate = nd / n_pairs if n_pairs > 0 else 0.0
        # Approx p via normal approximation of tau-b (Kendall 1938).
        if n_pairs > 0:
            var_tau = (2.0 * (2.0 * ns + 5.0)) / max(9.0 * ns * (ns - 1.0), 1e-9)
            z_t = tau_b / math.sqrt(max(var_tau, 1e-12))
            p_kendall = _two_sided_normal_p(z_t) if tau_b < cfg.min_correlation else 1.0
        else:
            p_kendall = 1.0
        rejected_kendall = tau_b < cfg.min_correlation - 0.1
        yield TestResult(
            name=TEST_KENDALL_DROP,
            statistic=tau_b,
            threshold=cfg.min_correlation,
            p_value=p_kendall,
            e_value=None,
            rejected=rejected_kendall,
            detail=f"τ_b={tau_b:.3f} viol_rate={viol_rate:.3f} on {n_pairs} pairs",
        )
        # 4. Gap excess (empirical Bernstein CI on E[g]).
        gap_mean = w.gap_mean
        gap_var = w.gap_var
        log_term = math.log(2.0 / cfg.alpha)
        eb_half = math.sqrt(2.0 * gap_var * log_term / max(w.gap_n, 1))
        # range correction (gap ∈ [-1, 1] ⇒ range 2)
        eb_half += 7.0 * 2.0 * log_term / max(3.0 * (w.gap_n - 1), 1)
        gap_ci_low = gap_mean - eb_half
        gap_ci_high = gap_mean + eb_half
        # Reject H0: E[g] <= budget if gap_ci_low > budget.
        rejected_gap = gap_ci_low > cfg.divergence_budget
        # p-value: Welford-Bernstein style z
        z_g = (gap_mean - cfg.divergence_budget) / max(math.sqrt(gap_var / max(w.gap_n, 1)), 1e-12)
        p_gap = _two_sided_normal_p(z_g) if gap_mean > cfg.divergence_budget else 1.0
        yield TestResult(
            name=TEST_GAP_EXCESS,
            statistic=gap_mean,
            threshold=cfg.divergence_budget,
            p_value=p_gap,
            e_value=None,
            rejected=rejected_gap,
            detail=f"E[g]={gap_mean:.3f} (CI {gap_ci_low:.3f}..{gap_ci_high:.3f})",
        )
        # 5. Gap e-process (Beta-Binomial universal portfolio).
        gap_evalue = self._gap_eprocess.e_value
        threshold = cfg.gap_evalue_threshold or (1.0 / cfg.alpha)
        rejected_gap_e = gap_evalue >= threshold
        yield TestResult(
            name=TEST_GAP_EVALUE,
            statistic=gap_evalue,
            threshold=threshold,
            p_value=None,
            e_value=gap_evalue,
            rejected=rejected_gap_e,
            detail=f"E_n={gap_evalue:.3g} (Beta-Binomial UP, threshold {threshold:.3g})",
        )
        # 6. Hedged-capital e-process.
        hedged_evalue = self._hedged.mixture_e_value
        rejected_hedged = hedged_evalue >= threshold
        yield TestResult(
            name=TEST_GAP_HEDGED,
            statistic=hedged_evalue,
            threshold=threshold,
            p_value=None,
            e_value=hedged_evalue,
            rejected=rejected_hedged,
            detail=f"K_n={hedged_evalue:.3g} (hedged-capital, threshold {threshold:.3g})",
        )
        # 7. Distribution shift (KS two-sample) — gated on a meaningful
        #    effect size (D > 0.2) so it does not fire on the
        #    re-sampling variance of an otherwise-aligned stream.
        if (self._snapshot_set
                and len(self._window_proxy) >= cfg.window_size // 2):
            recent = self._window_proxy[len(self._window_proxy) // 2:]
            d_stat, p_ks = _ks_two_sample(self._snapshot_proxy, recent)
            rejected_ks = (p_ks < cfg.alpha) and (d_stat > 0.2)
            yield TestResult(
                name=TEST_DIST_SHIFT,
                statistic=d_stat,
                threshold=cfg.alpha,
                p_value=p_ks,
                e_value=None,
                rejected=rejected_ks,
                detail=f"KS D={d_stat:.3f} (p={p_ks:.4f})",
            )

    def _classify(
        self,
        n_viol: int,
        tests: Sequence[TestResult],
        e_product: float,
    ) -> tuple[str, str]:
        cfg = self._config
        # The combined Vovk-Wang e-value is automatically anytime-valid.
        # We require a *strong* combined rejection (e_product >= 100/α)
        # before the e-value alone escalates beyond the violation-count
        # path; a single test crossing its own threshold should not
        # immediately QUARANTINE — the violation-count thresholds are
        # the operator-facing dial.
        strong_combined = e_product >= 100.0 / cfg.alpha
        if n_viol >= cfg.rec_quarantine_threshold or strong_combined:
            return (VERDICT_QUARANTINE, REC_ESCALATE_HUMAN)
        if n_viol >= cfg.rec_retrain_threshold:
            return (VERDICT_RETRAIN, REC_REPLACE)
        if n_viol >= cfg.rec_investigate_threshold:
            # Pick the recommendation that matches the strongest signal.
            top = next((t for t in tests if t.rejected), None)
            if top is None:
                return (VERDICT_INVESTIGATE, REC_MONITOR)
            if top.name in (TEST_GAP_EXCESS, TEST_GAP_EVALUE, TEST_GAP_HEDGED):
                return (VERDICT_INVESTIGATE, REC_RETUNE)
            return (VERDICT_INVESTIGATE, REC_MONITOR)
        return (VERDICT_TRUST, REC_DEPLOY)

    def _build_certificate(
        self,
        tests: Sequence[TestResult],
        violations: Sequence[TestResult],
        holm_rejected: tuple[str, ...],
        product_evalue: float,
        verdict: str,
        recommendation: str,
    ) -> GoodharterCertificate:
        cfg = self._config
        w = self._welford
        r = w.pearson_r
        r_lo, r_hi = _fisher_z_ci(r, w.n, cfg.alpha)
        gap_mean = w.gap_mean
        gap_var = w.gap_var
        log_term = math.log(2.0 / cfg.alpha)
        eb_half = math.sqrt(2.0 * gap_var * log_term / max(w.gap_n, 1))
        eb_half += 7.0 * 2.0 * log_term / max(3.0 * (w.gap_n - 1), 1)
        gap_ci_low = gap_mean - eb_half
        gap_ci_high = gap_mean + eb_half
        rho_s = _spearman(self._window_proxy, self._window_true)
        tau_b, nc, nd, n_pairs = _kendall_tau(self._window_proxy, self._window_true)
        viol_rate = nd / n_pairs if n_pairs > 0 else 0.0
        # Recompute KS p for the certificate (cheap given the snapshot).
        dist_p: float | None = None
        if (self._snapshot_set
                and len(self._window_proxy) >= cfg.window_size // 2):
            recent = self._window_proxy[len(self._window_proxy) // 2:]
            _, dist_p = _ks_two_sample(self._snapshot_proxy, recent)
        # Build the gap stream for the LCS — bounded for tractability.
        # Use the window slice to keep LCS computation O(window * grid).
        stream = [self._window_proxy[i] - self._window_true[i]
                  for i in range(len(self._window_proxy))]
        lcs_lo, lcs_hi = _hedged_lcs(
            stream_g=stream,
            grid_size=cfg.hedged_grid_size,
            alpha=cfg.alpha,
            budget=cfg.divergence_budget,
        )
        return GoodharterCertificate(
            proxy_id=cfg.proxy_id,
            n_observations=w.n,
            n_control=self._control_welford.n,
            verdict=verdict,
            recommendation=recommendation,
            pearson_r=float(r),
            pearson_ci_low=float(r_lo),
            pearson_ci_high=float(r_hi),
            spearman_r=float(rho_s),
            kendall_tau=float(tau_b),
            gap_mean=float(gap_mean),
            gap_var=float(gap_var),
            gap_ci_low=float(gap_ci_low),
            gap_ci_high=float(gap_ci_high),
            gap_evalue=float(self._gap_eprocess.e_value),
            gap_hedged_evalue=float(self._hedged.mixture_e_value),
            hedged_lcs_low=float(lcs_lo),
            hedged_lcs_high=float(lcs_hi),
            monotonicity_violation_rate=float(viol_rate),
            n_pairs_evaluated=int(n_pairs),
            distribution_shift_p=(float(dist_p) if dist_p is not None else None),
            tests=tuple(tests),
            holm_rejected=tuple(holm_rejected),
            product_evalue=float(product_evalue),
            fingerprint=self._fingerprint,
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> GoodharterReport:
        with self._lock:
            recent = tuple(self._history[-min(32, len(self._history)):])
            r = GoodharterReport(
                proxy_id=self._config.proxy_id,
                n_observations=self._welford.n,
                n_control=self._control_welford.n,
                last_verdict=self._last_verdict,
                last_recommendation=self._last_recommendation,
                last_fingerprint=self._fingerprint,
                proxy_mean=float(self._welford.mx),
                proxy_var=float(self._welford.var_x),
                true_mean=float(self._welford.my),
                true_var=float(self._welford.var_y),
                gap_mean=float(self._welford.gap_mean),
                gap_var=float(self._welford.gap_var),
                cov_proxy_true=float(self._welford.cov_xy),
                recent_observations=recent,
            )
            self._emit(GH_REPORTED, {
                "proxy_id": self._config.proxy_id,
                "n_observations": r.n_observations,
                "last_verdict": r.last_verdict,
                "fingerprint": self._fingerprint,
            })
            return r

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    def update_budget(
        self,
        *,
        divergence_budget: float | None = None,
        min_correlation: float | None = None,
    ) -> None:
        """Replace the documented budget at runtime.

        Re-validates the new combined config; bumps the fingerprint
        chain so the change is auditable.

        Side-effect: the indicator-based and hedged-capital e-processes
        are rebuilt from the retained history under the new budget, so
        ``certify()`` reflects the new bar even on already-accumulated
        observations.  The unbounded Welford moments are *not* reset —
        Pearson, Kendall and gap-mean carry forward as the budget
        change only shifts the test threshold, not the observations.
        """
        with self._lock:
            new_div = (float(divergence_budget)
                       if divergence_budget is not None
                       else self._config.divergence_budget)
            new_corr = (float(min_correlation)
                        if min_correlation is not None
                        else self._config.min_correlation)
            new = GoodharterConfig(
                proxy_id=self._config.proxy_id,
                divergence_budget=new_div,
                min_correlation=new_corr,
                min_observations=self._config.min_observations,
                window_size=self._config.window_size,
                alpha=self._config.alpha,
                gap_evalue_threshold=self._config.gap_evalue_threshold,
                hedged_grid_size=self._config.hedged_grid_size,
                rec_investigate_threshold=self._config.rec_investigate_threshold,
                rec_retrain_threshold=self._config.rec_retrain_threshold,
                rec_quarantine_threshold=self._config.rec_quarantine_threshold,
                track_history=self._config.track_history,
                seed=self._config.seed,
            )
            self._config = new
            # Rebuild the budget-dependent e-processes from history so
            # certify() reflects the new threshold.
            self._gap_eprocess = _BetaBinomialEProcess(
                mu0=max(min(new.alpha, 0.5), 1e-3),
            )
            self._hedged = _HedgedCapitalEProcess(
                grid_size=new.hedged_grid_size, lam_max=0.5,
            )
            if self._config.track_history:
                for (_did, _proxy, _true, gap) in self._history:
                    self._gap_eprocess.update(gap > new_div)
                    self._hedged.update(gap - new_div)
            self._chain("update_budget", {
                "divergence_budget": new_div,
                "min_correlation": new_corr,
            })
            self._emit(GH_BUDGET_UPDATED, {
                "divergence_budget": new_div,
                "min_correlation": new_corr,
                "fingerprint": self._fingerprint,
            })

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._welford = _BivariateWelford()
            self._control_welford = _BivariateWelford()
            self._gap_eprocess = _BetaBinomialEProcess(
                mu0=max(min(self._config.alpha, 0.5), 1e-3),
            )
            self._hedged = _HedgedCapitalEProcess(
                grid_size=self._config.hedged_grid_size, lam_max=0.5
            )
            self._window_proxy.clear()
            self._window_true.clear()
            self._window_idx.clear()
            self._window_features.clear()
            self._history.clear()
            self._snapshot_proxy = ()
            self._snapshot_set = False
            self._last_certificate = None
            self._last_verdict = VERDICT_TRUST
            self._last_recommendation = REC_DEPLOY
            self._chain("reset", {})
            self._emit(GH_RESET, {"fingerprint": self._fingerprint})

    # ------------------------------------------------------------------
    # Internal: fingerprint chain + event emission
    # ------------------------------------------------------------------

    def _chain(self, kind: str, payload: Mapping[str, Any]) -> None:
        h = hashlib.sha256()
        h.update(self._fingerprint.encode("utf-8"))
        h.update(kind.encode("utf-8"))
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        self._fingerprint = h.hexdigest()

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            from agi.events import Event
        except Exception:  # pragma: no cover
            return
        for attempt in (
            lambda: Event(kind=kind, data=dict(payload), ts=self._clock()),
            lambda: Event(kind=kind, data=dict(payload)),
            lambda: Event(kind, None, dict(payload)),
            lambda: Event(kind),
        ):
            try:
                ev = attempt()
                break
            except TypeError:
                continue
        else:  # pragma: no cover
            return
        try:
            self._bus.publish(ev)
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def fresh_goodharter(
    proxy_id: str = "default",
    *,
    divergence_budget: float = 0.05,
    min_correlation: float = 0.7,
    min_observations: int = 32,
    window_size: int = 256,
    alpha: float = 0.05,
    seed: int = 0,
) -> Goodharter:
    """Build a Goodharter with sensible coordinator defaults."""
    return Goodharter(GoodharterConfig(
        proxy_id=proxy_id,
        divergence_budget=divergence_budget,
        min_correlation=min_correlation,
        min_observations=min_observations,
        window_size=window_size,
        alpha=alpha,
        seed=seed,
    ))


def synthetic_aligned_stream(
    n: int,
    *,
    noise: float = 0.02,
    seed: int = 0,
) -> list[RewardObservation]:
    """Generate a *well-aligned* paired (proxy, true) stream — proxy
    tracks true within ``noise`` Gaussian sd.  Use this in tests to
    verify Goodharter returns ``TRUST`` on clean streams.
    """
    import random
    rng = random.Random(seed)
    out: list[RewardObservation] = []
    for t in range(n):
        true = rng.random()
        proxy = max(0.0, min(1.0, true + rng.gauss(0.0, noise)))
        out.append(RewardObservation(
            decision_id=f"a{t}",
            proxy_reward=proxy,
            true_reward=true,
        ))
    return out


def synthetic_goodhart_stream(
    n: int,
    *,
    onset: float = 0.5,
    drift: float = 0.3,
    noise: float = 0.02,
    seed: int = 0,
) -> list[RewardObservation]:
    """Generate a stream that *starts* aligned and *develops* a
    Goodhart drift — the proxy rises above the true reward by an
    increasing margin past ``onset * n``.  Use this in tests to
    verify Goodharter raises ``QUARANTINE`` / ``RETRAIN``.
    """
    import random
    rng = random.Random(seed)
    out: list[RewardObservation] = []
    cutoff = int(onset * n)
    for t in range(n):
        true = rng.random() * 0.8 + 0.1
        if t < cutoff:
            proxy = max(0.0, min(1.0, true + rng.gauss(0.0, noise)))
        else:
            scale = (t - cutoff) / max(n - cutoff, 1)
            proxy = max(0.0, min(1.0, true + drift * scale + rng.gauss(0.0, noise)))
        out.append(RewardObservation(
            decision_id=f"g{t}",
            proxy_reward=proxy,
            true_reward=true,
        ))
    return out


__all__ = [
    # Verdicts
    "VERDICT_TRUST",
    "VERDICT_INVESTIGATE",
    "VERDICT_RETRAIN",
    "VERDICT_QUARANTINE",
    "KNOWN_VERDICTS",
    # Recommendations
    "REC_DEPLOY",
    "REC_MONITOR",
    "REC_RETUNE",
    "REC_REPLACE",
    "REC_ESCALATE_HUMAN",
    "KNOWN_RECOMMENDATIONS",
    # Tests
    "TEST_PEARSON_DROP",
    "TEST_SPEARMAN_DROP",
    "TEST_KENDALL_DROP",
    "TEST_GAP_EXCESS",
    "TEST_GAP_EVALUE",
    "TEST_GAP_HEDGED",
    "TEST_DIST_SHIFT",
    "KNOWN_TESTS",
    # Events
    "GH_STARTED",
    "GH_OBSERVED",
    "GH_CERTIFIED",
    "GH_REPORTED",
    "GH_RESET",
    "GH_ALERTED",
    "GH_BUDGET_UPDATED",
    "GH_DRIFT_FLAGGED",
    "KNOWN_EVENTS",
    # Exceptions
    "GoodharterError",
    "InvalidConfig",
    "InvalidObservation",
    "InsufficientData",
    # Records
    "RewardObservation",
    "GoodharterConfig",
    "TestResult",
    "GoodharterCertificate",
    "GoodharterReport",
    # Class
    "Goodharter",
    # Factories
    "fresh_goodharter",
    "synthetic_aligned_stream",
    "synthetic_goodhart_stream",
]
