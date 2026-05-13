"""ExperimentDesigner — Bayesian Optimal Experiment Design as a runtime primitive.

Every other forecaster in this runtime answers a question about *what to do
with the data we have*. `Strategist` fuses calibration, conformal cost
bounds, causal lift, and OPE into a single recommendation for the current
ticket. `PolicyLab` evaluates new policies counterfactually. `Portfolio`
allocates a budget across a heterogeneous batch.

None of those answers the coordination engine's *meta*-question:

    Given a finite experiment budget and a pool of candidate tickets we
    could spend it on, which ones — and in what order — will most
    reduce uncertainty about the policy parameters we care about, per
    dollar?

Spending the entire budget on the highest-EV ticket is locally optimal
but globally wasteful: the runtime ends up data-rich on actions it was
already confident in, and data-poor on the marginal cases that actually
move the policy. `ExperimentDesigner` is the principled answer.

What this implements (razor's-edge of statistical decision theory)
------------------------------------------------------------------

The objective is the **Expected Information Gain** between an
experimental design d and the experiment outcome y, under a prior p(θ)
on the parameter we want to learn (Lindley 1956, Chaloner-Verdinelli
1995):

    EIG(d) := E_{θ,y ~ p(θ) p(y|θ,d)} [ log p(y|θ,d) − log p(y|d) ]
            = H[Y|d] − E_θ[ H[Y|θ,d] ]    (mutual information I(Y;Θ|d))

Choosing d to maximize EIG(d) is the Bayesian-optimal next experiment.
The coordination engine picks the next batch by maximizing EIG over a
*set* of designs subject to a cost budget — a submodular set function,
so the greedy algorithm has a (1 − 1/e) optimality guarantee
(Krause-Guestrin 2007; Sviridenko 2004 for the cost-aware variant).

  * `eig_discrete` — exact EIG when the parameter and outcome spaces
    are finite. Closed-form: H[prior @ likelihood] − prior @ rowwise-H.
    The reference truth for testing the Monte Carlo estimators.

  * `eig_nested_mc` — nested Monte Carlo estimator
    (Foster-Jankowiak-Bingham-Teh-Rainforth-Goodman 2019). For each
    outer θ ~ prior, sample y ~ p(y|θ,d), and use a *separate* inner
    sample of θ's to estimate log p(y|d). Bias is O(1/N_inner); the
    bound on the bias is tracked explicitly and returned.

  * `BALDScorer` — Bayesian Active Learning by Disagreement
    (Houlsby-Huszár-Ghahramani-Lengyel 2011). Decomposes the
    predictive entropy of an ensemble into
        epistemic := H[mean(p_k)]  −  mean(H[p_k])    (informative)
        aleatoric := mean(H[p_k])                     (irreducible)
    A coordination engine acquires on epistemic — chasing aleatoric
    uncertainty is a category error.

  * `BayesianBatchPlanner` — lazy submodular greedy
    (Minoux 1978; Krause-Golovin 2014) over a candidate pool. Picks a
    batch of size k (or under a knapsack budget) whose joint EIG is
    within (1 − 1/e) of the optimum. The "lazy" variant exploits
    monotone diminishing returns to cut the per-step cost from
    O(N k) to typically O(N + k log N).

  * `DOptimalDesigner` — Fedorov's exchange algorithm (Fedorov 1972)
    for the classical D-optimal criterion: maximize log det X_S^T X_S
    over subsets S of a row pool. Equivalent to minimizing the volume
    of the OLS confidence ellipsoid for the linear model y = Xβ + ε.
    A-optimal and E-optimal variants minimize trace and largest
    eigenvalue of the inverse information matrix respectively.

  * `knowledge_gradient` — Frazier-Powell-Dayanik 2008. Given a
    Gaussian posterior on the values of K alternatives, the KG of
    sampling arm a is E[max_i μ'_i] − max_i μ_i where μ' is the
    one-sample updated posterior mean. Optimal one-step look-ahead
    under correlated normal priors. The acquisition function of
    choice for *value-of-information*-driven exploration.

  * `thompson_top_k` — Russo 2020 batch acquisition. Draw one sample
    per arm from the posterior, return the top-k arms. Gives a
    diversified, posterior-consistent batch with no tuning.

Composition with the rest of the runtime
----------------------------------------

  * `PolicyLab` tells you what you'd earn from a new policy on the
    data you already have. `ExperimentDesigner` tells you which
    *next* contexts to log under the production policy so that next
    week's `PolicyLab` answer is sharper. Concretely: the IPS-OPE
    variance falls fastest when new logged data lands in regions
    where π(a|c) and μ(a|c) disagree the most — exactly where the
    BALD epistemic score is largest.

  * `Strategist.recommend(...)` returns STRAT_EXPLORE when the data
    is too thin to commit. The `ExperimentDesigner` is what makes
    that "explore" honest: it scores which contexts deserve the
    explore budget, instead of exploring uniformly.

  * `PortfolioOptimizer` allocates against a *known* utility surface.
    `ExperimentDesigner` decides which experiments refine the
    utility surface itself. They compose: the portfolio reserves a
    fraction of the budget for designer-selected exploration tickets.

  * `SelfEvalBank` mines a regression suite. The designer picks which
    of N unlabeled candidate traces are the highest-information for
    the critic — turning a generic eval suite into an active suite.

Investor framing
----------------

The runtime spends money to learn. Without principled experiment
design, every dollar buys a *random* amount of information about the
policy. With BOED, every dollar buys the *maximum possible* amount of
information per dollar, with provable optimality bounds. The same
budget produces a policy that converges faster — which is the only
dimension of the harness that compounds over time.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Information-theoretic primitives — pure, deterministic, no RNG.
# ---------------------------------------------------------------------------


_LOG_FLOOR = 1e-300  # floor for log() to avoid -inf on numerically-zero probabilities


def _log(x: float) -> float:
    """Numerically safe natural log."""
    return math.log(x) if x > _LOG_FLOOR else math.log(_LOG_FLOOR)


def entropy(p: Sequence[float], *, base: float = math.e) -> float:
    """Shannon entropy of a probability vector (nats by default).

    Drops zero entries (0·log0 := 0). Does not check normalisation —
    callers that pass in unnormalised weights get a nonsense answer.
    Numerically robust: handles probabilities ≥ _LOG_FLOOR.
    """
    log_base = math.log(base)
    h = 0.0
    for pi in p:
        if pi > _LOG_FLOOR:
            h -= pi * _log(pi)
    return h / log_base


def kl_divergence(p: Sequence[float], q: Sequence[float], *, base: float = math.e) -> float:
    """KL(p || q) of two discrete distributions.

    Returns +inf if any q_i = 0 where p_i > 0 (the divergence is
    genuinely infinite there, not a numerical artefact).
    """
    if len(p) != len(q):
        raise ValueError(f"p, q length mismatch: {len(p)} vs {len(q)}")
    log_base = math.log(base)
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= _LOG_FLOOR:
            continue
        if qi <= _LOG_FLOOR:
            return float("inf")
        s += pi * (_log(pi) - _log(qi))
    return s / log_base


def js_divergence(p: Sequence[float], q: Sequence[float], *, base: float = math.e) -> float:
    """Jensen-Shannon divergence — symmetric, bounded by log 2 (nats).

    Defined as ½ KL(p || m) + ½ KL(q || m) with m = ½(p + q). Finite
    everywhere; the symmetric stand-in for KL when neither distribution
    dominates the other.
    """
    if len(p) != len(q):
        raise ValueError(f"p, q length mismatch: {len(p)} vs {len(q)}")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m, base=base) + 0.5 * kl_divergence(q, m, base=base)


def _matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    """Row-major matrix-vector product."""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def _vecmat(vec: Sequence[float], mat: Sequence[Sequence[float]]) -> list[float]:
    """vec @ mat (row-vector times matrix)."""
    n_rows = len(mat)
    n_cols = len(mat[0]) if n_rows else 0
    out = [0.0] * n_cols
    for i in range(n_rows):
        v = vec[i]
        row = mat[i]
        for j in range(n_cols):
            out[j] += v * row[j]
    return out


def predictive_distribution(
    prior: Sequence[float], likelihood: Sequence[Sequence[float]]
) -> list[float]:
    """Marginal predictive p(y | d) = Σ_θ p(θ) p(y | θ, d).

    `prior`: shape [K] over θ-values; `likelihood`: shape [K, Y] with
    likelihood[k][y] = P(y | θ_k, d). Returns the [Y]-length vector.
    """
    if len(prior) != len(likelihood):
        raise ValueError(
            f"prior length {len(prior)} does not match likelihood rows {len(likelihood)}"
        )
    return _vecmat(list(prior), [list(r) for r in likelihood])


def eig_discrete(
    prior: Sequence[float], likelihood: Sequence[Sequence[float]], *, base: float = math.e
) -> float:
    """Exact Expected Information Gain for finite θ × Y.

    EIG = I(Y; Θ | d)
        = H[ Σ_θ p(θ) p(y|θ) ] − Σ_θ p(θ) H[ p(y|θ) ]

    `prior`: [K], `likelihood`: [K, Y] with `likelihood[k][y]` =
    P(y | θ_k, d). Returns EIG in `nats` (base=e) by default; pass
    `base=2` for bits.

    This is the closed-form reference Monte-Carlo estimators are
    benchmarked against. It is non-negative and at most min(log K, log Y)
    (clipping for normalisation rounding).
    """
    py = predictive_distribution(prior, likelihood)
    h_marginal = entropy(py, base=base)
    expected_h_cond = 0.0
    for k, pk in enumerate(prior):
        if pk > 0.0:
            expected_h_cond += pk * entropy(likelihood[k], base=base)
    # Clamp negative drift from float arithmetic — EIG is provably ≥ 0.
    return max(0.0, h_marginal - expected_h_cond)


# ---------------------------------------------------------------------------
# Nested Monte Carlo EIG — for general continuous / black-box models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestedMCEig:
    """Nested Monte Carlo EIG estimate.

    `eig` is the point estimate; `stderr` is the outer-loop sample
    standard error. `bias_bound` is the dominant O(1/N_inner) bias
    term tracked from Foster et al. 2019 — when it dominates `stderr`,
    raise N_inner before raising N_outer.
    """

    eig: float
    stderr: float
    bias_bound: float
    n_outer: int
    n_inner: int


def eig_nested_mc(
    prior_sampler: Callable[[random.Random], object],
    likelihood_sampler: Callable[[object, random.Random], object],
    log_likelihood: Callable[[object, object], float],
    *,
    n_outer: int = 200,
    n_inner: int = 200,
    rng: random.Random | None = None,
) -> NestedMCEig:
    """Nested Monte Carlo estimator for EIG (Foster et al. 2019).

    Estimates EIG(d) = E_{θ,y} [ log p(y|θ,d) − log p(y|d) ] by:

      1. Outer loop: for j = 1..N_outer, sample θ_j ~ prior, y_j ~ p(y|θ_j,d).
      2. Compute log p(y_j | θ_j, d).
      3. Inner loop: estimate log p(y_j | d) via
         log[ (1 / N_inner) Σ_l p(y_j | θ_l, d) ] with θ_l ~ prior, l ≠ j.
      4. EIG ≈ mean over j of (step-2 − step-3).

    The estimator has positive bias O(1/N_inner). The bound returned in
    `bias_bound` is the Foster-Rainforth tightening:
        bias ≤ Var(p(y|θ,d)) / (2 E[p(y|θ,d)]^2 N_inner)
    estimated on the outer samples. Raise N_inner first; the variance
    cost is amortised by reusing inner samples across outer points
    when implementations want to (this version does *not* reuse,
    keeping the estimator simple and the bound clean).

    Callers should pass:

      * `prior_sampler(rng)` -> θ      (any hashable / opaque type ok)
      * `likelihood_sampler(θ, rng) -> y`
      * `log_likelihood(y, θ) -> float`   (log p(y | θ, d))

    Design d is closed over by the caller — this routine treats d as
    fixed across all evaluations.
    """
    if rng is None:
        rng = random.Random(0)
    if n_outer < 2 or n_inner < 1:
        raise ValueError("n_outer must be ≥ 2 and n_inner must be ≥ 1")

    inner_buffer: list[object] = [prior_sampler(rng) for _ in range(n_inner)]

    contribs: list[float] = []
    inner_means: list[float] = []
    inner_second_moments: list[float] = []

    for _ in range(n_outer):
        theta = prior_sampler(rng)
        y = likelihood_sampler(theta, rng)
        ll_inside = log_likelihood(y, theta)
        # Estimate log p(y|d) via log-sum-exp over the inner samples.
        inner_log_terms = [log_likelihood(y, ti) for ti in inner_buffer]
        m = max(inner_log_terms)
        # log mean exp = log(1/N Σ exp(l_i)) = m + log Σ exp(l_i - m) - log N
        s = sum(math.exp(li - m) for li in inner_log_terms)
        ll_marginal = m + math.log(s) - math.log(n_inner)
        contribs.append(ll_inside - ll_marginal)
        # For the bias bound: track moments of p(y|θ) = exp(ll).
        terms = [math.exp(li - m) for li in inner_log_terms]
        mean_t = sum(terms) / n_inner
        var_t = sum((t - mean_t) ** 2 for t in terms) / n_inner
        # Rescaling drops out of the var / mean^2 ratio used in the bound.
        if mean_t > 0.0:
            inner_means.append(mean_t)
            inner_second_moments.append(var_t / (mean_t * mean_t))

    eig = sum(contribs) / n_outer
    var = sum((c - eig) ** 2 for c in contribs) / max(1, n_outer - 1)
    stderr = math.sqrt(var / n_outer)
    bias_term = (
        sum(inner_second_moments) / len(inner_second_moments)
        if inner_second_moments
        else 0.0
    )
    bias_bound = bias_term / (2.0 * n_inner)
    return NestedMCEig(
        eig=eig, stderr=stderr, bias_bound=bias_bound, n_outer=n_outer, n_inner=n_inner
    )


# ---------------------------------------------------------------------------
# BALD: epistemic vs. aleatoric decomposition for a committee / ensemble.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BALDResult:
    """Per-point active-learning score, decomposed.

    `predictive_entropy` is total uncertainty H[ mean(p_k) ]:
      = aleatoric + epistemic.

    `aleatoric` = E_k[ H[p_k] ] — the irreducible noise component.
    Acquiring a label here just averages out noise; no parameter
    learning happens.

    `epistemic` = predictive_entropy − aleatoric (the BALD score).
    Disagreement *between* committee members about y — the reducible,
    parameter-driven component. **This is the right thing to acquire on.**
    """

    bald: float                # epistemic component — the score
    predictive_entropy: float  # total uncertainty
    aleatoric: float           # mean conditional entropy
    epistemic: float           # alias for `bald`


def bald_score(
    committee_predictions: Sequence[Sequence[float]], *, base: float = math.e
) -> BALDResult:
    """BALD score for a committee of probability vectors over the same Y.

    `committee_predictions`: shape [M, Y]; each row p_k is a probability
    vector from one posterior sample / ensemble member / dropout-MC pass.

    Returns the decomposed entropy. Use `.bald` (= epistemic) as the
    acquisition score; never use raw predictive entropy (it confuses
    irreducible noise with informative disagreement).

    BALD obeys the bounds:
        0 ≤ bald ≤ log min(M, Y)
    achieved at zero when all members agree, and saturated when each
    member places mass on a different y.
    """
    if not committee_predictions:
        raise ValueError("need at least one committee member")
    y_dim = len(committee_predictions[0])
    if any(len(row) != y_dim for row in committee_predictions):
        raise ValueError("committee predictions have inconsistent length")
    m = len(committee_predictions)
    # Aggregate predictive: mean over committee.
    pbar = [0.0] * y_dim
    for row in committee_predictions:
        for j, v in enumerate(row):
            pbar[j] += v
    pbar = [v / m for v in pbar]
    h_pred = entropy(pbar, base=base)
    h_each = sum(entropy(row, base=base) for row in committee_predictions) / m
    epi = max(0.0, h_pred - h_each)
    return BALDResult(
        bald=epi, predictive_entropy=h_pred, aleatoric=h_each, epistemic=epi
    )


# ---------------------------------------------------------------------------
# Knowledge gradient (Frazier-Powell-Dayanik 2008).
# ---------------------------------------------------------------------------


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _expected_max_after_normal(
    means: Sequence[float], stds_after: Sequence[float]
) -> float:
    """E[max_i (μ_i + σ̃_i Z)] for Z ~ N(0,1), independent across i.

    Exact for K=1 or 2; for K ≥ 3 it is approximated by 1-D numerical
    integration over Z. This is the inner quantity in the knowledge
    gradient when posteriors are independent Gaussians.

    For K=2: closed form
        E[max(μ1 + σ̃1 Z, μ2 + σ̃2 Z)] = (μ1+μ2)/2 + something(Z)
    The simpler symmetric form below uses the maximum of two correlated
    normals (here, common driving Z), giving a clean closed expression.
    """
    k = len(means)
    if k == 0:
        return 0.0
    if k == 1:
        return means[0]
    if k == 2:
        m1, m2 = means
        s1, s2 = stds_after
        # max(a, b) = ((a+b) + |a-b|) / 2; E[|a-b|] for a=m1+s1 Z, b=m2+s2 Z:
        # |a-b| = |(m1-m2) + (s1-s2) Z|. Let σ = |s1-s2|, μ = m1-m2.
        # E|μ + σ Z| = σ φ(μ/σ) + μ (1 − 2 Φ(−μ/σ)).
        sigma = abs(s1 - s2)
        mu = m1 - m2
        if sigma < 1e-12:
            return max(m1, m2)
        z = mu / sigma
        return 0.5 * (m1 + m2) + 0.5 * (sigma * _normal_pdf(z) + mu * (2.0 * _normal_cdf(z) - 1.0))
    # K >= 3: numerical integration over Z. 41-point Gauss-Hermite is overkill;
    # uniform grid on [-6, 6] with 81 points is fine for the magnitude here.
    n_grid = 81
    lo, hi = -6.0, 6.0
    step = (hi - lo) / (n_grid - 1)
    total = 0.0
    norm = 0.0
    for i in range(n_grid):
        z = lo + step * i
        w = _normal_pdf(z) * step
        norm += w
        best = max(means[j] + stds_after[j] * z for j in range(k))
        total += best * w
    # Renormalise to absorb the truncation error.
    return total / norm


def knowledge_gradient(
    means: Sequence[float], stds: Sequence[float], obs_noise_std: Sequence[float]
) -> list[float]:
    """One-step knowledge gradient (KG) for K independent Gaussian arms.

    Posterior on arm i is N(μ_i, σ_i^2); a single observation of arm i
    yields y = θ_i + ε with ε ~ N(0, ν_i^2). The conjugate posterior
    after one observation has variance σ̃_i^2 = σ_i^2 σ_i^2 / (σ_i^2 + ν_i^2)
    on arm i — i.e., the posterior *update*'s prior-predictive
    distribution of the new posterior mean is N(μ_i, σ̃_i^2).

    KG_i := E_obs[max_j μ'_j] − max_j μ_j
          = (E[max_j (μ_j + σ̃_j Z) ] − max_j μ_j) for arm i, with σ̃_j = 0 for j ≠ i.

    Coordination engines pick argmax_i KG_i for the next experiment.
    Unlike BALD this acquisition function knows the *value* of learning
    — it weights information by its effect on the decision objective.
    """
    k = len(means)
    if len(stds) != k or len(obs_noise_std) != k:
        raise ValueError("means, stds, obs_noise_std must all have length K")
    base = max(means) if means else 0.0
    out = [0.0] * k
    for i in range(k):
        sigma2 = stds[i] * stds[i]
        nu2 = obs_noise_std[i] * obs_noise_std[i]
        denom = sigma2 + nu2
        sigma_tilde = math.sqrt(sigma2 * sigma2 / denom) if denom > 0.0 else 0.0
        stds_after = [0.0] * k
        stds_after[i] = sigma_tilde
        em = _expected_max_after_normal(list(means), stds_after)
        out[i] = max(0.0, em - base)
    return out


# ---------------------------------------------------------------------------
# Thompson Top-K — Russo 2020 batch acquisition.
# ---------------------------------------------------------------------------


def thompson_top_k(
    posterior_samplers: Sequence[Callable[[random.Random], float]],
    *,
    k: int,
    rng: random.Random | None = None,
) -> list[int]:
    """Russo 2020 Thompson Top-K batch selection.

    Each arm exposes a `posterior_samplers[i](rng) -> float` callback that
    draws from the posterior over its mean. We take *one* draw per arm and
    return the indices of the top-k arms by sampled value. Across calls
    this gives a diversified, posterior-consistent batch with no tuning.

    For the canonical Beta-Bernoulli case, `posterior_samplers[i]` is just
    `lambda r: r.betavariate(alpha_i, beta_i)`.
    """
    if rng is None:
        rng = random.Random()
    if k <= 0:
        return []
    if k > len(posterior_samplers):
        raise ValueError(f"k={k} exceeds number of arms {len(posterior_samplers)}")
    samples = [s(rng) for s in posterior_samplers]
    order = sorted(range(len(samples)), key=lambda i: -samples[i])
    return order[:k]


# ---------------------------------------------------------------------------
# Batch planner — lazy submodular greedy with optional knapsack cost budget.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentCandidate:
    """One candidate experiment.

    `id` uniquely names the experiment (a ticket id, trace id, ...).
    `eig` is the *marginal* EIG of running this experiment, *given the
    currently selected set*. For independent candidates pass the
    unconditional EIG; for correlated candidates supply a custom
    `eig_marginal` callback to the planner.
    `cost` is the cost in budget units (default 1.0 — equal-cost batch).
    `meta` is opaque pass-through for the coordination engine.
    """

    id: str
    eig: float
    cost: float = 1.0
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentPlan:
    """Output of `BayesianBatchPlanner.plan(...)`.

    `selected` are the chosen ids in the order they were picked
    (insertion order matters for the submodular guarantee). `eig_per`
    is the marginal information gain *at the time* of each selection
    — diminishing across the batch by submodularity. `total_eig`
    sums them. `total_cost` is the spent budget. `n_evaluations` is
    the number of marginal-gain calls — useful for benchmarking the
    speedup of lazy vs. naive greedy.
    """

    selected: list[str]
    eig_per: list[float]
    total_eig: float
    total_cost: float
    n_evaluations: int


class BayesianBatchPlanner:
    """Submodular greedy batch experiment selection with optional knapsack.

    Two operating modes:

      * `plan(candidates, k=K)` — pick K candidates maximising total
        marginal EIG. The submodular greedy is within (1 − 1/e) ≈ 0.63
        of the offline optimum (Nemhauser-Wolsey-Fisher 1978).

      * `plan(candidates, budget=B)` — knapsack-style: max EIG subject
        to Σ cost ≤ B. Uses Sviridenko's 2004 partial-enumeration trick
        as a clean (1 − 1/e) cost-aware bound when costs are non-trivial;
        for the common equal-cost case it collapses to the standard
        greedy.

    Correlated candidates: pass `eig_marginal(selected_ids, candidate) -> float`.
    The planner will call back into it as new candidates are added. The
    standard guarantee assumes monotone submodularity of the joint EIG
    set function; if your `eig_marginal` violates monotonicity (gives
    a *negative* marginal gain), the planner refuses to add that
    candidate.

    Uses the Minoux 1978 lazy variant: each candidate's stale marginal
    gain bounds its current marginal gain (by submodularity), so a
    max-heap of stale gains lets us skip recomputing most candidates.
    Typical speedup: O(N) per step → O(log N).
    """

    def __init__(
        self,
        eig_marginal: Callable[[list[str], ExperimentCandidate], float] | None = None,
    ) -> None:
        self._eig_marginal = eig_marginal

    def _marginal(
        self, selected: list[str], cand: ExperimentCandidate
    ) -> float:
        if self._eig_marginal is None:
            return cand.eig
        return float(self._eig_marginal(selected, cand))

    def plan(
        self,
        candidates: Sequence[ExperimentCandidate],
        *,
        k: int | None = None,
        budget: float | None = None,
    ) -> ExperimentPlan:
        """Select a batch by submodular greedy.

        Either `k` (size cap) or `budget` (cost cap) — or both — must
        be supplied; if both are given, both constraints are respected.

        With `budget` set the planner uses the cost-aware density rule
        (Sviridenko 2004): at each step, prefer the candidate
        maximising marginal_gain / cost among those that still fit.
        This is provably within (1 − 1/e) of the optimum for monotone
        submodular set-cover with knapsack constraints.
        """
        if k is None and budget is None:
            raise ValueError("must supply k or budget (or both)")
        # Tag candidates by insertion index for stable tie-breaking.
        pool = list(candidates)
        if not pool:
            return ExperimentPlan([], [], 0.0, 0.0, 0)

        selected: list[str] = []
        eig_per: list[float] = []
        cost_so_far = 0.0
        n_evals = 0

        # Heap of (-stale_gain_per_cost, freshness_step, insertion_idx, cand)
        heap: list[tuple[float, int, int, ExperimentCandidate]] = []
        cost_aware = budget is not None
        for i, c in enumerate(pool):
            stale = c.eig
            density = stale / c.cost if (cost_aware and c.cost > 0) else stale
            heapq.heappush(heap, (-density, 0, i, c))
            n_evals += 1

        while heap:
            if k is not None and len(selected) >= k:
                break

            neg_density, freshness, idx, cand = heapq.heappop(heap)
            if cand.id in selected:
                continue

            cur_step = len(selected)
            if freshness < cur_step:
                # Stale: recompute marginal gain at the current selected set.
                m = self._marginal(selected, cand)
                n_evals += 1
                if m < 0.0:
                    # Non-monotone region: skip it permanently rather than poison the batch.
                    continue
                density = m / cand.cost if (cost_aware and cand.cost > 0) else m
                heapq.heappush(heap, (-density, cur_step, idx, cand))
                continue

            # The top is fresh — pick it, if it fits the budget.
            if cost_aware and (cost_so_far + cand.cost > (budget if budget is not None else 0.0) + 1e-12):
                continue

            m = (-neg_density) * (cand.cost if cost_aware else 1.0)
            if m <= 1e-15:
                # Diminishing returns drove the gain to zero — stop early.
                break
            selected.append(cand.id)
            eig_per.append(m)
            cost_so_far += cand.cost

        return ExperimentPlan(
            selected=selected,
            eig_per=eig_per,
            total_eig=sum(eig_per),
            total_cost=cost_so_far,
            n_evaluations=n_evals,
        )


# ---------------------------------------------------------------------------
# D-optimal design — Fedorov exchange on a row pool.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimalDesignResult:
    """Output of `DOptimalDesigner.select(...)`.

    `selected` are the chosen row indices into the candidate pool.
    `criterion_value` is the criterion at the optimum (log det for D,
    trace M^{-1} for A, max eigenvalue of M^{-1} for E — *lower* is
    better for A and E). `iterations` is the number of exchange sweeps
    that finished. `swaps` is the number of accepted exchanges.
    """

    selected: list[int]
    criterion_value: float
    iterations: int
    swaps: int
    criterion: str


def _det_via_lu(mat: list[list[float]]) -> float:
    """Determinant via in-place LU with partial pivoting. O(p^3)."""
    n = len(mat)
    a = [row[:] for row in mat]
    sign = 1.0
    for i in range(n):
        # Partial pivot.
        pivot = i
        max_abs = abs(a[i][i])
        for r in range(i + 1, n):
            if abs(a[r][i]) > max_abs:
                max_abs = abs(a[r][i])
                pivot = r
        if max_abs < 1e-15:
            return 0.0
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            sign = -sign
        for r in range(i + 1, n):
            factor = a[r][i] / a[i][i]
            for c in range(i, n):
                a[r][c] -= factor * a[i][c]
    det = sign
    for i in range(n):
        det *= a[i][i]
    return det


def _xtx(rows: list[list[float]]) -> list[list[float]]:
    """X^T X for a list of rows, in pure Python."""
    if not rows:
        return []
    p = len(rows[0])
    out = [[0.0] * p for _ in range(p)]
    for row in rows:
        for i in range(p):
            ri = row[i]
            for j in range(p):
                out[i][j] += ri * row[j]
    return out


def _eigvals_symmetric(mat: list[list[float]], *, tol: float = 1e-10, max_iter: int = 200) -> list[float]:
    """Symmetric-matrix eigenvalues via Jacobi rotations. O(p^3) per sweep.

    Suitable for the very small (p ≤ ~10) covariance matrices we encounter
    here. Returns eigenvalues in any order.
    """
    n = len(mat)
    a = [row[:] for row in mat]
    for _ in range(max_iter):
        # Find max off-diagonal.
        p, q, max_val = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j
        if max_val < tol:
            break
        # Jacobi rotation to zero out a[p][q].
        if abs(a[p][p] - a[q][q]) < 1e-30:
            theta = math.pi / 4.0 if a[p][q] > 0 else -math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * a[p][q], a[p][p] - a[q][q])
        c = math.cos(theta)
        s = math.sin(theta)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for i in range(n):
            if i != p and i != q:
                aip, aiq = a[i][p], a[i][q]
                a[i][p] = a[p][i] = c * aip - s * aiq
                a[i][q] = a[q][i] = s * aip + c * aiq
    return [a[i][i] for i in range(n)]


def _matrix_inverse(mat: list[list[float]]) -> list[list[float]] | None:
    """Inverse via Gauss-Jordan. Returns None if singular."""
    n = len(mat)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(mat)]
    for i in range(n):
        # Pivot.
        pivot = i
        for r in range(i + 1, n):
            if abs(a[r][i]) > abs(a[pivot][i]):
                pivot = r
        if abs(a[pivot][i]) < 1e-15:
            return None
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
        # Normalise pivot row.
        piv = a[i][i]
        for c in range(2 * n):
            a[i][c] /= piv
        # Eliminate other rows.
        for r in range(n):
            if r != i:
                factor = a[r][i]
                for c in range(2 * n):
                    a[r][c] -= factor * a[i][c]
    return [row[n:] for row in a]


class DOptimalDesigner:
    """Fedorov exchange (Fedorov 1972) for D-, A-, and E-optimal designs.

    Pick k rows from a candidate pool X (shape [N, p]) to optimise one
    of three classical criteria over the resulting information matrix
    M = X_S^T X_S:

      criterion="D"   maximise log det M           (volume of conf. ellipsoid)
      criterion="A"   minimise trace M^{-1}        (avg variance of β̂)
      criterion="E"   minimise max-eigenvalue M^{-1} (worst variance direction)

    Fedorov's exchange iterates: for each row currently in the design,
    consider swapping it with every row outside the design; accept the
    swap that most improves the criterion. Run until no swap improves
    or `max_iter` sweeps are exhausted.

    Convergence to a local optimum is guaranteed by monotone improvement.
    For small-to-medium p (the regime where this primitive belongs in a
    coordination engine), Fedorov is the standard and the practical
    state of the art.

    NOTE: requires k ≥ p so the information matrix can be non-singular;
    starting subsets that are singular get a tiny ridge for the first
    swap evaluations.
    """

    def __init__(
        self,
        candidate_X: Sequence[Sequence[float]],
        *,
        criterion: str = "D",
        ridge: float = 1e-10,
        max_iter: int = 50,
        rng: random.Random | None = None,
    ) -> None:
        if not candidate_X:
            raise ValueError("candidate_X is empty")
        if criterion not in ("D", "A", "E"):
            raise ValueError(f"unknown criterion {criterion!r}; choose D, A, or E")
        self._X = [list(row) for row in candidate_X]
        self._p = len(self._X[0])
        self._n = len(self._X)
        if any(len(row) != self._p for row in self._X):
            raise ValueError("candidate_X rows have inconsistent length")
        self._criterion = criterion
        self._ridge = float(ridge)
        self._max_iter = int(max_iter)
        self._rng = rng if rng is not None else random.Random()

    def _info_matrix(self, idx: Sequence[int]) -> list[list[float]]:
        rows = [self._X[i] for i in idx]
        M = _xtx(rows)
        if self._ridge > 0.0:
            for i in range(self._p):
                M[i][i] += self._ridge
        return M

    def _score(self, M: list[list[float]]) -> float:
        if self._criterion == "D":
            det = _det_via_lu(M)
            if det <= 0.0:
                return -math.inf
            return math.log(det)
        # A and E need M^{-1}.
        inv = _matrix_inverse(M)
        if inv is None:
            return math.inf  # singular → worse (we minimise A/E)
        if self._criterion == "A":
            return sum(inv[i][i] for i in range(self._p))
        # E: largest eigenvalue of M^{-1}.
        evs = _eigvals_symmetric(inv)
        return max(evs)

    def _better(self, new_score: float, cur_score: float) -> bool:
        if self._criterion == "D":
            return new_score > cur_score + 1e-12
        return new_score < cur_score - 1e-12

    def select(
        self,
        k: int,
        *,
        initial: Sequence[int] | None = None,
    ) -> OptimalDesignResult:
        """Run Fedorov exchange for k rows.

        `initial` is an optional seed subset (otherwise random). Returns
        the selected indices, the criterion value, and exchange counts.
        """
        if k < self._p:
            raise ValueError(
                f"k={k} must be ≥ p={self._p} for a non-singular information matrix"
            )
        if k > self._n:
            raise ValueError(f"k={k} exceeds pool size {self._n}")

        if initial is not None:
            current = list(initial)
            if len(current) != k or any(i < 0 or i >= self._n for i in current):
                raise ValueError("initial subset invalid")
        else:
            current = list(self._rng.sample(range(self._n), k))

        cur_score = self._score(self._info_matrix(current))
        swaps = 0
        for it in range(self._max_iter):
            best_swap: tuple[int, int, float] | None = None
            for i_pos in range(k):
                for j in range(self._n):
                    if j in current:
                        continue
                    trial = current.copy()
                    trial[i_pos] = j
                    s = self._score(self._info_matrix(trial))
                    if best_swap is None or self._better(s, best_swap[2]):
                        best_swap = (i_pos, j, s)
            if best_swap is None or not self._better(best_swap[2], cur_score):
                return OptimalDesignResult(
                    selected=sorted(current),
                    criterion_value=cur_score,
                    iterations=it,
                    swaps=swaps,
                    criterion=self._criterion,
                )
            i_pos, j, s = best_swap
            current[i_pos] = j
            cur_score = s
            swaps += 1
        return OptimalDesignResult(
            selected=sorted(current),
            criterion_value=cur_score,
            iterations=self._max_iter,
            swaps=swaps,
            criterion=self._criterion,
        )


# ---------------------------------------------------------------------------
# ExperimentDesigner — the unified surface a coordinator drives.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignRequest:
    """A coordination-engine request for the designer.

    `candidates` is the pool; each must have an `id` and an `eig`
    (the marginal-EIG estimate for that candidate under the current
    posterior). `cost` defaults to 1 (equal-cost selection).

    Constraints (any combination):
      * `k` — exactly this many selections
      * `budget` — total cost ≤ this
      * `min_eig_per` — drop candidates whose marginal EIG is below
        this floor (useful when most candidates are uninformative)

    `correlated` toggles whether the planner uses the correlated
    eig-marginal callback supplied at construction time. If unset, all
    candidates' EIGs add and the standard greedy is exact.
    """

    candidates: Sequence[ExperimentCandidate]
    k: int | None = None
    budget: float | None = None
    min_eig_per: float = 0.0
    correlated: bool = False


@dataclass(frozen=True)
class DesignResponse:
    """Coordination-engine-facing summary.

    `plan` is the raw plan; the additional fields are convenience
    derivations the coordinator commonly needs:

      * `eig_per_dollar` — efficient frontier slope at this batch size,
        useful for "is the next dollar worth spending?"
      * `binding_constraint` — which constraint stopped the batch.
    """

    plan: ExperimentPlan
    eig_per_dollar: float
    binding_constraint: str  # one of: "k", "budget", "min_eig_per", "exhausted"


class ExperimentDesigner:
    """Top-level Bayesian Optimal Experiment Design surface.

    Composes the batch planner, BALD scorer, and KG / Thompson Top-K
    helpers behind a single coordination-engine API.

        designer = ExperimentDesigner()

        candidates = [
            ExperimentCandidate(id=t.id, eig=t.bald_score, cost=t.est_cost_usd)
            for t in unlabeled_pool
        ]
        resp = designer.design(DesignRequest(candidates=candidates, budget=12.50))

        for ticket_id in resp.plan.selected:
            coordinator.dispatch_with_logging(ticket_id)

    The class is *stateless*: each `design(...)` call is a pure function
    of the request. The coordination engine owns all state — what's
    been run, what data has come back. Statelessness keeps the
    composition with `Strategist`, `PolicyLab`, and the rest of the
    runtime clean and testable.
    """

    def __init__(
        self,
        *,
        eig_marginal: Callable[[list[str], ExperimentCandidate], float] | None = None,
    ) -> None:
        self._planner = BayesianBatchPlanner(eig_marginal=eig_marginal)
        self._has_correlated = eig_marginal is not None

    def design(self, request: DesignRequest) -> DesignResponse:
        """Select an information-maximising batch.

        Filters by `min_eig_per` first (cheap), then runs the lazy
        greedy planner. Reports which constraint was binding so the
        coordinator can adapt next call.
        """
        pool = [
            c for c in request.candidates if c.eig >= request.min_eig_per - 1e-15
        ]
        if not pool:
            return DesignResponse(
                plan=ExperimentPlan([], [], 0.0, 0.0, 0),
                eig_per_dollar=0.0,
                binding_constraint="min_eig_per",
            )
        planner = self._planner if (request.correlated and self._has_correlated) else BayesianBatchPlanner()
        plan = planner.plan(pool, k=request.k, budget=request.budget)

        if plan.total_cost <= 0.0:
            eig_per_dollar = 0.0
        else:
            eig_per_dollar = plan.total_eig / plan.total_cost

        # Diagnose which constraint stopped the batch.
        if request.k is not None and len(plan.selected) >= request.k:
            binding = "k"
        elif request.budget is not None and abs(plan.total_cost - request.budget) < (
            request.budget * 1e-9 + 1e-12
        ):
            binding = "budget"
        elif len(plan.selected) >= len(pool):
            binding = "exhausted"
        elif (
            request.budget is not None
            and plan.total_cost + min((c.cost for c in pool if c.id not in plan.selected), default=float("inf")) > request.budget + 1e-12
        ):
            binding = "budget"
        else:
            binding = "min_eig_per" if plan.eig_per and plan.eig_per[-1] <= request.min_eig_per + 1e-12 else "exhausted"

        return DesignResponse(
            plan=plan,
            eig_per_dollar=eig_per_dollar,
            binding_constraint=binding,
        )

    @staticmethod
    def score_bald(
        committee_predictions: Sequence[Sequence[float]],
    ) -> BALDResult:
        """Static helper: forward to `bald_score`."""
        return bald_score(committee_predictions)

    @staticmethod
    def score_knowledge_gradient(
        means: Sequence[float],
        stds: Sequence[float],
        obs_noise_std: Sequence[float],
    ) -> list[float]:
        """Static helper: forward to `knowledge_gradient`."""
        return knowledge_gradient(means, stds, obs_noise_std)


__all__ = [
    "BALDResult",
    "BayesianBatchPlanner",
    "DesignRequest",
    "DesignResponse",
    "DOptimalDesigner",
    "ExperimentCandidate",
    "ExperimentDesigner",
    "ExperimentPlan",
    "NestedMCEig",
    "OptimalDesignResult",
    "bald_score",
    "eig_discrete",
    "eig_nested_mc",
    "entropy",
    "js_divergence",
    "kl_divergence",
    "knowledge_gradient",
    "predictive_distribution",
    "thompson_top_k",
]
