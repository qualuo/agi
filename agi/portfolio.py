"""PortfolioOptimizer — allocate a fixed agent budget across many tickets.

A coordination engine that hands the runtime N tickets and a single
total budget B faces an allocation problem: which tickets get the
expensive (high-success) model, which get a cheap (lower-success)
model, and which are not worth running at all? Doing this by hand is
guesswork; doing it via per-ticket admission control is local — every
ticket sees only its own cost ceiling and not the portfolio.

`PortfolioOptimizer` solves it as a multiple-choice knapsack against
the live `PreflightEstimator` forecasts:

  - For each request × candidate model, look up an `Estimate`
    (cost_usd, p_success) from the estimator. Forecasts blend a prior
    with empirical history — so the optimizer naturally improves as
    the runtime accumulates experience.
  - Choose one candidate per request (or "skip") to maximize
    Σ value_weight_i · p_success_im  subject to  Σ cost_im ≤ B.
  - Solve exactly with DP for small portfolios (the common case);
    fall back to a value-per-dollar greedy for very large portfolios.

The plan is a pure recommendation: a coordination engine can dispatch
it via `RuntimeDriver.submit_portfolio`, alter it, or use it for a
quote-only "what would this cost?" answer.

Investor framing: the runtime acts like a portfolio manager for AI
compute. Given a fixed spend, it picks the model mix that maximizes
expected successful task completions — not just by per-ticket cost,
but by *marginal expected value per dollar*. The frontier API surfaces
the Pareto curve operators actually want: "what does another $1 buy
me?"

This module is pure (no I/O, no threading, no LLM calls). All
randomness comes from the supplied estimator. It composes cleanly
with everything else: `RuntimePool` for federation, `PolicyManager`
for tenant caps, `RuntimeDriver` for dispatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from agi.driver import TicketRequest
from agi.preflight import Estimate, PreflightEstimator
from agi.runtime import SessionConfig


# Default candidate set, cheap → expensive. Keep in sync with
# `agi.costs.PRICING` so unknown models don't surprise the estimator.
DEFAULT_CANDIDATE_MODELS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
)

# A synthetic "do not run" candidate. cost=0, p_success=0. The optimizer
# emits this when running a ticket has negative marginal value at the
# given budget, or when no model meets the value floor.
SKIP_MODEL = "__skip__"

# DP discretization: 1/10000 of a dollar (~ 0.01 cent). Plenty for
# budgets up to a few hundred dollars without blowing the state space.
_DP_UNIT_USD = 1e-4

# Switch greedy when the DP table would exceed this many cells. Tuned
# so a single call stays well under a second on a laptop.
_DP_CELL_BUDGET = 8_000_000


@dataclass
class PortfolioCandidate:
    """One (request, model) cell with its forecast.

    `score` is the value the optimizer assigns this candidate — by
    default `value_weight * p_success`. Exposed so callers can plot
    or audit the choice surface.
    """
    model: str
    estimated_cost_usd: float
    estimated_p_success: float
    estimated_duration_s: float
    estimate: Estimate | None = None
    score: float = 0.0
    is_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("estimate", None)
        return d


@dataclass
class PortfolioAllocation:
    """The optimizer's decision for one request."""
    request_index: int
    request: TicketRequest
    chosen: PortfolioCandidate
    candidates: list[PortfolioCandidate]
    value_weight: float = 1.0

    @property
    def skipped(self) -> bool:
        return self.chosen.is_skip

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_index": self.request_index,
            "intent": self.request.intent,
            "tenant_id": self.request.tenant_id,
            "value_weight": self.value_weight,
            "chosen": self.chosen.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "skipped": self.skipped,
        }


@dataclass
class PortfolioPlan:
    """Output of `PortfolioOptimizer.plan`.

    The plan is a *recommendation* — `expected_cost_usd` is the sum of
    forecasts, not a guarantee. The driver still enforces per-ticket
    `budget_usd` and the policy manager still enforces tenant caps.
    """
    total_budget_usd: float
    expected_cost_usd: float
    expected_value: float
    expected_p_success_sum: float
    allocations: list[PortfolioAllocation]
    method: str  # "dp" | "greedy"
    skipped_count: int
    candidate_models: tuple[str, ...]
    notes: list[str] = field(default_factory=list)

    @property
    def expected_p_success_mean(self) -> float:
        n = len(self.allocations)
        if n == 0:
            return 0.0
        return self.expected_p_success_sum / n

    @property
    def utilization(self) -> float:
        """Fraction of total budget the plan actually spends."""
        if self.total_budget_usd <= 0:
            return 0.0
        return min(1.0, self.expected_cost_usd / self.total_budget_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget_usd": self.total_budget_usd,
            "expected_cost_usd": self.expected_cost_usd,
            "expected_value": self.expected_value,
            "expected_p_success_sum": self.expected_p_success_sum,
            "expected_p_success_mean": self.expected_p_success_mean,
            "utilization": self.utilization,
            "method": self.method,
            "skipped_count": self.skipped_count,
            "candidate_models": list(self.candidate_models),
            "notes": list(self.notes),
            "allocations": [a.to_dict() for a in self.allocations],
        }


@dataclass
class FrontierPoint:
    """One point on the budget → expected-value Pareto curve."""
    budget_usd: float
    expected_cost_usd: float
    expected_value: float
    expected_p_success_sum: float
    dispatched_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioOptimizer:
    """Solve multiple-choice knapsack over (request × model) forecasts.

    Stateless w.r.t. requests; references an estimator. Safe to share
    across threads as long as the estimator is.
    """

    def __init__(
        self,
        estimator: PreflightEstimator,
        *,
        candidate_models: Sequence[str] | None = None,
        value_floor: float = 0.0,
    ) -> None:
        self.estimator = estimator
        self.candidate_models: tuple[str, ...] = (
            tuple(candidate_models) if candidate_models is not None
            else DEFAULT_CANDIDATE_MODELS
        )
        if not self.candidate_models:
            raise ValueError("candidate_models must be non-empty")
        # Per-request minimum p_success to consider a candidate eligible.
        # Defaults to 0 so the optimizer never silently drops a request
        # for "low confidence" reasons — the operator opts in.
        self.value_floor = value_floor

    # ---------- planning -------------------------------------------------

    def plan(
        self,
        requests: Sequence[TicketRequest],
        *,
        total_budget_usd: float,
        value_weights: Sequence[float] | None = None,
        candidate_models: Sequence[str] | None = None,
        allow_skip: bool = True,
        method: str = "auto",
    ) -> PortfolioPlan:
        """Optimize allocation. `value_weights[i]` weights request i's
        expected value contribution (default: 1.0 each — every
        successful task counts equally).

        method="auto" picks DP for small portfolios and greedy for
        large ones. The choice is recorded on the returned plan.
        """
        if total_budget_usd < 0:
            raise ValueError("total_budget_usd must be >= 0")
        if value_weights is not None and len(value_weights) != len(requests):
            raise ValueError("value_weights length must match requests")

        models = (
            tuple(candidate_models) if candidate_models is not None
            else self.candidate_models
        )
        weights = (
            tuple(value_weights) if value_weights is not None
            else tuple(1.0 for _ in requests)
        )

        # Forecast every (request, model) cell.
        per_request_candidates: list[list[PortfolioCandidate]] = []
        for i, req in enumerate(requests):
            cands = self._forecast_candidates(
                req, models, weight=weights[i], allow_skip=allow_skip,
            )
            per_request_candidates.append(cands)

        notes: list[str] = []
        chosen_method = self._select_method(method, requests, total_budget_usd, notes)

        if chosen_method == "dp":
            picks = _solve_mckp_dp(per_request_candidates, total_budget_usd)
        else:
            picks = _solve_greedy(per_request_candidates, total_budget_usd)

        allocations: list[PortfolioAllocation] = []
        total_cost = 0.0
        total_value = 0.0
        total_psuccess = 0.0
        skipped = 0
        for i, (req, cands, pick_idx) in enumerate(
            zip(requests, per_request_candidates, picks)
        ):
            chosen = cands[pick_idx]
            allocations.append(
                PortfolioAllocation(
                    request_index=i,
                    request=req,
                    chosen=chosen,
                    candidates=cands,
                    value_weight=weights[i],
                )
            )
            if chosen.is_skip:
                skipped += 1
            else:
                total_cost += chosen.estimated_cost_usd
                total_psuccess += chosen.estimated_p_success
                total_value += chosen.score

        return PortfolioPlan(
            total_budget_usd=total_budget_usd,
            expected_cost_usd=round(total_cost, 6),
            expected_value=round(total_value, 6),
            expected_p_success_sum=round(total_psuccess, 6),
            allocations=allocations,
            method=chosen_method,
            skipped_count=skipped,
            candidate_models=models,
            notes=notes,
        )

    def frontier(
        self,
        requests: Sequence[TicketRequest],
        *,
        budgets: Iterable[float],
        value_weights: Sequence[float] | None = None,
        candidate_models: Sequence[str] | None = None,
        method: str = "auto",
    ) -> list[FrontierPoint]:
        """Return one `FrontierPoint` per supplied budget.

        Operators use this to see "what does another $X buy me?" —
        the Pareto curve flattens once the cheapest viable model is
        already chosen for every request.
        """
        points: list[FrontierPoint] = []
        for b in budgets:
            plan = self.plan(
                requests,
                total_budget_usd=float(b),
                value_weights=value_weights,
                candidate_models=candidate_models,
                method=method,
            )
            dispatched = sum(1 for a in plan.allocations if not a.skipped)
            points.append(
                FrontierPoint(
                    budget_usd=float(b),
                    expected_cost_usd=plan.expected_cost_usd,
                    expected_value=plan.expected_value,
                    expected_p_success_sum=plan.expected_p_success_sum,
                    dispatched_count=dispatched,
                )
            )
        return points

    # ---------- internals ------------------------------------------------

    def _forecast_candidates(
        self,
        request: TicketRequest,
        models: Sequence[str],
        *,
        weight: float,
        allow_skip: bool,
    ) -> list[PortfolioCandidate]:
        base_cfg = request.config or SessionConfig()
        out: list[PortfolioCandidate] = []
        for model in models:
            cfg = _clone_session_config(base_cfg, model=model)
            est = self.estimator.estimate(request.intent, cfg)
            if est.p_success < self.value_floor:
                # Below operator floor — represent as a "skip-equivalent"
                # candidate by leaving p_success as forecast but score 0.
                score = 0.0
            else:
                score = weight * est.p_success
            out.append(
                PortfolioCandidate(
                    model=model,
                    estimated_cost_usd=float(est.cost_usd),
                    estimated_p_success=float(est.p_success),
                    estimated_duration_s=float(est.duration_s),
                    estimate=est,
                    score=score,
                )
            )
        if allow_skip:
            out.append(
                PortfolioCandidate(
                    model=SKIP_MODEL,
                    estimated_cost_usd=0.0,
                    estimated_p_success=0.0,
                    estimated_duration_s=0.0,
                    estimate=None,
                    score=0.0,
                    is_skip=True,
                )
            )
        return out

    def _select_method(
        self,
        method: str,
        requests: Sequence[TicketRequest],
        budget: float,
        notes: list[str],
    ) -> str:
        if method == "dp":
            return "dp"
        if method == "greedy":
            return "greedy"
        if method != "auto":
            raise ValueError(f"unknown method {method!r}")
        # Estimate DP cell count: N * (B+1) * K. K ≈ len(models)+skip.
        n = len(requests)
        b_cells = max(1, int(round(budget / _DP_UNIT_USD)) + 1)
        k = len(self.candidate_models) + 1
        cells = n * b_cells * k
        if cells <= _DP_CELL_BUDGET:
            return "dp"
        notes.append(
            f"falling back to greedy: DP would need ~{cells:,} cells "
            f"(> {_DP_CELL_BUDGET:,})"
        )
        return "greedy"


# ---------- solvers ----------------------------------------------------


def _solve_mckp_dp(
    groups: list[list[PortfolioCandidate]],
    budget_usd: float,
) -> list[int]:
    """Exact DP for multiple-choice knapsack.

    Returns a list parallel to `groups`: the chosen candidate index per
    group. Cost is discretized to `_DP_UNIT_USD`; rounding bias is
    biased *down* on cost (ceil), so the DP never under-budgets a
    candidate vs. the float forecast.
    """
    n = len(groups)
    if n == 0:
        return []
    cap = max(0, int(round(budget_usd / _DP_UNIT_USD)))

    # dp[b] = (best_value, parent_b, candidate_idx, group_idx) up to the
    # current group, using budget exactly <= b. We store choice traces
    # so we can reconstruct picks at the end.
    NEG_INF = float("-inf")
    # Two rolling rows keep memory small.
    prev = [NEG_INF] * (cap + 1)
    prev[0] = 0.0
    # `choice` records, for each (group, budget) cell, which candidate
    # the optimal solution picked. Reconstruction walks back through it.
    choice: list[list[int]] = []

    for g, cands in enumerate(groups):
        cur = [NEG_INF] * (cap + 1)
        choice_row = [-1] * (cap + 1)
        # Precompute candidate cost cells (ceil so a $0.0035 cost lands
        # in 35, not 34).
        cand_cells: list[tuple[int, float, int]] = []
        for ci, c in enumerate(cands):
            cell = (int(-(-c.estimated_cost_usd // _DP_UNIT_USD))
                    if c.estimated_cost_usd > 0 else 0)
            cand_cells.append((cell, c.score, ci))
        for b in range(cap + 1):
            best_val = NEG_INF
            best_ci = -1
            for cell, value, ci in cand_cells:
                if cell > b:
                    continue
                prev_val = prev[b - cell]
                if prev_val == NEG_INF:
                    continue
                v = prev_val + value
                if v > best_val:
                    best_val = v
                    best_ci = ci
            cur[b] = best_val
            choice_row[b] = best_ci
        choice.append(choice_row)
        prev = cur

    # Find best feasible final budget.
    best_b = 0
    best_v = prev[0]
    for b in range(cap + 1):
        if prev[b] > best_v:
            best_v = prev[b]
            best_b = b

    # Reconstruct.
    picks: list[int] = [0] * n
    b = best_b
    for g in range(n - 1, -1, -1):
        ci = choice[g][b]
        if ci < 0:
            # No feasible pick — fall back to first skip candidate, or 0.
            ci = _first_skip_or_zero(groups[g])
            cost_cell = 0
        else:
            cost_cell = (
                int(-(-groups[g][ci].estimated_cost_usd // _DP_UNIT_USD))
                if groups[g][ci].estimated_cost_usd > 0 else 0
            )
        picks[g] = ci
        b -= cost_cell
        if b < 0:
            b = 0
    return picks


def _solve_greedy(
    groups: list[list[PortfolioCandidate]],
    budget_usd: float,
) -> list[int]:
    """Greedy fallback for large portfolios.

    Initialise everything to skip; then sweep upgrades in order of
    marginal value per marginal dollar until the budget runs out.
    Optimal in the LP relaxation; off-by-one item from the true MCKP
    optimum in the worst case (standard greedy guarantee).
    """
    n = len(groups)
    picks: list[int] = [_first_skip_or_zero(g) for g in groups]
    # Remaining budget after the (free) skip baseline.
    remaining = budget_usd

    # Build candidate upgrades: (delta_value/delta_cost, group, ci).
    while True:
        best_ratio = 0.0
        best_g = -1
        best_ci = -1
        best_delta_cost = 0.0
        best_delta_value = 0.0
        for g, cands in enumerate(groups):
            cur = cands[picks[g]]
            for ci, c in enumerate(cands):
                if ci == picks[g] or c.is_skip:
                    continue
                d_cost = c.estimated_cost_usd - cur.estimated_cost_usd
                d_val = c.score - cur.score
                if d_cost <= 0:
                    # Dominated upgrade — same or cheaper AND better.
                    if d_val > 0:
                        ratio = float("inf")
                    else:
                        continue
                else:
                    if d_val <= 0:
                        continue
                    ratio = d_val / d_cost
                if d_cost > remaining + 1e-12 and ratio != float("inf"):
                    continue
                if d_cost > remaining + 1e-12 and ratio == float("inf"):
                    # Can't afford even a free-value upgrade.
                    continue
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_g = g
                    best_ci = ci
                    best_delta_cost = d_cost
                    best_delta_value = d_val
        if best_g < 0:
            break
        picks[best_g] = best_ci
        remaining -= max(0.0, best_delta_cost)
        _ = best_delta_value  # for clarity / future logging
    return picks


def _first_skip_or_zero(group: list[PortfolioCandidate]) -> int:
    for i, c in enumerate(group):
        if c.is_skip:
            return i
    return 0


# ---------- utilities --------------------------------------------------


def _clone_session_config(cfg: SessionConfig, **overrides: Any) -> SessionConfig:
    data = {**cfg.__dict__, **overrides}
    return type(cfg)(**data)
