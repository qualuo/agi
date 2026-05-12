"""TicketOracle — counterfactual replay + auto-tuning policy advisor.

The driver's `AdmissionAdvisor` is parameterised by a handful of policy
knobs: `min_p_success`, `max_cost_per_turn_usd`, and (per-request)
`allow_downgrade`. Picking those numbers used to be guesswork. After a
few thousand tickets, the runtime has the data to answer the question
mathematically: *given the receipts we already have, which set of
knobs would have minimised spend while preserving (or improving) the
hit rate?*

`TicketOracle` is that answer. It is the line between *"we run AI"*
and *"we run AI and the policy improves itself from history with a
provable counterfactual receipt."*

What it does, concretely
------------------------

  oracle = driver.oracle
  baseline, alt = oracle.compare(
      driver.tickets(),
      alt=PolicyKnobs(min_p_success=0.65, max_cost_per_turn_usd=0.08),
  )
  print(alt.projected_cost_savings_usd, alt.verdict_changes)

  rec = oracle.recommend(window=200)          # grid search
  print(rec.summary)                           # human-readable

  oracle.auto_tune(driver, window=500)         # closes the loop

  what_if = oracle.what_if(
      driver.tickets(),
      cost_multiplier=1.20,                    # provider raises prices 20%
  )

Why a coordination engine cares
-------------------------------

A coordination engine — local or remote — drives the runtime by
submitting `TicketRequest`s. It also has its *own* policies (when to
batch, when to retry, when to bump priority). The oracle gives the
coordinator a way to **backtest** those policies on the real
production receipt log *before* applying them. The driver exposes
the oracle as `driver.oracle`, so a coordinator that wants to
question its own past decisions has a one-call surface:

    >>> rec = driver.oracle.recommend()
    >>> if rec.improvement.projected_cost_savings_usd > 10.0:
    >>>     driver.oracle.apply(rec.knobs)

What the oracle does **not** do
-------------------------------

It does not re-run the LLM. Counterfactuals use the durable
`estimated_*` fields recorded on each receipt + the live
`PreflightEstimator` for alternative models. That keeps replay
deterministic and free — investor demos run in milliseconds, not
dollars — at the cost of trusting the estimator's calibration for
alt-model branches. For ADMITted branches that the runtime actually
ran, replay uses the **actual** outcome; only DOWNGRADE / DEFER /
REJECT branches use forecasts.

The oracle also does not invent new admission verdicts. It moves
existing receipts between the four `{ADMIT, DEFER, DOWNGRADE,
REJECT}` buckets that the advisor itself produces.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from agi.driver import (
    CANCELLED,
    COMPLETED,
    D_ADMISSION,
    D_DOWNGRADE,
    D_ESTIMATE,
    DEFERRED,
    FAILED,
    REJECTED,
    Receipt,
    Ticket,
)
from agi.preflight import (
    ADMIT,
    DEFER,
    DOWNGRADE,
    REJECT,
    AdmissionAdvisor,
    PreflightEstimator,
)


# --- knobs ----------------------------------------------------------

@dataclass(frozen=True)
class PolicyKnobs:
    """The parameters the oracle searches over.

    These map 1:1 onto `AdmissionAdvisor` construction arguments.
    `allow_downgrade` corresponds to `TicketRequest.allow_downgrade`
    in production — the oracle treats it as a tenant-wide default.
    """
    min_p_success: float = 0.55
    max_cost_per_turn_usd: float | None = None
    allow_downgrade: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- per-ticket counterfactual --------------------------------------


@dataclass
class TicketReplay:
    """Counterfactual outcome for one historical ticket.

    `baseline_*` mirrors what actually happened; `alt_*` is what the
    new knobs would have produced. Both `verdict` fields use the same
    constants as `agi.preflight` (`ADMIT/DEFER/DOWNGRADE/REJECT`) so a
    coordination engine can pattern-match without an extra dependency.
    """
    ticket_id: str
    tenant_id: str | None
    baseline_verdict: str
    alt_verdict: str
    baseline_cost_usd: float
    alt_cost_usd: float
    baseline_success: bool
    alt_p_success: float
    baseline_model: str | None
    alt_model: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def cost_delta_usd(self) -> float:
        """Positive = the alt knobs *saved* money."""
        return self.baseline_cost_usd - self.alt_cost_usd

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_delta_usd"] = self.cost_delta_usd
        return d


# --- aggregate counterfactual ---------------------------------------


@dataclass
class CounterfactualReport:
    """Result of replaying a population of tickets under one knob set."""
    knobs: PolicyKnobs
    n_tickets: int
    n_replayable: int
    n_skipped: int
    baseline_cost_usd: float
    alt_cost_usd: float
    projected_cost_savings_usd: float
    baseline_success_rate: float
    alt_success_rate: float
    verdict_changes: dict[str, int]
    replays: list[TicketReplay] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["replays"] = [r.to_dict() for r in self.replays]
        return d


@dataclass
class Recommendation:
    """Output of `oracle.recommend(...)`.

    `improvement` is the counterfactual report for `knobs`; the
    `summary` string is one line a UI or chat-ops bot can post
    verbatim.
    """
    knobs: PolicyKnobs
    improvement: CounterfactualReport
    baseline_knobs: PolicyKnobs
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "knobs": self.knobs.to_dict(),
            "baseline_knobs": self.baseline_knobs.to_dict(),
            "summary": self.summary,
            "improvement": self.improvement.to_dict(),
        }


@dataclass
class WhatIfReport:
    """Output of `oracle.what_if(...)`. Captures the impact of a
    forward-looking cost or pricing shock on the same population."""
    n_tickets: int
    baseline_cost_usd: float
    shocked_cost_usd: float
    cost_multiplier: float
    p_success_floor_breaches: int
    notes: list[str] = field(default_factory=list)

    @property
    def projected_cost_delta_usd(self) -> float:
        return self.shocked_cost_usd - self.baseline_cost_usd

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["projected_cost_delta_usd"] = self.projected_cost_delta_usd
        return d


# --- oracle ---------------------------------------------------------


# Default search space for `recommend`. Coordinators can pass their
# own to widen / narrow the grid.
DEFAULT_P_SUCCESS_GRID: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_COST_CAP_GRID: tuple[float | None, ...] = (None, 0.20, 0.10, 0.05, 0.02)


class TicketOracle:
    """Replay receipts under alternative admission policies.

    Construction takes the runtime's own `PreflightEstimator` so
    alt-model forecasts come from the same distribution that
    produced the originals. The oracle is otherwise stateless — pass
    receipts or tickets in, get reports back.
    """

    def __init__(
        self,
        estimator: PreflightEstimator,
        *,
        baseline_knobs: PolicyKnobs | None = None,
        history_path: str | Path | None = None,
    ) -> None:
        self._estimator = estimator
        self._baseline = baseline_knobs or PolicyKnobs()
        self._history_path = Path(history_path) if history_path else None
        self._lock = threading.Lock()
        # Receipts that the oracle has accepted via `record` (used when
        # tickets are no longer in memory but a coordination engine
        # wants to keep training the oracle's window from disk).
        self._receipts: list[Receipt] = []

    # --- baseline ---------------------------------------------------

    @property
    def baseline(self) -> PolicyKnobs:
        return self._baseline

    def set_baseline(self, knobs: PolicyKnobs) -> None:
        self._baseline = knobs

    # --- ingest -----------------------------------------------------

    def record(self, receipt: Receipt) -> None:
        """Ingest one receipt for later replay.

        A `RuntimeDriver` can be configured to call this on every
        ticket completion. The oracle keeps a rolling window
        in-memory; persistence is the receipts file itself.
        """
        with self._lock:
            self._receipts.append(receipt)
            if self._history_path is not None:
                try:
                    self._history_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._history_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(receipt.to_dict(), default=str) + "\n")
                except OSError:
                    pass

    def receipts(self, window: int | None = None) -> list[Receipt]:
        with self._lock:
            data = list(self._receipts)
        if window is not None and window < len(data):
            return data[-window:]
        return data

    # --- replay -----------------------------------------------------

    def replay(
        self,
        source: Iterable[Receipt | Ticket],
        knobs: PolicyKnobs,
        *,
        cost_multiplier: float = 1.0,
    ) -> CounterfactualReport:
        """Replay every receipt in `source` under `knobs`.

        `cost_multiplier` lets a coordination engine layer a pricing
        shock on top of the policy replay (used by `what_if`).
        """
        receipts = [_coerce_receipt(x) for x in source]
        replays: list[TicketReplay] = []
        verdict_changes: dict[str, int] = {}
        skipped = 0
        baseline_cost = 0.0
        alt_cost = 0.0
        baseline_success_count = 0
        alt_success_score = 0.0
        replayable = 0

        for r in receipts:
            replay = self._replay_one(r, knobs, cost_multiplier=cost_multiplier)
            if replay is None:
                skipped += 1
                continue
            replayable += 1
            replays.append(replay)
            baseline_cost += replay.baseline_cost_usd
            alt_cost += replay.alt_cost_usd
            if replay.baseline_success:
                baseline_success_count += 1
            # Alt contributes to hit-rate only when the alt admits the
            # ticket. REJECT/DEFER mean we didn't serve it, so it
            # doesn't count as a success regardless of the forecast.
            if replay.alt_verdict in (ADMIT, DOWNGRADE):
                alt_success_score += float(replay.alt_p_success)
            key = f"{replay.baseline_verdict}->{replay.alt_verdict}"
            verdict_changes[key] = verdict_changes.get(key, 0) + 1

        n = len(receipts)
        baseline_rate = (
            baseline_success_count / replayable if replayable else 0.0
        )
        alt_rate = alt_success_score / replayable if replayable else 0.0
        return CounterfactualReport(
            knobs=knobs,
            n_tickets=n,
            n_replayable=replayable,
            n_skipped=skipped,
            baseline_cost_usd=round(baseline_cost, 6),
            alt_cost_usd=round(alt_cost, 6),
            projected_cost_savings_usd=round(baseline_cost - alt_cost, 6),
            baseline_success_rate=round(baseline_rate, 4),
            alt_success_rate=round(alt_rate, 4),
            verdict_changes=verdict_changes,
            replays=replays,
        )

    def _replay_one(
        self,
        receipt: Receipt,
        knobs: PolicyKnobs,
        *,
        cost_multiplier: float = 1.0,
    ) -> TicketReplay | None:
        est_payload = _decision_payload(receipt, D_ESTIMATE)
        adm_payload = _decision_payload(receipt, D_ADMISSION)
        if est_payload is None or adm_payload is None:
            return None

        baseline_verdict = str(adm_payload.get("verdict", ""))
        if baseline_verdict not in (ADMIT, DEFER, DOWNGRADE, REJECT):
            return None

        est_cost = float(est_payload.get("cost_usd", 0.0)) * cost_multiplier
        est_cost_p90 = float(est_payload.get("cost_p90_usd", est_cost)) * cost_multiplier
        est_p_success = float(est_payload.get("p_success", 0.0))
        est_confidence = str(est_payload.get("confidence", "low"))
        est_samples = int(est_payload.get("samples", 0))
        est_model = str(est_payload.get("model") or receipt.model or "")

        # Re-derive the verdict under the alt knobs from the same
        # estimate the baseline saw. We don't re-call the estimator
        # for the same model (cheaper, deterministic).
        alt_verdict, alt_model, alt_cost, alt_p_success, notes = self._alt_advise(
            receipt=receipt,
            knobs=knobs,
            est_cost=est_cost,
            est_cost_p90=est_cost_p90,
            est_p_success=est_p_success,
            est_confidence=est_confidence,
            est_samples=est_samples,
            est_model=est_model,
            baseline_verdict=baseline_verdict,
            cost_multiplier=cost_multiplier,
        )

        baseline_cost_usd = _baseline_cost(receipt, cost_multiplier)
        baseline_success = receipt.status == COMPLETED

        # When the alt verdict is ADMIT and the baseline *also* ran an
        # ADMIT path that we observed, prefer the observed outcome over
        # the forecast — we know the truth. Forecast is only used for
        # branches the runtime didn't actually take.
        if (
            alt_verdict == ADMIT
            and baseline_verdict == ADMIT
            and receipt.status in (COMPLETED, FAILED, CANCELLED)
        ):
            alt_p_success = 1.0 if baseline_success else 0.0

        return TicketReplay(
            ticket_id=receipt.ticket_id,
            tenant_id=receipt.tenant_id,
            baseline_verdict=baseline_verdict,
            alt_verdict=alt_verdict,
            baseline_cost_usd=round(baseline_cost_usd, 6),
            alt_cost_usd=round(alt_cost, 6),
            baseline_success=baseline_success,
            alt_p_success=round(alt_p_success, 4),
            baseline_model=receipt.model or est_model,
            alt_model=alt_model,
            notes=notes,
        )

    def _alt_advise(
        self,
        *,
        receipt: Receipt,
        knobs: PolicyKnobs,
        est_cost: float,
        est_cost_p90: float,
        est_p_success: float,
        est_confidence: str,
        est_samples: int,
        est_model: str,
        baseline_verdict: str,
        cost_multiplier: float,
    ) -> tuple[str, str | None, float, float, list[str]]:
        """Return `(alt_verdict, alt_model, alt_cost_usd, alt_p_success,
        notes)`. The alt verdict uses the same decision rules as
        `AdmissionAdvisor.advise` but with knob overrides."""
        notes: list[str] = []

        # 1. Per-turn cost cap with optional downgrade fallback.
        if knobs.max_cost_per_turn_usd is not None and est_cost_p90 > knobs.max_cost_per_turn_usd:
            if knobs.allow_downgrade:
                alt = self._suggest_downgrade(
                    receipt=receipt,
                    cur_model=est_model,
                    cur_cost=est_cost,
                    knobs=knobs,
                    cost_multiplier=cost_multiplier,
                )
                if alt is not None:
                    notes.append(f"downgrade_to:{alt['model']}")
                    return (
                        DOWNGRADE,
                        alt["model"],
                        float(alt["est_cost_usd"]),
                        float(alt["est_p_success"]),
                        notes,
                    )
            notes.append("no_viable_downgrade")
            return (REJECT, None, 0.0, est_p_success, notes)

        # 2. p_success floor (only enforced when we trust the estimate).
        if est_p_success < knobs.min_p_success and est_confidence != "low":
            notes.append(
                f"p_success_below_floor:{est_p_success:.2f}<{knobs.min_p_success:.2f}"
            )
            return (REJECT, None, 0.0, est_p_success, notes)

        # 3. ADMIT — use the baseline's actual cost as the alt cost when
        #    the runtime actually ran it. Otherwise fall back to the
        #    estimated cost so a "would have ADMITted what was DEFERred"
        #    branch still bills sensibly in the report.
        alt_model = receipt.model or est_model
        if baseline_verdict == ADMIT and receipt.status in (COMPLETED, FAILED, CANCELLED):
            alt_cost = receipt.actual_cost_usd * cost_multiplier
        else:
            alt_cost = est_cost
            notes.append("alt_admit_uses_forecast")
        return (ADMIT, alt_model, alt_cost, est_p_success, notes)

    def _suggest_downgrade(
        self,
        *,
        receipt: Receipt,
        cur_model: str,
        cur_cost: float,
        knobs: PolicyKnobs,
        cost_multiplier: float,
    ) -> dict[str, Any] | None:
        """Look for a cheaper model whose p90 cost fits the cap and
        whose forecasted p_success clears the floor. We try the
        receipt's intent against the live estimator so the alt model
        gets a calibrated forecast even if it wasn't the original."""
        from agi.preflight import _MODEL_TIERS  # local import: stable list

        try:
            cur_idx = _MODEL_TIERS.index(cur_model)
        except ValueError:
            return None
        cap = knobs.max_cost_per_turn_usd
        if cap is None:
            return None
        prompt = receipt.intent
        for cheaper in _MODEL_TIERS[:cur_idx]:
            alt_est = self._estimator.estimate(prompt, _AltConfig(cheaper))
            shocked_cost = alt_est.cost_usd * cost_multiplier
            shocked_cost_p90 = alt_est.cost_p90_usd * cost_multiplier
            if shocked_cost_p90 <= cap and alt_est.p_success >= knobs.min_p_success:
                return {
                    "model": cheaper,
                    "est_cost_usd": shocked_cost,
                    "est_cost_p90_usd": shocked_cost_p90,
                    "est_p_success": alt_est.p_success,
                }
        return None

    # --- compare / recommend ---------------------------------------

    def compare(
        self,
        source: Iterable[Receipt | Ticket],
        alt: PolicyKnobs,
        *,
        baseline: PolicyKnobs | None = None,
    ) -> tuple[CounterfactualReport, CounterfactualReport]:
        """Replay the same population under two knob sets. Returns
        `(baseline_report, alt_report)`."""
        receipts = list(source)
        base = baseline or self._baseline
        return self.replay(receipts, base), self.replay(receipts, alt)

    def recommend(
        self,
        source: Iterable[Receipt | Ticket] | None = None,
        *,
        window: int | None = None,
        p_success_grid: Sequence[float] = DEFAULT_P_SUCCESS_GRID,
        cost_cap_grid: Sequence[float | None] = DEFAULT_COST_CAP_GRID,
        require_no_worse_success: bool = True,
        min_population: int = 5,
    ) -> Recommendation | None:
        """Grid-search the knob space against the supplied receipts
        (or the oracle's own rolling buffer) and pick the knobs with
        the highest projected savings that don't *meaningfully* hurt
        the alt success rate.

        Returns None when there is too little history to draw a
        conclusion — investors and operators get an explicit "we
        don't know yet" rather than a noisy recommendation.
        """
        receipts = (
            list(source)
            if source is not None
            else self.receipts(window=window)
        )
        if window is not None and len(receipts) > window:
            receipts = receipts[-window:]
        if len(receipts) < min_population:
            return None

        baseline_report = self.replay(receipts, self._baseline)
        if baseline_report.n_replayable == 0:
            return None

        best: Recommendation | None = None
        # baseline rate from the live receipts, not a heuristic.
        baseline_rate = baseline_report.baseline_success_rate
        for p in p_success_grid:
            for cap in cost_cap_grid:
                knobs = PolicyKnobs(
                    min_p_success=float(p),
                    max_cost_per_turn_usd=cap,
                    allow_downgrade=self._baseline.allow_downgrade,
                )
                report = self.replay(receipts, knobs)
                if report.n_replayable == 0:
                    continue
                if (
                    require_no_worse_success
                    and report.alt_success_rate + 1e-6 < baseline_rate
                ):
                    continue
                savings = report.projected_cost_savings_usd
                # Prefer larger savings; tie-break on alt_success_rate.
                if best is None or (
                    savings > best.improvement.projected_cost_savings_usd + 1e-9
                ) or (
                    abs(savings - best.improvement.projected_cost_savings_usd) < 1e-9
                    and report.alt_success_rate > best.improvement.alt_success_rate
                ):
                    summary = _summarise(self._baseline, knobs, report)
                    best = Recommendation(
                        knobs=knobs,
                        improvement=report,
                        baseline_knobs=self._baseline,
                        summary=summary,
                    )
        return best

    # --- what-if shock -----------------------------------------------

    def what_if(
        self,
        source: Iterable[Receipt | Ticket] | None = None,
        *,
        window: int | None = None,
        cost_multiplier: float = 1.0,
        knobs: PolicyKnobs | None = None,
    ) -> WhatIfReport:
        """Project the impact of a pricing or cost shock on the
        recorded population, optionally swapping knobs at the same
        time. Defaults to the baseline knobs."""
        receipts = (
            list(source) if source is not None else self.receipts(window=window)
        )
        if window is not None and len(receipts) > window:
            receipts = receipts[-window:]
        base_report = self.replay(receipts, knobs or self._baseline)
        shocked = self.replay(
            receipts, knobs or self._baseline, cost_multiplier=cost_multiplier
        )
        floor = (knobs or self._baseline).min_p_success
        breaches = sum(
            1 for r in shocked.replays if r.alt_p_success < floor and r.alt_verdict == ADMIT
        )
        return WhatIfReport(
            n_tickets=base_report.n_tickets,
            baseline_cost_usd=base_report.alt_cost_usd,
            shocked_cost_usd=shocked.alt_cost_usd,
            cost_multiplier=cost_multiplier,
            p_success_floor_breaches=breaches,
            notes=[f"replayable={shocked.n_replayable}"],
        )

    # --- apply ------------------------------------------------------

    def apply(self, advisor: AdmissionAdvisor, knobs: PolicyKnobs) -> PolicyKnobs:
        """Apply knobs to a live `AdmissionAdvisor`. Returns the
        previous baseline (callers may stash it for rollback)."""
        previous = self._baseline
        advisor._min_p_success = knobs.min_p_success
        advisor._max_cost_per_turn_usd = knobs.max_cost_per_turn_usd
        self._baseline = knobs
        return previous

    def auto_tune(
        self,
        driver: Any,
        *,
        window: int = 200,
        min_savings_usd: float = 0.0,
        dry_run: bool = False,
        p_success_grid: Sequence[float] = DEFAULT_P_SUCCESS_GRID,
        cost_cap_grid: Sequence[float | None] = DEFAULT_COST_CAP_GRID,
        require_no_worse_success: bool = True,
        min_population: int = 5,
    ) -> Recommendation | None:
        """Recommend + (optionally) apply the best knobs to the
        driver's advisor. Returns the recommendation it acted on, or
        None when no recommendation cleared the bar.

        Closed-loop: a coordination engine wires this on a timer
        (or after every N tickets) and gets a self-tuning runtime.
        """
        # Pull receipts from the driver's in-memory tickets and merge
        # with our own ingest buffer. The driver may be configured to
        # call `oracle.record(receipt)` itself; we don't assume.
        receipts: list[Receipt] = []
        try:
            for t in driver.tickets():
                if t.done:
                    receipts.append(t.receipt)
        except Exception:
            pass
        receipts.extend(self.receipts())
        if window is not None and len(receipts) > window:
            receipts = receipts[-window:]
        rec = self.recommend(
            receipts,
            window=window,
            p_success_grid=p_success_grid,
            cost_cap_grid=cost_cap_grid,
            require_no_worse_success=require_no_worse_success,
            min_population=min_population,
        )
        if rec is None:
            return None
        if rec.improvement.projected_cost_savings_usd < min_savings_usd:
            return None
        if not dry_run:
            self.apply(driver.advisor, rec.knobs)
        return rec


# --- helpers --------------------------------------------------------


def _decision_payload(receipt: Receipt, kind: str) -> dict[str, Any] | None:
    for d in receipt.decisions:
        # Decisions may be Decision dataclasses or already-serialized dicts.
        d_kind = getattr(d, "kind", None) or (d.get("kind") if isinstance(d, dict) else None)
        if d_kind != kind:
            continue
        payload = getattr(d, "payload", None) or (
            d.get("payload") if isinstance(d, dict) else None
        )
        if isinstance(payload, dict):
            return payload
    return None


def _baseline_cost(receipt: Receipt, cost_multiplier: float = 1.0) -> float:
    """Treat REJECTED/DEFERRED tickets as zero-cost baseline."""
    if receipt.status in (REJECTED, DEFERRED, CANCELLED):
        return 0.0
    return receipt.actual_cost_usd * cost_multiplier


def _coerce_receipt(x: Receipt | Ticket) -> Receipt:
    """Accept either a Ticket or a Receipt. Tickets that haven't
    finished are skipped during replay (their decisions list may be
    mid-flight)."""
    if isinstance(x, Ticket):
        return x.receipt
    return x


def _summarise(
    base: PolicyKnobs, alt: PolicyKnobs, report: CounterfactualReport
) -> str:
    parts = []
    if alt.min_p_success != base.min_p_success:
        parts.append(
            f"min_p_success {base.min_p_success:.2f}→{alt.min_p_success:.2f}"
        )
    if alt.max_cost_per_turn_usd != base.max_cost_per_turn_usd:
        a = "off" if alt.max_cost_per_turn_usd is None else f"${alt.max_cost_per_turn_usd:.3f}"
        b = "off" if base.max_cost_per_turn_usd is None else f"${base.max_cost_per_turn_usd:.3f}"
        parts.append(f"max_cost_per_turn {b}→{a}")
    knob_str = ", ".join(parts) if parts else "no knob deltas"
    return (
        f"{knob_str}: projected savings ${report.projected_cost_savings_usd:.4f} "
        f"on {report.n_replayable}/{report.n_tickets} tickets "
        f"(alt success {report.alt_success_rate:.2%}, "
        f"baseline {report.baseline_success_rate:.2%})"
    )


class _AltConfig:
    """Minimal SessionConfig stand-in for the estimator. We only need
    the attributes `_tools_signature` and `_prior_cost` inspect."""

    __slots__ = (
        "model",
        "max_tokens",
        "use_skills",
        "system_prompt_extra",
        "enable_web_search",
        "enable_web_fetch",
        "enable_tool_synthesis",
        "enable_delegation",
        "enable_reflection",
    )

    def __init__(self, model: str) -> None:
        self.model = model
        self.max_tokens = 16000
        self.use_skills = True
        self.system_prompt_extra = None
        self.enable_web_search = False
        self.enable_web_fetch = False
        self.enable_tool_synthesis = False
        self.enable_delegation = False
        self.enable_reflection = False


__all__ = [
    "DEFAULT_P_SUCCESS_GRID",
    "DEFAULT_COST_CAP_GRID",
    "PolicyKnobs",
    "TicketReplay",
    "CounterfactualReport",
    "Recommendation",
    "WhatIfReport",
    "TicketOracle",
]
