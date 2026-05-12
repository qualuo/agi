"""Preflight estimator — predict cost / duration / success before dispatch.

A coordination engine driving this runtime needs to make economic
decisions: which task to schedule now, which to defer, which to
downgrade to a cheaper model, which to reject as too uncertain. Those
decisions require *forecasts* of what a chat turn will actually cost
and how likely it is to succeed.

This module supplies that forecast. It is intentionally lightweight: a
heuristic prior (prompt length × model pricing × tool overhead) that
blends with empirical history once enough runs have landed. The
estimator is a passive observer of the runtime's event stream — wire
it once, and every chat that completes feeds back into future
estimates.

Why this is the missing piece:

  - `governance.PolicyManager.check_admission()` already accepts an
    `estimated_cost_usd` parameter, but nothing in the runtime
    produced it.
  - `policy.PolicyRouter` (Thompson-sampled bandit) chooses a
    *role*, not a *model tier* — model tier should be chosen by
    predicted cost/benefit.
  - A coordination engine running 100s of tasks across 10s of tenants
    cannot honestly say "this run will cost $X" without an estimator.

Surface:

  Estimate            — dataclass: cost/duration/p_success + quantiles + confidence
  AdmissionAdvice     — dataclass: ADMIT / DEFER / DOWNGRADE / REJECT + reason
  PreflightEstimator  — predict / record / persist / reset
  AdmissionAdvisor    — combines estimator + governance + capacity into an advice

Persistence is optional JSONL; the format is stable so a fleet of
runtimes can pool history into a shared file and benefit from each
other's calibration.

Honest about limits: this is a calibration tool, not a planner. It
predicts marginal cost-per-turn given observed history of similar
turns. It does NOT predict multi-turn task completion. Combine with
the `Coordinator` for that.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agi.costs import PRICING
from agi.events import CHAT_COMPLETED, CHAT_STARTED, ERROR, Event


# Tool overhead heuristic: each enabled high-level capability adds an
# expected fraction of additional output tokens (tool calls produce
# additional reasoning + tool result rounds). Numbers reflect typical
# observations on the eval suite, not provider claims.
_TOOL_OUTPUT_MULTIPLIER = {
    "web_search": 1.30,
    "web_fetch": 1.20,
    "tool_synthesis": 1.40,
    "delegation": 1.80,  # subagent spawns can substantially inflate
    "reflection": 1.10,
}

# How much input the system prompt + skill block adds, in tokens.
# Conservative: real runtimes with large skill libraries are higher.
_BASE_SYSTEM_TOKENS = 1500
_SKILL_TOKENS_PER_RETRIEVED = 400  # avg per skill loaded

# Per-turn duration prior (seconds), keyed by model.
_DURATION_PRIOR_S = {
    "claude-opus-4-7": 14.0,
    "claude-opus-4-6": 12.0,
    "claude-sonnet-4-6": 7.0,
    "claude-haiku-4-5": 3.0,
}

# Per-turn success-rate prior (purely heuristic; learnable).
_SUCCESS_PRIOR = {
    "claude-opus-4-7": 0.92,
    "claude-opus-4-6": 0.90,
    "claude-sonnet-4-6": 0.85,
    "claude-haiku-4-5": 0.78,
}

# Bayesian-blend weight: how many real observations a bin needs before
# we trust empirical over the prior. Standard "n / (n + k)" blend.
_BLEND_K = 8.0


def _prompt_length_bucket(text: str) -> str:
    """Coarse bucketing so we don't fragment history per unique prompt."""
    n = len(text)
    if n < 200:
        return "xs"
    if n < 1000:
        return "s"
    if n < 4000:
        return "m"
    if n < 16000:
        return "l"
    return "xl"


def _tools_signature(config: Any) -> str:
    """Stable string key for the set of enabled tools."""
    parts = []
    for attr, label in (
        ("enable_web_search", "ws"),
        ("enable_web_fetch", "wf"),
        ("enable_tool_synthesis", "ts"),
        ("enable_delegation", "dg"),
        ("enable_reflection", "rf"),
        ("use_skills", "sk"),
    ):
        if bool(getattr(config, attr, False)):
            parts.append(label)
    return "+".join(parts) if parts else "none"


def _bin_key(prompt: str, model: str, tools: str) -> str:
    return f"{model}|{_prompt_length_bucket(prompt)}|{tools}"


# Approximate token count without a tokenizer. Treat ~4 chars/token as
# the steady-state ratio for English code+prose. Coordinators that
# need exactness can plug in a real tokenizer via the prompt_tokenizer
# kwarg on PreflightEstimator.
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class CostBreakdown:
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float

    def total(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


@dataclass
class Estimate:
    """Forecast for a single chat turn.

    Quantile fields (p10/p90) capture uncertainty. The confidence flag
    tells callers how much weight to put on this estimate vs. their
    own fallbacks: `low` means the prior dominates (insufficient
    history); `high` means a dense bin of similar runs.
    """
    cost_usd: float
    cost_p10_usd: float
    cost_p90_usd: float
    duration_s: float
    duration_p10_s: float
    duration_p90_s: float
    p_success: float
    confidence: str  # "low" | "medium" | "high"
    samples: int
    breakdown: CostBreakdown
    model: str
    bin_key: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["breakdown"] = asdict(self.breakdown)
        return d


class _Bin:
    """Rolling stats for a (model, prompt-size, tools) bin."""

    __slots__ = ("costs", "durations", "successes", "max_history")

    def __init__(self, max_history: int = 200) -> None:
        self.costs: list[float] = []
        self.durations: list[float] = []
        self.successes: list[bool] = []
        self.max_history = max_history

    def add(self, cost: float, duration: float, success: bool) -> None:
        self.costs.append(cost)
        self.durations.append(duration)
        self.successes.append(success)
        if len(self.costs) > self.max_history:
            self.costs = self.costs[-self.max_history :]
            self.durations = self.durations[-self.max_history :]
            self.successes = self.successes[-self.max_history :]

    @property
    def n(self) -> int:
        return len(self.costs)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    # Linear interpolation between order statistics.
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


class PreflightEstimator:
    """Per-turn cost / duration / success forecaster.

    Thread-safe. Persistence is optional and append-only JSONL.
    """

    def __init__(
        self,
        *,
        history_path: str | Path | None = None,
        prompt_tokenizer: Callable[[str], int] | None = None,
        max_bin_history: int = 200,
    ) -> None:
        self._lock = threading.Lock()
        self._bins: dict[str, _Bin] = {}
        self._history_path = Path(history_path) if history_path else None
        self._tokenize = prompt_tokenizer or _approx_tokens
        self._max_bin_history = max_bin_history
        if self._history_path is not None and self._history_path.exists():
            self._load()

    # --- public surface ---------------------------------------------

    def estimate(self, prompt: str, config: Any | None = None) -> Estimate:
        """Forecast a chat turn. `config` may be a SessionConfig or any
        object exposing the same fields; None implies opus + skills only."""
        cfg = config if config is not None else _DefaultCfg()
        model = getattr(cfg, "model", "claude-opus-4-7")
        tools_key = _tools_signature(cfg)
        key = _bin_key(prompt, model, tools_key)

        # Heuristic prior.
        prior_cost, prior_break = self._prior_cost(prompt, cfg, model)
        prior_dur = _DURATION_PRIOR_S.get(model, 10.0)
        prior_succ = _SUCCESS_PRIOR.get(model, 0.85)

        with self._lock:
            b = self._bins.get(key)
            n = b.n if b else 0

            if n == 0:
                cost = prior_cost
                cost_lo = prior_cost * 0.55
                cost_hi = prior_cost * 1.80
                dur = prior_dur
                dur_lo = prior_dur * 0.5
                dur_hi = prior_dur * 1.8
                p_success = prior_succ
                confidence = "low"
            else:
                # Bayesian-ish blend on the mean; quantiles purely empirical
                # once we have enough samples.
                w = n / (n + _BLEND_K)
                emp_cost = statistics.mean(b.costs)
                cost = w * emp_cost + (1 - w) * prior_cost
                cost_lo = _quantile(b.costs, 0.10) if n >= 4 else prior_cost * 0.55
                cost_hi = _quantile(b.costs, 0.90) if n >= 4 else prior_cost * 1.80

                emp_dur = statistics.mean(b.durations)
                dur = w * emp_dur + (1 - w) * prior_dur
                dur_lo = _quantile(b.durations, 0.10) if n >= 4 else prior_dur * 0.5
                dur_hi = _quantile(b.durations, 0.90) if n >= 4 else prior_dur * 1.8

                succ_rate = sum(b.successes) / n
                p_success = w * succ_rate + (1 - w) * prior_succ

                confidence = "high" if n >= 20 else ("medium" if n >= 5 else "low")

        notes: list[str] = []
        if model not in PRICING:
            notes.append(f"unknown model {model!r}; pricing prior is $0")

        return Estimate(
            cost_usd=round(cost, 6),
            cost_p10_usd=round(cost_lo, 6),
            cost_p90_usd=round(cost_hi, 6),
            duration_s=round(dur, 3),
            duration_p10_s=round(dur_lo, 3),
            duration_p90_s=round(dur_hi, 3),
            p_success=round(p_success, 4),
            confidence=confidence,
            samples=n,
            breakdown=prior_break,
            model=model,
            bin_key=key,
            notes=notes,
        )

    def record(
        self,
        *,
        prompt: str,
        config: Any | None,
        cost_usd: float,
        duration_s: float,
        success: bool,
    ) -> None:
        cfg = config if config is not None else _DefaultCfg()
        model = getattr(cfg, "model", "claude-opus-4-7")
        tools_key = _tools_signature(cfg)
        key = _bin_key(prompt, model, tools_key)
        with self._lock:
            b = self._bins.setdefault(key, _Bin(max_history=self._max_bin_history))
            b.add(cost_usd, duration_s, success)
        if self._history_path is not None:
            self._append({
                "ts": time.time(),
                "key": key,
                "cost_usd": cost_usd,
                "duration_s": duration_s,
                "success": success,
            })

    def stats(self) -> dict[str, Any]:
        """Snapshot of all bins. Useful for an investor-facing
        dashboard: "we have N data points across K configurations
        and predict within ±X% on Y% of dispatches."""
        with self._lock:
            out: dict[str, Any] = {}
            for key, b in self._bins.items():
                if b.n == 0:
                    continue
                out[key] = {
                    "samples": b.n,
                    "mean_cost_usd": round(statistics.mean(b.costs), 6),
                    "mean_duration_s": round(statistics.mean(b.durations), 3),
                    "success_rate": round(sum(b.successes) / b.n, 4),
                }
            return {
                "bins": out,
                "bin_count": len(out),
                "total_samples": sum(v["samples"] for v in out.values()),
            }

    def reset(self) -> None:
        with self._lock:
            self._bins.clear()

    # --- attach to runtime ------------------------------------------

    def attach(self, runtime: Any) -> int:
        """Subscribe to the runtime's event bus so every completed
        chat updates the estimator. Returns the subscription id —
        keep it if you intend to `unsubscribe` later.

        We track CHAT_STARTED to capture the prompt + start time,
        then resolve on CHAT_COMPLETED with the session's cost delta.
        """
        pending: dict[str, dict[str, Any]] = {}
        pending_lock = threading.Lock()

        def on_event(ev: Event) -> None:
            sid = ev.session_id
            if sid is None:
                return
            if ev.kind == CHAT_STARTED:
                with pending_lock:
                    pending[sid] = {
                        "prompt": ev.data.get("user_input", ""),
                        "started_ts": ev.ts,
                        "cost_at_start": _session_cost(runtime, sid),
                    }
            elif ev.kind == CHAT_COMPLETED:
                with pending_lock:
                    rec = pending.pop(sid, None)
                if rec is None:
                    return
                cfg = _session_config(runtime, sid)
                cost_now = _session_cost(runtime, sid)
                cost_delta = max(0.0, cost_now - rec["cost_at_start"])
                duration = max(0.0, ev.ts - rec["started_ts"])
                critic = ev.data.get("critic_score")
                # Success: critic above 0.5 if scored; otherwise no-error
                # by virtue of having reached CHAT_COMPLETED.
                success = True if critic is None else float(critic) >= 0.5
                try:
                    self.record(
                        prompt=rec["prompt"],
                        config=cfg,
                        cost_usd=cost_delta,
                        duration_s=duration,
                        success=success,
                    )
                except Exception:
                    pass
            elif ev.kind == ERROR and ev.data.get("phase") == "chat":
                with pending_lock:
                    rec = pending.pop(sid, None)
                if rec is None:
                    return
                cfg = _session_config(runtime, sid)
                try:
                    self.record(
                        prompt=rec["prompt"],
                        config=cfg,
                        cost_usd=0.0,
                        duration_s=max(0.0, ev.ts - rec["started_ts"]),
                        success=False,
                    )
                except Exception:
                    pass

        return runtime.bus.subscribe(on_event)

    # --- internals --------------------------------------------------

    def _prior_cost(self, prompt: str, cfg: Any, model: str) -> tuple[float, CostBreakdown]:
        in_rate, out_rate = PRICING.get(model, (0.0, 0.0))

        prompt_tokens = self._tokenize(prompt)
        skill_tokens = _SKILL_TOKENS_PER_RETRIEVED * 3 if getattr(cfg, "use_skills", True) else 0
        system_tokens = _BASE_SYSTEM_TOKENS
        if getattr(cfg, "system_prompt_extra", None):
            system_tokens += self._tokenize(cfg.system_prompt_extra)

        # Default expected output: ~max(256, 1/4 of max_tokens).
        max_tok = int(getattr(cfg, "max_tokens", 16000) or 16000)
        base_out = max(256, max_tok // 4)
        out_mult = 1.0
        for attr, label in (
            ("enable_web_search", "web_search"),
            ("enable_web_fetch", "web_fetch"),
            ("enable_tool_synthesis", "tool_synthesis"),
            ("enable_delegation", "delegation"),
            ("enable_reflection", "reflection"),
        ):
            if getattr(cfg, attr, False):
                out_mult *= _TOOL_OUTPUT_MULTIPLIER.get(label, 1.0)

        expected_in = prompt_tokens + skill_tokens + system_tokens
        expected_out = int(base_out * out_mult)

        cost_in = expected_in * in_rate / 1_000_000
        cost_out = expected_out * out_rate / 1_000_000
        return cost_in + cost_out, CostBreakdown(
            input_tokens=expected_in,
            output_tokens=expected_out,
            input_cost_usd=round(cost_in, 6),
            output_cost_usd=round(cost_out, 6),
        )

    def _append(self, record: dict[str, Any]) -> None:
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _load(self) -> None:
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    b = self._bins.setdefault(r["key"], _Bin(max_history=self._max_bin_history))
                    b.add(float(r["cost_usd"]), float(r["duration_s"]), bool(r["success"]))
        except OSError:
            pass


# --- admission advice ------------------------------------------------

ADMIT = "ADMIT"
DEFER = "DEFER"
DOWNGRADE = "DOWNGRADE"
REJECT = "REJECT"


@dataclass
class AdmissionAdvice:
    """Recommendation for a coordination engine.

    `verdict` is one of ADMIT/DEFER/DOWNGRADE/REJECT. Coordinators may
    choose to ignore the advice (e.g. a high-priority job overrides
    DEFER), but at minimum they have a sound default. `alternative`
    carries any concrete suggestion (e.g. switch to a cheaper model)
    and is None when no alternative applies.
    """
    verdict: str
    reason: str
    estimate: Estimate
    alternative: dict[str, Any] | None = None
    retry_after_s: float | None = None
    governance_code: str | None = None

    def admit(self) -> bool:
        return self.verdict == ADMIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "estimate": self.estimate.to_dict(),
            "alternative": self.alternative,
            "retry_after_s": self.retry_after_s,
            "governance_code": self.governance_code,
        }


# Ordered cheap-to-expensive list for downgrade suggestions.
_MODEL_TIERS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
]


class AdmissionAdvisor:
    """Combines an estimator + (optional) governance policy + capacity
    signal into a single recommendation per request.

    The advisor is read-only with respect to governance — it never
    commits charges. A coordination engine should call `advise(...)`,
    act on the verdict, and (if it dispatched) record the actual
    outcome via the runtime event stream — which the estimator
    consumes automatically when attached.
    """

    def __init__(
        self,
        estimator: PreflightEstimator,
        *,
        policy: Any | None = None,
        runtime: Any | None = None,
        min_p_success: float = 0.55,
        max_cost_per_turn_usd: float | None = None,
    ) -> None:
        self._estimator = estimator
        self._policy = policy
        self._runtime = runtime
        self._min_p_success = min_p_success
        self._max_cost_per_turn_usd = max_cost_per_turn_usd

    def advise(
        self,
        *,
        prompt: str,
        config: Any | None = None,
        tenant_id: str | None = None,
    ) -> AdmissionAdvice:
        est = self._estimator.estimate(prompt, config)

        # 1. Hard per-turn cost ceiling (operator policy, not tenant policy).
        if (
            self._max_cost_per_turn_usd is not None
            and est.cost_p90_usd > self._max_cost_per_turn_usd
        ):
            alt = self._suggest_downgrade(prompt, config, est)
            if alt is not None:
                return AdmissionAdvice(
                    DOWNGRADE,
                    f"p90 cost ${est.cost_p90_usd:.4f} > per-turn cap "
                    f"${self._max_cost_per_turn_usd:.4f}; downgrade to {alt['model']} "
                    f"projected ${alt['est_cost_usd']:.4f}",
                    est,
                    alternative=alt,
                )
            return AdmissionAdvice(
                REJECT,
                f"p90 cost ${est.cost_p90_usd:.4f} > per-turn cap "
                f"${self._max_cost_per_turn_usd:.4f} and no cheaper viable model",
                est,
            )

        # 2. Quality floor.
        if est.p_success < self._min_p_success and est.confidence != "low":
            return AdmissionAdvice(
                REJECT,
                f"predicted p_success {est.p_success:.2f} below floor "
                f"{self._min_p_success:.2f} (confidence={est.confidence}, n={est.samples})",
                est,
            )

        # 3. Governance — uses the *p90* cost so we don't admit a job that
        #    has 10% chance of busting a tenant's daily cap.
        if self._policy is not None and tenant_id is not None:
            decision = self._policy.check_admission(
                tenant_id,
                kind="chat",
                estimated_cost_usd=est.cost_p90_usd,
            )
            if not decision:
                # Distinguish budget-style denials (defer-eligible) from
                # rate-limit denials.
                code = getattr(decision, "code", "denied")
                retry = getattr(decision, "retry_after_seconds", None)
                if code in {"daily_budget", "hourly_budget", "rate_limit_minute", "rate_limit_day"}:
                    verdict = DEFER
                else:
                    verdict = REJECT
                return AdmissionAdvice(
                    verdict,
                    f"governance: {decision.reason}",
                    est,
                    retry_after_s=retry,
                    governance_code=code,
                )

        # 4. Capacity — if the runtime is at its session cap, we suggest
        #    defer rather than admit-and-fail.
        if self._runtime is not None:
            cap = getattr(self._runtime, "max_concurrent_sessions", None)
            if cap is not None:
                active = sum(
                    1 for s in getattr(self._runtime, "_sessions", {}).values()
                    if not s.state.ended
                )
                if active >= cap:
                    return AdmissionAdvice(
                        DEFER,
                        f"runtime at session cap ({active}/{cap})",
                        est,
                        retry_after_s=5.0,
                    )

        return AdmissionAdvice(ADMIT, "ok", est)

    def _suggest_downgrade(
        self, prompt: str, config: Any, current: Estimate
    ) -> dict[str, Any] | None:
        """Find the cheapest tier whose p90 cost fits the cap and whose
        predicted p_success doesn't drop below the floor."""
        if self._max_cost_per_turn_usd is None:
            return None
        cur_model = current.model
        try:
            cur_idx = _MODEL_TIERS.index(cur_model)
        except ValueError:
            return None

        for cheaper in _MODEL_TIERS[:cur_idx]:
            alt_cfg = _shallow_clone(config, model=cheaper) if config else _DefaultCfg(model=cheaper)
            alt_est = self._estimator.estimate(prompt, alt_cfg)
            if (
                alt_est.cost_p90_usd <= self._max_cost_per_turn_usd
                and alt_est.p_success >= self._min_p_success
            ):
                return {
                    "model": cheaper,
                    "est_cost_usd": alt_est.cost_usd,
                    "est_cost_p90_usd": alt_est.cost_p90_usd,
                    "est_p_success": alt_est.p_success,
                    "expected_savings_usd": max(0.0, current.cost_usd - alt_est.cost_usd),
                }
        return None


# --- helpers ---------------------------------------------------------


def _session_cost(runtime: Any, sid: str) -> float:
    s = getattr(runtime, "_sessions", {}).get(sid)
    if s is None:
        return 0.0
    return float(getattr(s.state, "total_cost_usd", 0.0))


def _session_config(runtime: Any, sid: str) -> Any | None:
    s = getattr(runtime, "_sessions", {}).get(sid)
    if s is None:
        return None
    return s.state.config


def _shallow_clone(cfg: Any, **overrides: Any) -> Any:
    """Best-effort clone of a SessionConfig-like object with overrides.
    Used for what-if downgrade suggestions."""
    if cfg is None:
        return _DefaultCfg(**overrides)
    # If it's a dataclass, dataclasses.replace handles it cleanly.
    try:
        from dataclasses import is_dataclass, replace
        if is_dataclass(cfg):
            return replace(cfg, **overrides)
    except Exception:
        pass
    # Fallback: copy attributes onto a generic holder.
    holder = _DefaultCfg()
    for k, v in getattr(cfg, "__dict__", {}).items():
        setattr(holder, k, v)
    for k, v in overrides.items():
        setattr(holder, k, v)
    return holder


@dataclass
class _DefaultCfg:
    """Stand-in config for callers that pass nothing."""
    model: str = "claude-opus-4-7"
    max_tokens: int = 16000
    enable_web_search: bool = True
    enable_web_fetch: bool = True
    enable_tool_synthesis: bool = False
    enable_delegation: bool = False
    enable_reflection: bool = False
    use_skills: bool = True
    system_prompt_extra: str | None = None
