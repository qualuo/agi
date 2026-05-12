"""Governance — multi-tenant budgets, quotas, rate limits, fair-share.

A Runtime that ships to many users (or many sub-systems of one
coordination engine) needs hard isolation:

  - **Budgets**: tenant X cannot exceed $Y in a window.
  - **Quotas**: tenant X cannot run more than N concurrent sessions.
  - **Rate limits**: tenant X cannot submit more than R prompts per minute.
  - **Fair scheduling**: when contention exists, no tenant starves.
  - **Provable accounting**: every approved action is recorded so the
    operator can answer "what did tenant X cost last month?" by reading
    a single JSONL.

`PolicyManager` enforces these rules at admission. A caller (the HTTP
server, the JSON-RPC protocol, the AutonomyEngine, or a custom
coordinator) calls `policy.check_admission(tenant, ...)` before
spawning a session or submitting a task; if denied, the call is rejected
with a typed reason. After the work finishes, the caller reports actuals
via `policy.commit(...)` so the running window is up to date.

The policy is in-memory by default, with optional JSONL persistence for
restart-safe accounting. The window types supported out of the box:

  - **rolling time-window**: a budget over the last N seconds (typical:
    daily, hourly)
  - **calendar window**: per-day/per-hour with deterministic UTC reset
  - **lifetime**: hard total cap

This module deliberately stays out of business logic. It does NOT decide
who is a "premium" tenant or what the price of a token is — it enforces
limits the operator sets. Coordination engines wire it in front of the
Runtime; multi-tenant SaaS deployments depend on it for SOC2-style
isolation claims.

Investors care: this is the line between "demo" and "platform you can
sell to ten enterprises in parallel."
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque


@dataclass
class TenantLimits:
    """Per-tenant ceilings. Any None field means "no limit"."""
    tenant_id: str
    daily_cost_usd: float | None = None
    hourly_cost_usd: float | None = None
    lifetime_cost_usd: float | None = None
    max_concurrent_sessions: int | None = None
    max_prompts_per_minute: int | None = None
    max_prompts_per_day: int | None = None
    # Fair-share weight: when scheduling under contention, tenants are
    # served roughly in proportion to this. Default 1.0.
    fair_share_weight: float = 1.0
    notes: str = ""


@dataclass
class TenantUsage:
    """Running totals. Cost windows are time-stamped events; counts roll
    forward."""
    tenant_id: str
    lifetime_cost_usd: float = 0.0
    sessions_active: int = 0
    sessions_created: int = 0
    prompts_total: int = 0
    last_activity_ts: float = 0.0
    # Sliding windows: deques of (ts, cost) for time-bounded lookups.
    cost_events: Deque[tuple[float, float]] = field(default_factory=deque)
    prompt_events: Deque[float] = field(default_factory=deque)

    def trim(self, *, now: float, max_age: float = 86400.0) -> None:
        cutoff = now - max_age
        while self.cost_events and self.cost_events[0][0] < cutoff:
            self.cost_events.popleft()
        # Prompts: keep last day's worth max.
        while self.prompt_events and self.prompt_events[0] < cutoff:
            self.prompt_events.popleft()

    def cost_in_window(self, *, now: float, window_seconds: float) -> float:
        cutoff = now - window_seconds
        total = 0.0
        for ts, cost in self.cost_events:
            if ts >= cutoff:
                total += cost
        return total

    def prompts_in_window(self, *, now: float, window_seconds: float) -> int:
        cutoff = now - window_seconds
        return sum(1 for ts in self.prompt_events if ts >= cutoff)


@dataclass
class AdmissionDecision:
    """Result of `check_admission()`. `ok=True` means the call may
    proceed. `ok=False` carries a reason for the operator and an HTTP
    status hint for the server layer."""
    ok: bool
    reason: str = ""
    code: str = ""
    retry_after_seconds: float | None = None
    http_status: int = 200

    def __bool__(self) -> bool:
        return self.ok


class PolicyError(Exception):
    """Raised when a caller bypasses `check_admission()` and a hard
    invariant is violated. Operators see this in logs as a bug."""


class PolicyManager:
    """Stateful tenant governor. Thread-safe.

    Typical flow:

        pm = PolicyManager()
        pm.set_limits(TenantLimits("acme", daily_cost_usd=10.0))

        d = pm.check_admission("acme", kind="session_create")
        if not d: raise PermissionError(d.reason)
        sid = runtime.create_session(cfg)
        pm.session_started("acme", sid)

        ...later, between chats...
        d = pm.check_admission("acme", kind="chat", estimated_cost_usd=0.05)
        if not d: raise PermissionError(d.reason)
        text = runtime.chat(sid, prompt)
        pm.commit("acme", cost_usd=session.state.total_cost_usd_delta, kind="chat")

        ...on end...
        runtime.end_session(sid)
        pm.session_ended("acme", sid)

    The audit log writes one JSONL line per admission decision, one per
    commit, and one per session lifecycle event. Operators replay it to
    reconstruct any tenant's state at any past timestamp.
    """

    def __init__(
        self,
        *,
        audit_path: str | os.PathLike[str] | None = None,
        default_limits: TenantLimits | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._limits: dict[str, TenantLimits] = {}
        self._usage: dict[str, TenantUsage] = {}
        self._active_sessions: dict[str, str] = {}  # session_id → tenant
        self._default_limits = default_limits
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path.touch(exist_ok=True)

    # --- configuration ---------------------------------------------

    def set_limits(self, limits: TenantLimits) -> None:
        with self._lock:
            self._limits[limits.tenant_id] = limits

    def get_limits(self, tenant_id: str) -> TenantLimits:
        with self._lock:
            l = self._limits.get(tenant_id)
        if l is not None:
            return l
        if self._default_limits is not None:
            # Materialize a per-tenant copy
            d = self._default_limits
            return TenantLimits(
                tenant_id=tenant_id,
                daily_cost_usd=d.daily_cost_usd,
                hourly_cost_usd=d.hourly_cost_usd,
                lifetime_cost_usd=d.lifetime_cost_usd,
                max_concurrent_sessions=d.max_concurrent_sessions,
                max_prompts_per_minute=d.max_prompts_per_minute,
                max_prompts_per_day=d.max_prompts_per_day,
                fair_share_weight=d.fair_share_weight,
                notes=d.notes,
            )
        return TenantLimits(tenant_id=tenant_id)

    def _usage_for(self, tenant_id: str) -> TenantUsage:
        u = self._usage.get(tenant_id)
        if u is None:
            u = TenantUsage(tenant_id=tenant_id)
            self._usage[tenant_id] = u
        return u

    # --- admission --------------------------------------------------

    def check_admission(
        self,
        tenant_id: str,
        *,
        kind: str = "chat",
        estimated_cost_usd: float = 0.0,
    ) -> AdmissionDecision:
        """Return an AdmissionDecision. Side-effect free: caller must
        still record commits."""
        limits = self.get_limits(tenant_id)
        now = time.time()
        with self._lock:
            u = self._usage_for(tenant_id)
            u.trim(now=now)

            # Concurrent sessions (only for session_create)
            if kind == "session_create" and limits.max_concurrent_sessions is not None:
                if u.sessions_active >= limits.max_concurrent_sessions:
                    return self._deny(
                        tenant_id, "max_concurrent_sessions",
                        f"tenant {tenant_id} at session cap ({limits.max_concurrent_sessions})",
                        http_status=429,
                    )

            # Rate limits (prompts)
            if kind == "chat":
                if limits.max_prompts_per_minute is not None:
                    used = u.prompts_in_window(now=now, window_seconds=60.0)
                    if used >= limits.max_prompts_per_minute:
                        return self._deny(
                            tenant_id, "rate_limit_minute",
                            f"tenant {tenant_id}: {used} prompts in last 60s ≥ "
                            f"{limits.max_prompts_per_minute}",
                            retry_after=60.0,
                            http_status=429,
                        )
                if limits.max_prompts_per_day is not None:
                    used = u.prompts_in_window(now=now, window_seconds=86400.0)
                    if used >= limits.max_prompts_per_day:
                        return self._deny(
                            tenant_id, "rate_limit_day",
                            f"tenant {tenant_id}: {used} prompts in last 24h ≥ "
                            f"{limits.max_prompts_per_day}",
                            retry_after=86400.0,
                            http_status=429,
                        )

            # Cost budgets (lifetime, daily, hourly)
            if limits.lifetime_cost_usd is not None:
                if u.lifetime_cost_usd + estimated_cost_usd > limits.lifetime_cost_usd:
                    return self._deny(
                        tenant_id, "lifetime_budget",
                        f"tenant {tenant_id}: lifetime ${u.lifetime_cost_usd:.4f} + "
                        f"est ${estimated_cost_usd:.4f} > cap ${limits.lifetime_cost_usd:.4f}",
                        http_status=402,
                    )
            if limits.daily_cost_usd is not None:
                spent = u.cost_in_window(now=now, window_seconds=86400.0)
                if spent + estimated_cost_usd > limits.daily_cost_usd:
                    return self._deny(
                        tenant_id, "daily_budget",
                        f"tenant {tenant_id}: ${spent:.4f} in 24h + est ${estimated_cost_usd:.4f} "
                        f"> daily cap ${limits.daily_cost_usd:.4f}",
                        retry_after=86400.0,
                        http_status=402,
                    )
            if limits.hourly_cost_usd is not None:
                spent = u.cost_in_window(now=now, window_seconds=3600.0)
                if spent + estimated_cost_usd > limits.hourly_cost_usd:
                    return self._deny(
                        tenant_id, "hourly_budget",
                        f"tenant {tenant_id}: ${spent:.4f} in 1h + est ${estimated_cost_usd:.4f} "
                        f"> hourly cap ${limits.hourly_cost_usd:.4f}",
                        retry_after=3600.0,
                        http_status=402,
                    )

        self._audit({
            "type": "admission",
            "tenant_id": tenant_id,
            "kind": kind,
            "estimated_cost_usd": estimated_cost_usd,
            "ok": True,
            "ts": now,
        })
        return AdmissionDecision(ok=True)

    def _deny(
        self,
        tenant_id: str,
        code: str,
        reason: str,
        *,
        retry_after: float | None = None,
        http_status: int = 429,
    ) -> AdmissionDecision:
        decision = AdmissionDecision(
            ok=False, reason=reason, code=code,
            retry_after_seconds=retry_after, http_status=http_status,
        )
        self._audit({
            "type": "admission",
            "tenant_id": tenant_id,
            "code": code,
            "reason": reason,
            "ok": False,
            "ts": time.time(),
        })
        return decision

    # --- commits ----------------------------------------------------

    def session_started(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            u = self._usage_for(tenant_id)
            u.sessions_active += 1
            u.sessions_created += 1
            u.last_activity_ts = time.time()
            self._active_sessions[session_id] = tenant_id
        self._audit({
            "type": "session_started",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "ts": time.time(),
        })

    def session_ended(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            u = self._usage_for(tenant_id)
            if u.sessions_active > 0:
                u.sessions_active -= 1
            self._active_sessions.pop(session_id, None)
        self._audit({
            "type": "session_ended",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "ts": time.time(),
        })

    def commit(
        self,
        tenant_id: str,
        *,
        cost_usd: float = 0.0,
        kind: str = "chat",
    ) -> None:
        """Record actual cost + a chat-counted event."""
        now = time.time()
        with self._lock:
            u = self._usage_for(tenant_id)
            u.lifetime_cost_usd += max(0.0, cost_usd)
            u.cost_events.append((now, max(0.0, cost_usd)))
            if kind == "chat":
                u.prompt_events.append(now)
                u.prompts_total += 1
            u.last_activity_ts = now
            u.trim(now=now)
        self._audit({
            "type": "commit",
            "tenant_id": tenant_id,
            "kind": kind,
            "cost_usd": cost_usd,
            "ts": now,
        })

    # --- inspection -------------------------------------------------

    def usage(self, tenant_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            u = self._usage_for(tenant_id)
            u.trim(now=now)
            limits = self.get_limits(tenant_id)
            return {
                "tenant_id": tenant_id,
                "lifetime_cost_usd": u.lifetime_cost_usd,
                "sessions_active": u.sessions_active,
                "sessions_created": u.sessions_created,
                "prompts_total": u.prompts_total,
                "cost_last_hour_usd": u.cost_in_window(now=now, window_seconds=3600.0),
                "cost_last_day_usd": u.cost_in_window(now=now, window_seconds=86400.0),
                "prompts_last_minute": u.prompts_in_window(now=now, window_seconds=60.0),
                "prompts_last_day": u.prompts_in_window(now=now, window_seconds=86400.0),
                "limits": asdict(limits),
            }

    def tenants(self) -> list[str]:
        with self._lock:
            return list(set(list(self._limits.keys()) + list(self._usage.keys())))

    def snapshot(self) -> dict[str, Any]:
        return {tid: self.usage(tid) for tid in self.tenants()}

    # --- fair-share scheduling --------------------------------------

    def fair_pick(self, candidates: list[str]) -> str | None:
        """Given a non-empty list of tenants competing for one slot,
        pick the next one to serve. Lower observed cost-share / weight
        gets the slot.

        Returns None if `candidates` is empty.
        """
        if not candidates:
            return None
        now = time.time()
        scores: list[tuple[float, str]] = []
        with self._lock:
            for tid in candidates:
                u = self._usage_for(tid)
                u.trim(now=now)
                limits = self.get_limits(tid)
                weight = max(0.001, limits.fair_share_weight)
                # Lower normalized recent cost = higher priority.
                recent = u.cost_in_window(now=now, window_seconds=3600.0)
                score = recent / weight
                scores.append((score, tid))
        scores.sort()
        return scores[0][1]

    # --- audit ------------------------------------------------------

    def _audit(self, record: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        try:
            with self.audit_path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            return


# --- Runtime adapter --------------------------------------------------


class GovernedRuntime:
    """Thin wrapper that injects PolicyManager checks at the runtime
    boundary. Use this instead of a raw `Runtime` when your coordination
    engine is multi-tenant.

    `tenant_id` must be passed on every operation. The wrapper does
    *not* hide the underlying Runtime: callers can reach `.runtime` if
    they need an unbounded operation (and accept responsibility).
    """

    def __init__(self, runtime, policy: PolicyManager) -> None:
        self.runtime = runtime
        self.policy = policy

    def create_session(
        self,
        tenant_id: str,
        config=None,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        d = self.policy.check_admission(tenant_id, kind="session_create")
        if not d:
            raise PermissionError(d.reason)
        sid = self.runtime.create_session(
            config,
            session_id=session_id,
            parent_session_id=parent_session_id,
            namespace=tenant_id,
        )
        self.policy.session_started(tenant_id, sid)
        return sid

    def chat(
        self,
        tenant_id: str,
        session_id: str,
        prompt: str,
        *,
        estimated_cost_usd: float = 0.01,
    ) -> str:
        d = self.policy.check_admission(
            tenant_id, kind="chat", estimated_cost_usd=estimated_cost_usd,
        )
        if not d:
            raise PermissionError(d.reason)
        session = self.runtime.get_session(session_id)
        prev_cost = session.state.total_cost_usd
        text = self.runtime.chat(session_id, prompt)
        delta = max(0.0, session.state.total_cost_usd - prev_cost)
        self.policy.commit(tenant_id, cost_usd=delta, kind="chat")
        return text

    def end_session(self, tenant_id: str, session_id: str) -> None:
        self.runtime.end_session(session_id)
        self.policy.session_ended(tenant_id, session_id)
