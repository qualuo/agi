"""Multi-tenant coordination engine demo over `RuntimeDriver`.

The pitch this demo makes concrete:

    A coordination engine (orchestrator, web app, autonomous planner)
    drives this runtime through ONE entry point — `RuntimeDriver`. The
    driver gives the coordination engine, on every request, for free:

      - cost forecast (preflight estimate)
      - admission control (per-tenant budgets + quotas)
      - automatic model downgrade when cheaper tiers fit
      - live event stream
      - hard per-ticket budget ceiling
      - billing-grade receipt (JSON-serializable, includes the causal trace)

    Operators see receipts.jsonl; coordinators see decisions[] inline.
    Two tenants run concurrently against the same runtime under hard
    isolation; the system reports total fleet stats at the end.

Run:

    python examples/driver_multi_tenant_demo.py

Uses FakeAgent so no API key is required. Pass `--live` to swap in real
Opus (will incur cost; not enabled by default).
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.driver import RuntimeDriver, TicketRequest
from agi.governance import PolicyManager, TenantLimits
from agi.memory import Memory
from agi.preflight import AdmissionAdvisor
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# A coordination engine, in production, would either authenticate against
# the real Anthropic API or hold a stable of local model adapters. For
# this demo we use the same FakeAgent the test suite uses.
class _FakeUsage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        # Pretend opus 4.7 is expensive, haiku 4.5 is cheap, sonnet middle.
        if "haiku" in model:
            return self.input_tokens * 1e-6 + self.output_tokens * 5e-6
        if "sonnet" in model:
            return self.input_tokens * 3e-6 + self.output_tokens * 1.5e-5
        return self.input_tokens * 5e-6 + self.output_tokens * 2.5e-5


class _FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", critic_threshold=0.5, **kw):
        self.memory = memory
        self.model = model
        self.usage = _FakeUsage()
        self.last_critic_score = None
        self.extra_system = None
        self.messages = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        time.sleep(0.02)  # simulate work
        # Output size depends on model tier (cheaper → terser).
        if "haiku" in self.model:
            self.usage.input_tokens += 120
            self.usage.output_tokens += 40
            return f"[{self.model}] short answer to: {prompt[:50]}"
        if "sonnet" in self.model:
            self.usage.input_tokens += 200
            self.usage.output_tokens += 100
            return f"[{self.model}] balanced answer to: {prompt[:50]}"
        self.usage.input_tokens += 400
        self.usage.output_tokens += 200
        return f"[{self.model}] thorough answer to: {prompt[:50]}"

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = _FakeUsage()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="driver_demo_"))
    print(f"== Multi-tenant Coordination Engine demo ==  (workspace: {tmp})\n")

    # --- runtime setup ---------------------------------------------
    runtime = Runtime(
        memory=Memory(path=tmp / "memory.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=_FakeAgent,
    )

    # --- tenant policy ---------------------------------------------
    policy = PolicyManager(audit_path=tmp / "audit.jsonl")
    policy.set_limits(TenantLimits(
        tenant_id="acme",
        daily_cost_usd=10.00,            # generous: acme is the paying customer
        hourly_cost_usd=2.00,
        max_concurrent_sessions=5,
    ))
    policy.set_limits(TenantLimits(
        tenant_id="trial",
        daily_cost_usd=0.30,             # tight: trial gets squeezed mid-batch
        hourly_cost_usd=0.20,
        max_concurrent_sessions=2,
        max_prompts_per_minute=10,
    ))

    advisor = AdmissionAdvisor(
        runtime.estimator,
        policy=policy,
        runtime=runtime,
        max_cost_per_turn_usd=0.25,       # opus barely exceeds → triggers downgrade
        min_p_success=0.50,
    )

    driver = RuntimeDriver(
        runtime=runtime,
        advisor=advisor,
        policy=policy,
        receipts_path=tmp / "receipts.jsonl",
        max_concurrent=4,
    )

    # --- intents to dispatch ---------------------------------------
    intents = [
        ("acme",  "Summarize Q4 earnings call for tech sector"),
        ("acme",  "Outline a research approach to AI safety governance"),
        ("acme",  "Compare RAG vs fine-tuning for enterprise search"),
        ("trial", "Quick analysis: what is transformer architecture?"),
        ("trial", "Quick: best practices for prompt engineering?"),
        ("trial", "Survey: notable LLM benchmarks 2024-2025"),
        ("trial", "Cheap question: capital of France?"),
        ("trial", "Cheap question: explain entropy briefly"),
        ("acme",  "Long form: full analysis of multi-agent coordination patterns"),
        ("acme",  "Short check: validate this YAML schema"),
    ]

    print(f"Submitting {len(intents)} tickets across 2 tenants…\n")

    # Submit concurrently — driver returns Tickets immediately.
    tickets = []
    for tenant, intent in intents:
        # acme caps at $0.20/ticket, trial caps at $0.05/ticket.
        budget = 0.20 if tenant == "acme" else 0.05
        ticket = driver.submit(TicketRequest(
            intent=intent,
            tenant_id=tenant,
            budget_usd=budget,
        ))
        tickets.append(ticket)
        # Admission decision is added synchronously before submit() returns.
        adm = next((d for d in ticket.decisions() if d.kind == "admission"), None)
        verdict = adm.payload.get("verdict") if adm else "?"
        downgraded = any(d.kind == "downgrade" for d in ticket.decisions())
        tag = f"{verdict}{'→downgrade' if downgraded else ''}"
        print(f"  → ticket {ticket.id}  tenant={tenant:<5}  {tag:<22}  {intent[:42]!r}")

    print("\nWaiting for completion…\n")
    for t in tickets:
        t.result(timeout=30.0)

    # --- per-ticket breakdown --------------------------------------
    print("\n== Receipts ==")
    print(f"{'ticket':<14}{'tenant':<7}{'status':<11}{'model':<22}{'est$':>8}{'actual$':>10}{'dec':>5}")
    print("-" * 80)
    total_actual = 0.0
    total_estimated = 0.0
    for t in tickets:
        r = t.receipt
        total_actual += r.actual_cost_usd
        total_estimated += r.estimated_cost_usd
        print(f"{r.ticket_id:<14}{(r.tenant_id or '-'):<7}{r.status:<11}{(r.model or '-'):<22}"
              f"{r.estimated_cost_usd:>8.4f}{r.actual_cost_usd:>10.4f}{len(r.decisions):>5}")

    # --- tenant rollup ---------------------------------------------
    print("\n== Tenant rollup ==")
    by_tenant: dict[str, dict] = {}
    for t in tickets:
        r = t.receipt
        b = by_tenant.setdefault(r.tenant_id or "-", {"n": 0, "ok": 0, "rejected": 0,
                                                      "deferred": 0, "cost": 0.0})
        b["n"] += 1
        b["cost"] += r.actual_cost_usd
        if r.status == "completed":
            b["ok"] += 1
        elif r.status == "rejected":
            b["rejected"] += 1
        elif r.status == "deferred":
            b["deferred"] += 1
    for tenant, b in by_tenant.items():
        limits = policy.get_limits(tenant)
        cap = limits.daily_cost_usd
        print(f"  {tenant:<6}  submitted={b['n']:>2}  ok={b['ok']:>2}  "
              f"rejected={b['rejected']:>2}  deferred={b['deferred']:>2}  "
              f"cost=${b['cost']:.4f}  daily_cap=${cap:.4f}")

    # --- fleet stats -----------------------------------------------
    stats = driver.stats()
    print(f"\n== Driver stats ==")
    for k, v in stats.items():
        print(f"  {k:<14} {v}")
    print(f"  total_estimated  ${total_estimated:.4f}")
    print(f"  total_actual     ${total_actual:.4f}")
    if total_estimated > 0:
        variance = (total_actual - total_estimated) / total_estimated * 100
        print(f"  forecast error   {variance:+.1f}%")

    # --- causal trace of one ticket --------------------------------
    print("\n== Causal trace of first ticket ==")
    if tickets:
        for dec in tickets[0].decisions():
            payload = json.dumps(dec.payload, default=str, sort_keys=True)
            if len(payload) > 90:
                payload = payload[:87] + "..."
            print(f"  [{time.strftime('%H:%M:%S', time.localtime(dec.ts))}] {dec.kind:<12} {payload}")

    print(f"\nReceipts JSONL → {tmp / 'receipts.jsonl'}")
    print(f"Audit JSONL    → {tmp / 'audit.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
