"""End-to-end demo: AGI runtime driven by a coordination engine.

Wires the new platform layers together against a FakeAgent so it runs
without an API key:

  Coordinator ─► Runtime ─► Session (FakeAgent)
       │            │
       │            ├─ EventBus  ─►  KnowledgeGraph (auto-ingested)
       │            ├─ Memory
       │            └─ Skills
       │
  AutonomyEngine
       ├─ GoalQueue              # what to attempt next
       ├─ CapabilityRegistry     # observed performance over time
       ├─ PolicyRouter           # bandit routing  (optional)
       ├─ SelfEvalBank           # regression suite (grows from use)
       └─ PolicyManager          # multi-tenant budgets + quotas

Run:
    python examples/agi_autonomy_demo.py
"""
from __future__ import annotations

import json

from agi.autonomy import AutonomyEngine, GoalQueue
from agi.capabilities import CapabilityRegistry
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.events import Event
from agi.governance import GovernedRuntime, PolicyManager, TenantLimits
from agi.knowledge import KnowledgeGraph, attach_to_bus
from agi.policy import PolicyRouter
from agi.runtime import Runtime, SessionConfig
from agi.selfeval import EvalItem, SelfEvalBank


class FakeAgent:
    """Echo agent — passes acceptance tests that look for parts of the prompt."""
    def __init__(self, **kwargs):
        from agi.costs import Usage
        self.usage = Usage()
        self.usage.input_tokens = 250
        self.usage.output_tokens = 80
        self.messages: list = []
        self.last_critic_score = 0.85

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        return f"answer derived from: {prompt[:150]}"

    def reset(self) -> None:
        self.messages = []


def planner(goal: Goal) -> Plan:
    """A trivial decomposer: gather → summarize → answer."""
    return Plan(
        steps=[
            PlanStep(id="gather", prompt=f"gather facts: {goal.intent}", role="researcher"),
            PlanStep(id="answer", prompt=f"answer: {goal.intent}", role="executor",
                     depends_on=["gather"]),
        ],
        rationale="demo plan",
    )


def main() -> None:
    runtime = Runtime(agent_factory=FakeAgent)

    # --- Wire knowledge graph to the event bus ---
    kg = KnowledgeGraph(path="/tmp/demo_kg.jsonl")
    sub_id = attach_to_bus(kg, runtime.bus)

    # --- Capability registry + bandit router ---
    caps = CapabilityRegistry(path="/tmp/demo_caps.jsonl")
    router = PolicyRouter(caps)

    # --- Self-eval bank: starts empty, grows as goals succeed ---
    bank = SelfEvalBank(path="/tmp/demo_eval.jsonl")

    # --- Policy manager: limit acme tenant to $1 / day ---
    pm = PolicyManager()
    pm.set_limits(TenantLimits("acme", daily_cost_usd=1.0, max_prompts_per_minute=30))

    # --- Coordinator + autonomy engine ---
    coord = Coordinator(runtime, decomposer=planner)

    queue = GoalQueue()
    queue.push(Goal(intent="What does the file /etc/hosts contain?",
                    acceptance=lambda t: "answer" in t))
    queue.push(Goal(intent="Summarize the project's README",
                    acceptance=lambda t: "answer" in t))
    queue.push(Goal(intent="Plan a refactor of the runtime module",
                    acceptance=lambda t: "answer" in t))

    engine = AutonomyEngine(
        runtime, coord,
        goal_provider=queue.as_provider(),
        capabilities=caps,
        policy=router,
        eval_bank=bank,
        max_iterations=1,
        max_cost_per_tick_usd=0.50,
        mine_eval_items=True,
    )

    # --- Echo events to stdout so a coordination engine can see them ---
    def on_event(e: Event) -> None:
        if e.kind.startswith("autonomy.") or e.kind.startswith("coordinator."):
            print(f"  event: {e.kind} {json.dumps(e.data, default=str)[:140]}")
    runtime.subscribe(on_event)

    # --- Run the autonomy engine until the queue drains ---
    print("== Running autonomy engine ==")
    reports = engine.run_forever(max_ticks=5, heartbeat_seconds=0.0, idle_grace_ticks=1)

    print("\n== Autonomy metrics ==")
    print(json.dumps(engine.metrics(), indent=2, default=str))

    print("\n== Knowledge graph summary ==")
    print(json.dumps(kg.summary(), indent=2, default=str))

    print("\n== Capability registry stats ==")
    print(json.dumps(caps.stats(), indent=2, default=str))

    print("\n== Policy manager snapshot ==")
    print(json.dumps(pm.snapshot(), indent=2, default=str))

    print("\n== SelfEval bank ==")
    print(json.dumps(bank.stats(), indent=2, default=str))

    # --- Show governance enforcement ---
    print("\n== Multi-tenant governance demo ==")
    gr = GovernedRuntime(runtime, pm)
    pm.set_limits(TenantLimits("free-tier", daily_cost_usd=0.0001))
    sid = gr.create_session("free-tier", SessionConfig())
    try:
        gr.chat("free-tier", sid, "this will exceed the budget",
                estimated_cost_usd=0.10)
    except PermissionError as e:
        print(f"  blocked by policy: {e}")
    gr.end_session("free-tier", sid)

    runtime.bus.unsubscribe(sub_id)
    print("\nDone.")


if __name__ == "__main__":
    main()
