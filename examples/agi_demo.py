"""End-to-end demo: Runtime + Coordinator + AutonomousLoop + Fork + Capabilities.

What this shows in one runnable script (no API key required, uses a
deterministic stub agent):

  1. The Runtime is *the* coordination-engine surface: capabilities,
     sessions, event stream, cost accounting.
  2. The reference Coordinator decomposes a Goal into a Plan and drives
     it via the Runtime's task queue.
  3. The AutonomousLoop retries failed goals with accumulated lessons,
     mining a Skill candidate on success — durable improvement.
  4. SessionFork races N variants in parallel and picks the best by
     critic score — instant pass-rate lift on hard prompts.
  5. CapabilityRegistry records every observation; future routing
     learns from past success/failure — observed-performance routing.

Run:  python examples/agi_demo.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.autoloop import AutonomousLoop, promote_skill
from agi.capabilities import CapabilityRegistry
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.events import Event
from agi.fork import ForkVariant, SessionFork
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


# --- A small family of role-aware stub agents -------------------------------


class DemoAgent(FakeAgent):
    """Stub agent whose answer & critic-score depend on its assigned role.

    Lets us run the full pipeline (with retries, forking, capability
    routing) deterministically without burning API tokens. Production
    deployments swap this for the real `agi.agent.Agent`.
    """

    BANK = {
        "math":     ("the sum is 42", 0.92),
        "writer":   ("once upon a time", 0.60),
        "planner":  ("step 1: do thing", 0.55),
        "critic":   ("looks fine", 0.45),
        "junk":     ("",          None),  # empty result — should lose races
    }

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.received_prompts.append(prompt)
        self.usage.input_tokens += 100
        self.usage.output_tokens += 50
        self.usage.turns += 1

        role = "writer"
        if self.extra_system and "Role:" in self.extra_system:
            role = self.extra_system.split("Role:", 1)[1].split(".", 1)[0].strip()
        elif "math" in prompt.lower() or "sum" in prompt.lower() or "add" in prompt.lower():
            role = "math"
        resp, score = self.BANK.get(role, ("default", 0.50))
        self.last_critic_score = score
        # Simulate that "junk" sometimes returns "" then learns
        return resp


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agi_demo_"))
    print(f"workspace: {tmp}")

    # 1) Stand up the runtime ------------------------------------------------
    runtime = Runtime(
        memory=Memory(path=tmp / "memory.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=DemoAgent,
    )

    # Live event log — what a coordination engine subscribes to
    seen_events: list[Event] = []
    def log(e: Event) -> None:
        seen_events.append(e)
        if e.kind.startswith(("autoloop.", "fork.", "coordinator.", "task.")):
            print(f"  ⟶ {e.kind}  {dict((k, v) for k, v in e.data.items() if k != 'plan')}")
    runtime.subscribe(log)

    caps = CapabilityRegistry(path=tmp / "capabilities.jsonl")

    banner("1. runtime capabilities (what a coordinator discovers)")
    cap = runtime.capabilities()
    print(f"  models:    {len(cap['models'])} configured")
    print(f"  skills:    {len(cap['skills'])} in library")
    print(f"  active:    {cap['active_sessions']} sessions")

    # 2) Coordinator drives a multi-step plan --------------------------------
    banner("2. coordinator decomposes a Goal into a Plan and runs it")

    def planner(goal: Goal) -> Plan:
        return Plan(steps=[
            PlanStep(id="research", prompt=f"Research: {goal.intent}", role="planner"),
            PlanStep(id="answer", prompt=f"Answer: {goal.intent}", role="math",
                     depends_on=["research"]),
            PlanStep(id="check", prompt="Verify the answer above", role="critic",
                     depends_on=["answer"]),
        ])

    coord = Coordinator(runtime, decomposer=planner)
    goal = Goal(intent="compute the answer", budget_usd=0.10)
    cr = coord.run(goal)
    print(f"  steps run: {len(cr.outcomes)}   cost: ${cr.total_cost_usd:.4f}   success: {cr.success}")

    # 3) Autonomous loop pursues a harder goal -------------------------------
    banner("3. autonomous loop — retry-with-lessons until success")
    loop = AutonomousLoop(coord, max_iterations=3, capabilities=caps)
    hard_goal = Goal(
        intent="produce the canonical sum",
        acceptance=lambda t: "42" in t,
        budget_usd=0.10,
        metadata={"skill_name": "canonical_sum", "tag": "demo"},
    )
    auto = loop.pursue(hard_goal)
    print(f"  iterations: {len(auto.iterations)}   success: {auto.success}   cost: ${auto.total_cost_usd:.4f}")
    if auto.skill_candidate:
        print(f"  mined skill candidate: {auto.skill_candidate.suggested_name!r}")
        promoted = promote_skill(runtime, auto.skill_candidate)
        if promoted:
            print(f"  promoted skill into library: {promoted.name}")

    # 4) Fork — race 3 variants of the same prompt ---------------------------
    banner("4. fork — race N variants, pick best by critic score")
    fork = SessionFork(runtime, max_workers=3)
    variants = [
        ForkVariant("math",    SessionConfig(system_prompt_extra="Role: math.", role="math")),
        ForkVariant("writer",  SessionConfig(system_prompt_extra="Role: writer.", role="writer")),
        ForkVariant("planner", SessionConfig(system_prompt_extra="Role: planner.", role="planner")),
    ]
    race = fork.race("Give me the sum", variants)
    print(f"  winner: {race.winner.variant.name!r}   score: {race.winner.judge_score:.2f}   cost: ${race.total_cost_usd:.4f}")
    for o in race.outcomes:
        print(f"    {o.variant.name:>8s}  critic={o.critic_score!s:>5s}  result={(o.result or '')[:40]!r}")

    # 5) Capability registry now has data → recommend for next prompt --------
    banner("5. capability routing — what should the coordinator pick next?")
    # A coordinator typically counts an outcome as a success only above some
    # quality bar — here, critic_score ≥ 0.8. That's how the registry learns
    # to prefer the role that actually produces good answers, not just any
    # answer.
    QUALITY_BAR = 0.8
    for o in race.outcomes:
        caps.record(
            prompt="Give me the sum",
            role=o.variant.config.role or "executor",
            model=o.variant.config.model,
            skills_used=[],
            success=(
                o.status == "done"
                and bool((o.result or "").strip())
                and (o.critic_score is not None and o.critic_score >= QUALITY_BAR)
            ),
            cost_usd=o.cost_usd,
            duration_seconds=o.duration_seconds,
            critic_score=o.critic_score,
        )
    rec = caps.recommend("Give me the sum")
    print(f"  recommendation: role={rec.role!r}  model={rec.model!r}")
    print(f"  evidence:       {rec.evidence_count} similar trace(s)   confidence={rec.confidence:.2f}")
    print(f"  predicted:      {rec.expected_success_rate:.0%} success at ~${rec.expected_cost_usd:.4f}")
    print(f"  rationale:      {rec.rationale}")

    # 6) Runtime-level summary -----------------------------------------------
    banner("6. runtime telemetry — what a dashboard sees")
    m = runtime.metrics()
    print(f"  sessions created: {m['sessions_created']}")
    print(f"  chats completed:  {m['chats_completed']}")
    print(f"  total cost:       ${m['total_cost_usd']:.4f}")
    print(f"  total turns:      {m['total_turns']}")
    print(f"  uptime:           {m['uptime_seconds']:.2f}s")
    print(f"  events emitted:   {len(seen_events)}")

    print("\ndone.")


if __name__ == "__main__":
    main()
