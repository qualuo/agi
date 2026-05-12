"""Runtime-engine demo: PolicyRouter + Pool + Protocol + SelfEval + GoalCompiler.

The pitch in one runnable script (no API key required):

  1. **PolicyRouter** turns the capability registry into a learning
     bandit. The more it runs, the better the (role, model, effort)
     choice for the next prompt.
  2. **RuntimePool** federates many Runtimes behind one dispatch
     interface — capability-aware, load-aware, healthy-only routing.
  3. **CoordinationProtocol** exposes the Runtime over newline-JSON-RPC
     so any external coordination engine (in any language) can drive
     it like a subprocess.
  4. **SelfEvalBank** mines a regression suite from successful traces
     and gates skill/tool promotions on no-regression.
  5. **GoalCompiler** turns a high-level Goal into a parallel/sequential
     Plan automatically — heuristic-first, LLM-fallback.

Run:  python examples/runtime_engine_demo.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.capabilities import CapabilityRegistry
from agi.coordinator import Coordinator, Goal
from agi.goalc import chained_decomposer, heuristic_decomposer, llm_decomposer
from agi.memory import Memory
from agi.policy import Arm, PolicyRouter
from agi.pool import RuntimeNode, RuntimePool
from agi.protocol import CoordinationProtocol
from agi.runtime import Runtime, SessionConfig
from agi.selfeval import SelfEvalBank
from agi.skills import Skill, SkillLibrary
from tests.test_runtime import FakeAgent


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def _runtime(tag: str = "") -> Runtime:
    tmp = Path(tempfile.mkdtemp(prefix=f"agi_re_{tag}_"))
    return Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )


def section_policy() -> None:
    banner("1. PolicyRouter — Thompson-sampled adaptive routing")
    tmp = Path(tempfile.mkdtemp())
    reg = CapabilityRegistry(path=tmp / "caps.jsonl")
    # Seed history: one arm dominates in this context.
    for _ in range(20):
        reg.record(prompt="summarize a research paper", role="writer",
                   model="claude-sonnet-4-6", success=True,
                   cost_usd=0.005, duration_seconds=1.0)
    for _ in range(20):
        reg.record(prompt="summarize a research paper", role="executor",
                   model="claude-opus-4-7", success=False,
                   cost_usd=0.05, duration_seconds=1.0)
    router = PolicyRouter(
        reg,
        arms=[
            Arm(role="writer", model="claude-sonnet-4-6"),
            Arm(role="executor", model="claude-opus-4-7"),
        ],
        epsilon=0.0,
    )
    picks: dict[str, int] = {}
    for _ in range(30):
        d = router.decide("summarize the latest research paper")
        picks[d.arm.role] = picks.get(d.arm.role, 0) + 1
    print(f"  picks over 30 decisions: {picks}")
    print(f"  → router learned to prefer {max(picks, key=picks.get)!r}")


def section_pool() -> None:
    banner("2. RuntimePool — federate many runtimes")
    pool = RuntimePool()
    # Node A specialises in PDFs, node B is bare.
    rt_a = _runtime("pdf")
    rt_a.skills.save(Skill(name="summarize_pdf",
                           description="summarize a PDF document", body="..."))
    rt_b = _runtime("bare")
    pool.add_node(RuntimeNode(node_id="pdf-specialist", runtime=rt_a, tags=("gpu",)))
    pool.add_node(RuntimeNode(node_id="generalist",     runtime=rt_b, tags=("cpu",)))

    # Dispatch picks the skilled node.
    d = pool.dispatch("please summarize this PDF for me")
    print(f"  dispatch routed to: {d.node_id}  (cost ${d.cost_usd:.4f})")

    # Aggregate capabilities = federation-wide view.
    caps = pool.aggregate_capabilities()
    print(f"  federation: {caps['node_count']} nodes, "
          f"{len(caps['skills'])} unique skills, "
          f"{caps['healthy_node_count']} healthy")


def section_protocol() -> None:
    banner("3. CoordinationProtocol — JSON-RPC stdio surface")
    proto = CoordinationProtocol(_runtime("proto"))
    reader = io.StringIO("\n".join(json.dumps(r) for r in [
        {"jsonrpc": "2.0", "id": 1, "method": "version"},
        {"jsonrpc": "2.0", "id": 2, "method": "runtime.capabilities"},
        {"jsonrpc": "2.0", "id": 3, "method": "session.create",
         "params": {"max_iterations": 3}},
    ]) + "\n")
    writer = io.StringIO()
    proto.serve_streams(reader, writer)
    responses = [json.loads(l) for l in writer.getvalue().splitlines() if l.strip()]
    for r in responses:
        if "id" in r:
            print(f"  reply id={r['id']}: {json.dumps(r.get('result', r.get('error')))[:120]}")
        else:
            print(f"  notify {r.get('method')}: {json.dumps(r.get('params', {}))[:80]}")


def section_selfeval() -> None:
    banner("4. SelfEvalBank — regression suite mined from successes")
    rt = _runtime("se")
    bank = SelfEvalBank(path=Path(tempfile.mkdtemp()) / "se.jsonl")
    # Auto-mine an item from a "successful" trace
    item = bank.auto_mine(
        prompt="What is the capital of France?",
        final_text="Paris is the capital of France.",
        critic_score=0.95,
    )
    if item:
        print(f"  mined: {item.id}  prompt={item.prompt!r}")
    bank.add(prompt="hi", expect_substring="ok", source="explicit",
             tags=["smoke"])
    report = bank.run(bank.runtime_runner(rt))
    print(f"  ran {report.total} items: {report.passed} passed, "
          f"{report.failed} failed (pass rate {report.pass_rate:.0%})")
    ok, _ = bank.gate_promotion(bank.runtime_runner(rt), baseline_pass_rate=1.0,
                                allowed_regression=0.0)
    print(f"  gate (no-regression check): {'PASS' if ok else 'BLOCK'}")


def section_goalc() -> None:
    banner("5. GoalCompiler — Goal → Plan automatically")
    rt = _runtime("plan")
    coord = Coordinator(rt, decomposer=chained_decomposer(
        heuristic_decomposer,
        llm_decomposer(rt),  # falls back if heuristic was trivial
        min_steps=2,
    ))
    goal = Goal(intent="analyze the impact of LoRA on inference latency")
    result = coord.run(goal)
    print(f"  steps planned: {len(result.plan.steps)}  "
          f"({[s.id for s in result.plan.steps]})")
    print(f"  rationale: {result.plan.rationale}")
    print(f"  cost: ${result.total_cost_usd:.4f}  success: {result.success}")


def main() -> None:
    print("Runtime-engine demo — five investor-grade modules in one run\n")
    section_policy()
    section_pool()
    section_protocol()
    section_selfeval()
    section_goalc()
    print("\ndone.")


if __name__ == "__main__":
    main()
