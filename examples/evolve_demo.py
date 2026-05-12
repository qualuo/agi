"""Reference: closed-loop self-improvement via the Evolution Engine.

A coordination engine that wants the runtime to *get measurably better
over time* runs this loop:

  1. Bootstrap a SelfEvalBank from real successful traces (or hand-author
     a benchmark).
  2. Define seed strategies (population) — different (model, effort, role,
     skill overlay) configurations.
  3. Call `EvolutionEngine.evolve(...)` for N generations. Each generation:
     - evaluates every strategy on the bench,
     - selects top-k by fitness = pass_rate − cost_weight × mean_cost,
     - mutates parents into new candidate children,
     - eval-gates and promotes winners back into:
         * CapabilityRegistry  (durable trace of what worked)
         * PolicyRouter        (Thompson posteriors bias toward winners)
         * SkillLibrary        (mined skill from successful patterns)
         * SelfEvalBank        (regression items grow automatically).

The result is an `EvolutionResult` with per-generation curves a UI can
chart, a `best_strategy`, and a list of `PromotionRecord`s. This is the
artifact a coordination engine displays as "the system is working".

Run hermetically (no API key) with the FakeAgent runner below. To run
against real Opus, build the runner with `evolve.runtime_runner(rt)`
where `rt` is a real `Runtime()`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.capabilities import CapabilityRegistry
from agi.evolve import (
    EvolutionEngine,
    Strategy,
    default_seed_strategies,
)
from agi.policy import PolicyRouter
from agi.runtime import SessionConfig
from agi.selfeval import EvalItem, SelfEvalBank
from agi.skills import SkillLibrary


def _fake_landscape(strategy: Strategy, item: EvalItem) -> tuple[bool, str, float]:
    """A toy fitness landscape for the hermetic demo.

    Reality: opus + verifying nudge + skill overlay outperforms haiku.
    Cost grows with effort. Sonnet is in between. Mutations that combine
    a careful nudge with a strong model should dominate.
    """
    base_cost = {"low": 0.001, "medium": 0.005, "high": 0.02}[strategy.config.effort]
    model_strength = {
        "claude-haiku-4-5-20251001": 0.3,
        "claude-sonnet-4-6": 0.7,
        "claude-opus-4-7": 0.95,
    }.get(strategy.config.model, 0.5)
    nudge_bonus = 0.05 if strategy.system_prompt_extra and "Verify" in strategy.system_prompt_extra else 0.0
    skill_bonus = 0.05 * len(strategy.skill_overlay)
    pass_p = min(1.0, model_strength + nudge_bonus + skill_bonus)
    # Deterministic threshold based on item id so every strategy gets a stable score.
    threshold_seed = (hash(item.id) & 0xFF) / 255.0
    passed = pass_p >= threshold_seed
    text = "answer: " + (item.expect_substring or "ok") if passed else "I'm not sure"
    cost = base_cost * (1.5 if model_strength > 0.9 else 1.0)
    return passed, text, cost


def _bench() -> list[EvalItem]:
    return [
        EvalItem(id=f"q{i}", prompt=f"What is {i} doubled?",
                 expect_substring=str(i * 2))
        for i in range(1, 7)
    ]


def evolve_demo() -> None:
    print("== AGI Evolution Engine demo ==\n")
    print("This shows how a coordination engine asks the runtime to")
    print("self-improve, with eval-gated promotion of winning strategies.\n")

    bank = SelfEvalBank(path=Path("/tmp/agi-demo-selfeval.jsonl"))
    # Seed bench items
    bench = _bench()
    for it in bench:
        bank.add(prompt=it.prompt, expect_substring=it.expect_substring,
                 source="explicit", tags=["demo"])

    # Wire up promotion targets.
    registry = CapabilityRegistry(path=Path("/tmp/agi-demo-cap.jsonl"))
    policy = PolicyRouter(registry=registry)
    skills = SkillLibrary(path=Path("/tmp/agi-demo-skills"))

    engine = EvolutionEngine(
        runner=_fake_landscape,
        registry=registry,
        policy=policy,
        skill_library=skills,
        eval_bank=bank,
        cost_weight=2.0,
        seed=42,
    )

    seeds = default_seed_strategies()
    print(f"Starting population: {[s.name for s in seeds]}")
    print(f"Benchmark size: {len(bench)} items\n")

    result = engine.evolve(
        seeds,
        bench,
        generations=4,
        top_k=2,
        children_per_gen=3,
        promote_each_generation=True,
    )

    summary = result.summary()
    print("Per-generation fitness:")
    for i, (fit, pr, mc) in enumerate(zip(
        summary["fitness_curve"],
        summary["pass_rate_curve"],
        summary["mean_cost_curve"],
    )):
        bar = "█" * max(1, int(20 * pr))
        print(f"  gen {i}: fitness={fit:+.3f}  pass={pr:.0%} {bar}  mean_cost=${mc:.4f}")

    print(f"\nBest strategy: {result.best_strategy.name}")
    print(f"  model={result.best_strategy.config.model}")
    print(f"  effort={result.best_strategy.config.effort}")
    print(f"  role={result.best_strategy.config.role}")
    if result.best_strategy.system_prompt_extra:
        print(f"  nudge: {result.best_strategy.system_prompt_extra[:80]}")
    if result.best_strategy.skill_overlay:
        print(f"  skills: {result.best_strategy.skill_overlay}")
    print(f"\nFitness improved across run: {result.improved()}")

    print(f"\nPromotions made: {sum(1 for p in result.promotions if p.promoted)}/{len(result.promotions)}")
    for p in result.promotions:
        if p.promoted:
            mark = "+ NEW" if p.skill_was_new else "  (existing)"
            print(f"  [{p.strategy_name}] skill={p.skill_saved or '<none>'}{mark}, "
                  f"cap_records={p.capability_records}, arm={p.policy_arm}")
        else:
            print(f"  [{p.strategy_name}] REJECTED: {p.rejected_reason}")

    print(f"\nFinal state:")
    print(f"  CapabilityRegistry records: {len(registry.all())}")
    print(f"  PolicyRouter arms: {len(policy.arms)}")
    print(f"  SkillLibrary skills: {len(skills.all())}")
    print(f"  SelfEvalBank items: {len(bank.all())}")
    print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
    print(f"Total time: {result.total_duration_seconds:.2f}s")

    print("\n--- Coordination engine integration ---")
    print("A real coordination engine would now:")
    print("  1. Inspect `result.best_strategy` and use it as the default for")
    print("     similar prompts (the registry already biases recommend()).")
    print("  2. Read `result.promotions` to surface new skills for review.")
    print("  3. Display `summary['fitness_curve']` in a dashboard.")
    print("  4. Schedule the next evolve() run on the next eval window.")


if __name__ == "__main__":
    evolve_demo()
