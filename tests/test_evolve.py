"""Tests for `agi.evolve` — closed-loop self-improvement.

The runner is dependency-injected so every test stays hermetic. We
construct deterministic runners that decide pass/fail/cost based on
the strategy and item, then assert that fitness, selection, mutation,
and promotion behave correctly.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.capabilities import CapabilityRegistry
from agi.evolve import (
    EvolutionEngine,
    Strategy,
    StrategyOutcome,
    default_seed_strategies,
)
from agi.policy import PolicyRouter
from agi.runtime import SessionConfig
from agi.selfeval import EvalItem, SelfEvalBank
from agi.skills import Skill, SkillLibrary


def _bench(n: int = 3) -> list[EvalItem]:
    return [
        EvalItem(
            id=f"item-{i}",
            prompt=f"compute the value of {i}+{i}",
            expect_substring=str(i + i),
        )
        for i in range(n)
    ]


def _strategy(name: str, *, model="claude-opus-4-7", effort="high",
              role="executor") -> Strategy:
    return Strategy(
        name=name,
        config=SessionConfig(model=model, effort=effort, role=role,
                             use_skills=False),
    )


def _tmp_path(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


class TestStrategy(unittest.TestCase):
    def test_arm_extracts_routing_key(self):
        s = _strategy("s", model="claude-sonnet-4-6", effort="medium",
                      role="planner")
        arm = s.arm()
        self.assertEqual(arm.role, "planner")
        self.assertEqual(arm.model, "claude-sonnet-4-6")
        self.assertEqual(arm.effort, "medium")

    def test_materialized_config_merges_extras(self):
        s = Strategy(
            name="x",
            config=SessionConfig(system_prompt_extra="A"),
            system_prompt_extra="B",
        )
        cfg = s.materialized_config()
        self.assertIn("A", cfg.system_prompt_extra)
        self.assertIn("B", cfg.system_prompt_extra)

    def test_materialized_config_handles_no_extras(self):
        s = Strategy(name="x", config=SessionConfig())
        self.assertIsNone(s.materialized_config().system_prompt_extra)


class TestEvaluate(unittest.TestCase):
    def test_evaluate_records_one_outcome_per_pair(self):
        # Runner: every item passes for every strategy.
        def runner(s: Strategy, it: EvalItem) -> tuple[bool, str, float]:
            return True, it.expect_substring, 0.001

        eng = EvolutionEngine(runner)
        strategies = [_strategy("s1"), _strategy("s2")]
        bench = _bench(3)
        report = eng.evaluate(strategies, bench, generation=0)

        self.assertEqual(len(report.outcomes), 2 * 3)
        for s in strategies:
            self.assertEqual(report.pass_rate_by_strategy[s.name], 1.0)
            # cost_weight=5, mean_cost=0.001 → fitness = 1 - 0.005
            self.assertAlmostEqual(
                report.fitness_by_strategy[s.name], 1.0 - 5.0 * 0.001, places=6
            )
        self.assertEqual(report.total_cost_usd, 6 * 0.001)

    def test_runner_exception_counts_as_failure(self):
        def runner(s, it):
            raise RuntimeError("boom")

        eng = EvolutionEngine(runner)
        report = eng.evaluate([_strategy("s")], _bench(2), generation=0)
        self.assertEqual(report.pass_rate_by_strategy["s"], 0.0)
        # fitness = 0 - cost_weight*0 = 0
        self.assertEqual(report.fitness_by_strategy["s"], 0.0)
        for o in report.outcomes:
            self.assertFalse(o.passed)
            self.assertIn("RuntimeError", o.final_text)

    def test_best_strategy_picks_highest_fitness(self):
        # s1 always passes cheap; s2 always passes expensive.
        def runner(s, it):
            cost = 0.001 if s.name == "cheap" else 0.1
            return True, it.expect_substring, cost

        eng = EvolutionEngine(runner)
        report = eng.evaluate(
            [_strategy("cheap"), _strategy("expensive")],
            _bench(2),
            generation=0,
        )
        self.assertEqual(report.best_strategy_name, "cheap")

    def test_evaluate_rejects_empty_inputs(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0))
        with self.assertRaises(ValueError):
            eng.evaluate([], _bench(1), generation=0)
        with self.assertRaises(ValueError):
            eng.evaluate([_strategy("s")], [], generation=0)


class TestSelect(unittest.TestCase):
    def _report_with_fitness(self, fitness_map: dict[str, float],
                             pass_map: dict[str, float] | None = None):
        strategies = [_strategy(n) for n in fitness_map]
        from agi.evolve import GenerationReport
        return GenerationReport(
            generation=0,
            strategies=strategies,
            outcomes=[],
            fitness_by_strategy=fitness_map,
            pass_rate_by_strategy=pass_map or {n: 1.0 for n in fitness_map},
            mean_cost_by_strategy={n: 0.0 for n in fitness_map},
            best_strategy_name=max(fitness_map, key=lambda k: fitness_map[k]),
            total_cost_usd=0.0,
            duration_seconds=0.0,
        )

    def test_select_returns_top_k(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0))
        report = self._report_with_fitness({"a": 0.1, "b": 0.9, "c": 0.5})
        kept = eng.select(report, top_k=2)
        self.assertEqual([s.name for s in kept], ["b", "c"])

    def test_select_filters_low_pass_rate(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0))
        report = self._report_with_fitness(
            {"a": 0.9, "b": 0.8, "c": 0.5},
            pass_map={"a": 0.1, "b": 0.9, "c": 0.9},
        )
        kept = eng.select(report, top_k=2, min_pass_rate=0.5)
        self.assertEqual({s.name for s in kept}, {"b", "c"})

    def test_select_never_returns_empty(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0))
        # All fail the filter → still get the best one back.
        report = self._report_with_fitness(
            {"a": 0.9, "b": 0.5},
            pass_map={"a": 0.0, "b": 0.0},
        )
        kept = eng.select(report, top_k=2, min_pass_rate=0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].name, "a")


class TestMutate(unittest.TestCase):
    def test_mutate_produces_n_children(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0), seed=42)
        children = eng.mutate(
            [_strategy("p1"), _strategy("p2")],
            num_children=4, generation=1,
        )
        self.assertEqual(len(children), 4)

    def test_mutate_assigns_lineage(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0), seed=42)
        children = eng.mutate(
            [_strategy("parent")], num_children=3, generation=2,
        )
        for c in children:
            self.assertEqual(c.parent_name, "parent")
            self.assertEqual(c.generation, 2)
            self.assertTrue(c.name.startswith("g2-parent-m"))

    def test_mutate_seeds_are_deterministic(self):
        e1 = EvolutionEngine(lambda s, it: (True, "", 0.0), seed=7)
        e2 = EvolutionEngine(lambda s, it: (True, "", 0.0), seed=7)
        c1 = e1.mutate([_strategy("p")], num_children=4, generation=1)
        c2 = e2.mutate([_strategy("p")], num_children=4, generation=1)
        for a, b in zip(c1, c2):
            self.assertEqual(a.config.model, b.config.model)
            self.assertEqual(a.config.effort, b.config.effort)
            self.assertEqual(a.config.role, b.config.role)
            self.assertEqual(a.system_prompt_extra, b.system_prompt_extra)

    def test_mutate_changes_at_least_one_knob(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0), seed=1)
        parent = _strategy("p", model="claude-opus-4-7", effort="high",
                           role="executor")
        # Run many children; assert that *something* differs from the parent
        # in each child (mutation is non-no-op).
        children = eng.mutate([parent], num_children=20, generation=1)
        for c in children:
            differs = (
                c.config.model != parent.config.model
                or c.config.effort != parent.config.effort
                or c.config.role != parent.config.role
                or c.system_prompt_extra != parent.system_prompt_extra
                or c.skill_overlay != parent.skill_overlay
            )
            self.assertTrue(differs, f"mutation no-op for {c.name}")


class TestPromote(unittest.TestCase):
    def test_promotion_records_to_capability_registry(self):
        registry = CapabilityRegistry(path=_tmp_path("cap.jsonl"))
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            registry=registry,
        )
        report = eng.evaluate([_strategy("winner")], _bench(2), generation=0)
        promo = eng.promote(report, baseline_pass_rate=None)
        self.assertTrue(promo.promoted)
        self.assertEqual(promo.capability_records, 2)
        self.assertEqual(len(registry.all()), 2)
        self.assertTrue(all(r.success for r in registry.all()))

    def test_promotion_updates_policy_posteriors(self):
        registry = CapabilityRegistry(path=_tmp_path("cap.jsonl"))
        policy = PolicyRouter(registry=registry)
        before_arms = {a.key() for a in policy.arms}
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            registry=registry,
            policy=policy,
        )
        # Use a non-default arm so we can see it appended.
        s = Strategy(
            name="winner",
            config=SessionConfig(
                model="custom-model", effort="medium", role="custom-role",
                use_skills=False,
            ),
        )
        report = eng.evaluate([s], _bench(3), generation=0)
        promo = eng.promote(report, baseline_pass_rate=None)
        self.assertTrue(promo.promoted)
        new_arms = {a.key() for a in policy.arms}
        self.assertIn(("custom-role", "custom-model", "medium"), new_arms - before_arms)
        self.assertEqual(promo.policy_arm, ("custom-role", "custom-model", "medium"))

    def test_promotion_mines_skill_when_2plus_successes(self):
        skills = SkillLibrary(path=_tmp_path("skills"))
        eng = EvolutionEngine(
            lambda s, it: (True, "the answer is " + it.expect_substring, 0.001),
            skill_library=skills,
        )
        report = eng.evaluate([_strategy("winner")], _bench(3), generation=0)
        promo = eng.promote(report, baseline_pass_rate=None)
        self.assertIsNotNone(promo.skill_saved)
        self.assertTrue(promo.skill_was_new)
        self.assertEqual(len(skills.all()), 1)

    def test_promotion_skips_skill_when_single_success(self):
        skills = SkillLibrary(path=_tmp_path("skills"))

        def runner(s, it):
            # Only the first item passes
            passed = it.id == "item-0"
            return passed, it.expect_substring, 0.001

        eng = EvolutionEngine(runner, skill_library=skills)
        report = eng.evaluate([_strategy("winner")], _bench(3), generation=0)
        promo = eng.promote(report, baseline_pass_rate=None)
        self.assertIsNone(promo.skill_saved)
        self.assertEqual(len(skills.all()), 0)

    def test_promotion_does_not_clobber_existing_skill(self):
        skills = SkillLibrary(path=_tmp_path("skills"))
        # Pre-populate with a skill that has the name evolve will try.
        existing_name = "evolved_winner"
        skills.save(Skill(
            name=existing_name,
            description="existing",
            body="existing body",
        ))
        original_text = skills.get(existing_name).body
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            skill_library=skills,
        )
        report = eng.evaluate([_strategy("winner")], _bench(3), generation=0)
        promo = eng.promote(report, baseline_pass_rate=None)
        self.assertEqual(promo.skill_saved, existing_name)
        self.assertFalse(promo.skill_was_new)
        # Body was not overwritten.
        self.assertEqual(skills.get(existing_name).body.strip(), original_text.strip())

    def test_promotion_eval_gate_rejects_regression(self):
        eng = EvolutionEngine(lambda s, it: (False, "no", 0.001))
        report = eng.evaluate([_strategy("loser")], _bench(2), generation=0)
        promo = eng.promote(report, baseline_pass_rate=0.9)
        self.assertFalse(promo.promoted)
        self.assertIn("baseline", promo.rejected_reason)
        self.assertEqual(promo.capability_records, 0)

    def test_promotion_eval_gate_passes_when_meets_baseline(self):
        registry = CapabilityRegistry(path=_tmp_path("cap.jsonl"))
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            registry=registry,
        )
        report = eng.evaluate([_strategy("winner")], _bench(2), generation=0)
        promo = eng.promote(report, baseline_pass_rate=0.9)
        self.assertTrue(promo.promoted)

    def test_promotion_writes_to_selfeval_bank(self):
        bank = SelfEvalBank(path=_tmp_path("se.jsonl"))
        before = len(bank.all())
        # Long final_text so auto_mine accepts it.
        eng = EvolutionEngine(
            lambda s, it: (True, "the answer is " + it.expect_substring + " here is more text", 0.001),
            eval_bank=bank,
        )
        report = eng.evaluate([_strategy("winner")], _bench(2), generation=0)
        promo = eng.promote(report, baseline_pass_rate=None)
        self.assertTrue(promo.promoted)
        self.assertGreater(len(bank.all()), before)


class TestEvolveLoop(unittest.TestCase):
    def test_evolve_runs_full_loop(self):
        # Simple deterministic landscape: opus-high passes everything,
        # haiku-low fails everything.
        def runner(s, it):
            if s.config.model == "claude-opus-4-7" and s.config.effort == "high":
                return True, it.expect_substring, 0.01
            if s.config.model == "claude-haiku-4-5-20251001":
                return False, "weak", 0.001
            return True, it.expect_substring, 0.005

        eng = EvolutionEngine(runner, seed=1, cost_weight=0.1)
        result = eng.evolve(
            default_seed_strategies(),
            _bench(3),
            generations=3,
            top_k=2,
            children_per_gen=2,
        )
        self.assertEqual(len(result.generations), 3)
        self.assertEqual(len(result.fitness_curve), 3)
        self.assertEqual(len(result.pass_rate_curve), 3)
        self.assertEqual(len(result.mean_cost_curve), 3)
        self.assertEqual(result.benchmark_size, 3)
        # The best strategy should be one that passes all items.
        self.assertEqual(
            result.generations[0].pass_rate_by_strategy[
                result.generations[0].best_strategy_name
            ],
            1.0,
        )

    def test_evolve_with_promotion_grows_registry(self):
        registry = CapabilityRegistry(path=_tmp_path("cap.jsonl"))
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            registry=registry,
            seed=1,
        )
        result = eng.evolve(
            [_strategy("only")],
            _bench(2),
            generations=2,
            top_k=1,
            children_per_gen=1,
            promote_each_generation=True,
        )
        self.assertGreater(len(registry.all()), 0)
        self.assertGreaterEqual(len(result.promotions), 2)
        self.assertTrue(any(p.promoted for p in result.promotions))

    def test_evolve_summary_reports_all_curves(self):
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            seed=2,
        )
        result = eng.evolve(
            [_strategy("a")],
            _bench(2),
            generations=2,
            top_k=1,
            children_per_gen=1,
        )
        summary = result.summary()
        self.assertEqual(summary["generations"], 2)
        self.assertEqual(summary["benchmark_size"], 2)
        self.assertEqual(len(summary["fitness_curve"]), 2)
        self.assertEqual(len(summary["pass_rate_curve"]), 2)
        self.assertEqual(len(summary["mean_cost_curve"]), 2)
        self.assertIn("best_strategy", summary)

    def test_evolve_rejects_invalid_args(self):
        eng = EvolutionEngine(lambda s, it: (True, "", 0.0))
        with self.assertRaises(ValueError):
            eng.evolve([], _bench(1), generations=1)
        with self.assertRaises(ValueError):
            eng.evolve([_strategy("a")], _bench(1), generations=0)

    def test_evolve_skips_promotion_when_disabled(self):
        registry = CapabilityRegistry(path=_tmp_path("cap.jsonl"))
        eng = EvolutionEngine(
            lambda s, it: (True, it.expect_substring, 0.001),
            registry=registry,
            seed=3,
        )
        result = eng.evolve(
            [_strategy("a")],
            _bench(2),
            generations=2,
            promote_each_generation=False,
        )
        self.assertEqual(len(result.promotions), 0)
        self.assertEqual(len(registry.all()), 0)

    def test_improved_flag(self):
        # First gen passes 0 items; second gen passes all items.
        # Force "improvement" via a runner that consults the strategy name.
        def runner(s, it):
            if s.generation == 0 and s.parent_name is None:
                return False, "fail", 0.01
            return True, it.expect_substring, 0.001

        eng = EvolutionEngine(runner, seed=4)
        result = eng.evolve(
            [_strategy("seed")],
            _bench(2),
            generations=2,
            top_k=1,
            children_per_gen=2,
            promote_each_generation=False,
        )
        self.assertTrue(result.improved())


class TestDefaults(unittest.TestCase):
    def test_default_seeds_span_models(self):
        seeds = default_seed_strategies()
        models = {s.config.model for s in seeds}
        # Should hit at least three distinct models.
        self.assertGreaterEqual(len(models), 3)
        for s in seeds:
            self.assertEqual(s.generation, 0)


if __name__ == "__main__":
    unittest.main()
