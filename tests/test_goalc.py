"""Tests for the goal compiler — heuristic + LLM decomposers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.coordinator import Goal, Plan, PlanStep
from agi.goalc import (
    LlmDecomposerConfig,
    chained_decomposer,
    heuristic_decomposer,
    llm_decomposer,
    parse_plan_json,
)
from agi.memory import Memory
from agi.runtime import Runtime
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


class TestHeuristicDecomposer(unittest.TestCase):
    def test_analyze_produces_three_steps(self):
        plan = heuristic_decomposer(Goal(intent="analyze the impact of LoRA"))
        self.assertEqual(len(plan.steps), 3)
        ids = [s.id for s in plan.steps]
        self.assertIn("gather", ids)
        self.assertIn("analyze", ids)
        self.assertIn("summary", ids)

    def test_compare_produces_parallel_probes(self):
        plan = heuristic_decomposer(Goal(intent="compare LoRA and full fine-tuning"))
        roles = {s.role for s in plan.steps}
        self.assertIn("researcher", roles)
        # Both probes should be at the front with no dependencies.
        no_deps = [s for s in plan.steps if not s.depends_on]
        self.assertEqual(len(no_deps), 2)

    def test_build_pattern(self):
        plan = heuristic_decomposer(Goal(intent="build a regex compiler"))
        ids = [s.id for s in plan.steps]
        self.assertEqual(ids, ["design", "implement", "test"])

    def test_find_summarize_pattern(self):
        plan = heuristic_decomposer(Goal(intent="find references to attention"))
        self.assertEqual(len(plan.steps), 2)

    def test_no_match_falls_back_to_single_step(self):
        plan = heuristic_decomposer(Goal(intent="qrxz xyzzy"))
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].id, "root")


class TestParsePlanJson(unittest.TestCase):
    def test_parses_valid_plan(self):
        blob = json.dumps({
            "rationale": "split into research and write",
            "steps": [
                {"id": "research", "role": "researcher", "prompt": "find sources"},
                {"id": "write", "role": "writer", "prompt": "write up",
                 "depends_on": ["research"]},
            ],
        })
        plan = parse_plan_json(blob)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[1].depends_on, ["research"])

    def test_strips_code_fences(self):
        blob = (
            "```json\n"
            '{"rationale": "x", "steps": [{"id": "a", "prompt": "p"}]}\n'
            "```"
        )
        plan = parse_plan_json(blob)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 1)

    def test_handles_surrounding_prose(self):
        blob = (
            "Sure, here's the plan:\n"
            '{"rationale": "x", "steps": [{"id": "a", "prompt": "p"}]}\n'
            "End."
        )
        plan = parse_plan_json(blob)
        self.assertIsNotNone(plan)

    def test_rejects_garbage(self):
        self.assertIsNone(parse_plan_json("nope"))
        self.assertIsNone(parse_plan_json(""))
        self.assertIsNone(parse_plan_json("{"))

    def test_drops_steps_without_id_or_prompt(self):
        blob = json.dumps({
            "rationale": "",
            "steps": [
                {"id": "a", "prompt": "ok"},
                {"id": "", "prompt": "no id"},
                {"id": "c"},  # missing prompt
                {"id": "a", "prompt": "duplicate"},  # dup id
                {"id": "d", "prompt": "good"},
            ],
        })
        plan = parse_plan_json(blob)
        self.assertEqual(len(plan.steps), 2)

    def test_max_steps_caps_plan(self):
        blob = json.dumps({
            "rationale": "",
            "steps": [{"id": f"s{i}", "prompt": "x"} for i in range(10)],
        })
        plan = parse_plan_json(blob, max_steps=3)
        self.assertEqual(len(plan.steps), 3)


class _PlanReturningFakeAgent(FakeAgent):
    """FakeAgent that emits a JSON plan so the llm_decomposer can parse it."""
    _PLAN = {
        "rationale": "decompose",
        "steps": [
            {"id": "step1", "role": "executor", "prompt": "do thing one"},
            {"id": "step2", "role": "writer", "prompt": "do thing two",
             "depends_on": ["step1"]},
        ],
    }

    def chat(self, prompt, max_iterations=25):
        self.received_prompts.append(prompt)
        self.usage.input_tokens += 10
        self.usage.output_tokens += 10
        self.usage.turns += 1
        return json.dumps(self._PLAN)


class _BrokenFakeAgent(FakeAgent):
    def chat(self, prompt, max_iterations=25):
        self.received_prompts.append(prompt)
        self.usage.input_tokens += 10
        self.usage.output_tokens += 10
        return "not json at all"


def _runtime(factory=_PlanReturningFakeAgent) -> Runtime:
    tmp = Path(tempfile.mkdtemp())
    return Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=factory,
    )


class TestLlmDecomposer(unittest.TestCase):
    def test_llm_decomposer_uses_runtime(self):
        rt = _runtime()
        dec = llm_decomposer(rt)
        plan = dec(Goal(intent="do something complex"))
        self.assertEqual(len(plan.steps), 2)
        ids = [s.id for s in plan.steps]
        self.assertIn("step1", ids)
        self.assertIn("step2", ids)

    def test_llm_decomposer_fallback_on_garbage(self):
        rt = _runtime(_BrokenFakeAgent)
        dec = llm_decomposer(rt, config=LlmDecomposerConfig(fallback_on_error=True))
        # heuristic fallback on "analyze ..." → 3 steps
        plan = dec(Goal(intent="analyze something"))
        self.assertEqual(len(plan.steps), 3)


class TestChainedDecomposer(unittest.TestCase):
    def test_first_with_enough_steps_wins(self):
        # First decomposer always returns 3-step heuristic plan
        first = heuristic_decomposer
        second_called = []

        def second(g):
            second_called.append(g)
            return Plan(steps=[PlanStep(id="x", prompt="x")], rationale="")

        chained = chained_decomposer(first, second, min_steps=2)
        plan = chained(Goal(intent="analyze X"))
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(second_called, [])  # second was never called

    def test_falls_through_to_next(self):
        def first(g):
            return Plan(steps=[PlanStep(id="x", prompt="x")], rationale="")

        def second(g):
            return Plan(steps=[
                PlanStep(id="a", prompt="a"),
                PlanStep(id="b", prompt="b"),
            ], rationale="")

        chained = chained_decomposer(first, second, min_steps=2)
        plan = chained(Goal(intent="x"))
        self.assertEqual(len(plan.steps), 2)


if __name__ == "__main__":
    unittest.main()
