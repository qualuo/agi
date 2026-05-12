"""Tests for the RuntimePool — federation across runtimes."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.pool import (
    POOL_DISPATCH_COMPLETED,
    POOL_NODE_ADDED,
    PoolDispatch,
    RuntimeNode,
    RuntimePool,
)
from agi.runtime import Runtime, SessionConfig
from agi.skills import Skill, SkillLibrary
from tests.test_runtime import FakeAgent


def _make_node(node_id: str, *, tags: tuple[str, ...] = (), weight: float = 1.0,
               skills: list[Skill] | None = None) -> RuntimeNode:
    tmp = Path(tempfile.mkdtemp())
    skill_lib = SkillLibrary(path=tmp / "skills")
    for s in skills or []:
        skill_lib.save(s)
    runtime = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=skill_lib,
        agent_factory=FakeAgent,
    )
    return RuntimeNode(node_id=node_id, runtime=runtime, tags=tags, weight=weight)


class TestPoolLifecycle(unittest.TestCase):
    def test_add_remove_emit_events(self):
        pool = RuntimePool()
        events: list[str] = []
        pool.bus.subscribe(lambda e: events.append(e.kind))
        pool.add_node(_make_node("a"))
        pool.add_node(_make_node("b"))
        self.assertEqual(pool.nodes()[0]["node_id"] in {"a", "b"}, True)
        self.assertEqual(len(pool.nodes()), 2)
        self.assertIn(POOL_NODE_ADDED, events)
        self.assertTrue(pool.remove_node("a"))
        self.assertEqual(len(pool.nodes()), 1)

    def test_duplicate_node_raises(self):
        pool = RuntimePool()
        pool.add_node(_make_node("a"))
        with self.assertRaises(ValueError):
            pool.add_node(_make_node("a"))


class TestPoolSelection(unittest.TestCase):
    def test_select_prefers_skill_match(self):
        # Two nodes; one has a skill whose tokens overlap the prompt.
        s = Skill(name="summarize_pdf", description="summarize a PDF document", body="…")
        skilled = _make_node("skilled", skills=[s])
        bare = _make_node("bare")
        pool = RuntimePool()
        pool.add_node(skilled)
        pool.add_node(bare)
        chosen = pool.select("please summarize this PDF for me")
        self.assertEqual(chosen.node_id, "skilled")

    def test_select_filters_unhealthy(self):
        pool = RuntimePool()
        a = _make_node("a")
        b = _make_node("b")
        pool.add_node(a)
        pool.add_node(b)
        pool.mark_unhealthy("a")
        chosen = pool.select("anything")
        self.assertEqual(chosen.node_id, "b")

    def test_select_with_tag(self):
        pool = RuntimePool()
        pool.add_node(_make_node("gpu", tags=("gpu",)))
        pool.add_node(_make_node("cpu", tags=("cpu",)))
        self.assertEqual(pool.select("x", require_tag="gpu").node_id, "gpu")
        self.assertEqual(pool.select("x", require_tag="cpu").node_id, "cpu")

    def test_select_no_nodes_raises(self):
        pool = RuntimePool()
        with self.assertRaises(RuntimeError):
            pool.select("anything")

    def test_load_penalty_pushes_to_idle_node(self):
        pool = RuntimePool()
        busy = _make_node("busy", weight=1.0)
        idle = _make_node("idle", weight=1.0)
        busy.in_flight = 5
        pool.add_node(busy)
        pool.add_node(idle)
        self.assertEqual(pool.select("anything").node_id, "idle")


class TestPoolDispatch(unittest.TestCase):
    def test_dispatch_runs_on_some_node(self):
        pool = RuntimePool()
        pool.add_node(_make_node("a"))
        events: list[str] = []
        pool.bus.subscribe(lambda e: events.append(e.kind))
        d = pool.dispatch("hello")
        self.assertIsInstance(d, PoolDispatch)
        self.assertIsNone(d.error)
        self.assertEqual(d.final_text, "ok")
        self.assertEqual(d.node_id, "a")
        self.assertIn(POOL_DISPATCH_COMPLETED, events)

    def test_dispatch_increments_counters(self):
        pool = RuntimePool()
        node = _make_node("a")
        pool.add_node(node)
        for _ in range(3):
            pool.dispatch("x")
        info = pool.nodes()[0]
        self.assertEqual(info["total_dispatched"], 3)
        self.assertEqual(info["in_flight"], 0)


class TestPoolMetrics(unittest.TestCase):
    def test_aggregate_capabilities_unions_skills(self):
        s1 = Skill(name="skill_a", description="thing one", body="")
        s2 = Skill(name="skill_b", description="thing two", body="")
        pool = RuntimePool()
        pool.add_node(_make_node("a", skills=[s1]))
        pool.add_node(_make_node("b", skills=[s2]))
        caps = pool.aggregate_capabilities()
        skill_names = {s["name"] for s in caps["skills"]}
        self.assertEqual(skill_names, {"skill_a", "skill_b"})
        self.assertEqual(caps["node_count"], 2)

    def test_metrics_after_dispatches(self):
        pool = RuntimePool()
        pool.add_node(_make_node("a"))
        pool.dispatch("hello")
        m = pool.metrics()
        self.assertEqual(m["total_dispatches"], 1)
        self.assertEqual(m["successes"], 1)
        self.assertGreater(m["total_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
