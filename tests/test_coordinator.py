"""Coordinator DAG executor."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import EventBus, Memory, Runtime, SkillLibrary
from coordinator import DAG, Node, run
from tests._fakes import counting_factory


class TestDAG(unittest.TestCase):
    def test_topo_order_simple_chain(self):
        dag = DAG([
            Node("a"), Node("b", deps=["a"]), Node("c", deps=["b"]),
        ])
        order = [n.name for n in dag.topo_order()]
        self.assertEqual(order, ["a", "b", "c"])

    def test_cycle_detection(self):
        with self.assertRaises(ValueError):
            DAG([Node("a", deps=["b"]), Node("b", deps=["a"])]).topo_order()

    def test_unknown_dep_rejected(self):
        with self.assertRaises(ValueError):
            DAG([Node("a", deps=["nope"])])

    def test_duplicate_node_rejected(self):
        with self.assertRaises(ValueError):
            DAG([Node("a"), Node("a")])


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.factory, self.counter = counting_factory("done")
        self.rt = Runtime(
            skill_library=SkillLibrary(path=tmp / "s"),
            memory=Memory(path=tmp / "m.jsonl"),
            bus=EventBus(history=128),
            agent_factory=self.factory,
        )

    def tearDown(self):
        self.rt.close_all()
        self._tmp.cleanup()

    def test_runs_in_topo_order_and_threads_outputs(self):
        dag = DAG([
            Node("plan", role="planner", prompt="plan: {ask}"),
            Node("act",  role="executor", prompt="execute: {plan}", deps=["plan"]),
        ])
        results = run(dag, self.rt, inputs={"ask": "summarize README"})
        self.assertEqual([r.name for r in results], ["plan", "act"])
        self.assertEqual(self.counter["n"], 2)
        # all sessions closed
        self.assertEqual(self.rt.list_sessions(), [])

    def test_inputs_substitute(self):
        captured: list[str] = []
        dag = DAG([Node("solo", prompt="answer this: {q}")])
        run(dag, self.rt, inputs={"q": "what is 2+2?"},
            on_node_start=lambda _n, p: captured.append(p))
        self.assertIn("what is 2+2?", captured[0])

    def test_callbacks_fire(self):
        starts, finishes = [], []
        dag = DAG([Node("x"), Node("y", deps=["x"])])
        run(
            dag, self.rt,
            on_node_start=lambda n, _p: starts.append(n.name),
            on_node_finish=lambda n, _r: finishes.append(n.name),
        )
        self.assertEqual(starts, ["x", "y"])
        self.assertEqual(finishes, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
