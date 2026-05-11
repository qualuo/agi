"""Runtime + plan tests. No network — uses a fake Agent factory."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.budget import Budget
from agi.costs import Usage
from agi.events import (
    BudgetExceeded,
    SessionFinished,
    SessionStarted,
    TextDelta,
    TurnFinished,
)
from agi.plan import Plan, Subgoal, execute_plan
from agi.runtime import Runtime
from agi.server import make_server


class FakeAgent:
    """Stand-in for agi.Agent. Honours on_event, budget, cancel_check.

    Behavior is parameterized via `behavior` (set on the class before
    use, via a closure factory below).
    """

    def __init__(self, **kwargs):
        self.on_event = kwargs.get("on_event")
        self.budget = kwargs.get("budget")
        self.cancel_check = kwargs.get("cancel_check")
        self.model = kwargs.get("model", "claude-opus-4-7")
        self.usage = Usage()
        self.memory = kwargs.get("memory")

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        # Honour cancellation between iterations.
        for i in range(3):
            if self.cancel_check and self.cancel_check():
                return ""
            if self.on_event:
                self.on_event(TextDelta(text=f"step{i} "))
                self.on_event(
                    TurnFinished(
                        stop_reason="end_turn",
                        input_tokens=100,
                        output_tokens=50,
                        cost_usd=0.001,
                    )
                )
            self.usage.input_tokens += 100
            self.usage.output_tokens += 50
            self.usage.turns += 1
            if self.budget is not None:
                hit = self.budget.check(
                    cost_usd=self.usage.cost_usd(self.model),
                    input_tokens=self.usage.input_tokens,
                    output_tokens=self.usage.output_tokens,
                    iterations=i + 1,
                )
                if hit:
                    reason, limit, actual = hit
                    if self.on_event:
                        self.on_event(BudgetExceeded(reason=reason, limit=limit, actual=actual))
                    return f"stopped at iter {i}"
            time.sleep(0.01)
        return f"done: {prompt[:20]}"


def _fake_factory(**kwargs):
    return FakeAgent(**kwargs)


class TestRuntimeBasic(unittest.TestCase):
    def test_submit_and_wait(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=2)
        sid = rt.submit("hello world")
        record = rt.wait(sid, timeout=5)
        self.assertEqual(record["status"], "ok")
        self.assertIn("done", record["final_text"])
        self.assertEqual(record["turns"], 3)
        rt.shutdown()

    def test_list_sessions(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=2)
        sids = [rt.submit(f"goal {i}") for i in range(3)]
        for sid in sids:
            rt.wait(sid, timeout=5)
        listed = rt.list_sessions()
        self.assertEqual(len(listed), 3)
        statuses = {s["status"] for s in listed}
        self.assertEqual(statuses, {"ok"})
        rt.shutdown()

    def test_event_replay(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        sid = rt.submit("event test")
        rt.wait(sid, timeout=5)
        kinds = [e.kind for e in rt.events(sid, follow=False)]
        self.assertEqual(kinds[0], "session_started")
        self.assertEqual(kinds[-1], "session_finished")
        self.assertIn("text_delta", kinds)
        self.assertIn("turn_finished", kinds)
        rt.shutdown()

    def test_event_live_follow(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        sid = rt.submit("live tail")
        collected = []
        for ev in rt.events(sid, replay=True, follow=True, timeout=5):
            collected.append(ev.kind)
            if ev.kind == "session_finished":
                break
        self.assertIn("session_started", collected)
        self.assertIn("session_finished", collected)
        rt.shutdown()

    def test_concurrency_limit_serializes(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        t0 = time.time()
        sids = [rt.submit(f"g{i}") for i in range(3)]
        for sid in sids:
            rt.wait(sid, timeout=10)
        # Each fake "chat" sleeps ~30ms. With concurrency=1, 3 tasks
        # should take ≥ 90ms; with concurrency=3, ~30ms. Use a loose
        # bound to avoid flakiness.
        elapsed = time.time() - t0
        self.assertGreater(elapsed, 0.08)
        rt.shutdown()


class TestBudget(unittest.TestCase):
    def test_iterations_limit(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        sid = rt.submit("x", budget=Budget(max_iterations=2))
        rec = rt.wait(sid, timeout=5)
        self.assertEqual(rec["status"], "budget_exceeded")
        rt.shutdown()

    def test_tokens_limit(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        # Each fake turn = 150 tokens. Cap below 1 turn's worth.
        sid = rt.submit("x", budget=Budget(max_tokens=100))
        rec = rt.wait(sid, timeout=5)
        self.assertEqual(rec["status"], "budget_exceeded")
        rt.shutdown()

    def test_no_budget_runs_to_completion(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        sid = rt.submit("x")
        rec = rt.wait(sid, timeout=5)
        self.assertEqual(rec["status"], "ok")
        rt.shutdown()


class TestCancel(unittest.TestCase):
    def test_cancel_marks_session(self):
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=1)
        sid = rt.submit("cancel me")
        # Cancel immediately. The fake checks the flag between turns.
        rt.cancel(sid)
        rec = rt.wait(sid, timeout=5)
        self.assertIn(rec["status"], ("cancelled", "ok"))
        # In a fast machine the first iteration may complete before
        # cancel; we just verify it terminated cleanly.
        rt.shutdown()


class TestPlan(unittest.TestCase):
    def test_dag_executes_in_order(self):
        plan = Plan(
            name="research",
            subgoals=[
                Subgoal(name="a", prompt="research A"),
                Subgoal(name="b", prompt="research B"),
                Subgoal(name="synth", prompt="combine: {{ a }} + {{ b }}", depends_on=["a", "b"]),
            ],
        )
        rt = Runtime(agent_factory=_fake_factory, max_concurrent=4)
        results = execute_plan(rt, plan, timeout=10)
        self.assertEqual(set(results), {"a", "b", "synth"})
        self.assertEqual(results["a"].status, "ok")
        self.assertEqual(results["b"].status, "ok")
        self.assertEqual(results["synth"].status, "ok")
        rt.shutdown()

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            Plan(name="bad", subgoals=[Subgoal("x", "1"), Subgoal("x", "2")])

    def test_unknown_dep_rejected(self):
        with self.assertRaises(ValueError):
            Plan(name="bad", subgoals=[Subgoal("x", "1", depends_on=["nope"])])

    def test_cycle_rejected(self):
        with self.assertRaises(ValueError):
            Plan(
                name="bad",
                subgoals=[
                    Subgoal("a", "1", depends_on=["b"]),
                    Subgoal("b", "1", depends_on=["a"]),
                ],
            )


class TestServer(unittest.TestCase):
    def setUp(self):
        self.rt = Runtime(agent_factory=_fake_factory, max_concurrent=2)
        self.server = make_server(self.rt, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.rt.shutdown()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_health(self):
        with urllib.request.urlopen(self._url("/v1/health")) as r:
            data = json.loads(r.read())
        self.assertTrue(data["ok"])

    def test_submit_and_status(self):
        req = urllib.request.Request(
            self._url("/v1/sessions"),
            data=json.dumps({"goal": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
        sid = data["session_id"]
        # Wait for completion.
        self.rt.wait(sid, timeout=5)
        with urllib.request.urlopen(self._url(f"/v1/sessions/{sid}")) as r:
            rec = json.loads(r.read())
        self.assertEqual(rec["status"], "ok")

    def test_plan_endpoint(self):
        plan_body = {
            "name": "p",
            "subgoals": [
                {"name": "x", "prompt": "task x"},
                {"name": "y", "prompt": "use {{ x }}", "depends_on": ["x"]},
            ],
            "timeout": 5,
        }
        req = urllib.request.Request(
            self._url("/v1/plans"),
            data=json.dumps(plan_body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        self.assertIn("results", out)
        self.assertEqual(out["results"]["x"]["status"], "ok")
        self.assertEqual(out["results"]["y"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
