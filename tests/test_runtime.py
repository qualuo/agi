"""Tests for agi.runtime — exercises the runtime surface without making
API calls. The Agent class is monkey-patched so submit() can complete
end-to-end without ANTHROPIC_API_KEY."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

# Set a dummy key BEFORE importing agi (anthropic SDK may inspect env at import)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-runtime-tests")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi import runtime as runtime_mod
from agi.budget import Budget
from agi.events import Event
from agi.runtime import Runtime, TaskStatus


class _FakeAgent:
    """A drop-in stand-in for agi.agent.Agent that doesn't hit the API.

    Reads instructions from the prompt: if it starts with "FAIL", raises.
    If it starts with "SLEEP n", sleeps n seconds then returns "slept".
    Otherwise returns "fake: <prompt>".
    """

    last_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        _FakeAgent.last_kwargs = kwargs
        self.event_bus = kwargs.get("event_bus")
        self.task_id = kwargs.get("task_id")
        self.parent_task_id = kwargs.get("parent_task_id")

        class _Usage:
            input_tokens = 100
            output_tokens = 50
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0

            def cost_usd(self, _model: str) -> float:
                return 0.0042

        self.usage = _Usage()

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        if self.event_bus and self.task_id:
            self.event_bus.emit("task.started", self.task_id,
                                parent_task_id=self.parent_task_id,
                                prompt=prompt, model=self.kwargs.get("model"))
        cancel = self.kwargs.get("cancel_event")
        if prompt.startswith("FAIL"):
            raise RuntimeError("intentional failure")
        if prompt.startswith("SLEEP"):
            secs = float(prompt.split()[1])
            slept = 0.0
            while slept < secs:
                if cancel is not None and cancel.is_set():
                    return "cancelled"
                time.sleep(0.05)
                slept += 0.05
            result = "slept"
        else:
            result = f"fake: {prompt}"
        if self.event_bus and self.task_id:
            self.event_bus.emit("task.completed", self.task_id,
                                parent_task_id=self.parent_task_id,
                                final_text=result, cost_usd=0.0042)
        return result


class _RuntimeTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_agent = runtime_mod.Agent
        runtime_mod.Agent = _FakeAgent
        self.tmp = tempfile.TemporaryDirectory()
        self.rt = Runtime(
            memory_path=Path(self.tmp.name) / "mem.jsonl",
            skill_root=Path(self.tmp.name) / "skills",
            synth_root=Path(self.tmp.name) / "synth",
            trace_path=Path(self.tmp.name) / "traces.jsonl",
            max_workers=2,
        )

    def tearDown(self) -> None:
        self.rt.shutdown(wait=True)
        runtime_mod.Agent = self._orig_agent
        self.tmp.cleanup()


class TestRuntime(_RuntimeTestBase):
    def test_submit_and_wait_succeeds(self) -> None:
        h = self.rt.submit("hello world")
        snap = h.wait(timeout=5)
        self.assertEqual(snap["status"], TaskStatus.SUCCEEDED.value)
        self.assertEqual(snap["result"], "fake: hello world")
        self.assertGreater(snap["cost_usd"], 0)

    def test_failure_is_captured(self) -> None:
        h = self.rt.submit("FAIL please")
        snap = h.wait(timeout=5)
        self.assertEqual(snap["status"], TaskStatus.FAILED.value)
        self.assertIn("intentional failure", snap["error"])

    def test_cancel_running_task(self) -> None:
        h = self.rt.submit("SLEEP 5")
        time.sleep(0.1)
        self.assertTrue(self.rt.cancel(h.id))
        snap = h.wait(timeout=5)
        self.assertEqual(snap["status"], TaskStatus.CANCELLED.value)

    def test_list_tasks_filters(self) -> None:
        h1 = self.rt.submit("a")
        h2 = self.rt.submit("b")
        h1.wait(timeout=5)
        h2.wait(timeout=5)
        succ = self.rt.list_tasks(status=TaskStatus.SUCCEEDED)
        self.assertEqual({t["id"] for t in succ}, {h1.id, h2.id})

    def test_events_emitted_in_order(self) -> None:
        seen: list[Event] = []
        evt = threading.Event()

        def on(ev: Event) -> None:
            seen.append(ev)
            if any(e.kind == "task.completed" for e in seen):
                evt.set()

        self.rt.subscribe(on)
        self.rt.submit("hello").wait(timeout=5)
        evt.wait(timeout=3)
        kinds = [e.kind for e in seen]
        self.assertIn("task.submitted", kinds)
        self.assertIn("task.completed", kinds)
        # task.submitted comes before task.completed for the same task
        sub_idx = kinds.index("task.submitted")
        cmp_idx = kinds.index("task.completed")
        self.assertLess(sub_idx, cmp_idx)

    def test_parallel_batch(self) -> None:
        handles = self.rt.submit_batch(["SLEEP 0.2"] * 4)
        t0 = time.time()
        results = self.rt.wait_all([h.id for h in handles], timeout=10)
        elapsed = time.time() - t0
        self.assertTrue(all(r["status"] == TaskStatus.SUCCEEDED.value for r in results))
        # 4 tasks, 2 workers, ~0.2s each → ~0.4s wall, not 0.8s.
        # Allow generous slack on CI but assert better-than-serial.
        self.assertLess(elapsed, 0.7)

    def test_manifest_is_serializable_and_has_expected_keys(self) -> None:
        m = self.rt.manifest().to_dict()
        self.assertEqual(m["runtime_version"], runtime_mod.RUNTIME_VERSION)
        self.assertIn("models", m)
        self.assertIn("tools", m)
        self.assertIn("skills", m)
        self.assertIn("roles", m)
        # Role names present
        role_names = {r["name"] for r in m["roles"]}
        self.assertIn("planner", role_names)
        self.assertIn("executor", role_names)
        # Built-in tool names present
        tool_names = {t["name"] for t in m["tools"]}
        for name in ("read_file", "write_file", "run_bash", "save_memory"):
            self.assertIn(name, tool_names)
        # Optional features advertised
        self.assertTrue(m["features"]["events"])
        self.assertTrue(m["features"]["skills"])
        self.assertTrue(m["features"]["delegation"])

    def test_role_routing_changes_default_model(self) -> None:
        h = self.rt.submit("hi", role="critic")
        h.wait(timeout=5)
        # critic role defaults to haiku in roles.py
        self.assertEqual(_FakeAgent.last_kwargs["model"], "claude-haiku-4-5")

    def test_explicit_model_override(self) -> None:
        h = self.rt.submit("hi", role="executor", model="claude-sonnet-4-6")
        h.wait(timeout=5)
        self.assertEqual(_FakeAgent.last_kwargs["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
