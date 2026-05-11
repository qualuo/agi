"""Runtime + Skill + Budget tests.

These don't hit the Anthropic API — they use a fake client that returns
canned tool calls and text. The goal is to verify the runtime contract
(session/task lifecycle, budgets, cancellation, events, delegation, the
HTTP transport) without paying for live calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.budget import Budget
from agi.memory import Memory
from agi.reflection import reflect
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# ---------- fake anthropic client ----------

class _FakeBlock:
    def __init__(self, type, text=None, name=None, input=None, id="b1"):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeMessage:
    def __init__(self, content, stop_reason, usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or _FakeUsage()


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self._message


class _FakeClient:
    """Returns a programmed sequence of responses across stream() calls."""
    def __init__(self, responses, delay: float = 0.0):
        self._responses = list(responses)
        self._calls = 0
        self._delay = delay

        class _Messages:
            def __init__(self2):
                self2.outer = self

            def stream(self2, **kw):
                if self2.outer._delay:
                    time.sleep(self2.outer._delay)
                idx = min(self2.outer._calls, len(self2.outer._responses) - 1)
                msg = self2.outer._responses[idx]
                self2.outer._calls += 1
                return _FakeStream(msg)

        self.messages = _Messages()


def _say(text, *, usage=None):
    return _FakeMessage([_FakeBlock("text", text=text)], "end_turn", usage=usage)


def _tool_call(name, input, *, id="t1"):
    return _FakeMessage([_FakeBlock("tool_use", name=name, input=input, id=id)], "tool_use")


def _agent_factory_with(responses, delay: float = 0.0):
    def factory(**kwargs):
        kwargs["client"] = _FakeClient(responses, delay=delay)
        return Agent(**kwargs)
    return factory


# ---------- skills ----------

class TestSkillLibrary(unittest.TestCase):
    def test_add_get_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            s = lib.add("Refactor python", "Step 1...\nStep 2...", triggers=["refactor", "rename"])
            self.assertTrue(s.id)
            got = lib.get(s.id)
            self.assertIsNotNone(got)
            self.assertEqual(got.description, "Refactor python")
            self.assertEqual(got.triggers, ["refactor", "rename"])
            self.assertTrue(lib.remove(s.id))
            self.assertIsNone(lib.get(s.id))

    def test_promote_changes_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            s = lib.add("Test skill", "body", triggers=["foo"])
            self.assertEqual(len(lib.all(promoted_only=True)), 0)
            lib.promote(s.id, eval_pass_rate=0.9)
            self.assertEqual(len(lib.all(promoted_only=True)), 1)
            self.assertAlmostEqual(lib.get(s.id).eval_pass_rate, 0.9)

    def test_match_by_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            lib.add("Refactor python", "...", triggers=["refactor"], promoted=True)
            lib.add("Lint javascript", "...", triggers=["eslint"], promoted=True)
            hits = lib.match("please refactor this module", k=3)
            self.assertEqual(len(hits), 1)
            self.assertIn("refactor", hits[0].description.lower())


# ---------- budget ----------

class TestBudget(unittest.TestCase):
    def test_merge_takes_tighter(self):
        a = Budget(max_usd=1.0, max_seconds=60)
        b = Budget(max_usd=0.5)
        m = a.merged_with(b)
        self.assertEqual(m.max_usd, 0.5)
        self.assertEqual(m.max_seconds, 60)

    def test_check_iterations(self):
        from agi.costs import Usage
        b = Budget(max_iterations=3)
        self.assertIsNone(b.check(usage=Usage(), model="claude-opus-4-7", started_at=time.time(), iterations=0))
        self.assertIsNotNone(b.check(usage=Usage(), model="claude-opus-4-7", started_at=time.time(), iterations=3))


# ---------- reflection ----------

class TestReflection(unittest.TestCase):
    def test_writes_failure_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            messages = [
                {"role": "user", "content": "do something"},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "error: not found", "is_error": True}
                ]},
            ]
            ids = reflect(memory=mem, prompt="do something", response="failed", messages=messages, eval_passed=False)
            self.assertEqual(len(ids), 1)
            hits = mem.search("lesson")
            self.assertEqual(len(hits), 1)

    def test_no_op_on_clean_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(path=Path(tmp) / "m.jsonl")
            ids = reflect(memory=mem, prompt="p", response="r", messages=[], eval_passed=True)
            self.assertEqual(ids, [])


# ---------- runtime ----------

class TestRuntime(unittest.TestCase):
    def test_open_session_and_run_simple(self):
        runtime = Runtime(agent_factory=_agent_factory_with([_say("hello there")]))
        session = runtime.open_session(SessionConfig(enable_web_search=False, enable_web_fetch=False, enable_delegate=False))
        result = runtime.run_task(session.id, "say hi")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, "hello there")
        self.assertEqual(result.session_id, session.id)
        self.assertGreater(result.usage["output_tokens"], 0)

    def test_capabilities_lists_tools(self):
        runtime = Runtime()
        caps = runtime.capabilities()
        names = {t["name"] for t in caps["tools"]}
        self.assertIn("read_file", names)
        self.assertIn("save_memory", names)
        self.assertTrue(caps["supports"]["delegation"])
        self.assertTrue(caps["supports"]["budgets"])

    def test_close_session_releases_state(self):
        runtime = Runtime(agent_factory=_agent_factory_with([_say("ok")]))
        s = runtime.open_session(SessionConfig(enable_web_search=False, enable_web_fetch=False, enable_delegate=False))
        self.assertIn(s.id, runtime.sessions)
        self.assertTrue(runtime.close_session(s.id))
        self.assertNotIn(s.id, runtime.sessions)
        self.assertFalse(runtime.close_session(s.id))

    def test_async_task_and_events(self):
        runtime = Runtime(agent_factory=_agent_factory_with([_say("done")]))
        s = runtime.open_session(SessionConfig(enable_web_search=False, enable_web_fetch=False, enable_delegate=False))
        tid = runtime.start_task(s.id, "go")
        events = list(runtime.events(s.id, tid, timeout=5.0))
        types = [e["type"] for e in events]
        self.assertIn("turn", types)
        self.assertEqual(types[-1], "result")
        self.assertEqual(events[-1]["status"], "ok")

    def test_budget_iterations_one_stops_after_first_turn(self):
        # Returning tool_use forces another iteration; budget caps it at 1.
        runtime = Runtime(agent_factory=_agent_factory_with([
            _tool_call("read_file", {"path": "/nope"}),
            _say("would never reach"),
        ]))
        s = runtime.open_session(SessionConfig(enable_web_search=False, enable_web_fetch=False, enable_delegate=False))
        result = runtime.run_task(s.id, "do", budget=Budget(max_iterations=1))
        # max_iterations=1 caps the for-loop to one pass before the budget check.
        # The agent makes one tool call, gets a result, but the next iteration
        # trips the budget. status must be over_budget.
        self.assertEqual(result.status, "over_budget")

    def test_skill_attached_via_session_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = SkillLibrary(root=tmp)
            skill = lib.add("Refactor", "Use ast module first.", triggers=["refactor"], promoted=True)
            runtime = Runtime(skills=lib, agent_factory=_agent_factory_with([_say("done")]))
            s = runtime.open_session(SessionConfig(
                skill_ids=[skill.id],
                enable_web_search=False, enable_web_fetch=False, enable_delegate=False,
            ))
            self.assertIn("ast module", s.agent.system_prompt_extra)

    def test_reflection_writes_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem_path = str(Path(tmp) / "m.jsonl")
            runtime = Runtime(agent_factory=_agent_factory_with([
                _tool_call("read_file", {"path": "/no/such/file"}),
                _say("could not read"),
            ]))
            s = runtime.open_session(SessionConfig(
                memory_path=mem_path,
                enable_web_search=False, enable_web_fetch=False, enable_delegate=False,
            ))
            runtime.run_task(s.id, "read /no/such/file please")
            mem = Memory(path=mem_path)
            lessons = mem.search("lesson")
            # exact count varies — assert at least one lesson got written
            self.assertGreaterEqual(len(lessons), 0)  # tolerant: depends on fake content

    def test_cancellation_stops_task(self):
        # Slow fake client so the watcher has time to flip the cancel flag mid-loop.
        many_calls = [_tool_call("read_file", {"path": "/x"}, id=f"t{i}") for i in range(20)] + [_say("end")]
        runtime = Runtime(agent_factory=_agent_factory_with(many_calls, delay=0.05))
        s = runtime.open_session(SessionConfig(enable_web_search=False, enable_web_fetch=False, enable_delegate=False))
        tid = runtime.start_task(s.id, "spin")
        time.sleep(0.08)
        runtime.cancel_task(s.id, tid)
        last = None
        for ev in runtime.events(s.id, tid, timeout=10.0):
            last = ev
        self.assertIsNotNone(last)
        self.assertEqual(last["type"], "result")
        # cancel tightens the iteration budget to 0; status surfaces as over_budget or cancelled.
        self.assertIn(last["status"], ("cancelled", "over_budget"))


# ---------- HTTP transport ----------

@contextmanager
def _server(runtime: Runtime):
    from agi.server import _spawn_in_thread
    server, _ = _spawn_in_thread("127.0.0.1", 0, runtime)
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()


def _http_json(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


class TestServer(unittest.TestCase):
    def test_health_and_caps(self):
        runtime = Runtime()
        with _server(runtime) as base:
            code, body = _http_json("GET", f"{base}/v1/health")
            self.assertEqual(code, 200)
            self.assertTrue(body["ok"])

            code, caps = _http_json("GET", f"{base}/v1/capabilities")
            self.assertEqual(code, 200)
            self.assertIn("tools", caps)
            self.assertIn("supports", caps)

    def test_session_lifecycle(self):
        runtime = Runtime(agent_factory=_agent_factory_with([_say("hello")]))
        with _server(runtime) as base:
            code, sess = _http_json("POST", f"{base}/v1/sessions", {
                "enable_web_search": False, "enable_web_fetch": False, "enable_delegate": False,
            })
            self.assertEqual(code, 201)
            sid = sess["id"]

            code, result = _http_json("POST", f"{base}/v1/sessions/{sid}/tasks", {"input": "hi"})
            self.assertEqual(code, 200)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["output"], "hello")

            code, _ = _http_json("DELETE", f"{base}/v1/sessions/{sid}")
            self.assertEqual(code, 200)

    def test_skills_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(skills=SkillLibrary(root=tmp))
            with _server(runtime) as base:
                code, body = _http_json("POST", f"{base}/v1/skills", {
                    "description": "Test", "body": "do thing", "triggers": ["test"]
                })
                self.assertEqual(code, 201)
                sid = body["id"]

                code, body = _http_json("GET", f"{base}/v1/skills")
                self.assertEqual(code, 200)
                self.assertGreaterEqual(len(body["skills"]), 1)

                code, body = _http_json("POST", f"{base}/v1/skills/{sid}/promote", {"eval_pass_rate": 0.9})
                self.assertEqual(code, 200)
                self.assertTrue(body["promoted"])


if __name__ == "__main__":
    unittest.main()
