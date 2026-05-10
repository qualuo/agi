"""Tests for the runtime (Runtime, Session) — no API calls.

We don't exercise step() here because it hits Anthropic. We do verify
that capabilities, snapshot, list_sessions, end, inject, role config,
and the registry-restriction logic work correctly.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.runtime import DEFAULT_ROLES, Role, Runtime, Session
from agi.skills import SkillLibrary


class FakeClient:
    """Stand-in so Runtime can be built without an API key."""
    pass


def _runtime() -> Runtime:
    tmp = tempfile.mkdtemp()
    mem = Memory(path=Path(tmp) / "m.jsonl")
    skills = SkillLibrary(path=Path(tmp) / "skills")
    return Runtime(memory=mem, skills=skills, tracer=None, client=FakeClient())


class TestRuntime(unittest.TestCase):
    def test_create_general_session(self):
        rt = _runtime()
        sess = rt.create_session(role="general", goal="say hi")
        self.assertEqual(sess.meta.role, "general")
        self.assertTrue(sess.is_active)
        # Default tools present.
        self.assertIn("read_file", sess.agent.registry.handlers)
        self.assertIn("save_memory", sess.agent.registry.handlers)
        # Delegate is wired for the general role.
        self.assertIn("delegate", sess.agent.registry.handlers)
        # Synth is enabled.
        self.assertIn("make_tool", sess.agent.registry.handlers)

    def test_planner_role_restricts_tools(self):
        rt = _runtime()
        sess = rt.create_session(role="planner", goal="plan a thing")
        names = set(sess.agent.registry.handlers)
        # Planner allows only read-only tools.
        self.assertIn("search_memory", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("run_bash", names)
        self.assertNotIn("make_tool", names)

    def test_critic_role_read_only(self):
        rt = _runtime()
        sess = rt.create_session(role="critic")
        names = set(sess.agent.registry.handlers)
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("run_bash", names)

    def test_unknown_role_raises(self):
        rt = _runtime()
        with self.assertRaises(ValueError):
            rt.create_session(role="not-a-role")

    def test_list_sessions(self):
        rt = _runtime()
        a = rt.create_session(role="general")
        b = rt.create_session(role="planner")
        live = rt.list_sessions()
        ids = {s["id"] for s in live}
        self.assertEqual(ids, {a.id, b.id})

    def test_end_session(self):
        rt = _runtime()
        sess = rt.create_session(role="general")
        snap = rt.end_session(sess.id, reason="user_done")
        self.assertEqual(snap["end_reason"], "user_done")
        self.assertFalse(sess.is_active)
        # Stepping after end raises.
        with self.assertRaises(RuntimeError):
            sess.step("hi")

    def test_capabilities_shape(self):
        rt = _runtime()
        cap = rt.capabilities()
        self.assertEqual(cap["version"], 1)
        self.assertIn("tools", cap)
        self.assertIn("roles", cap)
        role_names = {r["name"] for r in cap["roles"]}
        self.assertEqual(role_names, set(DEFAULT_ROLES))
        tool_names = {t["name"] for t in cap["tools"]}
        self.assertIn("read_file", tool_names)
        self.assertIn("web_search", tool_names)

    def test_inject_observation(self):
        rt = _runtime()
        sess = rt.create_session(role="general")
        sess.inject_observation("the deploy completed")
        self.assertEqual(sess.agent.messages[-1]["content"], "the deploy completed")

    def test_snapshot_includes_metadata(self):
        rt = _runtime()
        sess = rt.create_session(role="executor", goal="do the thing")
        snap = sess.snapshot()
        self.assertEqual(snap["session_id"], sess.id)
        self.assertEqual(snap["role"], "executor")
        self.assertEqual(snap["goal"], "do the thing")
        self.assertTrue(snap["active"])
        self.assertEqual(snap["history_steps"], 0)

    def test_stats_aggregates_zero_state(self):
        rt = _runtime()
        rt.create_session(role="general")
        s = rt.stats()
        self.assertEqual(s["sessions"], 1)
        self.assertEqual(s["active_sessions"], 1)
        self.assertEqual(s["total_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
