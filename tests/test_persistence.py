"""Tests for session persistence + memory namespacing + metrics."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.persistence import SessionStore
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


def _make_runtime(tmp: Path, *, with_store: bool = True) -> Runtime:
    store = SessionStore(path=tmp / "sessions") if with_store else None
    return Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
        session_store=store,
    )


class TestPersistence(unittest.TestCase):
    def test_checkpoint_then_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rt = _make_runtime(tmp)
            sid = rt.create_session()
            rt.chat(sid, "first")
            rt.chat(sid, "second")
            ckpt = rt.checkpoint_session(sid)
            self.assertTrue(ckpt.exists())

            # New Runtime instance — simulate process restart
            rt2 = _make_runtime(tmp)
            rt2.restore_session(sid)
            session = rt2.get_session(sid)
            self.assertEqual(session.state.turn_count, 2)
            self.assertEqual(session.state.total_input_tokens, 200)

    def test_restore_without_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _make_runtime(Path(tmp), with_store=False)
            with self.assertRaises(RuntimeError):
                rt.restore_session("nope")

    def test_restore_unknown_session_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _make_runtime(Path(tmp))
            with self.assertRaises(KeyError):
                rt.restore_session("nope")

    def test_session_store_list_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(path=Path(tmp))
            rt = Runtime(
                memory=Memory(path=Path(tmp) / "m.jsonl"),
                skills=SkillLibrary(path=Path(tmp) / "skills"),
                agent_factory=FakeAgent,
                session_store=store,
            )
            sid = rt.create_session()
            rt.chat(sid, "hi")
            rt.checkpoint_session(sid)
            self.assertIn(sid, store.list_ids())
            self.assertTrue(store.delete(sid))
            self.assertNotIn(sid, store.list_ids())


class TestNamespacedMemory(unittest.TestCase):
    def test_namespaces_isolate_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            mem = Memory(path=path)
            tenant_a = mem.namespaced("tenant-a")
            tenant_b = mem.namespaced("tenant-b")

            tenant_a.save("secret-a")
            tenant_b.save("secret-b")

            self.assertEqual([n.text for n in tenant_a.all()], ["secret-a"])
            self.assertEqual([n.text for n in tenant_b.all()], ["secret-b"])
            # Default (no namespace) sees all (back-compat)
            self.assertEqual({n.text for n in mem.all()}, {"secret-a", "secret-b"})

    def test_runtime_passes_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _make_runtime(Path(tmp))
            sid_a = rt.create_session(namespace="tenant-a")
            sid_b = rt.create_session(namespace="tenant-b")
            # Sessions get distinct namespaced memory
            mem_a = rt.get_session(sid_a)._memory
            mem_b = rt.get_session(sid_b)._memory
            self.assertEqual(mem_a.namespace, "tenant-a")
            self.assertEqual(mem_b.namespace, "tenant-b")


class TestMetrics(unittest.TestCase):
    def test_metrics_increment_with_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _make_runtime(Path(tmp))
            self.assertEqual(rt.metrics()["sessions_created"], 0)
            sid = rt.create_session()
            self.assertEqual(rt.metrics()["sessions_created"], 1)
            rt.chat(sid, "hi")
            metrics = rt.metrics()
            self.assertEqual(metrics["chats_completed"], 1)
            self.assertEqual(metrics["active_sessions"], 1)
            self.assertGreater(metrics["total_cost_usd"], 0)
            rt.end_session(sid)
            metrics = rt.metrics()
            self.assertEqual(metrics["sessions_ended"], 1)
            self.assertEqual(metrics["active_sessions"], 0)


class TestConcurrencyCap(unittest.TestCase):
    def test_max_concurrent_sessions_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(
                memory=Memory(path=Path(tmp) / "m.jsonl"),
                skills=SkillLibrary(path=Path(tmp) / "skills"),
                agent_factory=FakeAgent,
                max_concurrent_sessions=2,
            )
            rt.create_session()
            rt.create_session()
            with self.assertRaises(RuntimeError):
                rt.create_session()


if __name__ == "__main__":
    unittest.main()
