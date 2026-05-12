"""Unit tests for KnowledgeGraph + event-driven ingestion."""
import os
import tempfile
import unittest

from agi.events import (
    CHAT_COMPLETED,
    SKILL_LOADED,
    SUBAGENT_COMPLETED,
    TOOL_RESULT,
    Event,
    EventBus,
)
from agi.knowledge import (
    Edge,
    KnowledgeGraph,
    Node,
    attach_to_bus,
)


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        )
        self.tmp.close()
        self.path = self.tmp.name
        self.kg = KnowledgeGraph(path=self.path)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def test_upsert_node_creates_and_updates(self):
        n1 = self.kg.upsert_node("file", "/tmp/x")
        self.assertEqual(n1.id, "file:/tmp/x")
        n2 = self.kg.upsert_node("file", "/tmp/x", attrs={"size": 42})
        self.assertEqual(n1.id, n2.id)
        self.assertEqual(n2.attrs.get("size"), 42)

    def test_edges_require_existing_nodes(self):
        self.kg.upsert_node("session", "abc")
        with self.assertRaises(KeyError):
            self.kg.add_edge("session:abc", "file:/missing", rel="read")

    def test_neighborhood_walks_relations(self):
        self.kg.upsert_node("project", "foo")
        self.kg.upsert_node("file", "/a")
        self.kg.upsert_node("file", "/b")
        self.kg.add_edge("project:foo", "file:/a", rel="contains")
        self.kg.add_edge("project:foo", "file:/b", rel="contains")
        q = self.kg.neighborhood("project:foo", hops=1)
        self.assertEqual(q.seed.key, "foo")
        self.assertEqual(len(q.nodes), 3)
        self.assertEqual(len({n.id for n in q.nodes}), 3)

    def test_shortest_path(self):
        self.kg.upsert_node("a", "1")
        self.kg.upsert_node("a", "2")
        self.kg.upsert_node("a", "3")
        self.kg.add_edge("a:1", "a:2", rel="next")
        self.kg.add_edge("a:2", "a:3", rel="next")
        path = self.kg.shortest_path("a:1", "a:3")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2)

    def test_path_returns_none_when_unreachable(self):
        self.kg.upsert_node("a", "1")
        self.kg.upsert_node("a", "2")
        self.assertIsNone(self.kg.shortest_path("a:1", "a:2"))

    def test_facts_attached_to_node(self):
        self.kg.upsert_node("user", "alice")
        self.kg.add_fact("user:alice", "prefers TypeScript", source="user")
        self.kg.add_fact("user:alice", "works on project foo", source="user")
        facts = self.kg.facts("user:alice")
        self.assertEqual(len(facts), 2)
        self.assertIn("TypeScript", facts[0].text)

    def test_query_text_finds_substring(self):
        self.kg.upsert_node("file", "/etc/nginx.conf")
        self.kg.upsert_node("file", "/etc/passwd")
        hits = self.kg.query_text("nginx")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].key, "/etc/nginx.conf")

    def test_summary(self):
        self.kg.upsert_node("file", "/a")
        self.kg.upsert_node("file", "/b")
        self.kg.upsert_node("url", "http://x")
        self.kg.add_edge("file:/a", "url:http://x", rel="references")
        s = self.kg.summary()
        self.assertEqual(s["nodes"], 3)
        self.assertEqual(s["edges"], 1)
        self.assertEqual(s["by_kind"]["file"], 2)
        self.assertEqual(s["by_rel"]["references"], 1)

    def test_persistence_roundtrip(self):
        self.kg.upsert_node("file", "/a", attrs={"size": 1})
        self.kg.upsert_node("file", "/b")
        self.kg.add_edge("file:/a", "file:/b", rel="depends_on")
        self.kg.add_fact("file:/a", "this is a fact")

        reloaded = KnowledgeGraph(path=self.path)
        self.assertEqual(len(reloaded.nodes()), 2)
        self.assertIsNotNone(reloaded.get_node("file", "/a"))
        self.assertEqual(len(reloaded.edges_from("file:/a")), 1)
        self.assertEqual(len(reloaded.facts("file:/a")), 1)

    def test_context_for_formats_neighborhood(self):
        self.kg.upsert_node("project", "agi")
        self.kg.upsert_node("file", "/r.md")
        self.kg.add_edge("project:agi", "file:/r.md", rel="contains")
        self.kg.add_fact("project:agi", "open-source agent runtime")
        ctx = self.kg.context_for("project", "agi")
        self.assertIn("project:agi", ctx)
        self.assertIn("open-source", ctx)
        self.assertIn("contains", ctx)

    def test_namespace_isolation(self):
        kg_a = KnowledgeGraph(path=self.path, namespace="tenant-a")
        kg_a.upsert_node("file", "/a")
        kg_b = KnowledgeGraph(path=self.path, namespace="tenant-b")
        kg_b.upsert_node("file", "/b")
        # Reload each with its namespace.
        a2 = KnowledgeGraph(path=self.path, namespace="tenant-a")
        b2 = KnowledgeGraph(path=self.path, namespace="tenant-b")
        a_keys = {n.key for n in a2.nodes()}
        b_keys = {n.key for n in b2.nodes()}
        self.assertIn("/a", a_keys)
        self.assertNotIn("/b", a_keys)
        self.assertIn("/b", b_keys)
        self.assertNotIn("/a", b_keys)


class TestKnowledgeGraphFromEvents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        )
        self.tmp.close()
        self.path = self.tmp.name
        self.kg = KnowledgeGraph(path=self.path)
        self.bus = EventBus()
        self.sub_id = attach_to_bus(self.kg, self.bus)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def test_tool_result_creates_file_node_and_edge(self):
        self.bus.publish(Event(
            kind=TOOL_RESULT,
            session_id="sess1",
            data={"tool": "read_file", "args": {"path": "/etc/hosts"}},
        ))
        self.assertIsNotNone(self.kg.get_node("file", "/etc/hosts"))
        edges = self.kg.edges_from("session:sess1")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].rel, "read")

    def test_subagent_completed_records_spawn(self):
        self.bus.publish(Event(
            kind=SUBAGENT_COMPLETED,
            session_id="parent",
            data={"child_id": "child1", "role": "researcher"},
        ))
        self.assertIsNotNone(self.kg.get_node("session", "child1"))
        edges = self.kg.edges_from("session:parent")
        self.assertEqual(edges[0].rel, "spawned")

    def test_skill_loaded_creates_skill_edge(self):
        self.bus.publish(Event(
            kind=SKILL_LOADED,
            session_id="s1",
            data={"name": "bisect_by_test"},
        ))
        self.assertIsNotNone(self.kg.get_node("skill", "bisect_by_test"))
        edges = self.kg.edges_from("session:s1", rel="used")
        self.assertEqual(len(edges), 1)

    def test_chat_completed_records_score(self):
        self.bus.publish(Event(
            kind=CHAT_COMPLETED,
            session_id="s1",
            data={"critic_score": 0.85},
        ))
        facts = self.kg.facts("session:s1")
        self.assertTrue(any("0.85" in f.text for f in facts))

    def test_buggy_event_does_not_crash_bus(self):
        # No 'data' field; ingest must swallow.
        self.bus.publish(Event(kind=TOOL_RESULT, session_id="s1", data={}))
        # Bus stays healthy
        self.bus.publish(Event(kind=TOOL_RESULT, session_id="s1",
                               data={"tool": "read_file", "args": {"path": "/x"}}))
        self.assertIsNotNone(self.kg.get_node("file", "/x"))


if __name__ == "__main__":
    unittest.main()
