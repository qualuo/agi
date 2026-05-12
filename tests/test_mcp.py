"""Tests for the MCP (Model Context Protocol) adapter."""
import io
import json
import unittest

from agi.mcp import MCP_VERSION, SERVER_NAME, McpServer
from agi.runtime import Runtime, SessionConfig
from agi.skills import Skill


class _FakeAgent:
    def __init__(self, **kwargs):
        from agi.costs import Usage
        self.usage = Usage()
        self.usage.input_tokens = 50
        self.usage.output_tokens = 30
        self.messages: list = []
        self.last_critic_score = None

    def chat(self, prompt, max_iterations=25):
        return f"reply: {prompt[:80]}"

    def reset(self):
        self.messages = []


class TestMcpServer(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(agent_factory=_FakeAgent)
        self.out = io.StringIO()
        self.server = McpServer(self.runtime, stdout=self.out)

    def _last_response(self) -> dict:
        lines = [ln for ln in self.out.getvalue().splitlines() if ln.strip()]
        return json.loads(lines[-1])

    def test_initialize_returns_protocol_version(self):
        self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        resp = self._last_response()
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["result"]["protocolVersion"], MCP_VERSION)
        self.assertEqual(resp["result"]["serverInfo"]["name"], SERVER_NAME)

    def test_tools_list_includes_core_tools(self):
        self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        resp = self._last_response()
        names = {t["name"] for t in resp["result"]["tools"]}
        for required in (
            "agi.create_session",
            "agi.chat",
            "agi.end_session",
            "agi.capabilities",
            "agi.metrics",
            "agi.save_skill",
            "agi.recall",
        ):
            self.assertIn(required, names)

    def test_create_session_then_chat(self):
        self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "agi.create_session", "arguments": {}},
        })
        resp = self._last_response()
        payload = json.loads(resp["result"]["content"][0]["text"])
        sid = payload["session_id"]
        self.assertTrue(sid)

        self.server.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "agi.chat",
                "arguments": {"session_id": sid, "input": "hi"},
            },
        })
        resp = self._last_response()
        text = resp["result"]["content"][0]["text"]
        self.assertTrue(text.startswith("reply:"))

    def test_capabilities_tool(self):
        self.server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "agi.capabilities", "arguments": {}},
        })
        resp = self._last_response()
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("models", payload)
        self.assertIn("active_sessions", payload)

    def test_save_skill_then_list(self):
        self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "agi.save_skill",
                "arguments": {
                    "name": "test_skill",
                    "description": "what",
                    "body": "1. do it",
                    "tags": ["t1"],
                },
            },
        })
        self.server.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "agi.list_skills", "arguments": {}},
        })
        resp = self._last_response()
        skills = json.loads(resp["result"]["content"][0]["text"])
        self.assertTrue(any(s["name"] == "test_skill" for s in skills))

    def test_recall_tool_with_no_kg_still_works(self):
        # Save a memory note first
        self.runtime.memory.save("the moon is made of cheese", tags=["fact"])
        self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "agi.recall", "arguments": {"query": "moon"}},
        })
        resp = self._last_response()
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("notes", payload)
        self.assertEqual(payload["graph"], [])

    def test_recall_with_kg_returns_graph_hits(self):
        from agi.knowledge import KnowledgeGraph
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            kg_path = f.name
        try:
            kg = KnowledgeGraph(path=kg_path)
            kg.upsert_node("file", "/etc/special-thing")
            server = McpServer(self.runtime, knowledge=kg, stdout=self.out)
            server.handle({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "agi.recall",
                           "arguments": {"query": "special-thing"}},
            })
            resp = self._last_response()
            payload = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual(len(payload["graph"]), 1)
            self.assertEqual(payload["graph"][0]["kind"], "file")
        finally:
            os.unlink(kg_path)

    def test_run_goal_returns_not_configured_without_coordinator(self):
        self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "agi.run_goal",
                "arguments": {"intent": "do something"},
            },
        })
        resp = self._last_response()
        self.assertTrue(resp["result"].get("isError"))
        self.assertIn("not_configured", resp["result"]["content"][0]["text"])

    def test_run_goal_with_coordinator(self):
        from agi.coordinator import Coordinator, Plan, PlanStep
        def planner(goal):
            return Plan(steps=[PlanStep(id="r", prompt=goal.intent, role="executor")])
        coord = Coordinator(self.runtime, decomposer=planner)
        server = McpServer(self.runtime, coordinator=coord, stdout=self.out)
        server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "agi.run_goal",
                "arguments": {
                    "intent": "do it",
                    "expect_substring": "reply",
                },
            },
        })
        resp = self._last_response()
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertTrue(payload["success"])
        self.assertIn("reply", payload["final_text"])

    def test_resources_list_includes_active_sessions(self):
        self.runtime.create_session(SessionConfig())
        self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
        resp = self._last_response()
        uris = [r["uri"] for r in resp["result"]["resources"]]
        self.assertTrue(any(u.startswith("agi://sessions/") for u in uris))
        self.assertTrue(any(u.startswith("agi://events/") for u in uris))

    def test_resources_read_session(self):
        sid = self.runtime.create_session(SessionConfig())
        self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "resources/read",
            "params": {"uri": f"agi://sessions/{sid}"},
        })
        resp = self._last_response()
        body = json.loads(resp["result"]["contents"][0]["text"])
        self.assertEqual(body["id"], sid)

    def test_unknown_method_returns_error(self):
        self.server.handle({
            "jsonrpc": "2.0", "id": 99, "method": "does/not/exist",
        })
        resp = self._last_response()
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)

    def test_unknown_tool_returns_error_payload(self):
        self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "agi.no_such_tool", "arguments": {}},
        })
        resp = self._last_response()
        self.assertTrue(resp["result"].get("isError"))

    def test_serve_forever_via_stdin_stream(self):
        request = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        }) + "\n"
        request += json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        }) + "\n"
        out = io.StringIO()
        server = McpServer(
            self.runtime, stdin=io.StringIO(request), stdout=out,
        )
        server.serve_forever()
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["id"], 1)
        self.assertEqual(json.loads(lines[1])["id"], 2)


if __name__ == "__main__":
    unittest.main()
