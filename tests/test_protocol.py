"""Tests for the JSON-RPC coordination protocol."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.memory import Memory
from agi.protocol import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    CoordinationProtocol,
)
from agi.runtime import Runtime
from agi.skills import SkillLibrary
from tests.test_runtime import FakeAgent


def _make_runtime() -> Runtime:
    tmp = Path(tempfile.mkdtemp())
    return Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )


def _exchange(proto: CoordinationProtocol, requests: list[dict]) -> list[dict]:
    """Send a batch of requests; return parsed responses."""
    reader = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    writer = io.StringIO()
    proto.serve_streams(reader, writer)
    responses: list[dict] = []
    for line in writer.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        responses.append(json.loads(line))
    return responses


class TestProtocolBasics(unittest.TestCase):
    def test_ping(self):
        proto = CoordinationProtocol(_make_runtime())
        out = _exchange(proto, [{"jsonrpc": "2.0", "id": 1, "method": "ping"}])
        # First message is the server's "ready" banner; we want responses with id.
        replies = [m for m in out if "id" in m and m["id"] == 1]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["result"]["pong"], True)

    def test_version(self):
        proto = CoordinationProtocol(_make_runtime())
        out = _exchange(proto, [{"jsonrpc": "2.0", "id": 2, "method": "version"}])
        replies = [m for m in out if m.get("id") == 2]
        self.assertEqual(replies[0]["result"]["protocol"], PROTOCOL_VERSION)

    def test_parse_error(self):
        proto = CoordinationProtocol(_make_runtime())
        reader = io.StringIO("not json\n")
        writer = io.StringIO()
        proto.serve_streams(reader, writer)
        out = [json.loads(l) for l in writer.getvalue().splitlines() if l.strip()]
        errors = [m for m in out if "error" in m]
        self.assertGreater(len(errors), 0)
        self.assertEqual(errors[0]["error"]["code"], PARSE_ERROR)

    def test_method_not_found(self):
        proto = CoordinationProtocol(_make_runtime())
        out = _exchange(proto, [
            {"jsonrpc": "2.0", "id": 3, "method": "nope"}
        ])
        errors = [m for m in out if m.get("id") == 3 and "error" in m]
        self.assertEqual(errors[0]["error"]["code"], METHOD_NOT_FOUND)

    def test_invalid_params(self):
        proto = CoordinationProtocol(_make_runtime())
        out = _exchange(proto, [
            {"jsonrpc": "2.0", "id": 4, "method": "session.create",
             "params": {"not_a_real_field": 123}}
        ])
        errors = [m for m in out if m.get("id") == 4 and "error" in m]
        self.assertEqual(errors[0]["error"]["code"], INVALID_PARAMS)


class TestProtocolSessionLifecycle(unittest.TestCase):
    def test_create_chat_end(self):
        proto = CoordinationProtocol(_make_runtime())
        reqs = [
            {"jsonrpc": "2.0", "id": 1, "method": "session.create",
             "params": {"max_iterations": 5}},
        ]
        out = _exchange(proto, reqs)
        replies = {m.get("id"): m for m in out if "id" in m}
        sid = replies[1]["result"]["session_id"]
        # Second exchange uses fresh streams but same proto state.
        reqs2 = [
            {"jsonrpc": "2.0", "id": 2, "method": "session.chat",
             "params": {"session_id": sid, "user_input": "hi"}},
            {"jsonrpc": "2.0", "id": 3, "method": "session.get",
             "params": {"session_id": sid}},
            {"jsonrpc": "2.0", "id": 4, "method": "session.end",
             "params": {"session_id": sid}},
        ]
        out2 = _exchange(proto, reqs2)
        r2 = {m.get("id"): m for m in out2 if "id" in m}
        self.assertEqual(r2[2]["result"]["final_text"], "ok")
        self.assertEqual(r2[3]["result"]["id"], sid)
        self.assertTrue(r2[4]["result"]["ok"])


class TestProtocolEvents(unittest.TestCase):
    def test_history_returns_events(self):
        rt = _make_runtime()
        proto = CoordinationProtocol(rt)
        sid = rt.create_session()
        rt.chat(sid, "hi")
        out = _exchange(proto, [
            {"jsonrpc": "2.0", "id": 1, "method": "events.history",
             "params": {"session_id": sid}}
        ])
        result = next(m for m in out if m.get("id") == 1)["result"]
        kinds = {e["kind"] for e in result}
        self.assertIn("chat.completed", kinds)

    def test_subscribe_emits_notifications(self):
        rt = _make_runtime()
        proto = CoordinationProtocol(rt)
        reader = io.StringIO(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "events.subscribe"
        }) + "\n")
        writer = io.StringIO()
        # Run protocol in a thread so we can trigger an event after subscription.
        t = threading.Thread(target=proto.serve_streams, args=(reader, writer))
        t.start()
        t.join(timeout=2.0)
        # Trigger an event after the reader is exhausted but writer captured
        # the subscribe ack.
        sid = rt.create_session()
        rt.chat(sid, "hi")
        # Re-run with empty reader to push pending notifications. The
        # subscriber already wrote ack; new events come through if we
        # keep the writer alive. We test ack only — full notification
        # streaming is covered by integration tests.
        out = [json.loads(l) for l in writer.getvalue().splitlines() if l.strip()]
        ack = [m for m in out if m.get("id") == 1]
        self.assertEqual(len(ack), 1)
        self.assertTrue(ack[0]["result"]["ok"])


class TestProtocolNotifications(unittest.TestCase):
    def test_notification_no_id_no_response(self):
        proto = CoordinationProtocol(_make_runtime())
        out = _exchange(proto, [
            {"jsonrpc": "2.0", "method": "ping"}  # no id ⇒ notification
        ])
        # Server "ready" banner and nothing else — no response for notification.
        notif_replies = [m for m in out if m.get("method") != "ready" and "id" in m]
        self.assertEqual(notif_replies, [])


if __name__ == "__main__":
    unittest.main()
