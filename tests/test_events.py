"""EventBus tests — pure stdlib, no API."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event, EventBus, stdout_printer


class TestEventBus(unittest.TestCase):
    def test_publish_to_single_subscriber(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append)
        bus.emit("text.delta", text="hi")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].kind, "text.delta")
        self.assertEqual(received[0].data, {"text": "hi"})

    def test_fanout_to_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe(a.append)
        bus.subscribe(b.append)
        bus.emit("turn.started")
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)

    def test_unsubscribe(self):
        bus = EventBus()
        received: list[Event] = []
        unsub = bus.subscribe(received.append)
        bus.emit("a")
        unsub()
        bus.emit("b")
        self.assertEqual([e.kind for e in received], ["a"])

    def test_subscriber_exception_is_isolated(self):
        bus = EventBus()
        survived: list[Event] = []

        def bad(_):
            raise RuntimeError("boom")

        bus.subscribe(bad)
        bus.subscribe(survived.append)
        bus.emit("k")
        # The second subscriber still got the event.
        self.assertEqual(len(survived), 1)

    def test_event_to_dict_is_json_safe(self):
        import json
        e = Event(kind="x", data={"n": 1, "s": "hello"})
        d = e.to_dict()
        # Roundtrip through JSON without raising.
        json.dumps(d)
        self.assertEqual(d["kind"], "x")
        self.assertEqual(d["data"]["s"], "hello")

    def test_stdout_printer_handles_known_kinds(self):
        # Just verify these don't raise; output goes to stdout.
        bus = EventBus()
        bus.subscribe(stdout_printer)
        for kind, data in [
            ("thinking.started", {}),
            ("thinking.delta", {"text": "x"}),
            ("text.started", {}),
            ("text.delta", {"text": "y"}),
            ("tool.requested", {"name": "run_bash"}),
            ("server_tool.requested", {"name": "web_search"}),
            ("turn.finished", {"usage_formatted": "0 in / 0 out"}),
            ("error", {"message": "x"}),
            ("critic.scored", {"score": 0.1, "threshold": 0.5}),
            ("some.unknown", {}),  # printer ignores unknown kinds silently
        ]:
            bus.emit(kind, **data)


if __name__ == "__main__":
    unittest.main()
