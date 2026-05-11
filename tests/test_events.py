"""EventBus tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus, collect, filter_types, make_event


class TestEventBus(unittest.TestCase):
    def test_emit_and_subscribe(self):
        bus = EventBus()
        events, _ = collect(bus)
        bus.emit("foo", k=1)
        bus.emit("bar", k=2)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "foo")
        self.assertEqual(events[1]["k"], 2)

    def test_unsubscribe(self):
        bus = EventBus()
        events, unsub = collect(bus)
        bus.emit("a")
        unsub()
        bus.emit("b")
        self.assertEqual(len(events), 1)

    def test_history_is_capped(self):
        bus = EventBus(history=3)
        for i in range(5):
            bus.emit("x", i=i)
        h = bus.history()
        self.assertEqual([e["i"] for e in h], [2, 3, 4])

    def test_no_history_when_disabled(self):
        bus = EventBus()
        bus.emit("x")
        self.assertEqual(bus.history(), [])

    def test_bad_subscriber_is_dropped(self):
        bus = EventBus()
        good_events: list = []
        bus.subscribe(good_events.append)

        def bad(_evt):
            raise RuntimeError("boom")
        bus.subscribe(bad)

        bus.emit("a")  # bad raises, gets dropped
        bus.emit("b")  # only good fires
        self.assertEqual([e["type"] for e in good_events], ["a", "b"])
        # subscribers count back down to 1
        self.assertEqual(len(bus), 1)

    def test_filter_types(self):
        evts = [make_event("a"), make_event("b"), make_event("a")]
        self.assertEqual(len(filter_types(evts, "a")), 2)
        self.assertEqual(len(filter_types(evts, "a", "b")), 3)


if __name__ == "__main__":
    unittest.main()
