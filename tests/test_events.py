"""Tests for the event bus."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import (
    CHAT_COMPLETED,
    CHAT_STARTED,
    SESSION_CREATED,
    Event,
    EventBus,
)


class TestEventBus(unittest.TestCase):
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append)
        bus.publish(Event(kind=SESSION_CREATED, session_id="s1"))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].kind, SESSION_CREATED)

    def test_session_filter(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append, session_id="s1")
        bus.publish(Event(kind=CHAT_STARTED, session_id="s1"))
        bus.publish(Event(kind=CHAT_STARTED, session_id="s2"))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].session_id, "s1")

    def test_kind_filter(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append, kind=CHAT_COMPLETED)
        bus.publish(Event(kind=CHAT_STARTED, session_id="s1"))
        bus.publish(Event(kind=CHAT_COMPLETED, session_id="s1"))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].kind, CHAT_COMPLETED)

    def test_unsubscribe(self):
        bus = EventBus()
        received: list[Event] = []
        sub = bus.subscribe(received.append)
        bus.publish(Event(kind="a"))
        self.assertTrue(bus.unsubscribe(sub))
        bus.publish(Event(kind="b"))
        self.assertEqual(len(received), 1)
        # unsubscribing again returns False
        self.assertFalse(bus.unsubscribe(sub))

    def test_broken_subscriber_does_not_break_bus(self):
        bus = EventBus()
        good: list[Event] = []

        def bad(_event):
            raise RuntimeError("boom")

        bus.subscribe(bad)
        bus.subscribe(good.append)
        bus.publish(Event(kind="x"))
        self.assertEqual(len(good), 1)

    def test_history_records_and_filters(self):
        bus = EventBus(history_limit=10)
        for i in range(3):
            bus.publish(Event(kind="x", session_id=f"s{i}"))
        bus.publish(Event(kind="y", session_id="s1"))
        self.assertEqual(len(bus.history()), 4)
        self.assertEqual(len(bus.history(session_id="s1")), 2)
        self.assertEqual(len(bus.history(kind="y")), 1)

    def test_history_bounded(self):
        bus = EventBus(history_limit=5)
        for i in range(20):
            bus.publish(Event(kind="x", data={"i": i}))
        h = bus.history()
        self.assertEqual(len(h), 5)
        # latest 5
        self.assertEqual(h[-1].data["i"], 19)

    def test_event_to_dict_round_trip(self):
        e = Event(kind="x", session_id="s", data={"a": 1})
        d = e.to_dict()
        self.assertEqual(d["kind"], "x")
        self.assertEqual(d["session_id"], "s")
        self.assertEqual(d["data"]["a"], 1)


if __name__ == "__main__":
    unittest.main()
