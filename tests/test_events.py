"""Tests for agi.events — pure-Python, no API key required."""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event, EventBus, new_task_id


class TestEventBus(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = EventBus()

    def tearDown(self) -> None:
        self.bus.close(timeout=0.5)

    def test_subscribe_receives_emitted_event(self) -> None:
        received: list[Event] = []
        evt = threading.Event()

        def on(ev: Event) -> None:
            received.append(ev)
            evt.set()

        self.bus.subscribe(on)
        self.bus.emit("task.started", "abc", foo="bar")
        self.assertTrue(evt.wait(timeout=2.0), "subscriber never called")
        self.assertEqual(received[0].kind, "task.started")
        self.assertEqual(received[0].task_id, "abc")
        self.assertEqual(received[0].data["foo"], "bar")
        self.assertGreater(received[0].seq, 0)

    def test_unsubscribe_stops_callbacks(self) -> None:
        received: list[Event] = []
        unsub = self.bus.subscribe(lambda e: received.append(e))
        self.bus.emit("task.text", "t1", text="hello")
        time.sleep(0.1)
        unsub()
        self.bus.emit("task.text", "t1", text="goodbye")
        time.sleep(0.1)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data["text"], "hello")

    def test_subscriber_exception_does_not_break_bus(self) -> None:
        seen_b: list[Event] = []

        def bad(_ev: Event) -> None:
            raise RuntimeError("kaboom")

        self.bus.subscribe(bad)
        self.bus.subscribe(lambda e: seen_b.append(e))
        self.bus.emit("task.completed", "x", final_text="ok")
        time.sleep(0.1)
        self.assertEqual(len(seen_b), 1)

    def test_event_to_dict_is_json_friendly(self) -> None:
        ev = Event(kind="task.tool_call", task_id="abc", seq=1, data={"name": "read_file"})
        d = ev.to_dict()
        self.assertEqual(d["kind"], "task.tool_call")
        self.assertEqual(d["task_id"], "abc")
        self.assertIn("ts", d)
        self.assertEqual(d["data"]["name"], "read_file")

    def test_seq_is_monotonic(self) -> None:
        seqs: list[int] = []
        evt = threading.Event()
        target = 5

        def on(ev: Event) -> None:
            seqs.append(ev.seq)
            if len(seqs) >= target:
                evt.set()

        self.bus.subscribe(on)
        for i in range(target):
            self.bus.emit("task.text", "t", text=str(i))
        evt.wait(timeout=2.0)
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), target)

    def test_new_task_id_is_unique(self) -> None:
        ids = {new_task_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
