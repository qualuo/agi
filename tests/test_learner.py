"""Tests for the learner package — only the GPU-free pieces.

Trace logger and filter are testable here. Training (`train.py`) and the
local agent require CUDA/MPS and large model downloads, so they're not
covered by these tests; run them manually on hardware.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.filter import eval_passing, filter_traces, min_quality, user_thumbs_up
from learner.traces import Trace, TraceLogger


class TestTraceLogger(unittest.TestCase):
    def test_log_and_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(path=Path(tmp) / "t.jsonl")
            logger.log(
                model="claude-opus-4-7",
                messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                final_text="hello",
                usage={"input_tokens": 5, "output_tokens": 1},
                metadata={"task_id": "math-1", "eval_passed": True},
            )
            traces = logger.all()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].model, "claude-opus-4-7")
            self.assertEqual(traces[0].final_text, "hello")
            self.assertEqual(traces[0].metadata["eval_passed"], True)
            self.assertEqual(traces[0].usage["input_tokens"], 5)

    def test_log_appends_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            TraceLogger(path=path).log(model="a", messages=[], final_text="1")
            TraceLogger(path=path).log(model="a", messages=[], final_text="2")
            self.assertEqual(len(TraceLogger(path=path).all()), 2)

    def test_log_serializes_complex_blocks(self):
        # Pydantic-like object simulating SDK content blocks
        class FakeBlock:
            def __init__(self, **fields):
                self._fields = fields
            def model_dump(self, exclude_none=False):
                return {k: v for k, v in self._fields.items() if not exclude_none or v is not None}

        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(path=Path(tmp) / "t.jsonl")
            logger.log(
                model="x",
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [
                        FakeBlock(type="text", text="hello"),
                        FakeBlock(type="tool_use", id="tu_1", name="run_bash", input={"command": "ls"}),
                    ]},
                ],
                final_text="hello",
            )
            # File should contain valid JSON we can parse back
            raw = (Path(tmp) / "t.jsonl").read_text()
            parsed = json.loads(raw.strip())
            self.assertEqual(parsed["messages"][1]["content"][0]["text"], "hello")
            self.assertEqual(parsed["messages"][1]["content"][1]["name"], "run_bash")


class TestFilter(unittest.TestCase):
    def _trace(self, **metadata) -> Trace:
        return Trace(id="t1", ts=0.0, model="m", messages=[], final_text="", metadata=metadata)

    def test_eval_passing_keeps_only_passed(self):
        traces = [
            self._trace(eval_passed=True),
            self._trace(eval_passed=False),
            self._trace(),  # no metadata
        ]
        kept = filter_traces(traces, eval_passing)
        self.assertEqual(len(kept), 1)

    def test_min_quality_threshold(self):
        traces = [
            self._trace(quality_score=0.9),
            self._trace(quality_score=0.5),
            self._trace(quality_score=0.7),
            self._trace(),
        ]
        kept = filter_traces(traces, min_quality(0.7))
        self.assertEqual(len(kept), 2)

    def test_min_quality_handles_non_numeric(self):
        traces = [self._trace(quality_score="high")]
        kept = filter_traces(traces, min_quality(0.5))
        self.assertEqual(len(kept), 0)

    def test_user_thumbs_up(self):
        traces = [
            self._trace(user_rating="up"),
            self._trace(user_rating="down"),
            self._trace(),
        ]
        kept = filter_traces(traces, user_thumbs_up)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
