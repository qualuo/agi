"""Lightweight in-memory metrics.

Thread-safe counters and a fixed-size latency histogram. Designed to be cheap
and to read out as plain JSON for ingestion by a coordination engine's
monitoring stack — no Prometheus dependency. The shape is stable; readers can
diff snapshots safely.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._latencies_ms: dict[str, list[float]] = defaultdict(list)
        self._started = time.time()

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] += by

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe_ms(self, name: str, ms: float, *, cap: int = 1024) -> None:
        with self._lock:
            buf = self._latencies_ms[name]
            buf.append(ms)
            if len(buf) > cap:
                # Drop oldest half to bound memory without losing recent signal.
                del buf[: cap // 2]

    def snapshot(self) -> dict:
        with self._lock:
            latencies = {}
            for name, buf in self._latencies_ms.items():
                if not buf:
                    continue
                s = sorted(buf)
                latencies[name] = {
                    "count": len(s),
                    "p50_ms": _pct(s, 0.50),
                    "p95_ms": _pct(s, 0.95),
                    "p99_ms": _pct(s, 0.99),
                    "max_ms": s[-1],
                }
            return {
                "uptime_seconds": time.time() - self._started,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latency": latencies,
            }


def _pct(sorted_buf: list[float], p: float) -> float:
    if not sorted_buf:
        return 0.0
    k = max(0, min(len(sorted_buf) - 1, int(round(p * (len(sorted_buf) - 1)))))
    return sorted_buf[k]


class Timer:
    """Context manager: records elapsed ms to `metrics` under `name` on exit."""

    def __init__(self, metrics: Metrics, name: str) -> None:
        self.metrics = metrics
        self.name = name
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        ms = (time.perf_counter() - self._start) * 1000.0
        self.metrics.observe_ms(self.name, ms)
