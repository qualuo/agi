"""SelfEval — agent-generated regression eval bank.

A frozen LLM with a growing skill library and registry of synthesized
tools has a real problem: how do you know a new skill *helps*? How do
you catch a regression when a new synthesized tool subtly breaks a
class of prompt the agent used to handle? The base model didn't change;
the *surrounding system* changed.

`SelfEval` is the answer. Every time a session ends successfully (high
critic score, acceptance criterion met, or explicit user thumbs-up),
the system can distill the (prompt, expected_substring | predicate)
pair into an `EvalItem` and add it to a growing bank. When a candidate
skill or synthesized tool is being considered for promotion, the
runtime *re-runs the bank* and refuses the promotion if the success
rate drops.

This closes a loop investors actually understand: "the more you use
this thing, the harder it is to break, because it grows its own
regression suite."

Storage is JSONL on disk (append-only, easy to inspect). The bank is
sharded into automatic items (mined from runs) and explicit items
(authored by humans or higher-level coordinators). A coordination
engine can mark explicit items as the "blessed" suite and use the
automatic items as soft signal.

The eval runner is intentionally pluggable: a real run hits the
runtime; tests pass a fake runner so we don't need an API key.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agi.runtime import Runtime, SessionConfig


@dataclass
class EvalItem:
    """One regression eval. Either substring-based or callable-based.

    For storage, callable predicates are converted to a substring spec
    (`expect_substring`). Callables are kept in memory only for the
    in-process bank and re-derived from the spec on reload.
    """
    id: str
    prompt: str
    expect_substring: str | None = None
    expect_regex: str | None = None
    expect_min_length: int | None = None
    source: str = "automatic"  # "automatic" or "explicit"
    tags: list[str] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    last_pass_ts: float | None = None
    last_fail_ts: float | None = None
    runs: int = 0
    passes: int = 0

    def predicate(self) -> Callable[[str], bool]:
        substr = self.expect_substring
        rx = re.compile(self.expect_regex) if self.expect_regex else None
        min_len = self.expect_min_length

        def _p(text: str) -> bool:
            if not isinstance(text, str):
                return False
            if substr is not None and substr.lower() not in text.lower():
                return False
            if rx is not None and not rx.search(text):
                return False
            if min_len is not None and len(text.strip()) < min_len:
                return False
            return True

        return _p

    def pass_rate(self) -> float:
        if self.runs == 0:
            return 0.0
        return self.passes / self.runs


@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    items: list[dict[str, Any]]
    pass_rate: float
    duration_seconds: float
    total_cost_usd: float

    @property
    def success(self) -> bool:
        return self.failed == 0


EvalRunner = Callable[[EvalItem], tuple[bool, str, float]]
# (passed, final_text, cost_usd)


class SelfEvalBank:
    """Append-only regression eval store with deduplication.

    Items are deduplicated by a hash of (prompt + expect_*) so the
    same successful trace doesn't get added twice. The bank persists
    to JSONL on disk and reloads on construction.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "selfeval.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, EvalItem] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    item = EvalItem(**d)
                    self._items[item.id] = item
                except Exception:
                    continue

    @staticmethod
    def _hash(prompt: str, substr: str | None, regex: str | None,
              min_len: int | None) -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update((substr or "").encode("utf-8"))
        h.update((regex or "").encode("utf-8"))
        h.update(str(min_len or "").encode("utf-8"))
        return h.hexdigest()[:12]

    def add(
        self,
        *,
        prompt: str,
        expect_substring: str | None = None,
        expect_regex: str | None = None,
        expect_min_length: int | None = None,
        source: str = "automatic",
        tags: list[str] | None = None,
    ) -> EvalItem | None:
        """Add or dedupe-skip an item. Returns the item iff newly added.

        Soft validation: if no predicate is given, the item is rejected
        — an eval with nothing to check would always pass.
        """
        if expect_substring is None and expect_regex is None and expect_min_length is None:
            return None
        item_id = self._hash(prompt, expect_substring, expect_regex, expect_min_length)
        with self._lock:
            if item_id in self._items:
                return None
            item = EvalItem(
                id=item_id,
                prompt=prompt,
                expect_substring=expect_substring,
                expect_regex=expect_regex,
                expect_min_length=expect_min_length,
                source=source,
                tags=list(tags or []),
            )
            self._items[item_id] = item
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(item)) + "\n")
        return item

    def auto_mine(
        self,
        *,
        prompt: str,
        final_text: str,
        critic_score: float | None,
        critic_threshold: float = 0.7,
        tags: list[str] | None = None,
    ) -> EvalItem | None:
        """Distill a regression item from a single successful trace.

        Heuristic: take the first non-empty line of `final_text` (≤80
        chars) as the expected substring. If the critic isn't
        confident, no item is added. This bias is deliberate — false
        positives in the bank are worse than missing some items.
        """
        if critic_score is not None and critic_score < critic_threshold:
            return None
        if not final_text:
            return None
        first_line = next((ln.strip() for ln in final_text.splitlines() if ln.strip()), "")
        if not first_line:
            return None
        # Use a 40-character window — long enough to be distinctive,
        # short enough to survive minor formatting drift.
        sample = first_line[:40]
        return self.add(
            prompt=prompt,
            expect_substring=sample,
            expect_min_length=max(1, min(20, len(final_text) // 4)),
            source="automatic",
            tags=tags,
        )

    def all(self) -> list[EvalItem]:
        with self._lock:
            return list(self._items.values())

    def by_source(self, source: str) -> list[EvalItem]:
        return [i for i in self.all() if i.source == source]

    def remove(self, item_id: str) -> bool:
        with self._lock:
            if item_id not in self._items:
                return False
            del self._items[item_id]
            # Rewrite the file (selfeval banks are O(10²-10³) items)
            with self.path.open("w") as f:
                for it in self._items.values():
                    f.write(json.dumps(asdict(it)) + "\n")
            return True

    def _persist_after_run(self) -> None:
        # Rewrite to reflect run counters. Selfeval is small enough
        # for this to be cheap.
        with self.path.open("w") as f:
            for it in self._items.values():
                f.write(json.dumps(asdict(it)) + "\n")

    def run(self, runner: EvalRunner, *, source: str | None = None,
            stop_on_failure: bool = False) -> EvalReport:
        """Run every item through `runner` and return a report.

        `runner(item) -> (passed, final_text, cost_usd)`. The bank's
        per-item counters are updated and persisted.
        """
        start = time.time()
        items = self.all() if source is None else self.by_source(source)
        results = []
        passed = 0
        total_cost = 0.0
        for it in items:
            try:
                ok, text, cost = runner(it)
            except Exception as e:
                ok, text, cost = False, f"{type(e).__name__}: {e}", 0.0
            it.runs += 1
            if ok:
                it.passes += 1
                it.last_pass_ts = time.time()
                passed += 1
            else:
                it.last_fail_ts = time.time()
            total_cost += cost
            results.append({
                "id": it.id,
                "prompt": it.prompt,
                "passed": ok,
                "final_text": text[:200] if isinstance(text, str) else "",
                "cost_usd": cost,
            })
            if not ok and stop_on_failure:
                break
        with self._lock:
            self._persist_after_run()
        return EvalReport(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            items=results,
            pass_rate=passed / max(1, len(results)),
            duration_seconds=time.time() - start,
            total_cost_usd=total_cost,
        )

    def runtime_runner(
        self,
        runtime: Runtime,
        *,
        config: SessionConfig | None = None,
    ) -> EvalRunner:
        """Build an `EvalRunner` that drives a real Runtime.

        Each item gets its own short-lived session so eval items can't
        cross-contaminate each other's context.
        """
        cfg = config or SessionConfig(use_skills=False)

        def _run(item: EvalItem) -> tuple[bool, str, float]:
            sid = runtime.create_session(cfg)
            try:
                text = runtime.chat(sid, item.prompt)
                cost = runtime.get_session(sid).state.total_cost_usd
            finally:
                try:
                    runtime.end_session(sid)
                except KeyError:
                    cost = 0.0
                    text = ""
            return item.predicate()(text), text, cost

        return _run

    def gate_promotion(
        self,
        runner: EvalRunner,
        *,
        baseline_pass_rate: float,
        allowed_regression: float = 0.0,
    ) -> tuple[bool, EvalReport]:
        """Run the bank; return (gate_ok, report).

        `gate_ok` is True iff the new pass rate is at least
        `baseline_pass_rate - allowed_regression`. Use this around
        skill promotions / tool synthesis to refuse changes that
        regress the regression suite.
        """
        report = self.run(runner)
        threshold = baseline_pass_rate - allowed_regression
        return report.pass_rate >= threshold, report

    def stats(self) -> dict[str, Any]:
        items = self.all()
        by_source: dict[str, int] = {}
        passing = 0
        total_runs = 0
        for it in items:
            by_source[it.source] = by_source.get(it.source, 0) + 1
            total_runs += it.runs
            if it.runs > 0 and it.passes == it.runs:
                passing += 1
        return {
            "items": len(items),
            "by_source": by_source,
            "total_runs": total_runs,
            "always_passing": passing,
            "path": str(self.path),
        }
