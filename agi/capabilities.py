"""Capability registry — observed-performance routing for coordinators.

A coordination engine that picks the right SessionConfig for each task
needs a model of *what tends to work*. The registry watches every
completed task and records:

  - prompt features (keyword tokens)
  - which role + model handled it
  - which skills were loaded
  - whether the result succeeded (and at what cost / latency)

It then offers `recommend(prompt)` so a planner can pick the (role,
model, skill hints) most likely to succeed on a new prompt, biased by
budget. Empty registry → falls back to defaults; the longer it runs the
better the routing.

This is the medium-timescale learning channel at the *runtime* level:
sessions, skills, and tools learn from individual traces; the
registry learns which combinations of them ship answers.

Storage is JSONL append-only on disk. Reload reconstructs the in-memory
stats; recommendation is O(records × terms) which is fine for the
working-set sizes coordinators typically have (≤10⁵ records).
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agi.runtime import SessionConfig


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str, *, min_len: int = 3) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= min_len}


@dataclass
class CapabilityRecord:
    """One observed task completion. Append-only."""
    id: str
    ts: float
    prompt_tokens: list[str]
    role: str
    model: str
    skills_used: list[str]
    success: bool
    cost_usd: float
    duration_seconds: float
    critic_score: float | None = None
    tag: str | None = None


@dataclass
class CapabilityRecommendation:
    """A coordinator-facing suggestion derived from past observations.

    `confidence` is in [0, 1]: 0 = no relevant evidence, 1 = many
    similar successful traces. A coordinator should fall back to a
    safe default below some threshold (e.g. 0.2).
    """
    role: str
    model: str
    skills_hint: list[str]
    expected_cost_usd: float
    expected_success_rate: float
    confidence: float
    evidence_count: int
    rationale: str

    def to_session_config(self, base: SessionConfig | None = None) -> SessionConfig:
        """Materialize this recommendation as a SessionConfig.

        If a base config is given, only role/model/cost-ceiling are
        overridden. Otherwise a sensible default is constructed.
        """
        base = base or SessionConfig()
        return SessionConfig(
            **{
                **base.__dict__,
                "model": self.model,
                "role": self.role,
                "system_prompt_extra": (
                    base.system_prompt_extra
                    if base.system_prompt_extra
                    else f"Role: {self.role}. Return a concise final answer."
                ),
            }
        )


class CapabilityRegistry:
    """Append-only observed-performance store with similarity-weighted routing."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        default_role: str = "executor",
        default_model: str = "claude-opus-4-7",
    ) -> None:
        self.path = (
            Path(path) if path else Path.home() / ".agi" / "capabilities.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.default_role = default_role
        self.default_model = default_model
        self._lock = threading.Lock()
        self._records: list[CapabilityRecord] = []
        self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._records.append(CapabilityRecord(**d))
                except Exception:
                    continue

    def record(
        self,
        *,
        prompt: str,
        role: str,
        model: str,
        skills_used: Iterable[str] | None = None,
        success: bool,
        cost_usd: float,
        duration_seconds: float,
        critic_score: float | None = None,
        tag: str | None = None,
    ) -> CapabilityRecord:
        rec = CapabilityRecord(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            prompt_tokens=sorted(_tokenize(prompt)),
            role=role,
            model=model,
            skills_used=list(skills_used or []),
            success=bool(success),
            cost_usd=float(cost_usd),
            duration_seconds=float(duration_seconds),
            critic_score=critic_score,
            tag=tag,
        )
        with self._lock:
            self._records.append(rec)
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(rec), default=str) + "\n")
        return rec

    def all(self) -> list[CapabilityRecord]:
        with self._lock:
            return list(self._records)

    def _similarity(self, query_tokens: set[str], rec_tokens: list[str]) -> float:
        if not query_tokens or not rec_tokens:
            return 0.0
        rec_set = set(rec_tokens)
        inter = len(query_tokens & rec_set)
        if inter == 0:
            return 0.0
        # Jaccard, plus a small recency-independent floor so that any
        # overlap registers; recency is applied separately below.
        return inter / len(query_tokens | rec_set)

    def recommend(
        self,
        prompt: str,
        *,
        budget_usd: float | None = None,
        min_similarity: float = 0.1,
        recency_half_life_s: float = 60 * 60 * 24 * 7,  # 1 week
    ) -> CapabilityRecommendation:
        """Pick the (role, model) bucket whose similar past traces look
        best on (success_rate × (budget − cost)). Ties broken by
        evidence count, then recency.

        With no relevant evidence, falls back to (default_role,
        default_model) at confidence 0.
        """
        query = _tokenize(prompt)
        now = time.time()
        # Bucket records by (role, model)
        buckets: dict[tuple[str, str], list[tuple[CapabilityRecord, float]]] = {}
        for rec in self.all():
            sim = self._similarity(query, rec.prompt_tokens)
            if sim < min_similarity:
                continue
            age = max(0.0, now - rec.ts)
            decay = math.exp(-age / recency_half_life_s)
            weight = sim * decay
            buckets.setdefault((rec.role, rec.model), []).append((rec, weight))

        if not buckets:
            return CapabilityRecommendation(
                role=self.default_role,
                model=self.default_model,
                skills_hint=[],
                expected_cost_usd=0.0,
                expected_success_rate=0.0,
                confidence=0.0,
                evidence_count=0,
                rationale="no relevant prior observations",
            )

        scored: list[tuple[float, tuple[str, str], list[tuple[CapabilityRecord, float]]]] = []
        for key, entries in buckets.items():
            total_w = sum(w for _, w in entries) or 1e-9
            wsuccess = sum(w for r, w in entries if r.success) / total_w
            wcost = sum(r.cost_usd * w for r, w in entries) / total_w
            evidence = len(entries)
            # Penalize buckets that exceed budget
            budget_ok = 1.0
            if budget_usd is not None and wcost > 0:
                budget_ok = min(1.0, budget_usd / max(wcost, 1e-6))
            score = wsuccess * budget_ok * math.tanh(evidence / 3.0)
            scored.append((score, key, entries))

        scored.sort(key=lambda t: (-t[0], -len(t[2])))
        best_score, (role, model), entries = scored[0]
        total_w = sum(w for _, w in entries) or 1e-9
        wsuccess = sum(w for r, w in entries if r.success) / total_w
        wcost = sum(r.cost_usd * w for r, w in entries) / total_w
        # Skill hint = skills present in ≥50% of successful records
        succ_records = [r for r, _ in entries if r.success]
        skill_counts: dict[str, int] = {}
        for r in succ_records:
            for s in r.skills_used:
                skill_counts[s] = skill_counts.get(s, 0) + 1
        skills_hint = [
            s for s, c in skill_counts.items() if c >= max(1, len(succ_records) // 2)
        ]
        confidence = min(1.0, math.tanh(len(entries) / 5.0))
        return CapabilityRecommendation(
            role=role,
            model=model,
            skills_hint=sorted(skills_hint),
            expected_cost_usd=wcost,
            expected_success_rate=wsuccess,
            confidence=confidence,
            evidence_count=len(entries),
            rationale=(
                f"chose (role={role}, model={model}) from {len(entries)} similar "
                f"trace(s): {wsuccess:.0%} success at ~${wcost:.4f}"
            ),
        )

    def stats(self) -> dict[str, Any]:
        recs = self.all()
        if not recs:
            return {
                "records": 0,
                "success_rate": 0.0,
                "total_cost_usd": 0.0,
                "by_role": {},
            }
        succ = sum(1 for r in recs if r.success)
        total_cost = sum(r.cost_usd for r in recs)
        by_role: dict[str, dict[str, Any]] = {}
        for r in recs:
            b = by_role.setdefault(r.role, {"n": 0, "successes": 0, "cost_usd": 0.0})
            b["n"] += 1
            b["cost_usd"] += r.cost_usd
            if r.success:
                b["successes"] += 1
        for b in by_role.values():
            b["success_rate"] = b["successes"] / max(1, b["n"])
        return {
            "records": len(recs),
            "success_rate": succ / len(recs),
            "total_cost_usd": total_cost,
            "by_role": by_role,
        }

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self.path.write_text("")
