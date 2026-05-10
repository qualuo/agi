"""Skill mining — turn successful traces into reusable skills.

The medium-timescale learning channel from ARCHITECTURE.md §3 needs a way
to actually *produce* skills. Two strategies:

  1. LLM-driven compilation: an LLM reads N successful traces for the
     same task family and writes a Skill.md that captures the procedure.
     Best quality, costs Opus tokens.
  2. Template-driven extraction: cluster traces by their user prompts,
     extract the common pattern of tool calls + final answers, emit a
     skeleton skill the user reviews. Cheap, deterministic.

This module ships (2) as a working baseline and exposes a hook for (1).
A real coordinator runs (2) automatically and queues candidates for
human approval; (1) runs on the most promising clusters.

The intent is that this is the *trigger* for durable improvement: when
the agent repeats a task class three times successfully, the next time
the user asks something similar, a skill is already waiting.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from agi.skills import Skill


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _top_terms(texts: Iterable[str], k: int = 5, min_len: int = 4) -> list[str]:
    """The k most-common content tokens across `texts` (length-filtered)."""
    counts: Counter[str] = Counter()
    for t in texts:
        for tok in set(_tokenize(t)):  # set: count one per text
            if len(tok) >= min_len:
                counts[tok] += 1
    return [tok for tok, _ in counts.most_common(k)]


@dataclass
class SkillCandidate:
    """A proposed skill awaiting review.

    Coordinators should *not* auto-commit candidates — surface them for
    human approval. Auto-committing risks training on poisoned patterns.
    """
    suggested_name: str
    suggested_description: str
    suggested_tags: list[str]
    body: str
    trace_count: int
    sample_prompts: list[str]

    def to_skill(self, name: str | None = None) -> Skill:
        return Skill(
            name=name or self.suggested_name,
            description=self.suggested_description,
            tags=list(self.suggested_tags),
            body=self.body,
        )


def cluster_traces_by_keyword(
    prompts: list[str],
    *,
    min_cluster_size: int = 3,
) -> list[list[int]]:
    """Cheap clustering: bucket prompts by their leading content keyword.

    For v1 this is enough to spot repeated task classes. Real clustering
    upgrades to embedding similarity later.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        tokens = [t for t in _tokenize(p) if len(t) >= 4]
        if not tokens:
            continue
        key = tokens[0]
        buckets[key].append(i)
    return [v for v in buckets.values() if len(v) >= min_cluster_size]


def propose_skill_from_cluster(
    prompts: list[str],
    responses: list[str],
    *,
    name_hint: str | None = None,
) -> SkillCandidate:
    """Build a SkillCandidate from a cluster of (prompt, response) pairs.

    The body is a deterministic template — readable, editable, honest
    about being a starting point. The user is expected to refine it.
    """
    assert len(prompts) == len(responses) and prompts, "need ≥1 example"
    tags = _top_terms(prompts, k=4)
    name = name_hint or (tags[0] if tags else "compiled_skill")
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_") or "compiled_skill"
    description = (
        f"Compiled procedure for tasks matching: {', '.join(tags) or 'this pattern'}"
    )
    body = (
        "## When to use\n"
        f"User asks about: {', '.join(tags) or '<keywords>'}\n\n"
        "## Procedure\n"
        "1. Restate the user's request in your own words to verify scope.\n"
        "2. Identify required inputs; ask for them if missing.\n"
        "3. Execute (mirror the approach in the example outputs below).\n"
        "4. Verify the result against the user's request before responding.\n\n"
        "## Example outputs (from successful past runs)\n\n"
        + "\n\n".join(
            f"- Prompt: {p[:120]!r}\n  Output: {r[:200]!r}"
            for p, r in zip(prompts[:3], responses[:3])
        )
    )
    return SkillCandidate(
        suggested_name=name,
        suggested_description=description,
        suggested_tags=tags,
        body=body,
        trace_count=len(prompts),
        sample_prompts=list(prompts[:3]),
    )


def mine_skills(
    pairs: list[tuple[str, str]],
    *,
    min_cluster_size: int = 3,
) -> list[SkillCandidate]:
    """Produce skill candidates from a list of (prompt, final_text) tuples.

    Typical caller: a coordinator that just observed N successful Tasks
    completing and wants to compile their patterns into skills.
    """
    prompts = [p for p, _ in pairs]
    responses = [r for _, r in pairs]
    clusters = cluster_traces_by_keyword(prompts, min_cluster_size=min_cluster_size)
    out: list[SkillCandidate] = []
    for idxs in clusters:
        cluster_prompts = [prompts[i] for i in idxs]
        cluster_responses = [responses[i] for i in idxs]
        out.append(propose_skill_from_cluster(cluster_prompts, cluster_responses))
    return out
