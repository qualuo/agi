"""Post-task reflection.

After a task completes the runtime can run a lightweight reflection pass that
writes a one-line lesson to long-term memory. The reflection is *deterministic
extraction*, not another LLM call — cheaper, predictable, and good enough as
a v1. An LLM-based reflection can replace this behind the same interface.

Each reflection writes one or more notes tagged `lesson` (failure mode) or
`procedure` (successful pattern). The next task's retrieval step picks them up.
"""
from __future__ import annotations

import re

from agi.memory import Memory


_FAILURE_HINTS = (
    "error:",
    "Traceback",
    "command not found",
    "Permission denied",
    "No such file",
    "[exit 1]",
    "[exit 2]",
    "TypeError",
    "ValueError",
)

_SUCCESS_HINTS = (
    "[exit 0]",
    "passed",
    "succeeded",
    "done.",
    "saved",
)


def _scan_tool_blocks(messages: list[dict]) -> tuple[list[str], list[str]]:
    """Walk assistant/tool messages and pull out failure and success snippets."""
    failures: list[str] = []
    successes: list[str] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") != "tool_result":
                continue
            text = b.get("content") or ""
            if not isinstance(text, str):
                continue
            sample = text[:240].replace("\n", " ")
            if any(h in text for h in _FAILURE_HINTS):
                failures.append(sample)
            elif any(h in text for h in _SUCCESS_HINTS):
                successes.append(sample)
    return failures, successes


def reflect(
    *,
    memory: Memory,
    prompt: str,
    response: str,
    messages: list[dict],
    eval_passed: bool | None,
    tags: list[str] | None = None,
) -> list[str]:
    """Write 0-2 notes to memory summarizing the task. Returns the note IDs.

    Strategy:
    - If the task failed (eval_passed=False), record the first failure snippet
      under tag `lesson` so the next attempt can read it.
    - If the task succeeded and tool errors were recovered from, record what
      worked under tag `procedure`.
    - If nothing notable happened, write nothing.
    """
    failures, successes = _scan_tool_blocks(messages)
    base_tags = list(tags or [])
    written: list[str] = []
    prompt_summary = _summarize(prompt)

    if eval_passed is False or (failures and not successes):
        if failures:
            note = memory.save(
                f"lesson on '{prompt_summary}': hit '{failures[0]}' — needed different approach",
                tags=[*base_tags, "lesson", "failure"],
            )
            written.append(note.id)
        elif eval_passed is False:
            note = memory.save(
                f"lesson on '{prompt_summary}': task failed — response was '{_summarize(response)}'",
                tags=[*base_tags, "lesson", "failure"],
            )
            written.append(note.id)

    if eval_passed is True and failures and successes:
        # recovered: worth recording the recovery pattern
        note = memory.save(
            f"procedure for '{prompt_summary}': recovered from '{failures[0]}' and succeeded",
            tags=[*base_tags, "procedure", "recovery"],
        )
        written.append(note.id)

    return written


def _summarize(s: str, n: int = 80) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"
