"""Reflection — turn lived experience into durable memory.

After each task the agent runs a small reflection step: a cheap
LLM call that looks at the user prompt, the assistant's final answer,
and a summary of tool usage, and proposes 0–3 lessons to commit to
long-term memory tagged `lesson`. The lessons surface on the next
related task via memory search.

This is the simplest channel of medium-timescale learning that survives
across sessions without weight updates: explicit, inspectable notes
the agent reads at the start of related work.

Cost ceiling: one Haiku call per turn, max ~500 tokens. Falls open if
the API call fails — reflection is a nice-to-have, not blocking.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from agi.memory import Memory


REFLECTION_SYSTEM = """\
You are a reflection module attached to an autonomous agent. After every
task the agent completes you receive (the user prompt, the agent's final
answer, and a brief summary of what tools were used). Your job is to
distill 0-3 short, durable lessons worth saving to long-term memory.

Output strict JSON: {"lessons": [{"text": "<one sentence lesson>", "tags": ["lesson", ...]}]}.

Guidelines:
- A lesson must be reusable by a future task — generalize, don't restate.
- Skip if the task was trivial or no transferable lesson exists. Empty
  list is the right answer most of the time.
- Prefer concrete patterns over platitudes. Bad: "be careful". Good:
  "when reading a file that may not exist, list_dir the parent first".
- Tag with `lesson` plus topic tags (e.g. `shell`, `web`, `planning`).
- Maximum three lessons. Keep each under 200 characters.
"""


@dataclass
class ReflectionResult:
    lessons_saved: int
    error: str | None = None


class Reflector:
    def __init__(
        self,
        memory: Memory,
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 600,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.memory = memory
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def reflect(
        self,
        *,
        user_prompt: str,
        final_text: str,
        tools_used: list[str] | None = None,
    ) -> ReflectionResult:
        if not final_text.strip():
            return ReflectionResult(lessons_saved=0)
        tools_summary = (
            f"Tools called: {', '.join(tools_used)}." if tools_used else "No tools called."
        )
        user_msg = (
            f"USER PROMPT:\n{user_prompt}\n\n"
            f"AGENT FINAL ANSWER:\n{final_text}\n\n"
            f"{tools_summary}\n\n"
            "Output JSON now."
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=REFLECTION_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            return ReflectionResult(lessons_saved=0, error=f"{type(e).__name__}: {e}")

        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")

        lessons = _parse_lessons(text)
        for lesson in lessons:
            tags = list(set((lesson.get("tags") or []) + ["lesson"]))
            self.memory.save(lesson["text"], tags=tags)
        return ReflectionResult(lessons_saved=len(lessons))


def _parse_lessons(text: str) -> list[dict]:
    """Extract the `lessons` list from a JSON blob in `text`. Permissive."""
    if not text.strip():
        return []
    # Try strict parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Find the first {...} block that parses
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    lessons = data.get("lessons") if isinstance(data, dict) else None
    if not isinstance(lessons, list):
        return []
    out: list[dict] = []
    for entry in lessons[:3]:
        if not isinstance(entry, dict):
            continue
        text_field = entry.get("text")
        if not isinstance(text_field, str) or not text_field.strip():
            continue
        tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
        tags = [t for t in tags if isinstance(t, str)][:6]
        out.append({"text": text_field.strip()[:300], "tags": tags})
    return out
