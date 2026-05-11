"""Reflection — per-task lesson distillation.

After a task completes (success or failure), an optional reflection pass
asks the model: "What's the durable lesson from this trace that's worth
remembering?" The lesson is saved to long-term memory with `lesson` tag and,
when the agent recognizes a reusable procedure, can be promoted to a skill.

Reflection is **optional and opt-in** at the runtime level — it costs an
extra model call per task. It's the cheapest learning channel that scales
without weight updates and is the substrate the skill-compilation loop reads.

Heuristics that keep reflections useful (not noise):
- Skip reflection on trivial tasks (≤ 1 tool call, ≤ 3 turns).
- Cap the lesson at ~200 chars to force compression.
- If the lesson is "no lesson", drop it — don't pollute memory.
"""
from __future__ import annotations

from dataclasses import dataclass


REFLECT_PROMPT = """\
The task is now complete (whether successfully or not). Reflect briefly:

1. What was the goal?
2. What worked? What didn't?
3. Is there ONE durable lesson worth remembering for similar future tasks?

If yes, respond ONLY with the lesson, one sentence, ≤200 chars, no preamble.
If there's no real lesson (trivial task, well-trodden territory), respond
with exactly: NO_LESSON

Do not save anything; the runtime will save your response.
"""


@dataclass
class Reflection:
    lesson: str | None  # None means NO_LESSON
    task_prompt: str
    response: str


def should_reflect(*, n_turns: int, n_tool_calls: int, response_chars: int) -> bool:
    """Heuristic gate. Don't reflect on trivial round-trips."""
    if n_turns <= 1 and n_tool_calls == 0:
        return False
    if response_chars < 40:
        return False
    return True


def parse_lesson(text: str) -> str | None:
    text = (text or "").strip()
    if not text or text.upper().startswith("NO_LESSON"):
        return None
    # Strip leading "lesson:" prefix if the model added it.
    for prefix in ("Lesson:", "lesson:", "LESSON:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Hard cap so reflections stay compressed.
    if len(text) > 240:
        text = text[:240].rsplit(" ", 1)[0] + "…"
    return text or None
