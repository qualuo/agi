"""Reflection journal.

After a task completes, write a compact one-paragraph lesson to memory under
the `lesson` tag. Lessons are surfaced at the start of related tasks via
memory search, giving the agent a cheap form of post-task learning that
doesn't require weight updates.

The reflector uses the existing Memory store rather than a parallel file: a
lesson IS a kind of memory, just with a known tag and a known authorship
("self"). Keeping them in one place means a single search hits both raw notes
and reflections.
"""
from __future__ import annotations

from dataclasses import dataclass

from agi.memory import Memory


REFLECT_SYSTEM = """\
You are reviewing one of your own task attempts to extract a durable lesson.

Write ONE concise paragraph (<= 60 words) capturing what worked, what didn't,
and what to do differently next time. Do not narrate; state the lesson. If the
task was trivial and there is no real lesson, output the single token NONE.
"""


@dataclass
class Reflection:
    task: str
    text: str
    passed: bool | None = None


class Reflector:
    """Turn (task, response) into a one-paragraph lesson and persist it.

    The actual LLM call is injected so this module stays decoupled from the
    Agent and trivial to test. A coordination engine can pass any callable
    matching the signature `(system, user) -> str`.
    """

    def __init__(
        self,
        memory: Memory,
        complete: "callable[[str, str], str] | None" = None,
        min_chars: int = 4,
    ) -> None:
        self.memory = memory
        self._complete = complete
        self.min_chars = min_chars

    def reflect(
        self,
        task: str,
        response: str,
        passed: bool | None = None,
        extra_tags: list[str] | None = None,
    ) -> Reflection | None:
        if self._complete is None:
            return None
        user = (
            f"Task: {task.strip()}\n\n"
            f"My response: {response.strip()}\n\n"
            f"Outcome: {'passed' if passed else 'unknown' if passed is None else 'failed'}\n\n"
            f"Lesson:"
        )
        try:
            text = self._complete(REFLECT_SYSTEM, user).strip()
        except Exception:
            return None
        if not text or text.upper() == "NONE" or len(text) < self.min_chars:
            return None

        tags = ["lesson"] + (extra_tags or [])
        if passed is True:
            tags.append("passed")
        elif passed is False:
            tags.append("failed")

        self.memory.save(text, tags=tags)
        return Reflection(task=task, text=text, passed=passed)
