"""Reference: how a coordination engine drives the agi Runtime.

A coordination engine is anything that sits above the runtime and:
  1. Discovers capabilities,
  2. Spawns sessions with policies (budget, tools, role),
  3. Reacts to the event stream,
  4. Persists durable artifacts (skills) across sessions.

This file is a runnable sketch using a FakeAgent so you can see the shape
of the integration without hitting the API. Swap the agent_factory for
None to use real Opus.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# Make the package importable when run directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.events import (
    CHAT_COMPLETED,
    ERROR,
    SESSION_CREATED,
    SUBAGENT_COMPLETED,
    TOOL_SYNTHESIZED,
)
from agi.runtime import Runtime, SessionConfig
from agi.skills import Skill


def coordinator_demo() -> None:
    # In production: leave agent_factory=None to use real Opus.
    # Here we pass a stub so the demo runs without an API key.
    from tests.test_runtime import FakeAgent
    runtime = Runtime(agent_factory=FakeAgent)

    # Observability — the coordinator's read on what the runtime is doing.
    counts: dict[str, int] = defaultdict(int)
    runtime.subscribe(lambda e: counts.__setitem__(e.kind, counts[e.kind] + 1))

    # Capabilities query: what models, skills, tools are available?
    caps = runtime.capabilities()
    print(f"runtime offers {len(caps['models'])} models, {len(caps['skills'])} skills")

    # Persist a durable procedure as a skill. Next sessions will load it
    # automatically when the user asks something relevant.
    runtime.save_skill(Skill(
        name="summarize_report",
        description="produce a 3-bullet summary of a technical report",
        body="1. Read the document.\n2. Extract claims, evidence, caveats.\n3. Emit exactly 3 bullets.",
        tags=["summarization", "writing"],
    ))

    # Policy: this session runs Opus with a hard budget, can delegate to
    # cheaper Haiku subagents, and uses the skill library.
    sid = runtime.create_session(SessionConfig(
        model="claude-opus-4-7",
        effort="high",
        enable_delegation=True,
        enable_tool_synthesis=False,
        use_skills=True,
        cost_ceiling_usd=2.50,
    ))

    runtime.chat(sid, "Summarize the attached quarterly report.")
    state = runtime.get_session(sid).to_dict()
    print(f"session {sid}: {state['turn_count']} turn, ${state['total_cost_usd']:.4f}")

    print(f"events seen: {dict(counts)}")


if __name__ == "__main__":
    coordinator_demo()
