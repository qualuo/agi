"""Skill compilation — distill recurring successful procedures into skills.

Reads the trace log, filters for high-quality completed traces, groups
them roughly by lexical signature, and asks an LLM to propose 0-5 new
skills (or refinements to existing ones). Output is *proposals*, not
committed skills — a human (or a downstream critic) approves before
SkillLibrary.add() is called.

Why proposals-only: skill drift is a real risk. A bad skill that gets
auto-loaded into every system prompt poisons many future tasks. The
gating belongs in human or trusted-critic hands, not the same LLM that
wrote the proposal.

Trigger: run periodically (cron, post-eval, end-of-day) — there's no
fast-path to call this from inside an active session.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

import anthropic

from agi.skills import Skill, SkillLibrary

try:
    from learner.traces import Trace, TraceLogger
    from learner.filter import eval_passing, filter_traces, min_quality
except ImportError:  # learner package optional
    Trace = TraceLogger = None  # type: ignore
    eval_passing = filter_traces = min_quality = None  # type: ignore


COMPILER_SYSTEM = """\
You are a skill compiler. You will be shown summaries of agent traces
that completed successfully. Your job is to propose 0-5 reusable skills
the agent could load on similar future tasks.

A skill is a short procedure: when to use it, what to do, and known
failure modes. It must be GENERAL — applicable to a class of tasks, not
a one-off. If the trace pool doesn't show a clear repeated pattern, the
right answer is an empty list.

Output strict JSON:

{"proposals": [
  {
    "title": "<short title>",
    "when": "<one sentence describing when to use it>",
    "procedure": "<numbered steps>",
    "failure_modes": "<known failure modes>",
    "triggers": ["<trigger>", "<keyword>"]
  }
]}

Existing skills (do NOT propose duplicates of these — propose
refinements instead, marked clearly):

{existing}

Aim for 1-3 high-confidence proposals over 5 mediocre ones. Empty
list is the right answer most of the time.
"""


@dataclass
class SkillProposal:
    title: str
    when: str
    procedure: str
    failure_modes: str
    triggers: list[str]

    def commit(self, library: SkillLibrary) -> Skill:
        return library.add(
            title=self.title,
            when=self.when,
            procedure=self.procedure,
            failure_modes=self.failure_modes,
            triggers=self.triggers,
        )


def _summarize_trace(t) -> str:
    """One-paragraph summary of a trace for the compiler prompt."""
    user_first = ""
    assistant_last = ""
    tool_calls: list[str] = []
    for m in t.messages:
        content = m.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    tool_calls.append(b.get("name", ""))
            text = " ".join(text_parts)
        else:
            text = repr(content)
        if m["role"] == "user" and not user_first:
            user_first = text[:300]
        elif m["role"] == "assistant":
            if text:
                assistant_last = text[:300]
    tools_str = ", ".join(tool_calls[:8]) if tool_calls else "(none)"
    return (
        f"USER: {user_first}\n"
        f"TOOLS: {tools_str}\n"
        f"FINAL: {assistant_last}"
    )


def collect_summaries(
    traces: Iterable,
    *,
    max_traces: int = 30,
) -> list[str]:
    summaries: list[str] = []
    for t in list(traces)[-max_traces:]:
        summaries.append(_summarize_trace(t))
    return summaries


def propose_skills(
    summaries: list[str],
    *,
    library: SkillLibrary,
    model: str = "claude-sonnet-4-6",
    client: anthropic.Anthropic | None = None,
    max_tokens: int = 2000,
) -> list[SkillProposal]:
    """Ask the model to propose new skills given trace summaries.

    Sonnet 4.6 by default — Haiku tends to over-propose, Opus is
    overkill for distillation. Tune via `model=`.
    """
    if not summaries:
        return []
    client = client or anthropic.Anthropic()

    existing = library.all()
    existing_block = "\n".join(
        f"- {s.title}: {s.when[:100]}" for s in existing
    ) or "(none)"
    system = COMPILER_SYSTEM.replace("{existing}", existing_block)

    user_msg = "Trace summaries:\n\n" + "\n\n---\n\n".join(summaries)
    user_msg += "\n\nPropose skills now. JSON only."

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")

    return _parse_proposals(text)


def _parse_proposals(text: str) -> list[SkillProposal]:
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    raw = data.get("proposals") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[SkillProposal] = []
    for entry in raw[:5]:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title")
        when = entry.get("when")
        procedure = entry.get("procedure")
        if not all(isinstance(x, str) and x.strip() for x in (title, when, procedure)):
            continue
        failure_modes = entry.get("failure_modes")
        if not isinstance(failure_modes, str):
            failure_modes = "(none recorded)"
        triggers = entry.get("triggers")
        if not isinstance(triggers, list):
            triggers = []
        triggers = [t.strip() for t in triggers if isinstance(t, str) and t.strip()][:6]
        out.append(SkillProposal(
            title=title.strip()[:100],
            when=when.strip()[:300],
            procedure=procedure.strip()[:1500],
            failure_modes=failure_modes.strip()[:500],
            triggers=triggers,
        ))
    return out


def compile_from_traces(
    *,
    library: SkillLibrary,
    trace_path: str | None = None,
    model: str = "claude-sonnet-4-6",
    client: anthropic.Anthropic | None = None,
    max_traces: int = 30,
    require_eval_pass: bool = True,
) -> list[SkillProposal]:
    """End-to-end: read traces, filter, summarize, propose. Does not commit.

    Returns proposals; caller decides which to `.commit(library)`.
    """
    if TraceLogger is None:
        raise RuntimeError("learner.traces not available; install learner extras")
    logger = TraceLogger(path=trace_path)
    traces = logger.all()
    if require_eval_pass:
        traces = filter_traces(traces, eval_passing)
    summaries = collect_summaries(traces, max_traces=max_traces)
    return propose_skills(summaries, library=library, model=model, client=client)
