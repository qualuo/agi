# Plan

> **The plan changed.** This document was originally a roadmap for the
> Opus-only scaffold. After settling on a working definition of AGI
> (generality × autonomy × durable improvement) and admitting that durable
> improvement requires **weight learning** that scaffolding can't fake, the
> direction shifted to the dual-track architecture in
> [`ARCHITECTURE.md`](ARCHITECTURE.md): a strong frozen-Opus harness for
> capability and a small open base + LoRA loop for actual learning, run
> head-to-head on the same eval suite.
>
> Read `ARCHITECTURE.md` first. The stage list below is being rewritten
> against it; what's there now is the older, scaffolding-only plan, kept
> for context until the rewrite.

---

This is an engineering plan for an agent harness on top of a frozen frontier
model. It is **not** a research plan for AGI — those problems aren't solved by
scaffolding, and pretending otherwise would be theater.

The plan is structured as stages with **measurable** exit criteria. The eval
suite is the only objective signal of progress; everything is gated on it
moving.

## What scaffolding can and can't deliver

Things engineering can buy:
- Better tool surface (more reach, safer dispatch, cheaper).
- Long-horizon coherence (memory, planning, reflection, self-extension).
- Faster iteration (evals, traces, cost ceilings).
- Composition (subagents, skill libraries, tool synthesis).

Things engineering **cannot** buy on top of a frozen model:
- Online weight learning from experience.
- Grounded world models built from interaction with environments.
- Robust out-of-distribution generalization.
- Coherent goals over weeks of unsupervised operation.

These are research problems. The plan acknowledges them and stops there. The
honest signal of "are we near AGI" is whether frontier model releases keep
moving capability — not whether our harness is clever.

## Stage 0 — Where we are (now)

- **98 unit tests passing**, no API calls required to run them
- 6-task eval suite (math, file ops, shell, recall, multi-step planning)
- Streaming, adaptive thinking with summarized display, cost tracking
- Persistent JSONL memory with keyword search
- Tools: file/shell/memory/skills + server-side web_search and web_fetch
- **Runtime contract shipped** (`agi.runtime`, `agi.server`, `agi.events`):
  embeddable Session/Budget API, HTTP+SSE wire shape, snapshot/restore
- **Coordination tools shipped** (`agi.coordination`): `delegate` (subagents
  with rollup accounting) and `reflect` (per-task lesson)
- **Skill library shipped** (`learner.skills`): markdown SOPs the agent can
  read, write, and search; threaded into the system prompt at session start

## Stage 1 — Tighten the loop

**Goal:** every change has a measurable effect on eval pass rate.

- [ ] Expand evals to 30+ tasks covering: planning, recall over long contexts,
      tool composition, error recovery, web research, code synthesis with tests.
- [ ] Eval CI: a `make eval` target that runs the suite, prints pass rate +
      total $, and exits non-zero on regression.
- [ ] Trace logging: every chat session writes a JSONL to `~/.agi/sessions/`
      for replay and post-hoc analysis. Replay tool: re-run a session against
      a different model / effort to compare.
- [ ] Per-task cost ceiling. Tasks that exceed it fail with a "ran out of
      budget" reason — measures planning efficiency, not just success.
- [ ] Headline metric: $/passed-task. Watch it move.

**Exit criterion:** eval pass rate is the primary indicator we trust. We can
detect regressions same-day.

## Stage 2 — Self-extension

**Goal:** the agent extends its own capabilities, measurably.

- [ ] **Tool synthesis.** Add a `make_tool(name, description, code)` tool. The
      agent writes a Python function, tests it in a sandbox, and on success it
      registers as a callable tool for the rest of the session. Optionally
      promotes to persistent on user approval. Eval: novel-task pass rate
      before/after `make_tool` is enabled.
- [x] **Skill library.** Successful task decompositions get distilled into
      named skills (`save_skill` tool writes markdown to `~/.agi/skills/`).
      Threaded into the system prompt at session creation. Eval pending: same
      task family should be cheaper second time.
- [x] **Reflection journal.** `reflect` tool writes a structured one-paragraph
      "what worked / what didn't / lesson" to memory tagged `lesson`; the
      `search_memory` tool retrieves them. Eval pending: error-recovery rate
      over time.

**Exit criterion:** on a held-out repeat workload, $/passed-task drops by ≥30%
between first and second exposure.

## Stage 3 — Memory that changes thinking

**Goal:** memory that actually shapes behavior, not just recall.

- [ ] Pluggable embedding backend (Voyage AI, sentence-transformers, etc. —
      Anthropic doesn't ship embeddings). `Memory.search()` becomes semantic.
- [ ] Split memory into episodic (events with timestamps), semantic (facts),
      and procedural (skills). Different access patterns, different retention.
- [ ] Idle-time consolidation: when sessions end, an LLM pass summarizes
      raw memory into compact higher-level notes; old raw memory ages out.
- [ ] Long-horizon recall eval: 100+ turn conversations testing whether
      facts from turn 5 are correctly invoked at turn 95.

**Exit criterion:** ≥80% recall on the long-horizon eval at 100 turns.

## Stage 4 — Multi-agent

**Goal:** parallel work and specialization, where it pays off.

- [x] Subagent spawning. A `delegate(role, task)` tool spawns a sub-Agent,
      runs it to completion, and returns its final answer.
- [x] Specialized roles: `planner`, `executor`, `critic`, `researcher`,
      `summarizer`. Different system prompts; per-call model override.
- [x] Honest accounting: subagent token usage rolls up to the parent.
- [ ] Eval: are decomposed runs actually beating flat runs on $/passed-task?
      (Build the comparison harness next.)

**Watch out for:** coordination overhead. If subagents make tasks slower and
costlier without raising pass rate, kill the feature. This is an *experiment*,
not an assumption.

**Exit criterion:** on multi-step tasks where decomposition is plausible,
subagent runs beat flat runs on $/passed-task. If they don't, abandon.

## What this plan won't deliver

A scaffold on a frozen model **cannot**:
- Update its weights from interaction.
- Build new world models from grounded experience.
- Robustly generalize beyond its training distribution.
- Pursue goals coherently across weeks of unsupervised operation.

The agent gets smarter when the underlying model gets smarter, or when the
harness lets it use existing intelligence more effectively. The harness has a
ceiling; the model has a (current) ceiling. Honest progress = eval scores
moving on a held-out suite that grows over time.

## Decision points

- After each stage, ask: did pass rate move? Did $/passed-task improve? If no
  to both, the stage was theater — revert or redesign.
- If a stage takes more than 5 iterations without measurable gain, retire
  that direction and pick something else.
- **Replace this scaffold** if a frontier release ships built-in versions of
  these primitives that beat ours. Don't be precious about the code; it's a
  means.

## What this plan deliberately doesn't have

- A timeline. Schedules on AGI-adjacent work are fiction.
- Capability projections. "Stage 4 will achieve general reasoning" is the
  kind of thing that needs to be *measured*, not claimed.
- An end state called "AGI." Reaching AGI is a research outcome, not the
  delivery of a roadmap.
