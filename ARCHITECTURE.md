# Architecture

A system that learns and adapts to new input. Honest about what's open research
vs. what's tractable engineering.

## The core insight: learning operates at multiple timescales

A frozen LLM is the wrong primitive to ask "does this system learn?" of, because
that question conflates four different timescales of adaptation:

| Timescale | What changes | How it changes | Cost |
|---|---|---|---|
| **Seconds** (per-turn) | Working memory | Append to context window | Free |
| **Minutes** (per-task) | Episodic memory | Write a JSONL note | Free |
| **Hours** (per-N-tasks) | Skill library | Distill successful procedures | Cheap LLM call |
| **Days** (per-batch) | Adapter weights | LoRA SFT on filtered traces | GPU hours |
| **Months** (frontier release) | Base model weights | Retraining by Anthropic/etc. | Out of scope |

Real-time per-token weight updates is **research-open** (catastrophic forgetting,
no clean signal, no rollback). The architecture sidesteps this by composing the
*next four faster timescales* and accepting that the slowest one happens elsewhere.

A system updating across all four faster timescales adapts substantially to new
input — not the same as biological learning, but durably more than scaffolding
alone.

## Components

```
                  ┌────────────────────────────────────────┐
                  │                Input                    │
                  │   (user msg, tool result, env state)    │
                  └──────────────────┬─────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │                Working Memory                     │
        │   (current task context, plan, partial results)   │
        └─────┬──────────────────────────────────┬─────────┘
              │ retrieve                         │ retrieve
              ▼                                  ▼
    ┌─────────────────┐                ┌──────────────────┐
    │    Long-term    │                │   Skill Library  │
    │     Memory      │                │   (procedural,   │
    │   (episodic,    │                │  named skills,   │
    │    semantic)    │                │   markdown SOPs) │
    └────────┬────────┘                └────────┬─────────┘
             │ retrievals                       │ skill ref
             └──────────────┬───────────────────┘
                            ▼
              ┌──────────────────────────────┐
              │       Reasoning Core          │
              │  (LLM: frozen Opus OR         │
              │   small base + LoRA adapter)  │
              └──────────────┬────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │       Action / Tools         │
              │  (file/shell/web/memory)     │
              └──────────────┬───────────────┘
                             ▼
                          Outcome
                             ▼
              ┌──────────────────────────────┐
              │     Critic / Verifier        │  ← can be a separate model
              │  (tests pass? eval pass?     │     or the same one in another
              │   user thumbs up?)            │     prompt; this is the signal
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   Trace Logger (durable)      │
              └──────────────┬───────────────┘
                             ▼
            ┌────────────────┼────────────────────┐
            ▼                ▼                    ▼
  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐
  │ Memory writes   │  │ Skill writes │  │ LoRA training     │
  │ (per-task)      │  │ (per-success │  │ (per-N-traces)    │
  │                 │  │   pattern)   │  │                   │
  └────────┬────────┘  └──────┬───────┘  └─────────┬─────────┘
           │                  │                    │
           ▼                  ▼                    ▼
       (long-term         (skill            (adapter
        memory)            library)          checkpoints)
```

### 1. Working memory

The active task context: current conversation, current plan, partial results,
pending tool calls. Lives in the LLM's context window, managed explicitly so
it doesn't grow without bound.

**v1:** in-memory list of messages. **Later:** explicit summarization, attention
focus, plan state separate from message history.

### 2. Long-term memory

Persists across sessions. Three sub-types, different access patterns:

- **Episodic** — events that happened ("on 2026-04-12 I helped user with X").
  Timestamped, mostly append-only. Searched by recency + relevance.
- **Semantic** — facts ("user prefers TypeScript", "project foo uses postgres").
  Often updates in place (newer fact replaces older). Searched by topic.
- **Procedural** — *not* in this module; lives in the Skill Library.

**v1:** keyword search over JSONL (already shipped). **v2:** semantic search via
embeddings (Voyage AI, sentence-transformers, etc. — Anthropic doesn't ship
embeddings). **v3:** consolidation pass that compresses raw memory into compact
higher-level notes during idle time.

### 3. Skill library

Successful task decompositions saved as named, retrievable skills. Each skill
is a markdown file with: when to use it, the procedure, and known failure
modes. Loaded into context when the current task matches the skill description.

This is the medium-timescale learning channel. When the agent solves a novel
class of task, it writes a SKILL.md so the next instance is cheaper.

**v1:** flat directory of `.md` files; LLM-as-retriever picks relevant ones.
**Later:** structured retrieval, automatic skill compilation from successful
traces, deprecation of stale skills.

### 4. Reasoning core

The LLM that takes (working memory + retrievals) and produces (text or
tool calls). The architecture is **agnostic** about which model — the same
agent interface wraps either:

- **Frozen frontier model** (Claude Opus 4.7): strong general reasoning,
  cannot be improved by training. Today's best capability ceiling.
- **Small open base + LoRA adapter** (Qwen 2.5 3B / Llama 3.2 3B etc.):
  much weaker general reasoning, but the adapter durably learns from
  experience.

Running both head-to-head on the same eval suite is the experiment. The
hypothesis worth testing: on narrow repeated workloads, the learning model
catches up to or beats the frozen one over time. On novel hard tasks, the
frozen model wins indefinitely.

### 5. Action layer

Tools the model can call: file/shell/web/memory. Already shipped in the Opus
harness. The local-base agent will use the same tool surface for parity in
evaluation.

### 6. Critic / verifier

Produces the **learning signal**. Without a reliable signal, the system trains
on noise. Three sources, in order of trust:

1. **Objective** (tests pass, eval task pass, file diff matches expected) — fully
   trusted, the gold standard.
2. **User feedback** (thumbs up/down) — high-signal but sparse.
3. **LLM self-critique** — cheap and dense but biased; never train on this alone.

The critic writes `eval_passed: bool` (or `quality_score: float`) into the
trace metadata. The trace filter reads it.

### 7. Trace logger

Durable JSONL of every interaction. Already shipped (`learner.traces`). The
filter reads from here; training reads from the filter.

### 8. Update mechanisms

Three feedback loops, three timescales:

- **Memory write loop** — per-task. After a task completes, the agent (or a
  separate reflection step) writes any durable lessons to long-term memory.
- **Skill compilation loop** — per-N-successful-tasks. Triggered manually or
  periodically: an LLM pass reads recent successful traces and proposes new
  skills (or refinements to existing ones).
- **Adapter training loop** — per-batch (e.g., nightly, or every N traces).
  Filter traces → format as SFT → train LoRA → validate against eval suite →
  deploy if pass rate improves, reject otherwise.

The third loop is **gated by eval pass rate**. An adapter that regresses on
the eval suite is rejected, no matter how much "learning" it did. This is the
rollback story.

## Information flow for a single task

1. User sends a message.
2. Working memory gets the message + current state.
3. Retrieval: relevant long-term memory + relevant skills loaded into context.
4. Reasoning core (Opus or local base+adapter) produces text or tool call.
5. Tools execute; results return to working memory.
6. Loop on (4–5) until end-of-turn.
7. Critic verifies outcome → trace gets `eval_passed` / `quality_score`.
8. Trace logger writes to disk.
9. Memory write loop runs (sync) — durable lessons saved.
10. (Async, periodic) skill compilation reads recent traces.
11. (Async, periodic) adapter training reads filtered traces, trains, validates,
    swaps adapter on success.

## Runtime engine — the control plane around the agent

The components above describe one *worker* — one Agent reasoning on one task
at a time. Production use needs a control plane around it: a stable, callable
surface that lets an external **coordination engine** (workflow system, DAG
scheduler, custom orchestrator) treat the agent as a service rather than a
script.

```
                ┌────────────────────────────────────────┐
                │       External coordination engine      │
                │ (Temporal, Airflow, in-house, or the    │
                │  Coordinator in agi.coordinator)        │
                └──────────────────┬─────────────────────┘
                                   │ Python API or HTTP/SSE
                                   ▼
            ┌──────────────────────────────────────────────┐
            │                   Runtime                     │
            │  sessions · jobs · budgets · cancel ·         │
            │  event streams · metrics · snapshot/restore   │
            └──────────────────┬───────────────────────────┘
                               │ drives chat_controlled(...)
                ┌──────────────┴───────────────┬────────────┐
                ▼                              ▼            ▼
          ┌─────────┐                    ┌─────────┐   ┌─────────┐
          │ Agent A │                    │ Agent B │   │ Agent C │
          │ (sess1) │                    │ (sess2) │   │ (sess3) │
          └────┬────┘                    └─────────┘   └─────────┘
               │ tool: delegate
               ▼
          ┌─────────┐
          │ Agent D │  ← spawned through the same Runtime.
          │ (child) │     Costs and traces roll up.
          └─────────┘
```

**Sessions** are long-lived agent instances with stable IDs, isolated memory,
optional role / system-prompt overrides. **Jobs** are units of work submitted
to a session. Each job has:

- a $ budget, checked between agent turns (`BudgetExceeded` short-circuits
  runaway loops — the most important production-safety knob)
- a cooperative cancel signal (worker stops on the next iteration)
- a structured event stream (text deltas, tool_use, tool_result, status
  transitions, usage updates)
- a final record (`status`, `output`, `cost_usd`, `input/output_tokens`,
  `error`, `iterations`)

Concurrency: a `ThreadPoolExecutor` runs jobs across sessions in parallel; a
per-session lock serializes jobs *within* a session so the same Agent isn't
re-entered. Snapshot/restore persists session and job records to disk; jobs
that were running at snapshot time are marked `failed` on restore (honest
about what we can and can't recover).

**The agent as coordinator.** When Agent runs under a Runtime, a `delegate`
tool is registered: the agent can spawn a child session/job and synchronously
wait for the result. This is how the same primitive supports both "external
orchestrator drives N agents" and "one agent decomposes a task into
subagents." Parent IDs are recorded in metadata so the call graph is
reconstructable from traces.

**The Coordinator** in `agi.coordinator` is a small DAG executor that uses
the Runtime: declare nodes with `depends_on` and `{upstream}` placeholders,
get parallel execution where the graph allows, with upstream-failure
short-circuit. It is one possible coordinator; the point of the Runtime is
that any other coordinator — written in Python or any language with an HTTP
client — can drive the same surface.

**The HTTP control plane** in `agi.server` exposes the Runtime as JSON+SSE
over stdlib `http.server`. A coordination engine in Go, TypeScript, or Rust
can `POST /v1/sessions/{id}/jobs` and `GET /v1/jobs/{id}/events` to drive a
fleet of agents without speaking Python.

**Why this lives in the architecture doc, not just engineering.** A learning
system that can't be deployed as infrastructure is a research artifact. The
runtime is the bridge between the components above and a real coordinator —
it's what lets the durable-improvement loops (memory, skills, adapters) be
operated, observed, and budgeted in production rather than only on a
researcher's laptop.

## Pluggable choices

The architecture commits to *shape*, not *specifics*. These swap independently:

| Slot | v1 choice | Alternatives |
|---|---|---|
| Reasoning core | Opus 4.7 (frozen) + Qwen 2.5 3B (learning) | Llama 3.x, Mistral, GPT-4o (no learnable weights), local Phi |
| Memory backend | JSONL + keyword search | SQLite, vector DB, Postgres, hybrid |
| Embedding model | None (v1) | Voyage AI, sentence-transformers, Cohere |
| Critic | Eval-pass + user thumbs | LLM judge, structured rubric, learned reward model |
| Adapter format | LoRA via PEFT | DoRA, full fine-tune, prefix tuning |
| Training framework | trl SFTTrainer | unsloth, axolotl, custom |
| Skill format | Markdown SOPs | Code (eval'd), YAML, structured plans |

## What this architecture does not do

Honest list of limitations:

- **No real online learning.** The adapter updates in batches, not per-token.
  Catastrophic forgetting is sidestepped by versioned adapters + eval-gated
  rollback, not by solving the underlying problem.
- **No grounded world model.** The system reads about the world; it doesn't
  build causal models from interaction.
- **No goal coherence over weeks.** Long-horizon planning beyond multi-day
  is unsolved at scale; the architecture has no special trick for it.
- **No emergent generalization beyond the base.** Adapters specialize; they
  don't increase the underlying model's capacity for novel reasoning.
- **No safety story for the learning loop.** If the trace pool is poisoned
  or the critic is wrong, the adapter drifts. v1 mitigates with conservative
  filters and eval-gated rollback; this is not a substitute for real
  alignment work on a continuously-updating system.

These are **research problems**, not features we forgot. Naming them keeps the
plan honest.

## Initial implementation roadmap

Building this in stages, smallest viable end-to-end loop first.

### Stage 1 — Trace + filter + LoRA training (skeleton)

- [x] `learner/traces.py` — append-only JSONL trace logger
- [x] `learner/filter.py` — quality gates (eval_passing, min_quality, user_thumbs_up)
- [x] `learner/train.py` — LoRA SFT script (runs on GPU)
- [x] Wire trace logging into `agi.Agent`
- [ ] `learner/local_agent.py` — load base + adapter, expose same chat interface

### Stage 1.5 — First specialist: trace-quality critic ← current

The critic is the verifier component (above). Building it as a CPU-tractable
specialist gives us: a real learning loop running today, a useful artifact
that plugs into the architecture, and a substrate that grows as we collect
real traces.

- [x] `learner/goals.py` — `Goal` abstraction; `Addition` as first concrete goal
- [x] `learner/synth.py` — synthetic labeled-data generators
- [x] `learner/critic.py` — char-ngram featurizer + tiny MLP, train/predict/save/load
- [x] `learner/train_critic.py` — CLI; verified accuracy climbs from chance to 85% train / 74% eval on synthetic addition
- [x] Plug critic into `agi.Agent` as an optional output filter (drop responses below P(passed) threshold). v1 annotates; future: regenerate / refuse / structured uncertainty.
- [ ] Train critic on real traces once a meaningful pool accumulates
- [ ] Critic-as-reward for the LoRA loop in stage 2

### Stage 2 — Eval-gated adapter deployment

- [ ] Eval runner accepts `--agent local|opus` so we can compare
- [ ] Adapter validation: only deploy a new adapter if it ≥ previous on eval pass rate
- [ ] Versioned adapter storage in `~/.agi/adapters/{vN}/`
- [ ] Manifest tracking which traces trained which adapter (provenance)

### Stage 3 — Skill library

- [ ] `learner/skills.py` — directory of markdown skills, retrieve by description
- [ ] Skill compilation: an LLM pass that proposes new skills from recent
      successful traces, with human review before commit
- [ ] Integrate into Agent: load top-K relevant skills into system prompt

### Stage 4 — Semantic memory

- [ ] Embedding backend abstraction (`Memory.search` becomes pluggable)
- [ ] Voyage AI integration (or sentence-transformers for local)
- [ ] Episodic vs semantic split

### Stage 5 — Idle-time consolidation

- [ ] Background job: summarize raw memory into compact notes
- [ ] Skill deprecation: skills that haven't been used in N tasks get archived

### Stage 6+ — Scale

Decide based on eval results. If learning model is closing the gap on Opus,
push harder: bigger base, more data, more compute. If not, the architecture
might still be right but the model size is too small.

## What success looks like

The architecture is "working" when:

1. **Eval pass rate trends up** on the held-out suite over months, with no
   manual intervention beyond the scheduled training loop.
2. **$/passed-task drops** for repeat workloads (skills + memory paying off).
3. **Adapter wins on narrow workloads** — there exists at least one task
   class where the local-base+adapter beats frozen-Opus after sufficient
   exposure.
4. **No regressions on the held-out novel suite** — adapters are not
   forgetting general capability while specializing.

The architecture is "wrong" — and we should redesign — if:

- Eval pass rate plateaus or regresses despite training.
- Adapters consistently degrade general capability (catastrophic forgetting
  not solved by our gating).
- Skill library is never used by the agent (LLM doesn't retrieve them well).
- Cost of running this loop exceeds the value of the improvement.

## Discussion / open questions

These are real questions, not rhetorical:

1. **Should the reasoning core be one model or two?** A "fast" small model
   for routine action + "slow" frontier model for hard reasoning, like
   System 1 / System 2. Adds complexity; might be worth it.
2. **Where does long-horizon planning live?** Not yet a component above —
   currently absorbed into the Reasoning Core. Probably needs its own
   module once tasks span days.
3. **Verifier robustness.** If the critic is an LLM, how do we keep it
   from drifting in lockstep with the agent? Holdout verifier? Multi-model
   ensemble? Rule-based for whatever can be?
4. **Trace privacy.** Every interaction is logged. For a personal tool,
   fine. For multi-user, this needs thinking through.
5. **Adapter merging.** If we train multiple specialized adapters, can
   they be merged? (Research area: model merging, MoE-style.)
6. **Sim-to-real for tools.** Should the agent train in a sandbox first,
   then transfer to real tools? Risk of harmful actions during learning
   is otherwise non-trivial.
