# Architecture

A system that learns and adapts to new input, exposed as a runtime that a
coordination engine can drive. Honest about what's open research vs. what's
tractable engineering.

## Two views

There are two ways to look at this system. Both are first-class:

**View 1 — the agent.** A goal-directed loop over a reasoning core, with
tools, memory, skills, and a world model. This is what you interact with
when you run `python -m agi`.

**View 2 — the runtime.** The same loop, exposed over a stable HTTP/JSON
protocol so an external coordination engine can submit tasks, dispatch task
graphs, stream events, and verify outputs. This is the integration surface.

The agent *is* the runtime. The HTTP server in `agi/runtime/` is just an
adapter that surfaces the same primitives to network callers. A coordination
engine can target the in-process Runtime directly, or talk to it over HTTP —
the protocol is the same.

## Runtime API (the contract for a coordination engine)

The runtime exposes the following over HTTP:

| Endpoint                       | Verb   | Purpose                                  |
|--------------------------------|--------|------------------------------------------|
| `/capabilities`                | GET    | Machine-readable descriptor              |
| `/tasks`                       | POST   | Submit a task (`kind`, `input`, `role`)  |
| `/tasks/{id}`                  | GET    | Task status + result                     |
| `/tasks/{id}/stream`           | GET    | SSE stream of task events                |
| `/tasks/{id}/cancel`           | POST   | Request cancellation                     |
| `/graphs`                      | POST   | Submit a typed task DAG (`GraphSpec`)    |
| `/graphs/{id}`                 | GET    | Latest event for the graph               |
| `/graphs/{id}/stream`          | GET    | SSE stream of graph events               |
| `/events`                      | GET    | SSE stream of all runtime events         |
| `/skills` , `/skills/{name}`   | GET    | List / read skills                       |
| `/healthz`                     | GET    | Liveness                                 |

Task kinds the runtime understands today:

- `chat` — full agent turn, returns `{text, critic_score?}`
- `plan` — decompose a goal into a `GraphSpec`
- `critique` — score a candidate response, returns `{score, explanation}`
- `skill.invoke` — run a named skill from the library
- `tool` — run a single registered tool directly (bypass LLM)
- `noop` — synthetic success, useful for graph-shape testing

Features baked into the protocol:

- **Idempotency** via `dedup_key` — resubmitting returns the same task id.
- **Budgets** per task (`budget_tokens`, `budget_seconds`).
- **Cancellation** — workers observe a cancel flag between LLM turns.
- **Parameter substitution** in graphs — node inputs can reference
  upstream outputs as `${node_id.field}`; the executor resolves before
  dispatch.
- **Per-node failure policy** — `fail_graph` (default), `skip`, `retry:N`.

The protocol's source of truth is `agi/runtime/capabilities.py`. Anything
not described there is not part of the contract.

## The agent's loop

A single `chat()` turn does the following:

1. On the first turn, retrieve top-K skills from the library by keyword
   match against the prompt; inject as a user-role context message.
2. Append the user message; call the model with the tool surface plus a
   role-aware system prompt.
3. Stream the response. On `tool_use`, dispatch the handler; on
   `pause_turn`, re-call; loop.
4. After `end_turn`, optionally score the final text with the trained
   critic (drops below threshold get an uncertainty annotation).
5. Persist the full trace (messages + usage + metadata) to JSONL for the
   learning loop.

The runtime adds: a thread-safe task store, a worker pool that pulls tasks
off a queue, and an event bus that emits structured events as each task
runs. A coordination engine subscribes to the bus to observe progress.

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

## The coordination engine

A coordination engine sits *above* the runtime. It owns the goal, decides how
to decompose it, dispatches into one or more runtimes, and aggregates the
result. We ship a reference implementation in `agi/coordination/` so the
protocol has at least one consumer in-tree.

```
┌────────────────────────────┐         submit graph        ┌────────────────────────┐
│   Coordination engine      │ ──────────────────────────▶ │     Runtime            │
│                            │                             │                        │
│  - owns the goal            │ ◀──────── events ────────── │  - workers             │
│  - plans (via runtime)      │       (SSE / in-proc bus)   │  - graph executor      │
│  - dispatches graphs        │                             │  - skills + memory     │
│  - verifies + revises       │ ──── critique task ───────▶ │  - trace log           │
│  - aggregates outcomes      │                             │                        │
└────────────────────────────┘                             └────────────────────────┘
```

The split matters:
- The **runtime** is stateless w.r.t. goals. It takes tasks, runs them, emits events.
- The **coordinator** owns the goal lifecycle, the revision policy, and the budget.
- One coordinator → many runtimes (sharding by capability, region, or trust).
- Many coordinators → one runtime (the runtime is the shared substrate).

This is the same shape as a process scheduler over CPUs, or a workflow engine
over executors. We're applying that pattern to LLM-driven work.

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

## Runtime module layout

The runtime layer is composed of small, replaceable pieces. Each one is the
in-process implementation of one row of the protocol contract.

| Module                          | Owns                                              |
|---------------------------------|---------------------------------------------------|
| `agi/runtime/tasks.py`          | `Task` state machine + `TaskStore` (dedup, parent/child) |
| `agi/runtime/events.py`         | `EventBus` — pub/sub with bounded per-topic history |
| `agi/runtime/worker.py`         | Agent-backed handlers for each task `kind`        |
| `agi/runtime/graph.py`          | DAG executor with `${node.field}` substitution and per-node failure policy |
| `agi/runtime/server.py`         | stdlib `ThreadingHTTPServer`, SSE streaming, JSON body parsing |
| `agi/runtime/capabilities.py`   | Single source of truth for the protocol descriptor |
| `agi/coordination/coordinator.py` | Reference coordinator: plan → execute → verify → revise |
| `agi/skills/library.py`         | Markdown-with-frontmatter SOPs, keyword retrieval |
| `agi/world_model.py`            | Observed-entity log: file/url/command + outcome  |
| `agi/tools_extension.py`        | `delegate`, `make_tool`, `plan_graph`, `invoke_skill`, `remember_observation` |
| `agi/planner.py`                | LLM-driven goal-to-GraphSpec decomposition       |

The contract is the HTTP protocol + the `GraphSpec`/`TaskSpec` JSON shapes.
Modules are swappable as long as the contract holds.
