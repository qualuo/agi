# agi

An agent runtime engine. Not AGI — that remains an unsolved scientific
problem. This is what you can credibly build today: a capable agent on top
of Claude Opus 4.7 with tools to act on the world, persistent memory and
skills across sessions, sandboxed self-extension via tool synthesis,
subagent delegation, and an event-driven runtime surface that a higher-level
coordination engine can drive.

## What's in here

```
agi/                # runtime + agent + reference coordinator
  runtime.py        # Runtime, Session, SessionConfig — the engine surface
  events.py         # EventBus + typed Event kinds (the coordination signal)
  server.py         # HTTP+SSE server exposing Runtime (stdlib only)
  protocol.py       # JSON-RPC 2.0 over stdio — drive the Runtime as a subprocess
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  coordinator.py    # reference Coordinator + Goal/Plan/PlanStep abstractions
  goalc.py          # Goal compiler: heuristic + LLM-based default decomposers
  autoloop.py       # AutonomousLoop — retry-with-lessons until goal accepted
  fork.py           # SessionFork — race N variants, pick winner by critic
  pool.py           # RuntimePool — federation: many runtimes, one dispatch surface
  capabilities.py   # observed-performance routing — learn which roles win where
  policy.py         # PolicyRouter — Thompson-sampled bandit on top of capabilities
  selfeval.py       # SelfEvalBank — agent-mined regression suite + promotion gate
  autonomy.py       # AutonomyEngine — continuous closed-loop self-improvement
  knowledge.py      # KnowledgeGraph — typed nodes + relations + facts
  governance.py     # multi-tenant budgets, quotas, rate limits, fair-share
  preflight.py      # cost/duration/p_success forecast + admission advisor
  mcp.py            # Model Context Protocol server: drive from Claude Desktop/Code
  evolve.py         # EvolutionEngine — closed-loop self-improvement over strategies
  contract.py       # TicketSLO + SLOCompiler + hedged execution + ComplianceLedger
  driver.py         # RuntimeDriver — single entry point with portfolio + SLO surfaces
  portfolio.py      # PortfolioOptimizer — fixed-budget allocation across many tickets
  scheduler.py      # ParallelScheduler — DAG-aware parallel plan execution
  skillmine.py      # mine reusable skills from successful trace patterns
  skills.py         # markdown skill library with retrieval (procedural memory)
  reflection.py     # per-task lessons-to-memory loop (medium-timescale learning)
  world_model.py    # observed-entity tracker (file/url/command + outcomes)
  toolsynth.py      # sandboxed Python tool synthesis (subprocess isolated)
  tasks.py          # Task / TaskQueue / TaskRunner — scheduled work
  persistence.py    # checkpoint sessions to disk and rehydrate
  memory.py         # persistent JSONL memory store + namespacing (multi-tenant)
  costs.py          # per-turn + cumulative token usage and $ tracking
  tools.py          # builtin tools: file, shell, web, memory (+ world auto-record)
  __main__.py       # CLI: python -m agi
learner/            # learning track — small open base + LoRA loop
  critic.py         # trace-quality critic (small MLP, trains on CPU)
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  goals.py          # Goal abstraction; Addition is the first concrete goal
  synth.py          # synthetic labeled data for critic warm-start
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner
tests/              # 290+ unit tests, all run without an API key
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## CLI

```sh
python -m agi                          # interactive REPL
python -m agi "summarize ./README.md"  # one-shot
python evals/run.py                    # run the eval suite
python -m agi.server --port 8765       # start HTTP runtime
```

## Coordinator — the reference driver

The `Coordinator` is a reference implementation of a coordination engine
sitting *above* the Runtime. It accepts a `Goal` (declarative intent +
budget), runs it through a pluggable `decomposer` to produce a `Plan`
of dependent `PlanStep`s, dispatches each step as a `Task` against the
runtime queue, and aggregates step results.

```python
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.runtime import Runtime

def planner(goal):
    return Plan(steps=[
        PlanStep(id="plan",       role="planner",    prompt=f"plan: {goal.intent}"),
        PlanStep(id="gather",     role="researcher", prompt=f"gather: {goal.intent}", depends_on=["plan"]),
        PlanStep(id="synthesize", role="writer",     prompt=f"summarize: {goal.intent}", depends_on=["gather"]),
    ])

result = Coordinator(Runtime(), decomposer=planner).run(
    Goal(intent="summarize LoRA adapters in production", budget_usd=1.0)
)
print(result.final_text)
```

The Coordinator talks to the Runtime only through its public API
(`create_session`, `chat`, `bus.subscribe`, `metrics`) — any other
planner can use the same surface. See `examples/coordinator_e2e.py`
for a full run including skill mining.

### Three drivers ship on top of the Runtime

These are coordination patterns shipped as small modules — a higher-level
coordination engine composes them or rolls its own:

- **`AutonomousLoop`** (`agi/autoloop.py`) — pursues a `Goal` across many
  attempts. Each failed attempt distills a lesson that is prepended to the
  next attempt's prompt; on success it mines a `SkillCandidate` from the
  winning trajectory. Halts on success, budget exhaustion, deadline, or
  iteration cap. Records every iteration to a `CapabilityRegistry` for
  downstream routing.

  ```python
  from agi.autoloop import AutonomousLoop, promote_skill
  from agi.capabilities import CapabilityRegistry

  caps = CapabilityRegistry()
  loop = AutonomousLoop(Coordinator(rt), max_iterations=4, capabilities=caps)
  result = loop.pursue(Goal(intent="…", acceptance=lambda t: "42" in t, budget_usd=1.0))
  if result.success and result.skill_candidate:
      promote_skill(rt, result.skill_candidate)  # graduate into the skill library
  ```

- **`SessionFork`** (`agi/fork.py`) — races N `SessionConfig` variants of the
  same prompt in parallel against the runtime's task queue and picks a
  winner via a pluggable `judge` (default: critic score, then succeeded,
  then cost). The cheapest way to lift pass rate on hard prompts.

  ```python
  from agi.fork import SessionFork, ForkVariant

  fork = SessionFork(rt, max_workers=4)
  race = fork.race("hard question", [
      ForkVariant("careful", SessionConfig(effort="high", role="planner")),
      ForkVariant("fast",    SessionConfig(effort="medium", role="executor")),
      ForkVariant("opus",    SessionConfig(model="claude-opus-4-7", role="reviewer")),
  ])
  print(race.winner.variant.name, race.winner.result)
  ```

- **`CapabilityRegistry`** (`agi/capabilities.py`) — append-only JSONL store
  of `(prompt_tokens, role, model, skills_used, success, cost, latency,
  critic_score)`. `recommend(prompt, budget_usd=…)` returns the best
  `(role, model)` bucket by similarity-weighted success rate, with a
  budget penalty. A coordinator queries this *before* dispatching work
  so each step picks the most-likely-to-succeed config.

  ```python
  rec = caps.recommend("compile this regex", budget_usd=0.05)
  cfg = rec.to_session_config(base=SessionConfig(max_tokens=8000))
  ```

See `examples/agi_demo.py` for an end-to-end narrated run that wires
all three together without an API key.

### Five more modules a coordination engine cares about

These extend the runtime into a federated, self-learning, externally
drivable engine — investor pitch: "the more you run it, the smarter,
cheaper, and harder-to-break it gets."

- **`PolicyRouter`** (`agi/policy.py`) — Thompson-sampling bandit over
  `(role, model, effort)` arms on top of `CapabilityRegistry`. Each
  decision draws from per-arm Beta posteriors conditioned on prompt
  similarity, penalised by expected cost. Real online learning at the
  routing layer; the policy converges to the right arm faster than
  the registry's similarity-weighted recommender.

  ```python
  from agi.policy import PolicyRouter

  router = PolicyRouter(caps, epsilon=0.05, cost_weight=5.0)
  decision = router.decide("compile this regex", budget_usd=0.05)
  cfg = decision.to_session_config()
  result = rt.chat(rt.create_session(cfg), "compile this regex")
  router.observe(prompt=..., decision=decision, success=True,
                 cost_usd=..., duration_seconds=...)
  ```

- **`RuntimePool`** (`agi/pool.py`) — federation layer. Add many
  `RuntimeNode`s (in-process today, HTTP/JSON-RPC out-of-process
  tomorrow); `pool.dispatch(prompt)` routes by skill match + node
  load + health. `aggregate_capabilities()` is the federation-wide
  view a coordinator sees.

  ```python
  from agi.pool import RuntimeNode, RuntimePool

  pool = RuntimePool()
  pool.add_node(RuntimeNode(node_id="gpu-1", runtime=rt1, tags=("gpu",)))
  pool.add_node(RuntimeNode(node_id="gpu-2", runtime=rt2, tags=("gpu",)))
  d = pool.dispatch("summarize this PDF", require_tag="gpu")
  ```

- **`CoordinationProtocol`** (`agi/protocol.py`) — newline-delimited
  JSON-RPC 2.0 over stdio. Any coordination engine (in any language)
  spawns `python -m agi` as a subprocess and drives it through:
  `runtime.capabilities`, `session.create/chat/cancel/end`,
  `tasks.submit/drain`, `events.subscribe/history`, `skills.save`,
  `tools.synthesize`. Notifications stream events back.

- **`SelfEvalBank`** (`agi/selfeval.py`) — mines `(prompt, expected
  substring/regex/min-length)` items from successful traces. Before
  promoting a new skill or synthesized tool, a coordinator calls
  `bank.gate_promotion(runner, baseline_pass_rate=...)` to refuse
  changes that regress the bank.

  ```python
  from agi.selfeval import SelfEvalBank

  bank = SelfEvalBank()
  bank.auto_mine(prompt=..., final_text=..., critic_score=0.95)
  ok, report = bank.gate_promotion(bank.runtime_runner(rt),
                                    baseline_pass_rate=1.0)
  ```

- **`goalc.heuristic_decomposer` / `goalc.llm_decomposer`** — the
  Coordinator's pluggable decomposer is now production-usable out
  of the box. The heuristic decomposer recognises common shapes
  (analyze / compare / build / find-and-summarize) and emits a
  multi-step DAG; the LLM decomposer asks a planner-role session to
  write a JSON Plan, reading the runtime's capabilities first. Use
  `chained_decomposer` to run heuristic-first, LLM-fallback.

  ```python
  from agi.goalc import chained_decomposer, heuristic_decomposer, llm_decomposer
  from agi.coordinator import Coordinator

  coord = Coordinator(rt, decomposer=chained_decomposer(
      heuristic_decomposer, llm_decomposer(rt), min_steps=2,
  ))
  result = coord.run(Goal(intent="analyze the impact of LoRA"))
  ```

See `examples/runtime_engine_demo.py` for a single narrated run that
exercises all five.

### Four platform layers for production deployments

- **`AutonomyEngine`** (`agi/autonomy.py`) — the *outer* loop. Pulls
  goals from a queue (anything that returns the next `Goal` or `None`),
  pursues each through `AutonomousLoop`, records outcomes to the
  `CapabilityRegistry`, mines skills from successes, gates promotion on
  `SelfEvalBank` regression, and writes new eval items back to the bank
  so the regression suite *grows from real use*. Run it as a heartbeat
  and the system measurably improves between invocations.

  ```python
  from agi.autonomy import AutonomyEngine, GoalQueue

  queue = GoalQueue()
  queue.push(Goal(intent="…", acceptance=lambda t: "42" in t))
  engine = AutonomyEngine(
      rt, Coordinator(rt),
      goal_provider=queue.as_provider(),
      eval_bank=bank,
      eval_runner=bank.runtime_runner(rt),  # gates skill promotion
      capabilities=caps,
      max_iterations=3, max_cost_per_tick_usd=0.50,
  )
  engine.run_forever(max_ticks=100, heartbeat_seconds=5.0, idle_grace_ticks=10)
  ```

  Emits `autonomy.tick_*`, `autonomy.goal_*`, `autonomy.skill_promoted`,
  `autonomy.skill_rejected`, `autonomy.evalbank_updated`, `autonomy.idle`.

- **`KnowledgeGraph`** (`agi/knowledge.py`) — typed nodes (`file`,
  `url`, `session`, `skill`, `project`, `user`, …) + directed relations
  (`depends_on`, `wrote`, `fetched`, `spawned`, …) + timestamped facts.
  `attach_to_bus(kg, runtime.bus)` makes the graph grow automatically
  from agent activity. `kg.neighborhood(node, hops=N)` and
  `kg.context_for(kind, key)` give a coordinator structured context
  to inject into the next prompt — real semantic memory, not keyword
  search.

  ```python
  from agi.knowledge import KnowledgeGraph, attach_to_bus

  kg = KnowledgeGraph()
  attach_to_bus(kg, rt.bus)
  ctx = kg.context_for("project", "agi", hops=2)  # ground next prompt
  ```

- **`PolicyManager` / `GovernedRuntime`** (`agi/governance.py`) — hard
  multi-tenant isolation. Per-tenant daily / hourly / lifetime cost
  caps, max concurrent sessions, prompts-per-minute / per-day rate
  limits, weighted fair-share scheduling across competing tenants, and
  an append-only JSONL audit log of every admission decision. The
  difference between a demo and a SaaS deployment.

  ```python
  from agi.governance import GovernedRuntime, PolicyManager, TenantLimits

  pm = PolicyManager(audit_path="/var/log/agi-audit.jsonl")
  pm.set_limits(TenantLimits("acme",
                             daily_cost_usd=10.0,
                             max_concurrent_sessions=5,
                             max_prompts_per_minute=60))
  gr = GovernedRuntime(rt, pm)
  sid = gr.create_session("acme", SessionConfig())
  text = gr.chat("acme", sid, "…")
  ```

- **`McpServer`** (`agi/mcp.py`) — exposes the Runtime as a Model
  Context Protocol server over stdio JSON-RPC. Claude Desktop, Claude
  Code, or any MCP-aware client connects with one config line and gets
  `agi.create_session`, `agi.chat`, `agi.run_goal`, `agi.recall`,
  `agi.autonomy.tick`, `agi.save_skill`, plus the live session/event
  resource feed. Distribution path: this runtime drops into any MCP
  host.

  ```python
  from agi.mcp import run_stdio
  run_stdio(rt, coordinator=coord, knowledge=kg, autonomy_engine=engine)
  ```

See `examples/agi_autonomy_demo.py` for an end-to-end run that wires
the autonomy engine, knowledge graph, capability registry, policy
router, self-eval bank, and policy manager together — no API key
needed.

### The closed loop: `EvolutionEngine`

- **`EvolutionEngine`** (`agi/evolve.py`) — the driver that turns the
  pieces above into an actual self-improvement loop a coordination
  engine can run on a schedule. Evolutionary search over agent
  `Strategy` variants (model × effort × role × system-prompt nudge ×
  skill overlay), scored on a benchmark from `SelfEvalBank` by
  ``fitness = pass_rate − cost_weight × mean_cost_usd``. Each
  generation: evaluate every strategy, select the top-k, mutate parents
  into children, eval-gate the winner, and *promote* — record outcomes
  in `CapabilityRegistry`, update `PolicyRouter` posteriors so future
  routing biases toward the winning arm, mine a skill from successful
  traces and save it to `SkillLibrary`, and grow the regression bank
  with newly-validated items.

  The artifact is an `EvolutionResult` with per-generation
  `fitness_curve`, `pass_rate_curve`, `mean_cost_curve` curves and a
  list of `PromotionRecord`s — what a UI displays as proof the runtime
  improves itself with use. Promotion is gated by the regression bank,
  so a generation that doesn't beat baseline is *rejected* and nothing
  contaminates the routing or skill layers.

  ```python
  from agi.evolve import EvolutionEngine, default_seed_strategies, runtime_runner

  engine = EvolutionEngine(
      runner=runtime_runner(rt),         # drives a real Runtime
      registry=caps, policy=router,      # closed-loop promotion targets
      skill_library=rt.skills, eval_bank=bank,
      cost_weight=2.0, seed=42,
  )
  result = engine.evolve(
      seed_strategies=default_seed_strategies(),
      benchmark=bank.all(),
      generations=4, top_k=2, children_per_gen=3,
  )
  print(result.summary())  # fitness/pass-rate/cost curves + promotions
  ```

  See `examples/evolve_demo.py` for a hermetic runnable demo that
  shows fitness climbing and cost falling across generations on a
  toy landscape.

## Runtime API — for a coordination engine

The `Runtime` is the integration point. A coordination engine (orchestrator,
planner, scheduler — anything sitting above) creates sessions, drives them,
observes the event stream, enforces budgets, and queries capabilities.

```python
from agi.runtime import Runtime, SessionConfig

rt = Runtime()

# Discover what this runtime can do
caps = rt.capabilities()
# → {models, skills, synthesized_tools, active_sessions, ...}

# Subscribe to the event stream before running anything
rt.subscribe(lambda e: print(e.kind, e.data))

# Spawn a session with a per-session budget
sid = rt.create_session(SessionConfig(
    model="claude-opus-4-7",
    effort="high",
    enable_tool_synthesis=True,   # agent can write new tools at runtime
    enable_delegation=True,        # agent can spawn subagents
    use_skills=True,               # relevant skills auto-loaded into prompt
    cost_ceiling_usd=5.00,         # session ends when budget is hit
))

result = rt.chat(sid, "Plan and execute: …")

# State + accounting available any time
print(rt.get_session(sid).to_dict())

# Persist a learned procedure as a durable skill
from agi.skills import Skill
rt.save_skill(Skill(
    name="bisect_by_test",
    description="locate a regression by running the test against bisected commits",
    body="1. Identify last-known-good commit.\n2. git bisect run …",
    tags=["debugging", "git"],
))
```

## Preflight — economic decisions before dispatch

A coordination engine driving the runtime needs *forecasts*: which task
to schedule now, which to defer, which to downgrade to a cheaper model.
The `Runtime` exposes a preflight estimator and an admission advisor
that produce those forecasts. The estimator self-trains on the runtime's
event stream — every completed chat refines future predictions.

```python
# Forecast cost / duration / p_success before committing to a chat
est = rt.estimate("Summarize this PDF and extract action items.")
# → Estimate(cost_usd=0.17, cost_p10/p90, duration_s=14.0, p_success=0.92,
#            confidence='low'|'medium'|'high', samples=N, breakdown=…)

# One-call admission decision combining preflight + governance + capacity
advice = rt.advise(
    "Render a long report",
    tenant_id="acme",       # optional — checks tenant budget/rate-limit
    config=SessionConfig(model="claude-opus-4-7"),
)
# advice.verdict ∈ {ADMIT, DEFER, DOWNGRADE, REJECT}
# DOWNGRADE carries a concrete alternative (cheaper model + expected savings)
# DEFER carries retry_after_s for the coordinator's scheduler
```

This is the missing piece for risk-aware coordination: instead of
dispatching blindly and burning budget on jobs that will be rate-
limited or fail, a coordinator can rank, defer, downgrade, or reject —
all from a single deterministic verdict.

See `examples/preflight_demo.py` for the full end-to-end walkthrough.

## RuntimeDriver — the one entry point a coordination engine uses

Preflight, admission, governance, dispatch, event streaming and billing
each have their own primitive. A coordination engine wiring them by hand
is brittle. `RuntimeDriver` collapses all of it into a single contract:

```python
from agi import RuntimeDriver, TicketRequest, PolicyManager, TenantLimits

policy = PolicyManager()
policy.set_limits(TenantLimits(tenant_id="acme", daily_cost_usd=10.0))

driver = RuntimeDriver(
    runtime=rt,
    policy=policy,
    receipts_path="receipts.jsonl",
    max_concurrent=8,
)

ticket = driver.submit(TicketRequest(
    intent="Summarize Q4 earnings call",
    tenant_id="acme",
    budget_usd=0.20,      # hard ceiling: passed through to the session
))

# Live progress
for ev in ticket.stream():
    ...

receipt = ticket.result()            # blocking; returns billing-grade Receipt
# receipt.status   ∈ completed | rejected | deferred | failed | cancelled
# receipt.decisions = [estimate → admission → (downgrade)? → route → dispatch → complete]
# receipt.estimated_cost_usd, receipt.actual_cost_usd, receipt.actual_duration_s
```

Every ticket carries a **causal decision trace** — the ordered list of
forks the driver took (estimate, admission verdict, optional downgrade,
node routing, dispatch, completion). The trace is what an operator
replays for audit, billing reconciliation, or post-hoc cost attribution.

Receipts are JSON-serializable and persist as JSONL — one line per
ticket — so a fleet of runtimes can stream billing into the same file
or pipe.

`RuntimeDriver` accepts either a single `Runtime` or a `RuntimePool`;
in the pool case the route decision records which node handled the
ticket, so a coordination engine can attribute cost across the fleet.

See `examples/driver_multi_tenant_demo.py` for the full demo: two
tenants, ten tickets, automatic model downgrade, hard per-ticket
budgets, and a fleet rollup at the end.

### Portfolio submission — fixed budget across many tickets

`RuntimeDriver.submit_portfolio` solves a different problem: you have N
tickets and **one shared budget**. Single-ticket admission is local
("can this one ticket afford to run?"); a portfolio decision is global
("which subset of these tickets, on which models, maximizes total
expected successes within $B?").

```python
requests = [TicketRequest(intent=t) for t in tasks]
tickets, plan = driver.submit_portfolio(
    requests,
    total_budget_usd=0.50,
    value_weights=priorities,    # weight each task's expected success
)

# `plan` is a JSON-serializable PortfolioPlan:
#   - one PortfolioAllocation per request with the chosen model
#     (or "skip" when no allocation is worth the marginal dollar)
#   - expected_cost_usd, expected_value, utilization
#   - `method` ∈ {"dp", "greedy"}; DP is exact, greedy is the fallback
#     for very large portfolios.
```

`driver.portfolio.frontier(requests, budgets=[0.05, 0.25, 1.00, ...])`
returns the budget → expected-value Pareto curve so operators can see
where the next dollar stops paying off.

See `examples/portfolio_demo.py` for an end-to-end walk-through: ten
tasks of varying priority, three budget tiers, a frontier curve, and a
live dispatch under shared accounting.

### SLO submission — declarative outcomes, hedged execution, compliance ledger

The portfolio API answers *"many tickets, one budget"*. The SLO API
answers the dual: *"one ticket, one objective"*. A coordination engine
declares what it wants — minimum success probability, maximum cost,
maximum latency — and the runtime compiles a concrete plan: one model
when feasible, a parallel hedge across several models when not.

```python
from agi import RuntimeDriver, TicketRequest
from agi.contract import TicketSLO

driver = RuntimeDriver(runtime=rt, compliance_path="compliance.jsonl")

slo = TicketSLO(
    min_p_success=0.95,        # I want >= 95% expected success
    max_cost_usd=0.40,         # spend up to 40 cents
    max_latency_s=30.0,        # finish in 30s wall-clock
    hedge_policy="auto",       # parallelize models if needed
    refund_on_breach=1.0,      # full refund credit on miss
)

slo_ticket = driver.submit_with_slo(TicketRequest(intent="..."), slo)

for ev in slo_ticket.stream():       # live progress (fan-in across hedges)
    ...
receipt = slo_ticket.result()        # SLOReceipt with compliance verdict
print(receipt.slo_status)            # "compliant" | "breached" | "infeasible" | "failed"
print(receipt.winner_model)          # which hedged candidate produced final_text
print(receipt.actual_cost_usd)       # aggregate cost across all hedged children
```

The compiler turns the SLO into one of two execution strategies:

  - **`STRAT_SINGLE`** — the cheapest single model whose forecast already
    meets the SLO floor. No hedge, no extra spend.
  - **`STRAT_HEDGE`** — when no single model is good enough within budget,
    greedily add candidates by uplift-per-marginal-dollar until the
    hedged success probability clears `min_p_success`. Children race;
    the first success wins and the rest are cancelled.

If the compiler reports `feasible=False` and the operator passes
`dispatch_infeasible=False`, the driver refuses up front — the SLO
ticket is returned already rejected, no spend, with `slo_status=infeasible`.

`driver.frontier_for_slo(request, slo, budgets=[...])` plots the Pareto
curve so an operator can size `max_cost_usd` on evidence — at $0.05 the
plan might be a single haiku at p≈0.78, at $0.20 it becomes a haiku +
sonnet hedge at p≈0.97, and the curve flattens above $0.50.

`driver.compliance_report()` rolls up the compliance ledger: hit rate,
breaches by kind (`cost` / `latency` / `infeasible_plan`), total
refund-eligible cost. A billing pipeline reads `compliance.jsonl` to
honor SLO refunds without bespoke plumbing.

See `examples/slo_contract_demo.py` for three scenarios — easy SLO,
tight quality (auto-hedge across three models), tight budget (infeasible,
rejected up front) — and the rolled-up compliance summary.

This is the surface a coordination engine actually wants: declarative
goals in, auditable outcomes out, with a paper trail you can bill against.

## HTTP / SSE surface

`python -m agi.server` exposes the Runtime over HTTP for out-of-process
coordinators:

| Method | Path                              | Purpose                                       |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/healthz`                        | Liveness check                                |
| GET    | `/capabilities`                   | What the runtime offers right now             |
| GET    | `/metrics`                        | Counters + totals for SLO/observability       |
| GET    | `/sessions`                       | List sessions                                 |
| POST   | `/sessions`                       | Create a session (body = SessionConfig + `namespace`) |
| GET    | `/sessions/{id}`                  | Inspect state                                 |
| POST   | `/sessions/{id}/chat`             | One turn; returns `{final_text, session}`     |
| POST   | `/sessions/{id}/cancel`           | Cancel between turns                          |
| POST   | `/sessions/{id}/reset`            | Clear conversation, keep session              |
| POST   | `/sessions/{id}/checkpoint`       | Persist session to the session store          |
| POST   | `/sessions/restore`               | `{session_id}` → reload from store            |
| DELETE | `/sessions/{id}`                  | End                                           |
| GET    | `/events`                         | SSE stream of all events                      |
| GET    | `/events?session_id=…&kind=…`     | Filtered SSE                                  |
| GET    | `/events/history`                 | Replay past events                            |
| GET    | `/tasks`                          | List queued/running/done tasks                |
| POST   | `/tasks`                          | Submit a task (prompt + budget + deadline)    |
| GET    | `/tasks/{id}`                     | Inspect a task                                |
| POST   | `/tasks/drain`                    | Run queued tasks (synchronous; `max_ticks`)   |
| POST   | `/skills`                         | Save a skill                                  |
| POST   | `/tools`                          | Synthesize a sandboxed tool                   |

Optional bearer-token auth via `AGI_AUTH_TOKEN` env var or `--auth-token`.

## stdio JSON-RPC surface

For coordinators that prefer spawning the runtime as a subprocess
(MCP-style), `CoordinationProtocol` exposes the same surface over
newline-delimited JSON-RPC 2.0 on stdin/stdout:

```python
from agi.runtime import Runtime
from agi.protocol import CoordinationProtocol
CoordinationProtocol(Runtime()).serve_stdio()
```

Methods: `ping`, `version`, `runtime.capabilities`, `runtime.metrics`,
`session.create/chat/cancel/end/get/list`, `tasks.submit/get/drain`,
`plans.submit/run/get/list/cancel`, `skills.save`, `tools.synthesize`,
`events.subscribe/unsubscribe/history`.
Notifications: `ready` (banner on connect), `event` (one per bus event
while subscribed).

### Parallel DAG plans — `ParallelScheduler`

`agi.scheduler.ParallelScheduler` is the coordination primitive when the
work has *shape*. Hand it a `Plan` (steps + dependencies) and it
dispatches independent steps in parallel up to `max_concurrent_steps`,
retries transient failures with exponential backoff, enforces per-plan
budget and deadline, and streams `plan.step.*` / `plan.completed` events.
The same surface is exposed over JSON-RPC as `plans.submit` / `plans.run`
for out-of-process coordinators.

```python
from agi.scheduler import ParallelScheduler, SchedulerConfig, RetryPolicy
from agi.coordinator import Plan, PlanStep

sched = ParallelScheduler(runtime, config=SchedulerConfig(
    max_concurrent_steps=4,
    retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.5),
))
result = sched.run(Plan(steps=[
    PlanStep(id="a", prompt="research X"),
    PlanStep(id="b", prompt="research Y"),
    PlanStep(id="c", prompt="synthesize", depends_on=["a", "b"]),
]), budget_usd=5.0)
```

See `examples/parallel_plan_demo.py` for a fan-out / fan-in walkthrough.

## Event kinds (the coordination contract)

The bus emits typed events. Coordinators pattern-match on `kind`:

- `session.created` / `session.ended`
- `chat.started` / `chat.completed`
- `usage.updated` — running token + cost totals
- `skill.loaded` — a skill was injected into a prompt
- `subagent.started` / `subagent.completed` — delegation
- `tool.synthesized` — agent extended itself
- `critic.scored` — gate fired with a confidence score
- `autoloop.iteration_started` / `autoloop.iteration_completed`
- `autoloop.completed` / `autoloop.failed` / `autoloop.budget_exhausted`
- `autoloop.skill_promoted` — a winning trajectory graduated into the library
- `fork.race_started` / `fork.race_completed`
- `pool.node_added` / `pool.node_removed` / `pool.node_unhealthy`
- `pool.dispatch_started` / `pool.dispatch_completed` / `pool.dispatch_failed`
- `plan.scheduled` / `plan.step.ready` / `plan.step.running`
- `plan.step.completed` / `plan.step.failed` / `plan.step.retry`
- `plan.completed` / `plan.failed` / `plan.budget_exhausted` / `plan.cancelled`
- `error` — including `CostCeilingExceeded` when budget runs out

Subagent token usage rolls up into the parent session for honest accounting.

## What it can do

- Read/write files, run shell commands
- Search the live web (`web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`)
- Load **relevant skills automatically** before answering (procedural memory)
- **Synthesize new tools mid-session** in a sandboxed subprocess (AST scan +
  banned imports + smoke test + per-call timeout)
- **Delegate subtasks to specialist subagents** with cost roll-up
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output and emit a typed event for every state transition
- Enforce per-session **cost ceilings** at the runtime layer
- Critic gate: scores final responses and annotates low-confidence ones
- **Learn which (role, model, effort) wins** on which prompts via a
  contextual Thompson-sampling bandit (`PolicyRouter`)
- **Federate over many runtimes** with skill- and load-aware dispatch
  (`RuntimePool`) — one coordinator, N runtime nodes
- **Speak JSON-RPC over stdio** so any external coordinator drives the
  runtime as a subprocess (`CoordinationProtocol`)
- **Mine its own regression suite** from successful traces and refuse
  promotions that regress it (`SelfEvalBank`)
- **Auto-decompose Goals** into multi-step DAGs via heuristic patterns
  or an LLM planner (`agi.goalc`)

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement on the base
model, robust out-of-distribution transfer, grounded world models, durable
goal pursuit across weeks of autonomy. The Opus reasoning core is frozen.
The `learner/` track is the path toward durable improvement via LoRA on a
small open base. See `ARCHITECTURE.md` for the design and what's open
research vs. tractable engineering.

## Testing

```sh
python -m unittest discover tests
# 230+ tests across events, skills, toolsynth, runtime, server, persistence,
# tasks, coordinator, autoloop, fork, capabilities, skillmine, agent, learner,
# policy, pool, protocol, selfeval, goalc
```

All tests run without an API key; they exercise the runtime, sandbox, and
HTTP server via a `FakeAgent` factory so CI doesn't burn budget.
