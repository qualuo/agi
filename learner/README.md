# learner

The learning side of the system. Builds a small open base + LoRA adapter that
durably learns from agent traces, alongside the frozen-Opus harness in `agi/`.

See [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the full design.

## What's here

| File | What it does | Status |
|---|---|---|
| `traces.py` | Append-only JSONL trace logger | Shipped |
| `filter.py` | Quality gates: `eval_passing`, `min_quality`, `user_thumbs_up` | Shipped |
| `train.py` | LoRA SFT script (HuggingFace transformers + PEFT + trl) | Shipped, needs GPU |
| `local_agent.py` | Local base + adapter inference, same chat interface as `agi.Agent` | TODO |

## Pipeline

```
agi.Agent            ──▶ traces.jsonl
   (with tracer)
                       │
                       ▼
                  filter.py  ──▶  filtered traces
                                       │
                                       ▼
                                  train.py  ──▶  adapters/v1/
                                                      │
                                                      ▼
                                            local_agent.py  ──▶  evaluate
                                                                     │
                                                                     ▼
                                                          (deploy if eval improves,
                                                           else reject and rollback)
```

## Running it (requires GPU)

```sh
# 1. Install learner deps
pip install -e ".[learner]"

# 2. Collect traces by running the eval suite with tracing on
#    (the eval runner needs the --tracer flag — TODO)
python evals/run.py --tracer

# 3. Train a LoRA adapter on filtered traces
python -m learner.train \
    --base Qwen/Qwen2.5-3B-Instruct \
    --filter eval \
    --out ./adapters/v1

# 4. Evaluate the adapter (TODO)
python evals/run.py --agent local --adapter ./adapters/v1
```

## Key design choices and their open questions

- **Base model:** default `Qwen/Qwen2.5-3B-Instruct` for fast iteration, but the
  CLI `--base` flag accepts any HF causal LM. Open: when do we move to 7B / 70B?
- **Quality signal:** v1 uses objective eval pass/fail. Open: how do we get
  signal on non-eval real-world tasks? User thumbs are sparse; LLM judges drift.
- **Catastrophic forgetting:** mitigated by eval-gated rollback (an adapter
  that regresses on the held-out suite is rejected). Not solved.
- **Agentic SFT:** v1 strips tool_use/tool_result blocks and trains text-only.
  This is a known simplification — agentic behavior won't transfer well. Coming
  back to this when basic loop is working.
- **Online vs batch:** v1 is batch (train periodically). True online weight
  updates is research-open and we're not pretending otherwise.

## Honest scope

The learning loop in this directory does **not** produce AGI. It produces a
small specialized model that durably adapts to specific workloads. Whether it
ever beats frozen-Opus on anything depends on (a) how narrow the workload is,
(b) how much trace data we collect, and (c) how good the quality signal is.
The eval comparison is the experiment.
