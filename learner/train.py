"""LoRA SFT on filtered agent traces.

Runs HuggingFace transformers + PEFT (LoRA) on a small open base. Reads
traces from the trace log, filters by quality, formats as chat data,
trains a LoRA adapter, saves to disk. The local agent loads the latest
adapter at inference.

GPU REQUIRED. Even a 3B base needs ~24GB VRAM for LoRA SFT at reasonable
batch size. For experimentation: A100 40GB, RTX 4090, H100 — anything
with enough VRAM and CUDA. Apple Silicon MPS works for tiny models.

Install deps:
    pip install -e ".[learner]"

Run:
    python -m learner.train --base Qwen/Qwen2.5-3B-Instruct --out ./adapters/v1

This is intentionally minimal. The data-quality problem is unsolved by
this script — it just trains on whatever passes the filter. Garbage in,
the adapter learns garbage. Tune the filter, not the optimizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from learner.filter import eval_passing, filter_traces, min_quality
from learner.traces import Trace, TraceLogger


def traces_to_sft_dataset(traces: list[Trace]) -> list[dict]:
    """Convert traces to HuggingFace SFT chat-format records.

    Strips tool_use / tool_result / thinking blocks for the first
    iteration — text-only SFT. This is a known simplification: agentic
    behavior is in the tool calls, and dropping them means the adapter
    won't learn to use tools well. Coming back to this when the basic
    loop is working.
    """
    rows: list[dict] = []
    for t in traces:
        msgs: list[dict] = []
        for m in t.messages:
            content = m.get("content")
            if isinstance(content, str):
                msgs.append({"role": m["role"], "content": content})
            elif isinstance(content, list):
                # Concatenate text blocks; drop tool_use, tool_result, thinking.
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                joined = "\n".join(p for p in text_parts if p)
                if joined:
                    msgs.append({"role": m["role"], "content": joined})
        if len(msgs) >= 2:
            rows.append({"messages": msgs})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT on agent traces")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct", help="Base model on HuggingFace")
    parser.add_argument("--traces", default=None, help="Path to traces.jsonl (default ~/.agi/traces.jsonl)")
    parser.add_argument("--out", default="./adapters/latest", help="Output directory for the LoRA adapter")
    parser.add_argument("--filter", choices=["eval", "quality", "all"], default="eval",
                        help="Which trace filter to apply (eval=pass-only, quality=score>=0.7, all=no filter)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"error: missing learner dependencies — install with `pip install -e \".[learner]\"`", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        return 2

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("warning: no CUDA or MPS device — training will be unusably slow on CPU", file=sys.stderr)

    logger = TraceLogger(path=args.traces)
    all_traces = logger.all()
    print(f"loaded {len(all_traces)} traces from {logger.path}")

    if args.filter == "eval":
        traces = filter_traces(all_traces, eval_passing)
    elif args.filter == "quality":
        traces = filter_traces(all_traces, min_quality(0.7))
    else:
        traces = all_traces
    print(f"after filter ({args.filter}): {len(traces)} traces")

    rows = traces_to_sft_dataset(traces)
    print(f"converted to {len(rows)} SFT rows")
    if not rows:
        print("error: no training data after filter — collect more traces or relax filter", file=sys.stderr)
        return 1

    dataset = Dataset.from_list(rows)
    print(f"dataset: {dataset}")

    print(f"loading base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"saved adapter to {out_path}")

    # Drop a small manifest so the loader knows what base this adapter trained on.
    manifest = {"base_model": args.base, "n_traces": len(traces), "n_rows": len(rows)}
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
