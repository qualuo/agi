"""CLI: train the trace-quality critic.

Default: trains on synthetic addition examples, reports accuracy on a held-out
split, saves the model. Real-trace training comes when traces accumulate.

    python -m learner.train_critic --n-train 2000 --n-eval 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make learner/ importable when running as `python -m learner.train_critic`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learner.critic import Critic, CriticConfig
from learner.synth import GENERATORS


def main() -> int:
    p = argparse.ArgumentParser(description="Train the trace-quality critic.")
    p.add_argument("--data", choices=list(GENERATORS), default="addition", help="Synthetic generator")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="./critic.pt", help="Where to save the trained model")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    gen = GENERATORS[args.data]
    train_data = gen(n=args.n_train, seed=args.seed)
    eval_data = gen(n=args.n_eval, seed=args.seed + 10_000)

    n_pos_train = sum(y for _, _, y in train_data)
    n_pos_eval = sum(y for _, _, y in eval_data)
    print(f"train: {len(train_data)} ({n_pos_train} pos / {len(train_data) - n_pos_train} neg)")
    print(f"eval:  {len(eval_data)} ({n_pos_eval} pos / {len(eval_data) - n_pos_eval} neg)")
    print()

    critic = Critic(CriticConfig())
    print(f"params: {sum(p.numel() for p in critic.model.parameters()):,}")
    print()

    critic.fit(
        train_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=True,
    )

    print()
    metrics = critic.evaluate(eval_data)
    print(f"eval: acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} (n={metrics['n']})")

    critic.save(args.out)
    print(f"\nsaved → {args.out}")

    print("\nspot-check predictions:")
    samples = [
        ("12+5=", "17", "correct"),
        ("12+5=", "18", "off-by-one"),
        ("12+5=", "I don't know", "hedge"),
        ("12+5=", "asdf", "garbage"),
        ("99+99=", "198", "correct hard"),
        ("99+99=", "200", "wrong hard"),
    ]
    for prompt, response, label in samples:
        prob = critic.predict_proba(prompt, response)
        print(f"  {prompt!r:14} → {response!r:20} {label:14}  P(passed)={prob:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
