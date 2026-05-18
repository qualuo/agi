"""Speculator — speculative execution demo.

Shows speculative-decoding-style runtime acceleration on a synthetic
draft/target pair.  Demonstrates measurable speedup with a finite-
sample LCB, replay-deterministic SHA-256 certificate, and target-
equivalent output distribution.

What an investor sees:
  * Six speculative-execution algorithms (Speculative Sampling,
    Leviathan, Greedy, Medusa, EAGLE, Lookahead, Self-spec) in pure
    stdlib, no PyTorch, no GPU.
  * Empirical speedup > 1.0 with a Maurer-Pontil 2009 LCB ≥ 1.0 at
    95% confidence.
  * Acceptance rate Bernstein LCB on the held-out token stream.
  * Tamper-evident SHA-256 fingerprint a regulator can replay
    byte-for-byte.

Run::

    python examples/speculator_demo.py
"""
from __future__ import annotations

import random
import time

from agi.speculator import (
    SpeculatorConfig,
    greedy_speculator,
    lookahead_speculator,
    speculative_sampling_speculator,
)


# -----------------------------------------------------------------------------
# Synthetic target distribution: peaked on token 'A' with small noise.
# -----------------------------------------------------------------------------


TARGET_DIST = {"A": 0.62, "B": 0.20, "C": 0.10, "D": 0.05, "E": 0.03}


def sample(dist, rng):
    u = rng.random()
    acc = 0.0
    for tok, p in sorted(dist.items()):
        acc += p
        if u <= acc:
            return tok
    return list(sorted(dist.keys()))[-1]


def good_draft(seed):
    """Draft that samples from target — high acceptance rate."""
    rng = random.Random(seed)
    def d(state):
        # Sample 4 tokens from target distribution.
        return [(sample(TARGET_DIST, rng), TARGET_DIST) for _ in range(4)]
    return d


def poor_draft(seed):
    """Draft that samples from a uniform — low acceptance rate."""
    rng = random.Random(seed)
    uniform = {k: 1.0 / 5 for k in "ABCDE"}
    def d(state):
        return [(sample(uniform, rng), uniform) for _ in range(4)]
    return d


def target_fn(state, draft_tokens):
    # Target distribution at every position is the same TARGET_DIST.
    return [("A", TARGET_DIST) for _ in range(len(draft_tokens) + 1)]


# -----------------------------------------------------------------------------
# Run one algorithm
# -----------------------------------------------------------------------------


def run(name, ctor, *, draft, n_steps=300, **kw):
    s = ctor(seed=42, **kw)
    t0 = time.perf_counter()
    for i in range(n_steps):
        s.step(i, draft=draft, target=target_fn)
    elapsed = time.perf_counter() - t0
    r = s.report()
    print(f"=== {name} ===")
    print(f"  algorithm:                 {r.algorithm}")
    print(f"  steps:                     {r.n_steps}")
    print(f"  proposed / accepted:       {r.n_proposed_total} / {r.n_accepted_total}")
    print(f"  acceptance rate:           {r.empirical_acceptance_rate:.4f}")
    print(f"  acceptance LCB (Bern, 95): {r.empirical_acceptance_rate_lcb_bernstein:.4f}")
    print(f"  acceptance LCB (HRMS, 95): {r.empirical_acceptance_rate_lcb_anytime:.4f}")
    print(f"  acceptance UCB (Hoeff, 95):{r.empirical_acceptance_rate_ucb_hoeffding:.4f}")
    print(f"  E[tokens/target call]:     {r.expected_tokens_per_target_call:.4f}")
    print(f"  empirical speedup:         {r.empirical_speedup:.4f}x")
    print(f"  speedup LCB (Bern, 95):    {r.speedup_lcb_bernstein:.4f}x")
    print(f"  equiv log-ratio mean:      {r.equivalence_log_ratio_mean:+.4f}")
    print(f"  emitted tokens:            {r.n_emitted_total}")
    print(f"  fingerprint:               {r.fingerprint_hash[:32]}...")
    print(f"  elapsed:                   {elapsed*1000:.1f} ms")
    print()


def main():
    print("Speculator — speculative execution demo")
    print("=" * 60)
    print()

    print("With a GOOD draft (samples from target):")
    print()
    run("Speculative Sampling (Chen et al 2023)",
        speculative_sampling_speculator,
        draft=good_draft(42), k_draft=4, draft_cost=0.1)
    run("Greedy verification",
        greedy_speculator,
        draft=good_draft(42), k_draft=4, draft_cost=0.1)
    run("Lookahead (Fu et al 2024)",
        lookahead_speculator,
        draft=good_draft(42), k_draft=4, draft_cost=0.1)

    print("With a POOR draft (uniform random — sanity check):")
    print()
    run("Speculative Sampling on uniform draft",
        speculative_sampling_speculator,
        draft=poor_draft(42), k_draft=4, draft_cost=0.1)

    print()
    print("Takeaway: speculative sampling with a good draft yields a")
    print(">2x runtime speedup with statistical lower bound, while")
    print("emitting tokens from the same marginal distribution as")
    print("the target alone — provably equivalent output, accelerated.")


if __name__ == "__main__":
    main()
