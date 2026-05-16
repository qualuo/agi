"""Predictor demo — universal sequence prediction via Context Tree
Weighting (CTW), as a runtime primitive.

Four scenarios demonstrate the four runtime use cases a coordination
engine actually drives the predictor for:

  1. Universal compressor — feed a structured binary stream; measure
     code length and compare to the naive ``n`` bits and the optimum.

  2. Entropy-rate estimation — watch the predictor converge to the
     true entropy rate of an unknown stationary source.

  3. Regime-change detection — switching CTW tracks a non-stationary
     source across a changepoint.

  4. Anytime-valid hypothesis test — the e-process rejects "uniform
     i.i.d." with strong evidence on structured data while remaining
     conservative on truly random data.

Run::

    python examples/predictor_demo.py
"""
from __future__ import annotations

import math
import random

from agi.predictor import Predictor, compress_binary_sequence


def scenario_compressor() -> None:
    print("=" * 66)
    print("1. Universal compressor on structured binary streams")
    print("=" * 66)
    streams = {
        "all-zeros (n=400)":          [0] * 400,
        "alternating 01 (n=400)":     [0, 1] * 200,
        "period-4 0011 (n=400)":      [0, 0, 1, 1] * 100,
        "Markov sticky (n=400)":      None,
        "uniform random (n=400)":     None,
    }
    rng = random.Random(0)
    markov = [0]
    for _ in range(399):
        markov.append(markov[-1] if rng.random() < 0.95 else 1 - markov[-1])
    streams["Markov sticky (n=400)"] = markov

    rng = random.Random(7)
    streams["uniform random (n=400)"] = [rng.randint(0, 1) for _ in range(400)]

    for name, seq in streams.items():
        bits, p = compress_binary_sequence(seq, depth=8)
        bits_per_sym = bits / len(seq)
        # Naive code: n bits.
        naive = float(len(seq))
        ratio = bits / naive
        print(
            f"  {name:<28}  CTW = {bits:>7.2f} bits "
            f"({bits_per_sym:.3f} bits/sym, {ratio:.2%} of naive)"
        )


def scenario_entropy_rate() -> None:
    print()
    print("=" * 66)
    print("2. Entropy-rate estimation on biased sources")
    print("=" * 66)
    for p_true in (0.5, 0.7, 0.9, 0.99):
        true_h = (
            0.0
            if p_true in (0.0, 1.0)
            else -p_true * math.log2(p_true) - (1 - p_true) * math.log2(1 - p_true)
        )
        rng = random.Random(int(p_true * 1000))
        seq = [1 if rng.random() < p_true else 0 for _ in range(5000)]
        pred = Predictor.create(alphabet_size=2, depth=4)
        pred.observe_many(seq)
        est = pred.entropy_rate_estimate().average_log_loss_bits_per_symbol
        print(
            f"  P(1) = {p_true:.2f}   true H = {true_h:.4f} bits/sym   "
            f"CTW H = {est:.4f} bits/sym   gap = {est - true_h:+.4f}"
        )


def scenario_regime_change() -> None:
    print()
    print("=" * 66)
    print("3. Regime-change detection — switching CTW vs vanilla CTW")
    print("=" * 66)
    rng = random.Random(2026)
    # Regime A: 300 samples of mostly 0s.  Regime B: 300 samples of mostly 1s.
    seq = []
    for _ in range(300):
        seq.append(0 if rng.random() < 0.95 else 1)
    for _ in range(300):
        seq.append(1 if rng.random() < 0.95 else 0)

    vanilla = Predictor.create(alphabet_size=2, depth=4)
    switch = Predictor.create(alphabet_size=2, depth=4, switching_rate=0.005)
    vanilla.observe_many(seq)
    switch.observe_many(seq)
    print(
        "  Vanilla CTW : code = {:>7.2f} bits   final P(1) = {:.3f}".format(
            vanilla.code_length_bits(), vanilla.predict().probs[1]
        )
    )
    print(
        "  Switching   : code = {:>7.2f} bits   final P(1) = {:.3f}".format(
            switch.code_length_bits(), switch.predict().probs[1]
        )
    )
    print(
        "  Truth: regime B (just-seen) is 95% ones, so P(1) ≈ 0.95 is correct."
    )


def scenario_e_process() -> None:
    print()
    print("=" * 66)
    print("4. Anytime-valid e-process for H_0: x_t iid uniform Bernoulli")
    print("=" * 66)
    rng = random.Random(31337)
    # Two sequences: uniform random, and a structured 0011 pattern.
    uniform = [rng.randint(0, 1) for _ in range(500)]
    structured = [0, 0, 1, 1] * 125

    for name, seq in [("uniform random", uniform), ("structured 0011", structured)]:
        p = Predictor.create(alphabet_size=2, depth=4)
        p.observe_many(seq)
        e = p.e_process_vs_uniform()
        verdict = (
            "REJECT H_0 at α = 1e-9"
            if e.p_value_upper_bound < 1e-9
            else "do not reject H_0"
        )
        print(
            f"  {name:<18}  e-value = {e.e_value:.3e}   "
            f"upper-p = {e.p_value_upper_bound:.3e}   {verdict}"
        )


def scenario_runtime_dispatch() -> None:
    print()
    print("=" * 66)
    print("5. Runtime dispatch: a coordination engine consumes Predictor events")
    print("=" * 66)
    from agi.events import EventBus

    bus = EventBus()
    seen = []
    bus.subscribe(lambda e: seen.append(e.kind) if e.kind.startswith("predictor.") else None)
    pred = Predictor.create(alphabet_size=2, depth=4, bus=bus)
    rng = random.Random(0)
    for _ in range(20):
        pred.observe(rng.randint(0, 1))
    rep = pred.report()
    map_tree = pred.map_tree()
    print(f"  emitted events: {len(seen)} ({', '.join(sorted(set(seen)))})")
    print(f"  final fingerprint: {rep.fingerprint[:24]}…")
    print(f"  MAP tree leaves: {map_tree.n_leaves}")
    print(f"  code length: {rep.code_length_bits:.2f} bits over 20 symbols")
    print("  Coordination engine can hash-verify the trace and re-route on either.")


def main() -> None:
    scenario_compressor()
    scenario_entropy_rate()
    scenario_regime_change()
    scenario_e_process()
    scenario_runtime_dispatch()


if __name__ == "__main__":
    main()
