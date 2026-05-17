"""Inducer demo — Levin universal search as a runtime primitive.

Five scenarios show the runtime use cases a coordination engine
actually drives the inducer for:

  1. Single-input arithmetic induction — n -> n²; the canonical
     three-instruction program (INP DUP MUL) is found by exhaustive
     length-1..L enumeration, with universal-prior mass and Levin
     complexity bounds reported.

  2. Two-input induction — (a, b) -> a + b; demonstrates how INP can
     read multiple values.

  3. Levin universal search — the same square problem run under the
     true Levin scheduler (each phase doubles the cumulative compute
     budget T; a program of length L gets ⌊T / 2^L⌋ steps).  This is
     Levin's dovetail in action.

  4. Posterior-over-programs — collect every consistent program in
     the search and combine their Kraft weights into a normalised
     Solomonoff posterior — the operation a coordination engine
     performs when it needs a model average rather than a single
     program.

  5. Composition: induction + held-out PAC bound — fit a program on a
     training spec, compute the Blumer-Ehrenfeucht-Haussler-Warmuth
     PAC bound on its generalisation error, and verify the bound by
     evaluating on a held-out spec.

Run::

    python examples/inducer_demo.py
"""
from __future__ import annotations

from agi.inducer import (
    ALPHABET_FULL,
    ALPHABET_STRAIGHT,
    Inducer,
    InducerConfig,
    Spec,
    induce,
    kraft_normalised_posterior,
    levin_runtime_bound,
)


# ----------------------------------------------------------------
# Scenario 1 — single-input arithmetic induction
# ----------------------------------------------------------------


def scenario_square() -> None:
    print("=" * 72)
    print("Scenario 1 — single-input arithmetic induction (n -> n²)")
    print("=" * 72)
    spec = Spec.from_pairs(
        [(i, i * i) for i in range(1, 8)],
        name="square",
    )
    cfg = InducerConfig(max_program_length=5, max_wallclock_s=5.0)
    rep = Inducer(cfg).search(spec)

    print(f"Spec               : {len(spec.examples)} examples, name={spec.name!r}")
    print(f"Program found      : {rep.program.disassemble()}")
    print(f"Length             : {rep.program.length} opcodes")
    print(f"Universal prior 2^-L: {rep.universal_prior_mass():.6f}")
    print(f"Kt(spec) upper bnd : {rep.levin_complexity():.3f} bits")
    print(f"Occam PAC ε (δ=.05): {rep.occam_bound():.6f}")
    print(f"Programs visited   : {rep.stats.programs_visited:,}")
    print(f"VM steps executed  : {rep.stats.steps_executed:,}")
    print(f"Wallclock          : {rep.stats.walltime_s * 1000:.2f} ms")
    print(f"Certificate prefix : {rep.certificate[:16]}…")
    # Generalisation: evaluate on held-out inputs
    print("Held-out evaluation:")
    for n in [9, 11, 17, 20]:
        print(f"  prog({n}) = {rep.eval([n])}  (expected {n * n})")
    print()


# ----------------------------------------------------------------
# Scenario 2 — two-input induction
# ----------------------------------------------------------------


def scenario_two_input_addition() -> None:
    print("=" * 72)
    print("Scenario 2 — two-input induction ((a, b) -> a + b)")
    print("=" * 72)
    pairs = [((1, 2), 3), ((3, 4), 7), ((10, 20), 30), ((5, 5), 10)]
    spec = Spec.from_pairs(pairs, name="add")
    cfg = InducerConfig(max_program_length=4, max_wallclock_s=5.0)
    rep = Inducer(cfg).search(spec)
    print(f"Program found    : {rep.program.disassemble()}")
    print(f"Universal prior  : {rep.universal_prior_mass():.6f}")
    print(f"Programs visited : {rep.stats.programs_visited:,}")
    print("Held-out evaluation:")
    for a, b in [(7, 8), (100, 200), (-3, 5)]:
        print(f"  prog({a}, {b}) = {rep.eval((a, b))}  (expected {a + b})")
    print()


# ----------------------------------------------------------------
# Scenario 3 — Levin universal search
# ----------------------------------------------------------------


def scenario_levin_universal_search() -> None:
    print("=" * 72)
    print("Scenario 3 — Levin universal search dovetail")
    print("=" * 72)
    spec = Spec.from_pairs([(i, i * i) for i in range(2, 8)], name="square-levin")
    cfg = InducerConfig(
        mode="levin",
        alphabet=ALPHABET_FULL,
        max_program_length=4,
        max_wallclock_s=10.0,
        levin_start_budget=64,
        levin_phase_doubling=2.0,
    )
    rep = Inducer(cfg).search(spec)
    print(f"Program           : {rep.program.disassemble()}")
    print(f"Phases completed  : {rep.stats.phases_completed}")
    print(f"Steps executed    : {rep.stats.steps_executed:,}")
    print(f"Wallclock         : {rep.stats.walltime_s * 1000:.2f} ms")
    print("Levin runtime bound (R ≤ K_U · 2^L · t):")
    for L, T in [(3, 4), (4, 8), (5, 16)]:
        print(f"  L={L}, t={T:>2} → R ≤ {levin_runtime_bound(L, T):,.0f} · K_U")
    print()


# ----------------------------------------------------------------
# Scenario 4 — posterior over programs (Solomonoff model average)
# ----------------------------------------------------------------


def scenario_posterior_over_programs() -> None:
    print("=" * 72)
    print("Scenario 4 — normalised Solomonoff posterior over consistent")
    print("           programs (model average for the coordination engine)")
    print("=" * 72)
    # Collect every program of length up to 4 that satisfies the spec.
    spec = Spec.from_pairs([(1, 1), (2, 2), (3, 3)], name="identity")
    cfg = InducerConfig(
        max_program_length=4,
        max_wallclock_s=5.0,
        early_stop=False,
        top_k=10,
        prune_constant_outputs=False,
    )
    rep = Inducer(cfg).search(spec)
    print(f"Top program       : {rep.program.disassemble()}")
    print(f"Consistent found  : {rep.stats.consistent_found}")
    print(f"Alternatives kept : {len(rep.alternatives)}")
    posterior = kraft_normalised_posterior(rep.alternatives)
    print("Normalised posterior weights:")
    for prog, w in posterior:
        print(f"  {prog.disassemble():<30}  weight={w:.3f}  (length {prog.length})")
    print()


# ----------------------------------------------------------------
# Scenario 5 — induction + held-out PAC bound
# ----------------------------------------------------------------


def scenario_pac_bound() -> None:
    print("=" * 72)
    print("Scenario 5 — induction + held-out PAC bound")
    print("=" * 72)
    train = Spec.from_pairs([(i, 2 * i) for i in range(1, 11)], name="double-train")
    holdout = [(i, 2 * i) for i in range(11, 51)]
    rep = Inducer(
        InducerConfig(max_program_length=4, max_wallclock_s=5.0)
    ).search(train)
    print(f"Program           : {rep.program.disassemble()}")
    print(f"Train size m      : {train.n_examples}")
    print(f"Occam PAC ε (δ=.05): {rep.occam_bound():.4f}")
    print(f"Occam PAC ε (δ=.01): {rep.occam_bound(delta=0.01):.4f}")
    # Empirical held-out error
    errors = sum(1 for x, y in holdout if rep.eval([x]) != y)
    print(f"Held-out error    : {errors}/{len(holdout)} = {errors/len(holdout):.4f}")
    print()


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------


def main() -> None:
    print("\nInducer — Levin Universal Search demo")
    print("(stdlib-only, deterministic, bit-for-bit reproducible)\n")
    scenario_square()
    scenario_two_input_addition()
    scenario_levin_universal_search()
    scenario_posterior_over_programs()
    scenario_pac_bound()
    print("All scenarios completed.")


if __name__ == "__main__":
    main()
