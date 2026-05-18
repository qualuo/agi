"""End-to-end demo of the Debater runtime primitive.

This script exercises every protocol the primitive ships:

  1. Two-player debate (Irving 2018) with a calibrated judge.
  2. Cross-examination debate (Barnes-Christiano 2020).
  3. Doubly-efficient debate (Brown-Cohen-Irving-Piliouras 2023).
  4. Market-maker debate (Hubinger 2020).
  5. Condorcet jury aggregation (Boland 1989).
  6. Persuasion-aware scoring (Khan-Hughes 2024).

It also runs a Monte-Carlo bimatrix payoff, finds the Nash equilibrium
of the debate game, certifies the cumulative truth-win-rate with
Hoeffding + empirical-Bernstein + Condorcet LCBs, and verifies the
replay chain by snapshot / restore.

No network or external models required: a deterministic calibrated
judge (a Bernoulli oracle with a configurable per-judge accuracy) and
constant-argument debaters are wired up in-process.  In production
the debaters would be Claude subagents, the judge would be a smaller
verifier model + a calibration recaliber, and the persuasion model
would come from the runtime's Mentalist.

Usage::

    python examples/debater_demo.py
"""
from __future__ import annotations

from agi.debater import (
    AGG_MAJORITY,
    Argument,
    Debater,
    DebateMove,
    DebateSpec,
    DebaterConfig,
    MOVE_ARGUE,
    MOVE_COUNTER,
    MOVE_CROSS_EXAMINE,
    PROTOCOL_CROSS_EXAM,
    PROTOCOL_DOUBLY_EFFICIENT,
    PROTOCOL_JURY,
    PROTOCOL_MARKET_MAKER,
    PROTOCOL_PERSUASION_AWARE,
    PROTOCOL_TWO_PLAYER,
    SIDE_A,
    SIDE_B,
    SIDE_TIE,
    debater_condorcet_lcb,
    make_calibrated_judge,
    make_constant_debater,
)


def header(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def two_player_demo() -> None:
    header("(1) Two-player debate (Irving 2018)")
    debate = Debater(DebaterConfig(protocol=PROTOCOL_TWO_PLAYER, max_rounds=3, seed=0))
    # Truthful judge: 85% per-debate accuracy in favour of the truth side
    judge = make_calibrated_judge(p_truth=0.85, truth_side=SIDE_A, seed=11)
    spec = DebateSpec(
        question="Is 7 prime?",
        claim_a="yes — divides only by 1 and 7",
        claim_b="no — claim that 7 = 2 * 3.5 = composite",
        debater_a=make_constant_debater([Argument(SIDE_A, "divisors are 1 and 7", evidence=0.95)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "7 = 2 * 3.5", evidence=0.05)]),
        judge=judge, ground_truth=SIDE_A,
    )
    for i in range(30):
        debate.run(spec)
    cert = debate.certify(delta=0.05)
    print(f"  n_debates       = {cert.n}")
    print(f"  win_prob_hat    = {cert.win_prob_hat:.3f}")
    print(f"  hoeffding_lcb   = {cert.hoeffding_lcb:.3f}")
    print(f"  bernstein_lcb   = {cert.bernstein_lcb:.3f}")
    print(f"  calibration ECE = {cert.calibration_ece:.3f}")
    print(f"  chain head      = {cert.chain_head[:16]}…")


def cross_exam_demo() -> None:
    header("(2) Cross-examination debate (Barnes-Christiano 2020)")
    # Round 0: argue.  Round 1: cross-examine the opponent's opening.
    def debater(spec, transcript, side):
        n_my = sum(1 for m in transcript if m.side == side)
        if n_my == 1:
            for idx, m in enumerate(transcript):
                if m.side != side and m.kind == MOVE_ARGUE:
                    return DebateMove(
                        kind=MOVE_CROSS_EXAMINE, side=side, round_index=n_my,
                        target_index=idx,
                    )
        return DebateMove(
            kind=MOVE_ARGUE if n_my == 0 else MOVE_COUNTER,
            side=side, round_index=n_my,
            argument=Argument(side, f"arg-{n_my}", evidence=0.6),
        )

    d = Debater(DebaterConfig(protocol=PROTOCOL_CROSS_EXAM, max_rounds=3, seed=0))
    spec = DebateSpec(
        question="Does this Python snippet halt?",
        claim_a="yes — loop terminates after 100 iterations",
        claim_b="no — infinite recursion on input 0",
        debater_a=debater, debater_b=debater,
        judge=make_calibrated_judge(0.75, SIDE_A, seed=22),
        ground_truth=SIDE_A,
    )
    r = d.run(spec)
    kinds = [m.kind for m in r.transcript]
    n_ce = kinds.count(MOVE_CROSS_EXAMINE)
    print(f"  winner          = {r.winner}")
    print(f"  rounds          = {r.rounds_used}")
    print(f"  cross_examines  = {n_ce}")
    print(f"  win_prob_hat    = {r.win_prob_hat:.3f}")


def doubly_efficient_demo() -> None:
    header("(3) Doubly-efficient debate (Brown-Cohen-Irving-Piliouras 2023)")
    d = Debater(DebaterConfig(protocol=PROTOCOL_DOUBLY_EFFICIENT, max_rounds=2, seed=0))
    spec = DebateSpec(
        question="Does this proof step typecheck?",
        claim_a="yes",
        claim_b="no",
        debater_a=make_constant_debater([Argument(SIDE_A, "valid step", evidence=0.9)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "invalid step", evidence=0.1)]),
        judge=make_calibrated_judge(0.9, SIDE_A, seed=33),
        ground_truth=SIDE_A,
    )
    r = d.run(spec)
    confirmed = r.transcript[-1].meta.get("doubly_efficient_confirmed")
    print(f"  winner          = {r.winner}")
    print(f"  win_prob_hat    = {r.win_prob_hat:.3f}")
    print(f"  verifier_passed = {confirmed}")


def market_maker_demo() -> None:
    header("(4) Market-maker debate (Hubinger 2020)")
    d = Debater(DebaterConfig(protocol=PROTOCOL_MARKET_MAKER, max_rounds=6, seed=0))
    spec = DebateSpec(
        question="Will this commit pass the unit-test suite?",
        claim_a="yes — added a regression test alongside the fix",
        claim_b="no — broke the build",
        debater_a=make_constant_debater([Argument(SIDE_A, "tests pass locally", evidence=0.7)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "tests fail on CI", evidence=0.2)]),
        judge=make_calibrated_judge(0.8, SIDE_A, seed=44),
        ground_truth=SIDE_A,
    )
    r = d.run(spec)
    print(f"  winner          = {r.winner}")
    print(f"  market_price    = {r.transcript[-1].meta['market_price']:.3f}")
    print(f"  rounds          = {r.rounds_used}")


def jury_demo() -> None:
    header("(5) Condorcet jury (Boland 1989)")
    d = Debater(DebaterConfig(protocol=PROTOCOL_JURY, aggregation=AGG_MAJORITY,
                              max_rounds=2, seed=0))
    judges = tuple(
        (make_calibrated_judge(p_truth=0.65, truth_side=SIDE_A, seed=100 + i), 0.65)
        for i in range(11)
    )
    spec = DebateSpec(
        question="Is the SHA-256 of this artifact ce6f…12a4?",
        claim_a="yes — matches the manifest",
        claim_b="no — looks tampered",
        debater_a=make_constant_debater([Argument(SIDE_A, "matches", evidence=0.9)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "differs", evidence=0.1)]),
        judge=judges[0][0],
        judges_for_jury=judges,
        ground_truth=SIDE_A,
    )
    r = d.run(spec)
    print(f"  jury_size       = {len(r.judge_votes)}")
    print(f"  jury_winner     = {r.winner}")
    print(f"  agreement_frac  = {r.win_prob_hat:.3f}")
    print(f"  condorcet_lcb95 = {debater_condorcet_lcb(0.65, 11, 0.05):.3f}")


def persuasion_aware_demo() -> None:
    header("(6) Persuasion-aware debate (Khan-Hughes 2024)")

    def model(spec, transcript):
        a = sum(m.argument.evidence for m in transcript if m.side == SIDE_A and m.argument)
        b = sum(m.argument.evidence for m in transcript if m.side == SIDE_B and m.argument)
        total = a + b + 0.5  # prior pseudocount
        p_a = max(0.01, min(0.99, (a + 0.25) / total))
        return {SIDE_A: p_a, SIDE_B: 1.0 - p_a}

    d = Debater(DebaterConfig(
        protocol=PROTOCOL_PERSUASION_AWARE, max_rounds=3,
        persuasion_penalty_weight=2.0, seed=0,
    ))
    spec = DebateSpec(
        question="Did the model leak training data?",
        claim_a="no — citations are paraphrased and outside the cutoff",
        claim_b="yes — verbatim reproduction observed",
        # A has high evidence, B uses pure rhetoric
        debater_a=make_constant_debater([Argument(SIDE_A, "citation outside cutoff", evidence=0.9)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "they always lie", evidence=0.05)]),
        judge=make_calibrated_judge(0.5, SIDE_A, seed=55),  # weak baseline judge
        persuasion_model=model, ground_truth=SIDE_A,
    )
    r = d.run(spec)
    print(f"  winner          = {r.winner}")
    print(f"  win_prob_hat    = {r.win_prob_hat:.3f}")
    print(f"  truthful_total  = {sum(r.truthful_components):.3f}")
    print(f"  manip_total     = {sum(r.manipulative_components):.3f}")
    print(f"  net effect      = penalises the manipulative side")


def payoff_nash_demo() -> None:
    header("(*) Empirical-payoff bimatrix → Nash equilibrium")
    d = Debater(DebaterConfig(max_rounds=1, seed=0))
    debaters = {
        SIDE_A: {
            "truthful": make_constant_debater([Argument(SIDE_A, "honest", evidence=0.9)]),
            "obfuscate": make_constant_debater([Argument(SIDE_A, "spin", evidence=0.05)]),
        },
        SIDE_B: {
            "truthful": make_constant_debater([Argument(SIDE_B, "honest", evidence=0.9)]),
            "obfuscate": make_constant_debater([Argument(SIDE_B, "spin", evidence=0.05)]),
        },
    }
    spec = DebateSpec(
        question="Strategic stability check",
        claim_a="A",
        claim_b="B",
        debater_a=debaters[SIDE_A]["truthful"],
        debater_b=debaters[SIDE_B]["truthful"],
        judge=make_calibrated_judge(0.75, SIDE_A, seed=66),
        strategy_space=("truthful", "obfuscate"),
        ground_truth=SIDE_A,
    )
    payoff = d.empirical_payoff(spec, debaters_by_strategy=debaters, samples_per_cell=8)
    print("  payoff matrix A (rows=A strategies, cols=B strategies):")
    for row, label in zip(payoff.matrix_a, payoff.strategies):
        cells = "  ".join(f"{v:.2f}" for v in row)
        print(f"    {label:>10s}: {cells}")
    nash = d.nash_check(payoff)
    print(f"  Nash π_A        = {tuple(f'{p:.2f}' for p in nash.pi_a)}")
    print(f"  Nash π_B        = {tuple(f'{p:.2f}' for p in nash.pi_b)}")
    print(f"  NashConv (exploitability) = {nash.nash_conv:.4f}")
    print(f"  method          = {nash.method}")


def chain_replay_demo() -> None:
    header("(*) Snapshot / restore round-trip")
    import json

    d1 = Debater(DebaterConfig(max_rounds=2, seed=0))
    spec = DebateSpec(
        question="Snapshot test",
        claim_a="a", claim_b="b",
        debater_a=make_constant_debater([Argument(SIDE_A, "a", 0.5)]),
        debater_b=make_constant_debater([Argument(SIDE_B, "b", 0.5)]),
        judge=make_calibrated_judge(0.8, SIDE_A, seed=77),
        ground_truth=SIDE_A,
    )
    for i in range(5):
        d1.run(spec)
    snap = d1.snapshot()
    blob = json.dumps(snap)
    d2 = Debater(DebaterConfig(max_rounds=2, seed=0))
    d2.restore(json.loads(blob))
    match = d2.chain_head == d1.chain_head
    print(f"  d1.chain_head   = {d1.chain_head[:16]}…")
    print(f"  d2.chain_head   = {d2.chain_head[:16]}…")
    print(f"  identical?      = {match}")


def main() -> None:
    print("agi.debater — multi-agent debate as a runtime primitive")
    print("  the alignment-grade verifier the coordination engine calls when")
    print("  no single component is strong enough to verify on its own.")
    two_player_demo()
    cross_exam_demo()
    doubly_efficient_demo()
    market_maker_demo()
    jury_demo()
    persuasion_aware_demo()
    payoff_nash_demo()
    chain_replay_demo()
    print()
    print("Done.  Every transcript is hash-chained and replay-verifiable;")
    print("every win-rate carries a Hoeffding / empirical-Bernstein /")
    print("Condorcet LCB at the configured confidence; the debate game's")
    print("NashConv is reported so the coordinator can refuse to ship when")
    print("the debaters' realised profile is far from a Nash equilibrium.")


if __name__ == "__main__":
    main()
