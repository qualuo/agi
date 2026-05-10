"""Synthetic labeled data generators.

Real traces accumulate slowly; the critic needs data on day one. Generators
here use the `Goal` abstraction to produce (prompt, response, label) triples
where the label is objective (we know the right answer because we generated
the problem).

Each generator emits both positive examples (correct responses) and several
flavors of negative examples (wrong number, off-by-one, hedging, garbage).
The mix matters — too many easy negatives and the critic learns "is this an
integer" instead of "does this match the prompt." Tune the mix for the
critic to be useful, not just accurate.
"""
from __future__ import annotations

import random

from learner.goals import Addition, Goal


def addition_examples(
    n: int = 1000,
    *,
    max_n: int = 99,
    seed: int = 42,
    pos_frac: float = 0.5,
) -> list[tuple[str, str, int]]:
    """Generate (prompt, response, label) triples from the Addition goal.

    Negative examples are a mix of:
    - Wrong number (random or off-by-one)
    - Hedging text ("I don't know", "let me think")
    - Garbage characters
    - Right number but with extra junk after
    """
    goal = Addition(max_n=max_n, train_seed=seed, eval_seed=seed + 1)
    rng = random.Random(seed + 2)
    out: list[tuple[str, str, int]] = []
    for _ in range(n):
        ex = goal.sample("train")
        # answer field includes the END terminator
        correct = ex.answer.rstrip("\n")
        if rng.random() < pos_frac:
            response = correct
            label = 1
        else:
            kind = rng.choice(["wrong_random", "off_by_one", "hedging", "garbage", "extra_junk"])
            if kind == "wrong_random":
                response = str(rng.randint(0, max_n * 2))
                # avoid accidentally producing the correct answer
                while response == correct:
                    response = str(rng.randint(0, max_n * 2))
            elif kind == "off_by_one":
                offset = rng.choice([-1, 1])
                response = str(int(correct) + offset)
            elif kind == "hedging":
                response = rng.choice([
                    "I don't know",
                    "let me think about it",
                    "the answer is unclear",
                    "I cannot compute this",
                    "?",
                ])
            elif kind == "garbage":
                response = "".join(rng.choices("abcdefxyz0!@#", k=rng.randint(1, 6)))
            elif kind == "extra_junk":
                response = f"{correct} extra noise here"
            label = 0
        out.append((ex.prompt, response, label))
    return out


GENERATORS: dict[str, callable] = {
    "addition": addition_examples,
}
