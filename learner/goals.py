"""Goal abstraction.

A Goal is a (problem distribution, scoring function, vocabulary) bundle. The
learning loop samples problems from it, the model tries to solve them, the
scorer grades the attempts, gradients update the model toward higher scores.

The same `Goal` interface works for arithmetic (today), modular arithmetic,
sequence reversal, parity, sorting (next iterations), and eventually text
tasks once the model is big enough.

Vocabulary lives on the goal because different goals need different alphabets.
The model is built around the goal's vocabulary.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Example:
    prompt: str       # everything the model sees at inference time
    answer: str       # what comes after the prompt (what we score)


class Goal(ABC):
    name: str
    vocab: list[str]
    max_seq_len: int  # prompt + answer + 1 (for end-of-answer terminator)

    @abstractmethod
    def sample(self, split: str = "train") -> Example:
        """Generate one (prompt, answer) example. `split` may be 'train' or 'eval'."""

    def score(self, answer: str, prediction: str) -> float:
        """Default: exact-match on stripped strings. Override for partial credit."""
        return 1.0 if prediction.strip() == answer.strip() else 0.0

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.vocab)}

    @property
    def itos(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.vocab)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


END = "\n"  # universal answer terminator across goals


class Addition(Goal):
    """Goal: given two non-negative integers, output their sum.

    Format: "a+b=" → "c\\n". Train and eval both sample uniformly from
    [0, max_n] but eval uses a separate RNG so train/eval are reproducible
    and disjoint by seed convention.
    """

    name = "addition"

    def __init__(self, max_n: int = 99, train_seed: int = 0, eval_seed: int = 1) -> None:
        self.max_n = max_n
        self.vocab = list("0123456789+= \n")
        # max prompt: "99+99=" = 6, max answer: "198\n" = 4, total 10
        digits = len(str(max_n + max_n))
        self.max_seq_len = (len(str(max_n)) * 2 + 2) + (digits + 1) + 1
        self._train_rng = random.Random(train_seed)
        self._eval_rng = random.Random(eval_seed)

    def sample(self, split: str = "train") -> Example:
        rng = self._train_rng if split == "train" else self._eval_rng
        a = rng.randint(0, self.max_n)
        b = rng.randint(0, self.max_n)
        prompt = f"{a}+{b}="
        answer = f"{a + b}{END}"
        return Example(prompt=prompt, answer=answer)
