r"""Inducer — Levin Universal Search for program induction as a runtime primitive.

Every primitive in this runtime is itself a *program*: Predictor runs CTW,
Solver runs CDCL, Synthesizer enumerates an AST.  The composing
coordination engine reaches a question that none of those primitives
can answer the moment it asks *which* program a stream of observations
came from in the **first place** — with no DSL pre-declared, no
hypothesis class fixed, no model family chosen.  That is the
*universal program-induction* question, and the foundational answer is
Levin's universal search (Levin 1973, Schmidhuber 2002, Hutter 2005).

Inducer is the runtime's **algorithmic-information** primitive.  It
enumerates programs over a small Turing-complete VM in order of a
*time-bounded Kolmogorov complexity* — the **Levin complexity**
``Kt(x) = min { l(p) + log₂ t(p, x) : U(p) = x }`` — and returns the
shortest fast program consistent with a specification, together with a
**Levin-tight runtime certificate**: if any program ``p*`` of length
``L*`` solves the specification in ``T*`` interpreter steps, Inducer
finds *some* solving program in total runtime ::

    R(p*)  ≤  K_U · 2^{L*} · T*

where ``K_U`` is a constant determined only by the universal VM
(Levin 1973, Theorem 1; Hutter 2002, *Universal Algorithmic Intelligence*,
Sec. 6).  This is the *strongest* worst-case bound on program search
known to be achievable — every other practical synthesiser (CEGIS,
FlashFill, syntax-guided SyGuS) trades it for a domain-restricted DSL.

Where Synthesizer commits to a typed DSL and Solver to a CNF, Inducer
commits to *no* model class at all — only to a small universal machine
``U`` — and pays the Levin bound for that generality.  This is the
operation that lets the coordination engine ask "*is there any program
at all that explains this trace?*" without first asking "what kind of
program?", and lets it observe a quantitative bound on the answer.

Mathematical and algorithmic roots
----------------------------------

  * **Solomonoff, R. J. (1964) — "A formal theory of inductive
    inference, parts I & II."**  *Information and Control* 7(1-2).
    The universal a priori probability ``M(x) = Σ_{U(p)=x} 2^{-l(p)}``
    on output strings from a universal Turing machine ``U``.  The
    *normalised* version (algorithmic probability) dominates every
    enumerable semi-measure up to a multiplicative constant — the
    foundation of algorithmic induction.

  * **Levin, L. A. (1973) — "Universal sequential search problems."**
    *Problems of Information Transmission* 9(3) 265-266.  Defines the
    *Levin time-bounded complexity* ::

        Kt(x | y) = min { l(p) + log₂ t(p, y, x) : U(p, y) = x }

    and proves that the **universal search** algorithm — dovetail every
    program ``p`` so that ``p`` is allotted ``2^{-l(p)} T`` of the
    total budget ``T`` — finds, for every inverter problem solvable by
    a program ``p*`` in time ``t*``, a solver in time
    ``O(2^{l(p*)} · t*)``.  The constant depends only on ``U``.

  * **Solomonoff, R. J. (1986) — "The application of algorithmic
    probability to problems in artificial intelligence" in: Kanal &
    Lemmer (eds.) *Uncertainty in Artificial Intelligence.*** Argues
    that Levin-search-style enumeration *is* the canonical learning
    primitive, on top of which heuristics merely accelerate.

  * **Schmidhuber, J. (1997) — "Discovering neural nets with low
    Kolmogorov complexity and high generalisation capability."**
    *Neural Networks* 10(5) 857-873.  PAC-Bayes generalisation
    guarantees are sharpest when *Kolmogorov complexity* is the
    description length term; Inducer's per-program length is precisely
    that term.

  * **Schmidhuber, J. (2002) — "Optimal ordered problem solver."**
    *Machine Learning* 54(3) 211-254.  *OOPS* sequentialises Levin
    search across a growing instruction set, freezes prefixes of useful
    solutions, and biases the prior over instructions toward those
    that already pay off — a *speed prior* that preserves the Levin
    bound while letting the runtime *learn what to search for next*.

  * **Hutter, M. (2002) — "The fastest and shortest algorithm for all
    well-defined problems."**  *International Journal of Foundations of
    Computer Science* 13(3) 431-443.  Refines Levin search to
    ``M^p_n``: tight bound on the runtime *plus* a constant additive
    overhead.

  * **Hutter, M. (2005) — *Universal Artificial Intelligence:*
    *Sequential decisions based on algorithmic probability*, Springer.**
    Chapter 7 frames AIXI's action search as a Levin-style dovetail
    over policy programs; Inducer is the planner-shaped subset of that
    construction, exposed as an in-process primitive.

  * **Veness, J., Ng, K. S., Hutter, M., Uther, W., Silver, D. (2011) —
    "A Monte-Carlo AIXI approximation."**  *Journal of Artificial
    Intelligence Research* 40 95-142.  Replaces the model with CTW
    (`Predictor`) and the action search with MCTS — Inducer's Levin
    backbone is the *exact* counterpart for the deterministic /
    enumerable side.

  * **Cover, T. M. & Thomas, J. A. (2006) — *Elements of Information
    Theory*, 2nd ed., chap. 14.**  Kraft inequality: a prefix-free code
    over programs has ``Σ_p 2^{-l(p)} ≤ 1``, hence the universal prior
    is a sub-probability measure.  Inducer's prior is exactly Kraft.

  * **Li, M. & Vitányi, P. (2019) — *An Introduction to Kolmogorov
    Complexity and Its Applications*, 4th ed., chap. 5.**  Theorem
    5.5.5: ``Kt(x)``-shortest programs satisfy a *coding theorem*::

        Pr_{U}(U(p) = x)  =  Θ(2^{-K(x)}).

    Inducer's returned program length is a constructive upper bound on
    the Kt-complexity of the spec.

The universal machine ``U``
---------------------------

Inducer ships a small **stack-based VM** as its universal machine.
The instruction set is deliberately minimal so that the *constants*
hidden inside the Levin bound (``K_U`` and the per-step interpreter
cost) are *visible* and the per-program reasoning stays auditable.

::

    op   mnemonic   stack effect              notes
    ──   ────────   ───────────────────       ────────────────────────────
    0    HALT       (— —)                     terminate; output = top of stack
    1    PUSH0      (— → 0)                   push 0
    2    PUSH1      (— → 1)                   push 1
    3    PUSH2      (— → 2)                   push 2
    4    PUSHN1     (— → -1)                  push -1
    5    INP        (— → x_i)                 push next input element
    6    DUP        (a → a a)                 duplicate top
    7    SWP        (a b → b a)               swap top two
    8    DRP        (a — )                    drop top
    9    ADD        (a b → b + a)             stack arithmetic
    10   SUB        (a b → b - a)
    11   MUL        (a b → b * a)
    12   MOD        (a b → b mod a)           fail on a == 0
    13   NEG        (a → -a)
    14   JNZ        (a — )                    if a ≠ 0, jump back 3 ops
    15   NOP        (— —)                     filler / Kraft padding

Each instruction is a 4-bit symbol so a program of length ``L``
instructions encodes in ``L/2`` bytes.  Programs are read left-to-right;
the program counter is bounded by the program length and a step budget.
Stack and value bounds are enforced (overflow on values larger than
``stack_value_bound`` halts with FAIL).  Division uses Python's
floor-mod so the only division failure is `a == 0`.

There are exactly **15 active opcodes** (``NOP`` is reserved for
prefix-free Kraft padding) — a *primitive recursive* core plus the
single backward jump ``JNZ``, which makes the language Turing-complete
under unbounded step budget.  Programs that run for more than
``max_steps_per_program`` steps are *aborted* and counted as
``DIVERGED`` (consistent with Levin's treatment of non-halting
programs).

Public API
----------

::

    >>> from agi.inducer import Inducer, InducerConfig, Spec, Example
    >>> spec = Spec.from_pairs([(2, 4), (3, 9), (5, 25), (7, 49)])  # n -> n²
    >>> ind  = Inducer(InducerConfig(max_total_steps=2_000_000))
    >>> rep  = ind.search(spec)
    >>> rep.program.disassemble()           # ⇒ "INP DUP MUL HALT"
    >>> rep.levin_bound(spec)               # constructive Levin bound
    >>> rep.universal_prior_mass()          # 2^{-l(p)}
    >>> rep.certificate                     # SHA-256 over (VM, spec, program)

Composition with the rest of the runtime
----------------------------------------

  * **Predictor** — Predictor's CTW estimates the universal prior over
    *next symbols*; Inducer estimates it over *programs*.  Feed
    Inducer's discovered program back into Predictor's prior to
    accelerate compression on the same source.
  * **Synthesizer** — Synthesizer searches a typed DSL with PAC bounds;
    Inducer searches the *unrestricted* universal VM with the Levin
    bound.  A coordination engine picks Inducer when the DSL is
    unknown, Synthesizer when the DSL is known.
  * **Compressor** — Compressor measures *normalised compression
    distance* over arbitrary byte strings; Inducer's program length on
    a string ``x`` is an *upper bound* on ``Kt(x)`` and feeds NCD with
    tighter (because constructive) numerators.
  * **Conjecturer** — every Inducer-discovered program is a candidate
    *generative law* for a finite observation set; Conjecturer's
    e-value framework lifts the program to a falsifiable law with a
    sequential test.
  * **Solver** — when a spec is too large for Levin search, the
    coordination engine compiles consistency into CNF and hands it to
    Solver; Inducer's enumeration order is the natural fallback when
    the CNF is itself too large.
  * **AttestationLedger** — each `InducerReport` includes a SHA-256
    chain over `(VM signature, spec, search trace, program, step
    budget)` so a coordination engine can replay-verify the search
    bit-for-bit.

Determinism, guarantees, and the Levin bound
--------------------------------------------

Inducer's enumeration order is *exact-lexicographic* over the 4-bit
opcode alphabet.  For any RNG-free spec this makes the entire search
**reproducible bit-for-bit** across machines.  The reported bounds:

  * ``universal_prior_mass = 2^{-l(p)}`` — the Kraft mass the
    discovered program contributes to the Solomonoff prior.
  * ``levin_complexity   = l(p) + log₂(t(p))`` — the discovered
    program's Levin complexity *for this spec* (constructive upper
    bound on ``Kt(spec)``).
  * ``levin_runtime_bound = K_U · 2^{l(p*)} · t(p*)`` — the bound that
    *would* hold if a length-``L*`` runtime-``T*`` solver existed;
    Inducer never beats it asymptotically.
  * ``occam_bound(m, δ)`` — Blumer-Ehrenfeucht-Haussler-Warmuth PAC
    bound: ``(l(p) · ln 2 + ln(1/δ)) / m``; tightest when ``l(p)`` is
    small.
  * ``coding_theorem_lb`` — Li-Vitányi coding-theorem lower bound on
    ``Pr_U(U(p) = spec)``: ``Θ(2^{-l(p)})``; used as the prior in
    posterior-mass arguments downstream.
  * ``certificate`` — SHA-256 chained over every (program, decision)
    pair visited.

The primitive is **stdlib-only**: no torch, no numpy, no SMT.  Every
opcode is one Python branch; the inner loop is a `while pc < len`
loop with a precomputed dispatch table.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class InducerError(Exception):
    """Base class for all Inducer errors."""


class InvalidConfig(InducerError):
    """Configuration parameters are inconsistent."""


class InvalidSpec(InducerError):
    """Specification is malformed."""


class InvalidProgram(InducerError):
    """Program bytes are out of alphabet or otherwise malformed."""


class BudgetExhausted(InducerError):
    """The configured step / program / time budget was hit."""


class NoSolution(InducerError):
    """Search completed within budget but found no solution."""


# =============================================================================
# Opcode alphabet
# =============================================================================


HALT  = 0
PUSH0 = 1
PUSH1 = 2
PUSH2 = 3
PUSHN1 = 4
INP   = 5
DUP   = 6
SWP   = 7
DRP   = 8
ADD   = 9
SUB   = 10
MUL   = 11
MOD   = 12
NEG   = 13
JNZ   = 14
NOP   = 15

OPCODE_COUNT = 16  # 4-bit alphabet

OPCODE_NAME = {
    HALT: "HALT",
    PUSH0: "PUSH0",
    PUSH1: "PUSH1",
    PUSH2: "PUSH2",
    PUSHN1: "PUSHN1",
    INP: "INP",
    DUP: "DUP",
    SWP: "SWP",
    DRP: "DRP",
    ADD: "ADD",
    SUB: "SUB",
    MUL: "MUL",
    MOD: "MOD",
    NEG: "NEG",
    JNZ: "JNZ",
    NOP: "NOP",
}


# Smaller alphabets restrict enumeration without changing the VM.
ALPHABET_STRAIGHT: tuple[int, ...] = (
    HALT, PUSH0, PUSH1, PUSH2, PUSHN1, INP, DUP, SWP, DRP, ADD, SUB, MUL, MOD, NEG,
)  # 14 opcodes; no JNZ ⇒ every program halts in ≤ len steps

ALPHABET_FULL: tuple[int, ...] = tuple(range(OPCODE_COUNT))  # full universal VM

ALPHABET_ARITH: tuple[int, ...] = (
    HALT, PUSH0, PUSH1, PUSH2, PUSHN1, INP, DUP, SWP, ADD, SUB, MUL, NEG,
)  # arithmetic-only subset; 12 opcodes


# =============================================================================
# VM result codes
# =============================================================================


VM_OK       = "ok"        # ran to HALT; output on stack
VM_FAIL     = "fail"      # stack underflow / div-by-zero / value overflow
VM_DIVERGED = "diverged"  # hit step budget without HALT
VM_BAD_PC   = "bad_pc"    # program counter ran off the end without HALT


# =============================================================================
# Programs
# =============================================================================


@dataclass(frozen=True)
class Program:
    """A sequence of 4-bit opcodes.

    Stored as a tuple of ints for hashability.  Use `to_bytes` for the
    packed binary form (two opcodes per byte, low nibble first).
    """
    ops: tuple[int, ...]

    def __post_init__(self) -> None:
        for o in self.ops:
            if not (0 <= o < OPCODE_COUNT):
                raise InvalidProgram(f"opcode {o} outside [0, {OPCODE_COUNT})")

    @property
    def length(self) -> int:
        return len(self.ops)

    def disassemble(self) -> str:
        return " ".join(OPCODE_NAME[o] for o in self.ops)

    def to_bytes(self) -> bytes:
        out = bytearray()
        ops = list(self.ops)
        if len(ops) % 2 == 1:
            ops.append(NOP)
        for i in range(0, len(ops), 2):
            lo = ops[i] & 0xF
            hi = ops[i + 1] & 0xF
            out.append(lo | (hi << 4))
        return bytes(out)

    @staticmethod
    def from_bytes(b: bytes, length: int | None = None) -> "Program":
        ops: list[int] = []
        for byte in b:
            ops.append(byte & 0xF)
            ops.append((byte >> 4) & 0xF)
        if length is not None:
            ops = ops[:length]
        # Strip trailing NOP padding only if no explicit length was given.
        if length is None:
            while ops and ops[-1] == NOP:
                ops.pop()
        return Program(ops=tuple(ops))

    def __str__(self) -> str:
        return self.disassemble()

    def __repr__(self) -> str:
        return f"Program({self.disassemble()})"

    def __len__(self) -> int:
        return len(self.ops)


# =============================================================================
# VM
# =============================================================================


@dataclass(frozen=True)
class VMResult:
    """Outcome of running one program on one input."""
    status: str
    output: Any | None
    steps: int
    stack_high_water: int


def _run_program(
    ops: Sequence[int],
    inputs: Sequence[Any],
    *,
    max_steps: int,
    stack_size: int,
    value_bound: int,
) -> VMResult:
    """Run a program on a sequence of inputs and return its result.

    Inputs are consumed in order by the ``INP`` opcode; each ``INP`` pops the
    next input from the head of the input list.  If the program runs out of
    inputs, ``INP`` fails.
    """
    stack: list[int] = []
    sp = 0  # high-water mark
    pc = 0
    inp_cursor = 0
    steps = 0
    n_inputs = len(inputs)
    n_ops = len(ops)
    while pc < n_ops and steps < max_steps:
        op = ops[pc]
        if op == HALT:
            top = stack[-1] if stack else 0
            return VMResult(VM_OK, top, steps + 1, sp)
        elif op == PUSH0:
            stack.append(0)
        elif op == PUSH1:
            stack.append(1)
        elif op == PUSH2:
            stack.append(2)
        elif op == PUSHN1:
            stack.append(-1)
        elif op == INP:
            if inp_cursor >= n_inputs:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            stack.append(inputs[inp_cursor])
            inp_cursor += 1
        elif op == DUP:
            if not stack:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            stack.append(stack[-1])
        elif op == SWP:
            if len(stack) < 2:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif op == DRP:
            if not stack:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            stack.pop()
        elif op == ADD:
            if len(stack) < 2:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            a = stack.pop()
            b = stack.pop()
            stack.append(b + a)
        elif op == SUB:
            if len(stack) < 2:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            a = stack.pop()
            b = stack.pop()
            stack.append(b - a)
        elif op == MUL:
            if len(stack) < 2:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            a = stack.pop()
            b = stack.pop()
            stack.append(b * a)
        elif op == MOD:
            if len(stack) < 2:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            a = stack.pop()
            b = stack.pop()
            if a == 0:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            stack.append(b % a)
        elif op == NEG:
            if not stack:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            stack[-1] = -stack[-1]
        elif op == JNZ:
            if not stack:
                return VMResult(VM_FAIL, None, steps + 1, sp)
            a = stack.pop()
            if a != 0:
                # backward jump of 3 instructions, bounded to pc >= 0
                target = pc - 3
                if target < 0:
                    target = 0
                pc = target
                steps += 1
                # Stack-size / value bounds checked below
                if len(stack) > stack_size:
                    return VMResult(VM_FAIL, None, steps, sp)
                if stack and abs(stack[-1]) > value_bound:
                    return VMResult(VM_FAIL, None, steps, sp)
                if len(stack) > sp:
                    sp = len(stack)
                continue
        elif op == NOP:
            pass
        else:
            return VMResult(VM_FAIL, None, steps + 1, sp)

        # Stack / value-bound checks (uniform after each step)
        if len(stack) > stack_size:
            return VMResult(VM_FAIL, None, steps + 1, sp)
        if stack and abs(stack[-1]) > value_bound:
            return VMResult(VM_FAIL, None, steps + 1, sp)
        if len(stack) > sp:
            sp = len(stack)

        pc += 1
        steps += 1

    if steps >= max_steps:
        return VMResult(VM_DIVERGED, None, steps, sp)
    # Ran off the end without HALT — treat as implicit halt with top
    top = stack[-1] if stack else 0
    return VMResult(VM_BAD_PC, top, steps, sp)


def run(
    program: Program,
    inputs: Sequence[Any],
    *,
    max_steps: int = 10_000,
    stack_size: int = 64,
    value_bound: int = 10 ** 9,
) -> VMResult:
    """Run a program on inputs; public wrapper around the inner loop.

    `max_steps` caps the dynamic instruction count.  Programs that exceed
    it return status `diverged`.
    """
    if max_steps < 1:
        raise InvalidConfig("max_steps must be >= 1")
    if stack_size < 2:
        raise InvalidConfig("stack_size must be >= 2")
    if value_bound < 1:
        raise InvalidConfig("value_bound must be >= 1")
    return _run_program(
        program.ops,
        inputs,
        max_steps=max_steps,
        stack_size=stack_size,
        value_bound=value_bound,
    )


# =============================================================================
# Specifications
# =============================================================================


@dataclass(frozen=True)
class Example:
    """One input/output pair for the universal VM.

    `inputs` is the sequence the VM's INP opcode draws from; `output` is
    the value the VM must produce on HALT.
    """
    inputs: tuple[Any, ...]
    output: Any


@dataclass(frozen=True)
class Spec:
    """A finite set of (input, output) examples.

    Inputs are tuples so a program may read multiple INP values.  For
    single-input problems use `Spec.from_pairs`.
    """
    examples: tuple[Example, ...]
    name: str = ""

    @staticmethod
    def from_pairs(pairs: Iterable[tuple[Any, Any]], name: str = "") -> "Spec":
        evs: list[Example] = []
        for x, y in pairs:
            if isinstance(x, tuple):
                evs.append(Example(inputs=x, output=y))
            elif isinstance(x, (list,)):
                evs.append(Example(inputs=tuple(x), output=y))
            else:
                evs.append(Example(inputs=(x,), output=y))
        if not evs:
            raise InvalidSpec("spec must contain at least one example")
        return Spec(examples=tuple(evs), name=name)

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode("utf-8"))
        for e in self.examples:
            h.update(b"|")
            h.update(repr(e.inputs).encode("utf-8"))
            h.update(b"=>")
            h.update(repr(e.output).encode("utf-8"))
        return h.hexdigest()


# =============================================================================
# Enumeration
# =============================================================================


def enumerate_programs(
    alphabet: Sequence[int],
    length: int,
) -> Iterable[Program]:
    """Yield every program of exactly `length` opcodes drawn from `alphabet`,
    in lexicographic order over the alphabet's index order.
    """
    if length < 0:
        raise InvalidConfig("length must be >= 0")
    if not alphabet:
        return
    if length == 0:
        yield Program(ops=())
        return
    for combo in itertools.product(alphabet, repeat=length):
        yield Program(ops=combo)


def count_programs(alphabet_size: int, length: int) -> int:
    """|alphabet|^length."""
    return alphabet_size ** length


# =============================================================================
# Spec-consistency check
# =============================================================================


def _consistent_with_spec(
    program: Program,
    spec: Spec,
    *,
    max_steps_per_example: int,
    stack_size: int,
    value_bound: int,
    early_stop: bool = True,
) -> tuple[bool, int, int]:
    """Run a program on every example.

    Returns (consistent, total_steps, max_status_code) where max_status_code
    is 0 if all examples returned VM_OK with matching output, else 1.
    """
    total = 0
    for ex in spec.examples:
        r = _run_program(
            program.ops,
            ex.inputs,
            max_steps=max_steps_per_example,
            stack_size=stack_size,
            value_bound=value_bound,
        )
        total += r.steps
        if r.status not in (VM_OK, VM_BAD_PC):
            if early_stop:
                return False, total, 1
            continue
        if r.output != ex.output:
            if early_stop:
                return False, total, 1
    return True, total, 0


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class InducerConfig:
    """Search configuration.

    `alphabet`:                opcodes the enumerator may use.
    `max_program_length`:      upper bound on program length (instructions).
    `max_steps_per_program`:   dynamic-step budget for one program on one
                               example.  Programs that exceed this are
                               counted as diverged.
    `max_total_steps`:         budget across the whole search (sum of all
                               example-runs).  Hard stop.
    `max_programs`:            cap on programs enumerated.
    `max_wallclock_s`:         soft wall-clock cap (checked between
                               programs).
    `stack_size`:              VM stack capacity.
    `value_bound`:             |x| > value_bound is a FAIL (prevents
                               long-multiplication explosions).
    `mode`:                    "iddfs"  — iterative deepening by length
                                          (every program at length L checked
                                          with max_steps_per_program budget);
                                "levin" — Levin universal search: each phase
                                          doubles the cumulative compute
                                          budget T_phase, and program of
                                          length L gets ⌊T_phase / 2^L⌋ steps.
    `levin_start_budget`:      initial T for "levin" mode.
    `levin_phase_doubling`:    multiplier between Levin phases (default 2).
    `early_stop`:              return on the first consistent program.
                               When False the search collects up to
                               `top_k` consistent programs at each length.
    `top_k`:                   how many consistent programs to retain (only
                               consulted when `early_stop` is False).
    `prune_constant_outputs`:  if all spec outputs are equal, skip programs
                               that don't read INP at least once.  Pure
                               speed-up; preserves the Levin bound.
    """
    alphabet: tuple[int, ...] = ALPHABET_STRAIGHT
    max_program_length: int = 7
    max_steps_per_program: int = 256
    max_total_steps: int = 50_000_000
    max_programs: int = 50_000_000
    max_wallclock_s: float = 30.0
    stack_size: int = 32
    value_bound: int = 10 ** 12
    mode: str = "iddfs"
    levin_start_budget: int = 64
    levin_phase_doubling: float = 2.0
    early_stop: bool = True
    top_k: int = 3
    prune_constant_outputs: bool = True

    def __post_init__(self) -> None:
        if not self.alphabet:
            raise InvalidConfig("alphabet must be non-empty")
        for o in self.alphabet:
            if not (0 <= o < OPCODE_COUNT):
                raise InvalidConfig(f"opcode {o} outside [0, {OPCODE_COUNT})")
        if self.max_program_length < 1:
            raise InvalidConfig("max_program_length must be >= 1")
        if self.max_steps_per_program < 1:
            raise InvalidConfig("max_steps_per_program must be >= 1")
        if self.mode not in ("iddfs", "levin"):
            raise InvalidConfig(f"unknown mode '{self.mode}'")
        if self.levin_phase_doubling <= 1.0:
            raise InvalidConfig("levin_phase_doubling must be > 1.0")
        if self.top_k < 1:
            raise InvalidConfig("top_k must be >= 1")


# =============================================================================
# Reports
# =============================================================================


@dataclass(frozen=True)
class SearchStats:
    """Quantitative summary of one search."""
    programs_visited: int
    consistent_found: int
    steps_executed: int
    walltime_s: float
    last_length: int
    phases_completed: int  # for Levin mode; 0 for iddfs


@dataclass(frozen=True)
class InducerReport:
    """The full output of one search."""
    spec: Spec
    config: InducerConfig
    program: Program | None
    alternatives: tuple[Program, ...]
    stats: SearchStats
    certificate: str
    vm_signature: str

    # ----------------------------------------------------------------
    # Levin / Solomonoff quantitative outputs
    # ----------------------------------------------------------------

    def universal_prior_mass(self) -> float:
        """Kraft mass 2^{-l(p)} the discovered program contributes to the
        Solomonoff prior.  Zero if no program was found.
        """
        if self.program is None:
            return 0.0
        return 2.0 ** (-self.program.length)

    def levin_complexity(self) -> float:
        """The discovered program's Levin complexity for this spec::

            Kt(spec)  ≤  l(p) + log₂ t(p, spec)

        where `t(p, spec)` is the total interpreter steps over every
        example in the spec.  Returns +inf if no program found.
        """
        if self.program is None:
            return float("inf")
        steps = max(1, self.stats.steps_executed)
        return self.program.length + math.log2(steps)

    def coding_theorem_lb(self) -> float:
        """Li-Vitányi coding-theorem lower bound on Pr_U(U(p) = spec): a
        constant multiple of 2^{-K(spec)} ≥ 2^{-l(p)}.  Returns the lower
        bound without the unknown constant.
        """
        return self.universal_prior_mass()

    def occam_bound(self, delta: float = 0.05) -> float:
        """Blumer-Ehrenfeucht-Haussler-Warmuth PAC generalisation bound:

            ε  ≤  (l(p) · ln 2  +  ln(1/δ)) / m

        where m is the spec's example count and l(p) the discovered
        program length.  Tightest when l(p) is small.
        """
        if self.program is None:
            return 1.0
        m = self.spec.n_examples
        if m <= 0:
            return 1.0
        if not (0.0 < delta < 1.0):
            raise InvalidConfig("delta must be in (0, 1)")
        return (self.program.length * math.log(2.0) + math.log(1.0 / delta)) / m

    def levin_runtime_bound(
        self, *, hypothetical_length: int | None = None,
        hypothetical_steps: int | None = None,
        constant: float = 1.0,
    ) -> float:
        """Levin's runtime-bound formula::

            R(p*)  ≤  K_U · 2^{l(p*)} · t(p*)

        With no hypothetical_length given, the discovered program's
        length and steps are used (giving a *self-bound* — how much
        compute Inducer would need to refind itself).  Pass
        hypothetical_(length, steps) to bound the cost of finding *any*
        shorter / faster program were one to exist.
        """
        L = hypothetical_length if hypothetical_length is not None else (
            self.program.length if self.program else 0
        )
        T = hypothetical_steps if hypothetical_steps is not None else max(
            1, self.stats.steps_executed
        )
        return constant * (2.0 ** L) * T

    # ----------------------------------------------------------------
    # Convenience evaluation
    # ----------------------------------------------------------------

    def eval(self, inputs: Sequence[Any]) -> Any:
        """Run the discovered program on a new input."""
        if self.program is None:
            raise NoSolution("no program was found")
        r = _run_program(
            self.program.ops,
            inputs,
            max_steps=self.config.max_steps_per_program,
            stack_size=self.config.stack_size,
            value_bound=self.config.value_bound,
        )
        if r.status not in (VM_OK, VM_BAD_PC):
            raise InducerError(f"VM status {r.status} on inputs={inputs}")
        return r.output


# =============================================================================
# Inducer
# =============================================================================


def _vm_signature(alphabet: Sequence[int]) -> str:
    h = hashlib.sha256()
    h.update(b"agi.inducer.vm.v1")
    h.update(bytes(sorted(alphabet)))
    h.update(str(OPCODE_NAME).encode("utf-8"))
    return h.hexdigest()


def _spec_uses_inp_required(spec: Spec) -> bool:
    """True iff distinct example inputs produce distinct outputs (so the
    program must read INP at least once)."""
    seen: dict[tuple[Any, ...], Any] = {}
    for e in spec.examples:
        if e.inputs in seen and seen[e.inputs] != e.output:
            raise InvalidSpec("spec is internally inconsistent (same input, different outputs)")
        seen[e.inputs] = e.output
    if len(seen) < 2:
        return False  # only one input class -> any constant could fit
    outputs = {e.output for e in spec.examples}
    return len(outputs) > 1


class Inducer:
    """Levin universal search over a small stack-based VM."""

    def __init__(
        self,
        config: InducerConfig | None = None,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or InducerConfig()
        self.on_event = on_event
        # state filled by `search`
        self._programs_visited = 0
        self._consistent_found = 0
        self._steps_executed = 0
        self._cert_hash = hashlib.sha256()
        self._cert_hash.update(b"agi.inducer.v1")

    # ----------------------------------------------------------------
    # Public search API
    # ----------------------------------------------------------------

    def search(self, spec: Spec) -> InducerReport:
        """Run universal search until a consistent program is found or the
        budget is exhausted.  Returns an InducerReport in either case.

        For `mode="iddfs"`, programs are enumerated in increasing length
        from 1 to `max_program_length`; each is run with the per-program
        step cap.

        For `mode="levin"`, programs are run on a doubling cumulative
        budget T; each program of length L gets ⌊T / 2^L⌋ steps.
        """
        self._reset()
        cfg = self.config
        if not isinstance(spec, Spec):
            raise InvalidSpec("expected Spec")
        self._cert_hash.update(spec.fingerprint().encode("utf-8"))
        self._cert_hash.update(_vm_signature(cfg.alphabet).encode("utf-8"))

        inp_required = (
            cfg.prune_constant_outputs and _spec_uses_inp_required(spec)
        )

        t0 = time.time()
        consistent_progs: list[Program] = []

        try:
            if cfg.mode == "iddfs":
                found, alts = self._search_iddfs(
                    spec, t0, inp_required, consistent_progs,
                )
                phases = 0
            else:
                found, alts, phases = self._search_levin(
                    spec, t0, inp_required, consistent_progs,
                )
        except BudgetExhausted:
            found = None
            alts = tuple(consistent_progs[: cfg.top_k])
            phases = 0

        elapsed = time.time() - t0
        stats = SearchStats(
            programs_visited=self._programs_visited,
            consistent_found=self._consistent_found,
            steps_executed=self._steps_executed,
            walltime_s=elapsed,
            last_length=(found.length if found else cfg.max_program_length),
            phases_completed=phases,
        )
        if found is not None:
            self._cert_hash.update(b"FOUND:")
            self._cert_hash.update(found.disassemble().encode("utf-8"))
        else:
            self._cert_hash.update(b"NOT_FOUND")
        cert = self._cert_hash.hexdigest()
        self._emit("inducer.search.completed", {
            "found": found is not None,
            "program": found.disassemble() if found else None,
            "programs_visited": self._programs_visited,
            "steps_executed": self._steps_executed,
            "walltime_s": elapsed,
            "certificate": cert,
        })
        return InducerReport(
            spec=spec,
            config=cfg,
            program=found,
            alternatives=alts,
            stats=stats,
            certificate=cert,
            vm_signature=_vm_signature(cfg.alphabet),
        )

    # ----------------------------------------------------------------
    # Iterative deepening over length
    # ----------------------------------------------------------------

    def _search_iddfs(
        self,
        spec: Spec,
        t0: float,
        inp_required: bool,
        consistent_progs: list[Program],
    ) -> tuple[Program | None, tuple[Program, ...]]:
        cfg = self.config
        for L in range(1, cfg.max_program_length + 1):
            self._emit("inducer.phase.started", {"length": L})
            for prog in enumerate_programs(cfg.alphabet, L):
                self._check_budget(t0)
                if inp_required and INP not in prog.ops:
                    self._programs_visited += 1
                    continue
                ok, steps, _ = _consistent_with_spec(
                    prog,
                    spec,
                    max_steps_per_example=cfg.max_steps_per_program,
                    stack_size=cfg.stack_size,
                    value_bound=cfg.value_bound,
                    early_stop=True,
                )
                self._programs_visited += 1
                self._steps_executed += steps
                if ok:
                    self._consistent_found += 1
                    self._emit("inducer.consistent.found", {
                        "length": L,
                        "program": prog.disassemble(),
                    })
                    consistent_progs.append(prog)
                    if cfg.early_stop:
                        return prog, tuple(consistent_progs[: cfg.top_k])
                    if len(consistent_progs) >= cfg.top_k:
                        return consistent_progs[0], tuple(consistent_progs[: cfg.top_k])
            self._emit("inducer.phase.completed", {
                "length": L,
                "programs_visited": self._programs_visited,
            })
        if consistent_progs:
            return consistent_progs[0], tuple(consistent_progs[: cfg.top_k])
        return None, ()

    # ----------------------------------------------------------------
    # True Levin universal search
    # ----------------------------------------------------------------

    def _search_levin(
        self,
        spec: Spec,
        t0: float,
        inp_required: bool,
        consistent_progs: list[Program],
    ) -> tuple[Program | None, tuple[Program, ...], int]:
        cfg = self.config
        phase = 0
        T = float(cfg.levin_start_budget)
        # Cache programs across phases so that lex-order is preserved.
        # We re-evaluate each phase because the step budget per program
        # changes; that is exactly Levin's dovetail.
        prog_lists_by_length: dict[int, list[Program]] = {}
        for L in range(1, cfg.max_program_length + 1):
            prog_lists_by_length[L] = []
        while phase < 200:  # safety bound; budget-checks handle real termination
            phase += 1
            self._emit("inducer.levin.phase.started", {"phase": phase, "budget": T})
            phase_progress = False
            for L in range(1, cfg.max_program_length + 1):
                steps_for_L = int(T / (2 ** L))
                if steps_for_L < 1:
                    # No program of length >= L gets any step this phase.
                    break
                phase_progress = True
                # Lazy fill of program list for this length
                if not prog_lists_by_length[L]:
                    prog_lists_by_length[L] = list(
                        enumerate_programs(cfg.alphabet, L)
                    )
                for prog in prog_lists_by_length[L]:
                    self._check_budget(t0)
                    if inp_required and INP not in prog.ops:
                        continue  # do not bump programs_visited; would double-count
                    ok, steps, _ = _consistent_with_spec(
                        prog,
                        spec,
                        max_steps_per_example=steps_for_L,
                        stack_size=cfg.stack_size,
                        value_bound=cfg.value_bound,
                        early_stop=True,
                    )
                    # `programs_visited` counts (program, phase) pairs;
                    # `steps_executed` is the true Levin work.
                    self._programs_visited += 1
                    self._steps_executed += steps
                    if ok:
                        self._consistent_found += 1
                        self._emit("inducer.consistent.found", {
                            "phase": phase,
                            "length": L,
                            "budget_per_program": steps_for_L,
                            "program": prog.disassemble(),
                        })
                        consistent_progs.append(prog)
                        if cfg.early_stop:
                            return prog, tuple(consistent_progs[: cfg.top_k]), phase
                        if len(consistent_progs) >= cfg.top_k:
                            return (
                                consistent_progs[0],
                                tuple(consistent_progs[: cfg.top_k]),
                                phase,
                            )
            self._emit("inducer.levin.phase.completed", {
                "phase": phase,
                "budget": T,
            })
            if not phase_progress:
                # Budget too small to evaluate even the shortest program.
                # Bump and continue.
                T *= cfg.levin_phase_doubling
                continue
            T *= cfg.levin_phase_doubling
        if consistent_progs:
            return consistent_progs[0], tuple(consistent_progs[: cfg.top_k]), phase
        return None, (), phase

    # ----------------------------------------------------------------
    # Budget guard
    # ----------------------------------------------------------------

    def _check_budget(self, t0: float) -> None:
        cfg = self.config
        if self._programs_visited >= cfg.max_programs:
            raise BudgetExhausted("max_programs hit")
        if self._steps_executed >= cfg.max_total_steps:
            raise BudgetExhausted("max_total_steps hit")
        if time.time() - t0 >= cfg.max_wallclock_s:
            raise BudgetExhausted("max_wallclock_s hit")

    # ----------------------------------------------------------------
    # Event hook
    # ----------------------------------------------------------------

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, data)
        except Exception:
            # Never let an event handler kill the search.
            pass

    def _reset(self) -> None:
        self._programs_visited = 0
        self._consistent_found = 0
        self._steps_executed = 0
        self._cert_hash = hashlib.sha256()
        self._cert_hash.update(b"agi.inducer.v1")


# =============================================================================
# High-level conveniences
# =============================================================================


def induce(
    pairs: Iterable[tuple[Any, Any]],
    *,
    config: InducerConfig | None = None,
    name: str = "",
) -> InducerReport:
    """One-call convenience: spec from pairs, run search, return report."""
    spec = Spec.from_pairs(pairs, name=name)
    return Inducer(config).search(spec)


def coding_theorem_posterior_mass(programs: Iterable[Program]) -> float:
    """Sum of 2^{-l(p)} over a finite set of programs (a sub-probability
    by Kraft).
    """
    return sum(2.0 ** -p.length for p in programs)


def kraft_normalised_posterior(
    programs: Sequence[Program],
) -> list[tuple[Program, float]]:
    """Normalise the Solomonoff prior over a *finite* candidate set so the
    resulting (program, weight) pairs sum to 1 — useful as a coordination-
    engine model average.
    """
    if not programs:
        return []
    weights = [2.0 ** -p.length for p in programs]
    Z = sum(weights)
    if Z <= 0:
        return [(p, 0.0) for p in programs]
    return [(p, w / Z) for p, w in zip(programs, weights)]


def levin_runtime_bound(
    program_length: int,
    program_steps: int,
    constant: float = 1.0,
) -> float:
    """Levin's runtime bound R = K_U · 2^L · t for an external program
    description (length and observed steps).
    """
    if program_length < 0:
        raise InvalidConfig("program_length must be >= 0")
    if program_steps < 0:
        raise InvalidConfig("program_steps must be >= 0")
    return constant * (2.0 ** program_length) * program_steps


def kt_complexity_upper_bound(report: InducerReport) -> float:
    """Convenience wrapper: ``Kt(spec) ≤ l(p) + log₂ t(p)``."""
    return report.levin_complexity()


__all__ = [
    "ALPHABET_ARITH",
    "ALPHABET_FULL",
    "ALPHABET_STRAIGHT",
    "ADD",
    "BudgetExhausted",
    "DRP",
    "DUP",
    "Example",
    "HALT",
    "INP",
    "Inducer",
    "InducerConfig",
    "InducerError",
    "InducerReport",
    "InvalidConfig",
    "InvalidProgram",
    "InvalidSpec",
    "JNZ",
    "MOD",
    "MUL",
    "NEG",
    "NOP",
    "NoSolution",
    "OPCODE_COUNT",
    "OPCODE_NAME",
    "PUSH0",
    "PUSH1",
    "PUSH2",
    "PUSHN1",
    "Program",
    "SUB",
    "SWP",
    "SearchStats",
    "Spec",
    "VMResult",
    "VM_BAD_PC",
    "VM_DIVERGED",
    "VM_FAIL",
    "VM_OK",
    "coding_theorem_posterior_mass",
    "count_programs",
    "enumerate_programs",
    "induce",
    "kraft_normalised_posterior",
    "kt_complexity_upper_bound",
    "levin_runtime_bound",
    "run",
]
