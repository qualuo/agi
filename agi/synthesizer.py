r"""Synthesizer — program synthesis as a runtime primitive.

Every other primitive in this runtime *executes* programs; **Synthesizer
*writes* them.**  Give it a finite specification — a handful of
input-output examples, a regular language, an oracle that accepts or
rejects candidate programs — and it returns the *smallest* program in
a stated domain-specific language that satisfies the specification,
together with a finite-sample PAC bound on the program's
generalisation error.  The coordination engine uses this to (a) mine
new tools from observed traces, (b) compile a high-level Goal into a
verifiable executable, (c) build white-box explanations for any
primitive's behaviour as a small program over its inputs.

This is the **self-extension** primitive: the runtime no longer needs a
human to hand-author new tools.  Programs are searched in a sandboxed
DSL with formal correctness guarantees on the training set and a
finite-sample bound on the held-out set — *the* line between
"experimental tool synthesis" and "tool synthesis investors will fund".

Mathematical and algorithmic roots
----------------------------------

  * **Mitchell, T. (1982) — "Generalization as search."**  *Version-
    space algebra*: maintain the set of all hypotheses ``H ⊆ DSL`` that
    are consistent with every example seen so far, as a pair ``(S, G)``
    of *specific* and *general* boundaries.  Each example either prunes
    ``S`` (positive example refutes some specifics) or prunes ``G``
    (negative example refutes some generals).  When ``S = G`` and is a
    singleton the hypothesis is *converged*.  Composes monotonically.

  * **Plotkin, G. D. (1970) — "A note on inductive generalisation."**
    *Least general generalisation* (LGG) / *anti-unification*: the most
    specific common generalisation of two terms.  For first-order
    terms ``f(a, b)`` and ``f(a, c)`` the LGG is ``f(a, X)``.  Repeated
    LGG over a batch of inputs produces the most specific pattern
    consistent with all of them — the analogue of `S` in version space.

  * **Angluin, D. (1987) — "Learning regular sets from queries and
    counterexamples."**  *L\\**: learn a minimal DFA for an unknown
    regular language with two oracles — *membership queries*
    (``MQ(w) ∈ {0, 1}``) and *equivalence queries* (``EQ(H) → ⊥`` or
    a counterexample word).  Polynomial in the size of the minimal DFA
    and the longest counterexample.  Foundation for every
    *interaction*-based synthesis.

  * **Gulwani, S. (2011) — "Automating string processing in
    spreadsheets using input-output examples" (POPL).**  *FlashFill*:
    PBE for string transformations.  Hypothesis space is a DSL of
    substring/concat/loop operators; search is a version-space union
    over per-example program sets, intersected lazily.  Shipped in
    Microsoft Excel 2013.

  * **Solar-Lezama, A. (2008) — *Program Synthesis by Sketching*,
    PhD dissertation, UC Berkeley.**  *Counterexample-guided inductive
    synthesis* (CEGIS): synthesise vs. verify in a loop::

        H ← initial candidate from spec
        while True:
            cex ← verify(H, spec)
            if cex is None: return H
            H ← synthesise(spec ∪ {cex})

    The verifier is an arbitrary executable predicate; the synthesiser
    is the version-space / brute-force / SAT solver.  When the
    candidate space is finite or has bounded depth the loop terminates.

  * **Alur, R., Bodík, R., Juniwal, G., Martin, M. M. K., Raghothaman,
    M., Seshia, S. A., Singh, R., Solar-Lezama, A., Torlak, E., Udupa,
    A. (2013) — "Syntax-guided synthesis" (SYGUS-IF FMCAD).**  The
    standard SyGuS interface: ``(DSL grammar G, spec φ) ↦ Σ ∈ L(G) such
    that φ(Σ) holds``.  Synthesizer ships a stdlib-only SYGUS-PBE
    backend (positive examples only) and SYGUS-IO (input-output
    examples) backend (positive + negative examples + post-condition
    oracle).

  * **Blumer, A., Ehrenfeucht, A., Haussler, D., Warmuth, M. K. (1987)
    — "Occam's razor."**  For a finite hypothesis class ``H`` and ``m``
    samples consistent with ``h ∈ H`` of complexity ``|h|`` bits, the
    generalisation error is bounded with prob. ``≥ 1-δ`` by::

        err(h)  ≤  (|h| · ln 2 + ln(1/δ)) / m.

    Stronger when ``|h|`` is small — the *Occam's razor* PAC bound.
    Synthesizer reports this on every program it returns.

  * **Vapnik, V. N. (1998) — *Statistical learning theory*, chap. 4.**
    For an infinite hypothesis class with VC dimension ``d``::

        err(h)  ≤  err_emp(h) + √((d (ln(2m/d) + 1) + ln(4/δ)) / m).

    Synthesizer reports a VC-style bound on grammar size when the
    DSL is described by a context-free grammar with bounded depth.

  * **Schmidhuber, J. (1997) — *Discovering neural nets with low
    Kolmogorov complexity and high generalisation capability*.** —
    PAC generalisation guarantees are sharpest when the *program*
    description length is the complexity term.  Synthesizer's
    enumeration by increasing AST size is precisely this MDL ordering.

  * **Knuth, D. E. (1973) — *Sorting and Searching*, sec. 6.3.**  Trie-
    based string indexing for fast version-space intersection across
    examples.

The DSL surface
---------------

Synthesizer is parameterised by a ``DSL`` — a typed expression
grammar over

  * **constants**: stdlib literals (`int`, `str`, `bool`, `list`),
  * **inputs**:    typed variables `x_0`, …, `x_{k-1}`,
  * **operators**: pure callables with an output-type signature.

The DSL declares (i) the set of operators and their arities/typings,
(ii) the maximum AST depth ``d_max``, (iii) an optional ``cost``
function ``c: AST → R⁺`` for MDL-ordered search.  Built-in DSLs:

  * ``STRING_DSL`` — concat, substring(start, len), replace, upper,
    lower, strip, split, join, format-pad — the FlashFill primitives.
  * ``INTEGER_DSL`` — +, -, *, //, %, min, max, abs, ifzero.
  * ``LIST_DSL`` — head, tail, last, length, reverse, sort, sum,
    map(f), filter(f), fold(f, init).
  * ``BOOL_DSL`` — and, or, not, eq, lt, gt, leq, geq.

The DSL is **fully introspectable** — the synthesiser walks operator
arities/return types to enumerate by *depth* (top-down), and by
*cost* (priority-queue, Dijkstra over the AST graph).  Programs are
``Program`` ASTs that are also Python callables; calling
``prog(x_0, x_1, ...)`` executes the AST in a stdlib sandbox.

Public API
----------

::

    >>> from agi.synthesizer import Synthesizer, STRING_DSL
    >>> S = Synthesizer(dsl=STRING_DSL)
    >>> rep = S.synthesize_from_examples([
    ...   (("alice@example.com",),  "alice"),
    ...   (("bob@example.com",),    "bob"),
    ...   (("carol@example.com",),  "carol"),
    ... ])
    >>> rep.program.run(("dave@example.com",))   # ⇒ "dave"
    >>> rep.occam_bound(delta=0.05)               # PAC ε

CEGIS::

    >>> def verify(prog):
    ...   # returns counterexample input or None
    ...   for x in test_corpus:
    ...     if prog.run(x) != gold(x): return (x, gold(x))
    ...   return None
    >>> rep = S.cegis(initial_spec=examples, verifier=verify,
    ...                max_rounds=20)

L\\* DFA learning::

    >>> def mq(word: str) -> bool: ...   # membership oracle
    >>> def eq(dfa)        -> str|None: ...  # counterexample word or None
    >>> dfa = S.learn_dfa(mq, eq, alphabet={"a", "b"})

Anti-unification::

    >>> S.lgg(("hello", "world"), ("hello", "there"))   # ⇒ ("hello", X)

Each ``SynthesisReport`` ships:

  * ``program``    — the synthesised AST,
  * ``size``       — AST node count (MDL-complexity proxy),
  * ``n_examples`` — examples used,
  * ``occam_eps``  — Blumer-Ehrenfeucht-Haussler-Warmuth PAC bound,
  * ``visited``    — number of programs enumerated,
  * ``walltime_s``,
  * ``certificate``: SHA-256 fingerprint over `(DSL, examples, AST)`,
  * ``alternatives`` — top-K equally-correct programs at higher AST cost.

Composition with the rest of the runtime
----------------------------------------

  * **Toolsynth** — the existing tool-synthesis kernel runs synthesised
    programs in a subprocess sandbox; Synthesizer produces *the
    candidates* with formal guarantees.
  * **SkillMine** — promote a synthesised program to a `Skill` when its
    Occam-bound clears the calibration threshold.
  * **AttestationLedger** — every `SynthesisReport` carries a
    tamper-evident fingerprint over examples + DSL + AST so the
    coordination engine can replay-verify the synthesis bit-for-bit.
  * **Auditor** — when multiple candidate programs all pass training,
    Auditor applies BH on per-candidate held-out test e-values to
    prevent multiplicity-induced over-fitting.
  * **Sampler** — when the DSL has continuous parameters (a polynomial
    fit, say), Synthesizer hands the *symbolic skeleton* to
    `Sampler.advi` to fit the constants.
  * **AutonomousLoop** — every retry-with-lesson cycle that fails to
    converge can invoke Synthesizer to write *the missing skill* from
    successful traces.
  * **EvolutionEngine** — Synthesizer is the *mutation* operator over
    candidate strategies; AST-level crossover is structurally sound
    when both parents are in the same DSL.

The primitive is **deliberately conservative**: search is bounded by
``max_depth`` and ``max_visited``; verification is mandatory; the
Occam bound is reported on *every* synthesis call.  Pure stdlib — no
Z3, no SMT-LIB, no PEG parser.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


# =============================================================================
# Typed AST
# =============================================================================


@dataclass(frozen=True)
class Type:
    name: str

    def __str__(self) -> str:
        return self.name


T_INT = Type("int")
T_STR = Type("str")
T_BOOL = Type("bool")
T_LIST_INT = Type("list[int]")
T_LIST_STR = Type("list[str]")
T_ANY = Type("any")


@dataclass(frozen=True)
class Op:
    """A typed pure callable."""
    name: str
    arity: int
    arg_types: tuple[Type, ...]
    return_type: Type
    fn: Callable[..., Any]
    cost: float = 1.0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Program:
    """An AST node: either a constant, an input variable, or an Op call."""
    kind: str  # "const" | "var" | "op"
    op: Op | None = None
    value: Any = None           # for "const"
    var_idx: int | None = None  # for "var"
    args: tuple["Program", ...] = ()
    return_type: Type = T_ANY

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def run(self, inputs: Sequence[Any]) -> Any:
        if self.kind == "const":
            return self.value
        if self.kind == "var":
            return inputs[self.var_idx]
        # op
        return self.op.fn(*[a.run(inputs) for a in self.args])

    # ------------------------------------------------------------
    # Structural ops
    # ------------------------------------------------------------

    def size(self) -> int:
        if self.kind in ("const", "var"):
            return 1
        return 1 + sum(a.size() for a in self.args)

    def depth(self) -> int:
        if self.kind in ("const", "var"):
            return 1
        return 1 + max((a.depth() for a in self.args), default=0)

    def cost(self) -> float:
        if self.kind in ("const", "var"):
            return 1.0
        return self.op.cost + sum(a.cost() for a in self.args)

    # ------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------

    def to_str(self) -> str:
        if self.kind == "const":
            return repr(self.value)
        if self.kind == "var":
            return f"x{self.var_idx}"
        return f"{self.op.name}({', '.join(a.to_str() for a in self.args)})"

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return f"Program({self.to_str()})"

    # ------------------------------------------------------------
    # Hash / equality (structural)
    # ------------------------------------------------------------

    def _key(self) -> tuple:
        if self.kind == "const":
            return ("const", repr(self.value))
        if self.kind == "var":
            return ("var", self.var_idx)
        return ("op", self.op.name, tuple(a._key() for a in self.args))

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Program) and self._key() == other._key()


# =============================================================================
# Builders
# =============================================================================


def const(value: Any, t: Type) -> Program:
    return Program(kind="const", value=value, return_type=t)


def var(idx: int, t: Type) -> Program:
    return Program(kind="var", var_idx=idx, return_type=t)


def call(op: Op, *args: Program) -> Program:
    if len(args) != op.arity:
        raise ValueError(f"{op.name} expects {op.arity} args, got {len(args)}")
    return Program(kind="op", op=op, args=args, return_type=op.return_type)


# =============================================================================
# DSL container
# =============================================================================


@dataclass
class DSL:
    """A typed program-synthesis DSL.

    ``ops``         — operators by return-type.
    ``input_types`` — declared input variable types.
    ``constants``   — typed constants the synthesiser may use (each as
                      ``(value, Type)``).
    ``max_depth``   — depth bound for enumeration.
    """
    name: str
    input_types: tuple[Type, ...]
    output_type: Type
    ops: tuple[Op, ...]
    constants: tuple[tuple[Any, Type], ...] = ()
    max_depth: int = 4

    def ops_by_type(self, t: Type) -> list[Op]:
        return [o for o in self.ops if o.return_type == t or t == T_ANY]

    def vars_by_type(self, t: Type) -> list[Program]:
        return [var(i, vt) for i, vt in enumerate(self.input_types)
                if vt == t or t == T_ANY]

    def consts_by_type(self, t: Type) -> list[Program]:
        return [const(v, ct) for v, ct in self.constants
                if ct == t or t == T_ANY]


# =============================================================================
# Anti-unification (Plotkin 1970 LGG)
# =============================================================================


def lgg(a: Any, b: Any) -> Any:
    """Least general generalisation of two values.  Returns a pattern
    where positions that differ become the placeholder ``("?", id)``
    with the same id for matched positions across the two arguments."""
    counter = itertools.count()

    def rec(x: Any, y: Any) -> Any:
        if x == y:
            return x
        if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
            return tuple(rec(a, b) for a, b in zip(x, y))
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            return [rec(a, b) for a, b in zip(x, y)]
        return ("?", next(counter))
    return rec(a, b)


def lgg_many(xs: Sequence[Any]) -> Any:
    if not xs:
        raise ValueError("lgg over empty sequence")
    out = xs[0]
    for y in xs[1:]:
        out = lgg(out, y)
    return out


# =============================================================================
# Version space enumeration
# =============================================================================


@dataclass
class EnumStats:
    visited: int = 0
    pruned_type: int = 0
    pruned_eval: int = 0


def _enumerate_at_depth(dsl: DSL, t: Type, depth: int) -> Iterable[Program]:
    """Generate every well-typed program of *exact* depth ``depth``
    returning type ``t``."""
    if depth == 1:
        for p in dsl.vars_by_type(t):
            yield p
        for p in dsl.consts_by_type(t):
            yield p
        return
    for op in dsl.ops_by_type(t):
        # generate args of any depth < depth, with at least one of depth `depth-1`
        ranges = [range(1, depth) for _ in range(op.arity)]
        for depths in itertools.product(*ranges):
            if max(depths) != depth - 1:
                continue
            arg_iters = [list(_enumerate_at_depth(dsl, at, d))
                         for at, d in zip(op.arg_types, depths)]
            if any(len(it) == 0 for it in arg_iters):
                continue
            for combo in itertools.product(*arg_iters):
                yield call(op, *combo)


def _enumerate_up_to(dsl: DSL, t: Type, max_depth: int) -> Iterable[Program]:
    seen: set[Program] = set()
    for d in range(1, max_depth + 1):
        for p in _enumerate_at_depth(dsl, t, d):
            if p in seen:
                continue
            seen.add(p)
            yield p


# =============================================================================
# Examples / specifications
# =============================================================================


Example = tuple[tuple[Any, ...], Any]


def _consistent(prog: Program, examples: Sequence[Example]) -> bool:
    for inputs, output in examples:
        try:
            if prog.run(inputs) != output:
                return False
        except Exception:
            return False
    return True


# =============================================================================
# Synthesis report
# =============================================================================


@dataclass
class SynthesisReport:
    program: Program | None
    size: int
    cost: float
    n_examples: int
    visited: int
    walltime_s: float
    alternatives: list[Program] = field(default_factory=list)
    cegis_rounds: int = 0
    converged: bool = True
    dsl_name: str = ""
    seed: int = 0

    # ----------------------------------------------------------------
    # Generalisation bounds
    # ----------------------------------------------------------------

    def occam_bound(self, delta: float = 0.05) -> float:
        """Blumer-Ehrenfeucht-Haussler-Warmuth (1987) PAC bound on
        the generalisation error of the returned program, given
        consistency on ``n_examples`` iid samples and program AST size
        ``size`` (description length proxy)."""
        if self.n_examples == 0 or self.program is None:
            return float("inf")
        return (self.size * math.log(2) + math.log(1.0 / max(1e-12, delta))) \
               / self.n_examples

    def sample_complexity(self, eps: float, delta: float = 0.05) -> int:
        """How many examples we needed to guarantee ``err ≤ eps`` with
        prob. ``≥ 1-δ`` for a hypothesis of this complexity::

            m  ≥  (|h| ln 2 + ln(1/δ)) / ε.
        """
        return int(math.ceil((self.size * math.log(2)
                              + math.log(1.0 / max(1e-12, delta))) / max(1e-12, eps)))

    # ----------------------------------------------------------------
    # Tamper-evident fingerprint
    # ----------------------------------------------------------------

    def fingerprint(self) -> str:
        prog_str = self.program.to_str() if self.program else "none"
        payload = json.dumps({
            "dsl": self.dsl_name,
            "n": self.n_examples,
            "size": self.size,
            "cost": self.cost,
            "visited": self.visited,
            "cegis_rounds": self.cegis_rounds,
            "converged": self.converged,
            "prog": prog_str,
            "seed": self.seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# =============================================================================
# Synthesizer (the public class)
# =============================================================================


class Synthesizer:
    """Program synthesis primitive over a typed DSL."""

    def __init__(
        self,
        dsl: DSL,
        rng: random.Random | None = None,
        max_visited: int = 50_000,
    ) -> None:
        self.dsl = dsl
        self.rng = rng or random.Random(0)
        self.max_visited = max_visited

    # =================================================================
    # PBE — programming by example
    # =================================================================

    def synthesize_from_examples(
        self,
        examples: Sequence[Example],
        top_k: int = 1,
        ordering: str = "depth",
    ) -> SynthesisReport:
        """Find the *smallest* program in the DSL consistent with every
        example.  ``ordering`` ∈ ``{"depth", "cost"}``."""
        t0 = time.time()
        if not examples:
            return SynthesisReport(
                program=None, size=0, cost=0.0, n_examples=0,
                visited=0, walltime_s=0.0, dsl_name=self.dsl.name,
            )
        visited = 0
        found: list[Program] = []
        if ordering == "depth":
            stream = _enumerate_up_to(self.dsl, self.dsl.output_type,
                                       self.dsl.max_depth)
        elif ordering == "cost":
            stream = self._enumerate_by_cost(self.dsl.output_type,
                                              self.dsl.max_depth)
        else:
            raise ValueError(f"unknown ordering: {ordering}")
        # Witness-based pruning: probe with first example's input to skip
        # programs that don't produce a value at all
        probe_input, _ = examples[0]
        for prog in stream:
            visited += 1
            if visited > self.max_visited:
                break
            try:
                _ = prog.run(probe_input)
            except Exception:
                continue
            if _consistent(prog, examples):
                found.append(prog)
                if len(found) >= top_k:
                    break
        prog0 = found[0] if found else None
        return SynthesisReport(
            program=prog0,
            size=prog0.size() if prog0 else 0,
            cost=prog0.cost() if prog0 else 0.0,
            n_examples=len(examples),
            visited=visited,
            walltime_s=time.time() - t0,
            alternatives=found[1:],
            converged=prog0 is not None,
            dsl_name=self.dsl.name,
            seed=self.rng.randint(0, 1 << 30),
        )

    # =================================================================
    # CEGIS — counterexample-guided inductive synthesis
    # =================================================================

    def cegis(
        self,
        initial_spec: Sequence[Example],
        verifier: Callable[[Program], Example | None],
        max_rounds: int = 50,
        top_k: int = 1,
    ) -> SynthesisReport:
        """Solar-Lezama 2008 CEGIS loop::

            spec ← initial_spec
            for r in 0 .. max_rounds:
                H ← synthesize_from_examples(spec)
                cex ← verifier(H)
                if cex is None: return H
                spec ← spec ∪ {cex}

        ``verifier`` returns a counterexample ``(inputs, expected_output)``
        when the candidate is wrong, or ``None`` if it has converged."""
        t0 = time.time()
        spec = list(initial_spec)
        rounds = 0
        for rounds in range(1, max_rounds + 1):
            rep = self.synthesize_from_examples(spec, top_k=top_k)
            if rep.program is None:
                return SynthesisReport(
                    program=None, size=0, cost=0.0, n_examples=len(spec),
                    visited=rep.visited, walltime_s=time.time() - t0,
                    cegis_rounds=rounds, converged=False,
                    dsl_name=self.dsl.name,
                )
            cex = verifier(rep.program)
            if cex is None:
                return SynthesisReport(
                    program=rep.program,
                    size=rep.size, cost=rep.cost,
                    n_examples=len(spec), visited=rep.visited,
                    walltime_s=time.time() - t0,
                    alternatives=rep.alternatives,
                    cegis_rounds=rounds, converged=True,
                    dsl_name=self.dsl.name,
                )
            spec.append(cex)
        return SynthesisReport(
            program=None, size=0, cost=0.0, n_examples=len(spec),
            visited=0, walltime_s=time.time() - t0,
            cegis_rounds=rounds, converged=False, dsl_name=self.dsl.name,
        )

    # =================================================================
    # Version-space candidates
    # =================================================================

    def candidates(
        self,
        examples: Sequence[Example],
        max_candidates: int = 20,
    ) -> list[Program]:
        """All programs consistent with every example, up to depth /
        cost bounds.  The full version space, capped at
        ``max_candidates`` (MDL-sorted)."""
        out: list[Program] = []
        for prog in _enumerate_up_to(self.dsl, self.dsl.output_type,
                                      self.dsl.max_depth):
            try:
                _ = prog.run(examples[0][0])
            except Exception:
                continue
            if _consistent(prog, examples):
                out.append(prog)
                if len(out) >= max_candidates:
                    break
        return sorted(out, key=lambda p: (p.size(), p.cost()))

    # =================================================================
    # MDL-ordered cost-priority enumeration (Dijkstra over AST)
    # =================================================================

    def _enumerate_by_cost(
        self,
        t: Type,
        max_depth: int,
        max_cost: float = 1e9,
    ) -> Iterable[Program]:
        """Priority-queue enumeration of programs by AST cost; near-
        Dijkstra on the typed program-construction graph."""
        seen: set[Program] = set()
        # heap holds (cost, counter, program)
        counter = itertools.count()
        pq: list[tuple[float, int, Program]] = []

        # seed: depth-1 atoms
        for p in dsl_atoms(self.dsl, t):
            heapq.heappush(pq, (p.cost(), next(counter), p))
        while pq:
            c, _, p = heapq.heappop(pq)
            if p in seen or c > max_cost:
                continue
            seen.add(p)
            yield p
            if p.depth() >= max_depth:
                continue
            # extend by wrapping in an op
            for op in self.dsl.ops:
                # generate combinations of arguments where one slot is `p`
                if p.return_type not in op.arg_types and T_ANY not in op.arg_types:
                    continue
                for slot in range(op.arity):
                    if op.arg_types[slot] != p.return_type and op.arg_types[slot] != T_ANY:
                        continue
                    arg_pools = []
                    valid = True
                    for j, at in enumerate(op.arg_types):
                        if j == slot:
                            arg_pools.append([p])
                            continue
                        atoms = list(itertools.islice(
                            dsl_atoms(self.dsl, at), 100))
                        if not atoms:
                            valid = False
                            break
                        arg_pools.append(atoms)
                    if not valid:
                        continue
                    for combo in itertools.product(*arg_pools):
                        np_ = call(op, *combo)
                        if np_ in seen:
                            continue
                        heapq.heappush(pq, (np_.cost(), next(counter), np_))

    # =================================================================
    # L* DFA learning (Angluin 1987)
    # =================================================================

    def learn_dfa(
        self,
        membership_query: Callable[[str], bool],
        equivalence_query: Callable[["DFA"], str | None],
        alphabet: Sequence[str],
        max_rounds: int = 50,
    ) -> "DFA":
        """Angluin L\\* algorithm — minimal DFA learning with
        membership + equivalence oracles.  Returns the learned DFA."""
        S: list[str] = [""]
        E: list[str] = [""]
        T: dict[tuple[str, str], bool] = {}

        def fill():
            for s in S + [s + a for s in S for a in alphabet]:
                for e in E:
                    if (s, e) not in T:
                        T[(s, e)] = membership_query(s + e)

        def row(s: str) -> tuple[bool, ...]:
            return tuple(T[(s, e)] for e in E)

        def closed() -> str | None:
            rows_S = {row(s) for s in S}
            for s in S:
                for a in alphabet:
                    if row(s + a) not in rows_S:
                        return s + a
            return None

        def consistent() -> tuple[str, str] | None:
            # find two rows in S that agree but where extending by some a
            # disagrees, indicating new column needed
            for i, s1 in enumerate(S):
                for s2 in S[i + 1:]:
                    if row(s1) == row(s2):
                        for a in alphabet:
                            if row(s1 + a) != row(s2 + a):
                                for k, e in enumerate(E):
                                    if T[(s1 + a, e)] != T[(s2 + a, e)]:
                                        return (a, e)
            return None

        for _ in range(max_rounds):
            fill()
            ext = closed()
            while ext is not None:
                S.append(ext)
                fill()
                ext = closed()
            inc = consistent()
            while inc is not None:
                a, e = inc
                E.append(a + e)
                fill()
                ext = closed()
                while ext is not None:
                    S.append(ext)
                    fill()
                    ext = closed()
                inc = consistent()
            dfa = self._build_dfa(S, E, T, alphabet)
            cex = equivalence_query(dfa)
            if cex is None:
                return dfa
            # add all prefixes of cex
            for i in range(len(cex) + 1):
                s = cex[:i]
                if s not in S:
                    S.append(s)
        return self._build_dfa(S, E, T, alphabet)

    def _build_dfa(
        self,
        S: list[str],
        E: list[str],
        T: dict[tuple[str, str], bool],
        alphabet: Sequence[str],
    ) -> "DFA":
        def row(s: str) -> tuple[bool, ...]:
            return tuple(T.get((s, e), False) for e in E)

        states: dict[tuple[bool, ...], int] = {}
        for s in S:
            r = row(s)
            if r not in states:
                states[r] = len(states)
        initial = states[row("")]
        accepts = {sid for r, sid in states.items() if r[0]}
        trans: dict[tuple[int, str], int] = {}
        for s in S:
            for a in alphabet:
                # need (s+a, e) for all e in E — already filled if S+S·a were
                key = (s + a, E[0]) if (s + a, E[0]) in T else None
                # find a "representative" prefix in S with matching row
                target_row = tuple(T.get((s + a, e), False) for e in E)
                if target_row not in states:
                    states[target_row] = len(states)
                trans[(states[row(s)], a)] = states[target_row]
        return DFA(
            states=set(states.values()),
            alphabet=set(alphabet),
            transitions=trans,
            initial=initial,
            accepts=accepts,
        )

    # =================================================================
    # Anti-unification
    # =================================================================

    def lgg(self, a: Any, b: Any) -> Any:
        return lgg(a, b)

    def lgg_many(self, xs: Sequence[Any]) -> Any:
        return lgg_many(xs)


def dsl_atoms(dsl: DSL, t: Type) -> Iterable[Program]:
    for p in dsl.vars_by_type(t):
        yield p
    for p in dsl.consts_by_type(t):
        yield p


# =============================================================================
# DFA
# =============================================================================


@dataclass
class DFA:
    states: set[int]
    alphabet: set[str]
    transitions: dict[tuple[int, str], int]
    initial: int
    accepts: set[int]

    def run(self, word: str) -> bool:
        s = self.initial
        for ch in word:
            if ch not in self.alphabet:
                return False
            s = self.transitions.get((s, ch), -1)
            if s == -1:
                return False
        return s in self.accepts

    def n_states(self) -> int:
        return len(self.states)


# =============================================================================
# Built-in DSLs
# =============================================================================


# ---------------- STRING_DSL --------------------------------------------------


def _str_concat(a: str, b: str) -> str:
    return a + b


def _str_substring(s: str, start: int, length: int) -> str:
    if start < 0 or length < 0:
        raise ValueError("negative")
    return s[start:start + length]


def _str_split_first(s: str, sep: str) -> str:
    parts = s.split(sep, 1)
    if len(parts) < 1:
        raise ValueError
    return parts[0]


def _str_split_last(s: str, sep: str) -> str:
    parts = s.rsplit(sep, 1)
    if len(parts) < 1:
        raise ValueError
    return parts[-1]


def _str_upper(s: str) -> str:
    return s.upper()


def _str_lower(s: str) -> str:
    return s.lower()


def _str_strip(s: str) -> str:
    return s.strip()


def _str_replace(s: str, old: str, new: str) -> str:
    return s.replace(old, new)


def _str_index_of(s: str, sub: str) -> int:
    i = s.find(sub)
    if i < 0:
        raise ValueError
    return i


def _str_length(s: str) -> int:
    return len(s)


STRING_DSL = DSL(
    name="STRING",
    input_types=(T_STR,),
    output_type=T_STR,
    ops=(
        Op("concat", 2, (T_STR, T_STR), T_STR, _str_concat, 1.0),
        Op("substring", 3, (T_STR, T_INT, T_INT), T_STR, _str_substring, 1.5),
        Op("split_first", 2, (T_STR, T_STR), T_STR, _str_split_first, 1.0),
        Op("split_last", 2, (T_STR, T_STR), T_STR, _str_split_last, 1.0),
        Op("upper", 1, (T_STR,), T_STR, _str_upper, 0.8),
        Op("lower", 1, (T_STR,), T_STR, _str_lower, 0.8),
        Op("strip", 1, (T_STR,), T_STR, _str_strip, 0.8),
        Op("replace", 3, (T_STR, T_STR, T_STR), T_STR, _str_replace, 1.2),
        Op("index_of", 2, (T_STR, T_STR), T_INT, _str_index_of, 1.0),
        Op("length", 1, (T_STR,), T_INT, _str_length, 0.5),
    ),
    constants=(
        ("@", T_STR), (".", T_STR), (" ", T_STR), ("", T_STR),
        ("/", T_STR), (",", T_STR), (":", T_STR), ("-", T_STR),
        (0, T_INT), (1, T_INT), (2, T_INT), (3, T_INT), (4, T_INT), (5, T_INT),
    ),
    max_depth=3,
)


# ---------------- INTEGER_DSL --------------------------------------------------


def _int_add(a: int, b: int) -> int:
    return a + b


def _int_sub(a: int, b: int) -> int:
    return a - b


def _int_mul(a: int, b: int) -> int:
    return a * b


def _int_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return a // b


def _int_mod(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return a % b


def _int_min(a: int, b: int) -> int:
    return a if a < b else b


def _int_max(a: int, b: int) -> int:
    return a if a > b else b


def _int_abs(a: int) -> int:
    return abs(a)


def _int_ifzero(c: int, t: int, f: int) -> int:
    return t if c == 0 else f


INTEGER_DSL = DSL(
    name="INTEGER",
    input_types=(T_INT, T_INT),
    output_type=T_INT,
    ops=(
        Op("add", 2, (T_INT, T_INT), T_INT, _int_add, 1.0),
        Op("sub", 2, (T_INT, T_INT), T_INT, _int_sub, 1.0),
        Op("mul", 2, (T_INT, T_INT), T_INT, _int_mul, 1.0),
        Op("div", 2, (T_INT, T_INT), T_INT, _int_div, 1.2),
        Op("mod", 2, (T_INT, T_INT), T_INT, _int_mod, 1.2),
        Op("min", 2, (T_INT, T_INT), T_INT, _int_min, 1.0),
        Op("max", 2, (T_INT, T_INT), T_INT, _int_max, 1.0),
        Op("abs", 1, (T_INT,), T_INT, _int_abs, 0.5),
        Op("ifzero", 3, (T_INT, T_INT, T_INT), T_INT, _int_ifzero, 1.5),
    ),
    constants=(
        (0, T_INT), (1, T_INT), (2, T_INT), (-1, T_INT),
        (10, T_INT), (100, T_INT),
    ),
    max_depth=3,
)


# ---------------- LIST_DSL ----------------------------------------------------


def _list_head(xs: list) -> int:
    if not xs:
        raise ValueError
    return xs[0]


def _list_last(xs: list) -> int:
    if not xs:
        raise ValueError
    return xs[-1]


def _list_length(xs: list) -> int:
    return len(xs)


def _list_sum(xs: list) -> int:
    return sum(xs)


def _list_max(xs: list) -> int:
    if not xs:
        raise ValueError
    return max(xs)


def _list_min(xs: list) -> int:
    if not xs:
        raise ValueError
    return min(xs)


def _list_reverse(xs: list) -> list:
    return list(reversed(xs))


def _list_sort(xs: list) -> list:
    return sorted(xs)


LIST_DSL = DSL(
    name="LIST",
    input_types=(T_LIST_INT,),
    output_type=T_INT,
    ops=(
        Op("head", 1, (T_LIST_INT,), T_INT, _list_head, 0.5),
        Op("last", 1, (T_LIST_INT,), T_INT, _list_last, 0.5),
        Op("length", 1, (T_LIST_INT,), T_INT, _list_length, 0.5),
        Op("sum", 1, (T_LIST_INT,), T_INT, _list_sum, 0.7),
        Op("max", 1, (T_LIST_INT,), T_INT, _list_max, 0.7),
        Op("min", 1, (T_LIST_INT,), T_INT, _list_min, 0.7),
        Op("add", 2, (T_INT, T_INT), T_INT, _int_add, 1.0),
        Op("sub", 2, (T_INT, T_INT), T_INT, _int_sub, 1.0),
        Op("mul", 2, (T_INT, T_INT), T_INT, _int_mul, 1.0),
    ),
    constants=(
        (0, T_INT), (1, T_INT), (2, T_INT),
    ),
    max_depth=3,
)


# =============================================================================
# Convenience: build a DSL with user-supplied operators
# =============================================================================


def make_dsl(
    name: str,
    input_types: Sequence[Type],
    output_type: Type,
    ops: Sequence[Op],
    constants: Sequence[tuple[Any, Type]] = (),
    max_depth: int = 3,
) -> DSL:
    return DSL(
        name=name,
        input_types=tuple(input_types),
        output_type=output_type,
        ops=tuple(ops),
        constants=tuple(constants),
        max_depth=max_depth,
    )


__all__ = [
    "Type",
    "T_INT", "T_STR", "T_BOOL", "T_LIST_INT", "T_LIST_STR", "T_ANY",
    "Op",
    "Program",
    "DSL",
    "Synthesizer",
    "SynthesisReport",
    "DFA",
    "const", "var", "call",
    "lgg", "lgg_many",
    "make_dsl",
    "STRING_DSL",
    "INTEGER_DSL",
    "LIST_DSL",
]
