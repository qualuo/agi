r"""Refuter — automated falsification as a runtime primitive.

Every other primitive in this runtime *makes claims*: Synthesizer says
"this program is correct on the spec", Forecaster says "this
distribution is calibrated", Sampler says "this is the posterior",
Cartographer says "this task is in the learner's frontier",
ConformalPredictor says "this interval covers", Submodular says "this
is a 1-1/e approximation".  The Refuter is the primitive that *tries
to break them* — Popperian conjecture-and-refutation as a runtime
mechanism.

The pitch
---------

A hypothesis ``H: X → bool`` is *refuted* by a single ``x ∈ X`` for
which ``H(x) = False``.  Refutation is asymmetric: one witness destroys
the hypothesis, while no number of confirmations proves it — they only
constrain the *rate* of failure.  Refuter operationalises this with
three modes:

  * **try_refute(H, space)** — hunt counterexamples in ``space`` using a
    portfolio of search strategies; return either a witness, or a
    quantified support claim ``Pr[H fails] ≤ ε``.

  * **try_refute_relation(f, relation, space)** — metamorphic mode.
    Many functions have no oracle but obey relations like
    ``sort(reverse(L)) == sort(L)`` or ``f(g(x)) == h(f(x))``.  Refute
    a *relation* between calls.

  * **try_refute_bound(scalar, threshold, space)** — refute a claim
    ``∀ x ∈ space: scalar(x) ≤ threshold`` (or ``≥``).  When the bound
    is tight, evolutionary search drives toward the boundary; the
    optimisation log doubles as a tightness audit.

Plus two compositional moves that turn it into a true epistemic engine:

  * **shrink(witness, space, H)** — QuickCheck-style minimisation.
    Given a refuting input, halve toward zero / shorten / drop fields
    while ``H`` still fails, returning the *minimal* counterexample
    found.  Minimal witnesses are dramatically more useful for
    diagnosis.

  * **refute_until(H, space, alpha)** — sequential mode with an
    anytime-valid e-process on accumulated trials.  ``Pr_{H₀}(∃t:
    e_t ≥ 1/α) ≤ α`` (Ville's inequality) so the user can keep
    pulling more samples and the support / refutation decision is
    valid under *any* stopping rule.

Mathematical and algorithmic roots
----------------------------------

  * **Popper, K. (1959) — *The Logic of Scientific Discovery*.**  The
    asymmetry between confirmation and falsification: a universal
    claim ``∀x: H(x)`` is logically refuted by one witness, but no
    finite confirmation set verifies it.  Refuter encodes this
    asymmetry directly: a single counterexample *closes* the search;
    confirmations only tighten the upper bound on the violation rate.

  * **Hughes, J. & Claessen, K. (2000) — *QuickCheck: A Lightweight
    Tool for Random Testing of Haskell Programs* (ICFP).**  Generators
    over typed search spaces; random sampling of properties;
    *shrinking* a failing witness to a minimal one by structural
    recursion.  Refuter ships Hughes-Claessen shrinking for every
    built-in space and exposes a `shrink_step` protocol so user spaces
    plug in.

  * **Chen, T. Y., Cheung, S. C., Yiu, S. M. (1998) — *Metamorphic
    testing: a new approach for generating next test cases* (HKUST
    TR-98-01).**  Test programs without oracles by validating
    *metamorphic relations* — algebraic identities the program must
    satisfy.  Refuter's `try_refute_relation` is exactly this: define
    transforms ``(t_in, t_out)`` such that
    ``t_out(f(x)) = f(t_in(x))`` and refute by counterexample to the
    identity.

  * **Solar-Lezama, A. (2008) — *Program Synthesis by Sketching*.**
    *Counterexample-guided inductive synthesis* (CEGIS) is the
    refute-then-resynthesise loop.  Refuter ships the *refute* half;
    Synthesizer ships the *synthesise* half; together they realise
    CEGIS over any DSL with no glue beyond a function call.

  * **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A.
    (2018) — *Towards Deep Learning Models Resistant to Adversarial
    Attacks* (ICLR).**  PGD: projected gradient descent on a
    *margin* objective drives inputs toward the decision boundary.
    Refuter generalises this to gradient-free settings (the evaluator
    is just a callable predicate; no autograd) via numerical
    one-sided finite differences and an evolutionary (1+λ) search on
    the *satisfaction margin* — the signed amount by which ``H``
    is satisfied, where the boundary lives at zero.

  * **Hansen, N. & Ostermeier, A. (2001) — *Completely derandomized
    self-adaptation in evolution strategies* (Evol. Comput.).**  Step-
    size adaptation in derivative-free optimisation: the proposal
    variance grows when offspring beat the parent and shrinks when
    they don't.  Refuter ships the *1/5-success-rule* (Rechenberg
    1973) — a stripped-down variant that needs no covariance matrix.

  * **Rechenberg, I. (1973) — *Evolutionsstrategie: Optimierung
    technischer Systeme nach Prinzipien der biologischen Evolution*.**
    The original 1/5-rule for adapting step size in evolution
    strategies: if more than 1/5 of mutations improve fitness, grow
    σ; else shrink.

  * **Clopper, C. J. & Pearson, E. S. (1934) — *The use of confidence
    or fiducial limits illustrated in the case of the binomial*.**
    Exact (non-asymptotic) confidence intervals on a Bernoulli rate.
    For ``k = 0`` failures in ``n`` trials, the (1-α) one-sided upper
    bound on the failure rate is
        p̄ = 1 - α^{1/n}.
    Refuter reports this on every support claim — the *quantified*
    Popperian statement "we tried n times and ε-failed-bound is p̄".

  * **Hanley, J. A. & Lippman-Hand, A. (1983) — *If nothing goes wrong,
    is everything alright?  Interpreting zero numerators* (JAMA).**
    The famous "rule of three": with ``k=0`` failures in ``n``
    trials, the 95% upper bound on ``p`` is ``≈ 3/n``.  This is the
    Taylor expansion of Clopper-Pearson at ``α = 0.05`` and gives
    the headline interpretation Refuter prints for non-statistical
    consumers.

  * **Wald, A. (1945) — *Sequential tests of statistical
    hypotheses*.**  The sequential probability ratio test (SPRT)
    accumulates evidence via a likelihood ratio with bounded type-I
    / type-II error under *any* stopping rule.  Refuter's anytime-
    valid e-process for binomial calibration is the modern Vovk-
    Wang reformulation of the same idea.

  * **Vovk, V. & Wang, R. (2021) — *E-values: calibration, combination
    and applications* (Ann. Statist.).**  An *e-value* is a non-
    negative test statistic with ``E[E | H₀] ≤ 1``.  Ville's
    inequality gives ``Pr_{H₀}(∃t: E_t ≥ 1/α) ≤ α`` for every
    stopping rule.  Refuter accumulates an e-value over Bernoulli
    refutation outcomes via *betting* against the null calibration
    rate; the user can run as long as desired and reject at the
    first ``E_t ≥ 1/α``.

  * **King, J. C. (1976) — *Symbolic execution and program testing*
    (Comm. ACM).**  Symbolic execution explores paths in a program by
    tracking *constraints*; concolic execution mixes symbolic with
    concrete inputs.  Refuter's *boundary corner enumeration* is a
    pragmatic stdlib-only variant: for each numeric coordinate, try
    ``{lo, lo+ε, mid, hi-ε, hi}`` and zero-crossings — covers the
    *common* corners without symbolic-execution machinery.

  * **Halton, J. H. (1960) — *On the efficiency of certain quasi-
    random sequences of points in evaluating multi-dimensional
    integrals*.**  Quasi-random low-discrepancy sequences cover a
    hypercube more uniformly than i.i.d. uniform, giving better
    coverage per sample.  Refuter ships a stdlib Halton sequence for
    the *random* strategy and falls back to Mersenne-Twister uniform
    when the space has no continuous coordinate.

  * **Nelder, J. A. & Mead, R. (1965) — *A simplex method for function
    minimization*.**  Derivative-free local search via reflect /
    expand / contract / shrink on a simplex of size ``n+1``.  Refuter
    ships a stdlib Nelder-Mead minimiser of the satisfaction margin
    for low-dimensional continuous searches as an alternative to
    (1+λ) ES.

  * **Goldberg, D. (1991) — *What every computer scientist should know
    about floating-point arithmetic*.**  IEEE-754 corners — ±0,
    subnormals, NaN, ±∞, ULP-boundaries — are notorious refutation
    seeds for numerical predicates.  Refuter enumerates these by
    default for any ``ContinuousSpace``.

Public API
----------

::

    >>> from agi.refuter import Refuter, ContinuousSpace, ProductSpace
    >>> R = Refuter(seed=0)
    >>> # Hypothesis: x² ≥ x for all real x in [-3, 3].  False at x ∈ (0, 1).
    >>> def H(x): return x["v"] ** 2 >= x["v"]
    >>> rep = R.try_refute(H, ProductSpace(v=ContinuousSpace(-3.0, 3.0)),
    ...                    n_trials=5_000, alpha=0.05)
    >>> rep.refuted, rep.counterexample.x        # ⇒ True, {"v": 0.4...}

Metamorphic mode::

    >>> # Sorting twice equals sorting once.
    >>> def f(L): return sorted(L)
    >>> def relation_holds(x, fx, x2, fx2): return fx2 == fx
    >>> from agi.refuter import ListSpace, IntegerSpace
    >>> S = ListSpace(elem=IntegerSpace(0, 100), max_len=8)
    >>> rep = R.try_refute_relation(
    ...     f, relation_holds, S, x_to_x2=lambda L: sorted(L), n_trials=2_000)

Bound mode (drive toward the boundary)::

    >>> def f(x): return x["v"] ** 2 - 4 * x["v"] + 3  # parabola, min = -1 at x=2
    >>> rep = R.try_refute_bound(f, threshold=-2.0, direction="<=",
    ...                          space=ProductSpace(v=ContinuousSpace(0.0, 5.0)),
    ...                          n_trials=2_000)
    >>> # Bound -2 is below the minimum -1, so unrefutable — rep.refuted is False
    >>> rep.tightness_margin   # how close did we get?  (positive = bound holds)

Shrinking a witness::

    >>> small = R.shrink(witness=rep.counterexample, space=S, H=H)

Composition with the rest of the runtime
----------------------------------------

  * **Synthesizer** — closes the CEGIS loop: ``while (cex := R.try_refute(
    prog, space)).refuted: prog = synthesizer.cegis_round(prog, cex)``.
  * **Forecaster** — refute calibration claims via metamorphic relations
    on the PIT (rank histogram uniformity).
  * **Sampler** — posterior-predictive cross-checks: refute the
    null that observed moments fall inside the posterior-predictive
    quantiles.
  * **ConformalPredictor** — coverage stress-test on adversarial
    held-out points.
  * **CausalDiscoverer** — refute conditional-independence claims by
    finding score-violating witnesses.
  * **AttestationLedger** — every `RefutationReport` carries a
    SHA-256 fingerprint over `(predicate signature, space, seed,
    witnesses)` for tamper-evident replay.
  * **Auditor** — multiple Refuter calls feed e-values into
    Auditor for FDR-controlled multiple-refutation control.
  * **AutonomousLoop** — every plan's preconditions can be refuted
    before action; every action's post-condition can be refuted
    before commit.

The primitive is deliberately conservative: every search is bounded by
``n_trials`` and ``walltime``; statistical claims report a *finite-
sample* upper bound on the failure rate.  Pure stdlib — no numpy, no
SciPy, no Z3.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


# =============================================================================
# Errors
# =============================================================================


class RefuterError(Exception):
    """Base class for Refuter-raised exceptions."""


class InvalidSpace(RefuterError):
    """A search space is malformed (lo > hi, empty, mistyped)."""


class InvalidHypothesis(RefuterError):
    """A predicate / margin function returned something unusable."""


class BudgetExhausted(RefuterError):
    """``n_trials`` reached without finding a counterexample (raised only
    in `refute_or_raise`)."""


# =============================================================================
# Search spaces
# =============================================================================


@dataclass(frozen=True)
class ContinuousSpace:
    """A closed real interval ``[lo, hi]`` with IEEE-754 corner support."""
    lo: float
    hi: float
    include_corners: bool = True

    def __post_init__(self) -> None:
        if not (self.lo <= self.hi):
            raise InvalidSpace(f"ContinuousSpace lo={self.lo} > hi={self.hi}")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.lo, self.hi)

    def boundary(self) -> list[float]:
        """IEEE-754 / interval corners — common refutation seeds."""
        lo, hi = float(self.lo), float(self.hi)
        mid = 0.5 * (lo + hi)
        eps = max(abs(lo), abs(hi), 1.0) * 1e-9
        cands: list[float] = [lo, hi, mid, lo + eps, hi - eps]
        # zero-cross if interval straddles zero
        if lo < 0.0 < hi:
            cands += [0.0, -eps, eps]
        # IEEE-754 corners (only if in range — NaN / inf are always
        # potential refutation seeds for numeric predicates)
        if self.include_corners:
            for sp in (float("inf"), float("-inf"), float("nan")):
                cands.append(sp)
        return cands

    def mutate(self, x: float, sigma: float, rng: random.Random) -> float:
        """Gaussian step, clamped to [lo, hi]."""
        y = x + sigma * rng.gauss(0.0, 1.0)
        if y < self.lo:
            y = self.lo
        if y > self.hi:
            y = self.hi
        return y

    def cardinality(self) -> float:
        return float("inf")

    def to_dict(self) -> dict:
        return {
            "kind": "continuous", "lo": self.lo, "hi": self.hi,
            "include_corners": self.include_corners,
        }


@dataclass(frozen=True)
class IntegerSpace:
    """A closed integer interval ``[lo, hi]``."""
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not (self.lo <= self.hi):
            raise InvalidSpace(f"IntegerSpace lo={self.lo} > hi={self.hi}")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.lo, self.hi)

    def boundary(self) -> list[int]:
        out = {self.lo, self.hi, (self.lo + self.hi) // 2}
        if self.lo <= 0 <= self.hi:
            out.add(0)
        if self.lo <= 1 <= self.hi:
            out.add(1)
        if self.lo <= -1 <= self.hi:
            out.add(-1)
        return sorted(out)

    def mutate(self, x: int, sigma: float, rng: random.Random) -> int:
        step = int(round(sigma * rng.gauss(0.0, 1.0)))
        if step == 0:
            step = rng.choice([-1, 1])
        y = x + step
        if y < self.lo:
            y = self.lo
        if y > self.hi:
            y = self.hi
        return y

    def cardinality(self) -> float:
        return float(self.hi - self.lo + 1)

    def to_dict(self) -> dict:
        return {"kind": "integer", "lo": self.lo, "hi": self.hi}


@dataclass(frozen=True)
class FiniteSet:
    """A finite ordered set of values."""
    values: tuple = ()

    def __post_init__(self) -> None:
        if len(self.values) == 0:
            raise InvalidSpace("FiniteSet must be non-empty")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def boundary(self) -> list[Any]:
        # first, last, middle
        n = len(self.values)
        if n == 1:
            return [self.values[0]]
        return [self.values[0], self.values[-1], self.values[n // 2]]

    def mutate(self, x: Any, sigma: float, rng: random.Random) -> Any:
        # ignore sigma — pick a different value with probability matched to sigma
        if rng.random() < min(1.0, max(sigma, 0.1)):
            return rng.choice(self.values)
        return x

    def cardinality(self) -> float:
        return float(len(self.values))

    def to_dict(self) -> dict:
        # repr the values, since JSON can't always round-trip them
        return {"kind": "finite", "values": [repr(v) for v in self.values]}


@dataclass(frozen=True)
class BoolSpace:
    """Just `{False, True}`."""

    def sample(self, rng: random.Random) -> bool:
        return rng.random() < 0.5

    def boundary(self) -> list[bool]:
        return [False, True]

    def mutate(self, x: bool, sigma: float, rng: random.Random) -> bool:
        if rng.random() < min(1.0, max(sigma, 0.2)):
            return not x
        return x

    def cardinality(self) -> float:
        return 2.0

    def to_dict(self) -> dict:
        return {"kind": "bool"}


@dataclass(frozen=True)
class ListSpace:
    """Variable-length list with elements drawn from ``elem``."""
    elem: Any  # a Space
    min_len: int = 0
    max_len: int = 8

    def __post_init__(self) -> None:
        if self.min_len < 0 or self.max_len < self.min_len:
            raise InvalidSpace(f"ListSpace bounds: {self.min_len}..{self.max_len}")

    def sample(self, rng: random.Random) -> list:
        n = rng.randint(self.min_len, self.max_len)
        return [self.elem.sample(rng) for _ in range(n)]

    def boundary(self) -> list[list]:
        out: list[list] = []
        # empty (if allowed)
        if self.min_len == 0:
            out.append([])
        # singleton with each elem-boundary
        for v in self.elem.boundary():
            out.append([v])
        # full of first-boundary
        first = self.elem.boundary()[0]
        out.append([first] * max(self.min_len, 1))
        if self.max_len > self.min_len:
            out.append([first] * self.max_len)
        return out

    def mutate(self, x: list, sigma: float, rng: random.Random) -> list:
        # one of three local moves with prob ∝ sigma:
        #   - mutate one element
        #   - append/pop (length jiggle)
        #   - swap two elements
        if len(x) == 0:
            if self.max_len > 0:
                return [self.elem.sample(rng)]
            return x
        y = list(x)
        r = rng.random()
        if r < 0.5 and len(y) > 0:
            i = rng.randrange(len(y))
            y[i] = self.elem.mutate(y[i], sigma, rng)
        elif r < 0.8:
            # length jiggle
            if len(y) > self.min_len and rng.random() < 0.5:
                y.pop(rng.randrange(len(y)))
            elif len(y) < self.max_len:
                y.insert(rng.randrange(len(y) + 1), self.elem.sample(rng))
        else:
            if len(y) >= 2:
                i, j = rng.sample(range(len(y)), 2)
                y[i], y[j] = y[j], y[i]
        return y

    def cardinality(self) -> float:
        ec = self.elem.cardinality()
        if math.isinf(ec):
            return float("inf")
        # sum of ec^k for k in [min, max]
        try:
            return sum(ec ** k for k in range(self.min_len, self.max_len + 1))
        except OverflowError:
            return float("inf")

    def to_dict(self) -> dict:
        return {
            "kind": "list", "elem": self.elem.to_dict(),
            "min_len": self.min_len, "max_len": self.max_len,
        }


@dataclass(frozen=True)
class ProductSpace:
    """Named product of subspaces.  Inputs are dicts."""
    children: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.children) == 0:
            raise InvalidSpace("ProductSpace must have at least one child")

    def sample(self, rng: random.Random) -> dict:
        return {k: s.sample(rng) for k, s in self.children.items()}

    def boundary(self) -> list[dict]:
        # Cartesian boundary corners would blow up — instead, emit one
        # boundary per coordinate axis with others at their *first* boundary.
        out: list[dict] = []
        keys = list(self.children.keys())
        defaults = {k: self.children[k].boundary()[0] for k in keys}
        # all-at-default
        out.append(dict(defaults))
        # one axis at a time
        for k in keys:
            for v in self.children[k].boundary():
                d = dict(defaults)
                d[k] = v
                out.append(d)
        return out

    def mutate(self, x: dict, sigma: float, rng: random.Random) -> dict:
        # mutate one random coordinate
        y = dict(x)
        keys = list(self.children.keys())
        k = rng.choice(keys)
        y[k] = self.children[k].mutate(y[k], sigma, rng)
        return y

    def cardinality(self) -> float:
        total = 1.0
        for s in self.children.values():
            c = s.cardinality()
            if math.isinf(c):
                return float("inf")
            total *= c
        return total

    def to_dict(self) -> dict:
        return {
            "kind": "product",
            "children": {k: s.to_dict() for k, s in self.children.items()},
        }


# Convenience constructor — clearer call site than `ProductSpace(children={...})`.
def Product(**children) -> ProductSpace:  # noqa: N802
    return ProductSpace(children=dict(children))


# =============================================================================
# Halton low-discrepancy sequence (Halton 1960)
# =============================================================================


def _halton(i: int, base: int) -> float:
    """The i-th element of the radical-inverse sequence in base ``base``."""
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def _halton_continuous(space: ContinuousSpace, idx: int, axis: int) -> float:
    """Map Halton(idx, prime[axis]) to [lo, hi]."""
    base = _HALTON_PRIMES[axis % len(_HALTON_PRIMES)]
    u = _halton(idx + 1, base)
    return space.lo + (space.hi - space.lo) * u


# =============================================================================
# Predicate kinds
# =============================================================================


@dataclass(frozen=True)
class Evaluation:
    """The outcome of evaluating a hypothesis at one input."""
    ok: bool                # True if the predicate held
    margin: float           # signed margin (>=0 means held; smaller = closer to violation)
    raw: Any = None         # the raw return value of the user's function (for debugging)
    error: str | None = None


def _coerce_to_eval(out: Any) -> Evaluation:
    """Accept any of:
      * bool                    →  (ok=bool, margin=+1 / -1)
      * (bool, float) tuple     →  (ok=bool, margin=float)
      * float                   →  (ok=margin>=0, margin=float)
      * Evaluation              →  as-is
    """
    if isinstance(out, Evaluation):
        return out
    if isinstance(out, bool):
        return Evaluation(ok=out, margin=1.0 if out else -1.0, raw=out)
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        m = float(out)
        return Evaluation(ok=(m >= 0.0), margin=m, raw=out)
    if isinstance(out, tuple) and len(out) == 2:
        a, b = out
        if isinstance(a, bool) and isinstance(b, (int, float)):
            return Evaluation(ok=a, margin=float(b), raw=out)
    raise InvalidHypothesis(
        f"Hypothesis must return bool, float, (bool, float), or Evaluation; "
        f"got {type(out).__name__}: {out!r}"
    )


# =============================================================================
# Counterexample
# =============================================================================


@dataclass(frozen=True)
class Counterexample:
    """A single refuting witness."""
    x: Any
    margin: float       # the (negative) margin at which H failed
    strategy: str       # which strategy found it ("random", "boundary", "evolutionary", "shrink", "metamorphic", "bound")
    trial: int          # which trial number (1-indexed)
    raw: Any = None     # the predicate's raw return for diagnostics

    def to_dict(self) -> dict:
        return {
            "x": _safe_jsonable(self.x),
            "margin": self.margin,
            "strategy": self.strategy,
            "trial": self.trial,
            "raw": _safe_jsonable(self.raw),
        }


def _safe_jsonable(v: Any) -> Any:
    """Best-effort JSON-able rendering of any value, for fingerprinting."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_jsonable(x) for k, x in v.items()}
    return repr(v)


# =============================================================================
# Statistical primitives
# =============================================================================


def clopper_pearson_zero_ucb(n: int, alpha: float) -> float:
    """For ``k = 0`` failures in ``n`` Bernoulli trials, return the
    one-sided (1-α) Clopper-Pearson upper bound on the failure rate:
        p̄ = 1 - α^{1/n}
    With ``n=0`` returns 1.0 (no information).
    """
    if n <= 0:
        return 1.0
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return 1.0 - alpha ** (1.0 / n)


def rule_of_three(n: int) -> float:
    """The rule of three (Hanley-Lippman-Hand 1983): with 0 failures in
    ``n`` trials, the 95% upper bound on the failure rate is ``≈ 3/n``."""
    if n <= 0:
        return 1.0
    return 3.0 / n


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _regularized_inc_beta(x: float, a: float, b: float, max_iter: int = 200,
                          eps: float = 1e-12) -> float:
    """Regularised incomplete beta I_x(a, b) via the Lentz continued
    fraction (Numerical Recipes 6.4).  Pure stdlib."""
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    # symmetry: use I_{1-x}(b, a) = 1 - I_x(a, b) for faster convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_inc_beta(1.0 - x, b, a, max_iter, eps)
    # prefactor
    lbeta = _log_beta(a, b)
    log_prefac = a * math.log(x) + b * math.log(1.0 - x) - lbeta - math.log(a)
    # Lentz CF
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(log_prefac) * h


def clopper_pearson_ucb(k: int, n: int, alpha: float) -> float:
    """One-sided (1-α) upper bound on the binomial rate with ``k``
    successes in ``n`` trials (Clopper-Pearson 1934).

    Exact: p̄ is the solution of ``∑_{j=0}^{k} C(n,j) p̄^j (1-p̄)^{n-j} = α``.
    We invert via the beta-binomial identity:
        p̄ = BetaInv(1-α; k+1, n-k).
    With ``k = n`` returns 1.0; with ``k = 0`` returns 1 - α^{1/n}.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of [0, n={n}]")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    # bisect on p where I_p(k+1, n-k) = 1 - α
    target = 1.0 - alpha
    a = float(k + 1)
    b = float(n - k)
    lo, hi = 0.0, 1.0
    # 60 bisections ≈ 1e-18 — pin to ≤1e-10
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _regularized_inc_beta(mid, a, b) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def hoeffding_ucb(mean: float, n: int, alpha: float, lo: float = 0.0,
                  hi: float = 1.0) -> float:
    """One-sided (1-α) Hoeffding upper confidence bound on a [lo, hi]-
    bounded random variable's mean from ``n`` samples with empirical mean
    ``mean``::

        mean + (hi - lo) * sqrt( ln(1/α) / (2n) )
    """
    if n <= 0:
        return hi
    width = (hi - lo) * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return mean + width


def e_value_binomial(k: int, n: int, p0: float) -> float:
    """Vovk-Wang (2021) e-value for the null ``Pr[fail] ≤ p0`` against
    the alternative ``Pr[fail] > p0`` using the *maximum-likelihood*
    betting fraction (the empirical rate, capped to a valid bet)::

        p̂ = max(p0 + δ, k / n)
        e   = (p̂ / p0)^k * ((1 - p̂)/(1 - p0))^{n - k}
    where δ is a tiny floor preventing the trivial bet ``p̂ = p0``.
    Returns 1.0 if ``n = 0``.

    Ville's inequality ⇒ Pr_{H₀}(∃t: e_t ≥ 1/α) ≤ α; *anytime-valid*.
    """
    if n <= 0:
        return 1.0
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1); got {p0}")
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of [0, n={n}]")
    rhat = k / n
    p = max(p0 + 1e-12, rhat)
    if p >= 1.0:
        p = 1.0 - 1e-12
    # work in log-space for numerical safety
    log_e = k * (math.log(p) - math.log(p0)) \
        + (n - k) * (math.log(1.0 - p) - math.log(1.0 - p0))
    # Cap to avoid overflow on extreme alternatives; an e-value of 1e300
    # is already astronomically above any rejection threshold.
    if log_e > 700.0:
        return math.exp(700.0)
    if log_e < -700.0:
        return 0.0
    return math.exp(log_e)


# =============================================================================
# Reports
# =============================================================================


@dataclass(frozen=True)
class RefutationReport:
    """The result of a Refuter call.

    Stable, immutable, JSON-serialisable, fingerprintable.
    """
    refuted: bool
    counterexample: Counterexample | None
    n_trials: int
    n_failures: int
    walltime_s: float
    alpha: float
    failure_rate_ucb: float        # (1-α) UCB on Pr[H fails] under sampling distribution
    failure_rate_emp: float        # k / n empirical failure rate
    e_value: float                 # anytime-valid evidence accumulator (1.0 = no evidence)
    strategy_counts: dict          # {strategy_name: n_trials_used}
    strategy_witnesses: dict       # {strategy_name: n_witnesses_found}
    near_misses: tuple             # tuple of (x, margin) for smallest-margin non-refuting inputs
    extra: dict                    # implementation-specific extras (e.g. tightness margin for bounds)
    fingerprint: str

    @property
    def supported(self) -> bool:
        return not self.refuted

    @property
    def failure_rate_rule_of_three(self) -> float:
        """The 95% rule-of-three bound on failure rate (used when α ≈ 0.05)."""
        return rule_of_three(self.n_trials)

    def support_claim(self) -> str:
        """A human-readable rendering of the support claim or refutation."""
        if self.refuted:
            assert self.counterexample is not None
            return (f"REFUTED in {self.n_trials} trials "
                    f"({self.walltime_s:.3f}s) "
                    f"by witness x={_brief(self.counterexample.x)} "
                    f"with margin {self.counterexample.margin:.6g} "
                    f"(strategy: {self.counterexample.strategy}).")
        return (f"SUPPORTED at (1-α)={1.0-self.alpha:.3f} confidence "
                f"after {self.n_trials} trials ({self.walltime_s:.3f}s); "
                f"empirical failure rate {self.failure_rate_emp:.4g}, "
                f"Clopper-Pearson UCB {self.failure_rate_ucb:.4g}; "
                f"anytime-valid e-value {self.e_value:.3g}.")

    def to_dict(self) -> dict:
        cex = self.counterexample.to_dict() if self.counterexample else None
        return {
            "refuted": self.refuted,
            "counterexample": cex,
            "n_trials": self.n_trials,
            "n_failures": self.n_failures,
            "walltime_s": self.walltime_s,
            "alpha": self.alpha,
            "failure_rate_ucb": self.failure_rate_ucb,
            "failure_rate_emp": self.failure_rate_emp,
            "e_value": self.e_value,
            "strategy_counts": dict(self.strategy_counts),
            "strategy_witnesses": dict(self.strategy_witnesses),
            "near_misses": [
                {"x": _safe_jsonable(x), "margin": m}
                for (x, m) in self.near_misses
            ],
            "extra": _safe_jsonable(self.extra),
            "fingerprint": self.fingerprint,
        }


def _brief(v: Any, max_len: int = 64) -> str:
    s = repr(_safe_jsonable(v))
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# =============================================================================
# Fingerprinting
# =============================================================================


def _hash_payload(payload: Any) -> str:
    """Deterministic SHA-256 of a JSON-able payload."""
    blob = json.dumps(
        _safe_jsonable(payload), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _predicate_signature(predicate: Callable, label: str | None) -> str:
    """A stable label for the predicate.  We can't fingerprint code,
    but we can fingerprint *what the user told us about it* — the name
    and an optional user-supplied label.  Combined with the witness list
    the fingerprint is sufficient for replay-verification."""
    name = getattr(predicate, "__qualname__", None) or getattr(predicate, "__name__", None) \
        or repr(predicate)
    if label:
        return f"{label}::{name}"
    return name


# =============================================================================
# Refuter
# =============================================================================


@dataclass
class _SearchState:
    rng: random.Random
    n_trials_target: int
    alpha: float
    walltime_budget: float | None
    start_t: float
    seen_inputs: set                # for de-dup by safe-jsonable key
    strategy_counts: dict           # {strategy: trials}
    strategy_witnesses: dict        # {strategy: cex_count}
    near_misses: list               # [(x, margin)]
    n_evals: int
    n_failures: int
    best_cex: Counterexample | None
    best_margin: float | None       # smallest margin seen (most violating)

    def time_left(self) -> float:
        if self.walltime_budget is None:
            return float("inf")
        return max(0.0, self.walltime_budget - (time.monotonic() - self.start_t))

    def has_budget(self) -> bool:
        return self.n_evals < self.n_trials_target and self.time_left() > 0.0


class Refuter:
    """Automated counterexample / falsification engine.

    A single Refuter is parametrised by:
      * ``seed`` — deterministic RNG seed (every call is fingerprintable).
      * ``max_near_misses`` — how many smallest-margin non-refuting inputs
        to retain.
      * ``cap_strategy_share`` — per-strategy maximum share of the trial
        budget (a safety against pathological strategies dominating).
    """

    def __init__(
        self,
        seed: int = 0,
        max_near_misses: int = 8,
        cap_strategy_share: float = 0.6,
        max_es_population: int = 16,
        es_initial_sigma: float = 0.3,
        es_min_sigma: float = 1e-9,
        es_max_sigma: float = 1.0,
    ) -> None:
        self._seed = int(seed)
        self._max_near_misses = max(1, int(max_near_misses))
        if not (0.0 < cap_strategy_share <= 1.0):
            raise ValueError(f"cap_strategy_share must be in (0, 1]; got {cap_strategy_share}")
        self._cap_share = float(cap_strategy_share)
        self._max_es_pop = max(2, int(max_es_population))
        self._es_initial_sigma = float(es_initial_sigma)
        self._es_min_sigma = float(es_min_sigma)
        self._es_max_sigma = float(es_max_sigma)

    # ----------------------------------------------------------------------
    # Core: try_refute
    # ----------------------------------------------------------------------

    def try_refute(
        self,
        predicate: Callable[[Any], Any],
        space: Any,
        n_trials: int = 1024,
        alpha: float = 0.05,
        walltime_s: float | None = None,
        strategies: Sequence[str] = ("boundary", "halton", "random", "evolutionary"),
        label: str | None = None,
    ) -> RefutationReport:
        """Hunt counterexamples to ``predicate`` over ``space``.

        Parameters
        ----------
        predicate : callable
            Called as ``predicate(x)`` where ``x`` is a point sampled from
            ``space``.  Must return one of:
              * ``bool`` — True ⇒ supported, False ⇒ refuted.
              * ``float`` — signed satisfaction margin (≥ 0 ⇒ supported).
              * ``(bool, float)`` — explicit (verdict, margin).
              * ``Evaluation`` — full record.
        space : Space
            Any of `ContinuousSpace`, `IntegerSpace`, `FiniteSet`,
            `BoolSpace`, `ListSpace`, `ProductSpace`.
        n_trials : int
            Maximum number of predicate evaluations.
        alpha : float
            Confidence level for support claims.  (1-α) Clopper-Pearson UCB.
        walltime_s : float | None
            Per-call wall-clock budget.  ``None`` = unlimited.
        strategies : sequence[str]
            Order in which strategies are invoked.  Each strategy gets at
            most ``cap_strategy_share * n_trials`` evaluations.
        label : str | None
            A user-supplied label folded into the fingerprint (so two
            distinct predicates with the same name don't collide).

        Returns
        -------
        RefutationReport
        """
        if n_trials <= 0:
            raise ValueError(f"n_trials must be positive; got {n_trials}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        state = _SearchState(
            rng=random.Random(self._seed),
            n_trials_target=int(n_trials),
            alpha=float(alpha),
            walltime_budget=walltime_s,
            start_t=time.monotonic(),
            seen_inputs=set(),
            strategy_counts={s: 0 for s in strategies},
            strategy_witnesses={s: 0 for s in strategies},
            near_misses=[],
            n_evals=0,
            n_failures=0,
            best_cex=None,
            best_margin=None,
        )
        # Run strategies in declared order.  Each is short-circuit:
        # if best_cex is set, subsequent strategies still run but only
        # to look for *smaller* witnesses (we don't early-return because
        # the user asked for n_trials, and we want to fingerprint over
        # the full computation).  However, for early-stop we accept the
        # *first* cex and break out: in practice that's what users want.
        for strat in strategies:
            if state.best_cex is not None:
                # already refuted; we still allow `shrink` to run later
                break
            if not state.has_budget():
                break
            self._run_strategy(strat, predicate, space, state, label)

        # If we found a cex, run the shrink strategy on it.
        if state.best_cex is not None and "shrink" not in state.strategy_counts:
            state.strategy_counts["shrink"] = 0
            state.strategy_witnesses["shrink"] = 0
            self._shrink_inplace(predicate, space, state)

        walltime = time.monotonic() - state.start_t
        return self._finalise_report(predicate, space, state, walltime, label=label,
                                     extra={"strategies": list(strategies)})

    # ----------------------------------------------------------------------
    # Metamorphic mode
    # ----------------------------------------------------------------------

    def try_refute_relation(
        self,
        f: Callable[[Any], Any],
        relation: Callable[[Any, Any, Any, Any], Any],
        space: Any,
        x_to_x2: Callable[[Any], Any] | None = None,
        x2_space: Any | None = None,
        n_trials: int = 1024,
        alpha: float = 0.05,
        walltime_s: float | None = None,
        strategies: Sequence[str] = ("boundary", "halton", "random", "evolutionary"),
        label: str | None = None,
    ) -> RefutationReport:
        """Refute a metamorphic relation between ``f(x)`` and ``f(x2)``.

        The user provides:
          * ``f`` — the function being tested,
          * ``relation(x, fx, x2, fx2)`` — returns bool / float / (bool, float),
          * either ``x_to_x2`` (a transform ``X → X``) or ``x2_space``
            (an independent sample).

        Predicate becomes ``H(x) = relation(x, f(x), x2, f(x2))``.

        Examples
        --------
        ``sort(reverse(L)) == sort(L)``::

            R.try_refute_relation(
                f=sorted,
                relation=lambda x, fx, x2, fx2: fx == fx2,
                space=ListSpace(IntegerSpace(0, 100), max_len=10),
                x_to_x2=lambda L: list(reversed(L)),
            )

        Idempotence ``f(f(x)) == f(x)``::

            R.try_refute_relation(
                f=trim,
                relation=lambda x, fx, x2, fx2: fx2 == fx,
                space=...,
                x_to_x2=lambda x: x,  # paired x2 = x ⇒ tests f(f(x)) vs f(x) when relation calls f again
            )
        """
        if x_to_x2 is None and x2_space is None:
            raise ValueError("must supply either x_to_x2 or x2_space")

        def predicate(x: Any) -> Any:
            try:
                fx = f(x)
            except Exception as exc:  # noqa: BLE001
                return Evaluation(ok=False, margin=-1.0, raw=("f(x) raised", repr(exc)),
                                  error=repr(exc))
            if x_to_x2 is not None:
                try:
                    x2 = x_to_x2(x)
                except Exception as exc:  # noqa: BLE001
                    return Evaluation(ok=False, margin=-1.0,
                                      raw=("x_to_x2 raised", repr(exc)),
                                      error=repr(exc))
            else:
                # Derive a deterministic RNG seed from `x` so the
                # metamorphic relation is replayable.
                seed = int(hashlib.sha256(repr(_safe_jsonable(x)).encode("utf-8"))
                           .hexdigest()[:16], 16)
                x2 = x2_space.sample(random.Random(seed))  # type: ignore[union-attr]
            try:
                fx2 = f(x2)
            except Exception as exc:  # noqa: BLE001
                return Evaluation(ok=False, margin=-1.0,
                                  raw=("f(x2) raised", repr(exc)),
                                  error=repr(exc))
            out = relation(x, fx, x2, fx2)
            return _coerce_to_eval(out)

        return self.try_refute(
            predicate=predicate,
            space=space,
            n_trials=n_trials,
            alpha=alpha,
            walltime_s=walltime_s,
            strategies=strategies,
            label=f"metamorphic::{label}" if label else "metamorphic",
        )

    # ----------------------------------------------------------------------
    # Bound mode
    # ----------------------------------------------------------------------

    def try_refute_bound(
        self,
        scalar: Callable[[Any], float],
        threshold: float,
        direction: str,                 # "<=" or ">="
        space: Any,
        n_trials: int = 1024,
        alpha: float = 0.05,
        walltime_s: float | None = None,
        strategies: Sequence[str] = ("boundary", "halton", "random", "evolutionary"),
        label: str | None = None,
    ) -> RefutationReport:
        """Refute a claim ``∀x ∈ space: scalar(x) [direction] threshold``.

        ``direction`` is ``"<="`` or ``">="``.  The satisfaction margin is::

            "<="  ⇒  margin = threshold - scalar(x)   (≥ 0 ⇒ supported)
            ">="  ⇒  margin = scalar(x) - threshold

        Reported extras: ``tightness_margin`` — the smallest observed
        margin (positive ⇒ how slack the bound is, negative ⇒ the
        magnitude of the worst violation).
        """
        if direction not in ("<=", ">="):
            raise ValueError(f"direction must be '<=' or '>='; got {direction!r}")
        if direction == "<=":
            def predicate(x: Any) -> Evaluation:
                try:
                    v = float(scalar(x))
                except Exception as exc:  # noqa: BLE001
                    return Evaluation(ok=False, margin=-1.0, raw=repr(exc), error=repr(exc))
                m = threshold - v
                return Evaluation(ok=(m >= 0.0), margin=m, raw=v)
        else:
            def predicate(x: Any) -> Evaluation:
                try:
                    v = float(scalar(x))
                except Exception as exc:  # noqa: BLE001
                    return Evaluation(ok=False, margin=-1.0, raw=repr(exc), error=repr(exc))
                m = v - threshold
                return Evaluation(ok=(m >= 0.0), margin=m, raw=v)
        rep = self.try_refute(
            predicate=predicate, space=space, n_trials=n_trials,
            alpha=alpha, walltime_s=walltime_s, strategies=strategies,
            label=f"bound{direction}{threshold}::{label}" if label else f"bound{direction}{threshold}",
        )
        # Add the tightness margin to extra
        if rep.refuted:
            tightness = rep.counterexample.margin if rep.counterexample else 0.0
        else:
            # smallest margin observed across near misses (positive)
            if rep.near_misses:
                tightness = min(m for (_, m) in rep.near_misses)
            else:
                tightness = float("inf")
        # Re-fingerprint with the tightness margin baked in
        return _replace_report_extra(rep, {"tightness_margin": tightness,
                                           "threshold": threshold,
                                           "direction": direction,
                                           **rep.extra})

    # ----------------------------------------------------------------------
    # Sequential / anytime-valid mode
    # ----------------------------------------------------------------------

    def refute_until(
        self,
        predicate: Callable[[Any], Any],
        space: Any,
        p0: float = 1e-6,
        alpha: float = 0.05,
        n_max: int = 100_000,
        block_size: int = 256,
        strategies: Sequence[str] = ("boundary", "halton", "random", "evolutionary"),
        label: str | None = None,
    ) -> RefutationReport:
        """Sequentially refute ``Pr[H fails] ≤ p0`` with an anytime-valid
        e-process; stop when ``e ≥ 1/α`` (reject ``H₀``) or ``n ≥ n_max``.

        Uses the same strategy portfolio as ``try_refute`` but runs in
        blocks so the e-value is recomputed between blocks, giving an
        *anytime-valid* sequential decision under arbitrary stopping
        (Ville 1939; Howard-Ramdas-McAuliffe-Sekhon 2021).
        """
        if not (0.0 < p0 < 1.0):
            raise ValueError(f"p0 must be in (0, 1); got {p0}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        target_e = 1.0 / alpha
        state = _SearchState(
            rng=random.Random(self._seed),
            n_trials_target=int(n_max),
            alpha=float(alpha),
            walltime_budget=None,
            start_t=time.monotonic(),
            seen_inputs=set(),
            strategy_counts={s: 0 for s in strategies},
            strategy_witnesses={s: 0 for s in strategies},
            near_misses=[],
            n_evals=0,
            n_failures=0,
            best_cex=None,
            best_margin=None,
        )
        while state.has_budget():
            # Run one block of evaluations across strategies
            block_left = block_size
            for strat in strategies:
                if not state.has_budget() or block_left <= 0:
                    break
                cap_orig = self._cap_share
                # within a block, give each strategy roughly equal share
                self._cap_share = 1.0
                try:
                    before = state.n_evals
                    self._run_strategy(strat, predicate, space, state, label,
                                       budget_override=block_left)
                    block_left -= (state.n_evals - before)
                finally:
                    self._cap_share = cap_orig
                if state.best_cex is not None:
                    break

            if state.best_cex is not None:
                break

            # Sequential decision: e-value vs 1/α
            e = e_value_binomial(state.n_failures, state.n_evals, p0)
            if e >= target_e:
                break

        if state.best_cex is not None and "shrink" not in state.strategy_counts:
            state.strategy_counts["shrink"] = 0
            state.strategy_witnesses["shrink"] = 0
            self._shrink_inplace(predicate, space, state)

        walltime = time.monotonic() - state.start_t
        rep = self._finalise_report(
            predicate, space, state, walltime, label=label,
            extra={"strategies": list(strategies), "p0": p0,
                   "target_e": target_e, "anytime_valid": True},
        )
        # If we stopped because e ≥ 1/α, mark refuted="rate" (failure rate exceeds p0)
        # only if we don't already have a witness.
        if not rep.refuted and rep.e_value >= target_e:
            rep = _replace_report_extra(rep, {**rep.extra, "rejected_by_e_value": True})
        return rep

    # ----------------------------------------------------------------------
    # Shrinking (QuickCheck-style)
    # ----------------------------------------------------------------------

    def shrink(
        self,
        witness: Counterexample,
        space: Any,
        predicate: Callable[[Any], Any],
        max_steps: int = 256,
        label: str | None = None,
    ) -> Counterexample:
        """Minimise a counterexample.  Returns the smallest still-refuting
        witness reachable by structural-shrink moves on the space."""
        cex = witness
        # Local state for shrink so we don't pollute caller state
        state = _SearchState(
            rng=random.Random(self._seed + 7919),
            n_trials_target=max_steps,
            alpha=0.05,
            walltime_budget=None,
            start_t=time.monotonic(),
            seen_inputs=set(),
            strategy_counts={"shrink": 0},
            strategy_witnesses={"shrink": 0},
            near_misses=[],
            n_evals=0,
            n_failures=0,
            best_cex=cex,
            best_margin=cex.margin,
        )
        self._shrink_inplace(predicate, space, state)
        return state.best_cex if state.best_cex is not None else cex

    # ----------------------------------------------------------------------
    # Strategy dispatch
    # ----------------------------------------------------------------------

    def _run_strategy(
        self,
        strat: str,
        predicate: Callable[[Any], Any],
        space: Any,
        state: _SearchState,
        label: str | None,
        budget_override: int | None = None,
    ) -> None:
        # ``budget`` is a *per-call* maximum on additional evaluations for
        # this strategy invocation.  The strategy is responsible for
        # decrementing it via the helper ``_eval_and_record``; we wrap that
        # with a simple counter.
        if budget_override is not None:
            budget = max(1, int(budget_override))
        else:
            budget = max(1, int(self._cap_share * state.n_trials_target))
        if strat == "boundary":
            self._strategy_boundary(predicate, space, state, budget)
        elif strat == "halton":
            self._strategy_halton(predicate, space, state, budget)
        elif strat == "random":
            self._strategy_random(predicate, space, state, budget)
        elif strat == "evolutionary":
            self._strategy_es(predicate, space, state, budget)
        elif strat == "nelder_mead":
            self._strategy_nelder_mead(predicate, space, state, budget)
        else:
            raise ValueError(f"unknown strategy: {strat!r}")

    # ---- Boundary strategy --------------------------------------------------

    def _strategy_boundary(self, predicate, space, state, budget):
        seeds = _enumerate_boundary(space)
        for x in seeds:
            if budget <= 0 or not state.has_budget() or state.best_cex is not None:
                return
            before = state.n_evals
            self._eval_and_record(predicate, x, "boundary", state)
            if state.n_evals > before:
                budget -= 1

    # ---- Halton (low-discrepancy) strategy ----------------------------------

    def _strategy_halton(self, predicate, space, state, budget):
        cont_axes = _continuous_axes(space)
        if not cont_axes:
            return self._strategy_random(predicate, space, state, budget)
        # Persistent Halton index across calls (uses strategy_counts as the
        # *cumulative* iteration count, which is exactly what Halton wants
        # for low-discrepancy continuation).
        start_idx = state.strategy_counts.get("halton", 0)
        idx = start_idx
        while budget > 0 and state.has_budget() and state.best_cex is None:
            x = _sample_halton(space, idx, state.rng)
            idx += 1
            before = state.n_evals
            self._eval_and_record(predicate, x, "halton", state)
            if state.n_evals > before:
                budget -= 1

    # ---- Random strategy ----------------------------------------------------

    def _strategy_random(self, predicate, space, state, budget):
        while budget > 0 and state.has_budget() and state.best_cex is None:
            x = space.sample(state.rng)
            before = state.n_evals
            self._eval_and_record(predicate, x, "random", state)
            if state.n_evals > before:
                budget -= 1
            else:
                # de-dup hit; advance budget anyway to avoid spinning
                budget -= 1

    # ---- (1+λ)-Evolutionary search on margin --------------------------------

    def _strategy_es(self, predicate, space, state, budget):
        if state.near_misses:
            seeds = [x for (x, _m) in sorted(state.near_misses, key=lambda t: t[1])[: 4]]
        else:
            seeds = [space.sample(state.rng)]
            b = _enumerate_boundary(space)
            if b:
                seeds.append(b[0])
        sigma = self._es_initial_sigma
        good = 0
        bad = 0
        pidx = 0
        while budget > 0 and state.has_budget() and state.best_cex is None:
            parent = seeds[pidx % len(seeds)]
            child = space.mutate(parent, sigma, state.rng)
            parent_eval = _safe_predicate_eval(predicate, parent)
            child_eval = _safe_predicate_eval(predicate, child)
            before = state.n_evals
            self._record_eval(parent_eval, parent, "evolutionary", state)
            if state.best_cex is not None:
                return
            self._record_eval(child_eval, child, "evolutionary", state)
            budget -= (state.n_evals - before)
            if child_eval.margin < parent_eval.margin:
                good += 1
                seeds[pidx % len(seeds)] = child
            else:
                bad += 1
            pidx += 1
            if (good + bad) >= 10:
                rate = good / (good + bad)
                if rate > 0.2:
                    sigma *= 1.22
                elif rate < 0.2:
                    sigma /= 1.22
                sigma = max(self._es_min_sigma, min(self._es_max_sigma, sigma))
                good = bad = 0

    # ---- Nelder-Mead simplex (optional alternative for low-D continuous) ----

    def _strategy_nelder_mead(self, predicate, space, state, budget):
        if not _continuous_axes(space):
            return self._strategy_es(predicate, space, state, budget)
        coords, sbox = _continuous_box(space)
        n = len(coords)
        if n == 0:
            return
        center = _sample_box(sbox, state.rng)
        simplex = [center]
        for i, name in enumerate(coords):
            v = dict(center)
            lo, hi = sbox[name]
            v[name] = min(hi, max(lo, center[name] + 0.1 * (hi - lo)))
            simplex.append(v)

        def margin_of(v: dict) -> float:
            nonlocal budget
            x = _coords_to_input(v, space, coords)
            ev = _safe_predicate_eval(predicate, x)
            before = state.n_evals
            self._record_eval(ev, x, "nelder_mead", state)
            budget -= (state.n_evals - before)
            return ev.margin

        margins = [margin_of(s) for s in simplex]
        if state.best_cex is not None:
            return
        alpha_, gamma_, rho_, sigma_nm = 1.0, 2.0, 0.5, 0.5
        while budget > 0 and state.has_budget() and state.best_cex is None:
            order = sorted(range(n + 1), key=lambda i: margins[i])
            simplex = [simplex[i] for i in order]
            margins = [margins[i] for i in order]
            if state.best_cex is not None or not state.has_budget():
                return
            # centroid of best n
            centroid = {c: sum(simplex[i][c] for i in range(n)) / n for c in coords}
            worst = simplex[-1]
            reflected = {c: _clip(centroid[c] + alpha_ * (centroid[c] - worst[c]),
                                  *sbox[c]) for c in coords}
            mr = margin_of(reflected)
            if state.best_cex is not None:
                return
            if margins[0] <= mr < margins[-2]:
                simplex[-1] = reflected
                margins[-1] = mr
            elif mr < margins[0]:
                expanded = {c: _clip(centroid[c] + gamma_ * (reflected[c] - centroid[c]),
                                     *sbox[c]) for c in coords}
                me = margin_of(expanded)
                if state.best_cex is not None:
                    return
                if me < mr:
                    simplex[-1] = expanded
                    margins[-1] = me
                else:
                    simplex[-1] = reflected
                    margins[-1] = mr
            else:
                contracted = {c: _clip(centroid[c] + rho_ * (worst[c] - centroid[c]),
                                       *sbox[c]) for c in coords}
                mc = margin_of(contracted)
                if state.best_cex is not None:
                    return
                if mc < margins[-1]:
                    simplex[-1] = contracted
                    margins[-1] = mc
                else:
                    # shrink toward best
                    best = simplex[0]
                    for i in range(1, n + 1):
                        simplex[i] = {c: _clip(best[c] + sigma_nm * (simplex[i][c] - best[c]),
                                               *sbox[c]) for c in coords}
                        margins[i] = margin_of(simplex[i])
                        if state.best_cex is not None:
                            return

    # ---- Shrinking ---------------------------------------------------------

    def _shrink_inplace(self, predicate, space, state):
        """Repeatedly shrink ``state.best_cex.x`` while predicate still
        fails.  Stops when no neighbour shrink-move yields a failing
        witness."""
        if state.best_cex is None:
            return
        cap = state.n_trials_target  # use remaining trial budget liberally
        steps = 0
        cur_x = state.best_cex.x
        cur_margin = state.best_cex.margin
        cur_raw = state.best_cex.raw
        while steps < cap and state.has_budget():
            improved = False
            for cand in _shrink_candidates(cur_x, space):
                key = _input_key(cand)
                if key in state.seen_inputs:
                    continue
                ev = _safe_predicate_eval(predicate, cand)
                state.seen_inputs.add(key)
                state.strategy_counts["shrink"] = state.strategy_counts.get("shrink", 0) + 1
                state.n_evals += 1
                steps += 1
                if not ev.ok:
                    state.n_failures += 1
                    state.strategy_witnesses["shrink"] = \
                        state.strategy_witnesses.get("shrink", 0) + 1
                    # shrinker prefers *smaller* witnesses, measured by complexity score;
                    # ties broken by smaller margin (more strongly violating)
                    if _is_smaller(cand, cur_x):
                        cur_x = cand
                        cur_margin = ev.margin
                        cur_raw = ev.raw
                        state.best_cex = Counterexample(
                            x=cur_x, margin=cur_margin, strategy="shrink",
                            trial=state.n_evals, raw=cur_raw,
                        )
                        improved = True
                        break  # restart from new center
                if not state.has_budget():
                    return
            if not improved:
                return

    # ----------------------------------------------------------------------
    # Eval + record helpers
    # ----------------------------------------------------------------------

    def _eval_and_record(self, predicate, x, strategy, state):
        key = _input_key(x)
        if key in state.seen_inputs:
            return
        state.seen_inputs.add(key)
        ev = _safe_predicate_eval(predicate, x)
        self._record_eval(ev, x, strategy, state)

    def _record_eval(self, ev: Evaluation, x: Any, strategy: str, state: _SearchState):
        state.n_evals += 1
        state.strategy_counts[strategy] = state.strategy_counts.get(strategy, 0) + 1
        if not ev.ok:
            state.n_failures += 1
            state.strategy_witnesses[strategy] = state.strategy_witnesses.get(strategy, 0) + 1
            if state.best_cex is None or ev.margin < state.best_cex.margin:
                state.best_cex = Counterexample(
                    x=x, margin=ev.margin, strategy=strategy,
                    trial=state.n_evals, raw=ev.raw,
                )
        else:
            # Track near misses: smallest-margin supporting inputs
            self._push_near_miss(state, x, ev.margin)
        if state.best_margin is None or ev.margin < state.best_margin:
            state.best_margin = ev.margin

    def _push_near_miss(self, state, x, margin):
        # Keep the K smallest-margin supporting inputs in a bounded list
        if len(state.near_misses) < self._max_near_misses:
            state.near_misses.append((x, margin))
            state.near_misses.sort(key=lambda t: t[1])
            return
        worst = state.near_misses[-1]
        if margin < worst[1]:
            state.near_misses[-1] = (x, margin)
            state.near_misses.sort(key=lambda t: t[1])

    # ----------------------------------------------------------------------
    # Report finalisation
    # ----------------------------------------------------------------------

    def _finalise_report(
        self, predicate, space, state, walltime, label, extra,
    ) -> RefutationReport:
        emp = state.n_failures / state.n_evals if state.n_evals > 0 else 0.0
        if state.n_failures == 0:
            ucb = clopper_pearson_zero_ucb(state.n_evals, state.alpha)
        else:
            ucb = clopper_pearson_ucb(state.n_failures, state.n_evals, state.alpha)
        # Default e-value uses p0 = α (a conservative anchor); refute_until
        # overrides via `extra["p0"]`.
        p0 = extra.get("p0", state.alpha) if isinstance(extra, dict) else state.alpha
        e = e_value_binomial(state.n_failures, state.n_evals, p0)
        sig = _predicate_signature(predicate, label)
        # The fingerprint covers: predicate signature, space shape, seed,
        # n_trials, alpha, the witness list, and the strategy counts.  The
        # raw predicate body isn't hashable but the signature + witness
        # values let an auditor re-run the call deterministically.
        cex_dict = state.best_cex.to_dict() if state.best_cex else None
        fp = _hash_payload({
            "kind": "refutation_report",
            "predicate_signature": sig,
            "space": space.to_dict(),
            "seed": self._seed,
            "n_trials_target": state.n_trials_target,
            "alpha": state.alpha,
            "n_evals": state.n_evals,
            "n_failures": state.n_failures,
            "best_cex": cex_dict,
            "strategy_counts": dict(state.strategy_counts),
        })
        return RefutationReport(
            refuted=(state.best_cex is not None),
            counterexample=state.best_cex,
            n_trials=state.n_evals,
            n_failures=state.n_failures,
            walltime_s=walltime,
            alpha=state.alpha,
            failure_rate_ucb=ucb,
            failure_rate_emp=emp,
            e_value=e,
            strategy_counts=dict(state.strategy_counts),
            strategy_witnesses=dict(state.strategy_witnesses),
            near_misses=tuple(state.near_misses),
            extra=dict(extra) if extra else {},
            fingerprint=fp,
        )


# =============================================================================
# Replace-extra helper (frozen dataclass workaround)
# =============================================================================


def _replace_report_extra(rep: RefutationReport, new_extra: dict) -> RefutationReport:
    return RefutationReport(
        refuted=rep.refuted,
        counterexample=rep.counterexample,
        n_trials=rep.n_trials,
        n_failures=rep.n_failures,
        walltime_s=rep.walltime_s,
        alpha=rep.alpha,
        failure_rate_ucb=rep.failure_rate_ucb,
        failure_rate_emp=rep.failure_rate_emp,
        e_value=rep.e_value,
        strategy_counts=dict(rep.strategy_counts),
        strategy_witnesses=dict(rep.strategy_witnesses),
        near_misses=rep.near_misses,
        extra=dict(new_extra),
        fingerprint=rep.fingerprint,
    )


# =============================================================================
# Halton sampling for product spaces
# =============================================================================


def _continuous_axes(space: Any) -> list[tuple]:
    """Walk the (nested) space; return a list of (path, ContinuousSpace)
    pairs for every continuous coordinate."""
    out: list[tuple] = []

    def rec(s, path):
        if isinstance(s, ContinuousSpace):
            out.append((tuple(path), s))
        elif isinstance(s, ProductSpace):
            for k, child in s.children.items():
                rec(child, path + [k])
        elif isinstance(s, ListSpace):
            # treat as continuous only if elem is continuous; we don't
            # attempt to sample list lengths via Halton — random for those.
            pass

    rec(space, [])
    return out


def _sample_halton(space: Any, idx: int, rng: random.Random) -> Any:
    """Sample ``space`` using Halton on continuous axes and random on
    discrete axes."""
    axes = _continuous_axes(space)

    def rec(s, path):
        if isinstance(s, ContinuousSpace):
            # find axis index for path
            for ai, (p, _) in enumerate(axes):
                if p == tuple(path):
                    return _halton_continuous(s, idx, ai)
            return s.sample(rng)
        if isinstance(s, ProductSpace):
            return {k: rec(child, path + [k]) for k, child in s.children.items()}
        return s.sample(rng)

    return rec(space, [])


def _enumerate_boundary(space: Any) -> list[Any]:
    """Enumerate boundary corners for a (possibly nested) space."""
    return space.boundary()


# =============================================================================
# Continuous-box helpers for Nelder-Mead
# =============================================================================


def _continuous_box(space: Any) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Return ([name_path_strs], {name_path_str: (lo, hi)}) for top-level
    continuous coordinates of a ProductSpace.  Only one level deep — the
    Nelder-Mead implementation handles flat continuous boxes."""
    coords: list[str] = []
    box: dict[str, tuple[float, float]] = {}
    if isinstance(space, ProductSpace):
        for k, child in space.children.items():
            if isinstance(child, ContinuousSpace):
                coords.append(k)
                box[k] = (child.lo, child.hi)
    elif isinstance(space, ContinuousSpace):
        coords.append("_")
        box["_"] = (space.lo, space.hi)
    return coords, box


def _sample_box(box: dict[str, tuple[float, float]], rng: random.Random) -> dict:
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in box.items()}


def _coords_to_input(v: dict, space: Any, coords: list[str]) -> Any:
    """Build the predicate input from a continuous-coord vector ``v``;
    non-continuous coordinates retain their default (lo) value."""
    if isinstance(space, ContinuousSpace):
        return v.get("_", space.lo)
    if isinstance(space, ProductSpace):
        out = {}
        for k, child in space.children.items():
            if isinstance(child, ContinuousSpace):
                out[k] = v[k]
            else:
                # leave at boundary[0]
                out[k] = child.boundary()[0]
        return out
    return v


def _clip(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


# =============================================================================
# Shrinking machinery
# =============================================================================


def _shrink_candidates(x: Any, space: Any) -> list[Any]:
    """Generate a list of structurally-smaller candidates from ``x``.

    QuickCheck-style: for each coordinate, propose halving toward zero,
    truncating, removing list elements, etc.
    """
    out: list[Any] = []
    if isinstance(space, ContinuousSpace) and isinstance(x, (int, float)):
        # halve toward 0
        if x != 0.0:
            out.append(_clip(x / 2.0, space.lo, space.hi))
            out.append(_clip(x / 4.0, space.lo, space.hi))
            out.append(_clip(0.0 if space.lo <= 0.0 <= space.hi else space.lo, space.lo, space.hi))
        # round to integer (often a "simpler" representation)
        try:
            r = float(int(x))
            r = _clip(r, space.lo, space.hi)
            if r != x:
                out.append(r)
        except (ValueError, OverflowError):
            pass
    elif isinstance(space, IntegerSpace) and isinstance(x, int):
        if x > 0:
            out.append(max(space.lo, x // 2))
            out.append(max(space.lo, 0) if space.lo <= 0 else space.lo)
        elif x < 0:
            out.append(min(space.hi, -((-x) // 2)))
            out.append(min(space.hi, 0) if space.hi >= 0 else space.hi)
    elif isinstance(space, BoolSpace):
        # only one alt: flip toward False
        if x is True:
            out.append(False)
    elif isinstance(space, FiniteSet):
        # earlier index = "simpler"
        try:
            i = space.values.index(x)
            if i > 0:
                out.append(space.values[0])
                out.append(space.values[i // 2])
        except ValueError:
            pass
    elif isinstance(space, ListSpace) and isinstance(x, list):
        # 1. drop any single element
        for i in range(len(x)):
            cand = x[:i] + x[i + 1:]
            if len(cand) >= space.min_len:
                out.append(cand)
        # 2. shrink one element
        for i, vi in enumerate(x):
            for vsm in _shrink_candidates(vi, space.elem):
                out.append(x[:i] + [vsm] + x[i + 1:])
        # 3. half the list
        if len(x) > space.min_len + 1:
            out.append(x[: max(space.min_len, len(x) // 2)])
    elif isinstance(space, ProductSpace) and isinstance(x, dict):
        # shrink one named field at a time
        for k, child in space.children.items():
            if k not in x:
                continue
            for vsm in _shrink_candidates(x[k], child):
                cand = dict(x)
                cand[k] = vsm
                out.append(cand)
    return out


def _complexity(x: Any) -> tuple:
    """A lexicographic complexity score: shorter / smaller-magnitude / fewer
    fields wins.  Returned as a tuple so Python's tuple ordering is the
    'smaller' ordering."""
    if x is None or isinstance(x, bool):
        return (0, int(bool(x)))
    if isinstance(x, int):
        return (1, abs(x))
    if isinstance(x, float):
        if math.isnan(x):
            return (10, 0.0)  # NaN is most "exotic"
        if math.isinf(x):
            return (9, 0.0)
        return (2, abs(x))
    if isinstance(x, str):
        return (3, len(x), x)
    if isinstance(x, (list, tuple)):
        if not x:
            return (4, 0)
        children = [_complexity(c) for c in x]
        return (4, len(x), tuple(children))
    if isinstance(x, dict):
        keys = sorted(x.keys(), key=lambda k: str(k))
        children = tuple(_complexity(x[k]) for k in keys)
        return (5, len(x), children)
    return (8, repr(x))


def _is_smaller(a: Any, b: Any) -> bool:
    """Strict structural ordering."""
    try:
        return _complexity(a) < _complexity(b)
    except TypeError:
        return False


# =============================================================================
# Predicate evaluation (with exception capture)
# =============================================================================


def _safe_predicate_eval(predicate: Callable, x: Any) -> Evaluation:
    """Call ``predicate(x)`` capturing exceptions as immediate refutations.

    A hypothesis that *raises* on a valid input is, by definition,
    refuted at that input: the user's claim "this predicate holds on
    space" cannot survive a predicate-evaluation crash."""
    try:
        out = predicate(x)
    except Exception as exc:  # noqa: BLE001
        return Evaluation(ok=False, margin=-1.0, raw=("predicate raised", repr(exc)),
                          error=repr(exc))
    try:
        return _coerce_to_eval(out)
    except InvalidHypothesis as exc:
        return Evaluation(ok=False, margin=-1.0, raw=("invalid return", str(exc)),
                          error=str(exc))


def _input_key(x: Any) -> tuple:
    """A hashable, stable de-duplication key."""
    return (type(x).__name__, repr(_safe_jsonable(x)))


# =============================================================================
# Quick helpers — common predicate constructors
# =============================================================================


def forall(predicate: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Identity wrapper that documents intent: "∀ x in space: predicate(x)"."""
    return predicate


def with_margin(
    predicate: Callable[[Any], bool],
    margin_fn: Callable[[Any], float],
) -> Callable[[Any], Evaluation]:
    """Pair a boolean predicate with a signed margin function.

    The margin signals "how close to the boundary" — negative magnitudes
    drive evolutionary / Nelder-Mead toward the violation.
    """
    def wrapped(x: Any) -> Evaluation:
        ok = bool(predicate(x))
        try:
            m = float(margin_fn(x))
        except Exception:  # noqa: BLE001
            m = 1.0 if ok else -1.0
        # If user gave a sign-inconsistent margin, fall back to ±1
        if ok and m < 0:
            m = abs(m) + 1e-12
        if (not ok) and m > 0:
            m = -abs(m) - 1e-12
        return Evaluation(ok=ok, margin=m, raw=None)
    return wrapped


# =============================================================================
# CEGIS — the canonical composition with Synthesizer
# =============================================================================


def cegis_loop(
    candidate0: Any,
    refute: Callable[[Any], RefutationReport],
    resynthesise: Callable[[Any, Counterexample], Any],
    max_rounds: int = 16,
) -> tuple[Any, list[Counterexample]]:
    """One-shot CEGIS scaffold (Solar-Lezama 2008).

    ``candidate0``     initial candidate hypothesis / program.
    ``refute(c)``      → RefutationReport on candidate ``c``.
    ``resynthesise(c, cex)`` → new candidate that handles ``cex``.

    Returns
    -------
    (final_candidate, list_of_witnesses)
    """
    witnesses: list[Counterexample] = []
    c = candidate0
    for _ in range(max_rounds):
        rep = refute(c)
        if not rep.refuted:
            return c, witnesses
        assert rep.counterexample is not None
        witnesses.append(rep.counterexample)
        c = resynthesise(c, rep.counterexample)
    return c, witnesses


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # exceptions
    "RefuterError",
    "InvalidSpace",
    "InvalidHypothesis",
    "BudgetExhausted",
    # spaces
    "ContinuousSpace",
    "IntegerSpace",
    "FiniteSet",
    "BoolSpace",
    "ListSpace",
    "ProductSpace",
    "Product",
    # core
    "Refuter",
    "Counterexample",
    "Evaluation",
    "RefutationReport",
    # statistical helpers
    "clopper_pearson_zero_ucb",
    "clopper_pearson_ucb",
    "rule_of_three",
    "hoeffding_ucb",
    "e_value_binomial",
    # predicate helpers
    "forall",
    "with_margin",
    # CEGIS scaffold
    "cegis_loop",
]
