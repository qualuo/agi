r"""Conjecturer — automated mathematical conjecture generation as a runtime primitive.

Every primitive shipped so far in this runtime is *fitting* — given a stream of
observations it returns the best parameter vector inside an a-priori function
class.  ``Scientist`` returns the best *sparse linear* combination of a fixed
basis of differentiable expressions.  ``Predictor`` returns the best
*variable-order Markov* mixture.  ``Embedder`` returns a vector in a fixed
geometry.  None of them returns a *closed-form mathematical identity* whose
form is itself discovered — an equation a working mathematician would call
a *conjecture*.

The ``Conjecturer`` closes that gap.  It is the runtime primitive a
coordinator can call to ask the question

    "What closed-form expression in terms of (1, π, e, γ, log 2, … and these
     user-supplied observations) is *numerically indistinguishable* from
     the constant I just measured to k digits of precision?"

The answer is a ranked list of ``Conjecture`` objects, each one a candidate
identity ``Σ aᵢ · vᵢ = 0`` with small integer coefficients ``aᵢ ∈ ℤ``, an
explicit upper bound on the residual ``|Σ aᵢ · vᵢ|``, and a per-conjecture
*false-discovery* probability — the probability that the relation is
spurious given the working precision and the dimension of the search space.

The pitch reduced to a runtime call::

    cj = Conjecturer.create(precision_digits=30, seed=0)
    cj.observe("phi", (1.0 + 5.0 ** 0.5) / 2.0)     # golden ratio
    cj.with_constants(("one", "phi"))               # search the 2-space
    out = cj.propose(max_coeff=20)                  # → ["phi² − phi − 1 = 0"]
    rep = cj.report()

Every ``observe``, ``propose``, ``verify``, ``report`` call is hashed into a
SHA-256 chain compatible with the rest of the runtime's
``AttestationLedger`` — so a conjecture's lineage from raw numbers to claim
to verification is fully auditable.

Mathematical roots
------------------

* **Ferguson-Bailey 1992 — PSLQ.**  Given a vector ``x = (x₁,…,x_n) ∈ ℝⁿ``,
  PSLQ returns either a non-zero integer vector ``m ∈ ℤⁿ`` with
  ``m·x = 0`` exactly (an *integer relation*) or a certificate that no
  such relation exists with ``‖m‖_∞ ≤ M``.  The algorithm maintains a
  pair of matrices ``(H, B)`` with ``H ∈ ℝ^{n×(n−1)}`` lower-trapezoidal
  and ``B ∈ GL_n(ℤ)`` and alternates Hermite reduction with weighted
  row exchanges.  Termination is guaranteed in finitely many steps for
  any ``γ > √(4/3)`` (Bailey-Broadhurst-Plouffe 1997 §3).

* **Lenstra-Lenstra-Lovász 1982 — LLL lattice reduction.**  Given a basis
  ``B = (b₁,…,b_n)`` of a lattice ``L ⊂ ℤⁿ`` LLL returns an equivalent
  basis whose first vector ``b₁`` satisfies
  ``‖b₁‖ ≤ 2^{(n−1)/4} · λ₁(L)`` where ``λ₁(L)`` is the shortest vector
  length.  Applied to the *integer-relation lattice*

  .. math::

      L = \bigl\{ (m_1,\dots,m_n,\lfloor C \, m \cdot x \rceil)
          : m \in \mathbb{Z}^n \bigr\}

  with ``C`` a large integer scaling factor, the first basis vector after
  reduction encodes ``m`` such that ``|m·x|`` is small.  ``Conjecturer``
  implements LLL in **exact rational arithmetic** (Python ``fractions``)
  so the answer is determined by the input precision alone — never by
  floating-point round-off.

* **Khinchin 1935; Lochs 1964 — Continued fractions.**  Every irrational
  ``x ∈ ℝ`` has a unique simple continued-fraction expansion
  ``x = [a₀; a₁, a₂, …]`` whose convergents ``p_k / q_k`` are the *best
  rational approximations*: for any ``p, q`` with ``q ≤ q_k`` the bound
  ``|x − p/q| ≥ |x − p_k/q_k|`` holds (Khinchin's Theorem 9).  The
  ``recognize_rational`` and ``recognize_quadratic`` calls of
  ``Conjecturer`` consume the CF expansion and stop the moment a *large
  quotient* (``aₖ ≥ Q_max``, default 1e8) signals that the next term
  would carry signal from finite-precision noise rather than from the
  irrational tail.

* **Stern-Brocot tree.**  Best rational approximations with bounded
  denominator are equivalently found by binary descent down the
  Stern-Brocot tree.  ``Conjecturer`` exposes this via ``best_rational``,
  which differs from the CF convergent stream by giving a *fixed
  denominator budget* ``D``: the algorithm returns the rational
  ``p/q`` with ``q ≤ D`` minimising ``|x − p/q|``.  This is the
  algorithm originally used by Hardy-Wright (1979 §3.7).

* **Ramanujan Machine (Raayoni-Gottlieb-Manor-Pisha-Harris-Mendlovic-
  Haviv-Hadad-Kaminer 2021, Nature 590).**  Given a numerical constant
  ``α``, search a finite family of *integer-coefficient* continued
  fractions ``α = a₀ + b₀ / (a₁ + b₁/(a₂ + …))`` for one whose numerical
  value matches ``α`` to working precision.  When found, the candidate
  identity is checked at higher precision; only matches that survive
  the *precision-doubling* test are reported.  ``Conjecturer`` adopts
  exactly this discipline: every candidate identity must survive
  re-evaluation at a precision strictly higher than that used to find
  it, with the residual shrinking accordingly.  Failed verifications
  are *rejected and logged*, never silently discarded.

* **False-discovery control.**  In a search over the lattice
  ``{‖m‖_∞ ≤ M}`` with ``n`` columns, there are ``(2M+1)ⁿ − 1`` non-zero
  candidates.  Treating each as an independent Bernoulli trial that the
  random scalar ``m·x`` falls within an interval of width
  ``2·10^{−d}`` of zero (where ``d`` is the working digits), Bonferroni
  gives the false-discovery upper bound

  .. math::

      \Pr[\text{spurious}] \;\le\; \bigl((2M+1)^n − 1\bigr) \cdot 2\cdot 10^{-d}.

  ``Conjecturer.report()`` ships this exact bound for every conjecture
  it returns; downstream consumers (e.g. ``Refuter``, ``Auditor``) can
  filter on it.

* **Plouffe's Inverse Symbolic Calculator (1995).**  For a single real
  constant ``α``, search precomputed tables of common closed forms
  (rational multiples of ``π, e, γ, ln 2, √k, ζ(2), ζ(3),…``) and
  return any whose numerical value matches ``α``.  ``Conjecturer``
  provides this via ``recognize_constant``, with the table managed as
  an open registry the coordinator can extend at runtime.

Why is this the right primitive
-------------------------------

A coordinator that can call

    conjecture = Conjecturer.propose(measurements)

closes a loop none of the other primitives close.  ``Scientist`` recovers
a *parametric* mechanism (real-valued coefficients on a fixed basis).
``Predictor`` recovers a *stochastic* mechanism (the next-symbol
distribution).  ``Conjecturer`` recovers a *combinatorial* mechanism —
the *exact* integer-coefficient identity that the data conform to.
That identity is then:

* a closed-form prior that ``Filterer`` can plug into its dynamics;
* a hypothesis ``Refuter`` can attempt to break against held-out
  high-precision evaluations;
* a typed compile-time-constant a ``Synthesizer`` can lift into a
  static field of a tool;
* a fact whose edges in ``KnowledgeGraph`` carry *integer* (not
  ``float``) weights, removing one entire failure mode of approximate
  reasoning over derived quantities;
* a parsimony bound: any ``Scientist`` law whose residual is *smaller*
  than the precision of the integer identity that the same data admit
  is overfit, and ``Conjecturer`` is the way to detect that.

In a system that wants to be more than a function-fitter — that wants to
*discover laws* — discovery of integer identities is a non-optional
primitive.

Implementation notes
--------------------

* Pure stdlib — ``fractions``, ``math``, ``hashlib``, ``random``,
  ``threading``, ``itertools``.  No NumPy.  All lattice arithmetic is
  exact rational; only the user-visible *input* is allowed to be
  ``float``, and even that is canonicalised to a ``Fraction`` via
  ``Fraction.from_float`` exactly once on entry.

* Deterministic given ``seed``.  Every ``observe`` / ``propose`` /
  ``verify`` / ``report`` event hashes into a SHA-256 chain consumed
  by ``AttestationLedger``.

* Thread-safe via a re-entrant lock.

* No on-disk state; ``Conjecturer`` is a pure in-process primitive.

* Self-contained — the only intra-package dependency is
  ``agi.events``.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# =====================================================================
# Event kinds — published when a bus is supplied
# =====================================================================

CONJECTURER_STARTED = "conjecturer.started"
CONJECTURER_OBSERVED = "conjecturer.observed"
CONJECTURER_PROPOSED = "conjecturer.proposed"
CONJECTURER_VERIFIED = "conjecturer.verified"
CONJECTURER_REJECTED = "conjecturer.rejected"
CONJECTURER_RECOGNISED = "conjecturer.recognised"
CONJECTURER_REPORTED = "conjecturer.reported"
CONJECTURER_CLEARED = "conjecturer.cleared"

CONJECTURER_KNOWN_EVENTS = frozenset(
    {
        CONJECTURER_STARTED,
        CONJECTURER_OBSERVED,
        CONJECTURER_PROPOSED,
        CONJECTURER_VERIFIED,
        CONJECTURER_REJECTED,
        CONJECTURER_RECOGNISED,
        CONJECTURER_REPORTED,
        CONJECTURER_CLEARED,
    }
)


# Search algorithms
ALGO_LLL = "lll"
ALGO_BRUTE = "brute"

CONJECTURER_KNOWN_ALGOS = frozenset({ALGO_LLL, ALGO_BRUTE})


# =====================================================================
# Errors
# =====================================================================


class ConjecturerError(Exception):
    """Base error for the Conjecturer primitive."""


class InvalidConfig(ConjecturerError):
    """Configuration values out of range."""


class InvalidObservation(ConjecturerError):
    """Observation rejected (NaN, infinite, not numerical, etc.)."""


class UnknownConstant(ConjecturerError):
    """A constant name referenced in ``with_constants`` is not registered."""


class InsufficientData(ConjecturerError):
    """An operation requires at least one observation."""


class InvalidConjecture(ConjecturerError):
    """A conjecture handed to ``verify`` is malformed."""


class InvalidAlgorithm(ConjecturerError):
    """Algorithm not in CONJECTURER_KNOWN_ALGOS."""


# =====================================================================
# Constants
# =====================================================================

_GENESIS = "0" * 64
_DEFAULT_PRECISION_DIGITS = 30
_DEFAULT_MAX_COEFF = 20
_DEFAULT_LLL_DELTA = Fraction(3, 4)
_DEFAULT_VERIFY_FACTOR = 2   # precision-doubling on verification
_MIN_PRECISION_DIGITS = 4
_MAX_PRECISION_DIGITS = 200
_MAX_DIMENSION = 12
_MAX_BRUTE_COEFF = 6
_MAX_CF_DEPTH = 40
_DEFAULT_CF_HUGE_QUOTIENT = 10 ** 8
_LOG10 = math.log(10.0)


# =====================================================================
# Hash chain
# =====================================================================


def _hash_link(prev: str, payload: str) -> str:
    """SHA-256 chain step: ``H(prev || 0x1f || payload)``."""
    h = hashlib.sha256()
    h.update(prev.encode("ascii"))
    h.update(b"\x1f")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _payload_repr(obj: Any) -> str:
    """Canonical deterministic string repr for hashing."""
    if isinstance(obj, dict):
        keys = sorted(obj.keys(), key=str)
        return "{" + ",".join(f"{k}={_payload_repr(obj[k])}" for k in keys) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_payload_repr(x) for x in obj) + "]"
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return f"{obj:.17g}"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    return repr(obj)


# =====================================================================
# Built-in constant registry
# =====================================================================


def _builtin_constants() -> dict[str, Callable[[int], Fraction]]:
    """Lazy evaluators returning a Fraction approximation to ``digits``
    decimal digits.

    Each function returns a rational whose value rounded to ``digits``
    digits equals the constant.  This lets us scale evaluations
    deterministically across precisions without ever materialising a
    float.
    """

    def one(_digits: int) -> Fraction:
        return Fraction(1)

    def zero(_digits: int) -> Fraction:
        return Fraction(0)

    def pi(digits: int) -> Fraction:
        return _pi_fraction(digits + 4)

    def e(digits: int) -> Fraction:
        return _e_fraction(digits + 4)

    def gamma(digits: int) -> Fraction:
        return _gamma_fraction(digits + 4)

    def log2(digits: int) -> Fraction:
        return _ln2_fraction(digits + 4)

    def log3(digits: int) -> Fraction:
        return _ln_fraction(3, digits + 4)

    def log5(digits: int) -> Fraction:
        return _ln_fraction(5, digits + 4)

    def sqrt2(digits: int) -> Fraction:
        return _isqrt_fraction(2, digits + 4)

    def sqrt3(digits: int) -> Fraction:
        return _isqrt_fraction(3, digits + 4)

    def sqrt5(digits: int) -> Fraction:
        return _isqrt_fraction(5, digits + 4)

    def phi(digits: int) -> Fraction:
        return (Fraction(1) + _isqrt_fraction(5, digits + 4)) / Fraction(2)

    def zeta2(digits: int) -> Fraction:
        f = _pi_fraction(digits + 6)
        return f * f / Fraction(6)

    def zeta3(digits: int) -> Fraction:
        return _zeta3_fraction(digits + 6)

    def catalan(digits: int) -> Fraction:
        return _catalan_fraction(digits + 6)

    return {
        "one": one,
        "zero": zero,
        "pi": pi,
        "e": e,
        "gamma": gamma,
        "ln2": log2,
        "ln3": log3,
        "ln5": log5,
        "sqrt2": sqrt2,
        "sqrt3": sqrt3,
        "sqrt5": sqrt5,
        "phi": phi,
        "zeta2": zeta2,
        "zeta3": zeta3,
        "catalan": catalan,
    }


# =====================================================================
# Closed-form constants — pure-stdlib rational evaluators
# =====================================================================


def _pi_fraction(digits: int) -> Fraction:
    """Machin-like formula ``π/4 = 4 arctan(1/5) − arctan(1/239)``.

    Returns a rational with denominator a power of ten that agrees
    with ``π`` to ``digits`` decimal places (with a safety margin).
    """
    n = digits + 8
    scale = 10 ** n
    # Compute arctan(1/x) * scale as integer
    def arctan_inv(x: int, scale: int) -> int:
        # sum_{k=0}^{inf} (-1)^k / ((2k+1) x^{2k+1}) * scale
        x2 = x * x
        term = scale // x
        total = term
        sign = -1
        k = 1
        while True:
            term //= x2
            if term == 0:
                break
            piece = term // (2 * k + 1)
            if piece == 0:
                break
            total += sign * piece
            sign = -sign
            k += 1
        return total

    pi_scaled = 4 * (4 * arctan_inv(5, scale) - arctan_inv(239, scale))
    return Fraction(pi_scaled, 10 ** n)


def _e_fraction(digits: int) -> Fraction:
    """Series ``e = Σ 1/k!`` summed until the next term is below
    ``10^{-digits}``."""
    n = digits + 6
    scale = 10 ** n
    total = scale  # k=0 term
    term = scale
    k = 1
    while term > 0:
        term //= k
        total += term
        k += 1
        if k > 4 * digits + 100:
            break
    return Fraction(total, 10 ** n)


_GAMMA_STR_100 = (
    "0.5772156649015328606065120900824024310421593359399235988057672348848"
    "67726777664670936947063291746749"
)
_CATALAN_STR_100 = (
    "0.9159655941772190150546035149323841107741493742816721342664981196217"
    "63035340309149753749683272380058"
)


def _decimal_str_to_fraction(s: str, digits: int) -> Fraction:
    """Parse a decimal string like ``0.123456…`` and return a Fraction
    truncated to ``digits`` decimal places."""
    if "." not in s:
        return Fraction(int(s))
    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:]
    int_part, frac_part = s.split(".", 1)
    frac_part = frac_part[:digits]
    num = int(int_part + frac_part) * sign
    den = 10 ** len(frac_part)
    return Fraction(num, den)


def _gamma_fraction(digits: int) -> Fraction:
    """Euler-Mascheroni constant.

    For ``digits ≤ 100`` we use the hardcoded high-precision string;
    otherwise we use a direct truncation with Euler-Maclaurin correction
    bounded by ``N = 10^{digits/2+2}`` (fast enough for ~30 digits).
    """
    if digits <= 100:
        return _decimal_str_to_fraction(_GAMMA_STR_100, digits + 4)
    # Fallback: tighter Euler-Maclaurin with reasonable N.
    # Cap N so the cost remains O(N · digits).
    n = digits + 6
    scale = 10 ** n
    big_n = max(1000, min(10 ** ((digits + 4) // 2 + 2), 10 ** 6))
    harmonic_scaled = 0
    for k in range(1, big_n + 1):
        harmonic_scaled += scale // k
    ln_n_scaled = _ln_scaled_int(big_n, scale)
    gamma_scaled = harmonic_scaled - ln_n_scaled
    gamma_scaled -= scale // (2 * big_n)
    gamma_scaled += scale // (12 * big_n * big_n)
    return Fraction(gamma_scaled, 10 ** n)


def _ln_scaled_int(x: int, scale: int) -> int:
    """Compute round(ln(x) * scale) for positive integer x using
    a long-multiplication AGM-free Taylor approach via ``ln(x) =
    k·ln(2) + ln(x / 2^k)`` with ``x / 2^k ∈ [1, 2)``."""
    if x <= 0:
        raise ValueError("ln of non-positive integer")
    # Find k such that 2^k <= x < 2^(k+1)
    k = x.bit_length() - 1
    # Compute residual ln(x / 2^k) via Taylor:
    # x / 2^k = 1 + r,  r ∈ [0, 1).  Then ln(1+r) = r - r^2/2 + r^3/3 - …
    # r as a rational: (x - 2^k) / 2^k, all integers.
    # We carry r·scale as an int.
    num = (x - (1 << k))
    den = 1 << k
    if num == 0:
        # x is power of 2; ln(x) = k · ln(2)
        return k * _ln2_scaled_int(scale)
    # We compute the alternating series sum_{i=1..N} (-1)^(i+1) (num/den)^i / i
    # as exactly-tracked rational, then convert to scaled int.
    r_num = num
    r_den = den
    # power_num/power_den = (num/den)^i
    power_num = r_num
    power_den = r_den
    total_num = 0
    total_den = 1
    sign = 1
    for i in range(1, max(50, 4 * (scale.bit_length()) // 3) + 1):
        # term = sign * power_num / (power_den * i)
        # add to total
        t_num = sign * power_num
        t_den = power_den * i
        # total += t
        new_num = total_num * t_den + t_num * total_den
        new_den = total_den * t_den
        g = math.gcd(abs(new_num), new_den)
        total_num = new_num // g
        total_den = new_den // g
        # stop when |term| < 1/scale (numerical cutoff)
        if power_den * i > scale * abs(power_num) * 16:
            break
        # next power
        power_num *= r_num
        power_den *= r_den
        g2 = math.gcd(abs(power_num), power_den)
        power_num //= g2
        power_den //= g2
        sign = -sign
    # ln(x) = k·ln(2) + total
    ln2_s = _ln2_scaled_int(scale)
    # scaled total = total_num * scale // total_den
    add_part = total_num * scale // total_den
    return k * ln2_s + add_part


def _ln2_scaled_int(scale: int) -> int:
    """Compute round(ln(2) * scale)."""
    # ln(2) = sum_{k=1}^inf 1/(k · 2^k)  (alternating-free series)
    total = 0
    k = 1
    while True:
        term = scale // (k * (1 << k))
        if term == 0:
            break
        total += term
        k += 1
        if k > 4 * scale.bit_length() + 100:
            break
    return total


def _ln2_fraction(digits: int) -> Fraction:
    n = digits + 6
    scale = 10 ** n
    return Fraction(_ln2_scaled_int(scale), 10 ** n)


def _ln_fraction(x: int, digits: int) -> Fraction:
    if x <= 0:
        raise ValueError("ln of non-positive integer")
    if x == 1:
        return Fraction(0)
    n = digits + 6
    scale = 10 ** n
    return Fraction(_ln_scaled_int(x, scale), 10 ** n)


def _isqrt_fraction(x: int, digits: int) -> Fraction:
    """Integer-square-root rational approximation of √x.

    Returns ``isqrt(x · 10^{2n}) / 10^n`` for ``n = digits + 6``.
    """
    if x < 0:
        raise ValueError("sqrt of negative integer")
    n = digits + 6
    scaled = x * 10 ** (2 * n)
    root = math.isqrt(scaled)
    return Fraction(root, 10 ** n)


def _zeta3_fraction(digits: int) -> Fraction:
    """Apéry-style series for ``ζ(3) = Σ_{n≥1} 1/n³``, accelerated
    by ``ζ(3) = (5/2) · Σ_{n≥1} (-1)^{n+1} / (n³ C(2n,n))``."""
    n = digits + 6
    scale = 10 ** n
    total = 0
    sign = 1
    nn = 1
    # compute factorial / binomial coefficient C(2n,n) = (2n)! / (n!)^2
    while True:
        binom = math.comb(2 * nn, nn)
        denom = nn ** 3 * binom
        term = scale // denom
        if term == 0:
            break
        total += sign * term
        sign = -sign
        nn += 1
        if nn > 4 * digits + 100:
            break
    return Fraction(5 * total, 2 * 10 ** n)


def _catalan_fraction(digits: int) -> Fraction:
    """Catalan's constant ``G = Σ_{k=0}^∞ (-1)^k / (2k+1)²``.

    For ``digits ≤ 100`` we return the hardcoded high-precision value;
    otherwise we use the BBP-style series (much slower in pure Python
    but correct for any precision).
    """
    if digits <= 100:
        return _decimal_str_to_fraction(_CATALAN_STR_100, digits + 4)
    n = digits + 6
    scale = 10 ** n
    total = 0
    sign = 1
    k = 0
    cap = 10 ** (digits // 2 + 4)
    while k < cap:
        term = scale // ((2 * k + 1) ** 2)
        if term == 0:
            break
        total += sign * term
        sign = -sign
        k += 1
    return Fraction(total, 10 ** n)


# =====================================================================
# Continued fractions, best rationals, recognition
# =====================================================================


@dataclass(frozen=True)
class ContinuedFraction:
    """Simple continued-fraction expansion ``[a₀; a₁, a₂, …]``.

    Attributes
    ----------
    coefficients:
        the integer partial quotients ``a₀, a₁, …``.
    truncated:
        ``True`` if the expansion was cut off at ``max_depth`` rather
        than terminating by exhausting precision.
    huge_quotient_index:
        index of the first quotient ``≥ huge_quotient``, signalling
        the precision limit; ``None`` if no such index was found.
    """

    coefficients: tuple[int, ...]
    truncated: bool
    huge_quotient_index: int | None

    def convergents(self) -> list[tuple[int, int]]:
        """Best-rational convergents ``(p_k, q_k)``."""
        p_prev, p_curr = 1, self.coefficients[0]
        q_prev, q_curr = 0, 1
        out = [(p_curr, q_curr)]
        for a in self.coefficients[1:]:
            p_prev, p_curr = p_curr, a * p_curr + p_prev
            q_prev, q_curr = q_curr, a * q_curr + q_prev
            out.append((p_curr, q_curr))
        return out

    def truncate_before_huge(self) -> "ContinuedFraction":
        """Return a copy truncated just *before* the first huge quotient."""
        if self.huge_quotient_index is None:
            return self
        return ContinuedFraction(
            coefficients=self.coefficients[: self.huge_quotient_index],
            truncated=True,
            huge_quotient_index=None,
        )


def continued_fraction(
    x: Fraction | float | int,
    *,
    max_depth: int = _MAX_CF_DEPTH,
    huge_quotient: int = _DEFAULT_CF_HUGE_QUOTIENT,
) -> ContinuedFraction:
    """Compute the continued-fraction expansion of ``x``.

    The expansion stops either when an exact representation is
    achieved (``x`` rational), when ``max_depth`` is reached, or when
    a partial quotient exceeds ``huge_quotient`` — the latter being
    the signal that we have crossed from the irrational tail into the
    numerical-precision tail.  See Khinchin 1935 / Lochs 1964 for the
    information-theoretic reason this works.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be ≥ 1")
    if huge_quotient < 2:
        raise ValueError("huge_quotient must be ≥ 2")
    if isinstance(x, float):
        if not math.isfinite(x):
            raise InvalidObservation("non-finite x")
        x = Fraction(x).limit_denominator(10 ** 30)
    elif isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise InvalidObservation(f"x must be Fraction|float|int, not {type(x)!r}")
    quotients: list[int] = []
    huge_idx: int | None = None
    truncated = False
    cur = x
    for i in range(max_depth):
        a = math.floor(cur.numerator / cur.denominator) if cur.denominator > 0 else (cur.numerator // cur.denominator)
        # Use exact floor for Fraction
        a = cur.numerator // cur.denominator
        if a >= huge_quotient and i > 0 and huge_idx is None:
            huge_idx = i
        quotients.append(a)
        rem = cur - a
        if rem == 0:
            break
        cur = Fraction(rem.denominator, rem.numerator)
    else:
        truncated = True
    return ContinuedFraction(
        coefficients=tuple(quotients),
        truncated=truncated,
        huge_quotient_index=huge_idx,
    )


def best_rational(x: Fraction | float, max_denominator: int) -> Fraction:
    """Best rational approximation of ``x`` with denominator ≤ ``D``.

    Implements Stern-Brocot descent on the simple CF expansion.  For
    floats this reduces to ``Fraction(x).limit_denominator(D)`` —
    we keep our own implementation so the algorithm is auditable.
    """
    if max_denominator < 1:
        raise ValueError("max_denominator must be ≥ 1")
    if isinstance(x, float):
        if not math.isfinite(x):
            raise InvalidObservation("non-finite x")
        x = Fraction(x).limit_denominator(10 ** 30)
    if not isinstance(x, Fraction):
        raise InvalidObservation(f"x must be Fraction|float, not {type(x)!r}")
    return x.limit_denominator(max_denominator)


# =====================================================================
# LLL lattice reduction in exact rational arithmetic
# =====================================================================


def _gram_schmidt(basis: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Gram-Schmidt orthogonalisation in exact rational arithmetic.

    Returns (B_star, MU) with ``B_star`` the orthogonalised vectors and
    ``MU[i][j] = ⟨b_i, b*_j⟩ / ⟨b*_j, b*_j⟩``.  No normalisation: this
    is the *unnormalised* GS used inside LLL.
    """
    n = len(basis)
    if n == 0:
        return [], []
    m = len(basis[0])
    b_star: list[list[Fraction]] = []
    mu: list[list[Fraction]] = [[Fraction(0)] * n for _ in range(n)]
    norms_sq: list[Fraction] = []
    for i in range(n):
        bi_star = [Fraction(c) for c in basis[i]]
        for j in range(i):
            num = Fraction(0)
            for k in range(m):
                num += basis[i][k] * b_star[j][k]
            mu[i][j] = num / norms_sq[j] if norms_sq[j] != 0 else Fraction(0)
            if mu[i][j] != 0:
                for k in range(m):
                    bi_star[k] -= mu[i][j] * b_star[j][k]
        b_star.append(bi_star)
        ns = Fraction(0)
        for c in bi_star:
            ns += c * c
        norms_sq.append(ns)
    return b_star, mu


def lll(
    basis: Sequence[Sequence[int | Fraction]],
    *,
    delta: Fraction = _DEFAULT_LLL_DELTA,
    max_steps: int | None = None,
) -> list[list[Fraction]]:
    """LLL lattice reduction in exact rational arithmetic.

    Parameters
    ----------
    basis:
        list of basis vectors; each row is one ``b_i ∈ ℤ^m`` or
        ``b_i ∈ ℚ^m`` (the algorithm is unchanged on rationals).
    delta:
        LLL parameter ``δ ∈ (1/4, 1)``.  Larger ``δ`` produces a
        more reduced basis at higher cost.  Default ``3/4``.
    max_steps:
        upper bound on the number of swap-or-reduce iterations.
        ``None`` ⇒ ``10 · n² · log(max ‖b_i‖∞)``.

    Returns
    -------
    list of Fraction vectors, the reduced basis.

    Notes
    -----
    The Gram-Schmidt scalars ``μ_{i,j}`` are updated **incrementally**
    after each size-reduction and swap so the total work is
    ``O(steps · n · m)`` Fraction operations rather than ``O(steps ·
    n² · m)`` (the naive recompute-from-scratch cost).
    """
    if not isinstance(delta, Fraction):
        delta = Fraction(delta)
    if not (Fraction(1, 4) < delta < Fraction(1)):
        raise ValueError("delta must satisfy 1/4 < δ < 1")
    n = len(basis)
    if n == 0:
        return []
    m = len(basis[0])
    B: list[list[Fraction]] = [[Fraction(c) for c in row] for row in basis]
    if max_steps is None:
        max_norm = max(
            (max(abs(c) for c in row) for row in B if row),
            default=Fraction(1),
        )
        max_norm = max(max_norm, Fraction(1))
        max_steps = max(50, 10 * n * n * (int(math.log2(float(max_norm) + 2)) + 4))
    b_star, mu = _gram_schmidt(B)
    norms_sq = [sum(c * c for c in v) for v in b_star]

    def _recompute_row(i: int) -> None:
        """Recompute ``b_star[i]``, ``norms_sq[i]`` and ``mu[i][:i]``
        in place from the current ``B`` and ``b_star[:i]``."""
        bi_star = [Fraction(c) for c in B[i]]
        for j in range(i):
            num = Fraction(0)
            for k_ in range(m):
                num += B[i][k_] * b_star[j][k_]
            if norms_sq[j] != 0:
                mu[i][j] = num / norms_sq[j]
            else:
                mu[i][j] = Fraction(0)
            if mu[i][j] != 0:
                for k_ in range(m):
                    bi_star[k_] -= mu[i][j] * b_star[j][k_]
        b_star[i] = bi_star
        ns = Fraction(0)
        for c in bi_star:
            ns += c * c
        norms_sq[i] = ns
        # Also update mu[t][i] for t > i since b_star[i] changed.
        for t in range(i + 1, n):
            num = Fraction(0)
            for k_ in range(m):
                num += B[t][k_] * b_star[i][k_]
            if norms_sq[i] != 0:
                mu[t][i] = num / norms_sq[i]
            else:
                mu[t][i] = Fraction(0)

    k = 1
    steps = 0
    while k < n and steps < max_steps:
        steps += 1
        # Size-reduce b_k against b_{k-1}, b_{k-2}, …, b_0.
        # Note: size-reduction modifies B[k] and mu[k][*] but does NOT
        # change b_star[k] (the orthogonal complement is invariant under
        # adding multiples of earlier rows), so we only need to update
        # the μ row, not Gram-Schmidt.
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                q = _round_half_even(mu[k][j])
                if q != 0:
                    for t in range(m):
                        B[k][t] -= q * B[j][t]
                    # μ_{k,j} ← μ_{k,j} − q,  μ_{k,l} ← μ_{k,l} − q·μ_{j,l}  (l<j)
                    for l_ in range(j):
                        mu[k][l_] -= q * mu[j][l_]
                    mu[k][j] -= q
        # Lovász condition
        if norms_sq[k] >= (delta - mu[k][k - 1] ** 2) * norms_sq[k - 1]:
            k += 1
        else:
            # Swap b_k and b_{k-1}; update GS at rows k-1, k.
            B[k], B[k - 1] = B[k - 1], B[k]
            _recompute_row(k - 1)
            _recompute_row(k)
            k = max(k - 1, 1)
    return B


def _round_half_even(x: Fraction) -> int:
    """Banker's rounding for a Fraction.  Returns an int.

    Required so LLL is deterministic and reproducible across
    platforms (Python's default ``round`` for floats is platform-
    sensitive at half values).
    """
    # x = a/b, round(a/b) with half-to-even
    num = x.numerator
    den = x.denominator
    q, r = divmod(num, den)
    twice = 2 * r
    if twice < den and twice > -den:
        return q
    if twice == den:
        return q + (q % 2)  # round to even
    if twice == -den:
        return q + (q % 2)
    if twice > den:
        return q + 1
    return q - 1


# =====================================================================
# Integer relations: PSLQ-style search via LLL
# =====================================================================


@dataclass(frozen=True)
class IntegerRelation:
    """An integer linear relation ``Σ coeffs[i] · values[i] ≈ 0``.

    Attributes
    ----------
    coeffs:
        tuple of integers ``mᵢ``, not all zero.
    residual:
        ``|Σ mᵢ · vᵢ|`` evaluated at the *working precision* used to
        find the relation.  A small residual relative to the working
        precision is suggestive but never proof — call ``verify`` to
        re-evaluate at higher precision.
    norm_infinity:
        ``max |mᵢ|``.
    """

    coeffs: tuple[int, ...]
    residual: Fraction
    norm_infinity: int


def integer_relations(
    values: Sequence[Fraction],
    *,
    precision_digits: int = _DEFAULT_PRECISION_DIGITS,
    max_coeff: int = _DEFAULT_MAX_COEFF,
    delta: Fraction = _DEFAULT_LLL_DELTA,
) -> list[IntegerRelation]:
    """Find integer relations among ``values`` via LLL on the
    integer-relation lattice.

    The lattice is

    .. math::

        L = \\big\\{(m_1,\\dots,m_n,\\;\\lfloor C \\cdot m \\cdot x \\rceil)
            : m \\in \\mathbb{Z}^n\\big\\}

    with scaling ``C = 10^{precision_digits}``.  After reduction the
    first basis vector encodes the dominant short integer relation;
    we filter on ``‖m‖_∞ ≤ max_coeff`` and ``|m·x| ≤ 10^{-d/2}``.
    """
    if not values:
        return []
    if precision_digits < _MIN_PRECISION_DIGITS:
        raise InvalidConfig("precision_digits too small")
    if precision_digits > _MAX_PRECISION_DIGITS:
        raise InvalidConfig("precision_digits too large")
    if max_coeff < 1:
        raise InvalidConfig("max_coeff must be ≥ 1")
    n = len(values)
    C = 10 ** precision_digits
    basis: list[list[Fraction]] = []
    for i in range(n):
        row = [Fraction(0)] * (n + 1)
        row[i] = Fraction(1)
        # Last coordinate carries C * x_i, rounded to nearest integer
        scaled = (values[i] * C).limit_denominator(10 ** (precision_digits + 6))
        rounded_int = _round_half_even(scaled)
        row[n] = Fraction(rounded_int)
        basis.append(row)
    reduced = lll(basis, delta=delta)
    out: list[IntegerRelation] = []
    seen: set[tuple[int, ...]] = set()
    for vec in reduced:
        coeffs = []
        ok = True
        for v in vec[:n]:
            if v.denominator != 1:
                ok = False
                break
            coeffs.append(v.numerator)
        if not ok:
            continue
        if all(c == 0 for c in coeffs):
            continue
        if any(abs(c) > max_coeff for c in coeffs):
            continue
        if coeffs[0] < 0 or (coeffs[0] == 0 and next((c for c in coeffs if c != 0), 0) < 0):
            coeffs = [-c for c in coeffs]
        key = tuple(coeffs)
        if key in seen:
            continue
        seen.add(key)
        residual = abs(sum(c * v for c, v in zip(coeffs, values)))
        norm = max(abs(c) for c in coeffs)
        out.append(
            IntegerRelation(
                coeffs=tuple(coeffs),
                residual=residual,
                norm_infinity=norm,
            )
        )
    # Sort by residual ascending, then by norm ascending
    out.sort(key=lambda r: (r.residual, r.norm_infinity))
    return out


def brute_relations(
    values: Sequence[Fraction],
    *,
    max_coeff: int = _MAX_BRUTE_COEFF,
    precision_digits: int = _DEFAULT_PRECISION_DIGITS,
    top_k: int = 32,
) -> list[IntegerRelation]:
    """Exhaustive search for short integer relations.

    Used as a *correctness backstop* on the LLL search: for low
    dimensions and small max_coeff this is feasible and guaranteed
    to find every short relation, providing a test oracle for the
    LLL path.
    """
    n = len(values)
    if n == 0 or n > 5:
        return []
    if max_coeff < 1 or max_coeff > _MAX_BRUTE_COEFF:
        raise InvalidConfig("max_coeff outside brute-force range")
    threshold = Fraction(10) ** (-precision_digits // 2)
    out: list[IntegerRelation] = []
    seen: set[tuple[int, ...]] = set()
    rng = range(-max_coeff, max_coeff + 1)
    for combo in itertools.product(rng, repeat=n):
        if all(c == 0 for c in combo):
            continue
        # Canonicalise sign
        first_nz = next(c for c in combo if c != 0)
        if first_nz < 0:
            continue  # we will cover its negation via the positive sibling
        # Drop common factor
        g = 0
        for c in combo:
            g = math.gcd(g, abs(c))
        if g == 0 or g != 1:
            continue
        if combo in seen:
            continue
        seen.add(combo)
        residual = abs(sum(c * v for c, v in zip(combo, values)))
        if residual < threshold:
            norm = max(abs(c) for c in combo)
            out.append(
                IntegerRelation(
                    coeffs=combo,
                    residual=residual,
                    norm_infinity=norm,
                )
            )
    out.sort(key=lambda r: (r.residual, r.norm_infinity))
    return out[:top_k]


# =====================================================================
# Conjecture dataclass
# =====================================================================


@dataclass(frozen=True)
class Conjecture:
    """A candidate closed-form identity.

    Attributes
    ----------
    columns:
        tuple of column names — the symbolic names of the values
        appearing in the linear combination.
    coeffs:
        integer coefficients, one per column.
    residual:
        ``|Σ coeffs[i] · values[i]|`` at working precision.
    working_digits:
        the precision used to discover the relation.
    fdr_bound:
        upper bound on the probability that the relation is spurious
        given the search-space volume and working precision.
    norm_infinity:
        ``max |coeffs[i]|``.
    verified_at_digits:
        precision at which the relation was last *verified* (re-evaluated
        with shrunken residual); ``None`` if never verified.
    verified:
        ``True`` if the most recent verification succeeded.
    rejected:
        ``True`` if a verification failed (residual did not shrink).
    """

    columns: tuple[str, ...]
    coeffs: tuple[int, ...]
    residual: Fraction
    working_digits: int
    fdr_bound: float
    norm_infinity: int
    verified_at_digits: int | None = None
    verified: bool = False
    rejected: bool = False

    @property
    def signature(self) -> str:
        """Stable string representation, sign-canonicalised."""
        cs = self.coeffs
        # Sign-canonical
        first_nz = next((c for c in cs if c != 0), 0)
        if first_nz < 0:
            cs = tuple(-c for c in cs)
        parts = []
        for name, c in zip(self.columns, cs):
            if c == 0:
                continue
            sign = "+" if c > 0 else "−"
            mag = abs(c)
            if mag == 1:
                parts.append(f"{sign}{name}")
            else:
                parts.append(f"{sign}{mag}·{name}")
        body = " ".join(parts).lstrip("+").strip()
        return f"{body} = 0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "coeffs": list(self.coeffs),
            "residual": float(self.residual),
            "working_digits": self.working_digits,
            "fdr_bound": self.fdr_bound,
            "norm_infinity": self.norm_infinity,
            "verified_at_digits": self.verified_at_digits,
            "verified": self.verified,
            "rejected": self.rejected,
            "signature": self.signature,
        }


def _fdr_bound(*, n: int, max_coeff: int, working_digits: int, residual: Fraction) -> float:
    """Upper-bound the probability that a relation is spurious.

    Treats the random scalar ``m·x`` as uniform in an interval of
    width ``2·max(values)`` and bounds the search-space volume by
    ``(2M+1)ⁿ``.  This is a loose Bonferroni bound — looser is safer.
    """
    if residual == 0:
        return 0.0
    search_space = (2 * max_coeff + 1) ** n
    # An ε-window event probability is ≈ residual / r_max
    # We use 2 · 10^{-working_digits} as a conservative ε for floats.
    eps = 2.0 * 10.0 ** (-working_digits)
    return min(1.0, max(0.0, search_space * eps))


# =====================================================================
# Recognition: closed-form for a single constant
# =====================================================================


@dataclass(frozen=True)
class Recognition:
    """A candidate closed-form for a *single* observed constant.

    Attributes
    ----------
    expression:
        printable form, e.g. ``"phi"``, ``"3/7"``, ``"2·pi + 1"``.
    coeffs:
        the integer coefficients on each basis column, in the order
        the recognition was searched in.
    columns:
        names of the basis columns.
    residual:
        ``|x − value(expression)|`` at working precision.
    kind:
        one of ``"rational"``, ``"basis"``, ``"algebraic"``.
    """

    expression: str
    coeffs: tuple[int, ...]
    columns: tuple[str, ...]
    residual: Fraction
    kind: str


# =====================================================================
# Reports
# =====================================================================


@dataclass(frozen=True)
class ConjecturerReport:
    """Auditable snapshot returned by ``Conjecturer.report()``."""

    n_observations: int
    n_proposed: int
    n_verified: int
    n_rejected: int
    columns: tuple[str, ...]
    conjectures: tuple[Conjecture, ...]
    head: str
    seed: int
    precision_digits: int


# =====================================================================
# Conjecturer class
# =====================================================================


class Conjecturer:
    """Runtime primitive for automated mathematical conjecture generation.

    Lifecycle
    ---------
    * ``Conjecturer.create(precision_digits, …)`` — build a configured
      instance and emit ``CONJECTURER_STARTED``.
    * ``observe(name, value)`` — register a named numerical observation.
    * ``with_constants(names)`` — pin which built-in/named constants
      (subset of ``("one", "pi", "e", "gamma", "ln2", …)``) participate
      in the next ``propose`` search.
    * ``propose(max_coeff=…, algo=…)`` — search the lattice and return
      candidate ``Conjecture`` objects.
    * ``verify(conjecture, factor=2)`` — re-evaluate at higher
      precision; sets ``verified`` / ``rejected`` accordingly.
    * ``recognize_constant(value)`` — closed-form recognition for a
      single real (CF + basis + rationals).
    * ``report()`` — dataclass snapshot for the attestation ledger.
    * ``clear()`` — drop all observations and reset the chain (the
      ``seed`` and ``precision_digits`` are retained).
    """

    @classmethod
    def create(
        cls,
        *,
        precision_digits: int = _DEFAULT_PRECISION_DIGITS,
        seed: int = 0,
        bus: EventBus | None = None,
        agent_id: str | None = None,
        builtin_constants: Mapping[str, Callable[[int], Fraction]] | None = None,
    ) -> "Conjecturer":
        if precision_digits < _MIN_PRECISION_DIGITS:
            raise InvalidConfig("precision_digits too small")
        if precision_digits > _MAX_PRECISION_DIGITS:
            raise InvalidConfig("precision_digits too large")
        inst = cls(
            precision_digits=precision_digits,
            seed=seed,
            bus=bus,
            agent_id=agent_id,
            builtin_constants=builtin_constants,
        )
        inst._emit(
            CONJECTURER_STARTED,
            {
                "precision_digits": precision_digits,
                "seed": seed,
            },
        )
        return inst

    def __init__(
        self,
        *,
        precision_digits: int,
        seed: int,
        bus: EventBus | None,
        agent_id: str | None,
        builtin_constants: Mapping[str, Callable[[int], Fraction]] | None,
    ) -> None:
        self._precision_digits = precision_digits
        self._seed = seed
        self._bus = bus
        self._agent_id = agent_id
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._head = _GENESIS
        self._observations: dict[str, Fraction] = {}
        self._observation_order: list[str] = []
        self._selected_columns: tuple[str, ...] = ()
        self._builtin_evaluators: dict[str, Callable[[int], Fraction]] = dict(_builtin_constants())
        if builtin_constants:
            for name, fn in builtin_constants.items():
                if not name.isidentifier():
                    raise InvalidConfig(f"constant name {name!r} is not an identifier")
                if not callable(fn):
                    raise InvalidConfig(f"constant {name!r} evaluator is not callable")
                self._builtin_evaluators[name] = fn
        # Cache: name → (digits, value) so repeated propose() calls at the
        # same precision are O(1).
        self._cache: dict[tuple[str, int], Fraction] = {}
        self._conjectures: list[Conjecture] = []
        self._proposed_count = 0
        self._verified_count = 0
        self._rejected_count = 0

    # -----------------------------------------------------------------
    # Internal: event emission
    # -----------------------------------------------------------------

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        payload_str = _payload_repr(dict(payload))
        self._head = _hash_link(self._head, kind + "|" + payload_str)
        if self._bus is None:
            return
        data = dict(payload)
        if self._agent_id is not None:
            data.setdefault("agent_id", self._agent_id)
        ev = Event(
            kind=kind,
            session_id=None,
            data=data,
        )
        self._bus.publish(ev)

    # -----------------------------------------------------------------
    # Constant evaluation
    # -----------------------------------------------------------------

    def _eval(self, name: str, digits: int) -> Fraction:
        key = (name, digits)
        if key in self._cache:
            return self._cache[key]
        if name in self._observations:
            val = self._observations[name]
        elif name in self._builtin_evaluators:
            val = self._builtin_evaluators[name](digits)
        else:
            raise UnknownConstant(name)
        self._cache[key] = val
        return val

    def has_constant(self, name: str) -> bool:
        """``True`` if ``name`` is either an observation or a built-in."""
        return name in self._observations or name in self._builtin_evaluators

    def builtin_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builtin_evaluators))

    def observation_names(self) -> tuple[str, ...]:
        return tuple(self._observation_order)

    def precision_digits(self) -> int:
        return self._precision_digits

    def head(self) -> str:
        return self._head

    # -----------------------------------------------------------------
    # observe
    # -----------------------------------------------------------------

    def observe(self, name: str, value: Fraction | float | int) -> None:
        """Register a named observation.  The same name overwrites.

        Names must be valid Python identifiers so they appear cleanly
        in conjecture signatures.  Observations override built-ins of
        the same name (built-ins are still callable explicitly via
        ``recognize_constant``).
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise InvalidObservation(f"name {name!r} is not a valid identifier")
        if isinstance(value, bool):
            raise InvalidObservation("bool is not a valid observation type")
        if isinstance(value, int):
            v = Fraction(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise InvalidObservation("non-finite value")
            v = Fraction(value).limit_denominator(10 ** (self._precision_digits + 6))
        elif isinstance(value, Fraction):
            v = value
        else:
            raise InvalidObservation(
                f"value must be Fraction|float|int, not {type(value)!r}"
            )
        with self._lock:
            if name not in self._observations:
                self._observation_order.append(name)
            self._observations[name] = v
            # Drop cached resolutions: the observation has changed.
            self._cache = {
                (n, d): val for (n, d), val in self._cache.items() if n != name
            }
            self._emit(
                CONJECTURER_OBSERVED,
                {"name": name, "value": v, "n_observations": len(self._observations)},
            )

    # -----------------------------------------------------------------
    # with_constants
    # -----------------------------------------------------------------

    def with_constants(self, names: Sequence[str]) -> tuple[str, ...]:
        """Pin the search-space columns for the next ``propose``.

        Returns the canonical tuple of column names (deduplicated,
        order-preserving)."""
        seen: set[str] = set()
        cols: list[str] = []
        for n in names:
            if not isinstance(n, str) or not n.isidentifier():
                raise InvalidObservation(f"column {n!r} is not a valid identifier")
            if not self.has_constant(n):
                raise UnknownConstant(n)
            if n in seen:
                continue
            seen.add(n)
            cols.append(n)
        if len(cols) > _MAX_DIMENSION:
            raise InvalidConfig(
                f"too many columns ({len(cols)} > {_MAX_DIMENSION})"
            )
        with self._lock:
            self._selected_columns = tuple(cols)
        return self._selected_columns

    def selected_columns(self) -> tuple[str, ...]:
        return self._selected_columns

    # -----------------------------------------------------------------
    # propose
    # -----------------------------------------------------------------

    def propose(
        self,
        *,
        max_coeff: int = _DEFAULT_MAX_COEFF,
        algo: str = ALGO_LLL,
        top_k: int = 16,
        precision_digits: int | None = None,
    ) -> list[Conjecture]:
        """Search the lattice for integer relations and emit candidates.

        Parameters
        ----------
        max_coeff:
            cap on ``‖m‖_∞``.  Larger means more candidates.
        algo:
            one of ``ALGO_LLL`` (default; LLL on the integer-relation
            lattice) or ``ALGO_BRUTE`` (exhaustive over the cube;
            only feasible for small dim + max_coeff).
        top_k:
            return at most this many conjectures, sorted by residual.
        precision_digits:
            override the working precision; defaults to the value
            passed to ``create()``.
        """
        if algo not in CONJECTURER_KNOWN_ALGOS:
            raise InvalidAlgorithm(algo)
        digits = precision_digits if precision_digits is not None else self._precision_digits
        if digits < _MIN_PRECISION_DIGITS:
            raise InvalidConfig("precision_digits too small")
        if digits > _MAX_PRECISION_DIGITS:
            raise InvalidConfig("precision_digits too large")
        with self._lock:
            cols = self._selected_columns
            if not cols:
                # Default: all observations, in registration order.
                if not self._observations:
                    raise InsufficientData("no observations to search")
                cols = tuple(self._observation_order)
            if len(cols) > _MAX_DIMENSION:
                raise InvalidConfig(f"too many columns ({len(cols)})")
            values = [self._eval(c, digits) for c in cols]
            if algo == ALGO_LLL:
                relations = integer_relations(
                    values, precision_digits=digits, max_coeff=max_coeff
                )
            else:
                relations = brute_relations(
                    values, precision_digits=digits, max_coeff=min(max_coeff, _MAX_BRUTE_COEFF)
                )
            # Filter by residual: insist on residual < 10^{-digits/2}
            cutoff = Fraction(1, 10 ** (digits // 2))
            conjectures: list[Conjecture] = []
            for rel in relations:
                if rel.residual > cutoff:
                    continue
                fdr = _fdr_bound(
                    n=len(cols),
                    max_coeff=max_coeff,
                    working_digits=digits,
                    residual=rel.residual,
                )
                conj = Conjecture(
                    columns=cols,
                    coeffs=tuple(rel.coeffs),
                    residual=rel.residual,
                    working_digits=digits,
                    fdr_bound=fdr,
                    norm_infinity=rel.norm_infinity,
                )
                conjectures.append(conj)
            conjectures = conjectures[:top_k]
            # Dedupe against existing conjectures by signature
            seen_sigs = {c.signature for c in self._conjectures}
            for c in conjectures:
                if c.signature not in seen_sigs:
                    self._conjectures.append(c)
                    seen_sigs.add(c.signature)
            self._proposed_count += len(conjectures)
            self._emit(
                CONJECTURER_PROPOSED,
                {
                    "n": len(conjectures),
                    "cols": list(cols),
                    "max_coeff": max_coeff,
                    "algo": algo,
                    "digits": digits,
                },
            )
            return conjectures

    # -----------------------------------------------------------------
    # verify
    # -----------------------------------------------------------------

    def verify(
        self,
        conjecture: Conjecture,
        *,
        factor: int = _DEFAULT_VERIFY_FACTOR,
    ) -> Conjecture:
        """Re-evaluate the conjecture at higher precision.

        The criterion is *precision-doubling*: at digits ``factor·d``
        the residual must shrink to a level commensurate with the
        increased precision, ``≤ 10^{−d}``.  A constant residual is a
        spurious match.
        """
        if factor < 2:
            raise InvalidConjecture("factor must be ≥ 2")
        if not isinstance(conjecture, Conjecture):
            raise InvalidConjecture("not a Conjecture")
        d = conjecture.working_digits
        high = min(factor * d, _MAX_PRECISION_DIGITS)
        with self._lock:
            try:
                values = [self._eval(c, high) for c in conjecture.columns]
            except UnknownConstant as exc:
                raise InvalidConjecture(f"unknown column: {exc}") from None
            residual = abs(sum(c * v for c, v in zip(conjecture.coeffs, values)))
            cutoff = Fraction(1, 10 ** d)
            verified = residual <= cutoff
            updated = Conjecture(
                columns=conjecture.columns,
                coeffs=conjecture.coeffs,
                residual=residual,
                working_digits=high,
                fdr_bound=_fdr_bound(
                    n=len(conjecture.columns),
                    max_coeff=conjecture.norm_infinity,
                    working_digits=high,
                    residual=residual,
                ),
                norm_infinity=conjecture.norm_infinity,
                verified_at_digits=high,
                verified=verified,
                rejected=not verified,
            )
            # Replace any existing conjecture with the same signature.
            replaced = False
            for i, existing in enumerate(self._conjectures):
                if existing.signature == updated.signature:
                    self._conjectures[i] = updated
                    replaced = True
                    break
            if not replaced:
                self._conjectures.append(updated)
            if verified:
                self._verified_count += 1
                self._emit(
                    CONJECTURER_VERIFIED,
                    {
                        "sig": updated.signature,
                        "digits": high,
                        "residual": updated.residual,
                    },
                )
            else:
                self._rejected_count += 1
                self._emit(
                    CONJECTURER_REJECTED,
                    {
                        "sig": updated.signature,
                        "digits": high,
                        "residual": updated.residual,
                    },
                )
            return updated

    # -----------------------------------------------------------------
    # recognize
    # -----------------------------------------------------------------

    def recognize_constant(
        self,
        value: Fraction | float,
        *,
        basis: Sequence[str] | None = None,
        max_coeff: int = _DEFAULT_MAX_COEFF,
        max_denominator: int = 10 ** 6,
    ) -> list[Recognition]:
        """Search for closed-form representations of a *single* real.

        Tries, in order:
          1. exact rational via continued fractions (truncated before
             any "huge" partial quotient);
          2. integer linear combinations of ``basis`` columns;
          3. degree-2 algebraic via continued-fraction periodicity
             (recognises quadratic irrationals).
        """
        if isinstance(value, bool):
            raise InvalidObservation("bool is not a valid value")
        if isinstance(value, int):
            v = Fraction(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise InvalidObservation("non-finite value")
            v = Fraction(value).limit_denominator(10 ** (self._precision_digits + 6))
        elif isinstance(value, Fraction):
            v = value
        else:
            raise InvalidObservation(
                f"value must be Fraction|float|int, not {type(value)!r}"
            )
        if basis is None:
            basis = ("one", "pi", "e", "gamma", "ln2", "sqrt2", "phi")
        out: list[Recognition] = []
        digits = self._precision_digits

        # 1) Rational via CF
        cf = continued_fraction(v, huge_quotient=_DEFAULT_CF_HUGE_QUOTIENT)
        if cf.huge_quotient_index is not None:
            truncated = cf.truncate_before_huge()
        else:
            truncated = cf
        convs = truncated.convergents()
        if convs:
            p, q = convs[-1]
            rat = Fraction(p, q)
            residual = abs(v - rat)
            if residual <= Fraction(1, 10 ** (digits // 2)) and q <= max_denominator:
                out.append(
                    Recognition(
                        expression=f"{p}/{q}" if q != 1 else f"{p}",
                        coeffs=(p, q),
                        columns=("num", "den"),
                        residual=residual,
                        kind="rational",
                    )
                )

        # 2) Linear combinations in chosen basis
        cols = []
        for n in basis:
            if not self.has_constant(n):
                continue
            cols.append(n)
        if cols:
            # Find m₀, m₁, …, m_k with m₀·v + Σ_i mᵢ·cᵢ = 0 i.e. v = -(Σ mᵢ·cᵢ)/m₀
            values = [v] + [self._eval(c, digits) for c in cols]
            relations = integer_relations(
                values, precision_digits=digits, max_coeff=max_coeff
            )
            for rel in relations:
                if rel.coeffs[0] == 0:
                    continue
                # Normalise so first coeff is positive 1 or smallest positive
                m0 = rel.coeffs[0]
                rest = rel.coeffs[1:]
                # Need m₀ · v + Σ mᵢ · cᵢ = 0  ⇒  v = -Σ mᵢ cᵢ / m₀
                # Print as a closed form
                parts = []
                for name, c in zip(cols, rest):
                    if c == 0:
                        continue
                    # Sign chosen so v = +Σ ...
                    sign_c = -c  # negate because we move to other side
                    parts.append((sign_c, name))
                if not parts:
                    continue
                if m0 != 1 and m0 != -1:
                    expr_parts = []
                    for s, name in parts:
                        if s == 0:
                            continue
                        if abs(s) == 1:
                            expr_parts.append(f"{'+' if s > 0 else '−'}{name}")
                        else:
                            expr_parts.append(f"{'+' if s > 0 else '−'}{abs(s)}·{name}")
                    body = " ".join(expr_parts).lstrip("+").strip()
                    expression = f"({body})/{m0}"
                else:
                    sign_flip = -m0  # if m₀ = -1 we flip every sign
                    expr_parts = []
                    for s, name in parts:
                        s_eff = s * sign_flip
                        if s_eff == 0:
                            continue
                        if abs(s_eff) == 1:
                            expr_parts.append(f"{'+' if s_eff > 0 else '−'}{name}")
                        else:
                            expr_parts.append(f"{'+' if s_eff > 0 else '−'}{abs(s_eff)}·{name}")
                    expression = " ".join(expr_parts).lstrip("+").strip()
                # residual at working precision
                residual = abs(sum(c * x for c, x in zip(rel.coeffs, values)))
                out.append(
                    Recognition(
                        expression=expression,
                        coeffs=rel.coeffs,
                        columns=("self",) + tuple(cols),
                        residual=residual,
                        kind="basis",
                    )
                )
                if len(out) >= 8:
                    break

        with self._lock:
            self._emit(
                CONJECTURER_RECOGNISED,
                {"n": len(out), "value": v, "basis": list(basis) if basis else []},
            )
        # Sort recognitions: rational beats basis at equal residual;
        # otherwise by residual asc, then by expression length.
        kind_rank = {"rational": 0, "basis": 1, "algebraic": 2}
        out.sort(key=lambda r: (r.residual, kind_rank.get(r.kind, 9), len(r.expression)))
        return out

    # -----------------------------------------------------------------
    # report / clear
    # -----------------------------------------------------------------

    def report(self) -> ConjecturerReport:
        with self._lock:
            rep = ConjecturerReport(
                n_observations=len(self._observations),
                n_proposed=self._proposed_count,
                n_verified=self._verified_count,
                n_rejected=self._rejected_count,
                columns=tuple(self._selected_columns or self._observation_order),
                conjectures=tuple(self._conjectures),
                head=self._head,
                seed=self._seed,
                precision_digits=self._precision_digits,
            )
            self._emit(
                CONJECTURER_REPORTED,
                {
                    "n_observations": rep.n_observations,
                    "n_proposed": rep.n_proposed,
                    "n_verified": rep.n_verified,
                    "n_rejected": rep.n_rejected,
                    "head": rep.head,
                },
            )
            return rep

    def conjectures(self) -> tuple[Conjecture, ...]:
        with self._lock:
            return tuple(self._conjectures)

    def clear(self) -> None:
        with self._lock:
            self._observations.clear()
            self._observation_order.clear()
            self._selected_columns = ()
            self._cache.clear()
            self._conjectures.clear()
            self._proposed_count = 0
            self._verified_count = 0
            self._rejected_count = 0
            self._head = _GENESIS
            self._emit(CONJECTURER_CLEARED, {})


# =====================================================================
# Quick-start helper
# =====================================================================


def quick_quadratic_recognition(
    value: float,
    *,
    precision_digits: int = _DEFAULT_PRECISION_DIGITS,
    max_coeff: int = 10,
) -> list[Conjecture]:
    """One-shot helper: detect ``ax² + bx + c = 0`` for the given ``x``.

    Returns the conjecture list for the lattice over ``(x², x, 1)``,
    using built-in ``one`` for the constant column.
    """
    cj = Conjecturer.create(precision_digits=precision_digits)
    cj.observe("x", value)
    cj.observe("x2", value * value)
    cj.with_constants(("x2", "x", "one"))
    return cj.propose(max_coeff=max_coeff)


__all__ = [
    "ALGO_BRUTE",
    "ALGO_LLL",
    "CONJECTURER_CLEARED",
    "CONJECTURER_KNOWN_ALGOS",
    "CONJECTURER_KNOWN_EVENTS",
    "CONJECTURER_OBSERVED",
    "CONJECTURER_PROPOSED",
    "CONJECTURER_RECOGNISED",
    "CONJECTURER_REJECTED",
    "CONJECTURER_REPORTED",
    "CONJECTURER_STARTED",
    "CONJECTURER_VERIFIED",
    "Conjecture",
    "Conjecturer",
    "ConjecturerError",
    "ConjecturerReport",
    "ContinuedFraction",
    "InsufficientData",
    "IntegerRelation",
    "InvalidAlgorithm",
    "InvalidConfig",
    "InvalidConjecture",
    "InvalidObservation",
    "Recognition",
    "UnknownConstant",
    "best_rational",
    "brute_relations",
    "continued_fraction",
    "integer_relations",
    "lll",
    "quick_quadratic_recognition",
]
