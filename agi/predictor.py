r"""Predictor — universal sequence prediction via Context Tree Weighting
(CTW), as a runtime primitive.

Every primitive that already lives in this runtime predicts *something*:
``Forecaster`` predicts calibrated outcome probabilities; ``Hedger``
predicts which expert will win; ``Abductor`` predicts which hypothesis
explains a fixed observation; ``Filterer`` predicts a hidden state;
``Compressor`` predicts which finite model class compresses the data
shortest.  Each one is *parametric* — it commits to a model family
(experts, hypotheses, latent SSM) before it sees the data.

The *non-parametric* question — "given a stream of symbols from an
**unknown source**, what is the predictive distribution of the next
symbol, with a universal-redundancy certificate that holds against
any tree source up to depth ``D``?" — is the universal-prediction
question (Solomonoff 1964; Cover-Thomas 1991; Rissanen 1984).  It is
the operation that turns a coordination engine from "I know my model
family" into "I will be no worse than the best model in a rich
non-parametric class — and I can prove it".

The ``Predictor`` is the runtime primitive that answers that
question.  It implements **Context Tree Weighting** (Willems-
Shtarkov-Tjalkens 1995 *The Context-Tree Weighting Method: Basic
Properties*; Willems 1998 *The Context-Tree Weighting Method:
Extensions*), the exact-mixture sequential predictor that *averages
over an exponentially large class of variable-order Markov models*
(every context tree of depth ≤ ``D``) in **O(D) per-symbol time**
and yields a strong redundancy guarantee:

  For every binary sequence ``x_1^n`` and every tree source ``S`` of
  depth ≤ ``D`` with ``|S|`` leaves,

      ``-log_2 P_CTW(x_1^n) ≤ -log_2 P_S(x_1^n) + Γ(|S|, D, n)``,

  where the redundancy ``Γ`` is the sum of a **parameter
  redundancy** ``(|S|/2) log_2(n / |S|) + O(|S|)`` (Krichevsky-
  Trofimov 1981 *The performance of universal encoding*) and a
  **model redundancy** ``|S| + (|S| - 1)``.  Per symbol the redundancy
  vanishes as ``O(log n / n)``.  *No* parametric primitive in this
  runtime achieves this against a class this large.

The pitch reduced to a runtime call::

    pred = Predictor.create(alphabet_size=2, depth=8, seed=0)
    for s in stream:
        pred.observe(s)
    p = pred.predict()                     # {0: p0, 1: p1}
    code_len = pred.code_length_bits()     # universal code length
    map_tree = pred.map_tree()             # Context-Tree Maximisation
    entropy = pred.entropy_rate_estimate() # bits per symbol
    bound = pred.redundancy_bound()        # universal certificate
    e = pred.e_process_vs_uniform()        # anytime-valid e-process
    report = pred.report()                 # everything + receipts

Every ``observe`` and every ``predict`` is hashed into a SHA-256
fingerprint chain compatible with ``AttestationLedger``.

Mathematical roots
------------------

* **Krichevsky-Trofimov 1981.**  The KT estimator is the universal
  Dirichlet-Bayes predictor with prior ``Dir(1/2, …, 1/2)`` over the
  per-context symbol distribution.  Its *parameter redundancy* against
  the best memoryless source is

      ``-log_2 P_KT(x) - (-log_2 P_θ*(x)) ≤ ((A-1)/2) log_2 n + O(1)``,

  for alphabet size ``A``, and is *minimax-optimal* — no other
  predictor achieves smaller worst-case redundancy on memoryless
  sources (Xie-Barron 1997 *Minimax redundancy for the class of
  memoryless sources*).

* **Willems-Shtarkov-Tjalkens 1995.**  The Context-Tree Weighting
  recursion mixes KT estimators at every node of the complete depth-
  ``D`` binary tree with prior weight ``1/2`` on "this node is a leaf"
  versus ``1/2`` on "this node splits".  The weighted probability at
  node ``s`` (context string) satisfies

      ``P_w(s) = (1/2) P_KT(s) + (1/2) P_w(0s) P_w(1s)``  for ``|s| < D``,
      ``P_w(s) = P_KT(s)``                                for ``|s| = D``.

  The root's weighted probability ``P_w(ε)`` is the Bayesian mixture
  over **all** tree models of depth ≤ ``D``.  The model redundancy
  is ``|S| + (|S|-1)`` bits — at most ``2|S|`` — independent of ``n``.

* **Krichevsky-Trofimov sequential update.**  In log-domain,

      ``log P_KT(x_1^{n+1}) - log P_KT(x_1^n) = log((c_{x_{n+1}} + 1/2) /
                                                  (n + A/2))``

  where ``c_a`` is the count of symbol ``a`` so far.  Per-symbol update
  is ``O(1)``; per-context update along the depth-``D`` suffix chain
  is ``O(D)``.

* **Willems 1998 — Extensions.**  The same recursion in log-domain,
  with internal-node mixing via ``log P_w = log((1/2)·exp(log P_KT) +
  (1/2)·exp(log P_w^{0} + log P_w^{1}))`` — ships log-sum-exp stable.

* **Volf-Willems 1998; Willems 1996; Veness-Hutter 2010 — Switching
  CTW.**  A switching prior in the meta-model class of "which depth-
  ``D`` tree source generated the data, allowing for a small number of
  changepoints" gives a strict refinement of CTW that *also* tracks
  non-stationary sources.  The recursion is the same with two
  internal-node states (split / leaf), priors mixed by ``α/(1-α)``
  per-symbol switch probability.  Ships at construction time.

* **Willems-Shtarkov-Tjalkens 1993 — Context-Tree Maximisation
  (CTM).**  Same recursion with ``max`` replacing ``(1/2) +
  (1/2)·prod`` — extracts the **MAP tree** under the same prior. Ships
  as ``map_tree()``.

* **Veness-Ng-Hutter-Uther-Silver 2011 — A Monte-Carlo AIXI
  approximation.**  MC-AIXI-CTW uses exactly this predictor as the
  generative model inside its UCT planner; the predictor's
  conditional log-loss is the agent's surprise term.  The
  ``Predictor`` here is the *universal-predictor half* of that
  architecture — the planner half is ``Composer`` / ``Active
  Inference`` in this stack.

* **Vovk 1990; Cesa-Bianchi-Lugosi 2006 — universal codes ⇔ on-line
  predictors.**  Any universal source code is a sequential predictor
  whose log-loss equals its code length; CTW is the *exact-mixture*
  predictor for its model class.  This lets ``Predictor`` issue an
  **e-process** ``e_T = 2^T · P_w(x_1^T)`` against ``H_0:`` "source
  is uniform i.i.d." (Vovk-Wang 2021 *E-values*) — which is anytime-
  valid by martingale arguments and ships as
  ``e_process_vs_uniform()``.

Algorithms shipped
------------------

* **CTW binary.**  Exact O(D) per-symbol mixture over the
  exponential class of binary depth-``D`` tree models, log-sum-exp
  stable, anytime.

* **CTW general alphabet.**  KT-Dirichlet ``Dir(1/2, …, 1/2)`` at every
  context, ``A``-way splitting at internal nodes.  O(A·D) per symbol.

* **Switching CTW.**  Volf-Willems switching prior with per-symbol
  switch rate ``α``; constructs the Bayesian mixture over piecewise-
  stationary tree sources.

* **Context-Tree Maximisation (CTM).**  Returns the MAP tree given
  the same prior — an interpretable summary of "what variable-order
  Markov model best explains the data".

* **Universal code length & redundancy bound.**  ``-log_2 P_w(x_1^n)``
  plus a closed-form upper bound on the regret against the best
  tree source.

* **Entropy-rate estimator.**  ``Ĥ_n = -log_2 P_w(x_1^n) / n`` — a
  consistent estimator of the entropy rate of any stationary ergodic
  binary source on alphabet ``A`` (Cover-Thomas 1991 §13).

* **Anytime-valid e-process.**  ``e_T = A^T · P_w(x_1^T)`` is an e-
  process for ``H_0: x_t`` iid uniform on alphabet of size ``A``; a
  coordination engine can stop at any data-dependent time and reject
  ``H_0`` at level ``α`` whenever ``e_T ≥ 1/α``.

* **Most-likely continuation.**  ``argmax_{x_{n+1:n+k}} P_w(x_{n+1:n+k}
  | x_1^n)`` via greedy / beam search on the predictive distribution.

* **Receipts.**  Every registration, observation, prediction, and
  selection appended to a hash chain matching the rest of the stack.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from agi.events import Event, EventBus


# ---------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------

PREDICTOR_STARTED = "predictor.started"
PREDICTOR_OBSERVED = "predictor.observed"
PREDICTOR_PREDICTED = "predictor.predicted"
PREDICTOR_SELECTED = "predictor.selected"
PREDICTOR_REPORTED = "predictor.reported"
PREDICTOR_CLEARED = "predictor.cleared"

PREDICTOR_KNOWN_EVENTS = frozenset(
    {
        PREDICTOR_STARTED,
        PREDICTOR_OBSERVED,
        PREDICTOR_PREDICTED,
        PREDICTOR_SELECTED,
        PREDICTOR_REPORTED,
        PREDICTOR_CLEARED,
    }
)

# Selectors a coordination engine can ask for
SELECT_MAP = "map"
SELECT_BAYES_MEAN = "bayes_mean"
SELECT_MIN_LOG_LOSS = "min_log_loss"
SELECT_SAMPLE = "sample"

PREDICTOR_KNOWN_SELECTORS = frozenset(
    {SELECT_MAP, SELECT_BAYES_MEAN, SELECT_MIN_LOG_LOSS, SELECT_SAMPLE}
)


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class PredictorError(Exception):
    """Base error for the Predictor primitive."""


class InvalidConfig(PredictorError):
    """Configuration values out of range."""


class InvalidSymbol(PredictorError):
    """Symbol outside [0, alphabet_size)."""


class InvalidObservation(PredictorError):
    """Observation sequence rejected (non-integer, wrong shape, …)."""


class InsufficientData(PredictorError):
    """Operation requires at least one observation."""


class UnknownSelector(PredictorError):
    """Selector name not in KNOWN_SELECTORS."""


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

_GENESIS = "0" * 64
_LN2 = math.log(2.0)
_NEG_INF = float("-inf")
_EPS = 1e-300


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _logsumexp2(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _logsumexp(xs: Sequence[float]) -> float:
    """Numerically stable log(sum_i exp(x_i))."""
    m = max(xs)
    if m == _NEG_INF:
        return _NEG_INF
    s = 0.0
    for x in xs:
        s += math.exp(x - m)
    return m + math.log(s)


def _sha256_step(prev: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update(prev.encode("ascii"))
    h.update(b"\x1f")
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _payload_repr(obj: Any) -> str:
    """Canonical, deterministic string repr for hashing."""
    if isinstance(obj, dict):
        keys = sorted(obj.keys())
        parts = [f"{k}={_payload_repr(obj[k])}" for k in keys]
        return "{" + ",".join(parts) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_payload_repr(x) for x in obj) + "]"
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return f"{obj:.17g}"
    return repr(obj)


# ---------------------------------------------------------------------
# CTW node
# ---------------------------------------------------------------------


@dataclass
class _CTWNode:
    """One node of the (lazily expanded) depth-D context tree.

    Stores the symbol counts at this context, the running KT log
    probability, and the running weighted log probability.  Internal
    nodes lazily spawn children only when a context first appears.
    """

    counts: list[int]
    log_pkt: float = 0.0
    log_pw: float = 0.0
    is_leaf: bool = False
    # Switching CTW: log-prob of "have already switched once at this node".
    # Only populated when switching mode is on.
    log_pw_switched: float | None = None


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """The predictive distribution over the next symbol.

    ``probs[a]`` is the CTW posterior predictive ``P(x_{n+1} = a |
    x_1^n)``.  ``log_probs[a]`` is the natural log of the same.

    The prediction is *exact* under CTW: averaging over an
    exponentially large class of variable-order Markov models.
    """

    probs: tuple[float, ...]
    log_probs: tuple[float, ...]
    n_observations: int
    alphabet_size: int

    def argmax(self) -> int:
        best_i = 0
        best_p = self.probs[0]
        for i in range(1, len(self.probs)):
            if self.probs[i] > best_p:
                best_p = self.probs[i]
                best_i = i
        return best_i

    def entropy_bits(self) -> float:
        """Predictive entropy of next symbol in bits."""
        h = 0.0
        for p in self.probs:
            if p > 0.0:
                h -= p * math.log(p) / _LN2
        return h


@dataclass(frozen=True)
class RedundancyBound:
    """Upper bound on CTW redundancy vs the best tree source.

    For every tree source ``S`` of depth ≤ D with ``|S|`` leaves,

      ``-log_2 P_CTW(x_1^n) ≤ -log_2 P_S(x_1^n)
                            + parameter_redundancy(|S|, A, n)
                            + model_redundancy(|S|, D)``.

    ``parameter_redundancy`` is the KT bound ``(|S|·(A-1)/2)·log_2 n
    + O(|S|)``; ``model_redundancy`` is ``2|S| - 1``.
    """

    n_observations: int
    alphabet_size: int
    depth: int
    leaves: int
    parameter_redundancy_bits: float
    model_redundancy_bits: float
    total_redundancy_bits: float
    per_symbol_redundancy_bits: float


@dataclass(frozen=True)
class EProcess:
    """Anytime-valid e-process for H_0: x_t i.i.d. uniform on alphabet.

    ``e = A^n · P_w(x_1^n)`` is a non-negative martingale under H_0
    with expectation 1; rejecting H_0 whenever ``e ≥ 1/α`` controls
    Type-I error at level ``α`` under *any* data-dependent stopping
    rule (Ville's inequality; Vovk-Wang 2021).
    """

    e_value: float
    log_e_value: float
    n_observations: int
    alphabet_size: int
    p_value_upper_bound: float


@dataclass(frozen=True)
class EntropyRate:
    """Plug-in entropy-rate estimate from the universal predictor.

    For any stationary ergodic source, ``-log_2 P_w(x_1^n)/n`` converges
    to the source entropy rate ``H`` (Cover-Thomas 1991, §13).  This
    object also reports the *next-symbol* predictive entropy, which is
    typically tighter on short prefixes.
    """

    average_log_loss_bits_per_symbol: float
    predictive_entropy_bits: float
    n_observations: int


@dataclass(frozen=True)
class TreeNode:
    """One node of the CTM (Context-Tree Maximisation) MAP tree.

    A leaf has ``children == ()`` and exposes its KT-MAP symbol
    probabilities.  An internal node has ``alphabet_size`` children;
    its KT estimate is *not* used in the MAP source.
    """

    context: str
    counts: tuple[int, ...]
    is_leaf: bool
    children: tuple["TreeNode", ...] = ()

    def map_probs(self) -> tuple[float, ...]:
        """Posterior-mean symbol probabilities under the KT Dirichlet."""
        n = sum(self.counts)
        a = len(self.counts)
        return tuple((c + 0.5) / (n + a * 0.5) for c in self.counts)


@dataclass(frozen=True)
class MAPTree:
    """The Context-Tree Maximisation MAP tree under the CTW prior.

    Willems-Shtarkov-Tjalkens 1993; same recursion as CTW with
    ``max`` instead of the ``1/2 + 1/2`` mixture.  Returns the
    interpretable variable-order Markov model with highest posterior
    weight.
    """

    root: TreeNode
    n_leaves: int
    alphabet_size: int
    depth: int
    log_map_prob: float

    def leaves(self) -> list[TreeNode]:
        out: list[TreeNode] = []

        def go(node: TreeNode) -> None:
            if node.is_leaf:
                out.append(node)
            else:
                for c in node.children:
                    go(c)

        go(self.root)
        return out


@dataclass(frozen=True)
class PredictorReport:
    """Comprehensive report on the Predictor's state.

    Bundles everything a coordination engine needs to make a logged,
    auditable decision off the predictor: the current predictive
    distribution, code length, entropy rate, redundancy bound,
    e-process, MAP tree leaf count, and the fingerprint of the trace
    so far.
    """

    n_observations: int
    alphabet_size: int
    depth: int
    code_length_bits: float
    average_log_loss_bits_per_symbol: float
    prediction: Prediction
    redundancy_bound: RedundancyBound
    entropy_rate: EntropyRate
    e_process: EProcess
    map_tree_leaves: int
    fingerprint: str
    config: dict[str, Any]


# ---------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------


def _select_map(prediction: Prediction) -> int:
    return prediction.argmax()


def _select_bayes_mean(prediction: Prediction) -> int:
    """Argmin of expected 0-1 loss = argmax of probs (same as MAP for 0-1)."""
    return prediction.argmax()


def _select_min_log_loss(prediction: Prediction) -> int:
    """Argmin of -log p — same as argmax of probs."""
    return prediction.argmax()


def _select_sample(prediction: Prediction, rng_state: list[int]) -> int:
    """Sample from the predictive distribution (PCG-style LCG)."""
    # Linear congruential RNG so the same seed gives the same trace.
    rng_state[0] = (rng_state[0] * 6364136223846793005 + 1442695040888963407) & (
        (1 << 64) - 1
    )
    u = ((rng_state[0] >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    acc = 0.0
    for i, p in enumerate(prediction.probs):
        acc += p
        if u < acc:
            return i
    return len(prediction.probs) - 1


# ---------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------


class Predictor:
    """Universal sequence predictor via Context Tree Weighting.

    A coordination engine treats this primitive as the workhorse
    *non-parametric* probability estimator: feed any symbol stream,
    get back a calibrated predictive distribution over the next
    symbol whose code length is universally no worse than the best
    variable-order Markov model of depth ≤ ``D``, up to ``O(log n)``
    redundancy.

    The predictor is **thread-safe** (one internal mutex), **anytime**
    (every method is well-defined after any number of observations,
    including zero), and produces **tamper-evident** receipts.
    """

    def __init__(
        self,
        *,
        alphabet_size: int,
        depth: int,
        bus: EventBus | None = None,
        switching_rate: float | None = None,
        seed: int = 0,
        primitive_id: str | None = None,
    ) -> None:
        if alphabet_size < 2 or alphabet_size > 256:
            raise InvalidConfig(
                f"alphabet_size must be in [2, 256], got {alphabet_size}"
            )
        if depth < 0 or depth > 64:
            raise InvalidConfig(f"depth must be in [0, 64], got {depth}")
        if switching_rate is not None and not (0.0 < switching_rate < 1.0):
            raise InvalidConfig(
                f"switching_rate must be in (0, 1) or None, got {switching_rate}"
            )
        self.alphabet_size = alphabet_size
        self.depth = depth
        self._switching_rate = switching_rate
        self._bus = bus
        self._lock = threading.RLock()
        self.id = primitive_id or uuid.uuid4().hex[:12]
        # state
        self._nodes: dict[str, _CTWNode] = {}
        self._history: list[int] = []
        self._fingerprint = _GENESIS
        self._n_events = 0
        self._rng = [int(seed) & ((1 << 64) - 1) or 1]
        self._seed = int(seed)
        # Empty-context node (depth-0) exists from the start so we always
        # have something to mix into the root.
        self._get_or_make_node("")
        self._emit(
            PREDICTOR_STARTED,
            {
                "alphabet_size": alphabet_size,
                "depth": depth,
                "switching_rate": switching_rate,
                "seed": seed,
            },
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        alphabet_size: int = 2,
        depth: int = 8,
        bus: EventBus | None = None,
        switching_rate: float | None = None,
        seed: int = 0,
    ) -> "Predictor":
        return cls(
            alphabet_size=alphabet_size,
            depth=depth,
            bus=bus,
            switching_rate=switching_rate,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return len(self._history)

    @property
    def switching_rate(self) -> float | None:
        return self._switching_rate

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Public surface — observe / predict
    # ------------------------------------------------------------------

    def observe(self, symbol: int) -> Prediction:
        """Process one new symbol; return predictive distribution AFTER.

        Returns the CTW predictive distribution over the *next* symbol
        given the just-extended history.  The returned ``Prediction``
        is consistent with ``predict()`` called immediately afterward.
        """
        with self._lock:
            self._validate_symbol(symbol)
            self._update_with_symbol(int(symbol))
            pred = self._predict_locked()
            payload = {
                "symbol": int(symbol),
                "n": self.n_observations,
                "log_pw_root": self._log_pw_root_locked(),
            }
            self._emit(PREDICTOR_OBSERVED, payload)
            return pred

    def observe_many(self, symbols: Iterable[int]) -> Prediction:
        """Process a batch of symbols. Returns prediction after the last."""
        with self._lock:
            seq = list(symbols)
            for s in seq:
                self._validate_symbol(s)
            for s in seq:
                self._update_with_symbol(int(s))
            pred = self._predict_locked()
            self._emit(
                PREDICTOR_OBSERVED,
                {
                    "batch_size": len(seq),
                    "n": self.n_observations,
                    "log_pw_root": self._log_pw_root_locked(),
                },
            )
            return pred

    def predict(self) -> Prediction:
        """CTW predictive distribution over the next symbol."""
        with self._lock:
            pred = self._predict_locked()
            self._emit(
                PREDICTOR_PREDICTED,
                {"probs": list(pred.probs), "n": self.n_observations},
            )
            return pred

    def predict_sequence(self, k: int) -> list[Prediction]:
        """Multi-step rollout.

        Returns ``k`` ``Prediction`` objects corresponding to a
        *greedy* most-likely rollout: at each step the predictor
        commits to the argmax of the current predictive, advances its
        internal state hypothetically, and emits the next predictive.

        State after the call is restored — this is non-destructive.
        """
        if k < 0:
            raise InvalidObservation(f"k must be ≥ 0, got {k}")
        with self._lock:
            saved_history = list(self._history)
            saved_nodes = self._snapshot_nodes_locked()
            try:
                out: list[Prediction] = []
                for _ in range(k):
                    pred = self._predict_locked()
                    out.append(pred)
                    self._update_with_symbol(pred.argmax())
            finally:
                self._history = saved_history
                self._restore_nodes_locked(saved_nodes)
            return out

    def most_likely_continuation(self, k: int) -> list[int]:
        """Greedy MAP continuation of length k."""
        rollouts = self.predict_sequence(k)
        return [p.argmax() for p in rollouts]

    def log_loss(self, future: Sequence[int]) -> float:
        """Cumulative log loss (natural log) of predicting ``future``.

        Returns ``-log P_w(future | history)`` in nats.  State unchanged.
        """
        if not future:
            return 0.0
        with self._lock:
            for s in future:
                self._validate_symbol(s)
            saved_history = list(self._history)
            saved_nodes = self._snapshot_nodes_locked()
            try:
                loss = 0.0
                for s in future:
                    pred = self._predict_locked()
                    loss -= pred.log_probs[int(s)]
                    self._update_with_symbol(int(s))
                return loss
            finally:
                self._history = saved_history
                self._restore_nodes_locked(saved_nodes)

    # ------------------------------------------------------------------
    # Public surface — selection
    # ------------------------------------------------------------------

    def select(self, rule: str = SELECT_MAP) -> int:
        """Bayes-decision selector.

        Available rules:
          * ``SELECT_MAP``         — argmax of predictive (also 0-1 loss).
          * ``SELECT_BAYES_MEAN``  — argmax of predictive (same as MAP).
          * ``SELECT_MIN_LOG_LOSS``— argmax of predictive (same as MAP).
          * ``SELECT_SAMPLE``      — sample from predictive.
        """
        with self._lock:
            pred = self._predict_locked()
            if rule == SELECT_MAP:
                pick = _select_map(pred)
            elif rule == SELECT_BAYES_MEAN:
                pick = _select_bayes_mean(pred)
            elif rule == SELECT_MIN_LOG_LOSS:
                pick = _select_min_log_loss(pred)
            elif rule == SELECT_SAMPLE:
                pick = _select_sample(pred, self._rng)
            else:
                raise UnknownSelector(f"unknown selector {rule!r}")
            self._emit(
                PREDICTOR_SELECTED,
                {"rule": rule, "pick": int(pick), "probs": list(pred.probs)},
            )
            return int(pick)

    # ------------------------------------------------------------------
    # Public surface — diagnostics
    # ------------------------------------------------------------------

    def code_length_bits(self) -> float:
        """Universal code length of the observed prefix in bits.

        ``-log_2 P_w(x_1^n)``.  Equals the cumulative log-loss of the
        CTW predictor in bits.
        """
        with self._lock:
            return -self._log_pw_root_locked() / _LN2

    def code_length_nats(self) -> float:
        """Universal code length in nats."""
        with self._lock:
            return -self._log_pw_root_locked()

    def redundancy_bound(self, *, leaves: int | None = None) -> RedundancyBound:
        """Upper bound on CTW redundancy vs the best tree source.

        If ``leaves`` is given, the bound is reported for a comparator
        with that many leaves; otherwise the worst-case over all
        comparators of depth ≤ D is reported (``leaves = A^D``).
        """
        with self._lock:
            n = self.n_observations
            A = self.alphabet_size
            D = self.depth
            ell = leaves if leaves is not None else max(1, A**D if D > 0 else 1)
            # Parameter redundancy (Krichevsky-Trofimov per leaf).
            if n > 0:
                per_leaf = 0.5 * (A - 1) * math.log2(max(n / ell, 1.0)) + (A - 1)
            else:
                per_leaf = float(A - 1)
            param_red = ell * per_leaf
            # Model redundancy: 2|S| - 1 bits for the CTW prior over trees.
            model_red = max(0.0, 2.0 * ell - 1.0)
            total = param_red + model_red
            per_symbol = total / n if n > 0 else float("inf")
            return RedundancyBound(
                n_observations=n,
                alphabet_size=A,
                depth=D,
                leaves=ell,
                parameter_redundancy_bits=param_red,
                model_redundancy_bits=model_red,
                total_redundancy_bits=total,
                per_symbol_redundancy_bits=per_symbol,
            )

    def entropy_rate_estimate(self) -> EntropyRate:
        """Plug-in entropy rate from the universal code."""
        with self._lock:
            n = self.n_observations
            if n == 0:
                return EntropyRate(
                    average_log_loss_bits_per_symbol=math.log2(self.alphabet_size),
                    predictive_entropy_bits=math.log2(self.alphabet_size),
                    n_observations=0,
                )
            avg = -self._log_pw_root_locked() / _LN2 / n
            pred = self._predict_locked()
            return EntropyRate(
                average_log_loss_bits_per_symbol=avg,
                predictive_entropy_bits=pred.entropy_bits(),
                n_observations=n,
            )

    def e_process_vs_uniform(self) -> EProcess:
        """Anytime-valid e-process for H_0: x_t iid Uniform({0,…,A-1}).

        ``e_T = A^T · P_w(x_1^T)``.  Under H_0, E[e_T] = 1 and ``e``
        is a non-negative martingale; reject H_0 at level α whenever
        ``e ≥ 1/α``, at any (data-dependent) stopping time.
        """
        with self._lock:
            n = self.n_observations
            log_e = n * math.log(self.alphabet_size) + self._log_pw_root_locked()
            e = math.exp(min(log_e, 700.0))
            p_bound = min(1.0, 1.0 / e) if e > 0 else 1.0
            return EProcess(
                e_value=e,
                log_e_value=log_e,
                n_observations=n,
                alphabet_size=self.alphabet_size,
                p_value_upper_bound=p_bound,
            )

    def map_tree(self) -> MAPTree:
        """Compute the CTM MAP tree (Willems-Shtarkov-Tjalkens 1993).

        Same recursion as CTW with ``max`` replacing the mixture; the
        tree returned is the one with highest posterior weight under
        the same prior.  Provides an interpretable summary of which
        variable-order Markov model best explains the observed data.
        """
        with self._lock:
            return self._compute_map_tree_locked()

    def report(self) -> PredictorReport:
        """Full snapshot — predictive, code, entropy rate, redundancy, e-process."""
        with self._lock:
            pred = self._predict_locked()
            entropy = self.entropy_rate_estimate()
            red = self.redundancy_bound()
            e = self.e_process_vs_uniform()
            map_tree = self._compute_map_tree_locked()
            rep = PredictorReport(
                n_observations=self.n_observations,
                alphabet_size=self.alphabet_size,
                depth=self.depth,
                code_length_bits=-self._log_pw_root_locked() / _LN2,
                average_log_loss_bits_per_symbol=entropy.average_log_loss_bits_per_symbol,
                prediction=pred,
                redundancy_bound=red,
                entropy_rate=entropy,
                e_process=e,
                map_tree_leaves=map_tree.n_leaves,
                fingerprint=self._fingerprint,
                config={
                    "alphabet_size": self.alphabet_size,
                    "depth": self.depth,
                    "switching_rate": self._switching_rate,
                    "seed": self._seed,
                },
            )
            self._emit(
                PREDICTOR_REPORTED,
                {
                    "n": rep.n_observations,
                    "code_length_bits": rep.code_length_bits,
                    "entropy_rate_bits": rep.average_log_loss_bits_per_symbol,
                    "fingerprint": rep.fingerprint,
                },
            )
            return rep

    def clear(self) -> None:
        """Drop all state, reset fingerprint chain."""
        with self._lock:
            self._nodes.clear()
            self._history.clear()
            self._fingerprint = _GENESIS
            self._n_events = 0
            self._rng = [int(self._seed) & ((1 << 64) - 1) or 1]
            self._get_or_make_node("")
            self._emit(PREDICTOR_CLEARED, {"genesis": _GENESIS})

    # ------------------------------------------------------------------
    # Internal — CTW update
    # ------------------------------------------------------------------

    def _validate_symbol(self, s: Any) -> None:
        if not isinstance(s, int) or isinstance(s, bool):
            if isinstance(s, bool):
                return  # bool is int subclass; allow
            raise InvalidSymbol(f"symbol must be int, got {type(s).__name__}")
        if s < 0 or s >= self.alphabet_size:
            raise InvalidSymbol(
                f"symbol {s} outside alphabet [0, {self.alphabet_size})"
            )

    def _context_string(self) -> str:
        """The relevant context for the *next* symbol — last D history."""
        if self.depth == 0 or not self._history:
            return ""
        D = self.depth
        if len(self._history) <= D:
            tail = self._history
        else:
            tail = self._history[-D:]
        # Encode each symbol as a fixed-width hex char (alphabet ≤ 256 means
        # 2 hex chars per symbol is sufficient; but for binary we want
        # human-readable context strings, so use a single char in radix 16
        # for alphabet ≤ 16 and 2-char hex otherwise).
        if self.alphabet_size <= 16:
            return "".join(f"{x:x}" for x in tail)
        return "".join(f"{x:02x}" for x in tail)

    def _encode_symbol(self, x: int) -> str:
        if self.alphabet_size <= 16:
            return f"{x:x}"
        return f"{x:02x}"

    def _suffixes(self, context: str) -> list[str]:
        """All suffixes of ``context``, longest first.

        For binary (and alphabet ≤ 16) every suffix length corresponds
        to a single context-char-per-symbol; for larger alphabets we
        encoded with 2 chars/symbol and so suffix lengths must be
        multiples of 2.
        """
        step = 1 if self.alphabet_size <= 16 else 2
        out = []
        # longest first: depths D, D-1, ..., 0
        max_len = len(context)
        # truncate to depth*step
        cap = self.depth * step
        if max_len > cap:
            context = context[-cap:]
            max_len = len(context)
        for L in range(max_len, -1, -step):
            out.append(context[-L:] if L > 0 else "")
        return out

    def _get_or_make_node(self, context: str) -> _CTWNode:
        node = self._nodes.get(context)
        if node is None:
            depth_here = self._depth_of(context)
            node = _CTWNode(
                counts=[0] * self.alphabet_size,
                log_pkt=0.0,
                log_pw=0.0,
                is_leaf=(depth_here == self.depth),
                log_pw_switched=(0.0 if self._switching_rate is not None else None),
            )
            self._nodes[context] = node
        return node

    def _depth_of(self, context: str) -> int:
        step = 1 if self.alphabet_size <= 16 else 2
        return len(context) // step

    def _update_with_symbol(self, x: int) -> None:
        """Apply one observation to the CTW tree."""
        context = self._context_string()
        suffixes = self._suffixes(context)  # depths D..0 (or up to history)
        # Process from longest suffix (deepest) upward to the root, so that
        # internal-node updates have already-updated children.
        # First update KT counts/log_pkt at every involved node.
        for s in suffixes:
            node = self._get_or_make_node(s)
            n_total = sum(node.counts)
            count_x = node.counts[x]
            log_p_kt_inc = math.log(
                (count_x + 0.5) / (n_total + 0.5 * self.alphabet_size)
            )
            node.log_pkt += log_p_kt_inc
            node.counts[x] = count_x + 1
        # Now recompute log_pw at every involved node bottom-up.
        for s in suffixes:
            node = self._nodes[s]
            if node.is_leaf or self._depth_of(s) >= self.depth:
                node.log_pw = node.log_pkt
                if self._switching_rate is not None:
                    node.log_pw_switched = node.log_pkt
                continue
            # Internal: combine KT with product over children.
            # Children are the per-symbol extensions on the *left* — i.e., the
            # context for any extension is (child_symbol_encoded + s).
            log_prod_children = 0.0
            for a in range(self.alphabet_size):
                child_ctx = self._encode_symbol(a) + s
                child = self._nodes.get(child_ctx)
                if child is None:
                    # Unseen child: its KT is empty so log_pw = 0.
                    continue
                log_prod_children += child.log_pw
            if self._switching_rate is None:
                # Standard CTW: log_pw = log( 1/2 e^log_pkt + 1/2 e^log_prod )
                node.log_pw = -_LN2 + _logsumexp2(node.log_pkt, log_prod_children)
            else:
                # Switching CTW (Volf-Willems): per-symbol mixing with switch rate α.
                # log_pw_split  = (1-α) · prod_children + α · pkt    (interpreted in prob)
                # log_pw_kt     = (1-α) · pkt + α · prod_children
                # log_pw        = 1/2 (split + kt)  (uniform prior on first state)
                a = self._switching_rate
                log_a = math.log(a)
                log_1ma = math.log(1.0 - a)
                log_split = _logsumexp2(
                    log_1ma + log_prod_children, log_a + node.log_pkt
                )
                log_kt = _logsumexp2(log_1ma + node.log_pkt, log_a + log_prod_children)
                node.log_pw = -_LN2 + _logsumexp2(log_split, log_kt)
                node.log_pw_switched = node.log_pw
        # Commit history *after* the update so context computed above
        # refers to the pre-observation context.
        self._history.append(x)

    def _log_pw_root_locked(self) -> float:
        return self._nodes[""].log_pw

    # ------------------------------------------------------------------
    # Internal — prediction
    # ------------------------------------------------------------------

    def _predict_locked(self) -> Prediction:
        """Predictive distribution over next symbol given current state.

        We compute log P_w(ε) for the current sequence and for each
        candidate next symbol (hypothetical extension), then take the
        normalised ratio.  This is implemented by hypothetically
        applying the update for each candidate without committing the
        history.
        """
        baseline = self._log_pw_root_locked()
        log_probs = [0.0] * self.alphabet_size
        for a in range(self.alphabet_size):
            self._update_with_symbol(a)
            new_root = self._log_pw_root_locked()
            log_probs[a] = new_root - baseline
            self._rollback_one_update(a)
        # Normalise: in theory sum_a exp(log_probs[a]) = 1; in practice
        # floating drift means we renormalise.
        lse = _logsumexp(log_probs)
        log_probs = [lp - lse for lp in log_probs]
        probs = tuple(math.exp(lp) for lp in log_probs)
        return Prediction(
            probs=probs,
            log_probs=tuple(log_probs),
            n_observations=self.n_observations,
            alphabet_size=self.alphabet_size,
        )

    def _rollback_one_update(self, x: int) -> None:
        """Undo the last call to _update_with_symbol — used by _predict_locked.

        We replay the inverse of the update at every touched node.
        Because we only do this for the *hypothetical-symbol* probe,
        the history list also has to be popped.
        """
        # history was appended; pop it.
        self._history.pop()
        # Now the context we used for the just-undone update is the
        # current context again.
        context = self._context_string()
        suffixes = self._suffixes(context)
        # Reverse KT update at each node, in any order; then recompute log_pw.
        for s in suffixes:
            node = self._nodes[s]
            node.counts[x] -= 1
            n_total = sum(node.counts)
            count_x = node.counts[x]
            log_p_kt_dec = math.log(
                (count_x + 0.5) / (n_total + 0.5 * self.alphabet_size)
            )
            node.log_pkt -= log_p_kt_dec
        # Recompute log_pw bottom-up.
        for s in suffixes:
            node = self._nodes[s]
            if node.is_leaf or self._depth_of(s) >= self.depth:
                node.log_pw = node.log_pkt
                if self._switching_rate is not None:
                    node.log_pw_switched = node.log_pkt
                continue
            log_prod_children = 0.0
            for a in range(self.alphabet_size):
                child_ctx = self._encode_symbol(a) + s
                child = self._nodes.get(child_ctx)
                if child is None:
                    continue
                log_prod_children += child.log_pw
            if self._switching_rate is None:
                node.log_pw = -_LN2 + _logsumexp2(node.log_pkt, log_prod_children)
            else:
                a = self._switching_rate
                log_a = math.log(a)
                log_1ma = math.log(1.0 - a)
                log_split = _logsumexp2(
                    log_1ma + log_prod_children, log_a + node.log_pkt
                )
                log_kt = _logsumexp2(log_1ma + node.log_pkt, log_a + log_prod_children)
                node.log_pw = -_LN2 + _logsumexp2(log_split, log_kt)
                node.log_pw_switched = node.log_pw

    # ------------------------------------------------------------------
    # Internal — snapshot/restore for non-destructive rollouts
    # ------------------------------------------------------------------

    def _snapshot_nodes_locked(self) -> dict[str, tuple[list[int], float, float, bool]]:
        snap: dict[str, tuple[list[int], float, float, bool]] = {}
        for k, n in self._nodes.items():
            snap[k] = (list(n.counts), n.log_pkt, n.log_pw, n.is_leaf)
        return snap

    def _restore_nodes_locked(
        self, snap: dict[str, tuple[list[int], float, float, bool]]
    ) -> None:
        # delete any nodes added since the snapshot
        for k in list(self._nodes.keys()):
            if k not in snap:
                del self._nodes[k]
        # restore everything else
        for k, (counts, log_pkt, log_pw, is_leaf) in snap.items():
            node = self._nodes.get(k)
            if node is None:
                node = _CTWNode(
                    counts=list(counts),
                    log_pkt=log_pkt,
                    log_pw=log_pw,
                    is_leaf=is_leaf,
                )
                self._nodes[k] = node
            else:
                node.counts = list(counts)
                node.log_pkt = log_pkt
                node.log_pw = log_pw
                node.is_leaf = is_leaf

    # ------------------------------------------------------------------
    # Internal — CTM (MAP tree)
    # ------------------------------------------------------------------

    def _compute_map_tree_locked(self) -> MAPTree:
        """Recursive CTM: at every internal node pick max(KT, prod children).

        For nodes that have not been visited (zero counts), the KT
        evidence is 1 and the children-product is 1; we treat them as
        leaves with vacuous counts.
        """
        # Build the MAP recursively starting from root.
        log_total = [0.0]

        def go(context: str, depth: int) -> TreeNode:
            node = self._nodes.get(context)
            if node is None:
                # Unseen subtree — represent as an empty leaf.
                empty = _CTWNode(counts=[0] * self.alphabet_size, log_pkt=0.0)
                return TreeNode(
                    context=context,
                    counts=tuple(empty.counts),
                    is_leaf=True,
                    children=(),
                )
            if node.is_leaf or depth >= self.depth:
                log_total[0] += node.log_pkt
                return TreeNode(
                    context=context,
                    counts=tuple(node.counts),
                    is_leaf=True,
                    children=(),
                )
            # Compute children-product log-prob if we split.
            log_prod_children = 0.0
            children_have_data = False
            for a in range(self.alphabet_size):
                child_ctx = self._encode_symbol(a) + context
                child_node = self._nodes.get(child_ctx)
                if child_node is not None:
                    log_prod_children += child_node.log_pw
                    children_have_data = True
            # MAP rule: choose split iff prod_children > KT (prior weight equal).
            if children_have_data and log_prod_children > node.log_pkt:
                log_total[0] += log_prod_children
                kids = tuple(
                    go(self._encode_symbol(a) + context, depth + 1)
                    for a in range(self.alphabet_size)
                )
                return TreeNode(
                    context=context,
                    counts=tuple(node.counts),
                    is_leaf=False,
                    children=kids,
                )
            log_total[0] += node.log_pkt
            return TreeNode(
                context=context,
                counts=tuple(node.counts),
                is_leaf=True,
                children=(),
            )

        root = go("", 0)
        # Count leaves.
        leaves = 0

        def count_leaves(t: TreeNode) -> None:
            nonlocal leaves
            if t.is_leaf:
                leaves += 1
            else:
                for c in t.children:
                    count_leaves(c)

        count_leaves(root)
        return MAPTree(
            root=root,
            n_leaves=leaves,
            alphabet_size=self.alphabet_size,
            depth=self.depth,
            log_map_prob=log_total[0],
        )

    # ------------------------------------------------------------------
    # Internal — events / fingerprint
    # ------------------------------------------------------------------

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        self._n_events += 1
        payload = {"kind": kind, "data": data, "seq": self._n_events}
        self._fingerprint = _sha256_step(self._fingerprint, _payload_repr(payload))
        if self._bus is None:
            return
        ev = Event(
            kind=kind,
            session_id=None,
            ts=time.time(),
            data={**data, "primitive_id": self.id, "fingerprint": self._fingerprint},
        )
        try:
            self._bus.publish(ev)
        except Exception:
            # Never let a flaky subscriber break the predictor.
            pass


# ---------------------------------------------------------------------
# Convenience: quick streaming compressor
# ---------------------------------------------------------------------


def compress_binary_sequence(
    sequence: Sequence[int], *, depth: int = 8
) -> tuple[float, Predictor]:
    """Return (code_length_bits, predictor) for a binary stream.

    Convenience constructor for the common case of "I have a binary
    stream; how many bits would CTW spend on it?".  Compares directly
    against the naive ``n * log2(2) = n`` bound: the difference is the
    structure CTW exploited.
    """
    pred = Predictor.create(alphabet_size=2, depth=depth)
    pred.observe_many(sequence)
    return pred.code_length_bits(), pred


def kl_divergence_bits(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in bits.  Helper for diagnostic code."""
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        s += pi * (math.log(pi) - math.log(qi))
    return s / _LN2


__all__ = [
    "PREDICTOR_STARTED",
    "PREDICTOR_OBSERVED",
    "PREDICTOR_PREDICTED",
    "PREDICTOR_SELECTED",
    "PREDICTOR_REPORTED",
    "PREDICTOR_CLEARED",
    "PREDICTOR_KNOWN_EVENTS",
    "PREDICTOR_KNOWN_SELECTORS",
    "SELECT_MAP",
    "SELECT_BAYES_MEAN",
    "SELECT_MIN_LOG_LOSS",
    "SELECT_SAMPLE",
    "Predictor",
    "Prediction",
    "RedundancyBound",
    "EProcess",
    "EntropyRate",
    "MAPTree",
    "TreeNode",
    "PredictorReport",
    "PredictorError",
    "InvalidConfig",
    "InvalidSymbol",
    "InvalidObservation",
    "InsufficientData",
    "UnknownSelector",
    "compress_binary_sequence",
    "kl_divergence_bits",
]
