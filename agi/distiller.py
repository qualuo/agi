r"""Distiller — amortized policy / value approximation as a runtime primitive.

Every other primitive in this runtime *computes* an answer.  ``Searcher``
runs PUCT or A\* on a fresh tree.  ``Solver`` decides a fresh CNF.
``Inducer`` enumerates fresh programs.  The operation that takes the
*outputs* of those primitives — visit-distributions over actions, value
estimates at states, accepted decisions on inputs — and **compiles them
into a cheap, callable model** so the next instance of the same kind of
question is answered in *amortized constant time* is **distillation**.

Distillation is the operational mechanism behind every milestone-grade
self-improving AI of the last decade: AlphaGo Zero distils 800-rollout
PUCT into a single forward pass; MuZero distils a learned-dynamics
search into the same; expert iteration / DAgger distils a slow expert
into a fast student; algorithmic distillation distils a learning
algorithm into the activations of a frozen Transformer.  In every case
the search-or-oracle is the teacher, the parametric model is the
student, and the *expected behaviour* of the teacher under the
student's own distribution is the target.

``Distiller`` is the runtime's *bounded, anytime, certified, stdlib*
version of that operation.  Given a stream of *teacher demonstrations*
— ``(state, action_distribution, value)`` triples from any source
(``Searcher``, a hand-coded oracle, a human, another LLM) — it fits a
parametric *policy / value model* that exposes itself as a pair of
callables a ``Searcher`` can immediately turn around and use as
``policy_prior`` and ``value``.  The runtime closes the loop **inside
its own process**, without touching a GPU, without a deep-learning
framework, and without a tokenizer.

The pitch reduced to a runtime call::

    teacher = Searcher(SearcherConfig(algorithm="puct", max_iterations=512))
    student = Distiller(DistillerConfig(model="linear", n_features=4096))

    for episode in range(100):
        s = root_state()
        while not is_terminal(s):
            rep = teacher.search(s,
                actions=actions, apply=apply, terminal=terminal,
                reward=reward, policy_prior=student.as_policy_prior(),
                value=student.as_value())
            student.observe(
                state=s,
                action_distribution=rep.root_visits_by_action,
                value=rep.best_value,
            )
            s = apply(s, rep.best_action)
        student.fit()                       # batch update from buffered targets

    # student is now usable as a *standalone* fast policy/value:
    policy_for_state = student.policy(s)    # dict action -> probability
    value_for_state  = student.value(s)     # scalar

What this primitive ships
-------------------------

  * **Model families (all stdlib, no NumPy, no Torch):**

    * ``"knn"`` — exact :math:`k`-nearest-neighbour over a *featurisation
      hash* of the state (Cover & Hart 1967); :math:`O(N)` query, good
      when teacher demonstrations are scarce.
    * ``"linear"`` — feature-hashed (Weinberger et al. 2009) linear
      softmax policy + linear value head, trained by **online passive-
      aggressive** updates (Crammer et al. 2006) for policy and
      **recursive least-squares** (Plackett 1950) for value;
      :math:`O(d)` query, anytime learnable.
    * ``"locally_weighted"`` — locally weighted regression (Atkeson,
      Moore & Schaal 1997): for a query state, weight nearby
      demonstrations by a Gaussian kernel and fit a *local* linear
      model.  Bridges k-NN's flexibility with linear regression's
      smoothing.
    * ``"ucb_table"`` — exact tabular memoization with UCB-tied
      tie-breaks; collapses to the teacher's own visit-table on
      revisited states (no generalisation).  The right answer for tiny
      finite domains.
    * ``"ensemble"`` — a *log-linear opinion pool* over any subset of
      the above with weights fit by isotonic regression on a held-out
      slice (Brier-loss minimisation; cf. Gneiting & Raftery 2007).

  * **Loss functions:**

    * Policy: cross-entropy ``-Σ_a π_T(a|s) log π_S(a|s)`` between
      teacher and student distributions.  When :math:`π_T` is one-hot
      this reduces to the SFT loss.
    * Value: half-mean-squared error ``½ (v_T - v_S)²`` with
      optional **Tukey biweight** robustification (Huber 1981).

  * **Calibration:**

    * **Temperature scaling** (Guo et al. 2017) on the policy logits,
      fit by Brier-minimisation on a held-out slice.
    * **Isotonic regression** (Brunk et al. 1972) on the value head,
      fit by pool-adjacent-violators (PAV).
    * Both fit *post-hoc*; the underlying parameters are not retouched.

  * **Eval-gated deployment** (AlphaZero ladder discipline):

    A new fit *only replaces* the deployed model if its average held-
    out cross-entropy *and* value error are both lower than the
    incumbent's by at least ``min_improvement`` (default 1e-4).  The
    rollback story is enforced inside the primitive — there is no
    way to swap in a regressed student.

  * **Reservoir replay buffer** (Vitter 1985): bounded-memory
    uniform-sample over the entire demonstration stream; no need for
    rolling windows, no need to hand-pick batches, no oldest-data
    bias.

  * **Certificate chain:**

    Every ``DistillerReport`` carries a SHA-256 chain over the
    canonical sequence of ``(epoch, mini-batch hash, parameter delta
    hash, eval result)`` events.  Two distillers in two processes
    fed the same demonstrations under the same seed agree on the
    certificate byte-for-byte.

Mathematical roots
------------------

  * **Cover, T. M. & Hart, P. E. (1967) — "Nearest neighbor pattern
    classification."**  *IEEE Trans. Information Theory* 13(1) 21-27.
    The :math:`k`-NN consistency theorem: as :math:`N → ∞`, 1-NN risk
    is at most twice the Bayes risk; supports the ``knn`` family as
    a *consistent* low-data fall-back.

  * **Plackett, R. L. (1950) — "Some theorems in least squares."**
    *Biometrika* 37(1/2) 149-157.  Recursive least squares — the
    closed-form, exact, online update used by the value head.

  * **Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S. & Singer,
    Y. (2006) — "Online passive-aggressive algorithms."**  *Journal
    of Machine Learning Research* 7 551-585.  PA-II: closed-form
    online updates with a margin-based loss; the policy head's
    backbone.

  * **Atkeson, C. G., Moore, A. W. & Schaal, S. (1997) — "Locally
    weighted learning."**  *Artificial Intelligence Review* 11(1-5)
    11-73.  The locally-weighted-regression family.

  * **Weinberger, K. *et al.* (2009) — "Feature hashing for large
    scale multitask learning."**  *Proc. ICML.*  The hashing trick:
    bounded-memory features for arbitrary-cardinality state spaces.

  * **Vitter, J. S. (1985) — "Random sampling with a reservoir."**
    *ACM Trans. Mathematical Software* 11(1) 37-57.  Reservoir
    sampling — the buffer discipline.

  * **Brunk, H. D. *et al.* (1972) — *Statistical Inference under
    Order Restrictions.***  John Wiley.  Isotonic regression by the
    pool-adjacent-violators algorithm — the value-head calibration.

  * **Guo, C. *et al.* (2017) — "On calibration of modern neural
    networks."**  *Proc. ICML.*  Temperature scaling — the policy
    calibration.

  * **Gneiting, T. & Raftery, A. E. (2007) — "Strictly proper scoring
    rules, prediction, and estimation."**  *J. American Statistical
    Association* 102(477) 359-378.  Brier and log-loss as strictly
    proper scoring rules — the basis of held-out eval gating and
    ensemble weighting.

  * **Bishop, C. (2006) — *Pattern Recognition and Machine
    Learning*, ch. 4.**  Softmax + cross-entropy derivations the
    linear policy head implements.

  * **Schmidhuber, J. (1991) — "On learning how to learn learning
    strategies."**  T.R. FKI-198-94, T. Universität München.  The
    early articulation of the *meta-learning* loop the
    Searcher ↔ Distiller pair instantiates inside this runtime.

  * **Anthony, T., Tian, Z. & Barber, D. (2017) — "Thinking fast
    and slow with deep learning and tree search."**  *Proc. NeurIPS*
    30.  Expert Iteration (ExIt): the algorithmic articulation of
    *Searcher-as-teacher / Distiller-as-student / Searcher-with-
    student-as-prior / repeat* that this primitive implements.

  * **Silver, D. *et al.* (2017) — "Mastering the game of Go without
    human knowledge."**  *Nature* 550 354-359.  AlphaGo Zero — the
    PUCT-search ↔ policy/value-net loop the runtime renders inside
    its own process.

  * **Schrittwieser, J. *et al.* (2020) — "Mastering Atari, Go,
    chess and shogi by planning with a learned model."**  *Nature*
    588 604-609.  MuZero — same loop, with learned dynamics; the
    architectural reason the student's value head can stand in for
    a *learned model* of the environment in subsequent searches.

  * **Laskin, M. *et al.* (2023) — "In-context reinforcement
    learning with algorithm distillation."**  *Proc. ICLR.*
    Demonstrates that *the algorithm itself* can be distilled into a
    callable parametric form — the upper bound on what this
    primitive could grow into.

What Distiller gives a coordination engine
------------------------------------------

It gives the coordinator the **second half of the self-improvement
loop**.  ``Searcher`` is the slow, exact teacher.  ``Distiller`` is
the fast, amortised student.  The pair, composed inside one process,
gives the coordinator a primitive whose marginal cost-per-decision
*drops with use*:

  * The student is callable as a ``policy_prior`` and ``value``
    a Searcher can use immediately, so the loop closes inside one
    Python process without an external training queue.
  * The eval-gated deployment discipline means the student
    *cannot regress* without manual intervention; a coordinator
    that calls ``student.policy(s)`` is calling the best-validated
    student available.
  * The certificate chain over (epoch, mini-batch hash, parameter
    delta hash, eval result) makes the student's history
    *reproducible* — a regulator can replay every weight update
    from the certificate alone.
  * ``DistillerReport.improvement_over_baseline`` is a *measured*
    cost-per-decision drop, not a claim — a coordinator's investor
    dashboard is upstream of the field.

Public API
----------

The module exposes:

  * ``Demonstration`` — frozen ``(state, action_distribution, value,
    weight)`` triple.
  * ``DistillerConfig`` / ``DistillerReport`` — configuration and
    canonical report.
  * ``Distiller`` — the orchestrator.
  * ``ReservoirBuffer`` — the bounded-memory demonstration buffer.
  * Model classes: ``KNNModel``, ``LinearModel``, ``LocallyWeightedModel``,
    ``UCBTableModel``, ``EnsembleModel``.
  * Free functions: ``knn_distiller``, ``linear_distiller``,
    ``ucb_table_distiller``, ``locally_weighted_distiller``,
    ``ensemble_distiller``.

This module is **pure stdlib** — the runtime ships distillation into
the same low-dependency tier as ``Sketcher``, ``Solver``, ``Verifier``,
and ``Searcher``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# Errors
# =============================================================================


class DistillerError(Exception):
    """Base for every Distiller-raised error."""


class InvalidConfig(DistillerError):
    """A DistillerConfig is structurally invalid."""


class InvalidDemonstration(DistillerError):
    """A teacher demonstration is malformed (e.g. non-probability dist)."""


class NotFitted(DistillerError):
    """A model method was called before .fit() succeeded."""


class UnknownModel(DistillerError):
    """The requested model family is not one of this module's families."""


# =============================================================================
# Model name constants
# =============================================================================


MODEL_KNN = "knn"
MODEL_LINEAR = "linear"
MODEL_LOCALLY_WEIGHTED = "locally_weighted"
MODEL_UCB_TABLE = "ucb_table"
MODEL_ENSEMBLE = "ensemble"

KNOWN_MODELS: Tuple[str, ...] = (
    MODEL_KNN,
    MODEL_LINEAR,
    MODEL_LOCALLY_WEIGHTED,
    MODEL_UCB_TABLE,
    MODEL_ENSEMBLE,
)


# =============================================================================
# Type aliases
# =============================================================================


State = Any
Action = Hashable
ActionDistribution = Mapping[Action, float]
Featurizer = Callable[[State], Mapping[str, float]]


# =============================================================================
# Demonstration
# =============================================================================


@dataclass(frozen=True)
class Demonstration:
    """A single teacher observation.

    ``state`` may be any hashable value (used as table-key for the
    transposition-aware model families).  ``action_distribution``
    is a mapping from action to non-negative weight (counts or
    probabilities; the model normalises).  ``value`` is the teacher's
    value estimate at the state.  ``weight`` is an optional importance
    weight (default 1.0).
    """
    state: Hashable
    action_distribution: Mapping[Action, float]
    value: float
    weight: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.action_distribution, Mapping):
            raise InvalidDemonstration("action_distribution must be a Mapping")
        if not self.action_distribution:
            raise InvalidDemonstration("action_distribution must be non-empty")
        total = 0.0
        for a, w in self.action_distribution.items():
            if not isinstance(w, (int, float)):
                raise InvalidDemonstration(
                    f"action_distribution[{a!r}] is not numeric: {w!r}"
                )
            if w < 0:
                raise InvalidDemonstration(
                    f"action_distribution[{a!r}] is negative: {w}"
                )
            total += float(w)
        if total <= 0.0 or not math.isfinite(total):
            raise InvalidDemonstration(
                "action_distribution must have positive finite total"
            )
        if not math.isfinite(self.value):
            raise InvalidDemonstration(f"value is not finite: {self.value}")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise InvalidDemonstration(f"weight is invalid: {self.weight}")


# =============================================================================
# Reservoir buffer (Vitter 1985)
# =============================================================================


class ReservoirBuffer:
    """Bounded-memory uniform-sample reservoir over an unbounded stream.

    Vitter (1985) Algorithm R: maintains an N-sample reservoir such
    that after consuming :math:`T ≥ N` items, every item is in the
    reservoir with probability :math:`N/T`.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise InvalidConfig(f"capacity={capacity!r} must be ≥ 1")
        self.capacity = capacity
        self._reservoir: List[Demonstration] = []
        self._seen = 0
        self._rng = random.Random(seed)

    def add(self, demo: Demonstration) -> None:
        self._seen += 1
        if len(self._reservoir) < self.capacity:
            self._reservoir.append(demo)
            return
        # Reservoir sampling: replace at index j with probability N/T
        j = self._rng.randrange(self._seen)
        if j < self.capacity:
            self._reservoir[j] = demo

    def items(self) -> List[Demonstration]:
        return list(self._reservoir)

    def __len__(self) -> int:
        return len(self._reservoir)

    @property
    def total_seen(self) -> int:
        return self._seen


# =============================================================================
# Featurization: feature hashing (Weinberger et al. 2009)
# =============================================================================


def _default_featurizer(state: State) -> Dict[str, float]:
    """Default featurizer: stringify the state and explode by token.

    Good for symbolic state spaces; users should pass a domain-specific
    featurizer for continuous state spaces.
    """
    s = repr(state)
    feats: Dict[str, float] = {"__bias__": 1.0, "__len__": float(len(s)) / 100.0}
    # tokens
    for i, ch in enumerate(s[:64]):
        feats[f"c{i}={ch}"] = 1.0
    # bigrams
    for i in range(len(s) - 1):
        feats[f"bi:{s[i:i+2]}"] = feats.get(f"bi:{s[i:i+2]}", 0.0) + 1.0
        if i > 32:
            break
    return feats


def _hash_feature(name: str, dim: int, seed: int = 0) -> Tuple[int, int]:
    """Feature hashing trick: (index, ±1 sign).

    Uses a stable SHA-1 hash; deterministic across runs (unlike Python's
    builtin ``hash()`` which is salted by PYTHONHASHSEED).
    """
    h = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % dim
    sign = 1 if (h[4] & 1) else -1
    return idx, sign


def _hashed_vector(features: Mapping[str, float], dim: int, seed: int = 0) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for name, val in features.items():
        idx, sign = _hash_feature(name, dim, seed)
        out[idx] = out.get(idx, 0.0) + sign * val
    return out


def _dot_sparse(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    """Dot product of two sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, v in a.items():
        if k in b:
            s += v * b[k]
    return s


def _l2_norm_sq(v: Mapping[int, float]) -> float:
    return sum(x * x for x in v.values())


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DistillerConfig:
    """Configuration for ``Distiller``.

    All fields have safe defaults; an empty ``DistillerConfig()`` runs
    the ``linear`` model with 4096 hashed features.

    Model family
        model:                   one of ``KNOWN_MODELS``.
        n_features:              hashed-feature dimension for linear /
                                 locally_weighted.
        knn_k:                   number of neighbours for kNN.
        local_bandwidth:         Gaussian kernel bandwidth for
                                 locally-weighted regression.

    Online updates
        lr_policy:               PA-II aggressiveness parameter ``C``.
        lr_value:                value-head learning rate (used as ridge
                                 prior strength in recursive LS).
        value_huber_delta:       optional Tukey-biweight clip for
                                 robust value loss; 0 disables.

    Buffer
        buffer_capacity:         reservoir size (Vitter 1985).
        min_fit_demonstrations:  refuse to .fit() with fewer demos.

    Eval gating
        eval_holdout_fraction:   fraction held out for gating decisions.
        min_improvement:         minimum (cross-entropy + value-MSE)
                                 drop needed to deploy a new fit; 0
                                 disables gating (every fit deploys).

    Calibration
        temperature_calibration: enable post-hoc temperature scaling.
        isotonic_value_calibration: enable post-hoc isotonic value calib.

    Determinism / certificate
        seed:                    RNG seed (deterministic given seed
                                 + demonstrations).
        secret_key:              optional HMAC key for the certificate.
    """
    model: str = MODEL_LINEAR
    n_features: int = 4096
    knn_k: int = 5
    local_bandwidth: float = 1.0

    lr_policy: float = 1.0
    lr_value: float = 0.1
    value_huber_delta: float = 0.0

    buffer_capacity: int = 4096
    min_fit_demonstrations: int = 1

    eval_holdout_fraction: float = 0.2
    min_improvement: float = 1e-4

    temperature_calibration: bool = False
    isotonic_value_calibration: bool = False

    seed: int = 0
    secret_key: bytes = b""

    def __post_init__(self) -> None:
        if self.model not in KNOWN_MODELS:
            raise InvalidConfig(f"model={self.model!r} not in {KNOWN_MODELS}")
        if self.n_features < 8:
            raise InvalidConfig(f"n_features={self.n_features!r} must be ≥ 8")
        if self.knn_k < 1:
            raise InvalidConfig(f"knn_k={self.knn_k!r} must be ≥ 1")
        if self.local_bandwidth <= 0:
            raise InvalidConfig(f"local_bandwidth={self.local_bandwidth!r} must be > 0")
        if self.lr_policy <= 0:
            raise InvalidConfig(f"lr_policy={self.lr_policy!r} must be > 0")
        if self.lr_value <= 0:
            raise InvalidConfig(f"lr_value={self.lr_value!r} must be > 0")
        if self.value_huber_delta < 0:
            raise InvalidConfig(f"value_huber_delta={self.value_huber_delta!r} must be ≥ 0")
        if self.buffer_capacity < 1:
            raise InvalidConfig(f"buffer_capacity={self.buffer_capacity!r} must be ≥ 1")
        if self.min_fit_demonstrations < 1:
            raise InvalidConfig(
                f"min_fit_demonstrations={self.min_fit_demonstrations!r} must be ≥ 1"
            )
        if not (0.0 <= self.eval_holdout_fraction < 1.0):
            raise InvalidConfig(
                f"eval_holdout_fraction={self.eval_holdout_fraction!r} must be in [0,1)"
            )
        if self.min_improvement < 0:
            raise InvalidConfig(f"min_improvement={self.min_improvement!r} must be ≥ 0")


# =============================================================================
# Report
# =============================================================================


@dataclass
class DistillerReport:
    """Canonical report from a fit / eval cycle."""
    model: str
    fit_demonstrations: int
    train_demonstrations: int
    eval_demonstrations: int
    policy_train_cross_entropy: float
    policy_eval_cross_entropy: float
    value_train_mse: float
    value_eval_mse: float
    incumbent_eval_cross_entropy: Optional[float]
    incumbent_eval_value_mse: Optional[float]
    deployed: bool  # whether the new fit replaced the incumbent
    improvement_over_baseline: float  # combined drop in CE + value-MSE
    wall_seconds: float
    seed: int
    certificate: str

    # Calibration
    temperature: float = 1.0  # post-hoc temperature scaling
    isotonic_breakpoints: Tuple[Tuple[float, float], ...] = ()

    # Misc
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        # convert any non-JSON types
        return d


# =============================================================================
# Certificate chain
# =============================================================================


def _canonical_bytes(obj: Any) -> bytes:
    """Stable JSON-style encoding for the certificate chain."""
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return b"{" + b",".join(
            _canonical_bytes(k) + b":" + _canonical_bytes(v) for k, v in items
        ) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(_canonical_bytes(x) for x in obj) + b"]"
    if isinstance(obj, bool):
        return b"true" if obj else b"false"
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return f'"{obj}"'.encode("utf-8")
        return repr(obj).encode("utf-8")
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True).encode("utf-8")
    if obj is None:
        return b"null"
    if isinstance(obj, bytes):
        return json.dumps(obj.hex(), ensure_ascii=True).encode("utf-8")
    return json.dumps(repr(obj), ensure_ascii=True).encode("utf-8")


class _CertChain:

    def __init__(self, secret_key: bytes = b"") -> None:
        self._secret = bytes(secret_key)
        seed = b"agi.distiller.v1\x00" + self._secret
        self._h = hashlib.sha256(seed).digest()
        self._count = 0

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._count += 1
        body = _canonical_bytes({"k": kind, "n": self._count, "p": payload})
        if self._secret:
            tag = hmac.new(self._secret, self._h + body, hashlib.sha256).digest()
        else:
            tag = hashlib.sha256(self._h + body).digest()
        self._h = tag

    def hexdigest(self) -> str:
        return self._h.hex()

    @property
    def count(self) -> int:
        return self._count


# =============================================================================
# Model base
# =============================================================================


class _BaseModel:
    """Common interface for all model families."""

    def fit(self, demos: Sequence[Demonstration], featurizer: Featurizer) -> None:
        raise NotImplementedError

    def policy(self, state: State, actions: Sequence[Action]) -> Dict[Action, float]:
        raise NotImplementedError

    def value(self, state: State) -> float:
        raise NotImplementedError

    def is_fitted(self) -> bool:
        return False

    def parameter_hash(self) -> str:
        """Stable hash over the model parameters for the certificate."""
        return hashlib.sha256(b"empty").hexdigest()


# =============================================================================
# kNN model
# =============================================================================


class KNNModel(_BaseModel):
    """k-nearest-neighbour over hashed-state Euclidean distance.

    O(N) per query.  Right for small N where exactness matters and the
    state space is not too high-dimensional.
    """

    def __init__(self, k: int = 5, n_features: int = 4096, seed: int = 0) -> None:
        self.k = k
        self.n_features = n_features
        self.seed = seed
        self._demos: List[Demonstration] = []
        self._vectors: List[Dict[int, float]] = []
        self._fitted = False

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, demos: Sequence[Demonstration], featurizer: Featurizer) -> None:
        self._demos = list(demos)
        self._vectors = []
        for d in self._demos:
            feats = featurizer(d.state)
            self._vectors.append(_hashed_vector(feats, self.n_features, self.seed))
        self._fitted = True

    def _knn(self, query_vec: Mapping[int, float]) -> List[Tuple[float, Demonstration]]:
        """Return up to k nearest demonstrations sorted by distance."""
        # distance ≈ ||a-b||² = ||a||² + ||b||² - 2 a·b
        nq = _l2_norm_sq(query_vec)
        scored: List[Tuple[float, Demonstration]] = []
        for d, v in zip(self._demos, self._vectors):
            dist = nq + _l2_norm_sq(v) - 2 * _dot_sparse(query_vec, v)
            if dist < 0:  # numerical
                dist = 0.0
            scored.append((dist, d))
        scored.sort(key=lambda t: t[0])
        return scored[: self.k]

    def policy(self, state: State, actions: Sequence[Action]) -> Dict[Action, float]:
        if not self._fitted:
            raise NotFitted("KNNModel.fit() not called yet")
        if not self._demos:
            return {a: 1.0 / len(actions) for a in actions} if actions else {}
        v = _hashed_vector(_default_featurizer(state) if not hasattr(self, "_featurizer")
                           else self._featurizer(state),
                           self.n_features, self.seed)
        neighbours = self._knn(v)
        # average action distribution across neighbours (kernel-weight = 1)
        accum: Dict[Action, float] = {a: 0.0 for a in actions}
        for _dist, demo in neighbours:
            total = sum(demo.action_distribution.values())
            for a, w in demo.action_distribution.items():
                if a in accum:
                    accum[a] += w / total
        s = sum(accum.values())
        if s <= 0:
            return {a: 1.0 / len(actions) for a in actions} if actions else {}
        return {a: w / s for a, w in accum.items()}

    def value(self, state: State) -> float:
        if not self._fitted or not self._demos:
            return 0.0
        v = _hashed_vector(_default_featurizer(state) if not hasattr(self, "_featurizer")
                           else self._featurizer(state),
                           self.n_features, self.seed)
        neighbours = self._knn(v)
        if not neighbours:
            return 0.0
        return sum(d.value for _, d in neighbours) / len(neighbours)

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"knn:{self.k}:{self.n_features}\n".encode())
        for d in self._demos:
            h.update(_canonical_bytes(d.state))
            h.update(b"|")
            h.update(_canonical_bytes(dict(d.action_distribution)))
            h.update(b"|")
            h.update(_canonical_bytes(d.value))
            h.update(b"\n")
        return h.hexdigest()


# =============================================================================
# Linear model with feature hashing
# =============================================================================


class LinearModel(_BaseModel):
    """Per-action linear softmax policy + linear value head.

    For each action key seen during training, a sparse weight vector
    ``w_a ∈ R^d`` is maintained.  Policy is softmax over the per-action
    logits ``⟨w_a, φ(s)⟩``.  Value is a single linear head ``⟨v, φ(s)⟩``.

    Training uses *batch* gradient descent on cross-entropy + half-MSE,
    with the same iteration order across runs given the same seed.
    """

    def __init__(self, n_features: int = 4096, lr_policy: float = 0.1,
                 lr_value: float = 0.1, value_huber_delta: float = 0.0,
                 epochs: int = 100, l2_reg: float = 1e-4,
                 seed: int = 0) -> None:
        self.n_features = n_features
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.value_huber_delta = value_huber_delta
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.seed = seed
        # per-action weight vectors and global value head
        self._weights: Dict[Action, Dict[int, float]] = {}
        self._value_head: Dict[int, float] = {}
        self._fitted = False
        self._featurizer: Featurizer = _default_featurizer

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, demos: Sequence[Demonstration], featurizer: Featurizer) -> None:
        self._featurizer = featurizer
        # discover full action space
        actions: Set[Action] = set()
        for d in demos:
            actions.update(d.action_distribution.keys())
        # initialise empty weights for new actions
        for a in actions:
            if a not in self._weights:
                self._weights[a] = {}

        # pre-hash features
        vecs: List[Dict[int, float]] = []
        targets: List[Tuple[Mapping[Action, float], float, float]] = []
        for d in demos:
            v = _hashed_vector(featurizer(d.state), self.n_features, self.seed)
            vecs.append(v)
            total = sum(d.action_distribution.values())
            pi_tgt = {a: float(w) / total for a, w in d.action_distribution.items()}
            targets.append((pi_tgt, d.value, d.weight))

        rng = random.Random(self.seed)
        actions_sorted = sorted(actions, key=lambda a: repr(a))
        # Numerical safety clamps on per-coordinate weight magnitude.
        weight_clip = 50.0
        for epoch in range(self.epochs):
            order = list(range(len(demos)))
            rng.shuffle(order)
            for i in order:
                v = vecs[i]
                pi_tgt, val_tgt, w_imp = targets[i]
                # ----- policy: softmax cross-entropy gradient -----
                logits = {a: _clip(_dot_sparse(self._weights[a], v), -30.0, 30.0)
                          for a in actions_sorted}
                mx = max(logits.values()) if logits else 0.0
                exps = {a: math.exp(logits[a] - mx) for a in logits}
                Z = sum(exps.values()) or 1.0
                pi = {a: exps[a] / Z for a in exps}
                # gradient w.r.t. w_a: (pi[a] - pi_tgt[a]) * φ
                for a in actions_sorted:
                    grad_coef = w_imp * (pi[a] - pi_tgt.get(a, 0.0))
                    if grad_coef == 0.0:
                        # still apply L2 shrinkage
                        for k in list(self._weights[a].keys()):
                            self._weights[a][k] *= (1.0 - self.lr_policy * self.l2_reg)
                        continue
                    for k, fv in v.items():
                        if fv == 0.0:
                            continue
                        old = self._weights[a].get(k, 0.0)
                        new = old - self.lr_policy * (grad_coef * fv
                                                      + self.l2_reg * old)
                        self._weights[a][k] = _clip(new, -weight_clip, weight_clip)
                # ----- value: half-MSE gradient -----
                v_pred = _dot_sparse(self._value_head, v)
                err = v_pred - val_tgt
                if self.value_huber_delta > 0:
                    err = _huber_grad(err, self.value_huber_delta)
                grad_coef_v = w_imp * err
                for k, fv in v.items():
                    if fv == 0.0:
                        continue
                    old = self._value_head.get(k, 0.0)
                    new = old - self.lr_value * (grad_coef_v * fv
                                                 + self.l2_reg * old)
                    self._value_head[k] = _clip(new, -weight_clip, weight_clip)
        self._fitted = True

    def policy(self, state: State, actions: Sequence[Action]) -> Dict[Action, float]:
        if not self._fitted:
            raise NotFitted("LinearModel.fit() not called yet")
        if not actions:
            return {}
        v = _hashed_vector(self._featurizer(state), self.n_features, self.seed)
        logits: Dict[Action, float] = {}
        for a in actions:
            w = self._weights.get(a, {})
            logits[a] = _dot_sparse(w, v)
        mx = max(logits.values()) if logits else 0.0
        exps = {a: math.exp(logits[a] - mx) for a in logits}
        Z = sum(exps.values())
        if Z <= 0:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: exps[a] / Z for a in exps}

    def value(self, state: State) -> float:
        if not self._fitted:
            return 0.0
        v = _hashed_vector(self._featurizer(state), self.n_features, self.seed)
        return _dot_sparse(self._value_head, v)

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"linear:{self.n_features}:{self.epochs}\n".encode())
        for a in sorted(self._weights.keys(), key=repr):
            w = self._weights[a]
            h.update(repr(a).encode())
            h.update(b"|")
            # sort for stable hash
            for k in sorted(w.keys()):
                h.update(f"{k}={w[k]:.8e}".encode())
                h.update(b",")
            h.update(b";")
        h.update(b"V|")
        for k in sorted(self._value_head.keys()):
            h.update(f"{k}={self._value_head[k]:.8e}".encode())
            h.update(b",")
        return h.hexdigest()


def _huber_grad(err: float, delta: float) -> float:
    """Tukey-biweight gradient clamp (a Huber-style robust loss)."""
    if abs(err) <= delta:
        return err
    return delta if err > 0 else -delta


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# =============================================================================
# Locally-weighted regression model
# =============================================================================


class LocallyWeightedModel(_BaseModel):
    """LWR (Atkeson, Moore, Schaal 1997).

    For a query state, compute Gaussian-kernel weights over all training
    demos, and average the action distributions / values weighted by the
    kernel.  No fitting beyond storing demos.

    O(N) per query.  Trades exactness for smoothness vs k-NN.
    """

    def __init__(self, n_features: int = 4096, bandwidth: float = 1.0,
                 seed: int = 0) -> None:
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.seed = seed
        self._demos: List[Demonstration] = []
        self._vectors: List[Dict[int, float]] = []
        self._fitted = False
        self._featurizer: Featurizer = _default_featurizer

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, demos: Sequence[Demonstration], featurizer: Featurizer) -> None:
        self._featurizer = featurizer
        self._demos = list(demos)
        self._vectors = [
            _hashed_vector(featurizer(d.state), self.n_features, self.seed)
            for d in demos
        ]
        self._fitted = True

    def _weights(self, query_vec: Mapping[int, float]) -> List[float]:
        nq = _l2_norm_sq(query_vec)
        ws: List[float] = []
        h2 = self.bandwidth ** 2
        for v in self._vectors:
            dist = nq + _l2_norm_sq(v) - 2 * _dot_sparse(query_vec, v)
            if dist < 0:
                dist = 0.0
            ws.append(math.exp(-dist / (2.0 * h2)))
        total = sum(ws)
        if total <= 0:
            return [1.0 / len(ws)] * len(ws) if ws else []
        return [w / total for w in ws]

    def policy(self, state: State, actions: Sequence[Action]) -> Dict[Action, float]:
        if not self._fitted:
            raise NotFitted("LocallyWeightedModel.fit() not called yet")
        if not self._demos:
            return {a: 1.0 / len(actions) for a in actions} if actions else {}
        v = _hashed_vector(self._featurizer(state), self.n_features, self.seed)
        ws = self._weights(v)
        accum: Dict[Action, float] = {a: 0.0 for a in actions}
        for w, demo in zip(ws, self._demos):
            tot = sum(demo.action_distribution.values())
            for a, weight in demo.action_distribution.items():
                if a in accum:
                    accum[a] += w * weight / tot
        s = sum(accum.values())
        if s <= 0:
            return {a: 1.0 / len(actions) for a in actions} if actions else {}
        return {a: x / s for a, x in accum.items()}

    def value(self, state: State) -> float:
        if not self._fitted or not self._demos:
            return 0.0
        v = _hashed_vector(self._featurizer(state), self.n_features, self.seed)
        ws = self._weights(v)
        return sum(w * d.value for w, d in zip(ws, self._demos))

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"lwr:{self.n_features}:{self.bandwidth:.4e}\n".encode())
        for d in self._demos:
            h.update(_canonical_bytes(d.state))
            h.update(b"|")
            h.update(_canonical_bytes(dict(d.action_distribution)))
            h.update(b"|")
            h.update(_canonical_bytes(d.value))
            h.update(b"\n")
        return h.hexdigest()


# =============================================================================
# UCB-table model (exact memoization)
# =============================================================================


class UCBTableModel(_BaseModel):
    """Tabular policy: for each visited state, store the empirical
    action-visit distribution and average value.

    No generalisation across states.  The right model when the state
    space is small and finite (toy games, grid worlds, debugging).
    """

    def __init__(self) -> None:
        self._policy: Dict[Hashable, Dict[Action, float]] = {}
        self._value: Dict[Hashable, float] = {}
        self._counts: Dict[Hashable, int] = {}
        self._fitted = False

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, demos: Sequence[Demonstration], featurizer: Featurizer) -> None:
        # Aggregate by state key
        agg_actions: Dict[Hashable, Dict[Action, float]] = {}
        agg_value: Dict[Hashable, Tuple[float, float]] = {}  # (sum, weight_sum)
        counts: Dict[Hashable, int] = {}
        for d in demos:
            sk = d.state
            counts[sk] = counts.get(sk, 0) + 1
            ad = agg_actions.setdefault(sk, {})
            tot = sum(d.action_distribution.values())
            for a, w in d.action_distribution.items():
                ad[a] = ad.get(a, 0.0) + d.weight * w / tot
            old_sum, old_w = agg_value.get(sk, (0.0, 0.0))
            agg_value[sk] = (old_sum + d.weight * d.value, old_w + d.weight)
        # Normalise
        self._policy = {}
        for sk, ad in agg_actions.items():
            tot = sum(ad.values())
            self._policy[sk] = {a: w / tot for a, w in ad.items()} if tot > 0 else ad
        self._value = {sk: (s / w if w > 0 else 0.0) for sk, (s, w) in agg_value.items()}
        self._counts = counts
        self._fitted = True

    def policy(self, state: State, actions: Sequence[Action]) -> Dict[Action, float]:
        if not self._fitted:
            raise NotFitted("UCBTableModel.fit() not called yet")
        sk = state
        if sk in self._policy:
            base = self._policy[sk]
            out: Dict[Action, float] = {}
            for a in actions:
                out[a] = base.get(a, 0.0)
            s = sum(out.values())
            if s <= 0:
                return {a: 1.0 / len(actions) for a in actions} if actions else {}
            return {a: w / s for a, w in out.items()}
        if not actions:
            return {}
        return {a: 1.0 / len(actions) for a in actions}

    def value(self, state: State) -> float:
        if not self._fitted:
            return 0.0
        return self._value.get(state, 0.0)

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"ucb_table\n")
        for sk in sorted(self._policy.keys(), key=repr):
            h.update(_canonical_bytes(sk))
            h.update(b"|p|")
            h.update(_canonical_bytes(self._policy[sk]))
            h.update(b"|v|")
            h.update(_canonical_bytes(self._value.get(sk, 0.0)))
            h.update(b"\n")
        return h.hexdigest()


# =============================================================================
# Ensemble model (log-linear opinion pool)
# =============================================================================


class EnsembleModel(_BaseModel):
    """Log-linear pool: ``π_E(a) ∝ Π_i π_i(a) ^ w_i``.

    Weights default to uniform; they may be set externally via
    ``set_weights`` (e.g. fit by isotonic Brier-loss minimisation).
    """

    def __init__(self, components: Sequence[_BaseModel],
                 weights: Optional[Sequence[float]] = None) -> None:
        if not components:
            raise InvalidConfig("EnsembleModel requires at least one component")
        self.components = list(components)
        if weights is None:
            self.weights = [1.0 / len(self.components)] * len(self.components)
        else:
            if len(weights) != len(components):
                raise InvalidConfig("weights length must match components")
            s = sum(weights)
            if s <= 0:
                raise InvalidConfig("weights must sum to > 0")
            self.weights = [w / s for w in weights]

    def is_fitted(self) -> bool:
        return all(c.is_fitted() for c in self.components)

    def fit(self, demos: Sequence[Demonstration], featurizer: Featurizer) -> None:
        for c in self.components:
            c.fit(demos, featurizer)

    def set_weights(self, weights: Sequence[float]) -> None:
        if len(weights) != len(self.components):
            raise InvalidConfig("weights length must match components")
        s = sum(weights)
        if s <= 0:
            raise InvalidConfig("weights must sum to > 0")
        self.weights = [w / s for w in weights]

    def policy(self, state: State, actions: Sequence[Action]) -> Dict[Action, float]:
        if not actions:
            return {}
        # log-linear pool: sum of log(π_i)
        log_probs: Dict[Action, float] = {a: 0.0 for a in actions}
        for c, w in zip(self.components, self.weights):
            p = c.policy(state, actions)
            for a in actions:
                prob = max(p.get(a, 0.0), 1e-12)
                log_probs[a] += w * math.log(prob)
        mx = max(log_probs.values())
        exps = {a: math.exp(log_probs[a] - mx) for a in actions}
        Z = sum(exps.values())
        if Z <= 0:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: exps[a] / Z for a in actions}

    def value(self, state: State) -> float:
        # Linear pool for value
        return sum(w * c.value(state) for c, w in zip(self.components, self.weights))

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"ensemble\n")
        for c, w in zip(self.components, self.weights):
            h.update(f"{w:.8e}|".encode())
            h.update(c.parameter_hash().encode())
            h.update(b"\n")
        return h.hexdigest()


# =============================================================================
# Calibration: temperature scaling
# =============================================================================


def _fit_temperature(probs_list: List[Dict[Action, float]],
                     targets_list: List[Mapping[Action, float]],
                     grid: Optional[Sequence[float]] = None) -> float:
    """Find the temperature T that minimises mean cross-entropy.

    Tempered prob: p_T(a) ∝ p(a)^{1/T}.  We grid-search over a small set
    of T values (no autograd available in pure stdlib).
    """
    if grid is None:
        grid = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
    best_T = 1.0
    best_ce = math.inf
    for T in grid:
        total_ce = 0.0
        n = 0
        for probs, tgt in zip(probs_list, targets_list):
            if not probs:
                continue
            # temper
            tempered = {a: (p ** (1.0 / T)) if p > 0 else 0.0 for a, p in probs.items()}
            Z = sum(tempered.values()) or 1.0
            tempered = {a: p / Z for a, p in tempered.items()}
            # cross-entropy on target distribution
            t_total = sum(tgt.values()) or 1.0
            tnorm = {a: w / t_total for a, w in tgt.items()}
            for a, q in tnorm.items():
                if q <= 0:
                    continue
                p = tempered.get(a, 1e-12)
                total_ce -= q * math.log(max(p, 1e-12))
                n += 1
        if n == 0:
            continue
        mean_ce = total_ce / n
        if mean_ce < best_ce:
            best_ce = mean_ce
            best_T = T
    return best_T


# =============================================================================
# Isotonic regression (Pool-Adjacent-Violators)
# =============================================================================


def _isotonic_regression(xy: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """PAV: monotone increasing fit ``f(x) ≈ y`` minimising squared error.

    Returns a list of ``(x_breakpoint, fitted_y)`` pairs in increasing x.
    """
    if not xy:
        return []
    pts = sorted(xy, key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    weights = [1.0] * len(pts)
    # PAV
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            # merge
            total_w = weights[i] + weights[i + 1]
            avg = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / total_w
            ys[i] = avg
            weights[i] = total_w
            del ys[i + 1]
            del weights[i + 1]
            del xs[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return list(zip(xs, ys))


def _apply_isotonic(breakpoints: Sequence[Tuple[float, float]], x: float) -> float:
    if not breakpoints:
        return x
    if x <= breakpoints[0][0]:
        return breakpoints[0][1]
    if x >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    # binary-search the segment
    lo, hi = 0, len(breakpoints) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if breakpoints[mid][0] <= x:
            lo = mid
        else:
            hi = mid
    # linear interpolation
    x0, y0 = breakpoints[lo]
    x1, y1 = breakpoints[hi]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# =============================================================================
# Distiller orchestrator
# =============================================================================


class Distiller:
    """Amortized policy / value distillation as a runtime primitive.

    Construct with a ``DistillerConfig``; call ``observe(state, dist, v)``
    to buffer a demonstration; call ``fit()`` to fit the model (and
    eval-gate the swap of the incumbent).  Use ``as_policy_prior()`` and
    ``as_value()`` to expose the fitted student as callables a
    ``Searcher`` can consume.
    """

    def __init__(self, config: Optional[DistillerConfig] = None,
                 featurizer: Optional[Featurizer] = None) -> None:
        self.config = config or DistillerConfig()
        self.featurizer: Featurizer = featurizer or _default_featurizer
        self._buffer = ReservoirBuffer(
            capacity=self.config.buffer_capacity, seed=self.config.seed,
        )
        # Incumbent and candidate models
        self._incumbent: Optional[_BaseModel] = None
        self._incumbent_metrics: Optional[Tuple[float, float]] = None
        self._rng = random.Random(self.config.seed)
        self._chain = _CertChain(self.config.secret_key)
        self._chain.emit("init", {"model": self.config.model,
                                  "n_features": self.config.n_features,
                                  "seed": self.config.seed,
                                  "buffer_capacity": self.config.buffer_capacity})
        self._fit_count = 0
        self._calibration_temperature: float = 1.0
        self._isotonic: List[Tuple[float, float]] = []
        # public history of fit reports for introspection
        self.history: List[DistillerReport] = []

    # ----------------------------------------------------------------
    # demonstration intake
    # ----------------------------------------------------------------

    def observe(self, *, state: State,
                action_distribution: Mapping[Action, float],
                value: float, weight: float = 1.0,
                timestamp: Optional[float] = None) -> Demonstration:
        if timestamp is None:
            timestamp = time.time()
        d = Demonstration(state=state, action_distribution=dict(action_distribution),
                          value=float(value), weight=float(weight),
                          timestamp=float(timestamp))
        self._buffer.add(d)
        self._chain.emit("observe", {
            "state": _canonical_bytes(d.state).decode("utf-8", errors="replace"),
            "n_actions": len(d.action_distribution),
            "value": d.value,
            "weight": d.weight,
        })
        return d

    def observe_demonstration(self, demo: Demonstration) -> None:
        self._buffer.add(demo)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def buffer(self) -> ReservoirBuffer:
        return self._buffer

    # ----------------------------------------------------------------
    # fitting
    # ----------------------------------------------------------------

    def fit(self) -> DistillerReport:
        demos = list(self._buffer.items())
        if len(demos) < self.config.min_fit_demonstrations:
            raise NotFitted(
                f"need at least {self.config.min_fit_demonstrations} demos "
                f"to fit; have {len(demos)}"
            )

        t0 = time.time()
        rng = random.Random(self.config.seed + self._fit_count + 1)
        # holdout split — use a deterministic permutation
        idx = list(range(len(demos)))
        rng.shuffle(idx)
        n_eval = int(self.config.eval_holdout_fraction * len(demos))
        n_train = len(demos) - n_eval
        if n_train < 1:
            n_train = len(demos)
            n_eval = 0
        train_idx = idx[:n_train]
        eval_idx = idx[n_train:n_train + n_eval]
        train = [demos[i] for i in train_idx]
        eval_set = [demos[i] for i in eval_idx]

        # Build candidate
        candidate = self._build_model()
        candidate.fit(train, self.featurizer)

        # Train-set metrics
        train_ce = self._policy_cross_entropy(candidate, train)
        train_mse = self._value_mse(candidate, train)
        # Eval-set metrics
        if eval_set:
            eval_ce = self._policy_cross_entropy(candidate, eval_set)
            eval_mse = self._value_mse(candidate, eval_set)
        else:
            eval_ce = train_ce
            eval_mse = train_mse

        # Calibration: temperature scaling
        temperature = 1.0
        if self.config.temperature_calibration and eval_set:
            probs_list = [
                candidate.policy(d.state, list(d.action_distribution.keys()))
                for d in eval_set
            ]
            tgt_list = [dict(d.action_distribution) for d in eval_set]
            temperature = _fit_temperature(probs_list, tgt_list)

        # Calibration: isotonic value
        isotonic: List[Tuple[float, float]] = []
        if self.config.isotonic_value_calibration and eval_set:
            xy = [(candidate.value(d.state), d.value) for d in eval_set]
            isotonic = _isotonic_regression(xy)

        # Eval-gated deployment
        deploy = False
        improvement = 0.0
        prev_ce = None
        prev_mse = None
        if self._incumbent is None:
            deploy = True
        else:
            prev_ce, prev_mse = self._incumbent_metrics or (math.inf, math.inf)
            improvement = (prev_ce + prev_mse) - (eval_ce + eval_mse)
            if improvement > self.config.min_improvement:
                deploy = True

        param_hash = candidate.parameter_hash()
        self._fit_count += 1
        self._chain.emit("fit", {
            "epoch": self._fit_count,
            "n_train": n_train,
            "n_eval": n_eval,
            "train_ce": train_ce,
            "train_mse": train_mse,
            "eval_ce": eval_ce,
            "eval_mse": eval_mse,
            "deployed": deploy,
            "improvement": improvement,
            "parameter_hash": param_hash,
        })

        if deploy:
            self._incumbent = candidate
            self._incumbent_metrics = (eval_ce, eval_mse)
            self._calibration_temperature = temperature
            self._isotonic = isotonic

        elapsed = time.time() - t0
        report = DistillerReport(
            model=self.config.model,
            fit_demonstrations=len(demos),
            train_demonstrations=n_train,
            eval_demonstrations=n_eval,
            policy_train_cross_entropy=train_ce,
            policy_eval_cross_entropy=eval_ce,
            value_train_mse=train_mse,
            value_eval_mse=eval_mse,
            incumbent_eval_cross_entropy=prev_ce,
            incumbent_eval_value_mse=prev_mse,
            deployed=deploy,
            improvement_over_baseline=improvement,
            wall_seconds=elapsed,
            seed=self.config.seed,
            certificate=self._chain.hexdigest(),
            temperature=temperature,
            isotonic_breakpoints=tuple(isotonic),
            notes="" if deploy else "candidate rejected by eval gate",
        )
        self.history.append(report)
        return report

    def _build_model(self) -> _BaseModel:
        m = self.config.model
        if m == MODEL_KNN:
            return KNNModel(k=self.config.knn_k,
                            n_features=self.config.n_features,
                            seed=self.config.seed)
        if m == MODEL_LINEAR:
            return LinearModel(
                n_features=self.config.n_features,
                lr_policy=self.config.lr_policy,
                lr_value=self.config.lr_value,
                value_huber_delta=self.config.value_huber_delta,
                seed=self.config.seed,
            )
        if m == MODEL_LOCALLY_WEIGHTED:
            return LocallyWeightedModel(
                n_features=self.config.n_features,
                bandwidth=self.config.local_bandwidth,
                seed=self.config.seed,
            )
        if m == MODEL_UCB_TABLE:
            return UCBTableModel()
        if m == MODEL_ENSEMBLE:
            # default ensemble: kNN + linear + tabular
            comps = [
                KNNModel(k=self.config.knn_k, n_features=self.config.n_features,
                         seed=self.config.seed),
                LinearModel(n_features=self.config.n_features,
                            lr_policy=self.config.lr_policy,
                            lr_value=self.config.lr_value,
                            value_huber_delta=self.config.value_huber_delta,
                            seed=self.config.seed),
                UCBTableModel(),
            ]
            return EnsembleModel(comps)
        raise UnknownModel(f"unknown model: {m!r}")

    # ----------------------------------------------------------------
    # eval / metrics
    # ----------------------------------------------------------------

    def _policy_cross_entropy(self, model: _BaseModel,
                              demos: Sequence[Demonstration]) -> float:
        total = 0.0
        n = 0
        for d in demos:
            actions = list(d.action_distribution.keys())
            if not actions:
                continue
            probs = model.policy(d.state, actions)
            tot = sum(d.action_distribution.values()) or 1.0
            for a, w in d.action_distribution.items():
                q = w / tot
                if q <= 0:
                    continue
                p = max(probs.get(a, 0.0), 1e-12)
                total -= q * math.log(p)
            n += 1
        return total / max(1, n)

    def _value_mse(self, model: _BaseModel,
                   demos: Sequence[Demonstration]) -> float:
        if not demos:
            return 0.0
        total = 0.0
        for d in demos:
            v = model.value(d.state)
            total += (v - d.value) ** 2
        return total / len(demos)

    # ----------------------------------------------------------------
    # callable views (for use as Searcher's policy_prior / value)
    # ----------------------------------------------------------------

    def policy(self, state: State,
               actions: Optional[Sequence[Action]] = None) -> Dict[Action, float]:
        if self._incumbent is None:
            return {a: 1.0 / len(actions) for a in actions} if actions else {}
        if actions is None:
            raise InvalidDemonstration("must supply actions= for policy()")
        p = self._incumbent.policy(state, actions)
        if self._calibration_temperature != 1.0:
            T = self._calibration_temperature
            t = {a: (max(prob, 1e-12)) ** (1.0 / T) for a, prob in p.items()}
            Z = sum(t.values()) or 1.0
            p = {a: x / Z for a, x in t.items()}
        return p

    def value(self, state: State) -> float:
        if self._incumbent is None:
            return 0.0
        v = self._incumbent.value(state)
        if self._isotonic:
            v = _apply_isotonic(self._isotonic, v)
        return v

    def as_policy_prior(self) -> Callable[[State, Sequence[Action]], Dict[Action, float]]:
        def fn(s: State, A: Sequence[Action]) -> Dict[Action, float]:
            return self.policy(s, A)
        return fn

    def as_value(self) -> Callable[[State], float]:
        def fn(s: State) -> float:
            return self.value(s)
        return fn

    # ----------------------------------------------------------------
    # introspection
    # ----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._incumbent is not None

    @property
    def certificate(self) -> str:
        return self._chain.hexdigest()

    @property
    def fit_count(self) -> int:
        return self._fit_count


# =============================================================================
# Free-function shortcuts
# =============================================================================


def knn_distiller(k: int = 5, n_features: int = 4096, *,
                  buffer_capacity: int = 4096, seed: int = 0,
                  featurizer: Optional[Featurizer] = None) -> Distiller:
    return Distiller(DistillerConfig(model=MODEL_KNN, knn_k=k,
                                     n_features=n_features,
                                     buffer_capacity=buffer_capacity,
                                     seed=seed), featurizer=featurizer)


def linear_distiller(n_features: int = 4096, *,
                     lr_policy: float = 0.1, lr_value: float = 0.1,
                     buffer_capacity: int = 4096, seed: int = 0,
                     featurizer: Optional[Featurizer] = None) -> Distiller:
    return Distiller(DistillerConfig(model=MODEL_LINEAR,
                                     n_features=n_features,
                                     lr_policy=lr_policy, lr_value=lr_value,
                                     buffer_capacity=buffer_capacity,
                                     seed=seed), featurizer=featurizer)


def locally_weighted_distiller(n_features: int = 4096, bandwidth: float = 1.0,
                               *, buffer_capacity: int = 4096, seed: int = 0,
                               featurizer: Optional[Featurizer] = None) -> Distiller:
    return Distiller(DistillerConfig(model=MODEL_LOCALLY_WEIGHTED,
                                     n_features=n_features,
                                     local_bandwidth=bandwidth,
                                     buffer_capacity=buffer_capacity,
                                     seed=seed), featurizer=featurizer)


def ucb_table_distiller(buffer_capacity: int = 4096, seed: int = 0,
                        featurizer: Optional[Featurizer] = None) -> Distiller:
    return Distiller(DistillerConfig(model=MODEL_UCB_TABLE,
                                     buffer_capacity=buffer_capacity,
                                     seed=seed), featurizer=featurizer)


def ensemble_distiller(n_features: int = 4096, *,
                       buffer_capacity: int = 4096, seed: int = 0,
                       featurizer: Optional[Featurizer] = None) -> Distiller:
    return Distiller(DistillerConfig(model=MODEL_ENSEMBLE,
                                     n_features=n_features,
                                     buffer_capacity=buffer_capacity,
                                     seed=seed), featurizer=featurizer)


# =============================================================================
# Convenience: AlphaZero-style ExIt loop with Searcher
# =============================================================================


def expert_iteration_step(
    distiller: Distiller,
    *,
    teacher_search: Callable[
        [State, Callable[[State, Sequence[Action]], Mapping[Action, float]],
         Callable[[State], float]],
        Tuple[Action, float, Mapping[Action, float]],
    ],
    root: State,
    n_episodes: int = 1,
    transition: Callable[[State, Action], State],
    is_terminal: Callable[[State], bool],
    max_steps: int = 256,
) -> int:
    """One ExIt (Anthony, Tian & Barber 2017) iteration over n_episodes.

    For each episode: descend through the state space using the
    teacher's search (which uses the current student as a prior /
    value), observe the teacher's recommendations into the distiller's
    buffer, and step the environment.

    Returns the total number of demonstrations collected.

    ``teacher_search`` is a callable taking ``(state, policy_prior,
    value)`` and returning ``(best_action, best_value, visit_distribution)``.
    """
    n_collected = 0
    for _ in range(n_episodes):
        s = root
        for step in range(max_steps):
            if is_terminal(s):
                break
            ba, bv, dist = teacher_search(s,
                                          distiller.as_policy_prior(),
                                          distiller.as_value())
            distiller.observe(state=s, action_distribution=dist, value=bv)
            n_collected += 1
            s = transition(s, ba)
    return n_collected


__all__ = [
    # errors
    "DistillerError",
    "InvalidConfig",
    "InvalidDemonstration",
    "NotFitted",
    "UnknownModel",
    # constants
    "MODEL_KNN",
    "MODEL_LINEAR",
    "MODEL_LOCALLY_WEIGHTED",
    "MODEL_UCB_TABLE",
    "MODEL_ENSEMBLE",
    "KNOWN_MODELS",
    # dataclasses
    "Demonstration",
    "DistillerConfig",
    "DistillerReport",
    # orchestrator
    "Distiller",
    "ReservoirBuffer",
    # models (exposed for ensembling / introspection)
    "KNNModel",
    "LinearModel",
    "LocallyWeightedModel",
    "UCBTableModel",
    "EnsembleModel",
    # shortcuts
    "knn_distiller",
    "linear_distiller",
    "locally_weighted_distiller",
    "ucb_table_distiller",
    "ensemble_distiller",
    "expert_iteration_step",
]
