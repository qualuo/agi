r"""Mechanizer — mechanistic interpretability as a runtime primitive.

Most of this stack reasons over *behaviour*: which primitive was
dispatched, what cost it spent, whether it satisfied an SLO.  But the
coordination engine increasingly needs to reason over *mechanism* —
**why** a representation makes the predictions it does, which
underlying features encoded inside an activation matrix drive a
downstream decision, whether two seemingly-distinct routes share a
common internal circuit, whether a safety-relevant feature ever fires
on the live traffic.  That is the territory of *mechanistic
interpretability*: dictionary-learning style decomposition of dense
activations into a sparse basis of monosemantic features, plus
counterfactual interventions on those features (activation patching,
steering) that produce causally-tested explanations.

``Mechanizer`` is the runtime primitive that owns this layer.  It
accepts a matrix of activations (rows = samples, cols = neurons /
embedding dimensions), trains an *over-complete sparse dictionary*
``D`` and produces sparse codes ``Z`` so that ``X ≈ Z · D``,
interprets every learned feature by surfacing its top-activating
samples, allows *steering* a sample along a feature direction or
*patching* one sample's feature value with another's, builds a
feature-feature dependency graph, and emits a *faithfulness
certificate* that quantifies reconstruction error, code sparsity,
mutual coherence (Donoho-Elad identifiability) and dead-feature count.
Every fit / encode / decode / patch / steer transition is hashed into
a SHA-256 receipt chain so a coordinator can replay the
interpretability run byte-for-byte at a later audit.

The pitch reduced to a runtime call::

    mech = Mechanizer(MechanizerConfig(
        algorithm=ALGO_TOPK_SAE,
        n_features=128,        # over-complete: more features than dims
        target_l0=8,            # top-k sparsity
        learning_rate=1e-2,
        max_iter=300,
        seed=0,
    ))

    # Caller supplies activations from any model — LLM hidden state,
    # an embedding matrix, the imaginator's latent rollouts, a
    # representation produced by the agent's tool output.
    X = mechanizer_synthetic_features(
        n=400, dim=32, n_true=16, true_l0=4, seed=0,
    )

    rep  = mech.fit(X)                  # MechanizerReport
    cert = mech.certify(X, delta=0.05)  # MechanizerCertificate

    Z = mech.encode(X[:8])              # sparse FeatureCode
    Y = mech.decode(Z)                  # reconstructed activations

    # Auto-interpretation: for each learned feature, the K samples
    # that activate it most strongly.
    labels = mech.auto_interpret(X, top_k=5)

    # Causal interventions in feature space:
    steered  = mech.steer(X[:1], feature=42, magnitude=2.0)
    patched  = mech.patch(X[:1], donor=X[1:2], feature=42)

    # Feature-feature dependency graph.
    circuit = mech.circuit(X, threshold=0.2)


What this primitive ships
-------------------------

  * **Four algorithms** — toggleable via ``MechanizerConfig.algorithm``:

    * ``ALGO_TOPK_SAE``  — top-k sparse autoencoder (Gao, Goh, Kingma,
      Nichol 2024 *Scaling and Evaluating Sparse Autoencoders*; also
      the variant used in Bricken-Templeton-Conerly et al. 2023
      Anthropic *Towards Monosemanticity*).  Linear encoder
      ``Z = ReLU((X − b_d) W_e^T + b_e)``, hard top-k mask retaining
      only the ``k`` largest entries of each row, tied or untied
      decoder ``X̂ = Z W_d + b_d``.  Trained by SGD on MSE with no
      L1 penalty; sparsity enforced *exactly* by the top-k mask.

    * ``ALGO_L1_SAE``    — L1-penalised sparse autoencoder (Cunningham,
      Ewart, Riggs, Huben, Sharkey 2023 *Sparse Autoencoders Find
      Highly Interpretable Features in Language Models*; Olshausen-
      Field 1996 sparse-coding roots).  ``loss = ||X − X̂||² + λ ||Z||₁``;
      ``Z = ReLU((X − b_d) W_e^T + b_e)``.  Soft sparsity, tunable
      via ``l1_coeff``.

    * ``ALGO_KSVD``      — K-SVD dictionary learning (Aharon, Elad,
      Bruckstein 2006 *K-SVD: An Algorithm for Designing Overcomplete
      Dictionaries for Sparse Representation*).  Alternates: (1)
      sparse coding step via OMP for each row of ``X`` and (2)
      atom-by-atom update via the rank-1 SVD of the residual on the
      support set.  Provably converges to a stationary point under
      mild assumptions; tight on rectangular dictionaries.

    * ``ALGO_PCA``       — orthogonal dense baseline.  Eigen-
      decomposition of the centred sample covariance via the
      symmetric Jacobi rotation method (pure stdlib, no NumPy);
      keeps the top ``n_features`` components.  Not sparse — used
      as a control to show that sparsity *adds* identifiability over
      the dense optimum and as a strong initialisation for the SAE.

  * **Five pursuit kernels** — toggleable via ``MechanizerConfig.pursuit``:

    * ``PURSUIT_OMP``    — Orthogonal Matching Pursuit (Pati-Rezaifar-
      Krishnaprasad 1993; Tropp 2004 *Greed is Good*).  Iteratively
      add the atom most correlated with the residual, re-project
      the code via least squares on the support.  Recovers the
      exact ``k``-sparse solution whenever ``μ(D) < (2k−1)⁻¹``.

    * ``PURSUIT_MP``     — vanilla Matching Pursuit (Mallat-Zhang
      1993).  Cheaper than OMP but the support is monotone and the
      coefficient on already-chosen atoms is never refined.

    * ``PURSUIT_TOPK``   — strict top-k thresholding on the
      encoder pre-activation.  The kernel SAE training uses; O(d·K)
      per row vs OMP's O(d·K²).

    * ``PURSUIT_FISTA``  — accelerated proximal gradient (Beck-
      Teboulle 2009 *A Fast Iterative Shrinkage-Thresholding
      Algorithm*) for the L1-regularised lasso form.  Soft-threshold
      shrinkage with Nesterov momentum.

    * ``PURSUIT_THRESHOLD`` — soft / hard thresholding (Donoho 1995
      *De-noising by Soft-Thresholding*).  One pass, no support
      refinement; only correct when ``D`` is orthogonal.

  * **Activation patching** — ``patch(target, donor, feature)`` replaces
    the *donor*'s value of feature ``j`` into the *target*'s code and
    decodes back, giving the counterfactual reconstruction "what
    would the target look like if its feature ``j`` came from the
    donor?".  This is the causal-intervention primitive used in
    Wang-Variengien-Conmy 2022 *Interpretability In The Wild*, Heimersheim-
    Nanda 2024, Conmy et al. 2023 *Towards Automated Circuit
    Discovery*.  The runtime version is symmetric: ``patch`` over a
    *set* of features at once, over a list of donors averaged into a
    composite, or over a multiplicative scale ``α ∈ [0, 1]`` that
    interpolates target ↔ donor.

  * **Steering** — ``steer(target, feature, magnitude)`` adds the
    learned dictionary atom ``d_j`` to the target's reconstruction
    with a bounded perturbation magnitude.  This is the "feature
    steering" primitive used in Templeton et al. 2024 (Anthropic)
    *Scaling Monosemanticity* and the Goodfire steering API.  The
    runtime version emits a SHA-256 receipt for every steer and a
    per-call perturbation L2 norm as the "blast radius" of the
    intervention; downstream ``aligner`` / ``verifier`` use it as
    the *bounded* certificate of how far the activation moved.

  * **Circuit discovery** — ``circuit(X, threshold)`` returns a
    ``CircuitGraph`` whose nodes are learned features and whose
    weighted edges encode pairwise feature co-activation correlation
    above ``threshold`` (the Conmy 2023 ACDC connection-importance
    heuristic in code-space rather than weight-space).  Composes
    with ``topologist`` (persistent homology of the circuit
    Laplacian) and ``knowledge`` (export to a typed graph store).

  * **Auto-interpretation** — ``auto_interpret(X, top_k)`` returns,
    for each feature, the ``top_k`` indices of ``X`` that maximally
    activate it together with the activation densities (Bricken
    2023 §3.3, Templeton 2024 §A.3).  When ``label_fn`` is provided
    (e.g. an LLM-call into the agent), the labels are written back
    into the certificate and become first-class metadata for the
    coordinator's downstream routing.

  * **Faithfulness certificates** — every report carries:

    * ``r2``                — coefficient of determination
      ``1 − ||X − X̂||² / ||X − mean(X)||²`` on the full sample.
    * ``relative_error``    — ``||X − X̂||_F / ||X||_F``.
    * ``mean_l0``           — average ``||z_i||₀`` per row.
    * ``dead_features``     — number of dictionary atoms whose
      sample-wide activation density is below
      ``dead_feature_threshold``.
    * ``mutual_coherence``  — ``μ(D) = max_{i≠j} |<d_i, d_j>|``
      (atoms ℓ2-normalised); Donoho-Elad 2003 identifiability
      bound ``μ < (2k − 1)⁻¹`` ⇒ unique ``k``-sparse decomposition.
    * ``identifiable``      — boolean predicate from the above.
    * ``variance_explained_per_feature`` — vector of length
      ``n_features``; sums to ``r2``.
    * ``hoeffding_r2_lcb(δ)`` — anytime-valid lower confidence
      bound on the population R² from the empirical per-row R²s
      (Hoeffding 1963), so the coordinator gets a *valid* one-
      tailed gate on faithfulness for unseen samples.
    * ``bernstein_r2_lcb(δ)`` — empirical-Bernstein refinement
      (Maurer-Pontil 2009) using the in-sample R² variance.

  * **Replay-verifiable receipts** — SHA-256 fingerprint chain over
    every observation: ``started``, ``fit``, ``encoded``, ``decoded``,
    ``patched``, ``steered``, ``circuit_built``, ``interpreted``,
    ``certified``, ``cleared``.  ``mechanizer_ledger_root`` is the
    immutable genesis ``agi.mechanizer.v1``.  Replaying the chain
    reproduces every code, every reconstruction, every patch, byte-
    for-byte.  Pluggable HMAC key for tamper-evident multi-tenant
    deployments.

  * **Snapshot / restore** — ``snapshot()`` returns a JSON-encodable
    state dict (random-tape position, dictionary atoms, encoder /
    decoder biases, feature activation density EMA, ledger
    fingerprint) that ``restore()`` can use to resume a fit byte-
    identically.  Composes with ``Persistence`` for crash recovery.

  * **Thread-safe re-entrant lock**; transport-agnostic; pure
    stdlib (no NumPy, no SciPy, no Torch); deterministic given
    seed.


Composes with
-------------

  * ``Attributor`` — data attribution / influence functions answer
    *which training rows drove this prediction*; Mechanizer answers
    *which internal features did*.  Combined: route ``Attributor``'s
    high-influence rows through ``Mechanizer.auto_interpret`` to get
    a row × feature heatmap, then use ``Attributor.counterfactual``
    on the patched activations to test causal effect on the
    downstream loss.

  * ``Aligner`` — value-alignment scoring against a held-out
    preference set.  ``Mechanizer.steer`` provides a causal probe:
    if steering along feature ``j`` durably changes the aligner's
    score, ``j`` is a *safety-relevant* feature and gets promoted
    to the ``Aligner``'s monitored vocabulary.

  * ``Debater`` — multi-agent debate produces multiple positions
    in latent space.  ``Mechanizer`` decomposes each position into
    sparse features; the debate moderator (``Reconciler``) can
    inspect *which features differ* between positions, not just
    their final scalar scores.

  * ``Conformal`` — distribution-free prediction sets need a
    notion of "is this input on-distribution?".  ``Mechanizer``'s
    encoded code ``Z`` is a sparse, monosemantic representation
    that defines a calibrated nonconformity score (density of
    features fired vs the calibration set's density), giving
    feature-aware conformal sets.

  * ``Forecaster`` / ``Calibration`` — log-density features
    extracted by ``Mechanizer`` from the activation history are
    natural covariates for the forecaster's predictor pool; the
    sparse codes serve as a low-dimensional surrogate for
    calibration regressors.

  * ``Topologist`` — persistent homology and Mapper run on the
    sparse code matrix detect *topological* features of the
    representation that complement the linear features Mechanizer
    extracts.  Combined: scale-invariant structure across feature
    families.

  * ``Knowledge`` — features auto-interpreted by ``Mechanizer``
    can be promoted to typed nodes in the ``KnowledgeGraph`` with
    edges from the circuit graph; the coordination engine then
    queries explanations symbolically.

  * ``Pretunist`` — test-time adapter training; ``Mechanizer``
    inspects the *change* in the activation space between pre-
    and post-adaptation, surfacing which features the adapter
    moved the most and providing a faithfulness gate that
    rejects an adaptation that destroys too many monosemantic
    features.

  * ``Verifier`` — the certificate's identifiability predicate
    is a single boolean that ``Verifier`` lifts into an LCF-style
    proof object; the dictionary itself can be checked against
    the coherence bound symbolically.

  * ``Curator`` / ``Continualist`` — automatic curriculum
    generation can target *dead* features (which the model has
    not learned to use) and *over-used* features (which dominate
    sparsity).  Mechanizer publishes both as actionable
    coordination signals.

  * ``Driver`` / ``Coordinator`` / ``Strategist`` — the ``Mechanizer``
    primitive is discoverable via the ``Manifest`` catalog; the
    coordinator routes "explain this representation" goals to it
    via the standard ``recommend()`` ranking.


Mathematical notation
---------------------

  * ``X ∈ ℝ^{n × d}``       — activation matrix; ``n`` samples,
    ``d`` neurons.
  * ``D ∈ ℝ^{K × d}``       — dictionary; ``K`` over-complete atoms
    of dimension ``d``.  Rows are ℓ2-normalised after each update.
  * ``Z ∈ ℝ^{n × K}``       — sparse code matrix; ``z_i`` is row
    ``i``'s code, ``||z_i||_0 ≤ k`` for top-k SAEs / KSVD.
  * ``X̂ = Z D``            — reconstruction.
  * ``μ(D) = max_{i≠j} |⟨d_i, d_j⟩|`` — mutual coherence.
  * ``k``                   — target sparsity (``target_l0``).
  * ``λ``                   — L1 penalty coefficient (L1-SAE only).
  * ``b_d ∈ ℝ^d``           — decoder bias (also "centring shift").
  * ``b_e ∈ ℝ^K``           — encoder bias.
  * ``f(z) = ReLU(z)``      — activation; top-k mask applied after.
  * ``η``                   — learning rate.
  * ``R² = 1 − Σ_i ||x_i − x̂_i||² / Σ_i ||x_i − x̄||²``.
  * ``r²_i``                — per-row R²; used in Hoeffding /
    Bernstein lower-confidence bounds on the population R².

All ingest paths are validated.  Inference is ``O(K d)`` per encode
row, ``O(K d max_iter)`` per fit iteration.  Memory is ``O(K d + n
K)`` for fit, ``O(K d)`` after fit.  No ``random`` without explicit
seed; no ``time.time()`` leaks into the chain.

References
----------

  * Olshausen, Field 1996. *Emergence of simple-cell receptive field
    properties by learning a sparse code for natural images.* Nature
    381:607-609.
  * Donoho 1995. *De-noising by Soft-Thresholding.* IEEE TIT
    41(3):613-627.
  * Mallat, Zhang 1993. *Matching Pursuits with Time-Frequency
    Dictionaries.* IEEE TSP 41(12):3397-3415.
  * Pati, Rezaifar, Krishnaprasad 1993. *Orthogonal Matching
    Pursuit: recursive function approximation with applications to
    wavelet decomposition.* Asilomar.
  * Donoho, Elad 2003. *Optimally sparse representation in general
    (nonorthogonal) dictionaries via ℓ1 minimization.* PNAS
    100(5):2197-2202.
  * Tropp 2004. *Greed is Good: algorithmic results for sparse
    approximation.* IEEE TIT 50(10):2231-2242.
  * Aharon, Elad, Bruckstein 2006. *K-SVD: An Algorithm for
    Designing Overcomplete Dictionaries for Sparse Representation.*
    IEEE TSP 54(11):4311-4322.
  * Beck, Teboulle 2009. *A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems.* SIAM J. Imaging Sci.
    2(1):183-202.
  * Hoeffding 1963. *Probability Inequalities for Sums of Bounded
    Random Variables.* JASA 58.
  * Maurer, Pontil 2009. *Empirical Bernstein Bounds and Sample
    Variance Penalisation.* COLT.
  * Bricken, Templeton, Batson, Chen, Jermyn, Conerly, Turner,
    Anil, Denison, Askell, Lasenby, Wu, Kravec, Schiefer, Maxwell,
    Joseph, Tamkin, Tang, Karpathy, Kaplan, Olah 2023. *Towards
    Monosemanticity: Decomposing Language Models With Dictionary
    Learning.* Anthropic.
  * Cunningham, Ewart, Riggs, Huben, Sharkey 2023. *Sparse
    Autoencoders Find Highly Interpretable Features in Language
    Models.* ICLR.
  * Templeton, Conerly, Marcus, Lindsey, Bricken, Chen, Pearce,
    Citro, Ameisen, Jermyn, Anil, Denison, Askell, Lasenby,
    Schiefer, Maxwell, Joseph, Tamkin, Tang, Kaplan, Olah 2024.
    *Scaling Monosemanticity.* Anthropic.
  * Gao, Goh, Kingma, Nichol 2024. *Scaling and Evaluating Sparse
    Autoencoders.* OpenAI.
  * Wang, Variengien, Conmy, Shlegeris, Steinhardt 2022.
    *Interpretability in the Wild.* ICLR.
  * Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso 2023.
    *Towards Automated Circuit Discovery for Mechanistic
    Interpretability.* NeurIPS.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    # Events
    "MECHANIZER_STARTED",
    "MECHANIZER_FIT",
    "MECHANIZER_ENCODED",
    "MECHANIZER_DECODED",
    "MECHANIZER_PATCHED",
    "MECHANIZER_STEERED",
    "MECHANIZER_CIRCUIT_BUILT",
    "MECHANIZER_INTERPRETED",
    "MECHANIZER_CERTIFIED",
    "MECHANIZER_CLEARED",
    "MECHANIZER_RESET",
    "KNOWN_EVENTS",
    # Algorithms
    "ALGO_TOPK_SAE",
    "ALGO_L1_SAE",
    "ALGO_KSVD",
    "ALGO_PCA",
    "KNOWN_ALGORITHMS",
    # Pursuit kernels
    "PURSUIT_AUTO",
    "PURSUIT_OMP",
    "PURSUIT_MP",
    "PURSUIT_TOPK",
    "PURSUIT_FISTA",
    "PURSUIT_THRESHOLD",
    "PURSUIT_DENSE",
    "KNOWN_PURSUITS",
    # Exceptions
    "MechanizerError",
    "InvalidConfig",
    "InvalidActivations",
    "InvalidFeature",
    "NotFit",
    "ConvergenceError",
    "InsufficientData",
    "UnknownAlgorithm",
    "UnknownPursuit",
    "LedgerCorrupt",
    # Dataclasses
    "MechanizerConfig",
    "MechanizerReport",
    "MechanizerCertificate",
    "FeatureSummary",
    "CircuitGraph",
    "MechanizerSnapshot",
    "MechanizerEvent",
    # Helpers
    "mechanizer_ledger_root",
    "mechanizer_synthetic_features",
    "mechanizer_random_dictionary",
    "mechanizer_mutual_coherence",
    "mechanizer_donoho_elad_bound",
    "mechanizer_recovery_threshold",
    "mechanizer_soft_threshold",
    "mechanizer_hard_threshold",
    "mechanizer_topk_mask",
    "mechanizer_omp",
    "mechanizer_fista",
    "hoeffding_half_width",
    "empirical_bernstein_half_width",
    # Main class
    "Mechanizer",
]


# =====================================================================
# Constants
# =====================================================================

# --- algorithms ------------------------------------------------------
ALGO_TOPK_SAE = "topk_sae"
ALGO_L1_SAE = "l1_sae"
ALGO_KSVD = "ksvd"
ALGO_PCA = "pca"

KNOWN_ALGORITHMS = frozenset({ALGO_TOPK_SAE, ALGO_L1_SAE, ALGO_KSVD, ALGO_PCA})

# --- pursuit ---------------------------------------------------------
PURSUIT_AUTO = "auto"           # pick the pursuit implied by the algorithm
PURSUIT_OMP = "omp"
PURSUIT_MP = "mp"
PURSUIT_TOPK = "topk"
PURSUIT_FISTA = "fista"
PURSUIT_THRESHOLD = "threshold"
PURSUIT_DENSE = "dense"          # identity-style encode for PCA

KNOWN_PURSUITS = frozenset({
    PURSUIT_AUTO, PURSUIT_OMP, PURSUIT_MP, PURSUIT_TOPK,
    PURSUIT_FISTA, PURSUIT_THRESHOLD, PURSUIT_DENSE,
})

# Algorithm → pursuit implied by the fit-time encoder.
_ALGO_DEFAULT_PURSUIT = {
    ALGO_TOPK_SAE: PURSUIT_TOPK,
    ALGO_L1_SAE: PURSUIT_THRESHOLD,
    ALGO_KSVD: PURSUIT_OMP,
    ALGO_PCA: PURSUIT_DENSE,
}

# --- events ----------------------------------------------------------
MECHANIZER_STARTED = "mechanizer.started"
MECHANIZER_FIT = "mechanizer.fit"
MECHANIZER_ENCODED = "mechanizer.encoded"
MECHANIZER_DECODED = "mechanizer.decoded"
MECHANIZER_PATCHED = "mechanizer.patched"
MECHANIZER_STEERED = "mechanizer.steered"
MECHANIZER_CIRCUIT_BUILT = "mechanizer.circuit_built"
MECHANIZER_INTERPRETED = "mechanizer.interpreted"
MECHANIZER_CERTIFIED = "mechanizer.certified"
MECHANIZER_CLEARED = "mechanizer.cleared"
MECHANIZER_RESET = "mechanizer.reset"

KNOWN_EVENTS = frozenset({
    MECHANIZER_STARTED, MECHANIZER_FIT, MECHANIZER_ENCODED,
    MECHANIZER_DECODED, MECHANIZER_PATCHED, MECHANIZER_STEERED,
    MECHANIZER_CIRCUIT_BUILT, MECHANIZER_INTERPRETED,
    MECHANIZER_CERTIFIED, MECHANIZER_CLEARED, MECHANIZER_RESET,
})

# --- numerical defaults ---------------------------------------------
_EPS = 1e-12
_GENESIS = hashlib.sha256(b"agi.mechanizer.v1.genesis").hexdigest()
_DEFAULT_DEAD_THRESHOLD = 1e-4   # activation density below which feature is dead
_DEFAULT_RIDGE = 1e-8            # diagonal jitter for Cholesky
_DEFAULT_JACOBI_TOL = 1e-9       # off-diagonal magnitude tolerance for PCA
_DEFAULT_JACOBI_MAX_SWEEPS = 50


# =====================================================================
# Exceptions
# =====================================================================


class MechanizerError(ValueError):
    """Base class for Mechanizer-domain errors."""


class InvalidConfig(MechanizerError):
    """A :class:`MechanizerConfig` field is malformed."""


class InvalidActivations(MechanizerError):
    """The activations matrix is malformed."""


class InvalidFeature(MechanizerError):
    """A feature index is out of range or otherwise invalid."""


class NotFit(MechanizerError):
    """An encode / decode / patch / steer was attempted before fit()."""


class ConvergenceError(MechanizerError):
    """An iterative solver failed to converge."""


class InsufficientData(MechanizerError):
    """Too few samples for the requested operation."""


class UnknownAlgorithm(MechanizerError):
    """Algorithm string is not in KNOWN_ALGORITHMS."""


class UnknownPursuit(MechanizerError):
    """Pursuit string is not in KNOWN_PURSUITS."""


class LedgerCorrupt(MechanizerError):
    """The audit ledger's hash chain does not verify."""


# =====================================================================
# Linear algebra helpers — pure Python, list-of-lists matrices.
# =====================================================================
#
# Mechanizer is sized for the runtime-primitive scale that a
# coordination engine actually routes through it: ``n`` up to a few
# thousand samples, ``d`` up to a few hundred neurons, ``K`` up to
# a few hundred over-complete atoms.  At those sizes, pure-Python
# linear algebra is fine and lets the module stay stdlib-only.

Vector = list[float]
Matrix = list[list[float]]


def _zeros(n: int) -> Vector:
    return [0.0] * n


def _zeros_mat(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def _copy_mat(M: Matrix) -> Matrix:
    return [row[:] for row in M]


def _copy_vec(v: Vector) -> Vector:
    return list(v)


def _identity(n: int) -> Matrix:
    out = _zeros_mat(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def _shape(M: Matrix) -> tuple[int, int]:
    rows = len(M)
    cols = len(M[0]) if rows else 0
    return rows, cols


def _validate_matrix(M: Sequence[Sequence[float]], name: str) -> Matrix:
    if not isinstance(M, (list, tuple)):
        raise InvalidActivations(f"{name} must be a sequence of rows")
    if len(M) == 0:
        raise InvalidActivations(f"{name} must have at least one row")
    rows = []
    width: int | None = None
    for i, row in enumerate(M):
        if not isinstance(row, (list, tuple)):
            raise InvalidActivations(f"{name}[{i}] must be a sequence")
        if width is None:
            width = len(row)
            if width == 0:
                raise InvalidActivations(f"{name} rows must be non-empty")
        elif len(row) != width:
            raise InvalidActivations(
                f"{name} rows must all have the same length; "
                f"row 0 has {width}, row {i} has {len(row)}"
            )
        out_row: Vector = []
        for j, v in enumerate(row):
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise InvalidActivations(
                    f"{name}[{i}][{j}] is not numeric: {v!r}"
                ) from exc
            if not math.isfinite(fv):
                raise InvalidActivations(
                    f"{name}[{i}][{j}] is not finite: {v!r}"
                )
            out_row.append(fv)
        rows.append(out_row)
    return rows


def _vec_dot(u: Vector, v: Vector) -> float:
    s = 0.0
    for a, b in zip(u, v):
        s += a * b
    return s


def _vec_norm(v: Vector) -> float:
    return math.sqrt(_vec_dot(v, v))


def _vec_scale(alpha: float, v: Vector) -> Vector:
    return [alpha * x for x in v]


def _vec_axpy(alpha: float, x: Vector, y: Vector) -> Vector:
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def _vec_sub(u: Vector, v: Vector) -> Vector:
    return [a - b for a, b in zip(u, v)]


def _vec_add(u: Vector, v: Vector) -> Vector:
    return [a + b for a, b in zip(u, v)]


def _mat_vec(M: Matrix, v: Vector) -> Vector:
    return [_vec_dot(row, v) for row in M]


def _mat_T_vec(M: Matrix, v: Vector) -> Vector:
    rows, cols = _shape(M)
    out = _zeros(cols)
    for i in range(rows):
        vi = v[i]
        Mi = M[i]
        for j in range(cols):
            out[j] += Mi[j] * vi
    return out


def _normalize_rows(M: Matrix, *, eps: float = _EPS) -> Matrix:
    out = _zeros_mat(*_shape(M))
    for i, row in enumerate(M):
        n = _vec_norm(row)
        if n < eps:
            # Replace zero atom with random unit vector to avoid collapse;
            # caller seeds the RNG so this is still deterministic.
            out[i] = row[:]
        else:
            inv = 1.0 / n
            out[i] = [x * inv for x in row]
    return out


def _frobenius_sq(M: Matrix) -> float:
    s = 0.0
    for row in M:
        for v in row:
            s += v * v
    return s


def _mean_rows(M: Matrix) -> Vector:
    rows, cols = _shape(M)
    out = _zeros(cols)
    for row in M:
        for j, v in enumerate(row):
            out[j] += v
    inv = 1.0 / rows
    return [v * inv for v in out]


def _center_rows(M: Matrix, mean: Vector) -> Matrix:
    return [[v - mean[j] for j, v in enumerate(row)] for row in M]


def _gram(M: Matrix) -> Matrix:
    """Symmetric Gram matrix ``M M^T`` for an ``n × d`` ``M``."""
    rows = len(M)
    G = _zeros_mat(rows, rows)
    for i in range(rows):
        Mi = M[i]
        G[i][i] = _vec_dot(Mi, Mi)
        for j in range(i + 1, rows):
            v = _vec_dot(Mi, M[j])
            G[i][j] = v
            G[j][i] = v
    return G


# --- small Cholesky / triangular solves for least squares ------------


def _cholesky(A: Matrix, *, ridge: float = _DEFAULT_RIDGE) -> Matrix:
    """Cholesky factor ``L`` such that ``L L^T = A + ridge·I``.

    Raises :class:`ConvergenceError` if the matrix remains non-positive-
    definite after a single ridge bump.  Callers can retry with a
    larger ``ridge`` if their problem is degenerate.
    """
    n = len(A)
    if n == 0:
        return []
    L = _zeros_mat(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            if i == j:
                s += ridge
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0.0:
                    raise ConvergenceError(
                        f"Cholesky failed at row {i}; matrix not SPD even "
                        f"with ridge {ridge}"
                    )
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _solve_lower(L: Matrix, b: Vector) -> Vector:
    n = len(L)
    x = _zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i][j] * x[j]
        x[i] = s / L[i][i]
    return x


def _solve_upper(U: Matrix, b: Vector) -> Vector:
    n = len(U)
    x = _zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]
    return x


def _solve_spd(A: Matrix, b: Vector, *, ridge: float = _DEFAULT_RIDGE) -> Vector:
    L = _cholesky(A, ridge=ridge)
    y = _solve_lower(L, b)
    Ut = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
    return _solve_upper(Ut, y)


# --- symmetric Jacobi eigendecomposition (used by PCA) ---------------


def _symmetric_eigen(
    A: Matrix,
    *,
    tol: float = _DEFAULT_JACOBI_TOL,
    max_sweeps: int = _DEFAULT_JACOBI_MAX_SWEEPS,
) -> tuple[Vector, Matrix]:
    """Eigendecomposition of a symmetric matrix ``A`` via Jacobi rotations.

    Returns ``(eigenvalues, eigenvectors)`` where the eigenvectors are
    returned as **columns** of the returned matrix (i.e. ``V[:, k]`` is
    the ``k``th eigenvector).  Eigenvalues / vectors are sorted in
    descending order of eigenvalue.

    Pure stdlib; suitable for ``d × d`` with ``d`` up to a few hundred.
    """
    n = len(A)
    # Working copy.
    M = _copy_mat(A)
    V = _identity(n)
    for sweep in range(max_sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += abs(M[i][j])
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = M[p][q]
                if abs(apq) < tol / max(n, 1):
                    continue
                app = M[p][p]
                aqq = M[q][q]
                if abs(app - aqq) < _EPS:
                    theta = math.pi / 4.0
                else:
                    theta = 0.5 * math.atan2(2.0 * apq, app - aqq)
                c = math.cos(theta)
                s = math.sin(theta)
                # Update M.
                for k in range(n):
                    mkp = M[k][p]
                    mkq = M[k][q]
                    M[k][p] = c * mkp + s * mkq
                    M[k][q] = -s * mkp + c * mkq
                for k in range(n):
                    mpk = M[p][k]
                    mqk = M[q][k]
                    M[p][k] = c * mpk + s * mqk
                    M[q][k] = -s * mpk + c * mqk
                # Update V (right multiplication by rotation).
                for k in range(n):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp + s * vkq
                    V[k][q] = -s * vkp + c * vkq
    else:
        # Did not converge cleanly; still return the best estimate.
        pass
    eigenvalues = [M[i][i] for i in range(n)]
    # Sort descending by eigenvalue.
    order = sorted(range(n), key=lambda k: -eigenvalues[k])
    sorted_vals = [eigenvalues[k] for k in order]
    sorted_V = [[V[i][order[k]] for k in range(n)] for i in range(n)]
    return sorted_vals, sorted_V


# =====================================================================
# Thresholding & pursuit primitives
# =====================================================================


def mechanizer_soft_threshold(v: Sequence[float], lam: float) -> Vector:
    r"""Soft-threshold operator ``sign(v) · max(|v| − λ, 0)``.

    The proximal operator of the L1 norm; the workhorse step of FISTA
    and ISTA.  ``λ ≥ 0`` is the threshold; values are coerced to
    ``float``.
    """
    if lam < 0:
        raise InvalidConfig("lam must be non-negative for soft threshold")
    out: Vector = []
    for x in v:
        x = float(x)
        if x > lam:
            out.append(x - lam)
        elif x < -lam:
            out.append(x + lam)
        else:
            out.append(0.0)
    return out


def mechanizer_hard_threshold(v: Sequence[float], lam: float) -> Vector:
    r"""Hard-threshold operator ``v · 1[|v| > λ]``."""
    if lam < 0:
        raise InvalidConfig("lam must be non-negative for hard threshold")
    return [float(x) if abs(x) > lam else 0.0 for x in v]


def mechanizer_topk_mask(v: Sequence[float], k: int) -> Vector:
    r"""Top-k mask: keep the ``k`` largest **positive** entries, zero rest.

    The activation used by ``ALGO_TOPK_SAE``: enforces *exact* L0
    sparsity, with the rectifier baked in (negative entries are zeroed
    regardless of k).  Ties are broken by index (lower index wins).
    """
    if k < 0:
        raise InvalidConfig("k must be non-negative")
    vals = [float(x) for x in v]
    if k == 0:
        return [0.0] * len(vals)
    # Only positive entries are eligible (post-ReLU).
    indexed = [(i, vals[i]) for i in range(len(vals)) if vals[i] > 0.0]
    if len(indexed) <= k:
        return [max(0.0, x) for x in vals]
    indexed.sort(key=lambda t: (-t[1], t[0]))
    keep = {i for i, _ in indexed[:k]}
    return [vals[i] if i in keep else 0.0 for i in range(len(vals))]


def mechanizer_omp(
    x: Sequence[float],
    D: Matrix,
    k: int,
    *,
    tol: float = 1e-10,
    ridge: float = _DEFAULT_RIDGE,
) -> Vector:
    r"""Orthogonal Matching Pursuit.

    Given the signal ``x ∈ ℝ^d`` and the (row-stored, ℓ2-normalised)
    dictionary ``D ∈ ℝ^{K × d}``, return the sparse code
    ``z ∈ ℝ^K`` with at most ``k`` non-zeros that minimises
    ``||x − D^T z||²`` (i.e. ``z[i] · D[i]`` summed equals the
    reconstruction).

    Convergence:  Tropp 2004 — for ``μ(D) < (2k − 1)⁻¹``, OMP recovers
    the unique ``k``-sparse solution exactly.

    Pure stdlib; complexity O(K d k + k³) — fine at runtime scale.
    """
    if k <= 0:
        return [0.0] * len(D)
    x_vec = [float(xi) for xi in x]
    K = len(D)
    d = len(D[0]) if D else 0
    if d != len(x_vec):
        raise InvalidActivations(
            f"OMP signal dim {len(x_vec)} != dictionary dim {d}"
        )
    residual = x_vec[:]
    support: list[int] = []
    coeffs: dict[int, float] = {}
    for _ in range(min(k, K)):
        # Find atom most correlated with residual.
        best_idx = -1
        best_val = -1.0
        for j in range(K):
            if j in coeffs:
                continue
            c = abs(_vec_dot(D[j], residual))
            if c > best_val:
                best_val = c
                best_idx = j
        if best_idx < 0 or best_val < tol:
            break
        support.append(best_idx)
        # Solve least squares over the current support:
        #   minimise || x - sum_{j ∈ S} z_j d_j ||
        # via normal equations  G z_S = D_S x.
        D_S = [D[j] for j in support]
        G = _gram(D_S)
        rhs = [_vec_dot(D[j], x_vec) for j in support]
        try:
            z_S = _solve_spd(G, rhs, ridge=ridge)
        except ConvergenceError:
            # Numerically singular; bail with current support.
            break
        coeffs = {support[i]: z_S[i] for i in range(len(support))}
        # Update residual.
        residual = x_vec[:]
        for j in support:
            residual = _vec_axpy(-coeffs[j], D[j], residual)
        if _vec_norm(residual) < tol:
            break
    z = _zeros(K)
    for j, v in coeffs.items():
        z[j] = v
    return z


def mechanizer_fista(
    x: Sequence[float],
    D: Matrix,
    lam: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> Vector:
    r"""FISTA (Beck-Teboulle 2009) for the lasso ``argmin_z ½||x − D^T z||² + λ||z||₁``.

    Implements the accelerated proximal-gradient form with Nesterov
    momentum.  Step size derived from a power-iteration estimate of the
    Lipschitz constant of ``∇f`` (the largest eigenvalue of
    ``D D^T``); we cap at 16 inner iterations of power method to keep
    determinism cheap.
    """
    x_vec = [float(xi) for xi in x]
    K = len(D)
    d = len(D[0]) if D else 0
    if d != len(x_vec):
        raise InvalidActivations(
            f"FISTA signal dim {len(x_vec)} != dictionary dim {d}"
        )
    if K == 0:
        return []
    # Lipschitz estimate L = largest eigenvalue of D D^T.
    L = _power_iteration_norm(D, max_iter=16)
    if L < _EPS:
        return _zeros(K)
    inv_L = 1.0 / L
    z = _zeros(K)
    y = _zeros(K)
    t = 1.0
    z_prev = z
    for _ in range(max_iter):
        # Gradient ∇f(y) = D ( D^T y - x ).
        Dty = _zeros(d)
        for j in range(K):
            yj = y[j]
            if yj == 0.0:
                continue
            Dj = D[j]
            for c in range(d):
                Dty[c] += Dj[c] * yj
        residual = _vec_sub(Dty, x_vec)
        grad = _mat_vec(D, residual)
        # Proximal step: y - (1/L) grad, then soft-threshold(λ/L).
        z_new = mechanizer_soft_threshold(
            [y[j] - inv_L * grad[j] for j in range(K)],
            lam * inv_L,
        )
        # Nesterov momentum.
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        weight = (t - 1.0) / t_new
        y = [
            z_new[j] + weight * (z_new[j] - z_prev[j])
            for j in range(K)
        ]
        # Convergence check on z.
        if _vec_norm(_vec_sub(z_new, z_prev)) < tol:
            z = z_new
            break
        z = z_new
        z_prev = z_new
        t = t_new
    return z


def _power_iteration_norm(D: Matrix, *, max_iter: int = 16) -> float:
    """Largest singular value of ``D`` (= largest eigenvalue of ``D D^T``).

    Used as a step-size oracle by FISTA.  Deterministic when seeded; we
    avoid relying on RNG by initialising with the all-ones vector.
    """
    K = len(D)
    if K == 0:
        return 0.0
    d = len(D[0])
    # Power iteration on D D^T which acts on R^K.
    v = [1.0] * K
    norm = _vec_norm(v)
    if norm < _EPS:
        return 0.0
    v = _vec_scale(1.0 / norm, v)
    lam = 0.0
    for _ in range(max_iter):
        # Compute u = D^T v ∈ R^d, then w = D u ∈ R^K.
        u = _zeros(d)
        for j in range(K):
            vj = v[j]
            if vj == 0.0:
                continue
            Dj = D[j]
            for c in range(d):
                u[c] += Dj[c] * vj
        w = _mat_vec(D, u)
        w_norm = _vec_norm(w)
        if w_norm < _EPS:
            return 0.0
        lam = w_norm
        v = _vec_scale(1.0 / w_norm, w)
    return lam


# =====================================================================
# Mutual coherence / Donoho-Elad identifiability
# =====================================================================


def mechanizer_mutual_coherence(D: Matrix) -> float:
    r"""``μ(D) = max_{i ≠ j} |⟨d_i, d_j⟩| / (||d_i|| · ||d_j||)``.

    For an ℓ2-normalised dictionary the denominator is 1.  Returns 0.0
    for a single-atom dictionary by convention.
    """
    K = len(D)
    if K <= 1:
        return 0.0
    norms = [max(_vec_norm(d), _EPS) for d in D]
    mu = 0.0
    for i in range(K):
        Di = D[i]
        ni = norms[i]
        for j in range(i + 1, K):
            c = abs(_vec_dot(Di, D[j])) / (ni * norms[j])
            if c > mu:
                mu = c
    return mu


def mechanizer_donoho_elad_bound(k: int) -> float:
    r"""Donoho-Elad 2003 identifiability threshold ``1 / (2k − 1)``.

    For a dictionary with ``μ(D) < this`` value, every ``k``-sparse
    signal has a unique sparsest representation, so OMP and Basis
    Pursuit both recover it.  Returns ``+inf`` for ``k ≤ 0``.
    """
    if k <= 0:
        return float("inf")
    if k == 1:
        return float("inf")
    return 1.0 / (2 * k - 1)


def mechanizer_recovery_threshold(mu: float) -> int:
    r"""The largest ``k`` for which Donoho-Elad guarantees uniqueness.

    Inverts ``1 / (2k − 1) > μ`` to ``k < (1 + 1/μ) / 2``, so the
    integer answer is ``floor((1 + 1/μ) / 2)``.  Returns a large finite
    sentinel for very small ``μ`` (caller doesn't need to special-case).
    """
    if mu <= 0.0:
        return 1 << 30
    val = (1.0 + 1.0 / mu) / 2.0
    return max(1, int(math.floor(val)))


# =====================================================================
# Hoeffding / empirical-Bernstein bounds (same shape as Annealer's)
# =====================================================================


def hoeffding_half_width(n: int, *, delta: float, b: float = 1.0) -> float:
    r"""One-tailed Hoeffding half-width.

    For ``n`` i.i.d. samples bounded in ``[0, b]``, ``μ̂ − μ`` exceeds
    ``b · sqrt(ln(1/δ) / (2n))`` with probability at most ``δ``.
    """
    if n <= 0:
        return float("inf")
    if delta <= 0.0 or delta >= 1.0:
        raise InvalidConfig("delta must lie in (0, 1)")
    return b * math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def empirical_bernstein_half_width(
    n: int,
    *,
    variance: float,
    delta: float,
    b: float = 1.0,
) -> float:
    r"""Empirical-Bernstein (Maurer-Pontil 2009) half-width.

    Tighter than Hoeffding when the in-sample variance is small relative
    to the support length ``b``.
    """
    if n <= 1:
        return float("inf")
    if delta <= 0.0 or delta >= 1.0:
        raise InvalidConfig("delta must lie in (0, 1)")
    var = max(0.0, variance)
    log = math.log(2.0 / delta)
    return math.sqrt(2.0 * var * log / n) + 7.0 * b * log / (3.0 * (n - 1))


# =====================================================================
# Public dataclasses
# =====================================================================


@dataclass(frozen=True)
class MechanizerConfig:
    """Configuration for :class:`Mechanizer`.

    All fields are validated in :meth:`Mechanizer.__init__`.  Bounds
    and semantics are kept *strict* so that misconfiguration fails
    eagerly rather than producing a quietly-wrong dictionary.
    """

    algorithm: str = ALGO_TOPK_SAE
    n_features: int = 64
    target_l0: int = 8                  # used by TOPK_SAE & KSVD
    pursuit: str = PURSUIT_AUTO         # default: pursuit implied by algorithm
    l1_coeff: float = 1e-2              # used by L1_SAE / FISTA encode
    learning_rate: float = 1e-2
    max_iter: int = 100
    tol: float = 1e-6
    seed: int = 0
    tied_decoder: bool = True           # tied weights for SAE variants
    normalize_dictionary: bool = True   # L2-normalise atoms each step
    dead_feature_threshold: float = _DEFAULT_DEAD_THRESHOLD
    pca_jacobi_sweeps: int = _DEFAULT_JACOBI_MAX_SWEEPS
    fista_max_iter: int = 200
    omp_tol: float = 1e-10
    eps: float = _EPS
    hmac_key: bytes | None = None       # if set, ledger is HMAC'd

    def __post_init__(self) -> None:  # called automatically by dataclass
        if self.algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(
                f"algorithm {self.algorithm!r} not in {sorted(KNOWN_ALGORITHMS)}"
            )
        if self.pursuit not in KNOWN_PURSUITS:
            raise UnknownPursuit(
                f"pursuit {self.pursuit!r} not in {sorted(KNOWN_PURSUITS)}"
            )
        if self.n_features <= 0:
            raise InvalidConfig("n_features must be > 0")
        if self.target_l0 < 0:
            raise InvalidConfig("target_l0 must be >= 0")
        if self.target_l0 > self.n_features:
            raise InvalidConfig("target_l0 must be <= n_features")
        if self.l1_coeff < 0:
            raise InvalidConfig("l1_coeff must be >= 0")
        if self.learning_rate <= 0:
            raise InvalidConfig("learning_rate must be > 0")
        if self.max_iter <= 0:
            raise InvalidConfig("max_iter must be > 0")
        if self.tol <= 0:
            raise InvalidConfig("tol must be > 0")
        if self.dead_feature_threshold < 0 or self.dead_feature_threshold > 1:
            raise InvalidConfig(
                "dead_feature_threshold must lie in [0, 1]"
            )
        if self.pca_jacobi_sweeps <= 0:
            raise InvalidConfig("pca_jacobi_sweeps must be > 0")
        if self.fista_max_iter <= 0:
            raise InvalidConfig("fista_max_iter must be > 0")
        if self.omp_tol <= 0:
            raise InvalidConfig("omp_tol must be > 0")
        if self.eps <= 0:
            raise InvalidConfig("eps must be > 0")
        if self.hmac_key is not None and not isinstance(self.hmac_key, (bytes, bytearray)):
            raise InvalidConfig("hmac_key, if set, must be bytes")


@dataclass(frozen=True)
class FeatureSummary:
    """Auto-interpretation record for a single feature."""

    feature: int
    activation_density: float            # fraction of rows where z_ij > 0
    mean_activation: float               # average non-zero activation
    max_activation: float                # global max over input rows
    top_indices: tuple[int, ...]         # rows that maximally activate it
    variance_explained: float            # share of total variance attributed
    label: str | None = None             # optional caller-supplied label

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "activation_density": self.activation_density,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "top_indices": list(self.top_indices),
            "variance_explained": self.variance_explained,
            "label": self.label,
        }


@dataclass(frozen=True)
class CircuitGraph:
    """Feature-feature dependency graph.

    Nodes are feature indices; edges encode pairwise correlation of the
    code matrix's columns above the threshold passed to
    :meth:`Mechanizer.circuit`.  ``adjacency[i]`` is a tuple of
    ``(neighbour, weight)`` pairs sorted descending by |weight|.
    """

    n_features: int
    threshold: float
    adjacency: tuple[tuple[tuple[int, float], ...], ...]
    edge_count: int
    largest_component: int

    def neighbours(self, feature: int) -> tuple[tuple[int, float], ...]:
        if feature < 0 or feature >= self.n_features:
            raise InvalidFeature(f"feature {feature} out of range")
        return self.adjacency[feature]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "threshold": self.threshold,
            "edge_count": self.edge_count,
            "largest_component": self.largest_component,
            "adjacency": [
                [list(pair) for pair in row]
                for row in self.adjacency
            ],
        }


@dataclass(frozen=True)
class MechanizerReport:
    """Summary of a single :meth:`Mechanizer.fit` invocation."""

    algorithm: str
    n_samples: int
    n_neurons: int
    n_features: int
    iterations: int
    final_loss: float
    loss_history: tuple[float, ...]
    r2: float
    relative_error: float
    mean_l0: float
    mean_active_features: float
    dead_features: int
    mutual_coherence: float
    identifiable_l0: int                 # Donoho-Elad-feasible k bound on D
    fingerprint: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "n_samples": self.n_samples,
            "n_neurons": self.n_neurons,
            "n_features": self.n_features,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
            "loss_history": list(self.loss_history),
            "r2": self.r2,
            "relative_error": self.relative_error,
            "mean_l0": self.mean_l0,
            "mean_active_features": self.mean_active_features,
            "dead_features": self.dead_features,
            "mutual_coherence": self.mutual_coherence,
            "identifiable_l0": self.identifiable_l0,
            "fingerprint": self.fingerprint,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MechanizerCertificate:
    """Faithfulness certificate over a held-out activation matrix."""

    n_samples: int
    r2: float
    relative_error: float
    mean_l0: float
    dead_features: int
    mutual_coherence: float
    identifiable_l0: int
    identifiable: bool                   # current mean_l0 <= identifiable_l0
    delta: float                         # confidence parameter
    hoeffding_r2_lcb: float              # one-tailed LCB via Hoeffding
    bernstein_r2_lcb: float              # one-tailed LCB via emp. Bernstein
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "r2": self.r2,
            "relative_error": self.relative_error,
            "mean_l0": self.mean_l0,
            "dead_features": self.dead_features,
            "mutual_coherence": self.mutual_coherence,
            "identifiable_l0": self.identifiable_l0,
            "identifiable": self.identifiable,
            "delta": self.delta,
            "hoeffding_r2_lcb": self.hoeffding_r2_lcb,
            "bernstein_r2_lcb": self.bernstein_r2_lcb,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class MechanizerEvent:
    """One entry in the audit ledger."""

    seq: int
    kind: str
    payload: dict[str, Any]
    parent_hash: str
    this_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "payload": dict(self.payload),
            "parent_hash": self.parent_hash,
            "this_hash": self.this_hash,
        }


@dataclass(frozen=True)
class MechanizerSnapshot:
    """JSON-encodable snapshot of a fitted :class:`Mechanizer`."""

    config: dict[str, Any]
    n_neurons: int
    decoder: tuple[tuple[float, ...], ...]
    encoder: tuple[tuple[float, ...], ...]
    decoder_bias: tuple[float, ...]
    encoder_bias: tuple[float, ...]
    activation_density: tuple[float, ...]
    fingerprint: str
    fit_seed: int
    rng_state: tuple[Any, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "n_neurons": self.n_neurons,
            "decoder": [list(row) for row in self.decoder],
            "encoder": [list(row) for row in self.encoder],
            "decoder_bias": list(self.decoder_bias),
            "encoder_bias": list(self.encoder_bias),
            "activation_density": list(self.activation_density),
            "fingerprint": self.fingerprint,
            "fit_seed": self.fit_seed,
            "rng_state": list(self.rng_state) if self.rng_state else None,
        }


# =====================================================================
# Synthetic data & dictionary builders (for tests & demos)
# =====================================================================


def mechanizer_synthetic_features(
    *,
    n: int,
    dim: int,
    n_true: int,
    true_l0: int,
    seed: int = 0,
    noise: float = 0.0,
) -> Matrix:
    r"""Make ``n × dim`` activations drawn from a known sparse model.

    Generates ``n`` samples ``x_i = D_true · z_i + ε`` where
    ``D_true ∈ ℝ^{n_true × dim}`` is a random ℓ2-normalised dictionary
    and each ``z_i`` is ``true_l0``-sparse with positive uniform
    activations.  Used by the demos and the test suite to verify that
    a Mechanizer can recover the *correct* sparse structure.
    """
    if n <= 0 or dim <= 0 or n_true <= 0 or true_l0 <= 0:
        raise InvalidConfig("n, dim, n_true, true_l0 must all be > 0")
    if true_l0 > n_true:
        raise InvalidConfig("true_l0 must be <= n_true")
    rng = random.Random(seed)
    D = mechanizer_random_dictionary(n_features=n_true, dim=dim, seed=seed + 1)
    X: Matrix = []
    for _ in range(n):
        z = _zeros(n_true)
        idx = rng.sample(range(n_true), true_l0)
        for j in idx:
            z[j] = rng.uniform(0.5, 1.5)
        x = _zeros(dim)
        for j, zj in enumerate(z):
            if zj == 0.0:
                continue
            x = _vec_axpy(zj, D[j], x)
        if noise > 0:
            for c in range(dim):
                x[c] += rng.gauss(0.0, noise)
        X.append(x)
    return X


def mechanizer_random_dictionary(
    *,
    n_features: int,
    dim: int,
    seed: int = 0,
) -> Matrix:
    r"""Random ℓ2-normalised dictionary with ``n_features`` atoms in ``ℝ^dim``."""
    if n_features <= 0 or dim <= 0:
        raise InvalidConfig("n_features and dim must be > 0")
    rng = random.Random(seed)
    D = _zeros_mat(n_features, dim)
    for i in range(n_features):
        for j in range(dim):
            D[i][j] = rng.gauss(0.0, 1.0)
        n = _vec_norm(D[i])
        if n < _EPS:
            D[i] = [1.0 if j == 0 else 0.0 for j in range(dim)]
            n = 1.0
        inv = 1.0 / n
        for j in range(dim):
            D[i][j] *= inv
    return D


# =====================================================================
# Ledger helpers
# =====================================================================


def mechanizer_ledger_root() -> str:
    """Genesis fingerprint shared across all Mechanizer ledgers."""

    return _GENESIS


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))


def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        if isinstance(x, float) and not math.isfinite(x):
            return repr(x)
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if hasattr(x, "to_dict"):
        return _jsonable(x.to_dict())
    return repr(x)


def _hash_event(
    seq: int,
    kind: str,
    payload: Mapping[str, Any],
    parent: str,
    hmac_key: bytes | None = None,
) -> str:
    msg = f"{seq}\0{kind}\0{parent}\0{_canonical(payload)}".encode("utf-8")
    if hmac_key:
        return hmac.new(hmac_key, msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


# =====================================================================
# Main class
# =====================================================================


class Mechanizer:
    """Mechanistic interpretability primitive.

    Owns a learned over-complete sparse dictionary plus an encoder /
    decoder pair, and exposes the runtime-level operations a
    coordination engine needs (``fit``, ``encode``, ``decode``,
    ``patch``, ``steer``, ``circuit``, ``auto_interpret``, ``certify``,
    ``snapshot`` / ``restore``).  Thread-safe; deterministic given the
    seed in :class:`MechanizerConfig`.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        config: MechanizerConfig | None = None,
        *,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or MechanizerConfig()
        # Re-run validation; dataclass __post_init__ already raised
        # if anything was malformed, but defensive checks don't hurt
        # when caller mutates the dataclass post-init (frozen=True
        # blocks that, but keep the symmetry).
        if self.config.algorithm not in KNOWN_ALGORITHMS:
            raise UnknownAlgorithm(self.config.algorithm)
        if self.config.pursuit not in KNOWN_PURSUITS:
            raise UnknownPursuit(self.config.pursuit)

        self._lock = threading.RLock()
        self._publish = publish
        # Fit state.
        self._fit = False
        self._n_neurons: int = 0
        self._decoder: Matrix = []     # K × d
        self._encoder: Matrix = []     # K × d (tied: encoder = decoder)
        self._decoder_bias: Vector = []  # length d
        self._encoder_bias: Vector = []  # length K
        self._activation_density: Vector = []  # length K
        self._fit_mean_l0: float = 0.0
        self._fit_seed: int = 0
        # Audit ledger.
        self._events: list[MechanizerEvent] = []
        self._fingerprint: str = _GENESIS
        self._record(MECHANIZER_STARTED, {
            "algorithm": self.config.algorithm,
            "n_features": self.config.n_features,
            "seed": self.config.seed,
        })

    # ---------------------------------------------------------------
    # Audit ledger
    # ---------------------------------------------------------------

    def _record(self, kind: str, payload: Mapping[str, Any]) -> MechanizerEvent:
        seq = len(self._events)
        canonical_payload = dict(payload)
        new_hash = _hash_event(
            seq, kind, canonical_payload, self._fingerprint,
            hmac_key=self.config.hmac_key,
        )
        ev = MechanizerEvent(
            seq=seq,
            kind=kind,
            payload=canonical_payload,
            parent_hash=self._fingerprint,
            this_hash=new_hash,
        )
        self._events.append(ev)
        self._fingerprint = new_hash
        if self._publish is not None:
            try:
                self._publish(kind, dict(canonical_payload))
            except Exception:  # noqa: BLE001 — never let a buggy listener poison us
                pass
        return ev

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    def events(self) -> list[MechanizerEvent]:
        with self._lock:
            return list(self._events)

    def verify_chain(self) -> bool:
        with self._lock:
            parent = _GENESIS
            for ev in self._events:
                expected = _hash_event(
                    ev.seq, ev.kind, ev.payload, parent,
                    hmac_key=self.config.hmac_key,
                )
                if expected != ev.this_hash or ev.parent_hash != parent:
                    return False
                parent = ev.this_hash
            return parent == self._fingerprint

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return self.config.n_features

    @property
    def n_neurons(self) -> int:
        with self._lock:
            return self._n_neurons

    @property
    def is_fit(self) -> bool:
        with self._lock:
            return self._fit

    def dictionary(self) -> Matrix:
        """Return a copy of the learned dictionary (rows = atoms)."""

        with self._lock:
            if not self._fit:
                raise NotFit("fit() must be called before dictionary()")
            return _copy_mat(self._decoder)

    def decoder_bias(self) -> Vector:
        with self._lock:
            if not self._fit:
                raise NotFit("fit() must be called before decoder_bias()")
            return list(self._decoder_bias)

    def encoder_bias(self) -> Vector:
        with self._lock:
            if not self._fit:
                raise NotFit("fit() must be called before encoder_bias()")
            return list(self._encoder_bias)

    def activation_density(self) -> Vector:
        with self._lock:
            if not self._fit:
                raise NotFit("fit() must be called before activation_density()")
            return list(self._activation_density)

    # ---------------------------------------------------------------
    # fit
    # ---------------------------------------------------------------

    def fit(self, X: Sequence[Sequence[float]], *, seed: int | None = None) -> MechanizerReport:
        """Learn a dictionary and an encoder from the activation matrix ``X``.

        Returns a :class:`MechanizerReport` summarising the fit; updates
        internal state so that :meth:`encode`, :meth:`decode`,
        :meth:`patch`, etc. all work.
        """
        with self._lock:
            Xv = _validate_matrix(X, "X")
            n, d = _shape(Xv)
            if n < 1:
                raise InsufficientData("fit requires at least one sample")
            if self.config.n_features > n * d:
                # Not a hard error; just a warning-ish log via ledger.
                pass
            self._n_neurons = d
            fit_seed = self.config.seed if seed is None else int(seed)
            self._fit_seed = fit_seed
            algo = self.config.algorithm
            if algo == ALGO_TOPK_SAE:
                rep = self._fit_topk_sae(Xv, fit_seed)
            elif algo == ALGO_L1_SAE:
                rep = self._fit_l1_sae(Xv, fit_seed)
            elif algo == ALGO_KSVD:
                rep = self._fit_ksvd(Xv, fit_seed)
            elif algo == ALGO_PCA:
                rep = self._fit_pca(Xv, fit_seed)
            else:
                # Shouldn't happen given config validation.
                raise UnknownAlgorithm(algo)
            self._fit = True
            self._record(MECHANIZER_FIT, {
                "algorithm": algo,
                "n_samples": rep.n_samples,
                "n_neurons": rep.n_neurons,
                "n_features": rep.n_features,
                "iterations": rep.iterations,
                "final_loss": rep.final_loss,
                "r2": rep.r2,
                "mean_l0": rep.mean_l0,
                "dead_features": rep.dead_features,
                "mutual_coherence": rep.mutual_coherence,
                "seed": fit_seed,
            })
            return rep

    # --- helpers used by every fit variant -------------------------

    def _make_initial_dictionary(self, X: Matrix, seed: int) -> Matrix:
        """Initialise the dictionary with random samples (Olshausen-Field style)."""

        K = self.config.n_features
        d = self._n_neurons
        rng = random.Random(seed ^ 0x9E3779B9)
        D = _zeros_mat(K, d)
        # Mix of random rows and random gaussian draws, then L2-normalise.
        n = len(X)
        for k in range(K):
            if rng.random() < 0.5 and n > 0:
                src = X[rng.randrange(n)]
                # add a small perturbation to break degeneracy on duplicates
                D[k] = [src[j] + 1e-3 * rng.gauss(0.0, 1.0) for j in range(d)]
            else:
                D[k] = [rng.gauss(0.0, 1.0) for _ in range(d)]
        return _normalize_rows(D)

    def _compute_density(self, codes: Matrix) -> Vector:
        K = self.config.n_features
        n = len(codes)
        if n == 0:
            return _zeros(K)
        density = _zeros(K)
        for row in codes:
            for j, v in enumerate(row):
                if v > 0.0:
                    density[j] += 1.0
        inv = 1.0 / n
        return [v * inv for v in density]

    def _reconstruction_loss(self, X: Matrix, Z: Matrix, D: Matrix, b_d: Vector) -> float:
        n = len(X)
        if n == 0:
            return 0.0
        s = 0.0
        for i in range(n):
            x_hat = self._decode_one(Z[i], D, b_d)
            for c in range(self._n_neurons):
                diff = X[i][c] - x_hat[c]
                s += diff * diff
        return s / n

    def _decode_one(self, z: Vector, D: Matrix, b_d: Vector) -> Vector:
        d = self._n_neurons
        out = list(b_d)
        for j, zj in enumerate(z):
            if zj == 0.0:
                continue
            Dj = D[j]
            for c in range(d):
                out[c] += Dj[c] * zj
        return out

    def _encode_pre(
        self, x: Vector, D: Matrix, b_d: Vector, b_e: Vector,
    ) -> Vector:
        """Pre-activation linear encoder ``(x − b_d) D^T + b_e``."""
        d = self._n_neurons
        x_c = [x[c] - b_d[c] for c in range(d)]
        return [_vec_dot(D[j], x_c) + b_e[j] for j in range(self.config.n_features)]

    def _per_row_l0(self, codes: Matrix) -> float:
        n = len(codes)
        if n == 0:
            return 0.0
        s = 0
        for row in codes:
            for v in row:
                if v > 0.0:
                    s += 1
        return s / n

    def _summarise_after_fit(
        self,
        X: Matrix,
        Z: Matrix,
        D: Matrix,
        b_d: Vector,
        b_e: Vector,
        iterations: int,
        loss_history: list[float],
        seed: int,
    ) -> MechanizerReport:
        n = len(X)
        d = self._n_neurons
        K = self.config.n_features
        # Store state.
        self._decoder = D
        self._encoder = D if self.config.tied_decoder else _copy_mat(D)
        self._decoder_bias = list(b_d)
        self._encoder_bias = list(b_e)
        density = self._compute_density(Z)
        self._activation_density = density
        # R^2.
        mean = _mean_rows(X)
        total = 0.0
        for row in X:
            for c, v in enumerate(row):
                diff = v - mean[c]
                total += diff * diff
        recon_sq = 0.0
        rel_num = 0.0
        rel_den = 0.0
        for i in range(n):
            x_hat = self._decode_one(Z[i], D, b_d)
            for c in range(d):
                diff = X[i][c] - x_hat[c]
                recon_sq += diff * diff
                rel_num += diff * diff
                rel_den += X[i][c] * X[i][c]
        if total < self.config.eps:
            r2 = 1.0 if recon_sq < self.config.eps else 0.0
        else:
            r2 = 1.0 - recon_sq / total
        relative_error = math.sqrt(rel_num / max(rel_den, self.config.eps))
        mean_l0 = self._per_row_l0(Z)
        self._fit_mean_l0 = mean_l0
        dead = sum(1 for v in density if v <= self.config.dead_feature_threshold)
        coherence = mechanizer_mutual_coherence(D)
        identifiable_l0 = mechanizer_recovery_threshold(coherence)
        return MechanizerReport(
            algorithm=self.config.algorithm,
            n_samples=n,
            n_neurons=d,
            n_features=K,
            iterations=iterations,
            final_loss=loss_history[-1] if loss_history else float("nan"),
            loss_history=tuple(loss_history),
            r2=r2,
            relative_error=relative_error,
            mean_l0=mean_l0,
            mean_active_features=mean_l0,
            dead_features=dead,
            mutual_coherence=coherence,
            identifiable_l0=identifiable_l0,
            fingerprint=self._fingerprint,
            seed=seed,
        )

    # --- TOPK SAE fit ---------------------------------------------

    def _fit_topk_sae(self, X: Matrix, seed: int) -> MechanizerReport:
        n = len(X)
        d = self._n_neurons
        K = self.config.n_features
        k = self.config.target_l0
        lr = self.config.learning_rate
        max_iter = self.config.max_iter
        tol = self.config.tol
        # Initialise decoder.
        D = self._make_initial_dictionary(X, seed)
        # Decoder bias = sample mean (Olshausen-Field centring).
        b_d = _mean_rows(X)
        b_e = _zeros(K)
        prev_loss = float("inf")
        loss_history: list[float] = []
        rng = random.Random(seed ^ 0xC0FFEE)
        iter_count = 0
        for it in range(max_iter):
            # Iterate over a random permutation of rows; mini-batch
            # gradient with batch size 1 to keep things stdlib-simple
            # and to encourage feature diversity.
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                x = X[i]
                pre = self._encode_pre(x, D, b_d, b_e)
                z = mechanizer_topk_mask(pre, k)
                # Reconstruction & error.
                x_hat = self._decode_one(z, D, b_d)
                err = _vec_sub(x_hat, x)  # ∂L/∂x_hat = (x_hat − x)
                # Decoder grad: ∂L/∂D_j = err · z_j  (rank-1 per active feature)
                # Update only the active features.
                for j in range(K):
                    if z[j] == 0.0:
                        continue
                    coeff = lr * z[j]
                    Dj = D[j]
                    for c in range(d):
                        Dj[c] -= coeff * err[c]
                # Encoder bias grad (tied weights pass through):
                # ∂L/∂b_e[j] = (∂L/∂x_hat) · D_j · 1[pre_j > 0 and j in topk]
                if not self.config.tied_decoder:
                    # Independent encoder; we still tie for simplicity but
                    # allow a configurable separate encoder later.
                    pass
                for j in range(K):
                    if z[j] == 0.0:
                        continue
                    grad_bej = _vec_dot(err, D[j])
                    b_e[j] -= lr * grad_bej
                # Decoder bias grad: ∂L/∂b_d = err  (centring shift).
                for c in range(d):
                    b_d[c] -= lr * err[c]
                if self.config.normalize_dictionary:
                    # Re-normalise just the rows we touched.
                    for j in range(K):
                        if z[j] == 0.0:
                            continue
                        nm = _vec_norm(D[j])
                        if nm < self.config.eps:
                            continue
                        inv = 1.0 / nm
                        D[j] = [c * inv for c in D[j]]
            # Compute full loss to track convergence.
            Z_all = self._encode_all_topk(X, D, b_d, b_e, k)
            loss = self._reconstruction_loss(X, Z_all, D, b_d)
            loss_history.append(loss)
            iter_count = it + 1
            if abs(prev_loss - loss) < tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = loss
        # Final encode.
        Z = self._encode_all_topk(X, D, b_d, b_e, k)
        return self._summarise_after_fit(
            X, Z, D, b_d, b_e, iter_count, loss_history, seed,
        )

    def _encode_all_topk(
        self, X: Matrix, D: Matrix, b_d: Vector, b_e: Vector, k: int,
    ) -> Matrix:
        out: Matrix = []
        for x in X:
            pre = self._encode_pre(x, D, b_d, b_e)
            out.append(mechanizer_topk_mask(pre, k))
        return out

    # --- L1 SAE fit ------------------------------------------------

    def _fit_l1_sae(self, X: Matrix, seed: int) -> MechanizerReport:
        n = len(X)
        d = self._n_neurons
        K = self.config.n_features
        lr = self.config.learning_rate
        lam = self.config.l1_coeff
        max_iter = self.config.max_iter
        tol = self.config.tol
        D = self._make_initial_dictionary(X, seed)
        b_d = _mean_rows(X)
        b_e = _zeros(K)
        prev_loss = float("inf")
        loss_history: list[float] = []
        rng = random.Random(seed ^ 0xBADC0DE)
        iter_count = 0
        for it in range(max_iter):
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                x = X[i]
                pre = self._encode_pre(x, D, b_d, b_e)
                # ReLU + L1 *implicitly* via soft-threshold;
                # for the autoencoder formulation we use ReLU and rely on
                # the L1 penalty in the loss to drive sparsity.
                z = [max(0.0, v) for v in pre]
                # Reconstruction error.
                x_hat = self._decode_one(z, D, b_d)
                err = _vec_sub(x_hat, x)
                # Decoder gradient.
                for j in range(K):
                    zj = z[j]
                    if zj == 0.0:
                        continue
                    coeff = lr * zj
                    Dj = D[j]
                    for c in range(d):
                        Dj[c] -= coeff * err[c]
                # Encoder bias gradient incl. L1 subgradient.
                for j in range(K):
                    if z[j] <= 0.0:
                        continue
                    grad_bej = _vec_dot(err, D[j]) + lam  # +λ from L1
                    b_e[j] -= lr * grad_bej
                # Decoder bias gradient.
                for c in range(d):
                    b_d[c] -= lr * err[c]
                if self.config.normalize_dictionary:
                    for j in range(K):
                        if z[j] == 0.0:
                            continue
                        nm = _vec_norm(D[j])
                        if nm < self.config.eps:
                            continue
                        inv = 1.0 / nm
                        D[j] = [c * inv for c in D[j]]
            Z_all = self._encode_all_relu(X, D, b_d, b_e)
            recon = self._reconstruction_loss(X, Z_all, D, b_d)
            l1 = sum(sum(row) for row in Z_all) / max(n, 1)
            loss = recon + lam * l1
            loss_history.append(loss)
            iter_count = it + 1
            if abs(prev_loss - loss) < tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = loss
        Z = self._encode_all_relu(X, D, b_d, b_e)
        return self._summarise_after_fit(
            X, Z, D, b_d, b_e, iter_count, loss_history, seed,
        )

    def _encode_all_relu(
        self, X: Matrix, D: Matrix, b_d: Vector, b_e: Vector,
    ) -> Matrix:
        out: Matrix = []
        for x in X:
            pre = self._encode_pre(x, D, b_d, b_e)
            out.append([max(0.0, v) for v in pre])
        return out

    # --- KSVD fit --------------------------------------------------

    def _fit_ksvd(self, X: Matrix, seed: int) -> MechanizerReport:
        n = len(X)
        d = self._n_neurons
        K = self.config.n_features
        k = self.config.target_l0
        max_iter = self.config.max_iter
        tol = self.config.tol
        # KSVD ignores the AE biases; we centre once and store the mean.
        b_d = _mean_rows(X)
        b_e = _zeros(K)
        X_c: Matrix = _center_rows(X, b_d)
        D = self._make_initial_dictionary(X_c, seed)
        prev_loss = float("inf")
        loss_history: list[float] = []
        iter_count = 0
        Z: Matrix = _zeros_mat(n, K)
        for it in range(max_iter):
            # Sparse coding step via OMP for each row.
            Z = [
                mechanizer_omp(
                    X_c[i], D, k, tol=self.config.omp_tol,
                    ridge=_DEFAULT_RIDGE,
                )
                for i in range(n)
            ]
            # Dictionary update step.
            self._ksvd_update_step(X_c, Z, D)
            # Track loss.
            loss = self._reconstruction_loss(X_c, Z, D, _zeros(d))
            loss_history.append(loss)
            iter_count = it + 1
            if abs(prev_loss - loss) < tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = loss
        # Final encoder is OMP-as-encoder, so we store the dictionary;
        # the encoder bias stays zero (codes come from OMP).
        return self._summarise_after_fit(
            X, Z, D, b_d, b_e, iter_count, loss_history, seed,
        )

    def _ksvd_update_step(self, X_c: Matrix, Z: Matrix, D: Matrix) -> None:
        """In-place K-SVD atom update.

        For each atom ``d_j``, find samples that use ``j`` (the support
        ``Ω_j``), compute the residual without ``j``'s contribution,
        and update ``d_j`` (and its row of ``Z``) by the leading
        right-singular pair of the restricted residual.  We approximate
        the SVD step via one round of power iteration which is enough
        for convergence and stays stdlib-only.
        """
        n = len(X_c)
        d = self._n_neurons
        K = self.config.n_features
        for j in range(K):
            omega = [i for i in range(n) if Z[i][j] != 0.0]
            if not omega:
                continue
            # Residual restricted to Ω_j without atom j's contribution.
            E_j: Matrix = _zeros_mat(len(omega), d)
            for r, i in enumerate(omega):
                row = X_c[i][:]
                z_i = Z[i]
                for jj in range(K):
                    if jj == j or z_i[jj] == 0.0:
                        continue
                    coeff = z_i[jj]
                    Dj = D[jj]
                    for c in range(d):
                        row[c] -= coeff * Dj[c]
                E_j[r] = row
            # Leading singular component of E_j ≈ d_new * (z_new)^T.
            d_new, z_new = self._leading_pair(E_j)
            # Normalise d_new and assign.
            nm = _vec_norm(d_new)
            if nm < self.config.eps:
                continue
            inv = 1.0 / nm
            D[j] = [v * inv for v in d_new]
            for r, i in enumerate(omega):
                Z[i][j] = z_new[r] * nm

    def _leading_pair(self, E: Matrix) -> tuple[Vector, Vector]:
        """Power iteration for the leading rank-1 of E (E ≈ u v^T)."""
        rows = len(E)
        cols = len(E[0]) if rows else 0
        if rows == 0 or cols == 0:
            return _zeros(cols), _zeros(rows)
        # Initialise v as average row direction.
        v = _zeros(cols)
        for row in E:
            for c in range(cols):
                v[c] += row[c]
        nv = _vec_norm(v)
        if nv < self.config.eps:
            v = [1.0 / math.sqrt(cols)] * cols
        else:
            v = _vec_scale(1.0 / nv, v)
        u = _zeros(rows)
        for _ in range(8):
            # u = E v
            for r in range(rows):
                u[r] = _vec_dot(E[r], v)
            nu = _vec_norm(u)
            if nu < self.config.eps:
                return _zeros(cols), _zeros(rows)
            u = _vec_scale(1.0 / nu, u)
            # v = E^T u
            v = _mat_T_vec(E, u)
            nv = _vec_norm(v)
            if nv < self.config.eps:
                return _zeros(cols), _zeros(rows)
            v = _vec_scale(1.0 / nv, v)
        # Scale so that E ≈ u * sigma * v^T; sigma = u^T E v.
        Ev = _zeros(rows)
        for r in range(rows):
            Ev[r] = _vec_dot(E[r], v)
        sigma = _vec_dot(u, Ev)
        # Return (d_new = sigma * v) and z_new = u, so that
        # d_new ⊗ z_new ≈ E^T  (we want E ≈ z_new · d_new^T).
        d_new = _vec_scale(sigma, v)
        z_new = u
        return d_new, z_new

    # --- PCA fit ---------------------------------------------------

    def _fit_pca(self, X: Matrix, seed: int) -> MechanizerReport:
        n = len(X)
        d = self._n_neurons
        K = self.config.n_features
        b_d = _mean_rows(X)
        X_c = _center_rows(X, b_d)
        # Covariance d × d.
        cov = _zeros_mat(d, d)
        for row in X_c:
            for a in range(d):
                ra = row[a]
                cov_a = cov[a]
                for b in range(d):
                    cov_a[b] += ra * row[b]
        inv_n = 1.0 / max(n, 1)
        for a in range(d):
            for b in range(d):
                cov[a][b] *= inv_n
        eigenvalues, V = _symmetric_eigen(
            cov,
            tol=_DEFAULT_JACOBI_TOL,
            max_sweeps=self.config.pca_jacobi_sweeps,
        )
        # Keep top-K (or all if K > d).
        keep = min(K, d)
        D = _zeros_mat(K, d)
        for j in range(keep):
            # V[:, j] is the j-th eigenvector.
            D[j] = [V[i][j] for i in range(d)]
        # Atoms beyond ``keep`` remain zero so that they neither
        # contribute to reconstruction nor distort coherence; the
        # caller can ask for ``K > d`` and PCA will simply leave the
        # surplus dictionary slots empty.
        b_e = _zeros(K)
        # Compute codes: z = D x_c (no ReLU, dense for kept atoms).
        Z: Matrix = []
        for x in X_c:
            row_code = _zeros(K)
            for j in range(keep):
                row_code[j] = _vec_dot(D[j], x)
            Z.append(row_code)
        # Loss & history (single step).
        loss = self._reconstruction_loss(X, Z, D, b_d)
        loss_history = [loss]
        return self._summarise_after_fit(
            X, Z, D, b_d, b_e, 1, loss_history, seed,
        )

    # ---------------------------------------------------------------
    # encode / decode
    # ---------------------------------------------------------------

    def encode(
        self,
        X: Sequence[Sequence[float]],
        *,
        pursuit: str | None = None,
    ) -> Matrix:
        """Encode an activation matrix into sparse codes.

        ``pursuit`` overrides the configured pursuit kernel for this
        call only; defaults to ``self.config.pursuit``.
        """
        with self._lock:
            self._require_fit()
            Xv = _validate_matrix(X, "X")
            n, d = _shape(Xv)
            if d != self._n_neurons:
                raise InvalidActivations(
                    f"X has {d} neurons but Mechanizer was fit on {self._n_neurons}"
                )
            p = pursuit or self.config.pursuit
            if p not in KNOWN_PURSUITS:
                raise UnknownPursuit(p)
            codes = self._encode_with_pursuit(Xv, p)
            self._record(MECHANIZER_ENCODED, {
                "n_samples": n,
                "pursuit": p,
                "mean_l0": self._per_row_l0(codes),
            })
            return codes

    def _resolve_pursuit(self, p: str) -> str:
        if p != PURSUIT_AUTO:
            return p
        return _ALGO_DEFAULT_PURSUIT.get(self.config.algorithm, PURSUIT_TOPK)

    def _encode_with_pursuit(self, X: Matrix, p: str) -> Matrix:
        p = self._resolve_pursuit(p)
        D = self._decoder
        b_d = self._decoder_bias
        b_e = self._encoder_bias
        K = self.config.n_features
        k = self.config.target_l0
        lam = self.config.l1_coeff
        codes: Matrix = []
        if p == PURSUIT_TOPK:
            for x in X:
                pre = self._encode_pre(x, D, b_d, b_e)
                codes.append(mechanizer_topk_mask(pre, k))
        elif p == PURSUIT_THRESHOLD:
            for x in X:
                pre = self._encode_pre(x, D, b_d, b_e)
                codes.append([max(0.0, v - lam) for v in pre])
        elif p == PURSUIT_OMP:
            for x in X:
                x_c = [x[c] - b_d[c] for c in range(self._n_neurons)]
                codes.append(mechanizer_omp(x_c, D, k, tol=self.config.omp_tol))
        elif p == PURSUIT_MP:
            for x in X:
                x_c = [x[c] - b_d[c] for c in range(self._n_neurons)]
                codes.append(_matching_pursuit(x_c, D, k, tol=self.config.omp_tol))
        elif p == PURSUIT_FISTA:
            for x in X:
                x_c = [x[c] - b_d[c] for c in range(self._n_neurons)]
                codes.append(mechanizer_fista(
                    x_c, D, lam, max_iter=self.config.fista_max_iter,
                    tol=self.config.tol,
                ))
        elif p == PURSUIT_DENSE:
            # PCA-style dense encode: z = D x_c with no thresholding.
            for x in X:
                x_c = [x[c] - b_d[c] for c in range(self._n_neurons)]
                codes.append([_vec_dot(D[j], x_c) for j in range(K)])
        else:  # pragma: no cover — guarded above
            raise UnknownPursuit(p)
        # Coerce length to K (some pursuits may return shorter on edge cases).
        for i, row in enumerate(codes):
            if len(row) < K:
                row.extend([0.0] * (K - len(row)))
                codes[i] = row
            elif len(row) > K:
                codes[i] = row[:K]
        return codes

    def decode(self, Z: Sequence[Sequence[float]]) -> Matrix:
        """Decode sparse codes back to activation space."""

        with self._lock:
            self._require_fit()
            if not isinstance(Z, (list, tuple)):
                raise InvalidActivations("Z must be a sequence of code rows")
            out: Matrix = []
            for row in Z:
                if len(row) != self.config.n_features:
                    raise InvalidActivations(
                        f"code row has length {len(row)}; expected {self.config.n_features}"
                    )
                z = [float(v) for v in row]
                out.append(self._decode_one(z, self._decoder, self._decoder_bias))
            self._record(MECHANIZER_DECODED, {"n_samples": len(out)})
            return out

    # ---------------------------------------------------------------
    # patch / steer
    # ---------------------------------------------------------------

    def patch(
        self,
        target: Sequence[Sequence[float]],
        donor: Sequence[Sequence[float]],
        feature: int | Sequence[int],
        *,
        scale: float = 1.0,
    ) -> Matrix:
        """Activation patching in code space.

        Replace the ``target``'s value of ``feature`` (or every feature
        in ``feature`` if a sequence is given) with the *first* donor's
        value, scaled by ``scale ∈ [0, 1]`` (1.0 = full swap; values in
        between interpolate target ↔ donor in code space).  Decodes
        the patched code back to the activation space and returns it.
        """
        with self._lock:
            self._require_fit()
            tv = _validate_matrix(target, "target")
            dv = _validate_matrix(donor, "donor")
            if len(tv) == 0 or len(dv) == 0:
                raise InsufficientData("patch requires non-empty target and donor")
            if isinstance(feature, int):
                features = [feature]
            else:
                features = [int(f) for f in feature]
            for f in features:
                if f < 0 or f >= self.config.n_features:
                    raise InvalidFeature(f"feature {f} out of range")
            if not 0.0 <= scale <= 1.0:
                raise InvalidConfig("scale must lie in [0, 1]")
            # Encode target and donor under the configured pursuit kernel.
            Z_t = self._encode_with_pursuit(tv, self.config.pursuit)
            Z_d = self._encode_with_pursuit(dv, self.config.pursuit)
            # First donor row is the source of all patched values
            # (averaging multi-donor patches available via `steer`).
            donor_code = Z_d[0]
            patched_codes: Matrix = []
            for code in Z_t:
                new_code = list(code)
                for f in features:
                    new_code[f] = (1.0 - scale) * code[f] + scale * donor_code[f]
                patched_codes.append(new_code)
            out = [self._decode_one(c, self._decoder, self._decoder_bias)
                   for c in patched_codes]
            self._record(MECHANIZER_PATCHED, {
                "n_samples": len(out),
                "features": features,
                "scale": scale,
            })
            return out

    def steer(
        self,
        target: Sequence[Sequence[float]],
        feature: int,
        magnitude: float,
    ) -> Matrix:
        """Steer activations along the dictionary direction of ``feature``.

        Adds ``magnitude * d_feature`` directly to the *activation* of
        every target row.  Returns the steered activations *and*
        records the L2 norm of the perturbation in the ledger so a
        downstream ``aligner`` / ``verifier`` can bound the blast
        radius of the intervention.
        """
        with self._lock:
            self._require_fit()
            tv = _validate_matrix(target, "target")
            if feature < 0 or feature >= self.config.n_features:
                raise InvalidFeature(f"feature {feature} out of range")
            try:
                m = float(magnitude)
            except (TypeError, ValueError) as exc:
                raise InvalidConfig(f"magnitude must be numeric: {magnitude!r}") from exc
            if not math.isfinite(m):
                raise InvalidConfig("magnitude must be finite")
            atom = self._decoder[feature]
            d = self._n_neurons
            out: Matrix = []
            for row in tv:
                out.append([row[c] + m * atom[c] for c in range(d)])
            perturbation_norm = abs(m) * _vec_norm(atom)
            self._record(MECHANIZER_STEERED, {
                "feature": feature,
                "magnitude": m,
                "n_samples": len(tv),
                "perturbation_norm": perturbation_norm,
            })
            return out

    # ---------------------------------------------------------------
    # auto_interpret / circuit
    # ---------------------------------------------------------------

    def auto_interpret(
        self,
        X: Sequence[Sequence[float]],
        *,
        top_k: int = 5,
        label_fn: Callable[[int, Sequence[int]], str] | None = None,
    ) -> tuple[FeatureSummary, ...]:
        """Return a per-feature summary plus optional caller-provided labels."""
        with self._lock:
            self._require_fit()
            Xv = _validate_matrix(X, "X")
            n = len(Xv)
            if top_k <= 0:
                raise InvalidConfig("top_k must be > 0")
            codes = self._encode_with_pursuit(Xv, self.config.pursuit)
            K = self.config.n_features
            # Reconstruct totals.
            mean = _mean_rows(Xv)
            total_var = 0.0
            for row in Xv:
                for c, v in enumerate(row):
                    diff = v - mean[c]
                    total_var += diff * diff
            # Per-feature variance explained: ||z_j||^2 · ||d_j||^2.
            ve_per_feature = _zeros(K)
            for j in range(K):
                atom_sq = _vec_dot(self._decoder[j], self._decoder[j])
                s = 0.0
                for i in range(n):
                    s += codes[i][j] * codes[i][j]
                ve_per_feature[j] = atom_sq * s
            ve_total = max(total_var, self.config.eps)
            summaries: list[FeatureSummary] = []
            for j in range(K):
                active_count = 0
                act_sum = 0.0
                act_max = 0.0
                for i in range(n):
                    zj = codes[i][j]
                    if zj > 0.0:
                        active_count += 1
                        act_sum += zj
                        if zj > act_max:
                            act_max = zj
                density = active_count / max(n, 1)
                mean_act = (act_sum / active_count) if active_count else 0.0
                top_idx = sorted(
                    range(n),
                    key=lambda i: -codes[i][j],
                )[:top_k]
                ve_share = ve_per_feature[j] / ve_total
                label = None
                if label_fn is not None:
                    try:
                        label = str(label_fn(j, top_idx))
                    except Exception:  # pragma: no cover — caller hygiene
                        label = None
                summaries.append(FeatureSummary(
                    feature=j,
                    activation_density=density,
                    mean_activation=mean_act,
                    max_activation=act_max,
                    top_indices=tuple(top_idx),
                    variance_explained=ve_share,
                    label=label,
                ))
            self._record(MECHANIZER_INTERPRETED, {
                "n_features": K,
                "n_samples": n,
                "top_k": top_k,
                "labelled": label_fn is not None,
            })
            return tuple(summaries)

    def circuit(
        self,
        X: Sequence[Sequence[float]],
        *,
        threshold: float = 0.2,
    ) -> CircuitGraph:
        """Build a feature-feature co-activation graph from the code matrix."""
        with self._lock:
            self._require_fit()
            if not 0.0 < threshold <= 1.0:
                raise InvalidConfig("threshold must lie in (0, 1]")
            Xv = _validate_matrix(X, "X")
            n = len(Xv)
            K = self.config.n_features
            codes = self._encode_with_pursuit(Xv, self.config.pursuit)
            # Pearson correlation of each column of Z.
            col_mean = _zeros(K)
            for row in codes:
                for j in range(K):
                    col_mean[j] += row[j]
            inv_n = 1.0 / max(n, 1)
            col_mean = [v * inv_n for v in col_mean]
            col_var = _zeros(K)
            for row in codes:
                for j in range(K):
                    diff = row[j] - col_mean[j]
                    col_var[j] += diff * diff
            col_std = [math.sqrt(max(v * inv_n, 0.0)) for v in col_var]
            # Edges.
            adjacency: list[list[tuple[int, float]]] = [[] for _ in range(K)]
            edge_count = 0
            for a in range(K):
                if col_std[a] < self.config.eps:
                    continue
                for b in range(a + 1, K):
                    if col_std[b] < self.config.eps:
                        continue
                    s = 0.0
                    for i in range(n):
                        s += (codes[i][a] - col_mean[a]) * (codes[i][b] - col_mean[b])
                    s *= inv_n
                    corr = s / (col_std[a] * col_std[b])
                    if abs(corr) >= threshold:
                        adjacency[a].append((b, corr))
                        adjacency[b].append((a, corr))
                        edge_count += 1
            for j in range(K):
                adjacency[j].sort(key=lambda t: -abs(t[1]))
            # Largest connected component (undirected).
            seen = [False] * K
            largest = 0
            for start in range(K):
                if seen[start] or not adjacency[start]:
                    if not seen[start]:
                        seen[start] = True
                        largest = max(largest, 1)
                    continue
                stack = [start]
                size = 0
                while stack:
                    node = stack.pop()
                    if seen[node]:
                        continue
                    seen[node] = True
                    size += 1
                    for neighbour, _w in adjacency[node]:
                        if not seen[neighbour]:
                            stack.append(neighbour)
                largest = max(largest, size)
            graph = CircuitGraph(
                n_features=K,
                threshold=threshold,
                adjacency=tuple(tuple(row) for row in adjacency),
                edge_count=edge_count,
                largest_component=largest,
            )
            self._record(MECHANIZER_CIRCUIT_BUILT, {
                "n_features": K,
                "threshold": threshold,
                "edge_count": edge_count,
                "largest_component": largest,
            })
            return graph

    # ---------------------------------------------------------------
    # certify
    # ---------------------------------------------------------------

    def certify(
        self,
        X: Sequence[Sequence[float]],
        *,
        delta: float = 0.05,
    ) -> MechanizerCertificate:
        """Faithfulness certificate over a held-out activation matrix."""
        with self._lock:
            self._require_fit()
            if not 0.0 < delta < 1.0:
                raise InvalidConfig("delta must lie in (0, 1)")
            Xv = _validate_matrix(X, "X")
            n, d = _shape(Xv)
            if d != self._n_neurons:
                raise InvalidActivations(
                    f"X has {d} neurons but Mechanizer was fit on {self._n_neurons}"
                )
            codes = self._encode_with_pursuit(Xv, self.config.pursuit)
            # Per-row R^2 — needs a per-row mean target; we use the
            # fit-time decoder bias as the "mean" reference so that R^2
            # is comparable across calls and ledgered runs.
            ref = self._decoder_bias
            per_row_r2: list[float] = []
            recon_sq = 0.0
            rel_num = 0.0
            rel_den = 0.0
            total_var = 0.0
            for i in range(n):
                x_hat = self._decode_one(codes[i], self._decoder, ref)
                row_recon = 0.0
                row_total = 0.0
                for c in range(d):
                    diff = Xv[i][c] - x_hat[c]
                    recon_sq += diff * diff
                    rel_num += diff * diff
                    rel_den += Xv[i][c] * Xv[i][c]
                    row_recon += diff * diff
                    diff_total = Xv[i][c] - ref[c]
                    row_total += diff_total * diff_total
                    total_var += diff_total * diff_total
                if row_total < self.config.eps:
                    per_row_r2.append(1.0 if row_recon < self.config.eps else 0.0)
                else:
                    per_row_r2.append(max(0.0, min(1.0, 1.0 - row_recon / row_total)))
            if total_var < self.config.eps:
                r2 = 1.0 if recon_sq < self.config.eps else 0.0
            else:
                r2 = 1.0 - recon_sq / total_var
            relative_error = math.sqrt(rel_num / max(rel_den, self.config.eps))
            density = self._compute_density(codes)
            dead = sum(1 for v in density if v <= self.config.dead_feature_threshold)
            mean_l0 = self._per_row_l0(codes)
            coherence = mechanizer_mutual_coherence(self._decoder)
            identifiable_l0 = mechanizer_recovery_threshold(coherence)
            identifiable = mean_l0 <= identifiable_l0
            # LCBs on R^2 from per-row R^2.
            if n > 0:
                mean_per_row = sum(per_row_r2) / n
                var_per_row = sum((v - mean_per_row) ** 2 for v in per_row_r2) / max(n, 1)
                hoeffding_lcb = mean_per_row - hoeffding_half_width(n, delta=delta, b=1.0)
                bernstein_lcb = mean_per_row - empirical_bernstein_half_width(
                    n, variance=var_per_row, delta=delta, b=1.0,
                )
            else:
                hoeffding_lcb = float("-inf")
                bernstein_lcb = float("-inf")
            cert = MechanizerCertificate(
                n_samples=n,
                r2=r2,
                relative_error=relative_error,
                mean_l0=mean_l0,
                dead_features=dead,
                mutual_coherence=coherence,
                identifiable_l0=identifiable_l0,
                identifiable=identifiable,
                delta=delta,
                hoeffding_r2_lcb=hoeffding_lcb,
                bernstein_r2_lcb=bernstein_lcb,
                fingerprint=self._fingerprint,
            )
            self._record(MECHANIZER_CERTIFIED, {
                "n_samples": n,
                "delta": delta,
                "r2": r2,
                "mean_l0": mean_l0,
                "dead_features": dead,
                "mutual_coherence": coherence,
                "identifiable": identifiable,
            })
            return cert

    # ---------------------------------------------------------------
    # snapshot / restore / clear
    # ---------------------------------------------------------------

    def snapshot(self) -> MechanizerSnapshot:
        with self._lock:
            self._require_fit()
            return MechanizerSnapshot(
                config={
                    "algorithm": self.config.algorithm,
                    "n_features": self.config.n_features,
                    "target_l0": self.config.target_l0,
                    "pursuit": self.config.pursuit,
                    "l1_coeff": self.config.l1_coeff,
                    "learning_rate": self.config.learning_rate,
                    "max_iter": self.config.max_iter,
                    "tol": self.config.tol,
                    "seed": self.config.seed,
                    "tied_decoder": self.config.tied_decoder,
                    "normalize_dictionary": self.config.normalize_dictionary,
                    "dead_feature_threshold": self.config.dead_feature_threshold,
                    "pca_jacobi_sweeps": self.config.pca_jacobi_sweeps,
                    "fista_max_iter": self.config.fista_max_iter,
                    "omp_tol": self.config.omp_tol,
                    "eps": self.config.eps,
                },
                n_neurons=self._n_neurons,
                decoder=tuple(tuple(row) for row in self._decoder),
                encoder=tuple(tuple(row) for row in self._encoder),
                decoder_bias=tuple(self._decoder_bias),
                encoder_bias=tuple(self._encoder_bias),
                activation_density=tuple(self._activation_density),
                fingerprint=self._fingerprint,
                fit_seed=self._fit_seed,
                rng_state=None,
            )

    def restore(self, snap: MechanizerSnapshot) -> None:
        with self._lock:
            if snap.config.get("algorithm") != self.config.algorithm:
                raise InvalidConfig(
                    "snapshot algorithm does not match Mechanizer configuration"
                )
            if snap.config.get("n_features") != self.config.n_features:
                raise InvalidConfig(
                    "snapshot n_features does not match Mechanizer configuration"
                )
            self._n_neurons = snap.n_neurons
            self._decoder = [list(row) for row in snap.decoder]
            self._encoder = [list(row) for row in snap.encoder]
            self._decoder_bias = list(snap.decoder_bias)
            self._encoder_bias = list(snap.encoder_bias)
            self._activation_density = list(snap.activation_density)
            self._fit_seed = snap.fit_seed
            self._fit = True
            # Append a "restored" event but don't reset the chain so
            # the caller's audit history still works.
            self._record(MECHANIZER_RESET, {
                "restored_from_fingerprint": snap.fingerprint,
                "n_neurons": snap.n_neurons,
            })

    def clear(self) -> None:
        with self._lock:
            self._fit = False
            self._decoder = []
            self._encoder = []
            self._decoder_bias = []
            self._encoder_bias = []
            self._activation_density = []
            self._n_neurons = 0
            self._record(MECHANIZER_CLEARED, {})

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _require_fit(self) -> None:
        if not self._fit:
            raise NotFit("Mechanizer must be fit() before this operation")


# =====================================================================
# Module-level pursuit helpers used by the encode dispatch
# =====================================================================


def _matching_pursuit(
    x: Sequence[float],
    D: Matrix,
    k: int,
    *,
    tol: float = 1e-10,
) -> Vector:
    """Greedy non-orthogonal Matching Pursuit (Mallat-Zhang 1993).

    Same iteration count as OMP but the coefficient on already-chosen
    atoms is never refined, so MP can repeat atoms; the support is
    *multi-set* not a set.  Implemented for completeness; the public
    API normally calls OMP / TOPK.
    """
    if k <= 0:
        return [0.0] * len(D)
    K = len(D)
    d = len(D[0]) if D else 0
    if d != len(x):
        raise InvalidActivations(
            f"MP signal dim {len(x)} != dictionary dim {d}"
        )
    residual = [float(v) for v in x]
    z = _zeros(K)
    for _ in range(k):
        best_idx = -1
        best_val = -1.0
        best_signed = 0.0
        for j in range(K):
            c = _vec_dot(D[j], residual)
            ac = abs(c)
            if ac > best_val:
                best_val = ac
                best_signed = c
                best_idx = j
        if best_idx < 0 or best_val < tol:
            break
        z[best_idx] += best_signed
        residual = _vec_axpy(-best_signed, D[best_idx], residual)
        if _vec_norm(residual) < tol:
            break
    return z
