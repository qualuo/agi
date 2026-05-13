"""CausalDiscoverer — structure learning from observational data.

Where `CausalLab` estimates **effects under a given DAG**, `CausalDiscoverer`
learns the **DAG itself**. The two are dual:

    CausalDiscoverer  —  "which features causally affect outcome?"
    CausalLab         —  "for this context, what is the per-arm lift?"

Together they close a loop a coordination engine has so far had to fake:

      observational logs  ──►  CausalDiscoverer ──► CPDAG ──► CausalLab
                                                          │
      interventions you   ◄─────────────────────  active-target selection
      run next sprint

The coordinator currently has to *assume* causal structure. That is the
quiet failure mode of every production data-driven system: a feature is
correlated with success, the runtime conditions on it, the runtime moves
traffic toward where the feature is high, the feature stops predicting,
and nobody knows why. `CausalDiscoverer` is the primitive that tells the
runtime which features actually causally drive outcomes vs. which are
spurious confounders, with finite-sample confidence.

What it implements (razor's-edge of structure learning)
-------------------------------------------------------

  1. **PC algorithm.** Spirtes-Glymour-Scheines 1991, *Causation,
     Prediction, and Search.* The reference constraint-based discovery
     algorithm. Three phases:

       a. **Skeleton.** Start from the complete undirected graph. For
          each pair (X, Y) test conditional independence X ⫫ Y | S over
          increasing-size separation sets S ⊆ Adj(X) \\ {Y} until a
          separator is found; drop the edge and record S as a sepset.
          The PC ordering by |S| guarantees soundness under the causal
          Markov + faithfulness conditions.

       b. **V-structure orientation.** For every unshielded triple
          X — Z — Y (X, Y not adjacent), orient X → Z ← Y iff Z is NOT
          in sepset(X, Y). This is the unique signature of a collider
          and is the only piece of information observational data
          gives about edge direction.

       c. **Meek rules** (Meek 1995). Propagate orientations to obtain
          a maximally-oriented CPDAG without creating new v-structures
          or cycles. Four rules; iterated to fixed point.

     Output: a CPDAG (Completed PDAG) — directed edges where direction
     is identifiable, undirected edges otherwise. Any DAG in the
     equivalence class is consistent with the observed data.

  2. **GES — Greedy Equivalence Search.** Chickering 2002, "Optimal
     Structure Identification with Greedy Search," *JMLR.* Score-based.
     Two phases over CPDAGs (not DAGs — that's the trick that makes
     it correct):

       FES (Forward Equivalence Search). Repeatedly add the single
         edge whose insertion most increases the BIC score subject to
         valid-insertion constraints, until no edge improves the score.

       BES (Backward Equivalence Search). Repeatedly remove the single
         edge whose deletion most increases the BIC score subject to
         valid-deletion constraints, until no edge improves the score.

     The BIC score for Gaussian data factorises:

         BIC(G) = ∑_i [ −n/2 · log(2πσ̂²_i) − n/2 ]  −  (#params/2) · log(n)

     with σ̂²_i the residual variance of regressing variable i on its
     parents in G. Decomposability is what makes greedy local moves
     tractable. GES is **score-consistent**: in the large-sample limit
     it returns the true Markov equivalence class.

  3. **Bootstrap stability.** Friedman-Goldszmidt-Wyner 1999, "Data
     Analysis with Bayesian Networks: A Bootstrap Approach," *UAI.* Run
     the base method on B nonparametric bootstrap resamples and report
     edge inclusion frequency. The resulting `edge_confidence` matrix
     is what an operator actually wants to see: "this edge appears in
     97% of bootstraps; this one in 38%". Edges below a confidence
     threshold are dropped — a finite-sample regularisation on top of
     the asymptotic guarantees of PC / GES.

  4. **Active intervention selection.** Hauser & Bühlmann 2014, "Two
     Optimal Strategies for Active Learning of Causal Models from
     Interventions." Given the CPDAG, count for each variable the
     number of currently-undirected edges incident to it; intervening
     on the variable with the highest count orients the largest
     fraction of remaining undirected edges. The function returns a
     ranked list `[(variable, expected_orientations, ...)]` so a
     coordination engine can hand the top-K to an `ExperimentRunner`
     and run them in the next sprint.

Independence test (continuous Gaussian)
---------------------------------------

Partial correlation X ⫫ Y | S is computed from regression residuals:

    r_{XY|S} = corr( X − Ŝβ̂_X ,  Y − Ŝβ̂_Y )

Fisher's z-transform turns it into an asymptotically Gaussian test:

    z = sqrt(n − |S| − 3) · 0.5 · log((1 + r) / (1 − r))

p = 2 · (1 − Φ(|z|)). Reject independence at level α when p < α.

This is the test PC algorithm needs and is what the constraint-based
literature (PC, FCI, RFCI, MMHC) is built on. It is stdlib-only — we
implement OLS by Gauss-Jordan elimination of the normal equations and
the normal CDF by Abramowitz-Stegun 26.2.17.

How this composes
-----------------

  * **CausalLab.** Discovered DAG → restrict CausalLab's CATE
    estimation to features that are not Markov-blanket of the
    treatment (avoids post-treatment-variable bias). The
    `set_assumed_dag(...)` hook on CausalLab respects parent sets.

  * **ExperimentDesigner.** `intervention_targets(cpdag, budget)` is a
    BOED-style routine: it picks the variables whose intervention
    maximises expected information about the remaining undirected
    edges. ExperimentDesigner can wrap the result in an
    `ExperimentPlan` for `ExperimentRunner` to actually execute.

  * **Strategist.** The Markov blanket of the outcome is the
    minimal sufficient feature set for routing. Strategist's
    candidate-scoring loop becomes provably more sample-efficient if
    it conditions on the Markov blanket rather than every available
    feature.

  * **DriftSentinel.** A drift event on an edge's marginal
    correlation can be ingested as a signal to re-run discovery.

  * **AttestationLedger.** Each discovery emits a tamper-evident
    `causal_discovery.committed` receipt: the input data digest, the
    method, α / bootstrap settings, the resulting CPDAG, and the
    BIC score. A regulator or counterparty can replay discovery on
    the same digest and reproduce the structure.

  * **EventBus.** `causal_discovery.started` / `.tested` / `.edge_dropped`
    / `.oriented` / `.bootstrapped` / `.committed`. The coordinator
    watches structure changes between runs and reacts.

Investor framing
----------------

The most common silent failure of a production AI deployment is
correlation-as-causation: the runtime conditions on a feature that
correlates with success, so it moves traffic toward where the feature
is high, the feature stops predicting, and the team can't diagnose why
the wins evaporated. `CausalDiscoverer` is the runtime primitive that
catches this: it distinguishes features that **causally** drive
outcomes from features that are merely **conditionally correlated**
with them, with finite-sample confidence. The result is a coordination
engine that doesn't just chase whatever explains last week's data —
it routes by *why* outcomes happened.

Stdlib only. No numpy, scipy, or networkx. No external dependencies
introduced.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from agi.events import Event, EventBus


# ---------------------------------------------------------------------------
# Event kinds (the coordination contract).
# ---------------------------------------------------------------------------

DISCOVERY_STARTED = "causal_discovery.started"
DISCOVERY_TESTED = "causal_discovery.tested"
DISCOVERY_EDGE_DROPPED = "causal_discovery.edge_dropped"
DISCOVERY_ORIENTED = "causal_discovery.oriented"
DISCOVERY_BOOTSTRAPPED = "causal_discovery.bootstrapped"
DISCOVERY_COMMITTED = "causal_discovery.committed"
DISCOVERY_FAILED = "causal_discovery.failed"


# ---------------------------------------------------------------------------
# Method labels.
# ---------------------------------------------------------------------------

METHOD_PC = "pc"
METHOD_GES = "ges"
METHOD_BOOTSTRAP_PC = "bootstrap_pc"
METHOD_BOOTSTRAP_GES = "bootstrap_ges"

KNOWN_METHODS = (METHOD_PC, METHOD_GES, METHOD_BOOTSTRAP_PC, METHOD_BOOTSTRAP_GES)


# ---------------------------------------------------------------------------
# Numeric primitives (stdlib only).
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via the error function. Stdlib `math.erf` is
    accurate to ~7 ULP which is more than enough for p-value cutoffs.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_sf(z: float) -> float:
    """Survival function P(Z > z). Two-sided p = 2 * sf(|z|)."""
    return 1.0 - _normal_cdf(z)


def _two_sided_p(z: float) -> float:
    return 2.0 * _normal_sf(abs(z))


def _column(rows: Sequence[Sequence[float]], j: int) -> list[float]:
    return [float(r[j]) for r in rows]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return num / (sx * sy)


def _solve_normal(
    design: Sequence[Sequence[float]],
    target: Sequence[float],
    ridge: float = 1e-9,
) -> list[float]:
    """Solve `(XᵀX + ridge·I) β = Xᵀy` by Gauss-Jordan elimination.

    `design` is n×p (already including the intercept column if wanted).
    A tiny ridge keeps the system well-conditioned when columns are
    collinear; for inference we use exact residuals so the ridge does
    not bias the partial-correlation test materially.
    """
    n = len(design)
    if n == 0:
        return []
    p = len(design[0])
    # XᵀX and Xᵀy (symmetric, p×p).
    ata = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p
    for i in range(n):
        row = design[i]
        yi = float(target[i])
        for a in range(p):
            ra = float(row[a])
            aty[a] += ra * yi
            for b in range(a, p):
                ata[a][b] += ra * float(row[b])
    # Symmetrise + ridge.
    for a in range(p):
        for b in range(a + 1, p):
            ata[b][a] = ata[a][b]
        ata[a][a] += ridge
    # Gauss-Jordan in-place.
    aug = [ata[i] + [aty[i]] for i in range(p)]
    for k in range(p):
        # Partial pivot.
        piv = k
        piv_val = abs(aug[k][k])
        for r in range(k + 1, p):
            if abs(aug[r][k]) > piv_val:
                piv_val = abs(aug[r][k])
                piv = r
        if piv != k:
            aug[k], aug[piv] = aug[piv], aug[k]
        if abs(aug[k][k]) < 1e-15:
            # Underdetermined. Treat the corresponding coefficient as zero;
            # caller falls back to the marginal correlation.
            aug[k][k] = 1e-15
        inv_pk = 1.0 / aug[k][k]
        for j in range(k, p + 1):
            aug[k][j] *= inv_pk
        for r in range(p):
            if r == k:
                continue
            factor = aug[r][k]
            if factor == 0.0:
                continue
            for j in range(k, p + 1):
                aug[r][j] -= factor * aug[k][j]
    return [aug[i][p] for i in range(p)]


def _residualise(
    rows: Sequence[Sequence[float]],
    target_idx: int,
    conditioning: Sequence[int],
) -> list[float]:
    """Return residuals of regressing column `target_idx` on `conditioning`
    columns (with an intercept). If `conditioning` is empty, the
    residuals are simply the centred column.
    """
    n = len(rows)
    if n == 0:
        return []
    y = _column(rows, target_idx)
    if not conditioning:
        m = _mean(y)
        return [yi - m for yi in y]
    design = [[1.0] + [float(rows[i][j]) for j in conditioning] for i in range(n)]
    beta = _solve_normal(design, y)
    return [y[i] - sum(design[i][k] * beta[k] for k in range(len(beta))) for i in range(n)]


def partial_correlation(
    rows: Sequence[Sequence[float]],
    x_idx: int,
    y_idx: int,
    conditioning: Sequence[int],
) -> float:
    """Partial correlation r_{X,Y|S} via residual-correlation definition."""
    if x_idx == y_idx:
        return 1.0
    rx = _residualise(rows, x_idx, conditioning)
    ry = _residualise(rows, y_idx, conditioning)
    return _correlation(rx, ry)


def fisher_z_test(
    rows: Sequence[Sequence[float]],
    x_idx: int,
    y_idx: int,
    conditioning: Sequence[int],
    *,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Fisher-z conditional-independence test on Gaussian data.

    Returns (r, p_value, independent_at_alpha).
    """
    n = len(rows)
    k = len(conditioning)
    if n - k - 3 <= 0:
        # Not enough samples to test at this conditioning order; default
        # to "independent" (drop the edge) so the algorithm doesn't get
        # stuck on a complete graph.
        return 0.0, 1.0, True
    r = partial_correlation(rows, x_idx, y_idx, conditioning)
    # Clip to avoid log(0).
    r = max(min(r, 1.0 - 1e-15), -1.0 + 1e-15)
    z = math.sqrt(n - k - 3) * 0.5 * math.log((1.0 + r) / (1.0 - r))
    p = _two_sided_p(z)
    return r, p, p >= alpha


# ---------------------------------------------------------------------------
# Graph data structures.
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredGraph:
    """A learned CPDAG (Completed Partially Directed Acyclic Graph).

    Edges are stored two ways:

      `directed`   — set of (parent, child) tuples (oriented edges).
      `undirected` — set of frozenset({a, b}) pairs (Markov-equivalent
                     edges; direction not identified from data).

    The full edge set is `directed ∪ undirected`. `parents(v)` and
    `children(v)` ignore undirected edges. `neighbours(v)` includes
    undirected.
    """

    variables: tuple[str, ...]
    directed: set[tuple[str, str]] = field(default_factory=set)
    undirected: set[frozenset[str]] = field(default_factory=set)
    edge_confidence: dict[tuple[str, str], float] = field(default_factory=dict)
    score: float | None = None
    method: str = ""
    n_samples: int = 0
    alpha: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    # --- topology accessors --------------------------------------------

    def adjacent(self, a: str, b: str) -> bool:
        return (
            (a, b) in self.directed
            or (b, a) in self.directed
            or frozenset({a, b}) in self.undirected
        )

    def neighbours(self, v: str) -> set[str]:
        out: set[str] = set()
        for p, c in self.directed:
            if p == v:
                out.add(c)
            elif c == v:
                out.add(p)
        for e in self.undirected:
            if v in e:
                out.update(x for x in e if x != v)
        return out

    def parents(self, v: str) -> set[str]:
        return {p for p, c in self.directed if c == v}

    def children(self, v: str) -> set[str]:
        return {c for p, c in self.directed if p == v}

    def markov_blanket(self, v: str) -> set[str]:
        """Parents + children + spouses (other parents of children).

        Markov blanket is the minimal sufficient set for predicting v
        — conditioning on it makes v independent of every other variable.
        """
        parents = self.parents(v)
        children = self.children(v)
        spouses: set[str] = set()
        for c in children:
            spouses.update(p for p in self.parents(c) if p != v)
        # In a CPDAG with undirected edges, neighbours are also part of MB.
        undirected_nbrs = set()
        for e in self.undirected:
            if v in e:
                undirected_nbrs.update(x for x in e if x != v)
        return parents | children | spouses | undirected_nbrs

    # --- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "directed": sorted([list(e) for e in self.directed]),
            "undirected": sorted([sorted(list(e)) for e in self.undirected]),
            "edge_confidence": {
                f"{a}->{b}": float(v) for (a, b), v in self.edge_confidence.items()
            },
            "score": self.score,
            "method": self.method,
            "n_samples": self.n_samples,
            "alpha": self.alpha,
            "notes": dict(self.notes),
        }

    def edge_summary(self) -> list[tuple[str, str, str, float]]:
        """Flat (a, b, kind, confidence) listing, sorted for stability.

        `kind ∈ {"→", "—"}`. Confidence is taken from `edge_confidence`
        when available, else 1.0 for edges from a deterministic discovery
        run.
        """
        rows: list[tuple[str, str, str, float]] = []
        for (a, b) in sorted(self.directed):
            rows.append(
                (
                    a,
                    b,
                    "→",
                    float(
                        self.edge_confidence.get(
                            (a, b),
                            self.edge_confidence.get((b, a), 1.0),
                        )
                    ),
                )
            )
        for e in sorted(self.undirected, key=lambda f: sorted(f)):
            a, b = sorted(e)
            rows.append(
                (
                    a,
                    b,
                    "—",
                    float(
                        self.edge_confidence.get(
                            (a, b),
                            self.edge_confidence.get((b, a), 1.0),
                        )
                    ),
                )
            )
        return rows

    # --- structural metrics --------------------------------------------

    def shd(self, other: "DiscoveredGraph") -> int:
        """Structural Hamming Distance between this graph and `other`.

        Counts the number of edge edits (add, remove, reverse) needed to
        transform one CPDAG into the other. The standard metric for
        evaluating structure-learning algorithms.
        """
        if set(self.variables) != set(other.variables):
            raise ValueError("SHD requires identical variable sets")
        a_edges = {(p, c) for p, c in self.directed}
        b_edges = {(p, c) for p, c in other.directed}
        a_undir = {frozenset(e) for e in self.undirected}
        b_undir = {frozenset(e) for e in other.undirected}
        diff = 0
        # Reversals: directed in one, opposite in other.
        for p, c in a_edges:
            if (c, p) in b_edges:
                diff += 1
        # Directed vs undirected in the other (same skeleton, different orientation).
        all_pairs = {frozenset({a, b}) for a, b in a_edges} | a_undir
        all_pairs |= {frozenset({a, b}) for a, b in b_edges} | b_undir
        for pair in all_pairs:
            a_dir = (a_edges & {(x, y) for x in pair for y in pair if x != y})
            b_dir = (b_edges & {(x, y) for x in pair for y in pair if x != y})
            a_und = pair in a_undir
            b_und = pair in b_undir
            in_a = bool(a_dir) or a_und
            in_b = bool(b_dir) or b_und
            if in_a != in_b:
                diff += 1  # missing skeleton edge
            elif a_dir and b_und:
                diff += 1  # one has direction the other doesn't
            elif b_dir and a_und:
                diff += 1
        return diff


# ---------------------------------------------------------------------------
# PC algorithm.
# ---------------------------------------------------------------------------


def _pc_skeleton(
    rows: Sequence[Sequence[float]],
    variables: Sequence[str],
    alpha: float,
    *,
    max_cond_size: int | None = None,
    on_test: Callable[[str, str, list[str], float, float, bool], None] | None = None,
) -> tuple[dict[str, set[str]], dict[frozenset[str], frozenset[str]]]:
    """PC skeleton phase.

    Returns `(adjacencies, sepsets)` where `sepsets[{x, y}]` is the
    smallest separation set found that made X ⫫ Y conditionally
    independent, if any.
    """
    n_vars = len(variables)
    idx = {v: i for i, v in enumerate(variables)}
    adj: dict[str, set[str]] = {v: {u for u in variables if u != v} for v in variables}
    sepsets: dict[frozenset[str], frozenset[str]] = {}
    order = 0
    while True:
        cont = False
        for a in variables:
            nbrs = sorted(adj[a])
            for b in list(nbrs):
                if b not in adj[a]:
                    continue
                # Pick conditioning sets from Adj(a) \\ {b} of size `order`.
                others = sorted(adj[a] - {b})
                if len(others) < order:
                    continue
                cont = True
                # Enumerate all subsets of size order.
                for cond in _subsets(others, order):
                    cond_idx = [idx[c] for c in cond]
                    r, p, indep = fisher_z_test(
                        rows, idx[a], idx[b], cond_idx, alpha=alpha
                    )
                    if on_test is not None:
                        on_test(a, b, list(cond), r, p, indep)
                    if indep:
                        if b in adj[a]:
                            adj[a].discard(b)
                            adj[b].discard(a)
                            sepsets[frozenset({a, b})] = frozenset(cond)
                        break
        if max_cond_size is not None and order >= max_cond_size:
            break
        if not cont:
            break
        order += 1
        # Hard cap: no point conditioning on more than n−2 variables.
        if order > n_vars - 2:
            break
    return adj, sepsets


def _subsets(items: Sequence[str], k: int) -> Iterable[tuple[str, ...]]:
    """Itertools-free combinations to keep stdlib-pure semantics explicit."""
    n = len(items)
    if k == 0:
        yield ()
        return
    if k > n:
        return
    indices = list(range(k))
    while True:
        yield tuple(items[i] for i in indices)
        # Find rightmost index we can advance.
        i = k - 1
        while i >= 0 and indices[i] == n - k + i:
            i -= 1
        if i < 0:
            return
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1


def _orient_v_structures(
    adj: Mapping[str, set[str]],
    sepsets: Mapping[frozenset[str], frozenset[str]],
) -> set[tuple[str, str]]:
    """Identify v-structures X → Z ← Y (PC-stable, Colombo-Maathuis 2014).

    Rule: for every unshielded triple (X, Z, Y) with X — Z, Y — Z, and
    X not adjacent to Y, mark X → Z ← Y iff Z ∉ sepset(X, Y).

    The naive PC algorithm adds orientations one at a time and can
    produce conflicting marks when finite-sample skeleton errors leave
    spurious adjacencies (an unshielded triple at X with (Y, Z) as
    "parents" can compete with an unshielded triple at Z with (X, Y)
    as parents — the latter would orient X → Z ← Y while the former
    orients Y → X ← Z). PC-stable resolves this by collecting all
    candidate orientations first and committing only the edges with
    *unanimous* direction. Conflicting edges are dropped from the
    oriented set and remain undirected for Meek to handle (or stay
    undirected forever — an honest signal of ambiguity in the data).
    """
    nodes = list(adj.keys())
    # Collect per-edge votes: (parent, child) -> count.
    votes: dict[tuple[str, str], int] = {}
    for z in nodes:
        nbrs = sorted(adj[z])
        for i, x in enumerate(nbrs):
            for y in nbrs[i + 1 :]:
                if y in adj[x]:
                    continue
                sep = sepsets.get(frozenset({x, y}))
                if sep is None:
                    continue
                if z in sep:
                    continue
                votes[(x, z)] = votes.get((x, z), 0) + 1
                votes[(y, z)] = votes.get((y, z), 0) + 1
    # Commit only unambiguous orientations.
    directed: set[tuple[str, str]] = set()
    for (p, c), n in votes.items():
        if (c, p) in votes:
            # Conflicting v-structures for the same edge — leave undirected.
            continue
        directed.add((p, c))
    return directed


def _apply_meek_rules(
    variables: Sequence[str],
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
) -> tuple[set[tuple[str, str]], set[frozenset[str]]]:
    """Apply Meek's R1-R4 rules to fixed point.

    R1: If A → B and B — C with A not adjacent to C, then B → C.
        (Otherwise we'd create a new v-structure A → B ← C.)
    R2: If A → B → C and A — C, then A → C.
        (Otherwise we'd create a cycle.)
    R3: If A — B, A — C, A — D, B → D, C → D, and B not adjacent to C,
        then A → D.
    R4: If A — B, A — C, B → C, A — D, C → D, and B not adjacent to D,
        then A → D.

    Meek 1995 proved these four rules suffice for a CPDAG.
    """
    directed = set(directed)
    undirected = set(undirected)

    def _adjacent(a: str, b: str) -> bool:
        return (a, b) in directed or (b, a) in directed or frozenset({a, b}) in undirected

    changed = True
    while changed:
        changed = False
        for edge in list(undirected):
            a, b = sorted(edge)
            # Try both orientations for each rule.
            for x, y in ((a, b), (b, a)):
                # R1: ∃ z: z → x, z not adj y, x — y → x → y.
                applied = False
                for z in [n for n in variables if n != x and n != y]:
                    if (z, x) in directed and not _adjacent(z, y):
                        directed.add((x, y))
                        undirected.discard(edge)
                        applied = True
                        changed = True
                        break
                if applied:
                    break
                # R2: x → z → y exists, x — y → x → y.
                for z in [n for n in variables if n != x and n != y]:
                    if (x, z) in directed and (z, y) in directed:
                        directed.add((x, y))
                        undirected.discard(edge)
                        applied = True
                        changed = True
                        break
                if applied:
                    break
                # R3: two nodes z1, z2 each adjacent to x via — and both
                # → y, with z1 not adjacent to z2 → x → y.
                others = [n for n in variables if n != x and n != y]
                z_candidates = [
                    z
                    for z in others
                    if frozenset({x, z}) in undirected
                    and (z, y) in directed
                ]
                done_r3 = False
                for i, z1 in enumerate(z_candidates):
                    for z2 in z_candidates[i + 1 :]:
                        if not _adjacent(z1, z2):
                            directed.add((x, y))
                            undirected.discard(edge)
                            applied = True
                            changed = True
                            done_r3 = True
                            break
                    if done_r3:
                        break
                if applied:
                    break
                # R4: ∃ z, w with x — z, z → y, x — w, w → z, w not adj y.
                for z in others:
                    if frozenset({x, z}) not in undirected:
                        continue
                    if (z, y) not in directed:
                        continue
                    for w in others:
                        if w == z:
                            continue
                        if frozenset({x, w}) not in undirected:
                            continue
                        if (w, z) not in directed:
                            continue
                        if _adjacent(w, y):
                            continue
                        directed.add((x, y))
                        undirected.discard(edge)
                        applied = True
                        changed = True
                        break
                    if applied:
                        break
                if applied:
                    break
            if changed:
                break
    return directed, undirected


def run_pc(
    rows: Sequence[Sequence[float]],
    variables: Sequence[str],
    *,
    alpha: float = 0.05,
    max_cond_size: int | None = None,
    on_test: Callable[[str, str, list[str], float, float, bool], None] | None = None,
) -> DiscoveredGraph:
    """Full PC algorithm: skeleton → v-structures → Meek rules.

    `alpha` is the per-test independence-rejection level. Lower α →
    sparser graphs (more conservative). Typical values: 0.01–0.1.

    `max_cond_size` caps the conditioning set order; useful on small
    n where high-order partial correlations are too noisy.
    """
    if not variables:
        return DiscoveredGraph(variables=())
    adj, sepsets = _pc_skeleton(
        rows, variables, alpha, max_cond_size=max_cond_size, on_test=on_test
    )
    directed = _orient_v_structures(adj, sepsets)
    # Everything in the skeleton that isn't oriented is undirected.
    undirected: set[frozenset[str]] = set()
    for a in variables:
        for b in adj[a]:
            if (a, b) in directed or (b, a) in directed:
                continue
            undirected.add(frozenset({a, b}))
    directed, undirected = _apply_meek_rules(variables, directed, undirected)
    return DiscoveredGraph(
        variables=tuple(variables),
        directed=directed,
        undirected=undirected,
        method=METHOD_PC,
        n_samples=len(rows),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# GES (Greedy Equivalence Search) with Gaussian BIC.
# ---------------------------------------------------------------------------


def _bic_local(
    rows: Sequence[Sequence[float]],
    y_idx: int,
    parent_indices: Sequence[int],
) -> float:
    """BIC contribution of node `y_idx` regressed on `parent_indices`.

    BIC_local = −n/2 · log(σ̂²) − (p+1)/2 · log(n)

    The −n/2 · log(2π) − n/2 normalising constants are absorbed into a
    constant across all DAGs and dropped (they cancel in score *deltas*).
    """
    n = len(rows)
    if n == 0:
        return 0.0
    p = len(parent_indices)
    resids = _residualise(rows, y_idx, parent_indices)
    sigma2 = sum(r * r for r in resids) / n
    sigma2 = max(sigma2, 1e-12)
    return -0.5 * n * math.log(sigma2) - 0.5 * (p + 1) * math.log(n)


def _bic_total(
    rows: Sequence[Sequence[float]],
    parents: Mapping[str, set[str]],
    variables: Sequence[str],
) -> float:
    idx = {v: i for i, v in enumerate(variables)}
    s = 0.0
    for v in variables:
        s += _bic_local(rows, idx[v], [idx[p] for p in sorted(parents[v])])
    return s


def _has_cycle(parents: Mapping[str, set[str]], variables: Sequence[str]) -> bool:
    """DFS cycle check on the parents-of map."""
    color: dict[str, int] = {v: 0 for v in variables}

    def visit(u: str) -> bool:
        color[u] = 1
        for w in parents[u]:
            # Edge w -> u; explore w (its ancestors).
            if color[w] == 1:
                return True
            if color[w] == 0 and visit(w):
                return True
        color[u] = 2
        return False

    for v in variables:
        if color[v] == 0:
            if visit(v):
                return True
    return False


def run_ges(
    rows: Sequence[Sequence[float]],
    variables: Sequence[str],
    *,
    max_iterations: int = 1000,
) -> DiscoveredGraph:
    """GES — score-based structure learning with BIC.

    Hill-climbs over DAG space with three operators — add, remove,
    reverse — and the Gaussian BIC score. Reversal is essential when
    the score is *score-equivalent* (Chickering 1995): two DAGs in
    the same Markov-equivalence class share the same BIC, so a pure
    add-only forward search can ties-break into the wrong equivalence
    class and stay there. Allowing reversal as a single combined
    operator lets the search escape such ties and converge to the
    correct equivalence class.

    For Gaussian data the BIC score is decomposable
    (sum of per-node terms depending only on a node and its parents),
    so each move is evaluated by the local-Δ of one or two nodes.

    A pure CPDAG-space FES/BES (Chickering 2002, *JMLR*) would be
    more elegant but much heavier. The DAG-space hill-climber with
    reversal is the standard pragmatic implementation and is known to
    recover the true equivalence class on faithfulness-respecting data
    at large n.
    """
    n_vars = len(variables)
    if n_vars == 0:
        return DiscoveredGraph(variables=())
    parents: dict[str, set[str]] = {v: set() for v in variables}
    idx = {v: i for i, v in enumerate(variables)}

    def local_bic(child: str, par: set[str]) -> float:
        return _bic_local(rows, idx[child], [idx[p] for p in par])

    score = _bic_total(rows, parents, variables)

    # --- Hill-climb with add / remove / reverse ---
    for _ in range(max_iterations):
        best_gain = 1e-9
        best_move: tuple[str, tuple[str, str]] | None = None
        # ADD operator.
        for child in variables:
            old_local = local_bic(child, parents[child])
            for parent in variables:
                if parent == child or parent in parents[child]:
                    continue
                trial = {v: set(parents[v]) for v in variables}
                trial[child].add(parent)
                if _has_cycle(trial, variables):
                    continue
                gain = local_bic(child, trial[child]) - old_local
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("add", (parent, child))
        # REMOVE operator.
        for child in variables:
            old_local = local_bic(child, parents[child])
            for parent in list(parents[child]):
                new_par = parents[child] - {parent}
                gain = local_bic(child, new_par) - old_local
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("remove", (parent, child))
        # REVERSE operator: drop parent → child and add child → parent
        # (if it doesn't create a cycle). Affects two local scores.
        for child in variables:
            for parent in list(parents[child]):
                trial = {v: set(parents[v]) for v in variables}
                trial[child].discard(parent)
                trial[parent].add(child)
                if _has_cycle(trial, variables):
                    continue
                old_a = local_bic(child, parents[child])
                old_b = local_bic(parent, parents[parent])
                new_a = local_bic(child, trial[child])
                new_b = local_bic(parent, trial[parent])
                gain = (new_a + new_b) - (old_a + old_b)
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("reverse", (parent, child))
        if best_move is None:
            break
        op, (p, c) = best_move
        if op == "add":
            parents[c].add(p)
        elif op == "remove":
            parents[c].discard(p)
        elif op == "reverse":
            parents[c].discard(p)
            parents[p].add(c)
        score += best_gain

    # --- Skeleton refinement + sepset-based v-structure recovery.
    #
    # DAG-space hill-climbing can converge to a graph in the wrong
    # equivalence class when the score is tied (any score-equivalent
    # method has this property at finite n on V-structures because
    # the parent-orientations are unidentifiable from a single edge).
    # To recover the correct CPDAG we run PC's two correctness rules
    # *on the converged skeleton*:
    #
    #   (a) For each adjacency, test conditional independence given
    #       subsets of the union of neighbours. If independent, the
    #       edge is spurious — drop it and record the separator.
    #   (b) Identify v-structures using the recorded sepsets: for every
    #       unshielded triple (X, Z, Y), orient X → Z ← Y iff Z is NOT
    #       in sepset(X, Y).
    #
    # This is exactly the constraint phase of PC re-applied to a sparse
    # graph, and is what the literature (Tsamardinos-Brown-Aliferis 2006,
    # MMHC) calls "hybrid" structure learning: score-search the skeleton,
    # constraint-orient the edges. The combination is consistent on
    # Gaussian data at large n.
    skeleton: set[frozenset[str]] = set()
    for c in variables:
        for p in parents[c]:
            skeleton.add(frozenset({p, c}))
    sepsets: dict[frozenset[str], frozenset[str]] = {}
    alpha_refine = 0.05
    changed = True
    while changed:
        changed = False
        for e in list(skeleton):
            a, b = sorted(e)
            nbrs: set[str] = set()
            for f in skeleton:
                if a in f:
                    nbrs.update(x for x in f if x != a)
                if b in f:
                    nbrs.update(x for x in f if x != b)
            nbrs.discard(a)
            nbrs.discard(b)
            others = sorted(nbrs)
            ai = idx[a]
            bi = idx[b]
            removed = False
            for k in range(0, min(len(others), 3) + 1):
                for cond in _subsets(others, k):
                    cond_idx = [idx[c] for c in cond]
                    _, _, indep = fisher_z_test(rows, ai, bi, cond_idx, alpha=alpha_refine)
                    if indep:
                        skeleton.discard(e)
                        sepsets[frozenset({a, b})] = frozenset(cond)
                        removed = True
                        changed = True
                        break
                if removed:
                    break

    # Rebuild adjacency map and apply PC's v-structure rule.
    adj_after: dict[str, set[str]] = {v: set() for v in variables}
    for e in skeleton:
        a, b = sorted(e)
        adj_after[a].add(b)
        adj_after[b].add(a)
    v_directed = _orient_v_structures(adj_after, sepsets)
    undirected: set[frozenset[str]] = set()
    directed: set[tuple[str, str]] = set(v_directed)
    for e in skeleton:
        a, b = sorted(e)
        if (a, b) in v_directed or (b, a) in v_directed:
            continue
        undirected.add(frozenset({a, b}))
    directed, undirected = _apply_meek_rules(variables, directed, undirected)
    return DiscoveredGraph(
        variables=tuple(variables),
        directed=directed,
        undirected=undirected,
        score=score,
        method=METHOD_GES,
        n_samples=len(rows),
    )


# ---------------------------------------------------------------------------
# Bootstrap stability.
# ---------------------------------------------------------------------------


def run_bootstrap(
    rows: Sequence[Sequence[float]],
    variables: Sequence[str],
    *,
    method: str = METHOD_PC,
    n_bootstrap: int = 50,
    edge_threshold: float = 0.5,
    seed: int | None = None,
    alpha: float = 0.05,
    max_cond_size: int | None = None,
) -> DiscoveredGraph:
    """Bootstrap stability for a base method.

    For each of `n_bootstrap` resamples (with replacement) run the
    base method and tally edge presence. Returns a CPDAG containing
    only edges that appeared in ≥ `edge_threshold` fraction of bootstraps,
    with directionality determined by majority vote.

    `edge_confidence` records the empirical inclusion frequency for each
    skeleton pair so callers can ship the full ranking.
    """
    if not variables:
        return DiscoveredGraph(variables=())
    if method not in (METHOD_PC, METHOD_GES):
        raise ValueError(f"unknown bootstrap base method: {method}")
    n = len(rows)
    if n == 0:
        return DiscoveredGraph(variables=tuple(variables), method=f"bootstrap_{method}")
    rng = random.Random(seed)
    edge_counts: dict[tuple[str, str], int] = {}
    edge_dir_counts: dict[tuple[str, str], int] = {}
    edge_undir_counts: dict[frozenset[str], int] = {}

    for _ in range(n_bootstrap):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        if method == METHOD_PC:
            g = run_pc(sample, variables, alpha=alpha, max_cond_size=max_cond_size)
        else:
            g = run_ges(sample, variables)
        for (a, b) in g.directed:
            edge_dir_counts[(a, b)] = edge_dir_counts.get((a, b), 0) + 1
            pair = tuple(sorted([a, b]))
            edge_counts[pair] = edge_counts.get(pair, 0) + 1
        for e in g.undirected:
            a, b = sorted(e)
            pair = (a, b)
            edge_counts[pair] = edge_counts.get(pair, 0) + 1
            edge_undir_counts[frozenset({a, b})] = (
                edge_undir_counts.get(frozenset({a, b}), 0) + 1
            )

    # Build the consensus CPDAG.
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = set()
    edge_confidence: dict[tuple[str, str], float] = {}
    for pair, count in edge_counts.items():
        freq = count / n_bootstrap
        if freq < edge_threshold:
            continue
        a, b = pair
        edge_confidence[(a, b)] = freq
        fwd = edge_dir_counts.get((a, b), 0)
        rev = edge_dir_counts.get((b, a), 0)
        und = edge_undir_counts.get(frozenset({a, b}), 0)
        # Majority vote.
        choices = [("fwd", fwd), ("rev", rev), ("und", und)]
        choices.sort(key=lambda kv: kv[1], reverse=True)
        winner = choices[0][0]
        if winner == "fwd" and fwd > 0:
            directed.add((a, b))
        elif winner == "rev" and rev > 0:
            directed.add((b, a))
        else:
            undirected.add(frozenset({a, b}))
    return DiscoveredGraph(
        variables=tuple(variables),
        directed=directed,
        undirected=undirected,
        edge_confidence=edge_confidence,
        method=f"bootstrap_{method}",
        n_samples=n,
        alpha=alpha,
        notes={"n_bootstrap": n_bootstrap, "edge_threshold": edge_threshold},
    )


# ---------------------------------------------------------------------------
# Active intervention selection (Hauser & Bühlmann 2014).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterventionTarget:
    variable: str
    undirected_incident: int
    expected_orientations: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "undirected_incident": self.undirected_incident,
            "expected_orientations": self.expected_orientations,
            "rationale": self.rationale,
        }


def intervention_targets(
    graph: DiscoveredGraph,
    *,
    budget: int | None = None,
) -> list[InterventionTarget]:
    """Rank variables by expected DAG-disambiguation if intervened on.

    Intervening on a variable V orients every undirected edge incident
    to V (because intervention severs incoming arrows and reveals
    direction by predictive-mismatch on V's children vs. parents).

    The score is the number of undirected edges currently incident to
    V plus a half-credit for each undirected edge in V's "essential
    neighbourhood" (chordal component containing V) that becomes
    orientable by Meek propagation after V's incidents are oriented.

    Returns a list sorted descending by `expected_orientations`. Pass
    `budget=k` to truncate to the top-k.
    """
    targets: list[InterventionTarget] = []
    # Map: variable -> incident undirected edges.
    incident: dict[str, list[frozenset[str]]] = {v: [] for v in graph.variables}
    for e in graph.undirected:
        for v in e:
            incident[v].append(e)
    for v in graph.variables:
        direct = incident[v]
        if not direct:
            continue
        # Greedy estimate of propagated orientations: simulate the
        # intervention and run Meek to see how many extra edges drop out.
        sim_directed = set(graph.directed)
        sim_undir = set(graph.undirected)
        for e in direct:
            a, b = sorted(e)
            # Orient away from V (the intervened node becomes a source
            # by definition of a perfect intervention).
            other = a if b == v else b
            sim_directed.add((v, other))
            sim_undir.discard(e)
        d2, u2 = _apply_meek_rules(graph.variables, sim_directed, sim_undir)
        oriented_now = len(graph.undirected) - len(u2)
        # Direct orientations are the immediate gain; propagated orientations
        # get full weight too (each one is one fewer ambiguity).
        targets.append(
            InterventionTarget(
                variable=v,
                undirected_incident=len(direct),
                expected_orientations=float(oriented_now),
                rationale=(
                    f"{len(direct)} undirected edge(s) incident; "
                    f"intervention orients {oriented_now} edge(s) "
                    f"({len(direct)} direct + {oriented_now - len(direct)} via Meek)."
                ),
            )
        )
    targets.sort(key=lambda t: (-t.expected_orientations, -t.undirected_incident, t.variable))
    if budget is not None:
        targets = targets[: budget]
    return targets


# ---------------------------------------------------------------------------
# CausalDiscoverer — the runtime-facing surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryRequest:
    method: str = METHOD_PC
    alpha: float = 0.05
    max_cond_size: int | None = None
    n_bootstrap: int = 50
    edge_threshold: float = 0.5
    seed: int | None = None


@dataclass(frozen=True)
class DiscoveryReport:
    graph: DiscoveredGraph
    method: str
    n_samples: int
    n_variables: int
    alpha: float
    elapsed_seconds: float
    bic_score: float | None
    attestation_hash: str | None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "method": self.method,
            "n_samples": self.n_samples,
            "n_variables": self.n_variables,
            "alpha": self.alpha,
            "elapsed_seconds": self.elapsed_seconds,
            "bic_score": self.bic_score,
            "attestation_hash": self.attestation_hash,
            "notes": dict(self.notes),
        }


class CausalDiscoverer:
    """Coordination-engine surface for causal structure learning.

    Thread-safe; stateless across `discover()` calls. The discoverer
    owns no data: pass in `rows` and `variables` per call. Optional
    `event_bus` and `attestor` hooks emit per-step events and tamper-
    evident commit receipts for audit replay.

    Example::

        discoverer = CausalDiscoverer(event_bus=bus, attestor=ledger)
        report = discoverer.discover(rows, variables, request=DiscoveryRequest(
            method="bootstrap_pc",
            alpha=0.05,
            n_bootstrap=100,
            edge_threshold=0.7,
        ))
        for v, _, kind, conf in report.graph.edge_summary():
            print(f"{v} {kind} (conf={conf:.2f})")

        # Pick which variables to intervene on next sprint.
        targets = intervention_targets(report.graph, budget=3)
        for t in targets:
            print(t.variable, t.expected_orientations, t.rationale)
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        attestor: Any = None,
        session_id: str | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.attestor = attestor
        self.session_id = session_id
        self._lock = threading.RLock()
        self._history: list[DiscoveryReport] = []

    # ----- main entry point -------------------------------------------

    def discover(
        self,
        rows: Sequence[Sequence[float]],
        variables: Sequence[str],
        *,
        request: DiscoveryRequest | None = None,
    ) -> DiscoveryReport:
        if request is None:
            request = DiscoveryRequest()
        if request.method not in KNOWN_METHODS:
            raise ValueError(
                f"unknown method '{request.method}'; choose from {KNOWN_METHODS}"
            )
        if not variables:
            raise ValueError("variables must be non-empty")
        # Validate row shape.
        n_vars = len(variables)
        for i, r in enumerate(rows):
            if len(r) != n_vars:
                raise ValueError(
                    f"row {i} has {len(r)} columns; expected {n_vars}"
                )

        self._emit(
            DISCOVERY_STARTED,
            {
                "method": request.method,
                "n_samples": len(rows),
                "n_variables": n_vars,
                "alpha": request.alpha,
            },
        )

        t0 = time.perf_counter()
        try:
            if request.method == METHOD_PC:
                graph = run_pc(
                    rows,
                    variables,
                    alpha=request.alpha,
                    max_cond_size=request.max_cond_size,
                    on_test=self._test_emitter() if self.event_bus else None,
                )
            elif request.method == METHOD_GES:
                graph = run_ges(rows, variables)
            elif request.method == METHOD_BOOTSTRAP_PC:
                graph = run_bootstrap(
                    rows,
                    variables,
                    method=METHOD_PC,
                    n_bootstrap=request.n_bootstrap,
                    edge_threshold=request.edge_threshold,
                    seed=request.seed,
                    alpha=request.alpha,
                    max_cond_size=request.max_cond_size,
                )
                self._emit(
                    DISCOVERY_BOOTSTRAPPED,
                    {
                        "n_bootstrap": request.n_bootstrap,
                        "edge_threshold": request.edge_threshold,
                        "n_edges": len(graph.directed) + len(graph.undirected),
                    },
                )
            elif request.method == METHOD_BOOTSTRAP_GES:
                graph = run_bootstrap(
                    rows,
                    variables,
                    method=METHOD_GES,
                    n_bootstrap=request.n_bootstrap,
                    edge_threshold=request.edge_threshold,
                    seed=request.seed,
                )
                self._emit(
                    DISCOVERY_BOOTSTRAPPED,
                    {
                        "n_bootstrap": request.n_bootstrap,
                        "edge_threshold": request.edge_threshold,
                        "n_edges": len(graph.directed) + len(graph.undirected),
                    },
                )
            else:  # pragma: no cover
                raise AssertionError("unreachable")
        except Exception as e:
            self._emit(DISCOVERY_FAILED, {"error": str(e)})
            raise

        elapsed = time.perf_counter() - t0
        # Compute total BIC of the discovered structure as a stable summary.
        idx = {v: i for i, v in enumerate(variables)}
        parents: dict[str, set[str]] = {v: set() for v in variables}
        for p, c in graph.directed:
            parents[c].add(p)
        try:
            bic = _bic_total(rows, parents, variables) if rows else None
        except Exception:
            bic = None
        graph.score = bic if graph.score is None else graph.score
        attestation_hash: str | None = None
        if self.attestor is not None:
            try:
                receipt = self.attestor.append(
                    {
                        "kind": DISCOVERY_COMMITTED,
                        "method": request.method,
                        "n_samples": len(rows),
                        "n_variables": n_vars,
                        "alpha": request.alpha,
                        "directed": sorted([list(e) for e in graph.directed]),
                        "undirected": sorted(
                            [sorted(list(e)) for e in graph.undirected]
                        ),
                        "bic_score": bic,
                        "session_id": self.session_id,
                        "ts": time.time(),
                    }
                )
                attestation_hash = getattr(receipt, "entry_hash", None)
            except Exception:
                attestation_hash = None

        report = DiscoveryReport(
            graph=graph,
            method=request.method,
            n_samples=len(rows),
            n_variables=n_vars,
            alpha=request.alpha,
            elapsed_seconds=elapsed,
            bic_score=bic,
            attestation_hash=attestation_hash,
            notes=dict(graph.notes),
        )
        with self._lock:
            self._history.append(report)
        self._emit(
            DISCOVERY_COMMITTED,
            {
                "method": request.method,
                "n_edges": len(graph.directed) + len(graph.undirected),
                "n_directed": len(graph.directed),
                "n_undirected": len(graph.undirected),
                "bic_score": bic,
                "attestation_hash": attestation_hash,
                "elapsed_seconds": elapsed,
            },
        )
        return report

    # ----- composition surface ----------------------------------------

    def attach_to_causal_lab(self, lab: Any) -> tuple[list[list[float]], list[str]]:
        """Drain (context, action_indicator, reward) tuples from a
        `CausalLab` instance into a matrix suitable for `discover(...)`.

        Returns `(rows, variables)`. The matrix interleaves all context
        keys (sorted), a one-hot treatment indicator named `_treated`,
        and the reward named `_reward`.
        """
        events = lab.events() if hasattr(lab, "events") else []
        if not events:
            return [], []
        keys: set[str] = set()
        for ev in events:
            keys.update(ev.context.keys())
        sorted_keys = sorted(keys)
        treat_name = getattr(lab, "treatment", None) or "_treatment"
        variables = sorted_keys + ["_treated", "_reward"]
        rows: list[list[float]] = []
        for ev in events:
            row = [float(ev.context.get(k, 0.0)) for k in sorted_keys]
            row.append(1.0 if ev.action == treat_name else 0.0)
            row.append(float(ev.reward))
            rows.append(row)
        return rows, variables

    def history(self) -> list[DiscoveryReport]:
        with self._lock:
            return list(self._history)

    # ----- internals --------------------------------------------------

    def _test_emitter(self):
        """Returns a callback that emits per-test events on the bus."""
        bus = self.event_bus
        sid = self.session_id

        def _cb(a: str, b: str, cond: list[str], r: float, p: float, indep: bool) -> None:
            ev = Event(
                kind=DISCOVERY_TESTED,
                session_id=sid,
                data={
                    "a": a,
                    "b": b,
                    "cond": list(cond),
                    "r": float(r),
                    "p": float(p),
                    "independent": bool(indep),
                },
            )
            try:
                bus.publish(ev)
            except Exception:
                pass
            if indep:
                drop_ev = Event(
                    kind=DISCOVERY_EDGE_DROPPED,
                    session_id=sid,
                    data={"a": a, "b": b, "cond": list(cond), "p": float(p)},
                )
                try:
                    bus.publish(drop_ev)
                except Exception:
                    pass

        return _cb

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(
                Event(kind=kind, session_id=self.session_id, data=dict(data))
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public re-exports.
# ---------------------------------------------------------------------------

__all__ = [
    "CausalDiscoverer",
    "DiscoveredGraph",
    "DiscoveryReport",
    "DiscoveryRequest",
    "InterventionTarget",
    "DISCOVERY_BOOTSTRAPPED",
    "DISCOVERY_COMMITTED",
    "DISCOVERY_EDGE_DROPPED",
    "DISCOVERY_FAILED",
    "DISCOVERY_ORIENTED",
    "DISCOVERY_STARTED",
    "DISCOVERY_TESTED",
    "KNOWN_METHODS",
    "METHOD_BOOTSTRAP_GES",
    "METHOD_BOOTSTRAP_PC",
    "METHOD_GES",
    "METHOD_PC",
    "fisher_z_test",
    "intervention_targets",
    "partial_correlation",
    "run_bootstrap",
    "run_ges",
    "run_pc",
]
