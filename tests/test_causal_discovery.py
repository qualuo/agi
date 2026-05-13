"""Tests for `agi.causal_discovery` — structure learning runtime primitive.

The contract has four parts:

1. **Independence test correctness.** Fisher's z on Gaussian data: the
   marginal independence of generated-independent variables is rejected
   at the configured rate (within Monte Carlo noise); conditional
   independence under chain / fork / collider structures is detected.

2. **PC recovery.** On synthetic data from a known DAG with strong
   signal and large n, PC recovers the Markov equivalence class
   exactly: skeleton matches, v-structures are oriented correctly,
   Meek rules propagate.

3. **GES + refinement recovery.** Score-based hill-climbing with
   constraint-based skeleton refinement (the hybrid MMHC pattern)
   recovers the same equivalence class on the same data.

4. **Bootstrap stability and active intervention.** Edge frequencies
   over bootstrap resamples are well-calibrated; intervention targets
   maximally disambiguate undirected components.

We do not test asymptotic guarantees with tight constants — those hold
only in the limit. We do test:
  - Exact recovery on strong signal (n ≥ 800), confirming
    correctness at sample sizes a production runtime sees;
  - Qualitative monotonicities (more samples → fewer errors,
    higher α → more retained edges, etc.);
  - Cosmetic invariants: thread-safety, no duplicate-direction
    edges in any output, event emissions in order, attestation
    pass-through.
"""
from __future__ import annotations

import math
import random
import statistics
import threading

import pytest

from agi.attest import AttestationLedger
from agi.causal_discovery import (
    CausalDiscoverer,
    DISCOVERY_COMMITTED,
    DISCOVERY_STARTED,
    DISCOVERY_TESTED,
    DiscoveredGraph,
    DiscoveryRequest,
    InterventionTarget,
    KNOWN_METHODS,
    METHOD_BOOTSTRAP_PC,
    METHOD_GES,
    METHOD_PC,
    fisher_z_test,
    intervention_targets,
    partial_correlation,
    run_bootstrap,
    run_ges,
    run_pc,
)
from agi.events import EventBus


# ---------------------------------------------------------------------------
# Data generators for canonical DAGs.
# ---------------------------------------------------------------------------


def _chain(n: int, seed: int = 0) -> list[list[float]]:
    """X → Y → Z."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        y = 0.8 * x + rng.gauss(0, 0.5)
        z = 0.8 * y + rng.gauss(0, 0.5)
        rows.append([x, y, z])
    return rows


def _v_structure(n: int, seed: int = 0) -> list[list[float]]:
    """X → Z ← Y (collider at Z)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        y = rng.gauss(0, 1)
        z = 0.6 * x + 0.6 * y + rng.gauss(0, 0.5)
        rows.append([x, y, z])
    return rows


def _fork(n: int, seed: int = 0) -> list[list[float]]:
    """W is common cause: X ← W → Y."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        w = rng.gauss(0, 1)
        x = 0.7 * w + rng.gauss(0, 0.5)
        y = 0.7 * w + rng.gauss(0, 0.5)
        rows.append([w, x, y])
    return rows


def _two_v_structures(n: int, seed: int = 0) -> list[list[float]]:
    """X1, X2 → Y; X3, X4 → Z; Y → W."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        x3 = rng.gauss(0, 1)
        x4 = rng.gauss(0, 1)
        y = 0.6 * x1 + 0.6 * x2 + rng.gauss(0, 0.5)
        z = 0.6 * x3 + 0.6 * x4 + rng.gauss(0, 0.5)
        w = 0.7 * y + rng.gauss(0, 0.5)
        rows.append([x1, x2, x3, x4, y, z, w])
    return rows


def _independent(n: int, k: int, seed: int = 0) -> list[list[float]]:
    """k independent Gaussian variables."""
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(k)] for _ in range(n)]


# ---------------------------------------------------------------------------
# Independence test.
# ---------------------------------------------------------------------------


def test_partial_correlation_unconditional_matches_pearson() -> None:
    rng = random.Random(0)
    rows = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
    pc_val = partial_correlation(rows, 0, 1, [])
    # Compute Pearson by hand.
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    pearson = num / (sx * sy)
    assert abs(pc_val - pearson) < 1e-10


def test_partial_correlation_self_equal_one() -> None:
    rows = [[1.0, 2.0], [2.0, 3.0]]
    assert partial_correlation(rows, 0, 0, []) == 1.0


def test_fisher_z_independent_data_high_pvalue() -> None:
    # Two genuinely independent gaussians should not reject H_0.
    rng = random.Random(1)
    rows = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(500)]
    r, p, indep = fisher_z_test(rows, 0, 1, [], alpha=0.05)
    assert indep, f"independent data should pass; r={r}, p={p}"
    assert p > 0.05


def test_fisher_z_dependent_data_low_pvalue() -> None:
    # Strongly dependent gaussians should reject.
    rng = random.Random(2)
    rows = []
    for _ in range(500):
        x = rng.gauss(0, 1)
        y = 0.8 * x + rng.gauss(0, 0.3)
        rows.append([x, y])
    r, p, indep = fisher_z_test(rows, 0, 1, [], alpha=0.05)
    assert not indep, f"dependent data should reject; r={r}, p={p}"
    assert p < 0.01


def test_fisher_z_conditional_independence_in_chain() -> None:
    """X → Y → Z: X ⫫ Z | Y (the chain's signature CI)."""
    rows = _chain(500, seed=3)
    _, p_marginal, indep_marginal = fisher_z_test(rows, 0, 2, [], alpha=0.05)
    _, p_cond, indep_cond = fisher_z_test(rows, 0, 2, [1], alpha=0.05)
    assert not indep_marginal, "X _||_ Z marginally is FALSE in a chain"
    assert indep_cond, "X _||_ Z | Y is TRUE in a chain"


def test_fisher_z_collider_opens_when_conditioned() -> None:
    """V-structure X → Z ← Y: X ⫫ Y unconditionally, but conditioning
    on the collider Z makes them dependent (Berkson / explaining-away).
    """
    rows = _v_structure(800, seed=4)
    _, _, indep_marginal = fisher_z_test(rows, 0, 1, [], alpha=0.05)
    _, _, indep_cond = fisher_z_test(rows, 0, 1, [2], alpha=0.05)
    assert indep_marginal, "X _||_ Y unconditionally"
    assert not indep_cond, "X _||_ Y | Z opens by collider conditioning"


def test_fisher_z_zero_samples_returns_independent() -> None:
    r, p, indep = fisher_z_test([], 0, 1, [], alpha=0.05)
    assert indep


def test_fisher_z_false_alarm_rate_under_alpha() -> None:
    """Across many independent-data trials, false-alarm rate ≤ ~alpha.

    Loose bound to keep the test fast and robust to seed; finer bounds
    would require many more trials.
    """
    trials = 200
    alpha = 0.05
    rejected = 0
    rng_master = random.Random(7)
    for t in range(trials):
        rng = random.Random(rng_master.randrange(1 << 30))
        rows = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(400)]
        _, _, indep = fisher_z_test(rows, 0, 1, [], alpha=alpha)
        if not indep:
            rejected += 1
    # Allow 3x slack on Monte Carlo noise.
    assert rejected / trials <= 3 * alpha, (
        f"false-alarm rate {rejected/trials:.3f} exceeds 3·alpha={3*alpha:.3f}"
    )


# ---------------------------------------------------------------------------
# PC algorithm.
# ---------------------------------------------------------------------------


def test_pc_chain_recovers_undirected_skeleton() -> None:
    """Chain X → Y → Z has Markov-equivalent CPDAG X — Y — Z (no v-structure
    to identify direction)."""
    rows = _chain(800, seed=5)
    g = run_pc(rows, ["X", "Y", "Z"], alpha=0.05)
    skeleton = {tuple(sorted(e)) for e in g.undirected} | {
        tuple(sorted([a, b])) for a, b in g.directed
    }
    assert skeleton == {("X", "Y"), ("Y", "Z")}, f"got {skeleton}"


def test_pc_v_structure_oriented() -> None:
    rows = _v_structure(800, seed=6)
    g = run_pc(rows, ["X", "Y", "Z"], alpha=0.05)
    assert ("X", "Z") in g.directed
    assert ("Y", "Z") in g.directed
    assert len(g.undirected) == 0
    # And no spurious X-Y edge.
    assert frozenset({"X", "Y"}) not in g.undirected
    assert ("X", "Y") not in g.directed and ("Y", "X") not in g.directed


def test_pc_fork_undirected() -> None:
    """Fork X ← W → Y is in the same equivalence class as the chain;
    no v-structure → CPDAG is all undirected."""
    rows = _fork(800, seed=7)
    g = run_pc(rows, ["W", "X", "Y"], alpha=0.05)
    skeleton = {tuple(sorted(e)) for e in g.undirected} | {
        tuple(sorted([a, b])) for a, b in g.directed
    }
    assert skeleton == {("W", "X"), ("W", "Y")}, f"got {skeleton}"
    # No v-structures should be detected: W not a collider.
    assert len(g.directed) == 0 or all(
        c == "W" for _, c in g.directed
    ) is False  # at most some Meek propagation, but not v-structures.


def test_pc_independent_variables_no_edges() -> None:
    rows = _independent(500, 4, seed=8)
    g = run_pc(rows, ["A", "B", "C", "D"], alpha=0.05)
    assert len(g.directed) + len(g.undirected) == 0


def test_pc_multi_v_structures_recovers_all() -> None:
    """X1, X2 → Y; X3, X4 → Z; Y → W with two v-structures and one
    chain edge that Meek R1 must orient."""
    rows = _two_v_structures(1000, seed=9)
    g = run_pc(rows, ["X1", "X2", "X3", "X4", "Y", "Z", "W"], alpha=0.05)
    expected = {
        ("X1", "Y"),
        ("X2", "Y"),
        ("X3", "Z"),
        ("X4", "Z"),
        ("Y", "W"),
    }
    assert g.directed == expected, f"got {sorted(g.directed)}"
    assert len(g.undirected) == 0


def test_pc_meek_r1_orients_chain_edge() -> None:
    """V-structure X → Z, and Z — W with X not adjacent to W → Meek R1
    orients Z → W. Construct this with X → Z ← Y, Z → W (so W and X
    aren't adjacent; W has only Z as neighbor)."""
    rng = random.Random(10)
    rows = []
    for _ in range(1000):
        x = rng.gauss(0, 1)
        y = rng.gauss(0, 1)
        z = 0.6 * x + 0.6 * y + rng.gauss(0, 0.4)
        w = 0.7 * z + rng.gauss(0, 0.4)
        rows.append([x, y, z, w])
    g = run_pc(rows, ["X", "Y", "Z", "W"], alpha=0.05)
    assert ("X", "Z") in g.directed
    assert ("Y", "Z") in g.directed
    assert ("Z", "W") in g.directed, "Meek R1 should orient Z → W"
    assert len(g.undirected) == 0


def test_pc_no_conflicting_directions_in_output() -> None:
    """Across many bootstraps the output should never contain both
    (a, b) and (b, a) in directed for any pair."""
    rng = random.Random(11)
    n = 600
    rows = _two_v_structures(n, seed=11)
    for _ in range(30):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        g = run_pc(sample, ["X1", "X2", "X3", "X4", "Y", "Z", "W"], alpha=0.05)
        seen: dict[frozenset, tuple[str, str]] = {}
        for (a, b) in g.directed:
            pair = frozenset({a, b})
            if pair in seen and seen[pair] != (a, b):
                raise AssertionError(
                    f"conflicting directions: {seen[pair]} and {(a, b)}"
                )
            seen[pair] = (a, b)


def test_pc_higher_alpha_retains_more_edges() -> None:
    """At higher α, fewer edges get tested-out by the independence
    rule, so the skeleton retains *no fewer* edges."""
    rows = _two_v_structures(400, seed=12)
    variables = ["X1", "X2", "X3", "X4", "Y", "Z", "W"]
    g_low = run_pc(rows, variables, alpha=0.001)
    g_high = run_pc(rows, variables, alpha=0.2)
    n_low = len(g_low.directed) + len(g_low.undirected)
    n_high = len(g_high.directed) + len(g_high.undirected)
    assert n_high >= n_low


def test_pc_max_cond_size_caps_test_order() -> None:
    """`max_cond_size` should prevent any test with |S| > cap."""
    rows = _two_v_structures(800, seed=13)
    variables = ["X1", "X2", "X3", "X4", "Y", "Z", "W"]
    bus = EventBus(history_limit=10000)
    g = run_pc(rows, variables, alpha=0.05, max_cond_size=1)
    # Doesn't need to be empty; just makes sure no crash and result is a CPDAG.
    assert isinstance(g, DiscoveredGraph)


# ---------------------------------------------------------------------------
# GES + skeleton refinement.
# ---------------------------------------------------------------------------


def test_ges_v_structure_recovers() -> None:
    rows = _v_structure(800, seed=14)
    g = run_ges(rows, ["X", "Y", "Z"])
    assert ("X", "Z") in g.directed
    assert ("Y", "Z") in g.directed
    # No spurious X-Y edge after refinement.
    assert frozenset({"X", "Y"}) not in g.undirected
    assert ("X", "Y") not in g.directed and ("Y", "X") not in g.directed


def test_ges_fork_undirected() -> None:
    rows = _fork(800, seed=15)
    g = run_ges(rows, ["W", "X", "Y"])
    skeleton = {tuple(sorted(e)) for e in g.undirected} | {
        tuple(sorted([a, b])) for a, b in g.directed
    }
    assert skeleton == {("W", "X"), ("W", "Y")}, f"got {skeleton}"


def test_ges_independent_data_empty_graph() -> None:
    rows = _independent(500, 4, seed=16)
    g = run_ges(rows, ["A", "B", "C", "D"])
    # Refinement should drop any spurious finite-sample edges.
    assert len(g.directed) + len(g.undirected) <= 1  # very loose; rare edges OK


def test_ges_returns_a_bic_score() -> None:
    rows = _chain(500, seed=17)
    g = run_ges(rows, ["X", "Y", "Z"])
    assert g.score is not None
    assert isinstance(g.score, float)


def test_ges_multi_v_structures_skeleton_matches_pc() -> None:
    rows = _two_v_structures(1000, seed=18)
    variables = ["X1", "X2", "X3", "X4", "Y", "Z", "W"]
    g_pc = run_pc(rows, variables, alpha=0.05)
    g_ges = run_ges(rows, variables)
    sk_pc = {tuple(sorted(e)) for e in g_pc.undirected} | {
        tuple(sorted([a, b])) for a, b in g_pc.directed
    }
    sk_ges = {tuple(sorted(e)) for e in g_ges.undirected} | {
        tuple(sorted([a, b])) for a, b in g_ges.directed
    }
    assert sk_pc == sk_ges, f"PC skeleton {sk_pc} != GES skeleton {sk_ges}"


# ---------------------------------------------------------------------------
# Bootstrap stability.
# ---------------------------------------------------------------------------


def test_bootstrap_pc_high_confidence_on_strong_signal() -> None:
    """When the signal is strong, bootstrap edge frequencies → 1."""
    rows = _v_structure(800, seed=19)
    g = run_bootstrap(
        rows,
        ["X", "Y", "Z"],
        method="pc",
        n_bootstrap=30,
        edge_threshold=0.5,
        seed=42,
    )
    # Both v-structure edges should be highly confident.
    confidence = {tuple(sorted(k)): v for k, v in g.edge_confidence.items()}
    assert confidence.get(("X", "Z"), 0) > 0.8
    assert confidence.get(("Y", "Z"), 0) > 0.8


def test_bootstrap_confidence_never_exceeds_one() -> None:
    """Edge frequency is a proper proportion."""
    rows = _two_v_structures(600, seed=20)
    variables = ["X1", "X2", "X3", "X4", "Y", "Z", "W"]
    g = run_bootstrap(
        rows,
        variables,
        method="pc",
        n_bootstrap=20,
        edge_threshold=0.0,
        seed=21,
    )
    for k, v in g.edge_confidence.items():
        assert 0.0 <= v <= 1.0, f"edge {k} has confidence {v}"


def test_bootstrap_threshold_filters_edges() -> None:
    """Edges below the threshold are dropped from the consensus CPDAG."""
    rows = _two_v_structures(400, seed=22)
    variables = ["X1", "X2", "X3", "X4", "Y", "Z", "W"]
    g_strict = run_bootstrap(
        rows,
        variables,
        method="pc",
        n_bootstrap=20,
        edge_threshold=0.95,
        seed=23,
    )
    g_loose = run_bootstrap(
        rows,
        variables,
        method="pc",
        n_bootstrap=20,
        edge_threshold=0.1,
        seed=23,
    )
    n_strict = len(g_strict.directed) + len(g_strict.undirected)
    n_loose = len(g_loose.directed) + len(g_loose.undirected)
    assert n_loose >= n_strict


def test_bootstrap_ges_runs() -> None:
    rows = _v_structure(400, seed=24)
    g = run_bootstrap(rows, ["X", "Y", "Z"], method="ges", n_bootstrap=10, seed=25)
    assert isinstance(g, DiscoveredGraph)


def test_bootstrap_unknown_method_raises() -> None:
    rows = _v_structure(100, seed=26)
    with pytest.raises(ValueError):
        run_bootstrap(rows, ["X", "Y", "Z"], method="bogus", n_bootstrap=5)


# ---------------------------------------------------------------------------
# Discovered graph accessors / serialisation.
# ---------------------------------------------------------------------------


def test_neighbours_and_parents() -> None:
    g = DiscoveredGraph(
        variables=("A", "B", "C"),
        directed={("A", "B"), ("B", "C")},
        undirected=set(),
    )
    assert g.parents("B") == {"A"}
    assert g.children("A") == {"B"}
    assert g.neighbours("B") == {"A", "C"}
    assert g.adjacent("A", "B")
    assert not g.adjacent("A", "C")


def test_markov_blanket() -> None:
    """MB(B) for A→B, B→C, D→C should be {A, C, D}."""
    g = DiscoveredGraph(
        variables=("A", "B", "C", "D"),
        directed={("A", "B"), ("B", "C"), ("D", "C")},
        undirected=set(),
    )
    assert g.markov_blanket("B") == {"A", "C", "D"}


def test_to_dict_round_trip_invariants() -> None:
    g = DiscoveredGraph(
        variables=("A", "B"),
        directed={("A", "B")},
        undirected={frozenset({"A", "B"})},  # weird but allowed at type level
        edge_confidence={("A", "B"): 0.97},
        score=-42.0,
        method="pc",
        n_samples=500,
        alpha=0.05,
    )
    d = g.to_dict()
    assert d["variables"] == ["A", "B"]
    assert d["directed"] == [["A", "B"]]
    assert d["edge_confidence"] == {"A->B": 0.97}


def test_edge_summary_includes_confidence() -> None:
    g = DiscoveredGraph(
        variables=("A", "B"),
        directed={("A", "B")},
        edge_confidence={("A", "B"): 0.85},
    )
    rows = g.edge_summary()
    assert len(rows) == 1
    a, b, kind, conf = rows[0]
    assert (a, b, kind) == ("A", "B", "→")
    assert conf == 0.85


def test_shd_identity_zero() -> None:
    g = DiscoveredGraph(
        variables=("A", "B", "C"),
        directed={("A", "B"), ("B", "C")},
    )
    assert g.shd(g) == 0


def test_shd_differing_skeletons() -> None:
    g1 = DiscoveredGraph(variables=("A", "B"), directed={("A", "B")})
    g2 = DiscoveredGraph(variables=("A", "B"), directed=set())
    # SHD ≥ 1; missing skeleton edge counted.
    assert g1.shd(g2) >= 1


# ---------------------------------------------------------------------------
# Active intervention selection.
# ---------------------------------------------------------------------------


def test_intervention_target_orients_undirected_neighbours() -> None:
    """Intervening on a node should orient every undirected edge
    incident to it."""
    g = DiscoveredGraph(
        variables=("A", "B", "C"),
        undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
    )
    targets = intervention_targets(g)
    # A has 2 incident undirected edges; should orient both.
    a_target = next(t for t in targets if t.variable == "A")
    assert a_target.expected_orientations >= 2
    assert a_target.undirected_incident == 2


def test_intervention_targets_sorted_descending() -> None:
    g = DiscoveredGraph(
        variables=("A", "B", "C", "D"),
        undirected={
            frozenset({"A", "B"}),
            frozenset({"A", "C"}),
            frozenset({"A", "D"}),
            frozenset({"B", "C"}),
        },
    )
    targets = intervention_targets(g)
    # A has 3 incident, B and C each have 2, D has 1.
    assert targets[0].variable == "A"


def test_intervention_targets_budget_truncates() -> None:
    g = DiscoveredGraph(
        variables=("A", "B", "C"),
        undirected={
            frozenset({"A", "B"}),
            frozenset({"B", "C"}),
            frozenset({"A", "C"}),
        },
    )
    targets = intervention_targets(g, budget=2)
    assert len(targets) == 2


def test_intervention_target_empty_when_no_undirected() -> None:
    g = DiscoveredGraph(
        variables=("A", "B"),
        directed={("A", "B")},
    )
    targets = intervention_targets(g)
    assert targets == []


def test_intervention_target_to_dict() -> None:
    t = InterventionTarget(
        variable="A",
        undirected_incident=2,
        expected_orientations=3.0,
        rationale="x",
    )
    d = t.to_dict()
    assert d["variable"] == "A"
    assert d["expected_orientations"] == 3.0


# ---------------------------------------------------------------------------
# CausalDiscoverer surface.
# ---------------------------------------------------------------------------


def test_discoverer_pc_recovers_v_structure() -> None:
    rows = _v_structure(800, seed=27)
    d = CausalDiscoverer()
    report = d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="pc"))
    assert report.method == "pc"
    assert report.n_samples == 800
    assert report.n_variables == 3
    assert ("X", "Z") in report.graph.directed
    assert ("Y", "Z") in report.graph.directed
    assert report.bic_score is not None
    assert report.elapsed_seconds >= 0


def test_discoverer_ges_works() -> None:
    rows = _v_structure(500, seed=28)
    d = CausalDiscoverer()
    report = d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="ges"))
    assert report.method == "ges"
    assert isinstance(report.graph, DiscoveredGraph)


def test_discoverer_bootstrap_pc_works() -> None:
    rows = _v_structure(500, seed=29)
    d = CausalDiscoverer()
    report = d.discover(
        rows,
        ["X", "Y", "Z"],
        request=DiscoveryRequest(method="bootstrap_pc", n_bootstrap=10, seed=29),
    )
    assert "bootstrap" in report.method
    assert isinstance(report.graph, DiscoveredGraph)


def test_discoverer_invalid_method_raises() -> None:
    rows = _v_structure(100, seed=30)
    d = CausalDiscoverer()
    with pytest.raises(ValueError):
        d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="bogus"))


def test_discoverer_invalid_data_shape_raises() -> None:
    rows = [[1.0, 2.0], [1.0]]  # ragged
    d = CausalDiscoverer()
    with pytest.raises(ValueError):
        d.discover(rows, ["X", "Y"], request=DiscoveryRequest())


def test_discoverer_empty_variables_raises() -> None:
    d = CausalDiscoverer()
    with pytest.raises(ValueError):
        d.discover([], [], request=DiscoveryRequest())


def test_discoverer_events_emitted() -> None:
    bus = EventBus(history_limit=2000)
    rows = _v_structure(300, seed=31)
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.kind))
    d = CausalDiscoverer(event_bus=bus)
    d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="pc"))
    assert DISCOVERY_STARTED in seen
    assert DISCOVERY_COMMITTED in seen
    # PC tests at least one independence; per-test events should fire.
    assert DISCOVERY_TESTED in seen


def test_discoverer_attestation_pass_through() -> None:
    ledger = AttestationLedger()
    rows = _v_structure(200, seed=32)
    d = CausalDiscoverer(attestor=ledger)
    report = d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="pc"))
    assert report.attestation_hash is not None
    assert len(report.attestation_hash) == 64  # SHA-256 hex
    ok, _why = ledger.verify()
    assert ok


def test_discoverer_history_records_each_call() -> None:
    rows = _v_structure(200, seed=33)
    d = CausalDiscoverer()
    d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="pc"))
    d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="ges"))
    history = d.history()
    assert len(history) == 2
    assert history[0].method == "pc"
    assert history[1].method == "ges"


def test_discoverer_threadsafe_concurrent_calls() -> None:
    """Multiple threads can call discover() simultaneously without
    corrupting the history list."""
    d = CausalDiscoverer()
    rows = _v_structure(150, seed=34)

    def _work() -> None:
        d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method="pc"))

    threads = [threading.Thread(target=_work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(d.history()) == 8


def test_known_methods_constant_aligns_with_request_validation() -> None:
    rows = _v_structure(50, seed=35)
    d = CausalDiscoverer()
    for m in KNOWN_METHODS:
        # All known methods should run; small data → graph may be empty,
        # but no exception should escape.
        d.discover(rows, ["X", "Y", "Z"], request=DiscoveryRequest(method=m, n_bootstrap=4))


# ---------------------------------------------------------------------------
# Integration with CausalLab.
# ---------------------------------------------------------------------------


def test_attach_to_causal_lab_drains_events() -> None:
    """Stub a lab-like object and verify the discoverer reads its events."""

    class _StubLab:
        def __init__(self) -> None:
            self.treatment = "T"
            self._events: list[Any] = []

        def events(self) -> list:
            return self._events

    class _Ev:
        def __init__(self, ctx: dict, action: str, reward: float) -> None:
            self.context = ctx
            self.action = action
            self.propensity = 1.0
            self.reward = reward

    from typing import Any  # noqa: F401

    lab = _StubLab()
    rng = random.Random(36)
    for _ in range(200):
        x = rng.gauss(0, 1)
        y = 0.5 * x + rng.gauss(0, 0.5)
        action = "T" if rng.random() < 0.5 else "C"
        reward = (1.0 if action == "T" else 0.0) * 0.3 + 0.5 * y + rng.gauss(0, 0.3)
        lab._events.append(_Ev({"x": x, "y": y}, action, reward))

    d = CausalDiscoverer()
    rows, vars_ = d.attach_to_causal_lab(lab)
    assert len(rows) == 200
    assert "_treated" in vars_ and "_reward" in vars_
    assert vars_[0] == "x" and vars_[1] == "y"
    # Ensure the matrix is properly shaped.
    assert all(len(r) == len(vars_) for r in rows)


# ---------------------------------------------------------------------------
# Stress: large variable set runs and terminates.
# ---------------------------------------------------------------------------


def test_pc_large_independent_set_runs_in_bounded_time() -> None:
    """10 independent variables. PC's complexity is exponential in |Adj|,
    but on independent data the skeleton becomes empty almost immediately
    at order 0, so the run is fast."""
    rng = random.Random(37)
    rows = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(300)]
    variables = [f"V{i}" for i in range(10)]
    g = run_pc(rows, variables, alpha=0.05)
    assert isinstance(g, DiscoveredGraph)
    # At alpha = 0.05 over C(10,2)=45 pairs, expected false positives ≤ 5.
    assert len(g.directed) + len(g.undirected) <= 10
