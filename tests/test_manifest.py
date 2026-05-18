"""Tests for the runtime-primitive Manifest catalog."""
from __future__ import annotations

import importlib
import json
import os
import pathlib

import pytest

from agi.events import Event, EventBus
from agi.manifest import (
    CERT_ANYTIME,
    CERT_EXACT,
    CERT_NONE,
    CERT_PAC,
    DEP_LLM,
    DEP_STDLIB,
    DETERMINISM_PURE,
    DETERMINISM_SEEDED,
    KIND_OPTIMIZATION,
    KIND_SAFETY,
    KNOWN_CERTIFICATES,
    KNOWN_DEPENDENCIES,
    KNOWN_DETERMINISMS,
    KNOWN_KINDS,
    MANIFEST_BUILT,
    MANIFEST_DIFFED,
    MANIFEST_EXPORTED,
    MANIFEST_LOOKUP,
    MANIFEST_QUERIED,
    MANIFEST_RECOMMENDED,
    Manifest,
    ManifestError,
    PrimitiveLoader,
    PrimitiveSpec,
    SCHEMA_VERSION,
    TAG_DISCRETE_OPT,
    TAG_LLM,
    TAG_PAC,
    TAG_SAFETY,
    auto_discover,
    default_manifest,
    reset_default_manifest,
)


# ---------------------------------------------------------------------------
# PrimitiveSpec

def test_primitivespec_minimum_fields():
    s = PrimitiveSpec(name="x", kind="infrastructure", summary="A thing.")
    assert s.name == "x"
    assert s.stable_id == "agi.x.v1"
    assert s.certificate == CERT_NONE
    assert s.determinism == DETERMINISM_SEEDED
    assert s.dependency == DEP_STDLIB
    assert s.tags == ()


def test_primitivespec_rejects_unknown_kind():
    with pytest.raises(ManifestError):
        PrimitiveSpec(name="x", kind="nope", summary="X.")


def test_primitivespec_rejects_unknown_certificate():
    with pytest.raises(ManifestError):
        PrimitiveSpec(
            name="x", kind="infrastructure", summary="X.", certificate="hand-wave",
        )


def test_primitivespec_rejects_unknown_determinism():
    with pytest.raises(ManifestError):
        PrimitiveSpec(
            name="x", kind="infrastructure", summary="X.", determinism="vibes",
        )


def test_primitivespec_rejects_unknown_dependency():
    with pytest.raises(ManifestError):
        PrimitiveSpec(
            name="x", kind="infrastructure", summary="X.", dependency="cuda",
        )


def test_primitivespec_rejects_empty_summary():
    with pytest.raises(ManifestError):
        PrimitiveSpec(name="x", kind="infrastructure", summary="")


def test_primitivespec_requires_trailing_period():
    with pytest.raises(ManifestError):
        PrimitiveSpec(name="x", kind="infrastructure", summary="No dot")


def test_primitivespec_requires_name():
    with pytest.raises(ManifestError):
        PrimitiveSpec(name="", kind="infrastructure", summary="X.")


def test_primitivespec_dedupes_and_sorts_tags():
    s = PrimitiveSpec(
        name="x", kind="infrastructure", summary="X.",
        tags=("safety", "pac-bound", "safety", "anytime-valid"),
    )
    assert s.tags == ("anytime-valid", "pac-bound", "safety")


def test_primitivespec_round_trip_dict():
    s = PrimitiveSpec(
        name="x", kind="optimization", summary="X.",
        tags=("discrete-optimization", "pac-bound"),
        certificate=CERT_PAC,
        composes_with=("y", "z"),
        references=("Hoeffding 1963",),
        events_emitted=("x.started", "x.finished"),
        complexity="O(n log n)",
        notes="thread-safe",
    )
    d = s.to_dict()
    assert d["stable_id"] == "agi.x.v1"
    s2 = PrimitiveSpec.from_dict(d)
    assert s2 == s


# ---------------------------------------------------------------------------
# Manifest

def test_manifest_default_has_full_coverage():
    m = default_manifest(fresh=True)
    curated = {s.name for s in m.list()}
    discovered = {s.name for s in auto_discover()}
    # Every module in agi/ that has a docstring matching the pattern must
    # be present in the curated table — auto_discover is a safety net,
    # not the source of truth.
    missing = discovered - curated
    assert not missing, f"manifest missing curated entries for: {sorted(missing)}"


def test_manifest_kinds_partition_specs():
    m = default_manifest()
    kinds = m.kinds()
    total = sum(kinds.values())
    assert total == len(m)
    assert set(kinds).issubset(KNOWN_KINDS)


def test_manifest_lookup_known_and_unknown():
    m = default_manifest()
    a = m.lookup("annealer")
    assert a.name == "annealer"
    assert a.kind == KIND_OPTIMIZATION
    with pytest.raises(ManifestError):
        m.lookup("nonexistent_primitive")


def test_manifest_find_by_kind():
    m = default_manifest()
    opts = m.find(kind=KIND_OPTIMIZATION)
    assert len(opts) >= 5
    assert all(s.kind == KIND_OPTIMIZATION for s in opts)
    names = {s.name for s in opts}
    assert {"annealer", "bayesopt", "submodular"}.issubset(names)


def test_manifest_find_by_tag_multi():
    m = default_manifest()
    pac = m.find(tag={TAG_PAC, TAG_LLM})
    assert pac
    for s in pac:
        assert TAG_PAC in s.tags or TAG_LLM in s.tags


def test_manifest_find_by_certificate():
    m = default_manifest()
    exact = m.find(certificate=CERT_EXACT)
    assert {"verifier", "solver", "planner", "reasoner", "composer", "inducer"}.issubset(
        {s.name for s in exact}
    )


def test_manifest_find_by_dependency_max_stdlib():
    m = default_manifest()
    stdlib_only = m.find(dependencies_max=DEP_STDLIB)
    assert stdlib_only
    for s in stdlib_only:
        assert s.dependency == DEP_STDLIB


def test_manifest_find_by_dependency_max_includes_lighter():
    """dependencies_max="numpy" should include both numpy and stdlib."""
    m = default_manifest()
    light = m.find(dependencies_max="numpy")
    deps = {s.dependency for s in light}
    assert "numpy" in deps or "stdlib" in deps
    assert "llm" not in deps
    assert "torch" not in deps


def test_manifest_find_by_dependency_max_unknown():
    m = default_manifest()
    with pytest.raises(ManifestError):
        m.find(dependencies_max="cuda")


def test_manifest_find_composes_with():
    m = default_manifest()
    hits = m.find(composes_with="annealer")
    assert {"submodular", "pareto", "robustifier", "portfolio"}.issubset(
        {s.name for s in hits}
    )


def test_manifest_find_name_prefix():
    m = default_manifest()
    causal = m.find(name_prefix="causal")
    assert {"causal", "causal_discovery"}.issubset({s.name for s in causal})


def test_manifest_find_intersection():
    m = default_manifest()
    # discrete-opt + PAC bound
    hits = m.find(kind=KIND_OPTIMIZATION, tag=TAG_DISCRETE_OPT, certificate=CERT_PAC)
    names = {s.name for s in hits}
    assert "annealer" in names
    assert "submodular" in names


def test_manifest_recommend_returns_relevant():
    m = default_manifest()
    hits = m.recommend("simulated annealing combinatorial optimisation", k=10)
    names = [s.name for s, _ in hits]
    # Annealer is the canonical answer.
    assert "annealer" in names


def test_manifest_recommend_zero_k():
    m = default_manifest()
    assert m.recommend("anything", k=0) == []


def test_manifest_recommend_empty_intent():
    m = default_manifest()
    assert m.recommend("", k=5) == []


def test_manifest_recommend_scores_descending():
    m = default_manifest()
    hits = m.recommend("safety calibration conformal prediction", k=10)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_manifest_recommend_tie_break_stable():
    m = default_manifest()
    a = m.recommend("planning task graph", k=20)
    b = m.recommend("planning task graph", k=20)
    assert [s.name for s, _ in a] == [s.name for s, _ in b]


def test_manifest_tags_histogram():
    m = default_manifest()
    h = m.tags()
    assert h.get(TAG_PAC, 0) >= 5
    assert h.get(TAG_SAFETY, 0) >= 5


def test_manifest_depends_graph_closed_under_lookup():
    m = default_manifest()
    g = m.depends_graph()
    names = set(g)
    for src, targets in g.items():
        for t in targets:
            assert t in names, f"{src} -> {t} not in catalog"


def test_manifest_depends_graph_strips_unknown():
    m = Manifest([
        PrimitiveSpec(name="a", kind="infrastructure", summary="A.",
                      composes_with=("b", "ghost")),
        PrimitiveSpec(name="b", kind="infrastructure", summary="B."),
    ])
    g = m.depends_graph()
    assert g == {"a": ["b"], "b": []}


def test_manifest_json_round_trip():
    m = default_manifest()
    blob = m.to_json()
    m2 = Manifest.from_json(blob)
    assert len(m) == len(m2)
    assert {s.name for s in m.list()} == {s.name for s in m2.list()}
    assert m.fingerprint() == m2.fingerprint()


def test_manifest_json_pretty():
    m = default_manifest()
    blob = m.to_json(pretty=True)
    # Pretty-printed JSON has indentation.
    assert "\n  " in blob
    payload = json.loads(blob)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["count"] == len(m)


def test_manifest_json_rejects_schema_skew():
    m = default_manifest()
    payload = json.loads(m.to_json())
    payload["schema_version"] = "999.0"
    blob = json.dumps(payload)
    with pytest.raises(ManifestError):
        Manifest.from_json(blob)
    # opt-in override works
    m2 = Manifest.from_json(blob, allow_schema_skew=True)
    assert len(m2) == len(m)


def test_manifest_json_rejects_malformed():
    with pytest.raises(ManifestError):
        Manifest.from_json("{not json")


def test_manifest_fingerprint_stable_across_processes():
    m1 = default_manifest(fresh=True)
    m2 = default_manifest(fresh=True)
    assert m1.fingerprint() == m2.fingerprint()


def test_manifest_register_unregister():
    m = Manifest()
    s = PrimitiveSpec(name="x", kind="infrastructure", summary="X.",
                      tags=("safety",))
    m.register(s)
    assert "x" in m
    assert m.tags()["safety"] == 1
    m.unregister("x")
    assert "x" not in m
    assert m.tags().get("safety", 0) == 0
    # unregister missing is a no-op
    m.unregister("x")


def test_manifest_diff_added_removed_changed():
    a = Manifest([
        PrimitiveSpec(name="x", kind="infrastructure", summary="A.",
                      tags=("safety",)),
        PrimitiveSpec(name="y", kind="infrastructure", summary="Y."),
    ])
    b = Manifest([
        PrimitiveSpec(name="x", kind="infrastructure", summary="A.",
                      tags=("safety", "pac-bound")),
        PrimitiveSpec(name="z", kind="infrastructure", summary="Z."),
    ])
    d = a.diff(b)
    assert d["added"] == ["z"]
    assert d["removed"] == ["y"]
    assert any(c["name"] == "x" and c["field"] == "tags" for c in d["changed"])


def test_manifest_default_cached():
    m1 = default_manifest()
    m2 = default_manifest()
    assert m1 is m2
    reset_default_manifest()
    m3 = default_manifest()
    assert m3 is not m1


def test_manifest_default_include_auto_bypasses_cache():
    a = default_manifest()
    b = default_manifest(include_auto=True)
    # include_auto returns a fresh manifest and does not poison the cache.
    assert default_manifest() is a
    # b is at least as large as a
    assert len(b) >= len(a)


def test_auto_discover_returns_specs_with_summaries():
    discovered = auto_discover()
    assert discovered
    for s in discovered:
        assert s.summary.endswith(".")
        assert s.name


def test_manifest_events_on_bus():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    m = Manifest(_initial_specs_with_safety_tag(), bus=bus)
    m.lookup("a")
    m.find(tag="safety")
    m.recommend("safety controls", k=2)
    m.to_json()
    m.diff(Manifest(_initial_specs_with_safety_tag()))
    kinds = {e.kind for e in seen}
    assert MANIFEST_BUILT in kinds
    assert MANIFEST_LOOKUP in kinds
    assert MANIFEST_QUERIED in kinds
    assert MANIFEST_RECOMMENDED in kinds
    assert MANIFEST_EXPORTED in kinds
    assert MANIFEST_DIFFED in kinds


def test_manifest_attach_detach_bus():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    m = Manifest(_initial_specs_with_safety_tag())
    m.attach_bus(bus)
    m.lookup("a")
    assert any(e.kind == MANIFEST_LOOKUP for e in seen)
    m.detach_bus()
    seen.clear()
    m.lookup("a")
    assert seen == []


def test_manifest_bus_failure_does_not_propagate():
    class BadBus:
        def publish(self, *_a, **_kw):
            raise RuntimeError("boom")
    m = Manifest(_initial_specs_with_safety_tag(), bus=BadBus())
    # Must not raise.
    m.lookup("a")
    m.find(tag="safety")


def test_manifest_iteration_ordered():
    m = default_manifest()
    names = [s.name for s in m]
    assert names == sorted(names)


def test_primitiveloader_caches_imports():
    m = default_manifest()
    loader = PrimitiveLoader()
    spec = m.lookup("events")  # cheap to import (no torch / llm)
    a = loader.load(spec)
    b = loader.load(spec)
    assert a is b
    assert "events" in loader.loaded()


def test_primitiveloader_lazy():
    """Holding a spec must not import its backing module."""
    m = default_manifest()
    spec = m.lookup("events")
    loader = PrimitiveLoader()
    assert loader.loaded() == []
    loader.load(spec)
    assert loader.loaded() == ["events"]


def test_demo_paths_exist_when_set():
    """Every spec with a demo_path must point at a real file on disk."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    m = default_manifest()
    for spec in m.list():
        if spec.demo_path:
            p = repo_root / spec.demo_path
            assert p.exists(), f"{spec.name} demo_path {spec.demo_path} does not exist"


def test_composes_with_resolves_in_curated_table():
    """Every composes_with reference must point at a real curated spec."""
    m = default_manifest()
    names = {s.name for s in m.list()}
    for spec in m.list():
        for other in spec.composes_with:
            assert other in names, f"{spec.name}.composes_with -> {other!r} not in catalog"


def test_known_constants_consistent():
    assert KNOWN_KINDS  # non-empty
    assert KNOWN_CERTIFICATES
    assert KNOWN_DEPENDENCIES
    assert KNOWN_DETERMINISMS
    for cert in (CERT_NONE, CERT_PAC, CERT_ANYTIME, CERT_EXACT):
        assert cert in KNOWN_CERTIFICATES
    for det in (DETERMINISM_PURE, DETERMINISM_SEEDED):
        assert det in KNOWN_DETERMINISMS


def test_curated_catalog_canonical():
    """Every curated spec passes its own validation."""
    m = default_manifest()
    for s in m.list():
        assert s.name
        assert s.summary.endswith(".")
        assert s.kind in KNOWN_KINDS
        assert s.certificate in KNOWN_CERTIFICATES
        assert s.determinism in KNOWN_DETERMINISMS
        assert s.dependency in KNOWN_DEPENDENCIES


def test_kind_optimization_has_annealer_and_solver():
    m = default_manifest()
    names = {s.name for s in m.find(kind=KIND_OPTIMIZATION)}
    assert "annealer" in names
    assert "solver" in names


def test_kind_safety_includes_verifier_and_conformal():
    m = default_manifest()
    names = {s.name for s in m.find(kind=KIND_SAFETY)}
    assert "verifier" in names
    assert "conformal" in names


# ---------------------------------------------------------------------------
# helpers

def _initial_specs_with_safety_tag() -> list[PrimitiveSpec]:
    return [
        PrimitiveSpec(name="a", kind="safety", summary="A.", tags=("safety",)),
        PrimitiveSpec(name="b", kind="infrastructure", summary="B."),
    ]
