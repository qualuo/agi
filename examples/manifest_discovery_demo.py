"""Manifest discovery — how a coordination engine introspects the runtime.

Run::

    python examples/manifest_discovery_demo.py

The demo plays the role of an external coordinator that connects to the
runtime, asks "what can you do?", filters the catalog by requirements,
ranks the catalog by free-text intent, and walks the
composes-with graph to plan a multi-primitive pipeline.  All of this
happens against pure metadata — no LLM call, no torch import, no
network — so a coordinator can evaluate hundreds of dispatch decisions
per second.
"""
from __future__ import annotations

import json
import sys

from agi.events import Event, EventBus
from agi.manifest import (
    CERT_ANYTIME,
    CERT_EXACT,
    CERT_PAC,
    DEP_STDLIB,
    KIND_OPTIMIZATION,
    KIND_SAFETY,
    PrimitiveLoader,
    TAG_PAC,
    TAG_SAFETY,
    default_manifest,
)


def section(label: str) -> None:
    print()
    print("=" * 72)
    print(label)
    print("=" * 72)


def main() -> int:
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(events.append)

    m = default_manifest()
    m.attach_bus(bus)

    section(f"Catalog overview: {len(m)} primitives")
    kinds = sorted(m.kinds().items(), key=lambda p: (-p[1], p[0]))
    for kind, n in kinds:
        print(f"  {kind:18s} {n:3d}")

    section("Tag histogram (top 12)")
    tags = sorted(m.tags().items(), key=lambda p: (-p[1], p[0]))[:12]
    for tag, n in tags:
        print(f"  {tag:24s} {n:3d}")

    section("Find: optimisation primitives with a PAC certificate")
    for s in m.find(kind=KIND_OPTIMIZATION, certificate=CERT_PAC):
        print(f"  {s.name:14s}  {s.summary}")

    section("Find: safety primitives with a formal guarantee")
    for s in m.find(kind=KIND_SAFETY, certificate=(CERT_EXACT, CERT_PAC, CERT_ANYTIME)):
        print(f"  {s.name:14s}  cert={s.certificate:8s}  {s.summary}")

    section("Find: stdlib-only primitives — cheapest deployment target")
    light = m.find(dependencies_max=DEP_STDLIB)
    print(f"  {len(light)} primitives runnable with no third-party dependency:")
    for s in light[:10]:
        print(f"    {s.name}")
    if len(light) > 10:
        print(f"    ... and {len(light) - 10} more")

    section("Recommend: rank by free-text intent")
    intents = [
        "I need to pick the best K from N candidates with a confidence interval",
        "estimate treatment effects from observational data",
        "find a Nash equilibrium in a sealed-bid auction",
        "produce a calibrated probabilistic forecast",
    ]
    for intent in intents:
        print()
        print(f'  intent: "{intent}"')
        for spec, score in m.recommend(intent, k=4):
            print(f"    score={score:.3f}  {spec.name:18s}  {spec.summary}")

    section("Plan via composes_with graph from `coordinator`")
    g = m.depends_graph()
    seen: set[str] = set()
    frontier = ["coordinator"]
    pipeline: list[str] = []
    while frontier:
        nxt: list[str] = []
        for node in frontier:
            if node in seen:
                continue
            seen.add(node)
            pipeline.append(node)
            nxt.extend(g.get(node, []))
        frontier = nxt
    print(f"  {len(pipeline)} primitives reachable from `coordinator` in the composes-with closure:")
    print(f"  {' -> '.join(pipeline[:18])}{'  …' if len(pipeline) > 18 else ''}")

    section("Lazy-load demo: instantiate a primitive only when needed")
    loader = PrimitiveLoader()
    print(f"  loaded modules: {loader.loaded()}")
    spec = m.lookup("events")
    mod = loader.load(spec)
    print(f"  loaded {spec.name!r} -> {mod}")
    print(f"  loaded modules: {loader.loaded()}")

    section("Export the catalog as JSON for a remote coordinator")
    blob = m.to_json()
    print(f"  {len(blob)} bytes; sha256={m.fingerprint()[:16]}...")
    payload = json.loads(blob)
    print(f"  schema_version={payload['schema_version']}  count={payload['count']}")
    first = payload["primitives"][0]
    print(f"  first entry name={first['name']!r} stable_id={first['stable_id']!r}")

    section(f"Audit trail: {len(events)} bus events emitted during discovery")
    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    for k, v in sorted(by_kind.items()):
        print(f"  {k:30s} {v}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
