"""Analogist demo — structure-mapping analogical reasoning as a runtime primitive.

Five scenarios show how a coordination engine actually drives the
Analogist at runtime, each producing an explicit, one-to-one,
parallel-connected mapping with a Structural Evaluation Score and
candidate inferences a downstream primitive can falsify:

  1. The Falkenhainer-Forbus-Gentner canonical: solar system ↔ atom.
     A coordinator that has the solar system as a base description
     transfers structural facts (revolves_around, greater-mass) onto
     the atom and predicts new facts (greater-temperature) for
     downstream Refuter / Conformal verification.

  2. Water-flow ↔ heat-flow.  The runtime's pattern for *cross-
     domain* transport: when one domain is rich and the other has
     gaps, analogy generates the gap-filling predictions.

  3. MAC/FAC retrieval: a probe (a new ticket) is matched against a
     memory of past cases.  MAC's content-vector dot-product picks a
     short-list in O(memory); FAC's SME ranks the short-list by
     structural similarity — the architecture that lets the runtime
     keep a large case base and still answer in bounded time.

  4. Karla-the-hawk ↔ Zerdia (Gentner's classic narrative-transfer
     stimulus).  Higher-order relations (cause, prevent) carry the
     systematicity that lets the runtime align two stories whose
     surface predicates only weakly overlap.

  5. Letter-string proportional analogy (a:b :: c:?) via
     ``ProportionalAnalogy``.  The Hofstadter / Copycat micro-domain
     as a symbol-stream pattern transfer primitive: a one-shot
     rewrite rule is inferred from one example and applied to
     another.

Run::

    python examples/analogist_demo.py
"""
from __future__ import annotations

import time

from agi.analogist import (
    Analogist,
    AnalogistConfig,
    ProportionalAnalogy,
    acme,
    sme,
)


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def _show_mapping(m, label: str = "mapping") -> None:
    print(f"  {label}: score={m.score:.3f}")
    print(f"    entities: {dict(m.entity_map)}")
    if m.support_breakdown:
        breakdown = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(m.support_breakdown.items())
        )
        print(f"    support : {breakdown}")
    if m.inferences:
        print(f"    candidate inferences ({len(m.inferences)}):")
        for inf, src in m.inferences:
            print(f"      {inf}   (from {src})")


# ----------------------------------------------------------------
# Scenario 1 — solar system ↔ atom (FFG 1989 canonical)
# ----------------------------------------------------------------


def scenario_solar_atom() -> None:
    banner("Scenario 1 — solar system ↔ atom (FFG 1989 canonical)")
    a = sme(hmac_key=b"demo-key")
    a.add_description("solar", [
        ("cause",
         ("attracts", "sun", "planet"),
         ("revolves_around", "planet", "sun")),
        ("greater", ("mass", "sun"), ("mass", "planet")),
        ("greater", ("temperature", "sun"), ("temperature", "planet")),
        ("yellow", "sun"),
    ])
    a.add_description("atom", [
        ("cause",
         ("attracts", "nucleus", "electron"),
         ("revolves_around", "electron", "nucleus")),
        ("greater", ("mass", "nucleus"), ("mass", "electron")),
    ])
    t0 = time.monotonic()
    rep = a.match("solar", "atom")
    dt = time.monotonic() - t0
    print(f"  ran in {dt * 1000:.1f} ms; "
          f"{rep.n_match_hypotheses} MHs, {rep.n_gmaps_explored} gmaps explored")
    print(f"  certificate: {rep.certificate[:32]}…")
    for i, m in enumerate(rep.mappings):
        _show_mapping(m, f"mapping #{i + 1}")


# ----------------------------------------------------------------
# Scenario 2 — water flow ↔ heat flow
# ----------------------------------------------------------------


def scenario_water_heat() -> None:
    banner("Scenario 2 — water flow ↔ heat flow")
    a = sme()
    a.add_description("water_flow", [
        ("cause",
         ("greater", ("pressure", "beaker"), ("pressure", "vial")),
         ("flows", "water", "pipe", "beaker", "vial")),
        ("greater", ("diameter", "beaker"), ("diameter", "vial")),
        ("liquid", "water"),
    ])
    a.add_description("heat_flow", [
        ("cause",
         ("greater", ("temperature", "coffee"), ("temperature", "ice_cube")),
         ("flows", "heat", "bar", "coffee", "ice_cube")),
    ])
    rep = a.match("water_flow", "heat_flow")
    print(f"  {rep.n_match_hypotheses} MHs, "
          f"{rep.n_gmaps_explored} gmaps explored, "
          f"{rep.n_inferences} inferences")
    if rep.mappings:
        _show_mapping(rep.mappings[0], "best mapping")
        # Coordinator's downstream use:
        if rep.mappings[0].inferences:
            print("  → these inferences would be handed to Refuter for "
                  "falsification or to Conformal for coverage testing.")


# ----------------------------------------------------------------
# Scenario 3 — MAC/FAC retrieval against a case base
# ----------------------------------------------------------------


def scenario_mac_fac() -> None:
    banner("Scenario 3 — MAC/FAC retrieval against a small case base")
    a = sme()
    # Build a case base.
    a.add_description("solar_system", [
        ("cause",
         ("attracts", "sun", "planet"),
         ("revolves_around", "planet", "sun")),
        ("greater", ("mass", "sun"), ("mass", "planet")),
    ])
    a.add_description("water_flow", [
        ("cause", ("greater", ("pressure", "tank"), ("pressure", "sink")),
         ("flows", "water", "pipe")),
    ])
    a.add_description("heat_flow", [
        ("cause", ("greater", ("temperature", "coffee"), ("temperature", "ice")),
         ("flows", "heat", "bar")),
    ])
    a.add_description("predator_prey", [
        ("hunts", "wolf", "rabbit"),
        ("greater", ("speed", "wolf"), ("speed", "rabbit")),
    ])
    # Probe: a new case the coordinator wants to retrieve analogues for.
    a.add_description("atom_model", [
        ("cause",
         ("attracts", "nucleus", "electron"),
         ("revolves_around", "electron", "nucleus")),
        ("greater", ("mass", "nucleus"), ("mass", "electron")),
    ])
    rep = a.retrieve("atom_model", k=3)
    print(f"  evaluated {rep.n_mac_evaluated} by MAC, "
          f"{rep.n_fac_evaluated} by FAC, in {rep.duration_s * 1000:.1f} ms")
    print(f"  top {len(rep.candidates)}:")
    for name, mac, fac, mp in rep.candidates:
        print(f"    {name:15s}  MAC={mac:.3f}  FAC={fac:.3f}")


# ----------------------------------------------------------------
# Scenario 4 — Karla-the-hawk ↔ Zerdia narrative transfer
# ----------------------------------------------------------------


def scenario_karla_zerdia() -> None:
    banner("Scenario 4 — Karla-the-hawk ↔ Zerdia (narrative transfer)")
    a = sme()
    # Base: Karla a hawk; Hunter wants feathers; Karla offers feathers
    # in exchange for not being shot.  Higher-order: causes, prevents.
    a.add_description("karla", [
        ("desires", "hunter", ("possess", "hunter", "feathers")),
        ("desires", "karla", ("not_killed", "karla")),
        ("cause",
            ("offer", "karla", "feathers", "hunter"),
            ("not_killed", "karla")),
    ])
    # Target: Zerdia, a country; another country wants its rare minerals;
    # Zerdia avoids invasion by exporting.
    a.add_description("zerdia", [
        ("desires", "imperium", ("possess", "imperium", "ore")),
        ("desires", "zerdia", ("not_killed", "zerdia")),
        ("cause",
            ("offer", "zerdia", "ore", "imperium"),
            ("not_killed", "zerdia")),
    ])
    rep = a.match("karla", "zerdia")
    print(f"  {rep.n_match_hypotheses} MHs, {len(rep.mappings)} gmaps")
    for i, m in enumerate(rep.mappings[:2]):
        _show_mapping(m, f"mapping #{i + 1}")


# ----------------------------------------------------------------
# Scenario 5 — proportional analogy (Copycat micro-domain)
# ----------------------------------------------------------------


def scenario_proportional() -> None:
    banner("Scenario 5 — letter-string proportional analogy (Copycat)")
    p = ProportionalAnalogy()
    cases = [
        ("abc", "abd", "ijk"),
        ("abc", "abd", "xyz"),
        ("abc", "bcd", "xyz"),
        ("aaa", "aaab", "xxx"),
        ("hello", "hella", "world"),
        ("ab", "ba", "cd"),
    ]
    print("  a:b :: c:? → answer (rule, score)")
    for a, b, c in cases:
        r = p.solve(a, b, c)
        print(f"    {a}:{b} :: {c}:{r.answer}  "
              f"({r.rule}, score={r.score:.2f})")


# ----------------------------------------------------------------
# Scenario 6 — ACME engine alternative
# ----------------------------------------------------------------


def scenario_acme() -> None:
    banner("Scenario 6 — ACME constraint-satisfaction-network engine")
    a = acme(iterations=80)
    a.add_description("solar", [
        ("cause",
         ("attracts", "sun", "planet"),
         ("revolves_around", "planet", "sun")),
        ("greater", ("mass", "sun"), ("mass", "planet")),
    ])
    a.add_description("atom", [
        ("cause",
         ("attracts", "nucleus", "electron"),
         ("revolves_around", "electron", "nucleus")),
        ("greater", ("mass", "nucleus"), ("mass", "electron")),
    ])
    rep = a.match("solar", "atom")
    print(f"  {rep.n_match_hypotheses} candidate MHs, "
          f"final SES = {rep.mappings[0].score:.2f}")
    _show_mapping(rep.mappings[0], "ACME mapping")


def main() -> None:
    scenario_solar_atom()
    scenario_water_heat()
    scenario_mac_fac()
    scenario_karla_zerdia()
    scenario_proportional()
    scenario_acme()
    print()
    print("=" * 72)
    print("done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
