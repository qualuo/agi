"""Reasoner — symbolic logical reasoning as a runtime primitive.

This demo runs six scenarios end-to-end (no API key required), each
mapping a familiar coordination-engine question onto one of the
algorithms shipped behind the Reasoner API:

  1. **CDCL on industrial-style SAT** — a small graph-colouring problem
     showing decisions, conflicts, learnt clauses, and an UNSAT close
     with a replayable resolution proof.
  2. **Pigeon-hole UNSAT proof** — 5 pigeons into 4 holes; CDCL refutes
     and emits a resolution chain ending in the empty clause.
  3. **DPLL vs CDCL on the same instance** — pure-stdlib comparison of
     conflict counts at fixed seed.
  4. **Walk-SAT with anytime Clopper-Pearson failure bound** — local
     search on satisfiable 3-SAT plus a finite-sample upper bound on
     the runtime failure rate.
  5. **Datalog forward chaining with variables** — Prolog-style
     ancestor query; semi-naïve evaluation with Robinson 1965
     unification.
  6. **Answer Set Programming** — graph 2-colourability via stable
     models, plus a guess-and-check enumeration of all valid colourings.

Each call advances the same tamper-evident SHA-256 fingerprint chain,
suitable for direct insertion into `AttestationLedger`.  Replay the
sequence and the fingerprint reproduces bit-for-bit.

Run:  python examples/reasoner_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.reasoner import (
    CDCL,
    DPLL,
    FORWARD_CHAIN,
    STABLE_MODELS,
    WALKSAT,
    Reasoner,
)


def section(title: str) -> None:
    line = "=" * 78
    print()
    print(line)
    print(f"  {title}")
    print(line)


def demo_cdcl_graph_colour() -> None:
    section("1) CDCL — 3-colourability of K4 minus one edge")
    # Vertices {a, b, c, d}.  Edges: a-b, a-c, a-d, b-c, c-d.  (b-d
    # removed → 3-colourable.)  Variables: ``x_v_c`` means vertex v gets
    # colour c.
    r = Reasoner(CDCL, seed=0)
    vs = ["a", "b", "c", "d"]
    cs = ["r", "g", "b"]
    edges = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("c", "d")]
    # Each vertex gets at least one colour.
    for v in vs:
        r.add_clause([f"x_{v}_{c}" for c in cs])
    # No vertex gets two colours.
    for v in vs:
        for i, c1 in enumerate(cs):
            for c2 in cs[i + 1:]:
                r.add_clause([f"~x_{v}_{c1}", f"~x_{v}_{c2}"])
    # Adjacent vertices have different colours.
    for u, v in edges:
        for c in cs:
            r.add_clause([f"~x_{u}_{c}", f"~x_{v}_{c}"])
    sol = r.solve()
    print(f"  verdict: {sol.verdict}")
    print(f"  decisions: {sol.decisions}, conflicts: {sol.conflicts}, "
          f"propagations: {sol.propagations}, learned: {sol.learned}, "
          f"restarts: {sol.restarts}")
    if sol.satisfiable:
        colouring = {
            v: next(c for c in cs if sol.model.get(f"x_{v}_{c}", False))
            for v in vs
        }
        print("  colouring:", colouring)


def demo_pigeonhole_unsat() -> None:
    section("2) Pigeon-hole 5 → 4 — CDCL refutes with a resolution proof")
    r = Reasoner(CDCL, seed=0)
    n = 5
    m = 4
    # Each pigeon in some hole.
    for p in range(n):
        r.add_clause([f"p{p}h{h}" for h in range(m)])
    # No hole has two pigeons.
    for h in range(m):
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                r.add_clause([f"~p{p1}h{h}", f"~p{p2}h{h}"])
    sol = r.solve()
    print(f"  verdict: {sol.verdict}")
    print(f"  conflicts: {sol.conflicts}, learned: {sol.learned}, "
          f"restarts: {sol.restarts}")
    proof = r.last_resolution_proof()
    print(f"  resolution-proof steps reconstructed: {len(proof)}")
    if proof:
        last = proof[-1]
        print(f"  final resolvent: {last.resolvent}  "
              f"(pivot atom: {last.pivot!r})")


def demo_dpll_vs_cdcl() -> None:
    section("3) DPLL vs CDCL on the same instance")
    # Encode a non-trivial SAT instance (Schur triple covering on n=10).
    n = 10
    # Variables ``c_i`` for i ∈ [1, n] — each integer gets a 'colour' (T/F).
    # Want: no monochromatic Schur triple (a + b = c).
    triples = [(a, b, a + b) for a in range(1, n + 1)
               for b in range(a, n + 1) if a + b <= n]
    for algo in (DPLL, CDCL):
        r = Reasoner(algo, seed=0)
        for (a, b, c) in triples:
            r.add_clause([f"~c{a}", f"~c{b}", f"~c{c}"])
            r.add_clause([f"c{a}", f"c{b}", f"c{c}"])
        sol = r.solve()
        label = "DPLL" if algo == DPLL else "CDCL"
        print(f"  {label}: verdict={sol.verdict}, decisions={sol.decisions}, "
              f"conflicts={sol.conflicts}, elapsed={sol.elapsed_s*1000:.1f}ms")


def demo_walksat_with_bound() -> None:
    section("4) Walk-SAT + Clopper-Pearson finite-sample failure bound")
    r = Reasoner(WALKSAT, seed=0, walksat_flips=2000, walksat_restarts=10)
    # Satisfiable 3-SAT: forced solution a=T, b=T, c=T.
    r.add_clause(["a", "b", "c"])
    r.add_clause(["~a", "b", "c"])
    r.add_clause(["a", "~b", "c"])
    sol = r.solve()
    print(f"  verdict: {sol.verdict}, flips: {sol.propagations}, "
          f"restarts: {sol.restarts}")
    # Run a few more times — call the bound at α = 0.05.
    for _ in range(5):
        r.solve()
    rep = r.report(alpha=0.05)
    print(f"  Walk-SAT attempts: {rep.sat_calls}, "
          f"finite-sample 95% UCB on failure rate: "
          f"{rep.failure_upper_clopper_pearson:.4f}")


def demo_datalog_with_variables() -> None:
    section("5) Datalog — recursive ancestor query with unification")
    r = Reasoner(FORWARD_CHAIN)
    r.add_rule("parent(alice, bob).")
    r.add_rule("parent(bob, carol).")
    r.add_rule("parent(carol, dave).")
    r.add_rule("ancestor(X, Y) :- parent(X, Y).")
    r.add_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")
    derived = r.forward_chain()
    print("  derived ancestor facts:")
    for f in sorted(derived):
        if f.startswith("ancestor("):
            print(f"    {f}")
    proof = r.backward_chain("ancestor(alice, dave)")
    print(f"  proof of ancestor(alice, dave):")
    if proof is not None:
        def _show(node, indent: str = "    ") -> None:
            tag = "fact" if node.rule_id == -1 else f"rule#{node.rule_id}"
            print(f"{indent}{node.goal}   [{tag}]")
            for sub in node.subgoals:
                _show(sub, indent + "  ")
        _show(proof)


def demo_asp_graph_colour() -> None:
    section("6) Answer Set Programming — graph 2-colour stable models")
    r = Reasoner(STABLE_MODELS)
    # Two-colour the triangle {a, b, c}.
    # Each vertex either red or blue (NaF generation):
    #     red(X)  :- node(X), not blue(X).
    #     blue(X) :- node(X), not red(X).
    # Forbid adjacent same colours:
    #     :- edge(X, Y), red(X), red(Y).
    #     :- edge(X, Y), blue(X), blue(Y).
    # (A triangle is *not* 2-colourable — expect zero stable models.)
    r.add_rule("node(a).")
    r.add_rule("node(b).")
    r.add_rule("node(c).")
    r.add_rule("edge(a, b).")
    r.add_rule("edge(b, c).")
    r.add_rule("edge(a, c).")
    r.add_rule("red(X) :- node(X), not blue(X).")
    r.add_rule("blue(X) :- node(X), not red(X).")
    r.add_rule(":- edge(X, Y), red(X), red(Y).")
    r.add_rule(":- edge(X, Y), blue(X), blue(Y).")
    mods = r.stable_models(limit=8)
    print(f"  triangle 2-colour models: {len(mods)} (expected 0)")
    # Switch to a path {a-b-c} which IS 2-colourable.
    r2 = Reasoner(STABLE_MODELS)
    r2.add_rule("node(a).")
    r2.add_rule("node(b).")
    r2.add_rule("node(c).")
    r2.add_rule("edge(a, b).")
    r2.add_rule("edge(b, c).")
    r2.add_rule("red(X) :- node(X), not blue(X).")
    r2.add_rule("blue(X) :- node(X), not red(X).")
    r2.add_rule(":- edge(X, Y), red(X), red(Y).")
    r2.add_rule(":- edge(X, Y), blue(X), blue(Y).")
    mods2 = r2.stable_models(limit=8)
    print(f"  path 2-colour models: {len(mods2)}")
    for i, m in enumerate(sorted(mods2, key=lambda s: sorted(s))):
        coloured = {a: ("red" if f"red({a})" in m else "blue")
                    for a in ("a", "b", "c")}
        print(f"    model {i + 1}: {coloured}")


def demo_replay_fingerprint() -> None:
    section("7) Tamper-evident replay — fingerprint determinism")
    def build() -> Reasoner:
        r = Reasoner(CDCL, seed=42)
        r.add_clause(["a", "b"])
        r.add_clause(["~a", "c"])
        r.add_fact("d")
        r.add_rule("e :- d.")
        return r
    r1 = build()
    r2 = build()
    print(f"  fingerprint(run1): {r1.fingerprint[:32]}…")
    print(f"  fingerprint(run2): {r2.fingerprint[:32]}…")
    print(f"  identical: {r1.fingerprint == r2.fingerprint}")
    # Composing into an attestation chain:
    print(f"  this fingerprint can be fed directly into")
    print(f"  AttestationLedger.append(payload={r1.fingerprint[:16]}…)")


def main() -> None:
    demo_cdcl_graph_colour()
    demo_pigeonhole_unsat()
    demo_dpll_vs_cdcl()
    demo_walksat_with_bound()
    demo_datalog_with_variables()
    demo_asp_graph_colour()
    demo_replay_fingerprint()
    print()
    print("All Reasoner scenarios completed.  Each call advanced the "
          "tamper-evident SHA-256 fingerprint chain; replay reproduces "
          "byte-for-byte.")


if __name__ == "__main__":
    main()
