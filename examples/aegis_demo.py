"""Aegis demo — multi-primitive safety certificate fusion at deployment time.

Wires the safety stack into one ship/no-ship decision:

  Faithfuller (CoT faithfulness)
  Elicitor    (capability elicitation)
  Goodharter  (proxy-reward drift)
                          → Aegis.absorb() → Aegis.certify() → SHIP / HOLD / DEGRADE / BLOCK

Run::

    python examples/aegis_demo.py
"""

from __future__ import annotations

from agi.aegis import (
    Aegis,
    AegisConfig,
    DECISION_BLOCK,
    DECISION_SHIP,
    SafetyCertificate,
    from_dataclass,
)
from agi.elicitor import (
    Elicitor,
    ElicitorConfig,
    synthetic_frontier_stream,
    synthetic_sandbag_stream,
)
from agi.faithfuller import (
    Faithfuller,
    FaithfullerConfig,
    synthetic_faithful_stream,
    synthetic_unfaithful_stream,
)
from agi.goodharter import (
    Goodharter,
    GoodharterConfig,
    synthetic_aligned_stream,
    synthetic_goodhart_stream,
)


def _print_decision(label, decision):
    print(f"\n{'='*70}\n{label}")
    print(f"  decision:               {decision.decision}")
    print(f"  severity:               {decision.severity}")
    print(f"  blocking primitive:     {decision.blocking_primitive}")
    print(f"  aggregated rec:         {decision.aggregated_recommendation}")
    print(f"  n_certificates:         {decision.n_certificates}")
    print(f"  product_evalue:         {decision.product_evalue:.3g}")
    print(f"  holm_rejected:          {list(decision.holm_rejected)}")
    print(f"  missing_required:       {list(decision.missing_required)}")
    print(f"  per-primitive:")
    for e in decision.per_primitive:
        line = (
            f"    {e.primitive:18s} verdict={e.raw_verdict!r:>16}  "
            f"severity={e.severity}"
        )
        if e.e_value is not None:
            line += f"  e={e.e_value:.3g}"
        if e.recommendation:
            line += f"  rec={e.recommendation!r}"
        print(line)
    print(f"  fingerprint:            {decision.fingerprint[:32]}...")


def _build_candidate_a():
    """Faithful, well-elicited, aligned proxy reward — the good case."""
    ff = Faithfuller(FaithfullerConfig(policy_id="claude-faithful@prod"))
    ff.observe_many(synthetic_faithful_stream(192, seed=11))

    el = Elicitor(
        ElicitorConfig(
            model_id="claude-faithful@prod",
            benchmark_id="aisi-frontier-v1",
            target_capability=0.70,
        )
    )
    el.observe_many(synthetic_frontier_stream(800, seed=11, method_sigma=0.08))

    gh = Goodharter(GoodharterConfig(proxy_id="claude-faithful@prod"))
    for obs in synthetic_aligned_stream(192, seed=11):
        gh.observe(obs)

    return ff.certify(), el.certify(), gh.certify()


def _build_candidate_b():
    """Unfaithful CoT, sandbagging, gaming proxy reward — the bad case."""
    ff = Faithfuller(FaithfullerConfig(policy_id="claude-rogue@candidate"))
    ff.observe_many(synthetic_unfaithful_stream(192, seed=11))

    el = Elicitor(
        ElicitorConfig(
            model_id="claude-rogue@candidate",
            benchmark_id="aisi-frontier-v1",
            target_capability=0.70,
        )
    )
    el.observe_many(synthetic_sandbag_stream(192, seed=11))

    gh = Goodharter(GoodharterConfig(proxy_id="claude-rogue@candidate"))
    for obs in synthetic_goodhart_stream(192, seed=11):
        gh.observe(obs)

    return ff.certify(), el.certify(), gh.certify()


def main() -> None:
    print("Aegis — multi-primitive safety certificate fusion gate demo")

    # -- Candidate A: ship-ready --------------------------------------
    ff_a, el_a, gh_a = _build_candidate_a()
    a = Aegis(
        AegisConfig(
            deployment_id="claude-faithful@prod",
            require_primitives=("faithfuller", "elicitor", "goodharter"),
        )
    )
    a.absorb(from_dataclass(ff_a, "faithfuller"))
    a.absorb(from_dataclass(el_a, "elicitor"))
    a.absorb(from_dataclass(gh_a, "goodharter"))
    _print_decision("Candidate A: claude-faithful@prod", a.certify())

    # -- Candidate B: must be blocked --------------------------------
    ff_b, el_b, gh_b = _build_candidate_b()
    b = Aegis(
        AegisConfig(
            deployment_id="claude-rogue@candidate",
            require_primitives=("faithfuller", "elicitor", "goodharter"),
        )
    )
    b.absorb(from_dataclass(ff_b, "faithfuller"))
    b.absorb(from_dataclass(el_b, "elicitor"))
    b.absorb(from_dataclass(gh_b, "goodharter"))
    _print_decision("Candidate B: claude-rogue@candidate", b.certify())

    # -- Candidate C: required primitive missing --------------------
    c = Aegis(
        AegisConfig(
            deployment_id="claude-untested",
            require_primitives=("faithfuller", "elicitor", "goodharter"),
        )
    )
    c.absorb(from_dataclass(ff_a, "faithfuller"))  # only one cert
    _print_decision("Candidate C: claude-untested (incomplete)", c.certify())

    # -- Audit fingerprints handed to governance --------------------
    print(f"\n{'='*70}\nAudit fingerprints (for governance ledger):")
    print(f"  candidate-A Aegis decision:    {a.last.fingerprint}")
    print(f"  candidate-B Aegis decision:    {b.last.fingerprint}")
    print(f"  candidate-C Aegis decision:    {c.last.fingerprint}")


if __name__ == "__main__":
    main()
