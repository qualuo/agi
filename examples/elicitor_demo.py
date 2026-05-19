"""Elicitor demo — capability elicitation with PAC certificates.

Three synthetic models — a well-elicited frontier model, a sandbagging
model, and an underperforming model — folded through
:class:`agi.elicitor.Elicitor`.

Run::

    python examples/elicitor_demo.py
"""

from __future__ import annotations

from agi.elicitor import (
    Elicitor,
    ElicitorConfig,
    synthetic_frontier_stream,
    synthetic_sandbag_stream,
    synthetic_under_stream,
)


def _print(label, cert):
    print(f"\n{'='*70}\n{label}: verdict={cert.verdict}  recommendation={cert.recommendation}")
    print(f"  observations:           {cert.n_observations}")
    print(f"  methods elicited:       {cert.n_methods}")
    print(f"  elicited capability:    {cert.elicited_capability:.3f}  via {cert.elicited_method_id!r}")
    print(f"  capability UCB / LCB:   {cert.capability_ucb:.3f}  /  {cert.capability_lcb:.3f}")
    print(f"  sandbag e-value:        {cert.sandbag_score:.3g}  (reject if > {cert.sandbag_threshold:.1f})")
    print(f"  fingerprint:            {cert.fingerprint[:32]}...")
    print(f"  per-method (top 3):")
    for r in cert.per_method[:3]:
        print(f"    {r.method_id:25s} n={r.n:4d} mean={r.mean:.3f} var={r.var:.3f} "
              f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}] cost=${r.total_cost_usd:.2f}")


def main() -> None:
    print("Elicitor — capability elicitation with PAC certificates demo")

    cfg = ElicitorConfig(
        model_id="claude-frontier",
        benchmark_id="aisi-frontier-v1",
        target_capability=0.70,
    )
    el = Elicitor(cfg)
    for obs in synthetic_frontier_stream(800, seed=2026, method_sigma=0.08):
        el.observe(obs)
    _print("FRONTIER (claude-frontier)", el.certify())

    cfg = ElicitorConfig(model_id="claude-sandbagger", benchmark_id="aisi-frontier-v1")
    el2 = Elicitor(cfg)
    for obs in synthetic_sandbag_stream(192, seed=2026):
        el2.observe(obs)
    _print("SANDBAGGER (claude-sandbagger)", el2.certify())

    cfg = ElicitorConfig(model_id="claude-underperforming", benchmark_id="aisi-frontier-v1")
    el3 = Elicitor(cfg)
    for obs in synthetic_under_stream(192, seed=2026):
        el3.observe(obs)
    _print("UNDERPERFORMING (claude-underperforming)", el3.certify())


if __name__ == "__main__":
    main()
