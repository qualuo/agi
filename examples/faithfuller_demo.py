"""Faithfuller demo — chain-of-thought faithfulness in 30 lines.

Runs two synthetic streams (a faithful policy and an unfaithful one),
folds them through :class:`agi.faithfuller.Faithfuller`, prints the
certificate the coordination engine would dispatch on.

Run::

    python examples/faithfuller_demo.py
"""

from __future__ import annotations

from agi.faithfuller import (
    Faithfuller,
    FaithfullerConfig,
    synthetic_faithful_stream,
    synthetic_unfaithful_stream,
)


def _show(label: str, cert) -> None:
    print(f"\n{'='*70}\n{label}: verdict={cert.verdict}  recommendation={cert.recommendation}")
    print(f"  observations:        {cert.n_observations}")
    print(f"  truncation sens.:    {cert.truncation_sensitivity:6.3f}  "
          f"[{cert.truncation_ci_low:5.3f}, {cert.truncation_ci_high:5.3f}]")
    print(f"  bias following:      {cert.bias_following_rate:6.3f}  "
          f"[{cert.bias_following_ci_low:5.3f}, {cert.bias_following_ci_high:5.3f}]")
    print(f"  edit response:       {cert.edit_response_rate:6.3f}  "
          f"[{cert.edit_response_ci_low:5.3f}, {cert.edit_response_ci_high:5.3f}]")
    print(f"  self-inconsistency:  {cert.self_inconsistency_rate:6.3f}  "
          f"[{cert.self_inconsistency_ci_low:5.3f}, {cert.self_inconsistency_ci_high:5.3f}]")
    print(f"  filler advantage:    {cert.filler_advantage:+6.3f}  "
          f"[{cert.filler_advantage_ci_low:+5.3f}, {cert.filler_advantage_ci_high:+5.3f}]")
    print(f"  mediation gap:       {cert.mediation_gap:+6.3f}  "
          f"[{cert.mediation_ci_low:+5.3f}, {cert.mediation_ci_high:+5.3f}]")
    print(f"  product e-value:     {cert.product_evalue:.3g}")
    print(f"  holm rejected:       {list(cert.holm_rejected)}")
    print(f"  fingerprint:         {cert.fingerprint[:32]}...")


def main() -> None:
    print("Faithfuller — chain-of-thought faithfulness certification demo")

    # Faithful policy.
    cfg = FaithfullerConfig(policy_id="claude-faithful@safety-v3")
    ff = Faithfuller(cfg)
    for obs in synthetic_faithful_stream(192, seed=2026):
        ff.observe(obs)
    cert = ff.certify()
    _show("FAITHFUL POLICY", cert)

    # Unfaithful policy (post-hoc rationalisation).
    cfg2 = FaithfullerConfig(policy_id="claude-unfaithful@candidate-r1")
    ff2 = Faithfuller(cfg2)
    for obs in synthetic_unfaithful_stream(192, seed=2026):
        ff2.observe(obs)
    cert2 = ff2.certify()
    _show("UNFAITHFUL POLICY", cert2)

    # Report summary.
    print(f"\n{'='*70}\nReport.recent_observations (faithful, last 4):")
    for did, verdict, rejected in ff.report().recent_observations[-4:]:
        print(f"  {did}: {verdict} rejected={rejected}")


if __name__ == "__main__":
    main()
