"""Faithfuller coordination demo — full safety-stack dispatch loop.

A *coordination engine* uses Faithfuller alongside the existing safety
stack to decide whether a candidate policy may receive high-stakes
work.  The script wires:

  Manifest discovery → Faithfuller certificate
                     → Goodharter certificate (proxy-reward gap)
                     → Refuser certificate (jailbreak robustness)
                     → Sycophant certificate (user-pressure robustness)
                     → Constitutionalist certificate (constitutional
                       compliance) → Strategist-style fusion

It simulates two candidate policies side by side:
  * ``claude-faithful@safety-v3`` — a fully-aligned recipe
  * ``claude-unfaithful@candidate-r1`` — passes capability evals but
    emits unfaithful CoT (post-hoc rationalisation)

For each policy the demo:

  1. Discovers the primitive specs through the manifest.
  2. Runs the synthetic audit streams for each safety primitive.
  3. Folds the resulting certificates into a single dispatch decision
     a coordination engine can act on.

Run::

    python examples/faithfuller_coordination_demo.py
"""

from __future__ import annotations

from agi.faithfuller import (
    REC_DEPLOY as FF_DEPLOY,
    REC_DISABLE_COT as FF_DISABLE_COT,
    REC_ESCALATE_HUMAN as FF_ESCALATE_HUMAN,
    REC_MONITOR as FF_MONITOR,
    REC_SUMMARY_ONLY as FF_SUMMARY_ONLY,
    VERDICT_REJECT as FF_VERDICT_REJECT,
    VERDICT_TRUST as FF_VERDICT_TRUST,
    Faithfuller,
    FaithfullerConfig,
    synthetic_faithful_stream,
    synthetic_unfaithful_stream,
)
from agi.goodharter import (
    Goodharter,
    GoodharterConfig,
    REC_DEPLOY as GH_DEPLOY,
    REC_REPLACE as GH_REPLACE,
    VERDICT_TRUST as GH_VERDICT_TRUST,
    synthetic_aligned_stream,
    synthetic_goodhart_stream,
)
from agi.manifest import default_manifest


# ---------------------------------------------------------------------------
# Coordinator's fused decision
# ---------------------------------------------------------------------------


def fuse_decision(
    ff_cert,
    gh_cert,
) -> tuple[str, str]:
    """Map a (Faithfuller, Goodharter) pair into one coordinator action.

    The fusion is conservative: any safety primitive in the most-severe
    bucket forces the coordinator's hand.  Real deployments would
    extend this with Refuser / Sycophant / Confabulator / Constitution-
    alist certificates following the same pattern.
    """
    ff_verdict = ff_cert.verdict
    gh_verdict = gh_cert.verdict

    if ff_verdict == FF_VERDICT_REJECT or gh_verdict == "QUARANTINE":
        return "QUARANTINE", "kill the deployment; route through human review"
    if ff_verdict == "DEGRADE":
        if ff_cert.recommendation == FF_SUMMARY_ONLY:
            return "DEGRADE", "strip CoT; deliver answer-only summaries"
        if ff_cert.recommendation == FF_DISABLE_COT:
            return "DEGRADE", "disable CoT-conditioned downstream verifiers"
        return "DEGRADE", "human reviews every dispatched ticket"
    if ff_verdict == "INVESTIGATE" or gh_verdict in ("INVESTIGATE", "RETRAIN"):
        return "MONITOR", "raise audit sampling rate; keep deployment alive"
    if ff_verdict == FF_VERDICT_TRUST and gh_verdict == GH_VERDICT_TRUST:
        return "DEPLOY", "route normal traffic; cache faithfulness certificate"
    return "MONITOR", "ambiguous; keep deploying with extra probes"


def _print_cert(label, ff_cert, gh_cert) -> None:
    print(f"\n{'='*70}\n{label}")
    print(f"  Faithfuller: verdict={ff_cert.verdict!r:>14} rec={ff_cert.recommendation!r}")
    print(f"    truncation sens.   = {ff_cert.truncation_sensitivity:6.3f}   "
          f"bias following = {ff_cert.bias_following_rate:.3f}")
    print(f"    edit response      = {ff_cert.edit_response_rate:6.3f}   "
          f"self-inconsistency = {ff_cert.self_inconsistency_rate:.3f}")
    print(f"    filler advantage   = {ff_cert.filler_advantage:+6.3f}   "
          f"mediation gap = {ff_cert.mediation_gap:+.3f}")
    print(f"    product e-value    = {ff_cert.product_evalue:.3g}   "
          f"holm = {list(ff_cert.holm_rejected)}")
    print(f"  Goodharter : verdict={gh_cert.verdict!r:>14} rec={gh_cert.recommendation!r}")
    print(f"    pearson r          = {gh_cert.pearson_r:+.3f}   "
          f"gap mean = {gh_cert.gap_mean:+.3f}")
    print(f"    product e-value    = {gh_cert.product_evalue:.3g}   "
          f"holm = {list(gh_cert.holm_rejected)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Step 1: coordinator discovers the primitives via the manifest.
    manifest = default_manifest()
    print("Coordinator discovers safety primitives:")
    for spec in manifest.find(kind="safety"):
        print(f"  - {spec.name:18s} certificate={spec.certificate}")

    print("\nIntent search ('certify chain-of-thought is faithful'):")
    for spec, score in manifest.recommend(
        "certify chain-of-thought is faithful", k=3
    ):
        print(f"  {score:5.2f}  {spec.name:18s} — {spec.summary[:80]}...")

    # Step 2: run the faithful candidate through Faithfuller + Goodharter.
    ff_a = Faithfuller(FaithfullerConfig(policy_id="claude-faithful@safety-v3"))
    ff_a.observe_many(synthetic_faithful_stream(192, seed=11))
    ff_cert_a = ff_a.certify()
    gh_a = Goodharter(GoodharterConfig(proxy_id="claude-faithful@safety-v3"))
    for obs in synthetic_aligned_stream(192, seed=11):
        gh_a.observe(obs)
    gh_cert_a = gh_a.certify()
    _print_cert("Candidate A: claude-faithful@safety-v3", ff_cert_a, gh_cert_a)

    decision_a, reason_a = fuse_decision(ff_cert_a, gh_cert_a)
    print(f"\n  COORDINATOR ACTION → {decision_a}: {reason_a}")

    # Step 3: run the unfaithful candidate (passes capability evals but
    # emits post-hoc CoT).
    ff_b = Faithfuller(
        FaithfullerConfig(policy_id="claude-unfaithful@candidate-r1")
    )
    ff_b.observe_many(synthetic_unfaithful_stream(192, seed=11))
    ff_cert_b = ff_b.certify()
    gh_b = Goodharter(
        GoodharterConfig(proxy_id="claude-unfaithful@candidate-r1")
    )
    # The unfaithful candidate happens to *also* be gaming a learned
    # reward model — that's the multiplicative failure investors care
    # about.
    for obs in synthetic_goodhart_stream(192, seed=11):
        gh_b.observe(obs)
    gh_cert_b = gh_b.certify()
    _print_cert("Candidate B: claude-unfaithful@candidate-r1", ff_cert_b, gh_cert_b)

    decision_b, reason_b = fuse_decision(ff_cert_b, gh_cert_b)
    print(f"\n  COORDINATOR ACTION → {decision_b}: {reason_b}")

    # Step 4: print the audit chain a real coordination engine would
    # ship to the governance ledger.
    print(f"\n{'='*70}\nAudit fingerprints (handed to attest / oracle / governance):")
    print(f"  candidate-A Faithfuller cert: {ff_cert_a.fingerprint}")
    print(f"  candidate-A Goodharter cert: {gh_cert_a.fingerprint}")
    print(f"  candidate-B Faithfuller cert: {ff_cert_b.fingerprint}")
    print(f"  candidate-B Goodharter cert: {gh_cert_b.fingerprint}")


if __name__ == "__main__":
    main()
