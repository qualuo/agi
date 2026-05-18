"""LatentReasoner demo — continuous-thought reasoning as a runtime primitive.

What this shows (in one runnable script, no API key required):

  1. Train a LatentReasoner from a tiny labelled dataset.
  2. Reason in latent space with greedy CoT and beam-of-K trajectories.
  3. Issue a Banach fixed-point convergence certificate + PAC-Bayes bound.
  4. Compose with the runtime: route the EventBus, pipe decoded
     posteriors through Reconciler-style Aumann agreement (manual sketch
     here because two LatentReasoners + an honest broker = an agreement
     demo on its own), and serialise to JSON for hot-reload by the
     coordination engine.

Run:  python examples/latent_reasoner_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.latent_reasoner import (
    LATENT_REASONED,
    LatentReasoner,
)


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


def section_train() -> LatentReasoner:
    banner("1. Train LatentReasoner — fit() on a tiny labelled dataset")
    # An ‘is this a yes/no question about water-related things’ classifier.
    train = [
        ("the sky is blue",          "yes"),
        ("the sun is hot",           "yes"),
        ("fire burns",               "yes"),
        ("the sky has clouds",       "yes"),
        ("the sun rises east",       "yes"),
        ("fire is dangerous",        "yes"),
        ("water is wet",             "no"),
        ("ice is cold",              "no"),
        ("snow falls in winter",     "no"),
        ("water flows downhill",     "no"),
        ("ice cracks under heat",    "no"),
        ("snow blankets the field",  "no"),
    ]
    lr = LatentReasoner(dim=48, anchors=("yes", "no"), seed=99,
                        learning_rate=0.1)
    summary = lr.fit(train, epochs=30)
    print(f"  trained on {summary['n_examples']} examples × {summary['epochs']} epochs")
    print(f"  final mean distance-to-anchor:  {summary['final_loss']:.4f}")
    print(f"  anchor coverage:                 {summary['anchor_coverage']:.0%}")
    print(f"  loss curve (first / mid / last): "
          f"{summary['loss_per_epoch'][0]:.3f} / "
          f"{summary['loss_per_epoch'][len(summary['loss_per_epoch'])//2]:.3f} / "
          f"{summary['loss_per_epoch'][-1]:.3f}")
    return lr


def section_greedy_reason(lr: LatentReasoner) -> None:
    banner("2. Greedy continuous-thought reasoning (beam = 1)")
    held_out = [
        ("the sun glows",         "yes"),
        ("the sky darkens at dusk", "yes"),
        ("fire crackles loudly",  "yes"),
        ("water freezes overnight", "no"),
        ("ice forms quickly",     "no"),
        ("snow covers everything", "no"),
    ]
    correct = 0
    for prompt, expected in held_out:
        r = lr.reason(prompt, beam=1, max_steps=10)
        ok = "✔" if r.answer == expected else "✗"
        if r.answer == expected:
            correct += 1
        print(f"  {ok}  {prompt!r:42}  "
              f"answer={r.answer}  conf={r.confidence:.2f}  "
              f"steps={r.n_steps}  γ̂={r.lipschitz:.3f}")
    print(f"\n  held-out accuracy: {correct}/{len(held_out)} = "
          f"{correct/len(held_out):.0%}")


def section_beam_reason(lr: LatentReasoner) -> None:
    banner("3. Beam-of-K trajectories (latent Tree-of-Thoughts)")
    prompt = "is the sun a star burning fire?"
    r = lr.reason(prompt, beam=4, max_steps=12)
    print(f"  prompt:     {prompt!r}")
    print(f"  winner:     answer={r.answer}  conf={r.confidence:.2f}  "
          f"entropy={r.entropy:.3f}  margin={r.margin:.3f}")
    print(f"  steps used: {r.n_steps}  γ̂={r.lipschitz:.3f}  "
          f"converged={r.converged}")
    print(f"  beams (sorted, winner first):")
    for i, t in enumerate(r.beams):
        print(f"    beam {i}: answer={t.answer}  conf={t.confidence:.2f}  "
              f"steps={t.steps}  γ̂={t.lipschitz:.3f}  "
              f"anchor_path={list(t.anchor_path)[:8]}{'…' if len(t.anchor_path) > 8 else ''}")


def section_certificate(lr: LatentReasoner) -> None:
    banner("4. Convergence + PAC-Bayes certificate")
    cert = lr.certificate()
    print(f"  γ̂ (Lipschitz on recent reasonings): {cert.gamma:.4f}")
    print(f"  ε (halting tolerance):              {cert.epsilon:.4f}")
    print(f"  anytime-valid Banach bound:         {cert.anytime_valid}")
    print(f"  PAC-Bayes bound on decode-KL (δ=0.05): {cert.pac_bayes_bound:.4f}")
    print(f"  anchor coverage:                    {cert.anchor_coverage:.0%}")
    print(f"  observations / reasonings:          "
          f"{cert.n_observations} / {cert.n_reasonings}")
    print(f"  chain head: {cert.chain_head[:24]}…")


def section_event_subscription() -> None:
    banner("5. EventBus integration — coordination engine subscribes")
    events: list[tuple[str, dict]] = []

    def cap(kind: str, data: dict) -> None:
        events.append((kind, data))

    lr = LatentReasoner(dim=16, anchors=("y", "n"), seed=7, publisher=cap)
    lr.fit([("affirmative input", "y"), ("negative input", "n")], epochs=3)
    lr.reason("subscribe demo")

    print(f"  captured {len(events)} events")
    kind_counts: dict[str, int] = {}
    for k, _ in events:
        kind_counts[k] = kind_counts.get(k, 0) + 1
    for k, n in sorted(kind_counts.items()):
        print(f"    {k:32} × {n}")

    # Show the one LATENT_REASONED payload — this is what the runtime
    # bus would forward to a coordination engine.
    for k, d in events:
        if k == LATENT_REASONED:
            print("\n  example latent_reasoner.reasoned payload:")
            print(f"    {json.dumps(d, indent=2)[:400]}")
            break


def section_hot_reload(lr: LatentReasoner) -> None:
    banner("6. JSON export → import_ round-trip (hot-reload by coordinator)")
    blob = lr.export()
    j = json.dumps(blob)
    print(f"  serialised size: {len(j):,} bytes")

    rebuilt = LatentReasoner.import_(json.loads(j))
    r1 = lr.reason("hot reload sanity check")
    r2 = rebuilt.reason("hot reload sanity check")
    print(f"  pre-export reason:  answer={r1.answer}  conf={r1.confidence:.3f}")
    print(f"  post-import reason: answer={r2.answer}  conf={r2.confidence:.3f}")
    print(f"  match: {r1.answer == r2.answer and abs(r1.confidence - r2.confidence) < 1e-6}")


def section_composition_sketch() -> None:
    banner("7. Composition sketch — two latents disagree, Aumann reconciles")
    # Two reasoners with the same training but different seeds.
    train = [
        ("blue or green", "yes"),
        ("orange or red", "no"),
        ("turquoise or teal", "yes"),
        ("vermillion or crimson", "no"),
    ]
    a = LatentReasoner(dim=24, anchors=("yes", "no"), seed=1)
    b = LatentReasoner(dim=24, anchors=("yes", "no"), seed=2)
    a.fit(train, epochs=15)
    b.fit(train, epochs=15)

    queries = ["azure or navy", "scarlet or maroon", "cyan or ruby"]
    for q in queries:
        ra = a.reason(q, max_steps=8)
        rb = b.reason(q, max_steps=8)
        # Aumann-style honest broker: average the two posteriors.
        merged = [(ra.distribution[i] + rb.distribution[i]) / 2.0
                  for i in range(2)]
        anchors = ("yes", "no")
        winner = anchors[merged.index(max(merged))]
        agree = "AGREE" if ra.answer == rb.answer else "SPLIT"
        print(f"  {q!r:24}  A={ra.answer}({ra.confidence:.2f})  "
              f"B={rb.answer}({rb.confidence:.2f})  {agree}  "
              f"→ merged={winner}({max(merged):.2f})")


def main() -> None:
    print("LatentReasoner demo — continuous chain-of-thought as a runtime "
          "primitive\n"
          "  paper: Hao et al. 2024, 'Training LLMs to Reason in a Continuous "
          "Latent Space'\n"
          "  here:  same idea, pure stdlib, anytime-valid Banach + PAC-Bayes "
          "certificates")
    lr = section_train()
    section_greedy_reason(lr)
    section_beam_reason(lr)
    section_certificate(lr)
    section_event_subscription()
    section_hot_reload(lr)
    section_composition_sketch()
    print("\ndone.")


if __name__ == "__main__":
    main()
