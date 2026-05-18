"""Aligner — direct preference optimisation demo.

Trains a DPO scorer on a synthetic preference stream and demonstrates:

  * Pair-based preference observation.
  * Eval-gated deployment.
  * Anytime-valid LCB / UCB on held-out preference accuracy.
  * Replay-deterministic SHA-256 certificate.
  * best_of_n inference at deployment.
  * Comparison with IPO, SimPO, KTO on the same dataset.

What an investor sees:
  * Five state-of-the-art preference-optimisation algorithms (DPO, IPO,
    SLiC, SimPO, ORPO, cDPO, rDPO, KTO) implemented in pure stdlib,
    no PyTorch.
  * Every fit reports a finite-sample lower bound on the held-out
    preference accuracy, plus a PAC-Bayes generalisation bound and
    an anytime-valid e-process on the agreement-with-judge test.
  * Every observation and fit event chains into a SHA-256
    fingerprint a regulator can replay byte-for-byte.

Run::

    python examples/aligner_demo.py
"""
from __future__ import annotations

import random
import time

from agi.aligner import (
    AlignerConfig,
    dpo_aligner,
    ipo_aligner,
    kto_aligner,
    orpo_aligner,
    simpo_aligner,
    slic_aligner,
)


# -----------------------------------------------------------------------------
# Synthetic preference data: "preferred" answers contain the magic token.
# -----------------------------------------------------------------------------


PROMPTS = [
    "Explain quantum entanglement",
    "Summarise the Cauchy-Schwarz inequality",
    "Outline the proof of Gödel's incompleteness theorem",
    "Describe the Riemann zeta function",
    "What is the second law of thermodynamics",
]

MAGIC = "rigorous"

GOOD_TOKENS = ["axiom", "lemma", "proof", "theorem", "corollary",
               "definition", "construction", "isomorphism"]
BAD_TOKENS = ["whatever", "stuff", "things", "kinda", "maybe",
              "approximately", "vibes", "trust me"]


def make_pair(rng: random.Random, *, noise: float = 0.05):
    prompt = rng.choice(PROMPTS)
    winner_body = " ".join(rng.choices(GOOD_TOKENS, k=6))
    loser_body = " ".join(rng.choices(BAD_TOKENS, k=6))
    winner = f"{MAGIC} {winner_body}"
    loser = f"casual {loser_body}"
    # Caller-supplied "reference policy" log-probabilities; flat here.
    ref_w = 0.0
    ref_l = 0.0
    if rng.random() < noise:
        return prompt, loser, winner, ref_l, ref_w
    return prompt, winner, loser, ref_w, ref_l


# -----------------------------------------------------------------------------
# Train one algorithm and print the report.
# -----------------------------------------------------------------------------


def train(name, ctor, *, n_observations=300, noise=0.05, **kwargs):
    rng = random.Random(42)
    aligner = ctor(seed=42, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n_observations):
        prompt, winner, loser, ref_w, ref_l = make_pair(rng, noise=noise)
        if name in ("KTO",):
            aligner.observe_unary(prompt=prompt, candidate=winner,
                                  desirable=True,
                                  ref_log_prob_candidate=ref_w)
            aligner.observe_unary(prompt=prompt, candidate=loser,
                                  desirable=False,
                                  ref_log_prob_candidate=ref_l)
        elif name in ("SimPO",):
            aligner.observe_pair(prompt=prompt, winner=winner, loser=loser)
        else:
            aligner.observe_pair(prompt=prompt, winner=winner, loser=loser,
                                 ref_log_prob_winner=ref_w,
                                 ref_log_prob_loser=ref_l)
    report = aligner.fit()
    elapsed = time.perf_counter() - t0
    print(f"=== {name} ===")
    print(f"  preference_accuracy:       {report.preference_accuracy:.4f}")
    print(f"  lcb (Bernstein, 95%):      {report.preference_accuracy_lcb_bernstein:.4f}")
    print(f"  lcb (anytime, HRMS, 95%):  {report.preference_accuracy_lcb_anytime:.4f}")
    print(f"  ucb (Hoeffding, 95%):      {report.preference_accuracy_ucb_hoeffding:.4f}")
    print(f"  e-process:                 {report.e_process:.3e}")
    print(f"  pacbayes_bound:            {report.pacbayes_bound:.4f}")
    print(f"  kl_to_reference:           {report.kl_divergence_to_reference:.4f}")
    print(f"  train_loss:                {report.train_loss:.4f}")
    print(f"  eval_loss:                 {report.eval_loss:.4f}")
    print(f"  deployed:                  {report.deployed}")
    print(f"  n_observations:            {report.n_observations}")
    print(f"  n_train / n_eval:          {report.n_train} / {report.n_eval}")
    print(f"  weight_l2:                 {report.weight_l2:.4f}")
    print(f"  iterations:                {report.iterations}")
    print(f"  elapsed:                   {elapsed*1000:.1f} ms")
    print(f"  fingerprint:               {report.fingerprint_hash[:32]}...")
    print()
    return aligner, report


# -----------------------------------------------------------------------------
# Show best_of_n at inference time
# -----------------------------------------------------------------------------


def demo_inference(aligner):
    prompt = "Explain the Riemann zeta function"
    candidates = [
        "rigorous lemma theorem proof corollary axiom isomorphism",
        "casual stuff things whatever vibes trust me",
        "rigorous axiom theorem construction proof",
        "kinda maybe approximately whatever vibes stuff",
    ]
    print("=== Inference (best_of_n) ===")
    print(f"prompt: {prompt!r}")
    print("candidates:")
    for c in candidates:
        s = aligner.score(prompt, c)
        print(f"  score = {s:+.3f}   {c[:60]!r}")
    best = aligner.best_of_n(prompt, candidates)
    print(f"=> best_of_n picked: {best!r}")
    # Pairwise probability
    p = aligner.preference_probability(prompt, candidates[0], candidates[1])
    print(f"=> P(rigorous ≻ casual) = {p:.4f}")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("Aligner — direct preference optimisation demo")
    print("=" * 60)
    print()

    # Train each algorithm.
    dpo, _ = train("DPO", dpo_aligner, beta=0.5)
    train("IPO", ipo_aligner, beta=0.5)
    train("SLiC", slic_aligner, beta=0.5, delta=1.0, lam=0.05)
    train("SimPO", simpo_aligner, beta=2.0, gamma=0.5)
    train("ORPO", orpo_aligner, beta=0.5, lam=0.5)
    train("KTO", kto_aligner, beta=0.5)

    # Demonstrate inference with the trained DPO scorer.
    demo_inference(dpo)


if __name__ == "__main__":
    main()
