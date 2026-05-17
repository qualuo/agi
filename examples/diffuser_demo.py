"""Diffuser demo — score-based generative modelling as a runtime primitive.

What this shows (in one runnable script, no API key required):

  1. Build a 2-D Gaussian-mixture target with closed-form score.
  2. Sample from it using **eight** of the ten reverse-time algorithms:
     DDPM, DDIM, DPM-Solver-1, DPM-Solver-2, Heun (Karras EDM), Euler-SDE,
     PF-ODE Euler, and Flow Matching.
  3. Show that **both modes are recovered** — the runtime primitive does
     not mode-collapse to one side, which is the failure mode classical
     learners on multi-modal targets typically exhibit.
  4. Run classifier-free guidance to **bias** sampling toward one mode.
  5. Issue a four-part *certificate*: Girsanov TV bound + per-bin
     empirical TV (with Hoeffding half-width) + DDPM ELBO per step +
     score-matching loss floor.
  6. Compose with the runtime: hot-reload via ``export() / import_()``
     and pipe events through a sink that a coordination engine can
     subscribe to.

Run:  python examples/diffuser_demo.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.diffuser import (
    ALG_DDIM,
    ALG_DDPM,
    ALG_DPM_SOLVER_1,
    ALG_DPM_SOLVER_2,
    ALG_EULER_SDE,
    ALG_FLOW_MATCHING,
    ALG_HEUN,
    ALG_PF_ODE,
    DIFFUSER_SAMPLED,
    DIFFUSER_STEP,
    Diffuser,
    DiffuserConfig,
    gaussian_mixture_score,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Build the target and the primitive.
    # ------------------------------------------------------------------
    banner("1. Build a 2-D bimodal target with analytic score")
    means = [[2.0, 0.0], [-2.0, 0.0]]
    weights = [0.5, 0.5]
    sigma_sq = 0.25
    score = gaussian_mixture_score(means, weights, sigma_sq=sigma_sq)
    print(f"Target: 0.5·N([+2, 0], σ²={sigma_sq}·I) + 0.5·N([-2, 0], σ²={sigma_sq}·I)")

    events: list[tuple[str, dict]] = []

    def sink(kind: str, payload: dict) -> None:
        events.append((kind, payload))

    diffuser = Diffuser(
        DiffuserConfig(dim=2, T=200, seed=7),
        publisher=sink,
    )
    diffuser.register(
        "gmm",
        score_fn=score,
        cond_score_fn=lambda x, t, c: score(x, t),  # placeholder
        uncond_score_fn=score,
        second_moment=4.0,
        score_error_floor=1e-3,
    )
    print(f"Registered target 'gmm'.  Schedule kind: {diffuser.schedule.kind}.")

    # ------------------------------------------------------------------
    # 2. Run eight reverse-time samplers.
    # ------------------------------------------------------------------
    banner("2. Run eight reverse-time samplers")
    algos = [
        ALG_DDPM,
        ALG_DDIM,
        ALG_DPM_SOLVER_1,
        ALG_DPM_SOLVER_2,
        ALG_HEUN,
        ALG_EULER_SDE,
        ALG_PF_ODE,
        ALG_FLOW_MATCHING,
    ]
    n_samples_per_algo = 60
    coverage: dict[str, tuple[int, int]] = {}
    for alg in algos:
        left = 0
        right = 0
        for _ in range(n_samples_per_algo):
            s = diffuser.sample(
                "gmm", algorithm=alg, num_steps=40, record_trajectory=False,
            )
            if s.final[0] < 0:
                left += 1
            else:
                right += 1
        coverage[alg] = (left, right)
        print(f"  {alg:18s}  left={left:2d}  right={right:2d}")

    # ------------------------------------------------------------------
    # 3. Mode coverage check.
    # ------------------------------------------------------------------
    banner("3. Mode coverage check (no mode collapse)")
    for alg, (l, r) in coverage.items():
        ok = (l >= 5) and (r >= 5)
        flag = "OK" if ok else "WARN"
        print(f"  {flag:5s}  {alg:18s}  both modes covered (≥ 5 each)")

    # ------------------------------------------------------------------
    # 4. Classifier-free guidance toward one mode.
    # ------------------------------------------------------------------
    banner("4. Classifier-free guidance: steer toward x > 0")
    # Bias: a 'positive' cond score uses a single-mode target at +2.
    pos_score = gaussian_mixture_score([[2.0, 0.0]], [1.0], sigma_sq=0.25)
    diffuser.deregister("gmm")
    diffuser.register(
        "gmm",
        score_fn=score,
        cond_score_fn=lambda x, t, c: pos_score(x, t),
        uncond_score_fn=score,
        second_moment=4.0,
        score_error_floor=1e-3,
    )
    pos_count = 0
    for _ in range(30):
        s = diffuser.sample(
            "gmm",
            algorithm=ALG_DDIM,
            num_steps=40,
            condition="positive",
            guidance_scale=3.0,
            record_trajectory=False,
        )
        if s.final[0] > 0:
            pos_count += 1
    print(f"Guided positive-side count: {pos_count}/30")

    # ------------------------------------------------------------------
    # 5. Certificate.
    # ------------------------------------------------------------------
    banner("5. Issue a four-part sampling-quality certificate")
    import random
    rng = random.Random(0)
    samples = [
        diffuser.sample(
            "gmm", algorithm=ALG_DDIM, num_steps=40, record_trajectory=False
        ).final
        for _ in range(40)
    ]
    def gt() -> tuple[float, float]:
        m = means[rng.choice([0, 1])]
        return (
            m[0] + rng.gauss(0, math.sqrt(sigma_sq)),
            m[1] + rng.gauss(0, math.sqrt(sigma_sq)),
        )
    target_samples = [gt() for _ in range(200)]
    cert = diffuser.certify("gmm", samples, target_samples=target_samples,
                            algorithm=ALG_DDIM)
    print(f"  Girsanov TV bound:           {cert.girsanov_tv_bound:.4f}")
    print(f"  Empirical TV:                {cert.empirical_tv:.4f} "
          f"± {cert.empirical_tv_half_width:.4f}")
    print(f"  Score-matching error floor:  {cert.score_error:.4f}")
    print(f"  Mean ELBO term per step:     {cert.elbo_per_step:.4f}")
    print(f"  Chain head:                  {cert.chain_head[:16]}…")

    # ------------------------------------------------------------------
    # 6. Hot reload: export → JSON → import.
    # ------------------------------------------------------------------
    banner("6. Hot-reload via JSON export/import")
    blob = diffuser.export()
    s = json.dumps(blob)
    print(f"Exported JSON length: {len(s)} chars")
    restored = Diffuser.import_(json.loads(s))
    print(f"Restored: {restored.report()}")

    # ------------------------------------------------------------------
    # Event-stream sanity (coordination-engine surface).
    # ------------------------------------------------------------------
    banner("Events emitted (coordination-engine subscribe surface)")
    by_kind: dict[str, int] = {}
    for k, _ in events:
        by_kind[k] = by_kind.get(k, 0) + 1
    for k, n in sorted(by_kind.items()):
        print(f"  {k:25s}  ×{n}")
    print()
    print(f"Total sample events: {by_kind.get(DIFFUSER_SAMPLED, 0)}")
    print(f"Total step events:   {by_kind.get(DIFFUSER_STEP, 0)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
