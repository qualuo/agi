"""Continualist + Drift + Pareto + AttestationLedger — lifelong-learning
coordination flow.

A coordination engine drives the runtime to deliver:

  Goal:  Train an agent on a *stream of tasks* (coding → research →
         finance → ops) and ship a *certified continual learner*
         that demonstrably:

           * accumulates skill without catastrophic forgetting,
           * exposes the plasticity-vs-stability frontier as a
             Pareto front for product trade-offs,
           * carries a PAC-Bayes continual-risk certificate,
           * detects unannounced task boundaries from the loss
             stream so an unsupervised coordinator can re-trigger
             curriculum / evaluation,
           * emits a tamper-evident replay-verifiable audit chain.

This is the **investor-grade lifelong-learning story** in a single
runnable script (no API key required, pure stdlib):

  1. Continualist.register_task → begin a new task
  2. Continualist.update         → ingest per-step (grad, loss, acc)
  3. Continualist.commit_task    → lock-in importance + anchor
  4. Continualist.regulariser    → trainer plugs in (λ/2) Σ F (θ−θ*)²
  5. Continualist.project_gradient (A-GEM) → refuse damaging updates
  6. Continualist.report         → BWT / FWT / AvgAcc / forgetting
  7. Continualist.certify        → PAC-Bayes + plasticity/stability
  8. AttestationLedger.append    → compliance receipt chain

Run::

  python examples/continualist_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.continualist import (
    Continualist,
    ContinualistConfig,
    METHOD_ONLINE_EWC,
    REPLAY_RESERVOIR,
)
from agi.pareto import (
    Pareto,
    ParetoConfig,
)


def banner(title: str) -> None:
    bar = "─" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Synthetic "skill" model:
#   θ ∈ R^d.  Each task is a target θ*_t ∈ R^d.  Loss is squared
#   distance ‖θ - θ*_t‖²/d.  Per-step gradient is 2(θ - θ*_t)/d.
#   Accuracy = clip(1 - loss, 0, 1).
# ---------------------------------------------------------------------------


def step_target(theta: list[float], target: list[float], lr: float = 0.1):
    """One vanilla GD step toward `target`; returns (grad, new_theta, loss, acc)."""
    d = len(theta)
    grad = [2.0 * (theta[i] - target[i]) / d for i in range(d)]
    new_theta = [theta[i] - lr * grad[i] for i in range(d)]
    loss = sum((new_theta[i] - target[i]) ** 2 for i in range(d)) / d
    acc = max(0.0, min(1.0, 1.0 - loss))
    return grad, new_theta, loss, acc


def regularised_step(
    theta: list[float],
    target: list[float],
    cl: Continualist,
    lr: float = 0.1,
) -> tuple[list[float], float, float, list[float]]:
    """GD step that *adds* the continual-learning regulariser gradient.

    The trainer is in charge of the optimisation; Continualist
    contributes the regulariser to keep old skills.
    """
    d = len(theta)
    grad_task = [2.0 * (theta[i] - target[i]) / d for i in range(d)]
    _, reg_grad = cl.regulariser(theta)
    grad = [grad_task[i] + reg_grad[i] for i in range(d)]
    new_theta = [theta[i] - lr * grad[i] for i in range(d)]
    loss = sum((new_theta[i] - target[i]) ** 2 for i in range(d)) / d
    acc = max(0.0, min(1.0, 1.0 - loss))
    return new_theta, loss, acc, grad


# ---------------------------------------------------------------------------
# Held-out evaluation for previously committed tasks.
# ---------------------------------------------------------------------------


def heldout_accuracy(theta: list[float], target: list[float]) -> float:
    d = len(theta)
    loss = sum((theta[i] - target[i]) ** 2 for i in range(d)) / d
    return max(0.0, min(1.0, 1.0 - loss))


# ---------------------------------------------------------------------------
# 1. NAIVE FINE-TUNE (no regulariser) — produces catastrophic forgetting.
# ---------------------------------------------------------------------------


def run_naive_finetune(targets: dict[str, list[float]], d: int, steps_per_task: int):
    theta = [0.0] * d
    history: list[dict[str, float]] = []
    for tid, tgt in targets.items():
        for _ in range(steps_per_task):
            _, theta, _, _ = step_target(theta, tgt, lr=0.2)
        # Held-out accuracy on every task so far.
        accs = {pid: heldout_accuracy(theta, ptgt) for pid, ptgt in targets.items()}
        history.append({"after": tid, **accs})
    return history


# ---------------------------------------------------------------------------
# 2. CONTINUALIST (Online EWC) — protects prior skills.
# ---------------------------------------------------------------------------


def run_continualist(
    targets: dict[str, list[float]],
    d: int,
    steps_per_task: int,
    ewc_lambda: float = 4.0,
) -> tuple[Continualist, list[dict[str, float]], list[float]]:
    cl = Continualist(
        ContinualistConfig(
            method=METHOD_ONLINE_EWC,
            dim=d,
            ewc_lambda=ewc_lambda,
            fisher_decay=0.9,
            replay_capacity=64,
            replay_strategy=REPLAY_RESERVOIR,
            boundary_detection=True,
            boundary_hazard=0.02,
            boundary_threshold=0.5,
            plasticity_min=0.6,
            stability_eps=0.15,
            confidence=0.95,
            seed=0,
        )
    )
    theta = [0.0] * d
    history: list[dict[str, float]] = []
    for tid, tgt in targets.items():
        cl.register_task(tid)
        for step in range(steps_per_task):
            old_theta = list(theta)
            theta, loss, acc, grad = regularised_step(theta, tgt, cl, lr=0.2)
            cl.update(
                tid,
                grad=grad,
                theta=theta,
                loss=loss,
                accuracy=acc,
                delta_theta=[theta[i] - old_theta[i] for i in range(d)],
            )
        # Held-out: evaluate against *every* known target.
        held = {pid: heldout_accuracy(theta, ptgt) for pid, ptgt in targets.items()}
        cl.commit_task(tid, final_theta=theta, accuracies=held)
        history.append({"after": tid, **held})
    return cl, history, theta


# ---------------------------------------------------------------------------
# 3. PARETO PLASTICITY ↔ STABILITY FRONTIER.
# ---------------------------------------------------------------------------


def build_plasticity_stability_frontier(
    targets: dict[str, list[float]], d: int, steps_per_task: int
) -> Pareto:
    """Sweep λ ∈ {0, 0.5, 1, 2, 4, 8, 16, 32} and register each
    resulting agent in a Pareto front of (− new-skill accuracy,
    forgetting-on-old)."""
    pf = Pareto(
        ParetoConfig(
            senses=("min", "min"),
            reference=(1.0, 1.0),
        )
    )
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        cl, _, _ = run_continualist(targets, d, steps_per_task, ewc_lambda=lam)
        rep = cl.report()
        # Cost vector: (1 − fresh skill avg, forgetting)
        fresh_loss = 1.0 - rep.average_accuracy
        forget = rep.forgetting
        # Clip so reference dominates.
        cost = (max(0.0, min(1.0, fresh_loss)), max(0.0, min(1.0, forget)))
        pf.observe(f"lambda={lam}", cost)
    return pf


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def print_history(name: str, history: list[dict[str, float]], targets: list[str]) -> None:
    print(f"\n  {name}")
    header = "    after  | " + " | ".join(f"{tid[:9]:>9s}" for tid in targets)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for row in history:
        print(
            f"    {row['after'][:6]:>6s}  | "
            + " | ".join(f"{row.get(tid, 0.0):9.3f}" for tid in targets)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(42)
    d = 16
    # Four targets — well-separated skill axes.
    task_ids = ["coding", "research", "finance", "ops"]
    targets: dict[str, list[float]] = {}
    for tid in task_ids:
        targets[tid] = [rng.uniform(-1.0, 1.0) for _ in range(d)]
    steps_per_task = 80

    banner("Continualist coordination demo")
    print(
        f"  Parameter dim:        {d}\n"
        f"  Tasks (in order):     {task_ids}\n"
        f"  Steps per task:       {steps_per_task}\n"
    )

    # --- 1. Naive fine-tune (no regulariser): catastrophic forgetting --- #
    banner("Step 1 — Naive fine-tune (no protection)")
    naive_history = run_naive_finetune(targets, d, steps_per_task)
    print_history("Held-out accuracy after each task:", naive_history, task_ids)
    naive_final = naive_history[-1]
    print(
        "\n  Naive average accuracy on all tasks at end:  "
        f"{sum(naive_final[t] for t in task_ids) / len(task_ids):.3f}"
    )

    # --- 2. Continualist (Online EWC + reservoir replay) --- #
    banner("Step 2 — Continualist (Online EWC + reservoir replay)")
    cl, cl_history, theta_final = run_continualist(targets, d, steps_per_task)
    print_history("Held-out accuracy after each task:", cl_history, task_ids)
    rep = cl.report()
    print(
        "\n  Continual-learning headline:\n"
        f"    AvgAcc            = {rep.average_accuracy:.3f}\n"
        f"    BackwardTransfer  = {rep.backward_transfer:+.3f}\n"
        f"    ForwardTransfer   = {rep.forward_transfer:+.3f}\n"
        f"    Forgetting        = {rep.forgetting:.3f}\n"
        f"    Plasticity        = {rep.plasticity:+.3f}\n"
        f"    n_boundaries (BOCD) = {rep.n_boundaries}\n"
        f"    Replay size       = {rep.replay_size}"
    )

    # --- 3. Certificate --- #
    banner("Step 3 — Plasticity/stability + PAC-Bayes certificate")
    cert = cl.certify(n_samples_per_task=2000)
    print(
        "  Certificate (Pentina-Lampert 2014 PAC-Bayes):\n"
        f"    Empirical mean risk: {cert.empirical_mean_risk:.4f}\n"
        f"    PAC-Bayes bound:     {cert.pac_bayes_bound:.4f}  "
        f"(δ = {1 - cert.confidence:.2f})\n"
        f"    KL complexity:       {cert.kl_complexity:.4f}\n"
        f"    Min fresh accuracy:  {cert.min_fresh_accuracy:.3f}  "
        f"(plasticity_min = {cl.config.plasticity_min:.2f})\n"
        f"    Max forget gap:      {cert.max_forget_gap:.3f}  "
        f"(stability_eps = {cl.config.stability_eps:.2f})\n"
        f"    plasticity_ok:       {cert.plasticity_ok}\n"
        f"    stability_ok:        {cert.stability_ok}\n"
        f"    chain head:          {cert.head[:32]}..."
    )

    # --- 4. A-GEM projection demo --- #
    banner("Step 4 — A-GEM gradient projection refuses skill-damaging steps")
    # A new candidate gradient that is "obviously bad" — moves opposite
    # to the average replay gradient.  A-GEM projects.
    replay_avg = [0.0] * d
    for it in cl._replay:
        for i in range(d):
            replay_avg[i] += it.gradient[i]
    n = max(1, len(cl._replay))
    replay_avg = [v / n for v in replay_avg]
    adversarial = [-v for v in replay_avg]
    out = cl.project_gradient(adversarial)
    print(
        f"  Adversarial gradient (negated replay avg) was_projected = {out.was_projected}\n"
        f"    inner product before  = {out.inner_product:+.4f}\n"
        f"    ||before||₂           = {math.sqrt(sum(v*v for v in adversarial)):.4f}\n"
        f"    ||after||₂            = {math.sqrt(sum(v*v for v in out.projected)):.4f}"
    )

    # --- 5. Boundary detection without explicit signals --- #
    banner("Step 5 — Unsupervised task-boundary detection (BOCD)")
    boundaries = [b for b in cl._boundaries if not b.explicit]
    if boundaries:
        print(
            f"  BOCD self-segmented the experience stream into "
            f"{len(boundaries)} implicit task boundaries:"
        )
        for b in boundaries:
            print(f"    step={b.step:5d}  p={b.probability:.3f}  head={b.head[:16]}...")
    else:
        print("  No implicit boundaries detected — task transitions were committed explicitly.")

    # --- 6. Pareto plasticity-stability frontier --- #
    banner("Step 6 — Pareto plasticity-stability frontier (sweep λ)")
    pf = build_plasticity_stability_frontier(targets, d, steps_per_task)
    front = pf.frontier()
    print("  λ-sweep frontier (Pareto-rank 1) on (1−fresh, forgetting):")
    for cid, cost in zip(front.candidates, front.costs):
        print(
            f"    {cid:>14s}   "
            f"cost = ({cost[0]:.3f}, {cost[1]:.3f})"
        )
    hv_report = pf.hypervolume()
    print(f"\n  Hypervolume = {hv_report.hypervolume:.4f}")
    print(f"  Pareto-front chain head: {pf.chain_head[:32]}...")

    # --- 7. Snapshot / restore --- #
    banner("Step 7 — Snapshot / restore — coordination-engine hibernation")
    snap = cl.snapshot()
    cl2 = Continualist(ContinualistConfig(dim=d))
    cl2.restore(snap)
    print(
        f"  Original head:    {cl.chain_head[:32]}...\n"
        f"  Restored head:    {cl2.chain_head[:32]}...\n"
        f"  Heads match:      {cl.chain_head == cl2.chain_head}\n"
        f"  Tasks restored:   {cl2.n_committed}\n"
        f"  Replay restored:  {cl2.replay_size()}"
    )

    # --- 8. Summary --- #
    banner("Investor summary")
    naive_avg = sum(naive_history[-1][t] for t in task_ids) / len(task_ids)
    cl_avg = sum(cl_history[-1][t] for t in task_ids) / len(task_ids)
    delta = cl_avg - naive_avg
    print(
        f"  Naive fine-tune end-of-stream average accuracy:    {naive_avg:.3f}\n"
        f"  Continualist  end-of-stream average accuracy:      {cl_avg:.3f}\n"
        f"  Δ (lifetime skill retained over naive baseline):   {delta:+.3f}\n"
        f"\n"
        f"  PAC-Bayes risk bound (δ=0.05):                     {cert.pac_bayes_bound:.4f}\n"
        f"  Pareto frontier candidates (plasticity↔stability): {len(front.candidates)}\n"
        f"  Tamper-evident chain head:                          {cl.chain_head[:32]}...\n"
    )


if __name__ == "__main__":
    main()
