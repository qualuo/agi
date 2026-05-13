"""Cartographer demo — self-directed curriculum kernel.

Walks an investor through the Cartographer primitive end-to-end:

  1. Register a DAG of skills with prereqs, values, and costs.
  2. Stream synthetic outcomes against a *latent ability* that grows
     with practice — exactly the regime intrinsic-motivation
     curricula (Oudeyer & Kaplan, 2007) are designed for.
  3. Watch each recommendation policy steer toward the
     zone-of-proximal-development: not the mastered skills, not the
     impossible ones, but the ones at the edge of competence.
  4. See mastery propagate through the DAG and unlock dependents.
  5. Inject a "drift event" on a mastered skill and watch the
     coordinator demote it back to the frontier.
  6. Print a calibration report comparing the cartographer's predicted
     competence to the true latent abilities.

Run:
    python examples/cartographer_demo.py

Stdlib-only and CPU-bound. ~250ms on a laptop.
"""
from __future__ import annotations

import random
import statistics

from agi.cartographer import (
    POLICY_INFOGAIN,
    POLICY_KNAPSACK,
    POLICY_LP,
    POLICY_THOMPSON,
    POLICY_UCB,
    STATUS_FRONTIER,
    STATUS_LOCKED,
    STATUS_MASTERED,
    Cartographer,
)
from agi.events import EventBus


CURRICULUM = [
    # (id,                value, cost, prereqs,                  latent_init, latent_gain_per_pull)
    ("add-1d",            1.0,   0.001, (),                       0.30, 0.020),
    ("add-2d",            2.0,   0.002, ("add-1d",),              0.05, 0.018),
    ("add-3d",            3.0,   0.004, ("add-2d",),              0.02, 0.014),
    ("mul-1d",            2.0,   0.003, ("add-1d",),              0.10, 0.022),
    ("mul-2d",            4.0,   0.006, ("mul-1d", "add-2d"),     0.04, 0.012),
    ("div-2d",            5.0,   0.008, ("mul-2d",),              0.02, 0.010),
    ("long-div",          8.0,  0.012, ("div-2d", "add-3d"),      0.01, 0.008),
]


class _World:
    """Drives synthetic outcomes from a learnable latent ability."""

    def __init__(self, curriculum, *, rng_seed: int = 7) -> None:
        self.rng = random.Random(rng_seed)
        self.ability: dict[str, float] = {}
        self.gain: dict[str, float] = {}
        self.pulls: dict[str, int] = {}
        self._mastered_prereqs: dict[str, tuple[str, ...]] = {}
        for tid, _v, _c, prereqs, init, gain in curriculum:
            self.ability[tid] = init
            self.gain[tid] = gain
            self.pulls[tid] = 0
            self._mastered_prereqs[tid] = prereqs

    def sample(self, tid: str, mastered: set[str]) -> float:
        # Prereqs unmet → outcome is bad regardless of pull count.
        if any(p not in mastered for p in self._mastered_prereqs[tid]):
            return 0.0
        self.pulls[tid] += 1
        # Practice raises latent ability toward 0.95 ceiling.
        self.ability[tid] = min(0.95, self.ability[tid] + self.gain[tid])
        return 1.0 if self.rng.random() < self.ability[tid] else 0.0

    def shock(self, tid: str, magnitude: float = 0.55) -> None:
        """Simulate distribution shift on a mastered task."""
        self.ability[tid] = max(0.10, self.ability[tid] - magnitude)


def _h(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_status_panel(cart: Cartographer) -> None:
    print(f"{'task':<14} {'status':<10} {'n':>4} {'mean':>6} "
          f"{'CI':>15} {'LP':>8} {'value':>6}")
    print("-" * 72)
    for tid in cart.task_ids():
        c = cart.competence(tid)
        ci = f"[{c.lower:.2f}, {c.upper:.2f}]"
        lp = f"{c.learning_progress:+.3f}" if c.learning_progress is not None else "  n/a "
        print(f"{tid:<14} {c.status:<10} {c.n:>4} {c.raw_mean:>6.3f} "
              f"{ci:>15} {lp:>8} {cart.spec(tid).value:>6.1f}")


def _run_batch(cart: Cartographer, world: _World, batch_size: int = 20,
               policy: str = POLICY_LP, budget: float | None = None) -> None:
    cur = cart.recommend(policy=policy, k=batch_size, budget=budget)
    mastered = {c.task_id for c in cart.mastered()}
    for item in cur.items:
        for _ in range(8):
            outcome = world.sample(item.task_id, mastered)
            cart.observe(item.task_id, outcome)
    cart.tick()


def main() -> None:
    print(__doc__)

    bus = EventBus()
    # Wire a console subscriber so transitions print as they happen.
    def _listener(evt):
        if evt.kind in ("cartographer.advanced", "cartographer.regressed",
                         "cartographer.frontier_changed"):
            payload = dict(evt.data)
            payload.pop("receipt_hash", None)
            print(f"  >> {evt.kind} :: {payload}")

    bus.subscribe(_listener)
    cart = Cartographer(
        bus=bus,
        entry_threshold=0.20,
        mastery_threshold=0.70,
        delta=0.05,
        window_recent=6,
        window_prior=6,
    )

    # 1. Register the DAG.
    _h("1. Register a multi-skill curriculum (DAG of prerequisites)")
    for tid, v, c, prereqs, *_ in CURRICULUM:
        cart.register_task(tid, value=v, cost=c, prereqs=prereqs)
    cart.tick()
    _print_status_panel(cart)

    world = _World(CURRICULUM)

    # 2. LP-greedy training loop.
    _h("2. Train under POLICY_LP (Oudeyer learning-progress greedy)")
    for round_ix in range(40):
        _run_batch(cart, world, batch_size=2, policy=POLICY_LP)
    _print_status_panel(cart)

    # 3. Switch to UCB to push through the frontier.
    _h("3. Switch to POLICY_UCB to convert frontier into mastery")
    for round_ix in range(50):
        _run_batch(cart, world, batch_size=2, policy=POLICY_UCB)
    _print_status_panel(cart)

    # 4. Knapsack: budget the next batch with a hard cost cap.
    _h("4. Budgeted scheduling: POLICY_KNAPSACK with budget=0.020 USD")
    cur = cart.recommend(policy=POLICY_KNAPSACK, k=8, budget=0.020)
    print(f"items chosen: {len(cur.items)}, total cost ${cur.total_cost:.4f}")
    for it in cur.items:
        print(f"  {it.task_id:<14}  score={it.score:+.3f}  cost=${it.cost:.4f}")

    # 5. Drift event: shock a mastered task.
    _h("5. Drift event on a mastered skill — coordinator demotes it")
    if cart.mastered():
        victim = cart.mastered()[0].task_id
        print(f"shocking {victim} (latent drop)")
        world.shock(victim, magnitude=0.55)
        for _ in range(40):
            outcome = world.sample(victim, mastered={c.task_id for c in cart.mastered()})
            cart.observe(victim, outcome)
        cart.tick()
        # The runtime would normally hear about this via DriftSentinel.
        # We surface the regression manually here.
        cart.regress(victim, rationale="drift")
    _print_status_panel(cart)

    # 6. Coverage report (calibration of predictions vs latent truth).
    _h("6. Coverage report — predicted competence vs latent ability")
    for tid, ability in world.ability.items():
        cart.record_truth(tid, true_mean=ability)
    rep = cart.coverage_report()
    print(f"n_tasks        : {rep.n_tasks}")
    print(f"n_predictions  : {rep.n_predictions}")
    print(f"signed err     : {rep.mean_signed_error:+.4f}")
    print(f"abs err        : {rep.mean_abs_error:.4f}")
    print(f"Brier          : {rep.brier_score:.4f}")
    print("counts by status:")
    for s, c in rep.n_by_status.items():
        if c:
            mean_n = rep.mean_n_by_status[s]
            mean_lp = rep.mean_lp_by_status[s]
            print(f"  {s:<10}  count={c}  mean n={mean_n:.1f}  mean LP={mean_lp:+.3f}")

    _h("Done.")


if __name__ == "__main__":
    main()
