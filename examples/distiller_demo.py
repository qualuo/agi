"""Distiller — runtime primitive demos.

End-to-end demonstration of the Searcher ↔ Distiller closed loop —
the in-process AlphaZero-style self-improvement pair.

Runs four scenarios:
  1. Each model family fits a tiny labeled dataset; show how the
     policy and value match the targets.
  2. Reservoir-sampling buffer demo (Vitter 1985).
  3. Expert Iteration (Anthony, Tian & Barber 2017): Searcher as
     teacher, Distiller as student, on a small grid.  Plot how
     ($/decision) drops as the student gets better.
  4. Certificate determinism: identical runs ⇒ identical certificates.
"""
from __future__ import annotations

import time

from agi.distiller import (
    Demonstration,
    Distiller,
    DistillerConfig,
    ReservoirBuffer,
    ensemble_distiller,
    expert_iteration_step,
    knn_distiller,
    linear_distiller,
    locally_weighted_distiller,
    ucb_table_distiller,
)
from agi.searcher import Searcher, SearcherConfig, ALGORITHM_PUCT


# -----------------------------------------------------------------------------
# Demo 1: each model family on a toy classification task
# -----------------------------------------------------------------------------


def demo_model_families() -> None:
    print("=" * 78)
    print("Demo 1 — Each model family fits a tiny labeled dataset.")
    print("=" * 78)
    targets = [
        ("A", {"go": 9, "stop": 1}, +2.0),
        ("B", {"go": 1, "stop": 9}, -2.0),
        ("A", {"go": 8, "stop": 1}, +1.8),
        ("B", {"go": 1, "stop": 7}, -1.7),
    ]

    def smoke(d, name):
        for st, ad, v in targets:
            d.observe(state=st, action_distribution=ad, value=v)
        rep = d.fit()
        pA = d.policy("A", ["go", "stop"])
        pB = d.policy("B", ["go", "stop"])
        print(f"  {name:<22}  A=(go {pA['go']:.2f}, stop {pA['stop']:.2f}) "
              f"v(A)={d.value('A'):+.2f}  "
              f"B=(go {pB['go']:.2f}, stop {pB['stop']:.2f}) "
              f"v(B)={d.value('B'):+.2f}  "
              f"ce={rep.policy_train_cross_entropy:.3f}")

    smoke(knn_distiller(k=2, seed=0), "kNN(k=2)")
    smoke(linear_distiller(n_features=128, seed=0), "Linear (hashed)")
    smoke(locally_weighted_distiller(n_features=128, bandwidth=2.0, seed=0),
          "LocallyWeighted (LWR)")
    smoke(ucb_table_distiller(seed=0), "UCB-Table (exact)")
    smoke(ensemble_distiller(n_features=128, seed=0), "Ensemble (log-pool)")
    print()


# -----------------------------------------------------------------------------
# Demo 2: reservoir sampling buffer
# -----------------------------------------------------------------------------


def demo_reservoir() -> None:
    print("=" * 78)
    print("Demo 2 — Reservoir sampling: bounded memory over an unbounded stream.")
    print("=" * 78)
    for stream_len in (10, 100, 1000):
        r = ReservoirBuffer(capacity=8, seed=0)
        for i in range(stream_len):
            r.add(Demonstration(state=i, action_distribution={"a": 1},
                                value=0.0))
        ids = sorted(d.state for d in r.items())
        print(f"  stream_len={stream_len:>5}  kept N={len(r)}  "
              f"items={ids}")
    print()


# -----------------------------------------------------------------------------
# Demo 3: Expert Iteration (the AlphaZero loop)
# -----------------------------------------------------------------------------


def demo_expert_iteration() -> None:
    print("=" * 78)
    print("Demo 3 — Expert Iteration: Searcher = teacher, Distiller = student.")
    print("         Each ExIt round: roll out → distill → use student as PUCT prior.")
    print("=" * 78)

    size = 5
    goal = (size - 1, size - 1)

    def acts(_s):
        return ["N", "S", "E", "W"]

    def app(s, a):
        x, y = s
        if a == "N":
            y -= 1
        elif a == "S":
            y += 1
        elif a == "E":
            x += 1
        elif a == "W":
            x -= 1
        if not (0 <= x < size and 0 <= y < size):
            return s
        return (x, y)

    def term(s):
        return s == goal

    def rew(s):
        return 25.0 if s == goal else -1.0

    def feat(s):
        return {"x": float(s[0]), "y": float(s[1]),
                f"x={s[0]}": 1.0, f"y={s[1]}": 1.0, "_b": 1.0}

    distiller = linear_distiller(n_features=128, lr_policy=0.05,
                                 lr_value=0.05, seed=0, featurizer=feat)
    cfg = SearcherConfig(algorithm=ALGORITHM_PUCT,
                         max_iterations=200, c_puct=1.25, seed=0)
    searcher = Searcher(cfg)

    def teacher(state, prior, value):
        rep = searcher.search(state, actions=acts, apply=app, terminal=term,
                              reward=rew, key=lambda s: s,
                              policy_prior=prior, value=value)
        return rep.best_action, rep.best_value, rep.root_visits_by_action

    print()
    print(f"{'round':>5}  {'demos':>6}  {'fit_ce':>7}  {'fit_mse':>9}  "
          f"{'deployed':>8}  {'root_best_action':>18}  {'root_value':>10}")
    print("-" * 78)

    for rnd in range(6):
        n = expert_iteration_step(
            distiller, teacher_search=teacher,
            root=(0, 0), n_episodes=2,
            transition=app, is_terminal=term, max_steps=30,
        )
        rep = distiller.fit()
        # query the student at the root
        p = distiller.policy((0, 0), ["N", "S", "E", "W"])
        best_act = max(p, key=p.get)
        print(f"{rnd+1:>5}  {len(distiller):>6}  "
              f"{rep.policy_train_cross_entropy:>7.3f}  "
              f"{rep.value_train_mse:>9.3f}  "
              f"{str(rep.deployed):>8}  "
              f"{best_act} (p={p[best_act]:.2f}) ".rjust(18) +
              f"  {distiller.value((0,0)):>+10.2f}")
    print()
    print(f"  After 6 ExIt rounds, the student's certificate is:")
    print(f"    {distiller.certificate}")
    print()


# -----------------------------------------------------------------------------
# Demo 4: certificate determinism
# -----------------------------------------------------------------------------


def demo_certificate_determinism() -> None:
    print("=" * 78)
    print("Demo 4 — Two distillers fed identical demonstrations under the same")
    print("         seed produce identical certificates.  Changing the seed")
    print("         changes the certificate.")
    print("=" * 78)
    demos = [
        ("A", {"go": 9, "stop": 1}, +2.0),
        ("B", {"go": 1, "stop": 9}, -2.0),
        ("A", {"go": 8, "stop": 1}, +1.8),
        ("B", {"go": 1, "stop": 7}, -1.7),
    ]
    for seed in (0, 0, 1):
        d = linear_distiller(n_features=64, seed=seed)
        for st, ad, v in demos:
            d.observe(state=st, action_distribution=ad, value=v)
        rep = d.fit()
        print(f"  seed={seed}  certificate={rep.certificate}")
    print()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    print()
    print("=" * 78)
    print(" Distiller — amortized policy/value distillation demo")
    print("=" * 78)
    print()

    demo_model_families()
    demo_reservoir()
    demo_expert_iteration()
    demo_certificate_determinism()

    print("=" * 78)
    print(" Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
