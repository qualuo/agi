"""Demo: PolicyImprover — safe off-policy optimization with HCPI.

A coordination engine routes turns to one of three models with the
goal of maximising p_success. It is already running a routing policy
in production; we have a log of (context, action, propensity, reward)
tuples from the past week. Can we ship a better policy *and prove it
won't regress*?

Run::

    python examples/policy_improver_demo.py
"""
from __future__ import annotations

import random

from agi.events import Event, EventBus
from agi.policy_improver import (
    HCPIReport,
    Improvement,
    MixturePolicySpace,
    PolicyImprover,
    SoftmaxPolicySpace,
    to_policy_candidate,
)
from agi.policy_lab import LoggedEvent, PolicyLab


def _ground_truth_reward(context: dict, action: str, rng: random.Random) -> float:
    """The (unobserved) true reward distribution.

    'difficulty' captures how hard the task is. Bigger models do better
    on hard tasks; cheap models suffice on easy ones. 'opus' is best
    everywhere except cost (not modelled here).
    """
    d = context["difficulty"]
    means = {
        "haiku": 0.95 - 0.6 * d,
        "sonnet": 0.85 - 0.2 * d,
        "opus": 0.95 - 0.05 * d,
    }
    return max(0.0, min(1.0, rng.gauss(means[action], 0.08)))


def _simulate_production_log(n: int = 4000, seed: int = 7) -> list[LoggedEvent]:
    """The production routing policy is uniform over the three models —
    a deliberate baseline that gives us logs across the full action
    space (high coverage)."""
    rng = random.Random(seed)
    actions = ["haiku", "sonnet", "opus"]
    log: list[LoggedEvent] = []
    for _ in range(n):
        ctx = {"difficulty": rng.uniform(0.0, 1.0)}
        a = rng.choice(actions)  # uniform logging policy
        r = _ground_truth_reward(ctx, a, rng)
        log.append(
            LoggedEvent(context=ctx, action=a, propensity=1 / 3, reward=r)
        )
    return log


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _print_improvement(label: str, imp: Improvement | HCPIReport) -> None:
    print(f"  [{label}]")
    print(f"    value  = {imp.value:.4f}    se = {imp.value_se:.4f}")
    print(
        f"    CI(V)  = [{imp.value_lcb:.4f}, {imp.value_ucb:.4f}]"
        f"  at δ = {imp.delta}"
    )
    print(
        f"    Δ vs baseline = {imp.improvement:+.4f}"
        f"   LCB(Δ) = {imp.improvement_lcb:+.4f}"
        f"   UCB(Δ) = {imp.improvement_ucb:+.4f}"
    )
    print(f"    verdict = {imp.verdict.upper():9s}  → {imp.rationale}")


def main() -> None:
    # ---------------------------------------------------------------
    # 1. Get production logs and evaluate the incumbent
    # ---------------------------------------------------------------
    _print_header("1. Production logs + baseline value (PolicyLab)")
    log = _simulate_production_log(n=4000)
    print(f"  collected {len(log)} logged events from uniform router")

    lab = PolicyLab()
    for ev in log:
        lab.record(ev)

    # Incumbent = uniform policy. Get its on-policy value (the average
    # reward in the log is the unbiased estimate of V(π_logging)).
    baseline_value = sum(e.reward for e in log) / len(log)
    print(f"  V(π_baseline) [uniform router]   = {baseline_value:.4f}")

    # ---------------------------------------------------------------
    # 2. Run PolicyImprover with a softmax policy space
    # ---------------------------------------------------------------
    _print_header("2. PolicyImprover — softmax policy space")

    bus = EventBus()
    promoted: list[Event] = []
    bus.subscribe(lambda ev: promoted.append(ev) if ev.kind.endswith("promoted") else None)

    improver = PolicyImprover(
        policy_space=SoftmaxPolicySpace(
            actions=("haiku", "sonnet", "opus"),
            feature_names=("difficulty",),
        ),
        baseline_value=baseline_value,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        event_bus=bus,
        seed=0,
    )
    improver.ingest_from_lab(lab)
    soft = improver.improve(n_restarts=5, n_iters=200, learning_rate=0.5)
    _print_improvement("softmax(difficulty)", soft)

    # Show the learned context-conditional policy.
    pi = improver.policy_space.to_policy(soft.parameters)
    print("  learned policy by difficulty:")
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = pi({"difficulty": d})
        chosen = max(p, key=p.get)
        print(
            f"    difficulty={d:.2f}  →  best={chosen:6s}"
            f"  p(haiku)={p['haiku']:.2f}  p(sonnet)={p['sonnet']:.2f}  p(opus)={p['opus']:.2f}"
        )

    # ---------------------------------------------------------------
    # 3. Cross-validate via PolicyLab (double-check the optimizer)
    # ---------------------------------------------------------------
    _print_header("3. Cross-validate the improved policy in PolicyLab")
    cand = to_policy_candidate(improver, soft, name="softmax_improved")
    for method in ("ips", "snips", "dr"):
        est = lab.evaluate(cand, method=method)
        print(
            f"  PolicyLab.{method:5s}: V̂ = {est.value:.4f}"
            f"  CI = [{est.ci_low:.4f}, {est.ci_high:.4f}]"
            f"  n_eff = {est.n_eff:.0f}"
        )

    # ---------------------------------------------------------------
    # 4. Mixture-policy improvement (provably-safe interpolation)
    # ---------------------------------------------------------------
    _print_header("4. MixturePolicySpace — provably-safe α-interpolation")

    def uniform(_ctx):
        return {"haiku": 1 / 3, "sonnet": 1 / 3, "opus": 1 / 3}

    def opus_always(_ctx):
        return {"haiku": 0.0, "sonnet": 0.0, "opus": 1.0}

    mix_improver = PolicyImprover(
        policy_space=MixturePolicySpace(
            uniform, opus_always, actions=("haiku", "sonnet", "opus")
        ),
        baseline_value=baseline_value,
        weight_clip=10.0,
        delta=0.05,
        reward_range=(0.0, 1.0),
        seed=0,
    )
    mix_improver.record_batch(log)
    mix = mix_improver.improve(n_restarts=4, n_iters=200, learning_rate=0.05)
    _print_improvement(f"mixture(α={mix.parameters[0]:.3f})", mix)

    # ---------------------------------------------------------------
    # 5. Safety-check an externally-produced candidate policy
    # ---------------------------------------------------------------
    _print_header("5. safety_check on hand-rolled rules from another team")

    def cheap_router(ctx):
        # Heuristic the data team proposed: always Sonnet.
        return {"haiku": 0.0, "sonnet": 1.0, "opus": 0.0}

    def difficulty_aware(ctx):
        # Heuristic the platform team proposed: gate by difficulty.
        d = ctx.get("difficulty", 0.5)
        if d < 0.2:
            return {"haiku": 1.0, "sonnet": 0.0, "opus": 0.0}
        if d < 0.6:
            return {"haiku": 0.0, "sonnet": 1.0, "opus": 0.0}
        return {"haiku": 0.0, "sonnet": 0.0, "opus": 1.0}

    for name, fn in [("always_sonnet", cheap_router), ("difficulty_aware", difficulty_aware)]:
        rpt = improver.safety_check(fn, name=name)
        _print_improvement(name, rpt)

    # ---------------------------------------------------------------
    # 6. Coordination-engine decision
    # ---------------------------------------------------------------
    _print_header("6. Coordination engine decision")

    candidates: list[tuple[str, Improvement | HCPIReport]] = [
        ("softmax(difficulty)", soft),
        (f"mixture(α={mix.parameters[0]:.3f})", mix),
        ("always_sonnet", improver.safety_check(cheap_router, name="always_sonnet")),
        ("difficulty_aware", improver.safety_check(difficulty_aware, name="difficulty_aware")),
    ]
    # Rank by LCB on improvement; refuse to ship anything not SAFE.
    safe = [(name, c) for name, c in candidates if (c.safe if hasattr(c, "safe") else c.verdict == "safe")]
    if safe:
        winner = max(safe, key=lambda nc: nc[1].improvement_lcb)
        print(f"  → coordination engine adopts: {winner[0]!r}")
        print(f"     LCB on improvement vs baseline = {winner[1].improvement_lcb:+.4f}")
    else:
        print("  → coordination engine keeps incumbent (no candidate certified SAFE)")

    print()
    print(f"  events that fired during the demo: {len(promoted)} 'promoted' events")


if __name__ == "__main__":
    main()
