"""Anticipator coordination demo: manifest discovery → forecaster →
allocate → precompute → serve → certify → replay.

This is the *runtime engine* story.  In one runnable script:

  1. A coordination engine receives an idle signal — the agent has
     finished its current turn and is awaiting the next user
     message.  Idle time is the cheapest compute it can buy.

  2. The engine queries the manifest for the ``anticipator``
     primitive and binds the loader.  It also discovers companion
     primitives (``scheduler``, ``economist``, ``costs``) for
     bookkeeping.

  3. A toy ``forecaster`` (one of the manifest-discoverable
     primitives in production; here a deterministic stub) enumerates
     candidate next queries.  The engine asks ``Anticipator`` to
     allocate a sleep-time budget under a hard FLOP cap.

  4. The engine runs the chosen pre-compute candidates against a
     toy ``answerer``.  Cached results are durable across the
     turn boundary.

  5. The user's actual next query lands.  The engine routes it
     through ``Anticipator.serve``: on a hit the cached answer is
     returned in microseconds; on a miss the engine falls back to
     the expensive fresh inference path.

  6. The engine emits a coordinator-facing certificate (hit-rate
     CIs, empirical-Bernstein LCB on saved cost, net value) and a
     replay-verifiable fingerprint chain.

  7. Over many turns the engine adapts the budget: hit rates
     above target ⇒ less spend next time; misses dominate ⇒ raise
     the similarity threshold or expand K.

Run:  python examples/anticipator_coordination_demo.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event, EventBus
from agi.manifest import default_manifest
from agi.anticipator import (
    Anticipator,
    AnticipatorConfig,
    Candidate,
    MATCH_HASH,
    KNAPSACK_EXACT,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Toy domain: customer-support assistant with three repeat-prone intents.
# ---------------------------------------------------------------------------


INTENTS = {
    "order_status": {
        "prior": 0.45,
        "miss_cost": 6.0e11,
        "pre_cost":  4.8e11,
        "answer": {"intent": "order_status", "answer": "in transit, ETA tomorrow"},
    },
    "return_policy": {
        "prior": 0.22,
        "miss_cost": 5.0e11,
        "pre_cost":  4.0e11,
        "answer": {"intent": "return_policy", "answer": "30 days no questions asked"},
    },
    "warranty": {
        "prior": 0.13,
        "miss_cost": 5.5e11,
        "pre_cost":  4.4e11,
        "answer": {"intent": "warranty", "answer": "2-year limited; serial required"},
    },
    "human_escalation": {
        "prior": 0.08,
        "miss_cost": 8.0e11,
        "pre_cost":  6.4e11,
        "answer": {"intent": "human_escalation", "answer": "routing to human agent"},
    },
    "refund_status": {
        "prior": 0.06,
        "miss_cost": 5.5e11,
        "pre_cost":  4.4e11,
        "answer": {"intent": "refund_status", "answer": "processed, 3-5 business days"},
    },
    "promo_codes": {
        "prior": 0.04,
        "miss_cost": 4.0e11,
        "pre_cost":  3.2e11,
        "answer": {"intent": "promo_codes", "answer": "WELCOME10 active until Jun"},
    },
    "shipping_carrier": {
        "prior": 0.02,
        "miss_cost": 4.0e11,
        "pre_cost":  3.2e11,
        "answer": {"intent": "shipping_carrier", "answer": "USPS / DHL by region"},
    },
}


def forecaster(ctx, k, rng):
    intents = ctx.get("intents", {})
    for name, row in list(intents.items())[:k]:
        yield Candidate(
            query={"intent": name},
            prior=row["prior"],
            est_miss_cost=row["miss_cost"],
            est_precompute_cost=row["pre_cost"],
        )


def answerer(ctx, query):
    intents = ctx.get("intents", {})
    name = query["intent"]
    row = intents.get(name)
    if row is None:
        return {"intent": name, "answer": "I'm not sure, can you rephrase?"}, 8.0e11
    return row["answer"], row["miss_cost"]


# ---------------------------------------------------------------------------
# A toy "coordination engine" that uses the manifest to discover the
# anticipator primitive, runs the sleep-time loop across N agent turns,
# and adapts the budget based on observed economics.
# ---------------------------------------------------------------------------


@dataclass
class CoordinationConfig:
    n_turns: int = 6
    serves_per_turn: int = 12
    initial_budget: float = 1.6e12
    target_hit_rate: float = 0.60
    seed: int = 17


@dataclass
class TurnRecord:
    turn: int
    budget: float
    chosen: int
    precompute_cost: float
    hits: int
    misses: int
    saved_cost: float
    net_value: float


def run_coordination(cfg: CoordinationConfig) -> tuple[Anticipator, list[TurnRecord]]:
    bus = EventBus()
    seen_events: dict[str, int] = {}

    def tally(e: Event) -> None:
        seen_events[e.kind] = seen_events.get(e.kind, 0) + 1

    bus.subscribe(tally)

    # 1. Manifest discovery.
    manifest = default_manifest(fresh=True)
    spec = manifest.lookup("anticipator")
    print(f"   discovered primitive: {spec.name} ({spec.kind})")
    print(f"     stable_id   = {spec.stable_id}")
    print(f"     certificate = {spec.certificate}")
    print(f"     composes    = {', '.join(spec.composes_with[:6])}, ...")

    # 2. Bind the primitive.
    ant = Anticipator(
        AnticipatorConfig(
            sleep_budget_per_ctx=cfg.initial_budget,
            sleep_budget_global=cfg.initial_budget * cfg.n_turns * 2,
            cache_size_limit=128,
            matcher=MATCH_HASH,
            knapsack=KNAPSACK_EXACT,
            cost_unit="flops",
            alpha=0.05,
            min_serves_for_certificate=4,
            seed=cfg.seed,
        ),
        bus=bus,
        instance_id="demo:anticipator:coordinator",
    )

    rng = random.Random(cfg.seed)
    records: list[TurnRecord] = []
    current_budget = cfg.initial_budget

    # 3. N turns.  Each turn:
    #    a. Register the idle context.
    #    b. Enumerate + allocate + precompute.
    #    c. Issue N simulated user serves drawn from the prior.
    #    d. Certify and adapt the budget.
    for turn in range(cfg.n_turns):
        ctx_id = f"turn:{turn}"
        ant.register_context(
            ctx_id,
            ctx={"intents": INTENTS},
            deadline_hint=time.time() + 30.0,
        )

        cands = ant.enumerate(ctx_id, forecaster, k=len(INTENTS))
        plan = ant.allocate(ctx_id, budget=current_budget)
        pre = ant.precompute(ctx_id, plan, answerer)

        # Issue serves drawn from the intent prior.
        intent_names = list(INTENTS.keys())
        weights = [INTENTS[i]["prior"] for i in intent_names]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        hits_this_turn = 0
        misses_this_turn = 0
        saved_this_turn = 0.0
        for _ in range(cfg.serves_per_turn):
            # 8% out-of-distribution intent that we never anticipated.
            if rng.random() < 0.08:
                intent = "novel_intent"
            else:
                u = rng.random()
                cum = 0.0
                intent = intent_names[0]
                for (name, w) in zip(intent_names, weights):
                    cum += w
                    if u <= cum:
                        intent = name
                        break
            r = ant.serve(ctx_id, {"intent": intent}, answerer=answerer)
            if r.hit:
                hits_this_turn += 1
                saved_this_turn += r.saved_cost
            else:
                misses_this_turn += 1

        # Certify at the per-turn level (anytime-valid → safe to query mid-run).
        cert = ant.certificate()

        net = saved_this_turn - pre.total_cost
        records.append(TurnRecord(
            turn=turn,
            budget=current_budget,
            chosen=len(plan.chosen),
            precompute_cost=pre.total_cost,
            hits=hits_this_turn,
            misses=misses_this_turn,
            saved_cost=saved_this_turn,
            net_value=net,
        ))

        print(f"\n   turn {turn:02d}:  budget={current_budget:.2e}  "
              f"chose={len(plan.chosen)}  hits={hits_this_turn}/{cfg.serves_per_turn}  "
              f"saved={saved_this_turn:.2e}  net={net:+.2e}")
        print(f"             running cert: hit_rate={cert.hit_rate:.3f} "
              f"wilson=[{cert.hit_rate_wilson_lo:.3f},"
              f"{cert.hit_rate_wilson_hi:.3f}]  "
              f"EB_lo={cert.saved_cost_eb_lo:.2e}")

        # 4. Adapt the next budget based on hit rate vs target.  This is
        # the "coordinator using the certificate" part — the engine
        # acts on the anytime-valid bound, not the noisy point estimate.
        if cert.hit_rate_wilson_lo > cfg.target_hit_rate:
            # We have headroom — shrink budget by 15%.
            current_budget = max(2.0e11, current_budget * 0.85)
        elif cert.hit_rate_wilson_hi < cfg.target_hit_rate:
            # Underperforming with confidence — grow budget by 25%.
            current_budget = min(5.0e12, current_budget * 1.25)
        # Else: CI straddles target → hold steady.

        # Invalidate the closed turn so the next one starts fresh.
        ant.invalidate(ctx_id)

    print("\n   events emitted:")
    for kind in sorted(seen_events.keys()):
        print(f"     {kind:36s}  {seen_events[kind]}")

    return ant, records


def main() -> int:
    banner("Anticipator coordination demo")
    print("Scenario: customer-support assistant; 6 consecutive idle windows;")
    print("the engine adapts sleep-time budget per turn based on Wilson CI.")

    cfg = CoordinationConfig()
    ant, turns = run_coordination(cfg)

    banner("Per-turn summary")
    print(f"   {'turn':>4} {'budget':>10} {'chose':>6} {'hits':>5} "
          f"{'misses':>6} {'saved':>10} {'net':>11}")
    for r in turns:
        print(f"   {r.turn:>4} {r.budget:>10.2e} {r.chosen:>6} "
              f"{r.hits:>5} {r.misses:>6} {r.saved_cost:>10.2e} {r.net_value:>+11.2e}")

    total_saved = sum(r.saved_cost for r in turns)
    total_spent = sum(r.precompute_cost for r in turns)
    print(f"\n   totals: saved={total_saved:.2e}  spent={total_spent:.2e}  "
          f"net={total_saved - total_spent:+.2e}")

    banner("Final certificate (anytime-valid across the whole window)")
    cert = ant.certificate()
    print(f"   serves              = {cert.n_serves}")
    print(f"   hits                = {cert.n_hits}")
    print(f"   hit rate (point)    = {cert.hit_rate:.3f}")
    print(f"   hit rate Wilson 95% = [{cert.hit_rate_wilson_lo:.3f}, "
          f"{cert.hit_rate_wilson_hi:.3f}]")
    print(f"   saved cost EB LCB   = {cert.saved_cost_eb_lo:.2e} flops")
    print(f"   precompute total    = {cert.precompute_cost_total:.2e} flops")
    print(f"   net value           = {cert.net_value:+.2e} flops")
    print(f"   fingerprint         = {cert.fingerprint[:48]}...")

    banner("Coordinator-consumable JSON (this is what the engine forwards)")
    payload = {
        "primitive": "anticipator",
        "instance_id": ant.instance_id,
        "fingerprint": ant.fingerprint_hash,
        "certificate": cert.to_dict(),
        "turns": [
            {"turn": r.turn, "budget": r.budget, "chosen": r.chosen,
             "hits": r.hits, "misses": r.misses,
             "saved_cost": r.saved_cost, "net_value": r.net_value}
            for r in turns
        ],
    }
    out = json.dumps(payload, indent=2, sort_keys=True)
    print(out if len(out) < 2500 else (out[:2200] + "\n   ..."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
