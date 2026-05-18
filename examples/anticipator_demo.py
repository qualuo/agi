"""Anticipator demo: sleep-time compute earns a perceived 5x latency win.

The pitch in one runnable script (no API key required):

  1. A chat assistant is mid-conversation: the user just said "give me a
     city-wide briefing for Tokyo for tomorrow", and the assistant is
     waiting for them to type the next message.  The runtime declares
     this state as an *idle context* and budgets a small slice of
     sleep-time compute against it.
  2. A toy forecaster enumerates the top-K plausible next queries
     ("...weather in Tokyo?", "...flights?", "...best ramen?",
     "...JR pass?", and a long-tail of low-prior queries).  Each
     candidate carries a prior probability and an estimated cost-to-
     answer.
  3. ``Anticipator`` runs a 0-1 knapsack over the (value, cost) pairs
     under a hard pre-compute budget, executes the chosen subset on
     a (toy) Answerer, and caches the results.
  4. The user's real next message arrives.  Some queries hit the
     cache → instant answers, the test-time miss-cost is *saved*;
     some miss → fresh compute.  The cumulative saved cost vs the
     up-front spend determines net value.
  5. The primitive emits a PAC-style certificate: 95% Wilson CI on the
     hit rate, Hoeffding LCB on the hit rate, empirical-Bernstein LCB
     on the per-serve saved cost.  Every step is fingerprinted into
     a Merkle-style chain for replay.

The narrative reduces to one number a coordination engine cares about:

   *net cost saved per dollar of sleep-time compute spent*.

Run:  python examples/anticipator_demo.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import EventBus
from agi.anticipator import (
    Anticipator,
    AnticipatorConfig,
    Candidate,
    MATCH_HASH,
    MATCH_SIMILARITY,
    KNAPSACK_EXACT,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Toy data.  In production, the forecaster is a small LM / a classifier
# / a Markov chain over recent queries.  The answerer is the expensive
# LLM call.  We use deterministic placeholders so the demo is hermetic.
# ---------------------------------------------------------------------------


# Each row: (query string, prior P(q|ctx), est_miss_cost in FLOPs-units,
# est_precompute_cost).  Pre-compute is slightly cheaper because the
# sleep-time scheduler can batch and use spot capacity.
CANDIDATES_TOKYO = [
    ("weather Tokyo tomorrow",          0.40, 5.0e11, 4.2e11),
    ("flights Tokyo tomorrow",          0.25, 6.0e11, 5.0e11),
    ("best ramen near Shibuya",         0.15, 4.5e11, 3.8e11),
    ("JR pass vs IC card",              0.08, 5.5e11, 4.6e11),
    ("Imperial Palace tour booking",    0.04, 6.5e11, 5.4e11),
    ("teamLab Planets tickets",         0.03, 4.5e11, 3.8e11),
    ("Tokyo earthquake forecast",       0.02, 5.0e11, 4.2e11),
    ("Tokyo wifi rental",               0.02, 4.0e11, 3.4e11),
    ("Tokyo currency exchange today",   0.01, 4.5e11, 3.8e11),
]

# Truthful "answers" — opaque payloads.  In production these are full
# LLM responses, retrieval payloads, or tool results.
ANSWERS = {
    "weather Tokyo tomorrow": {"forecast": "cloudy, 14C/22C, 30% rain"},
    "flights Tokyo tomorrow": {"hnd": 7, "nrt": 12},
    "best ramen near Shibuya": {"top": "Ichiran, Afuri, Tsuta"},
    "JR pass vs IC card": {"recommendation": "7-day pass if >3 long trips"},
    "Imperial Palace tour booking": {"slots": ["10:00", "13:30"]},
    "teamLab Planets tickets": {"availability": "limited"},
    "Tokyo earthquake forecast": {"risk": "low"},
    "Tokyo wifi rental": {"providers": ["NinjaWifi", "JapanWireless"]},
    "Tokyo currency exchange today": {"jpy_per_usd": 148.92},
}


def forecaster(ctx, k, rng):
    rows = ctx.get("candidate_rows", [])
    for (q, prior, miss, pre) in rows[:k]:
        yield Candidate(
            query={"q": q},
            prior=prior,
            est_miss_cost=miss,
            est_precompute_cost=pre,
        )


def answerer(ctx, query):
    """Toy answerer.  Pretends to think, returns a canned response, and
    bills the cost from the candidate row."""
    q = query["q"]
    rows = ctx.get("candidate_rows", [])
    miss_cost = next((m for (qq, _, m, _) in rows if qq == q), 5.0e11)
    answer = ANSWERS.get(q, {"unknown_query": q})
    return answer, miss_cost


def main() -> int:
    rng = random.Random(2026)

    banner("Anticipator — sleep-time compute as a runtime primitive")
    print("Scenario: chat assistant is mid-turn for a Tokyo-trip context.")
    print("Sleep-time budget: 1.5e12 FLOPs (about 3 of 9 candidates can be precomputed).")

    bus = EventBus()
    events_log: list[tuple[str, float]] = []
    bus.subscribe(lambda e: events_log.append((e.kind, e.ts)))

    ant = Anticipator(
        AnticipatorConfig(
            sleep_budget_per_ctx=1.5e12,
            cache_size_limit=64,
            matcher=MATCH_HASH,
            knapsack=KNAPSACK_EXACT,
            cost_unit="flops",
            min_serves_for_certificate=4,
            seed=2026,
        ),
        bus=bus,
        instance_id="demo:anticipator:tokyo",
    )

    ctx_payload = {
        "topic": "tokyo-trip",
        "recent": ["plan Tokyo trip", "10 days mid-May"],
        "candidate_rows": CANDIDATES_TOKYO,
    }
    ant.register_context("turn:42", ctx=ctx_payload, deadline_hint=time.time() + 30.0)

    banner("Sleep-time phase: enumerate, allocate, precompute")
    cands = ant.enumerate("turn:42", forecaster, k=9)
    for c in cands[:5]:
        print(f"   prior={c.prior:.2f}  miss_cost={c.est_miss_cost:.1e}"
              f"  value={c.value:.2e}  q={c.query['q']!r}")
    print(f"   ...{len(cands)} candidates total\n")

    plan = ant.allocate("turn:42", budget=1.5e12)
    print(f"   knapsack={plan.knapsack}  chose {len(plan.chosen)}/{len(cands)} candidates")
    print(f"   total expected saved value = {plan.total_value:.2e}")
    print(f"   total pre-compute cost     = {plan.total_precompute_cost:.2e}\n")

    pre = ant.precompute("turn:42", plan, answerer)
    print(f"   precomputed: {pre.succeeded}/{pre.requested} succeeded,"
          f" {pre.failed} failed, total cost {pre.total_cost:.2e}"
          f", cache size {pre.cache_size_after}")

    banner("Test-time phase: 10 simulated user queries land")
    # Mix of in-cache and out-of-cache queries, drawn from the same
    # prior distribution as the forecaster but with a 10% chance of
    # an unanticipated query slipping through.
    candidate_weights = [(row[0], row[1]) for row in CANDIDATES_TOKYO]
    total_w = sum(w for _, w in candidate_weights)
    candidate_weights = [(q, w / total_w) for q, w in candidate_weights]
    out_of_distribution = [
        "Tokyo embassy hours",
        "Tokyo COVID rules",
        "best onsen day-trip from Tokyo",
    ]
    n_serves = 10

    for i in range(n_serves):
        if rng.random() < 0.1:
            q = rng.choice(out_of_distribution)
        else:
            u = rng.random()
            cum = 0.0
            q = candidate_weights[0][0]
            for (qx, w) in candidate_weights:
                cum += w
                if u <= cum:
                    q = qx
                    break
        res = ant.serve("turn:42", {"q": q}, answerer=answerer)
        tag = "HIT " if res.hit else "miss"
        print(f"   #{i+1:02d}  {tag}  saved={res.saved_cost:.2e}"
              f"  served={res.served_cost:.2e}  q={q!r}")

    banner("Certificate")
    cert = ant.certificate()
    print(f"   serves               = {cert.n_serves}")
    print(f"   hits                 = {cert.n_hits}")
    print(f"   hit rate (point)     = {cert.hit_rate:.3f}")
    print(f"   hit rate Wilson 95%  = [{cert.hit_rate_wilson_lo:.3f}, "
          f"{cert.hit_rate_wilson_hi:.3f}]")
    print(f"   hit rate Hoeff. LCB  = {cert.hit_rate_hoeffding_lo:.3f}")
    print(f"   saved cost total     = {cert.saved_cost_total:.2e} {cert.cost_unit}")
    print(f"   saved cost / serve   = {cert.saved_cost_mean:.2e} {cert.cost_unit}")
    print(f"   saved cost EB LCB    = {cert.saved_cost_eb_lo:.2e} {cert.cost_unit}"
          "  (one-sided lower bound at alpha=0.05)")
    print(f"   precompute cost      = {cert.precompute_cost_total:.2e} {cert.cost_unit}")
    print(f"   NET VALUE            = {cert.net_value:+.2e} {cert.cost_unit}"
          f"  ({'savings' if cert.net_value > 0 else 'loss'})")
    print(f"   fingerprint          = {cert.fingerprint[:32]}...")

    banner("Report (JSON-serialisable; coordinator-consumable)")
    report = ant.report()
    out = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    # Trim very long payloads for stdout but show structure.
    print(out if len(out) < 2000 else (out[:1800] + "\n   ..."))

    banner("Event trail (audit)")
    counts: dict[str, int] = {}
    for kind, _ts in events_log:
        counts[kind] = counts.get(kind, 0) + 1
    for kind in sorted(counts.keys()):
        print(f"   {kind:32s}  {counts[kind]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
