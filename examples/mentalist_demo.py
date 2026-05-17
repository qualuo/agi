"""Mentalist — Bayesian theory-of-mind runtime primitive demos.

Three end-to-end scenarios that a coordination engine would drive:

  1. **Auction counterparty modelling.**  Watch an opponent in a sealed-bid
     auction; recover their reservation price and predict their next bid.

  2. **Customer-segment routing.**  A recommender uses Mentalist to track
     two latent user personas (price-sensitive vs. brand-loyal) and routes
     promotions to the right cohort.

  3. **Nested theory of mind in a negotiation.**  Mentalist models both
     the buyer and the seller, then computes what *the buyer believes the
     seller will do* — the recursive ToM that drives strategic concession.

Each demo prints the recovered utility vector, the predicted action
distribution at a representative state, the Clopper-Pearson confidence
interval on the agent's per-action success rate, and the certificate
hash that pins the inference.  The certificate composes with the rest of
the runtime via ``AttestationLedger``.
"""
from __future__ import annotations

import random

from agi.mentalist import (
    PREDICT_BAYES_AVG,
    PREDICT_MAP,
    PREDICT_SOFTMAX,
    PREDICT_THOMPSON,
    Mentalist,
    MentalistConfig,
)


# ---------------------------------------------------------------------------
# Demo 1: auction opponent modelling
# ---------------------------------------------------------------------------


def demo_auction_opponent() -> None:
    print("=" * 72)
    print("Demo 1 — Auction counterparty modelling")
    print("=" * 72)
    m = Mentalist(MentalistConfig(rng_seed=1, irl_max_iters=200))
    # The opponent's private state is their "true valuation tier" (low,
    # mid, high); the runtime infers this from observed bidding behaviour.
    m.register_agent(
        "opponent",
        states=["low_value", "mid_value", "high_value"],
        actions=["pass", "bid_low", "bid_high"],
        outcomes=["overpaid", "win_at_price", "lost_to_competitor"],
    )

    rng = random.Random(7)
    # Synthetic opponent: value-aware bidder who pays more when valuation is high.
    truth = {
        "low_value": {"pass": 0.7, "bid_low": 0.25, "bid_high": 0.05},
        "mid_value": {"pass": 0.2, "bid_low": 0.6, "bid_high": 0.2},
        "high_value": {"pass": 0.05, "bid_low": 0.2, "bid_high": 0.75},
    }
    outcomes_by_action = {
        ("low_value", "bid_high"): "overpaid",
        ("low_value", "bid_low"): "win_at_price",
        ("low_value", "pass"): "lost_to_competitor",
        ("mid_value", "bid_high"): "win_at_price",
        ("mid_value", "bid_low"): "win_at_price",
        ("mid_value", "pass"): "lost_to_competitor",
        ("high_value", "bid_high"): "win_at_price",
        ("high_value", "bid_low"): "lost_to_competitor",
        ("high_value", "pass"): "lost_to_competitor",
    }
    rewards = {
        "overpaid": -2.0,
        "win_at_price": 1.0,
        "lost_to_competitor": 0.0,
    }

    n_rounds = 120
    for _ in range(n_rounds):
        s = rng.choice(["low_value", "mid_value", "high_value"])
        # Sample an action from the (unknown to us) true policy.
        items, probs = zip(*truth[s].items())
        a = rng.choices(items, weights=probs)[0]
        o = outcomes_by_action[(s, a)]
        r = rewards[o]
        m.observe("opponent", state=s, action=a, reward=r, outcome=o)

    utility = m.infer_desire("opponent", force=True)
    print(f"  Recovered utility (zero-centred): {round_dict(utility)}")
    for s in ["low_value", "mid_value", "high_value"]:
        pred = m.predict("opponent", state=s, method=PREDICT_SOFTMAX)
        print(f"  P(action | {s})         = {round_dict(pred)}")
    eu_high = m.expected_utility("opponent", state="high_value")
    print(f"  EU(action | high_value) = {round_dict(eu_high)}")
    lo, hi = m.confidence("opponent", state="high_value", action="bid_high")
    print(f"  CI[ P(success | high, bid_high) ] = [{lo:.3f}, {hi:.3f}]")
    bound = m.pac_bayes_bound("opponent")
    print(
        f"  PAC-Bayes(δ={bound.delta}): empirical loss {bound.empirical_log_loss:.3f}, "
        f"upper {bound.upper_bound:.3f}, KL {bound.kl_to_prior:.2f}, n {bound.n}"
    )
    print(f"  certificate {m.chain_head[:24]}…")


# ---------------------------------------------------------------------------
# Demo 2: customer-segment routing
# ---------------------------------------------------------------------------


def demo_customer_segments() -> None:
    print()
    print("=" * 72)
    print("Demo 2 — Customer-segment routing")
    print("=" * 72)

    m = Mentalist(MentalistConfig(rng_seed=2, irl_max_iters=200))
    # Two personas: a price-sensitive shopper and a brand-loyal repeat customer.
    for persona in ("price_sensitive", "brand_loyal"):
        m.register_agent(
            persona,
            states=["browse", "cart", "checkout"],
            actions=["abandon", "discount_code", "buy"],
            outcomes=["bounce", "convert", "return"],
        )

    rng = random.Random(2024)
    # Price-sensitive: leaves at checkout unless a discount appears.
    for _ in range(60):
        # Discount nudges them to buy from checkout.
        m.observe(
            "price_sensitive",
            state="browse",
            action="abandon",
            reward=-0.2,
            outcome="bounce",
        )
        m.observe(
            "price_sensitive",
            state="cart",
            action="discount_code",
            reward=0.0,
            outcome="bounce",
        )
        m.observe(
            "price_sensitive",
            state="checkout",
            action="buy" if rng.random() < 0.6 else "abandon",
            reward=1.0 if rng.random() < 0.6 else -0.5,
            outcome="convert" if rng.random() < 0.6 else "bounce",
        )

    # Brand-loyal: buys directly, rarely discounted.
    for _ in range(60):
        m.observe(
            "brand_loyal",
            state="browse",
            action="buy" if rng.random() < 0.4 else "abandon",
            reward=1.0 if rng.random() < 0.4 else -0.1,
            outcome="convert" if rng.random() < 0.4 else "bounce",
        )
        m.observe(
            "brand_loyal",
            state="cart",
            action="buy",
            reward=1.0,
            outcome="convert",
        )
        m.observe(
            "brand_loyal",
            state="checkout",
            action="buy",
            reward=1.0,
            outcome="convert",
        )

    for persona in ("price_sensitive", "brand_loyal"):
        m.infer_desire(persona, force=True)
        print(f"  Persona: {persona}")
        report = m.report(persona)
        print(f"    utility:           {round_dict(report.utility_estimate)}")
        print(f"    P(action)|state    = (marginal)  {round_dict(report.action_distribution)}")
        print(f"    state dist:        {round_dict(report.state_distribution)}")
        print(f"    rationality β̂:    {report.rationality_mean:.3f}  ±  {report.rationality_var**0.5:.3f}")
        print(f"    CI[ P(buy) ]:      [{report.confidence_intervals['buy'][0]:.3f}, "
              f"{report.confidence_intervals['buy'][1]:.3f}]")
        print(f"    certificate:       {report.certificate[:24]}…")


# ---------------------------------------------------------------------------
# Demo 3: nested theory of mind
# ---------------------------------------------------------------------------


def demo_nested_tom() -> None:
    print()
    print("=" * 72)
    print("Demo 3 — Nested ToM (buyer reasoning about seller)")
    print("=" * 72)

    m = Mentalist(MentalistConfig(rng_seed=3))
    # Both agents share a (simplified) negotiation schema.
    for who in ("buyer", "seller"):
        m.register_agent(
            who,
            states=["wide_gap", "narrow_gap", "agreement_close"],
            actions=["concede", "hold", "walk_away"],
            outcomes=["bad_deal", "fair_deal", "good_deal"],
        )

    # The buyer's *own* observed history (the buyer's own decisions in
    # past sessions).  Asymmetric so utility is identifiable.
    for _ in range(30):
        m.observe("buyer", state="wide_gap", action="hold",
                  reward=0.2, outcome="bad_deal")
    for _ in range(5):
        m.observe("buyer", state="wide_gap", action="concede",
                  reward=-1.0, outcome="bad_deal")
    for _ in range(25):
        m.observe("buyer", state="narrow_gap", action="concede",
                  reward=0.5, outcome="fair_deal")
    for _ in range(8):
        m.observe("buyer", state="narrow_gap", action="hold",
                  reward=0.1, outcome="bad_deal")
    for _ in range(30):
        m.observe("buyer", state="agreement_close", action="concede",
                  reward=1.0, outcome="good_deal")
    for _ in range(3):
        m.observe("buyer", state="agreement_close", action="walk_away",
                  reward=-0.5, outcome="bad_deal")
    m.infer_desire("buyer", force=True)

    # First-person buyer prediction of own next action.
    print("  Buyer's own policy at agreement_close:")
    self_dist = m.predict(
        "buyer", state="agreement_close", method=PREDICT_SOFTMAX
    )
    print(f"    {round_dict(self_dist)}")

    # Nested: what does *the buyer* think *the seller* will do?
    # The buyer's record of the seller is shared with their own (the
    # buyer has watched the seller concede mostly when the gap is narrow).
    print("  Buyer's belief about seller (ToM_2) at agreement_close:")
    nested = m.nested_belief(
        observer="buyer",
        target="seller",
        state="agreement_close",
        method=PREDICT_SOFTMAX,
    )
    print(f"    {round_dict(nested)}")
    print()
    print(f"  Mentalist chain head: {m.chain_head[:24]}…")
    print(f"  Total observations:    {m.observation_count}")


# ---------------------------------------------------------------------------
# Demo 4: simulation rollout — anytime forecast of agent behaviour
# ---------------------------------------------------------------------------


def demo_simulation() -> None:
    print()
    print("=" * 72)
    print("Demo 4 — Bounded rollout of expected agent behaviour")
    print("=" * 72)

    m = Mentalist(MentalistConfig(rng_seed=4))
    m.register_agent(
        "trader",
        states=["risk_on", "risk_off"],
        actions=["buy", "sell", "hold"],
        outcomes=["pnl_up", "pnl_flat", "pnl_down"],
    )
    rng = random.Random(9)
    for _ in range(80):
        s = rng.choice(["risk_on", "risk_off"])
        if s == "risk_on":
            a, r, o = "buy", 1.0, "pnl_up"
        else:
            a, r, o = "sell", 0.5, "pnl_flat"
        m.observe("trader", state=s, action=a, reward=r, outcome=o)
    m.infer_desire("trader", force=True)

    # Deterministic transition kernel: regime flips with probability 0.5
    # at every step (using a seeded RNG inside the closure).
    flip = random.Random(0)
    def transition(_s: str, _a: str) -> str:
        return "risk_off" if flip.random() < 0.5 else "risk_on"

    rollout = m.simulate(
        "trader",
        start_state="risk_on",
        horizon=8,
        method=PREDICT_SOFTMAX,
        transition=transition,
        rng_seed=11,
    )
    print("  Predicted trajectory (state → action):")
    for i, (s, a) in enumerate(rollout):
        print(f"    t={i:2d}  {s:9s} -> {a}")
    print()
    print(f"  Trader's recovered utility: {round_dict(m.infer_desire('trader'))}")
    print(f"  Chain head: {m.chain_head[:24]}…")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_dict(d: dict, places: int = 3) -> dict:
    return {k: round(v, places) for k, v in d.items()}


if __name__ == "__main__":
    demo_auction_opponent()
    demo_customer_segments()
    demo_nested_tom()
    demo_simulation()
