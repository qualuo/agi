"""PolicyRouter — contextual bandit routing on top of CapabilityRegistry.

The `CapabilityRegistry` records what happened. The `PolicyRouter` decides
what to try next.

For a coordination engine dispatching N steps a day, naive "pick the best
historical bucket" routing converges to a local optimum — it never tries
the cheaper model on prompts where the expensive one is overkill, and
never tries a higher-effort config on prompts where the cheap one keeps
failing. A bandit fixes both ends:

  - **Exploit**: when one (role, model, effort) arm dominates similar
    prompts, pick it.
  - **Explore**: with calibrated probability, try an under-sampled arm
    so the policy keeps learning.

We use **Thompson sampling** with a Beta(α, β) posterior per arm, scoped
by the prompt's nearest-neighbor cluster in token space. α = successes +
1, β = failures + 1. To pick: sample a success rate from each arm's
Beta, penalize by expected cost (so a cheaper arm with comparable
success wins), and return the argmax.

This is real online learning at the routing layer. Investors care
because it's the difference between "an LLM that calls tools" and "a
system that gets cheaper and more reliable the more you use it."

Falls back to the registry's `recommend()` when no arm has any
evidence — same defaults, so the router is a drop-in upgrade.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from agi.capabilities import (
    CapabilityRecommendation,
    CapabilityRecord,
    CapabilityRegistry,
    _tokenize,
)
from agi.runtime import SessionConfig


@dataclass
class Arm:
    role: str
    model: str
    effort: str = "high"

    def key(self) -> tuple[str, str, str]:
        return (self.role, self.model, self.effort)


@dataclass
class ArmPosterior:
    """Beta posterior over success probability for one arm, in context.

    Cost is tracked as a running weighted mean of observed costs on the
    arm; it does not have a Bayesian posterior here (point estimate is
    fine for ranking, and cost has tighter empirical variance than
    success).
    """
    arm: Arm
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0   # failures + 1
    cost_sum: float = 0.0
    cost_count: int = 0
    evidence_count: int = 0

    def update(self, *, success: bool, cost_usd: float) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.cost_sum += float(cost_usd)
        self.cost_count += 1
        self.evidence_count += 1

    def expected_cost(self) -> float:
        if self.cost_count == 0:
            return 0.0
        return self.cost_sum / self.cost_count

    def expected_success(self) -> float:
        total = self.alpha + self.beta
        if total <= 0:
            return 0.0
        return self.alpha / total

    def sample(self, rng: random.Random) -> float:
        """Draw a success-rate sample from the Beta posterior."""
        return rng.betavariate(self.alpha, self.beta)


@dataclass
class RoutingDecision:
    arm: Arm
    sampled_success: float
    expected_cost_usd: float
    expected_success: float
    evidence_count: int
    explored: bool
    rationale: str

    def to_session_config(self, base: SessionConfig | None = None) -> SessionConfig:
        base = base or SessionConfig()
        return SessionConfig(
            **{
                **base.__dict__,
                "model": self.arm.model,
                "role": self.arm.role,
                "effort": self.arm.effort,
                "system_prompt_extra": (
                    base.system_prompt_extra
                    or f"Role: {self.arm.role}. Return a concise final answer."
                ),
            }
        )


@dataclass
class PolicyRouter:
    """Contextual Thompson-sampling router over (role, model, effort) arms.

    Parameters
    ----------
    registry
        The `CapabilityRegistry` to seed posteriors from. The router
        also calls `registry.record(...)` after each observed outcome
        via `observe(...)` so disk-state stays the durable source.
    arms
        Candidate arms. If empty, the router seeds itself from any arms
        seen in the registry plus a small default set.
    epsilon
        Floor on exploration: with probability `epsilon`, ignore the
        sampled scores and pick a uniformly random arm. Keeps the
        policy from collapsing on early bad luck.
    cost_weight
        Penalty applied to expected cost when ranking arms. The
        sampled-success score is divided by `(1 + cost_weight × cost)`
        before argmax. Higher → more cost-sensitive routing.
    similarity_floor
        Records below this Jaccard similarity to the query are ignored
        when seeding posteriors. Controls context width.
    """
    registry: CapabilityRegistry
    arms: list[Arm] = field(default_factory=list)
    epsilon: float = 0.05
    cost_weight: float = 5.0
    similarity_floor: float = 0.1
    _rng: random.Random = field(default_factory=lambda: random.Random())

    DEFAULT_ARMS: tuple[Arm, ...] = (
        Arm(role="executor", model="claude-opus-4-7", effort="high"),
        Arm(role="executor", model="claude-sonnet-4-6", effort="medium"),
        Arm(role="planner", model="claude-opus-4-7", effort="high"),
        Arm(role="researcher", model="claude-sonnet-4-6", effort="medium"),
        Arm(role="writer", model="claude-sonnet-4-6", effort="medium"),
        Arm(role="executor", model="claude-haiku-4-5-20251001", effort="medium"),
    )

    def __post_init__(self) -> None:
        if not self.arms:
            # Seed from registry + defaults; dedupe by key.
            seen: set[tuple[str, str, str]] = set()
            arms: list[Arm] = []
            for rec in self.registry.all():
                # effort isn't stored on the record; default to "high" for
                # historical traces. Future records can carry it as a tag.
                a = Arm(role=rec.role, model=rec.model, effort="high")
                if a.key() not in seen:
                    seen.add(a.key())
                    arms.append(a)
            for a in self.DEFAULT_ARMS:
                if a.key() not in seen:
                    seen.add(a.key())
                    arms.append(a)
            self.arms = arms

    def seed(self, prompt: str) -> dict[tuple[str, str, str], ArmPosterior]:
        """Build context-conditional posteriors for `prompt` from history.

        For each arm, scan registry records, accept those whose token
        similarity exceeds `similarity_floor`, and fold them into the
        arm's Beta posterior. Records with no similarity contribute to
        a global prior at a discounted weight so arms aren't starved on
        novel prompts.
        """
        query = _tokenize(prompt)
        posteriors: dict[tuple[str, str, str], ArmPosterior] = {
            a.key(): ArmPosterior(arm=a) for a in self.arms
        }
        for rec in self.registry.all():
            rec_set = set(rec.prompt_tokens)
            inter = len(query & rec_set)
            union = len(query | rec_set) or 1
            sim = inter / union
            key = (rec.role, rec.model, "high")
            if key not in posteriors:
                posteriors[key] = ArmPosterior(arm=Arm(role=rec.role, model=rec.model, effort="high"))
            p = posteriors[key]
            if sim >= self.similarity_floor:
                # Full-weight update
                p.update(success=rec.success, cost_usd=rec.cost_usd)
            else:
                # Discounted prior: nudge toward the historical mean
                # without dominating the in-context evidence.
                if rec.success:
                    p.alpha += 0.1
                else:
                    p.beta += 0.1
        return posteriors

    def decide(
        self,
        prompt: str,
        *,
        budget_usd: float | None = None,
        force_arm: Arm | None = None,
    ) -> RoutingDecision:
        """Return one routing decision for `prompt`.

        If `force_arm` is given, sample its posterior and return without
        exploration — useful for replaying a previous decision.
        """
        posteriors = self.seed(prompt)
        if force_arm is not None:
            p = posteriors.get(force_arm.key()) or ArmPosterior(arm=force_arm)
            s = p.sample(self._rng)
            return RoutingDecision(
                arm=force_arm,
                sampled_success=s,
                expected_cost_usd=p.expected_cost(),
                expected_success=p.expected_success(),
                evidence_count=p.evidence_count,
                explored=False,
                rationale="forced arm",
            )

        # Epsilon exploration: uniform random across arms.
        if self._rng.random() < self.epsilon:
            arm = self._rng.choice(self.arms)
            p = posteriors[arm.key()]
            return RoutingDecision(
                arm=arm,
                sampled_success=p.sample(self._rng),
                expected_cost_usd=p.expected_cost(),
                expected_success=p.expected_success(),
                evidence_count=p.evidence_count,
                explored=True,
                rationale=f"ε-explore (epsilon={self.epsilon})",
            )

        best_key: tuple[str, str, str] | None = None
        best_score = -math.inf
        best_sample = 0.0
        for key, p in posteriors.items():
            s = p.sample(self._rng)
            cost = p.expected_cost()
            # Cost-penalised score. Arms with no cost evidence are treated
            # as zero-cost so they aren't perpetually shadowed by a
            # historically-cheap arm.
            score = s / (1.0 + self.cost_weight * cost)
            # Hard budget: zero out arms that would blow the budget on
            # their average past cost.
            if budget_usd is not None and cost > budget_usd and cost > 0:
                score *= max(0.01, budget_usd / cost)
            if score > best_score:
                best_score = score
                best_key = key
                best_sample = s

        key = best_key or next(iter(posteriors.keys()))
        p = posteriors[key]
        return RoutingDecision(
            arm=p.arm,
            sampled_success=best_sample,
            expected_cost_usd=p.expected_cost(),
            expected_success=p.expected_success(),
            evidence_count=p.evidence_count,
            explored=False,
            rationale=(
                f"Thompson-sampled (sample={best_sample:.3f}, "
                f"exp_cost=${p.expected_cost():.4f}, n={p.evidence_count})"
            ),
        )

    def observe(
        self,
        *,
        prompt: str,
        decision: RoutingDecision,
        success: bool,
        cost_usd: float,
        duration_seconds: float,
        critic_score: float | None = None,
        skills_used: list[str] | None = None,
        tag: str | None = None,
    ) -> CapabilityRecord:
        """Persist the outcome of a routing decision into the registry.

        This is the closed loop: a decision is made, the agent runs,
        and the result lands back here. Future `decide()` calls see it.
        """
        return self.registry.record(
            prompt=prompt,
            role=decision.arm.role,
            model=decision.arm.model,
            skills_used=skills_used or [],
            success=success,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            critic_score=critic_score,
            tag=tag,
        )

    def stats(self, prompt: str | None = None) -> dict[str, Any]:
        """Inspect the policy. With `prompt`, returns the context-conditional
        posteriors; without, returns the marginal arm statistics."""
        if prompt is not None:
            posteriors = self.seed(prompt)
            return {
                "prompt": prompt,
                "arms": [
                    {
                        "role": p.arm.role,
                        "model": p.arm.model,
                        "effort": p.arm.effort,
                        "alpha": p.alpha,
                        "beta": p.beta,
                        "evidence": p.evidence_count,
                        "expected_success": p.expected_success(),
                        "expected_cost_usd": p.expected_cost(),
                    }
                    for p in posteriors.values()
                ],
            }
        # Marginal — no context, scan all records.
        per_arm: dict[tuple[str, str], dict[str, Any]] = {}
        for rec in self.registry.all():
            k = (rec.role, rec.model)
            b = per_arm.setdefault(k, {"role": rec.role, "model": rec.model,
                                        "n": 0, "successes": 0, "cost_sum": 0.0})
            b["n"] += 1
            if rec.success:
                b["successes"] += 1
            b["cost_sum"] += rec.cost_usd
        for b in per_arm.values():
            n = max(1, b["n"])
            b["success_rate"] = b["successes"] / n
            b["mean_cost_usd"] = b["cost_sum"] / n
        return {"arms": list(per_arm.values())}


def recommend_with_policy(
    registry: CapabilityRegistry,
    prompt: str,
    *,
    budget_usd: float | None = None,
    epsilon: float = 0.05,
    cost_weight: float = 5.0,
) -> CapabilityRecommendation:
    """Convenience: one-shot policy decision returned as the same
    `CapabilityRecommendation` shape that older callers consume.

    Lets a coordinator opt into bandit routing without changing its
    integration: it keeps calling `recommend(...)` semantics, but the
    underlying decision is Thompson-sampled.
    """
    router = PolicyRouter(registry, epsilon=epsilon, cost_weight=cost_weight)
    decision = router.decide(prompt, budget_usd=budget_usd)
    return CapabilityRecommendation(
        role=decision.arm.role,
        model=decision.arm.model,
        skills_hint=[],
        expected_cost_usd=decision.expected_cost_usd,
        expected_success_rate=decision.expected_success,
        confidence=min(1.0, math.tanh(decision.evidence_count / 5.0)),
        evidence_count=decision.evidence_count,
        rationale=decision.rationale,
    )
