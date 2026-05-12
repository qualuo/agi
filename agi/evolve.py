"""Evolution engine — closed-loop self-improvement over agent strategies.

The runtime exposes many knobs: model, effort, role, skill overlays, system
prompt nudges. A coordination engine that picks one fixed configuration is
leaving capability on the table — different prompts want different setups,
and the right setup drifts as skills accrete and tools get synthesized.

`EvolutionEngine` runs an evolutionary search over `Strategy` variants on a
benchmark drawn from `SelfEvalBank`, computes fitness as
``pass_rate − cost_weight × mean_cost_usd``, and *promotes* winners back
into the rest of the runtime:

  - Updates `PolicyRouter` posteriors so future routing prefers the
    winning (role, model, effort) on similar prompts.
  - Records every outcome in `CapabilityRegistry` so the registry's
    similarity-weighted recommendations include this generation's data.
  - Mines a skill from the winner's successful traces (via `skillmine`)
    and saves it to `SkillLibrary` if it doesn't already exist.
  - Eval-gates the promotion: if the winning strategy doesn't beat the
    baseline pass rate by `min_improvement`, no promotion happens. The
    bank stays the source of truth for "is the system actually better?".

Output is an `EvolutionResult` with per-generation fitness/pass-rate/cost
curves — the artifact a coordination engine (or an investor demo)
displays as proof that the runtime improves itself with use.

The engine is dependency-injected on the runner so tests stay hermetic:
pass a deterministic ``runner(strategy, item) -> (passed, text, cost)`` and
no Anthropic client is ever constructed. ``runtime_runner()`` builds a
runner that drives a real ``Runtime`` for production use.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agi.capabilities import CapabilityRegistry
from agi.policy import Arm, PolicyRouter, RoutingDecision
from agi.runtime import Runtime, SessionConfig
from agi.selfeval import EvalItem, SelfEvalBank
from agi.skillmine import SkillCandidate, propose_skill_from_cluster
from agi.skills import Skill, SkillLibrary


@dataclass
class Strategy:
    """A named candidate configuration the engine evaluates.

    A Strategy is *just data* — the engine never mutates a SessionConfig
    in place, it builds new Strategies from old ones via `mutate()`.
    Lineage is tracked via `parent_name` so the result includes a
    family tree.
    """
    name: str
    config: SessionConfig
    skill_overlay: list[str] = field(default_factory=list)
    system_prompt_extra: str | None = None
    parent_name: str | None = None
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def arm(self) -> Arm:
        return Arm(
            role=self.config.role or "executor",
            model=self.config.model,
            effort=self.config.effort,
        )

    def materialized_config(self) -> SessionConfig:
        """Return a SessionConfig with `system_prompt_extra` overlay applied."""
        base = self.config
        extra_pieces = [p for p in (base.system_prompt_extra, self.system_prompt_extra) if p]
        return SessionConfig(
            **{
                **base.__dict__,
                "system_prompt_extra": "\n\n".join(extra_pieces) or None,
            }
        )


@dataclass
class StrategyOutcome:
    """One (strategy, eval-item) result."""
    strategy_name: str
    item_id: str
    prompt: str
    final_text: str
    passed: bool
    cost_usd: float
    duration_seconds: float


@dataclass
class GenerationReport:
    """Aggregate of one generation's evaluation."""
    generation: int
    strategies: list[Strategy]
    outcomes: list[StrategyOutcome]
    fitness_by_strategy: dict[str, float]
    pass_rate_by_strategy: dict[str, float]
    mean_cost_by_strategy: dict[str, float]
    best_strategy_name: str
    total_cost_usd: float
    duration_seconds: float

    def best_strategy(self) -> Strategy:
        for s in self.strategies:
            if s.name == self.best_strategy_name:
                return s
        raise KeyError(self.best_strategy_name)


@dataclass
class PromotionRecord:
    """What the engine promoted after a generation, eval-gate permitting."""
    strategy_name: str
    skill_saved: str | None
    skill_was_new: bool
    capability_records: int
    policy_arm: tuple[str, str, str] | None
    rejected_reason: str | None = None

    @property
    def promoted(self) -> bool:
        return self.rejected_reason is None


@dataclass
class EvolutionResult:
    """The full audit trail of an `evolve()` run."""
    generations: list[GenerationReport]
    best_strategy: Strategy
    best_fitness: float
    fitness_curve: list[float]
    pass_rate_curve: list[float]
    mean_cost_curve: list[float]
    promotions: list[PromotionRecord]
    benchmark_size: int
    total_cost_usd: float
    total_duration_seconds: float

    def improved(self) -> bool:
        """True iff the best fitness in the final generation exceeds the first."""
        if len(self.fitness_curve) < 2:
            return False
        return self.fitness_curve[-1] > self.fitness_curve[0]

    def summary(self) -> dict[str, Any]:
        return {
            "generations": len(self.generations),
            "benchmark_size": self.benchmark_size,
            "best_strategy": self.best_strategy.name,
            "best_fitness": self.best_fitness,
            "first_fitness": self.fitness_curve[0] if self.fitness_curve else None,
            "improved": self.improved(),
            "fitness_curve": list(self.fitness_curve),
            "pass_rate_curve": list(self.pass_rate_curve),
            "mean_cost_curve": list(self.mean_cost_curve),
            "promotions": [p.strategy_name for p in self.promotions if p.promoted],
            "total_cost_usd": self.total_cost_usd,
            "total_duration_seconds": self.total_duration_seconds,
        }


# (strategy, item) -> (passed, final_text, cost_usd)
StrategyRunner = Callable[[Strategy, EvalItem], tuple[bool, str, float]]


_EFFORT_LADDER: tuple[str, ...] = ("low", "medium", "high")
_DEFAULT_MODELS: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
)
_DEFAULT_ROLES: tuple[str, ...] = ("executor", "planner", "researcher", "writer")
_NUDGE_HINTS: tuple[str, ...] = (
    "Be concise; one sentence final answer.",
    "Verify the answer against the prompt before responding.",
    "Decompose into sub-steps before answering.",
    "Prefer exact substrings from the prompt in your answer.",
)


class EvolutionEngine:
    """Evolutionary search over agent strategies, gated by an eval bank.

    The engine is intentionally dependency-injected: pass a `runner`
    callable and (optionally) a `CapabilityRegistry`, `PolicyRouter`,
    `SkillLibrary`, and `SelfEvalBank` to wire promotion in. Anything
    not supplied is treated as opt-out for that promotion channel.

    Determinism: when `seed` is supplied, mutation is reproducible. The
    runner itself is not seeded — that's the caller's responsibility.
    """

    def __init__(
        self,
        runner: StrategyRunner,
        *,
        registry: CapabilityRegistry | None = None,
        policy: PolicyRouter | None = None,
        skill_library: SkillLibrary | None = None,
        eval_bank: SelfEvalBank | None = None,
        cost_weight: float = 5.0,
        seed: int | None = None,
    ) -> None:
        self.runner = runner
        self.registry = registry
        self.policy = policy
        self.skills = skill_library
        self.bank = eval_bank
        self.cost_weight = float(cost_weight)
        self._rng = random.Random(seed)

    # ----- evaluation -----

    def fitness(self, pass_rate: float, mean_cost: float) -> float:
        return pass_rate - self.cost_weight * mean_cost

    def evaluate(
        self,
        strategies: list[Strategy],
        benchmark: list[EvalItem],
        *,
        generation: int = 0,
    ) -> GenerationReport:
        if not strategies:
            raise ValueError("evaluate() needs at least one strategy")
        if not benchmark:
            raise ValueError("evaluate() needs at least one benchmark item")
        start = time.time()
        outcomes: list[StrategyOutcome] = []
        for s in strategies:
            for item in benchmark:
                t0 = time.time()
                try:
                    passed, text, cost = self.runner(s, item)
                except Exception as e:  # runner crash = item failure
                    passed, text, cost = False, f"{type(e).__name__}: {e}", 0.0
                outcomes.append(StrategyOutcome(
                    strategy_name=s.name,
                    item_id=item.id,
                    prompt=item.prompt,
                    final_text=text,
                    passed=bool(passed),
                    cost_usd=float(cost),
                    duration_seconds=time.time() - t0,
                ))
        pass_rates: dict[str, float] = {}
        mean_costs: dict[str, float] = {}
        for s in strategies:
            mine = [o for o in outcomes if o.strategy_name == s.name]
            pass_rates[s.name] = sum(1 for o in mine if o.passed) / max(1, len(mine))
            mean_costs[s.name] = (
                statistics.fmean(o.cost_usd for o in mine) if mine else 0.0
            )
        fitness = {
            name: self.fitness(pass_rates[name], mean_costs[name])
            for name in pass_rates
        }
        best = max(fitness.items(), key=lambda kv: kv[1])[0]
        return GenerationReport(
            generation=generation,
            strategies=list(strategies),
            outcomes=outcomes,
            fitness_by_strategy=fitness,
            pass_rate_by_strategy=pass_rates,
            mean_cost_by_strategy=mean_costs,
            best_strategy_name=best,
            total_cost_usd=sum(o.cost_usd for o in outcomes),
            duration_seconds=time.time() - start,
        )

    # ----- selection / mutation -----

    def select(
        self,
        report: GenerationReport,
        *,
        top_k: int,
        min_pass_rate: float = 0.0,
    ) -> list[Strategy]:
        """Return the top-k strategies by fitness, filtered by pass rate."""
        ranked = sorted(
            report.strategies,
            key=lambda s: -report.fitness_by_strategy[s.name],
        )
        kept: list[Strategy] = []
        for s in ranked:
            if report.pass_rate_by_strategy[s.name] >= min_pass_rate:
                kept.append(s)
            if len(kept) >= top_k:
                break
        if not kept:
            # Always carry the best so search doesn't collapse.
            kept = [ranked[0]]
        return kept

    def mutate(
        self,
        parents: list[Strategy],
        *,
        num_children: int,
        generation: int,
        available_skill_names: list[str] | None = None,
    ) -> list[Strategy]:
        """Produce `num_children` mutated children from `parents`.

        Mutation operators (each child applies one):
          1. effort jitter (low/medium/high)
          2. model swap (within the default ladder)
          3. role swap
          4. nudge add (append a precision/decomposition hint)
          5. skill overlay add (from `available_skill_names`)
        """
        if not parents:
            raise ValueError("mutate() needs at least one parent")
        children: list[Strategy] = []
        for i in range(num_children):
            parent = parents[i % len(parents)]
            op = self._rng.randint(1, 5)
            child_name = f"g{generation}-{parent.name}-m{i}"
            new_cfg = SessionConfig(**parent.config.__dict__)
            new_overlay = list(parent.skill_overlay)
            new_extra = parent.system_prompt_extra
            if op == 1:
                new_cfg.effort = self._rng.choice(
                    [e for e in _EFFORT_LADDER if e != parent.config.effort]
                    or list(_EFFORT_LADDER)
                )
            elif op == 2:
                new_cfg.model = self._rng.choice(
                    [m for m in _DEFAULT_MODELS if m != parent.config.model]
                    or list(_DEFAULT_MODELS)
                )
            elif op == 3:
                new_cfg.role = self._rng.choice(
                    [r for r in _DEFAULT_ROLES if r != parent.config.role]
                    or list(_DEFAULT_ROLES)
                )
            elif op == 4:
                hint = self._rng.choice(_NUDGE_HINTS)
                new_extra = (new_extra + "\n" + hint) if new_extra else hint
            elif op == 5:
                pool = [n for n in (available_skill_names or []) if n not in new_overlay]
                if pool:
                    new_overlay.append(self._rng.choice(pool))
                else:
                    # Fall back to a nudge if no skills available
                    hint = self._rng.choice(_NUDGE_HINTS)
                    new_extra = (new_extra + "\n" + hint) if new_extra else hint
            children.append(Strategy(
                name=child_name,
                config=new_cfg,
                skill_overlay=new_overlay,
                system_prompt_extra=new_extra,
                parent_name=parent.name,
                generation=generation,
                metadata={"mutation_op": op},
            ))
        return children

    # ----- promotion (the closed loop) -----

    def promote(
        self,
        report: GenerationReport,
        *,
        baseline_pass_rate: float | None,
        min_improvement: float = 0.0,
    ) -> PromotionRecord:
        """Apply winner outcomes to the registry/policy/skills, eval-gated.

        The gate: if `baseline_pass_rate` is provided and the winning
        strategy's pass rate isn't at least
        ``baseline_pass_rate + min_improvement``, nothing is promoted.
        """
        winner = report.best_strategy()
        winner_pass_rate = report.pass_rate_by_strategy[winner.name]
        if (
            baseline_pass_rate is not None
            and winner_pass_rate < baseline_pass_rate + min_improvement
        ):
            return PromotionRecord(
                strategy_name=winner.name,
                skill_saved=None,
                skill_was_new=False,
                capability_records=0,
                policy_arm=None,
                rejected_reason=(
                    f"winner pass_rate {winner_pass_rate:.2f} < "
                    f"baseline {baseline_pass_rate:.2f} + min_improvement "
                    f"{min_improvement:.2f}"
                ),
            )

        cap_recorded = 0
        winner_outcomes = [o for o in report.outcomes if o.strategy_name == winner.name]

        # 1. CapabilityRegistry — durable trace of what worked.
        if self.registry is not None:
            for o in winner_outcomes:
                self.registry.record(
                    prompt=o.prompt,
                    role=winner.config.role or "executor",
                    model=winner.config.model,
                    skills_used=winner.skill_overlay,
                    success=o.passed,
                    cost_usd=o.cost_usd,
                    duration_seconds=o.duration_seconds,
                    critic_score=1.0 if o.passed else 0.0,
                    tag=f"evolve:gen{report.generation}",
                )
                cap_recorded += 1

        # 2. PolicyRouter posteriors — bias future routing toward winner.
        arm_key: tuple[str, str, str] | None = None
        if self.policy is not None:
            arm = winner.arm()
            arm_key = arm.key()
            # Add the arm if missing so routing can see it.
            if arm_key not in {a.key() for a in self.policy.arms}:
                self.policy.arms.append(arm)
            for o in winner_outcomes:
                # Build a synthetic decision so observe() wires through.
                decision = RoutingDecision(
                    arm=arm,
                    sampled_success=1.0 if o.passed else 0.0,
                    expected_cost_usd=o.cost_usd,
                    expected_success=winner_pass_rate,
                    evidence_count=len(winner_outcomes),
                    explored=False,
                    rationale=f"evolve:gen{report.generation}",
                )
                self.policy.observe(
                    prompt=o.prompt,
                    decision=decision,
                    success=o.passed,
                    cost_usd=o.cost_usd,
                    duration_seconds=o.duration_seconds,
                    critic_score=1.0 if o.passed else 0.0,
                    skills_used=winner.skill_overlay,
                )

        # 3. SkillLibrary — mine a skill from successful winner traces.
        skill_saved: str | None = None
        skill_was_new = False
        if self.skills is not None:
            successful = [(o.prompt, o.final_text) for o in winner_outcomes if o.passed]
            if len(successful) >= 2:  # need a small cluster to be a skill
                candidate = propose_skill_from_cluster(
                    [p for p, _ in successful],
                    [t for _, t in successful],
                    name_hint=f"evolved_{winner.name.replace('-', '_')}"[:48],
                )
                skill = candidate.to_skill()
                # Don't clobber an existing skill; coexist.
                if self.skills.get(skill.name) is None:
                    self.skills.save(skill)
                    skill_saved = skill.name
                    skill_was_new = True
                else:
                    skill_saved = skill.name
                    skill_was_new = False

        # 4. SelfEvalBank — every passed (prompt, text) becomes a regression item.
        if self.bank is not None:
            for o in winner_outcomes:
                if not o.passed or not o.final_text:
                    continue
                self.bank.auto_mine(
                    prompt=o.prompt,
                    final_text=o.final_text,
                    critic_score=1.0,
                    critic_threshold=0.5,
                    tags=[f"evolve:gen{report.generation}"],
                )

        return PromotionRecord(
            strategy_name=winner.name,
            skill_saved=skill_saved,
            skill_was_new=skill_was_new,
            capability_records=cap_recorded,
            policy_arm=arm_key,
        )

    # ----- the loop -----

    def evolve(
        self,
        seed_strategies: list[Strategy],
        benchmark: list[EvalItem],
        *,
        generations: int = 4,
        top_k: int = 2,
        children_per_gen: int = 3,
        min_pass_rate_for_selection: float = 0.0,
        promote_each_generation: bool = True,
        promotion_min_improvement: float = 0.0,
    ) -> EvolutionResult:
        """Run `generations` of evaluate → select → mutate → promote.

        The first generation evaluates the seed strategies as-is. Each
        subsequent generation keeps the top-k parents (elitism) and
        appends `children_per_gen` mutated children. Promotion runs
        after every generation by default; baseline = first generation's
        best pass rate, so the eval-gate only allows non-regressions.
        """
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if not seed_strategies:
            raise ValueError("evolve() needs at least one seed strategy")
        loop_start = time.time()

        all_reports: list[GenerationReport] = []
        promotions: list[PromotionRecord] = []
        fitness_curve: list[float] = []
        pass_rate_curve: list[float] = []
        mean_cost_curve: list[float] = []

        current = list(seed_strategies)
        baseline_pass_rate: float | None = None
        for gen_idx in range(generations):
            report = self.evaluate(current, benchmark, generation=gen_idx)
            all_reports.append(report)
            best_name = report.best_strategy_name
            fitness_curve.append(report.fitness_by_strategy[best_name])
            pass_rate_curve.append(report.pass_rate_by_strategy[best_name])
            mean_cost_curve.append(
                statistics.fmean(report.mean_cost_by_strategy.values())
                if report.mean_cost_by_strategy else 0.0
            )
            if baseline_pass_rate is None:
                baseline_pass_rate = report.pass_rate_by_strategy[best_name]

            if promote_each_generation:
                promotions.append(self.promote(
                    report,
                    baseline_pass_rate=baseline_pass_rate,
                    min_improvement=promotion_min_improvement,
                ))

            if gen_idx == generations - 1:
                break

            parents = self.select(
                report,
                top_k=top_k,
                min_pass_rate=min_pass_rate_for_selection,
            )
            available = [s.name for s in self.skills.all()] if self.skills else None
            children = self.mutate(
                parents,
                num_children=children_per_gen,
                generation=gen_idx + 1,
                available_skill_names=available,
            )
            current = parents + children

        # Best strategy across all generations (not just final).
        best_overall_name, best_overall_fitness = max(
            (
                (s.name, r.fitness_by_strategy[s.name])
                for r in all_reports for s in r.strategies
            ),
            key=lambda kv: kv[1],
        )
        # Pick a representative Strategy object for the winner name.
        best_overall: Strategy | None = None
        for r in all_reports:
            for s in r.strategies:
                if s.name == best_overall_name:
                    best_overall = s
                    break
            if best_overall:
                break
        assert best_overall is not None

        return EvolutionResult(
            generations=all_reports,
            best_strategy=best_overall,
            best_fitness=best_overall_fitness,
            fitness_curve=fitness_curve,
            pass_rate_curve=pass_rate_curve,
            mean_cost_curve=mean_cost_curve,
            promotions=promotions,
            benchmark_size=len(benchmark),
            total_cost_usd=sum(r.total_cost_usd for r in all_reports),
            total_duration_seconds=time.time() - loop_start,
        )


# ----- runtime adapter -----


def runtime_runner(runtime: Runtime) -> StrategyRunner:
    """Build a StrategyRunner that drives a real Runtime.

    Each (strategy, item) pair gets its own short-lived session so
    evaluations don't cross-contaminate each other's context. The
    skill overlay is honored via `system_prompt_extra` (not full skill
    loading, which is per-session): we inject the skill bodies inline
    so the strategy is reproducible from its data alone.
    """
    library = SkillLibrary()

    def _runner(strategy: Strategy, item: EvalItem) -> tuple[bool, str, float]:
        cfg = strategy.materialized_config()
        # Attach skill bodies inline if any are referenced.
        if strategy.skill_overlay:
            blocks: list[str] = []
            for name in strategy.skill_overlay:
                sk = library.get(name)
                if sk is not None:
                    blocks.append(sk.to_prompt_block())
            if blocks:
                inline = "\n\n".join(blocks)
                cfg.system_prompt_extra = (
                    f"{cfg.system_prompt_extra}\n\n{inline}"
                    if cfg.system_prompt_extra else inline
                )
        sid = runtime.create_session(cfg)
        try:
            text = runtime.chat(sid, item.prompt)
            cost = runtime.get_session(sid).state.total_cost_usd
        finally:
            try:
                runtime.end_session(sid)
            except KeyError:
                pass
        return item.predicate()(text), text, cost

    return _runner


def default_seed_strategies() -> list[Strategy]:
    """A small, opinionated starting population.

    Three strategies that span the tradeoff space: cheap-fast, balanced,
    expensive-careful. Evolution will mutate from here.
    """
    return [
        Strategy(
            name="seed-haiku-fast",
            config=SessionConfig(
                model="claude-haiku-4-5-20251001",
                effort="medium",
                role="executor",
                use_skills=False,
            ),
            generation=0,
        ),
        Strategy(
            name="seed-sonnet-balanced",
            config=SessionConfig(
                model="claude-sonnet-4-6",
                effort="medium",
                role="executor",
                use_skills=True,
            ),
            generation=0,
        ),
        Strategy(
            name="seed-opus-careful",
            config=SessionConfig(
                model="claude-opus-4-7",
                effort="high",
                role="executor",
                use_skills=True,
            ),
            system_prompt_extra="Verify the answer against the prompt before responding.",
            generation=0,
        ),
    ]
