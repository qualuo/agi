"""ExperimentRunner demo — A/B experiments with guardrails as a runtime
primitive.

The story this demo tells:

  EvolutionEngine + TicketOracle propose changes ("ship a cheaper
  model", "raise the cost cap", "swap the prompt"). ExperimentRunner
  is the *gate* that turns those proposals into measurable, reversible
  production rollouts.

  Every change ships behind an experiment with:
    * a frozen primary metric                  (the thing we're optimizing)
    * predeclared guardrails                   (what cannot regress)
    * deterministic traffic assignment         (same tenant → same arm)
    * Bayesian or Welch-test stopping rules    (no peeking, no p-hacking)
    * an append-only audit log of every assignment and decision

  Investors get one sentence: *"Every product change is a measurable
  experiment with predeclared guardrails. Nothing ships without a
  positive result on the primary metric without breaching a guardrail."*

Four scenes:

  Scene 1 — A clear win
      Treatment lifts the primary metric (success rate) by 30pp on
      synthetic traffic. The runner ships it automatically.

  Scene 2 — A clear loss
      Treatment regresses the primary metric. The runner kills it.

  Scene 3 — A guardrail breach
      Treatment is cheaper (the primary metric) but tanks success
      rate below a 5pp absolute floor. The runner triggers an
      emergency kill on the guardrail breach, *before* the primary
      metric converges.

  Scene 4 — Live end-to-end with RuntimeDriver
      Register an experiment, route 50 real (FakeAgent) tickets
      through `driver.submit_with_experiment`, and watch the runner
      auto-record observations as receipts come back.

No API key required — uses FakeAgent.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agi.driver import RuntimeDriver, TicketRequest
from agi.experiments import (
    DECISION_KILL,
    DECISION_SHIP,
    EXP_KILLED,
    EXP_SHIPPED,
    Experiment,
    ExperimentRunner,
    Guardrail,
    INTERPRET_ABS,
    INTERPRET_ABS_DELTA,
    METRIC_COST_USD,
    METRIC_LATENCY_S,
    METRIC_P_SUCCESS,
    Variant,
)
from agi.memory import Memory
from agi.runtime import Runtime
from agi.skills import SkillLibrary


# --------------------------------------------------------------------
# FakeAgent (no API key)
# --------------------------------------------------------------------


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score = None
        self.extra_system = None
        self.messages = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        if self.model == "claude-haiku-4-5":
            self.usage.input_tokens += 60
            self.usage.output_tokens += 25
        else:
            self.usage.input_tokens += 220
            self.usage.output_tokens += 95
        return f"answered with {self.model}"

    def attach_tool_synth(self, *a, **kw):
        pass

    def attach_delegation(self, *a, **kw):
        pass

    def reset(self) -> None:
        self.usage = FakeUsage()


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------


def hr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def render_status(runner: ExperimentRunner, name: str) -> None:
    s = runner.status(name)
    print(f"\nexperiment={s.name}  status={s.status}")
    for v in s.variants:
        print(
            f"  {v.name:>9s}  samples={v.samples:4d}  "
            f"{s.primary_metric}={v.primary_mean:7.4f}  "
            f"±{v.primary_std_err:.4f}"
        )
    if s.best_variant:
        ci = s.primary_lift_ci or (0.0, 0.0)
        print(
            f"  best={s.best_variant}  lift={s.primary_lift:+.3f}  "
            f"ci=[{ci[0]:+.3f}, {ci[1]:+.3f}]  "
            f"prob_treatment_better={s.prob_treatment_better:.3f}"
        )
    if s.guardrail_breaches:
        for g in s.guardrail_breaches:
            mark = "BREACH" if g.breached else "ok"
            print(
                f"  guardrail[{g.metric}]: observed={g.observed:.4f} "
                f"tol={g.tolerance} interpret={g.interpret} ({mark})"
            )
    print(f"  decision={s.decision}: {s.reason}")


# --------------------------------------------------------------------
# Scene 1 — clear win
# --------------------------------------------------------------------


def scene_clear_win() -> None:
    hr("Scene 1 — Treatment ships on a clear primary-metric lift")
    runner = ExperimentRunner(rng_seed=101)
    runner.register(Experiment(
        name="prompt-tweak-v3",
        variants=[
            Variant("control", description="current system prompt"),
            Variant("treatment", overrides={"system_prompt_extra": "Be concise. Be precise."},
                    description="add concise/precise nudge"),
        ],
        primary_metric=METRIC_P_SUCCESS,
        direction="max",
        traffic_split=[0.5, 0.5],
        min_samples_per_variant=200,
        posterior_samples=400,
    ))
    rng = random.Random(1)
    for i in range(800):
        a = runner.assign("prompt-tweak-v3", tenant_id=f"tenant{i % 32}")
        if a is None:
            break
        true_rate = 0.85 if a.variant == "treatment" else 0.55
        runner.record(
            "prompt-tweak-v3", a.variant,
            success=rng.random() < true_rate,
            cost_usd=0.05, latency_s=2.0,
        )
        if i % 200 == 199:
            outs = runner.evaluate_all()
            if outs.get("prompt-tweak-v3", ("",))[0] in (DECISION_SHIP, DECISION_KILL):
                break
    render_status(runner, "prompt-tweak-v3")
    e = runner.get("prompt-tweak-v3")
    print(f"\n→ outcome: status={e.status}, shipped={e.shipped_variant}")


# --------------------------------------------------------------------
# Scene 2 — clear loss
# --------------------------------------------------------------------


def scene_clear_loss() -> None:
    hr("Scene 2 — Treatment killed on a clear primary-metric regression")
    runner = ExperimentRunner(rng_seed=102)
    runner.register(Experiment(
        name="aggressive-quant-v2",
        variants=[
            Variant("control"),
            Variant("treatment", overrides={"model": "claude-haiku-4-5"}),
        ],
        primary_metric=METRIC_P_SUCCESS,
        direction="max",
        traffic_split=[0.5, 0.5],
        min_samples_per_variant=200,
        posterior_samples=400,
    ))
    rng = random.Random(2)
    for i in range(800):
        a = runner.assign("aggressive-quant-v2", tenant_id=f"tenant{i % 32}")
        if a is None:
            break
        true_rate = 0.35 if a.variant == "treatment" else 0.85
        runner.record(
            "aggressive-quant-v2", a.variant,
            success=rng.random() < true_rate,
            cost_usd=0.05, latency_s=2.0,
        )
        if i % 200 == 199:
            outs = runner.evaluate_all()
            if outs.get("aggressive-quant-v2", ("",))[0] in (DECISION_SHIP, DECISION_KILL):
                break
    render_status(runner, "aggressive-quant-v2")
    e = runner.get("aggressive-quant-v2")
    print(f"\n→ outcome: status={e.status}, decision_reason={e.decision_reason!r}")


# --------------------------------------------------------------------
# Scene 3 — guardrail breach kill
# --------------------------------------------------------------------


def scene_guardrail_breach() -> None:
    hr("Scene 3 — Guardrail breach overrides a tempting primary-metric win")
    runner = ExperimentRunner(rng_seed=103)
    # Optimizing cost — lower is better. Guardrail: success rate may not drop
    # more than 5pp below control. Treatment is much cheaper but tanks quality.
    runner.register(Experiment(
        name="cheaper-router-v1",
        variants=[
            Variant("control"),
            Variant("treatment", overrides={"model": "claude-haiku-4-5"}),
        ],
        primary_metric=METRIC_COST_USD,
        direction="min",
        traffic_split=[0.5, 0.5],
        min_samples_per_variant=200,
        posterior_samples=400,
        guardrails=[
            Guardrail(
                metric=METRIC_P_SUCCESS,
                direction="min",
                tolerance=-0.05,
                interpret=INTERPRET_ABS_DELTA,
            ),
            Guardrail(
                metric=METRIC_LATENCY_S,
                direction="max",
                tolerance=1.5,
                interpret="ratio",
            ),
        ],
    ))
    rng = random.Random(3)
    for i in range(800):
        a = runner.assign("cheaper-router-v1", tenant_id=f"tenant{i % 32}")
        if a is None:
            break
        if a.variant == "treatment":
            cost = rng.gauss(0.02, 0.002)
            success_rate = 0.50         # 35pp drop vs control — clear breach
            latency = rng.gauss(2.5, 0.1)
        else:
            cost = rng.gauss(0.08, 0.005)
            success_rate = 0.85
            latency = rng.gauss(2.0, 0.1)
        runner.record(
            "cheaper-router-v1", a.variant,
            success=rng.random() < success_rate,
            cost_usd=max(0.001, cost),
            latency_s=max(0.1, latency),
        )
        if i % 200 == 199:
            outs = runner.evaluate_all()
            if outs.get("cheaper-router-v1", ("",))[0] in (DECISION_SHIP, DECISION_KILL):
                break
    render_status(runner, "cheaper-router-v1")
    e = runner.get("cheaper-router-v1")
    print(f"\n→ outcome: status={e.status}, decision_reason={e.decision_reason!r}")


# --------------------------------------------------------------------
# Scene 4 — live driver integration
# --------------------------------------------------------------------


def scene_driver_e2e() -> None:
    hr("Scene 4 — RuntimeDriver routes 50 tickets through an experiment")
    tmp = Path(tempfile.mkdtemp())
    rt = Runtime(
        memory=Memory(path=tmp / "memory.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
    )
    driver = RuntimeDriver(runtime=rt)
    runner = driver.experiments
    runner.register(Experiment(
        name="cheap-router-live",
        variants=[
            Variant("control"),
            Variant("treatment", overrides={"model": "claude-haiku-4-5"}),
        ],
        primary_metric=METRIC_COST_USD,
        direction="min",
        traffic_split=[0.5, 0.5],
        min_samples_per_variant=20,
        posterior_samples=200,
    ))
    handles = []
    for i in range(50):
        req = TicketRequest(
            intent=f"task {i}: summarize topic #{i % 7}",
            tenant_id=f"tenant{i % 8}",
            budget_usd=0.50,
        )
        handles.append(driver.submit_with_experiment(req, "cheap-router-live"))
    for h in handles:
        h.result(timeout=10.0)
    render_status(runner, "cheap-router-live")
    # Auto-decide.
    runner.evaluate_all()
    e = runner.get("cheap-router-live")
    print(f"\n→ outcome: status={e.status}, decision_reason={e.decision_reason!r}")
    print(f"   driver.stats: completed={driver.stats()['completed']}, "
          f"submitted={driver.stats()['submitted']}")


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------


def main() -> None:
    scene_clear_win()
    scene_clear_loss()
    scene_guardrail_breach()
    scene_driver_e2e()
    print()
    print("=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
