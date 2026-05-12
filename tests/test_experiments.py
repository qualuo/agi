"""Tests for `ExperimentRunner` — A/B experiments with guardrails.

Hermetic. No API calls. The driver integration tests use a FakeAgent
identical in shape to `tests/test_oracle.py`.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.driver import COMPLETED, REJECTED, RuntimeDriver, TicketRequest
from agi.experiments import (
    BINARY_METRICS,
    CONTINUOUS_METRICS,
    DECISION_CONTINUE,
    DECISION_INCONCLUSIVE,
    DECISION_KILL,
    DECISION_SHIP,
    EXP_EVT_OBSERVED,
    EXP_EVT_REGISTERED,
    EXP_INCONCLUSIVE,
    EXP_KILLED,
    EXP_PAUSED,
    EXP_RUNNING,
    EXP_SHIPPED,
    Experiment,
    ExperimentRunner,
    Guardrail,
    INTERPRET_ABS,
    INTERPRET_ABS_DELTA,
    INTERPRET_RATIO,
    METRIC_BREACH_RATE,
    METRIC_COST_PER_SUCCESS,
    METRIC_COST_USD,
    METRIC_LATENCY_S,
    METRIC_P_SUCCESS,
    METRIC_REFUND_RATE,
    METRIC_REFUND_USD,
    METRIC_REJECT_RATE,
    MetricStats,
    Variant,
    _hash_bucket,
    _normal_cdf,
    _prob_better_binary,
    _two_proportion_p,
    _welch_t,
)
from agi.memory import Memory
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


# ---------- math helpers ----------


class TestMathHelpers(unittest.TestCase):
    def test_hash_bucket_deterministic(self) -> None:
        a = _hash_bucket("acme")
        b = _hash_bucket("acme")
        self.assertEqual(a, b)
        self.assertGreaterEqual(a, 0.0)
        self.assertLess(a, 1.0)

    def test_hash_bucket_distinct(self) -> None:
        a = _hash_bucket("acme")
        b = _hash_bucket("beta")
        self.assertNotEqual(a, b)

    def test_normal_cdf_known_values(self) -> None:
        self.assertAlmostEqual(_normal_cdf(0.0), 0.5, places=4)
        self.assertGreater(_normal_cdf(1.96), 0.97)
        self.assertLess(_normal_cdf(-1.96), 0.03)

    def test_welch_t_treatment_clearly_better(self) -> None:
        # Treatment mean is much higher with low variance and big samples.
        t, p = _welch_t(0.8, 0.01, 200, 0.5, 0.01, 200)
        self.assertGreater(t, 5)
        self.assertLess(p, 0.001)

    def test_welch_t_small_sample(self) -> None:
        # Treats <2 samples as non-significant.
        t, p = _welch_t(1.0, 0.0, 1, 0.5, 0.0, 1)
        self.assertEqual((t, p), (0.0, 0.5))

    def test_prob_better_binary(self) -> None:
        rng = random.Random(42)
        # Clearly better arm should win ~always.
        p = _prob_better_binary(180, 200, 80, 200, samples=400, rng=rng)
        self.assertGreater(p, 0.99)
        # Equal arms ~ 0.5.
        rng = random.Random(43)
        p = _prob_better_binary(100, 200, 100, 200, samples=400, rng=rng)
        self.assertGreater(p, 0.3)
        self.assertLess(p, 0.7)

    def test_prob_better_binary_empty(self) -> None:
        p = _prob_better_binary(0, 0, 0, 0, samples=200)
        self.assertEqual(p, 0.5)

    def test_two_proportion_p(self) -> None:
        # Clear win for a.
        p = _two_proportion_p(180, 200, 80, 200, one_sided_better=True)
        self.assertLess(p, 0.001)
        # Clear loss.
        p = _two_proportion_p(80, 200, 180, 200, one_sided_better=True)
        self.assertGreater(p, 0.99)
        # No samples.
        p = _two_proportion_p(0, 0, 0, 0)
        self.assertAlmostEqual(p, 0.5, places=2)


# ---------- variant / experiment validation ----------


class TestValidation(unittest.TestCase):
    def test_variant_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            Variant(name="")

    def test_variant_rejects_non_dict_overrides(self) -> None:
        with self.assertRaises(ValueError):
            Variant(name="x", overrides="not a dict")  # type: ignore[arg-type]

    def test_guardrail_rejects_unknown_metric(self) -> None:
        with self.assertRaises(ValueError):
            Guardrail(metric="not_a_metric", direction="max", tolerance=1.0)

    def test_guardrail_rejects_bad_direction(self) -> None:
        with self.assertRaises(ValueError):
            Guardrail(metric=METRIC_P_SUCCESS, direction="sideways", tolerance=1.0)

    def test_guardrail_rejects_bad_interpret(self) -> None:
        with self.assertRaises(ValueError):
            Guardrail(
                metric=METRIC_P_SUCCESS, direction="max",
                tolerance=1.0, interpret="rainbow",
            )

    def test_guardrail_rejects_negative_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            Guardrail(metric=METRIC_P_SUCCESS, direction="max", tolerance=-1.0)

    def test_experiment_requires_two_variants(self) -> None:
        with self.assertRaises(ValueError):
            Experiment(
                name="x",
                variants=[Variant("only")],
                primary_metric=METRIC_P_SUCCESS,
            )

    def test_experiment_unique_variant_names(self) -> None:
        with self.assertRaises(ValueError):
            Experiment(
                name="x",
                variants=[Variant("a"), Variant("a")],
                primary_metric=METRIC_P_SUCCESS,
            )

    def test_experiment_traffic_split_default(self) -> None:
        e = Experiment(
            name="x",
            variants=[Variant("c"), Variant("t1"), Variant("t2")],
            primary_metric=METRIC_P_SUCCESS,
        )
        self.assertEqual(len(e.traffic_split), 3)
        self.assertAlmostEqual(sum(e.traffic_split), 1.0)

    def test_experiment_traffic_split_normalises(self) -> None:
        e = Experiment(
            name="x",
            variants=[Variant("c"), Variant("t")],
            primary_metric=METRIC_P_SUCCESS,
            traffic_split=[2.0, 8.0],
        )
        self.assertAlmostEqual(sum(e.traffic_split), 1.0)
        self.assertAlmostEqual(e.traffic_split[0], 0.2)
        self.assertAlmostEqual(e.traffic_split[1], 0.8)

    def test_experiment_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            Experiment(
                name="x",
                variants=[Variant("c"), Variant("t")],
                primary_metric=METRIC_P_SUCCESS,
                traffic_split=[0.5, 0.3, 0.2],
            )

    def test_experiment_rejects_unknown_primary(self) -> None:
        with self.assertRaises(ValueError):
            Experiment(
                name="x",
                variants=[Variant("c"), Variant("t")],
                primary_metric="not_a_metric",
            )


# ---------- assignment ----------


def _basic_exp(**kw: Any) -> Experiment:
    kwargs = dict(
        name="exp1",
        variants=[Variant("control"), Variant("treatment", overrides={"model": "cheap"})],
        primary_metric=METRIC_P_SUCCESS,
        direction="max",
        traffic_split=[0.5, 0.5],
        min_samples_per_variant=10,
        max_samples_per_variant=1000,
        posterior_samples=200,
    )
    kwargs.update(kw)
    return Experiment(**kwargs)


class TestAssignment(unittest.TestCase):
    def test_register_and_get(self) -> None:
        r = ExperimentRunner()
        e = r.register(_basic_exp())
        self.assertIs(r.get("exp1"), e)

    def test_register_duplicate_rejected(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        with self.assertRaises(ValueError):
            r.register(_basic_exp())

    def test_assign_unknown_returns_none(self) -> None:
        r = ExperimentRunner()
        self.assertIsNone(r.assign("missing"))

    def test_assign_deterministic_per_tenant(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        a1 = r.assign("exp1", tenant_id="acme")
        a2 = r.assign("exp1", tenant_id="acme")
        a3 = r.assign("exp1", tenant_id="acme")
        self.assertEqual(a1.variant, a2.variant)
        self.assertEqual(a2.variant, a3.variant)

    def test_assign_distribution_roughly_matches_split(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp(traffic_split=[0.2, 0.8]))
        counts = {"control": 0, "treatment": 0}
        for i in range(1000):
            a = r.assign("exp1", tenant_id=f"tenant{i}")
            counts[a.variant] += 1
        # 20/80 split: control should be in [10%, 30%] over 1000 samples.
        self.assertGreater(counts["control"], 100)
        self.assertLess(counts["control"], 300)
        self.assertGreater(counts["treatment"], 700)

    def test_assign_paused_returns_none(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        r.pause("exp1")
        self.assertIsNone(r.assign("exp1", tenant_id="acme"))

    def test_assign_terminal_returns_none(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        r.ship("exp1", variant="treatment")
        self.assertIsNone(r.assign("exp1", tenant_id="acme"))


# ---------- apply_to_config ----------


class TestApplyToConfig(unittest.TestCase):
    def test_apply_overrides_known_field(self) -> None:
        r = ExperimentRunner()
        e = r.register(_basic_exp())
        cfg = SessionConfig(model="opus")
        v = e.variant("treatment")
        new_cfg = r.apply_to_config(v, cfg)
        self.assertEqual(new_cfg.model, "cheap")
        # Original unchanged.
        self.assertEqual(cfg.model, "opus")

    def test_apply_no_overrides_returns_same(self) -> None:
        r = ExperimentRunner()
        e = r.register(_basic_exp())
        cfg = SessionConfig(model="opus")
        new_cfg = r.apply_to_config(e.control, cfg)
        self.assertEqual(new_cfg.model, "opus")

    def test_apply_unknown_override_stashed_in_metadata(self) -> None:
        r = ExperimentRunner()
        exp = _basic_exp()
        exp.variants[1].overrides = {"max_iterations": 5, "no_such_field": "x"}
        e = r.register(exp)
        cfg = SessionConfig()
        new_cfg = r.apply_to_config(e.variant("treatment"), cfg)
        self.assertEqual(new_cfg.max_iterations, 5)
        self.assertEqual(
            new_cfg.metadata.get("experiment_overrides", {}).get("no_such_field"),
            "x",
        )


# ---------- record + status (binary) ----------


_STABLE_SEEDS = {
    ("exp1", "control"): 0xC07401,
    ("exp1", "treatment"): 0x73E471,
}


def _record_binary(runner: ExperimentRunner, exp: str, variant: str, n: int, success_rate: float, cost: float = 0.01, latency: float = 1.0) -> None:
    seed = _STABLE_SEEDS.get((exp, variant), abs(hash((exp, variant))) & 0xFFFF)
    rng = random.Random(seed)
    for _ in range(n):
        ok = rng.random() < success_rate
        runner.record(exp, variant, success=ok, cost_usd=cost, latency_s=latency)


class TestRecordBinary(unittest.TestCase):
    def test_record_increments_stats(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        for _ in range(50):
            r.record("exp1", "control", success=True, cost_usd=0.01, latency_s=2.0)
        s = r.status("exp1")
        ctrl = next(v for v in s.variants if v.name == "control")
        self.assertEqual(ctrl.samples, 50)
        self.assertAlmostEqual(ctrl.primary_mean, 1.0, places=3)
        cost = ctrl.metrics[METRIC_COST_USD]
        self.assertEqual(cost["samples"], 50)
        self.assertAlmostEqual(cost["mean"], 0.01, places=4)

    def test_record_unknown_variant_silently_dropped(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        # Should not raise.
        r.record("exp1", "ghost", success=True)
        s = r.status("exp1")
        for v in s.variants:
            self.assertEqual(v.samples, 0)

    def test_decision_continue_below_min_samples(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp(min_samples_per_variant=100))
        _record_binary(r, "exp1", "control", 20, 0.6)
        _record_binary(r, "exp1", "treatment", 20, 0.9)
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_CONTINUE)
        self.assertIn("samples_min", reason)

    def test_decision_ship_when_clear_win(self) -> None:
        r = ExperimentRunner(rng_seed=1)
        r.register(_basic_exp(min_samples_per_variant=200))
        _record_binary(r, "exp1", "control", 400, 0.50)
        _record_binary(r, "exp1", "treatment", 400, 0.80)
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_SHIP)
        self.assertIn("prob_treatment_better", reason)

    def test_decision_kill_when_clear_loss(self) -> None:
        r = ExperimentRunner(rng_seed=2)
        r.register(_basic_exp(min_samples_per_variant=200))
        _record_binary(r, "exp1", "control", 400, 0.80)
        _record_binary(r, "exp1", "treatment", 400, 0.50)
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_KILL)
        self.assertIn("prob_treatment_better", reason)

    def test_decision_inconclusive_at_cap_without_effect(self) -> None:
        r = ExperimentRunner(rng_seed=3)
        r.register(_basic_exp(
            min_samples_per_variant=50,
            max_samples_per_variant=200,
            minimum_detectable_effect=0.20,
        ))
        # Same true rate; at the cap, should be inconclusive.
        _record_binary(r, "exp1", "control", 200, 0.50)
        _record_binary(r, "exp1", "treatment", 200, 0.50)
        decision, reason = r.decide("exp1")
        self.assertIn(decision, (DECISION_INCONCLUSIVE, DECISION_CONTINUE))


# ---------- record + status (continuous) ----------


class TestRecordContinuous(unittest.TestCase):
    def test_continuous_metric_ship_when_lower_wins(self) -> None:
        r = ExperimentRunner(rng_seed=4)
        # primary=cost_usd, direction=min: lower is better.
        r.register(_basic_exp(
            primary_metric=METRIC_COST_USD,
            direction="min",
            min_samples_per_variant=200,
        ))
        rng = random.Random(123)
        for _ in range(300):
            r.record("exp1", "control", cost_usd=rng.gauss(0.10, 0.01))
            r.record("exp1", "treatment", cost_usd=rng.gauss(0.05, 0.01))
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_SHIP)

    def test_continuous_metric_kill_when_lower_wins_but_treatment_higher(self) -> None:
        r = ExperimentRunner(rng_seed=5)
        r.register(_basic_exp(
            primary_metric=METRIC_LATENCY_S,
            direction="min",
            min_samples_per_variant=200,
        ))
        rng = random.Random(124)
        for _ in range(300):
            r.record("exp1", "control", latency_s=rng.gauss(2.0, 0.2))
            r.record("exp1", "treatment", latency_s=rng.gauss(4.0, 0.2))
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_KILL)


# ---------- guardrails ----------


class TestGuardrails(unittest.TestCase):
    def test_guardrail_breach_kills(self) -> None:
        r = ExperimentRunner(rng_seed=6)
        r.register(_basic_exp(
            primary_metric=METRIC_COST_USD,
            direction="min",
            min_samples_per_variant=200,
            guardrails=[
                Guardrail(
                    metric=METRIC_P_SUCCESS,
                    direction="min",
                    tolerance=0.7,
                    interpret=INTERPRET_ABS,
                ),
            ],
        ))
        rng = random.Random(125)
        # Treatment is cheaper but tanks success rate.
        for _ in range(300):
            r.record("exp1", "control", cost_usd=rng.gauss(0.10, 0.01), success=True)
            r.record("exp1", "treatment", cost_usd=rng.gauss(0.05, 0.01), success=rng.random() < 0.30)
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_KILL)
        self.assertIn("guardrail", reason)

    def test_guardrail_ratio_breach(self) -> None:
        r = ExperimentRunner(rng_seed=7)
        r.register(_basic_exp(
            primary_metric=METRIC_P_SUCCESS,
            direction="max",
            min_samples_per_variant=200,
            guardrails=[
                Guardrail(
                    metric=METRIC_LATENCY_S,
                    direction="max",
                    tolerance=1.5,           # ≤1.5x slower
                    interpret=INTERPRET_RATIO,
                ),
            ],
        ))
        rng = random.Random(126)
        for _ in range(300):
            r.record("exp1", "control", success=True, latency_s=rng.gauss(2.0, 0.1))
            # 3x slower — breaches ratio.
            r.record("exp1", "treatment", success=True, latency_s=rng.gauss(6.0, 0.1))
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_KILL)
        self.assertIn("guardrail", reason)

    def test_guardrail_abs_delta(self) -> None:
        r = ExperimentRunner(rng_seed=8)
        r.register(_basic_exp(
            primary_metric=METRIC_COST_USD,
            direction="min",
            min_samples_per_variant=200,
            guardrails=[
                Guardrail(
                    metric=METRIC_P_SUCCESS,
                    direction="min",
                    tolerance=-0.05,   # treatment may drop at most 5pp below control
                    interpret=INTERPRET_ABS_DELTA,
                ),
            ],
        ))
        rng = random.Random(127)
        for _ in range(300):
            r.record("exp1", "control", cost_usd=rng.gauss(0.10, 0.01), success=True)
            # Treatment drops success rate to 0.7 (delta = -0.3 < -0.05).
            r.record("exp1", "treatment", cost_usd=rng.gauss(0.05, 0.01), success=rng.random() < 0.7)
        decision, reason = r.decide("exp1")
        self.assertEqual(decision, DECISION_KILL)


# ---------- ship / kill / pause lifecycle ----------


class TestLifecycle(unittest.TestCase):
    def test_ship_sets_terminal_state(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        e = r.ship("exp1", variant="treatment", reason="testing")
        self.assertEqual(e.status, EXP_SHIPPED)
        self.assertEqual(e.shipped_variant, "treatment")

    def test_double_ship_raises(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        r.ship("exp1", variant="treatment")
        with self.assertRaises(RuntimeError):
            r.ship("exp1", variant="treatment")

    def test_pause_resume(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        r.pause("exp1")
        self.assertEqual(r.get("exp1").status, EXP_PAUSED)
        r.resume("exp1")
        self.assertEqual(r.get("exp1").status, EXP_RUNNING)

    def test_kill_then_no_assign(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp())
        r.kill("exp1", reason="bad")
        self.assertEqual(r.get("exp1").status, EXP_KILLED)
        self.assertIsNone(r.assign("exp1", tenant_id="acme"))


# ---------- autopilot / evaluate_all ----------


class TestAutopilot(unittest.TestCase):
    def test_evaluate_all_ships_winner(self) -> None:
        r = ExperimentRunner(rng_seed=9)
        r.register(_basic_exp(min_samples_per_variant=200))
        _record_binary(r, "exp1", "control", 400, 0.50)
        _record_binary(r, "exp1", "treatment", 400, 0.85)
        out = r.evaluate_all()
        self.assertEqual(out["exp1"][0], DECISION_SHIP)
        self.assertEqual(r.get("exp1").status, EXP_SHIPPED)

    def test_evaluate_all_kills_loser(self) -> None:
        r = ExperimentRunner(rng_seed=10)
        r.register(_basic_exp(min_samples_per_variant=200))
        _record_binary(r, "exp1", "control", 400, 0.85)
        _record_binary(r, "exp1", "treatment", 400, 0.30)
        r.evaluate_all()
        self.assertEqual(r.get("exp1").status, EXP_KILLED)

    def test_evaluate_all_continues_when_undecided(self) -> None:
        r = ExperimentRunner()
        r.register(_basic_exp(min_samples_per_variant=100))
        _record_binary(r, "exp1", "control", 20, 0.6)
        _record_binary(r, "exp1", "treatment", 20, 0.7)
        r.evaluate_all()
        self.assertEqual(r.get("exp1").status, EXP_RUNNING)

    def test_autopilot_thread_starts_and_stops(self) -> None:
        r = ExperimentRunner(autopilot=False)
        r.start_autopilot(interval_s=0.05)
        self.assertTrue(r._autopilot_thread is not None and r._autopilot_thread.is_alive())
        r.stop_autopilot()
        self.assertTrue(r._autopilot_thread is None or not r._autopilot_thread.is_alive())


# ---------- persistence ----------


class TestPersistence(unittest.TestCase):
    def test_persist_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "experiments.jsonl"
            r = ExperimentRunner(persistence_path=path)
            r.register(_basic_exp())
            r.assign("exp1", tenant_id="acme")
            r.record("exp1", "control", success=True, cost_usd=0.01)
            r.ship("exp1", variant="treatment")
            lines = path.read_text().strip().splitlines()
            types = [json.loads(l)["type"] for l in lines]
            self.assertIn(EXP_EVT_REGISTERED, types)
            self.assertIn(EXP_EVT_OBSERVED, types)


# ---------- derived metric: cost_per_success ----------


class TestDerivedMetric(unittest.TestCase):
    def test_cost_per_success_treatment_wins(self) -> None:
        r = ExperimentRunner(rng_seed=11)
        r.register(_basic_exp(
            primary_metric=METRIC_COST_PER_SUCCESS,
            direction="min",
            min_samples_per_variant=200,
        ))
        rng = random.Random(128)
        # Control: cost ~$0.10, success 0.8 → cps ≈ 0.125
        # Treatment: cost ~$0.04, success 0.7 → cps ≈ 0.057 (still much cheaper per success)
        for _ in range(400):
            r.record("exp1", "control",
                     cost_usd=rng.gauss(0.10, 0.005),
                     success=rng.random() < 0.80)
            r.record("exp1", "treatment",
                     cost_usd=rng.gauss(0.04, 0.003),
                     success=rng.random() < 0.70)
        s = r.status("exp1")
        # The status uses the underlying cost-Welch as the p-value proxy.
        # Treatment cost is clearly lower → SHIP.
        self.assertEqual(s.decision, DECISION_SHIP)


# ---------- runner-on-bus / event sink ----------


class TestEventSink(unittest.TestCase):
    def test_event_sink_receives_kinds(self) -> None:
        seen: list[tuple[str, dict[str, Any]]] = []
        r = ExperimentRunner(event_sink=lambda k, p: seen.append((k, p)))
        r.register(_basic_exp())
        r.assign("exp1", tenant_id="acme")
        r.ship("exp1", variant="treatment", reason="test")
        kinds = [k for k, _ in seen]
        self.assertIn(EXP_EVT_REGISTERED, kinds)
        self.assertIn("experiment.assigned", kinds)
        self.assertIn("experiment.shipped", kinds)


# ---------- driver integration ----------


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    response = "fake-response"

    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score = None
        self.extra_system = None
        self.messages = []

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        # Simulate cost dependent on model: "cheap" charges less.
        if self.model == "cheap":
            self.usage.input_tokens += 50
            self.usage.output_tokens += 20
        else:
            self.usage.input_tokens += 200
            self.usage.output_tokens += 100
        return FakeAgent.response

    def attach_tool_synth(self, *a, **kw): pass
    def attach_delegation(self, *a, **kw): pass
    def reset(self): self.usage = FakeUsage()


def _make_driver(**overrides) -> tuple[RuntimeDriver, Path]:
    tmp = Path(tempfile.mkdtemp())
    rt = Runtime(
        memory=Memory(path=tmp / "m.jsonl"),
        skills=SkillLibrary(path=tmp / "skills"),
        agent_factory=FakeAgent,
        **overrides,
    )
    return RuntimeDriver(runtime=rt), tmp


class TestDriverIntegration(unittest.TestCase):
    def test_driver_experiments_property_is_lazy(self) -> None:
        driver, _ = _make_driver()
        self.assertIsNone(driver._experiments_runner)
        runner = driver.experiments
        self.assertIsNotNone(runner)
        self.assertIs(driver.experiments, runner)

    def test_submit_with_experiment_applies_overrides(self) -> None:
        driver, _ = _make_driver()
        runner = driver.experiments
        runner.register(_basic_exp(traffic_split=[0.0, 1.0]))  # force treatment
        req = TicketRequest(intent="test", tenant_id="acme", budget_usd=0.5)
        ticket = driver.submit_with_experiment(req, "exp1")
        receipt = ticket.result(timeout=10.0)
        # Receipt's session ran with the treatment override (model="cheap"),
        # which has reduced token usage.
        self.assertEqual(receipt.status, COMPLETED)
        self.assertLess(receipt.actual_cost_usd, 0.01)

    def test_completed_ticket_records_observation(self) -> None:
        driver, _ = _make_driver()
        runner = driver.experiments
        runner.register(_basic_exp(traffic_split=[0.0, 1.0]))
        req = TicketRequest(intent="test", tenant_id="acme", budget_usd=0.5)
        ticket = driver.submit_with_experiment(req, "exp1")
        ticket.result(timeout=10.0)
        s = runner.status("exp1")
        tv = next(v for v in s.variants if v.name == "treatment")
        self.assertEqual(tv.samples, 1)
        self.assertAlmostEqual(tv.primary_mean, 1.0, places=3)

    def test_submit_with_experiment_unknown_passes_through(self) -> None:
        driver, _ = _make_driver()
        req = TicketRequest(intent="test", tenant_id="acme", budget_usd=0.5)
        ticket = driver.submit_with_experiment(req, "no-such-exp")
        receipt = ticket.result(timeout=10.0)
        self.assertEqual(receipt.status, COMPLETED)

    def test_submit_with_terminal_experiment_passes_through(self) -> None:
        driver, _ = _make_driver()
        runner = driver.experiments
        runner.register(_basic_exp())
        runner.ship("exp1", variant="treatment")
        req = TicketRequest(intent="test", tenant_id="acme", budget_usd=0.5)
        ticket = driver.submit_with_experiment(req, "exp1")
        receipt = ticket.result(timeout=10.0)
        self.assertEqual(receipt.status, COMPLETED)
        # No observation recorded since the experiment is terminal.
        s = runner.status("exp1")
        self.assertEqual(sum(v.samples for v in s.variants), 0)


# ---------- end-to-end traffic ramp ----------


class TestEndToEnd(unittest.TestCase):
    def test_full_loop_ship_on_lift(self) -> None:
        r = ExperimentRunner(rng_seed=20)
        r.register(_basic_exp(min_samples_per_variant=200, posterior_samples=400))
        # Simulate 1000 tickets with treatment winning on success rate.
        rng = random.Random(2026)
        for i in range(1000):
            a = r.assign("exp1", tenant_id=f"tenant{i % 50}")
            if a is None:
                break
            true_rate = 0.85 if a.variant == "treatment" else 0.55
            r.record(
                "exp1",
                a.variant,
                success=rng.random() < true_rate,
                cost_usd=0.05,
                latency_s=2.0,
            )
            if i % 100 == 99:
                d = r.evaluate_all()
                if d.get("exp1", ("",))[0] in (DECISION_SHIP, DECISION_KILL):
                    break
        self.assertEqual(r.get("exp1").status, EXP_SHIPPED)
        self.assertEqual(r.get("exp1").shipped_variant, "treatment")


if __name__ == "__main__":
    unittest.main()
