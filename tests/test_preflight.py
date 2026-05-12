"""Tests for the preflight estimator + admission advisor."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.governance import PolicyManager, TenantLimits
from agi.memory import Memory
from agi.preflight import (
    ADMIT,
    DEFER,
    DOWNGRADE,
    REJECT,
    AdmissionAdvisor,
    PreflightEstimator,
    _DefaultCfg,
)
from agi.runtime import Runtime, SessionConfig
from agi.skills import SkillLibrary


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.turns = 0

    def cost_usd(self, model: str) -> float:
        return self.input_tokens * 0.000005 + self.output_tokens * 0.000025


class FakeAgent:
    def __init__(self, *, memory=None, model="claude-opus-4-7", **kw) -> None:
        self.memory = memory
        self.model = model
        self.usage = FakeUsage()
        self.last_critic_score: float | None = None
        self.extra_system: str | None = None

    def chat(self, prompt: str, max_iterations: int = 25) -> str:
        self.usage.input_tokens += 200
        self.usage.output_tokens += 100
        self.usage.turns += 1
        return "ok"


def _runtime(**kw) -> Runtime:
    tmp = tempfile.mkdtemp()
    return Runtime(
        memory=Memory(path=Path(tmp) / "m.jsonl"),
        skills=SkillLibrary(path=Path(tmp) / "skills"),
        agent_factory=FakeAgent,
        **kw,
    )


class TestEstimatorPrior(unittest.TestCase):
    def test_estimate_returns_nonzero_cost_for_known_model(self):
        est = PreflightEstimator()
        e = est.estimate("hello world", _DefaultCfg(model="claude-opus-4-7"))
        self.assertGreater(e.cost_usd, 0.0)
        self.assertEqual(e.confidence, "low")
        self.assertEqual(e.samples, 0)
        # p10 < cost < p90
        self.assertLessEqual(e.cost_p10_usd, e.cost_usd)
        self.assertGreaterEqual(e.cost_p90_usd, e.cost_usd)

    def test_estimate_scales_with_prompt_length(self):
        est = PreflightEstimator()
        short = est.estimate("x" * 100)
        long_ = est.estimate("x" * 100000)
        self.assertLess(short.cost_usd, long_.cost_usd)

    def test_estimate_cheaper_for_cheaper_model(self):
        est = PreflightEstimator()
        opus = est.estimate("hello", _DefaultCfg(model="claude-opus-4-7"))
        haiku = est.estimate("hello", _DefaultCfg(model="claude-haiku-4-5"))
        self.assertLess(haiku.cost_usd, opus.cost_usd)

    def test_unknown_model_emits_note(self):
        est = PreflightEstimator()
        e = est.estimate("hi", _DefaultCfg(model="nonexistent-model-x"))
        self.assertTrue(any("unknown model" in n for n in e.notes))
        self.assertEqual(e.cost_usd, 0.0)

    def test_tool_enablement_increases_cost(self):
        est = PreflightEstimator()
        base = est.estimate("hi", _DefaultCfg(enable_delegation=False))
        delegated = est.estimate("hi", _DefaultCfg(enable_delegation=True))
        self.assertGreater(delegated.cost_usd, base.cost_usd)


class TestEstimatorLearning(unittest.TestCase):
    def test_record_lifts_confidence_with_samples(self):
        est = PreflightEstimator()
        cfg = _DefaultCfg(model="claude-opus-4-7")
        for _ in range(25):
            est.record(prompt="quick task", config=cfg, cost_usd=0.01, duration_s=2.0, success=True)
        e = est.estimate("quick task", cfg)
        self.assertEqual(e.confidence, "high")
        self.assertEqual(e.samples, 25)
        # Empirical mean dominates with n=25 (w = 25/(25+8) ≈ 0.76)
        self.assertLess(abs(e.cost_usd - 0.01), 0.05)

    def test_quantiles_track_observed_distribution(self):
        est = PreflightEstimator()
        cfg = _DefaultCfg()
        costs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        for c in costs:
            est.record(prompt="task", config=cfg, cost_usd=c, duration_s=1.0, success=True)
        e = est.estimate("task", cfg)
        self.assertLess(e.cost_p10_usd, e.cost_p90_usd)
        # p10 should be in the lower tail of observed costs
        self.assertLess(e.cost_p10_usd, 0.01)
        # p90 should be in the upper tail
        self.assertGreater(e.cost_p90_usd, 0.05)

    def test_success_rate_learned(self):
        est = PreflightEstimator()
        cfg = _DefaultCfg(model="claude-haiku-4-5")
        for _ in range(20):
            est.record(prompt="hard", config=cfg, cost_usd=0.01, duration_s=1.0, success=False)
        for _ in range(5):
            est.record(prompt="hard", config=cfg, cost_usd=0.01, duration_s=1.0, success=True)
        e = est.estimate("hard", cfg)
        # 5/25 success with n=25 → blended toward ~0.2..0.35
        self.assertLess(e.p_success, 0.5)

    def test_persistence_round_trip(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "history.jsonl"
        est1 = PreflightEstimator(history_path=path)
        cfg = _DefaultCfg()
        for _ in range(10):
            est1.record(prompt="task", config=cfg, cost_usd=0.02, duration_s=3.0, success=True)
        # New instance reading the same path
        est2 = PreflightEstimator(history_path=path)
        e = est2.estimate("task", cfg)
        self.assertGreaterEqual(e.samples, 10)


class TestRuntimeIntegration(unittest.TestCase):
    def test_runtime_estimate_uses_session_config(self):
        rt = _runtime()
        sid = rt.create_session(SessionConfig(model="claude-haiku-4-5", use_skills=False))
        e = rt.estimate("hello", session_id=sid)
        self.assertEqual(e.model, "claude-haiku-4-5")

    def test_runtime_attaches_estimator_to_event_stream(self):
        rt = _runtime()
        sid = rt.create_session(SessionConfig(use_skills=False))
        e_before = rt.estimate("ping", session_id=sid)
        self.assertEqual(e_before.samples, 0)
        rt.chat(sid, "ping")
        e_after = rt.estimate("ping", session_id=sid)
        self.assertGreaterEqual(e_after.samples, 1)

    def test_capabilities_exposes_preflight_stats(self):
        rt = _runtime()
        caps = rt.capabilities()
        self.assertIn("preflight", caps)
        self.assertIn("bin_count", caps["preflight"])

    def test_advise_admit_by_default(self):
        rt = _runtime()
        advice = rt.advise("simple task")
        self.assertEqual(advice.verdict, ADMIT)
        self.assertTrue(advice.admit())


class TestAdmissionAdvisor(unittest.TestCase):
    def test_advise_admit_on_clean_path(self):
        est = PreflightEstimator()
        advisor = AdmissionAdvisor(est)
        advice = advisor.advise(prompt="hi")
        self.assertEqual(advice.verdict, ADMIT)

    def test_per_turn_cost_cap_triggers_downgrade(self):
        est = PreflightEstimator()
        # Cap chosen so opus p90 breaches but haiku p90 fits.
        advisor = AdmissionAdvisor(est, max_cost_per_turn_usd=0.08)
        advice = advisor.advise(
            prompt="x" * 5000,
            config=_DefaultCfg(model="claude-opus-4-7"),
        )
        self.assertEqual(advice.verdict, DOWNGRADE)
        self.assertIsNotNone(advice.alternative)
        self.assertNotEqual(advice.alternative["model"], "claude-opus-4-7")
        self.assertGreater(advice.alternative["expected_savings_usd"], 0.0)

    def test_per_turn_cap_with_no_viable_alternative_rejects(self):
        est = PreflightEstimator()
        advisor = AdmissionAdvisor(est, max_cost_per_turn_usd=1e-9)
        # Even cheapest model can't fit a microscopic cap.
        advice = advisor.advise(prompt="x" * 1000, config=_DefaultCfg(model="claude-haiku-4-5"))
        self.assertEqual(advice.verdict, REJECT)

    def test_quality_floor_rejects_when_history_low_success(self):
        est = PreflightEstimator()
        cfg = _DefaultCfg(model="claude-haiku-4-5")
        # 20 confirmed failures push the empirical p_success very low,
        # and n=20 yields medium/high confidence so the floor applies.
        for _ in range(20):
            est.record(prompt="impossible", config=cfg, cost_usd=0.01, duration_s=1.0, success=False)
        advisor = AdmissionAdvisor(est, min_p_success=0.6)
        advice = advisor.advise(prompt="impossible", config=cfg)
        self.assertEqual(advice.verdict, REJECT)
        self.assertIn("p_success", advice.reason)

    def test_quality_floor_skipped_when_confidence_low(self):
        est = PreflightEstimator()
        advisor = AdmissionAdvisor(est, min_p_success=0.99)  # impossibly high
        # No history → low confidence → floor does not apply.
        advice = advisor.advise(prompt="x", config=_DefaultCfg(model="claude-haiku-4-5"))
        self.assertEqual(advice.verdict, ADMIT)

    def test_governance_budget_breach_defers(self):
        policy = PolicyManager()
        policy.set_limits(TenantLimits(tenant_id="t1", daily_cost_usd=0.000001))
        est = PreflightEstimator()
        advisor = AdmissionAdvisor(est, policy=policy)
        advice = advisor.advise(prompt="task", tenant_id="t1")
        self.assertEqual(advice.verdict, DEFER)
        self.assertEqual(advice.governance_code, "daily_budget")
        self.assertIsNotNone(advice.retry_after_s)

    def test_governance_rate_limit_defers(self):
        policy = PolicyManager()
        policy.set_limits(TenantLimits(tenant_id="t2", max_prompts_per_minute=1))
        # Burn the budget for the minute window
        policy.commit(tenant_id="t2", cost_usd=0.0)
        est = PreflightEstimator()
        advisor = AdmissionAdvisor(est, policy=policy)
        advice = advisor.advise(prompt="task", tenant_id="t2")
        self.assertEqual(advice.verdict, DEFER)
        self.assertEqual(advice.governance_code, "rate_limit_minute")

    def test_capacity_at_session_cap_defers(self):
        rt = _runtime(max_concurrent_sessions=1)
        rt.create_session()
        # Now at cap. Advise on a hypothetical prompt should defer.
        advice = rt.advise("would create another")
        self.assertEqual(advice.verdict, DEFER)


class TestAdmissionDataclass(unittest.TestCase):
    def test_to_dict_is_jsonable(self):
        import json
        est = PreflightEstimator()
        advisor = AdmissionAdvisor(est)
        advice = advisor.advise(prompt="hi")
        s = json.dumps(advice.to_dict())
        self.assertIn("verdict", s)
        self.assertIn("estimate", s)


if __name__ == "__main__":
    unittest.main()
