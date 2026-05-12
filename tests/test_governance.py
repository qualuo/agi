"""Tests for the multi-tenant governance layer."""
import os
import tempfile
import time
import unittest

from agi.governance import (
    AdmissionDecision,
    GovernedRuntime,
    PolicyManager,
    TenantLimits,
)


class TestPolicyManager(unittest.TestCase):
    def setUp(self):
        self.pm = PolicyManager()

    def test_unconstrained_tenant_admits(self):
        d = self.pm.check_admission("acme", kind="chat", estimated_cost_usd=0.1)
        self.assertTrue(d)
        self.assertEqual(d.http_status, 200)

    def test_concurrent_session_cap(self):
        self.pm.set_limits(TenantLimits("acme", max_concurrent_sessions=1))
        self.assertTrue(self.pm.check_admission("acme", kind="session_create"))
        self.pm.session_started("acme", "s1")
        d = self.pm.check_admission("acme", kind="session_create")
        self.assertFalse(d)
        self.assertEqual(d.code, "max_concurrent_sessions")
        self.assertEqual(d.http_status, 429)
        # Free a slot
        self.pm.session_ended("acme", "s1")
        self.assertTrue(self.pm.check_admission("acme", kind="session_create"))

    def test_rate_limit_per_minute(self):
        self.pm.set_limits(TenantLimits("acme", max_prompts_per_minute=2))
        self.assertTrue(self.pm.check_admission("acme", kind="chat"))
        self.pm.commit("acme", cost_usd=0.0, kind="chat")
        self.assertTrue(self.pm.check_admission("acme", kind="chat"))
        self.pm.commit("acme", cost_usd=0.0, kind="chat")
        d = self.pm.check_admission("acme", kind="chat")
        self.assertFalse(d)
        self.assertEqual(d.code, "rate_limit_minute")

    def test_daily_budget_blocks_when_exceeded(self):
        self.pm.set_limits(TenantLimits("acme", daily_cost_usd=1.0))
        self.pm.commit("acme", cost_usd=0.99, kind="chat")
        d = self.pm.check_admission("acme", kind="chat", estimated_cost_usd=0.02)
        self.assertFalse(d)
        self.assertEqual(d.code, "daily_budget")
        self.assertEqual(d.http_status, 402)

    def test_lifetime_cap_blocks_forever(self):
        self.pm.set_limits(TenantLimits("acme", lifetime_cost_usd=0.50))
        self.pm.commit("acme", cost_usd=0.50, kind="chat")
        d = self.pm.check_admission("acme", kind="chat", estimated_cost_usd=0.01)
        self.assertFalse(d)
        self.assertEqual(d.code, "lifetime_budget")

    def test_default_limits_apply_to_unknown_tenant(self):
        defaults = TenantLimits("__default__", max_prompts_per_minute=1)
        pm = PolicyManager(default_limits=defaults)
        # First call OK
        self.assertTrue(pm.check_admission("brand-new", kind="chat"))
        pm.commit("brand-new", cost_usd=0.0)
        # Second call blocked by default policy
        self.assertFalse(pm.check_admission("brand-new", kind="chat"))

    def test_usage_reports_match_commits(self):
        self.pm.set_limits(TenantLimits("acme", daily_cost_usd=10.0))
        self.pm.commit("acme", cost_usd=0.05, kind="chat")
        self.pm.commit("acme", cost_usd=0.10, kind="chat")
        u = self.pm.usage("acme")
        self.assertAlmostEqual(u["lifetime_cost_usd"], 0.15, places=6)
        self.assertEqual(u["prompts_total"], 2)
        self.assertEqual(u["prompts_last_minute"], 2)

    def test_fair_pick_prefers_low_cost_tenant(self):
        self.pm.set_limits(TenantLimits("a", fair_share_weight=1.0))
        self.pm.set_limits(TenantLimits("b", fair_share_weight=1.0))
        self.pm.commit("a", cost_usd=1.0)
        # b has spent nothing so it should win
        winner = self.pm.fair_pick(["a", "b"])
        self.assertEqual(winner, "b")

    def test_fair_pick_respects_weights(self):
        self.pm.set_limits(TenantLimits("premium", fair_share_weight=10.0))
        self.pm.set_limits(TenantLimits("free", fair_share_weight=1.0))
        # premium spent 5x more than free — but its weight is 10x, so it
        # should still win.
        self.pm.commit("premium", cost_usd=5.0)
        self.pm.commit("free", cost_usd=1.0)
        self.assertEqual(self.pm.fair_pick(["premium", "free"]), "premium")

    def test_audit_log_records_decisions(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            pm = PolicyManager(audit_path=path)
            pm.set_limits(TenantLimits("acme", lifetime_cost_usd=0.5))
            pm.commit("acme", cost_usd=0.6)
            pm.check_admission("acme", kind="chat", estimated_cost_usd=0.01)
            with open(path) as fh:
                lines = [ln for ln in fh if ln.strip()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertTrue(any('"ok": false' in ln for ln in lines))
        finally:
            os.unlink(path)


class _FakeAgent:
    """Minimal stand-in for Agent so we can drive Session.chat without an API."""
    def __init__(self, **kwargs):
        from agi.costs import Usage
        self.usage = Usage()
        self.usage.input_tokens = 100
        self.usage.output_tokens = 50
        self.messages: list = []
        self.last_critic_score = None

    def chat(self, prompt, max_iterations=25):
        return f"echo: {prompt[:60]}"

    def reset(self):
        self.messages = []


class TestGovernedRuntime(unittest.TestCase):
    def setUp(self):
        from agi.runtime import Runtime, SessionConfig
        self.runtime = Runtime(agent_factory=_FakeAgent)
        self.pm = PolicyManager()
        self.gr = GovernedRuntime(self.runtime, self.pm)
        self.cfg = SessionConfig()

    def test_session_create_increments_active(self):
        self.gr.create_session("acme", self.cfg)
        self.assertEqual(self.pm.usage("acme")["sessions_active"], 1)

    def test_session_create_denied_at_cap(self):
        self.pm.set_limits(TenantLimits("acme", max_concurrent_sessions=1))
        self.gr.create_session("acme", self.cfg)
        with self.assertRaises(PermissionError):
            self.gr.create_session("acme", self.cfg)

    def test_chat_denied_over_daily_budget(self):
        self.pm.set_limits(TenantLimits("acme", daily_cost_usd=0.0001))
        sid = self.gr.create_session("acme", self.cfg)
        # estimated cost > daily cap
        with self.assertRaises(PermissionError):
            self.gr.chat("acme", sid, "hi", estimated_cost_usd=0.01)

    def test_chat_commits_actual_cost(self):
        sid = self.gr.create_session("acme", self.cfg)
        self.gr.chat("acme", sid, "hello")
        # Some non-zero cost should be recorded
        self.assertGreater(self.pm.usage("acme")["lifetime_cost_usd"], 0.0)

    def test_end_session_decrements_active(self):
        sid = self.gr.create_session("acme", self.cfg)
        self.gr.end_session("acme", sid)
        self.assertEqual(self.pm.usage("acme")["sessions_active"], 0)


if __name__ == "__main__":
    unittest.main()
