"""Tests for the Constitutionalist Constitutional-AI / RLAIF primitive."""
from __future__ import annotations

import json
import random

import pytest

from agi.events import Event, EventBus
from agi.constitutionalist import (
    ACTION_ACCEPT,
    ACTION_REFUSE,
    ACTION_REVISE,
    AGG_SOFT_MIN,
    AGG_WEIGHTED_GEOMETRIC,
    AGG_WEIGHTED_MEAN,
    AGG_WORST,
    CONSTITUTIONALIST_ACCEPTED,
    CONSTITUTIONALIST_BESTOF,
    CONSTITUTIONALIST_CERTIFIED,
    CONSTITUTIONALIST_JUDGED,
    CONSTITUTIONALIST_REFUSED,
    CONSTITUTIONALIST_REGISTERED,
    CONSTITUTIONALIST_REPORTED,
    CONSTITUTIONALIST_RESET,
    CONSTITUTIONALIST_REVISED,
    CONSTITUTIONALIST_STARTED,
    Constitution,
    Constitutionalist,
    ConstitutionalistCertificate,
    ConstitutionalistConfig,
    ConstitutionalistError,
    ConstitutionalistReport,
    Critique,
    InvalidConfig,
    InvalidConstitution,
    InvalidCritique,
    InvalidRevision,
    Principle,
    PrincipleCertificate,
    PrincipleScore,
    Revision,
    RevisionStep,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_VIOLATION,
    SEVERITY_WARN,
    STOP_MAX_ITER,
    STOP_NONINCREASING,
    STOP_THRESHOLD,
    UnknownItem,
    Verdict,
)


# ---------------------------------------------------------------------------
# Helper fakes for critic / reviser injection.
# ---------------------------------------------------------------------------


def keyword_critic(scoring):
    """Build a deterministic critic that scores by keyword presence."""

    def c(text, constitution, *, rng):
        out = []
        for p in constitution.principles:
            kw = scoring.get(p.principle_id, ("", 1.0, 0.0))
            keyword, hit, miss = kw
            score = float(hit) if keyword and keyword in text else float(miss)
            out.append(PrincipleScore(
                principle_id=p.principle_id,
                score=max(0.0, min(1.0, score)),
                rationale=f"keyword:{keyword}",
            ))
        return out

    return c


def substitution_reviser(subs):
    """Reviser that applies each (find, replace) sub in order, deterministic."""

    def r(text, critique, *, rng):
        t = text
        for find, repl in subs:
            t = t.replace(find, repl)
        return t

    return r


def basic_constitution():
    return Constitution(principles=(
        Principle("helpful", "Be helpful.", weight=1.0, threshold=0.5),
        Principle("honest", "Do not lie.", weight=2.0, threshold=0.7),
        Principle(
            "safe", "Refuse weapon help.",
            severity=SEVERITY_CRITICAL, weight=3.0, threshold=0.9,
        ),
    ))


# ---------------------------------------------------------------------------
# Principle / Constitution validation.
# ---------------------------------------------------------------------------


class TestPrinciple:
    def test_defaults(self):
        p = Principle("p1", "Be nice.")
        assert p.severity == SEVERITY_VIOLATION
        assert p.weight == 1.0
        assert p.threshold == 0.5
        assert not p.is_critical

    def test_critical_flag(self):
        p = Principle("p", "x", severity=SEVERITY_CRITICAL)
        assert p.is_critical

    @pytest.mark.parametrize("kwargs", [
        dict(principle_id="", statement="x"),
        dict(principle_id="p", statement=""),
        dict(principle_id="p", statement="x", severity="bogus"),
        dict(principle_id="p", statement="x", weight=-1.0),
        dict(principle_id="p", statement="x", threshold=-0.1),
        dict(principle_id="p", statement="x", threshold=1.1),
    ])
    def test_invalid(self, kwargs):
        with pytest.raises(InvalidConstitution):
            Principle(**kwargs)


class TestConstitution:
    def test_empty_raises(self):
        with pytest.raises(InvalidConstitution):
            Constitution(principles=())

    def test_duplicate_id_raises(self):
        with pytest.raises(InvalidConstitution):
            Constitution(principles=(
                Principle("p", "a"),
                Principle("p", "b"),
            ))

    def test_from_dicts(self):
        con = Constitution.from_dicts([
            dict(principle_id="a", statement="A", weight=1.0, threshold=0.5),
            dict(principle_id="b", statement="B", severity=SEVERITY_CRITICAL),
        ])
        assert con.principles[0].principle_id == "a"
        assert con.principles[1].is_critical

    def test_hash_stable(self):
        c1 = basic_constitution()
        c2 = basic_constitution()
        assert c1.constitution_hash == c2.constitution_hash

    def test_hash_changes_when_principle_changes(self):
        c1 = basic_constitution()
        c2 = Constitution(principles=(
            Principle("helpful", "Be helpful.", weight=2.0, threshold=0.5),
            *c1.principles[1:],
        ))
        assert c1.constitution_hash != c2.constitution_hash

    def test_get_unknown_raises(self):
        with pytest.raises(InvalidConstitution):
            basic_constitution().get("nope")


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = ConstitutionalistConfig()
        assert 0.0 < cfg.alpha < 1.0
        assert cfg.violation_threshold <= cfg.accept_threshold
        assert cfg.aggregator in (
            AGG_WORST, AGG_WEIGHTED_MEAN, AGG_WEIGHTED_GEOMETRIC, AGG_SOFT_MIN,
        )

    @pytest.mark.parametrize("kwargs", [
        dict(violation_threshold=-0.1),
        dict(violation_threshold=1.1),
        dict(accept_threshold=-0.1),
        dict(accept_threshold=0.4, violation_threshold=0.5),  # invariant violated
        dict(max_iters=-1),
        dict(aggregator="bogus"),
        dict(soft_min_temperature=0.0),
        dict(soft_min_temperature=-1.0),
        dict(alpha=0.0),
        dict(alpha=1.0),
        dict(min_items_for_certificate=0),
    ])
    def test_invalid(self, kwargs):
        with pytest.raises(InvalidConfig):
            ConstitutionalistConfig(**kwargs)


# ---------------------------------------------------------------------------
# Critique and judge.
# ---------------------------------------------------------------------------


class TestJudge:
    def test_unknown_item_raises(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("ok", 0.9, 0.4)})
        with pytest.raises(UnknownItem):
            c.judge("missing", critic=critic)

    def test_score_aggregate(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                aggregator=AGG_WEIGHTED_GEOMETRIC,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({
            "helpful": ("ok", 0.9, 0.4),
            "honest": ("true", 0.95, 0.6),
            "safe":   ("weapon", 0.2, 0.99),
        })
        cr = c.judge("a", text="ok and true", critic=critic)
        assert isinstance(cr, Critique)
        assert 0.0 <= cr.aggregate_score <= 1.0
        assert cr.worst_principle in ("helpful", "honest", "safe")
        # No violations for "ok and true".
        assert cr.violations == ()
        assert cr.critical_violations == ()

    def test_critical_violation_flagged(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({
            "helpful": ("ok", 0.9, 0.4),
            "honest": ("true", 0.95, 0.6),
            "safe":   ("weapon", 0.2, 0.99),
        })
        cr = c.judge("a", text="weapon answer", critic=critic)
        assert "safe" in cr.violations
        assert "safe" in cr.critical_violations
        # worst principle should be the lowest score.
        assert cr.worst_score == 0.2
        assert cr.worst_principle == "safe"

    def test_missing_principle_defaults_to_one(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def partial_critic(text, constitution, *, rng):
            # Only score 'helpful'; others default to 1.0.
            return [PrincipleScore("helpful", 0.3, rationale="x")]

        cr = c.judge("a", text="x", critic=partial_critic)
        scores_by_id = {s.principle_id: s.score for s in cr.scores}
        assert scores_by_id["helpful"] == pytest.approx(0.3)
        assert scores_by_id["honest"] == 1.0
        assert scores_by_id["safe"] == 1.0

    def test_critic_can_use_tuples(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def tuple_critic(text, constitution, *, rng):
            return [
                ("helpful", 0.8, "ok"),
                ("honest", 0.9),
                {"principle_id": "safe", "score": 0.95, "rationale": "ok"},
            ]

        cr = c.judge("a", text="x", critic=tuple_critic)
        assert cr.violations == ()

    def test_critic_raises_invalid_critique(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def bad(text, constitution, *, rng):
            raise RuntimeError("boom")

        with pytest.raises(InvalidCritique):
            c.judge("a", text="x", critic=bad)

    def test_critic_bad_shape_raises(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def bad(text, constitution, *, rng):
            return [None]

        with pytest.raises(InvalidCritique):
            c.judge("a", text="x", critic=bad)

    def test_critic_unknown_principle_silently_ignored(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def critic(text, constitution, *, rng):
            return [
                PrincipleScore("helpful", 0.6),
                PrincipleScore("nonexistent", 0.0),  # ignored
            ]

        cr = c.judge("a", text="x", critic=critic)
        assert all(s.principle_id != "nonexistent" for s in cr.scores)

    def test_judge_deterministic(self):
        cfg = ConstitutionalistConfig(seed=42, min_items_for_certificate=2)
        a = Constitutionalist(cfg, constitution=basic_constitution())
        b = Constitutionalist(cfg, constitution=basic_constitution())
        critic = keyword_critic({"helpful": ("ok", 0.9, 0.4)})
        c1 = a.judge("a", text="ok", critic=critic)
        c2 = b.judge("a", text="ok", critic=critic)
        assert c1.aggregate_score == c2.aggregate_score
        assert c1.text_hash == c2.text_hash
        assert c1.constitution_hash == c2.constitution_hash


# ---------------------------------------------------------------------------
# Aggregator semantics.
# ---------------------------------------------------------------------------


class TestAggregators:
    def _judge_with_agg(self, agg):
        c = Constitutionalist(
            ConstitutionalistConfig(
                aggregator=agg, min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )

        def critic(text, constitution, *, rng):
            return [
                PrincipleScore("helpful", 0.9),
                PrincipleScore("honest", 0.7),
                PrincipleScore("safe", 0.3),  # low — should pull aggregate down
            ]

        return c.judge("a", text="x", critic=critic)

    def test_worst_returns_min(self):
        cr = self._judge_with_agg(AGG_WORST)
        assert cr.aggregate_score == pytest.approx(0.3)

    def test_weighted_mean(self):
        cr = self._judge_with_agg(AGG_WEIGHTED_MEAN)
        # weights 1, 2, 3; scores 0.9, 0.7, 0.3
        expected = (1 * 0.9 + 2 * 0.7 + 3 * 0.3) / 6
        assert cr.aggregate_score == pytest.approx(expected, rel=1e-6)

    def test_weighted_geometric_lower_than_mean(self):
        gm = self._judge_with_agg(AGG_WEIGHTED_GEOMETRIC).aggregate_score
        am = self._judge_with_agg(AGG_WEIGHTED_MEAN).aggregate_score
        assert gm <= am + 1e-9

    def test_soft_min_between_worst_and_mean(self):
        sm = self._judge_with_agg(AGG_SOFT_MIN).aggregate_score
        wm = self._judge_with_agg(AGG_WEIGHTED_MEAN).aggregate_score
        worst = self._judge_with_agg(AGG_WORST).aggregate_score
        assert worst - 1e-6 <= sm <= wm + 1e-6


# ---------------------------------------------------------------------------
# Critique-revise loop.
# ---------------------------------------------------------------------------


class TestRevise:
    def test_no_iters_just_judges(self):
        c = Constitutionalist(
            ConstitutionalistConfig(max_iters=0, min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("yes", 0.9, 0.4)})
        reviser = substitution_reviser([("x", "y")])
        rev = c.revise("a", text="no", critic=critic, reviser=reviser)
        assert rev.steps == ()
        assert rev.final_text == "no"
        assert rev.stop_reason in (STOP_THRESHOLD, STOP_MAX_ITER)

    def test_converges_when_threshold_met(self):
        # Reviser produces "yes" which the critic scores high.
        c = Constitutionalist(
            ConstitutionalistConfig(
                accept_threshold=0.85,
                violation_threshold=0.5,
                max_iters=3,
                aggregator=AGG_WEIGHTED_MEAN,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({
            "helpful": ("yes", 1.0, 0.2),
            "honest":  ("yes", 1.0, 0.2),
            "safe":    ("yes", 1.0, 0.2),
        })
        reviser = substitution_reviser([("no", "yes")])
        rev = c.revise("a", text="no", critic=critic, reviser=reviser)
        assert rev.converged
        assert rev.stop_reason == STOP_THRESHOLD
        assert rev.final_text == "yes"

    def test_stops_on_noninc(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                accept_threshold=0.99,
                violation_threshold=0.5,
                max_iters=5,
                require_strict_improvement=True,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({
            "helpful": ("yes", 0.6, 0.6),  # constant — never improves
            "honest":  ("", 0.6, 0.6),
            "safe":    ("", 0.6, 0.6),
        })
        reviser = substitution_reviser([("a", "b")])
        rev = c.revise("a", text="aaa", critic=critic, reviser=reviser)
        assert not rev.converged
        assert rev.stop_reason in (STOP_NONINCREASING, STOP_MAX_ITER)

    def test_max_iter_budget_observed(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                accept_threshold=0.99,  # unreachable
                violation_threshold=0.0,
                max_iters=2,
                require_strict_improvement=False,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )
        # Critic returns scaling scores, reviser appends chars.
        counter = {"n": 0}

        def critic(text, constitution, *, rng):
            counter["n"] += 1
            base = 0.3 + 0.05 * counter["n"]
            return [
                PrincipleScore("helpful", min(0.9, base)),
                PrincipleScore("honest", min(0.9, base)),
                PrincipleScore("safe", 1.0),
            ]

        reviser = substitution_reviser([("x", "xx")])
        rev = c.revise("a", text="x", critic=critic, reviser=reviser)
        # Initial judge + at most 2 iteration judges = 3 critic calls.
        assert counter["n"] <= 3
        assert len(rev.steps) <= 2

    def test_revision_records_per_step_seed(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                accept_threshold=0.99, violation_threshold=0.0,
                max_iters=2, require_strict_improvement=False,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )

        def crit(text, constitution, *, rng):
            return [
                PrincipleScore("helpful", 0.5 + 0.1 * len(text)),
                PrincipleScore("honest", 0.6),
                PrincipleScore("safe", 1.0),
            ]

        rev = c.revise(
            "a", text="a", critic=crit,
            reviser=substitution_reviser([("a", "aa")]),
        )
        seeds = [s.seed for s in rev.steps]
        assert len(seeds) == len(set(seeds))  # all distinct

    def test_reviser_raises(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def bad(text, critique, *, rng):
            raise RuntimeError("nope")

        with pytest.raises(InvalidRevision):
            c.revise("a", text="hi", critic=keyword_critic({}), reviser=bad)

    def test_reviser_returns_non_string(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )

        def bad(text, critique, *, rng):
            return 123

        with pytest.raises(InvalidRevision):
            c.revise("a", text="hi", critic=keyword_critic({}), reviser=bad)


# ---------------------------------------------------------------------------
# Best-of-N revision.
# ---------------------------------------------------------------------------


class TestBestOf:
    def test_negative_n_raises(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        with pytest.raises(ConstitutionalistError):
            c.bestof(
                "a", text="x", n=0,
                critic=keyword_critic({}),
                reviser=substitution_reviser([]),
            )

    def test_picks_best_branch(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                accept_threshold=0.99,
                violation_threshold=0.0,
                max_iters=2,
                aggregator=AGG_WEIGHTED_MEAN,
                require_strict_improvement=False,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
        )

        # Critic that scores "good" substring as compliant, otherwise low.
        def crit(text, constitution, *, rng):
            score = 0.95 if "good" in text else 0.4
            return [
                PrincipleScore("helpful", score),
                PrincipleScore("honest", score),
                PrincipleScore("safe", 1.0),
            ]

        # Reviser sometimes inserts "good".  We make it consistent so it always inserts.
        rev = c.bestof(
            "a", text="hello", n=3,
            critic=crit,
            reviser=substitution_reviser([("hello", "good hello")]),
        )
        assert "good" in rev.final_text
        assert rev.final_critique.aggregate_score >= 0.9


# ---------------------------------------------------------------------------
# Gate.
# ---------------------------------------------------------------------------


class TestGate:
    def _setup(self, **cfg_overrides):
        defaults = dict(
            violation_threshold=0.5,
            accept_threshold=0.8,
            max_iters=3,
            aggregator=AGG_WEIGHTED_GEOMETRIC,
            min_items_for_certificate=2,
        )
        defaults.update(cfg_overrides)
        cfg = ConstitutionalistConfig(**defaults)
        return Constitutionalist(cfg, constitution=basic_constitution())

    def test_accept_path(self):
        c = self._setup()
        critic = keyword_critic({
            "helpful": ("ok", 0.95, 0.3),
            "honest":  ("ok", 0.95, 0.3),
            "safe":    ("weapon", 0.2, 0.99),
        })
        reviser = substitution_reviser([("x", "y")])
        v = c.gate("a", text="ok", critic=critic, reviser=reviser)
        assert v.action == ACTION_ACCEPT
        assert v.text == "ok"
        assert v.revision is None

    def test_critical_refuse(self):
        c = self._setup()
        critic = keyword_critic({
            "helpful": ("ok", 0.95, 0.3),
            "honest":  ("ok", 0.95, 0.3),
            "safe":    ("weapon", 0.1, 0.99),
        })
        reviser = substitution_reviser([("a", "b")])
        v = c.gate("a", text="weapon ok", critic=critic, reviser=reviser)
        assert v.action == ACTION_REFUSE
        assert "critical" in v.rationale
        assert v.text == ""

    def test_revise_path_succeeds(self):
        c = self._setup()
        # 'safe' stays compliant (>= 0.9 threshold) on both texts so the
        # gate doesn't trip the critical-refuse short-circuit.  Helpful
        # and honest score well below accept_threshold on "bad" and high
        # on "good", so the revise loop has work to do and then converges.
        critic = keyword_critic({
            "helpful": ("good", 0.95, 0.55),
            "honest":  ("good", 0.95, 0.55),
            "safe":    ("",     0.95, 0.95),
        })
        reviser = substitution_reviser([("bad", "good")])
        v = c.gate("a", text="bad", critic=critic, reviser=reviser)
        assert v.action == ACTION_REVISE
        assert "good" in v.text
        assert v.revision is not None
        assert v.revision.converged

    def test_refuse_when_iter_budget_exhausted(self):
        c = self._setup(max_iters=1, refuse_after_iters=True)
        critic = keyword_critic({
            "helpful": ("yes", 0.5, 0.2),
            "honest":  ("yes", 0.5, 0.2),
            "safe":    ("",    0.99, 0.99),
        })
        # Reviser doesn't insert "yes", so aggregate stays below violation_threshold.
        reviser = substitution_reviser([("x", "x")])
        v = c.gate("a", text="no", critic=critic, reviser=reviser)
        assert v.action == ACTION_REFUSE
        assert "iter_budget_exhausted" in v.rationale

    def test_revise_without_reviser_refuses_below_violation(self):
        c = self._setup()
        critic = keyword_critic({
            # All three principles score below their thresholds so the
            # aggregate sits well under violation_threshold.  No reviser
            # supplied → gate must refuse.
            "helpful": ("", 0.95, 0.1),
            "honest":  ("", 0.99, 0.1),
            "safe":    ("", 0.95, 0.95),  # compliant — avoids critical-refuse
        })
        v = c.gate("a", text="x", critic=critic, reviser=None)
        assert v.action == ACTION_REFUSE
        assert "below_violation_threshold" in v.rationale

    def test_revise_without_reviser_accepts_with_warning(self):
        c = self._setup(violation_threshold=0.3, accept_threshold=0.95)
        critic = keyword_critic({
            "helpful": ("", 0.95, 0.6),
            "honest":  ("", 0.99, 0.6),
            "safe":    ("", 0.99, 0.99),
        })
        v = c.gate("a", text="x", critic=critic, reviser=None)
        assert v.action == ACTION_ACCEPT
        assert "warn" in v.rationale


# ---------------------------------------------------------------------------
# Certificate.
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_too_few_items_raises(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=5),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("ok", 0.9, 0.4)})
        c.judge("a", text="x", critic=critic)
        with pytest.raises(ConstitutionalistError):
            c.certificate()

    def test_basic_cert_fields(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                min_items_for_certificate=2,
                aggregator=AGG_WEIGHTED_MEAN,
            ),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({
            "helpful": ("ok", 0.95, 0.4),
            "honest":  ("",  0.95, 0.95),
            "safe":    ("",  0.99, 0.99),
        })
        for i in range(6):
            c.judge(f"item{i}", text=("ok" if i % 2 == 0 else "bad"),
                    critic=critic)
        cert = c.certificate()
        assert isinstance(cert, ConstitutionalistCertificate)
        assert cert.n_items == 6
        assert 0.0 <= cert.aggregate_mean <= 1.0
        # eb_lo is no greater than the mean.
        assert cert.aggregate_eb_lo <= cert.aggregate_mean + 1e-9
        # Per-principle records map to constitution order.
        assert tuple(p.principle_id for p in cert.principles) == \
            tuple(p.principle_id for p in c.constitution.principles)
        # Wilson interval bounds the rate.
        for pc in cert.principles:
            assert pc.wilson_lo <= pc.violation_rate + 1e-9
            assert pc.wilson_hi >= pc.violation_rate - 1e-9
        # Fingerprint non-empty.
        assert len(cert.fingerprint) == 64

    def test_joint_correction_tightens(self):
        # Holm correction → adjusted alpha smaller → CI wider; assert that
        # for at least one principle the corrected alpha is < base alpha.
        c1 = Constitutionalist(
            ConstitutionalistConfig(
                min_items_for_certificate=2,
                joint_correction=True,
                aggregator=AGG_WEIGHTED_MEAN,
            ),
            constitution=basic_constitution(),
        )
        c2 = Constitutionalist(
            ConstitutionalistConfig(
                min_items_for_certificate=2,
                joint_correction=False,
                aggregator=AGG_WEIGHTED_MEAN,
            ),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({
            "helpful": ("ok", 0.9, 0.4),
            "honest":  ("ok", 0.9, 0.4),
            "safe":    ("",   0.99, 0.99),
        })
        for i in range(6):
            c1.judge(f"i{i}", text=("ok" if i % 2 == 0 else "bad"), critic=critic)
            c2.judge(f"i{i}", text=("ok" if i % 2 == 0 else "bad"), critic=critic)
        a1 = c1.certificate()
        a2 = c2.certificate()
        assert any(p.adjusted_alpha < a2.alpha
                   for p in a1.principles)

    def test_certificate_json_roundtrips(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("ok", 0.9, 0.4)})
        for i in range(5):
            c.judge(f"i{i}", text="ok", critic=critic)
        cert = c.certificate()
        s = json.dumps(cert.to_dict())
        d = json.loads(s)
        assert d["n_items"] == cert.n_items
        assert d["fingerprint"] == cert.fingerprint

    def test_report(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("ok", 0.9, 0.4)})
        for i in range(3):
            c.judge(f"i{i}", text="ok", critic=critic)
        rep = c.report()
        assert isinstance(rep, ConstitutionalistReport)
        assert rep.items == 3
        d = json.loads(json.dumps(rep.to_dict()))
        assert d["items"] == 3


# ---------------------------------------------------------------------------
# Preference mining.
# ---------------------------------------------------------------------------


class TestPreferenceMining:
    def test_only_improved_pairs(self):
        c = Constitutionalist(
            ConstitutionalistConfig(
                violation_threshold=0.3,
                accept_threshold=0.99,
                max_iters=2,
                aggregator=AGG_WEIGHTED_MEAN,
                min_items_for_certificate=2,
                require_strict_improvement=False,
            ),
            constitution=basic_constitution(),
        )

        # Critic: "good" -> 0.9, else 0.4.
        def crit(text, constitution, *, rng):
            score = 0.95 if "good" in text else 0.4
            return [
                PrincipleScore("helpful", score),
                PrincipleScore("honest", score),
                PrincipleScore("safe", 1.0),
            ]

        reviser_a = substitution_reviser([("bad", "good")])
        reviser_b = substitution_reviser([("bad", "bad")])  # no change
        c.revise("improved", text="bad", critic=crit, reviser=reviser_a)
        c.revise("no_change", text="bad", critic=crit, reviser=reviser_b)
        pairs = c.mine_preferences()
        ids = [p["item_id"] for p in pairs]
        assert "improved" in ids
        assert "no_change" not in ids
        p = next(p for p in pairs if p["item_id"] == "improved")
        assert p["rejected"] == "bad"
        assert "good" in p["chosen"]
        assert p["chosen_score"] > p["rejected_score"]


# ---------------------------------------------------------------------------
# Event bus integration.
# ---------------------------------------------------------------------------


class TestEvents:
    def test_started_published(self):
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(events.append)
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
            bus=bus,
        )
        kinds = [e.kind for e in events]
        assert CONSTITUTIONALIST_STARTED in kinds

    def test_judge_publishes(self):
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(events.append)
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
            bus=bus,
        )
        c.judge("a", text="ok", critic=keyword_critic({"helpful": ("ok", 0.9, 0.4)}))
        kinds = {e.kind for e in events}
        assert CONSTITUTIONALIST_JUDGED in kinds

    def test_gate_revise_path_publishes(self):
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(events.append)
        c = Constitutionalist(
            ConstitutionalistConfig(
                violation_threshold=0.5,
                accept_threshold=0.9,
                max_iters=3,
                aggregator=AGG_WEIGHTED_MEAN,
                min_items_for_certificate=2,
            ),
            constitution=basic_constitution(),
            bus=bus,
        )
        critic = keyword_critic({
            "helpful": ("good", 0.95, 0.4),
            "honest":  ("", 0.99, 0.99),
            "safe":    ("", 0.99, 0.99),
        })
        c.gate("a", text="bad",
               critic=critic,
               reviser=substitution_reviser([("bad", "good")]))
        kinds = {e.kind for e in events}
        # Has both REVISED and ACCEPTED for the revise-and-accept path.
        assert CONSTITUTIONALIST_REVISED in kinds
        assert CONSTITUTIONALIST_ACCEPTED in kinds

    def test_critical_refuse_publishes(self):
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(events.append)
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
            bus=bus,
        )
        critic = keyword_critic({
            "helpful": ("", 0.9, 0.9),
            "honest":  ("", 0.9, 0.9),
            "safe":    ("weapon", 0.1, 0.99),
        })
        c.gate("a", text="weapon", critic=critic,
               reviser=substitution_reviser([("x", "y")]))
        kinds = {e.kind for e in events}
        assert CONSTITUTIONALIST_REFUSED in kinds


# ---------------------------------------------------------------------------
# Fingerprint chain.
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_evolves(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        fp0 = c.fingerprint_hash
        c.judge("a", text="x",
                critic=keyword_critic({"helpful": ("x", 0.9, 0.1)}))
        fp1 = c.fingerprint_hash
        assert fp0 != fp1

    def test_fingerprint_deterministic_across_instances(self):
        cfg = ConstitutionalistConfig(seed=11, min_items_for_certificate=2)
        critic = keyword_critic({"helpful": ("x", 0.9, 0.1)})

        def trace():
            c = Constitutionalist(
                cfg, constitution=basic_constitution(),
                clock=lambda: 1.0,  # freeze time
            )
            c.judge("a", text="x", critic=critic)
            c.judge("b", text="y", critic=critic)
            return c.fingerprint_hash

        assert trace() == trace()


# ---------------------------------------------------------------------------
# Reset + threading safety smoke.
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_items_keeps_constitution(self):
        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("x", 0.9, 0.1)})
        c.judge("a", text="x", critic=critic)
        c.judge("b", text="y", critic=critic)
        before = c.constitution.constitution_hash
        c.reset()
        assert c.constitution.constitution_hash == before
        assert c.critiques == ()
        assert c.verdicts == ()
        assert c.revisions == ()


class TestThreadSafety:
    def test_concurrent_judges(self):
        import threading

        c = Constitutionalist(
            ConstitutionalistConfig(min_items_for_certificate=2),
            constitution=basic_constitution(),
        )
        critic = keyword_critic({"helpful": ("x", 0.9, 0.1)})

        def worker(i):
            for j in range(10):
                c.judge(f"t{i}-{j}", text=f"x{i}{j}", critic=critic)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(c.critiques) == 40
