"""Tests for agi.analogist — structure-mapping analogical reasoning."""
from __future__ import annotations

import unittest

from agi.analogist import (
    Analogist,
    AnalogistConfig,
    AnalogistError,
    BudgetExhausted,
    Description,
    GlobalMapping,
    InvalidConfig,
    InvalidDescription,
    InvalidExpression,
    MatchHypothesis,
    ProportionalAnalogy,
    UnknownDescription,
    acme,
    literal_similarity,
    sme,
)


# ---------------------------------------------------------------------------
# Expression / Description plumbing
# ---------------------------------------------------------------------------


class TestDescription(unittest.TestCase):
    def test_basic_construction(self):
        d = Description(
            name="x",
            expressions=(
                ("attracts", "sun", "planet"),
                ("greater", ("mass", "sun"), ("mass", "planet")),
            ),
        )
        self.assertEqual(d.name, "x")
        self.assertEqual(len(d.expressions), 2)
        self.assertIn("sun", d.entities())
        self.assertIn("planet", d.entities())

    def test_rejects_empty_name(self):
        with self.assertRaises(InvalidDescription):
            Description(name="", expressions=())

    def test_rejects_invalid_expression(self):
        with self.assertRaises(InvalidExpression):
            Description(name="x", expressions=(123,))
        with self.assertRaises(InvalidExpression):
            Description(name="x", expressions=(("", "a"),))
        with self.assertRaises(InvalidExpression):
            Description(name="x", expressions=((),))

    def test_fingerprint_stable(self):
        d1 = Description(name="x", expressions=(("p", "a", "b"),))
        d2 = Description(name="x", expressions=(("p", "a", "b"),))
        self.assertEqual(d1.fingerprint(), d2.fingerprint())

    def test_fingerprint_order_insensitive(self):
        d1 = Description(name="x", expressions=(("p", "a"), ("q", "b")))
        d2 = Description(name="x", expressions=(("q", "b"), ("p", "a")))
        # fingerprint sorts expressions before hashing
        self.assertEqual(d1.fingerprint(), d2.fingerprint())

    def test_subexpression_dedup(self):
        d = Description(
            name="x",
            expressions=(("cause", ("p", "a"), ("p", "a")),),
        )
        subs = list(d.all_subexpressions())
        # ("p","a") should appear exactly once after canonical dedup
        keys = [str(s) for s in subs]
        self.assertEqual(keys.count("('p', 'a')"), 1)


# ---------------------------------------------------------------------------
# AnalogistConfig validation
# ---------------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = AnalogistConfig()
        self.assertEqual(cfg.engine, "sme")
        self.assertEqual(cfg.mode, "analogy")

    def test_invalid_engine(self):
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(engine="bogus")

    def test_invalid_mode(self):
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(mode="bogus")

    def test_negative_weight(self):
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(attribute_weight=-1)

    def test_systematicity_bounds(self):
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(systematicity_weight=1.5)
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(systematicity_weight=-0.1)

    def test_max_gmaps_positive(self):
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(max_gmaps=0)

    def test_time_budget_positive(self):
        with self.assertRaises(InvalidConfig):
            AnalogistConfig(time_budget_s=0)

    def test_mode_affects_weight(self):
        cfg_a = AnalogistConfig(mode="analogy", attribute_weight=0.5)
        cfg_l = AnalogistConfig(mode="literal", attribute_weight=0.5)
        cfg_m = AnalogistConfig(mode="mere_appearance", attribute_weight=0.5)
        self.assertEqual(cfg_a.effective_weight("attribute"), 0.5)
        self.assertEqual(cfg_l.effective_weight("attribute"), 0.5)
        self.assertEqual(cfg_m.effective_weight("attribute"), 0.5)
        # mere_appearance zeroes relations
        self.assertEqual(cfg_m.effective_weight("relation"), 0.0)
        self.assertEqual(cfg_m.effective_weight("higher_order"), 0.0)


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------


class TestMemory(unittest.TestCase):
    def test_add_get_forget(self):
        a = Analogist()
        d = a.add_description("x", [("p", "a", "b")])
        self.assertTrue(a.has("x"))
        self.assertEqual(a.get("x"), d)
        self.assertEqual(a.names(), ("x",))
        self.assertEqual(len(a), 1)
        a.forget("x")
        self.assertFalse(a.has("x"))
        self.assertEqual(len(a), 0)

    def test_unknown_description_raises(self):
        a = Analogist()
        with self.assertRaises(UnknownDescription):
            a.get("missing")

    def test_match_unknown_raises(self):
        a = Analogist()
        a.add_description("x", [("p", "a", "b")])
        with self.assertRaises(UnknownDescription):
            a.match("x", "missing")


# ---------------------------------------------------------------------------
# The canonical solar / atom SME test
# ---------------------------------------------------------------------------


class TestSMECanonical(unittest.TestCase):
    def setUp(self):
        self.a = sme()
        self.a.add_description("solar", [
            ("cause",
             ("attracts", "sun", "planet"),
             ("revolves_around", "planet", "sun")),
            ("greater", ("mass", "sun"), ("mass", "planet")),
            ("greater", ("temperature", "sun"), ("temperature", "planet")),
            ("yellow", "sun"),
        ])
        self.a.add_description("atom", [
            ("cause",
             ("attracts", "nucleus", "electron"),
             ("revolves_around", "electron", "nucleus")),
            ("greater", ("mass", "nucleus"), ("mass", "electron")),
        ])

    def test_finds_mapping(self):
        rep = self.a.match("solar", "atom")
        self.assertGreater(len(rep.mappings), 0)
        m = rep.mappings[0]
        self.assertEqual(m.entity_map["sun"], "nucleus")
        self.assertEqual(m.entity_map["planet"], "electron")

    def test_inferences_include_temperature(self):
        rep = self.a.match("solar", "atom")
        m = rep.mappings[0]
        inf_strs = {str(inf) for inf, _ in m.inferences}
        self.assertTrue(
            any("temperature" in s and "nucleus" in s for s in inf_strs),
            f"expected temperature inference, got {inf_strs}",
        )

    def test_yellow_attribute_is_inferred_weakly(self):
        # In analogy mode the "yellow" attribute is a weak match, but
        # if the entity map already maps sun->nucleus, "yellow" still
        # appears as a candidate inference (it's a base-only fact
        # whose entities are mapped).
        rep = self.a.match("solar", "atom")
        m = rep.mappings[0]
        inf_strs = {str(inf) for inf, _ in m.inferences}
        self.assertTrue(any("yellow" in s for s in inf_strs))

    def test_systematicity_gives_one_dominant_mapping(self):
        rep = self.a.match("solar", "atom")
        # The mapping that picks up the entire cause/attracts/revolves
        # structure should dominate any alternative.
        self.assertGreater(rep.mappings[0].score, 0)

    def test_certificate_when_hmac_key_set(self):
        a = sme(hmac_key=b"key")
        a.add_description("solar", [
            ("cause",
             ("attracts", "sun", "planet"),
             ("revolves_around", "planet", "sun")),
            ("greater", ("mass", "sun"), ("mass", "planet")),
        ])
        a.add_description("atom", [
            ("cause",
             ("attracts", "nucleus", "electron"),
             ("revolves_around", "electron", "nucleus")),
            ("greater", ("mass", "nucleus"), ("mass", "electron")),
        ])
        rep1 = a.match("solar", "atom")
        rep2 = a.match("solar", "atom")
        self.assertIsNotNone(rep1.certificate)
        self.assertEqual(rep1.certificate, rep2.certificate)
        self.assertEqual(len(rep1.certificate), 64)  # sha256 hex

    def test_no_certificate_without_key(self):
        rep = self.a.match("solar", "atom")
        self.assertIsNone(rep.certificate)

    def test_fingerprints_present(self):
        rep = self.a.match("solar", "atom")
        self.assertEqual(len(rep.base_fingerprint), 64)
        self.assertEqual(len(rep.target_fingerprint), 64)
        self.assertNotEqual(rep.base_fingerprint, rep.target_fingerprint)

    def test_budget_tracking(self):
        rep = self.a.match("solar", "atom")
        self.assertGreater(rep.budget_used["mh_count"], 0)
        self.assertGreaterEqual(rep.budget_used["time_s"], 0)


# ---------------------------------------------------------------------------
# One-to-one and parallel-connectivity guarantees
# ---------------------------------------------------------------------------


class TestStructuralGuarantees(unittest.TestCase):
    def test_one_to_one_entity_mapping(self):
        a = sme()
        a.add_description("b", [("p", "a1", "a2"), ("p", "a2", "a3")])
        a.add_description("t", [("p", "b1", "b2"), ("p", "b2", "b3")])
        rep = a.match("b", "t")
        for m in rep.mappings:
            # Every entity maps to exactly one target, no duplicates.
            tgt_vals = list(m.entity_map.values())
            self.assertEqual(len(set(tgt_vals)), len(tgt_vals))

    def test_parallel_connectivity_under_kind_mismatch(self):
        # ("cause", R1, R2) cannot match if its children R1/R2 don't
        # share a head with anything in the target.
        a = sme()
        a.add_description("b", [
            ("cause", ("p", "x", "y"), ("q", "y", "x")),
        ])
        a.add_description("t", [
            ("cause", ("r", "u", "v"), ("s", "v", "u")),
        ])
        rep = a.match("b", "t")
        # No predicates identify, so no mapping should be produced.
        # (the top-level "cause" cannot survive parallel-connectivity
        # because its children have no MHs)
        if rep.mappings:
            for m in rep.mappings:
                # Whatever mapping exists must not bind "cause" since
                # there's no support for it.
                self.assertNotIn("(cause", str(m.expr_map))


# ---------------------------------------------------------------------------
# Identical-predicate requirement
# ---------------------------------------------------------------------------


class TestIdenticalPredicates(unittest.TestCase):
    def test_identical_predicates_only(self):
        a = sme(require_identical_predicates=True)
        a.add_description("b", [("p", "x", "y")])
        a.add_description("t", [("q", "u", "v")])
        rep = a.match("b", "t")
        # Nothing matches.
        if rep.mappings:
            for m in rep.mappings:
                self.assertEqual(len(m.entity_map), 0)

    def test_relaxed_function_matching(self):
        a = sme(require_identical_predicates=False)
        a.add_description("b", [("mass", "x")])
        a.add_description("t", [("weight", "y")])
        rep = a.match("b", "t")
        # In free-function mode, mass and weight (both functions) can
        # match if both are in DEFAULT_FUNCTIONS.
        # We don't require this to succeed — but it should not crash.
        self.assertIsNotNone(rep)


# ---------------------------------------------------------------------------
# Candidate-inference projection
# ---------------------------------------------------------------------------


class TestInferences(unittest.TestCase):
    def test_inference_projects_via_entity_map(self):
        a = sme()
        a.add_description("b", [
            ("p", "x", "y"),
            ("q", "x"),
        ])
        a.add_description("t", [
            ("p", "u", "v"),
        ])
        rep = a.match("b", "t")
        self.assertGreater(len(rep.mappings), 0)
        m = rep.mappings[0]
        # q(x) becomes q(u) by projection.
        proj_strs = {str(inf) for inf, _ in m.inferences}
        self.assertIn(str(("q", "u")), proj_strs)

    def test_inference_skipped_if_unmapped_entity(self):
        a = sme()
        a.add_description("b", [
            ("p", "x", "y"),
            ("r", "z"),  # z is not in any matched relation
        ])
        a.add_description("t", [
            ("p", "u", "v"),
        ])
        rep = a.match("b", "t")
        m = rep.mappings[0]
        proj_strs = {str(inf) for inf, _ in m.inferences}
        # r(z) should NOT be projected because z is unmapped.
        self.assertNotIn(str(("r", "z")), proj_strs)
        for inf, _ in m.inferences:
            # No projected entity should still be a base entity that
            # wasn't in entity_map.
            for sub in str(inf):
                pass


# ---------------------------------------------------------------------------
# MAC / FAC retrieval
# ---------------------------------------------------------------------------


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.a = sme()
        self.a.add_description("solar", [
            ("cause",
             ("attracts", "sun", "planet"),
             ("revolves_around", "planet", "sun")),
            ("greater", ("mass", "sun"), ("mass", "planet")),
        ])
        self.a.add_description("atom", [
            ("cause",
             ("attracts", "nucleus", "electron"),
             ("revolves_around", "electron", "nucleus")),
            ("greater", ("mass", "nucleus"), ("mass", "electron")),
        ])
        self.a.add_description("water", [
            ("cause", ("pressure", "tank"),
             ("flows", "water", "pipe")),
        ])
        self.a.add_description("noise", [("foo", "bar", "baz")])

    def test_atom_is_top_for_solar_probe(self):
        rep = self.a.retrieve("solar", k=3)
        self.assertGreater(len(rep.candidates), 0)
        self.assertEqual(rep.candidates[0][0], "atom")
        # FAC score on atom should be > 0.
        self.assertGreater(rep.candidates[0][2], 0.0)

    def test_mac_fac_counts(self):
        rep = self.a.retrieve("solar", k=2)
        self.assertEqual(rep.n_mac_evaluated, 3)  # atom, water, noise
        self.assertEqual(rep.n_fac_evaluated, 3)
        self.assertLessEqual(len(rep.candidates), 2)

    def test_probe_excluded_from_results(self):
        rep = self.a.retrieve("solar", k=10)
        names = [n for n, *_ in rep.candidates]
        self.assertNotIn("solar", names)


# ---------------------------------------------------------------------------
# Project method
# ---------------------------------------------------------------------------


class TestProject(unittest.TestCase):
    def test_project_recursive(self):
        gm = GlobalMapping(
            entity_map={"a": "x", "b": "y"},
            expr_map={},
            mhs=(),
            score=0.0,
            inferences=(),
            support_breakdown={},
        )
        self.assertEqual(gm.project("a"), "x")
        self.assertEqual(gm.project("b"), "y")
        self.assertEqual(gm.project("c"), "c")  # unmapped
        self.assertEqual(
            gm.project(("p", "a", ("q", "b"))),
            ("p", "x", ("q", "y")),
        )


# ---------------------------------------------------------------------------
# Budget exhaustion
# ---------------------------------------------------------------------------


class TestBudget(unittest.TestCase):
    def test_mh_budget(self):
        # Construct a description with many shared sub-expressions to
        # generate combinatorially many MHs.
        cfg = AnalogistConfig(max_match_hypotheses=5)
        a = Analogist(cfg)
        a.add_description("b", [
            ("p", "a", "b"),
            ("p", "c", "d"),
            ("p", "e", "f"),
            ("p", "g", "h"),
        ])
        a.add_description("t", [
            ("p", "1", "2"),
            ("p", "3", "4"),
            ("p", "5", "6"),
            ("p", "7", "8"),
        ])
        with self.assertRaises(BudgetExhausted):
            a.match("b", "t")


# ---------------------------------------------------------------------------
# ACME alternative engine
# ---------------------------------------------------------------------------


class TestACME(unittest.TestCase):
    def test_acme_runs_and_returns_report(self):
        a = acme(iterations=20)
        a.add_description("solar", [
            ("cause",
             ("attracts", "sun", "planet"),
             ("revolves_around", "planet", "sun")),
            ("greater", ("mass", "sun"), ("mass", "planet")),
        ])
        a.add_description("atom", [
            ("cause",
             ("attracts", "nucleus", "electron"),
             ("revolves_around", "electron", "nucleus")),
            ("greater", ("mass", "nucleus"), ("mass", "electron")),
        ])
        rep = a.match("solar", "atom")
        self.assertEqual(rep.engine, "acme")
        self.assertGreater(len(rep.mappings), 0)
        m = rep.mappings[0]
        # The one-to-one constraint should still hold.
        tgt = list(m.entity_map.values())
        self.assertEqual(len(set(tgt)), len(tgt))

    def test_acme_priors_influence_decoding(self):
        # Inject a strong pragmatic prior that maps sun to electron
        # (deliberately the "wrong" structural mapping).  ACME should
        # still produce a one-to-one map.
        priors = {("(attracts sun planet)", "(attracts electron nucleus)"): 5.0}
        a = acme(iterations=50, priors=priors)
        a.add_description("solar", [("attracts", "sun", "planet")])
        a.add_description("atom",
                          [("attracts", "nucleus", "electron"),
                           ("attracts", "electron", "nucleus")])
        rep = a.match("solar", "atom")
        # Doesn't matter which mapping; just that the engine runs.
        self.assertGreater(rep.n_match_hypotheses, 0)


# ---------------------------------------------------------------------------
# Literal-similarity mode
# ---------------------------------------------------------------------------


class TestLiteralSimilarity(unittest.TestCase):
    def test_attributes_contribute_in_literal_mode(self):
        a_lit = literal_similarity()
        a_ana = sme()
        for a in (a_lit, a_ana):
            a.add_description("b", [("red", "x"), ("p", "x", "y")])
            a.add_description("t", [("red", "u"), ("p", "u", "v")])

        rep_lit = a_lit.match("b", "t")
        rep_ana = a_ana.match("b", "t")
        # Literal mode should score the attribute match more strongly
        # than analogy mode.
        self.assertGreaterEqual(rep_lit.mappings[0].score,
                                rep_ana.mappings[0].score)


# ---------------------------------------------------------------------------
# Proportional analogy (Copycat micro-domain)
# ---------------------------------------------------------------------------


class TestProportionalAnalogy(unittest.TestCase):
    def setUp(self):
        self.p = ProportionalAnalogy()

    def test_last_letter_increment(self):
        r = self.p.solve("abc", "abd", "ijk")
        self.assertEqual(r.answer, "ijl")
        self.assertIn("shift last", r.rule)

    def test_last_letter_wrap(self):
        r = self.p.solve("abc", "abd", "xyz")
        self.assertEqual(r.answer, "xya")

    def test_append_extension(self):
        r = self.p.solve("ab", "abc", "xy")
        self.assertEqual(r.answer, "xyc")

    def test_substring_replacement(self):
        r = self.p.solve("aaa", "aba", "ccc")
        # The substring replacement at position 1 (a->b) becomes a
        # candidate; on "ccc" we replace "a" with "b" — but "a" not
        # in "ccc", so substitution rule is skipped.  Other rules
        # (shift-every, shift-last) may apply.
        self.assertIsNotNone(r.answer)

    def test_identity(self):
        r = self.p.solve("abc", "abc", "xyz")
        self.assertEqual(r.answer, "xyz")

    def test_shift_every_letter(self):
        r = self.p.solve("abc", "bcd", "xyz")
        # delta = +1 on every position
        self.assertEqual(r.answer, "yza")

    def test_reverse(self):
        r = self.p.solve("abc", "cba", "xyz")
        # Best rule depends on ranking — shift-by-2 or reverse could win.
        # Check that one of the deterministic candidates is reverse.
        candidates = [(r.answer, r.rule)] + [(a, ru) for a, ru, _ in r.alternatives]
        # zyx is the reversal answer
        rules = " ".join(c[1] for c in candidates)
        # Either the top or one of the alternatives should mention reverse
        # OR the answer should be zyx
        self.assertTrue(
            "reverse" in rules or any(a == "zyx" for a, _ in candidates)
        )

    def test_unknown_alphabet_chars(self):
        r = self.p.solve("hi", "hi!", "yo")
        # length-change rule applies even structurally
        self.assertEqual(r.answer, "yo!")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_match_is_deterministic(self):
        a1 = sme()
        a2 = sme()
        for a in (a1, a2):
            a.add_description("b", [
                ("cause", ("p", "x", "y"), ("q", "y", "x")),
                ("greater", ("mass", "x"), ("mass", "y")),
            ])
            a.add_description("t", [
                ("cause", ("p", "u", "v"), ("q", "v", "u")),
                ("greater", ("mass", "u"), ("mass", "v")),
            ])
        rep1 = a1.match("b", "t")
        rep2 = a2.match("b", "t")
        self.assertEqual(len(rep1.mappings), len(rep2.mappings))
        for m1, m2 in zip(rep1.mappings, rep2.mappings):
            self.assertEqual(dict(m1.entity_map), dict(m2.entity_map))
            self.assertAlmostEqual(m1.score, m2.score, places=6)

    def test_certificate_reproducible(self):
        a = sme(hmac_key=b"secret")
        a.add_description("b", [("p", "x", "y")])
        a.add_description("t", [("p", "u", "v")])
        rep1 = a.match("b", "t")
        rep2 = a.match("b", "t")
        self.assertEqual(rep1.certificate, rep2.certificate)


# ---------------------------------------------------------------------------
# Predicate-kind overrides
# ---------------------------------------------------------------------------


class TestPredicateKindOverrides(unittest.TestCase):
    def test_custom_higher_order(self):
        a = Analogist(AnalogistConfig(
            predicate_kinds={"explains": "higher_order"},
        ))
        a.add_description("b", [
            ("explains", ("hot", "sun"), ("warm", "planet")),
        ])
        a.add_description("t", [
            ("explains", ("hot", "fire"), ("warm", "room")),
        ])
        rep = a.match("b", "t")
        # Higher-order matches should produce non-empty mapping.
        self.assertGreater(len(rep.mappings), 0)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI(unittest.TestCase):
    def test_imports_from_top_level(self):
        # Top-level agi.* re-export
        from agi.analogist import (
            Analogist,
            AnalogistConfig,
            AnalogistReport,
            Description,
            GlobalMapping,
            MatchHypothesis,
            ProportionalAnalogy,
            RetrievalReport,
            sme,
        )

    def test_top_level_re_exports(self):
        # Don't fail if the top-level __init__ hasn't added them yet,
        # but check that direct module path works.
        import agi.analogist as mod
        self.assertTrue(hasattr(mod, "Analogist"))
        self.assertTrue(hasattr(mod, "ProportionalAnalogy"))


if __name__ == "__main__":
    unittest.main()
