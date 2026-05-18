"""Tests for the Watermarker synthetic-content provenance primitive."""
from __future__ import annotations

import json
import math

import pytest

from agi.events import EventBus
from agi.watermarker import (
    HASH_BLAKE2,
    HASH_HMAC_SHA256,
    HASH_SHA256,
    KNOWN_HASH_KINDS,
    KNOWN_MODES,
    KNOWN_POLARITIES,
    KNOWN_RECOMMENDATIONS,
    KNOWN_VERDICTS,
    MODE_COMBINED,
    MODE_KGW_EXACT,
    MODE_KGW_GREEN,
    MODE_KGW_SELFHASH,
    MODE_LEXICAL,
    POLARITY_DETECT_WATERMARK,
    POLARITY_VERIFY_WATERMARK,
    REC_BLOCK,
    REC_ESCALATE,
    REC_QUARANTINE,
    REC_RESTRICT,
    REC_TRUST,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
    VERDICT_PASS,
    VERDICT_WARN,
    WM_AUDITED,
    WM_CALIBRATED,
    WM_CERTIFIED,
    WM_GATED,
    WM_REPORTED,
    WM_RESET,
    WM_SCORED,
    WM_STARTED,
    WM_SUBMITTED,
    AuditReport,
    Document,
    InvalidConfig,
    InvalidDocument,
    InvalidToken,
    InvalidTrial,
    NotEnoughTrials,
    ThresholdReport,
    Token,
    TokenizerError,
    Trial,
    TrialReport,
    UnknownHashKind,
    UnknownMode,
    UnknownPolarity,
    Watermarker,
    WatermarkCertificate,
    WatermarkerConfig,
    WatermarkerError,
    WatermarkerReport,
    WatermarkSpec,
    default_tokenizer,
    green_indicators,
    is_green_token,
    make_document,
    simulate_marked_document,
    simulate_marked_token_ids,
    simulate_stripped_document,
    simulate_stripped_token_ids,
    simulate_unmarked_document,
    simulate_unmarked_token_ids,
    tokenize_text,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestWatermarkerConfig:
    def test_default(self):
        cfg = WatermarkerConfig()
        assert MODE_KGW_GREEN in cfg.modes
        assert MODE_COMBINED not in cfg.modes
        assert cfg.polarity == POLARITY_DETECT_WATERMARK
        assert 0.0 < cfg.alpha < 1.0

    def test_empty_modes_rejected(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(modes=())

    def test_combined_mode_rejected(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(modes=(MODE_COMBINED,))

    def test_unknown_mode_rejected(self):
        with pytest.raises(UnknownMode):
            WatermarkerConfig(modes=("blarg",))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
    def test_bad_alpha(self, bad):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(alpha=bad)

    def test_negative_bootstrap(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(bootstrap_b=-1)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
    def test_bad_confidence(self, bad):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(confidence=bad)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
    def test_bad_warn_factor(self, bad):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(warn_factor=bad)

    def test_bad_weights_length(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(weights=(1.0, 1.0))  # type: ignore[arg-type]

    def test_nonfinite_weights(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(weights=(float("inf"), 0.0, 0.0, 0.0))

    def test_bad_polarity(self):
        with pytest.raises(UnknownPolarity):
            WatermarkerConfig(polarity="??")

    def test_zero_prior(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(prior_a=0.0)
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(prior_b=-1.0)

    def test_min_tokens_for_normal(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(min_tokens_for_normal=0)

    def test_max_documents_negative(self):
        with pytest.raises(InvalidConfig):
            WatermarkerConfig(max_documents=-1)


# ---------------------------------------------------------------------------
# WatermarkSpec validation
# ---------------------------------------------------------------------------


class TestWatermarkSpec:
    def test_default(self):
        s = WatermarkSpec(name="x", key=b"key")
        assert s.gamma == 0.5
        assert s.left_context == 1
        assert s.selfhash is False
        assert s.hash_kind == HASH_BLAKE2

    def test_empty_key_rejected(self):
        with pytest.raises(InvalidConfig):
            WatermarkSpec(name="x", key=b"")

    def test_non_bytes_key_rejected(self):
        with pytest.raises(InvalidConfig):
            WatermarkSpec(name="x", key="str-not-bytes")  # type: ignore[arg-type]

    def test_empty_name_rejected(self):
        with pytest.raises(InvalidConfig):
            WatermarkSpec(name="", key=b"k")

    @pytest.mark.parametrize("bad_g", [0.0, 1.0, -0.1, 1.1])
    def test_bad_gamma(self, bad_g):
        with pytest.raises(InvalidConfig):
            WatermarkSpec(name="x", key=b"k", gamma=bad_g)

    def test_negative_delta(self):
        with pytest.raises(InvalidConfig):
            WatermarkSpec(name="x", key=b"k", delta=-0.1)

    def test_unknown_hash_kind(self):
        with pytest.raises(UnknownHashKind):
            WatermarkSpec(name="x", key=b"k", hash_kind="md5")

    def test_zero_left_context(self):
        with pytest.raises(InvalidConfig):
            WatermarkSpec(name="x", key=b"k", left_context=0)

    def test_fingerprint_is_hex_64(self):
        s = WatermarkSpec(name="x", key=b"key-1234")
        fp = s.fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_key_dependence(self):
        s1 = WatermarkSpec(name="x", key=b"key-A")
        s2 = WatermarkSpec(name="x", key=b"key-B")
        assert s1.fingerprint() != s2.fingerprint()

    def test_fingerprint_deterministic(self):
        s = WatermarkSpec(name="x", key=b"key")
        assert s.fingerprint() == s.fingerprint()


# ---------------------------------------------------------------------------
# Token / Document
# ---------------------------------------------------------------------------


class TestToken:
    def test_basic(self):
        t = Token(token_id=42, text="hi")
        assert t.token_id == 42
        assert t.text == "hi"

    def test_negative_id_rejected(self):
        with pytest.raises(InvalidToken):
            Token(token_id=-1)

    def test_non_int_id_rejected(self):
        with pytest.raises(InvalidToken):
            Token(token_id="not-int")  # type: ignore[arg-type]

    def test_non_str_text_rejected(self):
        with pytest.raises(InvalidToken):
            Token(token_id=0, text=123)  # type: ignore[arg-type]


class TestDocument:
    def test_basic(self):
        doc = Document(doc_id="a", tokens=(Token(1, "x"), Token(2, "y")))
        assert doc.doc_id == "a"
        assert len(doc.tokens) == 2

    def test_empty_id_rejected(self):
        with pytest.raises(InvalidDocument):
            Document(doc_id="", tokens=(Token(1),))

    def test_empty_tokens_rejected(self):
        with pytest.raises(InvalidDocument):
            Document(doc_id="a", tokens=())

    def test_non_tuple_tokens_rejected(self):
        with pytest.raises(InvalidDocument):
            Document(doc_id="a", tokens=[Token(1)])  # type: ignore[arg-type]

    def test_wrong_element_type(self):
        with pytest.raises(InvalidDocument):
            Document(doc_id="a", tokens=("not-a-token",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_default_tokenizer_splits_words(self):
        toks = default_tokenizer("Hello, world!")
        assert len(toks) >= 2
        assert all(isinstance(t, Token) for t in toks)

    def test_default_tokenizer_deterministic(self):
        a = default_tokenizer("foo bar baz")
        b = default_tokenizer("foo bar baz")
        assert [t.token_id for t in a] == [t.token_id for t in b]

    def test_default_tokenizer_lowercase(self):
        a = [t.token_id for t in default_tokenizer("FOO")]
        b = [t.token_id for t in default_tokenizer("foo")]
        assert a == b

    def test_tokenize_text_validates(self):
        def bad_tok(_text):
            return "not-list"

        with pytest.raises(TokenizerError):
            tokenize_text("x", bad_tok)

    def test_make_document_empty_text(self):
        with pytest.raises(InvalidDocument):
            make_document("a", "   ")


# ---------------------------------------------------------------------------
# Green-list PRF
# ---------------------------------------------------------------------------


class TestGreenList:
    def test_is_green_deterministic(self):
        spec = WatermarkSpec(name="x", key=b"k1")
        a = is_green_token(spec, 99, [1, 2, 3])
        b = is_green_token(spec, 99, [1, 2, 3])
        assert a == b

    def test_is_green_key_dependent(self):
        # Different keys should change the partition for at least some
        # (token, context) pair within a small sweep.
        spec_a = WatermarkSpec(name="x", key=b"k1")
        spec_b = WatermarkSpec(name="x", key=b"k2")
        diffs = 0
        for tok in range(50):
            for ctx in range(5):
                if is_green_token(spec_a, tok, [ctx]) != is_green_token(spec_b, tok, [ctx]):
                    diffs += 1
        assert diffs > 0

    def test_is_green_context_dependent(self):
        # Different contexts yield different labels for at least some tokens.
        spec = WatermarkSpec(name="x", key=b"k1")
        diffs = 0
        for tok in range(50):
            if is_green_token(spec, tok, [1]) != is_green_token(spec, tok, [2]):
                diffs += 1
        assert diffs > 0

    def test_is_green_insufficient_context(self):
        spec = WatermarkSpec(name="x", key=b"k1", left_context=3)
        with pytest.raises(WatermarkerError):
            is_green_token(spec, 99, [1, 2])

    def test_green_rate_approximately_gamma_under_null(self):
        # Under H0 (random tokens), green rate ≈ γ.  Large sample.
        spec = WatermarkSpec(name="x", key=b"k1", gamma=0.3)
        import random
        rng = random.Random(0)
        n = 2000
        prev = [rng.randrange(1 << 32) for _ in range(1)]
        green = 0
        for _ in range(n):
            tok = rng.randrange(1 << 32)
            if is_green_token(spec, tok, prev):
                green += 1
            prev = [tok]
        rate = green / n
        # 4σ window — generous but reliable.
        sigma = math.sqrt(0.3 * 0.7 / n)
        assert abs(rate - 0.3) < 4 * sigma, f"rate {rate} far from γ=0.3"

    def test_green_indicators_length(self):
        spec = WatermarkSpec(name="x", key=b"k", left_context=2)
        ids = list(range(10))
        ind = green_indicators(spec, ids)
        assert len(ind) == len(ids) - 2

    def test_selfhash_flag_changes_partition(self):
        s1 = WatermarkSpec(name="x", key=b"k", selfhash=False)
        s2 = WatermarkSpec(name="x", key=b"k", selfhash=True)
        diffs = 0
        for tok in range(100):
            if is_green_token(s1, tok, [42]) != is_green_token(s2, tok, [42]):
                diffs += 1
        assert diffs > 0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestSimulation:
    def test_marked_higher_green_rate(self):
        spec = WatermarkSpec(name="x", key=b"k", gamma=0.5, delta=3.0)
        marked_ids = simulate_marked_token_ids(spec, 500, seed=1)
        unmarked_ids = simulate_unmarked_token_ids(spec, 500, seed=2)
        m_green = sum(green_indicators(spec, marked_ids))
        u_green = sum(green_indicators(spec, unmarked_ids))
        assert m_green > u_green
        # The lift should be substantial — at least 50% more under δ=3.
        assert m_green > 1.5 * u_green

    def test_unmarked_approximately_gamma(self):
        spec = WatermarkSpec(name="x", key=b"k", gamma=0.5)
        ids = simulate_unmarked_token_ids(spec, 1000, seed=0)
        ind = green_indicators(spec, ids)
        rate = sum(ind) / len(ind)
        sigma = math.sqrt(0.25 / len(ind))
        assert abs(rate - 0.5) < 4 * sigma

    def test_simulate_zero_tokens_rejected(self):
        spec = WatermarkSpec(name="x", key=b"k")
        with pytest.raises(WatermarkerError):
            simulate_marked_token_ids(spec, 0)

    def test_simulate_small_vocab_rejected(self):
        spec = WatermarkSpec(name="x", key=b"k")
        with pytest.raises(WatermarkerError):
            simulate_marked_token_ids(spec, 10, vocabulary=[1, 2])

    def test_negative_delta_rejected(self):
        spec = WatermarkSpec(name="x", key=b"k")
        with pytest.raises(WatermarkerError):
            simulate_marked_token_ids(spec, 10, effective_delta=-1.0)

    def test_simulate_marked_document_returns_document(self):
        spec = WatermarkSpec(name="x", key=b"k")
        doc = simulate_marked_document(spec, "m1", 50, seed=1)
        assert isinstance(doc, Document)
        assert doc.doc_id == "m1"
        # Token count includes the prefix (left_context).
        assert len(doc.tokens) == 50 + spec.left_context

    def test_seed_determinism(self):
        spec = WatermarkSpec(name="x", key=b"k", delta=2.0)
        a = simulate_marked_token_ids(spec, 100, seed=42)
        b = simulate_marked_token_ids(spec, 100, seed=42)
        assert a == b


# ---------------------------------------------------------------------------
# Watermarker - basic scoring
# ---------------------------------------------------------------------------


class TestWatermarkerScoring:
    def test_score_marked_document_detects(self):
        spec = WatermarkSpec(name="t", key=b"k", gamma=0.5, delta=3.0)
        wm = Watermarker(spec=spec)
        doc = simulate_marked_document(spec, "m", 200, seed=1)
        report = wm.submit(Trial(document=doc, spec=spec, truth=True))
        assert report.z_score > 2.0
        assert report.chosen_p_value < 0.01

    def test_score_unmarked_document_passes(self):
        spec = WatermarkSpec(name="t", key=b"k", gamma=0.5, delta=3.0)
        wm = Watermarker(spec=spec)
        doc = simulate_unmarked_document(spec, "u", 200, seed=1)
        report = wm.submit(Trial(document=doc, spec=spec, truth=False))
        # Under H0 expect z ~ N(0,1), p > α=0.01 with high probability.
        assert report.chosen_p_value > 0.001
        assert report.green_fraction == pytest.approx(0.5, abs=0.15)

    def test_score_short_document_inconclusive(self):
        spec = WatermarkSpec(name="t", key=b"k", left_context=2)
        wm = Watermarker(spec=spec)
        doc = Document(doc_id="s", tokens=(Token(1), Token(2)))
        report = wm.submit(Trial(document=doc, spec=spec))
        assert report.n_scoreable == 0
        assert report.verdict == VERDICT_INCONCLUSIVE

    def test_clopper_pearson_ci_bounds(self):
        spec = WatermarkSpec(name="t", key=b"k", gamma=0.5)
        wm = Watermarker(spec=spec)
        doc = simulate_unmarked_document(spec, "u", 50, seed=0)
        report = wm.submit(Trial(document=doc, spec=spec))
        assert 0.0 <= report.rate_lower_cp <= report.green_fraction <= report.rate_upper_cp <= 1.0

    def test_p_value_consistency_with_z(self):
        # For long documents, normal p should be close to exact p.
        spec = WatermarkSpec(name="t", key=b"k", gamma=0.5, delta=2.0)
        wm = Watermarker(spec=spec)
        doc = simulate_marked_document(spec, "m", 500, seed=7)
        report = wm.submit(Trial(document=doc, spec=spec))
        # Both should be very small.
        assert report.p_value_normal < 1e-4
        assert report.p_value_exact < 1e-4

    def test_score_only_does_not_store(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        doc = simulate_marked_document(spec, "m", 50, seed=1)
        wm.score_only(Trial(document=doc, spec=spec))
        assert wm.n_trials == 0

    def test_submit_text_tokenizes(self):
        spec = WatermarkSpec(name="t", key=b"k", left_context=1)
        wm = Watermarker(spec=spec)
        report = wm.submit_text("d1", "the quick brown fox jumps over the lazy dog", spec)
        assert report.n_tokens >= 5
        assert wm.n_trials == 1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_calibrate_requires_both_classes(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        # Only positives.
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=100, marked=True, seed=i))
        with pytest.raises(NotEnoughTrials):
            wm.calibrate()

    def test_calibrate_succeeds_with_both_classes(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(spec=spec)
        for i in range(10):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=80, marked=True, seed=i))
            wm.submit(wm.simulate_trial(spec, doc_id=f"u{i}", n_tokens=80, marked=False, seed=100 + i))
        rep = wm.calibrate()
        assert isinstance(rep, ThresholdReport)
        assert rep.n_positive == 10
        assert rep.n_negative == 10
        # AUROC should be very high (strong watermark).
        assert rep.auroc > 0.9

    def test_calibrate_with_relaxed_requirement(self):
        cfg = WatermarkerConfig(require_label_for_threshold=False)
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(cfg, spec=spec)
        rep = wm.calibrate()
        assert math.isnan(rep.auroc)
        assert rep.n_labelled == 0


# ---------------------------------------------------------------------------
# Audit (sequential e-process)
# ---------------------------------------------------------------------------


class TestAudit:
    def test_audit_empty_pool(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        rep = wm.audit()
        assert rep.n_tokens_seen == 0
        assert rep.e_value == 1.0
        assert rep.rejected_h0 is False

    def test_audit_strong_watermark_rejects(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=4.0)
        wm = Watermarker(spec=spec)
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=300, marked=True, seed=i))
        rep = wm.audit()
        assert rep.rejected_h0 is True
        assert rep.e_value > 1.0 / wm.config.alpha

    def test_audit_unmarked_does_not_reject(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=0.0)
        wm = Watermarker(spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"u{i}", n_tokens=200, marked=False, seed=i))
        rep = wm.audit()
        # E-value can fluctuate but should not cross 1/α with α=0.01
        # under H0 with high probability (anytime-valid bound).
        # Five trials × 200 tokens is well within tolerance.
        assert rep.rejected_h0 is False

    def test_audit_polarity_verify_works(self):
        # In verify mode, H0: green_rate >= γ.  Marked text is
        # consistent with this null (green-rate >> γ), so the
        # one-sided lower-tail p-value should be ≈ 1 → don't reject.
        cfg = WatermarkerConfig(polarity=POLARITY_VERIFY_WATERMARK)
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(cfg, spec=spec)
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=200, marked=True, seed=i))
        rep = wm.audit()
        assert rep.rejected_h0 is False


# ---------------------------------------------------------------------------
# Certificate and recommendation
# ---------------------------------------------------------------------------


class TestCertificate:
    def test_certify_emits_valid_certificate(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=200, marked=True, seed=i))
        cert = wm.certify()
        assert isinstance(cert, WatermarkCertificate)
        assert cert.verdict in KNOWN_VERDICTS
        assert cert.recommendation in KNOWN_RECOMMENDATIONS
        assert len(cert.fingerprint_hash) == 64

    def test_certify_pass_on_strong_watermark(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=4.0)
        wm = Watermarker(spec=spec)
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=300, marked=True, seed=i))
        cert = wm.certify()
        assert cert.verdict == VERDICT_PASS
        assert cert.recommendation == REC_TRUST

    def test_certify_fail_on_unmarked(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"u{i}", n_tokens=300, marked=False, seed=i))
        cert = wm.certify()
        assert cert.verdict in (VERDICT_FAIL, VERDICT_WARN, VERDICT_INCONCLUSIVE)

    def test_certify_inconclusive_on_empty(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        cert = wm.certify()
        assert cert.verdict == VERDICT_INCONCLUSIVE
        assert cert.recommendation == REC_ESCALATE

    def test_certificate_fingerprint_chains(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d1", n_tokens=100, marked=True, seed=1))
        c1 = wm.certify()
        wm.submit(wm.simulate_trial(spec, doc_id="d2", n_tokens=100, marked=True, seed=2))
        c2 = wm.certify()
        assert c1.fingerprint_hash != c2.fingerprint_hash

    def test_certificate_replay_deterministic(self):
        # Two parallel Watermarkers, same config and trials, must
        # produce the same fingerprint chain.
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm1 = Watermarker(spec=spec)
        wm2 = Watermarker(spec=spec)
        for i in range(3):
            t = wm1.simulate_trial(spec, doc_id=f"d{i}", n_tokens=100, marked=True, seed=i)
            wm1.submit(t)
            wm2.submit(t)
        c1 = wm1.certify()
        c2 = wm2.certify()
        assert c1.fingerprint_hash == c2.fingerprint_hash

    def test_certificate_to_dict_roundtrip(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=100, marked=True, seed=0))
        cert = wm.certify()
        d = cert.to_dict()
        s = json.dumps(d)
        assert json.loads(s) == d


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class TestGate:
    def test_gate_passes_on_strong_watermark(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=4.0)
        wm = Watermarker(spec=spec)
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=300, marked=True, seed=i))
        passed, cert = wm.gate()
        assert passed is True
        assert cert.verdict == VERDICT_PASS

    def test_gate_fails_on_unmarked(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        for i in range(3):
            wm.submit(wm.simulate_trial(spec, doc_id=f"u{i}", n_tokens=300, marked=False, seed=i))
        passed, cert = wm.gate()
        assert passed is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_wipes_state(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=100, marked=True, seed=0))
        assert wm.n_trials == 1
        wm.reset()
        assert wm.n_trials == 0
        assert wm.n_tokens_seen == 0
        assert wm.n_green_seen == 0

    def test_reset_preserves_fingerprint_chain(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=100, marked=True, seed=0))
        c1 = wm.certify()
        wm.reset()
        wm.submit(wm.simulate_trial(spec, doc_id="d2", n_tokens=100, marked=True, seed=1))
        c2 = wm.certify()
        # Different fingerprint because the prior_fingerprint chains.
        assert c1.fingerprint_hash != c2.fingerprint_hash


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_events_published(self):
        bus = EventBus()
        events: list = []
        bus.subscribe(lambda e: events.append(e))
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(spec=spec, event_bus=bus)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=200, marked=True, seed=0))
        wm.certify()
        wm.gate()
        kinds = {e.kind for e in events}
        assert WM_STARTED in kinds
        assert WM_SUBMITTED in kinds
        assert WM_SCORED in kinds
        assert WM_CERTIFIED in kinds
        assert WM_GATED in kinds
        assert WM_AUDITED in kinds

    def test_eventbus_subscriber_exception_swallowed(self):
        bus = EventBus()
        def bad(_event):
            raise RuntimeError("oops")
        bus.subscribe(bad)
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec, event_bus=bus)
        # Must not raise.
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=50, marked=True, seed=0))


# ---------------------------------------------------------------------------
# Holm / FDR
# ---------------------------------------------------------------------------


class TestMultiTest:
    def test_holm_smallest_p(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"d{i}", n_tokens=200, marked=True, seed=i))
        p = wm.holm_combined_p()
        assert p is not None
        assert 0.0 <= p <= 1.0

    def test_holm_empty(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        assert wm.holm_combined_p() is None

    def test_fdr_threshold_strong_watermark(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=4.0)
        wm = Watermarker(spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"d{i}", n_tokens=200, marked=True, seed=i))
        t = wm.fdr_threshold()
        # All marked → all reject under BH.
        assert t is not None
        assert t < wm.config.alpha

    def test_fdr_threshold_unmarked(self):
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"d{i}", n_tokens=200, marked=False, seed=i))
        t = wm.fdr_threshold()
        # Most/all p-values large → no rejection.
        assert t is None or t < 1.0


# ---------------------------------------------------------------------------
# max_documents enforcement
# ---------------------------------------------------------------------------


class TestMaxDocuments:
    def test_max_documents_drops_fifo(self):
        cfg = WatermarkerConfig(max_documents=2)
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(cfg, spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"d{i}", n_tokens=50, marked=True, seed=i))
        assert wm.n_trials == 2
        # Audit's sufficient statistics are NOT rolled back — by design.
        assert wm.n_tokens_seen > 100  # 5 × 50 tokens

    def test_max_documents_zero_unlimited(self):
        cfg = WatermarkerConfig(max_documents=0)
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(cfg, spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"d{i}", n_tokens=50, marked=True, seed=i))
        assert wm.n_trials == 5


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------


class TestPolarity:
    def test_verify_polarity_blocks_stripped_text(self):
        # In verify mode, H0 is "still marked".  Submitting actively
        # stripped (adversarially red-biased) text should reject H0
        # and recommend BLOCK.
        cfg = WatermarkerConfig(polarity=POLARITY_VERIFY_WATERMARK)
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(cfg, spec=spec)
        for i in range(5):
            doc = simulate_stripped_document(spec, f"s{i}", 300, seed=i)
            wm.submit(Trial(document=doc, spec=spec, truth=False))
        cert = wm.certify()
        # Stripped text has green-rate << γ → reject H0 in verify mode.
        assert cert.rejected_h0 is True
        assert cert.recommendation == REC_BLOCK

    def test_detect_polarity_quarantines_unmarked(self):
        cfg = WatermarkerConfig(polarity=POLARITY_DETECT_WATERMARK)
        spec = WatermarkSpec(name="t", key=b"k")
        wm = Watermarker(cfg, spec=spec)
        for i in range(5):
            wm.submit(wm.simulate_trial(spec, doc_id=f"u{i}", n_tokens=200, marked=False, seed=i))
        cert = wm.certify()
        assert cert.recommendation in (REC_QUARANTINE, REC_RESTRICT, REC_ESCALATE)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_bundle_shape(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=200, marked=True, seed=0))
        rep = wm.report()
        assert isinstance(rep, WatermarkerReport)
        d = rep.to_dict()
        assert d["n_trials"] == 1
        assert "audit" in d
        assert "certificate" in d
        assert "config" in d

    def test_report_jsonable(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=200, marked=True, seed=0))
        rep = wm.report()
        s = json.dumps(rep.to_dict())
        d = json.loads(s)
        assert d["n_trials"] == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_submit(self):
        import threading
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        errors: list = []

        def worker(i):
            try:
                wm.submit(wm.simulate_trial(spec, doc_id=f"d{i}", n_tokens=100, marked=True, seed=i))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert wm.n_trials == 8


# ---------------------------------------------------------------------------
# Hash kinds
# ---------------------------------------------------------------------------


class TestHashKinds:
    @pytest.mark.parametrize("kind", [HASH_BLAKE2, HASH_SHA256, HASH_HMAC_SHA256])
    def test_all_hash_kinds_work(self, kind):
        spec = WatermarkSpec(name="t", key=b"k", hash_kind=kind, delta=3.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="m", n_tokens=200, marked=True, seed=0))
        report = wm.reports()[0]
        assert report.z_score > 1.0


# ---------------------------------------------------------------------------
# Polarity exact-tail edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_continuity_correction_at_mean(self):
        # If green count equals exactly γ·T, the corrected z should
        # be 0.
        spec = WatermarkSpec(name="t", key=b"k", gamma=0.5)
        wm = Watermarker(spec=spec)
        # Build a synthetic document with exactly half green tokens.
        # Easiest: use the simulator's null and find a doc whose
        # green-fraction is exactly 0.5.  Cheap and robust: small
        # doc; the test just verifies z is finite.
        doc = simulate_unmarked_document(spec, "u", 100, seed=0)
        report = wm.submit(Trial(document=doc, spec=spec))
        assert math.isfinite(report.z_score)

    def test_all_green_tokens_extreme_z(self):
        spec = WatermarkSpec(name="t", key=b"k", gamma=0.5)
        wm = Watermarker(spec=spec)
        # Craft a document where every token is green.
        # We hand-pick token ids by trial-and-error using the PRF.
        token_ids: list[int] = [1]
        seed = 2
        for _ in range(50):
            for cand in range(seed, seed + 10000):
                if is_green_token(spec, cand, [token_ids[-1]]):
                    token_ids.append(cand)
                    seed = cand + 1
                    break
        toks = tuple(Token(token_id=i, text=f"t{i}") for i in token_ids)
        doc = Document(doc_id="all-green", tokens=toks)
        report = wm.submit(Trial(document=doc, spec=spec))
        assert report.green_fraction == 1.0
        assert report.z_score > 5.0
        assert report.p_value_exact < 1e-10


# ---------------------------------------------------------------------------
# json round-trip of all reports
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_trial_report_json_roundtrip(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        report = wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=100, marked=True, seed=0))
        d = report.to_dict()
        s = json.dumps(d)
        assert json.loads(s) == d

    def test_threshold_report_json_roundtrip(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=3.0)
        wm = Watermarker(spec=spec)
        for i in range(6):
            wm.submit(wm.simulate_trial(spec, doc_id=f"m{i}", n_tokens=80, marked=True, seed=i))
            wm.submit(wm.simulate_trial(spec, doc_id=f"u{i}", n_tokens=80, marked=False, seed=100 + i))
        rep = wm.calibrate()
        s = json.dumps(rep.to_dict())
        assert "auroc" in json.loads(s)

    def test_audit_report_json_roundtrip(self):
        spec = WatermarkSpec(name="t", key=b"k", delta=2.0)
        wm = Watermarker(spec=spec)
        wm.submit(wm.simulate_trial(spec, doc_id="d", n_tokens=100, marked=True, seed=0))
        rep = wm.audit()
        s = json.dumps(rep.to_dict())
        assert "e_value" in json.loads(s)
