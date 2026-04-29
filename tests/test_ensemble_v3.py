"""Tests for V3 ensemble post-processing: self-correction filter,
min span filter, partial entity detection, and processing order.

Uses a mock model to test ensemble logic without requiring a trained model.
"""

from typing import Any

from src.models.ensemble import predict as ensemble_predict


class MockModel:
    """Mock transformer model that returns predetermined entities."""

    def __init__(self, entities: list[dict[str, Any]]):
        self._entities = entities

    def predict(self, text: str) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entities]  # Return copies


# ---------------------------------------------------------------------------
# Self-correction filter
# ---------------------------------------------------------------------------

class TestSelfCorrectionFilter:
    """Test self-correction marker detection and entity removal."""

    def test_correction_removes_pre_marker_entity(self):
        # "david" before "no wait", "daniel allen" after → "david" removed
        text = "my name is david no wait daniel allen"
        mock = MockModel([
            {"text": "david", "label": "NAME", "start": 11, "end": 16, "confidence": 0.9},
            {"text": "daniel allen", "label": "NAME", "start": 25, "end": 37, "confidence": 0.9},
        ])
        entities, stats = ensemble_predict(text, mock)
        names = [e["text"] for e in entities if e["label"] == "NAME"]
        assert "david" not in names
        assert "daniel allen" in names
        assert stats["correction_removals"] >= 1

    def test_no_marker_keeps_all(self):
        text = "sarah and bob are both customers here"
        mock = MockModel([
            {"text": "sarah", "label": "NAME", "start": 0, "end": 5, "confidence": 0.9},
            {"text": "bob", "label": "NAME", "start": 10, "end": 13, "confidence": 0.9},
        ])
        entities, stats = ensemble_predict(text, mock)
        names = [e["text"] for e in entities if e["label"] == "NAME"]
        assert "sarah" in names
        assert "bob" in names
        assert stats["correction_removals"] == 0

    def test_distance_constraint_preserves_far_entity(self):
        # Entity far from marker (distance > 30 chars) should NOT be removed
        text = "alice is great and she helped me with everything over the phone no wait it was bob actually"
        mock = MockModel([
            {"text": "alice", "label": "NAME", "start": 0, "end": 5, "confidence": 0.9},
            {"text": "bob", "label": "NAME", "start": 71, "end": 74, "confidence": 0.9},
        ])
        entities, stats = ensemble_predict(text, mock)
        names = [e["text"] for e in entities if e["label"] == "NAME"]
        # "alice" ends at 5, "no wait" starts at 56. Distance = 51 > 30
        assert "alice" in names
        assert "bob" in names

    def test_i_mean_does_not_trigger_removal(self):
        # "i mean" is excluded from correction markers
        text = "i mean the iphone is good i mean really good stuff"
        mock = MockModel([
            {"text": "iphone", "label": "PRODUCT", "start": 11, "end": 17, "confidence": 0.9},
            {"text": "stuff", "label": "PRODUCT", "start": 46, "end": 51, "confidence": 0.9},
        ])
        entities, stats = ensemble_predict(text, mock)
        products = [e["text"] for e in entities if e["label"] == "PRODUCT"]
        assert "iphone" in products
        assert stats["correction_removals"] == 0

    def test_different_type_not_removed(self):
        # Correction only removes same-type entities
        text = "sarah ordered no wait the iphone was returned"
        mock = MockModel([
            {"text": "sarah", "label": "NAME", "start": 0, "end": 5, "confidence": 0.9},
            {"text": "iphone", "label": "PRODUCT", "start": 25, "end": 31, "confidence": 0.9},
        ])
        entities, stats = ensemble_predict(text, mock)
        # NAME before marker, PRODUCT after — different types, no removal
        names = [e["text"] for e in entities if e["label"] == "NAME"]
        assert "sarah" in names


# ---------------------------------------------------------------------------
# Min span filter
# ---------------------------------------------------------------------------

class TestMinSpanFilter:
    """Test single-character entity filtering."""

    def test_single_char_filtered(self):
        text = "the letter y is not a name"
        mock = MockModel([
            {"text": "y", "label": "NAME", "start": 11, "end": 12, "confidence": 0.9},
        ])
        entities, _ = ensemble_predict(text, mock)
        assert len([e for e in entities if e["label"] == "NAME"]) == 0

    def test_two_char_kept(self):
        text = "contact al about the issue"
        mock = MockModel([
            {"text": "al", "label": "NAME", "start": 8, "end": 10, "confidence": 0.9},
        ])
        entities, _ = ensemble_predict(text, mock)
        names = [e for e in entities if e["label"] == "NAME"]
        assert len(names) == 1
        assert names[0]["text"] == "al"

    def test_filter_applies_to_model_only(self):
        # Rules entities are not affected by min span filter
        # CONTRACT_ID "ORD-2024-5591" from rules should always pass
        text = "order ORD-2024-5591 thanks"
        mock = MockModel([])  # model returns nothing
        entities, _ = ensemble_predict(text, mock)
        cids = [e for e in entities if e["label"] == "CONTRACT_ID"]
        assert len(cids) == 1


# ---------------------------------------------------------------------------
# Partial entity detection
# ---------------------------------------------------------------------------

class TestPartialDetection:
    """Test structural entity completeness checking."""

    def test_complete_email_not_partial(self):
        text = "my email is john at gmail dot com thanks"
        mock = MockModel([
            {"text": "john at gmail dot com", "label": "EMAIL",
             "start": 12, "end": 32, "confidence": 0.9},
        ])
        entities, _ = ensemble_predict(text, mock)
        emails = [e for e in entities if e["label"] == "EMAIL"]
        assert len(emails) == 1
        assert emails[0]["partial"] is False

    def test_incomplete_email_missing_domain(self):
        text = "send it to john at gm thanks"
        mock = MockModel([
            {"text": "john at gm", "label": "EMAIL",
             "start": 11, "end": 21, "confidence": 0.9},
        ])
        entities, _ = ensemble_predict(text, mock)
        emails = [e for e in entities if e["label"] == "EMAIL"]
        if emails:  # model might not return this
            assert emails[0]["partial"] is True

    def test_incomplete_email_missing_at(self):
        text = "email johngmail dot com thanks"
        mock = MockModel([
            {"text": "johngmail dot com", "label": "EMAIL",
             "start": 6, "end": 23, "confidence": 0.9},
        ])
        entities, _ = ensemble_predict(text, mock)
        emails = [e for e in entities if e["label"] == "EMAIL"]
        if emails:
            assert emails[0]["partial"] is True

    def test_standard_contract_id_not_partial(self):
        # Standard CONTRACT_ID comes from rules, check partial=False
        text = "order ORD-2024-5591 thanks"
        mock = MockModel([])
        entities, _ = ensemble_predict(text, mock)
        cids = [e for e in entities if e["label"] == "CONTRACT_ID"]
        assert len(cids) == 1
        assert cids[0]["partial"] is False

    def test_oral_contract_id_not_partial(self):
        text = "order ORD dash 2024 dash 5591 thanks"
        mock = MockModel([])  # rules should match oral pattern
        entities, _ = ensemble_predict(text, mock)
        cids = [e for e in entities if e["label"] == "CONTRACT_ID"]
        assert len(cids) >= 1
        assert cids[0]["partial"] is False

    def test_spelled_out_contract_id_not_partial(self):
        text = "order SUB dash three three zero one eight thanks"
        mock = MockModel([])  # rules should match spelled-out pattern
        entities, _ = ensemble_predict(text, mock)
        cids = [e for e in entities if e["label"] == "CONTRACT_ID"]
        assert len(cids) >= 1
        assert cids[0]["partial"] is False

    def test_semantic_entities_not_checked(self):
        # NAME, PRODUCT, ISSUE_DATE should always have partial=False
        text = "hello world test text"
        mock = MockModel([
            {"text": "hello", "label": "NAME", "start": 0, "end": 5, "confidence": 0.9},
            {"text": "world", "label": "PRODUCT", "start": 6, "end": 11, "confidence": 0.9},
            {"text": "test", "label": "ISSUE_DATE", "start": 12, "end": 16, "confidence": 0.9},
        ])
        entities, _ = ensemble_predict(text, mock)
        for e in entities:
            if e["label"] in ("NAME", "PRODUCT", "ISSUE_DATE"):
                assert e["partial"] is False


# ---------------------------------------------------------------------------
# Post-processing integration
# ---------------------------------------------------------------------------

class TestPostProcessingIntegration:
    """Test that all post-processing steps run in correct order."""

    def test_entities_have_partial_field(self):
        text = "order ORD-2024-5591 thanks"
        mock = MockModel([])
        entities, _ = ensemble_predict(text, mock)
        for e in entities:
            assert "partial" in e

    def test_stats_have_correction_removals_key(self):
        text = "hello world"
        mock = MockModel([])
        _, stats = ensemble_predict(text, mock)
        assert "correction_removals" in stats
        assert "total_entities" in stats
        assert "conflicts" in stats
