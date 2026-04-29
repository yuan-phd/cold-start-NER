"""Tests for BIOES tagging scheme (V3).

Covers: label definitions, char→label conversion, token alignment,
entity finalization, and predict decoding.
"""

import pytest
from pathlib import Path

from src.models.transformer_ner import (
    BIOES_LABELS,
    LABEL2ID,
    ID2LABEL,
    NERDataset,
    TransformerNER,
)


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

class TestBIOESLabels:
    """Verify BIOES label constants are correct."""

    def test_label_count(self):
        assert len(BIOES_LABELS) == 21  # O + 4 tags × 5 entity types

    def test_first_label_is_O(self):
        assert BIOES_LABELS[0] == "O"

    def test_all_entity_types_have_four_tags(self):
        for etype in ["NAME", "EMAIL", "CONTRACT_ID", "PRODUCT", "ISSUE_DATE"]:
            for prefix in ["B-", "I-", "E-", "S-"]:
                assert f"{prefix}{etype}" in BIOES_LABELS

    def test_label2id_id2label_consistency(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label
        for idx, label in ID2LABEL.items():
            assert LABEL2ID[label] == idx

    def test_label2id_length_matches(self):
        assert len(LABEL2ID) == len(BIOES_LABELS)
        assert len(ID2LABEL) == len(BIOES_LABELS)


# ---------------------------------------------------------------------------
# Character-level BIOES label generation
# ---------------------------------------------------------------------------

class TestEntitiesToCharLabels:
    """Test _entities_to_char_labels produces correct BIOES labels."""

    def test_no_entities_all_O(self):
        labels = NERDataset._entities_to_char_labels(None, "hello world", [])
        assert all(lab == "O" for lab in labels)
        assert len(labels) == len("hello world")

    def test_single_char_entity_gets_S(self):
        text = "a test"
        entities = [{"start": 0, "end": 1, "label": "NAME"}]
        labels = NERDataset._entities_to_char_labels(None, text, entities)
        assert labels[0] == "S-NAME"
        assert all(lab == "O" for lab in labels[1:])

    def test_two_char_entity_gets_B_E(self):
        text = "ab test"
        entities = [{"start": 0, "end": 2, "label": "NAME"}]
        labels = NERDataset._entities_to_char_labels(None, text, entities)
        assert labels[0] == "B-NAME"
        assert labels[1] == "E-NAME"
        assert all(lab == "O" for lab in labels[2:])

    def test_multi_char_entity_gets_B_I_E(self):
        text = "john smith is here"
        entities = [{"start": 0, "end": 10, "label": "NAME"}]
        labels = NERDataset._entities_to_char_labels(None, text, entities)
        assert labels[0] == "B-NAME"
        assert labels[9] == "E-NAME"
        assert all(lab == "I-NAME" for lab in labels[1:9])
        assert all(lab == "O" for lab in labels[10:])

    def test_multiple_entities(self):
        text = "john bought iphone"
        entities = [
            {"start": 0, "end": 4, "label": "NAME"},
            {"start": 12, "end": 18, "label": "PRODUCT"},
        ]
        labels = NERDataset._entities_to_char_labels(None, text, entities)
        # NAME: john [0:4]
        assert labels[0] == "B-NAME"
        assert labels[3] == "E-NAME"
        # gap
        assert all(lab == "O" for lab in labels[4:12])
        # PRODUCT: iphone [12:18]
        assert labels[12] == "B-PRODUCT"
        assert labels[17] == "E-PRODUCT"

    def test_invalid_span_skipped(self):
        text = "hello"
        entities = [{"start": -1, "end": 3, "label": "NAME"}]
        labels = NERDataset._entities_to_char_labels(None, text, entities)
        assert all(lab == "O" for lab in labels)

    def test_empty_span_skipped(self):
        text = "hello"
        entities = [{"start": 2, "end": 2, "label": "NAME"}]
        labels = NERDataset._entities_to_char_labels(None, text, entities)
        assert all(lab == "O" for lab in labels)


# ---------------------------------------------------------------------------
# Token-level BIOES alignment
# ---------------------------------------------------------------------------

class TestAlignLabelsWithTokens:
    """Test _align_labels_with_tokens handles subword tokenization."""

    def test_special_tokens_get_minus_100(self):
        offset_mapping = [[0, 0], [0, 5], [0, 0]]  # CLS, "hello", SEP
        char_labels = ["O"] * 5
        result = NERDataset._align_labels_with_tokens(None, offset_mapping, char_labels)
        assert result[0] == -100
        assert result[2] == -100

    def test_outside_tokens_get_O(self):
        offset_mapping = [[0, 0], [0, 5], [6, 11], [0, 0]]
        char_labels = ["O"] * 11
        result = NERDataset._align_labels_with_tokens(None, offset_mapping, char_labels)
        assert result[1] == LABEL2ID["O"]
        assert result[2] == LABEL2ID["O"]

    def test_single_token_entity_gets_S(self):
        # "john" as one token, S-NAME at char level
        char_labels = list("S-NAME") + ["O"] * 4  # not right, let me fix
        # Actually char_labels is per-character
        # text = "john rest" → "john" is [0:4]
        char_labels = ["S-NAME"] + ["O"] * 8  # "j"=S, but john is 4 chars
        # For single-char entity at position 0:
        char_labels = ["S-NAME", "O", "O", "O", "O"]
        offset_mapping = [[0, 0], [0, 1], [1, 5], [0, 0]]
        # token 1 starts at char 0 → S-NAME → (NAME, True)
        # token 2 starts at char 1 → O → ("O", False)
        # Span: token 1 alone → S-NAME
        result = NERDataset._align_labels_with_tokens(None, offset_mapping, char_labels)
        assert result[1] == LABEL2ID["S-NAME"]
        assert result[2] == LABEL2ID["O"]

    def test_multi_token_entity_gets_B_E(self):
        # "john smith" as two tokens
        # char_labels: B-NAME for 'j', I for 'ohn ', I for 'smi', E for 'th'
        # But we just need first char of each token
        # text = "john smith" [0:10]
        # char: j=B-NAME, o=I, h=I, n=I, ' '=I, s=I, m=I, i=I, t=I, h=E-NAME
        char_labels = ["B-NAME"] + ["I-NAME"] * 8 + ["E-NAME"]
        # Tokens: [CLS][john][smith][SEP]
        offset_mapping = [[0, 0], [0, 4], [5, 10], [0, 0]]
        # token "john": char[0]=B-NAME → (NAME, True)
        # token "smith": char[5]=I-NAME → (NAME, False)
        # Span of 2 tokens → B-NAME, E-NAME
        result = NERDataset._align_labels_with_tokens(None, offset_mapping, char_labels)
        assert result[0] == -100  # CLS
        assert result[1] == LABEL2ID["B-NAME"]
        assert result[2] == LABEL2ID["E-NAME"]
        assert result[3] == -100  # SEP

    def test_three_token_entity_gets_B_I_E(self):
        # "john james smith" as three tokens
        char_labels = ["B-NAME"] + ["I-NAME"] * 14 + ["E-NAME"]
        offset_mapping = [[0, 0], [0, 4], [5, 10], [11, 16], [0, 0]]
        result = NERDataset._align_labels_with_tokens(None, offset_mapping, char_labels)
        assert result[1] == LABEL2ID["B-NAME"]
        assert result[2] == LABEL2ID["I-NAME"]
        assert result[3] == LABEL2ID["E-NAME"]

    def test_adjacent_same_type_entities_separated(self):
        # "john bob" — two NAME entities
        # char: j=B, o=I, h=I, n=E, ' '=O, b=B, o=I, b=E
        char_labels = ["B-NAME", "I-NAME", "I-NAME", "E-NAME",
                        "O",
                        "B-NAME", "I-NAME", "E-NAME"]
        # Tokens: [CLS][john][bob][SEP]
        offset_mapping = [[0, 0], [0, 4], [5, 8], [0, 0]]
        # token "john": char[0]=B-NAME → (NAME, True) → span start
        # token "bob": char[5]=B-NAME → (NAME, True) → new span start!
        # Each is single-token span → S-NAME, S-NAME
        result = NERDataset._align_labels_with_tokens(None, offset_mapping, char_labels)
        assert result[1] == LABEL2ID["S-NAME"]
        assert result[2] == LABEL2ID["S-NAME"]


# ---------------------------------------------------------------------------
# Entity finalization
# ---------------------------------------------------------------------------

class TestFinalizeEntity:
    """Test _finalize_entity computes correct confidence."""

    def test_single_token_confidence(self):
        entity = {"_conf_sum": 0.9, "_tok_count": 1, "text": "x", "label": "NAME"}
        TransformerNER._finalize_entity(entity)
        assert entity["confidence"] == 0.9
        assert "_conf_sum" not in entity
        assert "_tok_count" not in entity

    def test_multi_token_confidence_is_mean(self):
        entity = {"_conf_sum": 2.4, "_tok_count": 3, "text": "x", "label": "NAME"}
        TransformerNER._finalize_entity(entity)
        assert entity["confidence"] == 0.8  # 2.4 / 3

    def test_no_internal_fields_means_noop(self):
        entity = {"text": "x", "label": "NAME", "confidence": 0.5}
        TransformerNER._finalize_entity(entity)
        assert entity["confidence"] == 0.5  # unchanged


# ---------------------------------------------------------------------------
# Predict method (integration — requires trained model)
# ---------------------------------------------------------------------------

MODEL_DIR = Path("results/models/bert_small")


@pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason="Trained BERT-Small model not found — run 'make train' first",
)
class TestBIOESPredict:
    """Integration tests for BIOES predict decoding."""

    @pytest.fixture(autouse=True, scope="class")
    def model(self):
        TestBIOESPredict._model = TransformerNER(str(MODEL_DIR))

    def test_returns_list(self):
        result = self._model.predict("hello world")
        assert isinstance(result, list)

    def test_entity_has_required_fields(self):
        result = self._model.predict("my name is sarah chen")
        if result:
            e = result[0]
            assert "text" in e
            assert "label" in e
            assert "start" in e
            assert "end" in e
            assert "confidence" in e

    def test_entity_offsets_match_text(self):
        text = "my name is sarah chen and my order is ORD-2024-5591"
        result = self._model.predict(text)
        for e in result:
            assert text[e["start"]:e["end"]] == e["text"]

    def test_confidence_between_0_and_1(self):
        result = self._model.predict("order ORD-2024-5591 for john smith")
        for e in result:
            assert 0.0 <= e["confidence"] <= 1.0
