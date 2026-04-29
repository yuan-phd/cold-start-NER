"""Tests for V3 oral-format CONTRACT_ID regex patterns in rules engine.

Covers: standard format, oral dash, short oral, filler-embedded,
spelled-out numbers, spelled-out letters, lowercase, and false positives.
"""


from src.models.rules import predict


def _get_contract_ids(text: str) -> list[str]:
    """Extract CONTRACT_ID entity texts from rules prediction."""
    entities = predict(text)
    return [e["text"] for e in entities if e["label"] == "CONTRACT_ID"]


class TestOralContractIDPatterns:
    """Test rules engine matches oral-format CONTRACT_IDs."""

    def test_standard_format(self):
        matches = _get_contract_ids("order is ORD-2024-5591")
        assert any("ORD-2024-5591" in m for m in matches)

    def test_standard_format_single_segment(self):
        matches = _get_contract_ids("reference SUB-33018 please")
        assert any("SUB-33018" in m for m in matches)

    def test_oral_dash_two_segments(self):
        matches = _get_contract_ids("order is ORD dash 2025 dash 0091")
        assert len(matches) >= 1

    def test_oral_dash_single_segment(self):
        matches = _get_contract_ids("order is ref dash 10042")
        assert len(matches) >= 1

    def test_filler_embedded(self):
        matches = _get_contract_ids("order is REQ dash uh 2025 dash like 1147")
        assert len(matches) >= 1

    def test_spelled_out_numbers(self):
        matches = _get_contract_ids("order is SUB dash three three zero one eight")
        assert len(matches) >= 1

    def test_spelled_out_letters(self):
        matches = _get_contract_ids("order is O R D dash 2024 dash 5591")
        assert len(matches) >= 1

    def test_lowercase_standard(self):
        # Case-insensitive matching
        matches = _get_contract_ids("order is ord-2024-6634")
        assert len(matches) >= 1

    def test_lowercase_oral(self):
        matches = _get_contract_ids("order is sub dash 99281")
        assert len(matches) >= 1

    def test_no_false_positive_on_plain_text(self):
        matches = _get_contract_ids("hello my name is john and i need help")
        assert len(matches) == 0

    def test_no_false_positive_on_short_word_dash(self):
        # "i dash" or "a dash" should not match (prefix too short)
        matches = _get_contract_ids("i dash to the store every day")
        assert len(matches) == 0

    def test_entity_offsets_correct(self):
        """Verify start/end offsets match the text."""
        text = "order is ORD-2024-5591 please"
        entities = predict(text)
        cids = [e for e in entities if e["label"] == "CONTRACT_ID"]
        for e in cids:
            assert text[e["start"]:e["end"]] == e["text"]
