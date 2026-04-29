"""Tests for Gemini data generation utilities: offset fixing and overlap detection.

Pure logic tests — no API key required. Tests the fix_offsets and _overlaps_any
functions from run_generate_gemini.py.
"""

import sys
from pathlib import Path


# Add scripts/ to path so we can import from run_generate_gemini
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_generate_gemini import fix_offsets, _overlaps_any


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

class TestOverlapsAny:
    """Test _overlaps_any span overlap checking."""

    def test_no_overlap(self):
        assert _overlaps_any(10, 15, [(0, 5), (20, 25)]) is False

    def test_overlap_at_start(self):
        assert _overlaps_any(3, 8, [(0, 5)]) is True

    def test_overlap_at_end(self):
        assert _overlaps_any(0, 5, [(3, 8)]) is True

    def test_contained(self):
        assert _overlaps_any(2, 4, [(0, 10)]) is True

    def test_containing(self):
        assert _overlaps_any(0, 10, [(2, 4)]) is True

    def test_adjacent_not_overlapping(self):
        # [0, 5) and [5, 10) do not overlap
        assert _overlaps_any(5, 10, [(0, 5)]) is False

    def test_empty_spans_list(self):
        assert _overlaps_any(0, 5, []) is False


# ---------------------------------------------------------------------------
# Offset fixing
# ---------------------------------------------------------------------------

class TestFixOffsets:
    """Test fix_offsets corrects LLM-generated entity offsets."""

    def test_correct_offsets_preserved(self):
        sample = {
            "text": "my name is john smith",
            "entities": [
                {"text": "john smith", "label": "NAME", "start": 11, "end": 21},
            ],
        }
        fixed = fix_offsets(sample)
        assert fixed is not None
        assert fixed["entities"][0]["start"] == 11
        assert fixed["entities"][0]["end"] == 21

    def test_wrong_offsets_corrected(self):
        sample = {
            "text": "my name is john smith",
            "entities": [
                {"text": "john smith", "label": "NAME", "start": 0, "end": 10},
            ],
        }
        fixed = fix_offsets(sample)
        assert fixed is not None
        assert fixed["entities"][0]["start"] == 11
        assert fixed["entities"][0]["end"] == 21
        assert fixed["text"][11:21] == "john smith"

    def test_invalid_label_skipped(self):
        sample = {
            "text": "hello world",
            "entities": [
                {"text": "hello", "label": "INVALID_TYPE", "start": 0, "end": 5},
            ],
        }
        fixed = fix_offsets(sample)
        # No valid entities remain → returns None
        assert fixed is None

    def test_entity_not_found_skipped(self):
        sample = {
            "text": "hello world",
            "entities": [
                {"text": "nonexistent", "label": "NAME", "start": 0, "end": 11},
            ],
        }
        fixed = fix_offsets(sample)
        assert fixed is None

    def test_no_entities_returns_none(self):
        sample = {"text": "hello world", "entities": []}
        fixed = fix_offsets(sample)
        assert fixed is None

    def test_duplicate_text_uses_different_positions(self):
        """Two entities with same text get different offsets."""
        sample = {
            "text": "john called john back",
            "entities": [
                {"text": "john", "label": "NAME", "start": 0, "end": 4},
                {"text": "john", "label": "NAME", "start": 50, "end": 54},
            ],
        }
        fixed = fix_offsets(sample)
        assert fixed is not None
        positions = [(e["start"], e["end"]) for e in fixed["entities"]]
        # First john at 0, second john at 12
        assert (0, 4) in positions
        assert (12, 16) in positions
        # No overlapping positions
        assert positions[0] != positions[1]

    def test_overlapping_entities_rejected(self):
        """Entities that would overlap after fixing are deduplicated."""
        sample = {
            "text": "john smith",
            "entities": [
                {"text": "john", "label": "NAME", "start": 0, "end": 4},
                {"text": "john", "label": "NAME", "start": 0, "end": 4},
            ],
        }
        fixed = fix_offsets(sample)
        assert fixed is not None
        # Should only keep one occurrence
        assert len(fixed["entities"]) == 1

    def test_case_insensitive_fallback(self):
        """Case-insensitive search finds entity in different case."""
        sample = {
            "text": "order for john smith",
            "entities": [
                {"text": "John Smith", "label": "NAME", "start": 0, "end": 10},
            ],
        }
        fixed = fix_offsets(sample)
        assert fixed is not None
        # Should find "john smith" (lowercase) via case-insensitive search
        assert fixed["entities"][0]["text"] == "john smith"
        assert fixed["entities"][0]["start"] == 10
        assert fixed["entities"][0]["end"] == 20
