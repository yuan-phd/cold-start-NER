"""Rule-based NER baseline — regex patterns for structured entity types.

This baseline uses handcrafted regex patterns and keyword matching to extract
entities. It serves as a lower bound: no ML, no training data required.
Expected to perform well on EMAIL and CONTRACT_ID (structured patterns),
poorly on PRODUCT and ISSUE_DATE (semantic understanding required).
"""

import re
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Email: standard pattern + common ASR variants
EMAIL_PATTERNS = [
    # Standard email
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    # ASR variant: "at" instead of @
    re.compile(
        r"[a-zA-Z0-9._%+\-]+\s+at\s+[a-zA-Z0-9.\-]+\s+dot\s+(?:com|org|net|edu|io)",
        re.IGNORECASE,
    ),
]

# Contract ID: alphanumeric with dashes, typically uppercase
CONTRACT_ID_PATTERNS = [
    # Standard format: ORD-2024-5591, SUB-33018, ACC-00192
    re.compile(r"\b[A-Z]{2,4}-\d{4,6}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,4}-\d{4}-\d{2,5}\b", re.IGNORECASE),
    # Mixed alphanumeric: A1B2C3 (6+ chars with both letters and digits)
    re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6,}\b"),
    # Oral format with "dash": ORD dash 2025 dash 0091, ref dash 10042
    re.compile(
        r"\b[A-Za-z]{2,4}\s+dash\s+\d{4,6}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[A-Za-z]{2,4}\s+dash\s+\d{4}\s+dash\s+\d{2,5}\b",
        re.IGNORECASE,
    ),
    # Oral format with fillers between parts: REQ dash uh 2025 dash like 1147
    re.compile(
        r"\b[A-Za-z]{2,4}\s+dash\s+(?:(?:uh|um|like|so)\s+)?\d{4}\s+dash\s+(?:(?:uh|um|like|so)\s+)?\d{2,5}\b",
        re.IGNORECASE,
    ),
    # Spelled-out oral: O R D dash 2024 dash 5591 (space-separated letters)
    re.compile(
        r"\b(?:[A-Za-z]\s+){2,4}dash\s+\d{4}\s+dash\s+\d{2,5}\b",
        re.IGNORECASE,
    ),
    # Spelled-out with word numbers: SUB dash three three zero one eight
    re.compile(
        r"\b[A-Za-z]{2,4}\s+dash\s+(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)\s*){3,8}\b",
        re.IGNORECASE,
    ),
]

# Name: keyword-triggered extraction
NAME_TRIGGERS = [
    re.compile(r"(?:my name is|this is|i'm|i am|speaking with|talking to|it's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
    re.compile(r"(?:name|caller)(?:\s+is)?\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
]

# Absolute dates
DATE_PATTERNS = [
    # Month Day (Year): January 15th, March 3rd 2024
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?\b",
        re.IGNORECASE,
    ),
    # MM/DD/YYYY or MM-DD-YYYY
    re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"),
    # YYYY-MM-DD (ISO)
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]

# Relative dates
RELATIVE_DATE_PATTERNS = [
    re.compile(r"\b(?:last|this past)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE),
    re.compile(r"\b(?:yesterday|today|the day before yesterday)\b", re.IGNORECASE),
    re.compile(r"\b(?:a|about|around|roughly)?\s*(?:a couple of|a few|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?)\s+(?:ago|back)\b", re.IGNORECASE),
    re.compile(r"\b(?:last|this past)\s+(?:week|month|year)\b", re.IGNORECASE),
    re.compile(r"\b(?:earlier|sometime)\s+(?:this|last)\s+(?:week|month)\b", re.IGNORECASE),
    re.compile(r"\b(?:end|beginning|start|middle)\s+of\s+(?:last|this)\s+(?:week|month|year)\b", re.IGNORECASE),
    re.compile(r"\babout\s+(?:a\s+)?(?:week|month)\s+ago\b", re.IGNORECASE),
]


def _find_all_matches(
    text: str,
    patterns: list[re.Pattern],
    label: str,
    group: int = 0,
) -> list[dict[str, Any]]:
    """Find all regex matches in text and return as entity dicts.

    Args:
        text: Input text to search.
        patterns: List of compiled regex patterns.
        label: Entity label to assign.
        group: Regex group to extract (0 for full match).

    Returns:
        List of entity dicts.
    """
    entities = []
    seen_spans = set()

    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                matched_text = match.group(group)
                if group > 0:
                    start = match.start(group)
                else:
                    start = match.start()
                end = start + len(matched_text)
            except IndexError:
                continue

            # Deduplicate overlapping matches
            span_key = (start, end)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            entities.append({
                "text": matched_text,
                "label": label,
                "start": start,
                "end": end,
                "confidence": 0.9,  # High confidence for regex matches
            })

    return entities


def _resolve_overlaps(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove overlapping entities, keeping the longest match.

    Args:
        entities: List of entity dicts, potentially overlapping.

    Returns:
        Non-overlapping entity list.
    """
    if not entities:
        return entities

    # Sort by start position, then by length (longest first)
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))

    result = [sorted_ents[0]]
    for ent in sorted_ents[1:]:
        prev = result[-1]
        # No overlap if current starts after previous ends
        if ent["start"] >= prev["end"]:
            result.append(ent)
        # If overlap, keep the longer one
        elif (ent["end"] - ent["start"]) > (prev["end"] - prev["start"]):
            result[-1] = ent

    return result


def predict(text: str) -> list[dict[str, Any]]:
    """Extract entities from text using rule-based patterns.

    Args:
        text: Input text string.

    Returns:
        List of entity dicts with text, label, start, end, confidence.
    """
    entities = []

    # EMAIL
    entities.extend(_find_all_matches(text, EMAIL_PATTERNS, "EMAIL"))

    # CONTRACT_ID
    entities.extend(_find_all_matches(text, CONTRACT_ID_PATTERNS, "CONTRACT_ID"))

    # NAME (uses capture group 1)
    entities.extend(_find_all_matches(text, NAME_TRIGGERS, "NAME", group=1))

    # ISSUE_DATE — absolute
    entities.extend(_find_all_matches(text, DATE_PATTERNS, "ISSUE_DATE"))

    # ISSUE_DATE — relative
    entities.extend(_find_all_matches(text, RELATIVE_DATE_PATTERNS, "ISSUE_DATE"))

    # Resolve overlapping matches
    entities = _resolve_overlaps(entities)

    return entities